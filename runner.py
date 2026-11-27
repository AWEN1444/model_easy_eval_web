"""Concurrent, resumable-record-friendly evaluation runs and honest summaries."""
import copy
import math
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from engine import DEFAULT_RUBRIC, evaluate, evaluator_names

_SECRET_FIELDS = {'key', 'api_key', 'apikey', 'authorization', 'password', 'token',
                  'access_token', 'secret', 'client_secret', 'headers'}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _is_secret_field(name):
    normalized = str(name).lower().replace('-', '_')
    return normalized in _SECRET_FIELDS or normalized.endswith(('_api_key', '_password', '_secret', '_token'))


def sanitize_config(value):
    """Strip credentials from configuration; never apply this to test expectations."""
    if isinstance(value, dict):
        return {key: sanitize_config(item) for key, item in value.items()
                if not _is_secret_field(key) and not str(key).startswith('_')}
    if isinstance(value, list):
        return [sanitize_config(item) for item in value]
    if isinstance(value, str) and value.startswith(('http://', 'https://')):
        try:
            url = urlsplit(value)
            host = url.netloc.rsplit('@', 1)[-1]
            query = [(k, v) for k, v in parse_qsl(url.query, keep_blank_values=True)
                     if not _is_secret_field(k)]
            return urlunsplit((url.scheme, host, url.path, urlencode(query), ''))
        except ValueError:
            return '[invalid URL]'
    return value


def _mean(values):
    valid = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)]
    return round(sum(valid) / len(valid), 4) if valid else None


def summarize_results(results, total=None, _include_categories=True):
    """Rates are fractions 0–1. Main pass rate always uses all scheduled cases."""
    total = len(results) if total is None else total
    good = [result for result in results if result.get('status') == 'success']
    passed = sum(result.get('passed') is True for result in good)
    cancelled = sum(result.get('status') == 'cancelled' for result in results)
    summary = {'total': total, 'processed': len(results), 'completed': len(good),
               'passed': passed, 'failed': len(good) - passed,
               'errors': len(results) - len(good) - cancelled, 'cancelled': cancelled,
               'completion_rate': len(good) / total if total else 0,
               'pass_rate': passed / total if total else 0,
               'graded_pass_rate': passed / len(good) if good else None,
               'mean_score': _mean([result.get('score') for result in good]),
               'mean_seconds': _mean([result.get('answer', {}).get('seconds') for result in results]),
               'categories': {}}
    if not _include_categories:
        return summary
    groups = {}
    for result in results:
        category = result.get('question', {}).get('category') or '未分类'
        groups.setdefault(category, []).append(result)
    for category, items in groups.items():
        item_summary = summarize_results(items, len(items), False)
        item_summary.pop('categories', None)
        summary['categories'][category] = item_summary
    return summary


def summarize_run(run):
    summary = summarize_results(run['results'], len(run['suite']['questions']))
    # Include queued and cancelled questions in each category denominator too.
    by_id = {result['question']['id']: result for result in run['results']}
    categories = {}
    for question in run['suite']['questions']:
        categories.setdefault(question.get('category') or '未分类', []).append(question)
    summary['categories'] = {}
    for category, questions in categories.items():
        items = [by_id[q['id']] for q in questions if q['id'] in by_id]
        data = summarize_results(items, len(questions), False)
        data.pop('categories', None)
        summary['categories'][category] = data
    return summary


def run_evaluation(suite, model, judge=None, rubric=DEFAULT_RUBRIC, workers=4,
                   retries=1, on_update=None, cancel_event=None, run_id=None, metadata=None):
    """Run a frozen suite against one training version; persist with on_update(run).

    Cancelling stops queued requests and retries. An in-flight HTTP request remains
    bounded by its configured timeout. Callbacks run serially on the caller thread.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 32:
        raise ValueError('并发数必须为 1–32 的整数')
    if isinstance(retries, bool) or not isinstance(retries, int) or not 0 <= retries <= 5:
        raise ValueError('重试次数必须为 0–5 的整数')
    if not suite.get('questions'):
        raise ValueError('题库不能为空')
    ids = [q.get('id') for q in suite['questions']]
    if any(not question_id for question_id in ids) or len(set(ids)) != len(ids):
        raise ValueError('每道题必须有唯一且非空的 id')
    evaluators = {question.get('evaluator', 'judge') for question in suite['questions']}
    unknown = evaluators - set(evaluator_names())
    if unknown:
        raise ValueError('题库包含未注册评测器：{}；请先加载对应插件'.format(', '.join(sorted(unknown))))
    if 'judge' in evaluators and judge is None:
        raise ValueError('judge 类型题目需要先配置评委模型')
    frozen_suite = copy.deepcopy(suite)
    runtime_model, runtime_judge = copy.deepcopy(model), copy.deepcopy(judge)
    cancel_event = cancel_event or threading.Event()
    metadata = copy.deepcopy(metadata or {})
    run = {'id': run_id or uuid.uuid4().hex,
           'name': metadata.get('name') or '{} · {}'.format(model.get('name', model['model']), suite['name']),
           'created': _now(), 'finished': None, 'state': 'running', 'rubric': rubric,
           'protocol_version': '2',
           'suite': frozen_suite, 'model': sanitize_config(model),
           'judge': sanitize_config(judge), 'results': [],
           'metadata': sanitize_config(metadata),
           'execution': {'workers': workers, 'retries': retries}}

    def publish():
        run['summary'] = summarize_run(run)
        if on_update:
            on_update(copy.deepcopy(run))

    publish()
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = {}
    try:
        for question in frozen_suite['questions']:
            if cancel_event.is_set():
                break
            future = pool.submit(evaluate, question, runtime_model, runtime_judge,
                                 rubric, cancel_event, retries)
            futures[future] = question
        for future in as_completed(futures):
            question = futures[future]
            if cancel_event.is_set():
                for pending in futures:
                    pending.cancel()
            if future.cancelled():
                result = {'question': question, 'model': model['model'],
                          'model_version_id': model.get('id', model['model']),
                          'status': 'cancelled', 'error': '任务已取消', 'error_kind': 'cancelled'}
            else:
                try:
                    result = future.result()
                except Exception as error:
                    result = {'question': question, 'model': model['model'],
                              'model_version_id': model.get('id', model['model']),
                              'status': 'evaluation_error', 'error': type(error).__name__,
                              'error_kind': type(error).__name__}
            run['results'].append(result)
            publish()
        if cancel_event.is_set():
            known = {result['question']['id'] for result in run['results']}
            for question in frozen_suite['questions']:
                if question['id'] not in known:
                    run['results'].append({'question': question, 'model': model['model'],
                                           'model_version_id': model.get('id', model['model']),
                                           'status': 'cancelled', 'error': '任务已取消', 'error_kind': 'cancelled'})
            run['state'] = 'cancelled'
        else:
            run['state'] = 'finished'
        run['results'].sort(key=lambda result: ids.index(result['question']['id']))
        run['finished'] = _now()
        publish()
        return run
    except BaseException as error:
        cancel_event.set()
        for pending in futures:
            pending.cancel()
        run['state'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        run['finished'] = _now()
        try:
            publish()
        except Exception:
            pass
        raise
    finally:
        pool.shutdown(wait=True)
