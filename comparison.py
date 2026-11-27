"""Paired, protocol-aware comparisons of immutable evaluation run snapshots."""
import math
from datasets import suite_hash, validate_suite

DEFAULT_THRESHOLDS = {
    'min_pass_rate': 0.80,
    'max_regression_rate': 0.05,
    'max_pass_rate_drop': 0.02,
    'max_category_drop': 0.05,
}


def _thresholds(overrides):
    values = dict(DEFAULT_THRESHOLDS)
    if overrides:
        if not isinstance(overrides, dict) or set(overrides) - set(values):
            raise ValueError('未知的质量门槛字段')
        values.update(overrides)
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('%s 必须为 0–1 数字' % name)
    return values


def _generation_protocol(model):
    model = model or {}
    return {
        'temperature': model.get('temperature', 0.0),
        'max_tokens': model.get('max_tokens', 4096),
        'system': model.get('system') or '',
        'extra': model.get('extra') or {},
    }


def _judge_protocol(judge):
    if not judge:
        return None
    return {
        'identity': {key: judge.get(key) for key in ('id', 'model', 'checkpoint', 'adapter', 'revision')},
        'generation': _generation_protocol(judge),
    }


def _index(run, question_ids):
    results = run.get('results', [])
    if not isinstance(results, list):
        return {}, ['结果列表无效']
    indexed, reasons = {}, []
    for result in results:
        if not isinstance(result, dict):
            reasons.append('结果记录无效')
            continue
        question = result.get('question') or {}
        qid = question.get('id') if isinstance(question, dict) else None
        if qid not in question_ids:
            reasons.append('结果包含未知或缺失的题目 ID')
        elif qid in indexed:
            reasons.append('结果包含重复题目 ID: %s' % qid)
        else:
            indexed[qid] = result
    return indexed, list(dict.fromkeys(reasons))


def _successful(result):
    if not result or result.get('status') != 'success' or not isinstance(result.get('passed'), bool):
        return False
    score = result.get('score')
    return not isinstance(score, bool) and isinstance(score, (int, float)) and math.isfinite(score) and 0 <= score <= 100


def _summary(indexed, questions):
    total = len(questions)
    valid = [r for r in indexed.values() if _successful(r)]
    passed = sum(r['passed'] for r in valid)
    categories = {}
    for question in questions:
        category = question['category']
        bucket = categories.setdefault(category, {'total': 0, 'completed': 0, 'passed': 0, 'failed': 0, 'errors': 0})
        bucket['total'] += 1
        result = indexed.get(question['id'])
        if _successful(result):
            bucket['completed'] += 1
            bucket['passed'] += int(result['passed'])
            bucket['failed'] += int(not result['passed'])
        else:
            bucket['errors'] += 1
    for bucket in categories.values():
        bucket['pass_rate'] = bucket['passed'] / bucket['total']
        bucket['completion_rate'] = bucket['completed'] / bucket['total']
    return {
        'total': total, 'completed': len(valid), 'passed': passed,
        'failed': len(valid) - passed, 'errors': total - len(valid),
        'completion_rate': len(valid) / total if total else 0,
        'pass_rate': passed / total if total else 0,
        'graded_pass_rate': passed / len(valid) if valid else None,
        'mean_score': sum(r['score'] for r in valid) / len(valid) if valid else None,
        'categories': categories,
    }


def _view(result):
    if result is None:
        return {'status': 'missing', 'passed': None, 'score': None, 'answer': ''}
    answer = result.get('answer') or {}
    return {
        'status': result.get('status', 'unknown'), 'passed': result.get('passed'),
        'score': result.get('score'), 'answer': answer.get('text', '') if isinstance(answer, dict) else '',
        'reason': result.get('reason', ''), 'error': result.get('error', ''),
    }


def compare_runs(candidate, baseline, thresholds=None):
    """Compare identical test protocols; never turn incomplete evidence into a pass.

    Rates use all planned questions as denominator. `improved` / `regressed`
    represent pass/fail transitions; score changes remain visible in each row.
    Threshold defaults are editable engineering examples, not statistical claims.
    """
    limits = _thresholds(thresholds)
    comparison = {
        'compatible': False, 'reasons': [], 'verdict': 'insufficient',
        'explanations': [], 'thresholds': limits,
        'counts': {key: 0 for key in ('improved', 'regressed', 'unchanged', 'unresolved')},
        'improved': [], 'regressed': [], 'unchanged': [], 'unresolved': [], 'summary': {},
    }
    if not isinstance(candidate, dict) or not isinstance(baseline, dict):
        comparison['reasons'] = ['请选择两个有效实验']
        comparison['explanations'] = list(comparison['reasons'])
        return comparison
    try:
        suite = validate_suite(candidate.get('suite'))
        baseline_suite = validate_suite(baseline.get('suite'))
        if suite_hash(suite) != suite_hash(baseline_suite):
            comparison['reasons'].append('题库 ID、版本或题目/评分规则不同，请在同一题库版本上补测')
    except (ValueError, TypeError, KeyError):
        comparison['reasons'] = ['实验缺少有效题库快照，旧版实验需要重新评测']
        comparison['explanations'] = list(comparison['reasons'])
        return comparison
    if candidate.get('protocol_version') != baseline.get('protocol_version'):
        comparison['reasons'].append('评测引擎协议版本不同')
    if _generation_protocol(candidate.get('model')) != _generation_protocol(baseline.get('model')):
        comparison['reasons'].append('被测模型的生成参数或系统提示词不同')
    uses_judge = any(q['evaluator'] == 'judge' for q in suite['questions'])
    if uses_judge:
        if candidate.get('rubric') != baseline.get('rubric'):
            comparison['reasons'].append('评委评分提示词不同')
        if not candidate.get('judge') or not baseline.get('judge') or _judge_protocol(candidate.get('judge')) != _judge_protocol(baseline.get('judge')):
            comparison['reasons'].append('评委模型版本或生成参数不同')
    question_ids = {q['id'] for q in suite['questions']}
    candidate_index, candidate_errors = _index(candidate, question_ids)
    baseline_index, baseline_errors = _index(baseline, {q['id'] for q in baseline_suite['questions']})
    comparison['reasons'].extend('新实验: ' + reason for reason in candidate_errors)
    comparison['reasons'].extend('基线实验: ' + reason for reason in baseline_errors)
    comparison['compatible'] = not comparison['reasons']
    if not comparison['compatible']:
        comparison['explanations'] = list(comparison['reasons'])
        return comparison
    for question in suite['questions']:
        before, after = baseline_index.get(question['id']), candidate_index.get(question['id'])
        complete = _successful(before) and _successful(after)
        row = {
            'id': question['id'], 'category': question['category'], 'question': question['question'],
            'critical': question.get('critical', False), 'before': _view(before), 'after': _view(after),
            'score_delta': after['score'] - before['score'] if complete else None,
        }
        if not complete:
            key = 'unresolved'
        elif not before['passed'] and after['passed']:
            key = 'improved'
        elif before['passed'] and not after['passed']:
            key = 'regressed'
        else:
            key = 'unchanged'
        comparison[key].append(row)
        comparison['counts'][key] += 1
    current = _summary(candidate_index, suite['questions'])
    previous = _summary(baseline_index, suite['questions'])
    category_deltas = {
        category: {'candidate': bucket['pass_rate'], 'baseline': previous['categories'][category]['pass_rate'],
                   'delta': bucket['pass_rate'] - previous['categories'][category]['pass_rate']}
        for category, bucket in current['categories'].items()
    }
    regression_rate = len(comparison['regressed']) / len(question_ids)
    comparison['summary'] = {
        'candidate': current, 'baseline': previous,
        'pass_rate_delta': current['pass_rate'] - previous['pass_rate'],
        'mean_score_delta': current['mean_score'] - previous['mean_score'] if current['mean_score'] is not None and previous['mean_score'] is not None else None,
        'regression_rate': regression_rate, 'category_deltas': category_deltas,
    }
    if comparison['unresolved'] or candidate.get('state') != 'finished' or baseline.get('state') != 'finished':
        comparison['explanations'] = ['两次实验必须完成全部题目评分；存在缺失、调用失败、评分失败或未结束的任务，请补测']
        return comparison
    failures = []
    epsilon = 1e-12
    if current['pass_rate'] + epsilon < limits['min_pass_rate']:
        failures.append('通过率低于最低要求')
    if regression_rate > limits['max_regression_rate'] + epsilon:
        failures.append('新增退步题比例超过允许范围')
    if previous['pass_rate'] - current['pass_rate'] > limits['max_pass_rate_drop'] + epsilon:
        failures.append('整体通过率下降超过允许范围')
    for category, delta in category_deltas.items():
        if -delta['delta'] > limits['max_category_drop'] + epsilon:
            failures.append('%s 类别通过率下降超过允许范围' % category)
    if any(q.get('critical') and not candidate_index[q['id']]['passed'] for q in suite['questions']):
        failures.append('存在未通过的关键题')
    comparison['verdict'] = 'fail' if failures else 'pass'
    comparison['explanations'] = failures or ['完整评测已满足所配置门槛；该结论不代表差异具有统计显著性']
    return comparison
