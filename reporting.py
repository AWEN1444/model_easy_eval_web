"""Baseline selection and portable reports shared by CLI and integrations."""
import csv
import io
import json
from pathlib import Path


def select_baseline(candidate, runs, policy='previous'):
    from comparison import compare_runs
    from runner import summarize_results
    eligible = []
    for run in runs:
        if run.get('id') == candidate.get('id') or not run.get('suite') or run.get('state') != 'finished':
            continue
        if run.get('model', {}).get('id') == candidate.get('model', {}).get('id'):
            continue
        if candidate.get('created') and run.get('created', '') > candidate['created']:
            continue
        if policy == 'base' and run.get('model', {}).get('role') != 'base':
            continue
        result = compare_runs(candidate, run)
        baseline_summary = result.get('summary', {}).get('baseline', {})
        if result['compatible'] and baseline_summary.get('completion_rate') == 1:
            eligible.append(run)
    if not eligible:
        return None
    if policy == 'best':
        return max(eligible, key=lambda r: (summarize_results(r['results'], len(r['suite']['questions']))['pass_rate'], r.get('created', '')))
    if policy not in ('previous', 'base'):
        raise ValueError('未知基线选择策略：' + policy)
    return max(eligible, key=lambda r: r.get('created', ''))


def build_report(run, baselines=None, thresholds=None):
    from comparison import compare_runs
    from runner import summarize_results, summarize_run
    comparisons = []
    for baseline in baselines or []:
        result = compare_runs(run, baseline, thresholds)
        result['baseline_run_id'] = baseline['id']
        result['baseline_name'] = baseline.get('name', baseline['id'])
        comparisons.append(result)
    if not comparisons:
        verdict = 'insufficient'
        explanations = ['尚无可比基线；此次结果可作为基线记录，不能据此判断训练进步。']
    elif any(item['verdict'] == 'fail' for item in comparisons):
        verdict = 'fail'
        explanations = ['至少一项基线对比未达到设定门槛。']
    elif any(item['verdict'] == 'insufficient' for item in comparisons):
        verdict = 'insufficient'
        explanations = ['存在不完整或协议不一致的对比，需要补测或统一评测条件。']
    else:
        verdict = 'pass'
        explanations = ['已完成的基线对比达到设定门槛；结论仅适用于本题库和评测协议。']
    total = len(run.get('suite', {}).get('questions', [])) or len(run.get('results', []))
    summary = summarize_run(run) if run.get('suite') else summarize_results(run.get('results', []), total)
    return {'schema_version': 2, 'run': run, 'summary': summary,
            'comparisons': comparisons, 'verdict': verdict, 'explanations': explanations}


def _safe_cell(value):
    if value is None:
        return ''
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    text = str(value)
    if text.lstrip().startswith(('=', '+', '-', '@')) and not isinstance(value, (int, float)):
        return "'" + text
    return text


def results_csv(run):
    output = io.StringIO()
    columns = ['question_id', 'category', 'evaluator', 'question', 'status', 'passed', 'score', 'reason', 'seconds', 'answer', 'error']
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    for result in run.get('results', []):
        question = result.get('question', {})
        answer = result.get('answer', {})
        row = dict(question_id=question.get('id'), category=question.get('category'), evaluator=question.get('evaluator', 'judge'),
                   question=question.get('question'), status=result.get('status'), passed=result.get('passed'), score=result.get('score'),
                   reason=result.get('reason'), seconds=answer.get('seconds'), answer=answer.get('text'), error=result.get('error'))
        writer.writerow({key: _safe_cell(value) for key, value in row.items()})
    return output.getvalue()


def report_markdown(report):
    run, summary = report['run'], report['summary']
    suite, model = run.get('suite', {}), run.get('model', {})
    lines = ['# Model Easy Eval 评测报告', '',
             '- 实验：%s (`%s`)' % (run.get('name', ''), run['id']),
             '- 模型版本：%s / checkpoint：%s' % (model.get('name', ''), model.get('checkpoint', '未登记')),
             '- 题库：%s @ %s' % (suite.get('id', 'legacy'), suite.get('version', '')),
             '- 状态：%s；结论：**%s**' % (run.get('state'), report['verdict']),
             '- 完成评分：%s/%s；通过：%s；答错：%s；错误：%s' %
             (summary['completed'], summary['total'], summary['passed'], summary['failed'], summary['errors']),
             '- 全题通过率：%.1f%%（分母包含未完成与调用失败的题）' % (100 * summary['pass_rate']), '',
             *report['explanations'], '', '## 分类结果', '',
             '| 类别 | 计划题数 | 已评分 | 通过 | 全题通过率 |', '| --- | ---: | ---: | ---: | ---: |']
    for category, values in summary.get('categories', {}).items():
        lines.append('| %s | %s | %s | %s | %.1f%% |' %
                     (category.replace('|', '\\|'), values.get('total', 0), values.get('completed', 0), values.get('passed', 0), values.get('pass_rate', 0) * 100))
    for comparison in report['comparisons']:
        lines.extend(['', '## 对比：%s' % comparison['baseline_run_id'], '',
                      '结论：**%s**' % comparison['verdict'], '', *comparison.get('explanations', []), *comparison.get('reasons', [])])
        counts = comparison.get('counts', {})
        lines.append('逐题变化：提升 %s，退步 %s，不变 %s，无法判断 %s。' % tuple(counts.get(k, 0) for k in ('improved', 'regressed', 'unchanged', 'unresolved')))
        for row in comparison.get('regressed', []):
            lines.append('- 退步题 `%s`：%s' % (row['id'], row['question'].replace('\n', ' ')))
    errors = [r for r in run.get('results', []) if r.get('status') != 'success']
    wrong = [r for r in run.get('results', []) if r.get('status') == 'success' and not r.get('passed')]
    if wrong or errors:
        lines.extend(['', '## 需排查的样本', ''])
        for result in wrong + errors:
            lines.append('- `%s` [%s] %s' % (result.get('question', {}).get('id', ''), result.get('status'), result.get('reason', result.get('error', ''))))
    lines.extend(['', '本报告不证明 DSL 编译或功能测试通过；python_syntax 仅检查 Python 语法。通用内置题库不是标准学术基准。', ''])
    return '\n'.join(lines)


def write_report(report, output_dir):
    from storage import sanitize_run
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    clean = dict(report)
    clean['run'] = sanitize_run(report['run'])
    (path / 'report.json').write_text(json.dumps(clean, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    (path / 'results.csv').write_text('\ufeff' + results_csv(clean['run']), encoding='utf-8')
    (path / 'report.md').write_text(report_markdown(clean), encoding='utf-8')
    return path
