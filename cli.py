"""Training-completion entry point. No Streamlit dependency is required."""
import argparse
import importlib
import json
import signal
import sys
import threading
from pathlib import Path

from storage import Store, initialize


EXIT_FAIL = 2
EXIT_INSUFFICIENT = 3


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _thresholds(args):
    return {key: getattr(args, key) for key in ('min_pass_rate', 'max_regression_rate', 'max_pass_rate_drop', 'max_category_drop')}


def _add_thresholds(parser):
    parser.add_argument('--min-pass-rate', type=float, default=.8, help='全题通过率下限，0–1')
    parser.add_argument('--max-regression-rate', type=float, default=.05, help='退步题比例上限，0–1')
    parser.add_argument('--max-pass-rate-drop', type=float, default=.02, help='全题通过率允许下降，0–1')
    parser.add_argument('--max-category-drop', type=float, default=.05, help='分类通过率允许下降，0–1')


def parser():
    result = argparse.ArgumentParser(description='Model Easy Eval：训练后回归评测')
    result.add_argument('--db', help='SQLite 路径；默认 EVAL_DB 或 eval.sqlite3')
    commands = result.add_subparsers(dest='command', required=True)
    commands.add_parser('init', help='创建数据库并安装内置题库')
    commands.add_parser('suites', help='列出题库版本')
    commands.add_parser('models', help='列出模型版本')
    add_suite = commands.add_parser('import-suite', help='导入完整的版本化 JSON 题库')
    add_suite.add_argument('path')
    add_model = commands.add_parser('register-model', help='登记不可变训练版本')
    add_model.add_argument('--config', required=True)
    run = commands.add_parser('run', help='执行评测，保存报告，返回质量门槛退出码')
    target = run.add_mutually_exclusive_group(required=True)
    target.add_argument('--model', help='已登记的模型版本 id')
    target.add_argument('--model-config', help='模型版本配置 JSON；不存在时自动登记')
    run.add_argument('--suite', default='general-quick')
    run.add_argument('--suite-version', help='省略时使用最近登记版本；固定流程建议显式指定')
    judge = run.add_mutually_exclusive_group()
    judge.add_argument('--judge', help='评委模型版本 id，仅 judge 题需要')
    judge.add_argument('--judge-config')
    run.add_argument('--rubric-file', help='评委系统提示词 UTF-8 文本')
    run.add_argument('--workers', type=int, default=4)
    run.add_argument('--retries', type=int, default=1)
    run.add_argument('--name')
    run.add_argument('--baseline', action='append', default=[], help='基线实验 id，可重复传入')
    run.add_argument('--baseline-policy', choices=['previous', 'best', 'base'], default='previous')
    run.add_argument('--record-baseline', action='store_true', help='仅建立基线，不做替换决策；完整完成时退出0')
    run.add_argument('--output', help='报告目录；默认 reports/<run-id>')
    run.add_argument('--plugin', action='append', default=[], help='导入本机自定义判定器模块，可重复')
    _add_thresholds(run)
    compare = commands.add_parser('compare', help='对比两个已有实验')
    compare.add_argument('candidate')
    compare.add_argument('baseline')
    compare.add_argument('--output')
    _add_thresholds(compare)
    report = commands.add_parser('report', help='导出已有实验及可选基线的报告')
    report.add_argument('run_id')
    report.add_argument('--baseline', action='append', default=[])
    report.add_argument('--output', required=True)
    _add_thresholds(report)
    return result


def _model(store, model_id=None, config_path=None):
    if config_path:
        raw = _read_json(config_path)
        data = store.save_model(raw)
        # Backward compatibility: inline key is ephemeral, never persisted.
        if raw.get('key'):
            data['key'] = raw['key']
        return data
    return store.get_model(model_id)


def _emit_report(report, output):
    from reporting import write_report
    path = write_report(report, output)
    print(json.dumps({'run_id': report['run']['id'], 'state': report['run']['state'],
                      'verdict': report['verdict'], 'summary': report['summary'],
                      'explanations': report['explanations'], 'report_dir': str(path.resolve())}, ensure_ascii=False))


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        store = Store(args.db)
        initialize(store)
        if args.command == 'init':
            print(json.dumps({'database': store.path, 'suites': len(store.list_suites())}, ensure_ascii=False))
            return 0
        if args.command == 'suites':
            print(json.dumps([dict(id=s['id'], name=s['name'], version=s['version'], questions=len(s['questions']))
                              for s in store.list_suites()], ensure_ascii=False, indent=2))
            return 0
        if args.command == 'models':
            print(json.dumps(store.list_models(), ensure_ascii=False, indent=2))
            return 0
        if args.command == 'import-suite':
            suite = store.save_suite(_read_json(args.path))
            print('%s@%s: %s questions' % (suite['id'], suite['version'], len(suite['questions'])))
            return 0
        if args.command == 'register-model':
            print(store.save_model(_read_json(args.config))['id'])
            return 0
        from reporting import build_report, select_baseline
        if args.command in ('compare', 'report'):
            run = store.get_run(args.candidate if args.command == 'compare' else args.run_id)
            ids = [args.baseline] if args.command == 'compare' else args.baseline
            report = build_report(run, [store.get_run(item) for item in ids], _thresholds(args))
            _emit_report(report, args.output or str(Path('reports') / (run['id'] + '-comparison')))
        else:
            if not 1 <= args.workers <= 32 or not 0 <= args.retries <= 5:
                raise ValueError('workers 必须在1–32之间，retries 必须在0–5之间')
            for value in _thresholds(args).values():
                if not 0 <= value <= 1:
                    raise ValueError('质量门槛必须为0–1之间的比例')
            if args.record_baseline and args.baseline:
                raise ValueError('--record-baseline 不能同时指定 --baseline')
            for module in args.plugin:
                importlib.import_module(module)
            from engine import DEFAULT_RUBRIC
            from runner import run_evaluation
            suite = store.get_suite(args.suite, args.suite_version)
            model = _model(store, args.model, args.model_config)
            judge = _model(store, args.judge, args.judge_config) if args.judge or args.judge_config else None
            explicit_baselines = [store.get_run(item) for item in args.baseline]
            rubric = Path(args.rubric_file).read_text(encoding='utf-8') if args.rubric_file else DEFAULT_RUBRIC
            cancel_event = threading.Event()
            old_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, lambda *_: cancel_event.set())
            try:
                run = run_evaluation(suite, model, judge=judge, rubric=rubric, workers=args.workers,
                                     retries=args.retries, on_update=store.save_run, cancel_event=cancel_event,
                                     metadata={'name': args.name or model['name'], 'source': 'cli', 'thresholds': _thresholds(args)})
            finally:
                signal.signal(signal.SIGINT, old_handler)
            baselines = explicit_baselines
            if not baselines and not args.record_baseline:
                auto = select_baseline(run, store.list_runs(10000), args.baseline_policy)
                baselines = [auto] if auto else []
            report = build_report(run, baselines, _thresholds(args))
            if args.record_baseline:
                report['verdict'] = 'baseline_recorded' if run['state'] == 'finished' and report['summary']['completion_rate'] == 1 else 'insufficient'
                report['explanations'] = ['本次仅记录基线，未执行版本替换判断。']
            _emit_report(report, args.output or str(Path('reports') / run['id']))
            if run['state'] == 'cancelled':
                return 130
        return {'pass': 0, 'baseline_recorded': 0, 'fail': EXIT_FAIL, 'insufficient': EXIT_INSUFFICIENT}[report['verdict']]
    except (ValueError, OSError, KeyError, ImportError) as error:
        # Do not echo configuration, endpoints or model output in machine-readable errors.
        print(json.dumps({'error': type(error).__name__, 'message': str(error) if isinstance(error, ValueError) else '请检查文件路径、依赖和配置。'}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
