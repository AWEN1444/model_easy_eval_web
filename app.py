"""Local training-evaluation dashboard."""
import copy
import json
import uuid

import pandas as pd
import streamlit as st

from comparison import compare_runs
from datasets import import_questions, validate_suite
from engine import DEFAULT_RUBRIC, evaluator_names
from jobs import JobManager
from runner import summarize_run
from reporting import results_csv
from storage import Store, initialize

st.set_page_config(page_title='Model Easy Eval', page_icon='🧪', layout='wide')


@st.cache_resource
def job_manager():
    return JobManager()


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def suite_label(suite):
    return f"{suite['name']} · v{suite['version']} · {len(suite['questions'])} 题"


def model_label(model):
    return f"{model.get('name', model['model'])} · {model.get('checkpoint') or model['model']}"


def run_label(run):
    model = run.get('model', {})
    return f"{run.get('name') or model.get('name') or run['id'][:8]} · {run.get('created', '')[:19]} · {run.get('state', '')}"


def notice(message, kind='success'):
    st.session_state['notice'] = (kind, message)


def render_notice():
    item = st.session_state.pop('notice', None)
    if item:
        getattr(st, item[0])(item[1])


def runtime_model(model, key):
    config = copy.deepcopy(model)
    if key:
        config['key'] = key
    return config


def result_rows(run):
    rows = []
    for result in run.get('results', []):
        question = result.get('question', {})
        answer = result.get('answer') or {}
        rows.append({
            '题目 ID': question.get('id', ''), '类别': question.get('category', ''),
            '问题': question.get('question', ''), '判定器': question.get('evaluator', 'judge'),
            '状态': result.get('status', ''), '通过': result.get('passed'),
            '评分': result.get('score'), '耗时（秒）': answer.get('seconds'),
            '回答': answer.get('text', ''), '理由': result.get('reason', ''),
            '错误类型': result.get('error_kind', ''), '错误': result.get('error', ''),
        })
    return rows


def show_report(run, prefix):
    if 'suite' not in run or 'model' not in run:
        st.info('这是旧版实验记录，可查看原始数据；缺少题库和训练版本快照，无法进行可信的跨版本对比。')
        st.json(run)
        st.download_button('导出旧版记录', json_text(run), f"{run['id']}.json", key=prefix+'legacy')
        return
    total = len(run['suite']['questions'])
    summary = summarize_run(run)
    a, b, c, d = st.columns(4)
    a.metric('题目通过率', f"{summary.get('pass_rate', 0):.1%}")
    b.metric('有效判定完成率', f"{summary.get('completion_rate', 0):.1%}")
    c.metric('通过 / 总题数', f"{summary.get('passed', 0)} / {summary.get('total', total)}")
    d.metric('调用或判定错误', summary.get('errors', 0))
    st.caption('通过率以全部计划题目为分母；接口错误不会提高通过率。错误与答错分别统计。')
    st.write(f"**模型版本：** {model_label(run['model'])}　 **题库：** {suite_label(run['suite'])}")
    if run.get('state') not in ('finished', 'completed'):
        st.warning(f"本次实验状态：{run.get('state', '未知')}。未完整完成的结果不能作为替换版本的充分依据。")
    rows = result_rows(run)
    if rows:
        frame = pd.DataFrame(rows)
        categories = []
        for category, bucket in summary.get('categories', {}).items():
            categories.append({'类别': category, '计划题数': bucket['total'],
                               '有效判定': bucket['completed'], '通过': bucket['passed'],
                               '答错': bucket['failed'], '调用或判定错误': bucket['errors'],
                               '取消': bucket.get('cancelled', 0)})
        if categories:
            st.dataframe(pd.DataFrame(categories), hide_index=True, width='stretch')
        failures_only = st.checkbox('只看未通过和异常题', key=prefix+'failures')
        shown = frame[frame['通过'] != True] if failures_only else frame  # noqa: E712
        st.dataframe(shown, hide_index=True, width='stretch')
        st.download_button('导出结果 CSV', results_csv(run).encode('utf-8-sig'),
                           f"{run['id']}.csv", key=prefix+'csv')
        with st.expander('查看逐题原始回答、评分及用量'):
            st.json(run.get('results', []))
    st.download_button('导出完整实验 JSON', json_text(run), f"{run['id']}.json", key=prefix+'json')


def overview(store):
    suites, models, runs = store.list_suites(), store.list_models(), store.list_runs()
    a, b, c = st.columns(3)
    a.metric('题库版本', len(suites))
    b.metric('训练版本', len(models))
    c.metric('实验记录', len(runs))
    st.subheader('每轮训练后的固定流程')
    st.write('登记 checkpoint → 选择固定题库 → 执行评测 → 对比基础模型、上一版或历史最佳 → 查看新增退步题。')
    st.info('内置 80 道核心题和其中 20 道快速检查题，覆盖基础推理、结构化输出、文本处理及代码理解。它们是通用工程回归集，不代表标准学术基准或 DSL 专项能力。')
    st.write('**开始使用：** 在“模型版本”登记训练产物，然后到“新建评测”运行。需要专项评测时，在“题库管理”导入真实任务并保存新版本。')
    if runs:
        st.subheader('最近实验')
        items = []
        for run in runs[:10]:
            summary = run.get('summary', {})
            items.append({'实验': run_label(run), '模型版本': run.get('model', {}).get('name', '旧版记录'),
                          '通过率': f"{summary['pass_rate']:.1%}" if 'pass_rate' in summary else '—',
                          '状态': run.get('state', '')})
        st.dataframe(pd.DataFrame(items), hide_index=True, width='stretch')
    st.caption('本地单用户使用。网页任务在当前服务进程内运行；关闭浏览器不终止任务，重启服务会中断任务，已保存的逐题结果仍可查看。')


def next_version(version):
    parts = str(version).split('.')
    if parts[-1].isdigit():
        parts[-1] = str(int(parts[-1])+1)
        return '.'.join(parts)
    return str(version)+'-next'


def question_frame(questions):
    rows = []
    for item in questions:
        rows.append({'id': item.get('id', ''), 'category': item.get('category', ''),
                     'question': item.get('question', ''), 'evaluator': item.get('evaluator', 'judge'),
                     'expected_json': json.dumps(item['expected'], ensure_ascii=False) if 'expected' in item else '',
                     'reference': item.get('reference', ''), 'criteria': item.get('criteria', ''),
                     'options_json': json.dumps(item.get('options', {}), ensure_ascii=False),
                     'critical': item.get('critical', False)})
    columns = ['id', 'category', 'question', 'evaluator', 'expected_json', 'reference', 'criteria', 'options_json', 'critical']
    return pd.DataFrame(rows, columns=columns)


def frame_questions(frame, originals=None):
    # Preserve plugin and dataset extensions that are not columns in the editor.
    by_id = {item.get('id'): item for item in originals or [] if item.get('id')}
    questions = []
    for row in frame.fillna('').to_dict('records'):
        if not str(row.get('question', '')).strip():
            continue
        edited = {key: str(row.get(key, '')).strip() for key in ('id', 'category', 'question', 'evaluator', 'reference', 'criteria')}
        item = copy.deepcopy(by_id.get(edited['id'], {}))
        item.update(edited)
        item['id'] = item['id'] or 'q-'+uuid.uuid4().hex[:12]
        item['evaluator'] = item['evaluator'] or 'judge'
        expected = str(row.get('expected_json', '')).strip()
        if expected:
            item['expected'] = json.loads(expected)
        else:
            item.pop('expected', None)
        item['options'] = json.loads(str(row.get('options_json', '')).strip() or '{}')
        item['critical'] = row.get('critical') is True
        questions.append(item)
    return questions


def suite_page(store):
    st.subheader('题库管理')
    suites = store.list_suites()
    source = st.selectbox('编辑来源', [None]+suites, index=1 if suites else 0,
                          format_func=lambda s: '新建空白题库' if s is None else suite_label(s), key='suite_source')
    token = f"{source['id']}:{source['version']}" if source else 'new'
    if st.session_state.get('suite_draft_source') != token:
        st.session_state['suite_draft_source'] = token
        st.session_state['suite_draft_questions'] = copy.deepcopy(source['questions']) if source else []
        st.session_state['suite_draft_metadata'] = copy.deepcopy(source or {})
        st.session_state['suite_revision'] = st.session_state.get('suite_revision', 0)+1
        st.session_state['suite_id'] = source['id'] if source else 'custom-'+uuid.uuid4().hex[:8]
        st.session_state['suite_name'] = source['name'] if source else '我的专项题库'
        st.session_state['suite_version'] = next_version(source['version']) if source else '1.0'
        st.session_state['suite_description'] = source.get('description', '') if source else ''
    st.caption('已保存的题库版本不可覆盖。修改后保存为新版本，历史实验继续使用原始快照。')
    left, right = st.columns(2)
    left.text_input('题库 ID', key='suite_id')
    right.text_input('新版本号', key='suite_version')
    st.text_input('题库名称', key='suite_name')
    st.text_area('用途与范围说明', key='suite_description', height=80)
    upload = st.file_uploader('导入 CSV / JSON / JSONL', type=['csv', 'json', 'jsonl'], key='suite_upload')
    if st.button('载入文件（替换编辑区）', disabled=upload is None):
        try:
            raw = upload.getvalue()
            st.session_state['suite_draft_questions'] = import_questions(raw, upload.name)
            if upload.name.lower().endswith('.json'):
                imported = json.loads(raw.decode('utf-8-sig'))
                if isinstance(imported, dict):
                    # Keep suite extensions without silently replacing the identity fields above.
                    extension = {k: v for k, v in imported.items()
                                 if k not in ('id', 'name', 'version', 'description', 'questions')}
                    st.session_state['suite_draft_metadata'].update(copy.deepcopy(extension))
            st.session_state['suite_revision'] += 1
        except (ValueError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
            st.error(str(exc))
    st.caption('expected_json 使用 JSON：字符串写为 "答案"，JSON 对象写为 {"result": 5}。options_json 可设置 pass_score 等参数。python_syntax 仅检查语法，不证明功能正确。')
    edited = st.data_editor(question_frame(st.session_state['suite_draft_questions']), num_rows='dynamic', hide_index=True,
                            width='stretch', key=f"suite_editor_{st.session_state['suite_revision']}",
                            column_config={'evaluator': st.column_config.SelectboxColumn('判定器', options=evaluator_names(), required=True),
                                           'critical': st.column_config.CheckboxColumn('关键题')})
    if st.button('保存题库新版本', type='primary'):
        try:
            draft = copy.deepcopy(st.session_state.get('suite_draft_metadata', source or {}))
            draft.update({'id': st.session_state['suite_id'], 'name': st.session_state['suite_name'],
                          'version': st.session_state['suite_version'], 'description': st.session_state['suite_description'],
                          'questions': frame_questions(edited, st.session_state['suite_draft_questions'])})
            suite = validate_suite(draft)
            store.save_suite(suite)
            notice(f"已保存 {suite_label(suite)}")
            st.rerun()
        except (ValueError, TypeError, KeyError) as exc:
            st.error(str(exc))
    if source:
        st.download_button('导出所选题库 JSON', json_text(source), f"{source['id']}-v{source['version']}.json", key='suite_export')


def models_page(store):
    st.subheader('模型与训练版本')
    models = store.list_models()
    if models:
        st.dataframe(pd.DataFrame([{'版本名称': m.get('name', ''), '角色': m.get('role', 'candidate'),
                                   'Checkpoint': m.get('checkpoint', ''), '服务 Model ID': m['model'],
                                   '训练数据版本': m.get('training_data_version', ''), '版本 ID': m['id']} for m in models]),
                     hide_index=True, width='stretch')
    st.caption('同一个 API Model ID 可以登记多个训练版本。每个版本保存独立 checkpoint 和配置；变更权重后应登记新版本。')
    with st.form('register_model', clear_on_submit=False):
        name = st.text_input('版本名称', placeholder='例如 DSL-SFT-v3 / checkpoint-1200', key='model_name')
        checkpoint = st.text_input('Checkpoint / Adapter 标识', placeholder='路径、commit 或权重哈希', key='model_checkpoint')
        a, b = st.columns(2)
        model_id = a.text_input('API Model ID', key='model_api_id')
        base_url = b.text_input('API Base URL', value='http://localhost:8000/v1', key='model_base_url')
        role = st.selectbox('版本角色', ['candidate', 'base', 'champion'],
                            format_func=lambda r: {'candidate':'候选版本', 'base':'基础模型', 'champion':'历史最佳'}[r], key='model_role')
        training_data = st.text_input('训练数据版本', key='model_training_data')
        training_config_raw = st.text_area('训练配置 JSON', '{}', key='model_training_config')
        api_env = st.text_input('API Key 环境变量名称（可留空）', placeholder='MODEL_API_KEY', key='model_api_env')
        a, b, c = st.columns(3)
        temperature = a.number_input('Temperature', 0.0, 2.0, 0.0, 0.1, key='model_temperature')
        max_tokens = b.number_input('Max tokens', 1, 131072, 2048, key='model_max_tokens')
        timeout = c.number_input('Timeout（秒）', 1, 1800, 120, key='model_timeout')
        system = st.text_area('System prompt', key='model_system')
        extra_raw = st.text_area('额外请求参数 JSON', '{}', key='model_extra')
        submitted = st.form_submit_button('登记模型版本', type='primary')
    if submitted:
        try:
            extra, training_config = json.loads(extra_raw), json.loads(training_config_raw)
            if not isinstance(extra, dict) or not isinstance(training_config, dict):
                raise ValueError('请求参数和训练配置必须为 JSON 对象。')
            model = {'id': uuid.uuid4().hex, 'name': name.strip(), 'checkpoint': checkpoint.strip(),
                     'model': model_id.strip(), 'base_url': base_url.strip(), 'role': role,
                     'training_data_version': training_data.strip(), 'training_config': training_config,
                     'api_key_env': api_env.strip(), 'temperature': temperature,
                     'max_tokens': int(max_tokens), 'timeout': timeout, 'system': system, 'extra': extra}
            if not model['name'] or not model['checkpoint']:
                raise ValueError('请填写版本名称和 checkpoint / adapter 标识，以便追溯实际权重。')
            store.save_model(model)
            notice(f"已登记模型版本：{name}")
            st.rerun()
        except (ValueError, TypeError) as exc:
            st.error(str(exc))
    st.caption('密钥不在此页面登记；可通过环境变量提供，或在启动评测时临时输入。不要把密钥写进 URL、提示词或额外参数。')


def evaluations_page(store):
    st.subheader('新建评测')
    models, suites = store.list_models(), store.list_suites()
    if not models:
        st.info('请先到“模型版本”登记被测模型。')
        return
    if not suites:
        st.info('请先保存一套评测题库。')
        return
    model = st.selectbox('被测训练版本', models, format_func=model_label, key='eval_model')
    suite = st.selectbox('评测题库', suites, format_func=suite_label, key='eval_suite')
    key = st.text_input('被测模型 API Key（仅本次会话）', type='password', key='eval_api_key')
    needs_judge = any(q.get('evaluator', 'judge') == 'judge' for q in suite['questions'])
    judge, judge_key = None, ''
    rubric = DEFAULT_RUBRIC
    if needs_judge:
        st.info('当前题库包含评委题，请选择已登记的评委模型。评委应掌握专项语言规范，并经过人工校准。')
        judge = st.selectbox('评委版本', models, format_func=model_label, key='eval_judge')
        judge_key = st.text_input('评委 API Key（仅本次会话）', type='password', key='eval_judge_key')
        rubric = st.text_area('评委评分规则', DEFAULT_RUBRIC, height=120, key='eval_rubric')
    else:
        st.caption('这套题库采用确定性判定，无需评委 API。')
    name = st.text_input('实验名称', value=f"{model['name']} · {suite['name']}", key='eval_name')
    a, b = st.columns(2)
    workers = a.slider('并发题数', 1, 16, 4, key='eval_workers')
    retries = b.slider('临时请求故障最大重试次数', 0, 5, 1, key='eval_retries')
    active_id = st.session_state.get('active_run_id')
    status = job_manager().status(active_id) if active_id else {'done': True}
    if st.button('启动评测', type='primary', disabled=not status['done']):
        try:
            run_id = job_manager().start(store, suite, runtime_model(model, key),
                                          judge=runtime_model(judge, judge_key) if judge else None,
                                          rubric=rubric, workers=workers, retries=retries,
                                          metadata={'name': name.strip() or model['name'], 'source':'web'})
            st.session_state['active_run_id'] = run_id
            st.rerun()
        except (ValueError, TypeError) as exc:
            st.error(str(exc))
    if st.session_state.get('active_run_id'):
        active_run_panel(store, st.session_state['active_run_id'])


@st.fragment(run_every='1s')
def active_run_panel(store, run_id):
    status = job_manager().status(run_id)
    try:
        run = store.get_run(run_id)
    except ValueError:
        run = None  # The worker may not have persisted its first snapshot yet.
    if not status['done']:
        if run:
            count = len(run.get('results', []))
            total = len(run['suite']['questions'])
            st.progress(min(count/total, 1.0), text=f'已记录 {count} / {total} 题')
        else:
            st.info('正在初始化评测任务…')
        if st.button('取消当前评测', key='cancel_'+run_id):
            job_manager().cancel(run_id)
            st.info('已请求取消。正在进行的 HTTP 请求会在返回或超时后停止。')
    elif status.get('error'):
        st.error('任务异常结束：'+status['error']+'。已写入的结果可在历史实验中查看。')
    elif not status.get('known') and run and run.get('state') == 'running':
        st.warning('执行进程已重启，本任务已中断。请重新启动评测；已记录结果仍可导出。')
    if status['done'] and st.button('准备下一次评测', key='next_'+run_id):
        st.session_state.pop('active_run_id', None)
        st.rerun(scope='app')
    if run:
        show_report(run, 'active_'+run_id)
    elif status['done'] and not status.get('error'):
        st.info('未找到当前任务记录，请在历史实验中查看或重新启动。')


def history_page(store):
    st.subheader('历史实验')
    runs = store.list_runs()
    if not runs:
        st.info('暂无实验。训练完成后，先在“新建评测”运行固定题库。')
        return
    run = st.selectbox('实验记录', runs, format_func=run_label, key='history_run')
    show_report(run, 'history_'+run['id'])


def candidate_summary(run):
    return summarize_run(run)


def comparison_page(store):
    st.subheader('实验对比与退步分析')
    all_runs = store.list_runs()
    runs = [r for r in all_runs if 'suite' in r and 'model' in r]
    if len(runs) < 2:
        st.info('至少需要两次带题库和训练版本快照的实验。请用相同题库与评测协议分别评测基础模型和新版本。')
        return
    candidate = st.selectbox('候选实验', runs, format_func=run_label, key='compare_candidate')
    others = [r for r in runs if r['id'] != candidate['id']]
    compatible = [r for r in others if compare_runs(candidate, r)['compatible']]
    mode = st.radio('基线选择', ['上一个兼容版本', '基础模型', '历史最佳', '手动选择'], horizontal=True, key='baseline_mode')
    baseline = None
    if mode == '手动选择':
        baseline = st.selectbox('基线实验', others, format_func=run_label, key='compare_baseline')
    else:
        eligible = [r for r in compatible if r.get('created', '') <= candidate.get('created', '')
                    and r.get('state') == 'finished'
                    and r['model'].get('id') != candidate['model'].get('id')
                    and candidate_summary(r)['completion_rate'] == 1.0]
        if mode == '基础模型':
            eligible = [r for r in eligible if r['model'].get('role') == 'base']
        if mode == '历史最佳' and eligible:
            baseline = max(eligible, key=lambda r: (candidate_summary(r).get('pass_rate', 0), r.get('created', '')))
        elif eligible:
            baseline = max(eligible, key=lambda r: r.get('created', ''))
        if baseline:
            st.write('**当前基线：** '+run_label(baseline))
        else:
            st.info('没有符合条件的历史基线。请先用相同题库与协议评测对应版本，或手动选择查看不兼容原因。')
            return
    with st.expander('替换标准（按业务需要调整）'):
        min_pass = st.slider('最低题目通过率', 0.0, 1.0, 0.8, 0.01, key='gate_min_pass')
        max_regress = st.slider('允许退步题占比', 0.0, 1.0, 0.05, 0.01, key='gate_max_regress')
        max_drop = st.slider('允许整体通过率下降', 0.0, 1.0, 0.02, 0.01, key='gate_max_drop')
        max_category_drop = st.slider('允许分类通过率下降', 0.0, 1.0, 0.05, 0.01, key='gate_max_category_drop')
        st.caption('这些默认值只是可调整的工程起点；不表示统计显著性。关键题未通过与评测未完成会单独影响结论。')
    compared = compare_runs(candidate, baseline, thresholds={'min_pass_rate':min_pass,
                            'max_regression_rate':max_regress, 'max_pass_rate_drop':max_drop,
                            'max_category_drop':max_category_drop})
    verdict = compared['verdict']
    getattr(st, {'pass':'success', 'fail':'error', 'insufficient':'warning'}[verdict])(
        {'pass':'满足当前替换标准', 'fail':'未满足当前替换标准', 'insufficient':'证据不足，不能据此决定替换'}[verdict])
    for reason in dict.fromkeys(compared.get('reasons', [])+compared.get('explanations', [])):
        st.write('• '+str(reason))
    if compared['compatible']:
        counts = compared['counts']
        a, b, c, d = st.columns(4)
        a.metric('新通过题', counts['improved'])
        b.metric('新增退步题', counts['regressed'])
        c.metric('判定不变', counts['unchanged'])
        d.metric('尚无有效对比', counts['unresolved'])
        delta = compared['summary'].get('pass_rate_delta')
        if delta is not None:
            st.write(f'整体通过率变化：**{delta:+.1%}**')
        categories = compared['summary'].get('category_deltas', {})
        if categories:
            st.dataframe(pd.DataFrame([{'类别': category, **values} for category, values in categories.items()]),
                         hide_index=True, width='stretch')
        labels = [('regressed', '新增退步题'), ('improved', '新通过题'), ('unresolved', '需补测题'), ('unchanged', '其余题目')]
        for field, label in labels:
            with st.expander(f"{label}（{len(compared[field])}）", expanded=field=='regressed'):
                for pair in compared[field]:
                    st.markdown(f"**{pair.get('id', '')} · {pair.get('category', '')}**")
                    st.write(pair.get('question', ''))
                    before, after = pair.get('before', {}), pair.get('after', {})
                    left, right = st.columns(2)
                    for col, title, item in ((left, '基线回答', before), (right, '候选回答', after)):
                        col.caption(f"{title} · {item.get('status', '')} · {'通过' if item.get('passed') else '未通过'}")
                        answer = item.get('answer', {})
                        col.text(answer.get('text', '') if isinstance(answer, dict) else str(answer))
                        if item.get('reason'):
                            col.write(item['reason'])
    st.download_button('导出对比报告 JSON', json_text(compared),
                       f"compare-{candidate['id'][:8]}-{baseline['id'][:8]}.json", key='compare_export')


store = Store()
initialize(store)
st.title('Model Easy Eval')
st.caption('训练后回归评测 · 固定题库 · 版本对比 · 退步追踪')
page = st.sidebar.radio('导航', ['概览', '题库管理', '模型版本', '新建评测', '历史实验', '实验对比'], key='navigation')
render_notice()
{'概览': overview, '题库管理': suite_page, '模型版本': models_page,
 '新建评测': evaluations_page, '历史实验': history_page, '实验对比': comparison_page}[page](store)
