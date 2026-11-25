import io
import json
import os
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import pandas as pd
import streamlit as st
from engine import evaluate, DEFAULT_RUBRIC

st.set_page_config(page_title='Model Easy Eval',layout='wide')
DB = os.environ.get('EVAL_DB','eval.sqlite3')
FIELDS = ['category','question','reference','criteria']

def save(run):
    with sqlite3.connect(DB) as c:
        c.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, data TEXT NOT NULL)')
        c.execute('INSERT OR REPLACE INTO runs VALUES (?,?)',(run['id'],json.dumps(run,ensure_ascii=False)))

def load():
    with sqlite3.connect(DB) as c:
        c.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, data TEXT NOT NULL)')
        return [json.loads(r[0]) for r in c.execute('SELECT data FROM runs ORDER BY rowid DESC')]

def config(title,prefix):
    with st.expander(title,expanded=True):
        base = st.text_input('API Base URL', 'http://localhost:8000/v1',key=prefix+'url')
        key = st.text_input('API Key',type='password',key=prefix+'key')
        model = st.text_input('Model ID',key=prefix+'model')
        a,b,c = st.columns(3)
        temp = a.number_input('Temperature',0.0,2.0,0.0,0.1,key=prefix+'temp')
        tokens = b.number_input('Max tokens',1,131072,4096,key=prefix+'tokens')
        timeout = c.number_input('Timeout (s)',1,1800,120,key=prefix+'timeout')
        system = st.text_area('System prompt（仅被测模型使用）',key=prefix+'sys') if prefix!='judge' else ''
        extra = st.text_area('额外请求参数 JSON（如 top_p）','{}',key=prefix+'extra')
        try:
            extra = json.loads(extra)
            if not isinstance(extra,dict): raise ValueError()
            if set(extra)&{'model','messages','stream','api_key','key','headers'}: raise ValueError()
        except (ValueError,TypeError):
            st.error('额外参数必须为 JSON 对象，且不得覆盖核心字段或包含密钥。')
            extra = None
        return dict(base_url=base,key=key,model=model,temperature=temp,max_tokens=int(tokens),timeout=timeout,system=system,extra=extra)

def sanitize(c):
    return {k:v for k,v in c.items() if k!='key'}

def report(run):
    results=run['results']
    rows=[]
    for r in results:
        rows.append(dict(model=r['model'],category=r['question']['category'],question=r['question']['question'],status=r['status'],score=r.get('score'),reason=r.get('reason',''),seconds=r.get('answer',{}).get('seconds'),answer=r.get('answer',{}).get('text',''),error=r.get('error','')))
    if rows:
        df=pd.DataFrame(rows)
        summary=df.groupby('model').agg(total=('status','size'),success=('status',lambda s:(s=='success').sum()),mean_score=('score','mean'),mean_seconds=('seconds','mean'))
        st.dataframe(summary)
        good=df[df.status=='success']
        if not good.empty:
            st.bar_chart(good.pivot_table(index='category',columns='model',values='score',aggfunc='mean'))
        st.dataframe(df,use_container_width=True)
        for i,r in enumerate(results):
            with st.expander(f"{i+1}. {r['model']} | {r['status']} | {r['question']['question'][:60]}"):
                st.json(r)
        st.download_button('导出 CSV',df.to_csv(index=False).encode('utf-8-sig'),f"{run['id']}.csv",key='csv'+run['id'])
    st.download_button('导出完整 JSON',json.dumps(run,ensure_ascii=False,indent=2),f"{run['id']}.json",key='json'+run['id'])

st.title('Model Easy Eval')
st.caption('训练后模型评测 · LLM-as-a-Judge · 多模型对比')
page=st.sidebar.radio('导航',['新建评测','历史实验'])
if page=='历史实验':
    runs=load()
    if not runs: st.info('暂无实验记录')
    else:
        chosen=st.selectbox('实验',range(len(runs)),format_func=lambda i:f"{runs[i]['name']} | {runs[i]['created']} | {runs[i]['state']}")
        report(runs[chosen])
    st.stop()

st.subheader('1. 模型配置')
n=st.number_input('被测模型数量',1,8,1)
targets=[config(f'被测模型 {i+1}',f'target{i}') for i in range(n)]
judge=config('评委模型','judge')
st.subheader('2. 评测数据集')
if 'questions' not in st.session_state:
    st.session_state.questions=pd.DataFrame([['通用能力','解释 TCP 和 UDP 的区别','','事实准确、表述清楚'],['代码生成','使用 Python 编写二分查找函数','','逻辑正确、边界处理正确']],columns=FIELDS)
if 'editor_version' not in st.session_state: st.session_state.editor_version=0
file=st.file_uploader('导入 CSV / JSON / JSONL',type=['csv','json','jsonl'])
if st.button('载入文件（替换当前数据集）',disabled=file is None):
    try:
        raw=file.getvalue().decode('utf-8-sig')
        if file.name.endswith('.csv'): data=pd.read_csv(io.StringIO(raw),dtype=str)
        elif file.name.endswith('.jsonl'): data=pd.DataFrame([json.loads(line) for line in raw.splitlines() if line.strip()])
        else: data=pd.DataFrame(json.loads(raw))
        if 'question' not in data: raise ValueError('缺少 question 字段')
        st.session_state.questions=data.reindex(columns=FIELDS).fillna('').astype(str)
        st.session_state.editor_version+=1
    except Exception as e: st.error(f'导入失败：{type(e).__name__}')
questions=st.data_editor(st.session_state.questions,num_rows='dynamic',hide_index=True,use_container_width=True,key=f"editor{st.session_state.editor_version}")
st.download_button('导出当前题库',questions.to_csv(index=False).encode('utf-8-sig'),'questions.csv')
st.subheader('3. 评测规则与执行')
rubric=st.text_area('Judge system prompt',DEFAULT_RUBRIC,height=140)
name=st.text_input('实验名称','训练后评测')
workers=st.slider('并发任务数',1,16,4)
if st.button('启动评测',type='primary'):
    configs=targets+[judge]
    qs=[q for q in questions.fillna('').astype(str).to_dict('records') if q['question'].strip()]
    if not qs or any(not c['model'].strip() or not c['base_url'].startswith(('http://','https://')) or c['extra'] is None for c in configs):
        st.error('请填写有效的模型 ID、API URL、请求参数，并至少保留一道题。')
        st.stop()
    if len({c['model'] for c in targets})!=len(targets):
        st.error('本版本要求被测模型 ID 不重复，以避免统计混淆。'); st.stop()
    run=dict(id=uuid.uuid4().hex,name=name,created=datetime.now(timezone.utc).isoformat(),targets=[sanitize(c) for c in targets],judge=sanitize(judge),rubric=rubric,questions=qs,results=[],state='running')
    save(run)
    total=len(targets)*len(qs)
    progress=st.progress(0,text=f'0 / {total}')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(evaluate,q,t,judge,rubric) for t in targets for q in qs]
        for i,f in enumerate(as_completed(futures),1):
            run['results'].append(f.result()); save(run)
            progress.progress(i/total,text=f'{i} / {total}')
    run['state']='finished'; save(run)
    report(run)
