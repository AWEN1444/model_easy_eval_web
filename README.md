# Model Easy Eval Web

Python + Streamlit + SQLite 的本地单用户模型评测 MVP。支持 OpenAI 兼容 Chat Completions 接口、多模型、可编辑题库、CSV/JSON/JSONL 导入、LLM Judge、并发评测、历史及导出。

## 启动

Python 3.10+：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.address 127.0.0.1
```

Windows PowerShell 激活命令：`.venv\Scripts\Activate.ps1`。

浏览器打开 http://localhost:8501 。API Base URL 示例 `http://localhost:8000/v1` 或供应商的兼容端点。填写实际模型 ID；API key 只保留在当前会话，不作为独立字段写入数据库。不要在额外参数、提示词或 URL 中填密钥。

## 题库

CSV 字段：category,question,reference,criteria。仅 question 必填。JSON 为对象数组，JSONL 为每行一个对象。示例见 examples.json。

## 评测说明

每个题目独立调用被测模型，随后把问题、参考答案、标准及候选回答交给评委。评分为 0–100；请求失败和无效评分不计入均值，另显示成功数。结果逐条写入 SQLite，保留原始回答、评委输出、配置快照及用量。每个模型 ID 必须不同。评委评分有偏差，不代表代码编译或运行通过。请使用独立测试集并进行人工校准。题目和回答将发送至所配置的评委 API，避免发送未经授权的数据。

## 限制

本地可信单用户使用；没有认证、后台任务队列、取消/重试、跨实验自动对齐比较或模型连通性测试。页面关闭或进程中断不保证任务继续，已写入的结果可在历史页查看。自定义 URL 不适合直接暴露给不可信用户。没有执行生成代码。不要公开部署本版本。

## 测试

```bash
python -m unittest discover -s tests -v
```
