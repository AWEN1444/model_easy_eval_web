# Model Easy Eval · v0.2

面向每轮模型训练的本地评测平台：登记 checkpoint → 运行固定题库 → 对比基础模型、上一训练版本或历史最佳 → 定位新增退步题 → 导出报告。

## 已实现

- **持久化题库**：CSV / JSON / JSONL 导入、网页编辑、稳定题目 ID、不可覆盖的题库版本、内容指纹及实验快照。
- **训练版本登记**：checkpoint / adapter、训练数据与配置、模型角色；不同权重允许使用相同的 API Model ID。
- **可选择的判定方式**：精确文本、严格 JSON、包含规则、正则、Python 语法、LLM Judge，以及本机自定义判定器插件。
- **可靠执行**：并发、暂时故障有限重试、网页后台运行与取消、逐题写入 SQLite。调用错误、截断、判定错误与模型答错分别记录。
- **逐题对比**：新通过、退步、未解决、需补测；整体与分类变化、关键题检查、可调整的替换门槛。
- **训练脚本入口**：CLI、自动基线选择、机器可读 JSON 结果和退出码；导出 JSON / CSV / Markdown 报告。
- **内置起步题库**：80 道通用工程回归题，8 类各 10 题；快检集是其中 20 道，**合计 80 道不同题**。涵盖数值推理、逻辑推理、结构化输出、信息提取、代码理解、边界条件、文本处理、接口参数。

内置题库有明确参考值，采用确定性判定，不需要评委 API。它用于基础回归，**不是标准学术基准，也不代表完整通用能力或自研 DSL 能力**。`examples.json` 保留旧版题库导入示例；启动页面现在使用持久化的内置题库。

## 安装与网页使用

Python 3.9+（本次验证环境 Python 3.9、Streamlit 1.50）：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.address 127.0.0.1
```

Windows PowerShell 激活：`.venv\Scripts\Activate.ps1`。打开 http://localhost:8501 。

1. 在“模型版本”登记基础模型及 checkpoint；填入实际 OpenAI 兼容 API 地址和 Model ID。
2. 在“新建评测”选择 `通用工程回归 · 快速20题`，运行基础模型，建立第一次基线。
3. 将训练后的权重加载到模型服务，登记一个**新的版本**，再运行同一题库。
4. 在“实验对比”选择基础模型／上一版／历史最佳，查看新增退步题和替换结论。
5. 需要专项评测时，在“题库管理”导入真实任务，保存为新题库版本；正式比较时两边都在该版本补测。

登记 checkpoint 不会自动加载权重，也无法从 API 别名证明服务端正在使用该权重。请在训练流程中确保实际服务与登记信息一致。页面支持单个版本的一次评测，多个版本分别执行后进行比较。

## 接入每次训练结束后的流程

先编辑 `examples/model-base.json` 和 `examples/model-candidate.json` 的模型地址、实际 checkpoint 和版本 ID。无需认证的本地接口可保留空的 `api_key_env`。商业 API 配置 `api_key_env` 为环境变量名称，并在运行环境中设置该变量；不要把真实密钥写入配置文件。

首次建立基础模型基线：

```bash
python cli.py init
python cli.py run \
  --model-config examples/model-base.json \
  --suite general-quick --suite-version 1.0.0 \
  --record-baseline --output reports/base-quick
```

每次训练完成并将该 checkpoint 加载到模型服务后，调用：

```bash
python cli.py run \
  --model-config examples/model-candidate.json \
  --suite general-quick --suite-version 1.0.0 \
  --baseline-policy base \
  --name sft-step-1200 --workers 4 --retries 1 \
  --output reports/sft-step-1200
```

每个新训练产物必须修改配置中的 `id`、`checkpoint`、训练数据及配置记录。相同版本 ID 不允许覆盖权重或推理配置。服务端 Model ID 可以相同。

- `--baseline-policy base`：同题库和协议下，最近的完整基础模型实验。
- `previous`（默认）：最近的、不同训练版本的完整实验。
- `best`：不同训练版本中全题通过率最高的完整实验。
- `--baseline RUN_ID`：显式指定实验；可重复传入多个基线。显式指定支持同版本重跑比较。
- `--judge MODEL_VERSION_ID` 或 `--judge-config FILE`：仅含 `judge` 题时需要。
- `--rubric-file FILE`：固定评委系统提示词。
- `--db PATH` 放在子命令之前，或设置 `EVAL_DB` 环境变量；网页和 CLI 使用同一个数据库即可共享历史。

进程向标准输出写一行 JSON，报告目录含 `report.json`、`results.csv`、`report.md`。退出码：

| 退出码 | 含义 |
| --- | --- |
| 0 | 满足所设替换门槛；或 `--record-baseline` 完整记录成功 |
| 1 | 输入、依赖或运行配置错误 |
| 2 | 完整可比的评测未满足门槛 |
| 3 | 证据不足：无基线、未完成或评测协议不一致 |
| 130 | 用户请求取消 |

第一次普通运行没有可比基线时返回 3，不会声称训练已进步。用 `--record-baseline` 明确创建初始基线。Ctrl+C 会请求取消并保存结果，已开始的 HTTP 请求最多等待其超时。

完整评测可改用 `general-core`，但需先为基础模型建立同题库基线。20 题和 80 题属于不同评测方案，总分不能直接横比。

已有实验的其他命令：

```bash
python cli.py suites
python cli.py models
python cli.py import-suite examples/custom-suite.json
python cli.py register-model --config examples/model-candidate.json
python cli.py compare CANDIDATE_RUN_ID BASELINE_RUN_ID --output reports/comparison
python cli.py report RUN_ID --baseline BASELINE_RUN_ID --output reports/export
```

## 题库结构与判定器

完整示例见 `examples/custom-suite.json`。CSV / JSONL 也可导入，旧版 `category,question,reference,criteria` 字段继续兼容，默认使用评委；未提供的题目 ID 由内容确定性生成。正式维护时建议显式提供稳定 ID。

```json
{
  "id": "my-regression",
  "name": "我的固定回归集",
  "version": "1.0.0",
  "questions": [{
    "id": "format-001",
    "category": "结构化输出",
    "question": "仅输出一个JSON对象，result为整数5。",
    "evaluator": "json",
    "expected": {"result": 5},
    "critical": true,
    "options": {"allow_fences": false}
  }]
}
```

| evaluator | 配置与语义 |
| --- | --- |
| `exact` | `expected` 字符串，默认去除首尾空白、区分大小写；`options.trim=false` 可检查空白 |
| `json` | `expected` 为完整 JSON 数据；忽略对象键顺序，严格比较字段、数组顺序、类型和值；默认拒绝围栏、额外文字、重复键、NaN；整数与浮点数严格区分 |
| `contains` | `expected` 为非空字符串数组，要求全部包含；适合明确片段规则，不等于语义正确 |
| `regex` | `expected` 为正则，使用 `re.search`；检查完整输出请自行使用首尾锚点 |
| `python_syntax` | 使用 Python AST 检查语法；不执行代码，也不判断功能、函数选择或业务正确性 |
| `judge` | 结合 `reference`、`criteria`、固定 rubric 评分；`options.pass_score` 默认 80 |

`exact` / `contains` / `regex` 可用 `options.case_sensitive=false`。`json` 可显式设置 `options.allow_fences=true` 放宽围栏要求。CSV 的 `json` / `contains` 题 `expected` 列、`options` 列应填写 JSON 文本。

题库编辑器中的 `expected_json` 填 JSON：字符串要加引号，例如 `"YES"`；对象填 `{"result":5}`。

### 接入 DSL 或其他真实判定器

现有项目不会假装掌握你的 DSL。需要提供语言规范、接口清单、真实任务、可信参考，以及可用的解析器／编译器／测试环境。可以先用有参考和明确标准的评委题，逐步接入确定性判定。

可信的本机插件可以注册一个判定器：

```python
# my_dsl_checker.py
from engine import register_evaluator
from your_test_adapter import check_candidate

def check(question, answer_text):
    passed, details = check_candidate(answer_text, question["options"])
    return {"score": 100 if passed else 0, "passed": bool(passed), "reason": details}

register_evaluator("dsl_tests_v1", check)
```

在题库中使用 `evaluator: "dsl_tests_v1"`，通过 `python cli.py run ... --plugin my_dsl_checker` 加载。模块需在 Python 导入路径中；上例的适配器由专项项目提供，不是本仓库内置依赖。插件逻辑变化时，更新判定器名称或题库版本／规则中的版本标记，防止新旧逻辑混比。插件在本机执行，生成代码的隔离和资源限制由实际测试适配器实现。网页默认不加载自定义插件。

## 指标与替换判断

- **全题通过率** = 通过题数 / 全部计划题数；调用失败和未完成不会缩小分母。
- **有效判定完成率** = 得到有效判定的题数 / 全部计划题数；答错也是有效判定。
- 条件平均分只反映已评分题目，不作为唯一替换依据。
- 按稳定题目 ID 对齐，分别展示“错→对”“对→错”“判定不变”“无法判断”。
- 题库 ID / 版本 / 内容、生成参数、系统提示词、评委版本与 rubric、引擎协议不一致时，结论为证据不足。确定性题库不要求评委相同。
- 两次实验均完整完成，才能判断是否满足替换门槛。关键题未通过会阻止替换。

默认门槛仅为可调整的工程起点：通过率 ≥80%、新增退步题比例 ≤5%、整体下降 ≤2个百分点、各分类下降 ≤5个百分点。网页可调整；CLI 使用 `--min-pass-rate`、`--max-regression-rate`、`--max-pass-rate-drop`、`--max-category-drop`，数值均为 0–1 比例。达到门槛不等于差异具有统计显著性，也不自动上线模型。

建议保存独立训练集、开发集和保留测试集。新失败题可沉淀到回归库；若加入训练，应另留未见样本验证泛化。正式结论需结合真实业务覆盖、人工校准及关键样本重复测试。平台不自动检测训练数据污染，也不会用一套通用题库代表所有能力。

## 持久化与运行边界

SQLite 默认 `eval.sqlite3`，保存模型配置、题库版本与逐题结果快照；API Key 从环境变量或网页当前会话读取，不作为独立凭证字段存入数据库或报告。不要把密钥放进提示词、题目、模型输出、训练配置或 URL。

网页后台任务运行在当前服务进程，关闭浏览器不终止执行；**服务重启不会自动恢复任务**，已保存的部分结果仍可查看和导出。取消不能强行中断已开始的 HTTP 请求；暂时故障（网络、429、5xx）才会有限重试。旧版实验继续可读，但缺少必要快照时需要重测才能比较。

本地单用户使用，尚无多人认证、分布式任务队列、自动权重部署、标准学术基准执行器或内置生成代码执行沙箱。不要直接公开部署。需要学术基准时，应接入其官方执行协议，不能把这里的评委分数当作对应基准分数。

## 验证

```bash
python -m unittest discover -s tests -v
```

测试覆盖确定性规则、严格 JSON、重试与取消、快照脱敏、完整分母、不可变题库和模型版本、协议兼容、逐题退步门槛、CLI 基线→候选→报告，以及 Streamlit 页面与后台任务交互。模型调用采用模拟响应验证，不能代替你的真实模型和专项 DSL 的验收。
