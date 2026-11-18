LIVE07 — worker 规范化预览通过却返回 exit 1（核查与小修复）

结论：已按主代理确认，在当前产品修复 worker 自检的退出码与提示契约。原来代码和测试有意保留 raw 未确认状态，但面对仅有机械格式 warning 的提案，实际工具层仍显示失败，与“格式交 Runner”不够一致。它不是 complete 的硬性阻断，也没有公开证据证明本轮因此卡住。r02 继续使用冻结旧 helper，没有中断或升级该运行。

核查对象：FA4 run `a9a16cd1-7232-46c0-9721-1e8d98a227d9`，P1 worker attempt `a5d2ac24-7cfc-4b1b-8273-958ec327c214`。只解析公开 `tool_use` / `tool_result` 块和读取代码、提示；未读取或复述模型隐藏推理。

- [原始 events.jsonl](../../.schema_extract/live_acceptance_20261117/runs/a9a16cd1-7232-46c0-9721-1e8d98a227d9/attempts/a5d2ac24-7cfc-4b1b-8273-958ec327c214/events.jsonl) 第 11296 行，`2026-11-17T15:09:49.315Z`：`raw_passed=false`、`preview_passed=true`、`exit_code=1`、`pending_corrections=true`；内层 `check.exit_code=0`、`check.diagnostics=[]`、`notices=[]`，来源 `verified`。17 条 corrections 全是 `severity=warning`，其中 16 条 YAML wrapper quote、1 条 layout。外层 Bash `tool_result.is_error=true`，内容以 `Exit code 1` 开头。
- 第 8324 行的前次检查确有重复 Contract 对象名、文本过长等失败；第 11296 行这些诊断已消失。因此本报告只针对最后这次仅剩自动格式调整的退出码。
- 已读后续公开工具调用至第 14533 行，`15:15:15.335Z`，均为 Read/Grep，涉及 receipts、证据与比较规则；没有再次手改格式、反复自检的公开证据。不能从这些调用推断 worker 的内部判断或停顿原因。

实际 [check-helper.py](../../.schema_extract/live_acceptance_20261117/runs/a9a16cd1-7232-46c0-9721-1e8d98a227d9/attempts/a5d2ac24-7cfc-4b1b-8273-958ec327c214/materials/check-helper.py) 只是绑定冻结 `check.json` 并调用 `reading.main()`，额外参数返回 125。冻结快照为 `db0e42767335da0a1a561b4ebd659c95017805df2bc58d93697f145190db7848`。修复前已核对涉及的 reading.py、prompts.py、receipts.py、check.md、worker.md、response.md 与当前对应文件字节相同；下面旧逻辑的行号是修复前版本。

- [reading.py](../../schema_extract/tools/reading.py) 第 64–66 行：worker 预览 check 通过后，仍以 `pending_corrections` 强制 `exit_code=1`；第 146 行直接将其作为进程退出码。[test_prompt_materials.py](../../schema_extract/tools/tests/test_prompt_materials.py) 第 62 行明确断言这个 1，说明不是偶发输出错误。
- [worker.md](../../schema_extract/prompt/roles/worker.md) 规定可确定格式由 Runner 整体规范化；[check.md](../../schema_extract/prompt/operations/check.md) 说明预览未落地、Runner 后续严格重读，并让 worker 直接读取退出码。它没有明确解释“仅格式 corrections 的 exit 1 仍允许完成提案”。JSON 中 raw/preview 的区别真实，但 Bash 的失败标记仍容易诱导不必要的人工修格式。
- [response.md](../../schema_extract/prompt/operations/response.md) 只要求 checks 如实记录；[receipts.py](../../schema_extract/tools/receipts.py) 第 89 行只限定整数，第 456 行仅要求未检查时说明理由，没有“checks 出现非零就拒绝 complete”的门槛。
- [runner.py](../../schema_extract/tools/runner.py) 第 563 行检查的是模型 CLI 最终原生退出状态，不是每条 Bash 检查命令；第 617 行以后对 worker 提案独立规范化，并用实际规范化产物、来源、回执和阶段门禁决定接收。[artifacts.py](../../schema_extract/tools/artifacts.py) 第 967–973 行对 complete/approve 严格重读。reviewer 在当前 raw 上核查，[publication.py](../../schema_extract/tools/publication.py) 第 54–65 行还要求当前 P6 reviewer 批准、完整检查及来源验证。不能把 worker 自检退出码视为最终批准。

已实施的最小修改：

1. `schema_extract/tools/reading.py`：普通 P1–P5 worker 的退出码服从已完成的规范化预览检查；预览通过且来源验证成功时返回 0，保留 `raw_passed=false`、`pending_corrections=true`、全部 warning 和摘要字段。结构、权限、来源、需人工精简的文本失败继续非零。按本次审查边界，reviewer 与 P6 的现有判定保持不变。
2. `schema_extract/prompt/operations/check.md`：明确 worker 的 0 表示本阶段预览检查通过；pending corrections 由 Runner 落地，不要求人工消除；满足其他阶段要求后可提交 complete。仍须如实记录退出码，预览不表示 raw 已落地或已批准。不给模板增加字段或状态层。
3. `schema_extract/tools/tests/test_prompt_materials.py`：更新现有 warning-only 断言，并用固定 helper 的进程退出值验证 0；保留原始字节不变、raw 仍 false、corrections 仍在和来源 verified 的断言。增加文本超长、P1/P6 reviewer 对错误字段大小写的拒绝断言；原有 P4 越权版本变更拒绝断言继续通过。

`prompts.py` 生成器和 `receipts.py` 未改，Runner、规范化器和发布门禁未改，没有新增字段、状态层或格式门槛。

实际草稿验证使用临时目录复制 workspace、baseline 和 helper，只有临时 helper 的执行模块指向当前产品；阶段规则和来源仍指向原冻结材料。真实 `root.yaml` 的 SHA-256 为 `6e36c832cb037ca9ac546ce647628c904e6dcea82462481e7731d64bde63ea3f`，与第 11296 行一致。

| 临时副本检查 | 进程退出码 | 结果 |
| --- | --- | --- |
| 原冻结 worker | 1 | preview=true、raw=false、pending=true、17 条 warning |
| 新 worker | 0 | preview=true、raw=false、pending=true、17 条 warning、source verified |
| 原冻结 / 新 reviewer | 均为 0 | 输出一致，raw=true，检查当前原稿 |
| 临时副本追加非法 `UNEXPECTED_FIELD`，新 worker | 1 | `P1 changed protected fields` |
| 同一非法字段变体，原冻结 / 新 reviewer | 均为 1 | 输出一致，明确 `field not allowed` |

每次检查均确认临时副本未被 helper 改写；验证结束时再次确认真实稿件字节不变。临时目录已清理。

验证纠正了一个预期：这里的 quote/layout warning 不使原有 raw 结构检查失败。worker 的 `raw_passed=false` 是“尚有规范化字节差异，因此没有确认 raw”的保守状态，不是一次独立 raw 检查失败；新旧 reviewer 均可按原规则通过这份原稿。本次保留该既有字段语义，未把无害格式差异升级成硬门禁。

定向回归：`python -B -m unittest schema_extract.tools.tests.test_prompt_materials`，4 tests，23.420s，OK。测试使用临时 fixture 与确定性 backend；没有运行真实模型或全量回归。实际草稿的副本实验补充验证了新 helper，本轮真实全流程仍由冻结旧 helper 执行，不能称为新版本全流程验收。

本次仅修改上述 3 个产品/测试文件和本报告，未修改旧 snapshot、真实 run/state/workspace 或 schema，未中断 CLI、未提交版本。按确认范围验证完成后停止修改。
