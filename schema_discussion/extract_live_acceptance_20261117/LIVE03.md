# LIVE-03：材料导航与检查器阅读观察

结论：确认一处非阻塞的示例导航落空；未发现本次所查规则与工具冲突，也没有证据表明必须先研究检查器才能写稿。本轮无需修改产品、扩充示例材料或调整权限，继续当前冻结运行。

本次只核查以下 P1 worker 的实际 `prompt.txt`、`materials/manifest.json` 及公开工具调用和结果：

| 任务 | run | attempt |
| --- | --- | --- |
| FA4 Prefill | `8794f043-7c53-4991-85bf-886afd417fed` | `5fedff12-8b1d-4bb0-ae94-05384a590a6f` |
| QSA Prefill | `7c31f6e1-e912-43d2-ada3-7b181f8245e5` | `7f12371a-9db5-4cf1-bcc1-acc7cbaef751` |

证据目录为 `.schema_extract/live_acceptance_20261117/runs/RUN/attempts/ATTEMPT/`。下列事件行号指各自的 `events.jsonl`；仅筛取 `tool_use` / `tool_result`，未读取或复述隐藏推理。所引用的 DOC0–2、Template README/工具说明、`symbols.py`、`text_policy.py` 及材料冻结/选择代码，当前文件与两个冻结快照的对应文件字节相同。

## FA4：示例链接存在，但示例未冻结

14:01:18 UTC，FA4 在冻结的 `rules/schema_template/examples` 搜索 `Execution:`，返回 `Path does not exist`（事件 3781–3782）。[Template README](../../schema_template/README.md) 第 3、91 行确实链接 `examples/README.md`；[rule_files](../../schema_extract/tools/snapshots.py) 没有冻结 examples，本次材料清单也没有它。这是共享作者说明中的可选示例入口在 Extract 冻结环境中落空。

它不构成 P1 必需定义缺失：[P1 阶段页](../../schema_extract/prompt/phases/01_contract.md) 已指向 DOC2 的执行单元规则，[DOC2](../../schema_docs/2.contract.hardware_specification.md) 第 55 行明确给出 `Execution` 的具名映射及摘要要求，且该文件已列入材料并被读取。失败后公开记录仍继续读取固定源码，未见重复尝试该缺失入口。无需为这一次可选示例查找扩充完整案例包或打断运行。

## QSA：所查约束已有规范或反馈入口

公开调用在 14:03:06–14:07:15 UTC 查询 `symbols.py`、`values.py`、`structure.py` 和 `text_policy.py`，涉及输入输出名称碰撞、shape/ref、nullable、文本预算及名称形式（事件 3576–4844）。其中名称问题在 [DOC1](../../schema_docs/1.contract.algorithm_requirement.md) 第 107 行已明确：常量、规模、输入、输出和 symbol 在同一契约内名称唯一；与 `symbols.py` 第 202–239 行一致。DOC1 第 125–149、218–234 行也已说明属性引用与 nullable。

预算细节仅在机器定义中维护是 [Template 工具说明](../../schema_template/tools/README.md) 第 29 行的现有约定；Template README 第 45 行要求填写后按反馈精简。实际提示已嵌入[只读自检](../../schema_extract/prompt/operations/check.md)，并提供本次固定 `check_command`，明确由阶段 helper 承接普通作者命令。提示没有要求预先阅读检查器实现；所查公开记录中也没有阶段自检报错与规范相冲突的反例。因此，这组源码阅读本身不足以认定材料缺失、约束矛盾或未写稿的原因。

QSA 对 271.2KB 验收 JSON 的整文件读取超限、随后 Python Bash 请求被 `dontAsk` 拒绝，之后已改用带 `limit` 的 Read 与 Grep（事件 275–312、539–541）。现有分段阅读可继续，不需要新增大包读取工具或放宽权限。

2026-11-17：仅新增本记录；未修改 `.schema_extract`、产品提示、检查器或 schema，未调用模型、运行测试或提交 git。本次结论限于上述实际现象，不推断运行速度的原因。诊断完成，停止写入。
