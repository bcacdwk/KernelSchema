# 框架问题台账

只有当现行规则会导致错误、遗漏或重复劳动，才登记能解决它的最小改动。一个可复查的反例就够。算子自身的实现差异和证据缺项写入任务回执。

| 类别 | 维护对象 |
| --- | --- |
| `schema_docs` | schema 字段和语义规则 |
| `schema_template` | 模板和检查器 |
| `kernel_wiki` | 算子资料、来源和导航 |
| `extract_workflow` | 提取步骤、角色和交接 |
| `extract_tools` | 执行、读取、恢复和巡检工具 |

**提报。** Worker、reviewer 和 Supervisor 在回执或巡检决定的 `framework_candidates` 中提报，没有必要的建议就返回 `[]`。主 agent 在清理运行记录前用 `python3 -m schema_extract candidates RUN_ID` 查看候选及其来源。

**登记。** 按稳定 ID 和根因查重，用 `apply_patch` 将合格的新项保存为 `<类别>/<id>.json`。字段为 `id`、`category`、`problem`、`evidence`、`proposal`、`necessity`。`id` 用小写 ASCII 字母数字，可用 `-` 或 `_` 分词。证据写到能独立复查，建议写清不改会带来什么影响。

**决定。** 维护者将文件移动到 `archive/<原类别>/<id>.json`，保留原字段并增加 `decision`。同一发现再次出现时沿用已有决定。用 `python3 -m schema_extract issues` 查看当前待决项和归档计数。

| `decision.outcome` | 必填的决定依据 |
| --- | --- |
| `declined` 不解决 | `reason` 说明理由 |
| `resolved` 完全解决 | `reason` 说明改动及核验结果 |
| `partially_adopted` 部分采纳 | `reason` 说明取舍，`adopted` 与 `remaining` 分别说明已落实和未采纳的范围 |

迁移前的单文件台账保留为[归档原文](archive/ISSUES.md)。
