# 框架问题决定台账

维护新问题和更新既有决定时，先读[问题维护入口](../README.md)。本目录按稳定 ID 保存正式决定；`partially_adopted` 的 `remaining` 与各项来源限制继续有效。

| 类别 | 内容 |
| --- | --- |
| [extract_workflow](extract_workflow/) | 提取流程、审阅、连续修订与 Design 引用 |
| [extract_tools](extract_tools/) | 调用、恢复、记账与巡检工具 |
| [schema_docs](schema_docs/) | 规范表达与 Contract 边界 |
| [schema_template](schema_template/) | 模板、结构检查和流水线图 |
| [kernel_wiki](kernel_wiki/) | 按证据记录的源实现或资料问题 |

`issues` 与 `candidates` 命令从这些 JSON 读取决定。沿用原 ID，按问题维护规则更新 `decision.outcome`、理由、版本和证据；归档整理不改变决定。

JSON 中的版本、行号与旧路径保留其证据时点含义。历史调查与旧路径的实际位置见[档案导航](../../../schema_discussion/archive/README.md)及[位置映射](../../../schema_discussion/schema_audit_and_cleanup_20261119/cleanup/LOCATIONS.json)。
