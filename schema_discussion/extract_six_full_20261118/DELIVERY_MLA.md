# MLA 交付核对

结论：`mla_expanded_fp8` 已完整交付，未发现本次核对范围内的交付阻塞。正式目录包含 [root.yaml](../../schema_wiki/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/root.yaml)、[中文 dispatch.md](../../schema_wiki/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/dispatch.md) 和 [.extraction/receipt.json](../../schema_wiki/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/.extraction/receipt.json)。本次仅核对既有交付、审批绑定和运行结清；没有重新提取或进行新的语义全审。

Run：`0d78eac5-1e45-4ac0-9725-ab885b5c52dc`；state root：`.schema_extract/v2`；最终 checkpoint：`6f4f6f303163fcb5d6d96a860a3e7337b133adea7e72a88416c76eaf8522960a`。精简机器证据见 [DELIVERY_MLA_EVIDENCE.json](DELIVERY_MLA_EVIDENCE.json)。

## 交付与审批

- 正式目录恰有上述三件套，均为普通文件。两个正文文件与最终 checkpoint 的对应文件逐字节相同，其 SHA-256 同时匹配交付 receipt、checkpoint manifest 和 P6 approval 的 artifact manifest。交付 receipt 内容等于 state 与 publication journal 保存的 proof。
- P6 reviewer attempt 为 `404ca07e-5d79-4c11-bcea-b344dba8d9f3`，回执为 `approve`，`issues=[]`、`changed_files=[]`。其 attempt/receipt 的规范 JSON 摘要与交付 receipt 绑定一致；该摘要采用 Extract 的 `digest`，不是格式化 JSON 文件的原始字节摘要。
- 当前 `ready / P6`、`published=true`；P1/P2/P3/P4/P6 均批准，P5 明确 `skipped`。P2 批准计划只选 `root`，P4 批准写作计划为 `[]`，正式完整设计共 1 份，完成 1/1。中文目录也明确没有计划中的其他完整 YAML；参数、局部和组合记录未计作完整设计。
- 共 19 个登记调用与 19 条账目，全部 `settled=true`、`outcome=completed`；`active_attempts`、`unconfirmed_attempts`、`unsettled_attempts` 均为空。最终 checkpoint、merged checkpoint、P6 批准及 P6 输入 checkpoint 一致。

| 正式文件 | SHA-256 |
| --- | --- |
| root.yaml | `4fec3eed4d4f2a2d7cf39aaa3b537e4b8485cecf10387c652b2a89acb01d583c` |
| dispatch.md | `977d6419f993afc3d7e8c1a2e2415c617ceb979f96d9e82ff7bf7bfdc87cc299` |

## 只读检查及限制

运行一次 `python3 -B schema_template/check_yaml.py schema_wiki/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/root.yaml --read-only --skip-hardware`：退出 0，1 YAML、0 规范化改动、没有写回；结构、文本预算、`from`、关联和 dispatch 检查通过。硬件区及依赖硬件的关系显式跳过，未运行 GPU 或有限图样本。

运行一次 `python3 -B schema_extract/extract_yaml.py audit 0d78eac5-1e45-4ac0-9725-ab885b5c52dc --state-root .schema_extract/v2 --verify --json`：退出 0，但 `verification.status=partial`、`errors=[]`，不能报告为全部证据已验证。范围为该 run 的 19 个登记调用及其输入/结果和当前/最终 checkpoint，共 13 个 checkpoint。两条 `unavailable` 为：

| 历史调用 | 审计保留的限制 | 后续批准 |
| --- | --- | --- |
| P2 reviewer `d3e4be0b-2e90-4d70-b01b-7bd6190aced8` | `receipt_repair`：approval 的短交接超过 1200 字符或 20 非空行限制；receipt 未被采纳，artifact validation 未完成 | P2 `3221dc8c-3413-4f8f-89e6-8e54adc69100` |
| P4 reviewer `27fa7617-d26a-4b21-8d19-c07b25f6439d` | 同上 | P4 `fef3f96f-16f6-4bac-91a8-6ea9a8ba2ab7` |

上述历史限制不等于当前调用未结清，也不替代最终批准绑定的独立核对。公开 audit 的权限声明为 `evidence_only; no approval, dispatch or publication`；本次没有创建或改变批准。

## 框架候选

公开 `candidates RUN --state-root .schema_extract/v2` 仅返回 `tuple-abi-input-typing`：候选历史仍为 `confirmed`，同时正确附带登记决定 `declined`，两者分别表示曾被 reviewer 确认提报和 Supervisor 的处置，不相互冲突。

[归档决定](../../schema_extract/issues/archive/schema_docs/tuple-abi-input-typing.json) 与运行维护状态一致：实例维护 `complete`，P1 任务 `done`，独立 verify 已批准；正式 root 的 `sage_attn_sfs` 使用 `python_tuple` 并说明四个 Python None，整数 tuple 保留 `python_int`，未由默认值擅自收窄有效域。示例建议已经并入仍待决的 [non-tensor-parameter-registration](../../schema_extract/issues/archive/schema_docs/non-tensor-parameter-registration.json)，不能把“实例维护完成”写成“示例建议已完成”。

本次只写本报告及精简证据 JSON；没有修改已发布文件、运行状态、回执、approval 或工具，也没有启动模型、新 run、维护或恢复。
