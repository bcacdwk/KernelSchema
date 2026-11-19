# GDN 交付核对

结论：`gdn_prefill_bf16_state_bf16` 已完整交付，本次核对未发现交付阻塞。正式目录包含 [root.yaml](../../schema_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/root.yaml)、[segmented.yaml](../../schema_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/segmented.yaml)、[中文 dispatch.md](../../schema_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/dispatch.md) 和 [.extraction/receipt.json](../../schema_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/.extraction/receipt.json)。本次仅核对既有交付、审批绑定与调用结清，没有重新提取或进行新的语义全审。

Run：`68edadab-4371-402c-8acf-f73eb5ec9f92`；state root：`.schema_extract/v2`；最终 checkpoint：`5ddf5963d2ad9da5875ad931980f326e7fde91ee428ed50e86c4a58554b7a243`。核对时间与精简证据见 [DELIVERY_GDN_EVIDENCE.json](DELIVERY_GDN_EVIDENCE.json)。

## 交付与审批

- 正式目录恰有上述四个普通文件。三份正文与最终 checkpoint 对应文件逐字节相同，SHA-256 同时匹配交付 receipt、checkpoint manifest 和 P6 approval 的 artifact manifest。交付 receipt 内容等于 state 与 publication journal 保存的 proof，发布摘要与 state 一致。
- P6 reviewer attempt `c56e0d66-fd1d-429f-9c46-a492e22925a9` 为 `approve`，`issues=[]`、`changed_files=[]`；attempt 与 receipt 的规范 JSON 摘要匹配发布绑定，receipt 摘要同时匹配 P6 approval 与 accepted receipt。摘要使用 Extract 的 `digest`，不是格式化 JSON 的原始字节摘要。最终 checkpoint、merged checkpoint、P6 approval 与 P6 输入 checkpoint 一致。
- 当前 `ready / P6`、`published=true`，P1–P6 均为 `approved`。P2 批准选择 root 整体式路径与 segmented 分段窄臂路径，P4 批准安排为 `[{id: segmented, wait_for: []}]`，与当前状态及交付选择一致。两节点均已批准、合入，完成 2/2；中文目录登记恰好两份完整设计，无“仅计划”或“记录计划”残留。参数、局部与组合记录不计作完整设计。
- 24 个登记调用与 24 条账目 ID 一致，全部 `settled=true`、`outcome=completed`；`active_attempts`、`unconfirmed_attempts`、`unsettled_attempts`、`pending_rechecks` 均为空。只读检查前后 state revision 均为 170。

| 正式文件 | SHA-256 |
| --- | --- |
| root.yaml | `f1e474efd50526bd1f4609ad9868a11f3a90d7d45a45a51cac590603adacedbc` |
| segmented.yaml | `6cc1f2a728c233e221f0ef05981f4a2c4804c6832747074a611d3ea3679c615f` |
| dispatch.md | `4df4f7bb92bb2ac4e2200e42249d4fe05fbce069fba7b61eadbe8dc3093b7804` |

## 只读检查及限制

当前 Template 执行一次：`python3 -B schema_template/check_yaml.py schema_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/root.yaml --read-only --skip-hardware`。退出 0，2 YAML、0 规范化改动、没有写回；结构、文本预算、`from`、关联与 dispatch 检查通过。硬件区与依赖硬件的关系显式跳过，未运行 GPU、有限图样本、数值或性能验证。

公开 audit 仅对本 run 执行一次：`python3 -B schema_extract/extract_yaml.py audit 68edadab-4371-402c-8acf-f73eb5ec9f92 --state-root .schema_extract/v2 --verify --json`。退出 0，但 `verification.status=partial`、`errors=[]`，不能报告为全部证据已验证。范围为 24 个登记调用及其输入/结果和当前/最终 checkpoint，共 15 个 checkpoint；五条 `unavailable` 均为 `receipt_not_accepted; artifact validation not completed`：

| 历史调用 | 审计保留的拒绝原因 |
| --- | --- |
| P1 reviewer `87b62d5b-1886-4b8e-a3f7-92c75fcc344d` | `proposal_rejected`：缺少四项 `document_*` 结论 |
| P2 reviewer `16f51bb2-9cb3-43f6-a654-97b48cd9c6d7` | `receipt_repair`：approval 短交接超过 1200 字符或 20 非空行限制 |
| P3 worker `4e20df78-0155-4cbe-a451-1ac94487a94a` | `proposal_rejected`：`missing bound implementation entry anchor` |
| P3 reviewer `6243ddba-9b7f-4eb3-a5a4-8a0cf56503cd` | `proposal_rejected`：同上入口证据缺项 |
| P5 segmented worker `0d8cc975-0ef4-499f-9347-a6c690cdd6f3` | `proposal_rejected`：同上入口证据缺项 |

相关阶段后续均已取得批准，后续批准 ID 保存在证据 JSON；历史证据缺口仍然保留，不等于当前调用未结清。最终批准与发布绑定已另行核对。公开 audit 的权限为 `evidence_only; no approval, dispatch or publication`，本次没有产生新的批准。

## 候选与来源限制

公开 `candidates RUN --state-root .schema_extract/v2` 返回 `[]`，已接受回执中的 `framework_candidates` 也全部为空；本 run 没有须登记或继续处置的 reviewer 确认候选。该结果仅覆盖本 run 的候选，不改变全局历史 issue 的处置状态。

正式正文继续披露两项既有来源限制：三个输出的验收容差为 `source-only`，缺少外部比较器公式、相对误差基准和逐输出路由；实现拒绝“`cu_seqlens` 为张量而 chunk 元数据为 `None`”的组合，Contract 的 nullable 有效域仍予保留。交付完整不表示这些来源限制已解决，本次也未重新裁决它们。

本次只写本报告和精简证据 JSON；未修改正式产物、工具、状态、回执或批准，未启动模型、新 run、维护或恢复，未提交或 push。
