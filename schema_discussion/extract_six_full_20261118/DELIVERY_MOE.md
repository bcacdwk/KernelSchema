# MoE 交付核对

结论：`nvfp4_fused_moe` 已完整交付，本次核对未发现交付阻塞。正式目录包含 [root.yaml](../../schema_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/root.yaml)、[mega.yaml](../../schema_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/mega.yaml)、[split_k.yaml](../../schema_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/split_k.yaml)、[中文 dispatch.md](../../schema_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/dispatch.md) 和 [.extraction/receipt.json](../../schema_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/.extraction/receipt.json)。本次仅核对既有交付、审批绑定与调用结清，没有重新提取或进行新的算法全审。

Run：`74bcac91-808a-41b2-aab4-85143935ab40`；state root：`.schema_extract/v2`；最终 checkpoint：`9d9a14ee44dcb57bb65f987b71b08b9e732f3cff0b7b64058c5ae2059c52fd9f`。核对时间与精简证据见 [DELIVERY_MOE_EVIDENCE.json](DELIVERY_MOE_EVIDENCE.json)。

## 交付与审批

- 正式目录恰有上述五个普通文件，无符号链接。四份正文与最终 checkpoint 及 P6 实际审阅 workspace 的对应文件逐字节相同；SHA-256 同时匹配交付 receipt、checkpoint manifest、P6 approval 与 P6 validated 的 artifact manifest。交付 receipt 内容等于 state 与 publication journal 保存的 proof，journal 及发布摘要均与 state 一致。
- P6 reviewer attempt `f3903b16-2d79-4e64-893b-2edb2f52dbee` 为 `approve`，`issues=[]`、`changed_files=[]`。attempt 与 receipt 的规范 JSON 摘要匹配发布绑定，receipt 摘要同时匹配 P6 approval、accepted receipt 与 validated 记录。摘要使用 Extract 的 `digest`，不是格式化 JSON 文件的原始字节摘要。最终 checkpoint、merged checkpoint、P6 approval 与 P6 输入 checkpoint 一致，checkpoint 本身的身份摘要也一致。
- 当前 `ready / P6`、`published=true`，P1–P6 均为 `approved`。P2 批准选择 root 的 cfg 97 + cfg2 81 流水线、split_k 的 cfg 99 两切片形式及 mega 的 cfg 92 单核；P4 批准安排为 `[{id: split_k, wait_for: []}, {id: mega, wait_for: []}]`。两份计划的批准回执摘要、当前状态与发布选择一致。三个节点均已批准、合入，完成 3/3；中文目录登记恰好三份完整设计，无“仅计划”或“记录计划”残留。参数、局部与组合记录不计作完整设计。
- 39 个登记调用与 39 条账目 ID 一致，全部 `settled=true`、`outcome=completed`；`active_attempts`、`unconfirmed_attempts`、`unsettled_attempts`、`pending_rechecks` 均为空。只读检查前后 state revision 均为 261，state 文件字节摘要未变。

| 正式文件 | SHA-256 |
| --- | --- |
| root.yaml | `b007b14a5f1ed883c276ed2e516b6e42bc3de76412a9e8efcf766cf81e09ba11` |
| mega.yaml | `edae27c6a5eb06decb290f465dcd177d9c6085a2f99c79bff3ae8933ef9f95a4` |
| split_k.yaml | `fad2b778ed0374fe7d9ff9844b3d511deb3503276c7f3709c89bcef72606f03b` |
| dispatch.md | `530c90a5908dfd44ca6247ab41919a85b782da5615381b91c15f73705ed2724d` |

## 只读检查及审计限制

当前 Template 执行一次：`python3 -B schema_template/check_yaml.py schema_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/root.yaml --read-only --skip-hardware`。退出 0，3 YAML、0 规范化改动、没有写回；结构、文本预算、`from`、关联与 dispatch 检查通过。硬件区与依赖硬件的关系显式跳过，未运行 GPU、有限图样本、数值或性能验证。

公开 audit 仅对本 run 执行一次：`python3 -B schema_extract/extract_yaml.py audit 74bcac91-808a-41b2-aab4-85143935ab40 --state-root .schema_extract/v2 --verify --json`。退出 0，但 `verification.status=partial`、`errors=[]`，不能报告为全部证据已验证。范围为 39 个登记调用及其输入/结果和当前/最终 checkpoint，共 24 个 checkpoint；九条 `unavailable` 均为 `receipt_not_accepted; artifact validation not completed`：

| 历史调用 | 审计保留的拒绝原因 |
| --- | --- |
| P2 reviewer `426c719f-196a-4d88-b500-bf4f2c871a04` | `receipt_repair`：approval 短交接超过 1200 字符或 20 非空行限制 |
| P3 worker `1bba285e-6616-4333-a4a0-da4f0b1419a2` | `proposal_rejected`：`missing bound implementation entry anchor` |
| P3 reviewer `49c0cfb0-60c4-4f64-952c-2770585c02a0` | `receipt_repair`：同上短交接长度限制 |
| P3 reviewer `9869bc70-3cc4-4ac1-a859-3efcd9f99639` | `receipt_repair`：同上短交接长度限制 |
| P4 reviewer `a55a6e39-8c33-49c7-bf2c-2d33c080ac65` | `receipt_repair`：同上短交接长度限制 |
| P5 split_k worker `a4ac10eb-cd4b-4c86-b789-4e1856fa9ded` | `proposal_rejected`：同上入口证据缺项 |
| P5 mega worker `ee3a59f1-de76-43b2-b2c2-1989b2c11bd5` | `proposal_rejected`：同上入口证据缺项 |
| P5 mega reviewer `d05cfb3b-1dc8-4745-840b-1b471c8b965b` | `proposal_rejected`：同上入口证据缺项 |
| P6 reviewer `62282994-9c30-43cf-a920-b25703fea430` | `proposal_rejected`：`conclusion/issue evidence index is outside source_evidence` |

相关阶段和节点后续均已取得批准，当前批准 ID 保存在证据 JSON；历史证据缺口仍然保留，不等于当前调用未结清。最终批准与发布绑定已另行核对。公开 audit 的权限为 `evidence_only; no approval, dispatch or publication`，本次没有产生新的批准；其调用范围也不包括 Supervisor 或外部账单，未据此汇总费用或宣称完整用量账单。

## 候选与既有边界

公开 `candidates RUN --state-root .schema_extract/v2` 返回两个 ID，均为 reviewer `confirmed`，并正确附带正式归档决定 `resolved / 2.5.1`：

- [normalizer-diff-not-visible](../../schema_extract/issues/archive/schema_template/normalizer-diff-not-visible.json)：当前工具已改善长文本规范化警告的差异展示；本 run 仍使用原冻结模板，未热更新。
- [inplace-output-input-name-diagnostic](../../schema_extract/issues/archive/schema_template/inplace-output-input-name-diagnostic.json)：当前工具已增强 Contract 撞名诊断，提示保留 Input/Output 角色并分别命名；该决定不改变合法性规则或冻结工具。

已接受回执共五次候选提及，ID 集合与公开结果一致；两项登记内容与命令返回一致，没有未登记候选。`confirmed` 表示提报确认，`resolved` 表示已登记的工具处置，两者不冲突；本次未重跑此前修复测试，也未重新裁决已归档决定。

正式正文及 P6 交接继续披露：外部验收比较器公式和相对误差基准缺失，Tolerance 为 `source-only`；activation 域的来源矛盾由 Contract 的两式并集表达；MEGA host 只检查 hidden_dim 为 256 倍数，而 kernel 需要 512 倍数，声明值 2048/4096/7168 均满足；ticket finalize 的八槽 staging 依赖声明域 `slot_count=8`。源码自述的 bit-exact 与性能比值没有在本次独立复现。交付完整不表示这些来源边界已消失。

P6 交接还保留两项不影响其放行的文字打磨：split_k 的 ACT `load:partials` 描述写成 `split_count float4 loads`，其交接指出源码为每切片 up/gate 各一条、共 `2*split_count`，精确数量可由 Math 恢复；Fc2 的未生产 act 字节措辞按 tile 范围阅读，全局陈旧行分析见 Slice reduction。这里仅保留既有 reviewer 判断，没有开展新的算法审阅或改稿。

本次只写本报告和精简证据 JSON；未修改正式产物、工具、state、frozen、回执、批准或 issue 台账，未启动模型、新 run、维护或恢复，未提交或 push。
