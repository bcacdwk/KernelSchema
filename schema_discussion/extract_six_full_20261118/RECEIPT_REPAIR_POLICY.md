# 回执有限修复的分类评估与当前源码修复

## 结论

**这是现有 repair 触发分类的遗漏，可以用很小的改动修复。** 三次提案都应继续被原校验拒绝；拒绝后可以把具体缺项交回同一角色，在既有调用余额、两次自动修复额度和精确会话约束内重交。修复的是恢复路由，不是把缺项当成已完成。

现有机制并不合理地局限于 JSON 语法：`receipt_repair.md` 原本就要求修正“提案结构或证据定位错误”，`ReceiptFormatError` 原有用途也包括缺 summary、缺检查说明和不合要求的 handoff。另一方面，`read_evidence` 把坐标问题和固定文件读取/摘要问题都报告为 `ValueError`，上层统一转换成普通 `ExtractError`；阶段必需来源分类、入口锚点和结论缺项也抛普通 `ExtractError`。因此这些错误直接进入 `proposal_rejected`，没有触发已有有限重交。

**本文及源码改动仅适用于之后以当前源码新建的运行。本批六算子的冻结 executor、rules、state、attempt、产物与进程均未改动，也未热更新。** 当前批次恢复继续由主 Supervisor 按冻结公开入口执行。本评估沿用已记录的原生成功、nonce/schema 有效及进程已结清事实，没有重复读取或操作运行状态。

## 本次三个实例

| 原提案问题 | 原先停止原因 | 当前源码的处理 |
| --- | --- | --- |
| GDN P1 reviewer 缺四个 `document_*` 明确结论 | 缺 supported stage conclusions，普通拒收 | 同一 reviewer 自行完成判断后重交；已有 artifact anchors 仍不能代替结论 |
| FMHA P1 worker 未给绑定入口提供 `implementation` 锚点 | 缺 bound implementation entry anchor，普通拒收 | 同一 worker 据实际阅读补齐/修正证据；不由工具重分类或代选源码 |
| FMHA P1 reviewer 引用 `root.yaml:167–205`，文件实际为 204 行 | `read_evidence` 范围越界，普通拒收 | 返回原样的引用索引、路径、范围和文件行数，由模型重新选取；不自动截断到 204 |

前两项事实与原 run 恢复依据见 [RECEIPT_RECOVERY_0810.md](RECEIPT_RECOVERY_0810.md)。第三项为后续记录的 attempt `8bee8bce-75e6-4f97-af52-dbcdfe16384b`；这里不改写其历史回执或账目。

## 最小改动边界

1. `reading.py` 新增 `EvidenceSelectionError(ValueError)`，只用于文件已通过原路径、读取稳定性和摘要检查之后的行范围、`quote_offset` 或空摘录选择错误。其他读取、路径、symlink、摘要和编码错误仍按原方式失败。
2. `receipts.py` 仅将这一明确类型转为已有 `ReceiptFormatError`；将通用回执中的证据索引越界、缺必需来源分类、缺分类的绑定入口、缺受支持的阶段结论、reviewer 缺必需文件引用/源码证据归入同一类。判断条件和诊断内容不变，也不按本次三个错误字符串匹配。
3. 更新既有 repair 提示，明确必需判断与证据须由当前角色核实；不能支持原决定时可据实选择该角色允许的退修或阻塞决定。执行协议补充分类边界。

没有新建重试框架，没有改 `runner.py`、`maintenance.py` 的恢复逻辑，没有改字段、动态 schema、证据/文字长度预算、模型身份或调用上限。工作区原有 maintenance 只读快路径和 scoped 恢复反馈修复均保留。

这不是将所有 `ExtractError` 视为可修：来源/身份/会话不符、权限与文件差异越界、实际 Design 来源诊断、计划或实例改变、比较器语义、reopen 权限、P6 revision scope、裁决与签收限制仍在原路径。`validate_comparisons` 等同时核对回执与产物语义的检查未重新分类。已生成 receipt 的摘要、quote、行范围等遭篡改时，原 `_anchor` 检查仍拒绝；仅模型 proposal 的合法文件内选择错误接入修复。

有效的 reviewer `revise` / `blocked` 继续作为真实决定处理。它们自身若含回执结构或证据定位错误，须先修正记录才能被接纳；本改动没有把具体缺陷、阻塞条件或真实语义争议改写成格式诊断，也没有自动保留或生成 approve。

## 为什么可复用现有恢复

`_receive` / `accept_result` / `_recover_call` 原有边界不变：

- 当前 attempt 与固定输入仍须匹配；原生成功、退出、终止结清、schema/nonce 和精确角色会话继续核验。新 attempt 使用新 nonce，不能接纳旧回执。
- worker 的原 workspace 保留；只有经原规范化、文件/字段权限和来源检查的合法草稿进入可恢复 checkpoint。reviewer 仍只读。下一次候选完整重新验收，worker 补正后仍须独立 reviewer。
- 同一阶段/节点原有 `receipt_repairs`、`invalid_receipts` 与 `calls` 持续累计，worker/reviewer 不各领新额度。原 `retry_limit=2` 最多允许两次自动追加重交；连续第三个错误记满失败次数并停为 `receipt_repair_limit_reached`。每次调用仍受原 calls 余额及审阅/裁决/签收预留限制。
- 异常候选没有 `validated.json`，不进入 `accepted_receipts`，不能成为阶段批准。修复失败或安全检查失败仍拒收，不回退共享计数。
- 现有 maintenance 会在修复记录后保持阻塞，仍需显式 resume；adjudicate/signoff 不获得新的重试循环。

## 验证

共 **30 项定向离线测试通过**，分两次执行（27 项 / 3 项）；`git diff --check` 通过。测试仅使用临时目录、确定性 backend 和显式 fake CLI，测试 PATH 中的真实 `claude` / `codex` 入口已阻断。

第一组覆盖：

```text
schema_extract.tools.tests.test_receipt_repair_policy
schema_extract.tools.tests.test_proposals
schema_extract.tools.tests.test_stage_flow.StageFlowTests.test_native_source_range_repairs_are_finite_and_resume_exactly
schema_extract.tools.tests.test_maintenance_receipts
```

新增集成测试验证 worker 漏项后的合法草稿、新 nonce 与精确会话、补正后再独立审阅、reviewer 漏结论后重交、共享修复计数、旧回执不被接纳，以及真实 revise/blocked 不进入 repair。现有 native fake CLI 用例改为连续三次越界，验证两次自动修复与第三次停止，仍保留会话隔离和结算检查。维护用例的期望计数相应增加，恢复权限未变。

第二组复核原硬边界：

```text
schema_extract.tools.tests.test_stage_flow.StageFlowTests.test_completed_label_does_not_replace_native_proof
schema_extract.tools.tests.test_stage_flow.StageFlowTests.test_extra_yaml_is_unread_in_normal_failed_and_interrupted_adoption
schema_extract.tools.tests.test_stage_flow.StageFlowTests.test_native_completion_cannot_adopt_another_scopes_session
```

没有运行完整离线套件或真实模型，因此这项结果只证明上述恢复与验收边界；不能据此宣称模型已补齐当前六算子的回执或本批已经通过。未提交、推送或启动整批运行。
