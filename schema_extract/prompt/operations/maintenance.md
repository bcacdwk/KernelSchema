# 当前运行的限域维护

只执行 maintenance.step 及已授权 fields。worker 随获准内容修改按[版本规则](../../../schema_docs/0.common.md#单份-plan-与元数据)更新 `META_DATA.Version`，该更新属于本次写权限，Runner 仅校验版本。阶段页说明应审阅的内容，不扩大本轮写权限；维护不替代缺失的正常阶段或最终 P6，不消耗或重新授予领域裁决与签收。

review/verify 的 reviewer 返回 approve、revise 或 blocked，独立审阅本阶段及当前前提。review 以显式给出的 `maintenance.fields` 为退修上限（`fields` 为 `[]` 时仅做只读复核、不可退修，发现问题或无法在上限内定位时返回 blocked）；退修在上限内给出 `revision_scope`（填 `[]` 则沿用 `maintenance.fields`）并为各目标文件提供本次 `artifact_evidence`，approve/blocked 及 verify 的 `revision_scope` 填 `[]`。review 最多进入一次 patch，再由独立 verify 验收；verify 仍有问题不继续修补循环。

patch 的 worker 返回 complete、continue 或 blocked，原样回传授权 revision_scope。continue 沿用当前步骤及调用次数。越界或上游错误返回 blocked，说明文件、字段和恢复条件，交 Supervisor 处理。
