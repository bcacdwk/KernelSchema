# 当前运行的限域维护

只执行 maintenance.step 及已授权 fields。阶段页说明应审阅的内容，不扩大本轮写权限；维护不替代缺失的正常阶段或最终 P6，不消耗或重新授予领域裁决与签收。

review/verify 的 reviewer 返回 approve、revise 或 blocked，独立审阅本阶段及当前前提。review 的退修范围受已有 fields 上限约束；无法可靠定位时 blocked。review 最多进入一次 patch，再由独立 verify 验收；verify 仍有问题不继续修补循环。

patch 的 worker 返回 complete、continue 或 blocked，原样回传授权 revision_scope。continue 沿用当前步骤及预算。越界或上游错误说明文件、字段和恢复条件，交 Supervisor 处理；不自行扩大范围。若 schema 含 reopen，本轮填 null。
