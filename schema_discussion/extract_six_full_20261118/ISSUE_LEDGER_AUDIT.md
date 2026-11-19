# 本批 issue 台账覆盖核对

2026-11-19。按 [台账规则](../../schema_extract/issues/README.md) 对照本批 [当前索引及原时间线](ISSUES.md)、索引所链诊断/修复/分流报告和全部正式 JSON 的 ID、问题、提案与决定查重；本次只整理记录，未重新核验 live 状态。

- **补齐四项遗漏**：新增 `maintenance-empty-poll-control-lock`、`monitor-registration-lock-retry`、`monitor-wait-stale-observation`；scoped 反馈遗漏沿原 `failed-node-recovery` 补正证据、影响及 2.5.1 实质修复决定，保留 2.3.0 历史依据。
- **按根因去重**：旧 `reminder-ack-delay` 的认证事件与未 ack 补送不同根，保留原决定；同类回执坐标/必填义务漏项统一归 `receipt-evidence-repair`，正常短交接 repair 不新增 issue。原 I/O 诊断、长字段 warning、维护 Version 指引已有正式记录。
- **实例与提案分开**：两项 `declined` 已补明拒绝的是规则扩展或合并主张；MLA tuple 与 Sparse/MoE/FMHA 目的地登记的实例维护均有单独完成依据，不能据 declined 忽略真实合规错误。非 Tensor 示例与存储层级两项仍 pending。
- **证据边界**：新活 scan owner 被抢占及旧 ack 时间为隔离复现，未记为线上事故；scoped 缺陷在本批实际限制局部恢复，但未执行的丢反馈路径不写成已发生。当前工作树修复不等于 frozen b8 或现有 daemon 已升级；持续等锁仍无时间上界。
- **不新增误分类项**：正常 P2 退修、已登记维护意图的 delivery busy、来源限制及历史拒收导致的 audit partial 直接引用原报告；不当作新工具 bug，也不以 errors 为空声称全证据通过。

已知本批发现均有索引、正式 issue 或明确的非工具问题报告去向，未发现仍仅在诊断报告而没有去向的实质项；这不是未知缺陷已排除或整批交付认证。39 份 JSON 的字段/唯一 ID/决定、两份索引文档的 43 个本地链接以及 8 个修改文件的空白与 JSON 格式核对通过，限定路径的 `git diff --check` 通过。未运行模型、提取或测试套件，未改源码、live、frozen、产物或批准。
