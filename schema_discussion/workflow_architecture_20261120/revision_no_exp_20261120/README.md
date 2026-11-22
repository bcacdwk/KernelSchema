# 双仓范围与可运行目标：计划修订

状态：**两位独立 xhigh 审查及另一位 xhigh 正式修改已完成**。D1–D3 保持；本轮将此前 17 个交付单元收敛为 K1–K8、A1–A7 共 15 个，保留 34 个子 gate，基准派工由 12 批收敛为 11 批。技术单元全部未开始，实现派工暂停。

当前完整目标有两条：AKA 提供 Schema on/off switch，开启后按阶段调用 planner 并执行合法后续行动，关闭保持原流程；KernelSchema 自身通过有限单 owner driver 完成 design 或 extract 消费→implement→至多一次 optimize→后续固定行动或 keep 停止。AKA fixed 是独立里程碑，不能代替 A6 动态目标。

- [范围审查](r1_scope_review.md)：已完成，提出职责、删项和依赖修改建议。
- [版本消费审查](r1_consumption_review.md)：已完成，仅只读核实来源、可选子模块及实际交付边界。
- [实际变更与检查](r2_change_report.md)：正式修改落点、采纳/调整、删并、计数与验证局限。
- [当前实施计划](../04_implementation_plan.md)、[交付单元](../05_work_packages.md)、[合同](../planning/r5_contract_execution_review.md)、[验收规范](../planning/r5_validation_delivery_plan.md)：唯一现行正文；范围外消费只见[主计划的简明边界](../04_implementation_plan.md#external-consumers)。
- `before/`、`baseline.json`：本轮修订前六份正文/导航/历史的精确快照与摘要，保持原样；历史报告的旧要求不构成当前交付门。

此前已建立 AKA 本地与远端 `feat/schema-workflows-v1`；KS 隔离工作树创建被用户中断，尚未确认完成，见[历史 H023/H024](../01_history.md)。本轮仅修订文档，没有改产品实现、活动提取、Git 或隔离工作树，没有运行模型/GPU、安装、源码提交或发布。
