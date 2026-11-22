# Schema 四流程与 AKA 接入：调研、决策与落地计划

状态：**四轮复审及正式修订已完成**，见[查漏、精简、裁决与修改记录](review_20261120/README.md)。D1–D3 保持；原 34 包收敛为 **17 个交付单元，全部未开始**。

本目录记录 2026-11-20 开始的长期规划讨论。本阶段完成了调研、决策、计划撰写、四轮复审与文档核对，没有修改 KernelSchema、AKA 或 AKA_experiment 的实现，没有运行 kernel 模型/GPU 实验或执行远端迁移。

## 阅读入口

- [本轮计划复审](review_20261120/README.md)：双 max 查漏 → high 精简 → xhigh 裁决 → 新 xhigh 修改均已完成，附修改前快照与逐项变更。
- **[当前实施计划](04_implementation_plan.md)**：方向、三仓职责、阶段交接与发布门。
- **[17 个交付单元](05_work_packages.md)**：唯一 owner/依赖表、改动区域、产物、验收归属和首批顺序，全部未开始。
- [开发代理派工](05_work_packages.md#agent-dispatch)：默认 xhigh、最多两位并发、12 个基准批次与里程碑独立审查；实际按前置和证据推进。
- **[公共合同](planning/r5_contract_execution_review.md)**：唯一 API、身份/选定闭包、源码、恢复/预算与结果规则。
- **[验收与交付规范](planning/r5_validation_delivery_plan.md)**：唯一 Task/案例参数、V01–V15、公开材料和有限 pilot。
- [用户需求与讨论边界](00_user_brief.md)
- [过程及决策历史](01_history.md)
- [调研开始时的三仓版本与工作区基线](baseline.json)
- `research/`：三轮、六位独立 max 子代理的报告（全部完成，见下表）。
- [初步架构与推进大纲](02_initial_outline.md)：由新的 xhigh 子代理整合，主代理审阅。
- [三个方向决策](03_decision_questions.md)：用户已明确回复“按推荐开始”；保留原推荐与备选作为历史。
- `planning/`：R4 保留历史计划稿；两个 R5 路径已就地收敛为现行合同/验收规范，原稿保存在本轮 before 快照中。

当前执行依据为上述四份分工明确的正文，不再靠主计划覆盖附录冲突。初纲、原始问题及研究报告保留为历史，不构成待确认问卷或当前规则。实施前记录基线并隔离；首批为 K1.materials 首题材料、K7.materials 真实题预审、A1.cpu 接缝验证。本阶段未开始执行这些单元。

## 分阶段流程

1. 第一轮：核实 KernelSchema 当前能力；核实上游 AKA 和 AKA_experiment 的真实结构。
2. 第二轮：基于证据设计独立四流程的边界与契约；研究 AKA 接入、轮次与状态管理。
3. 第三轮：从维护、发布、实验验收和反例角度交叉审查方案，提出分期建议。
4. 新的 xhigh 子代理整合初步大纲；主代理汇报必要决策。已完成，用户确认全部按推荐推进。
5. 用户决策后：两轮新的 max 子代理撰写落地计划，再由新的 xhigh 子代理整合完整计划。已完成：第一轮为 KS 独立能力、AKA 接入与三仓迁移；第二轮修订契约/依赖并补齐验收/交付/实验；最后整合为主计划与工作包清单。
6. 用户追加复审：双 max 查漏、单 high 精简、单 xhigh 独立裁决、另一位 xhigh 正式修改。已完成，详见本轮记录；复审交付时四份有效正文由 1127 行收敛到 688 行。此后按用户要求补充开发派工，合同/验收与17单元的技术边界保持。

每轮最多同时运行两个子代理。所有报告必须区分「已核实事实」「设计建议」「未验证假设」，保留必要的仓库路径及行号证据。讨论中的建议不自动成为用户决策。

| 轮次 | 分工与报告 | 状态 |
| --- | --- | --- |
| R1 | [KernelSchema 现状](research/r1_schema_inventory.md)；[AKA / 实验仓现状](research/r1_aka_inventory.md) | 完成，含代码证据与验证边界 |
| R2 | [独立四流程与契约](research/r2_standalone_contracts.md)；[AKA 接入与生命周期](research/r2_aka_lifecycle.md) | 完成，形成候选及取舍 |
| R3 | [维护、公开使用与实验验收](research/r3_delivery_evaluation.md)；[最小性与反例审查](research/r3_architecture_challenge.md) | 完成，修正候选前提 |
| 初步整合 | [xhigh 初步大纲](02_initial_outline.md)，[主代理决策页](03_decision_questions.md) | 完成，D1–D3 已确认 |
| R4 | [KS 历史计划稿](planning/r4_kernelschema_plan.md)；[AKA/迁移历史计划稿](planning/r4_aka_integration_plan.md) | 完成，仅作历史依据 |
| R5 | [原合同审查稿](review_20261120/before/planning__r5_contract_execution_review.md.txt)；[原验收计划稿](review_20261120/before/planning__r5_validation_delivery_plan.md.txt) | 完成，原稿已留快照 |
| 原最终整合 | [修订前主计划](review_20261120/before/04_implementation_plan.md.txt)、[修订前34包](review_20261120/before/05_work_packages.md.txt) | 完成，随后按用户要求复审 |
| 四轮复审与修订 | [独立裁决](review_20261120/r3_adjudication.md)、[实际变更](review_20261120/r4_change_report.md) | 完成，现行四正文/17单元；实现未启动 |

## 工作区保护

KernelSchema 开始时已有 AGENTS.md 修改、部分 wiki 文件删除及新增提取结果；这些属于既有工作，本讨论不改动。AKA 初始 `main` 工作区干净。AKA_experiment 位于 `schema-ablation-261120`，有大量既有修改、删除和未跟踪内容，调查必须区分当前工作树与已提交版本。后续仅在本目录新增和更新规划材料。
