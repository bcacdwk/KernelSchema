# 六个典型算子的提取研究档案

本档案收集 GEMM、MoE、FMHA、MLA、稀疏注意力和 GDN 的试用、评估、恢复及暂停记录。流程说明仍在 [Schema Extract 手册](../../README.md)，正式结果和历史结果从 [Schema Wiki](../../../schema_wiki/README.md)进入。

## 最近一次收口

[2026-11-09 暂停交接](20261109-pause/README.md)是本批最近一次状态快照：四项已交付，FMHA 与 MoE 保留部分成果后暂停。节点批准数不等于整算子已经完成 P6 或发布。

| 算子 | 本批收口状态 | 逐节点记录 |
| --- | --- | --- |
| GDN | 2.2.7 六节点版已交付 | [6/6](20261109-pause/gdn-nodes.md) |
| MLA | 2.5.0 两节点版已交付 | [2/2](20261109-pause/mla-nodes.md) |
| 分组 GEMM | 2.5.0 七节点版已交付 | [7/7](20261109-pause/gemm-nodes.md) |
| 稀疏注意力 | 2.5.0 八节点版已交付 | [8/8](20261109-pause/sparse-nodes.md) |
| FMHA | 26/43 已批准；4 个待审，13 个需继续写作或返工 | [FMHA 清单](20261109-pause/fmha-nodes.md) |
| MoE | 96/154 已批准；9 个待审，1 个需继续写作，5 个失败未成稿，43 个未开始 | [MoE 清单](20261109-pause/moe-nodes.md) |

完整字段、checkpoint 和最近尝试见[逐节点 CSV](20261109-pause/nodes.csv)。这批运行保持暂停，恢复范围和预算须由后续用户指示另行确定。

[2026-11-09 消耗复盘](20261109-usage-review/README.md)按原始调用和会话去重，分列重构以来与实际采用 2.5.0 规则的 token、缓存、耗时和用户给定单价下的费用；逐调用数据及统计范围一并保留。

## 按研究阶段查阅

| 阶段 | 关键文件 | 记录内容 |
| --- | --- | --- |
| 2026-11-03 初次试用及后续复核 | [试用记录](20261103-trial/TRIAL_REPORT.md)、[质量初验](20261103-trial/OPERATOR_REVIEW.md) | 初始六项交付、证据审计、表达保留意见，以及早期修订 |
| 2026-11-04 Reviewer 机制验证 | [验证材料](20261104-review-cases/README.md)、[样例集](20261104-review-cases/suite.json)、[验证结果](20261104-review-cases/VALIDATION.md) | 由试用问题形成的合成基线、十个语义样例、reference 及真实 GDN 隔离终审记录 |
| 2026-11-05 独立恢复评估 | [处置评估](20261105-recovery/README.md)、[审计 A](20261105-recovery/audit_a.md)、[审计 B](20261105-recovery/audit_b.md) | 哪些成果保留、哪些局部修订、哪些需要重审；原始独立意见保留 |
| 2026-11-05 推进组织 | [推进方案](20261105-recovery/CONTINUATION_PLAN.md)、[控制建议](20261105-recovery/organization_control.md)、[内容建议](20261105-recovery/organization_content.md) | Supervisor、独立工作包和各算子 Runner 的分工与顺序 |
| 2026-11-05 至 11-09 执行 | [执行记录](20261105-recovery/EXECUTION_LOG.md) | 桥接、正式修订、复核、交付和各次巡检的历史记录 |
| 2026-11-09 暂停收尾 | [暂停交接](20261109-pause/README.md)、[节点清单](20261109-pause/nodes.csv) | 最后状态、保留稿、未完成项、未开始项及原运行证据定位 |

六算子试用暴露的问题与机制设计，另见 [issue 归档](../../issues/archive/README.md)、[连续修订设计](../designs/2.4.0-continuous-revision.md)和 [Design 整区引用设计](../designs/2.5.0-design-region-derivation.md)。结果侧的版本对应关系和旧索引勘误见 [wiki 历史索引](../../../schema_wiki/archive/six-operators/README.md)。

## 阅读这些记录时

记录按其发生时点保留：早期“待迁移”“仍运行”以及旧预算不应覆盖后来的暂停结论。执行流水账不作为新的派发指令；实际恢复需要核对相应 run 的当时状态和当前用户授权。

本目录收集仓库中已有的文档和清单，没有从 Git 忽略的运行目录导入新的部分 YAML。清单内的本机 checkpoint 链接用于进一步核对原稿；正式交付仍以对应目录的回执为准。
