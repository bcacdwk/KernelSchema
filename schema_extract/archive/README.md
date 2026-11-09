# Schema Extract 历史归档

这里保存已经完成的设计、试用、评估、接续与暂停记录。现行操作从 [Supervisor 手册](../README.md)进入，提取顺序见[流程总览](../../EXTRACT_SCHEMA.md)，审阅方法见 [REVIEW](../REVIEW.md)。

| 材料 | 归档入口 | 用途 |
| --- | --- | --- |
| 六个典型算子的研究过程 | [六算子档案](six-operators/README.md) | 串联初次交付、质量初验、独立恢复评估、执行记录和暂停清单 |
| 2.4.0 连续修订设计 | [保进度连续修订](designs/2.4.0-continuous-revision.md) | 保留设计边界、实施分工与当轮验证依据 |
| 2.5.0 Design 整区引用设计 | [整区派生引用](designs/2.5.0-design-region-derivation.md) | 保留 DRD 决定、接口及整合验证；现行字段语义见公共规范 |
| 已处理的问题与决定 | [issue 归档](../issues/archive/README.md) | 按稳定 ID 和类别追踪决定、保留事项及来源证据 |
| Reviewer 验证材料 | [六算子试用验证集](six-operators/20261104-review-cases/README.md) | 基线 YAML、样例 JSON、reference、来源说明和 2.2.5 验证记录整体归档 |

## 归置方式

- 顶层的 README、REVIEW、phases 和 tools 继续承担当前流程与工具说明。已完成的设计、实施计划和批次记录进入 archive。
- 同一批算子的多次恢复、暂停和评估放在同一个研究档案下，按阶段日期保留，并从档案 README 统一导航。
- 归档正文中的“当前”“下一步”、预算和运行命令对应记录时点。它们不构成新的执行授权，也不证明后来版本已经通过验证。
- 稳定 issue JSON 保留原 ID、决定和历史证据；归档整理不改写决定版本。六算子的初始及修订结果统一保存在 [Schema Wiki 六算子归档](../../schema_wiki/archive/six-operators/README.md)，文件内容与原回执保留。
- `.schema_extract/` 是本机运行数据，不纳入这次目录归置。历史文档中指向该目录的链接保留为补充证据入口。

## 顶层文件迁移

| 原位置 | 现位置 |
| --- | --- |
| `schema_extract/TRIAL_REPORT.md` | [六类算子试用记录](six-operators/20261103-trial/TRIAL_REPORT.md) |
| `schema_extract/OPERATOR_REVIEW.md` | [算子结果初验](six-operators/20261103-trial/OPERATOR_REVIEW.md) |
| `schema_extract/IMPROVEMENT_PLAN.md` | [2.4.0 连续修订设计](designs/2.4.0-continuous-revision.md) |
| `schema_extract/DESIGN_REGION_DERIVATION.md` | [2.5.0 整区引用设计](designs/2.5.0-design-region-derivation.md) |
| `schema_extract/recovery_reviews/20261105/` | [2026-11-05 恢复评估与执行](six-operators/20261105-recovery/README.md) |
| `schema_extract/recovery_reviews/20261109-pause/` | [2026-11-09 暂停清单](six-operators/20261109-pause/README.md) |
| `schema_extract/review_cases/` | [Reviewer 验证材料整包](six-operators/20261104-review-cases/README.md) |
| `schema_extract/review_cases/RESULTS.md` | [整包内的历史验证记录](six-operators/20261104-review-cases/VALIDATION.md) |

历史 JSON、Git 版本定位和本机运行证据可能仍记载原文件名；核对历史行号时使用对应版本，查阅现有文档时使用本表。
