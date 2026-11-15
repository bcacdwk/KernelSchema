# Dispatch 设计与提取方法

现行方案已按用户最新决定落实到正式文档：每算子含 root 最多十份具体完整设计，其他配置参数、局部变化和组合在 `dispatch.md` 规范记录。共同区域直接 `from`，YAML 不容纳备选配置选路。

当前规则以 [common](../../schema_docs/0.common.md#派生节点与-dispatch) 为准；提取方法见 [EXTRACT_SCHEMA](../../EXTRACT_SCHEMA.md)。研究材料不替代正式定义。

## 最新落地

- [落地结果与四轮验收](concrete_design/README.md)
- [详细实施方案](concrete_design/PLAN.md)
- [用户约束与本轮范围](concrete_design/BRIEF.md)
- [执行器对齐状态](../../schema_extract/EXECUTOR_ALIGNMENT.md)：本轮只改文字，代码、starter、旧实例与运行状态未迁移

common 负责所有作者共用的设计身份和记录规则；extract 的阶段提示负责源码识别与推进。六阶段保留，P2 先识别机制再选 root，P4 整理有限目录与变体，P5 只写选中代表，P6 独立审完整集合及正式记录。

## 前期研究材料

以下文件保留各轮当时的判断与证据。后续决定已在最新落地中统一，不把早期候选意见当作现行规则；源码与统计的范围仍按各报告说明解释。

| 材料 | 内容 |
| --- | --- |
| [初始问题定义](00_problem.md) | 最初的职责混合与判断假设 |
| [MoE 证据](01_moe_evidence.md) | 初始显式配置与后续默认范围、组合矩阵、源码差异和成本边界 |
| [十算子对照](02_ten_operator_evidence.md) | 可合并特例、局部创新及整体协作反例 |
| [设计身份研究](03_design_identity.md) | 完整方案、范围、root 与来源关系的初期候选 |
| [共享与记录](04_sharing_and_records.md) | 整区复用的实际机会与局部说明试写 |
| [反例审计](05_adversarial_review.md) | 数值、在途访问、来源保证与记录边界 |
| [目录试写](06_case_trials.md) | MoE、FA4 早期候选目录，未迁移完整 YAML |
| [主代理前期审阅](REVIEW_NOTES.md) | 关键数据复核与当时的取舍 |
| [前次文档审阅](docs_review/README.md) | 上一轮 common 落地与独立演练 |

MoE 初始 60 份与后续 154 项计划的实现选择范围不同，不能作为同范围压缩对照；当前规则也没有声称这批旧产物已经迁移或全部组合已验证。提取流程与代码接续的下一步，以最新实施方案和执行器对齐页为准。
