# 提取方法与执行器对齐

当前 Markdown 定义少量具体完整设计、规范的 `dispatch.md` 与整区 `from`。现有 Python Runner、checker、starter、计划和回执尚未完成对接，**不能用当前工作树的 CLI 端到端运行这套新方法**。本页是本轮实施状态的唯一入口；更新方法文档不等于更新执行能力。

## 方法规范

文件与记录规则以 [common](../schema_docs/0.common.md#派生节点与-dispatch) 为准，提取职责见 [P1–P6 总览](../EXTRACT_SCHEMA.md)与 [REVIEW](REVIEW.md)。目标是含 root 至多十份各有独特价值的具体完整设计，其他参数、局部变化和已知组合在目录中记录；内容共享按实际整区来源判断。

## 手工推演与阶段结论

在本次授权范围内，源码阅读、数学推导、调度片段和候选评议可以按现行文档继续，不以启动 Runner 或通过旧 checker 为前提。记录实际使用的材料、已核实关系和未完成部分；片段按其说明的关系核对，候选按选择依据核对。

阶段准入、完整性和独立审阅要求用于正式放行。缺少前置阶段批准或仅核对片段时，结论止于实际覆盖对象，不记为阶段完成。正式运行与检查须使用匹配本次规则的执行器；尚无匹配工具时如实保留检查未完成，不为通过旧工具改写为旧语法。手工工作不改变阶段写权限、Worker/Reviewer 分工或冻结 run 的状态与批准。

## 当前执行器

以下差异已通过只读代码核实。本轮没有修改这些实现，也没有迁移现有算子或重新批准旧示例。

| 范围 | 当前实现事实 | 待对接的方法职责 |
| --- | --- | --- |
| 解析、检查与创建 | [resolver](../schema_template/tools/design_resolution.py) 和[字段检查](../schema_template/tools/rules/structure.py)仍使用 `parent/from_parent`，引用限定为直接父同名区；[starter](../schema_template/starter.yaml)还含 `phase`、`pipe.steady` 等旧结构。 | `from` 的实际内容引用图、使用上下文绑定、现行字段和 starter；修改引用名称不足以完成对接。 |
| P1 执行单元登记 | [artifacts.py](tools/artifacts.py) 的 `validate_changes` 仍要求整个硬件区与冻结 starter 相同，`_without_p1_fields` 也保留硬件区作受保护内容。 | P1 在现有 `execution` 中最小登记有据执行单元，其余硬件保持占位；当前门禁尚不接受这项流程权限。 |
| 目录与计划 | [artifacts.py](tools/artifacts.py)的 `validate_plan/index_prefix/validate_index` 要求七字段 plan、唯一根单父树和确定性的父表及覆盖缺口节。 | 有限完整集合、common 的记录格式、拟议内容与实际正文的区分。 |
| 调用、权限与证据 | [stages.py](tools/stages.py)按直接父批准解锁 P5；[artifacts.py](tools/artifacts.py)限制旧阶段写范围；[receipts.py](tools/receipts.py)按旧结论与依赖收集证据。 | P2/P3/P4 的批准对象与写权限、P5 实际来源依赖、普通记录的集中收尾和按影响返回。 |
| 修订、视图与交付 | 当前合入和复核沿旧父关系与代际推进；视图、回执、恢复和发布依赖登记的规则及执行能力。实现入口见[提取工具协议](tools/README.md)与[模板工具说明](../schema_template/tools/README.md)。 | 按实际定义和使用关系判断影响，保持 raw、完整来源身份、独立审阅、预算和最终文件集合一致。 |

## 冻结运行

旧 run 继续按其登记的 `source/config`、`rules` 和 `executor` 解释，身份含义见 [Supervisor 手册](README.md#输入与输出)。工作树的新文字不会自动改写旧状态、稿件、批准、回执或预算，也不能作为恢复旧 run 时替换冻结阶段页的依据。

现有命令及冻结协议仍由[操作手册](README.md)和[工具协议](tools/README.md)说明。语法、依赖或批准含义的改变需要实际工具支持和独立适用性验证；不能仅因现有 CLI 提供修订或升级入口，就把本次文字变更当作兼容运行层升级。

## 后续对接

后续代码工作须分别完成并验证：

1. 对齐 `from` 解析、整体引用图、使用上下文、raw/effective 来源身份、诊断、独立视图及 starter，并明确旧语法的冻结解释边界。
2. 解除七字段计划和直接父表的耦合，支持有限完整集合与正式记录；实现数量、文件和链接等可检查项。独特设计价值、局部差异是否闭合仍由人工审阅。
3. 对接阶段批准对象和写权限，包括 P1 的执行单元最小登记、实际内容依赖、P5 新事实的目录收尾及定点返回，保持 Worker/Reviewer 分离、有效全文审阅和受限写入。
4. 按实际定义、使用者及共享 Contract 处理合入、修改影响与复核；保留 raw、历史批准和原预算。普通记录增补不要求全部重开，重大选择变化也不能降格为局部编辑。
5. 对齐冻结能力、旧 run 可恢复性、语义采用和发布回执，再用结构正反例、旧运行回归及独立语义案例验证。

本页不预设新的命令、JSON 字段、状态或自动迁移策略。上述对接完成前，现有 checker 通过不能证明新方法已可执行。
