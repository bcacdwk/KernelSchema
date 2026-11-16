# 提取方法与执行器对齐

当前 Markdown 定义少量具体完整设计、规范的 `dispatch.md` 与整区 `from`。Template 的 checker、starter、只读 API 和统一作者 CLI 已对齐当前字段；Extract 的 Python Runner、计划、权限和回执仍待对接，**不能用当前 Extract CLI 端到端运行这套新方法**。本页记录两者的实际边界，Template 检查通过不等于提取执行器已经可用。

## 方法规范

文件与记录规则以 [common](../schema_docs/0.common.md#派生节点与-dispatch) 为准，提取职责见 [P1–P6 总览](../EXTRACT_SCHEMA.md)与 [REVIEW](REVIEW.md)。目标是含 root 至多十份各有独特价值的具体完整设计，其他参数、局部变化和已知组合在目录中记录；内容共享按实际整区来源判断。

## 手工推演与阶段结论

在本次授权范围内，源码阅读、数学推导、调度片段和候选评议可以按现行文档继续，不以启动 Runner 或通过旧 checker 为前提。记录实际使用的材料、已核实关系和未完成部分；片段按其说明的关系核对，候选按选择依据核对。

阶段准入、完整性和独立审阅要求用于正式放行。缺少前置阶段批准或仅核对片段时，结论止于实际覆盖对象，不记为阶段完成。正式运行与检查须使用匹配本次规则的执行器；尚无匹配工具时如实保留检查未完成，不为通过旧工具改写为旧语法。手工工作不改变阶段写权限、Worker/Reviewer 分工或冻结 run 的状态与批准。

## 当前执行器

Template 本轮已完成当前格式与工具更新。下列 Extract 差异仍未对接；本轮未修改 Runner、现有算子、冻结运行或批准状态。

| 范围 | 当前实现事实 | 待对接的方法职责 |
| --- | --- | --- |
| Template 解析、检查与创建 | [resolver](../schema_template/tools/design_resolution.py)、[字段检查](../schema_template/tools/rules/structure.py)和 [starter](../schema_template/starter.yaml)已采用当前格式；覆盖整区 `from` DAG、实际来源链、使用节点绑定、raw 文本预算和 dispatch 最小检查。默认作者 CLI 会统一规范化整个任务及其引用；规范化库可只返回内存候选。 | Extract 使用匹配的严格只读校验 API，或显式调用 `check_yaml.py FILE --read-only --skip-hardware` 预览并检查规范化候选；适配候选/实际来源的区别、诊断与来源身份。 |
| P1 执行单元登记 | [artifacts.py](tools/artifacts.py) 的 `validate_changes` 仍要求整个硬件区与冻结 starter 相同，`_without_p1_fields` 也保留硬件区作受保护内容。 | P1 在现有 `execution` 中最小登记有据执行单元，其余硬件保持占位；当前门禁尚不接受这项流程权限。 |
| 目录与计划 | [artifacts.py](tools/artifacts.py)的 `validate_plan/index_prefix/validate_index` 要求七字段 plan、唯一根单父树和确定性的父表及覆盖缺口节。 | 有限完整集合、common 的记录格式、拟议内容与实际正文的区分。 |
| 调用、权限与证据 | [stages.py](tools/stages.py)按直接父批准解锁 P5；[artifacts.py](tools/artifacts.py)限制旧阶段写范围；[receipts.py](tools/receipts.py)按旧结论与依赖收集证据。 | P2/P3/P4 的批准对象与写权限、P5 实际来源依赖、普通记录的集中收尾和按影响返回。 |
| 修订、视图与交付 | 当前合入和复核沿旧父关系与代际推进；视图、回执、恢复和发布依赖登记的规则及执行能力。实现入口见[提取工具协议](tools/README.md)与[模板工具说明](../schema_template/tools/README.md)。 | 按实际定义和使用关系判断影响，保持 raw、完整来源身份、独立审阅、预算和最终文件集合一致。 |

## 冻结运行

旧 run 继续按其登记的 `source/config`、`rules` 和 `executor` 解释，身份含义见 [Supervisor 手册](README.md#输入与输出)。工作树的新文字不会自动改写旧状态、稿件、批准、回执或预算，也不能作为恢复旧 run 时替换冻结阶段页的依据。

现有命令及冻结协议仍由[操作手册](README.md)和[工具协议](tools/README.md)说明。语法、依赖或批准含义的改变需要实际工具支持和独立适用性验证；不能仅因现有 CLI 提供修订或升级入口，就把本次文字变更当作兼容运行层升级。

## 后续对接

后续代码工作须分别完成并验证：

1. 接入 Template 已提供的当前字段、`from`、raw/effective 来源身份、诊断和独立视图接口；将外部检查调用改为 `--read-only --skip-hardware`，避免默认排版越过阶段写权限。冻结运行仍使用其登记的规则与执行器。
2. 解除七字段计划和直接父表的耦合，支持有限完整集合与正式记录；实现数量、文件和链接等可检查项。独特设计价值、局部差异是否闭合仍由人工审阅。
3. 对接阶段批准对象和写权限，包括 P1 的执行单元最小登记、实际内容依赖、P5 新事实的目录收尾及定点返回，保持 Worker/Reviewer 分离、有效全文审阅和受限写入。
4. 按实际定义、使用者及共享 Contract 处理合入、修改影响与复核；保留 raw、历史批准和原预算。普通记录增补不要求全部重开，重大选择变化也不能降格为局部编辑。
5. 对齐冻结能力、旧 run 可恢复性、语义采用和发布回执，再用结构正反例、旧运行回归及独立语义案例验证。

本页记录已有 Template 接口及 Extract 待对接事实，不改变 Runner 命令、JSON 字段、状态或冻结规则。上述对接完成前，Template checker 通过不能证明提取方法已可端到端执行。
