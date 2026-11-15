# 第 4 轮：提取流程独立验收

结论：**作为目标提取方法，当前文档满足预期，可以据此确定调查、写作、独立审阅和定点返回的下一步。未发现具有具体错误后果的必改问题，也不建议为本轮增加可选流程。** 这不表示当前 CLI 已能端到端运行新方法；现有执行器、检查器和 starter 仍待对接。

本次先独立阅读 `AGENTS.md`、`EXTRACT_SCHEMA.md`、Supervisor 手册、执行器对齐说明、六个阶段页及 `REVIEW.md`，形成初步判断后，才阅读本轮 `PLAN.md` 与 `04_workflow_audit.md`。随后按入口核对 common、模板与工具协议、验收来源记录，并只读抽查相关 Python 和一处 MoE 源码。没有阅读另一位第 4 轮验收者的草稿。

以下检查均判为“已有规则足够”，最小修正均为“无需修改”。

| 具体场景 | 当前规则如何给出可执行处理 | 依据 |
| --- | --- | --- |
| P1 已建立 root，但尚不知道应选哪套 Design。 | 只完成任务元数据、共享算法 Contract 和原样硬件占位；P1 审全部输出与外部验收材料，不批准 root 代表选择。P2 才处理具体来源与代表身份，P3 写真实 `when` 和完整 Design。不会因先创建 root 而提前锁定方案。 | [P1 准入、写权限与放行](../../../schema_extract/phases/01_contract.md)，第 3、13、27–35、45–52 行；[P2](../../../schema_extract/phases/02_dispatch_draft.md)，第 13、19–20 行；[P3](../../../schema_extract/phases/03_root.md)，第 15–17 行。 |
| host 有许多 selector、shape 阈值与配置公式；仍有代码未读。 | P2 先核实实际可进入的入口族，再追主计算、物化、输出归属、主要分配与协作。函数、文件和 launch 数不决定设计数；继续研究须有实际线索。影响选型的重要疑点须解决或明确阻塞，普通未读细节可以限定后停止，不要求预做全部 P3 或所有配对。 | [P2 操作与调查止点](../../../schema_extract/phases/02_dispatch_draft.md#调查止点)，第 15–28 行；[common 的差异判断](../../../schema_docs/0.common.md#参数变化与结构变化)。 |
| 已发现 12 个有价值的独立机制，但本次最多完整交付 10 份。 | 发现集合不受十份限制；按任务价值选择或替换拟完整代表，每份说明独特价值。未选中的已研究独立方案保留名称、机制、范围或出处及取舍理由，不假装参数、不藏第十一份完整 schema。只有用户明确要求逐套完整交付的独立方案超过十份才产生目标冲突。P6 反查机制去向，不要求全部源码路径都有 YAML。 | [P2](../../../schema_extract/phases/02_dispatch_draft.md)，第 18–22、36 行；[P4](../../../schema_extract/phases/04_dispatch_plan.md)，第 15–19 行；[P6](../../../schema_extract/phases/06_final.md)，第 15–19 行；[common 的完整集合](../../../schema_docs/0.common.md#文件与共享-contract)。 |
| P4 只有真实 root，拟议非 root 尚无文件、对象名或已证明的共享区。 | 已存在基线的记录可以正式落实；未来文件用普通文本并标“仅计划”，对应条目也只是记录计划。未来对象不得冒充已有定义，拟共享不得冒充同一性证明。只有命名尚待落实可交作者；值、生产者或完成条件仍靠猜测则继续研究或阻塞相关任务。P6 必须绑定实际正文并清理临时计划。 | [P4](../../../schema_extract/phases/04_dispatch_plan.md)，第 13–19、27–29 行；[P6](../../../schema_extract/phases/06_final.md)，第 18 行。最终格式仅链接 [common](../../../schema_docs/0.common.md#选择条件与索引)，未重新定义七字段计划或直接父表。 |
| P5 发现一条普通局部条件，或需要将自身某区完整本地化、改用已存在已审的相同整区。 | P5 只改自身 YAML。其他事实通过既有交接/问题保存事实、来源、基线与对象、建议落点和影响；没有目录权限不成为丢失理由。仅影响本节点的共享调整由当前 P5 审完整适用性及引用图，普通目录增补由 P6 reviewer 核对后交按需 worker 集中落实。不会为一次共享写法调整自动回 P4。 | [P4 的共享意向](../../../schema_extract/phases/04_dispatch_plan.md)，第 21 行；[P5 新事实与定点返回](../../../schema_extract/phases/05_branch.md#新事实与定点返回)，第 25–30、40–48 行；[P6](../../../schema_extract/phases/06_final.md)，第 23、29–31 行。 |
| P5 要增加未来 provider、替换代表，或发现共享定义和 Contract 错误。 | 改完整集合、名额、未来依赖或其他节点前提/安排先回 P4，依赖新前提的工作等待；动摇代表选择回 P2，Contract 回 P1，provider 的本地错误回实际定义所在 P3/P5。provider 正确而本节点不适用时调整本节点，不反改来源。重大改变不能拖到目录收尾，未受影响稿件保留。 | [P5 返回表](../../../schema_extract/phases/05_branch.md#新事实与定点返回)，第 31–36 行；[REVIEW 发现问题与批准](../../../schema_extract/REVIEW.md#发现问题与批准)，第 67–69 行。 |
| 新 P6 reviewer 尚不了解 P5 交接，需要补目录，但必须保持独立全文审阅。 | 先从当前 raw、实际引用及目录独立理解全部有效内容，再用过程记录反查去向；历史交接不能补足正文。确需修改才启用新的 worker，按明确文件/字段修稿后再独立审最终全文。共用定义可以按实际文件读一次，每个使用上下文仍全覆盖。没有给每条笔记另建完整审批，也没有把终审降为目录检查。单 root 仍终审。 | [P6](../../../schema_extract/phases/06_final.md)，第 3、7–23、29–37 行；[REVIEW 当前产物先行](../../../schema_extract/REVIEW.md#当前产物先行)，第 9–15 行及第 35–45 行。 |
| 验收器只提供部分判据、源码有支持限制、无 GPU，或硬件资料未填。 | P1 必须读取已提供材料并逐输出保留已知判据；只有外部验收定义缺失且不影响当前提取时才可保留来源记录继续。计算、数值或依赖需要猜测仍阻塞具体范围。硬件占位和 `--skip-hardware`、不要求 GPU 验收均明确限定于固定实现提取；实现不足不缩小 Contract。普通实现仍核对硬件、全部输出与任务要求的运行证据。 | [P1](../../../schema_extract/phases/01_contract.md)，第 27–41 行；[REVIEW 来源缺项](../../../schema_extract/REVIEW.md#来源缺项)；[总览边界](../../../EXTRACT_SCHEMA.md#来源缺项与验收边界)；[算法验收来源记录](../../../schema_docs/1.contract.algorithm_requirement.md#固定实现提取中的验收记录)；[普通实现的验证职责](../../../IMPLEMENT_SCHEMA.md#3-连通调用并迭代验证)。 |
| 读者沿新方法寻找运行命令，或准备恢复旧 run。 | 方法由总览/阶段页定义，审阅方法由 REVIEW 定义，正式目录格式由 common 定义，实施现状在对齐页，真实命令与冻结协议留手册/tools。对齐页首段明确新方法不可直接用当前 CLI 端到端执行；旧 run 按原 source/config、rules、executor 解释，工作树文字不改旧稿、批准、状态或预算，也不构成兼容运行层升级。 | [执行器对齐说明](../../../schema_extract/EXECUTOR_ALIGNMENT.md)，第 3、15–24、28–36 行；[手册的冻结身份](../../../schema_extract/README.md#输入与输出)；[工具协议入口](../../../schema_extract/tools/README.md)，第 3–5 行。 |

实际 MoE 片段提供了两个有用的触发点。[`bf16_fused_moe/candidate/dispatch.cuh`](../../../kernel_wiki/aka-cuda-operators-20261030/operators/moe/bf16_fused_moe/candidate/dispatch.cuh) 第 810–843 行按 token/expert 条件选择 `block_m` 并联动 `store_block_m`，第 846–911 行根据存储预算选择 `num_stages`。按当前 P2 和 common，不能仅因这些是输入公式就将所有配置留进一份 YAML，也不能按每个取值或配对直接建节点；应在核实主机制后选定实际实例，保存有据的参数与联动记录。该入口的 [`launch.cu`](../../../kernel_wiki/aka-cuda-operators-20261030/operators/moe/bf16_fused_moe/candidate/launch.cu) 第 199–203 行有可选累计状态输出，第 243–249 行有 debug 模式下的异步清零。P1 的全部输出/副作用核对、P2 的数值与调用线索调查、P3 的跨调用状态与已提交动作要求，均有明确职责承接这些事实；不能仅凭主输出 shape 或 host 返回概括。这里只验证文档能否指导追查，没有完成该算子的机制归类或重新批准其 schema。

工具现状的只读抽查与说明一致：`artifacts.py:274–336` 仍校验七字段单父计划，`:339–392` 仍校验确定性索引，`:395–415` 仍限制 P6 字段；`stages.py:182–202` 仍按直接父批准解锁 P5；`receipts.py:53–69` 仍收集节点与直接父的来源证据；resolver/structure 仍识别 `parent/from_parent`。这些是后续实现对接范围，不能用它们未支持新方法否定本轮文字方案，也不能靠更改说明假称已实现。P1 comparisons、P4 确定性索引与 P6 `revision_scope/affected_nodes` 的旧编码已分别留在 [tools 的对应小节](../../../schema_extract/tools/README.md#p1-的-comparisons)，没有要求新阶段页承担第二份机器格式定义。

本次只读检查了总览、手册、对齐说明、REVIEW、extract tools 和六个阶段页中的 194 个本地文件/小节链接，未发现缺失路径或不匹配锚点。没有运行提取、checker、GPU、绘图或生产测试；只新增本报告，没有修改正式规则、代码、starter、旧 YAML 或运行状态。

尚未审的范围包括全部算子与全部源码路径的实际提取质量、所有旧实例的现行合规性、完整运行器/冻结迁移行为，以及 DOC 1–6 和普通作者入口的全面交叉验收。后续工具对接仍须单独实现与验证；本报告只确认本轮提取方法及相关说明的职责和边界可以成立。
