# 派生节点的 Design 整区引用：问题记录

稳定 ID：[`design-region-derivation`](archive/extract_workflow/design-region-derivation.json)。状态：**2.5.0 已解决（resolved）**。记录日期：2026-11-05。下列调查原文、规范、代码和已提交示例的核对基线为 `KernelSchema@26a7947c8257401c95582a8a43676c5651c9a2ef`（2.4.1）；文中行号及“现行”“尚未”“待决定”均指该基线。链接用于导航，核对历史事实须读取该 Git 版本的对应路径，不能把当前实现当作旧行号的依据。

最终决定和实现见[实施设计](../DESIGN_REGION_DERIVATION.md)：直接父同名整区引用、子上下文解析、公共来源身份、审阅分工、换父和 provider 修复已随 2.5.0 贯通。以下保留调查事实与约束，不表示这些缺口在 2.5.0 仍未处理；真实历史产物未因框架修复自动获得新批准，`conditional-flow-topology` 按用户决定归档保留，产物研究另行开展。

用户提出的候选是：派生节点可以显式引用父级的完整 `math_reshape`、`work_schedule`、`tile_pipeline` 区域，让相同定义像共享 root Contract 一样只维护一份，同时保留真实不同的 dispatch、稳定节点命名和必要的独立审阅。这是待设计的语言与工作流变化，现行规则尚不允许；本文记录问题，不决定引用语法、解析接口或最终实施方案。

下文 `provider` 指某个被引用区域的实际定义节点；“原始视图”指作者保存的 YAML，“有效视图”指解析引用后、在使用节点上下文中理解的完整 Design。`DRD-01` 至 `DRD-10` 是同一聚合问题的稳定子编号，供后续方案引用，不另拆十份台账或验收清单。

## DRD-01：共享 Contract 不是现成的 Design 内容继承机制

**已证事实。** [0.common.md:104、121](../../schema_docs/0.common.md#派生节点与-dispatch) 明确 `parent` 只记设计来源，各节点写全 Design；[rules/__init__.py:31](../../schema_template/tools/rules/__init__.py) 从 root 取得统一 Contract 符号，再逐节点检查关系，没有 Design 内容展开器。区域之间也不独立：[math_reshape:164](../../schema_docs/3.design.math_reshape.md#引用与缺项) 允许数学符号 `bind` 到本节点的工作调度参数；[work_schedule:327](../../schema_docs/4.design.work_schedule.md#跨-phase-数据交接) 的 handoff 可以引用 pipeline 的资源和里程碑；[tile_pipeline:443](../../schema_docs/5.design.tile_pipeline.md#跨-phase-交接) 的 worker、资源和 role 身份又取决于 work.flow 中的提交及并发关系。

**待决定。** 内容引用如何与现有对象/属性 `ref:`、下游 `bind:` 区分；引入名称如何在使用节点中解析；什么关系由检查器验证、什么适用性仍由审阅承担。后续方案必须明确引入区域在子节点的完整上下文中绑定，并检查与另外两区的组合，不能把“区域原文相同”当作“在子节点中必然成立”。

**约束。** 子节点的 `when` 仍在 root Contract 全域独立求值，父 `when` 不是子节点前提。显式共享不能把父节点的选择域或参数取值一并隐式带入。

## DRD-02：引用粒度、来源关系与 Realization 边界尚未定义

**已证事实。** [fa4_decode/q1_cluster.yaml:8、31、124](../../schema_template/examples/fa4_decode/q1_cluster.yaml) 与 [q4_cluster.yaml:8、30、123](../../schema_template/examples/fa4_decode/q4_cluster.yaml) 的整个 math_reshape 经现有 YAML 解析器比较相同，但二者分别派生自 `q1_global` 和 `root`，并非父子；工作调度分别给出 `QueryTile=1`、`QueryTile=2`。相同公式可以绑定不同参数，能否复用还取决于允许的来源关系。

**待决定。** 首版是否只允许引用直接父的同名完整区域，是否允许指定祖先，以及直接父自身引用时如何传递解析；引用与本地完整区域是否互斥、root 是否必须全部本地定义、是否禁止缺省继承与局部 patch，都需要明确规则。若允许三区域选择不在同一祖先链上的 provider，内容依赖会成为不同于 dispatch 树的 DAG，现有树调度不能自然覆盖。

**约束。** [realization:7、57](../../schema_docs/6.realization.md) 是本节点面向上游对象的实现条件与适用反馈。三区域引用不能自动继承其 `require/prefer/forbid`、实验反馈或证据；本节点仍需判断这些约束是否适用。是否采用整区引用作为首版边界待方案确认，phase/role patch、兄弟共享和自动改选父节点不应被候选顺带引入。

## DRD-03：原始与有效视图、诊断与派生产物必须一致

**已证事实。** [structure.py:211](../../schema_template/tools/rules/structure.py) 要求三区域为完整映射；[schema.py:38](../../schema_template/tools/schema.py) 直接将读取结果交给结构和关系检查；[pipeline_graph.py:40、80](../../schema_template/tools/pipeline_graph.py) 直接消费该 Design，生成图时绑定的源码摘要只有本节点。现有[诊断 formatter:53](../../schema_template/tools/diagnostics.py) 依赖节点字段路径到原始文档位置的对应。[standalone.py:25](../tools/standalone.py) 只补 root Contract，来源回执和生成前后防漂移检查只覆盖 root 与指定节点。

**待决定。** 哪个共用入口返回原始/有效内容及来源信息，各消费方如何使用，结构错误与跨区组合错误分别怎样定位。传递引用加入后，standalone 需要完整展开 Design，图与阅读视图的来源身份、防漂移检查必须覆盖全部实际依赖；不能只因子 YAML 字节未变而复用旧结果。具体指纹或接口格式留给方案。

**约束。** 不能原地展开并覆盖作者稿件，也不能让检查、绘图、standalone 各自实现一套含义不同的展开。报错应能定位 provider 的实际定义及使用节点的引用/组合上下文，不能把继承字段误报为子文件中存在的行。任务目录中的全部直属 YAML 都被当作节点读取，最终文件集合还须与 plan 相同（[schema.py:17](../../schema_template/tools/schema.py)、[artifacts.py:208](../tools/artifacts.py)）；不能加入未计划的辅助 YAML 或在 worker workspace 中新增临时产物。

## DRD-04：P5 首稿、保稿与依赖失效不能混为一件事

**已证事实。** [Store.start_node:617](../tools/store.py) 在节点首次无稿时复制父整份文件，去掉 Contract 并设置身份；已有草稿、重试或曾批准内容则优先保留自己的稿件，只刷新只读祖先。私有上下文已带全祖先，[Store.verify_node:458](../tools/store.py) 校验全祖先文件摘要，[stages.affected_nodes:168](../tools/stages.py) 按树传播后代重审；兄弟不是该节点的依赖。checkpoint、调用依赖与串行合入也已有身份和越权校验。

**待决定。** 新节点的初始引用如何生成，worker 何时选择本地整区，父内容变化后怎样表示有效 Design 已变。原始 YAML 的摘要/版本，与展开后依赖组成的有效 Design 身份不是同一件事；两者的用途和变更规则需要写清，不能用某一方替代另一方。

**约束。** 沿 parent 链的方案可以先沿用现有全文件祖先摘要和保守后代失效，不必为减少重审立即细化到区域缓存或引入新 state 格式。新能力不应覆盖已有节点正文、改写不可变 checkpoint，或放宽 [artifacts.py:352](../tools/artifacts.py) 的文件/字段权限和 [Store.merged_workspace:665](../tools/store.py) 的只合入本节点限制。

## DRD-05：共享定义的审阅与子节点适用性审阅需要明确分工

**已证事实。** [required_artifact_evidence:52](../tools/receipts.py) 对 P5 当前只要求 root、本节点、直接父的文件证据；若直接父又引用非 root 祖先，实际 provider 可能漏出必要集合。全祖先虽然已在私有目录，不能因此认为必要证据已覆盖。[prompts.py:127](../tools/prompts.py) 会隐藏历史批准、交接与旧稿；[REVIEW.md:3、11、34](../REVIEW.md)、[P5](../phases/05_branch.md)、[P6](../phases/06_final.md) 均要求按当前完整产物独立审阅，现行 P5 还把父节点仅作为复制来源对照。

**待决定。** 对明确共享的定义本体，provider 与使用节点的 Reviewer 分别承担什么责任；子节点需要怎样呈现当前有效 Design、实际定义来源和必要证据。子节点必须覆盖选择域、当前参数绑定、三区域组合、输入到输出与 realization 的适用性，父批准不能直接替代这些判断。如何减少无意义重复，又保留完整理解，需要在 REVIEW、P5、P6 与必要文件证据中统一说明。

**约束。** 可利用已有 [`inherited_design` 结论主题](../tools/stages.py) 表达共用内容适用性；不要先另建审阅 IR 或一套重复证据账本。共享当前定义与依赖历史批准是两回事，不能恢复隐藏历史结论作为子节点通过理由。

## DRD-06：换父会改变引用语义，保留旧稿时尤其容易漂移

**已证事实。** [install_plan:282](../tools/stages.py) 更新 plan/state 中的父关系并保存旧稿、旧批准线索；[start_node:617](../tools/store.py) 随后仍保留子 YAML 原文。[P4](../phases/04_dispatch_plan.md) 只写索引和结构化 plan，[artifacts.py:401](../tools/artifacts.py) 又要求 P5 文件中的 `id/parent` 与当前计划一致，并由 [_versions:164](../tools/artifacts.py) 管内容修订。因此改 plan 并不会自动把旧稿内的 parent 或未来引用改好。

**待决定。** 换父后由谁、在哪个合法写入范围修改原始 parent/ref；旧 provider 是否保留、何时重新定向；原始版本、有效身份、重审和后代失效如何配合。若引用写“当前直接父”，单改计划就可能在子文件原文未变时改变有效 Design；若引用显式写 provider，旧目标又可能与新 parent 不一致。两类写法都需要完整处理，不能由解析器猜测作者意图。

**约束。** 原草稿和历史身份保留，不能以自动 retarget 掩盖设计变化，也不能借换父重置已有版本、调用计数或审批。不能为了机械去重自动重排派生树。

## DRD-07：缺少回到指定 P5 provider 修复的普通入口

**已证事实。** [receipts.REOPEN_TARGETS:42](../tools/receipts.py) 及普通 reopen schema 只允许回到 P1–P4 或 external；[runner._begin_reopen:419](../tools/runner.py) 按上游阶段批准处理。P5 无权改祖先，P4 也无权修派生 Design。子节点发现某个非 root 祖先定义错误时，现有常规通道不能直接指定该 P5 provider；引用使这个既有修复缺口更直接地影响依赖节点。

**待决定。** 如何把具体问题、当前证据和修复范围交回指定 provider，再独立重审并恢复受影响后代；如何与并行调用、有效批准和原稿保留衔接。root 问题仍须区分 Contract、root Design 与计划各自的合法入口。

**约束。** 不能靠改动无关 plan 文案触发失效来模拟 provider 修复，不能赋予子节点跨文件写权限。已有语义维护机制可以作为设计依据，但不能把每次普通内容修复都伪装成规则升级。

## DRD-08：派生身份、计划差异与选择条件保持各自含义

**已证事实。** [P4](../phases/04_dispatch_plan.md) 的 plan 是 `id/file/parent/workload/difference/selection/evidence` 七字段，索引是其固定投影；[validate_plan:208](../tools/artifacts.py) 核对唯一身份、父关系和文件集合。[0.common.md:119、135](../../schema_docs/0.common.md#选择条件与索引) 要求同一路径沿用 ID，所有节点（含 root 和中间节点）的 `when` 独立、互斥、共同覆盖契约全域。当前 `when` 仍为文本，引用机制不会把它变成条件求解器。

**待决定。** 区域引用仅作为 P5 的内容表达，还是需要 P4 提前承诺其来源；若后者需要额外计划信息，应说明必要性，不能不经决定便扩展七字段。`difference` 应描述有效设计相对父节点的差异，而非仅记录“改用引用”的语法事实。稳定命名应概括实际 workload/设计，不随复制、展开或参数取值变化。

**约束。** 真实提交、生产消费或事件集合不同的路径仍需有合法表达。引用不直接修复 [conditional-flow-topology](archive/extract_workflow/conditional-flow-topology.json) 记录的历史产物，也不允许正文删 launch、handoff 或事件；该问题归档保留，产物研究另行开展，不能因本候选而宣布历史算子表达已修复。

## DRD-09：必须界定新规则与历史冻结的边界

**已证事实。** [snapshots.rule_files:24](../tools/snapshots.py) 将规范、阶段页、检查器和执行工具纳入冻结规则；[compatibility.py:219](../tools/compatibility.py) 的 compatible runtime 要求字段、验收、批准、依赖、回执等含义不变。[revisions.py:198、254](../tools/revisions.py) 已区分阶段材料、原始批准与当前适用性，并收集 P5 的全祖先依赖。

**待决定。** 新规则修订中，哪些文档和工具必须共同发布，已有未完成任务怎样通过正式 semantic revision 采用新表达；旧内容继续本地写全时，与新引用内容的兼容边界怎样表述。具体规则版本号与迁移范围由方案及实施阶段确定。

**约束。** Design 内容解析和依赖含义的变化属于新 rules revision，不能仅标为兼容执行器修复。旧 run 按原冻结恢复时含义不变；采用新语义需走现有正式语义修订和适用性审阅，历史 checkpoint/receipt 不回写。当前保守祖先摘要可继续使用，不能以新功能为由无差别重跑全部历史任务。

## DRD-10：收益来自明确共享事实，不能靠抹平真实差异获得

**已证事实。** DRD-02 的 fa4 两个节点给出了可复现的完整数学区相同实例，也揭示了直接父引用不覆盖所有相同内容。相反，[MoE/w1_split_ws.yaml:75](../../schema_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/w1_split_ws.yaml) 写 gather 始终提交，[mono_split_ws.yaml:75](../../schema_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/mono_split_ws.yaml) 写某些模式不分配 Xq/Xs、从原张量读取；这里涉及物化与提交差异，不能删掉限定文本制造整区相同。这些历史产物的条件表达是否合规仍归关联 issue，不作为本候选已经通过的样板。[GDN 2.2.7 六节点索引](../../schema_wiki/revisions/2.2.7/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/dispatch.md) 也保留了直接 Q/K、排序和分段的真实设计差异。

**待决定。** 整区引用首版能在实际写作与审阅中减少哪些重复，如何让作者清楚地区分“同一定义在不同参数下成立”和“存在不能忽略的局部差异”。FMHA 的较大派生树是提出问题的极端提示，不据节点数量承诺普遍压缩比；多算子抽样中没有找到与直接父逐结构完全相同的整区，也不等于不存在语义共享。

**约束。** 不要求派生节点数量下降，不要求自动去重，不用文本相似度替代语义判断。局部 patch、兄弟共享、自动换父、通用条件求解可后置；先明确共享定义、有效差异和审阅分工，不为追求覆盖率扩成通用组合语言。

## 后续方案需要贯通的少量场景

方案可围绕四条完整路径检查上述决定是否闭合，无需把每个问题展开成大量独立验收项：

1. 传递引用在子参数环境中解析，贯通检查、诊断、绘图和 standalone；来源回执能识别实际 provider，生成时 provider 漂移可被发现。
2. 父内容变更与 P4 换父，贯通原稿保留、原始/有效身份、合法改稿、必要后代失效与重审，同时保留无关兄弟进度。
3. 子节点发现非 root provider 错误，能够合法返回指定 P5 定义节点修复、独立重审并继续工作，不靠无关计划变更绕路。
4. 旧规则恢复与正式采用新 rules revision 分别执行，旧 checkpoint/receipt 原样保留，新表达只由采用新语义的上下文解释。

调查时本记录仅表示问题与证据已整理，尚未决定语法或实现。此后 2.5.0 已按[实施设计](../DESIGN_REGION_DERIVATION.md)完成框架范围并归档；正式规则采用能力已验证，真实旧 run 的实际升级和历史产物修订未在本次执行。局部 patch、兄弟共享、自动换父/去重、通用条件求解等有意后置；条件流程 issue 已按用户决定归档，历史产物研究仍另行开展。
