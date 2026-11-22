# R2：独立四流程、最小契约与 implement 后端边界

日期：2026-11-20。性质：架构研究建议，全部新增接口、目录与阶段安排均未确认、未实现。
已读用户简报、历史、两份 R1 报告与 KernelSchema/AGENTS.md；本轮只读核对源码并写本文。
未改实现或既有 Schema，未运行模型、GPU、测试、安装、提交、远端操作或读取凭证。
证据前缀：KS=/home/shaohanyong.shy/KernelSchema；AKA=/home/shaohanyong.shy/atrex-kernel-agent；EXP=/home/shaohanyong.shy/AKA_experiment。路径行号指当前工作树。

## 1. 推荐边界及其适用条件

**建议选择 C 的分期形态：KernelSchema 拥有四流程产品入口、后端中立交接契约和有限但真实可用的参考 implement；AKA 拥有增强执行后端及自身的 schema 编排。**
首期参考 implement 可以采用 A 的薄循环思路，但不能把当前 AgentCLI 或 Agate 的存在误写成该循环已经可用。
只有用户接受“必须安装或接入 AKA 运行后端才能生成并验证代码”，才应把 B 作为长期唯一 implement 方案；这个选择本身合理，但公开承诺不同。
中立契约只约束交接的输入、产物、证据和续接身份；不要求两后端拥有同样的 scheduler、会话实现或晋升规则。
本轮建议的最大取舍：为真正独立的产品能力承担一小套 coder 调用与产物续接维护，换取 KernelSchema 不被 AKA 的完整 campaign 生命周期绑住。

| 独立的含义 | 最低可验收承诺 | 仅提供 AKA 包装能否满足 |
| --- | --- | --- |
| API/CLI 独立 | 用户从 KernelSchema 的明确入口提交四种任务、取得结构化结果 | 可以，前提是包装真实调用且返回事实；四份 Markdown 不够 |
| 运行依赖独立 | 无需本地 AKA 或远端 AKA 服务，也能完成声明范围内的设计、实现、验证 | B 不满足这个较强定义；A/C 可满足，但仍需要模型运行时和兼容验证环境 |
| 产品可用性 | 支持范围、安装条件、材料准备、失败返回和 demo 均可复现 | 取决于首发质量；一个不能编译或永远返回“请装 AKA”的入口不算独立 implement |

“独立”不意味着内置模型、免费 GPU 或支持所有编程栈；支持一个公开可取得的模型接口和一种明确执行环境即可构成真实首发。
托管 AKA 服务可让 B 的用户无需安装 AKA，改善使用体验；它仍有 AKA 后端运行依赖。用户若只要求独立前端，这完全可能是合适产品。
源码事实：三流程目前仍是方法文档；implement 明说使用任务提供的设施，仓库无统一 GPU Harness。KS:IMPLEMENT_SCHEMA.md:3、:46；R1 Schema 报告第 2 节。

## 2. 先准备任务契约，避免让四入口各自猜题

建议有一个轻量的任务准备步骤，既可读取调用者已经给定的任务，也可报告缺项；它不是新增第五个 agent 工作流。
算法与硬件 Contract 由任务方给定或确认，planner 只落实已授权的内容；已有确认可复用，不增设逐层审批。KS:README.md:48；KS:DESIGN_SCHEMA.md:7。
“任务方”可以是可信 benchmark manifest 或已授权项目配置，不必每次由人交互点确认；缺少信息时只停止依赖该信息的部分。

| 最小材料组 | 需要固定的内容 | 不应被替代成什么 |
| --- | --- | --- |
| 外部语义 | ABI、全部输出/副作用、有效输入域、dtype/布局/别名规则、参考入口与完整误差公式 | 现有 kernel 恰好支持的子域，或随机输入生成器的偶然分布 |
| 目标环境 | 设备、实现栈与版本、支持关键设计判断的硬件/编程资料；构建与执行可用性另记 | 一个 GPU 名称，或可解析但尚未绑定的 Wiki 占位 |
| 工作口径 | 必须实现的范围、优先 workload、比较实现、优化目标、计时边界 | 把优先 shape 当作整个 Contract；把主 kernel 时间当完整调用时间 |
| 实现与验证设施 | 可空的代码起点、任务提供的构建/验收/测量入口及身份、允许的工具能力 | 把 schema 当作 evaluator，或让 coder 自行放宽判据 |
| 调用控制 | 操作、角色、预算、产物位置、可选续接句柄；模型与后端实际选择留档 | YAML 新增 episode、预算、运行状态字段 |

任务已定义明确但仍有待实测的资源占用或性能假设，可继续设计并如实交接；不能要求 design 先证明最终性能。
关键数学/接口/可实现性前提缺失则返回缺项或草稿，不把它们包装成普通性能假设。KS:DESIGN_SCHEMA.md:7、:11、:13。
**extract 例外必须原样保留：**按固定来源提取，硬件除最小 Execution 登记外留占位，检查使用 skip-hardware；完整外部验收公式可缺失并保留来源记录。
这种输出可以是成功提取，但不能直接成为普通实现的充分前提。KS:EXTRACT_SCHEMA.md:44、:48、:50、:54；KS:schema_extract/tools/publication.py:82。
从 extract 转入 design/optimize/implement 时，由准备步骤报告硬件、验收和范围缺口；用已确认任务材料建立新的实现输入快照，保留提取原件和原回执。
不要求手写 Schema、人工审修成果或 design 产物持有 P6 receipt；receipt 是一种来源证明，不是通用输入资格证。R1 Schema 报告第 5 节。

## 3. 四流程最小输入、输出及返回语义

以下是拟议协议，不是已有 API 名或命令语法。可以先以版本化 JSON 和 Python 数据对象表达，无须先造公共 SDK。
共同结果至少包含：请求身份、状态、产物及摘要、检查范围与结论、普通反馈、消耗与停止原因；有可继续工作时再给续接句柄。
建议共同状态只表达本次交付：`completed / partial / blocked / failed`；另保留操作决定和逐项检查，避免一个 passed 同时表示格式、数值、性能全部通过。
`partial` 表示有可用草稿或代码但交付条件未齐；`blocked` 明确缺少外部前提；`failed` 表示执行/协议错误。预算结束通常是 partial 加原因，不等于设计不可实现。
外层进程正常结束只表示结果可读取；领域结果必须读结果对象。既有 extract 的原生状态与控制命令继续保留。
共同状态是交付摘要，不替代各后端 running/cancelled 等生命周期；诊断还要区分任务缺项、设计待决、代码缺陷、基础设施和协议错误。

| 流程 | 最小输入 | 最小输出与 completed 的含义 |
| --- | --- | --- |
| design | 已确认 Task；规则与参考材料；可选只含已确认 Contract 的起点；写作预算 | raw 任务目录、dispatch、有效视图来源清单、交接范围、待验证假设；必要决定已落实且任务检查完成，可交 coder；否则 partial 草稿 |
| extract | 现行 Runner 支持的固定来源及入口/IO/reference/acceptance/scope 绑定、输出选择 | 现行 raw、dispatch、原生 run/checkpoint/receipt；包装保留 ready 与是否 published 的区别，不改 P1–P6 或夸大提取验收 |
| optimize | 固定 Task 与当前完整 Schema；优化目标；可选已绑定的代码/实验/普通反馈 | `decision=revise` 的新任务快照与变更影响，或 `decision=keep` 的原身份及理由；两者都可 completed；不强迫产生 YAML 改动 |
| implement | 固定 Task、Schema/规则/选择范围；可空的代码起点；执行后端、验证设施和预算 | 代码快照、覆盖进度、构建结果、三条证据和普通反馈；选定范围已实现且规定的正确性/符合性验收完成，要求的测量也已取得，才可 completed |

design/optimize 的 completed 不是“数学被机器证明”或“已达到性能指标”；结果应保存静态检查与 planner 审阅的实际方法和未实测判断。
implement 的性能目标另报 `met / unmet / unknown`：代码正确且符合设计、已测但更慢，可以 completed + unmet；任务调用者仍可认定最终优化目标未完成。
completed 只覆盖输入明示的本次请求；若它只是总任务的合法子集，还须返回总任务未覆盖列表，汇总时不能升格为整个 Task 完成。
若只有代码、没有兼容 GPU，可交 partial，并把运行/测量标为 not_run；这是有用产物，但不得宣传为验证通过。
extract 若请求发布却出现冲突，须显示原生 publication outcome；`state.status=ready` 不自动变成“交付目录已发布”。KS:schema_extract/tools/publication.py:41、:139、:162。
optimize 无实测反馈时可做分析或保留设计；仅需要更多性能证据通常不是缺少任务前提的 blocked。KS:OPTIMIZE_SCHEMA.md:23、:39。

**具体接口用例（以下 T 为假想教学任务，未创建文件、未运行）：**

- T：对连续 FP32 矩阵逐行 softmax，M∈{1,32}、N∈{128,1024}，输入有限且在 [-20,20]；写满独立 out 并返回该对象。
- T 的材料固定目标设备/栈及 FP64 稳定求值参考；示例判据为转 FP64 后逐元素 `abs(y-g)<=2e-6+2e-5*abs(g)` 且拒绝非有限值；计时包含完整调用、输出预分配，并固定测试/测量入口。
- design(T)：选择稳定 softmax 的计算、每行工作组织与完整流水；输出 root/dispatch 和来源清单。只能落实数学、尚未决定工作归属时返回 partial，不能靠模板占位冒充 ready。
- extract(T 的固定已有实现)：走现行 Runner，交付原生 receipt；即便 published，若硬件仍占位，准备步骤仍列出 implement 前提缺项。
- optimize(T,S1,E1)：若真实 E1 仅说明实现漏写尾部，返回 keep 并交 coder 修复；若证据支持改变物化或工作分解，返回 revise 与受影响层。两种决定都不编造速度提升。
- implement(T,S1,空代码)：输出 C1 及证据；若两次 coder 调用后只覆盖 N=128，返回 partial、明确 N=1024 缺口及 C1 续接；再调用可从 C1 开始，不必重新从空代码生成。
- implement 的另一个合法结果：覆盖齐全、正确性与符合性按任务政策接受、性能已测但目标未达，返回 completed + objective=unmet，并把数据交给 optimize。

## 4. raw、effective、dispatch 与“以 root 为主干”

**已核实语义：**root 保存唯一 Contract，其他节点的三区可整区 from；when 可重叠或不覆盖 Contract，root 只是默认阅读入口。KS:schema_docs/0.common.md:136、:142、:148、:194。
dispatch 是目录和变体记录，不是机器 selector、patch 序列或 fallback 定义；当前 YAML 不受目录变体自动修改。KS:schema_docs/0.common.md:184、:186、:190。
因此实现范围应由 Task/Planner 交接明确，coder 据已定选择落实代码路由；“实现所有 dispatch”不是可直接执行的命令。

| 输入视图 | 最小用途 | 身份与限制 |
| --- | --- | --- |
| raw 完整任务 | planner 修改、审阅引用、定位原定义；小集合可整体冻结 | 保留每个文件原字节摘要和 META_DATA.Version；不能只复制目标节点 |
| effective 选定节点 | coder 理解完整计算与使用节点绑定 | 由同一规则和 raw 闭包生成，带 source_manifest/origins；用 effective_document/effective_yaml 补入 root Contract |
| dispatch 原文 | 理解当前完整设计、差异、候选及条件 | 单独记录摘要；明确本次采用哪些记录、具体选择与必要联动；它没有独立完整方案身份 |
| 外围范围选择 | 本次必须实现什么、顺序及尚缺什么 | 区分 Contract 域、要求实现域、node.when、测试样本；记录选择理由与显式重叠处理，不臆造通用路由器 |

非 root 的 `effective_task[node]` 不自动包含本地 Contract，自包含交接应取 `effective_document(node)`。KS:schema_template/tools/design_resolution.py:53。
建议 root-first 是可观察的工作顺序：先贯通 root 中代表性路径，再按已选定的必要节点/变化补齐；过程进度不等于任务完成。
若 root 不适用于优先 workload，可以先据它理解共用 Contract，再实现适用的目标节点；不能为“先 root”强行使用不合法方案。
实际反例：FA4 decode root 限 Q=4、B∈{15,16}、L∈[24576,131072] 等；完成它不代表覆盖其共享 Contract。KS:schema_template/examples/fa4_decode/root.yaml:8。
首期可用显式 case→node 的任务映射和未覆盖列表帮助测试；这只覆盖列举案例，不能据此声称已证明所有符号域的覆盖。
选择了只写在 dispatch 的变化时，交接必须固定其具体设计决定和条件；是否整理成完整 YAML 按现行任务要求办理，不强制为每次局部实验新增节点。

## 5. 身份与轮次：严格追溯，避免把字节变化误当优化

当前 effective_identity 哈希内容仅为 node、source_manifest、origins；source_manifest 含 root、自身及实际引用链 raw 摘要。KS:schema_template/tools/design_resolution.py:39、:47。
它不含规则版本、dispatch、Task 口径或环境，也不是纯 Design 语义哈希；仅排版、Feedback 或其他 raw 数据变化也可能改变它。
META_DATA.Version 按本文件数据递增，纯排版不递增；上游改动可改变消费者有效设计而不改变消费者 Version。KS:schema_docs/0.common.md:43。

建议外围只增加必要的身份关系，不新增“语义等价哈希”算法：

1. **任务身份**固定 Contract、ABI/reference/验收、目标环境与要求实现范围；优化目标/计时范围属于请求口径，一并追溯。
2. **Schema 快照身份**固定规则/Template 版本、raw 字节、dispatch 字节；保留每个选定节点的现有来源闭包和 effective_identity。
3. **本次选择身份**绑定该任务、快照、选定节点/目录变化与实现范围；否则相同 root 被不同 workload 或局部实验调用会混账。
4. **设计 revision**记录 planner 的有意修订及父 revision、原因、受影响层；相同设计的 coder 调优不增此计数，optimize=keep 保留原身份。
5. **实现 attempt**固定上述设计依据，持有可演进代码、覆盖进度和预算；它可包含多个 coder episode/model session。
6. **评测身份**绑定精确代码快照、上述输入依据及 evaluator/输入/环境/参数/计时范围；不能只挂在 episode 序号下面。

Feedback 追加或排版变化形成新的字节快照，但不自动算新 planner 设计 revision；记录变更性质，旧测量仍引用当时快照。
有证据的非设计改动可经过明确核对关联旧证据，避免每次排版都重测；不能直接重标旧记录为“新快照已实测”。
选择范围改变通常产生新 implement 请求；已有代码可显式作为起点，不强迫 raw 版本递增，也不沿用旧范围的完成状态。
新设计可以迁用旧代码作 seed，但须建立新 attempt、记录迁移与待复验范围；旧 CLI checkpoint 不能充当已采用新设计的证明。
最小续接承诺首先应是“从精确 C1 继续工作”，不必承诺任意后端会话或机器之间无损恢复模型内部上下文。

**AKA 的真实约束：**新 episode 从 canonical revision 创建 candidate，未晋升源码不是默认下一轮起点；终端源码可由 public-candidate 产物保留。
证据：AKA:src/aka/workflow/light/long_horizon/campaign.py:1785；AKA:src/aka/workflow/legacy/episode_output.py:52、:74、:103。
所以 AKA backend 必须增加明确的 partial seed 接纳/续接语义，或先限定为单次边界；不能宣称“加 max_iters 就能继续固定 schema 的半成品”。
AKA continuation 还冻结配置、资源和实现身份，restore 会逐值核对；EXP schema 另绑定 session。动态改写 manifest 再 resume 是违约路径。
证据：AKA:src/aka/bootstrap/continuation.py:57、:68、:161；EXP:tools/python/aka_experiment/schema_sessions.py:54；R1 AKA 报告第 4.5 节。
首个 AKA backend 可按不可变请求/新运行边界消费 revision，明确承担启动成本；更细的 campaign 内 revision 接纳应交由 AKA 生命周期设计验证。

## 6. 代码是共同锚点，验收保持三条证据链

每次结果首先保存实际交付代码内容或可取得的不可变引用、内容摘要、入口与构建环境；Git commit 可用，但不强迫未完成代码先成为 incumbent。
构建日志也是证据，不能仅留下 coder 的“已编译”文字；构建成功仍不等于后续三项通过。

| 证据链 | 最小结论与所需绑定 | 不能推出的结论 |
| --- | --- | --- |
| 正确性 | 对精确代码按固定 reference/公式检查全部输出与副作用；记录输入覆盖、判据、环境及实际结果 | 有限测试通过不能证明全域；也不能证明采用了指定算法/工作组织 |
| fidelity（设计符合性） | 对选定有效设计核对数值路径、工作归属、launch/handoff、role/等待/存储寿命与 Require/Forbid；记录方法、代码位置和未决点 | 文档投递成功、Schema checker 通过或 GPU 正确性通过都不等于代码符合设计 |
| 性能 | 固定比较代码、workload、环境和计时边界，记录实际延迟/指标、覆盖与不确定性 | 更快不自动授权改变设计；更慢也不抹掉正确、符合设计的实现产物 |

依据：KS:IMPLEMENT_SCHEMA.md:46、:50、:51、:52、:54。
首期 fidelity 可组合静态可核对项和独立只读审阅；复杂异步保证按需要补生成代码、轨迹或工具证据，不承诺通用数学证明。
结果建议逐项用 `pass / fail / not_run / inconclusive` 并给方法及覆盖；模型自评与独立审阅明确区分。可接受的方法由任务政策给定。
这允许先交付可审阅实现，也避免把“所有任务全自动可信 fidelity verifier”变成四流程第一版的前置研发项目。
Task 提供 evaluator 权威；模型可调用受控构建/评测工具，不能用修改验收器来换取通过。AKA 已有 source-bound Evaluation/Experiment，可复用其事实而非复制 GPU 链路。
真实反例：QSA 正式 Contract 使用 FP64、以实现值为相对基准；历史 harness 用另一 FP32/参考基准口径。复用同名 harness 仍须核对。KS:schema_template/examples/qsa_prefill/README.md:21。
AKA 的 candidate acceptance 与后续 ABBA 晋升亦是两道门；backend 返回实施证据与晋升结果两个事实。AKA:src/aka/workflow/light/host/journal.py:1304；AKA:src/aka/workflow/light/long_horizon/recorded_verifier.py:107、:124。

## 7. 普通反馈在外部，Prefer 替代证据才进入 Feedback

普通结果记录保持“观察→归因→建议”的区别，绑定相关节点/对象、触发输入、代码与证据，以及继续哪些工作、需要谁处理。
编译失败、预算耗尽、资源不足、Require 冲突、覆盖缺口、性能诊断及未实测建议均可交给调用者/planner；它们不是 REALIZATION.Feedback。
实现 bug 留给 coder，同一设计中的自由实现选择继续比较；确需改变数学、调度或执行条件时才请求 planner 修订。KS:OPTIMIZE_SCHEMA.md:15；KS:IMPLEMENT_SCHEMA.md:58。
一次工具链报错不足以证明设计不可实现；应报告可复核的具体矛盾和已尝试范围，不让 stop_reason 取代论证。
只有替代某条 Prefer、保持 Contract/Design 与硬条件、具有可比较的真实实验证据时，才允许追加 Feedback；不修改原 Require/Prefer/Forbid。
Item 对齐同一上游对象，Choice 明确替代哪项，Evidence 必须是 `repository@ref:path` 的不可变真实记录。KS:schema_docs/6.realization.md:44、:48、:52、:58。
本地临时日志、一个内容哈希或未提交实验目录尚不满足现行 Evidence 引用格式；先留在外部结果。取得合规不可变引用后再追加，不能为让流程完成伪造 ref。
推荐 coder 返回“拟追加条目+证据”，由宿主核对字段权限并应用；授权边界不改变，也不让 coder 获得整个 Schema 写权限。
先固定测量时的 S1/C1/E1，再追加 Feedback 得到 S1 的新字节快照；E1 继续指向测量时依据，避免“证据必须包含尚未写入的自身引用”的循环。
完整实验、预算与历史留在外部记录，不扩展 REALIZATION 成实验数据库。KS:schema_docs/6.realization.md:69、:73。

## 8. design / optimize 共用什么，怎样避免知识平台化

两者应共用同一写作内核：Task 准备、参考材料选择、规则阅读、Schema 草稿工作区、Template 检查、权限差异检查与交付；区别在起点与决策目标。
design 从 Contract 逐层确定 Math→Schedule→Pipeline；optimize 从证据定位最早受影响层，保留未受影响部分，可返回 keep。KS:DESIGN_SCHEMA.md:17、:27、:37；KS:OPTIMIZE_SCHEMA.md:43。
角色流程正文应继续作为提示方法的真源；运行时补请求事实、材料清单和上轮诊断，不复制一份很快漂移的“第二套规范”。
引用候选先用现有 catalog 的 family/precision/stage/关键词过滤与 JSON 输出，再由 planner 审查任务域、来源、验收和适用环境。
catalog 当前是 AND 过滤目录，不提供语义近邻排序；零命中或不相容参考应诚实返回，不为每次 design 强行凑一个“最近邻”。KS:kernel_wiki/tools/catalog.py:335、:377、:394。
schema_wiki 与 Template 案例可按现有目录定位；首期只需把选中的路径、版本/摘要、相关性理由与缺口带入上下文，无需先给全部资产建 embedding 服务。
查硬件/编程/微测/Profile 时沿任务绑定入口按问题阅读；DOC2 已允许已核对的上游入口，无须先统一复制成自建 Wiki。KS:schema_docs/2.contract.hardware_specification.md:117、:119。
校验直接使用 Template 的只读/规范化规划 API；raw 用于修订，effective 用于交接；完整文件入口负责 dispatch/本地引用检查。KS:schema_template/tools/README.md:33、:39、:41、:42。
建议流程为：模型草稿→显式规范化候选→冻结 Contract/角色权限差异检查→完整结构关联检查→受影响语义审阅→交付；草稿和最终快照分开。
自动规范化可能改共享定义及引用，不能未经差异核对写回调用者原目录；上游改动后按真实依赖闭包复查消费者，不全量重写每个节点。
不照搬 extract 的 P1–P6、approve/rework 预算、维护平台或发布授权；这些是源码忠实提取的具体政策，不是普通 planner 所需公共基础设施。
AgentCLI 已有 run(attempt,prompt,schema)，review-eval 在 P1–P6 外使用，说明可做小块复用。KS:schema_extract/tools/agent.py:209；KS:schema_extract/tools/review_eval.py:186。
但当前仅接受 worker/reviewer、稳定空 cwd 与 YAML 工件，Claude Bash/MCP 权限也限制很窄；不能直接换成 coder prompt 即获得代码/Agate工具。KS:schema_extract/tools/agent.py:27、:267；KS:schema_extract/tools/artifacts.py:224；KS:schema_extract/tools/backends.py:171。
优先验证新写作入口能否在现有调用面外做薄适配；若需共享底层 launcher/event/usage，先限定接口并保留 extract 行为，不把“抽公共运行时”变成前置重写。

## 9. A / B / C 的认真比较

| 维度 | A：自有薄 loop + AgentCLI/Agate 等 | B：仅薄入口调用外部 AKA | C：中立契约 + 参考后端 + AKA 后端，分期 |
| --- | --- | --- | --- |
| 首个可展示能力 | 单任务、单栈可以很直接，摆脱晋升策略约束 | 已有 AKA 环境方便，但 schema 语义仍需适配 | 首期与 A 相近，同时约束交接以免后续重写 |
| 独立运行 | 可以；模型与任务 evaluator 是显式依赖 | 不可以脱离 AKA；入口独立 | 基础能力独立，增强能力按需安装 |
| 现成资产复用 | Template 可直接用；AgentCLI 要适配；Agate 只提供执行 | 可复用 AKA Agent/Workspace/Evaluation/Experiment/Execution | 共享 Schema 交接与证据含义，分别复用适合的运行设施 |
| 最主要新增工作 | coder 权限/工具、有限循环、代码保存、预算和结果收敛 | 固定设计 prompt、fidelity、partial 续接、完成与晋升分离、配置绑定 | A 的小后端 + B 的适配，以及少量跨后端兼容验证 |
| 最容易失控之处 | 逐渐复制 AKA 的 campaign、恢复、评测调度与插件系统 | 以“调用已有 AKA”掩盖环境安装和生命周期改造 | 过早设计万能后端协议，或为了功能对齐维护两套完整平台 |
| 合适条件 | 目标始终是少量明确环境中的教学/研究工具 | 用户群主要已使用 AKA，接受依赖它 | 用户同时重视公开独立能力和 AKA 集成，允许分阶段投入 |

A 的最佳形式不是自写 GPU 平台：模型运行时负责原生 agent 工具循环，薄控制器只固定任务、调用、评估、保存并决定有限次继续。
本地任务 evaluator 或已安装 Agate 是执行适配对象；Agate 作业完成并不自动提供本任务正确性/计时/符合性，仍须明确任务适配和结果解释。
B 也绝非简单 import 一个 Light 文件：Light 依赖 Legacy，后者依赖 framework，且完整链路涉及 evaluator、资源与执行环境。AKA:packages/atrex-aka-light/pyproject.toml:10；R1 AKA 报告第 8 节。
B 的合理短期用途是先给已有 AKA 用户接通固定 Schema backend，同时明确独立参考后端尚未交付；若永久止步于此，应公开写明运行依赖。
C 只在契约维持很小时值得选；先实现一个后端、用第二后端的真实映射反证接口，不同时铺开两套完整生命周期。
反向证据也应保留：若小参考后端必须为 demo 重建复杂 remote recovery、隔离和数值验收，A/C 的“薄”假设已失败，应缩小支持范围或暂选 B。
若 AKA 适配只需固定请求即可导出完整实现证据、目标用户也已有 AKA，B 的投入回报可能更好；这属于产品目标判断，不是技术上天然劣等。

## 10. standalone implement 的责任上限与最小后端接口

建议参考后端承诺：单任务、固定 Schema 选择、串行 coder/验证、显式调用/评测预算、一个起点代码快照、可继续的未完成代码、可核对的结果。
implement 内不自主改 Schema；planner 的 optimize 发生在它返回之后，由外层调用者决定。首次设计可给更大的实现预算，不按每个 episode 强迫一次 optimize。
最少持久化输入摘要、调用/代码快照、已结算消耗、结果与续接指针即可；不要求新的数据库、事件总线或多任务调度器。
外层分别记录 planner/coder/reviewer 与评测消耗再汇总；续接不能抹掉已消耗额度。标明预算的硬停止或调用后结算边界，未知 token 不填零。
基础设施失败与设计问题分开；有未结算 GPU 作业时交给执行适配器恢复/结算，参考后端不能盲目重提后把重复实验当一次调用。
边界之外：多任务排程、incumbent/ABBA 晋升、复杂候选搜索、自动扩展方向、多主机隔离与长期精细恢复，优先由 AKA 承担。

| 拟议最小接口面 | 必须保持稳定的含义 | 后端私有部分 |
| --- | --- | --- |
| 请求/结果协议版本 | 不兼容版本明确拒绝，不静默丢字段 | Python 包组织、CLI 启动方式 |
| `run(request) → result` 的概念调用 | 请求含固定 Task/Schema/范围、代码起点、预算；结果含精确产物与证据 | 原生 episode、V0/V1、promotion、会话管理 |
| 能力声明 | 支持栈/评测方式、是否能续写指定代码、是否有原生 session resume、可报告哪些证据 | 资源注册表、特定模型参数与执行插件 |
| 可选 continuation | 绑定原请求/代码/预算；后端不支持则显式回退为新 session 从代码继续 | checkpoint 格式和存放位置；不可跨后端直接重用 |
| 资料读取与结果证据引用 | 固定规则/有效视图/原文可查，结果引用可取得并校验摘要 | 文件挂载、工具协议、远端产物存储 |

可先支持 Python 调用与显式指定的 JSON 子进程适配之一；不需要插件发现市场。跨语言/跨环境使用时再固定序列化细节。
KernelSchema 不暴露 AKA 内部 dataclass、campaign 目录或 vN 作为公共身份；AKA adapter 负责原生事实到上述结果的映射。
共享的是 Task/Schema 准备、planner 方法、范围/权限判断与证据要求，不是把两后端所有功能压成最低公分母；额外能力放有命名空间的扩展事实。
后端无法提供 fidelity 或完整覆盖时必须返回未决状态，不能由中立 wrapper 代签通过。

## 11. 三仓目录职责与依赖候选

下表“拟新增”均是职责候选，不表示路径已存在，也不是当前获准的改动清单。

| 仓库与目录候选 | 所有职责 | 避免混入 |
| --- | --- | --- |
| KS 现有 schema_docs/、schema_template/ | 语言、静态校验、规范化、有效视图与来源 | campaign、模型会话、GPU 调度 |
| KS 现有 schema_extract/ | 保留原提取工具、状态与回执；外层读取其交付 | 借本次任务改写 P1–P6 |
| KS 拟新增 schema_design/、schema_optimize/ | 两个真正可调用的 planner 入口；共享写作能力 | 两套独立检索器/提示规则/Schema parser |
| KS 拟新增 schema_implement/ | 公共入口、有限参考后端及实现结果合同 | AKA 私有 candidate/promote/checkpoint 逻辑 |
| KS 拟新增轻量公共包，例如 schema_workflow/ | 请求/结果、快照/范围、资料组装与权限核对；命名待定 | 完整 agent 框架、统一后台平台 |
| KS 现有 wiki + 拟新增流程 demo 目录 | 已有资料与可复现 Task、演示编排、真实证据；旧示例继续说明其原范围 | 实验矩阵、私有模型配置、把历史快照当当前验收 |
| AKA 可选 schema workflow/adapter 目录 | 调 KS planner、接 coder prompt/工具、native 状态协调、AKA implement backend | 实验 cohort、消融条件的业务政策 |
| EXP 现有 tools/python/aka_experiment/ 等 | 数据集/矩阵、批次预算、排程、基线/消融与资格验收；消费固定 AKA 适配版本 | 通用 planner、第二份 Schema 规则与永久 core patch 仓库 |

建议的源码/安装方向：

```text
AKA_experiment  ──固定版本──>  AKA 适配 fork  ──可选 submodule/package──>  KernelSchema
KernelSchema 核心  ──>  Python / Template 依赖
KernelSchema 参考后端  ──>  模型运行时 + 任务 evaluator（本地或 Agate）
```

KernelSchema 核心与参考后端不反向引入 AKA gitlink，也不把 AKA 列为必装依赖；中立协议可在 KS 中维护，无须第四仓。
选择 AKA backend 时允许运行调用为“KS 前端→外部已安装的 AKA adapter→AKA coder/eval”，adapter 可使用 KS 核心解析资料。
这不是递归编排：AKA backend 只执行固定实现请求；AKA 自身的 schema workflow 直接调 KS planner，不能再回调选择 AKA 的 KS implement 前端。
组件依赖应拆为前端/核心/adapter：两个运行主体都依赖核心，核心不导入主体；仅靠“可选”一词不能消除互相 import、自动启动或递归 submodule。
反例是 KS submodule AKA、AKA 再 submodule KS，或 KS 默认安装 AKA 而 AKA 安装时又要求完整 KS 主程序；前者应禁止，后者也应避免靠安装器碰巧容忍循环。
submodule 只固定来源，不自动解决 import 包名、运行依赖或资源分发；目前 KS 两个 tools 包需要明确命名空间，资源定位也不能依赖 cwd。R1 Schema 报告第 2 节。
当前 EXP 仍是 assets/aka 普通文件 vendor+overlay，并非这个图；迁移需保留来源/补丁台账，不能忽略其已有工作树成果。EXP:tools/lib/aka_vendor.py:70、:93；R1 AKA 报告第 2 节。
AKA schema-off 应不启动 planner、不安装/加载必需 Schema 运行路径，并保持选定原生流程行为；现有 EXP control 仅“不投递 schema”，不证明完整原生等价。R1 AKA 报告第 6 节。

## 12. 第一条可展示价值的路径与分期

建议第一条真实价值路径是：**资料与验收闭合的小型单节点任务→design→standalone implement→正确性/符合性/性能证据**，过程不安装 AKA。
可候选逐行归约/softmax 这类一份 root 可表达完整方案的教学任务；最终题目与设备尚未选定，不能把本文 T 当已存在 demo。
它需包含固定参考、完整比较器、明确 workload、设备/编程资料与可取得依赖；优先消除首次使用者无法复现的外部资产，而非选最复杂 YAML。
首轮无需宣称胜过最佳实现；真实实现指定设计、解释检查范围、保留未完成代码并能续写，已经证明四流程架构中当前缺失的产品能力。
随后以该次真实结果调用 optimize，再 implement 新 revision 或保留设计；要验证 revised 与 keep 两条结果，不以强制改 YAML 制造“优化成功”。
再选一个真实多节点任务验证来源闭包、root 子域、必要 dispatch 和未覆盖反馈；FA4 decode 可作范围反例，但当前资料缺口不自动满足端到端 demo 条件。
最后让既有 extract 产物通过同一准备步骤进入 implement，验证手写/design/extract 三种来源均可消费；extract 内部保持原样。
AKA backend 首先消费同一固定实现请求并回传同类事实，之后再实现其动态 revision 编排；不让独立产品首发等待完整 AKA campaign 重构。
公开演示应同时提供可阅读的真实产物与重跑条件；离线检查/回放对无 GPU 用户有价值，但清楚标明它没有新生成或新测量 kernel。

## 13. 少数产品选择与可先验证的技术未知

**需要用户判断的产品选择，建议控制在三项：**

1. 独立承诺：是否要求不依赖本地或远端 AKA 运行后端也能生成、验证一个支持范围内的 kernel？若是，推荐 C；若独立前端已足够，B 可先单独交付。
2. 首发支持面：先一个公开小任务/一种栈和环境，还是先接当前复杂 AKA 算子？推荐前者证明独立能力，再用后者检验集成；Agate 可是支持后端，但不应默认为所有公开用户已具备。
3. fidelity 首发验收：允许有明确范围的只读审阅及 inconclusive 返回，还是要求更强自动化才算完成？推荐先如实暴露方法与未决点，并保持正确性、性能各自有真实证据。

**技术未知可以在实施获授权后先做窄验证，无须先让用户决定内部实现细节：**

- 写作适配：AgentCLI 在非提取 scope 下可复用到什么程度；新增 planner prompt/角色权限与冻结资料能否独立，extract 回归是否保持。
- 公开安装：干净环境中的 Python/模型 CLI/Template 依赖、资料可取得性、两个 tools 命名冲突，以及是否真的不导入 AKA。
- 代码续接：用伪后端证明 partial C1 会成为下一次明确 seed；预算不重置、schema/Task 改动拒绝复用旧 continuation、旧证据不被重绑定。
- AKA 映射：从未晋升 terminal artifact 接纳 seed 的合法接口、正确但更慢的结果导出、V1 与后续 episode 是否都服从固定设计；未核实前不承诺可直接配置。
- 身份反例：仅 Feedback/排版变化、provider 改变消费者有效 Design、dispatch/Task 变化但裸 effective_identity 不变，均应产生正确追溯与复验决定。
- 评测适配：本地/Agate 返回能否固定 source/evaluator/计时范围并处理未结算作业；用 QSA 判据差异检验接口是否会错误接受“同名测试”。
- fidelity 可行性：在简单与异步复杂两类案例上分别评估机器核对、独立审阅的覆盖和成本；不足时返回 inconclusive，不扩大检查器结论。
- 后端可替换性：同一冻结请求由参考后端与 AKA adapter 消费，核心不依赖后端私有状态；AKA off 路径没有额外 planner 费用与行为串入。

以上为研究出的边界、取舍及可证伪假设，不是已完成接口或已批准落地计划；本轮没有性能、GPU 可用性或端到端成功的新实测结论。
