结论：保留 `launch` 与 `handoff` 的分工，现有核心表达能力基本够用；本批材料不支持把 flow 扩成通用控制语言。
不能合并启动许可、数据就绪与实际等待：GDN 的 PDL 独立准备是直接反例；也不能只留数据 DAG，GEMM 重算和 Sparse probe 有不传数据的启动边。
保留 `order` 的粒度区别，尤其 MoE direct 的 `per_worker`；它没有全体屏障，也不是逐工作一一配对。
建议从 flow 删除 `protocol` 字段：保留全部事件关系，具体启动协议与必要能力由已有 realization 绑定承担；这是新规则提案。
需要澄清三处默认规则：局部副本先按 worker 身份匹配；`wait_at` 可落在无工作也执行的 worker 入口；已有入口/存续初态不重复 `initialize`。
`phase` 的定义身份与本次执行身份继续合一且各提交一次，重复计算用不同 phase ID、相同内容用 YAML alias；无需另设 launch ID 或 phase 模板层。

本文只评估 `design.work_schedule.flow`，以本轮 [规则快照](/home/shaohanyong.shy/KernelSchema/schema_discussion/flow_review/rules/4.design.work_schedule.md:234) 为准。旧 YAML 的 `resource / milestone / waits_for` 仅用来恢复设计事实，不视为现行语法，也不重新讨论已确定的 storage、index、event、view.region 或 using_verb 设计。未读取其他本轮代理报告；静态阅读了十个 root 的 flow、相应 phase 输入/输出，并补看以下分支和固定源代码。没有改正式文件，没有作 GPU 正确性或性能结论。

**逐字段判断**

| 字段 | 建议 | 它独立保留的设计事实及边界 |
| --- | --- | --- |
| `launch[].phase` | 保留，单 phase 仍使用列表 | 标明实际计算 kernel 的分组边界。同一 kernel 可有多个 phase；不同 launch 即使 mapping 相同也有不同 worker/局部资源实例。不能从 phase 数推断 kernel 数。 |
| `launch[].order` | 保留条件必填；澄清与其他依赖取交集 | 区分每份工作接续、每个 worker 的整段接续、全体完成边界、不同 worker 并发。只描述同一 launch 内的组合；不能由 handoff 或列表相邻推导。 |
| `launch[].start_after` | 保留，省略仍不添加启动边 | 表达设备开始执行所需的控制前驱，包括无数据交接的串行重算、probe 后继以及 PDL 许可。它不是 host 阻塞，也不是数据就绪的另一名字。 |
| `launch[].protocol` | 从 flow 删除，必要绑定由 realization 承担；这是新规则提案 | `pdl` 提示如何实现早启许可与稍后消费的分离，但不自行建立启动边、数据边或等待位置。保留 `launch_ready` 和全部数据/等待事件后，不会丢失这三个时点的关系；具体实现方式与适用前提继续在 realization 明确绑定。 |
| `handoff.<value>.producer` | 保留必填 | 指向本次执行中哪一个 phase 交出该数学值。相同数学表达式可重算、相同数学标量可有多个局部副本；不能只按名称找到某个 target 就猜生产来源。 |
| `.consumer[]` | 保留必填 | 枚举实际消费参与者，确定每个消费者的等待和存续范围。MoE Counts 同时给 scan、fc1、fc2 使用；一条链的相邻关系推不出这些消费边。 |
| `.storage` | 保留显式填写 | 保存全局物化与 worker 局部接续之间最直接的设计区别。跨 kernel 的 global 常可推导，但保留成本标记仍比再加一套默认分类规则直观。若今后允许省略 global，必须明确是新省略规则，不能当前直接省略。 |
| `.initialize` | 保留窄用途；修改填写条件 | 记录当前交接版本中消费者需要、又未由生产或已有入口初态保证的补值。它不是分配 API 日志，也不应把首次清零改成每次清零；累加器初始化与同 phase 合并仍按既有职责处理。 |
| `.ready` | 保留现有默认与显式覆盖；删可证明被支配的重复 | 默认“相关生产 worktile 完成且区域可见”适合局部接续；显式 `kernel_done` 适合必须等整个 grid 的 PDL 数据保证。不能一律默认 kernel_done。 |
| `.wait_at` | 保留；填写条件应覆盖所有实际等待位置偏离默认的情形 | 保留消费者独立准备与后续等待的区别。引用等待的 `done` 允许等待先发起，引用读取的 `issue` 限制读取发起；不能只留动作名而丢时点。还须明确 worker 入口等待在空工作集合时依然存在。 |

“保留”不是要求全部字段每次都填写。当前单 phase 的最小壳已经只有 `launch: [{phase: [MAIN]}]`，见 [规则 4:236–243](/home/shaohanyong.shy/KernelSchema/schema_discussion/flow_review/rules/4.design.work_schedule.md:236)。根本没有 handoff 的 GEMM、MLA、FMHA 和 FA4 prefill root 不应增加空映射或占位协议。

**启动图与数据图必须分开，默认值不能偷偷把两者合并**

1. GDN root 的 prep 在输入到齐后触发后继许可，系数仍未生成完；chain 先读取 caller 的 initial_state，最后才等 prep 完成。`start_after: prep.launch_ready`、`ready: prep.kernel_done` 与 `wait_at: chain.prep_ready` 分别保留三个不同的限制。把 start_after 改成 kernel_done，会删除独立准备的重叠机会；把 ready 改成 launch_ready，会让系数读过早；删掉实际较晚的等待位置，又会把入口以前的独立工作挡住。

   证据：[GDN root flow:326–338](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml:326)、[生产端触发说明:509](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml:509)、[消费端独立准备与 join:599–609](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml:599)。源代码有实际触发 [gdn_kernel.cu:421](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu:421)，以及先 `sreg_load()`、后 `griddepcontrol.wait` 的分支 [gdn_kernel.cu:2211–2224](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu:2211)。FMHA 的 fscan 分支也在 GroupMax 等待前预取 Q/scales 并准备 TMEM，见 [tcg_d128_fscan_cp_direct_narrow.yaml:219–229](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/tcg_d128_fscan_cp_direct_narrow.yaml:219)、[同文件:543–547](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/tcg_d128_fscan_cp_direct_narrow.yaml:543)。这不是仅能在理论程序里构造的需要。

2. GEMM `wide_recovery_chain` 有 first、retry、final 三个提交，各自从原始输入重新计算并覆盖完整输出，彼此不读取旧输出，因此没有 handoff。启动完成边仍保护写后写次序，也保留重复计算的成本。若只根据 handoff 生成 launch DAG，这三个提交会失去必要次序。它已经用 `&wide_work / *wide_work` 和 `&wide_pipe / *wide_pipe` 复用内容，说明额外建立“phase 定义 + launch 实例 ID”两层未必减少登记。

   证据：[wide_recovery_chain.yaml:149–169](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gemm/wide_recovery_chain.yaml:149)、[同文件:227–249](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gemm/wide_recovery_chain.yaml:227)、[同文件:336–340](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gemm/wide_recovery_chain.yaml:336)。

3. Sparse `probe_hidden_packed` 的 prefix 不等 probe，remainder 只等 probe，不等 prefix；probe 的结果由 host 消费，两个 attention kernel 不读它。`start_after` 因而仍有意义，而伪造 ProbeRatio handoff 会改变数据归属。把 launch 列表默认为顺序执行，也会错误串行化 prefix。

   证据：[probe_hidden_packed.yaml:298–326](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/probe_hidden_packed.yaml:298)。四模板中也有更小的反例：[GDN m64_single_sequence.yaml:85–100](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/template_gdn_prefill/m64_single_sequence.yaml:85) 保留先提交的 index_cast，但 Cu32 被传参而不被该 core 读取，所以没有 handoff。这说明启动边与数据边并非偶然重复。

因此建议原样保留 [规则 4:260](/home/shaohanyong.shy/KernelSchema/schema_discussion/flow_review/rules/4.design.work_schedule.md:260) 的无相邻默认依赖，也保留 [规则 4:377–381](/home/shaohanyong.shy/KernelSchema/schema_discussion/flow_review/rules/4.design.work_schedule.md:377) 的设备启动、数据就绪、消费时点区分。没有边表示未禁止重叠，不保证实际同时驻留或加速。

**同 kernel 的最小组合：已有区别要保留，尚无证据需要组合树**

MoE direct 用 `launch: [{phase: [prelude, direct], order: per_worker}]`。每个 CTA 先处理自己归属的无 route 输出行，再独立扫描全体 route 形成自己的 Census；随后同一 CTA 处理其连续 route-unit 区间。这是真正不同工作分解之间的同 worker 延续：prelude 的 partition 是 replica，每个 worker 一份；direct 的 partition 是 unit，每个 worker 可多份、也可因 Census=0 没有 payload 工作。

证据：[direct.yaml:183–226](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/direct.yaml:183)、[同文件:230–247](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/direct.yaml:230)、[同文件:273–285](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/direct.yaml:273)。源代码只做 CTA 内的 census 同步，然后立即进入本 CTA 的区间，见 [source_nvfp4_moe.cu:1960–1985](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:1960)。

| 尝试替代 | 丢失或加入的关系 |
| --- | --- |
| 改成 `per_worktile` | prelude 的 replica 与 direct 的 unit 不是相同坐标集合、相同数量的一一配对；会混淆“一次 census 后多个 unit”和“每个 unit 一次 census”。 |
| 改成 `all` | 加入所有 CTA 互等的全体屏障，源代码没有，且破坏各 CTA 独立进入 direct 的设计。 |
| 改成两个 launch | 增加 kernel 边界；Census、Exchange 的局部资源身份不再自然存续。 |
| 改成一个无接续说明的 phase | 可以重新建模成 worker 入口加 unit 循环，但必须仍明确 replica census、route-less 输出归属、后续 unit 分配与局部存续；单纯删 prelude 不等价。 |

本轮所有根 flow 都是单 phase launch；有效分支中的同 kernel 多 phase 只观察到上述 direct 系列的 `per_worker`。这只限制证据强度，不证明其余枚举可删。`per_worktile / per_worker / all` 是不同完成范围，`concurrent` 是不同 worker 分工，都有独立、可读的语义。建议保留四者，不为假想的 A→(B∥C)→D 增加嵌套 seq/parallel、通用 DAG 节点或多层 worker 编组。

还应把两点说得更直接：

- `order`、`start_after`、handoff 默认依赖及显式事件依赖共同成立。`ready` 提前或 `wait_at` 后移不能取消 `all`、`per_worker` 已声明的次序；有依赖环或不可实现的等待是设计不成立，不能靠文字覆盖。
- `concurrent` 表示不同 worker 承担、无默认 phase 完成先后；不承诺测得的并行，也不自动证明轮询等待能取得进展。当前 [规则 4:381](/home/shaohanyong.shy/KernelSchema/schema_discussion/flow_review/rules/4.design.work_schedule.md:381) 已要求跨 worker 的可见性与进展依据，可把这一要求直接关联到枚举表。

这两项是规则澄清，不是新字段或新 order 取值。

**默认 handoff 的三个边界**

第一，先匹配局部存储所属的 worker/副本，再做数学值的区域覆盖。MoE direct 的 Census 是每个 CTA 一份、数值相同的 scalar，见 [direct.yaml:204–215](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/direct.yaml:204) 与 [局部资源:291–316](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/direct.yaml:291)。如果只因所有生产者都“覆盖同一 scalar”而让每个消费者等全部 Census，就凭空变成 grid barrier。当前 [规则 4:297、324](/home/shaohanyong.shy/KernelSchema/schema_discussion/flow_review/rules/4.design.work_schedule.md:297) 和 [规则 5:499–509](/home/shaohanyong.shy/KernelSchema/schema_discussion/flow_review/rules/5.design.tile_pipeline.md:499) 足以容纳正确解释，但 [区域匹配:301–305](/home/shaohanyong.shy/KernelSchema/schema_discussion/flow_review/rules/4.design.work_schedule.md:301) 应显式补这一步。归类：**已有能力，匹配说明应澄清**。

第二，默认入口等待不是整 kernel 启动等待。它仅对相关 consumer 的数据使用生效；GDN 的独立 state 加载、FMHA 的 Q/TMEM 准备可以早于该值的 ready。已有 `wait_at` 足以选择真实动作时点，不能因为默认很保守就删除它。反过来，若声明的启动完成边已支配全部消费，通常无需再列相同 ready，也无需为普通入口等待重复制造 event 别名。归类：**已有能力，冗余实例可以精简**。

第三，消费者的必需入口动作可能在工作集合为空时仍执行。MoE root 的 PDL entry 明确“所有参与线程先等待前驱完成，再由 warp 0 发出后继许可，worker 无 payload 工作也执行”，见 [root.yaml:576–580](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/root.yaml:576)；fc1 入口在全局 J=0 时不分配局部存储，后续工作域为空，见 [root.yaml:689–693](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/root.yaml:689)。现行 pipeline [规则 5:511](/home/shaohanyong.shy/KernelSchema/schema_discussion/flow_review/rules/5.design.tile_pipeline.md:511) 已保留这种动作，但 flow 默认只说 worktile 入口。

建议 `wait_at` 的填写条件改为“实际消费/协议等待落在默认工作入口以外时”，不只写“可以先做独立工作、稍后才等待”。引用循环外 worker 入口事件时，约束它实际存在的实例，不能随零 worktile 数消失。需要说明入口中为确定 worktile 集合而读取的上游 metadata 也受对应 handoff 保护。这是**跨区接口需要补闭合说明**；现有 event alias 可点名这些动作，暂无证据要求新增统一 control-edge 字段。若仅在 prose 写有 entry 等待，却无法从实际 handoff、事件或协议保证恢复其先后，应作为实例缺边核对，不能把“数据工作为空”当成删掉等待的理由。

`ready / wait_at` 使用的是已有 action event 的引用，不创建另一套执行事件。`kernel_done` 属于整个 launch，同 launch 任意 phase 的同名内建事件相同，不能用它指局部 phase 完成；需局部时点时使用现有 event alias。若它指的是 consumer 的等待动作，应约束 `done`；如果指实际读，则约束 `issue`。这是保留当前 [规则 4:295、367–377](/home/shaohanyong.shy/KernelSchema/schema_discussion/flow_review/rules/4.design.work_schedule.md:295) 与 [规则 5:503–511](/home/shaohanyong.shy/KernelSchema/schema_discussion/flow_review/rules/5.design.tile_pipeline.md:503)，不是重新设计 event。

**initialize 应留在“必要初态”而非“分配过程”上**

当前规则例子 `T: {producer: solve, consumer: [apply], storage: global, initialize: 0}` 表达生产者只写三角而消费者需要完整矩阵的零值，见 [规则 4:289、311–320](/home/shaohanyong.shy/KernelSchema/schema_discussion/flow_review/rules/4.design.work_schedule.md:289)。这个事实不能随 zeros/memset 的具体 API 一起删除；但本批固定 YAML 没有使用 initialize 的正例，因此不能仅凭字段闲置认定它没有必要，也不能声称本批已经实际验证其所有使用情形。

真实反例约束的是初始化频率。FA4 decode root 的 workspace 在首次独立使用前清零，见 [root.yaml:27](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/template_fa4_decode/root.yaml:27)；空 split 不覆写 Opartial，保留此前有限 FP32 值，combine 仍以零权读取这些槽，见 [同文件:167–185](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/template_fa4_decode/root.yaml:167)。其 [handoff:213](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/template_fa4_decode/root.yaml:213) 未填 initialize，不能简单判漏填，再增加每次调用的 `initialize: 0`：那会改变实际复用与存储成本。

建议将填写条件限定为：当前交接版本缺少必要的初态保证，且该保证未由入口值或已声明的状态存续提供时，填写本次必须建立的标量初值。若初态已由上游给定，flow 不重复发明初始化；若是保留上次内容，应沿用该既有状态事实，不能用一个新标量填充替代。归类：**填写条件过宽，需要澄清；不是要求新增 lifecycle 对象**。

MoE root 还有另一种零初值：count 前每次清零 Counts/Fill，之后用原子累加，见 [root.yaml:399–410](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/root.yaml:399)。这是合并机制的必要初态，当前 [规则 4:140–142](/home/shaohanyong.shy/KernelSchema/schema_discussion/flow_review/rules/4.design.work_schedule.md:140) 已允许在实际合并 phase 的 decomposition.strategy 说明谁、何时建立；flow.initialize 不应扩成所有初始化调用的目录。可以删 “cudaMemsetAsync” 这种 API 细节而保留“每次 count 开始前把 Counts/Fill 置零”的设计约束；具体 API 如确有必要，留在 realization。

**protocol 的整体取舍：建议删除字段，保留能力和绑定**

`protocol` 是本次唯一建议整体移出 flow 的字段。理由不是它使用次数少，而是其必要信息可以逐项找到已有归属：

- 后继何时获准开始，由 `start_after` 的事件边表达。
- 数据何时可用、何处等待，由 `ready / wait_at` 表达。
- 发布和等待动作实际保证什么，由 tile_pipeline 的动作声明表达。
- 哪个平台机制兑现这些保证，由 realization 绑定；GDN 已有 [root.yaml:684–686](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml:684)，FMHA 已有 [tcg_d128_fscan_cp_direct_narrow.yaml:660–667](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/tcg_d128_fscan_cp_direct_narrow.yaml:660)。

因此只删 flow 中的协议名字、保留上述内容，不等于删除 PDL 或允许任意 kernel 提前启动。`launch_ready` 仍是针对整个前驱 grid 的明确许可事件；其产生范围、进展条件与后继 launch 所需能力必须由实际动作和实现绑定保证。不能因为删了名字就把普通提交当作能实现这种依赖。

| 方案 | 保留什么 | 丢失或增加什么 |
| --- | --- | --- |
| **主建议：从 flow 删除 protocol** | 所有 kernel 分组、启动事件、数据 ready、等待位置及实际协议绑定 | 丢失的是 flow 内就地看到 `pdl` 的提示，不丢图中的关系。读者需到 realization 查机制；规则须把 launch_ready 的含义与保证来源写清，并将“采用分离协议时填写 protocol”改成“对应关系必须有可实现的绑定”。这是明确的新规则。 |
| 备选：保留但收紧含义 | 同样的事件关系，加上当前后继 launch 的机制/能力提示 | 少一次跳转，也给结构检查一个简单入口；但保留与 realization 的重复。必须禁止将 protocol 当作隐式数据依赖或单纯 API 标签，并澄清它写在需要该早启能力的当前 launch 上。 |

如果目标是最小且职责清楚的设计语言，我选第一项。第二项没有额外的关系表达能力，保留它的理由主要是就地提示与工具检查便利，不是不可替代的设计事实。当前样本的 `pdl` 有时写在首个 count 或纯完成串行 helper 上，恰好说明名字容易混入 API 使用记录。不能为了删字段而把这些 API 名移进 thesis；应保留已有、确有必要的 realization 绑定。

[01_gdn_no_protocol.yaml](/home/shaohanyong.shy/KernelSchema/schema_discussion/flow_review/sketches/01_gdn_no_protocol.yaml) 是**新规则提案**下的 GDN flow 片段；它只从原 flow 去掉 protocol，其他关系逐项相同，并用现行 alias 缩短重复。此片段需与原有 tile_pipeline 和 realization 一起理解，不是单独可执行或完整的 schema。

**可以立刻做的精简，不用新语法**

1. 同一组值共用一条交接契约时，使用已有 YAML anchor/alias。GDN root 的 Ti、Qd、b、bc、Ai、Dl、Qc、Kc 有八条完全相同的 handoff 属性；[01_gdn_handoff.yaml](/home/shaohanyong.shy/KernelSchema/schema_discussion/flow_review/sketches/01_gdn_handoff.yaml) 保留八个数学值键和全部依赖，用同一个 alias 共享属性。它是现行语法，依据 [公共规则:64](/home/shaohanyong.shy/KernelSchema/schema_discussion/flow_review/rules/0.common.md:64)，不把八个值变成一个新 tuple，也不隐含共用存储。

2. 完成启动边已经支配消费时，不重复 ready。FMHA `tc_split_flat_narrow` 的四次提交是完整 kernel_done 链，[flow:269–280](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/tc_split_flat_narrow.yaml:269) 同时为所有 handoff 重复了对应 kernel_done。保留原启动链即可保证全部 consumer 开始前已有完成与可见性，包括 MaxTail 到 combine 的传递依赖。其 pipeline 也明确普通 MMA launch 使整链按完成串行，见 [同文件:441–445](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/tc_split_flat_narrow.yaml:441)。[01_completion_chain.yaml](/home/shaohanyong.shy/KernelSchema/schema_discussion/flow_review/sketches/01_completion_chain.yaml) 保留全部 launch、数据值、producer、consumer 和 global 存储，去掉这些重复 ready。它是现行字段/默认值下的局部等效写法；应只在支配关系可证明时这样做，不能推广到 PDL 早启链。

3. 同一个 FMHA 例子还可在**现行规则内**删 scan/combine 的纯机制性 `protocol: pdl` 标记。最大值与 split 两个前驱没有早发信号、split 自身为普通 launch，因此 helper 的 PDL API 并未打开早启机会，见 [同文件:512–515](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/tc_split_flat_narrow.yaml:512) 和 [realization:677–682](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/tc_split_flat_narrow.yaml:677)。草图保留 realization 中的必要原语绑定；移除的是 flow 对 API 属性的重复记录。与 [现行填写条件“普通的完成依赖不填”:258](/home/shaohanyong.shy/KernelSchema/schema_discussion/flow_review/rules/4.design.work_schedule.md:258) 一致。若仍按现行 protocol 条件填写，不能据此也删 GDN 的 protocol；上一节的整体删除是另一个明确的规则变更提案。

`storage: global` 的省略是另一个可讨论的精简，但我不将它列为本轮主要建议。若采用“跨 kernel handoff 缺省 global，同 kernel 必须显式选择”的规则，事实可以保留，却会增加一个依赖 launch 分组的默认分支；这是**新规则**，不能把当前强制字段默默漏掉。将多个值改成 `values: [...]` 的 handoff 记录同样是新结构，已有 alias 能减轻重复，尚无必要为此更改 handoff 的身份方式。

**十个 root 的覆盖与证据强度**

| 固定 root | flow 观察 | 对评估的作用 |
| --- | --- | --- |
| [template_gdn_prefill:150](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/template_gdn_prefill/root.yaml:150) | index_cast → gdn；Cu32 全局交接 | 完成串行的最小普通案例；m64 分支能保留启动边、删无消费的数据边。 |
| [template_fa4_prefill:143](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/template_fa4_prefill/root.yaml:143) | 一个 attention launch | 无跨 phase 事实时，flow 不需要其他壳。 |
| [template_fa4_decode:206](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/template_fa4_decode/root.yaml:206) | partial 许可后早启 combine；三项 global handoff | 启动许可不等于 partial ready；首次初始化与跨次存续不能混同。 |
| [template_qsa_prefill:151](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/template_qsa_prefill/root.yaml:151) | preprocess 完成后 attention；Union/Count | 依靠完成启动边和默认交接的简短写法。 |
| [extract_gdn:326](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml:326) | 两次 PDL 提交、八项相同属性 handoff | 三时点不可合并；重复属性适合 alias。 |
| [extract_gemm:200](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gemm/root.yaml:200) | 一个 main launch | 重试分支才增加真实提交；不能只因为宿主构图/调用次数增加 phase。 |
| [extract_mla:291](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/root.yaml:291) | 一个 main launch | 没有依据为普通单 kernel 添加通用 flow 容器。 |
| [extract_sparse:283](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/root.yaml:283) | attention 完成后 combine；三项 handoff 无显式 ready | 已有默认值能正确减少完成串行路径的重复。 |
| [extract_fmha:390](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/root.yaml:390) | 一个 main launch | 分支用来补充 split、scan 和不同 PDL 拓扑，不把根的简单拓扑外推到整棵树。 |
| [extract_moe:375](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/root.yaml:375) | 七个计算 kernel；PDL 链与跨相邻阶段的多 consumer handoff | consumer 集合、启动边和入口协议各有独立含义；direct 分支提供同 kernel 接续反例。 |

单 phase 的旧实例有 `attention/main` 等名字，本轮不把与现行 `MAIN` 命名不同计作 flow 表达缺口。FMHA、MoE 的树尚未完成；分支仅用作已捕获的特定设计证据，不据分支数量推断语言覆盖率。

对草图作了有限 CPU 核对：GDN alias 展开后与原 flow 逐项相同；删除 protocol 的提案只改变该字段；FMHA 精简片段保留全部启动边与数据两端，删除的每项 ready 都由 kernel_done 启动路径传递保证。报告中的原始证据文件与行号均已检查存在。核对记录见 `data/01_semantics.json`；这不替代完整 schema 审查或 GPU 验证。

**给主代理的明确取舍**

- 保留：launch 的 kernel 分组、四种 order、显式启动 DAG（包括 launch_ready）；handoff 的数学值身份、两端、存储、必要初态、ready 与 wait_at；phase ID 一次提交身份和已有局部 event 引用。
- 删除字段：建议整体删除 `flow.launch[].protocol`，将具体协议/早启能力留在已有 realization 绑定，并修订 launch_ready 的保证来源说明。保留但收紧是备选，不是我的主建议；上表明确了两者的成本。
- 实例精简：无作用的空壳/重复说明、被完成启动链支配的重复 ready；完全相同的多值 handoff 属性优先用现有 alias 复用。若暂不改规则，至少删除只复述 API 使用而未分离启动/数据的 protocol。
- 修改：明确所有约束取交集；匹配局部交接先认 worker/副本；wait_at 包括循环外入口且不被零工作域抹掉；initialize 不重建已经给定或存续的初态；concurrent 不保证实际同时执行。
- 暂不增加：嵌套 phase 组合树、单独 launch 身份层、通用 host/stream/graph 控制语言、multi-producer 任意数据节点、为了合并几行属性的新 handoff 分组语法。

最重要的反例分别是 GDN 的先独立准备后 wait（不能合三时点）、GEMM 的无 handoff 串行重算（不能只留数据图）、MoE direct 的同 CTA census 后多 unit（不能抹掉接续粒度/局部副本），以及 FA4 decode 的首次清零后保留有限 scratch（不能把缺失写入都改成每次 initialize）。其中前三项说明当前能力该保留，第四项说明默认与填写条件要精确；它们都没有要求增加通用控制流。
