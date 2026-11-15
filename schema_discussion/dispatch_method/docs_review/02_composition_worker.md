# 当前 dispatch 文档的组合性独立审阅

结论：可以凭当前正式 docs 开始分析源码、选择完整主线并建立设计目录。当前没有剩余必改项。本轮独立发现的实质缺口是“局部记录没有直接界定自身适用范围”；主代理补充 common:150 后，我已复读并确认闭合。

唯一关键建议是保留这次最小补充。现有“具体接口相容”“完整 YAML 不受局部说明隐式修改”和“实际交付时闭合全部输出及依赖”已经承担其余约束，不需要新增字段、组合清单或逐配对展开要求。

本报告先读 AGENTS.md、common 和 DOC3/4/5/6 的相关规范，再读 YAML 与固定版本源码；没有读 dispatch_method 既有研究报告、旧 diff、其他审阅者报告或旧 dispatch.md。旧 YAML 只用于定位真实设计事实，不以其旧字段判断当前语法。

## 审阅依据

- 最终复核的 [common](/home/shaohanyong.shy/KernelSchema/schema_docs/0.common.md:104) SHA-256：`c5151d434d0794e5b7a9fd15ab1db5be85d9e96feb91fa5e1a0a35813d916fc5`。
- [DOC3](/home/shaohanyong.shy/KernelSchema/schema_docs/3.design.math_reshape.md:5)、[DOC4](/home/shaohanyong.shy/KernelSchema/schema_docs/4.design.work_schedule.md:5)、[DOC5](/home/shaohanyong.shy/KernelSchema/schema_docs/5.design.tile_pipeline.md:5)、[DOC6](/home/shaohanyong.shy/KernelSchema/schema_docs/6.realization.md:7)：核对完整区域、参数绑定、提交、初始化、动作依赖和本地 Realization。
- FA4 固定源码 ref 为 `01fff90d714abc6e84904ea9c44523ba66d34a0e`，从 `meta_data.source` 的入口沿调用只读 `git show`。下文短名都指此 ref 下的文件：

| 短名 | 仓库内路径 |
| --- | --- |
| P-I | `kernel_wiki/aka-four-problems-20261019/fa4_prefill/aka_solution/flash_attn/cute/interface.py` |
| P-K | `kernel_wiki/aka-four-problems-20261019/fa4_prefill/aka_solution/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py` |
| D-I | `kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution/qoder_greenfield_decode_attention/operator.py` |
| D-K | `kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution/qoder_greenfield_decode_attention/gqa_decode_paged.py` |
| D-R | `kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution/qoder_greenfield_decode_attention/gqa_decode.py` |

Sparse 的 S 指 [candidate/dsv4_kernel.cu](/home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/candidate/dsv4_kernel.cu:1)。已核对工作树 blob 与 YAML 固定来源 `6e4ca91bd91d998cac99e8dadd83838e153c679d` 的 blob 相同，均为 `d0105dff58540f8839694d6e4cae2441ecdc3129`。Sparse YAML 均来自 `schema_wiki/archive/six-operators/revisions/2.5.0/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/`。

## 八项真实差异的分类演练

以下是一次小规模取样的目录判断，不声称已经为全部原实现完成重提取。

### 1. Prefill：直接写输出与 split-KV 全局归并——完整主线；split 数量——参数

直接 attention 的工作块写最终 BF16 `Oblock`；split 版本改为写 FP32 `PartialO / PartialLSE`，再由独立 combine 工作产生最终输出。这里不只是多一个 producer：部分结果的含义、跨工作分解、存储和最终归一化都必须重新贯通。

证据：`splitkv_dual_load.yaml` 的 [部分值与合并公式](/home/shaohanyong.shy/KernelSchema/schema_template/examples/fa4_prefill/splitkv_dual_load.yaml:51)、[两次提交与交接](/home/shaohanyong.shy/KernelSchema/schema_template/examples/fa4_prefill/splitkv_dual_load.yaml:91)；P-I:565 的实现说明及 1013 的 FP32 partial 分配。P-I:578 明确 split 数是运行时参数，594–603 用相同规则选择数量。

我的记录：以直接输出的一份实际设计为 root，另留一个 split-KV 完整代表。不为每个 `Ssplit` 新建节点；写明 `Ssplit` 与 chunk 范围、partial 容量、combine 的关系。common:114/116/120 能直接作出这个判断。24 MB 是此来源的流量预算，不能当作数学或硬件容量上限；同一来源的 combine 入口另有 `num_splits <= 256` 检查（P-I:3891）。本次若只核对了默认最多 64 splits，可保守保留该子域，不必证明最大可行域。

### 2. Prefill：静态 worker 与持久 worker——完整主线；小批 LPT——局部记录

静态方案每个 CTA pair 只做一份工作；持久方案让有限数量的 pair 按步长处理多份工作，还将 KV 缓冲、最终统计量及 Y 的归还延伸到相邻 worktile。`packed_persistent.yaml` 的 [worker 分配](/home/shaohanyong.shy/KernelSchema/schema_template/examples/fa4_prefill/packed_persistent.yaml:66)、连续索引 `n={flatten:[t,i]}` 和 [Y 提前归还](/home/shaohanyong.shy/KernelSchema/schema_template/examples/fa4_prefill/packed_persistent.yaml:198) 足以说明这是整体组织变化；我会保留持久方案的完整代表。

在这条主线内，“请求按较长 KV 优先排序”只改已有请求映射及其准备方式，attention 的每份工作与结果定义不变，可以一次记录 LPT 决定。`small_static_lpt.yaml` 的 [排序与发布](/home/shaohanyong.shy/KernelSchema/schema_template/examples/fa4_prefill/small_static_lpt.yaml:95) 和 `small_persistent_lpt.yaml` 的 [一次建表、后续查表](/home/shaohanyong.shy/KernelSchema/schema_template/examples/fa4_prefill/small_persistent_lpt.yaml:95) 是两种已知接入方式，不需要重新复制整份 attention。

此处确实需要局部适用范围：P-K:1071 对请求数 1/2/3/4 分别处理；1 时跳过排序，源码直接指出否则会读越界；4 才使用五次比较网，超过 4 则沿用原请求映射（1145）。具体四项比较网不能凭“基于持久方案”就推广到 5–16 个请求。一般 LPT 算法当然可以支持更多请求，限制属于本记录采用的具体网络，不是 LPT 的最大适用域。新增 common:150 正好补上这件事。

### 3. Prefill：独立 K/V producer 与联合 loader——重要局部变化

`rect_joint_load.yaml` 的 [载入和消费协议](/home/shaohanyong.shy/KernelSchema/schema_template/examples/fa4_prefill/rect_joint_load.yaml:124) 用一个 loader 按 `K0,K1,V0,K2,V1,...` 推进，共用四 token 的取得/归还序列；独立加载版本则有 K、V 两个 producer 及各自环。P-K:1375 为独立 V producer 建立自己的 pipeline，1306–1314 也分开 descriptor 的准备职责。

我会以当前完整主线的 loader 协议为基线记录一次另一种加载决定，说明 role、载荷环、发起顺序和 operand 完成后的归还怎样联动。attention 的数学、工作归属和最终输出路径沿用基线。新的 V producer 不自动触发整份 YAML，也不能只写“改成双加载”：省去独立环和完成边会误读性能与复用安全。

已有静态、持久、split-KV YAML 提供相关实际接入的证据；这些接入不各产生新的完整节点。common:115/117/122/154 与 DOC5:200 的“完整节点仍须表达其实际依赖”已足够。

### 4. Prefill：共享 alpha——一次局部记录，必须包含载荷和消费者联动

基线 `Stats` 为两列 `(old maximum, base)`，Correction 重新计算 alpha；共享版本把它改成一列 alpha，由 Softmax 计算并同时用于分母与 Correction。真实证据是 [基线 Stats](/home/shaohanyong.shy/KernelSchema/schema_template/examples/fa4_prefill/small_persistent_lpt.yaml:45) 与 [共享 Stats](/home/shaohanyong.shy/KernelSchema/schema_template/examples/fa4_prefill/small_persistent_shared_rescale.yaml:45)。P-K:3523–3578 实际写出一列或两列，3644 选择复用 alpha，3689–3755 显示消费者先等待、捕获，再归还统计槽。

我会记录 `shared_rescale`：明确相对哪个 `Stats / Softmax / Correction` 基线、该值的 FP32 含义与一列形状、一次求值的复用关系、统计槽原有发布/捕获/归还，以及首块只确认而不读载荷的行为。不会声称仅凭使用同一个函数便已证明逐位等价，也不会把它压缩成无联动的“少算一个 exp”。

静态与持久的共享版本采用相同数学区域，我对解析后的区域作相等比较，结果为相同；各自需要的跨工作生命周期由本来的主线承担。因此 common:120 给的判断确实可用，不需要两份共享-alpha 完整代表。

范围必须区分事实：P-I:966–972 用 `B=4,T=16384` 选择该优化，是源码策略；它不证明共享 alpha 只能用于这个形状。本次可直接指向已核对的两个实例限定复用结论，不把 selector 填成普遍硬界，也不先宣称所有其他 loader 配对成立。

### 5. Decode：全局 FP32 partial 合并与 cluster BF16 reduce-add——不同完整主线

全局方案保留 FP32 partial，combine 最后一次窄化到 BF16；cluster 方案在成员间交换 max/sum，每份 split 先窄化为 BF16，再按完成顺序 reduce-add 到输出。后者还需要一个成员建立输出零值、通知其他成员，然后才允许相加。

证据：[global normalize 与 handoff](/home/shaohanyong.shy/KernelSchema/schema_template/examples/fa4_decode/root.yaml:176)、[cluster normalize 与成员归属](/home/shaohanyong.shy/KernelSchema/schema_template/examples/fa4_decode/q4_cluster.yaml:83)；D-R:255/285 的 XOR 对端交换，D-K:389 的 reduce-add 搬运选择、1878–1913 的零值发布、2049 的输出窄化。

我会保留 global 与 cluster 的完整代表，不能把归并后端当作同一 YAML 中一个任意枚举开关。global 内原策略选 9/10/12/14 等 split 数（D-I:59），只要同一分解与 combine 规则、容量和精度关系成立，就属于参数变化；cluster 则另有源码接口约束“2 的幂且不超过 16”（D-R:80）。这分别落在 common:114、116、122，已有规则足够。

### 6. Decode：compact 工作 split 与逐线程 combine——成套局部变化，不能任意拆配

compact 保留 nominal grid 和 scratch 的 `SplitCount`，但按请求决定较小的 `WorkSplit`，改变 block 到活跃 CTA 的步长。空 CTA 不写 `Opartial`。相应 combine 逐线程计算统计量，只对有效 partial 读取 payload；普通 warp-parallel combine 会读取每一个 nominal payload，包括权重为零的空项，并依赖其有限初值。

证据：D-K:1077–1106 的 active 分配，518–530 将 compact 与 `warp_parallel` 选择联动；D-R:467–529 无条件读取 nominal payload，532–554 则在 `max_partial > -inf` 内才读取。[compact 的工作分解](/home/shaohanyong.shy/KernelSchema/schema_template/examples/fa4_decode/q4_compact.yaml:145) 和 [combine](/home/shaohanyong.shy/KernelSchema/schema_template/examples/fa4_decode/q4_compact.yaml:350) 与之吻合；[global 的空项条件](/home/shaohanyong.shy/KernelSchema/schema_template/examples/fa4_decode/root.yaml:179) 明写空 payload 必须有限。

我会把 compact 记录为相对 global 主线的一项联动决定：实际工作步长、空工作写入集合、消费者读取集合、统计归约分组一起说明。两次提交及 partial-to-combine 的整体组织没有重建，不因改动横跨三区就必需整份 YAML。若交付 compact 的完整 plan，再把这些定义整理进去。

这里是 common:122 的实质用处：同名同 shape 的 `Opartial` 不足以证明可组合。例如保留“空 payload 无初值要求”的 compact 说明，却借入普通 combine，其 `0 * NaN` 仍可能进入输出。已有“有效区域、精度、生产消费与完成关系”及 common:154 的初始化要求可以拒绝这种配对，不需要新增兼容矩阵。

### 7. Sparse：单查询/头块与多查询 union packing——完整代表；不借旧 split 主线拼接

packing 改变的不只是矩阵行数：一个 worker 拥有多个查询，通过带成员位的 union 共享 latent，同时保留单个查询内的重复次数；数学遍历顺序、padding、QK 部分和物化的 `UnionKey / UnionMask` 都随之改变。`tc_packed.yaml` 的 gather/online 定义与 S:2147、2200 起的 union 构造是依据。

我会保留一份 packed 完整代表，与不 packed 的主线并列。packing factor 在这条定义和已核对容量内变化，不为 GT=2/4/8 各复制一份设计。

S:59/2115 的 union 容量是 2560；`pack_gt` 在 S:4801–4805 同时检查 `GT*IndexWidth <= 2560` 和工作量，这两项分别包含容量与性能策略，不能混为同一种阈值。S:1940 的 `static_assert(GT == 1 || !SPLIT)` 及 5008–5017 的单 split 决定说明该来源的 packed 路径没有实现 packed+split。不能把独立 split 主线与 packing 的文字说明拼成已有可用组合；这不证明今后无法设计 packed+split。

common:116/122/138 足够支持以上判断。完整主线也不必拥有一个数学上的共同父节点；Contract 同属 root 即可。

### 8. Sparse：serial probe 与隐藏 probe——前者可局部，后者需要整体协作代表

serial probe 可作为已选 attention 基线的一段重要前缀记录：八份控制结果、host 观测、同流后继顺序，attention 不读取 probe 的结果张量作为 operand。`probe_serial_packed.yaml` 的 [工作组织](/home/shaohanyong.shy/KernelSchema/schema_wiki/archive/six-operators/revisions/2.5.0/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/probe_serial_packed.yaml:294) 和 S:5582–5622 支持这种概括。增加一个控制 producer 和提交本身不强制完整复制。

隐藏 probe 则改变整次调用的输出归属：先在 caller stream 提交不可撤销的 packed prefix，同时在 side stream probe；观测返回后，remainder 在 side stream 上使用所选后端，最终 join 回 caller stream。S:6155–6190 给出完整事实。被拒绝时 prefix 仍然 packed，不能把全调用改读成 mma。`probe_hidden_mma.yaml` 的 [prefix/mma 数学分界](/home/shaohanyong.shy/KernelSchema/schema_wiki/archive/six-operators/revisions/2.5.0/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/probe_hidden_mma.yaml:212) 和 [三份工作及区域归属](/home/shaohanyong.shy/KernelSchema/schema_wiki/archive/six-operators/revisions/2.5.0/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/probe_hidden_mma.yaml:347) 是闭合所需的信息。

我会选择这套协作的一份实际完整代表，倾向用 rejected remainder 的混合方案暴露两个后端的交界。accepted remainder 可描述为在同一 remainder 区域采用已知 packed 决定，说明对应参数、数学和流水线联动，不仅因接受/拒绝组合另建节点。真正要完整交付另一个选择时才整理其完整区域。

这处界线可以用 common:115/116 判断：serial prefix 能附着在已有 attention 组织上说清；隐藏版本需要同时解释 `[0,K*GT)` 与 `[K*GT,T)` 的输出生产者、独立资源、probe 完成依赖及调用末尾 join。漏掉 join 会让后续 caller 工作在 remainder 输出可见之前消费；把 prefix 错写成等待 probe 会抹掉这项优化的意义。common:154/156 与 DOC4:261/312 已经阻止这两类遗漏。

## 局部记录的基线、来源、范围和组合是否够用

我原先发现的唯一缺口不是缺少“兼容性检查”原则，而是局部记录段落没有直接要求界定本项的适用范围。此前 common:150 只要求基线、对象、变化和联动；common:138 的范围要求放在完整设计 `when` 段落里。只按局部记录段落写上述四项比较网，可能让读者把基线支持的全部请求数视作这项新记录同样支持。

现行 [common:150](/home/shaohanyong.shy/KernelSchema/schema_docs/0.common.md:150) 已明确：

> 以及适用条件与必要联动；以明确条件或所指的已核实例界定本项范围。

这一最小改动已够。记录具体网络时写 B=4，或引用已经写出 B=2/3/4 各自规则的实例；记录共享 alpha 时可以指向本次核对的静态/持久实例，而不把 `B=4,T=16384` 升格为数学硬界。后续扩大记录的适用范围，需要新的实际分析，但不需要预先列出所有未分析配对。

其他部分已有足够约束：

- **基线**：common:150 要求明确对象与基线，可以引用唯一 rewrite 名称、值或 role。`Stats` 从双列变单列必须同时引用消费它的 Correction；只写“在 root 上优化 softmax”不满足该要求。
- **来源**：common:143 的来源说明加 common:146/150 的固定源码链接可以复用同一来源，不必每条局部记录复制元数据表。此处也不能把设计出处当成性能验证（common:102）。
- **组合**：common:122 要求实质核对，现行 150 界定其结论范围。Prefill 源码用 `not is_split_kv` 关闭 persistence（P-K:200–205）表示该来源未实现这项配对；不能凭两个来源各自存在就声称配对现成，也不能因此禁止 Agent 另行设计它。
- **不成为隐式 patch**：common:152 明确记录顺序无应用含义、完整 YAML 只由本地与整区引用决定，156 又要求交付的选定组合自己补齐定义。读完局部 `shared_rescale` 不会自动把任何基线 YAML 的 `Stats` 改成一列。

没有剩余必改项，也不建议再增一条逐项列举未知组合的规则。

## when 与完整方案闭合没有冲突

common:130 的闭合单位是“该节点自身范围内的全部输出”，不是“目录覆盖共同 Contract 的每个输入”。common:136 明确允许范围重叠、目录不全覆盖。因而我可以只保留直接 attention、persistent 和 split-KV 的几份代表，在已核对范围上讨论它们，而不重建原 selector 的完整路由表。

这不允许一份 hidden-probe 代表只交代 prefix：在它自己的范围内仍需覆盖 remainder 和调用边界。也不允许 root 接住未记录的情况；这是目录缺省行为与单份计算完整性的不同约束。

实际取样中，三种事实能分开处理：

| 事实 | 实际例子 | 我会怎样记录 |
| --- | --- | --- |
| 原实现性能选择 | P-I:966–979 的 shared-rescale 形状门和独立 K/V 加载的 96 Ki 门；P-I:598 的 partial 流量预算 | 有用时在索引给简短概括和来源。可以借其限定已核子域，但不称为最大可行域或硬件必需。 |
| 容量或所选接口约束 | Sparse 的 2560-entry union；D-I:227–244 的实际 workspace 字节容量；D-R:80 的 cluster split 集合 | 写进相关参数/存储/实现条件，`when` 必要时限制承担范围；不因 selector 曾避开小容量输入就省略条件。 |
| 动态观测或状态 | Sparse 的实际观测 `R`、捕获/重放所使用的既定选择 | 明确条件的来源，再在数学/调度/实现条件中说明实际生成和就绪关系，不能把 `when` 当作读值、等待或初始化动作。 |

动态观测尤其不能用源码注释的概括代替执行事实：S:5602–5614 在有界 poll 后累加最后扫描到的正值，允许部分样本均值；它不是“所有最终样本已完成的均值”。S:5600 还明确指出 pending probe 可与后一次 gate reset 竞争。common:138/154 要求如实保留这个来源限制，不能为得到漂亮的范围文字补造等待。当前规则已够，不需要为 timeout、缓存命中、录图入口各造一个完整节点。

## 无 parent 与三个不同 from 的实际演练

### 一个可接合但不值得新建身份的三来源例子

按当前语法理解各区域内容，考察 B=4 的静态共享-alpha 方案，作以下解析演练：

```yaml
design:
  math_reshape: {from: small_persistent_shared_rescale}
  work_schedule: {from: small_static_lpt}
  tile_pipeline: {from: small_static_shared_rescale}
```

这是检验三个来源如何绑定的示意，不提议建立一个重复节点；使用节点另有身份，不能让第三条成为自引用。

这组内容可以接合的原因能够从现有材料逐项确认：

1. persistent/static shared-rescale 的数学区域解析后完全相同；`M / Bk / N / sigma` 都指当前使用节点的调度定义。
2. `small_static_lpt` 与静态 shared-rescale 的调度差别在于前者还写有 B=2/3 的排序网；在 B=4 的当前范围内，它执行同一个五比较网络，工作参数、worker 数与分配相同。见两个文件各自的 78–102 行。这里并非整区原文完全相等，而是当前使用范围内的定义接合。
3. 使用的 pipeline 为静态方案，Softmax/Correction 一致采用单列 `Stats`，不携带持久工作循环。来源设计自己的 `when` 不能覆盖使用节点范围；尤其数学来源“选 persistent”的条件不随数学区域导入。

common:173 的使用节点绑定和 175 的不携带来源 `when` 使这个判断明确。结果只是已经存在的静态共享-alpha 方案的一种内容组织，common:175 的“引用来源不同本身不构成新设计”应阻止为了这张引用图多造一个节点。

此演练用现有完整 YAML 检验引用语义，不要求精简后的目录保留所有旧节点作 provider。若某项已只保留为局部记录，就不能成为 `from` 的目标；实际交付选定组合时按 common:171 整理必要的本地区域即可。当前旧实例尚未迁移，此演练也没有将旧 `phase/repeat/...` 当作当前合法字段，没有实际生成或检查这个 YAML。

### 三个必须拒绝的接合

- **双列数学配单列消费者**：从非 shared-rescale 数学借 `Stats=(old,base)`，同时借 shared pipeline。拼写相同仍是不同载荷；common:122/173 与 DOC3:151 可以判定冲突，不能按来源顺序覆盖。
- **持久调度配只做一次的静态流水线**：当前 worker 获配一串 worktile，却只执行一次正文，将漏掉后续输出；不是把 `worker_count` 重绑定一下就自动出现 `for(t)`。DOC5:255 已要求持久 worker 的整个生命周期在最外层工作循环表达。
- **为 Sparse 混合方案从 packed 数学“只借 prefix”**：`from` 只借整区，不会自动把作用域缩成 `[0,K*GT)`，也不会引入 remainder 的 `Qm / GatherM / ResultM`。这些名称在真实混合 YAML 中分别定义（212–286 行）。若完整交付混合方案，就本地整理相关完整区域，不能让 `from` 暗中承担切片、改名或逐 step 拼接。

没有发现需要新增参数重命名或覆盖语义的漏洞。无法在当前名字、版本与接口上直接接合时，common:171 已给出本地完整定义的做法；DOC3:135/151 与 DOC5:178/183 要求名称唯一且动作实例有明确身份。节点图的跨区环也由 common:169 明文排除，不依赖 YAML 文件顺序。

root 本地拥有 Contract 和三区不妨碍这些设计独立并列；它只是实际阅读入口和共享契约归属，不强迫所有来源成为它的子孙。Realization 仍需本节点按当前有效设计填写，DOC6:7 已明说不会随 `from` 带入；三条区域引用不自动提供 tcgen05/TMA 等必需实现条件。

## 开工所需指导与剩余结论

common:110 已经提供可操作的开工顺序：先定 Contract 与本次保留的实现族，再比较端到端值、物化、工作归属、协作，最后选择 root 和完整代表。上述演练实际可以落成以下取样结构：

- Prefill 保留静态、持久、split-KV 的完整代表；LPT、加载角色调整、共享 alpha 各记录一次，写明其已核范围与接入联动。
- Decode 的本次 Q4 取样保留 global 与 cluster 两条整体方案；compact 作为 global 的成套局部变化。原 split-count 调参不逐值建节点。
- Sparse 保留本次选取的基本 attention 代表与隐藏 probe 的整体协作代表；serial probe 和另一种 remainder 选择按基线说明，不把 host 观测结果、包装入口和缓存状态全部展开成节点。

这不是要求必须采用唯一目录划分。若某个项目主要研究加载协议，选择联合加载作为完整代表也成立；关键是必须能解释所保留的整体方案，以及局部记录所依赖的具体对象和范围。现有文档没有要求新增 producer、增加等待或跨区改动就升级成完整 YAML，也没有要求展开未选择的组合。

最终分级：**无剩余必改项；局部范围的原缺口已修复并复核；其余规则在本次真实取样上足够；无另提的可选文案修改。**

未审范围：FMHA、FA4 全部 Q1 分支、全部源码选路/开发开关/故障和 graph 状态；未做完整数值证明、性能验证或依赖图穷尽验证。未运行提取、检查器或 GPU，未修改正式 docs、模板、旧 YAML 或源码，只写本报告。
