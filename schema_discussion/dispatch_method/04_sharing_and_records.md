# 内容共享、局部记录与设计来源：最小方案试写

2026-11-13，第 2 轮独立研究。先读 DOC0、DOC3–6、FA4 和 MoE 原稿形成判断，再用 [MoE 证据](01_moe_evidence.md)与[十算子证据](02_ten_operator_evidence.md)定位核查。本文没有读取同轮身份研究草稿；没有修改正式规范、模板、wiki 或工具，也没有运行提取或 GPU 验证。历史稿的 `phase/resource/verb` 等只作证据；以下候选语法和片段均是 **proposal，不是现行语法，不是检查通过的完整 schema**。

- **建议把三个问题分别处理：少建完整设计、少写重复正文、少重复解释。** 整区引用主要解决第二项；即使所有正文都能引用，原来每个组合仍有一个选择节点时，组合数也不会下降。
- **建议只增加同任务、同名整区的 `from: <design_id>`，暂不增加 step/role/rewrite 引用或 patch。** 实际收益有限但明确：FA4 两条不同父分支的 alpha 数学区可以共用；MoE 暂停批准的 96 份稿中，现有引用已用尽逐区精确相等的机会，扩大目标额外省 0 区。
- **局部变化默认写在 `dispatch.md`，一项新决定只写一次。** 用已有数学值、step、role 和资源名称定位，只补新增决定及必要联动；不强制逐条填写输入输出、版本、全部等待、成本和来源长表，不增加独立审批单位。
- **建议把设计间来源说明移到 `dispatch.md`，不再要求单父树。** YAML 的 `from` 唯一负责可展开的内容来源；MD 用一句话说明 A 提供什么、B 提供什么。root 仍保存唯一 Contract 和三个本地区域，作为完整阅读起点。
- **共享原文在消费设计中重新绑定，名称和 shape 相同都不够。** 必须仍指向同一个数学对象、数据版本和索引关系，并满足当前 worker、资源及事件条件；provider 的选择条件、元数据、修订号和 Realization 均不随引用带入。
- **短记录可以保存重要创新，但不能把耦合变成可自由组合的开关。** alpha 要联动 Stats 与两个 role；GDN 排序要联动请求身份与 PRE0；FSCAN 要配加权 combine；Sparse hidden_mma 必须保留已执行的 packed 前缀和跨流 join。
- **维护应沿真实内容及解释依赖查影响，不能把谱系边当内容边。** 共用 Contract 的修改影响整个任务；某整区的修改影响引用它的设计及其当前跨区关系。本文不承诺最少节点数、压缩比例或费用收益，提取与审批机制留待后续。

## 1. 已核证据：整区共享实际能省什么

本轮用 PyYAML 读取每个目录的 YAML，按现有 `from_parent` 逐区展开，再分别比较三个区域的解析内容。比较包含全部公式、参数和说明字符串；不清洗 thesis、不合并不同表述，忽略 YAML 映射的书写顺序。只比较 Design，不混入元数据、`when` 和 Realization。

| 样本 | 文件数 | 精确不同的 math / schedule / pipeline | 按现 parent 可省的重复区域 | 任意同任务来源可省的重复区域 | 放宽目标的额外机会 |
| --- | ---: | --- | ---: | ---: | ---: |
| [初始 MoE][i-index] | 60 | 60 / 60 / 60 | 0 | 0 | 0 |
| [暂停批准 MoE checkpoint][q-index] | 96 | 76 / 84 / 83 | 45 | 45 | 0 |
| [FA4 prefill 模板][fa-index] | 11 | 3 / 9 / 9 | 11 | 12 | 1 |

“可省”只计一份完整区域正文改为引用的机会，不计字节或语义独立性。任意来源一列按每类相等内容保留一个本地定义计数；root 始终本地。MoE 的 45 区已经是实际引用，其中 math/schedule/pipeline 各 20/12/13 区；FA4 一列则是对当前父关系的可用机会，不表示历史稿已经采用引用。

**不能从第一行推断初始 MoE 有 60 种数学算法。** 其文字和局部变化足以让每个整区不同，却仍有大量重复的 FC1/FC2 正文。也不能把初始 60 和后续 154 计划当成同范围实验：前者包含显式 FC1/FC2 配置，后者重点增加默认入口的准备、缓存和验证前缀；此处的 96 仅为一个明确批准 checkpoint 的文件集合。

三个更有辨别力的局部对照也已重新解析核对：

| 对照 | 精确相同与不同 | 对取舍的支持 |
| --- | --- | --- |
| `small_static_shared_rescale` 与 `small_persistent_shared_rescale` | 数学区完全相同；直接父分别是 `small_static_lpt`、`small_persistent_lpt`，两父的 Stats 均为两列 | 真正的跨父整区共享机会；省一份数学正文，完整节点数不变 |
| `no_scan_fused_big` → `init_no_scan_fused_big` | 原 count、scan、scatter、gather、fc1、fc2、finalize 七个 pipeline 子树全同，仅新增 init_tables；三个整区却全部不同 | 整区共享无论指向父还是兄弟，都无法消除这七段复制 |
| MoE `root` → `fused_small` | gather、fc1、fc2、finalize 四个 pipeline 子树全同；count/scan/scatter 换成 prepare；三个整区全部不同 | 新准备决定值得保存，不必随每种 FC2 重写其余链路 |

这些结果与第一轮代表对照一致，但给“放宽引用能省很多”的推断加了限制。**本轮支持放宽引用的主要理由，是解除正文复用与谱系的捆绑；大量节点的减少仍依赖目录从完整选路表转为设计记录。**

## 2. 三种共享粒度的取舍

| 方案 | 完整节点数 | 重复正文 | 阅读和维护代价 | 本轮判断 |
| --- | --- | --- | --- | --- |
| 保持直接父同名整区 `from_parent` | 单靠引用不减少 | 能利用相等的父区域；MoE 当前 45 区已受益 | 一个父链容易追踪，但复制来源受设计谱系限制 | 可继续使用的保守方案，不足以解决组合放大 |
| 允许任意同任务设计的同名整区 `from` | 单靠引用不减少 | FA4 实际多省一个 alpha 数学区；也允许三区分别来自 A/B/C | 需要看三个明确来源及当前绑定；不必为复制正文改 parent | **推荐的唯一引用扩展**；不夸大其节约幅度 |
| step/role/rewrite 片段引用，或 override/patch | 保持完整选路义务时仍不减少 | 有望复用 MoE 七个未变阶段、反复配对的 FC2 等更小内容 | 需解决对象身份、自由名称、列表插删、跨区联动、资源实例和覆盖顺序；读者要重建修改后的完整关系 | 目前不推荐；局部知识先用一次记录保存，需要完整方案时本地写完整被改区 |

更细引用的障碍不只是检查器工作量。一个 `fc2` step 并不自带 ActQ/ActS 的生产者、route map、BM 和 finalize；一个 Softmax role 也不自带 Correction 的读取含义和 Stats 视图。`rewrite` 又是有定义顺序的列表，单条公式引用需要把所用符号及下游接口一起处理。仅规定“替换同名对象”没有解决这些问题。

原有 YAML anchor 仍可复用**单文件内**完全相同的内容。它不跨文档、不创造资源或提交身份，也不是跨设计片段库。本文不额外设计一套组件参数、rename、patch 顺序或组合运算符。

## 3. 候选整区语法及边界

只增加一个名字：`from`。所在位置已经是 `design.<area>`，值也限定为设计 ID，因而不必再写 `from_design`。若采用该方案，规范只保留选定名字；不同时接受 `from`、`from_design`、`from_parent` 三套别名。迁移不是本轮工作。

下面只展示三区来自不同完整设计的**结构片段**，不证明这个示意组合成立：

```yaml
design:
  math_reshape: {from: algorithm_a}
  work_schedule: {from: schedule_b}
  tile_pipeline: {from: pipeline_c}
```

| 项目 | 确切规则 |
| --- | --- |
| 字段位置、值 | 仅用于 `design.math_reshape`、`design.work_schedule`、`design.tile_pipeline`；区域映射恰有 `from` 一个键，值是同任务目录已定义的设计 ID 字符串 |
| 引用对象 | 目标设计的**同名完整区域**；目标可再引用，最终必须落到一个本地完整定义。不能取其中某个 step、role、rewrite 或任意 YAML 路径 |
| “已定义”与作用域 | 以任务内的唯一 ID 表查找，不以文件排列或创建先后查找。目标是已有完整设计，不是 MD 记录、片段库或另一目录的同名文件 |
| 本地与引用 | 每区二选一；不能在 `from` 旁加字段，不做缺省继承或覆盖。改一部分而要交付完整 YAML 时，将该区写为完整本地定义 |
| root | root 保存唯一 Contract，三个 Design 区全部本地；其他设计省略 Contract。不会因为来源关系改成网状，就复制多份 Contract |
| 无环 | 在**节点间内容引用图**上检查：任何一个区域的 `from: B` 都产生 A→B 边，整个图须无环。A.math→B、B.pipeline→A 也不允许；本轮没有必须支持这种交叉的案例，不加跨区例外 |
| 不随引用携带 | provider 的选择或适用条件、`meta_data`、源码来源、修订号、Realization；它们由消费者自己说明。来源关系也不隐含继承 |

### 绑定发生在使用者，不能按名字猜接口

引用展开的含义是：取得 provider 的区域原文，在**消费设计**的完整上下文中理解。引用链的中间节点不会把自己的参数、条件或符号环境带入消费者。

- Contract 的引用仍解析到本目录 root。`bind:design.work_schedule::Bk` 解析到消费者定义的 Bk，不取 provider 的 Bk；同样，launch、step/worker、role、存储副本和事件都以消费者为准。
- 一个外部名称必须在消费者中有唯一、相符的定义。相符包括数学含义、数据版本和索引关系，以及有关的初态与生产消费保证；不能仅因名字、shape 或 dtype 相同就重新接线。需要改名或新接口时，先在消费者的正式定义中说清；`from` 不负责自动 rename。
- 一个完整消费者仍需贯通审阅三个区域和自己的必要 Realization。共享某区不把 provider 的适用性结论转移过来，也不能用 MD 中“沿用”补齐 YAML 缺少的设计字段。

例如，FSCAN 前后的 `PartA/PartB` 可以保留相同名称、shape 和 FP32 dtype，含义却分别是未加权和已加权 partial。把旧 combine 原文绑定到新 `PartA` 并不会自动成为正确合并。MoE 中同名 ActQ 也须与本次 ActS、路由排列、量化及 BM 对齐，不能借一个名字连接两次调用的版本。

## 4. parent 放在哪里：保留与移出的比较

现 DOC0 已明确 `parent` 不产生内容继承，所以不需要用“多继承”解决多来源。可选的是要不要继续维护一个正式的单一主要来源。

| 选择 | 好处 | 仍需维护的事实 | 取舍 |
| --- | --- | --- | --- |
| 保留 YAML 单一 `parent`，只记主要来源 | 与现行目录接近，能继续画一棵主要派生树 | Sparse 等仍须在 MD 补另一来源的贡献；没有真实派生关系的设计也要解释为何指向 root。主要来源以 YAML 为准，不能在 MD 再维护另一套 parent | 可作保守过渡，但对本轮设计目录不是必需 |
| **删除强制 parent，设计来源说明放在 MD** | 每个设计在索引中一处说明参考了谁、借了什么；容纳独立设计和多来源组合，不需挑一个“唯一真正的父” | 放弃机器结构化的单父谱系；内容引用仍由 YAML 的 `from` 精确表达 | **推荐**；不另增 `parents`、继承列表或来源组件对象 |

这里移动的是**设计与设计之间的来源说明**。`meta_data.source` 仍记录固定源码或材料的实际依据，不移到 MD，也不能用它冒充本地内容引用。

关系各有一个事实来源：

| 关系 | 唯一事实来源 | `dispatch.md` 的职责 |
| --- | --- | --- |
| 共用 Contract | 任务目录的 root 规则 | 链接 root，不复制契约 |
| 完整区域的内容依赖 | YAML 的三个 `from` | 必要时展示或链接，不另手写一份独立可编辑依赖表 |
| 设计来源和贡献 | 对应设计的 MD 索引说明 | 用“参考 A 的前缀、B 的余段”这类具体文字写一次 |
| 局部变化的定义 | 对应具名 MD 记录，或已经承载它的完整 YAML | 索引链接权威定义；已有完整 YAML 时不再在 MD 复制整段规则 |

因此，来源图可以是阅读导航，内容图则服务于展开与影响分析；不要求画成同一张树。本报告沿用现行 ID 指称设计，ID/适用条件字段如何重新命名与设计身份研究合并决定，不为共享机制新增第二套身份字段。

## 5. 局部记录的职责与写法

**一条记录说明“相对已有设计，新决定是什么，哪些已有对象必须一起改变”。** 已经清楚的数学、资源、等待和实现条件直接引用；只补本次变化引入或破坏的必要关系。篇幅短不代表只是参数微调，跨两个区域也不自动意味着一份新完整算子。

默认就在 `dispatch.md` 的完整设计索引之后，以稳定名称开一个小节，例如 `shared_alpha`、`small_prepare`。不强制另建 `variants.md`。记录多到影响入口阅读时可以拆文件，但入口只链接它，正文仍只保留一处；文件拆分不增加语义、状态或审批身份。

定位用 `design.math_reshape::Stats`、`design.tile_pipeline::attention.Correction`、`flow.handoff.perm`、`storage.ScoreProbability` 等已有名称。指向 rewrite 时优先用它定义的符号，必要时附唯一 `rewrite.name`；指向 launch 时用其中的 step 名称。**不用 `line 183`、`rewrite[3]` 或 `launch[0]` 作为变化身份。** 固定源码证据可以有行链接，但那是证据定位，不是“如何修改 schema”的地址。

这些记录没有机器 patch 语义：记录次序不表示应用次序，没有默认“其余字段继承”，也不因写了 A、B 两个来源就生成 A+B 的有效 Design。需要把某组合交成完整 plan 时，才把被改区域完整落地、核对当前上下文；记录本身的完成不以生成所有组合为条件。

避免说明行数重新相乘，靠的是记录单位，而不是把 YAML 换成 MD：

- alpha 在 static/persistent 中出现，写一个 `shared_alpha`，附这两处已有实例的链接；不各写一份变体。
- 小准备后可以接不同 FC2，先写一次共同路由接口和准备决定；只有某种 FC2 新增了额外接口要求时，才补那一处耦合。
- GDN 排序的 PRE0 限制写一次；不用为三个后端再各建“排序版”说明。
- FSCAN 与 weighted combine 是同一个必须联动的决定；不把“scan”和“combine”两条记录默认为可任意排列的开关。

如果一条说明需要读者重新猜测主要算法、输出含义或整条执行关系，就保留一个完整代表设计。反过来，不因为一个局部协议重要，就强制为它与每个基线的组合再建完整稿。是否完整交付，取决于本次要直接学习或实施什么，不按记录行数设阈值。

## 6. 实际试写

以下分别尝试整区共享、同规则特例、局部记录和多来源编排。所有片段只覆盖所讨论对象，不能作为完整 schema 执行检查。对象沿用原稿；涉及现行字段定位时用 DOC3–5 的 `step/storage/role` 名称，不沿用历史 `phase/resource` 当候选语法。

### 6.1 FA4 alpha：一份跨父数学正文，一条跨 role 记录

**已核：**两份 shared-rescale 稿的数学区完全相等，两个父节点却都采用两列 Stats。固定源码也确实由 Softmax 写一列 alpha，分母和 Correction 复用它，见[生产][fa-alpha-produce]、[分母使用][fa-alpha-den]、[Correction 读取][fa-alpha-read]。

假如仍要保留 `small_static_shared_rescale` 为完整设计，它可以采用下面的**区域片段**，不必把主要设计来源改成 persistent：

```yaml
design:
  math_reshape: {from: small_persistent_shared_rescale}
```

消费者自己的 static schedule 和一列 Stats pipeline 仍应完整本地填写；上述引用不能把父 `small_static_lpt` 的两列 pipeline 变成一列。这一例只少写一个完整数学区，不少建一个完整节点。

若目标改为少量代表设计，`dispatch.md` 可以只写一次下面的记录，链接已有 static/persistent 基线及历史 alpha 实例：

> **shared_alpha** — 在 `design.math_reshape::Stats` 与 `attention.Softmax/Correction` 之间，共享 Softmax 已算出的每行 FP32 alpha；Stats 从 `(old maximum, base)` 两列改为一列 alpha，分母和 Correction 使用同一个值。`ScoreProbability` 的 Stats 视图相应占一列 `[32,33)`。沿用基线的 Stats 捕获后复用、前一 PV 完成后修正 Y 及首块处理协议；改变的是载荷与 alpha 的生产位置。这一决定已见于 [static][fa-static-alpha] 和 [persistent][fa-persistent-alpha] 两种实例。

这里没有再次列出所有输入、shape、等待和 Realization。关键新增信息仍在：一列的含义、唯一计算位置、两处使用和资源视图；已有协议通过明确名称接回基线。允许不再交付两个完整 alpha 组合，是**记录深度的改变**，不是 `from` 自动消除节点。

### 6.2 FA4 单请求 persistent：把特例还原到同一映射

**已核：**`packed_persistent` 与 `single_persistent_dual_load` 的 math、pipeline 相同；schedule 仅两个说明字符串不同。固定源码的二分和线扫都求同一个请求前缀区间；B=1 时 `Cpad=C_0`，有效位置必落在请求 0，线扫零次。[源码映射][fa-remap]与[两份稿][fa-packed]可对照。

可以把共同 `attention.schedule.strategy` 的相关正文试写为：

```yaml
strategy: |
  Preserve input request order. Let Prefix[0]=0 and
  Prefix[b+1]=Prefix[b]+C_b. For a position x in [0,Cpad),
  positions x>=Prefix[BatchCount] have no active work.
  Otherwise choose the unique b with Prefix[b]<=x<Prefix[b+1],
  set z=x-Prefix[b], and use m=2*z when z<ceil(C_b/2),
  otherwise m=2*(C_b-1-z)+1.
  Build the immutable request and prefix map once per CTA
  before processing its assigned worktiles.
```

`C_b`、`Cpad`、worker 数、起点和固定步长沿用基线正式定义；正文没有新增调度开关。线扫或二分如何求出同一个 b，可以留作实现选择；需要交接其成本理由时再用现有 Realization 字段。

**推断与建议：**这支持把已有 B=1 和 B=5–16 的两份身份合为一条“保持请求序的 folded persistent”规则，而不是只给两个身份改引用。这里没有证明 LPT、joint loader 或 split-KV 的任意组合成立，也没有据此扩大到所有 batch/长度的适用性结论。

### 6.3 MoE 小准备：保留新的工作组织，不复制后半条链

**已核：**`fused_small` 把 count/scan/scatter 三次准备改成单 CTA 的 prepare；gather、fc1、fc2、finalize 四段 pipeline 原文不变。prepare 的局部分组是一份 worktile 内的工作，不是新增独立 worktile。其输出、写入范围和下游就绪关系见 [prepare 定义][q-small]；实际提交及容量处理见[源码][moe-small-source]。

可写一条记录：

> **small_prepare** — 将 root 的 count、scan、scatter 改为一个 CTA 的 prepare：一次暂存路由 ID，在本地分组计数、前缀扫描及分配 rank，生成相互对应的路由和 tile map。Counts、两个前缀和 RoutePos 完整写出，packed/tile map 只写有效前缀，不再需要独立的 counts/fill 初始化。下游使用同一次 prepare 产生的 Offs、RoutePos、PackedToken/Slot/Expert、TilePrefix/Expert/Row，BM 与这些 map 的解释保持一致；不要求与大准备有相同的原子到达排列。`flow.handoff` 的相关 producer/ready 改接 prepare；PDL 的 launch_ready 仍只许可早启动，读取结果要等 prepare 完成。共同 gather、FC1、FC2 与 finalize 主体仍按基线阅读。可用范围与单 CTA 容量以 [fused_small 的已有设计][q-small]为依据。

这一条记录能解释为何少两次准备提交、哪些东西由谁生成，以及为何不能只换一个 producer 名字。它不逐 FC2 重写，也不假定相同 shape 的路由表可混用。若选择交付一个具体 small+flat 完整方案，则仍须把当前 prepare 与 flat FC2 的参数、输入版本和必要实现条件落进完整 Design。

查表前缀同理：`init_tables` 的三个只依赖格式码的表、模块寿命、实际初始化提交及完成边界可以记录一次；无需随主干复制七段 pipeline。[原稿][q-init]还明确该 packed 主干不消费这些表。面向设计知识可以把它归为调用准备记录，面向该次原调用复现则不能把已提交初始化删掉。

### 6.4 GDN 排序：只补排列与请求身份的联动

**已核：**cached、direct-QK、segmented 有各自的数据通路；排序由 prep 生产 `perm`，消费的真实请求身份在准备完成后解析。源码 PRE0 明确要求 `perm_ptr==nullptr`，并有 `!kqtma`、`!l2norm`、`Gseg==1`、`vtt==16` 等其他条件。[ordered 稿][gdn-ordered]和[选择源码][gdn-source]都保留这一耦合。

可用一条短记录，不建立三份“排序版”：

> **request_order** — 由 prep 生成按 chunk 数稳定降序排列的 `perm`，chain 按 `n=perm[rank]` 选择请求，状态、chunk 槽位与输出仍按原 n 编址。用于确定请求的 perm 必须在 `chain.prep_ready` 后重新读取；先前的 metadata warming 不提供该身份。启用此排列时禁用 PRE0 的预等待首 chunk 预取，因为预等待时尚不知道实际请求。其他数据通路沿用各自基线；本记录不产生 direct-QK 与 segmented 的新组合。

这里应保留的是请求身份出现的时机，不是排序网络的每一次 compare-swap；也没有错误地宣称 chain kernel 必须等 prep 完成才可启动。

### 6.5 Sparse hidden_mma：来源相加还不等于完整组合

这是整区引用能力的一个实际边界。[`dual_dispatch_hidden` 源码][sparse-source]与[组合稿][sparse-hidden]确认：probe 与 packed 前缀并行准备选路，拒绝 packing 后才选择 mma 余段；已经跑过的 packed 前缀不能消失。

设计索引中的来源说明可以写成一处，不新增 YAML 多父对象：

> **hidden_mma** — 参考 [tc_packed][sparse-packed] 的 packed 计算作为前缀，参考 [mma_direct][sparse-mma] 的不 split mma 计算作为余段；采用 hidden probe 的 fork/join 编排。两部分沿用各自数值路径，由下面的输出区间和接续关系组成同一次调用。

其新增编排至少要在完整代表设计中说清：

| 位置 | 必须保留的关系 |
| --- | --- |
| 切分 | `K=SMCount-8>=1`；packed 的 `GT>1`；前缀 `[0,K*GT)`，余段 `[K*GT,T)`；两后端均不 split，满足源码整批次、至少两波 group 的条件 |
| fork | side stream 先等 caller 已排入的输入生产，再执行 probe；caller 在等待 host 观察结果之前提交 packed 前缀 |
| 余段 | host 实际观察结果拒绝 packing 时，用 alt mma 参数计算余段；它接在 side stream 的 probe 之后，不等 packed 前缀完成 |
| join 与身份 | 两段写不重叠的 token 区间，各自建立局部状态；caller 后续工作经 side join 看到两段输出 |

这些事实共同形成新编排，所以值得保留一个完整组合实例。不能用 `math_reshape: {from: tc_packed}` 配 `tile_pipeline: {from: mma_direct}` 来表达：三区内部都涉及两套计算及新边界，任一整区来源都不足。MD 的多来源说明负责解释借用了什么；真实计算和接口仍在完整 Design 中定义。

FSCAN 给出另一种耦合反例：split-local scan 不再生成全局 MState，由各 split 写 PartM、末 split 写 MTail，PartA/PartB 尚未加权；combine 必须据二者有限性计算 `exp(PartM-MTail)` 或零，并在消费 partial 时加权。不能保留旧的已加权 partial combine，也不能让另一个 split 在本 kernel 内读刚发布的 MTail。[提交选择][fmha-launch]、[split epilogue][fmha-split]及[权重函数][fmha-weight]明确把这些变化绑在一起。它支持一条“scan 与合并一起迁移”的决定，不支持两个自由开关。

## 7. 修改共同定义后，重新理解哪些内容

| 修改 | 必须重新理解的范围 | 不应混为同一事实 |
| --- | --- | --- |
| root Contract | 全任务完整设计和引用该契约事实的局部记录 | 并非只有“root 的后代”才受影响；合同共用由任务归属决定 |
| 某 Design 区域 | 直接、间接 `from` 消费者的有效区域，以及它在各消费者当前数学、调度、流水线和必要 Realization 中的接合 | 来源谱系的所有后代不一定引用它，非后代却可能实际引用 |
| 消费者的 Bk、worker、map、物化或资源协议 | 本设计当前引入区域的重新绑定及有关接口；即使 provider 文件未变也要核对 | “provider 未改”不等于该引用仍适用 |
| 某局部记录所引用的基线对象 | 依靠该对象解释差额的记录，以及明确使用这项变化的说明 | 只写历史启发来源的链接不自动成为内容依赖 |
| 纯设计来源说明 | 该说明及其展示 | 不改变 YAML 有效 Design，不触发隐式内容继承 |

局部记录的基线链接有时是**解释依赖**：例如 shared_alpha 不重写 Stats 复用协议，就依赖基线仍提供那份协议。不能将其当无关的谱系链接而永不回看，也不必由此给每条笔记建立另一套机器展开身份。记录写明涉及对象，变更时就能有针对性地复核。

原稿修订号继续按本文件实际内容变化递增。provider 改变而消费者原稿未变时，不伪造消费者修订号增长；有效内容及来源身份可能已经变化，这一点沿用 DOC0 的现有区分。这里提出的是语义影响范围，**不改变现行审批仍保守绑定祖先的规则**。

关系索引只展示前述事实来源。若将来生成引用列表或图，应从 `from` 得出，不要求作者再维护一张可独立漂移的引用清单；来源贡献和局部记录则各留一处正文。解析器如何展开、如何定位 provider、如何保存有效内容身份，以及审批、缓存、重审与生成视图的细节均留 TODO。

## 8. 最小推荐与 DOC0–6 改动表

最小推荐由三件事构成：**完整设计仍用现有三区结构；整区共享改为同任务 `from`；非完整的局部变化与设计来源用一次 MD 记录。** 不建组件库，不给笔记加 patch 语义，也不把所有原选路组合再登记成“轻量节点”。目录是否继续承担互斥穷举，应与设计身份研究一并调整；否则共享只会让原组合表的文件更短。

| 文档 | 字段建议 | 规则建议与不变部分 |
| --- | --- | --- |
| DOC0 公共约定 | 三个 Design 区域的引用形式改为唯一 `from: <id>`；推荐删除 `meta_data.dispatch.parent` 的强制单父含义及字段，设计来源移到 MD；ID/适用字段与身份研究合并决定 | 明确同目录、同名整区、root 三区本地、内容图无环、使用者绑定、不继承条件/元数据/Realization。`dispatch.md` 从必画派生树改为完整设计入口与具名局部记录；不再由局部记录产生完整节点或覆盖承诺 |
| DOC1 算法 Contract | 不增字段 | 仍只在 root 保存一次；各设计、记录共同依靠同一契约，不复制或由局部变化收窄；对象名称与数学定义仍唯一 |
| DOC2 硬件 Contract | 不增字段 | 同任务共用 root 的硬件与实现环境；来源说明或引用不得偷换硬件契约 |
| DOC3 数学改写 | 顶层整区引用改用 `from`；不增 rewrite 片段引用、patch 或通用 component ID | 本地完整 `thesis/rewrite`、符号定义、retention 和精度规则不变；用已有符号或唯一 rewrite 名称定位局部记录 |
| DOC4 工作调度 | 顶层整区引用改用 `from`；不增 step override | 完整设计内仍如实定义实际 launch、step、handoff、工作实例和就绪关系；“新增一次准备只需局部记录”是交付层级选择，不是在完整 YAML 内用正文取消提交 |
| DOC5 微流水线 | 顶层整区引用改用 `from`；不增 role/resource patch | 现有 role、storage/view、动作和 waiting 规则不变；记录指出改变的对象及必要联动，完整实例仍须资源和事件闭合 |
| DOC6 Realization | 不增继承或笔记兜底字段 | 仍为本节点必要交接；没有新增条件可省略。引入区与消费者组合后的原语要求须独立判断，父/来源设计的反馈和证据不自动适用 |

**尚未证明的部分：**此处试写只验证了记录能保留哪些新增决定，以及引用能节省哪些精确相同正文；未证明全部历史组合可达、数值正确或能自由拼接。没有把“同 shape”当接口证明，没有给 FMHA/MoE 硬凑小于十的完整设计数，也没有用正文行数推算调用或费用比例。

### 核查入口

计数与比较的原始范围固定为下列三个目录。Q 的 96 个文件为同一 checkpoint，不包含 9 个其他候选、失败 seed 或未开始条目。两份第一轮报告用于定位证据；本轮重新解析了表中的精确相等结果，并阅读上述 FA4、MoE、GDN、Sparse、FMHA 的相关源码与对象定义。

[i-index]: ../../schema_wiki/archive/six-operators/initial/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/dispatch.md
[q-index]: ../../.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/dispatch.md
[fa-index]: ../../schema_template/examples/fa4_prefill/dispatch.md
[fa-static-alpha]: ../../schema_template/examples/fa4_prefill/small_static_shared_rescale.yaml
[fa-persistent-alpha]: ../../schema_template/examples/fa4_prefill/small_persistent_shared_rescale.yaml
[fa-packed]: ../../schema_template/examples/fa4_prefill/packed_persistent.yaml
[fa-remap]: https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_prefill/aka_solution/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py#L1213
[fa-alpha-produce]: https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_prefill/aka_solution/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py#L3523
[fa-alpha-den]: https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_prefill/aka_solution/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py#L3644
[fa-alpha-read]: https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_prefill/aka_solution/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py#L3689
[q-small]: ../../.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/fused_small.yaml
[q-init]: ../../.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/init_no_scan_fused_big.yaml
[moe-small-source]: ../../kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu
[gdn-ordered]: ../../schema_wiki/archive/six-operators/revisions/2.2.7/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/cached_ordered.yaml
[gdn-source]: ../../kernel_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/candidate/gdn_kernel.cu
[sparse-source]: ../../kernel_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/candidate/dsv4_kernel.cu
[sparse-hidden]: ../../schema_wiki/archive/six-operators/revisions/2.5.0/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/probe_hidden_mma.yaml
[sparse-packed]: ../../schema_wiki/archive/six-operators/revisions/2.5.0/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/tc_packed.yaml
[sparse-mma]: ../../schema_wiki/archive/six-operators/revisions/2.5.0/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/mma_direct.yaml
[fmha-launch]: ../../kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu
[fmha-split]: ../../kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_tcg.cuh
[fmha-weight]: ../../kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_common.cuh
