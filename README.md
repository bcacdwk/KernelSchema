# KernelSchema

本仓库设计一套描述 GPU kernel **流水线设计**的 schema：用结构化的 YAML 记录一个 kernel 要算什么、数学上怎么改写、工作怎么分配到硬件、tile 内部怎么流水，以及实现时必须遵守的条件。schema 位于所有 GPU 编程语言之上，由 Designer 填写、Coder 实现、人类专家审阅。

仓库记录 schema 各区的字段规则与样例。**当前进度：六个区的字段规则文档（`schema_docs/1–6`）已经写成，正在逐区完善和优化。**

## 0 术语

- **Designer**（原文也称 planner）：负责设计的 Agent 或人，唯一能修改 Design 区。
- **Coder**：把设计实现成代码的 Agent（例如 AKA），实现语言不限。
- **schema**：既指这套描述语言本身（分区与字段规则），也指针对某个 kernel 填好的 YAML 实例；后者也叫 plan。
- **template**：schema 的母版，定义有哪些字段、哪些必填、可选哪些词汇。计划用 CUE 编写，尚未开始；目前由 `schema_docs/` 的字段规则文档说明。
- **wiki / microbench**：完整的硬件资料、指令手册，以及吞吐与延时的实测结果。schema 只引用，不抄录。

## 1 背景与动机

### 1.1 Kernel Agent 的两个普遍问题

1. **难以优化**：无法判断结果是否接近最优。长程任务高度依赖 LLM 自身的能力，很难突破平台期，也不知道什么时候该停止迭代（需要在 token 消耗与 latency 下降之间取舍）。
2. **难以维护**：全部交给 Agent 后，人类专家无法观察、判断和介入，结果是纯黑盒，后期的生产、解释和迁移都很困难（也影响社区影响力，难以沉淀成可学习的 wiki）。

### 1.2 思路：把“策略意图”和“代码实现”分开

**策略意图**本质上是数学问题：在给定的算法要求下，怎样最好地安排硬件资源、吞吐、内存和计算顺序。这是纯“mental”的高层问题，不需要任何具体的 GPU 语言，用 YAML 这类结构化文本就能自然表达。

**代码实现**涉及大量工程细节：CuTe、CuTeDSL、CUDA、Triton 等语言，编译、精度验证、速度比较，以及数据 layout、语义接口约定、版本对齐等问题。

两者完全不是同一类问题。混在一起时，Agent 每提出一个新策略（例如新的 QK 计算顺序），就要推倒全部代码重来。这让 Agent 望而却步，也让它过早收敛到局部最优：代码太容易写错，负反馈太多，奖励太少。而策略意图恰恰是人类专家最擅长感知和判断、也最容易介入决策的“taste”问题。

好的流水线是 SOTA kernel 的必要条件。结构性很差的空间—时间—存储映射，不可能靠漂亮的 CuTe 模板、语法技巧或局部指令调整变成 SOTA。代码实现只能逼近一个设计的性能潜力，不能凭空创造不存在的数据复用、并行度和流水重叠。

### 1.3 核心科学问题与目标

> **是否存在一个比源代码明显更紧凑、更可解释，同时又保留绝大多数一阶性能因果关系的 Pipeline IR？**

换句话说，这个中间表示能否成为性能的某种“近似充分统计量”：

- 同一流水线下，不同合格实现的性能差异相对可控；
- 不同流水线之间的性能差异，显著大于同一流水线内部的实现差异；
- 根据流水线表示，就能大致判断候选方案的相对优劣；
- 代码可以被验证为该流水线的一个 refinement，而不是只看最终速度去猜它的实现。

本仓库的 schema 就是这个 Pipeline IR 的候选，阶段性目标是：

1. 完备、唯一地表达现在主流的 kernel：FA3/4、MoE、GEMM、MLA、GDN、KDA 等；
2. 人类能比较容易地读懂，并据此确定流水线；
3. 词汇尽可能精简，用尽可能少的词汇族表达整个流水线，不膨胀成真正的 DSL。

## 2 整体方案

### 2.1 从零设计一个 kernel 的步骤

| 步骤 | 内容 | 对应位置 |
| --- | --- | --- |
| 1 硬件 | 硬件 spec、执行单元与层级、基本 primitive、功能支持、指令集资料（如 PTX/SASS）；计算、访存、指令发射硬件在典型 tile size 下的吞吐与延时 microbench。可以先覆盖常用 primitive，再按迭代需要补测；每种卡的 spec 固定不变 | `contract.hardware_specification`；数值细节在 wiki / microbench |
| 2 算法 | 数学公式（如 SDPA）、PyTorch 算法（如 online softmax）、输入输出 shape 与精度规格；在不改变输入输出的前提下改写计算顺序与计算方法。这一步偏纯数学、算法层面的形式化验证与迭代，可能可以完全交给 AI；评估看各级存储（RMEM/SMEM/GMEM）的访存总量与 Tensor Core / CUDA Core 的计算总量 | `contract.algorithm_requirement`、`design.math_reshape` |
| 3 瓶颈 | 数据依赖图、roofline 估算、瓶颈判断，给定算法与硬件即可推导 | 估算器；`math_reshape` 中定量记录改写带来的计算量与访存量变化 |
| 4 流水线设计与搜索 | SM/Grid/CTA 任务划分、tile size、warp specialization 分工、RMEM/TMEM/SMEM 分配、barrier 与计算顺序、数据流转。这是最关键的设计：描述要完备、唯一，能近似评估性能上限，最好能快速生成可上机验证的 prototype。pipeline 设计天然有树状分叉，适合用 evolve 式的遗传策略维护谱系，甚至直接广搜、深搜 | `design.work_schedule`、`design.tile_pipeline`，Designer 负责 |
| 5 流水线实现 | 用 CUDA、CuTeDSL 等实现；精度 bench 保证正确，NCU、IKET 等 trace 工具确认实现符合设计，按任务 workload 评估端到端性能。与步骤 4 完全解耦，是经典的 kernel agent 任务：在固定的设计下写出最优的底层代码，适合 AKA 多轮迭代 | Coder |
| 6 反馈 | 分成两支：一支几乎不改顶层设计，只做 buffer 深度这类简单修改，继续演进直到满足停止条件；另一支总结这种设计的问题与阻力，反馈给 Designer，指导下一轮 pipeline evolution | `realization` 中的实验反馈，或回到 Design 区 |

![整体流程：硬件 spec 收集 → 数学算法 reshape → bottleneck 分析 → Planner 分析现有流水线并搜索、更新出多条候选流水线 → Coder（基于 AKA）为每条流水线迭代多个代码版本 → Verifier 把反馈送回 Planner](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/a/RNJroOxr3UGLN01e/33a775e0fcf74b40b9b61c2d32f795ea0521.png)

图中 Planner 一侧标注的是“通过 CAKE 表示”，本方案与 CAKE 的关系见 2.4。

### 2.2 角色

- **Designer** 在 Contract 的基础上做设计，只有它能修改 Design 区。
- **Coder** 在明确的硬件、算法任务和设计下，遵照 wiki 与 microbench 的信息逐代迭代实现。实现语言任意，可以是 TileLang、CuTeDSL、CUDA 等暴露 warp-spec 级细粒度的语言。Coder 不能修改 Contract 与 Design，只能在 realization 中附带有实验证据的反馈。
- **人类专家**随时介入、把握设计方向，只需要审阅 YAML 形式的 plan，不需要介入代码实现。预期的协作方式介于 vibe coding 与全自动 loop 之间。

### 2.3 Designer 的工具与 Coder 对称

Coder 和 Designer 是对称的，关键是怎样减轻 Designer 的工具负担：

| Coder 的工具 | 回答的问题 | Designer 的镜像工具 | 回答的问题 |
| --- | --- | --- | --- |
| CUDA/CuTeDSL 源码 | 实现是什么 | 设计 schema 实例：结构化声明表 + 受限伪代码，不可执行 | 设计是什么 |
| NVCC/ptxas | 编译能不能过 | 检查器（准入检查器组）：资源记账 + 合法性规则 + 活性检查；另一种分法是资源算术（A 档）+ 流水线模式库（B 档）+ verifier agent（C 档） | 设计可不可实现 |
| 精度测试 | 算得对不对 | License 引用审计 + metamorphic 测试生成 | 有没有越权重排 |
| NCU | 实际跑成什么样 | 估算器（双层评估器）：静态上界 + 离散事件仿真；另一种描述是 II / 稳态瓶颈分析 + 硬件常数表 + microbench 校准回路 | 理论上该跑成什么样 |
| — | — | Conformance 通道：NCU 签名 ↔ 设计预期 | 两边共享的桥 |

表格转写自原文档的两张图（[图一](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/4c70edf6-e134-48e4-b52f-e6f8c7d67d8a.png)、[图二](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/08398da4-d82a-4a15-8eeb-fda39b3fbc84.png)）。对三类工具的要求：

- **schema** 尽可能轻量：用尽可能少、相互正交的原语刻画现有的所有 kernel，以“结构化表格 + 不可执行伪代码”完备地描述所有可行的设计思路。
- **检查器**在不运行代码的前提下做静态校验，由确定性的 Python 与 Agent 混合完成。检查偏乐观、偏宽松：可以放过差的设计，但不能漏掉好的设计。
- **评估器**从多个维度快速估算。不要求准确，但要尽量保序，即“设计 A > 设计 B > 设计 C”的判断近似正确。

### 2.4 与 CAKE 的关系

[CAKE](https://github.com/huangzhilin-hzl/cake-ir)（[论文](https://arxiv.org/html/2608.12629v1)）之所以是大体量工程，是因为它做了确定性的 lowering/codegen 和全架构后端。本方案刻意把这些全部交给 Coder（自带 NVCC + NCU 反馈），去掉 codegen 的后半部分；剩下的检查器和估算器只是 Python + 一套 microbench + 一张硬件表，仍然停留在“Agent Harness 工程”的范畴内，不会变成开发 AI 编译器。

另一个本质差别是 ground truth。CAKE 的 IR 有可执行的 ground truth：每个移植的 kernel 都必须编译、跑对、复现原 kernel 的 physical schedule，抽象错了当场暴露。本方案没有这种 ground truth，“schema 是否正确刻画了这个 kernel 的流水线”只能靠 Agent 或专家读代码判断，事实性错误可能安静地沉淀进语料。对策是盲重建审计（见 6.3）。

## 3 Schema 的分区

### 3.1 总览

| 区 | 子区 | 记录什么 | 修改权限 |
| --- | --- | --- | --- |
| Contract | `contract.algorithm_requirement` | 算法要求：从外部看，算子必须满足的输入、输出与数学关系 | Designer 与 Coder 都不能改 |
| Contract | `contract.hardware_specification` | 硬件规格：目标设备的执行单元、核心能力与边界，本任务已选的实现栈，以及完整规格与 microbench 的引用 | 同上 |
| Design | `design.math_reshape` | 数学改写：为了在该硬件上实现，对计算本身做的改写 | 只有 Designer 能改，Coder 必须遵守 |
| Design | `design.work_schedule` | 工作调度：计算如何分解、由谁执行、怎样分配 | 同上 |
| Design | `design.tile_pipeline` | tile 流水线：一份工作内部 warp / warp group 的分工、顺序与依赖 | 同上 |
| Realization | `realization` | 实现交接：设计所依赖、但上面没有表达的实现条件，以及优先尝试的实现选择 | Designer 填写；Coder 只能附带实验反馈 |

六个子区依次对应 `schema_docs/1–6`。Design 三区的关系：

- **math_reshape 修改、优化要实现的任务本身；work_schedule 分治任务，规划子任务的流程以还原完整任务；tile_pipeline 确定子任务的具体细节与实现策略。**
- 三区从上到下排列：越靠上越难优化、收获越大，在源码中也越难看出来；越往下与代码绑定越深。
- math_reshape 是下游的前置条件：只有它定下来，下游才知道有哪些量、计算的先后顺序、哪些量跨迭代携带、哪些中间量写回全局内存，才能开始分配工作。
- work_schedule 与 tile_pipeline 需要交替协同：先定 macro，再定 micro，再改 macro，最后定 micro。流水线层面的发现可能反过来改变任务划分，例如 FA4 为了打满瓶颈环节，凑出 Q_high、Q_low 两个 tile，形成两个 softmax warp group。
- 三区顶部各有一个 `thesis`，用最简单的语言说明这一段为什么这样设计，尽量简短（可以设字数上限）。

分区经过三轮讨论才定下来：最初 Design 只有 macro tiling 与 micro pipeline 两层，math reshape 放在 Contract 区，另设一个 Designer 与 Coder 交织的“缓冲区”。后来 math reshape 移入 Design，macro tiling 演变为 work_schedule，micro pipeline 演变为 tile_pipeline，缓冲区演变为 realization。

### 3.2 各区要点

各区的字段、语法与案例以 `schema_docs/` 中的文档为准，这里只说明定位与边界。

**contract.algorithm_requirement（算法要求）**

- 从调用者的视角把算子当作黑盒：接收什么、返回什么、输入与输出之间必须满足什么数学关系。内容包括数学定义与记号、特殊语义、输入输出与内部变量的命名和含义、golden reference（固定版本的参考实现入口及各输出的误差判据）、代表性 shape 与 workload，以及优化目标（后两项的字段格式尚未确定）。
- 完备性标准：不看 Design，能否仅凭这一段写出一个朴素 reference，而不需要猜掩码、量化、路由和状态语义。
- 只写“要求算什么”，不写“怎样改写”：因果可见性、MoE 的量化函数、GDN 的状态递推属于这里；online softmax、按专家分组、块内三角求解属于 math_reshape。量化组长决定数值含义，不能当作普通的 tile size 调整。
- 计算定义里出现的中间量只用来定义数学关系，不要求实现物化。契约的边界来自选定的计算任务，不照搬现有 kernel 的拆分方式。

**contract.hardware_specification（硬件规格）**

- 记录目标设备的身份、执行单元与层级、会改变设计空间的核心机制及其边界，本任务已选的实现栈，以及完整规格与 microbench 的固定版本引用。执行单元是下游 worker 与 lane 的词汇来源：CTA、warp 只是 NVIDIA 上的实例，其他设备另有自己的单元。
- 要精简，不能变成 datasheet：SM 数量、资源容量、吞吐、延时等数值放在 wiki / microbench，这里只引用（不抄表，但对应的约束仍要检查）。机制的入选判据是删除测试——缺了这一条，读者会不会做出不可实现的设计，或错误地排除一个可行的设计。“本代新增”既不是入选理由，也不是排除理由；关键缺项可以显式写明不支持，用来阻止跨设备的默认继承。
- 设备事实独立于 kernel：同一设备、同一资料版本只对应一份身份、执行模型与机制摘要，从公共硬件资料中提炼，而不是从某个 kernel 用了哪些指令反推。最初的设想是记录“在这张卡上实现这个算子必须考虑的硬件特性”，后来改为与 kernel 无关的设备摘要。实现栈与资料绑定则按任务固定，换栈不改变设备事实。
- 只陈述条件，不替 Design 做选择：“支持 CTA-pair MMA”属于这里，“采用双 CTA”不属于。软件后端暂时无法表达某项能力，不等于硬件不支持。具体的编译目标参数属于实现选择，需要固定时写进 `realization` 的 `require`。

**design.math_reshape（数学改写）**

- 在不违反 Contract 的前提下改写要实现的任务本身：决定算出哪些值、这些值怎样定义、以什么精度存在、存在多久。它是从两份 Contract 过渡到下游设计的一层，做的是算法与硬件的 co-design：动机来自硬件，改写本身是纯数学的，能用接近数学公式的语言写出来，与流水线无关。
- 常见手段是交换计算顺序、分块、融合和改变表示，以降低理论上的时间、空间复杂度：用固定大小的状态代替随输入长度增长的中间量（online softmax）；避免物化中间矩阵和重复的扫描、搬运；把串行递推折叠成矩阵乘（GDN）；把零散的计算聚成能复用数据的稠密块（MoE 按专家分组）；换用更便宜的运算单元（把 CUDA core 的工作拼成 Tensor Core 的工作，把部分 exp2 从 MUFU 挪到 FMA）。
- 收益与代价都要定量记录：新增、消除或转换了哪些计算、访存、片上状态、串行深度、遍历趟数和误差。只说“转换后更快”不够。不要默认继承下来的算子边界、数据表示、kernel 拆分与融合方式不可改变，转换成本与 launch 成本也要算进整体目标。
- 精度是改写的一部分：可以接受计算次序、舍入点、累加精度这类有限精度差异，以及给出误差界的主动近似；不能为了加速省掉会改变结果的计算，也不能放宽 Contract 的精度要求。
- 这一层在源码中最难看出来（FA4 的滞后最大值在源码里只是一行阈值判断），往往要读论文或原作者的逻辑才能识别，但收益最高，也是遇到新的融合算子时 Agent 首先应该思考的空间。
- 曾考虑按 representation、preprocessing、decomposition、algorithm、schedule、dispatch 等类别，或按改写路线组织这一层。这些类别不完备也不正交，会逼着模型用不恰当的方式表达合理的设计，所以不设分类字段。

**design.work_schedule（工作调度）**

- 数学对象与硬件资源之间的映射、分配与调度，分三部分：decomposition 写计算怎样分解成可独立分配的工作单元 worktile，以及跨 worktile 的合并；mapping 写 worktile 由什么样的 worker 执行（CTA、CTA pair、cluster 等）、需要多少；schedule 写 worktile 怎样排列，以及 worker 怎样获得、接续工作、何时退出（静态分配、persistent、动态领取、负载均衡、任务窃取、分桶等）。
- 思考方式贴近 Triton / TileLang：按 tile 管理计算对象，按 CTA 分发（不细到 warp），同时考虑 wave、SM 数量、尾波负载均衡与 SM 资源平衡。这是最重要的 co-design 之一。
- 核心区分：worktile 的排列不等于 worker 的工作归属；预先获配的工作不等于运行时新领取的工作。
- `persistent`、`CLC` 等策略名可以帮助人类理解，但有约束力的是具体的决策。这一区不描述线程级实现，也不重复 tile_pipeline 的动作、依赖与缓冲协议。

**design.tile_pipeline（tile 流水线）**

- 一个 worktile、一个 worker 之内的微流水：warp / warp group 的角色划分（warp specialization），以及各角色在 prologue（填入）、steady（重复推进的最小单元，即 pace）、epilogue（排出）中的动作与依赖。只细到 warp 级，不深入 thread 访问的元素和 layout。
- 目标是在不死锁、不产生读写竞争的前提下，通过 warp 分工、异步逻辑与排布顺序，最大化计算与访存的重叠。一个 pace 的延时就是流水线的平均延时，设计围绕 microbench 给出的最慢环节 max(t1, t2, …) 展开，类似一张 Gantt 图；例如 FA4 在 SM100 上围绕 SFU 偏慢来设计。
- 三个关键语义：发射顺序不等于完成顺序，异步动作要区分 issue 与 done；除了“数据就绪”，还要记录“buffer 可以再次写入”的复用约束；循环体只写一次，跨迭代的依赖用迭代索引表达，展开后才是 DAG。
- 写法上曾参考 OSDI 2026 的 *Optimal Software Pipelining and Warp Specialization for Tensor Core GPUs*（Twill）：它便于分析，但人和 AI 都很难迅速抓住核心，最终改为按 lane（一个 warp 或 warp group）记录角色、动作序列与依赖。这种 YAML 可以用 Python 展开并检查（动作归属、重复 ID、缺失引用、有向环），再用 Graphviz 渲染成按 lane 分组的依赖图；依赖图不是时间线，节点间距不代表耗时。

**realization（实现交接）**

- Designer 与 Coder 之间的交接区，只记录 Contract 和 Design 尚未表达、但双方需要共享的实现决定。它不是第四个 Design 区，也不是源码剩余细节的收纳区。
- 按约束强度分三类：
  - `require`：既定设计成立所必需的硬条件，类似 lowering hint，尽量简短，Coder 不得替换。例如 Blackwell 上必须用 `tcgen05.mma`，不能用 `mma.sync` 或 `wgmma`；Hopper 及以后要用 TMA 搬运。这里记录的东西应当影响上面的设计，而不是被设计所影响。
  - `prefer`：与既定设计的代价直接相关的优先选择，能实现就优先实现。Coder 可以用实验证据支持替代方案，这是 Coder 唯一的反馈位点。
  - `forbid`：Designer 考虑过并明确排除的路径，即实现起来简单、但会走弯路白费力气的方案。尽量留空。
- 替换测试：如果 Coder 换掉这一选择，Design 的核心设计是否仍然成立？成立就留给 Coder，不成立才写进 realization。
- 这一区越长，说明设计越差：好的设计应当容易确定，而不是遗留大量细节，评估时篇幅越长扣分越多。反馈也不是修改上游设计的入口：替代方案需要移动舍入点、增加搬运、改变角色或 buffer 生命周期时，回到对应的 Design 区讨论；`require` 无法实现时报告阻碍，不能自行降为 `prefer`。

### 3.3 一个决定写在哪里

| 这条信息是什么 | 写在哪里 |
| --- | --- |
| 任务名称、设计出处与这份 plan 的修订号 | `meta_data`，位于 Plan 最前面，字段规则见 `schema_docs/1`，不属于 Contract |
| 算子从外部看必须满足的输入、输出与数学关系，包括掩码、量化、路由、状态与边界情况，以及验收的误差判据 | `contract.algorithm_requirement` |
| 目标设备有哪些执行单元、什么能力、边界在哪里，本任务用哪个实现栈 | `contract.hardware_specification`，数值规格在 wiki / microbench |
| 算出哪些值、如何定义、以什么精度存在、存在多久 | `design.math_reshape` |
| 计算怎样分解成 worktile、由什么 worker 执行、怎样分配与接续、怎样跨 worktile 合并 | `design.work_schedule` |
| worktile 内 warp / warp group 的分工、动作顺序、依赖与 buffer 复用 | `design.tile_pipeline` |
| 设计成立所依赖、但上面没有表达的实现条件（如必须使用的原语），以及优先尝试的实现选择 | `realization` |
| 只替换实现、数值差异在 fp32 ulp 以内的选择；代码组织、地址计算、barrier 的类型、编号与 phase、精确的 layout | 留给 Coder，不写进 schema（除非替换测试表明换掉后设计不再成立） |

## 4 编写原则

1. **奥卡姆剃刀，less is more。** 不必要的就删掉，不新增冗余的假设和文字；用最少的语言描述 kernel 的 insight 是最核心的需求。判断标准：删掉它，读者会不会读错或漏掉设计？不会就删。
2. **尽可能提炼信息熵。** 能讲清 kernel 设计的核心思路、内容与方法即可；不写一连串看起来有用、实际上没人看、也无法审计的伪代码或说明文字。
3. **一个决定只写一次，写在唯一合适的位置**（见 3.3）。下游直接引用，不重新解释，也不改变含义。
4. **不为格式对称填空壳。** 没有内容的字段和区域直接省略，不设 `notes` 式的兜底区域。
5. **定量、可核对。** 收益、代价和理由写成能核对的事实，不写“提升性能”“优化访存”这类口号；好坏的判断留给评审与实验。
6. **名称跨区一致。** 维度、张量、状态的名称属于当前算法实例，不是模板字段；同一个对象在各区用同一个名字。数学记号、精度、shape 与空值等共用写法定义在 algorithm_requirement，下游区直接引用，只补充本区特有的部分。
7. **对照真实源码的单一分支填写。** 来源固定到 `仓库@commit:路径`；不同分支不拼成一份，源码中每一处偏离精确计算的地方都要能在 schema 中找到。
8. **只陈述条件，不替下游做决定。** 每一层只固定本层必须固定的量，其余注明由哪一层决定。
9. **有约束力的是具体决策，不是名称。** `persistent`、`CLC`、`online softmax` 这类名字只帮助理解，不能代替具体规则。
10. **抓大放小。** Designer 必需的内容不能遗漏、不能出错；属于 Coder 的细节移出 schema；确实模糊、可以由 Coder 决定的内容放进 realization 的 `prefer`，并尽量减少。

**篇幅锚点。** 一份 plan 应在 200–300 行以内，人类能读懂。参照 CAKE 论文的 Table 4：FA4 forward 的 CUDA device 代码约 2369 行，Cake IR 表达为 430 行，可执行的 schedule 表示压缩了约 5 倍。schema 去掉操作流之后应当再小几倍，一个 SOTA attention kernel 的设计骨架大约在 50–150 行；写到 400 行，说明操作流细节漏了进来，应下放给 Coder。

## 5 文件格式：schema 用 YAML，template 用 CUE

### 5.1 schema 与 template 的关系

schema 和 template 要同时服务 Agent 和人类专家，所以要格式规整、易读、易维护：Agent 读写轻松，也便于用正则、匹配和逻辑操作批量处理 schema。

两者的关系像填表：template 定义表格的形状、表头文字和可选的填表词汇，类似幻灯片的母版；schema 是按 template 填好的表，直接关联 template，template 改动时 schema 随之调整，Agent 在填写 schema 时也不能随意修改表头。文件之间的映射关系有点像一个数据库系统。

**数据层要笨，智能放在校验层。** schema 文件越接近纯数据越好，类型、枚举、约束、母版关系都放在 template 和校验脚本里。限制 Agent 自由发挥，靠的不是文件格式本身，而是“改完必须通过校验”。

### 5.2 schema：YAML

YAML 的层级结构天然适合“表格 + 嵌套小节”；块标量 `|` 能容纳短文本和伪代码；支持注释，人类审阅时可以直接批注（定稿的 schema 实例不保留注释，以免内容无节制地膨胀）；在 Agent 的训练语料中占比大，编辑鲁棒；`yq` 与 Python 的批处理生态成熟；git diff 按行，人类可审。

淘汰的候选：JSON 不支持注释，多行文本要转义；TOML 表达深嵌套和重复结构（多个 role、多个 barrier）很笨拙；Markdown 无法约束，批处理只能靠猜；textproto 有强类型、枚举和注释，但需要 proto 编译链，太重；SQLite 等数据库不能 git diff，人类无法直接审阅，Agent 也无法直接编辑。需要的“数据库感”在校验关系上，不在存储上。

### 5.3 template：CUE（尚未编写）

CUE 本来就是为数据、约束、模板、验证和生成设计的，可以直接验证 YAML：template 写成 CUE definition，schema 仍然是普通 YAML，Agent 无法在实例中编程。它适合 (template, schema) 共同演化：

- definition 默认关闭未知字段，也可以在特定位置开放结构，从而关闭稳定区域、开放创新区域；
- 支持 optional / required、候选集合与默认值，可以表达简单的跨字段关系；
- 多个约束可以无顺序合并，template、额外约束与迁移补丁可以分文件组合，结果与应用顺序无关；
- 可以用 subsumption 判断新旧 template 的兼容关系；
- 比完整的 JSON Schema 紧凑，可以逐步增加约束，不必一开始写完。

代价是：团队需要理解 unification、definition、open/closed 等概念；template 的元数据与展示顺序需要额外约定；Python 侧通常要通过 `cue` CLI 或 Go API 交互，不如 JSON Schema 直接；如果允许 Agent 使用 CUE 的全部能力，template 也可能逐渐变复杂。早期的 CUE 示意（字段已过时）见归档文件。

### 5.4 编辑 schema 时的上下文

Agent 编辑一个 schema 时，上下文中同时有三层：

1. **骨架**：新 schema 从 template 生成的空白问卷起步。字段树从 CUE 导出，可选词汇和一行说明以 YAML 注释的形式内联在每个待填项旁边，让 Agent 填空而不是创作。
2. **template 本体**：常驻上下文，约 2k token，回答“有哪些字段、哪些必填、词汇表是什么”。
3. **最近邻 exemplar**：2–3 个同类 kernel 已验收的 plan，回答“这个字段在实际设计里长什么样、填到什么粒度”。

人类侧使用 CUE 自动生成的 JSON Schema 和编辑器补全，审阅的是纯 YAML 的 plan。两个受众、一个真源、零重复维护。

## 6 schema 的产出方法

### 6.1 数据驱动地提炼语言

提取现有 SOTA kernel 的流水线 schema，本质上是一个数据问题。一门语言的语法本身无法作为产物交付，最好的办法是用大量的参考样例来记录这门语言；把现有 SOTA kernel 的核心思想提取出来，本身就是造数据的过程。难点在于把提取出来的大量高层流水线再压缩成一门小语言。

目标是结合专家经验，从几十个来源、风格、实现各异的 kernel 中得到一个“数据驱动的 schema IR”：前期各份 schema 的风格可以不统一，但每一对 kernel ↔ schema 都必须是正确的。这类似于从大量数据中逐步学习、提取一门语言，为此要建立一个可迭代、能自驱动的轻量级 research Harness。

数据来源已整理到 Agent 能自行阅读和区分的程度，收集工作基本完成：

- Atrex bench 中清洗过的 challenge，以及 `data/opensource` 中原有的大量 kernel，覆盖 sm90、sm103、sm120 与各种算子规格；
- hpc-ops 的 `src`，覆盖 sm90、sm100，偏传统的经典 kernel；
- 其他开源算子库，例如 FlashInfer、DeepGEMM、grouped GEMM、CUTLASS 示例与 FA 系列。

计划：

1. 收集 NVIDIA 卡（以 H 卡、B 卡为主）上的开源 SOTA kernel，覆盖 FA3/4、MoE、GEMM、MLA、DSA/GDN/KDA 等，做必要的清洗。
2. 用一份初版 template（表格模板 + 原语词汇）统一风格，让多个 Agent 并行地把 kernel 提取成 schema。Agent 有权修改 template、增加原语，但要尽可能保守，且必须优先抓大放小；在单个 kernel 上多轮迭代、不断精简，不同 kernel 分 batch 并行。关键是让 Agent 完备地覆盖一个 kernel 的设计，把 macro tile 与 micro pipeline 都讲清楚；另设一个专门的 Agent 确认 Designer 必需的内容没有遗漏、没有错误，Coder 的细节已经去掉。
3. 先描述清楚几个典型算子（最初设想是 FA4、fusedMoE、GEMM，目前各区文档用的是 FA4、MoE、GDN），再逐步加入其他算子：一方面检查原语和词汇是否够用，另一方面检查有没有重复、冗余、可以精简的词汇。
4. 检查得到的几十份 schema 没有事实错误，然后精简：舍弃特例的原语和表格项，抓重点、抓 insight，确定最小的交集原语，最终抽象成一门固定风格的语言并附上大量示例。交付物是一个 template 加上所有 SOTA kernel 对应的 schema。
5. 与 CAKE-IR 的前端语言比较，看是否收敛到一致。
6. 最后再研究检查器和评估器：先有一门好的语言。

### 6.2 (template, schema) 共同演化

- Agent 写 schema 前先看 template 有没有不足以描述的地方；有就先最小化地修改 template，再写 schema。
- Harness 以单个 task 为粒度：一个 task 针对一个 kernel 迭代若干轮（例如 5 轮），产出它的 schema 和新版 template；下一个 kernel 以已有的 schema 为例、以新 template 为种子继续。template 只迭代版本，schema 不断积累；修改 template 后要回改旧的 schema（通常只有删改才需要，新增一般不需要）。从 v0 开始，10 个 kernel 可能到 v10，但不一定每轮都要改 template，可能到 v5 就差不多了。template 的版本线与 plan 自身的 `meta_data.version` 是两回事：前者记录语言有哪些字段，后者只记录这份设计修订到第几版。
- 第一轮先放开语言的自由度、增加原语，逐渐覆盖全部 kernel；第二轮再精简 template、优化 schema。template 应当小而精，而不是大而全：只保留 schema 的核心，把细节踢给 Coder（移到下层），让 template 越来越规范、原语越来越少，schema 同步瘦身。
- 待定：kernel 的引入顺序。从简单到困难逐步撑大语言空间，适合人类理解，但可能每次都要改 schema，一开始也抓不住核心；从最困难的开始，一开始语言空间就很大、之后逐渐稳定，但可能不容易收敛、形态各异。

这套方法与 CAKE 构建 Cake IR 的方法（论文附录 A）几乎一致：收集生产级 kernel 语料（CUTLASS、FlashInfer、DeepGEMM、FA4、Alpha-MoE 等，其他 DSL 先由 Agent 翻译成 CUDA）；Agent 从语料中提取重复模式（barrier 编排、pipeline staging、warp-role 划分、TMEM 生命周期），形成候选抽象；人类专家注入硬件先验（TMEM 是一等公民、warp-spec 是主要的并行 idiom）；按 8 条设计原则迭代验收；移植新 kernel 时要么验证抽象，要么暴露缺口并触发 IR 修订，且 IR 改动必须通过全语料回归。区别只是 CAKE 把终点推到了编译器，本方案停在 schema 层。这也说明“数据驱动地从语料里长出一门语言”是可行的（[原图](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/3417228d-0093-4e4f-8f77-1a38cf8492f3.png)）。

### 6.3 质量保证

- **盲重建审计**：审计 Agent 只拿 schema、不看原 kernel，复述出 macro tile 调度和 micro pipeline 的时序（谁在哪个 stage 等哪个 barrier、SMEM 何时被谁复用），再与原 kernel 对照。这样完备性就有了可操作的标准：一个没见过原代码的合格读者，能否从 schema 无歧义地还原设计骨架。这也正是对 Coder 的要求，等于提前排练了 Designer 到 Coder 的交接。
- **转写公开的 SOTA kernel**：像 CAKE 自底向上演化 IR 那样，把公开的 SOTA kernel（FA2/FA3 论文、CUTLASS 示例、DeepGEMM 等）用 schema 手工或半自动地转写成设计文档。这同时检验 schema 的表达完备性、填充流水线模式库，并直接构成 Designer 的验收测试集：能否在没有参考的情况下重新推导出这些设计。
- **数值等价**：改写是否与原定义等价，由实现与 golden reference 的数值对比来检验，schema 里不写证明。

## 7 现状

- **已有**：六个区的字段规则文档 `schema_docs/1–6`，是旧版草稿的优化版本，每份都按“核心思想 / 字段定义 / 语法规则 / 关键原则 / 具体案例”组织。案例共用三个固定版本的 kernel：FA4（SM100，Dao-AILab/flash-attention `ce088ab9`）、MoE（vLLM TritonExperts，FP8 block 量化，vllm-project/vllm `28282ffe`）、GDN（FLA Triton 分块前向，fla-org/flash-linear-attention `954438d1`）；硬件案例为 B200（SM100）、MI355X（gfx950）、昇腾 950PR（NPU arch 3510），分别绑定 CUDA C++/CUTLASS、HIP/Composable Kernel、Ascend C 三个实现栈。旧版草稿与 tile_pipeline 的渲染脚本原型 `render.py` 归档在 `schema_docs/archived/old_schema_drafts.md`；该脚本的键名与旧版 YAML 样例并未对齐，复用前要按现行字段更新。
- **正在做**：逐区优化 `schema_docs/1–6`。即使是优化后的版本，字段规则和样例仍有许多细节需要调优。
- **尚未开始**：template（CUE），包括整体设计思想与原则以及六个区各自的 template；把六个区串起来的完整案例（full case）；`schema_wiki/` 中的 wiki 样例；更多 kernel 的转写与整体精简；检查器与评估器。

进展记录：

- 26/09/01：第一轮分区讨论（Contract、Design、缓冲区）。
- 26/09/03：待办——核对 GDN 在 Atrex bench 上的题目对齐（FlashInfer 开源版本与川悟的 trace 是否与 Atrex benchmark 的题目匹配）；跑通 AKA 优化；先做 planner + AKA 的初版。
- 26/09/06：配置 AKA + Agate + Atrex bench 的迭代平台（基于睿博支持多文件迭代的版本），从 FlashInfer 的开源 fallback 版本出发，从零复现 SM103 上的 M64 GDN（CuTeDSL）。运行 76 轮后，10 个典型 shape 的精度与川悟手调的 M64 SOTA 版本对齐，其中 9 个 shape 更快，平均提速 4.5%。
- 26/09/08 起：从按角色分组的前驱表演进到按 lane 记录动作与依赖的 tile_pipeline 写法，并写出渲染脚本原型。
- 26/09/09：第二轮分区讨论，确定 Contract / Design / Realization 三区六个子区。
- 26/09/14：写出 work_schedule 的草稿；第三轮分区讨论聚焦 math_reshape。
- 此后：写出 math_reshape、algorithm_requirement、hardware_specification、realization 的草稿，再把六个区的草稿优化为 `schema_docs/1–6`。

## 8 仓库结构与阅读顺序

```text
.
├── README.md                      本文件：背景、动机、分区、原则与现状
├── schema_docs/                   六个区的字段规则与样例（当前的工作对象）
│   ├── 1.contract.algorithm_requirement.md
│   ├── 2.contract.hardware_specification.md
│   ├── 3.design.math_reshape.md
│   ├── 4.design.work_schedule.md
│   ├── 5.design.tile_pipeline.md
│   ├── 6.realization.md
│   └── archived/
│       └── old_schema_drafts.md   原设计文档第 8 节：各区旧版 YAML 草稿与案例（只读）
└── schema_wiki/                   wiki 样例（规划中，目前为空）
```

给后续 Agent 的阅读顺序：

1. 先读本 README，了解仓库的目的、分区边界和通用原则；
2. 再读要处理的 `schema_docs` 文档；改动涉及跨区的名称、记号或案例时，同时核对相关区的文档；
3. 需要了解某条规则的来历或旧版写法时，再查 `schema_docs/archived/`。归档文件只读，不再维护。

原来的长篇设计文档 `Kernel流水线设计old.md` 已拆分为本 README 与上述归档文件，原文保留在 git 历史中。
