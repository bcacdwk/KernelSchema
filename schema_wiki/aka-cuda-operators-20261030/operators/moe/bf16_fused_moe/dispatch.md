# bf16_fused_moe

本目录登记完整设计集合与实例身份，并把同一 kernel 模板的参数级变化一次记录；精确参数、`when` 与修订号以 [root](root.yaml) 为准。

## 范围

共享 Contract 固定的任务域见 [root](root.yaml) CONTRACT（此处不复述）：单 rank（`len(sym_buffer_ptrs)=1`、`rank_idx=0`）、`activation='swiglu'`、shared 权重与 stats 计数恒为 None、`fast_math=False`、1 ≤ T ≤ 8192、四组 (H, I, E) 几何。当前收录：一份完整设计 root，及其三条参数记录（token 分块阶梯、模型几何与堆容量、激活数值特化）；无非 root 完整设计，域内没有已研究但未完整展开的独立候选。十个冻结用例是代表 workload 而非全域（metadata.json），下文实验覆盖只登记实际到达的实例。

实现入口链（固定实现范围）：`candidate/solution.py::Model.forward` 原样转发 17 个参数给原生 `bf16_mega_moe`（launch.cu:136）；host 完成校验与归一（clamp None → +inf，launch.cu:159-160；K-major/连续/形状/容量断言，launch.cu:153-216），把 `sym_buffer` 按 MegaMoEBuffer ABI 切出操作数视图与 ring 缓冲（launch.cu:219-221），随后按 `arch_major == 10` 进入 `sm100_bf16_mega_moe`（dispatch.cuh:1372），否则 `DG_HOST_UNREACHABLE` 拒绝（launch.cu:239-241）。整个候选只有一个设备 kernel 模板 `sm100_bf16_mega_moe_impl`（kernel.cuh:45；solution.py:33 的唯一 tag）。

Host 配置与 kernel 取得：`get_mega_moe_config` 依 workload 选出 block_m/store_block_m/block_k/pull 大小/stages/smem（dispatch.cuh:810-980）；22 个模板实参作为配置键查 9 个内嵌预编译特化（dispatch.cuh:1470-1484；runtime.cuh:890-909），未命中时 solution.py 的 resolver 以同一模板即时编译 cubin（solution.py:68-124）；浮点 NTTP（clamp/alpha/beta）按 IEEE 位编码进配置（solution.py:78-91）。TMA descriptor 在 host 线程本地缓存（dispatch.cuh:1039-1101）。

单次 launch：grid = num_sms（由 solution.py:128-136 设为 SM 数 − 2，内嵌特化均为 146）、block = 512（128 dispatch + 128 non-epilogue + 256 epilogue）、cluster = 2、动态 smem（dispatch.cuh:1486-1493）；非 cooperative，固定 host 路径不启用 PDL（runtime.cuh:356-367 默认 false），kernel 内 `cudaGridDependencySynchronize()`（kernel.cuh:269）为惰性钩子。每次 forward 只有这一次 kernel 提交；输出 y 仅由 kernel 内 combine 段写回。机制细节由 root 的 DESIGN 与 REALIZATION 承载，目录不复述。

运行环境事实不是配置变体，已落 root 的 REALIZATION 与 DESIGN：num_sms = SM 数 − 2 且为偶数（cluster=2，kernel_helpers.cuh:1394）；每 CTA 独占 SM 与 TMEM（kernel.cuh:45, 851-852）；内嵌特化命中表与 resolver JIT、TMA descriptor 缓存；堆计数器零初态（input.py:104）与清理尾段跨 launch 恢复（kernel.cuh:582-649）。

Contract 有效域外、本任务不可达的来源能力（如实登记，不进入任何设计记录）：
- 多 rank EP：kNumRanks > 1 时经 `SymBuffer::map` 偏移对端堆地址做 NVLink pull、远端 combine 写与跨 rank barrier（kernel_helpers.cuh:381-414, 915-954）；单 rank 下 `map` 为恒等（kernel_helpers.cuh:406-408），全部退化为本地操作，机制骨架不变。
- 共享专家：kHasShared 时调度器追加 SharedLinear1/SharedLinear2 任务段、combine 追加第 K+1 槽（kernel_helpers.cuh:1589-1621, 1636-1640；kernel.cuh:1250-1251）；本任务 shared 权重恒 None，`num_shared_experts=0`。
- `situ` 激活：仅存在于 heap 尺寸 API 的断言集合（launch.cu:23），本入口 `DG_HOST_ASSERT(activation == "swiglu")` 拒绝（launch.cu:155）。
- `fast_math=True`：epilogue 的 `__expf`/快速倒数分支（kernel.cuh:987-997）；Contract 固定 False。
- stats 计数更新：`cumulative_local_expert_recv_stats` 非空时清理段累加（kernel.cuh:620-623）；本任务恒 nullptr。
- `DG_COMM_KERNEL_DEBUG` 环境副作用（调用后 memset 整个堆，launch.cu:243-250）：调试开关，非设计内容。

已知来源限制：clamp 启用时在 BF16 域取界，见「激活数值特化」条目的如实登记（root 实例不受影响，其 clamp 折叠）；验收判据为 source only（root Golden_reference.Tolerance），不以默认判据重释。

## 完整设计

| 设计 | 具体机制与保留理由 | 关系 |
| --- | --- | --- |
| [root](root.yaml) | 单次 launch 的持久 warp-specialized mega kernel：4 个 dispatch warp 把散乱路由聚集为按 expert 连续的池块（定容 ring、lap 计数门控复用）；leader CTA 的调度器 warp 从全局原子计数器动态领取 FC1/FC2 任务并发布到 2-CTA 集群；swap-AB 的 2-SM UMMA（权重为 A、激活为 B、FP32 TMEM 累加）；FC1 epilogue 做 SwiGLU 并乘入路由权重后写入 l2 ring，`l2_full_mask` 位让 FC2 按 K-block 提前开跑；FC2 epilogue 把 per-route BF16 输出行按 (slot, token) 散布到 staging；epilogue warp 任务耗尽后转 combine，按槽序 FP32 归约 ≤8 槽后一次 BF16 写 y（y 的唯一写点，空路由写零行）；清理尾段恢复计数器零初态。保留理由：来源中唯一的端到端方案——一个 kernel 模板贯通主计算、输出形成、工作归属与主协作组织，不存在可独立成立的第二套设计。 | 阅读入口，保存唯一 Contract |

拟完整交付集合仅 root 一份（含 root ≤ 10）。无第二个完整候选的依据是结构性的：候选内唯一 `__global__` 模板（kernel.cuh:45）、唯一 launch tag（solution.py:33）、host 单 arch 分支（launch.cu:224-241）；9 个内嵌特化与 resolver 即时编译产物全部是同一模板在不同 NTTP 取值下的实例（runtime.cuh:890-909），差异轴全部为同规则参数变化（见下节），无一需要重新贯通解释主要计算、输出形成、工作归属或主协作组织；域内也不存在已研究但未完整展开的独立候选。

实例身份登记：root 为内嵌特化 config_5 实例 `sm100_bf16_mega_moe_impl<1920, 2048, 768, 128, 0, 8, 128, 128, 64, 32, 21120, 8, 4096, 128, 128, 256, 146, 1, clamp=+inf, alpha=1.0f, beta=0.0f, fast_math=false>`（runtime.cuh:905），冻结用例 6（T=1536、H=2048、I=768、E=128、masked 路由，shape_cases.json:83-96）经 host 启发式（expected=96、covered≈105.8 → block_m=128）实际到达且验收通过（PRECISION.md:3-5）。`META_DATA.Source` 维持候选入口（solution.py::Model.forward）：root 实例是该入口沿固定 host 链到达的具体模板实例，实例身份登记于本目录，不改变设计来源；承担范围以 root 的 `META_DATA.Dispatch.when` 为准，其中 block_m=128 属设计自定参数，启发式选择域是来源事实、记录于下节变体。

## 变体与组合

以下三条均为同一 kernel 模板的参数级取值变化，基线对象按 root 已写实的 Design 定义定位；条目排列不产生应用顺序，实施时须整体代入所选取值及其联动。

### token 分块 block_m 与联动（参数）

- 基线：[root](root.yaml) 的 `DESIGN.Math_reshape::Route_pool_gather` 符号 `block_m`（值 128）与 `valid_m`；`DESIGN.Work_schedule::DISPATCH.Decomposition.Strategy` 的 ring 槽（mod 165）与 lap（÷21120）；`DESIGN.Work_schedule::GEMM.Schedule` 的计数器穷尽值（pool_blocks×6、pool_blocks×8）；`DESIGN.Tile_pipeline::GEMM.Storage` 的 `accum`（UMMA_N=128 × 2 stage 的 TMEM 列）与 `epi_stage`（STORE_BLOCK_M=32）。
- 差异与联动：来源选择条件——host 启发式按每 expert 期望路由 token 数 T·K/E 及其 +1σ 裕量 covered，从 {16, 32} ∪ {64, 128, 192, 240} 选 block_m（dispatch.cuh:810-833）。联动：store_block_m 阶梯 8/16/32/40（dispatch.cuh:833）改变 EPI 的每 warpgroup store 块数 WG_BLOCK_M/STORE_BLOCK_M 与 `epi_stage` 形状（kernel.cuh:868-874）；stage 数按 smem 余量重算（dispatch.cuh:908），`act_stage`/`w_stage` 环深度与相关 Waiting 窗口随之变化；UMMA_N = block_m，动态收缩为 align(valid_m,16) 且须 %16=0、≤256（kernel.cuh:152, 157, 765-768, 776）；TMEM 列数 = 2×block_m 向上对齐到 2 的幂，192/240 均对齐 512（kernel.cuh:183-185；cuda_primitives.cuh:2009-2016）；ring 块数 kNumRingBlocks = num_ring_tokens/block_m，DISPATCH 的 ring 槽模数、l1_empty/l1_full 计数目标（kernel.cuh:43, 526-531, 669-672）与 GEMM 的 lap 世代全部随之变化。kCandidateBlockM 全集 {8,16,32,64,96,128,192,240} 仅用于 ring 容量与 1920 对齐（kernel_helpers.cuh:423-427；launch.cu:36-44；input.py:133-141），启发式不产 8/96。
- 条件与依据：同模板不同 NTTP，三区机制不随 block_m 改变；各可达取值均有内嵌特化或 resolver 即时编译支撑（runtime.cuh:890-909；solution.py:68-124）。实验覆盖：冻结用例实际到达 block_m = 16（用例 0/1 → config_1）、32（empty 路由用例 7 → config_2）、64（用例 8 → config_8、用例 2 → config_6）、128（用例 3/4/6 → config_7/0/5）、192（用例 9/5 → config_4/3），均验收通过（PRECISION.md:3-5）；240 的选取条件为 num_blocks×192 < covered ≤ num_blocks×240（num_blocks = ceil(covered/240)，dispatch.cuh:822-831；bump 分支要求 E/ranks < 14，域内恒不触发），域内可达区域为 covered ∈ (192,240] ∪ (384,480]——(576,720] 段因域内 covered 上限 ≈534.6（T=8192、E=128）不可达，E=256 时 covered ≤ 272 仅前段适用；如 T=3072（covered≈206）或 T=6400（expected=400、covered=420、c=7680），无冻结用例，均走 resolver。

### 模型几何与堆容量 H/I/E/c（参数）

- 基线：[root](root.yaml) 的 `META_DATA.Dispatch.when` 几何（H=2048、I=768、E=128、c=1920）；`DESIGN.Work_schedule::GEMM.Decomposition` 的 ncl 分界 6 与 kb_count（FC1=32、FC2=12）；`DESIGN.Work_schedule::DISPATCH.Decomposition.Strategy` 的 ring 容量 21120；`DESIGN.Tile_pipeline::DISPATCH.Storage.pull_stage`（4096 字节）；`DESIGN.Work_schedule::COMBINE.Decomposition`（4096 字节单 chunk 整行）。
- 差异与联动：域内其余取值为 (H,I,E) ∈ {(4096,768,128), (4096,1536,128), (7168,256,256)}，c = 1920·ceil_div(T,1920) ∈ {3840, 5760, 7680, 9600}（input.py:100-101）。联动：每池块 FC1/FC2 的 N-cluster 数 = 2I/256、H/256，GEMM 的 ncl 划分、任务总数与计数器穷尽值随之变化（kernel_helpers.cuh:1371-1374, 1549, 1558）；kb_count = H/64、I/64；`l2_full_mask` 位宽 = 2I/128，受 ≤64 静态约束（kernel_helpers.cuh:1339-1342）；pull 分块 kNumBytesPerPull = H·2 折半至 ≤4096——H=7168 → 3584、每池行分 4 块（dispatch.cuh:941-946；kernel.cuh:516-519），`pull_stage` 容量随之变化；combine 分块 kNumChunks：H ≤ 4096 为 1，H=7168 为 2（kernel.cuh:1193-1195），COMBINE 的工作划分按 (token, chunk)，`combine_stage` 与 `reduce_acc` 容量随 chunk 字节变化（kernel.cuh:1188-1197）；ring 容量 = 全 block_m 候选集上最坏 live pool blocks 再按 1920 对齐（launch.cu:36-44；kernel_helpers.cuh:1243-1275），DISPATCH/GEMM 的 ring 槽模数与 lap 计数目标随之变化；E=256 时调度器每 lane 缓存 8 个 expert 计数（kernel_helpers.cuh:1370），清理段 expert 清理循环的步长恒为 kNumSMs−1，随 E 变化的是循环上界 kNumExpertsPerRank（即每 CTA 迭代次数，kernel.cuh:604）。SCHED 的 L1 warmup 波数上限由两级 N-cluster 数决定：四组 Contract 几何均 2I ≤ H，交错差项恒 0、上限保持 2（kernel_helpers.cuh:1228-1236），实际值再与 L1 总波数取 min（kernel_helpers.cuh:1492-1494），无随几何的额外联动。寄存器预算分支条件为 kNumExpertsPerRank > 64（kernel.cuh:308-311），Contract 的 E ∈ {128,256} 恒满足，root 的 96/88/160 分配不变。堆 ABI 操作数偏移与 c 的关系已由 root CONTRACT `Heap_operand_offsets` 固定，几何只改容量与循环计数，不改机制。
- 条件与依据：同模板不同 NTTP。代表实例：config_0（c3840/H7168/I256/E256）、config_6/7（H4096 两种 I）、config_3/4（c9600/5760）；c=7680（T ∈ (5760,7680]）无内嵌特化，经 resolver 即时编译到达，仍为参数级变化。实验覆盖：冻结用例 2/3（H4096）、4（H7168/E256）、5/9（c9600/5760，block_m=192）实际到达并验收通过（shape_cases.json；PRECISION.md:3-5）。

### 激活数值特化 clamp/alpha/beta（参数）

- 基线：[root](root.yaml) 的 `META_DATA.Dispatch.when`（activation_clamp_opt = None、host 归一为 +inf；alpha = 1.0；beta = 0.0）与 `DESIGN.Math_reshape::Pool_FC1_gated_activation` 的对应折叠。
- 差异与联动：三值按 IEEE 位编码进入 NTTP（solution.py:78-91；kernel.cuh:78-80），在 FC1 epilogue 内为 constexpr 分支：alpha ≠ 1 时 sigmoid 输入先乘 alpha（kernel.cuh:984-986）；beta ≠ 0 时 up 分支先加 beta（kernel.cuh:999-1000）；clamp ≠ +inf 时对 gate/up 取界（kernel.cuh:975-979）。无存储、调度或容量联动。已知来源偏差（如实登记，不得写成 FP32 clamp 语义）：kernel 在 BF16 域以 `__hmin2/__hmax2` 对已 BF16 化的 gate/up 与 BF16 化后的 L 取 min/max（kernel.cuh:974-979），而 root CONTRACT `FC1_and_activation` 规定对原 L 做 FP32 min/clamp；L 非 BF16 可表示时界值相差至多 L 的一个 BF16 量化步，差值再经激活与 ≤1 的路由权重缩放。
- 条件与依据：Contract 域 clamp ∈ [1,16]、alpha ∈ [0.5,2]、beta ∈ [0,1]，其余取值同法经 resolver 编译，无独立记录价值。代表实例 config_6（clamp=7.0、alpha≈1.702、beta=1.0）；冻结用例 2 实际到达且验收通过（shape_cases.json:28-43；PRECISION.md:3-5）——7.0 为 BF16 可表示值，该用例未触发上述偏差；非 BF16 可表示取值的数值影响未实测，不补造容差外验证。
