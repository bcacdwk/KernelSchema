# bf16_nvfp4_gemm

## 范围

**任务范围**由 [root.yaml](root.yaml) 的唯一 Contract 固定：m ∈ [1, 8192]，(n, k) ∈ {(4096,12288), (6656,2560), (8192,2560), (12288,4096), (24576,4096)}，alpha 可为 None，结果原地写满 out_buffer 的 (m, n) 逻辑区域并把同一张量原样返回。目录不改变该范围，也不解除任务要求的完整交付义务。

**本次收录范围**是固定实现中冻结入口真正能到达的部分。`candidate/solution.py::Model.forward` 只把 (a, b, b_descale, alpha, out, cfg) 交给扩展 `mm`（`kernels.cu::mm_bf16_fp4_v0` → `launch_mm`）；backend / out_dtype / block_size / enable_pdl 从不进入计算。cfg 取自环境变量 `BF16NVFP4_CFG`，默认 0（auto），冻结任务与输入生成器都不设置它，所以收录范围 = auto 分支在 Contract 域内实际选到的实例，加上 `launch_mm` 末尾无条件兜底的 `w4a16_kernel`。solution.py 自带的 `BF16NVFP4_GRAPH` 图层默认关闭，源码自述为已否决、仅留档的臂，不构成设计。`mm_ptr`、`bench_*`、`get_prof`/`clear_prof`/`last_use`/`graph_ok`/`graph_mode` 都在 forward 路径之外。入口校验（2D 连续、N%64==0、K%64==0、K==2·b.size(1)、scale 形状 (n,k/16)、out 形状 (m,n)）域内全部满足，不进 Contract。

**共用数值路线与真正的区分轴。** 所有 GEMM 臂共用一条数值路线：e2m1 码经 smem LUT 或 `cvt.rn.f16x2.e2m1x2` 解码，与 e4m3 组尺度相乘（≤6 位有效位，在 bf16/f16 中精确）得到 bf16 或 f16 的 B 操作数；A 保持 bf16（部分臂先由独立 prepass 转 f16）；乘加在张量核心中以 fp32 累加，最后乘 alpha 并单次 RNE 转 bf16 写回。方案之间的实质差别只在五处：张量核心代次与累加器位置（寄存器 / TMEM / 不用张量核心的 SIMT FFMA）、B 解码发生的位置（kernel 内 smem 环 / 独立 step 全局物化 / 每 lane 寄存器）、工作单元（128×64 与 128×128 tile、256×512 cluster pair tile、32 或 64 行权重 band、单行权重 warp）、K 是否切分及跨 CTA / 跨 kernel 归并、角色划分（对称 256 线程 / producer-issuer-dequant-epilogue 分工 / 每 warp 一行）。因此下面按这五处而不是按源码的近 60 个 namespace 划分设计。

**auto 分派在域内的实际去处**（`launch_mm`，按源码判定顺序；sm_count() 为设备 SM 数）：

- m == 1：N·K > 32e6 → cfg248（`gvt::gvt4_kernel<NB>`），否则 cfg170（`gv::gv_kernel<1,NB>`）；域内三个 k 都满足 K/512 ∈ {5, 8, 24}。
- 2 ≤ m ≤ 32：K ≥ 4096 且 N/64 ≥ sm_count()（域内即 (12288,4096) 与 (24576,4096)）→ cfg217/218/219（`smv13`，MP8/16/32）；否则 M ≤ 8 → cfg191（`smv9` MP8）、9 ≤ M ≤ 16 → cfg192（MP16）、17 ≤ M ≤ 32 → N ≤ 4096 时 cfg193（MP32）、N > 4096 且 N/64 ≤ sm_count() 时 cfg231（`smv14` MP32），再否则 cfg140（`smv5` MP32）。域内 f16_ok 门槛（K%256==0、N%64==0、M ≤ 16、(K/256)·ceil((N/64)/(2·sm_count())) ≥ 16）只对 (4096,12288) 的 M ≤ 16 成立，且给出的仍是 cfg191/192。上述去处与 cfg128/129/201/202/205/207 的不可达性以 128 ≤ sm_count() ≤ 192 为前提（cfg231 要 N/64 ≤ sm_count()，K2560 侧最大 N/64 = 128；cfg217-219 要 N/64 ≥ sm_count()，K4096 侧最小 N/64 = 192；源码自己的代价模型把 SM 数写成 148，正落在这个区间）；三个尾巴 cfg82/20/11 则与 SM 数无关地不可达，因为它们要求 K%256 != 0，而域内三个 k 都是 256 的倍数。SM 数未声明，两个方向的偏移都会改选同族的其他成员。低 SM 数侧：N/64 > 2·sm_count()（真多波）时 K2560 给 M ≤ 8 的 cfg128、M 9-16 的 cfg205、M 17-32 的 cfg207；只满足 sm_count() < N/64 ≤ 2·sm_count() 时 cfg231 门槛失效，M 17-32 落到 cfg140。高 SM 数侧：sm_count() > N/64 使 cfg217-219 门槛失效（域内即 K4096 且 N > 4096 的 (12288,4096) 与 (24576,4096)），M ≤ 16 给 cfg128（M ≤ 8）/ cfg129（M 9-16）。按引擎分：cfg128/129 是非持久的 `smv6` f16 臂，cfg205/207 是持久的 `smv11` f16 臂，cfg140 是非持久的 `smv5` bf16 臂；同分支的持久 `smv10` 臂 cfg201/202 需要 tsm6·waves6 < 16，域内 K ≥ 2560 走不到。这些成员与 cfg191-193/217-219/231 同属 weight_band_mma 一族，仍由「变体与组合」中的 f16 链、持久 band 循环与 MP/BNw 记录覆盖，不需要新的完整设计。
- m > 32：记 pairs13 = ceil(M/256)·(N/512)、tiles10 = ceil(M/128)·(N/128)。
  - pairs13 ≥ 120：记 G = min(pairs13, sm_count()/2)、R = ceil(pairs13/G)（pair 轮数）、W = G·R − pairs13（末轮空槽），则 K ≥ 8192 且 W ≥ 8R−5 时 cfg176（`ws50sk`），否则 cfg119（`ws50b`）。
  - 否则 K ≤ 4096 且 pairs13 ≥ 64：低 pair 占用（100·pairs13 < 6364·ceil(pairs13/74)）时，tiles10 ≥ 148 走 wall 模型（只有 K ≤ 2560 才真正比较 119 与 249 的代价，其余取 cfg249），tiles10 < 148 走 cfg76（`ws26`）；否则 pairs13 > sm_count()/2 走 cfg101（`ws29`）、K ≥ 4096 走 cfg94（`ws27e`）、其余 cfg119。该分支最先判定的 N/512 < 10 → cfg29（`ws13`）在域内不可达：K ≤ 4096 的四个 N 都 ≥ 6656。
  - 否则 split-K 代价模型给出 sk > 1 时 cfg144（`ws10sk`）。
  - 否则 ws10_big —— 域内恒成立（K%128==0、N%128==0、K ≥ 2560、tiles10 ≥ 32）—— tiles10 < 148 走 cfg28（`ws10`），多波时按 wall 模型在 cfg176 / cfg119 / cfg249 间选。
  - cfg10（`ws2`）与 cfg5（`ws`）两个分支域内不可达（ws10_big 先命中）。
- 设备 cc ≠ 10.3：全部专用门槛失效 → use = 1；任何未匹配的 use 也落到 `launch_mm` 末尾的 `w4a16_kernel`。这条兜底路径覆盖整个 Contract 域，是 root 的实际来源。
- 各设计的 when 只写自己已核对的输入与调用条件，范围允许重叠（如 split_k_reduce 与 umma_single_cta 的形状条件相交，weight_band_mma 与 simt_warp_row 都含 m == 1），目录不规定隐式优先级，也不复刻上面这套 selector 或其实测依据。六份 when 与 YAML 文本的对应：
  - root：承担整个 Contract 域，无设备门槛；m 尾块由尺寸门控零填充与 row0 < M 守卫处理。
  - umma_single_cta：cc 10.3、m > 32、n 与 k 为 128 倍数，且另限 ceil_div(m,128)·(n/128) 个 tile 至多填满一波；单提交、无 cluster、无工作区。
  - pair_two_pass：cc 10.3、m > 32、n 为 512 倍数、k 为 64 倍数；覆盖 auto 给两次提交的大形状，pair tile 在常驻 cluster 上跑多波。
  - pair_stream_k：cc 10.3、m > 32、n 为 512 倍数、k 为 64 倍数；覆盖 auto 切分的长 k 大形状，slot_count 个常驻 cluster worker 均分 pair stage。
  - split_k_reduce：cc 10.3、m > 32、n 为 128 倍数、k 为 256 倍数且 m·n 为 4 倍数；需要两次提交与 FP32 全局工作区。
  - weight_band_mma：m ≤ 16（mpad = min(m,16) 个 token 行入 stage），并要求源码的 cc 10.3 门槛。
  - simt_warp_row：只写 m == 1、k/512 ∈ {5, 8, 24} 与寄存器级机制需求（每 lane 寄存器、精确窄格式解码、warp 集体归约），不含 cc 门槛——其 SIMT 机制本身不依赖 10.3 特性，auto 分派到达它仍以上面 cc == 10.3 的 selector 分支为条件。

**跨方案的提交与寿命事实**（影响每一份设计的 Realization 与提交身份）：

- 每次调用在当前 PyTorch 流上提交 1 或 2 个 kernel，无事件、无显式同步，host 入队后立即返回同一 out 张量；完成关系只由流序给出。
- `launch_mm` 之上还有一层默认开启的 v20 CUDA-graph 复放：key = (a, b, s, alpha, out 指针, M, N, K, 解析后的 use, epoch)，同一 key 第 1 次普通执行、第 2 次在私有非阻塞捕获流上录制并实例化后立即复放、第 3 次起直接复放。录制本身不执行计算，实例化失败则回到普通流重跑一次，因此每次调用仍恰好提交一次工作；任何 API 失败 → 永久普通启动。域内被复放白名单覆盖的是 cfg119/144/28/176/191/192/193/217/218/219，以及 cfg170（条件是 N ≥ 8192 且 N·K ≤ 32e6，域内只有 (8192,2560) 满足，(6656,2560) 仍是普通启动）；cfg76/94/101/231/248/249 不在白名单内，普通启动。
- 多个臂使用进程级 grow-only 全局工作区（cudaMalloc）：`g_a16`（f16 A，smv6/smv13 系）、`g_ws10fw`（f16 A，cfg249）、`g_ws50b`（bf16 B 工作区，cfg119）、`g_ws75b` + `g_sk75`（bf16 B 工作区 + f16 stream-K 部分结果 + 4KB 标志页，cfg176）、`g_sk`（fp32 split-K 部分结果，cfg144）。跨调用复用的安全依据是流序；扩容时调用 `v20_invalidate()` 销毁全部已捕获图并递增 epoch。cfg119 与 cfg144 在分配失败时降级为 `ws10` 单 kernel；cfg176/249 与 f16 小 M 臂则直接 TORCH_CHECK 抛出。这些工作区由受影响节点自己的 `Flow.Handoff`（`storage: global`）与 Realization 表达，不在目录另立记录；cfg176 的标志页已由 pair_stream_k 的 REALIZATION 写清——分配时一次异步 memset 清零的初态与读后由同一 trailing segment 自复位，延迟 fixup 的自旋等待因此有可依据的初态。
- PDL 由 cfg 决定而非参数：第二个 kernel 以 `cudaLaunchAttributeProgrammaticStreamSerialization` 启动，第一个 kernel 在 store 后 `__threadfence` + `griddepcontrol.launch_dependents`，第二个 kernel 先 `griddepcontrol.wait`（无该属性时是 no-op）。Contract 的 enable_pdl 从不进入 `launch_mm`，任何设计都不得声称由该参数驱动 PDL。
- 不计算 Contract 结果的仪表路径（tc/tma256/ws/ws2 的 mode 1/2、`DMAONLY=1`、`PROF=1` 计数器）全部只能由强制 cfg 进入；域内生产启动一律 mode 0 / DMAONLY 0 / PROF 0。

**已识别机制的记录去处**：目录当前共七份完整 YAML——root 与 umma_single_cta、pair_two_pass、pair_stream_k、split_k_reduce、weight_band_mma、simt_warp_row 六份并列设计；六份的三个 Design 区域全部本地定义，引用图中没有 `from`，只共享 root 的 Contract。十五条已识别的参数、局部与组合记录的正式条目全部收在下面的「变体与组合」小节，每条直接定位至少一个当前完整 YAML 及其中已有对象，用数学符号、step、role、存储或唯一的 rewrite 名称定位，不以行号或列表序号为修改地址；组合条目直接列出完整基线，所叠记录只按名称互链，不层层组合，条目排列也不产生 patch 应用顺序。没有一条记录以 root 为基线：BM/BN/BK/STAGES/THREADS/GROUP_M 是 `w4a16_kernel` 唯一一次实例化用到的编译期常量，源码没有第二个配置，root 也没有全局工作区、PDL、图复放或第二次提交，因此这十五条没有一条是 root 自身已选值的另一种取值或围绕 root 已有对象的局部改动。各记录描述的多为 auto 未选中的强制臂或域内未到达的形状，其条件与依据按源码事实照实写在条目内，不补造验证。

**已研究但未完整展开的独立方案**（机制确实独立，本次不交付完整 schema）：

- `tcs::w4a16_small_kernel<16/32>`（cfg4）与 `smtc::smtc_kernel`（cfg160-162）：小 M 的互换操作数 tcgen05 方案——权重作 UMMA 的 M（64 行/CTA）、token 作 N（≤32），D 为 64 lane × N 列 TMEM，epilogue 按 D[w][a] → C[a][n0+w] 转置 scatter；smtc 进一步用 tid256 producer / tid257 issuer / 8 个解码 warp 的三角色划分与 64k 原子子块 BDQ 环。取舍理由：只能由 `BF16NVFP4_CFG` 强制进入，冻结入口不可达；其“权重为 M、token 为 N”的取向已由 weight_band_mma（mma.sync 版）代表，差额是张量核心代次与累加器位置。smtc 另有源码自述的否决依据（UMMA 每条 ~128-160 周期与 N 无关，token 作 N ≤ 32 浪费 >94%）。
- `tc::w4a16_tc_kernel<256/64>`（cfg2/3/12-15）：第一代 tcgen05 方案，cp.async 而非 TMA 装载、256 线程全部参与解码、tid0 发 UMMA、两次 `tcgen05.commit` 分别跟踪 BDQ 与 A 槽的释放。取舍理由：强制臂，机制已被 TMA + 角色化的 umma_single_cta 取代。
- `ws` / `ws2` / `ws3` / `ws4` / `ws5` / `ws6` / `ws7` / `ws8` / `ws13`-`ws26` / `ws28` / `ws51`：大 M 的实验谱系，含单 CTA TMA（`ws`）、cluster=2 pair（`ws2`）、cluster=8 且 A 的 32 行切片向 4 个 z 同伴多播（`ws3`）、cluster=4 跨两个 m-pair 的 B 多播（`ws51`）、N=512 单 CTA（`ws4`）、持久与环重平衡（`ws15`/`ws24`/`ws25`/`ws26`）。取舍理由：域内不可达或仅强制臂；其中仍被生产使用的机制（cluster pair、持久循环、专用 epilogue warp）已在入选设计与「变体与组合」的记录中表达，而 TMA 多播（cluster 4 / 8）确实未被任何入选设计覆盖，属已研究但未展开的独立协作机制。
- `sm` / `sm2` / `sm2m` / `sm3` / `sm29`（cfg11/19/80/81/20/82）：小 M 的早期 tcgen05 互换操作数方案（权重作 UMMA 的 M = 64 行、TMEM 累加、TMA 或 cp.async 装载），与上面的 `tcs`/`smtc` 同族。取舍理由：域内不可达（auto 表达式里 cfg82/20/11 这三个尾巴的前置条件 K%256 != 0 在域内恒假，其余为强制臂），机制差额只是张量核心代次与累加器位置。
- `smv5` / `smv6` / `smv10` / `smv11` / `smv12` / `smv13` / `smv14` / `smv16` / `smv17` / `smv18`：不是独立方案，而是 weight_band_mma 的同族成员，差额为 BNw（32/64）、BK（256/512）、环深、持久 band 循环、解码与 ldmatrix+mma 的软件流水、SF 走 LDG 而非 TMA 等，已按「变体与组合」的参数与局部记录表达；域内生产成员是 `smv9`（入选）、`smv13`（cfg217-219）与 `smv14`（cfg231），`smv5` 的 cfg140 只在设备 SM 数低于域内 N/64 时才被 auto 选到，其余为强制或未选中臂。
- `tma256::w4a16_tma_kernel`：源码中定义但从未被启动（`launch_mm` 无对应分支），只有它的 `tma_2d`/`tma_3d`/`mbar_arrive`/`mbar_arrive_expect_tx` 被所有 TMA 臂复用；不作为设计。

**未知与受限边界**：

- 冻结材料未声明设备 SM 数，而波数与占用门槛（148、sm_count()/2、2·sm_count()）都依赖它；`Hardware_specification.Device` 仍是占位。SELECTION.json 的 ncu 环境块给出了实际认证设备与工具链（NVIDIA L20D、build_target 10.3a、锁频 2032 MHz、torch 2.9.0+cu130、driver 580.105.08），同样没有 SM 数。本目录的分支边界按源码原样用 sm_count() 表达，具体数值路线随设备而变；只有 split-K 代价模型例外，它把波数写成硬编码常量 148，因此其判定与设备 SM 数无关。
- 全部专用臂要求 cc == 10.3，认证环境记录的 build_target 也正是 10.3a；冻结材料只有该设备上的采样验证（fixed 套件 shapes 0-9，每 shape 至多 2 个种子，另有 domain 套件 d0-d9），其他设备只走 root，这是源码事实而非已验证结论。
- 图复放在认证计时中是否实际生效无从判断：CERTIFIED_BEST.json 只记录通过与否和时延，未记录 launch 绑定，源码注释中的复放收益属未验证主张。
- 实验覆盖不是选择依据。认证材料两套都 full_pass，但都不记 launch 绑定，冻结材料也没有给出 d0-d9 的 shape，因此无从判断哪一份设计在哪些 shape 上被计时；七份设计的保留依据是 auto 分派在 Contract 域内确实会选到它们。例如 split_k_reduce：fixed 十个 shape 没有一个走到 cfg144（m ≤ 32 的三个落在小 M 分支，pairs13 ≥ 64 的六个被前面的 pair 分支接走，shape 3 的 tiles10 = 128 虽 < 148 但代价模型给出 sk = 1），而 m = 64、n = 8192、k = 2560 在域内且模型给出 sk = 2 → cfg144，该判定不含 sm_count()，所以这份设计的收录范围不依赖认证样本。
- 数值类别全部来自源码主张，本次不运行 GPU、未独立验证：f16 链在幅值小于 2^-14 的 subnormal 区间有 ≤1 ulp tie-flip（源码称冻结分布下 bf16→f16 精确、mismatch==0）；K 维切分的部分结果带来额外舍入与重结合，源码对 stream-K 的 f16 部分结果记录 rel_l2 3.3-5.9e-4（f16-RN 部分舍入类，比 5% rtol 小约两个数量级），对 K 切分族另记录 5.6-6.8e-5 的重新结合类，两者都声明 mismatch==0；`smv9` 的 4 路 kb 切分相对 `smv5` 的 2 路是部分和重结合（mismatch==0 而非逐位相同）。这些类别必须由各节点正文写进对应设计的精度说明，不能只写“精确”。
- 源码注释与 auto 分派有若干过期陈述（`ws10fw`/`smv9`/`smv14`/`ws27e` 的注释写 never auto、forced arm，但 auto 分支确实会选到 cfg249/191-193/231/94）。本目录一律以代码为准，注释只作导航。
- Contract 中 no other memory is written 一句按调用方可见内存理解：入选的 pair_two_pass、pair_stream_k、split_k_reduce 以及 f16 prepass 记录都会写实现自己 cudaMalloc 的全局工作区，那不是调用方可见的输出。这类设计自有全局物化由各节点的 `Flow.Handoff`（`storage: global`）或 REALIZATION 的工作区登记（pair_stream_k 的部分结果与标志页属后者）表达，不需要在 `Hardware_specification.Execution` 增设存储名，也不构成回 P1 改 Contract 措辞的理由。

## 完整设计

| 设计 | 具体机制与保留理由 | 关系 |
| --- | --- | --- |
| [root](root.yaml) | `w4a16_kernel`（cfg1，也是 `launch_mm` 末尾兜底）：单 kernel、一个 CTA 承担一个 128×64 输出 tile，256 线程对称分工——同一批线程做 cp.async 三级环装载（A 16B 块带 XOR swizzle 与尺寸门控零填充、packed B 4B 单元、尺度 4B/行）、按 256 项 nibble LUT 与 `mul.rn.bf16x2` 把 packed B 现场解码进单缓冲 BDQ、`ldmatrix.x4` + `mma.sync m16n8k16` 累加到每线程 32 个 fp32 寄存器、乘 alpha 后按 (row0, col) 与 (row0+8, col) 成对写 bf16，GROUP_M=8 栅格化。保留理由：唯一不依赖 tcgen05 / TMA / TMEM / cluster / 全局工作区的完整方案，覆盖整个 Contract 域（含 m 尾块的零填充与 row0 < M 守卫），机制最少因而是理解与比较其余方案的入口。 | 阅读入口，保存唯一 Contract；when 覆盖整个域、无设备门槛；本份不承载任何参数/局部/组合记录，理由见范围中的记录去处 |
| [umma_single_cta](umma_single_cta.yaml) | `ws10::w4a16_ws10_kernel`（cfg28，BM/BN/BK = 128/128/128，320 线程，grid = tiles）：tid256 为 TMA producer（A 与 packed B 各拆两个半盒，`expect_tx` 记 40960B），tid288 为 UMMA issuer（`tcgen05.mma cta_group::1 kind::f16`，每 stage 8 次 K16，`tcgen05.commit` 释放 A 槽与 BDQ 槽），warps0-7 为解码角色（每 2 线程一行，尺度按行 LDG 并预取下一 stage，`cvt` 链写 BDQ 双槽环）并在末尾内联做 epilogue（`tcgen05.ld 32x32b.x32` 读 TMEM、乘 alpha、`st.global.v8`）。保留理由：tcgen05 + TMEM 累加 + 角色化 mbarrier 流水的最小完整形态，输出由 TMEM 而非寄存器形成，是 pair 系与 split-K 系的公共骨架。 | 并列完整设计，只共享 root 的 Contract，三个 Design 区域全部本地定义。when：cc 10.3、m > 32、n 与 k 为 128 倍数，且 tile 至多填满一波（单提交、无 cluster、无工作区）；两遍 pair 与 k 切分链在工作区分配失败时重新发射这一单 kernel。五条记录以它为基线：持久 tile/band 循环与跨工作环复用、f16 操作数链与 A 转换 prepass、W-first preflight、tile 形状环深度与调度旋钮、cfg249 组合；PDL 启动配对记录以它为单提交反例点名，PDL 只经 cfg249 组合到达它 |
| [pair_two_pass](pair_two_pass.yaml) | `ws50b::deq_pass2_kernel` + `ws50b::w4a16_ws50b_kernel`（cfg119，cluster = 2，BMp/BNp/BK = 256/512/64，320 线程，grid = 2·min(pairs, sm_count()/2)）：第一个 step 把整块 B 精确解码成 bf16 全局工作区（每线程 2 个 16 元素尺度组，`st.global.v8` 写 64B）；第二个 step 中一个 256×512 pair tile 由 cluster 内两个 CTA 共担，rank0 的 tid288 发 `tcgen05.mma cta_group::2`（512 列 TMEM，每 stage 2 个 N 半区 × 4 次 K16），`tcgen05.commit ... multicast::cluster` 同时释放两 CTA 的 empty 屏障，rank1 的 tid289 用 `mapa` 远程 arrive rank0 的 pair 屏障，tid256 producer 先 `griddepcontrol.wait` 再发 TMA，warps0-7 为专用 epilogue（按 N 分半读 TMEM、乘 alpha、`st.global.v8` 写 C），pair 循环以 G = gridDim.x/2 持久推进并按 group_m 栅格化。保留理由：B 操作数的生产者移出 GEMM（跨 step 全局物化）、工作单元变成 cluster pair tile、输出由专用 warp 形成，三项都与单 CTA 方案不同。 | 并列完整设计，只共享 root 的 Contract，三区全部本地定义。when：cc 10.3、m > 32、n 为 512 倍数、k 为 64 倍数；覆盖 auto 给两次提交的大形状——pair tile 在常驻 cluster 上跑多波，需要 2·n·k 字节全局工作区，分配失败退回单 CTA 设计。四条记录以它为基线：cluster pair 内联解码、专用 epilogue warp 与 N 分半释放、PDL 启动配对、tile 形状环深度与调度旋钮 |
| [pair_stream_k](pair_stream_k.yaml) | `ws50sk::deq_pass_kernel` + `ws50sk::w4a16_ws50sk_kernel`（cfg176，cluster = 2，同一 pair 骨架）：工作划分改为 stream-K——把 U = pairs×T 个 stage 按 slot 均分（`seg_plan` 给出 [完整 pair][head][tail] 段序列，head/tail 是跨 slot 的部分 k 区间）；head 段把 f16 部分结果 dump 到全局 SK 并 `st.release.gpu` 置标志，tail 段先 `ld.acquire.gpu` 自旋等前一个 slot 的标志、读回部分结果加进自己的 TMEM 值再写 C，写后 `flag_clear`；epilogue 按 kind 三分支，ring 索引跨段连续。保留理由：K 维切分 + 跨 CTA 部分结果归并（release/acquire 握手、延迟 fixup、每 slot 的段序列与退出条件）改变了工作领取、退出与输出形成，必须整体贯通；G == pairs 时退化为 pair_two_pass 的调度。 | 并列完整设计，只共享 root 的 Contract，三区全部本地定义；与 pair_two_pass 同一 cluster pair 骨架（Work_schedule Thesis 自认），GEMM step 改为 stream-K 段序列。when：cc 10.3、m > 32、n 为 512 倍数、k 为 64 倍数；覆盖 auto 切分的长 k 大形状，两个全局 scratch 任一失败直接 raise。「tile 形状、环深度与调度旋钮」记录以它的 MAIN Mapping.Strategy group_m 为并列基线（cfg177/178 为其 GM16/GM4 对照臂）；PDL 启动配对记录以它为负向兄弟（cfg176 关：REALIZATION 明记无 serialization 属性，`await:deq_grid` 依流序立即解析） |
| [split_k_reduce](split_k_reduce.yaml) | `ws10sk::w4a16_ws10sk_kernel` + `ws10sk::sk_reduce_kernel`（cfg144，单 CTA 128/128/128，grid = tiles·sk，sk = sk_pick ≥ 2）：主 kernel 按 `blockIdx.x % sk` 领取 K 的一段（`t0 = sk_id·(T_all/sk)`，段长 T_all/sk），角色划分同 umma_single_cta，但把 tile 的 fp32 累加结果写进全局 P[s]；归并 kernel 以 PSS 启动、`griddepcontrol.wait` 后每线程处理 4 个元素，顺序累加 sk 份部分结果、乘 alpha、转 bf16 写满 C。保留理由：输出不由 GEMM kernel 形成，而由独立归并 kernel 在跨 kernel 全局部分结果上形成；sk 退回 1 或 scratch 分配失败时整条链退化为 umma_single_cta 的单 kernel 形态。 | 并列完整设计，只共享 root 的 Contract，三区全部本地定义；MAIN 与 umma_single_cta 同一 `ws10` 单 CTA 引擎与角色划分（Math_reshape Thesis 自认），只把 k 归约切成两段。when：cc 10.3、m > 32、n 为 128 倍数、k 为 256 倍数且 m·n 为 4 倍数；需要 8·m·n 字节 FP32 工作区与两次提交，分配失败重发 umma_single_cta 的单 kernel 形态。三条记录以它为基线：f16 操作数链与 A 转换 prepass、PDL 启动配对、split-K 路数与代价模型 |
| [weight_band_mma](weight_band_mma.yaml) | `smv9::w4a16_smv9_kernel<16,0,0>`（cfg192，同族 cfg191/193；BNw = 32，BK = 256，288 线程，`__launch_bounds__(288,2)`，grid = N/32）：每个 CTA 认领 32 行权重 × 全部 K × mpad = min(M,16) 个 token，M 方向不切分；tid256 producer 用 TMA 把 W（SW128 4KB）、SF（512B）、A（4 个 MP×128B 盒）灌进 8 / 4 深的环，8 个 mma warp 按 mb = warp&1（16 行权重）× kh = warp>>1（16 个 kb 中的 4 个）切分，权重现场解码成寄存器 a 片段、token 用 `ldmatrix x4/x2` 取成 b 片段（操作数互换），`mma.sync` 累加到 16 个 fp32 寄存器；结束后 kh ≥ 1 的 warp 把累加器写进 12KB smem 暂存区，kh = 0 的两个 warp 合并 3 个伙伴并按 C[token][wrow] 逐元素 scatter 写回。保留理由：工作单元是权重 band 而非输出 tile、操作数互换、输出经跨 warp smem 归并后转置写出，与其余方案无重叠。 | 并列完整设计，只共享 root 的 Contract，三区全部本地定义。when：m ≤ 16（mpad = min(m,16) 个 token 行入 stage，≥ m 的 token 写回丢弃），并要求源码的 cc 10.3 门槛、bulk tensor-map 拷贝、warp 级 BF16 矩阵乘与 82304 字节动态 smem 支持每 SM 双 CTA 共驻；无 cluster、tensor memory、全局工作区或第二次提交。六条记录以它为基线：持久 tile/band 循环与跨工作环复用、f16 操作数链与 A 转换 prepass、宽盒 BK512 三维 TMA 填充、token 块 MP 与环深度、权重 band 宽度与 kb 切分、cfg217-219 组合 |
| [simt_warp_row](simt_warp_row.yaml) | `gv::gv_kernel<1,NB>`（cfg170，NB = K/512 ∈ {5, 8, 24}，256 线程，grid = ceil(N/8)，无动态 smem）：不使用张量核心、smem、屏障、TMA 与 TMEM。一个 warp 拥有一行权重，32 个 lane 各取 8 字节 packed 权重与 1 个 e4m3 尺度并整行 LDG 预载进寄存器；按 16 元素尺度组做 `cvt.rn.f16x2.e2m1x2` → `cvt.f32.f16` 精确解码，与 bf16 激活的 fp32 值做两条奇偶 FFMA 链，尺度以 fp32 折进块和，块间顺序累加；最后 warp shuffle 归约，lane0 乘 alpha 转 bf16 写 C[n]。保留理由：唯一不用张量核心的完整方案，主计算、工作归属（warp = 一行权重）与输出形成（shuffle 归约 + 单点写）都独立成立。 | 并列完整设计，只共享 root 的 Contract，三区全部本地定义。when：m == 1、k/512 ∈ {5, 8, 24}；只要求每 lane 寄存器、精确窄格式解码与 warp 集体归约，未写 cc 门槛——机制本身不依赖 10.3 特性，auto 分派到达它仍经 cc == 10.3 的 selector 分支。两条记录以它为基线：TMA slab 输入物化、激活整行寄存器预载 |

## 变体与组合

以下十五条记录按局部、参数、组合排列。每条直接定位完整基线及其中已有对象；组合条目直接列出完整基线，所叠记录按名称互链，不层层组合；条目排列与引用不产生 patch 应用顺序，也不修改基线的参数或 when。

### 持久 tile / band 循环与跨工作环复用（局部）

- 基线：[umma_single_cta](umma_single_cta.yaml) 的 MAIN step（`Worker_count: worktile_count`、`Preassigned {start: worker_id, count: 1}`、`Refill: none`）、Tile_pipeline 的 `a_ring`/`bp_ring`（slot t%4）与 `bdq_ring`（slot t%2）、DECODE 的 `compute:barrier_init`/`acquire:tmem`；[weight_band_mma](weight_band_mma.yaml) 的 MAIN step（同形 Preassigned/Refill）、LOAD 的 `w_ring`（slot t%8）/`sf_ring`/`a_ring`（slot t%4）、MMA 的 `acc` 与 `stg`、EPI 的 `await:staged`。
- 差异与联动：grid 由 `Worker_count: worktile_count` 改为 min(工作数, sm_count()) 或 min(bands, 2·sm_count())，工作归属由每份工作一次 `Preassigned {count: 1}` 改为 stride 式持久领取（pair_two_pass MAIN 的 `Preassigned {start: worker_id, stride: worker_count}` 同形）加 CTA 内 tile/band 循环（`work += gridDim.x`），环 slot 索引由工作内 t 改用跨工作连续的全局 stage 计数，producer 因此能跨工作边界提前预取（umma 侧提前 A_STAGES 级），屏障不重新 init（`compute:barrier_init` 每 CTA 只一次）、parity 按 (t/RINGS)&1 继续翻转；累加器 `acc` 每个工作复位，tmem alloc（`acquire:tmem`）、屏障 init 与首段冷飞行每 CTA 只付一次；单缓冲的累加器与合并暂存区（weight_band 侧的 `stg` 与 EPI `await:staged` 合并门）新增 per-work 释放门，umma 侧有两种实际形式——`ws26`（cfg76）的 epi 屏障计数 8，由 8 个 epilogue warp leader（(tid&31)==0）在读完 TMEM 后到达，issuer 在 j>0 时等它再发本 tile 的首条 UMMA；`ws10fw`（cfg249）的 epi 屏障计数 256，由全部参与 epilogue 的解码线程在该 band 的 TMEM 读出后逐个到达；band 引擎侧则由持久 smv 族共用双缓冲 stg_done[2] 作每 band 合并门（`smv10`/`smv11`/`smv13` 计数 4 = kh=1 的四个 warp leader，`smv12` 计数 6；索引 jb&1、parity (jb>>1)&1，warps0-3 先等它再合并伙伴并写回），`smv10`/`smv11` 另用每 band 两次 `__syncthreads`（合并点与 band 末）保证暂存区复用；非持久成员（如入选的 `smv9`，即 weight_band_mma 基线）的 Cfg 没有 stg_done 槽，只在合并点用一次 `__syncthreads`（即基线唯一的 `await:staged` 合并门）。
- 条件与依据：适用于工作数超过驻留槽数的多波形状，不超过时源码称为结构性 no-op；域内 cfg76（`ws26`，bf16）、cfg249（`ws10fw`，f16）与 cfg217-219（`smv13`，N=24576 时 384 band 对 296 槽）采用；源码对 `ws10fw` 与 `smv10`/`smv11` 明确记录持久化前后逐位相同，`ws26` 未给数值等价声明，但其函数体已核对：UMMA 的累加标志用 tile 内相对 t（t==0 且 i==0 时清零累加器）、环索引 g 跨 tile 连续、issuer 在 j>0 时等 epi_b，故每 tile 的 k 序与指令序与非持久形式一致；这一条是本次读源码的结论而非源码声明，采用该机制的设计须按自己的定义核对。

### cluster pair 内联解码（局部）

- 基线：[pair_two_pass](pair_two_pass.yaml) 的两次提交结构：Work_schedule 的 DEQUANT step 与 `Flow.Handoff.wq`（storage: global）、MAIN step 的 `wq_block` 迭代块；Tile_pipeline MAIN 的 `b_ring`（存已解码 bf16 行）、PRODUCER 的 `load:wq_block(g,h)`、SIGNALER 的 `signal:pair_ready`、EPILOGUE_HALF0/EPILOGUE_HALF1 的角色划分，及 REALIZATION 中 PSS 启动属性的登记。
- 差异与联动：取消 DEQUANT step、`Flow.Handoff.wq` 与全局 B 工作区（Bws），`Flow.Launch` 收回单次提交；改由 TMA 灌 packed B——`b_ring` 换成 BP 环（2 槽，每槽 2×128×64B）另加 BDQ 环（2 槽 64KB），PRODUCER 的 `load:wq_block(g,h)` 换成 packed 字节盒装载；warps0-7（基线 EPILOGUE_HALF0/HALF1 的成员）兼作解码角色，先把 4 个 uint4 全读进寄存器就立刻 arrive empty（early-empty-arrive，提前一个转换+存储尾巴释放槽位），再写 BDQ 环；deq 屏障满 256 到达后由本 CTA 的 tid0 用 `mapa` 远程 arrive rank0 的 pair 屏障，取代专职 SIGNALER；e4m3 尺度不进 smem，按行 LDG 并预取下一 stage；BK 64→128（`bk` 与 `k_tile_count` 联动）、RING 4→2、无 PDL（REALIZATION 的 PSS 启动属性取消）、无专用 signaler；epilogue 回到解码 warp 内联执行（每 pair tile 末尾等 bdq 屏障后读 TMEM）。
- 条件与依据：域内 cfg94（`ws27e`）由 ws13_big 分支中 pairs13 ≤ sm_count()/2 且 K ≥ 4096 选中，K2560 走 cfg119（即 pair_two_pass 基线）；cfg77/78/96/98/100（`ws27`）是同一机制的强制对照臂，源码记录 cfg94 与 cfg77 逐位相同、只改释放时机。

### 专用 epilogue warp 与 N 分半释放（局部）

- 基线：[pair_two_pass](pair_two_pass.yaml) 的角色划分：EPILOGUE_HALF0（Member warp[0:4]）与 EPILOGUE_HALF1（Member warp[4:8]）、二者按半区的 `release:acc(t)` 与 ISSUER 按半区门控的 `await:acc_free(t,h)`、`acc` 的半区视图 acc(t,0)/acc(t,1)，以及 REALIZATION 登记的 320 线程与“每成员两个半区读取计数 4、member 0 两个半区累加数组计数 2”。
- 差异与联动：可与「cluster pair 内联解码」并列叠加在同一 cluster pair 骨架上；`ws29` 在源码中的直接基线是已含内联解码的 `ws27`（warps0-7 在那里已是解码角色），所以从 pair_two_pass 走到 cfg101 需要上一条（取消全局 deq pass）与本条两个差额并列，不能只记一条。本条的变化：THREADS 320→576，tid320-575 是 8 个只做 epilogue 的 warp（承担基线 EPILOGUE_HALF0/HALF1 的职能），与解码流并行排空；TMEM 的两个 256 列半区（基线 acc(t,0)/acc(t,1)）各自在读取完成后释放（warps10-13 → epip[0]，warps14-17 → epip[1]，对应基线两个半区的 `release:acc`），issuer 按半区分别门控（同基线 `await:acc_free(t,h)`）；epil 计数 4（半区 warp leader）、epip 计数 2（两 CTA 的 tid320/tid448 远程 arrive 到 rank0 宿主，与基线 member 0 两个半区累加数组的计数一致）；寄存器上限 113/线程。
- 条件与依据：域内 cfg101（`ws29`）由 ws13_big 分支中 pairs13 > sm_count()/2 选中；源码记录数值与 `ws27`-v8 逐位相同（同数据、同转换、同 alpha、同地址），只改角色划分与释放时机。

### f16 操作数链与 A 转换 prepass（局部）

- 基线：[umma_single_cta](umma_single_cta.yaml)、[weight_band_mma](weight_band_mma.yaml)、[split_k_reduce](split_k_reduce.yaml) 三份的 A/B 操作数格式：各自 Math_reshape 的 `wq`（bf16）与 `acc` 的 mma32 乘积定义、Tile_pipeline 的 bf16 `a_ring` 与解码操作数存储（umma_single_cta 与 split_k_reduce 的 `bdq_ring`，weight_band_mma 的寄存器 `wq_frag` 及 MMA 的 `matmul:wq_frag@a_block`），以及各自 REALIZATION 对 A/激活 tensor map 的登记（umma_single_cta 与 split_k_reduce：a as bf16 over (k, m)；weight_band_mma：mpad 行高的激活盒）。
- 差异与联动：新增一个同流 prepass step（在各基线的 `Flow.Launch` 前多一次提交）把 A 由 bf16 转 f16 写进进程级 grow-only 全局 scratch（`smv6::deq_a16_kernel` grid-stride、每线程一个 `__nv_bfloat162`；或 `ws27f::a2f_pass_kernel`、每线程一个 uint4），主 kernel 的 A tensor map 改指该 scratch；B 解码链去掉 `f16x2_to_bf16x2` 一步、`wq`（及 `bdq_ring`/`wq_frag` 内容）直接存 f16 乘积，IDESC 的 a_format/b_format 位改为 f16（`mma_f16` 取代 `mma_bf16`）；环字节数不变（f16 = bf16 = 2B），`mma_f16` 与 `mma_bf16` 的片段布局与发射率相同；scratch 扩容须调用 `v20_invalidate()`，复用依赖流序。
- 条件与依据：域内 cfg217/218/219（prepass trig=0、主 kernel 普通启动）与 cfg249（PSS 启动）走这条链，cfg128-131/154/204-207 为强制或未选中臂；源码声明冻结分布上 bf16→f16 精确（A 的幅值 ≤ ~350 ≪ 65504、零点映零点），仅幅值小于 2^-14 的 subnormal 区间有 ≤1 ulp tie-flip，属源码主张、本次未独立验证。

### W-first preflight（局部）

- 基线：[umma_single_cta](umma_single_cta.yaml) 的 PRODUCER `load:a_block`/`load:b_block`（一个 40960 字节公告计数覆盖两盒）、ISSUER 的 `await:stage_full`（等该单一计数落齐）、DECODE 的 `compute:barrier_init`（四个通知数组的到达计数）与 REALIZATION 的 tensor map 及计数登记。
- 差异与联动：与「f16 操作数链与 A 转换 prepass」「持久 tile / band 循环与跨工作环复用」两条叠加使用，所等的 `griddepcontrol.wait` 来自 f16 链引入的 prepass 提交。full 屏障到达数由 1 拆成 2（W 盒与 A 盒分别 arrive，`expect_tx` 字节数分别记，总字节数不变），producer 在 `griddepcontrol.wait` 之前先发出前 A_STAGES 个 stage 的 W 盒，使不依赖 a2f 的 W 冷飞行与 prepass 重叠；g ≥ 4 与后续 band 仍走两次 arrive 的正常路径。
- 条件与依据：仅 cfg249 采用，只改启动重叠顺序，源码记录与未 preflight 版本逐位相同。

### TMA slab 输入物化（局部）

- 基线：[simt_warp_row](simt_warp_row.yaml) 的每 lane 权重/尺度读取路径：Tile_pipeline 的整行寄存器预载存储 `w_row`/`s_row` 与 MAIN role 的 `load:b_chunk`/`load:s_chunk`，及其 REALIZATION 的整行寄存器预算；解码-FFMA-归约-写回主体（Math_reshape 的 `wq`/`sf`/`part`/`total` 与 `compute:wq_chunk`/`compute:group_acc`/`compute:row_total`/`store:out_element`）不变。
- 差异与联动：整块 W + SF 改由 TMA 3D/2D 盒搬进 smem slab（tid0 发、`expect_tx` 记数、full mbarrier 等待），每 lane 从 smem 读自己那 8 字节，取代 `load:b_chunk`/`load:s_chunk` 的整行 LDG 预载；A 仍按 lane LDG（L2 命中），`load:a_block` 不变。每 CTA 的盒数按成员不同：`gvt` 2 个（W 一个 3D 盒 + SF 一个 2D 或 3D 盒，单个 full mbar、expect_tx = NB·2304）；`gvt2` 与 `gvt4` 各 3 个（W 拆两个半盒 + 一个 SF 盒，两个 mbarrier）；`gvt3` 为 S+1 个（S 个 W 盒按块区间拆分、SF 随第 0 块，S=3 时 4 个、S=4 时 5 个），故该族的实际范围是 2-5 个盒。越界处理也随之改变——grid 仍覆盖 ceil(N/8) 或 ceil(N/16)，OOB 行由 TMA 零填充，但只有 `gvt`(v1) 的越界 warp 在等完 full 屏障后直接返回（全文件的 `n >= N` 早退只有它与基线 `gv` 两处）；`gvt2`/`gvt3` 的越界 warp 走完整条解码-FFMA-shuffle 链，只在写回处按 n < N 守卫那唯一一次 C 写，与 `gvt4` 对 n0 与 n0+1 两行各守卫一次是同一形式；基线 `gv` 没有屏障，在计算前就早退。slab 拆 2 半（`gvt2`，前半解码与后半冷飞行重叠）、拆 S 路（`gvt3`，首等待缩到约 1/S）、或每 CTA 16 行 / 每 warp 2 行（`gvt4`，grid = ceil(N/16)，两个累加器与两轮 shuffle 归约，slab 行距 256B → 4096B/chunk）。
- 条件与依据：域内 m = 1 且 N·K > 32e6 选 cfg248（`gvt4`，NB ∈ {5,8,24}），cfg236/237/238/245/247 是同族强制或未选中臂；源码记录各成员与 cfg170 的逐位相同门槛（block 顺序升序、同一条 fp32 链），smem 用量随 NB 与行数变化（NB8 18440B → `gvt4` NB24 109.4KB）。

### 宽盒 BK512 三维 TMA 填充（局部）

- 基线：[weight_band_mma](weight_band_mma.yaml) 的 bk = 256（MAIN Decomposition 与 `k_tile_count`）、`w_ring`（8 槽、32 行 × 128B）/`sf_ring`（512B/槽）/`a_ring`（4 槽、16×256）存储、LOAD 的 `load:w_block`/`load:scale_block`/`load:a_block`（A 每 k 块 4 个盒）与 REALIZATION 的事务字节计数登记。
- 差异与联动：BK 256→512（`k_tile_count` 与 Math_reshape 的 `block_count` 联动减半），W 与 A 各改用一个 3D 宽盒（W：128B × 64 行 × 2 chunk，SW128，z-major 拼成两个连续 SW128 半区；A：64 个 f16 × MP 行 × 8 chunk，SW128，globalDim1 = M 使 M..MP-1 行 OOB 零填充），SF 仍是 2D 盒但由 16B 加宽到 32B（SWIZZLE_NONE）；每 512 个 k 的填充由 12 次降到 3 次，`expect_tx` 等于盒字节数，stage 数 T = K/512。W/SF 宽盒的 64 行即 BNw = 64 取值（见「权重 band 宽度与 kb 切分（参数）」），A 盒的 f16 来自「f16 操作数链与 A 转换 prepass」，生产采用形与两者同现。
- 条件与依据：只在 K%512 == 0、K ≥ 4096 且 band 数 N/64 ≥ sm_count() 时启用，域内 (12288,4096) 与 (24576,4096) 的 2 ≤ M ≤ 32 走 cfg217/218/219；源码记录 K2560（T = 5 浅流水）与薄 grid 实测变差而被排除，深环强制臂 222-224 也不能翻转。

### PDL 启动配对（局部）

- 基线：[pair_two_pass](pair_two_pass.yaml) 与 [split_k_reduce](split_k_reduce.yaml) 相邻两个 step 之间的提交配对：前者的 `Flow.Launch` 两个条目（{step: [DEQUANT]} → {step: [MAIN]}）、`Flow.Handoff.wq`（wait_at: MAIN.wq_wait）、MAIN 的 `Event.wq_wait` 与 PRODUCER 的 `await:deq_grid`；后者的 `Flow.Launch` 中 {step: [REDUCE], start_after: MAIN.launch_ready}、`Flow.Handoff.part`（wait_at: REDUCE.part_wait）、REDUCE 的 `Event.part_wait` 与其 MAIN role 的 `await:grid_ready`。
- 差异与联动：后一个 kernel 以 `cudaLaunchAttributeProgrammaticStreamSerialization` 启动（两份基线的 REALIZATION 已把该属性与对应等待登记为启动事实），前一个 kernel 不做显式 `griddepcontrol.launch_dependents`，提前退出的 CTA 在退出处隐式触发，wait 在前一 grid 全部完成时才解析；后一个 kernel 的 producer 或归并线程先 `griddepcontrol.wait`，即两份基线的 `await:deq_grid` / `await:grid_ready` 动作；没有该属性时 wait 是 no-op（流序已保证前序完成）。部分强制臂把触发点提前：deq kernel 带 trig 参数，在自身 store 后 `__threadfence` + `griddepcontrol.launch_dependents`，使释放点从 grid teardown 提前到每 CTA 的 store 完成（cfg120/179 用之；pair_two_pass 的 deq_pass2_kernel 实例传 trig = 0，split_k_reduce 的前导 MAIN kernel 则根本没有触发指令）。下一次调用的前序 kernel 仍按流序排在其后，因此工作区复用安全。
- 条件与依据：由 cfg 决定而非参数——cfg119 开（cfg123 为无 PDL 对照）、cfg144 的归并 kernel 开（cfg148 为对照）、cfg176 关（cfg175/179 开）、cfg249 开；源码记录只改启动重叠、数值逐位不变，Contract 的 enable_pdl 从不进入 `launch_mm`。pair_stream_k 是本记录的负向兄弟：其 `Flow.Launch` 无 start_after，REALIZATION 明记无 serialization 属性，`await:deq_grid` 依流序立即解析；umma_single_cta 只声明一次提交（`Flow.Launch` 仅 {step: [MAIN]}），没有相邻 step 可配对，PDL 只经 cfg249 组合记录到达——该组合新增 a2f prepass 作第一次提交并以 PSS 启动主 kernel。

### 激活整行寄存器预载（参数）

- 基线：[simt_warp_row](simt_warp_row.yaml) 中激活的读取方式：Tile_pipeline 的 `a_chunk` 存储（depth 1 的单 chunk 槽）与 MAIN role 的 `load:a_block(i)`（在使用点发起）、Pipe 中紧随其后的 `compute:group_acc(i)`，及 Work_schedule MAIN 的 Worktile Iteration `a_block`（{k: 512}）。
- 差异与联动：对应源码模板参数 AP = 1：在 `load:b_chunk`/`load:s_chunk` 权重预载的同一前置循环里把整行激活也读进寄存器（areg0/areg1 各 NB 个 uint4，NB8 时 +64 寄存器），`a_chunk` 由单 chunk 槽扩为整行双数组；内层循环的 `compute:group_acc(i)` 从寄存器取值而不再逐块 LDG；仅在 MP == 1（即基线 when 声明的 m == 1）时启用，其余情况保持已验证的流式形式。
- 条件与依据：只对应 cfg171（强制臂，auto 只选 cfg170）；源码记录的动机是每 warp 重复读同一份 A 字节带来的 L1 广播压力，图复放白名单对 cfg170/171 用同一条件（N ≥ 8192 且 N·K ≤ 32e6）。

### token 块 MP 与环深度（参数）

- 基线：[weight_band_mma](weight_band_mma.yaml) 的 MP 选定值：Work_schedule MAIN Decomposition 的 “mp = 16; mpad = min(m, mp)”、Math_reshape 的 `mp` 符号（bind:DESIGN.Work_schedule::mp）与 step 2 乘积的 mp token 维、Tile_pipeline 的 `w_ring`（深度 8）/`a_ring`（深度 4）、LOAD 的 `load:a_block`（mpad 行高的盒）、EPI `store:out_band` 的 token 守卫，及 `when` 的 m ≤ 16。
- 差异与联动：MP ∈ {8, 16, 32} 决定 A_STAGE = MP×BK×2、mpad = min(M, MP)、`ldmatrix` x2/x4 的选择与写回 tile 数；A_RINGS（`a_ring` 深度）随 MP 取 8/4/2，W_RINGS（`w_ring` 深度）可加深（`smv5` 的 WR 模板参数 8→10/12/16，深环使 producer 免去 empty 等待）；A tensor map 的 box_rows = mpad，OOB token 行由 TMA 零填充并在写回处被 j0 < M 守卫丢弃（与基线 EPI `store:out_band` 的“只写小于 m 的 token”守卫同形）。
- 条件与依据：域内按 M 类选 cfg191（MP8）/192（MP16）/193（MP32），cfg231 = `smv14`（MP32 + A_RINGS 4，单 CTA/SM 类），cfg138-143/204-216 为强制或未选中臂；源码记录同一引擎不同 MP 之间数值同类而非逐位相同。

### 权重 band 宽度与 kb 切分（参数）

- 基线：[weight_band_mma](weight_band_mma.yaml) 的 BNw = 32 选定值：Work_schedule MAIN Partition 的 n tile 32（grid = n/32）、Tile_pipeline 的 `w_ring`（32 行 × 128B = 4KB/槽）/`sf_ring`（512B/槽）/`acc`（view 按 quarter 把 256 线程分四段各 64 线程）/`stg`（3 × 64 × 16 fp32 = 12KB）、MMA role 的 “warp w takes sub-band w mod 2 and k quarter w div 2” 切分与 Math_reshape step 2 的 `quarter_count = 4`、EPI（Member warp[0:2]）的 `compute:total` 三伙伴合并。
- 差异与联动：BNw 32↔64 使 W_STAGE 4KB↔8KB、SF_STAGE 512B↔1KB、grid = N/BNw 减半，8 个 warp 的切分由 mb = warp&1 × kh = warp>>1（4 路 kb、3 伙伴合并、12KB `stg`）变成 mb = warp&3 × kh = warp>>2（2 路 kb、1 伙伴合并、8KB 暂存），合并 warp 数（EPI 成员）由 2 变 4，step 2 的 kb 切分与合并次序随 `quarter_count` 由 4 联动为 2 路。
- 条件与依据：累加集合相同但部分和结合次序不同，源码记录为容差内（mismatch == 0）而非逐位相同；域内 BNw = 32 的成员是 cfg191-193，BNw = 64 的成员是 cfg231 与 cfg217-219，BNw = 32 的动机是薄 grid 的波量化（CTA 数翻倍填满机器）。

### tile 形状、环深度与调度旋钮（参数）

- 基线：[umma_single_cta](umma_single_cta.yaml)、[pair_two_pass](pair_two_pass.yaml) 与 [pair_stream_k](pair_stream_k.yaml) 的 tile 常量、环深度与调度参数选定值：umma 侧 bk = 128 与 128×128 Partition、`a_ring`/`bp_ring` 深度 4、`bdq_ring` 深度 2、Worktile_order 的 group: 8、DECODE 的 `store:out_tile` 及其 Prefer 登记的 2×32B 向量存储、REALIZATION 的 A/B tensor map 盒与 swizzle；pair_two_pass 侧 bk = 64 与 256×512 Partition、`a_ring`/`b_ring` 深度 4、Worktile_order 的 group: 8、EPILOGUE_HALF0/HALF1 的 `store:out_chunk` 2×32B 向量存储；pair_stream_k 侧的受影响对象是 MAIN Mapping.Strategy 中的 group_m = 8（Decomposition.Strategy 登记同名常量）——该设计唯一的 Partition 轴是 slot，group_m 出现在 p → blocks 的双射枚举里，因此这一参数在此基线不落在 Worktile_order 的 group 条目中，其 GM16/GM4 对照臂为 cfg177/178。
- 差异与联动：BK 64/128（各自 `bk` 与 `k_tile_count`）、A/BP/BDQ 环深度（如 4/4/2 与 4/2）、BM/BN 或 BMp/BNp（128×128、256×512，即 Partition tile 值）、group_m ∈ {1,2,4,8,16,32}（gid → tile 的遍历次序；在 umma_single_cta 与 pair_two_pass 侧是 Worktile_order 的 group 值，在 pair_stream_k 侧是 Mapping.Strategy 的 group_m；pair_two_pass 的发射按 m_pairs ≥ 10 在 8 与 16 之间选，pair_stream_k 固定取 8，各基线 YAML 选定 8）、st8 存储宽度（4×16B ↔ 2×32B `st.global.v8`，即 `store:out_tile`/`store:out_chunk` 的两种形式）、tensor map 的 L2 promotion（默认 128B，pair 引擎与 ws13 系改 256B）。
- 条件与依据：均为调度与访存粒度旋钮，源码记录 group_m 与 st8 的变化逐位相同、L2 promotion 是每次 launch 前重置的全局量；强制对照臂按基线归属：cfg124-127/262 对应 pair_two_pass（GM2/4/1/16 与 GM32 探针），cfg177/178 对应 pair_stream_k（GM16/GM4），cfg54/55/73 对应 umma_single_cta（GM4/GM16 与 L2 promotion 臂）。

### split-K 路数与代价模型（参数）

- 基线：[split_k_reduce](split_k_reduce.yaml) 的 sk 选定值：Math_reshape 的 `sk`（value: 2）、`seg_groups = group_count / sk`、`part`（{seg: sk} 平面）、Work_schedule MAIN 的 Partition `seg` 与 Schedule 的 “worker_id gives seg = worker_id mod sk”、REDUCE step，及 REALIZATION 的 `sk*m*n*4` 字节工作区登记。
- 差异与联动：sk 取 2..8 中整除 Ta 且 Ta/sk ≥ 8 的值（Ta = K/128），由 host 代价模型 cost(c) = waves(c)·(Ta/c·1200 + 2400) + c·M·N/500 选出，waves(c) = ceil(tiles10·c/148) 把 SM 数写成常量 148，且需比 sk1 好 5%；启用门槛为 tiles10 < 148 或（148 ≤ tiles10 ≤ 192 且 Ta ≥ 96）；P 大小 = sk·M·N·4B（工作区字节随之联动），归并 kernel 的 total4 = M·N/4，REDUCE 的工作数与 `part` 平面数随 sk 联动。
- 条件与依据：域内 Ta ∈ {20, 32, 96}，故 K = 2560 只可能取 sk = 2；sk = 1 或 scratch 分配失败时不启动归并 kernel，整条链退化为 umma_single_cta 的单 kernel 形态（split_k_reduce 的 when 与 REALIZATION 均已登记该退化）；域内由 m > 32 的第三个分支选中（cfg144），cfg145/146/148-155 是固定 sk、f16 部分结果与 PDL 对照的强制臂。

### cfg249：持久循环 + f16 链 + W-first preflight + PDL（组合）

- 基线：[umma_single_cta](umma_single_cta.yaml)（完整基线，直接定位），叠加「持久 tile / band 循环与跨工作环复用」「f16 操作数链与 A 转换 prepass」「W-first preflight」「PDL 启动配对」四条局部记录。
- 差异与联动：host 链为 `ws27f::a2f_pass_kernel` → `ws10fw::w4a16_ws10fw_kernel`（PSS 启动，grid = min(tiles, sm_count())），即在基线的单提交 `Flow.Launch` 前新增 a2f prepass 提交、主提交改 PSS 启动（PDL 配对以它为第二次提交）；A tensor map 指向 f16 scratch、IDESC 全 f16，full 屏障拆成 W/A 两次 arrive；四项接口相容（环索引连续性与 f16 scratch 无冲突，preflight 只改到达计数与发射顺序）。
- 条件与依据：域内由 ws13_big 的多 tile 分支与 ws10_big 的多波分支按 wall 模型选中（K ≥ 8192 时先与 cfg176/cfg119 比较），不在图复放白名单内；源码记录与 cfg28 属容差同类（f16-A 下溢 ~2.4e-7/元素，rel_l2 ≤ 1e-5 门槛）。

### cfg217-219：宽盒 BK512 + f16 链 + 持久 band 循环 + MP 类（组合）

- 基线：[weight_band_mma](weight_band_mma.yaml)（完整基线，直接定位），叠加「持久 tile / band 循环与跨工作环复用」「f16 操作数链与 A 转换 prepass」「宽盒 BK512 三维 TMA 填充」「token 块 MP 与环深度」四条记录；BNw 取「权重 band 宽度与 kb 切分（参数）」的 64 值，宽盒记录的 64 行 W/SF 盒即该值。
- 差异与联动：host 链为 `smv6::deq_a16_kernel`（trig = 0，无 PDL 触发）→ `smv13::w4a16_smv13_kernel<MP,0,0,0,0>`（普通启动，grid = min(N/64, 2·sm_count())），即在基线的单提交 `Flow.Launch` 前新增 f16 prepass 提交、两次提交都不带 PSS 属性；W 与 A 各一个 3D 宽盒、SF 一个 2D 盒（32B × 64 行），A tensor map 指向 f16 scratch，`mma_f16` 取代 `mma_bf16`；四项接口相容（宽盒只改 BK 与盒形状，f16 链只改操作数格式与 A 来源，持久循环只改 grid 与 band 遍历，MP 只改 A stage 与 mpad）。
- 条件与依据：域内 (12288,4096) 与 (24576,4096) 的 2 ≤ M ≤ 32 按 M 类选 MP8/16/32；在图复放白名单内，因此第二次同 key 调用起这两个 kernel 以一次图复放提交。
