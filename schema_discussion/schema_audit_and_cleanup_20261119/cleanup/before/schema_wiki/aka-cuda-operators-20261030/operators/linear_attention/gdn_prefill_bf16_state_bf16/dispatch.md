# gdn_prefill_bf16_state_bf16

## 范围

收录固定实现（KernelSchema@24d2b3d3ca9d4907c4e7efd9607cf78cfb1a52b2，candidate/）的生产构建全部分派路径：solution.py::Model.forward 仅在 cu_seqlens 为 None 时合成标准 chunk 元数据，随后调用 gdn_kernel.cu::gdn_forward；设备侧为可选 l2norm_qk、gdn_prep（源码称 P1，准备）与 gdn_chain（源码称 P2，链）三个 kernel。host 分派只依赖张量元数据与运行时查询的 sm_count、L2 字节数，无 D2H 同步。problem/input.py 的 prepare 是调用前的样本状态构造（32 个历史 token），不属于 Model.forward 计时计算；编译主要在 Model 构造阶段，forward 内仍含连续化、内部缓存、tensor-map 查询和结果分配。下文的 gdn_kernel.cu / gdn_common.cuh / solution.py 均指该固定来源 candidate/ 下文件，行号为固定版本行位。

主要机制（两设计共享的骨架）：64-token 分块块解是 Contract 逐 token 递推的代数等价形式。gdn_prep 以 256 线程 CTA 按 (chunk 槽, 头) 工作：4 个 SWIZZLE_128B TMA box 装载 k/q tile，cp.async 装载 g/beta（行不足谓词零填），warp 洗牌 cumsum 得 b 与 rho[j]=exp(b_{j-1}-b_j)，bf16 mma（fp32 累加）算 gram=bf16(K@K^T) 与保留 FP32 的 qk=Q@K^T，构建 system=bf16(I+tril(beta_i*exp(b_i-b_j)*gram,-1))，按 8/16/32/64 四级层次化块求逆得单位下三角 Tinv（8 级为一 lane 一列的寄存器前代；16/32 级为 Z=invD2@L21、X=-Z_hi@invD1 的单遍融合 mma，C 碎片寄存器直接重打包为 A 碎片；64 级为 Y=L21@invD1 写 bf16 平面后 X=-invD2@Y），向工作区槽按 (slot,hv) 发布三角打包的 TinvH 平面、QkdH=QSCALE*qk 的下三角衰减掩码平面（各 10 个 512B 半列分裂 tile）与 b/beta/Ai=exp(b_i)/Dl=exp(b63-b_i) 四组 fp32 向量，并把 32768B k/q swizzle 镜像 s2g 到 scratchq（KQTMA 档跳过）；启用 LPT 时 block(0,0) 顺带产出请求排列 perm。gdn_chain 每 CTA 负责 (request n, value head hv, v-tile vt) 的整条 chunk 链：fp32 状态 S[128,VTT] 按 step-8 累加器归属常驻寄存器 Sreg，sShi 是其 hi bf16 平面视图；NSTAGE 级 cp.async.bulk/TMA stage 按 mbar 奇偶相位双/三重缓冲装载 KQ 镜像、v box、平面与向量。每 chunk 依次：step2 kS/qS=[k;q]@S_hi → step3 rhs=beta*(v-Ai*kS) → step4 u=TinvH@rhsH → step5 du=Dl*u（输出启用时另写 u 平面）→ step6 attn2=QkdH@u → step7a o=Ai*QSCALE*qS+attn2 暂存 sRHh → step8 dS=K^T@du、S=exp(b63)*S+dS 并重建 sShi → step7b warp 本地 __syncwarp 后合并 STG.128 写 out（及 core）；链尾把 Sreg 经 smem 分级暂存转置写 final_state[n,hv,v,k]。两 kernel 以流序 + PDL 协作：prep 在其装载落地后显式 griddepcontrol.launch_dependents，chain 预启动执行只读输入的序幕（mbar 初始化、无 perm 时含 sreg_load，PRE0 档另发 chunk-0 box），griddepcontrol.wait 绑定 prep 网格完成后才读 perm/平面/scratchq。数值表示：q/k/v 为精确 bf16 进 mma；fp32 派生量（gram、system、状态平面、Tinv、rhs、u、du、Qkd）量化为单一 hi bf16 平面参与乘积（源码 V2f/V3d/V3e/E16 系列已移除 lo 残差平面，声明舍入类 ~2^-9；gdn_common.cuh 头部注释仍称 hi+lo 双平面 ~2^-17，为过期文本，数值事实以生产代码的单 hi 平面为准），mma 累加与跨 chunk 权威值保持 fp32（Sreg），final_state 直接来自 Sreg。host 资源机制（grow-only 持久工作区与 norm 缓冲、按 (指针,几何) 键控且仅含地址布局元数据的 tensor map 缓存、prep launch 后从单次新分配切出 out/final）不是独立设计轴：作为实现条件，root 已写入其 Realization（同流串行复用条件），segmented 的 REALIZATION 也已按 b_seg/a_seg 寿命（perm 尾垫之后的独立 transfers 区、pass A 网格完成后 pass C 才读）与 A→C 的 PDL 绑定独立填写。

边界与覆盖：GDN_HOSTPROF / GDN_P1PHASE / GDN_CPHASE 与 GDN_*_OFF、GDN_E1_SIMT16/32、GDN_E6_SERIAL_WAIT 等编译旗标只产出计量或归因回退文本，生产构建不编译（solution.py 编译选项不含这些宏），不属收录对象；chain 另支持 fp32 initial_state（state_is_fp32），任务域固定 bf16，按来源事实保留不进 Contract。仍在交付中的已知来源缺口：out/final_state/core_attn_out_updated 的验收容差为 source-only 记录（外部比较器的公式、基准与逐输出路由缺失，见 root.yaml 的 Golden_reference.Tolerance）；host 实现要求 cu_seqlens 为张量时 chunk_indices/chunk_offsets 必须一并提供，Contract 允许的“cu_seqlens 为张量而 chunk 元数据为 None”组合被拒绝。此 None 分支依据 reference.Model.forward 的默认参数及其完全不读取 ci/co、仅按 cu 切块的行为；标准非 None 枚举依据 input.py。候选对非连续 core 调用 contiguous()，之后没有 copyback，可丢失调用者写回；这些是实现覆盖缺口，Contract 的可空分支与前缀写回要求保持不变。设备身份（源码注 L20D/sm_103）与 SM 数、L2 容量、opt-in SMEM 上限等硬件事实由后续硬件绑定阶段登记，本目录分派条件只引用 sm_count 与 l2_bytes 两个运行时查询值（源码 census 注释按 148 SM 举例）。任务要求实现的范围即 Contract 有效域；本目录两份完整设计与下列记录覆盖 host 分派的每个选择分支（l2norm 旗标、vt16/vt32/vt64 档、vt64 迷你 chunk、Gseg 分段双臂、NSTAGE 深度、KQTMA、PRE0、perm、GQA/ratio/锁步、qhfast 次序），无未定去处的机制，也没有已研究但未完整展开的独立方案。两份设计的 when 均为已核对的保守子域：Contract 域内存在 GQA 与分段并行同时成立的可达成局（源码认证的分段形状 c8 即 gqa=true 且 Gseg=4），两份 when 都不承担该格局，它由“GQA 共享 prep”记录沿两条基线定位。

当前数值边界：g/beta 等未额外约束的浮点输入遵循 Contract 的任意有限值默认域，生成样本不隐含 |g| 或 |b| 的上界。零填 g/beta、TMA 边界零填和 i<Lc 输出掩码都有实际代码依据，但有限输入仍可能在归约、指数或状态乘积中溢出；0*inf 不会被 beta=0 消除。64 行 shuffle 扫描的零填尾部也未保证与最后有效 b 逐位一致。两稿据此保留实际扫描、BF16 RNE 舍入和未知的全域误差，不把 chunk 结果声明成 FP32 reference 的精确值。分段 G>=3 的 amax<=1e-3 门只看衰减积；它没有测量 k/beta 反馈或状态幅度，省略项并非仅 ~2^-24 的 FP32 重排误差。短请求若产生空段，pass A 发布 a_seg=1，pass C 由 segment 0 重跑整链，其他段退出；没有新输出空洞。

来源验收边界：阈值来自固定 problem/metadata.json 的 benchmark_contract.correctness_tolerances.default（atol=.01、rtol=.05、max_mismatch_fraction=0），可变输入观察对象由 problem/input.py 的 observable_inputs/writable_views 指定为 core 前缀；外部比较器本体及完整逐输出比较公式仍未随材料提供，未采用 DOC1 的默认判据来补齐。performance/CERTIFIED_BEST.json 记录候选 fixed/domain/reverse_fixed 三项有限作业通过；metadata.validation 的 numerical_failure 记录的是原库，两者不能互相覆盖。源码内注释的速度与精度数字是其所述实验记录，不是当前完整域实测。持久 workspace 与两份独立 norm 缓冲仅有同流复用顺序，源码没有跨流并发调用隔离；out/final 则是同一次新分配的互不重叠视图。所有已登记 Storage 的数量按每 CTA 总量计，不按每 warp 或每 thread 再乘；寄存器逻辑账不声称等于编译器最终寄存器占用。

## 完整设计

| 设计 | 具体机制与保留理由 | 关系 |
| --- | --- | --- |
| [root](root.yaml) | 整体式单链两 kernel 流水线。选定实例（真实分派结果，非拼接）：flag 为真时先 l2norm_qk；gdn_prep<GQA=false,RATIO=1,QHFAST=true>，grid (HV, slots_ub)，每 (槽, value head) 一个 CTA；gdn_chain<NSTAGE=2,VTT=32,MODE=0,KQTMA=false,PRE0=false> 且 perm=nullptr，grid (4, HV, N)，PDL 提交。机制即范围节骨架：prep 发布平面/向量/KQ 镜像，chain CTA 串行走完整条链，寄存器状态 + hi 平面 mma，step2–8 主链，step7b 合并写输出，链尾 epilogue 写 final_state；MODE0 的设备文本即共享树基准计算，分段机制全部编译期消除。保留理由：接口最简（一次 prep、一次 chain、无跨 launch 组合），端到端独立产出全部输出，是各档位与特性记录的公共参照。可达性示例：tp=4（H=4）、head_ratio=8（HV=32）、N=1、T=256，在 sm_count=148 下——slots_ub*H=20<sm_count 故非 GQA；N<8 且 chain_blocks=128<=sm_count 故无 perm；8*HV*N=256>sm_count 非 vt16，ctas32=128<=sm_count 非 vt64，得 vt32；avg_chunks=4<8 得深度 2；足迹远小于 L2 故非 KQTMA；vtt≠16 故无 PRE0。 | 阅读入口，保存唯一 Contract |
| [segmented](segmented.yaml) | 段并行两 pass 链（Gseg=G>=2，host 仅在 vt16 选中类内评估），选定实例为窄臂 vt16；宽臂 VTT=64 为参数差额记录，见“段并行宽 tile 臂 VTT=64”。选定实例（真实分派结果，非拼接）：gdn_prep<GQA=false,RATIO=1,QHFAST=false>（slot-fastest，grid (slots_ub, HV)）+ gdn_chain<NSTAGE_S=3,VTT=16,MODE=1>（pass A）→ gdn_chain<NSTAGE_S=3,VTT=16,MODE=2>（pass C），grid (8, HV, N*G)，两次 PDL 提交。机制：每 (r,h) 链按 seg_count 切段；pass A 每 CTA 只走本段（seg_lo=floor(chunks*sg/G) 起），删减 q 侧（qpred/attn2/o 与 out 写）、u_h 平面不写、bulk 装载跳过 q 半镜像与 qkd 平面，从 state0（sg=0）或零状态（sg>0）以传播速率推进 Sreg，段末 store:seg_transfer 向 perm 尾垫之后的 transfers 区发布 b_seg（段末状态 [key_dim][16] fp32，key-major）与 a_seg=∏ai(63)；pass C 的 PDL wait 绑定 A 网格完成，sg=0 从 state0 重跑段 0（与整体式逐位一致），sg>0 先 compose_state 按 i 降序近似组合 sc(seg_lo)=sum_{i<sg}(∏_{i<j<sg} a_seg(j))*b_seg(i) 再全速率重跑；全部输出只由 pass C 产出，final_state 由 sg=G-1 写。设备侧弱衰减回退：amax=max_sg a_seg > theta=1e-3（GDN_SEG_THETA）时 sg>0 直接退出、sg=0 以 nseg=chunks 整链重跑，逐位回到 MODE0 序列（含 final_state）。三区全部本地定义，不用整区引用：传播速率删减与 b_seg/a_seg 组合公式属数学层，整区引用不能附加 patch，故 Math_reshape 新增“Segment affine transfer and composition”步并本地重排状态推进步；Work_schedule/Tile_pipeline 按新的提交结构、工作归属、输出生产者与转移资源寿命本地定义 PASS_A/PASS_C step、depth-3 stage 环与 seg_transfer。保留理由：相对 root，工作归属轴（段取代整链）、launch 数与完成关系（A→C 跨 kernel 组合，b_seg/a_seg 跨工作寿命）、输出生产者（全部移至 pass C 重跑）与运行时改变工作归属的回退都需重新贯通解释，参数或局部记录不足以表达；有限运算下 G=2 组合只转移 b_seg(0)，保持相同 VTT 的状态计算，G>=3 除 FP32 结合顺序外，还把含 k/beta 反馈的矩阵状态转移替换为标量 a_seg；theta 只筛掉弱衰减，不提供 Contract 全域误差界。可达性示例：tp8（key_head_count=2）、head_ratio=2（value_head_count=4）、N=1、lengths=[2048]，在 sm_count=148 下——slots_ub=33，33*2=66<148 故非 GQA；vt_ctas=8*4*1=32<=148 入 vt16 类；avg_chunks=(2048+63)/64=32>=32；seg_count=min(8, floor(148/32)=4, floor(2111/1024)=2)=2；宽臂因 avg_chunks<192 不成立；N=1<8 且 chain_blocks=32<=148 故无 perm；KQTMA/PRE0 因 Gseg>1 关闭。源码认证的分段形状 c8（tp8、N=1、T=8192、head_ratio=2）不在本节点 when 内：slots_ub=129，129*2=258>=148 使 gqa 为真，其 prep 实为 gdn_prep<GQA=true,RATIO=2,QHFAST=false>（Gseg=4），该交叉格局由“GQA 共享 prep”记录定位。 | 独立并列，共享 root 的 Contract |

## 变体与组合

下列记录以 [root](root.yaml) 与 [segmented](segmented.yaml) 的实名对象（step、role 动作、存储、数学符号）定位基线，源码行号为固定版本证据行位。

### vtile 档 VTT=16（参数）

- 基线：[root](root.yaml) 的 CHAIN step——Work_schedule 的 vt 分区（tile=32）、Worker_count=(value_dim/32)*value_head_count*request_count，Tile_pipeline 的 CHAIN.MAIN for(c) 主循环与 stage_kq/stage_v/stage_tinv/stage_qkd/stage_vec 深度 2 stage 环。
- 差异与联动：vt 分区 tile 16，grid.x=8，Worker_count=(value_dim/16)*HV*N；NPER=1，matmul:kq_mirror@sc_hi 的 B 碎片改 x2 k-major 单 ldmatrix（ldm_b1_kmajor，字节减半，V3j）；KSPLIT 奇偶累加器对仍启用（VTT<=32）；stage 环深度恒 3（NSTAGE_S），各 stage_* 的 slot_index 改 c%3，smem 改为 165,016B；pred_acc/qpred_acc 保留奇偶两组但 value 轴改为 16，各占 8,192B，sreg 为 8,192B；stage_v 的 v box 为 {16,1,64} 无 swizzle；shi/uh/duh/rhh 平面步距 40 不变。
- 条件与依据：host 单波规则 8*HV*N<=sm_count（链串行延迟类，gdn_kernel.cu:3163）；launch 见 3462-3464，深度常量见 2915-2921。

### vtile 档 VTT=64（参数）

- 基线：[root](root.yaml) 的 CHAIN step——vt 分区（tile=32）、CHAIN.MAIN 的 matmul/compute 动作链、shi/uh/duh/rhh 平面与 stage_* 环、store:final_state 收尾。
- 差异与联动：vt 分区 tile 64，grid.x=2；NPER=4（B 碎片改 NPER/2 组 x4 装载，V3r-fix）；KSPLIT 关——matmul:kq_mirror@sc_hi 的奇偶累加器对 accKx/accQx 折叠为单累加器，Math_reshape “Chunk state advance and outputs” 的 Arithmetic 奇偶拆分声明随之失效，累加顺序改变使输出相对 vt32/vt16 为 margins 类而非逐位（源码注 E5a/V-J42）；shi/uh/duh/rhh 平面步距加宽为 72（SPS/USS），stage 字节随变；stage_v 改 SWIZZLE_128B box 并按 stage 实际地址取运行时相位 vph；trans 寻址辅助寄存器改每 chunk 重算（V-J5，控寄存器上限）；stage 环深度恒 2 同 root，dynamic smem 为 151,696B；pred_acc/qpred_acc 的 parity 轴各从 2 改 1、value 轴改 64，各仍 16,384B，sreg 为 32,768B。final_state 的 stage-0 暂存为 [128][65] FP32，末字节 33275，跨 stage_kq 的 32768B 并进入 stage_v 首部 508B；完整chunk路径的 v 读取已在 rhs/B1 后结束，K/Q 读取在 B3 前结束，不另分配存储；Lc=1迷你路径的退休限制见下条。
- 条件与依据：波次绑定（ctas32>sm_count）且 T>=4096 或 ceil(T/64)<=2N 或 w64*3<w32*2（gdn_kernel.cu:3147-3191）；launch 见 3465-3469，vph 见 2366-2368。

### vt64 迷你 chunk 尾块快路径（局部）

- 基线：[root](root.yaml) 的 Tile_pipeline CHAIN.MAIN for(c) 主循环（经 “vtile 档 VTT=64” 参数档；Lc 为本 chunk 有效行数，定义于 Math_reshape “Chunk layout and decay statistics”）。
- 差异与联动：Lc==1 时整 chunk 走退化路径：行 0 的属主 lane 以标量式直接算 rhs/u/o 并直写 out（及 core），u/du 的全平面往返省去，f_hi 后的 DU_0 借用 rhh 行 0 作跨 warp scratch（rank-1 dS 需跨 warp 读取，gdn_kernel.cu:2459/2496/2516），tinv@rhs_h、qkd@u_h 的 mma 与 7a staging/7b 合并写跳过；dS 退化为 k0 行与 DU_0 的 rank-1 外积，取代 kt@du_h；单个 __syncthreads（发布 DU_0）取代 rhs 与 u_planes 的两处 join；state_advance 与 shi 重建沿用完整路径的共享尾部文本。当中间运算有限时，beta 零填使无效 rhs/u/du 为零，退化计算具有相同的单项乘积；源码的逐位主张不能覆盖溢出或非有限中间量。输出生产者与工作归属不变。源限制：该路径在唯一 CTA barrier 之后才做 rank-1 的 K0 读取；若最终chunk落在stage-0且请求final，epilogue在覆盖stage-0前没有跨warp退休等待。例如合法 H=8/HV=64/N=1/T=4097 的末chunk cl=64 使用槽0，warp0的final暂存字节32与warp1的K0读取字节32相撞。所列源码没有建立此读写次序，不能把完整路径的B3退休保证直接套给迷你路径，也不能凭有限通过记录宣称全域逐位安全。
- 条件与依据：仅 VTT==64 编译期限定（vt16 热循环代码尺寸税实测超收益、vt32 中性，V-J10b）；每个 Lc=1 尾 chunk 省约 0.4-0.6us（gdn_kernel.cu:2438-2552）。

### 流水线深度 NSTAGE=3（参数）

- 基线：[root](root.yaml) 的 Tile_pipeline CHAIN stage 环——stage_kq/stage_v/stage_tinv/stage_qkd/stage_vec 的 depth=2 与 slot_index c%2，load:stage / await:stage 动作。
- 差异与联动：深度 2→3，各 stage_* 的 slot_index 改 c%3，mbar 相位仍为 (cl/NSTAGE)&1 奇偶规则；序幕预发 NSTAGE-1 个 chunk（root 深度 2 只发 1 个），循环内仍在每迭代顶发 cl+NSTAGE-1，获两拍预取松量；chain smem 随 ns*ST_BYTES 增大一槽（vt32 的 dynamic smem 从 123,024B 改为 171,160B）。
- 条件与依据：vt16 恒 3；vt32 仅 avg_chunks=(T+N*63)/(N*64)>=8 时取 3（单 chunk 类深度 3 实测回归 +3.3%/+3.8%，V-J9b）；vt64 恒 2（gdn_kernel.cu:2910-2941、3424-3430）。

### KQTMA 直读 q/k（局部）

- 基线：[root](root.yaml) 的 Tile_pipeline CHAIN.MAIN load:stage（镜像 bulk 发射路径）与 PREP.MAIN store:kq_mirror；Math_reshape 的 kq_mirror 字节定义不变。
- 差异与联动：load:stage 改发 4 个 {64,64,1} SWIZZLE_128B k/q box 直读输入张量，写入同一 stage_kq swizzle 镜像区，消费端 swz ldmatrix 与镜像字节逐位不变（仅发射机制差异）；expect_tx 与平面/向量 bulk 及 v box 合并为单一计数；prep 侧收 skip_q=1，store:kq_mirror 动作省去，工作区 q_bytes 区不写。
- 条件与依据：!l2norm && Gseg==1 && vtt>=32 && 工作区+q 输入足迹 > l2_bytes；vt16 实测受损被排除（+3 TMA op/chunk 落在长串行链上）；gdn_kernel.cu:3306-3326、2109-2128、651-667。

### PRE0 chunk-0 预等待预取（局部）

- 基线：[root](root.yaml) 的 Tile_pipeline CHAIN.MAIN 序幕（经 “vtile 档 VTT=16” 参数档；await:prep_grid 与 load:stage(0) 的先后）。
- 差异与联动：await:prep_grid 之前由 tid0 以输入 k/q/v tensor map 发 chunk-0 全套 box（4 k/q + 1 v），单一 expect_tx 预覆盖 chunk-0 全部字节（仅读输入，PDL 合法）；wait 后 load:stage(0) 改走 planes_only，只补 P1 输出的平面/向量 bulk；stage-0 字节与整体路径逐位一致，包含线不变。
- 条件与依据：!kqtma && !l2norm && Gseg==1 && vtt==16 && perm==nullptr（单波 vt16 bulk 类；gdn_kernel.cu:3349-3354、1998-2026、2100-2107）。

### LPT 请求排列 perm（局部）

- 基线：[root](root.yaml) 的 Work_schedule CHAIN n 分区（“n is the grid z index”）与 Tile_pipeline CHAIN.MAIN load:state0（n 解析点）、PREP.MAIN 的 block(0,0)。
- 差异与联动：host 在工作区尾垫分配 N 个 int；prep 的 block(0,0) tid0 按每请求 chunk 数降序稳定插入排序写 perm；chain 在 await:prep_grid 后取 n=perm[n_idx]（wait 前仅做丢弃值的 L2 预热读），load:state0 相应移回 wait 之后（n 来自 P1 输出，不可 pre-wait 读）；最长链拿到最小 blockIdx.z 先起跑，平衡波尾；Worktile_order 的 n 升序物理次序不变，逻辑请求归属由 perm 重排。
- 条件与依据：(8<=N<=32) 或 (2<=N<=32 且 chain_blocks>sm_count)；与 PRE0 互斥（gdn_kernel.cu:3331-3347、265-280、2041-2057）。

### GQA 共享 prep（局部）

- 基线：[root](root.yaml) 的 Work_schedule PREP（hv 分区 tile=1、Worker_count=value_head_count*slots_ub、Mapping grid x=值头）与 Tile_pipeline PREP.MAIN（一 CTA 解一个 (slot,hv)）；第二基线为 [segmented](segmented.yaml) 的同名 PREP step 与 Tile_pipeline PREP.MAIN——两节点 PREP 的动作与存储名完全相同（load:kq_image/g_beta、compute:decay_stats/gram/qk/system/qkd_publish/inv8/inv16/inv32/inv64/slot_publish、store:kq_mirror 与 vec_stage/gram_plane/sys_plane/tinv_plane），差别仅在其 Mapping 为 slot-fastest（grid x=slot、y=value head）。
- 差异与联动：grid 改按 key head——qhfast（Gseg==1）时 (key_head_count, slots_ub)、分段形状恒 slot-fastest (slots_ub, key_head_count)，每 (槽, key head) 一 CTA：load:kq_image 与 compute:gram/qk 每 kh 只执行一次，随后串行循环组内 cnt=ratio 个 value head 的 compute:system→inv8/inv16/inv32/inv64→slot_publish（每头独立发布自己的 (slot,hv) 平面与向量）；vec_stage 的 sb/sbeta/sRho 缓冲数随 RATIO 扩为 SBC 套（RATIO=0 时 2 套）；kq_mirror 仍每 kh 写一次——非 GQA 的“组首 hv 块写”与 GQA 的“单块每 kh”为同一互斥条件（hv0==qh*ratio）。该差额对两条基线都成立：GQA 与分段并行在 Contract 域内可同时成立——源码认证的分段形状 c8 即 gqa=true 且 Gseg=4（slots_ub=129，129*2=258>=sm_count）；多请求类如 tp8、head_ratio=2、N=2、total_tokens=4608，在 sm_count=148 下给 slots_ub=74，74*2=148>=sm_count 触发 GQA，而 vt_ctas=64<=148、avg_chunks=36>=32、seg_count=2、无宽臂。组合时以本条替换 segmented 的 PREP 网格与每 CTA 头归属，PASS_A/PASS_C 及 b_seg/a_seg 发布不受影响；两份设计的 when 均保守要求 slots_ub*key_head_count<sm_count，故 seg+GQA 为记录组合而非已交付实例。
- 条件与依据：slots_ub*H>=sm_count（小工作负载偏好每 value head 一块的短串行 ratio 循环；gdn_kernel.cu:34-47、256-259、3358、3391-3400）。

### prep ratio 特化与双头锁步（局部）

- 基线：[root](root.yaml) 的 Tile_pipeline PREP.MAIN 头维度动作与存储（load:g_beta、compute:decay_stats/gram/qk/system/inv8/inv16/inv32/inv64/slot_publish 与 vec_stage/gram_plane/sys_plane/tinv_plane）。本条是 “GQA 共享 prep” 局部差额之上的二级差额：组内头循环 cnt=ratio>1 仅在 GQA 档存在，root 实例（RATIO=1，cnt=1）不可达。
- 差异与联动：RATIO∈{2,3,8} 编译期特化——load:g_beta 在序幕一次预取全部头的 g/beta（每头一个 cp.async 组），cp_wait 立即数精确；compute:decay_stats 用 tid 128-191 在 K/Q box 在途期间计算全部头的 cumsum(b)/rho，生产 E6 路径在 gram 前全 CTA 汇合（RATIO>=4 改三相位分裂：局部扫描/跨 warp 修正/rho 写出），fp32 运算序逐位不变；RT2/RT3 将 compute:gram 与 compute:qk 融合为单遍共享 B 碎片（V24）；V-J44 组内双头锁步——P1NS=2 套 sys_plane/tinv_plane 并发、E19 一次性单位对角/垫列骨架初始化、E23 头共享 gram 装载、slot_publish 按组进行；各元素运算链逐字保留，发布字节逐位不变。RATIO=0 为域外比值的通用双缓冲回退（声明域 group_ratio∈{2,3,8} 内不可达，仍为登记实例）。
- 条件与依据：gdn_kernel.cu:43-47、150-168、313-399、427-432、466-648、672-1267。

### prep 网格次序 qhfast/slot-fastest（参数）

- 基线：[root](root.yaml) 的 Work_schedule PREP Mapping（“grid x = value head (head-fastest), y = slot”，即 QHFAST=true）。
- 差异与联动：grid x/y 与核内 blockIdx 取轴对调（QHFAST=false：slot=blockIdx.x、头=blockIdx.y），每 CTA 工作与发布字节逐位一致；联动仅 host 宏的 dim3 实参次序与核内三处 blockIdx 取轴。
- 条件与依据：qhfast=(Gseg==1)；分段形状恒 slot-fastest（官方 c8 实测 qh-fastest 致 +1.86us 双峰模态漂移；gdn_kernel.cu:180-202、3385-3400）。

### 段并行宽 tile 臂 VTT=64（参数）

- 基线：[segmented](segmented.yaml) 的窄臂 vt16 MODE1/2 实例——Work_schedule 的 PASS_A/PASS_C step（sg 分区与 b_seg/a_seg Handoff）、Math_reshape 的“Segment affine transfer and composition”步与 b_seg/a_seg/theta/seg_count 符号、Tile_pipeline 的 PASS_A/PASS_C stage_kq/stage_tinv/stage_qkd depth-3 stage 环、sreg 与 PASS_C 的 matmul 动作链（kq_mirror@sc_hi → tinv@rhs_h → qkd@u_h → kt@du_h），及 REALIZATION 的两 pass 统一 165,016B slab；共享骨架另见 [root](root.yaml) 的 Tile_pipeline CHAIN.MAIN matmul 动作链与 sreg 状态归属（两设计共享的公共树文本）。
- 差异与联动：同一 MODE1/2 设备文本按 VTT=64 实例化（gdn_chain<NSTAGE_V=2,64,1> / <2,64,2>），grid (2, HV, N*Gw)；stage 环深度 3→2，统一 slab 改按 chain_smem(64) 定尺（151,696B）；b_seg/a_seg 按 vtt 通用定尺（[2*64][64] fp32，key-major，vt16 数值字节相容）；“vtile 档 VTT=64” 参数档的全部差额（NPER=4、KSPLIT 关、平面步距 72、swizzle v box）同时适用。宽臂门槛：avg_chunks>=192、cols64*Gw<=sm_count、每段平均 >=32 chunk；臂选择按预测串行墙钟——Gw*100>Gc*205 时宽臂胜（速率比来自源码实测 census：vt16 1.82us/chunk vs vt64 3.73us/chunk）。
- 条件与依据：仅在 vt16 选中类内评估后 retile（V-J51/E5b）；数值类别同 segmented 节点（相同 VTT 的 G=2 不省略跨段反馈，G>=3 有标量转移近似；c8/c1 的源码认证注释不构成全域误差保证）；gdn_kernel.cu:3205-3250、3455-3457、2985-2988。

### 分派门相容关系（组合）

- 基线：[root](root.yaml)（含其上各参数/局部记录）与 [segmented](segmented.yaml)（窄臂实例；宽臂经其参数记录）。
- 差异与联动：L2NORM step 按输入 flag 与两设计组合，但排除 KQTMA 与 PRE0（两门均含 !l2norm）；KQTMA 排除 seg 与 vt16（门含 Gseg==1、vtt>=32）；PRE0 排除 perm、KQTMA、seg 与非 vt16；perm 可与 KQTMA、vt32/vt64 并存；perm 与 seg 在 host 门级不互斥（chain_blocks 为 retile 感知值），但在 Contract 域（value_head_count>=4、total_tokens<=16384，census 148 SM）内不可达——窄臂需 16*HV*N<=sm_count 只容 N<=2，宽臂需 avg_chunks>=192 只可能 N=1，且两分段臂的门槛都强制 chain_blocks<=sm_count/2，perm 的两条门（N>=8；或 N>=2 且 chain_blocks>sm_count）均不成立，与 segmented when 的 request_count<8（no permutation）一致；vt32 深度 3 与 KQTMA 可并存（登记实例 <3,32,0,true> 存在）；GQA 与 seg 同时成立的格局见“GQA 共享 prep”记录（两份 when 均排除，为记录组合）。组合只叠加已记录的局部发射/装载变化，不引入新的整体组织；全部可组合的 launch 实例均已在 set_smem_attrs 如实登记。
- 条件与依据：host 门条件 gdn_kernel.cu:3320-3321、3343-3354、3390、3450-3478；真实实例清单见 2943-2992。
