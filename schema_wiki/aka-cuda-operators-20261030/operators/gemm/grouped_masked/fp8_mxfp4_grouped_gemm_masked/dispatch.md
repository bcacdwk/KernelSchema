# fp8_mxfp4_grouped_gemm_masked

## 范围

**实现范围与提交。** 固定实现是 `candidate/solution.py::Model.forward`：首次调用用 nvcc 把 `candidate/kernel.cu` 编成共享库（按源码字节与编译 flag 的哈希缓存），ctypes 绑定 `fmggm_launch`；此后每次调用只做一次参数校验（或命中缓存后跳过）、组装 12 个标量/指针实参，在 PyTorch 当前 stream 上启动**一个** kernel，返回 None，结果就地写进调用方的 `d`。没有 GEMM/attention 库、DSL 或 Torch 数值回退，没有 CUDA graph 捕获，没有独立的 scale 重排 pass——f32→UE8M0 转换融合在被测 kernel 内。认证测量的 NCU 记录印证了这一点：每个 shape 的 `kernel_counts` 恒为 `[1,1,1,1,1]`，出现过的 kernel 名只有 `fmggm_kernel_dr` 与 `fmggm::fmggm_kernelT<(int)16,(int)5>`。

**默认构建即认证构建。** 取 `kernel.cu` 的默认 flag：DROUTE=1（主 kernel 是设备路由包装 `fmggm_kernel_dr`，其窄分支 = 文件作用域几何）、DEC16=1 且 DEC_STG=5、DUALN=1 且 N_B=2、BK=128、STAGES=4、ACC_STAGES=1、DESCROLL=1、DINTER=1、STAIL=1、ABOX=1、TCP=1、TMA_STORE=0，其余实验臂（MCAST/BK256/J0LA/J0LB/EDR/TJEB/LDBATCH/LDGSF/EBOX/BMERGE/BPF/LEVICT/DCS/PKELECT/PDL/FASTEXIT/MMC/PROMO/EPAD/EWSLEEP/GG/NKBH/EXPER）为 0；BMSHIFT 与 J0LB 因 DROUTE 被自动关闭。其中 SF_PF、DCS 在默认构建里没有使用点（见来源限制第 4 条）。NCU 实测 grid=(148,1,1)、block=(256,1,1)；主路由每 block 分配共享内存 232,448 B，其中动态部分 230,400 B，解码路由动态 214,016 B。两个动态值正好等于 `SharedStorageT<128,4>` 与 `SharedStorageT<16,5>` 逐字段相加、含屏障后向上对齐到 1024 B 的结果，且 232,448 B 同时是源码 static_assert 约束 `sizeof(SharedStorage)` 不得超过的 opt-in 上限，因此默认 flag 组合就是被认证与被测的那一份。

**主要机制（root 承担，对象名与 root.yaml 一致）。** 一次启动、每 SM 一个常驻 CTA（grid = `sm_count`，`__launch_bounds__(256,1)`），CTA 内 8 个 warp 组成四个专用 role：LOAD（warp0）的一个选举 lane 发全部 TMA（A/B/SFA/SFB），MMA（warp1）的一个选举 lane 发 UTCCP 与 tcgen05 UMMA，PACK（warp2/3）把 TMA 落地的 f32 scale 就地打包成 UE8M0，EPILOGUE（warp4-7）做写回。主计算是 swap-AB 的块缩放 MMA：权重（E2M1，TMA 以 `16U4_ALIGN16B` 硬件解 nibble 成 8-bit 容器）作 MMA A、M=128 权重行；激活（E4M3）作 MMA B、N 动态收缩到 `eff_a = align16(eff)`；每 stage K=128 拆成 4 条 UMMA_K=32 的 `matmul:wtile@atile`，scale 以 UE8M0 存在 TMEM（`sfw_tmem` 权重侧 2 组 4 列、`sfa_tmem` 激活侧 `ceil(eff_a/128)` 组 4 列）由硬件按 32 元素块施加；整块 k 累加在 `acc_tmem`，最后一条 k-block 的 `tcgen05.commit` 触发 `acc-ready`。输出形成是 TMEM→`epi_regs`（窄化 bf16）→`stmatrix.trans`→`cd_stage` 暂存→向量化 16B `st.global` 到 `d`（DINTER 下一个 si 同时写两个 j 半，16 行×256 列连续）。工作分配是持久 masked-grouped 调度：扁平位置 `pos = iter*sm_count + worker_id` 经 expert 累积 m-block 走查与 16 个 m-block 的子分组（`g_group`，选择规则见“调度分组因子与 m-block 计数”）解码到 (e, mb, np)，`nmb(e) = ceil_div(masked_m[e],128)`，空 expert 不产生 tile，一个 n-pair 含 2 个 128 行权重 n-tile 共享同一份激活 tile；STAIL 把尾波的 R 个 pair-tile 拆成 2R 个单 j 半 tile（`jlo`/`njl`，仅当 `1 <= R <= sm_count/2`）；各 role 各自持有调度状态走同一位置序列，不做跨 warp 的 tile 广播。协作与寿命由 4 深 smem ring（`a_ring`/`b_ring`/`sfa_ring`/`sfw_ring`）与五个通知界定（语义名＝源码名）：`full`（65 = 1 次 expect_tx + 64 个 PACK lane 到达；默认关闭的 PKELECT 会把计数与到达者换成 3 = 1 次 expect_tx + 2 个选举 lane，见“SF 打包到达规则”）、`scale-landing`（`sf_full`，1）、`slot-release`（`empty`，1，由 MMA 的 `tcgen05.commit` 到达，界定槽内 smem 读完成）、`acc-ready`（`tmem_full`，1，commit）、`acc-release`（`tmem_empty`，128，EPILOGUE 4 warp 全 lane 到达）；`acc_tmem` 与 `cd_stage` 均单缓冲，EPILOGUE 的 role 内 serial 顺序（对应源码两个 128 线程 named barrier）分隔 STSM 与 store。全空输入在 TMEM 分配之前直接退出（`if(any_active)` 为假）。

**root 选择与承担范围。** root 取窄几何主体 `fmggm_bodyT<BMV=128, STG=4>`，即 `fmggm_kernel_dr` 的 else 分支、`SharedStorage = SharedStorageT<128,4>`、启动时 dynamic smem = `sizeof(SharedStorage)`。理由：它是文件作用域的默认几何与全部布局常量的基准值，接口最清楚、便于比较；10 个固定用例中 6 个（1/2/3/5/6/8）直接由它承担；`-DFMGGM_DROUTE=0` 时它就是逐位相同的认证单几何构建，说明该几何本身在完整声明域上成立——宽/解码几何只是同一 body 模板的取值变化，不是正确性前提，因此不按它们拆完整设计。root.yaml 的 `when` 据此承担整个共享 Contract 域（附入口调用条件）；实际实现的解码门与设备路由是选择规则，按规范不写进 YAML，只记在“几何路由”条目，`when` 不承担原 selector 职责。当前完整设计集合只含 root 一份（含 root 共 1 份，不超过 10 份上限），没有其他已选或计划中的完整设计文件；其余全部配置差异按参数/局部/组合条目记录在本目录，不产生新 YAML。精确参数与 `when` 以 root.yaml 为准。

**已研究但未完整展开的独立方案。** `fmggm_kernel2sm`（`__cluster_dims__(2,1,1)`，配 `SharedStorage2`、STAGES2=7）是另一套整体组织，不是 root 的参数或局部变化：2 个 CTA 组成一个 cluster 共同做一个 256 权重行的 tile，UMMA 换成 `cta_group::2`（idesc 的 umma_m=256）；激活按 N 劈半，每 CTA 只取 `eff/2` 行（用 `map_a8/a16/a32/a64`）；激活 SF 每 CTA 全量复制 128 行，权重 SF 按 M 劈分各取自己的 128 行；A/B 用 cta_group::2 TMA，tx 累计到 leader 的 `full`（count 129），两个 CTA 的 pack warp 用 `mbarrier.arrive.release.cluster` 远程到达 leader 的 `full`；只有 leader 的 warp1 发 UTCCP 与 UMMA，`tcgen05.commit...multicast::cluster` 同时通知两个 CTA 的 `empty` 与 `tmem_full`，`tmem_empty` count 8（两 CTA 各 4 个写回 warp 的选举 lane 远程到达）；调度以 cluster 为 worker（`blockIdx.x>>1`、`num_clusters = num_sms/2`、n-tile 宽 256、`total_tiles=0` 因而关闭 STAIL）；进入与退出各一次 cluster barrier，TMEM 用 cg2 形式分配/释放；写回时每 CTA 存自己的 128 列。取舍理由：默认构建里 `g_use_2sm = 0`，只有设置环境变量 `FMGGM_KERNEL=2sm`（且 `N%256==0`、SM 数为偶数）才会启动，源码注明它在 L20D 上更慢（cluster barrier 链 ~280 ns/stage、cg2 TMA ~530 ns/stage），认证测量的 NCU 记录里也没有它出现；本次因此只登记机制、出处与条件，不交付完整 schema，root.yaml 的 REALIZATION.Forbid 已把这一否决登记为设计约束。若后续要求覆盖非默认路径，可按本段说明补一份完整设计。

**来源限制与已核实事实。**
1. 入口侧限制（不是 Contract 要求）：`k % 128 == 0`、六个张量 contiguous 且在 CUDA 上、dtype 与 scale shape 匹配、五个指针 16B 对齐、`E/M/N/K > 0`、`masked_m` 非空，否则 `forward` 抛错或 `fmggm_launch` 返回 −10…−13。
2. n-pair 枚举假定 `n % 256 == 0`：源码为 ragged n-tail 保留了 j 窗口 clamp，但注明冻结域下永不触发；`num_n_blocks2` 直接写 `N/(2*BN)`。
3. STAIL 的 m-block 汇总 pass 用 `tid < num_groups` 一次读完，只覆盖 `group_count <= 256`（THREADS），冻结域 `<= 64`；`total_tiles == 0` 时退回无界枚举。
4. 两个宏在所选构建里没有使用点，正文不要补造对应机制：`FMGGM_SF_PF`（默认 4）全树只有定义与 guard，没有任何使用点，故不存在 SF 预取深度；`FMGGM_DCS`（默认 0）唯一的使用点位于 `#if FMGGM_EDR` 块内部（kernel.cu 2485-2491 的 `st.global.cs`），而 root 实际走的两条写回路径——DINTER 的 store pass（2400-2418）与非 DINTER 的普通向量化 store（2668-2688）——都是裸 `*dst = val;`，因此 EDR=0 的默认构建下 `DCS=1` 不改变任何 store。
5. host 侧按 `(六个指针, E, M, N, K)` 缓存 8 槽 TensorMap 与整份 `Params`，命中只省描述符编码与结构体填充，不省任何设备工作；`Model.forward` 另按 `(data_ptr, shape, expected_m)` 缓存实参并跳过重复校验。TensorMap 一次编码后由 kernel 每次读取，`masked_m` 的内容始终在设备上按次读取，缓存不冻结掩码值。
6. 源码注释保留了两处已在当前树修复的宽几何缺陷（BMSHIFT 的 2 的幂前提在 `bm=224` 下失效导致走 `ceil(cnt/256)`；E116 的 `dcol_j1` 加数写成 128）。当前默认构建 BMSHIFT=0、`dcol_j1 = tmem + acc*(BM*N_B_PER_STAGE) + BM`，正文按当前源码写，不按注释里的历史版本写。
7. UE8M0 打包 `pack_sf4` 只取 f32 的指数字节，因此 scale 的转换在“输入 scale 恰为 2 的幂”时才无损；这一前提来自固定输入构造（Contract 已记录 UE8M0 值域），设计侧作为来源事实保留，不额外补造舍入。
8. 写回的逐 lane 寻址是实现边界：`tcgen05.ld` 的 `|0x00100000` 列位偏移、`cd_stage` 的 swizzle 异或、store pass 的 row/chunk 分解不在 root.yaml 展开；REALIZATION 只绑定原语（tcgen05.ld 16x256b、stmatrix.trans、16B 向量 store），`acc(j)` 到 `d` 行列的转置关系由 Math_reshape 与 `store:d_pair/d_half` 的定义唯一确定，逐 lane 地址留给实现。

## 完整设计

| 设计 | 具体机制与保留理由 | 关系 |
| --- | --- | --- |
| [root](root.yaml) | 窄几何主体 `fmggm_bodyT<128,4>`：常驻单 CTA + LOAD/MMA/PACK/EPILOGUE 四 role 的 tcgen05 块缩放 masked grouped GEMM。swap-AB UMMA（权重 E2M1 为 A、M=128；激活 E4M3 为 B、N=`eff_a=align16(eff)`）；PACK 把 f32 scale 在 ring 槽内就地转成 UE8M0，MMA 经 UTCCP 进 `sfa_tmem`/`sfw_tmem`；4 深 ring、每 stage 两个权重 n-tile 共享一份激活 tile（`acc_tmem` 的 `acc(j)` 双半区、单缓冲）；`pos` 位置的持久 masked-grouped 调度 + STAIL 尾波半 tile 拆分（`jlo`/`njl`）；EPILOGUE 按 capture→`release:acc`→stage（`cd_stage`）→store 转置写回，DINTER 交错 j 直写 `d`。保留理由：默认构建的主几何、认证 NCU 里实际运行的 kernel，一份即可从 Contract 输入贯通到全部输出，其余几何与实验臂都能围绕它的既有对象说明差额。 | 阅读入口，保存唯一 Contract |

## 变体与组合

### 宽激活 tile（参数）

- 基线：[root](root.yaml) 的 128 行激活 tile 与 4 深 ring——`Tile_pipeline` Storage `a_ring`（row=128、depth=4）及同深度的 `b_ring`/`sfa_ring`/`sfw_ring`，`Math_reshape` 的 `eff ∈ [1,128]`，`Work_schedule` Decomposition 的 `nmb(e)=ceil_div(masked_m[e],128)`。
- 差异与联动：`fmggm_bodyT<224,3>`（`FMGGM_DROUTE_BM=224`、`DROUTE_WIDE_STG=3`）。每 stage 涨到 69,632 B，深度只能是 3（源码 static_assert：深度 4 超 232,448 B 上限，且宽 body 必须装进窄 body 的 `sizeof(SharedStorage)`——启动 dynamic smem 取窄 body 的 4 stage）；激活 SF 变成 2 个 128 行组（`NSFA_K=2`，`sfa_ring` 槽 4,096 B，每 stage 多一次 UTCCP、多 4 列 TMEM）；TMEM 列位随实例化的影子常量（body 内的 `BM`/`SFA_STRIDE`/`SFW_COL`/`SFA_COL`）整体后移：j0=0、j1=224、SFW=448、SFA=456，共 464/512 列，不能按 root 文件作用域的 128 推导；A 与 SFA 改用 host 另行编码的 `map_a_wide`/`map_sfa_wide`（box 高 224），`TX_SFA` 按 224 行计（3,584 B），ABOX 阶梯多一档 128；调度器收到 `bm=224`：`nmb(e)` 变成 `ceil_div(masked_m[e],224)`、`eff` 上限 224，`g_group` 随之取 8（见“调度分组因子与 m-block 计数”）。
- 条件与依据：只由设备路由进入（见“几何路由”）。依据是 E117/E119/E121 的实测：`blk_ratio >= 1.40` 且多于一波时更优；固定用例 4/7/9 走它。含宽几何的构建整体关闭 BMSHIFT 与 J0LB。

### 解码 tile（参数）

- 基线：与“宽激活 tile”相同——[root](root.yaml) 的 128 行激活 tile（`a_ring` row=128）与 4 深 ring。
- 差异与联动：`fmggm_kernelT<16,5>`（DEC16，`FMGGM_DEC_STG=5`，配 `SharedStorageDec`）。`a_ring` 槽缩到 16 行（2,048 B），每 stage 40,960 B，5 深加 cd 暂存共 212,992 B（含屏障按 1024 B 对齐后 214,016 B）；调度器收到 `bm=16`：`nmb(e)` 变成 `ceil_div(masked_m[e],16)`，同一份权重可能被至多 8 个 m-block 重复读取；`sfa_ring` 槽仍保持 128 行大小（SFA TMA 与 PACK 不变），TMA 只填前 16 行，超出 UMMA N 的 TMEM lane 不被读；A 走 `map_a16`、`tx_a = 16*BK`；启动 dynamic smem 换成 `sizeof(SharedStorageDec)`，并对该实例单独 `cudaFuncSetAttribute`。
- 条件与依据：host 门 `expected_m >= 0 && expected_m <= 16 && M <= 128`（`M` 即 capacity，域内等价于 capacity=128）；固定用例 0 走它，NCU 里对应的 kernel 名就是 `fmggm_kernelT<(int)16,(int)5>`。依据是深 ring 覆盖冷而跨步的解码权重流（源码记录 case0 71.8→67.4 µs）。`expected_m` 只是路由提示，不影响结果（root.yaml 的 `when` 已声明其 advisory）。

### 几何路由（组合）

- 基线：[root](root.yaml)，加上“宽激活 tile”和“解码 tile”两条参数记录。
- 差异与联动：两级选择，接口相容、不引入新的整体组织。host 级在 `fmggm_launch` 里先判解码门，命中就启动 16 行实例并直接返回；否则启动 `fmggm_kernel_dr`，dynamic smem 取两个 body 的较大者（窄 body 的 4 stage）。设备级在包装 kernel 里让**每个 warp 冗余**遍历 `masked_m` 求 `b1 = Σ ceil(c/128)`、`b2 = Σ ceil(c/224)`，warp 内 shuffle 归约后按 `wide = (b2 > 0) && (b1*n_pairs > num_sms) && (5*b1 >= 7*b2)` 分支到 `fmggm_bodyT<224,3>` 或 `fmggm_bodyT<128,4>`；该值是 `(masked_m, num_groups)` 的纯函数，因而 CTA 一致、grid 一致，两个 body 的屏障/TMEM/ring 状态不会混用。归约刻意不用 barrier 与 smem 原子（源码实测三屏障版本在 6.2 µs 的 case 1 上多 0.8 µs）。`n_pairs = ceil(num_n_blocks/2)`。
- 条件与依据：路由是选择规则，按规范不写进任何 YAML，root.yaml 的 `when` 也不承担它的职责；固定用例的归属为 0→解码，4/7/9→宽，1/2/3/5/6/8→窄（全空的 case 1 因 `b2=0` 留在窄 body，与既有认证构建逐字节相同）。`blk_ratio` 不可在 host 计算（host 只拿到 groups/capacity/n/k/expected_m），这是设备路由存在的原因；7/5 阈值取自 E119 实测的胜负间隙，第二项“多于一波”是 E121 对预声明域的出样修正。

### stage 的 K 宽度与权重 n-tile 数（参数）

- 基线：[root](root.yaml) 的 stage K 宽度 128（`Tile_pipeline` Index `k` 上界 `k_dim/128`，MMA 每 stage 4 条 `matmul:wtile@atile`、UMMA_K=32，`sfa_ring`/`sfw_ring` 的 block=4 即 SF_SUB=1）与每 stage 两个权重 n-tile（`b_ring` out=256 及 `b_tile(s,j)` 视图、`acc_tmem` 的 `acc(j)` 双半区且 depth=1 即 ACC_STAGES=1、`cd_stage` 的 `cd_pair` 256 列、Mapping 的 `njl=2`）。
- 差异与联动：`BK256=1` 时 Index `k` 上界变成 `k_dim/256`、每 stage 8 条 matmul、SF_SUB=2（`sfa_ring`/`sfw_ring` block=8），`STAGES` 表达式的 BK256 分支优先，ring 深度固定为 3；源码又强制 `DUALN=0`（组合 `#error`），于是 `b_ring` out=128、`acc(j)` 双半区消失、`acc_tmem` 回到双缓冲（depth=2）、`njl` 恒 1、`cd_stage` 只用半区，n-pair 机制消失（只有 BK256 关闭而 DUALN=0 时才是 6 深）。`N_B=3` 时一个 stage 三个权重 n-tile（`b_ring` out=384）、三个 TMEM 累加器列组、`STAGES=3`，并与 DESCROLL 互斥（`#error`，即 root.yaml REALIZATION Prefer 登记的展开实现不可用）。
- 条件与依据：两者在默认构建里都不编译，属于源码保留的实验配置；接口相容性由源码的 `#error` 与 `NW_SUB`/`SFW_COL` 表达式约束。没有实测数据支持它们优于 root，仅作参数登记。

### 调度分组因子与 m-block 计数（参数）

- 基线：[root](root.yaml) 的 `Work_schedule`——Schedule.Strategy 中的分组常量 16（`d=16*n_pairs`、`nh=min(16, nmb(e)-16*grp)`、`mb=16*grp+ig%nh`）与 Decomposition 的 `nmb(e)=ceil_div(masked_m[e],128)`、`Math_reshape` 的 `eff = masked_m[e]-128*mb ∈ [1,128]`。
- 差异与联动：分组因子 `g_group` 在 {8,16} 上取 `min(cand*bm + ceil(num_sms/cand)*BN)` 的一方（bm=128、num_sms=148 时为 16，即 root 已固定的常量；宽 body 的 bm=224 时为 8），`FMGGM_GG>0` 可直接固定。BMSHIFT=1 时 `ceil_div(cnt,bm)` 换成 `>> bm_shift`、`eff` 的钳位换成 `1<<bm_shift`，只在 bm 为 2 的幂时值相同；DROUTE 构建里它被自动关闭，所以 root 实际走真除法。MMC=1 时 `masked_m` 从 STAIL 汇总那趟已经读到的寄存器写进 smem 快照供 walk 使用（多 256 B smem）。
- 条件与依据：BMSHIFT 与 MMC 都被源码记为值相同；E70 实测 BMSHIFT +0.10…0.22%、MMC 同为 +0.09% 但在 case 6 上 −0.50%，两者合用无增益，故只保留免费的一半；DROUTE 下 BMSHIFT 因宽 body 的 `bm=224` 而整体放弃，源码把这笔记为一次真实取舍。

### 尾波拆分开关（参数）

- 基线：[root](root.yaml) 的 STAIL 位置空间与半 tile 分支——Decomposition 的 `positions=T+R`/`split_from=T-R`（当 `2R<=sm_count`），Mapping 的 `h/f/jlo/njl` 解码，EPILOGUE 的 `njl(t)==1` 分支（`capture_half`/`stage_half`/`store:d_half` 与 `cd_half(r)` 视图；`r` 是跨 tile 持续的半区轮转序，对应源码 `tma_stage`，不是 Mapping 的半 tile 序号 `h`）及 `release:acc` 的到达规则。
- 差异与联动：`STAIL=0` 时 `positions` 就是 pair-tile 总数 T，Mapping 恒走 `h<0` 分支（`f=pos`、`jlo=0`、`njl=2`），EPILOGUE 只剩 pair 分支，`cd_half` 轮换与半 tile 到达分支消失，`acc-release` 每 tile 仍是一次到达。STAIL 的 T 依赖 `entry_scan` 第二趟顺带汇总的 m-block 总数（源码 `s_mblocks`）。
- 条件与依据：源码记录 STAIL 默认开启且值相同（同一 per-tile 数学，两个半 tile 兄弟靠 L2 共享 A/SFA），exp9 同作业 A/B 几何均值 ×0.9737；`STAIL` 与 `FASTEXIT` 互斥（`#error`，后者的早退路径不产生 `s_mblocks`），与 `J0LA` 互斥。

### 激活取数范围阶梯（参数）

- 基线：[root](root.yaml) 每 tile 的 A/SFA 取数范围——LOAD `load:ab_tiles` 的 A box 取 {16,32,64} 中覆盖 `eff_a` 的最小档、否则用满 128 行 box（`a_ring` 槽行数），及随之缩小的 `tx_a`。
- 差异与联动：`EBOX=1` 用满阶梯 {16,32,…,128}（`map_a_h[8]`/`map_sfa_h[8]`，`tx_a = eff_a*BK`、`tx_sfa = eff_a*16*SF_SUB`）；`EBOX=2` 把运行期下标换成 switch 以避开 `__grid_constant__` 的描述符取值；`EBOX=3` 只缩 SFA；全部关闭时恒用满 tile box。取数范围与写回域要分两段看：`eff <= i_l < eff_a` 的行仍由 A/SFA box 新取（阶梯 box 高恒 `>= eff_a`）、以 UMMA `N=eff_a` 参与乘加并随 tile 写入 `d`，即 root 的 padding 行语义（同一 expert capacity 内、无数值契约）；`i_l >= eff_a` 的行才不求值、不写回——满 128 行 box 时它们被新取却不用，EBOX 的精确 box 则不取、槽尾保留陈旧 smem 内容。
- 条件与依据：源码记录 EBOX 的 in-situ 数据为负（EBOX=1 的运行期下标描述符取值怀疑导致 case4 2×、case5 +42% 双峰；EBOX=3 显示非 2 的幂 box 高恒 −2.2%，怀疑走 TMA 慢路径），且与 DESCROLL 同一 E71 作业里以几何均值 0.99761 被否决，故默认关闭；ABOX 的三档是 root 的既有行为。

### SF 交付路径（局部）

- 基线：[root](root.yaml) 的 SF 链——LOAD `load:sf_tiles` 把 f32 SF 落到 `sfa_ring`/`sfw_ring` 的 `sfa_f32(s)`/`sfw_f32(s,j)` 视图并 post `scale-landing` 的 expect_tx，PACK `await:sf_landed(s)` → `load:sf_f32(s)` 读进 `pack_regs` → `compute:pack_sf(s)` 就地写回 `sfa_packed(s)`/`sfw_packed(s,j)` 视图后逐 lane fence 并到达 `full`。
- 差异与联动：`LDGSF=1` 取消 SF 的 TMA 请求与 `scale-landing` 那一跳（PACK 的 `await:sf_landed` 随之消失），改由 PACK 用 2 代寄存器前瞻直接 `LDG` f32（越界与超过 `eff_a` 的行读成 0），打包后的字仍落在 UTCCP 读取的同一 `q*128` 字偏移，`full` 的到达与 `slot-release` 门不变。
- 条件与依据：源码给出 EXPER 归因（SF 占 streams 4–11%、decode 15.7%）但未记录 LDGSF 的正式 A/B 结论，默认关闭；与 BK256 互斥（`#error`，前瞻代数按 BK128 定）。

### SF 打包到达规则（局部）

- 基线：[root](root.yaml) 的 `full` 计数 65 = 1 次 LOAD expect_tx + PACK 两个 warp 各 32 个 lane 的逐 lane 到达（MMA `compute:init_barriers` 与 REALIZATION Require 的 counts）；每 lane 在 `compute:pack_sf` 中各自 `fence.proxy.async` 再到达（kernel.cu 1455、1912-1918）。
- 差异与联动：`PKELECT=1` 把 64 次逐 lane 到达换成每 warp 一次 `__syncwarp()` + 选举 lane 到达（1914-1917），`full` 的初值随之由 65 改成 3 = 1 次 expect_tx + 2 个选举 lane（1455）；每 lane 的 `fence_proxy_async` 保留，由 syncwarp 把所有 lane 的 fenced store 排在选举 release 之前。计数与到达点必须同时改，否则 `full` 永不满足或提前满足。
- 条件与依据：默认关闭，源码注明该到达模式在 arm-b 血统里已验证，但本树没有实测结论。它直接改动 root 的一项主要协作事实（`init_barriers` 的 `full` 计数与 `pack_sf` 的到达者），故单独登记。

### 累加器释放与 drain 顺序（局部）

- 基线：[root](root.yaml) 的 EPILOGUE 写回链——`compute:capture_pair(si)` 按 si 同时读 `acc(j1)`/`acc(j0)` 两个半区的 TMEM（j1 先）并窄化 bf16 进 `epi_regs`，最后一个 si 的 capture 退休该 tile 的全部 TMEM 读；随后 `release:acc(t)` 由 128 lane 到达 `acc-release`（源码 `tmem_empty[acc]`，在最后一次 `tmem_ld` 之后到达，staging 与 store 仍在途）；`compute:stage_pair(si)` 把 BF16 STSM 转置进单缓冲 8 KB `cd_stage`（`cd_pair` 视图），动作末尾跨 warp 会合；`store:d_pair(si)` 4 趟 16B 向量直写 `d`。role 内 serial 顺序与 `stage_pair` 末尾会合等价于源码分隔 STSM 与 store 的两个 128 线程 named barrier（barrier 1 在 si 循环开头防上一 si 的 store 未读完就覆写 `cd_stage`，barrier 2 防 STSM 未完就读，kernel.cu 2292、2399）。`acc_tmem` depth=1（ACC_STAGES=1），MMA 经 `await:acc_free(t)` 在下一 tile 首条 `matmul` 前等上一 tile 的 `release:acc`。
- 差异与联动：`J0LB=1` 改成 j1 先 drain，`acc(j0)` 按 pair 奇偶在列 {0,268} 之间交替，新增 `tj1_empty` 与 `tj0_empty[2]`（count 128）作为“TMEM 读完成”的释放点替代单一 `acc-release`，MMA 只等读完成而不等 store，ring 的 `slot-release` 提交不变；`J0LA=1` 进一步用 j0 前瞻把下一 pair 的前若干 stage 提前发出，代价是整段持有 ring 槽（实测中性，已否决）；`EDR=1` 先把整个 j-tile 抓进寄存器立即释放 TMEM，再让打包/STSM/store 与下一 tile 主循环重叠；`TJEB=1`/`LDBATCH=1` 把一个 si 的 8 次/4 次 `tmem_ld` 批量发出后只等一次 `tmem_ld_wait`。
- 条件与依据：这些改动都围绕既有的 `acc_tmem` 列组、`epi_regs`/`cd_stage` 暂存与释放通知，主体组织不变。J0LB 要求 `DUALN && N_B==2 && BMV==128 && !DROUTE`，因此默认 DROUTE 构建里被自动关闭（源码记录 E104 测得 J0LB=0 反而 +0.110%，但先前信用为 +1.2…1.4%，作为一次真实取舍保留）；TJEB 在与 DESCROLL 相同的 E71 作业里以几何均值 0.99954 被否决（kernel.cu 293-294）；EDR/J0LA/LDBATCH 之间是 `#error` 互斥（EDR 对 LDBATCH|J0LA、J0LA 对 LDBATCH、J0LB 对 J0LA|EDR|LDBATCH），TJEB 则是在 EXPER/EDR/LDBATCH/TMA_STORE/J0LA 下被 `#undef` 自动关闭、并不报错。

### 输出写回代理（局部）

- 基线：[root](root.yaml) 的 EPILOGUE 写回代理——`stage_pair`/`stage_half` 把 BF16 STSM 进 `cd_stage` 后，`store:d_pair`（4 趟 ×128 线程 ×16B，16 行×256 列）/`store:d_half`（2 趟，16 行×128 列，读 `cd_half(r)` 当前半区）做向量化 `st.global` 直写 `d`；STSM 与 store 之间靠 role 内会合（源码 named barrier）。
- 差异与联动：`TMA_STORE=1` 时同一份 `cd_stage` 影像改由 `cp.async.bulk.tensor.2d` 写回（box `{64,16}`，`map_d` 无论如何都会被编码），加 `fence.proxy.async` + named barrier，由 ew==0 的 lane0 发两条 store 并 `cp.async.bulk.commit_group`，下一 si 前 `wait_group.read`；`cd_stage` 回到双缓冲（`tma_stage` 轮转；源码结构体把 cd 声明为 2 个 4 KB 槽，DINTER 把两槽连成一份 8 KB 单缓冲影像，TMA_STORE 则把它们用作两级轮转）。`DCS=1`（D 的 store 换成 `st.global.cs`）**不是** root 写回的差额：它只接线在 EDR 早释放 drain 自带的 store 里，默认的 DINTER 与非 DINTER 两条路径都没有这个分支，故默认构建下无使用点（见来源限制第 4 条）。
- 条件与依据：TMA_STORE 默认关闭。它与其余 drain 臂的关系按源码分三类，不能一概写成“被 guard 排除”：（i）DINTER 与 TJEB 由各自的自动关闭 guard 排除（319-323、605-608 的条件列表都含 TMA_STORE）；（ii）EDR 没有任何 guard 排除它，实际是路径优先——EDR 块（2428-2496）自带 store，`#else` 侧的 TMA_STORE 分支（2655-2666）根本不进入；（iii）J0LB 未被任何 guard 排除（其自动关闭条件是 `!DUALN || N_B!=2 || BMV!=128 || DROUTE`，两处 `#error` 只针对 J0LA|EDR 与 LDBATCH），它的 taddr/到达分支（2591-2653）与 TMA_STORE 的 store 分支是同一条 for-si 循环内的顺序两段、可同时编译，`cd_stage` 的 `tma_stage` 双缓冲轮转（2690）正是这一组合的联动。TMA_STORE 改变的是全局写的执行代理与完成关系（bulk group 的 commit/wait_group 而非 role 内 serial + named barrier + 直接 store），故记为局部而非参数。

### 权重请求合并（局部）

- 基线：[root](root.yaml) 的 LOAD 每 stage 6 条 `cp.async.bulk.tensor`——`load:sf_tiles` 的 SFA 1 条 + 每活跃 j 的 SFB 各 1 条（`sfw_f32(s,j)`），`load:ab_tiles` 的 A 1 条 + 每活跃 j 的 B 各 1 条（`b_tile(s,j)`）。
- 差异与联动：`BMERGE=1` 把一个 stage 的两个 `b_tile(s,j)` 合成一条 256 行 B 请求、两个权重 SF 合成一条 `{4,256}` 请求，每 stage 请求数从 6 降到 4；字节数、`expect_tx` 与 smem 影像不变（128B swizzle 下第 128 行正好开新的 1024 B atom，SFB 是 `SWIZZLE_NONE` 所以行连续）。
- 条件与依据：默认关闭。要求 `DUALN && N_B==2 && BK==128 && !LDGSF`、与 `LEVICT!=0` 互斥，且只在 `nj==N_B_PER_STAGE && jlo==0`（整 pair tile）时成立，STAIL 半 tile 退回逐 j 请求；源码把它记为“第一个减少请求数”的投递侧实验。

### 集群权重多播与对等中继（局部）

- 基线：[root](root.yaml) 的单 CTA worker（Mapping `Worker{unit: cta, unit_count: 1}`、`Worker_count=sm_count`）与每 CTA 自取的 `load:ab_tiles`/`load:sf_tiles` 请求、4 深 ring。
- 差异与联动：`MCAST>0` 时同一份 body 以 74 个 2-CTA cluster 启动（`cudaLaunchKernelExC` + cluster 属性，失败则回落普通启动并置 `pair_mode=0`）；相邻 worker id 天然解码到同一 `(group, n_blk)` 的相邻 m-block，故 rank r 只取权重 n-tile j=r 并多播给两个 CTA 的同一 smem 偏移，每 CTA 的影像、描述符、`expect_tx` 不变；新增 `pempty[STG]` 作为对等中继：每个 producer 在等完自己的 `slot-release`（`empty[S]`）后向 peer 的 `pempty[S]` 远程到达，再等 peer 的中继才发多播；`MCAST=5/7` 用硬件多播 `tcgen05.commit` 替代软件中继。`pair_ok` 由 Scheduler 对称判定，两侧永远一致。
- 条件与依据：默认不编译（`#if FMGGM_MCAST`）。两条 guard 必须分开归属：MCAST 自己的组合 `#error`（590-593）要求 `DUALN && N_B==2 && STAIL`，并与 BK256/BMERGE/LDGSF/J0LA 互斥；`MCAST=3`（只多播不中继）另有死锁 `#error`（602-604），需显式 `-DFMGGM_ALLOW_DEADLOCK=1`。与 DROUTE 的互斥、以及 EBOX/EDR/LDBATCH/DRAIN2/BK256/BMERGE/J0LA/LDGSF 的排除，属于 DROUTE 那条 guard（631-638，理由是“一次启动内实例化两个 body”），是 DROUTE 的约束而不是 MCAST 的：在 BMV=128 且 DROUTE=0 的构建里，MCAST 与 EBOX/EDR/LDBATCH 并不被拒绝。E101b 实测：cluster 启动本身免费，软件中继 +97%（265.30→523.28 µs），多播本身 −3.1%（523.28→506.88），因此整体被否决，root.yaml 已把这一否决登记为 REALIZATION.Forbid（2-CTA cluster 共享权重流）；保留登记是因为它给出了跨 CTA 共享权重流的确切协作与代价。

### 缓存与预取提示（参数）

- 基线：[root](root.yaml) 的 LOAD `load:ab_tiles` 对 B 载入用 `evict_last` L2 策略（TCP=1）、全部 tensor map 用 256B L2 promotion（两者已在 REALIZATION Prefer 登记），EPILOGUE `store:d_pair`/`store:d_half` 用普通 `st.global`，无 B 预取。
- 差异与联动：`LEVICT=1/2` 把 B 换成 `evict_first`（1 只在该 expert 只有一个 m-block 时）；`PROMO=1/2` 把 B 或 B+A+SFA+SFB 的 promotion 改成 128B；`BPF=n` 在每个 stage 额外发 n 个 k-stage 之后的 B `cp.async.bulk.prefetch.tensor.2d.L2`。`DCS` 不属于本条：它针对 D 的 store 而非载入侧提示，且在默认构建里没有使用点（见“输出写回代理”与来源限制第 4 条）。
- 条件与依据：TCP/LEVICT/PROMO/BPF 这四项都是值相同、只改载入侧缓存或预取行为的提示。TCP 默认开启（两次复现：case3 −4.8%、case6 −4.2%）；PROMO 的 in-situ 数据更差已关闭；BPF 实测 +33…50% 已否决；LEVICT 默认关闭、无正式结论。

### 启动前缀、退出与诊断臂（局部）

- 基线：[root](root.yaml) 的 Flow.Launch 单次启动（普通 `cudaLaunchKernel`、grid=`sm_count`，启动形状见 REALIZATION Require）、各 role 的 `compute:entry_scan`（`any_active` 在两个 barrier 下 OR，全空则 CTA 整体退出；第二趟汇总 m-block 总数）与 `if(any_active)` 分支、MMA `init_barriers` 的通知布局。
- 差异与联动：`PDL=1` 用 `cudaLaunchAttributeProgrammaticStreamSerialization` 启动，只在 EPILOGUE role 前加 `griddepcontrol.wait`（前驱只写 `d`，故 prologue/TMA/主循环可合法重叠）；`FASTEXIT=1` 把 `entry_scan` 的早退换成每 warp 冗余读 `masked_m` 后 `__any_sync` 归约再分支（无 barrier、无 smem，但必须在分支前归约，否则破坏后续 warp 集体指令）；`EPAD=1` 给每个通知数组 128B 对齐；`EWSLEEP=1` 把 EPILOGUE 的 `acc-ready` 自旋换成带 suspend hint 的 `try_wait.parity`；`NKBH=1` 把每 stage 重读 `p.num_k_blocks` 换成已提升的 `nk_`。
- 条件与依据：以上默认全关，但性质不同：EPAD/EWSLEEP/NKBH 是值相同的布局或等待提示；PDL 改的是启动属性与 EPILOGUE role 的前缀等待，其“结果不变”的依据是前驱只写 `d`，不能归为单纯提示；FASTEXIT 重写早退路径，与 STAIL `#error` 互斥（该路径不产生 `s_mblocks`）。`MTR`/`MTRE` 是 clock64 打点、`EXPER!=0` 是刻意产出错误结果的消融臂（跳 UTCCP/UMMA/SF/载入等），二者都不构成设计，只在需要解释源码注释里的数字时引用。
