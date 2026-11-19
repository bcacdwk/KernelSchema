# nvfp4_fused_moe

文件版本、来源与检查信息见[当前元数据](.extraction/current.json)。

## 范围

**固定来源与入口。** 各 YAML 固定到 `KernelSchema@24d2b3d3ca9d4907c4e7efd9607cf78cfb1a52b2`。入口 [candidate/solution.py](../../../../../kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/solution.py)`::Model.forward` 校验形状、选择 rung、分配 workspace 并组装 `MoEABI`（热路径经 `nvfp4_fast.c` 做对象身份校验），调用 [candidate/nvfp4_moe.cu](../../../../../kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu)`::moe_run` 原位覆写 `output_buffer` 后装入单元素列表返回。声明域固定 `K = slot_count = 8`、`T ∈ [1, 8192]`、`(H, I, E) ∈ {(2048,768,128), (4096,768,128), (4096,1536,128), (7168,256,256)}`（`H` 为 512 倍数、`I` 为 256 倍数）。

**规模口径与默认 Selector。** 令 `R = T * K` 为分配容量 `route_capacity`（**不是**有效路由数 `route_count`；`route_count` 仅标识非 padding 有效行），源码全部 host 分带门限均以 `R` 计量。声明域默认 selector 行为（`solution.py:811–845, 1260–1309, 1447–1495`）：
1. Capture-time 路由直方图为空且 `K == 8 && R <= 8192 && R * H <= 32e6` 时，整链替换为单核 `MEGA`（rung 92）；
2. 否则 GEMM1 按 `R <= 1536 ? 99 : 97` 选择，GEMM2 由同一张 `cfg2` 配对表独立选择 `81 / 57 / 44 / 23`；
3. 大/小路由准备、`fin4/8/16`、PDL 开关及零输入验证前缀作为独立轴叠加。域外非 packed 回退家族（`rungs 0–6`）、`E <= 32` 分支（`rung 43`）与旧 46 字段 ABI 历史快照均不在收录范围内。

**完整设计集合与代表配置配方。** 本目录收录三份完整 YAML：`root`、`split_k`、`mega`；`root.yaml` 保存唯一共享 Contract，`split_k.yaml` 与 `mega.yaml` 均三区本地定义，不使用整区 `from`。每份 `when` 绑定各自选定的工作组织与参数，不等同于默认 selector 的全部路径，也不自动叠加目录变体。对满足 `root.when` 的非全零输入，在进程启动前设置 `NVFP4_MOE_SMALL_THRESH=1`、`NVFP4_MOE_FIN16MB=1024`、`NVFP4_MOE_PDL=1`、`NVFP4_MOE_PDL_BIG_R=-1`（不覆盖 `CFG/CFG2`），即可通过既有源码开关固定命中 `root` 的大准备、`fin8` 与开启的 PDL；默认环境在部分形状下会改走小准备、`fin16` 或关闭 PDL。

**必要调用与数值限制。**
- **Capture-time 统计只定配置，运行重读数据：** `G2GATE`（GEMM2 真实 route-tile 计数）、`MEGA` 空路由判据与零输入探针均只在慢路径发布/捕图时统计一次；每次 eager 或 CUDA graph 重放仍重读设备当前字节（例如 `MEGA` 重放遇到非空 `ids` 会执行 direct SIMT 算术，空路由选择不证明该路径在全域数值验收通过）。
- **WS3 TMA 编码硬失败无回退：** `root`（97/81）与 `split_k`（99/81）所属的 WS3 家族要求 `cuTensorMapEncodeTiled` 可用且多维操作数 TMA box、`tile::gather4` map（99）、`rout` bulk drain map（81，需 `H >= 64 && R >= 32`）均可编码；编码失败直接返回错误码（`733 / 738`），**无对应 `cp.async` 回退**（`cp.async` 回退仅覆盖非 WS3 单体 rung 与 flat-23）。
- **MEGA 复用等待与隐含前提：** `MEGA` 不经 `launch_gemm`，不使用 tensor map、TMA 或 PDL，动态 shared 位于默认 48 KiB 内；其 `acc32` 与 `cnt` 仅在分配时清零、后续靠每次 ticket finalize 自清，**复用同一缓存 workspace 必须等待前次调用完成，无跨流并发隔离**（host 仍分配未消费的常规中间缓冲）。此外 `k_mega_body` 内循环步长隐含 `H % 512 == 0`、ticket staging 最多缓存 8 个 slot，均依赖当前声明域（`H ∈ {2048,4096,7168}`、`K = 8`）。
- **激活并集与验收边界：** `activation_type` 3 与 10 由共享 Contract 并集定义并由同一设备分支覆盖；外部验收比较器公式缺失，YAML 容差记为 `Source only`。

## 完整设计

| 设计 | 具体机制与保留理由 | 关系 |
| --- | --- | --- |
| [root](root.yaml) | 七节点 packed tensor-core 主链：`COUNT → SCAN_TILES → SCATTER → GATHER → GEMM1(97) → GEMM2(81) → FINALIZE(fin8)`，PDL 开启（`counts` 清零为 `initialize: 0`）。大路径三节点构建 `bm=128` 路由图并预聚集 `gathered_payload/gathered_scale`；GEMM1 取 WS3 七 warp（224 线程、2 CTA/SM），操作数保持 packed NVFP4 + E4M3 scale（`tcgen05.cp` 入 TMEM，`mxf4nvf4` MMA），`NPAIR` 单条 `m128n128k64` 合算 up/gate，`DEEP` 拆开深 3 payload 环与深 2 scale 环等待，`EPI` 融合 clamp、activation(3/10) 与 block-16 NVFP4 QDQ 直写 `act_payload/act_scale`；GEMM2 取 81（BN=128、DEEP、成对排空提前归还累加器、`sink` 2 KiB bulk TMA store `rout`）；`FINALIZE` 按固定 slot 升序每线程归约 8 元素。`when` 要求 `R > 1536`、最密路由下界直接满足 WS3 门（`ceil_div(R,128)*(H/128)*max(1,I/256) >= 3072`）、非 `MEGA` 且 payload 非全零。 | 阅读入口，保存唯一共享 Contract；三区本地定义 |
| [split_k](split_k.yaml) | 五节点 SK=2 链：`PREPARE → GEMM1(99) → ACT → GEMM2(81) → FINALIZE(fin4)`，PDL 开启。单块 1024 线程 `PREPARE`（`k_prepare_small`）一次产出路由图；GEMM1 取 rung 99（SK=2、关闭 `CPHOIST/EARLYDONE`），省去 `GATHER` 与预聚集缓冲，每 stage 发 `nbox = max(1, min(32, ceil(tile_rows/4)))` 条四行 clipped `tile::gather4` 直取 `hidden_states`（末 box 多余行重读首有效 token），工作项扩为 `(tile, k_slice, n_tile)`，`EPI` 写出 FP32 切片部分和 `part`；独立 `ACT`（`k_act_packed<true,2>`）沿切片升序归约后执行 clamp/activation/QDQ 产出 `act_payload/act_scale`；GEMM2 同 81，`FINALIZE` 取 4 元素 `k_finalize4`。`when` 要求 `32 <= R <= 1536`、`T <= 128`、capture-time 真实 route-tile 工作量达 WS3 门（`>= 3072`）、存在有效 `ids` 且 payload 非全零。 | 与 `root` 并列，共享 `root` 的 Contract；三区本地定义，无 `from` |
| [mega](mega.yaml) | 单核无 grid 级屏障 SIMT 方案（rung 92，`k_mega_direct = k_mega_body<false>`，含 CTA 内会合），单次提交替代整条节点链。512 线程 CTA 按连续段遍历原始 `(route r = t*K+k, 32 宽 intermediate chunk s)`：序幕清零无有效路由 token 的输出行并在全 CTA 无有效路由时直接返回；非空路由在 token 切换时解码 `x_dec` 到 padded shared FP32，在线解码 prepared 权重做 SIMT FP32 FC1 点积、shared 内完成 act+QDQ 反量化回 FP32，FC2 chunk 部分和经 `red.add.f32` 原子累加进 `acc32[r,h]`，由完成该 token 最后一个有效 unit 的 CTA 经设备栅栏 arrival ticket（`cnt[t]`）按 slot 升序求 BF16 路由加权和写 `output_buffer` 并自清 `acc32` 与 `cnt`。`when` 要求 capture-time 空路由、`R <= 8192`、`R * H <= 32e6`。 | 与 `root` 并列，共享 `root` 的 Contract；三区本地定义，无 `from`、无 PDL/TMA |

## 变体与组合

### GEMM2 叶子选择与 GEMM1×GEMM2 配对（组合）

- 基线：[root](root.yaml) 的 `Work_schedule::GEMM1/GEMM2`、`Tile_pipeline::GEMM2`（`PRODA/PRODB/MMA/EPI`、`sink`、`store:rout_bulk`、`await:stores_drained`）与 [split_k](split_k.yaml) 的 `Work_schedule::GEMM1/ACT/GEMM2`、`Tile_pipeline::GEMM2`。两份基线共享 `bm=128` 路由图与 `act_payload/act_scale` 接口，均固定叶子 `81`。
- 差异与联动及条件：GEMM1（`97 / 99`）仅决定 `act_payload/act_scale` 的生产者，GEMM2（`81 / 57 / 44 / 23`）仅决定 `rout` 的生产者，两轴按同一张 `cfg2` 表正交配对（`solution.py:1260–1309`）。令 `per_unit = (H / 128) * max(1, I / 256)`，当最密下界 `ceil_div(R, 128) * per_unit < 3072 <= min(E, R) * per_unit`（分散上界）时，`ws_g2` 由 capture-time `ids` 真实有效路由 tile 数 `real_tiles * per_unit >= 3072` 判定；与 `split_k` 组合非 `81` 叶子时，同步移除其 `rout` bulk drain map `Require` 及 `R >= 32` 条件：

| GEMM2 叶子 | 选择门限 | 相对基线 `81` 的机制变化与限制 |
| --- | --- | --- |
| `81`（两基线已选） | `ws_g2 && R >= 32` | WS3 七 warp + `DEEP` + 成对排空提前释放累加器 + `sink` 2 KiB bulk TMA store（要求 `H >= 64 && R >= 32` drain map，缺失硬报 `738`） |
| `57`（参数） | `ws_g2 && R < 32` | 同一 `f4_body_ws` 关闭 `TSTORE/DEEP/EPITB/CPHOIST/EARLYDONE`：`rout` 全走 `store:rout_masked`，移除 `sink`（动态 shared `114,960 → 106,640 B`）与 drain map 前提，`PRODB` 的 payload/scale 槽等待合并为一条 |
| `44`（局部） | `!ws_g2 && I != 256` | `k_gemm_f4_tma` 以 128 线程 4 warp 单一 `MAIN` 角色与每 stage 两次 CTA 会合取代七 warp mbarrier 分工；scale 改走深度 3 `cp.async`（`SFCP`），无 `sf_prefetch` 与 `sink` |
| `23`（局部） | `!ws_g2 && I == 256` | `f4_flat_g2` 在单体主体上取消 per-item 环启停，`g` 跨 item 连续递增（两拍 TMA 始终在途），新增 3 槽寄存器环按 `t % 3` 广播预存 item 元数据，将 item `t-1` 的排空移至 item `t` 首轮前次 MMA retire 后、下次 MMA 发射前。生产仅启动 `NK1=true` 特化，**仅在 `nk = I / 256 == 1`（`I == 256`）时成立，不可推广至 `nk > 1`** |

### 路由准备路径与聚集划分（局部）

- 基线：[root](root.yaml) 的 `COUNT / SCAN_TILES / SCATTER`（含 `counts` 的 `initialize: 0`）与 `GATHER`（`k_gather_f`），并列 [split_k](split_k.yaml) 的单块 `PREPARE`（`k_prepare_small`）。
- 差异与联动：小准备 `k_prepare_small` 用单块 1024 线程一次提交完成直方图、专家前缀、scatter 与 tile 图（无全局 `counts` 清零，`split_k` 已选）；大准备拆为清零 + `COUNT` + `SCAN_TILES` + `SCATTER`（`root` 已选）。两条路径确定性产出相同的 `counts/offs/tile_offsets/tile_expert/tile_first`；`route_of/route_token/route_slot/route_expert` 仅保证专家内置换等价（大准备含全局 `atomicAdd` 块到达序，run-to-run 不保证逐字节复现），`FINALIZE` 经 `route_of` 按固定 slot 升序归约故输出不变。内部等价划分：`SCATTER` 可换每 warp 原子 `k_scatter`；小准备可由 `NVFP4_MOE_PREP2=1` 换 `k_prepare_small2`（融合前段并按活跃 warp 收窄前缀）；`root` 的 `GATHER` 可换 `k_gather_t`（2368×64）或 `k_gather_a`（1184×128），给定同一 `route_of` 写出的有效行逐字节一致。
- 条件与依据：设备侧判据为 `R <= sthresh`（`nvfp4_moe.cu:9161–9163`）。未设 `NVFP4_MOE_SMALL_THRESH` 时，host 对 `3000 < R <= 6500` 且 capture-time 直方图非空取 `3000`、其余取 `6500`（`solution.py:1447–1453`）：因此**仅在默认阈值下** SK 带（`R <= 1536`）恒走小准备，而 `root` 形状域内的 `1536 < R <= 3000`（或 `(3000, 6500]` 空路由未进 `MEGA`）默认亦走小准备；`NVFP4_MOE_SMALL_THRESH` 可设为 `1` 而将 SK 带切至大准备路径。

### Finalize 宽度、环/归约旋钮与 Host 编排（参数）

- 基线：[root](root.yaml) 与 [split_k](split_k.yaml) 的 `FINALIZE`、`GEMM1/GEMM2` `Storage`/角色、`Flow.Launch`，[split_k](split_k.yaml) 的 `ACT.MAIN`，以及 [mega](mega.yaml) 的调用边界。
- 差异与联动：
  - **Finalize 每线程宽度：** `k_finalize4`（4 元素、`hidden_block=512`，`split_k` 已选）、`k_finalize`（8 元素、`1024`，`root` 已选）、`k_finalize16`（16 元素、`2048`、`.cs` streaming store）算术与 slot 升序一致，仅改线程映射与网格宽度。
  - **WS3 环深与通路旋钮：** `STAGES/SFS` 改 payload/scale 环深；`DEEP` 拆分 payload 与 scale 槽释放等待；`SFCH/SFPF` 改 B-scale 连续段搬运划分与 L2 预取距离；`NPAIR` 合并 up/gate 为单条 `m128n128k64`（与 `root.yaml` `Forbid` 的双累加器 `PP` 互斥）；`CPHOIST` 为 `MMA` 的跨项预取门（99 编译期关闭 `CPHOIST/EARLYDONE`）。
  - **ACT 切片归约形式：** `k_act_packed<true,2>`（`ACTV2`，`split_k` 已选）并发预发全部切片 `part` 读取后再解析 `route_expert`；`SKT=0` 按运行时循环逐切片加载，输出不变。
  - **网格上限、PDL 与调用缓存：** GEMM1/GEMM2 持久网格上限（默认 296 CTA）仅改 `gridDim.y` 跨步划分；关闭 `PDL` 把相邻启动边退回 `kernel_done`；CUDA graph 捕获重放与 `_hot → _fast → nvfp4_fast.c` 三层对象身份校验仅减少 host 开销。
- 条件与依据：Finalize 默认门限为 `R * H * 2 >= fin16_mb() << 20`（默认 40 MiB）选 `16`，否则 `fin_wide()` 开启且 `T <= fin_tmax()`（默认 128）选 `4`，其余选 `8`（如 SK 带 `128 < T <= 192` 选 `8`）。`CPHOIST` 运行时门 `R >= 16 * E` 在 SK 带 `R <= 1536` 下恒假。`NVFP4_MOE_ACTV2` 未设时默认返回 `1`（设 `0` 退回 `SKT=0`）。`PDL` 默认开启、`R >= 49152` 时关闭；`moe_run` 在同键第二次调用捕图（`NVFP4_MOE_NO_GRAPH=1` 走 eager），任何身份失配或连续化拷贝均回落慢路径；上述旋钮均混入 graph key。

### 零输入验证前缀与两图重放（局部）

- 基线：[root](root.yaml) 与 [split_k](split_k.yaml) 的 `Work_schedule::Flow.Launch` 及各角色 `await:predecessor`。
- 差异与联动：capture-time `moe_zero_probe` 确认 `hidden_states` payload 全零且相关 scale/scores/参数有限无 NaN 时，分配 `zflag[4]` 并在链首前置 `k_zero_verify`（不构成第四份完整 YAML）。每次运行 `k_zero_verify` 重扫当前字节、无条件清零 `output_buffer` 并置 `zflag[0] = 1(clean) / 0(dirty)`；链上各 kernel 在 `pdl_wait` 后经 `ZGUARD` 读到 `1` 即早退，dirty 时整链照常重算覆写输出。默认 `NVFP4_MOE_ZERO=2` 且 `T * H >= 2^20` 时拆为两张图：先重放仅含 `k_zero_verify` 的 clean 图并由 host 在 pinned 镜像 `zflag_host` 上有界自旋，仅在 dirty 或超时（`2e8` 次）时补发完整链图（外层处于图捕获时跳过自旋按序记录两图）。
- 条件与依据：仅在未设 rung override、`cfg ∈ {74, 78, 97, 99}`、`cfg2 ∈ {23, 44, 57, 81}` 且 `K == 8` 时启用，**可配 `97` 或 `99`，不配 `MEGA`**（`solution.py:1479–1553`）；`root.yaml` 与 `split_k.yaml` 的 `when` 均以“payload 幅值非全零”排除该前缀。

### 未交付的实验与 Override 分支（局部）

- 基线：[root](root.yaml)、[split_k](split_k.yaml) 与 [mega](mega.yaml) 对应模块。
- 差异与联动：
  - **SK 实验臂（`88 / 95 / 98`）：** 相对 [split_k](split_k.yaml)，`88`（`FUSE`）取消独立 `ACT` 节点，由 `GEMM1.EPI` 每 `(tile, n_tile)` 最后到达者归约切片；`95`（`SK4`）将 `split_count` 扩为 `4`（`H=2048` 时单片仅剩 2 个 stage，低于环深 3）；`98` 将 clipped `gather4` 换回 `root` 的 `GATHER` 预聚集 A。
  - **Staged MEGA 与网格上限（`91` / `MEGA_GRID`）：** 相对 [mega](mega.yaml)，rung 91（`k_mega_decode`）在 shared 增加 `48 * H` 字节 `w1/w2` staging 区经 `cp.async` 搬入；`MEGA_GRID` 仅改 `Worker_count` 上限（默认 296）。
  - **REDF 融合路由归约（`46 / 47`）：** 相对 [root](root.yaml) / [split_k](split_k.yaml) 配单体 GEMM2，侧流并行清零 `acc32[T, H]` 并在 `GEMM2` 启动前等待该侧流清零完成，`GEMM2.EPI` 将 per-route BF16 乘路由权重后经 `red.global.add.v4.f32` 原子累入 `acc32`（求和序由固定 slot 升序改为原子到达序），`FINALIZE` 缩为 `k_convert` 单次 BF16 舍入。
  - **SELFPREP 网格内准备（`96`）：** 省去独立准备节点，由 `GEMM1` 各 CTA 在 `pdl_wait` 前执行 `self_prep` 并发重算确定性规范序路由图。
- 条件与依据：全部默认关闭、仅环境变量 override 可达，未交付完整 YAML。`88/95/98` 由 `NVFP4_MOE_CFG` 强制；rung 91 由 `NVFP4_MOE_MEGA=2` 启用且要求 `R <= 64`、`min(E, R) * 3 * I * H <= 96_000_000` 及 `<= 227 KiB` opt-in shared（`H=7168` 被排除，见 `solution.py:1423–1425`）；`REDF` 由 `NVFP4_MOE_REDF=1` 启用（`nvfp4_moe.cu:9186–9191, 9330–9331`），跳过 WS3，TMA 编码失败硬报 `732`；`SELFPREP` 由 `NVFP4_MOE_SELFPREP=1` 启用，正确性硬上限 `R <= 256`（4×64 chunk），且 host 门要求默认 GEMM1 为旧 78（当前默认已为 99），在声明域内默认不可达。
