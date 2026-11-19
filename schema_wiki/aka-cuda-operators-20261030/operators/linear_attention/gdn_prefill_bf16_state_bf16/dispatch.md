# gdn_prefill_bf16_state_bf16

文件版本、来源与检查信息见[当前元数据](.extraction/current.json)。

## 范围

**固定来源与入口。** 各 YAML 固定到 `KernelSchema@24d2b3d3ca9d4907c4e7efd9607cf78cfb1a52b2`。入口 [candidate/solution.py](../../../../../kernel_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/candidate/solution.py)`::Model.forward` 在 `cu_seqlens is None` 时合成单请求标准 chunk 元数据，随后调用 [candidate/gdn_kernel.cu](../../../../../kernel_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/candidate/gdn_kernel.cu)`::gdn_forward`。设备端由可选 `l2norm_qk`、`gdn_prep` 与 `gdn_chain` 组成；host 分派仅读取张量形状元数据与运行时查询的 `sm_count`、`l2_bytes`，无 D2H 同步。`GDN_HOSTPROF`、`GDN_*_OFF` 等计量与归因编译宏不在生产构建中启用，不属收录对象。

**完整设计集合与覆盖边界。** 本目录收录两份完整 YAML：`root`（整体式 `MODE=0` 单链，选定 `VTT=32`、depth 2）与 `segmented`（段并行 `MODE=1/2` 两 pass 链，选定窄臂 `VTT=16`）；`root.yaml` 保存唯一共享 Contract，`segmented.yaml` 三区全部本地定义，不使用整区 `from`。两份 `when` 各自绑定已核对的保守子域，不单独覆盖整个 Contract（例如域内可达的 `GQA + segmented` 由下文记录组合承载），与变体及组合记录共同覆盖生产构建的全部分派分支。

**必要调用与数值限制。**
- **可空元数据与非连续 `core_attn_out` 缺口：** `solution.py:56–66` 仅在 `cu_seqlens is None` 时合成辅助元数据，Contract 允许的“`cu_seqlens` 为张量而 `chunk_indices / chunk_offsets` 为 `None`”会被 `gdn_forward` 拒绝；实现对 `core_attn_out` 调用 `contiguous()` 后写入临时缓冲但缺少 copyback，非连续视图的原位更新会丢失（`root.yaml:464–466`）。
- **持久缓冲无跨流隔离：** 内部 grow-only `workspace` 与两份 `norm_cache` 缓冲仅依赖同流串行顺序覆写复用，无跨流并发调用隔离（`root.yaml:475–480`）。
- **`vt64` 迷你尾块退休等待缺失：** 在 `VTT=64` 的 `Lc == 1` 最终 chunk 快路径中，各 warp 在该块唯一 `__syncthreads()` 之后仍从 `stage_kq` 读取 `k0` 行；若请求 `output_final_state` 且最终 chunk 落在 `stage 0`（偶序号尾块），epilogue 直接复用 `stage 0` 转置暂存 `final_state`，覆盖前缺少跨 warp 退休等待（`gdn_kernel.cu:2499–2550, 2819–2839`）。
- **算术与验收边界：** 未额外限幅的有限 `g / beta` 仍可能在累积衰减、指数或乘积中溢出（`beta = 0` 不消除 `0 * inf` 产生的 `NaN`），64 行 shuffle 扫描的零填尾部不保证与末有效值逐位一致；分段 `G >= 3` 的 `amax <= 1e-3` 门仅检查标量衰减积，不提供全域误差保证。外部比较器公式与逐输出路由缺失，YAML 容差记为 `Source only`。

## 完整设计

| 设计 | 具体机制与保留理由 | 关系 |
| --- | --- | --- |
| [root](root.yaml) | 整体式单链流水线：可选 `l2norm_qk` → `gdn_prep<false,1,true>`（grid `(HV, slots_ub)`）→ `gdn_chain<2,32,0,false,false>`（`perm=nullptr`，grid `(4, HV, N)`，流序 + PDL 衔接）。共享骨架以 **64-token 三角块解**替代逐 token 递推：`PREP` 计算衰减扫描与 8/16/32/64 四级层次化块求逆，发布三角打包平面 `tinv/qkd`、FP32 向量与 K/Q swizzle 镜像；派生中间量（`gram`、`system`、`tinv`、`qkd`、状态视图 `sc_hi`、`rhs_h`、`u_h`、`du_h`）均量化为**单一 hi BF16 平面**参与 BF16 tensor-core MMA（FP32 累加），跨 chunk 维护寄存器 **FP32 权威状态** `Sreg` 并在链尾直接写出 FP32 `final_state`。`when` 限定 `vt32`、depth 2、非 GQA、无 `perm`、镜像 staging 子域。 | 阅读入口，保存唯一共享 Contract |
| [segmented](segmented.yaml) | 段并行两 pass 链（窄臂 `VTT=16`）：可选 `l2norm_qk` → slot-fastest `gdn_prep<false,1,false>`（grid `(slots_ub, HV)`）→ `gdn_chain<3,16,1>`（`PASS_A`）→ `gdn_chain<3,16,2>`（`PASS_C`），链网格 `(8, HV, N*G)`，两次 PDL 串联。`PASS_A` 省去 `q` 侧与输出写，从 `state0`（`sg=0`）或零初态（`sg>0`）推进本段状态并发布段末 FP32 `b_seg` 与标量衰减积 `a_seg`；`PASS_C` 在 `PASS_A` 完成后对 `sg>0` 按降序仿射组合 `sc(seg_lo) = sum_{i<sg}(prod_{i<j<sg} a_seg(j))*b_seg(i)` 再全速率重跑产出全部 `out`（及 `core`），`sg=G-1` 写 `final_state`。有限运算下 `G=2` 仅转交前段状态 `b_seg(0)`，`G>=3` 以标量衰减 `a_seg` 近似含 `k/beta` 反馈的真实转移；`amax = max_sg a_seg > 1e-3`（含空段 `a_seg=1`）时 `sg>0` 退出、由 `sg=0` 串行重跑整链，该阈值无全域误差保证。`when` 限定 `vt16`、非 GQA、无 `perm`、`KQTMA/PRE0` 关闭。 | 与 `root` 并列，共享 `root` 的 Contract；三区全部本地定义，无 `from` |

## 变体与组合

### Chain vtile 档位、环深与尾块快路径（局部）

- 基线：[root](root.yaml) 的 `Work_schedule::CHAIN`（`vt` 分区 `tile=32`）与 `Tile_pipeline::CHAIN.MAIN`（深度 2 `stage_*` 环、奇偶累加器对 `pred_acc/qpred_acc`、`sreg`、步距 40 平面及 `store:final_state`）。
- 差异与联动及条件：

| 变体 | 条件与依据（`gdn_kernel.cu`） | 相对 `root` 的差异与参数联动 |
| --- | --- | --- |
| `VTT=16`（参数） | `vt16 = (8 * HV * N <= sm_count)`（`3163, 2915–2921, 3462–3464`） | `vt` tile `32 → 16`，`grid.x = 8`；`NPER=1`，`kq_mirror@sc_hi` 的 B 碎片改 x2 k-major 单 `ldmatrix`，保持 `KSPLIT` 奇偶累加器对；stage 环深度恒升至 `NSTAGE_S = 3`（`slot_index = c % 3`，smem `165,016 B`），`stage_v` 为 `{16,1,64}` 无 swizzle，`sreg/pred_acc/qpred_acc` 各减半至 `8,192 B`，平面步距仍为 `40` |
| `VTT=64`（参数） | `!vt16 && 4*HV*N > sm_count` 且满足 `T >= 4096`、`ceil(T/64) <= 2*N` 或 `ceil(2*HV*N/sm_count)*3 < ceil(4*HV*N/sm_count)*2` 之一（`3178–3191, 2366–2368, 3465–3469`） | `vt` tile `32 → 64`，`grid.x = 2`；`NPER=4`（B 碎片 x4 装载）；**关闭 `KSPLIT`**（`accKx/accQx` 奇偶对折为单累加器，`pred_acc/qpred_acc` 的 `parity` 轴 `2 → 1`，累加结合序改变）；`shi/uh/duh/rhh` 步距扩至 `72`；`stage_v` 改 `SWIZZLE_128B` 并按 stage 地址取运行时相位 `vph`；环深度保持 `2`（smem `151,696 B`，`sreg` `32,768 B`）；`final_state` 暂存 `[128][65]` FP32 跨过 `stage_kq` 占用 `stage_v` 首部 `508 B` |
| `vt32 NSTAGE=3`（参数） | 已选 `VTT=32` 且 `floor((T + 63*N) / (64*N)) >= 8`（`vt16` 恒 `3`，`vt64` 恒 `2`；`2910–2941, 3424–3430`） | `stage_*` 环深度 `2 → 3`、`slot_index = c % 3`，序幕预发 2 个 chunk，`CHAIN` 动态 smem 由 `123,024 B` 增至 `171,160 B` |
| `vt64 Lc=1` 迷你尾块（局部） | 仅 `VTT=64` 编译期启用，本 chunk 有效行数 `Lc == 1` 时进入（`2438–2552`） | 行 0 属主 lane 标量计算 `rhs/u/o` 并直写 `out`（及 `core`），省去 `u/du` 全平面、`tinv@rhs_h`、`qkd@u_h` 与 7a/7b 暂存合并写；`DU_0` 借用 `rhh` 行 0 经单次 `__syncthreads()` 发布，各 warp 随后读 `stage_kq` 的 `k0` 做 rank-1 外积更新 `Sreg` 与 `sShi`（偶序号末块写 `final_state` 时的 `stage 0` 退休等待缺失见“范围”） |

### Prep 组织、直读与序幕重排（局部）

- 基线：并列 [root](root.yaml) 与 [segmented](segmented.yaml) 的 `Work_schedule::PREP`、`Tile_pipeline::PREP.MAIN`，以及 [root](root.yaml) 的 `Work_schedule::CHAIN`、`Tile_pipeline::CHAIN.MAIN`。
- 差异与联动及条件：

| 变体 | 适用基线与条件依据（`gdn_kernel.cu`） | 具体变化与参数联动 |
| --- | --- | --- |
| GQA 共享 prep（局部） | `root`、`segmented`；`slots_ub * H >= sm_count`（`3358, 3391–3400`） | `PREP` 网格改按 `key_head_count`（`Gseg==1` 为 `(H, slots_ub)`，分段为 `(slots_ub, H)`）；每 `(slot, kh)` 一 CTA 仅做一次 `load:kq_image`、`compute:gram/qk` 与 `store:kq_mirror`，再串行循环组内 `ratio = HV/H` 个 value head 执行 `system → inv8..64 → slot_publish`，`vec_stage` 扩为多套。`GQA + segmented` 域内可达，作为无独立 YAML 的记录组合 |
| GQA `RATIO` 特化与双头锁步（局部） | GQA 共享 prep 之上，`RATIO = HV/H ∈ {2, 3, 8}`（域外落 `0`；`313–399, 466–648, 672–1267`） | 序幕一次预取全部头的 `g/beta`，空闲 warp 在 K/Q TMA 在途期间算完各头 `decay_stats`（`RATIO>=4` 分三相位扫描）；`gram` 与 `qk` 融合共享 B 碎片；以双套 `sys_plane/tinv_plane` 双头锁步求逆并按组发布，输出字节不变 |
| Prep 网格次序 `qhfast`（参数） | `root`（`true`）、`segmented`（`false`）；`qhfast = (Gseg == 1)`（`180–202, 3385–3400`） | `Gseg == 1` 取 head-fastest `(GX, slots_ub)`，`Gseg > 1` 取 slot-fastest `(slots_ub, GX)`；仅对调 `dim3` 与核内 `blockIdx.x/y` 取轴，单 CTA 工作与发布字节不变 |
| `KQTMA` 直读 q/k（局部） | `root`（`vtt >= 32`）；`!l2norm && Gseg == 1 && vtt >= 32 && need + qin_bytes > l2_bytes`（`651–667, 2109–2128, 3320–3321`） | `PREP` 收 `skip_q=1` 省去 `store:kq_mirror`（不写 `scratchq`）；`CHAIN` 的 `load:stage` 改发 4 个 `{64,64,1}` `SWIZZLE_128B` TMA box 直读输入 `q/k` 进 `stage_kq` 并计入同一 stage `expect_tx`，消费端 `ldmatrix` 图像不变 |
| `PRE0` chunk-0 预等待预取（局部） | `root`（经 `VTT=16`）；`!kqtma && !l2norm && Gseg == 1 && vtt == 16 && perm == nullptr`（`1998–2026, 2100–2107, 3353–3354`） | `CHAIN` 在 `await:prep_grid` 之前用输入 tensor map 预发 chunk-0 的 4 个 k/q box 与 1 个 v box（单次 `expect_tx`）；wait 后 `load:stage(0)` 走 `planes_only` 仅补发 `PREP` 平面与向量 bulk |
| LPT 请求排列 `perm`（局部） | `root`；`8 <= N <= 32` 或 `(2 <= N <= 32 && chain_blocks > sm_count)`（`265–280, 2041–2057, 3343–3347`） | `PREP` 的 `block(0,0)` 按每请求 chunk 数降序稳定插入排序写尾垫 `perm[N]`；`CHAIN` 在 `await:prep_grid` 后取 `n = perm[blockIdx.z]` 并将 `load:state0` 移至 wait 后，使长链优先占据低 `blockIdx.z` |

### 段并行宽 tile 臂 VTT=64（参数）

- 基线：[segmented](segmented.yaml) 的 `PASS_A/PASS_C` step、`Segment affine transfer and composition`、depth-3 `stage_*` 环与 `165,016 B` slab（共享计算见 [root](root.yaml) `CHAIN.MAIN`）。
- 差异与联动：同一 `MODE=1/2` 设备文本按 `VTT=64` 实例化为 `gdn_chain<2,64,1>` 与 `gdn_chain<2,64,2>`，链网格 `(2, HV, N*Gw)`；stage 环深度 `3 → 2`，动态 smem 改为 `151,696 B`，`b_seg` 按 `vtt=64` 扩为 `[128][64]` FP32，并同步应用 `VTT=64` 档的全部联动（`NPER=4`、关闭 `KSPLIT`、平面步距 `72`、`SWIZZLE_128B` `stage_v` 及 `Lc=1` 迷你尾块路径）。跨段组合近似与 `amax > 1e-3` 串行回退同 `segmented`，仅作为记录形式保留，无独立 YAML。
- 条件与依据：仅在 `vt16` 初始选中类内评估重分块；令 `cols64 = 2 * HV * N`，`Gw` 为满足 `Gw <= 8`、`cols64 * Gw <= sm_count` 且 `avg_chunks / Gw >= 32` 的最大整数，当 `Gw >= 2 && avg_chunks >= 192` 且（窄臂 `Gc < 2` 或 `Gw * 100 > Gc * 205`）时选宽臂（`gdn_kernel.cu:3222–3249, 3455–3457`）。

### 分派门相容关系（组合）

- 基线：[root](root.yaml) 与 [segmented](segmented.yaml) 及上述各参数/局部条目。
- 差异与联动：
  - `L2NORM` 按 `use_qk_l2norm_in_kernel` 与 `root` / `segmented` 任意路径前置组合，但强制关闭 `KQTMA` 与 `PRE0`（两门均含 `!l2norm`）。
  - `KQTMA` 排除 `segmented`、`VTT=16` 与 `PRE0`，可与 `VTT=32`（含 `NSTAGE=3`）、`VTT=64`、`perm`、`GQA` 并存。
  - `PRE0` 仅与无 `l2norm`、无 `perm` 的单波非分段 `VTT=16`（及可选 `GQA`）相容。
  - `perm` 可与 `VTT=32/64`、`KQTMA`、`GQA` 并存，与 `PRE0` 互斥；`perm` 门与分段门独立计算（`148 SM` 下因分段门限制 `N` 与 `chain_blocks` 而不共现，更大 `sm_count` 且 `N >= 8` 时可同时命中；完整 `segmented.yaml` 的 `when` 限定无 `perm`）。
  - `GQA`（含 `RATIO` 特化）与 `root` 各档位及 `segmented`（窄臂/宽臂）均可正交组合。
- 条件与依据：host 门控见 `gdn_kernel.cu:3320–3358, 3390, 3450–3478`；完整 launch 实例登记见 `2943–2992`。
