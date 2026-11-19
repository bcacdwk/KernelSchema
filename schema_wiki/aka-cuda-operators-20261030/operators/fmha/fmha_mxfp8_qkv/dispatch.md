# fmha_mxfp8_qkv

文件版本、来源与检查信息见[当前元数据](.extraction/current.json)。

## 范围

本目录含三份完整设计：[root](root.yaml)、[split_kv](split_kv.yaml)、[mma_sync](mma_sync.yaml)。仅 `root` 保存共享 Contract，三份设计的三个 Design 区域全部本地定义，无整区 `from` 引用。固定来源为 `KernelSchema@24d2b3d3ca9d4907c4e7efd9607cf78cfb1a52b2:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate`（下文源码路径相对该目录）。

**构建与分派入口。** `solution.py` 按 `{major}.{minor}a → {major}.{minor} → auto` 顺序编译并在 `compile()` 缓存一次 `tcg_available()`；是否发射 `tcgen05` 由加载二进制决定，与单次输入无关。arch-specific 构建（`tcgen05` 可用）按 `plan_geometry` 在单核 fused 与四核 split-KV 间分派；非 arch-specific 构建（`tcgen05` 体为 stub）走 `mma.sync` 路径。三份完整 YAML 的 `when` 互不重叠，均只承担 `head_dim = 128` 的代表子域，无隐式兜底。下列可达形式仅由下方记录承载，不立独立 YAML：`head_dim = 256` 的全部 fused / split / `mma.sync` 几何；`D = 128` 低波次 fused 的 key-split、group-split 及 `lpt ∈ {0, 2, 3}`（含无掩码多波几何）；tensor-map 不可用时的 `cp.async` 降级；split 链的 PSK、融合/两遍 scan、非空 `GMAX` chunk 不足 8 组的顺序体、wide combine 及 `mma.sync` split 组合。域外 scalar 路径与默认关闭的实验宏（含不可达的 Phase E 死分支）不属于完整设计集合。

**调用条件与已知限制。**
- **准备期固定参数**：softmax scale、`is_causal`、`window_size_left`（`None` 或负哨兵均无左窗）在准备期固定；逐次调用的 `scale_softmax` 实参被忽略，其余 inert 参数不进内核。
- **Workspace 回退与 CUDA Graph**：准备期 `scratch_workspace_bytes()` 与 `forward` 共用同一 `plan_geometry`；运行期仅在 `use_tc && nsplit > 1 && workspace 存在且 numel >= plan.workspace` 时走 split-KV，缺失或不足时静默回退 fused。CUDA Graph 按指针、stride、几何、掩码、scale 位及 `nsplit/gm_nsplit/use_split/maps_valid/workspace_bytes` 建键，同键第二次调用在侧流捕获 launch 序列并在调用流重放（不缓存结果，失败退回直接启动）。
- **源码缺陷与验证边界**：非 arch-specific 构建的 `mma.sync` split 链（`D = 128` 与 `256` 均受影响）将 `gm_grid.x` 设为 `qtiles128`，而 `fmha_tc_kernel` 按 `qph = 64/G`（`qtiles`）分行；当 `qtiles128 * qph < Sq` 时，未覆盖行的 `bm`（含首末 split 应写的 `-inf` 头尾）未初始化，`scan` 读取脏 workspace（`fmha_mxfp8_qkv.cu:1950–1979,1989–2032`）。既有采样与注释数据不证明完整动态域误差上界或外部验收器公式。

## 完整设计

| 设计 | 具体机制与保留理由 | 关系 |
| --- | --- | --- |
| [root](root.yaml) | `D=128` `tcgen05` block-scaled 单核 fused（`KSPLIT=1, TMA=1, QK64=1, lpt=1`）。1 CTA 覆盖 128 打包行（`qph128 = 128/G`）、128 线程一 lane 一行；每 128-key 组发两次 `N=64` block-scale MMA 写单块 64 列 `s_tile`（单次读出），按 `4·ln2` 延迟 rescale，E4M3 概率写 `p_tmem`，PV 累加进 `o_tmem`（总 256 TMEM 列，2 CTA/SM）；单核直接归一化原位写出 BF16 `o_tensor` 与可选 `lse_tensor`。 | 阅读入口，保存唯一 Contract；`when` 要求 `tcgen05` 可用、保持 fused、CTA 数 `> 2*sm_count`、准备期掩码开启且 K/V 满足 tensor map。 |
| [split_kv](split_kv.yaml) | `D=128` `tcgen05` 四核链 `GMAX → SCAN → SPLIT → COMBINE`（PDL 串联，`SPLIT` 的 `scan_wait` 下移至首次读 `m_table` 前）。`GMAX` 按 `gm_nsplit` 物化组最大值 `gmax_table`（首末段补 `-inf` 头尾）；单遍 flat `SCAN` 逐行重放延迟最大值递推得 `m_table/m_tail`；`SPLIT` 按 `nsplit` 分段累加并以 `exp(m_chunk_end - m_tail)` 加权写出 FP32 `num_part/den_part`；narrow `COMBINE` 一行一 CTA 求和归一化并生产 `o_tensor` 与 `lse_tensor`。 | 并列完整设计，三区本地定义；`when` 要求 `tcgen05` 可用、workspace 足够且 `nsplit > 1`、`base < 2*sm_count`、非融合 flat scan、非空 GM chunk `>= 8` 组、非 PSK、narrow combine 及 tensor map 可用。 |
| [mma_sync](mma_sync.yaml) | `D=128` 可移植 warp-level 单核 fused（`WARP=4`）。1 CTA 覆盖 `BM=64` 行，每线程持相距 8 行的两行 fragment；QK 用 `mma.sync m16n8k32` E4M3 配合每 32 通道块 E8M0 的 rank-1 折算，V 先按 key 轴 E8M0 解码为共享内存 BF16 再做 `m16n8k16` PV，K/V 与 scale 经两级 `cp.async` staging；输出驻寄存器，4-lane shuffle 归约分母后单核写出 `o_tensor` 与 `lse_tensor`。 | 并列完整设计，三区本地定义；`when` 要求加载二进制不能发 `tcgen05`，且保持 fused 或 workspace 缺失/不足。 |

## 变体与组合

下列记录定位对应完整 YAML 中的已有对象，不修改基线的参数或 `when`，也不构成独立完整 YAML。

### head_dim=256 fused 形式（局部）

- 基线：[root](root.yaml) 的 `Tile_pipeline::MAIN`（`k_ring/v_ring/s_tile/o_tmem/p_tmem/sf_cols` 与 `Role.MAIN`）、`Work_schedule::MAIN` 及 `Math_reshape` 的 PV / epilogue rewrite。
- 差异与联动：实例 `KSPLIT=2, TMA=1`（256 线程双 warpgroup）。QK 单发覆盖 128 键，score 读出与 softmax 按 64 键分半并经共享内存跨 warpgroup 合并组 `max`（精确）与组末 `den`（按 key 序折叠重结合），各自写 `p_tmem` 半段，PV 单发覆盖 `128×256`；`o_tmem` 占 `[128,384)`，TMEM 分配扩至 512 列（1 CTA/SM），rescale 与 epilogue 按 128 通道分半；Contract `bias` 由 16 变 1（magic 折叠退化为无操作）；启用 `SF_AHEAD`（scale atom 超前 3 组、3 槽 smem 环 + 双奇偶 TMEM 槽）与 `MIDPIPE`（迭代中段发 `QK(g+1)` 并退休 `PV(g-1)`，K 与 `SFB_V` 扩至 3 槽）。
- 条件与依据：`use_tcg && !use_split && D == 256` 时恒选，`lpt` 常驻槽按 `slots = sm_count` 计算（`fmha_mxfp8_qkv.cu:1761–1857`）。

### D=128 key-split 流水形式（局部）

- 基线：[root](root.yaml) 的 `Tile_pipeline::MAIN`（`Pipe`、`k_ring/v_ring`、`s_tile`、`o_tmem`、`Waiting`）与 `REALIZATION` 的 bulk staging 及常驻容量请求。
- 差异与联动：实例 `KSPLIT=2, TMA=0`（256 线程双 warpgroup，64 键/64 通道分半合并同 `D=256` fused）。K/V 环加深至 4 槽，新增共享第二块 128 列 score tile `S1=[128,256)`，改变 TMEM 布局——root 已在 TMEM 的 `o_tmem` 从 `[96,224)` 移至 `[256,384)`，总分配由 256 列扩至 512 列（1 CTA/SM）（`fmha_tcg.cuh:1894,1946`）；软件流水在迭代 `g` 中段向另一 score tile 发 `QK(g+1)`，`PV(g-1)` 推迟到 `g+1` 开头等待；staging 改用 16B `cp.async`。
- 条件与依据：`use_tcg && !use_split && D == 128 && fused_ctas <= sm_count && gspan < 8`（`gspan` 为最宽 CTA 实际遍历的 128-key 组数；`fmha_mxfp8_qkv.cu:348–364,1793–1818`）。

### D=128 group-split 形式（局部）

- 基线：[root](root.yaml) 的 `Tile_pipeline::MAIN`（`Role.MAIN`、`k_ring/v_ring`、`s_tile/p_tmem/o_tmem`）与 `Work_schedule::MAIN.Mapping`。
- 差异与联动：实例 `KSPLIT=2, TMA=0, GSPLIT=1`（256 线程双 warpgroup）。两 warpgroup 按奇偶交替处理整组 `cLo + w + 2k`，各自拥有独立 3 槽 K / 2 槽 V 环、独立 score tile 与 TMEM `P` tile（`o_tmem` 占 `[256,384)`，总 512 列、1 CTA/SM），同步收窄为 128 线程 warpgroup barrier；`(mstate, den)` 经共享内存与 `mbar_ms/mbar_den` 点对点交接（顺序 `den`，无折叠），每组 PV 等对侧上一组 PV 退休后再发射以按组序累加同一 `o_tmem`，末态由持 `cHi` 的一侧交接；恒用 16B `cp.async` staging，`lpt = 0`。
- 条件与依据：`use_tcg && !use_split && D == 128 && fused_ctas <= sm_count && gspan >= 8`，优先于 key-split 流水（`fmha_mxfp8_qkv.cu:1813–1818`）。

### K/V cp.async 降级（参数）

- 基线：[root](root.yaml) 与 [split_kv](split_kv.yaml) 的 `load:k_block / load:v_block` 及 `REALIZATION` 的 tensor-map bulk copy 条目。
- 差异与联动：`TMA = 0` 改用 16B `cp.async`（同 commit 组附带 scale atom）写入逐字节相同的 `SWIZZLE_128B` XOR 布局，越界键由 `src-size = 0` 归零，完成同步由每槽 `mbarrier expect_tx` 换成 `cp.async` `commit/wait_group`；在 `D=128` fused 上联动关闭 `DPV`，`PV(g)` 回到迭代脚部等待。
- 条件与依据：`encode_tile_map` 失败（驱动入口缺失、base/stride 非 16B 对齐或范围不合法，`maps.valid == false`）时自动降级；descriptor 按 base、3 个 stride 与 4 个 extent 在 8 槽缓存（`fmha_mxfp8_qkv.cu:2203–2206`）。

### CTA 派发序 lpt（参数）

- 基线：[root](root.yaml) 的 `Work_schedule::MAIN.Schedule.Worktile_order`（`lpt = 1`：全网格线性序反转后按 `(qtile 慢键, kv_pair 快键)` 双射重映射）。
- 差异与联动：`p.lpt` 另取 `0`（自然升序）、`2`（pair-class 互补序：前 `sm_count` 个重侧从末 tile 降序，其余轻侧升序并反转 head rank 以错开共享边界 tile）或 `3`（奇偶交错，默认不用）；均为网格双射，数值不变。
- 条件与依据：仅 `tcgen05` fused 分支设置（split 链与 `mma.sync` 恒 `lpt = 0`）。记常驻槽 `slots`（`root` 为 `2*sm_count`，`D=256`/key-split/group-split 为 `sm_count`）：有掩码且 `fused_ctas > slots` 取 `1`；有掩码且 `sm_count < fused_ctas <= slots` 取 `2`；其余（含无掩码）取 `0`（`fmha_mxfp8_qkv.cu:1826–1857,2181`）。

### 编译期 tile 与概率开关（参数）

- 基线：[root](root.yaml) 的 `s_tile`、`p_tmem`、`o_tmem` 及 `Math_reshape` “Biased probability and E4M3 quantization”。
- 差异与联动：`QK64 = 0` 改单发 `N=128` 128 列 score tile、每组三次读出，`O` 退回寄存器（仍 256 TMEM 列、2 CTA/SM）；`FMHA_P_IN_TMEM = 0` 将概率 tile 移入 swizzled smem 并补 `fence.proxy.async.shared::cta` + CTA barrier（`GSPLIT` 禁止该项）；`FMHA_BIAS_MAGIC_FOLD = 0` 将 `2^4` 偏置拆回独立乘法，`FMHA_UNIT_BIAS = 0` 改用运行期 `bias`。
- 条件与依据：默认构建恒取 `QK64 = 1`、`P_IN_TMEM = 1`、`BIAS_MAGIC_FOLD = 1`、`UNIT_BIAS = 1`，仅修改编译期宏时切换。

### 填充跳过行序 PSK（参数）

- 基线：[split_kv](split_kv.yaml) 的 `Work_schedule` `Step.GMAX` 与 `Step.SPLIT` 的 head-major 行映射。
- 差异与联动：`PSK = 1` 将 128 行置换为 query-major（`i = q0 + (r >> log2G), h = kh*G + (r & (G-1))`），使 `i >= query_len` 死行聚为整 warp 前缀；`!__any_sync(valid)` 的死 warp 整块跳过 softmax 计算与概率存储，保留全部 CTA barrier、`mbar_wait` 与对齐 `tcgen05` 操作，输出逐位不变；`D=128` 下同时联动关闭 FP32 partial 的 smem 转置 staging（改逐线程直写）。
- 条件与依据：仅 split 链启用，门为 `2 * Sq * G <= qtiles128 * 128`（每 tile 至少半数填充行；fused 不启用）。

### 融合 scan 变体（局部）

- 基线：[split_kv](split_kv.yaml) 的 `Step.SCAN`、`Flow.Launch`、`Step.SPLIT`（`load:m_init/m_next`、`scan_wait`）、`Step.COMBINE` 及 `Math_reshape` “Exact replay of the delayed maximum”。
- 差异与联动：取消独立 `SCAN` 启动（含两遍链）与 `scan_wait`；`SPLIT` 每线程从 `gmax_table` 前缀重放本段初态并直接携组最大值递推，以未加权（`wgt = 1`）写 `num_part/den_part`，另写段末态 `part_m[split][row]`（workspace 增 `off_pm`），由末段 `blockIdx.z == nsplit - 1` 唯一重发布 `m_tail[row]`；`COMBINE` 切 `FWGT = 1`，在共享内存内按 `exp(part_m - m_tail)` 同时加权分子与分母。
- 条件与依据：记两遍预判 `scan2p_gate = (T >= 128 && rows_total <= 512 && !has_window)`，`span` 为最宽 CTA 实际遍历的 128-key 组数（同 `gspan`）；融合优先，门为 `use_tcg && D == 128 && span <= (scan2p_gate ? 160 : 48)`（`D=256` 与 `mma.sync` 恒关闭；`fmha_mxfp8_qkv.cu:348–364,780–788,1640–1653`）。

### 两遍 scan 与块尺寸（局部）

- 基线：[split_kv](split_kv.yaml) 的 `Step.SCAN`（单遍 flat scan 与 `scan_block`）及 [root](root.yaml) 的延迟递推。
- 差异与联动：两遍 scan 以两次启动替代 flat scan——pass A `fmha_segmax_kernel`（每 16 组求段最大写 `segmax`，入口即发 PDL trigger），pass B `fmha_scan_kernel`（固定 256 线程，段最大不足以触发 `4·ln2` 跳变时跳过组最大值读取与比较，但仍向 `m_table` 全量写常量 `m` 保持无空洞，入口等待后发 PDL trigger）。走单遍 flat scan 时，`scan_block` 在 `T < 128` 取 256，否则取 `{32, 64, 128}` 中使 `ceil(rows_total / block) <= sm_count` 的最细值（都不满足取 256）。
- 条件与依据：融合优先，未融合时才执行两遍，门为 `!scan_fused && scan2p_gate`（即 `!scan_fused && T >= 128 && rows_total <= 512 && !has_window`；`fmha_mxfp8_qkv.cu:1640–1655,1707–1745,1973`）。

### GMAX 循环、partial 写出与 combine 形态（参数）

- 基线：[split_kv](split_kv.yaml) 的 `Step.GMAX`（流水循环、`gm_nsplit`）、`Step.SPLIT`（`store:part_a/part_b`）与 `Step.COMBINE`（narrow 128 线程、`den_smem`）。
- 差异与联动：`GMAX` 按 `gm_nsplit = clamp(resident/base, nsplit, T/2)` 切分；`D=128` 仅当本段非空且组数 `cHi - cLo + 1 >= 8`（`cHi - cLo >= 7`）走 3 槽软件流水体，不足 8 组或 `D=256` 走顺序体（`fmha_tcg.cuh:4093`）。`SPLIT` 的 FP32 `num_part` 在 `D=256` 或 `D=128 && !PSK` 时经空闲 smem 转置后合并写出，其余逐线程直写（`BF16` partial 因峰值分布超容差已默认禁用）。`COMBINE` 除 narrow（128 线程，`den_smem` 上限 `nsplit <= 128`）外，可切 `wide`（512 线程分 4 个 split-group 各归约 `ceil(nsplit/4)` 段再经共享内存合并，上限 `nsplit <= 512`，改变加法树）。
- 条件与依据：`wide` combine 门为 `(rows_total < 2 * sm_count && nsplit >= 8) || nsplit > 128`（`fmha_mxfp8_qkv.cu:931–932`）。

### head_dim=256 split-KV 与三轴相容组合（组合）

- 基线：[split_kv](split_kv.yaml) 的四核链与 [root](root.yaml)（双 warpgroup 计算机制同“head_dim=256 fused 形式”）。
- 差异与联动：`D=256` split 链的 `GMAX` 与 `SPLIT` 换用 `D=256, KSPLIT=2` 实例——`GMAX` 无 `O` 累加器，默认仍占 256 TMEM 列（最多 2 CTA/SM）并走顺序体，`SPLIT` 占 512 TMEM 列（1 CTA/SM）（`fmha_tcg.cuh:1885,1964–1966`）；`SCAN` 共用 flat 或两遍内核（不启用融合 scan）；`COMBINE` 仍按统一的 `(rows_total < 2 * sm_count && nsplit >= 8) || nsplit > 128` 规则选择 narrow 或 wide，并非固定 wide（`fmha_mxfp8_qkv.cu:916–932,1759`）。此外，在 `D=128` split 链上，融合 scan（携 `FWGT`）、`PSK` 与 `narrow/wide` combine 三轴接口正交相容，可按各自门控同时生效。
- 条件与依据：`use_tcg && use_split && D == 256`（`plan_geometry` 按 1 CTA/SM 常驻与 1.25 置信裕度定价 `nsplit`）。

### mma.sync 的 head_dim=256 与 split-KV 组合（组合）

- 基线：[mma_sync](mma_sync.yaml) 的 `fmha_tc_kernel` 与 [split_kv](split_kv.yaml) 的四核链及全局交接；共享 Contract 见 [root](root.yaml)。
- 差异与联动：`D=256` fused 实例（`fmha_tc_kernel<256,4,PASS_FUSED>`）保持 `BM=64`、每线程两行与两级 `cp.async` 组织，仅按 `D=256` 翻倍 32 通道块数（`4→8`）、寄存器输出列（每线程 `64→128` 个 FP32）、scale atom 与 `v_decoded` 尺寸，`bias` 由 16 变 1，动态 smem 升至 225344 B 使常驻上限由 `2*sm_count` 降为 `sm_count`。与 split-KV 组合时（`D=128` 与 `256` 同构），`GMAX` 与 `SPLIT` 换用 `fmha_tc_kernel<D,4,PASS_GROUP_MAX/PASS_SPLIT>`，复用 flat/两遍 `SCAN` 与 FP32 非融合 `COMBINE`（永不融合 scan 或用 BF16 partial）。
- 条件与依据：`!tcg_available()` 下按 `!use_split` / `use_split` 分派（`fmha_mxfp8_qkv.cu:1983–2040`）。两条 `D` 的 `mma.sync` split 链共有前述行覆盖缺陷：当 `qtiles128 * qph < Sq` 时，`GMAX` 漏写尾部行的 `gmax_table`（含 `-inf` 头尾），`SCAN` 读取未初始化 workspace。
