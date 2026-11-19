# dual_cache_latent_bf16

文件版本、来源与检查信息见[当前元数据](.extraction/current.json)。

## 范围

本目录含四份完整设计：[root](root.yaml)、[split_kv](split_kv.yaml)、[tcgen05](tcgen05.yaml)、[tcgen05_packed](tcgen05_packed.yaml)。仅 `root` 保存共享 Contract，四份设计的三个 Design 区域全部本地定义，无整区 `from` 引用。固定来源为 `KernelSchema@24d2b3d3ca9d4907c4e7efd9607cf78cfb1a52b2:kernel_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/candidate`（主实现 `dsv4_kernel.cu` 与入口 `solution.py`）。

**归口边界与构建分派。** 生产 `auto` 策略在共享 Contract 有效域内的全部执行机制均由四份完整设计加下方变体/组合记录归口，但四份 YAML 的 `when` 不拼成全域覆盖（如 `HB ∈ {16, 32}` 的非 split/split mma 几何、`CT = 128` combine，以及单次调用内按 token range 混用 `tcgen05_packed` 与替代 mma 设计的 dual 编排均由记录承载）。扩展按 `sm_103a → sm_100a`（带 `-DDSV4_TCGEN05`）`→ sm_103/sm_100` 顺序取首个编译成功目标；非 `'a'` 构建把全部 tc/packing 计划折叠回 `mma.sync` 且不触发 dual 探测。`root`（`cfg2`：`HB=64, KC=64, STAGES=2, SPLIT=false`）是 forced 家族基线（`DSV4_CFG=2`）与阅读入口，不是 `auto` 策略的落点——`auto` 下 `H >= 64` 满波在 `'a'` 构建走 `tcgen05`、无 tc 构建走 `cfg6`（`KC=128, STAGES=1`），不满波则收缩 `HB` 或转入 split（`dsv4_kernel.cu:4808–4813,5035–5086`）。

**未交付方案与研究组合。** `fused_combine`（`g_fuse_combine = 0`，按输出行 tagged 64-bit 计数在 `attn_kernel` 末位到达 CTA 内复用 `CombSmem` 直接归并，拒绝 `DQ > 1` 与 fp16 partial）与 `Paired cluster tile`（`PR`，force-only `cfg 61/62`，`cta_group::2` 双 CTA 合成逻辑 `M=128` tile 并经 cluster barrier 交换统计）为默认关闭或未晋升的独立方案，未交付完整 YAML；`tcgen05 + split_kv` 仅为 forced 研究组合；`naive_kernel`、测速/错值归因臂与 `dsv4_run_ptr` 不属生产路径。

**调用条件与真实源码缺陷。**
- **Host 与 Workspace**：`Model.forward` 要求 `kv_layout == 'HND'`、`cum_seq_lens_q` 非空且三个保留槽为 `None`；Python 仅在 `workspace_buffer` 为 CUDA 张量且 `>= 32 MiB` 时绑定（按对象 identity 缓存），否则静默替换为私有 32 MiB CUDA 缓冲，C++ 在计划所需字节仍超出已绑定容量时再另分配本次私有缓冲（`solution.py:152–171`、`dsv4_kernel.cu:6419–6423`；两份非 split tc 设计不读写 workspace）。
- **源码共享内存地址重叠（尚未修复）**：两份 tc 设计对应的固定源码 `tc_attn_kernel` 中，`bar_land`（`mbar + 4*STG`，`STG` 个 64-bit 字）与 `tmem_base_s`（同址 `mbar + 4*STG`）及紧随其后的 packed union 控制区 `ukey`（`tmem_base_s + 1`）同址重叠（`dsv4_kernel.cu:2111–2115,2497–2515`），packed 控制区同样受影响。Schema 抽象设计不复制该坏地址，也不据此声称固定源码已获全域正确性保证。

## 完整设计

| 设计 | 具体机制与保留理由 | 关系 |
| --- | --- | --- |
| [root](root.yaml) | `mma.sync` 非 split 单核方案（`cfg2`：`HB=64, WARPS=8, QKS=16, KC=64, STAGES=2, SPLIT=false`）。1 CTA 承担 `(token, 64-head block)`，双池 KV 行经 16B `cp.async` 聚集进两级 swizzle ring，前 16 个 k-step Q fragment 驻寄存器，块级 online softmax 写 BF16 `p_tile`，PV 后在 epilogue 折入 sink、归一化并单次提交直写 BF16 `out`（不读写 workspace）。 | 阅读入口，保存唯一 Contract，三区本地定义；forced 家族基线，`when` 承担 `H >= 64`、单整批提交且无 key split（含同 launch 的 graph replay）。 |
| [split_kv](split_kv.yaml) | `mma.sync` split-KV 双核方案（`HB=32, QKS=32, KC=64, STAGES=3, SPLIT=true` + `combine_kernel<32, CT=512, CM=1>`）。工作轴增 `key_split`（`2 <= splits <= 128` 且受 workspace 限制），`ATTENTION` 按连续 chunk 段写出未归一 FP32 `O` 与 base-2 `m/l`（空段写哨兵）至 `workspace_buffer`；第二核 `COMBINE` 一 CTA 一输出行，跨 split 归并、折入 sink、归一化并唯一写出 `out`。 | 并列完整设计，三区本地定义；`when` 要求 `HB=32`（`H >= 32`）、`splits > 1` 及整批启动（非 token range）。 |
| [tcgen05](tcgen05.yaml) | `tcgen05` warp 特化单核方案（`cfg90` 默认臂，`gt=1, CK=64, STG=2, RD=1, QIS2=true, SX2=true, TMA=true, MBA=true, REV=true`）。256 线程 CTA 承担 `(token, 64-head block)`，warps 4–7 以 TMA `gather4`（跨池四行组回退 `cp.async`）加载 KV，warps 0–3 做 `RD=1` 惰性 max softmax 且由 `tid32/tid64` 双路半 latent 发射 QK、`tid0` 发射 PV（TMEM 512 列含双 `O` 半块与双 score 缓冲），epilogue 折入 sink 归一化后经死 KV ring staging 写出 `out`。 | 并列完整设计，三区本地定义；仅存在于 `'a'` 后缀构建，`when` 要求 `H >= 64`、CTA 数满一波（`T * ceil_div(H, 64) >= sm_count`）、无 key split。 |
| [tcgen05_packed](tcgen05_packed.yaml) | `tcgen05` 多 token packing 单核方案（`cfg90` 的 `gt ∈ {2, 4, 8}` 分支）。1 CTA 将 `gt` 个连续 token 打包为 `<= 64` 个 `(token, head)` 行，先在共享内存开放寻址 hash 表构建组内去重 `union_key`（pool-major 放置，同 token 重复列出的同一行分配独立第二 entry 以保持重复索引逐次贡献）与位掩码 `member_mask`，再在共享键集上运行 `tcgen05` 流水并直写 `out`；`gt > 1` 静态禁止 key split。 | 并列完整设计，三区本地定义；仅存在于 `'a'` 后缀构建，`when` 要求 `H < 64`（2 的幂），`gt` 取满足 `gt*H <= 64`、`gt*index_width <= 2560` 且组数满一波的最大 2 的幂（允许组边界起算的 range 启动）。 |

## 变体与组合

下列记录定位对应完整 YAML 中的已有对象，不修改基线的参数或 `when`，也不构成独立完整 YAML。

### mma.sync 几何与 auto 选择链（参数）

- 基线：[root](root.yaml) 的 `Work_schedule::MAIN.Decomposition`、`Tile_pipeline::MAIN.Storage` 与 [split_kv](split_kv.yaml) 的 `ATTENTION` 同名对象及 `REALIZATION` 的 `cp.async` 组计数条目。
- 差异与联动：`HB ∈ {16, 32, 64}`（缩放头块划分与 `query_tile/p_tile/o_acc/softmax_ctrl` 行数）、`KC ∈ {64, 128, 192}`（缩放 `chunk_count` 与槽字节）、`STAGES ∈ {1, 2, 3}`（split 路径槽原点改为 `(c - chunk_lo) % STAGES`，且 `STAGES >= 3` 时越过 `chunk_hi` 必须提交空 `cp.async` 组以匹配 `wait_group` 组计数）、`QKS ∈ {0, 8, 16, 32}`（`QKS = 32` 全缓存 Q fragment，query staging 可复用 ring slot 0）。
- 条件与依据：生产 `auto` 选择链（`force_cfg <= 0`）：`'a'` 构建下 `H >= 64` 满波选 `tcgen05`；`H < 64`、`pack_gt > 1` 且替代 mma `splits == 1` 选 packing（直走或 dual）；其余走 mma——满波先切 `KC=128, STAGES=1`（`cfg6/7/8`），不满波时若收缩 `HB`（`64→32→16`）可增加 CTA 数（平局取少 split）则收缩；`nchunk_max > 4` 且 `base < sm_count` 时向一波分裂且不过一波（`base >= 4` 且不足半波允许每 split `>= 64` 键的亚 chunk 细分，受 128 与 workspace 上限钳制）；最后 `STAGES == 1` 且 `base < sm_count` 在满足 `solo / ring_fills / chunk_pads` 且不损失 CTA 时重选窄块深环（`cfg4/5`；`cfg6` 因 `base >= sm_count` 永不重选回 `cfg2`）（`dsv4_kernel.cu:4797–4885,5035–5086`）。

### EI 早发重排（局部）

- 基线：[root](root.yaml) 的 `Tile_pipeline::MAIN` 与 [split_kv](split_kv.yaml) 的 `Tile_pipeline::ATTENTION`（`compute:meta`、`load:kv`、`softmax_ctrl`）。
- 差异与联动：`EI_IDXPRE` 在 `index_width <= 1152` 时预取整行索引入新增 4624 B 共享内存供 `load:kv` 读取；`EI_METAPAR` 在 `request_count <= 64` 时借首块前空闲的 `warp_max/warp_sum` 并行加载 `cu_q/seq_lens/topk` 替代串行查询（超限各自回退全局/串行路径，输出逐位相同）。
- 条件与依据：`HB = 16` 的 `cfg5/cfg8` 及其 `DQ` 派生配置（`cfg123–126`）默认启用。

### DQ 输出列分区（局部）

- 基线：[root](root.yaml) 的 `Work_schedule::MAIN`、`Tile_pipeline::MAIN`（`matmul:pv`、`o_acc`、`store:out`）及 [split_kv](split_kv.yaml) 的 `ATTENTION` 对应对象。
- 差异与联动：`DQ ∈ {2, 4}` 个 CTA 共担同一 `(token, head_block[, key_split])`，重复 gather 与 phase-1 QK/softmax，各自仅计算并写出 `latent_dim / DQ` 的不相交输出列切片（`dq` 为最快网格维以命中 L2）；与 split 组合时各写自身 partial 列切片并重复写相同 `m/l`，由 `COMBINE` 整行归并（拒绝与 `fused_combine` 组合）。
- 条件与依据：仅在 `HB = 16`（`cfg5/cfg8`）且 `ctas * DQ <= sm_count`（一波内）启用，解析为 `cfg123–126`。

### Combine 形态与 partial 格式（参数）

- 基线：[split_kv](split_kv.yaml) 的 `Work_schedule::COMBINE`、`Tile_pipeline::COMBINE` 与 `Flow.Launch/Handoff`。
- 差异与联动：`CT ∈ {128, 512}` 按 combine 网格（`T * nb * HB`）不足一波取 512（4 个 128 线程段组归约），否则取 128（单组，省去跨组和）；生产固定 `CM = 1`（预取前 4 个 split 向量，零权重段既重定向读取段 0 又按值置零丢弃）、FP32 partial（`g_partial_fp16 = 0`）及普通流序双核启动（`g_pdl = 0`）；开发开关 `COPT`（去 barrier / 8 深预取）默认 0。
- 条件与依据：`launch_combine` 对全部 split 计划统一按上述网格门在 `CT = 512` 与 `128` 间选择。

### tc 几何与加载发射开发开关（局部）

- 基线：[tcgen05](tcgen05.yaml) 与 [tcgen05_packed](tcgen05_packed.yaml) 的 `Work_schedule::MAIN`、`Tile_pipeline::MAIN`（`LOADER`、`SOFTMAX`、`kv_ring`、`tmem_alloc`）。
- 差异与联动：生产 `cfg90` 固定 `CK=64, STG=2, RD=1, QP=true, REV=true, PREFETCH=true` 及 `TMA gather4 + MBA + QIS + QIS2 + SX2`（packed 侧另含 `IS`：在 `compute:union` 前将整组索引行 `cp.async` 入 `kv_ring` 闲置区）。Force-only 对照臂可切换 `RD=2`、`CK=32/STG=3..4` 或 `CK=128/STG=1`、`REV=false`、`SPIN` 轮询等待、16B `cp.async` loader、单路 QK（非 `QIS2`）或双路 PV（`PV2`），均须重算 smem 与 512 列 TMEM 预算。
- 条件与依据：非 `cfg90` 取值仅在 `DSV4_CFG` 强制指定时可达，`auto` 不选用。

### tc tile 与 split-KV 研究组合（组合）

- 基线：[tcgen05](tcgen05.yaml) 的 `Work_schedule::MAIN`、`Tile_pipeline::MAIN` 与 [split_kv](split_kv.yaml) 的 `key_split`、`Flow`、`COMBINE`。
- 差异与联动：`tc_attn_kernel` 在 `SPLIT = true` 时增加 `key_split` 网格维，epilogue 按 `part_stride<64>` 向 workspace 写出 TMEM 读出的原始未归一 FP32 `O` 及原始 `mrun`、`lrun`（虽预算局部 `inv`，但不沿用直接输出路径的本地 sink 折算与归一化）；随后 `run_tc` 强制 `pfp16 = 0` 调用 `launch_combine<64, false>`，sink 折入与最终归一化由 `COMBINE` 执行（`dsv4_kernel.cu:3467–3485,5459–5467`）。
- 条件与依据：仅 forced `cfg13/38`（`DSV4_CFG` / `DSV4_FORCE_S`）可达的研究组合，`auto` 不选用。

### Packing probe 门控与双流隐藏编排（组合）

- 基线：[tcgen05_packed](tcgen05_packed.yaml) 与 [root](root.yaml)（经“mma.sync 几何与 auto 选择链”解析且 `splits == 1` 的替代 mma 计划）。
- 差异与联动：`gt <= 2.6`（`gt = 2`）满足 `ratio <= gt` 先验免探测直走 `tcgen05_packed`；`gt > 2.6`（`gt ∈ {4, 8}`）成为 dual 计划，由 `overlap_probe_t` 采样 8 组索引求均值 `ratio = union * gt / sum|K_g|`（`<= 2.6` 选 packed，否则或超时选替代 mma）。当 `ngroups >= 2 * (sm_count - 8)` 且侧流/事件/pinned 可用时启用双流隐藏：probe 派往非阻塞侧流，主流同时先跑前 `(sm_count - 8) * gt` 个 token 的一波 `tcgen05_packed`（`gt * index_width <= 2560` 对任意重复率均合法），host 自旋取得结果后，剩余 token range（`Params.tok0/tok1/tile0`，两侧 `splits == 1`）按当选设计在侧流并发执行并以 event 汇合回主流，单次调用可按 token range 混用 `tcgen05_packed` 与替代 mma 设计；门不满足则同流串行先 probe 后整批启动。
- 条件与依据：`H < 64`、`'a'` 构建、`pack_gt > 1` 且替代 mma `splits == 1` 时触发。

### CUDA Graph 与注册 fastcall（组合）

- 基线：[root](root.yaml)、[split_kv](split_kv.yaml)、[tcgen05](tcgen05.yaml) 与 [tcgen05_packed](tcgen05_packed.yaml) 的 `Flow.Launch` 及 host 校验链。
- 差异与联动：普通 CUDA Graph 在同键（指针、尺寸、scale、plan 与开关）第二次调用时捕获 kernel 序列并在主流重放，仅固化启动配置与指针，重放仍读取当前张量内容；dual 计划、workspace 不足及 forced 计划不走普通 graph（任一失败永久回退直接启动）。注册 `fastcall` 默认关闭（`DSV4_FASTCALL=1` 开启），仅当参数对象同一且标量一致时重放；其中 dual 计划每次 fastcall replay 仍重做 probe 以读取当前索引内容。
- 条件与依据：Host 提交编排机制，不改变各设计的设备端计算与流序语义。
