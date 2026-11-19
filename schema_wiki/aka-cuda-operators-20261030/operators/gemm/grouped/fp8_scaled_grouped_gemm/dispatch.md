# fp8_scaled_grouped_gemm

文件版本、来源与检查信息见[当前元数据](.extraction/current.json)。

## 范围

**固定来源与入口。** 各 YAML 固定到 `KernelSchema@24d2b3d3ca9d4907c4e7efd9607cf78cfb1a52b2`。入口 [candidate/solution.py](../../../../../../kernel_wiki/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/candidate/solution.py)`::Model.forward` 仅做公开元数据检查（要求 `scale_granularity_mnk == (1, 128, 128)`、`scale_major_mode == 'MN'`，忽略 `mma_sm / backend / segment_alignment`）、编译加载与对象身份缓存，经 [candidate/fp8gg_bind.cpp](../../../../../../kernel_wiki/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/candidate/fp8gg_bind.cpp) 调用 [candidate/fp8gg_umma.cu](../../../../../../kernel_wiki/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/candidate/fp8gg_umma.cu)`::fp8umma_run` 并原样返回调用方传入的 `out`。

**完整设计集合与未收录分支。** `cc[0] == 10` 时编译 `fp8gg_umma.cu`，生产可达三个 kernel 实例。本目录收录两份完整 YAML：`root`（`engine == 0`，`umma_gemm<0>`）与 `wide`（`engine == 1`，`umma_wide<4,32768,0>`）；`root.yaml` 保存唯一共享 Contract，`wide.yaml` 三区全部本地定义，不使用整区 `from`。两份 `when` 在冻结形状域（`n ∈ {1536, 3072, 4096, 7168}` 均被 256 整除）上重叠，并列无隐式优先级，各自固定单一实例而不内嵌 host selector。第三个生产实例 `engine == 2`（`umma_wide<3,65536,1>`）是 `wide` 的环深与整块写回局部变化，不单列第三份完整 YAML。非 Blackwell 回退模块 `candidate/fp8gg.cu::fp8gg_bm64`（`mma.sync` + `cp.async`，`cc[0] != 10`）超出共享 Contract 硬件范围，未收录完整设计；当前冻结域五组 `(n, k)` 使 `kb_count = k / 128 ∈ {12, 16, 32, 56}` 恒为偶数，源码中的奇数 K 块尾对分支不在当前域内。

**源码 Host Selector（唯一分派入口）。** `sched_policy`（`fp8gg_umma.cu:1539–1655`）在缓存未命中时将 `m_indptr` D2H 并同步（失败退回 `engine = 0`），算出非空组数 `ne` 与总行块数 `mtiles = sum_g ceil(rows_g / 128)`，固定 `sareg = 0`、置 `pp = (mtiles == ne)`。当 `n % 256 == 0 && ne > 0 && nSM > 0` 时，令 `kb = kb_count = k / 128`、`items256 = mtiles * (n / 256)`，满足下表任一行即选 `engine = 1`，否则选 `engine = 0`：

| `kb` 范围 | 波次门限（`items256`） | 组内行块门限（`mtiles`） |
| --- | --- | --- |
| `kb >= 32` | `items256 >= 4 * nSM` | `mtiles >= 4 * ne` |
| `32 <= kb <= 48` | `2 * items256 >= 5 * nSM` | `mtiles >= ne` |
| `kb >= 16` | `items256 >= 8 * nSM` | `mtiles >= 2 * ne` |
| `8 <= kb <= 12` | `items256 >= 10 * nSM` | `mtiles >= ne` |
| `kb >= 16` | `items256 >= 8 * nSM` | `mtiles == ne` |
| `8 <= kb <= 20` | `items256 >= 4 * nSM` | `mtiles >= 2 * ne` |

命中 `engine == 1` 后，若 `kb <= 20 && 2 * mtiles >= 3 * ne` 则改判 `engine = 2`；`engine == 2` 启动失败时置 `g_w1g_off` 永久回退 `engine = 1`（`fp8gg_umma.cu:1631–1632, 1707–1762`）。

**必要调用与数值限制。**
- **后置 scale 算术边界：** 每个 128-K 块先在存储的 FP8 E4M3 上经 4 步 k32 `tcgen05.mma` 累加为 FP32 `blk`，再乘 `a_scale * b_scale` 做 FFMA 提升（`root.yaml:99–108`、`wide.yaml:31–40`）。结合次序不同于参考实现的先反量化后累加，无全域误差界；当两个有限 scale 的乘积在 FP32 溢出时，即使 payload 全零也可能因 `inf * 0` 产生 `NaN`。外部验收阈值（`atol=.01, rtol=.05, max_mismatch_fraction=0`）缺少比较器公式与相对基准，YAML 记为 `Source only`。
- **身份缓存与 `m_indptr` 陈旧性：** C 绑定层 weakref memo（`fp8gg_bind.cpp:63–84`）仅比对 6 个张量的 Python 对象身份，单节点 CUDA graph 缓存仅比对裸指针与尺寸；复用同一对象期间不得通过 `set_`/`resize_` 修改底层 storage 或 metadata。`sched_policy` 仅按 `(indptr, G, m, n, k)` 缓存 `engine` 与 `pp`（`fp8gg_umma.cu:1542–1548`），原位修改 `m_indptr` 内容会使 selector 选择陈旧，但设备端每次重读 `m_indptr` 构建 `sched_table`，不会复用旧计算结果。
- **单设备假设与运行时拒绝：** 编译架构查询、流获取、`fp8umma_smem_max` 及全局缓存均固定面向 `device 0`，不支持跨设备切换。合法形状仍可能在运行期因动态 shared 容量不足（返回 `1003`，`root/wide/engine 2` 分别需 `232448 / 230400 / 214016 B`）或主 tensor map 编码失败（返回 `1002`）被拒绝。

## 完整设计

| 设计 | 具体机制与保留理由 | 关系 |
| --- | --- | --- |
| [root](root.yaml) | Pair-batched 引擎 `umma_gemm<0>`（`engine == 0`，256 线程，`232448 B` 动态 shared）。工作项为 `(组内 128 行块, 128 列块)`，持久网格 `min((ceil_div(m,128)+group_count)*(n/128), sm_count)` 经设备侧 `sched_table` 前缀调度，无需全局 workspace。`PRODUCER`（warp0）按 K 块对搬入 3 深 `a_ring/b_ring`（每对 64 KiB）；`ISSUER`（warp1）每对发 8 个 128×128 k32 UMMA 写 `acc_pair` 的两个 128 列块并以单次 `tcgen05.commit` 发布；`EPILOGUE`（warp4–7，128 线程，`thread == row`）用 `sa_ring` 2 槽 cp.async 与 `sb_regs` 预取 scale，x16 旋转排空 TMEM 并在末批 32 列 FFMA（`promote_tail`）前提前 `signal:acc_free`，整块经 32 KiB `staging` 发 2 次 TMA store，残块谓词直写。固定 `dual=0, bstream=0, abox=1, sareg=0`。 | 阅读入口，保存唯一共享 Contract；与 `wide` 并列，范围重叠、无隐式优先级 |
| [wide](wide.yaml) | 宽块引擎 `umma_wide<4,32768,0>`（`engine == 1`，要求 `n % 256 == 0`，320 线程，`230400 B` 动态 shared）。工作项改为 `(组内 128 行块, 256 列块)`，持久网格 `min((ceil_div(m,128)+group_count)*(n/256), sm_count)`。同步粒度改为单个 128-K 块：`PRODUCER`（warp8）每轮向 4 深 `a_ring/b_ring` 搬入 16 KiB A 与 32 KiB B；`ISSUER`（warp9）每轮发 4 个 128×256 k32 UMMA 写 2×256 列 `acc_pair` 槽 `n%2` 并单次 `tcgen05.commit`；`EPILOGUE`（warp0–7，256 线程）按 `grp = warp >> 2` 分两列组，以 `sa_regs/sb_regs` 全寄存器两深预取按列半块提升，256 线程到达 `signal:acc_free` 后执行 `promote_tail`；整块写回按两半块顺序复用单个 32 KiB `staging`（每项 4 次 barrier、1–2 次读完成等待、2 组 store），残块谓词直写。固定 `bstream=0, abox=1`。 | 与 `root` 并列，共享 `root` 的 Contract；三区全部本地定义，无 `from` |

## 变体与组合

### wide 的单遍整块写回 engine 2（局部）

- 基线：[wide](wide.yaml) 的 `DESIGN.Tile_pipeline::MAIN.EPILOGUE`（`release:staging`、`store:staged_half`、`store:half_out`）、`Storage`（`a_ring`、`b_ring`、`staging`）及对应 `REALIZATION.Require`（`fp8gg_umma.cu:1316–1378`）。
- 差异与联动：实例化同一模板 `umma_wide<3,65536,1>`（`GSTAGES=3, GSTAGB=65536, GSTORE=1`）。`a_ring` 与 `b_ring` 环深 `4 → 3`（容量 `65536 → 49152 B`、`131072 → 98304 B`），`staging` 扩为 `65536 B`（`shape: [{row:128}, {col:256}]`，单整块视图），动态 shared 改为 `214016 B`（`3*49152 + 65536 + 1024`）。整块写回取消 `for(h=0:2)` 分半复用：每项仅 1 次前项 `cp.async.bulk.wait_group.read`、2 次 256 线程具名 barrier，两列组并发写入 4 个 16 KiB swizzle 象限，单次 `fence.proxy.async` 后一次提交 4 个 TMA box（`nt2*256 + {0,64,128,192}`）。数学、工作划分、2×256 列 `acc_pair` 与残块直写不变。
- 条件与依据：已命中 `wide` 且 `kb <= 20 && 2 * mtiles >= 3 * ne`；提交失败置 `g_w1g_off` 永久回退 `engine 1`（`fp8gg_umma.cu:1631–1632, 1707–1762`）。

### kb1 双发射 dual（局部）

- 基线：[root](root.yaml) 的 `DESIGN.Tile_pipeline::MAIN.ISSUER`（`Member: warp[1:2]`、`matmul:a_pair@b_pair`、`signal:pair_done`）、`stage_free(n) ← pair_done(n-3)` 及单次 commit/到达 `REALIZATION.Require`。
- 差异与联动：启用原本闲置的 warp2，`ISSUER` 扩为 `warp[1:3]`；warp1 发 `kb0` 的 4 个 k32 步写 `blk(n,0)`，warp2 发 `kb1` 的 4 个步写 `blk(n,1)`，两者各自等待 `stage_full` 与 `acc_free` 并各自 `tcgen05.commit` 到同一 stage 的 `pair_done`，该 barrier 到达计数由 `1` 变为 `2`。数学、工作划分、`PRODUCER`、`EPILOGUE` 与写回不变。
- 条件与依据：`engine == 0` 下 host 按 `wub * ceil(kb_count / 2) >= 48 * grid` 置 `Params.dual = 1`（`fp8gg_umma.cu:1829–1830`）。

### B 载入的 L2 evict_first 提示 bstream（参数）

- 基线：[root](root.yaml) `MAIN.PRODUCER` 的 `load:b_pair` 与 [wide](wide.yaml) `MAIN.PRODUCER` 的 `load:b_block`。
- 差异与联动：B 的 `cp.async.bulk.tensor.2d` 附加 `createpolicy.fractional.L2::evict_first.b64 1.0` 缓存提示；A 载入、预告字节数、barrier 与计算结果不变。
- 条件与依据：`sched_policy` 在 `mtiles == ne`（每个非空组恰占 1 个 m-tile、B 块仅读一次）时置 `pp = 1`，同时传入 `Params.bstream` 与 `ParamsW.bstream`（`fp8gg_umma.cu:1646`）。

### 固定 128 行 A box 降级 abox=0（参数）

- 基线：[root](root.yaml) `MAIN.PRODUCER` 的 `load:a_pair`、[wide](wide.yaml) `MAIN.PRODUCER` 的 `load:a_block` 及其自适应 box 高度（16/32/64/128 行匹配 `tA16/tA32/tA64/tA`）`REALIZATION.Require`。
- 差异与联动：四个 tensor map 指针全部指向 128 行 `tA`；`root` 每对恒搬 128 行并预告 `2*16384 + 2*16384 B`，`wide` 每轮恒预告 `16384 + 32768 B`。残块多载入的邻组行或 TMA 越界零填仍被 `EPILOGUE` 的 `valid` 谓词与写回边界排除，有效输出不变。
- 条件与依据：`tA64/tA32/tA16` 任一缩小 box 编码失败时置 `abox = 0` 降级继续（`fp8gg_umma.cu:1685–1686`）。

### 诊断构建分支（局部）

- 基线：[root](root.yaml) 的 `DESIGN.Tile_pipeline::MAIN`（`EPILOGUE`、`sa_ring`、`row_acc` 与 CTA 线程规模）。
- 差异与联动：`umma_gemm<1>`（宏 `FP8GG_FORCE_SAREG`）将 `a_scale` 改为每线程寄存器两深同步 `__ldg` 预取，`sa_ring` 的 2 KiB smem 保留未用而停用其 cp.async 提交与等待；8-warp 分栏 epilogue（宏 `FP8GG_V0_8EPI`）将 CTA 扩至 384 线程（`EPILOGUE` 为 `warp[4:12]` 分两列组、每线程 `row_acc` 缩为 64 列并改 x32 两深排空，`signal:acc_free` 到达计数 `128 → 256`）。
- 条件与依据：生产构建将 `Params.sareg` 恒置 `0` 且不启用上述诊断宏，两者默认均不进入（`fp8gg_umma.cu:1582, 1635–1639`）。
