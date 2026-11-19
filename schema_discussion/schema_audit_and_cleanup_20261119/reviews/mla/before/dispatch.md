# mla_expanded_fp8

## 范围

**实现范围。** 固定入口 `Model.forward` 把冻结 ABI 中实际有效的参数转给 `ext.mla_launch(..., _PV_MODE)`，进入 C++ `launch_impl`，最终只启动一个融合 kernel 模板 `mla_fp8_kernel<MODE, KVST>`；没有第二个 attention kernel，也没有 split-KV 的跨 kernel 归并。冻结入口实际可进入的部分由三个 launch 期决定划定：

- `MODE` 恒为 0：`_PV_MODE = 0` 是模块常量，`forward` 没有参数或环境变量能改它，只有 MODE0（E4M3 概率、V 直接取自 TMA 的 fp8 smem）可进入。扩展虽然也把 `mla_launch`/`mla_v16_prepare` 暴露给 Python，但本任务的入口只有 `Model.forward`。
- `KVST` 由 host 的 scatter 门控选择：活跃 K/V 字节 `min(kv_heads*batch, NSM) * max(kv_len) * 320`（NSM 为运行时查询的设备 SM 数，即 root.yaml 的 `sm_count`；320 = K 的 192 + V 的 128 字节）超过 3/4 L2（root.yaml 的 `l2_bytes`）时用 `<0,3>`，否则 `<0,4>`。
- 网格形态由 `grid_items = (items*2 <= NSM*3) ? items : NSM` 决定，kernel 内据此得 `queued = gridDim.x < items`：持久领取，或一个 CTA 一个 item。两个网格形态共用同一个 kernel 实例 `<0,4>`，区别只在 host 给的 grid 与 kernel 内由它推出的 `queued` 标志，不是两个二进制。

工作单元 item = (请求 b, 128 行 q tile, head h)，`items = tiles_total * heads`；kv 方向按 128 key 一个 tile 循环，`kv_tiles = max(1, ceil(vend/128))`，causal 时 `vend = min(kv_len, q_end + (kv_len - q_len))`（`q_end` 是该 tile 在请求内的行上界），即右下对齐。声明域内 `heads == kv_heads`，故 `kv_head(h)` 在域上是恒等映射，源码保留一般 `heads % kv_heads == 0` 形式；一个 item 只服务一个 head，K/V 的跨 head 复用只来自 L2 与表序。

**几何全部在 host 解析。** 每次 forward 重新执行：侧流 DtoH 拷 `seq_lens`/`cum_seq_lens_q`/`cum_seq_lens_kv` 并 `cudaStreamSynchronize` → host 建稠密 `ItemGeo` 表 → 可选 LPT 稳定计数排序 → HtoD 上传到 growth-only 的 `cudaMalloc` 设备表并记 event，主流 `cudaStreamWaitEvent` 后才 launch。侧流与调用者当前流之间没有任何 event join，三个元数据张量在拷贝发出时必须已经就绪，这一就绪只由调用者排序保证，root.yaml 的 `REALIZATION.Require` 把它记为来源限制而不是设计提供的等待。kernel 只读 `P.geo_table`；`Params` 的 `cu_q`/`cu_kv`/`seq_lens`/`batch`/`kv_heads`/`uniform_qlen` 在设备端从未被读取。因此每次调用含一次 device→host 元数据同步，调用序列不可整图捕获，网格与 item 数都是 host 决定的。张量 map 走 `encode_map` 缓存，键 `(base, seq, heads, dim, box0, swizzle)` 恰好覆盖 map 的全部输入，复用不引入陈旧地址；设备表与队列字只复用缓冲区，内容每次由当前输入重建。

**未收录的范围。**

- **MODE1（fp16 概率）是源码中完整存在但从冻结入口不可达的独立方案**，已研究、未交付完整 schema。独特机制：另起 `v16_prep_kernel` 先把 V 的 E4M3 精确转成 FP16 写进 caller 的 `workspace_buffer`（不满足条件时用私有 fallback 张量），PV 改 `kind::f16` 的 8×k16 步、V 阶段 32KB 且由两次 TMA 子 tile 拼成、`KVST` 被 `static_assert` 锁死为 3，概率不量化（`off` 不含 2^8，LSE 不减 ln 256）。取舍理由：只能由 `pv_mode == 1` 进入而冻结入口恒传 0；它与 root 的差额要重新贯通数值表示、V 的生产者（多一个 kernel 与一块跨 kernel 存活的 scratch）与流水线容量，不属当前任务可达范围的交付价值。源码把它称作保留 fallback。
- `forward` 里 `q.ndim == 4` 的 uniform 重排与 `out is None` 时自行分配是非冻结遗留路径：声明域只产生 packed 3-D 输入，冻结调用恒提供 caller-owned `out`/`lse`，二者都不进 Contract，也不构成设计。
- `window_left >= 0` 的左窗分支在设备端存在，但声明域固定 `window_left = -1`；Contract 保留一般公式，实测覆盖只在 -1，root 的掩码按一般式表达，声明域内下界恒不生效。
- 死代码不属任何设计的动作：`reg_dec_conv`（早期 in-kernel V 转换 warp 的寄存器预算，现已无转换 warp）、`store_o_row`（被 epilogue 内联的串行 drain 取代）、`M.geo_valid`（只写不读，padding CTA 早退已删）。

**影响理解或使用的边界。**

- 验收判据仍只有来源标签（P1 已登记 source_gap）：组合公式、相对误差基准、mismatch/rel-L2 归一化与 default→lse 路由都缺。因此 root 的数值取舍（P 量化到 E4M3、2^8 缩放、f16x2 tile sum、MAXFREEZE 0.75、tiny-row fp32 精确路径）机制有据，但数值保证只有源码自述的实测余量，没有可独立复核的外部判据。
- 硬件区仍是 P1 占位。NSM、L2 容量、opt-in 共享内存在源码中是运行时属性查询；注释里的 148 SM、132.6MB L2、99.5MB 阈值未经独立核实，本目录引用它们只为说明门控量级与推算固定用例走哪个实例。
- 源码注释有多处已过期，本目录一律按实际代码取值：模块 docstring 的 “Device-side ragged tile mapping … no host sync” 与上面的 host 表 + 每 forward DtoH 同步矛盾；“probabilities scaled by 448” 与常量名 `kLog2_448` 的实际值是 `8.0f = log2(256)`，448 只是 E4M3 上限、用作 MAXFREEZE 的余量依据。
- 固定 10 用例是样本而非完整域；按声明域的 `q_len == kv_len` 与 `q_len` 可低至 1，tiny-row 精确路径在域内可达（单 kv tile 且 `q_len <= 32` 的请求），但 10 个固定用例的最短请求是 257，都不触发它；10 个用例也全部 `sink = false`（`attention_sinks` 为 None），而声明域允许 `sink = true`，故 root 的 sink 重归一与 sink LSE 形式只有源码依据；degenerate 门与 scatter 门在固定用例上同样都不触发。证书声明 domain 套件 `full_pass`，而 domain 用例清单指向材料之外的路径，故无法核对哪些域点覆盖了这些分支；源码自述的实测余量只能当来源主张，root.yaml 与本目录都不把它写成已验证事实（相应的近似误差界在 root.yaml 中记为 unknown）。

## 完整设计

当前完整交付集合就是 [root](root.yaml) 一份，没有计划中的其他完整 YAML：其余差异都能由下面的参数、局部与组合记录表达，MODE1 已研究但不可达且未完整展开，均未借记录名义冒充完整交付。

| 设计 | 具体机制与保留理由 | 关系 |
| --- | --- | --- |
| [root](root.yaml) | 冻结入口唯一可进入的数值模式（E4M3 权重，源码的 `mla_fp8_kernel<0, 4>`）下的持久领取实例：`Worker_count = sm_count`、每 worker 一个 CTA、384 线程 12 warp。host 把几何解析成按 `kv_tile_count` 降序稳定排序的 item 表，worker 先做 `Preassigned` 的单个 item，再由 `Refill.atomic_queue` 从入口为 0 的设备计数器补领。QK 以 6 个 k32 步覆盖 192 通道（4 步 [0,128) 的 SW128 段 + 2 步 [128,192) 的 SW64 段，每操作数两张 tensor map），无 repack、不补零到 256 通道；`score_prob` 是 3 阶段 TMEM 环（与 `o_slot` 的 128 列合计占满 512 列），SOFTMAX 把按 2^8 缩放并转 E4M3 的 `p` 写回同一阶段的 [32,64) 列充当 PV 的 A 操作数；PV 4 个 k32 步直接吃 TMA 送来的 fp8 `v_tile`，`o_run` 以 fp32 累加在 TMEM，CORRECTION 按 `scale_relay` 中继的 `alpha` 就地重标定；epilogue 归一、可选 sink 重归一、bf16 直写 caller 的 `out` 并写 log2 空间 LSE，`tiny_unit` 的行改由 `tiny_scratch` 的精确 fp32 累加产出。保留理由：按源码自述的 148 SM 取值，固定 10 用例中 9 个走此实例（只有 192 item 的那个走静态网格），是这批用例里唯一同时含队列补领、跨 unit 寿命与 Q 预取的实例，最适合作比较基线；声明域内 `<0,3>` 的持久领取组合同样含这三项，只按下面记录改环深与表序，故该唯一性限于固定用例，其余实例相对它都是删减或参数改动。 | 阅读入口，保存唯一 Contract。三个 Design 区域与 `META_DATA.Dispatch.when` 均已本地写实，精确参数、`when` 与修订号以 YAML 为准。承担范围即其 `when`：E4M3 权重路径、`item_count*2 > sm_count*3` 的持久领取网格（入口队列字为 0）、活跃 K/V 未触发 scatter 门故 K/V 环保持 4 槽、无 `q_len(b) < 128` 的请求故表序为 LPT。下面四条记录分别放松其中一个条件，各自定位 root.yaml 中已写实的对象。 |

## 变体与组合

以下每条记录都直接定位 [root](root.yaml) 中已写实的对象（数学符号、step、role、storage 与 `REALIZATION` 条目），源码标识符只在需要对应固定来源时附注。记录不修改基线的参数或 `when`，条目排列也不构成应用顺序。

### K/V smem 环深 3（参数）

- 基线：[root](root.yaml) 的 `DESIGN.Tile_pipeline::MAIN.Storage.k_ring` 与 `v_ring`（`depth: 4`、`total_size` 98304/65536、`view` 的 `slot_index: n%4`）、`Tile_pipeline.Thesis` 中“K/V 环为 4 且需要自己的周期匹配释放”一句、`ISSUER` 的 `matmul:q_tile@k_tile`（“k_ring slot n%4”）、`await:kv_tile`（“ring slot n%4”）与 `release:kv_slot(n)`（PV 之后的第二次 commit）、`LOADER` 的 `load:kv_tile`（“Fill ring slot n%4”）、`await:kv_slot(n)`（“four key tiles after the slot was filled”，Waiting 的 Released 边前驱是 `release:kv_slot(n-4)`）与 Pipe 守卫 `if(n >= 4)`、`REALIZATION.Require` 中 209920 字节 opt-in 动态共享内存那一条，以及 `REALIZATION.Forbid` 的第一条。
- 差异与联动：环深字面量在基线中的全部落点都要改。`k_ring`/`v_ring` 的 `depth` 4→3、`total_size` 98304→73728 与 65536→49152、两个 `view` 的槽映射改为 `{k_tile(n): {slot_index: n%3}}` 与 `{v_tile(n): {slot_index: n%3}}`（`slot_index` 只写在 view 中，不由 `depth` 自动得出）；上述四处 describe 的 `n%4` 改为 `n%3`，`LOADER.await:kv_slot` 的“four key tiles after the slot was filled”改为三个 key tile 之后；`LOADER` 的 Pipe 守卫 `if(n >= 4)` 改为 `if(n >= 3)`，即首次必须等待的搬运就是 n=3，与源码的 `if (T >= KVST)` 一致——保留 `n >= 4` 会让 n=3 在 PV(0) 仍可能读取 `v_tile(0)` 时重写 slot 0，正是下面条件与依据所说的那类复用危险；`LOADER` 的 Waiting Released 边前驱由 `ISSUER.release:kv_slot(n-4)` 改为 `ISSUER.matmul:p(n-3)@v_tile(n-3)`，等待与 commit 的周期同为 3，相位差结构性 ≤1；`Tile_pipeline.Thesis` 改为 K/V 环 3 槽、与 S 环同期、不需要专用的周期匹配释放。`ISSUER.release:kv_slot` 整个动作连同 `Forbid` 第一条一起取消（该条禁止的正是用 PV 完成通知代替专用释放，环深与 S 环同期后这一禁令不再适用）；`Require` 第二条的账目改为 166224/168960 字节（未登记的通知对象与对齐仍是 2736 字节），`Meta` 换成无 `kv_free` 字段、偏移与认证版逐字节一致的布局；`QUEUE.compute:barrier_init` 初始化的通知对象少 5 个（4 个槽释放对象加第 4 号 `kv_loaded`），其“本 step 除查询搬运外的全部通知对象”表述与深度无关，不需改写。每 unit 的计算、输出形成、角色组成与 per-tile 协作不变，失去一格填充提前量。
- 条件与依据：host 的 scatter 门命中时选它，谓词为 `min(kv_heads*batch_count, sm_count) * max(kv_len) * 320 * 4 > l2_bytes * 3`（活跃 K/V 超过 3/4 L2 即视为 DRAM 排队受限）。声明域内可达（例如 `batch_count = 3`、`heads = kv_heads = 128`、每请求 5461 token 时活跃量约 258MB）；固定 10 用例的最大 scatter 为 97.0MB，低于按源码注释设备取值推算的约 99.5MB 阈值，全部走环深 4。源码记录的实测依据：深 scatter 配置上环深 4 因每 CTA 多 32KB 在飞字节退化 1.32%/1.77%，L2 常驻配置上环深 4 快 0.85~2.0%。环周期匹配是硬约束——非同期环若沿用周期 3 的 PV 完成通知作释放门，loader 的槽访问节奏 4 与 commit 节奏 3 会累积无界相位偏移并导致 parity 别名死锁，源码附有隔离复现记录。

### item 表序改回 head-major 重块优先（参数）

- 基线：[root](root.yaml) 的 `DESIGN.Work_schedule::MAIN.Schedule.Strategy`（表序即派发与补领序，按 `kv_tile_count` 降序的稳定排序）、`Worktile_order` 与 `Worker_assignment.Refill.atomic_queue`；受影响的安全性论证写在 `SOFTMAX.compute:row_final` 的 describe（tiny 重读“safe as later units hold one tile too”）。
- 差异与联动：host 跳过稳定排序，保留建表时的 head-major → 请求 → 重 q tile 优先顺序。`item` 分区、每 unit 的数学与 `Target` 的输出地址都不变（`out`/`lse` 按绝对地址写，数值与分配顺序无关），`Worktile_order` 仍是 item 升序、`Refill` 的领取协议与 pop 次数也不变；改变的只有派发顺序与每 head 的 K/V L2 复用局部性。
- 条件与依据：两个门任一命中即跳过。degenerate 门——任一 `q_len(b) < bq`（= 128）：子 tile unit 在 `q_valid < 128` 行上仍跑完整 CTA 机制，真实成本远超 `kv_tile_count` 权重模型，稳定 LPT 会把它们堆成暴露的尾波（源码预合并实测 +16.3%）；声明域允许 `q_len` 低至 1，固定用例最小为 257，故此门在域内可达、在固定用例上不触发。scatter 门——与环深 3 同一谓词：全局 LPT 把每波 CTA 摊到多达 `min(kv_heads*batch_count, sm_count)` 条 K/V 流，活跃集超过 3/4 L2 时按 head 的 K/V 复用塌到 DRAM（源码实测 +11.2% / +5.9%）。门只用当前输入几何与运行时设备属性，不含用例身份。
- 影响使用的缺口：`SOFTMAX.compute:row_final` 的 tiny 精确路径在 `release:score_slot` 之后重读 `score_prob` 的 `dot` 区域，唯一会覆盖它的是全局第 `n+3` 个 key tile 的 `ISSUER.matmul:q_tile@k_tile`。root 的 LPT 表序使 `kv_tile_count` 沿每个 worker 的序列不升，tiny unit（`kv_tile_count == 1`）之后仍是单 tile unit，`n+3` 落到三个 unit 之后，而其几何发布结构上晚于本 unit 的 `row_final`，故安全。head-major 表序不提供这一单调性：tiny unit 之后可紧跟 `kv_tile_count >= 4` 的 unit，`n+3` 落在下一个 unit 内，而该 unit 的几何发布由本 unit 最后一个 key tile 开头的 `signal:pop_request` 触发（早于 `row_final`），覆盖只靠“重读是精确路径第一条指令”的时序余量。degenerate 门命中（存在 `q_len < 128` 的请求）正是可能出现 tiny unit 的情形，因此选用本记录时必须补一条等待：精确重读完成后再 `release:score_slot`，或先把这 32 列 `dot` 拷进寄存器再释放阶段。固定 10 用例最短请求 257，既不触发 tiny 路径也不触发 degenerate 门，无可核对的实测覆盖。

### scatter 门控下的环深 3 与 head-major 组合（组合）

- 基线：[root](root.yaml)（同时引用上面两条参数记录）。
- 差异与联动：同一个 scatter 谓词在一次 launch 里同时选择 `k_ring`/`v_ring` 的 3 槽与 head-major 表序；两条记录的接口不相交（一个改 `LOADER.await:kv_slot` 的前驱与 smem 预算，一个改 host 建表后的排序），组合没有引入新的整体组织。degenerate 门只改表序，环深仍为 4。head-major 记录中 tiny 重读对表序单调性的依赖在本组合中同样存在。
- 条件与依据：host 一处 `scatter_gated` 判定同时驱动 kernel 实例选择与跳过 LPT；组合的实测依据是同一组深 scatter 域配置（环深 4 退化 1.32%/1.77%，LPT 退化 5.9%/11.2%）。声明域内本组合总与持久领取网格同时出现：scatter 门需要长 `kv_len` 与多条 K/V 流，`item_count` 随之远超 `1.5*sm_count`（该推算使用源码注释的 148 SM 与 132.6MB L2 取值，设备实际值仍待硬件区登记）。反之 degenerate 门与静态网格可以同时命中（例如 `heads = 12`、`batch_count = 2`、`q_len = kv_len = [32, 1024]` 时 `item_count = 108`），此时本条不适用，改由 head-major 与静态网格两条记录一并适用，且 tiny 重读的表序依赖消失——每 worker 只有一个 unit，重读之后不再有 `n+3` 的 QK。

### 静态网格：一个 CTA 一个 item（局部）

- 基线：[root](root.yaml) 的 `DESIGN.Work_schedule::MAIN.Schedule.Worker_assignment`（`Preassigned{start: worker_id, count: 1}`、`Refill.atomic_queue`、`Exit: [sentinel]`）、`DESIGN.Tile_pipeline::MAIN.Role.QUEUE` 的 `await:pop_request` 与 `compute:item_geo`、`SOFTMAX.signal:pop_request`、SOFTMAX/CORRECTION/ISSUER 三个角色的 `await:queue_end`、`Storage.geo_slot` 的双槽发布、`LOADER.await:next_item`/`await:q_consumed`/`load:q_tile(t+1)`，以及 `REALIZATION.Require` 中队列字初态那一条。
- 差异与联动：host 在 `item_count*2 <= sm_count*3` 时以 `grid = item_count` 启动，kernel 内 `queued` 为假。`Refill` 改为 `none`（无 atomic claim、无哨兵），`Preassigned` 仍是每 worker 一个 item，`Exit` 由 `[sentinel]` 改为 `assigned_done`；`Role.QUEUE` 只保留 `compute:barrier_init`（源码在 warp10 lane0 无条件执行），`await:pop_request` 与 `compute:item_geo` 删除，QUEUE 不进入服务循环；`SOFTMAX.signal:pop_request` 与 SOFTMAX/CORRECTION/ISSUER 三个角色的 `await:queue_end` 删除（`LOADER` 未声明 `queue_end`：它的哨兵是在 `await:next_item` 那次发布等待中读到的，`item_valid(t+1)` 为假即哨兵，而该动作在本条下文一并删除），四者都在第二个 unit 入口处直接退出；`SOFTMAX.compute:bootstrap_geo` 与 `signal:publish(0)` 保留并承担唯一的几何发布，`geo_slot` 仍是两槽布局但只有 slot 0 被写，双缓冲的覆盖安全条件不再被使用；`LOADER` 只剩 prologue 的 `load:q_tile(0)`，`await:next_item`、`await:q_consumed` 与跨 unit 预取删除，Q 仍是单缓冲；`Require` 中以 `Refill` 为 Item 的那一条失去对象：队列字初态与自复位部分随之删除，上传 event join 与 launch 后无 host 操作的部分仍成立、需改挂到 `Schedule`；队列字完全不被触碰（本模式因此既不依赖也不重置自复位）。`Index` 的 `t` 恒为 0、`n` 退化为 unit 内的 `i`，槽与 parity 表达式不变但不再跨 unit；每 unit 内部的阶段环、TMEM 布局、barrier 协议、tiny 路径与 epilogue 全部不变；`release:tmem` 前的 CTA 全体退出 join 仍成立，只是退出条件换了。
- 条件与依据：源码 host 阈值 `item_count*2 <= sm_count*3` 与 kernel 内 `queued = gridDim.x < item_count`。依据是 item 数接近 SM 数时哨兵往返与收尾（约 1us）超过队列收益（约 0.18us，1.3 item/SM）。声明域内可达（`item_count <= 1.5*sm_count`，例如 `heads = 12`、单请求 `q_len = kv_len = 2048` 时 `item_count = 192`）；按源码自述的 148 SM，固定 10 用例中只有 192 item 的那一个走此模式。相对 root 只删减机制、未新增任何主要计算、输出形成或工作归属，故按局部记录而不另立完整设计。
