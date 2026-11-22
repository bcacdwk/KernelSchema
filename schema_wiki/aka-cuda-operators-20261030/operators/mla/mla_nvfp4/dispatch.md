# mla_nvfp4

## 范围

实现范围。本次收录固定实现 `candidate/` 的十个文件（来源身份见 root.yaml 的 `META_DATA.Source`）：主机侧 `solution.py`（绑定入口 `Model.forward`）、`runtime.py`（校验、workspace 雕刻、参数绑定、启动）、`plan.py`（主机 work plan）；设备侧 `kernel.cu` 中唯一的解码主 kernel `kernel_cake_nvfp4_mla_decode_6260e249381d96576baa`、`combine.cu` 中的归并 kernel `kernel_cake_nvfp4_mla_decode_ff76ef6d8e6453ca5e3c`，以及 `launch.cu` 与 `combine_launch.cu`（TMA descriptor 编码、launch 配置与主机侧参数检查）、`cuda_primitives.cuh`（smem/TMEM 布局常量）、`ffi.cuh`、`host_checks.cuh`。

调用链。`Model.forward()` 无参调用构造期 `prepare_nvfp4_batch_decode_with_kv_cache_mla` 返回的 `NVFP4MLADecodeRunner.launch()`：同一 stream 上先发主 kernel，仅当 `plan.max_splits > 1` 再发 combine kernel，然后返回 out 或 (out, lse)。规划、`seq_lens_cpu` 缺席时的一次 `seq_lens.cpu()`、workspace 各区初始化拷贝以及 out/lse 的新分配全部发生在构造期，forward 内不分配、不主机同步。

主机接受与拒绝条件的去处。shape、整除、`seq_lens` 范围、page table 有效槽、连续性与同设备条件由 Contract 的 `Domain` 步承担；`backend` 必须是 `cake` 与 compute capability 必须是 `(10,3)` 是实现接受条件，前者写在 Contract 的 `Input.backend`（fixed impl cake、default cake），后者与 workspace 容量足够写在 root 的 `META_DATA.Dispatch.when`（`SUPPORTED_COMPUTE_CAPABILITIES` 只含 `(10,3)`，而报错文案写 10.0 或 10.3，`metadata.json` 宣称 SM100/103）；workspace 的 256 B 对齐雕刻与容量不足即拒绝写在 root 的 `REALIZATION.Require`。plan 侧的一致性校验不落 Contract 有效域：`max_splits <= 64` 由 `Tile_pipeline::COMBINE.MAIN` 的每 lane 两个 split 槽 × 32 lane 承担；`check_pairs` 的“每 unit 偶数个 item 且为同一 piece 的 v_half 对”由 `Work_schedule::MAIN.Decomposition.Strategy` 与 `Mapping.Member` 承担；`max_splits > 1` 时 `num_heads` 整除 128 与每 token 恰被一个 split 数覆盖，前者由 `Domain` 的 `num_heads` 取值集合（16 / 32 / 64 / 128 都整除 `row_tile`）保证，后者由 `Call_semantics` 的“划分恰好覆盖每个查询行一次”与 `COMBINE.Schedule.Strategy` 读的每 token split 数承担；`max_splits * rows_total * head_dim` 的 int32 元素偏移范围只在 balanced 域可达，记在下面的 balanced 条目。

只有一套可提取实现。`runtime.MODULES` 只登记 `cake_nvfp4_mla_decode_sm_103a` 一个模块（main 与 reduce 两个 stage），`kernel.cu` 与 `combine.cu` 各只有一个 `__global__`；没有第二条 kernel 路径，没有录图或结果缓存（`load_cake_nvfp4_mla_decode_module` 上的 `@cache` 只是编译产物缓存）。`problem/solution.py` 是生产对照，调用未随源提供的 FlashInfer 同名 prepared API（依赖修订 `d849eca46875d3561fa48f6093936304580a1698`），其实现不在固定源码内，不能作为设计候选。`cuda_primitives.cuh` 末尾的十个宏（`SKIP_XFORM / LAZY_RESCALE / IDENTITY_PAGES / SKIP_EXP / SKIP_KPUB / SKIP_PSTORE / SKIP_QK / SKIP_PV / SKIP_SMATH / TRACE`）与三个设备函数（`ex2_emulation_f32x2_value`、`ex2_emulation_f32x2`、`softmax_frag_exp2_cast`）全目录只有定义、无任何引用，不构成本源的编译期变体轴；其中 `softmax_frag_exp2_cast` 把结果打包成 BF16 而非 E4M3，不能读作 P 的量化路径。

已识别机制与去处。以下六项机制依据实际调用、模板实参与函数体识别（源码注释只作导航），现已全部落到 root.yaml 的正式定义，本目录只保留定位而不复述其内容：

1. 工作划分：piece = (request, row group, KV split)，一个 piece 写成相邻的 v_half 0/1 两个 item，由一个 2-CTA cluster 承担，item 携带 `FLAG_SEED_SINK` 与 `FLAG_DIRECT_OUT` → `Work_schedule::MAIN` 的 `Decomposition` / `Mapping` / `Schedule`，边界由 Contract 的 `Split_partition` 与 `Uniform_split_policy` 给出。
2. cluster 内操作数分割：rank 决定 64 行半区、每 page 中自己 32 个 key 的完整载荷与对端 32 个 key 的半载荷、以及 V 的通道半区；输出通道分工来自 CTA 内 softmax 象限（`n_half * 128` 与 `n_half * 128 + 256`），`cta_rank` 从不进入输出的通道地址 → `Work_schedule::MAIN.Mapping.Strategy / Member`、`Tile_pipeline::MAIN.Storage`（含 `tmem` 的 `s / o / q_sf / k_sf / q_op` 列区）、`LOAD.kv_raw` 与 `SOFTMAX` 的 `o_block` / `out_block` / `lse`。
3. 主计算：block-scaled NVFP4 的 QK、V 的就地 E4M3 转换、落后一拍的 PV 与共享 TMEM 累加器 → `Math_reshape::Block_scale_score_product`、`Tile_pipeline::MAIN` 的 `TRANSFORM` / `CORRECTION` / `MMA`，以及 `REALIZATION.Require` 的 MMA 发射与启动配置两条。
4. softmax 与在线归一化：延迟最大值、两 key 半区经 `xchg_max` / `xchg_sum` 交换、E4M3 概率、alpha 交 correction 就地回写 → `Math_reshape::Half_split_online_softmax` 与 `Score_scale_in_exponent`、`Tile_pipeline::MAIN` 的 `SOFTMAX` / `CORRECTION` 及其 `Waiting` 边。
5. 输出形成：由每个 piece 的 `FLAG_DIRECT_OUT` 决定直写 O/LSE 或写 BF16 partial 与 log2 域 lse，各分 bulk TMA 与逐行 guarded 两条写路径 → `Math_reshape::Reciprocal_split_output`、`Work_schedule::MAIN.Target`、`SOFTMAX` Pipe 中 `all_rows_valid` 的两个分支与 `epi_stage` 归还。
6. 归并与完成关系：同 stream 的第二次提交，每 warp 一行、每 lane 16 通道与 2 个 split 槽，split 数为 1 的行不写 → `Math_reshape::Log2_split_merge`、`Work_schedule::COMBINE` 与 `Flow.Launch / Handoff`、`Tile_pipeline::COMBINE.MAIN`。

各 mbarrier 的生产者集合与到达关系落在每个 role 的 `Waiting.Prepared / Released` 边与 `signal` / `release` 动词；18 组共 30 个 barrier（`raw_full` 与 `sf_empty` 各是其中 init 计数为 1 的一组）的具体到达计数属实现细节，不进 YAML。空工作与尾块以 `SOFTMAX` Pipe 中 `all_rows_valid` 的两个分支（`out_block` 的 bulk tensor 写与 `out_row` 的逐行 guarded 写）和 `LOAD.kv_raw` 的“奇数 page 重复一页”表达；无有效行的 member 仍跑完整条流水线，这一点写在 `Tile_pipeline.Thesis`。

承担范围与未覆盖面。root 承担 `META_DATA.Dispatch.when` 写实的 uniform 域：`max_b seq_lens[b] < 8192`（主机 plan 因此选 `Uniform_split_policy`，每个 cluster 恰领一个 piece）、cc `(10,3)`、workspace 足够。balanced 域（`max_b seq_lens[b] >= 8192`，`shape_cases.json` 十个固定用例中的八个）没有任何完整设计的 `when` 覆盖，只由下面的局部记录表达差额；目录不规定隐式优先级，root 不自动承担未记录的情况。

已知缺口与来源限制。`Hardware_specification` 的 `Device / Architecture / Stack / Wiki` 仍是占位，上述 cc 与 SM100/103 的声明差异随占位留给有权阶段。三处近似（`exp2`、`log2`、归一化倒数）没有实测误差界，按规范在 `Math_reshape` 的三条 `Arithmetic.Approx`（`Score_scale_in_exponent`、`Reciprocal_split_output`、`Log2_split_merge`）写 `bound unknown: not recorded`；任务材料明言近似指令不定义参考真值。验收状态 `passed_sampled`（`performance/ACCEPTANCE.json`）与 `PRECISION.md` 的 144 条检查 0 超差只覆盖固定样本，不作全域保证。

## 完整设计

| 设计 | 具体机制与保留理由 | 关系 |
| --- | --- | --- |
| [root](root.yaml) | uniform 划分实例下的 cluster 解码方案：一个 2-CTA cluster 承担一个 piece，主 kernel 以五个固定 warp 角色（softmax / correction / transform / load / mma）与 mbarrier 流水线完成 block-scaled NVFP4 QK、E4M3 V 的就地转换、E4M3 P 的 PV 与延迟最大值在线归一化；单 split 的 piece 直写 O/LSE，多 split 的 piece 写 BF16 partial 与 log2 域 lse，再由同 stream 的 COMBINE 提交归并。这是固定实现唯一的端到端方案，保留为默认阅读入口。 | 阅读入口，保存唯一 Contract |

完整交付集合含 root 共一份，也没有已研究但未完整展开的独立候选，理由：

- COMBINE 不是独立设计：它是同一方案里由 `plan.max_splits > 1` 触发的第二次提交，其输入（`partial_store` / `lse2_split` / 每 token split 数）的生产者、完成关系与输出归属必须与主 kernel 在同一份 `Work_schedule` 里贯通。
- direct-out 与 partial+combine 两条输出路径不是两套设计：由每个 piece 的 `num_splits` 经 `FLAG_DIRECT_OUT` 决定，属数据引起的执行分支；`plan.token_splits` 与 COMBINE 读的 `row_splits` 保证同一行只由一条路径写最终 O/LSE。
- uniform 与 balanced 只是主机划分不同：kernel、combine、work table 结构、role 与存储完全一致（同一份设备代码用 `unit_first` 同时实现两者），balanced 记为下面的局部变体。
- 生产对照 `problem/solution.py` 的 FlashInfer 库实现不在固定源码内，属来源限制而非未展开的候选。

## 变体与组合

### balanced 划分与每 worker 多 piece（局部）

- 基线：[root](root.yaml) 的 `DESIGN.Work_schedule::MAIN`（`Decomposition.Strategy / Partition.piece`、`Mapping.Worker_count`、`Schedule.Worker_assignment.Preassigned`）与 `DESIGN.Tile_pipeline::MAIN`（各 role 的 Pipe 只在一个 piece 内迭代 `tile_count`，`Preassigned.count: 1` 使跨 piece 的 Q 与 O 累加器归还边没有消费者）；划分式本身已由 `CONTRACT.Algorithm_requirement` 的 `Balanced_split_policy` 定义。`Math_reshape` 的全部改写、`Tile_pipeline` 的 role / 存储 / 事件、`Work_schedule::COMBINE` 与两条输出路径沿用不变。
- 差异与联动：`plan._balanced_plan` 取代 `_uniform_plan`——把每个 (request, row group) 的 `tiles(b)` 按 request-major、row-group-minor 拼成长度 `total` 的 tile 序列，取 `units = max(1, min(num_sms // 2, total // 8))` 并在 `ceil(max_b tiles(b) / (total / units)) + 1 > 64` 时折半，unit u 领 `[u * total // units, (u+1) * total // units)`，切口落在序列内部就把该 (request, row group) 切成多个 split；这些量与 `Balanced_split_policy` 的 `total / units / cuts / splits` 逐项对应。联动：`Partition.piece.size` 由切口数决定，`Uniform_split_policy` 的 `width` 与 `tiles_per_split` 不再存在（plan 记 `None`）；`Mapping.Worker_count` 由 piece 数改成 `units`（≤ SM 数），每 worker 仍是两个 CTA，grid 因而由 item 数缩到 `2 × units`，同一份设备代码由此真正持久；`Preassigned.count` 由 1 变成该 unit 的 piece 数，五个 role 各自的外层 worktile 循环因此迭代多次，循环外保留 stage/phase 计数，并补上跨 piece 的 Q 与 O 累加器归还边（`q_empty` 与 `o_empty`，root 用 `count: 1` 排除，只留 `SOFTMAX.release:o_acc`），每个 piece 仍靠首次 PV 的非累加谓词重置 O 累加器；`max_splits` 与每 token split 数随切口变化，BF16 partial 的舍入点因此与 `Balanced_split_policy` 的边界一致；COMBINE 及其 `Worker_count` 公式不变。
- 条件与依据：默认 `schedule="auto"` 在 `max_b seq_lens[b] >= 8192`（`BALANCED_MIN_KV`）时选它，`shape_cases.json` 的十个固定用例中八个落在此域（例如用例 0 的 8192、用例 4 的 65537）。缺口一：balanced 没有 uniform 的 0.35×KV partial 流量闸，`partial_o` 按 `max_splits * rows_total * head_dim * 2 B` 增长，声明域角落（batch 32 / q_len 16 / heads 128 / 各 seq 32768，得 `max_splits = 2`）仅 `partial_o` 就达 128 MiB，加上 partial lse、work table 等区即超过生成器固定提供的 128 MiB workspace，主机直接拒绝；按 `Uniform_split_policy` 的流量闸与 Contract 的 `sum_b seq_lens[b] <= 1048576`，uniform 域的同一上界约 50 MiB，容量拒绝在 root 的 `when` 域内不可达。缺口二：`max_splits` 可达 64 时主机另要求 `max_splits * rows_total * head_dim` 落在 int32 元素偏移内，超出即拒绝；root 的 `when` 域内 `width >= 8` 且 `tiles(b) <= 64` 使 `max_splits <= 8`，该条件同样只在 balanced 域可达。两条拒绝都属 `REALIZATION.Require` 的 workspace 雕刻条与本记录，不回填 Contract 有效域。

### schedule 与 tiles_per_split 覆写（参数）

- 基线：[root](root.yaml) 的 `CONTRACT.Algorithm_requirement` 中 `Uniform_split_policy` 折半规则算出的 `width`（源码 `plan.choose_tiles_per_split`），即 `DESIGN.Work_schedule::MAIN.Decomposition.Partition.piece.size` 与 `Mapping.Worker_count` 所依赖的那个宽度。
- 差异与联动：`prepare_nvfp4_batch_decode_with_kv_cache_mla` 另接受 `schedule ∈ {auto, uniform, balanced}` 与固定的 `tiles_per_split`。`tiles_per_split` 的单位是 `kv_tile`（128 token），不是 `plan.build_work_plan` 文档串所写的 page；它直接替换 `width`，只影响 item 数、`max_splits`、`unit_first`、`Worker_count` 与 BF16 partial 的舍入点，role、流水线与 COMBINE 机制不变。`schedule` 覆写则整体切到上面的 balanced 局部记录，此时 `tiles_per_split` 被忽略（balanced 划分不接受该参数）。
- 条件与依据：绑定入口 `Model.__init__` 不传这两个参数，从 Contract 的调用面不可达，只对直接调用 `prepare` 的测试与基准有效；覆写后的划分不再等于 `reference._split_ranges` 由公开 shape/device 策略重建的边界，而该边界决定标准答案（每个 split 边界都存在一份已存 BF16 partial），因此只作参数记录，不进入任何完整设计的 `when`。
