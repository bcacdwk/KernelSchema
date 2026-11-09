# 路径索引

| 节点 | 直接父 | workload | 相对父差异 | 选择依据 |
| --- | --- | --- | --- | --- |
| [root](root.yaml) | 无 | D128、tcgen05、非 split，融合 query 网格大于 SM 数。 | 基准：单 warpgroup 的 QK64 融合，完整 key-group 递推及 PV 分子留在 CTA 内，无跨 CTA partial/combine。 | HostPlan.use_tcg 且未实际 split；D128 大网格落入 qk64 分支，入口 launch_tcg<128,PASS_FUSED,1,true>。 |
| [fused_d256](fused_d256.yaml) | root | D256、tcgen05、非 split。 | 改用两 warpgroup 和 128-column QK，扩大 TMEM 输出；TMA 的 MIDPIPE 使用三槽 K、两槽 V 环并提前发射 QK，cp.async 保留自身循环。 | HostPlan.use_tcg 且未实际 split，D256 分支进入 launch_tcg<256,PASS_FUSED,2>；MIDPIPE 还受 TMA 限定。 |
| [fused_group_d128](fused_group_d128.yaml) | fused_key_d128 | D128、tcgen05、非 split，小融合网格且有效 key span 较长。 | 沿用两 warpgroup、双 score 槽和 TMEM 输出；由同组分 key 改为交替拥有整组，分别推进 QK，按组交接最大值、分母和 PV 完成关系。 | D128 的 key_split 候选中 gsplit 成立，默认 gspan>=8；入口 launch_tcg<128,PASS_FUSED,2,false,false,true>，使用 cp.async。 |
| [fused_key_d128](fused_key_d128.yaml) | root | D128、tcgen05、非 split，小融合网格且有效 key span 较短。 | 改用两 warpgroup 协作同组的 key 切片；四阶段环、双 score 槽让下一组 QK 和前组 PV 跨迭代在途，保留完整组最大值。 | D128 的 key_split 成立且 gsplit 不成立，默认 gspan<8；入口 launch_tcg<128,PASS_FUSED,2> 的 PIPE 循环，使用 cp.async。 |
| [fused_tc](fused_tc.yaml) | root | D128/D256，实际规划未用 tcgen05，非 split。 | 保留 CTA 内融合主线，改用 64-row tile、mma.sync QK 的 MX block 显式尺度折入，以及概率 FP8 舍入后转 BF16 的 PV；输出分子留在寄存器。 | launch_sequence 的 use_tc 分支且未实际 split；进入 launch_tc<D,4>(PASS_FUSED)，D 取 128 或 256。 |
| [split_prefix_d128](split_prefix_d128.yaml) | split_tcg | D128、tcgen05、实际 split，规划器选择 scan fusion。 | 保留 group-max，取消独立 scan；split CTA 重放 bm 前缀、写未加权 FP32 partials 和 part_m，末分片写 mtail，由 combine 计算归一化权重。 | HostPlan.use_tcg、实际 split 且 HostPlan.scan_fused；默认仅 D128，按 scan2p 门槛选 gspan 上限 48 或 160；launch_tcg_split 启用 FSCAN，combine 启用 FWGT。 |
| [split_tc](split_tc.yaml) | split_tcg | D128/D256，实际规划未用 tcgen05，且 workspace 允许 split。 | 保留 group-max、独立 scan、FP32 partials 和 combine；QK/PV 改为 mma.sync、64-row tile、显式尺度折入和 BF16 PV。保留 group-max 网格覆盖缺陷。 | launch_sequence 的 use_tc 分支且实际 split；launch_tc<D,4> 分别执行 PASS_GROUP_MAX/PASS_SPLIT，再 launch_combine<D>，不启用 FSCAN。 |
| [split_tcg](split_tcg.yaml) | root | D128/D256、tcgen05、实际 split，且未融合 scan。 | 由融合改为 group-max → 独立 scan → split QK/PV 重算 → combine，物化 bm、mstate、mtail 及 FP32 分子分母 partials；D128/D256 分别用 KSPLIT=1/2、寄存器/TMEM 分子。 | HostPlan.use_tcg、实际 split 且非 HostPlan.scan_fused；GM 使用 gm_nsplit，split 使用 nsplit；入口 launch_tcg(PASS_GROUP_MAX)、scan、launch_tcg_split、launch_combine。 |

## 覆盖缺口

表中 HostPlan 指固定源码 `fmha_mxfp8_qkv.cu::plan_geometry` 的实际返回值，包含 mask、代价、容量及运行时探测结果；实际 split 还须通过 forward 的非空 workspace 与字节数门槛。融合网格为 `fused_ctas=B*KH*HostPlan.qtiles128`，D128 小网格指其不超过 `device_sm_count()`。`gspan` 取 host 的 `useful_key_span`，不能用 `ceil_div(Skv,128)` 替换有 mask 的跨度。

直接父仅表示设计复制来源；各节点的条件在完整共享 Contract 域独立求值，不叠加父条件。精确条件与版本以各 YAML 为准，计划内派生文件在 P5 生成。

D/PASS 的组内参数、flat/分段 scan、combine 的宽窄组织、pad skip、分片数、CTA 顺序及允许的 TMA/cp.async 搬运变体留在对应节点内。它们不改变该节点是否物化跨 CTA 状态及归一化所在阶段；局部流水线差异仍须在节点设计中保留。Graph 捕获/重放与直接提交执行同一 launch_sequence，不新增节点。固定默认关闭 BF16 partials、GM-QK64、scan B1/B2 及 KSPLIT=4 实验，不将这些编译变体列为运行路径。

- 共享域的 head 分组为 G=1、4、16，均满足 use_tc；当前 use_tcg 赋值使 TCV 分支不可达，scalar 亦不可达。八个节点保留默认构建在该域内的 host 路由；运行时探测失败走实际 mma.sync 分支，workspace 不足走相应融合分支，不据此推断正确性或硬件可用保证。允许 TMA 的入口在描述符失败时有源码的 cp.async 变体。

- `split_tc` 的 group-max 使用 `qtiles128=ceil_div(Sq,128/G)`，kernel 却按 `Qph=64/G` 映射 query；当 `qtiles128<qtiles` 时，每个 head 后部 `i>=qtiles128*(64/G)` 的 bm 行未写入，后续 scan 仍遍历全部 rows。源码没有为此补写或初始化保证；该节点记录可达执行，但不声称在此子域满足 Contract 数值输出，也不虚构正确 fallback。

- P1 的外部比较器 source_gap 沿用 root.tolerance：已知 atol=0.01、rtol=0.05、max_mismatch_fraction=0.0；公式、相对误差基准、比较精度与 NaN/Inf 处理未提供。它只限制外部验收边界复现，不用于任何节点的数值或依赖保证。
