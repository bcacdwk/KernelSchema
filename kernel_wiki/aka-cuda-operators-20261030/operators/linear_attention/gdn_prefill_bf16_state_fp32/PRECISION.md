# gdn_prefill_bf16_state_fp32 精度记录

当前源码 `aeca6a148d2306a2036e64dfd1eb51b7413de0f69695c1ef06b31672d07c9e3f`：固定域 10 shape × 2 seed、声明域 10 shape × 2 seed、ABI 3 shape 全部通过冻结 checker；容差未放宽（atol 0.01、rtol 0.05、max_mismatch_fraction 0）。160 条记录中 80 条为数值检查，0 条存在超差元素，覆盖 6.41 亿个元素。逐元素判定为 `|candidate-reference| > atol + rtol*|candidate|`，允许超差数 0；同一 alias 可能对应两条记录，不累加成独立错误元素数。

| 阶段 / 输出 | 数值检查 | 最大绝对误差 | 最大 relative L2 | 最大逐元素相对差 |
|---|---:|---:|---:|---:|
| fixed output[0]（BF16 attention out） | 20 | 0.00195312 | 0.00449 | 0.0916 |
| fixed output[1]（FP32 final state） | 18 | 0.00816005 | 0.00362 | 0.4567 |
| fixed input.core_attn_out（alias 写入） | 2 | 0.000244141 | 0.00417 | 0.0122 |
| domain output[0] | 20 | 0.00390625 | 0.00417 | 0.0971 |
| domain output[1] | 20 | 0.0138087 | 0.00317 | 0.6036 |

output[1] 的最大绝对误差可以超过 atol，由 rtol·|candidate| 覆盖；全域最紧的一条检查容差占用率为 0.6121（见下）。完整数值、容差与逐 shape 明细见 [performance/ACCEPTANCE.json](performance/ACCEPTANCE.json) 的 `fixed` / `domain` / `abi`。

## ragged CP 最终 kernel 的 split-BF16 残差补偿已移除

起点源码在 ragged CP 路线（固定 shape 4/6，声明域 d3/d4）为 state、vks、nv、decay_v、qk 五个矩阵各维护 hi + residual 两份 BF16，每 64-token block 发 64 条 `tcgen05.mma`；等长 CP 路线只有 28 条。这是 shape 4/6 相对 FlashInfer 生产 kernel 的主要回退来源（生产对 shape 1/4/8 使用同一个无补偿 `CPDeltaRulePrefillTcgen05Sm100`）。移除依据（全部为本工作区实测证据）：

1. **冻结 checker（当前源码）**：abi / fixed 10 shape / domain 10 shape 全通过，0 超差元素。回执 `evidence/ncu-repair-20261021/raw/nores1-validate/results/`。
2. **runtime mask 扫描**：同一 cubin 只把残差数值置零（不改指令数、不改流水），mask ∈ {31,30,29,27,23,15,0} 在 shape 4/6 + d3/d4 全部 `mismatched_elements=0`。证据 `evidence/ncu-repair-20261021/raw/compmask-sweep/results/probe-summary.json`。
3. **配对余量研究**：8 个覆盖 generic 路线的域点 × 2 seed，同一 FP32 reference 配对比较（含 16383 单序列 ragged、4 序列 ragged、cancellation、slow_decay、zero/mixed/continuation、tp 1/2/4/8、head_ratio 2/3）。output[0] 最坏容差占用率：全补偿 0.0868 → 无补偿 0.1188；`max_abs_diff` 两者同为 0.001953，说明最坏逐元素误差由输出 BF16 量化主导。output[1]（由 fixup 扫描产出的 FP32 state）与补偿完全无关，占用率恒为 0.6121、`max_abs_diff` 恒为 0.013947。证据 `evidence/ncu-repair-20261021/raw/margin-probe/margin-probe.json`。

4. **等价性核对**：交付源码在 shape 4/6 的 8 条冻结检查上，output[0]/output[1] 的 relative L2 与 mask=0 变体逐位一致（12 位小数相同），output[1] 同时与 mask=31 一致 → 删除是外科手术式的，等价于把残差置零，没有引入额外数值变化。

5. **交付源码的补充种子余量研究**：5 个最难的 generic 路线域点（shape4/6 同构、d3 同构、16383 单序列 ragged、4 序列 ragged mixed）× 3 个附加种子（2026101227/29/31，非冻结种子）直接测交付源码：output[0] 最坏容差占用率 0.1191、最坏 `max_abs_diff` 0.00195312、最坏 rel_l2 0.0045、0 超差、全部有限；output[1] 最坏占用率 0.6464、最坏 `max_abs_diff` 0.0128、0 超差。与第 3 条（0.1188 / 0.6121）一致，说明余量对种子稳定。证据 `evidence/ncu-repair-20261021/raw/margin-probe-seeds/margin-probe.json`。

全域最紧的检查始终是 output[1]（FP32 final state，由 fixup 扫描产出，本轮源码逐字节未改），占用率 0.61–0.65；output[0] 有约 8× 余量。

结论：补偿只改变 output[0] 的 RMS 误差（rel_l2 0.0034 → 0.0046），不改变最坏逐元素误差；移除后 output[0] 仍有约 8× 容差余量，output[1] 逐位不变。未修改 reference、shape_cases、metadata、checker 或容差；未恢复 BF16 卷积乘积截断；未改动非 CP 路线与等长 CP 路线的数值处理；未调用任何原库计算。

限制：以上为采样证据（固定 10 shape + 声明域 10 shape，各 2 冻结 seed；8 个附加域点 × 2 冻结 seed；5 个最难域点 × 3 个附加非冻结 seed；SM103 实测），不构成全域零误差或逐 bit 相同的证明。
