# AKA GDN expert BF16/M64 solution

这是 `gdn_expert_m64_fp16_and_bf16_20260921.tar.gz` 中 AKA expert M64
实现适配到本题 `qwen38_max_gdn_prefill_sm103_core` 的自包含源码闭包。

## 文件

- `gated_delta_net_tile_scheduler.py`：CuTeDSL tile scheduler。
- `gated_delta_net_chunked.py`：完整 M64/M128 GDN device kernel。
- `gdn_prefill.py`：编译缓存、dispatch 与 launch adapter。
- `solution.py`：Agate 验证过的纯 M64 单文件候选，SHA-256 为
  `e216c76d8c434233cba9042be79500438a522da13cc42090281cc138ae427c69`。
- `build_solution.py`：从上面三个可读源码重建 `solution.py`。
- `AGATE_VALIDATION.json`：真实 L20D 验证摘要及原始失败证据。
- `PROVENANCE.json`、`PACKAGE_README.md`、`LICENSE`：来源与许可证。

## 题目 ABI 适配

`Model.forward(q, k, v, alpha, beta, initial_state, cu_seqlens, scale,
output, output_state)` 原地写满并返回 `(output, output_state)`。题目提供的
`cu_seqlens` 是 int64，kernel ABI 要求 int32，因此转换发生在计时边界内。

题目所有 tensor dtype、head 数和 head dimension 都满足 BF16 M64 selector：
BF16 Q/K/V/output，FP32 alpha/beta/state，HQ/HK/HV=4/4/32，DK=DV=128，
目标为 SM103。

## 重要边界修复

包内原始 M64 实现在 shape 3（B=1、T=14784=231×64）会额外执行一个
配对 padding chunk。旧 selector 仅以 `T % 64 == 0` 判定所有执行块完整，
从而关闭 padding 谓词，把越界 TMA zero-fill 的 `gate=0` 当成真实 gate，
最终将 recurrent state 衰减为零。

当前源码将静态 full-chunk 快路径收紧为：

```python
static_tma_full_chunks = use_static_tma_descriptors and T % 128 == 0
```

`Model.forward` 对所有题目 shape 强制 `value_rows_override=64`，不再回退
M128。非 128 对齐长度继续使用静态 TMA descriptor，但保留末块谓词，以
`gate=1、beta=0` 表示中性 padding。

## 构建与验证

```bash
uv run --with 'python-minifier==3.3.0' \
  python build_solution.py --rename-locals --output solution.py

agate eval --gpu L20D \
  --candidate solution.py \
  --reference-dir ../solution/agate_reference \
  --mode correctness_only --num-correctness-cases 6 --job-timeout 1800
```

最终作业 `ev_2a8e61d8f66f`：30/30 shape、180/180 case 全部通过题目
逐元素容差；详见 `AGATE_VALIDATION.json`。

## 性能（2026-09-22，L20D/SM103）

四个 `--mode full` Agate 作业串行提交，`--bench-iters 100
--num-correctness-cases 6 --lock-clocks
--atrex-bench-version 0.1.0.post20260917160311`，指标为 30 个 shape
中位延迟的几何平均。全部作业 30/30 compile、30/30 correctness、
180/180 case、30/30 performance 通过，SM 时钟 2032 MHz（锁频请求未生效）。

| 候选 | job_id | geomean | 相对 baseline | 回退 shape |
|---|---|---:|---:|---:|
| baseline（`../solution/solution.py`，FlashInfer M128 pin） | `ev_befb2cf452a2` | 142.328 us | 1.000x | - |
| 本目录纯 M64（run 1） | `ev_3bfc6d604a12` | 121.777 us | 1.169x | 5 |
| 本目录纯 M64（run 2，同候选复跑） | `ev_23ed7646f8bf` | 120.816 us | 1.178x | 5 |
| 同源码强制 M128 | `ev_69ee30458ff9` | 137.382 us | 1.036x | 0 |

纯 M64 两次运行均值 `121.297 us`，相对 baseline 为 **1.173x（低 14.78% 延迟）**，
两次运行相差 0.79%。本轮 baseline 与历史记录 `ev_4e1e2792ff14`（141.824 us）
相差 0.36%，说明两天数据可比。

同源码强制 M128 只有 1.036x 但 **0 回退**，且在全部 30 个 shape 上都不慢于
baseline（最低 1.003x）。逐 shape 取 `min(M64, M128)` 的上界是
`116.934 us`，即 **1.217x 且 0 回退**：M64 在 24 个 shape 上更优，M128 在
shape `0/12/18/26/27/29` 上更优。

### M64 的 5 个回退点

| shape | batch | sequence_lengths | baseline | M64 | 比值 |
|---:|---:|---|---:|---:|---:|
| 0 | 3 | `[4224, 4224, 4224]` | 140.61 us | 217.99 us | 0.645x |
| 12 | 4 | `[10560, 876, 724, 4224]` | 314.37 us | 360.54 us | 0.872x |
| 29 | 15 | 15 段，最长 912 | 113.89 us | 135.25 us | 0.842x |
| 18 | 3 | `[3184, 3833, 1622]` | 130.27 us | 136.11 us | 0.957x |
| 26 | 3 | `[22, 29, 31]` | 29.70 us | 30.77 us | 0.965x |

### 回退根因：persistent scheduler 的 wave quantization

`gated_delta_net_chunked.py` 在构造 scheduler 时把
`num_v_heads` 设为 `h_v * (2 if split_value_rows else 1)`，并在 M64 下把
`is_GQA` 置 false，于是 `total_tiles = num_seqs * num_o_heads` 变成
`B * 64`（M128 为 `B * 32`）。persistent grid 是
`min(total_tiles, num_sm)`，CTA `c` 依次处理 tile `c, c+P, c+2P, ...`，
tile 编号为 `batch * num_o_heads + head`。因此 makespan 等于「单个 CTA
分到的 chunk 总数」的最大值。当 `B * 64 > num_sm` 时 M64 需要第二个且
填不满的 wave，而 M128 仍是单 wave；同时 M64 会把 K 流量翻倍
（源码注释：`The M64 path doubles the CTA grid and therefore duplicates K traffic`）。

用上述规则写的静态成本模型（`../../../验证/gdn-prefill-perf-20260922/wave_model.py`）
取 `num_sm = 148`（`agate env L20D` 报告的 `sm_count`；MAPE 扫描的最优点在 129，
但 `129..148` 区间内预测完全相同，扫描本身无法识别设备，因此采用实测值），
在 30 个 shape 上拟合出 M128 单 chunk 成本约为 M64 的 2.370 倍，
R² 分别为 0.996（M64）与 0.9992（M128）、MAPE 为 9.0%/3.1%，并能正确预测
28/30 个 shape 的 M64/M128 优胜方；两个例外（shape 26、27）都是
几十个 token 的小 shape，固定启动开销项占主导。

这 5 个回退点属于调度层的 wave 边界效应，不是正确性问题：本轮已确认
两条路径都能通过 30/30 shape、180/180 case。若要同时拿到 1.2x 和 0 回退，
需要在 `Model.forward` 中按该成本模型逐 shape 选择 `value_rows_override`，
尚未实现与验证。
