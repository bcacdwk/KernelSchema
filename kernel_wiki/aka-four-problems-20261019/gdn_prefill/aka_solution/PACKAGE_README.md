# GDN expert kernel — FP16/M64 版 与 BF16 适配版

打包时间：2026-09-21（CST）。两份代码快照 + 完整验证/性能/NCU 证据。
两个目录都是 **8 个文件**、逐字节校验过，与 `BUILD_REPORT.json` 的 `hashes` 完全一致。

## 目录

| 路径 | 含义 | tree sha256（8 files） |
|---|---|---|
| `expert_flashinfer_fp16_m64/` | expert 的 **FP16 血统原版**（含战役自研的 M64 value-row 路径），取自父 payload `gdn_threeway_20260911` | `5e744b8b2499bed876b69fb23ccf325de5010d93955fef92a30c9f81b0e0f961` |
| `expert_flashinfer_bf16/` | **BF16 适配版**（selector 同时接受 FP16/BF16 IO，并暴露 `value_rows_override=None/64/128`），取自 `gdn_expert_bf16_m64_ablation_20260911` | `3f3f380fe277f1d8b9bc6112f189256f155cc91e8615a466a1a07299c05d832e` |
| `fp16_to_bf16_selector.diff` | 两者唯一差异：`gdn_kernels/blackwell/gdn_prefill.py`（168 行 unified diff） | — |
| `provenance/` | BUILD_REPORT ×2、validation ×2、core/e2e summary、NCU aggregate、上游 LICENSE 与溯源 | — |

> 两棵树只有 `gdn_kernels/blackwell/gdn_prefill.py` 一个文件不同；`cute_dsl/*`、
> `gated_delta_net_chunked.py`、`gated_delta_net_tile_scheduler.py` 与 4 个 `__init__.py` 完全一致。
> 即 **FP16→BF16 是 selector 层的适配，没有改动数值路径或 kernel 主体。**

## 血统（重要）

- 上游：`https://github.com/flashinfer-ai/flashinfer.git` @ `2ab910c58fdd2392914ea05e2a8714946ac0eef6`（2026-08-10），Apache-2.0（LICENSE 见 `provenance/upstream_seed/`）。
- 上游抽取策略只含"官方 non-CP SM100/SM103 GDN fallback 及其直接 helper"，且 `UPSTREAM_PROVENANCE.json` 明确
  **`excluded: ["all context-parallel GDN implementations", "all M64 implementations", ...]`**。
  → **M64 value-row 路径不是上游 flashinfer 的，是 AKA 战役自己加的。**
- expert 候选溯源：`accepted_candidate_commit=51d2d4b8e0adac56e1948d31edb3b6e86b378e55`、
  `promotion_commit=95a399d055c9a569fce9e84f7672e6f07a532515`、
  `canonical_source_commit=4102758d75535ac2b54c48421c3d91fe74e16c33`
  （源仓库 `GDN_AKA_REPRO_20260910/gdn_aka_runs/kernel_opt_chunk_gated_delta_rule_cutedsl_b300_production`，
  editable root = `vendor/flashinfer_gdn_seed/flashinfer`，故包名为 `expert_flashinfer`）。
- 同台对照 arm：`official` = 官方 flashinfer @`2ab910c5`；`current` = AKA 自研
  `qoder_greenfield_prefill_gdn` @`06b8e75394276db0f017f0617bd2c5eef6c5fc7d`（03_prefill_gdn 的 v25）；
  `fla` = 公开 vLLM v0.13.0 @`72506c98349d6bcd32b4e33eec7b5513453c1502`。

## BF16 契约与验证

- 契约：`BF16 Q/K/V/output` + `FP32 alpha/log_alpha/beta/state`；10 个 shape（id 0–9），拓扑沿用父 payload；
  correctness = finite + dtype match + `relative_l2 < 0.01`（output 与 final_state），`correctness_threshold_changed: false`。
- **最终验证：90/90 全通过**（`provenance/validation_bf16_passed_90of90.json`）：
  `official` 20/20、`expert_m128` 20/20、`expert_m64` 20/20、`current` 20/20、`fla` 10/10。
- 第一次全量校验（job `dv_efdb745569a3`）`passed: false`，但 8 条失败**全在 `current`（AKA 自研）arm 的 e2e**，
  且是适配器层形状错误而非数值错误：
  `ValueError: Mismatched q.shape[1] ... expected to be 4`，抛在 `qoder_greenfield_prefill_gdn/operator.py:53` 的 `normalize_qk(q3, k3)`。
  同轮 expert_m64 / expert_m128 / official / fla 全过 → **BF16 适配本身无问题**。修好后重跑即 90/90。

## 实测（NVIDIA L20D, capability (10,3), torch 2.9.0+cu130, CUDA 13.0, cutlass_dsl 4.6.1, python 3.12.3）

计时口径：warmup 10 / timed 100 per pass，CUDA Events 为正式计时器，NCU 仅用于解释（`ncu_role: explanation only`）。

### core（prepared normalized BF16 Q/K，log-alpha 在计时外预计算；8 rounds，8 个镜像顺序）

| arm | geomean µs | max_iteration_cv | max_pass_median_cv |
|---|---|---|---|
| official | 230.204 | 0.4339 | 0.0115 |
| expert_m128 | 224.888 | 0.0237 | 0.0047 |
| **expert_m64** | **184.559** | 0.1610 | 0.0099 |
| current (AKA v25) | 196.668 | 0.0207 | 0.0099 |

`expert_m64` 加速比：vs `expert_m128` **1.2185×**、vs `official` **1.2473×**、vs `current` **1.0656×**。`all_correct: true`，wall 96.6 s。

### e2e（raw BF16 Q/K/V + FP32 log-alpha/beta/state；10 rounds）

| arm | geomean µs | max_iteration_cv |
|---|---|---|
| official | 524.977 | 0.4025 |
| expert_m128 | 518.307 | 0.0375 |
| expert_m64 | 472.575 | 0.0453 |
| **current (AKA v25)** | **228.770** | 0.0418 |
| fla (vLLM) | 890.486 | 0.4569 |

`expert_m64` 加速比：vs `expert_m128` **1.0968×**、vs `official` **1.1109×**、vs `current` **0.4841×**。`all_correct: true`，wall 185.2 s。

### 解读注意

- **core 与 e2e 结论相反**：纯 kernel 口径 expert_m64 比 AKA v25 快 6.6%，但 e2e 口径 AKA v25 快约 2.07×。
- core 口径的 `max_iteration_cv` 偏大（official 0.434、expert_m64 0.161），**core 那 6.6% 的领先置信度低于 e2e**；
  e2e 各 arm cv ≤ 0.046，结论更硬。
- NCU：shape 0/4/9，`--clock-control none --cache-control none --profile-from-start off --metrics gpu__time_duration.sum`，
  每 arm×shape 2 个镜像 pass、profiler 内 1 次 forward（aggregate 见 `provenance/ncu_*_summary.aggregate.json`）。
- 网关 job：cuda_event `dv_4b3e9bfc812e`、ncu_core `dv_180752c996df`、ncu_e2e `dv_394ff7f0650d`、首轮 validation `dv_efdb745569a3`。

## M64 选择规则（BF16 版新增，逐条来自 `get_gdn_launch_metadata()`）

`natural_m64_eligible` 需同时满足：`capability == (10,3)`、`dk == 128`、`k.size(2) == 128`、`v.size(2) == 128`、
`hv >= hq`、`hv % hq == 0`、`initial_state`/`output_state` 均存在且为 `float32`、`not enable_checkpoints`、`not use_state_indices`。

- `value_rows_override=64` 但非自然 eligible → **直接 raise**（不静默降级）。
- `value_rows_override=128` → 关闭 M64。
- 返回值含 `requested_value_rows` / `selected_value_rows` / `natural_m64_eligible` / `use_m64_rows` /
  `use_static_tma_descriptors` / `static_tma_full_chunks` / `use_static_scale` / `compact_cg1_events`，适配器在计时前校验"请求 vs 实际选中"。
- 入参 dtype 硬校验：`q.dtype ∈ {float16, bfloat16}`，`k/v/output` 必须与 `q` 同 dtype，`gate/beta` 必须 `float32`。

## 已知遗留（打包时的状态）

1. `BUILD_REPORT.json` 的 `git_commit_created: false` —— BF16 适配从未落 commit，只存在于 payload 快照目录。
   **本包是这份代码的第二个持久副本**，请勿删除。
2. 当时没有生成任何 `.md` 报告（`results/` 下 7 个 md 对 `expert_m64` 零命中），结论只在 JSON 里；本 README 即补该缺口。
3. 未固化"10 个 shape 中哪些是自然 M64 eligible、哪些靠 override 强制"的清单。
4. 相关的后续 B 线（把 expert 的 SMEM stage 配比 Q3/K4/V2/O1 回植进 AKA 自己的 BF16 kernel）**判定为不采纳**：
   BF16 契约 `formal_long_speedup 1.0254`、FP16 参考契约 `1.0130`，两者均 `accepted: false`
   （`no_long_shape_regressions: false`，且 candidate 方差偏大）。该改动仍停在
   `/root/workspace/gdn-expert-bf16-port-20260911` 的未提交工作树里，未进 03_prefill_gdn 主线；主线最优仍是 v25。
