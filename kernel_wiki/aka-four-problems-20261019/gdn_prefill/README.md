# GDN Prefill SM103

这是 `bcacdwk/AKA_experiment@6e0d79ca` 的 `assets/bench-new` 中独立新增的
GDN Prefill 题，不属于 `atrex-bench-new-qwen38-fa4-qsa-20260920.tar.gz`
中的三道导入题。

本目录根部保留原题的输入、reference、metadata、shape 与固定 FlashInfer 提交记录。
`solution/` 保存从 FlashInfer `2ab910c58fdd2392914ea05e2a8714946ac0eef6`
抽取的 SM100/SM103 CuTeDSL kernel、题目 ABI adapter、依赖和真实 Agate 验证结果。

`aka_solution/` 保存 `gdn_expert_m64_fp16_and_bf16_20260921.tar.gz` 中的
AKA expert BF16/M64 kernel。目录内已修正 B=1、奇数个完整 64-token chunk 的
静态 TMA padding 谓词，所有题目 shape 均使用 M64、不回退 M128。最终产物经真实
Agate 作业 `ev_2a8e61d8f66f` 验证为 30/30 shape、180/180 case 通过。

2026-09-22 在同一块 L20D 上补测性能：`solution/solution.py`（FlashInfer M128
baseline）几何平均 `142.328 us`（`ev_befb2cf452a2`），`aka_solution/solution.py`
（纯 M64）两次运行均值 `121.297 us`（`ev_3bfc6d604a12`、`ev_23ed7646f8bf`），
即 **1.173x（低 14.78% 延迟）**，但有 5 个 shape 回退，最差 shape 0 为 0.645x。
回退根因是 persistent scheduler 的 wave quantization（M64 的 grid 是
`min(B*64, num_sm)`，M128 是 `min(B*32, num_sm)`）。逐 shape 明细、强制 M128
对照组与拟合成本模型见 `../../验证/gdn-prefill-perf-20260922/性能结论.md`。

完整上游说明与历史证据仍保存在：

`/Users/ruibo/work/AKA_experiment/assets/bench-new/GDN_PREFILL_SM103.md`
