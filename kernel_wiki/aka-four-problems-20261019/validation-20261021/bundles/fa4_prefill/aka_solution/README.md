# AKA solution: FA4 Prefill C05 v43

这是从 05 Prefill Attention Increment 战役冻结出来的 SM103 2CTA FA4 Prefill 实现，直接匹配本题的 FP8 E4M3、page64、head_dim=256 ABI。

- 入口：`solution.py::Model`
- 核心源码 SHA-256：`3cc52e747c956df415525bf597ea9dbc4d1a9ccb196500eea465dbaca35abadc`
- 来源 commit：`0b727fb40a9fd8b85e34e33a6f59f2f5330d744b`（`v43: long-horizon episode 122 candidate_ready`）
- Agate L20D job：`dv_afc2cc3e832e`
- 正确性：30/30 shapes passed，`atol=0.06`、`rtol=0.04`

`flash_attn/` 和 `quack/` 是该冻结版本运行所需的源码闭包；无需安装另一份 FlashAttention 源码包。`AGATE_VALIDATION.json` 是网关返回的逐 shape 结果，`CAMPAIGN_V43.json` 是战役原始记录（与战役 `memory/v43.json` 字节一致，SHA-256 `3ffb5aef923edd07df1fd4c81126fb3ae58c5947355d3a49520c156afb439386`）。

## 版本口径：v43 是 v37 的 carry-forward

`v43` 只是本包冻结的源码快照标签，不是战役最后一次成功晋升。读 `CAMPAIGN_V43.json` 时必须注意：

- `quality_gate.result = FAIL`，原因 `candidate improvement 0.5946143153829775 did not exceed 1.000%`；
- `optimization.expected_impact = episode exploration did not produce a promotable improvement`，episode 122 的描述是 “ships a bit-identical gated pipeline-depth candidate”、“incumbent kernel was preserved”；
- 因此它的性能字段是 `measurement_subject = incumbent`、`measurement_source = canonical_incumbent_carry_forward`、`carried_from_version = v37`，即这组数字描述的是 v37 既有 kernel，不是 episode 122 的新收益。

这条 kernel 线的**权威晋升记录是 v37**（战役 `memory/v37.json`，SHA-256 `569faec3bc13a42c0e986758118e7a70fecfada0de38085c30808475e696b9c8`）：

- `quality_gate.result = PASS`，`measurement_subject = candidate`，`measurement_source = authoritative_verification`；
- 30 个不透明 evaluator shape 的延迟几何平均 `1250.0801957223607 us`，`speedup_vs_ref_geomean = 1.0485106268462285`，`authoritative_improvement_pct = 2.2356189927478494`；
- 测量口径 A-B-B-A、warmup 10、timed_runs 100；commit `b883c6edb80e1d36f9477e92c5692f704cbb0abd`；
- v37 的改动是 `interface.py only (+113/-1)`，kernel 字节未变，所以这组数字可以合法地归属到本包冻结的 kernel。

v42/v43/v44 三个版本都是同一份 v37 carry-forward，quality gate 均为 FAIL。战役在 v43 之后继续到 v80 并累计 27 个 episode 未晋升，本包不采用 v43 之后的任何树。

## 战役后续：v53 已被接受，但没有打进本包

本包冻结于 v43（commit `0b727fb4`，`2026-09-13T01:10`）。战役在此之后又成功晋升过一次：

- **v53**，`2026-09-14T07:55`，commit `d75b780acc60d3fb20f1fe72cd7c891d7a7255c7`
- `quality_gate.result = PASS`，`measurement_subject = candidate`，`measurement_source = authoritative_verification`
- 30 个不透明 evaluator shape 的延迟几何平均 `1236.9776665613158 us`，`speedup_vs_ref_geomean = 1.0596168427750163`，`authoritative_improvement_pct = 2.4722130921941288`
- 记录已随包附上：`CAMPAIGN_V53_NOT_SHIPPED.json`（与战役 `memory/v53.json` 字节一致，SHA-256 `1009670b61a8881c07173ef5ec13a42583ff33244e8cda67c7827620a8631623`）

即 v53 比本包所依据的 v37 线**延迟低 1.05%**（`1236.98` vs `1250.08 us`）、加速比更高（`1.0596x` vs `1.0485x`）。

该战役共 81 个版本、7 个 PASS（v0/v1/v32/v33/v36/v37/v53），**最后一次 PASS 是 v53**。因此本包是四道题中唯一一个
落后于其战役最终接受版本的：Decode 交付的是该战役最后 PASS 的 v55，QSA 交付的是最后 PASS 的 v7，两者都对齐。

没有直接换成 v53 的原因：本包的正确性证据（`dv_afc2cc3e832e`，30/30 shape）是针对 v43 冻结树跑出来的，
v53 树未在本题上做过 Agate 复验。若要采用 v53，需要重新打包源码闭包并在本题上重跑 30 shape 正确性与性能。

## 与本题 production baseline 的区别

战役的 `ref` 不是本题的 production baseline。本题 `solution.py` 走 FlashInfer `trtllm_batch_context_with_kv_cache`，所以 `1.0485x` 不能当成“对本题生产实现的加速比”。AKA 与本题 production 的同进程对比在 `验证/ncu-aka-vs-production-20260922/`（3 个代表 shape）与 `验证/ncu-full-20260922/`（全量 shape）中单独给出，两者口径不同，不可混用。
