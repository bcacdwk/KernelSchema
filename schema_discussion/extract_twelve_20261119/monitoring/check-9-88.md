# 巡检 88

采样与完成时间为 2026-11-20 17:49:22 北京时间（UTC+8）；`clock.committed=true`，下次到期为 18:19:22。

已交付 10/12，本次新增 FP8 per-token/head FMHA 的 root、decode_split_kv、tcgen05_prefill 三个设计。核实 ready/published、全部调用结清、回执等于发布证明、run/checkpoint 匹配，并逐文件验证 SHA，见 [交付核验](../deliveries/fmha_fp8_per_token_head.json)。

KDA 的 stale 摘要再次以 `status --full` 复核，其权威交付状态和 checkpoint 不变，见 [权威状态](check-9-88-kda-status.json)。最后两个任务 `bf16_nvfp4_gemm` 与 `fp8_scaled_fused_moe` 均处于 P6，共 2 个在途调用，无当前异常，本次无需恢复。
