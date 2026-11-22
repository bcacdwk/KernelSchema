# 巡检 89

采样与完成时间为 2026-11-20 18:19:50 北京时间（UTC+8）；`clock.committed=true`，下次到期为 18:49:50。

已交付 11/12，本次新增 FP8 scaled MoE 的 root、mma_sync_fused、split_k_decode 三个设计。核实 ready/published、全部调用结清、回执等于发布证明、run/checkpoint 一致，并逐文件验证实际 SHA，见 [交付核验](../deliveries/fp8_scaled_fused_moe.json)。

KDA 的 stale 摘要以 `status --full` 再次确认权威状态及 checkpoint 不变，见 [权威状态](check-9-89-kda-status.json)。仅剩 `bf16_nvfp4_gemm` 在 P6，1 个在途调用，无当前异常。本次不恢复或干预正常审阅。
