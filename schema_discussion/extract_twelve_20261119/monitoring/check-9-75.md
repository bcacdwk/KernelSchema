# 巡检 75

采样与完成时间为 2026-11-20 11:13:29 北京时间（UTC+8）；`clock.committed=true`，下次到期为 11:43:29。

已交付 3/12：BF16 MoE、MLA NVFP4、masked GEMM。本次新增两项分别核实 ready/published、调用全部结清、正式回执等于发布证明、run/checkpoint 匹配，以及 root.yaml/dispatch.md 的实际 SHA 与回执一致。证据见 [MLA NVFP4](../deliveries/mla_nvfp4.json) 和 [masked GEMM](../deliveries/fp8_mxfp4_grouped_gemm_masked.json)。

其余 1 项 P6、8 项 P5，共 16 个在途调用。GDN 已从 P3 修订返回 P6。分页索引 Sparse 的 `P5:cluster_m128` 缺绑定实现入口，原 worker 正在自动补正，2/32 次；兄弟 `split8_merge` 正在 reviewer 审阅，4/32 次。[定向 inspect](check-9-75-inspect.json) 确认相关进程存活、无控制屏障或未确认终止，本次保持现有流程。
