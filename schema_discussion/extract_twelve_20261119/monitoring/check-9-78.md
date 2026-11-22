# 巡检 78

采样与完成时间为 2026-11-20 12:44:38 北京时间（UTC+8）；`clock.committed=true`，下次到期为 13:14:38。

已交付 4/12，本次新增 GDN：root、fused_single_launch、warp_row_reduction 三个设计均已发布。核实 ready/published、全部调用结清、回执与发布证明一致，并逐文件验证三个 YAML 和 dispatch 的 SHA。证据见 [GDN 交付核验](../deliveries/gdn_conv_mtp_bf16_state_bf16.json)。

其余 1 项 P6、7 项 P5，共 11 个在途调用。采样记录 3 项回执诊断；[随后定向 inspect](check-9-78-inspect.json) 中，吸收式 MLA 的 split_kv 已进入 reviewer，入口引用诊断消失。

仍在自动补正的两个范围：GEMM `split_k_reduce` 的引用结束行 276 超过实际 275 行，reviewer 在途，4/32 次；FP8 MoE `split_k_decode` 缺绑定实现入口，worker 在途，3/32 次。相关进程存活、无控制屏障或未确认终止，本次不额外恢复或增额。
