# 巡检 87

采样时间为 2026-11-20 17:18:54、完成时间为 17:18:55，均为北京时间（UTC+8）；`clock.committed=true`，下次到期为 17:48:55。

已交付 9/12，本次新增吸收式 MLA 的 root、simt、split_kv 三个设计。核实 ready/published、全部调用结清、回执等于发布证明、run/checkpoint 匹配，并逐文件验证 SHA，见 [交付核验](../deliveries/mla_absorbed_fp8.json)。

KDA 的摘要仍标 stale，已另用 `status --full` 核实权威交付状态与 checkpoint 不变，见 [权威状态](check-9-87-kda-status.json)。其余 2 项 P6、1 项 P5，共 3 个在途调用，当前无异常，本次无需恢复。
