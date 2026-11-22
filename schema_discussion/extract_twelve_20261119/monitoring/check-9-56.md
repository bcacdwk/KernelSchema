# 巡检 56

采样与完成时间为 2026-11-20 01:36:01 北京时间（UTC+8）；`clock.committed=true`，下次到期为 02:06:01。12 个任务均在 P2，共 12 个在途调用；此前 FMHA 基线链接诊断本次未再报告。

| 算子 | 当前回执诊断 | 补正角色 | P2 已用调用 |
| --- | --- | --- | --- |
| `fp8_scaled_fused_moe` | 结论或 issue 的证据索引超出 `source_evidence`。 | worker | 3/32 |
| `fp8_mxfp4_grouped_gemm_masked` | 回执引用 root.yaml 第 121 行，实际文件只有 120 行。 | reviewer | 3/32 |
| `paged_latent_mxfp8_nvfp4` | 批准回执的短交接说明不满足协议要求。 | reviewer | 4/32 |

01:36:22 的 [定向 inspect](check-9-56-inspect.json) 确认三个原 run 的补正调用均在运行，Runner 存活且持锁、后端进程存在，没有未确认终止、全局 hold 或维护请求。

Supervisor 决定：让原范围内的有限回执补正继续，由正常门禁核验结果；不并发追加同范围调用、不增补次数、不重开 run。三项均尚未确认为修复完成。
