# 巡检 68

采样与完成时间为 2026-11-20 07:40:25 北京时间（UTC+8）；`clock.committed=true`，下次到期为 08:10:25。

所有任务均已通过 P3：11 项 P4、1 项 P5。采样时混合精度 Sparse 停于 P4 短交接校验；GDN 的 `P5:warp_row_reduction` 停止，另一个 P5 分支仍在运行。

两个新的 xhigh 子代理分别核查后按原授权受控恢复：

| 范围 | 核验结果 | 既有用量与处理边界 |
| --- | --- | --- |
| `paged_latent_mxfp8_nvfp4` P4 | 最新 handoff 1368 字符/3 非空行，超限 168 字符。这是本 scope 首次长度诊断，前两次自动补正是证据索引问题。 | P4 6/32，receipt_repairs=3、invalid_receipts=3、revisions=1；调用已结清，无控制屏障。只做一次原 run 普通恢复。 |
| GDN `P5:warp_row_reduction` | starter Version=1、authored=false；首次 worker 原稿 Version=2，违反首稿版本必须从 1 开始的规则，验证器没有误判。 | 此节点 1/32，invalid_receipts=1；调用已结清且终止确认。仅允许正式 `resume --scope P5:warp_row_reduction`，不取消或重启健康兄弟。 |

范围恢复保留原 source/rules、实例与依赖、稿件、批准和账目；健康兄弟可自然推进，不为核验暂停它。不直接改候选快照或状态，不改规则/工具，不增加调用额度，不另建 run。

详细证据见 [Sparse P4](../recovery/mixed_sparse_p4_handoff/) 和 [GDN P5](../recovery/gdn_p5_warp_row_version/)。

两处均已完成一次受控恢复。Sparse P4 为 6→7/32，新 reviewer 调用 `1a7c4502-7e41-447d-ab69-2b7ee2147525` 沿用原会话并收到长度诊断；必要保存检查通过。GDN scoped 请求 `17dd09f1-042b-4158-9593-c8eca059d266` 已消费，仅该节点 1→2/32，新 worker `06b543bd-bc28-45c3-9763-15874fb5c232` 已启动并收到版本号诊断；健康兄弟 reviewer 的同一进程继续运行。

07:44:45 的[恢复后复核](check-9-68-after-recovery.json) 确认全部 12 项 running：11 项 P4、1 项 P5，合计 13 个在途调用，无当前诊断。两处新结果均尚待正常检查及审阅，不把运行恢复表述为已经通过门禁。
