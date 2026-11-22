# 巡检 58

采样与完成时间为 2026-11-20 02:36:46 北京时间（UTC+8）；`clock.committed=true`，下次到期为 03:06:46。

采样时 4 项已进入 P3；其余 8 项位于 P2，其中 `mla_nvfp4` 停于 `invalid_receipt`。总在途调用为 11 个。GDN 已通过 P2 补正并进入 P3。

`bf16_nvfp4_gemm` 与 `paged_indexed_latent_fp8` 的 P2 reviewer 正在补正短交接说明；[定向 inspect](check-9-58-inspect.json) 确认两个 Runner 存活、调用在途且后端进程存在，各范围已用 6/32 次调用，没有未确认终止、全局 hold 或维护请求。保持现有补正，不追加并发调用。

`mla_nvfp4` 的停止交给新的 xhigh 子代理核查及按授权恢复。初步证据：reviewer 的 handoff 为 3011 字符，超过 1200 字符上限；P2 已用调用为 6/32，`receipt_repairs=3` 超过自动补正上限 2。此前两次计数来自 worker 证据坐标修补，本次是 reviewer 首次 handoff 超长。所有调用均已结清，停止不等于整个调用预算耗尽。

恢复限定为原 run、原来源、原规则与原已用计数，必须保留正常阶段门禁；不直接改状态、回执或批准，不新建替代 run。恢复证据及最终决定记录在 [MLA 恢复目录](../recovery/mla_nvfp4_handoff/)。

MLA 已经通过一次正式 `resume` 恢复为 P2/reviewer/running；新调用为本范围第 7/32 次，原补正计数及既有账目保留。子代理验证新调用收到真实超长诊断、精确沿用原 reviewer 会话，35 项保留性检查通过。尚未认定 P2 获批。

02:41:45 的[恢复后复核](check-9-58-after-recovery.json) 确认 12 个任务均在 running，共 12 个在途调用；4 项 P3、8 项 P2。目前两个 GEMM 和 `paged_indexed_latent_fp8` 的短交接回执仍在正常自动补正，继续由原门禁处理。
