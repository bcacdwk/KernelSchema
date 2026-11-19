# 同线程提醒

先执行 ack_command。accepted=true 时执行一次 scan_command；already_acknowledged 时用同一 scan_command 续完，其余拒绝原因结束本提醒。

按 manual 核对阶段、在途调用与异常；正常简报即可。简报中的采样、完成和下次时间统一用北京时间（UTC+8），分别沿用 observed_at、completed_at 与 next_due_at 的含义；停用时不把保留的到期记录描述为仍在计时。需处理时遵守已有授权、作用域和已用次数。以 scan 的 clock.committed=true 或 reused=true 及 next_due_at 确认登记；scan_busy/scan_invalidated 时执行 status_command 核对。superseded_run_ids 表示该次采样已被更新观察超过，不把它描述为最新状态。

ack 后中断时显式续同一 scan，不重复 ack。后台观察独立于提醒确认；需要恢复时由 Supervisor 决定。完成巡检后结束提醒，继续尊重用户最新暂停、讨论及尚未完成的授权事项；不另建线程、闹钟或在对话中 sleep。
