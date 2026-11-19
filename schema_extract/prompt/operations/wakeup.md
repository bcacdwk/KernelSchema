# 同线程提醒

先执行 ack_command：accepted=true 时执行一次 scan_command；already_acknowledged（或 ack 后中断）时直接续同一 scan_command，不重复 ack；其余拒绝原因结束本提醒。以 scan 的 clock.committed=true 或 reused=true 及 next_due_at 确认完成；scan_busy/scan_invalidated 时执行 status_command 核对。

按 manual 核对阶段、在途调用与异常，正常简报即可。采样（observed_at）、完成（completed_at）和下次到期（next_due_at）统一用北京时间（UTC+8）表述；提醒停用后，保留的到期记录不表示仍在计时。superseded_run_ids 表示该次采样已被更新观察超越，不当作当前状态。

后台观察独立于提醒确认；是否恢复由 Supervisor 在已有授权、作用域和已用次数内决定。巡检完成后结束本提醒，继续尊重用户最新暂停、讨论及未完成事项；不另建线程、闹钟或在对话中 sleep。
