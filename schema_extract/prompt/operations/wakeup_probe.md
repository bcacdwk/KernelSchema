# 同线程唤醒验证

先执行 ack_command。accepted=false 时结束本提醒；accepted=true 时执行 status_command 核对线程确认，简短报告后结束本轮。

本次只验证唤醒，不巡检或控制提取，不改变正式周期，不另建线程、闹钟或在对话中 sleep。
