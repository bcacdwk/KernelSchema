# R7 B：运行与辅助功能裁剪

从实际用途重新检查运行、观察、审计和模型提示。先向主代理说明取舍，再修改；没有增加替代框架或兼容入口。

## 删改与取舍

- 删除 `inspect --start`。观察命令现在只读；排队运行统一通过已有 `resume` 启动，后台提取仍用 `--background`。
- 主动放弃 `monitor start --probe-delay` 的独立唤醒验证能力，删除其专用 prompt。正式提醒本身已有同线程 ack、真实 scan 和消息证据，不再另外排一条只验证唤醒的消息。相应删除消息 kind 分支、恒定的 scan_required/unfinished 字段、verified 状态和 last_check 副本；保留当前 reminder、有限 history 与最近 last_scan。迟到的消息 ID 仍补入历史及扫描记录。
- 删除 wakeup 测试对两份相同存储副本的重复断言及内部 sending 状态断言，继续检查真正重要的消息证据、已完成扫描、停止失效和恢复后的新提醒。
- 配合 A 删除阶段 `handoff.md` 的材料副本和无用 stage/node 元组。当前已接受 receipt 自带 handoff，当前决定和跨阶段事实仍可定位；中断恢复交接保留。
- 配合 A 移除新建输出目录的 `.gitkeep` 例外预检和说明。新目标须不存在，已登记运行及完整相同交付的重放保持原路径。
- 收短普通 worker/reviewer、恢复和签收提示：删除由 Runner 管理的会话退休、调用计数和预算实现说明，保留角色动作、权限、证据与决策要求。固定语义提示仍在 Markdown。

持久提醒与有限自动恢复仍有长程运行价值；audit 封存服务独立读取具体证据，review-eval 服务已知缺陷的独立审阅能力回归，原生用量观察保留真实口径，因此未删这些能力。默认 Qwen、显式 Codex、角色权限、精确会话、真实退出及成功判定、终止 fence、预算和独立批准均未改。

## 验证与未决

使用受限 `PATH=/usr/bin:/bin` 与 Python 3.13，先确认真实 claude/codex 不可用；模型调用仅用绝对路径 fake CLI，真实 queue 发送设为直接失败。

- wakeup 两项既有测试及 CLI 的 `test_resume_adds_finite_credit_after_expiry_without_refunding_calls`：3 项通过，26.541 秒。覆盖持久限次重试、ack/scan/stop 与迟到回执、冻结入口启动、普通 EX_TEMPFAIL 和预算不退款。
- 小检查通过：inspect 不创建不存在的状态目录；删去的参数明确拒绝；ack/scan 后 audit 仍有线程、消息 ID 及 queued 时间证据。
- A 同轮已验证当前 receipt.handoff 可达、真实临时发布及冲突保护，本代理未重复运行。最后的 approval 列表简化仅去掉未使用的元组成员。
- 五个修改的 Python 模块解析及限定 `git diff --check` 通过，测试根关联子进程剩余 0。

无已确认未修问题。没有真实模型、queue 投递、GPU、完整提取、发布或 commit；临时目录中的发布验证由 A 执行，不代表真实交付质量。

## 文件

- `schema_extract/tools/{cli,wakeup,audit,monitor,prompts}.py`
- `schema_extract/tools/tests/test_wakeup_retry.py`
- `schema_extract/README.md`
- `schema_extract/prompt/operations/{normal_worker,normal_reviewer,recovery,signoff,wakeup}.md`
- 删除 `schema_extract/prompt/operations/wakeup_probe.md`
- 本报告

未改正式 Schema 规则、Template、案例、Kernel、Wiki、archive、前轮报告或精简前快照。报告完成后停写。
