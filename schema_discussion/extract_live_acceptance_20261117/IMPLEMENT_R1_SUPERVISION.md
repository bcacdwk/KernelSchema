# 第一轮 B：监督与控制实施

2026-11-18。实施基于 `f48823b` 及保留 WIP；未修改活跃 MD/prompt、固定 schema、案例、Kernel 或旧 run，未启动真实 queue/monitor/模型，未提交推送。

## 已实施

- `wakeup.py` 的真实 queue 使用一份在途 `Popen`，stdout/stderr 写 `runtime/clock/<check>-<attempt>.stdout/.stderr`。每 tick 非阻塞 poll；`sending` 未真实结清时 `_next_send_at()` 返回 `None`，包括已 ack/scan 后移入 history 的投递，因此下一提醒也不能重叠启动。记录实际进程身份、已观察后代、退出及结果；仅真正退出/后代确认后采纳 bound thread/message UUID。原有限补送次数与提醒、观察、心跳节奏保留，没有 queue/清理/stop 时间杀停。
- clock 心跳先单独提交，观察继续使用轻量 summary/attempt 证据，不调用恢复深核验。迟到结果只更新原 attempt/history/check 的事实，不撤销 ack、scan 或新 generation。未能记录身份的在途投递保留 unknown，禁止据此启动第二份。
- `Clock.stop()` 先持久停用 generation，然后只清理核准 clock/queue，非阻塞返回 `committed`、`stop_status=pending|confirmed`、`termination`。不取消 Extract；身份观察失败、锁/身份不一致均不当作死亡。`/proc`、进程匹配与信号全部使用 A 的 `processes.py`，wakeup 的独立 Linux/ps/fallback 身份代码已删除。
- 删除 monitor 的 `configure_auto_restart`、`authorized_restarts`、clock binding 自动恢复授权及 CLI `--auto-restart-limit`。保留人工 scope 恢复的 `retries`、`starts`、failure ID 幂等及 pause/cancel/hold 优先。monitor 不再显示 deadline/overdue 或给补时建议；调用次数与一次性决定分类名改为 `call_limit_or_single_decision`。
- `common.SupervisorBusy(ExtractError)` 表示本次控制事务未取得锁。`Store.initialize()` 和 `Store.supervisor()` 的 flock 均立即 `LOCK_NB` 尝试；不按年龄破锁。clock tick 返回 `status=busy, committed=False`，后续 tick 可重试。CLI 锁忙不再表示控制成功。
- `Store.request_control()` 在请求和 ledger 都提交后返回 `committed=True`。同一当前 frozen binding 下重复 cancel 复用 request ID，继续观察同一交付；不是反复创建新请求。新增 `monitor.deliver_cancel(store, request)`：先登记交付身份，再独立于 Runner 写锁 KILL 核准 Runner，校验并交付已登记 attempt 的 gate 后代，保留 gate 锚。仅在 Runner 真正退出后，才按已捕获的归属树处理非 gate 辅助身份；未核准的 gate/进程保持 pending。setup 不被当作未启动，必须有 gate_reaped/launch_not_started 真实证据。未改 run state、session、fence 或验收记录。
- CLI cancel 不再调用 `run()` 或等待冻结执行器的深核验来交付控制；它沿已有 state/input-lock 指纹检查、请求 binding 与 attempt/invocation/launch/process 身份控制已拥有的进程，不推进 run 或放宽任何恢复/接收验收。返回 `{request, delivery}`；只有 `delivery.confirmed` 为真时 exit 0，pending exit 1。交付阶段锁忙仍明确请求已提交、交付待确认。
- `reserve_launch()` 对未确认的取消交付拒绝新启动；实际进程退出与恢复状态对账是下一轮集成要求，不能因释放 writer lock 就直接重派。
- 经 supervisor 同意，顺手删除 Store 中已无用途的 `validate_max_output_tokens` import 和单行 validation，避免已删除 backend API 阻断本轮独立 Store/clock 测试；其余输出字段由第二轮删除。

## 新接口和持久字段

`Clock.start(thread, run_ids=None, *, codex_binary=None, launch=True)` 不再接受自动重启参数。`monitor.request_restart(store, run_id, scope, repo)` 与 `restart_evidence(store, state, scope, *, table=None)` 不再接受 `automatic` 或 `clock_binding`；人工证据与控制消费仍必须核准。

supervisor 的默认/必需字段删除 `auto_restart`。`controls[run_id]` 增加：

- `binding={input_fingerprint, executor_digest}`：当前手动控制绑定的冻结 run。
- `deliveries[request_id]`：原 cancel 的交付事实，包含 `status`、`confirmed`、`requested_at`、`observed_at`、`runner`、`runner_descendants`、`targets[attempt_id]`、`attempts[attempt_id]`、`actions`、`unknown`，可带 `observer_error`。
- `targets` 只含 gate/launch identity 和已知后代或正向 `not_dispatched`，在信号前持久提交；`attempts` 是实际终止观察，非成功模型结果。上一轮仍未确认的 targets 会继续复查，即便 Runner 已从 active/unsettled 中移除它们。
- 后续 pause/resume 的 ledger 更新保留 `deliveries`。signal 已发送不意味着 `confirmed`；如最终交付观察写锁忙，返回 `evidence_committed=False` 且 pending。无新服务或通用任务系统。

`deliver_cancel` 是可重复的单轮交付/确认接口。第一次计划写锁忙可能抛 `SupervisorBusy`；CLI 已区分“请求已提交，交付计划忙”。它不自动恢复任务、消费请求、清 session fence 或覆盖 run state。

## 第二轮 C 必须对齐

1. `runner._consume_scope_retries()` 删除 `restart_evidence(...automatic=request.get(...))` 的旧参数。`recovery.scoped_restart_evidence()` 删除 automatic 模式分支，保留原人工 scope 证据路径、合法来源、failure ID、次数限制和精确权限；Runner 内部普通有限错误重试独立保留。当前两者尚未整合，不能宣称人工 scope 完整通过。
2. Runner 所有 `with store.supervisor()` 的进入/控制边界、scope retry 提交、派发/发布及 finally（当前约 1201/1224/1316/1408/1444/1525/1552 行）要把 `SupervisorBusy` 当控制争用退让重试，不能变成模型失败/丢控制。保持锁内短事务，不等待模型、子进程或深检查。
3. 取消可先停 Runner，state 尚未消费 request，attempt result/exit 可能缺失或仍是 unconfirmed。必须用 A 的 gate `setup`、真实 cli-exit/descendants_reaped、fresh owned observation 对账，保留未结/事务/旧线程 fence；不得为了方便伪造 result 或 session release。
4. 已提交 cancel 的交付通过重复 `deliver_cancel` 才把 `deliveries[id].confirmed` 更新为真。**显式 resume 应先对已有 pending cancel 做交付/观察对账，再过 reserve_launch guard**，否则进程实际已退但旧 ledger pending 会永远阻挡恢复。确认后沿原 explicit-resume 规则消费旧 request 并恢复，不能自动恢复；新 pause/cancel/hold 仍优先。state 尚未标 cancelled 也必须正确处理这条路径。
5. Runner 持久 `runner.identity` 建议明确带 launch_id 关联；派发前检查持久 cancel，避免控制提交到 Runner KILL 之间新 gate 漏登记。当前交付核准输入/执行器、state Runner identity，以及 matching starts.launch identity；启动身份未知保持 pending。
6. `process_table(cwd=...)`、`terminate_owned(...cwd/grace...)` 等旧调用、Agent 新 pending 返回、预算/时间字段由 C 对齐。调用启动/未知 gate 永不以 setup、年龄或锁空闲冒充已退出。

## 第二轮 D 必须对齐

- CLI 其余时间/输出字段和 budget import、resume/add-calls 文案/API，以及活跃 prompt/README 的必要事实删除；`monitor scan` help 可从“不默认恢复”改为仅观察。
- audit 删除 `auto_restart`/automatic/clock_binding 旧投影，保留 retries/starts 和必要取消交付事实。
- `test_scoped_recovery` 纯 clock 自动恢复场景删除，人工 scope/控制优先和恢复反例保留并按 C 新 API 修改；其它旧调用不要做兼容层。
- 最终明确：cancel 直接 KILL 会放弃最后刷盘机会；pause 保持正常边界；monitor 不额外自动重启；queue 自身挂住时不保证同一通道还能提醒。活跃正文最小事实更改由 D 统一做。

## 离线验证

`python -m unittest schema_extract.tools.tests.test_wakeup_retry schema_extract.tools.tests.test_monitor_evidence`：15 项通过（约 0.6 秒）。覆盖真实投递接口的 fake Popen、虚拟长时间 sending/心跳/不重发、归档在途阻止第二 queue、迟到 ack/scan/stop、有限补送、初始化和控制锁忙、后台启动真实退出语义的 fake Popen、取消持久化/写锁独立、unknown 不信号、setup gate 保锚与真实结清。

改动生产文件 `py_compile`、限定文件 `git diff --check` 已通过。第一次定向测试被 Store 已删除 backend import 阻挡，该最终应删项经 supervisor 同意删除；随后测试失败的两处旧夹具时序/进程表 mock 已修正。未运行完整 suite、提取流程、真实模型或真实提醒闭环。Runner/recovery/外层 CLI 的跨轮集成验收仍待第二轮；本轮离线通过不代表全链路取消/恢复或在线接管已验证。
