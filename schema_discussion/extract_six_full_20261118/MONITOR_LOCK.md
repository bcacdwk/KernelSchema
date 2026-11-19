# 六算子巡检锁竞争调查

调查时间：2026-11-18 07:04–07:09 UTC。基线 `24d2b3d`，运行身份见 [RUNS.json](RUNS.json)。本次只修改当前源工具和离线测试；未改运行状态、冻结快照、schema 字段或提取产物，未执行任何线上 ack/scan、进程控制、提交或推送。

## 结论与证据

这是多个 Runner 反复执行空维护轮询造成的短时写锁竞争，没有发现 supervisor 锁持续被某一进程占用。提醒送达本身没有在锁内等待模型、queue CLI 或进程收尾。两次历史 ack 失败时没有保留持锁进程的现场记录，因此不能给那两个瞬间指定某一个 PID；当前实测和代码共同定位了持续制造竞争的来源。

- `runner.py` 的模型等待循环每次 `events.get(timeout=0.1)` 超时后再次调用 `maintenance.consume_requests()`。原实现无条件进入 `store.control_transaction()`，即使根本没有维护请求也取得 supervisor 写锁、读取和深拷贝整个 ledger 后才返回。六个活跃 Runner 因此可产生约每秒 60 次无用锁申请。
- 07:06:51 读取的 ledger 中 `maintenance_requests={}`、`controls={}`、`active_hold=null`。ledger 约 27.6 KB；一次只读测量中读取中位耗时约 0.588 ms、深拷贝及比较约 0.652 ms，另有文件打开和锁操作开销。
- 07:06:22–07:06:28，对已有 supervisor.lock 进行 440 次非阻塞探测，其中 106 次忙、334 次可立即取得后释放。没有执行控制事务或写文件。忙时查看打开该文件的进程，候选持有者不断在下面六个 Runner 之间轮换，间隙持续恢复为空闲。约 24% 是本次探测的忙命中比例，受采样与 FD 观察开销影响，不能当作精确时间占比或生产统计。
- 本环境 `/proc/locks` 未列出已确认仍被持有的 Runner/clock 锁，不能仅凭缺行判定锁空闲。结论使用实际 `LOCK_NB` 探测及 `/proc/PID/fd`，FD 打开记录也只作为当时持有者的候选证据。

| 算子 | Runner PID | 07:06:51 阶段 |
| --- | --- | --- |
| fp8_scaled_grouped_gemm | 1563371 | P1 worker，1 次调用 |
| nvfp4_fused_moe | 1563568 | P1 worker，1 次调用 |
| fmha_mxfp8_qkv | 1563811 | P1 worker，1 次调用 |
| mla_expanded_fp8 | 1564062 | P1 worker，1 次调用 |
| dual_cache_latent_bf16 | 1565092 | P1 reviewer，2 次调用 |
| gdn_prefill_bf16_state_bf16 | 1564553 | P1 worker，1 次调用 |

六个 Runner 及对应 gate/model 进程仍存活；Sparse 已自然从 worker 进入 reviewer，其余模型事件文件仍有近期更新。没有将日志沉默作为退出或死锁证据。Clock PID `1565326` 存活，daemon 为 `running`，每 30 秒观察继续更新、最近观察无异常；它持有的是单实例 `monitor-clock.lock`，未发现 supervisor 锁泄漏或继承后不释放的证据。

## 提醒时间与锁范围

`check-1-3-29f45ffdd4423dbf64861d07`，线程 `01a1c900-a487-76a1-92ee-990cbff08276`：

| 事件 | UTC 时间 |
| --- | --- |
| 到期 | 06:33:17.860193 |
| 第一次 queue 启动 / 回执完成 | 06:33:18.059571 / 06:33:33.187344 |
| 第二次 queue 启动 / 回执完成 | 07:03:33.669562 / 07:03:48.823926 |

两次 queue 均 `exit_code=0`、`status=queued`。`_next_send_at()` 对已 queued 但未 ack 的提醒使用 `finished_at + ATTENTION_SECONDS`，其中 `ATTENTION_SECONDS=1800`；所以第二次投递符合规定的 30 分钟补送间隔，并非锁阻塞 30 分钟。失败的 ack 没有提交，因此它不能推进巡检或取消补送。

代码核对：`Clock._send()` 的 `Popen` 在 supervisor 事务外；`_poll_send()` 的进程观察、退出确认、收尾及回执文件读取也在事务外，随后 `_record_send()` 才短暂持锁记账。`observe()` 在锁外执行 `monitor.inspect_runs()`。`Store.supervisor()` 则对控制入口使用一次 `LOCK_NB`，竞争即报 `SupervisorBusy/committed=false`。这解释了为什么正常短锁竞争也能让前台 ack 立即退出；未改成长超时或通用重试。

## 源代码修复

[maintenance.py](../../schema_extract/tools/maintenance.py) 的 `consume_requests()` 先只读检查 `barrier(store, state)`；没有 `requested/draining` 请求时直接返回。存在待处理请求时仍取得原控制锁，并在锁内重新读取当前请求、检查状态、校验封印和冻结身份。

这不是缓存授权：新的请求可能在初次只读检查之后到达，但 Runner 在最后派发事务内仍用该事务的最新 ledger 执行 `maintenance.barrier(..., ledger=ledger)`；发布门禁也保留。新请求因此仍可阻止新调用/发布，下个循环继续消费。没有新增状态格式、配置参数、重试框架或后台进程。

## 离线验证

新增 [test_maintenance_polling.py](../../schema_extract/tools/tests/test_maintenance_polling.py)，3 项通过：另一个控制事务持锁时，无请求及已应用请求的空轮询仍完成且不改状态；取锁前请求状态变化时重新核对；待处理请求仍要求原封印身份。首次运行的封印测试夹具缺 `created_at`，补齐夹具后单独重跑 3 项全部通过。

另两项现有定向测试已通过：`WakeupRetryTests.test_busy_control_lock_does_not_ack_or_block_the_tick`，以及 `MaintenanceReceiptTests.test_resume_corrects_range_and_shape_in_exact_session_without_refunding_or_accepting_old_approve`。后者使用隔离目录中的本地假 CLI，验证维护审阅与原次数/回执边界；没有启动真实模型或整批流程。`git diff --check` 通过。

## 现有运行与恢复

本批六个 Runner 继续使用原冻结 executor `2.5.1`、digest `b8a1d5e368bf0e423081110ffc748945097a503b7eb2fd11fac6355650a58d8a`。当前源代码修复不会热替换其已加载代码，也不会改变已冻结的后续阶段。无需、也未请求停止、暂停、取消、恢复或重建任何 Runner 或 clock。现有批次仍可能遇到原空轮询引起的短时竞争。

现成的恢复方式由主 Supervisor 显式操作：先用 `monitor status` 核对 generation、线程、当前 reminder 和 ack 状态；未确认且仍为同一到期提醒时，重放一次该提醒的原 `monitor ack`。只有 `accepted=true` 才执行原绑定 `monitor scan --wake ID --thread THREAD`；若已确认则直接续同一 scan，不重复 ack。忙表示事务未取得，本次不伪造确认、不扫描绕过、不改已用次数，也不采用无限重试。该方式沿用 README 的“恢复由 Supervisor 决定”和现有 ack/scan 门禁，不需要等待新的 queue 或重启 monitor。

最后只读记录（07:09:04 UTC）：主 Supervisor 已在 07:08:19.254295 为上述 check3 提交真实 ack；`last_check_number` 仍为 2，check3 尚未提交 scan。因此当时的待续动作仅是原 check3 的绑定 scan，无需再次 ack。已确认状态会令 `_next_send_at()` 返回空，原预计 07:33:48 的下一次补送不再适用。本修复代理未执行该恢复操作；后续结果以主代理的巡检记录和 authoritative ledger 为准。

主 Supervisor 随后回报恢复完成：首次原 scan 仍遇 busy，先用 status 确认 `acknowledged_unfinished`，再续同一 scan 成功，期间没有重复 ack。Check3 为 `committed=true`，完成时间 `1794985751.0947406`（07:09:11.094740 UTC），下次到期 `1794987551.0947406`（07:39:11.094740 UTC）。六个 Runner 未动，继续原冻结 executor；此次不再需要进程恢复操作。
