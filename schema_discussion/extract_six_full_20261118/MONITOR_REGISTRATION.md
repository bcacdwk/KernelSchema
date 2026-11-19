# Monitor 登记锁竞争修复

2026-11-18。根据 [MONITOR_LOCK.md](MONITOR_LOCK.md) 已确认的短时锁竞争，前台 ack/绑定 scan 的登记适合复用现有 `Store.control_transaction()`。失败的单次 `LOCK_NB` 不再直接结束这一登记请求，避免正常短竞争让本次提醒无法登记、随后按原 30 分钟规则补送。本次没有操作 live runtime 或执行线上 ack/scan，也没有重启 clock 或 Runner。

[wakeup.py](../../schema_extract/tools/wakeup.py) 仅替换四处取锁调用：`Clock.ack()`，以及 `Clock.scan()` 的 claim、commit、异常清理。现有 `control_transaction()` 只捕获事务进入时的 `SupervisorBusy`，释放失败的事务后在锁外等待 0.05 秒，再尝试取锁；事务正文不被重放。线程身份、提醒 ID、代际、到期和发送/ack 前提、scan owner/claim 校验、次数与频率均保留。无参数的手工 `monitor scan` 不在本次修改范围。

每次成功取锁仍从当前 ledger 读取并校验：scan 的观察工作继续在两次事务之间执行，commit 继续核对 generation/thread、reminder ID、claim ID。等待期间 stop 或 stop/start 已经提交时，旧请求仍被拒绝或返回 `scan_invalidated`。身份拒绝、重复 ack、`scan_busy`、失效以及事务正文异常均不会转为锁重试；异常清理也只移除原匹配 claim。

这是对现有等待方式的局部复用：没有新增 timeout、重试框架、后台恢复或热补丁。若 supervisor 锁持续不释放，该前台 CLI 仍会等待，可通过中断终止；不提供等待时间上界。等待时不持有 supervisor 锁。已有证据显示本次是频繁短竞争而非持续死锁，因此该边界可接受。daemon 的 tick/observe、clock start/stop，以及人为 pause/cancel/resume 的取锁和即时 busy 语义均未改变。

隔离验证通过：`python -B -m unittest schema_extract.tools.tests.test_wakeup_retry.WakeupRetryTests -v`（10 项）及 `schema_extract.tools.tests.test_monitor_evidence.CancelDeliveryTests.test_busy_does_not_register_or_signal_cancel`（1 项）；`git diff --check` 通过。测试使用临时 state、虚拟时间、假身份与假 queue，并对真实 flock 竞争后的等待点释放锁或提交 stop/start：覆盖 ack、scan claim/commit/错误清理、失效后的无写入、观察不重复、重复 scan 复用，以及 tick/stop/cancel 即时 busy。原补送次数、时间间隔及晚到回执测试也通过；未运行模型或整批提取。

新的前台 monitor CLI 请求会加载当前源代码并受益。现有 daemon 已加载代码未被热改；六个冻结 Runner 仍使用 `b8a1d5e368bf0e423081110ffc748945097a503b7eb2fd11fac6355650a58d8a`，未修改或重启。此前 maintenance 空轮询修复仍只适用于未来使用新代码的运行。本报告不重复或代办已经完成的 check10 ack/scan，也不改变其 next_due。
