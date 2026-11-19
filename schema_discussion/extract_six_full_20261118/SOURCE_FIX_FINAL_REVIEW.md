# 源修复集中收尾审阅

2026-11-19。审阅时 HEAD 为 `24d2b3d`；仅核对本批未提交源修复及必要相邻调用链。补正了 monitor 等锁后复用旧观察的问题，其余改动无需扩展或重写。

- 维护空轮询的只读快路径等价于原来的无请求返回。待处理请求仍在锁内重读并校验封印；派发与发布仍用最新 ledger 检查维护门禁。scoped resume 在清除拒因前保留反馈，去重方式与普通 resume 一致，不改变计数或批准状态。
- 回执修复仅接入已验证文件内的坐标选择错误及通用必填判断/证据漏项。身份、权限、固定来源及字节、产物语义仍沿原校验路径；修复草稿仍经过文件/字段/来源门禁，不能成为批准。有效的 reviewer `revise` / `blocked` 仍按原决定处理，维护与一次性裁决/签收不获得新循环。
- 长字段 warning 只改变首差异的展示；Contract 撞名诊断复用位置表，保持原唯一命名约束，不推断或改写 I/O 关系。维护提示中的 Version 责任与现有 `_erase_fields`、`_versions` 一致；未改字段、版本或规范化规则。

本轮在 [wakeup.py](../../schema_extract/tools/wakeup.py) 另作两处小修正：ack 的时间在实际取得锁后读取；scan 判断当前 claim 的 owner 时重新观察存活状态，不再使用等待前的进程快照。否则，等待期间新出现的活 owner 会被旧快照误判为已退出，后来的 scan 会抢占 claim。存活检查仅在已有 claim 时发生；实际 run 观察仍在事务外。

仍复用原 `control_transaction()`，只重试事务进入；正文异常不重放。取得锁后继续校验提醒、线程、stop/代际和 claim；其他人为控制与 daemon 的即时 busy 语义不变。持续不释放的 supervisor 锁仍没有等待时间上界，本轮没有新增超时或重试框架。

[定向测试](../../schema_extract/tools/tests/test_wakeup_retry.py) 先复现 ack 时间偏旧与活 claim 被抢占两项失败，再验证修正。以下 **12 项全部通过**，使用隔离 state、虚拟时间与假发送器，未启动模型或 daemon：

```sh
python3 -B -m unittest schema_extract.tools.tests.test_wakeup_retry.WakeupRetryTests schema_extract.tools.tests.test_monitor_evidence.CancelDeliveryTests.test_busy_does_not_register_or_signal_cancel -v
```

此前验证按对应记录引用，未重复运行大套测试：[维护轮询](MONITOR_LOCK.md)、[scoped 反馈](RECEIPT_RECOVERY_0810.md)、[回执修复 30 项](RECEIPT_REPAIR_POLICY.md)、[模板 68 项](NORMALIZER_WARNING.md)、[I/O 诊断](IO_NAME_DIAGNOSTIC.md)、[版本提示](MAINTENANCE_VERSION_GUIDANCE.md)。`git diff --check` 通过。

本轮仅修改当前 monitor 源码、必要测试及本报告；未操作 `.schema_extract/v2`、冻结 executor/rules、在途或已发布产物、进程和调用计数，未提交或推送。各项修复供后续新 run 使用；新的前台 monitor CLI 请求可直接使用当前源码，已有 daemon 与本批 `b8a1d5e…` 冻结 Runner 不会热更新。未重跑提取，也未以源工具审阅代替算子语义验收。
