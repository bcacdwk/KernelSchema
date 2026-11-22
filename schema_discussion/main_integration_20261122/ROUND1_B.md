# 第一轮 B：控制提交事实与观察

W1/X-08 与 W4/X-09～12 已实施并完成共同最终源码树验收。唯一需求依据为已批准 FINAL_PLAN、COVERAGE、FINAL_REVIEW；执行与归属依 ARBITRATION_PLAN。没有扩展控制协议、历史迁移或真实运行范围。

## 实际改动

- `monitor.py`、`cli.py`、`store.py`、`wakeup.py` 保留本操作已持久登记的 request、registration、launch、hold 或 clock generation。后续 busy、spawn、signal 或观察写回失败分别报告；已确认提交为 true，明确未取得事务锁为 false，提交确认丢失为 null。`control_operation/control_error` 仅携带本次调用的返回事实，不写新日志，不以全局 ledger revision 推断本操作，也不重放整个命令、Popen、signal 或 apply。
- start/restart、maintain 后启动、extract 单项/批量登记后启动、hold release 后启动、resume/export 预约后运行、clock 登记后 spawn/signal 均保留原身份。cancel 沿原 request/delivery 分离。`inspect` 只读投影控制 pending/recent、按 scope 的 recovery pending/recent、registration 与 launch。较新 pause/cancel/hold 和被替代 launch 继续阻止旧预约 claim。
- `read_summary` 区分 `observation.cache.status` 与本次样本状态。stale/missing/invalid 缓存触发至多一次 state 读取；返回实际样本 revision、采样时间与含 ctime 的前后文件标记。读中变动标 changed，无法观测标 unknown，`deep_verified=false`，不写缓存或 state。
- `record_scan` 独自产生当前 anomalies，按同一 run 的最新观察汇总；旧 rows 和 superseded IDs 保留并标 historical。人读输出不把历史恢复建议作为当前行动；Clock 直接使用同一结果的 anomaly_count。较新 ready 后迟到旧失败不使 latest、当前计数或行动建议倒退；迟到旧 ready 同样不能掩盖较新失败。
- Clock 的 disabled→enabled 显式新代际重置 initial scan/due，初扫完成后排新的 due；同 enabled 代际 daemon 恢复保留提醒 ID、ack、scan claim 和原 due。旧发送未确认结清的 fence 与旧代际确认限制均保留。
- README 仅补输出叶目录必须不存在，以及上述缓存/样本、提交/投递、未知提交的字段解释。

## 验收

W1 按双方冻结 revision 2 完成，B 的 38 项通过证据为 [w1_b.json](w1_b.json)，根随后提交 `1a221e440f6f64cb83c4179f1cb9d22b92c3e812` 并释放 [W1_GATE](W1_GATE.json)。W4 未提前实施。

最终共同 [ROUND1_FROZEN](ROUND1_FROZEN.json) SHA256 为 `0eed4276bfe21214f7b4f3955e3a17e91a4a26ce9fd9c2a64976fea38fb5fce4`。在该树上以 `/opt/venv/python3.13.12/bin/python -B -m unittest -v` 运行以下对象，**46 项全部通过，exit 0**，包含 W1 回归：

- `schema_extract.tools.tests.test_cli_recovery.ControlCommitFactsTests`
- `schema_extract.tools.tests.test_monitor_evidence`
- `schema_extract.tools.tests.test_wakeup_retry.WakeupRetryTests`
- `schema_extract.tools.tests.test_wakeup_retry.ObservationTimingTests`
- `schema_extract.tools.tests.test_maintenance_polling`

具体命令、解释器、实际 import 来源、输入 SHA、测试后输入未变核对、退出码及日志 SHA 见 [final_verification.json](round1_b/final_verification.json)；逐项断言日志见 [final_unittest.log](round1_b/final_unittest.log)，需求到用例映射见 [coverage.json](round1_b/coverage.json)。精确产品/测试与轻量证据文件清单见 [final_manifest.json](round1_b/final_manifest.json)。

开发验证中两项旧 clock 测试仍预期显式 stop/start 沿用旧 due，或把旧 scan 的锁注入再次作用于新初扫；已按本次新代际契约修正等待时间和夹具绑定。失败和修正范围记录于 final_code_ready.json；最终共同冻结后没有失败，也没有产品或测试改写。

## 边界

所有验收使用自有临时 Store、假时钟、fake Popen/sender/signal；没有真实模型、GPU、Extract 提取或 monitor daemon，也没有发出外部消息。没有 Git 写动作、AKA/AKA_experiment/archive 访问、home 递归搜索或表外产品修改。测试未读取或写入原 `.schema_extract/v2`，不修 clock generation 10/disabled 或旧 daemon.status 字段，不重开 12 个已关闭 published run。旧账、receipt、原稿、固定来源提交和规则未改写；原运行态整体摘要由根任务统一核对。

这是局部 CPU 行为验收，不声称真实模型效率、GPU 正确性、完整产品全套测试或新的 Runner P6 认证。
