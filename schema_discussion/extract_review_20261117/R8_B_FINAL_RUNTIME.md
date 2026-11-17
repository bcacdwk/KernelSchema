# R8 B：执行、控制与提示最终验收

**签收：本范围修复后通过。** 独立对照当前代码、Git diff 与 `/tmp/schema-extract-pre-prune-ego_8hhb` 精简前快照，检查实际调用链与消费者，未仅复述前轮报告。最终发现并处理了交接未实际交付、提醒审计投影和测试旧签名三个具体遗漏；没有恢复 probe、inspect 启动入口或其他已同意删除的功能。

## 实质问题与处理

1. **恢复调用未执行，旧交接却已被占用。** Store 在准备时绑定 `next_attempt_id`。A 发现最终派发门因 pause 拒发后未释放绑定；B 进一步用真实 fake CLI 复现：前次结果丢失形成交接，下一调用在 gate 放行前取消，结果为 `cancelled / empty`，`handoff_session` 返回 `not_started`，旧交接仍被占用，后续无待交接项。live `exec_failed` 还有一层：结果为 `cli_error / 125 / empty`，原 `_recover_call` 不进入交接处理。
   - B 在 recovery 增加小函数 `release_unstarted_handoff`，只解除绑定当前未执行 attempt 的接收指针；setup、prepared、gate 未执行证明复用此处理。A 的最终拒发门调用同一函数，并让携带恢复交接但没有新 session 的普通失败进入现有 `handoff_session`。
   - 已执行中断仍须确认终止并退休 session；未知终止仍 fence。没有把缺结果或日志沉默当作未执行证明。旧交接、checkpoint、失败和调用账目保留，不增加状态或退款。
2. **停止后的迟到消息 ID 未进入审计包。** 实证场景为发送过程中 stop，随后 fake queue 返回 message ID：history 正确保留线程与消息，但无 scan 的提醒不会进入审计 `records`，`reminder_timing` 又丢弃 `thread_id/message_id`，归档只保存该投影，因而失去这次投递的身份。
   - 只在现有投影保留两个已有字段。queued 未 ack、已 ack 未 scan、完成 scan、stop 前未 ack 仍分别如实呈现；保留消息身份不产生 ack、scan 或恢复授权。
   - 扩展现有迟到回执用例，对 scan 与 stop 两支核对真正进入审计的线程和消息 ID；不恢复冗余存储副本。
3. **精简审计函数参数时漏改直接测试消费者。** `_verify_attempt` 删除 `store` 后，`test_current_recovery` 五处、`test_usage_observation` 两处仍传旧实参，实际会发生参数数量错误。已机械更新全部调用。
   - 用量投影单元测试原本利用 `store=None` 跳过原生结果结清；当前产品不再有此分支。该合成投影测试明确替身化单独的 `verified_result` 边界，仍执行实际投影、摘要、stream 和 invocation 绑定及篡改拒绝；真实原生结清由本轮四项 fake CLI 审计/恢复用例独立验证，没有恢复产品绕过开关。
   - 通过快照 AST 对比列出全部删参、删函数，再检索当前产品和测试消费者：artifacts 的 decision 参数、audit 参数、CLI 冻结入口、wakeup 参数，以及已删除的薄包装、scaffold、reader、Store.handoff 均无剩余悬空调用。A 另核对了 artifacts 的 13 处消费者。

## 精简为何保留必要能力

- 默认配置、CLI 构造和环境仍一致指向 Claude Code `qwen3.8-max[1m]`，两角色 `max`；显式 Codex 在创建前要求模型和两角色 effort，并使用自身原生协议。稳定 scope、精确 UUID、独立角色/操作、原生唯一终结、真实退出、schema/nonce 和 gate 后代结清仍分别核验。
- 删除恢复中的重复身份判断，依赖的是同次 `verify_execution(..., attempt=...)` 的完整绑定；当前 scope、generation、role、实例和实际来源、剩余预算、失败证据仍另核。pause/cancel/hold/维护屏障及持久有限恢复授权未合并掉。
- 已接受 receipt 仍必须有匹配 validation；未接受但有绑定拒绝/repair 的原稿只能归档为 `partial / evidence_only`。审计不写运行状态、不产生批准或恢复资格。失败、取消与重复读取不重置用量，Claude 增量/主循环/累计和 Codex 根观测仍分列，未知不补零。
- `inspect` 现在只有观察路径；排队任务仍可通过 `resume` 执行，新建后台任务仍经登记、预留和 `_background` 冻结入口。启动子进程实际早退时保留 `launch_failed`，下一次不因活 launcher 复用旧启动；已 claimed 的结果不被降级。
- 独立 probe 及其 prompt、kind、verified、last_check 副本确实已无独立用途。正式 reminder 自身仍完成绑定线程的 ack、显式 scan、有限重试、停止失效、迟到 ID 补记和有界历史审计。ack、完成扫描和自动恢复授权继续分开。
- 阶段 `handoff.md` 副本删除后，当前批准和知识去向均指向摘要核对的 accepted receipt，完整 `handoff` 仍随材料可读；中断恢复使用单独的 `recovery_handoff`。已确认并修复的提前占用问题不靠恢复副本解决。
- 新建输出须不存在，CLI 登记和创建预检与发布冲突规则、README 一致；已登记运行的复用仍走原身份检查。没有恢复 `.gitkeep` 接管。
- 当前 prompt 仍完整提供共同说明、角色、阶段、操作、响应协议、固定规范及动态身份。收短的普通 worker/reviewer 提示保留决策和当前动作，不是空壳；权限、源码/产物证据、四项判断、计划回传与 P6 定点授权仍有明确落点。固定语义继续在 Markdown，动态事实、机械 schema 和错误诊断留在程序。未发现需补回的无意义防御提示。

## 实际验证

统一使用 `PATH=/usr/bin:/bin /opt/venv/python3.13.12/bin/python3 -B`，先断言真实 claude/codex 不在 PATH；模型调用仅用临时绝对路径 fake CLI。提醒使用 fake sender，真实 `_send` 设为失败。

最终定向测试 **9 项通过，29.245 秒**：

- `CurrentRecoveryTests` 的 `test_native_result_recovery_requires_real_exit_nonce_schema`、`test_lost_gate_without_closed_descendant_evidence_stays_fenced`、`test_setup_failure_audit_preserves_missing_stream_and_rejects_forged_proof`、`test_setup_failure_keeps_the_previous_ready_session_and_usage_is_read_only`。
- 扩展 `CurrentScopeRecoveryTests.test_actual_transient_exit_recovers_only_its_scope_without_new_credit`：真实 EX_TEMPFAIL 75 → 原授权 scope 恢复 → gate exec_failed/125 → 下一次 prepare 仍收到原交接，且已用调用继续计费。
- `UsageObservationTests.test_bounded_native_delegation_preserves_root_delta_and_audit_binding`。
- 扩展 `WakeupRetryTests.test_ack_scan_and_stop_win_before_send_and_before_late_receipts`。
- `AuditReceiptRecoveryTests.test_rejected_receipts_archive_without_claiming_artifact_validation`：拒绝/repair 可保留部分证据，accepted 缺 validation 仍不封存，state bytes 不变。
- `MonitorLaunchTests.test_exited_background_before_claim_does_not_reuse_living_launcher`：真实 `/usr/bin/false` 子进程已 wait，登记可重试。

另执行以下小检查，没有再跑整组：

- 对修复前反例复核 setup 失败、gate 放行前取消及结果丢失三支，均保留原待交接项，不生成虚假新会话。
- inspect 不创建不存在的状态目录；删除的两项 CLI 参数拒绝；resume/background 入口仍可解析。queued、acked、scanned、stopped-before-ack 审计状态及旧 generation 的拒绝一致。
- 30 份当前入口、说明和 prompt 的本地导航无缺失文件；此检查不代替锚点和语义审阅。现行树未发现 probe、删去 CLI 选项或 handoff.md 的消费者。
- Extract 产品 Python AST 与本代理修改文件的 `git diff --check` 通过。
- 测试根 `/tmp/schema-extract-r8-b-tests-fcge5xr8` 及交接小检查关联子进程均 **0**，临时目录已清理。

A 在最终共同 helper 版本上另跑准备后 pause→resume 交接回归，**1 项通过，3.447 秒**，本报告不把它计入 B 的 9 项。

## 文件、限制与未决

B 实际修改：`schema_extract/tools/audit.py`、`recovery.py`、`tests/test_current_recovery.py`、`tests/test_usage_observation.py`、`tests/test_wakeup_retry.py` 及本报告。Runner 的最终门与普通失败接入由 A 修改，详见 R8 A。

本范围没有已确认而未处理的问题。结论限于代码、材料与小型离线行为；没有真实模型、真实 queue 投递、GPU、完整 P1–P6 或正式发布验证。未改正式 Schema 字段、Template/例子、Kernel、Wiki、archive、既有报告或精简前快照，没有提交 commit。报告完成后停写。
