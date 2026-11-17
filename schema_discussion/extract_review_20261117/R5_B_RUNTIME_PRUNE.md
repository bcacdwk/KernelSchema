# R5 B：执行、会话与恢复精简

完整阅读六个限定模块与 R1 B、R3 A、R4 B 结论；以 `/tmp/schema-extract-pre-prune-ego_8hhb` 只读快照区分本轮和前轮改动。先向主代理列明冗余及等价依据，再执行小幅精简，没有为减行数重构流程。

## 实际变化

- `recovery.scoped_restart_evidence` 删除紧接 `verify_execution(..., attempt=attempt)` 后的四项重复核对：directory、workspace、run_id、input_fingerprint。`runtime._attempt_identity` 已对同一份 attempt 核对相同值；仍保留请求的 attempt_id，以及当前 scope、generation、mode、role 等恢复专属约束。
- `recovery.handoff_session` 直接使用 `parse_events` 返回的 `session_id`。解析器已在且仅在 `trace.session_ids` 唯一时提供该值，删除重复的长度判断和索引；解析和原生成功判定未改。
- `backends.NativeEvents.finish` 删除 `usage["issues"]` 的无效初始赋值；该字段返回前仍无条件写入最终排序、去重结果。
- `observation._locate` 合并连续、同错误的索引路径拒绝条件；闭合判定复用已从同一不可变记录集合筛出并核对的 `endings`，不再重新筛选。
- `processes` 两处注释改为描述现行 ps 身份与 Linux boot/start token，去掉将其写成历史兼容的措辞；进程逻辑未改。

相对精简前快照，四个产品文件合计净减 5 行。`agent.py`、`runtime.py` 和所有测试保持原样。

## 保留的必要能力

启动前控制、放行后执行状态、读取结果时后代闭合、恢复时终止证明是不同时间点的检查，不能合并为一次判断。继续保留真实进程表观察、gate wait/ECHILD 证据、未确认时 fence，以及确证未执行时保留旧有效会话、已执行中断时退休和 exec_failed 返回 125 的前轮修复。没有恢复自建隔离。

默认 Claude Code `qwen3.8-max[1m]`、双角色 `max`、显式 Codex、角色权限、原生唯一成功/退出/session/schema/nonce、精确 scope resume、已有预算及失败用量不重置均未调整。Claude 增量与会话累计口径、Codex 根会话范围、未知不补零保持。只读审计和 accepted 回执验证不在本轮写入范围。

## 实际检查

使用 `PATH=/usr/bin:/bin /opt/venv/python3.13.12/bin/python3 -B`，执行前断言 `claude`、`codex` 不在 PATH；CLI 调用仅使用测试生成的绝对路径 fake 文件。以下既有用例共 **7 项通过，19.193 秒**：

1. `CurrentScopeRecoveryTests.test_actual_transient_exit_recovers_only_its_scope_without_new_credit`：真实 fake EX_TEMPFAIL、同 scope 恢复、兄弟进展不改变授权、预算不退款和 pause 屏障。
2. `CurrentRecoveryTests.test_missing_result_retires_private_binding_only_after_confirmed_exit`：缺结果时从事件取唯一旧会话，终止确认后退休，未知用量仍为 null。
3. `CurrentRecoveryTests.test_unstarted_gate_preserves_previous_session_even_when_result_is_missing`：放行前取消/exec_failed、结果存在/丢失均保留旧 UUID，后续 fake resume 保留历史。
4. `BackendRuntimeTests.test_native_success_schema_nonce_session_and_complete_trace_are_independent`：坏 schema/nonce、多 session、原生错误、缺失/重复 terminal、截断事件及非零退出继续拒绝成功；终止未知继续 fence。
5. `UsageObservationTests.test_bounded_native_delegation_preserves_root_delta_and_audit_binding`：闭合/开放 turn、标记异常、原生范围、根用量与摘要绑定。
6. `UsageObservationTests.test_native_index_rejects_paths_identities_and_other_scopes`：错误路径、UUID、symlink、home/scope 等仍拒绝观察。
7. `UsageObservationTests.test_claude_resume_uses_native_delta_and_never_sums_session_totals`：增量/累计分列、重复 attempt 去重及未知值口径。

检查测试临时根关联进程，剩余 **0**；六个限定 Python 模块 `ast.parse` 和限定 `git diff --check` 通过。没有新增或删减回归测试，没有真实模型、queue、GPU、完整 P1–P6、发布或 Git commit。

无已确认但未修的问题，无跨界待改项。此次小检查不代表真实 CLI 或完整提取已验证。本代理只修改 `backends.py`、`processes.py`、`recovery.py`、`observation.py` 及本报告；完成后停写。
