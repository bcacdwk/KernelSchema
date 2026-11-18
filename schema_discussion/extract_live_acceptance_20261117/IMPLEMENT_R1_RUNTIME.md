# 第一轮 A：运行时实施记录

2026-11-18。基于当前工作树完成分配范围内的修改；未改活跃 MD/prompt、案例、旧 run、主机配置，未提交或推送。没有真实模型调用、monitor 启动或提取全流程。

## 已实施与接口

- `ClaudeBackend.environment(home, effort)` 无条件写 `CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000`。默认常量只在 backend 定义，值仍进入 `configuration_digest`；Codex 不继承该设置。删除 `validate_max_output_tokens` 和本轮文件的额度透传/重复身份字段，会话配置一致性检查保留。
- `AgentCLI.run(*, attempt, prompt, schema, on_process=..., cancel_event=None)` 无 timeout 参数。耗时只统计。两个输出流直接写 `events.jsonl` / `stderr.txt`，不再有排空线程或 join。
- 版本查询进入同一 gate 的 setup，使用 `cli-tmp/version.txt`、`version-stderr.txt`、`version.json`。父侧轮询退出/取消；版本、可执行文件及会话核验完成后才放行模型。健康模型观察失败只写 `invocation.process_observation_error` 并保留最后可信 process 记录，不触发终止。
- `process_identity(pid)` 是单 PID 的唯一公共身份读取，返回 `{pid, ppid, pgid, stat, start_token}` 或真实消失时 `None`；`start_token=linux:<boot UUID>:<start ticks>`。权限/解析失败抛 `OSError`。`process_table()` 枚举 `/proc`，单 PID 消失跳过，其他失败不返回不完整表。无 ps/lstart/started 兼容及 cwd 参数。
- `process_identity_matches(identity, current)` / `process_alive(identity, table)` 保留三值：真、假、未知。`identity_key` 只使用 PID/start_token。
- `owned_processes(identity, known=(), *, table=None)`、`observe_owned(identity, known=(), *, table=None)` 可复用调用方可信快照。后者只观察，返回 `confirmed/actions/survivors/descendants`，必要时附 `reason`。`descendants` 是可跨次保留的已观察身份集合，可能包含 anchor；不是仅本次仍活着的进程。
- `signal_process(identity, sig=SIGKILL)` 发信号前重新核身份，保留 pidfd；返回真只证明已发信号。
- `terminate_owned(identity, known=(), *, include_anchor=False)` 一轮核准 KILL 后立即观察，没有 grace、等待或时间升级。`actions` 含 `pid/start_token/signal`。默认保留 gate；只有 clock/queue 等非 invocation gate 的根才用 `include_anchor=True`。观察失败/未退出如实 pending。接口已与第一轮 B 对齐。
- SQLite 使用 `timeout=0` 明确关闭 busy wait，沿用索引失败后的 UUID/日期文件查找。
- `evaluate(..., selected=None, cancel_event=None)` 删除 `call_seconds` / 输出额度，保留每 case 一次及统计；向 Agent 传同一取消事件，并在 case 边界停止。
- Template `local_repository_errors` 的实际 Git 检查删除 10 秒和 `TimeoutExpired` 分支；Git 结果仍决定检查。Graphviz 未修改。

## gate 与恢复必须共同保持的事实

所有 gate/版本/退出记录绑定 `protocol, launch_id, attempt_id, attempt_sha256`；launch/invocation 还绑定 run。`process.json` 仍为 `{identity, descendants, launch_id}`。

| 文件/状态 | 能证明什么 |
| --- | --- |
| 无 launch/process/log 的预检失败；invocation/result 有 `error`、termination `reason=not_started` | 原有正面未派发证据；session 预检仍在可结清范围内。 |
| `launch.json status=gated` | 尚未授权 setup。只有 creator 确实已死且重读标记不变，才能沿 `launch_not_started` 证明不会启动。 |
| `launch.json status=setup` | 已授权版本查询，模型尚未放行；可能有真实活着的辅助进程。不能由 creator 死亡直接推断范围结清。状态先持久化，再向 pipe 写 `S`。 |
| `cli-tmp/version.json` | 版本主进程及其被 gate 收养的所有后代已真正 wait 结清；含 `exit_code` / `cli_version` 和完整 gate 绑定。版本 stdout/stderr 不混入原生 events。 |
| `launch.json status=released` | 父侧已核验版本/会话并授权模型；不是模型已经退出。状态先持久化，再写第二个 pipe 字节 `1`。 |
| `cli-exit.json descendants_reaped=false` | 主 CLI 已真正退出，gate 仍有活后代；允许清理核准后代，不允许据此采纳成功。 |
| `cli-exit.json descendants_reaped=true` | gate 的 wait/ECHILD 证明全部收养后代已回收；仍需独立确认 gate/已知身份已退出。 |
| `launch.json status=not_started` | gate 未启动模型且 setup 后代已回收，或父侧从未授权 setup；`gate_closed/exec_failed/parent_aborted/spawn_failed` 正面原因继续保留。 |

`cli-exit.exit_code` 是主 CLI 的真实 Popen returncode（信号结束为负数）。`exit.json.exit_code` 是父侧已观察到的 gate returncode；gate 对主 CLI 的负 returncode 转成 `128 - code`。两者不要未经说明地等同。

取消和主 CLI 退出后的孤儿清理只有一轮非阻塞动作，所以 `result/exit` 可能先保存 `termination.confirmed=false`、`exit_code=null`；session 为 `fenced`，公开 outcome 沿现有 parser 为 `termination_unconfirmed`。这不是退出事实，也不是最终账目；后续 gate 可能补出真正的 cli-exit/not_started 证据。普通恢复观察应使用 `observe_owned`，不能无条件 KILL 活进程。

第二轮 C 需仅对尚未 finalized 的调用，核对原绑定、持久控制意图、最新 gate 闭合记录及 fresh 身份观察，再生成结清记录并读取真实日志。不能永久锁在首次 `exit_code=null`，不能把 KILL、函数返回或单独 `descendants_reaped=true` 当最终退出，也不能更改已 finalized 账目。现有 `verified_result` 会把未知原始终结意图保守归为 interrupted；若要恢复“原生成功后清理孤儿”的可采纳性，必须有真实原生成功、退出和未取消依据，不能猜测。

## 定向验证

通过 28 项，均为 fake CLI / 合成子进程 / 临时 Git/SQLite / fake Agent；不代表服务端接受 128000 或真实提醒闭环已验证。

- `BackendRuntimeTests` 17 项通过（当前文件除下述一项外全部）：精确 cwd/会话续用、读写角色、原生成功与提案验证独立、普通/脱离后代取消、成功 CLI 的孤儿清理、无启动与符号链接反例、session fence、固定输出与配置指纹、巨大 elapsed 仅统计、版本不匹配不启动、版本查询可取消、观察错误不杀模型、PID 身份失配不发信号。
- `UsageObservationTests` 6 项全部通过：`test_bounded_native_delegation_preserves_root_delta_and_audit_binding`、`test_real_codex_home_indexes_new_and_resumed_root_counters`、`test_locked_sqlite_index_falls_back_without_busy_wait`、`test_native_index_rejects_paths_identities_and_other_scopes`、`test_totals_deduplicate_disjoint_codex_observations`、`test_claude_resume_uses_native_delta_and_never_sums_session_totals`。
- `ReviewEvaluationTests` 4 项全部通过：`test_grading_requires_current_nonce_coverage_and_exact_field_boundary`、`test_evaluation_uses_registered_agent_backend_and_independent_read_only_scopes`、`test_cancel_event_stops_after_the_active_case`、`test_codex_does_not_receive_implicit_qwen_or_max`。
- Template：`PYTHONPATH=schema_template python -m unittest schema_template.tools.tests.test_schema.SchemaTests.test_local_repository_uses_available_commit_not_the_checkout` 通过（普通模块调用缺 `check_yaml` 搜索路径，已用项目路径重跑）。
- 本轮六个产品文件 compileall 与分配文件 `git diff --check` 通过。

上述 17 项 backend 通过用例的确切方法名：

```text
test_real_cwd_atomic_workspace_and_exact_resume_for_both_backends_and_roles
test_stable_cwd_rejects_untracked_mutable_material
test_codex_native_output_is_distinct_and_bound_to_last_message
test_native_success_schema_nonce_session_and_complete_trace_are_independent
test_cancel_kills_detached_descendants_before_retiring_session
test_launch_gate_and_symlink_inputs_fail_without_executing_cli
test_successful_cli_also_reaps_detached_descendants
test_cancel_ordinary_child
test_unconfirmed_termination_keeps_the_session_fence
test_registered_scope_and_missing_backend_never_use_defaults
test_controlled_commands_preserve_model_and_do_not_load_customizations
test_claude_default_output_is_fixed_and_environment_fingerprinted
test_large_elapsed_time_is_only_an_accounting_value
test_version_mismatch_stops_before_model_execution
test_version_query_is_owned_cancellable_and_never_reaches_model
test_observation_failure_does_not_terminate_a_running_model
test_proc_identity_errors_are_unknown_and_reused_pid_is_not_signalled
```

暂缓集成复跑的确切用例：`schema_extract.tools.tests.test_backends_runtime.BackendRuntimeTests.test_session_preflight_failure_is_bound_unstarted_evidence`。其运行时预检断言通过，进入旧 `recovery.termination_observed` 后因 `process_table(cwd=...)` 报 `TypeError`；未删测试或加兼容参数。

取消/孤儿测试允许首次返回 pending/fence，并由测试 harness 轮询真实闭合后结算；harness 的有限诊断等待不是产品时间限制。首次以 unittest 主入口执行可能报告 Popen 在非阻塞返回时仍活着的 ResourceWarning，测试随后已核准退出并回收；未为消除警告加入产品等待/后台回收框架。

## 第二轮调用点

- C：runner 删除 `validate_max_output_tokens` 导入/配置/调用，删除 `AgentCLI.run(timeout=...)`；store attempt/config/session 额度字段；recovery 的 `process_table(cwd=...)`、`terminate_owned(cwd=...)`、`identity.started`、无条件终止及上述 pending 结清。所有未结 scope 不得重派。
- D：cli review-eval 的 `max_output_tokens/call_seconds` 传参与外层参数；runtime/audit/receipts/prompts 等重复额度身份。环境 `configuration_digest` 和冻结执行器身份继续核验。
- B 已采用新公共进程接口；attempt gate 必须保持默认 `include_anchor=False`，即使取消 Runner 自身也不能误杀 gate。

必要文字事实变化仅为本实施记录；活跃文案中“128000 默认、无时间预算、cancel KILL 后可能 pending、每 case 一次”由第二轮 D 按用户现有正文最小修改。
