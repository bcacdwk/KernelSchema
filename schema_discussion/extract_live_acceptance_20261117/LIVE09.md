LIVE09 — 原生会话完成后回执拒绝导致 UUID 丢失，启动前错误误留终止 fence

结论：当前产品已修复两处生命周期边界。已正常完成、身份匹配的原生 session 在回执语义校验前独立登记；session 预检失败进入已有未启动结清流程。13 行来源证据仍被拒绝，未修改提案或放宽任何固定 schema 规则。8 项定向测试通过。真实 FA4 r02 仍被冻结旧执行器阻塞，不能把当前产品修复视为旧 run 已恢复。

真实现场：FA4 r02 run `a9a16cd1-7232-46c0-9721-1e8d98a227d9`，冻结快照 `db0e42767335da0a1a561b4ebd659c95017805df2bc58d93697f145190db7848`，executor version `2.5.1`、digest `9c9ab2494891c9a9a64d99cb45c541351102ec66553cb6e30328b5051e76c85f`。

- P2 worker attempt `98ca1cea-0547-483c-9685-d1d18893f1c7` 的 `result.json` 为 completed / exit 0 / native success / session ready / decision complete，UUID 为 `19ad9241-d038-4825-9786-98fdcd171534`。
- `source_evidence[50]` 指向 `kernel_wiki/aka-four-problems-20261019/fa4_prefill/aka_solution/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py:2114–2126`，共 13 行。12 行上限有效，原提案因此未获批准。
- 同一 P2 / root / worker / normal scope 的 `sessions/70370867819edc2c642451b272f9645d611ac0d9344df21bd019e016cba4ccd9/session.json` 正确登记 ready、原 UUID、原 attempt；state 中 `sessions.P2.worker` 却为 null。
- 正式 resume 准备 attempt `85686d53-656e-4b79-ae3c-7e29ad31c4bc` 时沿用该 null。`event.json` 在 0.007012650370597839 秒记录 `resume requires this exact registered role session UUID`；该 attempt 没有 invocation / launch / process / exit / result 文件，随后 state 保留 active 和 unconfirmed fence。

根因与修改：

1. [runner.py](../../schema_extract/tools/runner.py) 旧路径在回执、产物检查之后才 `_remember_session`。语义拒绝进入 `_reject`，虽然调用已确认结束、旧 active 被释放，native UUID 却未登记，下一次调度据此错误地申请新 session。现在 `_receive` 先核验 attempt 当前适用性，再要求 completed 或 receipt_error、confirmed termination、exit 0、native success、匹配 backend，并沿原 `_session` 校验精确 UUID、角色/阶段 scope、既有身份与 handoff 禁用记录。通过后先独立保存 session 登记，再检查回执与产物。该保存不创建批准、checkpoint 或通过回执，跨 scope UUID 仍被拒绝。
2. [agent.py](../../schema_extract/tools/agent.py) 旧 `run()` 在 `_run()` 的 try/结清范围外检查 session 登记；失败以异常返回 Runner，result=None 被当成终止未明。现在用一个 `ExitStack` 持有原 session 锁，把有效 attempt 的 CLI 参数、session 预检及设置放入已有 `_run` 边界；锁仍保持至结果和 session 结清。启动前失败返回带原 attempt 摘要的 invocation/result，outcome=cli_error、exit_code=null、termination.confirmed=true、reason=not_started、session_status=not_started。复用现有 `never_launched`、恢复和审计路径，不新增 marker、恢复状态机或迁移层。
3. 正常调用的 gate 登记、Popen、持久化释放许可、发送启动字节、回收后代进程的顺序未改。预检拒绝不写原 session，也不制作 launch/process/exit/events 文件；启动后终止不明的旧 session 仍 fenced。未启动是由仍在执行的调用边界记录，并非从“找不到 PID”推断死亡。准备/调用计数和 elapsed 继续保留，不退还预算，也不虚构 provider usage。

本次未改 `processes.py`、`recovery.py` 或固定 receipt/schema 字段；沿用已有 no-exec 证据与账目逻辑。原提案、所有来源证据和 12 行限制均保持原样。

定向验证：使用临时目录、真实产品 Store/Runner 接收逻辑和本地 fake Claude CLI，不调用真实模型。P1 fixture 先独立通过，再让 fake CLI 对 P2 返回原生 success/complete，提案含实际固定来源的 13 行范围；两次 P2 都被正常拒绝，第二次 CLI 实际检查 `--resume` 等于第一次 UUID。之后仅在测试 fixture 模拟旧状态丢 UUID，证明第三次预检失败可结清、原 session 字节不变、累计调用仍为 3 且该 attempt 的用时入账。

| 验证 | 结果 |
| --- | --- |
| P2 native success + 13 行证据拒绝 → 下一次同 scope 精确 resume | 通过；两次均未获批准，UUID 持久保留 |
| native success 返回另一个阶段的 UUID | 拒绝；P2 未登记该 UUID |
| 同 scope 缺失精确 UUID 的 session 预检 | 确认 not_started；恢复复验通过，原 session 不变 |
| 已启动但终止未确认 | 原 session 保持 fenced，再次调用未执行 CLI |
| 只有 dispatched、没有启动/退出正向证据 | 仍 termination unconfirmed |
| scope / backend 缺失与错配 | 拒绝，无默认身份回退 |
| 输出预算 session 绑定 | 错配仍拒绝 |
| gate 未释放、symlink 输入、原生成功/nonce/schema/trace 独立门禁 | 原有定向检查通过 |

实际测试命令：

```sh
python3 -B -m unittest \
  schema_extract.tools.tests.test_stage_flow.StageFlowTests.test_native_session_survives_rejected_source_range_and_resumes_exactly \
  schema_extract.tools.tests.test_stage_flow.StageFlowTests.test_native_completion_cannot_adopt_another_scopes_session \
  schema_extract.tools.tests.test_backends_runtime.BackendRuntimeTests.test_session_preflight_failure_is_bound_unstarted_evidence \
  schema_extract.tools.tests.test_backends_runtime.BackendRuntimeTests.test_timeout_and_unconfirmed_termination_keep_the_session_fence \
  schema_extract.tools.tests.test_backends_runtime.BackendRuntimeTests.test_registered_scope_and_missing_backend_never_use_defaults \
  schema_extract.tools.tests.test_backends_runtime.BackendRuntimeTests.test_output_budget_is_explicit_fingerprinted_and_bound_to_the_session \
  schema_extract.tools.tests.test_backends_runtime.BackendRuntimeTests.test_launch_gate_and_symlink_inputs_fail_without_executing_cli \
  schema_extract.tools.tests.test_backends_runtime.BackendRuntimeTests.test_native_success_schema_nonce_session_and_complete_trace_are_independent
```

结果：8 tests，17.055s，OK；`git diff --check` 通过。日志为 `/tmp/live09_tests.log`。未运行全套回归、GPU 或真实模型。

现场只读核对：读取公开结果、提案、状态、登记文件和冻结工具代码；未读取真实 events 或私有 session 内容中的隐藏推理。选定 27 个原 state / inputs.lock / attempt / result / session 登记及稿件文件的 SHA-256 在验证前后完全相同；冻结 agent.py、runner.py、recovery.py 仍与原 lock 匹配。核对记录为 `/tmp/live09_real_evidence.json`。

| 原文件 | SHA-256 |
| --- | --- |
| state.json | `8bb4b53e2eeff5a4eb934e162d316302dc613b864e169d9eb488a53f0f7e16f6` |
| inputs.lock.json | `a38dcfc6d035f3dcb2e417bb04f9d8513cb36cdb5e1df931b72c7e1d9de24c28` |
| 原 P2 result.json | `aa8e48b057036f23fc660a4c8b0f3247cbfcbacf4daf158cb8f160d8d43aa393` |
| 原 P2 attempt.json | `4a2d1e87af30c4ce962b2e501f5fc62f08675e060ec4296cb1cb30c2d185dd0a` |
| 恢复 attempt.json | `23761b2a52226e02e2279c6c4368c681b1dd2c4ce87d56bd3b2fe2d9d90208c4` |
| P2 session.json 登记 | `a789adec40dc24549f40dff4539e48f528445079103a7f03dada166a21a34cea` |

冻结旧 run 的正式处置建议：保留现场，新建 FA4 run。`cli.py` / `runtime.frozen_run_invocation()` 的正式 resume 路径会回到旧快照；旧恢复程序要求带 invocation/result 的正向未启动证据或 gate 证据，而本次旧预检异常未留下这些记录，单靠 event 错误和无进程文件不能满足其规则。普通 resume 不会获得当前修复，当前代码也不应替旧 run 猜补 UUID、改 state 或补造 no-exec 记录。maintain 仍须排空 active/unconfirmed 调用，不能借它绕过旧 fence。

新 run 可沿 [LIVE08](LIVE08.md) 与 [STATE](STATE.md) 已用的流程：用完整原 source descriptor、新输出路径和已授权 Claude Code `qwen3.8-max[1m]` / max 配置排队，再通过现有 `maintain` 的明确阶段/字段任务，在 reason 内嵌旧 Qwen 草稿原文、旧 run/attempt 与摘要，标明只作参考、必须重新核对新 run 固定来源。提醒新 worker 自行修复过长来源证据。旧 root/目录及 root+split_kv 方案不能继承批准或当作新 checkpoint；新 run 仍重新完成 P1–P6 和独立审阅。不要把旧草稿伪装成 acceptance/reference/scope，也无需新增 seed/迁移系统。本次只核查机制并建议，未创建任务或新 run。

本次仅修改 agent.py、runner.py、两处定向测试和本报告；保留已有 LIVE01/05/06/07/08 未提交变更。未修改 `.schema_extract` 任何状态、快照、原稿或私有 session，未操作旧 FA4 或当前 QSA r03，未撰写模型 schema/回执，未提交 Git。验证与报告完成，停止写入。
