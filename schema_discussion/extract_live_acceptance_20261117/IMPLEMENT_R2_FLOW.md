# 第二轮 C：执行流与恢复

2026-11-18。基于已合并 `f48823b` 及第一轮 WIP 实施，未改固定 schema、案例、旧 run 或主机配置，未提交或推送。除本记录外未编辑 MD/prompt。

## 已实施

- 删除 `budget.py`；Runner/Store/stages/recovery 删除算子、阶段、调用、裁决时间额度、deadline、预留秒数、检查/候选准备/merge 时间拒收、补时与按年龄终止孤儿。elapsed、processing、实际 usage 和重试间隔仍记录。
- Claude 输出配置由第一轮 backend 统一；本轮删除 config/attempt/summary/recovery 的专用输出字段，保留 `configuration_digest`、执行器和输入身份核验。
- 次数逻辑迁入既有 `stages.py`，默认 32 次、独审/裁决/签收调用预留、实质退修/修复次数和一次性决定均保留。没有替代的时间保护限额。
- Runner 统一 `AgentCLI.run(..., cancel_event=...)`，删除 transport 签名探测及 timeout 参数。普通孤儿恢复只观察；明确取消或真实主 CLI 已退出后的 owned 残留才能 KILL，健康兄弟继续保有其 fence 和计数。
- Agent 首次返回 pending 时不增加 invalid_receipts；未结清调用保留 active/unconfirmed/unsettled，取消不授予新的采纳或重派权限。
- 维护 worker 回执错误后的重提沿最近同 task/step、已结清 attempt 的 `result_checkpoint`。原 attempt/result、原 base/dependencies、当前 checkpoint 的实际字节、parent、attempt/stage/node 及 `repair.json` 摘要和 artifact_digests 均须匹配。新 attempt 从保存的新稿开始，沿同角色 UUID 重提；旧提案不被采纳，新完成仍须独立审阅。`maintenance.py` 由 D 实现严格默认接口。
- 新增内部 `Store.control_transaction()`，只在非阻塞锁获取失败时于锁外退让；事务体不重跑。Runner 与 D 的维护内部事务共用，外部控制提交和 Runner finally 仍单次尝试。经 supervisor 确认补位修改 `publication.py` 一处内部锁；取得锁后保留原最新控制、发布上下文与身份复核。

## 最终接口和结清事实

- `stages.call_limit(state)`、`call_exhaustion(state) -> str | None`、`validate_add_calls(calls) -> int`、`add_calls(state, calls, *, identity, origin) -> bool`。`state.call_additions` 保存 `{id, origin, run_id, at, calls}`；同授权不重复追加，异量重用拒绝。
- `monitor.reserve_launch(..., add_calls=0)`；launch 的 `add_calls` 为整数。summary 的 `calls` 包含 `limit/revision_limit/additions/scopes`；不保留混合 budget 对象。停止状态/理由为 `call_limit_reached`，修复次数理由后缀为 `_limit_reached`。
- `maintenance.validate_attempt(..., status="pending", current_checkpoint=None)` 默认严格检查原候选；Runner 仅在重提证据已核准后传当前 checkpoint。
- `confirm_termination(..., cancel=False)` 默认 `observe_owned`。统一 `start_token` 身份，无 cwd/grace/started 兼容。`setup` 不证明未启动；真实 gate 后代回收和 fresh 身份观察独立核验。
- `complete_pending_result` 只处理未 settled、未 accepted 调用。闭合后重读原生 trace，并用真实 `cli-exit.json` 结清；`exit_source="cli-exit.json"` 明确这是主 CLI returncode，不伪称 gate returncode。Agent 保存原 `terminal_outcome`；未知意图沿 interrupted，明确 cancel 不转换为成功。
- 现有 `exit.json` 同时绑定 `previous_result_sha256/settled_result_sha256`，用于“新 result 已落盘但 state 摘要尚未提交”的单一转换重放。须有明确 native exit 来源、原 attempt/launch、真实 gate 闭合、fresh 观察和原生验证。已 settled 记录不改。
- 版本查询的 `cli-tmp/version.json` 纳入已有 final evidence 摘要；D 将该 JSON 和版本 stdout/stderr 纳入审计封存。
- manual control request 与 cancel delivery 保存冻结 binding；历史 pending cancel 可复查，但被新 pause/hold 覆盖后只观察，不重发旧信号，不覆盖新控制。当前 Runner 记录明确 `launch_id`。CLI pending cancel 先交付/观察、再 reserve 的顺序由 D 完成。

## 定向验证

26 个不同短用例在定向批次通过，另有最后两个直接受影响用例复跑通过。只使用临时 fixture、fake CLI、合成进程/身份和虚拟时间；未跑大 suite 或完整提取流程。

- `test_call_limits` 5 项、`test_scoped_recovery` 3 项：调用追加幂等/不退款、独审与一次决定预留、很老的健康孤儿/P5 兄弟、人工 scope 来源变化及后续 pause。
- `test_monitor_evidence` 原 8 项及新增 `test_later_pause_only_observes_the_pending_cancel_delivery`：控制提交/锁忙/进程归属/gate 保锚，新 pause 保持原控制 ID 且不重放 KILL。
- 第一轮暂缓的 `BackendRuntimeTests.test_session_preflight_failure_is_bound_unstarted_evidence` 通过。
- `CurrentRecoveryTests`：原生 exit/nonce/schema、lost gate fence、receipt 原生成功及预检返回契约、pending cancel 真实 gate 结清/写入窗口重放/已 finalized 不写，共 4 项。
- `CurrentReturnsTests.test_large_checker_elapsed_preserves_success_and_independent_review`：虚拟 `10**12` 秒检查耗时仍正确接收并交独审。
- `StageFlowTests` 新增两项：P1 fake CLI cancel→真实闭合→显式 resume，旧调用不采纳、计数保留且 invalid_receipts 不增加；SupervisorBusy 退让后完成事件与回执不丢失。
- `CurrentDeliveryTests.test_publication_lock_retry_rechecks_new_pause_before_rename`：只隔离发布最终控制边界，锁争用期间新增 pause 后返回 controlled，未 rename；未跑六阶段流程。
- 本轮产品/测试 `py_compile` 与限定文件 `git diff --check` 通过。

并行编辑曾使冻结 executor 严格校验正确拒绝，已改为双方明确交接静止验证窗口后复跑。其余初次失败是旧 timeout/预检异常断言或测试预期，均按新真实接口修正，没有放宽冻结身份或 gate 核验。非阻塞取消首次返回时会产生 Popen ResourceWarning；测试随后已观察真实退出，没有为消除告警增加产品等待上限。

D 的最终集成意见已落实：历史 cancel 后续控制优先、共享内部锁退让、版本证据封存、双 hash 转换必须显式 native exit 来源。D 已报告维护 worker 保稿/同 UUID/独审及篡改反例、CLI 配置/取消顺序和审计归档通过。128000 服务端接受与真实提醒闭环由 supervisor 决定，本轮未执行，不能以离线结果代替。
