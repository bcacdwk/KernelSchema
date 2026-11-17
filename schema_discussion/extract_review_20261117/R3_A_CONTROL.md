# R3 A：命令、控制与审计接口

基线 `be4fc48`，保留前两轮改动。核对当前手册、执行协议、执行边界和 R1/R2 报告，串读 CLI、monitor/wakeup、审计、用量观察及恢复接口；只执行临时目录中的离线小场景。

## 确认问题与修复

1. **已结清但被拒绝的回执无法归档。** 原生调用成功后，Runner 先保存 `receipt.json` 和 `original/`，再核回执语义；证据索引越界或空 summary 会留下真实拒绝/修复记录，却不会产生 `validated.json`。`audit --attempt --verify --archive` 对所有 receipt 都要求该文件，因此将这类失败现场报为 incomplete。CLI 单调用复现得到 `invalid_receipt / proposal_rejected`，审计仅报 `No such file: validated.json`。
   - `audit.py:633` 仅对**未 accepted**、且 attempt 用量账目已绑定 `rejected.json` 或 receipt-repair 记录的回执开放此路径。原稿仍核精确文件集合和摘要，并重建回执核对原生提案、固定来源及原稿锚点。
   - 报告明确标记 `receipt_not_accepted`、缺少完成的 validation 及真实拒绝原因，结论为 `partial / evidence_only`。不产生接受记录、批准或恢复权限。
   - 已接受回执仍必须有匹配 `validated.json`；真实成功调用后删除该文件的反向场景继续拒绝归档。不存在笼统的缺文件跳过规则。
2. **后台 Runner 领取启动前退出，仍被当作 launching 复用。** 使用真实 `/usr/bin/false` 子进程并 wait 结清后，进程表已无该 PID；原 `start_background` 仍保存 launching，后续使用存活的 launcher 身份判定该启动可复用。长期存活的 monitor 下会一直卡在这一启动窗口。
   - `monitor.py:442` 读取已有 `Popen.poll()` 的真实退出事实，仅在同一 launch ID 且状态仍为 launching 时记录 `launch_failed` 和 exit code。已 claimed/finished 或被替换的启动不被降级。
   - 对明确的 `launch_failed` 直接产生启动失败观察，不再要求原 launcher 已退出。重试可登记新的 launch；run、预算和 attempt 保持原值。
3. **准备窗口内的普通 pause/cancel 未挡住最后派发门。** 最小离线复现：`prepare_attempt` 返回前持久登记 pause，随后仍发生一次 fake 调用和一次接受，最后才 paused。已转交 R3 B（Runner 所有者），本代理未写 Runner。
   - B 已在最终 supervisor 门复用 `consume_controls`，将当前控制动作纳入拒绝派发条件。
   - B 回报：P5 pause、P1 cancel→pause 均零调用、无新接受；调用及一次性额度、P5 `writing_started` 回退；已启动 reviewer 仍可结算，显式恢复可消费旧 pause。其相关边界及 R1 写作等待共 3 项通过，详见 B 报告。

## 必要离线验证

所有命令使用 `PATH=/usr/bin:/bin /opt/venv/python3.13.12/bin/python3 -B`，真实 claude/codex 不在测试 PATH。模型调用均由绝对路径的 fake 可执行文件替代；没有投递 queue。

- 新 audit 回归修复前：拒绝和 receipt-repair 两个子场景均因缺 `validated.json` 失败；反向的 accepted 缺文件场景保持拒绝。
- 修复后，`AuditReceiptRecoveryTests` 与既有单调用归档测试 `test_one_call_audit_archives_setup_failure_and_preflights_original_names` 共 **2 项通过，13.456 秒**。覆盖拒绝/修复证据、accepted 缺校验、启动失败无事件流、成功调用及原稿完整性边界。
- 新后台启动回归修复前：实际退出码 1，却得到 `('launching', None)`；修复后得到 launch_failed/1，观察报告启动异常，下一次登记不再复用旧 launch。
- 最终 `test_monitor_evidence` 与 `AuditReceiptRecoveryTests` 共 **5 项通过，7.989 秒**。三个原有并行节点诊断定位用例仍通过。
- 四个改动 Python 文件的 `ast.parse`、限定 `git diff --check` 通过。fake 早退子进程已 wait；最后检查 `offline-receipt` 剩余进程为 **0**。

## 保留行为与未决项

CLI 继续路由冻结执行器；scope 恢复绑定原 attempt、当前实例/来源和剩余额度，EX_TEMPFAIL 自动恢复仍需持久授权并受 pause/cancel/hold/维护屏障约束。提醒的 ack、已提交 scan 与启动授权保持分离。Claude 增量、主循环和累计口径及 Codex 根会话观察继续分列，失败账目不重置、未知值不填零。框架候选查询只取摘要匹配的 accepted receipts，reviewer 的候选数组对应当前提示要求的独立确认项。

本轮没有其他确认而未修的问题。没有扩大命令权限、增加审批/兼容层或重写用量协议；修复落实现有手册承诺，无必需文档改动。验证不代表真实模型审阅质量或完整提取已验证。

## 本代理修改文件

- `schema_extract/tools/audit.py`
- `schema_extract/tools/monitor.py`
- `schema_extract/tools/tests/test_current_recovery.py`（仅新增本轮 audit 回归，保留 R1 内容）
- `schema_extract/tools/tests/test_monitor_evidence.py`
- 本报告

未改 CLI、wakeup、observation、recovery、Schema 字段规则、Template、例子、Kernel、Wiki 或 archive；未执行真实模型、GPU、全流程试跑、正式发布或 Git commit。报告完成后停写。
