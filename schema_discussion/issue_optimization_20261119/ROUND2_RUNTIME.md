# 第二轮 xhigh 实施：运行与保进度续接

2026-11-19，北京时间。代理 `/root/issue_r2_runtime_xhigh`，`gpt-6-astra / xhigh`。依据 [ROUND1_RUNTIME.md](ROUND1_RUNTIME.md) 和两组共享接口约定实施；基线为当前未提交工作树，未重置已有改动。

## 结果与边界

四项局部修复保持原实现和门禁。续接现在核验保存内容的来源、原调用和权限，再报告未完成诊断，支持初始稿、P5 节点稿、P6 continue、回执补正稿和确认中断稿。P6 正常保存与裁决编辑共用节点 draft 更新规则；原执行器遗漏的指针按当前完整候选及原保存链恢复。

本轮仅修改 `executor_repairs.py`、`snapshots.py`、`store.py`、`cli.py`、`runner.py`、`stages.py`、两份 extract README 和四个对应测试文件。新增测试文件为 `test_executor_snapshots.py`。现有 `runtime.py`、`audit.py`、`publication.py`、`maintenance.py`、`artifacts.py`、`reading.py`、`receipts.py` 的已有必要修复保留；没有为了扩大差异重新改写它们。

未修改正式 issue 状态、版本、历史输入锁、原批准/回执/调用账或 `.schema_extract/v2`。未执行真实 extract/resume/monitor/cancel、模型、GPU、工作树提交或推送。FMHA 的原自动失败和 Sparse 取消 successor 不恢复。

## 五个稳定 ID

| ID | 本轮落实 | 验收重点 |
| --- | --- | --- |
| `maintenance-empty-poll-control-lock` | 保留无请求的只读快路径和锁内重读；保留派发/发布最终屏障。 | 持锁空轮询、预读后请求变化、封印身份负例。 |
| `failed-node-recovery` | 保留 scoped retry 在清理 reason 前补交反馈及原幂等关系。修正过时测试的实际故障注入和结清预设。 | 原失败信息仅补一次，健康兄弟、来源变化、pause、调用次数和未知终止 fence 不变。 |
| `p6-retained-scope-inventory` | 保留完整 YAML inventory 与局部内容解析的分工。 | retained scope 可命名未改文件；未知文件、root、Version、受保护字段仍拒绝。 |
| `receipt-evidence-repair` | 保留证据选择/必填格式错误回原角色有限重交；不代写证据或批准。 | 原预算和会话、真实 revise/blocked、合法保存稿；补验此类稿的续接及保存证据被改后的拒绝。 |
| `progress-preserving-continuation` | 实现来源/权限与完成度分层、全节点保稿、最新 P6 指针、严格 capsule 复用及统一审计标记。 | 见下文。继续限定工具修复，源码/任务/语义规则迁移不支持；正式归档交第四轮。 |

## 续接与共享接口

- `_saved_frontier` 复用 checkpoint、原 attempt、已 accepted receipt、validated/repair/recovery 及调用账中的证据。保存权限来自原调用中已哈希的 check 材料，保留原 `first_draft`、maintenance 字段和 P6 scope，不用当前下一轮 scope 推测历史权限。准备稿按原 starter、共享元数据初始化或已登记内容组合核验；reviewer 合并只能采用实际审过的字节。
- `probe` 核对当前及所有节点 draft、原批准来源，记录 `retained_drafts`、完成度退出码和诊断。普通未完成诊断不授予批准；核验无法运行、身份/来源/权限失效仍拒绝。apply 再核前沿、历史 attempt 和控制依据，保留 blocked、调用上限和待 signoff 状态，不自动 resume。
- `retain_p6_drafts` 只处理实际已验证的 changed YAML，被普通 worker、裁决编辑和恢复保存共用。旧指针修复比较所有选定非 root 文件，沿当前 P6 parent/producer 链定位保存者，apply 原子安装；不动原批准、来源身份和历史 checkpoint。
- `freeze_executor_patch` 对完整目标材料再读核验，并检测规则文件清单变化。并发发布复用输入快照已有的严格校验：lock、完整 manifest、无链接、封印状态均须匹配，才接受同内容 winner；保持父目录同步。目标语义材料新增或变更均拒绝。
- 新授权仅支持用户选定的 `continue_with_audit_marker` 和 `--authorization`。具名审计点必须与原 attempt 的真实阶段一致；未登记调用的停点用当前阶段和 `attempt_id: null`。删除未被现场历史使用的无标记授权及 `--review` 别名。授权可如实引用持续用户指令，无需伪造独立审查。
- 单一写者共享接口已交付：`Store.control_transaction(*, on_wait=None)` 仅有 callback 时测 monotonic；每次取锁失败后锁外回调，正文/退出异常不重放。CLI 前台 monitor 首次满 1 秒、其后每 30 秒最多提示一次，说明可中断并用同一提醒 ID 核对续接；没有等待超时预算。仅打印 monitor 结果时使用 `clock_display` 副本，人读 at/last 使用北京时间，`inspect --json` 不变。较旧 scan 的 `superseded_run_ids` 也在人读输出显式显示。Clock/helper 的实现及独立观察验收见 [ROUND2_MONITOR_SCHEMA.md](ROUND2_MONITOR_SCHEMA.md)。

## 离线验收

全部命令以仓库根目录为 cwd，使用：

```sh
PATH=/opt/venv/python3.13.12/bin:/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 python -B -m unittest ... -q
```

| 用例参数 | 最终结果 | 日志 |
| --- | --- | --- |
| `schema_extract.tools.tests.test_executor_repairs` | 11 项通过；1 项测试数据与现行归一化字节不符，已由下行修正验证；392.806s | `/tmp/r2-runtime-final.log` |
| `schema_extract.tools.tests.test_executor_repairs.ExecutorRepairTests.test_p6_adjudication_edit_preserves_draft_and_requires_signoff` | 1 项通过，42.514s | `/tmp/r2-runtime-adjudication-final.log` |
| `schema_extract.tools.tests.test_executor_snapshots` | 3 项通过，3.994s | `/tmp/r2-runtime-snapshots-final.log` |
| `schema_extract.tools.tests.test_maintenance_polling schema_extract.tools.tests.test_scoped_recovery schema_extract.tools.tests.test_receipt_repair_policy` | 9 项通过，10.218s | `/tmp/r2-runtime-local-fixes.log` |
| `schema_extract.tools.tests.test_artifact_transactions.ArtifactTransactions.test_p6_retained_scope_covers_unchanged_artifacts schema_extract.tools.tests.test_artifact_transactions.ArtifactTransactions.test_p6_changes_are_limited_to_the_reviewed_fields` | 2 项通过，1.118s | `/tmp/r2-runtime-inventory.log` |
| 下列四个原基线失败用例 | 4 项通过，6.870s | `/tmp/r2-runtime-baseline-repairs.log` |

最终共有 **30 个独立定向用例取得通过结果**。裁决用例从当前已归一化的 Thesis 读取精确 old 文本，不再假定末尾句点被保留；生产的精确单次匹配门禁不变。准备稿来源优先复用已核验的同名文件字节后，P5 节点草稿例也再次通过（`/tmp/r2-runtime-final-corrections.log`；该批另一个裁决数据错误已由上表单例结清）。

四个原基线用例：

```text
schema_extract.tools.tests.test_current_recovery.CurrentRecoveryTests.test_setup_failure_audit_preserves_missing_stream_and_rejects_forged_proof
schema_extract.tools.tests.test_current_recovery.CurrentRecoveryTests.test_setup_failure_keeps_the_previous_ready_session_and_usage_is_read_only
schema_extract.tools.tests.test_current_recovery.CurrentRecoveryTests.test_unstarted_gate_preserves_previous_session_even_when_result_is_missing
schema_extract.tools.tests.test_current_delivery.CurrentDeliveryTests.test_one_call_audit_archives_setup_failure_and_preflights_original_names
```

三个 setup_failure 使用真实 launch 前的 `ClaudeBackend.environment` 异常；取消/未启动及正常完成用实际 `confirm_termination`/结清路径收尾，保留精确 UUID、原 session、无 handoff 和所有伪证据负例。测试可出现 Python 对尚待回收的假 gate 子进程发出的 ResourceWarning；断言仍检查真实终止，不在生产流程加入同步等待。`test_executor_repairs` 改模块导入 fixture，避免重复采集整套 StageFlowTests。

续接组合验收覆盖：P1 初始诊断；P5 根游标下未完成节点与健康兄弟批准；P6 continue/回执补正/确认中断保存稿；两轮分别改 A/B 后恢复两份最新节点字节；实际裁决改稿待一次 signoff；两次绑定之间和之后的正常调用及新 session operation；提交前后中断幂等、提交前新 pause、错误审计阶段、保存证据变更、越原阶段字段、blocked/调用额度保留；正常维护→节点独立复核→P6 发布和正式 receipt 的原标记。

另完成 10 个修改 Python 文件的 AST 解析、所改 tracked 文件的 `git diff --check`。全量 extract 验收由根代理统一执行，本轮不重复运行全量。

## 历史事实与后续复核

只读调用 `Store.load`、`verify_execution`、`verify_bindings` 验证两条真实已封存历史：Sparse 原 run `8f0844f9-f279-4781-a19d-0216b7fc4ef5` 仍为 `ready/published:true`；FMHA 原 run `fb8b3758-92d7-4093-aa18-c2e5028f0a43` 仍为 `invalid_receipt/published:false`。两者各一条原 marked binding 通过验证；没有对这些 run 执行 probe/apply 或保存。

第三轮精简应保留保存类型的覆盖与来源/权限硬门槛，尤其不要把合法未完成诊断重新变成完成稿前置，也不要把它误当作普通阶段可跳过检查。原状态、正式 receipt 的 `pending_user_decision` 与后续专项的独立结论分别保留。最终 FMHA/Sparse 内容审计和正式 issue 归档由既定后续轮次承担。
