# R8 A：流程、产物与精简最终验收

**结论：签收修复后的当前版本。** 独立核对精简前源码、现行实现和必要反例后，未发现三轮精简撤销前四轮修复。发现一处未执行调用提前消费恢复交接的遗漏，已与 B 沿同一处理路径修复；没有恢复 `.gitkeep` 接管、普通分页读取或阶段 `handoff.md` 副本。

## 实质发现与处理

### 恢复交接在实际调用前被消费

`Store.prepare_attempt` 会将待交接记录的 `next_attempt_id` 绑定到本次准备。最后 pause/cancel 门拒发后，已有代码撤销未使用调用、一次决定、P6 开始计数及 P5 写作准入，却没有撤销交接领取。下一次准备只选择未绑定的交接，因此新上下文会漏掉这份恢复说明。

新增反例先保留真实 P1 中断草稿并登记已退休会话，再在下一次准备后提交 pause。修复前：没有派发任何调用，草稿和预算保持，但 `next_attempt_id` 仍指向未派发调用，断言失败。此遗漏已存在于精简前代码，不是删除普通阶段 `handoff.md` 引入的问题。

与 B 进一步核对发现：恢复调用的线程已经启动、CLI 却被原生 gate 确证没有执行时，也会留下同一领取。最终处理为：

- B 在 `recovery.py` 增加小函数 `release_unstarted_handoff`，仅清除 `next_attempt_id` 等于当前未执行 attempt 的绑定；原有确认未执行的恢复分支共用它。
- 本代理让最后拒发门调用同一函数；`_recover_call` 对携带恢复交接、尚未建立新会话的调用进入既有 `handoff_session` 核验。
- 已执行会话的退休、未确认终止的屏障、其他调用的交接、草稿与已用预算保持原语义。没有新增状态层或恢复协议。

最终反例确认：再次显式 resume 后，第一个实际派发的确定性 fixture 调用收到原交接，登记新会话，并只增加该次调用的计数。B 的真实假 gate 场景另确认 `exec_failed / 125 / empty` 不丢交接；下一次准备仍取得同一记录。

还逐项检查了准备阶段的其他状态写入：调用及一次决定额度、P6 开始计数、写作准入和维护 attempts 已有拒发回退；基线、候选、材料和固定 deadline 是保留的工作证据或时间边界，没有发现另一项需要撤销的提前消费。

## 精简保留功能的依据

先验证 `/tmp/schema-extract-pre-prune-ego_8hhb/manifest.json`：**95 份文件的 SHA256 与字节数全部匹配**，再逐文件比较本代理负责的 13 个核心模块。`receipts.py`、`budget.py`、`standalone.py`、`snapshots.py`、`comparators.py` 与精简前快照完全相同；其余差异逐项核到调用方及最终边界，没有只采用前轮报告的结论。

| 验收项 | 独立核到的当前行为 |
| --- | --- |
| P2 有限代表与独立批准 | `validate_plan` 要求唯一具体实例、root 及最多十份；P2 reviewer 不能改变待审选择。`verify_approval_origin` 追到已接受回执、reviewer 调用、独立签收、检查结果及不可变字节；删去无消费者状态投影不影响这些证据。 |
| P4 首次准入与实际依赖分开 | `writing_started` 只对应当前实例和排序后的等待集合；首次通过后不再把先写节点当持续内容依赖。`actual_sources`、`effective_nodes`、`validate_sources` 与 `verify_node` 仍持续核对实际整区来源、root 和当前批准；实例或等待集合改变重新准入。 |
| 当前维护与恢复 | `mark_drained` 从当前 checkpoint 起步，`normal_cursor` 保留待审 `plan_draft`；完整候选独立 review，未完成稿回普通写作，恢复时保留当前 working checkpoint。P5 恢复仍取自身草稿并刷新来源，最终仍须 P6。scope 查询合并没有改变原预算所有者。 |
| 早期文本、warning 与权限 | P1/P2 复用 Template 适用结构、算法及文本预算，warning 留在 notices；未来 Design 不要求提前完成。各阶段文件及字段边界、P1 最小 Execution、P6 明确范围及 reviewer 只读仍在。删去 `decision` 参数只涉及从未读取它的权限函数。 |
| 规范化、原稿及来源 | 原稿摘要与证据保留；整组候选落地前后核字节，严格重读后才接受。来源实际内容 manifest 与额外解析闭包继续分开；只读 provider、使用节点身份和当前批准不会被规范化绕过。 |
| P6 与发布 | `_final_inputs` 核精确选定集合、当前 P1–P4/P5 批准和实际来源；P6 检查整个目录。发布仍绑定当前独立 P6、完整检查和已审 raw，暂存后再通过最新控制门执行原子不覆盖 rename。删除 intent 不影响同结果重放：目标完整 manifest 含交付 receipt，rename 后状态写入中断可按同一证明重新登记。 |
| standalone | 使用 Template 代码 manifest 的 digest 选择解析缓存；生成与严格验证使用同一包，提交前重核代码身份。实际来源闭包、完整验证目录、节点自身上下文、来源漂移及目标冲突检查均保持。 |

第三轮三项主动裁剪均合理：

- **新任务使用新输出目录。** `create_run` 在冻结/创建 runtime 前拒绝已有目录。发布中的 `.gitkeep`、其他文件及空目录都由原子不覆盖规则保护；已发布同结果的当前 run 仍可幂等重放。旧 scaffold 的备份/inode/intent 恢复链不再有产品职责。
- **普通文件阅读交给原生工具。** 当前 prompt 明确 Read/Glob/Grep；`--evidence` 仍校验文件、摘要、行号和长行字符偏移，阶段自检仍在。删除分页游标没有删掉证据核验或只读检查。
- **短交接直接定位已接受 receipt。** `prepare_attempt_materials` 把当前批准回执列为 `current_decision`，校验回执摘要与调用身份，并保留跨阶段 `knowledge_destination`。新上下文可直接读取该回执的 `handoff`；独立的中断恢复交接也保留，并补齐上述未执行边界。

R4 的字段/工具接口映射与当前 Extract 使用方式一致。本轮只核接口及差异，没有重新设计字段或重跑全套图。相对 `be4fc48`，`schema_docs`、Template 实际工具及九份案例无修改；Template 的变化仅为前轮已记录的 README 表述修正。

## 本轮实际验证

测试统一使用 `PATH=/usr/bin:/bin /opt/venv/python3.13.12/bin/python3 -B`，确认 PATH 中没有真实 Claude/Codex。流程用临时材料、真实 Template 检查和确定性回执；没有运行完整 P1–P6、真实模型、queue、GPU 或正式发布。

**本代理执行的九个不同定向用例全部通过：**

| 组别 | 实际用例 | 结果 |
| --- | --- | --- |
| 新反例，最终 helper 版本 | `test_stage_flow.StageFlowTests.test_last_control_gate_preserves_pending_recovery_handoff` | 1 项，3.447 秒 |
| 前轮流程修复 | `StageFlowTests.test_last_control_gate_denies_calls_prepared_before_pause_or_cancel`；`StageFlowTests.test_writing_order_does_not_block_started_local_review`；`test_current_returns.CurrentReturnsTests.test_maintenance_restores_pending_p4_plan_after_upstream_review` | 3 项，37.784 秒 |
| 产物与接口 | `test_artifact_transactions.ArtifactTransactions.test_selection_and_waiting_are_distinct_bounded_decisions`；`test_early_checks_enforce_current_text_budgets_without_future_design`；`test_early_algorithm_warnings_are_reported_without_blocking`；`test_p1_normalizes_before_gates_and_preserves_original_evidence`；`test_standalone.StandaloneTests.test_repeated_views_use_the_current_template_validator` | 5 项，2.474 秒 |

补充的小检查均通过：已有 `.gitkeep` 输出在输入冻结前拒绝且不创建 runtime；`rename_new` 保留冲突目录与暂存字节并能提交到新目录；长 Unicode 行末证据引文及 SHA 精确。当前产物权限接口的 **13 处调用**经 AST 核对均无已删除的 `decision` 参数；改动 Python AST 和限定 `git diff --check` 通过。

新反例修复前确实失败。一次并行修改期间的旧控制门测试因 `executor differs from inputs.lock` 中止，是冻结后代码变更正确触发身份门；与 B 建立产品静止窗口后重跑通过，没有绕过身份核验。

共同恢复路径的扩展验证由 B 实际执行：`test_current_recovery.CurrentScopeRecoveryTests.test_actual_transient_exit_recovers_only_its_scope_without_new_credit` 覆盖临时失败、既有授权恢复、真实 gate 未执行及下一次交接。B 报告该组九项通过、子进程为零；不重复计入本代理上述九项。所有本代理测试解释器均已退出。

## 文件与剩余项

本代理本轮修改：

- `schema_extract/tools/runner.py`：最后控制门释放未交付的恢复交接；无新会话恢复调用进入既有核验。
- `schema_extract/tools/tests/test_stage_flow.py`：新增中断草稿、准备后 pause、再次 resume 的真实流程反例。
- 本报告。

共同 helper、未执行恢复分支及真实假 gate 扩展由 B 修改并在其报告登记。本代理未改其他产品模块、正式规则、Template、案例、Kernel、Wiki、archive 或旧报告，未提交 commit。

本代理范围内没有已确认而未修的产品问题。签收范围为当前静态实现与所列离线行为；真实模型审阅质量和正式提取仍属于此前明确未运行的工作。报告完成后停止写入。
