# R3 B：当前候选、维护返回与派发准入

基线 `be4fc48`，保留 R1/R2 既有改动。本轮从阶段批准、当前实例、实际来源、规范化与串行合入、P6/维护权限及预算衔接复核当前代码；没有执行真实模型、queue、GPU、完整 P1–P6 试跑或正式发布。

## 确认问题与修复

1. **维护取旧合入点，丢失当前上游草稿及待审计划。**
   - P2 worker 已 complete、尚待 reviewer 时提交当前 P2 review，`mark_drained` 原先优先取旧 `merged_checkpoint`。维护 reviewer 实际收到 P1 的 `root.yaml`，缺失刚完成的 `dispatch.md`；`prepare_workspace` 又将当前 `plan_draft` 替换为尚为空的 `selected_designs`。
   - 对未完成 P3 草稿提交 review，维护按规则返回普通 worker，但 `_resume_normal` 将当前 checkpoint 回退到 P2 合入点。原稿仍在历史 checkpoint，普通续写却拿不到它。
   - P4 worker 已完成、尚待 reviewer 时先维护 P2，P2 review 会占用并清空全局 `plan_draft`；恢复 P4 时再取尚为空的 `writing_plan`，丢失待审写作安排。
   - 三个定向回归在修复前全部失败。修复复用现有当前 `checkpoint`，在既有 `normal_cursor` 保存/恢复 `plan_draft`；维护当前 P2/P4 完整候选时读取这份待审计划。同阶段的普通 reviewer 继续审当前候选，已经批准的计划不留作待审草稿。P5 仍通过原 `start_node` 保留本节点草稿并刷新实际来源材料。没有另建候选选择机制或重置预算。

2. **准备调用期间到达的普通 pause/cancel 漏过最后派发门。**此问题由并行 R3 A 确认后交本代理修复。
   - 在 `prepare_attempt` 已保存调用后、启动线程前持久化 pause，旧最后门只检查 hold/maintenance，仍发出一次调用。新增回归分别确认 P5 pause 和 P1 cancel 后再 pause 都多发一次调用。
   - 最后 supervisor 门先登记 hold，再复用既有 `consume_controls`，将当前控制决定纳入拒发条件。使用同一控制优先级，cancel 不被后续 pause 弱化；已显式 resume 并消费过的旧 pause 不再次挡住维护调用。
   - 拒发沿用原收尾：保留准备证据，清 active/unsettled，退回没有使用的调用及一次性额度，并保留 R1 对 `writing_started` 的撤销。已经启动的调用仍可按 pause 规则结算；未扩大暂停或维护权限。

## 交叉复核结论

P2 的选定集合/具体实例、P4 的写作安排分别绑定独立 reviewer；P4 不能扩张选定集合。R1 的 `wait_for` 仅约束该实例和等待集合的首次准入，实际 `from`、共享 root 及当前来源批准仍持续核验。原稿捕获、整体规范化、字段权限、严格重读与串行合入保持现有边界。

P6 的目录更新不会错误推翻 P4：P4 的批准投影绑定选定集合与写作安排；P6 再审最终目录。P6 限域修改、实际影响者登记和 root 返回定义阶段的路径没有在本轮发现其他确认缺口。维护仍针对授权 P1–P5，完整候选可独立审阅，未完成稿回普通 worker，最终 P6 仍须完成；原阶段/节点的 deadline、已用次数和一次性决定保留。上述其余结论来自代码复核，不据此声明真实模型语义质量已验证。

## 必要离线验证

全部测试使用 `PATH=/usr/bin:/bin /opt/venv/python3.13.12/bin/python3 -B`，测试进程 PATH 不含真实 claude/codex。只使用临时目录与确定性回执；实际执行器配置为 `/usr/bin/true`，未调用真实 Agent CLI。

修复前：

- `test_maintenance_reviews_current_p2_candidate`
- `test_maintenance_preserves_unfinished_upstream_draft`
- `test_maintenance_restores_pending_p4_plan_after_upstream_review`

以上 **3 项均失败**，分别验证候选字节、未完成稿 checkpoint 和待审 plan 的丢失。

- `test_last_control_gate_denies_calls_prepared_before_pause_or_cancel` 的 P5 pause / P1 cancel→pause 两个子场景均失败，分别观察到实际调用数从 8 增至 9、从 0 增至 1。

修复后：

```sh
-m unittest \
  schema_extract.tools.tests.test_current_returns.CurrentReturnsTests.test_maintenance_reviews_current_p2_candidate \
  schema_extract.tools.tests.test_current_returns.CurrentReturnsTests.test_maintenance_preserves_unfinished_upstream_draft \
  schema_extract.tools.tests.test_current_returns.CurrentReturnsTests.test_maintenance_restores_pending_p4_plan_after_upstream_review \
  schema_extract.tools.tests.test_current_returns.CurrentReturnsTests.test_current_maintenance_uses_existing_scope_and_keeps_control_fence -v
```

**4 项通过，41.660 秒**。包含当前完整稿独立批准、原稿字节保留、未完成稿续写、P4 待审计划恢复、原维护边界及预算不重置。

```sh
-m unittest \
  schema_extract.tools.tests.test_stage_flow.StageFlowTests.test_last_control_gate_denies_calls_prepared_before_pause_or_cancel \
  schema_extract.tools.tests.test_current_returns.CurrentReturnsTests.test_resumed_maintenance_ignores_consumed_pause_and_settles_started_review \
  schema_extract.tools.tests.test_stage_flow.StageFlowTests.test_writing_order_does_not_block_started_local_review -v
```

**3 项通过，32.173 秒**。包含新请求拒发、cancel 优先、未用额度和 `writing_started` 回退、显式 resume 维护、已启动 reviewer 结算，以及 R1 实际来源与写作次序的区分。

改动 Python 文件 `ast.parse`、`git diff --check` 均通过。所有测试解释器已退出；没有真实 CLI 子进程或提取/发布任务留存。

## 文件与未决项

- `schema_extract/tools/maintenance.py`
- `schema_extract/tools/stages.py`：本轮仅 `resume_progress` 保留当前待审计划
- `schema_extract/tools/runner.py`：本轮仅最后派发门消费普通控制请求
- `schema_extract/tools/tests/test_current_returns.py`
- `schema_extract/tools/tests/test_stage_flow.py`：本轮仅准备后控制请求回归
- 本报告

当前说明已经承诺草稿保留、完整候选审阅和暂停准入，本轮修复落实这些既有行为，无须改写固定提示或产品文档。没有本轮已确认而未修的问题；未改并行代理所有的模块、Schema 字段、Template、案例、Kernel、Wiki 或既有报告，未提交 commit。报告完成后停止写入。
