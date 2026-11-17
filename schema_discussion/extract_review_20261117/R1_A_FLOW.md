# R1 A：阶段、批准与写作等待

基线 `be4fc48`。阅读当前总览、Supervisor 手册、执行协议和 REVIEW，审查六阶段推进、批准绑定、实际来源、返回、维护与预算；未读取 archive，未调用真实模型、queue、GPU 或执行真实六阶段流程。

## 确认问题与修复

1. **写作次序变成持续内容依赖。** `stages.py:215–232` 原先每次调度都要求 P4 `wait_for` 当前有效。已完成本地初稿、没有采用该来源的节点，也会被先写节点的后续退修挡住。新增回归在原实现中于 `dependencies_ready(consumer)` 断言失败。
   - 在节点中只记录 `writing_started = {instance, wait_for}`，`wait_for` 排序后比较；它表示该实例已经通过此安排的派发准入，不是新的批准。
   - `store.py:945` 登记准入。相同实例和安排的继续、复审、退修或恢复沿用此事实；实例或等待集合改变则重新核验。其他节点安排改变不会影响本节点。
   - `runner.py:1386,1416` 在控制门拒绝派发时恢复原记录；没有调用获准执行便不消费新的安排。
   - 实际 `from` 来源、共享 root 与有效批准仍持续核验；预算、会话和节点 generation 不重置。
2. **维护 schema 与可执行决定矛盾。** `receipts.py:119,152` 曾向维护模型提供 `reopen`，而验证器和维护操作明确拒绝该决定。现在 proposal 和完整 receipt 均不提供该选项/字段；普通阶段的返回能力保持。扩展现有协议测试，覆盖维护两种角色。
3. **Codex 不支持的 effort 到首次调用才失败。** 与 B 对齐，`runner.py:68` 在输入冻结前调用所选后端的 `validate_effort`。原提取入口可选项不扩大；Codex 的 `max` 立即拒绝，后端规则由 B 的 `backends.py` 方法维护。

P2/P4 各自绑定独立 reviewer 回执、P4 不改 P2 集合/实例、P6 前提和当前目录门槛、实际来源失效传播，以及返回不退还预算的现有路径均已检查。本轮没有对这些部分提出其他确认缺陷，也没有加入按关键词判断设计价值的规则。

## 必要离线验证

测试命令均使用前缀 `PATH=/usr/bin:/bin /opt/venv/python3.13.12/bin/python3`，真实 claude/codex 不在测试进程 PATH。流程用确定性回执和临时目录，只运行指定场景。

```sh
-m unittest schema_extract.tools.tests.test_stage_flow.StageFlowTests.test_writing_order_does_not_block_started_local_review -v
```

修复前：1 项失败，确认为持续等待问题。

```sh
-m unittest schema_extract.tools.tests.test_stage_flow.StageFlowTests.test_writing_order_does_not_block_started_local_review schema_extract.tools.tests.test_review_protocol.ReviewProtocolTests.test_current_maintenance_and_adjudication_keep_explicit_authority schema_extract.tools.tests.test_recovery_budget -v
```

修复后：写作回归与预算 5 项全部通过；新扩展的协议测试因原 fixture 缺少 `input_fingerprint` 报错，属于测试材料缺项，已补齐该调用的固定身份与 plan revision。

```sh
-m unittest schema_extract.tools.tests.test_review_protocol.ReviewProtocolTests.test_current_maintenance_and_adjudication_keep_explicit_authority schema_extract.tools.tests.test_current_returns.CurrentReturnsTests.test_current_maintenance_uses_existing_scope_and_keeps_control_fence -v
```

随后 2 项通过，包括真实 Runner 接收维护回执的离线路径。没有重复全量回归。

另运行离线 Python 断言：对 Codex worker/reviewer 分别设置 `max`，用 `patch('schema_extract.tools.runner.freeze_inputs')` 核对 `create_run` 均抛出 effort 错误且 `freeze_inputs.assert_not_called()`；结果通过。所有改动 Python 文件 `ast.parse` 通过，`git diff --check` 通过。

## 交下一轮的文字同步

- `schema_extract/tools/README.md`“并行合入与计划修订”：明确同一实例、同一 `wait_for` 只约束首次派发准入；实例或等待集合变化重新核验；实际来源始终需要当前有效批准。可在 P4/P5 阶段页相应一句中保持同义表述。
- `schema_extract/prompt/operations/maintenance.md`：删除末句“若 schema 含 reopen，本轮填 null”。维护 schema 已不包含此字段；越界/上游问题仍按该页要求 blocked 并交 Supervisor。
- CLI / review-eval 的 effort 入口一致性由主代理与 B 后续协调；本代理仅改 `create_run`。

未留下本轮已确认而未修的产品问题。验证仅证明所列离线场景，不代表真实模型审阅质量或整批提取已经验证。

## 本代理实际改动文件

- `schema_extract/tools/runner.py`
- `schema_extract/tools/stages.py`
- `schema_extract/tools/store.py`
- `schema_extract/tools/receipts.py`
- `schema_extract/tools/tests/test_stage_flow.py`
- `schema_extract/tools/tests/test_review_protocol.py`
- 本报告

未改 `budget.py`、`maintenance.py`，未修改产品文档/固定 prompt，未提交 Git commit。
