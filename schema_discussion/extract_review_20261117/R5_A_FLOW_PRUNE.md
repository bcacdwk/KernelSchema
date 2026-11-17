# R5 A：流程与状态精简

本轮仅记录相对 `/tmp/schema-extract-pre-prune-ego_8hhb` 的改动；不把基线 `be4fc48` 之后 R1–R4 的修复记为本轮成果。先读本目录 README、R1 A、R3 B 与 R4 B 结论，检查阶段推进、维护、回执及预算边界，直接删去已确认的重复表达，没有新增抽象层、兼容层或迁移逻辑。

## 实际删减及等价理由

1. **范围查询统一使用已有 `stages.scope_for`。** 删除 `runner._scope_at` 的重复 P5/普通阶段选择，以及 `maintenance._scope` 的纯转发包装；原调用位置直接调用 `scope_for`。所有调用都显式传入当前 attempt/task，因此仍取相同的阶段/节点计数对象。`owner_for` 保留：维护任务、普通阶段状态和 P5 节点是不同的状态所有者，不能和预算 scope 合并。同时删除 `Store.prepare_attempt` 对模块顶部已有 `next_calls` 的重复导入。
2. **批准动作只保留一个分支。** `stages.advance` 的普通 reviewer approve 与 adjudicate/signoff approve 均调用相同 `_approve`，现收束到一个条件。`adjudication_changed` 仍先于 approve 分支，将改稿交独立 signoff；blocked、revise、一次性计数及 mode/role 接受条件保持原语义。
3. **删除两个没有消费者的状态投影。** 删除 root/普通节点初始化与批准时写入的 `approved_checkpoints`，以及 invalidate 时写入的 `approval_required_generation`。已检索当前全仓使用并核对 summary、audit、回执、恢复及当前说明；前者没有读取，批准检查点已有 scope/node 的 `approval.checkpoint` 和不可变已接受回执记录；后者没有读取，恰好重复当前 `generation`。当前批准、`last_approval`、回执历史及代际核验全部保留。未更改历史运行文件，也没有加旧字段清理或迁移。

## 明确保留的边界

- `wait_for` 仍只约束当前实例和等待集合的首次派发，实际 `from`、root 与来源批准仍持续核验；P2 选定身份与 P4 写作安排各自绑定独立批准。
- 维护 `normal_cursor` 的当前草稿和 `plan_draft` 保存/恢复、完整候选独立 review、未完成稿回普通 worker、最终 P6 保留。维护不提供 reopen。
- 派发前后不同时间点的检查保留，包括最新 pause/cancel 的最后准入门、拒发时未用预算与 `writing_started` 回退；后端 effort 创建前校验保留。
- `last_draft_checkpoint`、`baseline_checkpoint`、`recoverable_checkpoint`、预算预留记录及候选历史保留：它们参与恢复、候选选择或已有验证，不因表面相似删除。
- receipt 的 proposal 身份、完整回执、原稿证据与最终落地字节检查对应不同输入或时点；本轮未改。预算保留调用、时间、审阅、裁决及签收各自额度。本轮未改 `receipts.py`、`budget.py`、发布代码或正式字段规则。

## 实际离线检查

只执行下面四项已有回归，没有新增或删除测试。测试进程 `PATH=/usr/bin:/bin`，不包含真实 Claude/Codex CLI；固定 fixture 使用 `/usr/bin/true` 注册执行器和内存确定性回执，没有调用真实模型、queue、GPU、完整提取或发布。

```sh
PATH=/usr/bin:/bin /opt/venv/python3.13.12/bin/python3 -B -m unittest \
  schema_extract.tools.tests.test_current_returns.CurrentReturnsTests.test_all_stages_reserve_signoff_and_p2_patch_binds_final_plan \
  schema_extract.tools.tests.test_current_returns.CurrentReturnsTests.test_p4_signoff_interruption_retires_session_without_new_decision_credit \
  schema_extract.tools.tests.test_current_returns.CurrentReturnsTests.test_current_maintenance_uses_existing_scope_and_keeps_control_fence \
  schema_extract.tools.tests.test_current_returns.CurrentReturnsTests.test_maintenance_restores_pending_p4_plan_after_upstream_review -v
```

**4 项通过，43.410 秒。** 覆盖普通批准、P2 改稿后独立签收及计划绑定、P4 签收中断不返还决定次数、P5 当前维护批准与代际失效、未完成稿回普通流程，以及维护后待审 P4 plan 恢复。测试解释器已退出；本轮没有创建须清理的假 CLI 子进程。

四个改动 Python 文件 `ast.parse` 通过；限本轮四个产品文件的 `git diff --check` 通过。逐文件对照精简前快照确认本轮增删；没有以完整 Git diff 冒充精简差异。

## 文件与未决项

- `schema_extract/tools/runner.py`
- `schema_extract/tools/maintenance.py`
- `schema_extract/tools/stages.py`
- `schema_extract/tools/store.py`
- 本报告

没有本轮已确认而未处理的问题；上述验证限于列出的离线行为，不代表真实模型审阅质量已验证。未改测试、固定 prompt、产品文档、其他代理文件或冻结快照，未提交 commit。报告完成后停止写入。
