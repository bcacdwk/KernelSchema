# R2 B：提示、材料与回执协议

基线 `be4fc48`，保留第一轮修复。本轮核对当前六阶段、角色与操作提示、动态材料、决定与证据 schema，以及 Supervisor/执行边界说明；未通读 archive，未调用真实模型、queue、GPU 或执行真实提取与发布。

## 确认问题与实际修复

1. **正式引用指南未进入调用材料清单。** `_linked_rules` 的白名单漏掉根 README、Extract 正式指南及 Template 工具说明。A 同步补冻此前缺失的执行边界与 Template 工具 README；B 按现有链接导航补齐这些正式指南，继续排除 archive、wiki 和案例。没有要求模型通读全部指南。
   - 扩展现有 P6 材料回归，在原导航实现上确认为失败，补齐后通过。完整冻结文档、动态身份及现有 P2/P4 worker 知识记录仍保留。
2. **提示未完整描述已有回执准入。** `response.md` 现在明确：普通/维护 P2、P4 reviewer 独立认可后原样回传 `plan_draft`；四项文档判断和纯文档问题如何使用真实 artifact 锚点；源码问题及 reopen 的来源要求；P6 的精确授权与实际受影响节点、批准时的 `pending_rechecks`；P2/P3/P4 approve 必填短交接。没有鼓励为文档缺陷填入无关源码依据，也没有增加回执字段。
   - 明确候选仅用 `id/scope/summary/evidence`，将条款、反例、最小建议和影响写入现有字段。reviewer 只将独立核实项写回候选数组，未确认项在结论说明；Supervisor 再整理为 issue 台账格式。同步 REVIEW、工具手册和 issues 入口。
3. **P6 的写作说明不够精确。** 阶段页现在直接列明目录及非 root 的现有可写字段；root 的任务、代表来源和定义分别返回 P1/P2/P3。已核对 `REOPEN_TARGETS`、`_begin_reopen` 和既有 REVIEW/reopen 职责：有对应返回、保留 raw、派发原职责 worker 和独立复审的路径。没有为了文字一致改变修稿权限。
4. **第一轮文字和固定指令位置未同步。** P4/P5 和工具手册已说明 wait_for 只约束本实例、本等待集合的首次派发准入；实例或集合改变重核，实际 from 来源批准持续核验。维护提示删除已不存在的 reopen 字段说明，越界/上游问题仍 blocked 交 Supervisor。
   - `prompts.py` 的历史材料 purpose 改为事实标签，解释保留在 P5 Markdown；经主代理授权删除 `receipts.py` 中重复的 summary.description 指令，summary 字段及非空语义验证保持。
5. **正式说明夹有过时写法与实施过程。** REVIEW 中的字段名称对齐当前规范；执行边界保留运行环境、角色权限、会话和验证含义，移除“本次移除隔离”的过程叙述，原验证报告及链接保留，未改历史证据。

新上下文已有的共同说明、角色、阶段、操作、应读规范和动态身份足以建立当前任务；现有 P2 独特价值审定、P4 计划、P5 实际来源及 P6 知识去向均保留。文字和命名原则继续由共同提示给出，具体字符预算与格式由 Template 维护。`review_eval` 的独立只读 scope、限定响应和创建前参数校验未发现本轮需要修改的缺陷。

## 必要离线验证

命令统一使用 `PATH=/usr/bin:/bin /opt/venv/python3.13.12/bin/python3`，测试进程 PATH 无真实 claude/codex；review-eval 使用 `RecordedAgent` 替身及 `/usr/bin/true` 的参数构造检查。

- 导航修复前，扩展的 `test_prompt_materials.PromptMaterialsTests.test_p6_receives_worker_facts_after_final_content_and_only_real_documents_are_provided` 在正式指南材料集合断言处失败。
- 修复后，同一材料回归、`test_issue_evidence` 全部用例及 `test_review_protocol.ReviewProtocolTests.test_known_source_misread_can_return_once_without_new_budget`：**15 项通过，12.750 秒**。覆盖文档锚点、无关/错误证据拒绝、P4 计划一致性、来源要求及 P6→P3 返回与原预算保留。
- `test_review_protocol` 的 `test_every_role_receives_current_task_decisions_sources_scope_and_budget`、`test_embedded_documents_are_exact_current_frozen_text_with_no_missing_read_specs`：**2 项通过，1.452 秒**。
- `test_review_eval.ReviewEvaluationTests.test_evaluation_uses_registered_agent_backend_and_independent_read_only_scopes`：**1 项通过，4.624 秒**，两个后端均为离线替身。
- 所改 Python 文件 `ast.parse`、`git diff --check` 通过。

最终相关检查共 **18 项通过**。只增加现有材料用例的实际导航断言，未为文案新增测试。上述结果不代表真实模型审阅质量或真实六阶段已经验证。

## 文件与未决项

本代理实际修改：

- `schema_extract/tools/prompts.py`
- `schema_extract/tools/receipts.py`：本轮仅 summary.description，其他差异属于第一轮
- `schema_extract/tools/tests/test_prompt_materials.py`
- `schema_extract/prompt/operations/response.md`
- `schema_extract/prompt/operations/maintenance.md`
- `schema_extract/prompt/phases/04_dispatch_plan.md`
- `schema_extract/prompt/phases/05_branch.md`
- `schema_extract/prompt/phases/06_final.md`
- `schema_extract/REVIEW.md`
- `schema_extract/EXECUTOR_ALIGNMENT.md`
- `schema_extract/tools/README.md`
- `schema_extract/issues/README.md`
- 本报告

没有本轮已确认而未修的代码问题。不改 schema 字段、Template、案例、Kernel、Wiki 或既有报告，没有恢复隔离、增加兼容层或提交 commit。报告完成后停写。
