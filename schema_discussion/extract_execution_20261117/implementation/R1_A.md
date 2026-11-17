# R1 A：当前产物、计划与回执接口

已实际修改 `schema_extract/tools/artifacts.py`、`receipts.py`、`comparators.py`，新增局部行为测试 `test_artifact_transactions.py`。没有修改主链、Template 规则、旧 run 或用户配置；没有模型/GPU 调用。`common.py` 无需改动。

## 实现

- 当前 `META_DATA / CONTRACT / DESIGN / REALIZATION` 和三区整区 `from`；删除本模块的单父计划、父表、换父及旧字段适配。
- P2 最小 `{id, instance}`，含 root 不超过 10，ID/具体实例唯一；P4 最小 `{id, wait_for}`，选定非 root 各一项，等待可空/多个，拒绝集合变化、自环、环。机器不裁决独特价值。
- P1 允许元数据、算法与最小 Execution；其他硬件固定为当前 starter。P2 仅自身 Source/Dispatch/Version 和目录；P3 允许完整 root Design/Realization/实际 when；P5 只改自己、禁止复制 Contract 或换 ID；P6 验证明确文件和字段，根定义走所属阶段。
- 规范化调用冻结 Template `normalize_documents`，整体处理根、目标、解析必需来源和受修改定义的实际使用者。原始文件/语义权限、source_hashes、版本及组内修正范围都通过后，才在新受控目录落地、严格重读、原子移入目标路径。越权整组拒绝，不修改原稿。
- 中间快照用 `validate_documents` 与适用的本地来源检查；目录登记、计划文件和实际文件清单另验。P6 用完整 `validate_resolved_file`；不会写 YAML。审批投影独立于原始全文摘要，来源变化可记录当前身份与严格等价证据。
- 原始 proposal/receipt 行号和 hash 始终指原稿，最终 reviewer 证据指最终 raw。保留一次 select/patch/blocked；裁决改稿使用同一规范化事务且记录 Runner 来源。

## 对外 API

全部来自 `artifacts.py`，`parser_root` 指冻结规则根：

| API | 参数和返回 |
| --- | --- |
| `validate_plan(stage, plan, *, parser_root, selected_designs=None)` | stage 为 P2/P4；P4 必须给已审 P2 列表。返回忽略条目/等待顺序的决定摘要；P2 可另给 selected_designs 核对身份/实例不变。失败抛 `ExtractError`。 |
| `validate_index(directory, selected_designs, *, parser_root, final=False)` | 当前中文目录使用 Template 检查；未来/未转正式登记项在完整设计表首列写普通 `<id>.yaml（仅计划）`，不得链接。返回实际文件、正式登记、计划及未建文件集合。最终集合必须精确且无计划项。 |
| `check_stage_snapshot(directory, *, parser_root, stage, selected_designs=None, node='root', checked_nodes=None)` | P1 可不传选择；其他阶段必传。P3/P5 自动补 root/目标与跨区解析闭包；额外完成节点由 checked_nodes 指定。返回 exit_code、diagnostics、checked_nodes/excluded_nodes、目录检查和严格 source context。P6 总是检查全部文件，不能借 checked_nodes 排除。 |
| `normalize_candidate(before, original, destination, starter, *, stage, role, decision, selected_designs=None, node_entry=None, revision_scope=None, first_draft=False, checked_nodes=None, maintenance_fields=None, adjudication=None)` | before 为本次基线；original 为需保留的原稿；destination 必须不存在且位于受控目录。通常 role=worker、decision 为 complete/continue/blocked/reopen，P5 node_entry 用 P2 条目。返回原始/候选/最终摘要、corrections、scope/check 和最终目录；拒绝抛 `CandidateRejected`，`.validation` 保留诊断和已经形成的证据。只有 complete/approve 执行完成快照检查；不完整稿仍有路径、身份、语义权限及重读门禁。 |
| `source_context(directory, *, parser_root, nodes=None)` | 解析来源，不宣称完成校验；按需节点自动补全解析材料。每节点返回 origins、source_manifest、effective_identity、resolution_nodes/manifest、raw/Contract/region/use_context 摘要。 |
| `align_source_identity(previous, current)` | current 必须来自成功严格检查；比较当前 consumer raw、真实 origins、Contract、有效三区和完整使用上下文。返回 equivalent、旧/新 identity、来源差异与可复查比较依据；不沿用旧 identity。 |
| `approval_projection(stage, directory, *, parser_root, plan=None, selected_designs=None, node='root')` | 返回阶段 subject_sha256、原始 raw_manifest 与适用的 source context；P2 用选择 plan，P4 用安排 plan + selected_designs。这里只提供投影，不代替独立审批。 |

保留的 `validate_artifacts(...)` 现在只做**规范 raw 的阶段权限门禁**；完成检查另用 `check_stage_snapshot`。旧 `root_revision / candidate_review / identity_migration` 参数已删除。`check_complete(directory, rules_root, *, selected_designs, filename='root.yaml')` 是最终完整文件 API，没有会改文件的 CLI 路径或旧 timeout 参数。

`receipts.validate_receipt(..., artifacts, changed_files, *, rules_lock=None, semantic_artifacts=None)` 新增 semantic_artifacts：worker 的 artifacts/changed_files 为原稿证据，semantic_artifacts 为规范化后落地目录；reviewer 两者必须相同字节。P2/P4 的普通 reviewer 都核对复用的 state.plan_draft。

`design_context` 保留调用装配接口，但来源资格来自实际定义者。P5 attempt.source_identities 为本次已提供、已审的 node→effective_identity；同时核对 state.nodes[node].status/approval.effective_identity 和实际依赖字节。不是阅读材料的永久依赖，也不推祖先。

## R2 必接线

1. 使用 state.selected_designs 与 state.writing_plan 的运行投影，分别绑定 P2/P4 独立 receipt；两个阶段复用 state.plan_draft。P3–P6 attempt 必须带 selection_signature=`validate_plan('P2', ...)`。proposal nonce 仍是顶层 attempt_id，无 nonce 别名。
2. 原稿先 `normalize_response` 固定 proposal 证据并保存原稿/receipt；调用规范化事务；再对原稿 receipt 使用 semantic_artifacts 做语义核验。保存 `.validation`、最终 checkpoint 后启动全新只读 reviewer。原始 changed_files 不要换成规范化后 changed_files。
3. P5 启动前从 starter 建自己身份与待写区域、删整个 CONTRACT；现有稿原样续写。stage_files/writable_files 改成 `<id>.yaml`；P2 dispatch.md 预建空文件。初稿传 first_draft=True，后续实际数据变化按 Version 检查，纯规范化无需增长。
4. Store 提供真实 material closure 与 source_identities，主链合入时另核实际采用来源在调用时已提供/已审、当前仍适用。来源等价对齐写入现有 validation；不要把解析材料或 P4 wait_for 全部当批准依赖。
5. P3/P5 用中间快照 API，并明确当前仍有效的额外 checked_nodes；P6/发布用最终 API。旧 `_merge_check` 不能再伪造完整目录或调用会写文件的默认 CLI。最终选择集合严格一致。
6. stages 的结论/提示移除 tree/inherited/parent 语义；P6 文档目录结论现在使用 `directory`。旧 index_prefix 已删除，不再由机器生成固定父表；P2/P4 的机制、差额及依据保留在目录与 conclusions/evidence/handoff。

## 裁决与后轮

Runner 应用已有 `{path, old, new}` 裁决后，调用同一 `normalize_candidate(..., role='runner', decision='approve', adjudication=已验证裁决receipt)`。事务要求 receipt 是 reviewer/adjudicate/approve/patch，先从 before 唯一匹配重放全部补丁，证明 original 没有额外改字节；核对 stage/P5 node，再按相同阶段权限规范化。记录 origin=runner_adjudication 和裁决 receipt 摘要。普通 reviewer 无写入入口；R3 必须在有改稿时启动有界独立 signoff，最终批准只能取签收回执。select 候选需要在裁决前完成规范化和身份调整。

主链尚未接这些新接口，旧计划/迁移测试也未转换；本轮不以兼容壳掩盖。恢复、实际来源失效、预算预留、最终签收与发布由后轮完成。本轮来源接口能解析和证明字节/内容事实，不替代 Store 的审批真实性与代际核验。P1 的“最小、有据”和代表的独特价值仍需 reviewer，不做关键词判断。

## 验证

`python -m unittest schema_extract.tools.tests.test_artifact_transactions -q`：12 项局部行为测试，使用临时目录及 Template 当前最小样本。覆盖选择/等待边界、当前字段权限、Execution、P6 限域、规范化/原证据、跨区链与环、同区 manifest 和解析闭包、实际 provider 资格、本地化、来源漂移等价、越权整组拒绝、未来/最终目录、裁决纯格式改稿、逐输出比较依据与阶段批准投影。

另执行三个修改模块的编译检查和 `git diff --check`。未运行真实模型、GPU、正式 Extract 或全库旧测试矩阵。
