# R2 A：六阶段状态、调度与正常主链

已修改 `schema_extract/tools/stages.py`、`store.py`、`runner.py`，新增 `tests/test_stage_flow.py`。正常 P1–P6 已接通至 `ready`，使用 R1 的产物/回执接口、`AgentCLI` 和 R2 B 的提示及材料接口。没有修改 Template、Schema Wiki、旧 run、用户配置；没有工作区 Git commit、真实模型或 GPU 调用。

## 已落实

- P2 的 `selected_designs` 与 P4 的 `writing_plan` 分开保存，分别经过 worker 提案、独立 reviewer 回执及职责投影验证；没有 `state.plan` 或节点 `parent`。P2 的 `{id, instance}` 含 root 至多十项，P4 的 `{id, wait_for}` 只安排这些非 root。机制和代表的独特价值仍由 reviewer 判断，程序只核身份、数量、权限和证据。
- P3–P6 调用绑定 `selection_signature`，P5/P6 另记录 `writing_signature`。P2/P4 的普通 reviewer 必须保留实际 `plan_draft` 的决定，不能悄悄换实例或扩大集合。独立批准同时保存原始回执、最终 checkpoint 与 `approval_projection`；后续合法写作不会因前期全文 SHA 改变而抹掉早期职责投影。
- 每阶段、角色、操作分别登记会话；P5 另按节点分开。`session_scope` 恰为 stage/node/role/operation，`session_directory` 在 attempt 外并按作用域摘要固定。只有结果的 `session_status=ready` 才可登记精确 UUID 续接；没有 P1→P2 共会话。
- 单 root 正常经过 P4，P5 明确 `skipped`，P6 先启动全新 reviewer。P6 未改稿时无 worker；目录仍有“仅计划”登记时，reviewer 可以给出具体 `dispatch.md` 范围，worker 修改后再由原 P6 reviewer 精确续接终审。
- P5 首稿从 starter 构造自己的元数据与待写区域，删除整个 Contract，复制的仅为任务 Name/Describe；不复制 provider 正文，也不生成默认 `from`。已有 raw 优先保留，准备后续调用只更新阅读材料。P4 不创建未来 YAML 或假链接。
- P4 等待只决定开工顺序。P5 可以看到已审设计作为材料，调用登记这些材料的 `source_identities`；合入资格只核实际 `source_manifest` 中采用的来源，以及它们调用时已提供、已审、当前仍有效的身份。缺失、未审、身份漂移及调用之后才出现的来源均拒绝。只复制本次目标 raw 到最新 merge，无关兄弟完成不会使正在审阅的结果过期。
- 使用 Template 的跨区解析闭包及全图检查，没有将同区来源 manifest 当成加载闭包。节点保存当前 `source_context`，影响计算采用真实来源记录，不采用写作等待或虚构父树。
- 正常候选先保存原 receipt 和独立 `original/` 证据副本，再整体规范化到新目录，严格重读、验证固定来源，再交 reviewer。原稿 SHA/行号仍指原稿；reviewer 固定读取最终 raw。所有 `continue`、检查失败后保留的草稿和 CLI 恢复草稿同样核固定来源，`/source/...` 阅读路径不能冒充最终 Source 身份。
- P1 Execution、P2 Source/Dispatch、P3 when、P4 目录、P5 本节点和 P6 明确字段范围都直接使用 R1 的门禁。P3/P5 用中间完成快照，P6 用最终完整目录检查；没有会写文件的默认 CLI 检查，也没有批准后的隐式规范化。
- Runner 接入统一原生结果：退出码、终结成功、隔离证明、schema/nonce、session 状态都必须成立。恢复中已移除以 Codex `turn.completed` 猜成功的逻辑，已有结果通过 `read_result` 和 `settle_session` 处理。状态提交仍只在原 Runner 串行循环，调用计数、代际、预算账目及进程 fence 保留。

## 小接口与接线

| 接口/数据 | 当前约定 |
| --- | --- |
| `DEFAULT_CONFIG` / `create_run` | 新建默认为 `backend=claude`、`agent_binary=claude`、`model=qwen3.8-max[1m]`、两角色 effort=max。实际登记绝对 executable。显式 Codex 必须指定模型和两角色 effort；不套 Qwen 默认。删除 `import_seed` 参数。 |
| `Store.create` / `Store.load` | 不接旧 seed，不接受旧 runtime/semantic 状态格式或缺失当前选择/安排/后端身份的 run。旧数据留原处，没有迁移壳。 |
| `Store.context(state, attempt=None)` | 返回当前同一冻结 lock 的 source_lock/rules_lock/executor_lock 与身份，验证固定输入；不沿历史 revisions 选择规则。 |
| `Store.verify_approval(..., directory=None)` | 先核真实 accepted receipt、Runner gate、immutable checkpoint，再核当前阶段职责投影。`verify_approval_origin` 只证明原批准真实性；`verify_node` 另核目标 raw 和当前实际来源。 |
| `Store.validate_sources(state, attempt, workspace, graph=None)` | 验证实际采用来源；返回 `design_resolution`、`adopted_source_identities` 和仅实际内容来源的 `dependency_digests`。材料全集仍在 attempt.dependency_digests，两者职责不同。 |
| `selection_signature` / `writing_signature` | 独立的、忽略列表和等待顺序的摘要。`selection_impact` / `install_selection` 接 P2；旧 `plan_signature` / `plan_impact` / `install_plan` 已移除。 |
| `prepare_attempt_materials` / `build_prompt` | Store 在封存 attempt.json 前调用 R2 B 的材料接口并合入返回字段，Runner 用原签名组装动态 stdin。逻辑路径由 B 提供；最终 Source 用冻结 repository@ref locator。 |
| `Store.handoff(..., checkpoint_id=...)` | Runner 显式传本次真实 checkpoint，避免从全局指针推测。P5 存 `stages/P5/<node>/handoff.md`；P2/P3/P4 仍按阶段保存。完整过程证据仍为 receipt。 |
| 固定指令 | Runner 恢复交接调用冻结 `render_operation('recovery', context, rules_lock=...)`；signoff feedback 只保存操作名和裁决 attempt，提示由文件加载。stage reviewer 方法指向 `prompt/roles/reviewer.md`。 |

R2 B 的 P6 材料已包含本代际 P2/P4/P5 的 worker/reviewer 回执和短交接，按“先当前产物，再过程记录”排列；不能只取最后 reviewer 的摘要而丢失 worker 记录的新事实。

## 行为验证

`python -m unittest schema_extract.tools.tests.test_stage_flow -v`：7 项通过。先在共享工作树真实接线通过（110.967 秒）；补齐 distinct implementation/reference 来源、草稿来源门禁和 P6 材料断言后，在字节核对过的临时隔离副本复验通过（103.153 秒），避免另一代理修改 executor 导致冻结身份失配。最后对明确 handoff checkpoint 和 reviewer 路径的小改动，仅重跑六节点用例，通过（48.429 秒）。

覆盖内容：

1. 单 root 的 Claude/Codex 两种后端登记，实际 Runner 调度 P1–P4 各自 worker/reviewer、跳过 P5、P6 reviewer 后 `ready`；会话不串用，原稿和规范化稿证据分别保留，批准检查不改字节。
2. 六节点、三区不同来源和额外跨区解析材料；consumer manifest 不包含只为解析所需的 hidden 节点，且不包含无关 sibling。consumer reviewer 在途时 sibling 完成，仍能合入本节点原审字节。
3. P6 首审发现目录计划项，启动只改目录的 worker，再精确续接 reviewer；材料含 P2/P4/P5 两角色记录，handoff 指向本节点真实批准 checkpoint。
4. P2 reviewer 换实例、P4 增项、在途选择变化、迟到的新来源、来源失审和审批身份漂移均拒绝。
5. 单有 `outcome=completed` 不能代替原生/隔离/响应证明；sandbox 阅读路径不能成为合法最终 Source 或可恢复 checkpoint。

另执行 R1 的 `test_artifact_transactions`：12 项通过，以及所有所辖 Python 的编译与 `git diff --check`。R2 B 的 namespace/helper 来源验证由其独立测试和报告覆盖。

替身 backend 提供确定性语义回执，只验证机器主链；合成多实例不证明真实设计具备不同机制。单 root `run()` 测试的发布函数只回传 `ready`，真实发布留 R3；没有宣称真实模型语义验收或完整交付已经完成。

## R3 / R4 必须闭合的事项

- **定点返回与当前 run 维护。** Runner 的返回入口仍保留旧“新证据”限制和较宽的阶段失效范围，需要按具体事实/已有误读、实际定义和使用关系重写。已有实际引用的失效消费者需要正确等待 provider 修复；不能只按 P4 wait_for 派发。P2 集合变更后的目录/已有稿件处理也由 R3 在审阅前完成。保留原预算与 generation，不重新建立父树。
- **来源等价与细粒度影响。** 当前身份漂移采取拒绝。R3 应接 `align_source_identity`，在严格证明定义、Contract 和使用上下文未变时记录当前身份与依据；不能直接沿用旧 identity。`affected_nodes` 目前按实际来源文件传递，区域/字段差异的更精确影响范围仍需 R3 收口。
- **裁决、一次签收与预算。** 所有实际 patch 已经转新 signoff，P5 未签收时不合入；但 budget 的非 P6 签收调用/时间预留尚未推广，恢复中一次性消费的深层验证仍归 R3。select 分支不会在 reviewer 批准之后改 Version 或再格式化；旧候选的版本、来源及规范化准备必须在提供给裁决之前完成。P2/P4 签收时的最终决定绑定也须与候选准备一起闭合。同步核对新的进程内 Template 检查与剩余检查预算的结算边界。
- **清除旧活路径。** 所辖正常路径已断开直接 `revisions.resolve_context`，但 Runner 仍接 `revision_control` 屏障和当前 maintenance 模块，Store 的 Supervisor ledger 尚有 continuation 检查，runtime 仍由旧模块提供执行核验。R3 A/B 需先保留排空、发布和维护的当前 run 责任，再删除旧模块及入口。maintenance/recovery/publication 等旧调用仍需改成新的选择、来源与检查 API；不要恢复已删的 plan/parent alias。
- **恢复与发布。** 无 result 的中断必须在确认终止后用 Agent 的 `retire_session` 清理真实 session 存储；不能仅清 state UUID。最终 publication/standalone 应使用当前 selected_designs 和完整目录 API、批准投影、来源身份；真实交付与幂等/冲突验证仍未由本轮执行。
- R4 再做适用 suite、当前文档、正式新入口和有限真实 CLI 的最终集成。当前状态只可称正常主链已接通，不能宣称执行器已完成全部上线验收。

本轮已完成并停止写入，后续共享文件交下一位新代理。
