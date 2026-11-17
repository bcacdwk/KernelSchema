# R4 B：Changelog 的 Extract、作者入口与接口核查

完整阅读 `schema_docs/CHANGELOG.md` 的 2026-11-11 至 11-17 条目及末尾文本草稿，按后续最终决定归并要求。检查当前根入口、Extract 阶段/角色/操作 prompt 和实际调用、材料、产物、批准、恢复、用量及发布接口；保留 R1–R3 已有修改，不把其测试结果作为本轮实测。DOC0–6、Template 内部规则和四个现行案例的逐项检查由并行 A 负责；本报告核对这些规则进入 Extract 与作者流程的接口。

确认两类遗漏，均在根作者入口：已取消的嵌套反馈写法，以及部分旧字段拼写。已定点修复；没有确认需要修改 Extract 代码的新问题。

## 有效要求与现行落点

表内“静态”表示本轮阅读当前实现与文本；V1–V7 是下文实际执行的离线检查。

| 有效要求（合并同义） | 当前实现与核查证据 | 状态 |
| --- | --- | --- |
| Design 围绕主要 workload 逐层完成具体 root，再判断其他完整方案是否值得并存；范围对照任务要求，输入默认值域与精度有明确入口 | [DESIGN_SCHEMA](../../DESIGN_SCHEMA.md) 第 1–5 节具备数学→调度→流水线的推进、返回、范围和资料导航；参数/局部/组合交 common 目录规则 | 静态一致；本轮修正 Wiki.Sota 字段拼写 |
| 含 root 至多十份具体完整设计，每份有独特完整价值；参数、局部变化、已知组合及未完整展开的独立候选诚实记录，不额外造 provider | [EXTRACT_SCHEMA](../../EXTRACT_SCHEMA.md)、P2/P4/P6、两份 reviewer 说明一致；`artifacts.validate_plan` 对 P2 的 `{id, instance}` 检查 root、上限、唯一 ID/实例，P4 不能增删选定身份 | 静态一致，V1 通过；独特价值仍由独立 reviewer 作语义判断 |
| P1 只确定任务元数据、共同算法 Contract、有据 Execution 最小登记，其余硬件保留 starter；Design 和 root 代表未在 P1 获批 | [P1](../../schema_extract/prompt/phases/01_contract.md)、`artifacts._validate_scope/_early_check`、`comparators.validate_comparisons` 对齐；算法与逐输出比较器单独核查，适用检查显式 skip-hardware | 静态一致，V2 验证早期门禁不要求未来 Design |
| P2 独立审定机制与代表取舍；P3 写具体 root，P4 落实已定安排；只在有具体事实或关键关系缺口影响取舍时回 P2 | P2/P3/P4、REVIEW 与 `receipts.REOPEN_TARGETS`、`runner._begin_reopen` 对应；正文漏项与普通记录按所属阶段修，未要求各阶段重新搜索全部候选 | 静态一致 |
| P4 只批准未来计划；正式记录指已有基线及对象，未来文件无假链接，等待安排无自环/循环 | [P4](../../schema_extract/prompt/phases/04_dispatch_plan.md)、`validate_plan/validate_index` 分别检查选定集合、等待图、当前链接及“仅计划”登记；P4 写权限只有 dispatch | 静态一致，V1 通过 |
| root 有唯一 Contract 与三个完整本地区域；非 root 使用同任务同名整区 from、无 patch、全图无环，名称/参数在使用节点绑定 | P3/P5/REVIEW 与 `artifacts._snapshot/resolve_workspace/_resolved_context` 调用冻结 Template 的真实解析器；scope 门禁拒绝非 root Contract、身份或受保护字段变化；Realization、when、Version 不继承 | 静态一致；旧 parent/from_parent 未在当前链路复活 |
| 写作等待只约束当前实例/等待集合的首次派发；实际 from 才是持续内容依赖，来源批准与消费者适用性分别核验 | `stages.dependencies_ready/affected_nodes`、`store.validate_sources`、`artifacts.align_source_identity` 使用实际来源链及严格当前快照；等价保留比较 Contract、区域、raw 与完整使用上下文，不改原批准 | 静态一致，V3 验证关联规范化不能越权改 provider |
| 每阶段、角色、P5 节点独立上下文；reviewer 只读并审有效全文、来源及每个使用上下文，历史批准不补足正文 | [REVIEW](../../schema_extract/REVIEW.md)、`prompt/roles/reviewer.md`、`stages.session_scope`、`backends` 与回执完整文件锚点要求一致；P6 先 reviewer，按需启动限域 worker | 静态一致，V5/V6 通过 |
| 必要知识跨新会话保留，P5 新事实汇集到 P6；先独立理解当前产物再反查过程记录 | `prompts.prepare_attempt_materials` 复制摘要绑定的已接受回执/短交接，保留代际、实例关系和知识去处；`build_prompt` 明列当前决定、范围、预算与阅读次序；P6 逐项核查最终去处 | 静态一致；交接不授予旧来源资格或新批准 |
| 框架候选由 worker 提出、reviewer 独立确认，Supervisor 按稳定 ID/根因查重和决定；源缺陷与来源缺项另行分流 | common/reviewer/response prompt 与 [issues](../../schema_extract/issues/README.md) 一致；`cli.candidate_report` 校验已接受回执摘要、保留双方报告，只有 reviewer 确认且 ID 合法才标 confirmed | 静态一致；未读写正式台账或投递消息 |
| 文本精简、常见明确命名；有效预算/行数、必要引号、大小写、标点与布局由 Template 统一处理，不能靠生僻缩写绕限 | `prompt/common.md`、worker/reviewer 与作者检查入口明确分工；`artifacts._early_check` 接入当前 text_policy 与算法规则，完成稿调用严格 Template；规范化 warning 与超长 error 分开 | 静态一致，V2 通过；末尾草稿中未被推翻的要求未整体排除 |
| 现行字段与依赖语义经同一规则/检查器进入 Extract，包括 Step/stride、Storage/Index/Event、View.region、八动词与 noun 模式、有序 for/if、显式 waiting、Flow 交接及逐 Rewrite Ledger | `snapshots.rule_files` 冻结 DOC0–6、完整 Template Python、starter 与 prompt；`stages.read_specs` 引向实际分区，`schema_for` 加载冻结 Template，P3/P5/P6 不保留另一套旧格式检查器 | Extract 接口静态一致；内部字段/有限图细节见并行 A 核查 |
| Coder 反馈为与 Require/Prefer/Forbid 平级的 REALIZATION.Feedback[]，只针对 Prefer，原约束不变 | README 与 DOC6 已正确；[IMPLEMENT_SCHEMA](../../IMPLEMENT_SCHEMA.md) 原仍写 `prefer[].feedback`，现改为独立分组并写明 Item/Choice/Evidence 及约束边界 | 本轮已修 |
| 作者使用当前规范字段拼写，工具统一规范化不能替代正确入口说明 | [DESIGN_SCHEMA](../../DESIGN_SCHEMA.md)、[IMPLEMENT_SCHEMA](../../IMPLEMENT_SCHEMA.md)、[OPTIMIZE_SCHEMA](../../OPTIMIZE_SCHEMA.md) 的 Wiki、DESIGN 三区、Ledger/Baseline、META_DATA.Source 已对齐 DOC2/3/6 | 本轮已修；未重设计优化/实现流程 |
| 候选保存原稿→全局必要关联规范化→权限/来源核对→受控落地→严格重读→独立审阅；预览不是已落地证据 | `runner.accept`、`artifacts.normalize_candidate`、`reading.stage_check` 保存原稿/修正/最终摘要，比较输入摘要并检查落地字节；跨文件规范化越权则整体拒绝；reviewer 的证据必须对应最终 raw | 静态一致，V3/V4 通过 |
| 固定模型提示从 Python 移入 prompt，Python 只装配动态事实与机械协议 | 实读 `prompts.py` 的 `_documents/build_prompt/render_operation`，并检索全部当前 Python 的 prompt/instruction/description 路径；阶段、角色、恢复、裁决/签收、维护、评估、巡检交接及提醒正文均来自 Markdown。Python 保留的状态诊断、CLI 帮助和合成评估目录不是阶段提示模板 | 静态一致，V5 验证冻结 Markdown 全文精确装入且阅读材料无漏项 |
| 默认 Claude Code qwen3.8-max[1m]，worker/reviewer 均 max；可选 Codex 显式登记模型与两角色 effort；禁止阶段内部委派 | README/EXTRACT/手册/执行边界与 `runner.DEFAULT_CONFIG/create_run`、`backends` 一致；参数在创建前校验，Claude effort 参数/环境同步，Codex 独立命令与原生协议，角色工具权限明确 | 静态一致；未调用真实 CLI |
| 精确 UUID resume，真实退出、唯一原生成功、session、schema/nonce 与 Codex 末条响应共同成立；中断确认后退休，确证未启动保留原会话；原预算不重置 | `agent._session/read_result`、`recovery`、`runtime.verify_execution` 和 `budget` 核实登记身份/当前 scope；不猜最近会话，未确认后代结清保持 fence | 静态一致，V6 验证各 scope 独立；未重跑真实恢复实验 |
| requested/reported 分列，用量按 attempt 去重；Claude 增量/主循环/累计值分开，缓存读写分列，未知不补零 | `backends.NativeEvents.finish`、`audit`、`observation`、`recovery.account` 保留实际计数范围及原生证据，session 累计成本不作为调用成本相加 | 静态一致，V7 通过 |
| 裁决改稿经同一事务后由全新 reviewer 一次签收；当前维护保留固定身份、草稿、预算和未受影响批准，最后仍须当前 P6 | `runner._adjudicated_workspace`、`stages.advance`、`store.verify_approval_origin` 拒绝未签收改稿作为批准；`maintenance` 使用原阶段预算与字段权限，review→至多一次 patch→独立 verify，不替代未完成阶段 | 静态一致，V4 验证裁决补丁走同一规范化与严格重读事务 |
| 发布核精确集合/来源/当前批准与完整目录，原子交付已审 raw；standalone 保留本节点身份并另置，PNG 按需、有限图不证明全部设计 | `runner._final_inputs`、`publication.publish`、`standalone.create_view` 使用完整 Template API、精确 manifest 与冲突保护；standalone 同时绑定来源闭包、整个目录和 Template 身份；手册明确 PNG 非统一发布硬门禁 | 静态一致；本轮未发布、导出或重画图，图内部和四例 PNG 由 A 核查 |
| 移除 Extract 自建 namespace/mount/proc/sudo 隔离，保留真实路径、原子编辑、CLI 只读角色与既有进程收尾 | `EXECUTOR_ALIGNMENT`、`backends`、`agent` 与 `processes._gate_main` 一致：无自建隔离/额外 proc 挂载，Linux 正常观察与 child-subreaper 仅追踪/回收所属后代 | 静态一致；没有恢复已否决方案 |

## 被替代、否决与延期

- 后续最终规则替代了早期 `dispatch.parent/from_parent`、按配置/配对拆大量完整节点、动态 `while`、View.dtype/part、旧字段名及旧动作记法。Extract 未提供兼容路径；旧讨论/归档只保留历史证据。
- 末尾文本草稿不是整体无效：未被推翻的长度/行数和文字原则仍由最终预算表核查。全禁引号、所有 entry 单行、Storage 固定单行等与后续决定冲突的写法不恢复；现行必要引号、固定 Storage 两行和正式字段/必填规则优先。Ledger 的 Baseline/Gain/Cost 均允许 YAML `null`；Ledger 出现时仍须保留 Baseline 键，Gain/Cost 可省略，空值或省略不表示零变化。
- 11-13、11-16 的“Runner/checker/starter/案例尚未对接”是当时状态；11-17 及当前代码已覆盖相应对接。不能据此把当前工具工作再次列为延期，也不能把历史测试数字当本轮测试。
- `schema_wiki` 重新转换、更多正式算子提取，以及用户延期的 optimize/implement 新流程设计继续延期。本轮只修三个作者入口的已定接口，不转写 Wiki、不新建流程。
- 完整真实模型 P1–P6 试跑此前按要求停止；本轮没有启动。GPU、源 kernel 修复、外部比较器或硬件资料缺口不在本轮执行范围；既有来源限制仍须保留，不能用本轮工具检查宣称源实现已验收。

## 本轮验证

七项已有定向测试实际执行，**7 项通过，4.946 秒**。测试进程用临时 `claude/codex` 阻断入口，并将 `AgentCLI.run` 替换为立即失败；无真实模型或 CLI 子进程，无完整 P1–P6 试跑，临时目录自动清理。

| 编号 | 本轮实际测试 |
| --- | --- |
| V1 | `test_artifact_transactions.ArtifactTransactions.test_selection_and_waiting_are_distinct_bounded_decisions` |
| V2 | `test_artifact_transactions.ArtifactTransactions.test_early_checks_enforce_current_text_budgets_without_future_design` |
| V3 | `test_artifact_transactions.ArtifactTransactions.test_normalization_does_not_rewrite_read_only_provider` |
| V4 | `test_artifact_transactions.ArtifactTransactions.test_adjudication_format_patch_uses_same_transaction_and_keeps_authorship` |
| V5 | `test_review_protocol.ReviewProtocolTests.test_embedded_documents_are_exact_current_frozen_text_with_no_missing_read_specs` |
| V6 | `test_review_protocol.ReviewProtocolTests.test_all_stage_role_node_and_operation_sessions_are_independent` |
| V7 | `test_usage_observation.UsageObservationTests.test_claude_resume_uses_native_delta_and_never_sums_session_totals` |

上述测试模块均位于 `schema_extract/tools/tests/`。另对 34 份当前根入口、Extract 说明与 prompt 做本地链接目标检查：缺失目标为 0；这不代替链接锚点或语义审阅。三个产品文件 `git diff --check` 通过。仅为文案做现行 DOC2/3/6 对照与旧字段检索，没有新增测试。

## 实际改动与未决

- `DESIGN_SCHEMA.md`：当前 Wiki.Sota.Schema/Kernel 拼写。
- `IMPLEMENT_SCHEMA.md`：独立 Feedback 分组、对应对象与权限边界；Wiki、Require/Forbid/Prefer、Source 拼写。
- `OPTIMIZE_SCHEMA.md`：当前 Wiki、DESIGN 三区、Ledger/Baseline 拼写。
- 本报告。

本代理范围内无已确认而未修的新问题。没有修改 Extract 产品代码、DOC0–6/Template、Kernel、Wiki、archive、既有讨论证据或历史 CHANGELOG，没有提交 commit。静态与离线结论不代表真实模型理解质量、全流程试跑或正式提取验收；报告完成后停止写入。
