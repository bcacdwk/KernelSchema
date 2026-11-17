# R3 A：当前运行返回、来源适用性与有界签收

本轮实现保留当前 run 的预算、草稿与独立批准，按具体职责、实例和实际区域来源返回。没有修改 Template 字段规则、Schema Wiki、旧 run 或个人配置，没有 Git commit、真实模型或 GPU 调用。R3 B 的恢复、发布、审计接口已联合接通；真实模型工具能力和完整语义审阅仍交 R4。

## 实际改动

- `runner/stages/store` 删除“必须首次发现新源码行”的返回限制。普通记录和本地化沿当前阶段继续；P1–P4 返回只先打开所属职责，原批准按阶段投影、当前实例和真实内容判断后续适用性。P5/P6 可从真实定义返回 P5 provider；consumer 绑定问题不能借 provider 返回权限越界。稳定 issue_id 在原 run 去重，换引文范围或版本不重授额度。
- P5 开工同时检查 P4 等待和已经形成的实际区域来源；provider 修复尚未独立批准时，旧 consumer 不会因写作等待已满足而重派。相关迟到结果退休，无关兄弟或选型变化不使仍适用的在途审阅失效。已移除与重新纳入的节点保留原草稿、计数和一次决定。
- P2 的选择与 P4 的安排不再通过全局签名粗暴废掉所有正文。签名仍封存在 invocation/原 receipt，P3/P5 另按当前实例、generation 与实际内容核准。保留未受影响的原批准；选择改变后只更新当前文件集合、真实使用者及必要安排，不自动改写 from。
- 接入 `align_source_identity`：只有严格检查后确认自身 raw、实际 origins、同名区域定义、共享 Contract 和完整使用上下文均未改变，才保留批准并登记当前身份。原批准的 identity 不改；当前适用性放入现有 `validated.json.retained_approvals`，节点仅存对应指针。核验同时检查原 receipt、执行身份、阶段决定、实例、原始严格来源记录、当前 checkpoint 和等价依据，拒绝未被 accepted 且成功检查的适用性记录。
- 区分 `source_manifest` 与解析闭包：前者是 root、自身和同名区域链，后者还可能需要 provider 其他区域继续引用的材料。完整 provider 身份或全文件 SHA 的漂移须核当前使用；不会把全部材料或 wait_for 当永久内容依赖。
- 删除 provider 后，当前 merge 不保留其 YAML，consumer 自身旧 raw 留存并交 worker 显式本地化或换用当前已审来源。必要历史材料来自原 immutable checkpoint 的完整、自洽链，独立放入只读 `historical` 材料；只有 baseline 规范化使用它，最终候选仍须闭合当前集合。不会借当前同名 provider 重新解释该历史链，也不恢复历史定义的当前来源资格。
- 所有阶段在原 scope 预留一次 signoff 调用和时间。patch 仍是唯一匹配 `{path, old, new}`，Runner 应用后共用 Template 事务与严格检查，再由全新 reviewer 签最终 raw；P2/P4 签收绑定最终 plan。中断、恢复、返回和追加预算不重置 adjudications/signoffs。
- select 只接受本次实际提供的规范候选。旧候选的版本、当前只读材料、规范化和权限检查在调用前完成，保留原 checkpoint、Runner 准备来源及版本调整记录；批准之后不改字节。签收另绑定裁决 checkpoint，允许刷新无关只读材料，禁止改变裁决目标 raw。
- 进程内检查记录实际耗时、初始允许时间与剩余时间；完成检查后再次结算，超额保留合法候选但不形成批准，不借用后续审阅/裁决/签收额度。Runner 的原 `processing/account` 继续承担恢复与累计结算。

## 当前维护与接口

重写 `maintenance.py`，断开本组文件对 `revision_control/revisions/continuation` 的引用；B 已删除旧模块。当前源、配置、规则和执行器不换绑。保留同一 Supervisor ledger、单写者、排空、发布屏障与有限 review → patch → verify；未完成稿交普通 worker，不以历史批准冒充当前稿完成。排空期间被移出当前集合的请求记为 superseded 并明确阻塞，避免永远卡在排空屏障。

| 接口 | 当前约定 |
| --- | --- |
| `maintenance.submit_request(store, run_id, *, issue_id, tasks)` | tasks 为 `{stage,node,action: review|patch,fields,reason}` 数组；只针对当前 P1–P5。不可变请求与 ledger 内容相互校验。P6 保留正常终审。 |
| `barrier / publication_barrier / consume_requests / mark_drained` | pending 请求先挡新调用和交付；Runner 排空后安装有限任务。普通 pause/cancel/hold 保持优先，预算与已用决定不变。 |
| `Store.source_identity(state, node_id)` | 返回当前 provenance identity；执行准入先调用 `verify_node`，不能仅信状态字段，也不能直接将旧 approval.identity 当作当前值。 |
| `Store.validate_sources(..., graph=None, reviewed_graph=None)` | 验证调用时确实提供且已审的实际来源，再核最新 merge；有漂移时返回完整使用等价依据，否则退休旧结果。材料全集仍只属于 invocation 输入。 |
| `Store.refresh_approvals(...)` | Runner 合入后复核现有批准，沿用 `validated.json` 保存当前适用性，失效者保留原稿后独立重审。 |
| attempt 的 `selected_designs / adjudication_candidates / adjudication_checkpoint / baseline_context_checkpoint` | 分别封存调用时集合、当次可选规范候选、签收的原裁决稿、确有必要的历史 baseline。没有旧字段 alias。 |

配套只改 `artifacts/receipts/prompts/reading` 的必要接线与返回/签收阶段提示。`Store.summary` 从登记 attempt 与已有用量 ledger 投影 provider usage；未结调用保留 unknown，Codex root 观测单列。Runner 已接 B 的 `recover_result` 与确认终止后 `handoff_session(..., termination=...)`，scope retry 核明确的 input_fingerprint/executor_digest。

## 验证与交接

使用临时固定副本避免另一代理修改执行器时破坏冻结身份；命令使用 `/opt/venv/python3.13.12/bin/python`。33 个不同用例在本轮经过验证：新 `test_current_returns` 11 项、R1 产物事务 12 项、R2 正常主链 7 项，以及 B 的五节点真实发布、原生离线全链归档和真实临时退出的单 scope 恢复各 1 项。最后的原批准追溯加固针对来源漂移、P2 签收、provider 等价/实际变化、select、在途选择变化、两条真实交付再复验；维护额外覆盖未完成旧稿和排空期间范围变化。具体命令、结果与后续小改复验见 `R3_A_checks.txt`。

这些测试使用真实 Template、checkpoint、原生离线 CLI/隔离、恢复与发布门禁；确定性语义回执不证明合成代表具有真实独特机制。未运行正式 Extract、批量 Wiki 转换或 GPU。R4 仍需真实后端工具/会话能力验证、完整动态 schema 与语义样本验收，并按最终当前协议处理其余旧 fixture。进程内 Template 检查为同步执行，超额在提交门结算；本轮没有宣称它提供硬实时抢占。

另有一个明确未闭合的字段归属边界交 R4 A：已有非 root 后，P1 更正 `META_DATA.Name/Describe` 会被整组规范化的任务元数据一致性门拒绝；P5 同时禁止修改这些字段。需要为仍属同一固定任务的元数据纠错确定由 P1 负责的受控同步与复审出口，不能靠放宽普通 P5 写权限或自动当作新任务来处理。本轮保持原门禁，未偷偷同步文件。最小复现及两侧拒绝见 `R3_A_metadata_boundary.txt`。

本轮代码与报告已完成并停止写入；上述字段同步边界、真实调用和其余现行测试整理交下一轮。
