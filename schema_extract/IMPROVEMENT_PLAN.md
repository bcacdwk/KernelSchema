# 进行中 Schema 的连续修订计划

本文件是已在 2.4.0 完成的设计与历史工程交接，原实施基线为当时未提交的 2.3.0。当前操作以 [Supervisor 手册](README.md)和[工具协议](tools/README.md)为准，问题决定见 [TRIAGE](issues/TRIAGE.md)。下文的代理分工、暂停真实运行、“不提交 Git”等均为当轮实施约束，不构成后续任务授权限制；具体完成证据保留原记录。本文件不加入 snapshots.rule_files。

## 目标与已定边界

- 只处理尚未完成的 extract：P1/P2 草稿、P3/P4 返工、P5 部分完成、P6 尚未批准。请求时 ready、published、已有 export 或发布 rename 已提交的任务拒绝进入；不开发已完成产物迁移。
- 默认保留现有稿件和有效进度。未知意味着待复核，不自动回 P1；明确的问题才允许修补或局部重提取。来源字节、输入角色绑定、任务目标和原 config/check_policy 不变。
- 进度连续采用调用边界切换：修订请求先拦新派发和发布，在途调用按原上下文完成、结算；然后切换规则。首版不运行新旧语义上下文并行调用，不终止健康兄弟来加速切换。
- 保留 P1–P6、独立 Reviewer、阶段全文审阅、结构化回执、单写者、checkpoint、依赖校验与原子发布。用小的维护待办补充接续，不另建工作流引擎。
- 每个独立实施任务使用新的 xhigh 子代理；主代理负责设计整合和差异审阅。中间不跑测试或逐步验收，最后集中少量必要场景。已有未提交改动不得覆盖，不提交 Git。
- 当前真实 .schema_extract/v2 仍保持暂停；本轮只实现和隔离验证，不启动真实模型、提醒投递或 GPU，不修改历史回执、快照或已发布目录。AGENTS.md 保持原样。

## 两个设计角度的整合结论

运行一致性角度要求：同 run 追加冻结语义上下文，原始证据与当前适用性分开；请求/drain/切换/恢复均复用现有控制账本与 Runner。语义角度要求：影响闭包只产生复核候选，父 Design 是复制来源，不能因祖先字节变化就覆盖孩子全文。两者共同指出 start_node 和 install_plan 会丢弃节点自己的草稿，必须与接续一起修正。

普通中断继续 resume；纯执行器修复继续 upgrade-runtime。本轮为语义修订增加独立入口，并使 continue 不再默默触发全量清批准。基线 runtime.INPUT_POLICY 只读原锁；2.4.0 已接入明确的语义规则上下文，prompt/checker 实际使用目标规则。

## 一个计划，三个独立决定

| 产物动作 | 批准处理 | 下一步 |
| --- | --- | --- |
| keep | 已批准内容有依据才采纳到新上下文；未完成内容只保留草稿 | 恢复原 phase/node/role 或等待依赖 |
| review | 暂挂当前有效性，保留旧批准及完整旧稿 | 直接 Reviewer，符合即结束该范围 |
| patch | 保留原稿，仅开放有证据的文件/字段 | Worker 修补，再由独立 Reviewer 审完整相关内容 |
| reextract | 明确失效的范围重做，其余内容和旧稿仍留存 | 对应 Worker，再 Reviewer；不隐式扩到全 run |

- 一个 run 可组合以上动作。字段范围限制写入，不能缩减当前阶段 Reviewer 的全文审阅职责。
- 修订计划绑定完整目标规则差异、具体 issue/证据、drain 后 frontier、所有已存在阶段/节点、原批准与草稿身份；机器生成摘要/依赖身份，模型负责动作、理由和定位。
- 未开始内容保留 pending/waiting；缺批准的 keep 不授予批准。未列清或依据不足的已有结论进入 review，不能默认为兼容或 redo。
- 复用 compatibility 的实际字节报告，支持 issue-only（规则未变）与规则变更。Supervisor 或现有独立 Reviewer 可提出计划，不固定新增一轮全量模型审阅。
- 计划本身不批准修改后的内容。旧批准适用性可有据采纳；有实质修改或原审核未覆盖的新要求，必须形成正常当前 Reviewer 的新批准。

## 影响与接续规则

| 直接变化 | 必须考虑的范围 | 默认 |
| --- | --- | --- |
| P1 共享 Contract/任务事实 | P2 选择域、P3 root、全部 P5 与共享前提的相容性、P6 | 保留全文，复核前提，只有错误字段才改 |
| P2 代表路径 | P3、P4 与实际相关 P5 | 完整旧设计作候选，确认路径变化才重做对应设计 |
| P3 root Design/when | P4 计划适用性、旧/新复制后代；when 的选择覆盖 | root 修补，后代先 review |
| P4 计划/分组/选父 | 旧/新树差异节点及后代、选择互斥/覆盖相关兄弟 | 保留节点正文，复核身份与复制适用性 |
| 某个 P5 的 Design/Realization | 本节点、实际复制风险后代、同一缺陷或条款涉及的节点 | 本节点修补，无关兄弟 keep |
| 回执/检查门禁，产物未变 | 失败调用及该门禁实际覆盖的结论 | 定向重新校验/Reviewer，不先发重写 Worker |

- plan_impact/affected_nodes 计算候选闭包，不直接代表重写集合。跨兄弟的共享 Contract、条款或选择覆盖关系须由计划显式覆盖。
- root Design 不等于全树共享 Design。子节点有自己的完整 Design；父变更时更新只读祖先输入，保留自己的文件并复核其适用性。
- P1 未完成保留非空 Contract；P2 未完成保留 P1 结论与 dispatch 草稿；P3/P4 保留候选、反馈和当前角色。P5 已写节点优先自己的 draft/approved checkpoint，不能仅保留版本号后再从父复制。
- P1–P4 使用现有局部 workspace；变更合回完整工作目录，不能拿仅有 root/dispatch 的 checkpoint 覆盖整树。
- P6 修订结束后在当前上下文完成最终审阅；保留 p6_starts 和已用单次决定。若修订请求先于在途 P6 完成，旧 P6 结果可留存，但请求屏障仍禁止旧结果发布。

## 身份、原稿和批准的持久模型

- 保留 run_id、原 inputs.lock.json、input_fingerprint、原 snapshot 和所有旧 attempt/checkpoint/receipt 字节。新增 state.semantic_revision 与追加的 semantic_revisions；缺字段是原语义上下文。
- revisions/ID 封存原状态/frontier、目标真实规则快照、完整差异/独立计划依据、原来源身份和执行器身份；先完整写入并封存，再原子更新状态指针。
- state.maintenance 保存当前修订、待办、原正常游标、每个 scope 的接续稿、动作进度和完成记录；不要复用 P6 revision_scope 承担整套维护。
- 新 attempt、receipt/validated、checkpoint 和批准适用性绑定 semantic_revision 与维护 task ID/purpose；旧对象缺字段保持旧格式含义，不回写。每个新调用始终使用自己登记的上下文。
- 统一 resolve_context(store, state, attempt=None) 返回原 source/config lock、semantic rules lock、executor lock、上下文身份。三个角色分开验证；不合成一个顶着旧 fingerprint 却替换 rules 的伪输入锁。
- prompt/phase/spec、YAML parser/checker、receipt 语义校验和所有 rules_root(state) 路径改用明确上下文。source 引文继续绑定原来源；实际新规则必须生效。
- verify_approval_origin 验原 receipt/Runner gate/不可变内容；verify_effective_approval 再验当前原生批准或 adoption，以及当前内容/依赖。Store.verify_approval 的执行准入默认走 effective，历史 audit 可按 origin。
- adoption 绑定真实旧批准、旧/新上下文、独立影响判断、当前 scope/文件/依赖摘要；不能为未获批准的草稿生成 adoption，也不能只靠 keep 标签放行。
- effective_nodes、final_eligible、prepare/accept、P6、发布共享有效性判断；旧原始证据可读，不代表当前可用于解锁后继。
- 语义修订与 runtime binding 两类历史都保留并可连续使用。新的 runtime 升级须以当时有效语义上下文评估；不再退回 revision 0 的规则作兼容依据。用一个上下文选择入口避免各模块各自拼锁。
- 增加明确 capability/state format，使旧冻结执行器拒绝新语义状态。当前执行器不支持请求屏障时，先按已有安全边界采用支持能力的运行层；不能向旧在途 Runner 注入它不认识的状态。

## 请求、切换和正常推进

1. request：CLI/监督器登记稳定请求 ID、目标规则快照与原因，使用现有 supervisor 控制事务；请求入账即拦新派发/启动/发布，pause/cancel/hold 继续优先。
2. drain：原 Runner 完成原上下文的在途调用、合法草稿保存和去重结算。未知终止保持 fence，旧调用仍受原 timeout/deadline。不能靠文件缺失认定已结束。
3. prepare：在 drain 后生成/核对具体修订计划。意图不绑定申请时易变的全 state_sha256；如早期计划涉及的稿或依赖变化，更新相关判断，不把兄弟正常完成当永久失败。
4. apply：同一 writer lock 下封存计划与 before state，原子切换当前语义/执行身份及 maintenance。目标代码确需更换时旧 Runner 正常退出，由同一已授权运行意图接续新冻结代码；暂停任务只登记不启动。
5. maintain：按 P1–P4 和父子就绪顺序执行计划中的有限待办；复用 prepare_attempt/回执/Checker/Reviewer，不走原 _approve 的无条件后续全阶段推进。
6. resume：待办和适用性闭包完成后恢复原未完成工作游标；原 scope 已由维护完成时选实际剩余工作。不能只恢复过期的 stage/role，也不能直接标 ready。最终 P6 仍负责正常交付。

- 一次维护任务用有界的 review →（发现问题才 patch → verify），或 patch/reextract → verify；keep 不启动模型。仍未通过则保留稿与具体问题，不自动无限返工或回 P1。
- 活动维护不能并发 runtime upgrade/另一个语义切换；重复同内容请求/计划幂等，同 ID 不同内容拒绝。第二次修订在前一次结束或安全暂停后明确追加。
- 所有角色在语义切换后重建会话并记录 handoff，保存稿件和反馈，不给 Reviewer 注入“以前已批准所以应通过”的提示。
- plan_revision 只因实际选择/结构变化更新；节点 generation 只更新受影响范围。迟到旧结果按原 attempt/usage 结算，已接收的幂等返回，过期上下文或依赖不得重新安装批准。
- 发布和 export 在 pending/draining/maintenance/有效性缺口期间均受同一锁内屏障；不能只在命令入口检查一次。

## 原有计数与单次决定

- calls、elapsed、deadline、revisions、adjudications/signoffs、reopen_requests、p6_starts、removed-node 账目和 budget_extensions 不重置；登记/keep/本地身份核验不计模型调用。
- 实际维护 Worker/Reviewer 用原 scope 计调用、时间和 usage。维护入场不是领域 reopen，只复核也不是领域退修；有限维护动作单独记录进度，不走已消耗的 adjudicate/signoff 通道。
- 相同 target diff + scope + issue/evidence + action 去重，不能换随机请求 ID 反复获得同一次维护动作。新的规则/证据可以形成新修订，记录它与原原因的关系。
- 原单次决定不重新授予；恢复原流程也不能重发中断的已用裁决/签收。维护自己的补丁必须预留独立 verify 的调用/时间，不侵占原领域预留。
- 只对实际要执行的 scope 检查剩余额度；不足保留进度并报告所需追加，沿现有显式预算接口，不因所有 P1–P6 都要重跑而误判。

## 崩溃与取消

- 封存记录先于 state 指针；未被指针引用的记录没有执行权。state 已提交但控制账本未确认时，按 revision/task ID补确认，不重复代际、计数或 handoff。
- 登记前取消可恢复旧正常游标；生效后取消仅停止维护，不倒退 before-state/用量。以后修订继续追加，不改写原锁或历史结果。
- 原旧门禁拒绝的 response/result/rejected 保留；重放仅作诊断/计划依据。完整候选可供新的必要提交/审阅使用，不改旧 nonce、不回填 accepted_receipts 伪造成功。
- 本轮不补外部任务材料、改输入域或修 kernel；真正需要这些变化时输出明确来源/要求问题，不作为维护兼容放行。

## CLI 与模块边界

- 保留 resume、resume --scope、upgrade-runtime 的明确职责。增加 revise 的 request/prepare/apply/status（可用简洁子命令），都以同一 run 为对象。
- continue 的默认入口改为连续修订准备/采纳，缺计划时返回待评估的具体计划，不再自动清批准退 P1。旧全量 successor 仅显式 full-restart + 非空理由可用，并同样只限未完成任务。
- 请求/准备/采纳的退出 0 表示对应动作完成或已登记，不等于 P6 交付；无需用户逐步确认，Supervisor 在既有授权内完成计划和执行。
- 数据/context/原始与有效批准在一个小模块集中；维护任务选择/完成与原 stages 分开小分支；各入口只调用共用函数，避免第二套状态账本和重复验证。

## 实施步骤与交接

| 步骤 | 负责内容及边界 | 状态 |
| --- | --- | --- |
| R1 | 修订计划/上下文/采纳数据与验证；Store、runtime、receipts、prompts 的必要身份和规则读取；先定准确 API | 已完成；上下文、origin/effective 与规则读取已集中验证 |
| R2 | 原稿保留、有限维护调度、reviewer-first、字段边界、正常游标接续；runner/stages/Store 的调度部分 | 已完成；保稿与完整树接续已集中验证 |
| R3 | 请求/drain/启动屏障、CLI continue/revise、发布和 audit；与 R2 约定 Runner 接口 | 已完成；请求、CLI 和发布/启动屏障已集中验证 |
| R4 | 文档/issue收口和最后一组集中验证，修实际集成问题，更新版本 | 已完成；2.4.0、5 个组合场景与 3 项原有回归通过 |

每个任务用新 xhigh agent，读取本文件与上一步交接即可；一个文件一个当前写者，主代理安排交接。改动中的公共 API 必须先简短告知后续任务；中间不跑测试。主代理负责关键取舍和最终差异审阅。

最终只选少量组合场景：P1/P2 未完成保稿；P4只复核接续不回P1；P5局部patch/父变更保子稿/兄弟保留；Contract未知review且当前新规则实际生效；活动drain/重复或崩溃/旧上下文拒绝/P6屏障与原计数。复用现有fixture/local stub，不运行真实模型或完整算子批次。

## 完成标准与后续范围

未完成 run 能在同一持续记录中采用新表达要求；无关阶段/节点可有据保留，有问题的范围基于自己的旧稿修订，必要复核完成后继续原工作。未知不自动生成空稿或重跑P1；所有新旧依据可追溯；真实源/外部要求及已完成Schema迁移保持本轮之外。

两个分析者均未发现当前必须由用户追加决定的产品取舍。本轮采用上述安全边界及原预算政策；若实施发现这些原则冲突，先给出具体冲突再问，不为普通工程选择另设审批。

## 2.4.0 集中验收记录

`tools/tests/test_semantic_revision.py` 使用临时 Store、固定 tiny source、合成历史批准和本地检查替身，按 5 个组合场景验证实际 request/prepare/apply/prepare_attempt/accept_result 链路：P1/P2 草稿及计数保留；P4 reviewer-first 且原阶段批准/完整树保留；P5 局部 patch→verify、子 review 保自己正文、兄弟 keep；目标规则 prompt 与实际 checker、原始/有效批准及 runtime 交错 audit；请求排空、幂等、原 deadline/单次决定和旧 P6 发布屏障。原始合成事件未提供 provider 用量时 audit 明确报告 partial，不伪造真实模型证据。

集成修复包括默认 continue 成功生成待评估计划返回 0/delivered:false、同一个维护 patch 在原余额内接受合法 continue，以及 audit 对先按 SHA 登记、后按 JSON 读取的同一证据正确解析并复核原字节身份。5 个组合场景均通过；另复用 full-restart 原账目/追加幂等、并发兄弟范围恢复、Reviewer 无历史暗示 3 项回归。未运行完整测试矩阵，失败只重跑相关组；JSON、相关 Python 语法和 diff 空白检查通过。

真实 `.schema_extract/v2`、旧回执/快照、已发布目录均未改写，真实旧 run 仍暂停。本轮没有运行模型、提醒投递或 GPU，也没有迁移已完成产物。历史 2.3.0 改动和原账目保留，AGENTS.md 未变，不提交 Git。
