# 第一轮 max 审计：运行、证据修复与保进度续接

2026-11-19，北京时间。审计代理：`/root/issue_r1_runtime_max`，`gpt-6-astra / max`。基线 `24d2b3d3ca9d4907c4e7efd9607cf78cfb1a52b2`；对象是当前实际工作树，而不是旧报告中曾经存在的实现。

结论：四项局部修复本身已基本足够，应保留。主要改进集中在 `progress-preserving-continuation`：当前续接预检把“保留进度的真实性”与“产物已经完成”混为同一个门槛，仍会拒绝合法的未完成工作；P6 draft 的保存与旧指针修复也需要从单次现场路径收敛为一致的候选稿规则。另有一个已复现的执行器快照并发复用缺口，以及两类过时的离线测试预设。无需重建迁移、自动重审或新调度系统。

## 现场与授权边界

- 已读 AGENTS、两个 extract README、issues README、全部五个正式 ID、[旧决策审阅](../extract_six_full_20261118/ISSUE_DECISION_REVIEW.md)、[用户续接决定](../extract_six_full_20261118/recovery_reassessment/USER_DECISION.md)、真实交付索引及相关修复差异。旧决定的“其他 issue 等指挥”已被本轮全面优化授权取代。
- `.schema_extract/v2` 只读核对：Sparse 原 run `8f0844f9-f279-4781-a19d-0216b7fc4ef5` 为 `ready/P6/published:true`，48 个 attempt；FMHA 原 run `fb8b3758-92d7-4093-aa18-c2e5028f0a43` 为 `invalid_receipt/P6/published:false`，47 个 attempt；两者 active/unsettled 均为空，各有一个真实 `continue_with_audit_marker` binding。
- Sparse binding 已将 `tcgen05_packed` 指向最新 `66ad31f3…` checkpoint；FMHA binding 已将 `mma_sync` 指向 `d3830a2c…`。这些历史指针、批准、调用账与输入锁不得因本轮源码改进而重写。当前状态目录共有两个已封存 repair authorization，均为 marked continuation。
- 六项逻辑交付、15 个设计成立：5 项正常 P6、FMHA 用户授权小修。取消的 Sparse successor `a362b85a-dce6-4ad4-b38e-3a186dd87c1c` 的 state 尚有历史 running/fence 字段，但取消确认是真实上位事实，不对它恢复或结算以“整理”现场。
- 本代理仅新增本报告；反例和测试在临时 Store、测试函数的内存副本及自建假 CLI 上运行。没有真实 extract/resume/cancel/monitor、模型、GPU、历史产物或 runtime 修改。
- 后续 FMHA/Sparse 专项全文审修已由用户授权另两位 xhigh 执行。本报告不声称完成那些语义审计，也不将它们前置为运行工具改进的许可条件。

## 五项正式 ID 的准确判断

| 稳定 ID | 现有决定 | 当前修复质量 | 第二轮动作与最终归档口径 |
| --- | --- | --- | --- |
| `maintenance-empty-poll-control-lock` | resolved | `consume_requests` 两行只读快路径处理真实根因；请求存在时锁内重读和封印校验仍在。 | 保留源码及三项定向测试。此项无需新增轮询、缓存或计时策略；可继续 resolved。 |
| `failed-node-recovery` | resolved | scoped 消费在清 reason 前保存反馈，与 whole-run 恢复相同；依赖/失败身份、未知终止 fence、控制优先及原次数保持。 | 保留三行补传和幂等测试。修正本轮识别的相邻旧测试预设，不能用同步等待或减弱 fence 迁就测试；可继续 resolved。 |
| `p6-retained-scope-inventory` | resolved | 用 `new_files` 的完整 YAML inventory 校验 retained scope，符合只读 reviewer 的事实；未扩大 root、字段、文件或最终 selected 集合权限。 | 保留 gate 及两个正反测试。P6 最新 draft 的相关修复属于续接/保稿链路，不能用此项 gate 已通过代替验收。该 ID 可继续 resolved。 |
| `receipt-evidence-repair` | resolved | 已把固定合法文件内的选择错误及模型必填义务漏项交回原角色有限重交；没有代写证据或形成批准。 | 保留异常分类、提示与现有回归。新续接测试补足此类保存草稿的交叉路径；不扩大自动修复到权限、身份、源码或语义错误。可继续 resolved。 |
| `progress-preserving-continuation` | partially_adopted | 同一 run 的冻结执行器绑定、历史身份、新会话、账本和正常最终发布链路已经真实使用；但实现主要验过已完成 P6 候选，尚未覆盖其公开承诺的未完成停点。 | 必须补齐下述窄范围实现。建议仍以 partially_adopted 归档：adopted 写明当前工具故障续接完成，remaining 明确源码/任务/语义规则迁移不支持，不把范围外主张写成待实现承诺。额外产物审计状态另据专项实际结果记录。 |

### 已足够的四项局部修复

空维护快路径读取的是原子保存的 Supervisor ledger。若它恰好读到“无请求”后才有新请求，该次返回只影响观察时点；派发前 `runner.py` 的最终 ledger 门禁、发布前 `publication.py` 的 `publication_barrier(..., ledger)` 仍会阻挡新调用/发布。因此不需要在快路径再取一次锁，也不能删掉最终门禁。已有测试覆盖持锁时空轮询立即返回、预读后状态改变、封印身份不符；足以验收这个很小的改动。

scoped 补反馈只改变失败责任的交接材料。`last_recovery` 与“先 state、后 ledger”的幂等提交关系保留；同一请求重放不会重复添加 feedback。健康兄弟、原 calls/revisions/receipt_repairs、批准、依赖变化和 pause/cancel 的原边界均保持。没有必要为了三行一致行为新增一套恢复抽象；若后续两处逻辑再分化才考虑提取共同 helper。

P6 gate 原来仅解析 root/changed nodes 是性能选择，不能被当作全目录清单。现在 inventory 与局部内容解析各司其职；`test_p6_retained_scope_covers_unchanged_artifacts` 同时拒绝未知文件、root、Version 和受保护 metadata 的显式 scope，修法足够。Version 的隐含更新权限及两个 prompt 归另一组核验，不能混成同一缺陷。

`EvidenceSelectionError` 的抛出顺序正确：先确认路径、文件和 SHA，再判断行号、quote_offset、空摘录。`ReceiptFormatError` 只请求原角色重新作答；次数来自原 scope、原精确会话，原失败不进入 accepted_receipts。`_recover_call` 对 worker 只保存通过正常范围/来源约束的候选；reviewer 仍只读。真实有效 `revise/blocked` 继续真实阶段流程，维护、adjudicate、signoff 没有获得额外自动次数。无需继续扩大异常归类或增加自动选行/补结论逻辑。

## 续接的根因与目标设计

### A. 将不可变身份核验与完成度分开

当前位置：`executor_repairs.py::probe` 第 248–270 行、`candidate_drafts` 第 207–245 行。

已实际复现两个反例，均由现有 `StageFlowTests` 临时 fixture 产生，不修改历史运行：

1. `create()` 得到合法的初始 P1 worker checkpoint，调用 `probe`。当前直接报 `current candidate fails target preflight: root.META_DATA.Name: expected a string; ...`，后续列出尚待填写的 Contract/硬件占位。这些是正常未完成稿，不是源码、checkpoint 身份或权限失效。
2. 两节点流程正常通过 P1–P5，在 P6 reviewer 授权后保存一个合法、已 accepted 的 worker `continue`。状态仍为 P6 worker，checkpoint 已推进，`candidate_drafts` 却报 `P6 candidate is not the checked result of an accepted worker completion`。仅 `complete` 的限定阻断了真正需要续接的中途进度。

同根静态缺口：receipt repair、CLI/中断恢复通过 `_draft` 保存的安全候选不一定有 accepted receipt；P6 reviewer 再次 revise 后当前授权范围可能已改变，最新 checkpoint 的 producer 仍是上一轮 worker。现函数还要求 producer 的旧 scope 等于当前 scope，因此即使 draft 已正确也可能拒绝。不能通过篡改 decision、复制旧批准或清空 scope 绕过这些问题。

目标是一个很小的“续接前沿”核验，而非跨版本内容迁移：

| 检查层 | 必须阻止 apply 的硬前提 | 可以保留并报告的未完成诊断 |
| --- | --- | --- |
| 执行/输入身份 | 当前支持格式；同 source/config/语义规则；完整目标 Python 差异和冻结身份；原 binding 链与历史 attempt 身份匹配 | 没有新语义审阅，不填“已证明兼容所有批准” |
| 控制与账本 | 单 writer；无在途/未确认/未结清调用、活动启动、控制/hold/retry/维护屏障；未交付；提交前重读相同依据 | 先前失败、调用数耗尽或真实 blocked 仍保留，换绑不自动清除 |
| 已保存内容 | 引用 checkpoint 可验证；文件集合/类型/字节、producer/base 和已登记证据匹配；保存动作未越原阶段文件/字段权限；已存在批准的 origin 真实 | 合法 starter、worker continue、回执补正稿、经确认终止保存的未批准草稿；不能因尚未完成而要求从 P1 重跑 |
| 正常阶段验收 | apply 不生成批准，不改原失败 decision、不扩大 scope、不退款；后续原阶段和 P6 照常执行真实 gate | `check_stage_snapshot` 的完成度诊断可列进 preflight，并说明对应责任尚未完成；它不是 apply 的完整稿批准前置 |

实现顺序应是先核完整性/来源/范围，再收集完成度诊断；不能简单把所有异常 catch 后继续。身份校验、读取可信依据或权限核验本身失败依然必须拒绝；目标执行器不能启动或无法运行核验也必须拒绝。未批准候选的完整稿 checker 诊断不等同上述错误。一个被换绑的 blocked/call-limit run 应保持该状态，普通 resume 原本不能绕过的真实 gate 仍不能绕过。

核验材料复用当前 checkpoint、`attempt.json`、`accepted_receipts`、`attempt_usage.evidence/result_checkpoint`、`validated.json`、`repair.json`、`recovery.json` 和 normalization 记录，不增加外部采用报告、批准迁移表或语义 compatibility 引擎。对已有初始/准备 checkpoint 按其原本类型核验，不强造一个 worker producer。对有 producer 的稿，验证真实 attempt/parent、实际保存的 manifest 和相应保存证据；是否 accepted 与是否可作草稿保留是两件事。

P5 的根游标常为 root，不能仅检查 `state.checkpoint/state.node` 就声称所有节点 draft 已核；所保留的各节点 draft 指针也要验证。此核验只证明保留的候选来源，不替它们补批准。已存在 approved 节点的 origin 验证继续复用 `Store.verify_approval_origin`，不在换绑时额外重跑模型语义审阅。

### B. P6 最新节点稿只用一套保存规则

当前新增代码在 `runner._draft` 和 `stages.advance` 各实现了一次 P6 分支 draft 更新；这是必要修复，但长期保留两份独立逻辑易再次漏掉一种保存路径。`candidate_drafts` 又只看最新 producer 相对其 base 的 diff：旧执行器下如果先改 A、再改 B，当前完整 checkpoint 同时有两份新稿，原 A 的 draft 指针仍可能保持旧值。

另一个静态可确认的遗漏是 P6 adjudication patch：`runner.accept_result` 为实际裁决改动创建新 checkpoint，但 `stages.advance` 的新增分支位于 `if role == "worker"` 内，reviewer/adjudicate 不会同步被改分支的 draft。它随后需要真实 signoff；保留这份未签收的新稿同样不能退回旧节点正文。

第二轮宜在 `stages.py` 提供一个纯粹的小 helper，处理“已验证的 P6 changed YAML → 节点 draft/last_draft 指针”，由普通 worker 落稿、实际 adjudication 编辑和 `_draft` 共用。它不负责批准、generation、scope、计数或流程迁移。传入实际通过权限检查后的改动文件集合；不要依赖模型自报 affected_nodes 来定位文件。

旧指针修复只在 prepare 的可审计划中计算、apply 原子安装：比较当前完整 P6 候选内**所有选定非 root 文件**与各自保留 draft 的实际字节，所需更新须有前述原始保存依据。不能只修最新一次 diff 中的节点，也不能因当前 revision_scope 已换成下一轮授权而遗失上一轮稿。可沿当前 P6 checkpoint 的已有 parent/producer 关系追到所需依据，范围止于当前 run 的登记 checkpoint；不引入跨 run 导入或通用图迁移。

只更新确实落后、且当前候选真实包含的新稿指针。原 approval/approved_sha256/source_context、历史 checkpoint、calls、accepted/failed receipts 均不改。节点后续重审依然审核最新未批准稿；普通 P6 继续直接使用当前完整候选。不能用 `merged_checkpoint` 的旧批准树覆盖未批准的新字节。

必须包含 `continue`、receipt repair、已确认中断保存稿，以及连续两次不同节点 P6 改动后重新进入 P5 review 的验收。现有“已 accepted complete → 一次 root 维护 → 节点复核 → P6”组合测试有价值，但不足以替代这些路径。

### C. 执行器快照的并发与中断仍复用现有持久化约定

当前 `freeze_executor_patch` 使用内容寻址且先 seal/sync 再 rename，方向正确；然而它未复用 `freeze_inputs` 已有的“另一个同内容发布者先成功”处理。实际在临时目录让两个线程都到达同一目标 rename 后放行：一个成功，另一个 `OSError [Errno 39] Directory not empty`。没有数据损坏，但正常同输入并行准备会无谓失败。

修复应沿现有 `freeze_inputs/_reuse_snapshot` 的严格原则：只在目标已存在、lock 和全部 manifest 完全一致、无链接且已封印时复用；不同/不完整/可写 winner 必须拒绝；完成必要父目录同步。不要靠吞掉所有 OSError 或再加全局锁。可以提取极小的共同发布/复用 helper，也可以复用现有代码的有限验证逻辑；不建立新快照框架。

显式 `--executor-tree` 应与输入冻结同样核验本次读取期间没有漂移，避免在准备时从仍在编辑的目标树拼出混合版本。目标树仍要求完整的原语义材料，非 Python 字节有任何变化即拒绝；文档解释应明确“从原 frozen rules 构造仅含已审 Python 修复的完整树”，不能静默忽略工作树中的新 prompt/Template。

保留当前 plan/binding 的提交设计：prepare 不授权运行；binding 封存但未被 state 引用时没有权力；state 更新后同 plan/authorization 重试幂等。当前提交前/后的故障注入测试有价值。补一次控制请求在预检后到达，以及 state 提交后重新加载/二次 repair 的验证即可，不新增事务日志系统。

### D. 故障标记必须真实且无需重复产品决策

当前 `continue_with_audit_marker` 正确绑定 plan，并随正式 receipt 保存 issue、沿用 checkpoint/批准及待审计点；Sparse/FMHA 的真实使用有效。新授权可直接引用用户已经给出的持久指令，无需每次修工具再向用户提同一问题，也不能为了旧候选协议伪造独立审查。

有两个很小的准确性缺口要随前沿整理修正：

1. `_authorization` 只检查 stage 是合法枚举、attempt_id 在 plan 中，没有核对两者实际对应；可把 P6 attempt 错标成 P1。对具名 attempt 必须查登记身份并匹配真实 stage。
2. 当前 audit_points 强制引用一个历史 attempt，无法表达第一个 attempt 尚未登记时的工具故障。允许此类前沿使用明确的 `attempt_id: null`（或等价既有格式内的缺省），stage 须对应封存的当前停点，并写清“未启动调用”；不要为通过校验伪引用一个别的阶段 attempt。

当前有一个旧候选分支 `compatible_executor_only`，会完全跳过 `continuation_audit`。它不是用户选定的“工具故障先完成、随交付留标记”的另一个产品流程。本轮新 apply 宜统一落实 marked continuation；已经真实发生的代码审查可以如实写进授权摘要/证据，但不能代替故障标记。现场两个已封存 binding 都是 marked 模式，不依赖这个候选分支。可在第二/三轮删除未被历史使用的并行授权形式、`--review` 别名及相应测试夹具，保留 `--authorization`；不要为此扩建兼容层。若实现者发现额外真实已封存旧格式，保持其只读核验，不重写它们。

标记仍只存交付元数据，不加 YAML 字段。`pending_user_decision` 表达当时未完成额外审计的真实事实；后续独立专项只能另记自己完成的审计范围，不改原 runtime receipt。正常流程出现新真实 reopen/invalid_receipt 时不能被 marker 或工具续接变成批准，FMHA 的历史就是必须保留的验收反例。

## 未提交 Git 变化逐项归属

以下包括新增文件；不是仅按 diff stat 或 resolved 标签判断。

| 文件/差异 | 根因归属与处理 |
| --- | --- |
| `tools/maintenance.py`、新增 `tests/test_maintenance_polling.py` | 空轮询快路径，保留。与 monitor 的取锁展示不同根因。 |
| `tools/runner.py::_consume_scope_retries`、`tests/test_scoped_recovery.py` | scoped feedback 补传，保留。 |
| `tools/artifacts.py::_validate_scope`、`tests/test_artifact_transactions.py` | P6 完整 inventory，保留。 |
| `tools/reading.py`、`tools/receipts.py` 的错误分类 | receipt evidence repair，保留。receipts 中 execution_binding 的 identity 变更属于续接，不能在精简分类时误删。 |
| `prompt/operations/receipt_repair.md`、`tools/README.md` 的 receipt 段 | 必填判断须原角色核实、不能支持原决定则真实退修/阻塞，保留。 |
| 新增 `tests/test_receipt_repair_policy.py`、`tests/test_proposals.py`、`tests/test_stage_flow.py`、`tests/test_maintenance_receipts.py` | 验证新分类、原次数/会话/稿/批准边界。maintenance 测试次数从 1→2 是遗漏证据也计入 repair 的实质变化，非无故改断言，保留。 |
| 新增 `tools/executor_repairs.py` | 唯一窄入口及封存链保留；重构 probe/candidate provenance/标记验证，删除单一已完成 P6 假设和无标记候选流程。 |
| `tools/snapshots.py` 新 capsule | 保留语义字节不变和独立 executor fingerprint，补并发严格复用/读取稳定性，删重复局部 `import shutil`。 |
| `tools/runtime.py` | 原输入与当前执行器分开、历史 attempt 精确映射、boundary 缓存和 frozen invocation，保留；复核二次 binding 和 evidence 收集，不重建版本迁移。 |
| `tools/store.py` | context 的 source/rules/executor 分离；checkpoint/approval/attempt identity 与新 session scope，保留；同文件另接入 monitor 可选等待回调。 |
| `tools/stages.py`、`tools/runner.py` 其余 diff | binding 进入会话、receipt/validation/approval，以及两份 P6 draft 更新；身份变更保留，重复 draft 规则收敛。 |
| `tools/audit.py` | 允许明确绑定链并报告身份历史，保留；必须验证历史/新 attempt 所指 executor 和归档证据，不能仅让新字段通过 shape。 |
| `tools/publication.py` | 正常发布追加真实 binding/历史/continuation_audit，保留。它是输出证据，不授予发布权限。 |
| `tools/cli.py` | prepare/apply/probe 的目标冻结跳转保留；授权名称和提示与最终支持范围一致；整合另一组只在 monitor 展示边界的小改。 |
| 新增 `tests/test_executor_repairs.py` | 保留真实目标进程、提交前后故障、审批与次数保持、正常发布组合；补未完成/恢复稿、多 binding、并发同内容。改为模块导入 `test_stage_flow`，避免直接导入 `StageFlowTests` 让 discover 重跑整类测试。 |
| `schema_extract/README.md`、`tools/README.md` 续接段 | 保留窄政策，改写为完成度与真实性分开的真实能力；加入完整授权样例和目标树来源说明，不增加决策问卷。 |
| 五个原稳定 issue 与 `archive/TRIAGE.md` | 本轮不改；第四轮根据最终实现/验证更新真实决定。不能沿用“其他 issue 待指挥”的旧暂停口径。 |
| `prompt/operations/maintenance.md`、`prompt/phases/06_final.md` | Version 隐含权限说明属于另一组；本组已核对与 gate 不冲突，保留。 |
| `wakeup.py`、`tests/test_wakeup_retry.py`、Template、三类规则、wiki diff/untracked deliveries | 由另一组/根代理/后续产物组负责。本组已将共享文件接口对齐；不把正式交付的 untracked 状态误认成可以重建的临时草稿。 |

## 全量基线失败的具体归因

根代理统一运行的基线为 **219 tests / 1237.150s / failures=6, errors=2**，日志 `/tmp/kernel-schema-issue-opt-20261119-zbocalki/baseline_extract_tests.log`。全部失败落在下列四个用例；不是本轮需要重写 process fence 的依据。

| 用例 | 基线异常 | 根因和保留的验收目标 |
| --- | --- | --- |
| `CurrentDeliveryTests.test_one_call_audit_archives_setup_failure_and_preflights_original_names` | setup_failure 返回 completed，1 fail | mock `agent.subprocess.check_output` 已不命中。保留真正 launch 前失败的无 events、用量未知、证据可归档、原状态不变、恶意原稿名拒绝。 |
| `CurrentRecoveryTests.test_setup_failure_audit_preserves_missing_stream_and_rejects_forged_proof` | expected cli_error 实得 completed，1 fail | 同一过时 version mock；保留伪造身份/终止/生存者/event link 的所有负例。 |
| `CurrentRecoveryTests.test_setup_failure_keeps_the_previous_ready_session_and_usage_is_read_only` | expected cli_error 实得 completed，1 fail | 同一过时 mock；保留原 session 字节/精确 UUID、未知用量和审计只读性。 |
| `CurrentRecoveryTests.test_unstarted_gate_preserves_previous_session_even_when_result_is_missing` | 3 fail + 2 error | 首个子项把取消后立即 confirmed 当作保证，先收到 termination_unconfirmed 后断言退出，未清 session fence；其余 before_release/exec_failed 子项及最终 resume 连锁失败。 |

用 `PATH=/opt/venv/python3.13.12/bin:/usr/bin:/bin` 对照仍复现，故不是根代理防真实模型的 CLI stub 导致。当前 `--version` 由 `processes._gate_main` 的真实子进程执行；不要再 mock 已移除的父进程调用，也不要把有 gate 的 version 失败伪装成“没有进程/没有 events”的 launch 前失败。

第二轮让前三个测试注入真正的 launch 前失败，例如 `ClaudeBackend.environment` 抛 OSError，保留原安全断言。该替换的临时内存实验中 setup 子项通过；正常 completed 子项也应按真实 gate 结清，因为正常返回同样可能短暂处于 termination_unconfirmed。版本子进程本身的失败/取消已有相邻测试，不必重复造一整套。

第四个测试的临时内存副本只将初始 outcome 允许为预期失败或 termination_unconfirmed，随后有界等待真实 `confirm_termination`，其余原断言全部保留；**四个组合和最终同 UUID resume 全部通过，1 test / 1.427s**。另一个真实假 CLI 实验得到 `termination_unconfirmed/fenced → confirmed → handoff:not_started → same UUID completed`。不要在生产代码中添加阻塞等待来恢复过时的同步返回假设，也不能在测试中直接伪设 confirmed。

直接导入 `StageFlowTests` 造成 unittest discover 重复采集的情况也应顺手消除；这是测试组织精简，不是减少独立覆盖。

## 第二轮文件边界与执行顺序

共享边界已与 monitor/schema max 代理确认，详见 [ROUND1_MONITOR_SCHEMA](ROUND1_MONITOR_SCHEMA.md)。

| 所有者 | 文件 |
| --- | --- |
| 运行组 xhigh | `executor_repairs.py`、`snapshots.py`、`runtime.py`、`runner.py`、`stages.py`、`artifacts.py`、`reading.py`、`receipts.py`、`audit.py`、`publication.py` 及对应运行测试；多数文件只需保留现有改动，不要求每个都再编辑。 |
| 运行组 xhigh，单一写者 | `store.py`、`cli.py`、`schema_extract/README.md`、`tools/README.md`。整合另一组明确接口，不让双方同时编辑。 |
| monitor/schema 组 xhigh | `wakeup.py`、`monitor.py` 的观察/scan 段、北京时间提示、规则/诊断及 MLA/GEMM 限域修订；不改运行恢复控制段。 |
| 根代理/第四轮 | 版本与当前索引整合、正式 issue 决定/归档；不改变历史冻结版本或交付凭据。 |

并行接口已固定：运行组实现 `Store.control_transaction(*, on_wait=None)`，锁外回调 `on_wait(elapsed_seconds)`，只有 callback 存在才计 monotonic；只重试 `__enter__`，正文/退出异常仍不重放。另一组给 Clock 提供 `on_wait(phase, elapsed)`（ack/scan_claim/scan_commit/scan_cleanup）以及 `format_beijing_time`、`clock_display`。CLI 前台 monitor 安装满 1 秒首次、此后 30 秒节流的 stderr 提示；使用另一组的纯输出 helper，不更改 ledger 或 inspect --json。具体展示路径由其报告定义，运行组只负责接入。

建议实施顺序：

1. 先落实小型前沿核验/诊断分层和 P6 保存 helper；用两个已复现反例确保不再要求假 complete。
2. 接入安全保存稿与所有相关节点 draft provenance、修复授权审计点真实性；既有真实 marked binding 保持可读可验。
3. 补执行器 capsule 并发严格复用和输入读取稳定性；保持原提交协议。
4. 修正四个过时测试及重复测试采集，再补必要的端到端失败路径回归；整合 monitor 的共享入口和短文档。
5. 两组各跑定向测试，根代理统一一次全量离线验证；第三轮新的 high 只删重复实现/注释/冗余变动，不删 gate 与必要失败覆盖；第四轮新的 xhigh 验证精简并更新原稳定 ID。

## 必须有价值的验收矩阵

不需要为每个字段写镜像测试。建议用少量组合用例覆盖这些实际失败边界：

- **未完成前沿**：P1 初始或合法中途 continue；P5 一个节点未完成、健康兄弟已批准；P6 accepted continue。prepare/apply 不批准、不改游标和计数，新正常调用从原停点/节点开始。
- **安全稿与真门禁**：P6 worker 有合法新稿但回执漏项、确认中断保存稿、两轮分别改 A/B、实际 adjudication 编辑等待 signoff；修工具后回到节点 review 看到最新字节，原失败不入 accepted；reviewer/root/未授权字段/额外文件越权仍拒；后续 genuine revise/blocked、单次裁决/签收、调用数上限仍有效。
- **身份与历史**：两次连续不同 executor 修复、夹有一次正常新调用；旧 attempt/receipt/approval/checkpoint 原字节不变，新调用带对应 binding、新会话 operation；第二次 plan 对第一次新调用映射正确。审核时能区分原 rules/source 与各 executor，并封存真正使用的 binding 证据。
- **停点和屏障**：现有提交前/后 crash 重试幂等；prepare 后 state 改变拒绝；封存后、提交前插入 pause/cancel/hold/maintenance/retry/launch 变化，apply 拒绝且旧状态不变；未知进程终止拒绝。无需为所有排列写重复大测试。
- **快照发布**：两个同内容 freezer 同时 rename 都可安全得到同一 capsule；已有不同、缺文件、可写或链接 winner 拒绝；目标语义字节或读入中的文件漂移拒绝。
- **标记与发布**：具名 audit point 的阶段匹配、未启动调用据实标识；未发生独立审计不得声称完成；正常 P6 后 receipt 记录真实工具问题/身份/沿用前提，真实新 P6 失败则保持失败，不能发布。
- **相邻已修回归**：维护空轮询与最终派发/发布屏障、scoped feedback 幂等、receipt repair 原预算/UUID/合法稿、完整 inventory 权限负例继续通过。不要扩大到真实模型能力回归或 GPU。

有限风险是前沿保存证据的类型比单一 accepted complete 丰富，精简时容易误删一个保存路径；因此先统一 provenance 与纯保存 helper，再扩大可续接停点。另一个风险是将允许保留未完成稿误写成“checker 失败也能批准”；以上验收刻意把 apply 与后续真实阶段分开。源码/任务/语义规则改变、跨 run 导入、已发布产物迁移和通用旧版兼容仍明确不支持。

第四轮归档时保留历史 2.5.1 修复依据与两次真实续接事实，并按本轮实际变更记录最终产品版本。不要将所有项强写 resolved，也不要留下已经被本轮授权取代的“等待用户指挥”。本组结论不要求再次向用户询问已决定的续接政策。
