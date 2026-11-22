# R4：AKA 可选集成与三仓维护实施计划

日期：2026-11-20；状态：用户确认 D1–D3 后的第一轮计划稿，交 R5 交叉审查与最终整合。
本文只规划后续开发，没有修改实现、运行测试/模型/GPU/服务、安装、提交、fork、fetch 或操作远端；没有读取凭证。
已读两仓 AGENTS、简报/历史/初纲/决策页和六份研究报告，并只读复核关键 AKA 代码与 EXP 已提交补丁。
证据前缀：KS=/home/shaohanyong.shy/KernelSchema；AKA=/home/shaohanyong.shy/atrex-kernel-agent；EXP=/home/shaohanyong.shy/AKA_experiment。
路径后的行号指现有文件；所有新 API、字段、目录、配置名和测试名均为**拟议**，不表示已存在。工期与端到端通过均不作推定。
已确认方向以 [决策页](../03_decision_questions.md) 和历史 H010 为准；本稿不重新提出 provider、V1 或运行时选择问卷。

## 1. 交付顺序与支持边界

固定路线是：MIG-01 保护/定基线 → AKA-01 接缝 spike → AKA-02～09 固定请求 backend → AKA-10 动态编排 → AKA-11 集成发行。
简写：R 是固定 Schema 及范围，C 是准确源码，G 是原生保留的 incumbent，e 是 native episode，A 仅指一次固定请求的实现阶段，不要求新增同名实体。
KS 自有参考 implement、design/一次 optimize demo 独立推进；它们不等待 AKA 动态循环。真实题材料预审从 AKA-02 同期开始。
首发共同候选为 Claude CLI、Triton、单个 kernel.py；AKA 仅声明 single_file 和一个已资格化的原生 Atrex-Bench 任务适配。
小任务与 KS 对齐为新建 row-softmax：FP32 输入、FP64 reference、完整误差式、预分配 out、返回对象/别名/副作用均由 Task 固定，具体数值由共同 Task 真源给出。
AKA 优先复用已有 Agate Execution；服务、worker、镜像、CLI/provider 和真实模型/GPU 资格尚未验证。KS 可另用已资格化本地 GPU 执行同一 evaluator。
AKA 不把任意 KS 本地 harness 当作已支持输入：Light 要求明确 initial_source/evaluation，非 SOL 时要求原生 Atrex-Bench runtime_root（AKA:src/aka/workflow/light/provider.py:145）。
首发不承诺 SOL、source_tree、cuda_manifest、Gluon/CuTeDSL 或多 provider；不支持时在产生模型/执行费用前拒绝，返回准确能力缺项。
FA4 decode 提前做材料/范围预审，保留原栈和 root 子域限制；它是后续真实多节点路径的候选，不强改成 Triton 教学题。
同 R 可跨多个 native episode；编译缺陷、未覆盖 dispatch 和预算结束先保留代码/继续实现，只有阶段完成或有据设计阻碍才准入 optimize。
每个 run 冻结一个顶层循环所有者：KS demo 调 AKA 固定 backend，或 AKA SchemaWorkflow 调 KS planner；任一侧都不能递归调用另一侧的顶层循环。

## 2. 模块、配置、资源及依赖

源码关系固定为 EXP → 固定 AKA fork → 可选 KS submodule；KS 核心/参考实现器不嵌入、不安装、不导入 AKA。
AKA fork 的通用接点不认识 Schema YAML；Schema 语义、Template、design/optimize 方法与公共交接合同仍由 KS 所有。
EXP 只拥有任务集合、实验条件、资料政策、排程、预算口径和结果分析；不在 EXP 再开发 planner 或另一份 Schema 校验器。

| 拟议位置 | 最小职责 | 边界 |
| --- | --- | --- |
| AKA:src/aka/workflow/schema/plugin.py、provider.py | 可选 composition、配置校验、生命周期组装 | 只在选择 schema profile 时加载 KS |
| 同目录 backend.py、bridge.py | 固定请求入口、与 ManagedCampaign/episode 边界的唯一桥接 | 不复制 startup、prepare、finish 或 native loop |
| 同目录 materials.py、prompts/、tools.py | KS 资料投递、固定任务正文、只读查询客户端 | 调 KS API，不另造 parser/dispatch 编译器 |
| 同目录 state.py、projection.py、eligibility.py | 请求/R 与 native 事实关联、结果投影、资格适配 | 源码/测量/晋升仍以 native 记录为权威 |
| AKA:packages/atrex-aka-schema/pyproject.toml | 单独可安装的 schema distribution | 基础三包不增加 KS 必装依赖 |
| AKA:optional/KernelSchema | 固定 KS commit 的可选 submodule | 仅安装所选能力/资料，不递归拉取全部第三方工程 |
| AKA:tests/schema/ | 少量接缝、恢复及 off fixtures | 新增测试位置；当前 checkout 未发现这套测试目录 |

拟议两种 on-profile 为 schema-fixed 和 schema-orchestrated；fixed 只接受固定实现请求，orchestrated 才允许调用 KS design/optimize。
配置仅冻结 KS 根/版本、协议版本、模式、Task/初始 R 引用、支持栈、资料许可、限额及 reviewer 政策；拒绝未知字段和不相容组合。
schema-on 要求明确有限的全局调用/执行上限和请求 episode 上限；公共“剩余为 0”表示禁止准入，不沿用 native 某些参数“0 表示不限”的编码。
off 直接选择原 Light provider/原 Agent/原资源；不构造空 SchemaAgent，不扫描 KS 目录，不校验 KS manifest，不增加 schema prompt/tool/mount/ledger 或模型调用。
静态资源只含固定 KS 包、规则、角色文档、选中参考与工具；运行中 R/结果置于 native 私有运行根下独立 schema 子目录，不进入 bootstrap 可变资源 fingerprint。
资源路径从显式配置或安装包解析，不依赖 cwd、作者 home 或两个同名 tools 包的搜索顺序；变更规则/实现/Task 使用新 run，不重写旧 capture。
现有 capture 会重算并比较 implementation/resources/config（AKA:src/aka/bootstrap/continuation.py:67、:161）；因此 off 行为相容不代表新版可恢复旧二进制捕获。
打包需扩展现有 tools/build_distributions.py:20 的 OWNERS 和 package-data；新 schema 包的资源随包发行，KS 选中资源独立留摘要/来源。

## 3. 与 KS 对齐的最小请求与返回

拟议公开入口为 run_fixed(request) → result，另有 capabilities()；Python/JSON 之一先落地，不先做插件市场、HTTP 服务或通用后台平台。
请求最少固定 Task/验收身份、Schema 字节快照/规则、选中节点与范围、完整 source seed、资料允许集合、预算和可选 opaque continuation。
Task 不能只剩输入 shape：需含 ABI、全部输出/副作用、reference/完整比较器、目标设备/栈、完整调用计时口径和硬门槛。
R 的 raw/dispatch/来源闭包字节身份、设计修订原因和本次范围分别保存；沿用 Template source_manifest/origins/effective_document，META_DATA.Version 不充当全局设计号。
设计交接的权限、全任务结构/语义审阅和缺项由 KS 准备结果给出；AKA 再核对执行 Task，设计 completed 不代替代码的三项验收。
手写、curated、design 和 extract 来源都可经同一准备进入，不要求 P6 receipt；extract 的 skip-hardware/验收缺项须补齐为新实现输入快照，原件/回执保留。
effective_identity 不含 Task/dispatch/规则且受 raw 字节影响（KS:schema_template/tools/design_resolution.py:39、:47）；排版/Feedback 变化留痕，不自动记一次 optimize。
source seed 是完整可校验代码清单：路径、字节/可取得引用、摘要、必要模式与 source format；删除由完整清单相对已声明可编辑集合确定，不能只传增量片段。
首期清单只允许 kernel.py；public-candidate 中的 solution.json 为辅助证据/元数据，不能直接合入 source contribution。
返回分别含最新稳定 C、当前 R 的合格 C、覆盖/缺口、构建、正确性/fidelity/性能证据、实际用量、停止原因和可选续接。
AKA 扩展单列 native episode/stop/promotion、G 的 source identity、canonicalGitHEAD 与原生导出状态；KS 核心不解释 private dataclass、vN 或 campaign 目录。
公共状态采用 completed/partial/blocked/failed；检查逐项 pass/fail/not_run/unknown，性能目标另报 met/unmet/unknown，R5 统一具体字段拼写。
completed 必须满足本次范围、Task 硬条件、规定正确性/fidelity 及要求的测量；较慢但合格 C 可 completed+unmet，G 晋升另算。
continuation 绑定原请求/原后端/代码与累计预算；换 Task/evaluator/规则拒绝旧续接，换 R 或范围形成新固定请求，允许显式迁入旧代码。
固定 backend 不调用 design/optimize/extract；verify_only 是同一入口的验收模式，不开改码会话、不伪造注释变化、不产生一次新晋升。

## 4. 收敛后的 AKA 修改面

下表逐项限定候选文件/既有函数；桥接或通用参数只有获相应工作包验证才加入，不能一次搬入 EXP 的全部差异。

| 修改项 | 现有接点/证据 | 计划中的最小变化 |
| --- | --- | --- |
| 受信任角色/context | contracts/agent.py:199；EXP:assets/aka/src/aka/contracts/agent.py:217 | 可选 role/context_files/scope 贯通 runtime.run、open_session/invoke，默认空值保持原行为 |
| 实际 Agent 投递 | agent/provider.py、agent/runtime.py、agent/native.py、agent/backends/base.py、agent/backends/claude.py | 同一有限字段走首次/重试/恢复/重开；不支持 provider 显式拒绝 |
| 固定任务正文 | workflow/light/long_horizon/campaign.py:_prompt:444；同目录 session.py:_run:145、:259 | 用受信任 assignment 替换自由优化主任务，保留原测量/报告/隔离片段 |
| 资料工具/工作区 | workflow/light/runtime_resources.py；workflow/light/orchestration/episode_workspace.py:119、:164 | 新增只读 schema 工具及外置材料挂载，scope 绑定当前 R，不放入可编辑 source |
| 初次准备/收尾 | workflow/light/orchestration/campaign.py:1186、:2045、:2572；workflow/light/long_horizon/main_adapter.py:30 | 按 AKA-05 决定 never+必要 helper 或受限 V1；C 与 G 各自校验/导出 |
| source seed | task/candidate_workspace/source.py:147；workflow/light/long_horizon/campaign.py:1790、:1831 | assignment/seed 安装事实先持久，再准入 draft；恢复已准入 draft 不覆盖 |
| 指定 episode 边界 | workflow/light/long_horizon/campaign.py:1698；workflow/light/search_driver.py:173；workflow/legacy/startup.py:155 | AKA-01 决定单步 façade 或原 loop 的窄准入/终态接点，保留一份 native 生命周期 |
| 合格结果/G | workflow/legacy/episode_output.py:capture:52；workflow/light/long_horizon/campaign.py:_record_terminal_episode:1101 | 读准确 C/receipts 投影结果，不以 accepted 或最后 vN 指标代替实现完成 |
| fidelity 资格 | workflow/light/long_horizon/campaign.py:_assess_terminal_handoff:1011、:1055 | native seal 后、ABBA 前增加可选资格结果；absent 时零新增行为 |
| durable 资格证据 | 上述 campaign.py:1162、:1231；同目录 promotion_audit.py:33、:78、audit_recovery.py:61 | 明确把资格引用/摘要写入 attempt/audit，恢复传回证据，不只加 active 临时字段 |
| 动态 R/恢复 | bootstrap/continuation.py:161；workflow/light/long_horizon/store.py:76、:99 | 运行产物发布与新请求准入分别持久；在原 active/archive 上加最少关联 |
| 总用量/暂停 | workflow/light/long_horizon/session.py:230、:293、同目录 store.py:85；workflow/light/orchestration/campaign.py:_account:653 | 索引真实调用收据，所有新调用/请求/episode 统一准入，保留 native 限额 |

上表无仓库前缀的路径统一相对 AKA:src/aka/；EXP 路径单独标注。
通用补丁优先参考 EXP@76dabfb 的 explicit-agent-role-context-v1 与 agent-session-private-boundary-r1；逐文件重建依赖/默认值测试。
storage-policy-r2/owner-lifecycle-r3 仅在所需隔离/导出生命周期证明依赖时迁入；condition/cohort、实验私有目录及全套重发布协调留 EXP。
R1 曾发现 41 条净差异、官方 verify 被缓存拒绝及未登记差异（research/r1_aka_inventory.md:43）；这是历史观察，不是当前差异台账或迁移授权。

## 5. AKA-01：第一个可开始的 spike

前置仅为 MIG-01 的隔离源码副本与记录基线；不依赖 KS planner 完成、不需要真实模型/GPU/服务。
产物为一份可运行 CPU fixture、归一化调用轨迹和 ADR；拟新增 tests/schema/test_episode_boundary.py、桥接草稿 bridge.py，文件名均非现成功能。
fixture 复用 native CandidateWorkspace/Store/恢复代码，用 fake Agent、Evaluation、ABBA 及可计数的 setup/finish；不把整个 Campaign 都 mock 掉。
先做一个指定 e 的窄入口 run_episode(assignment, target_episode)，返回持久 EpisodeOutcome；不存在按“这次新跑了一个”计数的语义。

| 最小输入/故障 | 必须观察到的判定 | 失败后的具体替代 |
| --- | --- | --- |
| e 已终结后再次请求 e | 只返回同一终态，无 e+1、无 V0 重建 | 在 native recover 后、while 准入前加入 target 已结算检查 |
| active e 已有 terminal handoff，恢复才完成记账 | 结算 e 后立即返回，不用恢复后的 starting_episodes 再跑一次 | 让原 loop 在 terminal callback 返回 stop，不再用 episode_limit=1 |
| 同时给 max_version/max_episodes；或请求/全局剩余为 0 | 两类上界及请求/全局预算都限制新准入 | 引入显式 admissible(target,state) 检查，不复用原 if/elif 组合表达新语义 |
| pause/预算关闭时有可恢复 e | 允许读取/吸收已完成事实，禁止新 coder/planner/review/GPU 提交 | bridge 的 settle-only 恢复分支；新副作用停在原 active 前提下 |
| 请求 e，但 active 指向另一 e/assignment | 返回身份冲突，零调用；不猜测“最近 episode” | 固定请求先绑定 native active，禁止多个 writer |
| 通过实际 startup/ManagedCampaign 构造入口启动与退出 | lease、record_context、execution、prepare 和 cleanup 各有唯一 owner | 保留原 startup，在 ManagedCampaign 接入受限 episode policy |
| 非晋升 e 后 canonical HEAD 因 vN 记录推进 | 下一 e 取新 HEAD，比较 G 源码仍可相同 | 删除以 A 初次 base_commit 作为永久 base 的桥接逻辑 |

事实依据：max_version/max_episodes 为 if/elif，starting_episodes 在 recovery 后赋值（AKA:src/aka/workflow/light/long_horizon/campaign.py:1718、:1727、:1734）。
通过条件是所有案例成立，且 façade 只经 bridge 复用 ManagedCampaign/prepare/settle/finish，不复制循环或分散读取新私有字段。
若为 façade 必须复制 startup/prepare/finish 或维护第二组恢复分支，**选择有限原 loop 接点**：admit_next(state,active) → assignment/stop，settled(outcome) → continue/stop。
该替代由原 loop 串行回调 schema policy；可选模块不再自建 episode while，调用点限定在恢复后/新 episode 前和持久终态后，Schema 知识仍在模块外。
终态回调先于 valid_blocked 的自动 fresh-episode retry（AKA:src/aka/workflow/light/long_horizon/campaign.py:1948），每次新调用再查暂停/预算，不能由原 continue 绕过。
若两种接法都不能保住身份/预算/恢复，停止动态方案扩展，交付问题定位与固定输入/verify-only 接口；KS 独立交付继续，不能用每 R 重启整场 campaign 掩盖缺口。
此 spike 只决定驱动形态；source 导入、V1 与 fidelity 的完整门禁分别在 AKA-05～09 完成，不能以 spike 通过宣传 backend 已可用。

## 6. 持久化所有权与原子边界

native frozen launch/Task/lease、e/session/draft、Experiment/Execution 收据、canonical 与 promotion audit 保持原所有者，不另建 W/R/A/e/s/C/G 八套账本。
schema 最少新增三类记录：不可变 R/固定请求发布件；native episode 到请求/seed 的关联；可重建的当前请求、latest_C/qualified_C 与用量索引。
拟议 schema/run.json 原子保存已准入请求、待发布/待准入操作、持久 pause/fence、吸收游标与收据索引；native active/attempt 只追加 opaque assignment_ref/seed_ready_ref/eligibility_ref。
保存请求/终态索引时以唯一 request_id、native e/branch、operation/receipt id 幂等吸收；原生 state 丢失或无法解析时核对 archive/audit，不能当新运行归零。
现 active/attempt 用 atomic rename，非 fsync durability（AKA:src/aka/workflow/light/long_horizon/protocol.py:11；store.py:106、:117）；新关键证据使用 core/filesystem.py:43 的 durable_write/replace。
需要加强的 on-path assignment/seed-ready/eligibility/吸收记录必须先写完整内容、fsync 文件与目录再发布引用；不能只在注释中称 atomic/durable。
R 的发布用暂存→完整清单/哈希核对→不可覆盖发布；发布成功不更改 admitted_request。新请求只在旧 native 事务结算、作用域关闭且预算/暂停允许后原子准入。
seed 先写 intent 与 source identity；新私有 worktree 的 link/clean 完成后按 source codec 安装/复读/持久化，写 seed-ready，才 register draft/准入 invocation。
native 当前已先落 exploring（AKA:src/aka/workflow/light/long_horizon/campaign.py:1802）；拟议 on-path 在 preparing 内完成 import/seed-ready 后再转换，恢复先走同一准备 helper；off 沿原路径。
安装完成但尚未准入的崩溃，可从不可变 seed 幂等重装；有 invocation 准入事实后，以已注册 draft 为权威，禁止以 seed-ready 重新覆盖 coder 内容。
准入事实缺失/矛盾而发现 draft 已变化时保留现场并拒绝新调用，不能把未知当“从未启动”。install_source 的内存异常回滚不覆盖 SIGKILL（source.py:185、:199）。
canonicalGitHEAD 记录 Git 当前版本；G 记录比较源码身份与其历史资格。未晋升 vN 也推进 HEAD，下一 e 必须重新取 HEAD，不能把这当 G 改变。

| 中断位置 | 权威事实/恢复动作 | 禁止行为 |
| --- | --- | --- |
| planner intent 后、R 发布前 | 同 operation 的草稿/收据留存；旧 admitted R 有效 | 重新免费调用 planner 或把草稿当发布 |
| R' 已发布、请求未准入 | 保留 R'；吸收旧 e，按明确 next_request 决策 | 仅看最新目录取消旧 R，或把发布等同准入 |
| R' 请求已准入、native e 尚未建立 | 用唯一 assignment 创建目标 e；旧会话已关闭 | 新开旧 R invocation，重复分配 e |
| seed intent 后、安装前/中 | 无已准入调用时完整重装并核 hash，再写 ready | 使用半装代码，借异常回滚冒称宿主崩溃安全 |
| ready 后、draft 创建/调用前 | 核对 seed/draft/准入记录，必要时完成同一准备 | 仅凭 phase 判断可覆盖 |
| coder 已准入/同 e 恢复 | 原 draft/session/checkpoint 延续；固定 assignment 不变 | 重装 seed、换 R 复用旧 session |
| C1 测过，随后改 C2 | 仅复用 subject/source/evaluator 全相符的 C1 收据 | 将最近 pass 贴给 C2 |
| seal 后、fidelity 返回前 | 同资格 operation 结算；未知费用/调用先保留 fence | 未知资格进入 ABBA，或无痕重评 |
| fidelity 已持久、ABBA 未完成 | 复用准确 C/R/scope 的资格收据；Experiment 结算原 ABBA | 重复扣同收据、重复提交未知执行 |
| ABBA 完成、promotion 未提交 | 从原收据恢复资格/比较，再按 native ancestry 提交 | 以 partial seed 替换 ABBA 的 G |
| promotion 已提交、state/返回未落盘 | 从 commit-bound audit 取资格与原费用，吸收终态 | 再晋升或新模型意见改写当时资格 |
| 已提交 audit 缺资格/仅 acknowledged missing | 保留 native 已提交事实，Schema 资格 unknown、阻止自动前进 | 伪造 pass、回滚历史或把人工承认缺档当 fidelity |
| native e 已终结、schema 分类未写 | 用 e/branch + source/receipts 重建同一结果与游标 | 再开 e、丢 partial、按 accepted 代替完成 |
| pause/预算耗尽 | 吸收已完成事实；新调用/请求/episode 禁止准入 | 新 R/新 e/keep 退款或清空已花额度 |
| unknown execution / inflight deadline | 原 Experiment/Execution 先结算；无法确认则 blocked+已有产物 | 盲目重提 GPU、猜测成功或重置 deadline |

普通/未提交恢复可共用 _assess_terminal_handoff；已提交晋升走 audit（AKA:src/aka/workflow/light/long_horizon/campaign.py:1393、:1516），两条路径必须各自验证。
_record_terminal_episode 重新构造 attempt，active 任意字段不会自动进入 audit（同文件:1162、:1231）；资格字段必须显式纳入哈希绑定证据。
原 production prelude 可能在 audit 前重调 reviewer，缓存仅在 Campaign 实例（main_adapter.py:43；orchestration/campaign.py:705、:831）。
新增 fidelity 对准确资格键去重；原 reviewer 若真实再次调用，必须算新成本。settle-only 在暂停时不能借 prelude 偷开审核。
总账索引 native coder/V0/V1/reviewer、KS planner/fidelity/extract、执行的实际收据；累积 receipt 取增量，不再叠加同一 state.tokens/tokens_spent 总数。
每次外部调用先持久 call_id/额度预约；失败、重试、恢复、review 新调用均结算，unknown 不填零。去重对象是同次调用的收据重放，不是内容相同的新调用。
全局调用/Token/执行上限、请求总 episode/active time、单操作上限取最紧约束；首次实现可分配更多请求预算，均从同一总剩余额度取得。
native Token 限额仅在调用间检查（session.py:230、:293）；没有 provider 硬保证时公开单次可能超额边界，以调用上限和保留额度准入，不虚称调用内硬停。

## 7. Coder 的具体执行路径与 V1 gate

宿主先把公共 Contract、root/dispatch、选中有效节点/来源、IMPLEMENT 文档及只读查询入口交给 coder；模型只修改授权 source。
Coder 先列实现范围→输入路径→节点/dispatch 决定→代码位置的短映射；随后贯通适用路径，再补本次必要节点/路由/边界，逐次记录准确 C。
root 用于读共享 Contract 和主干；root.when 不适用时实现所选适用节点，不造 root fallback。dispatch 选定变体须先有明确设计决定，不能自行应用目录笔记。
每轮编译/运行使用宿主 Task evaluator；编译 API/索引/精度实现错误及设计漏落实交 coder 修复，保留具体输入/代码/日志。
设计阻碍须指出 R 中的对象/硬约束及可核对推导、最小反例或真实编译/运行事实；一次超时、写不完或无晋升不触发 planner。
固定 R 禁止自行换算法/数值路径/工作组织/同步方案/实现栈；仅在未规定的代码细节调优。on-profile 关闭 goal widening/自动 conversion，并检查既有 G 的 Gluon 约束。
首发不新增 native terminal enum；budget 到期无合格 handoff 如实保留 invalid_handoff 等原事实，schema 层据稳定 C/进度返回 partial，不能滥用 blocked/pivot 代签完成。
如后续实际交付需要主动交出 partial，再单独引入无晋升含义的 checkpointed 接口；不以该扩展阻塞首条有限续写路径。
首次、handoff repair、限流重试、输出上限重开都沿同一固定 assignment；辅助 reviewer 不继承 coder home，新 R 不复用旧作用域。
role/purpose 与 context 由宿主调用链赋值，不采信模型自报角色或环境字符串；planner、schema coder、fidelity、native 辅助 reviewer 权限逐项固定。
prompt 审查覆盖首次正文、修复正文、WORKSPACE.md/CLAUDE.md、mode/资源指令，移除固定实现任务与自由换设计冲突的指令；不能只追加末尾 policy。
只读 schema read/effective/check/reference 工具直接调 KS；check 禁止默认规范化写回。原始材料/规则运行前后校 hash，挂载权限不等于 fidelity。
普通反馈在外部结果；仅有合规 repository@ref:path 实测证据的 Prefer 替代才成为 Feedback proposal，由宿主按 KS 权限核对后发布，固定 R 不热改。
实现材料只能来自 Task 的资料 allowlist；不向 coder/planner 交私有 evaluator、隐藏用例、额外优化原码、旧 transcript 或未声明 seed 以补结果。
verify-only 复用受信任 execute 的 candidate_source/reuse_completed 能力（AKA:src/aka/workflow/light/orchestration/supervisor_runtime.py:657），补新范围/fidelity，原有完整比较器不变。

AKA-05 先试 never 加所需准备/收尾；以下每行都有去向才允许固定 backend 发布，不能仅设置三个开关后宣告替代 V1。

| V1/原生职责 | 首发拟议处理/最小验证 | 不满足时退路 |
| --- | --- | --- |
| V0 公共 Task/私有 evaluator/基线 | 仍走原准备与测量；重启 fixture 不重复 V0 | 恢复同 ManagedCampaign 准备边界，拒绝换任务 |
| solution metadata | 按 native 格式 helper 处理并固定其身份；不存在且合法时明确记 absent | 必需 helper 写 canonical 时在分配 e 前原生 record 并重取 HEAD；仍不相容则接 V1 |
| 正确但更慢首版 | C 单独完成/导出，G 可不变；fake 慢 C 验证完整返回 | 增补结果投影，禁止降低 ABBA 阈值 |
| baseline pin | never 时明确没有 V1 pin，不伪造 speedup baseline；检查所选输出不依赖该 pin | 需要 pin 的任务转受限 V1，记录 bootstrap adoption 与 ABBA 胜出区别 |
| production/source/dependency gate | 对交付 C 的准确 staging 调同一检查；G 合规/导出另报 | 收缩为实现材料输出，不能宣称 native production 包成功 |
| reviewer/目标栈 | V0 prefetch 和 production 审核真实计费；G=Gluon 与 R=Triton 冲突提前拒绝 | 不通过换栈绕过；当前请求返回前提冲突 |

事实：never 在决策最前 skip（orchestration/campaign.py:1389）；V1 不要求 speedup（:2232），包含 metadata（:2045）和 pin（:2482）；_finish 只检查 canonical（:2583）。
若必须接 V1，改动限定 ensure_framework_baseline、_framework_baseline_prompt/_run_framework_baseline_agent/_recover_framework_baseline、_framework_baseline_external_gates/_commit_framework_baseline：同 R/context/累计预算，提交前 durable fidelity，保留失败源码与 pin 审计。
V1 fallback 本身另过恢复/慢首版/源码资格测试，不能算作已完成的轻接入；若超出当前支持范围，先拒绝该任务，保住 KS 独立路径。

## 8. 可交开发者的工作包

AKA-01 的内容和 gate 已在 §5 固定；其余工作包均要求代码、配置/接口说明、实际验证记录和已知限制一起交付。

### AKA-02 — 共同契约与原生 Task 资格

- 前置：KS 最小 Task/Schema/source/result 草案；不依赖动态 planner。
- 候选改动：拟新增 schema/backend.py、projection.py、单个 row-softmax Atrex-Bench task adapter；调用 provider.py:_load_problem/_campaign，保持三种输入角色分离。
- 产物：同一 KS Task 到原生 public/initial/evaluation 的映射、能力声明；FA4 decode 的材料/原栈/root 子域/必要 dispatch 缺口清单。
- 验证/完成：错误比较器、漏 out 写入/返回别名、副作用变化和不支持 source format 必须拒绝；真实 GPU 前只报材料/CPU 协议资格。
- 退路：无法保留完整 Task 判据则该 AKA task 不准入；不改 Task 降标准、不暴露私有 evaluator，KS demo 继续。

### AKA-03 — 可选模块与 off

- 前置：AKA-01 驱动选择、AKA-02 能力边界；KS 包命名/最小安装接口可读取。
- 候选改动：§2 新 plugin/provider/profile/distribution；现有 build_distributions.py:20；原 profile 不修改默认选择。
- 产物：fixed/on、orchestrated/on 的独立配置 schema，off 为原 provider 的组合图。
- 验证/完成：移走 KS/禁其 import 后，off 可解析/启动；归一化 prompt/工具/CLI/预算/晋升/导出轨迹与固定上游一致，新增 Schema 模型调用为零。
- 退路：去掉早期发现/全局资源声明，延迟装载；不得用空 KS 目录掩盖必需依赖。

### AKA-04 — 角色、固定 prompt 与只读资料

- 前置：AKA-02/03；选中 EXP 通用补丁的逐文件来源映射。
- 候选改动：§4 Agent 调用链、_prompt、session._run、runtime_resources/EpisodeWorkspace；拟新增 schema materials/tools/prompts。
- 产物：受信任 assignment 资料清单、四类角色权限、固定正文/修复正文与资料查询合同。
- 验证/完成：fake Claude 经实际 NativeAgent/投影隔离覆盖首调、resume、重开；伪造 role、caller context 抢占、Schema 写入和跨 R home 混用均拒绝。
- 退路：缺哪种 provider/恢复能力就声明 unsupported；必要 storage/owner 补丁按实际依赖增加，不整批复制 EXP。

### AKA-05 — 初次准备、慢实现及独立收尾

- 前置：AKA-01/02/04；执行 §7 的逐职责 gate。
- 候选改动：main_adapter.prepare_campaign、orchestration/campaign.py 的准备/收尾 helper；仅失败确需时接 V1 所列函数。
- 产物：明确选定的 on-profile 及 V0/V1/helper/pin/C/G 职责表，当前 R 合格代码的导出入口。
- 验证/完成：fake 慢且合格 C 返回成功实现而 G/ABBA 真实；首次和恢复都有合法 metadata/production 输出，真实 reviewer 调用不漏账。
- 退路：受限 V1 或拒绝该任务；无 native production 包时只交已说明范围的实现产物，不改原生事实。

### AKA-06 — source seed 与跨 episode 续写

- 前置：AKA-01/04/05；KS source manifest 编码固定。
- 候选改动：new-episode preparing/link/register 接点、source.py codec/install_source、EpisodeWorkspace.prepare；新增最少 seed intent/ready 关联。
- 产物：e1 未晋升 C1 → 同 R 的 e2 以精确 C1 开始；Git base 从新 HEAD 取，G 源码另绑定。
- 验证/完成：在 active 写入、安装前/中/后、ready、register/准入间进程中断；首次调用只能见完整 seed，已准入 draft 不被重装；solution.json 不能混入单文件 source。
- 退路：首发仅 single_file；无法确认安装/调用归属则受阻保留现场。多文件删除/固定文件测试留 AKA-12，不为首发预扩格式。

### AKA-07 — 资格、精确结果和 verify-only

- 前置：AKA-02/04/06、KS fidelity 方法/有限 reviewer 合同。
- 候选改动：_assess_terminal_handoff、_record_terminal_episode、promotion_audit/audit_recovery；source/receipt 投影，复用 trusted execute。
- 产物：准确 R/Task/scope/C/checker/reviewer 版本绑定的 durable eligibility receipt、C/G 分离结果、零改码验收模式。
- 验证/完成：快但违规 C 不进 ABBA；慢且合格 C 保留；C1 测后改 C2 无旧 pass；normal/未提交/已提交恢复各有原收据，缺资格不自动前进。
- 退路：有限人工/离线 evidence 可接入；无法完成审查返回 unknown，禁止以数值正确代签 fidelity 或以重新评审改写已提交历史。

### AKA-08 — 固定 backend 的完整返回

- 前置：AKA-01～07；一个请求覆盖多个 e，implement-only 始终零 planner 调用。
- 候选改动：schema/backend、projection；将 root/节点进度映射到请求结果，复用 episode_output/Workflow export 事实。
- 产物：能力查询、fixed/continue/verify-only 调用与可读取代码/证据；native stop 和 schema 交付状态分列。
- 验证/完成：缺 dispatch 为 partial；正确慢实现 completed+目标未达；native exit=0/无晋升不能误判；source unavailable 不冒充空 seed。
- 退路：保持有限固定后端，不接 planner；返回最后稳定 C/明确缺项，禁止静默从 G 重来。

### AKA-09 — 预算、暂停及完整故障恢复

- 前置：AKA-06～08；本包是 fixed backend 发布门，不是可延后的动态功能。
- 候选改动：schema/state、native usage/active/audit 接点、session 调用前准入、V0/V1/review _account；复用 Experiment settlement。
- 产物：§6 恢复表逐行 fixture、每角色收据索引、持久 pause/fence 与 settle-only 模式。
- 验证/完成：预算不因新 e/R/keep/reset session 增加；同收据仅一次、真实再 review 另计；未知 execution 不重提；损坏 state 不能回退为新 run。
- 退路：对未证明的恢复点返回 blocked 并保留原版本/现场；不得用清 state、删预算文件或关 gate 恢复。

### AKA-10 — KS planner 与动态请求准入

- 前置：fixed backend 通过 AKA-09；KS design/optimize 真入口与发布合同已经可调用。
- 候选改动：schema/provider/state，调用 KS 单次能力；R 产物置于运行目录，configuration_identity 始终固定。
- 产物：设计/已提取输入补条件 → fixed implement 若干 e → 有据 optimize → keep 或新 R/新请求；每个 run 一名循环 owner。
- 验证/完成：仅阶段完成/有据设计阻碍调用 planner；发布/准入各中断点不续错 R；keep 无新行动即停；新 scope、新证据适用性、旧 C/G 和总额度全部保留。
- 退路：动态修订作为下一显式固定请求，关闭自动循环；同 native Task 不能相容时新 run 迁代码并关联旧成本，不能宣称同 campaign 动态恢复已支持。

### AKA-11 — 有限真实集成及发行验收

- 前置：MIG-03/04、AKA-09；动态能力发布额外要求 AKA-10，独立 KS demo 不受此依赖。
- 候选改动：新公开配置/fixture、AKA README 和 docs/quickstart.md、docs/optional-light-modules.md；不直接改当前 EXP 正式 pilot。
- 产物：一个固定发行配对的真实 row-softmax 调用、精确产物与三项证据、按实际发生记录的续写；动态另展示一次 keep/revise。
- 验证/完成：先合格 fake/故障测试，再经支持环境真实模型/GPU；分别记录实现、fidelity、性能和原生晋升，未运行项保留 not_run。
- 退路：环境缺项只阻止对应真实验收，保留可运行 CPU 协议与部分产物；不能由 toy/fake 推出复杂任务或动态收益。

### AKA-12 — 后续扩栈/格式门禁

- 前置：固定首发已闭合且真实题预审证明确有需求；不作为前面工作包前置。
- 候选改动：能力声明、原生 source format codec/资料工具接线；每次只扩一个实际任务所需组合。
- 产物/验证：source_tree/cuda_manifest 才增加完整删除/immutable/声明恢复测试；FA4 decode 原栈另资格化，验证 root 子域、必要节点和输入覆盖。
- 完成/退路：新组合独立通过相同身份/续写/fidelity gate 才列支持；未知保持拒绝或 partial，不把首发 single_file 的结果外推。

## 9. MIG 工作包：fork、固定 KS 与 EXP 消费

以下全部是后续执行路线，本轮不执行 Git 写操作、安装或远端迁移，也不索取 GitHub 信息。远端名称/URL 在实际实施交接时记录，不阻塞本地 spike。

### MIG-01 — 保存现状与建立隔离基线

- 前置：已有 baseline.json 与两仓工作区保护要求；执行前重新观察，历史计数不当实时状态。
- 改动/产物：只在新审查目录记录三仓 commit/branch、已修改/删除/untracked 的归属及授权源码实际字节；含有意删除，不用 patch 遗漏 untracked。
- 建立新 AKA worktree/检查副本于 bf5557ac04f6ff9988901d511cf25e1cbe61a1bc；KS/EXP 另开隔离开发位置，当前树不 reset/clean/stash/批量 add，不覆盖缓存。
- 完成标准：所有拟迁入代码可追溯到固定提交或显式工作树快照；旧 capture/历史运行未动，凭证/Home 不在取样集合。
- 退路：归属不明的差异暂留原树并列清单，先以干净上游和已提交必要补丁开展 AKA-01。

### MIG-02 — fork 与补丁分层

- 前置：MIG-01、AKA-01 ADR；基线迁移与上游升级分开。
- 候选动作：后续从同一上游 SHA 建适配 fork；本地 schema/core-seams 与 schema/workflow 分层开发，经 schema/integration 汇总，不切换现有 main。
- 产物：EXP overlay id/固定文件摘要 → 新通用补丁/Schema 模块位置 → 必要测试的映射；只提取 assets/aka 相对根差异，不把 EXP 根提交 cherry-pick 到 AKA 根。
- 完成标准：role/context、episode/seed/eligibility 等通用补丁能独立审查，Schema 文件留可选包；未迁移 EXP 政策有明确 owner。
- 退路：争议通用扩展先留 fork，接口照旧限定；远端尚未安排时本地 patch series 可完成审查，不影响 KS 自有实现。

### MIG-03 — 固定 KS、安装与资源闭包

- 前置：KS 可交付提交/包、AKA-03/04；不将当前脏 KS 目录假装为一个干净 submodule commit。
- 候选动作：在 fork 加 optional/KernelSchema gitlink，固定评审过的 KS SHA；只初始化该子模块，按其安装入口安装核心/所选能力，再构建明确列出的 AKA 包。
- 产物：upstream SHA、fork SHA、KS SHA、实际包/规则/prompt/资料/evaluator 摘要的发行配对；选中资料的取得路径与来源/许可说明。
- 完成标准：新目录非原 cwd 安装；KS 无 AKA 可用，AKA off 无 KS 可用，schema-on 缺 KS 给出明确安装项；wheel/resources 与 checkout 配对。
- 退路：包接口未完成先使用明确固定的外部 KS 目录进行开发，不把它宣传为正式安装完成；保持单向依赖。

### MIG-04 — EXP 先以 vendor 消费 fork，并单独物化 KS

- 前置：MIG-02/03，固定 backend 的 CPU 门禁；只在新的 EXP 开发副本/捕获中操作。
- 候选改动：tools/lib/aka_vendor.py 的来源/映射使用固定 fork；control/aka-source、manifest、changes 留历史；新增 assets/kernel-schema 与 control/schema-source.json（均拟议）。
- 现 project() 排除 160000 与 .gitmodules（EXP:tools/lib/aka_vendor.py:93、:98）；从 fork tree 读取 optional/KernelSchema 的确切 gitlink SHA，再独立导出 KS普通文件/模式清单到 vendor 外。
- 新 schema-source manifest 记录父 fork、gitlink path/SHA、物化摘要和资源选择；不把 KS 塞进 assets/aka 后让官方 verify 误判为未登记源码。
- 同步候选改动：environment_entry.py、aka_dev_setup.py:DISTRIBUTIONS、launcher.py:_prepare_launch、runtime.py:check/SNAPSHOT_TREES、readiness/finalize 的新捕获逻辑。
- 完成标准：AKA verify、KS manifest verify、fresh-child 模块来源/metadata/resources、fake fixed 请求和新 capture restore 一致；旧 snapshot 用原 reader/工具链仍可定位。
- 退路：某项安装/捕获失败，继续用上一发行配对；不替换活动 run，不清除旧冻结资源，不冒称迁移已完成。

### MIG-05 — 停止重复开发并保留可回退消费

- 前置：MIG-04 与 AKA-11 对欲发布能力已通过；新 EXP 条件有独立资格记录。
- 默认收敛：EXP 保留可审计的**生成式 vendor**，产品更改只在 fork/KS 完成后固定导入；新 launch 不再叠套旧 SchemaAgent，旧模块只供原捕获恢复/历史读取。
- 产物：旧静态注入/新 fixed/新 dynamic 的条件版本映射，新的实验快照；旧 frozen capture、未完成 run、原结果目录不迁写。
- 完成标准：源码唯一 owner、无手改双份产品逻辑；source/evaluator/资料/预算身份可从新记录复核，实验 B/F/D 策略对齐，原码复验/seed续改/骨架重建分列。
- 后续若转 checkout/submodule 消费，在单独分支把 assets/aka 从普通文件索引改为固定 gitlink，调整同一套 provenance/build/capture 路径并复跑 MIG-04；不同时长期维护两条可手改源路径。
- 退路：新条件停用、下一新 run 回到上个配对；已启动新 run 保留它自己的冻结版本，不能换旧二进制继续同一 capture。

### MIG-06 — 同步上游与固定发布纪律

- 前置：已形成一组稳定通用补丁、可选模块和版本配对；不与首次搬迁混做。
- 候选动作：在独立更新分支读取目标上游提交及依赖，必要时才在获准实施阶段 fetch；逐补丁标记已吸收/冲突/仍需保留，准备新的 fork SHA 与 KS 配对。
- 产物：base/ours/theirs 审查记录、最小冲突修复、实际通过的 off/固定实现/受影响恢复 gate；EXP 只生成新的导入及捕获。
- 完成标准：先 off，再固定 source/资格/恢复，改动触及的任务另资格化；不要求重跑全部历史实验，不把未运行验证写入 ledger。
- 退路：不推广新配对，保留旧发布提交/源码和 frozen capture；仅向新 run 显式迁入旧 C，保留来源/历史成本，不假装 continuation 跨版本兼容。

## 10. 交付门槛与 R5 对齐清单

| 门槛 | 必须交付的证据 | 不能据此声称 |
| --- | --- | --- |
| 接缝确定 | AKA-01 指定 e/限额/startup ADR 和实际 fixture 轨迹 | GPU、真实 provider 或完整 backend 已通过 |
| 固定 backend | AKA-02～09、MIG-03/04 的身份/seed/off/资格/恢复证据 | 动态 R 或复杂格式已支持 |
| 真实实现 | AKA-11 的真实 C、构建、完整 Task 判据、有限 fidelity、性能 | 合格慢 C 已晋升、全域等价证明或普遍加速 |
| 动态编排 | AKA-10 发布/准入/keep/revise 与全角色费用；真实一次能力调用 | 每 e 改设计合理、动态策略有科研收益 |
| 实验消费 | 新固定捕获、等预算/资料/策略、全部样本及失败记录 | 旧静态消融就是新动态结果 |

R5 需把 KS 与 AKA 的协议版本、字段/unknown 拼写、source 字节编码及删除语义、Task 验收引用统一成一个小合同，禁止暴露 AKA 私有对象。
还需统一 KS 的 R 发布与设计修订计数、选中范围、fidelity 方法/审阅收据输入输出、keep 的停止规则和预算收据去重键。
共同 fixture 使用同一 row-softmax Task/源码身份与严格比较器；AKA native 包装不能改变 KS Task 意义，FA4 decode 仅预审已知范围/材料，不伪造支持。
AKA-01 的 façade/原 loop 选择和 AKA-05 的 never/V1 选择由技术 gate 决定；不再向用户发内部技术菜单。
本轮可审查结果就是本文的工作包、修改面和恢复表；实现、真实运行、发行配对与迁移的完成状态均留待各包产生实际证据后更新。

