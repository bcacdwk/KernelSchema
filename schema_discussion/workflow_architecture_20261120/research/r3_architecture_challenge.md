# R3：架构反例、最小性与可保留边界

日期：2026-11-20。性质：第三轮独立反例审查，所有新增行为仍是待决候选，不是实施计划或对 R2 的验收。
已读本仓 AGENTS、用户简报、历史与两轮四份报告，并重新检查下列原生代码；没有运行模型、GPU、真实服务、测试或安装，也没有修改实现、提交、远端操作或读取凭证。
本轮只写本文。`KS`=/home/shaohanyong.shy/KernelSchema；`AKA`=/home/shaohanyong.shy/atrex-kernel-agent；`EXP`=/home/shaohanyong.shy/AKA_experiment。
`事实`表示当前代码或规范可直接支持；`推论`表示据调用顺序构造的失效路径，尚未执行故障注入；`候选`表示未来可能采用的处理。
沿用 R2 的简写：R=指定设计及范围，A=一次固定设计实现尝试，e=原生 episode，C=准确源码，G=保留的 incumbent。简写不意味着要建同名数据库对象。

## 1. 审查结论：保留目标，收回“现成且更轻”的暗示

**可以保留：KS 拥有四流程入口和 Schema 语义；AKA 复用自己的一份执行/评测/晋升设施；EXP 消费版本并管理实验。**
**不能据候选直接下结论：参考 implement 必然很薄、Workflow provider 必然比少量 loop 接点轻、never 足以替代 V1、seed 安装已有崩溃事务、晋升恢复天然不再评审。**
R2 已多处注明待验证；本轮检查的是支撑推荐的前提，不把那些候选接口误写成 R2 声称已经实现的功能。
其中 seed 时序和恢复评审已找到具体反例；其余没有找到必然失败的证明，但也没有端到端成功证据。

| R2 命题 | 本轮处置 | 原因 |
| --- | --- | --- |
| 中立契约 + 有限参考 implement | 条件保留 | 支持一个明确环境可成立；若还承诺 AKA 级恢复、隔离、远端调度，“薄”已失真 |
| 可选 provider + 一份 native Campaign | 保留为组织边界 | composition 开关合理；provider 和 loop 接点不是互斥方案，前者仍需要后者 |
| 单 episode 驱动比接原循环轻 | 暂不排序 | Search driver 证明存在内部用法，未证明预算、恢复、startup、收尾可直接沿用 |
| schema-on 默认 never | 改为按支持任务验证的选项 | V1 包含接纳较慢首版、pin、元数据及独立门禁；不是纯写码 prompt |
| clean 后 install_source 即可续接 | 改写时序与恢复合同 | native 已提前写 exploring；崩溃可能留下未装或半装 seed |
| seal 后、ABBA 前 fidelity | 保留主要检查位置 | normal/未提交恢复共用入口；已提交晋升走另一条 audit 路径，须带持久收据 |
| R 是动态产物而非 config 热修改 | 条件保留 | 还需独立的“已发布”和“已准入”事实；EXP 当前 binding 不能直接动态改写 |

用户真正需要决定的是独立运行承诺、首发覆盖和成功/验收口径；provider 名称、phase 名称和 V1 接法应由技术验证消除，不宜让用户选内部补丁。

## 2. 中立契约是否在复刻 AKA

事实：KS 当前 AgentCLI 只接受 worker/reviewer，并在执行前检查 YAML 工件目录；不是已存在的 coder runtime。
证据：KS:schema_extract/tools/agent.py:27、:33、:217、:258；KS:schema_extract/tools/artifacts.py:224。
Claude 命令把写入限于声明的 YAML，Bash 只放行检查 helper，MCP 为空；换一个 prompt 不会获得构建或 GPU 工具。
证据：KS:schema_extract/tools/backends.py:171、:175、:179、:186。

**最小性反例：**实现器只做“调用 CLI→读 kernel.py→失败再调用”，第二次从 C1 继续时，没有保存 C1 字节、原输入和已花预算。
它已不能回答“本次用什么设计”“C1 是否测过”“暂停后为何又开始”；把这些补齐是必要编排，不能以“CLI 自带 agent loop”消除。
反过来，补齐这些事实不必复制 AKA：原生 CLI 负责一次会话内工具循环，任务提供 evaluator，KS 只控制有限调用与交付。

| 保留在 KS 的最少能力 | 不应由“中立”强迫它实现 |
| --- | --- |
| 固定输入与权限、有限 coder 调用、准确源码保存、明确停止/消耗 | CandidateWorkspace/Git 晋升、ABBA、trajectory、worker 调度 |
| 任务提供的构建/评测适配与证据解释 | 自建远端执行平台、自动探测任意栈、全部 AKA numerical policy |
| 从 C1 字节继续；可拒绝不支持的会话恢复 | 跨机器/provider 搬运原生 checkpoint、长期后台维护平台 |
| 反馈与三个验收维度 | 第二套 Schema parser、通用语义等价证明器 |

候选参考后端应先承诺单任务、单栈、串行、显式调用上限和一个可用 evaluator；原生会话恢复可以是可选能力。
但“进程崩溃后未知远端作业”不能用新 session 重提解决：要么复用执行后端的 operation id/结算能力，要么首期不承诺这种远端自动续跑。
失败 fallback 是缩小执行环境或暂时公开依赖外部 AKA 后端，不是把未知消耗归零或把 partial 改称 completed。
四流程中立的仅是请求/产物/证据含义；不应先建所有后端都实现的完整 run/session/attempt 方法集合。
第一后端的协议可先小到“冻结请求、代码起点、限额、结果”；第二后端暴露不匹配时再改，不先为不存在的替换场景建注册中心。
因此 C 的产品方向可讨论，但“首期同时完成参考后端与 AKA 动态编排”不由中立契约自动推出。

## 3. Workflow provider 与少量 loop 接点：不能虚构二选一

事实：Search driver 在单步时创建 LongHorizonCampaign 并以绝对 episode 数上限调用；它不是公开的 step 协议。
它先要求已验证 bootstrap，返回值只携带 accepted kernel，还包含 trajectory 初始化/广播；不能原封不动承担 schema partial。
证据：AKA:src/aka/workflow/light/search_driver.py:45、:56、:173、:194、:202、:217。
事实：公共 Campaign 协议只有 prepare/run，但原生 startup 还依赖 `_load_problem/_campaign`、execution、checkpoint、lease 和信号处理。
证据：AKA:src/aka/contracts/workflow.py:107；AKA:src/aka/workflow/legacy/startup.py:111、:149、:155、:169。

**预算反例一：**照抄 Search 的构造只传 max_episodes=e，LongHorizon 的 token_budget/max_stall 默认 0、max_version 默认 None。
普通 Campaign.run 则显式传 max_iters、token_budget、max_stall；不能称照抄之后仍自动保留原硬限额。
证据：AKA:src/aka/workflow/light/long_horizon/campaign.py:404；AKA:src/aka/workflow/light/orchestration/campaign.py:2559。
**预算反例二：**同时传 max_version 和 max_episodes 也不够；原 while 用 if/elif，只要 max_version 非空就不检查 max_episodes。
改用 episode_limit=1 仍有恢复问题：starting_episodes 在 `_recover_interrupted` 之后记录，恢复结算 e 后可能再跑 e+1。
证据：AKA:src/aka/workflow/light/long_horizon/campaign.py:1718、:1727、:1734、:1742、:1749。
这不是断言方案无法实现，而是证明需要一个正确的“只恢复/结算指定 e”边界，不能靠现有几个限额参数拼出已验证结论。

provider 外置 R/A 决策的优势是真实的；它仍须把指定任务、seed、准入身份、资格结果和终态交付接到 native loop。
反案是在原 loop 增加受限 assignment/settled 接口，由可选 provider 提供策略；同样能让 Schema 知识留在模块外。
两案共有的成本比差异更大：source 起点、固定任务 prompt、session scope、准确候选资格和恢复收据都无法省略。
候选选择标准应是“谁拥有每个事实、是否共用 normal/recovery、哪些原流程必须复制”，不是 provider 或 hook 的命名。
若单步 façade 必须复制 prepare/finish、读取大量私有字段或修补多处恢复分支，少量原循环接点可能更小，应允许切回。
若两条路都能把改动限制在同一窄接口，composition 选择 provider 可保留，内部驱动方式不必上升为产品决策。
不推荐外层每 R 新建整场 campaign 作为长期方案：它另外引入 V0、累计预算和 incumbent 的迁移问题。

## 4. never 的真实代价与初次实现

事实：`framework_baseline=never` 在决策最前面直接 skip，既不 run 也不 pin。
V1 的普通验收明确不要求 speedup；它可以接纳第一份正确、合规但更慢的框架实现。
证据：AKA:src/aka/workflow/light/orchestration/campaign.py:1387、:1203、:2230。
因此 **G=快的参考/既有实现、C=更慢的合规首版** 时，普通 e 的 ABBA 不晋升 C；原生 V1 却有不同的初次接纳语义。
保留独立的 implemented-for-R 结果能解决用户交付，但它并不使 native canonical 自动成为生产合格结果。

事实：原 `_finish` 对 canonical 运行 production 检查，并可能执行 SOL finalize；直接只调 engine.run 不执行这些收尾。
证据：AKA:src/aka/workflow/light/orchestration/campaign.py:2570、:2583、:2593。
候选 Schema 输出不能仅把 public-candidate 路径叫作“原生 production 导出”；需对所交 C 执行相应的源码/依赖/包装检查，或明确只交实现材料。
若没有晋升，原生 G 的合规情况与 C 的合规情况分别返回；既不能丢掉合格慢代码，也不能把原 G 冒充当前 R 的实现。

另有两项不是性能策略、但会被“跳过 V1”掩盖的事实：

- V1 会调用 seed_baseline_metadata；single_file 为部分 Python DSL 生成 solution.json，source_tree/cuda_manifest 的相应函数为空。
  证据：AKA:src/aka/workflow/light/orchestration/campaign.py:1215、:2045；AKA:src/aka/task/source/formats/single_file.py:39。
- V0 的 correctness prefetch 只看 reviewers 配置，不检查 never；若这些 reviewer 已开启，仍可能发生 V1 命名的额外模型调用。
  证据：AKA:src/aka/workflow/light/campaign.py:279；AKA:src/aka/workflow/light/orchestration/campaign.py:369、:1121。

不能反向夸大：未发现所有任务都必须有 solution.json 才能运行；production structure 在文件缺失时直接接受这一项。
证据：AKA:src/aka/workflow/light/orchestration/optimization_policy.py:164。因此这是逐格式核验点，不是 never 必然不可用。
候选优先支持准备条件已经闭合的任务；若复用准备 helper 足够，就保留 helper、跳过自由写码；若不足，再接 V1 或缩小首发。
自动 conversion 关闭也不撤销已有 Gluon incumbent 的阶段约束：普通候选 gate 仍据 G 要求 Gluon。
证据：AKA:src/aka/workflow/light/long_horizon/campaign.py:1040；AKA:src/aka/workflow/light/orchestration/optimization_policy.py:239。
因此不能只列 never/convert_after=0/goal=false 三个值就宣称固定 R 与原生策略已相容。

## 5. source seed 可以不改 incumbent，但现有导入点不具备完整恢复语义

事实：install_source 只写指定 workspace 的完整 source（含可编辑文件删除），不移动 Git；promote 另外要求当前 canonical 未前进且候选为其后代。
证据：AKA:src/aka/contracts/workspace.py:193；AKA:src/aka/task/candidate_workspace/source.py:147；AKA:src/aka/task/candidate_workspace/promotion.py:18。
所以“从当前 G 建 private candidate，再把 C1 装到 candidate”在静态源码路径上有支撑；无需先把 C1 晋升成 G。
ABBA 直接从 worktree.base_commit 和 candidate_commit 读已提交字节，没有把工作区 seed 当 baseline 的必要。
证据：AKA:src/aka/workflow/light/long_horizon/recorded_verifier.py:77、:81。

**反例：e1 partial C1→e2 导入 C1。** native materialize 后已写 active.phase=exploring，随后才 link/clean/register。
R2 的“clean 后 register 前装 seed、exploring 恢复绝不再装”在此有窗口：进程可能在 exploring 持久化后、seed 安装前崩溃。
也可能多文件已删除一份但未写齐；install_source 的回滚只在捕获 BaseException 时执行，原文件备份在内存中。
证据：AKA:src/aka/workflow/light/long_horizon/campaign.py:1799、:1802、:1811、:1831；AKA:src/aka/task/candidate_workspace/source.py:185、:195。
推论：单靠函数“transactional”文档不能证明 SIGKILL/宿主崩溃安全；本轮未运行故障注入。
候选最小修正是 seed intent/hash + 可验证的安装完成事实，或把 preparing→exploring 移到完整安装后；首次 Agent 准入必须依赖它。
正常恢复过的 Agent draft 不重装；判断依据应是“导入已完成且会话已准入”，不能只看现在这个 exploring 字符串。

**另一个具体格式反例：**public-candidate.json 为 single_file 也收集 solution.json；install_source 的 single_file 输入却只允许 kernel.py。
证据：AKA:src/aka/workflow/legacy/episode_output.py:57；AKA:src/aka/task/candidate_workspace/source.py:162。
因此不能把公开产物的全部 files 原样塞进 RevisionContent；要复用 source codec，辅助元数据按既有 Task/格式权威处理。
源码声明、immutable files 或 source format 不相容时应拒绝迁入；不为了复用旧 C1 改掉新任务的约束。

“canonical 不动”还需修辞校正：未晋升 e 也会提交 `.atrex_versions/vN.json`，Git HEAD 会前进，但 incumbent 的 kernel 字节不变。
证据：AKA:src/aka/workflow/light/long_horizon/campaign.py:556、:1278。
下一 e 的 Git base 必须重新取当前 canonical HEAD；G 的源码身份可以保持，不能把 A 初次 base_commit 永久复用。
这恰好说明无需为 seed 创造第二条 canonical 分支，只需分别保存工作起点、比较源码和 native 当前 revision。

## 6. fidelity 的检查位置合理，完整接口还缺哪些事实

事实：report 校验标准 full Evaluate 对应当前源码，然后独立 correctness，再 seal；不是 coder 一句 candidate_ready 就可晋升。
证据：AKA:src/aka/workflow/light/host/journal.py:877、:1304、:1315。
普通路径与未提交的终端恢复都会进入 `_assess_terminal_handoff`；在其候选/production 检查后、ABBA 前加额外资格，能覆盖这两条路径。
证据：AKA:src/aka/workflow/light/long_horizon/campaign.py:1011、:1055、:1393、:1909。
这能拦住“数值正确且更快，但数值路径、work schedule 或同步方案违反 R”的 C；代码正确不证明设计符合性。
证据：KS:IMPLEMENT_SCHEMA.md:38、:51。

**尚不完整的地方是 durable output，不是再找一个 callback 名字。**
现 `_record_terminal_episode` 重新构造 attempt 字典；向 active 任意加字段，不会自动被拷进 attempt 或 promotion audit。
晋升证据随后由这个 attempt+journal 写入私有 audit，再通过 commit trailer 绑定。
证据：AKA:src/aka/workflow/light/long_horizon/campaign.py:1162、:1231、:1243。
候选资格接口至少需携带输入 assignment/R/scope、sealed C、检查版本、结论/未知原因、收据引用和实际消耗，并被这次晋升证据绑定。
pass 才进入晋升；fail 保留违背位置，unknown 保留未完成验收/外部故障，不把后二者都投射为“schema 不可实现”。

**已晋升崩溃反例：**native canonical 已提交，host 在保存 state 前死掉；恢复走 `_recover_interrupted` 的 current_revision!=base 分支。
该分支用 audit 恢复 accepted，不再调用 `_assess_terminal_handoff`；若 fidelity 只存在内存/active，adapter 无法证明它曾通过。
证据：AKA:src/aka/workflow/light/long_horizon/campaign.py:1508、:1516、:1561。
audit 当前只校验 episode/version/base/branch/accepted；recover_promotion_audit 通常只回 `{status: verified}`，不是 schema 资格结果。
证据：AKA:src/aka/workflow/light/long_horizon/promotion_audit.py:78；AKA:src/aka/workflow/light/long_horizon/audit_recovery.py:61、:95。
候选应吸收原收据、恢复同一分类与消耗，不重新抽模型意见来改变历史；缺关键证据则保留已提交事实并报告证据受阻。

还不能说“原生恢复绝不会再评审”：engine.run 先 prepare_campaign，production 且 latest_version>0 时会检查 canonical，早于 audit 恢复。
production reviewer 的成功缓存是 Campaign 实例内字典；重建进程可在进入 audit 前重新发起 production-review。
证据：AKA:src/aka/workflow/light/long_horizon/campaign.py:1699；AKA:src/aka/workflow/light/long_horizon/main_adapter.py:43；AKA:src/aka/workflow/light/orchestration/campaign.py:705、:831。
audit helper 不重跑 Gate，与全恢复入口零额外评审是两回事；新 fidelity 的去重承诺要单独验证，已有 native 评审仍如实记账。
同一模型调用的账单重放只记一次；若原生真的重新调用了模型，这是真实新增成本，不能“去重”成未发生。

## 7. verify-only：需要的是无改码验收分支，不是第五套工作流

事实：single_file seal 对与 episode baseline 完全相同的源码报“no Kernel change”；该条件不应改成对所有格式的一概断言。
证据：AKA:src/aka/task/candidate_workspace/source.py:327、:330。
独立必要用例是 extract(G)→implement(R,G)，或已有 C 与新交接范围需要重新核验但无需编辑；成功不应靠加注释伪造代码变化。
另一用例是同字节 C 已有真实数值/性能证据，但缺新 R 的 fidelity 或新范围验收；需要补验收，不需要 coding episode。
候选可以复用受信任测量入口和准确源码 staging，按需复用旧测量；不产出新的 native promotion，也不新造一个 session。
现 Runtime 的 trusted execute 已有 reuse_completed、固定 Task/evaluator 和 candidate_source 入口，提供复用素材；尚未有完整 verify-only 接口。
证据：AKA:src/aka/workflow/light/orchestration/supervisor_runtime.py:648、:657、:687、:689。
因此保留结果语义/内部路径，但删掉“必须新建独立 verifier 服务或命令族”的额外复杂度。

## 8. 动态 R 与 frozen config/session：可兼容，但不能借路径伪装动态配置

事实：Bootstrap capture 冻结 effective config、实现、resources 和 host；restore 重新 capture 并逐值比较。
证据：AKA:src/aka/bootstrap/continuation.py:57、:67、:161。
将新 R 作为运行产物而非改写 config 在概念上成立；前提是产物不位于被资源 fingerprint 扫描的可变子树，规则/工具本身仍固定。
静态资源与运行产物分开只是第一步，不能因“路径字符串没变”就允许覆盖同一路径的 root.yaml。
EXP 当前 SchemaBinding 冻结 manifest，configuration_identity 包含 binding；session 和 checkpoint 也绑定其 schema digest。
证据：EXP:tools/python/aka_experiment/schema.py:183、:258；EXP:tools/python/aka_experiment/schema_sessions.py:71；EXP:tools/python/aka_experiment/schema_io.py:363。
所以当前 SchemaAgent 不能直接把 binding 从 R0 改成 R1；候选需固定 host 配置、按 invocation/会话准入传不可变 assignment。

**发布反例：**planner 已发布 R1，driver 恢复时只读旧 A0/active e，继续对 R0 调 coder；只记录“最新 R 指针”无法自动阻止它。
候选要区分“R1 bundle 已持久发布”和“R1 已成为下一实现请求”；发布文件本身不等于准入新设计。
最小权威记录应能核对当前请求、待吸收的旧 e、下一指定 e 与 R/seed；两个事实冲突时停止准入并调和，不按 mtime 猜版本。
先吸收旧事务已经完成的结果；新 R 若已准入，不再启动旧 R 的新 invocation。反之仅有未准入草稿/发布件，不取消合法旧 A。
新 R 用新会话 scope；同 R 的原 e 恢复可保留 session，输出上限重开也继续绑定同一个 R。
原生 session 已有同 e 换 context 的路径，应传同份固定 assignment，而非再设计独立“压缩后恢复”状态机。
证据：AKA:src/aka/workflow/light/long_horizon/session.py:255、:394、:401。

## 9. 必须保留的反例清单与判定

下表是应证伪的输入/时序，不是本轮已执行测试；native 证据说明为什么这些情形不能被一个 success 标志合并。

| 反例 | 应有结果/约束 | 最容易误用的依据 |
| --- | --- | --- |
| e1 只写好 root 部分，e2 才补齐 | 同 R/A；e2 起点为准确 C1；预算累计；C1 测量不自动属于 C2 | 新 e 默认从 G 起步，AKA:src/aka/workflow/light/long_horizon/campaign.py:1785 |
| C 正确、符合 R、已测但比 G 慢 | 实现可完成、目标未达、未晋升；交付 C 并可供 optimize | ABBA accepted 只表示保留策略，AKA:src/aka/workflow/light/long_horizon/campaign.py:1098 |
| C 更快但违背 design | fidelity fail，不能晋升为 R 的成功实现；通常先修码 | correctness/full Evaluate 无设计证明，KS:IMPLEMENT_SCHEMA.md:51 |
| root 是子域，dispatch/路由没补全 | 返回明确覆盖缺口；子请求完成不得升格为总任务完成 | FA4 root 限 B∈{15,16},Q=4，Contract 还含 B=1..17,Q=1，KS:schema_template/examples/fa4_decode/root.yaml:8、:38 |
| effective_identity 相同，Task 或 dispatch 变了 | 形成新请求/范围绑定并重判证据适用性 | 该 hash 只含 node/source_manifest/origins，KS:schema_template/tools/design_resolution.py:47 |
| provider 字节改，consumer Version 不变 | 冻结整个实际来源闭包；复核受影响项 | Version 按本 raw 独立，KS:schema_docs/0.common.md:43 |
| 只改 Feedback 或排版 | 新字节留痕但不自动算 optimize；旧证据保留原绑定，可明确关联复用 | source_manifest 按原字节，KS:schema_template/tools/design_resolution.py:39；Feedback 不豁免硬约束，KS:schema_docs/6.realization.md:44 |
| C1 测过，随后改成未测 C2 | C2 返回 not_run/unknown，不能读“最新日志”继承 pass | report 对当前准确 source digest 检查，AKA:src/aka/workflow/light/host/journal.py:898 |
| optimize=keep，反馈没新增仍反复调用 | keep 可直接完成优化操作；无可执行新证据采集则停止，不强迫再开 A | KS:OPTIMIZE_SCHEMA.md:39 允许保留；native blocked 会自动 fresh retry，AKA:src/aka/workflow/light/long_horizon/campaign.py:1948 |
| GPU/服务超时，被说成 schema 不可实现 | 保留外部故障/未决；同 operation 结算，不据超时改设计 | AKA:src/aka/workflow/light/orchestration/infrastructure_retry.py:105、:133 |
| 已发布 R1，driver 继续 R0 | 看准入权威与旧事务归属；不猜最新目录，拒绝冲突准入 | EXP session 明确绑定 schema，EXP:tools/python/aka_experiment/schema_sessions.py:71 |
| ABBA 晋升后崩溃 | 从绑定 audit/资格收据恢复；无第二次晋升；账单重放不重复加 | AKA:src/aka/workflow/light/long_horizon/campaign.py:1516；native prelude 的额外评审见第 6 节 |
| submodule 未初始化但 schema-off | 原 provider 能解析/启动，零 Schema planner/fidelity 调用；不能早期 import KS | resource_identity 能记录未初始化 gitlink，AKA:src/aka/core/lock.py:104；这尚非 schema-off 实测 |
| 用户暂停/全局预算耗尽后新建 A | 保持暂停或停止准入；新 A 不恢复余额；只吸收已有完成事实不代表获准新调用 | native durable budget 拒绝缺失或 inflight 退款，AKA:src/aka/workflow/budgets.py:51、:64 |

范围不能交给隐式 dispatcher：现规范明确目录笔记不自动应用、不补齐 YAML，root 也不自动 fallback。
证据：KS:schema_docs/0.common.md:148、:184、:186、:190。
仅 Feedback 改动也需核对内容性质，不能把字段名当成免审凭证；它可能改变供 coder 参考的替代证据，旧结论仍引用旧快照。
对“同一 C 换 R”可复用原测量，但若 Task/evaluator/环境/计时口径变了，源码同字节不足以复用。
同理，native vN 的未晋升记录可携带 incumbent 历史性能；应读 measurement_subject 或 source-bound 公开产物，不把 vN 性能当 C2。
证据：AKA:src/aka/workflow/light/long_horizon/campaign.py:886、:920；AKA:src/aka/workflow/legacy/episode_output.py:84。

## 10. 哪些状态能复用，哪些复杂度可以删

R2 的 W/R/A/e/s/C/G/F 适合解释关系，不宜逐个变成独立权威 ledger。

| 事实 | 优先复用 | 最小新增/限制 |
| --- | --- | --- |
| Task/config/工具版本 | 原 frozen launch、Task snapshot、resource identity | 外围引用 Schema 规则和选定范围；不复制 Task 私有 evaluator |
| e、session、draft、handoff、promotion | 原 active/archive、Agent scope、native audit | 只增加不可变 assignment 关联；不建第二个 e/session 计数器 |
| C 及其测量 | read_source/source_identity、public-candidate、Experiment receipts | 记录父 seed 和 R/范围关系；不另造 GPU 结果数据库 |
| G | canonical 的源码与 revision | 新设计不清空 G；区别 Git HEAD 变更和源码变更 |
| R 与 A | 没有现成 schema 生命周期可直接复用 | 不可变输入发布 + 固定请求的 e 归组 + latest/best-validated 两个引用足够起步 |
| 用量 | native usage_receipts 与外部模型/执行收据 | 一份总账索引各角色的增量；不把 native tokens 再加一遍 |
| 普通反馈 | 实现结果引用和检查诊断 | 只对合规 Prefer 替代产生 Feedback proposal，不复制 transcript |

public Workflow facts 已区分缺测、完整度、source hash 和 episode outcome，可借鉴其投影，不必把 native 私有 dict 暴露成 KS API。
证据：AKA:src/aka/contracts/workflow_export.py:1、:82、:145、:162。
来源闭包/有效视图直接复用 Template；不新建语义等价 hash、自动 dispatch 编译器或“全域证明”服务。
不为每次局部代码修复、新 e、keep 或排版生成新 A；R/请求实质变化才需要新的固定实现绑定。
不重建 P1–P6、不承诺全部 provider、不预先复制 EXP 全部隔离/存储补丁；使用场景需要的能力再选。
“pause 状态持久化”不能删：EXP pause-requested 目前只是新 e 准入检查；它不会自动阻止外加 planner/新 A。
证据：EXP:assets/aka/src/aka/workflow/light/long_horizon/campaign.py:1738。
原生每个 e 有自己的 active deadline；新 e 获得时间片不代表 A/W 已花时间清零，新 A 也只从 W 剩余额度获得授权。
全局准入控制也不能只读 native 空状态：load_state/load_active 的读错回退可能看似新 run，外层必须核对已有 archive/收据并报告损坏。
证据：AKA:src/aka/workflow/light/long_horizon/store.py:76、:99；用量增量去重可复用同文件:85。

## 11. 最小技术验证及失败 fallback（本轮均未运行）

下表只说明足以推翻架构假设的窄验证，不承诺工作量，也不构成正式实施顺序；先用 fake Agent/Evaluation/故障点，不需要真实服务。

| 验证问题 | 最小证据 | 失败 fallback |
| --- | --- | --- |
| 单步真的是指定 e | 恢复已结算 e 不启动 e+1；max_version/token/pause 仍生效；不重复 V0 | 小型原 loop 准入/终态接口，保留同一 provider 边界 |
| seed 仅改私有源码且可恢复 | 三种格式的删除/固定文件；在 phase 写入及每个 install 步骤故障后，准入只见完整指定 seed | 明确导入凭据/延迟 phase；首发限制一种格式，不能覆盖旧 draft |
| C 慢也能交付，C 快但违规不晋升 | fake 全正确慢候选得到完整实现结果；违规快候选 ABBA 未调用；G 各自正确 | 增加结果投影或使用较窄固定请求 backend，不误改 promotion |
| never 保留所需准备/收尾 | 选定任务的 V0、metadata、production 检查、C 导出、baseline 指标逐项有去向 | 复用准备/收尾 helper；必要时接 V1 或缩小首发支持 |
| fidelity 两类恢复一致 | seal→review→ABBA→promote 各中断点同一资格收据；已提交晋升无新增 fidelity 调用 | 先支持人工/离线检查并返回 unknown；不允许未知资格晋升 |
| R 与证据绑定 | Task/dispatch 变、provider 变、格式化/Feedback 变、C1→C2 均有不同的适用性判定 | 更保守地拒绝旧 continuation/重验受影响项，不伪称语义等价 |
| 动态 R 合法且不会续错 | 发布 R1 后各中断点，恢复只结算旧事务或准入明确 R；配置捕获保持不变 | 先固定 R backend；动态优化作为下一次显式请求，公开其边界 |
| off 真正无 KS 运行依赖 | 缺 submodule/缺 KS 依赖的干净安装仍可走原 profile；检查调用/工具/成本轨迹 | 延迟加载并缩小资源声明；不靠空 placeholder 掩盖依赖 |

可读代码已给出窄路径，不足以代替这些验证；尤其 install_source 的异常回滚和 audit helper 的注释不是故障注入结果。
若后续获准真实 demo，只需一个资料/验收闭合的小任务确认实际模型与执行接通；不能拿 fake 成功宣称数值/性能/fidelity 已验证。
EXP vendor 会排除 gitlinks 与 .gitmodules，未来消费带 KS 的 fork 还需显式处理资源物化；submodule 选择本身不解决分发。
证据：EXP:tools/lib/aka_vendor.py:93、:98。

## 12. 给主代理与 xhigh 整合的收敛

建议最终仅保留三项产品判断：

1. **独立运行承诺：**是否必须在不依赖 AKA 运行后端时生成并验证支持范围内的 kernel；若是，接受一个有边界的参考后端，而非两套全功能平台。
2. **首发覆盖与演示：**是否先以小任务/单栈证明固定设计的实现、源码续接与真实验收，再把同一结果用于 design/optimize 演示；不让公开可用性等待复杂多节点全自动化。
3. **成功与验收口径：**当前 R 的合格慢实现是否独立交付、何时授权 optimize、复杂 fidelity 允许何种有限证据和 unknown；这决定产品行为及成本。

不把“选 provider 还是 hook”“never 还是 V1”“是否新增 phase”列成用户必须拍板的内部菜单。
整合时应准确保留以下交接：

- provider 是开关/所有权边界，单步 engine 只是待验证驱动；Search 先例不能作为预算/恢复已通过的证明。
- source seed 的不移动 Git 属性有源码证据；崩溃窗口、公开产物与 source codec 差异有具体反例，须改写 R2 导入表述。
- 正确慢 C 与 G 分离、verify-only、三条证据链仍必要；它们不要求重建独立测量/晋升系统。
- fidelity normal/未提交 recovery 可共用一个点；已提交晋升必须靠持久证据恢复，不能补做模型审查冒充原历史。
- native production prelude 可再评审；新功能去重与原生真实额外调用要分别表述、分别计费。
- 复用 native e/session/source/receipts；新增最少的 Schema 发布与固定请求关联，不把概念关系图兑现成巨型状态机。
- 所有故障路径、公开安装与实际后端可用性仍未实测；本轮没有把“未发现反证”写成“已验证可行”。
