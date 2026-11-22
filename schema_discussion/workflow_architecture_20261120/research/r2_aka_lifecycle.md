# R2：AKA 的可选 Schema 接入、串行生命周期与最小改动

日期：2026-11-20。本文是第二轮架构研究，供用户选择方向；所有新增接口、路径、状态和策略均为**候选**，不构成已批准实施计划。
证据前缀：`AKA` = `/home/shaohanyong.shy/atrex-kernel-agent`；`EXP` = `/home/shaohanyong.shy/AKA_experiment`；`KS` = `/home/shaohanyong.shy/KernelSchema`。引用指当前本地工作树的 `路径:行号`。
已阅读两仓 AGENTS、用户简报、历史和两份 R1 报告；对下述关键结论重新核对源码。只新增本报告，未改实现、跑模型/GPU/测试、操作远端、提交或读取凭证；未派子代理。

## 1. 建议先对齐的方向

**推荐候选：可选 Schema Workflow provider，内部用一份 native Campaign 串行驱动单次 episode；KernelSchema 拥有设计产品能力，AKA 拥有实现与测量事务。**
这不是另写一个 AKA：保留 Task、Agent、CandidateWorkspace、Experiment、Evaluation、Execution、Journal、ABBA 和原生恢复，只增加窄的任务投递、源码起点和候选资格接口。
选择 provider 是为了让 schema 开关在 composition 边界生效；不为“插件化”增加任意事件总线、插件排序、动态 hook 注册或第二套 GPU 调度。
SchemaWorkflow 的候选逻辑应是 `初始化 R → implement(R) 若干 episode → 整理可信反馈 → optimize(R) → implement(R')`，每一时刻只运行一个 planner 或 coder。
固定 schema 的多轮实现是主路径；一次 episode 结束既不自动改 schema，也不自动调用 planner。
实现完成应返回“当前 R 的合格源码”；全局 incumbent 继续表示原生晋升保留的最好源码，两者可以不同。
首版可以显式使用原生 `framework_baseline=never`，将初次 bring-up 放进普通 episode；初版的较长工期由更大的实现尝试总预算承担。
这条选择减少 V1 专用接入改动，但需验收 production 收尾、基线指标与导出的差异，不能把它当成已经验证过的现成功能。

## 2. 必须正面处理的已核实约束

| 已核实事实 | 直接影响及代码证据 |
| --- | --- |
| 静态 SchemaAgent 依赖 EXP 核心补丁 | 原生请求只有 prompt/access 等；EXP 增加 role、context_files、session_namespace/storage_policy，Light 显式传 optimizer。不能仅复制 schema.py。AKA:src/aka/contracts/agent.py:199；EXP:assets/aka/src/aka/contracts/agent.py:217；EXP:assets/aka/src/aka/workflow/light/long_horizon/session.py:185。 |
| 原生 Light 没有 schema planner 调度点 | provider 创建 ManagedCampaign；native run 直接执行 prepare、恢复和 episode 循环。AKA:src/aka/workflow/light/provider.py:142；AKA:src/aka/workflow/light/long_horizon/campaign.py:1698。 |
| 有可借鉴的“单次 episode”驱动 | Search driver 用 `LongHorizonCampaign(max_episodes=episode).run()`，然后验证 episode 持久终态；不是只能整 campaign 调用。AKA:src/aka/workflow/light/search_driver.py:173、:194、:197。 |
| 新 episode 仍由 incumbent 创建 | base_commit 取 canonical current_revision，然后 plan/materialize；scratch 清空。未晋升 partial 不会自然续接。AKA:src/aka/workflow/light/long_horizon/campaign.py:1785、:1800。 |
| 已有完整源码安装接口 | `install_source` 包含删除、固定文件和声明校验；不是只能复制 kernel.py。AKA:src/aka/contracts/workspace.py:193、:196；AKA:src/aka/task/candidate_workspace/source.py:147、:177。 |
| partial 产物并非完全丢失 | 终态 capture 读取 working source，未提交变化令 revision=None，保留每文件 hash/base64 与相符 evaluations，并防并发变化。AKA:src/aka/workflow/legacy/episode_output.py:52、:74、:84、:92、:103。 |
| V1 先于普通 episode | prepare_campaign 调用 ensure_framework_baseline；后者有自己的模型、修复、评测和预算。AKA:src/aka/workflow/light/long_horizon/main_adapter.py:30、:42；AKA:src/aka/workflow/light/orchestration/campaign.py:1186。 |
| V1 与固定设计存在实际冲突 | 当前 V1 对 Triton/Gluon 家族要求先 plain Triton，Gluon 留给后续 escalation；不能只追加一个 Gluon schema。AKA:src/aka/workflow/light/orchestration/campaign.py:2171。 |
| 有现成策略关闭路径 | framework_baseline=never 直接 skip；convert_after=0 令自动转换条件为假；goal_episodes 默认 false。AKA:src/aka/workflow/light/orchestration/campaign.py:1389；AKA:src/aka/workflow/light/orchestration/hardware.py:126；AKA:src/aka/workflow/light/plugin.py:33。 |
| report accepted 不等于晋升 | 标准 full Evaluate/准确源码与正确性检查先于 seal；之后独立 ABBA 再决定 accepted/promotion。AKA:src/aka/workflow/light/host/journal.py:1253、:1304；AKA:src/aka/workflow/light/long_horizon/campaign.py:1055、:1098、:1248。 |
| 原生没有实现进度终态 | 仅 candidate_ready/pivot/blocked；有效 blocked 还会自动 fresh episode retry。AKA:src/aka/workflow/light/long_horizon/models.py:22；AKA:src/aka/workflow/light/long_horizon/campaign.py:1948。 |
| 配置/资源冻结不能热改 | restore 重做 capture，比较 effective config、implementation、resources、host；EXP session/checkpoint 又绑定 schema/policy/role。AKA:src/aka/bootstrap/continuation.py:67、:161；EXP:tools/python/aka_experiment/schema_sessions.py:54、:71；EXP:tools/python/aka_experiment/schema_io.py:356。 |
| Coder 只能落实指定设计 | 数值通过不证明 fidelity；Require/Forbid 不可自降，普通阻碍不写进 REALIZATION.Feedback。KS:IMPLEMENT_SCHEMA.md:38、:51、:63、:70；KS:schema_docs/6.realization.md:44。 |
| Schema 身份不只是 Version | effective_identity 哈希 node/source_manifest/origins，不含规则、dispatch、workload；非 root 自包含阅读用 effective_document。KS:schema_template/tools/design_resolution.py:39、:47、:53。 |

原生还拒绝“源码与 episode baseline 完全相同”的 single-file candidate，提示 pivot 或另选源码；所以已有实现验证不能强行伪造一个改码 episode。AKA:src/aka/task/candidate_workspace/source.py:327。
原生 production `_finish` 检查的是 canonical 最终源码；若只有未晋升的合格实现，不能把 canonical V0 包装成成功产品。AKA:src/aka/workflow/light/orchestration/campaign.py:2572、:2583。
EXP control 仍建立 SchemaAgent/DevelopmentAgent 包装，只是不投递 schema；它不是未来“完全不加载 schema 功能”的 off 合同。EXP:tools/python/aka_experiment/launcher.py:69、:81；EXP:tools/python/aka_experiment/schema.py:299。

## 3. 三种可维护方案的实际比较

| 候选方案 | 优点 | 必须承担的改动与代价 | 判断 |
| --- | --- | --- | --- |
| A. 薄 Agent decorator + native loop 的有限策略接口 | 静态 context/只读挂载可借 EXP；保留整个 Light 生命周期 | decorator 无法拥有 planner 串行顺序、source seed、attempt 预算和晋升前 fidelity；必须把准入/终态策略引入 native loop，容易让 Schema 状态与原生 state 混住 | 能做，但“只包 Agent”不够；若要求严格沿用 V1，可作为替代，接点会更多 |
| B. 可选 Workflow provider + 单次 native episode 驱动 | planner/attempt 状态有一个明确所有者；原生 episode、恢复、测量和晋升继续只有一个实现；off 选择原 provider | 需稳定一个窄驱动入口、提供固定任务/source seed、候选资格检查和结果投影；启动仍须遵守 managed 内部契约 | **推荐候选**；已有 search driver 证明驱动形态可行，不代表 schema API 已存在 |
| C. 外层脚本每轮启动一个完整 native campaign | 每个 schema 可绑定独立 frozen launch，快速演示静态输入；更少触碰内部 loop | 反复 V0/V1、重置 native 计数与 incumbent，partial 导入、跨 campaign ABBA、公平预算和恢复全需额外协调；自由优化 prompt 仍冲突 | 可作短期研究原型，不足以满足长期固定 schema 多 episode 的目标 |

B 与 C 的关键区别：B 在同一个 native workspace、Task、incumbent 和累计预算内前进，切换的是合法运行状态；C 重建整场运行。
B 不直接复用整个 EpisodeSearchDriver：它要求已验证 bootstrap，只返回 accepted kernel，还有 trajectory 复制/广播语义，不返回未晋升的 schema 完成结果。AKA:src/aka/workflow/light/search_driver.py:45、:100、:202、:217。
只借其“有限次调用现有 engine，读取持久终态”的模式，并补齐外层 budget/fidelity/partial 语义；不引入 PooledSearch、多 trajectory 或 epoch barrier。
候选 driver 的上界要绑定预期 episode 序号：若启动恢复已经结算了该 e，直接返回其结果，不因“本次尚未新跑一个 episode”而再开 e+1；所有硬预算仍来自同一 W。
也不能仅实现公共 Campaign Protocol 就接现有 startup：startup 私下调用 `_load_problem/_campaign`、`campaign.execution` 与 record_context。AKA:src/aka/workflow/legacy/startup.py:111、:155、:169。
候选 Schema provider 应复用 managed 构造/启动/清理适配，在固定 AKA 版本内提供小型显式 façade；不在运行时 monkey-patch 私有方法，不复制整个 orchestration/campaign.py。
是否抽出这一 façade 为通用上游接口，由首条垂直切片验证所需字段；不先宣布一套全面公共 plugin SDK。

## 4. B 所需的最小通用接点（全部为候选）

| 候选接点 | 内容及落点 | 默认/off 行为 |
| --- | --- | --- |
| 单次 episode driver / assignment | 一次调用只恢复或完成指定 native e；assignment 只传受信任的任务描述、source seed 引用、opaque 运行身份和预算上限；从 archive 返回 exact source/receipt/stop facts。候选新文件：AKA:src/aka/workflow/light/episode_driver.py | 不传 assignment 时原 run/_prompt/预算照旧；原生主流程无须改用该 façade |
| 可选 source seed 与任务 prompt | 在新 worktree 由 canonical 创建、runtime clean 检查后，首次 register draft 前安装完整 seed；固定任务正文替换自由探索目标，保留测量/报告/隔离指令。实际接缝：AKA:src/aka/workflow/light/long_horizon/campaign.py:1811、:1831、:444 | seed=None、prompt override=None 保留现状；禁止覆盖正在恢复的 draft |
| 候选额外资格检查 | 对已 seal 的准确源码，在 ABBA 前返回 pass/fail/unknown 与证据引用；core 只理解“是否可进入晋升”，不理解 Schema 字段。实际接缝：AKA:src/aka/workflow/light/long_horizon/campaign.py:1032、:1055 | absent 时无额外检查/调用；原 correctness/production/ABBA 条件不放宽 |
| 受信任 invocation context | role/purpose、只读输入和会话 scope 的有限契约，贯通首次、resume、输出上限重开；EXP role/context 与 namespace 补丁可借。候选修改面：AKA:src/aka/contracts/agent.py 及 Agent/Light 调用链 | 默认空值保持原 CLI 命令和存储策略；不支持的 provider 在 schema-on preflight 明确拒绝 |

这些是少数显式参数与一个资格回调，不是 before/after_everything hook framework；Schema 启停、planner 决策和 revision ledger 都留在可选 provider 内。
不要让 callback 改写 core state、当前配置、canonical HEAD 或 GPU 结果；结果只允许带对象引用、预算申请和可校验结论。
资格检查宜在 seal 后、ABBA 前；这样 report accepted 仍只代表原生报告合同，Schema fidelity 有独立状态，也不会让不符设计的“更快代码”先晋升。
资格检查结果须写入原生 active/终态证据，恢复与普通路径调用同一逻辑；已完成的晋升按原 native audit 恢复，不重新评审改写历史。
第一条切片不必新增 native terminal enum：真实 episode 预算耗尽可留下 `invalid_handoff + timeout` 原事实，Schema 层据稳定源码/进度另判 incomplete，不能改写原生事实为成功。
如果之后需要“工程进度已存好，主动让出 episode”而非等时间片结束，再研究 feature-gated `checkpointed` 通用终态（候选）；它必须无正确性/晋升含义，不能滥用 pivot/blocked。

## 5. 建议的数据对象与身份（全部为候选）

| 对象 | 最小内容与边界 |
| --- | --- |
| Workflow run W | 冻结公共 Task/Contract、workload/验收/计时范围、目标栈、KS/AKA 实现与规则版本、初始输入、策略和全局预算；拥有一个 native campaign |
| Schema revision R | 不可变 raw 文件集合及 dispatch、规则身份、选定节点/实现范围、source_manifest/origins/effective identities、父 R、变化说明、生成操作/反馈引用；不是 raw Version 或 vN |
| Implementation attempt A | 固定绑定一个 R 和实现范围；源码起点、预期完成条件、native episodes 列表、累计预算、最新 coherent source、当前 R 的最佳已验证 source；可跨多个 e |
| Native episode e | 保留 native 编号/版本/base_commit/报告/晋升事务；额外绑定 W/R/A/assignment 引用；一个 e 内可有多次编译、GPU eval 与模型 invocation |
| Agent session s | 绑定 W/R/A/e、角色、provider、上下文摘要与 workspace；同 e 可恢复或重开；新 R 必须新 scope/session；不把 session 当设计版本 |
| Source candidate C | 完整 source declaration/文件清单/字节 hash/来源 snapshot，记录 parent seed；可 incomplete、compile-failed、correct、fidelity-pass 等，不因保留而成为 incumbent |
| Global incumbent G | canonical native 晋升保留源码及其原 R/评测身份；initial G 可是任务 V0；跨 R 保留，不随 planner 创建新版本重置 |
| Feedback F | W/R/A/e/C/测量引用、观察/归因/建议分栏、覆盖与未执行项、已知缺口；trusted measurement 由 host 投影，coder 自述保留为 claim |

R 的 revision id 是外部发布身份；原有 raw Version 与 effective_identity 原样保留，不强行等同。任一完整 bundle 变动都可追溯，不靠哈希自动宣布发生一次语义优化。
同一固定 R 实现期间不热写 REALIZATION.Feedback；Coder 先输出有据反馈 proposal/sidecar，后续显式发布时才按 KS 权限纳入新 bundle。
工作流计数不进入 Schema YAML。KS 已明确 workload/目标/计时范围留在任务交接，普通交付不强制某份固定报告格式。KS:IMPLEMENT_SCHEMA.md:15、:74、:76。
以上对象是软件编排需保留的最小运行记录，不把它们回灌成语言的新必填字段，也不要求手工使用者填全部 ledger。

## 6. 串行状态机与决策条件（候选）

~~~text
INIT_TASK → [existing R | design | extract] → CHECK_AND_PUBLISH_R
  → OPEN_ATTEMPT_A → ADMIT_OR_RESUME_e → IMPLEMENT_AND_MEASURE
  → SETTLE_NATIVE_e → CLASSIFY(A, R, C, receipts)
      ├─ incomplete / repairable defect → CONTINUE_IMPLEMENT(same R, same A, next e)
      ├─ correct + required coverage + fidelity → IMPLEMENTED_R
      │    ├─ requested goal met / implement-only → STOP
      │    └─ comparable feedback + remaining optimize budget → OPTIMIZE
      ├─ evidence points to design conflict → OPTIMIZE_OR_REJECT_R
      ├─ infrastructure pending → WAIT_OR_RECOVER(same e/A/R)
      └─ global budget / cancellation / unresolved premise → STOP_WITH_ARTIFACTS
OPTIMIZE → [keep R and explicit coder action | publish R' and open A' | reject proposal | stop]
~~~

| 决策 | 具体触发及限制 |
| --- | --- |
| continue implement，同一 A/R | 未连通必要路径、缺 dispatch、代码违背既定设计、compile/correctness failure 有可执行修复，或时间片到期仍有余额/进展；优先同 episode 可用的原生恢复，终态之后才开下一 e |
| complete implementation | 当前准确 C 覆盖事先选定范围，Contract correctness 和要求的 fidelity 都有证据；性能要求若包含验收门槛亦须满足；是否胜过 G 单独报告 |
| optimize，已完成后 | 请求本来包含动态循环，已有可比全路径性能/数值/资源证据和具体设计假设，尚有 planner/实现预算；默认每次完整 A 最多一次 planner 决策，不按每条 GPU 反馈调用 |
| optimize，未完成前 | 已定位到 Schema 字段/角色/动作的缺项、冲突、资源不可能性或目标工具链能力矛盾；至少有可核对推导、最小反例或真实编译/运行证据。仅“写不出来”或“太慢”不够 |
| reject proposal | Planner 越权改 Contract/设备/误差、漏必需设计、输入身份不符或检查失败；保留原 R/A，退修受独立上限约束，不提前取消旧设计 |
| reject implementation | fidelity 失败、错误/覆盖不足、证据 source 不匹配；保存 C 和原因，通常继续同 R 修复，不自动提升到“设计不可实现” |
| reject current R | 经归因确认必须改变它才能实现，且政策允许设计回退；标记证据和被否定假设，再由 planner 出新 R。没有可用下一设计时返回 infeasible/undetermined，区分证明强度 |
| stop | implement-only 已完成；动态目标已达；用户取消；全局/attempt 预算耗尽；有限轮次无新证据/可行行动；必需 Contract 信息缺失。返回明确状态、最新 C 与 G，不用退出码 0 推断完成 |

Planner 返回 keep R 必须附具体继续行动/待收集证据；同一反馈签名、同一建议不能无限重复调用 optimizer。
无晋升次数只影响原生 stall 统计，不等于实现无进展；Schema 层看已连通路径、错误减少、设计映射和可信新证据，不只看速度。
设计不可实现须区分：任务前提矛盾交任务方；设计层选择冲突交 planner；API/编程错误交 coder；规则/checker 冲突交工具维护问题。
`blocked`、`pivot`、`invalid_handoff` 仅作为原生事实进入分类，不能一对一映射成 optimize、revision rejected 或“实现完成”。

## 7. 源码起点、固定 schema 多轮与新 revision

1. **第一次 implement（候选）：**任务自带源码/明确 skeleton 是 seed；若 extract 从 G 得到 R，应先核对 G 与 R 的来源/范围，允许做无修改的验证尝试，不能伪造代码变化满足原生 seal。
2. **同 e 的恢复（候选）：**使用原生 registered draft/session/checkpoint，不重装 seed；原生 prepare 根据 controller digest 保留 crashed invocation 的 draft。AKA:src/aka/workflow/light/orchestration/episode_workspace.py:119。
3. **同 A/R 的新 e（候选）：**默认续接最近 host 取得的稳定完整 source snapshot，即使未通过编译、尚未测量或未晋升；保留另一个 best_validated_for_R 引用以供显式回退。
4. **新 R'（候选）：**默认从父 R 的最近合格实现开始改；若父 R 未完成，planner 需指定可复用 partial 及失效部分；彻底换设计时可显式选 G/初始 skeleton，记录理由，不隐式清空代码。
5. **G 与 seed 分离（候选）：**每个 e 的 Git base 仍是当时 canonical G 的 commit；再用 install_source 装 seed 字节。candidate ancestry 保持对 G 有效，ABBA 比较对象仍是 G，不是 partial seed。
6. **导入点（候选）：**seed intent 先进入 active 身份；新 worktree 链接资源并通过 clean 检查后，首次创建 Agent draft 前安装。恢复 preparing 可幂等完成，恢复 exploring 绝不覆盖正在续写内容。
7. **导入完整性（候选）：**包括删除/清单与源码格式，不复制 `.git`、evaluator、session home、旧 active state；immutable/source policy 不相容时拒绝导入，而非修改冻结声明。
8. **任务范围（候选）：**先沿 root 连通完整调用，再落实明确选定的必要节点/dispatch；部分 root 只能记进度。root 的 when 可窄于 Contract，不能据文件名默认它覆盖剩余输入。KS:schema_docs/0.common.md:148。
9. **范围缺口（候选）：**没有合适设计覆盖必需输入时交 planner/任务方；不把旧 kernel fallback 偷当成该 R 的实现，也不把 dispatch 中文笔记当可执行 selector。KS:schema_docs/0.common.md:184、:190。
10. **输出（候选）：**同时给 `implementation_for_R` 与 `global_incumbent`，各自绑定源码/Schema/测量；正确但慢的 C 可作为 R 完成结果和 R' seed，不能靠修改原 ABBA 使它假晋升。

若 native capture 报 source unavailable，候选 adapter 不得把缺失产物当最新 seed：保留原 draft/故障证据并恢复取证，或显式回退到上一 coherent C；不能自动从 G 重来并抹掉原因。
源码安装并不授予晋升权：native promote 仍要求 canonical 没前进且 candidate 是其后代，固定上下文/source declaration 不得变化。AKA:src/aka/task/candidate_workspace/promotion.py:18、:21、:40。
若 seed 恰好就是当前 G 且任务仅需确认实现，候选 verify-only 路径复用 host 测量与 fidelity，结果明示“未运行改码 episode”；无变化不是失败，也不是一次新的 native promotion。

## 8. V0/V1、策略与预算处理

**推荐首版候选：**保留原生 V0/公共 Task/evaluator 冻结，Schema 初始化在首个 coder episode 准入前完成；schema-on 明确配置 `framework_baseline=never`、`goal_episodes=false`、`convert_after=0`，首版只支持串行、明确选定的 framework。
这些是启动前形成并冻结的 on-profile 策略，不是运行中改配置；未知 framework、与 R 硬条件冲突的目标、运行中切换 schema 开关均在准入时拒绝。
同时让 Schema driver 持有设计级 stop，避免 native `max_stall`/target_util 在当前 R 尚未实现时截断循环；原生硬预算仍必须遵守，不能绕过。
原生 goal 指令会要求探索 materially different approaches，conversion 可容忍性能回退；固定设计期间不应隐式启用。AKA:src/aka/workflow/light/long_horizon/campaign.py:510、:1075。
如果用户要求保留原生 V1 语义，备选是把 R prompt/context/预算/fidelity 同时接到 V1 和普通 e；V1 暂存/修复、baseline pin 和 Gluon 限制还需专项处理，维护面明显更大。
跳过 V1 不等于跳过正确性或 production policy；也不把成功实现强制 pin 为全局最优。候选 Schema provider 负责独立收尾：从已封存 C 导出当前 R 的实现/验收结果，不能调用只看 G 的 native `_finish` 来判定 schema 是否成功；原生 canonical 最终合规情况作为 G 的独立结果保留，off 继续原收尾。
候选预算使用 `B_initial > B_revision` 的**默认分配倾向**，不先定小时数；两者是 A 的总预算，均可容纳多个固定长度 native episode。
候选分层限额：W 的真实 elapsed 上限/总模型额度/总 GPU 额度，A 的 active time/episodes，单 planner 操作的调用/时间限额，native e 的现有 active deadline。
续 session、重开 e、Planner keep R、修复 R' 或恢复进程都不得退还已消费额度；新 A 从 W 剩余额度申请，不凭版本号得到全额新预算。
原生 record_usage 按 receipt 只累计增量，值得复用；但 native state.tokens 的默认统计不能覆盖新 planner/fidelity 调用。AKA:src/aka/workflow/light/long_horizon/store.py:85；AKA:src/aka/workflow/light/long_horizon/campaign.py:1724。
候选 W ledger 收集角色与操作级 receipts，去重累计 coder、planner、reviewer/fidelity、失败/重试用量；未知用量标 unknown，不能记零并继续无限准入。
原生 token 限额是 invocation 间检查，不能声称调用内严格硬停；候选需声明最大单次超额风险或使用后端可保证的上限，并在准入时保留相应额度。AKA:src/aka/workflow/light/long_horizon/session.py:230、:270、:293。
API 限流、GPU 队列、远端 unknown submission、恢复等待分别记账；等待既不是设计失败，也不自动清空 episode active budget。原生 active deadline 的 inflight 不确定性须明确调和。AKA:src/aka/workflow/budgets.py:28、:64。

## 9. Planner、Coder、只读材料与符合性证据

**Planner 输入候选：**已确认公共 Contract、目标硬件/编程资料与版本、workload/误差/计时范围、当前 R/来源闭包、选定范围、可信源码/评测快照、可复查反馈；design 另提供有限个近邻参考及来源理由。
近邻选择先复用 KS catalog/明确索引，完整知识库和所有历史 transcript 不必每轮灌入；参考只供借鉴，不得覆盖公共 Contract。
extract 初始化调用其已有入口并接收稳定交付；普通手写/curated/design 输入无须伪造 P6 receipt。extract 的 skip-hardware/不完整验收前提必须在 implement preflight 补齐。KS:schema_extract/tools/publication.py:71、:82；KS:DESIGN_SCHEMA.md:11。
Planner 只写候选 Schema 工作区；已发布 R、源码、evaluator、规则与参考为只读。host 检查 diff 权限、全任务结构和引用闭包，必要设计未完成则保留 draft，不能准入 coder。
Planner 使用 KS 自己的 design/optimize 能力和结果合同（尚待建设），不是将现有 `_run_session` 塞入新角色字符串；原生辅助 role 需在 WORKSPACE_LAYOUTS 注册。AKA:src/aka/workflow/light/campaign.py:103。
**Coder prompt 候选：**明确“实现 R 的指定范围；在未规定的代码细节上调优；不得修改 Contract/Design/Require/Forbid”；列 root→必要 dispatch 的进度顺序和这次继续点。
prompt 保留 native 工具/测量/report/隔离规则；删除/替换自由更换算法、扩大 roadmap、自动 DSL 转换的任务指令，不能靠最后追加一段 policy 去抵消矛盾正文。候选固定任务约束还须进入 handoff repair、timeout resume 和输出上限重开，不能只覆盖首次 `_prompt`。
Coder 得到完整 root/dispatch、选定节点有效视图及 raw/source mapping、规则/Template/IMPLEMENT 文档、上次 exact source/诊断；其余文档用只读查询按需读。
候选工具只需 `schema read/effective`、`schema check --read-only`、有限 reference 查询和外部 progress/feedback 记录；具体命令名未定，不另造 parser/展开器。
Template `validate_documents` 可直接复用，但明确 dispatch/local repository 检查另行调用；不能只调这一函数便声称整个交付包通过。KS:schema_template/tools/schema.py:75、:84。
检查工具必须禁止默认写回规范化；schema/docs/rules 作为独立只读输入挂载在 source tree 外，恢复每次校验摘要，不能被 coder 改名覆盖或通过同目录 writable parent 绕过。
read-only 文件模式只是一个层次；使用 native isolation/access 强制挂载及运行前后 hash 复验，不把 chmod 或“上下文已投递”当遵循证明。EXP:tools/python/aka_experiment/schema.py:107、:309、:323。
候选 coder session 仅获得当前 R；planner/editor 与 fidelity reviewer 各有独立 workspace/scope，无需将所有 EXP 消融 transcript 互斥政策推广到产品。

**候选 fidelity 证据分三层，逐项注明 checked/failed/unknown：**

- 身份/范围层：R/rules/source digest、选定节点/when、必要 dispatch 覆盖、Require/Forbid 是否全部有落实位置；机器可检查的部分直接检查。
- 设计落实层：数值路径/精度、work schedule/launch、role/同步/寿命与复用等关键承诺对应到准确源码和必要编译/trace 证据；Coder 映射只是输入，不能自签通过。
- 非机械语义层：独立模型或人工 reviewer 读取冻结 R+C+具体证据给有限结论，记录覆盖/不确定性/成本；Schema checker 和 GPU correctness 都不构成一般设计等价证明。
首条切片可选择能机械检查主要承诺的小案例；遇复杂流水线时必须显式引入有限 reviewer 或返回 fidelity unknown，不能把支持范围宣传为“已通用证明”。
fidelity cache 的键至少绑定 R/rules/C/检查器版本与范围；同字节可复用本次证据，不因新的 episode/session 重审全部材料；Schema 变更须重新评定受影响项。
一般 implementation failure/瓶颈反馈留在 F；只有针对 Prefer 的真实可比实验替代才产生 REALIZATION.Feedback proposal，且不豁免 Require/Forbid。KS:schema_docs/6.realization.md:44、:52、:58。

## 10. Exact source/evaluation 绑定与恢复原子边界

候选反馈链为 `R + scope + C(source identity) → native experiment/record → evaluator+input+target+policy+timing identity → observed facts`，ABBA 再增加确切 G 的身份。
原生 Runtime 已从 workspace snapshot 准备请求且保留 Task 输入权威；Agent 请求不能换 evaluator。复用此权威，不让 planner/coder 自报延迟 JSON 进入可信事实。AKA:src/aka/workflow/light/orchestration/supervisor_runtime.py:683、:689、:748。
原生 ABBA 要求 candidate、incumbent、inputs、evaluator 和完整 policy 匹配；从 committed candidate 读字节。AKA:src/aka/workflow/light/long_horizon/recorded_verifier.py:70、:81、:107。
candidate sealed 后 coder 的晚写不能改变评测对象；原 publish 已保留 sealed source 并给 integrity warning，Schema adapter 也须引用 seal 对象。AKA:src/aka/workflow/light/orchestration/episode_workspace.py:217、:234。
同一 C 在两个 R 下物理测量条件完全相同，可以复用 native measurement，但为新 R 独立记录绑定和 fidelity；无需仅因 schema 编号变化制造新的 GPU cache key。
不得用最后一份日志对应当前源码：缺 source/coverage/environment 身份的结果只能是诊断材料，超时或工具 unknown 也不能假判性能失败。

**候选原子边界及崩溃处理：**

1. W 冻结一次 code/config/rules/Task/policy；动态 R 是 append-only runtime artifact，不列入会随写入变化的 Bootstrap resource 目录 fingerprint。
2. Planner 先记 operation intent（输入 R/F hash、预算、角色/session）；draft 检查合格后写完整 immutable bundle，再原子发布 manifest/active R 指针；失败或崩溃保留旧 R。
3. 分配 A/e 前记录 assignment 与 seed/ref 的摘要；native 创建 active 时持有同一 assignment id，二者不一致则停止调和，不能“猜最近版本”。
4. 恢复时先结算 native active/report/promoting/audit，再把其已完成结果幂等吸收进 A；不能在上一 GPU/promotion 事务尚未定案时 optimize。
5. 终态 C/receipt 封存后才记录 A 分类与预算/下一动作；crash 后按 operation id 重放收据，不重复 episode、planner 发布、计费或 promotion。
6. 新 R' 仅在所有旧会话/相关工具完成或有明确保留 fence 后准入；新 session context 绑定 R'，旧 R session/checkpoint 不可复用。
7. 状态损坏不回退到“新 run”；adapter 检查 native archive/active/W ledger 的一致性，缺预算或关键 receipt 则明确受阻。原生 load_state/load_active 对解析失败会返回空，不能盲从。AKA:src/aka/workflow/light/long_horizon/store.py:76、:99。

关键区别：冻结的是实现/规则/任务/策略与启动选择；运行中新增已发布 R 是该固定状态机的产物。禁止修改 EXP 原 SchemaBinding.manifest 或让 Agent.configuration_identity 随 active R 变化来伪装热更新。
原生 restore 比较 capture 的规则保留不变；每个会话绑定它被准入时的 R。若实现库/规则/Task 本身变了，应新 run 或明确版本迁移，不能复写旧 selection。
不宣称一般分布式事务“exactly once”：用唯一 intent、幂等结果吸收和 native 已有晋升审计实现可核验恢复，远端未知执行由原 Experiment/Execution 先结算。

## 11. 三仓职责、EXP 可借部分与依赖

| 所有者 | 建议候选职责 / 放置位置 |
| --- | --- |
| KernelSchema | 四流程公共任务/产物合同、docs/template/wiki、design/optimize 执行能力、Schema 输入/权限/范围验证、fidelity 公共规则与 feedback 语义；standalone demo 编排使用同一 API。新包/目录由独立产品研究确定，本文不定最终名称 |
| AKA 通用 core | 单次 episode façade、可选 assignment/seed、准确 sealed source 的额外资格检查、必要的 Agent role/context/scope 传递与证据；候选落点见第 4 节，不出现 KernelSchema YAML 知识 |
| AKA 可选 schema adapter | composition/profile、SchemaWorkflow 串行 driver、R/A 与 native identity 映射、coder prompt/只读工具接线、测量反馈翻译/输出。候选路径：AKA:src/aka/workflow/schema/；可只在适配 fork 维护 |
| AKA_experiment | 数据集/实验矩阵、消融条件、run 选择、外部 wall-clock/公平成本统计、实验隔离与资格验证；消费固定 adapter/KS 版本，不拥有另一份完整 planner/schema 产品逻辑 |

**EXP 补丁的选择性借鉴（候选迁移判断）：**

- `explicit-agent-role-context-v1`：优先评估为独立通用补丁；含首次/恢复两条调用链及 Claude 原生 context，不能只搬 contracts。EXP:control/aka-changes.json:6；EXP:assets/aka/src/aka/agent/backends/claude.py:55。
- `agent-session-private-boundary-r1`：复用 namespace/implicit mount exclusions 的必要部分，保证 R/角色不混用；根据 schema session 测试验证，不要求照搬实验 condition 名称。EXP:control/aka-changes.json:237。
- `agent-storage-policy-r2` / `agent-owner-lifecycle-coordination-r3`：当实际隔离/共享 owner/导出需要时保留其通用能力；不默认把 EXP 全部 cross-condition provenance/重发布协调系统变成 Schema MVP 前置。EXP:control/aka-changes.json:379、:520。
- evaluator score carriers、诊断基线、pilot pause、Wiki 安全子进程和环境资格补丁：分别看所选任务/工具是否需要；它们不是 planner 生命周期，不能为了 schema-on 成批搬入。EXP:control/aka-changes.json:589、:993、:1078、:1109、:1135。
EXP 当前工作树仍可能有并行既有工作；本轮只确认台账/实现存在，未重新 verify 全部来源，不将 R1 的差异数量当本轮实时结论。

**依赖方向候选：**AKA fork 的可选模块 → 固定 KS submodule；EXP → 固定 AKA fork 及其 KS gitlink。KS 的基础安装/导入不反向依赖 AKA。
KS standalone implement 可以通过可选、外部安装的 backend/命令调用 AKA，属于运行时可选适配；KS 仓内不再嵌套 AKA，避免递归 submodule/构建循环。
每次运行只有一个循环所有者：候选 KS demo 调 AKA 的固定-R implement 后端时，AKA 不再自动启动另一套 optimize loop；AKA SchemaWorkflow 直接调用 KS design/optimize 能力，不回调 KS 顶层 demo。
候选 submodule 路径可为 `AKA:optional/KernelSchema`，这里只是位置示例；off 时不初始化、不 import、不校验该目录，schema-on 缺模块才报告清楚的安装需求。
公开分发应固定 AKA upstream SHA、fork patch SHA、KS SHA、规则/Prompt/工具身份；resources 可只包含实际使用文档/代码和已选 references，不必递归初始化所有第三方库。
现 EXP vendor 投影会排除 gitlinks 和 .gitmodules；消费未来带 KS 的 fork 时必须显式 materialize/固定它或改用 checkout，不可假设普通 vendor 已携带 KS。EXP:tools/lib/aka_vendor.py:93、:98。

## 12. 分期验证路线、off 与额外模型成本

**以下为候选探索顺序，用户定方向后才写完整落地计划：**

1. 在干净、明确固定的本地 AKA baseline 上形成最小补丁清单；EXP 仅作为材料，不直接把当前脏 vendor 当新产品基线；远端 fork/发布另按后续授权处理。
2. 首条垂直切片：一个已有完整 R、明确固定任务和单一已支持 Agent，运行“partial e1 → 同 R 从 C1 续接 e2 → correctness/fidelity 完成”；先用 fake Agent/Evaluation 验证状态，再在获准的独立阶段做真实任务。
3. 同切片覆盖“正确但比 G 慢”仍能返回 C2、“不符设计但更快”不可晋升、seed 安装/终态吸收处故障恢复、off 不依赖 KS；这比先接 design 更能暴露集成核心问题。
4. 加已有 extract 输出/手写/curated 输入；覆盖 zero-edit verification、root+必要 dispatch、不可比较反馈。Schema 文件语法通过不能替代任务 preflight。
5. 接 KS design→R0，再接一次 optimize→R1；验证旧 partial/G 不混用、旧 session 不复用、同 run 冻结 identity 不变、全角色成本累计。
6. 最后扩展有限动态循环、复杂 fidelity、多 provider 与公开 demo；把稳定通用小补丁独立准备 upstream，Schema orchestration 继续可选，EXP 只升级固定依赖和实验配置。

**候选 off 合同：**选择原 Light provider/原 Agent，完全不加载 KS 模块、没有 schema prompt/tools/mount/ledger、零 planner/fidelity 调用，保持原默认 V1/goal/conversion/预算/导出逻辑。
用固定输入的 fake Agent/Executor 对照 normalized 调用轨迹、prompt、CLI 参数、资源和预算/晋升结果；排除 UUID/时间差，不声称随机模型输出字节等价。
通用补丁改变了实现 fingerprint，因此旧运行必须用其原版本恢复；“off 行为保持”不等于新二进制可绕过旧 frozen continuation 的版本检查。
**候选额外成本界限：**off 的 schema 新增模型调用为 0；固定已有 R 的 implement 无 design/optimize/extract 调用，仅选用 fidelity reviewer 时有显式新增成本，context 变长也计实际 token。
动态模式最多一次初始化操作、每个已结算 A 一次受限 optimize 决策；每个准确 `(R,C,scope,checker)` 的 fidelity 不重复调模型，禁止每条 GPU record 自动调 planner。
候选调用上界可写为 `N_extract_actual + Q_design + O_max × Q_optimize + E_max × Q_fidelity`，各 Q 包含失败/退修/重试，现有 native reviewer/coder 成本另全部入 W 总账。
extract 的一次操作可能本来含很多模型调用，不能按“初始化 1 次”当成 1 次模型；未配置相应上限时不承诺公式给出硬成本保证。
若 reviewer 因预算/资源无法运行，fidelity=unknown，返回未完成相应验收；不得自动省掉 gate，也不需要为每次完成判断额外添一个无限制“管理模型”。

## 13. 需要用户选择的少数事项与可验证未知

**建议留给用户的选择（不是本轮已定）：**

1. 是否接受推荐 B，以及 schema-on 先停用原生 V1 自动代码生成、统一由 implement episodes 承担初次实现；若必须保留 V1，应接受更大的适配面。
2. 第一版目标是“固定 schema implement 可复用”还是同步展示 design→implement→一次 optimize；推荐先证明固定 schema 跨 episode，并选择一个验收/资料闭合的小任务。
3. 动态 optimize 的授权与成功目标：默认完成后按反馈改设计，明确设计阻碍可提前交 planner；同时确认需要的性能门槛与复杂案例的 fidelity 证据强度/额外模型预算。
4. 是否接受 AKA fork→KS submodule 的单向可选依赖及 EXP 只消费；具体目录/接口名和哪些通用补丁投上游不必由用户逐项设计。

**可以技术验证消除，不宜转成用户选择的问题：**

- 单次 driver 是否能无额外 V0/V1 模型重复地恢复同一个 campaign，是否正确限制 native version/token/episode admission；不能照抄 Search driver 省略其外层预算。
- `framework_baseline=never` 在所选 production task/格式下的 V0 provenance、manifest pin、native `_finish` 和 schema C 独立导出是否自洽。
- install_source 在三种 source format 下对删除/固定声明/崩溃恢复的表现；新 seed 不改变 canonical、ABBA base 或覆盖旧 draft。
- native report/额外 fidelity/ABBA 的顺序与恢复一致性；测过 C1 后改 C2、跨 R 重放 receipt、同字节 no-op 等反例必须保持身份边界。
- Agent 首次/resume/输出上限恢复及 sandbox 的真实 context 能力，旧 R/辅助 reviewer 是否可看到不应继承内容；unsupported provider 不静默退化。
- W/R/A 与 native active/archive 不一致、planner 发布中断、promotion 已提交未返回、预算 inflight 等故障点的幂等恢复；不能用空状态默认值退款重来。
- off 是否真正无 KS 路径依赖/额外调用，submodule 未初始化能否正常跑原 native 路径；分发时实际需要哪些 KS 文件与第三方资料。

本轮结论是推荐一个可验证的架构方向和最小必要接缝；接口定稿、默认预算数值、正式任务/demo 选择、远端 fork/submodule 操作及完整实施排期均保留给用户决策后的阶段。
