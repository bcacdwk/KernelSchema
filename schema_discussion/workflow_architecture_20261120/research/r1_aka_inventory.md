# 第一轮：AKA 与 AKA_experiment 现状调研

调研日期：2026-11-20。状态：只读现状报告，供下一轮架构研究使用；不是已批准的实施方案。

证据前缀：**AKA** 指 /home/shaohanyong.shy/atrex-kernel-agent，**EXP** 指 /home/shaohanyong.shy/AKA_experiment。下文“仓库前缀:相对路径:行号”均指本轮读到的当前工作树；涉及上游与本地补丁时分别标明。Git SHA、文件模式和文件哈希比较是本轮只读命令的观察事实，没有对应源码行时直接注明命令。

已阅读指定 AGENTS.md、讨论 README、00_user_brief.md 和 01_history.md。本轮没有修改实现、运行 GPU/模型/实验、读取凭据、操作远端或改变 Git 分支；只新增本报告。自动发现的本地 Skill 仅作为被调查的软件资源，没有执行其工作流。

## 1. 关键结论

1. **两个 AKA 目录不是同一个 checkout，也不是 submodule 关系。** EXP 把 AKA 的 bf5557a 固定 Git tree 导出成 assets/aka 普通文件，保留来源台账，再叠加本地修改。独立 AKA 工作树恰好也是这个 baseline。现有关系不能写成“实验仓已经消费一个适配 fork”。证据：EXP:control/aka-source.json:3、:5、:7、:19；EXP:tools/lib/aka_vendor.py:70、:93、:98；只读 git ls-files --stage 显示 assets/aka 内源码为 100644，而非 160000。

2. **AKA 已有实际模块化执行路径，但 Light 仍是持有状态的完整优化工作流。** Agent、Problem、Source、CandidateWorkspace、Evaluation、Experiment、Execution 可按 composition 选择；Light 的 episode 调度、prompt、report、晋升、预算和恢复还由内部 Campaign 代码直接组织，并没有现成 schema planner 生命周期接口。证据：AKA:src/aka/workflow/light/plugin.py:7、:12、:43；AKA:src/aka/workflow/light/provider.py:59、:112；AKA:src/aka/workflow/light/long_horizon/campaign.py:1698。

3. **EXP 已落地的是“固定 schema 材料消融适配”，不是四流程编排。** SchemaAgent 给受信任的 optimizer 角色投递同一任务 root.yaml / dispatch.md 全文，并挂载其余文档；只支持 Claude。没有调用 KernelSchema 的 design/optimize/extract，没有 schema revision 状态机，也没有强制 root-first / dispatch 完成度判定。证据：EXP:tools/python/aka_experiment/schema.py:23、:54、:141、:158、:299；EXP:tools/python/aka_experiment/launcher.py:120、:153。限定在所查运行模块的全文检索未发现 KernelSchema / schema_revision / design_schema / optimize_schema 的调用。

4. **现有 schema 接口依赖 EXP 的本地 core 补丁，不能原样搬到未修改的 AKA。** 上游 AgentRunRequest / open_session 没有 role、context_files、session_namespace、storage_policy；EXP 才添加这些契约，并让 Light 显式传 role="optimizer"，由 Claude backend 转为 --append-system-prompt-file。证据对照：AKA:src/aka/contracts/agent.py:199、:237；EXP:assets/aka/src/aka/contracts/agent.py:199、:217、:272；EXP:assets/aka/src/aka/workflow/light/long_horizon/session.py:185；EXP:assets/aka/src/aka/agent/backends/claude.py:55；EXP:control/aka-changes.json:6。

5. **episode、vN、model invocation、schema bundle 是四个不同单位。** 一个 Light episode 内可反复改码、GPU 测量、修复 report，甚至换新模型会话；每个终端 episode 都可写 vN，失败/pivot 也写，vN 不等于成功 kernel 版本。当前 schema bundle 则固定绑定在运行配置上。证据：AKA:src/aka/workflow/light/long_horizon/session.py:217、:255、:394；AKA:src/aka/workflow/light/long_horizon/campaign.py:1128、:1248、:1278；EXP:tools/python/aka_experiment/schema.py:190。

6. **原生跨 episode 机制不能直接满足“同一 schema 的未完成实现接着写”。** 新 episode 从 canonical HEAD 建 candidate；未通过晋升的源码不成为下一轮默认起点。终端 attempted source 有 public-candidate.json 可保留和导出，但没有“partial implementation”一等状态或自动续接规则。同一 episode 的会话重试/中断恢复才明确保留 draft。证据：AKA:src/aka/workflow/light/long_horizon/campaign.py:1785、:1248、:1264、:1320；AKA:src/aka/workflow/legacy/episode_output.py:52；AKA:src/aka/workflow/light/orchestration/episode_workspace.py:119。

7. **“正确实现指定 schema”与“AKA 晋升”必须分开讨论。** 原生 report acceptance 检查准确源码和完整正确性，后续 ABBA 再决定是否晋升；正确但更慢的实现会保留为尝试证据而不替换 incumbent。CLI 正常退出也可能只是预算用尽。证据：AKA:src/aka/workflow/light/host/journal.py:1304；AKA:src/aka/workflow/light/long_horizon/recorded_verifier.py:107、:124；AKA:src/aka/workflow/legacy/startup.py:169。

## 2. 仓库版本、来源与工作树边界

| 目录 | 本轮 Git 观察 | 调查解释 |
| --- | --- | --- |
| 独立 AKA | main，HEAD bf5557ac04f6ff9988901d511cf25e1cbe61a1bc；工作区干净；最近提交 2026-11-17，Fix workspace checkpoint locks and initialization ancestor aliases (#146) | 本报告“上游原生”均指这个本地固定版本；没有联网核实远端最新性 |
| EXP | schema-ablation-261120，HEAD 76dabfbf08e700e479b0262d448df578e74bfe35；最近提交 2026-11-19，Adapt AKA for ablation | 当前工作树含大量此前已有修改、删除和新增；不能把当前所有代码算作 HEAD 或已稳定功能 |
| EXP 的 assets/aka | 声明 baseline 与独立 AKA 相同；import_commit 为 9544bb34205bff78f2bb460d67574f9df45645af；无嵌套 .git / .gitmodules | 普通文件投影 + overlay，不由 git submodule 固定 |

上述 Git 观察来自 git status --short --branch、git rev-parse HEAD、git log -1、git ls-files --stage；详细初始数量以讨论目录 baseline.json 为准。EXP 在本轮开始已有的 76 修改、4767 删除、84 项 untracked 由主代理记录，不是本任务产生。

**已提交与当前新增的具体区别：**

- SchemaAgent、schema_sessions，以及 Agent role/context 契约和 Light optimizer 角色传递，与 EXP HEAD 字节一致；它们是已存在的静态 schema 适配。
- launcher.py、runtime.py、records.py、cli.py 当前有未提交修改；runner.py、bench_pipeline.py、bench_qualification.py、readiness.py、finalize.py 等当前是 untracked。本文关于正式 pilot 生命周期的说明是**当前工作树代码事实**，不能回推为 HEAD 已具备。
- 当前来源说明明确把固定 tree 导出、三方比较、受控 apply/record-overlay 作为维护路径，工具不自动 fetch/提交；feature 与完整 baseline 升级分开管理。证据：EXP:docs/aka-integration.md:5、:50、:65；EXP:tools/lib/aka_vendor.py:211。

**本轮额外只读审计结果：**

- 执行 python3 -B tools/lib/aka_vendor.py verify，退出码 2：现有 assets/aka/src/aka/__pycache__ 触发 non-source directory 拒绝。本轮没有清理。
- 随后仅为核实来源，按 manifest 对普通源码做内存哈希比对，排除生成缓存：baseline 1713 文件，当前 1714；41 个净差异路径，台账 recorded_files 为 39。新增普通源码为 src/aka/evaluation/performance_baseline.py。
- 另外两个当前源路径尚未与台账对应：src/aka/contracts/execution_environment.py、src/aka/execution/gateway/executor.py。这两处新增 EnvironmentAdmission 及受证据约束的 ABBA environment admission 路径，不是 schema planner；不能悄悄视作原生契约。
- 这项辅助比对**不等于 verify 通过**。官方校验拒绝缓存及未登记差异的逻辑在 EXP:tools/lib/aka_vendor.py:181、:211、:225；两个当前补丁入口在 EXP:assets/aka/src/aka/contracts/execution_environment.py:107、EXP:assets/aka/src/aka/execution/gateway/executor.py:53。

## 3. 实际架构与入口

### 3.1 原生默认 Light 路径

~~~text
aka optimize / aka workflow
  -> installed aka.commands entry point
  -> bootstrap.workflow_cli：解析 typed WorkflowLaunch + profile/patch
  -> bootstrap.host.run_selection：冻结选择、Core boot、startup.run
  -> legacy.startup.WorkflowStartup（Light 也复用这个 startup）
  -> LightWorkflow._load_problem / _campaign
  -> ManagedCampaign
       -> V0：冻结公共输入、私有 evaluator、初始 source 并测量
       -> V1 framework baseline：按策略实现/验证/固定首个目标 DSL kernel
       -> LongHorizonCampaign.run
            -> 恢复 active transaction，检查预算
            -> canonical HEAD -> private candidate worktree -> Git-free Agent draft
            -> _prompt -> LongSessionRunner -> Agent.open_session / invoke
                 -> tools/sandbox.py 或 tools/plugin.py
                 -> 私有 Tool Runtime（session capability + source snapshot）
                 -> Experiment 记账/恢复 -> Evaluation 准备/解释
                 -> Execution -> Agate GPU worker / SSH backend
                 <- 测量与错误 -> 同一 coder 会话修复
            -> episode-report -> full Evaluate 引用 + 正确性/适用 policy
            -> seal candidate -> native ABBA -> 晋升或记录未晋升
            -> vN + attempts/state + terminal candidate artifacts
       -> 预算/目标/暂停等 stop reason -> final retained source/export
~~~

调用证据：

- CLI 分发不是脚本名猜测：AKA:src/aka/cli.py:11；optimize/workflow entry point 属于 legacy distribution：AKA:packages/atrex-aka-legacy/pyproject.toml:12。
- bootstrap 启动与 dispose：AKA:src/aka/bootstrap/host.py:49、:54。
- Light 复用 WorkflowStartup：AKA:src/aka/workflow/light/plugin.py:59；startup 对 frozen request、campaign lease 的处理：AKA:src/aka/workflow/legacy/startup.py:104、:149。
- public Problem、initial source、evaluator 分离：AKA:src/aka/workflow/light/provider.py:88、:142；V0 实际流程：AKA:src/aka/workflow/light/campaign.py:182、:234、:279。
- V1 在普通 episode 循环前，且不是 schema-design：AKA:src/aka/workflow/light/long_horizon/main_adapter.py:30、:42；AKA:src/aka/workflow/light/orchestration/campaign.py:1186。
- LongHorizon engine 的 max_version 来自 max_iters：AKA:src/aka/workflow/light/orchestration/campaign.py:2559。

这里至少有三种不同的“worker/agent”：host 的工作进程、模型 CLI coding agent、Agate 的远端 GPU worker。原生 Python Supervisor 是状态/验证控制器，不应凭名称当作用户所需的 schema planner。

### 3.2 EXP 的外层包装

~~~text
bin/lab
  -> environment_entry：选专用解释器和 canonical source
  -> aka_experiment.cli
  -> runner.start：readiness/冻结快照/持久 start intent/lineage 预算
  -> _owner -> supervise -> _worker（另有 _guardian 清理所有权）
  -> launcher.run_launch
       -> compose light-sandbox
       -> 把 agent provider 替为 SchemaAgent(DevelopmentAgent(native Agent))
       -> 可把 execution 替为 ObservedExecutor(native Execution)
       -> native bootstrap.host.run_selection
       -> 原生 Light 生命周期

EXTERNAL stop/resume
  -> host hold/deadline/远端结算/ownership
  -> 原生 pause marker 或原 frozen continuation
  -> 原生 episode 恢复与原预算
~~~

证据：EXP:bin/lab:3；EXP:tools/lib/environment_entry.py:7、:25；EXP:tools/python/aka_experiment/cli.py:62；EXP:tools/python/aka_experiment/runner.py:712、:431、:994；EXP:tools/python/aka_experiment/launcher.py:149、:153、:257。

当前正式 pilot 配置并不允许随意开 schema：load_experiment 明确要求 matrix.schema_enabled 为 false；pilot_config 生成 condition="control"。正式启动还受 ready / final snapshot / root-clock 等实验门禁约束。证据：EXP:tools/python/aka_experiment/launcher.py:435、:457、:490；EXP:control/experiment-matrix.json:4；EXP:README.md:3、:13。这些门禁属于当前实验操作政策，不能自动变成 KernelSchema 独立产品的前置条件。

## 4. Episode、反馈、预算与恢复的真实语义

### 4.1 不应合并的计数单位

| 单位 | 代码语义 | 对 schema 架构的影响 |
| --- | --- | --- |
| V0 / V1 | 原始 source 测量；按策略建立第一份目标框架实现 | “design 后较长 implement”要覆盖 V1 入口，不能只改后续 episode prompt |
| vN / max_iters | 每个终端 episode 写一个 canonical version fact；包括没有晋升的结果 | 不是 schema revision，也不是成功 kernel 数 |
| episode | 从一个 incumbent 起步的串行优化事务；允许多次 source 编辑、测量与修复，最终交一次 terminal report | 一个 schema 可以需要多次此类事务；现有循环没表达这种外层关系 |
| model invocation/session | 一个 episode 可 resume 同 CLI session；输出上限恢复可换新 context 而保留 episode/draft | “新模型会话”不等于“新 schema”或“新 episode” |
| measurement / Kernel ID | 源码、输入/evaluator、target、policy 等身份绑定的单次任务/记录 | 一个 episode 可有多个 candidate 和多次 GPU eval |
| schema bundle | EXP 的 task 文档快照 digest + policy，绑定 run/condition | 当前只有静态 bundle identity，没有父子 revision/optimize 事务 |

证据：AKA:src/aka/workflow/light/long_horizon/campaign.py:1785、:556、:1128、:1278；AKA:src/aka/workflow/light/long_horizon/session.py:255、:394；AKA:src/aka/workflow/light/measurements.py:347；EXP:tools/python/aka_experiment/schema.py:158、:190。

新 episode 清空 scratch，默认可继承自定义 skills/tools；这种 inheritance 与 candidate 晋升独立，**不会自动把未晋升 kernel 当作下一轮 source**。源码起点是新建时的 canonical HEAD。证据：AKA:src/aka/workflow/light/long_horizon/campaign.py:1785、:1801；AKA:src/aka/workflow/light/orchestration/episode_workspace.py:177、:189、:209。

### 4.2 状态及成功判定

| 事件/状态 | 当前后果 |
| --- | --- |
| 普通 compile/eval 失败 | 返回工具诊断；prompt 要求同 episode 修复，不能仅因一次失败结束 |
| episode-report 被拒 | Journal 留在 in_progress，coder 修正字段或 evidence 后重交 |
| candidate_ready | 必须引用准确当前源码的标准 full Evaluate，并通过独立正确性/适用 gate；不代表已晋升 |
| report accepted + ABBA 未过 | 不晋升；记录结果和终端尝试源码，增加 stall/version |
| pivot | 结束当前工程方向，写结果；不是自动调用 schema optimize |
| blocked | 文本/协议上是基础设施或缺权限；当前有效 blocked 记录后启动一个 fresh episode retry，甚至先于 max_stall 判断 |
| invocation timeout/无有效 handoff | 在有余额和 resume 额度时可续会话；否则成为失败/invalid_handoff 类终端结果 |
| 有效 handoff 后 CLI 挂到超时 | session runner 可认定 handoff 权威，改记成功退出，不把 CLI 尾部挂起当作 kernel 失败 |
| max_iters / token_budget 用尽 | 停止 campaign；Startup 可正常返回 0，因此 process exit=0 不是“schema 已实现” |

证据：AKA:src/aka/resources/prompts/research-episodes/episode.md:3、:67；AKA:src/aka/resources/prompts/research-episodes/runtime_report.md:16；AKA:src/aka/workflow/light/host/journal.py:877、:1210、:1304；AKA:src/aka/workflow/light/long_horizon/session.py:317；AKA:src/aka/workflow/light/long_horizon/campaign.py:1222、:1264、:1948；AKA:src/aka/workflow/legacy/startup.py:169。

没有原生 terminal status 叫 implementation_incomplete、schema_infeasible、schema_fidelity_failed。现有三个 report 状态是 candidate_ready / pivot / blocked；不能只重命名它们就认为具备了 schema 层语义。证据：AKA:src/aka/workflow/light/long_horizon/models.py:22；AKA:src/aka/workflow/light/host/journal.py:1239。

### 4.3 GPU 与可信反馈闭环

Runtime 从 Agent draft 取 source snapshot，并使用 retained Task 的输入权威；Agent 不能通过请求换 evaluator。SharedMeasurements 打开共享 Experiment；Experiment 准备并持久记录各步骤意图，调用 Execution，再把原始输出交 Evaluation 解释。Agate gateway 执行已准备的请求并保留 execution facts。证据：AKA:src/aka/workflow/light/orchestration/supervisor_runtime.py:678、:689、:748；AKA:src/aka/workflow/light/measurements.py:240、:347；AKA:src/aka/experiment/runner.py:370、:404、:438；AKA:src/aka/execution/gateway/executor.py:52、:92、:149。

Report 的标准 full Evaluate 引用检查在 GPU acceptance 之前；随后六 case 正确性、适用 production policy、可选 supplemental numerical 测量，再 seal 候选。晋升另走同 allocation ABBA，严格比较配置的 latency improvement threshold。普通 Evaluate、正确性通过、诊断 score 都不能代替 ABBA。证据：AKA:src/aka/workflow/light/host/journal.py:897、:1304；AKA:src/aka/workflow/light/candidate_gate.py:10、:33；AKA:src/aka/workflow/light/long_horizon/recorded_verifier.py:70、:107、:124。

可复用的反馈载体不止日志：measurement record、public typed result、report diagnosis、版本事实和终端 source artifacts 已存在。AKA:src/aka/workflow/legacy/episode_output.py:52 读取终端 working source，:74 明确未提交内容的 revision=None，:103 保留 base64 source 与绑定 evaluations；因此“未晋升就完全丢失代码”也不准确。缺口是如何把该产物授权为下一次 implement 起点，以及哪些诊断要交 planner。

### 4.4 预算边界

- 原生默认：Light episode 4 小时 active time，framework baseline 8 小时；没有默认 campaign-wide wall-clock 上限。见 AKA:src/aka/contracts/workflow.py:16、:26；AKA:src/aka/workflow/light/plugin.py:33；AKA:docs/design.md:225。
- active allowance 用 durable deadline 文件累计，resumed 状态不能缺文件，旧 wall-clock 格式或 inflight=True 需要人工/明确调和，不能重置续跑。见 AKA:src/aka/workflow/budgets.py:28、:51、:64；AKA:src/aka/workflow/light/deadline.py:12。
- 认证 Tool Runtime 等待暂停 owned Agent 进程树时可排除其队列/handler 时间；GPU、Wiki、queue、recovery 的 elapsed-time limits 仍独立。见 AKA:docs/optional-light-modules.md:137。
- Light token_budget 检查的是 state.tokens；runner 使用 terminal_usage.total_tokens or 0，每次 invocation 后记入 replay-safe receipt，再决定是否续调。它不是单次调用内 token 硬停，也不自动覆盖未来 planner 成本。V1 / review 的 _account 另累加 tokens_spent；下一阶段需验证全成本口径，不能把现有名字“native_total_tokens”视为所有角色总用量。见 AKA:src/aka/workflow/light/long_horizon/campaign.py:1721、:1754；AKA:src/aka/workflow/light/long_horizon/session.py:230、:293；AKA:src/aka/workflow/light/long_horizon/store.py:85；AKA:src/aka/workflow/light/orchestration/campaign.py:653。
- EXP 的 StopBudget 目前只真正映射 version_index、native_total_tokens、episode_active_seconds；lifetime_terminal_episodes 与 output_tokens 虽是可枚举单位，native_options 会明确拒绝。见 EXP:tools/python/aka_experiment/records.py:84。
- EXP 的 host 外层有独立 wall-clock window、lineage 累计、pause/drain/settlement，属于实验要求。当前 pilot 把 episode 与 framework baseline 都配置成 3600 秒，不能沿用原生默认来解释它。见 EXP:control/experiment-policy.json:1；EXP:tools/python/aka_experiment/runner.py:486、:768、:790。

### 4.5 恢复不是任意热切换

原生先恢复未完成 promotion/report 事务；相同 base、registered worktree、phase 等满足条件时再用旧 draft 继续，否则做中断归档或晋升审计恢复。版本、mode 和预算不会因 provider/session 重启自然重置。证据：AKA:src/aka/workflow/light/long_horizon/campaign.py:1351、:1418、:1461、:1497。

Bootstrap capture 将有效配置、implementation、resources、Core/Bootstrap/Contracts 身份一起冻结；restore 重新 capture 并逐值比较，变化即拒绝。Core 的可替换 provider 不等于运行中安全 hot swap。证据：AKA:src/aka/bootstrap/continuation.py:57、:126、:161；AKA:docs/plugin-framework.md:43、:100。

EXP 又把 session / Home / capture / checkpoint 绑定到 run、condition、task、schema digest、policy 和 role。修改 schema 后重用同 session ID 或 restore 旧 binding 会失败；这种失败是现有隔离合同的设计结果，不应绕过它实现 optimize。证据：EXP:tools/python/aka_experiment/schema_sessions.py:54、:71、:99；EXP:tools/python/aka_experiment/schema_io.py:290、:356。

EXP clean resume 还要求没有 active_episode，interrupted resume 则恢复原 selection；未结算 GPU job、原配置不一致或 frozen source 变更都会阻挡恢复。当前新增的 pause-requested 判断只关闭新 episode admission，不打断已恢复事务。证据：EXP:tools/python/aka_experiment/runner.py:663、:840；EXP:assets/aka/src/aka/workflow/light/long_horizon/campaign.py:1738。

## 5. 模块与自然扩展点

下面“可插入”是架构候选，不是已存在的 schema API，也不表示已选择 fork 实施。

| 所有者/入口 | 已有能力与证据 | schema 接入的自然位置与局限 |
| --- | --- | --- |
| Bootstrap / Core composition | profile/patch 选择 provider，校验配置、依赖及清理；AKA:src/aka/bootstrap/host.py:54，AKA:src/aka/workflow/light/plugin.py:43 | 可用于启动时选择 schema 工作流/装饰器；patch 替换整份 config，必须保留 startup 等配置；不能直接塞未知 schema 字段 |
| Problem / Source | 冻结公共描述、initial source、私有 evaluator；AKA:src/aka/workflow/light/provider.py:88 | 可明确 design/implement 的 Task 输入；“从零”仍需要公共算子合同和可验证入口，不能把 schema 当完整 evaluator |
| CandidateWorkspace | source formats、draft、seal、archive、promotion、read working/revision；AKA:src/aka/workflow/light/campaign.py:103，AKA:src/aka/workflow/legacy/episode_output.py:52 | 可支持 root-first 实现产物与 partial snapshot；新的 schema docs 默认不应混成可编辑 kernel source |
| Resources / local tool manifest | 明确 catalog、tools/Skills/instructions/resources；AKA:docs/plugins.md:8、:87；AKA:src/aka/resources/bundles.py:57 | 适合发布 docs/template/read-only schema 查询与校验工具；单加工具不会自动串行调度 planner |
| Agent 服务 | scoped process/session/tool runtime、isolation、checkpoint；AKA:src/aka/agent/provider.py:481 | 适合投递 coder schema/context 和独立 planner 角色；现有可靠 role/context seam 是 EXP patch，需先决定通用接口的归属 |
| LongHorizonCampaign._prompt | 明确 episode/version/directions/evaluator/sandbox/research 片段；AKA:src/aka/workflow/light/long_horizon/campaign.py:444 | 最自然的 coder 任务契约改写点；目前指令仍是自由优化，静态材料附加无法撤销其语义 |
| LongHorizonCampaign.run / V1 baseline | owned lifecycle、预算、recover/new draft/report/promotion；AKA:src/aka/workflow/light/long_horizon/campaign.py:1698，AKA:src/aka/workflow/light/orchestration/campaign.py:1186 | planner/design/optimize 的串行触发和 schema attempt 状态应由明确 orchestration 拥有；目前没有公开 before_episode/after_evaluation hook |
| Experiment / Evaluation / Execution | 共享准确 source 测量和恢复，typed facts；AKA:src/aka/experiment/runner.py:370 | 应复用，不应复制 GPU 后端；planner 只接所需公共 feedback，不接 credentials/evaluator 私有权威 |
| Report / CandidateGate / verifier | 分层 correctness、source seal、ABBA；AKA:src/aka/workflow/light/host/journal.py:1304 | 可在明确新契约下另加 schema fidelity evidence；当前 gate 没有这个输入/判断 |
| Export / observations | 终端候选源码与绑定结果、canonical 版本事实；AKA:src/aka/workflow/legacy/episode_output.py:103 | 可形成 standalone implement 的返回值和下一 planner 输入；有候选 artifact 并不自动等于已验证 schema 完成 |

资源层与服务层是两套接口：local plugin.json 贡献 JSON stdin/stdout 工具和资源；Core plugin 则 provide/inject 服务。不要把“已能安装一个 Skill/plugin”误写成“已能插入两个 Agent 串行阶段”。见 AKA:docs/plugin-framework.md:26、:31；AKA:docs/plugins.md:87。

## 6. 现有 schema 适配：已做什么，还缺什么

### 已核实实现

- create_snapshot 按 task 复制文档、记录每文件 SHA-256、生成 bundle hash，文件/目录设只读；拒绝 symlink、git metadata、非普通文件、空 root/dispatch。**它不解析 KernelSchema 语法或验证 dispatch 引用。** 证据：EXP:tools/python/aka_experiment/schema.py:54、:80、:116。
- render_context 不截断 root/dispatch，完整保留 UTF-8/空白；其余分支留在 /opt/solution-schema 按需读。policy 是独立可选文字，默认 neutral/空。证据：同文件 :130、:141。
- schema_enabled 与 condition 必须相符；只对 role="optimizer" 授权。control/reviewer/unknown 不投递 schema；调用者不能自带 context_files 抢占权威。证据：同文件 :168、:262、:299。
- runtime().run 与 open_session().invoke 两条调用路径均有包装；每次 invocation 重验 snapshot，记录 prepared/exited/failed；记录自称 input_preparation_only，没有冒充模型阅读或遵循证明。证据：同文件 :314、:336、:382、:453。
- 原始 transcript、Home、checkpoint、publication 与 ownership 的隔离是外部 adapter + 泛化 core patch 共同完成。补丁台账分 role/context、session boundary、storage policy、owner coordination 四组。证据：EXP:control/aka-changes.json:6、:237、:379、:520。
- V1 framework baseline 的 coding path 在 EXP 也显式传 optimizer；production/numerical/baseline reviewer 则用各自 role。故静态 schema 不只影响后续 episode，也可影响 V1。证据：EXP:assets/aka/src/aka/workflow/light/orchestration/campaign.py:1948、:756、:1685；EXP:assets/aka/src/aka/workflow/light/orchestration/numerical_policy.py:221。

### 已存在的验证材料的准确范围

test_aka_schema_contract.py 用假的 Claude CLI 在实际 Agent/bwrap 路径检查参数、上下文 hash、分支只读、private 隔离、resume/compact 等；test_aka_launcher.py 验证完整 composition 确实借用选中的 SchemaAgent，但直接调用 Agent fixture，未运行完整 GPU campaign。证据：EXP:tests/test_aka_schema_contract.py:25、:118、:159；EXP:tests/test_aka_launcher.py:58、:81、:85。

这些测试正文足以说明实现不是单纯文档设想；本轮没有重跑测试，台账中的历史通过次数也未独立复验。它们不能证明真实 provider 1M 上下文容量、真实压缩后语义保持、实现 fidelity 或 GPU 性能。EXP:docs/aka-integration.md:46、:73 也明确保留此界限。

### 尚未存在的能力

- planner/design/optimize 的可调用服务与结果合同；
- 从 KernelSchema docs/template/wiki 组装 planner context 的入口；
- 动态 schema revision、父版本、变更原因、反馈绑定与原子提交；
- coder 只能实现指定方案的任务语义、root-first/dispatch 进度、fidelity gate；
- 同一 schema 跨终端 coder episodes 的 partial artifact 延续与预算；
- 正确但不如 incumbent 快的“实现完成”结果返回政策；
- 多 provider 的 schema context 投递；
- schema revision 变化时与旧 Agent checkpoint / launch selection 协调的续跑策略。

特别是 control 条件依然运行 SchemaAgent / DevelopmentAgent、同一 isolation 与观察包装；它表示“不提供 schema 材料”，不是“完全回到未修改 AKA 的字节等价执行”。证据：EXP:tools/python/aka_experiment/launcher.py:69、:81；EXP:tools/python/aka_experiment/schema.py:299。未来可选开关的 off 合同需另外明确和验收。

## 7. “AKA 有没有 planner”的限定回答

**Light 默认没有独立的 schema planner。** 它把调查、下一步假设、实现和测量交给同一个 coding session，prompt 明说无须单独 draft/plan/profile 文档。goal_episodes 仅在持续 stall 后扩大同一 episode 的搜索指令，不改变 Journal/acceptance 合同，默认关闭。证据：AKA:src/aka/resources/prompts/research-episodes/episode.md:46、:55；AKA:src/aka/workflow/light/long_horizon/campaign.py:422、:510；AKA:src/aka/workflow/light/plugin.py:33。

**AKA 的其他路径确实存在“planning”，但不是需求中的 schema 设计层：**

- Legacy kernel-development bundle 安装 gen-plan，runtime_directives 生成 draft.md -> plan.md 的 Skill 调用，Fast trial 指令要求其已配置的 review flow。这是原有 kernel 优化计划资源/执行约定；不产出 KernelSchema。证据：AKA:src/aka/resources/bundles.py:31；AKA:src/aka/workflow/legacy/runtime_directives.py:44；AKA:src/aka/resources/prompts/kernel-development/fast_episode.md:47。
- Light 默认 resource bundle 是 measurement-session / research-records，而非 kernel-development / gen-plan。证据：AKA:src/aka/workflow/light/profiles/compositions/light.json:8。
- 可选 numerical planner/reviewer 读取 candidate 和 trusted public input contract，输出 action=complete/probe 与测试 suite；其职责是补充数值测试设计，不是优化 kernel schema。证据：AKA:src/aka/workflow/light/orchestration/numerical_policy.py:180、:202。

因此不能说“AKA 完全没有规划”，也不能把既有 gen-plan 或 numerical-planner 当成可直接配置的 design/optimize 实现。

## 8. 三仓职责与依赖候选（建议，未决定）

| 候选职责 | 来自本轮事实的依据 | 保留给后续阶段的选择 |
| --- | --- | --- |
| KernelSchema 拥有 schema 语言、验证、design/optimize/extract/implement 公共合同及独立 demo | AKA 当前不认识此 schema，仅有普通 Task/source/evidence；EXP 只消费固定文档 | standalone implement 后端做多薄、共享何种 planner/prompt/feedback 包；需结合另一份 KernelSchema 调研 |
| AKA 的可选接入层拥有 schema-aware orchestration 与 coder 对接 | Light 已拥有 episode、draft、工具、GPU 验证、恢复；schema 操作必须与这些状态协调 | 独立 workflow provider、少量通用 hook、或 adapter branch 中明确的扩展；本轮不承诺 fork |
| AKA_experiment 拥有 dataset/matrix、消融 policy、run 排程、外层预算、成本/结果比较、资格验收 | 当前 launcher/runner/records 已实际做这些事；硬编码 pilot/model/schema-off 是实验专属 | 哪些通用 role/context/storage patch 可移出实验仓，哪些保留为实验隔离 |
| 单向固定版本依赖 | EXP 现为 ordinary-file vendor；AKA 原生有 submodules/资源身份冻结，但没有 KernelSchema 依赖 | AKA -> KernelSchema submodule/package 与反向调用 AKA 的独立 implement 需一起设计，避免强制安装/构建循环 |

**事实约束：** Light package 依赖 legacy==0.2.0，legacy 又依赖 framework==0.2.0；Light 内部复用 Legacy startup、checkpoint/export 等。把“独立 implement 复用 AKA”解释成简单复制一个 Light 文件，会隐含拉入多包和执行环境。证据：AKA:packages/atrex-aka-light/pyproject.toml:10；AKA:packages/atrex-aka-legacy/pyproject.toml:10；AKA:src/aka/workflow/light/plugin.py:59。复用已有 Agent/Evaluation 契约和复用整个 campaign 是不同维护成本，需分别比较。

## 9. Submodule、许可与分发约束

- AKA 的 .gitmodules 声明 reference-projects 和 gpu-wiki/3rdparty；固定 tree 有 25 个 gitlink。既有清单中没有 KernelSchema。部分 URL 用 git@github.com:t-head/...，本轮没有验证它们的公开可访问性或初始化完整性。证据：AKA:.gitmodules:1、:65、:71；EXP:control/aka-manifest.json 的 excluded_gitlinks。
- EXP 投影明确排除全部 gitlinks 和 .gitmodules；保留其 commit/URL/materialized=false 元数据。这使当前实验无需自动携带完整参考工程，但未来 submodule 安装流程不能照抄“完整 clone”假设。证据：EXP:tools/lib/aka_vendor.py:93、:98、:104；EXP:control/aka-source.json:10。
- AKA 声明 Apache-2.0；重新分发须携带许可证、修改声明、相关 attribution/NOTICE。第三方 gen-plan 与 reference-kernels 保留各自许可，不能只看根 LICENSE 概括所有资源。证据：AKA:README.md:121；AKA:LICENSE:90；AKA:NOTICE:7、:17；AKA:src/aka/resources/skills/gen-plan/LICENSE:12。
- checkout 中 plugins/gpu-wiki、参考工程等与可安装 package 不是同一个分发集合。资源目录被 fingerprint/lock 引用，不能在已有 campaign 下随便变更；引入 KernelSchema 文档/wiki 时必须明确其是否 packaged、外部路径还是 submodule。证据：AKA:docs/design.md:69；AKA:docs/plugins.md:47；AKA:src/aka/bootstrap/continuation.py:68。
- 当前 EXP 根目录未见独立顶层 LICENSE；assets/aka 内保留上游许可。EXP 自身 adapter、bench/solution 数据和未来公开分发的授权边界，本轮没有完成许可证盘点，不能据此推定全部资产都能直接公开。此项是文件清单观察及未验证范围，不是阻塞规划的许可审批要求。

## 10. 下一阶段最需验证的风险和未知

以下是待验证问题，不是已确认缺陷或已批准方案：

1. **最初实现与后续优化的接缝。** schema design 在 V1 前还是另一个明确 phase；V1 失败时如何留存 partial source/预算，不能只在 LongHorizon 普通 episode 前插一段 prompt。
2. **状态粒度。** 需要分别建 schema revision、implementation attempt、native episode/session、measurement 身份；schema unchanged 的多 episode 与 schema changed 的新 attempt 应能恢复并正确计费。
3. **实现完成不依赖胜过 incumbent。** 正确但更慢、未完成、只实现 root 未覆盖 dispatch、不可实现方案各如何报告/返回，以及哪些才触发 optimize；原生三状态+ABBA 不能直接回答。
4. **持久产物与下一轮 seed。** public-candidate.json 可取到终端 working source，但新 episode 默认重置为 canonical source。应验证可复用的候选读写/导入接口以及中途未测量修改，避免失去已完成工作或把它误当成功晋升。
5. **schema fidelity。** 现有只读 snapshot 和 prompt 注入只证明交付，不证明实现遵守 root/dataflow/dispatch。checker 的职责、机器可验证部分和人工/模型判断部分需明确。
6. **角色与 provider。** 复用 EXP adapter 需携带尚未上游的通用 patch；扩展 planner role、Codex/Pi/DSH 等后端与 baseline/reviewer 权限时，不能依赖环境字符串猜角色。
7. **switch 与原生策略的相容性。** 明确 off 是无 planner/no extra model cost，还是要求上下文、工具、状态、预算和结果均与固定上游一致；当前消融 control 不是完整原生等价基线。开启 schema 后还需处理已有 goal widening、mandatory Triton→Gluon conversion 路径与“实现指定 schema”的冲突，不能让 stall 策略静默改变被指定的实现方案。证据：AKA:src/aka/workflow/light/long_horizon/main_adapter.py:102；AKA:src/aka/workflow/light/long_horizon/campaign.py:1734。
8. **冻结配置与动态 schema 的冲突。** 不能编辑已绑定 manifest 或重写 recorded selection 来换 schema。新的 revision 应作为哪一级合法状态变化，需针对 capture/restore/checkpoint 设计并验证。
9. **故障分类和预算。** provider throttling、GPU unknown submission、compile/correctness failure、budget exhaustion 与设计问题分离；planner 成本必须显式进入全局预算，不能借新 phase 获得未计费余额。
10. **记录损坏的边界。** CampaignStore.load_state / load_active 对缺失或解析错误返回默认值，而 durable active-time budget 更严格。需要测试二者与 Task/continuation/Journal 的组合，不能只根据“resumable”文档承诺任意损坏可恢复。证据：AKA:src/aka/workflow/light/long_horizon/store.py:76、:99；AKA:src/aka/workflow/budgets.py:51。
11. **现有实验工作树尚未可冻结为新基线。** 缓存导致 vendor verify 失败，另有两项 ledger 差异，pilot 仍 schema-off / formal_launch_enabled=false。本轮不能拿这份工作树直接作为最终可发布适配接口或端到端性能证据。
12. **维护路线与资产发布。** 普通文件 vendoring 已有审计工具；fork/submodule/package 方案需比较 upstream 同步、复现 SHA、工作树 patch 承载和第三方资源集合，当前没有做远端决定。

## 11. 本轮核验范围与交接

完成：本地 Git/来源文件核对，关键原生与 EXP 调用路径正文审查，schema adapter 与通用 core 补丁逐文件比较，episode/report/ABBA/预算/恢复代码审查，测试正文与许可/分发声明审阅，以及只读 vendor 审计。

未执行：任何模型/GPU/付费调用、benchmark、测试套件、完整 example/campaign、网络搜索、远端 Git 操作或许可法律结论。没有将已有准备记录当作本轮现场验收。

下一轮架构研究可直接从本报告第 4、5、6 节建立契约；最关键的约束是：**静态 bundle 投递已存在，但 dynamic schema lifecycle 尚未存在；原生 episode 有可靠的局部续跑能力，却不等于固定 schema 的多 episode 实现状态机。**

