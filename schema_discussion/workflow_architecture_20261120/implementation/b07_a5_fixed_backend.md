# B07 / A5.interface + A5.cpu — AKA 固定 backend 实现

日期：2026-11-20 UTC。本文是本批唯一写入原 KS main 的报告。**A5.interface/CPU 固定 backend 已实现并通过适用验证**；真实模型/GPU/profile 资格仍按下述边界保留。

## 固定输入与范围

- AKA：`/home/shaohanyong.shy/atrex-kernel-agent`，分支 `feat/schema-workflows-v1`，HEAD 保持 A4 的 `106400d5e0d8ed26e33d7696bde578daa5242a9f`。本批不提交、不推送、不切 AKA 分支、不改 Git 配置。
- 初始固定 KS 为 `189a541`。按主代理后续明确授权先配 `067b37d`，最终子模块 detached 于已公开 `b1049e19403238125d4210dd275a8f3e2874ba47`。最后一次从公开 origin fetch 成功；此前 067 网络等待时仅 fetch 原 KS 已固定 Git 对象，没有从活跃工作树安装。origin 仍为 `https://github.com/bcacdwk/KernelSchema.git`，无 alternates。
- `tools/freeze_schema_release.py` 已刷新 293 个选定文件。新进程 `python -I` 核实实际 KS import/resource root 均为 AKA 子模块，manifest 为 `sha256:d108ae06e45fb405eb9ba6c900404550b454958d53b628ad8a90e01df16e5da1`。
- 仅在 AKA 实现固定 backend、必要原生/A2–A4 接点、CPU 测试与说明；未访问/修改/运行 EXP，未派生代理，未新增模型/GPU/Agate 作业，未实现 A6 planner。
- 复用 Python 3.13 隔离环境 `/tmp/schema-b05-a3-venv`。以原 `tools/build_distributions.py:owned` 的 ownership 分包，本地 `build --no-isolation` 构建 Framework/Light 的 sdist→wheel，并用 `pip --no-deps` 安装这两份当前源码包，使 `python -I` 包含真实新接点。未下载解释器或依赖栈。Schema 继续使用原 A3 的显式 editable 包；不是 A7 完整独立发行签收。

## 已实现入口与配置

默认工厂 **`aka.workflow.schema.backend:create_workflow(selection, NativeModules)`** 已存在，返回保留的 Light Workflow/ManagedCampaign 合同。实际原 CLI 经 Bootstrap/WorkflowStartup 走到该工厂，不再是 missing factory 或接口计划。

```sh
aka workflow --repo-root "$AKA" --problem /absolute/bundle \
  --workspace-base /absolute/native-work \
  --platform H20 --framework triton --framework-baseline never \
  --convert-after 0 --token-budget 0 --schema \
  --schema-config /absolute/selection.json
```

selection 仍为 A3 的 `aka.schema.selection.v1`，字段是原 request 路径、private run_root、固定 factory、实际 `backend.py` 的 sha256 和 `parameters={}`。执行参数放在 immutable backend profile 的 `parameters.aka`，不静默忽略 selection 参数。真实命令 CPU smoke 使用完整原 Bootstrap；缺实际资格时退出 0 并发布规范 zero-fee blocked result，领域状态以 result.json 为准。

KS 方向由 core 的显式 `implement + backend.name="aka"` 映射惰性加载 **`aka.workflow.schema.backend:workflow_backend()`**。其 `capabilities/identity/validate_resume/execute` 实际实现直接消费原 RunContext，并调用同一 FixedController；不再调用一次 `api.implement`，不创建第二个顶层循环或可增额父账本。

```sh
kernelschema implement --request /absolute/bundle/request.json --out /absolute/private/run
kernelschema implement --continue /absolute/private/run
```

`profile.parameters.aka.native` 可固定 `workspace/name/settings/runtime`，使用原 WorkflowSettings 和 CampaignRuntimeSettings；相对 workspace 对原 KS run 根解析。默认私有 `<run>/native`、Triton、never、无转换、20 个 native versions；实际请求应显式填写平台和原生限制。小数 native 参数用合同十进制字符串输入并在边界转原 DTO 数值。

实现文件：`schema/backend.py`（双方向入口/固定来源）、`controller.py`（固定事务和原 engine）、`state.py`（定位/收据持久状态，无余额实现）、`calls.py`（原 RoleSession 调用/恢复）、`projection.py`（公共结果）、`comparison.py`（原 ABBA callback）、`agate.py`（具体真实服务资格消费）。完整使用说明见 AKA `docs/schema-fixed-backend.md`。

## 固定行为、C/G 与费用

固定 R 可跨多个 coder e。普通失败、未完成、fidelity fail/unknown 继续同 R；零 planner。NativeRoleResult 的完整 NativeInvocation 在原 NativeSession 返回前先持久化；随后 A4 以原 authority 安装准确 `kernel.py`。超限/无效输出保留费用和原稳定 C，下一次调用是独立 owner 操作，不伪装原 native context resume。

每个操作在实际费用前具有 run、原 owner/reservation、role、purpose、准确输入 C、e、key、native output/scope locator。私有 state 保存事务与资格，不复制 KS 余额/时间算法。

| 实际接点 | 原所有者与本批接入 |
| --- | --- |
| V0 | 原 ManagedCampaign.setup_baseline → measure_for_acceptance → 原 Runtime/CommandEvaluator → A2 SchemaExecution；baseline 是 Task.baseline_source，不用 starter/C0 测 V0 |
| 并行 V0 prefetch | 原 `_run_v0_evaluator_with_correctness_prefetch` 的线程结构保留；原 KS `reserve_group` 原子预约 evaluator/reviewer，恢复完整未启动组才用原 `claim_group`；每个 reviewer 走独立 RoleSession/receipt |
| 每次 coder/reviewer/repair | 原 Agent/NativeSession + A3 NativeCallAuthority/RoleSession；每次真实 host invoke 计一次，内部 events 不与 terminal total 重算 |
| 完整 Task | A2 Evaluator/CommandEvaluator 双槽实际注入；冻结准确 C、完整 Task/scope，真实 KS prepare/submit/query/settle；原 CPU 仍 UNAVAILABLE/all_pass=false |
| fidelity | 直接调用固定 KS 的 obligations/review_input/report 与 Rules 冻结方法；独立只读 reviewer，不复制 AST/parser/评测方法、不用 coder 自评代签 |
| production/source/dependency | 原 `_production_kernel_violations` 结构检查保留，独立原生 reviewer 使用显式 C 和实际 solution 元数据；原 `_validate_production_review` 解析结果，原 native digest 绑定缓存，不能同 C 换 metadata 复用旧 verdict |
| ABBA | 原 RecordedABBAValidator.verify 的显式 callback；单逻辑 comparison receipt，原顺序/GeoMean policy/晋升路径保留 |
| finish | 在实际固定阶段结束调用原 `_finish`；另外 stage/check 精确 C。C 的生产检查失败清除 qualified；native G export 失败单列，不把 G 当成 C |

原 KS OwnerAccount 是唯一 account、活动时钟和累计吸收实现。Model/评测/收尾保持最低保留额度；费用未知不归零。原终态只结算自己的操作，不证明同组或远端 sibling 清理完成。实际 launch、原 Agent during-work 检查、stdin shim exec 前、A2 提交及 Agate POST 前均使用原 authorization/check_active 和原绝对截止；不重授时、不换 key/channel 绕 unknown。任何未连接固定 role 的旧 `_run_session` helper 明确拒绝；V1 auto/always 在收费前拒绝，没有自由探索回落。

本首发 Schema profile 将 native prefetch 标签作为 advisory reviewer slot，实际都调用请求明确选定的受限 Claude reviewer；标签保留在 operation purpose，不冒称调用了 Codex/Qoder。其他 reviewer runtime 须另给真实 profile/实现。

公共 `latest/qualified/assessments/performance_goal` 与 native G/HEAD/accepted/promotion 独立。完整五项 Task/current-R 检查可 completed，软目标 unmet 或原生 ABBA 比 G 慢仍可完成。verify_only 要求显式非空 seed，零 coder、零改码 episode、零新晋升，必要 evaluator/reviewer 仍计费。

## 三 audit 与恢复

| 边界 | 实际实现与 CPU 证据 |
| --- | --- |
| 普通 seal 后、ABBA 前 | 调用原 CandidateWorkspace.seal_candidate，durable qualification 绑定 run/owner/Q/R/rules/scope、精确 C/native identity、base/candidate commit、assessment 与原操作收据，随后进入原 ABBA |
| 未提交恢复 | 原 e/worktree/draft/operation 回收；重建 attempt 保留原资格绑定。candidate_sealed、qualification_durable、native_handoff_ready 等故障点无新模型/重复评测 |
| 已提交恢复 | 从 commit-bound promotion audit 读取原 schema_qualification 和含 ABBA 的 schema_operations，核对当前 committed source 并吸收原用量，不新审 fidelity、不重复晋升 |
| 已提交 audit 缺资格 | 保留 native accepted/commit 历史，Schema unknown/blocked，停止自动推进；测试用原 native commit digest 绑定一个确实缺资格的 audit |
| 首 result 前丢失 | native_invocation_retained、episode_settled 和 V0 已结算但 native return 丢失均恢复原操作；完成的 e 不误开 e+1 |
| 原调用无收据 | 只查询原 operation/native locator，blocked/settle_only，零替代调用；不把原生进程结束或空返回认作所有统计已齐 |

Late statistics 通过原 OwnerAccount 累计序列吸收；首次 terminal 事实保持不可变，后续收据另留。已终止、未限额 token=null 可以不是在途；公共 usage 保留 null，补齐后不重复计算调用。外部 Python 子进程已实际从 `KS_BUDGET_AUTHORITY` 授权/吸收同一原预约并重放零新增；没有复制 account descriptor 自建余额。

本批额外闭合了三个真实组合缺口：A3 `agent_plugin/problem_plugin` 的 name 必须符合原 Boot basename；RoleDelivery.roles 的 tuple 在持久 grant 中转合同 list；原启动 native lease/checkpoint 元数据不应被 A4 当成已做 V0。A4 的例外只接受原 writer 锁持有、无原 intents、且只有 public-identity/writer-state/checkpoint-context 的 bootstrap 根；Git/active/其他工作仍拒绝重新造 seed。

最后又核对了 KS 已明确保留的初始化间隙：AKA selection 现在按原 run 是否已存在记录 continuing，不因 backend state 尚不存在就以 create=True 补账户。旧 run 无原 budget binding 明确拒绝；新增反例证明不会生成 owner-account.json/budget binding。KS Backend.execute 也显式传递原 continuing 标志。

## 实际 comparison 能力与真实服务边界

`ComparisonCallback` 是具体的受信接口：service 提供 admission、locator、prepare、validate、submit(authorize)、settle、measurement；原 spec 固定双方 Source、Task/scope、原 schedule、timeout、qualification digest 和原 reservation。每个 row 必须来自同一 allocation，保留原完整 Task report 和全 coverage 测量。原 kernel-only candidate 约束、score/promotion 决策不变。独立公共 timing/不同 allocation 不能冒充 ABBA。

固定树中的 `GatewayABBAValidator` 只有配置构造，**没有独立 verify 实现**；实际运行入口为 RecordedABBAValidator。Schema 其他 baseline/任意命令请求仍在 A2 routing 拒绝。CPU 用明确 FakeCudaExecutor/FakeComparison 工程服务走原 Task/source/evaluator 接口、同一比较 spec 和原 native promotion；它们没有执行 GPU，也没有形成真实资格。

具体 `aka.workflow.schema.agate:workflow_services` 已接 K6 的 AgateChannel、原六次 qualification capture 校验、真实环境/target、原 AgateDevExecutor 和 POST 前剩余截止。另要求 **AKA 自身** `aka.native.qualification.v1` coder/reviewer 原 NativeInvocation+locator capture，固定本次 native profile/executable/source hashes；这些 Ref 明确放 host-only Materials `roles=[]`，不把私有 capture 给模型。K3 的公共 profile 不能代签。

本批没有真实 AKA native/model/GPU capture，也未取得真实 same-allocation Agate ABBA 资格。内置 Agate services 因此不选 comparison；接好真实资格后可以执行完整 Task/fidelity 并交付 qualified、未晋升的 C，不能宣称已验证真实晋升。不存在 fake Atrex-Bench 目录/harness、普通 allclose 替换或 CPU→CUDA PASS 修改。Agate 认证/ACL 未验证不等于服务不可用；复用现成 `KS_AGATE_CONFIG`，不广搜凭证或另搭环境。

## CPU 验证与工件

- 当前源码 A1/A3/A4 + default-off：**73 passed / 222.56s**，在 067 配对阶段；原 default-off 禁止 optional import/KS checkout I/O/network 的探针通过。
- A2 原完整 CPU Task/坏候选、Source、实际 Evaluator/Executor/CommandEvaluator：**16 passed / 148.25s**，日志 `/tmp/schema-b07-a5-a2-regression.log`。
- b104 新配对的 role/host Tools/原 owner/实际 composition 定点：**3 passed / 12.37s**，日志 `/tmp/schema-b07-a5-b104-role-regression.log`。
- A5 主集成：**23 passed / 278.89s**，保留根 `/tmp/schema-b07-a5-final-stable`，日志 `/tmp/schema-b07-a5-final-stable-tests.log`。
- 随后补初始化间隙的账户保护，仅重跑受影响的真实 CLI、KS 外部入口/continue、V0 恢复及新反例：**4 passed / 50.52s**，保留根 `/tmp/schema-b07-a5-entry-final`，日志 `/tmp/schema-b07-a5-entry-final-tests.log`。当前测试集为 24 个 case，全部已通过适用检查。
- 工件与最终源码哈希索引：`/tmp/schema-b07-a5-evidence.json`。测试实际使用原 Store/GitCandidateWorkspace/EpisodeWorkspace/Agent/NativeSession/RoleSession/Source/Evaluator 接口；模型/GPU服务是明确 synthetic transport，不形成真实资格。
- 较早一轮 22 passed/1 failed 的原证据保留于 `/tmp/schema-b07-a5-final-tests.log` 和 `/tmp/schema-b07-a5-final`：唯一失败是测试运行中收紧 controller 的最低收尾预留，被外部 continue 的 implementation hash 守卫正确拒绝。核对 anchor 只差 controller.py；未改历史或放宽守卫，冻结代码后 23 项重跑全通过。
- 本地当前 Framework wheel sha256 `bcdf59e57e376425f3363551caf99b3fe996d07221aa9f84338bd9a5cbe381f1`；Light wheel sha256 `9a192996b03221a7a6f7016b0f28c2fa4c5c7bc2a37c1e730bfd9d0306027d53`，都在 `/tmp/schema-b07-a5-dist`，并有各自 build log。
- 新测试精确 `git add -N -f tests/test_schema_backend.py` 可见；没有暂存内容。其余新增实现和说明保持未跟踪，交主代理统一审查/提交。
- 最终 `git diff --check` 通过；AKA HEAD 与分支未变，KS 子模块工作树干净，未提交/推送。

## A6 所需接口与未验证项

A6 直接消费 `Backend.execute(original_RunContext)` / `FixedController.execute()` 和公共 result；原事务完成/结算、旧 scope 关闭后构造新不可变 request。新 R 或 mode 不可热改旧 Q；用原 `use_authority`/`KS_BUDGET_AUTHORITY` 连接同父余额。阶段已 qualified 或有独立设计阻碍才由 A6 判断 planner；普通编译、数值、fidelity unknown、未晋升都不是固定 backend 的 planner 触发器。

Controller 沿用 A1/A4 的原 native writer lease；AKA CLI 由 WorkflowStartup 持锁，KS 由 Backend.execute 持锁。A6 已持原锁时用已有 controller，不要嵌套获取第二个 lease。

尚未验证：实际 AKA no-tools CLI profile/capture、实际选定 GPU/Task 全套与 fidelity、真实 Agate同allocation ABBA、真实 production交付、所宣告的外部原生进程终态定位资格、A6动态迭代、A7独立源码发行。失去 owner binding 的初始化间隙仍按原 KS 明确失败，A5 没有静默补建账户。有限 token hard enforcement、V1、source_tree、多后端自动迁移不属于此首发 profile。
