# B09 / A6 有限动态工作流工程交付

**A6.cpu 范围内代码与测试已交付，未提交；A6.real 未签收。** 最终 AKA 源码
配对 KS `8e846d7dd6321dc226436ac8fc80fbf569b0965f`。完整新配对动态组
**26 passed**，没有 skip，覆盖真实 API/原 native 对象与 synthetic transports。

2026-11-20。AKA 工作树 `/home/shaohanyong.shy/atrex-kernel-agent`，分支
`feat/schema-workflows-v1`，开工 HEAD
`879cfeaf4648726b31321b751d25460ff3aa5d66`；开工 KS gitlink
`b1049e19403238125d4210dd275a8f3e2874ba47`。本代理不提交、推送或更改全局配置，
不派生代理，不读取/修改 EXP 源码、配置或捕获，不运行 EXP，不新增解释器或大栈。
只编辑 AKA 所属实现/测试/使用说明，以及这份指定主仓审计记录；不编辑活跃 KS
源码，不手工修改子模块内源码。KS 配对升级只在主代理给出新的固定 SHA 后进行。

本批是真实 API/原生对象下的 **CPU synthetic 工程验证**。新增真实模型调用
**0**、GPU/Agate 作业 **0**、新增真实费用 **0**。测试里的 Claude 子进程是
KS 既有 synthetic fake CLI；AKA Agent 与原生 Campaign/Store/Git 对象是真实
对象，模型返回和 CUDA 观测是显式 fake transport。旧 AKA coder probe 的一次
120 秒超时、tokens/cost unknown 仍未资格；K3 运输、历史反馈及本批 synthetic
轨迹均不签 A6.real/A5.real。

## 交付实现与入口

- 原默认 factory `aka.workflow.schema.backend:create_workflow` 保留。
  `parameters={}` 仍为零 planner fixed backend；仅 `parameters.dynamic`
  选择有限动态实现。原 `--schema` CLI 路径实际执行它；默认/off 不导入它。
- 新 `dynamic.py` 只拥有阶段、原请求/result 定位、publication ID 和 handoff
状态；实际写作用 KS `writing.request`、`api.design/optimize`，实际实现用
  原 `FixedWorkflow`/`FixedController`。没有复制 KS 写作、Schema parser、
  evaluator、预算账本或固定 coder loop。
- 已有 Schema/extract 可用原合法 implement request，明确 seed 或 verify_only。
  从零时 selection.request 本身是合法 design_request；implementation 描述
  明确给 profile/rules/materials/budget/mode/seed，**不含预造 Schema**。
  KS 完成写作后才形成带真实输出 R/Scope 的合法 implement Q。
- optimize 的固定输入模板可为合法 design 或 optimize request：只冻结其
  Task/目标/Materials/Rules/profile/budget，当前 R/Scope 和已结算原证据从
  前一固定阶段取得。因此从零准备有限链也不需要虚构未来 R。
- 仅支持 optional design → fixed → 至多一次 optimize → 至多一次 fixed
  followup。extra 配置和无限调度不支持。optimize keep/none 停止；有行动才
  从 `action_inputs`/`action_evidence` 建新 Q，verify 精确非空 Source 且零 coder，
  Task/default_start 明确 `seed=null`，不暗取旧 C。

公开配置/真实命令见 AKA `docs/schema-dynamic-workflow.md`；已有 optional、
fixed、source-release 说明均链接该入口。实际 CLI 测试从非仓库 cwd 以
`python -I -m aka workflow|optimize ... --schema --schema-config ...` 启动，
从零 design fake CLI 写作后继续实际 fixed admission；无真实资格时固定阶段
发布原规范 blocked/zero-native-fee 结果，不把默认 LocalCPU 当 GPU PASS。

## 原生跨阶段选择及恢复理由

采用**同一原生工作区的显式交接**，没有独立 native campaign 仓库或累计上限
重置。原 Startup 在外层持有唯一 writer lease；动态调用 retained controller，
不调用会再取 lease 的 `Backend.execute`/`api.implement`。原同步 KS→AKA
Backend 仍直接使用它收到的 RunContext。

`handoff.close_stage` 要求旧 fixed 为真实 done/finished，原 operation 均已终止
或释放，无 owner fence、无 active episode；保存原 run/Q/R、准确 C/qualified、
HEAD、完整 native checkpoint 摘要、source-state 摘要、native 设置/上限及同一
authority 路径。新 source lifecycle 在原累计 e 之后开始，继承已结算 V0 证据；
禁止重复 V0/预取 reviewer。旧生命周期、attempt/commit audit、G 和全部 native
history 保留。新 draft 从明确 seed 完整安装，不覆盖任何已准入旧 draft。

实际整链测试发现 Light 原 `_load_problem` 会拒绝已有仓库却缺新的 snapshot。
这条保护保留。只有显式新 Q、旧 handoff 可核关闭、该新 Q 无任何 intent/source
state 时，调用原 SchemaProblemProvider 捕获新的完整 Task/current Scope snapshot，
然后进入原 `_campaign`。已经开工的阶段丢 snapshot 仍拒绝补造；没有静默复用旧
scope，也没有修改旧 Q/R/mode/profile/native limits。

动态 state 原子持久化并带完整摘要，固定 selection/初始 request 摘要、implementation
和 parent descriptor。新 request 的 publication ID 在发布前持久；ready、账户
binding ready 但未建 e、新 scope snapshot、source ready/admitted、原结果未返回、
旧 scope closed 与 writing published 各有可中断边界。先从 current 原 child
run 恢复 finish/model/job/receipt，完成结算才向下一阶段移动。已有 run 缺 budget
binding 不自动 create，missing result.json 不换 run。恢复实例不能重复 planner、
重发新 R 或追加一次 revision。

主代理进一步明确：OwnerAccount.register/for_context 的新 Q 绑定不是付费准入。
断点已改用 `new_request_bound`，不保留误导的 admitted 别名；没有为预检制造假
operation、跨阶段 group 或续额度。新 R/请求绑定可先持久，实际 reserve/authorize
仍由原 FixedController 管理。pause/余额拒绝保留“新 R 尚未执行”及旧 qualified
的准确归属；不能用新请求创建成功冒称实际工作已准入。

新 R 写作完成但尚未进入固定阶段时，summary 的状态为 partial，current_qualified
为空；旧 qualified 在带原 Q/R 的 history 中保留。native_result.schema 标明
该结果所属请求，native incumbent G 单独列出，不能将其当作当前 R 的资格。
普通 failure/unknown、预算结束、无 promotion 只停原固定阶段；当前没有独立
可信 design-blocker 协议，所以只启用 completed+明确目标的 D3 路径。

## KS 依赖与原账户

所有阶段要求同一个显式已存在的 OwnerAccount/parent Ref；不从公开 descriptor
复制新余额。所有原生、评测和 KS 写作费用都走原 account 的 reserve、reserve_group、
check_active、receipt absorption 与单时钟。native 原三条 qualification audit、
late usage、unknown 及 source 规则不变。

发现 K5 原 writing._operate 只保留当前 reviewer floor，不能由外壳另预留一个
operation 来保护后续 fixed：那会违反 writing state 的 operation 所有权，并
触发串行 fence。已向主代理提交精确缺口，KS 唯一 writer 接受新增固定 profile
参数 `writing_finish_floor`：严格原 cost 完整向量，planner floor=该向量+原
reviewer 最低量，reviewer 也保留它，默认零保持独立写作。无 authority 迁移。

AKA 在开始动态收费前检查该 floor 至少覆盖所支持的下一 fixed stage。普通
leaderboard 默认最少 coder 1、reviewer 1、evaluator jobs 2、active_ms 4000；
production 和 V0 预取按实际 native 配置增加。该数值是 policy minimum，不是
另一个余额/计时实现。写作请求自身 limits 与 parent/native 最紧边界也必须能
覆盖这份 floor，不能只增加 parent。新的固定 KS 配对已到位，实际原 reserve 的
floor/clock 断言已在最终动态组通过；配对及最终整组结果见下节。

## 已发生的测试轮

解释器为既有 `/tmp/schema-b05-a3-venv/bin/python`（Python 3.13），实际 import
AKA Schema editable 源码、原 Framework/Legacy/Light 安装及 AKA 固定 KS 子模块。
没有真实模型/GPU依赖或联网安装。

| 轮次 | 事实 |
| --- | --- |
| 首轮接口调试 | 先发现测试 Rules.files 未排序；后发现启动外壳缺 command-capable evaluator；均在零业务测试费用处失败并修复。 |
| 首条完整链 | schema → native fixed → keep/none 通过；随后同 R 行动暴露 Light snapshot guard，上述显式交接修复后继续。 |
| 动作/初始设计轮 | **4 passed / 1 deselected，161.38s**：keep→implement、keep→verify、revise→implement、design→fixed。该轮设计入口随后改为真正无预存 Schema。 |
| 无 Schema 起点及 CLI 轮 | **7 passed，295.16s** 后主动中断剩余恢复测试，以先补齐发现的发布状态、初始 Q 摘要和后续 writing floor 校验。通过项为 5 个整链测试和 2 个实际 CLI 测试；不是全套通过。 |
| 原 Startup 成功链 | **1 passed / 5 deselected，50.44s**；走原 Startup.prepare/run 和真实 native 两阶段，确认整个 lease/prepare 路径也执行 followup。原 ManagedCampaign.prepare 只做目录/recovery-ready，不另启动付费工作。 |
| factory 导入反例 | **1 failed / 2 deselected，7.91s**；发现本轮新增 selection 详细校验提前导入 backend。已移回通过源哈希验证的 factory 之后，保持 A3 的 pre-import 边界。 |
| 恢复主轮 | **11 passed / 1 failed，487.36s**；已过四个新 R/请求中断、原 result 丢失、finishing、bootstrap binding 丢失、同 R 多 e、ordinary fail/unknown 与 pause。唯一失败是测试错误地要求 unknown 的诊断理由字符串也不更新；原 op/reservation/receipt 事实未变。 |
| 修复及剩余定点轮 | **4 passed / 19 deselected，81.45s**；fresh-child factory 导入、unknown 原预约/收据不重复、planner 预算耗尽零启动、额外 action_inputs 明确拒绝。 |
| 实际付费 finish 恢复 | **1 passed / 13 deselected，40.58s**；production reviewer 原 NativeInvocation 已落盘但未吸收时中断，原 finishing 先恢复并将原 receipt 只吸收一次，随后才调用唯一 optimize。 |

完整动态组主命令：

```sh
cd /home/shaohanyong.shy/atrex-kernel-agent
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /tmp/schema-b05-a3-venv/bin/python -B -m pytest \
  tests/test_schema_dynamic.py tests/test_schema_dynamic_cli.py \
  tests/test_schema_dynamic_recovery.py -q -x --tb=short
```

已完成针对新 R published/ready/bound-before-e、原 result 丢失、finishing 先恢复、
bootstrap binding 丢失、同 R 多 e、ordinary fail/unknown、暂停、未知 native owner
和 planner 耗尽的实际恢复测试。最终计数、配对 SHA、固定/off 回归及源码检查见下。

## 最终固定配对及增量验证

主代理给出已公开推送的 KS 固定提交
`8e846d7dd6321dc226436ac8fc80fbf569b0965f` 后，AKA 子模块切到该 detached SHA，
原公开 origin URL 未改。对 GitHub 的第一次 fetch 约一分钟无输出后主动终止，
随后从 KS 主仓的 **Git 对象库** fetch 同一个指定 SHA；未读取另一代理活跃工作树
的未提交字节。运行原 freeze 工具后，release.json 覆盖 **297** 个选定文件，
manifest SHA 为 `94d2457ea5128ecd72d78448abaf489a3ae2c098d370b7748c77101c5bd57af7`。

从 `/tmp` 执行现有 venv 的 `python -I`，实际 `kernelschema.__file__` 与
`checkout_root()` 均指向 AKA 子模块，`verify_release()` 返回上述固定 SHA 和
manifest。没有重新安装大栈、改解释器或使用 KS 活跃工作树的 editable 路径。
新 KS 自身包含 FA4/tree 能力并不扩大 AKA 本批的 row-softmax/single_file 范围。

新 KS 增加 `text_alias_of` 与公开 `runtime.material_text`。AKA 的 RoleContext
仍保留并重建原紧凑 payload，RoleReader 单 ID read 仅通过该 helper 解析该
payload 内的合法 alias，返回已获授正文，不查询其它 payload 或磁盘资料，不
恢复被移出模型输入的 host implementation。新增实际角色输入/reader 测试
核对正文原字节及越权 ID 拒绝。

| 最终配对相关轮次 | 实际结果 |
| --- | --- |
| 升级前 fixed / source lifecycle / off 完整回归 | **62 passed，485.47s**；运行在 b104 配对，准确保留这个范围。 |
| 新配对 A3 / Task / probe / off 首轮 | **31 passed / 1 failed，46.77s**；所有已到达的功能检查通过，失败仅为 KS 错误文字从 `extra files` 变成 `extra paths`。实际多余源码仍拒绝。 |
| 上述受影响组修复后 | **23 passed，165.06s**；只将 A2 测试 regex 收窄到稳定语义前缀 `Source payload contains extra`，没有放宽源码边界或修改 KS。包含完整 Task adapter、native probe synthetic、off。 |
| 新配对 fixed 原 audit / V0 group / 外部原账户 / late usage 增量 | **6 passed / 20 deselected，85.18s**；原 ABBA promotion/commit recovery、V0/prefetch 原收据恢复、KS 同步入口及 late unknown 原账均保持。 |
| 完整新配对动态组 | **26 passed，887.44s，0 skipped**；包含真正无 Schema 起点、已有 Schema、同 R 多 e、keep/verify/revise 实际 followup、所有上述中断、实际 writing floor/单时钟、paid production finish、两条 CLI、alias 和 pre-import 边界。 |

真实命令：

```sh
git -C optional/KernelSchema fetch /home/shaohanyong.shy/KernelSchema 8e846d7dd6321dc226436ac8fc80fbf569b0965f
git -C optional/KernelSchema checkout --detach 8e846d7dd6321dc226436ac8fc80fbf569b0965f
/tmp/schema-b05-a3-venv/bin/python -B tools/freeze_schema_release.py --commit 8e846d7dd6321dc226436ac8fc80fbf569b0965f
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /tmp/schema-b05-a3-venv/bin/python -B -m pytest tests/test_schema_backend.py tests/test_schema_seed_lifecycle.py tests/test_schema_off.py -q -x --tb=short
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /tmp/schema-b05-a3-venv/bin/python -B -m pytest tests/test_schema_optional_module.py tests/test_schema_task_adapter.py tests/test_schema_native_probe.py tests/test_schema_off.py -q -x --tb=short
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /tmp/schema-b05-a3-venv/bin/python -B -m pytest tests/test_schema_task_adapter.py tests/test_schema_native_probe.py tests/test_schema_off.py -q -x --tb=short
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /tmp/schema-b05-a3-venv/bin/python -B -m pytest tests/test_schema_backend.py -q -x --tb=short -k 'promotion or prefetch or late_unknown or KS_external'
```

实际写作 owner operations 的 planner floor 为 coder 1 / reviewer 2 /
evaluator jobs 2 / active_ms 5000，reviewer floor 为 coder 1 / reviewer 1 /
evaluator jobs 2 / active_ms 4000；各原 receipt 只吸收一次。原账户的 active_ns
严格等于其各请求贡献之和，完成后 active_run=null；重复运行不增加 operation
或 receipt。它验证原 OwnerAccount，未实现或复制另一份计时/余额逻辑。

最终所有 **16 个改动文本文件** 的 Python AST（适用者）、逐行空白、本地
Markdown 链接检查及 `git diff --check` 通过；另外的第 17 项是精确 gitlink。
子模块 `git status --short` 为空。完整文件 SHA-256、基础 AKA HEAD/KS HEAD 和
实际测试计数保存在 `/tmp/schema-b09-a6-evidence.json`；不含这份主仓审计文件。
AKA backend.py 当前 SHA-256 为
`d37620bbffdaaa5b29bccad89af3e4b82c146667e34d4a6f1e313f1e80e613db`。

没有提交/推送 AKA，没有改全局 Git 配置，没有新真实模型/GPU调用；旧实际失败
记录未改写。本交付仅签上述 CPU 工程范围，不能签真实 design/optimize 数学判断、
AKA native transport、GPU/full Task、A6.real 或 FA4/多文件 AKA 支持。下一步可由
主代理对当前稳定字节独立审查，再固定 AKA 提交与完整发布配对。
