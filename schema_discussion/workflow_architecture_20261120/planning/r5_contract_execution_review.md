# 当前合同：API、身份、恢复、预算与资格

日期：2026-11-20。本文是 `ks.workflow.v1` 的唯一合同正文；本文件定义合同，实际完成与资格状态见[实施记录](../implementation/README.md)，不能由规范文字代签。路径保留供既有链接使用；原 R5 推导见[审查前快照](../review_20261120/before/planning__r5_contract_execution_review.md.txt)，此前依据见[R3 裁决](../review_20261120/r3_adjudication.md)，本轮范围及归属澄清见[变更报告](../revision_no_exp_20261120/r2_change_report.md)。
[主计划](../04_implementation_plan.md)负责方向、阶段与发布门，[交付单元](../05_work_packages.md)负责 owner 和唯一依赖，[验收规范](r5_validation_delivery_plan.md)负责 Task、V01–V15、公开材料与消费模式；不以文件优先级覆盖另一份错误规则。

<a id="api"></a>
## 1. 公共入口与固定请求

唯一新 CLI 为 `kernelschema {design|optimize|implement} --request <request.json> --out <run目录>`；恢复分别使用 `kernelschema {design|optimize|implement} --continue <原run目录>`。Python API 为同实现的 `design`、`optimize`、`implement`，接受 `request/out` 或 `continue_run` 两组互斥参数，共用验证、运行定位与结果协议。没有已发行 v1 的兼容别名。
新启动只创建新 run；已有运行根拒绝并指向 continue。continue 核对 operation、原请求、后端/profile、owner 与账目，不能借命令覆盖输入、模型、mode 或预算。三个操作均不依赖首份 result 已存在；恢复细则见[持久边界](#recovery)。
`kernelschema extract -- …` 透传原命令；原输入、P1–P6、ready/publication/receipt、恢复与退出语义保持。消费桥引用原 raw/receipt，补齐 Task 后建立固定请求；手写、curated、design 来源同样合法，不要求统一 P6 门票。
新入口退出 0 仅表示规范 result 已发布，领域状态仍须读取；参数/协议无法准入或宿主无法发布结果时非零并给诊断，不伪造 Q 或收据。

每个新规范 JSON 带精确 `protocol=ks.workflow.v1`；请求/结果用 `operation`，manifest/收据用 `kind`。缺版本、未知版本/核心字段在收费前拒绝；原生报告保留原协议。仅 `extensions.<owner>` 容纳扩展，纯信息扩展可保留/忽略；影响权限、执行或验收的扩展必须由所选后端明确支持，不得削弱硬条件。

| 对象 | 必需字段及限制 |
| --- | --- |
| Task，kind=task | `task_id`；`contract/acceptance/evaluator/environment/workload/baseline_source:Ref`；`starter_source:Ref/null`。闭包包括 ABI、全部副作用、比较式、输入分区、硬条件及计时；数值 reference 与性能 baseline 分开 |
| Schema，kind=schema | `task_sha256`、`raw:[{node,file:Ref}]`、`dispatch:Ref/null`、`design_revision`、`parent:Ref/null`、`change`、`change_report:Ref/null`。raw 保留完整任务与实际解析来源；effective 是阅读产物，不能替原件 |
| Rules，kind=rules | `files:[Ref]`、`tools:[{name,version,source:Ref}]`；规则、Template、角色方法和有限 fidelity 方法及工具/prompt 版本可定位，版本字符串对应实际字节 |
| Materials，kind=materials | `entries:[{id,file:Ref,roles,origin,license}]`；roles 是 planner/coder/reviewer 的显式集合，可为空；来源说明不能替可读文件；仅收当轮所需资料 |
| Scope，kind=scope | `coverage_ids` 为 Task 输入分区，`bindings:[{coverage_ids,node,variant}]`，variant 可 null。design 输入可空 bindings，完成输出及 implement 不可空 |
| Source，kind=source | `format/encoding/entrypoint/files:[{name,mode,size,sha256}]`；完整快照，不是 overlay；首题格式见[源码规则](#source) |
| Backend profile | 后端有版本的固定配置 Ref，含运行时、显式模型、参数、目标环境/能力、必要 native 上限，无凭证；不靠私有字段判公共完成 |
| 请求共有 | `request_id/operation/task/rules/materials/scope/budget/backend`；对象位置均为 Ref，`backend={name,profile:Ref}`；budget 见[预算](#budget) |
| implement 请求 | 另必有 `schema:Ref`、`seed:Ref/null`、`mode=code|verify_only` |
| design 请求 | 不含 schema、seed、mode；使用公共输入与当轮目标 |
| optimize 请求 | 另必有 `schema:Ref`、`feedback:[Ref]`、`evidence:[Ref]`；两列表可空，不含 seed/mode |
| Check | `{status:pass|fail|unknown|not_run,receipt:Ref/null,reason}`；pass/fail 有证据，unknown 有原因，not_run 不伪造收据 |

`request_id` 是显示/关联标识，不充当全局去重键。`backend.capabilities()` 声明实际支持的协议、operation、mode、source 格式、empty_start、恢复边界和 token enforcement；查询不自动试跑 GPU，未知组合先拒绝。
Task 的 Contract/acceptance 对模型公开；私有 evaluator、隐藏用例只给宿主执行器。资料按角色 allowlist 构造，摘要/路径可见不等于读文件授权，不直接挂整个输入根。coder 首调、修复、限流重试、resume/reopen 均保持固定 assignment；来源中的自由优化指令不能覆盖它。
design/optimize 的 Materials 必须有且只有一个保留 `id=workflow_objective` 条目，file 使用 Ref，至少对 planner/reviewer 可见。其内容给当轮目标、优先 workload、计量范围，可写“按 Task 默认目标”；它是当前任务指令而非普通参考，不能修改 Contract、scope 或硬判据。修改目标改变 materials→Q，无须改变 Task 或增加 design_revision；implement 若需要该说明，通过自身 allowlist 显式投递。`performance_goal` 始终按 Task 固定口径计算。
Scope binding 在调用前核对 when、当前设计决定和必要 dispatch；root 不是全任务 fallback，coverage_id 不是测试编号。dispatch=null 仅用于原规则允许且无需变体选择的交接；缺实际所需设计/目录则拒绝准入并列缺项。
Template 复用既有解析、规范化、校验与来源工具，放到唯一包的私有命名空间，不复制解析器或依赖两个裸 tools 的 sys.path 顺序。非 root 阅读采用含 root Contract 的 effective_document；固定 Task 材料可供写作，不要求 GPU 资格先通过。

<a id="identity"></a>
## 2. 字节、路径与身份

- `Ref={path,sha256}`，摘要为 `sha256:` 加 64 位小写十六进制，始终核所指原文件字节；mtime、文件名、Git SHA 不能代替它。
- bundle 根为 request.json 所在目录；全部 Ref，包括嵌套 manifest 的 Ref，均相对此根。路径必须是规范 POSIX 相对路径，无绝对路径、`.`/`..`、空段、反斜线、控制字符或符号链接穿越，仅接受普通文件。固定快照不与可写工作区硬链接。
- JSON 拒绝重复键、NaN/Infinity、非 Unicode 标量；规范发布用 RFC 8785 JCS UTF-8，无 BOM/末尾换行。计数为安全范围非负整数，小数用十进制字符串；采用同一小型 JCS 实现及共同字节向量，不用普通 sort_keys 冒充规范。
- `Q=SHA256(JCS(request))`；request 不含自身摘要。Q 绑定固定输入，首次启动另分配 durable `run_id`；同 run 恢复沿用。显式新 run 可使用相同 Q，独立产生真实费用，不承诺跨任意目录自动去重。
- `R` 表示一个具体 Schema manifest 及设计修订关系，不能只用 revision 数字识别；`C` 是规范 Source manifest 的原字节摘要。producer/operation 身份包含实际 run 命名空间，避免相同 Q 的两个真实运行撞号。
- YAML/Markdown/源码/原生报告保留原字节，不 trim、换行转换或 Unicode 归一化。规范化 YAML 另成候选/新快照，不能摘要后原地修改。
- 集合数组按键排序并拒绝重复：source.files 按 name、schema.raw 按 node、rules.files 按 path、materials.entries 按 id、scope.coverage_ids 按 ID；其余数组保留声明顺序，JCS 不替应用排序。

<a id="source"></a>
### 源码与 seed

首题 source 支持 Linux `single_file`、严格 `utf-8`、唯一 `kernel.py`、mode=`0644`；BOM/非 UTF-8 拒绝，CRLF、无尾换行和非 ASCII 注释逐字节往返。files 的 name 是工作区逻辑名称；实体位于 source manifest 同目录的 `files/<name>`，这是唯一不同于 Ref 的路径规则，取用和安装均复核 hash/模式/size。
Source manifest 不含父 C、Q、时间或 native revision；关系放收据。相同字节/模式清单得到同一 C。公共 C→native source identity→实际评测来源须显式映射，不能重编码 public-candidate 整包冒充原 C，额外 solution.json 也不暗中成为 kernel.py 的一部分。
`seed=null` 唯一含义是使用 Task.starter_source；二者皆 null 才为空起点。后端不支持 empty_start 则收费前拒绝。不可读 seed/摘要不符不能退回骨架或 G；删除必需文件时保存损坏 draft 诊断，latest 仅指最近完整稳定 C，并说明不是当前损坏草稿。
`verify_only` 必须提供非空 seed，零 coder、零改码 episode、零新晋升；必要执行/审阅照常计费。改 mode 是新 Q，不可通过 continue 热改。
F1 实际需要的新 source 格式在该案例内明确能力、完整文件清单/删除/模式/固定文件隔离及原子安装规则；未支持组合拒绝。数组存在不等于已承诺任意源树。

<a id="publication"></a>
## 3. 不可变发布、选定闭包与写作顺序

首次冻结前分配并持久化随机 128 位 `publication_id`，所有可被 Ref 引用的文件从首次发布起使用 `artifacts/<publication_id>/` 稳定前缀；同发布操作恢复复用该 ID，新发布分配新 ID。原路径、字节、模式及摘要不可改写。
输入快照、Schema/source、原始报告、用量、不可变 request/result 历史副本可位于此前缀；顶层 request.json/result.json 只是入口/投影，不作为可导入 Ref 目标，`.runtime/` 为私有恢复状态。完整 bundle 可移动读取；移动 artifacts 不承诺私有 run 可跨机恢复。
导入根由调用者显式选择 Schema、source 或证据。按受支持对象语义递归取实际依赖：Ref、source.files 实体、Schema.parent/change_report、raw 的真实解析来源，以及原生协议声明的依赖；不能只扫 JSON 字典，也不把普通文本路径全部升格为 Ref。缺无关历史或未选私有日志不拒绝，缺真实依赖拒绝。
选中证据须能定位其 Q/Task/R/C/规则/方法/evaluator 等实际主体；subject 的裸摘要不是主体已随包存在的证明。当前 request 中的原件可复用；历史证据缺主体时须补选不可变 manifest/request 快照，不能猜路径或重贴新 Q。只导入 C 不要求带历史 Q，私有历史的限制不传播给无关 C。
先暂存并逐 Ref/source/原生依赖核验，再发布。已存在同路径同字节/模式可幂等复用；同路径异字节/模式、缺依赖、非法路径整次拒绝并保持目标不变，不重写旧路径再沿用旧摘要。
公开说明包、可导入产物包、私有完整归档的边界见[公开材料](r5_validation_delivery_plan.md#public)。脱敏 L′ 是新字节，不能替 L 满足旧 Ref；选中证据实际依赖 L 时不能声称原证据完整闭包公开可导入。

写作发布使用以下无环顺序：固定输入与草稿 → 规范化候选及全任务/权限/设计影响检查 → 冻结 change_report → 冻结 raw、scope、Schema → writing_review → 结果。
change_report 是冻结前 diff/影响检查，可绑定输入及待发布 raw 内容，Schema 引用它；终态 writing_review 绑定已冻结输出 Schema/scope 和输入 Q，作为结果证据，Schema 不反向引用它。审阅可复用前述事实，无须生成两套完整语义审阅；若终态检查失败，候选保持不可变但不得作为 completed 发布，修正另冻结。
keep 默认复用原 Schema/scope；有据的同设计表示/合法 Feedback 更新才另发 unchanged。design/optimize 的结构、权限、语义、发布四检查均为完成门；写作成功不证明代码或性能。

下面仅展示选定 source 根的内部一致布局；摘要与 size 由示例文件实际字节计算。JSON 块便读排版，实际 manifest 使用 JCS。其它真实模板由 K2/K6 生成完整可核 bundle，不提交占位摘要执行。

```text
request.json
artifacts/00000000000000000000000000000001/source/manifest.json
artifacts/00000000000000000000000000000001/source/files/kernel.py
```

kernel.py 内容为 5 字节 `pass` 加 LF；manifest 如下。这只说明 codec，不是合格 kernel。

```json
{
  "protocol": "ks.workflow.v1",
  "kind": "source",
  "format": "single_file",
  "encoding": "utf-8",
  "entrypoint": "kernel.py",
  "files": [{"name": "kernel.py", "mode": "0644", "size": 5, "sha256": "sha256:9f56e761d79bfdb34304a012586cb04d16b435ef6130091a97702e559260a2f2"}]
}
```

<a id="revision"></a>
## 4. 设计变更与证据适用性

| 变化 | Schema.change / design_revision | 后续处理 |
| --- | --- | --- |
| 首次设计/新 Task，或 ABI、比较器、硬件/栈等 Task 身份变化 | `new_task`，新 Task 内从 0 开始；parent 可记录原出处 | 新固定请求/所需 native Task；旧 C 可显式 seed，旧费用保留 |
| 同 Task 授权改变设计/约束、when、dispatch 的实际决定 | `new_revision`，parent+1，必有 diff/影响检查 | 发布新快照，旧事务结算与作用域关闭后另准入；重判 fidelity |
| keep/同字节/合规纯格式或 Feedback 更新 | `unchanged`，计数不变，优先复用原件 | 新字节仍改变 Q；旧证据保留原绑定，不自动全继承 |
| 选择已有合法 scope/variant，或仅变规则/材料/目标/seed/mode | Schema 修订数不变 | 新 Q；未作设计决定的 variant 不能直接当已有选项 |

字节相等可自动识别，纯格式/允许字段差异可机械核对；其他实质变化不能凭 planner 自称等价跳过影响审查，未决不准入。Q、revision、optimize 次数、coder e 次数分别计数，keep 的真实调用仍计费。
Task 方固定 Contract/evaluator；planner 只改授权 Design/Realization 及元数据/目录，coder 只写授权 source。普通反馈留外部；REALIZATION.Feedback 仅收针对 Prefer、带 `repository@ref:path` 不可变实测证据的合规提案，不能放宽 Require/Forbid。
执行须固定 staging 并核实际装入 C/evaluator，避免可写工作区竞态与旧 import 缓存，返回受信来源摘要、实际环境及原始终态。提交前 hash 一次不足以证明实际测了这些字节。
数值/性能复用至少核准确 C、Task/完整 evaluator、输入/种子、实际环境、计时及 backend/profile/adapter 的适用性，保留旧报告并另记当前适用依据；文件名或相同 shape 不够。fidelity 另绑定当前完整 Schema、rules、scope 与方法；新 R 必须重判，不能只换收据 hash。数值报告可按自己的规则独立复用。
同一后端/主体/方法/profile 的既有真实记录可支持多个适用声明，变动只补受影响路径；KS pass 不能代签 AKA，跨后端声明须双方自己的真实证据。

<a id="fidelity"></a>
## 5. 有限 fidelity、报告与原生资格

Rules 冻结一个有限检查方法及 checker/prompt/工具版本。方法读取当前完整 Schema 与 scope，解释适用的数值路径/精度、work ownership、读写与寿命、Require/Forbid、覆盖及必要设计决定；不能把 R0 专属条目硬编码成所有 R 的常量。
每次报告列实际义务、Schema 定位、准确代码位置、pass/fail/unknown 与理由。读取完整来源用来解释当前 scope，不要求有限 scope 实现未选节点；新增适用义务无法由方法判断时为 unknown，不能由 planner/coder 删除义务豁免，也不增加清单 planner。
一份 `kind=fidelity` 报告含 `coverage={status,mappings:[{coverage_id,node,variant,source_locations}],task_uncovered}` 与 `fidelity={status,items:[{id,obligation,schema_locations,source_locations,status,reason}]}`；status 使用四种检查状态。assessment 的 coverage/fidelity 可引用同一报告 Ref，但两个判定独立。数值 pass 不能代替覆盖/设计符合，缺 dispatch 与设计违规分别诊断。
报告还必须有 `operation_id`、`subject={request_sha256,task_sha256,schema_sha256,rules_sha256,scope_sha256,source_sha256}`、`method={spec:Ref,reviewer:{runtime,requested_model,reported_model,implementation,scope_id}}`、`limitations`、`evidence:[Ref]`、`usage_receipts:[Ref]`。身份由宿主核对，未知报告模型可 null，三个笼统 pass 标签或 coder 自评不够。
机器项只判断能证明的部分，独立只读 reviewer 给可追溯的有限审阅；AST/只读权限本身不证明完整 fidelity。必需 fail/unknown 不 qualified，记录人工/离线证据也必须满足绑定与适用性，不建设通用证明服务。R0/Rd 配对反例见[V02/V09](r5_validation_delivery_plan.md#validation)。
`kind=evaluation` 报告有 operation_id、subject 的 Task/source/evaluator/inputs/environment/timing 摘要、build/correctness/performance 状态及诊断、case_ids、samples Ref/null、raw_receipts 与 usage_receipts；environment 必须是实际回报快照。`kind=writing_review` 按[发布顺序](#publication)绑定四项检查，汇总不能替原始事实。
AKA 三条路径分别闭合：普通 seal 后/ABBA 前验资格；未提交恢复重建 attempt 时保留同一资格绑定；已提交恢复从 commit-bound promotion audit 吸收原资格/用量。资格先 durable，再写入 attempt/晋升事务，不能只在 active 加松散字段。
已提交 audit 缺资格时保留 native 历史，Schema 资格 unknown，停止自动推进；不能新问模型改历史或补造 pass。native production 检查如果确需新审阅，按新真实操作准入计费；吸收旧 fidelity 不表示全部原生恢复免费。

<a id="budget"></a>
## 6. 单 owner 预算与用量

请求 budget 必有 `parent:Ref/null` 和 `limits`：`calls.{planner,coder,reviewer,extract}`、`evaluator_jobs`、`active_ms` 为有限上限，`tokens` 为整数/null（null 仅表示未设 token 上限）；可选总 `model_calls`、`deadline_utc`（RFC3339 UTC）。0 禁止准入。
parent 声明形状为 `{protocol,kind:"budget_account",id,owner,limits,deadline_utc:null或时间}`，只描述账户，余额/预约由唯一 owner 的私有持久状态掌握。parent=null 的顶层 run 自持 owner；外壳创建的子请求即使新 out/Q/R 也沿用该 owner，不能重置余额。首期只需顶层 owner 与串行子请求，不承诺任意深度账户树。
KS 可同步调用外部已安装 AKA，AKA 可调 KS 单次 planner，不限同进程。调用前 owner 持久预约受限额度，子后端依同一授权/pause/fence 做内部准入；可持久授权内消费和 native 状态，不能另复制可自由加额的父账本。恢复原预约，不重复授权；不能贯穿限额/暂停的外部 profile 不合格。
全部收费接点取父/请求/native 最紧限额，含准备、V0/V1/helper、coder、native 预取 reviewer、fidelity、production/终态评测、失败重试和真实恢复重评。native 内部 V0 与 reviewer 可并发，两者都预约/记账，外壳仍只有一个 owner。
一次可产生 token 的宿主 Agent/CLI invoke（含真实 resume/reopen/review）计一次角色调用；内部多轮/物理重试另录实际消耗。每个不同 V0/ABBA/测量逻辑 operation 按实际提交计 jobs，内部物理 attempts/GPU 单列；token 不支持调用内硬停时不得冒称硬限保证。
准入前预留已知最低收尾额度；不足则 partial/not_run，不能预算外评测后代签 completed。已发生超额记录真实用量并禁新准入，已花额度不退款；只有证明未启动的预约可释放，未知在途额度不释放。

用量按 `producer（含run命名空间）+operation_id` 关联，receipt_id/sequence/摘要判重与冲突，累计收据只吸收增量。同操作收据重放零新增费用；同输入新真实调用使用新 operation_id 并计费。显式 continue 在原操作结算、确有下一行动且原限额允许时可准入新操作。
`usage.scope=request_cumulative`，含 `receipts:Ref`、角色 calls、evaluator_jobs、tokens、active_ms、unsettled；索引行记录 `{producer,operation_id,role,purpose,receipt_id,sequence,receipt:Ref/null,state:settled|unknown,calls,evaluator_jobs,tokens,gpu_ms}`，未知统计为 null 并保留已知下界。总 token 有未知相关分项即 null；不把分项与 total 或 child/native totals 重复相加。
操作是否终止/结算与统计是否齐全分别记录：已终止操作某项 token/GPU 统计 null 不自动意味着仍在途；若某个现有限额无法核定剩余额度，仍须 fence。检查 unknown 的补证规则见[状态](#state)。
owner 维护一个活动时钟与活动/静止转换锚点，包含模型、排队、GPU、review 及外部等待；父等待与子调用不重复加时间。暂停后在途仍计时，确认全部静止才停；无法证明静止的崩溃间隔保守计入。无须给每收据强制 active_ms 或建设一般区间合并引擎。
请求 active_ms 从同一 owner 时钟的当前 Q 活动段投影，不填整个父账户历史；父总时间直接累计，不再加子投影。native 仅计 Agent 的 active-time 保留原名作为扩展/额外约束，wall_ms 可另报含暂停的日历时长。
来源中的历史 design/extract 初始化费用与当前请求用量分列，不能伪装为当前新增调用；人工或缺失历史用量保留未知及原因，不能记零。

<a id="recovery"></a>
## 7. run 定位、持久边界与恢复

首笔费用前持久化本地 run 锚点：run_id、operation、原请求/Q、后端/profile/实现身份、预算 owner、原生运行定位。每个收费动作先保存 operation_id、输入 C、角色/purpose、预约、目标 e/提交键，再执行；终态保存同 operation_id 的产物、真实用量和未知项。
外部 AKA 先由宿主固定运行/操作键，adapter 在执行前持久保存 native run 实际路径、版本、键及必要私有句柄的关联。首个返回或私有句柄传输丢失时仍能按该关联取回原操作；只在最终 result 才返回不可重取句柄的接法不合格。私有句柄不是公共恢复门票。
三个新操作均按原 run 目录恢复：先核身份/所有原操作及 owner，取回、结算并重建 result；不得因 result 缺失重提。写作同样至少保证结算/重建发布结果；只有原请求未完成、有明确下一行动、原操作已结算且原额度允许时才可再付费。completed/keep 无行动不会因 continue 再收费。
丢失或损坏 owner/运行定位的旧 run 不能当恢复重新发额度；保留可读产物并拒绝新准入。恢复承诺限合格本地 Linux 文件系统和保留的后端安装/私有状态，不承诺迁机、移动私有 runtime 或隐藏 CLI 上下文搬运。

| 中断点 | 必须保留的事实及恢复结果 |
| --- | --- |
| seed intent→ready→register/admitted | 完整暂存/核 hash、模式并落盘后才 ready；未准入可幂等重装；已准入或 draft 变化后禁止 seed 覆盖 |
| coder/外部 job 身份或终态未知 | 保留原进程/会话/operation/key 和草稿；先查询原事实，404/取消接受/超时不证明未执行，不能换通道重提 |
| pause/耗尽/未结算或余额不明 | 持久 fence，禁止新调用/e/R；查询吸收原事实允许，任何新模型/GPU 恢复动作须另过准入 |
| R′ 发布→新请求准入 | 旧事务结算/旧 scope 关闭/余额允许后，持久 admitted_request；仅发布不取消旧 R，不读 latest 猜新请求 |
| 资格→晋升→result | C/资格先 durable；已提交从原 audit 恢复、不重晋升；result 是可重建投影，返回丢失不丢费用或重新开 e |

输入、意图、ready、资格、准入和发布事实须 flush/fsync 文件及相关目录，同文件系统 rename 后同步父目录；不能把 atomic rename 本身称为断电 durable。单文件也需这些边界。保存稳定 C 后可在新会话续写，未知进程未停前不能以开新会话绕过 fence。

<a id="state"></a>
## 8. 公共结果与阶段状态

结果共有 `protocol/operation/run_id/request_id/request_sha256/status/stop_reason/usage/feedback/recovery`；无产物用 null/空列表。recovery 仅为 `none|continue|settle_only|new_request` 的动作提示，不授权新增余额。公共核心不传恢复 token。
implement 另有 `code:{latest:Ref/null,qualified:Ref/null}` 和 `assessments`；每个准确 `source_sha256` 一份当前汇总，列 coverage、task_uncovered、build、correctness、fidelity、performance、performance_goal，原报告全部保留。performance_goal 为 met/unmet/unknown；performance=pass 仅指规定测量完整，Task 另定的性能硬项仍须满足。
写作另有 `schema:Ref/null`、`scope:Ref/null`、`writing_checks:{structure,permissions,semantics,publication}`；optimize 有 `decision=keep|revise`、reason、`next_action=none|implement|verify`，非 none 必须给 action_reason 与所需输入/证据。无行动的 keep 正常完成。
latest=C2 未测、qualified=C1 合格可以并存并明确交付 C1；C2 不继承 C1 的 pass。qualified 只针对当前 Task/R/rules/scope 的适用资格，不能指历史 G 冒充当前合格。原生 e/G/HEAD/promotion/stop 放扩展分列；下一 e 由当前 HEAD 准备并安装准确终态 C，G 只作 native 比较基准。
新 R 已准入时，当前结果只投影其请求与适用资格；未完成的新 R 仍是 partial/blocked，旧 `(R,C)` 可保留并明确标为历史，不能填成当前 R 的 qualified。旧 Q/R 的迟到原收据只按原 operation/主体/账户吸收一次，不改当前 admitted_request，不把旧 pass 贴到新 R，也不绕过未结算 fence。这是现有结果与恢复的归属规则，不新增跨请求候选排序或结果 API。

| 事实 | status / recovery 与下一步 |
| --- | --- |
| 当前范围完整 qualified，或四项写作检查通过 | completed/none；软目标 unmet 仍可完成；无优化行动停止 |
| 代码/覆盖/必需检查未完成、编译/数值/fidelity 缺陷 | partial；原操作已结算、有具体行动和余额才 continue，否则 none；同 R 修复 |
| 已终止并结算的检查 unknown | partial，仍不 qualified；有具体补证及余额可同 R continue 或新固定 verify_only；不是设计阻碍 |
| 能力/材料缺失 | blocked；按缺项决定 none/new_request，不能虚构执行证据 |
| 操作身份/终态/计费未闭合，或现有限额余额无法确定 | blocked/settle_only；保留产物与历史 qualified，但不能 completed 掩盖恢复责任、换 R 或换通道绕过 |
| pause/耗尽 | 禁新准入；有未结算事实先按上一行处理，否则未完成为 partial/none；已完成交付可 completed |
| 独立可核对的设计阻碍 | blocked/new_request；仅交已授权外壳决定 planner，fixed backend 自身零 planner |
| 输入/协议/摘要/原 run 身份损坏或不相容 | failed，none 或 new_request；已有产物保留，不能静默重置 |

D3 的阶段交接要求：当前实现阶段完整合格且有已授权优化目标，或有独立可核对的设计阻碍；同时必须无 fence、有外壳授权和余额。普通编译失败、超时、无晋升、“写不出”、fidelity unknown 都不足。
keep 若提出明确补证则建立 verify_only；提出同 R 改码则新固定请求或合法继续原请求，父预算不变。revise 冻结新 R 并按[持久边界](#recovery)另准入；固定 backend 始终不调用 planner。
