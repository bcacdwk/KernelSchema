# 第二轮：正确性、控制提交与故障恢复复审

2026-11-21；基线 HEAD `484902855fe7be82a74fe7c394503d1d7d222dbb`。本报告是独立反证与实施计划，**没有实施产品修改，也没有生成新的批准**。已读根 AGENTS、审计 README/BASELINE、第一轮两份报告及 FINDINGS、十二项 RECAP/FINAL_REPORT/DELIVERY_INDEX，按问题回查实际代码、原回执和必要恢复记录。只写本报告与自己的 evidence；未运行真实 extract/resume/monitor、模型或 GPU，未改正式 YAML、dispatch、receipt、运行态、issue 注册表，未提交或 push。既有用户修改和 workflow architecture 讨论保持原意。

三项核心缺陷均成立，而且证据更强：S-06 已通过临时固定来源下的真实 `accept_result` 路径进入未批准 checkpoint；X-08 已在真实临时持久账本上重现授权后及模拟启动后的错误 `committed=false`；X-15 的两份原 reviewer 工作目录换用实际 worker 计划后，当前只读 P2 检查均通过。

需要收窄第一轮的两处推论：仅修公式左值的 `related` 判定不能解决 S-06；仅看到坏回执被自动补正，也不能声称身份/批准门禁失守。**拒绝采纳、独立保留未批准草稿、允许原角色重交是三件事。** 普通模型记录错误宜继续自动补正；原生结果通道不一致、实际来源漂移及未确认终止须分别处理。

仍未证实：33 份成品发生过此类名称腐化、独立 reviewer 会漏检该反例、线上发生过重复模型调用、KDA 的 ctime 改变原因。禁止把这些推测升级为本轮事实或全量重提取的理由。

## 1. 第一轮逐项去向

`accept` 同意事实及必要行动；`refine` 保留问题但缩小/改进修法；`reject` 仅否定所列推论；`defer` 不借本轮扩查。

| 项目 | 第二轮判断 | 当前修正与建议的去向 |
| --- | --- | --- |
| S-06 | accept + refine，优先 | 保留事务/原稿/只读保护；修名称身份和不明绑定的替换。拒绝“只改 related 即完整修复”及“已证明全批腐化”。 |
| X-01 | accept | 保留 P2/P3/P4 **approve** 的 1200 字符/20 非空行门禁；同一条件生成提示和字段说明，不强推 backend 条件 schema。 |
| X-02 | accept | 保留 category→绑定 path 门禁；直接展示 implementation→entrypoint 等义务。当前是每类命中其允许 locator 之一，不偷偷改为全部 locator 或 AST 范围证明。 |
| X-03 | refine | 保留真实字节/坐标校验；改进索引、行数、offset 诊断。区分非法引用选择与实际来源/权限破坏，不自动改证据。 |
| X-04 | refine | 合并安全可判定的机械诊断，保留 native 错误细目；不维护另一套宽松 validator，不把所有 schema enum 错误一律升级人工恢复。 |
| X-05 | refine | 推荐普通回执补正只受**现有有效 scope 调用预算**约束；删除该类的累计两次停止条件。传输/CLI、维护、单次裁决/签收不随之扩大。详见第 5 节。 |
| X-06 | refine | 版本、字段范围、目录门禁及原稿证据保留；“安全规范化”要以 S-06 修复为前提。正文语义/Version 不由工具钳制。 |
| X-07 | accept | scoped recovery 与 tools-only 续接保留；不用新恢复控制器替换。去掉每次复制全历史作为例行前置的建议。 |
| X-08 | accept，优先 | 改提交/投递结果与关联，保留现有 request、launch、单写者和 claim 门禁；不要重放整段副作用。 |
| X-09 | accept + 简化 | 区分缓存过期与已从 state 读到的样本；不忽略 ctime、不新增修缓存命令或服务。 |
| X-10 | accept + 简化 | 只在 `record_scan` 生成有效 anomalies；`Clock._complete` 已消费此数组，不再复制过滤算法。旧 rows 保留并先标历史再展示。 |
| X-11 | accept + refine | 新显式停启代际重新 initial scan/due；同一 enabled 代际 daemon 恢复保留原提醒/claim/due。当前已停 monitor 不启动。 |
| X-12 | accept，无代码必要 | 无覆盖发布和 requested/reported 模型身份保留；补输出空占位约定即可，不据告警换模型或声称实际上下文/effort。 |
| X-13 | accept + 简化 | 文档说明已有输出形状、结算边界；summary 投影已有计数。不要添加通用脚本适配层或把未知 usage 填零。 |
| X-14 | accept，排除 kernel 修复 | 保留源风险、性能例外、证据缺口；正文是否失真由同伴定点审阅，不由恢复代码改发布字节。 |
| X-15 | accept，优先 | reviewer 检查实际 worker 候选；保留只读、approve 计划相等和真实目录检查。首次待审不标成 removed。 |

其余 S 项的正文、规范及当前维护清单由 [SCHEMA_AND_CONTRACT_REVIEW](SCHEMA_AND_CONTRACT_REVIEW.md) 主审；这里不把同伴结论冒称已再次逐行核验。

## 2. S-06：成立于有效数学身份，正常采用路径也会经过它

**输入边界。** DOC0:63 要求节点 ID 和具名登记键小写，没有规定任意 Math 临时量都是大小写不敏感的别名；DOC0:75 允许非 Python 数学记法，DOC3:134 允许未登记名称在 Math 中定义。`V` 与输入 `v` 是两个明确对象。即便选择对它们提出风格诊断，也不允许工具把二者合并。

本轮把原 scalar fixture 的说明、Contract、Work/Pipe 改成一致任务，使用 `int32`、输入 `range: [0,16]`，排除浮点舍入或溢出争议。原 `V=v+1; y=V` 在顺序赋值读法下只观察 y 会偶然同值，因此不单靠它证明输出变化。补充的窄变体是 **`V=v+1; y=V+v`，Contract 及临时固定实现/reference 均为 `2*v+1`**；归一后即使按顺序赋值理解，输出也变成 `2*v+2`（v=1 时 3→4），不存在这一争议。实际结果见 [复现脚本](correctness_evidence/s06_repro.py)和[结果](correctness_evidence/s06_result.json)：

| 验证层 | 实际结果 | 能证明什么 |
| --- | --- | --- |
| 完整 `validate_documents(skip_hardware=True)`，归一前后 | 原例与窄变体都通过；窄变体变为 `v=v+1; y=v+v`；再次规范化无变化 | 完整静态检查及幂等性不保证数学身份或输出保真。 |
| 真实 `normalize_candidate(P3, worker, complete)`，临时目录 | `landed`，严格 P3 check=0；原稿与基线不变 | 事务的字节与权限保护没有拦下这个语义改写。 |
| 现有 StageFlow fake backend，真实 `accept_result`，固定源 `2*v+1` | P3 worker 回执被接受，checkpoint 含 `v=v+1; y=v+v`；下一角色 reviewer，P3 approval=null | 固定来源/原稿/规范化/回执门禁后的实际采用路径可到达；没有证明独立语义审阅会批准。 |
| 仅在内存取消 `_formula_scopes` 的 `related` 豁免 | 普通 `V =` 被拒；开放记法 `V := v+1; y=V` 仍改成小写且通过 | 单点修改不足；`_tokens/_Names.find` 的不精确 casefold 命中也在问题路径上。 |

**最小保真修法。** 修改现有 [normalization.py](../../../schema_template/tools/normalization.py) 的 `_Names`、`_formula_scopes`、`_math_tokens/_tokens`，不新增数学证明器：

1. 区分声明身份、原始拼写和目标拼写；casefold 相同只能是潜在碰撞，不能自动证明 related。重复的同一登记/引用可以同步；不同定义不得多对一合并。
2. 对 Math 只改有明确绑定依据的引用。未登记的数学临时名和无法判定的开放文本保留原文；若必需的登记名改写会穿过不明绑定、捕获另一个原有目标名，则整组拒绝，并给出两个位置。不要增加 `:=`、`let` 等逐个正则补丁。
3. 重用已有局部参数、comprehension、Pipe/formal 作用域；保留目前有效的 I/i、T/t 捕获拒绝。确定的字段拼写、必要引号和布局整理继续工作；不把整体关闭规范化作为修法。
4. 保持 `normalize_documents` 整组先规划、Extract 事务比较摘要后落地；拒绝时不写原稿、provider 或兄弟。当前未知文本没有必须改的名字时，不为了“能解析全部 Math”增设新门禁。

验收需覆盖：上表两种定义；外层/局部同目标名；反向大小写（登记 `V` 而正文另有 `v`）；两个局部定义归一后撞名；同一 provider 在不同 consumer 上的绑定分歧；精确同对象的安全更名；第二次无改动。捕获反例必须保留两个对象或在写入前拒绝，不能只断言 checker 成功。个别既有测试把未证实的 `T` 当作形参 `t` 的别名，应按新保真契约修正其期望，不能拿旧期望反过来批准捕获。

**迁移。** 这是 Template 改动，现有 `executor_repairs._target` 只允许 `schema_extract` 非测试 Python，不能把本修复偷渡进 tools-only 续接。新 run 使用新规则快照；当前十二项保持已交付状态。生产影响、独立 reviewer 命中率均 defer，不自动重新规范化 33 稿。

## 3. X-08：提交是操作级事实，busy 只说明某次事务未取到锁

真实路径为 `cli.main → request_restart` 先提交 `ledger.retries[id]`，随后 `start_background → reserve_launch` 登记 `starts[run].launch`，然后 Popen、再次取锁写回。`SupervisorBusy` 描述的是**当前取锁**失败；`cli.py:592` 把它升级为整个命令 `committed=false`，范围错误。

[独立复现](correctness_evidence/control_commit_repro.py)使用真实临时 `Store.supervisor` 落盘，只替代静态预检、进程观察与 Popen，不使用内存假账本：

| 注入点 | 已持久 retry | 已登记 launch | 模拟 Popen 次数 | 当前 CLI |
| --- | --- | --- | --- | --- |
| 首次登记前 | 0 | 无 | 0 | busy / false，正确 |
| retry 已登记，reserve_launch 取锁前 | 1，authorized | 无 | 0 | busy / false，错误 |
| launch 已登记，Popen 后写回取锁前 | 1，authorized | 有，retry_id 正确关联 | 1 | busy / false，错误 |

第三种 fixture 中再执行同一请求，现有逻辑复用了原 request/launch，模拟 Popen 总数仍为 1。这说明已有去重/claim 保护值得保留，**不能把回执错误等同已证实重复启动**。现场 [FMHA 记录](../../extract_twelve_20261119/recovery/p3_handoffs_check66/group_a/fmha_bf16/README.md)确已证明第一次零提交、第二次只加一次调用；本发现不改写那段历史。

**实现职责与接口。** 在 [monitor.py](../../../schema_extract/tools/monitor.py) 的已有操作函数保留已提交对象并处理后续投递失败；CLI 只呈现结果，不从异常类型猜提交事实。沿用 `{request, launch/delivery}`：

- `committed=false` 只在确认该操作的请求/登记均未提交时给出；提交过则返回 true 和原 ID；无法确定则 null/unknown，附 run 与已知 ID。launch 预约、进程启动、request consumed、模型调用结清分别展示，不拿其中一个代替其他几个。
- 优先复用 `retries.id/failure_id`、`launch.id/retry_id`、输出登记 request_key、hold ID、clock generation/daemon ID；不用新的通用控制日志或请求服务。busy 后继续等锁只围绕一个待进入/待写回事务，Popen、signal、增额、apply 都不能包进重试循环。
- `inspect_runs` 已读取 supervisor，可增加相关 scope 的最近/待消费请求投影，使返回丢失时仍能按 run/scope/request/launch 对账；复用现有 ledger，不把“重试同命令”做成查询接口。
- 顶层兜底在没有局部提交证明时保守返回 unknown。非零退出、空 stdout、超时都不等于未提交；不输出可能误导整命令重放的 false。

同类边界须在本次小改中逐一核对，但无需重构其机制：`request_restart`、`start_background`、maintain 提交后启动、extract 登记/创建后启动、hold release 后逐成员启动、普通 resume/export 预约后运行、clock start/stop 登记后 spawn/signal。cancel 已有 request/delivery 分离的局部处理，复用其表达，不倒退成统一 false。

验收按上表故障点核实际 ledger、同 ID 对账、Popen/signal/增额次数；再加提交后 pause/cancel/hold 覆盖、响应丢失后只读对账、旧 launch 不能 claim 新预约。控制确认必须绑定本操作，不能仅看到 supervisor.revision 增长就称提交成功。历史请求不回写；未知投递仍受既有归属 fence，不能因显示修复启动第二模型调用。

## 4. X-15：检查对象错误，不是选集被回滚

直接读取两份 accepted reviewer 回执及 check，并在**原工作目录**仅换用此前已接纳 worker 的 plan 运行当前 `check_stage_snapshot(P2)`：

| run / reviewer attempt | 原决定/计划 | 真实 worker 计划 | 原 check | 按实际候选重查 |
| --- | --- | --- | --- | --- |
| `2b64ba84…` / `b5e8fa14-ef9a-4d29-b3e0-5f9673d83f17` | revise / [] | `3684aa2e-b01a-44d9-a02f-12fc24148990`，7 项含 root | exit1，缺 root | exit0，root + 6 个仅计划节点 |
| `5be021ee…` / `7a274da1-14ab-456f-80af-f76f6403807e` | revise / [] | `541c0592-84a4-47bc-a161-834e1e62b4cc`，3 项含 root | exit1，缺 root | exit0，root + 2 个仅计划节点 |

`receipts.validate_receipt` 对合法 revise 不要求 plan 等于候选；`runner.accept_result:610` 却把空 plan 当候选；`stages.advance` 随后把假 checker_input_error 加入真实退修意见，增加 check_failures。原 `plan_draft` 没被清空，没有因此新批准。

最小改动只选对参数：普通/维护 P2 reviewer 的候选检查使用已读 worker 的 `state.plan_draft`；approve 先保留已有计划签名相等验证，再检查同一候选。adjudication 的显式 select/patch 路径保持自己的已核候选身份。不要跳过 `validate_index`、要求 revise 伪填批准计划、或移走前置 reviewer `_artifact_gate`。`prompts.receipt_record` 在首次尚无正式 selected_designs 时用当前草案/待审关系，正式移除后仍保留 removed 的含义；标签不制造批准。

必要回归：worker complete→reviewer revise/[]→下个 worker，无假 check_failures/缺 root 反馈且真实意见保留；真实坏目录仍报错；approve 擅改 instance 仍拒；reviewer 改 raw 仍拒。历史 check、accepted_receipts 和计数不重算，只记录旧工具归属错误。

## 5. X-01～X-05/X-07：可自动补正，但不能把所有错误当成同一成本类别

### 5.1 独立反证后的三条边界

[定向实验](correctness_evidence/receipt_classification_repro.py)用现有 fake backend 生成真实结构校验结果，再交当前 `_receive`：缺 summary、manifest 外的引用字符串、worker 写 approve，均进入 receipt_repair；错 nonce 直接拒绝；Codex native_output_matches=false 也进入 receipt_repair。所有案例 **accepted_receipts 不增、approval=null**。这是离线边界实验，不是线上原生事件的复放。

不能据此声称权限绕过。`_recover_call` 在保稿前独立执行：`store.validate_attempt`、精确 session、固定 run 输入、原 checkpoint；worker 用 `_normalize(..., continue)` 核文件/字段/Version/Source，reviewer 用 `_artifact_gate` 核原字节只读。只有这些门禁通过的稿件才成为**未批准草稿**。有意义的内容可以暂不通过完整完成检查；这不等于可信来源或权限可以缺省。

`agent.read_result:188–197` 在 receipt_error 前已经核退出 0、唯一 native success、cwd、精确 session；坏 nonce 在后续身份检查仍被拒。因此 schema-invalid 既不代表安全可采纳，也不自动代表必须人工恢复。不能把第一轮保留“身份/来源硬门禁”的要求扩写成所有 enum 错误或非法选择字符串都需人工处理。

| 实际问题 | 接纳/批准 | 未批准草稿 | 下一步推荐 |
| --- | --- | --- | --- |
| 缺普通字段、类别/结论漏项、固定文件内行号/offset/索引错、适用阶段 handoff 超长 | 全部必需项通过前拒绝 | 独立保稿门禁通过才保存 | 原角色、新 nonce、有界重交；不能工具代写结论/引用或截短含义。 |
| decision 误写为本角色不允许的值；引用选择字符串不在固定清单 | 拒绝；不执行所声称决定、不读取清单外路径 | 同上，与坏选择分开检查 | 可让原角色按本次允许值/固定清单更正；不授予新权限、不扩大来源。与下一行区别明确。 |
| nonce/调用身份/会话错误，机器字段冒认，实际文件 hash/来源/版本/权限/只读完整性失败 | 拒绝 | 不把未过门禁的新稿变为 checkpoint；原 workspace 留作取证 | 保持既有停止/责任入口；源码或权限问题不能靠修回执绕开。 |
| 有效 revise、blocked、reopen | 按各阶段真实决定处理 | 沿既有稿件/退修规则 | 不是 receipt repair；不能借余额把语义阻塞变为继续。 |
| native 末条结果与 output-last-message 不一致，但退出、归属、session、终结已证实 | 当前回执仍拒绝 | 可独立保留已核的未批准草稿，不挑一个矛盾响应当批准 | 精确报告输出通道不一致，推荐沿**现有 CLI/传输失败有限策略**；不随普通 receipt 取消两次 cap 而一并扩大。 |
| 未确认终止、缺可靠 native 完成/实际 exit、归属未知 | 不接受、不视为已结清 | 留证据；采用仍须原恢复门禁 | 保持 fence；确认后走原中断恢复/精确来源核验，不盲重放。 |

原生输出不一致已由现有 `test_codex_native_output_is_distinct_and_bound_to_last_message` 明确测试为 receipt_error；所以**不是新发现的批准漏洞**。本轮确认的问题是诊断将它错说成 schema 错误，且 X-05 若按异常大类直接取消 cap 会顺带扩大传输异常的自动成本。默认修法可在 `agent.read_result` 或 `_receive` 分出精确原因，复用 `cli_failures` 的现有有限策略和会话恢复；不用再造每种错误的新计数器。这个选择不要求用户逐次批准普通模型补正。

### 5.2 一个机械义务入口，一次反馈足够多的事实

在现有 `receipts.py` 集中很小的条件义务/诊断入口，供 `proposal_schema` 的 description、`prompts.build_prompt` 动态材料和最终 gate 共用。handoff 只在 P2/P3/P4 approve 生效；字符按 Python `len`，行数按非空 splitlines。绑定类别展示来自当前 lock，不能与模型提示另存一套推测。

结构错误复用同一个 jsonschema 实现及禁止外部 schema refs 的约束；返回稳定 code、字段 path、实际/允许计数。安全前置失败时不继续执行依赖它的取证；坏容器中的未知部分明确未检查。对独立、可安全判定的错误尽量合并，并保留总数/省略信息。保留原数组序号，不先删除坏锚点使其他索引移动；最终仍对新 proposal 完整验证。

当前 `_schema_check` 已有 path/validator 列表，`_receive` 却只回一句泛化错误；先贯通现有细目，再补坐标/绑定/handoff 计数，不为此增加通用证据服务、proposal 文件写权限或第二套宽松预检。字段缺失、非法选择、真实固定字节漂移分层报告。工具只能记录从真实字节取得的 hash/quote，不能钳制末行、改 offset 为 0、搜索相似原文替换引用或自动复制 host 证据充作 implementation 判断。

### 5.3 对“取消累计两次 cap”的最终判断

**refine 后推荐采用。** 只对上表普通模型记录补正，删除 `_recover_call` 中 `receipt_repairs > retry_limit` 的额外停止条件；`receipt_repairs/invalid_receipts/calls/usage` 继续累计。不要把 `retry_limit` 整个删掉，它仍服务既有 CLI 策略。实际上限用 `call_limit`，默认 32，但必须尊重用户配置和已授权 additions；不是另给补正 32 次。

现有 `call_exhaustion` 必须原样保留：尚未用裁决/签收时，normal worker 预留后续 review+裁决+签收 3 次，normal reviewer 预留 2 次；adjudicate 留 signoff，已使用的裁决/签收不能再来一次。32 默认下 worker 已用 29、reviewer 已用 30 会停止普通派发。补正预算不能侵占这些保留调用，追加额度也不能重置单次决定。

自动补正只发生在原运行的正常循环中；新派发前现有两处 control gate 核 pause/cancel/hold/维护，未知在途调用保持 fence。不要因状态里累计 repairs>2 就清零、退款、换角色、恢复已停止历史 run，或把维护/CLI/裁决/签收转成普通补正。重复机械失败确可能比旧策略消耗更多已授权调用；这是成本取舍，尚无真实模型返工率证据。单一现有调用预算是推荐默认，不再增加按角色/错误 ID/进展评分的计数平台。

scoped recovery 继续以真实 failure/attempt、generation、当前实例、来源和批准、终止/结算证据绑定，消费前重核最新用户控制。保留 `last_recovery.request_id` 对状态先提交的去重，兄弟自然推进不使目标恢复失效；不能以任意 latest attempt 猜 P5 失败者。`repair-runtime` 继续要求未 ready/未交付、已结清、无屏障、原 rules 树与真实授权/审计点；不能利用修显示或补正策略清掉 pending audit marker。

## 6. X-09～X-11/X-13：只改观察与调度含义

**KDA。** 本轮只读复查仍是 summary/state revision=155、同一 checkpoint、ready/published=true。inode、size、mtime 相同；仅 ctime 从 `1795160724445403052` 到 `1795160724450241587`。`Store.read_summary` 已返回 source=state，却沿用 status=stale。确认的是缓存 stamp 不同，未确认 ctime 的系统原因，也不是已证实返回了旧交付内容。

最小改动是 fallback 成功后标明本次 state 样本；缓存的 stale/missing/invalid 另放 observation 的缓存说明，deep_verified 仍为 false。保持一次只读 fallback 和 revision/采样时间；如读期间检测到变化，标明变化/未知，不循环追逐“绝对最新”。不忽略 ctime，不写回 summary 修显示，不恢复已发布任务。覆盖同字节 metadata 变化、真实 revision 变化、缓存缺失/坏 JSON、并发替换及零写入。

**superseded。** `record_scan` 的时间比较已经保护 latest；保留。当前函数仍汇总全部 rows 的 anomalies，因此旧失败可出现在“当前”总数中。只过滤返回的有效 anomalies 与行动提示，原 rows/superseded IDs 留证据；CLI 展示前先标历史。`Clock._complete` 已读取 `result.anomalies`，只需随结果验证，无需再加过滤器。

**显式 stop/start。** 原 start.stdout 的 generation=9，却沿用 generation=7 的 last_scan、last_check=48 和旧 due，迟到约 24909.8 秒；代码 disabled 分支确未重置这些字段。推荐新显式代际清本代际扫描/到期起点、做一次 initial scan 后排 due，旧提醒继续用已有 history 归档。enabled 同代际 daemon 故障恢复保留原 ID/ack/claim/due；旧投递未结清不能启动新发送，旧代际不能 ack 新提醒。这个常规默认无需添加用户选项或另造时钟系统。

**轻量读数。** `summary.calls.scopes` 增加已记录的 receipt_repairs/invalid_receipts/cli_failures/check_failures/tool_failures 即可。展示以具体 reason 与真实计数为准，`framework_or_protocol` 不等于工具故障，`call_limit_or_single_decision` 不等于 32 次耗尽。读取脚本按既有 JSON 形状和 settled 取数，不修成宽松字符串/对象互转。以上均为读取/展示改动，不影响冻结、ready、published 或批准状态。

## 7. 可直接交实施的顺序、兼容与删减

| 顺序 | 负责文件/接口 | 完成标准 |
| --- | --- | --- |
| 1 | `normalization.py` 及现有 Template/Extract 事务测试 | 两种 S-06 反例不能改义落地；绑定/捕获、provider/consumer、幂等与只读保护通过。 |
| 2 | `monitor.py`、`cli.py`、相关 Store/Clock 提交边界 | 已提交请求始终可关联；busy 的本操作事实准确；副作用计数、用户控制和原去重保持。 |
| 3 | `runner.accept_result`、`prompts.receipt_record` | P2 revise 检真实候选；approve 相等/只读门禁不丢；历史不重算。 |
| 4 | `receipts.py`、`reading.py`、`agent.py`、`prompts.py`、`runner._receive/_recover_call`、现有说明 | 同一义务→准确批量反馈→原角色重交；明确三轴边界和传输原因后，再取消普通回执的额外累计 cap。 |
| 5 | `Store.read_summary/summary`、`monitor.record_scan`、CLI 展示、`Clock.start` | 缓存/本次样本/旧样本/新代际分清；没有运行授权副作用。 |
| 6 | 同伴明确的正文、规则、维护清单 | 定点纠正文义后记录当前实际字节与有限审阅范围；旧 publication receipt/proof 不改，不伪称新 P6。 |

保留的现有修正包括：receipt 真实取证、failed-node-recovery 的目标/反馈/去重、tools-only 的保进度与真实 audit marker、单写者和最后控制门、只读 reviewer、P5 来源/健康兄弟、无覆盖原子发布、monitor 最新观察与 claim 保护。它们是新修复的边界，不是要被重写的框架。

可删除/不新增：公式 casefold 即同一身份的推断；全命令统一 false；按非零退出整命令重试；副本式义务数字表；第二套 validator；额外普通补正次数平台；多处 superseded 过滤；强制反复 status --full/复制全历史；新缓存服务、通用控制日志、AST 证据定位器和全量重提取。不能以精简删除真实引用、来源限制、作者权限或必要独立审阅。

旧格式只增加可选诊断/观察字段，历史 receipt/request/check 不迁移。新 prompt/docs/Template 规则只进入新 run 快照；对另一个符合条件且已获工具续接授权的未完成 run，仍从其原 rules 树只加入允许的 Extract Python 修复。当前十二项全部已交付、monitor 已停止，任何本轮显示/策略修复都不触发它们的 resume、repair-runtime 或 republish。

本报告范围没有必须另问用户的例行决策。推荐默认是有限 scope 预算下的普通自动补正、传输异常沿既有有限策略、新显式监控代际新起点；是否实施产品修改仍由主代理依当前授权统一处理。若用户另设更低成本上限，使用一个明确上限即可，不加策略平台。

## 8. 实际验证与未验证

- 三份本轮离线复现脚本：S-06 的公共 API、临时落地及真实 accept_result；控制三处事务故障注入；五类回执接收分类。均零真实模型/GPU、零正式运行/产物写入；控制脚本 Popen 全为假对象。
- 两份真实 P2 工作目录只读 check：换为对应 worker plan 后均 exit0；原回执、check 与账目保持。KDA 的只读 state/summary/stat/fallback 复查如第 6 节。
- **18 项既有定向回归通过，27.083 秒**：安全规范化/局部捕获、只读 provider/原稿、nonce/机器字段、正常补正/真实 revise-blocked、P2 approve 不能改计划、五项调用预算、两项局部恢复/用户控制、两项 native 通道/终结身份。环境 `PATH=/opt/venv/python3.13.12/bin:/usr/bin:/bin`、`PYTHONDONTWRITEBYTECODE=1`、`PYTHONPATH=.:schema_template`，显式断言真实 claude/codex 不在 PATH。
- 核对 BASELINE 的 **319 个保护文件 SHA 全匹配**。当前修改只有审计目录中的新报告/证据；旧用户改动未动。
- 未执行新方案实施后的测试、真实 reviewer 漏检率/返工率测试、33 稿名称腐化调查、GPU、ctime 原因调查或新发布。现有 18 项通过不能抵消已复现的 S-06/X-08/X-15，也不能冒充建议已修好。

机器去向与紧凑证据摘要见 [CORRECTNESS_DECISIONS.json](CORRECTNESS_DECISIONS.json)。
