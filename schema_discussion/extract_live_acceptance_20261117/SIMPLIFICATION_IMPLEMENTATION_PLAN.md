# Extract 精简与两轮实施整合方案

2026-11-18。只做评估；本轮仅新增本文，没有修改产品、历史报告、冻结 run、其他仓库或全局配置，没有运行模型、提醒投递或提取流程。依据当前 Git diff、`ATTEMPT_REPORT.md`、`TOKEN_DEFAULT_PLAN.md`、`NO_TIMEOUT_SUPERVISION_PLAN.md`、`LIVE_FIX_REVIEW_PLAN.md` 和当前源码。第三份评估确认的维护 worker 缺口列于下文，本轮没有重复探针。

## 1. 结论与对初评的收紧

两份初评没有政策上的实质冲突：Claude 默认请求 **128000** 单响应输出；删除 Extract 自设时间截止；保留次数、角色和真实验收；由 supervisor 根据证据判断 hang。需要收紧的是实现范围，不能用去 timeout 的机会换来更多线程、公共 subprocess 框架、状态协议和旧运行兼容。

推荐采用方案 B，并作以下精简：

1. `budget.py` 整体删除。仅把现有次数资格搬入 `stages.py`，calls-only 的幂等追加放入已有控制/恢复路径。不保留 `timeout=None`、无限大数、软期限、退出保护时间或新名字的时间余额。
2. 输出值只在 `ClaudeBackend.environment()` 定义并设置一次。删除可选参数的跨模块透传和重复身份字段。输出环境仍进入现有 `configuration_digest`；冻结执行器已经包含默认常量。
3. `/proc` 替代 `ps`，并删除 `wakeup.py` 自己的第二套进程身份实现及 Unix fallback。日志直接写普通文件，删除 drain 线程。继续使用现有 gate、launch 和 session fence，不另起日志服务、进程监管服务或通用任务系统。
4. **删除 monitor 额外的自动恢复功能。** 后台 clock 只观察和提醒；被唤醒的 supervisor 判断后使用原 `resume` / `resume --scope`。保留 Runner 内已有有限错误重试和回执修复。这是明确的功能删减：`--auto-restart-limit` 默认关闭，上轮为 0；不是删掉全部恢复。它直接移除 clock 的深度核验/启动阻塞点，比为此再造异步恢复执行体更小。
5. queue 仅需要 clock 内一份在途 `Popen` 和原 reminder attempt 的事实记录。不能因为投递久未结束再发一份，不能建通用 job 状态机。supervisor 锁立即尝试，忙即报告或下轮重试。
6. cancel 直接 KILL 已核准 owned 后代，非阻塞报告 pending/confirmed；pause/hold 继续在正常边界停。KILL 由明确控制触发，不由耗时触发。保留 gate 锚的后代回收证明，不能一起先杀掉锚再把“看不到子进程”当作结清。
7. **排除 Graphviz。** `schema_template/tools/pipeline/render.py:184` 的 PNG timeout 不在 Extract 常规提取/检查链路，不纳入本次改造。`schema_template/tools/rules/values.py:494` 的 Git timeout 在实际检查链路内，纳入删除。不要扩成全仓库、GPU 或测试运行器的 timeout 重写。
8. TOKEN 初评提出的真实 128000 短探针不进入本次最低验收。当前离线证据足以确认 CLI 请求值；服务端接受能力仍未知，不能把它当作已验证事实或为了验证启动真实模型。

无需重新裁决是否去时间预算。最终讨论只需明确看到 cancel 的刷盘取舍和 monitor 自动恢复删减；本文没有实施这些行为变化。

## 2. 保留、删除、重写

以下路径除明确标注外，均相对 `schema_extract/tools/`。

| 对象 | 处理 | 原因及边界 |
| --- | --- | --- |
| `budget.py` | 删除整个文件 | 时间公式、review/signoff 秒数预留、check 120 秒、补时与 deadline 全部消失。保留 calls 所需几段纯判断，不重建同名框架。 |
| `runner.py` 的派发、检查、接收与 orphan 逻辑 | 删除全部余额分支，保留原语义处理 | 慢检查通过后正常接收 complete/approve；P5 merge、裁决候选与规范化不因 elapsed 改按 continue；活 orphan 不因年龄被终止。 |
| `stages.py`、`store.py` 的 scope/state/attempt | 删除 deadline、`timeout_seconds`、`*_reserved_seconds`；保留 started/elapsed/counts | `max_calls=32`、revision limit、两次修复/失败重试和一次裁决/签收仍约束循环。检查处理和账目不是停止条件。 |
| `recovery.py`、`cli.py`、`runtime.py` 的追加 | 改为 calls-only | 保留 `--add-calls`、launch ID 幂等和已用账目；删除 `--add-seconds`、seconds 对象及 `budget_extension.v1`。不接受旧参数、不转换旧预算对象。 |
| `backends.py` 与所有 `max_output_tokens` 透传 | 常量化并删除透传 | 只保留 Claude env=128000 与既有配置摘要；`environment(home, effort)` 恢复两参数。Codex 不收到 Claude env，仍需显式 model/effort。 |
| `agent.py` 版本核验 | 重写等待方式，保留核验 | 版本探测必须有同一次 attempt 的 owned 身份并能取消；不能把 `check_output(timeout=15)` 简单变成无限同步等，也不能为简化跳过版本与精确会话校验。优先扩用原 gate 的 setup，不做通用子任务协议。 |
| `agent.py` 日志与退出 | 文件直写、非阻塞退出观察 | 删除 `drain`、`threads`、`pipe_errors` 及 join 等待。真实关闭后刷新/封存日志；原生终结、exit、后代结清和 schema/nonce 校验继续全部执行。 |
| `processes.py`、`wakeup.py` 身份 | 合并为一套 Linux boot/start token | 删除 `ps` subprocess、lstart 依赖、`PROC_IDENTITIES` 与第二份 `/proc` 解析。`identity_key`、恢复、clock、launcher 的消费者一起更新；不做旧格式 fallback。 |
| `terminate_owned` 与 cancel | 一次可重入的核准、发信号、观察 | 删除 TERM→等待→KILL 的时间升级；未结继续 fence。巡检读取失败不能落入清理路径杀健康模型。 |
| `monitor.py` 额外自动恢复 | 删除 | 删除 `configure_auto_restart`、`authorized_restarts`、policy/cursor/used、自动授权及 clock-binding 分支；保留 `request_restart` 的人工 scope 路径、`retries`、`starts`、failure identity 和消费时复核。 |
| `wakeup.py` | clock 只做观察、心跳、一份 queue 投递 | 删除三处 `authorized_restarts` 调用和 recovery 展示。无须新增恢复 helper/线程/守护进程。保留 thread/generation/reminder/ack/scan 及迟到结果不可覆盖新控制的约束。 |
| `Store.initialize/supervisor` | 非阻塞取锁 | 忙是明确控制争用，不是模型失败。未持久提交不能回复“cancel 已登记”或“ack 已完成”；不按锁年龄破锁。 |
| `runner.py` 的 fake transport 适配 | 删除 `inspect.signature` / `accepts_cancel` 分支 | 产品 transport 一律接受同一 `cancel_event` API；既有测试替身同步修改。无需维持旧 fake API。 |
| `observation.py` SQLite | 明确零 busy wait | `timeout=0` 是关闭锁等待；锁忙走现有可选索引 fallback，不添重试期限，不伪造 usage。 |
| Template Git 检查 | 删除 10 秒及 TimeoutExpired 分支 | 保留 Git 实际结果。不要另造全仓 subprocess helper。Runner 同步检查的取消交付由现有持久控制及独立进程身份处理，不承诺任意 Python 指令点可温和抢占。 |
| `audit.py`、summary、prompt 动态上下文 | 删除预算投影；保留统计 | 不留 `budget_policy=wall_clock_deadline`、`waiting_consumes_budget`、deadline/seconds 扩展。原生 usage、处理耗时、并发与未知口径、哈希继续保留。 |

输出配置删除面具体为 `backends.py` 的 `validate_max_output_tokens` 和参数；`cli.py` 两个入口；`runner.py` 默认/校验；`agent.py` binding、invocation、result 比较和 env 调用；`store.py` 配置/attempt/summary；`runtime.py` admission；`recovery.py` invocation/ledger；`receipts.py` identity；`audit.py`、`monitor.py`、`review_eval.py` 展示与传参。**不随此删除 `recovery.verified_result` 的 `configuration_digest` 校验。** 若 invocation 保留一个实际请求值，只作诊断，不回流为身份、配置接口或模型必填回执；最小默认可完全依靠 env digest 和冻结源码，不必加诊断字段。

## 3. 哪些真实修复仍有必要

| 前轮修复 | 建议 |
| --- | --- |
| LIVE01 失效 `EXECUTOR_ALIGNMENT.md` 引用 | 保留删除与现有协议链接；不恢复该文件、不留“曾删除”的活跃提示。 |
| LIVE05 输出额度参数 | 按上节替换为单一默认；删除相关参数接受/拒绝矩阵和重复字段断言。修复评估确认其为唯一合法 nullable 标量身份，删除输出身份字段后可删 `receipt_schema` 新增的 null 类型分支；`reopen` 的 object/null 与正式 YAML nullable IO 不是同一契约，保留。 |
| LIVE06 新登记前校验配置 | 保留，仍有 model/effort/backend/次数/可执行文件等非法输入。规范化函数只有一份；不为避开一次重复调用增设缓存或 trusted 参数。 |
| LIVE07 worker 预览通过但机械格式 warning 返回失败 | 保留 P1–P5 worker 预览退出码；raw 原稿、pending corrections、reviewer/P6 严格边界不变。 |
| LIVE08 Python/Markdown 被误当 JSON | 保留仅 `.json` 声明解析及精确键名提示；坏 JSON、矛盾声明和真实来源仍检查。不要为了缩提示删除实际修复。 |
| LIVE09 原生 session UUID 先于提案采纳登记、启动前失败结清 | 保留真实成功和精确 scope 的登记时机；新版本探测改造必须保住“模型未启动”与“探测进程已退出”分别证明。不能把已发信号、函数返回或输出文件存在当终止证明。 |
| LIVE10–12 正式恢复、孤儿结清 | 保留现有恢复语义与证据；这些记录没有要求添加新的兼容功能。没有原生成功的结果不能通过恢复变成 approve。 |
| LIVE13 维护回执恢复及索引定位 | 保留，并补维护 worker 保存新稿后的恢复缺口，见下段。不能只保住 reviewer 无改稿用例。 |
| LIVE15 删除任意 12 行上限 | 保留。文件边界、offset、精确摘录、摘要和独审仍有效；不能改成另一任意范围上限，也不重新写旧证据。 |

`LIVE_FIX_REVIEW_PLAN.md` 已离线复现：维护 worker 的合法新稿先被保存，随后回执被拒；`task.draft_checkpoint` 已变，但 `_resume_maintenance_receipts` 仍拿旧 attempt 的 `base_checkpoint` 当当前草稿校验，触发 `StaleResult`。最小修复是核对同 task/step 的最近已结清 attempt，以及现有账目 `result_checkpoint` 与当前 task 草稿的对应链条，复用现有 checkpoint/repair 核验；原 base/dependency 仍证明这次调用从哪里开始。**不得把旧 attempt 改成新 checkpoint、跳过候选核验、选择任意较新稿或采纳旧 approve。** 不复制出第二套恢复检查器。此项由第二轮 `runner.py` owner 实施，整合代理只复核和给定位，不同时编辑。

正常调用的真实身份、冻结执行器、manifest、原稿/规范化稿、checkpoint、独立 reviewer、P5 当前依赖和 P6 发布屏障都是功能本体。不能以“重复检查”为名删去不同信任边界上的检查；只删同一事实的纯透传与失效政策。原生成功先登记 UUID 与“产物已被批准”仍是两个不同事实。

## 4. 最小运行与监督接口

实施前固定以下接口；中间分支可暂不运行，不用兼容参数让半成品看似可用。

- `AgentCLI.run(attempt=..., prompt=..., schema=..., on_process=..., cancel_event=...)`：没有时间参数；真实 backend、exit、native result、usage、session 与 termination 仍返回。
- `Store.prepare_attempt(..., call=...)`：没有 timeout/output 参数；固定身份由原 manifest/config/env digest 承担。
- 次数判断只返回“是否可派/原因”，不返回假 timeout。`call_limit`、`reserved_calls` 使用当前 scope 原计数；calls 追加按既有 launch ID 一次消费，不退款。
- 进程身份在 `processes.py` 一处定义，以 boot/start token 判断；读不到是 unknown，PID 消失与 PID 复用分开；对部分 `/proc` 扫描失败不得用“表里没有”证明全体死亡。
- 发信号接口保留原 `confirmed/actions/survivors/reason`，没有 grace；发 KILL 与确认退出分离。gate 与原 Runner/调用线程尚在时不许重派同 scope。
- supervisor 锁忙的结果必须可区分“未提交”和“已提交、等待消费”。Runner 下轮处理，clock 下轮观察；交互命令不伪装成功。只需要一个具体的 busy 错误/返回约定，不建设通用重试层。

queue 的一次投递状态使用原 reminder 的 `attempts`：启动后保存其身份、stdout/stderr 文件和绑定；tick 只 poll，在原 attempt 中补真实 exit/queue UUID。`_next_send_at()` 在最后一次 **sending 尚未终结** 时返回 None；不能继续沿现有 `started_at + RETRY_SECONDS` 发下一次。真实失败可沿原有限补送规则，已 queued 未 ack 的提醒节奏也继续保留。owner 死亡后，先核对原 queue 身份与结果；仍活或未知不重复发送。

`monitor stop` 先持久停用 generation，再请求原 clock/queue 退出，返回实情；不等固定秒数、不因“到点未退”升级。stop 只停止监督，不 cancel Extract。新的 clock 不得与原未结投递争用。queue 自己挂住时，clock 仍能观察，但同一损坏通道未必能通知主线程；不为此新增第二投递系统。

同步 Python 检查、Git 和文件 I/O 的任意点抢占不属于这次最小实现。cancel 应先通过独立控制入口落盘，再按现有 Runner/attempt 身份交付强制终止；不能等拿到 Runner 写锁才报告控制。如果尚未交付/结清，公开 pending，不宣称任意阻塞都可温和退出，也不重新加保护 timeout。

如用户明确要求继续保留 monitor 无需 ack 的额外自动恢复，才恢复讨论一个局部 owned 执行体与旧 `retries/starts` 的结合；这是备选，不是本方案默认，不引入 force 命令或恢复调度器。

## 5. 活跃文档、注释和测试删改清单

| 路径 | 精简动作 |
| --- | --- |
| `schema_extract/README.md` 参数、冻结、角色、恢复、monitor 节 | 删除三类秒数、逐次输出选项、补时、等待消耗额度和 `--auto-restart-limit`。一处说明 Claude 请求 128000、实际服务边界未知；保留 Codex 示例。cancel 写清直接 KILL 与 pending，pause 写清正常边界。 |
| `schema_extract/tools/README.md` 恢复、backend、usage 节 | 删除 `agent.run(..., timeout)`、签收时间预留和“超额候选不获批准”；用次数和账目描述，不再多段重复“保留原预算”。 |
| `EXTRACT_SCHEMA.md:20`、`schema_extract/REVIEW.md:97` | 将执行预算职责改为调用次数/明确控制；不动设计容量预算、独审责任与 schema 语义。 |
| `schema_extract/prompt/operations/{normal_worker,retry,recovery,receipt_repair,maintenance,reopen,monitor_handoff,wakeup}.md` | 删除原/剩余时间预算要求；一次说清保留身份、权限、草稿和已用次数。hang 交 supervisor 判断，不教 worker 因快到时而提前收尾。 |
| `schema_extract/tools/prompts.py:280` | 删除动态 `budget.deadline/call_seconds`；只给任务所需的已有次数/角色信息，不新造预算展示协议。 |
| `schema_extract/prompt/operations/{check,response}.md`、`prompt/phases/01_contract.md` | 保留 LIVE07/08/15 的当前行为说明；需要时压成短句，不删精确边界。 |
| 根 `README.md`、`schema_template/README.md`、`schema_template/examples/README.md` | 保留前轮正确的链接修复；未发现应重写的时间策略，不为凑统一措辞大改导航。 |
| `agent.py/processes.py/wakeup.py` | 删已移除 ps/drain/TERM 宽限的注释、Unix fallback 和“older POSIX”分支。Linux gate 已依赖 prctl，不继续维护未验证可移植路径。 |
| `runner.py:_reconcile_orphans`、`recovery.py:elapsed_on_disk`、`stages.py` scope 恢复注释 | 删除接管旧 timeout、固定 deadline、不重置预算等失效表述；只解释当前 invariant。 |
| `wakeup.py:_next_send_at`、`observation.py` 顶部、相关 CLI help | 区分重试次数、观察范围/内存界限和时间停止；不把所有 bounded 字样机械删除。 |

活跃文件没有必要加入 LIVE 编号、补丁过程、旧值/新值对照或“为修复上次……”段落。`patch` 本身是正式维护/裁决动作，不是历史注释，不能按词删除。`historical` 的原稿或共享来源基线、已用决定和不可重授的批准是实质语义，也不能按词删除。`schema_discussion/**`、`schema_extract/archive/**`、`.schema_extract/**` 及其他 archive 是证据，保持原样；`schema_docs` 固定字段、YAML 长度规范、Template 文本/图复杂度规则不动。

测试精简以行为为单位：

- `test_recovery_budget.py` 删除纯 `_remaining/review_seconds/check_timeout/deadline` 公式用例；迁移其中调用保留、calls-only 幂等、fence/一次性决定不重置、P5 健康兄弟的用例到现有恢复/阶段测试，然后删除旧预算测试文件，不额外建大套件。`test_current_returns.py::test_checker_time_after_normalization_cannot_spend_reserved_review_time` 改为慢检查仍按实际结果接收。
- `test_cli_recovery.py::test_output_budget_is_frozen_passed_to_cli_and_rejected_for_codex`、`test_backends_runtime.py::test_output_budget_is_explicit_fingerprinted_and_bound_to_the_session` 以及 `test_review_eval.py/test_stage_flow.py` 中可选额度透传断言撤掉。保留前者独立的非法配置不占登记断言；默认输出只需一处环境隔离/摘要测试和两个入口的调用契约核验。
- `test_backends_runtime.py` 中靠 `timeout=0.8` 触发结束的用例改用明确 cancel 事件；保留 detached descendant、原生 exit 0 仍有后代、PID 复用、真实 fence 反例。无需为多个旧秒数复制同一取消测试。
- `test_current_recovery.py/test_scoped_recovery.py` 中 monitor 自动额度/clock 自动恢复专用部分删除，人工 scope 恢复、实际失败绑定、健康兄弟与 pause/cancel 覆盖继续保留；`test_wakeup_retry.py` 不再 mock 已删除的 `authorized_restarts`。
- 保留 `test_maintenance_receipts.py` 的 reviewer 恢复、semantic blocked/一次性决定/fence 反例，并加一个“worker 新稿已保存、回执被拒”用例；不要复制整份 stage-flow fixture。
- `test_proposals.py/test_source_gaps.py/test_prompt_materials.py` 保留 JSON/非 JSON、真实越界与超过 12 行合法证据、失效链接、worker preview 等必要行为。已有正反例足够，不为文案变更另铺快照测试。
- 测试 harness 的短等待/失败保护与产品时间政策分开，测试用虚拟时钟和事件同步；不为了 literal `timeout` 零命中改写所有测试、第三方工具或 GPU 验证。

## 6. 四位新实施代理，两轮明确文件所有权

四位均按用户要求新建 xhigh 代理；同时两位。先约定上节接口，再分别编辑。每轮同一文件只有一个 owner；轮间是明确接力，不保留跨轮兼容代码。第一轮不宣布全系统可用，第二轮完成接口和定向验收后才交付。

| 轮次 / 代理 | 独占文件与工作 | 对另一代理的交付 |
| --- | --- | --- |
| 第一轮 A：调用与进程 | `agent.py`、`processes.py`、`backends.py`、`observation.py`、`review_eval.py`；`schema_template/tools/rules/values.py`；`test_backends_runtime.py/test_usage_observation.py/test_review_eval.py`。实现默认输出、无调用/版本 timeout、Linux 身份、文件日志、显式 KILL/真实 fence、零 SQLite wait。 | 无时间参数的 AgentCLI、统一进程身份和非阻塞终止接口；review-eval 前台取消约定。此任务承担实际进程边界，不只改一个常量。 |
| 第一轮 B：监督与控制 | `wakeup.py`、`monitor.py`、`common.py`；本轮 `store.py` 的锁/监督 ledger 及 `cli.py` 的 monitor/cancel 控制；`test_wakeup_retry.py/test_monitor_evidence.py`。删除 monitor 自动恢复，单在途 queue，非阻塞控制提交/stop，准确 pending。 | 给第二轮交付 lock-busy 结果、queue 记录、人工 `request_restart` 接口及剩余旧预算调用点列表。绝不并发编辑 A 的 `processes.py`；直接消费已约定接口。 |
| 第二轮 C：主流程与恢复 | `runner.py`、`store.py`、`stages.py`、`recovery.py`、删除 `budget.py`；`test_stage_flow.py/test_recovery_budget.py/test_current_recovery.py/test_current_returns.py/test_scoped_recovery.py`。统一删时间、保次数、慢检查正常采纳、活 orphan 不被年龄杀、calls-only 追加，并实施维护 worker 恢复链条修复。 | 给 D 确定的 config/state/count/status/API；D 对 runner 的正确性意见由 C 实施。第一轮基础运行/监督文件如需集成修正，也明确由 C 接管，D 只报问题。 |
| 第二轮 D：正确性、外层接口与精简 | `cli.py`、`runtime.py`、`audit.py`、`receipts.py`、`reading.py`、`comparators.py`、`maintenance.py`、`snapshots.py`、`prompts.py`；活跃 README/MD/prompt；`test_cli_recovery.py/test_proposals.py/test_source_gaps.py/test_prompt_materials.py/test_maintenance_receipts.py`。删输出透传和旧能力/文案，落实其所持文件的真实修复，复核 C 的身份/恢复边界。 | 最终用户行为、静态删除核对与最小验证记录；不同时编辑 C 的 runner/store/recovery。 |

`store.py` 和 `cli.py` 的轮间接力必须等第一轮落地后再开始。第二轮 C 在动手前接收第三份修复评估，把涉及 `runner.py/recovery.py` 的必要修复纳入自身任务；D 持有的 `maintenance.py` 只按双方约定接口修改。第一轮的测试若因第二轮 API 有必要改动，由 C 持有，D 不同时改写。分工不会把“只有输出常量”的小任务分给一人、把所有进程/流程/监督交给另一人。

## 7. 最低离线验收

不启动实际模型、queue、后台 monitor 或旧 frozen run；不跑新全流程。只运行受影响的现有定向测试，加少量必要反例，测试通过不声明真实监督接管或完整提取已完成。

1. **入口与默认。** 两个入口都不接受旧时间/输出参数；Claude 默认 env=128000、主机输出/thinking 不漂移；Codex fake backend 可调用且不收到 Claude 配置。原生假 API 的请求捕获至多一组，不扩成能力数据库或真实服务探测。
2. **流程不按耗时改判。** 虚拟时钟越过原调用/检查截止，合法 complete/approve、P5 merge 与裁决候选仍按内容处理；调用次数和 reviewer/一次性决定仍执行。原有独立 session 测试即可，不新增完整提取试跑。
3. **进程与取消。** 短合成进程覆盖版本未返回、日志落盘、主 CLI 退出且后代还在、cancel KILL、`/proc` 失败/复用。cancel 先有持久 request，只有真实退出和 gate 结清才 confirmed；healthy observer error 不触发 kill。
4. **监督独立。** 假 Popen 长期 sending、supervisor 锁忙、stop/ack 与迟到结果交错：tick 能继续，未结束不重发，未提交不报成功；monitor 自动恢复函数/选项已消失，而人工 scope 恢复继续可用。
5. **必要回归。** 维护 worker 新稿恢复、reviewer scope UUID、启动前失败不造 fence、JSON/非 JSON 与真实证据边界、worker preview 原稿保留；失败/取消/恢复 usage 按 attempt 高水位去重，calls 追加不重复消费。
6. **一次静态收口。** 只搜活跃 Extract 和实际 Template Git 路径的旧预算参数、deadline、timeout/grace、输出透传、auto_restart；逐项解释剩余的轮询/提醒/elapsed/零 busy wait。复核 frozen 清单/活跃链接及变更文件编译。没有新变化或失败，不再扩大测试轮次。

最终需知会用户：直接 KILL 放弃 TERM 的最后刷盘机会，已落盘稿件/日志仍保留，未落盘内容不能承诺恢复；monitor 不再替 supervisor 自动重启，正常 Runner 有限重试不变；提醒通道自身挂住时保持观察也不保证能用同一通道通知。除此以外，不存在必须重新讨论的时间预算策略冲突。
