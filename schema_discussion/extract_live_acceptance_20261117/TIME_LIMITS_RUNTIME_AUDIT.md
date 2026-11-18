# Extract 调用、工具和监督时间限制只读审计

审计日期：2026-11-18。对象是当前工作区生产代码，不是各历史 run 冻结代码的逐字重审。只读取本地代码、活跃说明和既有 LIVE04/LIVE05/ATTEMPT_REPORT 捕获记录；没有启动提取、模型请求、提醒投递或修改产品/运行配置。本文是本次唯一新增文件。

整算子、阶段/P5 节点、审阅/裁决预留、检查处理和 review-eval 的完整预算推导见同目录 `TIME_LIMITS_FLOW_AUDIT.md`。本篇负责执行这些预算的底层机制、附加小超时、监控和输出参数。数值均以秒计，另有说明除外。

## 核心判断

1. `--max-output-tokens` 是 **Claude 单个原生模型响应的可选输出上限覆盖**，不是整个 CLI 调用、阶段或全场的累计 token 限额。Extract 没有“累计 output_tokens 达到 N 就停止”的逻辑。
2. 从默认 32000 提到 65536 是可选调优，目的是让一次响应有更大的连续生成空间；不是修复丢配置，不是节省总 token 的预算开关，也没有证据证明能加速或保证完成。CLI 曾自行处理输出触顶并续写；LIVE04 两个 attempt 真正终止于 1800 秒调用超时。
3. 监控的 300 秒停滞提示、75 秒心跳陈旧、1800 秒 ack 提示只标记状态，不终止模型。不能将这些节奏当成任务预算。
4. 最值得检查的非显性风险是 `ps` 的 **10 秒超时会在 AgentCLI 活跃调用内沿异常路径触发进程清理**；它不是单纯观察失败。另有版本查询 15 秒、局部 Git 检查 10 秒以及退出/日志排空 3 秒等可靠性等待。下面明确区分其影响。
5. 当前代码也有无时间上限的同步等待：来源 Git、私有 Git 快照、supervisor 锁、Python 检查/发布过程等。设置 `--call-seconds` 并没有使整条流水线具备精确可抢占的总截止。

上述第 4 项是本次**静态代码发现**；未发现既有记录证明试跑命中过该 `ps` 超时，也不据此解释 ATTEMPT_REPORT 中原因未查明的 Runner 消失。

## A. 会截断 agent 工作或使有效工作不能继续的时间条件

| 项目 | 当前值 / 是否可配 | 到期动作及实际范围 | 出处 |
| --- | --- | --- | --- |
| AgentCLI 整次调用截止 | `timeout` 必须有限且 >0；Runner 传入预算计算结果。产品 `call_seconds` 默认 1800，历史试跑后期 7200；不支持 None/无限 | 从 CLI 设置前开始 monotonic 计时；放行 launch gate 前、运行循环中检查。到期记录 `timeout` 并进入 owned-process 清理。包括设置和版本查询等开销，非纯模型执行时间。清理/解析可能使实际返回晚于截止 | `agent.py:220,236,245,354,371,383`；`runner.py:44,1426,1438` |
| 每次 CLI `--version` | `min(15, timeout)`，15 固定，无单独 CLI 参数 | `subprocess.check_output` 超时转 `cli_error`；发生在模型主调用启动前。不会给正在写作的模型设置额外 15 秒预算，但会让本次 attempt 启动失败并使用已登记次数 | `agent.py:285-287,407-409` |
| 进程表 `ps -axo ...` | 每次最多 10，硬编码，无公共配置 | `process_table()` 在 agent 启动、每次后代扫描、终止/恢复以及 monitor 中共用。**活跃 AgentCLI 的 `record()` 抛 TimeoutExpired 时，进入 finally，尝试终止本次 CLI/后代，再记 cli_error；清理不能确认则保持 fence。** 在 monitor 的只读路径中捕获为 unknown/observer_error，不因此杀模型 | `processes.py:44-48,131-132`；`agent.py:307,319-324,348,374-386,407`；`monitor.py:31-35,208` |
| 检查器局部 Git 元数据命令 | 每条命令 10，硬编码，无 Extract CLI 参数 | Template `local_repository_errors` 调用的 `git config --get remote.origin.url`、`rev-parse`、`ls-tree` 超时转 ToolError。Extract snapshot checker 把 ToolError 记 exit 2，可阻断检查/发布；不是给模型整段工作设 10 秒上限 | `schema_template/tools/rules/values.py:491-498,521,530,546`；`schema_template/tools/schema.py:68-70`；`artifacts.py:706,721,726-728` |
| 完成后的检查处理额度 | 最多 120，另受主预算和预留扣减；详见流程审计 | 同步 Python 检查并非运行到 120 秒被抢占；Runner 在边界拒收超额的 complete/approve 并保留稿件。这项会造成“模型已经完成，机器处理超额而未获采纳” | `budget.py:61-62`；`runner.py:554-555,737-740`；候选准备 `store.py:765-771` |

普通 `cli_error`/`timeout` 是否继续重试由 Runner 原有剩余额度/重试规则决定；本表不把所有超时都等同于整 run 永久失败。其入口在 `runner.py:849-852`。已确认结束与回执可采纳仍是不同条件。

局部 Git 检查的触发条件也有限：文档已知 locator 字段出现 repository 引用，而且被检查 YAML 的父目录链中确有 `.git` 时，才查询本地 origin；仓库名称/origin 与引用匹配后才解析本地 revision 和路径。找不到上下文仓库时跳过，没有网络 fetch/clone。因此不能理解成“每次机器检查都联网或固定等 10 秒”。见 `schema_template/tools/rules/values.py:505-550`。

## B. 底层可靠性等待和回收

| 项目 | 当前值 / 是否可配 | 到期动作及影响 | 出处 |
| --- | --- | --- | --- |
| owned invocation 的 TERM→KILL 回收宽限 | `grace=1.0`，每个信号阶段各 1；Python 函数参数可改，所有生产调用沿用默认，无公共 CLI 参数 | 对核实归属的后代先 TERM 后 KILL；0.05 间隔重查，gate 作为归属锚不被该函数直接发信号。仍存活/归属不明则 `confirmed=false`。这是已结束/取消/超时后的清理，不是正常工作 1 秒上限 | `processes.py:165-200`；`agent.py:385`；`recovery.py:262-264` |
| AgentCLI 等待 wrapper 退出 | 3，硬编码 | `process.wait(timeout=3)` 到期仅标记终止未确认；不把未知退出改成成功 | `agent.py:388-396` |
| stdout/stderr 排空线程 join | 每个线程 3；两条顺序 join 最多名义 6，硬编码 | 仍存活则 `termination.confirmed=false`，后续保持 fence。发生在模型主进程已退出或清理启动之后，不是正常生成过程的静默上限 | `agent.py:326-347,397-400` |
| 正式恢复确认进程结束 | 没有独立新超时；复用上述 TERM/KILL 和 `ps` 等待 | 归属或结清证据失败就保持 `confirmed=false`；没有按时间自动解除 fence | `recovery.py:250-285` |
| 提醒投递 `codex queue` | 30，硬编码，无 `monitor start` 时长选项 | 从 `Popen` 返回后开始 monotonic 计时，communicate 反复等待；超时标 uncertain，清理投递子进程，之后按提醒重试规则处理。只影响提醒投递，不是 stage backend 的调用截止 | `wakeup.py:30,444-458,520-521` |
| 提醒投递进程回收 | TERM 后 wait 1，必要时 KILL 后 wait 1，硬编码 | 向该投递进程组发信号，只有父进程的有限 wait，不等脱离进程的 pipe EOF；不追杀其他进程组。结束投递的资源等待 | `wakeup.py:460-479` |
| 显式 `monitor stop` 的时钟进程退出 | TERM 2，KILL 1，0.05 观察间隔，硬编码 | 只停止 clock 自身；不 cancel 已启动 Extract。无确认时返回实际状态，非强行标记成功 | `wakeup.py:600-621`；`schema_extract/README.md:172` |
| Codex usage 索引 SQLite 锁等待 | `timeout=0.2`，硬编码 | SQLite busy/error 时走有界 UUID/日期文件查找；整体 usage observation 出错记录 unavailable/错误，不判模型失败。这是数据库争用等待，不是整条查询/文件观察的通用 0.2 秒超时 | `observation.py:73-109`；`agent.py:301-306,428-437` |

“名义 1+1 秒回收”不是整个回收调用的严格上界：每次观察会先执行最多 10 秒的 `ps`；信号/文件系统调用也在循环边界之外。版本查询和清理都使用 Python subprocess 的子进程超时机制，并非独立覆盖所有 I/O/后代的系统级绝对截止。当前保守 fence 不应因为把宽限改为提醒而解除。

## C. 仅观察节奏、退避与提醒条件

| 项目 | 当前值 / 是否可配 | 发生什么；不发生什么 | 出处 |
| --- | --- | --- | --- |
| AgentCLI 取消/退出/截止轮询 | `sleep(0.05)`，硬编码 | 在正常工作循环间让出 CPU；不是一次工作必须在 0.05 内完成 | `agent.py:362-377` |
| AgentCLI 归属进程扫描 | 每次成功扫描结束后 +0.5，硬编码 | 更新后代身份记录；单次扫描受上表 `ps` 10 秒影响，不是 0.5 秒生成截止 | `agent.py:374-376` |
| CLI/回执失败重试退避 | `min(60, 5 × 2^min(4,max(0,failures−1)))`：5、10、20、40、60…；硬编码 | 延后下一次合法重试；不终止健康调用。墙钟主预算继续流逝。重试次数限制另算 | `recovery.py:32-33`；`runner.py:790` |
| Monitor 无进展提示 | `min(300, call_seconds or 300)`；300 固定、call_seconds 间接影响 | 仅当两次观察 state_revision 相同、运行中、无 active/unsettled、Runner 确认活且持锁，标 `possible_stall`。不自动杀进程/重试；模型有在途调用时不因日志沉默命中此条件 | `monitor.py:224-243` |
| Monitor deadline/overdue 展示 | operator/scope/调用已登记截止的最小值 | 只报告超期。普通 scan 本身不执行因超期而杀模型的动作；主预算执行见 Runner/AgentCLI | `monitor.py:86-104,119-127,166-171` |
| 后台 observation | 每次完成或错误后 +30；硬编码 | 每轮最多 16 个 run，轮转；只读观察后可执行此前明确授权、符合条件的恢复。提醒未 ack 也独立继续观察 | `wakeup.py:32-33,378-429` |
| Supervisor 提醒计划 | 初始 scan #0 立即；#1/#2/#3 各在前一完成后 300；#3 完成后下一次起为 1800；硬编码 | 计时从 scan 完成开始，不是追赶绝对日程。提醒不会使提取多等 5/30 分钟，Runner 自行推进 | `wakeup.py:26-27,278-302` |
| 投递失败/不确定后的补送 | 上次 started/finished 较晚值 +120；硬编码 | 相同提醒最多 3 次投递尝试；不增加模型重试次数，不续任务预算 | `wakeup.py:28-29,127-135,504-507` |
| 投递已确认、尚未 ack 的补送 | 上次 started/finished 较晚值 +1800；硬编码 | 仍在每条提醒最多 3 次的共同计数内；入队成功不是线程已接管 | `wakeup.py:127-135` |
| Clock 心跳/空闲 tick | 最多等待 15；按最近观察/投递时间缩短，最短 0.05；硬编码 | `Event.wait` 可由停止信号唤醒，tick 后更新 heartbeat；实际 tick 的工作耗时可使心跳间隔更长 | `wakeup.py:31,495,549,587-592` |
| 投递 communicate 轮询片段 | `min(0.25, max(0.01, remaining))`；硬编码 | 为 30 秒总投递截止与停止信号提供响应机会，不是单独的 0.25 秒投递失败限制 | `wakeup.py:451-459` |
| 心跳陈旧提示 | `age > 15×3+30 = 75`；硬编码 | 标 `heartbeat_stale` / needs_attention，不杀 clock、不杀模型，也不自动认领仍活的旧锁 | `wakeup.py:628-635,669-670` |
| 已 ack 但 scan 未完成提示 | ack 后 ≥1800；硬编码 | 标 `acknowledged_unfinished`；ack 后不再补送同一提醒，没有过时后自动完成/退款 | `wakeup.py:127-130,654-660` |
| 已入队但无人 ack 提示 | 首次 queued 的 finished_at 后 ≥1800；硬编码 | 标 `queued_without_ack`，按上述有限补送规则处理；不停止 Extract | `wakeup.py:648-668` |
| Runner 循环观察/收尾 | orphan +1；events.get 0.1；空闲/退避 wait 0.01–0.25；退出线程 join 3 | 由流程审计详列。前几项是调度节奏；join 是退出回收等待 | `runner.py:1387-1391,1490-1496,1549` |

自动恢复不是“等够某个时间就重启”：monitor 默认关闭自动恢复；其已授权路径要求具体 transient_execution、EX_TEMPFAIL 75、进程结清和原剩余额度，不能把 stall、overdue、日志静默当成授权依据。见 `recovery.py:23-29,166-194`，`schema_extract/README.md:174`。

## D. 没有数值时间上限的等待与缺省行为

| 路径 | 现状及边界 | 出处 |
| --- | --- | --- |
| SourceTree Git | `subprocess.run(...capture_output=True)` 不传 timeout；本地历史源码 `show`、`ls-tree`、解析 revision/HEAD 等调用。不存在另一个 source.py 超时层 | `inputs.py:23-28,30-47`；`snapshots.py:39-44` |
| 私有 Git snapshot | 初始化、逐文件 hash-object/update-index、write-tree/commit-tree/update-ref 都不传 timeout；不是网络 clone/fetch | `snapshots.py:48-77` |
| Python checker/normalize | Runner 直接导入冻结 Template 执行，不为整个 checker 创建有总 timeout 的 subprocess；同步 CPU、读文件、校验不可由 `check_timeout` 立即打断。内部局部 Git 10 秒见 A 表 | `artifacts.py:45-71,677-734`；`schema_template/tools/errors.py:17` 的 bounded 只处理 RecursionError |
| 模型工具 check-helper | 生成的可执行 Python 直接进 `reading.main → stage_check`，没有独立 Extract 工具时长参数；它作为 CLI 后代仍可能被外层调用超时一并回收。原生 CLI 对 Bash/Read 等工具的内部默认值未在 Extract 明确设置，不能声称为无限或推断一个固定值 | `prompts.py:224-234`；`reading.py:126-146`；`backends.py:187-201` |
| 发布/export | Python 同步复核完整 P6、来源、复制/哈希/fsync/原子 rename；没有统一独立发布 timeout，也不启动 Graphviz | `publication.py:40-65,90-110` |
| standalone | 同步 validate_resolved_file / validate_standalone_file、复制、同步、rename；没有总时间截止 | `standalone.py:28-99` |
| supervisor.lock | 初始化和控制事务用阻塞 `flock(LOCK_EX)`，无 timeout；“短事务”是代码设计意图而非强制时间上限。持锁进程阻塞可拖延监控/控制，主预算墙钟继续走 | `store.py:77-89,112-127` |
| runner/session/clock 锁 | 非阻塞 LOCK_NB，立即获得或报告 busy/held/reuse；无按年龄解除锁、TTL、定时抢占。恢复依赖进程身份与实际锁证据 | `store.py:129-143,267-280`；`agent.py:60-65`；`wakeup.py:160-204,553-563` |
| background Runner 启动/claim | Popen 后观察一次进程身份/poll，持久记录 launching/claimed；没有“启动满 N 秒则杀掉”的宽限定时器 | `monitor.py:401-457` |
| clock 启动/claim | 继承锁启动后记录身份，无启动确认总超时；starting 无 heartbeat 时也没有独立年龄限制 | `wakeup.py:190-260,567-573,630-633` |
| scan/observation claim | 原 owner 仍活或不确定时返回 busy；没有 claim TTL。需要确认旧 owner 退出才换 claim | `wakeup.py:346-350,395-402` |
| launch gate | 等待父 pipe 一字节；父端关闭则不启动，随后等主 CLI、收养后代直到 ECHILD；均无自身独立 timeout，靠父 AgentCLI/恢复清理。未结清不能假定结束 | `processes.py:272-305` |
| 归属与日志/用量读取 | 文件读写、fsync、事件解析及受字节/条数约束的 usage 观察没有统一墙钟截止；`elapsed_seconds` 是事后统计，不是每个操作的硬 timer | `agent.py:326-333,422-438`；`observation.py:28-30,73-109` |
| backend 原生网络/重试/流静默 | Extract 未显式为 Claude 配置 SDK 请求时间、Bash 默认时间或流静默 cutoff，也不设置相应环境变量；用外层调用截止包住 CLI。Codex 会完整保留所选 provider 配置，若该配置本身带原生超时会随之保留；本轮未读取私人配置，数值未知 | `backends.py:80-84,164-181,224-245` |

Template 的 Graphviz 有 60 秒渲染超时（`schema_template/tools/pipeline/render.py:178-186`），但只由显式图形 PNG 路径调用，Extract 阶段/发布/check-helper 常规路径不调用；因此不计入 Extract 正常提取预算。LIVE04 离线捕获的 25 秒、LIVE05 探针的 180 秒及 `--max-turns 1` 都是历史诊断 harness 配置，不是产品默认限制。

## E. 输出预算逐项澄清

| 问题 | 代码与既有证据支持的回答 |
| --- | --- |
| `--max-output-tokens` 究竟限制什么？ | 输入只在 Claude backend 设置 `CLAUDE_CODE_MAX_OUTPUT_TOKENS=N`，CLI 请求里对应 `max_tokens`。每个 CLI invocation 可以有多轮工具调用/原生响应；N 不是这些响应总和，也不是全阶段累计值。`backends.py:154-181,194-206`；`cli.py:112-113,160-161`。 |
| 会限制整场 token/费用吗？ | 没有累计用量达到 N 即停止的判断，也没有生产 `--max-budget`/`--max-turns` 参数。Extract 只按 attempt 记录/聚合实际 usage；时间、调用次数另有限额。一个阶段甚至一个 CLI 调用累计输出均可以超过 N。`recovery.py:288-367`；`audit.py:199-216,824-851`。 |
| None/default 是什么？ | 产品默认 None；CLI 不显式注入该环境值，也不继承主机 env/settings 的同名值，所以回到当前 Claude CLI 的缺省策略。它不表示 max_tokens 无限，更不表示无任何输出边界。`runner.py:44`；`backends.py:28-34,164-181`。 |
| 为什么曾从 32000 提到 65536？ | 本机 Claude Code 2.1.316 在该 Qwen 模型路径默认请求 32000，显式覆盖后请求 65536；目的是扩大一次连续响应空间，降低遇到同类单响应边界的机会。不是扩大整场“可花 token 总额”。`LIVE04.md:3,40-50,68-85`；`LIVE05.md:21-26`。 |
| 是否必须保留 65536？ | 不必须。CLI 默认曾命中边界并自动续写，因此不是原生无法继续。既有真实短 JSON 成功只证明该配置下短响应可用；未生成满 65536、未证明长提取更快或更可靠。可保留可选覆盖入口，默认继续 None；不能把 65536 当成必须修复的缺陷。`LIVE04.md:16-18,89`；`LIVE05.md:30-58`。 |
| 只保留统计、不显式配置会怎样？ | 现有产品直接不传参数即可，实际 usage 统计仍在；本机已捕获的同版本/同模型条件下会恢复请求 32000，未来 CLI/服务配置可能不同。达到原生单响应上限仍可能触发其续写/截断分支，Extract 不再主动覆盖它。“只统计”可以取消我们自设的累计停机条件，但不能消除原生 API/模型本身的有限响应边界。 |
| 与 thinking / 上下文窗口的关系？ | 本地捕获同时有 adaptive thinking、effort=max、max_tokens。参数不设置固定 MAX_THINKING_TOKENS；[1m] 是上下文窗口选项，不自动给 1M 输出。不能从既有捕获推断 provider 具体内部映射或隐藏 token 计费。`LIVE04.md:44-50,74-85`；`LIVE05.md:11,19,26`。 |
| 能在同一个冻结 run 随便换吗？ | 当前实现把值写入 config/attempt/session binding/fingerprint，绑定不符会拒绝；统计本身不要求这些策略，但现有固定身份契约如此。只是解释当前实现，本次不更改旧记录或重建运行。`agent.py:49,75-76,280-283`；`runtime.py:85`；`store.py:897`。 |

ATTEMPT_REPORT 把 LIVE05 写作“缺少显式原生输出预算配置”（第 86 行），应理解为新增可选控制能力；LIVE04 已明确没有丢失既有配置或 effort 失效（第 3、28-30 行）。不能把“没有这个可选入口”改述成已证明导致失败的执行器 bug。历史调用真正结束的依据是两份 timeout / exit143 / confirmed（`LIVE04.md:14`），不是看到一次原生响应触顶就推断整个 agent 已停止。

## F. 建议供共同决定，不在本轮实施

- **保留统计，输出覆盖维持可选。** 不为整个阶段/算子新增累计 token 杀停；不要求每次指定 65536，也不把它当完成或加速承诺。若目标就是依 CLI 默认，省略参数即可，无须先删代码。
- **减少会拒收完成工作的时间层次。** 优先共同讨论流程审计中的阶段/预留/检查 120 秒门槛和单调用硬停策略；不把所有下层等待一律删除。A 表的 `ps` 10 秒失败应优先与活模型终止路径解耦：一次观察失败可报告并重试，暂不能确认身份时仍禁止不安全信号/放行；不能仅加大到另一个随意常数后称问题消失。
- **保留明确用于退出回收的等待。** TERM/KILL、wait/join 和队列投递的边界防止清理无限阻塞；若 1/3 秒导致已退出的有效结果误成 fence，再基于实际慢清理证据调整。它们不是应改为“无限生成”的业务预算。
- **监控阈值继续只提醒。** 300/75/1800 秒已经是观察提示，不需为了让模型充分工作而取消；可按监督体验简化节奏，不能据此授权自动杀停。
- **不要用新增一批短超时填所有空白。** 来源 Git、Python 校验/发布和 supervisor 锁目前无统一时间上限，先让其等待/耗时可见；需要可取消的可靠性机制时针对具体阻塞点处理，避免再引入“模型做完但本地一步稍慢就重跑”的门槛。

检索覆盖：`schema_extract` 非 archive/非 tests 全部生产 `.py`、活跃 README/REVIEW/prompt 和 `EXTRACT_SCHEMA.md`，检索 timeout/deadline/seconds/interval/sleep/monotonic/heartbeat/grace/backoff、subprocess/Popen/wait/join/communicate/flock 及 token 限额，随后追踪 Extract 实际使用的 Template 检查 API。未把测试 stub、GPU/kernel 算法时间、Graphviz 旁支或历史诊断等待算作当前提取限制。
