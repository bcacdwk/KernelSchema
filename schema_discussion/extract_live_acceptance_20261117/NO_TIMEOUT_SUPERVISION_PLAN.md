# 无 timeout 与现有巡检接管：最小实施方案

2026-11-18。只读评估当前工作树；依据 `UNLIMITED_EXECUTION_BRIEF.md`、三份 `TIME_LIMITS_*` 报告和相关源码复核。未改产品、冻结 run 或既有报告，未启动 monitor、提醒投递、review-eval 或模型。本文中的实施方案尚未实现、未验证。

**默认方向：删掉全部 Extract 自设时间截止和短命令 timeout，不保留单调用硬上限，不用大常数、`infinity`、`None` 配置分支或改名后的“保护时间”替代。** 正常工作由真实结果推进；supervisor 依据证据巡检。取消是显式控制，终止确认是另一个事实，二者不能因为取消了等待上限而合并为“成功退出”。前轮报告中“保留单调用硬 timeout”“保留退出宽限”的建议已被用户最新决定覆盖。

## 1. 直接删除与继续保留

| 删除 | 保留 |
| --- | --- |
| O/S/C：整算子、阶段/P5 节点、调用秒数；裁决/签收秒数和独审时间预留 | P1–P6、P5 节点责任、worker→独立 reviewer、裁决/签收角色及真正的语义验收 |
| 所有剩余秒数门、派发缩额、超时拒收、过期孤儿接管 | 调用/处理的真实开始、结束、耗时和用量；未知/估计口径必须标明 |
| 120 秒完成回执处理门与裁决候选准备门 | 检查器实际结果、原始回执、候选、规范化稿、checkpoint 和审计关联 |
| `review-eval` 240 秒默认、1800 秒上界及参数 | 每 case 一次独立 reviewer、只读材料、真实结果评分和逐 case 落盘 |
| 版本 15 秒、`ps`/Git 10 秒、queue 30 秒、wrapper/drain/Runner join、TERM/KILL 宽限、SQLite 正数 busy wait | 可取消或非阻塞的等待；真实进程身份、gate 的后代结清证明、未结 fence |
| 时间追加、时间 capability、deadline/超时提示及 prompt 中的剩余时间要求 | 原有次数规则、明确人工控制和有条件的有限自动恢复；不是无限退修 |

轮询、退避、心跳和提醒间隔继续保留。其到期只安排下一次观察/提醒，不能杀任务、拒收结果、释放 fence 或凭年龄替换仍在运行的操作。`queue.Queue.get(timeout=0.1)` 一类调度轮询可换成 `get_nowait` + 可取消事件等待以减少误解，但其本身不是任务上限。不要全局删去名为 `budget` 的 schema 字段/字符/图复杂度规则。

## 2. 主流程删除面

以下按文件列删除点，不重复前轮逐行清单。

| 文件 | 最小修改 |
| --- | --- |
| `schema_extract/tools/runner.py` | `DEFAULT_CONFIG`、`normalize_config` 删除 O/S/C 与 `adjudication_seconds`；派发只检查原次数/一次性决定，不计算 timeout；`invoke`/`cli.run`/`prepare_attempt` 不传秒数。删除 `_record_check_budget`、`_check_timeout`、`budget_empty`、因余额把 `complete` 改按 `continue` 处理的分支，以及最终检查超额拒收。`validation.checking` 只保留真实 elapsed。删除 `_reconcile_orphans` 的三个 deadline 和 `expired`，仅按明确 cancel 或已观察到结束来结清，健康孤儿不因年龄被终止。移除本系统生成的 `timeout` outcome 恢复分支；provider 的真实错误仍按真实 `cli_error` 处理。 |
| `schema_extract/tools/budget.py` | 删除整个时间预算模块，包括 `_remaining`、`_reserve_seconds`、`review_seconds`、`check_timeout`、混合 `addition/extend`、时间停止集合和补时文案。必要的次数判断迁入已有 `stages.py`；calls-only 追加归入已有恢复/控制代码，不另建预算框架。详见第 5 节。 |
| `schema_extract/tools/store.py` | 新 run 不写 operator deadline；scope 只登记 `started_at`，不写 deadline 和三类 `*_reserved_seconds`。`prepare_attempt` 删除 `timeout_seconds` 参数/字段。裁决候选遍历不因 120 秒停止，仍去重并执行相同检查。summary 删除 `budget` 的时间投影，原次数和各 scope 进度用简单 count/progress 字段呈现，elapsed/usage 留在原账目。 |
| `schema_extract/tools/stages.py` | scope 初值删除 deadline/时间预留字段；停止状态区分次数耗尽与真实失败。`resume_progress`、reopen/维护继续保留计数、generation、批准和一次性决定，不出现“恢复重置时钟”。 |
| `schema_extract/tools/agent.py` | `run/_run` API 删除 timeout 参数及 finite-positive 校验；删除 launch 前和运行循环中的两处超时判断。monotonic 仍用于耗时统计、扫描节奏。版本、日志、退出和取消改法见第 3 节。 |
| `schema_extract/tools/cli.py` | extract 删除 `--operator-seconds/--stage-seconds/--call-seconds`；review-eval 删除 `--call-seconds`；resume 删除 `--add-seconds`，删除旧时间 capability 检查、参数转发和混合 budget 字段。新 CLI 对这些旧参数直接报未知参数，不静默接受。文本 status 不再打印 deadline。 |
| `schema_extract/tools/review_eval.py` | 删除 `call_seconds` 形参、范围校验、冻结配置和 AgentCLI 传参；逐 case 结果/耗时保留，复用改后的可取消 AgentCLI。它目前不是 Store 中的常规 run，现有 `monitor --run` 不直接监督它；前台取消和其 `report.json/attempts` 观察必须单独说明，不能声称删 timeout 后自动获得 clock 接管。 |
| `schema_extract/tools/prompts.py`、`prompt/operations/*.md` | 动态材料不再给 deadline、call_seconds；只给次数、已用统计和本轮实际任务。清理 normal_worker、retry、recovery、receipt_repair、maintenance、reopen、monitor_handoff、wakeup 的“原时间预算/剩余预算”。改为保留原身份、草稿、计数及授权；不得暗示无时间限制就可跳过独审或草率收尾。 |
| `schema_extract/tools/recovery.py`、`monitor.py` | scoped recovery 的 `_budget` 调用改为次数及决定资格检查；恢复不要求剩余秒数、不补时。monitor 删除 invocation deadline/overdue、`deadline_exceeded` 及补时恢复建议。`possible_stall` 的 300 秒提示与已删 `call_seconds` 解耦，只保留观察阈值。 |
| `schema_extract/tools/runtime.py` | capabilities 删除 `budget_extension.v1`；如保留手动加 calls，登记清楚的 count-only 能力，不接受旧混合授权对象。冻结新执行器仍保留；不为旧参数/旧 run/旧预算格式搭兼容层。 |
| `schema_extract/tools/audit.py` | 删除 `budget_policy=wall_clock_deadline`、`waiting_consumes_budget`、deadline、seconds extension。保留 `run_wall_seconds`、各 CLI/processing 秒数、并发不相加的说明、控制等待与估计区间、provider/root usage、session 和证据哈希。 |
| 当前使用文档 | `schema_extract/README.md`、`tools/README.md`、`REVIEW.md`、根 `README.md/EXTRACT_SCHEMA.md` 中相关预算/调用 API/恢复说明统一；review-eval 示例删旧秒数。历史试跑报告、archive、冻结 run 原样保存。 |

不应只删除 120 秒最终 if：当前还会在开始检查、P5 merge 和 normalization 决策处因余额改变处理方式。这些路径全部删除后，慢检查通过的稿件才会按原决策采纳，不必让模型重写已经完成的工作。检查中断仍不等于检查通过；未完成处理继续落在原 `unsettled_attempts`/`processing.json` 恢复路径。

## 3. 全部短 timeout 的替代，不把无限阻塞带进控制面

### 3.1 进程身份：直接使用已有 `/proc` 能力

**建议用本机 Linux `/proc` 完成 process table，删除 `ps` 子进程。** 当前 `processes.py` 已有 `_boot_id()`、`_linux_process()`、boot/start token、pidfd 发信号和子进程 gate；gate 本来已依赖 Linux `prctl`。只读 `/proc` 与此前删除的隔离能力无关。

- 枚举本次 `/proc` 数字 PID，以 stat 中 ppid/pgid/state/start ticks 和 boot ID 建表；单 PID 消失按扫描竞态处理。权限/解析失败保留 unknown，不能把不完整表当作所有缺失 PID 都已退出。
- 已知身份优先直接查该 PID；发信号前仍核 token，保留 pidfd，不能退成只凭 PID/进程名 kill。
- 当前 `process_identity_matches` 和 `recovery.confirm_termination` 还要求 `started` 为 ps lstart 字符串；新身份校验一起改为 boot/start token，不能只换表生成器而漏掉消费者。新 run 不需要 lstart 旧格式兼容。
- AgentCLI 的例行观察失败只记录 observer error、保留最后可信 owned 集合和 fence，继续检查 cancel/真实 exit；不能再由 `record()` 的 `ps` 异常进入清理并终止健康模型。启动前连归属都无法建立时仍不放行 gate。

这既删掉 10 秒，又消掉“为了避免 ps 无限卡住再给 ps 加新 timeout”的循环。

### 3.2 模型、版本和日志

1. 模型主体保留原 gate，父线程只做退出/取消/身份轮询，没有总截止。gate 的 `child.wait/os.wait` 可以等待：它就是 owned 子进程锚，不是 supervisor 主循环；父侧和独立监督必须仍能控制它的子进程。
2. 15 秒版本检查不能只改成无 timeout 的 `check_output`。版本探测应在已有调用归属之内执行，用 Popen + poll，让 cancel 可以到达，版本输出与原生模型 events 分开。最省归属分支的方向是在原 gate 的 setup 阶段完成版本查询/核验，再启动模型，仍由同一锚收养后代；父侧可在版本未完成时观察和取消。保持“可执行文件/版本/精确会话一致才启动模型”，不能为省事取消版本核对。若实现选择单独辅助子进程，也必须绑定原 attempt/launch、复用原 fence，不得留下不可恢复的裸 check_output。
3. 建议让 Popen stdout/stderr **直接写现有 `events.jsonl`/`stderr.txt` 普通文件**，删两条 `shutil.copyfileobj` drain 线程及两次 3 秒 join。当前读取/观察已经基于这些文件；不必再建日志收集服务。成功仍需原生终结、真实 exit、gate 后代结清和最终日志校验；直接写文件不构成提前验收理由。
4. wrapper 的 `wait(timeout=3)` 改为非阻塞 `poll`/`waitid(WNOHANG)`；未退返回/记录原 `termination.confirmed=false` 与 survivors，而非无限 wait。wrapper 已退出后才读取真实 return code。Runner finally 不无限 join，也不保留 3 秒；线程尚活就保留原未结集合及 `termination_unknown`，继续由 owner/恢复流程结清。
5. 去掉 timeout 后不能用“调用线程结束”“主 CLI exit 0”“文件不再增长”替代 gate 的 `descendants_reaped`。此前的进程归属修复是长期需要，继续保留。

### 3.3 Git、同步检查与其他本地等待

| 当前点 | 删除及最低可取消方案 |
| --- | --- |
| `schema_template/tools/rules/values.py` 每条本地 Git 10 秒 | 删 timeout 与 TimeoutExpired 诊断。Git 结果仍决定真实仓库检查；不得遇慢即跳过检查。若执行处需要响应取消，用一个小的 Popen/poll 等待函数，检测调用方的取消并记录辅助进程身份；独立 Template CLI 响应用户中断。不要为每个 Git 子命令建后台作业或 ledger。 |
| `inputs/snapshots` 来源及私有 Git、Python normalization/check/publication | 这些本来没有统一 timeout。检查/prepare 在 Runner 主线程同步执行，现有 cancel_event 不能打断它们，不能因为 AgentCLI 可取消就宣称全链路可取消。循环/外部命令边界可消费已有持久 cancel；机器处理前已有 `processing.json`/unsettled 记录继续保留。真正卡住边界时，独立 supervisor 能先写 cancel、返回 pending；经授权的控制交付可核准原 Runner/辅助进程身份后终止并让已有事务/恢复对账，不能等待拿到 Runner 写锁才发控制。无需把整个 schema 求值改造成新调度系统。 |
| Codex SQLite 0.2 秒 busy wait | 取消“等待一小会再放弃”。SQLite 明确设置零 busy wait（Python 参数 `timeout=0` 是关闭锁等待），争用立即走已有 UUID/日期文件查找，不因可选索引忙阻断调用。不能只删参数，因为 sqlite 默认会重新引入 5 秒等待。失败仍记录 usage unavailable，不能伪造用量。 |
| `schema_template/tools/pipeline/render.py` Graphviz 60 秒 | 这是共享 Template 的可选 PNG 路径，**不在常规 Extract 提取/检查链路**，也未发现上轮触发。建议作为相关共享工具的小范围清理，一并删除这一处已知产品 timeout，并保留用户取消/子进程退出路径；不扩大为全仓库/GPU/测试 harness 的超时重写，不改字段规则。 |

runner 同步纯 Python/文件 I/O 不是一个“删 timeout 就自动变可取消”的问题。最低承诺是控制提交和独立巡检不等 Runner 返回；外部取消后保留真实未结现场。若产品要求连 Python 任意指令点都优雅抢占，那会扩成更大的执行隔离改造，不属于本轮最小方案。也不声称用户态轮询能保证在内核不可中断 I/O 或整个磁盘失效时及时结束。

### 3.4 TERM/KILL、显式 stop 与不等待结清

删除 `terminate_owned(grace=1)` 的两段截止、queue 清理的 1+1 秒、`monitor stop` 的 2+1 秒及其按时间自动升级。**不能把循环改成一直等 TERM，然后永远没有机会退出或发 KILL；也不能改为 sleep 一个短常数后照样 KILL。**

建议将现有终止函数缩成一次可重入动作：核归属、对本次确认的目标发指定信号、做非阻塞观察、返回原 `confirmed/actions/survivors/reason`。已经结清自然返回 confirmed；未结清用现有 unconfirmed/unsettled 和 session fence 等待后续观察，不新增另一套“pending job”模型。

**最小默认建议：现有 `cancel` 就执行强制取消，直接对核准 owned 后代发 KILL，异步确认。** 当前 cancel 本来会在短宽限后升级 KILL；删掉宽限后直接使用同一终止意图，比 TERM 后再新增 force API 更小。不按 elapsed 决定发信号，触发只能是明确 cancel，或主 CLI 已真实结束后的剩余 owned 后代清理。保留 gate 锚直到后代结清；不对“整个机器/所有 claude”发宽泛信号。公开结果可以是“取消请求已登记，终止仍待确认”，不能把发过 KILL 当作退出证明。

这一默认会放弃 CLI 收到 TERM 后的最后刷盘机会；已有落盘草稿、checkpoint、原始输出和用量证据保留，未落盘内容不能承诺恢复。应在最终方案中明确知会用户这一行为变化。希望让当前调用自然完成时使用已有 pause/hold；不增加 force 标志，也不引入新的等待时间。

`monitor stop` 先原子停用 clock generation，使迟到投递/scan 不能覆盖新控制，再通知原 clock 停止并立即返回观察事实。非阻塞 clock 可自行退出，自己的在途 queue 以显式 stop 的授权清理，不等待 1/2 秒；确需直接终止时只针对已核准 clock/queue 身份。其本身不 cancel Extract。未确认原 clock/queue 已退时不得另起一份竞争实例。

这是不保留任何短退出等待的默认。第 7 节明确列出需知会的行为变化，不为“可靠性”擅留 1/2/3 秒例外。

## 4. supervisor 现状与最小接管闭环

### 4.1 已证实与未证实

已静态确认的工具仍在：`monitor start/status/stop/ack/scan`、`inspect`、`pause/cancel/resume`、全局 hold、scope recovery、`monitor.py`、`wakeup.py`；`wakeup.md`、`monitor_handoff.md` 仍由 `render_operation` 使用。stage backend 与 supervisor `codex queue` 控制通道分开。clock 绑定 thread、generation、reminder ID、scan claim；已入队、ack、committed scan 各有持久证据。runner.lock、session lock、clock lock 是非阻塞取得；普通 inspect 不需要拿 Runner 写锁，也不做深度 schema/Git 验收。

上轮真实证据只证明 clock 曾启动、有进程/锁/心跳和初始观察。`PERIODIC_SUPERVISION.md` 明确写着首提醒投递及本线程 ack/scan 尚待验证；随后用户在 02:45 UTC 取消，而记录的首 due 在约 02:46 UTC。`ATTEMPT_REPORT.md` 是终止结论。**不能宣称已经真实完成 queue→主线程收到→ack→committed/reused scan 的接管闭环。** 本轮不会启动它来补证。

默认仍用已有节奏：每轮观察后 30 秒；心跳最多 15 秒；初始 scan 后前三次提醒 5 分钟、后续 30 分钟。提醒故障补送/ack 注意阈值和次数保持既有语义。75 秒 heartbeat stale 原表达式含 `QUEUE_TIMEOUT_SECONDS`，删除 queue timeout 后将陈旧提示独立表达为观察参数；其只显示异常，不能自动重启 clock 或杀提取。

### 4.2 当前会使“独立巡检”卡住的点

1. `process_table()` 的同步 ps：上面以 `/proc` 删除。
2. `Store.initialize/supervisor()` 的 `flock(LOCK_EX)` 会无限等；clock 心跳、scan commit、ack、control 提交都可能被同一个锁卡住。注释写“short transaction”并不等于实际非阻塞。
3. `Clock.tick()` 实际串行 `observe()`→queue `_send()`；现在的 30 秒只是让它最终返回。直接删除后无限 communicate 会连观察和心跳一起停。
4. `observe/scan` 后的 `authorized_restarts()` 会同步做 frozen execution/失败身份/依赖等核验，且可能启动恢复；即使只读 inspect 很轻，这个已授权动作仍会阻塞 clock。
5. Runner 的准备、候选检查、处理、Git 和发布在写者侧同步运行。它们不必影响只读 inspect，但会推迟消费 cancel；若错误地把深检搬进 inspect，监督也会一起卡。

### 4.3 最少改动的控制面

- supervisor 写锁改为立即尝试；锁忙返回明确 busy。clock 下次 tick 再试，交互命令返回未提交/待重试，不伪称 ack、scan 或 cancel 已登记。Runner 对短控制争用只退让后重试，不把它归为模型失败。锁内仍只做小的持久记录事务；不运行 ps/Git/模型/检查或等待子进程，不按锁年龄破锁。
- queue 投递改为**已有 clock 内的一份在途 Popen**：启动后保存 reminder/attempt 绑定及进程身份；每次 tick 非阻塞 poll/读取现有投递输出，同时照常观察、心跳。可以用普通文件接 stdout/stderr，避免继承 pipe 的后代让 communicate 永不 EOF。至多一份未结投递，不起线程池/新守护系统。
- queue 真正 exit 后核对 thread/message UUID 才登记 queued；未完成仍是 sending，不能到 30/120/1800 秒就改成失败、kill 或再发一份。观察阈值可以提示“投递未结束”。只有实际失败、明确取消或原 owner 死亡使投递不确定，才进入既有有限补送路径。stop/失效后的迟到结果只补充原 attempt 事实，不撤销已 ack/scan 的结果。
- 默认 `auto_restart` 仍关闭，不新增“检测 hang 自动 kill/resume”。用户已授权的自动恢复继续只处理原有已确认终止的 EX_TEMPFAIL 75、普通作用域、未被新 pause/cancel/hold 覆盖且仍有调用次数的失败。把可能长期阻塞的 recovery 核验/启动交给一份现有后台执行入口/短辅助进程，保留原 `retries`、failure ID、clock binding 和消费校验；clock 不同步等待它。至多一份在途恢复交付，未结就下轮观察，不建新的恢复调度器。人工 ack/scan 的提交不等待其完成。
- 保留现有 summary 原子投影、state stat 一致性检查、单层 run/attempt 文件读取；禁止周期观察做全仓 Git/整树哈希/重新验收。观察到未知证据返回 unknown，而不是把未知转成死亡或批准。

实现应优先让原有 `sending/observation_claim/retries/termination_unknown` 和进程记录承担事实，不另建通用任务生命周期。上述 queue 与可选恢复的独立执行是为了保证 clock 不被它自己调用的外部操作堵住，不是新调度系统。

### 4.4 每次巡检看什么，如何控制

| 证据 | 能支持的判断/行动 |
| --- | --- |
| `summary/state_revision`、阶段/role/checkpoint、active/unconfirmed/unsettled、控制 request ID/消费记录 | 判断卡在排队、准备、模型、处理、暂停边界还是结清；确认旧控制是否已经登记/被覆盖。状态不变本身只意味着需要核对。 |
| Runner launch ID + boot/start token、writer lock；attempt launch/process/cli-exit/exit/result；gate 后代闭合 | 区分活 Runner、已退出 Runner、健康孤儿、主 CLI 已退但后代未退、身份不明。仅 Runner 死亡不能推出模型结束；仅 lock free 也不够。 |
| events/stderr 的大小与 mtime、公开 native 状态/工具命令、`processing.json` 起止、可核准的辅助进程及必要的 `/proc` 状态 | 用来定位长等待发生在哪里；不读取隐藏推理。CPU 低、日志静默、D/S 状态、长 elapsed 都不能单独证明死锁。若存在真实错误/已退出子命令/失去消费者/重复可复现阻塞等相互支持的证据，再按已授权范围处理。 |
| clock identity/lock/heartbeat、sending attempt、queue ID、ack、committed/reused scan、next_due_at | 区分观察正常、投递卡住、投递确认但线程未接收、ack 后 scan 未完成；提醒故障不能转成提取故障。 |

处理顺序：正常长调用继续；信息不足保持 unknown 并进一步只读核对。普通 pause/hold 只阻止新派发并等边界，不能用来打断 hang。在已授权取消时先用现有控制入口持久提交，再核实控制被消费；若 Runner 卡在同步步骤，控制交付可以独立核准 owned 进程并发取消信号，不能依赖取得它的 writer lock。取消若需要退出 Runner 本身，先留下已绑定控制与在途进程记录，保留旧写者/线程退出核验，不能因 writer lock 释放就立刻把旧线程仍可能落盘的 attempt 交给新写者。没有时间到点自动 kill。

确认进程结清之后，才走原恢复：校验真实 exit/native result/schema/nonce/精确 session、冻结身份、checkpoint 和依赖；仅有真实成功的完整原结果才能重放接收。中断会话退休后从安全 checkpoint/交接继续；未启动的 gate 可复用已有合法会话。unconfirmed、unsettled 或后代归属未知时不得重派同一作用域、清 fence 或伪造终止。使用现有 `resume --scope` 处理符合条件的普通失败并保留健康兄弟；当前 cancel 是 run 级，不擅自新增 scope-cancel 语义。

## 5. 删除 budget.py 不等于删除次数规则

默认保留：每 scope 32 次调用、实质退修 5 次后进入裁决、两次自动回执修复/执行失败重试、裁决/签收各一次、review-eval 每 case 一次，以及独审/裁决/签收的**调用次数**保留。它们约束循环和角色责任，仍有意义；当前用户只明确撤销时间/累计 token 截断，没有授权无限续写、无限退修或重授一次性决定。

最小迁移：将 `call_limit/reserved_calls` 和“本次是否可派”的纯次数判断放进已有 `stages.py`；计数耗尽给出清楚 `call_limit_exhausted`，不再与时间混称 `budget_exhausted`。CLI 自动失败与回执修复次数耗尽也用 retry/repair limit 描述。Prompt 显示次数和角色责任，不能继续显示“剩余秒数”。

`--add-calls` 是既有明确恢复能力，建议保留，避免删时间机制时顺手删掉用户可控制的次数追加。把原混合 extension 缩成 calls-only 的一次性授权记录（ID、run、origin、at、calls），删除 seconds、旧 deadline、scope_deadlines；复用原 resume launch ID 幂等，不能每次 resume 都重复追加。放在已有 recovery/control 模块即可，不需要新的 budget.py、调度器或兼容旧对象。scope/hold 释放仍不隐式加 calls；所有已用计数、一次性决定保持。

若后续明确要连总 calls 上限也取消，那是另一个策略决定：需同时说明 worker `continue`、回执修复、人工恢复的循环边界。不能借本次删除 timeout 默认做这件事，也不能改 schema 派生/引用求值规则。

## 6. 两轮实施落点与最低验证

本方案与输出配置评估、前轮修复审查、精简整合共四份评估合并讨论后实施。建议两轮共四位新实施代理，不让同一共享文件被并发编辑。

**第一轮：输出配置实施 + 无 timeout/巡检实施。** 输出代理主要持有 `backends.py` 及其输出配置测试；本方案代理持有 `budget.py` 删除、`runner/store/stages/recovery/monitor/wakeup/processes/observation` 和 Template 两处 timeout。共享 `agent.py/cli.py/review_eval.py/prompts.py/runtime.py/audit.py` 由主代理指定单一文件 owner，另一代理提交具体改动要求后串行整合。主预算删除、AgentCLI 无 timeout API、全部短等待替换必须在这一轮一起落地，不能中间把“模型无限运行、监督仍可堵死”宣称为可用成品。先完成核心行为及下面的关键离线检查。

**第二轮：前轮正确性修复实施 + 精简/整合实施。** 在第一轮结果上验证已定的源码修复及前述取消/接收边界，修复必要项；另一代理清理过时 capability/字段/prompt/文档/测试、统一状态与审计口径，交叉复核取消交付与不重复派发。凡第一轮接口再变，相关定向测试随之重跑；不增加旧 run 迁移、旧预算参数支持或新的 schema 规则。

最低离线验证（不是当前轮执行项目）：

1. **预算已消失**：静态搜索当前生产代码的秒数配置、deadline、timeout、grace、bounded wait 与 prompt 字段；逐项确认保留下来的时间值仅为轮询/提醒/耗时。旧 CLI 秒数选项被拒；新配置/attempt/summary/audit 无旧时间预算字段。SQLite 的 0 要明确是关闭 busy wait，测试不误判成残留正数上限。
2. **长工作仍能成功**：用假 transport/虚拟时钟跨过原 120/240/1800/3600/7200 秒，不真实等待；complete/approve、P5 merge、裁决候选不会因 elapsed 改判。确认独立 reviewer 和现有次数门仍执行，不能只测没有 timeout 字符串。
3. **可取消与真实 fence**：短合成子进程分别模拟版本未返回、CLI 退出但后代还在、直接日志、忽略 TERM、wrapper 仍活；确认 cancel 无宽限即发核准信号，但只有真实退出及 gate 结清才确认；PID 复用/`/proc` 读取缺失保持 unknown，不误杀。原线程仍活时不得重复恢复同一 attempt。用事件同步，不跑分钟级 sleep。
4. **恢复与统计**：虚拟年龄很大的健康 orphan 不被终止；已结束 native success 可正常重放，取消/不完整原结果不可批准；调用、processing、usage 按 attempt 最高水位去重；calls-only 授权回放不重复追加；维护、单次裁决、健康 P5 兄弟和原 schema 语义保持。
5. **巡检独立性**：只离线实例化 Clock，注入假 sender/假 Popen；投递永不完成、supervisor 锁忙、恢复核验未返回时，新的 tick 仍能观察/报告 pending，stop 不无限 wait，不产生重复发送。已有 thread/generation/ack/scan 竞态及幂等测试继续通过；真实 queue 与模型都不启动。
6. **review-eval/Template**：fake CLI 验证去掉时间参数仍记录每 case 一次及 usage；本地 Git/Graphviz 替身验证无时间拒绝且可中断。无需实际图渲染、完整提取或全量测试。

优先改造已有 `test_recovery_budget.py` 中有价值的次数/幂等测试并删除纯时间公式测试；使用已有 `test_backends_runtime/test_current_recovery/test_scoped_recovery/test_monitor_evidence/test_wakeup_retry/test_review_eval/test_stage_flow` 的相关定向用例。测试通过只支持离线协议与边界，**不等于真实接管或全提取成功**。

真实 queue→接收→ack→scan 必须留作用户以后授权的独立、小范围接管验收；本轮及两轮实施不以启动真实模型或冻结旧 run 来“顺便验证”。

## 7. 需知会的行为变化与明确遗留

1. **cancel 的强度与返回语义。** 现有 cancel 直接 KILL 核准 owned 进程、返回真实 pending/confirmed；代价是放弃 TERM 刷盘机会，温和边界停止继续用 pause。实施前汇总方案应让用户看到这一变化；无需新增 force API 或短退出等待。
2. **queue 一直活着但无完成证据。** 推荐让 clock 保持观察/心跳，并报告 sending，等待真实结果或显式终止该投递；不按 30 秒取消，也不按提醒重试间隔重叠发新 queue。这意味着提醒通道本身卡住时，不能保证同一个通道还能通知主线程。无新增第二通道/外部守护的前提下，应如实保留这一边界，不能声称无人值守绝对可达。

其余按默认方案直接落地，不需要重新询问是否保留某个硬 timeout。平台/CLI 服务端自身未暴露的网络、工具或模型限制不由本仓库控制；实施应删尽本仓库自设项，不能把“本仓库不再设时间截止”夸大成服务端也已无限。

当前没有真实 hang 判定样本或完整提醒接管样本；因此本方案不为 hang 拍一个新的统一秒数。持续时间只产生“应查看的事实”，结束工作仍由模型真实结果、明确控制与进程终止证据决定。
