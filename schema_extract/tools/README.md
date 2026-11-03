# Schema Extract：工具维护与回执协议

当前版本 `2.2.0`，单点定义见 [__init__.py](__init__.py)。

[Supervisor 手册](../README.md)保留入口、参数、状态与恢复；各阶段页定义写审方法和阶段专有字段。[stages.py](stages.py)定义可机械判断的阶段读写范围，[receipts.py](receipts.py)定义回执 schema 与门禁。本页和运行代码一同进入 `rules/` 冻结清单。

## 写入与完整性

[runner.py](runner.py)持有运行写锁并串行提交结果；[store.py](store.py)负责不可变 checkpoint、已审依赖和权威状态。`_StateGuard` 只保护当前 Runner 线程最后提交的 `state.json` 字节：恢复、读取和写入 state、派发、接收结果及发布边界都核对当前文件。其他线程不能更新这份可信字节。历史日志和旧 workspace 不参与常规扫描；引用的关键对象在实际采用时核验。

发现 state 外改时，Runner 从最后可信字节保存 `authority_changed`、路径和预期摘要，保留原 checkpoint、预算、在途 fence 和未结算调用。`cancel` 仍可确认旧进程终止；显式恢复继续核对未结调用、计时和会话退休。普通日志变更不会触发这个错误；这份保护不承诺所有历史文件保持完整。

| 采用边界 | 核验对象 |
| --- | --- |
| 创建、进入冻结执行器与恢复 | 输入锁与 state 的指纹、配置、快照路径；源码、规则及执行器身份。 |
| 接收调用结果 | 已登记调用身份、许可差异、回执摘要、精确证据与引用、真实产物；陈旧或越权提案仍拒绝。 |
| 读取 checkpoint、批准或父依赖 | checkpoint ID、manifest 与实际字节；引用的 receipt、validated 记录、阶段/节点代际和祖先内容。 |
| 保存 checkpoint 与更新指针 | 复制后核对 manifest，先完整持久化不可变对象，再原子保存 state。 |
| 子节点、合入与 P6 准入 | 当前计划、有效批准集合、真实父依赖和上游阶段放行。 |
| 发布或 export | 匹配本 run 的输入锁、独立 P6 放行与最终 checkpoint；待发布副本必须匹配已审清单和 Runner 生成的交付回执字节。 |

恢复只读取 state 登记的 `unsettled_attempts`、`active_attempts` 与 `unconfirmed_attempts`。`unsettled_attempts` 在准备调用时登记 `prepared`，启动线程前先持久化 `dispatched`，直到结果处理和计时全部落盘才清除；结果已合入但计时未完成的崩溃也能对账。仅明确的 `prepared` 且没有相反发射证据可证明调用未启动；发射后缺少进程身份和终止证据仍保留 fence，不能凭文件缺失判断退出。

## 轻量观察投影

`Store.save` 原子保存 state 后，尽力原子写出 `summary.json`（`schema_extract.summary.v1`）。投影包含 run/input 身份、执行器版本与摘要/能力、`state_revision`、`generated_at`、`updated_at`、阶段和节点分类、在途与未结调用、原预算、停止原因、checkpoint 及发布状态。`approved_nodes` 等字段只表达 state 登记的进展，`content_verified: false` 明示没有深验；调度、恢复与发布不能把投影作为批准。

`store.summary(store, state)` 是纯 state 投影，`runner.summary` 保留原调用入口；两者不逐节点验证。`Store.read_summary(run_id)` 正常只读取小投影并 stat 当前 state，不读取 attempts/checkpoints。`observation.status` 为 `current/stale/unknown`，`source` 为 `summary/state`：版本戳不符显示 stale，投影缺失或损坏显示 unknown，默认仅回退读取一次该 run 的 state，保留上述标记且不补写、不深验。`fallback=False` 可禁用回退。旧 state 缺少执行器身份或修订号时显示未知，不能代入当前工具身份。投影写失败不会回滚权威 state 或阻塞调度；下一次观察由版本戳识别陈旧，后续正常 state 提交会再次生成投影。

`monitor.inspect_runs` 每轮只取一次进程表，只查看 state 登记的活跃/未结 attempt 的小记录与活动日志 stat。`monitor.scan` 复用它并持久化最新短观察及最多 64 条变化；默认从不恢复任务。未知状态隔离到单个 run；已经有 successor 的前任只显示历史状态和接续定位。冻结能力缺项保持未知，不由新版入口代入。

## 多任务控制与维护接续

`Store.supervisor()` 是 state-root 下唯一 `supervisor.json`/`supervisor.lock` 的短事务。账本只保存启动意图、控制修订、hold 影响、接续关系、观察、时钟和授权重试，不复制阶段权威或批准。正常控制均经过统一 CLI。需要 run 写锁时先在控制锁外取得；深验、冻结、等待进程和 exec 都不持控制锁。

后台先按目的地登记固定 run UUID/创建时刻，再在对应写锁下创建。启动令牌先持久化，wrapper 认领进程 PID/启动身份后进入冻结执行器；重复命令复用原身份。continue 同时核对旧写锁与启动意图，不能以“writer 尚空闲”忽略 wrapper 到旧冻结 exec 的窗口。新 run 构建中的接续关系同样是启动 fence。

hold 首先单独提交 `active_hold`，随后追加暂停请求。派发授权及发布 rename 与该闸门共享短锁；准备产物、生成提示词和复制/验证发布目录都在锁外。影响记录区分 queued_blocked、pause_requested、paused_by_hold、completed_before_hold 和保留/未知项。发布提交身份与控制修订先持久化，因此 state 还未更新时也能识别 hold 前完成。请求文件可能先于控制账本提交；释放和 Runner 启动还核对实际请求及已消费 ID，避免崩溃后漏掉手动 pause/cancel。

启动登记的 `mode=resume` 是明确人工恢复，可按原预算重试普通失败 scope；`hold/queued/retry` 不走此全任务分支。所有计数与期限保持累计，external_confirmation、原预算和已用一次性决定仍是硬停止。`assert_constructed` 同时供启动和再次 continue 使用，禁止从尚未提交预算继承的半成品建立新接续。

`continue_run` 深验选中 checkpoint 与旧冻结来源后，为唯一 successor 冻结当前规则；源码逐文件摘要和原 config 必须相同。每个旧 scope 全字段复制，包含动态预留和裁决/签收原文；所有 removed 节点保留。有效批准、候选/基线指针、角色会话和新调用列表重新建立，P1 独立重审。重新初始化 baseline 不重置已有 started_at/deadline。旧 state、lock 和选中稿件保存于 `continued/`，原运行本身不改写；已继承 `attempt_usage` 带 origin_run_id，供后续按来源去重审计。

`scan(store, run_ids=None)` 始终只观察。`configure_auto_restart(store, run_ids, limit)` 只配置显式授权；`authorized_restarts(store, rows, repo, clock_binding=...)` 在初始/周期扫描提交后应用它。时钟绑定在请求、后台启动预留和认领的控制事务内重核，stop 或新代际不能继续颁发旧扫描的启动授权，已耗次数不退款。当前正向暂态标记仅来自真实已终止的 EX_TEMPFAIL 75，默认停在该范围；每次失败先记原 cli_failures，超过普通 retry_limit 后不可自动重发。授权按 lineage 持久记账，failure/scope/attempt/result 摘要/控制修订在 Runner 内重核，其他错误节点保持原停止状态。

## 时钟与单轮提交

[wakeup.py](wakeup.py) 只在 supervisor 账本中添加 `clock`，格式为 `schema_extract.clock.v1`。`last_check_number`、`last_scan`、`next_due_at` 保存唯一进度；`reminder` 绑定 generation/thread/number/due 的稳定 ID，最多三个 attempt，分别记录发送前时刻、响应结果与 message_id，ack 单独保存。已完成或失效提醒保存最后一项及最多 64 项历史，累计发送次数不清零；没有另一份提取运行状态。

`Clock.start` 首次在前台观察并提交 #0；后台 `_clock` 只调用 tick/queue。start 持进程文件锁并将同一 fd 传给 daemon，父进程只关闭 fd，不能 `LOCK_UN` 解掉子进程的锁。Linux 优先用 PID/启动 tick/boot ID；其他 Unix 复用 `processes.process_table` 的 PID/启动时间，控制事务所需快照提前在锁外采集，读取失败保持 unknown。监督状态不能由单独 PID 或持久 enabled 推断存活；活进程缺锁标记 `identity_lock_mismatch`，未恢复的 tick 异常标记 `clock_error`。queue 子进程及其等待在控制锁外，SIGTERM 能中断等待；清理仅有界等待已知 queue 父进程并关闭本端管道，脱离进程组的子进程保留管道也不会导致永久等 EOF，送达仍按不确定记账。

`Clock.ack` 在控制锁内校验启用、代际、线程、当前 ID、到期和发送登记，重复 ack 返回 false。周期 scan 必须显式带线程，认领记录包含扫描者进程身份；普通 scan 不获得该认领。`monitor.inspect_runs` 在锁外采集，`monitor.record_scan` 在同一个短事务内登记观察和完整下一次时间；不存在观察已提交而周期只提交一半的窗口。进程在提交前退出可续读观察，提交后重放同 ID 直接返回已完成，不再消费轮次或重复自动恢复。失败的采集保留已接受身份。

ack/scan 可先于 queue 响应提交；晚到响应只补该 attempt 与已归档检查的消息身份，不把 accepted/completed 回退成 queued。发送成功但落盘前退出则保留 sending 和已耗次数，重启后仍间隔至少 120 秒按同 ID 有界重送。stop 先在控制事务中停用并推进代际，再于锁外终止时钟，旧提醒和未提交扫描随即失效。

audit 使用 `last_scan.id/number/generation/thread_id`、`started_at/observed_at/completed_at/elapsed_seconds`、`message_ids` 和历史 reminder 的 `ack.at` 定位检查及唤醒耗时。入队响应与实际同线程 ack 是两种证据；可选 probe 单独记在 `clock.probe`，不改正式周期。生产间隔固定 300/300/300/1800；测试通过 `Clock(..., now=callable, sender=callable)`、`start(..., launch=False)` 或可执行 `--codex-bin` 替身完成，不改生产时间参数。

## Usage、时间与证据封存

[audit.py](audit.py) 提供 `parse_events(path)`、`audit_run(store, run_id, *, attempt_id=None, session_id=None, verify=False, archive=None)` 与短文本格式化。新冻结 run 声明 `audit.v1`；统一入口可只读旧 v2 记录，能力或字段缺失保持未知。没有实时增量日志消费者，故不保存 cursor：正常 CLI 关闭后一次按完整行解析，保存完整文件 SHA-256、字节数、完整行边界、错误行与 completion 行/偏移/原始行摘要。未换行末尾保留在文件摘要内，不消费为完成事件；巡检不反复扫描它。

| 记录 | 口径与绑定 |
| --- | --- |
| `result.json.usage` | `schema_extract.usage.v1`；逐条 `turn.completed.usage` 原值和规范化 input/cached-input/output 字段，reported/partial/unknown，原始事件定位。缓存包含于 input；`counter_scope=provider_counter_semantics_unverified` 时所有 `additive_tokens` 为 null。 |
| `result.json.trace` | 本次关闭观察的日志摘要及完整行边界；字段存在不代表调用成功或终止已确认。`event_types` 为出现过的类型，次数在 `event_counts`。 |
| `attempt_usage[attempt_id]` | 原 `elapsed_seconds` 高水位，新增 origin、角色/阶段/节点、requested/actual session、usage 小摘要、CLI/processing 时间及证据摘要。原始 provider 对象仍只在 result/events，不另存调用历史。 |
| `processing.json` / `finished.json` | 现有可继续结算的时间记录；检查崩溃开放区间关闭时单记 `estimated_seconds`。分项与 budget 高水位校对，不能把它们当永久不变的文件摘要。 |
| `state.session_handoffs` | 每 origin run 唯一会话退休/接续事实；不复制可变交接进多个 usage 条目。继续运行的原文也保留在 `continued/state.json`，审计按 origin 定位。 |

`account` 仅把新增耗时差额计入原 scope，不重置 deadline、调用计数或预留。重复接收、恢复时 result 缺失或耗时样本变小时，保留既有 role/session/usage/origin 和时间高水位。不可变证据摘要首次绑定，后续不一致留诊断并保留原摘要；处理函数返回后，仍须核对 active/unconfirmed/unsettled 才标记 settled。CLI result 关闭后不再追加检查时间，以保留失败恢复所绑定的 result 摘要；后续检查由 processing 和 usage 定位。

默认 audit 只读 state 中已记账条目和小 Supervisor 账本。接续仅使用 successor 已继承的逐 attempt 账目，不把 predecessor 总数再次相加。计数语义未确认时展示逐调用原值，阶段/角色/节点可比较已记录 CLI 与处理累计时间；并发调用之和不称墙钟。`operator_wall_seconds` 仅使用可核实的原创建/发布提交锚点；无最终锚点、暂停区间或纯模型采样时明确不可得。时钟检查属于共享观察且历史有界，晚到 message ID 只补原检查，不新记成本。

有界真实 bf16 P1 调用已确认 `turn.completed.usage` 及原始扩展字段可保存并深验；仅一次 worker 提案通过完整门禁，独立 reviewer 超时，P1 批准仍为空。新会话与 resume 计数能否相加尚未确认，超时调用缺失的 usage 不补作零。

显式 `--verify` 读取选中的调用证据，使用 `verify_run_inputs`（含 aliases、私有 Git 来源）、`Store.verify_checkpoint` 与现行回执引文规则；不会启动检查器、提取或授予批准。`_receipt_response` 按已登记 attempt 的 `response_format` 核验原始输出：语义提案从固定来源与原产物 checkpoint 重新规范化并逐值比对完整 receipt，旧完整回执直接比对。reviewer 使用实际读过的起始 checkpoint，裁决补丁的结果另验；worker 使用已登记结果 checkpoint。`response.json` 与原始 `result.response` 仍须一致，原文件不改写。

`--archive` 在相同选择范围完成门禁复验，再流式采集 prompt/events/result/回执/检查记录、关键 checkpoint 以及来源/规则身份。所选对象可为复验与采集分别读取；这仅发生于显式审计，不引入热路径全扫。临时流只用于压缩包对象去重；不保留额外副本或游标。包的 `manifest.json.id` 由稳定范围与实际对象清单计算，不含本次观察时间；对象位于 `objects/<sha256>`，包含实际大小、原位置，并验证压缩包内每个对象。重复目标只在同清单且全部对象可验证时复用，否则保留冲突目标；目录链接、特殊文件、越界目标、缺失对象和活动范围不能被宣称完整。包不是恢复授权，不删除 runtime，也不读取 Codex/provider 私有会话库。

## 角色与会话

Supervisor 在主窗口发起任务、读状态、补充外部材料、处理已明确的任务要求变更和发布冲突。Runner 是运行状态的唯一写入者：选择阶段页、派发调用、验证回执、保存 checkpoint、合入和发布。worker 只修改调用中列明的产物；reviewer 通常只读对应 checkpoint，按阶段门槛给出结论，不通过修改状态自行放行。

reviewer 独立对照固定 kernel，判断数学与数值、工作组织、依赖、寿命、复用和适用范围是否被忠实压缩。每个决定只在合适位置表达一次，正确且必要信息齐备就直接 `approve`；不要求找到问题、退修或增加篇幅。阶段重点由六张任务页定义，P5 复用 P3 的逐段方法。

| 具体情形 | 判断 |
| --- | --- |
| 简短设计准确交代计算、工作归属、依赖与条件，重复细节已省略 | `approve`；篇幅短不构成缺项。 |
| 源码下一迭代覆盖缓冲前仍须等待上一迭代消费者，设计漏掉该依赖 | `revise`；引用对应生产/消费范围，说明覆盖风险及需补的关系。 |
| 源码有未排空或覆盖不足，设计忠实记录实际行为、代价和范围 | 可 `approve`；不替源码补等待、fallback或缩小 Contract。 |
| 标准答案、必要输出或关键外部验收依据无法确定 | `blocked` 并说明缺失材料、影响和恢复条件；非关键未知可如实备注。 |

| 阶段 | worker 与 reviewer 的上下文 |
| --- | --- |
| P1 | 分别新建，两个角色独立。 |
| P2 | 分别续用 P1 对应的会话，但明确注入 P2 页并单独验收和计退修。 |
| P3、P4 | 每阶段各开全新一对。 |
| P5 | 每个节点各开全新一对；同节点返工续用本节点原角色。 |
| P6 | 全新 reviewer 先审；确需修改才启动全新 worker。 |

Runner 直接启动 `codex exec`，正常续写用 `codex exec resume <会话 UUID>` 精确指向原角色，禁止 `--last` 或会话名查询。每次只传当前任务页和定位信息：角色、阶段、节点、固定源码入口、起始 checkpoint、计划修订、可写范围、应读产物和本轮退修意见。换页时不因会话已有历史而省略新任务页。P1 的两个会话必须不同，P2 分别复用；P3/P4、每个 P5 节点和 P6 全部换新，各角色不得复用别的阶段/节点 UUID。reviewer 使用独立产物副本和只读 sandbox；裁决时也只提交补丁提案，由 Runner 应用。未完成调用恢复时，先确认终止，再从安全 checkpoint 建新上下文；`session_handoffs` 记录旧 UUID、新尝试和新 UUID，已退休 UUID 不能复用。正常暂停后已完成的会话仍按上表续接。

## 回执与短交接

模型输出为 `schema_extract.proposal.v1`，每次调用的 `response.schema.json` 是字段清单，定义见 [receipts.py](receipts.py) 的 `proposal_schema`。`attempt_id` 是本次已登记的随机 UUID，也是必须原样回传的 nonce；同角色续会话时每次调用仍换 nonce。模型只写语义判断、证据选择和当前页的阶段字段，通过 CLI 最后输出返回 JSON，不创建额外报告。

- `decision/summary/conclusions` 表达完成、通过、可执行退修或已核实阻塞；`issues` 写具体范围、影响、来源依据与恢复条件。
- `source_evidence` 选择 `category/path/line_start/line_end/quote_offset/note`；`artifact_evidence` 使用同一定位字段而无 category。`note` 说明该范围支持的字段或决定，`evidence` 从 0 索引本次 `source_evidence`。
- `checks` 只记真实命令和退出码；未执行为 `[]` 并说明 `checks_note`。Runner 的结构检查另记在 `validated.json`，不采信模型声明代替检查。
- 阶段专有的 `comparisons`、`plan`、`revision_scope/affected_nodes` 按当前任务页填写；`framework_candidates` 无必要建议为 `[]`，遵循[分流规则](../../EXTRACT_SCHEMA.md#框架问题分流)。

`normalize_response` 从已登记 attempt、许可差异及真实字节生成 `schema_extract.receipt.v2`，交现有完整回执门禁验证。工具填写 `identity` 的 run、输入、阶段/角色/节点、计划修订、checkpoint、依赖摘要、mode、generation 和 node_signature；填写完整交付文件 `artifacts`、字典序实际 `changed_files`，以及证据的固定 source_id、完整文件 SHA-256 和精确短引文。P5 只提交本节点 YAML，root/祖先仍可作为证据；其他阶段包含全部交付文件。模型不得提供这些工具字段，显式错误不会被替换后继续批准。

原始 `response.json` 与 `result.response` 保持一致且不改写，规范化结果另存 `receipt.json`。reviewer 的完整 workspace 必须等于其起始依赖字节；裁决选稿或补丁另保存结果摘要。Runner 仍核验活动 attempt、阶段/节点代际、真实父依赖、许可范围与全部来源锚点，批准只绑定实际审过的内容。草稿 checkpoint 只用于恢复，不能解锁后续工作。新 run 以 `proposal.v1` capability 和 attempt 的 `response_format` 标记；旧冻结执行器继续使用自己的协议。

`required_evidence/required_conclusions` 由 [stages.py](stages.py) 给出，complete/approve 必须覆盖；每类证据至少定位一次绑定入口，调用链文件可补充。`required_artifact_evidence` 由同一个接收门禁生成具体文件名：reviewer 必须引用本阶段全部所需产物，P5 包括 root、直接父与当前节点。机械定位准确不证明语义成立，结论与所选源码范围的关系仍由 reviewer 独立核实。

按范围读取使用冻结的 [reading.py](reading.py)，例如 `python3 -B <rules>/schema_extract/tools/reading.py --root <source> --path input.py --start-line 1 --end-line 80`。返回真实行列片段、全文件 SHA-256 和 `next_cursor`；续读保留原范围并传 `--cursor`，避免长行丢失，不保存阅读状态。

选择证据时加 `--evidence` 并指定 `--end-line`，可用 `--quote-offset N` 预览生成的精确锚点。行号从 1 开始，范围最多 12 行；`quote_offset` 是所选行以换行符连接后的从 0 开始的字符偏移，通常填 0。工具从该位置取最多 320 字符，支持单行 JSON 后半段的声明；不自动寻找关键词、改范围或替模型补证。路径、范围、偏移和实际文件摘要均须有效。

P2→P3、P3→P4、P4→P5 换上下文时，由 reviewer 在回执中给出短交接，Runner 在对应运行记录生成 `handoff.md`。通常约 10 行，最多 20 个非空短行，正文建议不超过 1,200 字符。只写已定事项及理由、关键源码位置、待核实缺口和下一步入口；checkpoint 和来源标识由 Runner 自动附上。P5 读取 P4 交接、root 和已审直接父节点，完成时只留节点回执；子节点直接读取父产物，P6 汇总有效结论。

P2–P6 的普通回执另有必填 `reopen` 字段，无重开时填写 `null`。有新固定来源发现时可返回 `decision: "reopen"`，并给出 `reopen: {issue_id, target, reason, evidence}`；`evidence` 指向本轮 `source_evidence`，`issues` 仍须有具体范围、证据和恢复条件。`target` 只能回到上游：P2→P1，P3→P2/P1，P4→P3/P2/P1，P5/P6→P4/P3/P2/P1。外部任务要求变化用 `target: "external"`，保存需要任务方确认的明确条件。来源明确的转录错误可以引用已读过的源行：`issues.kind` 用 `transcription`，`scope` 定位错误产物，并以 `artifact_evidence` 引用当前错误字段。相同问题和固定源码范围只重开一次，换注释或短引文不能重新计次。

## 退修与裁决

普通 worker 的 `blocked` 是独立审阅请求。通过来源、版本和写入范围门禁的草稿仍交本阶段 reviewer，附上具体缺项；reviewer 独立判断通过、退修或真正基础阻塞。草稿检查退出 `1` 不妨碍 reviewer 返回 `revise`，该次意见照常计入实质退修并附检查诊断；检查退出 `2` 仍作为工具错误停止。完成和批准始终要求完整的阶段证据及必需检查退出 `0`。P1 的 `first_draft` 与初版门禁都按起始 checkpoint 是否仍为 starter 判断，初稿提交版本为 1；本轮先复制再填写不另算一次已接受修订。

每阶段单独累计实质退修，P5 按节点计数。reviewer 给出有证据、可执行的修改后，Runner 登记一次退修；低于上限时原 worker 修改、原 reviewer 再审。`continue`、提案格式修复和 CLI 暂时失败分别记账，均受原调用与时间预算约束。nonce 正确的普通字段格式错误与临时 CLI 错误各最多重试两次，保留原角色 UUID 及经过产物门禁的合法草稿；必须重新提交并完整验证才能推进。

缺失/陈旧 nonce、显式工具字段、非法来源或范围、越权和真实字节变化硬拒绝。引文由已选择的范围确定性生成，不存在改好错误引文便沿用未经复核 approve 的路径；范围不能支持结论时，模型必须重新核对自己的判断。裁决和签收不获得格式或传输重试；未确认终止的尝试始终保留进程 fence。没有模型响应的中断恢复仍按已登记进程、固定 attempt 和产物门禁保存可恢复草稿，不授予批准。

达到上限（默认第 5 次退修）后，原 reviewer 接手一次裁决：选定已有写法，或提交限于当前产物的补丁，由 Runner 在隔离副本应用并检查。满足阶段门槛就记录裁决和修订后推进；关键事实仍缺、补丁无法应用或必需检查仍不通过，就保留草稿、阻塞原因和恢复条件。裁决只有推进或 `blocked` 两个出口。回执额外包含 `adjudication: {action, checkpoint, patches}`：`action` 为 `select/patch/blocked`；select 只能选本阶段/节点已保存的候选，patch 基于当前 checkpoint，条目为 `{path, old, new}`，`old` 必须非空且唯一匹配。回执的产物摘要仍绑定裁决者实际读取的只读副本；Runner 在隔离副本处理并另存结果摘要。恢复旧稿会按实际变化保留或提高当前版本，不回滚权威状态、预算和计数。补丁越界、应用失败、回执错误或检查失败即阻塞，裁决不重试。正常审阅已核实必须等待外部输入时，可直接 `blocked`，无需耗尽退修。

如果 P6 裁决者亲自改了最终交付，Runner 再开一位独立 reviewer 做一次签收，覆盖最后改动及受影响结论，仅判 `ready` 或 `blocked`。签收不再启动返工、裁决或传输重试；原裁决和签收分别保存身份与结论。选择已有 worker 写法不额外触发此签收。普通 P6 worker 修订仍走正常审阅回合。

退修次数、已用时间、截止条件和裁决所需预算持久化；超时保存最后可恢复 checkpoint。恢复、重开计划继续原计数和预算，同一问题及同一证据只触发一次重开。分支阻塞只让其后代等待，独立分支继续；所有可继续的工作结束后仍有必需节点阻塞，整个算子以 `blocked` 结束并交付进展及具体原因。

## 并行、合入与计划修订

Runner 立即派发所有满足依赖的节点，不设固定 jobs 数或 worker/reviewer 槽。每个节点只有一个在途调用，先写再审；兄弟节点可同时写、同时审。节点通过并合入后立即解锁自己的子节点，不等整层完成。阻塞节点仅使其后代等待，独立兄弟继续。

每个分支在私有目录工作，只读已审 root 和所需祖先，只写自己的 YAML。检查器会扫描同目录，因此兄弟草稿必须隔离。Runner 串行合入同一算子的结果：核对尝试、输入指纹、阶段、节点、计划修订、checkpoint 和实际依赖摘要；将许可差异合入最新已审目录，复查目录与依赖，先保存不可变 checkpoint，再原子更新权威指针。重复回执只提交一次。仅兄弟先完成时可保留既有审阅；父节点、Contract 或选择条件变化时重新核对受影响结果。

P5 发现树或直接父节点需要改变，以新证据申请重开 P4，由 P4 worker/reviewer 修订并验收。要改选代表路径先回 P2，再验 P3；现有 root 的 Design 或选择条件变化要复核 P3；涉及共享 Contract 先回 P1，实际外部要求变更交任务方确认。重开期间，旧 P5 调用可以结束，返回结果暂存。P4 通过后，Runner 比较条目和真实依赖内容，保留未受影响的已审节点及合法在途结果；失效回执记录在 `stale_results`。旧调用只有确认终止才能释放对应 `active_attempts` 身份，随后才允许同节点的新调用。

计划删除节点会移出当前产物集合，历史节点账本仍保留。重新加入相同 ID 时，沿用原退修、调用、已用时间、截止条件和一次裁决记录，版本继续原稿。重开与节点重加不产生新的预算，也不重新授予已用裁决。P4 的审阅计划与影响记录保存在 `plan_history`，不写入交付索引。

P6 仅在当前计划节点集合与当前有效、已合入节点集合相同，且 P1–P4 放行仍有效时启动。完成数从该集合推导，草稿和失效的旧审阅不计入。

## 离线验证

从仓库根目录检查 CLI 入口并运行保留的检查器测试：

```sh
python3 -B schema_extract/extract.py --version
python3 -B schema_extract/extract.py --help
python3 -B -m unittest discover -s schema_extract/tools/tests
python3 -B -m unittest discover -s schema_template/tools/tests -t schema_template
```

这些命令不会调用模型。提取工具仅保留 nonce、只读字节、原响应绑定、长行证据与封存身份的小型回归；`tools/tests` 已排除出冻结清单。发布、恢复或回执门禁的临时探针使用仓库外的合成输入与运行目录，需要调用替身时显式指定受控的 `--codex-bin`。已提取的 YAML 使用 `schema_template/check_yaml.py --skip-hardware` 检查。
