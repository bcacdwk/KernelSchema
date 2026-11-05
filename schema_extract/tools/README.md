# Schema Extract：工具维护与回执协议

[Supervisor 手册](../README.md)说明命令、参数和恢复操作；六张阶段页定义提取方法与阶段权限。版本在 [__init__.py](__init__.py)，本页和执行器一起冻结到每个 run 的 `rules/`。

## 角色与会话

Runner 唯一写运行状态、派发、检查、合入和发布。worker 只修改本次许可产物；reviewer 独立读取 checkpoint 和固定源码，通常只读。裁决补丁也由 Runner 应用，reviewer 不直接改 workspace。

| 阶段 | 上下文 |
| --- | --- |
| P1 | worker/reviewer 分别新建，两个角色独立 |
| P2 | worker 精确续用 P1 会话；reviewer 新建阶段会话，从当前产物独立验收 |
| P3、P4 | 每阶段新建一对 |
| P5 | 每个节点新建一对，节点内返工续用原角色 |
| P6 | 全新 reviewer 先审，需要修改才启动新 worker；裁决改稿另开独立签收 |

[agent.py](agent.py)直接启动 `codex exec`，正常续写用 `codex exec resume UUID` 精确指向原角色。禁止用 `--last`、会话名或其他阶段/节点的 UUID。中断后须先确认终止，再以安全 checkpoint 重建上下文；`session_handoffs` 记录旧 UUID、后续尝试和新 UUID，已退休 UUID 不能复用。普通格式修复和已完成调用后的续写保留原角色会话。

审阅围绕源码语义与信息压缩，按[统一缺项边界](../../EXTRACT_SCHEMA.md#来源缺项与验收边界)及[源缺陷设计分流](../REVIEW.md#源实现缺陷与设计分流)判断：有据的阶段内容完整、忠实表达来源限制时可 approve 并携带 issues；已知事实写错或遗漏则 revise。blocked 须在现有 detail/scope 中说明缺失事实、无法确定的具体计算/数值/设计关系，以及为何现有事实和明确限制不足以继续。仅缺外部验收器定义、但不影响当前提取时记录 source_gap，不要求补齐后才推进；P6 仍复核缺口没有被用作保证。结构检查和精确引文不能代替语义判断。

[REVIEW.md](../REVIEW.md) 是各阶段共用的文档验收方法，随规则冻结并嵌入 reviewer 提示。Reviewer 初始输入移除历史 feedback/handoff、前序批准、旧稿与已审摘要，保留当前规范、产物、计划及固定源码入口；P4 按需提供当前节点文件，裁决保留完整候选。文档全文审阅覆盖本阶段已完成内容，源码核查再按变化和风险深入。

## 回执与短交接

模型通过 CLI 最后输出返回 `schema_extract.proposal.v1` JSON，字段以本次 `response.schema.json` 为准。`attempt_id` 是必须原样回传的本次随机 nonce；续会话也要更换，不另建回执文件。

| 模型负责 | 含义 |
| --- | --- |
| `decision/summary/conclusions`、`issues` | 模型填写的语义结论；`summary` 必须简短且非空，问题给出范围、影响、对应事实的证据与恢复条件 |
| `source_evidence` | category/path/line_start/line_end/quote_offset/note；note 解释所选范围支持的结论 |
| `artifact_evidence` | 同样的定位字段，无 category；引用实际读取的当前产物 |
| `checks/checks_note` | 只记亲自执行的命令和退出码；交 Runner 检查时用空列表并说明 |
| 阶段字段 | comparisons、plan、revision_scope/affected_nodes 等按当前页填写；无框架建议时 framework_candidates=[] |

[receipts.py](receipts.py)的 `normalize_response` 从已登记调用和真实字节生成 `receipt.v2`：身份、文件清单、实际差异、文件 SHA-256、固定来源及精确引文均由工具填写。模型不得提交这些工具字段；语义字段 `summary` 仍由模型填写。P5 只提交本节点 YAML，root/祖先可作为证据；其他阶段覆盖该次工作目录。P1–P4 局部目录合回完整树时，批准仍按阶段职责解释，audit 按 Reviewer 实际读取的 checkpoint 重建原回执。

原始 `response.json` 与 `result.response` 保持一致，规范化结果另存 `receipt.json`。reviewer 的整个 workspace 必须等于起始依赖字节，裁决结果另记摘要。批准绑定实际审过的内容；合法草稿只用于恢复。

complete/approve 须覆盖 [stages.py](stages.py)定义的 required_evidence 和 required_conclusions，每类至少引用一次绑定入口；调用链文件可补充。reviewer 的 artifact_evidence 须覆盖门禁给出的全部文件，P5 包括 root、直接父和当前节点。conclusions/issues 等的 evidence 从 0 索引本次 source_evidence。

revise/blocked 保留“至少一条具体且有依据的 issue”门槛。仅当前产物的提交格式问题 `submission_format` 或规范表达问题 `schema_conformance` 可以令 `issues[].evidence=[]`，以该问题所指文件的有效 `artifact_evidence` 支持：`scope` 必须独立写全相对文件名，可接 `:字段/行号`；其他文件、同名片段或旧目录别名不能借作依据。例如索引有节点而本次结构化 plan 为空，可将 scope 写为 `proposal.plan 与 dispatch.md 节点表` 并引用当前表格，无须补上不相干的源码引文。

该分流不免除 reviewer 的真实源码证据与全部阶段文件引用，也不改变 complete/approve、计划一致性或四项文档批准门槛。源码缺口、源码缺陷、转录、计算、数值及路径选择等问题仍使用有效源码索引；工具只核对问题类别、文件绑定和引文字节，不证明类别或解释正确。混合 issues 中任意一条有依据的完整问题仍可支撑退修，其他问题不会因此获得证据豁免。blocked 的恢复条件仍必填；reopen 继续遵循原源码依据及单次重开规则，不适用此处的产物证据替代。

Reviewer 的 approve 还须给出 `document_self_contained / document_layering / document_consistency / document_relevance` 四项非空结论，由当前产物的 artifact_evidence 支持，来源 evidence 可为 `[]`。缺任一结论或必需文件锚点都不能放行；这是范围与证据门禁，不是自动理解正文的语义判定。Worker 不承担这四项批准结论的填写义务。

P6 的 `tree` 结论描述当前目录的文件集合、父关系、链接与拓扑时，可以仅依据完整的 `artifact_evidence`，来源 `evidence` 用 `[]`。Runner 先核对实际文件集合与计划相同、全部文件已有有效产物引用，再执行完整树检查；选择、覆盖等源码语义结论仍须有效源码索引，全目录 schema 检查与独立语义审阅继续执行。

固定材料的范围读取使用 [reading.py](reading.py)：

```sh
python3 -B RULES/schema_extract/tools/reading.py --root SOURCE --path input.py --start-line 1 --end-line 80
```

返回真实行列片段、全文件摘要及 next_cursor。需要续读同一范围时保留起止行并传 `--cursor`，长行不会丢失；`--max-bytes` 默认 6000，可选 1024..24000。reader 不保存阅读状态。

选择证据时加 `--evidence --end-line N`，可用 `--quote-offset N` 预览锚点。行号从 1 开始，范围最多 12 行；quote_offset 是所选行以换行符连接后的字符偏移，从 0 开始。工具取最多 320 字符，不替模型寻找依据、改范围或补证。

P2/P3/P4 reviewer 批准时提供短 handoff：最多 20 个非空行、1200 字符，只写已定事项及理由、关键入口、缺口和下一步。Runner 保存对应 handoff.md 并附 checkpoint/来源身份；其他阶段无交接时用空串。

## 退修与裁决

P3/P5/P6 的完成、批准及退修由 Runner 自动执行冻结检查器，放行须退出 0。输入诊断为 1，工具/环境问题为 2。模型可按需自检，无需两个角色各自重复执行。worker 的 blocked 将合法草稿与缺项交独立 reviewer；reviewer 可对未完成草稿给出计数的 revise，不能把检查失败当作批准。

每阶段、每个 P5 节点独立累计实质退修。达到 revision_limit（默认 5）后，由原 reviewer 一次裁决，只有 approve 或 blocked 两个出口。格式修复不算实质退修；nonce 正确的普通格式错误和普通 CLI 失败各最多重试两次，所有尝试均消耗调用及时间预算。

P4 确定性核对表格与 plan，表格后允许先写必要说明，再写唯一的覆盖缺口节，不因段落位置消耗格式修复调用。表格内容、父关系、来源依据、该节正文和链接仍须通过门禁；表格差异报告首个不同的行列位置。

裁决额外返回 `adjudication: {action, checkpoint, patches}`。action 为 select/patch/blocked：select 只选本范围已有候选；patch 基于当前 checkpoint，条目为 `{path, old, new}`，old 非空且唯一匹配。Runner 在隔离副本应用、核对版本并检查；越权、无法应用、缺关键事实或检查失败即阻塞，裁决不重试。

P6 裁决亲自改稿后，另开全新 reviewer 一次签收，覆盖最后改动和受影响结论，只判 approve/blocked。选择已有 worker 写法不触发额外签收。裁决和签收均不享有格式或传输重试，恢复或重开不能重新授予。

[budget.py](budget.py)统一计算调用与检查余额。未使用的裁决和 P6 签收各预留一个调用及 `A = min(call_seconds, stage_seconds/4, operator_seconds/12)` 秒。每次普通 worker 派发另为后续 reviewer 留一个调用及 `R = min(call_seconds, (min(stage_seconds, operator_seconds)-A)/2)` 秒。模型与检查不能动用后续预留。

显式预算追加要求 `budget_extension.v1`：resume 请求绑定持久 launch ID，由冻结 Runner 在写锁内应用；同一 ID 只记账一次，每次新 resume --add-* 是新授权。默认 continue 保留原余额；显式 full-restart 先复制原账目，再按唯一 successor 应用追加。budget_extensions 保存来源、额度和前后截止时间，不改初始 config、已用时间或单次决定；普通恢复、hold 和自动重试不增加额度。具体参数见[手册](../README.md#命令与参数)。

## 并行、合入与计划修订

Runner 派发所有依赖就绪节点，不设固定槽位，每节点同时只有一个调用。私有工作目录只放已审 root、所需祖先和本节点，避免同目录检查混入兄弟草稿。节点通过并串行合入后立即解锁子节点，阻塞只传给后代。

合入核对调用、输入、阶段/节点代际、计划和真实父依赖，将许可差异合入最新已审目录，复查后先保存不可变 checkpoint，再更新状态。仅兄弟先完成不使合法结果过期；父节点、Contract 或选择条件改变时重新核对依赖，迟到结果记入 stale_results。重复回执只提交一次。

P2–P6 普通提案的 reopen 无请求时为 null；请求格式为 `{issue_id, target, reason, evidence}`。target 只能回上游：P2→P1，P3→P2/P1，P4→P3/P2/P1，P5/P6→P4/P3/P2/P1；external 表示任务要求变更，须外部确认。issues 仍要给具体范围和恢复条件。

树或选父回 P4，代表路径回 P2，root Design/when 回 P3，共享 Contract 回 P1。重开须有固定来源新发现；来源明确的转录错误或当前产物的规范表达错误可用已读源码，issues.kind 分别标为 transcription 或 schema_conformance，scope 定位错误产物并提供 artifact_evidence。规范错误还须在 detail 指明条款。同一问题和固定证据只重开一次，换 note/引文不能重计。

上游重开时，旧 P5 调用可结束并暂存；P4 批准后按计划和依赖判断保留或失效。未确认终止的调用仍保留 fence。删除节点保留账目，再加相同 ID 沿用原计数、版本和已用裁决。计划修订记录放 plan_history，不写入交付索引。

P6 准入要求当前计划节点集合等于有效已合入集合，且 P1–P4 放行仍有效。草稿或失效批准不计入完成数。

## 写入与完整性

[runner.py](runner.py)持写锁并串行提交；[store.py](store.py)管理状态和 checkpoint。`_StateGuard` 核对当前 Runner 线程最后写出的 state.json 字节。发现外改时保存 authority_changed、最后可信状态及全部在途身份；取消仍可确认终止。

| 边界 | 必须核验 |
| --- | --- |
| 创建、进入冻结执行器、恢复 | state 与输入锁一致，源码/规则快照完整 |
| 接收结果 | 活动调用身份、许可差异、nonce/回执、引文及实际字节 |
| 使用 checkpoint 或批准 | manifest、receipt、validated 记录及当前依赖 |
| 子节点、合入、P6 | 已审直接父、代际、计划及有效批准集合 |
| 发布/export | 独立 P6 放行、最终 checkpoint、待发布副本与交付回执 |

引用的对象在实际采用时核验，历史日志和旧 workspace 不参与热路径全扫。checkpoint 完整落盘后才更新指针。

恢复只对账 state 登记的 active/unconfirmed/unsettled 调用。unsettled 在准备时为 prepared，派发前持久化 dispatched，处理与计时都完成后才清除。只有 prepared 且无相反发射证据可认定未启动；发射后缺进程身份不能认定已终止。

Codex 调用通过 pipe gate 先持久登记启动与进程身份，再放行 payload；登记失败时关闭 gate。进程身份绑定 PID 与系统启动/进程启动 token，收尾保留 leader 身份以核对后代，未知归属继续保留 fence。执行器、检查器与诊断子进程的 cwd 位于 run/attempt，冻结 rules 仅供读取；不忽略 manifest 额外文件，也未修改宿主 core handler。

## 轻量观察投影

Store.save 提交权威 state 后，尽力原子写 summary.json。投影包含状态修订/文件版本戳和 `content_verified: false`，不能用于批准或调度。写投影失败不回滚 state；read_summary 发现陈旧/缺失时显示 stale/unknown，至多回读一次 state，不补写或深验。

inspect 每轮只取一次进程表，读取投影及已登记活跃调用的小记录和日志 stat。scan 另保存最新观察与最多 64 条变化，不读全历史、不自行恢复。冻结能力缺失保持未知。

工程交接优先使用失败原因绑定的 `attempt_id`。旧 P5 记录缺少该身份时只列状态文件与节点范围，不能用算子级最近调用代替失败节点；需要时再按节点读取原调用记录。

## 多任务控制与维护接续

supervisor.json 是唯一控制账本，保存启动、控制修订、hold、接续、观察、时钟和重试授权。Store.supervisor 的短事务不包含冻结、深验、等待或 exec；需要 run 写锁时先在控制锁外取得。

后台先持久登记目的地、run UUID、创建时间和启动令牌，再由 wrapper 认领进程身份并进入冻结执行器。语义切换须同时核对旧写锁及启动窗口，full-restart 的半成品 successor 也禁止启动或再次接续。

hold 先提交 active_hold，再追加暂停请求；派发授权和发布 rename 共用控制锁。释放与启动核对实际请求、控制修订和已消费 ID，以后续手动 pause/cancel 为准。只有 mode=resume 授权全任务普通失败恢复，hold/queued/retry 不解封其他范围；预算和已用决定继续生效。

`resume RUN_UUID --scope P5:NODE_ID` 使用既有 retries 账本，绑定失败 attempt/result、scope 代际、父依赖、执行绑定和控制修订；兄弟的状态修订不使请求过期。唯一 Runner 先结算再消费，失败范围之外的健康调用继续；失去原 Runner 的调用按原 created_at + timeout_seconds、阶段及算子截止时间收尾。旧记录缺少单次 timeout 时沿用原 config 上限，接管不重置时间或次数。

默认 continue 使用 [revision_control.py](revision_control.py) 的 request/prepare/apply：控制事务登记即建立派发与发布屏障；原 Runner 结算在途调用后准备计划，同一写锁内先封存 revisions/ID，再提交 semantic_revision 指针。重复请求和 apply 幂等，登记/排空/待评估计划返回 0 与 `delivered:false`。旧 successor 仅显式 `--full-restart --reason ... --checkpoint ...` 可用；它保存原 state/lock/稿件和账目，从 P1 重审。v1 兼容导入由 [legacy.py](legacy.py)核对原冻结执行器、源码、checkpoint 和终止证据。

[revisions.py](revisions.py) 的 `normalize_plan` 将独立 proposal 绑定完整目标 diff、issue/evidence 和 drain 后所有已有 scope/原稿/批准；未列清的已存在结论默认 review。每项 `scope/action/reason/issues/evidence/fields` 分开记录产物动作，keep 的批准采纳另需 `assessment` 声明内容未变、要求已覆盖、依赖相容，并完整覆盖变化路径和 issue。未完成草稿没有可采纳批准。`resolve_context(store, state, attempt=None)` 分别返回 source/config lock、semantic rules lock、executor lock 与身份；缺 semantic_revision 的历史对象继续按原锁解释，不能把新 rules 填进旧 fingerprint 的伪锁。

`Store.verify_approval_origin` 验原调用/receipt/Runner gate/不可变内容；`verify_effective_approval` 再验当前原生批准或封存的 adoption、阶段职责投影及实际依赖。默认 `verify_approval` 使用有效性门禁。P1 Contract、P2 选择、P3 Design/when、P4 计划分别投影，合法的后阶段增补不会反向改写旧批准；旧证据可 audit 不代表可以解锁新工作。

[maintenance.py](maintenance.py) 复用 Store.prepare_attempt、receipt、Checker 和原阶段 Reviewer。state.maintenance 保存真实 normal_cursor、完整 working_checkpoint 及有限 task；调用/回执/检查/批准绑定 semantic_revision、maintenance_task_id、purpose、maintenance_step。完整 review 候选先 Reviewer；patch/reextract 只开放明确文件/字段，再独立 verify。未完成 P1/P2 返回原 worker，P1–P4 局部工作目录合回完整树，P5 优先自己的稿件，只刷新只读祖先。依赖影响闭包产生复核候选，不等于重克隆后代。维护沿原 scope 计调用和时间，保留单次决定；相同 target/scope/issue/evidence/action 不因换 ID 获得新一轮。只有 paused/cancelled 且无未结调用可追加替代，封存旧维护并继承真实正常游标。

同一个 patch 步骤允许 Worker 提交合法 continue，保存稿件并沿原调用/时间余额续写；后续 verify 仍预留，实际失败或已结束步骤不能重新获得一次动作。只接受请求时未完成、无 export/已提交发布的 run；当前 P6 仍负责最终交付。新旧语义调用不并行，pause/cancel/hold 和原截止时间继续有效。能力 `semantic_revision.v1` 及新 state format 使旧冻结执行器拒绝接管；当前真实旧 run 未运行本轮迁移。

## 兼容差异与审阅记录

[compatibility.py](compatibility.py) 的 `build_report(store, run_id, checkpoint_id, target_repo=None)` 只读生成确定性报告。报告绑定原 state、inputs.lock、checkpoint、完整源码集合/绑定、原 config 和目标规则 manifest；目标文件集合与字节在读取前后复核，包含未提交修改。报告中的 `id` 是去掉自身后对象的规范 JSON SHA-256。`compatibility-report RUN_UUID --checkpoint DIGEST` 输出 `report`，可用 `--target-repo PATH` 选择目标工作树。

独立 Supervisor 阅读完整差异后单独保存审阅 JSON，不能把它加入冻结规则集合。下面是只有一项执行器修复时的形态；实际 `changes` 必须覆盖报告的全部差异，包含相同的 before/after SHA：

```json
{
  "format": "schema_extract.compatibility_review.v1",
  "report_id": "报告的完整 id",
  "supervisor": {
    "role": "Supervisor",
    "identity": "实际审阅者标识",
    "independent_review": true,
    "record": "独立审阅会话或保存记录的引用"
  },
  "decision": "compatible_runtime",
  "reason": "完整差异仅修复执行器行为，保留已批准结论的含义。",
  "preserves": {
    "field_meanings": true, "acceptance": true, "source_and_task": true,
    "approval_meanings": true, "dependency_meanings": true,
    "receipt_meanings": true, "budget_policy": true
  },
  "changes": [{
    "path": "schema_extract/tools/agent.py",
    "before_sha256": "报告中的原 SHA", "after_sha256": "报告中的目标 SHA",
    "classification": "runtime_fix",
    "reason": "具体说明修复影响及为何不会改变任何已有批准结论。",
    "evidence": ["具体代码位置、审阅推理和相应检查证据"]
  }]
}
```

每个文件按实际影响分类：`runtime_fix`、`usage_documentation` 可提供运行兼容依据；`paths/design/parent_selection` 分别提示 P2/P3/P4 及依赖范围；`contract/acceptance/source_or_task/policy/checker` 从共享前提考虑，`unknown` 保持信息不足。这些兼容报告建议供连续修订计划评估，不直接触发全量重写；具体保留、复核、修补或局部重提取由独立计划绑定证据。同一文件含多类变化时在理由中说明全部影响。README 须逐段区分命令说明和政策变化；文件名、扩展名、相邻版本或测试通过不能代替语义理由。真实来源/任务变化不能借维护放行。

`verify_review(report, review)` 返回分别列出的产物、批准和执行器影响及 `review_sha256`，拒绝遗漏、额外变更、摘要不符和以 compatible 覆盖语义重审；CLI 的 `--review FILE.json` 调用同一接口。身份和独立声明只是审阅来源记录，API 不认证人或把两个不同名字当成独立性的证明；采用方必须取得实际独立 Supervisor 的明确审阅。核验记录本身不检查旧批准是否有效，也不授予执行权限，原批准仍走既有统一 verifier。

接入方在冻结前调用 `verify_report(store, report, target_repo)`，冻结后调用 `verify_report(store, report, target_lock=frozen_lock)` 再调用 `verify_review`；两次都要求原 state/checkpoint 仍匹配。冻结目标须严格匹配报告的完整 manifest，并保留源码字节/绑定、原 config 和检查政策。报告和审阅记录应随执行器绑定保存；本接口不改原 run、inputs.lock 或批准身份，也不复制跨 run 批准。任何额外变化都需重新报告和审阅。

[runtime.py](runtime.py) 的 `upgrade_runtime(store, run_id, report, review, target_repo=...)` 在原 run 写锁内冻结目标并重新核对报告，最后以短控制事务提交唯一状态指针；不启动模型、不接收旧失败结果、不追加额度。`state.execution_bindings` 按顺序保存历史 ID，`execution_binding` 指向最新记录。升级状态采用 `schema_extract.run.v2.runtime.v1`，使不认识绑定的旧冻结入口拒绝写入；新 Store 同时读取旧 v2。原 `inputs.lock.json`、receipt、checkpoint 和批准均不重写。

每个 `runs/RUN/runtime/BINDING/` 封存 `binding.json`、`before.json`、`report.json`、`review.json`、`inputs.lock.json` 和 `target.lock.json`。绑定记录包含前驱、输入/目标/报告/审阅摘要及登记前各 attempt 摘要；未被 state 指针引用的中断准备目录没有执行权限。连续升级以当时有效语义锁为基准，历史报告按封存的 before-state 核验，完整保留各次修订和实际执行代码。已用会话通过 `session_handoffs` 退休，后续角色从原安全 checkpoint 新建上下文，不撤销已有有效批准。

`verify_execution(store, state, executor=False, attempt=None, attempts=None, evidence=None)` 是启动、调用、批准/P6、发布和深审计的共用校验入口，返回精确执行 lock；`executor=True` 另验证当前 Python 来源。封存报告始终对照封存的 before-state 核验，不对比后来推进的 state 摘要。一次调用边界内复用校验结果，祖先遍历不重复扫描快照。新 attempt 写入实际 `execution_binding`，历史 attempt 由封存摘要关联；领域规范、检查器、提示材料和来源分别由 resolve_context 选择。目标中所有运行代码及依赖仍需完整独立审阅，不能用文件路径自动豁免语义变化。

目标版本、状态格式和 `EXECUTOR_CAPABILITIES` 从目标冻结 Python 的字面量声明读取，不执行目标代码来获取元数据，也不沿用旧 executor 的能力表。新建与兼容升级使用同一能力来源；目标缺少所需能力时相应恢复入口仍拒绝。

`frozen_run_invocation(store, state, arguments)` 返回绑定执行器的绝对入口、冻结 PYTHONPATH 和 run 目录 cwd；后台启动也使用该入口，并在控制锁内拒绝过时的绑定观察。新发布/export 的交付回执写明执行历史和 P6 批准所属绑定；已发布的同一 checkpoint 保留原交付回执。`audit --verify/--archive` 复用绑定校验，并封存迁移证据、各目标快照及报告选定 checkpoint。普通 summary/audit 观察只展示已记录身份，不冒充深验证。

## 时钟与单轮提交

[wakeup.py](wakeup.py)在控制账本保存 clock，后台 tick 做有界观察、已授权范围恢复及 queue 投递，观察不等待 ack。单实例进程锁的 fd 传给子进程，父进程只关闭 fd；不能用 LOCK_UN 释放子进程仍需持有的锁。进程身份观察和 queue 等待均在控制锁外，存活证据不足时显示 unknown。

reminder 绑定 generation/thread/number/due 的稳定 ID。ack 在锁内校验身份、启用和到期条件；scan 在锁外观察，再于同一短事务提交观察、完成号和下一次时间。提交前退出可续做，提交后重放同 ID 不再推进。晚到 queue 响应只补消息身份，不能回退已接受/完成状态。

发送前记次数；最后一次尝试为 uncertain/sending 时至少等 120 秒，为 queued 且未 ack 时等 1800 秒，同线程沿用同 ID 补送。tick、后台等待与 status 共用下一次发送时间计算，保留首次入队事实；所有发送共用总上限 3 次，重建时钟不退款。ack 或耗尽后不再发送，耗尽也不推进周期；status 显示下一次尝试、剩余次数和 ack 后未完成诊断。发送前重核当前提醒及 ack，晚响应只补事实。stop 先停用并推进代际，再于锁外终止时钟，旧提醒及未提交扫描失效。操作与实际投递边界见[提醒手册](../README.md#同线程持久提醒)。

历史 HTTP 403 的配置原因已由用户确认：主/子 Agent API key 曾不同步，后续已同步修正。本次有界补送作通用兜底，不将该配置故障继续列为待办，也不保证绝对投递或处理完成。

自动重试只接受已终止的 EX_TEMPFAIL 75，并重核 failure/scope/attempt/result 摘要、控制修订和剩余预算。授权绑定时钟代际，按 lineage 累计；stop、手动控制或新代际不能延续旧扫描授权，已用次数不退款。

## Usage、时间与证据封存

[audit.py](audit.py)在 CLI 关闭后一次流式解析完整事件行。result 保存提供者 usage 原值、字段来源、行号/字节锚点、日志摘要及完整行边界；未换行末行不算完成事件。

| 记录 | 口径 |
| --- | --- |
| result.usage / trace | reported/partial/unknown；事件存在不代表调用成功或终止确认 |
| usage-observation.json | 独立的根会话增量证据；root exact/partial/unavailable，整体最多 partial |
| attempt_usage | 按 origin_run_id/attempt_id 记录角色、会话、证据及 elapsed_seconds 高水位 |
| processing / finished | Runner 检查与处理时间；崩溃开放区间单记 estimated_seconds |
| session_handoffs | 会话退休及后续尝试，按原 run 定位 |

account 只计新增耗时差额；恢复不重复累加或把缺值改为零。首次绑定的不可变证据摘要不能覆盖，active/unconfirmed/unsettled 都结清才标 settled。result 关闭后不再写入后续检查时间。

旧 result.usage v1 不变。[observation.py](observation.py)只按本次 UUID 有界读取 native ledger，记录 CLI 版本、文件/事件身份及 resume 前基线；新会话的零基线也须由首个累计与本次计数一致证明。单 turn 的 native 增量必须与最终输出累计一致，重复 counter 不重复计；缺基线、截断、回退和来源冲突使 cursor 无效，观察失败不改变模型结果。下一次调用重新取基线，避免重放用量。

`observation_totals` 统计本 run 已登记的 CLI attempts（含继承来源），按 origin_run_id/attempt_id 去重，再拒绝同会话的重叠计数区间。只加有据根会话增量，分列 exact/partial/unknown；`known_attempts=exact_attempts+partial_attempts` 保持兼容。非空范围全部 exact 才给出 root_status=exact，有据小计但不完整为 partial，无可加增量为 unavailable。cached input 已含于 input；旧 result/receipt 不补写观察值，继承账目不再叠加 predecessor 总数。

2.4.1 的可选 delegation 投影随 observation 摘要绑定，invocation/result 显式记录 `delegation_policy=forbidden`，旧字段缺失为 unknown。已有 native 调用范围仅提取已识别 response_item.function_call 和 item_completed SubAgentActivity/CollabAgentToolCall 的工具/活动类型、事件锚点及关联 UUID；不保留参数或正文，不读取子会话和用户历史。summary 只保留策略、状态、短原因及标记/关联线程数量，完整 UUID 锚点留在由 SHA-256 绑定的 usage-observation.json，不把事件列表复制进账本。observed/not_observed_in_range/unavailable 分列委派证据；格式不识别或范围不完整保持 unavailable，委派观察失败不抹掉已证明的根增量。禁止委派、未观察到标记、关联线程数量及 parent_thread_id 均不证明完整子会话集合。子会话 token 与完整调用用量仍未知；`cost=null` 的原因为无可靠计价/账单来源，不以现价反推实付。旧 observation v1 与 summary 原字节/摘要语义不变，旧冻结执行器须经显式运行层升级才会采集未来调用的新字段。

audit 分列 wall/CLI/check-processing 与 lifecycle 等待，文本及 JSON 明示 `budget_policy=wall_clock_deadline` 和等待计入预算；pause/hold/排队/退避按已知区间并集计算，缺边界保持未知。范围等待可能与兄弟工作并行，不能相减推算纯模型时间；usage observation 耗时是 CLI 时间的子集。原 config、墙钟截止、预算预留及追加语义不变，观察不赠送或退还额度。

verify 复用快照、checkpoint 和引文门禁：提案按原始输入/产物重新规范化后与 receipt 比对，reviewer 绑定实际读过的起始 checkpoint，裁决结果另验。核对 response.json 与 result.response 一致，不启动模型或检查器。

archive 在同一选定范围复验后收集证据，按 SHA-256 去重。manifest 身份只取稳定范围和实际对象，不含观察时间；已有包须同清单且对象完整才复用。缺失、活动或未结范围不能冒充完整。封存不扫 provider 私有历史、不删除 runtime，也不提供执行恢复授权。

## 离线验证

```sh
python3 -B schema_extract/extract.py --version
python3 -B schema_extract/extract.py --help
python3 -B -m unittest discover -s schema_extract/tools/tests
python3 -B -m unittest discover -s schema_template/tools/tests -t schema_template
```

测试使用本地合成输入和可执行替身，不调用真实模型。`tools/tests` 不进入冻结清单。已提取的 YAML 用 `schema_template/check_yaml.py --skip-hardware` 检查；提取质量仍须独立语义审阅。

Reviewer 的[语义回归集](../review_cases/README.md)另用真实 CLI 独立调用，记录规范与提示身份、结果和用量，测试合法文本及结构检查无法识别的表达错误。它与上述离线测试分别报告，不能将模拟回执通过当作 Reviewer 已识别语义问题。
