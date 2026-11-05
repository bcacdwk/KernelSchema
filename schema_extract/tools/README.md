# Schema Extract：工具维护与回执协议

[Supervisor 手册](../README.md)说明命令、参数和恢复操作；[REVIEW.md](../REVIEW.md)定义共同审阅方法，六张阶段页定义提取内容与权限。版本在 [__init__.py](__init__.py)，本页随执行器冻结到 run 的 `rules/`。

## 角色与会话

Runner 唯一写运行状态、派发、检查、合入和发布。worker 修改许可产物，reviewer 独立读取当前 checkpoint 和固定源码；裁决补丁也由 Runner 应用。Supervisor 的工程维护与 extract 两角色的职责边界见[角色说明](../README.md#角色与会话)。

| 常规阶段 | 上下文 |
| --- | --- |
| P1 | worker/reviewer 各自新建，两角色独立 |
| P2 | worker 精确续用 P1；reviewer 新建阶段会话 |
| P3、P4 | 每阶段新建一对 |
| P5 | 每节点新建一对，节点内返工续用原角色 |
| P6 | 全新 reviewer 先审，需修改才启动 worker；裁决改稿另开独立签收 |

[agent.py](agent.py)直接调用 `codex exec`，续写用 `codex exec resume UUID` 精确指向原角色。中断后先确认终止，再从安全 checkpoint 重建上下文；`session_handoffs` 记录退休 UUID 与后续尝试，退休会话不复用。正常续写、普通格式修复和已结束调用的普通恢复按已登记会话处理。

Reviewer 初始输入保留当前规范、产物、计划和固定源码入口，移除前序批准、旧稿及历史 feedback/handoff；维护另提供本轮问题和字段边界。先独立理解本阶段已完成的全文，再核对源码。完整且忠实表达限制的内容可以携带 issues 获批；未完成草稿和已知转录错误按[共同审阅方法](../REVIEW.md)处理。

## 回执与短交接

模型通过 CLI 最后输出返回 `schema_extract.proposal.v1` JSON，以本次 `response.schema.json` 为准。`attempt_id` 是须原样返回的本次随机 nonce，续会话也会更换；无需另写回执文件。

| 模型负责 | 内容 |
| --- | --- |
| `decision/summary/conclusions`、`issues` | 语义决定、简短非空 summary、具体范围/影响/证据及恢复条件 |
| `source_evidence` | category/path/line_start/line_end/quote_offset/note；note 说明证据支持的结论 |
| `artifact_evidence` | 同样的定位字段，无 category；引用实际读取的当前产物 |
| `checks/checks_note` | 仅记亲自执行的命令与退出码；交 Runner 检查时使用空列表并说明 |
| 阶段字段 | comparisons、plan、revision_scope/affected_nodes 等按阶段页填写；无框架建议时 `framework_candidates=[]` |

[receipts.py](receipts.py)的 `normalize_response` 根据已登记调用和实际字节生成 `receipt.v2`，填写身份、文件清单、差异、SHA-256 和精确引文。模型只填写提案字段。P5 只提交本节点 YAML，root/祖先可作为证据；其他阶段覆盖该次工作目录。原 `response.json` 与 `result.response` 保持一致，规范化结果另存 `receipt.json`。Reviewer workspace 与起始依赖字节一致，裁决产物另记摘要；合法草稿只提供恢复进度。

worker 的 `decision=continue` 保存合法稿件，在原阶段/节点继续写作，消耗原调用和时间预算，不表示完成或批准。它与 CLI `continue RUN_UUID` 的连续修订请求无关；维护 patch 中也适用，见[维护协议](#多任务控制与维护接续)。

complete/approve 须覆盖 [stages.py](stages.py)中的 required_evidence 和 required_conclusions，每类至少引用一次绑定入口，调用链文件可补充。conclusions/issues 的 evidence 从 0 索引本次 source_evidence。reviewer 的 artifact_evidence 覆盖门禁列出的全部文件，P5 包括 root、直接父和当前节点。

revise/blocked 至少有一条具体且有依据的 issue。仅提交格式 `submission_format` 和规范表达 `schema_conformance` 可用有效 artifact_evidence 代替该 issue 的源码索引，`scope` 须独立写全相对文件名，可附字段/行号。例如结构化 plan 漏掉当前索引节点，可引用 `dispatch.md` 中的对应表格。其他源码、计算、数值、转录与路径问题仍给有效源码索引；blocked 写明缺失事实、不能确定的具体关系及恢复条件。类别与引文门禁不代替语义判断，也不豁免 Reviewer 的阶段源码证据及完整文件引用。

Reviewer approve 另需 `document_self_contained/document_layering/document_consistency/document_relevance` 四项非空结论，由当前产物锚点支持，来源 evidence 可为 `[]`；worker 无此填写义务。P6 的 `tree` 若只描述文件集合、父关系、链接及拓扑，也可仅依据全部产物引用；选择与覆盖等源码语义结论仍需来源索引。

固定材料用 [reading.py](reading.py)按范围读取：

```sh
python3 -B EXECUTOR/schema_extract/tools/reading.py --root SOURCE --path input.py --start-line 1 --end-line 80
```

EXECUTOR/SOURCE 分别使用调用列出的 executor_root/source_root；产物取证则将 `--root` 指向当前 workspace。返回真实片段、文件摘要和 next_cursor；续读保持起止行并传 `--cursor`，长行可续读。`--max-bytes` 默认 6000，可选 1024..24000。取证加 `--evidence --end-line N`，可用 `--quote-offset N` 预览：行号从 1 开始，最多 12 行；offset 是所选行以换行符连接后的字符偏移，从 0 开始，工具最多取 320 字符。工具不替模型选依据或补证。

P2/P3/P4 reviewer 批准时提供最多 20 个非空行、1200 字符的 handoff，仅写已定事项、理由、关键入口、缺口和下一步。Runner 保存 `handoff.md` 并附身份；其他阶段无交接时用空串。

## 退修与裁决

常规流程的 P3/P5/P6 完成、批准及退修由 Runner 执行冻结检查器，放行须退出 0；1 是产物诊断，2 是工具/环境问题。模型可按需自检，无需两角色重复运行。worker blocked 将合法草稿与缺项交 Reviewer；Reviewer 可给计数的 revise，未完成或检查失败的稿件不能获批。

每阶段、每个 P5 节点独立累计实质退修。达到 revision_limit（默认 5）后由原 reviewer 一次裁决，只能 approve 或 blocked。格式修复不计实质退修；nonce 正确的普通格式错误和普通 CLI 失败各最多重试两次，均消耗调用及时间。

裁决返回 `adjudication: {action, checkpoint, patches}`。action 为 select/patch/blocked：select 只选本范围已有候选；patch 以当前 checkpoint 为基准，条目 `{path, old, new}` 的 old 非空且唯一匹配。Runner 在隔离副本应用并核对权限、版本和检查；失败即阻塞，裁决不重试。P6 裁决改稿后由全新 reviewer 一次签收，覆盖改动和受影响结论，仅 approve/blocked；选择已有 worker 候选不触发签收。裁决、签收均无格式或传输重试，已用决定不会因恢复、重开或追加预算而重置。

[budget.py](budget.py)统一计算调用和检查余额。未使用的裁决、P6 签收各预留一个调用及 `A = min(call_seconds, stage_seconds/4, operator_seconds/12)` 秒；普通 worker 派发另为后续 reviewer 留一个调用及 `R = min(call_seconds, (min(stage_seconds, operator_seconds)-A)/2)` 秒。模型与检查使用扣除这些预留后的余额。

显式追加由支持 `budget_extension.v1` 的冻结 Runner 在写锁内按持久 launch ID 记账一次；full-restart 继承账目后按唯一 successor 追加。原 config、已用额度及单次决定保留，墙钟在排队、暂停和维护等待中继续。参数和重复授权语义统一见[命令与参数](../README.md#命令与参数)。

## 并行、合入与计划修订

Runner 派发所有依赖就绪节点，不设固定槽位，每节点同时一个调用。节点私有目录只放已审 root、所需祖先和本节点。首次创作从已审直接父复制；续写、恢复和维护优先保留本节点自己的稿件，只刷新只读祖先。

合入核对调用、输入、阶段/节点代际、计划和真实父依赖，将许可差异合入最新已审目录，复查并保存不可变 checkpoint 后才更新状态。兄弟先完成不使结果过期；父、Contract 或选择条件变化时重核依赖，迟到失效结果写入 `stale_results`。节点有效合入后解锁子节点，阻塞仅传给后代。

常规 P2–P6 的 `reopen` 默认 null，请求为 `{issue_id, target, reason, evidence}`。树/选父回 P4，root Design/when 回 P3，代表路径回 P2，共享 Contract 回 P1；只能回到当前阶段上游。`external` 表示任务要求变更，须外部确认。

重开依据固定来源新发现；来源明确的转录错误或当前产物的规范表达错误可用已读源码，issues.kind 分别为 transcription/schema_conformance，scope 与 artifact_evidence 定位错误，规范错误另指明条款。同一问题和固定证据只重开一次。上游重开时旧 P5 调用可结束并暂存，P4 批准后按依赖判定保留或失效；删除后再加同 ID 仍沿用原计数、版本与已用裁决，历史记录保存在 `plan_history`。

P4 门禁核对表格与结构化 plan、父关系、来源、覆盖缺口和链接；必要说明可放表格与唯一覆盖缺口节之间。P6 准入要求当前计划集合等于有效已合入集合，且 P1–P4 批准仍有效，草稿不计完成数。

## 写入与完整性

[runner.py](runner.py)持写锁串行提交，[store.py](store.py)管理状态和 checkpoint。`_StateGuard` 核对当前 Runner 最后写出的 state 字节；发现外改则保留 `authority_changed`、最后可信状态及在途身份。

| 采用边界 | 核验内容 |
| --- | --- |
| 创建、启动、恢复 | state 与输入锁、源码/规则快照、实际执行绑定 |
| 接收结果 | 活动调用、许可差异、nonce、回执、引文与字节 |
| checkpoint、批准与合入 | manifest、receipt、validated、代际、计划及当前依赖 |
| 发布/export | 当前 P6 放行、最终 checkpoint、待发布副本与交付回执 |

对象在实际采用时验证，历史日志和旧 workspace 不参与每次热路径全扫；checkpoint 完整落盘后才更新指针。

恢复只对账 state 登记的 active/unconfirmed/unsettled。unsettled 从 prepared 到 dispatched，结果处理与计时完成后才清除；只有 prepared 且无相反发射证据能认定未启动。Codex 调用由 pipe gate 在启动与进程身份持久登记后放行，登记失败关闭 gate。进程身份含 PID、系统及进程启动 token，保留 leader 身份核对后代；发射后身份缺失或归属不明继续保留 fence。执行器、检查器及诊断进程的 cwd 位于 run/attempt，冻结 rules 只读。

## 轻量观察投影

Store.save 提交权威 state 后尽力原子写 `summary.json`，包含修订/文件版本戳及 `content_verified: false`，不用于批准或调度。投影写失败不回滚 state；read_summary 对陈旧/缺失显示 stale/unknown，最多回读一次 state。

inspect 每轮只取一次进程表，读取投影、登记的活跃调用小记录和日志 stat；scan 另存最新观察与最多 64 条变化，默认不恢复任务。工程交接优先定位失败 reason 绑定的 `attempt_id`；旧 P5 缺该身份时只列节点与状态文件，再按节点查证。

## 多任务控制与维护接续

`supervisor.json` 是启动、控制、hold、修订、观察、时钟和重试的唯一控制账本。短控制事务不包裹冻结、深验、等待或 exec；run 写锁在控制锁外取得。后台启动先登记 UUID 和令牌，再认领进程身份。派发授权与发布 rename 共用控制锁，后来的 pause/cancel/hold 优先。

普通恢复、单范围恢复、运行层升级和语义修订的操作条件见[恢复与交付](../README.md#恢复与交付)。`resume --scope` 复用 retries 账本，绑定失败 attempt/result、scope 代际、父依赖、执行绑定和控制修订；兄弟状态变化不使它过期。接管使用原 `created_at + timeout_seconds` 及阶段/算子截止，旧记录无单次 timeout 则用原 config。

[revision_control.py](revision_control.py)执行 request → drain → prepare → apply。请求入账即建立派发与发布屏障；旧调用在自己的上下文中结算，未知终止保持 fence；排空后绑定实际计划。先封存 `revisions/ID`，再提交 `semantic_revision` 指针；重复请求和 apply 幂等。仅请求时未完成且无 export/已提交发布的 run 接受修订；显式 full-restart 则创建封存原账目的 P1 successor。

[revisions.py](revisions.py)的 `resolve_context` 分别返回原 source/config lock、当前 semantic rules lock、实际 executor lock 及身份。历史 attempt 选择登记时的上下文，原锁、回执及批准不回写。缺 `semantic_revision` 的历史对象按原语义解释；目标能力不足的冻结执行器拒绝接管。

`normalize_plan` 将独立 proposal 绑定完整目标差异、issue/evidence 及 drain 后的所有已有 scope、稿件与批准。Supervisor 从 prepare 返回的计划评估，并保存独立 proposal（或包含该 proposal 的准备结果），核心字段为：

| 字段 | 要求 |
| --- | --- |
| `reason`、`reviewer` | 非空理由；reviewer.role 为 Supervisor/Reviewer，identity/record 非空，independent_review=true；声明须对应真实独立评估 |
| `issues/evidence` | 本次问题及依据，可包含规则未改变的具体语义问题 |
| `scopes[]` | 每项独立填写 scope/action/reason/issues/evidence/fields；scope 如 P3:root、P5:node |
| `fields[]` | patch/reextract 的明确写入范围，如 `{"path":"root.yaml","fields":["design.work_schedule"]}`；dispatch.md 使用 `["*"]` |
| `assessment` | 仅采纳旧批准时使用：decision=adopt，unchanged_content/requirements_covered/dependencies_compatible 均为 true，并附 reason/evidence、覆盖全部差异路径的 covered_changes 和全部相关 issue 的 covered_issues |

产物动作与批准适用性分开决定：

| action | 执行与批准 |
| --- | --- |
| keep | 保留稿件；已有批准还须完整独立 assessment，否则转 review；未获批准的草稿不因此获批 |
| review | 完整候选先由原阶段的新 Reviewer 复核，符合即完成该范围 |
| patch / reextract | 保留旧稿，按有据的文件/字段范围修补或局部重提取，再独立 verify |

已有范围未列清时默认 review，未开始范围 keep。`Store.verify_approval_origin` 验原调用、receipt、Runner gate 与不可变内容；`verify_effective_approval` 再验当前批准或封存 adoption、阶段职责投影及实际依赖。执行准入默认用 effective：P1 Contract、P2 选择、P3 Design/when、P4 计划分别投影，后阶段合法增补不改写旧批准。可审计的旧证据不自动具备当前解锁权限。

[maintenance.py](maintenance.py)保存真实 `normal_cursor`、完整 `working_checkpoint` 和有限 task，沿用 prepare_attempt、回执、Checker 及原阶段 Reviewer。调用/检查/批准绑定 semantic_revision、maintenance_task_id、purpose 和 maintenance_step。未完成稿件继续正常 authoring，P6 继续正常终审；维护不能批准残稿。P1–P4 局部目录合回完整树；P5 保留自己的正文，只刷新祖先。影响闭包产生复核候选，有依赖缺口的任务等待上游完成。

维护动作为 `review →（确需修改时 patch → verify）` 或 `patch/reextract → verify`。patch 的合法 `decision=continue` 保存进度并续写同一步骤，后续独立 verify 仍有预留；complete 才进入 verify。实际失败或结束的动作不会因恢复或换 ID 获得新一轮。维护不使用领域 reopen、裁决或签收通道，所有实际调用与时间计入原 scope，保留退修、P6 starts 和已用单次决定。

待办结束后按当前有效依赖恢复实际剩余工作，最终仍需当前 P6。已有维护只能在结束后，或 paused/cancelled 且全部调用结清时，由新请求替代；封存旧维护、继承真实正常游标，并按 target/scope/issue/evidence/action 去重。新旧语义调用不并行，原控制和预算始终有效。

## 兼容差异与审阅记录

[compatibility.py](compatibility.py)的 `build_report` 只读比较当前有效规则与目标完整文件集合，绑定原 state、inputs.lock、checkpoint、源码/角色和 config。目标读取前后复核实际字节，包含未提交修改；报告 id 是去掉自身后的规范 JSON SHA-256。

独立 Supervisor 阅读全部差异后单独保存审阅 JSON，不把审阅文件加入冻结规则集合。单项执行器修复的形态如下，实际 changes 必须覆盖报告中的全部新增、删除和修改及其 before/after SHA：

```json
{
  "format": "schema_extract.compatibility_review.v1",
  "report_id": "报告的完整 id",
  "supervisor": {
    "role": "Supervisor", "identity": "实际审阅者",
    "independent_review": true, "record": "独立审阅记录引用"
  },
  "decision": "compatible_runtime",
  "reason": "完整差异只修复执行器行为，保留既有结论含义。",
  "preserves": {
    "field_meanings": true, "acceptance": true, "source_and_task": true,
    "approval_meanings": true, "dependency_meanings": true,
    "receipt_meanings": true, "budget_policy": true
  },
  "changes": [{
    "path": "schema_extract/tools/agent.py",
    "before_sha256": "报告中的原 SHA", "after_sha256": "报告中的目标 SHA",
    "classification": "runtime_fix",
    "reason": "具体修复及其兼容理由",
    "evidence": ["代码位置、审阅判断与验证依据"]
  }]
}
```

`runtime_fix/usage_documentation` 可支持兼容登记；`paths/design/parent_selection` 提示 P2/P3/P4 及依赖范围，`contract/acceptance/source_or_task/policy/checker` 涉及共享前提，`unknown` 保持待评估。报告建议不直接触发重写。一个文件含多类变化时说明全部影响；README 逐段区分用法与政策，文件名、版本或测试通过不代替语义判断。真实来源/任务变化另建提取。

`verify_review` 拒绝差异遗漏、额外变化、摘要不符及用 compatible 覆盖语义变化，输出产物、批准和执行器影响及 review_sha256。身份声明只是来源记录，不能证明实际独立性，也不授予运行权限。冻结前后分别 `verify_report`，确保 state/checkpoint 与完整目标 manifest 仍匹配；任何额外变化须重新报告和审阅。

[runtime.py](runtime.py)的 `upgrade_runtime` 在原 run 写锁内冻结并核验目标，再用短控制事务提交执行绑定。`runtime/BINDING/` 封存 binding、before-state、报告、审阅及原/目标锁；未被状态指针引用的准备目录无执行权。保留原 inputs.lock、checkpoint、批准与账目，退休旧角色会话。连续升级以当时有效语义规则评估，历史报告按封存的 before-state 验证。

`verify_execution` 是启动、调用、批准、发布和深审计的共用入口，返回精确执行锁；`executor=True` 另验当前 Python 来源。目标版本、状态格式及 `EXECUTOR_CAPABILITIES` 从冻结 Python 字面量读取。`frozen_run_invocation` 给出绝对入口、冻结 PYTHONPATH 和 run cwd，后台启动同样核对绑定。新交付回执记录执行历史及 P6 所属绑定，已发布同一 checkpoint 保留原回执；audit 封存相关证据。

## 时钟与单轮提交

[wakeup.py](wakeup.py)在控制账本保存 clock。tick 的观察、已授权范围恢复与 queue 投递不依赖提醒 ack。单实例进程锁 fd 传给子进程，父进程仅关闭自己的 fd；进程观察及 queue 等待在控制锁外，证据不足为 unknown。

reminder 以 generation/thread/number/due 绑定稳定 ID。ack 在锁内核验身份、启用和到期；scan 在锁外观察，随后短事务一次提交观察、完成号及下次时间。提交前中断可续做，提交后重放不再推进；晚到 queue 响应只补事实。发送前持久计次并重核当前提醒与 ack，重建不退还发送次数；补送间隔和上限统一见[提醒操作](../README.md#同线程持久提醒)。stop 先停用并推进代际，再在锁外终止时钟。

自动重试仅接受确认终止的 EX_TEMPFAIL 75，重核 failure/scope/attempt/result 摘要、控制修订及预算。授权绑定时钟代际并按 lineage 累计；stop、手动控制或新代际使旧授权失效，已用次数保留。

## Usage、时间与证据封存

[audit.py](audit.py)在 CLI 关闭后流式解析完整事件行，result 保存提供者 usage 原值、字段来源、行号/字节锚点、日志摘要及完整行边界；未换行末行不视为完成事件。

| 记录 | 口径 |
| --- | --- |
| result.usage / trace | reported/partial/unknown；有事件不等于调用成功或已确认终止 |
| usage-observation.json | 根会话增量证据；root exact/partial/unavailable，整体调用最多 partial |
| attempt_usage | 按 origin_run_id/attempt_id 记录角色、会话、证据与 elapsed_seconds 高水位 |
| processing / finished | Runner 检查处理时间；崩溃开放区间另记 estimated_seconds |
| session_handoffs | 退休会话与后续尝试，按原 run 定位 |

account 只计新增耗时差额，首次绑定的不可变证据不覆盖；active/unconfirmed/unsettled 结清后才标 settled。关闭 result 后不把后续检查时间写回。

[observation.py](observation.py)按本次 UUID 有界读取 native ledger，记录 CLI 版本、事件身份和调用前基线。新会话零基线也须首个累计与本次计数一致；单 turn native 增量须与最终输出累计一致。缺基线、截断、回退或冲突使 cursor 无效，下一调用重新取基线；观察失败不改模型结果。

`observation_totals` 覆盖本 run 登记的 CLI attempts（含继承来源），按 origin_run_id/attempt_id 去重，并拒绝同会话重叠计数区间。仅累加有据根增量，exact/partial/unknown 分列，`known_attempts=exact_attempts+partial_attempts`。非空范围全部 exact 才有 `root_status=exact`；有据小计但不完整为 partial，无可加增量为 unavailable。cached input 是 input 子集；继承账目不再叠加 predecessor 总数，旧 result/receipt 不补写观测值。

2.4.1 invocation/result 记录 `delegation_policy=forbidden`，旧字段缺失为 unknown。可选 delegation 投影只从已读 native 范围提取可识别工具/活动类型、事件锚点及关联 UUID；不保留参数/正文，不读取子会话与用户历史。summary 只含短状态与数量，完整锚点留在 SHA-256 绑定的 observation 文件。

observed/not_observed_in_range/unavailable 描述委派证据；格式未知或范围不完整为 unavailable，委派观察失败不抹去已证明的根增量。禁止委派、未见标记、关联数量及 parent_thread_id 都不能证明子会话集合完整。子会话 token 与完整调用用量仍未知；`cost=null` 表示缺少可靠计价/账单来源，不用现价反推实付。旧 observation v1 和摘要字节保持原义，新字段只由采用该能力的执行器采集未来调用。

audit 分列 wall/CLI/check-processing 与 lifecycle 等待，明示 `budget_policy=wall_clock_deadline`。pause/hold/排队/退避按已知区间并集计，缺边界保留未知；范围等待可与兄弟工作并行，不能相减推算纯模型时间。观察耗时属于 CLI 时间，观察不改变预算或追加记录。

verify 从原始输入/产物重新规范化提案并比对 receipt，核对 Reviewer 实际读取的 checkpoint、裁决结果及 response/result 一致性，不启动模型或检查器。archive 对同一范围复验后按 SHA-256 去重封存，manifest 仅含稳定范围和实际对象；同清单且对象完整才复用已有包。缺失、活动或未结范围不冒充完整，封存不提供恢复授权。

## 离线验证

工程维护者按改动范围选择必要验证：

```sh
python3 -B schema_extract/extract.py --version
python3 -B schema_extract/extract.py --help
python3 -B -m unittest discover -s schema_extract/tools/tests
python3 -B -m unittest discover -s schema_template/tools/tests -t schema_template
```

测试使用本地合成输入和可执行替身，不调用真实模型，`tools/tests` 不进入冻结清单。已有 YAML 用 `schema_template/check_yaml.py --skip-hardware` 检查；质量仍须独立语义审阅。

[Reviewer 语义回归](../review_cases/README.md)另用真实 CLI 独立调用，记录规范/提示身份、输入、结果和用量。它与离线测试分别报告，模拟回执通过不能作为模型已识别语义问题的证据。历史验收与当前操作分开，索引见[Supervisor 手册](../README.md#验证边界)。
