# Extract 工具实施规划 A：阶段、完整设计与实际来源

本报告独立形成，未阅读其他规划代理报告。仅静态走查，不实现工具，不运行真实模型、Extract Runner 或 GPU。
基线为主代理提供的 `27f36e5`；当前规范和 Template 已就绪，旧执行器协议不作为新方法约束。

## 结论

建议四轮实施，每轮两位全新 MAX：先统一最小内部接口与 Template 边界，再完成阶段推进和 Claude 调用，随后收口返回、记录与交付，最后整体验证和删冗余。
保留可信运行骨架，重写单父计划和相关门禁；不要在旧协议外再叠一层“新模式”。
数量上限只是硬门禁；防止参数膨胀的主关口是 P2 独立判断、批准有限完整集合，以及禁止 P4/P5 自行扩张该集合。

## 已证实的实现事实

| 证据 | 现状与实施影响 |
| --- | --- |
| `EXTRACT_SCHEMA.md`、`phases/02_dispatch_draft.md`、`REVIEW.md` | P2 审定机制与代表；P4 落实写作安排；P5 真正共享按实际来源，不能阶段切换便重选。 |
| `artifacts.py:load_yaml/validate_artifacts/validate_plan` | 仍读小写字段，要求七字段 `parent` 计划；P1 禁止全部硬件变化，P2 禁止 source 变化，P3 首次不普遍允许写 when。 |
| `artifacts.py:index_prefix/validate_index` | 强制旧父表和覆盖缺口节，链接必须包含未来计划文件；与正式 dispatch 及“未来文件不做假链接”相冲突。 |
| `stages.py:next_calls/affected_nodes/plan_impact`、`store.py:start_node` | 按父批准解锁、按后代失效；首次从父复制并自动写三个 `from_parent`。必须整体替换。 |
| `runner.py:_begin_reopen/_merge_check/_final_inputs` | 上游返回清后续阶段，合入/终审核对祖先；合入检查目录只复制 YAML，不能直接承接新 Template 的完整目录检查。 |
| `receipts.py:required_artifact_evidence`、`artifacts.py:design_context` | P5 强制当前节点和直接父的来源证据，provider 返回仅容许严格祖先；应改为真实区域来源及使用上下文。 |
| `artifacts.py:check_complete`、`prompts.py:build_prompt` | 外部检查未加 `--read-only`；当前 Template 默认会规范化全任务，存在越过阶段权限的实际入口。 |
| `schema_template/tools/schema.py` | `validate_documents` 是零 I/O 的严格快照检查；完整文件 API 另检查 dispatch、本地可确定来源；规范化 API 只返回候选。 |
| `schema_template/tools/design_resolution.py:ResolvedTask` | 提供 raw/effective、origins、来源链、source_manifest/effective_identity；图是跨区无环 DAG，没有共同设计父。 |
| `agent.py`、`audit.py`、`observation.py` | 实际调用和完成事件仍专属 Codex；不能把 executable 改为 claude 就接入。 |
| `prompts.py`、`runner.py:_recover_call`、`review_eval.py`、`wakeup.py` | 固定注入文字不止阶段页；搬迁须覆盖继续、修复、评审和保留的提醒入口。 |
| `cli.py:candidate_report` | 当前直接汇集所有 accepted receipt 的 framework 候选，没有 worker→reviewer 甄别门。 |

历史只定向读取 `BRANCH_DERIVATION.md` 与 `ROOT_ANALYSIS.md`：前者区分节点膨胀、重复写作和保守传播，后者没有证明 root 检查 CPU 是主要成本。
因此不规划新的缓存平台、不减独立审阅、不许以少文件或短 prompt 代替语义完整性；不承诺未经实测的成本降幅。

## 保留、删除、修改、新增

- 保留：固定源码/规则/配置摘要、角色独立、原始事件和精确证据、不可变 checkpoint、单写者、调用 nonce、预算、取消结算、迟到结果隔离、原子发布。
- 保留其职责而精简实现：本次运行的阶段返回、草稿恢复、来源变化后的复审、后台执行和必要状态观察。
- 删除：旧格式导入、旧包/入口布局探测、旧 standalone 兜底、兼容执行器升级、跨旧版本语义采用与专门迁移分支；不为它们保留测试矩阵。
- 删除：`parent/from_parent`、确定性父表、换父迁移状态、以历史继承为目的的层；不可变证据和本次运行修订记录仍是正确性资料。
- 删除 reviewer 裁决补丁路径：reviewer 始终审查，修稿一律交有权 worker；达到限制仍可有一次只读裁决，之后明确阻塞，不把 reviewer 变作者。
- 修改：入口直接改名为 `schema_extract/extract_yaml.py`，无 `extract.py` 兼容壳；真实命令、冻结启动、monitor/wakeup、测试与当前文档调用点一起改。
- 修改：`artifacts/stages/store/runner/receipts/prompts` 的主路径，移除旧字段假设；`comparators` 同步当前算法字段但保留逐输出验收职责。
- 新增：一个明确 Claude backend 和少量统一 transport 结果字段；固定提示集中 `schema_extract/prompt/`，阶段页移入 `prompt/phases/`。
- 模块按职责收口，不追求机械文件数：状态/调度、产物/Template、角色/提示、进程/预算、发布/观察；不另建通用插件框架。
- `legacy/compatibility/continuation` 及跨版本 runtime/revisions/maintenance 代码退出当前链路；其中必要当前身份核对归入 snapshots/store，返回规则归入 stages/runner。

## 先冻结的最小共享接口

以下都是 Runner 内部协议，不能加入 schema 字段；详细机制、差额与依据复用 dispatch、既有 conclusions/evidence/handoff。

| 对象 | 建议与权限 |
| --- | --- |
| `selected_designs` | P2 审定的有限列表，最小为 `{id, instance}`；instance 定位真实调用/模板实例，文件名由 ID 派生。root 必有，含 root ≤10；P4 不得增删或换实例。 |
| `writing_plan` | P4 的非 root 写作安排，最小为 `{id, depends_on}`；只含已选 ID，允许多个或零个等待项，必须可完成。拟共享区域和差异留当前目录/交接。 |
| 实际来源 | 由当前 raw 的 `from` 和 Template resolver 唯一计算；不可从 writing_plan 生成“批准的共享”。 |
| 调用上下文 | 明确来源/规则/产物摘要、角色、允许文件及字段、当前选择、当前节点安排、实际材料和本轮问题；不附全量历史账目。 |
| 批准 | 绑定该阶段批准对象、当前 raw/checkpoint、实际来源摘要与代际；P2 选择、P4 安排和 P3/P5 正文不是同一批准对象。 |
| 运行结果 | backend、requested/actual model、effort、session、completed/outcome、response、termination、usage、原始事件定位；Runner 不理解某一家 CLI 的事件名字。 |

P2 worker/reviewer 必须就同一完整候选集合形成提案与独立决定；reviewer 要求降级时回 worker 改目录，再批准，不能暗中改集合而目录仍是旧稿。
P4 只批准安排及目录落实。保持 P2 选择签名可防止 P4 用“重新规划”重新列一批参数组合；不要只比较 root 一项。
`depends_on` 表示 P4 认可的写作等待，不表示设计继承或永久内容依赖；P5 实际来源可以与拟共享不同。

## 六阶段完整预走

1. 创建运行：冻结当前源码、规范、工具与 prompt，建立 root starter 和预算；不导入旧 run，不启动 GPU。
2. P1：只写任务元数据、算法 Contract、最小 Execution 登记；其余硬件与 starter 相同。算法区独立检查，逐输出验收材料不能留给 P3。
3. P2：worker 续用 P1，reviewer 新建；允许改 root Source/Dispatch/Version 和 dispatch 草案，不写 Contract/Design 或非 root YAML。
4. P2 批准：主要机制、真实 root、有限完整代表、参数/局部/组合去处及重要疑点均有据；候选发现数不限于10，最终完整选择才有上限。
5. P3：新角色对，写 root 全部本地 Design、必要 Realization、自身实际 when/Version；source 与选择不变。检查完整 root，审当前有效全文。
6. P4：新角色对，只写 dispatch 和既有过程计划；读完整 root 与相关现有基线，落实 P2 判断，未来文件仅普通文本“仅计划”。
7. P5：按 writing_plan 的必要等待派发，每个代表新角色对；首次由 starter 建自身身份与待写区域，不复制父正文或自动宣称三整区共享。
8. P5 worker 只写自己的 YAML；已有稿件原样续写。引用来源必须存在且已独立审阅，Contract 唯一在 root，Realization/when/Version 不继承。
9. P5 reviewer 审本节点有效全文、真实链和全部使用绑定，核对差异、数值、工作、接口、生命周期及输出贯通；批准后才成为未来来源。
10. P6：全部当前选定设计已审且集合一致，新 reviewer 先审全文，再反查机制和 P5 新事实去处；需要才加入限域 worker，之后重新终审。
11. 单 root 同样经过 P4 正式目录与 P6；P5 显式跳过，不绕过最终全目录检查。
12. 当前 P6 与最终 raw/目录/来源一致才 ready；交付另核对来源、批准、文件集合与图，原子发布后才 published。

P2 防膨胀的必要审查：相对最有意义的完整基线新增什么主体机制、为什么目录差额不足、为何值得当前完整保留。
仅参数、局部和已知组合须降级；没有新增知识则合并；真正独立但不展开的候选诚实留在范围说明，不为 cap 改称参数。
程序只核数量、身份、权限和证据完整性，不用关键词或源码分支数量冒充这种语义判断。

## Template、规范化与证据的最小闭环

- 所有字段直接采用当前 Template 的唯一形式，如 `META_DATA/CONTRACT/DESIGN`；不增加大小写兼容 parser、字段别名或第二套 formatter。
- P1 用严格算法 API 加阶段权限门禁；P2/P4 用阶段目录门禁，不要求未来 Design 通过完整检查。
- P3/P5 用 `validate_documents` 校验受控规范 raw 快照和有效关系，显式做适用来源核对；P6/发布用完整目录 API 校验 dispatch 与最终集合。
- 自检 CLI 明确 `--read-only --skip-hardware`；P5 私有来源集合不能直接套用要求完整目录的 CLI，不能伪造完整 dispatch 来凑检查。
- worker 原提案先按原字节验证、保存；Runner 调用 Template 的 normalize_documents/normalize_directory 生成候选，生成一个机器转换记录。
- 整个候选涉及的文件/字段都在当前写权限内、原始 source_hashes 未变时，才在隔离副本安全落地并严格重读，作为下一 reviewer 的 checkpoint。
- 原 worker 证据继续指原稿；规范化前后摘要、corrections 与新 checkpoint 由机器关联。不得把候选行号或候选哈希冒充当时 raw 证据。
- reviewer 对落地后的 raw 取证和批准；不能在 approve 后再次规范化而保留旧批准。纯规范化不伪造 Version 增长，真实内容变化按原规则处理。
- 跨文件重命名/引用同步超出当前权限时不拆散候选强写；保留诊断并返回有权阶段或明确 P6 范围处理，不让 Agent 手工维护大小写/标点。
- 规范化规则和保真判定全部复用 Template；Extract 只决定时机、权限、事务和批准身份。

## 实际 DAG、合入与定点返回

可读材料集合、完成检查集合和有效来源身份须分别清楚，但无需三套独立台账。
P5 可以提供当前已审设计作为只读素材，便于更换真正相同来源；这些素材不能全部变成当前节点的永久批准依赖。
接收时从实际 raw 重算来源：每条采用的来源当时已供给且已审、当前仍有效；非授权文件未改。合入只取本节点 raw，再按最新实际图检查。
`ResolvedTask.source_manifest` 只含本节点、root 及各同名区域真实链；provider 其他区域可能另有引用，完整解析素材需补齐，但不能把它们伪称当前区的来源。
所有 `from` 边合起来查环，不能只逐区域各查无环；跨区循环同样不合法。
修改影响从真实 provider/区域和共享 Contract 计算；来源字节身份仍严格核对。无法可靠判断时扩大到真实来源闭包，不退回全体父树传播。
未受影响的稿件、批准和预算保留；依赖失效的在途结果退休，完整旧稿可先给 reviewer，新批准不能仅靠旧哈希/旧结论恢复。

| 新事实影响 | 路由及机器要求 |
| --- | --- |
| 普通记录增补，不改选择或义务 | 当前节点继续，既有 handoff/issues 汇集，P6 集中落实并审，不产生新 YAML。 |
| 本节点整区本地化，或换成已存在已审的相同来源 | 当前 P5 审有效全文及实际图；不回 P4，P6 对齐目录摘要。 |
| 已定代表的登记、未来等待或其他节点安排变了 | 回 P4 只修受影响安排；新依赖未解决的任务等待，其他任务保持。 |
| 机制或代表判断错误、必要关系无法支持取舍 | 有据回 P2 修该判断；集合改变再由 P4 落实，不默认重选 root/所有代表。 |
| provider 定义本身错误 | 回真实定义所在 P3/P5；资格来自 actual origins，不限祖先。正确但不适用则修 consumer。 |
| 共享 Contract 错误 | 回 P1；复核真正受影响的选择/正文，不固定重跑所有 worker。真实来源或任务改变另建运行。 |
| P6 有局部修稿 | reviewer 明列文件与字段，writer 不取得全目录任意权限；根定义、Contract 和选择纠错仍走所属阶段。 |

去掉“必须新读到此前没引用的源码行才准返回”的机械限制；已有固定源码的具体误读或必要关系缺口也可能推翻判断。
保留稳定问题与证据去重、原预算和阶段代际，防止同一无新进展请求形成无限重开。

## 记录、图与交付

P5 新事实使用原有 conclusions/issues/handoff；Runner 只汇集当前有效节点的相关回执位置，P6 在独立理解产物后读取，不把历史批准注入首次理解。
最终核对每项重要知识落在完整 YAML、规范变体记录或未展开独立候选说明；没有额外永久知识台账和固定终审报告。
Framework 闭环：worker 在现有 framework_candidates 给稳定 ID、具体条款/反例/影响；reviewer 独立核实，只把确认项沿同字段提交；Supervisor 按 ID/根因接收处理。
候选汇总必须区分未经 reviewer 确认的 worker 提案；空候选不增加报告，候选正确也不自动修改规范或阻塞本来合格的产物。
最终 schema 集合精确等于批准选择，dispatch 遵 common 最小格式；临时/备份/未计划 YAML 不进入交付，图和回执单独列在 manifest。
图使用 Template 的 `pipeline.configuration/analyze` 与现有 GraphOutput，显式样本从当前贯通实例提供，保留样本、raw 来源链和 effective identity。
图写入 Runner 的交付暂存区，不能在角色工作区运行默认 `--graph png` 顺带改 YAML；出图前后再次核对来源摘要。
有限图缺参数或展开未完成不能冒充通过；图只证明实际样本，不增设 GPU 验收，也不把动态图参数不足误称源码缺项。
standalone 继续在创作目录外生成有效全文与来源凭据，删除旧 exporter fallback；它不增加完整设计名额或替代 raw。

## Backend 与提示材料

实际 phase 默认且明确登记为 Claude Code `qwen3.8-max[1m]`、worker/reviewer 均 effort=max；命令参数与 CLAUDE_CODE_EFFORT_LEVEL 同值。
保留显式 Codex backend，冻结 backend/model/effort；不继承歧义默认，不静默换模型，不把 Codex 会话 ID 当 Claude 会话。
Claude 需校验成功 result、structured_output、session、进程退出与终止、nonce/schema；保留原 JSONL，usage 缺项明确未知。
会话遵阶段规则；精确 `--resume`，不使用最近会话猜测。工具 allowlist 之外仍需文件/目录隔离、权限和差异门禁，reviewer 不开放任意可写 shell。
固定 common/roles/phases/重试等提示全部进入 prompt；Python 只装配动态事实、路径和模板，不夹带大段自然指令。
冻结清单包含实际注入的全部 prompt；provided_documents 保留，避免同一已嵌入文档再次要求读取。搬迁同步当前链接，不批量改历史归档或 schema_wiki。

## 四轮分工与验收

| 轮次 | 新 MAX A | 新 MAX B | 本轮共同出口 |
| --- | --- | --- | --- |
| 1 | Template 适配、阶段字段权限、规范化事务、只读检查 | 最小 selection/writing/approval 接口、store 草稿与集合基础、删除旧兼容入口 | 锁定共享协议；P1–P4 静态/合成路径可用，不依赖真实模型。 |
| 2 | P2/P4 批准约束、P5 DAG 调度/合入与实际证据 | Claude/Codex transport、结果审计、全部 prompt 外置与冻结 | 两后端替身走新六阶段；参数记录不会自行晋升节点。 |
| 3 | 定点返回、来源失效、P5 新事实与 framework→reviewer→Supervisor | P6 限域修稿、规范目录终审、图/视图/原子交付 | 多来源、并行迟到、返回修复后仍能正确终审交付。 |
| 4 | 集成走查与故障路径收口，修缺失职责 | 入口直接改名、当前手册同步、删除失用模块与旧测试 | 唯一新入口、无残留旧树语义、适用检查通过；不运行真实提取。 |

同轮共享文件指定唯一 owner；另一人只提交明确接口需求，避免两个代理同时改 runner/store/receipts。第1轮接口确定后后续不各自重建模型。
每轮只保留有意义的行为验证，不为搬文件或逐行实现写镜像测试；末轮整体运行一次适用 suite。

必要验收集中六组：
1. 单 root 和多个完整代表的完整替身闭环；P2 超10拒绝、P4 不能恢复已降级参数项、未来文件无假链接，P6 最终集合精确。
2. P1 Execution 可写但其他硬件不可写；P2 Source 可写、Contract 不可写；P3 when 可写；P5 不可改 dispatch/provider；P6 scope 外拒绝。
3. 两来源加跨区链、整区本地化、跨区循环、来源未批准/漂移、无关节点变化；合入与失效只影响实际范围，草稿及预算保留。
4. 规范化跨文件依赖超 scope 不落地、许可修正自动落地、原证据不冒充新位置、reviewer 读到最终 raw、批准后无隐式写入。
5. Claude/Codex 成功、精确续会话、错误/截断/缺 structured_output、取消与未确认终止；模型/effort 不被环境或 fallback 改写。
6. P5 新事实到 P6 目录、provider 定点修复、Graph provenance、最终全目录检查、发布冲突/来源漂移和 standalone 来源复核。

四轮后两位全新 XHIGH 独立验收：一位侧重方法与语义权限，一位侧重真实工具事务与后端；不得以同一替身结果代替代码独立走查。
再一轮全新 MAX 修明确意见，最后原 XHIGH 逐项复验；此时才判断端到端 ready，真实算子运行另按已授权范围开展。

## 确定、待核实与用户决定

- 已确定：当前 Template 的唯一格式、六阶段职责、≤10、无旧兼容要求、Claude 指定模型/max、独立 reviewer、无 GPU、无 wiki 数据改动。
- 本报告建议：内部 selection 与 writing_plan 的最小拆分、worker 后 reviewer 前规范化、四轮 owner 分工；整合代理可在这些职责不丢失前提下调整命名。
- 实施时核实：本机 Claude 精确参数/事件样例与隔离能力；图样本交给 Runner 的最小编码；现有 tests 中哪些仍保护当前不变量。只读帮助/替身可先验证。
- 图是否每个最终设计默认必须出 PNG，方法正文尚未把它列为硬放行项；建议接通并验证能力，最终运行缺样本时明确未完成，不自造取值或静默跳过。
- 当前无需向用户追问才能实施；真正独立且被明确要求全部交付的方案超过10时，才有需要用户决定的目标冲突。
