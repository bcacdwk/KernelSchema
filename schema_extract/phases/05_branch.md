# P5：派生节点

本页每次只完成一个计划内的派生节点。

以下准入、推进与重开用于普通提取；`maintenance` 只执行本次任务，写入还受调用的文件/字段授权限制，不能据本页扩大范围。

## 准入与应读材料

P1–P4 放行有效，本节点属于当前计划；其计划指定的直接父节点已经独立审过并合入。Runner 提供节点 ID/文件名、直接父节点、计划修订、P4 已审清单中本节点条目及短交接、已审 root 与所需祖先、固定路径源码、起始 checkpoint、依赖摘要及可写范围。首次节点新开一对独立 worker/reviewer，节点内返工续用原角色；已有完整候选可先交 reviewer，不要求重新创作。

本节点条目的 `evidence` 编号属于 P4 回执；worker 调用中的 `node_evidence` 已解析出对应固定源码锚点，`plan_receipt` 可只读回查。reviewer 从当前产物及固定来源入口自行定位，本轮回执的证据独立编号。

worker 定位包含本节点、`ancestor_entries` 中的实际祖先、P4 短交接和当前反馈；reviewer 不接收历史交接、批准或旧稿内容，审阅当前文件、共享 Contract 和显式 Design 来源链。`design_resolution` 提供当前来源图、manifest 和有效身份，不代替 raw 原文阅读或完整检查。不为普通派生调用加载无关兄弟的计划或审批正文。裁决另提供本节点候选 checkpoint 的只读位置。

worker 先读 P4 交接与本节点清单，再读当前共享 Contract、直接父与实际 provider；reviewer 先按[当前文档验收](../REVIEW.md)理解完整有效 Design，自行理解后再核对本路径源码及共享定义。按[公共派生规则](../../schema_docs/0.common.md#派生节点与-dispatch)核对身份；撰写复用 [P3 的节点逐段方法](03_root.md#节点逐段方法)及规范 3–6，**不继承 P3 的 root 准入条件和写入权限**。

## worker：可写范围与操作

Runner 在私有目录放入已审 root 和所需祖先。首次确无自己任何稿件的新节点从直接父元数据模板建立 `<id>.yaml`：移除 Contract 和父 realization，设置自己的 `dispatch.id`、显式直接 `parent`、空 when 与 `version: 1`，三个 Design 区域均为 `{from_parent: <直接父ID>}`。已有草稿、重试、曾批准或可恢复稿完整保留自己的 raw，只刷新只读祖先；不自动转换引用或落地本地。worker 核对准备结果并补齐自己的选择条件，不能自行改选父节点。私有目录不放兄弟的在写草稿。

worker 只可改本节点 YAML 并返回验证回执。root、祖先、其他节点、`dispatch.md`、源码、规范、检查器和运行状态只读。新节点保留与 root 相同的 `meta_data.name` 和 `describe`，核对 `source` 指向本路径实际设计依据；节点 ID 与文件名、parent 与计划一致。初版从 1 开始，后续本节点内容修订按[版本规则](../../schema_template/README.md#3-填写与修订)递增，不能每次返工重置为 1。

调用中的 `first_draft` 区分首次撰写与已有节点的修订。已有 ID 因计划重开或删除后再加入时，沿 Runner 提供的版本继续，原调用、退修、时间与裁决账本仍有效。

调用带 `maintenance` 时，按当前 action/step 和 fields 接续；完整旧稿的独立复核、有限修补及批准采纳依[共用方法](../REVIEW.md#发现问题与批准)。父变化只更新只读前提，不能覆盖自己的正文。

调用带 `identity_migration` 时，旧 raw 的 parent/ref 与新 plan 可暂时不符，完整检查前必须由 worker 显式迁移。旧 checkpoint 只在 workspace 外作为只读恢复线索：逐区选择引用新直接父，或取旧有效内容落地整个 local 区域，再修改本地 parent 并按原版本演进；解析器不自动 retarget。迁移完成后按新计划、父依赖和有效身份独立获批，不能恢复旧批准。maintenance 缺少 parent/ref/落地区域写权限时报告缺少的 scope，不自行扩大范围。

1. 写本节点自己的非空英文 `when`，在 root 的整个契约有效域内独立给出充要选择条件。必要派生量和运行时事实在本节点说明，不叠加父 `when`，不复制或收窄 Contract。
2. 对照 P4 指定的有效差异和实际源码，依次执行 [P3 第 4–7 段](03_root.md#节点逐段方法)。各区选择完整本地定义或显式同名整区引用，可沿直接父传递。需要局部变化时，用 `check_yaml.py --show-effective` 获取有效内容，将整个区域落为本地后修改；不能附加 patch、隐式继承或跨到兄弟/外部。原文相同仍须核实本节点的数学、参数绑定、调度、流水线与选择域。
3. 每个节点形成完整有效 Design，并独立填写适用的本地 Realization。父的 metadata、when、版本、realization 和反馈不随引用继承；已有本地实现交接也重新核实适用性。只有当前 Contract 与有效 Design 之外没有必要实现交接时才按规范省略；与父相同但本节点仍必要的约束也须在本节点显式填写，不能以“相对父没有新增”为由省略。源实现缺陷沿用[设计分流](../REVIEW.md#源实现缺陷与设计分流)，不以问题记录代替完整设计。
4. 选择本节点范围内的调用，使用完整有效 Design 按 P3 方法贯通输入到全部输出，覆盖 launch/producer/事件、handoff 与资源闭合；记录相对直接父的有效差异、共用内容适用性、关键 raw 源码锚点和未决项。引用写法本身不构成 difference。硬件只存在于只读 root，保持 starter 占位；GPU 验收与实测不在本阶段范围。

计划或直接父选择错误时，返回源码新发现或具体纠错依据和申请重开 P4 的理由；不自行换父、改索引或写兄弟。本节点 `when` 的错误在本节点修正；root Design/when 回 P3，共享 Contract 回 P1，交 Runner 按[修订规则](../tools/README.md#并行合入与计划修订)路由。实际 provider 定义错误与仅在子上下文不适用的情况按下段区分。算子自身缺项与框架候选遵循[分流规则](../../EXTRACT_SCHEMA.md#框架问题分流)。

普通 normal P5 发现实际非 root provider 的本地定义错误时，可用 `{issue_id, target: 'P5:<id>', reason, evidence}` 请求本轮 `provider_reopen_targets` 中的目标。证据须定位该 provider 区域的当前 raw 定义，并提供固定源码新发现或具体转录/规范纠错依据。关联 issue 的 `scope` 同时写 provider 真实文件名与 `design.<区域>`，其 `evidence` 与 `reopen.evidence` 至少相交一项本次源码索引；reopen 至少有一条有效 issue 的 `recovery_condition` 非空，见[协议片段](../tools/README.md#普通-p5-的-provider-修复)。仅因共享定义不适用于本子上下文时，落地本节点整个 local 区域后修订。Runner 在提交 workspace 重新计算来源与资格；已本地落地的区域不能借旧链重开 provider。维护、裁决和签收没有此入口。

有效请求先保存请求者的合法 raw 草稿，再由 provider 新 generation 的 worker 修复、独立 reviewer 新批准；即使字节相同也不恢复旧批准。provider 后代保守失效，完整稿待 reviewer、未完成或待迁移稿交 worker，按父先子后恢复；兄弟与 P1–P4 保留进度。P6 当前资格失效，原预算、退修、P6 starts 和已用单次决定不重置；同一 provider、稳定 issue 与固定源码证据不能靠改文案重复重开。

## reviewer：放行门槛

按[当前文档验收](../REVIEW.md)覆盖本节点全文及当前共享 Contract，核对改变结构的各类条件。只读上游问题按本轮模式处理；维护不使用普通重开决策。

检查扫描同目录直属全部 YAML，因此私有目录必须具备 root 和所需祖先，并隔离兄弟草稿。Runner 在接收完成/批准时自动检查；来自只读祖先的问题交回处理，不能越权修复。

独立对照本路径源码，按 [P3 的逐段方法](03_root.md#节点逐段方法)审定 `differences`、独立条件及完整有效设计。provider 本地定义的通用推导可按当前文件读一次；本节点 `inherited_design` 必须覆盖选择域、名称/参数绑定、跨区接口、数值路径、工作归属、依赖、资源复用和本地 realization。父批准不能证明适用性，`walkthrough` 仍贯通本节点输入到全部输出。确认无 Contract，ID/parent/文件名及引用正确，其他文件未改；父来源或计划变化后重新验证。

## 回执与下一位

按[回执约定](../tools/README.md#回执与短交接)提交本节点相对直接父有效 Design 的差异、`inherited_design` 适用性、完整 `walkthrough`、实际检查与未决项。reviewer 的 `artifact_evidence` 覆盖 child 与直接父的 source manifest 并集，再并入 root/child/直接父文件。批准时还须对 child 和直接父分别逐区域覆盖每条引用 chain 的全部节点：该 consumer 自己的 `from_parent` raw 行、所有中间引用 raw 行及实际 provider 本地定义 raw 行。每文件一个任意锚点不足以满足此门禁；同一真实范围可满足多项要求，无需重复列相同锚点，不以有效视图虚拟行号代替。本阶段不另生成 `handoff.md`。

reviewer 通过后，Runner 核对尝试、许可差异与全部祖先依赖，只替换候选自己的 raw 文件。完整 merged 树保留全部计划稿件；完整检查在 workspace 外受控临时目录运行，按[合入检查协议](../tools/README.md#并行合入与计划修订)取有效节点、候选与祖先闭包。候选 raw SHA 异于已批准值时，先排除后代再取闭包，记录实际集合与摘要；维护父候选也遵循此时序，不等待批准后的状态失效。未包含的稿件不因此获批。保存不可变 checkpoint 后才解锁子节点；独立兄弟继续工作。P6、ready 和发布仍要求整棵计划树完整有效。
