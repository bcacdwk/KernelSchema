# 第一轮 Extract 工具、回执协议与运行控制审计

日期：2026-11-21。审计角色：第一轮 max / Extract。基线：`484902855fe7be82a74fe7c394503d1d7d222dbb`，保留当前工作区既有未提交内容。

这是取证和修复方案，**没有实施产品修改**。本轮未启动 extract、resume、repair-runtime、monitor 或真实模型/GPU，未修改正式 schema、历史回执、运行状态、输入锁或批准。十二项、33 份设计已交付的事实引用 [FINAL_REPORT](../../extract_twelve_20261119/FINAL_REPORT.md)，不把本次工具审计当作对 33 份设计的再次语义签收。monitor 仍应保持已停止状态。

已读根 AGENTS、`schema_extract/README.md`、`schema_extract/tools/README.md`、本轮 RECAP、相关恢复/巡检记录、必要小型回执和当前实现；复审相关已归档修复，不以 `resolved` 代替验证。既有 workflow_architecture 讨论只作背景，没有将其中计划转为本轮实施授权。

## 结论与收敛方向

最主要的成本链是：**模型不知道全部机械义务 → 一份回执只得到第一个错误 → worker/reviewer 共用终身累计的两次自动补正额度 → 尚有大量 32 次预算时反复停到 Supervisor。** 单独加一句提示、单独扩大补正次数或持续人工 resume 都不完整。建议在现有 Runner 内闭合三件事：统一暴露回执义务；一次报告可安全判定的独立机械错误及实际计数；普通回执补正使用现有 scope 调用预算，保留累计账目和所有硬门禁。

另确认两处应独立修复的代码缺陷：

- **X-08：**多步骤控制命令已提交授权后，后续锁竞争仍可能返回 `committed=false`。本轮以纯内存注入重现，不声称本批已发生重复模型调用。
- **X-15：**P2 合法 `revise / plan=[]` 被误当成待检选集，产生假 `checker_input_error` 并发给 worker。两项真实 run 的回执和 check 均证实；没有因此错误批准。

KDA 的读数则需要纠正问题表述：当前 `summary.json` 和 `state.json` 都是 revision 155、同一已发布结果。缓存 stamp 的 `ctime_ns` 不同导致 fallback；`read_summary` 返回的是当前 state 投影，却继续标 `stale`。这不是已证实的错误交付状态，而是缓存与返回值时效混在一个标记中。

不建议增加通用策略平台、旧版本兼容系统、新控制服务或一套手写语义校验器。既有固定身份、单写者、规范化事务、scoped recovery、工具修复续接和最终批准门禁应保留。

## RECAP 流程项覆盖

| RECAP 项 | 审计编号 | 判断 |
| --- | --- | --- |
| 输出目录已有空占位 | X-12 | 既定无覆盖发布约束；现场处理正确，非执行器缺陷 |
| CLI 模型名提示 | X-12 | 非阻断 CLI 提示；不能据此证明上下文或 effort |
| 回执入口类别遗漏 | X-02 | 回执缺项；模型材料和诊断可显著改善 |
| 证据坐标错误 | X-03、X-04 | 模型稿件错误；已有严格引用校验正确，反馈不足 |
| dispatch 链接、首稿 Version、文字预算 | X-06 | 规则有据的稿件错误；不可借修工具自动改语义 |
| handoff 超长 | X-01、X-04、X-05 | 提示/schema 与运行门禁不一致、计数反馈不足 |
| 累计补正额度触顶 | X-05、X-07 | 当前有意策略的可用性缺陷；未耗尽 32 次 |
| 控制入口竞争 | X-08 | 本次未提交重放正确；另有已提交后误报的代码缺陷 |
| stale / superseded / 重启旧 due | X-09、X-10、X-11 | 三个不同问题，不能合并成“监控不可靠” |
| 取证脚本自身错误 | X-13 | 多数为调用方错误；补少量接口说明/字段即可 |
| kernel 来源风险、来源证明缺口 | X-14 | 排除 kernel 修复，保留证据与范围 |
| 从 framework_candidates 补充的 P2 退修问题 | X-15 | RECAP 未单列的真实诊断归属缺陷 |

数学/精度、布局/容量、同步/寿命、调度/适用域、ABI/目录五类 schema 修正由同轮 [SCHEMA_AUDIT](SCHEMA_AUDIT.md) 主审；本报告只核对其在 Extract 中的责任与门禁，不重复宣称源码审计。

## X-01：handoff 的条件门禁没有完整进入模型契约

**事实。** `receipts.py:461–464` 只对 **P2/P3/P4 的 `approve`** 要求非空、`len(handoff) <= 1200`、`splitlines()` 后非空行数不超过 20。`receipt_schema:94` 和 `proposal_schema:134–153` 只有普通 string；`response.md:17` 写“简短、必须提供”，没有数值。`prompts.py:268–280` 暴露结论、类别及调用数，但没有 handoff 数值。实际 indexed Sparse 两份完整提示也没有 `1200`，见 [观察](../../extract_twelve_20261119/recovery/indexed_sparse_p3_anchor/handoff_prompt_observation.json)。MLA 3011→1224、BF16 FMHA 2270→1235 的现场说明已有反馈后仍有模型计数失准，不能一概称反馈丢失。

**影响与当前处理。** 先消耗完整审阅调用，才发现隐含格式约束；手工有限恢复保住严格门禁，但没有消除下一批的同类成本。P5 split_k_reduce 的 2841 字/8 行 handoff 合规于该阶段，见 [GEMM 恢复](../../extract_twelve_20261119/recovery/gemm_p5_split_k_anchor/README.md)；禁止将 1200/20 泛化到 P5、P6 或 P2/P3/P4 的 revise/blocked。

**最终行为与修改位置。** 在 `receipts.py` 集中一个小型“本调用机械义务”函数，让 `proposal_schema` 的字段说明、`prompts.build_prompt` 动态数据和本地检查使用同一数值/条件；`response.md` 明确按该条件计数。诊断同时返回实际字符数、实际非空行数及限值，空白、换行也按现行 Python 语义统计。可建议作者留摘要余量，但不能把建议 800 字变成新硬门禁，也不能由 Runner 截短文本。

JSON schema 若要机械约束，应使用 **decision 条件分支**，并与实际支持的 backend schema 能力一起验证；不得给整个 reviewer P2/P3/P4 的 handoff 无条件加 `maxLength`。非空行计数不值得另写一套复杂正则：schema 字段说明明确此条件，Runner 用同一函数执行完整规则即可。`validate_shape` 当前也不实现 `maxLength`/条件组合，不能只往生成 schema 加关键字便宣称两端一致；结合 X-04 统一结构校验，或明确 provider schema 只承担其支持的结构子集。

**验证与边界。** 本轮用既有 ProposalTests fixture 实测：1200 字、20 非空行通过；1201 字、21 非空行被本地 gate 拒绝；四者均通过当前 JSON schema。下一轮增加 stage × decision 矩阵，尤其 P5 长 handoff、P2 revise 长 handoff、Unicode 与空白行，并检验生成提示含相同值。纯澄清/诊断是小改；不更改语义规则、历史回执或门禁限值。可删除散落的数字副本和只说“短”的模糊错误文案。

## X-02：同一绑定文件承担 implementation 与 host，义务表达不够直接

**事实。** `STAGES` 为 P3/P5 要求 implementation、host；`receipts.py:401–406` 将 implementation 映射到 `bindings.entrypoint`，逐类别检查绑定 **path**。本批 indexed Sparse、masked GEMM、GEMM split_k_reduce 的失败证据均只满足其中一种类别，另一类只引下游 CUDA；拒绝正确。当前提示只给 `required_evidence` 与通用 `inputs`，需模型自行推导 implementation→entrypoint，并意识到同址仍需分别登记。

相关真实候选：`required-evidence-category-binding-same-path`（FP8 MoE `abf696ae-ac13-4ce3-9005-368b95b272a6`）、`required-evidence-bound-entry-category-hint`（同 run `1759c2ba-0958-4ef2-a101-b52968c23161`）、`receipt-impl-entry-anchor`（indexed Sparse `4a60aac4-d688-43e1-a85f-635c0d5363db`）。这些候选归一到本项，不建三套修法。

**责任与影响。** 绑定入口覆盖和具体节点 `META_DATA.Source` 是两种关系：P5 的实现入口可以在 kernel 文件，但仍须解释任务级绑定入口如何进入它。保持每角色分别核实，不能自动复制 host 证据生成 implementation 判断。机器当前证明的是类别/文件及真实坐标，**没有证明引文落在绑定 symbol 的 AST 范围，更没有证明引文支持语义**。候选中要求“行号落在 symbol 范围”的建议强于当前 gate，不能冒称现有契约已经机械保证。

**最终行为与步骤。** 复用 X-01 的小型义务数据，向每次调用给出 `{category, binding_key, locators:[{path,symbol}]}`；报错列出缺失类别和期望绑定位置。`response.md` 补一条同址例：同一原文范围可由模型在两类各登记一次；其余调用链锚点仍按事实选取。保留单 category 的 proposal.v1，不为此升级成多类别证据系统。明确建议补项追加到数组尾部，再核对索引，减少无意义的全表重排。

**验证/迁移/复杂度。** 覆盖同址、不同址、多 locator、绑定入口与 P5 raw Source 不同、只有下游类别而缺绑定路径等正反例；合法同址两类通过、单类仍拒绝。本輪相关证据义务测试通过。无需新 C++ 解析器或自动语义锚点选择。旧回执不改；新增动态事实是现有固定绑定的可读投影，不改变来源。工具仅检查文件关系的边界应写清，不增加无根据的“已证明符号覆盖”结论。

## X-03：坐标与本次索引是有效硬边界，减少人工算术而不改证据

**事实。** `reading.py::read_evidence` 核固定路径、真实 hash、1-based 行区间、0-based 字符偏移，再从指定范围截取既定最多 320 字符的真实 quote；`normalize_response` 由实际 bytes 补 hash/source_id/quote。GEMM raw 只有 275 行却引至 276、FP8 MoE 574/575、MLA quote_offset 无效以及多项 source_evidence 索引越界均有现场记录。不是由正常规范化把作者证据偷偷换到新行号所致。

**当前修正是否足够。** 2.5.1 的 `receipt-evidence-repair` 正确地将固定文件内的位置错交原角色补正，并保持外部路径、hash 漂移、权限越界为硬拒绝。它没有消除“索引错误只报 outside”“偏移错误只报 within range”的定位成本。现有读数 helper 本身具备精确 quote，但 Claude 的可执行 helper 是无额外参数的 `check-helper.py`；不能向用户声称角色已经有任意可执行的引用计数工具。

**最终行为与位置。** 在 `receipts.py` 的索引检查记录具体 JSON path、实际索引、允许范围；在 `reading.py` 的位置错误给实际文件行数/选择范围长度/偏移单位。保持原数组位置，不删除坏锚点来让后续索引“凑齐”。与 X-04 一起返回同批错误。首轮方案无需为计数引入 proposal 文件写权限、通用 Bash、MCP 取证服务或新 evidence-ID 协议；精准动态材料和返回的计数先闭环。

**必须保留的边界。** 不钳制末行、不把失效 offset 自动改成 0、不搜索相似文本后偷偷换锚点，不按下游 helper 自动补入口类别。模型选择支撑断言的原文；工具证明所选字节真实；独立 reviewer 判断支持关系。既有 320 字符摘录协议不用于“修复”错误引用，也不替代完整行区间和原文。

**验证/迁移/成本。** 本轮位置/offset/长行迟后声明/非法路径/假 quote 的集中现有测试均通过。新增测试检查每个错误 JSON path、多个坏锚点不改变其他编号、跨规范化 raw/preview 坐标仍分开。纯工具诊断可沿工具修复政策接续；保存的原稿和 receipt 不重写。改动小，可删除恢复脚本里手算每个末行和偏移的重复代码；不必恢复已交付 run。

## X-04：首错即停和丢失已有 schema 错误明细形成多轮修稿链

**事实。** `normalize_response:196–219` 在第一个非法锚点就退出；`validate_receipt:390–464` 先索引，再类别/入口/结论，最后 handoff。masked GEMM 依次暴露末行、入口类别、handoff；indexed Sparse 的缺 host 与 2282 字 handoff 同时存在却只送前者。`agent._schema_check:137–152` 已收集前 12 个结构错误的 path/validator，但 `runner._receive:805–807` 把它们压成一句“did not satisfy the scoped proposal schema”。

**根因/影响。** 独立、无副作用的机械条件被一条异常链串联；调用者不知道后置错误存在。补正计数却按整 scope 累计，交错的作者/审阅者为同一份完整回执重复支付调用。当前人工诊断虽然能看到后置问题，但普通 resume 只传登记的旧反馈；不应靠修改 state 或临时补丁 prompt 解决。

**最终行为与具体步骤。**

1. 先验证 nonce、调用身份、固定来源、文件类型/权限与读取可信性。来源漂移、越权、未知终止不能混进可补正集合。
2. 在 `receipts.py` 抽出一个无状态的机械诊断入口，共用已有 JSON schema 校验；对于类型/结构已能安全读取的字段，累计坐标、索引、类别/绑定、必填结论、checks_note、条件 handoff 等独立问题，给稳定 code、字段位置、observed/expected。上限用于限制反馈体积，同时记录总数/未检查原因，不声称已检查未知部分。
3. 任何必需项失败均不生成接受/批准。不能因为某锚点无效就删除它后继续归一化；诊断可引用原始位置，最终仍由新 proposal 重新全验。
4. `ReceiptFormatError` 携带诊断列表；`_recover_call` 在原 reason.summary 之外保存/传递细目；原 `result.response_validation.schema_errors` 也进入反馈。保留可信错误和可补正错误的分类，不把所有 ExtractError 一概自动重试。

**验证。** 至少构造同一回执同时缺绑定类别、索引越界、末行越界和 handoff 超长：首轮应列全部可判定项，raw 不变、无批准；修正后仍完整校验。身份错与普通缺字段同时出现时仍硬拒绝。遇损坏容器/非 UTF-8/越权文件只报告可安全确认的事实。覆盖 P5 from 链引用漏项：其机械遗漏可统一反馈，但解析/实际来源错误仍是 gate，不降级为格式成功。

**复杂度/迁移。** 中等、应局部重构，不能新增一套与最终 gate 不同的“宽松预检”。当前 `validate_shape` 与 `agent._schema_check` 重复维护结构语义，优先复用 jsonschema 的同一实现，保留显式身份/安全前置；删掉逐条拼接的泛化异常文案。只增加未来 reason 细目，不改历史 schema/receipt。工具修复续接可采用这类 Python 修正，冻结 prompt 的语义保持。

## X-05：累计两次自动补正与 32 次 scope 预算之间的职责混淆

**事实。** `effective_config` 固定 `retry_limit=2`；`_recover_call:742–765` 每次先增加 scope.receipt_repairs，超过 2 即停，worker 和 reviewer 共用 `scope_for`，成功不会重置。indexed Sparse P3 首次 reviewer 缺 host 时已被两个 worker 修补耗尽；mixed Sparse P4 首次 handoff 错误在前两次索引补正后立即停止。实际只用 4、5、6、7/32 次的多份恢复记录吻合。`test_receipt_repair_policy` 还明确断言共享累计策略，故这不是计数器偶发错误。

**当前处理。** 普通 resume/scoped resume 没有退款，仍保留补正累计；现场基于新诊断或明确收敛逐次恢复是正确止血。但已超过 2 后每个新机械问题都会再次停，且 `call_limit_or_single_decision` 容易被误读为 32 次耗尽。原 issue `receipt-evidence-repair` 所称“已解决”只覆盖错误分类，未覆盖此成本链。

**推荐默认。** 普通、已确认完成且安全保稿的 **receipt repair** 使用现有 `call_exhaustion` 限制，不再用终身累计 `receipt_repairs > 2` 作为第二个停止条件。累计 receipt_repairs/invalid_receipts 永远保存、继续增长。32 包含续写、补正、退修、失败，且现有 worker 后续 review/一次裁决/一次 signoff 预留照旧；不是给补正另开 32 次。CLI 故障、维护任务、adjudicate/signoff 的原一次性规则不因本项扩大。

这样无需再引入按角色、错误 ID、连续次数、进展评分四套计数器；本来已经授权的 scope 调用上限就是成本边界。真实 blocked/reopen、身份/权限/来源错误、未知终止、暂停/取消/hold、工具/检查器故障及原调用预算耗尽仍交既有责任方。Supervisor 处理这些例外和显式额度追加，不代替 Runner 逐项批准普通计数修稿。

**修改与验证。** 修改 `runner._recover_call` 的 receipt 分支、相应说明及 `monitor` 历史停止码的人读解释；不要改 `scope.calls`、`call_limit`、调用预留或 resume 的权限。现有测试中“共享累计记账”应继续通过，只替换“累计次数是第二硬阈值”的期望。新增跨角色成功后再错、已恢复且累计>2、剩余调用不足、pause/unknown termination 抢先、被修坏的原稿权限、有效 revise/blocked 不进入补正等案例。

**取舍与兼容。** 这是自动推进默认行为的产品取舍，非数据修复。推荐采用统一调用预算；代价是若模型持续机械失败，可能比旧两次上限多消费已授权的调用，收益是无需人工搬运每个新诊断。不能用“有 32 次”自动恢复已停止或已发布历史 run；仅新代码下自然触发，旧停点仍经已授权正式 resume/repair-runtime。无需清零历史累计或构建版本兼容层。若用户明确另要求更低的自动成本上限，再作单一产品决定；不应把每次普通恢复重新变成审批。

## X-06：产物机械规则与规范化责任要分开

**事实与归属。** GDN 新节点首次 raw Version=2，基线/starter=1、authored=false；P5 页明确首稿1，`artifacts._versions:523–533` 拒绝有据。dispatch 缺当前完整基线直链、正文超字符/行数，同样不能因为模型想完成就取消限制。已发生的数学、同步、适用域修订是作者/独立 reviewer 的语义工作，不是 Runner 可自动修补的格式。

**保留的正确机制。** `normalize_candidate` 原稿→预览→授权范围→受控落地→严格重读，`receipt` 锚原稿而 reviewer 审最终 raw；只读 provider 不被暗改。`pending_corrections` 不等于失败或已批准。此前 normalizer 首差异提示修复有价值，但不能用它担保大小写、引用、定义的转换永远保真；该实质边界由同轮 SCHEMA_AUDIT 深审，本报告不作重复结论。

**改善范围。** `layout-correction-without-field-path`（FP8 MoE `abf696ae-...`，对应 SCHEMA_AUDIT 的 S-08）反映上游只读 Contract 的 layout warning 不可行动。现行 `check.md` 已允许 worker 在合规预览通过后提交，由 Runner 落地；不应强迫 P3 改 P1。可让 Template 诊断给首差异字段/顶层区域，Extract 只将其和当前写权限对照，说明可写/由 Runner 机械落地/需要上游责任。该增强是体验项，不能扩写权限或增加“修全部 warning 才 complete”的新门禁。

P2/P4 的未来基线线索与正式 variant 直链的关系，按 SCHEMA_AUDIT 的 S-09 处理：草案把线索留在范围/计划中，真实基线完成后落实正式记录；不生成空 YAML、不把无关 root 伪称基线。已归档 `p4-index-layout-hard-stop` 取消了无意义段落位置要求，当前 `validate_index` 已不再要求 canonical 字节前缀；保留这个简化，不恢复专用布局异常层。

**验证/迁移/成本。** 本轮已跑首稿规范化保存原证据、只读 provider 不变、P4 说明段布局等集中测试，均通过。新增诊断只验证路径/范围与结果不变；语义规范化修法由另一审计报告规定反例与保真要求。历史产物不重新规范化；不能为了清理文案改写已经发布的字节。没有依据把 GDN 版本错误归档为工具 bug，或用自动 Version 钳制代替作者修稿。

若后续经授权人工修订已发布的正式字节，原 publication proof、receipt 和 run 保持历史原意；新字节另记当前 manifest、修改范围与真实审阅来源，不伪造新的 P6 批准。当前清单的最小建议见 SCHEMA_AUDIT 的 S-09，不为此追加通用发布框架。

## X-07：scoped recovery 与 tools-only 修复续接应保留，减少重复人工证明

**事实。** GEMM `P5:split_k_reduce` 恢复后只增加目标 reviewer 一次，健康 pair_two_pass 身份未变，343 项历史文件核对保持；GDN `P5:warp_row_reduction` 仅增目标 worker，健康兄弟 reviewer 保持。入口核实际失败 attempt、generation、实例/来源、当前批准与真实终止；消费时再次核控制权限，原 reason 先进入 feedback，再清空，state 先提交、`last_recovery.request_id` 去重。见 `recovery.py:172–240`、`runner.py:1144–1209` 和两份恢复 README。

**旧修正复审。** `failed-node-recovery` 的反馈丢失修复当前仍在；本轮四个 scoped 回归通过。`executor_repairs._target` 只允许 Extract 非测试 Python 变化，保留 source/任务/prompt/Template/docs；`_saved_frontier/probe/apply` 区分合法保存稿和完成度，保留原权限、批准来源、最新 P6 草稿及累计次数，提交前核前沿和控制。新增审计标记不生成批准。两项集中 executor_repair 测试本轮通过：全 P5 草稿/健康批准保留；blocked 决定及耗尽预算不被解除。

**最终行为。** 保持普通恢复、局部恢复、tools-only 修复续接的现有分工。健康 P5 兄弟在途时，选唯一失败 scope；不为一个 handoff 或 Version 错调用维护、更换执行器或重开 run。真工具故障已获持续授权时，用原 rules 的完整树加审过的 Python 补丁，prepare/apply 后普通 resume，随交付留真实阶段/attempt 的 audit marker；额外语义审计不前置阻塞。

**必要修正/验证/成本。** 本项核心实现无需重构，控制返回缺陷转 X-08。若改 X-04/X-05，只跑本来有意义的回归：目标失败身份/来源变更、健康兄弟自然推进、pause/cancel/hold 优先、未知终止 fence、replay 不新增第二调用、稿件/批准/次数保持。不要每次普通恢复都复制几百个文件并完整重读 native events；标准恢复入口及必要 audit 已承担机械核验。历史脚本可保留为证据，不升级成第二个运行控制器。

**兼容。** 现有十二项均已交付，不适用 repair-runtime；不动它们。prompt/docs 或真实语义规则变更仍不混进 tools-only 续接。无需再次请求用户批准已明确的工具故障保进度政策，也不恢复已删除的一般来源/语义迁移系统。

## X-08：控制事务的 `committed=false` 不覆盖整个多步骤命令

**现场与新发现要分开。** FMHA 的一次 busy 已由真实 state、calls 和 attempt 空增量证明未提交，串行重放只增一次调用，现场处理正确。但 `cli.py:592–594` 对所有逃逸的 SupervisorBusy 一律报 `committed=false`。`monitor.request_restart:751–777` 已先提交 retries 授权，`780–784` 再启动 background；后者任一锁竞争都可逃逸。`start_background:586` 先登记 launch，`594–599` 可在 Popen 后再取锁；此时“未提交”甚至不能代表没有启动进程。

**本轮重现。** 在纯内存 MemoryStore 上调用真实 `cli.main`→`request_restart`，仅把静态预检和进程启动边界替为假对象，在 second-lock 注入 SupervisorBusy：CLI exit=1、stderr `{"status":"busy","committed":false}`；ledger 已有 **1 条 `authorized` retry**。无模型、无真实进程、无真实运行写入。测试证明回执误报，不证明重复调用已在线上发生。

**最终行为与步骤。**

1. 只在确知尚未发生任何该操作的持久提交时返回 `committed=false`。登记后启动/投递阻塞，应返回原 request/launch ID 和 `committed=true`、delivery pending/unknown；终止未知不能重发。
2. 在 `request_restart`、`start_background`、maintain、hold release、extract 登记/启动、普通 resume、monitor start/stop 的已有提交点逐一检查。使用已经登记的 request/launch 作为幂等身份，不增加一套控制日志。
3. 已提交后的锁等待只重试事务进入或回报 pending，不重放整段副作用。直接复用 `Store.control_transaction` 的取锁机制时，不把 Popen、signal、额度增加或 apply 重跑进循环。
4. 顶层 SupervisorBusy 兜底若无法证明整个命令未提交，必须标未知并指向 status/inspect；不能继续默认 false。控制输出沿已有 request + delivery 形状解释，文档说明哪些字段证明登记、哪些只证明启动/结清。

**验证。** 必须覆盖登记前 busy、登记后/Popen 前 busy、Popen 后登记回写 busy、返回中断后按同 ID 对账、重复 CLI、后来的 pause/cancel/hold。断言真实 durable ledger、进程启动计数、调用额度增量，而非只 mock 返回 JSON。本轮现有“busy 不登记或 signal cancel”测试通过，但它只覆盖首次取锁，不能作为本缺陷反证。

**迁移/复杂度。** 中等、优先级高。改未来返回语义，不回写旧请求；原库的 request_id/failure_id/launch_id 已够用。不为每次 busy 再问用户批准；已有操作授权仍有效，关键是先确定提交事实。可删除凭“nonzero exit/没stdout/committed=false”整命令盲重试的脚本分支，以及一概 false 的兜底断言。

## X-09：KDA 的缓存 stale 与返回内容时效混为一谈

**当前离线观察。** run `bda2a4ba-47f7-4ae2-8a9b-9e0081c41cf9` 的 summary/state 都为 revision **155**、`ready/published=true`、checkpoint `d398e8618598d6d4bbaeffb419cb065530361c78f2fc5e091ae76376af38f647`。summary 所存 state inode=38745417、size=318852、mtime_ns=1795160724442225334，均与当前 state 相同；只 `ctime_ns` 从 **1795160724445403052** 变为 **1795160724450241587**，相差约 4.84ms。ctime 变化的底层原因本轮未证实，不能编成保存异常或缓存损坏。

`store.py:275–294` 检测 stamp 不同后读取 state 重新 summary，但只把 observation.source 改成 state，保留 status=stale、reason=state changed after projection。现场 check-9-84-inspect 已是这个结构；本轮只读调用重现同样结果。因此多次 status --full 的临时做法可靠，却在重复验证已经从权威 state 返回的结果。

**最终行为。** 区分 `cache_status` 与本次返回投影的 `observation.status/source`。fallback 成功读取稳定的一次 state 后，结果明确为 source=state、当前采样，缓存另标 stale/missing；若读前后 state 真变化，则保守标变化/未知，不宣称深验。不得仅忽略 ctime 来掩盖真实漂移，更不能为修显示 resume 已发布任务。

**修改与验证。** `Store.read_summary`、`cli._print_observations` 及相应说明。保留轻量 stamp 快路径、原 state revision 和一次 fallback；可记录缓存生成失败的可见诊断，但不让观察缓存失败回滚已提交状态。测试同 bytes metadata 改变、实际内容/revision 改变、缓存缺失/坏 JSON、state 读取期间变化、读取不产生写入。

**迁移/复杂度。** 小改，不必增加缓存修复命令/后台服务或让 inspect 写 summary。原 summary.v1 缺新增字段按读取事实展示，历史记录不改。fallback 可能仍需读约 319KB 的 state，这是可接受的只读成本；没有证据要求为本例设计另一种持久缓存体系。删除将“缓存过期”一律解释为“返回状态过期”的人工流程。

## X-10：superseded 保护 latest 正确，但聚合告警仍包含旧样本

**事实。** `monitor.record_scan:219–222` 已按每 run 实际 observed_at 阻止旧样本覆盖 latest；本轮时序回归通过。现场 check-9-57 按 superseded_run_ids 再 inspect，处理正确。但函数 `243–244` 仍从全部 rows 汇总 anomalies，`Clock._complete` 再计该数组；人读输出先打印旧行和 next 建议，之后才提示 superseded。

**本轮隔离重现。** 先记录 revision2/ready/t=200，再提交 revision1/invalid_receipt/t=100 的样本：latest 仍 ready，superseded 包含该 run，而顶层 anomalies 仍带旧 framework_or_protocol。它是展示/聚合语义的残余问题，不是旧状态重新夺取运行授权；monitor 本身不自动 resume。

**最终行为。** 保留原始 rows 与 superseded IDs 作证据；“本轮有效 anomalies/计数/行动提示”只来自未被 supersede 的采样。被超越的行在人读呈现前就标历史，避免打印看似当前的恢复建议。不把 latest 的旧完整对象冒充新的现场采样，不为此循环 scan 或改时钟。

**位置/验证/成本。** `monitor.record_scan`、`Clock._complete` 的计数和 CLI 展示；补旧失败被新成功取代、混合多 run 一部分 superseded、采样元数据不制造 changes 的测试。小改、无 state/receipt 迁移；保留既有旧观察保护和身份未知不抢 claim 的修复。可删 Supervisor 为“总体告警含旧失败”反复解释的模板性步骤。

## X-11：显式 stop/start 的新监控代际继承旧到期时间

**事实。** `Clock.start:229–244` 在 disabled→enabled 时增加 generation、改 thread/run_ids，却保留 last_check_number、next_due_at、last_scan；只有 last_check_number<0 才 initial scan。`_arm` 用旧 due 构造新代际提醒。现场 `monitoring/start.stdout` 的新 generation9 启动后 due 仍为前批的 2026-11-19 15:06:08（北京时间），已迟 24909.8 秒，所以立即提醒；不是 Extract 已失败，也不是队列重复投递。

**当前处理与推荐。** 现场按真实 ack/scan 登记没有丢失控制事实，但调度起点不符合新绑定的直觉。推荐：**显式 stop 后 start 新代际应先做一次新 initial scan，再从本次完成时刻安排 due**。同一 enabled 代际的 daemon 故障重启应保留原提醒 ID/ack/claim/due，不将它当新周期。两者必须区分。

**位置/步骤/验证。** 在 `Clock.start` 的 disabled 分支重置本代际序号/next_due，旧提醒按既有 history 归档；重新 initial scan，而不是拿旧 last_scan 当新范围已检查。不清空历史或未确认 queue 归属。测试新 thread/新 run scope、同 scope 显式停启、enabled daemon 接管、旧提醒不能被新代际 ack、stop 后无活动 due、旧发送未结清时仍拒绝启动。

**迁移/成本。** 小改，未来显式停启才生效；没有理由现在启动 monitor 验证。保持 epoch 和北京时间显示分工，disabled 状态保留的 due 仍只作历史。无需用户再次批准已授权的提醒范围；新外部投递授权仍由原上层约定承担。

## X-12：空目录与 unrecognized_model 的正确问题归属

**事实。** `runner.create_run`、`monitor.register_start` 要求 output 不存在；`publication.py` 的原子 no-replace 连已有空目录也不覆盖。当前仓库用 .gitkeep 预置叶目录，启动前已精确确认并删除十二个空占位及空目录，随后正常注册。它是两种入口约定碰面，不是内容丢失 bug。

**最终行为。** 保留无覆盖交付；在开始说明中明确输出叶目录须不存在、catalog 占位需先确认清理。不要自动 rmtree 输出、默许“看起来空”覆盖，或为这个正常前提增加通用合并输出模式。现有授权足以清理明确空占位，不必制造新确认。

**模型提示。** 本批 stderr 的 unrecognized_model 没有阻断调用；原生 reported 为 qwen3.8-max，requested 为 qwen3.8-max[1m]。请求 model/effort、原生报告和未知项的现有分层正确。既不擅改模型，也不把无 error 等同于服务端确实提供 1m 上下文或 max effort。无需真实模型调用来“证明告警可以忽略”。

**验证/迁移/成本。** 保留已存在目录/符号链接/不同发布内容拒绝及同 manifest 重放的现有测试；若仅补启动说明，不新增实现镜像测试。无需代码修改、历史迁移或 issue 声称已修模型服务。可删针对此告警无依据自动重试/换模型的建议。

## X-13：取证脚本错误不能归咎协议，但读取接口可少量补齐

**事实。** indexed Sparse 脚本把默认人读 audit 当 JSON；GDN 把子命令参数放在前面、把 accepted_receipts 的摘要字符串当对象；GEMM 比较 proposal 与 Runner 派生 receipt 完全相等、过早读取新 attempt_usage。它们都是调用方对现行格式/结算边界的误用，修脚本后没有多发模型调用，见两份恢复 README。`audit --json` 已存在；status 默认 JSON，inspect/audit 是否 JSON 各有明确参数；不能因为一次脚本写错就新增全套适配层。

**具体最小改进。** `tools/README` 增一段机器读取约定/现有命令样例：proposal 的选择字段与 receipt 的机器字段对应、accepted_receipts 映射到摘要、usage 以 settled 判断完整性、CLI 非零不代表控制未提交（X-08）。脚本先检查输出形状与 settled，再选需核对的语义字段；不要接受任意字符串/对象互转来吞掉格式错误，也不要把缺 usage 填零。

`Store.summary:1158–1159` 的 scope_fields 目前省略 receipt_repairs/invalid_receipts/cli_failures/check_failures/tool_failures。推荐把这些**已有账目**加入 summary.calls.scopes，让轻量 inspect 可区分调用预算与补正策略停止；无需新 ledger、全量 native replay 或新增“恢复审核 API”。诊断里的准确失败 attempt 已有，继续按它读，不退回 run.last_attempt_id 猜 P5 兄弟。

`framework_or_protocol`、`call_limit_or_single_decision` 是当前粗分组，不是工具故障或32次预算耗尽的证明。展示应优先给精确 reason.code、可补正/硬拒绝的具体诊断与实际计数；不能据这个分组直接决定 repair-runtime、重开 run 或追加额度。

**验证/迁移/复杂度。** 小改：记录字段输出类型与 null/未结清值、失败时 JSON/stderr 语义、正常 busy前/后提交差异。summary 增加可选展示字段不变更原账目，脚本应按 schema版本和字段含义读取，不改历史 proposal/receipt。优先删除临时脚本里重复标准 audit/恢复校验的代码；旧脚本只作当次证据，不纳入长期公共 API。

## X-14：不处理 kernel_wiki 自身缺陷，保留来源/提取责任分界

本轮 RECAP 中 GDN fence→arrival 缺聚合、indexed Sparse 零 tile 屏障、BF16 FMHA warp/split_range/fused 状态条件、KDA scale=0 参考差异及 mixed Sparse 外部比较器/逐输出/机制 A/B/NCU 缺口，均不在本轮 kernel 修复范围。Extract 必须表达固定实现及其条件，不能补造 GPU 证明、验收通过或理想同步行为。

当前已保留的源码风险及用户性能例外，不因十二项交付成功而消失，也不因后续工具修正自动变成 source-only 的新批准。是否 schema 正文过强或遗漏条件由同轮 SCHEMA_AUDIT 核对；若有正文问题，应按其授权/修订方式处理，不能以恢复工具偷偷替换已发布 bytes。

本项无需修改、迁移或新测试；保留来源证据及排除理由。与之相关的 `source-defect-abstraction-boundary` 既有决定不等于允许转录错误。删除“发布成功=kernel风险修复/全域验证”的含混总结即可。

## X-15：P2 revise 的空 plan 被错当作待检查选集

**真实证据。** GEMM run `2b64ba84-...` 的 reviewer `b5e8fa14-ef9a-4d29-b3e0-5f9673d83f17`、FP8 MoE run `5be021ee-...` 的 reviewer `7a274da1-14ab-456f-80af-f76f6403807e` 均是 `revise / plan=[]`，均被 accepted_receipts 接纳；其 `check.json` 均为 exit1 / `P2 selection requires root and at most 10 designs including root`。此前 worker plan_draft 分别含 root 的 7 项/3 项，非空且合法。来源候选是 `p2-revise-empty-plan-misattributed` 和 `p2-revise-empty-plan-snapshot-diagnostic`。

**根因与影响。** `validate_receipt` 对非最终决定不要求 plan 等于已认可选择，这是合理的；`runner.accept_result:610–612` 却对 approve/revise 都以 receipt.plan 传入 P2 snapshot，`artifacts.validate_index:432` 因空选集报错。revise 对退出1的容忍使流程继续，但 `check_failures` 增加，`stages.advance:640–641` 将假 checker_input_error 发给 worker，诱发对本来正确选择的误改。P2 的只读 helper 用的是 plan_draft，因此还出现“作者自检无法复现”的反差。

**最终行为/位置。** P2 reviewer **revise** 的结构/目录检查对象是实际读到的 worker 计划 `state.plan_draft`；approve 仍核其 plan 与 plan_draft 一致后检查同一候选。reviewer 想改变选择时用问题和正常退修表达，不把它的空/建议 plan 作为已经安装的选择。不跳过整个 validate_index，也不要求退修者伪填已批准计划。修改 `runner.accept_result` 这个参数选择并补集中测试，真实坏目录仍产生真实 checker 错。

**材料展示的相关次级问题。** `prompts.receipt_record:136–147` 用正式 selected_designs 判断 instance_relation。首次 P2 尚未批准时 selected_designs 为空，旧 worker plan 的所有实例就显示 removed；这是“尚未选定”和“已从正式选择移除”混淆，不是候选声称的真实批准回滚。P2 当前草案应标待审/当前 plan_draft，正式选择已存在时才按其关系说明；这些标签不产生批准。原 `plan_draft` 本身并未被这次 revise 清空。

**验证/迁移/成本。** 小改。需要从真实 P2 worker complete→reviewer revise/空 plan→下个 worker 的流程测试：合法目录无假 check_failures，原计划保留，真实退修意见仍在；坏目录仍报错；approve 改 plan 仍拒绝；首次待审不叫 removed，已移除旧实例仍叫 removed。历史 check/receipt 不重算或删账，只注明旧工具诊断归属错误；未来工具续接保留审计点。无须增加一次额外 review 或重开 P2。

## 相关既有修正的再评价

| 既有稳定 ID/修正 | 本轮结论 |
| --- | --- |
| receipt-evidence-repair | 错误分类与安全补正保留；X-01～X-05 解决它未涵盖的契约、诊断与累计策略 |
| failed-node-recovery | reason 去重送达、目标身份、健康兄弟/计数保护保留；不是新增恢复框架的理由 |
| progress-preserving-continuation | tools-only 范围已采用，非等待用户重选；合法草稿与完成度分层保留 |
| monitor-wait-stale-observation / registration-lock-retry | 最新观察、等待后重核 claim、取锁后时间均正确；X-10 只补聚合展示，不推翻原竞态修复 |
| monitor-beijing-time-display | 显示约定保留；X-11 是新代际调度，不是时区问题 |
| p5-handoff-attempt-locator | 失败 attempt 定位仍正确，旧 P5 缺身份时不猜兄弟 |
| p4-index-layout-hard-stop | 已删无意义排版前缀，当前相关回归通过；不重建布局异常补丁 |
| proposal-summary-ownership | 已明确 summary 由模型填；不能顺手改为工具自动生成语义结论 |
| normalizer-diff-not-visible | 首差异可见性有用；其保真边界交本轮 schema 反例审计，不能仅据旧测试默认安全 |
| p6-retained-scope-inventory | 完整文件 inventory 与局部解析的职责保留；不把允许命名等同允许修改 |
| maintenance-empty-poll-control-lock | 无请求读快路径/有请求锁内复核保留；X-08 不应以放弃锁来修返回语义 |

## 最小实施顺序与验收闭环

1. **先修 X-08、X-15 两个事实错误。** 改控制提交回执、P2 检查对象和材料关系标签；独立验证每个副作用边界和真实退修流程。
2. **一组完成 X-01～X-05。** 机械义务数据→精准/累计诊断→原角色有界重交，贯通 schema、提示、接收、反馈和 scope 预算。不要分成十个不共享检查逻辑的 patch。X-06 只按 schema 审计采纳必要澄清。
3. **一组完成 X-09～X-11、X-13。** 缓存/采样/代际三个时间概念清楚，聚合不再将旧告警算成当前，输出已有计数并说明机器读取约定。
4. **按变更执行必要验证。** targeted 离线负例→相关完整现有模块→fake backend 的六阶段收口。确认 prompt/JSONschema 条件在所支持 backend 输出协议上可用；没有真实模型授权/必要时，不以 fake CLI 测试冒称模型实际补正成本已经下降。只有发生新变更/失败才重复扩测。

全程保持：固定 source/任务/语义 rules；原次数不退款；无身份或权限异常的稿件只在正常事务中采用；失败不变批准；健康兄弟继续；未知进程归属保持 fence；暂停/取消/hold 优先；P6 和发布不跳过。工具修复续接沿已决定的 prepare/apply/resume 与 audit marker 政策，不将额外审计或新执行器身份变成默认重跑。

当前十二项已发布，不实施修缓存式 resume、不对旧回执补数字/类别、不自动删 continuation_audit，不重写最终 schema。若新提示/docs 修改与工具修改一并开发，它们只进入新 run 的规则快照；对任何另一个未完成 run 的 tools-only 换绑，仍须以该 run 原规则树为基底，只放入 Python 修复。

本轮不需要新的源码/语义迁移或每次恢复审批决定。X-05 的统一预算是本报告推荐的自动推进默认值；若上层确有更低成本诉求才需定另一个阈值，而不是引入多套策略。是否在本次审计后实施，由主代理按当前用户指令统一汇总，不影响本报告已完成的取证。

## 本轮实际验证与未验证

- **23 项现有定向测试通过。** 第一组21项，19.375秒：证据义务/坐标/长行/非法路径、P4排版、receipt repair、scoped recovery、call limits、两项 monitor竞态、cancel首次锁竞争、原稿规范化与只读provider。第二组2项，37.236秒：executor repair 保留全P5草稿/健康批准，保持 blocked/耗尽预算。测试 PATH 屏蔽真实 claude/codex；第一组显式断言两者不可查到。
- **三类离线验证另列。** P2 handoff 的1200/1201、20/21边界与JSONschema差异；纯内存控制“已提交但false”；纯内存 superseded 聚合旧告警。它们是反例重现，不是产品修复后的通过结果。
- **只读取证。** KDA summary/state/stat/read_summary；两项P2 revise原回执/check；相关候选、恢复小JSON/README。没有扫大型原生events，未重新审计所有模型调用或33份schema。
- **未验证。** 上述方案尚未实施；没有验证新方案的真实模型返工率、服务端schema关键字能力、GPU正确性/性能、ctime变化的底层来源。现有23测试通过不覆盖新发现的控制提交后锁竞争和P2空plan反例，不能抵消这两项事实缺陷。

机器覆盖表与精确测试/反例摘要见 [EXTRACT_FINDINGS.json](EXTRACT_FINDINGS.json)。
