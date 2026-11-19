# 本轮 issue 的用户决策审阅

2026-11-19，独立只读审阅。结论是：**本轮 14 个新增或实质补正的正式 issue 中，`progress-preserving-continuation` 需要用户讨论；其余 13 项可以按已有目标处理，不必再请求产品决定。** 续接需要决定的是长期支持的变化范围、沿用批准的证据标准，以及何时必须重新独立语义审阅。用户要保留有效进度已经明确，不再询问是否保进度，也不把 run UUID 或实现模块变成用户题目。

“无需用户决定”不等于“已完全解决”。本轮还看到三类须继续观察或验收的缺口：monitor 登记改成了无等待上界的取锁重试；MLA 的 singleton 类型错误已消除，但容器与元素的表达仍需核对；P6 最新候选的节点 draft 可能回退，正在随续接代码补正。它们都有可按现有目标确定的处理方法。

## 观察范围与证据等级

- 首次 Git 观察：2026-11-19 05:48 UTC；范围复核：05:52 UTC。实际 HEAD 为 `24d2b3d3ca9d4907c4e7efd9607cf78cfb1a52b2`，不是凭交接猜测的基线。
- 使用 `git status --short`、`git diff`、`git diff --cached` 和 `git ls-files --others --exclude-standard`，同时检查 tracked、staged、unstaged 和 untracked。复核时无 staged 改动，35 个 tracked 路径有差异；128 个 untracked 文件中，本批讨论目录 95 个、正式 issue 12 个、工具及测试 4 个、当前交付 16 个、archive 总索引 1 个。本报告尚未计入这些数字。
- 14 个正式 issue 为 2 个修改的历史 JSON 加 12 个新增 JSON：9 个登记为 `resolved`、2 个 `declined`、2 个 pending、1 个 `partially_adopted`。另核对了本批明确引用但未修改的 `reminder-ack-delay`，作为相关历史决定保留，不计作第 15 个本轮修复。
- 本批讨论目录整体 untracked，不能因 `git diff` 没显示正文而排除。覆盖其 36 份顶层 Markdown、恢复复盘、问题索引、分流/修复/交付报告和相关证据 JSON；`sparse_successor/inputs/` 内复制的源材料和旧稿按历史输入包识别，不把复制文件当成新的源码修改。撤回的 successor 方案和启动记录都有历史提示，本次没有采用其中失效的执行建议。
- 已读仓库 `AGENTS.md`、[issue 维护规则](../../schema_extract/issues/README.md)、[本批索引](ISSUES.md)、[台账覆盖核对](ISSUE_LEDGER_AUDIT.md)，再逐项对照正式 JSON、实际 diff、必要的相邻代码和实际发布文件。`archive/TRIAGE.md` 中本轮只改续接一行；其余大量历史条目不因出现在同一文件便全算作本輪待办。

本次只写本报告。没有调用 extract/monitor、操作运行或冻结状态、修改正式 issue/规则/源码/产物、运行模型或新增子代理；也没有重跑测试。下文测试结果明确引用已有报告，并以当前代码阅读复核其对应逻辑，不能解释为本次重新执行通过。续接代码与测试由另一代理并发编写，**观察到代码和测试文件不等于实现已验收**；主代理本轮最新约束是仅离线实施与验证，真实旧 Sparse 的 prepare/apply/maintain/resume 和新模型调用均尚待讨论后的安排。

源码覆盖关系如下，避免只审台账而漏掉报告中的改动：

| Git 变更组 | 本轮关联 |
| --- | --- |
| `maintenance.py`、`runner.py` 的 scoped feedback、`wakeup.py` | 空维护轮询、失败节点恢复、monitor 取锁重试和等锁后观察 |
| `reading.py`、`receipts.py` 的错误分类、`receipt_repair.md`、工具 README | 有限回执补正 |
| `artifacts.py` 的 P6 inventory | 合法保留 scope 被局部文件集合误拒 |
| Template 的 `yaml_format.py`、`rules/symbols.py` | 首差异展示、I/O 撞名诊断 |
| `maintenance.md`、`06_final.md` | 版本职责及隐含权限说明 |
| `executor_repairs.py`，以及 `runtime/snapshots/cli/store/audit/publication/receipts/runner/stages.py` 中执行身份、会话和候选指针差异 | 正在进行的续接；不混记为先前已通过的局部工具修复 |
| 7 个 tracked 测试差异及 `test_maintenance_polling.py`、`test_receipt_repair_policy.py`、`test_executor_repairs.py` | 对应上述改动；其中续接测试仍属进行中验证材料 |
| 6 个 `.gitkeep` 删除、四项当前交付和 archive 索引 | [输出准备/历史归档](ARCHIVE_INPUTS.md)及四项交付；删除占位不是六项交付证明，也没有证明 FMHA/Sparse 已交付 |

## 14 项正式 issue 的逐项判断

### 1. maintenance-empty-poll-control-lock — 无需用户决定

[正式记录](../../schema_extract/issues/archive/extract_tools/maintenance-empty-poll-control-lock.json)登记 `resolved 2.5.1`。旧 Runner 即使无维护请求也每 0.1 秒申请全局控制锁，六项并行放大争用；现场证明频繁短竞争，没有证明永久死锁，也未记录每次历史失败瞬间的唯一持锁者。

当前 `maintenance.consume_requests()` 先只读 `barrier`，无 requested/draining 请求即返回；有请求仍在原事务内重读、验封印。派发和发布的最新 ledger 门禁仍在，因此初次只读以后新到的请求没有被当成已经处理。这个修法符合已有维护契约，没有新的产品边界。

固定验收是：别的控制事务持锁时空轮询仍能返回且不写 ledger；请求竞争变化在锁内重核；新请求不能越过派发/发布。已有 3 项轮询及相邻验证见 [MONITOR_LOCK](MONITOR_LOCK.md)，恢复争用见 [MOE_RESUME_CONTENTION](MOE_RESUME_CONTENTION.md)。原冻结 b8 执行器仍可能发生旧争用，源码 resolved 不表示本批已升级。

### 2. failed-node-recovery — 无需用户决定

[正式记录](../../schema_extract/issues/archive/extract_tools/failed-node-recovery.json)从历史 2.3.0 补正为当前修复 `resolved 2.5.1`。此次新问题是 scoped retry 消费时先清 `owner.reason`，没有把拒因放进下一次 feedback；健康兄弟在途时，这使本来可用的定点恢复缺少修稿信息。本批因此等待 whole-run 条件，没有证据表明执行了丢反馈的 scoped 请求。

当前 `_consume_scope_retries` 在清理 reason 前按普通恢复同样的规则追加一次反馈；没有增加调用额度、清批准或改变终止 fence。应保留这个小修，不把历史“超时接管”政策误写为当前运行策略。验收看拒因持久化、重放不重复、健康兄弟与原计数/批准不变；已有定向验证及真实等待影响见 [首次回执恢复](RECEIPT_RECOVERY_0810.md)、[MoE 局部恢复判断](MOE_PARTIAL_RECOVERY.md)。不需要用户重新选择自动恢复策略。

### 3. monitor-registration-lock-retry — 无需用户决定，但 resolved 有活性边界

[正式记录](../../schema_extract/issues/archive/extract_tools/monitor-registration-lock-retry.json)登记 `resolved 2.5.1`。当前 diff 将 ack、绑定 scan 的 claim、commit、异常清理四处从一次非阻塞取锁改为 `control_transaction()`；后者只重试事务进入，失败后锁外 sleep 0.05 秒，事务正文不会重放。取得锁后仍核当前提醒、线程、代际和 claim。

**这确实改变了前台命令的等待行为。** `while True` 没有等待上界，持续占锁时 ack 会一直等，scan 的提交和异常清理也会一直等；不能用“只是复用原函数”把这个边界抹掉。不过没有发现新的模型调用循环，且实际根因是短争用。用户没有要求增加调用时长或累计 token 的停机预算，不能把这里当作设置此类预算的理由，也不必让用户决定一个武断的锁超时数值。

保留锁外重试的方向。工程验收还应覆盖持续争用时的可观察等待、CLI 可中断、无半完成确认、scan 旧 claim 可按真实进程身份处理，以及同一 reminder 后续能够续完、不重复观察或推进次数；如果有缺口，在这个控制请求范围内修正。当前 [修复报告](MONITOR_REGISTRATION.md)和 [集中审阅](SOURCE_FIX_FINAL_REVIEW.md)有短争用、stop/代际变化、四个入口等隔离验证，**没有提供持续占锁加真实中断场景的完整验证**。这是待补工程证据，不是新增用户审批。

### 4. monitor-wait-stale-observation — 无需用户决定

[正式记录](../../schema_extract/issues/archive/extract_tools/monitor-wait-stale-observation.json)登记 `resolved 2.5.1`。这是上一项等待行为引出的正确性问题：ack 使用取锁前旧时间；scan 使用旧进程表时可能抢走等待期间新出现的活 owner 的 claim。两者在隔离测试复现，不是已观察到的线上抢占事故。

当前 `Clock.ack` 在锁内取时间，scan 在当前 ledger 已有 claim 时重新检查 owner 存活，实际 run 观察仍在锁外。修复方向由真实时序和唯一 claim 所有权确定。验收看等待期间新活 owner 不被抢、ack 时间属于实际登记、stop/新代际使旧请求失效；[集中审阅](SOURCE_FIX_FINAL_REVIEW.md)记录先失败后通过的 12 项定向验证。本项没有消除第 3 项的持续等锁边界，不能混为同一项“全部监控可靠性已解决”。

### 5. p6-retained-scope-inventory — bug 修法无需用户决定；旧 run 采用修复归续接讨论

[正式记录](../../schema_extract/issues/archive/extract_tools/p6-retained-scope-inventory.json)登记 `resolved 2.5.1`。P6 reviewer 无写入时只解析 root，却以局部 `new` 文档集合检验完整的非 root 保留 scope，造成合法批准在进入回执校验前被拒。

当前 `_validate_scope` 使用完整 `new_files` 的 YAML 文件集合。相邻文件集合、字段权限、root/reviewer 只读和最终 selected-manifest 检查仍保留；无需放宽 scope、清掉合法历史 scope 或替模型改回执。固定验收是无写入 reviewer、仅目录修改 worker 可通过，缺文件、root 写入、受保护字段及最终集合缺项仍拒绝。[P6_APPROVE_SCOPE](P6_APPROVE_SCOPE.md)记录两项测试及真实副本的原 bug 复现、修复后 gate/回执/四节点严格检查。

该报告末尾当时把新 run 当作主要出路，现应以[恢复复盘](recovery_reassessment/RECOMMENDATION.md)为最新结论。正式 JSON 仍保留“正核对新运行接续”的较早措辞；这是状态文案应同步的工程工作，不是仍应启动已取消 successor 的指令。修复后机械通过不使旧 P6 response 成为正式批准，也不免除 Sparse root 的真实内容修正。

### 6. receipt-evidence-repair — 无需用户决定

[正式记录](../../schema_extract/issues/archive/extract_tools/receipt-evidence-repair.json)登记 `resolved 2.5.1`。已验证文件内的坐标/摘录选择和回执必需证据/结论漏项原来没有进入既有有限 repair，造成反复等 Supervisor。

当前 `EvidenceSelectionError` 只在路径、稳定读取和摘要等检查之后产生；上层把它及通用必填义务漏项转为 `ReceiptFormatError`。仍由原角色重新判断、重新取证并提交新 attempt；不能支持原结论时应退修或阻塞。原调用与 repair 次数累计，两次自动 repair 额度不扩展，第三次同类失败停止；维护仍需原显式恢复，裁决/签收不获得新循环。

验收看精确角色会话、新 nonce、合法旧稿、独立 reviewer、原计数、有效 revise/blocked 不变，以及身份/来源漂移/越权不被降级成格式错误。[分类报告](RECEIPT_REPAIR_POLICY.md)有 30 项定向验证，当前 diff 与这个边界吻合。[EOF 核对](EVIDENCE_LINE_ENDS.md)证实部分 +1 来自 Read 的虚拟 EOF 行，但不支持自动截断或 +1 容忍；Sparse 另一例 +2 的具体成因仍未证实，不应统一归为 EOF。增加可能实际发生的补正调用是利用已有额度，不是无限隐藏重试。

### 7. normalizer-diff-not-visible — 无需用户决定

[正式记录](../../schema_extract/issues/archive/schema_template/normalizer-diff-not-visible.json)登记 `resolved 2.5.1`。长字段旧 warning 只展示相同前缀；当前 `_shown_change` 显示首个 Unicode 字符差异的 1-based 位置和局部上下文，供 identifier/reference 和句号警告共用。

代码只改变诊断，不改变转换结果或字段限制；明确“first change”，没有假称展示完整 diff。固定验收是长字段前/中/尾、Unicode、空白保留及规范化结果字节一致。[报告](NORMALIZER_WARNING.md)记录 68 项检查和真实 968 字符案例的相同输出 SHA。无需扩展成完整 diff 产品，也不能把新展示说成旧冻结日志已被改善。

### 8. inplace-output-input-name-diagnostic — 无需用户决定

[正式记录](../../schema_extract/issues/archive/schema_template/inplace-output-input-name-diagnostic.json)登记 `resolved 2.5.1`。MoE 曾为消除 I/O 重名而删掉必需的目的地 Input。当前 `symbols.contract` 复用位置表，报冲突双方路径；Input/Output 冲突时提示保留两个角色、分别命名，并且仅在确为写回目的地时说明关系。

没有依据 shape/dtype 推断别名，没有删改对象，也没有改变统一命名空间。固定验收是不同组的冲突仍拒、异名合法、shape/dtype 相同或不同都不推断关系、只读输入不变。[诊断报告](IO_NAME_DIAGNOSTIC.md)记录相应组合和既有用例通过。这个修复不能机械证明以后每次模型都不会漏 Input，但不足以据此要求新的字段或审批流程。

### 9. maintenance-version-bump-outside-field-cap — 无需用户决定，含 P6 Version

[正式记录](../../schema_extract/issues/archive/extract_workflow/maintenance-version-bump-outside-field-cap.json)登记 `resolved 2.5.1`。维护 worker 曾误称 Runner 代填 Version；Sparse P6 reviewer 又把 Version 单列为授权字段而被拒。这两例同根，但第二例不是第 5 项 inventory bug。

当前维护与 P6 提示各补一句：worker 随获准内容改动递增版本，Runner 检查；P6 隐含版本权不单列 scope。既有 `_erase_fields` 和 `_versions` 未改，无理由另建 Version 独立写入口。验收看有权改内容时版本可递增、改内容不增版本仍拒、root/reviewer 仍只读、显式 Version scope 仍按原契约拒绝。[维护说明](MAINTENANCE_VERSION_GUIDANCE.md)、[P6 核对](P6_VERSION_SCOPE.md)保留文字/AST 和六项边界验证。实际冻结提示未更新，不能把新一句话作为旧模型已经收到的说明。

### 10. tuple-abi-input-typing — 不重开类型扩展决策；实例仍有表达核对

[正式记录](../../schema_extract/issues/archive/schema_docs/tuple-abi-input-typing.json)为 `declined`，拒绝的是新增类型机制或把 `python_none` 扩成 tuple 的提案。DOC1 已允许有明确定义的开放 dtype，而 `python_none` 明确只表示单个 Python None；原 `(None,None,None,None)` 的 singleton 登记确有错误，不能因 decline 忽略。

[维护和交付记录](DELIVERY_MLA.md)证明原流程完成了限域修订与独立 verify；本次实际读到 [MLA root:36](../../schema_wiki/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/root.yaml:36) 为 `shape: [{element: 4}], dtype: python_tuple`、meaning 为“4 Python Nones”，相邻整数 tuple 是相同 shape 加 `python_int`。**原 singleton 矛盾已消除，但当前写法没有明说 python_tuple 是整个容器类型还是元素类型，不宜称表达已完全无疑义。** [TUPLE_TRIAGE](TUPLE_TRIAGE.md)第一条原本就要求核清此点。

这是应继续核对的表达缺口，当前证据不足以仅凭形状一行宣布新硬错误或撤销交付。现有字段已有无歧义的处理方向：把完整非数值对象作为明确定义的整体值，或明确逻辑轴与元素类型；不能同时把 dtype 用于整容器、把 shape 用于元素而不说明。数值 tuple 可按逻辑元素表达，类型/meaning/必要约束需一致；default 不得被当成固定有效域。无需发明新的封闭类型表或泛化 ABI 字段。与第 12 项一起做文档样例和语义核对，若正式产物确需修订，再走其正常修订验收，本轮保持只读。

### 11. inplace-output-input-registration — 无需用户决定

[正式记录](../../schema_extract/issues/archive/schema_docs/inplace-output-input-registration.json)为 `declined`，针对“需要新增双登记规则”或放宽 Input 定义的主张。DOC1 已要求调用者全部对象登记为 Input、原地结果另列 Output、名称全契约唯一；目的地旧值是否参加计算不豁免该义务。

Sparse、MoE、FMHA 三项实例缺项通过原限域维护处理，新增 Input 异名并用 existing meaning/Compute 绑定输出。MoE 当前正式 [root:44](../../schema_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/root.yaml:44)可直接看到 `output_buffer` 与 `Output.output` 的关系。另两项维护完成的证据来自本批索引/进展与原维护记录，不把它们误写成两项正式交付。

固定验收是输入对象、输出对象、同一/分配关系、nullable/default 和旧内容有效域与来源同时一致。Sparse 的 NaN 初态必须容纳，不能默认有限，也不能收窄成仅 NaN；其余五项已核对 guarded 有限初态，不移用 Sparse 的特殊事实。不能从较宽 candidate 签名增加任务没有的 None/缺省分支。见 [分流](OUTPUT_BUFFER_TRIAGE.md)、[同类核对](OUTPUT_BUFFER_CROSSCHECK.md)。这些都是现有语义下可验证的修正，无须重新讨论“目的地算不算输入”。

### 12. non-tensor-parameter-registration — pending，但无需用户决定

[正式记录](../../schema_extract/issues/archive/schema_docs/non-tensor-parameter-registration.json)只提固定序列、框架 dtype 参数的 shape/dtype/meaning 示例，明确保持开放 dtype、不新增字段/类型登记表。当前 GEMM 的三整数 tuple 与 `torch.dtype` 已按此开放语义获审，MLA 的示例建议并入此 ID。

应直接补少量能分清层次的样例：数值序列的逻辑轴与元素类型；整体非数值对象的类型和值含义；`torch.dtype` 值与它指代的结果元素格式；tuple 内含 None 与整个对象 nullable/default 的区别。验收不能只看 YAML checker 通过，还应由独立读者从字段唯一还原对象/元素层次、缺省行为及有效域，尤其覆盖第 10 项现有写法。开放类型不等于类型含义可以含混，也不要求所有 tuple 强行使用同一 shape 写法。

没有发现需要用户选择“完整 ABI 或仅有语义的参数”的真实未决点：DOC1 的全部调用者对象义务已经明确。只有将来要让机器统一遍历异构容器、增加结构化类型语法，才是不同产品需求；本轮提案没有要求这些能力。保持 pending 直到示例和核对真正完成，而不是因为无需讨论就立即标 resolved。

### 13. location-storage-level-registration — pending，但无需用户决定

[正式记录](../../schema_extract/issues/archive/extract_workflow/location-storage-level-registration.json)指出 P1 仅最小登记执行层级，DOC5 又要求 Storage.location 沿用硬件区名称。GEMM 当前硬件区列 smem/tmem，Design 使用 register，但没有对应明确名称登记，造成作者反复判断。

这是有事实依据的指引缺口，但现有目标已足够约束修法：DOC2 的 Execution 摘要可说明执行者的物理资源归属；DOC5 明列 register，并将存储名设为开放取值；提取的硬件占位豁免不提供容量保证。应澄清源码明确的存储名称及其所属层级怎样在现有最小登记/摘要中保持一致，不把纯存储名伪造为新执行角色，不为了登记一个名称搜全套设备参数。

具体句子/放置位置属工程文档选择。固定验收是：P1 作者能只凭给定材料登记名称，后续 location 可准确追到该名称及所属范围；缺容量/编译资源报告仍不构成占位提取阻塞，普通非豁免硬件校验不被放宽。寄存器设计槽位也不能因此升级成“编译后确无 spilling”。无需用户决定增加硬件调查、字段或强制测量门禁；本项当前尚未改文档，不能因 GEMM 已交付而算已解决。

### 14. progress-preserving-continuation — 必须用户讨论，当前仅部分采纳/离线进行中

[正式记录](../../schema_extract/issues/archive/extract_workflow/progress-preserving-continuation.json)现为 `partially_adopted`，version 保留历史 2.4.0。旧能力确曾实现，20261117 精简确实删除了换绑/跨身份承接；当前固定身份内 maintain/恢复仍在。这次补正不是已经重新实现的版本发布，Sparse 个案恢复也不能替代更广能力验收。

实际缺口有三层：工具 bug 修复无法被旧冻结 run 使用；原批准、当前适用性和最新候选没有被分别判断，导致“换身份”被推成从 P1 重来；产品精简后能力说明与 issue 状态未同步。源码 31 文件、完整四稿和目录、47 次原调用都可追踪，不存在仅由这次 P6 gate 故障推出全流程重写的依据。原被拒 P6 approve 仍不是批准，root 的寄存器声明也确有应修范围。

05:52 UTC 观察到新增 `executor_repairs.py` 及相关运行/审计/发布身份链、P6 节点 draft 同步代码。候选方向是当前格式、同源码/语义规则、未交付且已结清的任务显式采用新冻结执行器，历史调用留原身份、新调用记新身份，保留原账，随后维护和新 P6。**这是待审方案和进行中实现，不是政策已经通过或真实续接已经成功。**

尤其不能把“非 Python 规则文件没变”当成语义兼容的充分证明：当前 `_target`/freeze 白名单允许 `schema_extract/*.py` 改动，而 Python 也承载批准、权限、来源和检查含义；`_review` 校验指定文件清单和一份兼容结论的字段，不能机械证明审阅者真已独立判断全部影响。必须有具体补丁审查证据。当前 `probe` 明确仅做机械 preflight/原批准来源核验，不会生成新的语义批准，这是应保留的边界。

固定工程验收包括原/新身份可追溯，真正完整 diff 受独立审查，最新候选不回退，批准适用性按范围核验，调用账不重置，重复登记幂等，封存与提交中断后可继续同一次恢复，取消/暂停/在途/发布屏障有效，后续同类 P6 改动仍能正常验收。旧失败 response 不回填，发布须新的当前 P6。当前观察到的测试文件覆盖了局部合成路径，不能代替完整中断/身份/真实个案验收；本报告不宣称任何一个进行中测试已通过。

## 报告里还有哪些事项，为什么不增加用户议题

| 事项与证据 | 独立判断、固定处理及验收 |
| --- | --- |
| P6 最新候选未同步节点 draft：[技术复盘](recovery_reassessment/TECHNICAL_RECOVERY.md)；`runner._draft`、`stages.advance` 的新 diff | 是真实的进度回退风险，不在较早 ISSUE_LEDGER_AUDIT 的四项补记中。最新候选是 `66ad…`，packed draft 原指较早 `71ec…`。应只把真实已接受 P6 worker 改过的节点指向最新待审稿，保留前指针和 producer，不提升 approval/merged。这是第 14 项明确应覆盖的工程子问题，不能漏记，也不需要用户选择要不要保留最新稿。验收需回 P3/P5 后仍读到最新 packed 修改。 |
| Sparse root 的 208 / without spilling：[REGISTER_CLAIM_TRIAGE](REGISTER_CLAIM_TRIAGE.md) | 208 来自三个主数组的 64+16+128 个 32 位槽小计，不能笼统删除成“无来源”，也不能当总寄存器或无 spill 证明。按现有 P3 规则保留设计账目，收窄实现保证，独立审相关全文和依赖，再新 P6。没有匹配 cfg2 的编译/测量证据；无需为修掉夸大保证要求用户启动 GPU 调查。 |
| MoE 已交付 P6 留下两项措辞：[DELIVERY_MOE](DELIVERY_MOE.md) | `split_count float4 loads` 与 up/gate 两路各一条的总量、Fc2 未生产 act 字节范围，均已有可查语义与 reviewer 边界。报告不能将它们抹成“已完美无缺”；按实际 Math/source 核准后做相应文字修订/验收即可，不构成新字段或算法路线选择。本次没有重新完整审 MoE，不能凭交付核对给出新的语义批准。 |
| P2 的正式条目缺直接 baseline 链接：[P2_INDEX_CHECK](P2_INDEX_CHECK.md) | 正常退修。纯未来线索移回计划说明，实际 root 相关正式条目补真链接；已有前后稿和检查证据证明未要求 P2 提前完成未来 YAML。无需修改 checker 或放宽阶段门禁。 |
| maintain 返回 busy/后续 delivery 忙：[OUTPUT_MAINTENANCE_REGISTRATION](OUTPUT_MAINTENANCE_REGISTRATION.md) | 必须区分意图登记、派发、内容完成。只用相同载荷/稳定请求身份核对幂等，不把 delivery 失败当成完全未登记而新建重复任务；applied 不是 complete。没有必要另造工具 bug 或让用户批准再试同一请求。 |
| 单次模型生成较长、权限拒绝：[RUNTIME_PROGRESS_0706](RUNTIME_PROGRESS_0706.md) | 当时所有已观察工具调用都有返回，长间隔主要在模型生成；被拒扩展 Bash 命令有既有允许读法。事件活跃只证明活动，不证明内容或一定完成。没有当前无限工具挂起证据，不由此引入时间/token 停机预算或扩大权限。 |
| 四份交付核对的 audit partial：[GDN](DELIVERY_GDN.md)、[GEMM](DELIVERY_GEMM.md)、[MLA](DELIVERY_MLA.md)、[MoE](DELIVERY_MOE.md) | 历史拒收回执的 artifact validation 未完成，被诚实报告为 unavailable。errors 为空、退出 0、当前发布绑定正确都不能把 partial 改称完备证据证明；另一方面不能仅因旧拒收存在便撤销已另行核验的当前发布。保留原因和范围，按各自证据验收，无新产品选择。 |
| source-only 比较器、实现域限制和上游差异：同上四报告 | 已有来源缺项/事实分流边界仍适用。交付不证明数值/GPU/性能已复现，声明来源缺口不等于可把它当设计保证。不是因出现在新报告便自动新开历史全部 kernel issue。 |
| 运行归档、占位删除与 successor 历史输入：[ARCHIVE_INPUTS](ARCHIVE_INPUTS.md)、[已取消确认](recovery_reassessment/SUCCESSOR_CANCELLED.md) | 保留归档映射、旧稿和失败/取消用量。仅以新身份/真实发布证明当前交付；旧指令按时间作废。Supervisor 交接文件中较早“勿恢复旧 Sparse/已暂停新 run”的文字须在后续按最新决定同步，这属第 14 项的说明一致性，不单独向用户请示。 |

未修改的 [reminder-ack-delay](../../schema_extract/issues/archive/extract_tools/reminder-ack-delay.json)已另读：它处理历史认证配置事件和入队未 ack 的同 ID 有限补送，当前发送次数/间隔逻辑不在本轮 diff，和锁竞争及旧观察不同根。保留原决定和“不保证绝对投递/主管处理”的范围，不借本轮现象重复登记或重新询问用户。

## 需要用户决定的两条原则

以下都归第 14 项。候选实现提供具体可评估材料，但不能倒过来替用户决定长期边界。选择以后，实际每次兼容审查、测试及恢复由工程按原则执行，不增加逐步用户审批。

### 决策一：首版保进度续接承诺覆盖哪些变化？

**决策问题：本轮把“当前格式任务采用工具修复后保进度继续”确定为稳定能力；是否同时承诺规则语义变化、甚至源码/任务变化也能按影响承接？**

| 方向 | 实际承诺与代价 |
| --- | --- |
| A. 本轮只支持执行器修复（建议） | 固定源码、任务和语义规则；只为修复执行行为/恢复既有契约的补丁提供显式续接，未交付且已结清。原稿/原账保留，批准按下一条原则处理。范围最小、可用当前 Sparse 验收；规则或源码变化仍保留材料与影响分析，但不宣称已有自动承接能力。 |
| B. 同时支持当前格式内的规则语义修订 | 源码/任务不变；规则本身变化也记录新语义身份，并逐阶段/节点独立判断哪些批准可用、哪些需审/改。更适合规则频繁迭代，但必须新增语义差异与当前适用性的完整验收，不能用“只改 Python”或文本哈希代替。 |
| C. 再纳入源码或任务变化 | 允许真实新任务身份承接旧内容，必须判断输入域、接口、来源、代表选择及依赖失效范围。复用面最广，也最接近通用迁移系统；不是本次同源码 P6 bug 所必需。 |

建议 A，B/C 明确留在同一 issue 的 remaining，未来有具体反例再扩大。依据是本次阻塞完全可以由最小工具续接加既有维护解决；20261117“不建旧版本兼容层”的目标也无需牺牲当前长任务恢复。无论选哪项，都不允许“没有现成入口”直接推成重做所有阶段；应先给出真实失效范围和剩余成本。

### 决策二：修复了验收 gate 后，什么证据足以继续沿用已批准内容？

**决策问题：当新 gate 能把一个属性机械重核为合格，但旧 gate 对该属性曾有漏检/误通过可能时，是允许按该属性的完整证据保留批准，还是对曾受该 gate 影响的范围必须再做一次独立语义审阅？** 这不是“保进度或全重跑”二选一，两种方向都保留原稿、账本和明确未受影响的批准。

| 方向 | 实际信任边界与取舍 |
| --- | --- |
| A. 逐属性证明，确定项保留（建议） | 独立工程审查先证明补丁恢复哪条既有义务、覆盖哪些批准对象；原批准来源、当前正文/依赖和新机械门禁均能完整核验时保留。涉及来源解释、批准结论可信度或无法机械证明的影响，必须定向独立语义审阅。节省能够严格确定的重复审核，但依赖补丁影响分析的质量。 |
| B. 对“曾可能误通过”的 gate 影响范围强制语义复审 | 单纯误拒/可用性 bug 仍可有据保留；只要修复暴露旧验收可能不可靠，就对实际受该 gate 覆盖的阶段/节点再做独立全文审阅，即使新机械检查已通过。不重写无问题稿件，不审无关范围；证据标准更保守，增加定向 reviewer 调用。 |

建议 A，并明确下列硬边界：修补身份/来源验证时若原证据真实性已无法恢复，不能称为机械可证；影响判断“未知”先对最小相关责任范围独立全文审阅，有跨阶段依赖证据才扩大；发现正文错误才派修稿，不先重写来解决审阅不确定。所有需要派审的阶段仍审其有效全文，不只审补丁几行。最终 P6 继续是当前完整成果的新独立终审。

本次 Sparse 是具体校准例：inventory bug 为合法只读候选的误拒，本身没有证据推翻 P1/P2/P4；root 寄存器保证是真实语义缺口，必须 P3 小修及独立复核；packed 后来经过 P6 修改，旧 P5 批准不覆盖新字节，必须审最新候选；另外两个分支是否保留看当前依赖适用性证据；旧 P6 approve 未被接纳，最终必须新 P6。具体需要几次调用是验收后的估计，不能提前承诺四次必成。

## 面向用户的精简议程

1. **续接的首版产品范围**：建议先稳定支持同源码、同任务/语义规则、当前格式的执行器修复；本轮是否要同时纳入规则语义修订，源码/任务变化是否明确后置？
2. **批准沿用的信任边界**：建议可完整证明的机械属性保留，语义影响或证据不确定定向独立全文复审；对于修复过“可能误通过”的 gate，是否额外要求其实际影响范围一律再做语义复审？

其余 13 个正式 issue、候选 draft 回退及已指出的文字/观察缺口按上述固定方案和证据验收推进。它们不需用户逐项审批，但尚未完成的工作仍保持真实未完成状态；代码修复、个案续接、独立语义审核、机械审计和正式交付分别记证据。
