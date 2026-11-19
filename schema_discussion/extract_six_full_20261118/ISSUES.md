# 本批问题与处置

## 当前问题索引与状态

北京时间2026-11-19 16:30收口：六项逻辑交付完成。Sparse走完正常P6，FMHA按用户最新指令由xhigh完成定点小修交付，原自动P6的`reopen/invalid_receipt`历史保留，不再打回循环。工具故障与潜在审计点随交付receipt保留，额外检查及其他issue修改等待用户指挥。见 [交付汇总](README.md) 与 [FMHA小修授权](recovery_reassessment/FMHA_USER_MINOR_FIX_DECISION.md)。

用户新增待办：[monitor-beijing-time-display](../../schema_extract/issues/archive/extract_tools/monitor-beijing-time-display.json)。巡检面向用户的时间统一采用北京时间（UTC+8），只改展示及说法；本次仅登记，内部调度时刻不变。此项新增于14项issue审阅之后。

2026-11-19 06:18 UTC：用户已明确决定工具故障修复后沿已有进度先完成，随schema留故障阶段及潜在独立审计标记，完成后再由用户决定额外检查/重来；其他issue修改等后续指挥。决定见 [USER_DECISION](recovery_reassessment/USER_DECISION.md)。旧Sparse与FMHA已分别通过公开工具完成执行器补丁换绑并实际进入新的正常P6 reviewer（总调用48/47），未追加P3/P5维护、未重做P1；[实际运行核对](recovery_reassessment/ACTIVE_CONTINUATION_STATUS.json)。待查标记随正式receipt交付，当前尚未宣称两项已完成。

2026-11-19 06:05 UTC巡检新增同根实例：FMHA P6 reviewer返回approve后，也被旧冻结执行器的局部inventory校验误拒，当前46次调用、无在途，完整三稿与目录保留。临时副本已精确复现旧gate失败/当前gate通过，真实状态未变；证据见 [FMHA复现](FMHA_P6_SCOPE_REPRO.json)。沿 `p6-retained-scope-inventory` 补记，不增加issue数量。Sparse与FMHA均未恢复或重跑，继续尊重续接方案的讨论边界。

2026-11-19 用户已授权原技术调查 xhigh 子代理负责旧 Sparse 的续接实施与监管，直至正式交付；此前仅研究、不恢复的阶段已结束，新 successor 的取消仍有效。用户同时要求把续接能力作为待讨论的流程 issue 记录清楚；原 `progress-preserving-continuation` 已补齐当前缺口、Supervisor决策原因、窄范围实施边界、较广产品问题及验收条件，保持 `partially_adopted`，不因委派或一次个案恢复宣称全部解决。

2026-11-19 04:59 UTC 补充：用户要求取消新 Sparse 调用，公开 cancel 已确认进程完全终止；回到旧进度续接研究，未恢复旧 run。见 [取消核对](recovery_reassessment/SUCCESSOR_CANCELLED.md)。原先从 P1 重建的计划已撤回，下面按时间保留的启动记录仅代表当时操作。

本次另查明原 [progress-preserving-continuation](../../schema_extract/issues/archive/extract_workflow/progress-preserving-continuation.json) 的 `resolved 2.4.0` 不代表当前能力：20261117 精简删除了执行器换绑及跨身份进度承接，固定执行器内维护仍在；当前格式长任务遇到工具缺陷后的续接缺口重新暴露。已沿原 ID 补正为 `partially_adopted` 并同步 TRIAGE，保留 2.4.0 历史采纳和版本，不新增已解决版本。事实及决策原因见 [独立复盘](recovery_reassessment/SUPERVISOR_DECISION.md)；主代理仅落实了其中的台账补正，工具/手册/提示改动仍待方案讨论。此项未实现或宣称恢复完成。

2026-11-19 按本批报告核对正式台账。此前四项工具发现仅在报告中，现已补齐三个新 ID，并沿 `failed-node-recovery` 补正一次既有功能遗漏；核对范围见 [台账审计](ISSUE_LEDGER_AUDIT.md)。下表是问题处置状态，不代替实时运行状态或算子交付验收。

| 问题与影响 | 正式 issue | 当前处置及证据 |
| --- | --- | --- |
| 无请求维护轮询高频争控制锁，造成登记失败、重复重试 | [maintenance-empty-poll-control-lock](../../schema_extract/issues/archive/extract_tools/maintenance-empty-poll-control-lock.json) | 当前源码 resolved 2.5.1；[锁竞争调查](MONITOR_LOCK.md)、[恢复争用](MOE_RESUME_CONTENTION.md) |
| scoped 恢复清拒因却未转反馈，妨碍定点修稿并限制并行节点恢复 | [failed-node-recovery](../../schema_extract/issues/archive/extract_tools/failed-node-recovery.json) | 沿原 ID 补正，当前源码 resolved 2.5.1，保留 2.3.0 历史依据；[首次发现](RECEIPT_RECOVERY_0810.md)、[本批 P5 等待影响](MOE_PARTIAL_RECOVERY.md) |
| ack/绑定 scan 的单次取锁失败浪费本轮提醒处理机会 | [monitor-registration-lock-retry](../../schema_extract/issues/archive/extract_tools/monitor-registration-lock-retry.json) | 当前源码 resolved 2.5.1；四处登记复用锁外重试，持续占锁无等待上界；[修复报告](MONITOR_REGISTRATION.md) |
| 等锁后旧时间、旧进程表会使 ack 时序失真或误抢新活 owner 的 scan claim | [monitor-wait-stale-observation](../../schema_extract/issues/archive/extract_tools/monitor-wait-stale-observation.json) | 当前源码 resolved 2.5.1；隔离测试复现后修复，未观察到线上抢占事故；[集中审阅](SOURCE_FIX_FINAL_REVIEW.md) |
| P6只读gate把局部解析误作完整清单，合法approve被拒且旧冻结run无法重交恢复 | [p6-retained-scope-inventory](../../schema_extract/issues/archive/extract_tools/p6-retained-scope-inventory.json) | 当前源码resolved 2.5.1，旧Sparse仍受阻、未交付；[真实复现与修复](P6_APPROVE_SCOPE.md) |
| 回执坐标、必需证据/结论漏项未进入已有有限重交，提取反复停等 | [receipt-evidence-repair](../../schema_extract/issues/archive/extract_tools/receipt-evidence-repair.json) | 当前源码 resolved 2.5.1；同类反复错误合并本项，不逐 attempt 新建；[分类修复](RECEIPT_REPAIR_POLICY.md)、[EOF 证据](EVIDENCE_LINE_ENDS.md) |
| 长字段 warning 看不见实际差异，造成反复自检 | [normalizer-diff-not-visible](../../schema_extract/issues/archive/schema_template/normalizer-diff-not-visible.json) | 当前源码 resolved 2.5.1；只改差异展示；[报告](NORMALIZER_WARNING.md) |
| Input/Output 撞名诊断不足，曾诱发删除必需输入对象 | [inplace-output-input-name-diagnostic](../../schema_extract/issues/archive/schema_template/inplace-output-input-name-diagnostic.json) | 当前源码 resolved 2.5.1；明确双方位置及保留角色的写法；[报告](IO_NAME_DIAGNOSTIC.md) |
| 限域维护未明确 worker 更新 Version 的已有职责 | [maintenance-version-bump-outside-field-cap](../../schema_extract/issues/archive/extract_workflow/maintenance-version-bump-outside-field-cap.json) | 当前提示 resolved 2.5.1；单句澄清，不改版本规则；[报告](MAINTENANCE_VERSION_GUIDANCE.md) |
| MLA tuple/None 类型矛盾，涉及真实接口登记合规性 | [tuple-abi-input-typing](../../schema_extract/issues/archive/schema_docs/tuple-abi-input-typing.json) | 规范扩展主张 declined，示例并入下列待决项；实例限域维护及独立 verify 已完成；[分流](TUPLE_TRIAGE.md)、[同类核对](TUPLE_CROSSCHECK.md)、[交付核对](DELIVERY_MLA.md) |
| Sparse、MoE、FMHA 缺调用者目的地 Input，散文不能替代字段义务 | [inplace-output-input-registration](../../schema_extract/issues/archive/schema_docs/inplace-output-input-registration.json) | 新增规范主张 declined；三项实例维护已 complete，仍分别遵循后续审阅和交付要求；[分流](OUTPUT_BUFFER_TRIAGE.md)、[同类核对](OUTPUT_BUFFER_CROSSCHECK.md) |
| 非 Tensor 序列、框架 dtype 值的登记示例不足 | [non-tensor-parameter-registration](../../schema_extract/issues/archive/schema_docs/non-tensor-parameter-registration.json) | pending，非阻塞说明建议；包含 MLA 合并建议，未改规则；[分流](TUPLE_TRIAGE.md)、[GEMM 候选核对](DELIVERY_GEMM.md) |
| P1 硬件最小登记与 DOC5 存储位置名称口径不够明确 | [location-storage-level-registration](../../schema_extract/issues/archive/extract_workflow/location-storage-level-registration.json) | pending，非阻塞说明建议；未新增字段或改变 register 产物；[GEMM 候选核对](DELIVERY_GEMM.md) |

`resolved 2.5.1` 表示当前工作树源码或提示修复，不表示本批执行器升级。本批仍使用 frozen `b8a1d5e368bf0e423081110ffc748945097a503b7eb2fd11fac6355650a58d8a`，旧轮询、反馈及回执路由限制仍可能影响它；新前台 monitor CLI 可加载修复，已有 daemon 未热改。上述工具问题没有数据丢失或错误发布的现场证据，也不能由此保证后续绝无风险。`declined` 针对规则扩展或合并提案，不是否认或忽略实例错误；实例维护与框架说明待办分别记录。

旧 [reminder-ack-delay](../../schema_extract/issues/archive/extract_tools/reminder-ack-delay.json) 记录认证事件后的未 ack 有限补送，与本批控制锁登记及观察竞态不同根；原决定保留。正常 [P2 目录退修](P2_INDEX_CHECK.md)、[维护意图已登记而 delivery busy 的幂等核对](OUTPUT_MAINTENANCE_REGISTRATION.md) 不另建工具 bug。来源限制及审计 `partial` 继续按 [GDN](DELIVERY_GDN.md)、[GEMM](DELIVERY_GEMM.md)、[MLA](DELIVERY_MLA.md) 报告披露：历史拒收回执未完成 artifact 验证不能写成全部证据已验证，`source-only` 等来源限制也未因交付消失；这些不冒充新工具缺陷。

## 按时间保留的原记录

| 时间（UTC） | 问题 | 处置与状态 |
| --- | --- | --- |
| 2026-11-18 06:16 | 批量启动时稀疏注意力返回 supervisor busy，注册已保留但尚未派发 | 核对无在途调用后，单项重试同一 extract 请求；复用 `8f0844f9-f279-4781-a19d-0216b7fc4ef5` 成功启动，无重复运行。暂未发现需要修改锁语义的依据，继续观察 |

worker/reviewer 候选随各 run 的 `candidates` 收集；正式框架问题按稳定 ID 和根因归入 `schema_extract/issues/`，不将提取产物或源实现差异混为工具问题。

- 2026-11-18 07:04:50 UTC：check3 首次提醒及补送均 ack 返回 supervisor busy / committed=false，未执行 scan。已委派新 xhigh 查锁占用与根因；当前运行未被暂停或重启。

- 2026-11-18 07:09 UTC：确认根因为无维护请求时 Runner 仍每 0.1 秒争用全局控制锁，六任务放大无用竞争。当前源 `maintenance.consume_requests` 已增加只读快路径，锁内核对和派发前屏障保留，针对性验证通过。详情见 [锁竞争分析](MONITOR_LOCK.md)。本批已冻结 executor 不变，未热改或重启；监督使用原公开命令恢复同一提醒，check3 已 committed，下一次 07:39:11 UTC。

- [首轮耗时抽查](RUNTIME_PROGRESS_0706.md)：所有观察到的工具调用均已返回，单次最长约 2.8 秒；存在 14–30 分钟的模型生成间隔。文本长度/命名自检失败正在由 worker 修稿，历史权限拒绝已有替代读法，尚未发现持续工具拒绝循环。该观察不等于当前阶段通过。

- 2026-11-18 07:40 UTC：GEMM worker 提出 `non-tensor-parameter-registration` 候选，关注 Python tuple 和 torch.dtype 等非 Tensor 输入的登记示例；reviewer 尚未确认，不修改字段规则、不作为已证实缺陷。GDN/Sparse 当前无 framework candidates。候选原回执为 run `06f3ed2c-daa0-46f0-a0dd-a7e1f4f4df31` / attempt `cf14892a-8c64-47af-b944-78d741b45a8f`。

- 2026-11-18 08:10 UTC：GDN reviewer 缺少四项 document_* 结论；FMHA worker 将绑定入口只归为 host，implementation 证据未覆盖绑定入口。原生结果与会话有效，属于回执遗漏。按 [恢复分析](RECEIPT_RECOVERY_0810.md) 公开续原运行，未放宽校验，未增加调用额度；FMHA 原稿保留，但需原 worker 在新授权目录续交，未由 Supervisor 合入。恢复命令登记见 RECOVERIES.jsonl。

- 本次诊断另发现 scoped resume 清除拒因而未传入下一次反馈；当前 runner.py 已与 whole-run 的去重反馈行为对齐，4项定向检查通过。本批恢复使用原冻结执行器的 whole-run 路径，未热改快照。

- GEMM 的 `non-tensor-parameter-registration` 已获独立 reviewer 确认，登记到 `schema_extract/issues/schema_docs/`，作为非阻塞说明建议；字段规则未改，提取继续。

- 2026-11-18 08:41 UTC：GDN 补齐回执后 P1 已获批准并进入 P2；FMHA 补齐回执的 worker 稿已采纳，进入独立 P1 reviewer。两项均沿原 run 续修，无需更改冻结规则或重建整个算子。

- 2026-11-18 08:43 UTC：MLA P1 reviewer 确认 `tuple-abi-input-typing`，涉及 `(None,None,None,None)` 与数值 tuple 的登记一致性；与 GEMM 候选可能同根。已委派独立 xhigh 区分现有开放 dtype 规则误读、说明缺口和当前稿件语义错误，再决定是否需原 run 的限域维护；当前字段规则与产物未动。

- 2026-11-18 09:13 UTC：MLA 已登记 `tuple-abi-input-typing` 限域维护请求 `31a684d9-bfba-484d-9692-6d5178e2a79f`，在当前调用边界后由 Qwen review/必要patch/verify，仅授权两个 tuple Input 条目。整数 tuple 的值域由任务材料核实，不按样本擅自收窄。公开任务为 MLA_TUPLE_MAINTENANCE.json；规则扩展主张已按现有开放类型语义驳回，示例建议并入 non-tensor-parameter-registration，产物维护仍待完成。

- 2026-11-18 09:18 UTC：FMHA reviewer attempt `8bee8bce-75e6-4f97-af52-dbcdfe16384b` 的 artifact_evidence[15] 写到 root.yaml:205，而已审文件实为204行；原生调用成功且终止已确认。按已验证公开resume路径续原reviewer session `07d04ece-63bb-4b21-99ac-9a1dc44ab0e8`，P1已用次数从3继续，不代写证据。新增xhigh另评估既有有限receipt repair为何未处理这类可定位漏项，不改本批冻结内容。

- MoE 的 `normalizer-diff-not-visible` 已获独立 reviewer 确认。已修当前模板工具：长字段警告显示首差异位置与短上下文，案例重新规范化与原产物逐字节一致；相关检查通过，六份冻结模板均未改变。详情见 NORMALIZER_WARNING.md。

- 2026-11-18 09:50 UTC：Sparse P2同形证据末行109写成110，已核对native成功、终止结清和精确session，公开resume后第2次P2调用在途。两例末行问题的根因已核实为Claude Read额外显示虚拟EOF空行，不是稿件错绑或工具重写坐标；见 EVIDENCE_LINE_ENDS.md。不加+1容忍、不自动替模型改证据，使用有限重交提示真实行数。

- 回执错误分类源码修复已完成并登记 `receipt-evidence-repair`：复用原有限修复与调用额度，30项定向验证通过；身份/权限/固定来源等硬边界不变，本批冻结执行器不热更新。

- 六算子同类类型核对完成：128个Input、10个Output中31个python_none Input，仅MLA已登记条目存在tuple/None矛盾，其他五项无需新增维护。见 TUPLE_CROSSCHECK.md。

- 2026-11-18 10:16 UTC：GEMM P2 reviewer 把root.yaml末行112写成113，GDN P2 reviewer把137写成138；与已调查EOF显示差异同形。均验证native完成、终止和会话后公开续原run，已确认在途，无调用额度追加。

- Sparse P2检查要求10条variant/combination有现有完整YAML的直接baseline链接，worker正在正常处理。新xhigh核对草案实际格式及P2/P4阶段边界，尚不预断工具错误；不干预当前稿件。

- Sparse P2目录诊断已核实为合理退修：4条只有未来基线的内容移回普通计划线索，6条涉及现有root的正式格式记录补直接链接后通过检查，仍有3份明确未来YAML未创建；没有把P4完整正文义务强加P2。无需改工具，见 P2_INDEX_CHECK.md。

- 2026-11-18 11:20 UTC：MLA维护verify的root.yaml末行183写成184。冻结执行器已有维护续接逻辑，可保留patch草稿与原verify会话、带入拒因并累计原P1次数；公开whole-run resume已启动第6次P1调用，未重做review/patch。Sparse P2 reviewer的dispatch末行91写成92，也已按原会话恢复。

- 巡检登记另作最小源码改进：ack及绑定scan的claim/commit/清理共4处复用既有锁外取锁重试，避免正常竞争直接浪费本轮提醒；身份/代际/频率/次数与其他外部控制语义不变，11项隔离验证通过。下一次新CLI请求可受益，daemon与frozen Runner均未热改。见 MONITOR_REGISTRATION.md；未在线假造新的提醒或scan。

- 2026-11-18 11:45 UTC：MLA tuple条目维护完成并恢复正常P2 reviewer；Sparse P2审定4套完整代表并进入P3。已读四项新增P2批准后的candidates，无新增框架ID。FMHA P2 reviewer再次出现已知EOF末行206/207问题，已核对结清并续原会话。

- 2026-11-18 12:17 UTC：GEMM/GDN P3 worker回执缺bound implementation entry anchor；同属已登记可修回执义务漏项。核对native成功、终止和精确会话后公开resume，均已实际进入第2次P3 worker调用；不放宽入口证据、不重建运行。

- 2026-11-18 12:47 UTC：MoE P3 worker同类入口证据漏项，已验证退出、终止和精确会话后公开续交，P3原调用次数从1继续；与GEMM/GDN同类，不另建重复工具issue。

- 2026-11-18 13:17 UTC：Sparse P3 worker attempt ebd7d0c4-f38e-4d11-91a8-7a02bf02fea4 的证据末行543超过当前541行，属于既有证据选择修复类别；本次未单独核实两行差异的具体成因，不将其直接归为已验证的单行EOF显示差异。核对native/终止/会话后公开续交，原范围检查不变。

- 2026-11-18 13:54 UTC：新增reviewer确认的非阻塞 `location-storage-level-registration`，登记于extract_workflow。问题是P1最小执行单元登记与DOC5存储名称来源的文字口径，不新增字段、不改当前register产物；与非Tensor参数示例一起待收口评估。

- 2026-11-18 14:49 UTC：Sparse reviewer确认 `inplace-output-input-registration`，提出只写回的可选调用者输出缓冲是否需要Input/Output双登记的说明问题；当前Compute已写完整分配/别名/覆写语义。已交独立xhigh判断是既有规则缺项还是文字歧义，不据确认标签直接改规范或产物。

- `inplace-output-input-registration` 独立分流结论：现行Input/Output规则已经明确，属于稿件登记缺项，不新增规范义务。Sparse维护request5677bbf8-286e-4169-b91e-59aa0f8e6445；同类核对新增MoE request4d304fe6-bfdf-49ab-9b1f-5c861b5c6f8c、FMHA request1170387d-57bb-483e-9ffa-1e960c39ab2c。均仅登记输入角色/别名文字，输出ID及Design保留；其余三项无新增。

- 维护命令遇busy时可能已完成意图登记、仅后续delivery未取得锁；本次按公开inspect与同载荷幂等maintain核对，没有创建重复维护或退款。具体结果见 OUTPUT_MAINTENANCE_REGISTRATION.md。

- 2026-11-18 17:24 UTC：三个输出目的地登记维护均已complete，原运行继续其当前阶段/必要重审；未重新选分支或建立新run。

- 2026-11-18 20:56 UTC：FMHA已确认 `maintenance-version-bump-outside-field-cap`。代码已允许并要求worker随授权内容修改递增Version，维护提示未明确这项隐含写权，曾出现误称Runner会代填的交接；交新xhigh只做必要一句文字澄清，不改字段或代码，不重跑当前已完成维护。

- `maintenance-version-bump-outside-field-cap` 已resolved归档：只补维护提示一句，worker更新Version的职责与既有隐含写权限一致；代码、字段和本批冻结运行未动。

- 2026-11-19 02:33 UTC：Sparse P6显式把Version放进revision_scope被拒收，与既有版本隐含权限说明同根，沿maintenance-version-bump-outside-field-cap补充。原方案无root写入；已由公开resume续原reviewer(P6 2→3)。源码仅P6提示一句，不放宽validator或代改scope。见P6_VERSION_SCOPE.md。

- 2026-11-19 04:10 UTC：Sparse合法P6 approve被局部inventory误拒，已复现并修当前工具；这是实际框架bug，不是模型scope错误。旧frozen没有能有效继续的公开路径，未盲目重交或改快照，正在核对只为该算子建立新身份的恢复方案。

- 同时独立核对root寄存器陈述：208来自64+16+128个主数组槽的静态账目，并非全无来源，但不能推出实际总寄存器数或without spilling；需要保留设计存储成本、收窄未经证实的实现保证，详见REGISTER_CLAIM_TRIAGE.md。尚未修改当前稿件或伪造新的批准。

- 2026-11-19 04:23 UTC：已按SPARSE_SUCCESSOR_PLAN通过公开extract为Sparse单独建新run `a362b85a-dce6-4ad4-b38e-3a186dd87c1c`，从P1真实重审；31源码和五稿字节保留，不能继承旧批准。旧run `8f0844f9-f279-4781-a19d-0216b7fc4ef5`仍如实invalid_receipt，未热改/清scope/伪造批准。新run将正常发布后再public export到标准目标，谱系见SPARSE_SUCCESSOR_LINEAGE.json。其余五项未重建。
