# 最终可执行计划：Extract 与 Schema 审计

**审计及方案已完成；下列产品修复全部待实施。** 本次仅写 `schema_discussion/extract_schema_review_20261121/`，没有修改 Extract/Template、规范、正式 YAML/dispatch/receipt、旧运行态、kernel 或 issue 注册表，也没有启动提取、恢复、monitor、真实模型/GPU、提交或 push。当前十二项已交付、monitor 已停，维持该状态。

流程按用户指定完成：**2 个新 max → 2 个新 max → 1 个新 high → 1 个新 xhigh**。最终入口以本文件为准；[覆盖表](COVERAGE.json)逐项交代 15 个 X、11 个 S、24 个候选/104 次出现以及 ISSUE_RECAP 全部类别，[删减复核](FINAL_REVIEW.md)说明最终取舍。前轮报告保留为证据，不再形成并列备选方案。

最重要的确认缺陷与证明范围：

- **S-06：规范化会改变数学身份。** 有界 int32 的 `V=v+1; y=V+v`，实现/reference 与 Contract 都为 `2*v+1`；真实 StageFlow/`accept_result` 将改成 `v=v+1; y=v+v` 的稿件存入待审 checkpoint，v=1 从 3 变为 4。仍没有证明生产 33 稿已受损或独立 reviewer 必然漏检。仅删除 `related` 豁免不足，未证实绑定的 token 也不能猜测改名。
- **X-08、X-15：工具报告/检查对象与事实不符。** 控制请求在真实临时账本已持久提交后仍可返回 `committed=false`；合法 P2 `revise/plan=[]` 被当作空候选制造“缺 root”。没有证据表明本批因此重复调用或回滚了已批准选择。
- **S-01/02/03/05/07/09：当前正式内容仍有明确残留。** 定点为九份 YAML、两份 dispatch，其中五个 root 的九个输出漏掉已知验收标签。需要正文、必要规则及新字节清单同包更正；旧 P6 只证明旧字节。
- **X-01～05、X-09～11：协议反馈和观察含义需贯通。** 普通回执补正使用既有 scope 总预算，native 通道不一致保留有限传输策略；缓存状态、实际样本、被超越样本和新监控代际分别表达。

唯一推荐四包及顺序：

| 工作包 | 目标 | 执行顺序 |
| --- | --- | --- |
| W1 修正采用对象与提交事实 | 名称身份、控制登记/投递、P2 真实候选 | 先修；三项各有离线反例闭环。 |
| W2 定点更正 Schema、规范与当前字节清单 | 九 YAML、两 dispatch、必要短规范、有限审阅和 current | W1 后执行；正文、预算、检查、Version 与 current 一次收口。 |
| W3 贯通回执义务、诊断与原角色补正 | 同一义务→准确反馈→原角色有界重交 | W1 后；先分清失败原因，再取消普通补正的额外累计 cap。 |
| W4 修正观察与监控代际 | 缓存/样本、当前告警、显式新代际 | W1 后；与 W2/W3 无依赖。 |

W2/W3/W4 可独立验收，不要求合成一轮新的六阶段提取。**没有必须用户决定的技术或语义分叉。** 推荐默认与现有权限、预算和语义契约一致；本轮的审计/方案范围并未扩展为产品实施。

## 已有修正的最终评价

这些是已经存在、继续有效的行为，不是本计划新完成的工作。只对右列残留追加修复。

| 已有修正，继续保留 | 仍需调整 |
| --- | --- |
| 真实来源/坐标、nonce/会话/角色权限、reviewer 只读；原稿与规范化稿分开、整组事务；首稿与续修 Version、完整目录/from 检查、P6 inventory、无覆盖发布 | 名称规范化的数学身份；回执义务/诊断/补正边界；P2 reviewer 检查对象 |
| scoped recovery 的真实 failure/attempt、generation、request 去重、健康兄弟与用量保留；tools-only 原规则树、真实授权与 audit marker；单写者、claim、最后用户控制门、维护空轮询快路径 | 提交后的控制答复不能再声称整命令未提交 |
| dense GEMM 完整 K 归约；KDA `ki_f32` 两条窄化路径；masked 对齐行实际写回；stream-K 同 lane serial/实际前驱；MoE、Sparse、MLA 已修动作存在性、复用和退休；既有 worker/存储归属、GDN 对齐与 vec==0、默认值/workspace、selector 范围 | masked clamp/等式域/精确性文字，FP8 FMHA 副本，KDA 未读 ABI 指针，五个 root 的来源验收标签 |
| 最新观察保护、等待后重核 claim、北京时间显示、原提醒的归属与确认 | 缓存与返回样本含义、旧样本聚合、新显式代际起点 |

两种 FMHA 已有全掩码 softmax、归约/近似指数和 mixed Sparse 的 FMA/FTZ/量化路径修正也保留；没有据此重证全域数值正确性。P4 已取消的排版硬停、模型自己填写 summary、P5 精确失败 attempt、规范化首差异提示、P6 完整 inventory 等既有修正继续使用，不新建替代机制。

原始 source bug、性能例外和材料缺口继续准确保留。GDN fence→arrival 缺聚合、indexed Sparse 零 tile 屏障、BF16 FMHA warp/split_range/fused 状态条件、KDA scale=0 差异及 mixed Sparse 资料/性能例外均不列为 kernel 待修。Schema 必须表达实际来源，不能虚构屏障、比较器、全域通过或通过收窄任务域掩盖问题。上述“保留”不等于本轮重证所有源实现正确。

## W1：修正采用对象与提交事实

### 名称规范化（S-06）

**位置：** `schema_template/tools/normalization.py` 的 `_Names`、`_formula_scopes`、`_math_tokens/_tokens`，以及现有 Template normalization、Extract artifact transaction/stage flow 测试。

声明身份、原始拼写、目标拼写分开；casefold 相同只表示潜在碰撞。Math 只同步有明确绑定依据的引用；未登记临时量和无法判定的开放记法保留原文。若必须改的登记名会捕获其他名称，或其改写穿过无法确认的绑定，则在整组写入前拒绝并指出冲突位置。沿用局部参数、comprehension、Pipe/formal 作用域与事务摘要检查；确定的字段、引号和布局整理继续工作。

**验收：** 复用第二轮的真实 `normalize_candidate → accept_result` fixture，`V=v+1; y=V+v`（Contract 为 `2*v+1`）及 `:=` 变体必须保留两个对象或写入前拒绝，不能变为 `v=v+1; y=v+v`；v=1 的 3→4 是语义判据。覆盖反向大小写、外层/局部及两个局部撞名、同 provider 不同 consumer 绑定；安全的同对象更名仍成功，第二次无变化，原稿/provider/兄弟不变。旧测试中把未证实的大小写名称当别名的期望随契约修正。

### 控制提交事实（X-08）

**位置：** `schema_extract/tools/monitor.py` 的现有操作函数/`start_background`/`inspect_runs`，`cli.py` 的结果呈现；相关 `Store`/`wakeup.Clock` 多步入口只修其提交后答复。

操作函数保留已经持久提交的 request/launch 及 ID，CLI 呈现请求提交与后续投递两个事实。确认本操作无提交才报 `committed=false`；已提交报 true 和原 ID，后续 pending/unknown；无法确认报 null/unknown。沿用 retry/failure、launch/retry、request_key、hold、clock generation 的既有关联。`inspect_runs` 投影相关 scope 的待消费/最近请求，使返回丢失后可只读对账。等待锁只包一个事务，不重放 Popen、signal、增额或 apply。

同一修改中核对 restart/start、maintain 后启动、extract 登记后启动、hold release 后启动、resume/export 预约后运行、clock 登记后 spawn/signal；cancel 沿用已有 request/delivery 分离。无需重构它们的控制机制。

**验收：** 在真实临时持久账本注入登记前、登记后且 Popen 前、Popen 后写回前三处 busy；核对 request/launch ID、实际 ledger 和副作用次数。返回丢失后可读对账，重复请求不增第二次启动/额度；较新的 pause/cancel/hold 有效，旧 launch 不能 claim 新预约。不能用 ledger 总 revision 变化代替本操作的提交证据。

### P2 候选身份（X-15）

**位置：** `schema_extract/tools/runner.py:accept_result`、`prompts.py:receipt_record`，现有 stage flow/maintenance receipt 测试。

普通及维护 P2 reviewer 检查已读取的 worker `state.plan_draft`；approve 仍先核 plan 相等，revise 可以合法为空。adjudication 的显式 select/patch 保持自己的候选身份。首次尚无正式选择的稿件标待审；只有真实移除才标 removed。

**验收：** worker complete→reviewer revise/[]→下个 worker，保留真实退修意见且不新增假“缺 root”/check_failures；坏目录、approve 偷改计划、reviewer 改 raw 仍拒绝。第二轮两份真实 reviewer 工作目录按真实 worker plan 重查均通过，作为回归依据，不重算历史 check/计数。

## W2：定点更正 Schema、规范与当前字节清单

### 规则与提示的最小改动

| 文件 | 必要变更 |
| --- | --- |
| `schema_template/tools/text_policy.py` | 仅把 `Golden_reference.Tolerance.*` 的现有 `BUDGETS` 改为 300 Unicode 字符/2 逻辑行；沿用 `text_size/check_values`。 |
| `schema_docs/1.contract.algorithm_requirement.md` | 来源验收记录不得为长度丢弃已知字段；在现有命名段各给一个短例：Input buffer/异名 Output result 及 Tolerance 输出对应；真实 scalar Input 与异名 extent 的等式。 |
| `schema_docs/4.design.work_schedule.md`、`5.design.tile_pipeline.md` | concurrent 使用分别映射的逻辑 worker，不隐含列表次序/同时驻留；物理单元先后接管须写明对应、非同时活跃范围与旧存储/在途访问退休，复用地址不自动合并存储身份。保留现有四个 enum 和其余 order 规则。 |
| `schema_docs/5.design.tile_pipeline.md` | Storage 容量按已登记区域同时存活的独立副本计，相同值不能去重，仍区别整 kernel 寄存器分配；if 段补明 and/or 从左到右短路，未求值分支本次无需有定义，已求值表达式仍须有效。 |
| `schema_extract/prompt/phases/02_dispatch_draft.md`、`04_dispatch_plan.md` | 未来基线线索放现有“范围”，正式变体条目只直链实际完整基线；P6 沿原规则落实最终目录。 |

P3 已有逐段方法、P5 已有复用和首稿/续修说明继续使用；定点审阅按实际变更检查完整归约/窄化、访问域/结果域/capacity、存活副本、退化条件下动作和最终退休。

### 确切内容范围

下表路径均相对于 `schema_wiki/aka-cuda-operators-20261030/operators/`；版本是当前基线→建议值，实施时再读实际字节。

| 文件 | Version | 要改的事实 |
| --- | --- | --- |
| `gemm/grouped_masked/fp8_mxfp4_grouped_gemm_masked/root.yaml` | 4→5 | 实际非空 worktile 用 `eff=min(128,masked_m[e]-128*mb)`、`eff_a=16*ceil_div(eff,16)`；Contract 输出等式仅在 `i_l<eff` 成立。`eff<=i_l<eff_a` 保留实际计算/窄化/写回，不引用未定义 y；撤销“全域解码/乘积精确、只改变归约顺序”的无据保证，分别写 FP32 求值与实际 block-scaled 路径。 |
| `fmha/fmha_fp8_per_token_head/root.yaml` | 4→5 | `m_run`、`l_run` 各 512→2048 B，`page_regs` 512→2048 B；用现有 shape 复制轴并同步 total_size，数学行数不变。 |
| `fmha/fmha_fp8_per_token_head/decode_split_kv.yaml` | 2→3 | `m_run`、`l_run` 各 256→1024 B，保留 slice:4 并计同行四 lane 副本。decode `page_regs` 的 1024 B 活跃 payload 保留。 |
| `linear_attention/kda_raw_prefill_state_bf16/prepare_chain.yaml` | 2→3 | Forbid 只约束不启用状态索引间接寻址/FP32 state，CHAIN 按 request 取 state slot；删未读占位 state-index 指针必须 null 的断言。保留 `ki_f32` 分流。 |
| `moe/bf16_fused_moe/root.yaml` | 4→5 | `y_result`（源 `input.y`）恢复 mode 与显式 `max_rel_l2=null`。 |
| `mla/mla_absorbed_fp8/root.yaml` | 4→5 | `attention_output`/`attention_lse`（out/lse）恢复 mode，lse 的 max_rel_l2 从 absent 改回显式 null。 |
| `mla/mla_nvfp4/root.yaml` | 2→3 | out/lse 恢复 mode、显式 max_rel_l2=null 与完整 max_mismatch_fraction 标签。 |
| `sparse_attention/paged_indexed_latent_fp8/root.yaml` | 4→5 | out 恢复 mode、完整 max_mismatch_fraction/max_rel_l2 标签。 |
| `linear_attention/kda_raw_prefill_state_bf16/root.yaml` | 3→4 | attention_output/final_state/checkpoint_states 恢复 mode 与显式 max_rel_l2=null。 |
| `sparse_attention/paged_latent_mxfp8_nvfp4/dispatch.md`；`linear_attention/gdn_conv_mtp_bf16_state_bf16/dispatch.md` | 无 YAML 增版 | 将“本次未运行/已登记 issue”等过程口吻换为当前来源支持范围与限制；执行过程留维护证据。保留具体同步风险、比较器/逐输出/机制 A/B 缺口、NCU 6/10 与 shape 0/1/3/9 性能例外。 |

五个 root 共九条 Tolerance，逐项保持固定 JSON 的数值、全标签、输出别名、`relative_to=candidate`、null/缺失区别及 source-only 限制。来源对象、JSON pointer 与 SHA 复用 [第二轮结果](round2/schema_contract_evidence/result.json)。标签不提供比较器完整定义；null 不推断“免检/通过”。masked 的更正不缩小输入域、不改阈值，也不声称 GPU 已失败。

### 验收与已发布字节

1. 改前固定涉及文件的旧摘要、Version、Source 与 receipt；保全旧字节或引用已验证可还原的 checkpoint。修改 YAML prose 也是数据变更，增该文件 Version；纯格式、未改文件不虚增版。
2. 按来源独立复核这十一处改动及真实 provider/consumer 影响。masked 复用已完成的 4616 输入索引枚举；FMHA 核四 lane/四 warp 副本及 total_size，区别 active payload 与编译器总分配；KDA 核未读占位地址和两条窄化路径；九条 Tolerance 对原 JSON 逐项核对，并验证新预算接受完整记录、超限仍拒绝。concurrent/短路用已有正反例核对文义，无 evaluator 改动。
3. 对所有受影响**完整目录**运行现有只读、跳过硬件的检查与 dispatch 链接检查，记录实际规则/检查器身份、命令和退出码。Template 命令模板为 `python3 -B schema_template/check_yaml.py <受影响目录>/root.yaml --read-only --skip-hardware`；它读取完整目录，`--all` 仅用于显式请求的有限图分析，不另加全域图检查。
   **检查对象须与 current 字节一致：**`--read-only` 校验的是规范化预览。若仍有预览与 raw 的差异，先在候选副本中审阅并处理必要的规范化差异；最终落盘后重读并检查，确认有限语义审阅指向同一内容，再固定实际字节摘要。不得把仅预览通过记成实际交付字节通过；规范化若影响了已审内容，重核该影响后才完成。结构检查与有限语义审阅分开记，不增加新的交付机制或历史重提取。
4. 各改动目录新增/更新 `.extraction/current.json`，复用 `schema_wiki.current_schema_manifest.v1`，列全当前 YAML/dispatch、Version、Source/源 SHA、真实依赖闭包、有限审阅与检查证据及旧 receipt 路径/SHA，明确 `runner_p6_receipt:false`。本讨论目录下将来的 `CURRENT_SCHEMA_INDEX.json` 仅投影这些 current；未改目录继续使用与实际字节匹配的 receipt。旧 publication proof、receipt、DELIVERY_INDEX 和 run 保持历史原义。

此包完成的证据是新字节清单及有限源审阅，不是新的 Runner P6 批准。

## W3：贯通回执义务、诊断与原角色补正

**位置：** `schema_extract/tools/receipts.py`、`reading.py`、`agent.py`、`prompts.py`、`runner.py:_receive/_recover_call`；`prompt/operations/response.md`、`receipt_repair.md` 的对应说明。

1. 在 `receipts.py` 集中现有 handoff 条件/数值，供 gate、schema description、动态 prompt 共用：**P2/P3/P4 且 approve**，非空、1200 字符/20 非空行。类别提示直接投影当前 STAGES/bindings；P3/P5 implementation 对应入口，同文件承担 host 不抵消类别义务，每类仍只需命中其允许 locator 之一。handoff description 只修改自己的 schema 副本。
2. 先贯通已有 schema_errors，再在原校验路径合并可独立安全判定的缺类别、坐标、索引与 handoff 错误。给原 JSON path、实际行数/选段长度/索引上界/字符行数，保留原数组索引；安全前置未过时不读取依赖它的材料；反馈若截断，标明省略数量与未检查部分，不能把未知写成通过。当前 proposal 必须完整重验。
3. 在 `agent.read_result`/`_receive` 明确 native 双通道不一致的原因，沿既有 CLI/传输有限策略；再只对普通回执补正移除 `_recover_call` 的累计 `receipt_repairs > retry_limit` 停止条件，沿有效 scope 的 `call_exhaustion`。累计次数与 usage 不清零，不另给额度。

三件事分别裁定：**坏回执拒绝采纳/批准；独立通过真实身份、固定来源、checkpoint、Version、权限/只读门禁的草稿可未批准保留；原角色是否可重交按实际原因处理。** 普通缺项、坐标/索引、错误 decision 或 manifest 外的选择字符串可让原角色用新 nonce 更正，但不能执行错误决定或读取清单外材料。真实 nonce/身份/权限/来源/字节门禁失败仍停；有效 revise/blocked/reopen 沿语义流程，不当补正。native 通道矛盾仍拒绝当前回执；未确认终止/归属的调用保持 fence。

正常补正保留 worker 后续 review+裁决+签收、reviewer 后续裁决+签收的调用预留和单次裁决/签收约束；默认 32 只是既有总额度，受用户配置和授权 additions 控制。每次派发前仍核 pause/cancel/hold/维护。维护、CLI 故障、裁决/签收和历史停点不因普通补正策略扩大。代价是机械失败可能多消耗已有额度；这不构成新的必要审批，也不声称真实返工率已降低。

**验收：** 复用现有 receipt/proposal/call-limit 测试和 fake backend，覆盖阶段×decision 的 handoff 边界、Unicode/空白行、同址双类别、多个独立错误同时反馈、坏容器与固定来源漂移。普通补正第三次在有可用额度时继续，预留额度/一次裁决/用户控制仍阻止违规派发；错误 decision/非法选择与坏 nonce、native 通道矛盾分别落入正确路径。全过程没有工具代写引文/结论，批准不提前，合法未批准草稿按事务保存。

## W4：修正观察与监控代际

| 位置 | 最小行为 | 验收 |
| --- | --- | --- |
| `schema_extract/tools/store.py:read_summary` | fallback 成功显示本次 state 样本，将 stale/missing/invalid 限定为缓存状态；保留 revision/采样时间、ctime 检测、`deep_verified=false`。只读一次，读中变化报变化/未知。 | 同字节 metadata 变化、真实 revision 变化、缺失/坏缓存、并发替换；无写入。 |
| `schema_extract/tools/monitor.py:record_scan`、`cli.py` 展示 | 只在 `record_scan` 生成当前有效 anomalies；保留旧 rows/superseded IDs，展示前标历史。`Clock._complete` 消费同一结果。 | 新 ready 后到旧失败样本，latest、当前计数和行动提示不倒退，历史仍可查。 |
| `schema_extract/tools/wakeup.py:Clock.start` | disabled→显式新代际重置扫描/到期起点，initial scan 完成后排 due；同 enabled 代际 daemon 恢复保留原 ID/ack/claim/due，旧投递未结清仍受原归属 fence。 | 假时钟分别覆盖显式停启与同代际重启；旧代际不能确认新提醒，不重复发送。 |

`schema_extract/README.md` 只补输出叶目录须不存在的启动前提，并随上述字段解释缓存/样本与提交/投递区别；已有 inspect/status 用法继续使用。

## 共用完成标准与兼容边界

各包修改后只运行相关现有模块及上述有意义的新增反例；通过后不例行扩成全仓库、全量六阶段或真实模型/GPU 审计。正常权限、来源和目录门禁属于回归条件，不能因本包目的缩小而跳过。

观察/诊断新增字段保持可选；控制结果保留已知的 true/false，以 null 明确未知提交，调用方不能据未知重放。新 prompt/docs/Template 规则进入新 run 快照；已有 receipt、request、check 和账目不迁移。若另一个未完成 run 已获 tools-only 续接授权，仍只在它原 rules 树内加入允许的 Extract Python 修复，不能带入本包 Template/规范更改。十二项均已交付、monitor 已停，以上修复不触发它们的 resume、repair-runtime、republish 或监控启动。

## 实施时的必要验证位置

以下是**未来修复后**的验证安排，不是本轮已经通过的结果。按改动运行相关现有模块与必要反例；不为纯文案、可选诊断或实现镜像另造测试。

| 工作包 | 复用位置与必要增量 |
| --- | --- |
| W1 | `schema_template/tools/tests/test_normalization.py`；`schema_extract/tools/tests/test_artifact_transactions.py`、`test_stage_flow.py`、`test_maintenance_receipts.py`；第二轮 S-06 和控制故障注入夹具。验证改义稿不能落地、真实 request/launch 可对账、P2 revise 检查真实 worker；既有权限与 approve 相等检查不退化。 |
| W2 | 现有 text budget/dispatch/比较输出绑定测试；九 YAML 与两 dispatch 的有限源审阅及所有受影响完整目录检查。复用第二轮 4616 个 masked 输入算术结果、FMHA 副本算术与九条来源 JSON；只在公式或来源变化时重算。新预算的完整记录与超限反例、真实 provider/consumer 和最终字节对齐是完成条件。 |
| W3 | `test_proposals.py`、`test_receipt_repair_policy.py`、`test_call_limits.py`、`test_backends_runtime.py` 及现有证据/坐标测试。覆盖阶段×decision、批量独立错误、普通第三次补正、预留/单次决定、用户控制、nonce/权限/来源、native 通道异常；保稿与批准分别断言。 |
| W4 | `test_monitor_evidence.py`、`test_wakeup_retry.py` 及现有状态观察测试，使用临时 Store/假时钟。验证 fallback 零写入、迟到失败不倒退当前告警、显式新代际与同代际恢复有别，不启动真实 monitor。 |

最终实施交付应能逐包回答：改了哪些实际文件/行为、必要反例和回归的命令/退出码、剩余限制；W2 另给当前字节清单与有限审阅范围。未经实施不得把这些条目改成已修复。

## 已经执行的离线检查与证据边界

下表区分原轮次的实际工作和本轮的最终整理；不把既有通过记录相加成一次独立全套测试。

| 轮次 | 实际已执行 | 证据入口 |
| --- | --- | --- |
| 第一轮 2 max | Extract 23 项既有定向测试及离线反例；Schema 63 项既有测试、内存规范化反例；读取原回执、成品和来源。 | [Extract 审计](round1/EXTRACT_AUDIT.md)、[Schema 审计](round1/SCHEMA_AUDIT.md) |
| 第二轮 2 max | 真实临时规范化/accept_result、控制登记前后故障注入、回执分类、两份 P2 原工作目录只读重查，18 项既有测试；独立复核 372 份回执的 24 候选/104 次出现及 45 成品/12 receipt；4616 个索引输入、5 个全 K 组号样本、23 例 handoff 矩阵、4 个短路案例，8 项既有测试。 | [正确性复审](round2/CORRECTNESS_REVIEW.md)、[Schema/契约复审](round2/SCHEMA_AND_CONTRACT_REVIEW.md)、[强反例结果](round2/correctness_evidence/s06_result.json)、[Schema 结果](round2/schema_contract_evidence/result.json) |
| 第三轮 high | 26 项发现、24 候选和 16 组删减核对，319 个保护文件摘要与链接检查；未重跑产品测试。 | [精简计划](round3/SIMPLIFIED_PLAN.md)、[删减去向](round3/TRIM_DECISIONS.json) |
| 第四轮 xhigh | 回查上述报告、关键离线结果和现有 P3/P5/checker 文义；最终去向/JSON/文档链接检查，对 BASELINE 的 319 个保护文件作最后 SHA 检查，并核前轮报告和旁支讨论未变。**未重跑产品测试或新复现实验。** | [最终复核](FINAL_REVIEW.md)、[最终验证](FINAL_VALIDATION.json)、[基线](BASELINE.json) |

现有测试通过与已发布身份不能抵消确认缺陷，也不能冒充计划已实现。仍未证实且不阻塞这四包的是：33 稿是否受过此类名称腐化、独立 reviewer 命中率、普通补正的真实成本收益、ctime 改变原因、编译器最终寄存器分配及全域 GPU 正确性/性能。不据此增加全量审计、迁移或 kernel 待办。

只有未来用户另要求更低自动补正成本上限或新的 Runner P6 认证时，才处理该明确新增要求；当前推荐继续使用一个既有 scope 总预算，并以有限审阅/current 清单证明本批新字节。旧 receipt、历史 check/计数、原 rules 和旧运行态都保持历史原义。
