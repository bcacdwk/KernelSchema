# main 整合调研：最终修订计划、Schema 语义与 workflow v1

本报告是首轮第二位调研者的只读结论，尚未实施任何产品修订。唯一推荐：**先固定十二项原交付为可还原的 Git 历史基线，再合入 workflow 原分支，随后用两轮、每轮两位执行者完成 W1–W4；第二轮必须完成新字节的独立有限源复核、完整目录检查和 current 清单。** 保留默认 write-only 与标准 AKA 委托路径，不增加模型效果审阅或 GPU 验收门。

最终需求依据是 [审计 README](../extract_schema_review_20261121/README.md)、[FINAL_PLAN](../extract_schema_review_20261121/FINAL_PLAN.md)、[COVERAGE](../extract_schema_review_20261121/COVERAGE.json) 和 [FINAL_REVIEW](../extract_schema_review_20261121/FINAL_REVIEW.md)。前轮报告仅作证据，已删建议不重新成为待办。本轮用户已授权实施；最终计划也明确没有必须用户另选的技术或语义分叉。

## 1. 本次直接核到的事实

- main 基线为 `484902855fe7be82a74fe7c394503d1d7d222dbb`。按审计 BASELINE 的精确路径比较，允许读取的 **277 个保护文件 SHA 全部相同**。另 42 个含 `archive` 路径的条目仅从基线元数据过滤，未访问其目标。因此旧修正已经存在，而 W1–W4 确实仍未实施。
- 12 个正式目录的 **33 YAML、12 dispatch、12 receipt** 均与 [DELIVERY_INDEX](../extract_twelve_20261119/DELIVERY_INDEX.json) 的 SHA 一致。12 个 `.extraction/current.json` 均尚不存在。
- 本次逐一直接读取精确 12 个 `state.json`：均为 `ready / P6`、顶层 `published=true`、`publication.status=published`，active/unconfirmed/unsettled attempts 均为 0。12 个正式 receipt 逐对象等于各自 `publication.proof`。没有调用 Extract 状态命令、修改账本或启动任何流程。
- supervisor 的 clock 仍为 `enabled=false, generation=10`。其中 `daemon.status=running` 是保留的登记字段，不能据它推断进程仍活跃；原关闭报告记录的 `process_alive=false / stopped` 是当时进程观察。本报告未重新启动或修正该记录。
- 对将修改的 **9 个完整目录（19 YAML）** 调用现有 Template `normalize_directory(..., skip_hardware=True)` 作纯内存只读预览，全部 `raw == preview`。这只是改前事实；W1/W2 后仍须重新比较实际交付字节，不能复用该结果充当改后验收。
- 5 份待补 Tolerance 的固定 `ACCEPTANCE.json` 仍与第二轮证据所列源 SHA 一致。masked clamp、FMHA 副本以及 KDA 非空占位指针也已按精确源码位置抽查，残留与最终计划相符。
- 本轮没有访问/枚举 `AKA_experiment` 或任何 `archive` 目录，没有递归搜索 home，没有运行真实模型、GPU、extract、monitor，也没有 checkout、提交、stash、reset 或清理。仅新增本报告及配套覆盖 JSON。

逐文件旧 SHA、Version、current 存在性、状态与分工映射见 [research_revision_coverage.json](research_revision_coverage.json)。这些是研究时快照，不是实施完成声明。

## 2. 四包必须项、精确文件与验收

| 包 / 发现 | 必须落地 | 主要文件 | 必要验收对象 |
| --- | --- | --- | --- |
| W1 / S-06 | 声明身份、原始拼写、目标拼写分开；casefold 仅发现潜在碰撞。Math 仅改有明确绑定的引用；未登记临时量/开放记法保留；必须改名但可能捕获时整组写入前拒绝 | `schema_template/tools/normalization.py`：`_Names`、`_formula_scopes`、`_math_tokens/_tokens` | 真实 `normalize_candidate → accept_result`：`V=v+1; y=V+v` 与 `:=` 不得从 `2*v+1` 改成 `2*v+2`；v=1 保持 3 或落盘前拒绝。反向大小写、外层/局部、两个局部、provider/consumer、安全改名、幂等、原稿和兄弟保护一起核 |
| W1 / X-08 | 返回本操作已持久提交的 request/launch 事实及原 ID，与投递分开。明确未提交才 false，确认提交 true，无法确认 null/unknown；只读可对账，不能全命令重试副作用 | `schema_extract/tools/monitor.py`、`cli.py`；相关 `store.py` / `wakeup.py` 多步入口 | 临时真实账本在登记前、登记后/Popen 前、Popen 后/写回前注入 busy；校验 ID/账本/副作用次数，重复请求不多启动/增额，旧预约不能 claim 新预约，最新 pause/cancel/hold 优先 |
| W1 / X-15 | 普通及维护 P2 reviewer 检查已读 worker `state.plan_draft`，合法 revise/[] 不构造空候选；approve 仍相等校验；adjudication 显式 select/patch 独立；首稿无正式选择标待审，仅真实移除标 removed | `runner.py:accept_result`、`prompts.py:receipt_record` | worker complete → reviewer revise/[] → 下个 worker，无假缺 root/check_failure；坏目录、approve 偷改 plan、reviewer 改 raw 仍拒绝；旧计数不重算 |
| W2 / S-01/02/03/05/07/09 | 第 3 节九 YAML、两 dispatch、九 current；真实来源/依赖/Version/新字节闭环 | `schema_wiki/...` 精确文件见下表 | 独立有限源审阅＋9 完整目录只读 skip-hardware 检查＋dispatch 链接＋最终 raw/preview 一致；旧 receipt 不变 |
| W2 / S-05/07/10/11 | 仅 Tolerance 提升 300 Unicode 字符/2 逻辑行；命名两短例；来源标签不得为长度丢弃；concurrent 的逻辑 worker/物理接管责任；Storage 独立副本；and/or 左到右短路 | `schema_template/tools/text_policy.py`；`schema_docs/1.contract.algorithm_requirement.md`、`4.design.work_schedule.md`、`5.design.tile_pipeline.md` | 现有预算 helper：完整记录接受，超限拒绝；保留其余预算；既有 concurrent/短路正反例验证文义，evaluator 不改 |
| W2 / S-09 | 未来基线线索放“范围”，正式变体条目仅直链实际完整基线 | `schema_extract/prompt/phases/02_dispatch_draft.md`、`04_dispatch_plan.md` | 保持现有 P6 完整目录约束；不新增发布机制 |
| W3 / X-01/02 | 同一 handoff 义务投影 gate/schema description/prompt：P2/P3/P4 且 approve，非空、1200 字符、20 非空行；类别来自 STAGES/bindings，同文件 host 不抵消 implementation，任一合法 locator 即可 | `receipts.py`、`prompts.py`，必要 `reading.py`；`prompt/operations/response.md`、`receipt_repair.md` | stage×decision、Unicode、空白行、同址双类别、schema 副本不污染；P5 不套 handoff 限制 |
| W3 / X-03/04 | 贯通已有 schema_errors；在原路径聚合独立安全错误，给真实 JSON path、行数、选段长度、索引上界和字符/行数；原数组索引不移动；安全前置失败则依赖项不读；截断说明省略数和未检查项 | `agent.py`、`receipts.py`、`reading.py`、`runner.py:_receive`、`prompts.py` | 多错误同时反馈、坏容器/来源漂移、安全前置、反馈省略标识；新 proposal 必须完整重验，工具不生成引用和语义结论 |
| W3 / X-05 | 先区分 native 双通道不一致，再仅去掉普通回执补正累计 cap；沿有效 scope `call_exhaustion`，保留累计用量/次数和后续调用预留 | `agent.py:read_result`、`runner.py:_receive/_recover_call` | 第三次普通补正有额度可继续；nonce/身份/权限/来源/字节失败仍停；native/CLI 有限策略、未终止 fence、单次裁决/签收、维护边界、用户控制都保留；坏决定不执行、未批准保稿与批准分开 |
| W4 / X-09 | 一次只读 state fallback 返回本次样本；stale/missing/invalid 是缓存状态，保留采样时间/revision/ctime、`deep_verified=false`；读中变化标变化/未知 | `store.py:read_summary` | 同字节 metadata 变化、真 revision 变化、缓存坏/缺、并发替换；零写入 |
| W4 / X-10 | 仅 `record_scan` 聚合当前有效 anomalies；旧 rows/superseded ID 保留，展示前标历史；Clock 消费同一结果 | `monitor.py:record_scan`、`cli.py`、`wakeup.py:Clock._complete` | ready 新样本后到达失败旧样本不倒退 latest/当前计数/行动提示；历史仍可查，不在下游再复制过滤器 |
| W4 / X-11/12 | disabled→显式新代际重置 initial scan/due；同 enabled 代际恢复保留原 ID/ack/claim/due；旧发送归属 fence 保留。README 补输出叶目录不存在和字段解释 | `wakeup.py:Clock.start`、`schema_extract/README.md` | 假时钟显式停启 vs 同代际 daemon 恢复；旧代际不能确认新提醒、不重复发送；不启动实际 monitor |

X-08 的多步入口不能漏：restart/start、maintain 后启动、extract 登记后启动、hold release 后启动、resume/export 预约后运行、clock 登记后 spawn/signal；cancel 沿已有 request/delivery 分离。不应借此重构控制系统。ledger 全局 revision 变更不是本操作提交证明。

W3 的三个判决始终分开：坏回执不批准；独立通过身份/来源/checkpoint/Version/权限门的草稿可未批准保留；原角色能否重交按失败原因。错误 decision、清单外选择字符串可以补正，但当前错误决定不得执行或读取清单外资料。普通补正不另加预算、不清历史次数，也不放宽裁决/签收和维护路径。

## 3. 十二项交付保留与定点更正

以下路径以 `schema_wiki/aka-cuda-operators-20261030/operators/` 为基准。全部 12 receipt、原 Source、来源固定提交、原 rules/executor/checkpoint 和发布账目保留。**Source 不改成新 merge HEAD**；本次没有修改其来源实现。未改 YAML 不增 Version。

| 算子 | 当前 YAML / Version（全部保留，箭头为唯一计划增版） | 本次更正与来源边界 |
| --- | --- | --- |
| `gemm/dense/bf16_nvfp4_gemm` | root 2；pair_stream_k 4；pair_two_pass 2；simt_warp_row 2；split_k_reduce 1；umma_single_cta 2；weight_band_mma 3 | 无正文变更；保留全 K 归约、同 lane serial/真实前驱、SM 数方向/工作区回退 |
| `gemm/grouped_masked/fp8_mxfp4_grouped_gemm_masked` | **root 4→5** | 非空 tile `eff=min(128,masked_m[e]-128*mb)`，`eff_a=16*ceil_div(eff,16)`；Contract y 对应仅 active `i_l<eff`；额外非活跃行仍真实计算/窄化/写回，不能引用未定义 y。撤销全域解码/乘积精确及仅归约序不同的无据保证，分别记 FP32 求值与实际 block-scaled 路径。保留原 Contract 域/阈值，不声称 GPU 失败 |
| `moe/bf16_fused_moe` | **root 4→5** | y_result←input.y 恢复完整 mode、显式 `max_rel_l2=null`；其余原子/队列/物理先后接管描述保留 |
| `moe/fp8_scaled_fused_moe` | root 3；mma_sync_fused 2；split_k_decode 3 | 无正文变更；保留 cp.async 空组和小形状解码修正 |
| `fmha/fmha_bf16` | root 4；mma_sync_decode 2；split_kv_prefill 2；tc_prefill_cpasync 1 | 无正文变更；保留全掩码/数值链及 warp 通知、split_range、fused 跨调用条件风险 |
| `fmha/fmha_fp8_per_token_head` | **root 4→5、decode_split_kv 2→3**；tcgen05_prefill 1 | root m_run/l_run 各512→2048B、page_regs512→2048B；decode m_run/l_run各256→1024B。shape 显式复制轴且 total_size 一致；Math 行数不变；decode page_regs 的1024B活跃payload不再乘四，不据此推编译器分配/驻留 |
| `mla/mla_absorbed_fp8` | **root 4→5**；simt 2；split_kv 2 | attention_output/attention_lse 恢复 mode；lse 的 absent 改显式 null。保留23默认值、workspace约束与TMEM退休 |
| `mla/mla_nvfp4` | **root 2→3** | out/lse 恢复 mode、显式 null 与完整 max_mismatch_fraction 标签 |
| `sparse_attention/paged_indexed_latent_fp8` | **root 4→5**；cluster_m128 2；split8_merge 1 | out 恢复 mode、完整 max_mismatch_fraction/max_rel_l2；保留非 split 零 tile 屏障来源风险 |
| `sparse_attention/paged_latent_mxfp8_nvfp4` | root 4；tcgen05_tmem 2 | **仅 dispatch 更正文风**：当前来源支持范围/缺口，去“本次未运行”等过程口吻；保留原题验证回执/最终题目哈希、比较器/逐输出/机制 A/B 缺口、SM103采样边界、NCU6/10及shape0/1/3/9性能例外 |
| `linear_attention/gdn_conv_mtp_bf16_state_bf16` | root 3；fused_single_launch 2；warp_row_reduction 4 | **仅 dispatch 更正文风**；保留out4B对齐、vec==0、fence→tid0 arrival缺块内聚合与调度来源假设，不能虚构新屏障/硬件保证 |
| `linear_attention/kda_raw_prefill_state_bf16` | **root 3→4、prepare_chain 2→3** | root三输出恢复mode/null；prepare_chain Forbid仅约束不启用状态索引间接寻址/FP32 state、CHAIN按request取slot，删除未读state-index占位指针必须null。保留ki_f32先计算再分别窄化、root重读BF16路径和scale=0 Gap |

九条 Tolerance 的共同记录是 `atol=0.01, rtol=0.05, relative_to=candidate, mode=elementwise_and_l2`；完整 `max_mismatch_fraction` 与 `max_rel_l2` 按下表填写，均保留 source-only/比较器定义缺失/不推默认语义。null 不是 absent，也不意味着免检或通过。

| 输出 | max_mismatch_fraction | max_rel_l2 |
| --- | --- | --- |
| BF16 MoE y_result | 0.0 | null |
| absorbed MLA attention_output | 0.001 | 0.01 |
| absorbed MLA attention_lse | 0.0 | null |
| NVFP4 MLA out、lse | 0.0 | null |
| indexed Sparse out | 0.001 | 0.01 |
| KDA attention_output、final_state、checkpoint_states | 0.0 | null |

来源精确路径、SHA、JSON pointer、别名直接复用并核对 [第二轮 result.json](../extract_schema_review_20261121/round2/schema_contract_evidence/result.json)，不能给另7个未提供同组标签的算子补默认值。

有限复核的最小来源位置：masked `candidate/kernel.cu:1181` 两分支 clamp 后 align16；FMHA `candidate/fmha.cu:582` 的行映射、`rm[MT][2]/rl[MT][2]`、608附近各warp MetaRegs和1192附近发布lane；KDA `runtime.py:99`、`solution.py:208`、`chain_launch.cu:281` 的非空占位地址，`prepare.cuh:803` 和839附近两条窄化。复用已完成的4616 masked索引枚举、FMHA副本算术，不改变公式/来源时不重复耗费。

完整目录与依赖不能退化为只查9文件：7个YAML改动目录＋2个dispatch改动目录，共19 YAML/9 dispatch。所有非root节点隐式共享root Contract；五个Tolerance root共覆盖10个有效节点，其他raw不改也须复核其继承结果。上述9目录显式 `from` 仅见 GDN fused_single_launch 的 Math_reshape→root（当前未修改provider），仍在清单中记录；不能因为没有显式from就漏掉root Contract依赖。由原 Template resolver 记录真实闭包，不自造第二解析器。

每个改动目录的 `current.json` 复用 `schema_wiki.current_schema_manifest.v1`，列完整当前 YAML/dispatch SHA、Version、Source/源SHA、真实依赖、有限复核范围及证据、checker命令/身份/退出码、原receipt路径/SHA，明确 `acceptance.runner_p6_receipt:false`。旧receipt保留原路径。新增的 `schema_discussion/extract_schema_review_20261121/CURRENT_SCHEMA_INDEX.json` 只是9个current的投影；未改的3目录继续用与字节一致的原receipt。原DELIVERY_INDEX和旧审计COVERAGE是历史快照，不改成新的批准或完成记录。

## 4. 已修、保留、明确排除

继续保留：真实引用/坐标、nonce/会话/角色权限、reviewer只读、原稿与规范化稿分离、整组事务；首稿/续修Version、完整目录/from、P6 inventory、no-replace发布；scoped recovery、真实failure/attempt、generation/request去重、健康兄弟/用量、tools-only原规则树和audit marker、单写者/claim/最新用户控制、维护空轮询快路径。P4排版硬停已取消，summary仍由模型写，P5失败attempt精准定位、规范化首差异提示均继续使用。

无单独实现项：X-06/07/13/14，S-04/08。它们分别是保留现有修正、调用方按既有格式读取或明确排除的来源修复；不能漏掉保留回归，也不能以这些编号新造机制。

必须排除：全Math解析器、仅删related豁免的小补丁、通用诊断服务/第二validator、自动补引文/坐标、强推provider条件schema、所有receipt_error无差别解除cap、新控制日志/全命令重放、五项计数摘要、P3/P5重复checklist、新Tolerance DSL/全标签parser/全域文字放宽、新concurrent enum/接管表/evaluator、别名新类型/删除必需Input、新恢复/缓存命令/额外时钟选项、自动换模型、输出目录合并器、全量历史搬迁/改签receipt/current发布器、重开六阶段提取/monitor、kernel修复、GPU/全33稿重审、可选Execution摘要和KDA slot_start文案。保留来源风险不等于把其修复延期成隐藏必须项。

没有必须新增的真实模型/GPU调用。W1/W3/W4以真实临时账本、fake backend、假时钟验收；W2必须有**独立有限源复核**，由第二轮D针对C冻结后的实际字节完成，足以满足计划，不要求Runner原worker/reviewer/P6重新跑。W3规定的是未来运行时原角色补正的行为，可先用CPU仿真完成。只有用户另外明确要求新Runner P6认证或新的更低自动成本上限，才新增那项流程；当前不需要重复审批。

## 5. workflow v1 的共享缝与不应改变的行为

已直接检查 worktree 的 `kernelschema/{api,template_api,extract_bridge,resources,writing,_writing_policy,materials,iterations}.py` 和相关测试。[m01 Git调研](research_git_merge.md)确认workflow只在新增API/docs/examples/tests等148路径变更，未修改本轮核心Extract/Template/schema_docs；与main共同祖先即4849028。无文本冲突不代表共享语义已验收。

| 共享点 | 实际接口与兼容要求 |
| --- | --- |
| Template | `template_api.tools()` 私有命名空间动态加载**同一原Template**；W1会影响bridge/strict normalization。保留 `normalize_directory` 返回plan、`validate_directory`/resolve接口和异常类型；不复制parser、不以全局sys.path覆盖两套tools |
| 默认design/optimize | profile默认`write_only`；`write_only_files`保留原始YAML并给诊断，不调用Contract派生、Template normalize/precheck或独立review。W1/W2不能把默认writing改为强制checker、返工或效果/GPU门。现有测试明确允许长文本、保留坏YAML、reviewer=0/evaluator_jobs=0 |
| 显式strict兼容 | `_writing_policy.normalize`才调用Template，`text_policy`从真实BUDGETS投影并核冻结源SHA。W2预算新值应自动进入**新建**strict Rules，不加第二预算表。旧固定请求按原身份继续或正确拒绝漂移，不能改写其Rules/usage来迁移 |
| Schema文档 | `resources.rule_files/freeze_rules`冻结实际schema_docs与Template源码；write_only默认仅选已配置guidance，并不要求所有规范转成验收门。文档澄清是书写依据，不是新资格证明 |
| Extract | `extract_bridge.extract`独立解释器原样转交原CLI/退出码；W1/W4新增可选观察字段、committed=null语义不应被wrapper解释成可重放的false；不改原P1–P6/恢复生命周期 |
| consume_raw/receipt | receipt可省略；若提供，必须是原`schema_extract.delivery.v2`且files与被导入字节相同。W2后旧receipt对新字节应继续拒绝；current不能塞进receipt参数。修订目录可按`origin="curated", receipt=None`导入，current另作普通可追溯材料保留。无需修改发布器/放松原receipt校验/制造新P6 |
| 两种版本 | YAML `META_DATA.Version`是本次九文件版本；workflow `design_revision`属于新workflow Schema对象。二者不能互相替代，不能用workflow revision覆盖来源Version或旧批准 |
| 标准implement | `api.implement(schema=...)`只导入`aka.workflow.schema.standard`并原样委托；与legacy request/continue互斥。不能要求Extract receipt/current、旧qualification、Template或写作review批准才能进入标准AKA。本轮不读/改AKA仓库 |
| iterate | `rounds`计Schema版本，每版完整标准AKA campaign；optimize keep停止；保留原result/feedback；异常或未结清停，不能自动复放。它不是新P1–P6，也不是W3补正循环，不把Extract retry/call cap套进去 |
| runtime/accounting | Extract agent/receipt 与workflow native runtime是不同协议；W3只改Extract，保留workflow operation身份、fence、真实usage及原回执，不统一格式、不迁移历史账户 |

因此推荐的“先交付历史基线→merge→修订”经语义反证成立：先更正文稿会失去易引用的原字节提交；在feature旧基线上修订再merge更易漏掉本地12项；先merge不会触发模型或旧run迁移，且能让所有改后测试针对最终共享Template。仅merge后声称完成则明显不成立，277个原保护文件仍原样且缺陷仍可定位。

## 6. 两轮四执行者的唯一推荐归属

四位执行者均使用新上下文；每轮两位并行，不再spawn。读/运行另一位测试可以，产品写入按下列唯一owner。主代理只负责全局保全、交接、Git收束和审定，不亲自并行改同文件。

### 合并前主代理步骤

按m01的精确allowlist固定57个正式交付文件（33 YAML/12 dispatch/12 receipt）及12个被替换.gitkeep删除，保留可还原原字节。AGENTS本地修改不stage；历史discussion/原运行态/调用账目原位保留。然后将workflow原37提交合入main（m01已验证没有incoming路径覆盖和文本冲突）。不带入AKA、不跑部署。Git具体命令及push授权范围以m01/主代理掌握的用户指令为准。

### 第一轮：先W1屏障，再W3/W4

| 执行者 | 唯一写入归属 | 同轮内部顺序 | 输出/交接 |
| --- | --- | --- | --- |
| A：采用对象与回执 | `schema_template/tools/normalization.py`；`schema_extract/tools/{runner,prompts,receipts,reading,agent}.py`；两个operations提示；Template `test_normalization.py`，Extract `test_artifact_transactions.py`、`test_stage_flow.py`、`test_maintenance_receipts.py`、`test_proposals.py`、`test_receipt_repair_policy.py`、`test_call_limits.py`、`test_backends_runtime.py`、`test_prompt_materials.py` 的必要增量 | 先W1 S-06/X-15及反例通过；通知B/主代理。两边W1都通过后继续W3，先诊断/native分类，最后普通cap修订 | 精确改动、定向命令/退出码、普通补正/安全拒绝分类、原稿与未批准保留测试、接口变化说明 |
| B：控制事实与观察 | `schema_extract/tools/{monitor,cli,store,wakeup}.py`；`schema_extract/README.md`；`test_monitor_evidence.py`、`test_wakeup_retry.py`及必要控制/只读观察测试 | 先W1 X-08全部入口及三时点故障注入；通知A/主代理。两边W1都通过后继续W4 | request/launch关联/未知提交返回契约、late sample/缓存/代际结果、假时钟/临时Store日志、旧run零写入证明 |

A不得改B四个产品文件，B不得改A的runner/prompts/receipt链；如发现最小跨界修复，由owner实施并回交，不双写。共同W1屏障是顺序条件，不能为了同轮并行把W3/W4先提交为已完成。

### 第二轮：内容更正与真正集成收口

| 执行者 | 唯一写入归属 | 依赖与必须收口 |
| --- | --- | --- |
| C：W2全文、规范及清单 | `text_policy.py`；DOC1/4/5；P2/P4 phase提示；本报告明确九YAML/两dispatch；九`.extraction/current.json`；新增旧审计目录`CURRENT_SCHEMA_INDEX.json`；必要Template预算/比较/dispatch测试；本轮内容更正/检查证据 | 第一轮完成后开始。先最小规则/规范，再候选正文和必要规范化；固定19YAML/9dispatch最终SHA并交D。D有限复核如有问题，C更正并再次冻结。D报告固定后C生成current与索引，不先写“已审”占位 |
| D：workflow集成＋独立有限复核＋最终检查 | 仅确有必要的`kernelschema/`兼容修补与`tests/workflows/{test_template_extract,test_write_only,test_writing,test_iterations,test_standalone,test_api}.py`增量；README/`schema_wiki/README.md`导航；本轮独立源复核/兼容回归/总coverage证据 | 可以先跑workflow CPU相关测试、准备源对照；**必须等待C最终实际字节**后复核十一处事实及真实消费者，核raw==review==current。保留原receipt拒绝改字节测试，curated无receipt可导入测试，write_only一调用无review/GPU、standard委托与keep停回归。D不直接改C正文；发现问题交C闭环。最后核九current完整闭包/源SHA/证据SHA及总索引 |

C与D的交接不以“稍后补审阅”结束：C先提供candidate字节清单和完整目录日志；D输出绑定相同subject SHA的独立有限复核；C据固定D证据构造current；D只读核current与index。语义报告不要反向包含current自身摘要，避免哈希环。任何review后字节修改都必须重核受影响内容与摘要。

主代理最终收齐四包完成证据、确认所有必需项有落实文件与退出码，再审定main提交范围。原审计计划/覆盖/交付记录保持历史事实；本轮新实施coverage承担完成状态。无必须项可以只写“后续”。

## 7. CPU验收、运行环境与收口判据

当前系统`python3`缺PyYAML，不能把其import失败判为产品回归。现成 `/opt/venv/python3.13.12/bin/python` 有PyYAML/jsonschema，可跑unittest。m01确认 `/tmp/schema-b18-model-budget-usage-20261121/venv/bin/python` 另有pytest9.2.0，但默认editable指向旧`/tmp/.../fixed-ks`；使用前必须显式`PYTHONPATH=/home/shaohanyong.shy/KernelSchema`并核`kernelschema.__file__`及Template来源确为最终main。不要仅运行旧console脚本后声称新main通过。

新`pyproject.toml`默认pytest testpaths仅`tests/workflows`，**一次根pytest不覆盖Extract/Template**。推荐分别记录：

1. A/B运行第2节相关现有unittest模块及最少有意义反例；使用`PYTHONDONTWRITEBYTECODE=1`，PATH仅所选Python、`/usr/bin:/bin`，fake backend/假时钟/临时Store，真实claude/codex不可查到。控制Popen/signal均故障注入或fake，不运行实际monitor。
2. C用最终main的原Template对9目录逐一执行`python -B schema_template/check_yaml.py <dir>/root.yaml --read-only --skip-hardware`，校验完整目录/dispatch；不擅加`--all`全图。先在候选副本处理必要规范化差异，最终落盘后重读比较raw/preview和语义subject，记录实际checker/规则SHA、命令、exit0与日志SHA。
3. D显式指定workflow相关测试文件；原有fake-runtime测试可执行本地fixture子进程，不调用真实模型/AKA/GPU。至少涵盖原Template私有loader、strict预算投影与规范化、原Extract桥CLI纯透传/错误码、receipt字节绑定、write_only、标准implement mock委托、有限iterate。已有相关测试过后无新失败，不例行扩大到真实环境资格认证或全仓库GPU。
4. 清单核验：9 YAML恰按表增版；2 dispatch只修定点事实；另外24 YAML原字节保留；33+12原交付仍可从历史基线恢复；12旧receipt/所有原run和用量不改；9current覆盖19YAML/9dispatch，源/依赖/证据与实际字节一致；另3目录14YAML/3dispatch仍与旧receipt匹配。若必要规范化导致额外字节变化，必须逐项列出原因和复核范围，不能默算入原9文件完成证明。

现有测试与只读预览只证明有限机械性质。最终必须诚实区分：结构检查、有限源语义复核、历史P6、write-only完成、真实AKA结果；不声称33稿全域正确、GPU性能通过或真实模型返工率下降。本次没有发现阻断实施的缺数据或需求冲突。
