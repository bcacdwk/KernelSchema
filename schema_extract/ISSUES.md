# Schema Extract 问题记录

记录执行器设计、实现和提取过程中需要维护的问题，包括已自动解决的争议。一个问题一个稳定 ID；重复出现更新原条目，相关任务通过 ID 引用。状态使用 `open`、`in_progress`、`blocked`、`resolved`、`ignored`；ignored 表示当前范围外，不表示证据补齐。

**当前用户政策覆盖以下历史设计记录：** 硬件区只保留 starter 占位，撰写／搜索／检查／审阅／收尾全部跳过；硬件问题 ignored。不存在 agent 时长、token、调用、修订、续接或无进展额度。非硬件的已知实现缺陷与外部证明缺失可作为 note 保留，open 不自动等于禁止提取；只有基础任务定义不明或必要源码缺失才 needs_input。本轮只做静态审查，未运行新政策测试、模型、提取、真实 schema 检查或 Supervisor 调度。

条目包含：状态与日期、任务／阶段／节点／字段范围、现象和依据、影响、已做处理、下一步与关闭条件。运行中的条目还需引用 run、检查点、来源和产物版本。正文保留足以复查的依据，不只链接可能被清理的临时日志。并发 worker 不直接编辑本文件，由执行器串行汇总；尚未实现时由维护 Agent 更新。设计层面关闭的问题不代表相关实现已验证，运行资格仍以任务记录和实际核验为准。

条目包括设计调研、执行器实现和真实无 kernel CLI 探针；真实算子提取结果另按任务记录。

<a id="se-001"></a>

## SE-001 本地 Codex 调用端到端兼容性验证

- **状态／日期：** `resolved`（本机 CLI 探针）；2026-11-01。
- **范围：** `codex_cli` 调用层、会话续接、中断恢复。
- **现象与依据：** 前期只读调研在本机 Codex CLI 0.163.0 上确认了 `exec`、指定 ID 的 `resume`、JSONL、结构化回执及 `--no-daemon` 等入口，并核对了参数解析和模型目录；未启动推理会话，尚未验证账户实际调用、事件字段、参数生效和取消行为。
- **影响：** 不能把接口声明视为调用、续接和停止均可用的保证。
- **已做处理：** [执行器说明](README.md#流程和记录)将 CLI 适配与执行器恢复分开，恢复依据为文件检查点；所有调用保存明确的会话 ID。
- **关闭依据：** 已使用 `codex-cli 0.163.0`、`gpt-6-astra / xhigh` 完成真实小探针。`.schema_extract/probes/20261101-cli-01/` 保存 new（`01a1970f-0b06-7a80-8e96-1342a2310e20`）、同 ID resume（文件值 8→9）、cancel（`01a1970f-ce0e-7570-9725-9ce19072b210`，CLI 和两层子进程均已 SIGTERM 且确认无存活）、新目录新会话 checkpoint 8→18、独立只读复核。每次记录实际 argv、版本、事件、退出状态、已报告用量与结果。
- **最终参数补验：** 初版误带 `--ignore-rules`，经审查删除；最终适配保留用户／项目执行规则。在 `.schema_extract/probes/20261101-cli-final/` 再用新会话 `01a19711-d553-7fa0-9637-4dd22927194b` 恢复为 18，只读复核 `01a19712-6925-7af3-bb0e-d8f924a3110e` 通过。worker 对自建可写区外 canary、reviewer 对目录内 canary 的正常写入均被系统拒绝，未提权或重试。CLI JSONL 未呈现这些失败的嵌套调用，因此从明确的两份 rollout 提取 4 条相关调用／拒绝事件到 `permission-evidence.json`；没有复制认证信息。
- **剩余边界：** 当前只验证本机 macOS 及此 CLI 版本；模型目录刷新曾超时，但各推理会话成功完成。长任务与真实提取成本继续由 SE-005 跟踪；help 或探针通过不自动授予真实产物交付资格。

<a id="se-002"></a>

## SE-002 运行区与正式库的硬件检查策略差异（历史）

- **状态／日期：** `ignored`（当前提取不处理硬件）；2026-11-01。
- **当前处理：** extract 固定使用显式 `--skip-hardware`，整个硬件区与依赖硬件的关联检查都排除；普通检查默认行为不变。下面保留旧政策的发现与验证，不作为当前阻塞条件，也不声称硬件已验证。新参数只静态审查，未运行。
- **范围：** 检查策略记录、阶段交接、产物交付资格。
- **现象与依据：** [硬件占位豁免](../schema_template/tools/README.md#硬件占位豁免)只覆盖未归档 `schema_wiki`；仓库外运行区和现行示例仍做完整检查。同一份缺硬件资料的草稿可能因所在位置得到不同检查结果。
- **影响：** 如果只记录退出码，或为取得通过而移动产物，会误把暂缓的硬件检查当成已完成。
- **已做处理：** 设计规定保存实际检查策略、诊断与缺口；运行区不复制豁免，缺项不阻止独立部分继续提取，但阻止完整交付。
- **关闭依据／后续：** 执行器固定现有检查器，在仓库外尝试目录逐轮执行完整检查并记录 `strict-no-hardware-waiver`；发现豁免提示直接拒绝。`test_runtime_never_borrows_hardware_waiver` 用缺硬件材料验证实际检查退出 1，检查点可保存但无交付资格；发布在正式目标之外再次检查，因此不能靠搬入 schema_wiki 获得通过。具体算子资料缺口仍另按任务建立条目。

<a id="se-003"></a>

## SE-003 持久状态与异常恢复机制需要收敛

- **状态／日期：** `resolved`（设计与小规模实现验证）；2026-11-01。
- **范围：** 任务状态、检查点、并发写入、恢复和发布。
- **问题与依据：** MAX subagent 的运行架构评审指出，原方案同时维护 SQLite 状态和文件检查点，并要求核对原地半成品；对单机单调度者而言，主要风险是多份状态与恢复分支，而非数据库本身。此项为架构判断，尚无运行故障数据。
- **处理：** 父 Agent 综合后采用每任务一份权威记录、不可变检查点、单 runner、异常后新目录新尝试；首版只发布到新目录，不做旧进程接管、双存储状态修复或已有目录原位更新。
- **关闭依据／后续：** `task.json` 是唯一状态权威；完整只读检查点先落盘，再原子替换该记录。故障注入验证在状态提交前崩溃时，孤立快照不会被接受；新尝试只复制权威记录指向的检查点，半写入文件不被恢复。单 runner 锁、真实进程树终止、小型发布幂等与冲突保留均已测试；进程所有权不明时拒绝恢复，不接管旧调用。

<a id="se-004"></a>

## SE-004 缺资料任务需要稳定的本轮结束条件（历史策略）

- **状态／日期：** `resolved`（设计与小规模实现验证）；2026-11-01。
- **范围：** 批次退出、带缺口产物、重复执行和修订预算。
- **问题与依据：** 两份独立评审均认为，区分执行与交付还不够；若相同输入下的草稿反复重入，或批次等待无法推进的任务，会产生无新增信息的循环。这是设计风险，不是已发生故障。
- **处理：** 可支持的提取完成后正常结束为带缺口草稿，释放名额；无可运行工作时结束批次。相同输入复用该结果，资料改变或问题解决后才显式恢复；回退、换会话和同一计划重入不重置预算。
- **关闭依据／后续：** 合成缺口任务结束为 `draft_with_gaps`，重复 run 没有第二次 CLI 调用，export 被拒绝。中断产物不接受；同一会话达到两次续接时如仍有实质进展，换新目录／会话继续，已用调用数与预算不重置。仍需真实两算子试点观察一个任务缺资料时其他任务的实际推进与成本。
- **当前政策变更：** 上述续接、预算、无进展次数限制已全部移除；由合作者判断有用工作是否完成，诚实备注不自动否决收尾。已结束记录仍不因重复 run 自动重做；用户暂停／取消仍须显式恢复。

<a id="se-005"></a>

## SE-005 会话切分与复核成本尚需小规模实测

- **状态／日期：** `open`；2026-11-01。
- **范围：** 节点提取、独立复核、交接和自动修订效率。
- **问题与依据：** 设计评审认为，默认按分区重建上下文、重复 root 与最终复核，以及重复记录正文事实，可能增加成本。首轮两题真实调用仅形成 Contract 草稿，未进入节点和独立复核；尚不能据此断言某种切分更优。
- **已做处理：** 保留 Contract 常规独立复核；单节点合并 root 与最终复核；多节点复用仍适用的已核验结论，但所有节点实质语义仍须独立核验。实际工作收敛为 contract／node／index，node 内部逐段读写，不固定每个 schema 分区启动会话。交接引用正文，程序生成机械元数据；执行器不建立通用语义依赖图。提取、复核、修订统一 `xhigh`，切分与轮数列为试点参数。
- **首轮实测与配置调整：** `pilot_xhigh_20261101`：GDN 414.694 秒 / 1,577,482 reported token，FP8 405.694 秒 / 1,099,329；各 1 次调用，Contract 草稿检查点均保存，review / Design 均为 0。reported token 为 input + output，包含缓存输入。依据父端授权，第二轮每任务显式设 8,000,000 token / 18 次调用，壁钟／并发不变，复用时导入全部已用量（两题累计 820.388 调用秒），不以新会话清零。原始核验：`.schema_extract/runs/pilot_xhigh_20261101/supervisor-verification.json`。
- **第二轮历史结果：** GDN root v12，Contract/final 均独立审阅为 `accept_with_gaps`，已完成旧政策草稿闭环；FP8 root v10、小 M v8，Contract/两节点均已审阅，停在 index 前，整体 final 因旧 8M 预算未派发。两题没有 export 或 GPU 实验。第二轮核验见 `.schema_extract/runs/pilot_xhigh_20261101_r2/supervisor-verification.json`。8M/18次及原900秒等限额现已移除；本轮不继续 FP8。
- **下一步／关闭条件：** 在固定来源、模型和检查标准下，以单节点和多节点任务观察耗时、调用／用量、实质遗漏、有效修订与返工；仅在一个代表性任务对照会话切分。确认效率收益没有以漏掉副作用、数值路径、复制错误或证据缺口为代价，再确定默认参数。

<a id="se-006"></a>

## SE-006 新派生节点导致已有复核被错误撤销

- **状态／日期：** `resolved`；2026-11-01。
- **范围：** 检查点失效规则、节点复核调度、最终交付资格。
- **现象与依据：** 实现审查发现，将任意节点清单变化都视为全量失效会使正常新增派生 YAML 清空 Contract/root 结论，而旧调度不补回这些复核，最终永远无法满足交付条件。
- **处理与关闭依据：** Contract 变化仍撤销全部结论；普通新增节点保留未变的 Contract/root；已有条件或 root 改变保守撤销相关节点，并显式排入独立复核。`test_two_node_flow_reaches_ready_with_independent_reviews_and_receipt_repair` 从初始任务真实推进到两节点 `ready`，检查 Contract、root、派生、final 全部复核与一次有界回执纠正；不是只测试失效函数。

<a id="se-007"></a>

## SE-007 执行器与检查器的 YAML 解析语义必须一致

- **状态／日期：** `resolved`；2026-11-01。
- **范围：** Contract 比较、版本递增与语义摘要。
- **现象与依据：** 实现审查发现 `yaml.safe_load` 会把 on/off、日期等按默认 YAML 规则转型，小数也可能丢失精度，与现有 `yaml_source.load` 不同，可能漏判或误判语义变化。
- **处理与关闭依据：** 直接加载固定检查器的 `yaml_source.load`，不另定义 YAML 语义；摘要保留类型和 Decimal 原始精度。回归验证 off 保持字符串、长小数精确区别、重复键拒绝，以及 Contract 更正撤销全部节点复核。

<a id="se-008"></a>

## SE-008 局部 Contract 缺资料被作者判断提前终止整个提取

- **状态／日期：** `resolved`（旧政策下第二轮已进入独立审阅与 Design）；2026-11-01。
- **范围：** Contract 草稿、独立复核裁决、Design 继续和预算保留。
- **现象与依据：** 首轮 GDN / FP8 的 worker 回执均说已保存当前可支持的 Contract，却因硬件／文档、比较器精度与非有限处理等缺口返回 `blocked`，并宽泛标记 `contract/node/index`。`Runner.work` 在提交检查点后直接结束，独立 Contract review 从未执行；这不符合继续提取不依赖缺项的设计要求。检查点分别为 `gdn_postconv_mtp_bf16_state_fp32-9ecd8a03d384/cp-0001-566e9b8e17` 与 `fp8_scaled_gemm-1efaafba78b1/cp-0001-9e60b8bf88`，来源 HEAD `448ab852500e244b07dff024418c27e6d097f320`。
- **处理：** worker 的 blocked 只请求独立裁决；可解析 Contract 和节点草稿进入独立审阅。Contract reviewer 明确记录支持 Design 草稿的已核实 IO／计算／实现依据及未获支持的保证，或确认真正基础阻塞；不把宽泛 blocks 列表直接当依赖证明。严格检查与交付条件保持原样。`--seed-run` 逐文件核对旧语义输入，保留原产物／回执／问题身份与已用预算，清空旧复核与交付资格，直接独立复核，避免重做昂贵 Contract 提取。
- **验证／关闭条件：** 合成集成测试已验证“Contract worker blocked→独立 accept_with_gaps→有依据的 Design（worker 仍 blocked）→独立 final→draft 终态”和重复 run 无新增调用；真正基础语义阻塞仍由 reviewer 确认。复用测试验证来源／规范被改时拒绝、旧任务不变、问题更新原 marker。第二轮只用原两题检验真实裁决和 Design 推进；不以强行通过严格检查作为目标。
- **后续事实：** 第二轮两个算子均完成实质节点提取和独立审阅，GDN 完成 final；FP8 的剩余 index/final 是旧预算停止，不再是作者 blocked 跳过审阅。当前 reviewer 进一步改为合作补充者，硬件永不纳入，非硬件备注不自动否决进度；新政策尚未运行验证。

<a id="se-009"></a>

## SE-009 固定输入遗漏算子库的公共任务背景与证据说明

- **状态／日期：** `ignored`（硬件身份提取已不在当前范围；保留历史来源记录）；2026-11-01。
- **范围：** 固定输入、目标设备背景与单算子读取范围。
- **现象与依据：** 首轮只冻结逐题和批次说明，未包含 `kernel_wiki/README.md`。其首页明确库面向 L20D（B300），同时声明具体环境以各题证据为准；缺失这层背景使 worker 只能看到逐题 SM103 标识，报告了部分本可从已有公共来源取得的身份缺口。它仍不能证明具体配置、执行能力、容量或比较器行为。
- **处理与关闭依据：** 固定总 README、EVIDENCE／GUIDE、CATALOG／FACETS、IMPORT 与当前分类 README，并在新会话明确读取任务背景／证据边界。公共索引只定位当前算子或相关分类，不加载全库上下文；不从 SM 编号推断设备能力。旧来源摘要必须保持，新增背景经独立复核裁决；首轮六条真实问题不由实现者直接关闭或改写。

<!-- extraction:gdn_postconv_mtp_bf16_state_fp32-9ecd8a03d384:hardware_evidence_missing -->

## SE-gdn_postconv_mtp_bf16_state_fp32-9ecd8a03d384-hardware_evidence_missing

- **状态：** `ignored`；类型 `evidence_gap`。
- **当前政策：** 硬件区不在提取范围，不再要求补齐或据此阻止 ready/export；未声称硬件资料已解决，旧运行记录保持原样。
- **任务／阶段：** `gdn_postconv_mtp_bf16_state_fp32` / root.yaml:contract.hardware_specification and dependent Design execution/resource bindings；run `pilot_xhigh_20261101_r2`。
- **现象与依据：** The library target statement and SM103 label do not establish the concrete execution configuration or version-matched hardware/compiler guarantees. Required device, execution and feature fields remain absent. Design appropriately records source organization and payloads without claiming verified capacities or synchronization guarantees. Preserved provenance: gdn_postconv_mtp_bf16_state_fp32-9ecd8a03d384, first checkpoint cp-0001-566e9b8e17, inherited through cp-0001-seed-68f26e20.
- **证据：** kernel_wiki/README.md:3; kernel_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_postconv_mtp_bf16_state_fp32/problem/metadata.json:37-43; kernel_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_postconv_mtp_bf16_state_fp32/performance/ACCEPTANCE.json:12-15; root.yaml:210-225; root.yaml:454-462; dispatch.md:42-47
- **基线／检查点：** `79b23ffb37c27955834f313e5f643ba1e9be22a4ca7981b64dfed54c770dc9d9` / `cp-0002-76a3966c70`。
- **问题来源：** `gdn_postconv_mtp_bf16_state_fp32-9ecd8a03d384` / `cp-0001-566e9b8e17`；保留原问题身份。
- **影响与重启条件：** Completion of required hardware Contract fields and strict validation, Verified capacity, residency and primitive execution/visibility guarantees, Final delivery as a complete schema；Supply fixed execution-environment records and version-matched hardware, compiler, programming, ISA, microbenchmark, implementation-index and profiling documentation.

<!-- /extraction -->

<!-- extraction:gdn_postconv_mtp_bf16_state_fp32-9ecd8a03d384:acceptance_semantics_missing -->

## SE-gdn_postconv_mtp_bf16_state_fp32-9ecd8a03d384-acceptance_semantics_missing

- **状态：** `open`；类型 `evidence_gap`。
- **任务／阶段：** `gdn_postconv_mtp_bf16_state_fp32` / root.yaml:contract.algorithm_requirement.golden_reference.tolerance；run `pilot_xhigh_20261101_r2`。
- **现象与依据：** The candidate-relative thresholds are faithfully preserved. The frozen checker is unavailable, so comparison precision and nonfinite handling remain unresolved. Preserved provenance: gdn_postconv_mtp_bf16_state_fp32-9ecd8a03d384, first checkpoint cp-0001-566e9b8e17, inherited through cp-0001-seed-68f26e20.
- **证据：** kernel_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_postconv_mtp_bf16_state_fp32/performance/ACCEPTANCE.json:33-68; root.yaml:184-209; dispatch.md:43-47
- **基线／检查点：** `79b23ffb37c27955834f313e5f643ba1e9be22a4ca7981b64dfed54c770dc9d9` / `cp-0002-76a3966c70`。
- **问题来源：** `gdn_postconv_mtp_bf16_state_fp32-9ecd8a03d384` / `cp-0001-566e9b8e17`；保留原问题身份。
- **影响与重启条件：** Complete numerical acceptance criterion, Acceptance conclusions depending on unspecified checker behavior；Supply the frozen checker or authoritative fixed documentation defining comparison precision and NaN/infinity handling.

<!-- /extraction -->

<!-- extraction:gdn_postconv_mtp_bf16_state_fp32-9ecd8a03d384:candidate_padding_mismatch -->

## SE-gdn_postconv_mtp_bf16_state_fp32-9ecd8a03d384-candidate_padding_mismatch

- **状态：** `open`；类型 `implementation_mismatch`。
- **任务／阶段：** `gdn_postconv_mtp_bf16_state_fp32` / Root dispatch coverage of Contract state-slot padding；run `pilot_xhigh_20261101_r2`。
- **现象与依据：** The declared padding domain includes every integer from 0 through 65536, while the candidate rejects FP32 slot strides not divisible by four elements and requires a 16-byte-aligned state address. The draft preserves the uncovered domain without narrowing Contract or inventing another path. Preserved provenance: gdn_postconv_mtp_bf16_state_fp32-9ecd8a03d384, first checkpoint cp-0001-566e9b8e17, inherited through cp-0001-seed-68f26e20.
- **证据：** kernel_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_postconv_mtp_bf16_state_fp32/problem/shape_range.json:55-59; kernel_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_postconv_mtp_bf16_state_fp32/candidate/launch.cu:190-198; root.yaml:7-12; root.yaml:101-108; dispatch.md:19
- **基线／检查点：** `79b23ffb37c27955834f313e5f643ba1e9be22a4ca7981b64dfed54c770dc9d9` / `cp-0002-76a3966c70`。
- **问题来源：** `gdn_postconv_mtp_bf16_state_fp32-9ecd8a03d384` / `cp-0001-566e9b8e17`；保留原问题身份。
- **影响与重启条件：** Candidate conformance for legal non-four-element slot padding, Complete dispatch coverage；Provide a corrected fixed candidate supporting the declared padding domain or an explicitly authorized requirement revision.

<!-- /extraction -->

<!-- extraction:fp8_scaled_gemm-1efaafba78b1:hardware_identity_and_bindings -->

## SE-fp8_scaled_gemm-1efaafba78b1-hardware_identity_and_bindings

- **状态：** `ignored`；类型 `evidence_gap`。
- **当前政策：** 硬件区不在提取范围，不再要求补齐或据此阻止 ready/export；未声称硬件资料已解决，旧运行记录保持原样。
- **任务／阶段：** `fp8_scaled_gemm` / root.yaml:120-139; small_m_transposed.yaml:44-51,139-180,181-188,461-472,527-601; dispatch.md:18,33-35；run `pilot_xhigh_20261101_r2`。
- **现象与依据：** Independently confirmed open. Source operations support factual extraction, but the supplied material does not establish complete device configuration, authoritative execution/capability and numerical/synchronization specifications, host compiler identity or matched documentation. The checker reproduces missing execution/feature diagnostics. Preserved provenance: seeded task fp8_scaled_gemm-1efaafba78b1, first checkpoint cp-0001-9e60b8bf88, through cp-0001-seed-78ded785, cp-0002-4fc9939402, cp-0003-0bba1db270 and cp-0004-f30ad38bb3.
- **证据：** kernel_wiki/README.md:3; kernel_wiki/aka-cuda-operators-20261030/operators/gemm/dense/fp8_scaled_gemm/performance/ACCEPTANCE.json:97-115; kernel_wiki/aka-cuda-operators-20261030/operators/gemm/dense/fp8_scaled_gemm/candidate/kernel.cuh:469-516; schema_docs/2.contract.hardware_specification.md:48-69; small_m_transposed.yaml:461-472; dispatch.md:35
- **基线／检查点：** `3b1f8e0f41dd25c111a922ba4461c496e9aeebbbd30eae6f4d34733b39029436` / `cp-0004-f30ad38bb3`。
- **问题来源：** `fp8_scaled_gemm-1efaafba78b1` / `cp-0001-9e60b8bf88`；保留原问题身份。
- **影响与重启条件：** Completion of hardware execution/feature fields and dependent named-relation checks., Resource-capacity and residency certification., Certification of primitive precision, descriptor boundaries, buffer reuse, cancellation ownership, progress and terminal completion.；Supply fixed local device configuration, authoritative execution/capability and numerical/synchronization specifications, host compiler identity/version and version-matched documentation bindings.

<!-- /extraction -->

<!-- extraction:fp8_scaled_gemm-1efaafba78b1:acceptance_checker_semantics -->

## SE-fp8_scaled_gemm-1efaafba78b1-acceptance_checker_semantics

- **状态：** `open`；类型 `evidence_gap`。
- **任务／阶段：** `fp8_scaled_gemm` / root.yaml:111-119; small_m_transposed.yaml:71-76,104-105; dispatch.md:19；run `pilot_xhigh_20261101_r2`。
- **现象与依据：** Independently confirmed open. All recorded numerical tolerances use candidate-relative rtol, zero permitted mismatches and no relative-L2 threshold, but comparison precision and NaN/Inf handling remain unspecified. The node appropriately withholds numerical certification. Preserved provenance: seeded task fp8_scaled_gemm-1efaafba78b1, first checkpoint cp-0001-9e60b8bf88, through cp-0001-seed-78ded785, cp-0002-4fc9939402, cp-0003-0bba1db270 and cp-0004-f30ad38bb3.
- **证据：** kernel_wiki/aka-cuda-operators-20261030/operators/gemm/dense/fp8_scaled_gemm/performance/ACCEPTANCE.json:125-160; kernel_wiki/aka-cuda-operators-20261030/operators/gemm/dense/fp8_scaled_gemm/PRECISION.md:3-11; small_m_transposed.yaml:71-76; dispatch.md:19
- **基线／检查点：** `3b1f8e0f41dd25c111a922ba4461c496e9aeebbbd30eae6f4d34733b39029436` / `cp-0004-f30ad38bb3`。
- **问题来源：** `fp8_scaled_gemm-1efaafba78b1` / `cp-0001-9e60b8bf88`；保留原问题身份。
- **影响与重启条件：** Complete numerical acceptance semantics., Acceptance guarantees for non-finite results and overflow., Numerical certification beyond recorded samples and the supported finite-value rule.；Supply the fixed numerical checker or an authoritative specification of comparison precision, elementwise decisions and NaN/Inf handling.

<!-- /extraction -->

<!-- extraction:fp8_scaled_gemm-1efaafba78b1:declared_alias_and_masking_domain -->

## SE-fp8_scaled_gemm-1efaafba78b1-declared_alias_and_masking_domain

- **状态：** `open`；类型 `evidence_gap`。
- **任务／阶段：** `fp8_scaled_gemm` / Shared Contract's additional declared branches; small_m_transposed.yaml:77-88; dispatch.md:8,20；run `pilot_xhigh_20261101_r2`。
- **现象与依据：** Independently confirmed open. Metadata declares additional branches and optional C aliasing, while the supplied input, reference and selected entrypoint expose no C input. logical_m does not affect reference computation. The node correctly records all physical M rows and caller-buffer identity without resolving or deleting broader declarations. Preserved provenance: seeded task fp8_scaled_gemm-1efaafba78b1, first checkpoint cp-0001-9e60b8bf88, through cp-0001-seed-78ded785, cp-0002-4fc9939402, cp-0003-0bba1db270 and cp-0004-f30ad38bb3.
- **证据：** kernel_wiki/aka-cuda-operators-20261030/operators/gemm/dense/fp8_scaled_gemm/problem/metadata.json:4-30; kernel_wiki/aka-cuda-operators-20261030/operators/gemm/dense/fp8_scaled_gemm/problem/input.py:63-81; kernel_wiki/aka-cuda-operators-20261030/operators/gemm/dense/fp8_scaled_gemm/problem/reference.py:2-16; small_m_transposed.yaml:77-88; dispatch.md:20
- **基线／检查点：** `3b1f8e0f41dd25c111a922ba4461c496e9aeebbbd30eae6f4d34733b39029436` / `cp-0004-f30ad38bb3`。
- **问题来源：** `fp8_scaled_gemm-1efaafba78b1` / `cp-0001-9e60b8bf88`；保留原问题身份。
- **影响与重启条件：** Complete semantics of additional layout, accumulation, masking and optional-C alias branches., Implementation coverage and acceptance across those additional branches.；Supply fixed ABI facts defining the additional branches, or authoritative clarification of which metadata declarations apply.

<!-- /extraction -->

<!-- extraction:gdn_postconv_mtp_bf16_state_fp32-79b23ffb37c2:numeric_bounds_missing -->

## SE-gdn_postconv_mtp_bf16_state_fp32-79b23ffb37c2-numeric_bounds_missing

- **状态：** `open`；类型 `evidence_gap`。
- **任务／阶段：** `gdn_postconv_mtp_bf16_state_fp32` / root.yaml:design.math_reshape arithmetic and error ledger；run `pilot_xhigh_20261101_r2`。
- **现象与依据：** BF16 raw-readout staging, reordered query scaling, grouped reductions, fast functions and explicit paired FTZ operations are faithfully extracted. Version-matched numerical behavior and propagated error bounds over the complete Contract domain remain unestablished. Sampled acceptance cannot close this gap. Preserved first checkpoint: cp-0002-76a3966c70.
- **证据：** kernel_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_postconv_mtp_bf16_state_fp32/candidate/solution.py:26-28; kernel_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_postconv_mtp_bf16_state_fp32/candidate/kernel.cuh:130-144; kernel_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_postconv_mtp_bf16_state_fp32/candidate/kernel.cuh:454-505; root.yaml:278-374; dispatch.md:45
- **基线／检查点：** `79b23ffb37c27955834f313e5f643ba1e9be22a4ca7981b64dfed54c770dc9d9` / `cp-0002-76a3966c70`。
- **影响与重启条件：** Quantified numerical guarantees across the complete Contract domain；Supply version-matched numerical documentation and fixed analysis establishing applicable function, FTZ and propagated-error bounds.

<!-- /extraction -->

<!-- extraction:fp8_scaled_gemm-3b1f8e0f41dd:declared_target_device_transcription -->

## SE-fp8_scaled_gemm-3b1f8e0f41dd-declared_target_device_transcription

- **状态：** `ignored`；类型 `transcription`。
- **当前政策：** 旧政策下的设备身份转录曾修正；当前不再填写或审阅硬件区，只保留 starter 占位。下面保留历史证据，不据此宣称硬件已经验证。
- **任务／阶段：** `fp8_scaled_gemm` / root.yaml:contract.hardware_specification.device; dispatch.md:18；run `pilot_xhigh_20261101_r2`。
- **现象与依据：** The current fixed library explicitly identifies L20D (B300), but root omits device and dispatch.md says the inputs identify only SM103 with no concrete model. Record L20D (B300) specifically as the library-declared target, update that gap description, and increment root's version for the YAML change. Do not promote the library statement into proof of per-run configuration, capacities or capabilities. Keep the remaining hardware evidence gap open.
- **证据：** kernel_wiki/README.md:3; schema_docs/2.contract.hardware_specification.md:5-11; schema_docs/2.contract.hardware_specification.md:48-59; root.yaml:120-135; dispatch.md:18
- **基线／检查点：** `3b1f8e0f41dd25c111a922ba4461c496e9aeebbbd30eae6f4d34733b39029436` / `cp-0001-seed-78ded785`。
- **影响与重启条件：** Faithful transcription of contract.hardware_specification.device, Acceptance of the current root Contract revision；A worker records the fixed library target with its limited provenance and corrects the claim that no target model is identified, without inventing missing hardware facts.

<!-- /extraction -->

<a id="se-010"></a>

## SE-010 Codex resume 用量是同 thread 累计值

- **状态／日期：** `in_progress`（差分观察代码已写，当前未运行验证）；2026-11-01。
- **实证：** CLI 0.163.0 的 GDN thread `01a1974b-256d-7132-9f4f-16839fbb298b` 首调用累计 1,760,088，续接累计 2,053,803；定向 rollout 中续接两次内部请求分别 144,630 / 149,085，增量 293,715。原小探针亦为 64,652→132,889，增量 68,237。旧 runner 逐次累加累计值，GDN 历史总和 7,634,402 多计 1,760,088，实证差分总和 5,874,314。
- **当前处理：** 原始 result/events 和旧 task 状态不改写。新调用按 thread 保存上一累计值，只增加 input/output 差额，不重复加 cached/reasoning。重复观察幂等；缺失、未知基线或下降记为观测异常，不产生负增量，也不影响流程。没有计量校准或预算扩展入口。两组真实值与异常测试代码已更新但未执行。

<a id="se-011"></a>

## SE-011 旧 900 秒超时与预算中断已移除

- **状态／日期：** `in_progress`（逻辑已删除，实际新政策行为待验证）；2026-11-01。
- **实证：** 第二轮 FP8 root 有一次旧 900 秒自动超时，随后从已接受检查点的新目录恢复；该次用量未知。最终 FP8 已报告 8,777,991 token、9 次调用、4045.044 秒，在 index 派发前触发旧 8M 上限，两个节点均已协作审阅。
- **当前处理：** 不再有 agent 时长/token/调用/修订/无进展/续接上限，也不读取历史 plan 的旧限制字段。明确 pause/cancel、真实错误与进程清理保留；CLI 版本查询或明确取消后的短等待不是 agent 运行时限。超过 60 分钟仅供 Supervisor 调查活动和产物，不自动停止。长节点可自行选择有用检查点，无固定交接分钟数；本轮不启动后续提取。

<a id="se-012"></a>

## SE-012 新政策与旧冻结计划的恢复兼容边界

- **状态／日期：** `open`（留待新上下文审计）；2026-11-01。
- **范围：** 旧 accepted 草稿、输入冻结、seed/resume。
- **现象与依据：** 新政策修改了 EXTRACT_SCHEMA、检查器 API／策略、协议和执行器内容摘要。旧计划绑定这些文件，正常 resume 会拒绝静默采用新程序；seed 的旧语义文件／检查器／政策一致性核验同样会拒绝跨此次变更复用。取消限额不等于允许偷换旧前提。
- **处理与边界：** 旧 task/CP/回执/产物与两轮证据全部保留，未改写或继续。没有复杂迁移或12M扩额入口。后续需明确审计新计划与旧草稿如何衔接；当前README中的通用新任务命令不能被当成旧试点已验证可续跑的承诺。

<a id="se-013"></a>

## SE-013 Supervisor 调度尚未做运行验证

- **状态／日期：** `open`；2026-11-01。
- **已实现：** 独立 `supervisor.py` 的只读 compact 巡检与可选15分钟 macOS LaunchAgent；明确 thread ID 续接、独立 run 锁、绝对路径，固定程序只可请求已选 run 的 queued 工作。不得恢复用户 pause/cancel，不自动 export；长时活动标记不触发 kill。
- **验证边界：** 默认未启用，本轮未执行 enable/disable/tick、模型或后台 runner。仅静态阅读、AST 与差异检查；定时启停、跨次唤醒、后台进程存活和端到端链路留待新上下文获准后实测。
