# Schema Extract 问题记录

记录执行器设计、实现和提取过程中需要维护的问题，包括已自动解决的争议。一个问题一个稳定 ID；重复出现更新原条目，相关任务通过 ID 引用。状态使用 `open`、`in_progress`、`blocked`、`resolved`、`ignored`；ignored 表示当前范围外，不表示证据补齐。

**当前用户政策覆盖以下历史设计记录：** 算子已经真实跑过 GPU；当前算子和 schema 都不需要 GPU 正确性验收。extract 完全忽略 GPU 验收与实测，不列为待办、缺口、交付条件或收尾提醒。硬件区只保留 starter 占位，撰写／搜索／检查／审阅／收尾全部跳过；硬件问题 ignored。不存在 agent 时长、token、调用、修订、续接或无进展额度。已知实现缺陷与尚未确定的源码关系可作为 note 保留，open 不自动等于禁止提取；只有基础任务定义不明或必要源码缺失才 needs_input。执行器试跑与检查的依据记录在下列条目。

条目包含：状态与日期、任务／阶段／节点／字段范围、现象和依据、影响、已做处理、下一步与关闭条件。运行中的条目还需引用 run、检查点、来源和产物版本。正文保留足以复查的依据，不只链接可能被清理的临时日志。并发 worker 不直接编辑本文件，由执行器串行汇总；尚未实现时由维护 Agent 更新。设计层面关闭的问题不代表相关实现已验证，运行资格仍以任务记录和实际核验为准。

条目包括设计调研、执行器实现和真实无 kernel CLI 探针；真实算子提取结果另按任务记录。

**历史运行清理（2026-11-02）：** 按用户要求，确认 5 个旧 FP8 GEMM／GDN task 均 ended、逐调用进程身份无存活、runner 锁空闲且相关 LaunchAgent 已卸载后，删除 `pilot_xhigh_20261101`、`pilot_xhigh_20261101_r2`、`policy_xhigh_20261101_fp8` 的 run/task、专属输入快照与仓库外 work-root，共 16 个目录、735 个文件、25,283,991 字节。随后补清探针目录中 3 份仅属于旧算子试跑的 seed／历史文件清单报告，以及混合汇总中的旧算子部分。源码（含探针脚本）、正式产物和本台账保留；以下旧检查点／日志路径仅为历史身份，不再保证可打开，历史结论以条目内的实证摘要为准。主目录清理清单与保护文件摘要见 `.schema_extract/maintenance/cleanup_20261102.json`。独立 CLI／Supervisor 探针与汇总中的对应证据保留。

<a id="se-001"></a>

## SE-001 本地 Codex 调用端到端兼容性验证

- **状态／日期：** `resolved`（本机 CLI 探针）；2026-11-01。
- **范围：** `codex_cli` 调用层、会话续接、中断恢复。
- **现象与依据：** 前期只读调研在本机 Codex CLI 0.163.0 上确认了 `exec`、指定 ID 的 `resume`、JSONL、结构化回执及 `--no-daemon` 等入口，并核对了参数解析和模型目录；未启动推理会话，尚未验证账户实际调用、事件字段、参数生效和取消行为。
- **影响：** 不能把接口声明视为调用、续接和停止均可用的保证。
- **已做处理：** [当时的执行器说明](https://github.com/bcacdwk/KernelSchema/blob/38d5a389173b099a8694cffd3d892dc25f6fa049/schema_extract/README.md#流程和记录)将 CLI 适配与执行器恢复分开，恢复依据为文件检查点；所有调用保存明确的会话 ID。
- **关闭依据：** 已使用 `codex-cli 0.163.0`、`gpt-6-astra / xhigh` 完成真实小探针。`.schema_extract/probes/20261101-cli-01/` 保存 new（`01a1970f-0b06-7a80-8e96-1342a2310e20`）、同 ID resume（文件值 8→9）、cancel（`01a1970f-ce0e-7570-9725-9ce19072b210`，CLI 和两层子进程均已 SIGTERM 且确认无存活）、新目录新会话 checkpoint 8→18、独立只读复核。每次记录实际 argv、版本、事件、退出状态、已报告用量与结果。
- **最终参数补验：** 初版误带 `--ignore-rules`，经审查删除；最终适配保留用户／项目执行规则。在 `.schema_extract/probes/20261101-cli-final/` 再用新会话 `01a19711-d553-7fa0-9637-4dd22927194b` 恢复为 18，只读复核 `01a19712-6925-7af3-bb0e-d8f924a3110e` 通过。worker 对自建可写区外 canary、reviewer 对目录内 canary 的正常写入均被系统拒绝，未提权或重试。CLI JSONL 未呈现这些失败的嵌套调用，因此从明确的两份 rollout 提取 4 条相关调用／拒绝事件到 `permission-evidence.json`；没有复制认证信息。
- **剩余边界：** 当前只验证本机 macOS 及此 CLI 版本；模型目录刷新曾超时，但各推理会话成功完成。长任务与真实提取成本继续由 SE-005 跟踪；help 或探针通过不自动授予真实产物交付资格。

<a id="se-002"></a>

## SE-002 运行区与正式库的硬件检查策略差异（历史）

- **状态／日期：** `ignored`（当前提取不处理硬件）；2026-11-01。
- **当前处理：** extract 固定使用显式 `--skip-hardware`，整个硬件区与依赖硬件的关联检查都排除；普通检查默认行为不变。下面保留旧政策的发现与验证，不作为当前阻塞条件，也不声称硬件已验证。本轮 145 项检查器测试通过，旧 GDN root 与 FP8 两节点的实际非硬件检查均返回 0。
- **范围：** 检查策略记录、阶段交接、产物交付资格。
- **现象与依据：** [硬件占位豁免](../../../schema_template/tools/README.md#硬件占位豁免)只覆盖未归档 `schema_wiki`；仓库外运行区和现行示例仍做完整检查。同一份缺硬件资料的草稿可能因所在位置得到不同检查结果。
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

- **状态／日期：** `resolved`（具体返工改进已验收，独立成本对照不列待办）；2026-11-02。
- **范围：** 节点提取、独立复核、交接和自动修订效率。
- **问题与依据：** 设计评审认为，默认按分区重建上下文、重复 root 与最终复核，以及重复记录正文事实，可能增加成本。首轮两题真实调用仅形成 Contract 草稿，未进入节点和独立复核；尚不能据此断言某种切分更优。
- **已做处理：** 保留 Contract 常规独立复核；单节点合并 root 与最终复核；多节点复用仍适用的已核验结论，但所有节点实质语义仍须独立核验。实际工作收敛为 contract／node／index，node 内部逐段读写，不固定每个 schema 分区启动会话。交接引用正文，程序生成机械元数据；执行器不建立通用语义依赖图。提取、复核、修订统一 `xhigh`，切分与轮数列为试点参数。
- **首轮实测与配置调整：** `pilot_xhigh_20261101`：GDN 414.694 秒 / 1,577,482 reported token，FP8 405.694 秒 / 1,099,329；各 1 次调用，Contract 草稿检查点均保存，review / Design 均为 0。reported token 为 input + output，包含缓存输入。依据父端授权，第二轮每任务显式设 8,000,000 token / 18 次调用，壁钟／并发不变，复用时导入全部已用量（两题累计 820.388 调用秒），不以新会话清零。原始核验：`.schema_extract/runs/pilot_xhigh_20261101/supervisor-verification.json`。
- **第二轮历史结果：** GDN root v12，Contract/final 均独立审阅为 `accept_with_gaps`，已完成旧政策草稿闭环；FP8 root v10、小 M v8，Contract/两节点均已审阅，停在 index 前，整体 final 因旧 8M 预算未派发。两题没有 export。第二轮核验见 `.schema_extract/runs/pilot_xhigh_20261101_r2/supervisor-verification.json`。8M/18次及原900秒等限额现已移除，旧记录保持原样。
- **新政策实测：** `policy_xhigh_20261101_fp8` / `fp8_scaled_gemm-dcb921aefcb5` 复用旧草稿，经 Contract、root、小 M、index/final 达到 `ready`；最终 `cp-0003-0bd256ca55`。接入政策后的两份 YAML 全程原字节保留，实际 worker 只修正并精简索引（8,858→4,452字节），两项非硬件 note 保留。新增13次调用、累计 CLI 1,804.8秒、增量 reported tokens 5,322,265（含缓存输入）；无新额度或导出。完整分项见该 run 的 `verification.json`。
- **本轮效率问题：** 6次额外续接都用于回执路径修正：5次把版本化来源引用写入 `evidence.path`，1次改成了绝对路径；最终相对路径通过，产物未改。索引旧检查状态的小修还触发了一次节点重新复核。后续新计划优先明确回执路径与来源引用的区别，并减少仅改索引时的重复上下文；本轮记录保留这些实际成本，不据单个任务断言默认切分已最优。
- **2026-11-02 复现与改进：** 新 masked Contract reviewer `a-0002-6218cc08/call-01` 再把 4 个来源路径写成当前 `KernelSchema@2574b7d054a930b6406ef0eeec7e24329e4c96e0:...`，触发仅修格式的 call-02，额外 39.492 秒／73,477 增量 tokens、没有工具调用或产物改动。回执 schema／提示已明确两种路径；程序现在仅把精确匹配当前来源前缀或当前目录内绝对路径转为相对路径，完整校验文件、行范围、原文后留下转换记录，原始回执不变。对这份真实被拒回执的离线重放已通过，依据为该 run 的 `receipt-replay.json`；错误 commit、越界和不匹配引文仍拒绝。其他回执纠错只发具体错误及必要规则，不重复整套指令。新 MLA 将验证实际调用路径，不将离线重放当作已节省的真实用量。
- **当前处理／关闭条件：** 按用户最新要求，不单独再做会话切分或成本对照实验。落实 SE-018／SE-025 的具体读取、绑定和修订流程改进，并随功能验收观察是否减少截断补读和无效重审；统计只用于解释真实改动，不为降低数字删减必要源码核验。
- **新 masked 基线完成：** `trial_20261102_masked_baseline` 从零达到 ready，最终 `cp-0005-949e9eb079`，root v8／single_cta v5／mma_sync v4，Contract、三个节点、final 共五项审阅 complete。28 次调用、12,758,962 差分 reported tokens（含缓存输入）、3,641.907 CLI 秒；其中 18 次回执修正占 2,387,484 tokens／659.988 秒，实际首错分类 17 path＋1 quote。逐调用终止确认与 131 个历史 PID 身份检查均无残留，未导出。父端在冻结旧执行器中再次验证 eligible，报告为该 run 的 `verification.json`。不同算子不能直接给出总成本的因果降幅。
- **新 MLA 改进版完成：** `trial_20261102_mla_improved` / `mla_expanded_fp8-aa60ab8be0d4` 从空目录、新 plan、新 thread 达到 ready，最终 `cp-0002-b297748059`、root v8。4 次调用、6,016,647 差分 reported tokens、1,761.009 CLI 秒，无未知用量、回执拒绝或纠错。作者 Contract→root 沿用 thread `01a19834-fdd3-70f1-a823-e75fd493ca47`，两次 reviewer 均独立；单节点完整 final 已核验，算法 Contract 字节不变，非硬件检查通过。五项源材料 note 保留。所有调用终止确认、44 个记录 PID 无残留、runner 锁释放；未导出／提交／推送。父端在冻结执行器中复验 eligible，依据为该 run 的 `verification.json`；仍有输出截断，见 SE-018。

- **本轮关闭依据：** `refine_20261102_fp8` 从零完成 Contract、root、small_m_transpose、index 及相应独立审阅，8 次调用达到 ready，无回执修正或重派已完成节点。真实读取和 final 范围见 SE-018／SE-025。分页有工具往返，统计留在 run 记录；不以不同任务总 token／耗时推断因果降幅，不再为找“最优切分”另开实验。

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
- **后续事实：** 第二轮两个算子均完成实质节点提取和独立审阅，GDN 完成 final；FP8 的剩余 index/final 是旧预算停止，不再是作者 blocked 跳过审阅。当前 reviewer 进一步改为合作补充者，硬件永不纳入，非硬件备注不自动否决进度；新政策的合成工作流已通过，本轮真实结果见 SE-005。

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

<a id="se-017"></a>

## SE-017 新会话在空产物目录搜索固定规范，阶段阅读入口不明确

- **状态／日期：** `resolved`（本轮新 MLA 路径使用已验证）；2026-11-02。
- **真实依据：** 全新上下文试跑 `trial_20261102_masked_baseline` / `bf16_grouped_gemm_masked-7ffc871299d8`，初次 Contract 调用 `a-0001-13300e6b/call-01`，thread `01a197fc-7406-7622-92f3-06c6f6b30b45`：`events.jsonl` 的 item_1 在空 workdir 执行 `pwd`／`rg --files` 搜索规范，退出 1；item_2 才改到固定 snapshot 并成功。原提示虽列出两个目录，读取指令却只给裸文件名。第二个全新 Agent 的实际准备也指出 README 缺少工具维护期间如何启动冻结执行器的说明。
- **处理：** 提示直接说明产物目录初始为空，给出固定规范／starter 的绝对路径和当前阶段；Contract 只列算法与公共规范，node 列 Design 规范，index 列公共规范，具体交叉检查仍可读其他非硬件章节。验收文件使用当前 catalog 路径，不套用别题文件名。README 补充冻结运行方式。
- **同轮补充：** 独立 Contract reviewer 的 `a-0002-6218cc08/call-01` item_11 跟随算子 README 全库接口清单链接搜索 `MANIFEST.json`，因快照不含全库清单退出 2。提示现明确本次只冻结所选材料，catalog 的当前条目已含接口／路径，ABI 直接核对 entrypoint/input/reference；不为消除一次导航失败而载入全库。
- **工作目录误认：** `single_cta` worker `a-0005-fcb2b6a6/call-01` item_37 在普通产物目录运行 `git status --short`，退出 128；macOS git shim 还触发无用的 xcrun 缓存权限错误。未提权或改文件。初始与作者续接提示均已明确 workspace 无 Git 元数据，直接读文件并运行检查器。
- **跨根路径混用：** `mma_sync` worker `a-0007-f4a059ba/call-01` item_11 在 snapshot cwd 把 `sed single_cta.yaml && sed kernel_wiki/...` 拼接；产物不在该目录，首条失败还跳过源码读取。item_13/14 分开定位后成功。提示补充混合读取两个根时都使用绝对路径。
- **验证／关闭依据：** 新 MLA 四次实际调用首次规范读取即定位 snapshot，使用本题 CERTIFIED_BEST，混合读取时对 workspace 产物使用绝对路径；未出现 Git、缺 MANIFEST 或跨根失败。仍有空目录盘点的正常 rg exit 1，以及一次无匹配 rg 与 wc 用 && 连接而跳过 wc 的低影响空操作，不把这些计成已消除，也不据此扩展规范读取范围。

<a id="se-018"></a>

## SE-018 批量读取的外层截断导致重复读取，CLI 事件不足以诊断

- **状态／日期：** `resolved`（受支持入口已验收；任意外部合批不在保证内）；2026-11-02。
- **真实依据：** 同一 masked Contract thread `01a197fc-7406-7622-92f3-06c6f6b30b45` 的定向 rollout 第 22 行并行读取规范，每条内层命令指定 16k／18k 输出，却没有调整 `functions.exec` 外层预算；第 27 行回包报告 `truncated output (original token count: 14605)`，第 31 行重新读取 common／starter／模板。第 45 行又发生 14,196-token 外层截断。CLI `events.jsonl` 保留了完整内层命令结果，单看它会误把重读归因于模型重复理解。
- **处理：** 提示要求按行段／JSON key 读长文件，批量调用同时考虑外层组合输出预算；截断后仅补缺失片段。观察者只定位本次 thread 的日志，不搜索或加载历史会话库。
- **前一轮 MLA 的反证：** 首次 Contract 作者无外层截断，但独立 Contract reviewer 的定向 rollout 第 25 行因合批 13,926 tokens 截断并重读；作者 node 续接第 129 行又因 11,863 tokens 截断，随后自行提高外层输出预算并定点补读。当时只补充预算语法，未据此关闭问题。下列改动保留规范逐段读写回查，转为有界入口处理。
- **本轮机制改进：** 新增标准库 `reading.py`，整批共享每页 6,000 包装后 UTF-8 字节上限；保留原行／列、换行、内容 SHA、准确未读位置和可重放游标。每次校验整批内容与请求，改变文件、坏游标、越界路径均明确拒绝。普通完整行省略默认列号／完成标记，避免元数据挤占正文。固定调用模板一页一外层调用，不再靠临时猜总输出预算；任意绕过入口的大合批仍不在保证内。
- **真实验收补修：** 新 agent 独立实现，父端复审后压缩重复元数据并补 shell 参数／调用模板验证。`refine_20261102_fp8` / `fp8_scaled_gemm-97d141747cf0` 在内容页之外用普通 `rg -n` 查 12 个关键词，两文件合计命中 600 行，输出 161,243 字节（内层 reported 40,311 tokens），在 2,500 的工具预算被截断。现为同一读取器增加 `find` 字面子串定位，只返回原行号与总数，条件、请求、整批文件摘要均绑定游标。对同一冻结输入的真实查询重放一次返回全部 600 个有序行号，与 `rg` 完全一致，最终 v2 含一次包装 3,833 字节。另将调用模板改为由 JS 保存原命令、游标与重放位置，避免手抄错误。这些补修在 run 冻结后实现，不冒充已被该 run 的模型使用。
- **完整验收：** 5 个实际 thread 的 324 个工具回包中，读取器成功返回 288 页（242 次续页），每页均可解析且不超过 6,000 包装字节；读取器没有截断标记，未出现外层截断。另有 2 次手抄坏游标被拒、3 次绕开入口的直接 `rg` 内层截断。对后者逐项重放：600 行、532 行两组定位各在一页返回，原有序行号均相同；等价于读全文的第三组分 5 页完整重组，原文逐字节相同。
- **位置完整性：** 父端进一步指出，仅校验批次摘要和坐标范围还不足以拒绝“误改后仍合法”的坐标。v2 游标将版本、摘要和坐标一起校验，固定 92 字符；损坏报 `invalid_cursor`、输入变化报 `stale_cursor`、未校验的旧版本要求重建批次。校验用于检测损坏，不是权限认证。24 项读取回归通过；父端用现行模板的真实工具调用验证客户端自动续到同一原行第 74 列，并逐字相同地重放该页。完整 run 使用冻结时的分页版本，当前定位与游标补修由真实输入重放及边界回归验证。依据：`.schema_extract/runs/refine_20261102_fp8/verification.json`。

<a id="se-019"></a>

## SE-019 status 混淆活跃调用与已结束调用统计，并输出全部历史

- **状态／日期：** `resolved`；2026-11-02。
- **真实依据：** masked 首次 Contract 调用已持续工作且写入 root 草稿，原 `status` 仍仅显示累计耗时／token 为 0，没有当前 thread 或调用年龄；每次直接输出整个 task，随回执、用量观察和审阅增长。观察 Agent 只能另查 events/process 文件才能判断正在推进。
- **处理：** `status` 默认输出紧凑只读视图，区分 `calls_started`、已结束调用耗时与 token，以及 `active_call` 的 PID、thread、持续时间、事件字节／更新时间；`tokens_include_active_call=false` 明示计量边界。`--full` 保留原完整 JSON 入口。无新持久状态、推理调用或超时判定。
- **验证／关闭依据：** 已对正在运行的 masked 调用真实查询，显示 thread、271.7 秒和 280,423 字节事件，同时 token 仍如实为 0；47 项执行器／巡检回归通过，覆盖不输出大段历史、不改 task、完整模式保持内容，以及启动时半写事件不致失败。
- **规模实测：** masked 进入 `single_cta`、已开始 10 次调用时，完整 task JSON 为 31,163 字节，紧凑 status 约 1,091 字节，仍含当前 thread／105.7 秒调用年龄、Contract/root 已审阅范围；不输出重复的历史证据正文。后续会话／回执改进后执行器与巡检共 50 项回归通过。

<a id="se-020"></a>

## SE-020 问题投影为两条 note 重写整份台账

- **状态／日期：** `resolved`（真实补丁生成与应用）；2026-11-02。
- **真实依据：** masked 首个 accepted Contract 有两条 note。实际调用 `sync_issues` 生成 90,703 字节／680 行补丁，把 44,010 字节已有台账整体删后重加；审阅者需重新面对大量不变历史，也更容易与同时维护的台账冲突。
- **处理／验证：** 改用最小上下文差异，仍只生成可审阅的内置 `apply_patch` 格式，不直接改台账。同一真实 task 生成 2,297 字节／33 行补丁（减少 97.5%），父 Agent 已阅读并通过内置 `apply_patch` 成功应用；原条目保持不变。回归要求补丁包含实际新增问题且小于原台账，已有问题沿用原 marker。

<a id="se-021"></a>

## SE-021 独立审阅后重新创建作者会话，重复建立相同背景

- **状态／日期：** `resolved`（真实作者续接与独立审阅验证）；2026-11-02。
- **真实依据：** masked Contract 已完成并独立审阅后，root worker `a-0003-207b092a` 新 thread `01a19804-f36d-74d0-9bd9-c7cdde2f1f3f` 的 item_4 重读 EXTRACT／模板（10,311 字符），item_6 重读 common／math（18,199 字符），item_10 重读 candidate 2195–2497（19,250 字符），与 Contract worker item_16 的 2195–2495 几乎重合；item_11 再读整个 metadata 和 reference。Design 新内容的阅读合理，但重建刚核对过的 ABI／默认选择背景可以减少。
- **索引阶段复现：** `a-0009-031bb070/call-01` 的 item_3 新上下文整读 dispatch 与三份 YAML，单次输出 79,826 字符；item_4 又完整读取 EXTRACT／模板。此时 Contract/root/single_cta/mma_sync 均已有 complete 审阅，三份 YAML 与 `cp-0004-06f1ff3c9a` 的 SHA 一致。
- **处理：** 同次不中断 run 内保留作者的目录和明确 thread ID，审阅者仍新建独立只读会话；每次回到作者前核对原目录与当前 accepted checkpoint 完全一致。作者续接只接收当前阶段／节点、检查结果与新反馈，仍执行逐段读写回查。内存引用不跨恢复；暂停、取消、失败后保持新目录／新会话规则。
- **验证／关闭依据：** 多节点合成工作流覆盖作者跨四阶段同 thread、reviewer 独立、坏引文纠错、暂停后新目录／会话恢复。真实 MLA 在同一 `a-0001-251ba0f5` 中由 Contract 续接 root，未重读 EXTRACT、公共规范、输入生成与 reference；两次 reviewer 使用独立 thread。未把不同算子的总 token／时长作为严格 A/B 对照；续接原提示仍被诊断撑长，另由 SE-026 处理。

<a id="se-022"></a>

## SE-022 全空 experts 的零工作数量误写进正数 range

- **状态／日期：** `resolved`（worker 局部修正与检查通过）；2026-11-02。
- **真实依据：** masked root worker `a-0003-207b092a/call-01` 的 item_24 检查报告 `design.work_schedule.phase.main.decomposition.partition.work.range[0]: literal must be >= 1; Actual: 0`。源码和 Contract 允许所有 expert 为空，不能把有效域下界改成 1 来过检查。
- **处理／验证：** worker 删除可选 range，保留 `work: {size: T, tile: 1}`；当时 root.yaml:191–196 明确 `T=NB*sum_e Mblocks(e)`、`T=0` 对应空工作列表，:48 保留 all experts empty，:159 保留输出不变，pipeline 保留 empty return。观察 Agent 比较算法 Contract 与 `cp-0001-0fc0b65419` 完全相同；item_27 非硬件检查退出 0。这是有效的局部表达修正，未改检查器、未缩小要求，整体节点仍按正常流程独立审阅。

<a id="se-023"></a>

## SE-023 引文缩进转录错误被逐次暴露，回执诊断缺少锚点位置

- **状态／日期：** `resolved`（真实回执重放、合并诊断回归及新回执验证）；2026-11-02。
- **真实依据：** masked `single_cta` 作者 `a-0005-fcb2b6a6/call-03` 的路径终于修对后，又被 `evidence quote does not match fixed contents: single_cta.yaml` 拒绝。`artifact_evidence[0]` 的 42–43 行跨行引文遗漏第二行缩进；下一次调用重新读文件改成单行精确引文，并将 release 锚点行号更正为 406。只修回执，无产物改动。此前 root 作者 call-01 的原回执也有真实引文不匹配，旧验证先拒绝来源前缀而掩盖了它，不能把所有回执往返都算作无意义路径修复。
- **处理：** schema／提示建议短的单行精确引文；验证一次汇总所有无效 evidence 锚点，并指出 `source_evidence[i]`／`artifact_evidence[i]`、文件和行范围，避免逐次才发现下一处。仍逐字检查原文，不自动改写引文，也不将“语义相近”视为通过。
- **验证／关闭依据：** 回归同时注入来源与产物错误引文，一次报告两处且均不接受。真实旧回执重放见下；新 MLA 四份回执路径与引文均直接通过，没有纠错，也未触发机械归一化。不能因此把未触发的归一化分支称为 MLA 实跑覆盖。
- **已有验证：** 50 项回归通过；最终 18 份真实被拒回执，按原 prompt 身份／范围及 receipt-only 守护的原文件摘要离线重放，13 份完成路径归一化后严格通过，另 5 份仍因真实引文不匹配拒绝，且现在指出具体数组锚点／行范围。记录保存在 masked run 的 `receipt-replay.json`；这是离线验证，不等同于已经节省 13 次真实调用。

<a id="se-024"></a>

## SE-024 只读 reviewer 的 heredoc 隐式写临时文件失败

- **状态／日期：** `resolved`（新 MLA 只读调用未复现）；2026-11-02。
- **真实依据：** masked `single_cta` reviewer `a-0006-34ed3de7/call-01` item_8 使用 `python3 -B - <<'PY'` 只读解析 root，zsh 报 `can't create temp file for here document: operation not permitted`、退出 1；item_9 改用 `python3 -B -c` 成功。没有提权或改文件，非硬件检查已通过。
- **处理：** 读取提示建议 Python `-c` 或管道，避免 shell heredoc 的隐式临时文件；沙箱和网络政策不变。
- **验证／关闭依据：** 原运行自行用 `-c` 完成同项读取；新 MLA 的作者与两次独立 reviewer 均未发生 heredoc 临时文件失败。没有增加预检、临时目录权限或额外 agent。

<a id="se-025"></a>

## SE-025 索引阶段扩展为重新理解已审阅的 Design

- **状态／日期：** `resolved`（权威绑定、修订回路及真实多节点收尾验收通过）；2026-11-02。
- **真实依据：** masked index `a-0009-031bb070/call-01` 除整读 79,826 字符产物外，item_8/11 两次读 root 281–561 行完整 pipeline/realization，item_7/12 完整读 math 规范，item_15 读 schedule/pipeline 规范，item_16 读 root 源码 1240–1591 行。此时三个节点已有 complete 独立审阅。最终实际只将 dispatch.md 从 9,292 字节精简为 6,480 字节，三个 YAML 的 SHA 与 `cp-0004-06f1ff3c9a` 相同，item_25 非硬件检查及 item_26 链接／文件集合检查通过。
- **final 复现：** `cp-0005-949e9eb079` 正确保留四项既有审阅且直接进入 final，没有另派节点审阅；但新 reviewer `a-0010-9c4b2377/call-01` item_4 又读 53,214 字符 Design 规范，item_5 打印 root 全部非硬件正文，item_7/9 读 single_cta 1–453 全文，item_13 读 mma_sync 全文，随后再读各路实现／参考。此时三份 YAML 与此前已独立审阅版本完全一致。
- **处理：** 作者提示明确显示已完成审阅的范围，index 只处理索引与选择关系，有具体矛盾才重新展开 Design。多节点 final 只有在每个节点已有同 fingerprint、同 Contract 的 complete 审阅且未被失效时，才使用独立审阅摘要并聚焦当前 metadata／index／host selector；缺任意审阅或单节点 final 保持完整实质审阅。所有当前文件锚点和 ready 语义绑定条件不变。
- **前一轮验证范围：** 当时 52 项回归覆盖多节点 final 路由及缺失审阅后的完整范围。MLA 为单节点，当时只实证了单节点完整 final，故保留此项待后续多节点验收。
- **本轮机制改进：** `review_current` 统一用于复用、补审和 ready 判断；checkpoint 原子保存完整 dispatch 声明，选择关系修订号随既有 when／parent 改变递增。单纯 reason 润色只撤销 final，不撤销节点。索引上下文由当前 checkpoint 派生；仅索引小修用 `impact=index`，实际修改节点则补独立审阅，完成后直接 final，避免再派一次 index 作者。
- **父端验收修正：** 拦下了把自然语言 reason 也当成全量失效依据的初稿。还实证 AST 直接序列化会把 `!!str 080` 等变成数字；现改为使用固定 parser 取值与读回严格比较，新增 `yaml_source.loads` 供内存复核，解析规则只有一份，不写临时文件。核心类型、alias／merge、失效与暂停恢复的回归已覆盖。
- **真实关闭依据：** 两个全新实现 agent 分工，父端集成与验收。`refine_20261102_fp8` / `fp8_scaled_gemm-97d141747cf0` 达到 ready，最终 `cp-0004-36afb82b91`。root v8、small_m_transpose v4 的首次独立审阅均完成。`cp-0003-d2537e9996`→最终检查点只改 dispatch.md，两份 YAML 的 SHA 不变。index 作者只展开索引和 host 选择源码，final 读取 root 的 metadata／Contract 最远至第 130 行（Design 从 144 行开始），派生节点只读第 1–9 行（Design 从 10 行开始）；没有重新展开两个节点的 Design 正文或分区规范。两次调用各有由当前检查点派生的 `review_context.json`，共 8 次调用，无额外节点重审。最终 eligible 已用冻结执行器复验。
- **验收边界：** 真实 final 当时仍读了三份通用说明。其后把有效索引上下文的规则入口收窄至 `schema_docs/0.common.md` 的 dispatch／索引章节，已在同一真实上下文离线核对路由；未把后续提示冒充冻结 run 行为。索引小修、实际 YAML 变化补审、reason 润色、when／parent 变化、旧选择版本、暂停恢复与标量读回等分支另有回归。最终执行器相关 95 项、检查器 147 项测试通过。

<a id="se-026"></a>

## SE-026 原始检查诊断的重复绝对路径撑大作者续接提示

- **状态／日期：** `resolved`（真实提示离线重放与内容守恒验证）；2026-11-02。
- **真实依据：** 新 MLA 作者复用了原 thread，却收到 13,093 字符的 node 续接提示，长于首次 10,151 字符。主要增量是前一 Contract 检查的 18 条 Design 占位诊断，每条重复长的工作区与规范绝对路径；单个 check JSON 占 7,890 字符。不能把“未重发背景”误称为“提示一定更短”。
- **处理：** 只在发给 agent 的副本中，将诊断位置、See／Shared root 的重复目录前缀缩成相对路径，并一次说明两类根。保留每条字段、行列、错误、Actual 值与 Fix 提示；Actual 中的绝对路径文本不改。原 task、check.json、argv、检查点和交付判据保持完整。
- **验证：** 真实 `a-0001-251ba0f5/call-02/prompt.txt` 离线等效转换后 13,093→9,153 字符（约 30.1%），check 部分 7,890→3,950；18 条错误及 Actual／Fix 逐字相等，原文件未改。报告在 MLA run 的 `feedback-replay.json`。新增回归覆盖多节点位置、来源文档、Actual 绝对路径不误改及成功检查不增加说明。此修改在 MLA 冻结运行开始后完成，未冒充该 run 已使用的提示。

<a id="se-027"></a>

## SE-027 收尾简写替换了问题首次发现中的具体源码依据

- **状态／日期：** `resolved`（两份真实台账投影与回归验证）；2026-11-02。
- **真实依据：** masked 与 MLA 最终 reviewer 把部分 note 写成 Retained unchanged，并仅给 dispatch／产物行号。原 `sync_issues` 会用这些当前字段替换初次 metadata／candidate 源码锚点；当前小结虽准确，台账却失去可独立复查的具体初次发现。
- **处理：** 投影按已有 `first_checkpoint`／`ledger_task_key` 从可用的不可变 worker 回执保留初次 detail 与 evidence，同时显示当前结论和检查点。校验 checkpoint ID 与原 task key；旧记录已清理或找不到原问题时不臆造来源。不新增状态存储、不改历史回执，也不加载整个历史会话。
- **验证：** 两题 7 项 note 的实际小补丁已由父端审阅并应用，均保留初次发现与源码证据；回归模拟最终仅剩简写／dispatch 引用，确认具体来源仍在补丁且 task 字节不变。完整执行器／巡检 52 项测试通过。此投影改进在两次运行结束后直接使用，不冒充模型的新判断。

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

- **状态：** `open`；类型 `evidence_gap`；处理 `note`。
- **任务／阶段：** `fp8_scaled_gemm` / root.yaml:111-119; dispatch.md:16,19；run `policy_xhigh_20261101_fp8`。
- **现象与依据：** Comparison precision and NaN/Inf handling remain unspecified. Candidate-relative tolerance, zero permitted mismatches and no independent relative-L2 threshold are preserved. Origin remains task fp8_scaled_gemm-1efaafba78b1, first checkpoint cp-0001-9e60b8bf88; subsequent provenance is preserved in dispatch.md:19.
- **证据：** kernel_wiki/aka-cuda-operators-20261030/operators/gemm/dense/fp8_scaled_gemm/performance/ACCEPTANCE.json:125-160; dispatch.md:16,19
- **基线／检查点：** `dcb921aefcb5896535d13ce6f7018d5ac4b8ebf9a0d06bde22cdfc85087e7388` / `cp-0003-0bd256ca55`。
- **问题来源：** `fp8_scaled_gemm-1efaafba78b1` / `cp-0001-9e60b8bf88`；保留原问题身份。
- **影响与重启条件：** Complete numerical acceptance semantics and certification beyond recorded evidence.；Supply the fixed numerical checker or authoritative comparison precision and non-finite handling semantics.
- **本轮复现：** `refine_20261102_fp8` / `fp8_scaled_gemm-97d141747cf0` 的本地 note `comparison_policy_details` 复现同一来源问题，沿用本条稳定 ID。新输入基线 `97d141747cf0e43826efcc96d0d65c82f4e0c363d738abcde9d6d568af1de588`，初次 `cp-0001-51b2c0fb7a`，最终 `cp-0004-36afb82b91`；依据为本题 `performance/ACCEPTANCE.json:135–141`、`root.yaml:123–130`、`dispatch.md:14`。未另建重复条目。

<!-- /extraction -->

<!-- extraction:fp8_scaled_gemm-1efaafba78b1:declared_alias_and_masking_domain -->

## SE-fp8_scaled_gemm-1efaafba78b1-declared_alias_and_masking_domain

- **状态：** `open`；类型 `evidence_gap`；处理 `note`。
- **任务／阶段：** `fp8_scaled_gemm` / Shared Contract’s additional declared branches; dispatch.md:10,17,19；run `policy_xhigh_20261101_fp8`。
- **现象与依据：** Metadata declarations exceed the supplied ABI: no C input exists and logical_m does not mask reference rows. No additional semantics or fallback were invented. Origin remains task fp8_scaled_gemm-1efaafba78b1, first checkpoint cp-0001-9e60b8bf88; subsequent provenance is preserved in dispatch.md:19.
- **证据：** kernel_wiki/aka-cuda-operators-20261030/operators/gemm/dense/fp8_scaled_gemm/problem/metadata.json:4-30; kernel_wiki/aka-cuda-operators-20261030/operators/gemm/dense/fp8_scaled_gemm/problem/input.py:63-81; kernel_wiki/aka-cuda-operators-20261030/operators/gemm/dense/fp8_scaled_gemm/problem/reference.py:2-16; dispatch.md:17,19
- **基线／检查点：** `dcb921aefcb5896535d13ce6f7018d5ac4b8ebf9a0d06bde22cdfc85087e7388` / `cp-0003-0bd256ca55`。
- **问题来源：** `fp8_scaled_gemm-1efaafba78b1` / `cp-0001-9e60b8bf88`；保留原问题身份。
- **影响与重启条件：** Complete semantics and implementation coverage of the additionally declared branches.；Supply fixed ABI facts defining the additional branches or authoritative clarification of their applicability.
- **本轮来源核对：** `refine_20261102_fp8` 的最终 `dispatch.md:12` 保留了 metadata 中的可选 C 别名描述、实际 ABI 无 C 参数以及 logical_m 不缩小写区的区别。索引的分派覆盖结论限于已定义的共享 Contract 域，不据此关闭本条来源差异。

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

- **状态／日期：** `resolved`（差分观察回归与真实续接验证）；2026-11-01。
- **实证：** CLI 0.163.0 的 GDN thread `01a1974b-256d-7132-9f4f-16839fbb298b` 首调用累计 1,760,088，续接累计 2,053,803；定向 rollout 中续接两次内部请求分别 144,630 / 149,085，增量 293,715。原小探针亦为 64,652→132,889，增量 68,237。旧 runner 逐次累加累计值，GDN 历史总和 7,634,402 多计 1,760,088，实证差分总和 5,874,314。
- **当前处理／验证：** 原始 result/events 和旧 task 状态不改写。新调用按 thread 保存上一累计值，只增加 input/output 差额，不重复加 cached/reasoning；重复观察幂等，异常只记录、不阻断。回归测试通过；本轮真实 thread `01a197ba-655c-7720-9e95-6db63b94e8bb` 累计 65,326→117,033，程序观察增量 51,707，无重复加和。依据：`.schema_extract/probes/policy_20261101/cli/` 与 `verification.json`。

<a id="se-011"></a>

## SE-011 旧 900 秒超时与预算中断已移除

- **状态／日期：** `resolved`（限额移除及当前恢复路径已验证）；2026-11-01。
- **实证：** 第二轮 FP8 root 有一次旧 900 秒自动超时，随后从已接受检查点的新目录恢复；该次用量未知。最终 FP8 已报告 8,777,991 token、9 次调用、4045.044 秒，在 index 派发前触发旧 8M 上限，两个节点均已协作审阅。
- **当前处理／验证：** 不再有 agent 时长/token/调用/修订/无进展/续接上限，也不读取旧 limit 字段。46 项执行器／巡检测试通过，包含旧限制无效、长时观察与恢复边界。真实 xhigh 探针完成新会话 7→8、同 ID 8→9、取消三层进程确认、仅从 accepted 8 新会话恢复为18、独立只读复核；未接受半成品不参与恢复。超过60分钟的观察使用合成时钟测试，没有声称完成60分钟真实长跑。CLI 版本预检与取消后的清理等待不构成 agent 额度。
- **2026-11-02 补验：** 新 MLA 作者同 thread 的 `a-0001-251ba0f5/call-02` 实际持续 1,048.449 秒并自然完成，最终节点通过非硬件检查；未被旧 900 秒边界中断。这验证当前政策在真实较长节点调用中的生效，不代表做过单次超过 60 分钟的调用测试。

<a id="se-012"></a>

## SE-012 新政策与旧冻结计划的恢复兼容边界

- **状态／日期：** `resolved`（窄政策过渡已审计并实际建立新检查点）；2026-11-01。
- **范围：** 旧 accepted 草稿、输入冻结、seed/resume。
- **现象与依据：** 新政策修改了 EXTRACT_SCHEMA、检查器 API／策略、协议和执行器内容摘要。旧计划绑定这些文件，正常 resume 会拒绝静默采用新程序；seed 的旧语义文件／检查器／政策一致性核验同样会拒绝跨此次变更复用。取消限额不等于允许偷换旧前提。
- **处理与边界：** 新增显式 `--seed-policy-update skip-hardware`，仅放行已审计的 EXTRACT_SCHEMA、根 README、模板 README、check_yaml.py、tools/schema.py 五组旧／新 SHA；其余原有来源、schema_docs、starter、检查规则逐文件不变。普通 seed 与旧 plan 直接续跑仍拒绝，不能借此放行其他政策或语义变更。
- **实际依据：** 新 run `policy_xhigh_20261101_fp8`，task `fp8_scaled_gemm-dcb921aefcb5`，seed `cp-0001-seed-a3b062bf`。只在新副本恢复 starter 硬件占位，root v10→v11；算法 Contract、Design、realization 不变，小 M v8 原字节保留，旧回执／用量／issue 身份继承，旧 reviews 清空，非硬件检查返回0。已有完整 Design 直接重新复核，避免重新派写。父端核验见 `.schema_extract/probes/policy_20261101/seed-verification.json`；135个历史运行文件摘要保持不变。

<a id="se-013"></a>

## SE-013 Supervisor 周期唤醒与会话续接验证

- **状态／日期：** `resolved`（本机 macOS 真实周期与后台派发已验证）；2026-11-01。
- **已实现：** 独立 `supervisor.py` 的只读 compact 巡检与可选15分钟 macOS LaunchAgent；明确 thread ID 续接、独立 run 锁、绝对路径，固定程序只可请求已选 run 的 queued 工作。不得恢复用户 pause/cancel，不自动 export；长时活动标记不触发 kill。
- **当前实测：** 独立合成 run 两次真实巡检相隔900.29秒，第二次由 LaunchAgent 自然唤醒，明确续接 thread `01a197b6-80a9-73e1-a3ab-8e555925791f`；两次均使用 `gpt-6-astra / xhigh`、只读调用，结束后无所属进程残留。pause/cancel 下不调用模型；运行中 cancel 也确认进程停止且 task 未改。探针 disable 的 bootout 返回0、plist已删除、系统查询确认服务不存在。依据：`.schema_extract/probes/policy_20261101/verification.json`。
- **实际派发：** 新 FP8 run 的真实 tick 返回 `start_queued` 后正常退出，固定程序启动后台 runner；独立查询确认锁已持有、reviewer 进程存活并产生事件。900秒后系统再次唤醒同一 Supervisor 会话，观察到节点 worker 和产物在推进，返回 `observe` 后退出，没有重复派发或恢复旧暂停任务。两个 worker 忙时仍可独立只读巡检的回归通过；大规模与其他系统平台不在本轮验证范围。
- **收尾：** FP8 达到 ready 后 runner 锁释放，所有已记录调用确认结束；实际 run 的 LaunchAgent 也已卸载，plist删除、服务不存在，未留下巡检 pending。依据：该 run 的 `supervisor-final-status.json`。

<a id="se-014"></a>

## SE-014 未启动调用被误当作终止未确认

- **状态／日期：** `resolved`；2026-11-01。
- **范围／依据：** runner 等待 reviewer 名额前登记 call，随后 pause 会留下不存在的调用；CLI 版本预检或 Popen 失败也会留下无 process/result 的记录。旧恢复与 seed 不能确认终止，导致实际没有 agent 的任务无法再用。
- **处理／验证：** 调用登记移至取得名额并复查控制请求后；已确定未启动的预检／spawn 错误保存 `preflight_error`／`spawn_error` 与 `process_not_started` 证据。允许使用既有明确恢复终止记录，仍不把“缺 process 文件”当安全证明。暂停竞态、不存在的可执行文件→显式恢复／seed、spawn失败与版本预检超时的回归通过。

- **本轮收尾：** index/final 的上下文与提示先准备完成，再复查 pause/cancel，最后才登记和派发调用。故障注入验证准备失败不会增加调用计数或留下未启动的 process，可从原 accepted checkpoint 显式恢复；准备期间收到 pause/cancel 也不派发模型。

<a id="se-015"></a>

## SE-015 坏回执绕过文件边界或污染新尝试

- **状态／日期：** `resolved`；2026-11-01。
- **范围／依据：** 重复坏回执能在校验文件前覆盖 receipt-only 基线，reviewer 修改产物也可能先进入回执重试；pause 后恢复新目录却保留旧 `receipt_correction`，会禁止 worker 重建未接受的文件。
- **处理／验证：** 先核对 reviewer／receipt-only 文件边界，再处理回执；新 attempt 清除旧回执修复限制。加入“坏回执→pause→新目录恢复→重新建立 accepted CP”等回归。`revise`／`needs_input` 须有对应具体问题，Contract 接受须明确 `proceed`，不增加证明或格式要求。MAX 修改后由独立 xhigh 审阅确认。
- **2026-11-02 会话复用补充：** 合法回执接受后立即解除该次 receipt-only 文件禁改标志，避免作者后续正常 continue／跨阶段写入被旧标志拒绝；验证顺序仍是先检查文件未改，再校验回执，最后解除。新增“坏回执→同会话只修回执→保存 continue→正常修改并提交下一检查点”回归，50 项执行器／巡检测试通过。

<a id="se-016"></a>

## SE-016 巡检取消与调度切换的进程边界

- **状态／日期：** `resolved`；2026-11-02。
- **范围／依据：** tick 未传 CLI cancel 回调；resume 在 thread.started 前被取消时，收尾还可能把取消／未确认终止覆盖为 session_mismatch。
- **处理／验证：** tick 传入 run 的 cancel 请求，pause 保持调用边界语义；取消与未确认终止保留真实状态，正常退出也确认所属进程清理。真实巡检取消返回 `cancelled`、终止确认、task 字节不变，后续 tick 为 idle；记录见 `supervisor-cancel.json`。回归同时覆盖两个 worker 占满时的独立巡检。
- **收尾补充：** 原 enable/disable 可在 tick 活跃时调用 bootout，存在结束父进程后遗留独立 Codex 调用的风险。调度更新现全程持有 tick 锁，活跃时明确拒绝变更；没有自动取消 run。真实 flock 回归覆盖活跃时不调用 launchctl、不改配置，以及空闲启停全过程阻止新 tick；Supervisor 15项测试通过。

<!-- extraction:fp8_scaled_gemm-dcb921aefcb5:dispatch_checker_status -->

## SE-fp8_scaled_gemm-dcb921aefcb5-dispatch_checker_status

- **状态：** `resolved`；类型 `transcription`；处理 `fix`。
- **任务／阶段：** `fp8_scaled_gemm` / dispatch.md:1,28,35,37；run `policy_xhigh_20261101_fp8`。
- **现象与依据：** Replace the superseded current-stage checker summary with the observed --skip-hardware exit-0 result covering both YAML nodes. The index currently says named relationship checks have not run. Align its opening status and issue count with the current policy and two retained evidence-gap notes. Historical checks may remain clearly labeled as superseded. No YAML or preserved source-binding changes are requested.
- **证据：** dispatch.md:28: 具名关系检查因此尚未运行; dispatch.md:35: 具名关系检查仍未运行; Read-only checker rerun: check_yaml.py --skip-hardware root.yaml exited 0; 2 YAML nodes, whole task checked.
- **基线／检查点：** `dcb921aefcb5896535d13ce6f7018d5ac4b8ebf9a0d06bde22cdfc85087e7388` / `cp-0001-seed-a3b062bf`。
- **影响与重启条件：** Accurate current-stage status in the task index.；Review the corrected dispatch status against the current checker result.
- **关闭依据：** worker 在 `cp-0002-78dc54bc16` 仅修改 dispatch.md，独立 reviewer 确认当前 `--skip-hardware` 检查返回0；最终 `cp-0003-0bd256ca55` 进一步精简索引并通过整体复核，两份 YAML 未改动。

<!-- /extraction -->

<!-- extraction:bf16_grouped_gemm_masked-7ffc871299d8:source_storage_001 -->

## SE-bf16_grouped_gemm_masked-7ffc871299d8-source_storage_001

- **状态：** `open`；类型 `implementation_mismatch`；处理 `note`。
- **任务／阶段：** `bf16_grouped_gemm_masked` / Shared write boundary and candidate epilogues；run `trial_20261102_masked_baseline`。
- **现象与依据：** Retained from cp-0001-0fc0b65419 through cp-0005-949e9eb079. All three paths can overwrite inactive tail rows through rounded block stores. The artifacts record this defect while preserving the strict Contract.
- **证据：** dispatch.md:15; root.yaml:68-75; mma_sync.yaml:54-65
- **初次发现（`cp-0001-0fc0b65419`）：** Metadata requires inactive storage preservation, but candidate epilogues write complete row blocks. The Contract retains the strict boundary; subsequent Design must describe actual writes.
- **初次证据：** problem/metadata.json:19; candidate/solution.py:208-215; candidate/solution.py:617-652; candidate/solution.py:1491-1517,1574-1578,1732
- **基线／检查点：** `7ffc871299d8cff6e06f9b222b0e81b689bde01f7bbee8b06fdb50e54417ee08` / `cp-0005-949e9eb079`。
- **影响与重启条件：** ；Reassess if authoritative requirements or candidate source change.

<!-- /extraction -->

<!-- extraction:bf16_grouped_gemm_masked-7ffc871299d8:comparison_evidence_001 -->

## SE-bf16_grouped_gemm_masked-7ffc871299d8-comparison_evidence_001

- **状态：** `open`；类型 `evidence_gap`；处理 `note`。
- **任务／阶段：** `bf16_grouped_gemm_masked` / External comparator provenance；run `trial_20261102_masked_baseline`。
- **现象与依据：** Retained from cp-0001-0fc0b65419 through cp-0005-949e9eb079. External comparator precision and nonfinite handling remain unverified; the artifacts explicitly identify the schema conventions used.
- **证据：** root.yaml:83-90; dispatch.md:16
- **初次发现（`cp-0001-0fc0b65419`）：** Metadata supplies tolerance thresholds but the operator bundle lacks the external comparator implementation. Schema FP32/reference-relative conventions are explicit; external precision and nonfinite handling remain unverified.
- **初次证据：** problem/metadata.json:31-35; performance/CERTIFIED_BEST.json:3-14; root.yaml:83-90
- **基线／检查点：** `7ffc871299d8cff6e06f9b222b0e81b689bde01f7bbee8b06fdb50e54417ee08` / `cp-0005-949e9eb079`。
- **影响与重启条件：** ；Reassess if the fixed comparator source becomes available.

<!-- /extraction -->

<!-- extraction:mla_expanded_fp8-aa60ab8be0d4:domain_summary_drift -->

## SE-mla_expanded_fp8-aa60ab8be0d4-domain_summary_drift

- **状态：** `open`；类型 `transcription`；处理 `note`。
- **任务／阶段：** `mla_expanded_fp8` / contract.algorithm_requirement；run `trial_20261102_mla_improved`。
- **现象与依据：** Retained unchanged: Contract follows the explicit local dynamic domain despite conflicting catalog prose.
- **证据：** dispatch.md:21
- **初次发现（`cp-0001-a0202f19df`）：** Catalog prose differs from the explicit local dynamic domain. Contract follows shape_range.json: H in {12,24,32,96,128}, with noncausal attention and optional sinks allowed.
- **初次证据：** problem/metadata.json explicitly assigns the dynamic domain to shape_range.json.; problem/shape_range.json defines the correlated head geometry and boolean options.
- **基线／检查点：** `aa60ab8be0d4ad37cd003bfc3d24c8212815a3d2595b0f241986bf7bfc651005` / `cp-0002-b297748059`。
- **影响与重启条件：** ；Task authority changes or clarifies the explicit dynamic domain.

<!-- /extraction -->

<!-- extraction:mla_expanded_fp8-aa60ab8be0d4:comparator_evidence -->

## SE-mla_expanded_fp8-aa60ab8be0d4-comparator_evidence

- **状态：** `open`；类型 `evidence_gap`；处理 `note`。
- **任务／阶段：** `mla_expanded_fp8` / contract.algorithm_requirement.golden_reference.tolerance；run `trial_20261102_mla_improved`。
- **现象与依据：** Retained unchanged: comparator details beyond supplied threshold declarations remain unverified.
- **证据：** root.yaml:135-153; dispatch.md:22
- **初次发现（`cp-0001-a0202f19df`）：** Preserved metadata thresholds and the candidate numerical-policy declaration. The frozen comparator is unavailable, leaving the relative-error operand, L2 definition, comparison precision, nonfinite handling, and returned-alias checks unverified.
- **初次证据：** problem/metadata.json:38-49; candidate/solution.py:41-53
- **基线／检查点：** `aa60ab8be0d4ad37cd003bfc3d24c8212815a3d2595b0f241986bf7bfc651005` / `cp-0002-b297748059`。
- **影响与重启条件：** ；The fixed-version frozen comparator becomes available.

<!-- /extraction -->

<!-- extraction:mla_expanded_fp8-aa60ab8be0d4:source_comment_drift -->

## SE-mla_expanded_fp8-aa60ab8be0d4-source_comment_drift

- **状态：** `open`；类型 `transcription`；处理 `note`。
- **任务／阶段：** `mla_expanded_fp8` / design；run `trial_20261102_mla_improved`。
- **现象与依据：** Executable x256 weights, maximum freezing, host geometry and dedicated short-row scratch are faithfully preserved despite stale comments.
- **证据：** dispatch.md:23
- **初次发现（`cp-0002-b297748059`）：** Executable definitions use x256 weights, maximum freezing, host-built geometry and dedicated short-row scratch. Design preserves these facts despite stale introductory comments describing earlier choices.
- **初次证据：** candidate/solution.py:453-458,484-490; candidate/solution.py:1785-1817; dispatch.md:23
- **基线／检查点：** `aa60ab8be0d4ad37cd003bfc3d24c8212815a3d2595b0f241986bf7bfc651005` / `cp-0002-b297748059`。
- **影响与重启条件：** ；Executable definitions change in a newly fixed source revision.

<!-- /extraction -->

<!-- extraction:mla_expanded_fp8-aa60ab8be0d4:numerical_ordering_evidence -->

## SE-mla_expanded_fp8-aa60ab8be0d4-numerical_ordering_evidence

- **状态：** `open`；类型 `evidence_gap`；处理 `note`。
- **任务／阶段：** `mla_expanded_fp8` / design.math_reshape and realization；run `trial_20261102_mla_improved`。
- **现象与依据：** Approximate arithmetic and source-dependent MMA ordering remain explicitly qualified; no full-domain numerical bound or independent ordering proof is asserted.
- **证据：** root.yaml:260-268; root.yaml:680-682; dispatch.md:24
- **初次发现（`cp-0002-b297748059`）：** Recorded approximate arithmetic and source-dependent cumulative MMA ordering without asserting a full-domain numerical bound or an independent primitive-ordering proof.
- **初次证据：** candidate/solution.py:1015-1031; candidate/solution.py:1254-1364; candidate/solution.py:2054-2058; dispatch.md:24
- **基线／检查点：** `aa60ab8be0d4ad37cd003bfc3d24c8212815a3d2595b0f241986bf7bfc651005` / `cp-0002-b297748059`。
- **影响与重启条件：** ；Matching numerical bounds or primitive-ordering evidence becomes available.

<!-- /extraction -->

<!-- extraction:mla_expanded_fp8-aa60ab8be0d4:invocation_ordering -->

## SE-mla_expanded_fp8-aa60ab8be0d4-invocation_ordering

- **状态：** `open`；类型 `evidence_gap`；处理 `note`。
- **任务／阶段：** `mla_expanded_fp8` / design.work_schedule.phase.attention.schedule；run `trial_20261102_mla_improved`。
- **现象与依据：** Missing incoming producer synchronization and shared static invocation state remain documented without adding guarantees or narrowing Contract.
- **证据：** root.yaml:392-407; dispatch.md:25
- **初次发现（`cp-0002-b297748059`）：** The nonblocking metadata-copy stream has no explicit incoming producer-event wait, and shared static tables, queue and caches lack invocation isolation. Actual synchronization and persistent side effects are preserved; concurrent-call safety is not asserted.
- **初次证据：** candidate/solution.py:1638-1681; candidate/solution.py:1725-1779; candidate/solution.py:1895-1898,2010-2011; dispatch.md:25
- **基线／检查点：** `aa60ab8be0d4ad37cd003bfc3d24c8212815a3d2595b0f241986bf7bfc651005` / `cp-0002-b297748059`。
- **影响与重启条件：** ；Supporting invocation-order guarantees or revised synchronization source becomes available.

<!-- /extraction -->
