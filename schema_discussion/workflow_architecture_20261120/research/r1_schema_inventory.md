# R1：KernelSchema 现状、可复用资产与边界

调研日期：2026-11-20。仓库：KernelSchema；盘点 HEAD 为 `484902855fe7be82a74fe7c394503d1d7d222dbb`。本报告依据当前工作树中的活动文档、实现及少量只读运行验证；历史记录仅说明当时发生的事。未改实现、既有 Schema、提取状态或用户未提交成果，未运行模型、GPU、安装依赖、提交 Git 或读取凭证。本文不是实施计划，也不代表用户已批准架构。

## 1. 已核实的总体判断

1. **仓库已有实质工具能力，但四项功能的自动化程度不同。** extract 有真实六阶段 Runner、模型传输、审阅、恢复、发布；design、optimize、implement 有内容明确的角色工作方法，却没有各自的可调用执行入口、任务输入协议或端到端编排。不能把 README 中“四个任务入口”解释成四个软件 API。证据：README.md:13、schema_extract/tools/cli.py:87、DESIGN_SCHEMA.md:3、OPTIMIZE_SCHEMA.md:3、IMPLEMENT_SCHEMA.md:3。
2. **最直接的共享资产是当前语言规则与 Template API。** 它们已经解决整区引用、使用上下文、原文/有效视图、来源闭包、规范化、结构关联检查、有限事件图等问题；新流程无须另造一套 Schema 解析器或“展开器”。证据：schema_template/tools/README.md:31、schema_template/tools/schema.py:75、schema_template/tools/design_resolution.py:29。
3. **extract 的完成只证明相应提取门禁通过。** 它故意保留硬件占位、跳过硬件检查，并允许真实外部验收公式缺项；不能由 extract 的 ready、P6 或 receipt 推出新设计可实现、代码正确或性能合格。证据：EXTRACT_SCHEMA.md:44、schema_extract/tools/publication.py:71、schema_docs/1.contract.algorithm_requirement.md:254。
4. **实现迭代与设计迭代目前只有文档上的职责区分，没有统一执行状态模型。** 现有 Version 是单个 raw 文件修订号；dispatch 节点是当前并存的具体设计；两者都不是 coder episode。证据：schema_docs/0.common.md:37、schema_docs/0.common.md:43、OPTIMIZE_SCHEMA.md:73、IMPLEMENT_SCHEMA.md:22。
5. **示例是真实来源的设计教材、静态检查与局部事件图 demo，尚非“由设计生成代码并优化”的产品 demo。** 仓库明确没有统一 GPU Harness；历史包内虽有 harness 与复现脚本，不能自动视为现行通用评测后端。证据：schema_template/examples/README.md:3、README.md:108、IMPLEMENT_SCHEMA.md:46、kernel_wiki/EVIDENCE.md:70。

## 2. 当前能力矩阵

下表“已实现”指当前存在可执行代码及相应职责；本轮是否实际执行，另见第 7 节。

| 能力 | 当前真实入口与产物 | 已实现程度与限制 |
| --- | --- | --- |
| design | DESIGN_SCHEMA.md；从已确认 Contract 形成 root、必要完整节点与 dispatch | **方法文档**。明确数学→调度→流水线逐层写作与最早受影响层回退，尚无模型驱动命令或固定请求/结果协议。DESIGN_SCHEMA.md:9、:17、:45 |
| extract | schema_extract/extract_yaml.py extract；支持 catalog ID、目录/来源描述、前台/后台/排队 | **完整工具流程**。P1–P6、worker/reviewer、P5 就绪节点并行、原子发布；输出 raw YAML、中文 dispatch、receipt。schema_extract/tools/cli.py:102、schema_extract/tools/stages.py:283 |
| optimize | OPTIMIZE_SCHEMA.md；审查反馈后修订当前节点或保留设计 | **方法文档**。允许无真实反馈时作有依据的分析，也允许本轮不改设计；没有迭代策略执行器、实验选择或预算管理。OPTIMIZE_SCHEMA.md:13、:23、:39、:77 |
| implement | IMPLEMENT_SCHEMA.md；代码、构建/验证结果与有据反馈 | **方法文档**。规定代码符合性和正确性/性能分别验收，但没有 coder 工具调用循环、GPU/Agate 适配或多 episode 续跑实现。IMPLEMENT_SCHEMA.md:24、:40、:56 |
| 校验与规范化 | schema_template/check_yaml.py；严格只读 Python API | **已实现**。CLI 默认可能改写全任务 YAML；只读 API 与规范化规划分开。结构/文本预算/引用/具名关系并非数学或性能证明。schema_template/README.md:36、:87 |
| 有效视图与来源 | check_yaml.py --show-effective；ResolvedTask；extract_yaml.py standalone | **已实现**。show-effective 明示 validation:not_run；standalone 是先完整检查再导出的独立阅读视图，不是 standalone implement。schema_template/check_yaml.py:270、schema_extract/tools/standalone.py:28 |
| 微流水线图 | check_yaml.py --graph check/png | **已实现的有限分析**。显式样本、每 step 代表 worker 的局部事件；不跨 step 建完整执行图，不解析动作正文来证明存储安全。schema_template/README.md:81、:85；schema_template/tools/README.md:57 |
| 参考检索 | kernel_wiki/tools/catalog.py list/show/check/render | **已实现的目录检索**。44 题、分类/精度/阶段/验收状态与关键词 AND 过滤，JSON 输出；不是语义近邻排序器或自动“检索→设计”链。kernel_wiki/tools/catalog.py:332、:376 |
| 模型调用 | schema_extract/tools/agent.py、backends.py | **已实现的内部 CLI 适配**。Claude/Codex 两种；固定角色、精确会话、原生结果/JSON/nonce 校验与用量。没有通用公开 planner/coder SDK。schema_extract/tools/agent.py:137、:155 |
| 状态、恢复、控制 | extract 的 status/inspect/pause/cancel/resume/maintain/repair-runtime/audit/monitor | **已实现，但绑定 extract**。run、scope、attempt、checkpoint 与 P1–P6 批准紧密关联；没有跨 design→implement→optimize 的总运行状态。schema_extract/README.md:94、:100；schema_extract/tools/runtime.py:110 |
| 独立审阅与回归 | extract REVIEW、冻结 reviewer prompt、review-eval、离线 tests | **已实现提取审阅协议及小型能力回归**。review-eval 固定为合成标量复制文档，真实调用模型；不是 kernel 正确性评估或通用语义证明。schema_extract/tools/README.md:118、schema_extract/tools/review_eval.py:145 |
| 公共安装与集成 | 当前以仓库内 Python 脚本、文件路径调用 | 盘点未发现根目录 pyproject/setup、统一依赖清单、.gitmodules 或三流程专用模块。Template 与 Extract 都有 tools 包，不能把它们当一个同名模块无条件拼接；现有 Extract 用独立 import 命名加载冻结 Template。schema_extract/tools/artifacts.py:46 |

## 3. 四流程共同输入输出必须尊重的现行语义

### 3.1 设计节点、dispatch 和范围

**已核实事实：**一个任务一个目录；root 保存唯一 Contract 且 Design 三区全本地，其他节点省略 Contract，三区各自完整本地定义或仅写 `{from: id}`；跨区整体引用图无环。完整集合含 root 至多十份；节点必须有独特完整设计价值，不能作纯 provider、参数组合库或优化历史。证据：schema_docs/0.common.md:134、:192；schema_template/tools/design_resolution.py:15、:155。

dispatch.md 是中文设计目录。它记录完整方案、参数/局部/组合差异和依据，**不执行 selector、不应用 patch、不规定优先级，也不补齐 YAML 缺失的设计**。when 的范围可重叠、可不覆盖整个 Contract；root 是默认阅读入口，不自动作为未覆盖输入的 fallback。证据：schema_docs/0.common.md:146、:180、:190。

这与“以 root 为主干逐步实现必要 dispatch”的方向相容，但目前没有机器机制决定“必要”指哪些节点、哪些笔记，以及何时达到任务要求的覆盖。真实例子：FA4 decode root 的 when 只取 B∈{15,16} 与较长 KV，不能把实现 root 宣称为覆盖共享 Contract 的全部 B/Q/L。证据：schema_template/examples/fa4_decode/root.yaml:8、:38。

### 3.2 版本身份不是一个数

**已核实事实：**Version 只随本 raw 数据改动增长；共享来源改动可改变 consumer 的有效设计，却不增长 consumer Version。ResolvedTask.source_manifest 包括 root、自身、三区实际链上的中间节点和 provider；origins 保留各区 provider/chain。证据：schema_docs/0.common.md:43、:211；schema_template/tools/design_resolution.py:39、:242。

更细的接口事实：**effective_identity 是 node + source_manifest + origins 的哈希，不含规则版本、dispatch、workload 或运行环境；它也不是仅对 Design 语义取哈希。** 任一来源文件的原始字节改变，包括反馈、元数据或排版，都可能改变它。新旧 episode 的设计是否相同，不能只看 Version，也不能无解释地把每次 effective_identity 变化都算一次 Planner 优化。证据：schema_template/tools/design_resolution.py:47；schema_template/README.md:34、:64。

库调用还有一个具体区别：非 root 的 effective_task[node] 仍不本地保存 Contract；需要自包含输入时，应使用 effective_document(node) 或 effective_yaml，而非直接拿该映射交给 coder。证据：schema_template/tools/design_resolution.py:53、:259。

**建议：**后续架构沿用这些来源标识，并把规则版本、任务选择/范围、实现版本、评测口径与 episode 身份放在外围交接或运行记录中研究；不要先往 YAML 新增进度、预算或实验状态字段。现行文档明确要求 workload、目标与计时范围留在任务材料/交接说明。证据：DESIGN_SCHEMA.md:13、IMPLEMENT_SCHEMA.md:74。

### 3.3 角色权限和验收

**已核实事实：**任务方确认 Contract；Planner 修改 Design 与 REALIZATION.Require/Prefer/Forbid；Coder 仅能追加针对 Prefer、且有不可变真实实验证据的 Feedback。一般阻碍、Require 无法满足、待办与未实测建议不进入 Feedback。现有结构检查不能识别作者身份或判定反馈确有实测，三流程目前没有独立的角色差异门禁。证据：README.md:48、schema_docs/6.realization.md:9、:42、:69；schema_template/tools/rules/__init__.py:29。

Math_reshape 定计算/数值/逻辑寿命；Work_schedule 定工作/提交/跨 step 交接；Tile_pipeline 定 role、动作完成、依赖和复用。Coder 的“更快/更准”不自动授权改变这些选择。图中的 done 是完成保证，不是时间戳；Prepared 与 Released 分别回答可读与可覆盖，不能用“图无环”代替代码符合性。证据：schema_docs/3.design.math_reshape.md:7、schema_docs/4.design.work_schedule.md:9、schema_docs/5.design.tile_pipeline.md:206、:398、schema_docs/6.realization.md:142。

**关键前置差异：**普通设计/优化/实现需要可用硬件资料、明确接口与完整验收；extract 则强制除 Execution 最小登记外保留 starter 硬件占位，全部检查走 skip-hardware，并允许 Tolerance 留“Source only…no defaults”来源记录。要从 extract 进入实现，必须识别这些缺口，不能把本次提取豁免沿用为实现成功条件。证据：schema_extract/tools/artifacts.py:565、:715；schema_docs/1.contract.algorithm_requirement.md:254；schema_docs/2.contract.hardware_specification.md:123。

正确性、设计符合性、性能是三条证据链。QSA 是现成反例：正式 Schema 采用原验收器“FP64、以实现值为相对基准”的判据，20261021 快照 harness 却采用另一个 FP32/参考基准口径。一个通用 allclose 或现成 harness 名称不足以证明本次任务验收一致。证据：IMPLEMENT_SCHEMA.md:46、schema_template/examples/qsa_prefill/README.md:21。

## 4. extract 可借鉴与不可直接照搬的部分

### 4.1 已存在的可靠机制

| 机制 | 已核实实现证据 | 对其他流程的潜在价值与边界 |
| --- | --- | --- |
| 明确来源角色与冻结字节 | inputs.py:142、:186；snapshots.py:124 | entrypoint/host/io/input/reference/acceptance/scope 不靠文件名猜；脏工作树按实际字节冻结。可借鉴输入身份，但 design 不应被迫提供“待提取实现”。 |
| 冻结规则与执行器 | snapshots.py:23、:277；runtime.py:102 | 固定代码、规则和 prompt，恢复回到原执行器；任务/来源/语义规则改变另建 run。不能把不同设计 revision 硬装进一次冻结 extract run。 |
| 固定 Markdown + 动态事实 | prompts.py:35、:248；stages.py:14 | 提示、角色、材料、反馈分开，实际 stdin 留档，前阶段会话不是隐含输入。当前冻结清单未显式纳入 DESIGN_SCHEMA.md、OPTIMIZE_SCHEMA.md、IMPLEMENT_SCHEMA.md，不能直接改命令名就用于三流程。 |
| 可注入的模型传输 | agent.py:209；runner.py:1244；review_eval.py:186 | AgentCLI.run 接受 attempt/prompt/schema；review-eval 已在 P1–P6 以外使用它。存在小块复用基础，但仍是 extract 内部模块和 worker/reviewer 角色。 |
| 精确 scope/session/attempt | stages.py:137；agent.py:27、:50 | 分阶段、节点、角色与操作建立会话；同 scope 才精确 UUID resume。可借鉴身份隔离，不能直接等同 AKA coder episode 或 planner revision。 |
| 原稿→规范化候选→权限/严格检查→独审 | artifacts.py:884；runner.py:191 | 原稿证据不被自动修正覆盖，normalization 不是语义批准；continue 可保存合法草稿而不冒充完整通过。适合借鉴交付事务边界。 |
| 实际内容依赖与写作依赖分开 | stages.py:248、:265；artifacts.py:739 | wait_for 只管首派准入，from/origins 管内容来源与失效；不把“先写过谁”当设计继承。可供多节点和增量复核使用。 |
| 单写者、恢复与结算 | store.py:114；common.py:15；agent.py:155；recovery.py:89 | 成功必须真实 exit、唯一原生完成、session、schema/nonce 都成立；未确认后代结清保持 fence。不能把日志沉默、文本“完成”或主 PID 消失当成功。 |
| 原子非覆盖发布 | publication.py:41、:98 | ready 与 published 分开；相同 manifest 幂等，冲突保留，批准后不隐式改 raw。是输出事务范例，不是通用项目发布系统。 |
| 统计与可审计记录 | backends.py:340；tools/README.md:106 | requested/reported 分开，增量/累计、缓存与未知值区别对待；失败和恢复不抹账。可以借鉴字段含义，无须复制全部维护平台。 |

本表省略共同前缀：Python 文件位于 schema_extract/tools/；tools/README.md 指 schema_extract/tools/README.md。

### 4.2 不能原样套到 design/optimize/implement

- **六阶段是提取方法，不是四流程公共状态机。** P1 建立固定来源 Contract、P2 机制选型、P4 写作安排、P5 分节点审来源、P6 全算子终审，有明确源码忠实性要求。普通 Planner 文档要求逐层自检，并未要求每层照搬 Extract 的固定源码调查与独立审批。证据：EXTRACT_SCHEMA.md:9；DESIGN_SCHEMA.md:7；schema_discussion/dispatch_method/design_extract_workflow/README.md:13。
- **写作权限与 workspace 格式不适合直接实现 kernel。** artifact_inventory 只接受直属 root.yaml、其他 .yaml、dispatch.md，拒绝子目录和代码文件；Claude 的 Bash 只许可固定检查 helper，网络/委派工具禁用。若复用传输，需要清楚地区分代码工作区和 Schema 保护，而非假设现成 worker 就是 coder。证据：schema_extract/tools/artifacts.py:224、schema_extract/tools/backends.py:171。
- **现行预算不是时间/Token 限额。** 默认每阶段或 P5 节点 32 次调用、5 次实质退修，并预留裁决/签收；无单次、阶段、整算子时间上限，耗时与累计 token 是统计。implement 的 episode、GPU 等待和性能停滞政策尚未实现。证据：schema_extract/README.md:68、:84；schema_extract/tools/stages.py:81。
- **现行恢复拒绝旧协议，并依赖冻结绝对路径与原身份。** runtime 校验明确拒绝 semantic_revision/continuation/imported_draft 等旧字段；不能把历史恢复脚本或归档 run 当当前接入 API，也不能假设任意目录搬家后仍可 resume。证据：schema_extract/tools/runtime.py:75、:110；schema_extract/tools/cli.py:222。
- **standalone 当前也保留提取政策。** create_view 固定使用 skip_hardware=True，只校验/导出，不调用 coder；其 .source.json 记录 Template 代码身份与源目录来源，不能将名称误读成独立实现后端。证据：schema_extract/tools/standalone.py:43、:62。
- **CLI 权限不是宿主隔离。** 当前已移除自建 namespace/sandbox，保留真实路径、私有 CLI home 和原生角色权限；文档明确不保证宿主文件系统不可写。Coder/GPU/远端工具授权边界须由使用方明确承担。证据：schema_extract/tools/README.md:86；schema_discussion/extract_execution_20261117/isolation_removal/REVIEW.md:25。
- **模型默认值和原生事件契约是环境相关依赖。** 当前默认 Claude Code + qwen3.8-max[1m]/max，Codex 需显式指定；代码依赖这些 CLI 的参数与事件字段。本轮没调用真实 CLI，因此外部公开环境的可用模型、参数兼容与恢复覆盖仍未知。证据：schema_extract/tools/runner.py:39、schema_extract/tools/backends.py:182、:235、:340。

## 5. 当前示例、Wiki 与历史的可用范围

**已核实事实：**

- Template 当前有 **4 个案例、9 份 YAML、9 张 PNG**：FA4 prefill 3、decode 4、GDN 1、QSA 1；只两份非 root 整区引用 root 的数学区。可演示本地完整设计、整区共享、dispatch 笔记、多 kernel/多 worker 和局部事件图。来源为独立 20261019 四题快照，不能与 44 题主批次混用实现或验收。证据：schema_template/examples/README.md:5、:12、:16。
- kernel_wiki 主批次 **44 个算子，20 AKA + 24 参考候选**，有 44×7 题目文件、源码、采样与性能证据、机器索引和检索工具。存在 3 题最终回执缺项、2 题人工性能例外，两个主要计时口径也不同；这是一批可选择的有证据资产，不是统一 SOTA 或跨题排行榜。证据：kernel_wiki/README.md:3；kernel_wiki/EVIDENCE.md:9、:19、:29、:38。
- schema_wiki 导航登记的是 **6 个已审修算子、15 份当前 YAML**。首次盘点另有 **3 个未提交的新 Runner root 交付**（masked FP8×MXFP4 GEMM、MLA NVFP4、BF16 MoE），当时为 **9 个目录、18 份 YAML**；报告收尾时，原有并行提取又发布 GDN conv/MTP 的 3 份设计，实际已为 **10 个目录、21 份 YAML**。这些是同时进行的既有提取成果，全部未修改；未见输出不等同任务失败，批次状态需由负责其运行的 Supervisor 核实，本报告不维护实时提取看板。证据：schema_wiki/README.md:3；schema_discussion/extract_twelve_20261119/README.md:7、:21；输出目录枚举及 schema_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_conv_mtp_bf16_state_bf16/.extraction/receipt.json:13、:177。
- 六份 curated 目录采用 .extraction/current.json。它们的验收基础已是直接源码审修/只读检查，明确 runner_p6_receipt=false，并保留历史 receipt 指针。新的 Runner 输出则是 schema_extract.delivery.v2 receipt。**未来读取设计不能要求所有合法输入都带“当前 extract run P6 receipt”。** 证据：schema_wiki/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/.extraction/current.json:16、:44；schema_wiki/aka-cuda-operators-20261030/operators/mla/mla_nvfp4/.extraction/receipt.json:2、:88。
- 仓库确有历史 GPU 测试资产，例如 kernel_wiki/aka-four-problems-20261019/validation-20261021/bundles/gdn_prefill/harness.py:44、:111；它们属于输入快照/复现材料，没有成为 KernelSchema 活动 implement 入口，且前述 QSA 判据差异说明不能直接推广。
- docs 定义 Wiki.Hardware/Programming/Microbench/Sota/Profile 的绑定，但本轮未发现一套已经就绪、可替所有独立 design/implement 提供保证的通用资料库；示例仍保留 Wiki 尖括号占位。近邻参考可先使用 catalog 和固定源码/Schema，完整硬件与实现栈资料的来源仍要核实。证据：schema_docs/2.contract.hardware_specification.md:62、:113、:123；schema_template/examples/README.md:16。

**历史辨别：**

- role_prompt_trial 是角色文字试走，明确未完整交付设计、未运行 checker/GPU；其“工具尚未适配”是当时前提。concrete_design 与 design_extract_workflow 里的 schema_extract/phases、EXECUTOR_ALIGNMENT 等旧路径和未接通描述，不是当前接口。证据：schema_discussion/role_prompt_trial/README.md:14、:22、:49；schema_discussion/dispatch_method/concrete_design/README.md:42。
- 20261117 的真实 FA4/QSA 全流程验收停止在 P6 之前，不能称该次验收通过；但 **20261118 六算子记录已有五项正常 P6/Runner 发布**，之后又有人工审修及本轮新产物。不能将早期“没有真实全链通过”照搬为仓库现在完全没有实跑证据，也不能把旧批准迁给审修后的新字节。证据：schema_discussion/extract_live_acceptance_20261117/README.md:3；schema_discussion/extract_six_full_20261118/README.md:3、:24；schema_discussion/schema_audit_and_cleanup_20261119/README.md:16。
- schema_discussion/archive 下保留旧规范、运行、版本与计划；正式 issue 决定台账 schema_extract/issues/archive 仍被当前工具读取，不能因为同名 archive 就一并视作废弃。证据：schema_discussion/archive/README.md:3；schema_extract/tools/cli.py:26。

## 6. 后续架构调研的精确交接

### 给“独立四流程边界与契约”子代理

1. 从三份角色流程与 Template 只读 API 出发，研究最小外部任务封装如何明确已确认 Contract、规则身份、选定设计/实现范围、workload、优化目标、参考/验收与预算；不要把 workload、episode 等塞进当前固定 YAML 结构。优先阅读 schema_template/tools/README.md:31、DESIGN_SCHEMA.md:9、OPTIMIZE_SCHEMA.md:7、IMPLEMENT_SCHEMA.md:7。
2. 分别定义“有合法草稿”“完整设计可交 Coder”“实现已连通”“数值通过”“设计符合”“性能已测”的最小可观察结果，避免一个 passed/ready 承担所有含义；普通 Planner 不能直接拿 extract 的 skip-hardware 门禁作最终验收。
3. 明确 root→必要 dispatch 的选择由谁作出、怎样防止把局部目录笔记自动变成实现变更，以及任务范围尚未覆盖时如何交接。现行 when/dispatch 是文本语义，不提供可执行选路器。
4. 将复用候选分成“可以直接调用的 Template API”“需轻度适配的 AgentCLI/prompt/事务模式”“仅借鉴的 extract 状态/维护政策”。尤其注意独立命名空间、规则冻结清单和代码工作区权限，避免为了三条轻量能力抽出整套 extract 平台。

### 给“AKA 接入、轮次与状态”子代理

1. 研究 coder 输入应怎样携带固定规则、raw 任务、选定节点/范围、来源闭包与有效阅读视图；effective_identity 不含规则/dispatch/任务目标，且会随反馈等 raw 字节变化。不能用一个裸 YAML 路径或 Version 代替版本身份。
2. 明确同一设计跨多个 coder episode 的续跑，与 Planner 优化后的新 revision 是两条轴；应按什么证据判断“实现未完成”“实现缺陷”“设计需要改变”“预算停止”，目前 KernelSchema 只有文档分工，没有可以直接接上的状态协议。重点对照 OPTIMIZE_SCHEMA.md:15 与 IMPLEMENT_SCHEMA.md:56。
3. coder prompt/工具应遵守 Contract/Design 保护与 Require/Prefer/Forbid 的边界，并能查正文规则；不能直接给 AKA 使用 extract worker prompt。代码产物、编译/运行日志、GPU 实验结果不适合放入 extract 的 YAML-only workspace。
4. 研究如何分别接收一般阻碍和有实验支持的 Prefer 替代：后者才是 REALIZATION.Feedback，其余应走外部结果。评测反馈需绑定代码、完整 Schema 身份、输入/环境与计时范围；QSA 两套比较器口径是首批适配必须防止的具体反例。
5. generic implement 必须能消费普通手写/设计产物和 curated current.json 集合，不应依赖 .schema_extract 私有运行态或某次旧 receipt。依赖方向、AKA API 和 Agate 细节以同轮 AKA 调研为准，本报告未替其作假设。

### 仍需进一步核实的问题

- **资料与验收：**第一条独立 design→implement demo 用哪份已确认的完整硬件/编程资料、golden reference、比较器与 workload？现有复杂案例包含真实源限制，不能只挑一个能过 checker 的 YAML。
- **实现后端：**当前 AKA 能否绑定上述身份、隔离 Schema 写权限、继续同设计的多个 episode、回传构建/数值/符合性/性能证据？需要 AKA 子代理核实真实 API 与 prompt，不能从本仓库“Coder 如 AKA”推断已集成。
- **公开可用性：**Python/CLI 版本、模型选择、依赖安装、仓库路径与资料解析的最小支持范围是什么？当前是源码树内调用，尚无统一包/稳定集成版本契约。脏源码提取还会生成只在快照 source.git 中存在的 private_snapshot_commit，而正常发布只复制批准产物与 receipt；“已固定来源”不自动等于第三方可取得该来源。需研究公开交付时如何定位必要材料，本轮不实施打包或许可证改动。证据：schema_extract/tools/snapshots.py:48、:97；schema_extract/tools/publication.py:98。
- **语言既有边界：**历史 FIELD_ISSUES 的 F1（动态设备规模来源）与 F2（host 观察后条件提交）未见明确闭合结论。当前 DOC4 仍要求设备规模资料来源、每 step 一次 launch；DOC5 仍只支持 for/if、不支持 while。应核具体目标是否真的需要这些能力，不能把“实现所有 dispatch”暗中扩大为修改语言。证据：schema_discussion/role_prompt_trial/FIELD_ISSUES.md:3、:24、:66；schema_docs/4.design.work_schedule.md:9、:248；schema_docs/5.design.tile_pipeline.md:238。
- **批准与执行证据的迁移：**输入可能是 raw 手稿、Template 案例、人工审修 current manifest 或新 Runner receipt；需要什么最低验收，怎样保存已有 provenance 而不伪造新批准，尚无跨四流程的统一答案。

**简短建议（尚非决策）：**先围绕现有 Template 与角色规则研究三个最小可调用能力，保留 extract 已有流程。优先定清输入身份、选定范围和三类实现验收，再决定需共享多少运行设施；演示选择应以资料与验收闭合为依据，不以现有 YAML 规模或图是否漂亮为依据。

## 7. 本轮实际验证与未验证范围

本轮只执行了不会写回的命令；未运行离线测试套件（其临时文件不是本轮必要产物），没有把历史 suite 通过数冒称本轮结果。

| 检查 | 本轮结果 |
| --- | --- |
| 默认 python 的 Template 只读检查 | 退出码 2，/usr/local/bin/python 缺 PyYAML；未安装或修改环境。 |
| 使用现有 /opt/venv/python3.13.12/bin/python3，对四个 Template root 逐一 --read-only | 四目录、全部 9 份 YAML 均退出 0；规范化预览均为 0 would change，dispatch 检查通过。 |
| MLA FP8 curated root --read-only --skip-hardware | 退出 0，0 would change；明确跳过硬件。 |
| FA4 decode q1_global --show-effective | 退出 0、validation:not_run；Math 来源链为 q1_global→root，另两区本地，source_files 包括两份 raw。 |
| GDN root --read-only --graph check --set 'chunk_count(0)=2' --worktiles 1 | 退出 0；INDEX_CAST 1 动作/1 边、GDN 68 动作/242 边/19 次声明边界跳过，所给样本无环。 |
| extract_yaml.py --help | 退出 0，真实 parser 展示 extract/恢复/观察/standalone/review-eval 等命令，无 design/optimize/implement 命令。 |
| catalog.py list --family mla --json | 退出 0，得到 4 个明确 MLA ID；未导入 CUDA。 |

这些运行只核实现入口与所述静态能力。真实模型兼容、当前全套恢复/发布路径、GPU 正确性和性能、本轮新交付全部历史证据均未重演；公开 receipt 与历史报告按其实际覆盖范围引用。
