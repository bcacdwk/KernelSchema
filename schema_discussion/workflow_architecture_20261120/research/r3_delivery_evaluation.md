# R3：公开交付、三仓维护与实验验收交叉审查

日期：2026-11-20。性质：第三轮独立研究，供初步大纲与用户方向决策使用。
本文只新增本文件；未改实现、既有 Schema、运行记录或工作树状态，未运行模型/GPU/实验/测试，未安装、提交、操作远端或读取凭证。
已读用户简报、历史、两份 R1、两份 R2，以及相关安装说明、来源投影、交付代码和实验配置。
标记含义：**事实**为已读材料可支持的现状；**审查结论**为本轮判断；**候选建议**尚未批准；**用户选择**保留产品取舍；**技术 spike**须在后续获准实施阶段验证。
路径前缀：KS=/home/shaohanyong.shy/KernelSchema；AKA=/home/shaohanyong.shy/atrex-kernel-agent；EXP=/home/shaohanyong.shy/AKA_experiment。
R1S、R1A、R2S、R2A 分别指本讨论目录 research/ 下的 r1_schema_inventory.md、r1_aka_inventory.md、r2_standalone_contracts.md、r2_aka_lifecycle.md。
以下路径行号是本轮所读工作树的定位；既有并行工作可能继续变化，初始脏文件数量不是实时看板。

## 1. 审查结论：保留方向，收紧首发与论证

**审查结论：**“KS 薄独立实现 + AKA 增强后端”符合用户公开产品诉求，但只能作为分期方向；现在不应承诺两套完整执行系统同时交付。
**审查结论：**可选 Workflow provider 比静态 SchemaAgent 更适合拥有动态 revision，但单 episode 驱动只是接缝证据，不能据此估成“配置即可完成”的低成本集成。
**审查结论：**首个工程切片、首个公开 demo、首次科研收益结论应是三个门槛；把它们合成一个“端到端成功”会掩盖问题。

| R2 建议 | 本轮审查与所需修改 |
| --- | --- |
| C：公共契约 + 参考后端 + AKA 后端（R2S:10、:192） | 保留分期；先形成一个能运行的窄后端，再以 AKA 映射检验契约。删去任何首期双后端功能对齐的暗示。 |
| design→standalone implement 为第一路径（R2S:252） | 适合作为首个公开展示；工程上先固定一份审查过的 R0 打通实现/续写，以便区分 planner 缺陷与 coder 缺陷。 |
| 最后才放公开 demo（R2A:237） | 若指全部项目则应改序；独立 demo 不必等待 AKA 动态循环、多 provider 或复杂 fidelity。 |
| 最后才验证 extract 输入（R2S:258） | extract 内部不重做，但真实既有 Schema 的兼容与材料缺口应提前探测；否则 toy 接口可能长期偏离现有资产。 |
| framework_baseline=never 等 on-profile（R2A:142） | 保留候选；它是行为选择且影响实验对照。先验证首发任务可用，不能直接拿当前 V1-always pilot 当等条件基线。 |
| 多层完整身份与恢复记录（R2A:75） | AKA 动态路径需要这些语义；KS 参考后端先存请求、代码、证据与累计消耗，不复制 W/R/A/G/native 全部目录和状态。 |
| fidelity 作为独立门（R2S:140、R2A:172） | 保留；首发选有限可审查承诺与独立审阅，删去“通用自动等价证明”为发布前提的可能解读。 |

**候选建议：**立项承诺是“支持明确范围内的真实独立能力，并允许 AKA 增强”，不是“实现一个新的通用 kernel agent 平台”。
**用户选择：**若用户实际上只要独立前端而允许强制 AKA 后端，B 仍是合理方案；不能用名称上的 standalone 掩盖该取舍。

## 2. 下载 KernelSchema 的用户，当前与首发分别能做什么

**事实：**README 的四个入口链接到角色方法文档；可运行的提取入口是 schema_extract/extract_yaml.py。三项新能力尚无对应执行入口。KS:README.md:13、:24；R1S:19。
**事实：**Template 需要 Python 3.10+、PyYAML，PNG 另需 Graphviz；其依赖文件目前只列 PyYAML。KS:schema_template/tools/README.md:7；schema_template/tools/requirements.txt:1。
**事实：**Extract 声明 Linux、Python 3.11+、PyYAML、jsonschema、Git 和已认证 CLI；默认 Claude Code / qwen3.8-max[1m]，本身不运行 GPU。KS:schema_extract/README.md:5、:9、:20。
**事实：**这些默认模型名来自现有环境，不能证明任意公开用户可以取得同一服务或上下文容量。KS:schema_extract/README.md:76；R1S:89。
**事实：**当前无统一 GPU Harness，也无统一根安装包；默认解释器缺 PyYAML 曾使 R1 只读检查失败，换既有 venv 才通过。R1S:30、:141、:142。

| 使用条件 | 当前真实能力 | 候选首发对外承诺 |
| --- | --- | --- |
| 只下载仓库、无模型/GPU | 阅读方法、案例、Wiki 与历史证据 | 提供完整小任务材料、示例产物、结果解释；回放必须明示没有新生成/测量。 |
| 配好 CPU Python 依赖 | 结构检查、effective 视图、catalog；可选绘图 | 一条可重复的检查/材料准备路径，明确缺项，不要求安装 AKA。 |
| 有支持的模型运行时、无 GPU | 现有 extract；design/implement 目前仍须外部人工或 agent 执行方法 | 真正可调用 design/optimize；implement 可交代码与未执行项，运行验收保持 not_run/partial。 |
| 再有明确支持的 GPU/栈与任务 evaluator | 现有材料尚不能自动构成通用实现 demo | 独立完成一个固定设计的生成、构建、验证、测量及代码续写，并返回证据。 |
| 另安装适配 AKA | 当前未交付对应通用 backend | 后续消费同一固定实现请求，增加长程 episode、测量事务与原生恢复。 |

**审查结论：**首发独立的实质是第四行存在一条不需要 AKA 的真实路径；GPU、模型和任务 harness 是明确条件，并非必须由 KS 免费提供。
**候选建议：**首发统一声明一个经过验证的 Linux/Python 组合、一个模型 CLI/provider、一个 GPU 实现栈和一种 evaluator 适配方式；其他组合标为未验证。
模型选择要求公开用户可自行取得兼容运行时与服务，认证由用户自己的环境提供；不把当前私有路由、1M 上下文或服务器别名当可移植依赖。
发布记录固定模型“请求值/报告值”、CLI 版本与必要参数；无法固定服务端模型权重时，承诺记录条件与结果，不能承诺逐字生成相同代码。
安装说明应分开 CPU 材料工具、模型调用、GPU 执行三步，使无 GPU 的用户仍能得到可用的检查与阅读成果。
一种任务提供的本地 harness 足够首发；Agate 接口可以后续补。若先支持 Agate，必须写明账号、SDK、GPU/镜像和任务 evaluator 条件，不能称下载即能跑。
Agate 只解决执行通道，不自动解决 ABI、reference、完整误差公式、输出副作用或计时范围；后者始终属于 Task。
不需要首发就支持任意用户脚本或自动猜测 evaluator；先支持一个固定输入/结果协议及一个公开样例，扩展任务由显式适配承担。

### 2.1 可取得材料比“已有 SHA”更严格

**事实：**脏来源提取会在 run 快照内部创建独立 source.git，标记 private_snapshot_commit；发布只复制批准工件和 receipt。KS:schema_extract/tools/snapshots.py:48、:97；publication.py:98。
**审查结论：**回执里的 Git SHA 和仓库名不能单独证明第三方能取得那些源码；一个可追溯的本地提取不自动是可公开重跑的 demo。
**候选建议：**首发小任务应随选定发行物提供 Task、参考/比较器、输入生成、计时入口、初始骨架、资料索引和示例 Schema；必要字节给实际位置与摘要。
外部资料固定 URL/版本或可分发快照，并记录取得方式；代码和 Schema 引用不得只落到作者 home、临时目录或私有 snapshot repository。
若某份资料只可阅读不可随包分发，明确额外取得步骤；选中 demo 的必要材料必须形成可复现闭包，不要求先清理整个 Wiki。
本轮活动目录文件名枚举未发现 KS 根 LICENSE/统一 pyproject；公开发行时应明确所选代码/数据的许可与来源，本文不自行决定许可，也不推定整库均可再分发。
真实示例产物可以随包发布；模型 Home、原始会话、个人配置和整个实验工作区没有成为公开 demo 的必要性。

### 2.2 自有 loop 的责任上限

**候选建议：**参考后端只拥有单任务、固定 R、串行调用、受限验证、代码快照、累计预算和结构化返回；模型的原生工具循环交给已支持运行时。
它应保证从准确 C1 开新会话仍能继续 C1；无需首发保证跨机器恢复模型隐藏上下文，或承诺所有 CLI checkpoint 可互换。
同一 R 下编译失败、缺 dispatch 或预算到期先返回进度/诊断；implement 本身不自主修改设计，不自动调用无限 optimize。
一次运行只允许一个外层 loop 所有者：KS demo 调固定 implement backend，或 AKA SchemaWorkflow 调 KS planner；两者不得相互启动顶层循环。
最小持久化是输入/产物摘要、已结算用量、最后稳定代码、调用状态与结果；不要先造数据库、多租户调度、远端作业平台或统一事件总线。
运行时若有未结算作业，先交执行适配器恢复/结算；参考后端无法确认时返回 pending/blocked，不能重提并把账目归零。
**收缩条件：**若首个小任务已要求复制 AKA 的隔离、ABBA、分布式恢复或复杂 candidate search，应缩小支持环境；用户也可改选 AKA 依赖型产品。
“薄”应按新增职责和长期维护面判断，不能仅靠初版代码行数命名。

## 3. 三仓所有权与迁移的真实成本

**事实：**EXP 当前为 assets/aka 普通文件 vendor + overlay，基线 bf5557a，与已读独立 AKA HEAD 相同；没有 AKA fork→KS submodule 的现成消费关系。R1A:11、:29。
**事实：**初始 EXP 有 76 修改、4767 删除、84 untracked，KS 也有既有变更；这些不由本任务产生。KS:schema_discussion/workflow_architecture_20261120/01_history.md:8、:17。
**事实：**R1 vendor verify 被生成缓存拒绝，辅助比对另发现两项未登记源码差异；该辅助比对不是官方 verify 通过，也不是本轮重测结果。R1A:43。
**事实：**EXP 的 launcher/runner、解释器选择、metadata 构建、资源身份及冻结安装共同依赖 canonical source；替换 assets/aka 不等于完成迁移。EXP:docs/aka-integration.md:26、:38、:48。

| 仓库所有者 | 候选长期职责 | 迁出/避免的重复职责 |
| --- | --- | --- |
| KernelSchema | 语言/Template、四能力的交接语义、planner 写作、参考 implement、有限 demo 与公开资料 | 不拥有 AKA native state/promotion，也不维护 EXP 批次策略。 |
| AKA 适配 fork | 通用 role/context/seed/资格接口，可选 SchemaWorkflow，native 执行/恢复与输出翻译 | 不拷贝一份 KS parser/planner；实验 condition 和 cohort 不进入 core。 |
| AKA_experiment | 固定版本消费、任务/矩阵、预算与成本比较、资格校准、实验产物及历史 | 不再拥有第二份通用 planner；过渡 overlay 保留来源映射并逐步归位。 |

候选源码依赖图：
~~~text
EXP 的一次固定实验捕获
  → 固定 AKA fork 提交
      → 可选 KS submodule 固定提交
KS 核心/参考后端 → Python、模型运行时、Task evaluator
KS 选择 AKA backend → 外部已安装 adapter（固定实现请求）
~~~
**审查结论：**上图最后一条是可选运行调用；KS 基础 import/安装不得反向依赖 AKA，也不能把两个仓库互相设为 submodule。
AKA off 应能在 KS 子模块未初始化时工作；submodule 固定源码，不自动解决包命名、资源定位和安装依赖。R2S:244、:246。
目录名与公共 Python 包名可以后定；先禁止依赖 cwd 和两个 tools 包互相遮蔽，避免把私有 dataclass 暴露成 KS 公共接口。

### 3.1 保护现有成果的候选迁移序列

1. 在用户确认方向后的实施阶段，从明确提交建立独立工作树/检查副本；现有活动工作树继续保留，不以 reset/clean/stash 或批量 add 为迁移步骤。
2. 先核对现有差异归属：有效未提交改动、预期删除、未跟踪新增、运行缓存分别记录；旧删除不能因“还原干净基线”被自动撤销。
3. 从同一 AKA baseline 提取经审查的通用补丁和 schema adapter，建立旧 overlay id/原字节/新文件或提交的映射；不把 EXP root 提交直接套到 AKA root。
4. 第一轮只迁移所有权与入口，尽量保持 AKA 基线不变；升级上游和变更 workflow 策略分开，使失败能定位到一次变化。
5. 用一个新实验捕获消费新依赖，核对加载来源、资源、数据/evaluator、预算和导出身份；活动旧 run 继续用原捕获恢复。
6. 新路径通过必要验收后，才停止在 EXP 继续开发重复产品逻辑；旧台账、旧捕获、旧结果和 reader 仍可定位，不用新结果覆盖旧目录。

上述是阶段关系，不是本轮授权执行；不要求用户把全部既有工作立即整理成提交才可以讨论架构。
必要时冻结实际字节及摘要；只有 HEAD 不足以代表脏工作树，保存 patch 也不能遗漏 untracked 内容与有意删除。
迁移保留旧来源/构建身份与新消费身份的对照，旧结果只增加外部索引或新读取说明，不回写旧测量为新版本成果。
**事实：**EXP 已有 base/ours/theirs 审查、来源漂移拒绝和整版/feature 分离工具，明确不自动 fetch/切分支/提交。EXP:docs/aka-integration.md:50、:58、:65。
**候选建议：**保留其来源审计思想，不必把当前所有 overlay 工具永久带入 KS；产品代码归位不等于实验可复现台账失去价值。

### 3.2 vendor→消费与上游升级

**事实：**project() 遇到 mode=160000 会记录 excluded_gitlinks 并跳过，.gitmodules 也不投影。EXP:tools/lib/aka_vendor.py:93、:98。
因此将“带 KS gitlink 的 fork”交给当前 vendor 流程，得到的仍是不含 KS 的普通文件树；需要显式物化并验证 KS，或改用固定 checkout 消费。
**候选建议：**先允许 vendor 过渡，只从固定 fork 取产品代码、另按其 gitlink 物化 KS；长期可换 checkout/submodule，不能同时永久维护两份手改代码。
若直接切换 checkout，一次性代价还包括解释器/metadata、资源路径、快照捕获与来源校验的适配；不是一次 Git 命令。
版本记录至少包含 AKA upstream/fork SHA、KS SHA、实际代码/规则/prompt 摘要、选中资料、evaluator、模型 CLI 与执行环境；有 dirty 字节则另外声明。
每次上游升级在新捕获中审查通用补丁被吸收/冲突的状态，先过 off 与固定 implement 接缝，再做新任务资格确认；不重跑所有历史实验。
不承诺升级后二进制能恢复旧 frozen continuation；保留原工具链或显式新 run 迁入旧源码，预算/历史关联不得丢失。R1A:169；R2A:241。
**维护判断：**fork+submodule 减少复制漂移，却增加版本配对和安装验证；只有稳定小接口、固定发行配对与单一代码所有者才能抵消此成本。

## 4. 小任务与复杂已提取算子的取舍

**事实：**现有案例含真实来源和复杂约束，不代表材料已适合独立生成。GDN 单 root 已包含成对块内求逆、片上递推和跨工作接续。KS:schema_template/examples/gdn_prefill/README.md:3、:16。
GDN 仍记录多序列 TMA descriptor 顺序、缓存并发等限制，Wiki 占位未闭合；“一份 YAML”不能估作简单题。同文件:20。
FA4 decode root 的 when 仅覆盖 Q=4、B∈{15,16}、长 KV 等，而共享 Contract 更宽。KS:schema_template/examples/fa4_decode/root.yaml:8、:38。
QSA 的正式比较器采用 FP64、以实现值作相对误差基准，历史 harness 使用不同口径。KS:schema_template/examples/qsa_prefill/README.md:21。

**审查结论：**教学小任务最适合验证产品闭环与安装；真实复杂任务用于反证表达/实现能力。二者不可相互替代。
第一题应按材料、栈与验收闭合选择，不按 YAML 数量选择；复杂 extract 成果也不能免去任务准备和来源限制核对。
已有源实现的限制若与必须满足的 Task 冲突，应显式形成 planner 修订或缩小经授权范围，不能让 coder 偷改设计后仍声称忠实实现原 R。

### 4.1 首条切片候选

选择逐行 softmax/归约这类单 root、有限域、完整接口的小任务，先固定 Task、reference、比较器、计时入口、硬件资料和一个审查过的 R0。
从公开骨架开始，在不安装 AKA 的环境调用 implement；只提供 Task、规则、R0 与限定参考，不预置目标优化实现。
交付准确源码、build 日志、运行正确性、有限 fidelity 对应和实际性能记录；性能可以不优于参考，只要结论真实且符合完成政策。
给同一请求一个不足以完成全部工作的受控调用边界，保存 C1；下一调用从 C1 继续，并证明旧完成范围/已消费预算没有丢失。
“partial→continue”可先用伪后端验证，再在实际运行中保留真实发生的记录；不能把人为改写成功日志当续写证据。
这证明独立 implement 的工程价值，不证明 Schema 相对自由编码提高成功率，也不证明复杂异步流水线 fidelity。
随后用同一 Task 实际调用 design，再 implement 其 R0；再调用一次 optimize，允许 keep 或 revise，按返回决定继续，不能强造速度提升。
首个四入口展示可另附一个小来源的现行 extract 调用说明/已有合格交付；extract 结果转实现必须补齐硬件与验收，不重设计提取内部。
公开 demo 必须展示真实调用入口与失败返回；只放生成后的文件、录屏或“让 agent 读 IMPLEMENT_SCHEMA.md”不足以交付自动能力。

### 4.2 后续验收阶梯候选

| 阶梯 | 少量选择 | 能支持的结论 |
| --- | --- | --- |
| L0：材料/协议 | 小 Task、伪模型/评测、精确 C1 续接 | 状态与安装材料契约可审查；无 GPU/模型成功结论。 |
| L1：独立产品 | 小任务真实固定 implement，再接 design/一次 optimize | 声明环境内的独立能力与四流程交接可用；无普遍收益结论。 |
| L2：真实既有 Schema | 按材料资格选一项现有 GDN/MLA/GEMM，优先一个完整已提取设计 | 真实设计消费、隐含前提、实现成本与 fidelity 缺口；不能推广全部算子。 |
| L3：多节点/新 revision | 一个有必要 dispatch 的任务，旧源码续改并有限修订 | 范围覆盖、引用失效、revision 与 episode 分离，明确保留失败/unknown。 |
| L4：受控收益研究 | 预先选定少量代表性真实任务、独立重复、预算受控对照 | 固定 Schema、动态修订与信息/成本贡献各有范围明确的证据。 |

L2 的材料预审应与 L0/L1 同期开始，真实实现可在 L1 之后；不要等所有 AKA 动态功能完成才第一次读取真实 Task 缺口。
GDN 单 root 是候选而非已选定 demo；设备、现有 source 限制和 evaluator 是否闭合可能使其不适合首个真实题。
FA4 decode 与 QSA 可先作为范围/判据反例使用；不必为了测试这些已知边界先跑整项昂贵实现。
对复杂 fidelity 返回 unknown 是合法研究结果，但意味着相应“已按设计实现”验收尚未完成。

## 5. 控制实验：把三个问题拆开

**事实：**现有 SchemaPolicy 默认 neutral/空，SchemaAgent 投递全文材料并绑定静态 bundle，不执行动态 planner。EXP:tools/python/aka_experiment/schema.py:130、:141、:190；R1A:15。
**事实：**当前 matrix 仍 schema_enabled=false，各题标 runtime_not_validated；policy 指定 framework_baseline=always、L20D、Triton，formal_launch_enabled=false。EXP:control/experiment-matrix.json:4、:14；control/experiment-policy.json:1。
**审查结论：**现有静态适配和 CPU fixture 是可借资产；既不能算动态 planner 收益实验，也不能证明固定 Schema 完成度/fidelity 或 schema-off 原生等价。

### 5.1 核心对照与有限诊断组

| 条件候选 | 固定内容/处理 | 主要问题 |
| --- | --- | --- |
| B：无 Schema coder | Task、同一骨架、同一资料集合和执行策略；无额外 R0 | 基准实现能力及成本。 |
| F：固定 Schema implement | 相同 Task/资料，加冻结 R0 与固定设计实现政策；不许 optimize | 提供该设计及约束后的整体变化。 |
| D：动态 Schema | 与 F 共用同一 R0，允许受限 optimize；相同总资源上限 | 在既定起点上动态修订的净贡献。 |
| P：同设计普通文字 | 只在选定诊断题上给与 R0 决策对齐的普通文字计划 | F 的优势是否来自设计信息，而非结构化表达；转换须审查。 |
| C：原实现信息对照 | 只在 extract 诊断题上向基准提供同一被提取原源码，固定读/seed 方式 | Schema 压缩交付与直接提供源码的差异。 |

核心先 B/F/D；P/C 只针对要解释的结论补充，不要求全任务×全资料×全预算的完整笛卡尔积。
F−B 衡量“设计材料 + 约束实现方式”的总效应，不能直接归因 YAML 语法或流水线表示。
D−F 在等总资源上限下衡量动态策略价值；若还要宣称超越一般两角色规划，需要额外同预算普通 planner 对照，未做就保留结论边界。
P 与 F 的事实对齐、长度和表达差异无法天然完全相等，应保留人工核对与 token 统计；不能把一份劣质摘要用作公正普通计划基线。
静态 SchemaAgent 可另作兼容诊断，但其自由优化 prompt 与固定实现政策不同，不宜把旧实验直接改名 F。
各组最终统一评定 Task 正确性/性能；F/D/P 另外报告所分配设计的符合性，B 没有固定 R，不能因无 fidelity 对象被判任务失败。

### 5.2 Token、资料和策略必须可比较

采用同一模型/CLI、相同 Task/evaluator/目标硬件、相同骨架、资料允许集合、执行与基线策略；顺序轮换或配对，记录实际服务差异。
R2 on-profile 关闭 V1/自动转换，而当前 pilot 开 V1；要隔离 Schema 效应，应让 B/F/D 共享这部分策略。保留原生 defaults 的比较只能报告系统整体差异。
主对照使用相同总资源上限，计入 planner、coder、fidelity/reviewer、失败/重试、缓存/输入输出 token、GPU 与真实 elapsed；未知用量不能填零。
F 可把同等总预算用于更多 coder 工作；“coder token 相同、D 额外免费 planner”只能回答追加投入的收益，必须另标，不能充当效率证据。
不靠无意义 padding 把 token 凑齐；比较质量/完成率随累计成本的结果，另外报告实际花费与未用余额。
R0 的 design/extract 成本在 F/D 间同口径：同时报告已有 Schema 的边际使用成本与连初始化的端到端成本，注明摊销次数。
相同 Wiki 可访问不等于实际输入完全相同；首轮固定选中资料包与版本，记录读取/检索结果及计费，开放网络的实用效果研究另报。
dynamic 不能独占新的原实现、profile 或私有 Wiki；若允许新资料，给对照相同访问机会并把信息变化列为处理的一部分。
预先定少量真实任务、预算和独立重复，保留全部启动样本/失败/超时，报告按题结果与测量波动；不从成功 demo 里挑赢家后宣称普遍收益。
任务数、重复次数与具体预算留到用户选择支持范围后决定；当前无新实测可估计提升或方差。

### 5.3 extract→implement 的原代码混淆

**审查结论：**extract 由原实现得到设计是任务本义；混淆发生在评估“只凭 Schema 能否重建”时 coder 又读取或直接继承原实现。
区分三种运行：原源码保持不变的 verify-only；原源码作为 seed 的适配/优化；从统一骨架、未暴露被提取实现的重建。三者分别记账与命名。
前两种有真实产品价值，但不能独立证明 Schema 足以传递实现设计；也不能把已有源码本来正确算成新生成成功。
严格重建条件下，语义 reference 可按统一规则提供；被提取的优化源码、annotated 副本、build 产物、旧 transcript/seed 不进入 coder 可见材料。
仅要求“不要看原代码”不够：资料 allowlist、文件挂载、检索/网络策略与新会话应限制实际访问，并保存投递/读取证据。
Schema 的来源引用可保留在宿主 provenance；若提供遮蔽定位信息的阅读视图，记录视图摘要与原 Schema 对应，不悄悄改写原工件或原 receipt。
公开仓中源代码仍可能被模型预训练见过，这类实验只能约束本次显式暴露；不宣称证明模型从未知道该实现。
C 组让相同原信息以源码形式出现，P 组让同一设计以普通文字出现；需要哪个对照取决于要主张“压缩价值”还是“可执行产品价值”。

## 6. 七项有限的必要验收

以下为候选发布/集成门禁，按所发布能力适用；不是本轮已执行的测试，也不要求每项展开成大规模测试平台。
共同锚点是 Task + R/范围 + 确切 C + evaluator/环境/资料身份；一次 GPU 结果不能自动证明另一代码或 revision。

| 必要承诺 | 最小正向证据与关键反例 | 足够范围/适用阶段 |
| --- | --- | --- |
| 运行正确性 | 固定比较器检查完整调用、全部输出/副作用及规定输入；错口径或缺输出不能通过 | 首发小任务及选中真实题；有限样本不叫全域证明。 |
| 语义 fidelity | 核对数值路径、工作组织、必要同步/寿命与 Require/Forbid，并有独立有限审阅；更快但换设计不得代签 | 逐项 pass/fail/unknown；复杂未知保留，不以 checker/GPU 通过替代。 |
| 性能 | 精确源码、同设备/输入/计时边界与参考的真实记录；正确但更慢仍返回实现，目标单报未达 | 首发无需胜过 SOTA；复用 QSA 判据反例防止错配。 |
| 代码续写 | C1 未完成且未晋升，下一同 R 调用确实继承全部文件/删除并继续；范围/预算不重置 | KS 可用新 session；AKA 必须额外证明跨 terminal episode。 |
| 失败恢复 | 在代码/结果交界处中断后保留同一身份与累计消耗；已完成作业不重复提交，未知作业先结算 | 本地参考后端覆盖自身持久化边界；AKA/Agate 再覆盖所承诺的远端边界。 |
| off 等价 | 同一固定 native 配置下，归一化调用轨迹/prompt/工具/预算/输出选择与上游一致；无 KS 仍可启动，新增 Schema 模型调用为零 | 伪 Agent/Executor 验工程合同；不承诺随机模型文本相同，EXP control 不能代替此验收。 |
| 公开可用性 | 干净目录按说明安装最小依赖、取得必要材料、执行真实入口并读懂失败结果；无 AKA 路径/import 依赖 | 声明一种环境即可；离线产物另标，不能由作者机器成功推定第三方可重跑。 |

复用有限反例而非无限列测试：C1 测后改 C2 防结果错绑；Schema 原文/hash 变化防旧身份续接；正确但违背设计防“速度即通过”。
故障模拟与伪后端用于事务/权限/预算，真实模型/GPU 用于能力与数值性能；两类证据分列，不能互相替代。
一般失败留在外部结果；只有符合现行 Prefer 替代规则且有合规不可变实测引用时才进入 REALIZATION.Feedback。R2S:153。

## 7. 顺序分歧、最小阶段依赖与待选项

**审查结论：**R2S 的 design-first 适合用户看到完整产品；R2A 的 fixed-first 更利于定位实现、续写和原生晋升冲突。两个结论针对不同门槛，可用分支路线兼容。
若先把动态 planner、复杂任务、AKA 生命周期和独立安装一起接通，失败无法归因；若永远只做 fixed toy，又无法兑现用户要求的设计产品。
**候选推荐：**工程先 fixed，公开展示尽早 design→standalone，一项真实既有 Schema 紧随；AKA 单 episode 适配独立验证，动态收益最后做受控比较。

~~~text
用户确认独立承诺/首发支持面
  → Task/材料闭包与最小请求/结果
      ├→ KS 固定 R 薄实现 + C1 续写 → design/一次 optimize → 独立公开 demo
      ├→ 真实既有 Schema 材料预审 ────────────────→ 真实固定实现/必要 dispatch
      └→ AKA 单 episode/seed/off 技术验证 → 固定实现 backend → 动态 provider
       独立/AKA 的固定实现证据 + 同一 R0/预算政策 → 受控 dynamic 收益研究
~~~
箭头表示必要前提；图不要求三线同时投入，也不把 AKA 动态能力设为独立 demo 前置。
extract 使用现有入口，产物经 Task 准备消费；不需要等待其内部重构，也不能跳过硬件/验收缺口。

**用户选择一：独立承诺与投入。**推荐“不依赖 AKA 的一个真实支持路径 + 后续 AKA 增强”；若只要求统一入口，可减少自有 loop，但需对外明确强制后端依赖。
**用户选择二：首发支持面与展示顺序。**推荐一个资料闭合的小任务/单栈先公开、一个复杂已提取题紧随；若复杂现有算子优先，应接受公开 demo 更晚与更高 fidelity 成本。
**用户选择三：维护路线与验收强度。**推荐 AKA fork→KS 可选 submodule、EXP 固定消费，允许有限独立审阅且 unknown 不冒充完成；更强自动 fidelity 是后续研究投入。
默认 V1 是否可停用、必须达到的性能目标与可投入模型/GPU预算随这些选择明确；接口/目录名和内部对象字段不必交用户逐项设计。

**技术 spike 一：公开闭包。**验证一个干净环境的包命名/资料定位/公开模型组合与 Task harness，确认没有隐含 AKA 或作者绝对路径。
**技术 spike 二：薄后端上限。**固定 R→partial C1→新会话 C2，核清代码/删除、累计消耗、必要只读约束和不支持时的返回，判断是否仍足够薄。
**技术 spike 三：AKA 接缝。**检验 single-episode 恢复、partial seed、比 incumbent 慢的完成产物、never-V1 收尾与 off；失败时再评估替代接入，不能用 search_driver 存在作通过证据。
**技术 spike 四：迁移闭包。**在独立副本验证 fork+KS 物化、source/metadata/resources 选中路径与旧捕获可读性；不在当前脏树上试升级。
**技术 spike 五：真实题资格与信息边界。**选一个现有复杂 Schema，核对 comparator/硬件/源限制及 coder 原码可见性；据此确定 L2 与实验对照，避免材料不足被误报为 Schema 失败。

上述 spike 均为用户方向确认后可安排的窄验证，不是本轮实施或实测承诺；本文保留替代方案与失败后的收缩条件，不锁定最终落地计划。
