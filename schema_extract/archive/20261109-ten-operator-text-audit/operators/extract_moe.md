# extract_moe 全量文字审计（v1.1）

本算子的现有设计**依赖开放正文确定核心行为**，但不能把所有正文都当成语言缺口。W/T/R 中有 600/992 个精确文本组含已确认的非公式核心信息；另有实现原语绑定的核心重要性未获独立证明。分块、预分配、launch/handoff、环槽与大量 reuse 边已有结构。主要剩余文字负责复合动作完成承诺、空动作、跨调用存储与主机服务；具体全设备 host 完成事件是本轮提出的结构缺口候选。

审计读完 2555/2555 精确文本组、106/106 创作 YAML 的有效组合及全部 dispatch；另读 5 份无创作结果骨架、列出 43 个未启动节点。没有抽样、未审默认标签或按关键词分类。人工分批记录在 [manual.json](../work/extract_moe/manual.json)，覆盖与 source 固定身份见 [coverage_complete.json](../work/extract_moe/coverage_complete.json) 和 [source_identity.json](../work/extract_moe/source_identity.json)。

## 材料与口径

固定范围为 pause inventory 当前计划：96 approved、9 review_candidate、1 needs_authoring，均为语义样本；其余 48 个不进入简洁/密度计算。106 个 meta_data.source 全部绑定 `SchemaExtractSource_c98aa54e58ac@ce1b7d5e8e7560b5c80abd87549bc2328c52df34` 的 `candidate/solution.py::Model.forward`，其 `_SRC` 指向同快照 `nvfp4_moe.cu`。源在 `.schema_extract/v2/snapshots/c496cab3…/source`，复制到 work 的 solution/CUDA/reference/metadata 字节均通过 inputs.lock alias 的 SHA256 核对。未使用当前树不匹配的源码行号。

已读冻结 README、DOC0、DOC1、DOC3–6、REVIEW 和 examples README。硬件区为明确的 skip-hardware 空值骨架（root:197–208），没有 hardware 开放文本组；它不提供完整硬件能力证明。approved 是冻结状态，不是本轮 GPU/数值或动态全域验证。

字符按统一空白归一化的 Python len 计；字符串总量含公式、引用和标识。英文 YAML 与中文 dispatch 分列。声明量每个源标量计一次，alias 实例展开量另列；from_parent 的有效视图不与 raw 相加。精确相同文字按清单去重读取，但另读 166 种局部结构、73 种 flow、289 种数学属性定义及全部使用节点。词级差分包可完整还原原组，不作为语义相等证明。

| 范围 | 精确组 | 去重字符 | 声明数 | 声明字符 | 含非公式核心组 | 保守短写候选字符（声明量） |
|---|---:|---:|---:|---:|---:|---:|
| YAML 全字段 | 2,381 | 1,029,922 | 13,693 | 3,103,483 | 851 | 75,898 |
| W/T/R 主研究范围 | 992 | 514,934 | 5,799 | 1,919,372 | 600 | 72,097 |
| math | 1,209 | 309,959 | 7,606 | 965,684 | 143 | 765 |
| dispatch.when | 106 | 196,613 | 106 | 196,613 | 106 | 111 |
| dispatch.md（中文） | 174 | 34,576 | 174 | 34,576 | 7 | 0 |
| hardware | 0 | 0 | 0 | 0 | 0 | 0 |

W/T/R 的 600 组对应 3642 次创作声明、1,550,479 字符的**字段面积**，不是不可删字符数；MIXED 中的重复片段仍可能可缩。全 YAML 含核心的非公式组为 851，dispatch.md 另有 7 组。93 组/494 次声明的核心重要性为 uncertain；UNRESOLVED 组、未决上下文和未决声明均为 0。这里的 0 只表示分类/绑定已完成，源缺陷和未知验收并未消失。

| raw/effective 量 | 全 106 节点 | root |
|---|---:|---:|
| 全部字符串声明字符 | 3,668,894 | 35,832 |
| 全部字符串 alias 实例字符 | 4,744,703 | 55,300 |
| 开放字段声明字符 | 3,103,483 | 30,245 |
| 开放字段 alias 实例字符 | 4,107,277 | 47,902 |
| 有效 Design 字符实例（独立口径） | 4,834,026 | 43,100 |

raw pipeline.verb 声明 1,158,927 字符、alias 实例 2,162,721 字符；相差 1,003,794 是使用位置展开，不能说成又写了一百万字。实际本地 provider 数为 math 86、W 94、T 93。root 有 184 个开放文本组，其中 W/T/R 48 组、26 组含非公式核心；大树 W/T/R 的 600 组不能用重复字符压倒这个 root 基线。`no_scan_fused_small_cp` 的 raw W/T/R 没有独占核心组，但有效视图引入其父的动作定义；这正说明不能按空 raw 区断言设计更简单。

## 全字段分类

| 类别 | 精确组 | 去重字符 | 声明数 | 声明字符面积 |
|---|---:|---:|---:|---:|
| DUPLICATE | 33 | 11,455 | 244 | 68,470 |
| NONCORE | 0 | 0 | 0 | 0 |
| STRUCTURABLE | 15 | 5,537 | 97 | 25,797 |
| ESSENTIAL_TEXT | 269 | 94,708 | 2,126 | 669,564 |
| FORMAL | 377 | 97,241 | 2,801 | 329,064 |
| SOURCE_LIMIT | 10 | 3,091 | 10 | 3,091 |
| EXPLANATION | 1,188 | 176,718 | 5,929 | 429,502 |
| MIXED | 663 | 675,748 | 2,660 | 1,612,571 |
| UNRESOLVED | 0 | 0 | 0 | 0 |

每组的 reason、原始行号、finding 和 occurrence 清单位于 [annotations/extract_moe.json](../annotations/extract_moe.json)。NONCORE 为 0 不表示所有源码细节都核心：布局/API等内容大多与必要事实混在同一字段，归 MIXED，或列为核心重要性 uncertain；没有把混合字段整体判为可删。

## 可复核判断

| finding | 涉及组/实际 YAML 节点 | 判断（集合可重叠，不能相加） |
|---|---:|---|
| extract_moe-F01 | 1/106 | 同名/describe的106次声明遵守共享任务身份要求，不是可任选删除的106份赘文。 |
| extract_moe-F02 | 1186/106 | 名称、释义、摘要、理由和中文索引主要支持阅读；字段面积不等于可删除字符。 |
| extract_moe-F03 | 377/91 | 公式、参数谓词、成本关系、step与ledger.by引用属于正式内容，不能算自然语言冗余或结构缺口。 |
| extract_moe-F04 | 2/1 | Contract中的调用边界/输出别名与相关输入域使用开放compute/meaning正文；不是公式字符串本身的漏洞。 |
| extract_moe-F05 | 113/106 | 106个when独立定义ABI缓存、已保存选择、实际提交和capture/replay作用域；没有继承父when或共享谓词库。 |
| extract_moe-F06 | 65/85 | 实际数值求值、主动近似、重复计算和padding规则有必要文字；不能据长度截断。 |
| extract_moe-F07 | 132/89 | 逻辑M/J视图与R容量、模块表/ABI存储、版本共址及尾部读写须区分；retention不等于物理分配。 |
| extract_moe-F08 | 335/93 | 复合动作的输入组成、done承诺、CTA协作、guard、空动作与资源退役是有意开放的动作语义。 |
| extract_moe-F09 | 152/92 | 许多长动作契约把必要完成语义与重复公式、shape/slot数、线程lowering放在一起，字段为MIXED。 |
| extract_moe-F10 | 80/106 | Realization固定tcgen05/TMA/cp.async/原子与可见性原语；这种绑定有意开放，核心重要性单列uncertain。 |
| extract_moe-F11 | 5/11 | ordinary-stream或verify-only路径中的inert依赖helper/trigger记录来源事实，不计为已证明的额外核心依赖。 |
| extract_moe-F12 | 9/87 | 9组fc2 strategy的独有BN2=128可以放进已有hidden.partition.tile，并用现有binding取用该字段。 |
| extract_moe-F13 | 6/10 | 6个自定义scan/check动作全部计算已有同名上游结果，可改用通用compute，避免为同一计算重写动作词定义。 |
| extract_moe-F14 | 200/94 | W中参数/分片/服务正文与重复phase概览混合；已有partition、worker_assignment、launch、handoff不能被摘要替代。 |
| extract_moe-F15 | 195/95 | 主机服务、probe读回/清理、缓存发布与存储owner通过开放正文联系GPU工作；这是相关段落集合，不表示其中每个API都已证明是一阶性能核心。具体全设备host等待的缺事件候选另见F25。 |
| extract_moe-F16 | 13/86 | 13组通用compute/produce契约没有独占核心；上游同名计算与当前worktile/resource已定义其全部设计含义。 |
| extract_moe-F17 | 4/32 | 4组wait.meaning重述已显式给出的q域、前驱和pipe位置。 |
| extract_moe-F18 | 7/63 | 7组gather strategy只能局部缩写：保留Offs[E]的读取绑定；扁平chunk、128分块、无route、尾块和guard的其余复述可缩。 |
| extract_moe-F19 | 6/10 | 6组第二验证strategy仅重述先前参数/分片定义、独立重算和已在math/flow/动作中给出的版本覆盖。 |
| extract_moe-F20 | 9/9 | 9组table strategy完全重述单worktile、完整码域、模块存续及无消费者。 |
| extract_moe-F21 | 1/3 | direct.schedule.strategy复述已由math、bounded pipe和assigned_done给出的无效unit/最后到达/nullable边界。 |
| extract_moe-F22 | 75/88 | 未知近似误差、比较器、unchecked读回、plain同址写和失败前缀是必要来源限制；不把源缺陷当核心字段缺失。 |
| extract_moe-F23 | 1/1 | verify_only明确提示HostObserved可能来自先前verification；当前镜像无代际且host reset不等前GPU writer。 |
| extract_moe-F24 | 1/3 | 一组publish契约的末句把optional mirror也笼统写成前请求完成后才复用，需明确这只限同流device访问；host sentinel reset并无该等待。 |
| extract_moe-F25 | 84/30 | 具体缺事件候选：当前flow的本请求kernel_done不能结构化等同host对整个device的同步完成与随后ready-set发布。这里列出相关init节点的when、主要W定义/摘要和显式Realization；字段面积不等于缺口大小。 |

F05 的 when/索引集合包含中文说明，节点数排除了 `dispatch.md` 这个文件标签。F15 是广泛的主机服务相关正文集合；F25 才是具体全设备 host 等待候选（84 个相关字段、30 个初始化节点），并非这 84 个字段的每个字符都缺结构。所有完整 unit ID、节点、反例及已有字段写法在机器明细。

已有结构足以承载的例子有两类：BN2 固定值可落在 `phase.fc2.decomposition.partition.hidden.tile: 128`，数学符号用现有 `binding: bind:design.work_schedule::fc2.decomposition.partition.hidden.tile`；6 个纯计算的自定义 scan/check 可改成通用 `compute:<已定义结果>`，保留 serial 与后续发布。两个独立副本 [bn2_candidate.yaml](../work/extract_moe/bn2_candidate.yaml) 和 [generic_compute_candidate.yaml](../work/extract_moe/generic_compute_candidate.yaml) 均通过 `check_yaml.py --standalone --skip-hardware`。这只是结构验证；若在 raw 上落实 BN2，需要同步受影响的所有 W provider，不能让子节点保留 `tile: BN2` 而形成循环。

表达成本的例子是全局别名与生命周期：片上 TensorStorage/ScaleTM、GroupTable 的共址已经用 `resource.view/part/slot_index` 表达；R 容量/M 有效视图、Counts/Fill 一次清零、两个 verdict/output 版本和模块表跨调用存续仍有正文。已有 `retention` 能表达逻辑保留，不能把它误当成分配器或地址模型；全局 allocation/view 的紧凑复用是候选，不是已证明必须扩语言。

具体缺事件候选是 `torch.cuda.synchronize(device)`：`tables.kernel_done` 只约束本请求中的 kernel；在其他 stream 有长任务时，它不表达 host 还要等整个 device 才发布 ready-set、继续配置与 probe。把该 API 换个封装不会改变核心，取消这条 host 完成边界却会改变主时序。这是 F25 的反例。当前仍可在 strategy/choice 合法保留，尚未设计或采用新字段。

有意开放的核心包括原语组合的 done 承诺、可选输入/空动作、零状态初始化与复用、四个 epilogue 读者全部退役、bulk source-read 与 global-completion 分离，以及 last-arrival 的可见性范围。现有 pipe/repeat/边已经表达大部分关系；没有把换一种 barrier API 当作核心字段缺失。原语名与 ISA 兼容绑定单列 uncertain，不能用它们证明需要新语言。

必须保留的边界：Clean=1 时有的 bounded repeat 不产生动作，有的已列动作保留 no-op issue/done；J=0 与 J>0 但 worker 无 item 的 TMEM 分配/回收不同；三槽 payload 与两槽 scale 跨 work 接续，flat FC2 另有 metadata ring 与 Pending；空 copy-group 仍参与 wait_group 选择；split activation 在 expert 检查前读 R-row partial 尾部，不能补零；未消费的 3072-byte 格式表仍是实际工作；纯录图不产生执行事件；BeginCapture 失败后的未捕获私有流前缀不能抹掉。

源限制与设计缺口分开：固定源码确认 probe 的普通同址写、未检查 D2H copy，graph_replay 的无代际 host mirror，以及 BeginCapture 失败仍调用私有流 run_pipeline。verify_only 的 HostObserved=1 可来自旧 GPU writer，不证明当前 Clean 或零输出；F24 还标出 3 个上下文共用的一条 mirror 复用说法需限定 device 顺序。它们是来源/表述问题，不是新增 barrier 或 generation 字段已经被证明必要；本轮没有修源或补造恢复方案。

## 保守短写候选

27 个精确 quote→replacement 候选合并后，去重层面净减 5,950 字符，创作声明层面净减 **75,898** 字符；全部属于英文 YAML，约占开放声明的 2.45%。其中 W/T/R 为 72,097，math 765，when 111，其他为 metadata/Contract。approved 68,239、review_candidate 7,166、needs_authoring 493。没有计入 alias 展开或 from_parent 有效视图增益；不是独立认证的最终安全删除量。

候选保留必填字段和执行事件。例如：

- `Drain(t,q) completes after the four explicit prior-work accumulator releases. q identifies the one noninitial-work wait; FC1 performs this wait before its first scale copy.` → `Wait for the explicit predecessors.` 保留长形式的 `execution_mode: serial`、q 域、pipe 位置和四个 release 前驱；不删除整个 wait 定义。
- `One worktile covers all 256 byte codes and writes every table entry.` → 删除这条可选 strategy；公式、partition/target、实际 table launch 保留。
- 任务 describe → `MoE with precomputed routing, prepared NVFP4 weights, gated activation, activation QDQ and BF16 output.` 同时用于全部 106 节点，保持必填与同名描述要求。
- tolerance 仅缩写重复措辞，保留 allclose 参数次序、两组 mismatch 阈值、max_rel_l2、输出别名检查身份、全部未提供定义及不采用 schema 默认公式的声明。完整短稿见 JSON。

Supervisor末轮复核收紧了7组gather strategy（63次声明）：Offs输入只有prefix:1，无offset，相应gather动作也未另行明确Offs[E]。M=Offs[E]的数学等式不足以替代这次读取映射，因此撤回整段删除，保留读取句及各组的Clean/guard限定，只压缩flatten、128分块、空域和裁剪等重复说明。分类从DUPLICATE改为MIXED；声明级净减量较上一版减少3374字符，去重级减少504字符。核对记录见 [gather_offset_qa.json](../work/extract_moe/gather_offset_qa.json)。

没有按长字段比例扣减。特别是 106 个 when、last-arrival/emit 合约和 host service 长段仍含必要信息；MIXED 面积不可整体算作可删或必需。

## 每节点覆盖

下表逐一列出全部 154 计划节点。`self` 表示本地 provider；M/W/T 为三个实际 Design provider；所有有创作内容节点都已核对当下 when、完整有效 Design 与本地 Realization。每个角色/资源/模式/索引/边和 flow 的结构明细在 [context_records.json](../work/extract_moe/context_records.json)、[phase_structures.json](../work/extract_moe/phase_structures.json) 和 [flow_groups.json](../work/extract_moe/flow_groups.json)。

| 节点 | 状态 | 版本 | M / W / T provider | 开放声明字符 | 覆盖 |
|---|---|---:|---|---:|---|
| [direct](../inputs/extract_moe/direct.yaml) | approved | 1 | self / self / self | 15787 | 已读；2 phases |
| [double_verify_fused_big_cp](../inputs/extract_moe/double_verify_fused_big_cp.yaml) | approved | 1 | self / self / self | 34729 | 已读；9 phases |
| [double_verify_fused_big_flat](../inputs/extract_moe/double_verify_fused_big_flat.yaml) | approved | 1 | self / self / self | 44976 | 已读；9 phases |
| [double_verify_fused_big_mono](../inputs/extract_moe/double_verify_fused_big_mono.yaml) | approved | 1 | self / self / self | 36182 | 已读；9 phases |
| [double_verify_fused_serial](../inputs/extract_moe/double_verify_fused_serial.yaml) | approved | 1 | self / self / self | 35257 | 已读；9 phases |
| [double_verify_fused_small](../inputs/extract_moe/double_verify_fused_small.yaml) | approved | 1 | self / self / self | 32585 | 已读；7 phases |
| [double_verify_fused_small_cp](../inputs/extract_moe/double_verify_fused_small_cp.yaml) | review_candidate | 1 | self / self / self | 45460 | 已读；7 phases |
| [double_verify_fused_small_flat](../inputs/extract_moe/double_verify_fused_small_flat.yaml) | review_candidate | 1 | self / self / self | 45521 | 已读；7 phases |
| [double_verify_fused_small_mono](../inputs/extract_moe/double_verify_fused_small_mono.yaml) | approved | 2 | self / self / self | 39680 | 已读；7 phases |
| [double_verify_root](../inputs/extract_moe/double_verify_root.yaml) | approved | 1 | self / self / self | 32479 | 已读；9 phases |
| double_verify_split_cp | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| double_verify_split_flat | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| [double_verify_split_store](../inputs/extract_moe/double_verify_split_store.yaml) | approved | 1 | self / self / self | 57633 | 已读；7 phases |
| [fused_big_cp](../inputs/extract_moe/fused_big_cp.yaml) | approved | 1 | self / root / self | 20299 | 已读；7 phases |
| [fused_big_flat](../inputs/extract_moe/fused_big_flat.yaml) | approved | 1 | root / root / self | 12771 | 已读；7 phases |
| [fused_big_mono](../inputs/extract_moe/fused_big_mono.yaml) | approved | 1 | root / root / self | 11890 | 已读；7 phases |
| [fused_serial](../inputs/extract_moe/fused_serial.yaml) | approved | 1 | root / self / self | 14093 | 已读；7 phases |
| [fused_small](../inputs/extract_moe/fused_small.yaml) | approved | 1 | self / self / self | 22225 | 已读；5 phases |
| [fused_small_cp](../inputs/extract_moe/fused_small_cp.yaml) | approved | 1 | self / fused_small / self | 21052 | 已读；5 phases |
| [fused_small_flat](../inputs/extract_moe/fused_small_flat.yaml) | approved | 1 | fused_small / fused_small / self | 15226 | 已读；5 phases |
| [fused_small_mono](../inputs/extract_moe/fused_small_mono.yaml) | approved | 1 | fused_small / fused_small / self | 12306 | 已读；5 phases |
| [init_no_scan_direct](../inputs/extract_moe/init_no_scan_direct.yaml) | approved | 1 | self / self / self | 19219 | 已读；3 phases |
| [init_no_scan_fused_big](../inputs/extract_moe/init_no_scan_fused_big.yaml) | approved | 1 | self / self / self | 26139 | 已读；8 phases |
| [init_no_scan_fused_big_cp](../inputs/extract_moe/init_no_scan_fused_big_cp.yaml) | approved | 1 | self / self / self | 27605 | 已读；8 phases |
| [init_no_scan_fused_big_flat](../inputs/extract_moe/init_no_scan_fused_big_flat.yaml) | approved | 1 | self / self / self | 27204 | 已读；8 phases |
| [init_no_scan_fused_big_mono](../inputs/extract_moe/init_no_scan_fused_big_mono.yaml) | approved | 2 | self / self / self | 27052 | 已读；8 phases |
| [init_no_scan_fused_serial](../inputs/extract_moe/init_no_scan_fused_serial.yaml) | approved | 1 | self / self / self | 25210 | 已读；8 phases |
| [init_no_scan_fused_small](../inputs/extract_moe/init_no_scan_fused_small.yaml) | approved | 1 | self / self / self | 25035 | 已读；6 phases |
| [init_no_scan_fused_small_cp](../inputs/extract_moe/init_no_scan_fused_small_cp.yaml) | approved | 1 | self / self / self | 27897 | 已读；6 phases |
| [init_no_scan_fused_small_flat](../inputs/extract_moe/init_no_scan_fused_small_flat.yaml) | approved | 1 | self / self / self | 32289 | 已读；6 phases |
| [init_no_scan_fused_small_mono](../inputs/extract_moe/init_no_scan_fused_small_mono.yaml) | approved | 1 | self / self / self | 27983 | 已读；6 phases |
| init_no_scan_split_cp | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| init_no_scan_split_flat | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| [init_no_scan_split_mono](../inputs/extract_moe/init_no_scan_split_mono.yaml) | needs_authoring | 1 | self / self / self | 32045 | 已读；6 phases |
| [init_no_scan_split_store](../inputs/extract_moe/init_no_scan_split_store.yaml) | approved | 1 | self / self / self | 30764 | 已读；6 phases |
| init_no_scan_split_ws | attempted_without_authored_result | 1 | — | — | 无创作结果骨架；空 when，三域 from_parent |
| [init_probe_double_verify_fused_big](../inputs/extract_moe/init_probe_double_verify_fused_big.yaml) | approved | 1 | self / self / self | 42530 | 已读；11 phases |
| init_probe_double_verify_fused_big_cp | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| init_probe_double_verify_fused_big_flat | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| init_probe_double_verify_fused_big_mono | attempted_without_authored_result | 1 | — | — | 无创作结果骨架；空 when，三域 from_parent |
| [init_probe_double_verify_fused_serial](../inputs/extract_moe/init_probe_double_verify_fused_serial.yaml) | approved | 1 | self / self / self | 44629 | 已读；11 phases |
| [init_probe_double_verify_fused_small](../inputs/extract_moe/init_probe_double_verify_fused_small.yaml) | approved | 1 | self / self / self | 40775 | 已读；9 phases |
| init_probe_double_verify_fused_small_cp | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| init_probe_double_verify_fused_small_flat | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| init_probe_double_verify_fused_small_mono | attempted_without_authored_result | 1 | — | — | 无创作结果骨架；空 when，三域 from_parent |
| init_probe_double_verify_split_cp | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| init_probe_double_verify_split_flat | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| init_probe_double_verify_split_store | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| [init_probe_fused_big](../inputs/extract_moe/init_probe_fused_big.yaml) | approved | 1 | self / self / self | 30107 | 已读；9 phases |
| [init_probe_fused_big_cp](../inputs/extract_moe/init_probe_fused_big_cp.yaml) | approved | 1 | self / self / self | 32288 | 已读；9 phases |
| [init_probe_fused_big_flat](../inputs/extract_moe/init_probe_fused_big_flat.yaml) | approved | 1 | self / self / self | 39598 | 已读；9 phases |
| [init_probe_fused_big_mono](../inputs/extract_moe/init_probe_fused_big_mono.yaml) | approved | 1 | self / self / self | 31444 | 已读；9 phases |
| [init_probe_fused_serial](../inputs/extract_moe/init_probe_fused_serial.yaml) | approved | 1 | self / self / self | 29595 | 已读；9 phases |
| [init_probe_fused_small](../inputs/extract_moe/init_probe_fused_small.yaml) | approved | 1 | self / self / self | 30272 | 已读；7 phases |
| [init_probe_fused_small_cp](../inputs/extract_moe/init_probe_fused_small_cp.yaml) | approved | 1 | self / self / self | 33040 | 已读；7 phases |
| [init_probe_fused_small_flat](../inputs/extract_moe/init_probe_fused_small_flat.yaml) | approved | 1 | self / self / self | 38369 | 已读；7 phases |
| [init_probe_fused_small_mono](../inputs/extract_moe/init_probe_fused_small_mono.yaml) | approved | 1 | self / self / self | 32255 | 已读；7 phases |
| init_probe_split_cp | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| init_probe_split_flat | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| init_probe_split_mono | attempted_without_authored_result | 1 | — | — | 无创作结果骨架；空 when，三域 from_parent |
| [init_probe_split_store](../inputs/extract_moe/init_probe_split_store.yaml) | review_candidate | 1 | self / self / self | 37972 | 已读；7 phases |
| init_probe_split_ws | attempted_without_authored_result | 1 | — | — | 无创作结果骨架；空 when，三域 from_parent |
| [init_probe_verify_fused_big](../inputs/extract_moe/init_probe_verify_fused_big.yaml) | review_candidate | 1 | self / self / self | 37210 | 已读；10 phases |
| init_probe_verify_fused_big_cp | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| init_probe_verify_fused_big_flat | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| [init_probe_verify_fused_big_mono](../inputs/extract_moe/init_probe_verify_fused_big_mono.yaml) | review_candidate | 1 | self / self / self | 41069 | 已读；10 phases |
| [init_probe_verify_fused_serial](../inputs/extract_moe/init_probe_verify_fused_serial.yaml) | approved | 1 | self / self / self | 40411 | 已读；10 phases |
| [init_probe_verify_fused_small](../inputs/extract_moe/init_probe_verify_fused_small.yaml) | approved | 1 | self / self / self | 37620 | 已读；8 phases |
| init_probe_verify_fused_small_cp | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| init_probe_verify_fused_small_flat | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| [init_probe_verify_fused_small_mono](../inputs/extract_moe/init_probe_verify_fused_small_mono.yaml) | review_candidate | 1 | self / self / self | 43020 | 已读；8 phases |
| init_probe_verify_only | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| init_probe_verify_split_cp | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| init_probe_verify_split_flat | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| init_probe_verify_split_mono | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| init_probe_verify_split_store | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| init_probe_verify_split_ws | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| [no_scan_direct](../inputs/extract_moe/no_scan_direct.yaml) | approved | 1 | direct / self / direct | 5745 | 已读；2 phases |
| [no_scan_fused_big](../inputs/extract_moe/no_scan_fused_big.yaml) | approved | 1 | root / self / root | 7825 | 已读；7 phases |
| [no_scan_fused_big_cp](../inputs/extract_moe/no_scan_fused_big_cp.yaml) | approved | 1 | self / self / fused_big_cp | 15464 | 已读；7 phases |
| [no_scan_fused_big_flat](../inputs/extract_moe/no_scan_fused_big_flat.yaml) | approved | 1 | root / self / fused_big_flat | 6819 | 已读；7 phases |
| [no_scan_fused_big_mono](../inputs/extract_moe/no_scan_fused_big_mono.yaml) | approved | 1 | root / self / fused_big_mono | 7504 | 已读；7 phases |
| [no_scan_fused_serial](../inputs/extract_moe/no_scan_fused_serial.yaml) | approved | 1 | root / self / fused_serial | 6922 | 已读；7 phases |
| [no_scan_fused_small](../inputs/extract_moe/no_scan_fused_small.yaml) | approved | 1 | fused_small / self / fused_small | 6687 | 已读；5 phases |
| [no_scan_fused_small_cp](../inputs/extract_moe/no_scan_fused_small_cp.yaml) | approved | 1 | fused_small_cp / fused_small / fused_small_cp | 4275 | 已读；5 phases |
| [no_scan_fused_small_flat](../inputs/extract_moe/no_scan_fused_small_flat.yaml) | approved | 2 | fused_small / fused_small / fused_small_flat | 5737 | 已读；5 phases |
| [no_scan_fused_small_mono](../inputs/extract_moe/no_scan_fused_small_mono.yaml) | approved | 1 | fused_small / self / fused_small_mono | 7723 | 已读；5 phases |
| no_scan_split_cp | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| no_scan_split_flat | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| [no_scan_split_mono](../inputs/extract_moe/no_scan_split_mono.yaml) | approved | 1 | split_store / self / split_mono | 7971 | 已读；5 phases |
| [no_scan_split_store](../inputs/extract_moe/no_scan_split_store.yaml) | approved | 1 | split_store / self / split_store | 8583 | 已读；5 phases |
| [no_scan_split_ws](../inputs/extract_moe/no_scan_split_ws.yaml) | approved | 1 | split_store / self / split_ws | 7433 | 已读；5 phases |
| [probe_double_verify_fused_big](../inputs/extract_moe/probe_double_verify_fused_big.yaml) | approved | 1 | self / self / self | 39567 | 已读；10 phases |
| probe_double_verify_fused_big_cp | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| probe_double_verify_fused_big_flat | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| [probe_double_verify_fused_big_mono](../inputs/extract_moe/probe_double_verify_fused_big_mono.yaml) | approved | 1 | self / self / self | 40676 | 已读；10 phases |
| [probe_double_verify_fused_serial](../inputs/extract_moe/probe_double_verify_fused_serial.yaml) | approved | 1 | self / self / self | 41247 | 已读；10 phases |
| [probe_double_verify_fused_small](../inputs/extract_moe/probe_double_verify_fused_small.yaml) | approved | 1 | self / self / self | 37467 | 已读；8 phases |
| probe_double_verify_fused_small_cp | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| probe_double_verify_fused_small_flat | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| [probe_double_verify_fused_small_mono](../inputs/extract_moe/probe_double_verify_fused_small_mono.yaml) | approved | 1 | self / self / self | 52812 | 已读；8 phases |
| probe_double_verify_split_cp | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| probe_double_verify_split_flat | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| probe_double_verify_split_store | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| [probe_fused_big](../inputs/extract_moe/probe_fused_big.yaml) | approved | 1 | self / self / self | 27682 | 已读；8 phases |
| [probe_fused_big_cp](../inputs/extract_moe/probe_fused_big_cp.yaml) | approved | 2 | self / self / self | 29499 | 已读；8 phases |
| [probe_fused_big_flat](../inputs/extract_moe/probe_fused_big_flat.yaml) | approved | 2 | self / self / self | 36367 | 已读；8 phases |
| [probe_fused_big_mono](../inputs/extract_moe/probe_fused_big_mono.yaml) | approved | 1 | self / self / self | 28221 | 已读；8 phases |
| [probe_fused_serial](../inputs/extract_moe/probe_fused_serial.yaml) | approved | 1 | self / self / self | 26295 | 已读；8 phases |
| [probe_fused_small](../inputs/extract_moe/probe_fused_small.yaml) | approved | 1 | self / self / self | 27940 | 已读；6 phases |
| [probe_fused_small_cp](../inputs/extract_moe/probe_fused_small_cp.yaml) | approved | 1 | self / self / self | 29624 | 已读；6 phases |
| [probe_fused_small_flat](../inputs/extract_moe/probe_fused_small_flat.yaml) | approved | 1 | self / self / self | 36198 | 已读；6 phases |
| [probe_fused_small_mono](../inputs/extract_moe/probe_fused_small_mono.yaml) | approved | 1 | self / self / self | 29103 | 已读；6 phases |
| probe_split_cp | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| probe_split_flat | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| [probe_split_mono](../inputs/extract_moe/probe_split_mono.yaml) | approved | 1 | self / self / self | 35166 | 已读；6 phases |
| [probe_split_store](../inputs/extract_moe/probe_split_store.yaml) | approved | 2 | self / self / self | 45816 | 已读；6 phases |
| [probe_split_ws](../inputs/extract_moe/probe_split_ws.yaml) | approved | 1 | self / self / self | 42929 | 已读；6 phases |
| [probe_verify_fused_big](../inputs/extract_moe/probe_verify_fused_big.yaml) | approved | 1 | self / self / self | 34214 | 已读；9 phases |
| [probe_verify_fused_big_cp](../inputs/extract_moe/probe_verify_fused_big_cp.yaml) | review_candidate | 1 | self / self / self | 45915 | 已读；9 phases |
| [probe_verify_fused_big_flat](../inputs/extract_moe/probe_verify_fused_big_flat.yaml) | review_candidate | 1 | self / self / self | 46711 | 已读；9 phases |
| [probe_verify_fused_big_mono](../inputs/extract_moe/probe_verify_fused_big_mono.yaml) | approved | 1 | self / self / self | 37815 | 已读；9 phases |
| [probe_verify_fused_serial](../inputs/extract_moe/probe_verify_fused_serial.yaml) | approved | 1 | self / self / self | 37190 | 已读；9 phases |
| [probe_verify_fused_small](../inputs/extract_moe/probe_verify_fused_small.yaml) | approved | 1 | self / self / self | 34700 | 已读；7 phases |
| probe_verify_fused_small_cp | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| probe_verify_fused_small_flat | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| [probe_verify_fused_small_mono](../inputs/extract_moe/probe_verify_fused_small_mono.yaml) | approved | 1 | self / self / self | 40337 | 已读；7 phases |
| probe_verify_only | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| probe_verify_split_cp | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| probe_verify_split_flat | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| probe_verify_split_mono | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| [probe_verify_split_store](../inputs/extract_moe/probe_verify_split_store.yaml) | review_candidate | 1 | self / self / self | 40710 | 已读；7 phases |
| probe_verify_split_ws | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| [root](../inputs/extract_moe/root.yaml) | approved | 7 | self / self / self | 30245 | 已读；7 phases |
| [split_cp](../inputs/extract_moe/split_cp.yaml) | approved | 1 | self / split_store / self | 30411 | 已读；5 phases |
| [split_flat](../inputs/extract_moe/split_flat.yaml) | approved | 1 | split_store / split_store / self | 17131 | 已读；5 phases |
| [split_mono](../inputs/extract_moe/split_mono.yaml) | approved | 1 | split_store / split_store / self | 16788 | 已读；5 phases |
| [split_store](../inputs/extract_moe/split_store.yaml) | approved | 1 | self / self / self | 25384 | 已读；5 phases |
| [split_ws](../inputs/extract_moe/split_ws.yaml) | approved | 1 | split_store / split_store / self | 21519 | 已读；5 phases |
| [verify_fused_big_cp](../inputs/extract_moe/verify_fused_big_cp.yaml) | approved | 1 | self / self / self | 31230 | 已读；8 phases |
| [verify_fused_big_flat](../inputs/extract_moe/verify_fused_big_flat.yaml) | approved | 1 | self / self / self | 33107 | 已读；8 phases |
| [verify_fused_big_mono](../inputs/extract_moe/verify_fused_big_mono.yaml) | approved | 1 | self / self / self | 32432 | 已读；8 phases |
| [verify_fused_serial](../inputs/extract_moe/verify_fused_serial.yaml) | approved | 1 | self / self / self | 30932 | 已读；8 phases |
| [verify_fused_small](../inputs/extract_moe/verify_fused_small.yaml) | approved | 1 | self / self / self | 29040 | 已读；6 phases |
| [verify_fused_small_cp](../inputs/extract_moe/verify_fused_small_cp.yaml) | approved | 1 | self / self / self | 31417 | 已读；6 phases |
| [verify_fused_small_flat](../inputs/extract_moe/verify_fused_small_flat.yaml) | approved | 1 | self / self / self | 32565 | 已读；6 phases |
| [verify_fused_small_mono](../inputs/extract_moe/verify_fused_small_mono.yaml) | approved | 1 | self / self / self | 34556 | 已读；6 phases |
| [verify_only](../inputs/extract_moe/verify_only.yaml) | approved | 2 | self / self / self | 9202 | 已读；1 phases |
| [verify_root](../inputs/extract_moe/verify_root.yaml) | approved | 1 | self / self / self | 28250 | 已读；8 phases |
| verify_split_cp | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| verify_split_flat | not_started | — | — | — | 未启动；仅检查计划/dispatch覆盖 |
| [verify_split_mono](../inputs/extract_moe/verify_split_mono.yaml) | approved | 1 | self / self / self | 34526 | 已读；6 phases |
| [verify_split_store](../inputs/extract_moe/verify_split_store.yaml) | approved | 1 | self / self / self | 34300 | 已读；6 phases |
| [verify_split_ws](../inputs/extract_moe/verify_split_ws.yaml) | approved | 1 | self / self / self | 36762 | 已读；6 phases |

## 验证与限制

- 154节点中仅106有创作语义；5份空when的from_parent骨架与43未启动不纳入语义密度。全部154状态及dispatch已读，缺口不能解释成简洁。
- 根硬件区是skip-hardware空值骨架，hardware文本组为0；CTA/SMEM/TMEM等来源称呼不构成完整硬件Contract或设备能力验收。
- 对2555精确文本组逐组人工分类；使用可还原词级差分仅减少显示重复；另全读166局部结构、73 flow、289数学属性定义及106使用节点。
- 源码核对固定ce1b7d...私有snapshot的matching字节，按source_identity.json列出的风险/机制定点亲读；没有逐行复审全部约万行CUDA实现或全部外部ISA/benchmark。
- 没有GPU、数值验收、全域dispatch穷尽证明或新语义批准；approved标签仅是冻结状态。
- 原语名字的核心重要性按uncertain单列；SOURCE_LIMIT及MIXED字符面积不是可删量。0 UNRESOLVED仅表示本轮文字分类和绑定已覆盖，不表示源正确性问题已解决。
- 删改量仅来自精确quote候选；未独立执行全套短稿等价审阅，不能称最终安全可删除量。BN2短稿仅在work副本做结构验证；raw provider更新需同步所有消费者。
- 局部原语保证采用源实现及现有契约，不把同一依赖图的不同barrier/API直接统计为核心结构缺口。host全设备完成事件为有据候选，是否扩语言仍需范围决策。

统一汇总命令：`python3 -B scripts/summarize.py --operator extract_moe`（在审计根目录运行）。结果 VALID：2555/2555 组、106/106 语义节点，0 格式/覆盖错误、0 未决上下文；93 组核心重要性 uncertain。完整准确统计见 [audit_summary.json](../data/extract_moe/audit_summary.json)。输入 YAML、规范、Runner、AGENTS.md、kernel 与运行状态均未修改。
