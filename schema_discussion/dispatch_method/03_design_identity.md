# 设计身份、适用范围与 root 的最小职责

2026-11-13，第 2 轮独立研究。先读现行 DOC0、DOC3–6 判断语言职责，再核对第 1 轮证据；没有采用其候选分组作为既定结论。本稿仅提出规则建议，不修改正式文档、模板、wiki 或提取工具，不运行提取或 GPU 验证。

给主代理的建议：

1. **完整 YAML 记录一套端到端设计，完整性以其声明范围为界。** 它应把共同 Contract 的输入接到全部输出，并交代所需的数值路径、工作归属、物化与协议；不再把原实现的每个选择结果当作一个设计身份。
2. **区分新增决定与记录深度。** 参数实例化、重要局部决定、新的整体组织、无新增决定的已知组合是不同情况。生产者、launch 或三区内容变化都是检查线索，不是自动生成完整 YAML 的判据。
3. **重要局部决定只记录一次，附在设计索引中。** 只补新增决定及必要关联条件，不设置一套每条必填的输入输出、存储、事件、成本小模板；它不获得独立完整设计、自动展开或实施批准的身份。
4. **删除 `when` 的实际选中充要条件、互斥和全域覆盖要求。** 最小方案保留这个字段名，但唯一语义改为本记录承担的设计范围。允许范围重叠和未覆盖；范围声明不等于正确性证明、已验证集合或最大可行域。
5. **root 保留唯一 Contract 容器和默认阅读入口，删除强制祖先职责。** 比较可选主父与无 parent 两案后，本稿倾向删除 `parent`：设计来源关系统一在 `dispatch.md` 说明，内容引用另外明确 provider，避免两处维护来源关系。
6. **放弃路径穷举，不放弃影响结果的事实。** 缓存键、包装调用和无新增计算的组合可概括；输出生产者、舍入、物化、初始化、所有权、发布与复用保证不能随之删除。验证输出和 host 镜像竞态尤其不是普通包装。
7. **保留少量具体的整体组合实例来解释真实耦合。** FSCAN 与加权 combine、Sparse packed 前缀与 mma 余段等应贯通表达；链接已有部件不能替代组合后的工作范围、值接口与接续。
8. **不预设完整设计总数，也不承诺节省比例。** 主要收益来自取消重复组合的创作和独立审阅义务。内容共享有自身价值，但不能替代设计身份改革。

## 1. 独立读现规范得到的问题

| 现行规则 | 语言职责上的问题 | 应保留的部分 |
| --- | --- | --- |
| DOC0 要求包括 root 在内的 `when` 独立求值、互斥并穷尽 Contract | 设计目录被迫承担一个固定实现的完整 selector；竞争设计必须人为分割输入，缓存和失败前缀也可能增加节点 | 每份设计明确自己的范围；条件不从目录位置或其他节点暗中继承 |
| 不可参数化的实际提交、生产者、物化或事件差异要拆成完整节点 | 从“必须表达真实差异”直接跳到“必须再写完整算子”，缺少记录深度的判断 | 不能把多条结构并集放进 YAML，再靠正文删提交、换生产者、取消等待 |
| 每个节点单父、最终连 root；共享只限直接父 | Contract 归属、阅读组织、设计来源和内容来源被绑在一起 | Contract 只存一份；引用必须显式；新上下文仍要检查兼容性 |
| DOC3/4/5 对值、工作和事件要求完整 | 这本身不是过度细化；问题是完整性要求被重复施加到每个路径组合 | 精度、物化、工作归属、初态与依赖均须够用；DOC5 已允许设计级动作合并，DOC6 已有删除测试 |

依据：[DOC0 节点字段][doc0-node]、[选择条件与索引][doc0-select]、[参数变化与结构变化][doc0-split]、[DOC3 数学步骤][doc3-unit]、[DOC5 动作粒度][doc5-grain]、[DOC6 删除测试][doc6-delete]。

因此应先改变目录的交付承诺，再讨论引用范围或紧凑写法。继续要求“任何来源调用都恰有一个完整节点”，就仍需解释那些调用的全部差异；换引用语法不能消除这个工作。

## 2. 完整 YAML 的单位与可操作判据

**设计身份不是原始字段的相等关系，也不是原创性判断。** 对当前任务，一份完整 YAML 应是一套读者能贯通理解、能据以实施的端到端组织：采用哪些数学值与精度，怎样产生和归并结果，工作归谁，必要资源与依赖怎样闭合。一个重要局部设计仍是真实的设计知识，只是没有因此取得另一份完整算子的交付身份。

建议 Agent 按以下顺序判断，先比较实际含义，再决定记录层级：

1. **实例化检查。** 变化能否由同一套已有的数学、索引和协议规则，在合法参数下直接得到？必须能指出参数的语义、取值来源和约束。把几套组织藏在一个枚举开关里不算参数化。
2. **删去差异的后果。** 如果只保留已有设计，读者会不会采用错误的数值路径、工作/存储所有权、等待或复用关系，或者失去这项变化的明确性能因果？会则必须保留新信息；不会才考虑实现自由或来源说明。
3. **局部边界检查。** 能否围绕已有的数学值、step、role 或一项明确协议，说明这次替换及它改变的接口，而不重新决定其余工作的归属与结果形成方式？能则优先短记。边界可以跨分区，不能为了“局部”把整条计算链临时命名成一个部件。
4. **组合检查。** 源码中出现的具体组合是否只采用了已有决定，且接口匹配没有增加新的跨工作协调？是则不新增完整身份。没有检查过的组合是未研究对象，不能标成已证明冗余，更不能自动推广成各轴任意组合。

| 类别 | 判定与记录方式 | 会触发进一步检查的反例 |
| --- | --- | --- |
| **新完整设计** | 新的算法/工作组织必须贯通解释，尤其重新决定部分结果、输出区域、跨 worker 的合并或状态归属；或者需要把多种计算架构组织成一次新的完整调用。选择一个有据实例写全 | 只因新增一个查表 kernel 就升级；只因三个区域都变了就升级 |
| **重要局部变体** | 一个可命名的决定有完整的局部解释；受影响接口可说清，边界外仍使用已知组织。记一次变化及关联条件，不要求可机械施加的 patch | 只写 `alpha=true`；只写“更快写回”而遗漏源释放与目标完成的区别 |
| **参数或实现选择** | 同一规则的合法数量、映射参数留在原 YAML；不改变既定设计保证的查找方式、API 或代码组织留给实现或必要的 Realization | 把不同舍入点当 dtype 参数；把 Queue server 消失只当 worker 数变化 |
| **无新增设计信息的组合** | 已有决定在一个已核对的具体上下文中共存，没有新的接口或协调决定；必要时给一个来源链接，不另复制完整计划 | 同 shape 的 Act 其实分别是 BF16 与 packed FP4；FSCAN 接了未加权语义不匹配的旧 combine |

“新的整体组织”不以代码范围衡量。Sparse 的新增编排代码很短，却改变了输出区域归属和跨流接续；MoE 查表前缀使三个区域都变了，却不改变后面的两 GEMM 主干。反之，一个局部变化可以涉及多个 role、多条事件，不能因为需要几句准确解释就强迫它成为完整算子。

一套数量无关的共同规则可以包含现有 `for/if`，例如环槽索引和非首轮等待。若读者必须分别理解两套工作归属、发布意义或归并方式，给它们一个总开关并不会减少设计决定，应按上表分开完整记录或局部说明。

## 3. 用已核案例检验，而不直接采纳第一轮分组

以下事实来自两份第一轮报告；本轮另读了所列的代表 YAML 和源码片段。这里判断的是记录层级，没有重新验证其 GPU 正确性。

| 案例 | 建议与判据应用 | 证据及范围 |
| --- | --- | --- |
| FA4 prefill 单/多请求 folded persistent | **合并同一设计身份。** 前缀映射、工作归属与流水线相同；B=1 是同一映射的退化。线扫/二分是查找实现。可先保留已有已说明子域的并集，不据此扩展到所有 batch、所有 loader 和排序组合 | [多请求映射][fp-packed]、[单请求映射][fp-single]；源码求解过程见 [02 §3.1][cross-prefill] |
| FA4 alpha 共享 | **重要局部记录。** Stats 从两列旧 max/base 改为一列 alpha，Softmax 计算供自身分母和 Correction 复用；Stats 捕获/归还与 Y 交接仍须说明。改变生产者不自动成为另一条完整 attention 主线 | [普通数学及 Stats][fp-root-stats]、[共享数学][fp-alpha-math]、[共享资源][fp-alpha-storage] |
| MoE `fused_small_flat` 与 `no_scan_fused_small_flat` | **不保留第二份完整 Design。** 三个有效区域相等；新增信息是重建服务、workspace 初态/绑定/寿命。记录这些条件一次，不把缓存状态当核心计算身份，也不把新增要求默认为所有调用已有保证 | [01 §3][moe-diff]、[新增 Realization][moe-services]。内容相等仅证明重复，不证明这些条件可在所有主干无条件复用 |
| MoE `init_no_scan_fused_big` | **查表准备的一次局部记录，而非三区都要重学的设计。** 后续七段 pipeline 原文相同。若完整实例声明包含这次提交，就准确保留它；若代表实例只覆盖准备已完成的调用，明确前提，来源额外工作可留在一次准备说明中 | [01 §3][moe-diff]、[新增查表及 flow][moe-init]。不能把未消费的计算反写成源码未执行 |
| FMHA FSCAN | **优先保留完整代表。** 取消全局 scan 状态，split 写未加权 partial、PartM，末 split 产生 MTail，combine 才归一到共同参考。这里要重新理解部分结果与全局归并的协作，不是一个可独立套用的 scan 开关 | [FSCAN 生产与说明][fmha-fscan]、[02 §6][cross-fmha]。只核了代表，不证明所有 copy/store/combine 配对 |
| Sparse `probe_hidden_mma` | **保留一个具体完整组合。** 调用已经在 caller stream 计算 packed 前缀，余段才采用 mma 并在 side stream 运行；输出区间分割、fork/join 和余段不等前缀是新增的整体组织 | [实际函数][sparse-hidden]、[02 §5][cross-sparse]。仅无 split、无既有 range 等已核条件；不能泛化到任意后端 |
| MoE REDF | **优先保留完整代表。** RouteResult/finalize 改成 Acc32/convert，并连带 BF16 后加权、FP32 原子归并、旁流清零和 join；这是输出形成方式改变。只记录“FC2 融合 finalize”不够 | [01 §4][moe-axis]、[REDF 数学/输出][moe-redf]。拒绝描述符时不能照旧回退 rout producer |
| GEMM 多次恢复重算 | **保留一次恢复事实，不按重算次数形成新的完整 GEMM。** 已有 G/F 计算重复、同流顺序和后次覆盖输出需要记录；各次重复没有创造新的矩阵算法 | [02 GEMM 对照][cross-gemm]。原有不同 GEMM 架构仍各有设计内容；未证明所有错误组合可达 |

对第一轮建议的三处修正：

- [01 的默认四组、扩展五组][moe-groups]可以作为试写集合，不能作为最少数量或固定分类。其“跨 math/schedule/pipeline”理由应改成上面的具体协作理由，否则 `init_tables` 也会被误判为整体新设计。selfprep、不同 BF16 链等仍需按目标阅读粒度和接口实际试写，不能从该表自动得到必须完整保留的数量。
- 01“局部说明至少给出输入输出接口、存储版本、等待提交……”应改为**只补本项变化造成的关联条件**。alpha 不必复写 attention 输入输出和完整存储表；缓存重建也不必再给一遍 FC1/FC2 数学。必须核实事实，不代表每条要走一套完整 schema 填表。
- QSA 只能说明“一个 root 不表示泛化到所有同名任务”。当前 [QSA Contract][qsa-contract]已经固定 Rows、单请求、索引生成模式，不能把它当作现成的“宽 Contract、窄 root Design”实例。后一能力应由新规则明确允许。

另有几类必须防止的误合并：

| 不充分的合并理由 | 实际会丢什么 |
| --- | --- |
| topology 一样、都是两次 launch | FA4 decode compact 改为只读有效 split；普通 global 会读空 split，要求相应 scratch 有限。还改变局部归约组织，见 [02 §4][cross-decode] |
| 最后都生成 BF16 Act / 最后 shape 相同 | MoE BF16 融合与独立激活的中间 FP32 物化不同；packed/BF16 也不能混接。见 [01 §4][moe-axis] |
| 都是 persistent、worker 数不同而已 | MLA 的 Queue server、终止描述符、补领和跨工作预取在 bootstrap 中消失。见 [02 MLA 对照][cross-mla] |
| 已有 A、B，两者分别成立就能拼起来 | FSCAN partial 不能接旧加权位置的 combine；MoE REDF 不能接写 rout 的回退；Sparse 不接受“probe 失败，所以前缀未执行” |

## 4. `when` 只表示记录范围，不承担 selector

### 必须分开的四种事实

设共同 Contract 的调用域为 C。这里的集合只解释职责，不要求 YAML 新增集合语言。

| 事实 | 含义 | 最小落点 |
| --- | --- | --- |
| **本记录的设计范围 R** | 这份 Design 的定义和约束承担哪些输入及必要调用状态。它可以是保守的研究/实施范围，不声称是最大正确性域 | 唯一字段 `meta_data.dispatch.when`；正文相关字段说明参数、数据就绪和资源条件，避免重复 |
| **已有验证 V** | 哪份实现、环境、workload 和检查实际通过；有限样本不证明整个 R，更不证明 R 外不能用 | 固定的实验/审阅记录，由索引链接。`realization.feedback` 仍只做其已有的替代反馈，不改成通用状态表 |
| **原实现被选中 S** | 固定来源的 host 条件、缓存/捕获状态、probe 观察等，使实际调用选择这条路径 | `dispatch.md` 中必要的来源选择说明及固定源码链接；不另加 `selector`、优先级、fallback 字段 |
| **目录尚未覆盖 U** | C 中没有任何完整记录范围承担的调用；局部说明也不因此升级为完整覆盖 | 索引一次说明主要边界。不能推断为原实现不支持、调用非法或自动退到 root |

R、V、S 不应相互替换。来源选到了一个路径，不证明它满足 Contract；测试过一些点，不自动扩大记录范围；没有被原 selector 选中，也不证明竞争设计不能工作。已知源码缺陷必须明确保留，不能以范围声明使其获得正确性保证。

**推荐保留 `when` 这个现有字段名，只保留上述 R 的语义。** 不并列新增 `scope`，不在同一字段中交替使用“适用条件”和“精确原选路”。旧稿中的 `Select exactly when` 必须按新职责重审，不做机械措辞替换。若正式决策选择更名，也应一次替换，而非维护两种字段。

### 允许重叠、空缺与单 root 的窄范围

- 两套完整设计可以对同一输入成立并竞争。目录不选胜者、不设隐式优先级，root 也不是运行时默认 fallback；实际实现者按实施任务或证据选方案。
- 所有完整记录共用原 Contract。R 较窄只限制这份设计承担的调用，不改变 C 的合法输入、输出、误差或硬件要求。
- 单 root 同样可以显式填写窄 `when`。例如**假设**通用 GEMM Contract 允许任意正 N，本次只记录处理 N 为 128 倍数的一套设计：root 的范围可以这样限制，Contract 不必被改成只允许整除输入。这里是规则能力示例，不是已核某个 GEMM kernel 的适用结论。
- 为维持单文件最小写法，单 root 仍可省略 `dispatch`，但其唯一缺省意义应为“本记录承担整个 Contract 域”，不能兼有“作者未填写范围”。有任何额外限制时必须显式写出。

`when` 仍用简洁自然语言和已定义量，不承担执行程序。范围所需事实的提供者和就绪条件必须可定位；一次 probe 的结果不能以“已知条件”替代实际 probe 或完成等待。来源恢复路径需要按已发生的结果叙述时，应标清这是来源行为，而不是提前可用的安全判据。

来源阈值分两种：FMHA narrow combine 的共享存储容量界限属于设计成立条件；仅据一次性能比较选出的阈值属于选择依据。两者可能写在同一个源码 `if` 中，也要分别解释。可以保守地只记录来源已研究的子域，但不能把那个阈值称为设计的最大有效边界，也不能未经分析删除所有门后宣称更广泛适用。

## 5. root、归属、来源：哪种关系真的需要进入语言

| 职能 | 建议 |
| --- | --- |
| 唯一 Contract 归属 | 保留。所有完整 YAML 的 Contract 名称固定解析到同目录 `root.yaml`，不依赖祖先路径 |
| 默认阅读入口 | 保留。root 仍是一份实际完整设计，选择易理解、便于比较的起点；不要求最宽范围、最优性能或覆盖全部调用 |
| 全体设计祖先 | 删除。独立的 mma.sync 与 TMEM 组织无需伪装成同一设计的派生历史；共享数学与硬件即可同目录 |
| 内容来源 | 显式引用实际 provider 并检查消费上下文；它不推出设计祖先、适用域继承、实现兼容或自动组合 |

这里的“不相关架构”指同一硬件 Contract 下的不同实现架构。如果实际更换设备或已固定的实现栈，不能借取消 parent 合并成一个任务；仍按 DOC2 的共同前提处理。

### 可选 parent 与删除 parent 的比较

| 方案 | 能做什么 | 代价与判断 |
| --- | --- | --- |
| 保留可选 `parent`，仅表示主要设计来源；省略不默认 root | 提供一个机器可读的比较起点；简单线性改进可直接导航 | A+B+C 仍要在 MD 说明其余来源，形成 YAML 主来源与 MD 其余关系的双重维护；“主”常是人为选择。当前 `from_parent` 依赖它是现有语法造成的需求，不是设计本身的必需 |
| **删除 `parent`，设计关系统一在 `dispatch.md`** | 索引用链接说明独立、替换、借鉴和已知组合；引用字段只记录内容来源 | 放弃机器校验单父设计祖先的能力；需要使整区引用直接定位 provider。没有发现用户目标要求可机器消费的单一祖先，因此更简洁 |

**推荐第二案。** `meta_data.source` 继续定位主要外部设计依据；同目录设计之间的来源关系只在索引说明，不再把其中一个来源重复写入 YAML。`dispatch.id` 仍标识完整文件，修订号沿用原规则。区域引用指向哪里属于内容依赖，和索引的设计关系不是重复声明同一件事。

具体组织上：root 链接共同 Contract 和一个实际设计；独立架构直接并列列出；局部决定附在所关联设计之后或集中一次链接。所有文件须从索引可发现，但不要求有一条设计祖先边连回 root，不造只装 Contract 的抽象根。

A+B+C 有两种处理：若只是已核对的既有决定共存，说明已见的组合和必要边界即可，不给每个组合造节点；若组合新增整体组织，就保留一个具体完整 YAML，在索引说明各部分来源及新增协作。Sparse hidden_mma 是后一种。无需 `parents: [A,B,C]`、合成顺序或多继承框架，来源链接也不授权自动把部件拼起来。

## 6. 有价值设计覆盖与必须保留的完整性

目录的覆盖对象应是**选定研究范围中的设计决定**，不是所有源码路径。开始时说明本次研究哪些固定来源和实现族；遇到不同数学路径、物化方式、工作归属或协作协议，就按判据决定完整记录或局部记录。相同决定只记一次，已核组合只给必要关联，未研究的实现族和组合说明边界，不凑完整选路表。

| 信息 | 完整性要求与可采用的抽象 |
| --- | --- |
| 范围与数值结果 | 完整 YAML 在声明范围内交代全部输出、写回副作用、边界分支和实际精度路径；不能用局部误差或“同公式”抹掉中间舍入、量化和归约方式 |
| 物化、生产消费与所有权 | 必要中间值、有效区域、版本、存续期、初始化及生产者必须明确。普通分配 API、地址计算和不影响约束的缓存容器可以省略 |
| 提交与协议 | 被该完整设计表示的提交、数据就绪、启动许可和资源释放按 DOC4/5 闭合；不能靠正文从已有 flow 取消事件。全局新协作不能只由索引链接承担 |
| 外部准备和缓存 | 可概括真正提供的值、初态、就绪和寿命保证，不逐项复制查表/包装调用。前提必须有来源依据；cache hit 不自动证明内容新鲜、跨流排他或输入版本匹配 |
| 初始化前缀 | 对结果或协议必要的初始化必须保留其值、位置、作用范围与先后；普通建立服务可按 DOC4 的外部供给/初态抽象。原实现额外做了未被主干消费的查表，可一次说明其事实与成本范围，不说成源码未执行 |
| 失败、超时、恢复 | 已经发生的计算、写回、异步访问和占用不能因后续失败而消失。相同计算的重复可一次概括顺序、覆盖和存续要求；不为每种返回码组合再独立写全算子 |

主动放弃的是“完整重建原调用轨迹”的承诺。如果代表设计只覆盖已准备资源的调用，冷启动必须明示在范围之外；不能一面删掉初始化，一面仍把 cold 调用和其端到端计时当成该完整实例已经表达的内容。源码没有提供的初值、等待或隔离，不能因为采用抽象服务边界而补造出来。

### MoE 验证是这条边界的必要检验

1. **验证可能计算真正的输出。** verify 写零，带 guard 的已提交主干可能早退；因此不能统一列为无关 host 包装，也不能把所有已提交 kernel 从记录中删除。
2. **两次验证有两次生产与不同数据版本。** 双图调用不能压成一次 verify；一份共同协议说明可以避免它与每种 FC2 重复，但不能抹掉实际顺序、生产者和版本。
3. **host 观察不等于本次 device 判定。** [源码 `graph_replay`][moe-replay]写 -1 后提交并有限轮询；[GPU 发布][moe-publish]可能在更晚时刻写同一 word。没有代次标签且 CPU 重置不等待旧 writer，前次超时的发布可以被后次观察。源码注释声称不会陈旧，不是其协议成立的证明。[`verify_only` 已记录限制][moe-verify-only]。
4. **记录缺陷不等于修复缺陷。** 不能新增“所有调用隔离”“host 读到 1 就证明当前 Clean”来获得漂亮的适用范围。可选一份确实值得贯通理解的验证主干完整记录，并把 host 选择、双图和已知限制集中说明；存在已知反例的来源行为不能被宣称为已保证正确的可用设计。

本轮只是静态源码交错分析，不报告 GPU 复现或故障率。验证的设备计算、输出写入和 guard 属于设计；主机镜像的版本缺口是实现正确性事实。两者都应保留，但无需为每个主干前缀配对重复整个解释。

## 7. 两套可以执行的组织方案

两案都先放弃完整 selector，允许设计范围重叠；共同 Contract、必要事实和固定来源都保留。

| 方案 | 实际怎么写、怎么使用 | 主动放弃什么 | 成本和风险 |
| --- | --- | --- | --- |
| A. 精选完整设计目录 | 合并同规则实例，省去没有新决定的组合；每种决定保留一个承载它的完整实例。alpha、compact、不同主循环分工等若选入目录，就在某个代表中完整表达，不要求与每条主线配对 | 不重建原 selector，不覆盖未选组合；不承诺根节点通吃 | 每项入选变体都有可直接实施的完整上下文，但完整实例仍可能不少；需要完整审阅的对象较多 |
| **B. 完整主线 + 一次性局部记录** | 整体新组织有完整 YAML；有清楚局部边界的决定在索引中短记并关联已有完整设计；已见组合只保留必要来源与条件 | 除 A 的放弃项，还不承诺每条局部记录能直接生成独立实现、能自动展开或已完成组合验收 | 更接近少量精炼 schema 的目标；实现者选中某个局部组合后，才为该具体目标补成完整方案并检查上下文 |

本稿推荐 B；如果用户要“目录中每种变化都可直接领取实施”，则 A 更匹配，不能一面要求这项承诺，一面把完整工作成本算成已省去。

B 避免成本搬家的具体规则：

- 局部记录以一个新增决定为单位，不为每个旧 YAML/后缀自动生成一条；多个已有适用实例以链接或一组名称说明，不扩成组合矩阵。
- 不设必填小 schema，不复制背景说明；alpha 的记录只围绕一列 Stats 的生产、复用及原有捕获/Y 握手，缓存记录只补新增服务与存续条件。
- 局部事实需要核实，但默认随所属设计/索引一并审读，**不为每条启动完整独立的创建、展开、审阅、生成和发布任务**。它也没有完整节点的语义批准身份。
- 真正需要实施某个组合时，只补全该组合并复核引用后的数值、资源和事件；不预先展开所有潜在组合。若局部说明反复需要重新安排整体工作，按前面的判据把它升为完整设计。

这些规则减少重复决策和重复任务，不减少真实新机制所需的理解。无需人为设置节点上限，也无需扩大 `if` 到可以切换全部后端。现有成本数据没有隔离这种组织方式的节省比例，本稿不预测美元、时间或返修量。

## 8. DOC0–6 的保留、删除与改写清单

| 文档 | 保留 | 删除或改写 |
| --- | --- | --- |
| **DOC0** | 一个任务目录、实际 root、唯一 Contract、ID/版本/固定来源、显式内容引用及消费上下文检查 | 删除 when 的选中充要条件、互斥穷尽、单 root 必须全域和全树回 root；推荐删除 parent。when 改为唯一的记录范围语义，允许单 root 显式窄范围。参数/结构规则改为本稿的记录层级判断；索引改为设计目录，不再必须派生树 |
| **DOC1** | 调用者数学、全部输出、边界与副作用、golden reference 和共同误差；不允许为实现方便改 Contract | 不新增每节点 Contract 或容差；明确设计范围较窄不改变合法输入域，记录范围也不是已验证域 |
| **DOC2** | 唯一目标硬件/栈和资料绑定，设备事实与设计选择分开 | 不为不同 kernel 架构复制硬件；真正更换设备/栈仍不能伪装成同 Contract 的局部选择。无须其他结构变更 |
| **DOC3** | 从输入到全部输出、数值路径、物化/retention、参数归属与精度例外 | 完整性按选中完整设计的声明范围检查，不要求顺带表示其他来源路径。调整整区引用开头的 parent 限制；不因局部记录存在而放松正式数学字段 |
| **DOC4** | 实际表示的 step/launch 身份、工作归属、合并执行者、初始化、版本、存续、启动与数据就绪区别 | 把“提交/交接变化则按 dispatch 拆完整节点”改为记录层级判断；继续禁止正文删 flow。明确普通准备过程可由有据的外部供给概括，影响输出/所有权的工作不可隐藏 |
| **DOC5** | 设计级动作、有限 for/if、跨 role 等待、真实动作存在性、资源与事件闭合 | 调整“不能参数化就拆 dispatch”的自动升级；仍不得用巨型条件并集或一句局部说明改变已交付完整 YAML 的事件 |
| **DOC6** | require/prefer/forbid/feedback 的现有职责、删除测试、所有适用硬条件共同成立、必要寿命和原语保证 | 每个完整设计仍填写自己必要的交接；不再因缓存/入口选择不同自动产生一份必须独立填写的节点。局部记录与一般验证状态不塞进 feedback，不新增兜底 notes 字段 |

上述删除 parent 要同步使内容引用明确 provider，但具体共享字段与语法不在本稿展开。来源关系的删减不应自动放松内容依赖、证据固定和消费上下文的检查。

### 候选正式规则文字

> 一个任务共用一份数学与硬件 Contract，root 保存 Contract 并作为实际设计的默认阅读入口。每份完整 YAML 记录其 `when` 声明范围内的一套端到端 Design；范围可以重叠或未覆盖整个 Contract，既不表示原实现选路，也不证明正确性、性能或最大适用域。参数能由同一套规则表达时不另建设计；新增整体计算、结果归属或协作组织时保留完整设计；具有明确局部边界的新增决定可在设计索引中只记录其变化及必要条件，不作为可自动展开或独立获批的完整计划。已有决定的组合仅因选择条件不同不新增身份，新的接口与协作仍须表达。所有完整设计共用同目录 root 的 Contract，不要求设计祖先关系；设计来源在索引中说明，内容共享显式指向其定义。完整设计内的数值路径、物化、生产消费、初始化和必要协议不得因省略源码选路而缺失；已知来源限制照实保留，不补造保证。

### 提取与工具的必要 TODO

- 先确认新的交付承诺，再修改规划/审阅要求：计划单位是完整设计与少量局部决定，停止自动补齐缓存、失败和后端组合的完整节点矩阵。
- 同步 when 的单一语义、单 root 窄范围、parent/索引职责和区域 provider 引用；现行实例逐项判断迁移，不用字段别名兼容两种语义。
- 完整设计继续检查其范围内的输出、资源、依赖及引用上下文；取消 selector 穷尽检查后，另以目录边界声明区分尚未研究范围，不能把不检查覆盖解释成已覆盖。
- 局部记录不进入独立完整节点的生成/批准流水线；实际实施一个组合时才生成该完整目标，仍固定全部有效内容来源和必要证据。

## 9. 事实与未验证范围

本稿直接复读了 DOC0、DOC3–6 的相关规则及 DOC1/2 的职责与边界；复读了 FA4 前缀映射/alpha 的代表 YAML、QSA Contract、Sparse hidden dispatch、FMHA FSCAN epilogue、MoE host/GPU 验证和既有 `verify_only` 限制。其余横向事实沿用第一轮明确列出的固定来源和复现记录，未在本轮重新统计所有文件、运行复现脚本或重新批准历史稿。

仍未验证：全部 MoE/FMHA 组合可达性与正确性、局部记录支撑实际实施的充分程度、各候选完整设计分组的最小性、方法的净成本收益。本文的独立结论是职责和判断规则的建议，不是十算子的最终迁移方案。

[doc0-node]: /home/shaohanyong.shy/KernelSchema/schema_docs/0.common.md:43
[doc0-select]: /home/shaohanyong.shy/KernelSchema/schema_docs/0.common.md:144
[doc0-split]: /home/shaohanyong.shy/KernelSchema/schema_docs/0.common.md:162
[doc3-unit]: /home/shaohanyong.shy/KernelSchema/schema_docs/3.design.math_reshape.md:15
[doc5-grain]: /home/shaohanyong.shy/KernelSchema/schema_docs/5.design.tile_pipeline.md:20
[doc6-delete]: /home/shaohanyong.shy/KernelSchema/schema_docs/6.realization.md:157
[cross-prefill]: /home/shaohanyong.shy/KernelSchema/schema_discussion/dispatch_method/02_ten_operator_evidence.md:68
[cross-decode]: /home/shaohanyong.shy/KernelSchema/schema_discussion/dispatch_method/02_ten_operator_evidence.md:92
[cross-sparse]: /home/shaohanyong.shy/KernelSchema/schema_discussion/dispatch_method/02_ten_operator_evidence.md:107
[cross-fmha]: /home/shaohanyong.shy/KernelSchema/schema_discussion/dispatch_method/02_ten_operator_evidence.md:126
[cross-mla]: /home/shaohanyong.shy/KernelSchema/schema_discussion/dispatch_method/02_ten_operator_evidence.md:164
[cross-gemm]: /home/shaohanyong.shy/KernelSchema/schema_discussion/dispatch_method/02_ten_operator_evidence.md:170
[moe-diff]: /home/shaohanyong.shy/KernelSchema/schema_discussion/dispatch_method/01_moe_evidence.md:86
[moe-axis]: /home/shaohanyong.shy/KernelSchema/schema_discussion/dispatch_method/01_moe_evidence.md:106
[moe-groups]: /home/shaohanyong.shy/KernelSchema/schema_discussion/dispatch_method/01_moe_evidence.md:128
[fp-packed]: /home/shaohanyong.shy/KernelSchema/schema_template/examples/fa4_prefill/packed_persistent.yaml:74
[fp-single]: /home/shaohanyong.shy/KernelSchema/schema_template/examples/fa4_prefill/single_persistent_dual_load.yaml:74
[fp-root-stats]: /home/shaohanyong.shy/KernelSchema/schema_template/examples/fa4_prefill/root.yaml:116
[fp-alpha-math]: /home/shaohanyong.shy/KernelSchema/schema_template/examples/fa4_prefill/small_persistent_shared_rescale.yaml:39
[fp-alpha-storage]: /home/shaohanyong.shy/KernelSchema/schema_template/examples/fa4_prefill/small_persistent_shared_rescale.yaml:113
[qsa-contract]: /home/shaohanyong.shy/KernelSchema/schema_template/examples/qsa_prefill/root.yaml:8
[sparse-hidden]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/candidate/dsv4_kernel.cu:6140
[fmha-fscan]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_tcg.cuh:5181
[moe-services]: /home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/no_scan_fused_small_flat.yaml:80
[moe-init]: /home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/init_no_scan_fused_big.yaml:229
[moe-redf]: /home/shaohanyong.shy/KernelSchema/schema_wiki/archive/six-operators/initial/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/mono_split_redf.yaml:170
[moe-replay]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:9471
[moe-publish]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:1764
[moe-verify-only]: /home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/verify_only.yaml:145
