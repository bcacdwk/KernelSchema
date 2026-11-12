# 跨调用状态：删除反例与设计代价复核

**结论：修改原建议；保留“不加新模块”，撤回“scratch 只写本次重写范围与复用次序”及“容量一概删除”。**
这两个量不足以确定读哪个版本、未写槽的初态、共享对象身份、最后一次读取/写入是否结束，以及容量是否覆盖真实访问。
这些缺失事实在本次核查的实例中都有现行入口；必要新增字段 **0**，必要规范语义增量 **0**，正式 doc0–6 可以完全不改。
不加模块是比较现行能力与反例后的选择，不是调查前提；它也不表示现有字段已结构化全部跨请求存储关系。

## 阅读范围与判断口径

主入口是 [SCHEMA_VERBOSITY:80–89](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/six-operators/20261109-methodology-review/SCHEMA_VERBOSITY.md:80) 与 [十算子审计:64–94](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/README.md:64)，不是按 cache/malloc 关键词判必要性。
现行规范只用本轮 `rules/0–6`；[旧缺口复核:76–86](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/SCHEMA_LANGUAGE_GAPS.md:76) 只作定位，逐条重新对照。
语义阅读聚焦 GDN root、MLA root、MoE root/init_no_scan_direct/verify_only、Sparse probe_serial_mma、模板 QSA root、FA4 decode root 的有关段落。
这些八份 YAML 均与 corpus 摘要一致：前六份是固定 approved 输入，后两份是 template；MoE 使用暂停 working 输入，没有借未完成节点证明语言缺口。
源码核查覆盖上述四个 extract 的必要片段；另读 QSA 两文件、FA4 四文件，均逐字匹配固定 commit `01fff90d714abc6e84904ea9c44523ba66d34a0e`。
完整路径、SHA256、实际阅读段及核查结果在 [06_state_deletion.json](/home/shaohanyong.shy/KernelSchema/schema_discussion/language_compression/review_20261112/data/06_state_deletion.json)。没有运行 GPU、改正式文件、读同轮报告或研究分支组织。

## 对照当前条文去重

| 事实 | 当前入口 | 本次判断 |
| --- | --- | --- |
| 调用者的算法状态、原地结果 | [DOC1:43–44](/home/shaohanyong.shy/KernelSchema/schema_discussion/language_compression/review_20261112/rules/1.contract.algorithm_requirement.md:43)：input/output，原地关系写 meaning | 已有能力；内部 scratch 不冒充调用者参数或输出。 |
| 值的保留域、初值、重置、未写区域 | [DOC3:96–108](/home/shaohanyong.shy/KernelSchema/schema_discussion/language_compression/review_20261112/rules/3.design.math_reshape.md:96)：retention 与 math | 已有能力；retention 不等于写入次数，也不限于 invocation。 |
| 外部对象的引用 | [DOC3:145–155](/home/shaohanyong.shy/KernelSchema/schema_discussion/language_compression/review_20261112/rules/3.design.math_reshape.md:145)：整个 binding 与属性 ref 分开 | 已有能力；属性相同不表示共享，binding 也不自动生成复用边。 |
| 外部准备、初态与共享复用前提 | [DOC4:191、264–266](/home/shaohanyong.shy/KernelSchema/schema_discussion/language_compression/review_20261112/rules/4.design.work_schedule.md:264)：相关 schedule.strategy | 原建议“不加模块”的主体已经落实；可短写身份及必要前提。 |
| 跨 phase 初值与存续 | [DOC4:295、315–326](/home/shaohanyong.shy/KernelSchema/schema_discussion/language_compression/review_20261112/rules/4.design.work_schedule.md:315)：handoff.initialize 及在途访问结束 | 已有能力；省略 initialize 不等于清零，已有初态不重复初始化。 |
| 本次动作的版本、范围与完成 | [DOC5:185–200](/home/shaohanyong.shy/KernelSchema/schema_discussion/language_compression/review_20261112/rules/5.design.tile_pipeline.md:185)：using_verb.<verb>.<noun>.describe 与 waiting | 已有能力；不能以 schedule 长文代替已经命名的必要动作依赖。 |
| 物理容量、别名与未结束访问 | [DOC5:355–428](/home/shaohanyong.shy/KernelSchema/schema_discussion/language_compression/review_20261112/rules/5.design.tile_pipeline.md:355)：worker 片上 storage/view | 只覆盖片上账；不可把它误称通用 global allocation 表。 |
| 实现机制和额外 ABI 容量约束 | [DOC6:9–13、44](/home/shaohanyong.shy/KernelSchema/schema_discussion/language_compression/review_20261112/rules/6.realization.md:9)：require 引用上游对象 | 可绑定机制；不可在此首次定义状态或依赖，reason 也无此权力。 |

特别是 [DOC4:408](/home/shaohanyong.shy/KernelSchema/schema_discussion/language_compression/review_20261112/rules/4.design.work_schedule.md:408) 已明确禁止把必要初值随分配细节一起删掉，本轮不再建议重复增加该规则。
MoE 的 [keep_value device_module](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/init_no_scan_direct.yaml:52) 是现成跨调用保留实例；旧“retention 只支持单次调用”归因必须排除。
真正没有固定结构的是跨请求全局分配的 owner/epoch/access 图；这是真实结构边界，但尚无证据要求它成为本轮必备模块。

## 可删的真实文字：保住共享关系，删除无关的缓存策略

[GDN 原文 root:381–383](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml:381)：
> The current CUDA stream orders prep, chain and reuse of the grow-only workspace across calls. Every consumed handoff slot is rewritten. The source provides no cross-stream event, lock or reentrancy protocol.

现行合法的 `design.work_schedule.phase.prep.schedule.strategy` 短写候选：
```yaml
strategy: >-
  Shared workspace is reused by ordered caller-stream executions; prep rewrites
  all consumed handoff regions. Cross-stream or reentrant exclusion is not provided.
```
`prep→chain` 的本次启动/读就绪关系仍由 [flow:326–338](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml:326) 保留；这里不重新决定它们。
源码 [3254–3274](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu:3254) 确认 static ws_cache、容量不足或设备变化才重分配、scratchbuf 延寿及 caller stream。
源码 [3316–3321](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu:3316) 使用本次 `need+qin_bytes`，没有用旧分配的 numel 作工作集判据。
因此在本次容量已满足、共享与寿命不变的设计域，可删 grow-only 的缓存管理细节；不能连同本次需求字节量、访问边界和设备适用性一起删。
这不是承诺全进程的历史峰值内存、换设备行为或任意并发调用都等价；若这些成为设计要求，须另保留其必要约束。
另有无条件可删重复：[MoE:292](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/init_no_scan_direct.yaml:292) 的 “RouteAccum has R*H FP32 elements and Arrivals has T int32 elements.” 已由该文件 165–169 的 shape/dtype 完整表达。
这项删减不改变计算量、数值、存储身份/范围、参与者、必要先后、重叠或初始化条件。

## 绝不可整删：首次置零、每次改写、末次自清不是同一件事

[MoE init_no_scan_direct:144–161](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/init_no_scan_direct.yaml:144) 的必要事实：
> New workspace establishes zero in all R*H route entries and T counters; completed ordered requests retain zero for the next execution even if route contents change.

`RouteAccum` 从零做 FP32 atomic sums；`Arrivals` 从零计到 `n(t)*S`，最后一票触发读取、写输出和自清。
无效 route 行、无 route token 的 counter 本次不写；它们仍须靠初次全零及此前执行的自清保持下次可用，而不是靠本次写集保证。
源码 [Python:938–975](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_solution.py:938) 只在 workspace miss 建零；[CUDA:2201–2254](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:2201) 给出 atomic、最后一票、读后清零及 counter 复位。
源码 [9096–9133](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:9096) 确认这一执行没有 per-execution memset。
反例：相同当前输入、相同写集和流顺序，初次将 acc32 改成未初始化内容，会改变累加结果；Arrivals 残值会改变谁取得最后一票。
反向“修补”为每次全量清零虽可恢复某些初态，却新增 `4*R*H+4*T` 字节写入和前置工作，不是等价精简。
相邻事实中，普通 MoE [root:401–403](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/root.yaml:401) 的 Counts/Fill 确实每次清零，[CUDA:9230](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:9230) 可核；不能从“都是 workspace”推广一种初始化时机。
最小落点是已有 math 初态/更新/终态、symbol.retention 和最后到达动作的 describe；相关 schedule.strategy 只补共享身份及先前完整访问结束的复用前提。
“按 shape/config 查 cache、miss/hit 的分支过程”可以缩成上述保证；ABI cache miss 不等于得到新 workspace，不能把两个 cache 的身份混成一个。
这是必要短定义与既有字段用法，不是缺一个 `initialize_each_call` 字段。

## 未重写不等于无影响：FA4 的旧值与容量

[FA4 root:164–185](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/template_fa4_decode/root.yaml:164) 规定空 split 写 `Mpartial=-inf,Lpartial=0`，不覆写 Opartial；combine 仍读该槽并乘零。
因此未写 Opartial 必须是有限 FP32；零权重不能消除 `0*NaN`/`0*Inf`。同一当前注意力数学输入，旧槽为有限值或 NaN 会改变该实现的数值合法性。
源码 [paged:1138–1145](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/template_fa4_decode/source_gqa_decode_paged.py:1138) 跳过空 split 的主体；[combine:503–529](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/template_fa4_decode/source_gqa_decode.py:503) 在这条 warp_parallel 路径读每个 nominal slot。
首次零值有 [input.py:83](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/template_fa4_decode/source_input.py:83) 依据，已有 input.meaning/math 保存它；不能替换成每次 `handoff.initialize: 0`。
现有 [workspace require:443–446](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/template_fa4_decode/root.yaml:443) 保留 `WorkspaceBytes%4=0` 和最小 `4*B*q*H*(SplitCount*(D+2)+1)` 字节。
这里 B/q/H 分别为 BatchCount/q_len_per_req/HeadCount；[operator.py:227–254](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/template_fa4_decode/source_operator.py:227) 明确检查并切分，额外 `+1` statistics 区即使不被核读取仍参与 ABI 切片。
删除最小容量会把相同数值设计下的可用 buffer 与越界/拒绝的 buffer 混为一谈；它是必要访问/绑定条件，不是缓存增长政策。
该外部 workspace 原本就是真实输入，可沿用其 meaning/range 及已有 require；不得据此把 GDN/MoE 内部 scratch 都升格为 Contract 输入。
此例已有描述，不要求新字段；冷启动失败恢复及任意调用间重新解释同一字节布局的有限性仍未获完整证明，不能补造安全承诺。

## 不读旧算法结果，也可能必须保留协议版本

QSA [root:189–193](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/template_qsa_prefill/root.yaml:189) 保留了 device ticket 与 host base 的配对状态，串行 eager 每次共消耗 `worktile_count+worker_count` 张票。
源码 [operator:104–115、155–160](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/template_qsa_prefill/operator.py:104) 分别建立 ticket/base 及推进 base；[kernel:324–333](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/template_qsa_prefill/qsa_kernel.py:324) 使用 `atomic_add(ticket)-tbase`。
若 base 只加工作数而漏掉 W=worker_count 张 terminal ticket，下次首编号为 W，会跳过前 W 份工作；只重置 ticket 也破坏对应关系。
所以“当次写 counter、调用串行”不足以定义工作覆盖；票数、base 版本及 wrap-safe 差值须保留在已有 `schedule.strategy/refill`。
其旧算法输入不跨次保存，计数协议仍是设计状态；本证据只支持声明的 eager 域，不推出 capture/replay 自动修正 host base。
MLA [root:344–352](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/root.yaml:344) 是另一种协议：初次零，最后一次 pop 内联归零。
源码 [1461–1464](/home/shaohanyong.shy/KernelSchema/schema_discussion/phase_grammar/sources/kernel/extract_mla/solution.py:1461) 和 [1730–1736](/home/shaohanyong.shy/KernelSchema/schema_discussion/phase_grammar/sources/kernel/extract_mla/solution.py:1730) 支持它；不能给所有缓存 counter 套“每次零”或“永不零”的统一字段默认值。

## 共享身份与 host 观察：前提缺失不是新增字段的证明

MLA 的 Geo 数值已由 [Contract:114–115](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/root.yaml:114) 及 [Geo:299](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/root.yaml:299) 绑定当前 metadata；不能把“旧表不得顶替新表”再次报成缺定义。
但 static d_table 的身份与表版本的读取寿命不是 Geo 数学公式。[源码:1725–1728](/home/shaohanyong.shy/KernelSchema/schema_discussion/phase_grammar/sources/kernel/extract_mla/solution.py:1725) 在扩容时 cudaFree 有同步作用，cache hit 没有这一动作。
[源码:1895–1898](/home/shaohanyong.shy/KernelSchema/schema_discussion/phase_grammar/sources/kernel/extract_mla/solution.py:1895) 在 nonblocking side stream 覆写同一 d_table；[2010–2026](/home/shaohanyong.shy/KernelSchema/schema_discussion/phase_grammar/sources/kernel/extract_mla/solution.py:2010) 只让本次 main 等本次上传，然后异步返回。
核查范围未见“前次 main 的表读取结束→下次 side 上传”的边；连续 host 调用、主核同流，均不能单独推出外部 side 写入安全。
应保留需要排除重叠读写的调用前提或明确源缺口；不能从偶发扩容同步制造普遍保证，也不能为修源默默新增同步。
容量扩张倍率、malloc 次序可删到必要保证为止；若一条分配调用承担了设计依赖，其保证仍须保留，不能因名称像实现细节而删除。

Sparse [probe_serial_mma:89–99、184–188](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/probe_serial_mma.yaml:89) 已区分最终逐槽观察、非共同快照、旧 writer 与下次 reset 的竞争。
[源码:5582–5614](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_sparse/dsv4_kernel.cu:5582) 是共享 mapped gate、host 写 sentinel、有限 poll、最后普通读取；poll 退出不是 GPU 完成，64-word 分配不改变仅八槽的数学观察。
MoE [verify_only:145–156](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/verify_only.yaml:145) 明确 HostObserved 可来自旧请求；[源码:9480–9488](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:9480) 的 reset 在 GPU launch 前，但未等待旧 GPU writer。
因此保留“来源可能旧、不能认证本次 Clean/输出完成”是必要来源限制；加 generation 字段或把观察值改成 kernel_done 都不能如实修复这段代码。
具体 host 循环可以压成观察语义和边界；若保留相同观察/工作选择，循环 API、分配包装和 cache 容器本身不必保留。

## 最小落点与新增模块的代价

| 需要补/保留的实例事实 | 最小合法落点 | 相对只写写集和次序的增量 |
| --- | --- | --- |
| 算法状态与外部别名 | input/output.meaning，必要 math | 返回/写回对象与值版本。 |
| 初次零、保留旧值、末次自清 | math、retention；实际动作的 describe | 读前初态、未写区域及下次初态的来源。 |
| ticket/base 等领取状态 | worker_assignment.refill + schedule.strategy | 工作计数、终止票、跨次版本对应。 |
| global scratch 共享与访问寿命 | 相关 schedule.strategy，引用既有值/对象；图内依赖用 flow/waiting | 共享身份、准备动作也受约束、全部相关访问结束。 |
| 合法容量与格式要求 | 既有 shape/范围；必要额外 ABI 条件用 require | 有效读取边界和布局兼容；不记录一般增长算法。 |

这里需要的是整理实例和修正原表，不是让每个对象都补五项表头；已有公式、symbol、调用边界或动作保证覆盖时直接引用。
新增 shared_storage 至少还要定义 owner、请求实例、版本、区域、初始化、退役以及与既有绑定的关系，且仍不能自动解释 QSA 票数或 Sparse 的非快照观察。
这会同时增加规则和实例成本；本组反例都能由一处短前提加既有值/动作定义区分，尚未证明新模块有更高收益。
短契约的代价是 global 共享与跨请求版本仍主要靠人工审阅；本报告不把“可写文字”冒称静态存储/并发验证。
若未来确需组合并发请求并引用它们的区域完成端点，应另给短契约无法唯一表达的实证，再讨论结构；本轮不预先建 runtime。

建议原表改为：**算法状态沿用输入输出；内部状态沿用 math、retention 与 refill。共享 scratch 只补影响设计的身份/版本、读写及保留范围、初态来源和复用前提；保留必要容量，删除不改变这些事实的缓存查找、替换和分配过程。**
理由改为：**现行字段足以承载本组必要状态事实；本次写集与调用次序不能代替初态、旧版本和访问寿命。**
正式 docs：不改。新增字段：无。新增规范语义：无。需要收窄的是原表删除口径，以及个别实例尚未给出的共享复用前提。
