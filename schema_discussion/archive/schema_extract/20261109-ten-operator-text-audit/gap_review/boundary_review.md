# 调用、外部存储与状态边界：第 4 类候选复核

本报告只复核既有十算子审计的相关 finding，不重做审计、不提取、不运行 GPU。全量检索了 `findings_catalog.json` 的 182 条记录，按机制选取外部身份、全局 backing、host 准备、caller 边界、跨调用状态的相关条目；逐条结论见 `boundary_findings.json`。读了根 README、审计 README/METHOD、相关算子报告，并复核关键 YAML、规则和冻结源码。当前七份 `schema_docs/*.md` 与审计 `rules/schema_docs` 逐字相同。本报告不修改它们或原标注。

**最明确的现状是：GPU plan 与外部执行环境之间缺少现行可引用的事件端点；必须新增端点字段尚未证明。** Sparse F05 的 caller fork/join、MLA F18 的 Geo 上传完成到主 kernel 启动，已有确切数据可见性与重叠关系；现行事件引用却只能落到本次计算的 phase。应比较“一次定义的请求边界/外部供给约定”与“实例显式引用外部端点”两个方案，不需要把 host 包装器变成完整执行 DSL。

跨调用物理存储身份也确实没有统一结构，但“没有结构”与“值得加入 Pipeline IR”必须分开。GDN F16、Sparse F06、MoE F07 包含真实不能删的事实，也混有 allocator、缓存策略、源缺陷与外围 API 约束；现有材料不足以把它们整体升级为必须扩语言。外部输出 alias/write-set 则优先统一简短语义，不因缺少 `alias` 字段就判成语言缺陷。

## 判定使用两个轴

1. **当前是否有结构载体。** 结构本身加共享规范能否表达关系？把正文挪到 `choice`、给对象起名、或引用一个仍靠长文定义的对象，均不算建立了新的结构关系。引用确实能绑定已经定义的整个对象，不能一概说 ref 没有身份能力。
2. **是否需要扩展。** 该事实是否直接决定数学、GPU 工作、内存合法性、关键等待或主要复用；短而统一的语义定义是否足够；新增词汇是否比保留少量边界条件更有价值？源代码有这个 API 或跨文件重复很多次，本身不能回答这三个问题。

`confirmed_structural_gap` 在本报告仅确认相关核心关系没有现行显式结构端点，**不表示第二轴中的新增固定字段必要性已通过**。两条对应记录的 `new_fixed_field_required` 均为 `not_proven`。此前把“两轴均有较强支持”当作该标签含义过强，本次一致性复核予以收窄。`scope_decision` 表示没有结构载体已确定，但是否纳入本 IR 的范围仍须单独决定。`shared_semantic_definition` 表示不能删的语义需要短定义，未证明新增固定字段更好。其余结论保留既有结构、表示成本、来源不确定或无缺口的区别。

根 [README:44](../../../../README.md:44)、[README:46](../../../../README.md:46) 的目标是保留绝大多数一阶性能因果且不膨胀成 DSL；[README:69](../../../../README.md:69) 的删除测试同时要求不漏设计。这两个要求必须一起使用。

## 规则上的硬边界与反向证据

| 现行规则 | 实际能支持什么 | 不能据此推出什么 |
| --- | --- | --- |
| [DOC1:41](../../../../schema_docs/1.contract.algorithm_requirement.md:41) 要求原地结果登记 output 并在 meaning 写明 | 外部原地含义有明确、合法的唯一入口 | 没有 alias 固定字段，因此 API 身份毫无表达能力 |
| [DOC1:135](../../../../schema_docs/1.contract.algorithm_requirement.md:135) 属性 ref 等价于复制属性，不表示共享存储 | shape/dtype 共享 | 同 shape 就是同对象、同地址 |
| [DOC3:168](../../../../schema_docs/3.design.math_reshape.md:168)、[DOC3:176](../../../../schema_docs/3.design.math_reshape.md:176) 可引用固定材料，binding 引用整个对象 | 绑定已存在且唯一可解析的外部提供者，而不只是抄属性 | 被引用对象自动成为 flow 的合法外部事件；绑定数学值就建立了地址、owner、跨请求复用顺序 |
| [DOC3:100](../../../../schema_docs/3.design.math_reshape.md:100)、[DOC3:108](../../../../schema_docs/3.design.math_reshape.md:108) retention 以逻辑实例定义并与空间区分 | 逻辑值的跨迭代/实例保留；初始化和不改写区域由 math 给出 | retention 只允许 invocation；或 keep_value 自动指定物理地址/owner |
| [DOC4:275](../../../../schema_docs/4.design.work_schedule.md:275)、[DOC4:309](../../../../schema_docs/4.design.work_schedule.md:309) phase 与事件绑定一次实际计算，事件形如 phase.milestone | 同请求内多 kernel 的实际工作和顺序 | caller-prior、host 观察、跨请求 writer 自然获得事件身份 |
| [DOC4:317](../../../../schema_docs/4.design.work_schedule.md:317) start_after 只限制设备开始，不阻塞 host | kernel 启动下界 | host 返回、ready-set 发布或全设备 CPU 同步完成 |
| [DOC4:341](../../../../schema_docs/4.design.work_schedule.md:341) global 不规定地址、分配 API 或 owner | 跨 phase 全局物化 | 一个全局 allocation 的容量、视图、跨调用共用身份 |
| [DOC4:357](../../../../schema_docs/4.design.work_schedule.md:357)、[DOC4:366](../../../../schema_docs/4.design.work_schedule.md:366) initialize 只补消费者所需但 producer 未写区域 | 新版本必要的初值 | 每调用清零、首次分配清零、真实 memset 提交三者等价 |
| [DOC5:257](../../../../schema_docs/5.design.tile_pipeline.md:257)、[DOC4:370](../../../../schema_docs/4.design.work_schedule.md:370) resource 按 worker 成员持有，跨 kernel 不延续局部身份 | 局部容量、别名与槽位 | 仅把 space 改成 global 就合法登记 process 级共享 owner |
| [DOC0:156](../../../../schema_docs/0.common.md:156)、[DOC0:164](../../../../schema_docs/0.common.md:164) when 可使用有定义的运行时事实，但不代替实际 probe | 当前选择、已观察值、实际提交范围 | 必须新增每一种缓存/探测专属字段；或 when 能替代运行中的事件边 |

特别纠偏：MoE [init_no_scan_direct.yaml:52](../inputs/extract_moe/init_no_scan_direct.yaml:52) 已给 module table 保留定义，并在 [同文件:59](../inputs/extract_moe/init_no_scan_direct.yaml:59) 使用 `keep_value device_module`。规范的逻辑实例不是封闭的 invocation 枚举。不能把“逻辑实例表达”混为“物理存储身份、owner、观测版本”。

## A. 外部对象身份、可写区域和返回形式

覆盖 template GDN F03、FA4 prefill F01、FA4 decode F13、QSA F01、extract GDN F01、Sparse F15、MoE F04；MLA F03、FMHA F08 是相同关系的重复说明对照。

**最小事实。** 返回值与调用方提供对象的关系、允许写入的区域、其他输入保持、旧目标内容是否参与计算。不要把 Python 包装对象身份、底层 allocation 别名、逻辑更新后的值三者写成同一个 “alias”。例如模板 GDN [root:26](../inputs/template_gdn_prefill/root.yaml:26)、[root:29](../inputs/template_gdn_prefill/root.yaml:29)、[root:43](../inputs/template_gdn_prefill/root.yaml:43) 分别约束旧值、返回关系和写集。extract GDN [root:68](../inputs/extract_gdn/root.yaml:68) 的 fresh Out 与 [root:70](../inputs/extract_gdn/root.yaml:70) 的 caller prefix 是两种不同结果。

**删除反例。** 同值的新张量返回且原缓冲不写、写原缓冲但返回另一个包装对象、只写 active prefix 却破坏 guard，这些实现不能由同 shape/dtype 或数值等式区分。把旧 output 当累加初值在 NaN 预填时也不等价；后者已有输入范围和输出公式可直接约束。

**排除现有方案。** shape/dtype ref 明文不建 storage alias；`resource.view` 是 worker 局部；handoff 是内部数据交接；`initialize` 不能定义对调用方的可变性。整个对象的 binding 可以引用已经定义的外部对象，但返回顺序、allowed writes、fresh/nonalias 各关系仍须有明确含义，不能只写 `binding: ref:...` 然后让读者猜。

**结论：shared_semantic_definition。** DOC1 已主动选择由 output.meaning 表达原地结果。这里确有未结构化的关系，但大部分是固定算法 API 边界，一次短定义就够；没有证据表明给每一项加字段会明显提高 Pipeline IR 的压缩或推理。可统一写法的含义，例如“returned object is input out; writes out[region]; other input storage unchanged”，并明确“对象身份”“共享 backing”“更新后值”三词。它是建议的语义模板，不是已存在的新语法；不宣布现状已经完全结构化。K/V ABI 的 stride/base delta 若决定实际访存，则保留定量关系，不能随 Python 返回包装一起删。

## B. 全局 backing 与跨调用缓存：缺载体确定，范围不能混判

覆盖 extract GDN F16、MoE F07、template FA4 decode F14、template GDN F17。同机制的 raw-byte 暖读为 extract GDN F13。

**最小事实。** 值的逻辑区域与合法访问的 backing 区域有区别；同一 backing 的旧字节可能留给下次调用；若设计依赖复用，该存储有确定共享范围。GDN 源 [gdn_kernel.cu:3258](../work/extract_gdn/gdn_kernel.cu:3258) 只在容量不足或设备变化时重新分配；[3267](../work/extract_gdn/gdn_kernel.cu:3267) 固定 perm/seg 的 backing 区域。MoE [double_verify_split_store.yaml:154](../inputs/extract_moe/double_verify_split_store.yaml:154) 的 Part 逻辑 M 行与 backing R 行不同，[同文件:179](../inputs/extract_moe/double_verify_split_store.yaml:179) 明确先读取所有 R 行的 partial，再检查 expert，不能把无当前 route 含义当成未访问。

**删除反例。** 只分配 M 行可使 R 行实际访问越界；把保留旧值改成每次 zero 增加工作并改变允许的旧字节；为两个版本各分配独立存储会改变地址复用与流量。GDN 的每次重分配、同 process 多 stream 共享、按当前 device 重查 L2 也各自不同，但后两项不能直接声称在已授权调用域内必然造成错误。源码没有锁不等于任务允许并发。

**已有结构的真实能力。** `math` 可定义 R 域的旧值、视图、保留或更新公式；`retention` 可表达逻辑实例；handoff 可引用相同数学值及版本。把 raw backing 定义成可审阅对象，再引用其视图，能缩短很多文字。DOC4:341 明确仍不约束它们是否同一地址、归谁持有；resource 的容量/part 不可直接移到全 process。`initialize: 0` 会按新版本适用，不能伪装只在首次 allocation 时执行的初始化；它也不登记真正的 memset 工作。

**binding/ref 的复核结论。** DOC3 允许 `repository@ref:path::symbol` 和整个对象绑定，因此外部固定材料若已经定义 cache/owner，可合法提供来源与对象身份；“只能在本文件 prose 命名”说得过强。但是引用的目标若只是 allocator 函数，不能替代本 plan 的数学定义（DOC3:180）；若目标确是 ws_cache，该引用也不自动给出 capacity/view/owner 关系的共享规则，不能供 `start_after` 跨请求引用。要靠外部固定短定义表达这些条件仍然是可行的压缩方向，不能忽略后径直扩语言。

**结论。** GDN F16 主判 `scope_decision`；MoE F07 保持 `representation_cost`。当前确实没有全局物理 backing/owner 关系表，但未证明需要完整跨调用 allocator 模型。GDN direct-QK [strategy:257](../inputs/extract_gdn/direct_qk_ordered.yaml:257) 的不缩减分配容量应保留为来源选择，不能仅因“少分配也不同”就认定一阶流水线不同；source 注释中稳定 norm buffer 避免编码成本的例子位于 l2norm 分支，而本批 Contract [root:60](../inputs/extract_gdn/root.yaml:60) 固定 false，不能借不在域内的成本证据升级当前结论。L2 无设备键缓存直接影响 when，先按 C 节处理。

如未来取得**同一有效域内**“同样逻辑值与内部事件、只改变 backing 共享关系就必然多一次大搬运/改变合法复用”的成对实例，可优先扩一个最小 global storage/view 关系，复用已有值、版本、容量和事件概念；不要同时引入分配器策略语言。跨调用 owner 另需明确 key、首次状态、复用前提及调用并发域。

template GDN F17 的跨 stream 并发许可未知，FA4 decode F14 的整个 native 冷启动/失败入口未被模板范围完整覆盖，两者判 `source_or_uncertain`，不能拿缺审计范围证明缺语言。GDN F13 的 raw 暖读则已有 DOC4:368 的合法地址/丢弃语义边界，适合统一短原语定义；“提前读 raw 字节”不得改叫读取已就绪的数学 Perm。

## C. host 选择缓存与录图快照

覆盖 extract GDN F05、GEMM F03、FMHA F04、MoE F05，以及 MLA F05 的 cache/排序部分。

**最小事实。** 选择函数的值来自哪一次缓存/观察；key 是地址还是当前内容；图录制固定的 host 标量/对象绑定与 replay 时设备读取的内容不同。删掉后相同 shape、不同旧 cache 或 metadata 内容会错误选择同一方案。

这些事实可用现有 `when` 的具名局部量和谓词精确表达；DOC0:164 明确允许已说明来源的运行时事实。数学 Geo/order 可定义稳定排序与 Scatter 谓词，再由 `worktile_order: Item` 消费；不能把 N 冒充新的独立工作轴。源码引用可固定实际选择函数，但不能让 `when` 变成无定义的函数名，也不能隐式继承父 when。

**结论：representation_cost / shared_semantic_definition。** 这里的缺点主要是每节点重写完整来源与谓词。DOC0:43、45 已统一 capture/replay 时点；不存在再加“capture 完成事件”的理由。GEMM F04 的失败后重复计算已经可用不同 phase 与 `kernel_done` 串接，这是应排除的“边界缺口”。是否允许共享选择函数定义是复用设计问题，不能与外部执行事件合为一项大扩展。

## D. caller fork/join：明确缺少现行端点，新字段必要性待比较

**覆盖：extract Sparse F05 的 hidden 路径部分。结论：confirmed_structural_gap。**

冻结 [probe_hidden_packed.yaml:300](../inputs/extract_sparse/probe_hidden_packed.yaml:300) 和 [同文件:429](../inputs/extract_sparse/probe_hidden_packed.yaml:429) 给出真实关系。源 [dsv4_kernel.cu:6155](../work/extract_sparse/dsv4_kernel.cu:6155) 记录 caller fork 后让 side 等待；[6188](../work/extract_sparse/dsv4_kernel.cu:6188) 在 side 提交 remainder 后记录 join、让 caller 后继等待。后者跟 prefix 的 caller 顺序共同保护完整输出，CPU 返回不等于 GPU 输出完成。

不能删除的最少关系（下列为研究关系式，**不是合法 YAML 扩展提案**）：

```text
caller_prior -> probe.start
caller_prior -> prefix.start
probe.kernel_done -> remainder.start
{prefix.kernel_done, remainder.kernel_done} -> caller_subsequent
无 prefix.kernel_done -> remainder.start
```

两端都是调用外部的执行位置，内部真正只有三个 kernel。删去边界正文后，现有 `launch` 三条既容许正确 fork/join，也容许 side 抢读旧输入或 caller 后继过早读 remainder 输出。添加 prefix→remainder 会强行串行，失去已证设计意图；不是修复表示。

**为什么现有字段不够。** `start_after`、`ready/wait_at` 的目标必须是本 plan phase 里程碑；handoff 的 producer/consumer 也必须是 phase，且这里没有外部计算值要伪造为新 GPU producer。`refill` 只管理 worker 领取，math/ref/storage 不建执行边，`kernel_done` 不指 caller 环境；假造空 phase、host worker 或 undefined `caller_prior.done` 都不是现有语法。Realization 可以绑定 event API 的具体保证，但 DOC6:13 不允许用它重新决定依赖；因此最关键的关系现在只能留在正文。

**两个待比较的最小方案。** 一是统一规定请求的相关输入访问排在 caller 前序之后，caller 后续消费排在本请求相关输出生产者之后，并明确这种排序不阻塞 CPU 返回、不在内部输出生产者之间增加顺序。二是允许 plan 显式引用外部端点、逐项绑定输入和输出完成关系。前者可能已足够表达本例；目前没有具体反例排除它。简单的“所有 output 都完成才返回”确会错误增加 host 阻塞，但只能排除这个错误约定，不能排除经过正确定义的统一边界约定。两方案都须保留 prefix/remainder 的重叠，不规定 CUDA event API 或枚举 caller 的其他工作。

F05 还包含 capture 前 `cudaStreamSynchronize(user)`（源码 [6384](../work/extract_sparse/dsv4_kernel.cu:6384)），这部分单独归入 G 节的范围决策，不能凭 F05 主结论把所有 host 等待都升级。

## E. MLA host 几何表：保留数据版本与 ready 边，不复制整个 host 算法

**覆盖：extract MLA F18。结论：confirmed_structural_gap，仅限外部供给边界。**

样本 [root:322](../inputs/extract_mla/root.yaml:322)、[bootstrap:59](../inputs/extract_mla/bootstrap.yaml:59) 明确当前 metadata 的 D2H、host 构表、H2D 和主 kernel 等待。固定 snapshot 源 `solution.py:1669` 为 D2H，`:1895` 为上传与 event record，`:2011` 为 main stream wait（完整路径保存在 JSON 对应证据中）。`:1638` 的注释解释 side 传输避免等待 caller drain；这里复核的是源码提交与等待，不把注释中的测量数当本轮性能验证。

必须进一步收窄独占事实：**Geo 上传完成 → 整个 main grid 开始。** [root:114](../inputs/extract_mla/root.yaml:114) 已在 Contract 要求每次从当前输入与 metadata 计算、无跨调用数值保留，[root:299](../inputs/extract_mla/root.yaml:299) 已定义 Geo 坐标。因此不能声称“只删这段 host 前缀就允许任意重用旧 Geo”：剩余值定义仍排斥错误的旧表。录制固定 host 绑定、回放读取当前 table 内容的区别也已有 DOC0:45，不能再把整项时点语义算作独占缺口。

更严格的反例是：A 按源码在 grid 启动前等待上传；B 先启动 grid，在真正读取 Geo 前才等待上传。两者都能计算相同的当前值，但 worker 开始占用设备的时点和允许的重叠窗口不同。现有 Geo 数学定义、通用 load 的正确值要求不能区分这两个启动边界。缺少的是一个能被 `start_after` 引用、绑定已有 Geo 的外部 ready 端点；它不是某个已存在 compute phase 的 milestone。用 main 里的普通 load 等待只约束读取，未表达源实现的整 grid 启动下界，也无法把 side upload 定义为该 phase 事件。

可以与 D 节共用外部边界定义，无需专设 D2H、host-sort、H2D、event-record 四种字段。若规范统一规定“该类外部供给在所属计算 grid 启动前就绪”，并让本例已有 Geo 定义明确采用这一约定，也能区分启动前等待与读前等待；本轮没有反例证明它一定不够。实例端点是否值得增加，要与这种短供给前提比较，不能仅由现行端点缺失推出。是否要计量 host sort 的内部算法、缓存 resize、每个 API 的耗时仍是范围问题；本 finding 不证明应扩大到完整 host service 图。

## F. 跨调用 scheduler、host observation 与 process gate

覆盖 QSA F10、MLA F10、Sparse F06、MoE F23/F24，补充 Sparse F13/F17 的来源与空分支区别。

**scheduler 状态最小事实。** QSA [root:189](../inputs/template_qsa_prefill/root.yaml:189) 的 device ticket 与 host base 初值为 0，每个串行 eager 请求消耗 `Delta=worktile_count+worker_count`，下一 base 增加 Delta；源 [operator.py:104](../work/template_qsa_prefill/operator.py:104) 是 workspace key，[157](../work/template_qsa_prefill/operator.py:157) 是 base 推进。MLA [root:343](../inputs/extract_mla/root.yaml:343) 的私有 queue 最后一次 pop 复位下一有序请求的 counter。

这类关系可以在已存在 `refill` 中用短协议/公式统一：领取结果、terminal、初值、复位和跨次前提。DOC4:225 明确要求 refill 说明计数由谁维护；其示例也已允许全局 counter（DOC4:232）。不能声称需要先造 local resource 才能说 global counter。物理 key/owner 没有结构表是真的，但尚未证明一个短协议定义不能压缩好这些案例。两条主判 `shared_semantic_definition`。QSA 重放固定 tbase 而 ticket 持续增加可使第二次全部拿 terminal；这说明不能外推到 replay 安全，不是创建 replay 修复字段的理由。

**Sparse 观察最小事实。** [dsv4_kernel.cu:5588](../work/extract_sparse/dsv4_kernel.cu:5588) 先把同一 process gate 写 sentinel 再提交 probe；[5602](../work/extract_sparse/dsv4_kernel.cu:5602) 是有界轮询；[5610](../work/extract_sparse/dsv4_kernel.cu:5610) 另作最终普通 scan，仅对实际读取的正值求平均。poll 退出不是共同快照，也不是 probe 完成；之后的 GPU attention 仍由内部 `probe.kernel_done` 排序。timeout 并不保证最终扫描没有正值。它可能在 probe pending 时与下一调用 reset 冲突。

**删除反例。** 把 R 改为八个最终 GPU 值的完整平均，会改变实际 dispatch；把 host scan 完成当 gate 可复用，会错误证明无竞争；每次私有 gate 则改变共享事实。MoE [source_nvfp4_moe.cu:9481](../work/extract_moe/source_nvfp4_moe.cu:9481) 的 host sentinel reset 同样没有等待先前 GPU writer，不能把同 stream 设备顺序套到 CPU store；HostObserved=1 不证明当前 Clean=1。

**结论。** Sparse F06 主判 `scope_decision`：R 的数学/实际观察定义由 `when` 与 math 可以精炼，录制时点有共享规则，内部 probe→attention 已有结构；全 process gate 身份、跨请求 writer 与 host 观察端点则确无统一结构。若范围需要对跨请求竞争做可组合检查，这才与 B/D 的存储和外部事件关系一起成为扩展对象。当前应保留短的观察语义和源限制，不能把源码缺 generation、缺同步改写成需要新增安全 generation 字段。MoE F23/F24 属 `source_or_uncertain`（已证源/表述问题，而非事实本身不确定）；Sparse F17 的 no-work、no-op、stale、已提交但观察不完整，已有公式/domain/flow/动作语义各自表达。

## G. 全设备 host 等待与 ready 发布：事实确定，IR 范围待定

**覆盖：MoE F25、F15 的对应子事实，以及 Sparse F05 的 capture 前同步。结论：scope_decision。**

MoE [source_solution.py:1015](../work/extract_moe/source_solution.py:1015) 明确 `torch.cuda.synchronize(index)` 后才更新 ready set；[init_no_scan_fused_small_flat.yaml:871](../inputs/extract_moe/init_no_scan_fused_small_flat.yaml:871) 保存了完整条件。放一个很长的既有任务在另一 stream：只等 tables.kernel_done 会让 host 提前构建后续工作，原实现则继续阻塞。这是有效的时序反例，非 API 名称差异。

`start_after: tables.kernel_done` 明文不要求 host 阻塞，不能替代全设备等待。`handoff.initialize` 也不能表达 ready-set 发布或一个真实 memset；若消费者区域已全部被 kernel 写出，不应补初始化。没有现有 host worker 或 service phase，可供直接塞入 flow。

但该等待发生在冷初始化/包装器中，还等待本 plan 没有产生的任意外部任务。把这项连同所有 allocator、host cache、返回异常、同步 API 都建成一等执行图，会改变 IR 的目标与复杂度。保留一句必需来源条件是合理现状；若明确要比较完整请求的 host enqueue/cold-start 时序，可复用 D/E 的外部事件边界，并区分 host continuation 与 device start，不必另造完整 host 服务分类体系。这一范围决定不能由“84 个字段、30 个节点”或正文长度替代。

## 六条 missing_field_or_event 的收敛结果

| 原 finding | 本次主判 | 当前确无结构的部分 | 是否直接推动 schema_docs |
| --- | --- | --- | --- |
| template_gdn_prefill-F03 | shared_semantic_definition | API identity/write-set 固定关系 | 统一短定义；暂不新增固定字段 |
| extract_gdn-F16 | scope_decision | 全局 backing 的地址/owner/跨请求关系 | 先界定需要保留的一阶物理关系和有效调用域 |
| extract_mla-F18 | confirmed_structural_gap | 外部 Geo-ready → main-start | 比较统一供给前提与实例端点；新字段必要性未证 |
| extract_sparse-F05 | confirmed_structural_gap | caller-prior 与 caller-subsequent 端点 | 比较统一请求边界与实例端点；capture host-sync 子项单独范围决策 |
| extract_sparse-F06 | scope_decision | process gate 与跨请求 host 观察/写者身份 | 当范围包含跨请求协议检查时再扩；R 和内部依赖已有入口 |
| extract_moe-F25 | scope_decision | 全设备 host completion → ready 发布 | 保留来源事实，先决定完整 host 时序是否在目标内 |

本报告最支持的下一步，是对 **外部供给 ready、caller 输入顺序、caller 输出汇合** 做一个共用最小定义，拿 MLA/Sparse 两例比较统一边界约定与实例端点的总表示成本、适用域和可组合性。**当前没有排除前者的具体反例，故不能把两条明确的现行端点缺口说成两条必须新增固定字段的证明。** 数值正确性仍由现有数学定义约束；需要收束的是等待位置、可见性与执行接续。最应排除的是“多几次真实 kernel 提交”“纯 capture 单独完成事件”“shape ref 就能 alias”“每次 initialize 修复历史 scratch”“给所有 host/cache API 都加字段”。跨调用全局存储有真实欠表达，尚需更窄的核心实例才适合升级。
