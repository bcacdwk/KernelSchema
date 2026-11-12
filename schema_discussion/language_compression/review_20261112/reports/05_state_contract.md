# 跨调用状态：最小契约与现有字段的边界

**结论：修改原表。保留“不加新模块”和“算法状态用已有输入输出”；收窄“共享 scratch 只写复用次序和本次重写范围”；撤回无条件删除全部缓存容量的说法。**
本轮建议新增固定字段 **0**、删除固定字段 **0**、新增封闭取值 **0**；去重后必须新增的规则 **0**。
需要修的是实例的归属与短定义。当前 DOC3/4/5/6 已覆盖初态、保留、版本、区域、外部供给和复用条件；旧调查中“缺统一边界”的判断不能直接沿用。

## 阅读范围与判断口径

入口是 [SCHEMA_VERBOSITY:80](../../../../schema_extract/archive/six-operators/20261109-methodology-review/SCHEMA_VERBOSITY.md:80)，再读十算子 README、SCHEMA_DISCUSSION 的状态段、SCHEMA_LANGUAGE_GAPS 及 boundary_review。
按 `inputs.json → flow_review/corpus.json` 定位并定点阅读八份 YAML：GDN root/direct_qk_ordered、FA4 decode root、QSA root、MLA root、MoE init_no_scan_direct/verify_only、Sparse probe_hidden_packed。
模板保持 template 身份；四个 extract 任务中的所选节点均为 approved；MoE 来自暂停 working 输入，不能把局部已批准说成整树完成。
核源限于这些状态的创建、更新、消费及复用路径；FA4/QSA 另核十算子归档的固定源码身份。读取范围及摘要核验在 [JSON](../data/05_state_contract.json)。
没有全量遍历、GPU 运行或源码正确性证明；没有阅读搭档或本轮其他报告。分支只提供状态差异证据，不讨论继承或 dispatch 组织。
以下“不能删除”指删后会放开不同设计；相同事实已有唯一来源时删重复，不要求每层复述。

## 现行能力：已经足够，但各字段不能互相冒充

| 载体 | 本轮可直接使用的能力 | 不能从它自动推出 |
| --- | --- | --- |
| [DOC1:43](../rules/1.contract.algorithm_requirement.md:43)、[269](../rules/1.contract.algorithm_requirement.md:269) | 调用者输入、输出、原地关系、算法初末状态；内部 workspace 不是输出 | 每个私有 counter、cache 都必须进 Contract |
| [DOC3:73](../rules/3.design.math_reshape.md:73)、[96](../rules/3.design.math_reshape.md:96) | `symbol.meaning/retention` 登记逻辑状态，`math` 定义初值、未写区域、消费者和终点 | `keep_value` 等于全局分配，或等于每次覆盖全部槽 |
| [DOC3:145](../rules/3.design.math_reshape.md:145) | 整个对象的 `binding` 可引用已定义提供者；属性引用只沿用属性 | allocator 函数名已给出状态定义、物理别名及完成顺序 |
| [DOC4:191](../rules/4.design.work_schedule.md:191)、[266](../rules/4.design.work_schedule.md:266) | 相关 `schedule.strategy` 给外部供给、共享身份所涉及的调用前提与复用条件 | 当前 stream 的设备顺序自动约束所有 host store 或其他 stream |
| [DOC4:213](../rules/4.design.work_schedule.md:213)、[309](../rules/4.design.work_schedule.md:309)、[315](../rules/4.design.work_schedule.md:315) | `refill` 定义计数协议；交接匹配同值同版本同区域；存续覆盖访问和在途操作 | poll 退出等于最后使用，`initialize` 等于每次 memset |
| [DOC5:178](../rules/5.design.tile_pipeline.md:178)、[185](../rules/5.design.tile_pipeline.md:185) | noun 的 `describe` 绑定本地对象、来源、版本、区域和原语保证；已有动作间必要边仍用 `waiting` | 同名 noun 具有全局分配身份；短说明可以取消必要边 |
| [DOC6:5](../rules/6.realization.md:5)、[159](../rules/6.realization.md:159) | 只补上游尚未表达的实现兼容条件；删无设计依据的源码偏好 | 把状态过程搬进 `choice` 就完成了压缩 |

`retention` 的实例不是封闭的 invocation 枚举；但仅写 `keep_value device_module` 也不建立跨请求物理地址或同步协议。
当前 [DOC5:355](../rules/5.design.tile_pipeline.md:355) 的 `storage` 是 worker 片上存储；无需伪造一个跨请求 worker，把全局 cache 塞进这张表。

## 最小闭合需要什么

1. **对象与共享范围**：调用者对象直接用原名；私有对象只在实际定义处命名一次，明确哪些请求共用同一 backing。无需登记每个 Python 容器。
2. **初态与值来源**：区分算法输入、首次准备的零、上次消费者留下的零、允许保留的有限旧值。只保留消费者真正依赖的性质。
3. **本次更新范围**：哪些区域重写、哪些不写、哪些虽无数学贡献仍实际读取；由 `math`、target 与动作描述分担。
4. **所读版本**：是本次生产值、原地址已有内容，还是 host 最终实际观察值；不要求为无关旧值标出精确请求编号。
5. **最后使用**：覆盖全部实际读写和在途操作；已有 handoff/动作依赖可推出时不重复加边。
6. **下次复用**：只说明源设计成立所需的调用顺序或残留性质；来源未给同步时记录限制，不补 generation、lock 或新安全承诺。

这不是六个新字段或一张每个对象都必填的清单；它是本次删除测试。原表只剩“次序＋写集”时，第二、第四项会实质丢失。

## GDN：递推状态与可复用工作区是两回事

[root:43](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml:43) 的 `initial_state` 与 [123](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml:123) 的 `FinalState` 已闭合算法状态：读入 BF16、返回可选 FP32，输入不原地更新。
`core_attn_out` 的 active prefix 是已有可见写回；它与 `Out` 的非别名关系也已在 Contract。不能用一个笼统“state”合并这些身份。
[源码:3254](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu:3254) 则是私有 `ws_cache`：容量够且设备相同才复用，prep 重写 chain 所消费区域，分配本身不提供跨 stream 排他。
本例保留共享 backing、prep→chain 的当前值、顺序复用前提；算法结果、工作量及并行不会因删掉 `new torch::Tensor`、静态析构规避等过程而变化。
真实可缩短处是 [schedule.strategy:379](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml:379)。保留全部事实的现行合法替文如下：

```yaml
strategy: >-
  One grow-only workspace is reused by calls ordered on the same CUDA stream.
  Prep rewrites every consumed handoff slot;
  cross-stream or reentrant reuse has no source guarantee.
```

归一化字符 **207→175**；共享身份、所读版本、最后消费前不可覆盖及来源限制不变，也没有新增等待。仅计这一处，不外推整体节省比例。
[direct_qk_ordered:259](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/direct_qk_ordered.yaml:259) 仍预留 `SU*(11264*HV+32768*H)+512` 字节，其中 Q/K image 不再读写。
该未用区应判为**来源选择、按显存预算条件保留**：源码未证明算法、合法访存或复用依赖它；不能只因字节数大就判核心必留。
若正在比较实际峰值，保留一句“direct reads 未缩减分配”，避免把流量下降误报为显存下降；否则可删未用区的精确大小。`perm` 的实际读取范围及存活条件另行保留。
grow-only 的历史最高容量同理；无主要占用或成本依据，不把整个扩容过程升级成 Design。l2norm 稳定指针的编码成本不能用于当前 `false` 域的证明。

## FA4 decode：首次零、有限旧 payload、每次清零不能互换

[root:27](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/template_fa4_decode/root.yaml:27) 已把 caller 提供的 `workspace_buffer` 列为输入；它不是算法递推状态，也不是返回结果。不能因它叫 scratch 就删掉真实接口。
[materialize:164](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/template_fa4_decode/root.yaml:164) 对空 split 只写 `M=-inf,L=0`，不写 O；[normalize:183](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/template_fa4_decode/root.yaml:183) 却读取所有 O 槽并乘权重。
源码 [early exit:1138](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/template_fa4_decode/source_gqa_decode_paged.py:1138) 与 [combine:503](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/template_fa4_decode/source_gqa_decode.py:503) 支持这个区别；[输入构造:83](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/template_fa4_decode/source_input.py:83) 建立初始零。
最小契约是：O 使用 caller workspace 的 FP32 视图；活动槽本次写，空槽保留已有有限值，combine 实际读取它们；首次准备负责未触及槽的有限性。
删除有限性会允许 `0*NaN/inf` 污染输出；改成每次清零增加写入工作；改为完全不读空槽则改变求值/访问路径。三者不是同一种精简。
现有 `math + Opartial.symbol.retention + schedule.strategy` 已够；[DOC3:108](../rules/3.design.math_reshape.md:108) 和 [DOC4:317](../rules/4.design.work_schedule.md:317) 已明文支持“不全写、已有初态不重复 initialize”。
不应增加 `handoff.Opartial.initialize: 0` 来假装首次准备；也不能仅凭首次零就保证同一 allocation 换 shape/视图后仍满足每个空槽的有限性。
[workspace 切片:230](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/template_fa4_decode/source_operator.py:230) 给实际 O/L/M 范围；这些可读容量要保留。额外未消费 m 区只支撑当前入口兼容，不自动成为主要资源设计。

## QSA 与 MLA：私有调度状态不需要算法输出

QSA [strategy/refill:189](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/template_qsa_prefill/root.yaml:189) 已写初始 ticket/base=0、每个串行 eager 调用消耗 `W+P`、host base 同增、无 counter reset。
[operator:104](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/template_qsa_prefill/operator.py:104) 证明 ticket/base 与同一个 workspace entry 关联；[157](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/template_qsa_prefill/operator.py:157) 推进 base，[kernel:327](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/template_qsa_prefill/qsa_kernel.py:327) 领取相对编号。
保留初值、signed-int32 差值、每 CTA 首次 terminal 停止及 `W+P`；它们决定工作覆盖、退出和跨调用复用，不能删作 host cache bookkeeping。
用一个短 `refill.atomic_counter` 定义共享 pair 与领取公式，再由 `schedule.strategy` 限定调用顺序即可；无需给 Contract 增加 ticket 输入输出。
该结论限既有串行 eager 说明；固定 replay 参数不意味着 host base 每次重新执行，不能据此宣称 replay 安全。
MLA [refill:343](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/root.yaml:343) 则用私有 uint32 queue：`Item=P+pop`，总共 W 次 pop，返回 `W-1` 的 popper 归零。
[源码:1461](../../../phase_grammar/sources/kernel/extract_mla/solution.py:1461) 确有末次 pop 复位，[1730](../../../phase_grammar/sources/kernel/extract_mla/solution.py:1730) 只在创建时清零；queue 与 caller Workspace 不是同一对象。
两种协议都有既有 `refill` 入口；把它们统一成“每调用计数器清零”会增加工作或改变所读状态，不是少写一句等价说明。
QSA 还存在先 raw-load 后 mask 的 [尾段读取:885](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/template_qsa_prefill/qsa_kernel.py:885)：合法 backing 范围不能只按 Count 的数学值判断；被丢弃字节无需因此初始化。
但源码分配的额外 128 words **未证明在固定模板域必需**：末块只有六行、MaxE 按十行留空间。保留真实访问范围，不把这个预留常数当硬件或算法义务。

## MLA 几何表：缓存存储不等于缓存 metadata

[Geo:299](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/root.yaml:299) 从当前 metadata 定义，Contract [114](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/root.yaml:114) 已排除旧数值延续；不必再把 SeqLens/CuQ/CuKV 复制一套跨调用输入。
共享 d_table 每次被上传；[源码:1895](../../../phase_grammar/sources/kernel/extract_mla/solution.py:1895) 与 [2011](../../../phase_grammar/sources/kernel/extract_mla/solution.py:2011) 明确完整上传后主 grid 才启动。
可把 host 分配/拷贝叙事收成“shared Geo buffer contains this call's table before main starts”，保留原排序定义及录制绑定/实际执行读取内容的区别。
其影响是当前工作集合、工作顺序、访问版本及启动重叠；重建必须保留，容量扩成 `2*items+4096` 的具体策略没有独立的一阶依据，可以删除。
共享表只有在旧消费者结束后才可安全覆盖；源码只显示 upload→main 的等待，不能把它反推成任意请求并发时的 main→下一次 upload 保证。
这里需要的是短的有效调用前提或来源限制；不为补安全创造跨调用事件，也不把所有元数据缓存做成新模块。

## MoE：零态闭环、module tables、ready 标记分别定义

[init_no_scan_direct:144](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/init_no_scan_direct.yaml:144) 定义 RouteAccum/Arrivals 初零、原子贡献、末到者读出、已消费行归零，未用行不写。
[workspace:941](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_solution.py:941)、[971](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_solution.py:971) 是首次零；[kernel:2241](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:2241)、[2254](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:2254) 在读出后归零。
保留“首次建立 R*H 个 FP32 零及 T 个计数零；有序完成保留零态供下次使用”，允许 route 内容变化；删 reset 会改变数值/末到者，改成每次全清又改变工作量。
数学初终关系已有 `math/symbol.retention`；`direct.schedule.strategy` 只补同一 workspace entry 的身份和复用前提。ABI cache 命中/失效不能替代这份零态条件。
[源码:1507](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_solution.py:1507) 按 shape/configuration 取 workspace；不同 ABI 可共用它。完整字典构建和 Python 引用计数过程可删；未使用缓冲只在主要占用有依据时保留总量。
E2Table/ScaleTable/ScalePairTable 是 Design 保存的格式常量表；已有 [keep_value device_module:52](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/init_no_scan_direct.yaml:52)，源码 [116](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:116) 明确不含问题数据。
`_tables_ready` 是跳过建表的控制状态，既不是表内容，也不是 workspace 清洁证明。[ensure_tables:1005](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_solution.py:1005) 在全设备同步后才发布 ready。
保留表值有效、ready 发布时点和相应工作是否执行；把全设备 host 同步等同 `tables.kernel_done` 会改 host 时序。具体 API 由 realization 绑定，容器操作可删。

## Sparse gate：实际观察不是本次最终值，也不是最后使用

[math:124](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/probe_hidden_packed.yaml:124) 将 R 定义成最终扫描实际读到的正值均值；它不是八个本次最终 ProbeRatio 的完整平均。
源码 [gate:5561](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_sparse/dsv4_kernel.cu:5561)、[arm/poll/scan:5588](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_sparse/dsv4_kernel.cu:5588) 使用一份映射 host/device 存储；先写 -2，再有界 poll，另作普通最终读取。
保留共享 gate、PS 个活动槽、sentinel、实际观察定义，以及 poll 结束后 GPU 仍可能写的限制；删除会改变选择、工作量或复用判断，不能写成“读本次完成版本”。
64-float 分配中只有 PS=8 个活动槽；没有证据表明另外 56 槽影响访问、主要内存或并行，精确 64 可以从 [choice:1027](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/probe_hidden_packed.yaml:1027) 删除。
保留映射别名与所依赖的 host/device 可见性机制；最终 scan 的普通读和 poll 的 volatile 读已影响实际观察，不能把它们一并删成笼统“等待完成”。
MoE [verify_only:145](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/verify_only.yaml:145) 有同类旧 mirror 问题，[源码:9481](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:9481) 的 CPU reset 没有等待旧 GPU writer。
这是来源限制：`HostObserved=1` 不证明本次 Clean=1。新增 generation/lock 会更换源协议，不能用它证明现行语言必须扩展。

## 建议改规则的去重与最终边界

| 候选“新规则” | 已有直接覆盖 | 本轮处理 |
| --- | --- | --- |
| 算法初末状态用 input/output，私有 scratch 不做输出 | DOC1:43–44、269 | 已落实，不改 |
| 状态说明初值、未写范围、消费者、最终关系 | DOC3:104、108 | 已落实，不另加初始化/版本字段 |
| 共享状态的外部准备与复用前提放相关 strategy | DOC4:191、264–266 | 已落实，不再追加防御性清单 |
| 初始化不能冒充首次零，最后使用含在途访问 | DOC4:295、315–326 | 已落实，修实例即可 |
| 动作说明来源/版本/区域，已有局部前驱入图 | DOC5:185–200、500–507 | 已落实，不再加 read_version/last_use 字段 |
| 只保留必要实现兼容条件，删除 allocator 过程 | DOC4:408、DOC6:157–168 | 已落实，按删除测试而非按“cache”词判删 |

建议把原行改为：“算法状态沿用输入输出；私有值和调度状态沿用 math/symbol、refill、相关 strategy 与动作定义。只补缺少的共享身份、初态/所读内容及复用前提。分配、替换和缓存过程默认删；实际访问、主要占用或有据成本依赖的事实短保留。”
没有证明需要跨调用 storage/owner 模块，也没有声称跨请求关系已能自动组合验证；这些例子的关键差别已能用有归属的短定义保留。
字段总增删均为零；真正规则增量为零。必要新增的实例内容是缺少的事实，不是为了旧实例错误再加一套规范义务。
