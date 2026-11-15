# MAX B：提取流程核实与文字修订

本报告覆盖 [RAW_INDEX](../RAW_INDEX.md) 的全部 52 个编号：26 个提取侧问题在下文逐项处理，26 个设计侧问题明确交 A。数量是原始观察数量，不是文档缺陷数量。本次没有重新提取算子、批准任何 P1–P6 阶段或改写 medium 结论。

已读 AGENTS、MAX_BRIEF、RAW_INDEX、8 份 ISSUES，完整读 M02/M04/M06/M08 DRAFT，定向核对现行规范及固定源码。先形成判断，再与 A 交换交界问题。M08 是独立 reviewer；其 REDF 判断和新增 cfg88 线索保留为独立观察，不把它改写成 M06 当时已知，也不强制得到唯一的目录数量。

## 本次实际修改

| 位置 | 变化及依据 |
| --- | --- |
| [P1](../../../schema_extract/phases/01_contract.md) | 在算法要求开头说明怎样对齐任务范围材料、签名、生成器和候选说明，把分歧定位到具体对象或条件。针对 M02-04，使用已有 `hardware_specification.execution` 最小登记给定材料明确的执行单元；其余硬件保持占位。 |
| [P2](../../../schema_extract/phases/02_dispatch_draft.md) | 从实际调用、模板实参和函数体建立机制概览；跨多个计算段的候选分别说明已选机制。直接说明 P2 轻量线索与 P4 正式基线定位的关系，区分未研究、已研究但未展开及未来正文计划；reviewer 独立判断调查止点。 |
| [P3](../../../schema_extract/phases/03_root.md)、[P4](../../../schema_extract/phases/04_dispatch_plan.md) | P3 将动作内对象、跨动作接口和说明深度直接导航到 DOC5 主定义；P4 将暂拟名称落实、未来展开与实际机制仍未知分开。未复制 schema 语法。 |
| [REVIEW](../../../schema_extract/REVIEW.md) | 说明另行授权的候选评议/局部片段核对只产生相应范围的结论；正式阶段仍按原完整性和前提放行。明确调用边界是设计所需关系，不能作为源码已提供输入就绪、元数据有效或跨调用复用保证的证据。 |
| [EXECUTOR_ALIGNMENT](../../../schema_extract/EXECUTOR_ALIGNMENT.md) | 集中写明手工推演不以启动旧工具为前提，限域结论不成为阶段批准；明确旧 Runner 仍整体保护硬件占位，尚不支持本次 P1 最小登记流程。 |
| [EXTRACT_SCHEMA](../../../EXTRACT_SCHEMA.md)、[Supervisor 手册](../../../schema_extract/README.md)、[工具说明](../../../schema_extract/tools/README.md) | 连接手工工作与实际执行能力入口；EXTRACT 同步 P1 最小执行单元登记，命令说明仍如实描述现行协议。 |

P5/P6 的独立有效全文审阅、受限写入、十份上限、P6 新终审和按影响返回规则没有改动。独特价值、变体格式、字符串与 ragged 工作表达的主定义由 A 在 schema_docs 处理，本报告不另造一套规则。

P1 执行单元登记是本轮明确授权的流程调整：修改已有字段的写入权限，未新增字段、类型、默认值或引用规则。A 已同步 DOC2、模板 README 和模板工具 README；下游执行名仍必须来自 `execution`。动态 `nSM` 不由这项修法解决，见后文。

## 证据简称

下文行号是本次读取的实际源码位置，不继承原稿中偶有偏移的行号。只读核实没有提供运行结果或硬件正确性证明。

| 简称 | 实际来源 |
| --- | --- |
| G | [grouped FP8 GEMM](../../../kernel_wiki/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/)；`G/U` = `candidate/fp8gg_umma.cu`，`G/B` = `candidate/fp8gg_bind.cpp`，`G/S` = `candidate/solution.py` |
| L | [MLA expanded FP8](../../../kernel_wiki/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/)；`L/S` = `candidate/solution.py` |
| O | [NVFP4 MoE](../../../kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/)；`O/C` = `candidate/nvfp4_moe.cu`，`O/S` = `candidate/solution.py` |

共同规则依据为 common 的“参数变化与结构变化”“选择条件与索引”“局部记录与完整方案”，DOC1 的验收来源记录，DOC4 的执行映射/提交身份/调用边界，DOC5 的动作声明、事件引用与连续序号。相应提取方法原已位于 P1/P2/P4 和 REVIEW；本轮主要缩短定位路径，并将使用反馈中容易混淆的职责接起来。

## 提取侧逐项处理

### 任务与验收材料

| 原始编号 | 核实、归因与处理 |
| --- | --- |
| M02-01 | **输入域尚不完整，不是 nullable/default 字段不足。** G 的 metadata 同时声称 exact upstream API，并明确 shape_range 是动态域材料；本地 reference `expected` 直接访问 `p['out'].shape`，签名却给 `out=None`，生成器只提供 BF16 out。shape_range 可以裁定其声明的动态维度，不能自动裁定未给的上游可选 ABI。继续保留已确认的 caller-owned out 路径，正式共享 Contract 的缺省域仍须有任务依据。P1 新文字用于按对象定位这种分歧；未替任务方补出 `out=None` 的语义或 C alias。 |
| M04-02 | **来源概括不一致，已有明确范围依据可继续。** L metadata 指定 shape_range；该文件列头数 12/24/32/96/128、causal false/true，并在末尾要求 `query_lengths == kv_lengths`。输入生成器的更宽断言、candidate 注释所称 q<k 不能覆盖该声明。M04 的回读纠正有据，不必为同一事实再次索要确认。与 M02-01 同属材料对齐，但这部分已有裁定依据；P1 统一改进流程，未改 Contract 字段。 |
| M04-03 | **外部比较器材料不足，现行来源记录已能处理。** L metadata 保留 default 阈值及 `mutated_inputs.out` 的 mismatch/rel-L2 override；reference 全文是计算、归一化和返回包装，没有给出完整验收路由。candidate 头部的公式不能升级为外部比较器定义。DOC1/P1/REVIEW 已允许在对应 tolerance 保留已知声明与明确限制，不要求无限追索未给的 runner；没有新增字段或放宽数值要求。 |
| M06-03 | **与 M04-03 同组，但已有声明更具体。** O metadata.description 明写 `torch.allclose(reference_float, candidate_float, atol=0.01, rtol=0.05)`，benchmark default 与 `mutated_inputs.output` 的 mismatch/rel-L2 又不同；不能用 schema 默认简写抹掉实参顺序或 override。返回列表与原位 output 的实际检查路由仍未给出。现行 tolerance 来源记录足够，本轮不伪造比较器、不把缺项塞进 `golden_reference.gap`；P1 的原有逐输出门槛保持。 |

### 当前调用与源码保证

| 原始编号 | 核实、归因与处理 |
| --- | --- |
| M02-05 | **源码声称的前提与固定调用材料之间有缺口，未证实为实测 kernel 缺陷。** G/B:69–83 只核六个 weakref 对象身份后使用缓存指针/维度，不能由此推出 tensor storage/metadata 跨调用不变；G/U:1542–1547 缓存策略按地址和 shape 命中，G/U:1723–1759 在 graph/direct 错误后继续回退。已读代码支持这些路径存在，不支持“任何失败都未提交工作”或“同对象天然隔离”的保证。正常成功路径的一次主提交可继续提取，其余范围保留来源限制。REVIEW 新段澄清调用约定不能当源码证据，不改源码或错误 API 语义。 |
| M04-07 | **输入/调用上下文缺项与潜在源码问题尚需区分，不能直接套低层缺陷豁免。** L/S:1646–1681 创建 nonblocking side stream 并 DtoH 当前元数据；1895–1898 上传静态 d_table 并 record event；2010–2011 当前 stream 等该 event。上传→主 kernel 的边有据；在已读 host 路径中，side-stream 读取前的输入生产者完成、下一调用覆盖表前的旧读者完成未有对应证据。`Geo is ready`、固定 harness 注释或 DOC4 调用约定不能补成实际就绪/隔离保证。影响所有使用该共享准备的代表，但内部已确认的工作与计算可继续记录。REVIEW 与 M02-05 共用修订；未宣称发生过竞态，未要求 GPU 测试。 |

上述两项不能一概称为“只是 wrapper”。判定可继续的依据仍是 REVIEW 现有规则：若源声明、生产消费和使用点已经支持完整逻辑设计，而偏差仅在抽象边界外，可记录源缺陷后忠实提取；若需要新造版本、等待、容量或复用策略才能解释，受影响范围仍不能批准。本轮没有用文案预先替两例选择修复方案。

### 机制识别、独特价值与局部记录

| 原始编号 | 核实、归因与处理 |
| --- | --- |
| M02-02 | **已有“决定什么”的规则足够，缺少贴近读者的例子。** G/U:377–396 在已选 abox-on 机制中按当前有效行上取整到 16/32/64/128，容量和 pair 主体不变，actual bytes 跟随搬运区域；这一离散尾块范围可留正文。G/U:1682–1686 的 abox on/off 是另外的机制选择，仍属目录。A 已在 common 加短例；B 不把每种 box 新建完整文件，也不隐藏失败回退。 |
| M02-03 | **分支尺度需要算法判断，现规则支持局部候选。** G/U:336–339、428–511、838–855 表明单/双 issuer 仍处理同一 pair、相同 payload/TMEM 槽和工作序列；dual 分开两次 dot，并用双方 commit 汇合，odd-K 无第二 payload 时第二方仍履行通知。producer、promotion 与输出关系可围绕原 pair 接口沿用，故倾向“局部”，不只是参数。不是因为多一个 role 就必须完整，也不是因为改动短就可降级；本轮未完成全部变体协议验收。A 修 common 判据，P2/P4 继续按实际基线复核。 |
| M04-04 | **边界裁量，不是未知字段。** L/S:698–735、1078–1087、1182–1216 确认 tiny 从 S/V 另算 FP32 分母和 SMEM PV，再由 epilogue 写出；1320–1362 的普通 P/PV 仍执行。M04 的“围绕现有 role/存储闭合”是可保留的局部候选理由，但原 PV 仍在并不能证明主要输出计算仍成立。B 倾向将替换实际 PV/输出链视为有独立价值的候选，最终选集须结合完整基线与域判断；本轮不宣布唯一分类或要求立刻增文件。A 的 common 改文强调沿实际输出来源判断，原 M04 观察不改。 |
| M04-05 | **可表达的局部协议变化，不能只记 depth 参数。** L/S:1329–1330、1361–1362、1392–1412 显示 3 槽直接等 pv_done，4 槽有对应周期的 kv_free；QK/PV/softmax、领取主体保持。容量、环索引、release 来源与跨工作连续性都需随局部记录保存。当前已有 storage/index/waiting 表达；完整相位证明仍是研究工作，本轮不从少量源片段批准全部同步。 |
| M06-02 | **复合问题：已有调查法有用，剩余主要是研究未完成与判断尺度。** M06 正确避免枚举全部 cfg；REDF 的具体输出关系当时未读，不能称资料未提供。B 核 O/C:3948–3987、8233–8261、9164–9192、9330–9338 后确认关键事实；M08 独立补读得到的判断另见 M08-02。mono/BF16 不能整族降为参数，cfg99/44 已含 mono G2 又须与尚未研究的整链区分。P2 改为按实际实例分段写机制，区分未研究、已研究取舍与未来正文；不按规模无限加调查任务。 |
| M08-02 | **独立 reviewer 的完整保留判断有据，不是目录字数规则。** 相对 cfg99/44，cfg99/46 保持 mono G2 主计算，却把 route 独占 BF16 rout→finalizer 拉取归并改为 G2 向 token 共享 acc32 原子归并，再 convert；BF16 边界在 atomic 前（O/C:3975–3981），清零分支及 join 位于 9186–9191、9330，最终转换位于 9334–9338。输出的所有者、冲突、初值、完成与求和次序需要贯通，故 M08 建议完整保留合理。保留其相对最近基线的独立判断，不把 M06 的当时未研究状态改掉，也不以此批准最终数值或全局选集。主判据由 A/common 收束。 |
| M08-04 | **草稿标签与实际已选实例不对应。** O/C:8233–8261 的 cfg44/46 属同一 mono G2 系列；cfg99/44 已承载一段 mono。把“单体 FP4、BF16/warp MMA”整行称未研究，容易同时重复计数和漏掉主协作。P2 增加跨多个计算段逐段标明实际机制的写法；没有规定新的方案表字段，没有要求 WS×mono 全配对展开。 |
| M08-06 | **独立调查发现的真实缺项，原有止点规则本已要求处理。** O/C:9300 排除 cfg88 的独立 act，8031–8060 存在 FUSE 实例，5270–5313 将 fixed-slice 归并/activation 交最后到达者，并置于 s_epidone 发布后。该线索不能被“已选 split-K”自动覆盖，也不因 MEGA 已有 ticket 就成为已知组合。初始化、全部 partial 可见性和后继接口尚未完整核实，保留具体研究任务，不先定新文件数。P2 reviewer 文字明确调查记录不限定独立发现或止点。 |
| M08-07 | **源码 selector 算术与任务/设计保证不同，现规则足够。** O/S:1260–1273 用 `ceil(R/bm_g)*per_unit` 决定是否免去 route_hist readback；masked/empty 当前路由仍使实际有效工作少于这一值。草稿示例至多支持选中 cfg81 的实例说明，不能作为所有调用实际非空工作数的下界。common 已区分 selector、when 与实验覆盖；本轮未把它修成新门槛，也未要求在限域 P2 试走中完成全部 P3 when。 |

### 阅读、记录与局部表达

| 原始编号 | 核实、归因与处理 |
| --- | --- |
| M04-01 | **输入注释陈旧，medium 已自纠；不是应新增设计节点。** L/S:1646–1681 的 host 几何准备、2008 的 grid 与 1450 附近 pop server 不符合头部旧概览；实际 softmax/tiny scratch 也需以使用点核实。P2 明确由实际调用/实例/函数体建立机制图，入口注释作导航。没有要求每轮逐行重读整个文件。 |
| M06-04 | **同类注释陷阱，已局部解决。** O/S:798–838 有默认返回 99/97；G2 选择位于 1308–1309。O 的 MEGA 实际 `p.I >> 5` 与旧 I/64 概括不同。实际实例与主体算式优先，本轮不继承性能或 bit-exact 注释为验证。与 M04-01 共用 P2 修改，不把其余注释一概认定错误。 |
| M04-06 | **现行表达已足够，主定义例子由 A 改进。** DOC4 已允许 schedule.strategy 描述运行时排序键；将真实 ItemGeo 条目作为 item 轴、写清条目到原坐标的映射与 LPT 排列，就无需把 kv_tiles 冒写为 partition 轴。L/S:1782–1820 建实际条目，1822 起排序，2008 决定提交数，支持这种区分。A 在 DOC4 给现有表达示例；B 不在阶段页复刻语法。 |
| M04-08 | **已通过回读解开的阶段导航问题。** P2 需要机制线索和计划，不要求不存在的完整基线已经可引用；P4 才落实正式记录，依赖未来正文的条目仍为计划。P2 第 6 项与 P4 第 5 项现在直接连接这两个时点，不提前增加 Design 写权限，不造空 provider YAML。 |
| M08-01 | **试走输入与正式准入不同，原规则无工具硬阻塞。** 所给 M06 DRAFT 未完成 P1，本次明确只做候选评议；不能转为正式 P2 approve，也无需借机补做完整 P1/P3。REVIEW/EXECUTOR_ALIGNMENT 明确限域结论与正式批准的区别；P1→P2 正式前提保留。 |
| M08-05 | **片段正确性与完整性应分别判断，既有语言能表达。** O/C:4628–4660、4950–4968 支持 operand 环的 g+3/g+2 复用距离，DOC5 flatten 保留跨 item 接续。两条 released 边不能证明 prepared、accumulator/epilogue、CPHOIST 或 TSTORE sink 已完整。REVIEW 明确片段核对的实际结论范围，P3 导航到动作声明；未以缺完整 action 表为限域任务追加 P3。 |
| M02-06 | **已自查解决，现有 store/release 足够。** G/U 的 read-wait 与 full-wait 分别保护 staging 可覆盖与最终 global 完成；DOC5 的 OSlot 例子已提供相应表达。无需新增事件种类或拆完整设计，尚未展开的 full/tail 条件等待仍是该草稿后续工作。本轮不为已解决概念另添规定。 |
| M06-05 | **现行规则应用已解开，不是 runtime 条件统一合法或统一非法。** O/C:4855–4915 的 CPHOIST 决定 scale 搬运相对 accumulator 退役的策略，应按选定配置与局部差额处理；g 跨 item 可用已有 flatten。Clean 决定本次验证后是否需要动作，与此不同；但它不因此自动取得 host 条件提交语法。A/common 处理判据，B 保留 ZERO2 的独立议题。 |

## 固定表达与政策边界

### M02-04：执行单元已用流程修复，动态设备数量仍未解决

原卡点有两部分。DOC4 的 `worker.unit` 和 DOC5 的成员执行名必须来自 `hardware.execution`；提取 P1 原先却要求全区 starter 占位原样，starter 只有 `Unit: null`。M04 未单列地认为豁免足够直接写 CTA，与 M02 的犹豫不同；跳过检查不能证明名字已绑定。

本次采用无需改字段的路径：允许 P1 在 DOC2 已有开放具名映射中最小登记给定材料明确的执行层级。CUDA 源的 `__global__`/block 内共享存储、`threadIdx.x`、`warp = tid >> 5` 及当前 launch 提供执行者和分组的来源线索；登记不把具体 block 大小变成设备上限，也不推断其他协作能力。P1 人工核来源，其他硬件字段保持占位；后续新增执行名回 P1。主代理已明确确认这项流程调整在本轮授权内，A 同步 DOC2/template，B 同步 EXTRACT/P1。

当前 Python 仍有两个实际限制：`artifacts.py:463–466` 要整个硬件区等于冻结 starter，并在 P1 保护硬件；`_without_p1_fields` 也未排除 execution。已在 EXECUTOR_ALIGNMENT 单列，不改 Python，不称新流程已能执行。

**剩余 `nSM` 不是同一个问题。** G/S:126–129 从设备属性取得 `multi_processor_count`，`candidate/fp8gg.cu:340–344` 用它决定 grid。DOC2 将设备级数量/容量放在 `wiki.hardware`，DOC4 要设备规模来自绑定资料；最小登记执行名不提供这条数值来源。不能把 nSM 改成外部算法输入、写死成某设备数值，或塞入 execution 摘要冒充容量绑定。

需要区分的后续方案：

1. 保持现有来源规则，通过既有 `wiki.hardware` 取得与目标设备匹配的数量/查询依据。这是补材料或调整硬件资料政策，不必新增字段；本轮没有该绑定，也不启动调查。
2. 允许固定实现提取在既有 mapping 说明中以源码的运行时设备查询定义数值符号，要求查询来源与使用范围明确、没有数值默认。字段名可不变，但会放宽 DOC4 的数值来源/绑定规则，须作为语义选择交用户，不能仅称措辞澄清。

因此 M02-04 是“已解决执行名登记 + 仍待决定设备数值来源”，不是全部都需要新增字段，也不是全部已经解决。

### M06-01、M08-03：ZERO2 的 host 条件提交

两个独立角色遇到的是同一具体反例。O/C:9471–9491 先实际 launch clean graph；host 将 pinned word 设 -1、有限轮询，只有观察值不是 1（dirty 或 timeout）才 launch full graph。full 自含另一次 verify（9157–9159）；outer capture 时先记录 clean，再记录 full，执行发生于随后 replay。

已有 DOC4 能表达固定提交、启动等待、工作域为空以及手工登记多次 verify 的不同 step 身份；DOC5 的 if 能表达 role 内动作条件。这些能力不足以直接表达本例某次 host 决策后整组提交不存在：

- 固定列 clean/full，再在说明写“仅 dirty/timeout”，改变了 launch 的既定存在性；`start_after` 只约束何时开始。
- 提交 full 后让工作为空，与源码不提交 full 不同；用 no-op 取消动作事件也不成立。
- 把 host 伪装为 CUDA worker，或把同一次轮询结果拆成数份 `when` 设计，没有解释该 host 机制如何作本次决定。
- 普通 graph cache 是提交方式；ZERO2 的验证产物与分支决定本次输出生产和实际工作，不能降成无关包装来消除问题。

B 与 A 均未找到满足现行字段限制的完整表达。未更改 `flow.launch`、step 唯一身份、事件、if 或说明字符串的语义；M06 的 worker 疑问与 M08 的独立复核分别保留。这不是要求重跑工具或补 GPU 证据。

可供用户选择的方向：

| 方向 | 具体内容与影响 |
| --- | --- |
| 保持语言范围 | 若当前任务允许，完整展开普通机制，在目录如实保留 ZERO2 的已研究独立机制与未展开原因。无需字段变化；不能在任务明确要求完整 ZERO2 时用此方式宣称交付完成。 |
| 正式增加条件提交能力 | 在 launch 层定义条件存在性，或定义可容纳 host 决策的外层控制结构。需一并说明条件来源及求值时点、dirty/timeout 的含义、未提交时的 step/事件实例、输出生产者覆盖、handoff/等待条件和 graph capture/replay 身份。只加一个没有这些关联语义的布尔字段不够。 |

本次只验证了变量提交的表达反例，未完整审 ZERO 的 flag 发布、全部并发调用或异常前缀。因此任何新表达都不能附带推定输入已就绪、pinned verdict 不陈旧或跨调用天然隔离的保证。

## 设计侧 26 个编号的责任映射

以下均已读原始问题，主处理与作者侧修改交 A；B 不在 EXTRACT 文档替普通设计补材料或改语义。完整处理记录见 A_TRIAGE.md（由 A 生成）。

| 原始编号 | B 的交界判断与责任去处 |
| --- | --- |
| M01-01、M03-03、M07-01 | 普通设计的硬件能力/容量/实现栈资料不足；不能套提取豁免。A 处理设计推进与资料边界。 |
| M01-02、M07-07 | split-KV workspace 容量与可否内部申请不足，首次清零也不等于每次保证。A 保留输入事项，不补 scratch 下界。 |
| M01-03、M05-01、M07-03 | workload 权重/目标不足；M05 还含硬件相对代价。A 处理，不能把自选焦点写成已知最优或新增 workload 固定字段。 |
| M01-04、M03-01、M07-02 | DOC1 默认有限输入已存在，M07 的 NaN 输入疑问不能继续当作缺输入；M01 还涉及 retention，M03 仍有有限输入导致 FP32 中间溢出的实际问题。A 分开核实，不合并抹掉差别。 |
| M01-05、M03-05、M05-06、M07-05、M07-08 | 现有 serial/成员作用域、数学 matmul、prepared/released 与索引窗口已有表达；具体实现保证仍须有据。A 在 DOC5 处理例子与导航；B 的 P3 只链接动作声明主定义。 |
| M03-02 | online U 的溢出反例属于方案数值推导不足，不能通过文案或字段修掉；A 记录算法结论。 |
| M03-04、M05-07、M07-04 | 输入重排、输出融合与 ragged 两-query 分工是不同的局部/整体边界；A 在 common 核判据和示例，不靠 launch 数或参数名判定。 |
| M05-02、M05-03 | binary64 scale 消费精度与 forward 边界转换分别需要既有数学精度规则、固定接口事实；A 主处理，B 未独立批准该设计计算。 |
| M05-04、M07-06 | retention/invariant/target/storage 与局部对象的放置，部分已由原规则解开；A 用主定义和短例改善，B 不重复造字段。 |
| M05-05 | ragged partition 的表达范围由 A 核；扁平真实 item 轴或有明确定义的上界工作已有表达，不能偷偷改变 partition 的乘积计数语义。 |
| M05-08 | 本轮手工设计原本没有执行器硬前置；A 处理作者入口，B 的 EXECUTOR_ALIGNMENT 提供共用结论范围说明，不称修复了一个实际执行失败。 |

## 验证范围

所有改动限于上述 Markdown 与本报告，保留已有未提交修改。B 开始编辑前的授权文件副本位于 `/tmp/schema_role_max_b_before_iFlhsg`，用于区分本轮增量和此前修改。源码、medium 原稿、inputs、starter、原 YAML/PNG、运行状态及 archive 未写入。

临时文档检查覆盖 B 修改的 9 份说明和本报告：197 个本地链接及锚点可定位，正文无行末空白，报告的原始编号集合与 RAW_INDEX 的 52 个编号一致；授权说明文件的 `git diff --check` 通过。对采集时 `raw.sha256` 的静默校验通过，medium 与 inputs 原始记录未被改写。

本轮仅做源码/规范只读核对、文本差异审阅和临时 Markdown 检查；未运行提取器、schema checker、构建、GPU 或生产测试。没有把命令未执行或旧执行器不支持当作算法失败，也没有把文字修订当成工具能力已实现。最终逐项统一登记、跨文件去重及字段授权复核由第三位 MAX 完成。
