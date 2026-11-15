# A：作者侧核实与文字修订

本报告覆盖全部 52 个原始编号，按原始记录与采集时规则归因。文字修订不表示相应草稿已完成，也不构成提取阶段批准、硬件保证或数值验收。提取流程与来源问题由 B 主核；下表保留交界与各编号的去处，不替代 B 的报告或第三位 MAX 的最终登记。

## 核实范围与修改边界

已通读八份 ISSUES 与 DRAFT、MAX_BRIEF、RAW_INDEX，读取 DOC 0–5、作者入口及 DOC 6 的相关条款；定向复核 grouped GEMM 的 pair/box/dual 路径、MLA tiny 与三/四槽路径、MoE ZERO2 的实际 host 提交。GDN 仅沿输入的固定参考版本读取 oracle，没有用旧 Design 回填新设计。

开工时所有相关正式文档已有未提交修改；A 的起点快照为 `/tmp/role_max_a_before_noivxsrh/`，以下“原规则”均指该快照。本轮保留先前内容，只作增量修订。没有修改原始 medium、inputs、算子 YAML/PNG、starter、代码或运行状态，没有运行 extract、checker、GPU 或生产测试。

已修改的正式文字：

| 位置 | 目的及沿用的语义 |
| --- | --- |
| [common：参数变化与结构变化](../../../schema_docs/0.common.md#参数变化与结构变化) | 用固定容量内的尾块传输说明“决定对象”；用输入重排与输出工作融合说明局部闭合。沿用具体选择、必要联动、边界外主体仍成立的原判据，不改变完整设计上限或目录身份。 |
| [DESIGN §1](../../../DESIGN_SCHEMA.md#1-接收-contract建立设计起点) | 直接导航到默认输入域与计算精度，避免重复寻找或从 dtype 推断 NaN 域；没有改变任何默认。 |
| [DOC 3：符号属性](../../../schema_docs/3.design.math_reshape.md#符号属性) | 连接数值接口、本地对象和工作区域的既有定义，避免“所有名字都登记”与“都塞 describe”两种误读。 |
| [DOC 4：工作分解](../../../schema_docs/4.design.work_schedule.md#工作分解)、[工作排列与分配](../../../schema_docs/4.design.work_schedule.md#工作排列与分配)、[数据块描述](../../../schema_docs/4.design.work_schedule.md#数据块描述) | 给出现有 item 轴的紧凑变长编号、表排序的落点，区分映射函数与实际数组、逻辑产出与驻留容量。未允许新的 partition 域或排序字段。 |
| [DOC 5：describe](../../../schema_docs/5.design.tile_pipeline.md#describe-要说明什么)、[执行模式](../../../schema_docs/5.design.tile_pipeline.md#执行模式) | 标明数学值、坐标、本地对象、存储各自定义位置，给出同步协作 matmul 短例；明确成员来自 worker，而非 role 内每个 warp。动作完成、引用与等待语义沿用原规则。 |
| [DOC 2 开头](../../../schema_docs/2.contract.hardware_specification.md)、[模板 README](../../../schema_template/README.md)、[工具 README：硬件占位豁免](../../../schema_template/tools/README.md#硬件占位豁免) | 按主代理明确授权，与 B 的 P1 政策同步：在现有 execution 中最小登记给定材料明确的执行者，其余硬件占位保留。是提取流程修订，未改变 worker.unit 的引用位置；动态 nSM 来源仍单列未决。 |
| [CHANGELOG](../../../schema_docs/CHANGELOG.md#2026-11-14) | 记录以上说明与流程修订，不声称工具已迁移。 |

DOC 1、DOC 6、OPTIMIZE、IMPLEMENT、根 README 和 examples README 没有本轮 A 增量修改：现有规则或本次交界分工已覆盖问题，没有为逐条回应重复增加文字。

## 证据缩写

- G：[grouped GEMM fp8gg_umma.cu](../../../kernel_wiki/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/candidate/fp8gg_umma.cu)；其余 GEMM 文件在同目录。
- L：[MLA candidate/solution.py](../../../kernel_wiki/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/candidate/solution.py)。
- C：[MoE nvfp4_moe.cu](../../../kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu)，S 为同目录 `solution.py`。
- R：GDN 输入固定的 `01fff90d714abc6e84904ea9c44523ba66d34a0e:kernel_wiki/aka-four-problems-20261019/gdn_prefill/reference.py`，通过 `git show` 读取。该路径不在当前工作树，未伪造本地文件链接。

G/L/C 的行号取当前未修改的源码；正式规则用章节定位。B 主核的来源事实在下表明确交 B，不把 A 未独立展开的源码检查写成 A 已完成。

## 逐原始编号处理

### M01：decode 设计

原稿：[ISSUES](../medium/M01_design_decode/ISSUES.md)、[DRAFT](../medium/M01_design_decode/DRAFT.md)。

| ID | 归因、证据与处理 |
| --- | --- |
| M01-01 | **输入不足，原稿已正确处理。** 输入 wiki 为 pending，只能做所选 64-token、8-warp 路线的需求账，不能据此证明驻留、异步原语或 CuTe 支持。原 DESIGN 推进原则与 DOC 2 占位规则已允许继续独立部分。未新增硬件默认或 require 占位；提取的最小 execution 政策不适用于这里，也不解决容量缺口。 |
| M01-02 | **scratch 与调用边界输入不足。** 输入只有 WorkspaceBytes，没有容量下界；`16,512 * BatchCount * q_len_per_req * Smax` 是该 split 候选的需求。PART 写满消费者区域可避免依赖旧清零，但不能保证容量。原 DOC 4 存续/初始化与 common when 足够表达已有条件；不假定额外申请免费、不缩小共享 Contract。未改文字。 |
| M01-03 | **workload 输入不足，非字段缺失反例。** 原模板入口已明确 workload/目标尚无固定格式。q_len=1/4 在该 query_tail partition 下只改变工作数；长 KV 不单独证明 split 有收益。保留缺失分布、设备并发和提交成本的记录，不制造阈值或 workload 字段。 |
| M01-04 | **已解的 shape/retention/有限域误读，另有数值验证未完成。** 原 DOC 1 默认浮点有限；原 DOC 3 carry_forward 与递推下标不入 shape 已可表达 m/z/u。补 DOC 3–4 导航和 DESIGN 默认域链接。FP8 幅值上界只排除所述量级溢出，不能证明长归约、抵消和 exp 的原 tolerance。未把数值缺口改成状态字段。 |
| M01-05 | **已解，原依赖语义足够。** 在动作确实保证完整 CTA 协作的前提下，全部 serial 的页内与跨页顺序可保护读取和单槽覆盖，无须重复 waiting。DOC 5 新说明只点明这一前提及 worker 成员范围，未增加隐式屏障或强制边。 |

### M02：grouped GEMM 提取

原稿：[ISSUES](../medium/M02_extract_gemm/ISSUES.md)、[DRAFT](../medium/M02_extract_gemm/DRAFT.md)。

| ID | 归因、证据与处理 |
| --- | --- |
| M02-01 | **材料的任务域/接口边界未闭合，交 B 主核。** 生成器总传 out、reference expected 读取 out.shape、较宽签名及 metadata 的 upstream 描述并不自动互相覆盖。现有 nullable/default 能表达已确认的两种接口；不能由 candidate 分配行为替任务补 None 分支。A 不改字段或 Contract。 |
| M02-02 | **已有语义可表达，原提示不够具体。** G:377–396 在固定 abox-on 机制与固定 stage 容量中，用有效行数选择 16/32/64/128 传输 box，联动 expect_tx；这是当前尾块执行。G 的 host abox 开关/创建失败回退另属配置与来源条件。common 增加此类例子，不把所有 descriptor 选择都归一种类别。 |
| M02-03 | **可定位的局部协作差额，不是角色数量参数。** G:336–339、415–523：同一 pair 的两次 dot 从一名 issuer 改为两名，各 commit，empty_bar 汇合数为 2；odd-K 的第二方无 payload 仍 commit。producer、pair 工作序列、TMEM 两块与 promotion 入口保持。就已核差额可作局部记录，须显式保留双完成汇合；多一个长期 role 本身不证明整体重组。common 改进闭合判定提示，未要求先写两套完整 YAML。 |
| M02-04 | **原流程冲突，已修执行名部分；设备数值来源仍未决。** 原 P1 要全区原样占位，原 DOC 4/5 又要求执行名来自 execution；`Unit: null` 不提供 CTA/warp。现按授权调整 P1，并同步 DOC 2/template，使来源明确的执行名在既有字段登记，未放宽 unit 引用。`nSM` 则由源码 host 查询，原 DOC 4 仍要求设备规模来自绑定资料；详见后文 F2，不能说本项已全部解决。B 负责 P1/EXTRACT 的政策正文。 |
| M02-05 | **调用前提与失败语义的来源限制，交 B。** weakref 身份不证明 metadata/storage 不变，返回错误不证明此前没有设备提交，graph/direct 正常同主 kernel 也不消除跨调用生命周期。原 common/DOC 4 已禁止补造保证。保留可确认成功路径与受影响调用范围；没有给缓存路径复制完整方案，也不以 when 新造安全前提。 |
| M02-06 | **已自行解决，保留正面证据。** 原 DOC 5 OSlot 示例已区分 store 的目标写回 done 与 release 的源读取结束。G 的 `bulk_wait_group_read0`/`bulk_wait_group0` 正是该区别。完整 full/tail 条件事件仍未展开；概念解开不等于 pipe 完整。未增加动作词或重复解释。 |

### M03：QSA 设计

原稿：[ISSUES](../medium/M03_design_qsa/ISSUES.md)、[DRAFT](../medium/M03_design_qsa/DRAFT.md)。

| ID | 归因、证据与处理 |
| --- | --- |
| M03-01 | **有限输入误读已解；极端有限值的参考/验收问题真实。** q/gate 默认有限，mapped K/V 的 range 也只保证有限。取 q、K 元素为 BF16 可表示的 `2^64`，单个乘积即超过 FP32 有限上界，max-shift 不能修复已有 Inf。现行 tolerance 没有“双方 NaN 即通过”的许可。需要明确该有效域上的任务/参考处理，不能换为 FP64 后自行宣布遵守原判据。只补默认域导航，未改输入域或浮点语义。 |
| M03-02 | **候选算法有具体反例，非 schema 缺口。** q=K=0，2048 个有效 V 都为 BF16 最大有限值：归一化权重是精确的 `1/2048`，其 FP32 加权和可保持有限；未归一化 U 在累加两份 V 时已可溢出，甚至不必等完整长序列。原 DRAFT 撤回初版论据正确。归一化均值或宽 U 是待分析候选，未实施、未替换原 tolerance，未把它们写成已获修复。 |
| M03-03 | **硬件/DSL 输入不足。** 32 KiB K/V 槽和 12 KiB U 是需求，feature 名不提供原语 shape、容量、完成协议或 CuTe 绑定。原 DOC 2/6 与 DESIGN 已处理此类缺项。DOC 5 同步 matmul 例只说明表达，不提供本案例缺失的硬件证明。 |
| M03-04 | **可作局部变化，已有判据缺少易用案例。** DRAFT 的 PackCache 仅改变既有 Ksel/Vsel 来源、全局物化及交接，MAIN 工作归属、递推和 gate 不变，512 MiB 最低附加流量仍纳入整体比较。common 用此类输入重排解释局部闭合；新增一次 kernel 本身不要求完整身份。若后续改变主要 attention 工作组织，须重新判断。 |
| M03-05 | **原稿“已解”仍残留成员误读。** 原 DOC 5 明定 member 是 worker.unit；此处 worker=一个 CTA，`MAIN[*]` 只选该 CTA 的 role 实例，不枚举四个 warp。两条边不能替代动作内部协作保证。可让 load/Update 明确承担整个 role 的必要完成与可见性，或按真实独立职责划 role 后表达交接；不能靠 serial 标签证明屏障。DOC 5 执行模式处直接连接成员定义；原 DRAFT 保持原样。 |

### M04：MLA 提取

原稿：[ISSUES](../medium/M04_extract_mla/ISSUES.md)、[DRAFT](../medium/M04_extract_mla/DRAFT.md)。

| ID | 归因、证据与处理 |
| --- | --- |
| M04-01 | **陈旧源码说明，已自行纠正，交 B 归档。** 文件头的 fused/device mapping 概览与实际 host ItemGeo、pop server、数值链不符。应从真实入口/模板实例核实，现有提取规则已要求；不在 common 增加源码调查清单。 |
| M04-02 | **输入材料不一致，交 B。** 报告已按材料明确指定的 shape_range 细化域，仍需在 P1 保留 metadata、生成器、candidate 注释之间的差异；不能把更宽签名或样本当任务全集。不是 schema 新增字段需求。 |
| M04-03 | **外部验收来源不足，交 B。** candidate 的阈值公式不是外部 evaluator 定义，metadata 的已知阈值也不能丢。原 DOC 1 允许明确来源记录，未要求复刻未提供 runner。A 不改 tolerance 默认或 gap 职责。 |
| M04-04 | **局部降级理由不足，保留边界判断。** L:1078–1087、1182–1217 显示 tiny 额外计算 FP32 PV/分母并由 SMEM 实际输出；普通 PV 仍执行并不表示原输出形成主体仍有效。common 增加沿实际输出来源判断的提示。倾向作为独立候选继续核，但 A 不声称“只要换输出来源就必有完整 YAML”；完整保留仍须相对基线说明主体变化与独特价值。与 B 交换，未据文件数裁决。 |
| M04-05 | **协议有局部新决定，不能只写 depth 参数。** L:1329–1330、1361–1362、1393–1414 的四槽 kv_free 与三槽 pv_done 并非同一循环周期的替换常量；额外通知/等待须进入局部差额。已有 QK/PV/领取主体保持是局部候选的依据，完整跨工作相位证明仍未完成。common 闭合说明沿用此判断，不新增同步字段。 |
| M04-06 | **已有表达可用，定位成本真实。** 原 schedule.strategy 已允许运行时排列键，by 则引用 partition 维。DOC 4 增加 ItemGeo 表项编号 + by:item + 明确稳定排序的例子；item 必须有实际工作坐标，准备、上传与复用不能被“LPT”名称隐藏。未添加 sort_key 或把 kv_tiles 伪装成独立轴。 |
| M04-07 | **跨调用/side-stream 来源前提不足，交 B 主核。** 报告能确认 upload→kernel 边，不能确认输入生产者→side-stream 或旧读者→下次覆盖。不能用 strategy 写“Geo ready”就创造缺失保证。保留受影响范围，既不声称已实测竞态，也不假定 harness 总排空。 |
| M04-08 | **原阶段职责可解，仍有导航改善，交 B。** P2 是轻量线索与计划，P4 才复核正式局部记录对实际完整基线的定位。原稿回读后处理正确；B 修阶段提示，A 不把 common 的最终要求扩成 P2 前置或建立假 provider。 |

### M05：GDN 设计

原稿：[ISSUES](../medium/M05_design_gdn/ISSUES.md)、[DRAFT](../medium/M05_design_gdn/DRAFT.md)。

| ID | 归因、证据与处理 |
| --- | --- |
| M05-01 | **workload 与硬件相对代价缺失。** B3 的 FP32 S_entry 上界约 1.03 GiB，不能只凭递推链变短判断更快。原 DESIGN 成本口径已要求纳入物化、准备与调用边界，原稿也保留了反证风险。没有新增字段、假精度路径或性能结论。 |
| M05-02 | **精度选择/来源衔接未完成，不是已证必须改字段。** 输入 scale 明确 binary64；默认 FP32 不由输入 dtype 自动升为 FP64，改变数值的窄化又须显式。R 的 Model.forward 把 q/k/v 转为 FP32，blockwise 函数用 Python scale_factor 乘 FP32 张量；可定位真实消费点。固定 PyTorch 2.9.0 标量输入的具体舍入在本轮未独立核完，不能把 FP32 结果 dtype 当作完整舍入证明。现有 `fp32(scale)` 与 arithmetic 已能写出明确候选路径，仍须按原 oracle/tolerance 核对，不能擅自宣布某种新默认。详见后文表达请求汇总。 |
| M05-03 | **调用边界输入仍需落实。** 输入显式把 Model.forward 与 int64→int32 转换计入 timing；R 的 oracle 则从原 int64 边界求差并 tolist，没有该转换。因此 oracle 不能单独证明新实现可删除固定 wrapper，也不能要求无依据多加一个 launch。需明确任务允许替换的 forward 边界，再用现有 step/动作记录实际转换及成本；A 未改 Contract 或调度语义。 |
| M05-04 | **已解的跨区对象放置，导航已加强。** 原 DOC 3 的 S carry_forward、DOC 4 输入块和 DOC 5 storage 已足够：S0 可为 invariant，S 不是不变输入，O 的整段 target 不等于片上输出预算。DOC 3–5 连接此例，无需“内部携带值”新栏目。 |
| M05-05 | **变长域写法未选定，现有字段可继续。** 原 worktile_count 是 partition 各轴工作数之积，不能直接将依赖 b 的 Nc_b 当统一维度大小而不说明域。DOC 4 新例用 Items=sum_b Nc(b)、唯一前缀映射恢复 b,c；映射函数不自动物化数组。表准备与上界空工作有不同成本，须实际选择。没有新增 ragged/lookup 字段或放宽 size 的作用域。 |
| M05-06 | **动作粒度已大体解开，完成依据仍缺资料。** Advance 引用上游递推并保留真正的输入槽交接即可，不必把 DK 归约每项拆开；语义保证可写在草稿，实际 CuTe 绑定未完成时不能称为完整可实现交付。DOC 5 补名字位置与同步协作例，沿用 DOC 6 删除测试，不捏造 API 或扩张 serial。 |
| M05-07 | **整体候选有实质依据，非提交数判断。** B3 的独立 READOUT 变为 STATE 内逐 chunk 计算，取消 S_entry/U 的全局存活与交接，输出的独立工作轴、生产者及主链改变。common 用该类变化说明须重新贯通为整体候选；共享大部分公式或差额很短不消除其意义。未要求 B2/B3 必须同时交付完整文件。 |
| M05-08 | **已解的已知工具边界，不重报为发现。** 原 DESIGN 与模板允许中间稿及独立核对，EXECUTOR_ALIGNMENT 只限定实际工具能力。未运行或迁移工具；不为旧 checker 使用旧字段。 |

### M06：MoE worker

原稿：[ISSUES](../medium/M06_extract_moe/ISSUES.md)、[DRAFT](../medium/M06_extract_moe/DRAFT.md)。

| ID | 归因、证据与处理 |
| --- | --- |
| M06-01 | **真实 host 条件提交表达阻碍，合并 F1。** A 核 C:9471–9491：先提交 clean，非 capture 时 host 有界等待，只有观察值非 1 才提交 full；capture 分支只记录两图。现有 launch/start_after/role if 不能直接承载这层存在条件。未改控制流语义，B 同结论。 |
| M06-02 | **调查未完成与分支尺度混合，分别保留。** REDF 的已知输出归并变化支持完整候选，M08/B 补读后消除了部分数值疑点；不能继续笼统写“未读 epilogue”。mono/BF16 与 cfg88 的去处仍要看实际机制与已有代表覆盖，不按 cfg 数或未读代码推定新方案。common 改局部判据提示；P2 止点与后续定向调查由 B 处理。 |
| M06-03 | **已知多层验收声明及缺失路由，交 B。** allclose 的实参顺序、default 与 mutated_inputs.output 的覆盖阈值都需保留；不能压成默认 reference-relative atol/rtol。原 DOC 1 来源记录可表达，不新增 comparator 字段或把缺口放 gap。 |
| M06-04 | **实际实例与旧注释不符，已自行解决，交 B。** cfg99/81 的实际可达性与 MEGA 的 I/32 粒度按调用/算式核，不采纳“override-only”或 I/64 概览。未把历史性能数字当验收。common 不加入此类调查步骤。 |
| M06-05 | **配置与执行边界已读通；连续环也有现成表达。** CPHOIST 选择搬运相对 accumulator 退役的策略，不能因在 kernel 内判断就都留正文；当前验证 Clean 是本次状态，另受 F1 的提交表达限制。g+3/g+2 的 flatten 接续适合不重置的环，未新增事件种类。common 新尾块例服务同一“决定对象”原则，不反推所有运行时判断都是配置。 |

### M07：prefill 设计

原稿：[ISSUES](../medium/M07_design_prefill/ISSUES.md)、[DRAFT](../medium/M07_design_prefill/DRAFT.md)。

| ID | 归因、证据与处理 |
| --- | --- |
| M07-01 | **硬件/DSL 输入不足。** 已给执行词汇足够写 CTA/warp 草稿，但 72 KiB SMEM、16 KiB 累加器的需求账不是驻留、异步或低精度原语证据。按原推进原则保留未绑定范围；同步 matmul 例不替本题补能力。 |
| M07-02 | **原稿归因为输入不足不成立，应纠正为未定位默认规则。** 原 DOC 1 字段表与 range 段两处已写浮点默认有限；query/kv_cache 没有 range，因而不含 NaN。out 明许任意旧 bits 且不读取，与算术输入不同。原稿的有限 FP8 范围分析可针对现有输入域继续，但仍不是 tolerance 证明。只加 DESIGN 默认域直达链接，不请求任务方重复批准 finite，也不放宽/收窄 Contract。 |
| M07-03 | **优先 workload 未给定。** 明确自选设计焦点后可比较四类情景，不能把它写成用户权重或精确 dispatch 阈值。原模板已说明所需材料及尚无固定格式；不增 workload 字段或要求四类 shape 各建完整节点。 |
| M07-04 | **不能按参数名称判断，现有工作域表达足够。** 全局 token tile=2 可能跨请求，原“共享同一 KV”的入口不再成立；请求内紧凑 item、矩形空工作或新任务表各有不同实际差额。DOC 4 新例帮助落实 b,c 映射；common 要沿改变后的工作归属判断局部/整体。未选具体组织前，不强裁完整文件数量。 |
| M07-05 | **matmul 与硬件指令的误读已解，粒度提示已补。** 原 DOC 5 已允许同步协作实现矩阵数学；现增加直接示例，内部线程归约/同步无需机械拆分，实际发布或独立交接仍须表达。没有提供 FP8 Tensor Core 能力或凭空证明七 warp 原语可实现。 |
| M07-06 | **多种名字混淆与片段未补全。** 原 DOC 3 数值接口须登记，原 DOC 5 本地地址/packing 可由动作说明，跨动作输入仍须唯一、就绪；现用表连接这些位置。原 offset 已明定数学偏移列表，DRAFT 自行删除源切片字符串是正确修正。m/l、U 及跨动作输出寄存器应按实际存活补账；普通地址/标量临时量不逐项列 storage，不能以“本地对象”隐藏已有数学接口。未新建对象表。 |
| M07-07 | **scratch 与计时输入不足，与 M01-02 同组。** `4*S*Hq*(D+2)` 是候选容量，不是 WorkspaceBytes 的保证。给出单独调用条件也不改变完整任务覆盖义务；不能假定内部申请免费或首次清零保证所有重入。未改字段。 |
| M07-08 | **已自行解决，现有窗口与依赖规则有用。** QK 等 load(i)，load(i+2) 等 PV(i)，serial 保护 ScoreSlot；N=1/2/3 的手推只证明已写片段。原声明窗口可排除超域移位，域内实际缺失前驱仍是错误。未新增 allowed 或重复等待。 |

### M08：MoE reviewer

原稿：[ISSUES](../medium/M08_review_moe/ISSUES.md)、[DRAFT](../medium/M08_review_moe/DRAFT.md)。

| ID | 归因、证据与处理 |
| --- | --- |
| M08-01 | **本次手工任务与正式阶段输入不同，已按授权处理，交 B。** P1 未通过不能形成正式 P2 approve；本次又明确无需补完整 P1/P3，所以候选评议可继续。未为试走改变正式机器批准协议。 |
| M08-02 | **有据的独立复审，与 M06-02 合并。** M08/B 的 C:3948–3987、8234–8262、9164–9192、9329–9338 证据区分了 route 独占 rout+finalizer 与 G2 向 token acc32 原子推送+convert；相对 cfg99/44 可隔离 REDF 自身变化。完整候选建议有据，不因差额短降为局部。common 提醒沿实际主体判断；A 未宣布唯一完整选集。 |
| M08-03 | **独立发现同一 F1 条件提交阻碍。** 与 M06-01 保留两个来源编号，不以二者表述相近删掉 reviewer 的独立确认。原 waiting 只给时点，不提供不提交语义；未新增 guard。 |
| M08-04 | **worker 机制概览缺项，交 B。** cfg99/44 已包含 mono G2，整族“单体未研究”标签掩盖已有覆盖；也不能由已有 mono G2 宣称整条 mono/BF16 主线都已收录。分清真实 G1/G2/输出实例后再判断完整方案，非字段问题。 |
| M08-05 | **片段边界，现有规则有解，交 B。** 两条 released 边只说明 operand 环复用；不能推出 prepared、accumulator、CPHOIST、TSTORE 或跨 step 交接完整。本次是 P2 局部示范，无需因缺完整角色表补做 P3，也不能批准完整 pipe。 |
| M08-06 | **新增实际调查线索，交 B。** cfg88 的 FUSE last-arriver 改变 activation 生产与跨 item 进展，不能因已有 split-K 代表就忽略；当前规则已要求沿真实输出、完成与生命周期继续有限调查。不是缺少新字段或要求穷举 cfg。 |
| M08-07 | **selector 算术与保证的边界，交 B。** S 的 G2 下界用于 selector 决策，不能成为含 masked/empty 路由的全部输入实际非空 tile 下界。原 common 已区分 selector、样本和 when；后续按实际 counts/map 固定机制及必要条件，A 不增加重复规则。 |

## 保留的字段、语义与表达请求

### F1：本次 host 观察决定是否提交后续计算

原始编号：M06-01、M08-03。

**原规则与真实反例。** DOC 4 的 launch 条目枚举一次实际计算提交，每个 step ID 恰出现一次；start_after 表示额外启动等待，说明字符串不提供条件删边。DOC 5 的 if 只控制 role 内动作。C:9471–9491 中 clean graph 已实际提交，host 将 pinned verdict 初始化为 -1 后等待；观察到 1 即返回，0 或 timeout 才再提交 full graph。full 自带第二次 verify，不能把两次求值合成同一 step 身份。outer capture 路径只记录，不能当场计成已执行。

**现有表达的限度。** 固定列 full 会多报 clean 路径，省略 full 会漏 dirty/timeout；start_after 不使提交消失，零 worktile 也不等于未提交。把 host 冒充 CUDA role、用描述字符串删边或让 when 承担未来 host 判定，都没有现行依据。common 允许本次验证状态控制必要动作，不能据此推导任意层级已有 if 语法。本轮没有找到忠实表达整个机制的合法完整 flow。

**候选与取舍，均未实施。**

- 讨论 flow 级条件提交/分支结构：需同时定义条件的生产、读取与稳定值、分支内提交身份、未发生事件、输出生产者与 handoff 关系。表达贴合反例，但会扩控制流语义，不能只添加一个未定义的布尔字段。
- 讨论有限的带条件 launch：表面改动较小，但若不处理条件依赖与互斥生产者，仍无法完整说明 ZERO2 的输出责任。
- 在当前语言下诚实保留为未展开独立候选，仅交付可表达的其他方案：可作为明确的收录范围决定，但没有解决 ZERO2，也不自动解除用户要求的全部覆盖义务。

交第三位 MAX 复核，需实质规范选择时留用户决定；不能以正文澄清名义实施。

### F2：执行名最小登记已修，动态设备数量来源仍需判断

原始编号：M02-04。

**已完成部分。** 原“硬件全区原样占位”与 execution 绑定冲突，现用原有 execution 具名映射登记材料明确的执行者，保持 DOC 4/5 的引用规则。主代理已明确授权这项流程修订，B 修改 P1/EXTRACT，A 同步 DOC 2/template。并未新增字段或从源码推断额外能力。

**剩余反例。** GEMM `solution.py::_Engines` 从 `device_properties.multi_processor_count` 取得 nSM，`fp8gg.cu:340–344` 据此限制实际 grid。DOC 4 原文要求设备规模数值来自 hardware 绑定资料，DOC 2 将设备级数量归 wiki.hardware；仅登记 CTA/warp 并没有建立这个符号的来源。它不是算法输入，也不能硬编码成某个设备 SM 数。

候选是：由符合现行要求的已绑定资料明确这项查询，或讨论在固定提取中将源码运行时设备查询方法作为受限来源的流程/引用政策。后者须确认是否改变已有来源约束，且不把动态查询结果升级为稳定容量、实际驻留或性能保证。本轮不改 DOC 4 的设备数值来源句；第三位 MAX 优先判断现有字段是否已有足够落点，再决定是否需要用户讨论。不得把它合并成“skip-hardware 已解决”。

### 已有字段足够，或尚未证明需要改语言的请求

| 原始编号 | 判断与可用表达 |
| --- | --- |
| M05-05、M04-06 | item 轴、既有 strategy、明确数学映射与实际准备/交接可表达紧凑变长工作及 LPT。不为此增加 ragged、lookup 或 sort_key。若未来要求 partition.size 直接依赖另一坐标，需另讨论工作数、排序和实例域的语义；当前反例不迫使采用这种改动。 |
| M05-04、M07-06 | 既有 math symbol/retention、worktile 输入/target、storage/view、noun describe 可分别表达状态、数学接口和实现对象，不需要新增 state/对象表。跨动作使用数学数值不能借“本地”免掉接口定义。 |
| M01-05、M02-06、M03-05、M05-06、M07-05/08、M08-05 | 动作完成、serial、prepared/released、release、flatten 与成员引用已有表达；应修动作保证或草稿完整性，不新增事件类别。M03-05 的 warp/worker 成员混淆尤其不能通过改变 `[*]` 意义解决。 |
| M02-02/03、M03-04、M04-04/05、M05-07、M07-04、M08-02 | 属于现有方案身份判据的应用与必要证据，不要求新增配置或模式字段。局部差额应保留数值、生产消费、完成及资源联动；不能为降低完整文件数改标签。 |
| M05-02 | 现有显式 cast 与 arithmetic 可表达选定的 scale 路径；输入或 oracle 的实际舍入仍须核对。本轮没有给 FP64 标量补“默认先转 FP32”或“默认保持 FP64 乘法”的新规则。若第三位 MAX 判定需要统一改变默认解释，那属于语义决定，不能当普通文案修改。 |
| M03-01/02、M07-02 | M07 的输入 NaN 误读由原有限域规则解决；QSA 的极端有限输入与新引入 U 溢出分别是任务/参考问题和算法反例。现有 math、range、tolerance 能记录已确认决定；不得以缩域、更换精度或更换比较目标掩盖它们。 |

## 交界与未完成项

B 主处理 M02-01/05、M04-01/02/03/07/08、M06-03/04 及 M08 的阶段边界、机制遗漏与选择范围；A 已把自己的 common、工作域和动作结论发给 B。M02-04 最小 execution 政策双方同步，nSM 单独保留；ZERO2 双方都未更改表达语义。

第三位 MAX 应独立检查实际增量是否仍沿用原规则，特别是 common 例子未变成机械节点分类、DOC 5 没有创建新的成员事件、提取最小 execution 政策没有暗中解决 nSM。还需合并 B 的材料证据与阶段修订，逐一登记原始编号，而不是用本报告代替最终 ISSUE_REGISTER。

GDN 标量的固定版本底层消费尚未核完：尝试读取 PyTorch v2.9.0 对应源码时连接被重置；本地只有 2.10.0 头文件，未将其当作 2.9.0 证据。已读取的 oracle 消费位置可以缩小后续核对范围，未执行任何数值探针或 GPU 测试。

## 文档核对

- 对开工快照比较，A 的正式增量恰为上表 9 个文件；增量补丁保存于 `/tmp/role_max_a_before_noivxsrh/A_DOC_DIFF.patch`，未将此前修改计成本轮成果。
- 两段新增 YAML 例子均经 PyYAML 普通解析成为映射；既有 text/结构轮廓代码块未变。没有调用仓库 checker 或声称符合完整 Plan。
- 新增/改动文字及本报告的 46 个本地链接均能定位文件及所写章节；限定文件的 `git diff --check` 通过。
- 原始 ISSUES 提取到 52 个唯一编号，本报告覆盖 52 个，无漏号或额外编号。
- `/tmp/schema_role_trial_before_mfYTjS/raw.sha256` 核对全部 20 份 medium/input 文件一致。没有执行数值、构建、GPU 或生产测试；上述检查只覆盖文档与原稿完整性。
