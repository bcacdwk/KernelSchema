# 角色试走综合问题登记

本页是本轮唯一的综合登记；[原始索引](RAW_INDEX.md)、八份原始观察/草稿和 [A](max/A_TRIAGE.md)、[B](max/B_TRIAGE.md) 的独立报告保持原样。C 复核实际规则、增量及关键源码后作下列归因，复核范围见 [C_REVIEW](max/C_REVIEW.md)。固定字段及语义选择另见 [FIELD_ISSUES](FIELD_ISSUES.md)。

## 数量与结论口径

- 8 份观察共有 **52 个唯一原始编号**：设计侧 26 个、提取 worker/reviewer 侧 26 个。下表每个编号恰有一行，复合观察可同时进入多个组。
- 分成 **17 个主题组**以便追踪；主题组数、原编号数都不是已证规范漏洞数。已解误读、输入不足、真实数值反例、源码限制和研究未完成分别保留，不统计一个混杂的“已修复率”。
- 本轮正式文档增量涉及 **18 份 Markdown**，按采集前快照 `/tmp/schema_role_trial_before_mfYTjS/` 核对，未把更早的未提交改动算入。本轮保持字段结构与既定解释；P1 的最小执行单元登记是已授权的流程修订。
- 仍有 **2 项固定规则选择**，直接来自 M02-04、M06-01、M08-03 三个原始编号，尚未实施。另有任务材料、数值路线、来源保证及调查事项未完成，它们不是新增字段的审批清单。
- `medium/` 与 `inputs/` 共 **20 份原始文件**经采集时 SHA256 核对未变。没有提取阶段批准、全算子正确性验证、checker/构建/GPU 运行或源码修复。

## 合并组

| 组 | 主题与处理边界 |
| --- | --- |
| G01 | 普通设计的硬件/DSL 资料：可继续独立推演，资源需求不等于容量、驻留或原语保证。 |
| G02 | workspace 与调用资源：需求公式已有，容量下界、允许分配及计时条件不能补造。 |
| G03 | workload 与成本选择：自选焦点可推进，未给权重不能变成性能阈值。 |
| G04 | 输入默认域与数值安全：有限域误读已解；QSA dot 和 online U 的溢出、其他误差证明分别保留。 |
| G05 | 数学接口、递推状态、工作产出与存储：已有分层足够，连接定义并收紧新导航的措辞。 |
| G06 | 动作粒度、成员、完成与等待：沿用现有事件语义；纠正把同一 CTA 内 warp 当 worker 成员的读法。 |
| G07 | 变长工作与排序：用真实 item 轴、唯一坐标映射及既有 strategy 表达；不扩 partition 域。 |
| G08 | 配置、局部差额与整体候选：改善判据和例子，逐案保留数值、归属、协议联动及未完成核实。 |
| G09 | 执行词汇与设备规模来源：最小 execution 登记已修；动态 nSM 来源仍见字段议题 F1。 |
| G10 | 任务域与可选 ABI：按材料明确的职责裁定，缺少裁定依据的具体分支保留。 |
| G11 | 外部验收来源：已知声明完整保留，未知比较路由不借默认值补齐。 |
| G12 | 缓存、侧流与跨调用保证：调用约定不能代替源码证据；源缺陷与设计缺项按实际影响分流。 |
| G13 | 注释、族名与真实实例：用调用、模板实参和使用点核实，原稿已纠正的观察也保留。 |
| G14 | 阶段、正式记录与限域手工工作：不追加完整 P1/P3 义务，也不将片段结论升级为正式批准。 |
| G15 | 调查止点、遗漏机制与 selector：继续有线索的定向调查，不穷举，也不把尚未研究写成来源未提供。 |
| G16 | GDN 标量消费与 forward 边界：现有表达可写具体路径；固定版本舍入和允许替换的接口仍未核完。 |
| G17 | ZERO2 host 条件提交：可保存固定图事实与目录关系，单份完整 flow 的变量提交能力仍见 F2。 |

## 证据位置

规范中的本轮主改动位于 [common](../../schema_docs/0.common.md#参数变化与结构变化)、[DESIGN](../../DESIGN_SCHEMA.md#1-接收-contract建立设计起点)、[DOC3](../../schema_docs/3.design.math_reshape.md#符号属性)、[DOC4](../../schema_docs/4.design.work_schedule.md#工作分解)、[DOC5](../../schema_docs/5.design.tile_pipeline.md#describe-要说明什么)、[P1](../../schema_extract/phases/01_contract.md)、[P2](../../schema_extract/phases/02_dispatch_draft.md)、[P4](../../schema_extract/phases/04_dispatch_plan.md)、[REVIEW](../../schema_extract/REVIEW.md)。未改的 [DOC1](../../schema_docs/1.contract.algorithm_requirement.md) 默认值域、精度与验收规则也作为核实依据。

源码简称按当前未修改文件定位；行号用于查证，不作为目录差额的修改地址：

| 简称 | 来源 |
| --- | --- |
| G/S、G/U、G/B、G/F | grouped GEMM 的 [solution.py](../../kernel_wiki/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/candidate/solution.py)、[fp8gg_umma.cu](../../kernel_wiki/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/candidate/fp8gg_umma.cu)、[fp8gg_bind.cpp](../../kernel_wiki/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/candidate/fp8gg_bind.cpp)、[fp8gg.cu](../../kernel_wiki/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/candidate/fp8gg.cu)。 |
| L | MLA [candidate/solution.py](../../kernel_wiki/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/candidate/solution.py)。 |
| O/C、O/S | MoE 的 [nvfp4_moe.cu](../../kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu)、[solution.py](../../kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/solution.py)。 |
| R | GDN 固定参考 `01fff90d714abc6e84904ea9c44523ba66d34a0e:kernel_wiki/aka-four-problems-20261019/gdn_prefill/reference.py`，由 `git show` 读取；当前工作树不存在该路径，不伪造文件链接。 |

### M01：decode 设计

原始材料：[观察](medium/M01_design_decode/ISSUES.md) · [草稿](medium/M01_design_decode/DRAFT.md) · [给定 Contract](inputs/fa4_decode.md)。

| 原始 ID | 合并组 | 最终归因、实际处理与剩余边界 | 证据 |
| --- | --- | --- | --- |
| M01-01 | G01 | **输入不足，原稿已正确继续。** CTA/warp 词汇已给，64-token、8-warp、单槽可作具体初稿选择；需求账不能证明驻留、MMA 或 CuTe 支持。未补硬件事实，原推进规则足够。 | 输入 wiki 占位；DESIGN §1、DOC2 占位规则；A M01-01。 |
| M01-02 | G02 | **资源与计时前提未给。** PART 写满消费区域可解除旧清零依赖，但不能保证 `WorkspaceBytes` 足够，亦不能推出可免费额外申请。保留无 workspace 主线与 split 候选边界，未改字段或缩域。 | 草稿 split 需求 `16,512*BatchCount*q_len_per_req*Smax`；DOC4 存续与初始化、common when。 |
| M01-03 | G03 | **缺 workload 分布，非字段表达不足。** q_len=1/4 在该分解中只变工作量；长 KV 不独自决定 split 收益。模板已说明材料与格式边界，未新增 workload 字段或虚构阈值。 | 草稿工作数/归并成本；模板 §1、common 配置与工作量。 |
| M01-04 | G04、G05 | **概念已解，数值验收未完成。** 当前 m/z/u 的 retention 和 shape 不需要 page 快照轴，浮点默认有限已存在。补默认域与状态/存储导航；长递推、抵消、exp 对 tolerance 的误差仍未证明。 | DOC1 range、DOC3 符号与保留、DOC4 数据块；草稿数值自检。 |
| M01-05 | G06 | **现有 serial 顺序可用，但须有完整动作保证。** 单角色、单槽的读完再覆盖可由顺序推出，无需重复 waiting；若动作只是各 warp 局部完成，serial 本身不补 CTA 屏障。DOC5 新例与成员提示明确该前提。 | DOC5 执行模式、默认依赖；草稿页内/跨页动作序列。 |

### M02：grouped GEMM 提取

原始材料：[观察](medium/M02_extract_gemm/ISSUES.md) · [草稿](medium/M02_extract_gemm/DRAFT.md)。

| 原始 ID | 合并组 | 最终归因、实际处理与剩余边界 | 证据 |
| --- | --- | --- | --- |
| M02-01 | G10 | **可选 ABI 尚无足够裁定依据。** shape_range 可以确定所声明动态维度，不能自动决定较宽签名的 out=None/out_dtype 分支。reference 直接读 out.shape、生成器总传 out，不足以补全上游 API。P1 改为按具体对象对齐材料；nullable/default 现有表达足够，未替任务作决定。 | GEMM problem/metadata.json、reference.py:9/26；B“任务与验收材料”。 |
| M02-02 | G08 | **现规则可表达，例子已补。** 固定 abox-on、固定槽容量内按有效行取 16/32/64/128 box 是本次尾块搬运，实际字节和完成计数须随之说明；host 的 abox 开关和失败回退另行记录。未将每个 box 建成配置节点。 | C 复核 G/U:377–396；common 新尾块例；G/U:1682–1686 的配置分支。 |
| M02-03 | G08 | **有据的局部候选，完整协议未验收。** 两个 issuer 处理原 pair 的两块，分别 commit，odd-K 无第二 payload 也通知；producer、pair 序列、TMEM 与 promotion 接口可沿用。可围绕原接口记录局部协作，不能简化成 role 数量参数。common 强化闭合判据，未要求先写两套完整 YAML。 | C 复核 G/U:336–339、428–511、838–855；A/B 均保留完整变体同步未核完。 |
| M02-04 | G09 | **两部分不能合并结案。** 已授权的 P1 最小 execution 登记解决 CTA/warp 名称来源，DOC2/template/EXTRACT 已同步，旧 Runner 仍不接受。nSM 在 G/S 初始化时查询设备 0 并缓存，设备规模的绑定资料仍缺；F1 保留来源规则选择，未硬编码、改成算法输入或塞入 execution。 | C 复核 G/S:126–129、G/F:340–344；DOC2 具名映射/wiki.hardware、DOC4 核心思想；FIELD_ISSUES F1。 |
| M02-05 | G12 | **调用前提与异常语义来源不足。** 六个 weakref 命中不能证明 storage/metadata 不变；graph/direct 报错后回退也不能证明前缀零提交。REVIEW 明确调用约定不是源码保证。正常已确认路径可继续，其余具体生命周期与错误范围未批准；未修源码或制造隔离条件。 | C 复核 G/B:69–83、G/U:1542–1547、1723–1759；REVIEW 源实现缺陷与设计分流。 |
| M02-06 | G06 | **原稿已自解，保留正面使用证据。** store.done 与源槽 release.done 已有不同含义，read-wait 可保护槽复用，最终 full-wait 才排空目标写回。不新增事件或完整节点；full/tail 条件组合仍未写完。 | DOC5 OSlot 例；G/U 的 bulk_wait_group_read0/bulk_wait_group0；原观察与 A/B。 |

### M03：QSA 设计

原始材料：[观察](medium/M03_design_qsa/ISSUES.md) · [草稿](medium/M03_design_qsa/DRAFT.md) · [给定 Contract](inputs/qsa_prefill.md)。

| 原始 ID | 合并组 | 最终归因、实际处理与剩余边界 | 证据 |
| --- | --- | --- | --- |
| M03-01 | G04 | **有限域误读已解，参考/验收的极值问题真实。** 有限 BF16 q、K 可取 `2^64`，单个乘积 `2^128` 已越过 FP32 有限上界；max-shift 不能修复之前的 Inf。默认 tolerance 没有双方 NaN 自动通过的许可。只补默认域导航，未缩域、改参考或把宽 dot 宣布为修复。 | C 对输入与 DOC1 手推；A/B 对相同反例的独立判断；DESIGN 数值自检。 |
| M03-02 | G04 | **新算法的确定溢出反例，仍未修复。** q=K=0、2048 个 V 为 BF16 最大有限值时，归一化权重可精确为 1/2048、加权和保持有限；未归一化 FP32 U 累加两份 V 就会溢出。归一化均值或宽 U 仍需选定并分析，不能登记成“只待跑误差”。 | C 手推 BF16 最大值及 FP32 范围；草稿 §3 已撤回原论据；DOC3 精度规则足够表达后续方案。 |
| M03-03 | G01 | **资料不足，非新增 realization 字段需求。** 32 KiB K/V 槽和 12 KiB U 是需求；MMA 形状、容量、完成原语与 CuTe 支持未绑定。同步 matmul 文案只改善表达，未补本题能力或性能依据。 | 输入 wiki 占位；DOC2/6；草稿资源预算。 |
| M03-04 | G08 | **当前差额可作局部，不能只数 launch。** PackCache 改既有 Ksel/Vsel 来源、物化和 handoff，MAIN 归属、递推、gate 保持。common 增加输入重排例；最低 512 MiB 准备流量仍纳入比较。后续若改变主体工作组织须重新判断。 | 草稿候选与 handoff；C 对照 common 局部条件；A/B 的对象边界判断。 |
| M03-05 | G06 | **纠正原稿残留误读，尚非已修好 pipe。** worker 只有一个 CTA，MAIN[*] 选一个成员上的 role 实例，不枚举四个 warp。原稿两条边不能替“this warp”动作补全 CTA 协作。DOC5 明示 worker 成员范围及 serial 前提；后续须明确整个 role 的完成保证或按实际独立职责分 role，未改原稿。 | C 复读草稿 §5、DOC5 事件引用与执行模式；采集前规则已有该成员含义；采纳 A 的纠正。 |

### M04：MLA 提取

原始材料：[观察](medium/M04_extract_mla/ISSUES.md) · [草稿](medium/M04_extract_mla/DRAFT.md)。

| 原始 ID | 合并组 | 最终归因、实际处理与剩余边界 | 证据 |
| --- | --- | --- | --- |
| M04-01 | G13 | **陈旧注释造成误图，原稿已纠正。** host ItemGeo、动态 grid、pop server 与当前数值链不能按文件头概览确定。P2 直接提示从实际调用、模板实参与函数体建机制图；不改源注释，也不把历史模式另算候选。 | L:1646–1681、1450 附近、2008；原观察；A/B 来源核实。 |
| M04-02 | G10 | **材料表述不一，但这部分已有明确裁定依据。** metadata 指定 shape_range 的职责；其中头数 tuple、causal 两值及 query_lengths==kv_lengths 可据以使用。较宽生成器/候选注释不扩域。P1 改材料对齐提示，不要求用户再批准同一已给事实。 | C 核 metadata.description、shape_range:137–139；与 M02-01 的未定可选 ABI 区分。 |
| M04-03 | G11 | **比较器材料缺项，现行来源记录足够。** 已知阈值和 mutated_inputs.out override 保留；candidate 公式不能替代未给 evaluator，返回/LSE/原地值路由也不能猜。未改 tolerance 默认或 gap 职责；不追加追索外部 runner 的硬前置。 | MLA problem/metadata.json/reference.py；DOC1 固定实现验收记录；B 的逐输出核实。 |
| M04-04 | G08 | **局部降级尚不充分，保留整体候选判断。** tiny 的实际 PV/分母改由 FP32 路径与 SMEM 结果供 epilogue；普通 P/PV 虽继续执行，已不提供该实际输出。A/B 倾向独立候选，C 认可其理由比“原计算仍在”充分；最终完整保留仍须对照基线与范围，不因更换输出来源机械增文件。 | C 核 L:698–735、1078–1087、1182–1216 与普通 PV；common 实际输出来源提示；保留 M04 原局部倾向。 |
| M04-05 | G08 | **局部协议变化可表达，不能只登记 depth。** 3 槽直接等 pv_done，4 槽另有匹配周期的 kv_free；容量、索引、发布/等待须随差额保留。QK/PV/领取主体保持支持局部候选，完整跨工作相位证明仍未完成。 | C 核 L:1329–1330、1361–1362、1392–1412；DOC5 storage/index/waiting；A/B。 |
| M04-06 | G07 | **已有表达足够，新例已明确坐标身份。** item 是排序后实际表项的位置，每项唯一恢复原工作坐标；by:item 升序并不等于按原 batch 排序，kv_tiles 只作排列键。DOC4 增补映射、稳定次序及准备/上传/复用落点，未增 sort_key。 | C 核 L:1782–1820、1888–1898；DOC4 工作排列；C 补“排序后的位置”以消除歧义。 |
| M04-07 | G12 | **必要调用保证仍缺来源。** upload→kernel 边有据，输入生产者→nonblocking side-stream 读取、旧 table 消费→下一次覆盖尚缺证据。REVIEW 澄清不能用 Geo ready 或调用约定补等待。影响使用该共享准备的完整代表，但不阻止记录已确认内部事实；未称实测竞态。 | C 核 L:1646–1681、1895–1898、2010–2011；原稿与 A/B；REVIEW。 |
| M04-08 | G14 | **原阶段职责可解，导航已直接连接。** P2 保存线索和暂拟对象，P4 基于实际完整基线落实正式定位；未来文件/对象仍仅计划。P2/P4 已修文案，未增加 Design 写权限、空 provider 或提前完整 YAML 要求。 | P2 第6项、P4 第4–5项；原稿回读后的正确处理。 |

### M05：GDN 设计

原始材料：[观察](medium/M05_design_gdn/ISSUES.md) · [草稿](medium/M05_design_gdn/DRAFT.md) · [给定 Contract](inputs/gdn_prefill.md)。

| 原始 ID | 合并组 | 最终归因、实际处理与剩余边界 | 证据 |
| --- | --- | --- | --- |
| M05-01 | G01、G03 | **workload 与硬件相对代价未给。** B3 的 S_entry 上界约 1.03 GiB 是实质成本，不能以主链缩短直接判快。原规则已要求整体成本，未增字段、假定低精度或硬件吞吐；R/B 性能取舍仍待材料。 | 草稿 §2 快照/物化账；DESIGN §1–3、模板资料入口。 |
| M05-02 | G16 | **表达能力已有，固定版本消费事实未核完。** binary64 scale 的输入类型不自动决定乘法精度；R:249–251 的 Python scale_factor 乘 FP32 张量可定位，但结果 dtype 不能证明所有舍入点。现有显式 cast/arithmetic 可写具体候选路径，未宣布默认先转 FP32 或默认 FP64，也未把设计称为数值收束。 | C 核 R:249–251、274–283，输入 scale；DOC1/3 精度；A 明示 PyTorch 2.9.0 消费语义未独立核完。 |
| M05-03 | G16 | **forward 边界仍需任务事实，不能由 oracle 代裁。** 给定 Contract 把 int64→int32 计入 timing，R:275 却直接求差并 tolist。未据此删除固定 wrapper，也未添加无依据转换 launch。既有 step/动作/调用边界能表达确认后的处理和成本，无字段变更。 | C 核输入计时句与 R:274–283；草稿 §4 明示未补 boundary/scale。 |
| M05-04 | G05 | **原误读已解，跨区导航补齐。** S0 可是 invariant；随 token 变化的 S 为 carry_forward，实际容量记 storage；覆盖整段的 O target 不要求片上保留整段。DOC3–5 连接这些位置，不新增内部 state 栏。 | 草稿 §3–5；DOC3 保留、DOC4 数据块、DOC5 storage。 |
| M05-05 | G07 | **未选具体 ragged 组织，现有表达有合法路径。** 按前缀把真实 (b,c) 编成 item，工作数为 Items*H，恢复坐标不再重复相乘；P 是数学映射，不自动有数组。DOC4 新例交代准备归属，未允许 size 捕获另一 partition 坐标或新增 lookup。 | C 核新例与原乘积计数规则；草稿 B3 的 PREP/STATE/READOUT 差别。 |
| M05-06 | G06、G01 | **动作粒度大体解开，实现保证未就绪。** Advance 可引用上游计算，不必逐归约项拆动作；真实发布与复用边界仍写明。同步 matmul 例示设计层完成保证，C 压掉重复数学、补具体槽绑定；资料缺失时不捏造 CuTe API，草稿不等于完整可实现交付。 | DOC5 describe/执行模式、DOC6 删除测试；草稿双 role 与未绑定保证。 |
| M05-07 | G08 | **整体候选理由充分，未强制交付两份。** READOUT 移入 STATE 后独立 chunk 输出轴、生产者、主链和 S_entry/U 的全局存活均改变。common 用此例说明重新贯通的必要，不能按少一个 launch 或共享公式多就降级。 | C 复读草稿 B3/B2；common 新例；A 的对象/寿命比较。 |
| M05-08 | G14 | **已知边界，原稿已正确处理。** 手工设计没有旧工具硬前置；检查未完成不等于数学停止，也不能为旧 checker 填旧字段。EXECUTOR_ALIGNMENT 集中说明结论范围，未迁移工具或重报一个实际运行失败。 | 原 DESIGN 中间稿规则；执行器对齐“手工推演与阶段结论”。 |

### M06：MoE worker

原始材料：[观察](medium/M06_extract_moe/ISSUES.md) · [草稿](medium/M06_extract_moe/DRAFT.md)。

| 原始 ID | 合并组 | 最终归因、实际处理与剩余边界 | 证据 |
| --- | --- | --- | --- |
| M06-01 | G17、G12 | **真实条件提交能力问题；有价值的事实仍可记录。** clean 已提交，host 观察非1才提交 full，固定 launch/等待/空工作不能忠实表示整个算法。现有目录可保留固定图、相容关系与未展开独立候选，不等于完整 ZERO2 YAML。F2 留决策；mirror 版本/跨调用竞态另属源码保证，新增条件字段不能修复。 | C 核 O/C:1707–1793、9157–9159、9471–9491；DOC4 提交存在性、DOC5 if、common 目录分工；FIELD_ISSUES F2。 |
| M06-02 | G08、G15 | **研究未完成与分类裁量并存。** 原稿避免 cfg 穷举正确；REDF 当时未读的数值/输出关系已由 M08、B 及 C 定点补核，但 mono/BF16 的整链去处、cfg88 等未完。P2 区分实际混合、未研究、已研究取舍与未来写作；未宣称全范围 P2 已完成。 | 原稿停止点；O/C REDF/实例/输出代码；M08-02/04/06 各自保留。 |
| M06-03 | G11 | **已知验收声明比简单 atol/rtol 更具体。** 原 allclose 实参顺序、default 与 mutated_inputs.output 的 mismatch/rel-L2 差异均须保留；返回列表与原位对象的完整检查路由未给。现有 tolerance 来源记录足够，不压成 reference-relative 默认，不挪入 gap。 | C 核 MoE metadata.description/benchmark_contract；DOC1 验收来源记录；B 逐输出核实。 |
| M06-04 | G13 | **旧注释误导，原稿已局部纠正。** 实际 cfg99/81 可达、MEGA 用 I/32；不按 override-only/I64 概览分类，也不继承性能或 bit-exact 注释为本次证据。P2 与 M04-01 共用实际实例提示，未宣称所有注释已审完。 | O/S selector、O/C MEGA 主体；原观察；B 的定点行号。 |
| M06-05 | G08、G06 | **两部分均有现成表达。** CPHOIST 决定搬运/退役策略，属配置与局部差额；本次 Clean 是执行状态，但仍受 G17 的提交语法限制。跨 item 不重置的环可用 flatten 的 g+3/g+2，未新增事件或 while。 | common 决定对象；DOC5 flatten；O/C:4855–4915、4628–4660、4950–4968；M08 的独立片段核对。 |

### M07：prefill 设计

原始材料：[观察](medium/M07_design_prefill/ISSUES.md) · [草稿](medium/M07_design_prefill/DRAFT.md) · [给定 Contract](inputs/fa4_prefill.md)。

| 原始 ID | 合并组 | 最终归因、实际处理与剩余边界 | 证据 |
| --- | --- | --- | --- |
| M07-01 | G01 | **能力/容量/DSL 资料未给。** 72 KiB SMEM、16 KiB 累加器的需求账不能证明驻留、异步重叠或低精度矩阵路径；保守数学选择可继续。未因新同步例而声称本题已有硬件依据。 | 输入执行词汇与 wiki 占位；DESIGN §1、DOC2/6。 |
| M07-02 | G04 | **纠正“输入缺项”的原归因。** DOC1 原已规定浮点默认有限，query/kv_cache 未写 range 因而不含 NaN；允许任意旧 bits 且不读取的 out 不改变输入算术域。DESIGN 新增直达链接，无需任务方重复批准 finite。有限域分析仍不等于 tolerance 证明。 | 采集前 DOC1 字段表/range 两处；输入 out 含义；A/B 纠正，C 复核。 |
| M07-03 | G03 | **优先 workload 未给，已有推进路径。** 明确自选多 query 焦点后可比较候选，不能转成用户权重或精确 dispatch 阈值；不为四类情景机械造四份 YAML，也不增 workload 字段。 | 原稿焦点与候选成本；模板 §1、DESIGN §1。 |
| M07-04 | G07、G08 | **Bq 名称不决定差异等级。** 全局 token tile=2 可跨请求，必须重新确定请求内工作集合与映射；紧凑 item、上界空工作和新任务表各有不同联动。DOC4 例子提供可表达路径，common 判据保留；实际方案未选时不预判文件数量。 | 原稿 Bq=2 候选；DOC4 item/工作数；common 工作归属与局部闭合。 |
| M07-05 | G06 | **matmul 不是单条硬件指令，完成保证仍须落实。** DOC5 允许同步协作实现数学乘积，内部线程同步不必机械拆边。新例保留 QSlot/KSlot/ScoreSlot 及全参与 warp 的 done 范围；未证明七 warp 或异步 load 已可实现。 | DOC5 既有基础动词与合理粒度、新同步例；原稿的资料限制。 |
| M07-06 | G05 | **名字职责与草稿缺项并存。** 数学接口按原 DOC3 判据登记，局部地址/判定/packing 按动作定义，存储按真实容量/寿命列账；C 改导航，避免凡跨动作对象都新增 symbol。offset 是相对工作区域的偏移列表，原稿删除源切片写法正确；m/l 与寄存器账仍未补全。 | DOC3 原判据、DOC4 offset、DOC5 本地对象；原稿自纠与 A 导航、C 收束。 |
| M07-07 | G02 | **scratch 及调用资源输入不足。** `4*S*Hq*(D+2)` 是 split 需求，不是 WorkspaceBytes 下界；首次清零不保证每次重入，内部申请也不能假定免费。未缩域或补默认，主方案可继续。 | 输入 workspace、原稿 split 候选；与 M01-02 合并但保留独立观察。 |
| M07-08 | G06 | **原稿已自解，片段证据有效。** load(i+2) 等 PV(i)、QK(i) 等 load(i) 分别保护覆盖与就绪；声明窗口处理首尾超域。N=1/2/3 手推只支持该片段，不是全 pipe 验收；未增 allowed 或重复等待。 | DOC5 默认依赖/索引窗口；原稿局部序列。 |

### M08：MoE 独立 reviewer

原始材料：[观察](medium/M08_review_moe/ISSUES.md) · [草稿](medium/M08_review_moe/DRAFT.md)。M08 当时未读 M06 的 ISSUES，以下独立发现不追写为 worker 当时已知。

| 原始 ID | 合并组 | 最终归因、实际处理与剩余边界 | 证据 |
| --- | --- | --- | --- |
| M08-01 | G14 | **限域任务与正式准入不同，已按授权处理。** 未完成 P1 的 M06 草稿可接受本次候选评议，但不形成正式 P2 approve。REVIEW/执行器说明明确结论范围，未要求补整个 P1/P3，也未放宽正式 W/R 与前置批准。 | 本次委派范围；M06/M08 开头；REVIEW 阶段覆盖与限域段。 |
| M08-02 | G08 | **独立整体候选判断有据，保留其增量。** 相对 cfg99/44，cfg99/46 保持 mono G2，却把 route 独占 rout→finalizer 改为 token 共享 acc32 原子归并→convert；BF16 边界、初值、冲突、完成与求和次序均相关。C 认可完整候选理由，未批准最终数值或唯一选集。 | C 核 O/C:3948–3987、8233–8261、9164–9192、9330–9338；M08 的最近基线判断，不混入 WS→mono 差别。 |
| M08-03 | G17 | **独立确认同一条件存在性问题。** start_after 只给启动时点，role if 不能取消 host 提交；不同 when 也不执行本次轮询选择。与 M06-01 合并 F2，保留独立来源。目录可保存事实与组合说明，但不能补成一个缺失控制流的完整 YAML。 | C 核实际 graph_replay 与 DOC4/5；FIELD_ISSUES F2。 |
| M08-04 | G13、G15 | **worker 族名概览有缺项。** cfg99/44 已含 mono G2，不等于整条 mono/BF16 都已研究。P2 增加逐计算段说明已选机制的提示，避免重复计数和漏掉主体差别；不要求 WS×mono 全配对或新字段。 | C 核 O/C:8233–8261；M08 定向追 mono G2 的独立发现。 |
| M08-05 | G06、G14 | **局部正确性与完整性分别处理。** g+3/g+2 只说明 operand 环复用，未证明 prepared、accumulator/epilogue、CPHOIST、TSTORE 或跨 step 全部关系。REVIEW 明确按限域对象核对；不将缺全 action 表当成必须补 P3，也不批准完整 pipe。 | DOC5 flatten、动作完成与窗口；M08 只核 O/C:4628–4660、4950–4968 的实际范围。 |
| M08-06 | G15 | **新增真实调查线索，现规则已要求处理。** cfg88 有入口、跳过独立 act，并由 last-arriver 归并/QDQ；不能被“已有 split-K”或另一处 ticket 自动覆盖。P2 强化 reviewer 独立止点；初始化、全部 partial 可见性与后继接口未完整核实，保留具体研究，不先增文件。 | C 核 O/C:9299–9300、5270–5313；M08/B 核实例 8031–8060；不继承源码 bit-exact 注释。 |
| M08-07 | G15 | **selector 示例不构成全域工作量保证。** `ceil(R/bm_g)*per_unit` 用于决定是否 readback，masked/empty 当前路由仍可能更少。原 common 已区分 selector、when 和样本；不在本次限域评议追加完整 P3，也不批准草稿算式为所有输入下界。 | O/S:1260–1273；M08 对草稿范围的独立纠正；common 选择条件。 |

## 尚未闭合事项的去处

F1 影响依赖设备规模查询的完整 mapping；除 M02-04，C 在 L:2001–2008 看到同类查询及源码 fallback=148，作为同组佐证，不新增 medium 编号，也不将其变成规范默认。F2 影响要求以单份完整 YAML 表达 ZERO2 数据相关 host 提交算法的任务；固定图、目录关系和已研究未展开候选仍有现行记录位置。两项选择及推荐均在 [FIELD_ISSUES](FIELD_ISSUES.md)。

其他剩余事项按性质继续：QSA 数值路线与参考极值处理，decode/prefill 的误差保证，GDN 标量舍入与 forward 边界，workspace/workload/ABI 材料，缓存/侧流/mirror 的来源保证，以及 MoE 尚未完成的机制调查。它们不因文案修订自动通过，也不一概归为字段缺陷。M04 的已指定动态域、M07 的默认有限域和原稿已解的事件用法无须重复用户批准。

正式 P1–P6 仍保留当前有效全文、自查与独立 reviewer、独特价值与必要降级、按真实影响返回和 P6 新终审；本轮没有运行这些流程。现有工具差异集中在 [EXECUTOR_ALIGNMENT](../../schema_extract/EXECUTOR_ALIGNMENT.md)，文字正确性与执行能力分别报告。
