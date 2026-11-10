# 动作语义的结构缺口复核

本复核以根 README 的 Pipeline IR 目标为筛选条件：删掉一条事实后，是否会容许不同的主要计算分配、资源寿命、并行重叠或一阶工作量；然后检查现有结构及已定义语义能否保留该事实。**动作范围内没有确认必须新增固定字段或事件的缺口。最有证据、最值得优先优化 schema_docs 的内容，是少量共享动作语义及其绑定规则；其次是角色内分工的紧凑表示成本。** 这不表示当前结构已经独立表达全部动作事实，也不表示现有长文本不能删减。

本次筛过原目录全部 182 条 finding，其中 106 条纳入动作复核、76 条只作范围外筛查；动作复核覆盖十算子及其相关数值、实现绑定、来源边界。逐项记录见 [action_findings.json](action_findings.json)。范围外记录仅为 `screened_out_of_scope`，没有据此判定全局无缺口。106 条主判分别为 43 条 existing_structure、28 条 shared_semantic_definition、12 条 representation_cost、15 条 source_or_uncertain、6 条 no_gap、2 条 scope_decision；这是 finding 子论断的分流，不是独立缺口数，也不把次级判定相加。各 finding 的原始 unit 集合、全部 dispatch/节点覆盖与来源锚点沿用原审计；本次不重新运行全量审计，不声称重新逐字审阅全部原始文本组，也不把抽取状态当作来源正确性证明。

## 判定尺度与现有能力

读过现行 README、DOC3/4/5/6，并按原 finding 追到冻结样本、算子报告及必要源码。规则中已有的能力必须先扣除：

| 已有能力 | 规则锚点 | 不可据此推导的东西 |
| --- | --- | --- |
| 上游计算唯一归属，数学视图、显式舍入、归约与近似例外 | [DOC3:81](../../../../schema_docs/3.design.math_reshape.md#符号属性)、[DOC3:225](../../../../schema_docs/3.design.math_reshape.md#精度与辅助运算) | 数学等值不自动说明这次物理读取来自何处、只读一次、或由哪个 warp 执行 |
| `compute / produce / load / publish / release / wait` | [冻结 DOC5:80](../rules/schema_docs/5.design.tile_pipeline.md#动作与契约) | 自造对象名不自动定义 bundle；`release` 不代表目的写入完成 |
| `issue / done`、异步及累积完成、局部同步例外 | [冻结 DOC5:100](../rules/schema_docs/5.design.tile_pipeline.md#role-内的顺序) | done 有序不等于先读旧操作数再覆盖；serial role 不等于线程在每个语句锁步 |
| `role.member / repeat / waits_for / member事件引用` | [冻结 DOC5:139](../rules/schema_docs/5.design.tile_pipeline.md#pipe-与局部重复)、[188](../rules/schema_docs/5.design.tile_pipeline.md#成员与名称) | 成员参数不能随 pipe 变量改变；CTA 成员索引不是 warp 索引；不能将未触发 warp 的等待加给整组 |
| `resource.view / part / slot_index` 与全读者退役 | [冻结 DOC5:273](../rules/schema_docs/5.design.tile_pipeline.md#视图) | view 只给存储对应，不产生读取、发布、同步或输出组成 |
| `kernel_done / launch_ready / handoff / milestone` | [冻结 DOC4:305](../rules/schema_docs/4.design.work_schedule.md#事件与启动) | 启动许可不是值可用，kernel_done 不是 host 全设备同步 |
| 原语自身协作、读写顺序保证留在短契约，具体提供者由 realization 绑定 | [冻结 DOC5:96](../rules/schema_docs/5.design.tile_pipeline.md#动作与契约)、[DOC5:137](../rules/schema_docs/5.design.tile_pipeline.md#role-内的顺序)、[DOC6:11](../rules/schema_docs/6.realization.md#核心思想) | “允许写文字”不等于已有结构能解析语义；换一种等价 API 不自动构成不同核心设计 |

`confirmed_structural_gap` 的门槛是：唯一核心事实可证明；现有字段、合法表达式和已定义通用语义不能保留；不能仅因需要展开较多 role、需要定义一个算法、没有专用字段或图工具不解析文字而升级。本轮没有动作 finding 同时通过这些条件。

## 最小机制及逐项反证

### 1. 动作接口：读写集合、合成结果与捕获副本

这是动作部分最强的优化候选，判为 **shared_semantic_definition，P1**。代表项为模板 GDN F09/F10、FA4 prefill F14/F15/F17、decode F03、MLA F09/F12、FMHA F12、MoE F08。事实不是“有一个动作”，而是该动作实际读写的对象、共同发布的结果、捕获到何处、捕获以后还要保留多久。

模板 GDN 的 `StateOperand` 只在 [root.yaml:269](../inputs/template_gdn_prefill/root.yaml:269) 明确从 incoming S 每 subtile 读一次，同时产生 Sbar 和原地 Sdec；[303](../inputs/template_gdn_prefill/root.yaml:303) 定义 Gate 同时发布 G/Gamma。若将所有承载该事实的位置一起删除，相同动作名和等待边可以允许仅发布 Sbar、漏掉 Sdec，或者为两个结果重复读取 S。原 `Arena.view` 已表达同址，但不表达这些动作效果。不能只把名字改成 `produce:StateOperand` 就称已结构化。

捕获反例更直接：[GDN root:240–243](../inputs/template_gdn_prefill/root.yaml:240) 的 row Beta 在寄存器中继续使用，column Beta 仍要从 SMEM 读到 Tinv 发布；[274](../inputs/template_gdn_prefill/root.yaml:274) 的 U 捕获先释放 KS/U TMEM，再形成两个窄化版本。若捕获只是晚读 view，原有 reuse 边会容许覆盖仍需读取的值。FA4 prefill 的 Stats/ell 捕获和 MLA 的末 32 列输出捕获具有相同寿命机制。

现有结构尝试的结果：

- `view` 能给源/目标存储，不能给本动作的读写集合；`milestone` 只能给已有事件组命名，不能定义一次动作有哪些结果。
- 数学定义可将 FinalSum 这类单纯别名定义完整，然后直接 `publish:FinalSum`，见 FA4 prefill F12。不能因此断言所有混合类型、不同驻留的 bundle 都已有相同解法。
- 对单值捕获，可定义清楚的源值/快照关系，登记 register 资源，使用已有 `load` 与 `release`，并让后续计算引用快照；这是一条现有结构的展开方向。不得仅造 Captured 名字，不交代其来源、驻留和消费者。
- 多结果融合可展开为多个具名计算/发布，但必须检查是否新增了中间可见性或发射顺序、是否丢掉单次读取复用。完整等价展开尚未审定，所以不能把展开方向写成已完成的删改。

最小不丢事实的**概念表示**是：`StateOperand：读 S_old 一次 → {Sbar, Sdec 原位覆盖 S}；done 保证两者可用`；`捕获：指定源片段 → role 的寄存器副本；done 结束本 role 对源的读取；副本保留到指定消费者`。这不是现行新增 YAML 字段的示例，而是共享规范必须解释、实例只需绑定的参数。参数至少包括输入片段、输出集合及目的位置、读次数/复用要求、完成边界；没有这些参数的“capture profile”仍是文本垃圾桶。

### 2. 完成保证、组窗口与全读者退役

判为 **shared_semantic_definition，P1**；这与机制 1 可以由同一套动作契约规范覆盖，不建议再造 read_done、retired、ready-only 等通用事件。

FA4 prefill F13、QSA F06、MLA F11 具有同一个强反例：PV 读取旧 P 以后，后发 QK 才可覆盖别名 S/P。仅保持 `async_ordered_completion`，仍可能出现后发 QK 先写、最后才 done，从而满足完成顺序却损坏旧 P。直接加 `PV.done → QK.issue` 又会取消原语允许的在途重叠。应保留一次定义的“前一乘积的操作数读取先于后一写入”保证及提供该保证的 primitive 绑定；无需为其假造可观察读事件。

GEMM F09/F10、GDN F08、decode F03/F04、FMHA F12/F13、MoE F08 则涉及完成所覆盖的事务组与读者范围：尾部空 commit 仍计入 group 窗口；PRE0 的同一 transaction 分两时段填充但不能再 arrival；Staging 归还要覆盖所有此前 bulk store 的源读取，跨 tail 不能丢失历史。`repeat` 与连续索引能表达对应动作，不能自行定义这些原语窗口。

现有 `release` 已完整容纳“源已读完、目的写仍在途”。[GEMM root 的输出段](../inputs/extract_gemm/root.yaml) 和 [MLA root 的 OutputState](../inputs/extract_mla/root.yaml) 都已有 release 与后续尾写；为了少写一句而改等 store.done 会加强等待。`kernel_done` 已提供本 kernel 的提交完成和必要可见性，不能把它重复记成动作缺事件。

最小概念表示：`done覆盖哪些已发射事务/哪些成员`、`空payload是否仍提交一组`、`源读完与目的完成分别由哪个已有动作保证`、`原语提供的读先于写规则及适用范围`。实例中的具体组深度、域、槽数、前驱已经有结构时不再重述。原语无法提供或源码未建立的保证必须留为来源限制，不可补成通用默认。

### 3. role 内部分工、子集协作与独立推进

主要判为 **representation_cost，P2**。模板 GDN F12、decode F04/F07/F08、QSA F09、提取 GDN F10/F12、GEMM F08、Sparse F09/F10、FA4 prefill F10/F18、MLA F13 都落在这一族。

删除 GDN inverse 的逐级 warp 分工，会允许四个 warp 串行做两个 chunk，也会允许每两个 warp 并行做一个 chunk；删除 decode global/compact 的分配句，在相同 `member: warp[0:4]` 和 pipe 下，可能变成每 warp 合作求权重，也可能变成 128 线程各自重算 exp2。后者在 [q1_global:363–367](../inputs/template_fa4_decode/q1_global.yaml:363) 与 [q4_compact:353–356](../inputs/template_fa4_decode/q4_compact.yaml:353) 有明确对照，确属主要计算分配与访问量，不只是线程 layout 拼写。

已有 `role.member`、数学切片、`repeat`、`waits_for` 能把静态 warp 子集展开，按层列出真实局部动作及 rendezvous；固定 cluster S 可以列全 `mapping.member`，XOR 集体可以按轮展开。不能用随 repeat 变化的 member 或未定义的 `Role[r xor …]` 记法冒充现行语法。可表达方向不等于完整展开已经简洁、等价或值得采用。

FA4 prefill F18 还有更细的边界：[splitkv_dual_load:285–294](../inputs/template_fa4_prefill/splitkv_dual_load.yaml:285) 按线程拥有的 row group 裁剪晚期 copy，读取为线程私有，所以无需逐 stage CTA barrier。拆成八个 warp 并不能自动保留每线程独立裁剪/就绪。DOC5 有意把线程协作放在原语内部；在当前粒度下，一句“私有分区、谁读谁等”是合理语义定义，尚不足以要求全新的线程事件系统。

最小概念表示是“成员子集 → 已定义计算/值区域；哪些完成需要子集汇合；哪些子集可独立继续；主要重复计算次数”。优先在真实案例上比较短契约和静态展开的总成本，再决定是否需要一个通用分工接口。不要先加入 per-thread 字段。

### 4. 原子协议与无数学值的预取

这两类都需要精确语义，但不能因难以逐冲突展开就一律升级固定结构。

**原子协议：shared_semantic_definition，P2。** QSA F05 的 one-CAS/group、每个插入者 OR、一次 reservation/warp 是计算与竞争量；Sparse F11 的 key 创建和 membership 首次置位可以来自不同 occurrence，决定 ProbeCount。注意 [probe_hidden_mma:108–112](../inputs/extract_sparse/probe_hidden_mma.yaml:108) 已有 Creator/First 公式；它不是缺“计数公式”。仍需保留 attention 插入者先发布 canonical position，其他同 key 参与者等待 position 再用的并发协议。源码见 [dsv4_kernel.cu:2346](../work/extract_sparse/dsv4_kernel.cu:2346) 与 [3972](../work/extract_sparse/dsv4_kernel.cu:3972)。删协议会容许读取未发布 position；把 probe 简化成 distinct count 会改变结果。

数学公式给输出与并发顺序参数，role/资源/边给外层协作，短算法定义给原子返回值、线性化与发布保证；这是一条闭合表达路线。原子冲突次数依赖运行时状态，逐次写事件图没有已证必要性。共享语义应只定义 CAS/OR/reservation 的必要返回及可见性保证，实例仍定义算法，不把一个未定义的 `atomic_build` 当答案。

**预取：shared_semantic_definition，P1 的接口特例。** GDN F13 的 [warm_metadata:479–488](../inputs/extract_gdn/cached_ordered.yaml:479) 在 PDL wait 前合法读地址、丢弃读值，wait 后重读 Perm 并禁止编译器复用旧值。若改写成普通 `load:Perm`，就会把“地址合法的无值预取”误作“当前数学 Perm 已可用”，要求额外初始化/等待，或错误复用旧值。DOC4:368 已明确区分分配字节与交接数学值，因此这是已有规范需要收束的访问语义，不是发现了一个全新的事件类别。

最小概念表示：`合法地址域；读值丢弃且不进入结果/控制/后续地址；有用读取重新发生在既有ready之后；是否产生实际访存/仅hint`。FA4 prefill F18、decode global/compact、Sparse F17 的空 split 访问必须分别保留“读但丢弃”“跳过 payload 读取”“读取有限旧值参加零权算术”，不能统一改成 empty no-op。全局 allocation 身份/跨调用初值是否要结构化由另一范围复核，不在此处凭空造 owner。

### 5. native 近似与实现条件

FMHA F06、MoE F06、FA4 prefill F07、decode F09、MLA F06 的主判定是 **existing_structure**：显式 `cast / fma32 / mm32 / mma32`、`arithmetic.approx / accumulate` 与数学分情形已有位置。native block-scale 输入解释、FTZ、累加分组和实际舍入不能从 fp32 dtype 推导，但需要精确定义数值运算，本来不等于需要新的动作字段。

未知 native 内部行为、误差界或比较器定义应判 `source_or_uncertain`。不允许用新函数名遮住未知数值语义。近似原语的具体实现提供者由 realization 绑定，不能以 API 名称隐式允许其它 FTZ 或 contraction。

实现绑定里只纳入真正提供上述保证的部分。alignment/swizzle 兼容条件可能是必要交接；barrier 编号、wrapper 或等价 API 替换没有独立的 Pipeline IR 删除反例。FMHA F19 的动态 SMEM 额外预留可能影响驻留，但属于真实 launch 预算与资源记账范围，已转为 `scope_decision`，不伪造无用数学对象来填 resource。

## 必须排除的误报和未决事实

| finding | 本轮判定 | 不能提出的“修复” |
| --- | --- | --- |
| template_fa4_decode F06 | existing_structure | 条件 `wait` 加现有 `Correction[0]/zero:output@done` 可承载；不能说缺清零事件，也不能把它提前成所有 K/V 的全局启动 barrier |
| template_qsa_prefill F09 | existing_structure，次级 representation_cost | 四个真实静态 warp、0/1 repeat、PV 前驱可以表达；需要保留完整 slice/rendezvous。不能给未触发 warp 增加等待，不能把局部展开方向说成已验证全文修复 |
| template_gdn_prefill F11 | no_gap，仅通知流身份子论断 | 已有发布/全读者关系未缺；独立/共用流、ready-only 身份没有已证不可传递等待或一阶成本反例，不恢复早已撤回的 missing_field_or_event 结论 |
| GDN F09/F11、GEMM F11、FMHA F13/F16、MoE F13/F16/F17 | existing_structure | 更新所有 action 引用后可用 wait/compute/produce/模式/已有域；不能只删动词定义留下未知动作 |
| GDN/FA4/Sparse/FMHA 的资源 alias findings | existing_structure | 同一资源多 view 已表达字节共用；读者完成仍需真实边/保证，view 不自动同步 |
| PageIndex 环、HalfMax、GDN final mini、Sparse barrier 地址别名、FMHA cold/xpose 退休 | source_or_uncertain 或含此次级判定 | 不造一条来源没有的 release/reuse 边，再宣称 schema 修好了 kernel |
| release 与 kernel_done | 已存在 | 不再提缺 read_done / retirement / kernel completion 事件 |

模板 GDN F09/F10 等删除反例按“删除所有同义承载后”成立；如果 realization 或另一动作仍完整定义同一保证，某一段当前只是重复，不能把两段都分别算独占核心。MoE F18 也说明数学值等式不等于具体读取绑定：`Offs[E]=M` 不自动补齐当前 `invariant` 的缺失 offset。本次仅认可保留读绑定的原审计结论，不把整段删掉。

## schema_docs 的建议次序

1. **P0：收紧现有写法规则及反例。** 已命名跨 role 前驱入 waits_for；通用计算/等待优先用已有词；已有公式、slot、域、view 和事件不再在 verb 重写；来源未知保持未知。对 `done`、`release`、no-payload/空域的现行区分给一组可复用的短反例。
2. **P1：编写少量共享动作语义定义，不先改 YAML 固定字段。** 合并上述机制 1、2 和预取特例；清楚列出“通用定义负责什么、实例必须绑定什么、现有字段负责什么、哪些保证必须有原语来源”。这样能减少每个实例重述，同时保住操作数、发布、捕获和退役的一次定义。仅把长文搬到公共 meaning 或建立一个无参数的宏库不算完成。
3. **P2：用同源完整节点验证紧凑分工。** 首选 GDN 两 chunk inverse、decode global/compact、QSA 条件 warp 退役和 prefill 私有 combine。比较保留短契约、现有结构完整展开、少量新通用接口的总表示成本。比较项包括共用规范长度及新增边/对象数，不只比较实例字符数。没有证明前不设计动态 member/线程 DSL。
4. **数值与来源独立处理。** 数值放已有 math/arithmetic；atomic 的业务算法保留在数学/短动作定义；host、跨调用 allocation/缓存、预算范围另决。不得通过扩大“动作接口”吞下完整主机运行时。

共享语义定义能回答用户对“文本垃圾桶”的主要关切：把重复的、通用的语义移入有边界且可核对的规范，让实例只留下不可推导的参数和算法决定。它不能让所有文字消失，也不能让读写集合自动变成机器可解析结构。若未来明确要求结构独立比较/检查这些集合，才需要基于这里的反例设计一个通用动作接口；当前证据支持先优化 schema_docs 的语义和写法，不支持立刻增加一批 operator 专用字段。

本产物只新增研究文件，未修改 schema_docs、样本 YAML、原标注或生产提取结果；未运行提取、GPU 或全量审计。
