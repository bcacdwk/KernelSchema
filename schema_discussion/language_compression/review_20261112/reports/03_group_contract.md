# 组内分工：修改原建议表，正式规则不改

**判断：修改原建议表；正式 docs 不改。** 现行 DOC4/5 已明确三种 member、必要等待与动作粒度；需修正的是原表的混称及“普通原语内部协作不记录”的过宽措辞。
影响负载、局部顺序、提前发布或存储复用的分工，不能因没有独立 role 就删掉；现有字段足够表达本次核查的反例，尚无新增 subgroup 字段的必要性证据。
“零新增”不等于把所有细节搬进 describe：已命名跨角色前驱必须成为 waiting；其余只保留有辨别力的短定义。

本报告静态核读模板 GDN、提取 GDN、模板 QSA、模板 decode 的定点材料，未运行 GPU，也不批准源算子正确性。
固定 YAML 来自 corpus：GDN/QSA/decode 为 template，提取 GDN root 为 approved；状态不替代本次必要性判断。
FMHA/MoE 未作语义复核；没有把旧规则、旧审计的“缺口”直接当作当前能力结论。
主线定位依据为 [SCHEMA_VERBOSITY:84、228、236][V]；旧十算子材料仅帮助定位，下面的保留和删减均核到当前规则与实际材料。

## 先把 member 分清

| 位置 | 现行含义 | 不能推导的事实 |
| --- | --- | --- |
| `work_schedule.phase.<p>.mapping.member` | 对 worker 内各 `worker.unit` 成员登记局部数学区域；列表位置为成员号 | 一个 CTA 内四个 warp 怎样随层次重分工；它不是 warp 调度表 |
| `tile_pipeline.phase.<p>.role.<r>.info.member` | 固定选取承担此 role 的执行单元；可参数化，但整个 phase 不随循环或动作结果改变 | 每个 warp 对应哪个 chunk、先后处理哪些块、哪些 warp 需要等上一轮 |
| `Role[member]::verb:noun(...)` | 选择该 role 在哪个 worker 成员上的事件；`[*]` 要求所选实例全部成立 | `SoftmaxA[0]` 不能表示 SoftmaxA 的第 0 个 warp；单 CTA 中它仍是整组 role 的事件 |

依据：[DOC4:149–183][R4M]、[DOC5:82–110][R5R]、[DOC5:570–589][R5M]。worker.unit 若为 CTA，事件下标就是 CTA 成员号。
`storage.member` 同样选择 worker 单元的存储副本；容量不是按 role 的 warp 数重复计算（[DOC5:355–376][R5S]）。
因此不把 GDN 的两个 warp 组写成两个独立 worktile，也不改 worker.unit 来获得方便的 warp 事件编号。
`info.describe` 只概括角色职责；具体分摊、动作输入、存储绑定和完成保证归各 `info.using_verb.<verb>.<noun>.describe`（[DOC5:185–200][R5D]）。
`storage.meaning` 解释区域用途，`index.meaning/definition` 定义索引身份与取值；它们都不产生线程分工或等待。

## 何时短定义，何时结构化

| 事实 | 最小载体 | 删除或合并会怎样 |
| --- | --- | --- |
| 相同参与者中，分摊与重复计算不同 | noun.describe 中一句负载定义 | 改变工作量估算，即使结果公式一样 |
| 动作内部按层重分工，只有整组入口/出口被外部使用 | noun.describe 的层次、局部顺序、必要汇合 | 否则看不出并行与串行；无需生成每条 shuffle/MMA 的事件 |
| 某些参与者独立等待、推进或提早发布 | 固定 role、`if`、动作及 `waiting` | 整组等待会丢失原有推进机会，隐藏前驱会漏依赖 |
| 统计量捕获、读完归还、部分发布供后续独立引用 | 现有基础动作、storage/view 与边 | 不能合并为更晚的整块 done，也不能降低 store 的 done 保证 |

[DOC5:20、28–31][R5G] 已明确上述主要粒度；[DOC5:137–146、198–200][R5D] 允许计算内普通读写，要求可引用交接使用结构。
三种 member 的区分已有 DOC4:149–183 和 DOC5:570–589；“describe 中的负载分摊”也已见 DOC5:28。原表和旧实例应使用这些规则，正式 docs 无需再加同义澄清。

## GDN Inverse：一个动作可保真，原短句仍需澄清

真实旧文 [模板 root:242][TG]：
> Invert both chunks in place with group rendezvous between levels; two warps per chunk at widths 8/32/64, all four across each chunk's 16x16 diagonal blocks.

其 member 是 `warp[0:4]`；数学已经给定 8→16→32→64 及 BF16 中间舍入，Tring 给定 A/Tinv 的同槽关系。
源码 `_partial_pair_inverse` 与 `_finish_pair_inverse` 连续调用；执行分摊如下（[solution.py:496、518–526][TGC]）。

| 层 | 实际分工 | 应保留的顺序 |
| --- | --- | --- |
| 8 | warp 0/1 处理第一 chunk，2/3 处理第二；每 chunk 八个对角块 | 进入前和本层后四 warp 汇合 |
| 16 | 每个 warp r 做两 chunk 的第 r 个对角块 | **每个 warp 自己**先第一 chunk、后第二；两次调用之间没有整组屏障 |
| 32/64 | 恢复每 chunk 两 warp；32 层各处理一个块，64 层两 warp 合作 | 层末四 warp 汇合；merge 原语本身的必要协作保证仍成立 |

不能把 width 16 改成“全组完成 chunk 0，再开始 chunk 1”；这会添加源码没有的交叉先后。
可以把三层映射与这一局部顺序写入一个 noun.describe；分层变化不要求动态 `info.member`，四 warp 始终承担这个整体动作。
数学公式、BF16 舍入、槽位复用沿用既有定义；不在契约中再抄 8×8 消元过程或线程地址。
保真的代价是依赖人工核查该短映射；现有事件图不会自动生成每个 warp 的内部事件。

只取 width 8 比较，两种写法都保持整组入口与层末汇合；不是以局部片段声称整个 Inverse 已验证等价。

| 方案 | width 8 的新增表达 | 得到什么 |
| --- | --- | --- |
| 在现有 Inverse 内保留 Pair8 短契约 | 1 个 noun 声明、1 个 pipe 调用；不新增 role/跨 role 边 | 足够保留两 chunk 的并行负载分摊；整段原 invert 内融合时还可省掉这个切点 |
| 静态拆 Pair0/Pair1 两 role | 2 个 role 外壳、2 个 noun/pipe 调用、2 条入口规则、1 条含两前驱的出口规则 | width 8 的两组进度可以独立引用；没有真实消费者时只是多出接口 |

完整局部对照和统计口径见 [03 草图][SK]；共用物理 warp 只分时使用，不把它们计成新增 warp。
若继续展开全部层，16 层仍须表达四份“本 warp 的 chunk 0→1”，不能用两份整组 serial 动作代替。

## GDN finish_chunk：必须留下的是作用域与来源限制

[提取 root:638–660][EG] 把 full/mini 的 readout、更新、sweep 与最终 staging 合在一个动作；“各 warp 独立推进”有实际意义。
full 路径每 warp 只 sweep 自己先前写的 Out 字节，源码只用 `__syncwarp()`；跨 warp 的下一轮复用由下轮入口 CTA join 保护（[gdn_kernel.cu:2753–2785][EGC]）。
若机械拆为整组 serial 的 readout→update→sweep→staging，会在原先各 warp 之间添出全组先后；可留一个有这些作用域保证的动作，或确有需要时静态拆 warp。
mini 路径的 DU0 发布后才读 Kc row 0，随后直接进入收尾；final-state stage 0 的写入前没有新的 CTA 读退役保证。
具体条件是 VTT=64、末 chunk 长度为 1、请求 FinalState 且 `(c_n-1)%NSTAGE=0`（[root:651–660][EG]；[源码:2499–2550、2819–2839][EGC]）。
可直接复算冲突：warp 1 读取 key 16 的 Kc row 0，地址为 stage 0 +32 字节；warp 0 staging 的 key 0/value 8 写同一地址。
这是旧读与新写的物理别名，加上缺少前置跨 warp 顺序；末尾 staging 后的 CTA join 不会倒过来保护旧读。
不能把这段限制删成“原语内部协作”，也不能用整动作 done 伪造内部先后；保留短限制，不在研究中加源代码没有的 barrier。
这是**来源缺陷/保证不足**，不是新增 subgroup 或 read 事件的证据；完整源码正确性仍未由静态阅读证明。

## QSA 与 decode：已有表达需要真正用上

[QSA root:324–325][QY] 的原文是 “Only warps with move_ref true wait for PV(n-1) when i>0”。
源码先逐 warp vote，再仅在 `i>0 && vote && isA` 时等待 pvdone；A 修正该 warp 的 Y 行后发布 P（[qsa_kernel.py:517–565][QC]）。
等待的是已命名 MMA/PV 的事件，原 waits_for 只有 peer HalfMax，没有这个前驱；现行 [DOC5:198–200、500–507][R5D] 已不允许靠说明加入它。
精确展开需要四个固定 A-warp role，各自先保留 vote/weights，条件 await 既有 PV.done，必要时 rescale，然后发布对应 P 行片。
不能写 `SoftmaxA[r]` 来寻址 warp，也不能把四个不同 vote 合成整组无条件 await；不触发的 warp 原本可继续。
代价是四份动作声明、跨新动作保留的寄存器副本及相关 HalfMax、P、ticket ack、Q、drain 的端点重接；本报告给局部边，不宣称完成整份 QSA 改写。
这属于**实例没用好 + 展开成本**；已有 static role/if/waiting 能表达，未证明新字段更便宜或必要。

decode 的 [root combine:434–440][DR] 与 [q4_compact:353–358][DC] 都是 `warp[0:4]`，参与规模相同，工作方式却不同。
前者每 warp 分摊 M/Z 归约并广播保留的权重；后者每个线程扫描所有 maxima、为 active split 重算权重与 Z。
源码两支分别在 [gqa_decode.py:467–530、531–555][DCode]，调用选择也核到 paged.py:528。
所以须留“warp 合作并广播”与“逐线程重复统计”的短区别；不能删后让两份设计看起来只是相同的四 warp compute。
此外前者读所有 nominal payload，后者跳过 inactive；此读取范围影响流量与有限值前提，也不能当线程细节删除。
没有独立消费者要求逐 lane 的权重事件，拆成 128 个 role 不增加必要设计信息。

decode cluster 提供另一个 member 正例：多个 CTA 是同一 worker 的成员，活跃成员的 reduce-add 等待成员 0 的 output-zero 可见性。
其 `Correction[0]` 确实表示 CTA 0，不是第一 correction warp；[q1_cluster:217、307、315–323][DCluster] 现有文字与边未完全对应。
最小修复是在 active 条件下，原输出消费位置增加 `await:OutputZero` 及来自 `Correction[0]::compute:OutputZero` 的 done 前驱；保留 K/V ramp 的提前执行。
源码 [paged.py:1389–1397、1856–1887][DPaged] 支持这个晚等待；无需增加所有 CTA 的启动屏障或新的成员字段。

## 一处可安全缩短的真实原文

[提取 GDN root:663–666][EG] 的 final_store 局部原文，按空白归一后为 211 字符：
> Per thread prefetch one float4 into a register; in each of VTT/8 rounds load the next float4 before storing the current one. The last round has no next load and its subsequently assigned current value is unused.

可替换为 102 字符（本处净减 109，不外推全仓比例）：
> Per thread, stream VTT/8 float4 blocks with one-block lookahead, skipping lookahead on the last round.

保留每线程、VTT/8 次 float4、一个块前视及末轮不越读，删掉已蕴含的赋值过程和死值解释；[源码:2844–2863][EGC] 可直接核对。
原字段其余的条件输出、转置方向、无后续 worktile/推测 chunk、kernel completion 退役与无额外 drain 全部保留。
未删除前视机制：源码明确用它重叠 load/store；只删死赋值说明不损失该成本选择。对应现行载体为 `using_verb.store.FinalState.describe`。

## 字段、规则与原表的最小改动

| 项目 | 决定与依据 |
| --- | --- |
| 新增字段、动词、事件种类 | **0**；本次反例可由现有角色、动作、条件、存储和事件引用表达 |
| 删除正式字段 | **0**；不删除 info.describe、noun.describe、storage.meaning 或 index.meaning/definition |
| 三种 member 的规则增量 | **0，不改 DOC4/5**；DOC4:149–183 已定 worker 单元分摊，DOC5:82–91、570–589 已定角色成员与事件副本；只改原表混称 |
| 命名前驱进 waiting 的规则增量 | **0，不改 DOC5**；198–200、500–507 已禁止说明替代必要前驱，并规定条件用 if；QSA/decode 要改旧实例 |
| 动作粒度/负载分摊的规则增量 | **0，不改 DOC5**；20、28–31、137–146、198–200 已覆盖拆分边界、负载说明和原语抽象；按既有规则审实例 |
| 实例修正 | QSA 条件 PV 等待、decode output-zero 等待用现有结构；GDN Inverse 写清每 warp 顺序；GDN 来源限制保持可见 |
| 普通实现文字删减 | 只做上面的 109 字符局部替换；没有对所有内部协作、layout 或 barrier 文本作批量删除授权 |

原表建议改成：**“组内分工暂不加字段：区分工作分摊与角色成员；短契约保留影响负载、局部顺序和复用的分工。独立推进、部分发布、捕获或归还用现有 role/动作、if 与 waiting；仅删无设计影响的原语步骤。”**
规则增量去重结论：本主题没有发现正式 docs 仍缺的一句不可替代说明；不为这些旧实例问题重复增补防御性条款。
当前能力、实例未用、短定义、来源限制分别已有证据；**真实结构缺口未证实**，不把较长展开或旧稿不合法直接升级为缺字段。
完整证据、读取范围、状态与局部计数在 [03 数据][DATA]；语法草图仅核局部语义，不是十算子全量审计或可直接替换的完整 Plan。

[V]: ../../../../schema_extract/archive/six-operators/20261109-methodology-review/SCHEMA_VERBOSITY.md:84
[R4M]: ../rules/4.design.work_schedule.md:149
[R5R]: ../rules/5.design.tile_pipeline.md:82
[R5M]: ../rules/5.design.tile_pipeline.md:570
[R5S]: ../rules/5.design.tile_pipeline.md:355
[R5D]: ../rules/5.design.tile_pipeline.md:185
[R5G]: ../rules/5.design.tile_pipeline.md:20
[TG]: ../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/template_gdn_prefill/root.yaml:242
[TGC]: ../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/template_gdn_prefill/source/aka_solution/solution.py:518
[EG]: ../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml:638
[EGC]: ../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu:2499
[QY]: ../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/template_qsa_prefill/root.yaml:324
[QC]: ../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/template_qsa_prefill/qsa_kernel.py:517
[DR]: ../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/template_fa4_decode/root.yaml:434
[DC]: ../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/template_fa4_decode/q4_compact.yaml:353
[DCode]: ../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/template_fa4_decode/source_gqa_decode.py:467
[DCluster]: ../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/template_fa4_decode/q1_cluster.yaml:217
[DPaged]: ../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/template_fa4_decode/source_gqa_decode_paged.py:1389
[SK]: ../sketches/03_group_contract.md
[DATA]: ../data/03_group_contract.json
