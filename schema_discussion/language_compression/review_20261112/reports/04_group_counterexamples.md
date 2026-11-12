**结论：修订原建议，保留“不加新字段”；“普通原语内部协作不记录”须限定为可删除内部指令，不能删除设计依赖的分摊、局部先后、发布和退休保证。**

建议新增字段 **0**、删除字段 **0**，新增事件种类／执行模式 **0**，正式规则修改 **0 处**。DOC0–6 保持现状；现行规则已覆盖本主题需要的能力，剩下的是原表措辞修订与旧实例粒度、必要保证的整改。

首要入口是 [SCHEMA_VERBOSITY:84](../../../../schema_extract/archive/six-operators/20261109-methodology-review/SCHEMA_VERBOSITY.md:84)：GDN 的 1,300 字符后缀和 FMHA 的 619 字符 advance 集中了组内推进及退休语义，不能按长度整段删。
本次语义核对限定为五份固定 YAML：模板 GDN、QSA、FA4 splitkv_dual_load，以及提取 GDN、FMHA 的当前 root；FMHA 是暂停 working 中的 approved 节点，不代表整树完成。
输入由 [inputs.json](../inputs.json) → [corpus.json](../../../flow_review/corpus.json) 定位；六份使用到的源码均核对身份。corpus 未列模板源码，其三份副本另按模板 meta_data.source 与固定 commit 的 git show 核对字节一致。
没有运行 GPU、改正式规则或算子，也没有把局部短写视作完整迁移或正确性验收；精确范围、哈希、原文和计数见 [证据 JSON](../data/04_group_counterexamples.json)。

**现行规则究竟已经解决什么。**

| 旧问题／易误读处 | 本轮 rules 的能力 | 本次判断 |
| --- | --- | --- |
| 一个 verb 字符串混装多个对象及模式 | [DOC5:150](../rules/5.design.tile_pipeline.md:150) 已按 verb/noun 分别声明 execution_mode、describe | 旧的多对象契约容器问题已改善，不再建议另加动作说明字段。 |
| 总体职责无短写位置 | [DOC5:82](../rules/5.design.tile_pipeline.md:82) 已有必填 info.describe | 稳定分工可写这里；随动作改变的分工写对应 noun.describe。 |
| 提前归还被误作整个写入完成 | [DOC5:128](../rules/5.design.tile_pipeline.md:128)、[234](../rules/5.design.tile_pipeline.md:234) 区分 load/store/release 和局部使用权 | 已有答案；不加 read-complete 事件或全员完成字段。 |
| 条件等待、独立推进 | [DOC5:200](../rules/5.design.tile_pipeline.md:200)、[286](../rules/5.design.tile_pipeline.md:286) 有 waiting、if 和逐 noun 模式 | 条件不能藏进另一角色已命名事件的短契约；必要时增加实例。 |
| 局部范围与存储退休 | [DOC5:355](../rules/5.design.tile_pipeline.md:355)、[389](../rules/5.design.tile_pipeline.md:389)、[426](../rules/5.design.tile_pipeline.md:426) 有 storage.meaning、view.region 和全必要读者约束 | 新 meaning 不是退休协议，view 也不产生同步。 |
| domain 旧写法 | [DOC5:328](../rules/5.design.tile_pipeline.md:328) 已用 index 的 meaning/definition 包装 | 解决索引定义，不能把索引变量当动态 warp 成员选择。 |

三个 member 必须分清：[DOC4:156](../rules/4.design.work_schedule.md:156) 的 mapping.member 是 worker 单元的数学区域；DOC5 info.member 是执行单元选择；[DOC5:589](../rules/5.design.tile_pipeline.md:589) 的 Role[member] 下标也是 worker 成员，**不是 warp 编号**。
例如单 CTA 的 SoftmaxA:warp[0:4] 只有 CTA 成员 0，SoftmaxA[2] 不能指第 2 个 warp。动作实例粒度见 [DOC5:183](../rules/5.design.tile_pipeline.md:183)；member 不自动产生四条独立 pipe。
done 是所声明动作结果／使用范围的保证；serial 只约束该动作完成到后续发起，不推出动作内部每一步都经过 CTA join。反过来，也不能用 describe 把一个已声明的全组 serial 完成偷偷改成四个局部完成。

**G1 · GDN 两 chunk：分摊要留，内部独立推进未必值得拆。**

[原文:242](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/template_gdn_prefill/root.yaml:242)：“Invert both chunks in place with group rendezvous between levels; two warps per chunk at widths 8/32/64, all four across each chunk's 16x16 diagonal blocks.”
实际分工由 [源码:518](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/template_gdn_prefill/source/aka_solution/solution.py:518) 和 [524](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/template_gdn_prefill/source/aka_solution/solution.py:524) 确认：8/32/64 层按 warp 对分两个 chunk；16 层每 warp 各做两个 chunk 的同号对角块，自己先第一个再第二个。
删掉这句而只留 member，会容许四 warp 逐 chunk 整组处理，或重复处理同一块；成员数相同，工作分配与重叠关系不同。
把 16 层改成两个整组 serial 动作，则强加“全部第一个 chunk 完成后才发起第二个”。源码两次 _blockwise_diagonal_8x8_to_16x16 调用之间没有该组屏障，不能这样展开。
上游 [inverse 数学:103](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/template_gdn_prefill/root.yaml:103) 已给算法和舍入；[pipe／消费:244](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/template_gdn_prefill/root.yaml:244) 只在整个 inverse 后发布 Tinv，没有外部消费者引用某个内部逆矩阵层。
因此保留一个 compute:Pair(t,p)，用现有 noun.describe 澄清分摊即可；Pair 的两项仍是 n[t,2*p]、n[t,2*p+1]，A 仍原地写回 Tring。

```text
Invert both A in place. Widths 8/32/64: warp[0:2] takes the first chunk
and warp[2:4] the second. Width 16: warp r takes diagonal block r in
the first chunk, then the second, without an intervening group join.
Keep the level rendezvous.
```

这条消歧文字是 **156 → 236 字符**，增加 80 字符而非声称压缩；它保留的就是局部先后。层间与 merge 原语已有保证仍保留。
[旧局部展开:58](../../../../schema_extract/archive/20261109-ten-operator-text-audit/gap_review/code_examples_actions.md:58) 那种方案仅抽出 width 8，就新增两个静态 role、两个计算族及一个汇合动作，还要处理同一物理 warp 分时承担旧／新角色。按当前规则补齐 info.describe 和逐 noun 声明后也不会免费。
这里 **新增 role／动作 0** 更合适；不增加动态 member、组内 partition 或线程子程序字段。

**G2 · FMHA：可以真正缩短，同时留下看似低级的必要参与范围。**

[原 advance:603](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/root.yaml:603) 的完整 619 字符原文：

```text
Apply normalize and the numerator rescale using the retained Score and RowState.
After the first group, any warp with an alpha unequal to one reads/multiplies/stores
its NumState rows in 32-channel chunks and waits for the stores; all lanes of that
warp participate. The first group never reads NumState. Compute probabilities and
denominator partials, pack Prob8 into TMEM, then wait for its stores. A before-thread-sync
fence and CTA barrier publish all probability and scale writes before thread 0's next PV.
Score and group temporaries are now dead; MaxState, DenState, Prob8 and NumState retain their defined uses.
```

按当前字段给 StateProb 的局部短契约如下；旧 read_score 对应 compute:Score，其他 noun、存储绑定及完整 pipe 的迁移不在此片段中：

```yaml
compute:
  StateProb:
    execution_mode: serial
    describe: >-
      Apply normalize to register Score from both compute:Score halves and carried RowState.
      After the first group, a warp rescales its TensorMemory.NumState rows iff any lane has alpha!=1;
      all its lanes participate. The first group never reads NumState.
      Complete NumState and TensorMemory.Prob8 writes, then join the CTA to publish Prob8 and preceding
      scale writes to the PV issuer. Retire Score and group temporaries;
      retain RowState, Prob8 and NumState.
```

值文本 **619 → 450，减 169 字符**。计算／舍入沿用上游，TMEM 与寄存器位置沿用原存储，PV(g-1) 的退休位置仍在两次 QK 之后；不能只应用短句而漏迁输入来源与依赖。
删除的是 32-channel 指令展开、fence 拼写等已有 [realization:663](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/root.yaml:663) 绑定的重复细节；没有删除集体参与、写完成或 CTA 发布。
“所有 lane 参加”不能按低级细节删：[源码:4976](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_fmha/source/candidate/fmha_tcg.cuh:4976) 用 __any_sync 决定整个 warp 的 RMW；[wrapper:133](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_fmha/source/candidate/fmha_tcg_ptx.cuh:133) 实际是 sync.aligned TMEM 读。改成各 lane 独自按 alpha 分支不满足该原语的参与要求。
首轮不读 NumState、Prob8/scale 到 PV issuer 的发布也分别由 [4974](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_fmha/source/candidate/fmha_tcg.cuh:4974)、[5052](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_fmha/source/candidate/fmha_tcg.cuh:5052) 确认。它们归现有动作契约和实现绑定，不需要拆四个 rescale role。

**G3 · QSA：短契约不能替代真实的跨角色条件等待。**

[原文:325](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/template_qsa_prefill/root.yaml:325) 写“Only warps with move_ref true wait for PV(n-1) when i>0”。[源码:546](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/template_qsa_prefill/qsa_kernel.py:546) 确为 i>0、vote、isA 三层条件；B 侧不做 Y 修正。
最小删除反例：i=1，只有 A 的 warp 0 触发 vote，前一 PV 尚未完成。删等待允许 A0 改写 PV 仍在更新的 Y；无条件整组 await 则禁止其他 A warp 先完成各自 P 分片。
后者虽然最终 PV 仍要等全部 P 发布，却改变了未触发 warp 的局部推进；“消费者最终需要全块”不能推出“所有生产者都必须提前等同一前驱”。
现行可用办法是把需要分别引用的 A 执行者静态拆为 A0–A3，member 分别是 warp[0:1]…warp[3:4]；保持每 warp 的整个必要先后链。以下只列 A0 的条件边界：

```yaml
pipe:
  - "for(t)":
      - "for(n[t])":
          - "compute:Weights(n)"
          - "if(i > 0 and move_ref(i,0))":
              - "await:PriorPV(n)"
          - "compute:P_A0(n)"
waiting:
  released:
    - "{done await:PriorPV(n)} ← [{done MMA::matmul:P(n-1)@Vblk(n-1)}]"
```

这是现行语法的**局部方案**：Weights 必须先捕获 peer maximum／形成 vote；await 声明 serial；P_A0 绑定 packed rows [0,32) 的 A 半列并按上游 alpha 修正 Y。其余三个 role 使用对应常量行起点。
完整拆分还要重接 HalfMax、P 四份发布、ticket ack、Q refill 和 drain，MMA 仍等待四份 A 加 B。净增 A 角色 3 个，不是新增字段；本次没有伪称已完成整份等价展开。
尤其 [HalfMax:327](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/template_qsa_prefill/root.yaml:327) 已声明读退休未获保证；源码 [509–511](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/template_qsa_prefill/qsa_kernel.py:509) 是先写、join、读 peer，下一轮写前缺少额外 read-complete 保证。不能让角色拆分顺手把源缺陷“修好”。

**G4 · 局部读取退休：已有 release 够用，但不能改成全块写完。**

QSA 的 [drain 契约:331](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/template_qsa_prefill/root.yaml:331) 在末片装入寄存器后归还 Y 的半区，最后 gate/store 仍未完成；源码 [615–624](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/template_qsa_prefill/qsa_kernel.py:615) 先发 odrained/qready，再在 [625](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/template_qsa_prefill/qsa_kernel.py:625) 后计算并写尾片。
当前规则可用 load/compute 捕获加 release:YHalf 表达，view.region 指明 A/B 两半；下一 worktile 首 PV 保留两半 release 的前驱。不能把 release 替成最终 store.done，否则额外等待尾部 gate/store；也不能只等一个 warp 的读完成。
FA4 combine 的 [原文:293](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/template_fa4_prefill/splitkv_dual_load.yaml:293) 则是另一种粒度：“Reads are thread-private, so no per-stage CTA barrier is needed.”
源码 [600–630](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/template_fa4_prefill/flash_fwd_combine.py:600) 按每线程持有行的末有效 split 裁剪，各自 cp_async_wait_group 后读取自己的分片；LSE/权重交换处的 CTA join 则在 [499](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/template_fa4_prefill/flash_fwd_combine.py:499)、[598](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/template_fa4_prefill/flash_fwd_combine.py:598)。
私有拷贝→本人读取的关系不能删，否则可误成共享读者必须整组等待；也不能把八 warp 名字当成八套足以表达所有线程私有完成的事件。
最小保留句是“Each thread waits for and reads only its own copied slice; no per-stage CTA join.”；四槽、三次预取、尾部裁剪及非正权跳过仍由对应结构／契约保留。
若需要改变粗粒度 pipe，优先把无外部引用的私有小流水封装在一个高层动作内，明确私有分片和循环内先后；不要把逐 split 的 serial 全组完成宣称为源中的逐线程退休。这沿用 DOC5:20／200 的现有动作粒度，需要改旧实例，未证明需要线程事件系统。

**G5 · 删除元解释可以，删除源缺陷不可以。**

GDN [finish_chunk:638](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml:638) 可删原句：“Keeping this suffix in one action does not order those reads before other warps' writes.”
短写是其余文字原样保留，特别保留“无 CTA join”、最终 mini 冲突条件和“source supplies no read-retirement guarantee”；归一化字段 **1,300 → 1,211，减 89 字符**。这些相邻句已完整给出该句解释的事实。
本轮额外核实了具体重叠：mini 只存在于 [VTT=64、Lc=1](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu:2448)；最终 c%NSTAGE=0 时，warp 4 的 Kc 行 0/key 64 读地址是 stage 0+8192 B，warp 1 的 FinalState[key=31,value=33] 写也是 +8192 B。
读地址来自 [2508](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu:2508)，写地址来自 [2827](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu:2827)，其中 EPITCH=65；mini 在 [2550](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu:2550) 直接 continue，最终状态的 CTA join 在写后 [2839](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu:2839)。
这证明的是跨 warp 读取退休保证缺失和真实字节冲突，不是测得竞态发生。追加一个 ownership／join 字段不能产生源码没有的保证；增加写前 join 也属于改变源实现，不能记作精简收益。

**归类与实际修改范围。**

| 归类 | 本次落点 |
| --- | --- |
| 当前已有能力 | 逐 noun 契约、静态 role、if/waiting、release 与 region；G1/G2/G4 可直接使用。 |
| 实例没用好 | QSA 将另一角色已命名 PV 的条件等待藏在 publish 文字中；G3 需重接现有结构。 |
| 实例仍需短定义 | 必要分摊、warp 集体参与、私有读取，以及局部先后不扩大为全组完成；不增加规范字段或解释义务。 |
| 真实结构缺口 | 本次未证成；精确线程事件图超出已取的动作粒度，不等于必须新增字段。 |
| 来源缺陷 | GDN final-mini 重叠读写、QSA HalfMax 退休未证实；不能借新结构补成事实。 |
| 实现细节 | 指令展开、fence 拼写及重复元解释，在保证和必要原语绑定仍在时可缩。 |

正式规则无需再改：成员选择已由 [DOC5:83](../rules/5.design.tile_pipeline.md:83)／[589](../rules/5.design.tile_pipeline.md:589) 定义，动作的来源成员和存储绑定由 [187](../rules/5.design.tile_pipeline.md:187) 覆盖；不必再增加“member 不是分工表”的说明。
局部动作与已命名跨角色依赖的边界已有 [DOC5:198–200](../rules/5.design.tile_pipeline.md:198)，done／serial 与内部读写关系已有 [204](../rules/5.design.tile_pipeline.md:204)／[465](../rules/5.design.tile_pipeline.md:465)；GDN/QSA 的旧实例问题不构成新规则缺口。
省略普通实现细节、保留必要原语保证，也已有 [DOC5:20](../rules/5.design.tile_pipeline.md:20)／[191](../rules/5.design.tile_pipeline.md:191) 和 [DOC6:13](../rules/6.realization.md:13)。本轮建议是执行这些规则：GDN 补最小分摊消歧、FMHA 缩短重复细节、QSA 修正条件等待，并保留源缺陷。
不采用新增 action.member：它单独不能给 QSA 建立每 warp 独立先后及条件事件，也不能解决 FA4 的线程私有读取；仍需定义作用域与完成含义。GDN 只需增加 80 字符消歧，尚无证据支持为此扩语言。
原表第 14 行建议改为：“必要分摊用现有 member、info.describe 和逐 noun 短契约；需要独立引用的推进、发布或复用边界才拆 role／动作，并保留实际参与范围。普通原语的内部指令省略，但设计依赖的协作和退休保证保留。”
上述两处删短合计只针对被引字段的文字，不代表整份迁移后的净字数；新增的消歧也单列计算。没有把结构展开更长自动判作应新增语法。
阅读定位还使用了十算子 TEXT_INVENTORY 的所选条目、[SCHEMA_DISCUSSION:108](../../../../schema_extract/archive/20261109-ten-operator-text-audit/SCHEMA_DISCUSSION.md:108)、[SCHEMA_LANGUAGE_GAPS:88](../../../../schema_extract/archive/20261109-ten-operator-text-audit/SCHEMA_LANGUAGE_GAPS.md:88)、既有 phase_grammar/view_review 的局部退休线索；这些只作定位，本报告的关键结论均回到上述当前规则、固定 YAML 和源码。
