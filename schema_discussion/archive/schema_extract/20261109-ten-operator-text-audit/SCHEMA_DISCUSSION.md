# Schema 文本为什么写长：五个值得讨论的问题

**可以先抓住两个问题：常见动作需要作者反复解释，计算与外部之间的等待关系缺少统一写法。** 但“调用边界不明确”容易让人误以为我们不知道源码怎么执行。更准确的是：我们知道它怎样执行，重要的先后关系却只能写在说明里。

围绕文本膨胀，我建议分成下面五件事讨论。**前两项优先补清规范，第三、四项改善写起来太长的问题，第五项先决定 schema 应该管到哪里。**

每项末尾都有代码对照。**“现有字段”表示不需要扩充字段；“新写法草图”表示供讨论的提案，当前不能直接提交。** 代码只展示相关部分，省略的计算、对象定义和依赖仍需保留；没有改动原始 schema。

## 1. 同类动作的说明书，每个算子都要重新写

**问题：** 像工厂里每张工单都写“备料”，旁边又各自解释：拿哪些材料、交出哪些东西、什么时候可以收走原材料。Kernel 里也一样：一个动作可能只读一次数据，却同时生成两个结果；也可能先把数据复制出来，让原来的缓冲区提前交给别人使用。这些区别会影响访存量和能否并行，不能随便删。

**成因：** Schema 已有“读取、计算、写出、释放”等常用动作。但超出通用含义的部分，由作者在 `info.verb` 中自己说明。一些反复出现的做法还没有共同定义，于是每个实例都重新解释，并夹带公式、槽数、已有等待关系等重复内容。

**影响：** 五项中，这项覆盖最广：十个算子，包括四个模板，都有相关事实。它直接影响单份 schema 的篇幅；删错还可能丢掉缓冲区何时能覆盖、哪些结果已经可用等核心信息。并不是整段都必须保留，通常是少数关键句混着大量解释。

**改法：** 改[微流水线的动作说明](rules/schema_docs/5.design.tile_pipeline.md:68)，并与[实现交接](rules/schema_docs/6.realization.md:11)配合。先把反复出现的做法定义一次，例如“复制出独立副本后，原空间可以归还”；实例只说明从哪里复制、结果放在哪里、后面谁要使用。已有字段能表达的数量和等待关系直接引用。

**预期效果：** 同一原理只解释一次，各算子的动作说明缩成必要的差异和参数。先做好这件事，再判断是否还缺字段；不能只给长段落换一个更短、但没人定义过的名字。

**代码对照：先把一个长动作说明整理清楚。**

这是[模板 GDN 的真实动作说明](inputs/template_gdn_prefill/root.yaml:267)。`publish` 同时解释好几个目标，所以一段文字里挤了多件事：

```yaml
verb:
  publish: 'StateOperand reads each incoming S subtile once, snapshots Sbar and decays S in place before publication; R captures/releases KS; Ubar and
    Udec are written from the captured FP32 U before their separate publications; O rounds Y into SMEM and releases its TMEM slot. StateOperand, R,
    Ubar and Udec have independent notification streams. Producers omit empty-stage acquisition: subsequent KS/U/state completion transitively protects
    their fixed operand slots.'
```

**第一步，仍用现有字段，只留下每个目标的独有行为。** 下方保留了这个字段负责的全部目标；公式和存储位置继续引用原有定义：

```yaml
verb:
  publish: >-
    StateOperand: read each incoming S subtile once; form {Sbar, Sdec},
    with Sdec overwriting S; publish both together.
    R captures/releases KS. Ubar/Udec use the captured FP32 U and publish separately.
    O rounds Y into SMEM and releases Y's TMEM slot.
    StateOperand/R/Ubar/Udec have independent notification streams and no empty-stage acquisition;
    subsequent KS/U/state completion protects their fixed slots.
```

不用先看懂所有符号。重点是第一句留下了三个决定：**S 只读一次、得到两个结果、其中 Sdec 覆盖原来的 S。** 这些不能删；长解释可以压缩。归一化空白后，这个字段从 457 缩到 402 个字符。模板这段本来就不算长，因此只改措辞的收益有限。

这一步还没有解决跨算子的共用。下一步应把反复出现的“复制后何时可归还原空间”等规则写入规范，实例只写具体数据与例外；在共同规则尚未定义前，不能直接把上面换成一个含义不明的短名字。

## 2. 内部工序排好了，外面何时送货、何时取货却写在备注里

**问题：** 现在可以表达“本算子中的计算 A 完成后，计算 B 才能开始”。但如果前提是“CPU 生成的表已经上传”，或结果要交给“调用者接下来的工作”，就缺少同样明确的表达方式。

两个实测例子能说明区别：

- **MLA：** CPU 先生成调度表并上传，上传完成后才能启动主 kernel。“先启动 kernel，到真正读表时再等”是另一种安排，会改变资源占用和并行机会。现在这个启动条件主要靠文字说明。
- **Sparse：** 两路工作可以并行，外部后续使用者必须等两路都写完。不能为了省掉说明，直接把两路改成前后排队；那会改变原来的设计。

**成因：** [工作调度的执行顺序](rules/schema_docs/4.design.work_schedule.md:273)主要连接 schema 内已登记的 GPU 计算。外部供给和外部后续工作没有对应的引用方式，作者只能在策略说明中交代。

**影响：** 明确的强例集中在 MLA、Sparse，覆盖面比第一项小；但关系一旦漏掉，可能提前读数据，或者丢掉本来可以并行的安排。它首先是设计能否说完整的问题，不能据此断言它贡献了大量总字数。

**改法与预期效果：** 在[公共约定](rules/schema_docs/0.common.md:39)和工作调度中明确：外部什么条件满足后，哪一步才可开始；哪些结果完成后，外部下一步才可使用。简单情况可以统一约定，确实需要各算子分别连接时，再增加引用外部条件的写法。这样能用短关系替代长说明，同时保留原来的等待位置。还要分清“给后续 GPU 工作安排好等待”和“让 CPU 停下来等完”，两者不是一回事。

**代码对照：把“谁等谁”从段落变成明确关系。**

这是[Sparse 的真实写法](inputs/extract_sparse/probe_hidden_packed.yaml:299)。`probe` 做探测，`prefix` 写前一段输出，`remainder` 写剩余输出：

```yaml
thesis: |-
  Fork after the caller's preceding work, run the full-batch probe on a side stream
  and a packed prefix on the caller stream, then submit the accepted packed remainder
  behind the probe. The remainder need not wait for the prefix. Queue a side-to-caller
  join after the remainder and behind the prefix, so subsequent caller work observes
  all output rows. One probe and two attention launches
  plus the event pair permit overlap without promising simultaneous residency.
flow:
  launch:
  - phase: [probe]
  - phase: [prefix]
  - phase: [remainder]
    start_after: probe.kernel_done
```

下面的结构只写出了“remainder 等 probe”。输入接入、输出交接仍在上面那段文字里。

**预期写法草图：如果决定让每份 schema 明确连接外部条件，可以长这样。** `boundary` 和 `caller` 引用都是新提案：

```yaml
flow:
  boundary:
    input_after: caller.prior_done
    applies_to: [probe, prefix]
    output_before: caller.next_start
    waits_for: [prefix.kernel_done, remainder.kernel_done]
    block_host_for_outputs: false
  launch:
  - phase: [probe]
  - phase: [prefix]
  - phase: [remainder]
    start_after: probe.kernel_done
```

前两行条件表示“调用者前面的 GPU 工作完成后，probe、prefix 才开始”；后面的条件表示“两段输出完成后，调用者后续 GPU 工作才开始”。**没有给 prefix 和 remainder 增加前后顺序，也没有让 CPU 等两路输出完成。** 源码原有的探测轮询仍保留。

这个草图让关系显式可见，不一定减少 YAML 行数。另一个更轻的方案是把这些要求定义成共同约定，实例继续保留原来的三条 `launch`。要比较的是哪一种更清楚、更少重复，不是预先决定必须增加 `boundary`。

## 3. 只写“这组人负责”，组内怎样分工就容易塞进长说明

**问题：** 四个线程小组处理两个数据块，可以两组负责一块、同时推进；也可以每组分别处理两个块中的一部分。参与的人一样，工作安排却不同。GDN 等例子里，这类分工还会随计算阶段变化。

**成因：** [微流水线中的执行角色](rules/schema_docs/5.design.tile_pipeline.md:38)可以登记谁参与，也能拆成更小的角色、动作和等待关系。但复杂分工展开后可能很繁琐。规范又有意保留较大的动作，避免把每条指令都抄进 schema，于是重要分工也容易一起落入文字。

**影响：** 多个算子和模板都有。它可能影响并行、等待和重复计算的数量；不过，多数相关关系已有表达方向，目前不能说 schema 完全不支持。这项与第一项相关：第一项问“这个动作做什么”，这里问“谁做哪部分、能否各自推进”。

**改法与预期效果：** 先选一个完整例子，把两种写法并排比较：用现有字段拆开；用简短、定义清楚的分工规则保留大动作。只保留会改变工作量、数据交接或并行关系的分工。若现有写法确实太长，再研究通用的简写。目标是让人看出串行与并行的区别，同时不把 schema 展开成线程级程序。

**代码对照：这里保留短说明，可能比拆出很多字段更合适。**

[模板 GDN](inputs/template_gdn_prefill/root.yaml:233) 把四个线程小组登记成一个执行角色。下面只摘出成员和求逆动作，其他动作未展示：

```yaml
Inverse:
  info:
    member: "warp[0:4]"
    verb:
      invert: Invert both chunks in place with group rendezvous between levels; two warps per chunk at widths 8/32/64, all four across each chunk's 16x16 diagonal blocks.
```

`warp[0:4]` 只说明“四个小组参与”，具体如何分工全在 `invert` 那一句里。

**预期写法：仍用现有字段，把必要分工列清楚。** 只替换 `invert` 的文字，原来的计算顺序、输入等待和存储复用关系保留：

```yaml
invert: >-
  Invert the two chunks' A in place.
  Widths 8/32/64: warp[0:2] handles the first chunk, warp[2:4] the second.
  Width 16: each warp r handles block r of the first chunk, then block r of the second.
  No extra group barrier between those two width-16 chunks.
  Rendezvous all four warps before width 8 and after each level;
  retain the merge primitive's internal cooperation.
```

这里 `r` 是小组编号 0～3。宽度 16 时，是**每组自己先处理第一块的一部分，再处理第二块的一部分**，并没有“所有人做完第一块，再一起开始第二块”的额外等待。

这份候选比原句更长，但更明确。因此这例不能支持“加结构一定更短”。如果拆成小角色，还要补两组的入口、出口和后续交接，代码会更多；[局部展开对照](gap_review/code_examples_actions.md)保留了具体写法。我们要先决定哪些分工值得记录，再选择文字或结构。

## 4. 分支只改了一部分，共同内容却得反复抄

**问题：** 像几份表单只有一栏不同，却要分别填写大段共同内容。一个算子的不同选择分支可能共用数学定义或大部分动作，但仍各自保存完整段落。

**成因：** 现有[设计引用规则](rules/schema_docs/0.common.md:125)允许子节点引用父节点的**整个设计区域**，文件内也可以复用相同内容。但某个区域只要部分不同，就需要本地写全；各分支的选择条件也要独立定义，不能自动继承父条件。这保护了每个分支的完整性，也带来了复制成本。

**影响：** GDN、Sparse、FMHA、MoE 都有例子，分支多时尤其明显。它会放大整棵分支树的总篇幅，也增加“同一个定义改了几处、漏改一处”的风险。**它解释的是相同内容被写很多遍，不能用来解释每一段为什么本身写得很长。**

**改法与预期效果：** 已有整块引用能解决的复制，先用起来。剩下的再讨论能否让少量公共定义只写一次，各分支明确引用并写出差异。位置仍在公共约定中的设计引用和选择条件规则。预期效果是减少跨文件复制，并让共用定义只需维护一处；代价是必须把引用和差异规则讲清楚，避免看不出某个分支最终用了什么。

**代码对照：同一段算法写两遍，能否改为一份定义、两处使用。**

GDN 的 [root](inputs/extract_gdn/root.yaml:237) 和 [cached_ordered](inputs/extract_gdn/cached_ordered.yaml:98) 中，完整的 `inverse` 步骤相同；但前面的准备步骤不同，因此不能引用整个数学区域。两份文件都重复了下面这段的完整版本，这里只展示开头：

```yaml
- step: inverse
  name: Hierarchical rounded inverse with different merge order at the final level
  math: |
    Initialize Ti to the identity. In each aligned 8x8 diagonal block,
    for every local column j, z[0]=1 and, for d=1..7,
    z[d]=-sum32_{q=0..d-1}(Sys[base+min(j+d,7),base+j+q]*z[q]),
```

不必在这里读懂公式。关键是：**后面的合并、舍入、补齐和输出定义也完全相同，却要跟着复制。**

**预期写法草图：允许引用直接父节点的一个完整步骤。** root 保留原来完整的 `inverse`，cached_ordered 中对应的那一个条目改为：

```yaml
- from_parent_step: inverse
```

`from_parent_step` 是新提案，目前不支持。它应当引用父节点中该步骤的**完整定义**，包括公式和符号；不允许只引用一段摘要，也不隐含继承父节点的选择条件。本分支其他步骤仍按自己的实际设计填写，引用后的参数也要在本分支重新核对。

**已有能力的对照：如果整个区域都一样，现在就能这样写。** 例如模板 FA4 prefill 的 [rect_joint_load](inputs/template_fa4_prefill/rect_joint_load.yaml:15) 与直接父 root 的完整数学区域相同：

```yaml
design:
  math_reshape: {from_parent: root}
```

最后这三行使用现有规则；它与上面的“只共享一个步骤”是两种不同情况。

## 5. 多次调用共用一张“工作台”，交接条件缺少统一写法

**问题：** 一块临时内存可能被多次调用轮流使用。第一次计算还在写，下一次就把它清空，会互相干扰；CPU 读到的结果也可能来自上一次。GDN 的共用临时内存、Sparse 和 MoE 的探测/观察状态中都有相关事实。

**成因：** Schema 已能说明某个逻辑值要保留多久；困难是另一层：几次调用是否使用同一块实际内存，下一次何时可以接着用，读取的是哪一次的结果。这些关系目前没有统一写法，也还没决定 schema 要覆盖多少内存分配、缓存和 CPU 行为。作者于是把相关过程一并写进说明。

**影响：** 忽略真正的共享关系，可能影响复用安全和路径选择；但不能把所有缓存实现都当作核心。有些例子还是源代码自身缺少同步，不能靠增加字段把它描述成安全。这项有实质问题，但“必须扩哪些规则”的把握低于前两项。

**改法：** 涉及[值的保留](rules/schema_docs/3.design.math_reshape.md:98)、[跨计算的数据交接](rules/schema_docs/4.design.work_schedule.md:323)及公共约定。先确定最少要保留的内容：共用哪份存储或状态、读到哪次的值、再次使用要满足什么条件。再判断这些内容需要专门字段，还是短而统一的前提说明就够用。

**预期效果：** 重要的共享和复用条件不会丢；普通的分配器、缓存代码也不会整段挤进 schema。这里先要确定范围，再设计语法。

**代码对照：把“哪几次调用用同一块内存”单独说清。**

这是[GDN 的真实说明](inputs/extract_gdn/root.yaml:379)。同一设备上，缓存的空间不足才换成更大的分配；各次调用依靠同一个 CUDA 执行队列的先后关系复用它：

```yaml
schedule:
  strategy: >-
    The current CUDA stream orders prep, chain and reuse of the grow-only
    workspace across calls. Every consumed handoff slot is rewritten.
    The source provides no cross-stream event, lock or reentrancy protocol.
```

这段其实已经不长。问题是它同时交代了存储共享、本次数据来源和跨次使用条件，现有存储字段不能独自承担这些含义。

**预期写法草图：只有决定把跨调用存储纳入 schema，才考虑这种记录。**

```yaml
shared_storage:
  workspace:
    identity: process.ws_cache
    reuse_if: present && same_device && capacity_bytes >= requested_bytes
    replace_if: absent || !same_device || capacity_bytes < requested_bytes
    rewrite_each_call: consumed_handoff_slots
    reuse_order: same_cuda_stream
    cross_stream_order: not_provided
```

整块 `shared_storage` 都是新提案。它想表达：**缓存对象由进程共用；当前设备相同且空间够时沿用；本次要消费的数据本次重写；跨次复用依靠同一执行队列，源码没有提供跨队列的同步。** 同一设备上的分配只增不减，换设备时则可能重新分配更小的空间。

`not_provided` 只是如实记录缺少保证，不能自动补锁或宣布并发安全。后续还要定义每份交接数据对应这块空间的哪一部分，才能成为完整设计。若我们决定不把这层存储管理纳入 schema，保留前面的短说明就足够；这也是一个合理选择。

---

**另有两种情况，可以直接处理，不必再列成大问题。**

一是已有公式、参数、等待关系被反复解释，或明明有字段却只写文字。这属于作者和审阅没有落实现有规则：先放回正确位置，再删重复内容。

二是少量必要的接口说明，例如“写回并返回调用者提供的 out，其他输入不变”。这类事实不能丢，但一句就能说清，统一短写方式即可。

**原来的问题提报机制容易漏过这些情况。** 规范本来就允许用说明补充动作和策略。能合法写进文字，未必会触发“schema 有问题”的提报。需要增加一个检查角度：重要关系如果只能靠长段解释维持，就要检查是否缺少共同定义或合适的表达方式。

**我建议的讨论顺序：先定第 1 项的动作说明应该只留下什么，再用第 2 项确定内外等待的写法；接着处理分工和分支复用，最后决定跨调用状态的范围。** 对整篇文字直接限长，应放在这些职责分清之后。

以上来自四个模板算子、六个提取算子及全部选择分支的审计。“广泛”“明显”指观察到的范围和影响，不是已经测出的字数贡献率。详细依据保留在[全量统计](README.md)和[技术复核稿](SCHEMA_LANGUAGE_GAPS.md)，讨论时可以从本页开始。
