# 第 1、3 项的前后代码候选

供主讨论稿整合；这里只比较局部写法，不改冻结 YAML，不把候选写法当作完整等价改写或已验收的新语法。路径均相对于本文件。

## 第 1 项：同类动作的说明书反复写

**当前真实 YAML。** [GDN root:262](../inputs/template_gdn_prefill/root.yaml:262) 的 `Recurrence.info.verb.publish` 同时承载多个对象的动作契约。下面是 [267](../inputs/template_gdn_prefill/root.yaml:267) 起的完整 `publish` 值，仅缩进调整；没有截掉这个动词负责的其他对象。

```yaml
verb:
  publish: 'StateOperand reads each incoming S subtile once, snapshots Sbar and decays S in place before publication; R captures/releases KS; Ubar and
    Udec are written from the captured FP32 U before their separate publications; O rounds Y into SMEM and releases its TMEM slot. StateOperand, R,
    Ubar and Udec have independent notification streams. Producers omit empty-stage acquisition: subsequent KS/U/state completion transitively protects
    their fixed operand slots.'
```

**怎么改。** 先在 DOC5 说明：一个合成目标必须列清输入与输出集合、物理位置、单次读取的复用要求，以及 `done` 究竟保证哪些结果可用；捕获必须是真正独立于源存储的副本，源的复用仍受全部读者约束。数学公式沿用上游，位置沿用 `resource.view`，不以对象名字猜效果。下面先给**不用新增字段、也不依赖未定义短名**的局部短写。它保留了这个 `publish` 值的全部子契约，主要压缩措辞并显出双输出。

**预期写法（现有 `info.verb` 字符串）。**

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

这里 `{Sbar, Sdec}` 是字符串中的明确枚举，不是新 YAML 对象类型。`StateOperand` 沿用原名，但不再靠名称隐含“两个结果”。公式来自 [root:122](../inputs/template_gdn_prefill/root.yaml:122)，S/Sdec 同址和 Sbar 位置来自 [root:211](../inputs/template_gdn_prefill/root.yaml:211)。实例依然保留 [MMA1 的消费动作与等待:254](../inputs/template_gdn_prefill/root.yaml:254)、[Recurrence 的动作序列与前驱:276](../inputs/template_gdn_prefill/root.yaml:276) 以及 [realization 的通知实现:323](../inputs/template_gdn_prefill/root.yaml:323)：未删 KS、QS、S 等任何消费者，未将独立通知变成统一等待。

这个例子说明先规范“动作契约应该留下什么”。它只能有限缩短这个已经相当紧凑的实例；若要把共同捕获规则也压成可复用短绑定，必须先给该绑定定义输入片段、目标驻留、消费者寿命和完成边界，不能直接换成一个未定义的 `capture_profile`。

## 第 3 项：组内分工挤进说明

**当前真实 YAML。** 下列为 [GDN root:233](../inputs/template_gdn_prefill/root.yaml:233) 起的 `Inverse` 动作定义原文；完整 pipe 与边在 [244](../inputs/template_gdn_prefill/root.yaml:244)。

```yaml
Inverse:
  info:
    execution_mode: serial
    member: "warp[0:4]"
    register: 224
    verb:
      prepare: Form W for both chunks, return their gate stages, and retain W through Bmat publication.
      capture: Capture both chunks' row Beta; keep column Beta live in SMEM through Tinv publication.
      read: Capture KK/QK in registers and return its TMEM slot before subsequent weighting arithmetic.
      invert: Invert both chunks in place with group rendezvous between levels; two warps per chunk at widths 8/32/64, all four across each chunk's 16x16 diagonal blocks.
      publish: Tinv applies column Beta and returns its beta stage; Bmat weights the captured QK using retained W.
```

**怎么改。** 先用现有字段静态展开一个局部，确认分工确实能保留，再比较紧凑契约的成本。下面只展示宽度 8 这一层的替换片段。`invert8` 在两处明确绑定上游已有算法：对 A 对应 J 的全部八个 8×8 对角块求逆，FP32 Gauss–Jordan 后舍入为 BF16，写回原 A 槽。没有新的数学算法或动态 `member`。

**现有语法可以怎样展开（局部示意）。**

```yaml
Inv8First:
  info: {member: "warp[0:2]", execution_mode: serial, register: 224,
    verb: {invert8: "In-place width-8 stage of math_reshape step inverse, for all eight diagonal blocks of the named A."}}
  pipe: {repeat: t, steady: [{repeat: p, steady: ["invert8:A(n[t,2*p])"]}]}
  waits_for: {sync: {"invert8:A(n[t,2*p])@issue": ["Inverse/compute:A(n[t,2*p+1])@done"]}}
Inv8Second:
  info: {member: "warp[2:4]", execution_mode: serial, register: 224,
    verb: {invert8: "In-place width-8 stage of math_reshape step inverse, for all eight diagonal blocks of the named A."}}
  pipe: {repeat: t, steady: [{repeat: p, steady: ["invert8:A(n[t,2*p+1])"]}]}
  waits_for: {sync: {"invert8:A(n[t,2*p+1])@issue": ["Inverse/compute:A(n[t,2*p+1])@done"]}}
# 合并进原 Inverse；原有 data/reuse 边全部保留。
Inverse:
  waits_for:
    sync:
      "wait:Pair8(t,p)@issue": ["Inv8First/invert8:A(n[t,2*p])@done", "Inv8Second/invert8:A(n[t,2*p+1])@done"]
```

这不是整份可直接替换的 role 定义。配套改动必须同时做：在原 `Inverse.pipe` 的同一位置，把 `invert:Pair(t,p)` 换成 `wait:Pair8(t,p)`、`invert_tail:Pair(t,p)`；`Pair8` 只是等待这两组八层结果的局部协议对象，`invert_tail` 明确定义为原来的 16/32/64 层及其原语内部协作。原来生成两个 A 的动作、后续 Tinv/Bmat 发布、全部数据/复用边均保留。入口等待第二个 `compute:A`，利用原 `serial` 顺序保留“两个 A 都准备好”这个整组入口；出口等待两组八层动作，再进 16 层。共享物理 warp 的原 Inverse 与新增 role 只在这段交接中分时执行，不能把它们当成额外的 warp 资源。整个展开尚需验证角色交接和事件替换，不能据这个片段声称全文等价。

**更适合主讨论稿的预期短写（仍是现有契约，不是新字段）。** 只替换 `info.verb.invert`，其余动作、pipe 和 waits_for 全部保留：

```yaml
invert: >-
  Pair(t,p) contains chunks n[t,2*p] and n[t,2*p+1]; invert their A in place.
  Widths 8/32/64: warp[0:2] handles the first chunk, warp[2:4] the second.
  Width 16: each warp r handles diagonal block r of the first chunk,
  then block r of the second; no extra group barrier between those chunks.
  Rendezvous warp[0:4] before width 8 and after each level; retain the merge primitive's internal cooperation.
```

这里的 `r` 是这句契约中指代 warp 编号 0..3 的数学记号，不是可用于 `info.member` 的 pipe 变量。两两分组在 8/32/64 层成立；16 层是**每个 warp 自己先第一块、后第二块**，不能新增“全组完成第一个 chunk 才能开始第二个”的等待。准确来源为 [solution.py:518](../work/template_gdn_prefill/source/aka_solution/solution.py:518)（8、16 层）及 [524](../work/template_gdn_prefill/source/aka_solution/solution.py:524)（32、64 层）。这份短契约并没有让机器自动展开分工，只是让必要分工显式、紧凑且可审阅。

若主文还要画出**未来新简写**，应另标为“待设计语法”，不能把它混同上面的静态展开或现有 `info.verb` 字符串。特别是 `info.member: "warp[2*j:2*j+2]"` 随 repeat 的 `j` 改变，违反 [DOC5:194](../rules/schema_docs/5.design.tile_pipeline.md:194)；不能作为现有能力示例。本例暂不发明 `partition` 等字段，以免省下文字却留下未定义的展开、交接和冲突规则。

核对范围：冻结 YAML、冻结 DOC5、动作复核及上述源代码分工入口；未修改样本或规范，未运行 GPU、提取或全量等价验收。
