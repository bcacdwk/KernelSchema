# 片上存储声明审计

**建议保留存储、共享索引、跨 phase 事件组这三个概念，不增加存储字段。** 在本次六 root 和两个有针对性的分支中，`depth/location/shape/dtype/member + view` 能表达需要保留的片上容量、分槽、区域共用和数值副本。主要可改的是名称、派生字节数的填写方式，以及“设计区域”和“实际分配请求”的边界；没有找到必须新增 `alias`、`lifetime`、`allocator` 或缓存字段的最小反例。

当前规则明确不登记 barrier 存储、对齐余量和编译器临时量，也明确允许同 worker 跨 phase 共用一份资源账（规则 5，L338–358、L492–500）。因此不能把旧稿仍写着 `space/member_slot_shape/slot_count` 当作表达能力问题，更不能把旧审阅批准当作本轮新规则认证。

实际材料范围如下。六 root 重点阅读 Design、相关数学值与 realization；分支只为验证特定边界。源码只审阅有关的分配、读写和交接段落，没有重新提取六算子或运行 GPU。

| 算子 | 根稿中的主要存储事实 | 核对的关键源码 |
| --- | --- | --- |
| [GDN root L446–586][gdn] | 头间复用、混合类型 stage、RhsOutput 共用、末尾 stage 0 改存 FinalState | `gdn_kernel.cu` L1902–1935、L2499–2511、L2819–2839 |
| [GEMM root L272–329][gemm] | 三槽输入、两槽产品、两槽寄存器读出、单份输出 staging | `fp8gg_umma.cu` L315–345、L592–665、L744–763 |
| [MLA root L368–464][mla] | 三槽 S/P 别名、不同 SMEM 环、寄存器副本、两槽 Geometry | 内嵌 CUDA `solution.py` L464–470、L817–844、L912–945、L1218–1237 |
| [Sparse root L384–421][sparse] | Q/Gather/FP32 partial 同区复用；Q 寄存器复制；全局 partial 跨 kernel | `dsv4_kernel.cu` L878–896、L979–990、L1162–1173、L1424–1465 |
| [FMHA root L465–525][fmha] | TMEM 分区、两槽 K/V、Q 提前预取、保留完整 score 的寄存器容量 | `fmha_tcg.cuh` L1822–1852、L3278–3320；`fmha_mxfp8_qkv.cu` L818–833 |
| [MoE root L638–682、L850–871][moe] | packed payload/scale 不同环深、C1/C2 与 scale 的分区、跨 kernel 同名定义 | `source_nvfp4_moe.cu` L4349–4367、L4908–4921、L5050–5064；同 worker 反例另读 direct 分支 |

分支深入范围为 **MoE/direct L1–421** 和 **FMHA/tcg_d128_flat_tma_xpose_narrow 的 split 部分 L492–657**，两者均为旧审阅 `approved`。另有跨文件关键词定位，以及 GDN/segmented、MoE/fused_small、Sparse/tc_heads、tc_packed 的资源扫描，未据此声称完整语义审阅。corpus 的 172 份有设计稿、5 份未创作 scaffold、43 个未开始节点是材料统计，不是本报告审阅数量。所审实例均为单 CTA worker，`member` 的部分成员情形没有被这些样本充分检验，暂无删除依据。

已读 [SCHEMA_VERBOSITY][verbosity] 的核心发现与文字/结构/来源限制部分，以及 [SCHEMA_DISCUSSION][discussion] 的动作、外部边界、跨调用状态等讨论；它们用于定位，下面的判断回到固定 YAML 和源码。完整路径、原文、行号、源码 SHA256 在 [01_storage.json](../data/01_storage.json)，每条对应同名 ST 编号。

**ST01 — view 三项应保留，单值空 view 可按现有规则减掉。** MLA 的 `P(n)` 占每行第 `[32,64)` 个 FP32 单元对应的字节，以 FP8 解释；这是每行 128 个 FP8 元素，不是“整块某段同样大小的连续字节”。槽数不决定 `n%3`，等待边也不决定放在哪个槽。GDN stage 里 BF16、FP32 共用 uint8 区域，再次说明 `dtype` 不能随 `part` 删除。类别是 `existing_structure`：不需要扩展，适合把 `slot_index` 缩为 `slot`，不适合发明隐含取模。单槽、整槽且名称就是上游值的 `M`、`D` 可以免 view；S/P 的 TMEM 与寄存器副本仍须各自绑定。

证据：[MLA L387–424][mla-view]、[源码 L464–470][mla-map]；[GDN L571–586][gdn-stage]、[源码 L1902–1907][gdn-size]。

**ST02 — 不宜把所有共用都强制化成同形状的环形槽。** Sparse 的 ring 开头给 Q、稳态给多槽 Gather、末尾给 FP32 partial。源码有 `HB*D*4 <= STAGES*KC*D*2` 的整区容量检查；后者可以跨原来的 KV 槽。根稿用一块 `depth=1` 的字节 arena 和不同 `part` 已表达清楚。这里的 1 是所选分配单位的槽数，不是“流水线只有一层”。这是 `existing_structure`；保留一个 arena 比新增跨槽 alias 字段简单。Q 读完后的首次覆盖、所有 Gather 读完后的末尾覆盖仍必须有保证，view 不代替同步。

证据：[Sparse L385–393、L444–498][sparse-ring]；[源码 L878–884][sparse-fit]、[L1162–1173][sparse-query]、[L1424–1465][sparse-stage]。

**ST03 — 设计副本与编译器临时量要区分。** GEMM `DrainRegister` 的双缓冲允许下一读与当前 FMA 重叠，最后一读完成后先归还 TMEM、再做最后一组 FMA。FMHA `QueryPrefetch` 让 Q 读取先于 TMEM 分配；Sparse `Q_cached` 的 `copy: WC1` 表示实际复制量，并跨 recurrence 保留。删去这些会丢掉容量、寿命或重叠事实，适合保留资源加短契约，类别为 `concise_semantics`。普通转换打包标量、指令寄存器名、barrier 字节则不需要逐项登记；也不能把逻辑寄存器值字节再加到已经包含它们的 `role.register` 预算上。未发现必须把现有聚合 scratch 拆成每个内部数组的理由。

证据：[GEMM L315–322、L426–446、L481–489][gemm-drain]、[源码 L592–665][gemm-register]；[FMHA L548–555][fmha-prefetch]、[源码 L3278–3320][fmha-prefetch-source]；[Sparse L395、L453–455][sparse-cache]、[源码 L1162–1173][sparse-query]。

**ST04 — 同 worker 跨 phase 与跨 kernel 的边界已经可表达。** MoE/direct 的一次 `order: per_worker` 提交包含 prelude、direct。Census 和 Exchange 只在 prelude 登记；direct 读取 Census，并把已经消费完的 `Exchange.WarpCounts` 改放 `FinalRoutes`。源码是同一个 `s_warp[16]` 的两次用途。现有 launch、handoff 和一处 view 足够；默认交接无须额外 milestone。root 的 fc1、fc2 则属于不同 kernel，即使 YAML alias 复用了 PayloadA 定义，资源实例仍独立。类别是 `existing_structure`，不建议增加 `owner_phase` 或 `scope` 字段。

证据：[MoE/direct L188–199、L291–306、L372–375][direct]；[源码 L1929–1976][direct-census]、[L2218–2237][direct-reuse]；[MoE root L375–395、L850–870][moe-flow]。

**ST05 — total_size 可以减轻作者负担，但不能改成实际占用承诺。** 按规则它是 `depth × 单槽字节 × worker 持有成员数`，不是独立选择。建议变为工具派生展示，或允许省略、填写时校验；保留时可短称 `bytes`。这是 `existing_structure` 的减冗余建议，不是旧稿漏字段诊断。六 root 使用旧格式，不能从它们推算“手填 total_size 的出错率”。

FMHA 提供了实际限制：根稿登记 SMEM 为 `16384 + 2×16384 + 2×16384 + 2×512 + 2×512 = 83968` 字节；源码初始动态请求为 **102912** 字节，多出的 18944 来自未用 P/Q-scale 预留、exchange 预留和对齐余量，之后 helper 还可能以 4096 字节递增。这个差别真实影响容量，但适合保留在 realization，不能往资源表里造几个假数学值。也不能把 SMEM、TMEM、寄存器字节混加，或把多 CTA 合计直接当单 SM 驻留条件。

证据：[FMHA L466–494、L685–702][fmha-memory]；[布局源码 L1822–1852][fmha-layout]、[请求调整 L818–833][fmha-request]；[规则 5 L338–358][resource-rule]。

**ST06 — 值得补清的是“分配账”的单位。** MLA 只调用一次 512 列 TMEM 分配，但根稿把互不重叠的 S/P 区与 Y 区分列。现有语法也能把它们合为一个大区域再写偏移，能力并不缺；强制追随分配 API 会让原本清楚的三槽结构更啰嗦。建议把 `一块分配只登记一次` 补清为：**设计上同一共用区域只计一次；互不重叠的子区可以分列；分配 API 如何打包它们属于实现。** 类别为 `underspecified_rule`，不是 `missing_capability`。

padding 也按这个边界处理。MoE/direct 写出 `H+H/8` 的 InputCache，MLA TinyY 却保留逻辑 `32×ValueDim` 而不抄入源码 130 的行距。这些旧稿写法不能自动证明新字段缺失。普通布局余量可交实现，保留缓存值量、寿命和别名；若余量显著影响驻留，像 ST05 那样保留实际请求的简短实现约束。**也不能删除所有未存值的容量：MoE root fc1 实用 192 列，却按有效分配粒度申请 256 列 TMEM；资源容量仍应保留 256，不能虚报省下 64 列。** 不用为这 64 列另造 view。未知的编译器复用不能当成已成立的别名，所有短暂逻辑值的和也不是物理寄存器峰值。

证据：[MLA L387–400][mla-view]、[源码 L817–844][mla-alloc]；[MoE/direct L324–352][direct-padding]、[源码 L1924–1928][direct-padding-source]；[MLA L443–448][mla-tiny]、[源码 L1080–1084][mla-tiny-source]；[MoE root L669–676][moe-tmem]、[源码 L4349–4367][moe-tmem-source]。

**ST07 — 跨调用 cache 需求不归 docs5 兜底。** GDN grow-only workspace 的身份与同 stream 复用、MoE/direct 的 RouteAccum/Arrivals 自清零、MLA queue word 的跨调用存续都是真实语义。它们分别属于全局物化、初值、调度或调用约定，类别为 `other_layer`。保留“用哪次值、下一次复用前什么要完成”的短说明；不建议让片上 storage 新增 process、cache key、allocator 字段。缺少跨 stream 同步也不能靠本地事件组自动补出。

证据：[GDN L330–338、L379–383][gdn-global]、[源码 L3254–3274][gdn-cache-source]；[MoE/direct L126–145][direct-global]、[源码 L2239–2256][direct-clear]；[MLA L342–352、L626–633][mla-queue]。

**ST08 — 有区域声明仍可能有源同步缺陷。** GDN 最后一个 mini-chunk 在 stage 0 的 Kc 行读尚未全部结束时，可能写入 FinalState；FMHA 空 split 在 Query 写者完成前可能覆写 OperandArena。原稿里的长说明主要承载来源保证问题，属于 `source_or_draft_issue`。FMHA 分支已用 prepare→stage_partials 边记录应有关系；新增 view/allocator/milestone 不会修复源码。GDN 如果另做安全设计，需要先给必要读者完成找准动作边界，本报告不擅自修源或认证。

证据：[GDN L637–659][gdn-gap]、[源码 L2499–2511、L2543–2552][gdn-mini]、[L2819–2839][gdn-final]；[FMHA 分支 L503–512、L604–649][fmha-gap]、[源码 L5388–5424][fmha-gap-source]。

三个名称的独立建议如下，不要求形式整齐。

| 当前名 | 首选 | 其他候选的取舍 |
| --- | --- | --- |
| `resource` | **`storage`** | `buffer` 易懂但容易只想到搬运缓冲；`allocation` 更像分配 API/动态生命周期，且更长；`resource` 又容易把 warp、引擎、barrier 和许可全装进来。storage 同样涵盖 SMEM、TMEM、register 的值与状态。 |
| `domain` | **`index`** | `range` 适合区间，对 worktile 与 flatten 偏窄；`domain` 数学上正确，但易与算子适用域混淆。index 表只登记共享索引，不生成循环或槽。 |
| `milestone` | **`event`** | 必须首句定义为“全部已有事件发生时成立的对外事件名”，防止误读为新建 CUDA event。更看重复合性可用 `event_group`；`milestone` 也可保留。`ready` 不合适，wait_at 也可引用读取动作的 issue。 |

静态 `storage` 不登记使用权协议；`acquire/release` 和 waiting 约束对这块存储或区域的使用。FMHA prepare 中 **归还 TMEM allocation permit 后仍继续使用 TMEM**，finish 才 deallocate，正说明许可归还、读者归还与物理销毁不能混为一件事（root L548–555、L636–643；源码 L3316–3320）。因此 `storage` 比 `allocation` 更适合作为该表名称。

以下是**建议草图，不是现行格式**。三段都示范 `resource→storage`、`total_size` 改为派生展示；第一段还示范 `domain→index`、`slot_index→slot`。仅展示相关声明，省略部分的动作、条件、来源绑定和依赖仍须保留。

MLA：保留真正的别名，减掉可由同名推定的单值空 view。

```yaml
index:
  w: worktile
  t: [0, "N(w)"]
  n: {flatten: [w, t]}
storage:
  ScoreProbability:
    depth: 3
    location: TMEM
    shape: [row: Bq, column: Bk]
    dtype: fp32
    view:
      "S(n)": {slot: "n%3"}
      "P(n)": {slot: "n%3", part: {column: [32, 64]}, dtype: fp8_e4m3fn}
  M: {depth: 1, location: register, shape: [query_token: Bq], dtype: fp32}
  D: {depth: 1, location: register, shape: [query_token: Bq], dtype: fp32}
```

Sparse：保留整块 arena，不为三个不同生命周期重复记账。这段依旧需要 `part`，不能仅用一句“reuse ring”代替。

```yaml
storage:
  ring:
    depth: 1
    location: SMEM
    shape: [byte: "2*Stages*KC*D"]
    dtype: uint8
    view:
      "Gather(c)": {part: {byte: ["2*((c-c0)%Stages)*KC*D", "2*(((c-c0)%Stages)+1)*KC*D"]}, dtype: bf16}
      Q_ring: {part: {byte: [0, "2*HB*D"]}, dtype: bf16}
      PartialStage: {part: {byte: [0, "4*HB*D"]}, dtype: fp32}
```

MoE/direct：同一 worker 的资源只登记一次，后续 phase 直接绑定 `Exchange.FinalRoutes`；Census 的默认交接不用额外造事件组。

```yaml
work_schedule:
  flow:
    launch: [{phase: [prelude, direct], order: per_worker}]
    handoff:
      Census: {producer: prelude, consumer: [direct], storage: tile_pipeline.phase.prelude.storage.Census}
tile_pipeline:
  phase:
    prelude:
      storage:
        Census: {depth: 1, location: SMEM, shape: [], dtype: int32}
        Exchange:
          depth: 1
          location: SMEM
          shape: [element: 16]
          dtype: int32
          view:
            WarpCounts: {}
            FinalRoutes: {part: {element: [0, K]}}
```

证据复核脚本为 [01_storage.py](../scripts/01_storage.py)：检查引用片段和固定输入哈希，复算 ST05 的两种字节数。它不做新语法验证，也不证明源码安全。

[gdn]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml#L446
[gemm]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gemm/root.yaml#L272
[mla]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/root.yaml#L368
[sparse]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/root.yaml#L384
[fmha]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/root.yaml#L465
[moe]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/root.yaml#L638
[verbosity]: ../../../schema_extract/archive/six-operators/20261109-methodology-review/SCHEMA_VERBOSITY.md#L174
[discussion]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/SCHEMA_DISCUSSION.md
[mla-view]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/root.yaml#L387
[mla-map]: ../sources/kernel/extract_mla/solution.py#L464
[gdn-stage]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml#L571
[gdn-size]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu#L1902
[sparse-ring]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/root.yaml#L385
[sparse-fit]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_sparse/dsv4_kernel.cu#L878
[sparse-query]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_sparse/dsv4_kernel.cu#L1162
[sparse-stage]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_sparse/dsv4_kernel.cu#L1424
[gemm-drain]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gemm/root.yaml#L315
[gemm-register]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gemm/fp8gg_umma.cu#L592
[fmha-prefetch]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/root.yaml#L548
[fmha-prefetch-source]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_fmha/source/candidate/fmha_tcg.cuh#L3278
[sparse-cache]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/root.yaml#L395
[direct]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/direct.yaml#L188
[direct-census]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu#L1929
[direct-reuse]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu#L2218
[moe-flow]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/root.yaml#L375
[fmha-memory]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/root.yaml#L685
[fmha-layout]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_fmha/source/candidate/fmha_tcg.cuh#L1822
[fmha-request]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_fmha/source/candidate/fmha_mxfp8_qkv.cu#L818
[resource-rule]: ../sources/rules/5.design.tile_pipeline.md#L338
[mla-alloc]: ../sources/kernel/extract_mla/solution.py#L817
[direct-padding]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/direct.yaml#L324
[direct-padding-source]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu#L1924
[mla-tiny]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/root.yaml#L443
[mla-tiny-source]: ../sources/kernel/extract_mla/solution.py#L1080
[moe-tmem]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/root.yaml#L669
[moe-tmem-source]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu#L4349
[gdn-global]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml#L330
[gdn-cache-source]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu#L3254
[direct-global]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/direct.yaml#L126
[direct-clear]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu#L2239
[mla-queue]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/root.yaml#L342
[gdn-gap]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml#L637
[gdn-mini]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu#L2499
[gdn-final]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu#L2819
[fmha-gap]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/tcg_d128_flat_tma_xpose_narrow.yaml#L604
[fmha-gap-source]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_fmha/source/candidate/fmha_tcg.cuh#L5388
