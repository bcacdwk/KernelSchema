# 长文本的事实审计：冗余、域外分支与表达粒度

2026-11-09。接续 [SCHEMA_VERBOSITY.md](SCHEMA_VERBOSITY.md)，从“何时变长”推进到“哪些事实应当留下”。本次新增研究记录，不改现有 YAML、规范或执行流程，不恢复提取。两个新的 xhigh 子代理分别核查动作契约和 host 文本；主代理核对规范、MoE 样例与 Sparse 容量推导。

**初步判断：这些样本混合了重复表达、未按当前选择域收窄的源码分支、审计说明和真实的执行边界。现有证据不足以把普遍长文归为语法缺失。** 应先判断事实是否属于当前设计、是否已有唯一定义，再讨论句子怎么缩短。模板用于比较表达方式，不能充当同源语义基准。

| 样例 | 本次得到的具体判断 | 处理方向 |
| --- | --- | --- |
| MoE `fc1.BProducer.load` | 三个 payload 槽、两个 scale 槽已由资源和复用边表达，契约再次复述 | 删除重复，压缩措辞，保留搬运对象及完成条件 |
| Sparse `realization.require[1].choice` | 当前自动策略且 `S>1` 时容量已足够，“另分配更大 scratch”不在本节点可达路径内 | 按选择域消去通用分支；不是补新语法 |
| MLA `main.schedule.strategy` | `Scatter` 同时决定排序和 KV 槽数；host 拷贝、等待、缓存也不是一概可删 | 保留必要规则，分清定义、结构重复与解释 |
| GDN `finish_chunk` | 无 CTA join 的逐 warp 推进、源读取退役和 final-state 别名风险是真信息；长段也夹带对写法本身的辩护 | 压缩独有语义；分别审定粒度和来源缺陷，不用删字掩盖 |
| GDN `final_store` | 记录末轮局部变量赋值后无人使用，没有增加设计约束 | 删除源码变量叙述，保留真正的预取/写回顺序 |

**1. MoE：一个不需要扩语法的局部短写。**

[原契约](../../../../.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/root.yaml#L730)共 245 个归一化字符：

> Stage PayloadA from GatherQ and PayloadB as the two contiguous up/gate weight boxes. Arm the full-stage transaction count for 32768 payload bytes. Done covers both payload transfers. Payloads use three slots independently of the two scale slots.

末句已由同文件 L639–662 的 `slot_count`、`g%3`、`g%2`，以及 L750–751 的复用距离表达。保留其余事实的一句候选为 167 字符：

> Load GatherQ into PayloadA and the two contiguous up/gate weight boxes into PayloadB; arm the full-stage transaction count for 32768 bytes; done covers both transfers.

这次缩短来自重复删除和措辞压缩，未改变动作、资源或依赖。32768 字节和完成范围刻意保留；本例无需先裁定是否还能从 shape 推导并省掉字节数。字符口径同前报告，不是 token 数；此局部候选不能外推全文件可删比例，也不构成正式语义批准。

**2. Sparse：较强的新证据是“当前路径不需要这些事实”。**

[root 的选择条件](../../../../.schema_extract/v2/runs/8f1be03a-af4a-44f2-a2ba-b8deda6ae3e7/checkpoints/a2e71c99c568d4c81e6a91a14a598e03b6962f3b592864ecda33ed32f18264fa/artifacts/root.yaml#L7)取实际 workspace 容量 `W`，固定默认自动策略，且 `P.splits > 1`。但 [choice 的 L558–562](../../../../.schema_extract/v2/runs/8f1be03a-af4a-44f2-a2ba-b8deda6ae3e7/checkpoints/a2e71c99c568d4c81e6a91a14a598e03b6962f3b592864ecda33ed32f18264fa/artifacts/root.yaml#L558)仍描述容量不足时另分配，以及 graph 的容量 eligibility 条件。

源码给出了收窄依据：

- [cta_plan L4884–4891](../../../../kernel_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/candidate/dsv4_kernel.cu#L4884)计算 `unit=T*NB*part_bytes(HB)`、`fbytes=fuse_bytes(T,NB,HB)`，用 `max(1, floor(max(W-fbytes,0)/unit))` 限制 split 数。
- `S>1` 排除了 `unit=0` 和 cap 因空间不足而被强抬到 1 的情况，因此 `S*unit+fbytes <= W`。
- [make_plan L5125–5142](../../../../kernel_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/candidate/dsv4_kernel.cu#L5125)的 DQ 调整只改配置；最终 `P.ws_bytes=S*unit+fbytes`。此节点的容量回退条件不成立。

这不是说 fallback 在整个算子中不存在，而是本节点已经有更强的不变量。可删掉此节点的 fallback 叙述，并在需要解释容量时仅保留 `P.ws_bytes <= W` 及其设计作用；推导和源码证明属于研究/审计依据，无须全部搬入 YAML。

同段其他事实不能连带删除。[graph_capture L6384–6398](../../../../kernel_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/candidate/dsv4_kernel.cu#L6384)表明首次 capture 前确实同步 caller stream，BeginCapture 失败则在 dispatch 前返回，实例化本身也不执行所录 kernel。首次捕获的额外完成边界、后续 replay 使用当前输入及 workspace 绑定仍可能影响正确性或执行代价；不需要把每个失败 API 的控制流逐句重述。

另有一个需收窄的措辞：[solution.py L160–170](../../../../kernel_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/candidate/solution.py#L160)只对合格 workspace 保留参数身份；不合格输入复用 private scratch，但把 `_ws_obj` 设回 `_UNSET`。因此笼统写“caches workspace selection by argument identity”容易让实现细节说明本身产生歧义。

主代理核实上述两个工作树源码与 root 的来源 commit `6e4ca91bd91d998cac99e8dadd83838e153c679d` 字节一致。`solution.py` SHA256 为 `2411a8fd28e09be34b14fa103ee616d8b6c3e2f2177d7f22b1089b9e0e491525`；`dsv4_kernel.cu` 为 `c9389387d35bd26954754a5ae13e2d202df06fc9799a77fddfb0654306909eb2`。

**3. MLA：host 细节必须按其后果区分。**

[main.schedule.strategy L321](../../../../.schema_extract/v2/runs/531567f1-bcae-45ad-acf6-fd674c055fdc/checkpoints/05f6f97e9f84c73f3af897059f8a331e08626e326ba026f8d9b2bd0ee6d0558c/artifacts/root.yaml#L321)混合了 metadata 拷贝、host 等待、排序、缓存参数、Geo 上传、录图与可选输出。不能因它们来自 host 就整体移出：`Scatter` 不只影响 stable sort，还通过 [Loader.load L599](../../../../.schema_extract/v2/runs/531567f1-bcae-45ad-acf6-fd674c055fdc/checkpoints/05f6f97e9f84c73f3af897059f8a331e08626e326ba026f8d9b2bd0ee6d0558c/artifacts/root.yaml#L599)令 `KVSlots=3/4`，作用到 L379、385 的真实资源容量。

因此应保留 Scatter 的定义、排序生效条件和必要的 host/device 顺序；把已经由输出条件、工作集合和 flow 表达的重复单独核查。移到另一个长 `choice` 中不能算压缩，也不能用“只是 locality heuristic”作为删掉它的理由。

实际槽数选择见 [solution.py L2019–2024](../../../../kernel_wiki/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/candidate/solution.py#L2019)。子代理核实该源码与 `.schema_extract/v2/snapshots/fa6531820137d090dc39755c2756ad1072c8f3b21aa722d3b7725e1b225c2ed4/source/` 中同路径文件字节一致，对应 root 固定来源 `228934a2253bde4d6f0c4463dd77aeada1bf07a8`，SHA256 为 `78cb2427001cd23d296e907f40f28c9743ba46471eb17ee80a38d584bf45dff8`。

**4. GDN：要区分局部契约、跨动作顺序与表达成本。**

[finish_chunk L638–660](../../../../.schema_extract/v2/runs/45f9bb51-77f3-4285-b7a7-251cf2744fa1/checkpoints/349f0fdbad54effd79e2557888dffcc88de418c142438fd2989db2b4937d0a2d/artifacts/root.yaml#L638)把 full/mini 输出、状态更新、warp 内同步、非末块接续、final-state staging 和源码风险集中在一个动作中。其主要边界是多个 warp 各自按序推进，彼此没有所述中间 CTA join。

[动作粒度规则 L22、98](../../../../schema_docs/5.design.tile_pipeline.md#L22)允许保留内部计算和线程协作。把这段机械拆成多个覆盖全部八个 warp 的 `serial` 动作，会引入各动作间的 `done→issue` 保证，未必保留原来的独立推进。显式建八个真实 warp role 并表达实际汇合也并非语法禁止；但重复的 role 和汇合关联可能使全稿更长。这支持“表达成本/粒度值得研究”，尚不能证明“没有合法表达”。

另有独立的语义审查点：同文件 `setup` L599–600 说只有显式 CTA/warp joins 同步成员，但 [serial 定义 L106](../../../../schema_docs/5.design.tile_pipeline.md#L106)仍规定跨动作默认顺序。如果该说明意图连动作之间也只保留逐 warp 顺序，就不能靠文字取消默认边。动作内部无 join 的合法性，不等于跨动作的弱顺序也已表达正确。

来源缺陷与设计事实也须分开。末次 mini chunk 的 stage-0 覆盖风险应保留有据限制；“Keeping this suffix in one action does not order those reads…”则包含对当前写法的说明。按 [REVIEW.md 的分流条件](../../../REVIEW.md#源实现缺陷与设计分流)，应先确定完整设计关系是否已有依据，才能把缺陷详证移出、在 YAML 留必要的一句限制。本轮没有正式改判或修订这份历史稿。

风险有具体静态反例：`VTT=64`、请求 FinalState、末次 mini chunk 使用槽 0 时，warp 0 写 `stg[0*65+8]` 会覆盖字节 32，另一个 warp 1 的 `Kc[0,16]` 读取也访问该处。见 [gdn_kernel.cu L2508–2511](../../../../kernel_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/candidate/gdn_kernel.cu#L2508)、[L2833–2839](../../../../kernel_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/candidate/gdn_kernel.cu#L2833)：CTA join 发生在覆写后，不能保护这次旧值读取。这是顺序缺失的反例，不是 GPU 实测，也不表示每次执行都发生竞争。该源码及相邻 host/header 与 root 声明的 `6e4ca91bd91d998cac99e8dadd83838e153c679d` 字节一致；CUDA 文件 SHA256 为 `1817841153fb05dae9acf9f9a9700d00aea67d9de2d40a4c00c90e924f994191`。

同时，数学区 L274–297 已定义 full/mini 的计算、精度和输出目的地；`solve` L637 已定义预载和读退役；`advance` L620–628 已定义下一轮复用边界。利用这些既有定义，原 1300 字符可先形成以下 668 字符的纯文本候选，保持原有动作和结构：

> Each warp executes the full-path readout into RhsOutput, updates R and rebuilds Rh, then warp-syncs and sweeps its own valid Out bytes with coalesced 16-byte stores to the declared destinations. On mini chunks, rhs has stored Out; use its published DU0 and load Kc row 0 for the rank-1 R update and Rh rebuild. Warps proceed independently until the next advance or the final-state publication join. On the last chunk, requested FinalState is first staged in slot 0 at FP32 pitch VTT+1, then CTA-published. Source limitation: for a final mini chunk in slot 0, that overwrite can precede other warps' Kc row-0 read retirement. Ordinary output stores need not drain here.

候选保留了 warp 自有输出范围、16 字节访存、pitch 和精确风险条件，不要求新增语法；它仍依赖上述定义及 `final_store` 继续存在。它未经独立等价审阅，也不解决前述 serial 跨动作抽象问题，不能将这次长度变化当成已证明的无损压缩。

相邻 [final_store L665–666](../../../../.schema_extract/v2/runs/45f9bb51-77f3-4285-b7a7-251cf2744fa1/checkpoints/349f0fdbad54effd79e2557888dffcc88de418c142438fd2989db2b4937d0a2d/artifacts/root.yaml#L665)的“subsequently assigned current value is unused”则更接近可删的代码阅读笔记：它对应 [源码 L2860–2863](../../../../kernel_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/candidate/gdn_kernel.cu#L2860)末轮后无人消费的 `cur=nxt`。这里应保留“末轮没有下一次 load”及此前 load/store 的软件流水顺序，无需交接局部变量最后一次赋值。

**对下一步研究的约束。**

现行规范其实已经要求 [契约只补名称和上游推不出的事实](../../../../schema_docs/5.design.tile_pipeline.md#L96)、[strategy 不复述结构](../../../../schema_docs/4.design.work_schedule.md#L380)、[realization 先做删除测试](../../../../schema_docs/6.realization.md#L21)。单纯再加一句“简洁”尚无新机制；检查器只验非空也不代表规范认可任意长文。

下一组同源短稿实验应逐条回答：这个事实在当前选择域内是否可达；是否改变数值、工作、顺序、资源或主要性能代价；是否已在其他字段有唯一定义；若必须保留，现有结构能否表达；剩下的自然语言最少需要说明什么。只有最后仍出现具体表达障碍，才据最小反例讨论语法扩展。

对照应检查首末/空分支、全部读者退役、存储别名和跨调用状态，不能只比较长度或 checker 退出码。本次完成的是有源码依据的局部审计与短写候选，没有完成全部长字段的逐句覆盖，也没有进行同源短稿盲审，因而不估计全量可删比例或提示改动的因果效果。
