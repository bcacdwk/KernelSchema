**主张：继续使用 `launch + handoff`；先保持现有关系与默认粒度，用清楚的约定和 YAML alias 降低阅读、填写成本，不新增递归 phase、独立 launch ID 或通用外部运行时。**
`protocol` 是可以讨论下移到 `realization` 的字段；其余字段的能力大多有独立作用，但不等于每个实例都必须显式重复填写。
外部供给和调用者完成边界确有结构化引用缺口；轻量方案可以承认这个边界，不能宣称所有重要关系已能由当前 `flow` 机械解析。

材料以 [corpus.json](../corpus.json) 的 2026-11-12 捕获为准；仓库基线为 `bd24b58b853ed4b3deca9acebcce028a75d85b22`，规则含捕获时未提交的修改。
第 4 篇采用 [rules 快照](../rules/4.design.work_schedule.md)，SHA256 为 `d5a6beaab41dd1dee49336326f32f5778bc3b9ba1b161fdf423f8d42418d0dd6`，不以旧 YAML 的 pipeline 写法当成现行语法。
读取十个 root 的原始 flow，并用 census 定位少量反例；六提取算子采用 2026-11-09 暂停稿，FMHA/MoE 未完成整树不代表已穷尽其拓扑。
复用旧 `SCHEMA_DISCUSSION.md` 的边界线索及 `SCHEMA_VERBOSITY.md` 对 alias 的计量区分；未读本轮其他报告或综合建议。
所引 YAML 的节点版本、完整哈希、源码固定 ref、实际阅读范围与核对结果见 [05_minimality.json](../data/05_minimality.json)；未引入额外模板源码。

## 1. 两种关系不宜合成一种

`launch` 回答计算边界与 worker 是否延续，`handoff` 回答哪个值由谁交给谁、走哪类存储、何时可消费。
它们不是同一张图的重复绘制：有启动约束而没有设备数据交接，也有交接而没有新 kernel。
[Sparse probe_hidden_packed v2，L306–323](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/probe_hidden_packed.yaml:306) 中，probe 的值由 host 选路使用，remainder 仍须等待 probe 完成；这不是一个可以从设备输入自动推得的 handoff。
[MoE direct v1，L188–199](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/direct.yaml:188) 则在一个 launch 内交接 Census。
删去 launch、仅从数据边推提交，会丢失同 kernel 身份；删去 handoff、只保留先后，则读不出全局流量、局部存续或多消费者。
把两者统一成通用 event graph 理论上可以编码同样事实，却要读者从边反推 kernel 边界和数据含义，不能算更轻。

最小单 phase 外壳维持 `flow: {launch: [{phase: [MAIN]}]}` 即可，不需要新的执行区域。
已有单 kernel 模板也只有这个量级，如 [FA4 prefill root v6，L143–144](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/template_fa4_prefill/root.yaml:143)；其历史 phase 名不作为现行命名规范。
`phase` 不必与数学步骤一一对应：同一分解、映射和分配规则里的多段计算可以交给一个 phase 和其 pipeline。

## 2. 最容易损坏设计的“简单默认”

| 拟简化规则 | 删除/合并测试 | 判断 |
| --- | --- | --- |
| launch 列表默认按完成顺序执行 | Sparse 的 prefix 会被排到 probe 后，remainder 又被排到 prefix 后 | 拒绝，改变重叠机会 |
| 数据依赖自动变成整个消费者 launch 的启动等待 | GDN 的输入状态加载与局部初始化不能提前执行 | 拒绝，等待层级不同 |
| 省略 ready 一律等 producer.kernel_done | MoE direct 在同一 kernel 内等待自己完成 | 拒绝，形成自等待 |
| 省略 wait_at 自动等到第一次读取 | 需要重建实际输入、预取及事件；与现行入口等待含义不同 | 不改默认；晚等待显式填写 |
| 省略 storage 表示 global | 可定义且不必丢能力，但隐藏了最常看的物化决定 | 可选压缩，不是必要改动 |

Sparse 的真实 flow 如下，未添加新语法：

```yaml
flow:
  launch:
  - phase: [probe]
  - phase: [prefix]
  - {phase: [remainder], start_after: probe.kernel_done}
```

[固定 Sparse 源码 L6153–6190](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_sparse/dsv4_kernel.cu:6153) 明确 fork 后并行提交 probe/prefix，将 remainder 放在 probe 后，最后 join 回 caller。
所以列表省略边应继续表示“未声明额外启动等待”，既不表示串行，也不保证并发驻留。
`start_after` 的必要性来自启动约束本身，不来自是否使用 CUDA stream 或 graph；这些具体载体可以留在 realization。

[GDN root v11，L326–338](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml:326) 同时具有 `prep.launch_ready`、`ready: prep.kernel_done` 与 `wait_at: chain.prep_ready`。
[源码 L1977–1987、L2180–2224](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu:1977) 在依赖等待前初始化局部资源、读取输入 initial state；随后才消费 prep 结果。
这里需要三个坐标：启动许可、结果可用、消费者等待点。改成一个 `after` 字段会丢坐标，或把同样三项藏回长字符串。
普通串行案例不应照抄这三项：[QSA root v9，L151–155](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/template_qsa_prefill/root.yaml:151) 已用完整启动等待和默认 handoff，无需另填 ready/wait_at。

MoE direct 的 `order: per_worker` 也不能从 mapping 相同自动推得：同一 CTA 先扫描一次 routing，再处理自己获配的多个 unit。
[源码 L1946–1985](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:1946) 是 CTA 内 census 后接 unit 循环，没有全 grid 屏障。
因此不能换成 `all`，也不能把它写成 `per_worktile`，后者会要求两 phase 逐项配对；其 replica 与 unit 本来就是两种工作集合。
历史片上路径 `tile_pipeline.phase.prelude.resource.Census` 需要按现行 storage 规则迁移，但这属于语法版本问题，不削弱该实例的工作组织证据。

## 3. 字段必要性与能力必要性

**producer / consumer。** 生产与消费关系必须保留；从完整数学与 worktile 定义理论上可能恢复部分关系，不代表省掉字段更易读。
[MoE root v7，L385–395](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/root.yaml:385) 的 Counts、Offs、TilePrefix 分别连向不同消费集合；不能把“下一个 launch”当成默认消费者。
别名、数学视图和 tile_reduction 又使“找同名 target”不是可靠的通用推导法。保留具名值下的显式端点，成本比另建隐式数据流规则低。
**storage。** 同 kernel 也可以选择全局或片上交接；相同 mapping 不证明局部资源身份相同。该决定影响流量、容量与可融合性，不能删除能力。
跨 kernel 只能沿当前支持的全局路径交接时，`global` 可推得；允许其缺省是一种可选排版政策，非表达能力改进。本轮倾向保留以便直接看见物化。
**initialize。** 消费区中未被生产者写出的初值必须有归属；但本样本没有使用此字段，不能把它宣传成样本证明的新增需要。
保留现行窄义即可；[MoE root L400–403](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/root.yaml:400) 的 Counts/Fill 清零是归约与调度状态初值，不能把全部 memset 一律迁成 handoff.initialize。
数学初值、未写区域默认值、物理缓存的跨调用复用是三个问题；一个 `initialize: 0` 不能同时解决它们。

**protocol。** 必要的是提前启动所需的协议保证，字段名本身不是不可替代的能力。
现行 `launch_ready` 的意义由启动协议给出；直接删 protocol 而不交代这个事件的语义来源不成立。
若公共规则先把它定义成抽象的整 grid 后继启动许可，并要求 realization 绑定支持该关系的机制，则协议名可移至 realization，无需新字段。
[FA4 decode root v9，L454–456](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/template_fa4_decode/root.yaml:454) 已在那里绑定 PDL publication/wait；GDN root L685 也有绑定。
这是本轮最合理的可删字段候选，但收益是抽象层更清楚；不是可以删除启动许可、完成可见性或协议进展要求。
如选择保留，则只放有共同定义的协议名，不把形状分支、stream/API 调用过程塞进 protocol；API 名字和动作兑现仍归 realization。

## 4. 重复交接属性先用已有 alias

GDN root 的八个值确实重复同一整份交接属性，FA4 decode 的三个 partial 也相同；这种重复不值得先改成新的 handoff 分组语言。
下方是现行 flow 语法的内容复用，完整片段见 [05_gdn_alias.yaml](../sketches/05_gdn_alias.yaml)：

```yaml
handoff:
  Ti: &prepared {producer: prep, consumer: [chain], storage: global,
                  ready: prep.kernel_done, wait_at: chain.prep_ready}
  Qd: *prepared
  b: *prepared
  bc: *prepared
  Ai: *prepared
  Dl: *prepared
  Qc: *prepared
  Kc: *prepared
```

解析后与原 flow 完全一致；八个值仍各有身份，只少写七次属性，不建立共享存储或对象别名。
不能因为同属一条计算链就强行合组：值若有不同消费者、ready 或 wait_at，应保留差异。MoE root 的 Counts、Offs 与 TilePrefix 消费集合就不同。
新增 `value: [..]` 的交接组可减少键的重复，但会改结构、定位规则与逐值覆盖检查；先证明 alias 的阅读成本仍不可接受，再考虑这一步。
也不建议把短字段全面更名：`start_after / ready / wait_at` 已能读出三个时点，`order` 配一个展开顺序例子即可。

## 5. 新能力的门槛与真实边界

**外部端点：有真缺口，但不是必须立即增加一套边界语言。**
[MLA root v4，L321–336](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/root.yaml:321) 的 CPU 构表、上传和 main 启动关系不能写成当前 phase-only `start_after`。
[固定源码 L1895–1898、L2011–2023](../../../schema_discussion/phase_grammar/sources/kernel/extract_mla/solution.py:1895) 证实是上传完成后允许整个 kernel 启动，不能改成 kernel 内首次读表才等。
轻量的人读方案可以保留一句“Geo fully uploaded before main starts”，将表构造与工作分配理由留在 schedule；不用为 CPU 搬运伪造一个 GPU phase。
共同边界约定还可说明：声明的入口输入/元数据前提在要求的启动点满足；输出交给后续工作前，全部相应输出区域的生产执行须完成且可见。
后者是设备侧完成接续，不是 host 阻塞，也不是给各 launch 添加顺序。Sparse 的 prefix/remainder 仍可独立推进，caller 必须观察到二者的完整输出。
不能把上述约定扩成“所有外部输入先就绪，再串行跑所有 launch”，也不能把输出可消费当作所有后台访问及共享存储都可回收。
若目标要求机械校验独立外部供给、异步返回多个结果或任意边界等待位置，共同约定不够；这时才为现有 start_after/完成关系增加最少的外部引用目标。
当前保留短说明是有意选范围；不是现行结构已经表达了这些外部事件。

**独立 launch ID：目前不需要。** 现行每个 phase 恰好属于一个 launch，phase.kernel_done 已唯一定位其提交；同 launch 的多个 phase 共享同一完成事件。
[MoE double_verify_fused_small_mono，L309–319](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/double_verify_fused_small_mono.yaml:309) 已用 verify_first / verify 表达两次执行及状态接续。
[源码 L9471–9488](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:9471) 的两次 graph launch 是真实重复执行，不是需要把同一个 phase ID 复用两次的证据；相同定义可用 anchor。
单独 launch ID 仅换一个事件寻址层，还会迫使下游区分 phase 定义、提交实例和 worker 实例；样本没有因此消失的表达缺口。

**多个生产者：先分清多个 writer、多个输出 phase 与同一 handoff 多生产者。**
MoE direct 的 prelude 写 route-less Y，direct 写非空 token 的 Y；Sparse 的 prefix/remainder 也共同覆盖最终输出。这些 phase 的 target 与数学范围已经能记录多个输出生产者。
它们没有内部消费者需要一条 `handoff.Y` 同时引用两者，不能据此证明 producer 必须改成列表。
不同 phase 拼接一个中间值再由第三 phase 消费，纯分区可考虑已有数学视图分成两条交接；不能用 alias 伪造不同数值版本或同区域原子合并。
真正在同一版本、同一区域共同生产并交给第三 phase，是当前 singleton 的边界；本轮所核材料没有这样的必要实例，先列能力储备，不新增结构。

**递归组合：本样本不要求。** direct 的一次 census 后处理多个 unit，现有 per_worker 已够；同份工作的多段计算通常应先检验能否落在一个 phase。
若真有同 kernel 的 A/B 并行、随后 C 复用部分原 worker，且各段的分解不能合成一份规则，单个扁平 order 未必够；不能证明所有混合拓扑都可随意合 phase。
这是理论上合法的边界，不是已发现的样本能力缺失；本轮不为它预建递归执行树。

**跨调用状态：保留有因果作用的短前提，不发明一个调度器来修源码。**
[GDN root L379–383](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml:379) 已用三句区分同 stream 复用、本次重写、没有跨 stream 排他。
MoE double-verify L305–308 更指出下一请求的 host sentinel reset 不受设备 stream 排序；增加一个完成默认不能凭空补上这个约束。
身份、所读版本、复用前提需要保留；分配 API、缓存命中过程和不存在的同步不能升级为 flow 的执行语法。

## 字段建议与仍需取舍的范围

| 字段/结构 | 最小建议 | 不能丢的事实 |
| --- | --- | --- |
| launch[].phase | 保留，单次提交引用 phase | kernel 边界、worker 实例 |
| launch[].order | 多 phase 必填，保留现有粒度 | 逐工作、逐 worker、全体、并行的区别 |
| launch[].start_after | 保留可选；列表无隐式顺序 | 整个提交的启动条件 |
| launch[].protocol | 可讨论移至 realization；先定义 launch_ready 语义 | 启动许可所需保证 |
| handoff 的值键、producer、consumer | 保留；相同整项用 alias | 数值身份、真实端点及区域关系 |
| handoff.storage | 保留；global 缺省仅作可选压缩 | 全局物化或片上存续 |
| handoff.ready / wait_at | 保留可选；不改成 kernel 完成/首次读取默认 | 数据可用与等待位置 |
| handoff.initialize | 保留现行窄义，不扩成 memset/服务描述 | 消费区的必要初值 |
| 外部端点、launch ID、producer[]、递归组合 | 本轮不加入；外部端点列为有据可选扩展 | 不以省字段之名否认边界 |

仍需用户取舍的是：flow 是否只需让人完整审阅计算组织，还是还必须机械校验调用边界与跨次状态；前者用共同约定和短前提，后者需要另定清楚的外部范围。
其次是 protocol 名是否属于设计语言，以及是否愿意用隐式 global 换少量字符。其余建议无需先扩语言：保留关系粒度、纠正填写、应用 alias 即可。
