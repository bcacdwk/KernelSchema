# Flow 的外部前置、调用汇合与跨次状态

**短结论：先统一边界语义，不急于增加外部执行图。** 对十个算子已描述的成功单次计算，现有内部 flow 加上共同 caller 入口/出口约定，以及 MLA、MoE 各一句明确的供给/初始化前提，可以保留本轮关注的设计事实。**只有 caller 入口/出口一句话还不够**；它不能自动表达 Geo 上传、Counts/Fill 清零，更不能证明共享 scratch 的跨次安全。若目标包含完整 host 调用、失败恢复或任意跨请求组合，这套最轻方案不够。

| 处置 | 本轮建议 |
| --- | --- |
| 保留 | 内部启动依赖、启动许可与可消费时点的区别、Sparse 两支之间没有完成屏障、必要初值与逻辑版本存续。 |
| 澄清 | 共同 caller 边界仅安排设备执行；外部供给按实际使用它的提交就绪；CPU 返回、host 观察与 GPU 完成互不等同。 |
| 新增 | 优先新增上述共同语义说明；当前十算子未证明必须新增固定字段。若要求机械关联外部前置，最小增量是让 `start_after` 引用已有值的外部 ready，而非新增 host phase。 |
| 不纳入 | 完整 allocator/cache/stream/capture 语言、跨请求锁和 generation 协议、源码未提供的同步；这些不因文字较长就应成为 flow。 |

范围依据 [METHOD](../METHOD.md:1)、[corpus](../corpus.json:1)。四模板 root 与六提取 root 为主；六提取 root 均是已批准工作稿，但 FMHA、MoE 整树仍暂停。补查 Sparse hidden、MLA bootstrap、MoE 初始化/verify/direct 等已批准分支；未把未开始节点或重复继承计成证据。统计及精确来源集中在 [03_boundaries.json](../data/03_boundaries.json:1)。静态核对，无 GPU 正确性或性能结论。

## 1. 现行能力与真正缺少的端点

[规则 4](../rules/4.design.work_schedule.md:247) 将每个 phase 绑定于一次计算执行；[launch](../rules/4.design.work_schedule.md:255) 只登记计算 kernel，[事件](../rules/4.design.work_schedule.md:367) 只能是 phase 的内建完成/许可或本地动作别名。因此，caller 前序、caller 后继、host 供给完成都没有原生端点。

但 [说明字符串](../rules/4.design.work_schedule.md:387) 可以承载结构表达不了的那条规则。当前是“关系缺少统一、可引用的结构载体”，不是“自然语言完全不能表达”。反过来，不能把 caller、memcpy、host sort 假造为 GPU phase，也不能用 [realization](../rules/6.realization.md:11) 首次决定上游漏写的依赖。

已有 `handoff.ready/wait_at` 解决内部值的就绪和消费位置，不能代替外部就绪；`binding` 给出定义来源，不产生执行边；`retention` 给出逻辑保留，不等于跨请求的同一物理 allocation。相关规则见 [4:285](../rules/4.design.work_schedule.md:285)、[3:104](../rules/3.design.math_reshape.md:104)、[3:153](../rules/3.design.math_reshape.md:153)。

## 2. Sparse hidden：共同调用边界足以表达本例

[YAML](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/probe_hidden_packed.yaml:299) 的结构只写 `probe → remainder`；fork/join 在 thesis 和 describe。[源代码](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_sparse/dsv4_kernel.cu:6153) 则明确先记录 caller fork，再让 side 等待；prefix 在 caller 提交，remainder 在 side 提交，最后把 side join 排入 caller。

必须保留的关系如下；这是关系草图，不是新 YAML：

```text
caller_prior_done ─┬─> probe ─────> remainder ─┐
                   └─> prefix ────────────────┴─> caller_next
prefix 与 remainder 之间没有完成依赖。
CPU 可以在两路输出完成之前返回。
```

“不加边”表示保留重叠机会，不能推导同时驻留。源码 [6180–6190](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_sparse/dsv4_kernel.cu:6180) 特意将 remainder 放到 side；加上 `prefix.kernel_done → remainder` 会让它等待最慢 prefix CTA，改变设计。

最轻方案是把 caller 入口/出口定义为共同请求约定，实例继续使用现有合法片段：

```yaml
flow:
  launch:
    - phase: [probe]
    - phase: [prefix]
    - phase: [remainder]
      start_after: probe.kernel_done
```

共同约定需要明说：入口提交接在 caller 已排入的前序设备工作之后；caller 后续设备工作等待本请求相关输出完成；不新增内部提交之间的边，也不要求 host 阻塞等输出。由此得到 prefix/remainder 汇合，既不用第四个 kernel，也不用虚构输出 merge 或 `ProbeRatio` handoff。ProbeRatio 实际只供 host 选路径，见 [YAML:315](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/probe_hidden_packed.yaml:315)。

**真实删除反例：** 只保留三条 launch、删去所有共同或实例边界语义，side 可以早于 caller 的输入生产者读取，caller 后继也可以早于 remainder 消费输出。**没有发现本例必须逐项自定义 caller 端点的反例。**

## 3. MLA：还需一句“当前 Geo 在 main 启动前供给”

[YAML:322–336](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/root.yaml:322) 混合了回传、排序、设备属性缓存、上传和录图。核源得到三件不同的事：

- [1669–1681](../../phase_grammar/sources/kernel/extract_mla/solution.py:1669)：metadata 在非阻塞 side 上 D2H，host 等这批复制后构表。
- [1895–1898](../../phase_grammar/sources/kernel/extract_mla/solution.py:1895)：Geo 的 H2D 后记录事件；[2010–2024](../../phase_grammar/sources/kernel/extract_mla/solution.py:2010)：current stream 等该事件，然后启动整个 main grid。
- [1654–1681](../../phase_grammar/sources/kernel/extract_mla/solution.py:1654) **没有** caller→side 的 fork；Python [forward](../../phase_grammar/sources/kernel/extract_mla/solution.py:2134) 也未添加这条同步。不能把 Sparse 的 fork 当作所有供给操作的隐含行为。

最少保留的实例前提可放在现有 `schedule.strategy`：

```yaml
strategy: >-
  Build and upload Geo from current metadata on each host invocation; MAIN starts
  after the upload event. Side-stream preparation does not wait for preceding
  caller work.
```

这是**现行字段的短写草图**，单 phase 名用现行 `MAIN`；Geo 定义/排序规则仍须在各自位置保留。它不声称 metadata 在源码中已同步，也不声称 replay 重新执行 host 构表。录制绑定与执行时读取内容要保留原 [YAML:334](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/root.yaml:334) 的区分，不能把所有 replay 都写成“当次重建”。

**最小反例：** 等上传后再启动 main，与先启动 main、读 Geo 前才等上传，可以读取相同当前值，却有不同的 worker 占用和重叠窗口。Geo 数学定义、普通读取正确性及 caller 边界均不能区分；“在该提交启动前供给”可以，无须拆出 D2H/sort/H2D 三个 service。

**不能采用的共同约定：** “本调用的所有输入访问都排在 caller 前序之后”。它会给 MLA 的 side D2H 补出源码没有的等待，也会消除源码允许它与 caller 前序工作的重叠。若 caller 正在改写这些 metadata，必须单列输入就绪前提/来源限制，不能用默认值替源码证明安全。

## 4. 两种最轻方案的取舍

| 方案 | 实例成本 | 能保留什么 | 放弃什么 |
| --- | --- | --- | --- |
| 共同 caller 边界 + 每例短前提 | Sparse 不加字段；MLA 留一句 Geo 供给，MoE 留一句清零前提 | 当前成功单次计算的 fork/join、按提交的供给、既有内部 PDL/分支 | 外部事件不可机械寻址；不描述完整 host 排队时序、任意异步入口或分片输出协议 |
| 最小外部 ready 引用 | 对特殊提交多一个 `start_after` 引用，并明确值的供给身份 | 机械区分启动前就绪和启动后读前等待；可表达供给只约束部分提交 | 仍不能自动解决物理缓存归属、跨次复用、host 阻塞或失败恢复 |

第二种只需扩展引用语法，示意如下；**这是讨论提案，当前非法**：

```yaml
flow:
  launch:
    - phase: [MAIN]
      start_after: external.Geo.ready
```

`external.Geo.ready` 必须绑定已定义的 Geo 当前版本，表示供给完成且对该 grid 可见；不能是随意的未知事件名。是否需要额外声明表，应先看上游能否明确供给身份。共同出口已够 Sparse，不必为它同时增 `boundary`、stream ID、host-block 标志、event-record 动作。只有目标允许非默认输出汇合或局部输入延迟，才继续设计那些能力。

第一种方案的“启动前供给”必须**按使用它的提交**解释；不能让一个分支等待另一个分支专属的前置，也不能把内部 handoff 重新归入“全输入先齐备”。若把普通 caller 入口固定为完成边界，也主动放弃跨请求 PDL 提前启动；本次未发现要求这种外部协议的有效实例，内部 PDL 仍由已有 flow 保留。

这是一项需要明确选择的接口范围，**不是从源码自动推出的普遍保证**。样本没有定义任意 caller 的外部 PDL 端点，也未证明所有包装器都在任意前驱环境下完成后才启动。若要保留这种组合自由，必须区分外部启动许可与实际输入等待，不能用普通入口默认给已有实现补屏障。

## 5. 初始化：保留事实，不把所有 zero 合成同一默认

| 情形 | 最小归属与结论 |
| --- | --- |
| 生产者没覆盖、消费者仍要读的交接区域 | 保留 `handoff.initialize` 语义；它在生产开始前作用于新版本。共同入口不能提供具体初值。冻结语料未使用该字段不是删除依据。[规则 4:311](../rules/4.design.work_schedule.md:311) 的三角矩阵反例已足够区分。 |
| MoE Counts/Fill 每次归零 | 是实际工作与原子累加前提，不能删成分配细节。[root:401](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/root.yaml:401)、[source:9230](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:9230) 给出每次 memset；[k_count:599](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:599) 只原子添加非零贡献。最轻保留“每次 count 启动前完成 Counts/Fill 清零”即可。 |
| 首次 allocation 为零、此后自己复位 | 应保留首次状态和复用前提；不能改成每次 `initialize: 0`。MLA queue、MoE direct 的 RouteAccum/Arrivals 属此类。 |
| 完整覆盖或合法丢弃的旧 backing | 不凭 `zeros` API 或旧字节存在就添加初值；先检查真实读写及版本。[规则 4:320](../rules/4.design.work_schedule.md:320) 已有边界。 |

MoE 的 memset **不能完整塞入现行 handoff.initialize**：Counts 的所有原子更新都要累加身份值，不只关心未写区域；Fill 的初始零供 scatter 原子预约使用，却不是 count 计算出来的 handoff。现行 prose 可简明表达，但它不成为可寻址 service 事件。若要比较清零与独立计算的重叠，应再升级为供给/前置引用；当前 root 不要求为此枚举 memset API。

## 6. 跨次 scratch：短协议够当前阅读，不是并发安全证明

**GDN：** [root:379](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml:379) 三句已接近最小：复用 workspace、本次重写消费区域、无跨 stream 排他。[source:3258–3274](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu:3258) 确认 process 静态缓存按 device/容量换配并取 current stream。保留一处短来源条件即可；`storage: global` 不代表独占 allocation，也不授予跨请求重叠安全。

**MLA：** [refill:343](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/root.yaml:343) 的首次零、每 CTA 一张 terminal、总共 WorkCount 次 pop、末次 pop 复位可合在一个队列协议中。[source:1730](../../phase_grammar/sources/kernel/extract_mla/solution.py:1730) 与 [1461](../../phase_grammar/sources/kernel/extract_mla/solution.py:1461) 支持它。复位只供下一次有序使用；共享 Geo table 也不能因为 queue 复位就被证明可以被下一次 side 上传安全覆盖。

**QSA：** [root:189](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/template_qsa_prefill/root.yaml:189) 的 device ticket 和 host base 均从零开始，每次串行 eager 调用消耗 `worktile_count+worker_count`；[source:104–115](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/template_qsa_prefill/operator.py:104)、[157–160](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/template_qsa_prefill/operator.py:157) 确认缓存与推进。放进一个 `refill` 说明，不需第二套全局资源表。固定旧 base 的图重复回放会让增长后的 ticket 直接落在 terminal 域；共同入口/出口不能把它变成 replay 安全。

**MoE：** [direct:239](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/direct.yaml:239) 保存首次零与末次贡献者自清，无每次 memset；[workspace 源码](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_solution.py:969) 支持。不同方案的旧字节也不能混成默认零：[split-store:154](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/double_verify_split_store.yaml:154) 定义 R 行 backing，[179](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/double_verify_split_store.yaml:179) 明确先读后判定。保留 backing 合法范围及读取后果，不能用数学 shape 或 blanket initialize 代替。

**host gate 的反例：** Sparse [5588–5614](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_sparse/dsv4_kernel.cu:5588) 的 sentinel reset、有限轮询、最后一次普通 scan 不等于 probe 完成；超时也可能读到正值并接受部分平均。MoE [9481–9488](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:9481) 的 CPU reset 同样没等待旧 GPU writer，[verify_only:149](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/verify_only.yaml:149) 已保留来源限制。caller 的 GPU 顺序不排序下一次 CPU store。补 generation、锁或 host wait 是修源码，不是补语法。

## 7. 十算子覆盖与最轻方案的边界

以下“够用”仅指已定义的成功**单次计算组织**；仍保留上面的实例前提及源码限制，不表示所有调用组合成立。

| 固定 root | 共同 caller 约定之外仍保留的内容 |
| --- | --- |
| 模板 GDN | index_cast→gdn；显式 incoming state 与输出写集。 |
| 模板 FA4 prefill | 单计算提交；其内部工作/局部流水由原字段负责。 |
| 模板 FA4 decode | partial/combine 的 PDL 许可与真正读取等待。 |
| 模板 QSA | preprocess→attention；ticket/base 的串行 eager 前提。 |
| 提取 GDN | prep/chain 的 PDL；共享 workspace 的有序复用限制。 |
| 提取 GEMM | 正常单 kernel；实际发生的恢复提交按原节点保留，失败本身不产生事件。 |
| 提取 MLA | 当前 Geo 的启动前供给、metadata 侧流无 caller fork、queue 的有序复位。 |
| 提取 Sparse | root 的 attention→combine；hidden 分支的共同入口/出口和独立 prefix；host 观察语义。 |
| 提取 FMHA | root 单 kernel；分支内部 partial/combine 仍按原关系，不能以统一边界修复局部源码缺陷。 |
| 提取 MoE | 每次 Counts/Fill 清零、内部 PDL、首次初始化/自清状态；不外推 host mirror 的正确性。 |

**完整 host 请求确有反例。** MoE [ensure_tables:1015](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_solution.py:1015) 等整个 device 后才发布 ready；[已批准分支](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/init_no_scan_fused_small_flat.yaml:871) 明确保留。若其他 stream 有长任务，只等 `init_tables.kernel_done` 会提前继续 host 工作，源代码不会。因此标准 caller 输出汇合不能代替这个 host 边界；本轮可保留一句来源条件，不必让 flow 模拟所有外部任务。

**失败路径也不能套“正确默认”。** MoE [9533–9538](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:9533) 在 BeginCapture 失败后仍调用私有 stream 的 pipeline，随后 [9577](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:9577) 可能 eager 重提，源码没有在这里补 caller fork/join。root [20–22](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/root.yaml:20) 已排除该额外执行；不能当成 root 必须新增字段的证据，更不能隐式连边把失败路径描述成安全。

因此，本轮最轻方案足够覆盖上述有限目标；代价是放弃描述任意外部提前启动、各输出独立交付、完整 host 阻塞时序和跨请求物理资源组合。若用户要比较其中某项，再针对真实案例扩一个外部 ready/完成引用，比预先引入完整 boundary 对象更容易写对、读懂。

复用的调查是 [SCHEMA_DISCUSSION:60–106、192–231](../../../schema_extract/archive/20261109-ten-operator-text-audit/SCHEMA_DISCUSSION.md:60)、[SCHEMA_VERBOSITY:80–89、198–219](../../../schema_extract/archive/six-operators/20261109-methodology-review/SCHEMA_VERBOSITY.md:80) 和旧 [boundary_review:75–146](../../../schema_extract/archive/20261109-ten-operator-text-audit/gap_review/boundary_review.md:75)；本报告的关键关系均另行核对上述 YAML 和固定源码，未阅读本轮其他代理报告。
