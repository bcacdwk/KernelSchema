# 四模板与 phase 组合的最小语言

结论：保留「kernel 分组、组内接续粒度、启动前驱、数据交接」四种关系；不能把它们压成一条顺序列表。可优先简化的是跨 kernel 必然为 `global` 的存储字面量，以及已在 realization 绑定的协议名。`ready` 与 `wait_at` 不宜合并或删除。真实的边界问题主要是首次初始化与跨调用共享状态，增加递归 phase 组合并不能解决它们。

本报告只读本轮规则、冻结 YAML、两份既有调查及必要源码，没有读取本轮其他报告。已用 CPU 解析 21 份模板，并逐字节核对当前文件；全部一致。没有 GPU 正确性或性能结论。覆盖清单及源文件定位见 [04_templates.json](/home/shaohanyong.shy/KernelSchema/schema_discussion/flow_review/data/04_templates.json)。

为缩短引用，以下路径前缀均相对仓库根，后缀 `:a–b` 是原文件行号：

- `R/` = `schema_discussion/flow_review/rules/`。
- `I/` = `schema_extract/archive/20261109-ten-operator-text-audit/inputs/`。
- `S/` = `schema_extract/archive/20261109-ten-operator-text-audit/work/`。

现行规则是 `R/4.design.work_schedule.md:234–324,365–381`。冻结模板的 `resource/milestone`、非 `MAIN` 单 phase 名称是历史写法，不作为新语法依据；本报告展示的 flow 片段本身符合对应标注的语法。

## 最小覆盖矩阵

| 模板分支 | 文件数 | 实际 flow | 能证明的关系与定位 |
| --- | ---: | --- | --- |
| FA4 prefill：root 与非 split 分支 | 9 | 一个 kernel、一个 phase | persistent、多 CTA worker、warp 分工不自动产生 phase；`I/template_fa4_prefill/packed_persistent.yaml:66–97` |
| FA4 prefill：两个 splitkv 分支 | 2 | attention 完成 → combine；全局 partial | 生产是 CTA 对，消费是单 CTA；分解与数量也不同；`I/template_fa4_prefill/splitkv_dual_load.yaml:91–145` |
| FA4 decode：q1/q4 direct | 2 | 一个 kernel、一个 phase | 单块 KV 归约留在 worktile 内；`I/template_fa4_decode/q4_direct.yaml:69–85` |
| FA4 decode：q1/q4 cluster | 2 | 一个 kernel、一个 phase | split 是 worker 的成员分工，统计交换和输出累加在内部；`I/template_fa4_decode/q4_cluster.yaml:115–139` |
| FA4 decode：root/q1_global/q4_compact | 3 | PDL partial → combine；全局 partial | 启动许可、grid 完成、读取等待各自登记；`I/template_fa4_decode/root.yaml:204–257,425–440` |
| GDN：root 与 m64_single_sequence | 2 | index_cast 完成 → gdn | root 消费 Cu32；m64 核心不读它，仍保留启动顺序；两文件分别 `:148–169`、`:83–103` |
| QSA prefill：root | 1 | preprocess 完成 → attention | 一个行块的 Union/Count 被多个 KV-head worktile 使用；静态生产与动态消费；`I/template_qsa_prefill/root.yaml:149–193` |

合计 13 个单 kernel、5 个普通完成依赖双 kernel、3 个 PDL 双 kernel。21 份都没有同 kernel 多 phase，16 个 handoff 条目全部写 `global`；9 个 PDL handoff 同时写 `ready/wait_at`。这些是文件覆盖数，不能当成独立设计样本数。

补充的真实同 kernel 多 phase 是 MoE `direct/no_scan_direct/init_no_scan_direct`；代表文件 `I/extract_moe/direct.yaml:183–285` 使用 `per_worker` 与 SMEM Census。四模板及这项补充均没有证明 `per_worktile/all/concurrent` 的实际使用，也未覆盖多个启动前驱、同一值的多消费者等待覆盖、早于生产 worktile 完成的局部发布或局部/全局混合交接。

## kernel 分组不能从数学步骤或数据边推出

FA4 prefill split 的 attention worker 是 2 CTA，combine 是 1 CTA，工作数量、行块大小、轴顺序都不同。把两个 launch 改成同 launch `order: all` 不等价：现行规则要求前三种 order 共享 worker 组成、数量及 placement；原设计依靠 kernel 边界重新选择映射。证据：`I/template_fa4_prefill/splitkv_dual_load.yaml:109–145`；`R/4.design.work_schedule.md:268–277,324`。

反过来，decode cluster 中每个 worker 是一个 CTA cluster，内部成员完成 split 与合并。其 partition 没有 split 轴，直接产生 output；不能因为出现两种数学步骤就强拆为两个 phase。direct、cluster、global 三类分支的区别已经由工作分解、worker 和 launch 表达，不需要 `fused: true`。证据：`I/template_fa4_decode/q4_cluster.yaml:128–138`，对照 `I/template_fa4_decode/root.yaml:228–253`。

以下为**现行语法**的 split/combine 最小片段；省略的 phase 定义仍采用原文件中的两种映射。

```yaml
flow:
  launch: [{phase: [attention]}, {phase: [combine], start_after: attention.kernel_done}]
  handoff:
    PartialO: {producer: attention, consumer: [combine], storage: global}
    PartialLSE: {producer: attention, consumer: [combine], storage: global}
```

`start_after` 也不能由 handoff 自动生成。GDN m64 先做 Cu32 转换，但核心传而不读 Cu32；没有 handoff，仍有完成前驱。省掉转换或启动边都改变已捕获设计；源码入口确实先调用 `.to(torch.int32)`。证据：`I/template_gdn_prefill/m64_single_sequence.yaml:86–103`；`S/template_gdn_prefill/source/aka_solution/solution.py:808–812`。

相邻 launch 默认串行同样不成立：Sparse `probe/prefix/remainder` 只要求 remainder 等 probe，不要求 prefix 与 remainder 串行。证据：`I/extract_sparse/probe_hidden_packed.yaml:299–324`。保留「列表仅枚举；前驱全部满足；省略不添加等待」比添加默认 stream/顺序再逐处撤销更简单（`R/4.design.work_schedule.md:255–260,375–381`）。

## order 四值：区分能力，不把每种能力都变成新字段

| 值 | 真正区分的设计事实 | 能否靠其他已有字段省掉 |
| --- | --- | --- |
| `per_worktile` | 同 worker 的 `A(t0),B(t0),A(t1),B(t1)`；一份中间值可消费后复用 | 两段确实共享完整工作定义时可合为一个 phase；只说 same worker 不够。无真实样本证明必须拆成两个 phase |
| `per_worker` | 每个 worker 先完成自己的 A 集合，再处理 B 集合；worker 互不等齐 | MoE 一份 Census 接多份 unit，不能改成一一配对，也不能默认 grid barrier |
| `all` | 所有 A 工作完成后才能开始 B；同 kernel 保持 worker 身份 | 用数据边只覆盖某块输入不能自动得到全体屏障；拆 kernel 会丢 worker 延续。当前仅规范能力，未找到实际样本 |
| `concurrent` | 不同 worker 实例分别承担 phase，数据边只限制必要消费 | 不等于同一 worker 的 role 并行，也不保证硬件同时驻留；当前仅规范能力，未找到实际样本 |

四值语义及限制的依据是 `R/4.design.work_schedule.md:268–277`；同 worker 的存储合账、不同 worker 的独立实例见 `R/5.design.tile_pipeline.md:499–501`。不建议因未出现而删未覆盖的三种能力，也不建议自动把所有局部计算步骤拆成 phase 以制造使用率。

MoE 是区别 `per_worker` 的最小实证：prelude 的 partition 是 replica，每个 worker 一份；direct 的 partition 是 unit，每个 worker 一个长度可变的连续区间。Census 留在本 CTA 的 SMEM，零 Census 的 CTA 直接结束。源码在本 CTA 的 census/barrier 后进入 unit 循环，没有 grid barrier。证据：`I/extract_moe/direct.yaml:201–226,273–293`；`S/extract_moe/source_nvfp4_moe.cu:1946–1985`。

因此保留单一 `order` 字段即可，不必再拆 `same_worker`、`barrier_scope`、`loop_scope` 三个互相约束的字段。多 phase 不设默认 order。建议保留现名；`all` 可在文档标题直接称「全体工作完成」，无需为阅读改善立刻迁移 enum。将 `concurrent` 的释义说清为允许独立 worker 推进，避免把允许重叠读成实测同时运行。

一个 launch 内「一组逐 worktile 串接、另一组同时独立运行」超出当前统一 order。现有样本没有证明这种混合组合不可用合理的单 phase/role 定义表达；本轮不据此引入递归 `seq/par/repeat` 或一般 host/stream 图语言。

## PDL：三个时点不能压成一个 after

decode partial 在 statistic store 发起后发布 PDL 许可，部分写入仍可在途；combine 要等待前驱 grid 完成后才读 partial。源 combine 在 wait 前有本地构造和累加器清零，但模板没有把这些地址准备单独抽成 phase，这一点合理。证据：`I/template_fa4_decode/root.yaml:343–358,427–440`；`S/template_fa4_decode/source_gqa_decode.py:393–442`。

更强的独立准备证据来自提取 GDN：chain 在等待 prep 前初始化 barriers、读取 initial_state，并可预取直接来自输入的 Q/K/V；这些不是已经就绪的 prep 产物。`ready=prep.kernel_done`、`wait_at=chain.prep_ready` 精确保留机会。证据：`I/extract_gdn/root.yaml:593–619,670–674`；`S/extract_gdn/gdn_kernel.cu:1977–2020,2176–2224`。

以下为**现行语法**，只展示 GDN 的一个代表值；其他交接值同理。

```yaml
flow:
  launch:
    - {phase: [prep]}
    - {phase: [chain], start_after: prep.launch_ready, protocol: pdl}
  handoff:
    Ti: {producer: prep, consumer: [chain], storage: global,
         ready: prep.kernel_done, wait_at: chain.prep_ready}
```

其约束是 `prep.launch_ready → chain 可启动 → 独立输入准备 → 等待完成 → 消费 Ti`，等待完成还须晚于 `prep.kernel_done`。把 `ready` 合到 `start_after` 会推迟整个 kernel；删除 `wait_at` 会把等待移到 worktile 入口；把 ready 默认解释成「有 PDL 许可即有数据」会提前读取。三个改动都不保真（`R/4.design.work_schedule.md:289–295,371–381`）。

## 字段取舍与可省略条件

| 字段 | 建议 | 能力与字段必要性的区别 |
| --- | --- | --- |
| `launch[].phase` | 保留，维持引用而不内联定义 | kernel 分组必要；单 phase 的外壳只有一行。允许省略整个 flow 是可行语法糖，但收益小，不主张额外维护第二种写法（R4:236–247） |
| `launch[].order` | 保留，多 phase 必填 | 接续粒度必要；四值共享一个字段已经比多个布尔开关简单（R4:268–277） |
| `launch[].start_after` | 保留，省略不添等待 | 启动边可以没有数据边；事件列表保留多前驱能力，即使模板未使用（GDN m64 与 Sparse 反例见上） |
| `launch[].protocol` | 可从 flow 移走，机制只绑定到 `realization.require` | 启动许可能力必要；协议名不是独立的先后边。现有 GDN realization 已绑定 PDL launch 属性、release 与 wait（`I/extract_gdn/root.yaml:684–686`） |
| `handoff` 的值键 | 保留数学值身份 | shape 相同不能推出同一值、版本或区域；不新增形状、索引映射的第二份登记（R4:301–305） |
| `producer/consumer` | 保留显式端点，可用 YAML anchor 复用相同条目内容 | 唯一生产者且输入登记完整时确可推导，但重复提交、视图、多个消费者会增加隐式解析规则；省下两个名字不值得。consumer 是 phase 集合，不是 worktile 一一匹配 |
| `storage` | 保留能力；仅纯跨 launch 的交接默认 `global` | 当前片上 worker 身份不能跨 kernel，故这类字面量可推导；同 kernel 的 global/on-chip 选择仍须写，不能由 `order: all` 或 shape 推断（R4:288,297,324） |
| `initialize` | 保留窄的缺口初值含义，省略不清零 | 初值能力必要；数学公式的初值不等于实现已经写出未覆盖区域。模板没有这个字段的实际实例，不能据此删掉规范能力（R4:289,311–322） |
| `ready` | 保留当前默认与显式覆盖 | 已有 `start_after: producer.kernel_done` 时通常无需重写 ready；PDL 下 grid 完成不能从相关 worktile 完成推出（R4:290–295） |
| `wait_at` | 保留按 consumer 覆盖 | 默认入口或已有启动/事件前驱已足够时可省；即使存在独立工作，若启动完成链已覆盖全部 consumer 动作，也无需重复。真正需要在入口后继续独立推进、稍后才等时填写（R4:291–295,381） |

表中 `R4` 指完整路径 `R/4.design.work_schedule.md`。producer 与 consumer 的多对多区域关系已有定义，不应因 QSA 一个 union 供多个 head worktile 读取而新增 `fanout` 字段；它们仍属于同一个 consumer phase（`I/template_qsa_prefill/root.yaml:157–187`）。

上述优先简化形成第三个片段，明确是**新提案**，不是可直接提交的现行实例：

```yaml
flow:
  launch:
    - {phase: [prep]}
    - {phase: [chain], start_after: prep.launch_ready}
  handoff:
    Ti: {producer: prep, consumer: [chain],
         ready: prep.kernel_done, wait_at: chain.prep_ready}
```

仅省掉两项：端点跨 launch 推出 global；既有 `realization.require` 继续绑定 PDL。三个时点和两个端点原样保留。若删除 protocol 字段，须同时将 `launch_ready` 定义为由明确绑定的启动原语提供的全 grid 许可；不得仅凭事件名猜测 PDL。当前 `R/4.design.work_schedule.md:372` 对协议定义的引用也要同步调整。这是字段迁移，不是删除实现前提。

storage 默认的代价是：少写一个字面量，却要先解析 producer 与全部 consumer 的 launch 归属；同 kernel 或混合归属仍须显式写存储。统一必填更便于逐条阅读，条件默认更紧凑。这里选择后者，但「可推导」本身并不证明净阅读成本一定降低。

不建议同时把所有跨 kernel ready 默认改成 kernel_done：它虽能再缩短 9 个模板条目，却改变原来的按区域就绪默认，使已有可表达的逐块交接需要补显式事件。无需为缩短几个字增加另一组依拓扑变化的时间默认。重复的三个/八个同构 handoff 可先用 YAML anchor；anchor 只复用内容，不合并数学值或物理存储身份（`R/0.common.md:64`）。

## 初始化频率与共享 scratch：真实边界在哪里

FA4 prefill 的空 split 不写 PartialO，combine 可预取未指定 payload，但仅在权重为正时进行算术使用；不由此要求 initialize。证据：`I/template_fa4_prefill/splitkv_dual_load.yaml:49–50,69–72,285–294`；源码分支见 `S/template_fa4_prefill/flash_fwd_combine.py:627–639`。这与 `R/4.design.work_schedule.md:322` 的「合法且无结果/控制影响的预取」边界一致。

FA4 decode 则不同：空 split 保留旧的有限 Opartial，combine 仍做零权乘法；需要有限值前提，不能把未写的 payload 当作任意比特。输入构造只在首次创建 scratch 时置零，wrapper 只切片；写成每次生产前 `initialize: 0` 会额外要求反复清零并改变流量。证据：`I/template_fa4_decode/root.yaml:27,161–185`；`S/template_fa4_decode/source_input.py:80–87`、`source_operator.py:227–254`、`source_gqa_decode.py:502–529`。

因此「值的每个新版本必须满足的缺口初值」「物理分配首次建立」「每调用重新置零」是三回事。decode 的有限 scratch 前提现在由输入 Contract 与数学说明承载，不能说整个 schema 完全无法描述；但单个 scalar initialize 不能结构化表达这三种频率，也不能表述“任意有限旧值”。不应为它悄悄解释一个 `once` 默认。

对 **initialize 字段本身** 的建议是暂保留，必要性证据弱于 ready/wait_at：它给「生产开始前，消费者要读而生产者不写的区域应有什么值」一个正式承载点，单靠数学公式不能证明该义务已由存储提供。等价替代可以是明确的输入/存储前提，未必仍叫 initialize；若有这种替代字段，它可以删除。规范的下三角例（`R/4.design.work_schedule.md:311–320`）说明现行字段如何工作，但本轮真实材料没有使用该标量字段的实例，不能以“初值语义必要”冒充“此字段已获实证支持”。

QSA 还提供了实际的反复使用但不重置：device ticket 与 host base 初始为零，后续调用共同推进；不属于 Union/Count 的 handoff 初值。证据：`I/template_qsa_prefill/root.yaml:189–193`；`S/template_qsa_prefill/operator.py:104–115,152–160`。把所有初始化汇总为 handoff.initialize 会混淆调度器状态与交接数据。

提取 GDN 的缓存 scratch 每次覆盖全部消费区域，却跨调用复用同一分配并依靠当前 stream 排序；`storage: global` 不能表示这个物理身份或跨调用复用条件。证据：`I/extract_gdn/root.yaml:379–383`；`S/extract_gdn/gdn_kernel.cu:3254–3274`。现有存续规则只保证某一数据版本的读写存续，不能自动证明下一调用不会覆盖它（`R/4.design.work_schedule.md:309–324`）。

这是 **flow 层的真实结构缺口，且涉及调用边界范围选择**。本轮建议保留短而明确的共享/串行使用前提；不为填满 flow 创建假的初始化 kernel，也不加入完整 allocator/cache 模型。若以后决定结构化跨调用状态，至少先确定具名物理存储身份、哪些值映射到它、哪个调用实例可复用；仅给 initialize 加频率枚举仍不完整。

现有十算子证据足以支持上述两项窄简化及保留边界；不足以要求递归组合、分支 flow、自动串行化或一般生命周期语言。凡改变实际 launch 集合、producer、物化或事件关系的分支，继续依当前 dispatch 规则分开，不能用注释取消结构边（`R/0.common.md:158–162`）。
