# extract_sparse 全量文字审计（v1.1）

8 个现有节点都依赖一定的开放文字才能确定完整设计，但不能把全部长段归为语言缺口。数学谓词、舍入和 union 多重性是正式计算定义；多数局部发布、条件执行和资源复用已经有结构。较强的结构候选是 **调用外部 fork/join 端点与 host gate 的跨调用观察/复用生命周期**。纯 API 改名不构成核心反例。

已完成 427/427 精确文本组、8/8 YAML、dispatch、全部本地不同定义和 alias 使用上下文。统一校验 `python3 -B scripts/summarize.py --operator extract_sparse` 为 VALID，0 个 UNRESOLVED。分类中的 139 组包含非 FORMAL 核心片段；仅看 work_schedule/tile_pipeline/realization 是 101/149 组，覆盖全部 8 节点。这个数包括 MIXED 与 SOURCE_LIMIT，不能解释成“139个语言缺字段”，也不代表整段不可压缩。

## 身份与覆盖

输入为 pause inventory 选中的 current-plan：root v13、mma_direct v3、probe_hidden_mma v3，其余版本见表，8 个状态均为 approved。所有输入 SHA256 与 manifest 相符，记录在 [input_hash_checks.json](../work/extract_sparse/input_hash_checks.json)。来源均锁定 `source@6e4ca91bd91d998cac99e8dadd83838e153c679d`；candidate/solution.py 与 dsv4_kernel.cu 的固定提交原字节已与当前文件比对一致，SHA256 分别为 `2411a8fd28e09be34b14fa103ee616d8b6c3e2f2177d7f22b1089b9e0e491525`、`c9389387d35bd26954754a5ae13e2d202df06fc9799a77fddfb0654306909eb2`。引用的源码行号均来自 [work/extract_sparse](../work/extract_sparse/) 中的固定副本。

已读全部精确组正文与每个节点完整结构；为避免文本重复，结构阅读视图只把已读文本换成索引，保留全部其他字段，并对 packed 派生逐属性比较。scalars 中目录外的 257 种字符串均为引用、标识、成员选择、保留范围、shape/整数表达式等，未发现漏掉的长正文。硬件文字为零是因为 root.hardware_specification 仍为 null 骨架（root.yaml:153 起），不是“硬件无需说明”。既有 approved 状态不证明这些缺项已满足冻结规范。

规则使用本审计冻结 README、DOC0/1/3/4/5/6、REVIEW 和 examples 状态说明。读取固定 reference.py 全文及 input.py/metadata/shape_range 的相关契约材料；实现源码阅读覆盖相关数学、主 dispatch、probe、hash、ring、TMA/QIS2、combine 和图缓存，未复审所有强制/诊断/实验配置。未运行 GPU、Runner 或提取流程。

| 节点 | 状态/版本 | raw全部字符串声明 | raw全部字符串实例 | 开放字段声明字符 | 开放字段实例字符 | effective Design字符串实例 | verb声明/实例 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mma_direct | approved / 3 | 15632 | 15632 | 14380 | 14380 | 12018 | 9/9 |
| probe_hidden_mma | approved / 3 | 44631 | 44631 | 40673 | 40673 | 37596 | 35/35 |
| probe_hidden_packed | approved / 2 | 36410 | 49303 | 32980 | 44026 | 42684 | 27/53 |
| probe_serial_mma | approved / 2 | 18413 | 18413 | 17090 | 17090 | 15247 | 11/11 |
| probe_serial_packed | approved / 2 | 33868 | 33868 | 30731 | 30731 | 27894 | 27/27 |
| root | approved / 13 | 27131 | 27131 | 24737 | 24737 | 16547 | 13/13 |
| tc_heads | approved / 2 | 16517 | 20846 | 14113 | 18437 | 17312 | 11/24 |
| tc_packed | approved / 1 | 21808 | 26531 | 19032 | 23755 | 22213 | 13/26 |
| 合计 | 8 approved | 214410 | 236355 | 193736 | 213829 | 191511 | 146/198 |

所有 Design 三区都是本地定义，没有 from_parent，因此 effective Design 的差别只来自本文件 alias 展开，不应将 effective 与 raw 相加。`tc_heads`、`tc_packed`、`probe_hidden_packed` 有 alias；后者的 remainder 复用整个 packed pipeline，声明量 32980 与实例量 44026 相差 11046 个开放字段字符。alias 不共享两个 kernel 的物理资源。root 的开放字段仅24737字符、114个去重组；不能拿大树跨文件复制量代替 root 复杂度。

逐节点重点：

- root：split partials→combine，S/DQ及不同 Stages/QKS/Early；空 split 留未写 PartO，combine 的四次早取值与后段 split0 地址替代，padded head 四个 no-op 动作。
- mma_direct：直接 unsplit 与 gate 不可用且未提交 probe 的回落；P_run 参数、DQ重复QK、Q-ring与output staging别名。
- probe_serial_mma：8个probe CTA、有界观察后 cfg8 完整 batch；probe→attention 的设备完成边独立于host看到哪些样本。
- probe_serial_packed：GT4/8，独立 probe 后全 batch packed；跨request/尾组由attention正常处理，probe可给-1，不能混为删除工作。
- probe_hidden_mma：固定 packed prefix 保留，R拒绝只替换余段；余段使用针对完整T预先规划的cfg8，不能重新按短tail规划。
- probe_hidden_packed：probe/prefix/remainder三次提交，余段只等probe；两段独立worker、不同全局group偏移，共享文字不共享存储。
- tc_heads：GT1、HB64、所有head均真实；双QK issuer、单PV issuer、四个softmax读者、两个score轮转槽与loader tail汇合。
- tc_packed：GT2免probe；pool-major runtime union，query内重复次数保留；empty union用v/e域，hash/index/KV/output先后复用ring。

## 统一统计

字符按归一化后长度，包含公式、标识和引用，不是 token 或纯自然语言字数。英文 YAML 与中文 dispatch 分列。以下范围直接复用统一 `scope_statistics`：

| scope | 去重组 | 去重字符 | 声明数 | 声明字符 | 非FORMAL含核心组 | 候选缩减声明字符 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 427 | 131452 | 831 | 196649 | 139 | 5613 |
| yaml | 412 | 128539 | 816 | 193736 | 137 | 5562 |
| dispatch_md | 15 | 2913 | 15 | 2913 | 2 | 51 |
| main_WTR | 149 | 63361 | 270 | 107575 | 101 | 5043 |
| math | 215 | 46243 | 491 | 66295 | 23 | 301 |
| dispatch_when | 8 | 11930 | 8 | 11930 | 8 | 218 |

| 分类 | 精确组 | 去重字符面积 | 声明数 | 声明字符面积 |
| --- | ---: | ---: | ---: | ---: |
| FORMAL | 70 | 22771 | 184 | 27634 |
| EXPLANATION | 209 | 21020 | 378 | 31361 |
| DUPLICATE | 8 | 1144 | 10 | 1412 |
| NONCORE | 1 | 82 | 1 | 82 |
| STRUCTURABLE | 0 | 0 | 0 | 0 |
| ESSENTIAL_TEXT | 10 | 2250 | 11 | 2401 |
| MIXED | 127 | 83056 | 245 | 132630 |
| SOURCE_LIMIT | 2 | 1129 | 2 | 1129 |
| UNRESOLVED | 0 | 0 | 0 | 0 |

MIXED 的 83056 去重字符同时含核心、重复与来源说明，不能整体算可删或不可删。FORMAL 的 70 组含step/by正式身份、谓词和成本公式；部分数学长段因夹有执行绑定归MIXED。STRUCTURABLE全字段为0，不表示现有结构无用：当前混合字段中可迁移的局部定义已在F07/F10给出具体候选，尚不能把整项动作契约替换成纯结构。

## 18 个机制 finding

所有精确unit IDs、全部使用节点、证据、反例和结构候选见 [annotations/extract_sparse.json](../annotations/extract_sparse.json)。下面的组数是finding涉及组，不互斥、不相加；包含比较上下文或理由的组不都属于缺口的核心载体。

| Finding | 涉及组 | 判断及适用范围 |
| --- | ---: | --- |
| F01 | 103 | 正式公式/谓词/成本与身份：全8节点。保留slot多重性、sink、FP32分母/BF16概率、两半QK、lag、grouping、padding访问；不把公式字符串叫漏洞。 |
| F02 | 209 | 标题、meaning、thesis、reason及导航：无独占核心；多数必填，只做合法短写。 |
| F03 | 21 | 7个schedule.strategy（9次声明）完全重复worktile_order/preassigned，可省整个可选字段；其余参数、range偏移和空工作片段不能跟着删。 |
| F04 | 29 | 同文档anchor已有支持；跨文件只能整区from_parent，probe/hidden/packed不同phase和参数造成局部复用成本，属于表达成本候选。 |
| F05 | 18 | 外部事件端点：两个hidden节点的caller prior fork及最终join；另4个非dual节点的capture前caller同步。内部flow无合法外部producer/consumer。 |
| F06 | 17 | 四个probe节点的有界host观察、录制时选择、process gate跨调用复用；没有host观察/复用完成端点。不得把poll exit当kernel_done。 |
| F07 | 21 | ring、TMEM、控制slab已有resource/view，局部叙述可缩；union两段物理映射可用正式Upos公式+索引view.part进一步结构化。 |
| F08 | 25 | 现有结构已承载四读者、PV→reuse、empty v/e、内部start_after和final汇合；保留动作完成保证，不添加stage→loader尾提交等待。 |
| F09 | 42 | Main内异步priming/Q缓存/metadata/CTA rendezvous属于开放动作契约；serial外壳不定义prepare.done或enqueue.done。未提出只改动作名字的伪结构解。 |
| F10 | 41 | 固定数学参数可用symbol.value；role行/列分工可用精确区间公式短写。位置仍是已有开放语义，非证明缺新字段。 |
| F11 | 10 | probe与attention的哈希原子发布不同：前者Creator/First顺序可交错；后者等待完整canonical position；保留一次next-index预取/累积flush。 |
| F12 | 32 | QK共享通知与PV独立累积完成、Mask发布与数据落地分离、convergent TMEM读写的契约必须保留；事件边接这些保证，不凭API名推导。 |
| F13 | 17 | 源实现限制：5个TC相关节点的allocation-return/union-key与bar_land地址重叠，root的DQ非原子m/l重写，probe gate竞争。 |
| F14 | 18 | tolerance公式/相对基准/NaN/Inf/路由未知及fast-math无全域误差界；来源限制，不是任意精度许可。 |
| F15 | 7 | 共享Contract的out对象身份、允许写集、scratch无最终数值；meaning/compute有意开放语义，input意义的重复可缩，不能删主定义。 |
| F16 | 46 | 实现兼容与普通API区分：改变等待/可见性才是核心反例；cudaEvent包装、barrier编号或没有独立成本依据的layout不自动算核心。 |
| F17 | 48 | 空split、全invalid、空union、partial/cross-request probe、未观察全量与未提交的边界校验；现有数学/域/动作契约组合能保留。 |
| F18 | 2 | hardware null骨架是证据/填表不足；metadata与dispatch只是上下文锚点，没有漏掉硬件正文。已有硬件字段足够，不是语言缺字段。 |

最强缺口反例（F05）：删除hidden的fork/join叙述后，以下现有结构同时容许A正确等待caller先前的输入写入并让后继caller观察两段输出，和B抢读旧输入或后继提前读余段。区别是数据可见性，不是换了event API：

```yaml
flow:
  launch:
    - {phase: [probe]}
    - {phase: [prefix]}
    - {phase: [remainder], start_after: probe.kernel_done}
```

`start_after`只接受已定义phase事件。新增“caller_prior”字符串而无定义、造无计算的第四phase、或把`prefix.kernel_done`加到remainder都会伪造或改变设计。当前应保留有据文字；是否扩展调用接口/外部事件需另行评估。

F06的具体反例：两个调用共用gate，前一次probe未完成时后一次host reset，随后普通scan可以读到部分样本。改成每调用隔离gate、强制完成后再reset或把八个最终值当共同快照，都会改变原有状态/选择行为。正确内部设备顺序已有`start_after`，它不能替代host有界观察及跨调用存储身份。

现有结构替代实例（F07/F10）：

```yaml
# 数学正文先定义 Upos(k)=k if k<Ns, else 2560-Ng+k-Ns, 0<=k<Nu(grp)。
resource:
  UnionStore:
    space: SMEM
    member_slot_shape: [entry: 2560]
    dtype: int32
    slot_count: 1
    view:
      "UnionKey(k)": {part: {entry: ["Upos(k)", "Upos(k)+1"]}}
```

这只把原两段存储映射落到已有view语法，不代替hash位置发布及成员计数契约；原UnionKey对象不另配第二份存储。固定MR=64/KC=64可由其现有symbol移除bind、改为value并让下游引用，但GT、P.alt_*等来源关系仍需保留。不能将长文搬到另一个choice就称结构化。

## 保守缩减与限制

30个精确quote→replacement互不重叠，合计 **5613个声明字符**：YAML 5562，中文dispatch 51；去重候选净减少3103字符，root自身811字符。约占开放文本声明量196649的2.85%，不含alias展开倍增。main_WTR净候选5043/107575=4.69%。这是本次逐条证据支持的候选下界，既不是全文压缩潜力上界，也没有独立短稿等价认证。

典型候选：

- `Four such arrivals jointly authorize PV and score-slot reuse.` → 删除。保留前句的本warp行发布/score读取结束；PV data与QK reuse已列全4个前驱。
- `For c>0 wait for the preceding PV even if every scale is one; then conditionally rescale` → `For c>0 conditionally rescale`。现有无条件reuse边仍要求上一PV完成；c0的no-op契约保留。
- packed build最后一整句65536-byte table与ring容量证明 → 删除。容量/part及GT*IndexWidth门槛都已明确，原build rendezvous和ring归还句保留。
- `The source's allocation-return word overlaps ...` → `Source limitation: the allocation-return word overlaps a live transfer barrier; this design does not depend on that alias.`。来源限制仍在，不把坏地址作为必要设计。

不能按长度截断的字段包括prepare（最长完整组之一，root1266字符）、packed build（1497字符）、probe produce/build、root merge数学、hidden调度正文。它们分别承载异步完成边界、原子位置发布、观察模型、空split实际访问和调用外部完成关系。三类来源问题也不能用加同步或强行清零“修好”后宣布文本可删。

特别保留源码与抽象Design的区别：dsv4_kernel.cu:2104–2115的`bar_land`和`tmem_base_s/ukey`重叠，可直接从地址算式复核；2493–2515确有活跃barrier初始化和allocation。冻结REVIEW允许在逻辑对象预算、依赖和生命周期已有来源依据时记录边界外实现偏差继续提取。这不证明原字节代码运行正确，也不要求把坏地址写成复用视图。root的硬件Contract缺项、完整外部comparator和原始实测记录仍是限制，未重新运行GPU。

机器明细与统一统计：[annotations](../annotations/extract_sparse.json)、[audit_summary](../data/extract_sparse/audit_summary.json)。全部工作在本算子专属产物中完成；原schema、规范与运行状态未改。
