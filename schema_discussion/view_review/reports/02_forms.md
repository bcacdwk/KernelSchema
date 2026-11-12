建议保留可选的一层 `view`，但把它缩成**只记录占用区域的 `slot_index? / part?`**，完全删除 `view.dtype`。父 `storage.dtype` 仍是容量坐标的计量单位；值的数值格式沿用数学定义，本地对象和硬件 packing 沿用对应动作或 realization 的唯一绑定。省略 `part` 表示使用整槽的保守复用包络，较小值不必填满；只有并存、局部释放或必须保持的源布局依赖某个子区时才细化。

这比“上游类型优先、否则父类型默认、再允许 view override”更一致，也确实能少写一些定位。它没有把所有复杂共享压成一个简单标签：多槽不同映射、同时存活的子区、异地副本仍要表达。**不推荐把所有逐值属性提升到父 storage，也不推荐另建 flat storage + backing/alias 体系。** 普通单值、独立不相交区域已经可以平列；无需为了它们新增另一套语法。

本轮以 [inputs.json](../inputs.json) 固定的现行 `storage/index/event` 规则为准。旧 `01_storage` 报告只用于定位，以下判断重新回查冻结 YAML 和 Kernel。这里只提交规则研究，不修改规范、算子或源代码；下面所有新写法都是草图，尚未完成全文迁移和合法性证明。

实际覆盖如下。所有 root 和四个分支都是 corpus 中原 working checkpoint 的 `approved`，这一状态不等于新语法认证。没有把 `review_candidate` 或未创作节点当作已批准设计，也没有重审整树。

| 范围 | 实读的相关约束 | 独立回查的源段 |
| --- | --- | --- |
| GDN root | 单值 R；Hrhs/Out/DU0；多槽混合 Stages；stage 0 最终状态；三角物化 | `gdn_kernel.cu` 1902–1935、2495–2511、2822–2839；`gdn_common.cuh` 50–65 |
| GEMM root；mma_sync 分支 | 普通 F；TMEM/寄存器不同双缓冲；分支 A/B/AS 同槽并存、Output 固定槽 0、末尾空 copy groups | `fp8gg_umma.cu` 315–345、592–665、744–768；`fp8gg.cu` 33–38、177–212、263–280 |
| MLA root | 三槽 S/P、3/4 槽 K/V、寄存器副本、跨工作持续序号、terminal | 冻结 `solution.py` 464–470、905–945、1080–1100、1259–1277、1333–1374 |
| Sparse root；tc_packed 分支 | root 的 Q/Gather/Partial arena；分支 HashTable/IndexStage 并存、双槽 Gather、输出跨槽 | `dsv4_kernel.cu` 878–896、1162–1173、1424–1470、1946–1951、2210–2235、3530–3545 |
| FMHA root；tcg_d128_flat_tma_xpose_narrow 分支 | root 的 QK/PV 在途重叠和独立 TMEM 子区；分支整个 operand arena 改存 partial | `fmha_tcg.cuh` 1815–1975、2987–3005、4888–4923、5394–5424 |
| MoE root；direct 分支 | payload/scale 不同环，寄存器本地索引与 SMEM 连续索引；direct 跨 phase 共用 Exchange | `source_nvfp4_moe.cu` 1929–1976、2218–2237、4349–4367、4491–4514、4908–4921 |

十稿做了完整结构扫描：125 个 storage 条目、139 个 view、49 处 slot_index、82 处 part、41 处 view.dtype。这只是定位和填写量统计，**不是 41 个 dtype 都已证明可自动删掉**。十八份源码全部复核 corpus hash 并做关键词扫描，其中八份实读了上述相关计算/布局段；其余 wrapper/helper 不声称完成语义审阅。MLA 源内 MODE1 的 FP16 路径只作为表示宽度旁证，不冒充 root 的 MODE0 范围。精确路径、行号、原文和覆盖清单见 [02_forms.json](../data/02_forms.json)。

候选方案用同一组六类片段比较，不能只看缩进。

| 候选 | 单值 F / 同类 Exchange | MLA S/P | Sparse arena / GEMM InputRing | FMHA 独立子区 | 结论 |
| --- | --- | --- | --- | --- | --- |
| A：区域型精简 view，整槽包络为缺省 | 单值免 view；Exchange 两个空条目即可 | 保留两条映射；不复写 dtype；本轮保守保留 P.part | 整区阶段用途可省 part，实际环槽和并存子区仍明确 | 可保留必要 part，也可按既有规则分列独立区域 | **推荐，一个结构** |
| A 的类型 fallback 变体 | 简单例子短 | 要规定 P 的上游 FP8 胜过父 FP32 | 本地名又要判断是否回退父类型或写 override | uint8 override 仍不表达 E8M0 atom | 多一组优先级规则，没有找到必须如此的反例 |
| B：逐值属性提升到 storage | 单值可浅一层；Exchange 换成 values 列表 | 共同 `n%3` 能少一次复写 | `Output(t)=0` 与 `Stage(t,p)=p%3` 不同；Gather 与整个 arena 也不同 | 每值不同区域/格式必须恢复逐值记录 | 只在同质子集上简洁，统一后要默认/覆盖或并行表 |
| C：逐值 flat 条目 + shared_with/backing | Exchange 两值尚可 | 仍需 TMEM 共享身份和两个独立寄存器副本 | 需要 backing、容量一次计账、逐值区域、跨槽含义及引用作用域 | 已有平级 storage 足够 | 把父子关系改成引用，没有删掉所需事实 |
| D：仅整区分时复用组，不允许 part | 很合适 | 完整读取交接若被证明才可用；当前 tiny 分区不能自动抹掉 | 稳态多个 Gather、A/B/AS 并存仍须另造子区/模式 | 会抹掉允许重叠的 QK/PV 关系 | 只把其整区语义吸收到 A，不新增第二语言 |

**F01：可删的是 view 的类型职责，不是数值格式和 packing 事实。** [MLA 数学](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/root.yaml#L199) 已给 P 的 E4M3 转换和 dtype；[源码](../../sources/kernel/extract_mla/solution.py#L464) 同时给出 S 的 FP32 容量列、MODE0 的 32 列 FP8 P、MODE1 的 64 列 FP16 P。`u32 pk[32]` 是四个 FP8 的搬运载体，不是另选一次 P 的数值格式。省掉 `view.dtype` 后这些表示照旧由值/转换与实现绑定确定；父 `dtype: fp32` 只把 128×128 容量算成 65536 字节，不再是 P 的默认解释。

最强的本地反例也不要求在 view 保留类型：[Sparse/tc_packed 的 build](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/tc_packed.yaml#L363) 已写 `8192-word uint64 HashTable`；[FMHA realization](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/root.yaml#L674) 已绑定 E8M0 atom 的 TMEM 表示。view 中的 `uint64/uint8` 有重复，其中 uint8 本身还不足以说明 atom 格式。若别的稿件只有 view 中留着这一事实，迁移时必须把它放入创建该对象的已有动作契约，或对应 realization 绑定；这叫移动一个必要事实，不算减少作者决定的内容。不新增一张类型表，也不允许多个消费者各自选择格式。

反过来，不能把父 dtype 全删掉、保留原 shape/part 的数字不变。MLA 的 column 32 指 32 个 FP32 容量单元，换成 bytes 需要整体换算。把 `shape+dtype` 改成 `size_bytes` 对无 part 的 S/P 可少一个容量属性，但有逐行区域时必须补回坐标单位/shape/stride，不能靠一个连续 byte 区间代替。脚本复算了两个同为 16384 字节的集合：每行 `[32,64)` FP32 列，与整个 tile 的连续 `[16384,32768)` 字节，交集只有 4096 字节。等容量不是同布局。

还有一个防止“漂亮推导”越界的事实：[GDN 的 Ti](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml#L264) 是 64×64 BF16 数学值，dense 量是 8192 字节；[源码](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_common.cuh#L50) 只物化十个下三角 16×16 tile，共 5120 字节。[现有 realization](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml#L688) 还记录每 tile 的两个 16×8 半列块。dtype 可唯一确定，实际物化足迹却不能由完整数学 shape 机械算出。

**F02：整区先后使用，不必写较小用途的首地址。** [MoE/direct](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/direct.yaml#L291) 用一个 `s_warp[16]` 先保存各 warp 计数，Census 消费并完成 CTA join 后，再在最终路由收集中使用前 K 个元素；[实际写读](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu#L2218) 没有要求剩余计数继续存活。决定是“这 64 字节复用”，不是“FinalRoutes 必须从 element 0 开始”。同样，[Sparse](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/root.yaml#L444) 先把 Q 搬入寄存器并 join，随后才使用其 ring；[末尾](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/root.yaml#L489) 在所有 Gather 读完后改存 FP32 partial。入口/出口用途都可以只占有整个 arena 的复用生命周期。

省略 part 的精确定义应是：**这一用途的访问必须容纳于整槽，分析上整槽都视为可能冲突；不声称它把槽填满，也不为未写字节自动授予并发用途。** 当设计依靠两个小值同时存在时必须给出不相交区域，或分列独立 storage。这样 whole reuse 是同一 view 的缺省，不需要 `epoch/mode/reuse_group` 三种新记法。

**F03：省掉槽公式的文本重复，可能只是增加了绑定规则。** MLA 的 S(n)、P(n) 可以共用一个父 `n%3`；但 [GDN Stages](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml#L575) 的九类 c 版本与无 c 参数的 FinalState=0、[GEMM mma_sync](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gemm/mma_sync.yaml#L163) 的四类 p%3 与 Output(t)=0，都不是一个父公式。若再允许逐 view 覆盖，就要解析父形参、继承条件和覆盖优先级；把属性各自做成按值映射则要反复 join 同一值，仍没有更少的关联事实。

例如下面这个候选，只有第一段真正省了一次重复；第二段已经恢复了逐值例外，并要解释父 p 在 Output(t) 处如何不求值。这不是推荐语法：

```yaml
ScoreProbability:
  values: ["S(n)", "P(n)"]
  slot_index: "n%3"
# P 的不同 part 仍未表达，不能把这个 slot 聚合片段当作完整替换。

InputRing:
  values: ["Stage(t,p)", "A(t,p)", "B(t,p)", "AS(t,p)", "Output(t)"]
  slot_index: "p%3"
  overrides:
    "Output(t)": {slot_index: 0}
# A/B/AS 的不同 part 还没有找到同级且无需关联的存放处。
```

这里必须区分两种“重命名”。把 view 内形参 `S(n)` 改成 `S(j)`，同时把映射改成 `j%3`，只是局部形参换名，不改动作实例。把工作内 `p` 改成跨工作 `n`，或把 `AS(t,p)` 的动作签名改成 `AS(n)`，则可能改变版本身份。[MLA 源码](../../sources/kernel/extract_mla/solution.py#L1259) 的 T 跨 item 持续，局部 t 每 item 重来；N=[1,2,1] 时连续槽为 `[0,1,2,0]`，局部取模为 `[0,0,1,0]`。[GEMM 的两个终端空 group](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gemm/mma_sync.yaml#L189) 仍有事件且被 drain 等待；KB=4、两份工作时，用 `n=t*KB+p` 代替它们的 `(t,p)` 会把 12 个 load 身份压成 10 个，其中上一工作的空 group 与下一工作的真实 load 碰撞。正确的另一个 flatten 可以保留身份，但不能声称“重命名自然解决”。[MoE 源码](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu#L4491) 也在同一循环里分别用 `g%SFS` 取 SMEM scale 槽、`ks&1` 取寄存器槽。

同槽不同代次也不能消失。MLA 一个槽先有 `P(n-3)`，随后 `S(n)`，再是 `P(n)`，之后 `S(n+3)`。当前 S 的常规读、tiny 的后续首32列读取、旧 P 的 operand-read 退休和下一轮覆写是不同约束。ScoreRegisters 接走 S 不能自动释放旧 P；固定 P 的 `[32,64)` 列与 tiny 的 `[0,32)` 列本来不相交，不能把 tiny 退休擅自强化成所有 P 发布的全局前置；早期 P 读与较晚 QK 写之间依靠绑定的同一有序矩阵流。[root 479–491、565–591](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/root.yaml#L479) 和 [源码 1080–1100、1302–1309](../../sources/kernel/extract_mla/solution.py#L1080) 给出了这些关系。删 dtype 不删对应代次和依赖。

**F04：需要保留的是空间共存关系；绝对位置可以有条件地晚定。** [FMHA root](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/root.yaml#L495) 的 ScoreHalf、scales、NumState、Prob8 是不相交子区。[源码 DPV](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_fmha/source/candidate/fmha_tcg.cuh#L2991) 允许 PV(g-1) 在当前两次 QK 的准备和读回下继续执行，到中途才退还 V/Prob8/NumState。这些区域不能一律改成“整个 TensorMemory 轮流使用”。

但“NumState 必须和 ScoreHalf 不相交”不等于 Agent 必须先选 column 96。现行规则已允许独立设计区域分列、实现合并申请：若不依赖部分别名，可把这些子区独立列账，让 Coder 选择兼容位置。这会增加容量头的复写，不能宣传成总字段更少。若设计要在它们中的某几块上再放一个跨区对象，继续用父区域+part 更直接；否则就需要额外的 backing 身份或区域组合语言。

固定源码提取与前向设计也要区分：源码已经选择的 part 是事实记录，可以留在 realization；前向草图省略起点不意味着任意新布局都合法。[GDN 最后 mini-chunk](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml#L651) 的 stage 0 读取退休缺口仍存在，[实际写状态](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu#L2822) 前没有对应的 CTA join。扩大或删去 part 不能把它认证为安全。

MLA 的 P.part 也因此在本轮主草图中保留。虽然各 warp 的本地代码先做 tiny 再写自己的 P，但这不等于所有 warp 先读完再开始任何 P 写；[源码](../../sources/kernel/extract_mla/solution.py#L852) 还存在已知的 `tb=tb0` 读地址与 P 的 warp 行偏移不一致。完整每行所有权、跨 warp 读取退休尚未在本轮证明。若以后证明这些关系允许省掉列位置，可以再采用整槽包络；若为了省一个 part 必须新增全量等待、改变重叠，则不应采用这项省略。不能把“字段可选”当作对这个具体实现的认证。

**F05：flat+共享标签没有消灭 arena 的父身份。** 最简 Exchange 可以写 `FinalRoutes.shared_with=WarpCounts`，但这马上需要两类条目：一个负责容量/location/depth，另一个继承这些属性且不计第二遍。MLA 再出现 `S_tmem/S_reg`、`P_tmem/P_reg` 时，必须保持四个副本身份以及其中两个共用 TMEM 的关系。若每条都独立列容量，则还需 backing 标识和一致性规则；若只有首条负责容量，首条就隐含变成父区域。

部分重叠不是一个“同址等价类”。[Sparse/tc_packed](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/tc_packed.yaml#L246) 的双槽 Gather 分别在 `[0,65536)` 和 `[65536,131072)`；OutputStage 的带 padding 包络是 `[0,66560)`。输出与两槽各相交，两槽彼此不相交。HashTable 与 IndexStage 在同一 arena 中还必须同时放得下且不相交。`same_address` 会说错，`same_backing` 又没有告诉人哪些范围重叠。补回 backing+part+容量所有权后，表达能力回来，但已重新建立 view 的关联，未减少事实。

**F06：不精确定位仍可做有边界的容量和生命周期核对。** 保留父区域容量，只计一次；核对各活动集合的物化足迹和可能同时存在的槽；整区交接前核对旧写已完成且所有必要读者退休。依赖只记录最后读取，若旧异步 fill 还会落地，同样不能覆盖。以下都是可核对的事实，而不是让 Coder 随便塞：

| 实例 | 可以现在核对的容量/交接 | 不能据此宣称的结论 |
| --- | --- | --- |
| MLA root S/P | S 每槽 65536 B，P 为 16384 B，三槽共 196608 B；当前有明确每行 P 子区 | 全部 S 读者已在任何 P 写前退休；16384 B 任意位置都合法；MODE1 属于该 root |
| Sparse root cfg2 | HB=64、KC=64、Stages=2、D=512；arena=131072 B，FP32 partial=131072 B，大于一个 Gather 槽 65536 B | 能把全部用途都硬改成 `depth=2` 的等大独占单槽，而不处理 partial 跨槽 |
| Sparse root cfg4 | HB=32、KC=64、Stages=3；Q=32768 B，搬完到寄存器并 join 后才让 ring 用作 Gather | Q 和 Gather 本来需要共存，或任意 Q 缓存策略都成立 |
| GEMM mma_sync | 单输入槽 24832 B，BF16 输出带行 padding 的包络 17408 B；drain wait-all+CTA join 后可覆盖 | 空 copy group 没事件；矩阵读完便能忽略尚在途的 copy |
| Sparse tc_packed | HashTable+最大 IndexStage 为 75776 B，arena=131072 B；build join 后可改用途 | 一个同址标签足以表达两者并存；两个 Gather 版本只算一份 |
| FMHA xpose 分支 | 81920 B operand arena 容纳 67584 B FP32 staging 包络 | 空 split 自动没有 Query 写者；源码已保证写者退休 |
| MoE fc1 | 实用 C1+双 scale 区共 192 列，源码按有效粒度申请 256 列 | 删除 view 中未占用的 64 列即可把驻留账改成 192 列 |

容量/交接依据分别是 [Sparse static_assert](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_sparse/dsv4_kernel.cu#L878)、[cfg 表](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_sparse/dsv4_kernel.cu#L4463)、[GEMM drain](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gemm/mma_sync.yaml#L201)、[xpose staging](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/tcg_d128_flat_tma_xpose_narrow.yaml#L604)、[MoE 分配粒度](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu#L4349)。其中 xpose 的空 split 源缺口和 GDN 的末轮缺口都保留为来源问题。新语法不制造同步保证。

**F07：同一个值的副本不是重复格式声明。** [MLA root](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/root.yaml#L387) 的 TMEM S/P 与两个寄存器副本有不同生命周期；[GEMM DrainRegister](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gemm/root.yaml#L315) 的两个槽支持下一次 TMEM 读与当前 FMA 重叠。[实际代码](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gemm/fp8gg_umma.cu#L654) 在最后读完成后归还 TMEM，再做最终 FMA。格式只定义一次，并不意味着存储副本也只有一份。各动作仍须点名具体 storage/view，普通临时指令寄存器不因此逐项入账。

下面是同例 before/after。before 将冻结字段名归一到当前 `depth/location/shape/total_size`，total_size 依现行规则补算；不把旧命名当缺口，也不声称下面是原 YAML 原文。before/after 的动作、域、访问绑定和等待全部需要保留，片段只显示这里比较的 storage。

MoE/direct：去掉 FinalRoutes 的固定前缀，跨 phase 仍引用同一个 Exchange；格式由计数/路由对象与动作定义承担。

```yaml
# before
Exchange:
  depth: 1
  location: SMEM
  shape: [element: 16]
  dtype: int32
  total_size: 64
  view:
    WarpCounts: {}
    FinalRoutes: {part: {element: [0, K]}}
```

```yaml
# after，建议而非现行语法
Exchange:
  depth: 1
  location: SMEM
  shape: [element: 16]
  dtype: int32
  total_size: 64
  view:
    WarpCounts: {}
    FinalRoutes: {}
```

MLA：两值仍绑定同一组三槽，P 不继承父 FP32。P 的 FP8 表示已由 weights 定义；本轮保留原 P 子区，不把尚未证明的全槽读退休当作事实。旧 P(n-3)、S(n)、P(n)、S(n+3) 的关系仍要核对。

```yaml
# before
ScoreProbability:
  depth: 3
  location: TMEM
  shape: [row: Bq, column: Bk]
  dtype: fp32
  total_size: "3*Bq*Bk*4"
  view:
    "S(n)": {slot_index: "n%3"}
    "P(n)": {slot_index: "n%3", part: {column: [32, 64]}, dtype: fp8_e4m3fn}
```

```yaml
# after，删除类型职责，保留本轮仍需记录的部分区域
ScoreProbability:
  depth: 3
  location: TMEM
  shape: [row: Bq, column: Bk]
  dtype: fp32
  total_size: "3*Bq*Bk*4"
  view:
    "S(n)": {slot_index: "n%3"}
    "P(n)": {slot_index: "n%3", part: {column: [32, 64]}}
```

Sparse root：统一用同一 arena 的 view，没有另造“阶段复用声明”。只有稳态 Gather 的多个同时存在版本还需要区域映射。Q_ring/PartialStage 的数据来源和格式分别由既有 prepare 的 Q 绑定、finish 的 FP32 Acc staging 绑定明确；若迁移工具不能解析这些本地名，须报告未解析，不能回退成父 uint8。

```yaml
# before
ring:
  depth: 1
  location: SMEM
  shape: [byte: "2*Stages*KC*D"]
  dtype: uint8
  total_size: "2*Stages*KC*D"
  view:
    "Gather(c)": {part: {byte: ["2*((c-c0)%Stages)*KC*D", "2*(((c-c0)%Stages)+1)*KC*D"]}, dtype: bf16}
    Q_ring: {part: {byte: [0, "2*HB*D"]}, dtype: bf16}
    PartialStage: {part: {byte: [0, "4*HB*D"]}, dtype: fp32}
```

```yaml
# after
ring:
  depth: 1
  location: SMEM
  shape: [byte: "2*Stages*KC*D"]
  dtype: uint8
  total_size: "2*Stages*KC*D"
  view:
    "Gather(c)": {part: {byte: ["2*((c-c0)%Stages)*KC*D", "2*(((c-c0)%Stages)+1)*KC*D"]}}
    Q_ring: {}
    PartialStage: {}
```

GEMM mma_sync 的 after 刻意保留不同槽映射和所有并存子区。这里 AS 的来源是已给 FP32 类型的 a_scale，Output 来自 F 到 BF16 输出的转换；删去两个 view.dtype 不改变这两条数据绑定。

```yaml
# before 的 view，父区域每槽 E=(Bm+Bn)*ScaleK+4*Bm 个 FP8 容量单元，depth=3
view:
  "Stage(t,p)": {slot_index: "p%3"}
  "A(t,p)": {slot_index: "p%3", part: {element: [0, "Bm*ScaleK"]}}
  "B(t,p)": {slot_index: "p%3", part: {element: ["Bm*ScaleK", "(Bm+Bn)*ScaleK"]}}
  "AS(t,p)": {slot_index: "p%3", part: {element: ["(Bm+Bn)*ScaleK", "(Bm+Bn)*ScaleK+4*Bm"]}, dtype: fp32}
  "Output(t)": {slot_index: 0, dtype: bf16}
```

```yaml
# after，同一父区域
view:
  "Stage(t,p)": {slot_index: "p%3"}
  "A(t,p)": {slot_index: "p%3", part: {element: [0, "Bm*ScaleK"]}}
  "B(t,p)": {slot_index: "p%3", part: {element: ["Bm*ScaleK", "(Bm+Bn)*ScaleK"]}}
  "AS(t,p)": {slot_index: "p%3", part: {element: ["(Bm+Bn)*ScaleK", "(Bm+Bn)*ScaleK+4*Bm"]}}
  "Output(t)": {slot_index: 0}
```

填写负担按“语义赋值项”核对，不数行数或缩进：每个完整 shape 算一项、一个范围映射算一项、一个值绑定算一项；共同不变的五个容量字段都计入。这里没有通过省 total_size 或给 depth 加缺省来制造收益。它们的派生性另有明确现行规则，不属于本次选择 view 结构的必要变化。

| 同例 | 现行归一片段 → A | 真正减少了什么 | 仍必须决定/记录什么 |
| --- | --- | --- | --- |
| GEMM 单值 F | 5 → 5 | 无；原规则已能免 view | 容量、位置、存续；类型沿用上游 |
| Exchange | 8 → 7 | 一个可晚定的具体前缀选择 | 两个用途、同一容量、前读者结束 |
| MLA S/P | 11 → 10 | 一处已定类型复写；进一步到9的 part 省略尚未证明 | P 子区、三槽、两条映射、各代次复用关系和副本 |
| Sparse ring | 14 → 9 | 两项入口/出口定位、三处格式复写 | arena 总量、Gather 环内映射、每阶段绑定及交接 |
| GEMM InputRing | 20 → 18 | 两处已由 load/convert 绑定确定的类型复写 | p%3 与固定0、三个并存子区、terminal group |
| FMHA TensorMemory | 24 → 19 | 五处 view 格式；真实 atom 格式仍在 realization | 七个区域和相应在途关系；未减少布局选择 |

若为了绝对扁平把 FMHA 七个子区全部独立列为 storage，五个容量字段变成 35 项，虽然可以把七处起点交给实现，文字赋值项却比 24 更多。这个方向的收益是减少提前定位，不是所有填写量都减少；它也必须保留实际合并申请和有效粒度约束。类似 MoE 的 192/256 列场景，在没有处理粒度账之前不能简单拆散并声称省了容量。相反，flat+shared_with 的条目可能更短，但增添的 backing 引用、容量主从和相对区域仍是作者/审阅者要跟踪的事实。

最终建议只有一套规则：`storage` 管容量，`view` 可选且只管区域/版本映射，值/动作/realization 管各自已经有定义权的表示。整槽复用采用 view 缺省；局部例外继续用 part；同名普通单值沿用现有免 view 规则。必须保证的数据绑定、格式、物化足迹和同步不能靠省字段变成未知。若所有读者/旧写已经退休且容量足够，offset 可以晚定；若合法性依赖局部共存而当前没有定位或独立区域约束，就还没有完成设计，不能装作已证明。

[02_forms.py](../scripts/02_forms.py) 已核对 7 份现行规则、10 份选定 YAML、18 份源码的固定 hash，验证 JSON 中 98 处逐行原文，并运行容量、区域集合、持续槽及 terminal 身份的有限反例。它不解析新语法、不证明完整事件图、不验证 TMEM 指令地址/对齐/swizzle，也不修复已知源缺口；未做 GPU 执行。对其他分支、本地对象格式迁移和全域布局合法性仍保留这些明确边界。
