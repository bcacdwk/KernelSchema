# View：应保留占用与别名，删除独立的数值解释

建议**完全删除 `view.dtype`**，把 view 定义为“值的某个版本绑定到哪块字节区域／占用区域”，不再让它决定数值解释。数值类型属于上游选定的值；本地协议对象和物理编码由其生产动作或被引用的 realization 契约定义一次。没有发现本轮真实案例必须把格式字段放在 view 层。格式信息仍然必要，但这不等于 `view.dtype` 必须存在。

同时建议把相同的槽映射提升为 storage 级显式公共规则；把 `part` 的必填条件收紧到**确实需要区分同时存活子区或部分覆盖**的情况。已证明所有冲突读写都退休的整块顺序复用，可以按整槽 reservation 登记，即使前后值的 shape、dtype、实际写入量不同，也不必先定偏移。**MLA S/P 未满足这个证明条件，本报告的最终草图保留其 part 和行分工。** 复杂部分别名仍需区域约束，不能只写一句“共享”。本轮不修改现行规范或算子。

## 范围与核验方法

当前规则取自 `view_review/inputs.json` 指定的七份 `view_review/rules` 快照；不把父目录改名前的 `sources/rules` 当现行规范。重点实读当前第 5 篇 storage/view/index、第 3 篇数值接口与本地物化、第 1 篇参考 dtype。旧 YAML 的 `resource/space/slot_count` 只按相应存储语义读取。

六个暂停 root 的 storage/view 映射全部解析；再实读相关上游对象、生产／消费／复用动作和源码。只补四个有必要的 approved 分支：MoE/direct、Sparse/tc_packed、GEMM/mma_sync、FMHA/tcg_d128_flat_tma_xpose_narrow。没有读取另一位代理本轮报告，没有重新提取整树，没有 GPU 执行。

| Root | storage | view | view.dtype override | 本轮关注 |
| --- | ---: | ---: | ---: | --- |
| GDN | 14 | 22 | 11 | 混合 arena、三角 pack、rhs/output、最终状态覆盖 |
| GEMM | 9 | 9 | 0 | 不同环的版本轴、寄存器副本、mma_sync 的整块复用 |
| MLA | 15 | 18 | 1 | S/P、不同驻留副本、持续环、Geometry |
| Sparse | 15 | 12 | 5 | Q/Gather/partial、metadata/reduction scratch、packed union |
| FMHA | 9 | 13 | 5 | 持续 NumState、scale 编码、xpose 的跨槽 arena |
| MoE | 20 | 14 | 2 | E4M3 scale packing、不同节奏的环、Exchange |
| 合计 | **82** | **88** | **24** | |

六 root、四分支、十八份源码和七份新规则的 hash 均与输入清单一致；十八份源码中实读语义相关段落的文件为八份，其余只作 hash 核对或定位。逐个 override 的分类、原文行号、quote 与 hash 保存在 [01_semantics.json](/home/shaohanyong.shy/KernelSchema/schema_discussion/phase_grammar/view_review/data/01_semantics.json)。[01_build_data.py](/home/shaohanyong.shy/KernelSchema/schema_discussion/phase_grammar/view_review/scripts/01_build_data.py) 可重建数据并验证这些 hash；它不是算子测试。

## 三种类型／格式不应混成一个继承链

| 信息 | 本轮真实例子 | 唯一来源应在哪里 |
| --- | --- | --- |
| 数学／设计值的数值类型 | MLA `S:fp32`、`P:fp8_e4m3fn`；GDN `Ti:bf16` | 被当前设计实际采用的值定义及转换公式。Contract 的参考中间量不能直接替代选定的 Design 精度。 |
| backing 的容量与坐标单位 | `Stages.dtype:uint8` 的 element 是字节；TMEM 的 `dtype:fp32` 使 column 对应四字节单元 | storage 的容量账。它不意味着 arena 内每个对象都按该数值类型读取。 |
| 存放编码、打包、物理布局 | E4M3 四字节打包；E8M0 `0x7f`；key/position 的单个 64 位原子字；三角 tile pack | 本地对象的生产协议／被引用的 realization 契约。格式影响数值时，转换还必须反映在数学路径中。 |

当前规则规定 `part` 先按父 storage 的 shape/dtype 选字节，随后按 view.dtype 解释。**既然选字节本来就不依赖 view.dtype，删除它不破坏 part 的坐标定义。** 需要同步删除“省略时继承父 dtype”的访问解释；否则直接删 YAML 字段会把 P 或 bf16 arena 子区错误解释成父 dtype。

这也回答“是否把 dtype 提升到 storage 同级”：容量单位已经在那里；异构值的数值类型不能再折叠成一个父类型。应取消 view 的解释职责，而不是把一份异构格式表换一层缩进。

### 六 root 的每类 override

| 项目 | 数量 | 能否从当前绑定名直接唯一取得 | 删除字段之后保留什么 |
| --- | ---: | --- | --- |
| GDN `MergeScratch.Y` | 1 | 可以，math `Y:bf16` | Y 的 BF16 数值定义；带 padding 的物理 span |
| GDN `Stages.Kc/Qc/Vc/Ti/Qd` | 5 | 可以，均有 BF16 定义 | 值绑定；Ti/Qd 的十 tile pack 及各自容量 |
| GDN `Stages.b/bc/Ai/Dl`、`FinalState` | 5 | 可以，四向量和输出均为 FP32 | 每个值的份数／span；FinalState 的复用条件 |
| MLA `ScoreProbability.P` | 1 | 可以，math 明确 E4M3FN | 同版本 S/P 共享；P 的转换已在公式里 |
| Sparse `Gather` | 1 | 可以，math 明确 BF16 | 与 QK/PV 读取相同 gather 版本的绑定 |
| Sparse `Q_ring`、`PartialStage` | 2 | 不能只按名字；动作分别绑定 Q、Acc/PartO 后可以 | 短而明确的物化绑定；DQ 切片／full-D stride 的物理占用 |
| Sparse `RequestBounds/RequestLengths` | 2 | 不是已登记同名数学值 | 当前 metadata 动作明确的 int32 输入副本／组合；与 RowMax/RowSum 的复用 |
| FMHA `Prob8` | 1 | 可以，math 明确 E4M3FN | 数值定义和 TMEM P operand 编码契约 |
| FMHA `QueryScale/KeyScale/ValueScale` | 3 | 本地 copy/pack 名，不是上游同名对象 | 明确绑定 sf_q/sf_k/sf_v 的 E8M0 code；硬件 scale placement |
| FMHA `UnitScale` | 1 | 数学常数 1 不唯一决定存储编码 | 选定 E8M0 primitive 的单位编码 `0x7f`，不是再写一次 uint8 |
| MoE 两 phase 的 `ScaleTM` | 2 | 本地聚合 pack，不是一个上游 tensor | AScale/BScale 的 E4M3 codes 与 tcgen05.cp/MMA 表示契约 |

**14/24** 是与已登记同名上游对象完全一致的重复 dtype；没有发现这 14 项与上游冲突。其余 10 项没有统一的“按名字猜”规则：其中有别名缺少结构化绑定、本地 tuple，以及硬件编码。它们不能用作“数学值 dtype 重复不是问题”的理由，也不能假装已经全部由一个上游同名 symbol 定义。

但进一步审查后，这 10 项也**不要求 view 有格式字段**：定义物化对象／生产协议即可，而且现在不少格式事实已经明确写在生产动作和 realization 中。删除 `view.dtype` 不等于删除这些已有事实。

## 七组较强证据

### E1 — MLA：S/P 混合数值类型，不构成 view.dtype 的必要性

[root 的数值定义](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/root.yaml:188)明确 S 是 FP32，P 是经过 E4M3 舍入的值；[存储条目](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/root.yaml:387)再重复 `P.dtype:fp8_e4m3fn`。每槽是 128×128 FP32，P 占每行 32 个 FP32 columns，即每行 128 个 FP8 bytes。[源码列图](/home/shaohanyong.shy/KernelSchema/schema_discussion/phase_grammar/sources/kernel/extract_mla/solution.py:464)与实际写回一致。

这里有三件不同的事实：S 是 64 KiB，P 的有效 FP8 payload 是 16 KiB；二者属于相同的三槽 backing；P 的实际起点是每行 column 32。第一件需要类型／表示，第二件需要共享与槽映射，第三件是具体实现位置。

**定向反证后，撤回将本例改为无 part 整槽 reservation 的建议。** [pipe](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/root.yaml:493)中 `produce:Final` 在 `publish:P` 前，以及[源码](/home/shaohanyong.shy/KernelSchema/schema_discussion/phase_grammar/sources/kernel/extract_mla/solution.py:1078)先 tiny 重读后本 warp P 写回，都只提供相应成员的顺序，不能推出所有 Softmax warp 的 S 读取已经结束。此前据此写成“P 写入时没有本轮仍需保留的 S 子区”过强。

源码里的同步对象不能补出这一全组边：[初始化](/home/shaohanyong.shy/KernelSchema/schema_discussion/phase_grammar/sources/kernel/extract_mla/solution.py:805)使 `s_free`、`p_ready` 都需要四次 arrive，但各 warp 读完两个 half 后只是[通知 s_free](/home/shaohanyong.shy/KernelSchema/schema_discussion/phase_grammar/sources/kernel/extract_mla/solution.py:918)，P 写者没有等待它；这个 arrive 还早于可选的 tiny 重读，也不能单独充当全部必要 S 读取的退休事件。[MMA 发行者](/home/shaohanyong.shy/KernelSchema/schema_discussion/phase_grammar/sources/kernel/extract_mla/solution.py:1273)才在后续 QK 覆盖时等待 s_free，并在读取 P 前等待 p_ready；[p_ready 的 arrive](/home/shaohanyong.shy/KernelSchema/schema_discussion/phase_grammar/sources/kernel/extract_mla/solution.py:1090)发生在 P store/wait 之后，不能倒过来保护写前的 S 读者。`o_done` 保护上一 worktile 的输出／final scratch，`pv_done` 保护旧 P 的消费，都不是本轮 S 全组读完。

一个允许的源码级交错是：首次工作、T=0 且至少两个 KV tile；QK(0) 已完成，warp 0 读完自己的两个 half 后计算并写 P，warp 1 仍停在第一次 S load 之前。T=0 没有旧 PV 等待，且这不是 final 分支；没有源码边禁止这个交错。在 YAML 声明的“每 warp 独占自己的连续 32 行”设计下，这本来可以靠各行读写范围互不冲突来重叠。若无 part 的 reservation 被解释为 **128 行整个 S 都退休之后才允许任何 P 写入**，就会新增一个全组等待／更强的执行约束；它不是等价的书写缩减。

实际源码另有已记录的 row-address 限制：[S 读取使用 `tb=tb0`，P 写地址却加入 `warp<<21`](/home/shaohanyong.shy/KernelSchema/schema_discussion/phase_grammar/sources/kernel/extract_mla/solution.py:852)。因此不能连“每 warp 的实际 S 地址各不相同”也未经证明就假定成立；相同 load 地址使潜在读者集合与期望的行分工不一致。现有 `[32,64)` 也不能单独修复这个来源问题，不能由这个缺陷反推可放宽 P 的位置。

如果“整槽”只表示允许放入的容量范围而不建立全组互斥，新增等待可以避免，但所需的行映射、32-column P span、与仍活跃读者的相交关系就必须在别处明确保留；一句“共享这个槽”不够。本轮保守保留现有 `part:{column:[32,64]}`、上游行分工和源码限制，只删除 dtype 并合并共同槽公式。未来可以另证一个保持行对应、占用宽度、所有冲突读者退休与硬件编码的晚定位置方案；本轮没有证明列原点 32 永远不可替换，也不再声称可以直接把它省成整槽。

源码还包含 MODE1 的 FP16 P，使用 64 columns；但 `_PV_MODE=0` 是本 root 的选定 scope，不能拿未选中的 MODE1 充当当前 P 的独立存储精度选择。当前规则的 BF16 S/P 示例同理：上游若已经定义 BF16 P，重复字段仍无必要；本轮没有把该说明性例子冒充六 root 的实测 BF16 P 别名。

### E2 — GDN：所有数值 dtype 可知，packed 容量和真实覆盖不能省

[Stages](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml:571)的九类输入已经各有 BF16／FP32 定义，FinalState 也由输出接口定义 FP32。`uint8` 是这一 arena 的容量单位，不是第十二种数值选择。

**Ti 的逻辑 shape 不等于其物理占用。** 上游定义 64×64 BF16 Ti，但[冻结 common 源码](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_common.cuh:50)只连续存放十个 16×16 BF16 tile：`10*512 = 5120` 字节，而非 `64*64*2 = 8192`。两者 dtype 一样；不同的是选择／pack。`MergeScratch.Y` 也有 32×32 BF16 的逻辑值，源码按 32×40 BF16 行距存放，所需 span 是 2560 字节；父区域按 1280 FP32 单元预留 5120 字节。删除重复 dtype 合理，拿逻辑 shape×dtype 重算全部容量不合理。

stage 内九个输入子区同时存在，不能把所有 `part` 直接删成九个独占全槽的对象。至少要保留它们各自的占用、同时不重叠，以及共享同一代 stage 的关系；精确排列次序和各段原点可以由一次兼容的 pack 决定。批量传输连续性、1024-byte swizzle phase 等固定实现约束要保留为 representation 契约，而非静默放任任意重排。

最终状态是更强反例。[YAML](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml:651)已经保留最后 mini chunk 的 stage-0 竞争；[源码先读取 Kc row 0](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu:2499)，分支 `continue` 后没有加入结束这些读取的 CTA join，随后各 warp 就[把 R 写到 stage 0](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu:2819)。当最后 chunk 落 stage 0 时，某 warp 的输出写可以覆盖另一 warp 尚未完成的 Kc 读取。post-write barrier 不能补救。

这证明必须保留冲突区域与读完条件，并不证明必须提前规定偏移 0。VTT=64 时，一槽 52224 字节、final reservation 33280 字节；仅从范围关系看，把后者放在 `[16384,49664)` 可避开完整 Kc 区 `[0,16384)`，也不触及尾部 Ai/Dl。是否采用该位置仍要满足所有访问、对齐与表示约束；这是说明**约束允许晚定具体位置**的构造，不是本轮已修复或 GPU 验证的实现。

### E3 — MoE：uint8 ScaleTM 不是 E4M3 到整数的转换

[AScale/BScale](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/root.yaml:651)是 E4M3，`ScaleTM` 却写 `dtype:uint8`。若把每个 override 都视为独立数值格式，这会产生一次不存在的转换。

[生产动作](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/root.yaml:764)说的是 scale copy。[实际代码](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:4918)执行 `tcgen05.cp` 并以对应 scale 指针发 MMA；[表示说明](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:3012)明确每个 TMEM word 的字节还是 E4M3 code。uint8 只是 raw-byte 容器拼写。

必须保留的是特定的 TMEM scale placement、两套 scale 版本与单份持续 accumulator 不重叠、scale-copy/MMA 的顺序。每代 ScaleTM 预留 128 rows×32 FP32 columns = 16384 字节，而两个逻辑 128×16 E4M3 scale 矩阵只有 4096 payload bytes；硬件表示使逻辑元素数不足以替代容量账。`view.dtype:uint8` 既不能独自说明此差额，也不能说明哪个 byte 对应哪个 block，删掉它并不损失实际编码契约。

### E4 — Sparse：本地 metadata 格式需要定义，但不必归 view

[root.control](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/root.yaml:399)让 RequestBounds/RequestLengths 分时使用 RowMax/RowSum 的区域。源码清楚地[把 cu_q、topk、seq_lens 的 int32 words 写入两个未来的 FP32 scratch slab](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_sparse/dsv4_kernel.cu:964)。数学上的请求边界并不天然指定存储位宽；这里的类型来自具体被复制的输入和本地物化协议。

`tc_packed` 更清楚：[build 动作](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/tc_packed.yaml:363)已定义 8192-word uint64 HashTable 和 key/position 的单字发布；[源码](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_sparse/dsv4_kernel.cu:2346)以高低 32 位组合 key 和 position，使用 64 位 CAS/Exch。删除 view 里的 uint64 不会删除这些已有事实。不能把“union 是集合”推成 uint64，也没必要往 math_reshape 增加一个伪数学 HashTable 来配合类型推导。

同一分支的 `ScoreWork` 同形状 FP32 `Scores/E` 依次复用：[源码](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_sparse/dsv4_kernel.cu:3196)原位把 `s[j]` 改为指数。两条空 view 已足够，不需要字段。`alpha/inv` 也共享一个 slab；真正应保留的是其他仍存活区与它的独立关系，以及最后一次 PV／rescale 使用后才能换用途。把它们拆成持续状态与复用 scratch，可避免抄写 `3*MR` 的地址。

`tc_packed` 原稿指出的 protocol/union 错误别名仍是来源限制；本轮没有把这些错误地址升级成必须保留的设计约束。

### E5 — GEMM：不同 dtype、shape 的后用途可以独占同一 reservation

`mma_sync` 的 [InputRing](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gemm/mma_sync.yaml:158)每槽保存 A、B 和 FP32 a_scale，尾部把 F 转为 BF16 output 重用其空间。[源码常量](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gemm/fp8gg.cu:28)给出一槽 24832 字节，BF16 输出的带 padding span 是 `64*(128*2+16)=17408` 字节。源码 `Cs` 旁“overlays stage 0/1”的旧注释并不准确：该 span 放得进一个槽；判定依据应是尺寸和实际寻址。

[drain 与写回](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gemm/fp8gg.cu:263)先等完所有异步输入并 join，再形成 output；Schedule 在 ring 外继续存活。若声明 output 独占已死的某个槽，不必精确指定它的有效数据止于哪一个 byte。仍应检查 `17408 <= 24832`、所有输入读写已退休，以及下一 worktile 不早于 output source reads 结束。

反过来，GEMM root 的 DrainRegister 不能以“同一个 Fragment 值”删掉物理副本。[源码交替发下一读再做当前 FMA](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gemm/fp8gg_umma.cu:866)，[最后 read wait 后先归还 TMEM 再做 FMA](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gemm/fp8gg_umma.cu:912)。两个寄存器槽的占用、代次和完成边是设计事实，slot 的物理编号不是。

### E6 — FMHA：同时不重叠共存可分列，跨多区复用仍要一个共同账

[root.TensorMemory](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/root.yaml:495)里 ScoreHalf、四个 scale 区、NumState、Prob8 全部不重叠。QueryScale/UnitScale 跨全部 group 存活；NumState 也是持续状态；旧 PV 的 ValueScale/Prob8 读取需要在后续写之前退休。**只保留“七个值共享 TensorMemory”会丢掉这些可以同时活着的区域。**

但这些区域可以分列为互不重叠的设计 storage。总容量仍是 `(64+4*8+128+32)*128*4=131072` 字节；实现可一次申请 256 columns，未必要让设计填写 64、72、80、88、96、224 这些原点。[源码](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_fmha/source/candidate/fmha_tcg.cuh:1935)本身证明了“32 对齐但非 128 对齐”的 accumulator 原点也合法，具体选择有多个，不应把历史地址当数学要求。

`xpose` 分支相反：[OperandArena](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/tcg_d128_flat_tma_xpose_narrow.yaml:503)容纳一份 Q、两份 K、两份 V，然后整块给 FP32 PartA staging。81920-byte arena 容纳 67584-byte 的带行距结果，[代码的断言和地址计算](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_fmha/source/candidate/fmha_tcg.cuh:5225)明确跨越原先的多个 operand 子区。不能强制所有 backing 都变成同构 depth=2 的环，也不能拆成独立分配后把 PartA 再加一遍账。

这里可晚定 Q/K/V 的具体排列与 PartA 原点，但必须保留：五块在计算期并存、K/V 的两代轮转、PartA 在它们都结束后独占同一 arena、带 padding 的占用界限。空 split 缺少 pre-staging join 的来源问题继续保留，不能因抽象化而变成“必然安全”。

### E7 — 槽映射：很多公式可提升，有些差异其实是命名视角

GDN 的九类 `X(c)` 都用 `c%NSTAGE`，GEMM 的 Operands/A_pair/B_pair 都用 `n%3`；这些不需要在每个 view 重复。父 storage 可显式声明一次共同模板。GDN 的 FinalState 不带 c，选固定槽是例外；不能让它意外捕获某个 role 的当前 c。

MLA Geometry 的 [Item(w)→w%2、Next(w)→(w+1)%2](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/root.yaml:457)不证明有两种独立的物理算法。[源码](/home/shaohanyong.shy/KernelSchema/schema_discussion/phase_grammar/sources/kernel/extract_mla/solution.py:1447)按 publication epoch 选 `M.mi[epoch&1]`。可以只定义本地存放对象 `Descriptor(g)` 和统一 `g%2`，然后绑定：

| 原动作／访问名 | 对应存放对象 |
| --- | --- |
| latch:Item(w) | 读取 Geometry.Descriptor(w) |
| publish:Next(w) | 写入 Geometry.Descriptor(w+1) |

`Descriptor` 必须包括成功和终止描述符；最后的 Next 不一定产生一个新 worktile。这样动作索引 w 完全不必改变，也不必为对齐槽表达式硬改动作名称。其本地类型来自[十个 int 字段](/home/shaohanyong.shy/KernelSchema/schema_discussion/phase_grammar/sources/kernel/extract_mla/solution.py:493)，不是上游数学整数 Item 的 dtype。源码中的早期 descriptor reuse 问题仍需独立核对所有实际读者，不能由槽公式自行保证。

共同模板也不能扩展为“一律第一个参数取模”。GEMM `Fragment(n,d)` 按 d 轮转；`AS(t,p)` 按 p；MoE 的 APrefetch 按每 worktile 的 i，而 SMEM payload/scale 按跨 worktile 连续 g。[源码](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:4501)同时出现 `g%SFS` 和 `ks&1`。MLA 的 K/V 深度 3 或 4，与 S/P 的深度 3 还有[不同 notification 周期](/home/shaohanyong.shy/KernelSchema/schema_discussion/phase_grammar/sources/kernel/extract_mla/solution.py:1388)。选择哪个有序版本轴、是否跨 worktile 连续、什么时候允许覆盖必须显式；在一致重命名下把物理槽 0 换成槽 1 通常可以晚定。

## 保留信息与早期填写要求的判定

| 属性／事实 | 建议 | 必须早定 | 可以晚定／删除的内容 |
| --- | --- | --- | --- |
| backing 身份、location、份数／depth、容量界限 | 保留 | 共用账与独立副本，预算与目标重叠深度 | API 分配拆并、地址 |
| storage.dtype | 保留其容量单位职责 | 混合 arena／TMEM 物理轴的单位，不能含糊 | 普通同构 dense storage 可引用已有类型；不要再当所有 view 的访问类型 |
| view 值／本地对象绑定 | 非平凡时保留 | 哪个版本、哪个物理副本 | 同名单值全区沿用现有省略规则；空映射无须制造 |
| view.dtype | **删除整个字段** | 需要知道的格式仍有唯一来源 | 不再从父区域继承数值解释；不设稀疏的 dtype override 特例 |
| slot_index | 公共规则提升，局部例外保留能力 | 选哪个版本序号、周期、重置域、同槽冲突关系 | 同一公式的重复；一致的物理槽编号重命名 |
| part | 改成按需区域约束 | 同时活跃子区的 span、包含／相交／不相交；会覆盖谁 | 全部冲突读写已退休的整块复用时可省；保留 MLA 当前 part，不能从 per-warp 顺序推出全槽边界 |
| 物理 pack、padding、有效分配粒度 | 保留／引用 | 足以证明占用与选定传输／计算可实现的约束 | 具体 swizzle 与线程布局，若不改变这些约束则交实现 |
| 读完、写完、发布与复用顺序 | 保留 | 冲突字节上的所有读者和异步写者如何退休 | 不要求每个原语动作都独立一行，但不可由 alias 声明自动补出同步 |

这不是强迫所有实现都记录一个完整地址求解问题。优先让不重叠区域分列；让已证明整块退休的顺序复用使用整个 reservation；只有抽象后仍有交叉覆盖／共存的 arena，才需要结构化区域约束。多个成员各自执行同一份 pipe，并不自动形成整个区域的共同生命周期边界；MLA 是这个区别的反例，GEMM 的 wait-all 加 CTA join 才提供了相应的整块退休依据。

### 同一案例删项会怎样

| 操作 | MLA S/P | GDN Stages |
| --- | --- | --- |
| 删除 view.dtype，并取消 view 的数值解释职责 | 不丢信息；S/P 格式和转换在 math | 11 项格式全部有上游定义；Ti pack 另保留 |
| 只删 dtype，但仍保留“默认继承父 dtype” | 错误：P 被当作 FP32 | 错误：BF16/FP32 输入被当作 uint8 数值 |
| 删除精确 part，改成整槽 reservation | 本轮不成立：严格全槽互斥增加跨 warp 约束；仅作容量容器则未保留必要行／列别名关系 | 九类输入同时存在，不能各自独占全槽；需保留分区或 typed aggregate 的 disjoint-pack 契约 |
| 只留“这些值共享” | 不够：还需三槽代次、P 子区、行读写范围及实际退休保证；现有 row-address 限制不能被抹掉 | 不够：丢失输入间不重叠、final overlay、活动 Kc 的冲突 |

## 三个最小改写对照

以下是**同一套建议语义的候选片段**，不是宣称当前规则已经允许 parent `slot_index` 或新的省略语义。旧语料字段已按当前名字展示；没有把计算／同步事实移进长 describe。

**1. MLA：删除独立格式，公共槽规则只写一次，保留 part。** 上游 S/P 定义、三槽容量、行分工、各动作的寄存器／TMEM 绑定、全部复用依赖和来源限制保留；这份草图不增加全组 S 退休假设。

```yaml
# 当前规则的表达
ScoreProbability:
  depth: 3
  location: TMEM
  shape: [row: 128, column: 128]
  dtype: fp32
  total_size: 196608
  view:
    "S(n)": {slot_index: "n%3"}
    "P(n)": {slot_index: "n%3", part: {column: [32, 64]}, dtype: fp8_e4m3fn}
```

```yaml
# 建议：view 绑定字节区域；不承担数值解释
ScoreProbability:
  depth: 3
  location: TMEM
  shape: [row: 128, column: 128]
  dtype: fp32
  total_size: 196608
  slot_index: "n%3"
  view:
    "S(n)": {}
    "P(n)": {part: {column: [32, 64]}}
```

parent 表达式是显式的形式参数模板：其自由形式参数必须在每个继承它的 view 键中被绑定；实参仍由本次访问提供。缺少形式参数的 view 必须显式给例外映射，不能捕获 role 局部变量。`n%3` 没有从 depth 推出来。P 的 FP8 类型来自 P 本身；part 继续选父 FP32 坐标下每行的 32 columns，避免把区域约束的删除混进 dtype 的删除。

**2. MoE/direct：小值顺序使用大区域，不必标出尾界。** 当前的 `[0,K)` 只是 FinalRoutes 在已死 WarpCounts 空间里的具体占用；[生产顺序](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/direct.yaml:370)已明确先消费 census 的 WarpCounts。

```yaml
# 当前
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
# 建议：两种用途先后独占同一个 64-byte reservation
Exchange:
  depth: 1
  location: SMEM
  shape: [element: 16]
  dtype: int32
  total_size: 64
  view: {WarpCounts: {}, FinalRoutes: {}}
```

FinalRoutes 的有效 K、局部 int32 route index 格式和发布／读取条件仍在既有生产协议中。若后来要让 Exchange 的剩余部分同时保存别的活值，必须重新显式划分子区；不能继续使用这份“独占整个 reservation”的抽象。

**3. FMHA：把不重叠设计区域分列，免去 API arena 的偏移表。** 左侧为当前 TensorMemory 内的区域宽度和原点；右侧采用同一容量单位，storage 条目之间默认不重叠。

| 对象 | 当前 part columns | 建议：一个成员的一槽容量 |
| --- | --- | --- |
| ScoreHalf | [0,64) | 128×64 FP32 capacity cells |
| QueryScale | [64,72) | 128×8 FP32 capacity cells |
| KeyScale | [72,80) | 128×8 FP32 capacity cells |
| UnitScale | [80,88) | 128×8 FP32 capacity cells |
| ValueScale | [88,96) | 128×8 FP32 capacity cells |
| NumState | [96,224) | 128×128 FP32 capacity cells |
| Prob8 | [224,256) | 128×32 FP32 capacity cells |

每项均 `depth:1, location:TMEM`，无 view；容量之和仍是 256 columns。数值／raw-code 格式由原对象和 scale/P primitive 给出，完全不从该表的 FP32 容量单位读取。realization 约束它们放进一次合法的 256-column allocation，实际原点由 Coder 定；选定的 row/scale 表示仍要一致。若固定实现提取要求复现原列图，则在 realization 引用那份列图一次。

## 统一的替代来源与边界

删除字段后采用一条一致规则：**view 不产生格式；每次访问必须绑定一个有定义的值或本地存放对象，并能追到唯一的数值／表示来源。**

1. 上游有该值：解析当前 Design 的 dtype／ref／明确转换；输入输出按其 Contract 格式。不能用 Contract 参考中间量覆盖 Design 的改写精度。
2. 只是切片、副本或无损 byte packing：明确指向原值与一个表示契约。布局可以变，值的数值格式不会因为父 arena 是 uint8 或 FP32 而改变。
3. 本地 descriptor、混合 tuple、atomic word、硬件 scale image：在生产它的动作或被引用的 realization 中定义一次字段／code 含义、格式和 footprint。其他读写绑定这一对象；不得各自猜测。Sparse HashTable 现有 build 描述、MoE 现有 scale-copy 契约就是可直接沿用的实例，不需要发明一个新上游数学 symbol。
4. 格式改变了数值：必须由定义过的转换／量化路径承担。view 删除或区域重解释不会自动执行转换。

对局部格式只在 prose 中定义的情况，现有人工 Reviewer 可以核对，纯结构检查器不能声称已自动完成类型验证。这是当前 noun/describe 模型已有的能力边界，不是需要恢复 view.dtype 的证据；一个孤立 dtype 字段也验证不了 key/position packing 或 E8M0 code。若未来要求机械化验证，应结构化**唯一的存放对象／编码契约**，而不是让同一格式在上游、view 和每个消费者各维护一份。

part 晚定也采用明确义务：子区必须容纳已知物理 extent／上界；同时存活的独立值不重叠，除非已有同 bits 别名定义；重叠写必须在所有冲突读取和未完成写入之后，证明范围包括其他 warp／成员，不能把本成员的顺序提升成全组顺序；部分保留必须指出仍存活的子区；容量每块 backing 只记一次。Coder 可以选择满足这些条件的偏移、排列和 swizzle，不能借“晚定”改变 buffer 深度、副本数量、设计重叠窗口或突破容量预算。若放宽区域后只能通过新增全组 join 才安全，就已改变执行约束，不能称为单纯布局晚定。

这套规则支持：普通单值不写 view；同类型同形状顺序复用写最小绑定；不同 shape/dtype 整块复用保留容量与顺序；同时不重叠共存优先分列；跨区 arena 和部分保留用明确区域关系；同构环共用显式轮转模板并保留例外。它减少的是不必要的早期决定，同时保留审查实际内存与流水线设计所需的信息。
