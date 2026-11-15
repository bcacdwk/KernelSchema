# M06：NVFP4 fused MoE 提取试走草稿

本稿是一次只读源码、手工推演的 EXTRACT_SCHEMA worker 试走。只做到 P1 事实摘要与 P2 机制/代表判断，并试写极短依赖片段；没有完成或通过任何阶段，没有完整 YAML、正式 dispatch 或数值保证。未执行 Python、extract.py、checker、GPU、构建或测试。没有阅读本算子旧 schema Design/dispatch、其他试走报告或以往 schema_discussion 研究。

## 范围和证据身份

根目录为 `/home/shaohanyong.shy/KernelSchema`。以下 `O/` 缩写指 `kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/`；`C` 指 `O/candidate/nvfp4_moe.cu`，`S` 指 `O/candidate/solution.py`。行号均指本次工作树实际文件。

已读取 AGENTS.md、EXTRACT_SCHEMA.md、P1/P2 阶段页、EXECUTOR_ALIGNMENT.md；定向读取 common 的参数/局部/整体判断、DOC1 的 Contract/验收、DOC4 提交/交接、DOC5 动作/依赖/跨工作索引及 REVIEW 的保留降级与缺项规定。P3 阶段页作导航，starter 仅核对占位与已知旧结构；没有据旧 starter 发明表达方式。

本次 `git rev-parse HEAD` 为 `0c662ebcf2cb57d6b417c12498c3adbf39071a44`。对 problem 及两个主源码的定向 `git status --short` 无输出。主证据 SHA256：

| 文件 | SHA256 |
| --- | --- |
| S | `0a71613fe4beec9314570cf128f2d455e91adf7182467c261a36ac11e45f0688` |
| C | `cf0b42e2e4809cf71b2442f1b64285da4cc0a69214f43eb82efbdac938978f6f` |
| O/problem/reference.py | `4fccbab4f84e063d70fc019d4ba77a63f3464b41fd5c5965aa2eb586e8289833` |

主要入口是 S 的 `Model.forward`，其构建 `_SRC=nvfp4_moe.cu`，通过 ABI 调用 C 的 `moe_run`。CFAST 默认可启用：本次仅用 `nvfp4_fast.c` 的定位搜索核对它读取 `_hot`、检查身份、调用同一 `moe_run` 与 fallback 的线索，未全文审计 C 扩展。备用 `solution_*` 和 `nvfp4_moe_*` 文件不是 S 的 `_SRC`，没有据文件数量扩大调查。

## P1：能核实的任务事实

外部任务是给定已选路由和已准备 NVFP4 权重的局部 MoE 输出，不含 router top-k 的计算。输入构造 `input.py:101–126`、reference 全文、metadata 的 benchmark_contract、shape_range 全文、shape_cases 的实例及 baseline 入口已经读取。`problem/solution.py` 是 FlashInfer 原库 baseline；`problem/reference.py::Model.forward/expected` 才是独立参考入口，不能互换。

令 T 为 token 数，K=8，H 为 hidden，I 为本地 intermediate，E 为本地 expert 数。已声明的相关几何为 `(H,I,E,expert_tp)`：`(2048,768,128,1)`、`(4096,768,128,2)`、`(4096,1536,128,1)`、`(7168,256,256,8)`；T 为 1…8192。shape_cases 的 10 个样本不是完整域。路由可 balanced/hotspot/skewed/empty，可含 masked route，生成器把 invalid ID 写为 -1 且对应 score 为 0。

关键对象：IDs 是 int32 `[T,K]`，scores 是 FP32 `[T,K]`；hidden payload 是 uint8 `[T,H/2]`，每字节两枚 E2M1；hidden scale 为 E4M3FN `[T,H/16]`。FC1 payload `[E,2I,H/2]`、scale `[E,2I,H/16]`，FC2 payload `[E,H,I/2]`、scale `[E,H,I/16]`；权重实际存储已经含行排列与 scale swizzle。准备与逆解码分别见 input.py:82–99 和 reference.py 的 `weight`，不能按原始连续逻辑行直接解释权重字节。

外部计算关系（此处不是完整 Contract 字段稿）：

1. 解码 hidden/weights 为 E2M1 值乘块 16 的 E4M3 scale；逻辑 FC1 每个有效 `(token,slot,expert)` 得 up/gate 两个 I 维结果，参考用 FP32 收缩，TF32 关闭。
2. 可选 limit 时，up 截到 `[-limit,+limit]`，gate 只上截到 limit。activation_type=10 时为 `alpha*tanh(gate/alpha)*sigmoid(gate) * beta*tanh(up/beta)`；否则为 `gate*sigmoid(alpha*gate)*(up+beta)`。
3. 每条 route、连续 16 个 intermediate 值独立 NVFP4 QDQ：`sf=E4M3FN(clamp(max(abs(x))/6,0,448))`；sf=0 输出零，否则 E2M1 就近偶数舍入并饱和到 ±6，解码回 `q*sf`。norm_const=1。此边界是数学要求，不要求特定 CTA 或 kernel 数。
4. FC2 FP32 收缩后，每条 route 先 BF16 舍入，再转 FP32 乘路由 score 并求和；最后输出 BF16 `[T,H]`。没有有效 route 的 token 输出零。参考按 expert 遍历并 index_add，candidate 常按 slot 顺序 FMA，不能把相同数学误写成已证 bit-exact。
5. `output` 必须由调用方给出；原位写入并以 `[output]` 返回，身份/别名属于要求。`input.py:114–126` 只声明 output 为可写、可观察、replay 恢复对象；guarded backing 的边界区域不在许可写区内。workspace 不是外部输出。TP 分片后的结果需要调用方在本 callable 外合并。

本次未逐项正式转录 37 项签名。已核实生成器中的 GEMM biases、routing_bias、LoRA 等为空，三个 output scale 为 1，routed_scaling_factor=1、do_finalize=True。注意初读 shape_range 的 activation_alpha/beta 不能直接变成 alpha/beta 张量值域：input.py:111–112 在 situ 模式实际构造 alpha=4、beta=25；普通分支是 alpha 默认1、beta默认0。这是生成参数和外部张量的区别，已通过回读解决。

验收材料存在两层声明，不能只抄一个 atol/rtol：metadata.description 给 `torch.allclose(reference_float,candidate_float,atol=.01,rtol=.05)`；benchmark_contract.default 同阈值且 max_mismatch_fraction=0；`mutated_inputs.output` 给 max_mismatch_fraction=.001、max_rel_l2=.01。这里 reference 是 allclose 的第一个实参，不能套 DOC1 默认把 reference 当相对基准。提供材料未给出完整 runner 比较/路由实现，尤其返回列表和 mutated output 分别如何采用这些规则、rel_l2 归一化和非有限政策，尚未核实。应在相应 tolerance 保留来源记录和限制；不是删掉已知声明，也不妨碍当前机制调查。本稿未声称还原了外部验收器。

## 已读的实现主线

| 定位 | 已核实内容和用途 |
| --- | --- |
| S:1023–1180 | forward、身份缓存、shape 检查、连续化、ABI 参数、实际 E/H/I 推导。E>256、H%128 或 I%64 等被拒绝；这不缩小共享 Contract。 |
| S:506–875、1180–1505 | 默认 cfg、cfg2、有效性/表示兼容检查，route_hist host readback，MEGA/zero/小路径门槛。这里混有历史注释，以实际 return/if 为准。 |
| S:918–1004、1507–1575 | 路由/act/rout/partial/acc32 等 workspace 分配与 ABI，zero pinned-host flag 和 keepalive。 |
| C:9078–9364 | run_pipeline 的实际 launch、归并责任、特殊单核出口及 side stream memset。 |
| C:4190–4510、4622–4705、4838–4995 | 分工式 FP4 GEMM、实际模板形参、持久 item 序列、环缓冲及跨 item 协作；未逐行审完所有 epilogue 模式。 |
| C:7730–7820、8330–8359 | cfg97/99 与 cfg81 对应的真实 kernel 实例，避免仅据旧注释分类。 |
| C:1352–1427、1464 起及 finalize 定位 | split-K 固定 slice 顺序 FP32 相加、activation/QDQ，route buffer 与 finalizer 的关系；未完成所有向量化 finalizer 的逐行审阅。 |
| C:1899–1990、2160–2268 | MEGA 单核 direct/staged 主体、32-wide I chunk、跨 CTA 累加、最后到达者写 output 并自清理。 |
| C:1707–1835、9450–9618 | zero scan/verify 的当前字节扫描、输出清零、ticket/flag、真实双图 replay 与 capture 行为。 |
| C:180–216、6774–6818、6916–6934 | PDL trigger 只放行后继启动，wait 才给此前工作完成/可见性；默认启用且 R>=49152 关闭属性。 |

### 主机制理解

普通主链先按 expert 统计有效 route、形成 offsets、scatter 得 `ptok/pslot/pexp/route_of` 和 M tile map，再做 FC1、activation/QDQ、FC2、按 token/slot 归并。大准备路径含 memset/count/scan_tiles/scatter，小准备路径用一个 CTA 完成同类 map。单有“launch 少了”不能决定新设计身份。

cfg>=7 的主要族直接用 packed FP4 payload+scale 供 block-scaled MMA；act 也是 packed payload+scale。cfg<7 存在先 decode hidden、BF16 中间量和不同 GEMM 路径；本次只核实了入口与表示族边界，没有把这些备用主循环全部归为参数。

分工式 FP4 主循环：一个 CTA 是 worker，7 个 warp，0 做 A scale，1 做 payload 和 B scale，2 做 MMA，3–6 做 epilogue。item 是 `(M tile,N tile)` 或 `(M tile,K slice,N tile)`；每 CTA 从线性 CTA index 开始按 grid 大小 stride 处理多个 item。它是预分配循环，不是运行时队列补领。`g` 跨 item 连续递增，payload 三槽、scale 两槽。DEEP 把 payload 复用等待放到 g-3，scale 保持 g-2；MMA stage commit 与 epilogue done 分别保护 operand 槽和累加器复用。不能在每个 item 尾部人为排空整个流水线。

S 中实际表示族检查（1367 附近）防止 FP4 act 与 BF16 act 不兼容配对。cfg97 FC1 融合 activation/QDQ，cfg99 则 SK=2 写 FP32 partial，独立 k_act_packed 才相加并量化。分片不仅改 grid 数，改变了主要中间结果、写出者和归并。

### 拟代表与记录去处

所有文件名都只是计划，没有相应 YAML 或可引用的完整基线。以下是暂定选择，不是 P2 放行。

| 候选 | 真实实例与机制 | 暂定去处、保留理由/疑点 |
| --- | --- | --- |
| packed WS 融合主链 | 拟 root：cfg97/81，SK=1；用 fixed_3 几何 T512/H4096/I1536/E128/K8、skewed、普通非零输入作为真实可达说明。R4096，非空时 small_thresh=3000，走大准备；R>1536 选97；G2下界 `ceil(4096/128)*(4096/128)*(1536/256)=6144` 已过3072，选81；pre-gather，FC1融合act/QDQ，G2 TSTORE 到 BF16 rout，再 finalizer。 | 拟 `root.yaml`。主要 packed 表示、持久协作和输出接口明确，便于对照。root 的正式 when/参数须在 P3 固定：例如该范围 CPHOIST 的 R>=16E 成立，不能把门槛开关继续留在单份正文里。实际非零/成功 TMA 等前提仍应写实，未模拟生成某个 seed。 |
| packed WS split-K 主链 | cfg99/44；fixed_1 几何 T32/H2048/I768/E128/K8 hotspot 是可达例。R256、SK2、AMODE3 clipped gather4 不预物化 hidden；route_hist 约8个 M tile，G2 gate选44；FC1 partial→独立act/QDQ→G2→finalizer。 | 拟 `split_k.yaml`。新 K-slice 工作轴、全局 partial、固定 slice 合并和额外消费者需要贯通解释。与 root 同时改变了 G2 配对，不意味着每种 G2 配对都要完整文件；其余配对拟组合记录，须再核实接口。 |
| MEGA direct | cfg92；S:1417–1422 对当前 empty histogram 且满足范围默认选中。C:9099–9136 实际只提交一个 kernel；每 unit 为 raw route×I/32，CTA 连续切块，SIMT FP32 计算，route acc32 原子累加，token 最后 ticket 的 CTA 完成 BF16 route边界/score求和/output，自清理acc32/cnt；无有效路由时 pass0清零后退出。 | 拟 `mega_direct.yaml`。输出生产者、工作切分和跨工作状态完全不同，保留有明确独立价值。不能只写“空输入清零”；cached replay 变非空会走同一 kernel 的实际计算路径。源码历史说明写I/64，实际 `S=p.I>>5` 是32，本稿取后者。 |
| 零值验证与双图 host 决策 | 已经接受的 zero probe 创建 zflag，T*H>=2^20 时默认 ZERO2 再有 pinned host mirror；graph hit先 replay verify-only clean graph，host bounded spin，dirty/timeout再提交 full graph；outer capture记录两图，实际执行在以后 replay。 | 是主要提交/输出机制候选，拟完整保留价值高，但当前无法按 DOC4 忠实写出单份完整 launch 结构；受阻，见下一节。不是已降为参数。 |
| REDF | cfg2=46/47，C:9163起 side-stream清零 `[T,H]` acc32，G2负责加权原子归并、k_convert最终舍入，替代 `[R,H]` BF16 rout+finalizer。 | 尚不能判断局部记录是否足够。已知新增输出归并/初始化/完成关系，倾向独立候选；但未读其具体 epilogue，尚未核实 route BF16边界在atomic前的位置，不能据注释保留或降级。下一步应只追该epilogue，不枚举所有cfg配对。 |
| 单体 FP4、BF16/warp MMA 族 | C 的 k_gemm/k_gemm_tc/f4_body 与 S 的表示族/override/fallback有实际入口。 | 来源范围内的已发现线索，未完整研究，暂未纳入拟完整集合，也未假称普通参数。需确认是否仅保留“未完整展开独立方案”足够，尤其单体与角色分工改变主协作。 |

可合理降级的线索：DEEP 的复用等待改变（相对相同角色/存储基线）；同机制的 payload 槽深、tile/向量宽度及联动；NPAIR 的两枚 n64 改一枚 n128 及 scale image/TMEM 容量联动；普通 graph 缓存与 CFAST host 调用优化。后两者分别是局部/来源行为候选，不是仅凭 bit-exact 就降级。NPAIR 的实际值、操作数及资源差额仍需记录。

有保留条件的局部候选：预 gather vs clipped gather4 可围绕 hidden route view 的物化、生产者消失与 GEMM load 改动描述；但必须同时保存 padding rows不取/不写、transaction byte计数和后继边变化。小/大 route准备可以围绕同一 map接口记录，但还没完成每个map值的写入/初始化/PDL出口核查，不能把这个暂定直接当批准。

CPHOIST 初读时我想把 `if Rmax>=16E` 当普通运行时执行条件留在 root；重新读 common:108–124 后改为：它选择 scale搬运相对epilogue退役的策略，是配置差额。root选实际 hoist-on 子域，hoist-off另记局部变化。与 Clean 等本次验证状态分支区别在“决定什么”，不在 runtime/compile-time。

## 极短表达试走和一个未解决的表达阻碍

### 可表达的普通局部依赖

以 cfg97 的 GEMM1 为例。上游已经定义同一 CTA worker 的预分配 item 序列，K stage数 `Nk=H/256`；payload三槽、scale两槽。g是跨item连续序号，不是全局item编号。下段仅展开 ProducerB 的两个覆盖等待，不是完整 pipe：缺少的role、动作和storage在此用文字定义，不据此宣称完整schema。

```yaml
index:
  t: {meaning: Local worktile ordinal, definition: worktile}
  i: {meaning: K stage within a worktile, definition: [0, Nk]}
  g: {meaning: Continuous stage ordinal across worktiles, definition: {flatten: [t, i]}}
# 位于 ProducerB 的 waiting；两个 load 分别写 payload(g%3) 与 B scale(g%2)。
waiting:
  released:
    - "{issue load:Payload(g+3)} ← [{done MMA::matmul:A(g)@B(g)}]"
    - "{issue load:BScale(g+2)} ← [{done MMA::matmul:A(g)@B(g)}]"
```

这里 `MMA::matmul:A(g)@B(g)` 汇总该stage的4枚NPAIR MMA，done取 `mma_commit(s_empty+g%3)` 完成保证，须包含此前该warp的scale读取退役；不是“发射就可覆盖”。两个load是独立完成机制，不能只因都在ProducerB就使用跨noun累积完成；完整稿仍须写 prepared 和 epilogue/accumulator 等边。源码 C:4628–4659、4950–4968 明确支持两种复用距离。最初想用每item局部i+3，但会丢跨item环接续；DOC5的flatten表达解决了这一点。

### ZERO2 的实际提交无法直接套当前字段

实际 C:9471–9491：先真实执行 clean graph；只有 host观察值不是1（含timeout）才再 `cudaGraphLaunch(full)`。full自身含另一次 verify，故dirty/timeout路径不能复用同一 step ID；C:9473–9485 的outer capture是仅记录两个图，不能当场算两次计算已执行。

最初尝试以下固定条目，然后发现它说错：

```yaml
flow:
  launch:
    - {step: [verify_clean]}
    - {step: [verify_full], start_after: verify_clean.kernel_done}
    # 再列 full 链的实际 kernel；此处故意省略。
```

问题不是缺一个后继等待：clean verdict=1时后面根本不提交。DOC4:243–259说每个step恰好出现一次、列表枚举提交；DOC4:395说“说明字符串也不提供条件删边或替换生产者的语义”。给verify_full写“仅dirty/timeout”会违反这一点。把full各kernel提交后空工作，与实际host不提交也不同。把graph capture计作执行则违反同节“录图只记录执行结构”。

common:112明确允许“根据本次产生的Clean决定必要动作”留在当前方案，但 DOC5 的if只控制worker内动作，没有看到它可取消host的整个kernel提交。把host虚构为CUDA worker/role，或新增 launch.when/guard 字段，均无依据，本稿没有这样做。

一种可能是分为clean-only/dirty/full/outer-replay数份设计并用when写条件，但host timeout不能由输入域预先决定，这会把同一机制的运行结果分支当不同完整配置，还没有说明谁承接本次host决策。因此我目前不能确认这是合规解法。可以先继续普通root，不能据此宣布ZERO2已完整表达或已找到全算子的调查止点。

## 明确的未决边界和停止位置

目前拟具体完整集合只有普通root、split-K、MEGA三个较有据候选，加上受阻ZERO2；RED F与其他主循环保留为待判断，不把总数凑到10或先承诺全展开。普通root的代表依据已经相对稳固，但若要求整个S可达实现范围的P2草案完整，尚不满足“没有动摇选择的重要疑点”。

下一轮最有价值的定向读取是 REDF 的G2 epilogue/BF16边界、单体FP4到WS的主要协作差异，以及ZERO2怎样容纳于现行提交表达。普通参数全表、所有override pair和全部备用文件没有现在展开的理由。

本稿还没有形成全部IO正式字段、全部tolerance路由、每个候选完整math/schedule/pipe，也没有核实所有源失败前缀的在途副作用。源码的历史性能/bit-exact注释仅作线索，未当成本次执行证据。硬件资料和GPU验收按提取豁免跳过，既未推断硬件保证，也不把它们列为待办。
