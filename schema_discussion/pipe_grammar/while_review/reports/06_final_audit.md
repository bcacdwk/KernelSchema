# 06：while 核心资格的最终逆向审计

**单一建议：从当前核心 pipe 移出 while，核心保留有序列表、for、if；保存现有 while 语义和本轮反例，作为重审依据。** 这是针对目前已落地工作负载的能力范围选择，不是“良好算子不该有 while”的定理，也不要求修改源码里的 while/break。本报告是研究建议；当前 README 的 for/if/while 规则尚未因此变更。

本审计直接回查原始源码、基线职责、当前控制/动作/调度规则，并重新运行 04 的 CPU 见证；不按报告人数投票。已读入 01–05 的报告与对应数据，并对新增 07 的 SSSP 证明、源码与模型做了聚焦复核；最终建议以原材料和明确的表达条件为依据。全部证据锚点、指纹和结论类型记于 [06_final_audit.json](../data/06_final_audit.json)。

**可以做出的结论与不能做出的结论。**

| 可以做出的结论 | 不能随之推出的结论 |
| --- | --- |
| 十个 root 的已审执行路径有来自输入、既有调度或动作职责的自然解释，没有发现必须新增 pipe while 的机制。 | 十个 trial 已与源码逐事件完全等价；所有分支、193 份旧稿或一般 GPU 算法都已证明不用 while。 |
| CG/BFS/SSSP 有真实跨动作反馈，已审来源在明确前提下又有真实 cap 或自然有限轮界，可以用 for/if 保留它。 | 反馈不存在、算法设计有漏洞，或“有上界就能机械改写”。 |
| Gamma 的局部采样、原语等待和 DMR/MIS 的 host 提交循环不直接要求这个层次的 pipe while。 | 一切反馈都能塞入 compute/await/work_schedule；这些算法已被当前核心完整覆盖。 |
| 本次真实材料尚未建立一个已闭合的、当前目标必须使用 while 的迁移案例；SSSP 的自然轮界也已在执行契约下补证。 | 旧 CUDA 的所有实际执行都已认证；没有自然上界的合理协议不存在；删掉 while 就得到终止性或静态完整展开。 |

**十例的负向证据不是“旧试写里没有 while”。** 下表是本审计重新回读的控制来源和边界。路径 T 指 `trial_roots/sources/`，E 指冻结提取 baseline；完整路径在 JSON。这里确认循环机制及其抽象层，不冒充第二次全源码/数值/并发证明。

| 路径 | 原始锚点与回查结果 | 当前 pipe 判断 |
| --- | --- | --- |
| 模板 FA4 prefill | T 的 hd256 kernel 1829、1903–1921；tile_scheduler 1303–1345；入口/选择器也已核对。root 的非 persistent 分支处理初始工作后置失效，KV 范围来自长度与 mask。 | 单份工作和有限 KV 遍历；不能把旁边的 persistent/CLC/sparse vendor 路径混成 root。 |
| 模板 FA4 decode | paged kernel 1104–1107、1651–1657、1707–1726；operator 379。次数来自长度/split；BLASST 即使启用也只选择当前 tile 的后续工作，root 传 None 禁用。 | 有限域加 if。try-wait token 不表示重新计算整轮。 |
| 模板 GDN | 可读 core 1530–1553、1627–1630，scheduler 237、263–270；压缩入口 782、812。工作 ID 按 resident CTA 步长推进，chunk 数来自请求长度。 | 状态递推依赖前块，不以该状态决定是否有下一块；padding 协议仍需保留。 |
| 模板 QSA | kernel 324–334；baseline work_schedule 149–193；preprocess 221–232。真实票据流和 Hash 内 CAS 探测是不同职责。 | for(t) 保留 first/next/terminal 票、确认与排空；Hash 探测留在既有 build 内，不伪造“成功轮次域”。 |
| 提取 GDN | CUDA 2058–2063、2227–2240、2319–2343。weak fallback 在进入第二遍前确定全请求或分段范围。 | Count/Run 已就绪后作有限遍历，不是反复求解直到 weak 消失。 |
| 提取 GEMM | UMMA 363–388。外层固定工作步长，组查找沿 prefix 递增，内层 PAIRS 来自 K。 | 有限工作与归约；组查找属于既有坐标解析，barrier poll 属于等待。 |
| 提取 MLA | solution 863–870、1458–1483；baseline 340–353。Geo 在 launch 前上传，Queue 发布成功描述符或负 tile-count terminal。 | 真实动态 for；不能把最后一次失败领取、terminal 发布或消费者读取删掉。 |
| 提取 Sparse | CUDA 1002–1025、1198–1206、2347–2380；packed Setup 契约 363–384。union 完成后 Nu 已定；同 key 的位置发布等待在既有构建动作内。 | 有限域、空路径和原语等待各自表达；8192 个槽不能充当位置等待的次数上限。 |
| 提取 FMHA | TCG 4462–4489；分段 scan 1423–1497。外层 group 范围有输入界；分段 Cut 在进入后续范围前算出。 | 数值影响后续读取量也不必 while；仍须写全部承诺的 snapshots 并保留先后。 |
| 提取 MoE | CUDA 4371–4384、4448–4461、4501；direct 2210–2256。root 为固定步长；direct 的最后到达票只决定一次 finalize。 | for+if 能保留独立贡献、到达、合并与清零；不存在失败后重做 Partial 的反馈。 |

01 的 21 个 family YAML 是 dispatch/work_schedule 范围检查；02 的六 root 加 14 个代表分支也不是全部提取树审查。十 root 的 while 结论应保留这个范围。尤其，控制外壳足够不等于已有 noun 契约和 waiting 都完整正确。

**广查增加了什么，没有增加什么。** 05 从 44 个 candidate 目录的 428 份代码筛选机制；其数据登记 29 个题目具有某种机制证据，其中 25 个不属于前六提取题目，分成 21 个机制记录。案例的审阅层级仍不同，不能说“44 题全部分支均证明消去”。这些目录实际集中在 GEMM 15、FMHA 8、linear attention 8、MoE 5、MLA 4、sparse attention 4 个题目；它们扩展了当前张量 kernel 的采样，不代表随机、几何、图算法等一般空间。

本审计另回查了 MegaMoE task schedule/top-k reduce、DeepGEMM grouped scheduler、conv-GDN producer/consumer 等待、paged-latent fused merge，以及 05 新补到的 FP8 sparse CLC。几个特别能约束结论的事实是：

- `dual_cache_latent_fp8` 在原 while/do 和字面 for(;;) 索引中均为零命中，但 [SM103 exact kernel](../../../../kernel_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_fp8/candidate/fp8_h8_h16_source_exact_sm_103a_kernel.cu) 214、1250 行确有带初始化和步进的无限 for。runtime 178–201 行可选这条路径；scheduler 1266 行提交 CLC，consumer 518–529 行及 scheduler 1321–1332 行均先归还响应槽再按 invalid 退出。下一工作是已提交 grid 中被接管的 CTA，attention 数值不生成新任务。动态 for 正当，但 terminal 响应与归还必须显式。
- [MegaMoE](../../../../kernel_wiki/aka-cuda-operators-20261030/operators/moe/bf16_fused_moe/candidate/kernel_helpers.cuh) 1436–1446、1623–1644 行已有 task descriptor、full/empty 协议和 sentinel；[top-k reduce](../../../../kernel_wiki/aka-cuda-operators-20261030/operators/moe/bf16_fused_moe/candidate/kernel.cuh) 1248–1308 行每次去掉既有 mask 的一个 bit，先预取下一项再累加当前项。两者都不能仅因写成 while 就当作本地无界求解，也不能合并成一个不透明动作。此处已回查 solution.py 登记的 normal kernel；kernel_situ 被登记或留有代码，不代表当前 dispatch 选择它。
- FP8 FMHA 的 host 1879–1888、1980–1989、2080–2108 行确实允许 MT 动态 pool，却只给 TC 建 static pool；static 不可行时回退普通 launch。MT loader 1375–1407 行领取与发布的对象来自 host 事先建立的输入几何工作表。不能从残留 TC dynamic 分支或源码注释推出它是当前选定路径。
- 944 条 while/do、18 条字面 for(;;)、补扫的 27 条空条件 for，以及 218 条 bra/goto 定位都不是语义机制数。原索引漏掉 CLC 已是现实的反证。PTX 的向前 DONE 分支和等待回跳也不能合并统计成反馈循环；05 已区分这些限制。通用多 rank 模板存在，不等于 single-rank 验收合同已经验证多 GPU。

**CG/BFS 的有界改写成立，但成立的对象要说准确。** 我直接回读了保存的 [CUDA cooperative CG](../sources/03_cuda_samples/cpp/4_CUDA_Libraries/conjugateGradientMultiBlockCG/conjugateGradientMultiBlockCG.cu) 277–340、469–528 行，以及 [BFS](../sources/03_galois/lonestar/analytics/gpu/bfs/bfs.cu) 305–361 行和真实 [Worklist](../sources/03_galois/libgpu/include/worklist.h) 223–266 行。

CG 的 k 从 1 开始，最多执行 10000 次 body；每轮方向发布、SpMV、dot scratch 复用和残差读取之间确有 grid sync。当前残差由同步后的显式读取缓存，不再执行 body 后它也不会自己变化。因此 `for(r=0:10000)` 内 `if(residual2 > tol2)` 可以保留同一个活跃动作前缀，不必新增 Alive。必须保留真实浮点类型、算术和原来严格比较，不能把“条件为 false”直接定义为成功：host 的成功标准是 `sqrt(r1) < tol`。**用满 cap 也不自动意味着失败**：最后一次更新可能恰使结果通过；最终残差检查决定原 PASS/FAIL，不能新增或丢失返回行为。若导出活跃次数，须保留显式状态，不能读取 for 的退出索引。

正确的逐层 BFS 最多处理 `eccentricity(source)+1 <= |V|` 个非空 frontier；必须包括处理最远叶节点、产生空 frontier 的最后一轮。03 已把源码竞争/容量的未证部分分开，并给出有向树这一无需竞争写假设的合法输入族。不能由这个族声称旧实现对任意图都正确。改写只在 Count>0 时清备用 counter、展开边、flip、grid sync、换队列、推进 LEVEL、读取新 Count；Count 为 0 后不再发这些动作。原最后的 grid sync 和 leader store 仍在循环之后，LEVEL 按实际活跃轮次更新。

这两例跨 CTA 的依赖首先需要把合作 grid 登记成同一个多成员 worker，满足可驻留/同步前提。当前 `unit_count` 容许这种解释，但还没有完成对应完整 schema/waiting 迁移；while 不能修复跨 worker 寻址或驻留问题，for 也不能。

可保持的是动作实例、发起次序、原有完成/交接约束和终态，而非每一次 guard 观察、物理完成时间或所有 GPU 调度结果逐一相等。直接解释有界 for 可能做较多空 guard；描述/检查成本也可能上升。**没有 GPU 必慢的结论**：pipe 记录设计约束，Coder 仍可用源码 while/break 跳过已稳定且没有动作的尾部。

04 的一般前提经审计仍必要：真实 cap/有意义轮界、初始与后继条件已就绪、false 后不再被其他角色重新激活、尾部不读取未产生的 `Continue(r)` 版本、条件无隐式读/等待/随机数、退出状态与最终交接保留。若为了改写必须添加锁存动作或改造终态接口，应承认这是设计变化，不能声称仅换语法。我重新运行其 CPU 脚本，72 个原 MaxIter 例与其中 45 个保证界例的动作/结果比较通过；四个错误改写见证也通过。它们是串行合成见证，不是 CG/BFS 或跨 role GPU 正确性测试。

**排除 Gamma、DMR、MIS 的理由合法；SSSP 已有条件成立的自然轮界。**

PyTorch [sample_gamma](../sources/03_pytorch/aten/src/ATen/native/Distributions.h) 114–142 行没有 MaxAttempts，CUDA [functor](../sources/03_pytorch/aten/src/ATen/native/cuda/Distributions.cu) 92–121 行显示 normal/uniform、候选与判断都属于各输出元素的线程局部状态，中间没有交给另一个 role 的候选缓冲。把它保留在既有 compute 粒度没有吞并设计交接；`philox_cuda_state(10)` 不是十次 cap。这个例子证明合理随机算法可有无 cap 的源码重试，不证明这个 pipe 必须展开重试。

[DMR](../sources/03_galois/lonestar/scientific/gpu/delaunayrefinement/dmr.cu) 216–238 行和 [MIS](../sources/03_galois/lonestar/analytics/gpu/independentset/mis.cu) 159–168 行的反复工作是 host 的多次 kernel launch。当前 work_schedule 以每个 phase ID 对应一次提交，role pipe 不负责重发这组 launch；排除不需要发明大 compute。DMR kernel 内的 ownership/commit barrier 仍是真实边界。此结论也没有证明整个 KernelSchema 已能自然描述无界 host orchestration。MIS 的 priority 只在 185 行生成一次，配合 ID 打破平手；不能改写成“每轮随机，所以可能永无进展”的证据。

[SSSP](../sources/03_galois/lonestar/analytics/gpu/sssp/sssp.cu) 344–378 行的 retry/deferred 两层循环确在同一 persistent kernel 中，伴随独立去重与 grid 同步。[07 专项证明](07_sssp_bound.md)补出了 `outer≤V`、`inner(b)≤V`；本审计重新推导关键不变量并回查源码后接受这一**有前提的自然轮界**，撤回此前“轮界未决”的判断。它使用已有 `graph.nnodes`，没有改成同步 Bellman–Ford，也没有新建工作调度器。

成立的执行契约必须与结论一起保留：固定有限图与合法节点、非负整数权重和正 DELTA；所有实际候选加法及 i/delta 更新不溢出；完整扫描每个当前 frontier 在该实现中选定的固定边域；atomicMin 可线性化，距离读取沿减小方向且不退回前一完成屏障以前；队列容量足够、通知和 payload 不丢失，去重保留每个顶点的代表，集体同步和 count 观察成立。这些是算法级读取/队列契约，不能由普通 load 与 atomicMin 混用的旧源码自动推定已兑现。

独立复核中最重要的三步是：

1. **允许 atomic 失败后仍入队。** 源码 237–248、275–287 行的检查、atomic 参数和分类会分别读 source。契约给出 `c_class≤c_atomic≤c_check<d_entry[v]`；因此一次通知只证明该目标在本 wave 内由某个任务变小过，不证明本次 atomic 成功。真正写入阈值内值的任务必通知 re，但 atomic 后 source 再下降可以把仍高于 delta 的目标也送进 re。07 没有误用“re 全是低值点”。
2. **外层保证路径传播。** 每个 epoch 入口，尚有出边可松弛的 source 都在 frontier；每次变小有 re/out 通知，无新变化的已扫描 source 则变为 clean。out 在下一 epoch 立即处理，不等 delta 追上其距离。因而实际 epoch 变换至少达到一次入口 Bellman 松弛的效果；非负简单路径至多 V−1 条边，再加一次处理过时通知的 epoch，得到 outer≤V。Bellman 映射是证明用比较对象，不是新增全图扫描。
3. **内层只需证明阈值内距离稳定。** 把高于 delta 的距离视为无穷，阈值内尚有松弛机会的 source 必在当前 frontier。每个 wave 至少推进一次截断后的松弛，V−1 个 wave 后低值稳定。此后若仍产生 re：atomic 候选已低，意味着目标发生低值改善；atomic 候选尚高而分类变低，意味着 source 在该 wave 降到或进一步改善低值。两者都与稳定矛盾，故第 V 个 wave 排空 re。它仍可生成 out，外层去重、屏障和 i/delta 更新不得省略。

两个 V 界均包括最后一次通知排空；零权链和单位权链分别能达到单个界。不能把它们误合为全程最多 V 个 wave，也没有证明 V×V 同时紧。缓存 OuterCount/InnerCount 的 for/if 形式保留每次实际 count load，包括最后一次读到 0；空尾只重复读缓存 false，不多发 load、atomic、队列交换或 barrier。最终 i/curdelta 仍由真实活跃 body 更新，不能赋成 V，也没有新增“cap 耗尽正常返回”。

我用[最新版 CPU 模型](../scripts/07_sssp_bound_model.py)重跑了 1,456 个三点图/参数组合和 3,000 个随机协议比较，输出与 data07 的记录逐项一致。**模型仅枚举已定义的 guard/atomic/classify 三阶段交错**：guard 的两个操作数一起读取，atomic 和分类后入队各为一个不可分阶段，读取取当前模型值；完全枚举还把队列按多重集合归一化。它不是全部 CUDA 执行的过近似，也不验证每个 CTA 的屏障实现、普通 load/atomic 的内存模型或固定 `2E` 队列容量。上述一般轮界来自不变量证明，不能由有限通过数目推出。

源码 192 行还有 outdegree/DEGREE_LIMIT 选择。因此终止界可以对实际完整扫描的固定边域成立；要称输出是原图 SSSP，还须证明所需边都被处理。**已解决的是所列契约下的轮界与控制投影；真实 CUDA 契约兑现、完整原图正确性和多成员 schema 迁移仍需另审。** 这些尚缺部分不再等同于“没有找到自然轮界”，也不能反过来被轮界证明覆盖。

**动态 for 的准入不是“任何 while 都能叫生成器”。** 正当解释需要能从原设计指出工作输入、结果目标、身份与分配规则，并保留实际产生/领取/发布/确认/资源归还和终止检测。QSA 的输出工作、MLA 的 Geo、CLC 的待接管 CTA 及 MegaMoE 的 task descriptor 都满足这一检查。若本轮确实生成有这些身份的子任务，已有真实 scheduler 可以接收它们；这不等于消除了系统反馈，只说明调度职责本来承担它。

反之，一份求解的残差轮次、拒绝采样的第 r 次尝试，不能仅因事后能写成列表就成为上游工作域。新增队列、把本地判定迁到 scheduler、把整个跨 role 协议合成一个 compute，可能是另一种合理设计，但必须承认工作组织和接口改变；不能据此证明原设计的 while 有漏洞。BFS 一轮内的已形成 frontier 可以遍历，也不自动给出了外层“所有未来 frontier”的合法域。

**方法性反例足以否定普遍消去，不足以独自决定当前核心。** 可以构造分批拒绝采样：Proposal 生成候选并发布缓冲，Validation 读取/验证并发布 verdict，Proposal 显式读取 verdict、确认并归还本轮资源，双方在同一已就绪接受结果上决定是否再做一轮。终止轮也完成确认，故没有理由因为有反馈就称其协议设计坏了。

若每轮独立且接受概率固定为 `p∈(0,1)`，则任意有限 B 后仍未接受的概率为 `(1-p)^B>0`，而无限拒绝的概率为 0。原契约是几乎必然取得合格样本，没有预算失败出口；加入 cap 并返回 Exhausted 是新契约，返回未接受候选是错误，伪造“成功前各轮”域或吞掉发布/验证/确认也违反本研究边界。单说“每轮概率大于零”而没有独立性或统一进展保证，还不足以推出几乎必然终止。

这是**设计构造，不是 PyTorch Gamma 实现的多 role 版本，不是已找到的本库 kernel**。接受常见的几乎必然终止随机契约时，它反驳“所有好算子都不需要 while”；若范围只接纳有最坏轮界的协议，排除的是契约类别，不是证明原算法不好。当前核心是否覆盖这个类别仍是工程选择。不能把 Gamma 的无 cap、CG 的 grid 协作拼接成已有实证。

**选择移除的收益、代价与范围。** 最强的保留理由是：CG/BFS/SSSP 展示了真实且自然的跨动作反馈，while 可直接表达首次停止，不必寻找轮界或维护停止状态。这是实质的可读性与表达能力收益，不是语言完备性口号。此次仍推荐移除，是因为当前十例及广查选中机制没有实际迁移缺口，CG/BFS 的替代表达只需现成条件与真实域，07 又为 SSSP 的精确队列算法补出了有前提的 V/V 界；没有证据要求当前核心为一般反馈承诺完整支持。SSSP 的补证有真实审阅成本，while 的书写便利性依然成立，不能把这种成本说成不存在。

删除的收益限于减少一个尚未被真实 trial 使用的控制形式及其写作/检查规则。删除不会自动简化所有活性问题：动态 worktile、await/acquire 和异步依赖仍然存在。代价是放弃直接书写无合适轮界的本地反馈，并可能使部分有界反馈描述更繁琐；这笔代价必须公开，不得转嫁成巨大 cap、假域、隐藏赋值或不透明大动作。

建议明确写下这样的能力承诺：

> 当前核心覆盖输入/元数据定义的有序遍历、真实工作调度提供的动态 worktile 流，以及能够保持条件就绪、停止状态、动作交接与退出契约的有界门控。原语内部等待/重试及没有独立设计交接的局部算法仍由动作契约承担。当前核心不承诺自然表达所有无合适有限轮界的跨动作本地反馈；遇到此类合理设计，应标明范围外并重审语法，不要求作者伪装成已支持。检查预算耗尽表示未验证完，不能当作正常退出或算法失败。

**最小重审/准入条件。** 不要求未来作者证明 for/if 在数学上绝对不可能。只需出现一份当前确实要描述的真实 kernel：每轮结果决定后续多个可引用动作，且 for/if 需要虚构轮域、改变协议/退出，或增加明显不自然的锁存/终态处理，即足以重审。具有真实 cap 但表达负担显著的例子也有资格，不必只找严格无界算法。

重新纳入核心前，完成至少一份固定源码对应的完整反馈 pipe：工作/成员作用域合法，初始和每轮条件由明确动作建立，所有角色的轮次身份、零轮/提前停止/正常结束/原有预算失败，以及最后通知、消费完成、资源释放均闭合。有限展开/CPU 轨迹只验证其标明范围；进展依据可以是 cap、排名量、调度完成保证或注明的概率保证，不要求普遍终止定理。

**证据修正与仍缺的材料。** 没有发现足以推翻以上范围内结论的硬源码错误；下列限制必须进入最终汇总。

1. 原词法索引漏掉带初始化/步进的空条件 for，05 已补检；因此“零 while 命中即无动态工作”明确不能成立。
2. QSA baseline 328 行的 terminal Qnext “no-op”需要澄清：源码 621–624 行只跳过 Q copy 的等待/栅栏，仍发 qready；trial 76 行仍保留 publish:Qnext。按既有规则，无 payload 动作仍可有 issue/done，不能仅凭旧文一句推定通知事件被删、必要消费者失联或运行时错误。它提醒契约解释需核对，不能升级成坏设计证据。
3. CG cap、终止比较与最终 PASS/FAIL必须分别保留；04 的 CUDA v12.5 与 03 保存的另一固定 commit 本来就是两个版本，行号/散列不同不是来源损坏。本审计直接核对的是 03 本地版本。
4. SSSP 此前的轮界未决已由 07 在明确执行契约下解决；仍未完成的是其 CUDA 可见性、固定队列容量、完整边覆盖和 cooperative worker 映射，以及一般图上的 BFS 竞争/容量证明。数学轮界与实现契约兑现不能混写。
5. 仍缺一份实际运行在 kernel 内、跨 role 发布/验证/确认且无合适轮界的采样或重试实现。若要对 SSSP 做完整源码认证，还缺上述执行契约的实现级核验；若要纳入动态细化，需实际 persistent 源码，不能把现有 host DMR 图搬入 role pipe。
6. 已核对本轮的小修订：04 的 QSA preprocess 片段为 344–359 行，正文为 346–359，文件确有 359 行；03 改为未编译/执行下载源码及外部二进制，避免与已下载原文件矛盾。07 的模型措辞已限定为三阶段抽象，Worklist2::pop_id 的定位以现稿 525–533 行为准。这些修订未改变算法，材料表和报告指纹均已刷新。

本审计重新核对了 104 份 trial 源码、10 份 baseline、14 份提取代表分支和 25 份公开保存文件的指纹，另核对 broad 索引的全部 428 份代码；均匹配。指纹核对不是逐体语义审查。没有运行 GPU、编译 kernel 或执行下载的外部源程序；没有修改正式规则、源码、旧报告或 manifest。

研究启动版本 `38bc1ea` 与期间用户另行提交的 `f0bc432` 已分别冻结到 `sources/00_rules_at_start`、`sources/00_rules_later`，由 manifest 的 document_snapshots 登记。本次核对两套指纹，并检查后续规则仍保留真实动态 worktile、已就绪条件及原语/动作边界的区分；历史 baseline 沿其记录版本解释，无关字段名称不在本报告迁移，也没有覆写用户的新文档。
