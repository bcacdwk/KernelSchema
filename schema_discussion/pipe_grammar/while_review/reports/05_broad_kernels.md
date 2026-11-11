# 第五份独立调查：扩展到 Kernel Wiki 的 44 题

**独立建议：在当前 `for` 已允许真实 `work_schedule` 提供动态 worktile 序列的前提下，当前 pipe 核心可以移除 `while`。** 扩大材料后，已审阅路径仍未给出必须用它表达的本地多动作反馈：未见计算失败后重试整个动作段，也未见 FC1、attention 或状态递推的数值结果再生任务。最强候选分别落在已有工作调度、输入位图遍历、哈希探测和就绪等待。这个结论不支持“合理算子的源码不该出现 while”，更不是所有算子的普遍证明。

本报告先读 [METHOD](../METHOD.md)、AGENTS.md、当前 grammar、catalog 与源码，未读 01–04 的结论。源码位置、分支状态、谓词来源、进展依据和片段 SHA256 收在 [05_broad_kernels.json](../data/05_broad_kernels.json)；可用 [05_build_data.py](../05_build_data.py) 重新生成并核验。

## 覆盖边界

**44 题完成词法筛查；部分路径完成机制审阅。** 复核了定位索引中全部 **428 份 candidate 代码/头文件的 SHA256**，均与索引一致。原索引的 **944 条 while/do 候选行**含注释、宏和历史变体；**18 条字面 `for(;;)` 候选行**中有两行只是注释。它们不是循环或设计机制的数量。

本次另查到 **27 条 `for(init;;step)` 候选**，其中包含原索引没有定位出的 FP8 双缓存 attention 的七角色 CLC 循环及其另一架构版本。还补扫 **218 条 `bra/goto` 候选行**：217 条的相邻片段含 mbarrier，另一条是 `goto` 注释；其中也有向前的 `DONE` 分支，不能按 217 个回跳计数。抽查的实际 PTX 回跳是 barrier 等待。额外 `brx/jmp` 搜索的 38 条命中均为 FMHA 的 C++ 位图变量 `jmp`，其数值语义另行审阅，未当汇编跳转。

JSON 有 **21 个机制记录，关联 29 题的部分路径**，其中 25 题不在此前六个 Wiki root 内。其余四个旧 root 来自另一批四题模板。这里的“关联”包括原语和 host 规划审阅，不表示 29 题全 forward 或全部编译组合完成证明。

| family / 总题数 | 本次下钻的路径 | 机制判断 |
| --- | --- | --- |
| GEMM / 15 | BF16 grouped 的 contiguous/PSUM scheduler；FP8×MXFP4 masked 的组定位、MMA 环首尾；scaled FP8 的 CLC；MXFP4 的末波分割；BF16×NVFP4 flag；MXFP8 PTX wait | 输入 tile 域、真实调度及原语等待。BF16 helper 里的 masked/KGrouped 分支不能冒充该题选定路径。 |
| MoE / 5 | 三种 reference normal kernel 的 scheduler、combine、通信计数；scaled FP8 的前缀搜索/发布等待；NVFP4 的 barrier 和 host verdict 等待 | fixed routed work、输入 top-k 遍历、await。三种 reference 合同均明确 single-rank；`kernel_situ` 登记存在不代表当前 dispatch 选它。 |
| FMHA / 8 | FP8 persistent pool；per-token-head 的 persist/fold 和 timeout helper；NVFP4 QK/PV prefix wait；FP16 PTX wait；MXFP8 数值 rescale scan | 动态 worktiles、有限几何域、await，以及真实数值条件下的 `for/if`。 |
| MLA / 4 | expanded BF16 的 tc2p 与退役分支区分；expanded FP8 五个 role loop；absorbed 的 SIMT 前缀定位；NVFP4 host plan | 有限 item pool/终止票、metadata 搜索和 host 规划。 |
| 稀疏 attention / 4 | FP8 双缓存 CLC；indexed FP8 的有真实上界 CLC；BF16 union 插入/重复项；MXFP8/NVFP4 fused merge | 工作接管、有限输入收集、原语重试和一次发布后的等待。 |
| 线性 attention / 8 | BF16 conv-MTP 的 FK flag 协议；FP32 postconv-MTP token 递推；FP32 prefill fixup；KDA engine/chain；decode PTX wait | 数值状态沿真实 token/chunk 域递推；conv→GDN 的通知/等待/归还是独立事件。 |

## 强候选一：reference MoE 的 scheduler 和 `do_reduce`

[MegaMoEScheduler](../../../../kernel_wiki/aka-cuda-operators-20261030/operators/moe/bf16_fused_moe/candidate/kernel_helpers.cuh#L1436) 有真正的跨角色协议：scheduler warp **等空 slot → atomic claim → publish TaskInfo**；消费者取得 metadata 后分别发 TMA、MMA、epilogue；[epilogue 在 TMEM full 后归还 task-info](../../../../kernel_wiki/aka-cuda-operators-20261030/operators/moe/bf16_fused_moe/candidate/kernel.cuh#L883)。末尾还有一次无效领取和单独发布的 `BlockPhase::None` 终止票。这不是“一个 while 里面只有整数加一”。

但工作域的来源也很具体：`fetch_expert_recv_count` 等完整路由计数，按专家 token 数形成 `num_total_m_blocks`；L1/L2 的任务数分别乘各层 N cluster 数。`get_next_task` 的 warmup/交替策略改变领取顺序，L2 还等所需 L1 **已被领取**；真正数据就绪另由 full count / K-ready mask 保护。没有任何 FC1/FC2 数值进入下一任务判定，也没有 enqueue/requeue。三个 reference 的相应 scheduler 片段逐字节相同。

因此可遍历这个**原本就存在的 TaskInfo 工作序列**。pipe 中仍须显式保留 claim、publish、receive、release、invalid claim 和 sentinel；特别不能把 claim 提前到 `await empty` 之前，或让 `for` 头隐式领取。把它归入 work_schedule 的依据是源码已有任务单位、全局分配器、专用 scheduler warp 和共享发布槽，不是临时定义“直到成功的轮次”。

另一个容易误判的名字是 [`while(do_reduce)`](../../../../kernel_wiki/aka-cuda-operators-20261030/operators/moe/bf16_fused_moe/candidate/kernel.cuh#L1248)。`total_mask` 来自输入 top-k 槽有效性；`move_mask_and_load` 清最低有效位并发 TMA，`do_reduce` 表示这一份是否存在。FP32 `reduced[]` 的数值不改变 mask。自然表述是：非空时发首项；`for(有效 top-k 槽，按 ffs 次序)` 中先预取下一项，再等当前 load、累加并推进原 stage/phase；末尾 cast/store。空 mask 的零输出也必须保留。这里无需把 load、await 和 accumulate 合成一个“大归约动作”。

`pending_mask` 与 `ld_acq` 计数循环是 producer/consumer readiness；rank min-peeling 的算数对象是已经收齐的路由计数，未命中一轮至少消去一个非空 rank。`num_bytes_per_pull` 则是 **host** 把载荷字节数缩半的配置计算，根本不是反复尝试通信。

## 强候选二：FP8 双缓存 attention 的 CLC 循环

[`runtime.py`](../../../../kernel_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_fp8/candidate/runtime.py#L178) 确实可选 H8/H16 exact 路径。[SM103 scheduler](../../../../kernel_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_fp8/candidate/fp8_h8_h16_source_exact_sm_103a_kernel.cu#L1239) 的 `for(init;;step)` 发起 `clusterlaunchcontrol.try_cancel`，等异步响应、读取 valid 和 CTA 坐标；七个 role 围绕它处理初始 CTA 和接管的工作。它比单纯遍历 host array 更接近所寻找的反馈。

仍然有明确的工作来源：**已发射网格中尚未执行的 CTA**。每个成功响应接管一个既有工作；invalid 立即终止，没有失败后重试直到成功，也不根据 softmax 数值生成新 CTA。`active_topk/num_steps` 另从输入长度得到。自然域是 initial CTA 加硬件调度器实际交付的 worktiles，不是事后收集“所有成功轮次”。

这里不可省略的细节是 [consumer 对 invalid 响应也先归还 `work_empty` 再退出](../../../../kernel_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_fp8/candidate/fp8_h8_h16_source_exact_sm_103a_kernel.cu#L518)。cancel 发起、completion wait、response read、各 role 的 return、terminal 处理都应保持显式，不能只写一个 `for(tile)` 就宣告协议完成。scaled FP8 GEMM 有同类独立 CLC scheduler；indexed FP8 虽无 while 命中，已有 `total_work_items+1` 的 CLC 遍历——这是实际 launch 工作数加终止票的上界，不能推广为任意反馈的 cap 技巧。

## 强候选三：稀疏 union 的 CAS、发布和重复项

BF16 双缓存的 [packed union build](../../../../kernel_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/candidate/dsv4_kernel.cu#L2303) 包含 CAS、dense position 分配、`ukey/umask` 写、位置发布、等待其他线程发布，以及输入重复项的补写。不能把整个段落都叫成“CAS 原语”后消失。

逐条追踪后，**返回重试边只经过槽探测**：碰到不同 key 就换下一个槽；成功占槽后执行一次分配→写→发布，立即 break；碰到同 key 则只等已占槽线程完成位置发布，再更新 membership，必要时加一次重复项，随后 break。它没有在一次发布之后重新执行整个插入动作段。外层 item 数来自输入列表，`TCUCAP=2560` 与 `TCHT=8192` 是具体收集容量和哈希槽域；CTA barrier 后固定 `nkeys`，才开始后续 attention。

可保留 `for(输入 items)`，让 `LocateOrReserveSlot` 承担原有槽探测，之后用 owner/non-owner 分支**显式**写和发布，或 await 已发布位置；最后保留 CTA 完成边界及 union-key 遍历。原子探测内部仍可以是合法 while。`DB=true` 是 cfg65 对照分支；默认 `overlap_probe` 是真实 planner 的 metadata probe，不能因名字含 probe 全判成调试；而 `fuse_step` 那个 CAS 归约融合由默认关闭的 `g_fuse_combine` 控制，三者状态须分开。

## 强候选四：FMHA 的动态 pool 与数值条件

FP8 FMHA 的 [当前 host gate](../../../../kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_fp8/candidate/fmha_fp8.cu#L1879) 默认 AUTO：MT 长链可以用动态 ticket pool，短链用 static segment；TC 路径用 static。`fmha_tc_pers.cuh` 留着 dynamic 分支，但 host 在 TC static 不可行时不给它造动态 flat pool，而是回退普通 launch。因此不能只凭头文件或旧“experimental”注释判断选定状态。

[MT loader](../../../../kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_fp8/candidate/fmha_tcmt.cuh#L1375) 等前一任务 metadata 被读完，再领取、写 aid、发布序号；MMA 和 softmax 读取后分别确认。整个 pool 的元素在 launch 前按 B/Hkv/query tile/causal KV 几何构造。计数器清零是给下一次 graph replay 使用，本次不回流。采用动态 worktile 遍历时仍须保留这些确认、终止票和 reset，而不是只保留最终 attention 输出。

还专门追了 MXFP8 FMHA 的 [rescale scan](../../../../kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_tc.cuh#L1464)：这里**确有数值局部反馈**，`segment_max - m > threshold` 决定写常值前缀还是读取并扫描后缀，`m` 也逐段更新。但外层仍遍历固定输入 segments，后缀范围来自本段边界，源码已自然使用 `for/if`。保留这种分支就能保留跳过的读取及输出次序；没有必要强制重算所有段，更不能把它说成没有数据依赖条件。GDN/KDA 同样存在数值状态递推，但处理下一个 token/chunk 的依据仍是输入序列长度。

## 尚未证明的部分

本次给出了所列路径的控制来源、动作边界和自然替代表述，**没有执行新的 schema 迁移或一般事件轨迹等价证明**；也没有 GPU 运行。原语等待的进展依赖对应 producer/consumer 与调度契约，不由删除一个语法节点自动保证。mixed-format sparse fused merge 的 one-wave/驻留 gate、GDN FK 对小网格和经验 block 顺序的依赖，都已在 JSON 记录；未把源注释提升成通用 CUDA 调度保证。

采样验收、历史分支、调试 timeout 与正常出口不能混用。三种 reference MoE 的多 rank 通信模板超出这里的 single-rank 验收；29 题所列片段之外的数值实现、更多编译组合及其他合理算子仍可能提供新证据。当前建议是收紧已有证据支持的核心，而不是宣布反馈算法不存在；如果出现真实的多动作重试、结果驱动任务再生或无自然工作域的协作迭代，应重新评估扩展 `while`。
