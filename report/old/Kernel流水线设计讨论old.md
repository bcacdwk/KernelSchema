# Kernel流水线设计

## 1 背景和动机

**Kernel Agent 现在的两个普遍问题：**

1）难以优化，无法保证是否是优解，长程任务非常依赖LLM自己的智力，突破平台期很困难，不知道什么时候停止迭代（token消耗&latency下降 的平衡）

2）难以维护，人类专家无法干涉和插入，无法观察和判断，直接全部交给agent，后期生产和解释困难，纯黑盒（社区影响力、可学习到wiki等迁移能力）

**解决的思路：**

“策略意图”本质上是真正的数学问题，是在已有的算法要求下，最好的规划和安排硬件资源、吞吐内存、计算顺序，这些内容是纯“mental”的高层次问题，不需要任何具体的GPU语言，用 yaml 等等就可以自然表达

“代码实现”本质上涉及到了大量的计算机代码细节实现，类似 CuTe、CuTeDSL、CUDA、Triton 等等，都涉及到了编译器+精度验证+速度比较，还会有许多 数据layout拓扑、语义接口约定、版本对齐 等问题

以上的两个问题，完全不是同类问题；如果 agent 提出了一个“新的QK计算顺序” 策略，那么就需要推倒所有代码实现重来，这不仅让 agent“望而却步”，也让它过早收敛到局部最优（太容易写错代码），负反馈太多，奖励太少

**“策略意图”恰恰是，人类专家最擅长、最容易感知/判断；同时也最容易介入决策的“taste”问题**

好的流水线是 SOTA Kernel 的必要条件：一个结构性很差的空间—时间—存储映射，不可能靠漂亮的 CuTe 模板、语法技巧或局部指令调整变成 SOTA。代码实现只能逼近一个设计的性能潜力，不能凭空创造不存在的数据复用、并行度和流水重叠。

真正有研究价值、最核心、最值得验证的科学问题：

**是否存在一个比源代码明显更紧凑、更可解释，同时又保留绝大多数一阶性能因果关系的 Pipeline IR？**

**换句话说，这个中间表示是否能成为性能的某种“近似充分统计量”：**

在同一个流水线下，不同合格实现的性能差异应当相对可控；

不同流水线之间的性能差异应当显著大于同一流水线内部的实现差异；

根据流水线表示，应当能大致判断候选方案的相对优劣；

代码可以被验证为该流水线的一个 refinement，而不是只看最终速度猜测它的实现

2026/08/17:

## 2 初步想法

**预期的kernel设计步骤（from scratch）：**

1）硬件spec、基本primitive、功能支持、不同类型的PTX/SASS指令；计算硬件、访存硬件、指令发射硬件在不同典型tile size的吞吐/延时的microbench

【可以先探索常用primitive，后期逐步沉淀，根据迭代需求及时测试、事后增补，每种卡对应不变的spec】

2）数学公式（SPDA）、Pytorch算法（online-softmax）、输入输出shape规模、精度规格的理解

【这个设计工作是不是可以完全交给AI？感觉是纯数学/算法层面的形式化验证和迭代；也就是在不改变任何 函数 y -> f() 的输入输出前提下，调换改写计算顺序/计算方法的trick】

如何评估？

返回RMEM/SMEM/GMEM不同memory的访存总量、Tensor Core/CUDA Core的计算总量

模块化，在KDA/GDN等可选的优化步骤？

3）bottleneck确定，数据先后依赖图，roofline估算（各个存储器层级对应的计算操作）

【偏向于算法-算子实际需求workload，和硬件、算法都相关的前置信息；也就是给定算法+硬件就可以推导出来，每种卡的特定kernel对应不变的spec】

**4）流水线设计和搜索：SM/Grid/CTA的任务划分、数据Tiling Size决策、Warp Spec工作分配、R/T/SMEM存储资源分配、barrier计算顺序、数据流转和更新过程**

**【最关键的design，可以用特定某种DSL来表达、近似评估性能上限；需要保证这个DSL描述的完备性、唯一性；而且最好能够快速的生成可被GPU机器验证的prototype code；这个地方非常适合用 envolve 的遗传策略来维护谱系，因为pipeline design具备Tree的分叉行为，甚至可以直接广搜/深搜】**

单步内微流水 + 搜索维护方法 + 评估微流水方法

跨步的任务划分和SM/CTA调度

**5）流水线实现：将上述流水线描述，通过实际的 CUDA、CuTeDSL 等逐步实现，通过精度bench测试保证正确性，通过NCU/IKET等trace工具保证符合流水线设计，通过任务需求的workload评估端到端算子性能分数**

【和 (4) 完全解耦、经典的kernel agent；这个地方其实最适合用AKA进行多轮迭代，要么是一口气写好，要么是渐进式披露；任务很明确，就是在给定fixed pipeline设计后，完成得到最优的lower-level DSL代码】

6）根据实际实现的效果，在一方面，微调流水线进行深入演进（也就是几乎不改变pipeline顶层design，只改变例如buffer深度等较简单的代码修改）；在另一方面，总结和梳理这一种流水线设计存在的问题和阻力，反馈 (4) 的流水线设计designer，为下次pipeline envolution给出指导

【在此处分叉，一个继续前进直到某种条件停止；另一个重新回到Loop里面】

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/a/RNJroOxr3UGLN01e/5564c2f836774f549bc7d5349a2dcedc0521.png)

## 3 关于CAKE

CAKE（https://arxiv.org/pdf/2608.12629）恰好是符合我们需求的一种 Intermediate Representation，非常适合用于描述“流水线”：**Python 表面的 kernel 语言：重 schedule、轻 layout**。比 Triton 低（warp 角色、barrier、pipeline、内存层级都要写出来），比 CUDA 高（地址、phase、TMEM offset、warp identity 由 lowering 推导）。

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/a64bb41d-2a1a-402a-8564-49dd58d185ca.png)

Agent 只需要负责 schedule：流水线切分、load MMA、buffer 位置、barrier 怎么交接。具体的底层细节交给编译器，**默认不改动生成的 CUDA**。仅仅是最关键的 指令形态、pipeline 深度这些性能决策在 IR 里，agent 拿诊断改 schedule，不是改 CUDA 代码

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/b5e99a38-61c0-4f0f-971c-49b13d365c57.png)

CAKE 和 CuTe DSL 比较相近，但故意不做 layout 代数（layout 写成具体承诺，由编译器查合法性）；编译器不只查错：先 type-check / verifier / cost model，**再通过确定性的编译生成得到 lower-level 的 CUDA/PTX**，然后走 nvcc 编译运行。Agent 默认只写 Cake IR。

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/ceafaa10-e3d2-43ce-a599-0e143c56e128.png)

问题：这个依然是一种“需要被执行”的语言。换句话说，这个应当落在“Coder”的层级，而不是“designer”，依然是Coder在开发代码使用的工具；并没有真正的抽象成一个“顶层的设计”

CAKE-IR的第三方实现参考

https://github.com/huangzhilin-hzl/cake-ir/tree/main

## 4 想法

Coder 和 Designer 其实是对称的，关键是，我们怎么样“减轻 designer 的工具负担”

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/4c70edf6-e134-48e4-b52f-e6f8c7d67d8a.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/08398da4-d82a-4a15-8eeb-fda39b3fbc84.png)

**本质上，可以要求：**

对于 schema，要保证这个尽可能的“轻量”，要尽可能用少的、正交的原语，来刻画现有的所有kernel，也就是通过“结构化表格”+“不可执行伪代码”，完备的描述所有可行的设计思路

对于 检查器，一定要保证在“不运行代码”的前提下，进行静态校验，可以通过确定性python+不确定Agent混合快速校验 的内容，可以“放过差的设计”，但不能“漏掉好的设计”，也就是要检查的偏乐观、偏宽松

对于 评估器，需要通过多种不同维度进行估算，这个地方的核心在估算的快速，而且，要尽可能保证估算的“保序”，不要求它准确，但是能尽可能保证 评估结果“设计A>设计B>设计C” 是近似准确的

**CAKE 之所以是一个大体量工程，是因为它做了确定性的 lowering/codegen 和全架构后端；但是，在这一个架构里，这些实现全部被刻意移交给 Coder 实现（自带 NVCC+NCU 反馈）；在我们的设计中，去掉 codegen 的后半部分，前面剩下的检查器和估算器是 Python + 一套 microbench + 一张硬件表，依然保证了在"Agent Harness 工程"的范畴内，不会变成开发 AI 编译器**

_AI总结：总的来说_

_(1) 要描述语言，但它是"由检查器反推字段"的结构化 Schema，表格+伪代码只是它的渲染；_

_(2) 检查器可以按性质划分，资源记账和合法性查表必做且便宜，活性检查便宜到不该砍（它拦 hang），数学不变性不机械化、用 license+metamorphic+下游精度兜底，agent 只当审计员不当 gate；_

_(3) 评估器做"乐观上界 + 事件仿真"双层，乐观性换来 sound 剪枝，争用折进组合成语能力卡，校准目标是"均匀地松"；_

_(4) 不自建可执行语言；CAKE 已经替你验证了那条路的价值和成本，我们的差异化恰恰在于把 codegen 外包给 Coder LLM 和现有 DSL 梯子，只拥有 Schema + 检查器 + 评估器 + 能力卡这四样别人没有、也不会有人替我们做的东西_

_如果用一句话向别人解释这个架构：Coder 手里有 CUDA+NVCC+NCU，Designer 手里有 Schema+检查器+仿真器；前者是 NVIDIA 给的，后者是我们造的——而后者之所以造得起，是因为它是编译器的前半截，后半截由 LLM 时代免单了_

因此，第一个工作是：像 CAKE 自底向上进化 IR 那样，先拿公开的 SOTA kernel（FA2/FA3 论文、CUTLASS 例子、DeepGEMM）用 schema 手工/半自动地"转写"成设计文档。这样可以同时验证：1）验证 schema 的表达完备性、2）填充流水线模式库、3）直接构成 Designer 验收测试集（能否无参考地重新推导出这些设计）

这个schema的阶段性成果：

1）能够完备、唯一的表达现在主流的所有kernel，FA3/4、MoE、GDN、KDA、GEMM、MLA等

2）能够被人类较为容易的读懂、确定流水线

3）词汇尽可能的简洁，用尽可能少的词汇族，表达出整个流水线，而不能膨胀到真正的DSL

大概的分段：

1）算法要求（不可变）

算法声明、数学公式的计算流程

要求的workload

2）硬件规格（不可变）

存储资源限制

计算吞吐、带宽延时等

硬件新特性

3）核心设计（designer可变，Coder不可变）

任务划分

调度策略

**任务步骤**：

1）应当先收集至少现在H卡、B卡的开源SOTA（不SOTA也可以，开源就行）重要算子，主要是：FA3/4-type，MoE-type、GEMM-type、MLA-type、DSA/GND/KDA等。

2）然后再让Agent去设计初版的语言，首先的任务是能够描述清楚几个典型算子，我们就暂定例如FA4、fusedMoE、GEMM。然后，在这个基础上，再考虑：把其他的算子加入考虑，一方面看这个原语和词汇是不是足够的/有没有要补充的，另一方面看词汇的设置有没有重复、冗余、类似、可以精简的。然后最终先实现一个能表达所有的开源算子流水线的语言。

3）在这个初步阶段之后，再逐轮次优化这个语言的一致性，增加规范性，同时继续精简。

## 5 Schema的设计

第一，收集的SOTA算子：

1）atrex bench里面的 challenge（清洗过），和 data/opensource（原来的大量kernel），针对sm90、sm103、sm120，和各种算子规格都有

2）hpc-ops里面的src，针对sm90、sm100，偏向于传统的经典kernel

第二，建立一个可以迭代的轻量级Harness，可以 自驱动的research Harness

先做了一个简单的尝试：从已有的kernel中提取“有影响的设计决策”

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/593ce918-07c3-45a3-af9e-fe3a447720ac.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/7851a69d-ba61-4bb2-9d02-31d45b28b47d.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/86e63ee9-9e46-4c9d-8c3b-e8b09927c7d6.png)

```json
{
  "claims": [
    {
      "id": "S-C1",
      "text": "A unit of work is the triple (m_block, num_head, batch_size) taken from work_tile.tile_idx; from it the CTA builds exactly one Q tile at (m_block, 0) and un-indexed K/V tile views over the KV sequence axis, then iterates over KV blocks.",
      "status": "source_fact",
      "attribution": "designer",
      "evidence": [
        "C1:786..788",
        "C1:830..833",
        "C1:1029..1036",
        "C1:1056..1063"
      ],
      "counterfactual": "Splitting the KV axis across CTAs (split-KV) would require cross-CTA combination of partial outputs and LSE; keeping the whole KV sweep in one CTA keeps the online-softmax state CTA-local. Coupled to K/V multi-buffering (S-C13) and Q's mainloop residency (S-C23).",
      "uncertainty": "The `_` fourth component of tile_idx (a split index, num_splits=1 at C1:696) is never used in the cited kernel range; scheduler internals are absent."
    },
    {
      "id": "S-C2",
      "text": "The launch grid shape is produced entirely by TileScheduler.get_grid_shape and is not visible in the provided evidence; grid extents, axis folding, and any intra-scheduler load-balancing policy cannot be determined from these files.",
      "status": "unknown",
      "attribution": "uncertain",
      "evidence": [
        "C1:692..709",
        "C1:743..748",
        "C1:36"
      ],
      "counterfactual": "Not asserted: no claim is made that a particular grid dimensionality enables or prevents persistence.",
      "uncertainty": "tile_scheduler.py is not provided; the only grid facts visible are that grid_dim comes from the scheduler and block is [num_threads, 1, 1]. Scheduler semantics must not be inferred from class names."
    },
    {
      "id": "S-C3",
      "text": "Grid shape and work-tile identity are delegated to a scheduler class chosen at compile time by whether cu_seqlens/seqused Q tensors are present.",
      "status": "source_fact",
      "attribution": "coder",
      "evidence": [
        "C1:683..686",
        "C1:709",
        "C1:786..788"
      ],
      "counterfactual": "Computing the grid inline would fix the mapping policy in the kernel and prevent swapping in a different scheduler for varlen inputs.",
      "uncertainty": ""
    },
    {
      "id": "S-C4",
      "text": "This path is not persistent: the kernel body fetches exactly one work tile via TileScheduler.create/initial_work_tile_info and contains no scheduler-advance call or outer work loop through prologue, mainloop, finalize and epilogue; one CTA processes one m-tile.",
      "status": "source_fact",
      "attribution": "designer",
      "evidence": [
        "C1:786..788",
        "C1:958..1089"
      ],
      "counterfactual": "A persistent loop would let one CTA process several tiles and amortize launch/scheduler cost, but would require re-initializing accumulators, softmax state and the aliased smem buffers between tiles; coupled to S-C52, S-C58 and the aliasing structure (S-C49). No claim is made that persistence would amortize a tile's Q load or prologue across unrelated work tiles.",
      "uncertainty": "Absence is asserted over the cited kernel-body range; scheduler class internals are not provided."
    },
    {
      "id": "S-C5",
      "text": "Default tiling is tile_m=128, tile_n=128, num_stages=1, and head dimensions are padded up to a multiple of 16 into tile_hdim/tile_hdimv, with the padding decision also setting the hdim out-of-bounds predication flags.",
      "status": "source_fact",
      "attribution": "designer",
      "evidence": [
        "C1:51..55",
        "C1:80..89",
        "C1:819..821"
      ],
      "counterfactual": "Smaller tile_m reduces SMEM/register footprint and raises occupancy but shortens the K/V reuse of each Q tile; larger tile_n lengthens each MMA/softmax step and increases per-stage SMEM. Coupled to the SMEM capacity pre-check (S-C29) and buffer depth (S-C13).",
      "uncertainty": "Actual launch/autotune values are not in the evidence (metadata.json omitted); these are constructor defaults only."
    },
```

要提取现有SOTA kernel的流水线schema设计，本质上是一个数据问题

我们需要先把大量的kernel的流水线给抽象出来，通过给定的语言，先尽可能的把语言的自由度提高，增加primitive，然后逐渐的涵盖全部的kernel；之后再一轮一轮的精简语言；

本质上“一个语言”的语法是没办法“交付成为产物”的，最好的方法就是“以大量的参考语言“本身，来记录一个语言。换句话说依然需要大量的准备工作，把现有的SOTA kernel的核心思想提取出来，这本身就是造数据的过程；但是我们的核心问题是，把提前出来的“这么多高层流水线”，再压缩

所以问题大概分为：

1）收集开源SOTA kernel（nvidia卡），进行必要的数据清洗

2）把这些kernel，都通过一个类似的“模板”+“原语”（先给一个模板+类似于CAKE-IR的已有风格减少工作量，可以提前把大家统一起来）；然后让多个Agent并行的进行提取，抽象，变成高层次的schema；Agent自己具备“改动模板/增加原语”的权利，但是要尽可能的保守。强调Agent必须优先“抓大放小”；在单个kernel上，进行多轮次的迭代，不断的精简，交付。不同的kernel可以分batch并行。这里的关键是“要让Agent完备的cover一个kernel的设计”，也就是 macro tile + micro pipe 这两个东西搞清楚

3）对得到的几十个算子的schema，进行统一，舍弃特例的primitive，确定最小的交集primitive。最后抽象成一个最终的固定风格的语言，并且附上大量的示例风格代码

4）后面再研究检查器、评估器的事情（先有一个好的语言）

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/a/RNJroOxr3UGLN01e/85d2a01e34454c67a771c98a1bae438f0521.png)

**一方面：往上走，通过Harness来把大量的kernel抽象成schema，变成高度精简的语言**

**一方面：往下走，通过类似的Loop，把L20C/D的kernel自动的翻译成CUDA代码，并且速度对齐**

## 6 Harness的设计

为了实现 SOTA kernel 的向上抽象提取、向下规范统一，需要一个自动化的、长期的 Agent Harness，来不断的完成整个任务。因此首要的问题变成了：设计一个 全自动、轻量级、可靠稳定、智能的Harness

**早期版本：**

假设手里面有几个roles，对应的每个roles可以对应不同的foundation model（GPT，Claude，Qwen等等），这个先不管，可以自由配置他们的模型、强度等等。

一个最小的循环（或者说，一次round）：

1）Director承上启下、统揽本轮工作：对上，判断Advisor上一轮的反馈建议，结合自己的判断，和全部的轮次历史、演化情况，给出下一轮的目标建议，相当于知道round1/2/3/4等等的全部流程，知道全部的信息，做宏观决策。同时，对下，把他确定的具体的任务，按照不同的语言风格，同时并行的唤起多个worker独立的工作

2）Worker并行赛马工作：对上，接收Director的命令，开始真正的工作，而且他绝对只能看到自己的工作区、Director需要让他看到的工作区，上一版完成的统一结果，以及必要的、被允许的全局资料。每次的worker是并行工作的，他们的工作环境完全独立，知道“还有别的worker”，所以不能访问他们；他们会有自己的工作成果、简要结论和md（简洁，只记录重点），这里要设置timeout免得无限迭代

3）All2All互相通信：新加入的，也就是当所有的worker工作结束后，每个worker现在会继续之前的session，但是，他们被允许去主动访问和观察其他worker的成果

4）Worker：根据其他人的结果，进行辩证的反思，自己的不足，思考大家的路径差异，并且作为信息，继续再迭代一轮自己的工作

5）Synthesizer进行总结挑选：它负责把大家的分歧整合，或者挑选最好的结果（outlier），把大家的共同结果沉淀下来，确定最终提交的版本，可运行、正确无误，剔除或者无视掉错误的结果。这里很重要的是要注意少数人的正确观点，也要注意多数人的大体主干。也就是它会得到一个具体的round1/2/3/4的正式结果了

6）Advisor负责本轮统揽，他只看本轮提供的信息，不看之前轮次的历史。判断Director的目标与最终synthesizer的结果的gap、问题，提出建议和问题，注意建议的精简性和凝练性，把最重要的3-5个建议总结出来。同时审查当前结果是否有hacking的欺骗情况，是否真的能跑起来。他也看worker具体的工作细节，看看有没有什么有趣的worker碰到的insight、经验值得记录，或者对整个Harness提出小的建议（供人类最外层调整Harness进化）。

7）在最外层，可以有一个operator负责监控整个流程。它可以看情况，发起多个任务。可以是并行的5个loop，做5个不同的工作，然后当一个loop的5轮结束后，可以继续把目前的结果定版，整理成比如ver1，然后重新唤起Director，又开始5轮，但是旧的Director就不需要看之前的老结果了。也就是串行/并行，负责和用户来对接。这个更多的不需要一个具体的流程，而是用Agent.md来告诉他怎么办就可以，他也可以看情况停下来要不要修改Harness的问题、处理问题情况。

但是，这里的设计依然是比较老旧的思路，参考了之前的Harness调研+之前一些关于Multi-Agent的研究，以及和 Fable 5 max + Sol 5.6 Pro 的讨论，**最终确定的设计思路：**

**带事务化 commit 的 quality-diversity 搜索器：**

**以动词而非 role 定义系统；只做 selection、不做合并；证据优先于观点；预算制取代固定轮次；状态外置到文件、不依赖对话记忆；Advisor 的本质是 falsifier；主干只接受单向、事务化的 commit。**

**设计框架与细节：**

**一块 blackboard、三个 role、外层两条 rule**

blackboard 就是一个 git 仓库：

*   `CONTRACT.md`：由人撰写，包含目标、约束、评测命令、promotion 规则与预算，epoch 内冻结；
    
*   `eval/`：两级评测——cheap gate 的脚本对所有 Worker 可见，hard gate（隐藏测试、随机化输入）只存在于 Promoter 的环境中；
    
*   `ledger.md`：一页纸的全局状态——当前 champion、archive 索引、已证伪假设的失败指纹、未决风险、停滞计数、剩余预算；
    
*   `cards/`：每个候选一张结构化的 result card——假设、最小 diff、实测数字、失败条件、可复用的部件；
    
*   `archive/`：top-k 精英分支，champion 打 tag。
    

三个 role 全部无状态，随用随建、用毕即弃：**Director** 读 ledger、写出 K 份简报；**Worker** × K 在各自隔离的 worktree 中并行工作；**Promoter** 负责运行 hard gate、执行 commit、更新账目，审计被触发时切换到另一家族的基座模型构造反例。

blackboard 是共享的记忆介质。

一轮的流程：

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/a/RNJroOxr3UGLN01e/8456cdd4749f42a982ab7fa2593fa00d0521.png)

完整的流程：

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/a/RNJroOxr3UGLN01e/cb0602e0bc1d4a31a2217ca0a7e03d500521.png)

核心：**每一轮就是一个带两道 gate 的漏斗——K 路隔离发散，m 路凭证据深化，至多1次单向 commit**

```text
CONTRACT ─ human-written, frozen per epoch
   │
   ▼
1  DIRECT    read ledger → K orthogonal briefs
2  EXPLORE   K workers · isolated · timeboxed             → K candidates
3  GATE₁     cheap eval, re-run by driver                 → m+1 survive
4  PULL      survivors read cards (evidence only; 1 hermit abstains)
5  DEEPEN    remaining budget on survivors, same sessions
6  GATE₂     hidden tests · random inputs · certificate   → ≤1 winner
7  COMMIT    beats champion → promote, else NO_COMMIT       (one-way ratchet)
8  RECORD    ledger · top-k archive · dead-end fingerprints
   │            └─ stall / score jump / metric conflict → AUDIT (cross-family falsifier)
   ▼
next round │ budget out or contract met → tag version
           │ stalled × s  → restart from elite / swap hypotheses / escalate to human



人：CONTRACT（目标·约束·评测·promotion 规则·预算，epoch 内冻结）
        │
┌───────▼──────────────────────────────────────────────┐
│ ① 定向   Director 读 ledger → K 份正交假设简报          │
│ ② 发散   K 个 Worker 隔离并行（worktree + 限时）        │
│ ③ 初筛   驱动脚本复跑 cheap gate → 存活 m+1 个          │
│ ④ 拉取   存活者只读 cards/（一个隔离分支除外）           │
│ ⑤ 深化   剩余预算集中于存活者，延续原 session            │
│ ⑥ 严测   隐藏测试·随机输入·重复测量·certificate 校验     │
│ ⑦ 提交   胜过 champion 才前移，否则 NO_COMMIT           │
│ ⑧ 记账   更新 ledger·cards·archive·失败指纹·停滞计数     │
│    └─ 触发式审计：停滞/分数跳变/指标冲突 → 跨家族构造反例 │
└───────┬──────────────────────────────────────────────┘
        │
外层两条rule：预算耗尽或 contract 达成 → 定版打 tag；
             停滞 s 轮 → 换假设族 / 从 archive 精英重启 / 上报 human
```

*   **State**（一个 git 仓库即 blackboard）：`CONTRACT.md` · `eval/`（GATE₁ 公开、GATE₂ 密封）· `ledger.md` · `cards/` · `archive/`
    
*   **Roles**（无状态、用毕即弃）：Director 只分配，Worker 只探索，Promoter 只验证与提交，审计时换一个模型家族构造反例
    
*   **Defaults**：K = 3–5（等于本轮正交假设数）；m = 2 加 1 个 hermit；explore : deepen ≈ 1 : 2；s = 2
    

形状即含义：发散端制造方差，两道 gate 消灭噪声与作弊，commit 端只保留单调。K → m → 1 的漏斗走完一遍，就是一轮。

需要固定的数字只有几个：K = 3–5，等于本轮值得检验的正交假设数，并非常数；cheap gate 后存活 m = 2，另加一个不参与交流的隔离分支；发散与深化的预算比约为 1:2；Worker 自报的数字一律由驱动脚本复跑核实；s = 2。

实现规模：几百行驱动脚本，加 git worktree，加 headless agent CLI 调用；不需要服务端、数据库或消息总线。更换任务领域时只需更换一样东西——把评测命令换成相应的 certificate 校验器（Lean checker、数值验证脚本、反例校验器），loop 本身零改动：**loop 不需要理解任务，它只认 certificate**。

**Advisor 的 Gate 演化路径：**

trigger ──▶ Advise ──▶ 决定是否写出新的gate1/2测试case

                └─▶ 新测试的检查（确定性 + baseline 需要通过）

                      ├─ FALL ──▶ 丢弃，记入 ledger

                      └─ PASS ──▶ 立刻对 champion + archive 再检验

                            ├─ champion 挂 ──▶ 撤销，回滚到通过的最佳 elite

                            └─ champion 过 ──▶ 计入 eval/extra，下一轮生效

**设计理由与数学原则**

先立总纲。这个 loop 要解决的核心数学问题是：**在相关采样、可欺骗的测量、有限预算这三重约束之下，最大化可验证的单调进步。** 整个设计是四条不变量的机械化——不回退、不重复、不丢失、随时可停。以下分层说明理由。

**生成层：为什么是 K 份假设正交、材料各异的简报，而不是 N 个换了语气的 Worker。** 等相关样本的有效样本量为 (N\_{\text{eff}} = N/(1+(N-1)\rho))，当 N 趋于无穷时上界为 (1/\rho)：单纯增加数量没有收益，降低相关性才有收益。推理阶段模型权重固定，唯一可操纵的是条件分布，因此多样性工程只能在条件空间中做正交化——不同的假设、不同的参考材料、不同的基座模型。更换 prompt 的语气并不改变条件分布，所以无效。

**测量层：为什么要 certificate、两级 gate 与复跑。** 将测量分解为 (\hat{s} = s + \varepsilon + h)。噪声项 (\varepsilon) 依靠重复测量以 (\sqrt{n}) 的速度收敛，代价低廉；hack 项 (h) 是对抗性的，且随搜索强度增长——[Gao et al.](https://arxiv.org/abs/2210.10760) 的 best-of-N 定律 (R(d)=d(\alpha-\beta d)) 表明，对一个固定且不完美的指标施加越强的选择压力，真实质量先升后降。由此得到全设计中最重要的一条推论：**gate 的硬度必须随搜索强度同步升级**，否则增加 Worker 反而降低真实分数。certificate 原则把校验权交给确定性代码，令 (h) 项无隙可乘：Sakana 那个 memset kernel 骗过的是单输入测试，骗不过随机输入加参照校验。两级 gate 的本质是按信息价值分配测量预算——多数候选只值一个比特（keep or drop），不必为它们支付 hard gate 的成本。

**通信层：为什么采用 pull、只传 card、只做一次。** 互相阅读就是在做平均：DeGroot 动力学下，个体间偏差按 (\lambda\_2^k) 衰减，交流层数越多趋同越快，方差在 selection 发生之前就已清零；实证数据是 85% 的从众采纳率、七成原本正确的推理被同伴带偏、初始多样性越大从众反而越强。一条消息的净价值等于它带来的后验改善减去相关性代价：实测数字与失败条件的比值最高，叙事与信心最低。因此 card 中只允许出现证据，采纳他人思路必须援引具体测量；push 改 pull 直接消灭了"说服者"这一位置；保留一个不读 card 的隔离分支，既是对照组，也是保险。

**分配层：为什么宽进窄出、允许空转、允许重启。** 困难搜索中进步事件的等待时间服从重尾分布，这蕴含两件事：successive halving 式的"低成本广泛试探、按证据逐级加注"接近最优预算策略；在同一路径上持续投入劣于重启（SAT 求解器的 Luby restart 定理）。No Free Lunch 定理则说明不存在普适最优的拓扑与轮数，所以 K、m、轮次都应当是配置而非架构，唯一固定的是预算上限。强制跑满轮次会制造"为迭代而迭代"以及对 evaluator 的过拟合，NO\_COMMIT 必须是合法结局。

**累积层：为什么要 ratchet、失败指纹与精英 archive。** 经过验证的 commit 使 best-so-far 成为有上界的单调序列，由单调收敛定理几乎必然收敛：方向确定，速率随机。这正是"不依赖温度"的可实现形式——要求终态分布收紧，放弃路径复现。失败指纹是 loop 中唯一只增不减的知识：一次附带实例的反驳，按其先验质量确定性地缩小假设空间（证实有噪、证伪无噪，这是 Popper 不对称性的信息论表述）。精英 archive 用于对抗欺骗性的适应度地形：全局最优所在的盆地可能从当前 champion 出发不可达，必须保留生态位——这是对你 outlier 哲学的结构性兑现，单主干无法兑现它。旧的 Director 可以丢弃，负知识不可以。

**治理层：为什么 contract 冻结、harness 由人管理、随时可停。** 目标漂移是最高层级的 Goodhart：如果"什么算更好"每轮都可以被系统内部改写，ratchet 就失去了参照系。所以 contract 按 epoch 冻结，修改 contract 等于开启新的 epoch，必须由人签署。loop 的 anytime 性质——任意时刻停机都持有已验证的最优解——正是"可人为管控"的数学形式：管控不要求盯守过程，只要求能读 ledger、有权随时中止，且中止无损。harness 自身的验收沿用他给出的那组指标（success@budget、跨种子方差、outlier recall 等），再加一条递归的规则：任何新机制先过消融再进主干——harness 自己也要走 ratchet。

最后收束为四句，对应四条不变量：**LLM 负责稀有的可能性（方差），certificate 负责真实性（判别力），ratchet 负责方向（单调性），ledger 负责不遗忘（记忆与信道合一）。** role、model、K、round 都只是这四件事的宿主，可以随意更换；四者缺任何一件，loop 便无法运转。

一些优化：

1）发起Spira.py后，有一些可视化界面：

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/ad02ddf1-949a-4560-b3b8-f5d7f5e89f47.png)

2）worker运行的可配置多个模型。用户可以指定可以“使用的”多个模型，Director自己决定dispatch。worker可以一轮固定4-8个进行选择

3）gate1、gate2的自进化升级？reward hacking检查

4）Harness的可控自进化？

5）最开始接入的任务拆分

6）每个具体任务的Agent规则？

7）可以被合理记录的历史log、模型选用、耗时问题等等

2026/08/24:

**第一次测试，直接让agent写FA4算子而没有任何参考（在我们的Harness上面）：**

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/5e1a0156-f0a2-450e-b89c-f992f3610f1d.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/f6cd6bc0-9e8d-40ae-8359-d6c169d6cc43.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/1f5cda04-69b7-459c-b97b-8362696a0fec.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/d48263a7-22c2-4a1a-b51d-6e99f8b440cc.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/8a7f7e1e-f8c4-40fc-81e9-16032235440c.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/823aa098-b3d2-4032-99f1-76a9445e2dae.png)

**从这里至少可以看出：**

1）GLM/Kimi的思考时间显著长于Opus，但是质量不如Opus

2）在没有参考流水线设计的情况下，30 round～30小时的表现依然很差，只有SOTA版本的60%速度，而且每次的性能收益都在逐渐减少；

3）有少数round提出了流水线的结构性设计，包括修改流水线设计、累加方法、KV warp设计、新的异步指令和tcgen05的命令，但是都因为  时间不够、或者没有真正写对（结构性重构需要的时间是很长的），而被淘汰掉了

8/25，让Harness的worker严格参考cutedsl版本的FA4进行cuda开发，而不是自己创造：

**第二次测试：**

Agent分工和额度的考虑

**supervisor**：codex GPT5.6-sol-xhigh  来源： idealab API

**director**：Qoder ultimate high  来源：自己的credits

**worker**：CC Qwen3.8max high  来源：百炼API

**promotor**：Qoder ultimate high  来源：自己的credits

**advisor**：codex GPT5.6-sol-xhigh 来源： idealab API

**用 Qwen 跑了 24个小时，一共10个round，最终达到cutedsl的41%**

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/24e9f7bd-5d8d-4ecf-9b13-9c0b8c3910ea.png)

在compute-bound的shape下甚至依然慢了 8x

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/7abe4cdd-e36b-40f2-af47-1bf61d1c1d17.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/5d6d3fb0-ab27-4090-8358-7f698d58d101.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/240feecb-61bd-43b9-a835-5dc17863a3ef.png)

1）Qwen的思考时间还是太久了，平均一个任务，45分钟 + 90分钟 都很难够，有一半都会timeout；整体来说可能是因为没有注入wiki（但是明明有确切的cutedsl代码作为参考设计）；

2）Cursor 的 Grok 4.6 high 平均的时间要快很多；但是他更加倾向于“快速完成”，一旦尝试失败就立刻回退而不再参考真正的流水线。换句话说，过于正确导向

3）目前的Harness，无论是gate1还是gate2，都强调对于正确性的维护，以及对于速度提升的要求，而没有对于“长期工作、尚未完成、结构忠实”的保护。Qwen就像是一个超慢版本的Opus48？看起来是不是不应该在gate1引入一个“真正的验证”，而是通过Advisor等等来进行“fake”的路线验证？

4）

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/9853af38-708c-44c7-ac2d-a413bd8a1976.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/5960e44f-6804-4ef4-be54-2e759233a49c.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/2b3a2480-b9a7-4090-be05-3783f5b00be7.png)

**先在agent模式走通 cutedsl -> cuda 的去封装的流程，可以通过sass/ptx来作为展开逼近**

同理，cutedsl -> schema 的 高层抽象流程？开源的CAKE-IR，tilelang的warp-spec，比如把算子抽象成tilelang？

## 7 现有DSL的调研和思考

不能简单采用 CAKE 论文 Related Work 中“TileLang 把关键硬件细节隐藏掉”的字面判断。CAKE 的这个分类描述了 TileLang 的默认高层使用方式，但不是当前 TileLang 的表达能力上限。CAKE 把 TileLang、Triton、Helion、cuTile 放在高/中层 tile DSL 一侧，认为它们经常通过自动调度隐藏 warp specialization、barrier choreography 和 memory placement；但 TileLang 当前文档已经明确提供了上述手工控制路径。

当前 TileLang 文档中已经有：

*   `ClusterKernel` 和 thread-block cluster；
    
*   persistent tile scheduler、L2 swizzle、cluster-aware tile traversal；
    
*   Blackwell CLC 的 cancel/query 原语；
    
*   `T.Pipelined` 的自动 pipeline，以及 `order / stage / sync / group` 手工调度注解；
    
*   显式 mbarrier、named barrier、producer-consumer 同步；
    
*   显式异步 Hopper WGMMA 与 Blackwell TCGEN05 MMA；
    
*   编译失败而非静默回退的精确 instruction path；
    
*   Developer-level tile primitives 和 Expert-level thread primitives。[TileLang](https://tilelang.com/autoapi/tilelang/language/kernel/index.html)
    

因此，从**可表达性**来看，TileLang 已经能够覆盖你所说的大量宏观和微观决策：

| Designer 决策 | TileLang 当前表达能力 |
| --- | --- |
| Grid / CTA 划分 | 可以 |
| Cluster 划分 | 可以 |
| persistent kernel | 可以 |
| tile traversal / L2 swizzle | 可以 |
| CLC / 动态尾部负载均衡 | 可以，已有低层原语 |
| tile size | 可以 |
| pipeline stage / buffer depth | 可以 |
| warp specialization | 可以自动生成，也可以在低层手工控制 |
| producer-consumer barrier | 可以 |
| WGMMA / TCGEN05 异步发射与等待 | 可以 |
| SMEM / TMEM / local storage | 很大程度可以表达 |
| 具体 layout、descriptor、instruction 细节 | 可以逐步下降到 Expert 接口处理 |

所以不能简单采用 CAKE 论文 Related Work 中“TileLang 把关键硬件细节隐藏掉”的字面判断。CAKE 的这个分类描述了 TileLang 的**默认高层使用方式**，但不是当前 TileLang 的表达能力上限。CAKE 把 TileLang、Triton、Helion、cuTile 放在高/中层 tile DSL 一侧，认为它们经常通过自动调度隐藏 warp specialization、barrier choreography 和 memory placement；但 TileLang 当前文档已经明确提供了上述手工控制路径。

但是**现在的社区并不缺“能够写出细粒度 GPU 流水线的语言”；**

**真正缺的是“把流水线本身变成一个可读、可比较、可留白、可演化、可由人类干预的设计对象”的表示**

TileLang 的正确定位：它是 Coder 后端的强候选（尤其对非 frontier 的 kernel，可以让 Coder 少写很多脚手架；frontier 级别如 FA4 类流水线目前仍是 CuTeDSL/CUDA 的领地），也是快速 prototype 验证 pipeline 设计的候选语言

因此，TileLang可以作为一种“Plan”的载体，schema的Plan应当比他的表达更高；但是它的 knob 词汇表（`num_stages`、memory scope、warp policy、swizzle annotation）可以直接进 template 的"核心决策"清单——这是免费的词汇来源

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/93b2ae9a-f1e7-4f00-be9c-6ecd7d62a90d.png)

大概的想法：

我的想法是，需要先把大量的kernel的流水线给抽象出来。在第一轮（我们要先做的），先通过给定的语言，先尽可能的把语言的自由度提高，增加primitive，然后逐渐的涵盖全部的kernel，在这个步骤中，我的想法是，维护一个schema的规范template，也就是每个Agent在写schema的时候，会先看看这个template是不是有不足以描述的地方，可以先最小化修改template，然后再实现这个schema，相当于是 (template, schema）共同迭代进化；之后第二轮再考虑精简template，优化schema，最终变成一个template，对应所有的schema。这里要求template不应该“大而全”的包含所有，而是应该小而精，把schema的核心拿出来，细节踢掉变成Coder的工作（移到下层），然后再对应的让template越来越清晰规范，原语越来越少，schema也同步减轻、瘦身。

最后，交付的时候，就是一个template，和SOTA kernel的所有schema，把现有的SOTA kernel的核心思想提取出来，这本身就是造数据的过程。

所以计划大概分为：

1）收集开源SOTA kernel（nvidia卡），进行必要的数据清洗（我觉得现在的情况，已经整理到了能被Agent自己阅读+区分的状态，也就是说比较干净了）。目前这个可以认为做得差不多了。

2）把这些kernel，都通过一个类似的template（“表格模板”+“原语词汇”）（可以之后我们先讨论一个初版的template，和schema的case示例，比如类似于CAKE-IR的已有风格减少工作量，可以早期提前把大家统一起来）；然后让多个Agent并行的进行提取，抽象，变成高层次的schema；Agent自己具备“改动模板/增加原语”的权利，但是要尽可能的保守。强调Agent必须优先“抓大放小”；在单个kernel上，进行多轮次的迭代，不断的精简，交付。不同的kernel可以分batch并行。这里的关键是“要让Agent完备的cover一个kernel的设计”，也就是 macro tile + micro pipe 这两个东西搞清楚。而且需要有一个专门的Agent负责确定：必要的Designer的内容没有遗漏、没有错误，而细节的Coder的内容要去掉；对于模糊的内容应当专门存放保持“可自己决定”，但是应该尽可能减少不确定的设计。

举一个例子，假设有50个各种来源、风格、实现的kernel，我们的任务是，根据这些数据，结合专家的经验，来实现一个“数据驱动的schema IR”。而且是前期的、可能风格不统一的、但是每个kernel <--> schema 都已经是正确的。

这就涉及到了一个类似于“从大量的dataset中逐渐学习、训练和提取一门语言”的问题。我个人的计划是，我们的Harness会有一个“单task”的粒度。对于一个task，就是要针对单个优化特定轮次（比如round=5轮），最终实现一个kernel的template、schema。然后把第一个实现的作为例子+新的template优化种子，开始下一个kernel的迭代，得到新的template+2个schema（新的+旧的）。然后类似的，template只会迭代version版本，而schema则会积累，Agent如果改了template，还需要调整之前的schema（当然理论上，只有删改才需要，增加是不太需要的）。我们假设初始template是v0，那么10个kernel可能就是最终到v10（但是理论上可能到v5就差不多够了，因为不一定每轮都要动template）。然后尽可能的鼓励他减少template，在有限的template语言体系下，来描述schema。

这种“渐进式学习”的方法相当于是这样会让Agent逐渐的学习到所有kernel的“核心流水线”设计思路。因为kernel的差异可能非常大。想法有两个：要么先从较为简单的开始，然后慢慢变难，慢慢增加，逐渐的把语言的空间撑大；这样可能循序渐进适合人类，但是也可能会带来每次都要改schema、最开始抓不住重点和核心的问题；要么从最困难的算子开始，往简单的走，先一来学很大，然后慢慢的稳定；但是这样可能最开始不容易收敛、大家形态各异。这个可能需要考虑顺序的问题。

3）然后，对于对得到的几十个算子的schema + template，进行检查，看看是不是都正确无误（没有事实错误）；然后开始精简，舍弃特例的primitive/表格项，抓重点，抓insight，确定最小的交集primitive。最后抽象成一个最终的固定风格的语言，并且附上大量的示例风格代码（一个template，几十个对应的schema）。同时，最后去比较一下，类似于现在开源版本的CAKE，也就是https://github.com/huangzhilin-hzl/cake-ir  的前端语言(论文 https://arxiv.org/html/2608.12629v1  ），有什么变化，是不是收敛到一致了。

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/3417228d-0093-4e4f-8f77-1a38cf8492f3.png)

尺寸锚点： 论文的 Table 4 FA4 forward 的 CUDA device core 是 2369 行，Cake IR 表达为 430 行——可执行的 schedule 表示压缩了约 5 倍。你们的 schema 去掉操作流之后应该再小几倍，一个 SOTA attention kernel 的设计骨架大概落在 50–150 行的量级。如果哪个 schema 写到 400 行，那说明操作流细节漏进来了，该下放给 Coder。

CAKE 和你们的方案有一个本质差别：**CAKE 的 schema 正确性有可执行的 ground truth**——每个 port 的 kernel 必须编译、跑对、性能复现原 kernel 的 physical schedule，抽象错了当场暴露。你们第一轮的 (template, schema) 没有这个 ground truth："schema 是否正确刻画了这个 kernel 的流水线"只能靠 agent/专家读代码判断，事实性错误可能安静地沉淀进语料。

不需要为此建编译器，**但建议在第 2 步的 Harness 流程里加一个便宜的替代物：盲重建审计——审计 Agent 只拿 schema（不看原 kernel）**，要求它复述出 macro tile 调度和 micro pipeline 的时序（谁在哪个 stage 等哪个 barrier、SMEM 何时被谁复用），再与原 kernel 对照。schema 的完备性标准就变成可操作的："一个没见过原代码的合格读者，能否从 schema 无歧义地还原设计骨架"。这恰好也是你们对 Coder 的要求，等于提前排练了 Designer→Coder 的交接。

现有系统大致分成两类：

1.  **把流水线写成实现代码；**
    
2.  **让编译器替你推导流水线。**
    

你们希望研究的是第三类：

> 把流水线作为一个独立、可显式操作的设计计划，但暂时不要求它自己完成代码生成。

基于目前公开资料，我没有看到一个系统同时覆盖你们要求的全部性质。不过存在若干很接近的方向

| 系统类别 | 代表系统 | 与你们的关系 |
| --- | --- | --- |
| 可执行 expert kernel DSL | TileLang、Pallas Mosaic GPU、Gluon、CuTe DSL | 可以实现细粒度流水线，但仍然是 Coder 层 |
| 调度语言或低层 Tensor IR | Fireiron、Graphene | 把数据移动、tile/thread mapping 等结构化，但目标仍是编译和生成代码 |
| 自动 pipeline/warp specialization 编译器 | Cypress、Tawa、Twill | 能推导或搜索流水线，但倾向于把 Designer 决策交给 compiler/solver |
| Agent-facing physical schedule IR | CAKE IR | 与你们最接近，但比你们预期的轻量 Plan 更低层、更严格 |

**一个语言能替代 Schema，至少应当满足以下条件**

### 1. 能表达不完整设计

Designer 应该可以写：

```plaintext
QK pipeline depth: 2 or 3, selected by resource fit
MMA atom: coder-selected within WGMMA-compatible family
SMEM swizzle: unconstrained, must satisfy TMA/WGMMA compatibility
producer role: dedicated 1 warpgroup
```

这里同时存在：

*   已确定的硬约束；
    
*   候选集合；
    
*   偏好；
    
*   Coder 自由度；
    
*   尚未解决的问题。
    

可执行 TileLang 程序通常必须选择一个具体值，才能进行 lowering 和 compilation。它没有把 `fixed / preferred / free / unresolved` 作为一等语义。

### 2. 对等价实现进行归一化

相同的 Pipeline 可能有十种 TileLang 写法：

*   使用自动 pipeline；
    
*   使用手工 pipeline annotation；
    
*   手工展开 producer/consumer；
    
*   使用 scheduler helper；
    
*   自己写 persistent loop；
    
*   使用不同 buffer declaration 和 control-flow spelling。
    

这些代码在 Designer 看来可能是同一个点，在代码搜索空间中却是不同程序。

### 3. 保存理由，而不只是动作

人类专家真正关心的通常不只是：

```plaintext
stages = 3
```

而是：

```plaintext
stages = 3
because:
  2 stages underfill TMA latency;
  4 stages exceeds desired SMEM occupancy;
evidence:
  TMA latency estimate;
  SMEM budget;
  expected overlap diagram.
```

这些理由会决定下一次演化应该改什么，也能阻止 Agent 在后续轮次无意中破坏关键 insight。

普通代码注释可以写这些内容，但它不是结构化设计语义，无法稳定参与比较、mutation、审查和谱系维护。

### 4. 宏观调度在 TileLang 中经常只是“普通代码”

TileLang 能够实现 persistent、swizzle、cluster 和 CLC，但这些常常表现为：

*   一个循环；
    
*   一段 index decoding；
    
*   几个 CLC intrinsic；
    
*   一个 scheduler helper；
    
*   host-side grid 计算。
    

它们是可执行的，但不一定是可直接比较的设计对象。

Designer 真正希望看到的可能是：

```plaintext
dispatch:
  policy: persistent_clc
  logical_work: q_tiles × kv_splits
  resident_workers: 1 cluster / SM pair
  initial_assignment: static raster
  tail_balance: CLC steal
  traversal: M-major, L2 panel=8
```

这比读一段控制流更适合专家判断，也更适合进化搜索。

### 5. 后端变化不应使 Plan 失效

一个好的 Plan 应该能够在以下变化下保持稳定：

*   TileLang 版本升级；
    
*   TileLang lowering 改变；
    
*   从 TileLang 切换到 CuTeDSL；
    
*   某个新 idea 只能用 inline PTX；
    
*   某个后端不支持一个 primitive；
    
*   新 GPU 引入新的 instruction family。
    

如果 Schema 直接绑定 TileLang AST，那么 TileLang 的表达边界会反过来定义 Designer 的想象边界。

尤其重要的是：

> 当某个新 Pipeline idea 无法用 TileLang 实现时，它应该被记录为 `TileLang projection gap`，而不是被判定为非法 Pipeline。

否则你们最终只能发现“TileLang 容易表达的新设计”，而不是开放设计空间中的新设计。

### Designer、建议项和 Coder 自由度可以切分

| Designer 必须决定 | Designer 可以建议 | Coder 默认决定 |
| --- | --- | --- |
| workload/shape/precision contract | MMA instruction family | exact MMA atom |
| macro tile decomposition | preferred TMA granularity | descriptor encoding |
| Grid/CTA/cluster mapping | layout或swizzle类别 | exact layout algebra |
| persistent/CLC/load-balance policy | vectorization方向 | exact vector width |
| warp role topology | register budget目标 | register packing |
| buffer depth与生命周期 | preferred memory primitive | pointer arithmetic |
| producer-consumer event graph | atom-level implementation hint | barrier address/ID/phase mechanics |
| 计算、搬运、归约的 overlap | tail strategy偏好 | template/compiler workaround |
| 必须保持的性能 invariant | 可接受候选集合 | 具体代码组织 |

这个边界不是永久固定的。某个通常属于 Coder 的选择，如果被证明是该 Pipeline idea 的核心，例如某种特殊 swizzle 才能支撑 TMA multicast，那么它就应当被提升为 Designer 决策。

这正是 Schema 比固定 DSL 灵活的地方：**字段的语义可以表达“这个选择为什么在本例中重要”，而不是仅由语言设计者事先决定它永远属于哪一层。**

## 8 schema的具体讨论分析

schema和template这两个东西作为同时要服务agent，和人类专家的东西，最好格式规整，容易阅读，而且也容易维护，是结构化的表达。agent读起来、改起来要轻松，而且做一些正则/匹配/逻辑操作（后续对schema进行批量操作）也能很方便。

至少能够某种意义上限制agent去自由的发挥。就像是给一个人填表，在schema中，至少是一个表头很清晰的情况；而在template中，则可以定义schema对应的“表格形式”。比如说，template本身应该记录了：这个表本身的形状，表头的文字，可以选择的填表词语。而schema会直接被template关联，因而直接产生改动，同时限制agent在操作schema的时候肆意发挥修改表头。就类似于幻灯片的母版。这种文件的映射关系有点像一个数据库系统

### schema：YAML文件

| **候选** | **判定** |
| --- | --- |
| **YAML** | **推荐。** 层级结构天然适合"表格 + 嵌套小节"；块标量 `\|` 优雅容纳 prose 和伪代码；支持注释（人类审阅批注必需）；agent 训练语料中占比极大，编辑鲁棒；`yq`/Python 批处理成熟；git diff 按行、人类可审 |
| JSON | 淘汰：**不支持注释**，多行文本要转义，人写人读都痛苦 |
| TOML | 淘汰：深嵌套和重复结构（多个 role、多个 barrier）表达笨拙 |
| Markdown | 你已正确否决：无法约束，批处理靠猜 |
| textproto (Protobuf 文本格式) | 谷歌内部就是用它做"表单"的，强类型+enum+注释都有；但需要 proto 编译链，对你们是多余的重量 |
| SQLite/数据库 | 淘汰：不可 git diff，人类无法直接审阅，agent 无法直接编辑——你要的"数据库感"只在校验关系上，不在存储上 |

这里有一个值得记住的原则：**数据层要笨，智能放在校验层。** schema 文件本身越接近纯数据越好，所有"聪明"（类型、enum、约束、母版关系）都放在 template 侧和校验脚本里。这正是你说的"限制 agent 自由发挥"的正确实现位置——不是靠文件格式本身限制，而是靠"改完必须过校验"限制。

### template：CUE文件

CUE 本来就是为“数据、约束、模板、验证和生成”设计的。它可以直接验证 YAML 文件，Template 可以是一个 CUE definition，而 Kernel Plan 仍然只是普通 YAML

CUE 的几个性质特别适合 `(template, schema)` 共同演化：

*   definition 可以默认关闭未知字段；
    
*   又可以在特定位置通过开放结构保留扩展空间；
    
*   支持 optional / required；
    
*   支持候选集合和默认值；
    
*   可以表达简单跨字段关系；
    
*   多个约束可以无顺序合并；
    
*   可以用 subsumption 思路判断新旧定义的兼容关系
    

例如 Template 可以保持接近这种密度：

```plaintext
#PipelinePlan: {
    spec_version!: string

    workload!: {
        input_dtype!:  "fp16" | "bf16" | "fp8"
        accum_dtype!:  "fp32" | "fp16"
        causal!:       bool
        variable_len?: bool
    }

    macro_pipeline!: {
        tile!: {
            q?:  int
            kv?: int
            m?:  int
            n?:  int
            k?:  int
        }

        dispatch!: "static" | "persistent" | "clc" | "custom"
    }

    novel_idea?: {
        ...
    }
}
```

它明显比完整 JSON Schema 紧凑。

优点

*   Template 简洁；
    
*   很适合逐步增加约束，而不必一开始把约束写完；
    
*   可以关闭稳定区域、开放创新区域；
    
*   Template 版本之间更容易做逻辑上的兼容比较；
    
*   Plan 仍然是普通 YAML，Agent 无法在实例中编程；
    
*   Template、额外约束和迁移补丁可以分文件组合，CUE 的合并结果与应用顺序无关
    

缺点

*   团队需要理解 CUE 的 unification、definition、open/closed 等概念；
    
*   Template 元数据和展示顺序需要额外约定；
    
*   Python 侧通常通过 `cue` CLI 或 Go API 交互，不如纯 JSON Schema 直接；
    
*   如果允许 Agent 使用 CUE 全部能力，Template 也可能逐渐变复杂。
    

**agent 编辑一个 schema 时，上下文里同时有三层，各司其职:**

1.  骨架：新 schema 从生成的空白问卷起步（从 CUE 导出字段树，enum 可选词和一行说明以 YAML 注释形式内联在每个待填项旁边）——填空而非创作；
    
2.  template 本体：常驻上下文，2k token 量级，回答"有哪些字段、哪些必填、词汇表是什么"；
    
3.  最近邻 exemplar：2–3 个同类 kernel 的已验收 plan，回答"这个字段在实际设计里长什么样、填到什么粒度"。
    

人类侧则用 CUE 自动生成的 JSON Schema + 编辑器补全，审查面是纯 YAML 的 plan。两个受众、一个真源、零重复维护。

### 26/09/01 schema的分区讨论：I

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/a/RNJroOxr3UGLN01e/302bc1676d69499e81c67343c3c43cf20521.png)

coder agent：在给定的明确硬件、明确算法任务、明确算子设计的情况下，遵照wiki和microbench的信息，严格的一代又一代进行迭代

human随时会介入和把握设计方向，目前预期是介于 vibe coding 和 自动loop 之间。

plan一定是位于所有编程语言之上的，200-300行以内的描述，最好可以被human看明白

进行

human不需要介入代码实现，完全的loop engineering

coder的语言是任意的，可以是TileLang、CuTeDSL、CUDA等暴露了warp-spec级别细粒度的语言

**第一，应该是contract区，也就是基本上不变，或者不期望发生变化的内容，无论是planer还是coder都没有资格修改，简单来说我认为就是一切和预期算法精度/硬件相关的，planer不应该探索、应当早期确定的设计：**

**具体想法 draft （schema的YAML文件的格式、记录内容、层级顺序）：**

**部分**

1）算法要求的部分algorithm IO。也就是，用类似于数学公式的语言，记录算子的核心流程；这个地方可能有公式。以及各种被操作数的预期大小（比如head dim等等可以被固定的）和意义等等。也就是纯数学的内容，是从外侧的视角看过去，算子作为黑盒，应当满足的contract。也就是纯数学目的的考虑。比如对于Fused MoE、Fused RMSnorm + ADD等等都应该有确切的、精简的数学表达，和最经典的Pytorch实现能够对应。甚至这里也可以把Pytorch本身作为一种“表达语言”，比数学公式更加确定一些，还可以和kernel的golden reference进行关联

2）硬件规格的部分hardware spec。简练的记录这一代硬件的参数，比如，计算单元的各种计算硬件和参数、访存单元的硬件和参数、存储单元的各种、指令单元的各种新特性，以及其他的各种精简的、这一张卡的特别之处。注意，这里应当精简，而不要变成“纯datasheet”，因为我们之后应该会提供一个microbench来记录所有的原语+吞吐速度，这里感觉写太多不合适。换句话说，应当是“在这张卡实现这个算子必须要考虑的硬件特性”。这里很容易膨胀，所以要好好考虑该怎么“抓住核心内容”。比如，“SM数量”和“TMA吞吐”这种东西我们应该放到wiki/microbench里面，而不应该记录到这里。而“引入TMEM”，“支持2CTA-pair”这种显著影响后续决策的，应该被考虑进来。

硬件

3）数学变换的部分math reshape。这个地方的特点在于，它属于非常高阶的数学优化技巧。它记录了在contract中，硬件和算法之间本征矛盾所带来的妥协和创新设计。这里有非常典型的例子，比如，Attention里面的online-softmax就是因为SMEM等资源限制、减少中间矩阵流量的目的，而设计出来的，也就是Flash Attention1、2、3、4里面很重要的online更新。注意，这里是纯算法的内容。也就是说，这个地方应当不需要任何硬件，可以用Pytorch的语言和数学公式描述出来。又比如，FA4里面提出的rescale跳过的设计，也应当属于这里（虽然这里有精度问题，我们可能平时不欢迎这种取巧，因此这一点被放在了contract区，因为涉及到了精度的妥协问题，但是我们绝大多数情况都不能接受为了加速而随意的跳过计算，因此就认为这种应该是“很早就定下来的”contract）；还比如，分块矩阵乘法，也就是在各种GEMM中，一般会把输出矩阵C按照Tile来划分，然后通过分块的思路来得到一个C\_tile，而避免对底层的“按列按行内外循环嵌套索引求GEMM”。这个部分的难度最高、收益最大。我觉得可能agent训练会天然的记住Attention、MoE算子的优化，但是如果有一些新奇的融合算子，那么这里就是agent首先应该思考的空间——怎么避免显式化中间值、通过分块、分步的方法来在数学上，削减算法实现的空间/时间复杂度。这里应当是纯数学的，hardware-aware的math reshape和optimize

**第二，应该是最重要的design区，也就是agent应当付出最高的努力，来进行调整、设计、思考、判断的内容，只有planer能修改，coder必须遵守的区域。分为不同层次：**

**部分**

1）任务划分的部分 macro tiling。建立在数学变化的基础上，我们已经可以把一个大的数学算法，分治成许多的单元。比如说，面对GEMM，agent就需要在这里考虑，C\_tile和SM/CTA之间的关系。它不需要思考warp，而是想清楚，这是要1CTA？2CTA？同时这里要怎么分配SM/CTA的关系？每个SM/CTA负责多大一块计算任务？这就好比给C\_tile分配一个SM，或者一个CTA。或者，给Attention里面的Q/O\_tile分配任务。这里就是最核心的事情，需要把硬件资源和数学计算算法给mapping起来。这里涉及到了，第一，怎么mapping，第二，怎么调度？比如CLC、persistent、static这种很直观的，也可能是更有趣的方法。但是在这里，一切的逻辑都贴近“tilelang”或者“triton”，把要计算的对象按照tile作为管理，把要负责的硬件作为CTA（最底层，还不能到warp/warp group）来分发，同时负责管理好数学对象/算法数据在物理对象/硬件单元上面的调度、衔接。比如考虑好wave，考虑好SM数量的情况，处理好尾波负载均衡、SM资源平衡等等。这里就是最重要的co-design了

2）计算流水的部分 micro pipeline。当上面的任务确定好了，那么这里负责已经把一个数学tile的划分好的任务、一个确定好的CTA Block的硬件，作为最高的研究平台，来研究在这里面，warp/warp group的行为。具体来说，他要描述清楚在这样的tile+小硬件单元上面，warp之间是怎么协作、怎么设计、怎么分配任务给确定好。这里需要规范的最底层细节，仅仅是到“warp”级别，而不需要深入到每个thread访问的element、thread和memory的layout等等。它的最核心的研究目标就是：通过分配warp-spec、异步逻辑、流水线pipeline排布顺序等等，最大化的重叠计算和访存，提高硬件的并行度，在不产生hang的dead-lock和读写竞争的情况下，设计出一个好的流水线。这个流水线里面最小的重复pace就是这个流水线的平均延时。我们可以理解为是在这个pace内，warp都在做不同的事情——流水线的最小不可分割单元；而pace之间，就是简单的重排+填入+排出。这里就是最重要的，类似于Gantt图的内容，根据microbench的结果，类似于通过 max(t1,t2,t3,...) 来围绕延时最高的瓶颈进行design。这个地方就像FA4在sm100发现SFU速度偏慢、FA3发现sm89的WGMMA速度很慢，来围绕这个进行设计。这里很有可能要协同前面的tiling协同设计；因为可能要在这里才意识到如何修改tile策略才能把瓶颈过程给尽可能的打满，比如FA4会凑出Qhigh和Qlow两个tile来一起计算，形成两个softmax WG。不过这个其实也是可以理解为micro pipeline的内容。只是说，这里的决策会反过来改变前面的内容，不过，这里的决定显然是，先定macro，再定micro，再改macro，最后定micro这样交替进行

写到这里 我感觉似乎拆分成两个也有些“粗”。一方面，macro tiling这边同时耦合了硬件-算法的资源调度、工作窃取、持久驻留等等，也会涉及到分块的工作，有一些混杂；而另一方面，我在想是不是漏掉了些什么设计？换句话说，kernel design的部分感觉似乎应当有三层比较符合直觉，可能这里缺了一些描述的同时也把一些描述不恰当的归为一类了。这里可能是最重要的，我觉得还需要一个点。因此上面的两个拆法也不一定是完全准确的。

**第三，应该是“缓冲”区，记录了planer认为这里应当被确定、但是不属于design部分的细节**

**吧**

可能涉及到了planer和coder的交织地带。目的是那些，可能会和上层设计约束耦合的底层细节。简单来说，在这个区域内，coder也有资格进行改写。但是这里的“重要性”其实不大，因为它可能有些游离于design的最高层设计之外，但是又对整个kernel有较重要的内容。我们要警惕agent可能会再次随意发挥、不断拓展的喜好。这里越多，反而说明这个kernel design越差——因为好的kernel设计应该更多是容易确定的，而不是遗留了大量的设计细节。所以未来这里的文字长度越长，我想评估的时候扣的分越多。但是我暂时没有想清楚哪些东西是planer和coder之间的模糊地带；对于明确应该coder决定的东西（比如，barrier要选择什么类型）不应该放在这里，但是mma指令要选择什么（因为这涉及到了乘法规格、硬件使用、调度分块），似乎就不得不在这里甚至是planer决定。

我想到了一些可能应当放在这里的内容，比如说

1）指令原语，必须在Blackwell用tcgen05.mma，而不能用mma.sync或者wgmma，必须在Hopper+用copy.bulk这种TMA的原语；总之是planer的design区域在设计的时候就已经考虑好使用的primitive。或者在这一代硬件上要SOTA所必须使用的primitive（简单来说，这个东西会影响design）。但是不能把所有的primitive都搬过来。我们认为只有和design本身相关的才值得考虑。剩下的都应该是coder的工作，进行探索

2）layout布局，这个地方可能更加麻烦；但是我在思考是不是有一些特殊的layout可能会和流水线设计本身耦合；就是如果要实现这个design、这个pipeline，那么就必须用这个特殊的布局格式。当然这个也不一定，因为我也没有想得特别清楚，甚至这个缓冲区是否是必要的也有待考量。

26/09/03

1）GDN Atrex bench的问题；flashinfer开源版本 -> 川悟的trace（有没有和atrex benchmark的题目匹配对齐）

2）跑通AKA优化

3）先做 planer + AKA 初版

![f6d63a56788017942f0cf1f8df0c84d6.jpeg](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/1aa6afbb-4e8c-4717-ad09-0b6e7c9ea5c4.jpeg)

26/09/06：

配置AKA+Agate+Atrex bench的迭代平台，基于睿博的支持多文件迭代版本，尝试从Flashinfer的开源fallback版本开始从零复现M64 GDN sm103 CuTeDSL

目前运行76轮，在10个典型shape上精度已经和川悟手调的M64 SOTA版本对齐，且9/10个shape速度更快，总体平均提速4.5%

**26/09/08：**

思考什么样的语言可以承载“warp-spec”的异步流水线的信息，简洁干净、容易记录和理解

OSDI 2026：Optimal Software Pipelining and Warp Specialization for Tensor Core GPUs

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/9511888e-97a9-4325-a90f-a88d868d241a.png)

**这个做法虽然看起来能分析，但是完全没办法迅速的get核心，AI去理解也会很慢**

**按角色分组的前驱表**

核心只需要这条约定：

> `**操作: [前驱操作……]**`**，表示所有前驱完成后，该操作才允许开始。**

按角色把这些条目分组，已经很接近 Excel 风格了：

```plaintext
kernel: fa4

warp_spec:
  version: 1
  inputs: [m_prev, y_prev]

  ops:
    tma:
      q:  []
      k0: []
      k1: [k0]

    umma:
      qk0: [q, k0]

    softmax:
      load_s0:     [qk0]
      max_update0: [load_s0, m_prev]
      alpha_ready0: [max_update0]
      exp0:       [alpha_ready0]

    correction:
      rescale0: [alpha_ready0, y_prev]
```

这是**局部示例，不是某个 FA4 版本的完整还原**：省略了 V、PV、其他迭代和 buffer 复用关系；`alpha_ready0` 在这里包含 alpha 的计算及发布，表示 correction 已可消费它。

这个片段明确表达了：`k0` 完成后，`k1` 搬运和 `qk0` 计算可以具备重叠条件；`alpha_ready0` 之后，`exp0` 和 `rescale0` 分别向下推进，correction 不必等整个 softmax 结束。**没有依赖表示不被这张图禁止重叠，不表示硬件必定同时执行。**

这里最重要的是：**角色分组不产生任何隐含的顺序边。** `q`、`k0` 的发射顺序没有在这个片段中记录，不能从它们的文本排列顺序推断。

这不是一个现成命名的行业标准，而是一个很小的项目 schema。不过其核心“节点→前驱集合”就是常见图表示；Python 标准库 `graphlib.TopologicalSorter` 接受的正是这种关系，分组展开后可以直接用于拓扑处理。

**为什么我更倾向这种写法**

每个节点只声明一次，角色和依赖都在附近，Agent 改一条关系不需要调整坐标，也不需要重排整张表。根操作用 `[]`，窗口外的状态放 `inputs`，含义也清楚。

以后增加属性，可以把单个条目展开，而不改变其余节点：

```plaintext
exp0:
  after: [alpha_ready0]
  latency_cycles: 1024
```

这里的 `1024` 只是格式示例，不是 FA4 实测值。项目约定短写 `[alpha_ready0]` 等价于长写 `{after: [alpha_ready0]}` 即可。

**建议用** `**load_s0**`**、**`**alpha_ready0**` **这样的稳定 ID，而不是纯数字编号。** 增删一步不应导致后面的编号全部变化。

**比语法更重要的三个语义问题**

1. 发射顺序，不能误写成完成顺序

这是 warp-spec 最容易画错的地方。异步指令可以先返回控制权，工作随后才完成；PTX 也明确区分发起与完成观测机制。

因此：

**“同一 warp 先发射 A，再发射 B”不等于“A 完成后，B 才能开始”。**

需要表达这种区别时，把关键操作拆成 `a_issue`、`a_done` 两个节点：后续发射依赖 `a_issue`，数据消费者依赖 `a_done`。不必把所有操作都拆开，只拆那些确实存在“发射后继续干别的事”的边界。

这也是为什么不建议把 `TMA: [Q, K0, K1]` 自动解释成一条完成依赖链。**这串列表只说明任务分配，还没有说清楚顺序的类型。**

2. 不仅要记录“数据 ready”，还要记录“buffer 可以再次写”

如果只记录 `load_k0 → qk0`，图会漏掉复用约束。对于双缓冲，同一个物理槽再次装载 `K2` 前，必须确认旧的 `K0` 已经不再被消费者使用。CUTLASS 的 producer acquire / consumer release 正是在表达这类关系。

所以需要类似：

`**k0_read_done → load_k2**`

这里应使用“真正允许复用”的事件，而不是随意替换成某个更早或更晚的操作。它会直接影响你分析出来的重叠空间。

初版可以把这些关系都放进前驱表；需要区分原因时，再给边增加 `data`、`order`、`reuse` 等类型。

3. 展开后的图是 DAG，折叠后的循环模板不一定是

把前四个迭代展开，使用 `k0、k1、k2、k3`，很适合核对启动阶段。

但长期要避免重复记录，可以使用编译器依赖图里的 **iteration distance**：例如一条 `u → v, distance: 2`，表示 `u[i] → v[i+2]`，不是同一迭代里的回边。Twill 使用的循环依赖模型就包含这种跨迭代信息

### tile pipeline的YAML语言表述

#### 1. 核心设计

**一个 lane 对应一个 warp 或 warp group，在同一处记录：它是什么角色、按什么顺序做事、每个动作在等谁。**

目标是用最少的文字表达 kernel 的流水线依赖，而不是描述执行代码或绘图坐标。这些是我们约定的语法规则，循环和依赖不是 YAML 自带的功能。

每个 lane 固定分成三块：

| 字段 | 内容 |
| --- | --- |
| `info` | 角色属性：同步/异步模式 `mode`、顺序约束 `order`、动作词表 `verbs`，以及可选的每线程寄存器数 `registers`。 |
| `pipe` | 动作序列：`prologue` 写开头，`steady` 写重复部分，`epilogue` 写收尾。 |
| `edge` | 当前 lane 的动作所需的额外前驱，分为 `data / reuse / sync`。 |

例如：

```plaintext
TMA warp:
  info:
    mode: async
    order: [issue]
    verbs: [load]

  pipe:
    prologue: [load:K(0), load:Q_high, load:Q_low]
    steady:   [load:V(i), load:K(i+1)]
    epilogue: [load:V(N-1)]

  edge:
    reuse:
      load:V(i+1): [mma:Q_low@K(i)]
      load:K(i+2): [mma:P_low(i)@V(i)]
```

#### 2. 语法规则

**动作命名：**`**动词:对象_说明(序号)**`**。** 例如 `load:S_high(i)`、`compute:alpha_high(i)`；矩阵乘法写成 `mma:P_high(i)@V(i)`。动作名称同时作为引用 ID，展开后必须唯一。名称负责可读性，**不会自动推导数据依赖**。

**流水线只写“开头＋循环＋收尾”。** `prologue`、`epilogue` 各展开一次；没有内容就写 `[]`。`steady` 的 `i` 从 0 开始，只有整轮动作的索引都落在 `0..N-1` 内才展开。因此，含 `K(i+1)` 的循环自然少一轮，不必单独声明循环范围。

**列表顺序有意义，但发射顺序不等于完成等待。** 同步 lane 按动作完成顺序推进；异步 lane 的 `order: [issue]` 只约束发射先后，`[issue, done]` 则分别约束发射顺序和完成顺序，**不额外加入“前一个完成后才能发射下一个”**。这里的同步模式不妨碍不同 lane 并行。

**依赖统一写成** `**当前动作: [前驱动作……]**`**，表示所有前驱均须就绪。** 左侧必须属于当前 lane 的 `pipe`，右侧可以来自其他 lane；同一动作命中多条规则时，前驱取并集。三类名称只是区分等待原因：

| 分类 | 等待什么 |
| --- | --- |
| `data` | 操作数或计算结果就绪。 |
| `reuse` | 旧内容使用结束，允许覆盖存储。 |
| `sync` | 额外的同步协议放行。 |

含 `i` 的依赖也按索引窗口展开；越界实例整条跳过，窗口内引用不存在则报错，不能静默忽略。

**这样压缩的是重复描述，不是必要语义：** lane 内顺序只写一次，循环体只写一次，额外依赖放在等待方附近；high/low 属于不同 WG，就分别明确记录。

#### 3. 如何渲染成 DAG

```plaintext
YAML
  → Python 展开循环和依赖
  → 检查动作归属、重复 ID、缺失引用和有向环
  → 构建事件 DAG：异步动作拆成 issue / done
  → 输出 DOT：按 lane 分组，按依赖类型设置线型
  → Graphviz 的 dot 自动布局
  → SVG / PNG
```

**Python 负责语义，Graphviz 负责排版。** YAML 不记录节点位置；脚本使用统一的分组和样式规则，不需要逐节点摆放。

最终得到的是**带 lane 分组的依赖图，不是实际时间线**：节点间距不代表耗时，同一高度也不代表同时执行。

#### 4. 具体case

目前的，在Warp-spec这边的YAML效果：

```yaml
kernel:
  name: FA4
  arch: B200-SM100
  dtype: bf16
  head_dim: 128
  tile: [128, 128]
  q_stage: 2
  cta_group: 1
  causal: false
  kv_smem_slots: 3

warp_spec:
  version: 3
  parameters: {N: 4}

  # steady 和含 i 的 edge：i 从 0 起，索引须在 0..N-1；越界整轮/整边不展开。
  # 固定项越界、窗口内引用缺失、或 edge 左侧不属于本 lane，直接报错。
  # pipe 自带 lane 内顺序；edge 的左侧属于本 lane，右侧是它等待的前驱。
  # data=结果就绪；reuse=允许覆写；sync=额外协议放行。
  # 沿用操作级模型：P 整块发布，未展开 split-P 和最终输出阶段。
  lanes:
    TMA warp:
      info:
        mode: async
        order: [issue]
        verbs: [load]
      pipe:
        prologue: [load:K(0), load:Q_high, load:Q_low]
        steady:   [load:V(i), load:K(i+1)]
        epilogue: [load:V(N-1)]
      edge:
        # 此配置的 K/V 共用 3 个 SMEM 槽，覆盖前必须等旧数据读完。
        reuse:
          load:V(i+1): [mma:Q_low@K(i)]
          load:K(i+2): [mma:P_low(i)@V(i)]

    UMMA warp:
      info:
        mode: async
        order: [issue, done]
        verbs: [mma]
      pipe:
        prologue: [mma:Q_high@K(0), mma:Q_low@K(0)]
        steady:   [mma:P_high(i)@V(i), mma:Q_high@K(i+1),
                   mma:P_low(i)@V(i),  mma:Q_low@K(i+1)]
        epilogue: [mma:P_high(N-1)@V(N-1), mma:P_low(N-1)@V(N-1)]
      edge:
        data:
          mma:Q_high@K(i):         [load:Q_high, load:K(i)]
          mma:Q_low@K(i):          [load:Q_low, load:K(i)]
          mma:P_high(i)@V(i):      [store:P_high(i), load:V(i)]
          mma:P_low(i)@V(i):       [store:P_low(i), load:V(i)]
          # 后续轮的 PV 还要等历史 Y 完成 correction；首轮不需要。
          mma:P_high(i+1)@V(i+1):  [rescale:Y_high(i+1)]
          mma:P_low(i+1)@V(i+1):   [rescale:Y_low(i+1)]

    Softmax WG0:
      info:
        mode: sync
        order: [done]
        verbs: [load, rowmax, compute, exp2, store, rowsum]
      pipe:
        prologue: []
        steady:   [load:S_high(i), rowmax:S_high(i), compute:alpha_high(i),
                   exp2:S_high(i), store:P_high(i), rowsum:L_high(i)]
        epilogue: []
      edge:
        data:
          load:S_high(i): [mma:Q_high@K(i)]
        # stats 的放行约束，不是 rowsum 的数学输入。
        sync:
          rowsum:L_high(0):   [wait:alpha_high(0)]
          rowsum:L_high(i+1): [rescale:Y_low(i+1)]

    Softmax WG1:
      info:
        mode: sync
        order: [done]
        verbs: [load, rowmax, compute, exp2, store, rowsum]
      pipe:
        prologue: []
        steady:   [load:S_low(i), rowmax:S_low(i), compute:alpha_low(i),
                   exp2:S_low(i), store:P_low(i), rowsum:L_low(i)]
        epilogue: []
      edge:
        data:
          load:S_low(i): [mma:Q_low@K(i)]
        sync:
          rowsum:L_low(i):   [rescale:Y_high(i+1)]
          rowsum:L_low(N-1): [release:stats_low(N-1)]

    Correction WG:
      info:
        mode: sync
        order: [done]
        verbs: [wait, rescale, release]
      pipe:
        # 首轮只消费通知；这里的 alpha(0) 是初始化通知的抽象名称。
        prologue: [wait:alpha_high(0), wait:alpha_low(0)]
        steady:   [rescale:Y_high(i+1), rescale:Y_low(i+1)]
        epilogue: [release:stats_low(N-1)]
      edge:
        data:
          wait:alpha_high(0):  [compute:alpha_high(0)]
          wait:alpha_low(0):   [compute:alpha_low(0)]
          # rescale 包含检查、按需缩放、发布就绪；动作仍属于 Correction。
          rescale:Y_high(i+1): [compute:alpha_high(i+1), mma:P_high(i)@V(i)]
          rescale:Y_low(i+1):  [compute:alpha_low(i+1), mma:P_low(i)@V(i)]
```

**第一，同一个动作命中多条依赖规则时，取前驱的并集。** 例如 `mma:P_high(1)@V(1)` 同时需要 `store:P_high(1)`、`load:V(1)` 和 `rescale:Y_high(1)`，后面的规则不会覆盖前面的规则。

**第二，Correction 用** `**(i+1)**` **表达“从第 1 轮开始修正”。** 这样既不需要写循环范围，也不会凭空产生第 0 轮 rescale。Softmax 的全 `(i)` 列表则自然覆盖全部四轮。

**检查脚本（render.py）：**

```python
#!/usr/bin/env python3
"""Render a small warp-specialization YAML into a checked event DAG and DOT.

Requirements: Python >= 3.9, PyYAML, Graphviz executable (dot).
No model calls, network requests, kernel-specific graph edges or image editing.
"""
from __future__ import annotations
import argparse
import ast
import re
from dataclasses import dataclass
from graphlib import TopologicalSorter, CycleError
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import yaml


class UniqueLoader(yaml.SafeLoader):
    """Reject duplicate mapping keys rather than silently replacing entries."""


def unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


@dataclass(frozen=True)
class Event:
    id: str
    op: str
    lane: str
    phase: str


# Bound expansion; only N, i, integer constants and +/- are permitted.
LIMIT = 20000


def check_keys(value, allowed, where):
    if not isinstance(value, dict):
        raise ValueError(f'{where}: expected a mapping')
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f'{where}: unsupported keys {sorted(map(str, unknown))}')


def integer(expression, env):
    """Safe integer expressions: literals, parameter names, + and -. No eval."""
    text = str(expression)
    if len(text) > 128:
        raise ValueError('Integer expression is too long')
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.Name) and type(env.get(node.id)) is int:
            return env[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            a, b = visit(node.left), visit(node.right)
            return a + b if isinstance(node.op, ast.Add) else a - b
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            a = visit(node.operand)
            return a if isinstance(node.op, ast.UAdd) else -a
        raise ValueError(f'Unsupported integer expression: {text!r}')
    try:
        return visit(ast.parse(text.strip(), mode='eval').body)
    except (SyntaxError, RecursionError) as exc:
        raise ValueError(f'Invalid integer expression: {text!r}') from exc


# All parenthesized numbers are iteration indices, not physical buffer slots.
# N is global; i is local to each complete steady block or dependency rule.
INDEX = re.compile(r'\(([^()]*)\)')
TEMPLATE = re.compile(
    r'^[a-z][a-z0-9]*:[A-Za-z_][A-Za-z_0-9]*(?:\([^()]+\))?'
    r'(?:@[A-Za-z_][A-Za-z_0-9]*(?:\([^()]+\))?)*$'
)


def inspect_operation(text, params):
    """Validate syntax before bounds filtering, including inactive boundary rules."""
    if not isinstance(text, str) or not TEMPLATE.fullmatch(text):
        raise ValueError(f'Expected verb:noun(index), with no space after colon: {text!r}')
    uses_i = False
    for expression in INDEX.findall(text):
        integer(expression, dict(params, i=0))
        tree = ast.parse(expression.strip(), mode='eval')
        uses_i |= any(isinstance(n, ast.Name) and n.id == 'i' for n in ast.walk(tree))
    return INDEX.sub('()', text), uses_i


def instantiate(text, env):
    """None means index out of domain, never an unknown operation or syntax error."""
    values = [integer(expression, env) for expression in INDEX.findall(text)]
    if any(not 0 <= value < env['N'] for value in values):
        return None
    iterator = iter(values)
    return INDEX.sub(lambda _: f'({next(iterator)})', text)


def compile_templates(document, overrides=None, trace=None):
    """Version 3 lane-local syntax -> the unchanged concrete v1 event graph."""
    if not isinstance(document, dict) or 'warp_spec' not in document:
        raise ValueError('Expected kernel and warp_spec mappings')
    spec = document['warp_spec']
    check_keys(spec, {'version', 'parameters', 'lanes'}, 'warp_spec')
    if spec.get('version') != 3:
        raise ValueError('Expected warp_spec.version: 3')
    check_keys(spec.get('parameters'), {'N'}, 'parameters')
    params = dict(spec['parameters'])
    for name, value in (overrides or {}).items():
        if name not in params:
            raise ValueError(f'Unknown parameter override: {name}')
        params[name] = value
    if type(params.get('N')) is not int or not 1 <= params['N'] <= LIMIT:
        raise ValueError(f'N must be an integer in 1..{LIMIT}')
    lanes = spec.get('lanes')
    check_keys(lanes, lanes if isinstance(lanes, dict) else (), 'lanes')
    if not lanes:
        raise ValueError('lanes must be nonempty')
    flat = {'version': 1, 'lanes': {}, 'async_lanes': [], 'completion_order': [],
            'after': {}, 'reuse_after': {}, 'sync_after': {}}
    report = {'index_domain': [0, params['N'] - 1], 'lanes': {}, 'rules': []}
    families, owners = {}, {}
    for lane, definition in lanes.items():
        if not isinstance(lane, str) or not lane:
            raise ValueError('Lane names must be nonempty strings')
        check_keys(definition, {'info', 'pipeline', 'depends'}, lane)
        info = definition.get('info', {})
        check_keys(info, {'mode', 'verbs', 'completion_order', 'registers_per_thread'}, f'{lane}.info')
        mode, verbs = info.get('mode', 'sync'), info.get('verbs')
        if mode not in ('sync', 'async'):
            raise ValueError(f'{lane}: mode must be sync or async')
        if not isinstance(verbs, list) or not verbs or not all(
            isinstance(v, str) and re.fullmatch(r'[a-z][a-z0-9]*', v) for v in verbs
        ) or len(verbs) != len(set(verbs)):
            raise ValueError(f'{lane}: provide distinct verbs')
        ordered = info.get('completion_order', False)
        if type(ordered) is not bool or (ordered and mode != 'async'):
            raise ValueError(f'{lane}: completion_order must be boolean and requires async mode')
        registers = info.get('registers_per_thread')
        if registers is not None and (type(registers) is not int or registers <= 0):
            raise ValueError(f'{lane}: registers_per_thread must be a positive integer or null')
        pipeline = definition.get('pipeline')
        check_keys(pipeline, {'prologue', 'steady', 'epilogue'}, f'{lane}.pipeline')
        families[lane] = set()
        parts = {}
        for part in ('prologue', 'steady', 'epilogue'):
            items = pipeline.get(part)
            items = [] if items is None else items
            if not isinstance(items, list):
                raise ValueError(f'{lane}.{part}: expected a list or an empty value')
            uses_i = False
            for text in items:
                family, symbolic = inspect_operation(text, params)
                families[lane].add(family)
                uses_i |= symbolic
                if text.split(':', 1)[0] not in verbs:
                    raise ValueError(f'{lane}: undeclared verb in {text!r}')
            if part != 'steady' and uses_i:
                raise ValueError(f'{lane}.{part}: i is only allowed in steady')
            if part == 'steady' and items and not uses_i:
                raise ValueError(f'{lane}.steady: a nonempty steady must contain i')
            parts[part] = items
        sequence, active = [], []
        for part, items in parts.items():
            contexts = (dict(params, i=i) for i in range(params['N'])) if part == 'steady' and items else [params]
            for env in contexts:
                block = [instantiate(text, env) for text in items]
                if any(op is None for op in block):
                    if part != 'steady':
                        raise ValueError(f'{lane}.{part}: index outside 0..N-1')
                    continue  # Skip the entire boundary block, never individual operations.
                if part == 'steady' and items:
                    active.append(env['i'])
                for op in block:
                    if op in owners:
                        raise ValueError(f'Duplicate operation ID: {op}')
                    owners[op] = lane
                    sequence.append(op)
                    if len(owners) > LIMIT:
                        raise ValueError('Expanded operation limit exceeded')
        if not sequence:
            raise ValueError(f'{lane}: the expanded pipeline is empty')
        flat['lanes'][lane] = sequence
        report['lanes'][lane] = {'steady_i': active, 'operations': len(sequence)}
        if mode == 'async':
            flat['async_lanes'].append(lane)
        if ordered:
            flat['completion_order'].append(lane)

    all_families = set().union(*families.values())
    for lane, definition in lanes.items():
        groups = definition.get('depends') or {}
        check_keys(groups, {'data', 'reuse', 'sync'}, f'{lane}.depends')
        for kind, rules in groups.items():
            rules = {} if rules is None else rules
            check_keys(rules, rules if isinstance(rules, dict) else (), f'{lane}.depends.{kind}')
            section = {'data': 'after', 'reuse': 'reuse_after', 'sync': 'sync_after'}[kind]
            for target, sources in rules.items():
                if not isinstance(sources, list) or not sources:
                    raise ValueError(f'{target}: predecessors must be a nonempty list')
                target_family, uses_i = inspect_operation(target, params)
                if target_family not in families[lane]:
                    raise ValueError(f'{lane}: target not declared in this pipeline: {target}')
                for source in sources:
                    family, symbolic = inspect_operation(source, params)
                    uses_i |= symbolic
                    if family not in all_families:
                        raise ValueError(f'{lane}: unknown predecessor family: {source}')
                active, skipped = [], []
                for i in range(params['N']) if uses_i else [None]:
                    env = dict(params, i=i) if i is not None else params
                    block = [instantiate(text, env) for text in [target, *sources]]
                    if any(op is None for op in block):
                        if not uses_i:
                            raise ValueError(f'{lane}: fixed dependency index outside 0..N-1: {target}')
                        skipped.append(i)
                        continue
                    dst, *preds = block
                    if owners.get(dst) != lane:
                        raise ValueError(f'{lane}: dependency target not in this pipeline: {dst}')
                    for src in preds:
                        if src not in owners:
                            raise ValueError(f'{lane}.{dst}: unknown predecessor: {src}')
                    flat[section].setdefault(dst, []).extend(preds)
                    active.append(i)
                report['rules'].append({'lane': lane, 'kind': kind, 'target': target,
                                        'applied_i': active, 'boundary_skipped_i': skipped})
    if trace is not None:
        trace.update(report)
    return {'kernel': document.get('kernel', {}), 'warp_spec': flat}, params


def expand(document):
    spec = document['warp_spec']
    if spec.get('version') != 1:
        raise ValueError('Expected warp_spec.version: 1')
    lanes = spec['lanes']
    if not isinstance(lanes, dict) or not lanes:
        raise ValueError('lanes must be a nonempty mapping')
    async_lanes = set(spec.get('async_lanes', []))
    ordered = set(spec.get('completion_order', []))
    if not async_lanes <= set(lanes) or not ordered <= async_lanes:
        raise ValueError('Unknown async lane, or completion_order is not an async lane')
    ops, events, entry, done = {}, {}, {}, {}
    for lane, sequence in lanes.items():
        if not isinstance(sequence, list) or not sequence:
            raise ValueError(f'{lane}: expected a nonempty operation list')
        for op in sequence:
            if not isinstance(op, str) or not op:
                raise ValueError(f'{lane}: operation IDs must be nonempty strings')
            if op in ops:
                raise ValueError(f'Duplicate operation ID: {op}')
            ops[op] = lane
            phases = ('issue', 'done') if lane in async_lanes else ('done',)
            for phase in phases:
                eid = f'{op}::{phase}'
                events[eid] = Event(eid, op, lane, phase)
            entry[op] = f'{op}::{phases[0]}'
            done[op] = f'{op}::done'

    edges = set()
    def add(src, dst, kind):
        if src == dst:
            raise ValueError(f'Self-dependency: {src}')
        edges.add((src, dst, kind))

    for lane, sequence in lanes.items():
        for op in sequence:
            if lane in async_lanes:
                add(entry[op], done[op], 'async')
        for a, b in zip(sequence, sequence[1:]):
            add(entry[a] if lane in async_lanes else done[a], entry[b], 'program')
            if lane in ordered:
                add(done[a], done[b], 'completion')

    for section, kind in [('after', 'data'), ('reuse_after', 'reuse'), ('sync_after', 'sync')]:
        values = spec.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f'{section} must be a mapping')
        for op, predecessors in values.items():
            if op not in ops:
                raise ValueError(f'{section}: unknown operation {op}')
            if not isinstance(predecessors, list):
                raise ValueError(f'{section}.{op}: expected a list')
            for pred in predecessors:
                if pred not in ops:
                    raise ValueError(f'{section}.{op}: unknown predecessor {pred}')
                add(done[pred], entry[op], kind)

    predecessors = {eid: set() for eid in events}
    for a, b, _ in sorted(edges):
        predecessors[b].add(a)
    try:
        order = list(TopologicalSorter(predecessors).static_order())
    except CycleError as exc:
        raise ValueError(f'Cycle detected: {exc.args[1]}') from exc
    return lanes, ops, events, sorted(edges), predecessors, order


def q(s):
    return json.dumps(str(s), ensure_ascii=False)


# Appearance belongs to the renderer, not to kernel metadata.
STYLES = {
    'async': ('#718096', 'solid', 1.0),
    'program': ('#8291A1', 'dashed', 1.0),
    'completion': ('#A0AEC0', 'dotted', 1.0),
    'data': ('#244866', 'solid', 1.3),
    'reuse': ('#A16112', 'dashed', 1.2),
    'sync': ('#7752A3', 'dashed', 1.2),
}
FILLS = ['#E7F0F7', '#E5F0EB', '#FAEBDD', '#F7E4EA', '#ECE6F6']


def dot_text(document, lanes, events, edges, params):
    k = document['kernel']
    title = " | ".join(str(k[x]) for x in ("name", "arch", "dtype") if x in k)
    if params:
        title += " | " + ", ".join(f"{key}={value}" for key, value in params.items())
    out = ['digraph WarpDAG {',
           f'graph [rankdir=TB, compound=true, newrank=true, bgcolor="white", pad=0.25, nodesep=0.18, ranksep=0.24, label={q(title + chr(10) + 'Operation-level event DAG | NO TIME SCALE' + chr(10) + 'Solid: data / async completion   Dashed gray: program order' + chr(10) + 'Dotted: completion order   Orange: reuse   Purple: synchronization')}, labelloc=t, fontname="DejaVu Sans", fontsize=14];',
           'node [shape=box, style="rounded,filled", fontname="DejaVu Sans", fontsize=10, margin="0.07,0.04", color="#BAC6D1"];',
           'edge [arrowsize=0.55];']
    for idx, lane in enumerate(lanes):
        out.append(f'subgraph cluster_{idx} {{ label={q(lane)}; fontname="DejaVu Sans"; color="#D9E0E8";')
        for event in events.values():
            if event.lane == lane:
                label = event.op + ('\nISSUE' if event.phase == 'issue' else '\nREADY' if f'{event.op}::issue' in events else '')
                out.append(f'{q(event.id)} [label={q(label)}, fillcolor={q(FILLS[idx % len(FILLS)])}];')
        out.append('}')
    for a,b,kind in edges:
        color,style,width=STYLES[kind]
        out.append(f'{q(a)} -> {q(b)} [color={q(color)}, style={q(style)}, penwidth={width}, tooltip={q(kind)}];')
    out.append('}')
    return '\n'.join(out)+'\n'


def run_graphviz(dot_path, base):
    engine = 'dot'
    exe=shutil.which(engine)
    if not exe:
        raise RuntimeError(f'{engine} not found; install Graphviz and add it to PATH')
    outputs=[]
    for ext in ('svg','png'):
        target=Path(str(base)+'.'+ext)
        args=[exe]
        args += [f'-T{ext}', str(dot_path), '-o',str(target)]
        proc=subprocess.run(args, capture_output=True, text=True)
        if proc.returncode:
            raise RuntimeError(proc.stderr or f'{engine} failed')
        if proc.stderr.strip(): print(proc.stderr.strip(),file=sys.stderr)
        outputs.append(target)
    return outputs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('yaml_file',type=Path)
    parser.add_argument('--out',type=Path,default=Path('out'))
    parser.add_argument('--set', action='append', default=[], metavar='NAME=INT')
    args=parser.parse_args()
    try:
        raw=args.yaml_file.read_bytes()
        document=yaml.load(raw,Loader=UniqueLoader)
        overrides = {}
        for item in args.set:
            name, value = item.split('=', 1)
            overrides[name] = int(value)
        trace = {}
        concrete, params = compile_templates(document, overrides, trace)
        lanes,ops,events,edges,pred,order=expand(concrete)
        args.out.mkdir(parents=True,exist_ok=True)
        dot=args.out/'dag.dot'
        dot.write_text(dot_text(concrete,lanes,events,edges,params),encoding='utf-8')
        run_graphviz(dot,args.out/'dag')
        (args.out/'inference.json').write_text(json.dumps(trace, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
        (args.out/'expanded.yaml').write_text(yaml.safe_dump(concrete, sort_keys=False, allow_unicode=True), encoding='utf-8')
        expanded={'events':[vars(e) for e in events.values()],
                  'edges':[{'from':a,'to':b,'kind':kind} for a,b,kind in edges]}
        (args.out/'events.json').write_text(json.dumps(expanded,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        report={'operations':len(ops),'events':len(events),'edges':len(edges),'acyclic':True,
                'parameters':params,
                'yaml_sha256':hashlib.sha256(raw).hexdigest(),
                'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'graphviz':subprocess.run(['dot','-V'],capture_output=True,text=True).stderr.strip(),
                'pyyaml':yaml.__version__,'python':sys.version.split()[0]}
        (args.out/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(report,indent=2))
    except (KeyError,ValueError,TypeError,OSError,yaml.YAMLError,RuntimeError) as exc:
        parser.exit(1,f'Error: {exc}\n')


if __name__=='__main__':
    main()

```

流程仍然是：

```plaintext
lane-local YAML
    → 按统一边界规则展开
    → 检查动作归属、引用、重复 ID 和环
    → issue / done 事件 DAG
    → DOT
    → Graphviz SVG / PNG
```

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/09750d35-ba40-4cbf-ac92-75ff4d1561c8.png)

DAG在Graphviz的渲染效果，已经接近了预期的流水线样式：

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/3bbbbd5e-0f92-4c86-b200-c0bda1af4a42.png)

### 26/09/09 schema的分区讨论：II

**核心的原则：奥卡姆剃刀原理，如果他不是必要的，那就去掉它，不要新增冗余的假设、文字来描述。用最少的语言描述kernel的insight是最核心的需求。less is more**

**预期的分层方法：**

**Contract**区，包括algorithm requirement，以及 hardware specific，这是planer和coder都不可修改的，纯数学要求，纯硬件规格的内容。

1）algorithm requirement：算法要求的计算I/O、从外向内作为黑盒抽象的算子定义。包括：math definition，也就是数学语义的公式表达、数学记号、特殊语义；输入输出和内部涉及的变量命名、意义状态；基于Pytorch等实现的golden reference的oracle（通过文件索引地址来记录，精简代码）；代表性的shape，输入的workload，目标优化的指标latency/throughput、精度的容忍范围等特殊要求和target；

2）hardware specific：硬件平台的具体信息。硬件的具体性能参数和指令集手册（wiki，地址索引）；microbench关键计算/访存指令的细粒度延时吞吐测试结果（wiki，地址索引）；该schema的design设计时所利用的硬件和特性；

**Design**区，包括math reshape，work schedule，tile pipeline三个部分，是设计最核心的三段内容，是planer最需要聚焦的、按照层次排序，优化起来由难到易、但是收获由多到少的地方，但是coder不能修改：

1）math reshape：算法为了在该硬件实现，而对数学流程本身实现的优化和技巧。一般是较为高阶的数学技巧，记录了因为contract中硬件和算法之间本征矛盾所带来的妥协和创新设计。例如，Attention的online-softmax、动态阈值的rescale跳过、GEMM的分块矩阵乘法、计算步骤的融合等，优化存储/计算等等的复杂度；主要涉及到了对数学计算先后顺序的变化、乘法精度/累加精度等优化，一般可能会涉及到计算精度的不同程度的妥协；

2）work schedule：数学对象与硬件资源的映射、分配、调度。首先涉及到整个计算过程中对于logical work unit的分治，也就是完整计算流程中的可分割单元任务、分割的方向和顺序。然后还涉及到具体的，每个logic unit映射到的execution group是什么？SM/CTA/CTA pair/cluster等等。以及具体的分发任务的调度策略，负载平衡、任务窃取、分桶计算等等；以及跨logic unit的merge和reduction方法。这里我想了一下子分区的名字大概可能是：decomposition，mapping、schedule、coordination（这个名字可能还要再考虑考虑）

3）tile pipeline：在已经被分治后的logic unit、execution group的层级之内的具体计算流程，也就是完成一个tile计算对应的微流水。这里相对不太好界定子分类，但是大体上，肯定是

包括了role角色划分的warp/warp group的warp-spec（最重要，任务进一步的拆分和细化），

然后是：prologue，也就是流水线的填入的计算顺序，

steady，也就是最重要的一个最单元的pace推进的逻辑工作（可以以第i个pace为例将i-1和i+1），

以及epilogue，流水线排出的顺序。对于tile pipeline的表述，我们现在采用了一个YAML的方式来实现，记录了每一个lane（warp）的工作内容，依赖关系。这个小部分我们已经有一些小的实现能独立的画出来并且自动渲染成DAG图了

对于这个地方我现在非常关注这三个，这是核心中的核心，而这里面该怎么“表达”、“描述”，是最重要的。我的核心是，我们要保证信息本身首先是精简的，也就是足够描述清楚kernel设计的最核心的思路、内容、方法。而且不能说那种一连串下来写了大量的伪代码、大量的看起来有用，实际上人也不会看、懒得看、没办法审计的东西。也就是说我觉得要“尽可能提炼信息熵”，高度浓缩。但具体的策略可能还要思考。在这三个部分的最上方都有一个thesis的字段，来用最简单的语言记录这个东西的设计主张和目的，仅仅描述这个kernel的这一段为什么要这样做。尽可能的简洁（比如有字数上限）

**Realization**区，分为require、perfer和forbid，相当于是涉及到plan能否真正work的关键primitive原语（比如必须用tcgen05，不能用mma/wgmma）或者其他的必要建议。这是planer和coder的过渡区，planer在这里把一些和非设计的、但是实现所必要的信息、建议的方法记录在此，而coder也可以根据实验的结果在这里反馈，

1）require：design设计所必须要符合的内容，类似于lowering的hints，但是要尽可能的简短记录这里必要的东西，比如原语的使用。简单来说的原则是：这里记录的，应该会“影响上面design区的设计”，而不是“被上面的design区所影响”。

2）forbid：design设计所坚决排除的方案，这里不是把所有的“不可行路径”都拉过来写上去，而是只记录那些“planer在思考require的时候考虑到、但是放弃的”，而且“这些内容虽然实现起来简单，但是会走弯路和白费力气，哪怕写出来了都没有用”。但是我觉得这里应该尽可能留空吧。毕竟，less is more，不要额外添加复杂度。

3）perfer：planer考虑的、建议的设计，这部分会和planer的design有一定的耦合，需要在可能的情况下优先实现；除非明确无法实现。这也是coder唯一能够反馈的位点（用证据说明），是coder的选择自由

判断 primitive 是否进入 plan，可以使用反事实测试：如果 Coder 替换这一 primitive，Design的核心设计是否仍然成立？如果成立，留给 Coder；如果不成立，进入 realization

9/14：

### work schedule的YAML语言表述

总的来说，分成了三个部分：

1）decomposition：任务在数学/算法层面的分治。通过partition描述数学对象的工作划分，通过worktile描述每个任务的具体信息，通过target描述每个任务的预期输出和格式

2）mapping：任务在编程对象/物理硬件的映射。通过worker/placement给出一个具体的worker的资源和形式，数量和范围

3）schedule：分治后的子任务（worktile），如何在具体的硬件对象（worker）上面进行任务调度。包括计算的先后顺序order，以及任务的分配方式assignment（动态/静态等等）

初版的草稿示例（有很多问题，表达不完备或者不合理）

```yaml
work_schedule:
  thesis: "双 Q 共用 KV，有限 CTA 按固定步长接续任务。"

  partition:
    batch: {extent: batch_count, per_task: 1}
    head:  {extent: head_count, per_task: 1}
    query: {extent: query_length, per_task: 256}
    kv:    {extent: kv_length, per_task: all}

  within_task:
    query: {block: 128, names: [Q_high, Q_low]}
    kv: {block: 128, order: descending, shared_by: [Q_high, Q_low]}
    tail: {query: clip, kv: mask}

  result:
    tensors: [O, LSE]
    region: task_query_rows
    kind: final

  placement:
    unit: CTA
    units_per_task: 1
    grid: ["min(sm_count, task_count)", 1, 1]

  assignment:
    fastest_first: [query, head, batch]
    direction: ascending
    first_index: cta_id
    index_step: cta_count
    index_limit: task_count
```

核心：worker\_assignment

**它应该描述一个 worker 获配的工作序列，以及这个序列如何被补充、何时处理完毕；而不只是“第一项是什么、下一项怎么算”。**

把原来的：

```plaintext
first: ...
next: ...
end: ...
```

改成：

```plaintext
initial: ...   # 初始获配的工作序列
refill: ...    # 是否从别处补充工作
exit: ...      # worker 的退出条件
```

一个 worker 已经获配了 `0、8、16、24`，那么从 8 走到 16，只是在消费自己的序列；不需要再用一个“领取机制”描述它。只有从共享工作池、待启动 worker 等来源取得额外工作，才属于 `refill`。

这个区分比 `persistent/static/dynamic` 更接近真正的设计决定。

#### 1. 核心思想

**Work schedule 是一张设计决策表，描述：计算如何分解、由谁执行、怎样分配。** 它不描述线程级实现，也不重复 tile pipeline 的动作、依赖和缓冲协议。

两个基本对象：

**Worktile**：可以独立分配的一份计算工作，由 `decomposition.partition` 定义。它可以产生完整结果，也可以产生待合并的部分结果。

**Worker**：承担 worktile 的协作执行单元，可以是单个 CTA、多个 CTA 或其他平台的执行组织。一个 worker 可以连续处理多个 worktile。

核心分离是：

> **Worktile 的排列，不等于 worker 的工作归属；预先获配的工作，不等于运行时新领取的工作。**

#### 2. 固定字段

| 区域 | 字段 | 记录内容 |
| --- | --- | --- |
| 顶层 | `thesis` | 整体设计主张及目的。 |
| 三个主区 | `strategy` | 一句简短英文，概括该区的策略，不重复逐项解释。 |
| `decomposition` | `partition` | 真正产生独立 worktile 的划分维度。每项使用 `size / tile`，必要时加 `range`。 |
|  | `worktile.invariant` | 在当前 worktile 的局部迭代中保持不变的输入块。 |
|  | `worktile.iteration` | 随局部迭代更换的输入块，记录其 `shape / dtype / direction`。 |
|  | `target.single_tile` | 一个 worktile 的产物及其 `shape / dtype`。 |
|  | `target.cross_tile` | **可选。** 多个 worktile 合并后的产物。 |
|  | `target.tile_reduction` | **可选。** 合并的维度、输入、输出和数学方法引用。 |
| `mapping` | `worker` | 一个 worker 的组成：`unit / count`。 |
|  | `placement` | **可选。** 一个 worker 所需的物理资源范围，不登记没有设计意义的绑定信息。 |
|  | `count` | 本次提交的 worker 总数，可以是固定值或必要的运行时数量关系。 |
|  | `members` | **可选。** 协作成员之间的数学区域分工。 |
| `schedule` | `worktile_order` | Worktile 的逻辑排列规则。 |
|  | `worker_assignment` | Worker 的 `preassigned / refill / exit`。 |

`partition` 下的维度名、operand 名、产物名都是**当前算法的名称，不是模板专属字段**。名称与 Contract、数学定义和 tile pipeline 保持一致。

#### 3. 数据与范围的写法

**固定参数写实值，运行时规模写引用及适用范围。**

```plaintext
head:  {size: 8, tile: 1}
query: {size: QueryLength, range: [1, null], tile: 256}
```

`size` 是本次调用的实际规模；`range` 是允许的取值范围，`null` 表示本记录不额外限定该端的边界。已有 Contract 明确规定的范围不必重复填写。**数量取决于输入，不意味着划分策略未确定。**

`partition` 只列参与独立工作划分的维度。完整遍历 KV 时不写 `kv: all`；沿 KV 切成独立 worktile 时才加入 KV 维度。

Shape 使用**有序、带名称的维度列表**：

```plaintext
Q_high: {shape: [query: 128, head_dim: 128], dtype: bf16}
Q_low:  {shape: [query: 128, head_dim: 128], dtype: bf16, offset: [128, 0]}
```

`offset` 可选，按 shape 的维度顺序表示相对当前 worktile 的数学偏移，不是物理地址。名称与偏移的关系只需明确一次，不再增加 `tensor / views` 等重复描述。

所有 shape 默认以**整个 worktile**为作用域；成员局部范围由 `mapping.members` 给出。成员列表按编号排列，区域使用左闭右开区间。

`invariant` 不表示编译期常量；`iteration` 不重复登记可由运行时输入推导的迭代次数。合法迭代范围来自既有数学语义和输入定义，不能因省略次数而变成无约束访问。

#### 4. 排列与分配的语法

`worktile_order`：只定义宏观排列

```plaintext
worktile_order:
  - {by: [batch, head], group_size: 8, direction: ascending}
  - {by: query, direction: descending}
  - {by: [batch, head], direction: ascending}
```

列表从外层到内层读取：后续条目只在前面排序键相同的范围内继续排列，无需另写优先级编号。

`by` 可以是单个名称或有序名称组合。`[batch, head]` 按 batch 外层、head 内层展平；`group_size: 8` 表示**每组八个** `**(batch, head)**` **坐标对**，不是两个维度分别分八组。末组只包含实际存在的成员。

这里排列的是 worktile，**不包含 K/V 的内部遍历，也不隐含跨 worker 的启动顺序、完成顺序或同步依赖。**

`worker_assignment`：从 worker 视角描述工作归属

```plaintext
worker_assignment:
  preassigned: {start: worker_id, step: worker_count}
  refill: none
  exit: assigned_done
```

`**preassigned**` 描述预先确定的工作序列，而不只是第一项。`start + step` 表示有序 worktile 列表中的等步长序列，直到列表末尾；`start + count` 表示从起点开始的指定数量。也可以引用已有的预分配表。它不要求物化一个队列。

`**refill**` 描述运行时额外取得工作的机制。沿预分配序列前进不属于 refill；`none` 表示不补充工作。预取容量只在确实属于设计决定时记录。

`**exit**` 描述当前 worker 的退出条件。`assigned_done` 包含预分配及后续取得的全部工作；条件列表表示同时满足，例如：

```plaintext
exit: [refill_failed, assigned_done]
```

局部领取失败不等于全局完成。

`worker_id` 和 worktile 编号均从零开始；worktile 编号由 `worktile_order` 决定。`worker_count` 是提交数量，不是同时驻留数量。

#### 5. 取舍原则

**一个决定只写一次。** 最终算子语义属于 Contract；局部计算与同步属于 tile pipeline；本区只保留工作分解、执行组织和分配策略。

没有跨 tile 合并，就省略 `cross_tile / tile_reduction`；没有成员分工，就省略 `members`。不为格式对称填空壳，也不添加 `notes` 式的兜底区域。

`persistent / CLC` 等名称可以出现在 `strategy` 中帮助人类理解，但**有约束力的是下面的具体决策，不是策略名称**。

**精简的是重复描述与实现细节，必须留下的是：一份工作是什么、由谁承担、如何取得后续工作，以及何时结束。**

#### 6. 具体case

**FA4在sm100的3个变体（FA开源库的版本）**

这组示例选择：**每卡 8 个 MHA head，Q/K/V 的 head dimension 均为 128，BF16 输入和 O，FP32 LSE**。这里的 8 是这组 Plan 选择固定的参数，不是说所有 FA4 都必须固定为 8 个 head。

运行时量则保持为：

| 名称 | 含义 | 本组示例的范围 |
| --- | --- | --- |
| `BatchCount` | 本次调用的 batch 数 | 正整数，不额外限定上界 |
| `QueryLength` | 本次调用的 Q 长度 | 正整数，不额外限定上界 |
| `KVLength` | 本次调用的 KV 长度 | 正整数，不额外限定上界 |
| `SMCount` | 目标设备的 SM 数 | 来自 hardware profile |

同一次调用内使用等长序列，但长度可以在调用之间变化；**不预设** `**QueryLength = KVLength**`。三份都不包含 paged KV、varlen、split-KV 或 local window。第一份非 causal，后两份 causal，具体因果对齐语义沿用 Contract。

**版本一：双 CTA 协作，预分配固定步长的 worktile 序列**

这份保留原来的双 CTA 分支：每个 worktile 覆盖 512 行 Q，一个 worker 内两个 CTA 协作；提交的 worker 数受 worktile 总量和 SM 数量限制。源码的 cluster 级任务计数与步长分配对应这一组织。

```yaml
work_schedule:
  thesis: "两个 CTA 协作处理双 Q 块，按固定步长接续 worktile。"

  decomposition:
    strategy: "Partition Q rows into 512-row worktiles."
    partition:
      batch: {size: BatchCount, range: [1, null], tile: 1}
      head:  {size: 8, tile: 1}
      query: {size: QueryLength, range: [1, null], tile: 512}
    worktile:
      invariant:
        Q_high: {shape: [query: 256, head_dim: 128], dtype: bf16, offset: [0, 0]}
        Q_low:  {shape: [query: 256, head_dim: 128], dtype: bf16, offset: [256, 0]}
      iteration:
        K: {shape: [kv: 128, head_dim: 128], dtype: bf16, direction: descending}
        V: {shape: [kv: 128, head_dim: 128], dtype: bf16, direction: descending}
    target:
      single_tile:
        O:   {shape: [query: 512, head_dim: 128], dtype: bf16}
        LSE: {shape: [query: 512], dtype: fp32}

  mapping:
    strategy: "Map each worktile to a two-CTA worker."
    worker: {unit: CTA, count: 2}
    placement: {unit: SM, count: 2}
    count: "min(worktile_count, floor(SMCount / 2))"
    members:
      - {query: [[0, 128], [256, 384]]}
      - {query: [[128, 256], [384, 512]]}

  schedule:
    strategy: "Preassign strided worktiles to persistent workers."
    worktile_order:
      - {by: batch, direction: ascending}
      - {by: head, direction: ascending}
      - {by: query, direction: ascending}
    worker_assignment:
      preassigned: {start: worker_id, step: worker_count}
      refill: none
      exit: assigned_done
```

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/099ee316-5940-4f89-a8fc-5797a7e464b1.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/ab9d8d70-17f4-40e2-9288-8936104f9cab.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/d348e2ab-3394-4d0d-9c67-0ecc01f09f7b.png)

**版本二：单 CTA 单 worktile，按 KV 工作集分组排序**

这一版有一个必须随运行时范围同步修改的地方：原来的 group\_size: 8 只是 KVLength 为 8192 时的结果，不是这个源码分支在所有输入上的固定参数。

源码按完整 K/V 工作集大小调整分组宽度，使用 50 MiB 调度预算，并向下取 2 的幂。因此，扩大为运行时 KV 范围后，这里需要保留一条输入相关规则。50 MiB 是调度预算，不是物理 L2 容量的声明。

```yaml
work_schedule:
  thesis: "按 KV 工作集分组，组内较长 worktile 优先编号。"

  decomposition:
    strategy: "Partition Q rows and iterate over causal-visible KV blocks."
    partition:
      batch: {size: BatchCount, range: [1, null], tile: 1}
      head:  {size: 8, tile: 1}
      query: {size: QueryLength, range: [1, null], tile: 256}
    worktile:
      invariant:
        Q_high: {shape: [query: 128, head_dim: 128], dtype: bf16, offset: [0, 0]}
        Q_low:  {shape: [query: 128, head_dim: 128], dtype: bf16, offset: [128, 0]}
      iteration:
        K: {shape: [kv: 128, head_dim: 128], dtype: bf16, direction: descending}
        V: {shape: [kv: 128, head_dim: 128], dtype: bf16, direction: descending}
    target:
      single_tile:
        O:   {shape: [query: 256, head_dim: 128], dtype: bf16}
        LSE: {shape: [query: 256], dtype: fp32}

  mapping:
    strategy: "Map each worktile to one CTA."
    worker: {unit: CTA, count: 1}
    placement: {unit: SM, count: 1}
    count: worktile_count

  schedule:
    strategy: "Group KV working sets and number longer worktiles first."
    worktile_order:
      - {by: [batch, head], group_size: "max(1, floor_pow2(102400 / KVLength))", direction: ascending}
      - {by: query, direction: descending}
      - {by: [batch, head], direction: ascending}
    worker_assignment:
      preassigned: {start: worker_id, count: 1}
      refill: none
      exit: assigned_done
```

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/573bfd67-38c0-4b9e-af8e-0fb9a09def1e.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/464cc4bb-144d-4bd4-823f-6309ca400778.png)

**版本三：单 CTA 预分配一项工作，运行时接管更多工作**

这一版的数学分解与第二版相同，取消工作集分组。每个 worker 预先对应一个 worktile，随后可以取消尚未启动的 worker 并接管其工作。源码的这个分支使用原始提交坐标对应的工作，并进行 Q 块反向映射，不走上一版的分组重排。

```yaml
work_schedule:
  thesis: "每个 worker 先分配一项工作，再提前接管未启动 worker 的工作。"

  decomposition:
    strategy: "Partition Q rows and iterate over causal-visible KV blocks."
    partition:
      batch: {size: BatchCount, range: [1, null], tile: 1}
      head:  {size: 8, tile: 1}
      query: {size: QueryLength, range: [1, null], tile: 256}
    worktile:
      invariant:
        Q_high: {shape: [query: 128, head_dim: 128], dtype: bf16, offset: [0, 0]}
        Q_low:  {shape: [query: 128, head_dim: 128], dtype: bf16, offset: [128, 0]}
      iteration:
        K: {shape: [kv: 128, head_dim: 128], dtype: bf16, direction: descending}
        V: {shape: [kv: 128, head_dim: 128], dtype: bf16, direction: descending}
    target:
      single_tile:
        O:   {shape: [query: 256, head_dim: 128], dtype: bf16}
        LSE: {shape: [query: 256], dtype: fp32}

  mapping:
    strategy: "Map one preassigned worktile to each CTA."
    worker: {unit: CTA, count: 1}
    placement: {unit: SM, count: 1}
    count: worktile_count

  schedule:
    strategy: "Prefetch work by cancelling unstarted workers."
    worktile_order:
      - {by: batch, direction: ascending}
      - {by: head, direction: ascending}
      - {by: query, direction: descending}
    worker_assignment:
      preassigned: {start: worker_id, count: 1}
      refill: {cancel: unstarted_worker, prefetch: 1}
      exit: [refill_failed, assigned_done]
```

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/87afd7c6-c30c-4bf6-b26a-52cfdab1a904.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/d2150a3e-1dce-43ef-a383-2df8e8635c77.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/57a350c7-b7b0-450e-b58d-4faf7cbb6633.png)

**以上展示的过程中，可以发现，其实work schedule也能画图（类似于tile pipeline），但是感觉会比较麻烦（看起来信息量似乎和纯文字的差不多），所以似乎没有必要专门在这里绘图了，YAML已经足够清楚和简洁**

### 26/09/14 schema的分区讨论：III

**reshape是修改、优化要具体实现的任务本身**

**schedule是分治任务、规划子任务流程来还原完整任务**

**pipeline是确定子任务的具体细节和实现策略**

目前，已经确定好了work schedule + tile pipeline的格式字段和细节示例。现在的问题在于math reshape的部分。这一部分最难描述，最难界定，但是最核心，效果也最重要。本质上来说，我认为这个的核心目标就是在保持contract部分的算法不变性的情况下（或者，一定程度的精度妥协，仅仅是类似于计算先后顺序、乘法累加等等的妥协；而不是直接不算、跳过的妥协），通过交换计算顺序、分块、融合等等方式，降低算子在理论上的复杂度，特别的，比如类似于online softmax、分块矩阵乘法等等，都可以认为是，一方面，降低local的显式量来把问题简化到一个GPU可以接受的复杂度（比如，SMEM可以放得下），或者，避免重复的扫描过程和数据搬运，或者，避免显式化的得到某个中间矩阵，直接在高层级的cache来进行复用和计算，或者，直接把一些计算融合、先后顺序交换、把CUDA core的问题拼凑成Tensor Core的问题来提高利用率。

所以现在的核心难题就变成了，我们其实对于work schedule、tile pipeline是有感知的，因为SOTA kernel的代码是很明确的会显式化这些东西。简单来说，越往下走，和代码绑定的越深。所以你可以看得出来，tile pipeline有的地方的描述和变量的名字已经完全接近了。但是，对于math reshape这一层，甚至在代码里很难看出来，他已经抽象到了“需要亲自翻论文”或者去研究原作者的代码逻辑，才能看出来这里是不是用到了某种技巧。越贴近数学和顶层，在代码中暴露的信息就越少。但是，收益会越高。这里本质上就是，解决和优化算法的时间复杂度、空间复杂度，涉及到了大量的计算机技巧和数学知识。所以，我在考虑的就是，一方面，如何表达？有哪些字段？这里是不是其实没有太多的文字？通过几个字符串来表达就行？因为我一下子想不明白这里有什么“需要被规范的东西”。另一方面，他本身要涵盖什么？要表达什么？哪些东西应该作为需要纳入的对象？哪些又不属于这里？主要的点在于，似乎这里要用“纯文字”和“大段的描述”和latex公式来写了？但是我觉得，如果想让人阅读明白，这个东西是怎么实现的、有什么好处和意义，还是需要更清晰的YAML来表达。我们考虑在不画图的情况下，纯YAML，怎么样把这个事情说明白？写伪代码也可以，但是伪代码的阅读应该也要清晰、容易理解，而不是有非常高的阅读成本。而且，这个math reshape应该完全和kernel的流水线无关。它是上游，是决定下游的前置条件。在我们思考和确定reshape方法的时候，会“感知和aware”下游的schedule/pipeline设计，但是更多的是感知硬件，确定这个东西一方面可以降低理论的复杂度，另一方面可以在这一代GPU硬件上面work起来。在我们的定义里面，math reshape就是最重要的，从algorithm+hardware这两个spec的contract，衔接过渡的层，它要做到算法-硬件的co-design，而且是纯算法层面的改进。当这一层确定后，它的输出，会作为work schedule的输出。因为只有math reshape确定好了，才知道我们的tile的名字，计算先后顺序、数学过程，这样才能开始分配工作，才能确定好任务的分治。

我这里想到了一些例子。比如，或许可以从math reshape的分类出发来描述？

representation、preprocessing、decomposition、algorithm、schedule、dispatch

不要默认继承下来的算子边界、paged 表示、kernel 分解和融合方式不可改变；同时要把转换成本和 launch 成本算入整体目标。然后，也要区分出来哪些是外部的约束，哪些是继承的实现，哪些是可以自由发挥的不影响精度的选择？

可能的路线：

direct\_native 直接处理现有输入表示，能做到什么程度？

representation\_transform 先转换、gather 或重排，是否能换来更好的后续计算路径？

multi\_stage 主动拆成多个阶段，会不会比强行融合更好？

hybrid\_dispatch 不同输入区域是否应该选择不同路线？

以及，每次变换可能的Gain和Cost？

每条 route 记录：transformation\_cost、unlocked\_fast\_path、winning\_regimes、losing\_regimes

也就是说，不能只说“转换后计算更快”，还要说明为了得到这个路径付出了什么，以及在哪些输入区域值得做。这次变换新增了什么工作，消除了什么工作，或把什么昂贵操作换成了什么便宜操作？例如，重计算换存储不一定减少 FLOPs，却可能减少中间张量流量；表示转换可能增加一个阶段，却降低后续访问成本。

以上的内容和例子只是一些抛砖引玉和浅层思考，因为，这些分类大概率是不完备的，不正交的，如果一定要这样写，要么不能表达潜在的优化，要么会逼着模型把一些合理的东西用不恰当的方法来表达。我现在没有特别好的思路。但是我认为这件事情最好是data-driven的。我希望你可以认真的去看看，github的开源仓库，包括grouped GEMM、Flashinfer、deepGEMM、HPC-ops、Atrex算子库，等等各种算子库，特别是最复杂的，FusedMoE、KDA、GDN、FA、MLA、各种GEMM。我觉得或许我们可以从中挑选出3个，FA4+GDN+MoE，作为case，来实际看看这个东西该怎么书写。我觉得关键在于，你要给一个不太做过这个的人看明白他的优化技巧。而且要写的清楚，凝练，而避免大量的、无法检验的字符串纯文字。但是数学公式、latex肯定也是需要的，毕竟要描述清楚一个数学过程。所以就是，表达内容+表达形式，这两个请你好好的、深入的展开思考，要考虑到我们的schema的通用性/可泛化性，简洁性/高度抽象性，和易读性/人类友好。你可以先给出你的一些思考、想法、讨论，然后再根据这三个我要求的case来试试，类似的YAML是什么样子的？我们可能会迭代和讨论好几版。所以，一步一步来。

26/09/21：

### AKA的baseline实验

**B300 ：FA4、GDN，从零写（允许给wiki，提前准备离线的sm103a的资料，不联网，不看repos），A/B（插入plan开关；对比迭代轮次/上限）**

**baseline：AKA内源两个分支main/light（不引入任何开源repos，wiki准备好B300的知识），\* GND/FA4两个题目；CC（Qwen38max），codex（不用），qoder（不用），不打开frontier模型的review功能；**

先跑baseline，2\*2=4个实验的baseline，**文件隔离**；episode设置1000（不限），每个拉起3个工作区并行同时开始。共4\*3=12个进程

QSA（还没有题目，届时可以enable全部功能，验证集）

比较 度行提供的简化版simplified和现在main主版AKA，题目是睿博给的FA4和GDN，任务是L20D的cutedsl，条件是完全删除了所有3rd party的外部project源代码，让他从零迭代，关闭了联网fetch的功能。

然后每个实验并行开了3个duplicate，相当于2\*2\*3=12个测试，文件隔离应该是没有问题的。刚刚因为L20D缩容我把实验停下来做了一个分析，画了两个图。分别是算子随着episode而对应的geomean latency的下降（有的版本可能没接受就留空），以及输入token、输出token的历史累积总量

总体来说看起来，节省的是输入，而不是输出。而且简化版倾向于前期一直没有accept，但是接受的一般都是更好的性能。

![fa4_trace.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/3a38ff86-eba7-4d24-b9ff-e7c898268aaa.png)

![gdn_trace.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/3127e344-3aed-4e88-9cd6-4106bf1b653e.png)

**新的题目：**

FA4 Prefill（30个shape）

FA4 Decode（30个shape）

GDN Prefill（17个shape）

QSA Prefill（4个shape）

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/b3756ef6-1018-48d7-bcd9-d8838f06f7ba.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/3b563bc5-4a14-44a7-b5f6-d05d14d8abcc.png)

![fa4_prefill_trace.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/v9kqDea8L9oWaOVx/img/ee6ea4d0-1083-4c64-a212-2fc5cbf609a7.png)

![fa4_decode_trace.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/v9kqDea8L9oWaOVx/img/aa087003-f804-4d89-94b3-da82d434b95e.png)

![gdn_prefill_trace.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/v9kqDea8L9oWaOVx/img/21130688-3e23-4ec1-9d35-93b3eef961e0.png)

![qsa_prefill_trace.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/v9kqDea8L9oWaOVx/img/8f84f10e-86a9-4500-ad31-faed7c0185c2.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/v9kqDea8L9oWaOVx/img/72478f22-74b7-458c-894d-351f3f04f59b.png)

**基于上述已有seed的进一步优化：**

**26/10/21，基于已有实验的最佳版本的进一步24路优化（FA4 Prefill/Decode，GDN/QSA Prefill），大概一共历时 120小时**

累计2105个终结episode、511次晋升，每路46–198/1000

**FA4 Prefill的情况：**

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/02135bf9-0852-47c6-b55f-098da3700f2c.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/1d2b6d2b-90c5-4814-8ea8-44070c2a6873.png)

**FA4 Decode的情况：**

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/8e5b4f22-2697-4516-9b34-ef4f13d1acf2.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/4383d63d-b1cf-4b26-b720-87216447feb7.png)

**GDN Prefill的情况：**

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/58095213-1aab-4c84-89f6-88b418789288.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/d271add0-3ba3-493e-926c-413dd3e3edee.png)

**QSA Prefill的情况：**

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/4c5bf5ab-a27d-4c92-8cd4-7e7ef12894f1.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/b20e0b36-766d-4b44-8b9b-11540a94095d.png)

### AKA的schema对照实验

24路，都使用AKA light（非主线的旧版本light），是否开启schema注入作为对照（没有其余任何wiki信息）

**四个算子的从零迭代结果：**

![fa4_prefill_trace.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/d4f430a9-620e-491e-8112-59be56c75ff2.png)

![fa4_decode_trace.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/02e57c82-c6a9-4681-ac3c-96e7dbe23278.png)

![gdn_prefill_trace.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/f5abd2a9-8cbe-4ddf-8d19-bd8dde145975.png)

![qsa_prefill_trace.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/6c7f1928-a80f-4536-8d7a-e0aa00b18d72.png)

**对照schema的分析情况：**

![01_early_convergence.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/8f8aaa5b-e565-4464-9d39-25317d226583.png)

![02_stability.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/4b8c6e54-e65e-4732-b7da-c15c51f236aa.png)

![03_best_performance.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/d08a7a91-5a0a-4fdb-80f6-99f0245d336a.png)

![appendix_attainment_profile.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/b1ed5115-0de4-48bd-b05d-a44fe5ef92f2.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/69b32806-7a53-4f2b-ab31-2e8786371661.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/c198a392-2efa-41a1-b018-eedb1082486b.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/3aead0b2-ffc6-40ad-a5c4-af143dae59a6.png)

### math reshape的YAML语言表述

总的来说，比较类似于tile pipeline，以多个lane并行为核心，这里也是以多个step为核心：

1）thesis简单交代核心的设计思路

2）rewrite内部由多个串行/并行的step1、2、3组成，每个step可以理解为是“一段关键的数学变换和优化”，有自己的name（称呼）、math（这里通过接近数学公式的语言来描述）、symbols（新引入的数学记号以及他们的含义）、arithmetic（这里的计算的关键数学精度/舍入/累加等等的选取）

3）ledger作为账本，通过不同的类别记录这个reshape对应的收支取舍，比如memory的traffic、计算、深度、存在的精度妥协、减少的计算/访存，强调这个方法带来的理论优势和开销

#### 1. 核心思想

**1.1 这一层做什么**

Plan 的 Design 部分有三层：math\_reshape 修改、优化要实现的任务本身；work\_schedule 把任务分治成可以分配的 worktile；tile\_pipeline 确定每个 worktile 内部的执行细节。math\_reshape 在最上游：在不违反 Contract 的前提下，决定算出哪些值、这些值怎样定义、以什么精度存在、存在多久。

只有这一层定下来，下游才知道有哪些量、哪些量跨迭代携带、哪些中间量要写回全局内存，才能开始划分工作。例如 FA4 的改写确定了“沿 kv 分块迭代、每行携带 (m, L, Y)”，work\_schedule 才能把 K、V 写成迭代输入，把查询行写成划分维度。

这一层在源码里最难看见。调度器、warp 角色和 barrier 在代码里都是显式的，数学改写却常常只是几行 FMA、一个阈值判断或一次类型转换：FA4 的滞后最大值，在源码里只是 `softmax.py` 中的一行 `if acc_scale_ >= -self.rescale_threshold`。把它单独写出来，专家和 Agent 才能直接讨论、修改、比较 kernel 的算法形态，而不必先读懂或重写代码。

**1.2 改写带来什么**

这一层是算法与硬件的协同设计：动机来自硬件，改写本身用数学表达。常见的收益如下：

| 收益 | 做法 | 案例 |
| --- | --- | --- |
| 状态放得进片上 | 用固定大小的状态代替随输入长度增长的中间量 | FA4：每行 Lk 个分数压成 d + 2 个 fp32 |
| 避免物化与重复搬运 | 中间量只以块的形式存在 | FA4：S、P 不写回 HBM |
| 串行工作变成矩阵乘 | 把逐步递推折叠成块内的矩阵运算 | GDN：L 次秩一更新变成 L/64 次块更新 |
| 分散的工作变得稠密 | 改变表示，把零散的计算聚成能复用数据的块 | MoE：逐槽位 GEMV 变成按专家分组的 GEMM |
| 换用更便宜的运算单元 | 在精度允许的范围内使用近似或低精度 | FA4：部分 exp2 从 MUFU 挪到 FMA；MoE：MMA 直接吃 fp8 码值 |

收益总有代价：GDN 多出约 1.8 倍的 MAC 和五趟之间写回 HBM 的中间量，MoE 多出补齐行和每 token 约 250 KB 的中间张量。代价与收益一样记在 ledger 里。

**1.3 精度是改写的一部分**

改写几乎总会改变舍入点、累加次序或数据格式，一次错误的舍入就可能让整个 kernel 不可用。本规则区分三类，并给每一类固定的位置：

*   **精确等价**：实数下与原定义相同的重排，例如在线 softmax、UT 变换、把路由权重移进 GEMM2。它们直接体现在计算本身。
    
*   **有限精度差异**：舍入点与累加精度的变化，例如 P 舍入到 bf16、W 和 U 以 bf16 存储。它们由 math 中的显式 cast 表达。
    
*   **主动近似**：函数求值的近似，例如多项式 exp2。它们写在 `arithmetic.approx`，并给出误差界。
    

这一层不能为了性能放宽 Contract 的精度要求。Contract 规定的量化（例如 MoE 的 q(·)）直接引用，本层只决定它作用在哪里。

**1.4 核心约定**

**math 只记录 kernel 实际执行的计算。** 不写证明、恒等式、不变量或解释。改写是否与原定义等价，由实现与 golden reference 的数值对比来检验；设计为什么值得，由 ledger 的定量增减来体现。这样 math 的每一行都能对照源码或实现逐行核对。

**1.5 边界：一个决定属于哪一层**

| 这个决定改变了什么 | 归属 |
| --- | --- |
| 算出哪些值、如何定义、什么精度、存在多久 | math\_reshape |
| 谁在何时计算哪一块 | work\_schedule / tile\_pipeline |
| 只替换实现，数值差异在 fp32 ulp 以内 | realization |

*   FA4 的多项式 exp2 误差 8.8e-5，超出 fp32 ulp，属于本层；`rcp.approx`、`lg2.approx` 只有 ulp 级差异，属于 realization。
    
*   FA4 的 kv 块长、MoE 的每块槽位数、访问顺序属于 work\_schedule；GDN 的块长 64 决定三角系统的规模与串行步数，属于本层。
    

#### 2. 字段划分

```yaml
math_reshape:
  thesis: ...          # 一两句：改写的主张与目的
  rewrite:             # 有序步骤，从 Contract 的输入写到 Contract 的输出
    - step: 1          # 互不依赖的兄弟步骤写 1(A)、1(B)
      name: ...        # 这一步做的决定
      math: |          # 实际计算：定义、递推、分情形、显式 cast
      symbols:         # 公式里看不出的信息
      arithmetic:      # 公式表达不了的精度例外
  ledger:              # 相对 baseline 的客观增减
    baseline: ...
    <resource>: [{item, change, by, when}]

```

**2.1 thesis**

一两句写出改写的主张与目的，也点出主要代价。读者只看 thesis，就应知道这份改写的核心想法；“提升性能”“优化访存”之类的泛化口号不写。

**2.2 rewrite、step 与 name**

rewrite 是有序的步骤列表，从 Contract 的输入一直写到 Contract 的输出。每个 step 是一个设计决定：

*   `step` 是编号，供 ledger 回指。互不依赖的兄弟步骤写 1(A)、1(B)，例如 MoE 的输入量化与槽位分组。列表顺序表达数据依赖，不承诺执行必须串行。
    
*   `name` 用一句话写出这一步做的决定，例如“参考最大值允许滞后 τ”，而不是“softmax 阶段”。
    

**2.3 math**

kernel 实际执行的计算：定义、递推、分情形、显式 cast。它是 step 的主体，也是这一层最需要逐行审阅的部分，写法见第 3 节。

**2.4 symbols**

只登记公式里看不出的信息。公式已经定义清楚、又没有下面这些属性的中间量，不登记。

*   `life` 表示一个量存在多久，是本层最重要的属性：
    
    *   省略：只在本步的一次计算中存在，例如 FA4 的 E、P，GDN 的 A、R；
        
    *   `carried`：跨迭代携带的状态，例如 FA4 的 m、L、Y，GDN 的 S。它们的总大小决定片上能否放下；
        
    *   `stored`：写回全局内存、由后续 pass 读取的中间量，例如 GDN 的 G、T、W、U、S̄、R̄。它们决定额外的访存，也标出了 pass 的边界。
        
*   `value`：本层固定的取值，例如 GDN 的块长 64。
    
*   `shape`、`dtype`：公式推不出时才写。
    
*   `meaning`：公式看不出含义时写一个短语；由下游决定的量在这里注明，例如“kv 块长（由 work\_schedule 决定）”。
    

用 `life` 而不另设一个“物化”区，是因为物化和状态压缩正是访存与片上容量的来源。把它们写在记号定义处，每个量的代价一眼可见。

**2.5 arithmetic**

只记公式表达不了的精度例外：

*   `approx`：函数求值的主动近似，写方法与误差界，例如 FA4 的 3 次多项式 exp2；
    
*   `accumulate`：非默认的累加方式，例如 GDN 求逆时子块合并用 tf32 输入，MoE 的 FP8 MMA 累加器位宽低于 fp32。
    

fp32 计算、fp32 累加与 RNE 舍入是默认值，不写；窄化由 math 中的 cast 表达。四个案例合计只有 5 条 arithmetic。

**2.6 ledger**

客观记录改写相对 baseline 多了什么、少了什么、换成了什么。

*   `baseline`：一句话写清被比较的直算方式，例如 GDN 的“逐 token 递推，fp32 状态常驻片上”。资源类条目以它为基准，`error` 以精确计算 Contract 公式为基准。
    
*   资源键是固定词表，只写有变化的资源：
    
    | 键 | 计量 | 为什么单列 |
    | --- | --- | --- |
    | `matmul`、`vector`、`transcendental` | 张量核、通用 ALU、特殊函数单元上的计算 | 把工作从一种单元换到另一种，本身就是常见的改写：GDN 是 vector → matmul，FA4 是 transcendental → vector |
    | `traffic` | 片外访存字节 | 决定带宽开销 |
    | `footprint` | 每个并行单元必须常驻片上的状态 | 决定能否放下，与 traffic 是两回事 |
    | `depth` | 串行依赖链的长度 | 决定可用的并行度，例如 GDN 从 L 缩短到 L/64 |
    | `passes` | kernel 数或全量扫描趟数 | 包含 launch 开销与多趟中间量 |
    | `comm` | 跨 SM 或跨设备的通信 | 本轮案例未用到 |
    | `error` | 相对精确计算新增的误差来源 | 精度代价也是代价 |
    
*   条目有四个键：
    
    *   `item`：对象；
        
    *   `change`：用 `+`（新增）、`−`（消除）、`A → B`（转换），并且定量；公式用 Contract 与 rewrite 的记号写，具体形状下的数值附在括号里；
        
    *   `by`：回指造成它的步骤；
        
    *   `when`：只在特定输入区域成立时写，例如 MoE 的 `T·k·4 ≤ E`。
        

不叫“得失”，是因为同一个变化在不同场景下好坏不同：GDN 的 matmul 增加恰恰是收益。ledger 只记事实，判断留给评审与实验。要求定量，是因为定性描述无法核对，而且会无限膨胀。

#### 3. 语法规则

**3.1 记号**

| 记号 | 含义 |
| --- | --- |
| `X(i)`、`X(c)` | 迭代 i、块 c 上的实例；`X(−1)` 或 `X(0)` 为初值 |
| `X[·]`、`blk(j)` | 下标；`blk(j) = [j·B, (j+1)·B)` 为第 j 块的位置区间 |
| `·`、`⊙`、`ᵀ` | 矩阵乘或数乘、逐元素乘（向量按行广播）、转置 |
| `Σ v`、`Σ(范围) 表达式`、`max v` | 向量元素求和、按范围求和、向量元素取最大 |
| `⟨a, b⟩` | 一个量化组内的内积 |
| `cumsum`、`tril`、`‖`、`⌊·⌋`、`⌈·⌉` | 前缀和、保留 r ≥ s、拼接、取整 |
| `bf16(·)`、`fp8(·)`、`q(·)` | 舍入到 bf16、舍入到 fp8（饱和）、Contract 规定的量化 |
| `a 若 条件；否则 b` | 分情形，也用于按输入区域分派 |

**3.2 迭代与递推**

迭代 i 上的值写成 X(i)，跨迭代依赖直接写成从 X(i−1) 到 X(i) 的递推，不另设循环语法；初值写成 X(−1) 或 X(0)。例如 FA4 的 `L(i) = α(i)·L(i−1) + Σ E(i)`，`L(−1) = 0`。块内位置用方括号，块的位置区间用 blk(j)；访问顺序记为 σ，由 work\_schedule 决定。

**3.3 分情形**

条件更新与按输入区域分派都用“若 … 否则 …”，各分支纵向对齐。FA4 的参考最大值：

```plaintext
m(i) = m(i−1)   若 m(i−1) > −∞ 且 c·(m̂(i) − m(i−1)) ≤ τ
       m̂(i)     否则

```

MoE 按批量大小选择两种分组方式：

```plaintext
B(b) = {b},    ε(b) = e(b)                                            若 T·k·4 ≤ E
B(b) = π[b·BM, (b+1)·BM),    ε(b) = e(π[b·BM])                        否则

```

**3.4 精度**

*   窄于 fp32 的值一律由显式 cast 定义：`bf16(·)`、`fp8(·)`，或 Contract 的 `q(·)`。没有 cast 的计算默认为 fp32 计算、fp32 累加；舍入为 RNE，`fp8(·)` 饱和到 ±448。
    
*   精度分叉必须分别命名。GDN 的修正量 R 有两个 bf16 版本：`R̄(c) = bf16(R(c))` 存给输出，`bf16(δ(c) ⊙ R(c))` 用于状态更新。两者都由 fp32 的 R 算出，不能互相代替；如果共用一个名字，读者就无法知道状态更新用的是哪一个。
    
*   只在最终存储时舍入的值，把 cast 直接写在输出上，例如 `O = bf16(Y(N−1) / L(N−1))`。
    

**3.5 记号来源与命名**

*   Contract 的记号直接使用，例如 Q、K、V、scale、visible、q(·)，不重新定义。
    
*   局部视图用公式命名，例如 GDN 的 `X(c) = X[blk(c)]`，而不是用文字描述“第 c 块的输入”。
    
*   新记号与 tile\_pipeline 对齐：FA4 的 S、P、α、L、Y、N 对应 tile pipeline 中的 `S_high(i)`、`alpha_high(i)`、`L_high(i)`、`Y_high(i)` 与参数 N。希腊字母在下游可用拼写名。
    

**3.6 不写什么**

*   不用 LaTeX：原始的 LaTeX 在 YAML 里不会渲染，读者要先在脑子里编译一遍。
    
*   不用 torch 写法：view、transpose、dtype 参数带着实现的形状，会掩盖数学关系。
    
*   不写说明性的文字：math 中只保留“若 / 否则”，含义放进 `symbols.meaning`。
    

**3.7 示例**

一个完整的 step（取自 FA4）：

```yaml
- step: 3
  name: 分母累加 E，分子累加 bf16(E)
  math: |
    E(i) = exp2(c·S(i) − c·b(i)),    P(i) = bf16(E(i))
    L(i) = α(i)·L(i−1) + Σ E(i),       L(−1) = 0
    Y(i) = α(i)·Y(i−1) + P(i)·V(i),    Y(−1) = 0
  symbols:
    L(i): {dtype: fp32, life: carried, meaning: 分母}
    Y(i): {shape: [d: 128], dtype: fp32, life: carried, meaning: 分子}
  arithmetic:
    approx: 1/8 的 E 元素（每块每行 128 列中的 16 列）用 exp2(x) ≈ 2^⌊x⌋·p₃(x − ⌊x⌋)，p₃ 为 [0, 1) 上的 3 次 minimax 多项式，最大相对误差 8.8e-5

```

*   第一行把同一个指数结果分成两种精度：E 保持 fp32，P = bf16(E)。
    
*   第二、三行是跨块递推：分母 L 累加 E，分子 Y 累加 P·V，两者共用同一个缩放因子 α(i)。
    
*   symbols 只登记两个 carried 状态；E、P、α 由公式定义、只在本步存在，不登记。
    
*   arithmetic 只有一条：部分 exp2 用多项式近似，给出方法与误差界。
    

一条 ledger 记录：

```yaml
traffic:
  - {item: S 与 P, change: "− 16 · Lq · Lk B：fp32 的 S、P 各写一次、读一次", by: [1, 3]}

```

`item` 写对象，`change` 写带符号的定量变化，`by` 指向造成它的步骤 1（只访问 kv 块）与步骤 3（在线累加）。

#### 4. 原则

1.  **Less is more。** 一条信息只写一次，写在唯一合适的位置；不为格式对称填空壳。判断标准是：删掉它，读者会不会读错或漏掉设计？不会就删。按这条标准，本规则删去了 given、恒等式、不变量与数值范围，也删去了 FA4 非因果分支里不会出现的空行分支。
    
2.  **只写实际计算。** 不写恒等式、不变量、数值范围，也不写“次序任意”这类声明。证明改写正确不是这一层的任务；math 只需能逐行对照源码或实现核对。
    
3.  **一步一个决定。** 一个 step 对应一个可以单独审阅、替换或否决的决定，在两种地方切开：
    
    *   pass 边界，即产生 `stored` 之后；
        
    *   迭代结构变化处，即“每块一次”与“结束时一次”之间。
        
    
    精度分叉不单独成步，而是在同一步里用两个名字表达。例如 FA4 的步骤 2 可以单独撤销，令 τ = 0 就退回每块都重缩放的在线 softmax；GDN 的五步正好对应五个 kernel。
    
4.  **精度写进公式。** 每个窄于 fp32 的值都由显式 cast 定义，读者从公式本身就能看出每个值走哪条精度路径。舍入点因此成为设计的一部分，Coder 不能悄悄移动。`arithmetic` 只记公式表达不了的例外。
    
5.  **省略有边界。** 未改变的计算只引用、不重写，例如 SiLU、q(·)、mask；改写后的计算完整写出，一直写到 Contract 的输出。
    
    *   在 per-tensor FP8 的 MoE 里，GEMM 可以不展开，因为 scale 只是标量；
        
    *   在 block FP8 的 MoE 里，GEMM 必须展开，因为“scale 提到组外”这一改写就发生在 GEMM 内部。
        
6.  **不替下游做决定。** 块大小、访问顺序由 work\_schedule 决定，只在 `symbols` 中注明。只有数学上必须固定的量才写 `value`：GDN 的块长 64 决定了三角系统的规模、串行步数与中间量的大小，所以由本层固定。
    
7.  **ledger 客观、定量、单一基准。** 只写有变化的资源，不记流水重叠、指令选择与定性描述。基准必须写明：同一个 bf16 中间张量，相对“全融合”是新增，相对“逐步不融合”则不是变化，基准不明就会时算时不算。
    
8.  **不设分类字段。** 常见的改写类别彼此不正交：FA4 的在线 softmax 同时是分解、算法与状态表示，MoE 的按专家分组同时是表示变换、分派与分解。强行归类只会误导，而这些类别都能从结构直接读出：
    
    | 类别 | 结构中的签名 |
    | --- | --- |
    | 状态压缩 | `life: carried` |
    | 多阶段、物化 | `life: stored` |
    | 表示变换 | 引入索引类记号，数据张量本身不重排 |
    | 按输入区域分派 | math 中的分情形，配合 `ledger.when` |
    | 重算代替存储 | 派生量在多步中现算，没有 `stored` |
    | 跳过计算 | 迭代域收缩 |
    | 精度分叉 | 同一个量的两个 cast 版本 |
    
9.  **按真实源码的单一分支填写。** 源码中每一处偏离精确 fp32 计算的地方，都必须出现在 math 的 cast 或 `arithmetic` 里；不同分支不拼成一份。
    
    *   FA4 的 work\_schedule 版本二、三调度不同，数学完全相同，共用一份；版本一在块集合、空行分支与多项式比例三处不同，单独一份。
        
    *   一份独立写出的版本漏掉了多项式 exp2。这类近似藏在一次函数调用里，不逐处核对就看不见。
        

#### 5. 具体case

**1）FA4 · SM100 · 因果单 CTA**

*   **来源**：Dao-AILab/flash-attention `ce088ab9`，`flash_attn/cute/flash_fwd_sm100.py`、`softmax.py`；因果、单 CTA、hdim 128、bf16，对应 work\_schedule 版本二与版本三。
    
*   **内容**：4 步。只访问含可见 key 的 kv 块；参考最大值允许滞后 τ = 8；分母累加 fp32 的 E、分子累加 bf16(E)，其中 1/8 的 exp2 用多项式；结束时一次归一化得到 O 与 LSE。ledger：消除 S、P 的 HBM 往返，每行状态压到 d + 2 个 fp32，3 个 pass 合为 1 个。
    

```yaml
math_reshape:
  thesis: >-
    每行只带 (m, L, Y) 按 kv 块在线合并，S、P 从不整体存在；参考最大值允许落后至多 2^τ 倍，
    历史状态只在最大值跃升时重缩放；因果下整块不可见的 kv 块不参与计算。

  rewrite:
    - step: 1
      name: 只访问含可见 key 的 kv 块
      math: |
        blk(j) = [j·Bk, (j+1)·Bk)
        J = { j | ∃ q ∈ R, k ∈ blk(j)：visible(q, k) },    N = |J|
        S(i) = Q[q] · K[blk(σ(i))]ᵀ,    V(i) = V[blk(σ(i))],    S(i)[k] = −∞  若 ¬visible(q, k)
      symbols:
        R:  {meaning: 共用同一组 kv 块的查询行，q ∈ R（由 work_schedule 决定）}
        Bk: {meaning: kv 块长（由 work_schedule 决定）}
        σ:  {meaning: J 中块的访问次序（由 work_schedule 决定）}

    - step: 2
      name: 参考最大值允许滞后 τ
      math: |
        c = scale · log2(e),    τ = 8
        m(−1) = −∞,    m̂(i) = max(m(i−1), max S(i))
        m(i) = m(i−1)   若 m(i−1) > −∞ 且 c·(m̂(i) − m(i−1)) ≤ τ
               m̂(i)     否则
        b(i) = m(i)   若 m(i) > −∞；否则 0
        α(i) = exp2(c·(m(i−1) − b(i)))
      symbols:
        m(i): {dtype: fp32, life: carried, meaning: 参考最大值}

    - step: 3
      name: 分母累加 E，分子累加 bf16(E)
      math: |
        E(i) = exp2(c·S(i) − c·b(i)),    P(i) = bf16(E(i))
        L(i) = α(i)·L(i−1) + Σ E(i),       L(−1) = 0
        Y(i) = α(i)·Y(i−1) + P(i)·V(i),    Y(−1) = 0
      symbols:
        L(i): {dtype: fp32, life: carried, meaning: 分母}
        Y(i): {shape: [d: 128], dtype: fp32, life: carried, meaning: 分子}
      arithmetic:
        approx: 1/8 的 E 元素（每块每行 128 列中的 16 列）用 exp2(x) ≈ 2^⌊x⌋·p₃(x − ⌊x⌋)，p₃ 为 [0, 1) 上的 3 次 minimax 多项式，最大相对误差 8.8e-5

    - step: 4
      name: 结束时一次归一化
      math: |
        O = bf16(Y(N−1) / L(N−1)),    LSE = ln2 · (c·m(N−1) + log2 L(N−1))      若 L(N−1) > 0
        O = 0,    LSE = −∞                                                   若 L(N−1) = 0

  ledger:
    baseline: 按 Contract 直算：物化 fp32 的 S = Q·Kᵀ 与 P = softmax(S)，再算 P · V
    traffic:
      - {item: S 与 P, change: "− 16 · Lq · Lk B：fp32 的 S、P 各写一次、读一次", by: [1, 3]}
    matmul:
      - {item: 不可见块, change: "− 约一半的 4 · Lq · Lk · d FLOP（Lq = Lk 时；跨对角线的块仍整块计算）", by: [1]}
    vector:
      - {item: Y 重缩放, change: "+ 每行每次参考更新 d 次乘法；首块之后每行最多 c · (M − m(0)) / τ 次，M 为该行最终最大值", by: [2, 3]}
    transcendental:
      - {item: exp2, change: "访问到的每个分数一次 ex2，其中 1/8 改为约 6 条 FMA/ALU 指令（含 packed f32x2）", by: [3]}
    footprint:
      - {item: 每行状态, change: "Lk 个分数 → d + 2 个 fp32：Y 的一行、m、L", by: [2, 3]}
    depth:
      - {item: kv 方向, change: "对 Lk 的并行归约 → N 次串行合并；并行只来自行组、head 与 batch", by: [2, 3]}
    passes:
      - {item: kernel, change: "3 → 1", by: [1, 2, 3, 4]}
    error:
      - {item: P 舍入, change: "+ P = bf16(E) 进入 P · V（2^−8），而分母 L 累加未舍入的 E", by: [3]}
      - {item: exp2 近似, change: "+ 1/8 的 E 元素带 ≤ 8.8e-5 的相对误差", by: [3]}

```

**2）FA4 · SM100 · 非因果双 CTA**

*   **来源**：同一 commit；非因果、双 CTA（要求 Lq > 256，且非 varlen、非 split-KV），对应 work\_schedule 版本一。
    
*   **内容**：与 5.1 相同的四步，差别只有三处：访问全部块，只屏蔽尾块中 k ≥ Lk 的位置；没有空行分支；多项式 exp2 覆盖 3/16 的元素。
    

```yaml
math_reshape:
  thesis: >-
    每行只带 (m, L, Y) 按 kv 块在线合并，S、P 从不整体存在；参考最大值允许落后至多 2^τ 倍，
    历史状态只在最大值跃升时重缩放。

  rewrite:
    - step: 1
      name: kv 分块
      math: |
        blk(j) = [j·Bk, (j+1)·Bk),    N = ⌈Lk / Bk⌉
        S(i) = Q[q] · K[blk(σ(i))]ᵀ,    V(i) = V[blk(σ(i))],    S(i)[k] = −∞  若 k ≥ Lk
      symbols:
        Bk: {meaning: kv 块长（由 work_schedule 决定）}
        σ:  {meaning: 块的访问次序（由 work_schedule 决定）}

    - step: 2
      name: 参考最大值允许滞后 τ
      math: |
        c = scale · log2(e),    τ = 8
        m(−1) = −∞,    m̂(i) = max(m(i−1), max S(i))
        m(i) = m(i−1)   若 m(i−1) > −∞ 且 c·(m̂(i) − m(i−1)) ≤ τ
               m̂(i)     否则
        α(i) = exp2(c·(m(i−1) − m(i)))
      symbols:
        m(i): {dtype: fp32, life: carried, meaning: 参考最大值}

    - step: 3
      name: 分母累加 E，分子累加 bf16(E)
      math: |
        E(i) = exp2(c·S(i) − c·m(i)),    P(i) = bf16(E(i))
        L(i) = α(i)·L(i−1) + Σ E(i),       L(−1) = 0
        Y(i) = α(i)·Y(i−1) + P(i)·V(i),    Y(−1) = 0
      symbols:
        L(i): {dtype: fp32, life: carried, meaning: 分母}
        Y(i): {shape: [d: 128], dtype: fp32, life: carried, meaning: 分子}
      arithmetic:
        approx: 3/16 的 E 元素（每块每行 128 列中的 24 列）用 exp2(x) ≈ 2^⌊x⌋·p₃(x − ⌊x⌋)，p₃ 为 [0, 1) 上的 3 次 minimax 多项式，最大相对误差 8.8e-5

    - step: 4
      name: 结束时一次归一化
      math: |
        O = bf16(Y(N−1) / L(N−1)),    LSE = ln2 · (c·m(N−1) + log2 L(N−1))

  ledger:
    baseline: 按 Contract 直算：物化 fp32 的 S = Q·Kᵀ 与 P = softmax(S)，再算 P · V
    traffic:
      - {item: S 与 P, change: "− 16 · Lq · Lk B：fp32 的 S、P 各写一次、读一次", by: [1, 3]}
    vector:
      - {item: Y 重缩放, change: "+ 每行每次参考更新 d 次乘法；首块之后每行最多 c · (M − m(0)) / τ 次，M 为该行最终最大值", by: [2, 3]}
    transcendental:
      - {item: exp2, change: "每个分数一次 ex2，其中 3/16 改为约 6 条 FMA/ALU 指令（含 packed f32x2）", by: [3]}
    footprint:
      - {item: 每行状态, change: "Lk 个分数 → d + 2 个 fp32：Y 的一行、m、L", by: [2, 3]}
    depth:
      - {item: kv 方向, change: "对 Lk 的并行归约 → N 次串行合并；并行只来自行组、head 与 batch", by: [2, 3]}
    passes:
      - {item: kernel, change: "3 → 1", by: [1, 2, 3, 4]}
    error:
      - {item: P 舍入, change: "+ P = bf16(E) 进入 P · V（2^−8），而分母 L 累加未舍入的 E", by: [3]}
      - {item: exp2 近似, change: "+ 3/16 的 E 元素带 ≤ 8.8e-5 的相对误差", by: [3]}

```

**3）GDN · FLA Triton 分块前向**

*   **来源**：fla-org/flash-linear-attention `954438d1`，`fla/ops/gated_delta_rule/` 与 `fla/ops/common/` 中的五个 kernel；chunk 64、dk = dv = 128、bf16 q/k/v；未安装 FlashQLA 时的默认路径。
    
*   **内容**：5 步，每步一个 pass。块内重置的 log2 累积衰减；严格下三角系数与以 bf16 存储的 (I + A)⁻¹；同一个逆作用于两个右端项得到 W、U；块间状态推进，状态 S 保持 fp32，快照 S̄ 与修正量 R 的两个 bf16 版本分别命名；由快照与块内项得到输出。ledger：串行链从 L 缩短到 L/64，递推从 vector 转到 matmul；代价是 MAC 约 1.8 倍，访存约为最小值的 4.5 倍。
    

```yaml
math_reshape:
  thesis: >-
    把逐 token 的秩一 delta 更新改写成 64-token 块上的 UT 变换：块内修正由一次单位下三角求逆解出，
    串行链从 L 缩短到 L/64，主体计算全部变成矩阵乘；代价是约 1.8 倍的 MAC 和写回 HBM 的中间量。

  rewrite:
    - step: 1
      name: 块内重置的累积衰减
      math: |
        blk(c) = [c·C, (c+1)·C),    NT = ⌈L / C⌉,    X(c) = X[blk(c)]，X ∈ {g, β, Q, K, V}
        G(c) = log2(e) · cumsum(g(c))
        γ(c) = exp2(G(c)),    δ(c) = exp2(G(c)[C−1] − G(c)),    Γ(c)[r, s] = exp2(G(c)[r] − G(c)[s])  (r ≥ s)
      symbols:
        C:    {value: 64, meaning: 块长；尾块取有效长度}
        G(c): {shape: [C], dtype: fp32, life: stored}

    - step: 2
      name: 块内依赖写成单位下三角系统
      math: |
        A(c)[r, s] = β(c)[r] · Γ(c)[r, s] · (K(c)·K(c)ᵀ)[r, s]  (r > s)；0  (r ≤ s)
        T(c) = bf16((I + A(c))⁻¹)
      symbols:
        T(c): {shape: [C, C], life: stored}
      arithmetic:
        accumulate: (I + A)⁻¹ 由 16×16 对角块前代（fp32）与子块合并（tf32 输入、fp32 累加）求得

    - step: 3
      name: 同一个逆作用于两个右端项
      math: |
        W(c) = bf16(T(c) · bf16(β(c) ⊙ γ(c) ⊙ K(c)))
        U(c) = bf16(T(c) · bf16(β(c) ⊙ V(c)))
      symbols:
        W(c): {life: stored}
        U(c): {life: stored}

    - step: 4
      name: 块间只推进一次状态
      math: |
        S(0) = S_init,    S̄(c) = bf16(S(c))
        R(c) = U(c) − W(c) · S̄(c),    R̄(c) = bf16(R(c))
        S(c+1) = γ(c)[C−1] · S(c) + K(c)ᵀ · bf16(δ(c) ⊙ R(c)),    S_final = S(NT)
      symbols:
        S(c): {shape: [dk: 128, dv: 128], dtype: fp32, life: carried, meaning: 进入块 c 前的状态}
        S̄(c): {life: stored}
        R̄(c): {life: stored}

    - step: 5
      name: 输出由快照与块内项组成
      math: |
        B(c) = bf16(tril(Γ(c) ⊙ (Q(c) · K(c)ᵀ)))
        o[blk(c)] = bf16(scale · (γ(c) ⊙ (Q(c) · S̄(c)) + B(c) · R̄(c)))

  ledger:
    baseline: 按 Contract 逐 token 递推：fp32 状态常驻片上，每个 token 做一次状态衰减、两次矩阵-向量乘与一次秩一更新
    depth:
      - {item: 串行链, change: "L 次秩一更新 → L/C 次块更新；步骤 1、2、3、5 对所有块并行，步骤 4 只在 batch、head 与 S 的列块之间并行", by: [2, 3, 4]}
    vector:
      - {item: 逐 token 递推, change: "− 每 token 3 · dk · dv 次 MAC 与 dk · dv 次乘法（约 4.9 万 MAC + 1.6 万乘法）", by: [4]}
    matmul:
      - {item: 块化计算, change: "+ 每 token 3 · dk · dv + C · (2.625 · dk + 2 · dv) 次 MAC（约 8.7 万；不计子块合并）", by: [2, 3, 4, 5]}
    transcendental:
      - {item: 衰减因子, change: "每 token 1 次 exp → 约 107 次 exp2；Γ 在步骤 2、5 各重算一次，而不是存储 C × C 矩阵", by: [1, 2, 5]}
    traffic:
      - {item: 多趟中间量, change: "每 token 每 head 约 1.0 KB → 至少 4.6 KB：G、T、W、U、S̄、R̄ 的写回与重读，K 读 4 次", by: [1, 2, 3, 4, 5]}
    passes:
      - {item: kernel, change: "1 → 5，每个 step 一个 kernel", by: [1, 2, 3, 4, 5]}
    error:
      - {item: bf16 中间量, change: "+ 五个存储量 T、W、U、S̄、R̄ 与四个矩阵乘操作数（β ⊙ γ ⊙ K、β ⊙ V、δ ⊙ R、B）各舍入到 bf16（2^−8）；S 与 R 本身保持 fp32", by: [2, 3, 4, 5]}

```

**4）MoE · vLLM TritonExperts · FP8 block**

*   **来源**：vllm-project/vllm `28282ffe`，`fused_moe/experts/triton_moe.py`、`fused_moe.py`；SM90、TP = 8、权重 128×128 块 scale、激活 1×128 动态量化，DeepSeek-V3 路由专家形状。
    
*   **内容**：6 步。1(A) 在展开前量化 X，与 1(B) 并行：槽位按专家成块、只存索引，小批量时每槽位一块；GEMM1 把两个 scale 提到 128 组内积之外；SiLU·mul 之后直接量化，不经 bf16；GEMM2 把路由权重乘在 fp32 累加器上；最后 top-k 合并。ledger：权重读取从每槽位一次降到每块一次，MMA 直接吃 fp8；代价是补齐行、6 个 pass，以及每 token 约 250 KB 的中间张量。
    

```yaml
math_reshape:
  thesis: >-
    把每个 (token, 专家) 槽位上的两次 GEMV 改写成按专家分组的 GEMM：分组只存在于索引表，不产生重排副本；
    FP8 的 scale 按 128 组提到内积之外，MMA 直接吃 fp8 码值；路由权重借线性并入 GEMM2 的 fp32 累加器。

  rewrite:
    - step: 1(A)
      name: 输入在展开前量化
      math: |
        (Xq, sX) = q(X)
      symbols:
        Xq: {shape: [T, H], dtype: fp8e4m3, life: stored}
        sX: {shape: [T, H/128], dtype: fp32, life: stored}

    - step: 1(B)
      name: 槽位按专家成块，只存索引
      math: |
        p = t·k + j,    t(p) = ⌊p / k⌋,    e(p) = idx[t, j],    w(p) = w[t, j]
        B(b) = {b},    ε(b) = e(b)                                            若 T·k·4 ≤ E
        B(b) = π[b·BM, (b+1)·BM),    ε(b) = e(π[b·BM])                        否则
        π = pad(P(0)) ‖ … ‖ pad(P(E−1)),    P(e) = { p | e(p) = e }
      symbols:
        pad: {meaning: 以哨兵 T·k 补到 BM 的倍数}
        BM:  {meaning: 每块槽位数，由 work_schedule 按 T 选取}
        π:   {shape: [≤ T·k + E·(BM − 1)], dtype: int32, life: stored}
        ε:   {shape: [块数], dtype: int32, life: stored}

    - step: 2
      name: scale 提到 128 组内积之外
      math: |
        H1[p, n] = bf16( Σ(g < H/128) sX[t(p), g] · S13[e][⌊n/128⌋, g] · ⟨Xq[t(p), g], W13[e][n, g]⟩ ),    p ∈ B(b)，p < T·k，e = ε(b)，n < 2I
      symbols:
        H1: {shape: [T·k, 2I], life: stored, meaning: "[gate | up]，按槽位存放"}
      arithmetic:
        accumulate: ⟨·, ·⟩ 在 FP8 MMA 的累加器中求和，Hopper 上其位宽低于 fp32（约 14 位）

    - step: 3
      name: 激活后直接量化，不经 bf16
      math: |
        A[p, i] = SiLU(H1[p, i]) · H1[p, I + i],    i < I
        (Aq, sA) = q(A)
      symbols:
        A:  {shape: [T·k, I], dtype: fp32}
        Aq: {shape: [T·k, I], dtype: fp8e4m3, life: stored}
        sA: {shape: [T·k, I/128], dtype: fp32, life: stored}

    - step: 4
      name: 路由权重乘在 fp32 累加器上
      math: |
        Y3[p, n] = bf16( w(p) · Σ(g < I/128) sA[p, g] · S2[e][⌊n/128⌋, g] · ⟨Aq[p, g], W2[e][n, g]⟩ ),    e = ε(b)，n < H
      symbols:
        Y3: {shape: [T·k, H], life: stored}
      arithmetic:
        accumulate: 同步骤 2

    - step: 5
      name: top-k 合并
      math: |
        Y[t] = bf16( Σ(j < k) Y3[t·k + j] )

  ledger:
    baseline: 按 Contract 逐槽位直算：每个槽位在片上完成 GEMV → SiLU·mul → q → GEMV → 加权累加，专家权重按槽位各读一次
    traffic:
      - {item: 专家权重, change: "T·k·3·H·I B → Σ(e) ⌈n_e / BM⌉ · 3·H·I B，n_e 为专家 e 的槽位数（不计 L2 复用）", by: ["1(B)", 2, 4], when: "T·k·4 > E"}
      - {item: 中间张量, change: "+ 每 token 2·k·(2·2I + I + 2·H) B：H1、Aq、Y3 各写一次读一次；本形状约 250 KB，其中 Y3 占 92%，T = 4096 时约为权重流量的 1/3", by: [2, 3, 4, 5]}
    matmul:
      - {item: FP8 MMA, change: "反量化后的 bf16 MMA → 直接用 fp8 码值的 MMA（SM90 峰值 2 倍）", by: [2, 4]}
      - {item: 补齐行, change: "+ 每个 GEMM 至多 E·(BM − 1) 行空算", by: ["1(B)", 2, 4], when: "T·k·4 > E"}
      - {item: 空行, change: "+ 每槽位 BM − 1 行空算", by: ["1(B)", 2, 4], when: "T·k·4 ≤ E"}
    vector:
      - {item: 反量化, change: "每槽位 3·H·I 次逐元素反量化 → 3·H·I / 64 次 scale 乘法（每个输出元素每组 2 次）", by: [2, 4]}
    passes:
      - {item: kernel, change: "1 → 6：量化 X、分组、GEMM1、激活与量化、GEMM2、合并", by: ["1(A)", "1(B)", 2, 3, 4, 5], when: "T·k·4 > E"}
      - {item: kernel, change: "1 → 5：不做分组", by: ["1(A)", "1(B)", 2, 3, 4, 5], when: "T·k·4 ≤ E"}
    error:
      - {item: 中间舍入, change: "+ H1 与每个槽位的 Y3 各舍入一次到 bf16（2^−8）；A 由 fp32 直接量化，不增加舍入", by: [2, 3, 4]}

```

### Contract algorithm的YAML语言表述

**Contract**区，包括algorithm requirement，以及 hardware specific，planner和coder都不可修改

纯数学要求，纯硬件规格

1）algorithm requirement：算法要求的计算I/O、从外向内作为黑盒抽象的算子定义。包括：math definition，也就是数学语义的公式表达、数学记号、特殊语义；输入输出和内部涉及的变量命名、意义状态；基于Pytorch等实现的golden reference的oracle（通过文件索引地址来记录，精简代码）；代表性的shape，输入的workload，目标优化的指标latency/throughput、精度的容忍范围等特殊要求和target；**不看 Design，能否仅凭这一段写出一个朴素 reference，而不需要猜掩码、量化、路由和状态语义**

从 FA4、MoE、GDN 三个已有 case 的调用边界与 reference 出发，先写出各自的算法契约，再归纳共同字段：`constants / inputs / outputs / compute / reference`。

`compute` 沿用 math reshape 中 `rewrite` 的步骤格式，但表达的是**黑盒必须满足的计算**，不是优化后的实现方法。

#### 1. 核心思想

**1.1 这一层做什么**

algorithm\_requirement 属于 Contract：从调用者的视角，定义算子接收什么、返回什么，以及输入与输出之间必须满足的数学关系。Planner 与 Coder 都不能自行修改。

判断是否写完整，不是看它有没有覆盖源码中的每个函数，而是：**不看 Design，能否仅凭这一段写出一个朴素 reference，而不需要猜掩码、量化、路由和状态语义？**

一个算法契约可以跨越多个源码函数，也可以只覆盖某个函数的一种配置。边界来自选定的计算任务，不直接照搬现有的 kernel 划分。

**1.2 格式怎么来的**

三个 case 分别暴露了不同的必要信息：

| Case | 不能省略的外部语义 |
| --- | --- |
| FA4 | Q/KV 长度、因果对齐、无可见 key 时的结果、LSE 定义。 |
| MoE | 量化码值与 scale 的解释、给定路由的使用方式、输出是否只是权重分片的部分和。 |
| GDN | 门控输入的含义、初始状态、更新与读出的先后关系、最终状态。 |

这些差异不需要变成三套字段。固定数值放进 `constants`；对象含义与取值限制跟随 `inputs / outputs`；计算关系写进 `compute`；对照代码与参数映射放进 `reference`。

因此，不再设置顶层 `symbols` 或单独的 `domain`；不在 `reference` 外套一层 `check / oracle`；也不写没有新增约束的 `target: end_to_end_latency`。

**1.3 与 Design 的边界**

`algorithm.compute` 写“要求算什么”，`math_reshape.rewrite` 写“怎样改写这份计算”。

例如，因果可见性、MoE 的量化函数、GDN 的状态递推属于 Contract；在线 softmax、按专家分组、块内三角求解属于 Design。量化组长决定数值含义，不能当成普通的 GEMM tile size 自由调整。

`compute` 中出现一个中间量，只是在定义数学关系，**不要求实际分配、保存或物化这个量**。因此，本区不继承 `life / arithmetic / ledger`，也不记录 kernel 数量、存储位置或执行组织。

#### 2. 字段划分

| 字段 | 记录内容 |
| --- | --- |
| `constants` | 本份契约固定的数值，每项使用 `value / meaning`。 |
| `inputs` | 调用者提供的对象，每项使用 `meaning / shape / dtype`，必要时加 `range / default`。 |
| `outputs` | 调用者得到的对象，每项使用 `meaning / shape / dtype`。 |
| `compute` | 从输入到输出的计算定义，由 `step / name / math / symbols` 组成。 |
| `reference` | 对照代码的位置，以及契约对象与函数参数、返回值之间的映射。 |

**2.1 constants、inputs 与 outputs**

`meaning` 用一两个词或一个短语表达对象的角色，例如“隐层维度”“专家索引”“分片部分和”，不重复解释公式。

`constants` 只放固定值。函数在首次使用处定义，dtype 在对象上声明，不把函数说明、类型解释混进常量表。固定尺寸可以直接写在 shape 中，不必为了命名而重复登记。

输入、输出与常量的名称属于当前算法实例，不是模板新增的专属字段。

**2.2 compute**

每个 step 包含：

| 字段 | 内容 |
| --- | --- |
| `step` | 步骤编号；互不依赖的兄弟步骤写 `1(A)`、`1(B)`。 |
| `name` | 一句话说明这一步的计算关系，例如“下投影后按原路由权重求和”。 |
| `math` | 定义、递推、分情形与显式 cast，是本步的主体。 |
| `symbols` | 可选。只补充公式中看不出的局部记号含义。 |

`symbols` 不重复登记已经由公式说明的中间量，也不填写存储生命周期。没有额外信息就省略。

**2.3 reference**

`reference` 固定使用三个字段：

*   `source`：`仓库@commit:文件路径::函数名`，固定代码版本与入口。
    
*   `inputs`：左侧为函数参数名，右侧为契约对象、转换表达式或固定参数。
    
*   `outputs`：左侧为契约输出名，右侧为函数返回值或转换表达式；`result` 表示返回值。
    

这里的映射是调用说明，不是可执行的 YAML。未显式指定的参数采用固定版本的默认值；改变本例语义的非默认参数必须写出。

#### 3. 语法规则

**3.1 Shape 与范围**

Shape 使用有序、带名称的维度列表：

```yaml
X: {meaning: 输入特征, shape: [token: T, hidden: H], dtype: bf16}

```

固定规模写实值或引用 `constants`；运行时规模使用符号。同名维度在同一份 algorithm 内表示相同的大小，不同 case 的符号作用域独立。

`range` 只在需要限制输入取值时填写。`[0, 1]` 表示两端包含；含符号或开端点的区间写成字符串，例如 `"[0, E)"`。`default: 0` 用于张量时，表示缺省为给定 shape 的全零张量。

本节三个 case 均限定为正整数规模、有限数值输入；没有写出的运行时尺寸上界不额外限定。公式明确规定的 `−∞` 掩码与输出不受有限输入约定影响。

**3.2 步骤与依赖**

步骤按计算关系排列，从 Contract 的输入一直写到全部输出。列表顺序表达数学依赖，不承诺执行必须串行；`1(A) / 1(B)` 也不要求拆成两个 kernel。

递推直接写出初值与状态转移，例如：

```text
S(0) = S_init
S(t+1) = ...

```

编号不是执行调度。GDN 可以先定义完整的状态递推，再定义各位置如何读出；这不要求实现先保存全部状态、再启动输出计算。

**3.3 数学表达**

记号沿用 math reshape：`X[t]` 表示下标，`S(t)` 表示递推实例；`· / ⊙ / ᵀ / ‖` 分别表示矩阵乘或数乘、逐元素乘、转置、拼接；按范围求和写成 `Σ(范围) 表达式`。

公式可以省略相互独立的 batch/head 下标，但归约维度、迭代范围与边界分支必须明确。`exp` 使用自然底数，`logsumexp` 使用自然对数；FA4 的 softmax 与 logsumexp 均沿 key 维计算。

辅助函数在首次使用处定义，后续直接引用。`dq(q(z))` 表示把 `q` 返回的码值与 scale 传给 `dq`，不要求真的生成反量化副本。

`math` 不写 LaTeX 源码、torch 的视图操作或说明性段落；对象解释放在 `meaning`，分支使用“若 / 否则”。

**3.4 精度**

沿用已有约定：没有显式 cast 的计算默认使用 FP32 计算与累加，舍入为 RNE；`bf16(·)` 表示舍入到 BF16，`fp8(·)` 表示舍入到 E4M3FN 并饱和到 ±448。

`q(·)` 不等于 `fp8(·)`：前者还包含分组、scale 计算及数值编码规则，必须由 Contract 定义。

这里给出的是参考计算的精度口径，不要求实现使用相同的指令或物化相同的中间量。Design 引入的有限精度差异仍须满足另外确定的误差要求。

#### 4. 原则

1.  **一个对象、一处定义。** 固定值、外部对象、局部记号分别放在自己的位置；下游直接引用，不重新解释或改变含义。
    
2.  **一步一段计算关系，不按源码 pass 切分。** Contract 不描述优化主张，也不因当前实现拆成多个 kernel 就增加多个步骤。
    
3.  **删重复，不删边界。** 无可见 key、路由权重的使用位置、量化 scale 的解释、初末状态等会改变结果，必须留下；泛化目标、实现细节与没有内容的字段不写。
    
4.  **数学定义、reference 与数值验收不能混为一谈。** Reference 必须与选定的计算边界和配置一致，不能直接把待优化实现当作独立 golden。输出 dtype 也不能代替误差指标与阈值；本轮尚未定稿的 `accuracy` 不填空壳，不据此宣称验收已经完成。
    

#### 5. 具体case

下面仅展开 `contract.algorithm`，省略外层 `contract`。三个例子均包含常量、输入、输出、计算定义与 reference 映射。

**1）FA4 · SM100 · 因果**

**来源**：Dao-AILab/flash-attention `ce088ab9`；对应已有 FA4 因果设计，reference 为 `flash_attn/cute/testing.py::attention_ref`。

**范围**：8 个 MHA head，Q/K/V 均为 128 维；同次调用内各序列等长，但 Q 与 KV 长度可以不同。因果采用右下对齐，返回 O 与自然对数 LSE。Reference 以 FP32 输入调用，避免 LSE 被转换回 BF16。

```yaml
algorithm requirement:
  constants:
    scale: {value: "1/√128", meaning: 分数缩放}

  inputs:
    Q: {meaning: 查询向量, shape: [batch: BatchCount, query: QueryLength, head: 8, d: 128], dtype: bf16}
    K: {meaning: 键向量,   shape: [batch: BatchCount, kv: KVLength, head: 8, d: 128], dtype: bf16}
    V: {meaning: 值向量,   shape: [batch: BatchCount, kv: KVLength, head: 8, d: 128], dtype: bf16}

  outputs:
    O:   {meaning: 注意力输出,   shape: [batch: BatchCount, query: QueryLength, head: 8, d: 128], dtype: bf16}
    LSE: {meaning: 对数归一化项, shape: [batch: BatchCount, head: 8, query: QueryLength], dtype: fp32}

  compute:
    - step: 1
      name: 按右下对齐定义因果分数
      math: |
        visible(q, k) = 0 ≤ k < KVLength 且 k ≤ q + KVLength − QueryLength
        Z[q, k] = scale · Q[q] · K[k]ᵀ  若 visible(q, k)；否则 −∞

    - step: 2
      name: 对可见位置归一化并加权求和
      math: |
        O[q] = bf16(softmax(Z[q]) · V),    LSE[q] = logsumexp(Z[q])  若 ∃ k：visible(q, k)
        O[q] = 0,                        LSE[q] = −∞               否则

  reference:
    source: Dao-AILab/flash-attention@ce088ab9:flash_attn/cute/testing.py::attention_ref
    inputs: {q: "fp32(Q)", k: "fp32(K)", v: "fp32(V)", causal: true, return_lse: true}
    outputs: {O: "bf16(result[0])", LSE: "result[2]"}

```

**2）MoE · vLLM TritonExperts · FP8 block**

**来源**：vllm-project/vllm `28282ffe`；对应已有 DeepSeek-V3 路由专家、TP=8 的分片规格，reference 入口为 `tests/kernels/utils.py::torch_experts`。

**范围**：输入量化、专家计算与本分片内的 top-k 加权合并均在边界内；路由选择与跨分片归约在边界外。`I=256` 是本分片的中间维度，`Y` 是分片部分和，不是完整模型层的最终输出。`W13` 按门控、上投影的顺序拼接。

**量化基准**：采用本轮选定的普通 FP32 scale，amax 下界为 `1e−10`，scale 不取整为 2 的幂。两次激活量化引用同一个 `q`，不把融合 kernel 特有的数值规则反过来当成算法要求。

```yaml
algorithm requirement:
  constants:
    H: {value: 7168, meaning: 隐层维度}
    I: {value: 256,  meaning: 分片中间维度}
    E: {value: 256,  meaning: 专家数}
    k: {value: 8,    meaning: 路由槽位数}
    G: {value: 128,  meaning: 量化组长}

  inputs:
    X:   {meaning: 输入特征, shape: [token: T, hidden: H], dtype: bf16}
    idx: {meaning: 专家索引, shape: [token: T, slot: k], dtype: int32, range: "[0, E)"}
    w:   {meaning: 路由权重（原值）, shape: [token: T, slot: k], dtype: fp32}
    W13: {meaning: 门控与上投影权重, shape: [expert: E, out: 2I, in: H], dtype: fp8e4m3fn}
    W2:  {meaning: 下投影权重,       shape: [expert: E, out: H, in: I], dtype: fp8e4m3fn}
    S13: {meaning: W13 块缩放, shape: [expert: E, out_group: 2I/G, in_group: H/G], dtype: fp32, range: "(0, +∞)"}
    S2:  {meaning: W2 块缩放,  shape: [expert: E, out_group: H/G, in_group: I/G], dtype: fp32, range: "(0, +∞)"}

  outputs:
    Y: {meaning: 分片部分和, shape: [token: T, hidden: H], dtype: bf16}

  compute:
    - step: 1(A)
      name: 输入按行分组量化
      math: |
        blk(g) = [g·G, (g+1)·G)
        q(z) = (zq, s)
          s[g] = fp32(max(max|z[blk(g)]|, 1e−10) / 448)
          zq[i] = fp8(z[i] / s[⌊i/G⌋])
        dq(zq, s)[i] = fp32(zq[i]) · s[⌊i/G⌋]
        x[t] = dq(q(X[t]))
      symbols:
        z: {meaning: 一行待量化值}

    - step: 1(B)
      name: 按块 scale 解释专家权重
      math: |
        Ŵa[e, r, c] = fp32(Wa[e, r, c]) · Sa[e, ⌊r/G⌋, ⌊c/G⌋],    a ∈ {13, 2}

    - step: 2
      name: 按给定路由计算门控与上投影
      math: |
        u[t,j] ‖ v[t,j] = x[t] · Ŵ13[idx[t,j]]ᵀ
      symbols:
        u: {meaning: 门控分量}
        v: {meaning: 上投影分量}

    - step: 3
      name: 门控乘积重新量化
      math: |
        A[t,j] = dq(q(SiLU(u[t,j]) ⊙ v[t,j]))

    - step: 4
      name: 下投影后按原路由权重求和
      math: |
        Y[t] = bf16(Σ(0 ≤ j < k) w[t,j] · (A[t,j] · Ŵ2[idx[t,j]]ᵀ))

  reference:
    source: vllm-project/vllm@28282ffe:tests/kernels/utils.py::torch_experts
    inputs: {a: "fp32(X)", w1: W13, w2: W2, topk_weight: w, topk_ids: idx,
             w1_scale: S13, w2_scale: S2, quant_dtype: fp8e4m3fn, block_shape: [G, G]}
    outputs: {Y: "bf16(result)"}

```

这里以 FP32 的 X 调用 reference，是为了让非量化中间量保持 FP32，而不把 BF16 测试路径的中间舍入固定为算法要求。

**待完成**：`torch_experts` 仍调用共享的量化与激活 helper，需要固定上述普通 scale 分支并隔离这些 helper。当前记录的是 reference 的适配起点，不是已经独立、已经验收的 golden；现有 Design 与这份量化基准的相容性仍需验证。

**3）GDN · FLA Triton 分块前向**

**来源**：fla-org/flash-linear-attention `954438d1`；对应已有 GDN 分块前向设计，reference 为 `fla/ops/gated_delta_rule/naive.py::naive_recurrent_gated_delta_rule`。

**范围**：等长序列、相同 head 数、dk=dv=128。Q/K 不在本算子内归一化；g 是单步自然对数衰减，β 是已激活的更新系数；接收初始状态并返回最终状态。这里定义逐 token 的数学递推，不固定 chunk=64 或分块算法。

```yaml
algorithm requirement:
  constants:
    scale: {value: "1/√128", meaning: 读出缩放}

  inputs:
    Q:      {meaning: 查询向量, shape: [batch: B, token: L, head: H, dk: 128], dtype: bf16}
    K:      {meaning: 键向量,   shape: [batch: B, token: L, head: H, dk: 128], dtype: bf16}
    V:      {meaning: 值向量,   shape: [batch: B, token: L, head: H, dv: 128], dtype: bf16}
    g:      {meaning: 单步对数衰减, shape: [batch: B, token: L, head: H], dtype: fp32, range: "(-∞, 0]"}
    β:      {meaning: 更新系数（已激活）, shape: [batch: B, token: L, head: H], dtype: fp32, range: [0, 1]}
    S_init: {meaning: 初始状态, shape: [batch: B, head: H, dk: 128, dv: 128], dtype: fp32, default: 0}

  outputs:
    o:       {meaning: 读出向量, shape: [batch: B, token: L, head: H, dv: 128], dtype: bf16}
    S_final: {meaning: 最终状态, shape: [batch: B, head: H, dk: 128, dv: 128], dtype: fp32}

  compute:
    - step: 1
      name: 先衰减历史状态，再写入 delta 修正
      math: |
        S(0) = S_init
        Ŝ(t) = exp(g[t]) · S(t)
        r(t) = β[t] · (V[t] − K[t] · Ŝ(t))
        S(t+1) = Ŝ(t) + K[t]ᵀ · r(t),    0 ≤ t < L
      symbols:
        S(t): {meaning: 写入 token t 前的状态}

    - step: 2
      name: 从更新后状态读出，并返回最终状态
      math: |
        o[t] = bf16(scale · Q[t] · S(t+1)),    0 ≤ t < L
        S_final = S(L)

  reference:
    source: fla-org/flash-linear-attention@954438d1:fla/ops/gated_delta_rule/naive.py::naive_recurrent_gated_delta_rule
    inputs: {q: Q, k: K, v: V, g: g, beta: β, scale: scale,
             initial_state: S_init, output_final_state: true}
    outputs: {o: "bf16(result[0])", S_final: "result[1]"}

```

### Contract hardware的YAML语言表述

#### 1. 定位

`hardware_specific` 属于 Contract，绑定**目标设备、核心硬件能力，以及本任务使用的 wiki 入口**。Planner 与 Coder 可以查阅资料、选择设计，但不能自行更换设备、实现语言或任务指定的资料版本。

`device / arch / features` 描述硬件事实，同一卡型共用一份，不随算法或 Design 改变。`wiki` 绑定当前设备与已确定实现栈所需的资料；只记录位置，不在 Plan 中复制手册、测试结果或参考实现。

**能力摘要与资料入口分开。** `features` 让读者迅速了解可行的计算、存储和协作方式；完整参数、编程规则、实测表现、参考案例与调试方法由 wiki 提供。

#### 2. 字段与填写规则

| 字段 | 内容 |
| --- | --- |
| `device` | 明确的设备型号，必要时包含产品版本。 |
| `arch` | 设备架构标识。 |
| `features` | 能力名称到简短描述的映射，说明关键行为、作用域或限制。 |
| `wiki` | 固定的资料入口，分为 `hardware / programming / microbench / sota / profile`。 |

**2.1 features：能力摘要，不是设计选择**

每条用一句话说明能力及必要边界，不只列名称，也不展开成手册。筛选标准是：**这条信息是否改变可行的计算、存储或协作方式？** 不是“当前 kernel 使用了什么”。

“支持 CTA-pair MMA”属于这里；“采用双 CTA”“使用三重缓冲”“某单元是当前瓶颈”属于 Design。软件栈暂时无法表达某项能力，不能写成硬件不支持。

摘要不是完整白名单。未列出不代表不支持；SM 数量、资源容量、分配粒度、完整类型组合等查 `wiki.hardware`，不能因摘要省略而忽略约束。本次运行额外限制的可用资源，不改写设备规格。

**2.2 wiki：按知识层次绑定入口**

| 字段 | 对应资料 |
| --- | --- |
| `hardware` | 当前卡型的完整 datasheet：计算、存储、访存、通信、同步与执行组织的规格、能力及限制。绑定一个具体文件。 |
| `programming.language` | 本任务选定的一种语言或 DSL 的语法、编程模型、接口与编译使用规则，例如 CUDA C++ 或 CuTe DSL。 |
| `programming.template` | 当前实现栈使用的模板库及抽象接口，例如 CUTLASS、CuTe 的使用方法与约束。 |
| `programming.virtual_isa` | 虚拟指令集的语义、操作数、支持条件与使用规则，例如 PTX。 |
| `programming.machine_isa` | 目标架构机器指令的含义、编码或反汇编表示，例如 SASS。 |
| `microbench` | 软件操作与硬件表现的实测关联；包含测试配置、结果与复现实验入口。 |
| `sota.kernel` | 当前卡型上的算子库与参考实现索引。 |
| `sota.schema` | 当前卡型上的 Kernel schema 设计案例。 |
| `profile` | 性能剖析与正确性调试的方法：工具使用、数据采集、报告与 dump 的含义，以及分析和定位流程。 |

`hardware` 由统一的硬件规格仓库维护，按代际组织目录、按卡型保存文件，例如 `Blackwell/B200.md` 或 `Blackwell/B200.json`。B200 与 B300 分别绑定自己的条目；不为不同 kernel 或软件版本重复建立 datasheet。

`programming` 保留四个入口，分别对应语言、模板库、虚拟 ISA 和机器 ISA。每个入口可以指向独立仓库，也可以指向仓库内的目录。`language` 绑定已经选定的实现语言，不是供 Agent 自由切换的候选列表。

`microbench` 的内部分类由仓库维护，本区不继续展开。`sota` 区分参考代码与设计案例；它们供查阅和迁移，不自动成为本任务的 golden reference，也不要求照搬其设计。

`profile` 暂不设子字段。NCU、NSYS、Sanitizer、PyTorch Profiler、IKET 等资料在该入口下按工具和诊断问题组织。Schema 不逐项登记工具，也不存放本次任务产生的报告或 dump。

**2.3 引用：每个叶子字段只记录一个位置**

引用统一使用字符串：

```text
repository@ref:path

```

`ref` 可以是约定的 tag、branch 或 commit；`path` 定位文件或目录，仓库根目录可写 `.`。版本由引用确定，不再增加独立的 `version` 字段；编程资料、microbench 与实际实现环境应匹配。

每个叶子字段只放一个入口，不写网址列表，也不展开成 `repo / version / path` 对象。一个入口需要覆盖多份资料时，由对应 wiki 的索引组织，Agent 进入后自行检索。

适用但尚未整理的资料使用尖括号占位；不适用或未选用的项直接省略，例如没有使用模板库时省略 `programming.template`。占位不表示资料已经存在，省略也不表示硬件不支持某项能力。

#### 3. 具体 case：NVIDIA B200 · CUDA C++

下面省略外层 `contract`。示例选择 CUDA C++，语言资料以 `cuda13.0` 为版本标签；所有 wiki 位置均为占位，不表示仓库或资料已经建立。

```yaml
hardware_specific:
  device: NVIDIA B200
  arch: SM100

  features:
    tcgen05.mma: Single-thread asynchronous issue; TMEM accumulators; single-CTA or CTA-pair operation within a cluster.
    TMEM: Dedicated Tensor Core storage accessed through specialized instructions, distinct from general SMEM.
    TMA: Asynchronous tensor-map transfers between GMEM and SMEM; cluster multicast.
    cluster: Cross-CTA shared-memory loads, stores, and atomics within a cluster.
    CLC: Attempts to cancel unstarted CTAs or clusters in the same grid and returns their indices on success.
    L1/SMEM: Shared physical capacity; SMEM carveout trades against available L1 capacity.

  wiki:
    hardware: "<wiki-hardware@ref:Blackwell/B200.md>"

    programming:
      language: "<wiki-language@cuda13.0:CUDA-Cpp/>"
      template: "<wiki-template@ref:CUTLASS-CuTe/>"
      virtual_isa: "<wiki-virtual-isa@ref:PTX/>"
      machine_isa: "<wiki-machine-isa@ref:SASS/SM100/>"

    microbench: "<wiki-microbench@ref:B200/>"

    sota:
      kernel: "<wiki-sota-kernel@ref:B200/>"
      schema: "<wiki-sota-schema@ref:B200/>"

    profile: "<wiki-profile@ref:NVIDIA/>"

```

### Realization的YAML语言表述

#### 1. 核心思想

**1.1 这一层做什么**

Realization 是 Planner 与 Coder 之间的实现交接区：只记录 **Contract 和 Design 尚未表达、但需要双方共享的实现决定**。它不是第四个 Design 区，也不是源码剩余细节的收纳区。

Contract 规定任务与硬件；Design 确定数学改写、工作分配与流水线；Realization 补充既定设计依赖的实现条件，以及少数值得优先尝试的实现选择。具体代码组织、地址计算、barrier 编号与 phase、精确 layout 等，默认留给 Coder。

**1.2 格式怎么来的**

沿用前面的方法：先对照三个固定版本的源码，分别写出 realization，再归纳共同字段，不预先建立指令、布局、编译器等分类。

三个 case 暴露了不同的边界：FA4 的原语选择与专职 warp、异步发射及 TMEM 数据流耦合；MoE 的 FP8 计算路径已经由 math\_reshape 表达，没有必要再写一遍；GDN 的指数求值可以保留为与重计算代价相关的实现建议。

因此，沿用 `require / prefer / forbid` 三类，将每条决定统一压成 `item / choice / reason`。**源码采用了某个选择，只能证明它存在；是否需要交接、是否必须采用，还要根据它与 Design 的关系判断。**

#### 2. 字段划分

**顶层只区分约束强度，不区分实现类别：**

| 字段 | 记录内容 |
| --- | --- |
| `require` | 既定 Design 成立所需的硬条件，Coder 不得自行替换。 |
| `prefer` | 与既定设计的代价直接相关的优先选择，Coder 可以用实验支持替代方案。 |
| `forbid` | Planner 已考虑并明确排除、值得避免再次尝试的路径。 |

**三类条目共用相同结构：**

| 字段 | 记录内容 |
| --- | --- |
| `item` | 选择作用于哪个对象或计算，直接引用上游名称。 |
| `choice` | 具体的实现选择或兼容性要求。 |
| `reason` | 这一选择依赖哪个设计前提，或影响哪项既定代价。 |

`item` 可以是单个名称或名称列表；同一选择和理由适用于多个对象时合写，不逐项复制。对象名称与原语名称是当前实例的内容，不为它们新增模板字段。

本区不另设 `thesis`：设计主张已经在 Design 中，单项选择的依据由 `reason` 说明。

#### 3. 填写与反馈规则

**一个决定只写一次。** 数学公式、舍入点、tile 大小、warp 角色、buffer 深度和依赖边已经由上游确定，不在这里重复。Realization 的省略不取消这些约束。

**是否保留，先做替换测试。** 换掉这一选择后，若既定角色、数据流或重叠方式不再成立，进入 `require`；若设计仍然成立，但有具体理由优先采用，进入 `prefer`；其余留给 Coder。`reason` 必须说明这种关系，不能只写“提升性能”。

**空区直接省略。** 三类字段均可省略；整个区域没有新增决定时，可以省略 `realization`，示例中用 `realization: {}` 明确表示没有额外交接项。`forbid` 不机械地列出 `require` 的反面，也不收集所有可能的错误实现。

**反馈只附在** `**prefer**` **条目下。** 可选字段 `feedback` 内只保留 `choice / evidence`：前者记录 Coder 提出的替代选择，后者引用固定版本的实验记录。原条目的 `choice / reason` 保留，完整的编译配置、精度结果与性能数据放在实验记录中；没有实验依据就不填写反馈。

`feedback` 不是修改上游设计的入口。若替代方案需要移动舍入点、增加搬运、改变角色或 buffer 生命周期，应回到对应的 Design 区域讨论；若 `require` 无法实现，应报告阻碍，不能自行降为 `prefer`。

#### 4. 具体 case

下面只展开三个案例的 `realization`，Contract 与 Design 沿用前文。FA4 选择 work\_schedule 版本二，不混入版本三的 CLC 调度。本轮是固定版本源码的静态核对，没有 GPU 实验反馈；`prefer` 是实现起点，不是已经验收的性能结论。

**1）FA4 · SM100 · 因果单 CTA**

**来源**：Dao-AILab/flash-attention `ce088ab9`，`flash_attn/cute/flash_fwd_sm100.py`、`softmax.py`；因果、单 CTA、hdim 128、bf16，对应 work\_schedule 版本二。

保留两项原语硬条件与一项求值建议，不重复单 CTA、双 Q、流水线顺序和缓冲关系。

```yaml
realization:
  require:
    - item: [Q·Kᵀ, P·V]
      choice: tcgen05.mma
      reason: 独立 MMA warp 发射，累加器经 TMEM 由 Softmax、Correction 消费。

    - item: [Q, K, V]
      choice: TMA
      reason: 专职搬运 warp 通过 tensor map 发射整块传输，不承担逐段搬运。

  prefer:
    - item: E 的多项式求值
      choice: fma.rn.f32x2
      reason: 成对求值，压低从 SFU 转移到 ALU 的指令发射开销。

```

多项式本身、覆盖比例及误差界已经属于 math\_reshape；这里仅建议其求值方式。替换 packed FMA 不等于允许取消多项式近似或修改其精度要求。

**2）MoE · vLLM TritonExperts · FP8 block**

**来源**：vllm-project/vllm `28282ffe`，`fused_moe/experts/triton_moe.py`、`fused_moe.py`；SM90、TP=8，权重 128×128 块 scale、激活 1×128 动态量化，沿用已有分片规格。

```yaml
realization: {}

```

空区是完整结果，不是待填项。FP8 码值直接参与 MMA、scale 在组内积之后相乘、激活后直接量化、路由权重在最终舍入之前相乘，都已经由 math\_reshape 表达，再写 `require` 或 `forbid` 没有新增信息。

本例不额外固定具体 MMA 指令族，也不把源码使用 Triton 变成实现限制。空区只表示没有额外交接项，不表示已有 Design 已通过数值验收；它与 Contract 量化基准的相容性仍沿用前文的待验证状态。

**3）GDN · FLA Triton 分块前向**

**来源**：fla-org/flash-linear-attention `954438d1`，`fla/ops/gated_delta_rule/`、`fla/ops/common/` 及 `fla/ops/utils/op.py`；chunk 64、dk=dv=128、bf16 Q/K/V，沿用已有 Triton 分块前向分支。

```yaml
realization:
  prefer:
    - item: [γ, δ, Γ]
      choice: ex2.approx
      reason: 块内衰减因子反复求值，优先采用硬件指数路径而非通用函数实现。

```

已有 ledger 记录了衰减因子重计算带来的指数求值代价，因此保留这一建议；块长、五个 pass、状态与中间量的精度、求逆合并中的 TF32 输入均已在 Design 中确定，不在这里重复。

`ex2.approx` 表示优先尝试的原语家族，不额外授权 `ftz` 或放宽误差要求。是否采用仍须接受既有数值验收。

## 9 工作

**四个主要的待做：**

1）完整的初版schema的语法和字段（结束，关键问题在于dispatch的拆分逻辑、schema版本管理）

2）kernel的算子库+schema库的建设，并在这个过程中优化template（通过什么？GPT pro？）

3）wiki的分门别类的整理，在B300建设比较完整的wiki

4）planner在AKA的具体引入，对于AKA的修改，以及消融实验

**对于（1）**

*   [x] schema的dispatch逻辑、version和分支的处理
    
*   [x] 完整的仓库README。介绍schema的背景逻辑，核心设计，预期的阅读和使用方法。预期的仓库大概包括，转换前的sota Kernel、转换后的schema wiki、转换的具体指导/规则/模板、实现批量转换的基础设施
    
*   [x] 基于6个md的完整schema的 template文件，主要包括 md指导+约束模板+Python检查器，其中模板负责字段合法性约束，而PY负责Kernel本身合法性的约束，以及用来画pipeline的py
    
*   [x] 分类的流程文件（skill/agent规则）；主要有三类，第一，指导agent怎么离线的基于已有的sota Kernel提取schema（提取）；第二，指导planner怎么在线根据schema wiki、md文件来从零撰写、或者逐步优化schema，两个步骤；第三，指导coder怎么根据已有的schema设计分别来从零实现schema对应的Kernel、或者逐步根据设计修改优化Kernel。也就是 extract、design、optimize和implement schema四个MD
    
*   [ ] 进一步在AKA的典型算子上开展schema注入的消融实验，包括GEMM、MoE、FMHA、MLA、Sparse、Linear Attn六类典型算子，基于AKA外源的main新的主线（light）进行消融比对
    

**这一步消融实验的预期想法是，首先告诉agent遵照extract的schema，开始进行implement，也就是所有设计好的schema应当支持让agent从零开始写**

**几个预期的实验方法：**

1）从零开始，无schema，直接迭代kernel；停止条件是到该算子的已知SOTA延时的110%

2）从零开始，有schema，先明确要求implement而没有自由权；停止条件同上

3）从零开始，有schema，但是只要求作为参考，不要求复现；停止条件同上

4）从零开始，无schema，但是先design schema，再迭代kernel进行复现，并optimize；停止条件同上

\*\*5）从SOTA开始，无schema，直接迭代；停止条件是该算子已知SOTA延时的80%

\*\*6）从SOTA开始，有schema，先明确要求optimize然后再implement的两阶段，停止条件同上

\*\*7）从SOTA开始，有schema，直接同时迭代schema+kernel，不强制，草稿辅助作用，停止条件同上

**选定的6个实验算子，在B300收集的SOTA版本**

GEMM：fp8\_scale\_group\_gemm

MoE：nvfp4\_fused\_moe

FMHA：fmha\_mxfp8\_qkv

MLA：mla\_expanded\_fp8

Sparse：dual\_cached\_latent\_bf16

Linear：gdn\_prefill\_bf16\_state\_bf16

**对于（2）**

*   [x] sota Kernel算子库（Attn、MoE、GEMM等），睿博提供的AKA产出的算子库
    
*   [x] 基于轻量Harness的自动化schema extract转换工具，可以支持随时断点继续，以及任意dispatch，supervisor+worker+reviewer，6个phase的阶段（非agent产物），能保证较好的提取效果
    
*   [ ] sota schema库（对应的，更新可能的template的不足）
    
*   [ ] megakernel的支持
    

**在6个典型算子并行提取的实际extract流程发现的问题：**

1）dispatch的分支太多，导致消耗的token明显，周期长，时间久

2）schema的一些字段，agent倾向于加入大量的文字描述

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/4ceadce7-12b7-4915-9df6-d20df1eb9e46.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/e922423b-85e5-456c-91c7-621c3e24d8b0.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/0ddbf2e9-6c74-4a28-9c0d-5441cd0b158a.png)

**对于（3）**

*   [ ] 整理之前的B300 microbench，包括所有潜在的microbench，重新收集
    
*   [ ] 现在已有的wiki，以及之前AKA已有的大量内源和外挂wiki，重新组织
    
*   [ ] nv的Blackwell和相关的其他资料
    
*   [ ] 整个wiki库的建设
    

**对于（4）**

*   [ ] 基于AKA-light的额外planner层插入？
    
*   [ ] 基于planner设计的，Kernel性能和SOL评估方法？
    

**L20D典型问题：**

FA4 Prefill

FA4 Decode

GDN Prefill

QSA Prefill

其他的MLA/MoE/GEMM？

睿博的题目/算子库：https://code.alibaba-inc.com/tre-infra/atrex-bench/tree/ruibo\_full\_bench

**AKAlight +- planner（AB测试）**

**1）相同效果的token消耗**

**2）相同token的latency表现**

**3）稳定性和可复现能力**

**4）性能上界（平台ceiling）**

 \*可解释性（泛化和通用性） 和 简洁性？

**从零写 还是 从SOTA写，可以两个都进行一些测试？**

（sota schema wiki？）

a）kernel和schema都从零写，只有模板，没有任何相关资料

b）kernel从零写，但是有sota schema

c）kernel和schema都是sota版本开始迭代

同时维护schema和kernel的研究

**CUDA代码库 --> schema代码库**