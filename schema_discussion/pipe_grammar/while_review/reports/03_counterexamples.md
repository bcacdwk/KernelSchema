# 03：公开 GPU 实现中的反馈反例调查

本报告独立阅读 `METHOD.md`、当前 pipe/waiting 规则和下列公开实现后形成判断，未读取 01/02 报告结论。只做源码与契约分析，没有编译、执行 GPU，也没有修改原算子、正式规范、既有 trial 或上级 README。

**找到的最强正向证据是：合理的 GPU 算法确实存在必须保留在多个可引用动作之间的反馈，但本次没有找到同时证明“处于当前 pipe 作用域内，而且 for/if 没有合适改写”的干净实证。** CUDA cooperative CG 和 persistent BFS 的循环跨越真实的全网格同步、共享结果读取或队列复用，不能一概降为等待原语内部的轮询；它们又分别有真实 `10000` 次 cap 和自然的 `|V|` 轮上界。Gamma 拒绝采样没有接受次数 cap，却适合留在单个 compute 内部。Delaunay 动态再生更难给出轮次域，但这份实现的反馈在 host 的多次 kernel 提交之间。

这三个结论不能混为一谈：

| 命题 | 本次证据 |
|---|---|
| 合理算法可以含依结果决定继续的 while | 是。CG、拒绝采样、BFS 等都有明确用途，不因语法而有设计漏洞。 |
| 在当前设计动作粒度，while 有时更自然 | 是，但 CG/BFS 需先把协作 grid 合理登记为一个多 CTA worker，并明确同步与条件就绪。 |
| 只用 for/if 无法合适表达一个已核实的当前 pipe 设计 | 尚未找到足够证据。不能把 gamma 的无 cap 与 CG 的跨动作边界拼成一个不存在的实现。 |

完整文件、上游仓库、固定 commit、路径、下载 URL、字节数、Git blob SHA 和 SHA-256 见 [03_sources.json](../sources/03_sources.json)；逐例结构化记录见 [03_counterexamples.json](../data/03_counterexamples.json)。源文件原有版权说明未删改，另保存了 PyTorch/Galois 顶层 LICENSE。未编译或执行下载的源码，也未运行外部二进制程序。

## 证据与比较的共同边界

当前 `for(t)` 可遍历**实际上游调度提供的**动态工作序列。它不自动表示“在本轮生成下一轮，再决定下一轮是否存在”。本报告把一轮内已经形成的 frontier 元素遍历与产生下一 frontier 的外层反馈分开。`while`/`if` 的条件都使用此前显式动作建立的缓存值；特别是 Galois `nitems()`，设备端是 volatile load，host 端是 `cudaMemcpy`，不能原样放入当前 pipe 条件。

下文 YAML 是用于比较的**动作边界投影**，不是提交到 trial 的完整设计。每个有意义的计算、读取、barrier 和复用边界仍须有 noun 契约、执行模式与完整 waiting。未把数学微步骤、每条 atomic 或 warp 内调度都升级为 role；也未虚构源实现没有的专职 Load/Compute role。

单 GPU cooperative CG 和 persistent BFS/SSSP 可以作如下作用域解释：一次提交只有一个联合 worker，它含启动时已经确定的 `G` 个 CTA 成员，`MAIN` 在每个成员上各有实例，全体共同推进一次求解/图遍历。`G` 的实际值及可同时驻留约束来自 launch/occupancy。`worker.unit_count` 允许多成员，`MAIN[*]` 才能引用这个 worker 的全体成员。**如果坚持每个 CTA 是独立 worker，当前 waiting 不能通过添一个 while 跨 worker 寻址；必须先解决这个作用域问题。** 多 GPU 的情况还需明确设备组成，不可直接假定 `[*]` 自带跨设备含义。

这不是 while 独有的问题：改成 `for + if` 后同样需要全体成员看到一致、已经就绪的条件，同样需要 grid barrier、共享结果可见性及安全复用。while 不赋予驻留保证，也不自动等待。

本报告区分三种比较：有相同返回值；有相同设计动作实例及依赖；有相同所有物理观察行为。纯 guard 的尾部检查不产生设计动作，所以前两者可能保持，但严格的运行时间、kernel 完成时刻和资源占用时长未必保持。当前 pipe 也没有规定必须把每次空 guard 实际发射成 GPU 指令，因此下面的控制开销是直接实现时的代价，不是已测量结论。

## A. 收敛迭代：NVIDIA cooperative conjugate gradient

来源：`NVIDIA/cuda-samples@0ec046d5e99d7214cc8e9fbac2cc1e66f3019878`。

- [单 GPU 完整源码](../sources/03_cuda_samples/cpp/4_CUDA_Libraries/conjugateGradientMultiBlockCG/conjugateGradientMultiBlockCG.cu)：device solver 262–341；cap 277；条件 298；host 初始化 404–449；occupancy/合作启动 465–478；结果判定 483–528。
- [多 GPU 完整源码](../sources/03_cuda_samples/cpp/4_CUDA_Libraries/conjugateGradientMultiDeviceCG/conjugateGradientMultiDeviceCG.cu)：`PeerGroup::sync` 197–257；solver cap 354、反馈 388–445；合作启动 737–759；最终判定 767–811。
- [单 GPU README](../sources/03_cuda_samples/cpp/4_CUDA_Libraries/conjugateGradientMultiBlockCG/README.md)。固定源码已保存，不依赖随 master 改变的链接。

**实际计算与正常/失败语义。** 单 GPU sample 生成对称三对角矩阵：对角约 `[10,11]`，非对角约 `[0,1]`，严格对角占优且正定；`N=1048576`、初始 `x=0`、右端项全 1、`tol=1e-5`。求解结果直接写入全局 `x`，迭代残差平方保存在 `dot_result`。device 停止条件是 `r1 <= tol²` 或已执行 10000 轮；host 用 `sqrt(r1) < tol` 决定 PASS/FAIL 并返回对应退出码。恰等于 tol 时 device 可以退出而 host 仍报告 FAIL，这就是源码的边界，不在本报告中修正。错误硬件能力会 waive；CUDA API 错误由检查宏处理。这是演示程序的契约，不冒称它是对所有 CSR、NaN、非正定矩阵都完备的生产求解 API。

**进展与真实 cap。** 正定条件支持 CG 的数值求解用途；浮点下不把“理论 N 步精确收敛”当退出证明。源码的 `max_iter=10000` 是真实 cap，足以给出有限迭代次数，仍需各次 grid 同步能完成。host 检查 cooperative launch 并用 occupancy 选择可驻留的 grid（389–395、469–476）。这部分不是无界轮询。

**跨动作反馈证据。** 每轮依次产生方向 `p`、SpMV `Ax`、`p·Ax`、步长 `a`、更新 `x/r`、新残差 `r·r`；后者决定是否存在下一轮。下列边界是真实的，而非把一条数学公式拆成许多名字：

| 源码位置 | 可独立引用的边界与作用 |
|---|---|
| 299–307 | 更新本 CTA 负责的方向片段，然后全 grid 同步；SpMV 可读取其他 CTA 的方向分量。 |
| 309–320 | SpMV 写 `Ax`，leader 清共享 dot 累加器，全 grid 同步，再做各 CTA 部分归约/atomicAdd；第二个 grid sync 后每个参与者读取共同分母。 |
| 322–332 | 更新全局 `x/r`；所有旧分母读取结束后才能清零同一累加器，清零又必须完成后才开始下一次归约。 |
| 334–338 | 新残差归约、全 grid 同步、显式读取共同 `r1`；下一轮控制只使用这个已取得的值。 |

把整个 solver 变成 `compute:Solve` 会丢掉 `p/Ax` 的发布边界、dot 累加器的两次复用，以及跨 CTA 的读取完成要求。在以这些边界为设计对象时，这不是正当的删循环方法。相反，`gpuDotProduct` 内部的 warp/block reduction（211–243）可以留在这个动作内部。

一个自然的反馈片段如下。`InitialResidualState` 对应初始三步计算及同步之后的 295 行读取；`leader` 是当前 worker 的 0 号 CTA 内实际执行清零的线程所属动作条件；`residual2`、`tol2` 是已就绪标量。单个动作的线程细节按契约处理。

```yaml
pipe:
  - "load:InitialResidualState"
  - "while(r, residual2 > tol2 and r < 10000)":
      - "compute:Direction(r)"
      - "wait:DirectionReady(r)"
      - "compute:SpMV(r)"
      - "if(leader)":
          - "compute:ClearDot(r,0)"
      - "wait:SpMVReady(r)"
      - "compute:Dot(r,0)"
      - "wait:DotReady(r,0)"
      - "load:DotValue(r,0)"
      - "compute:Alpha(r)"
      - "compute:UpdateXR(r)"
      - "wait:UpdatedVectors(r)"
      - "if(leader)":
          - "compute:ClearDot(r,1)"
      - "wait:ResidualScratchReady(r)"
      - "compute:Dot(r,1)"
      - "wait:DotReady(r,1)"
      - "load:ResidualState(r+1)"
```

`load:ResidualState` 取得新标量并由契约建立下一判断的输入，不在条件中偷偷读全局内存。`compute:Direction(r)` 用 `r=0` 的初始分支和之后的 `r1/r0` 更新；`UpdateXR` 包含源码没有中间交接的两次 vector 更新和保存旧残差。所有 wait 保留源码的 collective 边界。例如：

```yaml
waiting:
  data:
    - "{done MAIN::wait:DotReady(r,0)} ← [{done MAIN[*]::compute:Dot(r,0)}]"
    - "{issue MAIN::load:DotValue(r,0)} ← [{done MAIN::wait:DotReady(r,0)}]"
  reuse:
    - "{issue MAIN::compute:ClearDot(r,1)} ← [{done MAIN[*]::compute:UpdateXR(r)}]"
```

这里只列出说明问题的边；它们不是完整的 ready/reuse 验证。对于带 if 的 leader 动作，实际身份必须是 member 0，并保留其发生条件。

**尽力 for/if 改写。** 完全使用源代码已经存在的 cap：把外层头换成下面两层，其余动作条目原样置于 if 内。

```yaml
pipe:
  - "load:InitialResidualState"
  - "for(r=0:10000)":
      - "if(residual2 > tol2)":
          - "compute:Direction(r)"
          - "wait:DirectionReady(r)"
          - "compute:SpMV(r)"
          - "if(leader)":
              - "compute:ClearDot(r,0)"
          - "wait:SpMVReady(r)"
          - "compute:Dot(r,0)"
          - "wait:DotReady(r,0)"
          - "load:DotValue(r,0)"
          - "compute:Alpha(r)"
          - "compute:UpdateXR(r)"
          - "wait:UpdatedVectors(r)"
          - "if(leader)":
              - "compute:ClearDot(r,1)"
          - "wait:ResidualScratchReady(r)"
          - "compute:Dot(r,1)"
          - "wait:DotReady(r,1)"
          - "load:ResidualState(r+1)"
```

这里**不需要额外 Alive 标志**：第一次 guard 为假后没有动作更新 `residual2`，它保持为假；活跃动作恰构成前缀，轮次身份也不变。初始不需迭代、第一次收敛、恰耗尽 10000 次，都覆盖。若原 design 还导出 `k`，则把它作为既有显式状态仅在活跃 body 内更新，不能用 for 的隐含末值代替。

- 结果/动作：同一初值和每轮数值结果下，活跃动作与依赖可保持；不增加归约、barrier 或 RNG 消耗。
- 新状态：无需必然添加标志；不是“所有消去都需要复杂状态机”。
- 空迭代：直接控制实现可能在收敛后检查到 10000；没有 payload 也不应发射 body 里的 barrier。
- 观察/寿命：host 的 event 时间、kernel 返回时刻可能延后；原共享结果与全 grid worker 生命周期持续到退出，不把尾部空检查说成零物理成本。
- 可读性：只是多一层 if；while 更直接表达数值退出，for+if 仍属合理表达。

多 GPU 版本没有解除这个 cap。它的 `PeerGroup::sync` 内两处 polling while（241、252）属于 acquire/release 同步原语；外层 solver while（388）才是上述反馈。前者不是再添两个 pipe while 的理由。

**判定：强烈反驳“含 while 的求解器就没定义好”；支持在显式 grid 数据交接粒度用 while；不构成 for/if 必须失败的反例。**

## B. 随机接受：PyTorch CUDA Gamma 的 Marsaglia–Tsang 拒绝采样

来源：`pytorch/pytorch@157115e6506269aa62dbb2ef50f3a4a21fa5bd6d`。

- [公共 Gamma 接口](../sources/03_pytorch/torch/distributions/gamma.py)：正参数约束 37–40；`rsample` 到 `_standard_gamma` 88–100。
- [CUDA host 入口](../sources/03_pytorch/aten/src/ATen/native/cuda/Distributions.cpp)：`_s_gamma_cuda` 51–55 创建输出并调用 launcher。
- [CUDA launcher/functor](../sources/03_pytorch/aten/src/ATen/native/cuda/Distributions.cu)：92–122 每个线程建立自己的 Philox 状态和 sampler；170–180 分派。
- [实际抽样实现](../sources/03_pytorch/aten/src/ATen/native/Distributions.h)：114–142，包含算法引用和原 MIT 版权说明 87–111。

**输入输出。** 公共 Gamma 需要正 concentration/rate；本例采用有限正 concentration、rate=1 的普通有限数值范围。每个输出元素调用标准 Gamma sampler，再按 dtype 的最小正正规数下限裁剪（CUDA 116–117）；公共 API 除 rate 后再 clamp（Python 97–99）。实现不是“无限精度连续 Gamma 的逐比特精确实现”。内部 helper 还特别处理 alpha=0 返回 0；这不扩张公共正参数契约。非法参数/NaN 的全行为不是这里的反例依据。

**真实循环。** `Distributions.h:128` 的外层甚至写作 `for(;;)`，不是源码字面 while：先重采样 normal 到 `y>0`（130–133），再读 uniform 并进行 squeeze/log 接受判断（135–140）；拒绝就重试，没有 MaxAttempts。`philox_cuda_state(10)` 是生成器状态预留参数，不是“最多采样十次”的循环 cap。

**进展及失败语义。** 理想独立随机变量模型下，该接受拒绝算法有正的接受概率，通常很快并且几乎必然终止；任意有限尝试次数仍可能全部拒绝。实现没有“预算用完返回失败”的分支。此处没有穷举真实 Philox 的有限 seed/state，也没有证明每个 seed 的最坏尝试次数；不能把概率模型的几乎必然终止改写为对每个 GPU seed 的总正确性证明。用户若把“良好定义”限定为每次调用都必须有已证明最坏时限，需要另立这种实时契约。

**为什么这不是 pipe 的强反例。** normal/uniform 采样、`x/y/v/u`、接受标志全在一个元素的线程局部求值中。期间没有交给别的 role 的候选缓冲，没有共享 ring 的发布/归还，也没有跨 CTA 的 acceptance reduction。实际 `CUDA_tensor_apply2` 只是调用该 functor，结束后写该元素（CUDA 115–121）。自然 pipe 是：

```yaml
pipe:
  - "load:AlphaTile"
  - "compute:GammaTile"
  - "store:GammaTile"
```

`GammaTile` 的动作契约保留每个参与线程自己的随机采样语义。把它内部人为切成下面的片段，能描述算法步骤，但没有充分的设计边界依据：

```yaml
pipe:
  - "compute:InitialAcceptanceState"
  - "while(a, not Accepted)":
      - "compute:PositiveProposal(a)"
      - "compute:Acceptance(a)"
  - "store:AcceptedSample"
```

这里 `PositiveProposal` 还含线程局部 `y>0` 的重试；不能用“我也能把它拆出来”证明两层都该进入 pipe。没有为了删循环而吞并既有发布、归还或握手，因而将这些步骤保留在 compute 是正当的抽象边界。

**尽力 for/if。** 若要求在同一抽样步骤层次消去，可以写 `for(a=0:B)` 下的 `if(not Accepted)`，但 B 在原契约中不存在。预算后仍未接受时：返回失败新增失败行为；返回最后 proposal 破坏接受分布；无限续批又恢复了反馈。为保随机轨迹，已接受后不能继续推进其 RNG；若单纯把所有 lane 固定抽 B 次，随机状态/输出行为也可能改变。公开返回值“分布相同”还弱于按 seed 复现相同执行。

将实际无限 `for(;;)` 叫作当前的 `for` 也无效：它并非已定义有序域。反过来，不在 pipe 中展开这段算法，不需要 B、不新增状态、不产生空迭代、不改变抽样分布或 RNG；SIMT lane 长短不齐属于该 compute 的完成契约。

**判定：证明合理随机算法可以无真实迭代 cap；不证明 pipe 需要 while。这是必须承认的反例弱点。**

## C. 动态 frontier：Galois 的 persistent GPU BFS

来源：`IntelligentSoftwareSystems/Galois@b67f94206a8c47fd414446621f6633a31c49fd98`。

- [BFS 完整源码](../sources/03_galois/lonestar/analytics/gpu/bfs/bfs.cu)：`gg_main_pipe_1_gpu_gb` 305–325；host wrapper 326–347；入口及初始 frontier 349–361。
- [输出与选项](../sources/03_galois/lonestar/analytics/gpu/bfs/support.cu)：默认关闭额外 load balancer，输出逐顶点距离及 unreachable sentinel。
- [真实队列实现](../sources/03_galois/libgpu/include/worklist.h)：223–239 reset/flip counter slots；253–266 `nitems()` 的真实读取；269–287 reserve/store。
- [队列身份交换](../sources/03_galois/libgpu/include/pipe.h)：63 `advance2()` 只交换 in/out，70 `retry2()` 只交换 in/re。

**输入输出与证据限制。** 算法目标是有限 CSR 图从合法 source 的无权最短距离，不可达为 INF。源码写距离的多处 `if (dist==INF) { dist=LEVEL; push; }` 没有 CAS（189–195、227–233、256–262），队列又按 `nnodes` 容量分配（358）；本报告不把这个旧研究实现宣称为对任意大图都已完成数据竞争/重复入队容量证明。作为明确可讨论的合法输入族，可取出度很小的有根有向树，所有计数可表示、source 在范围内、队列足够、硬件可运行它使用的 grid barrier；各节点只有一个父节点，不需依赖竞争写正确性。这已经有不同深度/宽度、动态产生的多个 frontier。没有错误地把潜在一般输入问题归因于 while。

**真实反馈位置。** 默认无额外 load balancer 时，host 344 行启动一个持久 kernel；条件在设备 310 行。每轮：leader 清输入队列备用 counter、按当前 frontier 遍历边并写下一 frontier、翻已消费队列的 counter slot、全 grid `gb.Sync()`、交换 in/out、推进 LEVEL，之后再测试下一 frontier 是否为空。输出队列不是提前给出的静态“顶点顺序”。全 grid 收到同一 frontier，才可共同决定下一轮是否存在。

`gb.Sync()` 既提供下一批队列内容的完成交接，也确保旧输入读取结束后才被复用为输出。源码中的 `nitems()` 设备端实际上从全局 counter 读数，因此当前 pipe 投影显式写 `load:FrontierCount`，不能把这个加载藏在条件中。

```yaml
pipe:
  - "load:FrontierCount(0)"
  - "while(r, Count > 0)":
      - "if(leader)":
          - "compute:ClearNextCounter(r)"
      - "compute:ExpandAndAppend(r)"
      - "compute:FlipConsumedCounter(r)"
      - "wait:FrontierGrid(r)"
      - "compute:PromoteFrontier(r)"
      - "compute:AdvanceLevel(r)"
      - "load:FrontierCount(r+1)"
  - "wait:AllExited"
  - "if(leader)":
      - "store:FinalLevel"
```

`ExpandAndAppend` 对应一轮 `bfs_kernel_dev`，其 done 包括本 CTA 所有队列 payload 的写完；atomic reserve 本身不等于发布完成。这个动作内部的邻接表遍历和 warp 分工可保留为实现细节。`wait:FrontierGrid` 对应源码 316 行而不是凭 waiting 引用制造出来的 barrier；其完成要求所有成员的 append/消费阶段到达。`PromoteFrontier` 对应 `advance2`，不会自行等待，也不会清空 ring。最后两项对应源码 320–324。

例如必须保持：

```yaml
waiting:
  data:
    - "{done MAIN::wait:FrontierGrid(r)} ← [{done MAIN[*]::compute:ExpandAndAppend(r)}]"
    - "{issue MAIN::load:FrontierCount(r+1)} ← [{done MAIN::wait:FrontierGrid(r)}]"
  reuse:
    - "{issue MAIN::compute:ExpandAndAppend(r+1)} ← [{done MAIN[*]::wait:FrontierGrid(r)}]"
```

索引移位只在下一轮实际存在的声明窗口实例化。还须按具体队列 alias 补齐 counter 操作的约束；这不是完整验证过的 Galois 迁移稿。

**上界与 for/if。** 在正确的逐层 BFS 中，处理 source 的轮次也计一轮。每个非空 frontier 是一个距离层，最多 `eccentricity(source)+1 <= |V|` 轮；路径图最后还要处理最远叶节点并得到空 frontier，因此简单写 `|V|-1` 会少一轮。源码没有显式 cap，但 `|V|` 是数学输入规模给出的自然上界，不是硬件有限状态大界。

```yaml
pipe:
  - "load:FrontierCount(0)"
  - "for(r=0:V)":
      - "if(Count > 0)":
          - "if(leader)":
              - "compute:ClearNextCounter(r)"
          - "compute:ExpandAndAppend(r)"
          - "compute:FlipConsumedCounter(r)"
          - "wait:FrontierGrid(r)"
          - "compute:PromoteFrontier(r)"
          - "compute:AdvanceLevel(r)"
          - "load:FrontierCount(r+1)"
  - "wait:AllExited"
  - "if(leader)":
      - "store:FinalLevel"
```

Count 变为 0 后停止更新，可以复用原状态而无需新 Alive 位。活跃轮次的动作、slot parity、LEVEL 更新、barrier 和最终 store 相同；空尾轮**不能**交换队列、推进 LEVEL 或参加 body barrier。树较宽、深度较小时，直接实现会增加约 `|V| - 实际层数` 个 guard。实际 kernel 返回、队列和 CTA 生命周期可能推迟；若只把 pipe 视作动作发生域描述而优化掉纯尾部 guard，则不能从文本断言这个物理成本必然发生。可读性多一层，但仍能清楚表达。

**work_schedule 能否消去外层？**

1. 一轮内对已经形成的 frontier 顶点/边进行 `for(t)`，有真实工作域，合理。
2. 若另有独立 scheduler 真实提供 worktiles、处理结束票，消费者当然可 `for(t)`；必须写明这个 scheduler 的职责、提交和退出协议。
3. 本源码的下一 frontier 在 `ExpandAndAppend(r)` 中产生，并通过同一 kernel 的 `gb.Sync/advance2` 变成下一轮输入。只把它改名成“上游提供的所有 frontier 轮次”没有提供新的调度层，仍然藏着本地反馈。将一轮包装成动作尚可，将整个 BFS 及其 grid barrier/复用封成一个 opaque compute 则丢掉当前选择的设计边界。

**判定：找到必须跨真实动作保留的反馈，且可以在明确联合 worker 作用域后用当前 pipe 表达；while 自然，但 `for(|V|)+if` 是有根据的替代。**

## D. 更强工作队列尝试：Galois delta-style SSSP

同一固定 Galois commit；[完整源码](../sources/03_galois/lonestar/analytics/gpu/sssp/sssp.cu)、[host 参数/输出](../sources/03_galois/lonestar/analytics/gpu/sssp/support.cu)。

实际 device 外层和内层都在 `gg_main_pipe_1_gpu_gb` 344–378；host 400 行启动这个 persistent kernel，421–430 初始化 source 和容量 `2*nedges` 的三个队列。`DELTA` 默认 10000，选项要求大于 0。每次边 relaxation 使用 `atomicMin`（237–248、275–287）：较小候选进入 re 队列，其他更新进入 out 队列；内层 drain re，外层交换 out、去重，再推进 delta。

**输入输出/进展边界。** 预期契约是有限图、非负边权、合法 source、所有距离与 `curdelta` 算术可表示、队列足够、grid barrier 可完成，输出最短距离及 INF。不存在迭代预算耗尽状态。距离严格改善（竞争会产生重复通知）和有限可达路径是分析进展的基础；负环、溢出、队列不足不能当正常输入然后用它们制造“while 必要”。该研究代码的这些前提未由本报告完成普遍 GPU 正确性证明。

**它确有更多独立边界。** 356–359 进行 relaxation、翻 counter、grid 同步和 `retry2`；361–362 内层排空后交换 deferred queue；`remove_dups_dev` 34–63 在 marks 写入与代表节点输出之间有另一个 grid barrier（49）；365–368 再做去重发布、grid 同步、队列交换。这不是单纯 `atomicMin` 内部重试。

以下是其关键控制骨架。`InnerCount/OuterCount` 都由显式 load 建立，初始 inner count 在每个外层 body 入口从已就绪当前队列建立；完整 reset/flip/publish 细节见源码，不把下面骨架冒充完整 pipe。

```yaml
pipe:
  - "load:OuterCount(0)"
  - "while(b, OuterCount > 0)":
      - "load:InitialInnerCount(b)"
      - "while(r, InnerCount > 0)":
          - "compute:RelaxAndAppend(b,r)"
          - "wait:RelaxationGrid(b,r)"
          - "compute:PromoteRetryQueue(b,r)"
          - "load:InnerCount(b,r+1)"
      - "wait:RetryDrain(b)"
      - "compute:PromoteDeferredQueue(b)"
      - "compute:MarkRepresentatives(b)"
      - "wait:RepresentativeGrid(b)"
      - "publish:UniqueDeferred(b)"
      - "wait:DeferredGrid(b)"
      - "compute:PromoteUniqueQueue(b)"
      - "compute:AdvanceDelta(b)"
      - "load:OuterCount(b+1)"
```

**没有把普通 Bellman–Ford 上界硬套到它。** 普通同步 BF 在第 k 次 sweep 后已有至多 k 条边路径的信息，所以有 `|V|-1` 次 relax sweep 的证明；必要的末次空检测另算。这里不是每 sweep 扫同一全部边域，而是内外队列、阈值分类、就地更新、延后发布和去重构成的调度。若为这个具体调度证明 `B_outer`、`B_inner`，两个 while 都能改成对应范围加原 Count guard；必须保留外层的 compaction、计数和 barrier，即使内层已排空。该改写不必增加很多布尔状态，却可能产生两级空尾。

本报告**没有完成对这个精确调度的紧致/合适自然上界证明**，因此不宣称 `for(V)×for(V)` 已等价，也不宣称它一定没有这种上界。按正权路径/传播层分析可能得到这样的界，是进一步工作；“32 位整数只能改善有限次”的巨界不能替代这项证明。把算法换成同步 BF 虽可证明结果相同，也改变 relaxation 次序、队列事件、去重和同步，不能自动当作源设计的控制消去或“修好了坏算子”。

**work_schedule/作用域。** 内层 re 是本轮 relax 产生；外层 out 是内层多轮累积产生；它们不是凭名称就成为外部工作域。声明真实生产者/调度协议可以分担职责，但不能把反馈定义搬到一个不透明的域里。与 BFS 一样，只有在 grid 联合 worker 解释下，`MAIN[*]` 才能表达这些相互交接。

**判定：值得继续研究的复杂反馈候选，支持 while 的可读性；没有用缺失的上界证明或旧代码的未验证条件冒充不可消去反例。**

## E. 动态任务再生尝试：Galois Delaunay mesh refinement

同一固定 Galois commit：

- [README](../sources/03_galois/lonestar/scientific/gpu/delaunayrefinement/README.md) 说明采用 Chew 算法变体。
- [dmr.cu](../sources/03_galois/lonestar/scientific/gpu/delaunayrefinement/dmr.cu)：refine kernel 68–175；host `refine_mesh` 176–259；真正 worklist 反馈在 216–238。
- [main.inc](../sources/03_galois/lonestar/scientific/gpu/delaunayrefinement/main.inc)：29 读 mesh；32 调用 refine；36–38 验证/写出。
- [dmr.h](../sources/03_galois/lonestar/scientific/gpu/delaunayrefinement/dmr.h)：23 `MINANGLE=30`；[dmrggc.inc](../sources/03_galois/lonestar/scientific/gpu/delaunayrefinement/dmrggc.inc)：280–323 添加节点/单元及容量断言，77–81 验证 TODO。
- [meshfiles.h](../sources/03_galois/lonestar/scientific/gpu/delaunayrefinement/meshfiles.h)：90–106 由输入节点数和 maxfactor 分配容量。

**契约与实际行为。** 输入 `.node/.ele/.poly` 表示初始平面三角网格/边界，目标细化质量不好的三角形，输出新网格；其有效使用需几何条件、容量和局部 cavity/boundary 大小足够。host 参数 maxfactor 默认 20，是容量扩张因子，不是循环次数上限。refine 成功时插入节点、新单元，修改邻接关系并删除旧 cavity（137–159）；冲突或原三角形仍需处理则 repush（163–172）。随后 `check_triangles` 对新增单元检测坏角并发布更多工作（230–237）。任务数可以增长，不能只按初始三角形个数遍历一次。

**独立设计边界。** kernel 内明确有三段 collective：两阶段 ownership 竞争之间/之后的 `gb.Sync`（118、121），以及修改 mesh 完成后、repush 前的 `gb.Sync`（168）。这些边界不可用泛称“本地重试”抹掉。不过该 kernel 自身按本次输入 worklist 的有限位置 `for`（87）执行；局部 cavity 的 do/while（102–107）不是整个动态再生反馈。

**关键反例失败点：作用域。** 真正的“细化 → 更新全局尺寸 → 检查新坏三角形 → 决定是否再细化”是 host 216–238 的 while，包含多次 kernel launch 和 device-to-host 计数读取。当前 `flow.launch` 记录计算提交身份，role pipe 不能直接重发一组 phase。因而这里没有可直接给出的合法单 phase while pipe；把下面 host 顺序画成 pipe 会偷换层次：

```text
read current work count
while cached count is nonzero:
    launch refine; wait/read new node and element counts
    launch check_triangles for new elements
    rotate worklists; read new bad count and work count
```

如果未来设计一个融合的 persistent refinement kernel，外层 while 可能跨 ownership、commit、queue publication 等动作，但这需要新的实际实现与进展/资源证明；本报告没有这样的源码实证。

**上界、失败和 for/if。** 不能把 `maxfactor` 当已有 MaxIter。固定容量、每次成功至少插一节点及全局进展若一并证明，可能导出资源相关界，但冲突重试、几何终止与存储失败语义都须一并核对；只说内存有限不够。该代码有 `CAVLEN=256`、`BCLEN=1024` 和节点/单元容量断言，容量不足不是正常收敛，断言编译掉时还不能承诺安全失败。`verify_mesh` 留有 Delaunay property TODO（dmrggc 77–81）；没有对所有输入的终止/质量证书。本报告因此不宣称已核实“无设计漏洞的全输入 DMR”或“没有任何有限界”。

最佳 for/if 尝试是：在**host 层**给出经证明的 refine 轮数 B，用 `if(cached_count>0)` 保留原 launch、尺寸同步、工作队列交换和日志；现在缺少这样的 B。凭空加入预算以及失败返回会改变语义，尾部仍 launch 空 kernel 又会改变动作和观察行为。移到真实 work scheduler 可以管理坏三角形的领取/重投，但仍需负责 mesh commit 的互斥、发布及终止检测，不是简单改域名。

**判定：真实的动态再生算法与跨 kernel 反馈；不足以决定当前单 role pipe 的 while 去留。**

## F. 反向检查：带随机优先级的 maximal independent set

[Galois mis.cu](../sources/03_galois/lonestar/analytics/gpu/independentset/mis.cu) 在同一固定 commit 中提供另一个容易被误判的候选。host 185 行只生成一次随机 priority；`mark_nodes` 89–95 用 vertex ID 打破同 priority 的平手，随后 105–150 删除选中点及其邻居。while 在 host 159–168，重复启动 mark/drop 两个 kernel。

对于这个算法所要求的有限无向图和有效优先级比较，剩余集合的全局最大 `(priority,id)` 至少选中一个点，所以非空集合每轮减少，有 `|V|` 的自然界；“使用随机数”不代表每轮可能永远零进展。该旧实现还有 `unsigned prio` 与局部 `int max_prio` 混用（83）等应另行核对的实现细节，本报告不据此建立全输入正确性证明。

输出是 maximal independent set 的节点标记，已有 `STEPS` 是实际活跃轮数；`for(r=0:V)+if(count>0)` 的 host 改写须只在活跃 body 更新它，保留 mark/drop 交接及同一 priority。增加一个空尾范围，通常无需额外 Alive 状态。这里同样没有合法的、直接对应现实现的单 phase while pipe，因为反馈位于多次 host 提交。换用每轮重新抽签且允许零进展的另一种 MIS，可以得到不同性质，但不再是这份源码的证据。

**判定：主动剔除的“随机所以无 cap”假强例。**

## 对“while 意味着算子没定义好”的分层判断

1. **算子结果层：不成立。** CG 的近似结果及 cap 后失败、Gamma 的概率输出、BFS 的固定点，都能明确说明算什么以及何时正常退出。终止证明可以是数值预算、排名函数、工作集合耗尽或几乎必然接受，不需要都长成固定次数的 for。
2. **总正确性/实时契约层：可能提示必须补充条件，但不能靠语法定罪。** 非正定矩阵、负环、溢出、容量不足、无公平性或随机接受的最坏时限都需在各自契约中处理。加一个任意 cap 后返回错误，是新设计选择，不自动证明原来的几乎必然终止算法没定义好。
3. **pipe 抽象层：这种怀疑经常很有用。** 大量 while 只是线程内算法、等待原语、有限工作领取或者 host orchestration；Gamma/DMR/MIS 就不能被直接用来证明当前 pipe 必须有 while。
4. **显式跨动作反馈层：不能一概删除事实。** CG 的新残差和 BFS 的新 frontier 决定后续动作是否存在，且伴随真实 grid 可见性/复用边界。可用 for+if 表达的，是同一个反馈条件，而不是“反馈其实不存在”。

本次最明确的比较结论是：**有限范围加纯 if 的消去通常比想象中简单，CG/BFS 不必新增 Alive 标志；而让动态 for 的域定义吞掉本地产生的下一轮，或把所有同步/发布封进一个 compute，并不是同样的合法消去。** 当前证据支持“while 可让某些合理设计更自然”，尚不足以声称“删除 while 必然无法表达某个已核实的当前 pipe 算子”。因此无论最终保留或删除，都应基于当前作用域与表达成本，而不应用“出现 while 就是坏设计”作为理由。

## 主要文件指纹与复核方式

全部指纹在 [03_sources.json](../sources/03_sources.json)，这里列最直接的实现文件：

| 文件 | SHA-256 |
|---|---|
| NVIDIA `conjugateGradientMultiBlockCG.cu` | `6dc0afd278dee9c445a97a52fc651ecb53c6ba91a26c7863579a0bd45298fe4e` |
| NVIDIA `conjugateGradientMultiDeviceCG.cu` | `a2c6ce43ab62c9a9b92b506cfe4035fa82dec30264df6c85e3a2dd655bc134f1` |
| PyTorch `aten/src/ATen/native/Distributions.h` | `7134912f1397a90e451787b3c3bdfff126737b059b8c9a60927ef1b566ac6770` |
| PyTorch `aten/src/ATen/native/cuda/Distributions.cu` | `cef5ee8525ae2fd2e6a68eca69eeaf1b0a479cc8b7668c4d81f8cd92acf18719` |
| Galois `bfs.cu` | `495291ea15108c5c8cbb493c77b2b81a035e172ac9c818c4e78ee04c18eb0630` |
| Galois `sssp.cu` | `966ad1fa15ab6bf14e3273e4e4dc33bdf843ec03067873122962446d64ad6fe1` |
| Galois `dmr.cu` | `987c23be57cce9cf955fd7329b27576a1273a4df0cc0a804a851542884ea416d` |

复核可按 manifest 对每个保存文件重新计算 SHA-256，再按上文行号审阅调用链和退出条件。本报告只证明这些静态证据与相应分析，不声称已验证任意 GPU 并发轨迹、数值正确性或普遍终止性。
