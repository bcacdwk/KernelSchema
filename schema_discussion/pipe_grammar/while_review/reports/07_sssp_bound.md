# Galois SSSP 专项复核：两个反馈循环都有输入规模上界

在下文明确的非负权、算术、完整波次与可见性契约下，这份 persistent delta-style SSSP 有

\[
B_{\mathrm{outer}}=|V|,\qquad
B_{\mathrm{inner}}(b)=|V|.
\]

两者都包括最后一次处理已有通知、发现不再产生后继的波次或外层轮次。**不是 `V−1`，也不是把源码换成同步 Bellman–Ford。** 两个界只需已有的 `graph.nnodes`；不用先求最短路，不用扫描 `Wmax`，不用把反馈藏到新调度器或大动作中。`V` 对两个循环各自都是可达到的上界。

本例因此支持“当前 pipe 核心可以只保留 `for/if`”的表达能力判断。源码 `while` 仍然直接表达运行时排空，阅读与实现都很自然；这与“没有它便不能合理表达该设计”是不同判断。此处解决 [03 报告 D 节](03_counterexamples.md) 留下的具体上界问题，不据此推断所有合理算子都可删除 while。

这不是完整 GPU 算子规范化或 CUDA 正确性认证；没有编译、运行 GPU，也没有改原源码或正式规则。一般上界来自以下证明，有限 CPU 枚举只是查错证据。

## 1. 来源与真正影响证明的源码语义

固定来源为 `IntelligentSoftwareSystems/Galois@b67f94206a8c47fd414446621f6633a31c49fd98`。使用的四个原文件与 [03_sources.json](../sources/03_sources.json) 中指纹一致。

| 源文件 | 本次核对的位置和用途 |
| --- | --- |
| [sssp.cu](../sources/03_galois/lonestar/analytics/gpu/sssp/sssp.cu) | 15–31：有符号整数、`INF=INT_MAX`、初始距离；185–199：固定当前 frontier 的完整索引遍历；237–248、275–287：检查、atomicMin、分类三次 source 读取；344–378：两个 device while 与退出；423–429：初始 `i=curdelta=0`、source 队列、`2*nedges` 容量 |
| [support.cu](../sources/03_galois/lonestar/analytics/gpu/sssp/support.cu) | 10、20–22：默认 `DELTA=10000`，参数分支要求 `DELTA>0` |
| [pipe.h](../sources/03_galois/libgpu/include/pipe.h) | 36–43：三个物理队列；63、70：`advance2` 只交换 in/out，`retry2` 只交换 in/re |
| [worklist.h](../sources/03_galois/libgpu/include/worklist.h) | 222–239：两个 counter slot；253–266：device `nitems()` 为 volatile load，host 为 cudaMemcpy；269–287、330–349、392–395：先原子保留位置、后写 payload；525–533：`pop_id` 按 ID 读取，不逐项递减 count |

这里将外层一次 body 称为 **epoch**，内层一次 `sssp_kernel_dev` 加完成屏障、队列翻转称为 **wave**。这只是讨论用语，不增加 schema 节点。

一个 wave 处理的是入口已经形成的整个 `in` frontier，多重通知也各有输入位置。它不在这个 wave 内把新 `re` 继续弹出。新通知在 `gb.Sync()` 后，经 `retry2` 成为下一 wave 的输入；`out` 在当前 epoch 的全部 wave 之间持续累积。队列状态为：

| 位置 | in | re | out |
| --- | --- | --- | --- |
| epoch 入口 | 上次 out 去重结果，首次为 source | 空 | 空 |
| 每个 wave 完成并 retry2 后 | 本 wave 新 re 通知，可重复 | 刚消费的输入队列，当前 counter 已为 0 | 本 epoch 累积 deferred 通知，可重复 |
| 内层退出时 | 空 | 空 | 本 epoch 累积 deferred 通知 |
| 外层去重并两次 advance2 后 | deferred 的去重结果 | 空 | 空 |

每个物理 worklist 只有一份 payload，两个 slot 是计数器。`reset_next_slot` 清另一个计数器，`swap_slots` 改本地选中的计数器；跨 grid 屏障之后才会复用旧 payload。不能把它误读成两个独立 payload 缓冲，或把 slot 翻转当作可删除的空操作。

**外层并不等待距离进入当前 bucket。** `sssp_kernel_dev` 不检查 `d[node] <= curdelta` 才扫描节点。上个 epoch 的全部 out 在下个 epoch 就被扫描，即使其距离仍远大于 `curdelta`。这是能得到不依赖权值大小的 `V` 界的关键。

## 2. 证明使用的执行契约

1. 图、边权和邻接关系固定，`V>=1`，source 与队列中的节点合法。边权是非负整数。无额外生产者在屏障之外注入工作。
2. 距离相关的**所有实际执行的加法**、`i++` 和 `curdelta += DELTA` 均不溢出。仅“最终最短距离可表示”不够：较差 tentative 值加边权也可能溢出。若要求正确区分所有有限答案，有限最短距离须严格小于源码 INF。
3. 一个 wave 完整处理当前 frontier 上源码所选择的固定边域，每条已分配的边任务最终完成。工作列表容量足够，push 不丢项、不越界，count 与 payload 在屏障后均已发布；去重为每个出现的顶点留下一个代表。
4. `atomicMin` 可线性化，距离只减不增。普通距离读取遵守其单调变化；同一边任务在检查、atomic 参数、分类中的后续 source 读取不会读到比先前更大的过期值。进入下一 wave 后，读取不退回前一完成屏障以前的旧状态。屏障可完成，且所有成员看到一致的就绪队列 count。

第 4 项是常用算法执行模型的明确化，不是已经为这份旧 CUDA 源完成的内存模型证明。源码混用普通 load 与 atomicMin，marks 也有竞争写入；冻结材料没有使任意 C++/CUDA 数据竞争行为自动成为此证明的模型。不能一边取消这些可见性保证，一边把剩下的失效执行称作有效 SSSP 的正常反例。

`sssp.cu:192` 还带有 `outdegree < DEGREE_LIMIT` 的选择。本文的上界可对**实际完整扫描的固定边域** `E_scan` 证明，即使它是原图边域的子集；所有写距离的边均在这个域内。要再声称结果是原图 SSSP，须另外保证需要的边都得到处理，包括生成代码的 degree/load-balancing 配置。上界不靠忽略这个选择，本文也不以终止证明替代这项完整算子核验。

以下 `G` 可以理解为该固定扫描图。未达到 INF 的路径按普通整数距离处理；INF 在数学记号里视作“尚无可表示路径”，不在证明中执行源码没有执行的 `INF+w` 加法。

## 3. 三个不变量：允许重复通知，允许 re 中暂时有高值

### 3.1 每个实际写入的有限距离有简单路径见证

初始 source 的见证是长度 0 的路径。一次真正减小 `d[v]` 的 atomicMin 使用某次读取的 `d[u]` 加 `w(u,v)`。归纳假设该 source 值来自简单路径 `P`。

若 `v` 已在 `P` 中，见证中到 `v` 的前缀距离曾经写入 `d[v]`，此后它只能减小。非负后缀意味着本次候选不小于该前缀，因此不可能再严格改善 `d[v]`。所以真正改善时 `v` 不在 `P` 中，`P+(u,v)` 仍简单。

这证明的是**实际保存的值**，不是“每个入队通知都是一次 atomic 成功”。若 `Wmax` 是边权上界，则有限值满足

\[
0\le d[v]\le H=\min((V-1)W_{\max},\mathrm{INF}-1).
\]

计算这个表达式须用可容纳乘积的数学整数或加宽整数，不能在证明之外新增一次有符号溢出。

### 3.2 每次扫描保证相对于 wave 入口距离的边松弛不等式

记入口距离为 `d0`，完成距离为 `d1`。若本 wave 扫描了有限 source `u` 的边 `(u,v,w)`，则

\[
d_1[v]\le d_0[u]+w
\]

对可表示候选成立。检查为假时，读到的目标已经不大于所检查候选；检查为真时，atomicMin 后的目标不大于 atomic 使用的候选。之后目标只会减小，而 source 在 wave 内的读取不大于其入口值。

这个论证不要求 source 在整轮内保持不变，不要求所有边在同一时刻读距离，也不改变现有 frontier 调度。

### 3.3 通知性质比“atomic 成功才入队”弱，但足够

设检查使用 `c_check`，atomic 参数使用 `c_atomic`，分类使用 `c_class`。契约给出

\[
c_{\rm class}\le c_{\rm atomic}\le c_{\rm check}.
\]

检查通过即有 `c_check < d_read[v] <= d_entry[v]`。所以即使本次 atomicMin 没有成功，完成时的 `d[v]` 仍严格小于本 wave 的入口值：一定有本任务或同 wave 竞争任务完成了改善。因此：

- 每次真正改善都会完成一次 re/out 通知；不是每份通知都对应独立改善。
- 某个 wave 通知的目标，确实在这个 wave 中变小过；同目标可以收到多份通知。
- 如果一次 atomic 把目标变成 `<=delta`，其后分类也 `<=delta`，所以它必进入 re。特别是一个顶点在本 wave 达到的最终低值，不会只留在 out。
- 反向命题不成立：分类可以在 atomic 之后看到更小 source，把一个仍高于 delta 的目标送入 re。

最后一点不能略过。以下四点图满足非负权和小整数算术，`DELTA=1`，容量 `2E=8` 足够：

```text
s=0 --2--> u=1 --0--> v=3
  \--1--> x=2 --0--> u=1
```

epoch 0 后，`d=(0,2,1,INF)`，下个 epoch 的 `delta=1`、输入为 `{u,x}`。允许的源级交错是：

| 次序 | 边任务 | 动作和结果 |
| --- | --- | --- |
| 1 | u→v | 检查 INF>2，atomic 把 v 写为 2 |
| 2 | x→u | 检查 2>1，atomic 把 u 写为 1，分类进入 re |
| 3 | u→v | 重新读取 u=1，分类进入 re，但此时 v 仍是 2>delta |

这是源级三次读取模型允许的轨迹，不声称已经检查某次编译的 SASS 一定重载这些值。若编译实现重用较早读取，证明仍覆盖其不增的特例。不能用“re 只含低值点”证明当前源码。

## 4. 外层界：每个 epoch 至少推进一条路径边

称 source `u` 为 dirty，若存在一条可表示的出边候选还未反映到目标，即 `d[v] > d[u]+w`。

**epoch 入口不变量：所有 dirty source 都在当前 frontier 中。** 初始化时唯一可能 dirty 的有限 source 是 s，正好在队列中。一个 wave 完成后，一个没有变小的已扫描 source 由 3.2 变 clean；其他目标的减小不会让它重新 dirty。变小的 source 则由 3.3 留下 re 或 out 通知。因此 wave 边界有

\[
\mathrm{Dirty}(d)\subseteq \mathrm{in}\cup\mathrm{out}.
\]

内层结束时 in/re 为空，dirty 都在 out；去重保留每个 source 一个代表，所以下个 epoch 入口仍成立。此处用的正是原始 barrier、队列翻转和去重，未添加全图扫描。

令 `T` 为数学上的一次 Bellman 松弛映射，`T(d)[v]` 是自身旧值与所有可表示前驱候选的最小值。它仅用于比较，不是要求执行的算法。对于 epoch 入口 `d_b`：

- 入口 frontier 中的每个 source 在首个 wave 被扫描，由 3.2 可用入口 source 值界定 epoch 结束目标值。
- 不在入口 frontier 中的 source 已 clean，已有距离就满足相同不等式；目标之后变小只会使它更容易成立。

所以原算法实际 epoch 变换满足

\[
d_{b+1}\le T(d_b).
\]

初值只有 source 为 0，归纳后，完成 `k` 个 epoch 至少包含所有不超过 `k` 条边的可表示路径所能给出的距离。非负权最短路可取简单路径，`V−1` 条边足够；简单路径见证同时保证不会低于真实最短距离。因此至多 `V−1` 个 epoch 后距离已经稳定。

但此时 out 可能仍含已过时的通知，不能直接退出。若下个 epoch 仍存在，它会处理这些通知；稳定距离下，所有检查均失败，不再产生 re/out。仍照常执行去重、slot 翻转、屏障、`i++`、`curdelta += DELTA`，然后下一次 outer count 为 0。因此

\[
N_{\rm outer}\le V.
\]

退出更早的情况，空 frontier 与 clean 不变量已经构成固定点。证明中的 Bellman 映射不产生任何额外源码动作。

## 5. 内层界：证明低值的传播，而不是假定所有 retry 都低

固定一个 epoch 的 delta，定义截断距离

\[
D_r[v]=\begin{cases}
d_r[v],&d_r[v]<\mathrm{INF}\text{ 且 }d_r[v]\le\delta,\\
\infty,&\text{其他情况}.
\end{cases}
\]

令 `T_delta` 只考虑候选 `D[u]+w <= delta` 的 Bellman 映射。活跃 epoch 在无溢出且 `DELTA>0` 的契约下，delta 本身低于 INF。

**wave 入口不变量：所有尚有阈值内松弛机会的低值 source 都在当前 frontier。** epoch 入口由上一节的不变量得到。之后，如果一个 source 变小到低值，3.3 保证它在下个 re frontier；若未变小，则本 wave 扫描它会消除其低值松弛机会，不扫描且原已 clean 的点仍 clean。因此即使 out 存在、即使 re 中还含高值点，这个不变量也保持。

与上一节相同的逐边论证给出

\[
D_{r+1}\le T_\delta(D_r).
\]

现在把 epoch 入口的每个有限 `D_0[u]` 视为一个已有初值。由这些初值出发得到的阈值内最小距离记作 `L_delta`。非负权保证高于阈值的初值无法产生低值；后续每个低值仍来自某个初始低值加一条路径，所以始终 `D_r >= L_delta`。最优路径去除非负环后至多 `V−1` 条边；其每个前缀也不超过最终阈值。因此

\[
T_\delta^{V-1}(D_0)=L_\delta,\qquad
D_{V-1}=L_\delta.
\]

若内层仍未结束，第 `V` 个 wave 中不能再有任何低值改变。这个 wave 也就不能产生 re：

1. 若某条入 re 的边在 atomic 时的候选已 `<=delta`，由 3.3 目标必在该 wave 内发生低值改善，矛盾；即使该 atomic 自己失败，竞争者也必须完成这种改善。
2. 若 atomic 候选 `>delta`，但之后分类 `<=delta`，非负边权意味着 source 在该 wave 中降到或进一步改善了低值，仍矛盾。

所以第 `V` wave 的 re 必为空，随后执行原 inner 的失败 count 读取和 outer 收尾。该 wave 仍可把高值工作送入 out，不能把 outer 收尾删除。于是

\[
N_{\rm inner}(b)\le V.
\]

这不是按每个原子更新数计数。重复通知可以增加一个 wave 内的工作量与队列容量需求；它们无法在低值已稳定后再复制出无限 retry 波次。`V−1` 后的**一个排空 wave**已经包含需要处理的重复通知，没有再加一个按重复数量增长的尾部。

两节构成同一个归纳，而不预设 inner 会终止：第一个 epoch 的入口 clean 不变量由初始化成立；本节保证该 epoch 的 inner 在 V 波内完成；上一节的队列不变量保证下一 epoch 的入口。于是既有每个 epoch 的完成，也有外层的 V 界。

## 6. 可选距离阈值界、紧致性和真正的反例

3.1 的 `H` 还能给出一个较弱但有算法意义的阈值界。不能简单说“写入值<=H，因此每次分类值<=H”，因为 atomic 可能失败、三次读取又不同。应分情况：

- 若目标在 wave 入口已经有限，检查通过时 `c_check < d_read[v] <= d_entry[v] <= H`。
- 若目标在入口仍 INF，则输入 source 的有限简单路径见证不能经过它。该 source 路径至多 `V−2` 条边，故入口 source 加当前边至多 `(V−1)Wmax`。读取单调性把这个界传到检查、atomic、分类；检查又保证候选 `<INF`。

因此每个分类候选确实 `<=H`。epoch 入口的 out 为空；一旦 `b*DELTA >= H`，该 epoch 不会生成 out，内层完成后外层退出。可取

\[
N_{\rm outer}\le
\min\left(V,\ 1+\left\lceil\frac{H}{\mathrm{DELTA}}\right\rceil\right).
\]

这是路径长度、输入权值和实际阈值增量推出的界，与“整个机器只有有限多个状态”不同。仅用 INF 也能给出范围界，但小 DELTA 时仍非常大。若 Wmax 未作为输入元数据给出，这份调用路径并没有现成计算；额外扫描边权会增加工作和事件。**采用 `V` 界完全避开该额外扫描。** 阈值界可作已知权值范围下的收紧，不能强迫算法真的等待阈值追上距离。

| 图/调度 | 实际行为 | 排除的误判 |
| --- | --- | --- |
| 两点 `s --0--> t`，初始 delta=0 | 一个 epoch 内先扫描 s，再扫描 t 才排空，共 2 个 inner wave | `B_inner=V−1` 少了末次处理；一般 V 点零权链达到 V |
| 两点 `s --1--> t`，DELTA=1 | epoch 0 将 t 放 out；epoch 1 扫描 t 后退出，共 2 个 epoch | `B_outer=V−1` 少了 deferred 排空；V 点单位权链达到 V |
| 两点 `s --10^9--> t`，DELTA=1 | 仍只需 2 个 epoch，最终 curdelta=2 | 此实现不会等待十亿个 bucket |
| 两条平行零权边都先读到目标 INF，再依次 atomicMin | 第二次 atomic 不改变距离，但两份 t 通知都进 re；下一 wave 一起处理 | “每个入队都对应一次严格改善”错误；它没有突破 V 波次 |
| 第 3.3 节四点图 | 允许 re 中暂时含 `d[v]>delta`；另有有效模型调度得到各 epoch wave 数 `[1,3,1]` | 不能假设 re 全低；也不能把两个 V 界误合并为全程仅 V 个 wave |

这些小图的数值和所列队列长度均在源码 `2E` 容量内；不靠负环、溢出、外部补充任务或不公平调度制造反例。这里“紧”是说两个循环各自的 V 可达到，不声称 `V×V` 的总上界同时紧致。

## 7. 保留事件与退出状态的 for/if 表达

`V` 已是图输入规模。以下是控制层比较稿，动作仍需完整定义它们的作用域、执行模式、复用依赖和 grid 联合 worker 成员；不冒充已经通过 schema 验证的完整 GPU 算子。

`OuterCount`、`InnerCount` 是由显式 load 更新的**当前缓存值**；为 0 后，空尾 guard 继续读这个已有值，不重复读取 worklist，也不查询不存在的 `Count(b,r)` 实例。

```yaml
pipe:
  - "load:InitialCounters"
  - "load:OuterCount(0)"
  - "for(b=0:V)":
      - "if(OuterCount > 0)":
          - "load:InitialInnerCount(b)"
          - "for(r=0:V)":
              - "if(InnerCount > 0)":
                  - "compute:ResetInputNextCounter(b,r)"
                  - "compute:RelaxAndAppend(b,r)"
                  - "compute:FlipInputCounter(b,r)"
                  - "wait:RelaxationGrid(b,r)"
                  - "compute:PromoteRetryQueue(b,r)"
                  - "load:InnerCount(b,r+1)"
          - "wait:RetryDrain(b)"
          - "compute:PromoteDeferredQueue(b)"
          - "compute:ResetDeferredNextCounter(b)"
          - "compute:MarkRepresentatives(b)"
          - "wait:RepresentativeGrid(b)"
          - "publish:UniqueDeferred(b)"
          - "compute:FlipDeferredCounter(b)"
          - "wait:DeferredGrid(b)"
          - "compute:PromoteUniqueQueue(b)"
          - "compute:AdvanceDeltaAndIteration(b)"
          - "load:OuterCount(b+1)"
  - "wait:FinalGrid"
  - "store:ClosedCounters"
```

其中 reset 和最终 store 保留原 tid 0 执行者契约；其他成员的集体参与由原 grid/barrier 契约说明。`MarkRepresentatives → RepresentativeGrid → UniqueDeferred` 对应 `remove_dups_dev:41–61` 的两个扫描及中间屏障。最后一轮即使 deferred 空，也保留去重函数的屏障、counter 翻转及外层的 i/delta 更新。

一般事件投影依据如下：设原执行有 K 个 epoch，第 b 个有 L_b 个 wave。证明给出 `K<=V`、`L_b<=V`。两个受缓存 count 控制的 guard 为真时，各 body 与原序列完全相同，且使用同一个原始边任务调度、atomic 结果、去重代表选择。因此归纳保持相同的距离、三份 payload、两个 counter slot、in/out/re 别名、i 与 curdelta。真实 body 的事件索引仍恰为 `b=0..K−1`、`r=0..L_b−1`。

每个 body 后的显式 count load 包括原 while 的末次失败读取。即使某个真实长度恰等于 V，也已经取得那个 0，不会漏掉退出读取。剩余范围仅求值已缓存的假条件，不发起原子操作、load、等待、发布或队列翻转。投影删除的是无事件的范围推进和重复布尔 guard，保留全部原设计动作及其交接边界。

没有新增“预算耗尽=正常退出”的状态，也没有把 `i` 设置为 V。达到上界后 count 必为 0 是契约推论；若验证不能确认此前提，应报告该实现/输入不在已证明范围，不能静默按 cap 返回。

这个表达增加了界的论证和两层 guard。若机械地让源码跑完整个 `V×V` 范围，可能产生大量无效控制开销；它不是必要的实现策略。源码可保持当前 while，或由编译器对缓存 count 为 0 的后缀提前退出。此处的 for 是有输入与算法根据的有限发生域，不要求实现真的执行空尾。while 在阅读上更贴近“排空”，但在允许上述实现关系的设计语言中，它不是该例的不可替代核心控制。

## 8. CPU 查错与限制

[模型脚本](../scripts/07_sssp_bound_model.py) 将每个边任务分成 guard、atomicMin、分类三个阶段，允许阶段间交错与 source 重读，保留通知多重性，只有外层去重。完全枚举时将队列按多重集合归一化；随机协议比较保留实际模型入队次序和被新写入覆盖之外的旧 payload。guard 内两个操作数在模型中一起读取，atomic 与分类后入队也各作为一个阶段；这不是全部 GPU 指令级交错。证明本身允许源代码两个操作数不同时间读取，只使用上文的不增性与 wave 入口界。

[结构化结果](../data/07_sssp_bound.json) 记录：

- 三点、无自环/平行边的 728 个非空有向图，边权 0/1，DELTA 1/2，共 1,456 个图与参数组合；完全枚举模型的边任务阶段交错，累计 3,616 个宏状态、22,520 个微状态。观察到最大 inner=3、outer=3，全部终态距离正确。
- 单独枚举四点 source 重读图与两点平行边图的模型交错；显式保存“atomic 失败仍入队”和“re 中仍有高值”两条见证。
- 固定种子 20261111 的 3,000 个随机图，2–12 点，边权 0–9，DELTA 1–10。对相同任务调度比较原 while 与受 V 限制的 for/if；全部模型动作序列、count 读取、三物理队列与双 counter slot、别名、旧 payload、i/curdelta、最终距离相同。观察到最大 outer=8、最大 inner=6。
- 对这些运行额外检查 `H` 距离界和阈值 outer 界。它们是数学证明的补充，枚举数量不被当作一般证明。

复现模型检查：

```bash
python schema_discussion/pipe_grammar/while_review/scripts/07_sssp_bound_model.py --exact --random-cases 3000
```

输出对应 JSON 的 `cpu_model` 部分。算法模型使用足够的 CPU 队列存储，不证明 Galois 固定 `2E` 容量对任意图和并发重复数均足够。协议模型的一个 barrier 事件只表示源码顺序边界，不模拟每个 CTA 的到达、内存栅栏实现或驻留。模型未编码实际 lane/warp 的全部调度约束，也未覆盖所有 GPU 读取与栅栏行为，不能据此宣称是全部 CUDA 执行的过近似。上述限制与未运行 GPU、未完整规范化跨角色 waiting 的限制保持。

**本例最终判断：自然上界问题已在明确执行契约下解决；不能再把它列为“仅因未找到上界而可能需要 while”的强反例。保留 while 可以基于书写清晰度，删除当前核心 while 也不会因这个例子失去经证明的控制表达能力。**
