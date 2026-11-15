# M01：fa4_decode 文字设计草稿

本次从给定 Contract 出发，未读该算子的旧 Design、dispatch 或历史研究。下面是实际选出的初稿，不是完整 schema，也不是可直接交给 Coder 的已验证方案。只做文本推导；没有性能、编译、数值验收或 schema 工具结果。

## 计算路线

先选单请求、单个 query-tail token、合并 16 heads 的在线 softmax 作为主要阅读路线。KV 沿逻辑 token 顺序按一页 64 token 推进；不物化全长 score 或 probability；同一 KV 块在 16 heads 间共享。q_len_per_req=4 时让四个 tail token 成为四份独立工作，先用这部分并行度，代价是 KV 可能重复读四遍。无分段主线不使用 workspace，能表达短、长 KV 的全部语义；它不保证少 batch、长 KV 的性能。

直接求值参照是：每个 (b,t,h) 计算全可见 score，物化 score 和 probability，再与 V 相乘。若两个 FP32 中间量各写、读一次，中间流量为每个可见 key、每个 query/head 16 字节。在线求值去掉这笔全长物化，增加每页对行统计量及输出累加器的重缩放，并留下跨页依赖链。

暂选 FP32 分数、指数、统计量和 PV 累加，不把 probability 窄化为 BF16/FP8。实现方向先是普通协作 FP32 运算；不因 Contract 写有 TMEM 而直接断言特定 Tensor Core 形状可用。Tensor Core 路线暂未选定，见候选判断。

以下定义对应拟写的 `math_reshape.rewrite`，但此处用公式草稿表达，不冒充已经填全字段的区域：

```text
Independent b,t, with 0 <= b < BatchCount and 0 <= t < q_len_per_req.
Nbt = seq_lens[b] - q_len_per_req + t + 1.
Bk is bound to design.work_schedule::Bk.
Nb = ceil_div(Nbt,Bk).
K,V use the unchanged Contract page lookup.
For 0 <= i < Nb, J(i) = {k in integers | i*Bk <= k < min((i+1)*Bk,Nbt)}.
For each h, initialize m_h(-1)=-inf, z_h(-1)=0, u_h(-1)[d]=0.

For each page iteration i:
  S_i[h,k] = bmm1_scale * dot32(query[b*q_len_per_req+t,h,:], K[b,k,:]), k in J(i).
  c_i[h] = max_{k in J(i)} S_i[h,k].
  m_h(i) = max(m_h(i-1),c_i[h]).
  alpha_i[h] = 0 if i=0, otherwise exp(m_h(i-1)-m_h(i)).
  E_i[h,k] = exp(S_i[h,k]-m_h(i)), k in J(i).
  z_h(i) = alpha_i[h]*z_h(i-1) + sum32_{k in J(i)} E_i[h,k].
  u_h(i)[d] = alpha_i[h]*u_h(i-1)[d] + dot32(E_i[h,J(i)],V[b,J(i),d]).

output[b*q_len_per_req+t,h,d] = bf16(bmm2_scale*u_h(Nb-1)[d]/z_h(Nb-1)).
out_after=output; write the destination out and return that same tensor object.
```

S/E 每轮用完，没有全局物化；m/z/u 在 (b,t) 内跨 i 携带，最终输出后死亡。若正式登记：m、z 是 `[head: HeadCount]` FP32，u 是 `[head: HeadCount, d: D]` FP32，retention 为 `carry_forward (b,t); i=-1..Nb-1`。S/E 是下游计算接口，shape 为 `[head: HeadCount, kv: Bk]`、dtype 为 FP32；不足 Bk 的尾部是无效区域，不能读取无效页或先算非法值再掩码。K/V 的数学视图沿用 Contract；下游块中的无效 lane 不属于 J(i)。具体 FP32 归约树尚未定，误差需要实现验证。

有限精度自检：DOC 1 的默认浮点有效域为有限值，因此 fp8_e4m3fn 输入幅值至多 448。内积绝对值至多 `256*448^2=51,380,224`，乘 0.0625 后至多 3,211,264，远离 FP32 溢出。非空可见域保证 Nb>=1、每个实际 J(i) 非空；指数参数不大于零，首轮 alpha 独立分支不计算无意义的无穷差。精确算术下 z<=1,051,776，|u[d]|<=448*1,051,776，重缩放系数在 [0,1]，没有明显中间量溢出路线。此分析不证明误差阈值：长链舍入、score 误差及正负 V 抵消仍须按原 atol/rtol 验收。

## 有意义的候选与变化

| 方案或变化 | 当前判断 | 原因与成本 |
| --- | --- | --- |
| 上述不分 KV 主线 | 首先展开的一份实际方案 | 无跨 CTA 合并，无 scratch 容量前提；工作数只有 BatchCount*q_len_per_req，少 batch 时利用率可能不足。 |
| split-KV，每段固定 1024 tokens，单独 kernel 合并 | 值得独立 schema；本次只展开必要关系 | 输出由多个部分统计量形成，需要完整解释生产、物化、消费和第二次提交。可缩短单 worker 链并增加工作，增加 scratch、重复 query 读取与归并。 |
| Bk=32 或 128，或 heads 每份改成 4/8 | 参数变化，需关联缓冲、状态和工作数，不新建完整节点 | 前者改变循环/缓冲；后者增加并行度但重复搬运 KV。同一个在线算法、输出所有权和单 CTA 主组织仍成立。 |
| q_len_per_req=1 与 4 | 当前方案的输入工作量变化 | query_tail partition 大小由输入确定，未选择不同算法、worker 或 buffer。 |
| 把四个 t 合并以复用 KV | 暂作分块参数候选 | 同一 CTA 的输出行数与状态增加四倍，独立工作减少；若随后改变主协作组织才需重新判断独立方案。 |
| 双缓冲并拆 Load/Compute role | 未选的局部候选 | 可以围绕同一 MAIN、K/V 块和状态说明发布/复用；是否有收益取决于硬件原语、SMEM 与驻留。 |
| Tensor Core 的 QK/PV 路线 | 未展开候选，暂不能判断是局部变化还是独立方案 | 输入原为 FP8，但 PV 权重是 FP32；若需要窄化 E，必须回数学层记录精度，若需要新的主角色组织也不能当作单个指令替换。缺少支持形状、精度行为和 CuTe 接口资料。 |

这里不选择按 max_seq_len 的某个阈值自动切路线。缺少 BatchCount/seq_lens 分布、设备并行规模和提交成本，不能把“长 KV”直接等同于“split 一定快”。

split-KV 的具体数学候选：固定分段长度 C=1024（设计初始选值，不是硬件已知最优值），段 s 覆盖 `[s*C,min((s+1)*C,Nbt))`。每段用同一个在线求值得到 `(M_s,Z_s,U_s)`；空段直接写 `(-inf,0,0)`，不执行空 softmax。合并仅遍历 Z_s>0 的段，令 `M=max M_s`，`beta_s=exp(M_s-M)`，`Z=sum32 beta_s*Z_s`，`U[d]=sum32 beta_s*U_s[d]`，最终 `bf16(bmm2_scale*U/Z)`。至少一段非空。部分值需按 (b,t,s) 保留到合并消费结束。

每个 (b,t,s) 的 FP32 部分结果为 `HeadCount*(D+2)*4=16,512` 字节，写出、读回约 33,024 字节。不计 cache，每份 (b,t) 的 KV 输入约 `512*Nbt` 字节，query 4,096 字节；分段将 query 重读到每段一次。满 1024-token 段的部分结果往返约为其 KV 输入的 6.3%，还有第二次提交、统计归并及尾段空工作。该比值不是延迟收益证明。

若使用 `Smax=ceil_div(max_seq_len,C)` 的矩形 grid，scratch 至少需要 `BatchCount*q_len_per_req*Smax*16,512` 字节，另有对齐；当前 WorkspaceBytes 没有相应下界。候选可写明容量成立时的范围，但不能把它伪装成覆盖整个 Contract 的完整实现。是否允许实现自行分配额外 scratch、其分配是否计时，当前输入未明确。

## 主线的最小 work_schedule 尝试

数学关系收束后，选 Bk=64，单 worktile 对应 (b,t)，覆盖全部 heads 和 d；一个 worker 是一个 CTA，一份工作处理完就退出。此选择把一页 KV 读入后复用于 16 heads；没有跨 worker 结果依赖。分配按 batch、query_tail 展平，并不假设相邻 CTA 同时执行或 L2 一定命中。

```yaml
work_schedule:
  thesis: Reuse each page across all heads within one CTA and avoid partial-result traffic; small batches may provide insufficient parallelism.
  step:
    MAIN:
      describe: Compute all attention heads for one query-tail token and overwrite its output rows
      decomposition:
        strategy: Bk = 64. Work coordinates are b and t. Iterate i over [0,Nb), with Nbt and Nb from the mathematical definition; only J(i) is valid.
        partition:
          batch: {meaning: Request index b, size: BatchCount, tile: 1}
          query_tail: {meaning: Tail-token index t, size: q_len_per_req, tile: 1}
        worktile:
          invariant:
            query: {meaning: All query heads for the selected b and t, shape: [head: HeadCount, d: D], dtype: 'ref:contract.algorithm_requirement::query.dtype'}
            seq_lens: {meaning: Length of the selected request, shape: [], dtype: 'ref:contract.algorithm_requirement::seq_lens.dtype'}
          iteration:
            block_tables: {meaning: Page-table entry b and i for the current logical page, shape: [], dtype: 'ref:contract.algorithm_requirement::block_tables.dtype', direction: ascending}
            K: {meaning: Keys in the current J(i), shape: [kv: Bk, d: D], dtype: 'ref:contract.algorithm_requirement::K.dtype', direction: ascending}
            V: {meaning: Values in the current J(i), shape: [kv: Bk, d: D], dtype: 'ref:contract.algorithm_requirement::V.dtype', direction: ascending}
        target:
          single_tile:
            output: {meaning: Final rows written into out and returned through its alias, shape: [head: HeadCount, d: D], dtype: 'ref:contract.algorithm_requirement::output.dtype'}
      mapping:
        worker: {unit: CTA, unit_count: 1}
        worker_count: worktile_count
      schedule:
        worktile_order:
          - {by: batch, direction: ascending}
          - {by: query_tail, direction: ascending}
        worker_assignment:
          preassigned: {start: worker_id, count: 1}
          refill: none
          exit: assigned_done
  flow:
    launch: [{step: [MAIN]}]
```

人工覆盖核对：`BatchCount*q_len_per_req` 份工作恰好覆盖全部 query；每份覆盖 16 heads 和 256 输出列；J(i) 无重叠且合起来等于该 t 的可见前缀；out 不先读后写；无外部计数器、旧 scratch 或额外输出。每 worker 只做一份预分配工作，所以没有跨工作缓冲寿命。

粗预算：一页原 FP8 K/V 共 32 KiB，一份 query 4 KiB；拟缓存到 SMEM，合计 36 KiB。u 为 16 KiB，若逐页点积结果单独暂存再加到 u，另需 16 KiB；S/E 各最多 4 KiB，m/z/alpha 等为少量行向量。按这份保守方案，寄存器逻辑值约 40 KiB/CTA，尚未算指令临时量、地址和分配粒度。拟用 8 warps 协作；这些是需求量与设计选择，尚无硬件资料证明驻留。若资源不合适，应先减少 heads 分块或改页内计算组织，不声称这个配置一定合法或足够快。

split 候选的调度关系另写，不混进主线 YAML：PART 的 partition 为 batch/query_tail/segment，segment.size=Smax、tile=1，一 CTA 一段；无效段仍写中性部分值。MERGE 的 partition 为 batch/query_tail，一 CTA 合并 Smax 份统计量，最终写 out。MERGE 才填写 `tile_reduction`，dimension 为 segment，input 指向上游 M/Z/U 部分结果，method 指向上述分段合并定义。

其最小提交和交接形状如下；Mpart/Zpart/Upart 的正式 symbol、两个完整 step 与 scratch 字节布局尚未填写，故是候选片段：

```yaml
flow:
  launch:
    - step: [PART]
    - step: [MERGE]
      start_after: PART.kernel_done
  handoff:
    Mpart: {producer: PART, consumer: [MERGE], storage: global}
    Zpart: {producer: PART, consumer: [MERGE], storage: global}
    Upart: {producer: PART, consumer: [MERGE], storage: global}
```

PART 全部覆盖消费者读取的槽，包括空段，所以不依赖首次隔离使用前的清零，也不填写同义 initialize。MERGE 的就绪来自明确的 kernel_done，不能只靠两个 launch 条目的排列顺序。Scratch 保留到 MERGE 的全部读取结束。

## 最小 pipe 尝试与停止点

选一个 MAIN role、8 warps，所有动作 serial，先建立可讨论的完整局部顺序。没有声称重叠，也不为展示 waiting 而虚构另一个角色。下段位于 `tile_pipeline.step.MAIN.role`；S/E/u/W/m/z 以及 query/K/V 的完整 storage 声明尚未写，所以它只是 pipe 与动作契约片段。m/z/u 采用上面的数学定义；W 是当页 `dot32(E,V)` 的结果，不改变上游数值路径。

```yaml
MAIN:
  info:
    describe: Cooperatively load pages and compute all heads for this worktile
    member: warp[0:8]
    using_verb:
      load:
        QueryLength:
          execution_mode: serial
          describe: Read the selected query and seq_lens from the invocation inputs; retain query for this worktile and make Nbt and Nb available
        KV:
          execution_mode: serial
          describe: Read block_tables[b,i], then load only valid K and V rows in J(i) into the current page buffers; done includes CTA consumer visibility
      compute:
        Init:
          execution_mode: serial
          describe: Initialize m, z and u for this worktile by the upstream initial values without reading their old contents
        Weights:
          execution_mode: serial
          describe: Read the current Scores result and previous m and z; produce current m, z, alpha and E by the online-softmax definition
        UpdateU:
          execution_mode: serial
          describe: Read current WeightedValue W, current alpha and previous u; produce current u by the upstream recurrence and finish all reads of W and alpha
        Output:
          execution_mode: serial
          describe: Read final u and z, normalize and narrow by the upstream definition, and overwrite this worktile's out region with final output
      matmul:
        Scores:
          execution_mode: serial
          describe: Read the query retained by QueryLength and current K supplied by KV; produce S for every valid head and key by the upstream scaled dot product
        WeightedValue:
          execution_mode: serial
          describe: Read current E from Weights and current V from KV; produce FP32 W by the upstream dot32 reduction, with all operand reads finished at done
  pipe:
    - load:QueryLength
    - compute:Init
    - for(i=0:Nb):
        - load:KV(i)
        - matmul:Scores(i)
        - compute:Weights(i)
        - matmul:WeightedValue(i)
        - compute:UpdateU(i)
    - compute:Output
```

这里各动作的 serial 完成顺序已经覆盖数据就绪与单槽复用，按 DOC 5 省略 waiting。首轮 Init 已完成；页内读写依次完成；下一页覆盖 K/V 前，该页 Scores/WeightedValue 已结束；末轮完成后才 Output；Nb 非空。动作内部的 CTA 协作可见性需要 CuTe 实现提供，片段没有绑定具体 barrier API。

停止在“已选计算、初步工作组织、同步 pipe 片段”。没有继续填完整 storage、realization 或异步版本，因为对应硬件容量、执行原语和数值路径支持没有可靠资料；也没有把缺项填成伪造参数。当前可审阅主要方向和局部顺序，但完整实现还需补资源/原语依据、精度实现与验收，以及 split 的 scratch 条件。
