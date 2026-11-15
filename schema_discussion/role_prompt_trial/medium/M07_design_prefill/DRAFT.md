# M07：prefill Planner 手工试走草稿

这是从给定 Contract 独立推导的局部草稿，不是完整可交给 Coder 的 Design。仅阅读 AGENTS.md、DESIGN_SCHEMA.md、schema_template/README.md 及正式 schema_docs 的有关章节；未读该算子旧 Design、dispatch、其他试走报告或旧研究，未运行 Python、checker、构建、GPU 或测试。下面英文 YAML 是表达尝试，中文是推导与未完成项。

## 1. 起点与第一次选择

任务是 MQA：16 个 query head 共用一个 KV head，D=256、Page=64，右对齐因果域。一个 query 至少能看到一个 key。调用整体计时；新增任务表准备、拆分结果物化、归并 launch 都要付费。`out` 初值忽略，返回相同对象；主方案不使用 scratch，因而不依赖它首次隔离调用后是否仍为零。

没有实际 shape 集合、长度分布、优化指标权重。我把“多请求或较长 query 已提供相当多独立行时，先利用同一 query 各头共享 KV”作为主路线的优先情景，而不声称这是用户最重要的实测 workload。少请求/短 query/长 KV 明确比较另一种工作分解。

直接参照：按每个 `(b,q,h)` 求分数、softmax 和加权和；若完整物化分数/概率，令 `T=sum_b sum_q count(J(b,q))`，仅一份 FP32 分数就是 `4*16*T` 字节。独立按 head 处理还可能重复读取 KV；这是调度成本，先不写成数学账本收益。

选在线 softmax，不物化完整分数/概率；不利用输入内容相关性，不截断有效历史。主方案沿 KV 正向扫描。暂不选择 BF16/FP8 概率窄化，所有 softmax 状态和 PV 路径都按 FP32 表达。两个乘积都可以由 SIMT FP32 运算实现其数学关系；本草稿没有给它们虚构 Tensor Core 吞吐。此选择使性能前景较弱，但保留了一条明确的数值路线；快速矩阵路径是尚缺实现证据的候选。

## 2. 数学改写与局部自检

对全局 query 编号 `a`，唯一查到 `b` 满足 `cum_seq_lens_q[b] <= a < cum_seq_lens_q[b+1]`，再得 `q=a-cum_seq_lens_q[b]`。定义 `C(a)=L(b)-Qlen(b)+q+1`，`N(a)=ceil_div(C(a),Bk)`，块 `i` 的有效 key 为 `[i*Bk,min((i+1)*Bk,C(a)))`。`Bk` 交由 work_schedule 绑定。每行的 16 个 head 独立递推、共享 K/V 输入。

数学定义用 FP32 默认求值；下列符号中 `h` 覆盖全部 16 个头，`d` 覆盖全部 D 通道：

```text
m_h(-1) = -inf; l_h(-1) = 0; U_h,d(-1) = 0.
S_h,j(i) = 0.0625 * dot32(query[a,h,:], Kref[b,j,:]).
r_h(i) = max over valid j in block i of S_h,j(i).
m_h(i) = max(m_h(i-1), r_h(i)).
alpha_h(0) = 0.
alpha_h(i) = exp(m_h(i-1)-m_h(i)) for i > 0.
E_h,j(i) = exp(S_h,j(i)-m_h(i)) for valid j; zero otherwise.
l_h(i) = alpha_h(i)*l_h(i-1) + sum32(E_h,valid(i)).
Uscaled_h,d(i) = alpha_h(i)*U_h,d(i-1).
U_h,d(i) = mma32(E_h,valid(i), Vref[b,valid(i),d], Uscaled_h,d(i)).
out_after[a,h,d] = bf16(U_h,d(N(a)-1)/l_h(N(a)-1)).
output is the same tensor object as out.
```

`Kref/Vref` 在这里重新按 Contract 的逻辑页映射定义为相同数学对象；未复制表项或重排 cache。S/E 只有当前块有逻辑寿命；`m/l/U` 为 `carry_forward (a); i=-1..N(a)-1`。Q 在工作内保持；完整输出只写一次。`alpha` 和 `Uscaled` 是当前轮的中间值。

接口登记尝试（只是 math_reshape.symbol 的局部，不是一份完整 YAML）：

```yaml
Bk: {meaning: Key iteration length, binding: bind:design.work_schedule::Bk}
m: {meaning: Running row maxima, shape: [head: Hq], dtype: fp32, retention: 'carry_forward (a); i=-1..N(a)-1'}
l: {meaning: Running exponential sums, shape: [head: Hq], dtype: fp32, retention: 'carry_forward (a); i=-1..N(a)-1'}
U: {meaning: Running unnormalized output, shape: [head: Hq, channel: D], dtype: fp32, retention: 'carry_forward (a); i=-1..N(a)-1'}
S: {meaning: Current score block, shape: [head: Hq, key: Bk], dtype: fp32}
E: {meaning: Current exponential block, shape: [head: Hq, key: Bk], dtype: fp32}
```

有限 FP8 输入的初步分析：绝对值至多 448，分数绝对值至多 `256*448^2/16=3,211,264`，指数自变量非正。有效域 `C>=1`，首块非空，不计算 `-inf-(-inf)`；显式首轮 alpha=0。若 L 是正 int32，则精确数学 l 至多 L，U 的绝对值至多 `448*L`，无 FP32 溢出量级风险。最后一页只加载有效 j，无效槽置零；S 的无效位置不参与最大值和指数计算，E 的无效位置置零。这样不以“后面乘零”掩盖越界或 NaN。

这不是有限精度误差证明：长归约的舍入、在线重缩放与 golden 的不同顺序仍需按既有 atol/rtol 验证。更关键的是 Contract 没有限定 query/KV 为有限值；上述分析不是完整域保证。NaN 的语义与验收缺口记录在 M07-02。不能把有限子域当作已获得的新 Contract。

数学收益是避免完整 S/P 物化、状态随 key 长度不增长；新增每块运行最大值、alpha 重缩放和约 Hq*D 的 U 缩放。没有微基准，不能断言这比两遍流式读取或特定矩阵路径更快。

## 3. 已选工作组织

每份工作是一条全局 query 的 16 个头、全部输出通道，沿 KV 页局部迭代。一个 CTA 一个工作，提交 `TotalQ` 个 CTA。每个工作通过 query 前缀数组二分定位请求；不在 host 读取长度，不预建任务表，不用持久队列。KV 的单次块加载服务 16 个头；Q 对所有 KV 块复用。不同 query 之间的 KV 复用仅可能来自缓存，未计为保证。

```yaml
work_schedule:
  thesis: Reuse each KV page across all query heads within one query-row worker; use independent query rows for parallelism without a task-list preparation pass.
  step:
    MAIN:
      describe: Produce all output heads and channels for one global query row.
      decomposition:
        strategy: |
          Bk = 64. Work coordinate a is a global query index.
          Find b by binary search of cum_seq_lens_q and derive q, C(a), and N(a) by the math definition.
          Iterate i from 0 to N(a)-1; page lookup uses block_tables, and only j < C(a) is read.
        partition:
          token: {meaning: Global query row, size: TotalQ, tile: 1}
        worktile:
          invariant:
            Q: {meaning: Query row query[a,:,:] for all heads, shape: [head: Hq, channel: D], dtype: fp8_e4m3fn}
          iteration:
            K: {meaning: Shared logical key page, shape: [key: Bk, channel: D], dtype: fp8_e4m3fn, direction: ascending}
            V: {meaning: Shared logical value page, shape: [key: Bk, channel: D], dtype: fp8_e4m3fn, direction: ascending}
        target:
          single_tile:
            out_after: {meaning: Completed output row, shape: [token: 1, head: Hq, channel: D], dtype: bf16}
      mapping:
        worker: {unit: CTA, unit_count: 1}
        worker_count: worktile_count
      schedule:
        strategy: Output ownership is unique; the callable returns the caller out object after scheduling the output-producing kernel.
        worktile_order:
          - {by: token, direction: descending}
        worker_assignment:
          preassigned: {start: worker_id, count: 1}
          refill: none
          exit: assigned_done
  flow:
    launch: [{step: [MAIN]}]
```

从大 token 编号开始，仅让每个请求内较重的后部 query 倾向于先编号，不保证长请求优先，也不保证硬件按编号启动。跨请求严重倾斜时此排列不够强；我没有把“排列”说成负载均衡保证。每个 CTA 一份工作，因此空闲 SM 可承接尚未启动的 CTA，但已启动的一个很长工作仍不可切分。

主要算术约为每个工作 `2*Hq*C(a)*D` MAC，加上在线缩放与 softmax；有效 KV 逻辑载荷为 `2*C(a)*D` 字节，Q 为 `Hq*D` 字节，输出 `2*Hq*D` 字节。这里按工作计复用，未声称 DRAM 流量。

## 4. 局部流水表达尝试

选择 Load 的 1 warp 和 Calc 的 7 warp，合计 256 线程。不是因为知道 7 warp 最优，而是给局部重叠一个具体预算；工作与分支选择不依赖此精调。Q 单槽、KV 双槽。Calc 串行执行 QK、softmax/缩放、PV，Load 可领先两页。这里的 `matmul` 是数学动作，不保证由异步矩阵硬件实现。

当前资源账：Q 4 KiB；双槽 FP8 KV 64 KiB；S/E 同槽 4 KiB，总 SMEM 至少 72 KiB。U FP32 为 16 KiB，m/l/alpha 约 192 B，另有地址与临时寄存器。下述 UState 是全 CTA 的聚合寄存器容量，不是每线程 16 KiB。硬件容量/寄存器预算未给定，尚不能证明一 CTA 能驻留。未登记屏障编号和具体原语；因此这仍是“待绑定实现保证”的局部草稿。

```yaml
tile_pipeline:
  thesis: Overlap paged KV transfers with the current score and output update using two KV slots; keep the numerical recurrence serial within the compute role.
  step:
    MAIN:
      index:
        i: {meaning: Key-page iteration of this query-row worktile, definition: [0, 'N(a)']}
      storage:
        QSlot: {meaning: Query row shared by computation, depth: 1, location: SMEM, shape: [head: Hq, channel: D], dtype: fp8_e4m3fn, total_size: 4096}
        KVSlot:
          meaning: Two key-value page slots
          depth: 2
          location: SMEM
          shape: [kind: 2, key: Bk, channel: D]
          dtype: fp8_e4m3fn
          total_size: 65536
          view:
            'K(i)': {slot_index: 'i%2', region: {kind: [0, 1]}}
            'V(i)': {slot_index: 'i%2', region: {kind: [1, 2]}}
        ScoreSlot:
          meaning: Score storage overwritten by exponentials after each score is consumed
          depth: 1
          location: SMEM
          shape: [head: Hq, key: Bk]
          dtype: fp32
          total_size: 4096
          view:
            'S(i)': {}
            'E(i)': {}
        UState: {meaning: Distributed running output accumulator, depth: 1, location: register, shape: [head: Hq, channel: D], dtype: fp32, total_size: 16384}
      role:
        Load:
          info:
            describe: Resolve this query row and stage its operands for Calc.
            member: warp[0:1]
            using_verb:
              compute:
                Coordinates:
                  execution_mode: serial
                  describe: Read contract cum_seq_lens_q and seq_lens to derive this role's b, q, C(a), and N(a) by the work mapping; no shared metadata is consumed.
              load:
                Q:
                  execution_mode: async_unordered
                  describe: Read contract query[a,:,:] into QSlot; done guarantees all source reads and visibility to Calc are complete.
                KV:
                  execution_mode: async_unordered
                  describe: Read the contract block_tables entry and the valid K and V page region into KVSlot views K(i) and V(i); zero unused tail locations; done guarantees both regions are visible to Calc and all source reads are complete.
          pipe:
            - compute:Coordinates
            - load:Q
            - for(i):
                - load:KV(i)
          waiting:
            released:
              - "{issue load:KV(i+2)} ← [{done Calc::matmul:E(i)@V(i)}]"
        Calc:
          info:
            describe: Compute the online attention recurrence and store the completed query row.
            member: warp[1:8]
            using_verb:
              compute:
                Initialize:
                  execution_mode: serial
                  describe: Independently resolve b, q, C(a), and N(a) from contract metadata; initialize m, l, and UState by the math initial conditions without reading old accumulator or out contents.
                Update:
                  execution_mode: serial
                  describe: Read S(i) in ScoreSlot and previous m, l, UState; form current m, alpha, l and Uscaled by the online recurrence; replace ScoreSlot with E(i) only after its needed S values are consumed.
                Output:
                  execution_mode: serial
                  describe: Normalize the final UState by final l and convert to BF16 by the math output relation; place the result in distributed registers for store:O.
              matmul:
                Q@K:
                  execution_mode: serial
                  describe: Read QSlot and KVSlot view K(i), compute the FP32 score relation into ScoreSlot view S(i), and finish all operand reads before done; invalid keys are excluded.
                E@V:
                  execution_mode: serial
                  describe: Read ScoreSlot view E(i), KVSlot view V(i), and Uscaled in UState; update UState by the FP32 weighted-sum relation; done ends all E and V reads.
              store:
                O:
                  execution_mode: serial
                  describe: Read the BF16 result of compute:Output and write contract out[a,:,:]; done guarantees the output writes are complete and visible at the callable output boundary.
          pipe:
            - compute:Initialize
            - for(i):
                - matmul:Q@K(i)
                - compute:Update(i)
                - matmul:E(i)@V(i)
            - compute:Output
            - store:O
          waiting:
            prepared:
              - "{issue matmul:Q@K(i)} ← [{done Load::load:Q}, {done Load::load:KV(i)}]"
```

手工核对：首两次 KV 搬运无需前轮释放；第三次覆盖槽 0 要等 PV(0)，包含旧 V 读完，同时 Calc 的串行顺序已保证旧 K 读完。QK 等同一页 K 和 V 都完成，较保守但容易闭合。Calc 的串行 PV→下一轮 QK 保护 ScoreSlot，未另加重复边。最后一轮必有实际已发布的 KV 前驱；Load 结束发射不能让 CTA 提前销毁 SMEM，CTA 结束仍需 Calc 的输出完成。没有跨 worktile 接续，因为每 worker 仅一份工作。

仍缺：m/l/alpha 和 Output 寄存器的完整 storage 登记；所有本地名字和上游数学接口的完整注册；page copy/可见性原语及其 CuTe 支持；数值验收。这些未以空字符串或无根据的实现保证补齐。ScoreSlot 的空 view 参照 DOC 5 视图章节单槽 `S: {}` 的例子，表示整槽绑定，非通用占位。

## 5. 候选与分支判断

| 工作情景/变化 | 候选与取舍 | 当前记录判断 |
| --- | --- | --- |
| 多请求，或少请求但 query 很长 | 当前一 query/CTA，Hq 内复用。CTA 数为 TotalQ；长 prefill 自然提供很多工作。 | 主方案；没有实测最优声明。 |
| 少请求、短 query、长 KV | split-KV：每 query 切若干连续 key 段，各得 `(m_s,l_s,U_s)`；另一个 kernel 用共同最大值重标定合并。 | 新的完整候选，不能只把 split 因子作为当前 YAML 参数。 |
| 少请求、query 和 KV 都短 | 按 head 子组分工作增加并行度，牺牲 K/V 重读；更少线程、单槽可降低开销。 | head 分组看起来可定位现有分块/角色/存储的局部变化，但尚未闭合记录；不保证短工作值得额外 CTA。 |
| 同请求多个相邻 query 共用 KV | 每 CTA 两条 query、全部头；进一步复用 KV，U/score/Q 大致翻倍、工作数减半。 | 不能直接把当前全局 token tile 从 1 改成 2：可能跨请求，需要请求内任务映射。参数/局部边界尚有疑问（M07-04）。 |
| 极不均衡请求、长短 query 混合 | 持久 CTA 动态领取，可减少尾波；需要任务来源、领取、退出及跨工作存储寿命。 | 如引入整条领取主循环和跨工作重叠，应作为独立方案候选。没有资料支持固定 worker_count。 |
| Load/Calc 双角色改成单角色提前发射 | 一个角色可先发下一页、再做本页串行计算；要重新写控制流及两槽关系。 | 围绕现有动作可闭合的局部变化，不因 role 数变化自动新建完整方案。 |
| BF16/FP8 概率路径、原生低精度矩阵乘 | 可能大幅改善 PV 吞吐，但概率窄化、累加与误差需说明；输入 FP8 不等于中间概率可随意 FP8。 | 暂未选择；必须先回数学层，不由流水 noun 的 dtype 暗改。 |

split-KV 的初步关系为 `M=max_s m_s`，`w_s=exp(m_s-M)`，`out=bf16(sum_s w_s*U_s / sum_s w_s*l_s)`。若有 S 个分段，每 query 的 FP32 部分结果约 `4*S*Hq*(D+2)` 字节，写加读约两倍；额外提交和每段重复 Q 读取均计入。空段必须跳过或定义显式空状态并避免 `-inf-(-inf)`。workspace 的容量没有下界，不能据现有 Contract 保证这份候选可用 caller scratch；本草稿不补造可用字节数或分配策略。

没有构造 selector 阈值，也没有交付第二套完整 schema。若继续，应先补输入域/硬件与主要 workload，再确定是否值得完整展开 split-KV；当前主方案在全 Contract 上的验收仍未完成。
