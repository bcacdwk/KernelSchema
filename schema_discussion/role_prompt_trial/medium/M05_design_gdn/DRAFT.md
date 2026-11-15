# M05：GDN prefill Planner 试走草稿

本稿只依据给定 `inputs/gdn_prefill.md` 的 Contract 和当前正式文档推进。没有读取本算子的旧 Design、dispatch、其他试走或研究报告；没有执行 Python、extract、checker、GPU、构建、测试。阅读了 AGENTS、DESIGN_SCHEMA、作者入口 `schema_template/README.md`，按表达需要查阅 DOC 0–6，并读执行器对齐说明。以下是手工设计进展，**不是完整 schema，不代表正确性或性能验收**。

## 1. 任务边界与设计起点

- 保留 value-by-key 的 FP32 状态，更新后 inclusive readout，全部覆盖 O 和 S_final；不读取两个预分配输出的旧内容。显式 initial_state 不能当作零。
- 独立轴是 sequence b、value/state head h、value row j。hq=floor(h/8)。同一个 j 的 DK 归约可以协作，但同一 (b,h,j) 的 token 顺序是真依赖。
- B=1..16、T<=16384；序列严格非空。没有给出各长度、B 的频率或性能计分权重。特别考虑 B=1 且长序列，以及 B>1 且长度不均匀；不把这些例子当成新增 when 限制。
- alpha 可为零、可接近一。不能假设每隔固定长度就能忽略前缀，也不能从“通常衰减快”推出截断状态是合法优化。
- 对照值来自 FP32、block_size=32、log-space decay 的 oracle。逐 token FP32 也不是逐位一致的 oracle；误差要求保持 O 的 0.01/0.01 与 S_final 的 0.005/0.001。
- 计时必须包括实际 forward、边界处理和所有辅助提交。没有性能数据，以下数字是逻辑载荷或手算容量，不是 DRAM 实测。

直接求值参照：每 token/head/value row 做一次 DK 状态预测归约、一次 DK 更新、一次 DK 输出归约，并对 DK 状态元素衰减。约 3*DK*DV MAC 加 DK*DV 衰减乘法/token/head（不把 MAC 算两次 FLOP）。若状态每 token 读回写回全局，额外载荷约 8*T*HV*DV*DK bytes；本次候选将这类状态往返消除。这不是声称给定 oracle 真有该访存行为。

## 2. 候选推导与比较

### R：直接递推，状态驻留，value row 分片

保留 Contract 的递推；一份工作处理一个 (b,h) 的 J 个 value row 和该序列全部 token。DK 不跨 worktile 切分，没有跨工作归约。每份状态只从 initial_state 读入一次，最后写 S_final；O 随 token 写出。q/k 在同一份工作内由 J 行共享。

收益是取消状态随 token 的全局物化，并以 DV/J 增加工作数。代价是 q/k、alpha/beta 被不同 j 分片重复读取，token 依赖仍长 L_b。J=16 时 B*32*8=256B 个 CTA 工作，B=1 也有 256 份；不能把提交数当成同时驻留数。相比一个 (b,h) 的 64 KiB 状态，当前工作状态为 8 KiB。

这是当前**暂选的下推路线**：数学与状态接口最清楚，可以形成独立可核对的调度和流水片段；它不是已经证明最快的 long-prefill 设计。下面的 B、S 比较没有被遗忘，也没有因文档要求完整节点而强行填完。

### B：无衰减除法的 chunk 三角改写

以下省略 b,h，保留 DV×DK 状态方向。C 为工作调度绑定的 chunk 长，考虑 C=32；每个 chunk 有 m=min(C,remaining) 个真实 token，局部 i=0..m-1。所有展示的矩阵乘均用 mm32；辅助乘加默认 FP32/RNE，不引入 BF16/TF32 状态或工作量。

定义 a_i=alpha_i+epsilon。定义 p(i,j) 为从 j+1 到 i 的 a 的递增顺序乘积，p(j,j)=1，p(i,-1) 为 chunk 首到 i 的乘积。只在合法局部范围求值。**不使用 p(i,-1)/p(j,-1)**；长乘积下溢后不会形成 0/0。

用行形式 K_i、Q_i 和 V_i，先求：

```text
F[i,j] = beta_i * p(i,j) * dot32(K_i,K_j),  0 <= j < i < m
R[i,:] = beta_i * V_i - sum32_{j<i}(F[i,j] * R[j,:])
W[i,:] = beta_i * p(i,-1) * K_i - sum32_{j<i}(F[i,j] * W[j,:])
U = R - mm32(W, S_inᵀ)
E[i,:] = p(m-1,i) * K_i
S_out = p(m-1,-1) * S_in + mm32(Uᵀ,E)
H[i,j] = p(i,j) * dot32(Q_i,K_j), 0 <= j <= i < m
Y[i,:] = p(i,-1) * mm32(Q,S_inᵀ)[i,:]
         + sum32_{j<=i}(H[i,j] * U[j,:])
O_chunk = bf16(scale * Y)
```

R/W 是单位下三角系统的前代求值，空和为零。为了固定实际数值路线，j 按升序前代，行间求和默认 FP32；没有把“inverse”当作未说明算法。末 chunk 只算真实 m 个 token，不让补齐 token 更新状态。S_in 来自 initial_state 或前 chunk 的 S_out；最后一块产生原 S_final。

推导核对：展开原递推得 State_i=p(i,-1)S_in+sum_{j<=i}p(i,j)U_j K_j。代回第 i 个 delta 的预测项得到 (I+F)U=diag(beta)V-diag(beta*p(i,-1))K*S_inᵀ，因而 U=R-W*S_inᵀ。输出使用更新后的包含 j=i 的状态。这是实数等价推导，不是有限精度证明。

数值自检到这里：a 的有效路径非负且很小值被 epsilon 稳定；局部乘积没有增长性倒数，没有奇异主对角求逆；乘积仍可能下溢。R/W 的前代与 oracle 的 block inverse/log-space 会有不同舍入。仅凭 FP32 不能保证误差，归一化 BF16 k 也不等于精确单位长度。没有硬假设 q/v 有固定最大值，Normal 中心描述本身不给确定性有限常数。因此不声称已证明整个 Contract 的误差界；这不妨碍核对接口及物化量。

可能的组织 B3：

1. PREP 对 (b,h,chunk) 独立算 R、W，保留供状态工作消费。
2. STATE 对 (b,h,j-slice) 顺序遍历 chunk，算 U 和 S_out，保存各 chunk 的 FP32 S_in 快照 S_entry。
3. READOUT 对 (b,h,chunk,j-slice) 独立计算 H、Y、O。最终状态由 STATE 写出。

它把主要状态链从 L_b 个 token 过渡改成 ceil_div(L_b,C) 个块；PREP 的前代仍有 C 级局部依赖，不能把总依赖长度直接说成除以 C。R/W 的前代有约 O(C²*(DK+DV)) MAC/chunk，Gram 有 O(C²*DK)，STATE 与 READOUT 的块乘仍有 O(C*DK*DV)。是否可转到更合适的执行单元，要看已选 CuTe/硬件精度路径；不凭 Tensor Core 的名字假定 FP32 输入有便宜等价指令。

最大 T 时，chunk 总数 Nc_total=sum_b ceil_div(L_b,32)<=527。仅 FP32 S_entry 上界是 4*Nc_total*HV*DV*DK bytes，约 1.03 GiB；U 或 R 各为 4*T*HV*DV bytes，最大 256 MiB；W 为 4*T*HV*DK bytes，也最大 256 MiB。S_entry 至少一写一读约 2.06 GiB 逻辑载荷，尚未计 R/W/U、H、输入、输出。这个开销足以改变方案选择，不能只记录链变短。

B2：将 READOUT 放回 STATE 的同一 chunk 内，保留 R/W 的 PREP，U 与 S_in 留在工作内直到输出完成。可去掉快照与 U 的全局交接，但失去 chunk 级输出并行。进一步把 PREP 也放入同一工作则省 R/W 物化，又会按 j 分片重复 Gram/W 构造。这三个组织都值得用明确账本比较，当前没有算力/带宽相对成本，尚不挑选 B 的最终组织。

### S：完整 affine chunk scan

对单 token 定义右作用矩阵 A_i=a_i*(I-beta_i*mm32(K_iᵀ,K_i))、B_i=beta_i*mm32(V_iᵀ,K_i)，有 S_i=mm32(S_prev,A_i)+B_i。相邻段 (A_L,B_L)、(A_R,B_R) 合成为 (mm32(A_L,A_R), mm32(B_L,A_R)+B_R)。这些关系在实数算术下满足结合律，可 scan 得各 chunk 入口，然后重算各 chunk 输出；mm32 实际舍入下不声称结合律成立。

它是真正跨序列内 chunk 并行的候选，独特价值是降低长单序列的主状态依赖；代价是 DK×DK 的稠密变换与 DV×DK 的偏移物化、矩阵矩阵组合，以及改变结合顺序。逐层 scan 的边界、scratch 复用、所有段前缀的数值误差尚未展开。当前不完整设计它；这不是 R 的参数变化。仅根据 alpha 通常小去近似剪掉 A 不获准。

## 3. R 的数学接口草案

这是设计表达片段；计算沿用 Contract，不改变状态意义。S 是当前独立 (b,h) 的实现状态视图，保持 value-by-key 排列；它的每个 DV 分片独立携带相应行。DOC 3 要求 Design symbol 的 shape 写一个实例，因此不照抄 Contract 的 [B,HV,DV,DK] 总体 symbol。

```yaml
math_reshape:
  thesis: Keep the contract recurrence and retain state rows across the sequence to avoid token-wise state materialization.
  rewrite:
    - name: Retain the inclusive recurrent state
      math: |
        Evaluate the Contract compute "Decay and update the state before inclusive readout".
        S(t) denotes its State(t) for one fixed (b,h).
        S(-1) = initial_state[b,h].
        S_final[b,h] = S(L_b-1).
      symbol:
        S:
          meaning: Current value-by-key state for one sequence and value head
          shape: [dv: DV, dk: DK]
          dtype: fp32
          retention: "carry_forward (b,h); t=-1..L_b-1"
```

这段没有给全部公式重新发明状态。数学默认求和不固定具体线程归约树，若实现改成 TF32 或截断乘法输入必须回来修改数学区。scale 的 binary64 输入与默认 FP32 运算的具体结合仍列为 M05-02；该点使本稿还不是可直接交付的完整数值设计。

## 4. R 的 work_schedule 片段与覆盖

```yaml
work_schedule:
  thesis: Split independent value rows while retaining the complete token recurrence in each worker.
  step:
    RECURRENCE:
      describe: Produce output rows and final state rows for one sequence, value head and value-row slice
      decomposition:
        strategy: |
          J = 16. A worktile selects b, h and a J-row interval in dv.
          Iterate local token t from 0 through L_b-1 without splitting its dependency chain.
          Use hq=floor(h/8). The input boundary values determine L_b and the packed token offset.
        partition:
          batch: {meaning: Sequence identity, size: B, tile: 1}
          head: {meaning: Value and state head identity, size: HV, tile: 1}
          dv: {meaning: Independent value rows, size: DV, tile: J}
        worktile:
          invariant:
            initial_state: {meaning: Initial state rows for this worktile, shape: [dv: J, dk: DK], dtype: ref:contract.algorithm_requirement::initial_state.dtype}
          iteration:
            q: {meaning: Query row at the current token and associated query head, shape: [dk: DK], dtype: ref:contract.algorithm_requirement::q.dtype, direction: ascending}
            k: {meaning: Key row at the current token and associated key head, shape: [dk: DK], dtype: ref:contract.algorithm_requirement::k.dtype, direction: ascending}
            v: {meaning: Value rows at the current token, shape: [dv: J], dtype: ref:contract.algorithm_requirement::v.dtype, direction: ascending}
            alpha: {meaning: Decay at the current token and value head, shape: [], dtype: ref:contract.algorithm_requirement::alpha.dtype, direction: ascending}
            beta: {meaning: Update strength at the current token and value head, shape: [], dtype: ref:contract.algorithm_requirement::beta.dtype, direction: ascending}
        target:
          single_tile:
            O: {meaning: All sequence outputs for this value-row slice, shape: [token: L_b, dv: J], dtype: ref:contract.algorithm_requirement::O.dtype}
            S_final: {meaning: Final state rows for this worktile, shape: [dv: J, dk: DK], dtype: ref:contract.algorithm_requirement::S_final.dtype}
      mapping:
        worker: {unit: CTA, unit_count: 1}
        worker_count: worktile_count
      schedule:
        worktile_order:
          - {by: batch, direction: ascending}
          - {by: head, direction: ascending}
          - {by: dv, direction: ascending}
        worker_assignment:
          preassigned: {start: worker_id, count: 1}
          refill: none
          exit: assigned_done
  flow:
    launch:
      - step: [RECURRENCE]
```

尚需补充入口 boundary/scale 的工作块登记及确定性数值消费约定，见 M05-02/03；上述只展开主要输入。S 不是 invariant input，也不是每轮从外部读的 iteration input，它是 math 的 carry_forward 状态，初值列 invariant，实际驻留下放 storage。single_tile 的 O 是该工作产生的完整结果范围，不意味着片上保存 L_b*J 个输出。

手工覆盖：每个 (b,h,j) 恰归一份工作，t 全部覆盖 [0,L_b)，所有 final-state 元素恰写一次；HQ 的组映射不拆分、不归约。cu_seqlens 严格增保证每份至少有一次更新。J=16 整除128，DV 无尾片；token 长度完全按输入。没有序列间合并或跨 CTA 状态依赖。

长短序列：此方案让硬件 grid 调度独立 CTA，不另外造持久队列。长序列最后可能仍有256个长工作，不能依靠短序列空出的 CTA 再分摊同一条 token 链。按长度降序排列值得作为局部调度候选，但产生排序/索引表的开销与计时必须归属，不能只把 `direction: descending` 套到 b 就声称按 L_b 排序。

## 5. R 的小 pipe / waiting / 存储片段

为看清双缓冲的发布与覆盖，在同一个 CTA 选 Loader 一个 warp、Compute 四个 warp。角色规模是具体试选，无已验证驻留结论。Compute 协作保存 J*DK 个 FP32 状态元素，均摊到128线程是64个状态寄存器/线程（仅状态，不是总寄存器预算）。Loader 使用普通同步协作加载也可跨 role 与计算重叠；不要求未知的异步搬运原语。

```yaml
tile_pipeline:
  thesis: Overlap loading the next token with recurrent computation using two input slots and one state copy.
  step:
    RECURRENCE:
      index:
        t: {meaning: Local token ordinal shared by loader and compute, definition: [0, L_b]}
      storage:
        StateRows:
          meaning: Resident state rows carried across local tokens
          depth: 1
          location: register
          shape: [dv: J, dk: DK]
          dtype: fp32
          total_size: 4*J*DK
          view: {S(t): {}}
        QuerySlots:
          meaning: Alternating query input slots
          depth: 2
          location: SMEM
          shape: [dk: DK]
          dtype: bf16
          total_size: 4*DK
          view: {q(t): {slot_index: t%2}}
        KeySlots:
          meaning: Alternating key input slots
          depth: 2
          location: SMEM
          shape: [dk: DK]
          dtype: bf16
          total_size: 4*DK
          view: {k(t): {slot_index: t%2}}
        ValueSlots:
          meaning: Alternating value input slots
          depth: 2
          location: SMEM
          shape: [dv: J]
          dtype: bf16
          total_size: 4*J
          view: {v(t): {slot_index: t%2}}
        GateSlots:
          meaning: Alternating decay and update input slots
          depth: 2
          location: SMEM
          shape: [element: 2]
          dtype: fp32
          total_size: 16
          view:
            alpha(t): {slot_index: t%2, region: {element: [0, 1]}}
            beta(t): {slot_index: t%2, region: {element: [1, 2]}}
      role:
        Loader:
          info:
            describe: Load token inputs into the two shared input slots
            member: warp[0:1]
            using_verb:
              load:
                Token:
                  execution_mode: serial
                  describe: |
                    Read the worktile's q,k,v,alpha,beta at token t from Contract inputs.
                    Write QuerySlots.q(t), KeySlots.k(t), ValueSlots.v(t), GateSlots.alpha(t) and GateSlots.beta(t).
                    Done guarantees all these writes are complete and visible to Compute in this CTA.
          pipe:
            - for(t):
                - load:Token(t)
          waiting:
            released:
              - "{issue load:Token(t+2)} ← [{done Compute::compute:Advance(t)}]"
        Compute:
          info:
            describe: Carry state rows, produce each token output and store the final state
            member: warp[1:5]
            using_verb:
              load:
                Initial:
                  execution_mode: serial
                  describe: Load the assigned initial_state rows into StateRows.S(-1); do not read output_state.
              compute:
                Advance:
                  execution_mode: serial
                  describe: |
                    Consume token t inputs from Loader's declared slot views and the prior StateRows version.
                    Evaluate the upstream recurrence for the assigned rows, replace StateRows with S(t), and write the assigned O token slice.
                    Done includes all reads of token t slots, the state update, and the output write required by this action.
              store:
                Final:
                  execution_mode: serial
                  describe: Store the final StateRows value as the worktile's S_final slice; do not read the destination.
          pipe:
            - load:Initial
            - for(t):
                - compute:Advance(t)
            - store:Final
          waiting:
            prepared:
              - "{issue compute:Advance(t)} ← [{done Loader::load:Token(t)}]"
```

普通内部 DK 归约和状态更新没有为形式逐项拆成动作。状态在 Compute 的 serial 顺序中由初值、上一轮、最终 store 串接，不额外写同义状态等待。Advance 包含输出落位，所以这里没有声称输出存储还能与下一 token 重叠。

边界核对：t=0/1 的槽没有旧读者，不产生负 token 的覆盖等待；t>=2 的加载等待 t-2 的 Advance 结束。L_b=1 时只有一次加载、一次 Advance、一次 Final。Loader 可先装两轮，第三轮开始前受 released 约束，不能覆写正在读的槽。不存在从 Compute(0) 回等 Compute(1) 的环。退出要求各 role 全部职责完成；本稿不使用跨 worktile 连续序号，因为每 worker 只处理一份工作。

已列资源：SMEM 两槽输入 4*(DK+DK+J)+16=1104 bytes；寄存器状态8192 bytes/CTA。还需归约、delta、地址、输出临时值和实际分配粒度，不能把8192 bytes当总寄存器占用。初版可能以单 role 减少专职 Loader 的成本；两个 role 是否真有收益尚未测量。

此处 done 的跨 warp 可见性需要已选栈的真实同步原语实现。Contract 只给出共享存储/屏障能力摘要，尚无绑定资料。本稿没有在 realization 发明 API；可见性是该片段明确的实现条件，不是已证实能力。完整交付还需绑定和容量复核。

## 6. B 的工作接口能走到哪里

当前仅 B3 的部分提交/交接关系可以先明确，尚未完整排列所有工作字段：

```yaml
flow:
  launch:
    - step: [PREP]
    - step: [STATE]
      start_after: PREP.kernel_done
    - step: [READOUT]
      start_after: STATE.kernel_done
  handoff:
    R: {producer: PREP, consumer: [STATE], storage: global}
    W: {producer: PREP, consumer: [STATE], storage: global}
    U: {producer: STATE, consumer: [READOUT], storage: global}
    S_entry: {producer: STATE, consumer: [READOUT], storage: global}
```

PREP 独立维 (b,h,chunk)，STATE 独立维 (b,h,j-slice) 且 chunk 是 iteration，READOUT 恢复独立 chunk 轴。这里 `order: per_worktile` 不适用，因为三类工作不是一一配对；也不能只靠 launch 列表上下位置表达先后。R/W/U/S_entry 都由生产者写满真实消费区域，不因 allocation 未清零而无条件添加 initialize:0。

R/W/U 是每 chunk 保留的值，math symbol 应带 keep_value (b,h,c)；S 是 chunk 递推状态 carry_forward；S_entry 的逻辑实例需要说明它确实为多个 chunk 保存快照。C 用 bind:design.work_schedule::C，在对应 decomposition.strategy 固定32，不同时维护 math value:32。

变长 chunk 的 PREP/READOUT 展平方式未定：可用 Nc_total 加显式 (b,c) 映射，也可用 B×ceil_div(T,32) 上界并早退无效工作。前者需要索引映射准备/消费，后者有空工作和额外提交量。这个决定不影响三角数学或上面的物化字节推导，故先保留，而不是给 partition 写来源不明的 `size: Nc`。

## 7. 完整 schema、局部与参数的本次判断

| 变化/候选 | 暂定分类 | 本次保留理由与限制 |
| --- | --- | --- |
| R 驻留直接递推 | 值得一份完整 schema 的当前主路线 | 状态、输出和独立工作闭合；本次仅局部片段，精度/入口/资料仍有缺口。 |
| B 的块三角计算 | 独立候选，若选定值得完整 schema | 新计算值和主要状态/输出组织需重新贯通；不能叫 R 的 C 参数。未在 B2/B3 中选最终具体方案。 |
| S 稠密 affine scan | 独立但未选完整展开的候选 | 主状态依赖、合并执行者和多层 scratch 都不同；独特但开销与误差未解决。 |
| R 的 J=8/32 | 参数 | 同分片机制，须联动状态容量、CTA数和q/k重复读取；当前YAML仍只选16。 |
| R 的一槽/三槽、Loader规模调整 | 参数或局部，具体差额要展开 | 槽数在同机制下是参数；取消专职Loader会改协作边，可写局部，不等于换算法。 |
| 预先按长度排序 R 工作 | 局部候选 | 需明确排列映射来源与计时成本；未取得可免费访问的host长度。 |
| R 分片参数与输入槽深度配对 | 组合记录 | 需核对寄存器/SMEM、动作读完边；不为每个配对造完整节点。 |
| B3 合并 STATE/READOUT 为 B2 | 分类尚不确定 | 可清楚列出省快照和全局U，但改变整个输出生产位置和并行轴；倾向独立整体组织而非因“少一launch”就归局部。见 M05-07。 |

不会为了共享数学区而造只有 provider 用途的节点；如果最终并列完整 B2/B3 且 math 物化不同，也不能整区 from 一个含快照的版本后用 prose 删快照。当前没有形成合法完整基线，因此这些是候选分类草稿，不是假装已合规的 dispatch 变体条目。

## 8. 停在这里的原因与下一步所需证据

实际做到：直接和块化的计算/状态推导、候选成本比较、R 的独立工作与有界输入环、B3 的主要工作和交接片段、完整/局部/参数分类。没有完成完整 schema，也没有证明性能或契约全域数值验收。

阻止完整交付的具体点：scale 的实际计算精度及边界转换归属未闭合；硬件容量/真实可见性原语与 CuTe支持资料缺口；B 路线的数值和成本证据不足以选最终组织；完整数学符号和辅助工作尚未登记齐全。这些不妨碍上面已完成的手工推导。

工具不是手工设计的门槛。DESIGN_SCHEMA 允许中间稿和继续独立核对，作者入口也允许信息不足时继续不受影响部分。执行器对齐说明限制的是 CLI 端到端运行和验收含义，本次按实验要求不运行它们，没有因旧 checker 去退回旧字段。
