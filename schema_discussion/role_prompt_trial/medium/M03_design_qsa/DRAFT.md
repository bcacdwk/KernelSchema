# M03：从 Contract 开始的 QSA Planner 试走

本稿独立从给定 Contract 推演。阅读了 AGENTS.md、DESIGN_SCHEMA.md、README 的作者入口、schema_template/README.md 和 DOC 0–6 的相关段落；未读本算子旧 Design/dispatch、旧实现、其他试走报告。未运行 Python、checker、构建、GPU 或测试。下文是局部草稿，不是可交付的完整合法 YAML，也没有实测性能结论。

## 1. 实际任务与第一判断

目标是整个 callable 的成本，包括 selection 处理、attention、gate、必要的输出分配。4096 行、每行 24 个 query heads、每 12 个 query heads 共用一个 KV head；每行有效 token 数是 2048–2051。

第一种理解是：每个 `(row, kv_head)` 合作处理 12 个 heads，逐块 gather 同一批 K/V，在线累积 attention，最终在片上完成规定的 BF16 舍入和 gate。这避免把同一行的 K/V 为 12 个 heads 分别搬运，并避免全局 attention 中间结果 A。不是先选复杂预处理或持久队列。

已保证与未保证的区别：

- 一行的前 2048 个 slots 是 512 个连续四元组，每组四个 token 连续，组号互异。可只读每个 quad 的首 token 并恢复后三个；这不是假设组号有序。
- 每个 token 的 D 维连续，相邻 token 的 stride 是 512 BF16 元素；K/V 共用 allocation，但 V 在同一页的 K 基址之后 65536 元素。不能按紧凑的 `[page,token,kv,d]` 默认 strides 寻址。
- `rho` 不需要显式得到；有效计数与实际 indices 足以执行。尾部按 count 处理。
- 所有实际选中 token 都小于 131072，所以 logical page 为 0–1023，physical page 可用 `1352+logical_page`。因此没有必要读 -1 页表项，也没有理由扫描未映射的 cache 存储。
- 没有跨行 quad 选择重合率、顺序、排序或 query 相似性的保证。相邻行共享缓存内容只是机会，不能当成保证收益。
- DOC 1 的默认 range 使 q/gate 为任意有限 BF16 值。我最初误以为未给 range 可能包含 NaN，查阅后已修正；有限不等于通常测试中的小数值。

## 2. 候选比较与当前选择

近似采用每行 2048 tokens 算量，不把 cache 命中当作已知。一次 K/V payload 为 `2048*256*2*2 = 2 MiB`，其中两个 2 分别是 BF16 字节数和 K/V 两份。

| 候选 | 减少的成本 | 新增成本/限制 | 当前判断 |
| --- | --- | --- | --- |
| 每 head 独立直接 attention | 工作简单、每个 worker 状态小 | 无片上跨 head 复用时约 192 GiB K/V 请求 payload；实际 DRAM 量依 cache 而定 | 作为直接参照，不选为主路线 |
| 每 `(row,kv_head)` 一份工作，12 heads 共享 gather，在线 attention + gate | K/V 请求 payload 约 16 GiB；省去全局 A 写读约 96 MiB | 一份工作持有 12 个 heads 的状态；online 重缩放；数值问题见后文 | 当前工作组织主路线，数值部分尚未完整成立 |
| 先把每行选中的 K/V 打包为稠密数组，再分组 attention | 后续加载规则化，可单独调整 gather 并行度 | pack 仍读约 16 GiB，再写约 16 GiB、attention 再读约 16 GiB；约 16 GiB scratch；多提交 | 当前分组已复用 12 heads，pack 没有额外复用可摊销这些流量，不优先 |
| 每次调用先重排全部映射 cache，再按 indices gather | compact 页布局，可能改善后续 gather/缓存访问 | 映射 cache 为 256 MiB，预处理最少增加 256 MiB 读 + 256 MiB 写；仍需按行 gather，增加 scratch 和提交 | 有意义，不能凭“预处理贵”直接排除；需要后续每 token 成本节省能覆盖该项 |
| split-KV，多 worker 处理同一 `(row,kv_head)` 后归并 | 单工作递推链变短、更多并行度 | 部分 m/l/U 的读写、归并、Q 重读与额外提交；主路线已有 8192 份工作 | 保留独立候选，当前没有证据表明需要更多工作数量 |
| 跨行排序/合并相同 selections | 若重合大可能进一步复用 K/V | 排序或哈希、重排和逆映射；没有保证的重合率 | 不作为当前方向依据 |

主路线 QK 与 PV 合计约 206 GFLOP（乘加算两次），不是减少了 attention 数学乘法。16 GiB 只是 gather payload 估算，不是对 L20D DRAM traffic 的测量。q、gate、out 各约 48 MiB；这些都计入调用。选择 index 也有处理成本：每个 KV worker 读 512 个 quad 首 ID、count 和至多三个 tail ID，可将两份 KV 工作合计的 selection 读请求压到约 16 MiB，且不用全局预处理。若实际实现读完整 index，则约 64 MiB，不能混用这两个口径。

完整映射 cache 重排的盈亏可写成 `T_pack + T_attention_packed < T_attention_direct`。缺硬件/微基准资料时，我无法把字节变化换算成可靠时间。其 page-stride 空洞本身不会产生等量 payload 传输，不应把省下虚拟地址跨度误算为带宽收益。

## 3. 数学路线与不能绕过的数值缺口

数学上先定义选中视图 `Ksel(r,v,j,d)`、`Vsel(r,v,j,d)`，地址按 Contract 解码，不物化整份视图。将 j 分成 B=32 的块 `i`；`N(r)=ceil_div(count(r),B)`，有效域仅包含真正选中的 slots。

最初拟写的 OnlineAttention：对每个 `(r,h)`，初态 `M=-inf, L=0, U[d]=0`。当前块 scores 为 Contract 的 dot/scale。令 `m_i=max(scores)`，`Mnew=max(M,m_i)`，`alpha=0` 在首块，否则 `exp(M-Mnew)`，`P_j=exp(score_j-Mnew)`。更新 `Lnew=alpha*L+sum(P_j)`、`Unew=alpha*U+sum(P_j*V_j)`，终点 `A=bf16(U/L)`，然后严格保留 `O=bf16(A/(1+exp(-gate)))`。M/L/U 的逻辑寿命是 `(r,h)` 内跨块携带；A 只在最终 gate 之前保留，不落全局。

初版不将 P 主动窄化到 BF16；QK/PV 的数值操作先按 FP32 解释，矩阵的数学形式不等于已选定 Tensor Core 原语。Tensor Core 的 P 窄化路径有独立误差决定，不能在 tile_pipeline 偷加。

这段 FP32 数学尚不能成为整个 Contract 的合法设计，已有两个具体反例类别：

1. q 和 mapped K 为大有限 BF16 时，点积可以溢出 FP32。softmax 稳定化发生在点积之后，不能修复已经得到的 infinity。原 Contract 默认 FP32 自身在这一类别如何与 tolerance/参考比较，需要明确，不能私自改成限制输入幅值。
2. 即使 q=0、K=0，使 scores=0，取全部 V 为最大有限 BF16，参考的归一化权重求和结果在数量级上仍为 V；上述 online U 却会累加约 `2048*V` 而溢出。这是我改写新引入的可分析风险，不只是“还没跑精度测试”。

第二项可以探索用归一化均值进行块合并，或采用更宽的 U，避免先形成巨大未归一化加权和；但会改变 PV 精度、额外除法或累加成本。FP64 dot/U 也不能未经分析就称为参考一致：既有 FP32 溢出结果与比较规则尚不清楚，且硬件支持/吞吐未绑定。这一轮没有选出完整数值修复，不伪装成可交付实现。

后面的 schedule/pipe 是在同一在线状态接口下的局部表达试写；工作覆盖、合法 gather 和 gate 舍入可独立推演，不表示上述数值缺口已经解决。

## 4. work_schedule 小段

选 B=32 是为了做具体容量估算，不声称最优。8192 个 CTA，一份工作负责一行的一个 KV head 对应的 12 个 query heads。不 split KV，64 或 65 轮，最后一轮只含 1–3 个 token；count=2048 时只有 64 轮。无动态队列、不假设 worktile 相邻就同时驻留。

下面是局部字段片段。Qgroup/Ksel/Vsel/Ggroup/Ogroup 是上文逻辑值的相应视图，正式稿需要补齐它们的 math symbol；本段没有用目录替代这些缺失定义。

```yaml
work_schedule:
  thesis: Reuse each gathered key and value across the twelve query heads of one row and fuse the final gate.
  step:
    MAIN:
      describe: Resolve selections and compute grouped attention followed by the required rounding and gate.
      decomposition:
        strategy: B is 32. The row and kv coordinates select heads h=12*kv+a for 0<=a<12. Iterate over N(row) selected-token blocks; only valid slots participate.
        partition:
          row: {meaning: Query row, size: Rows, tile: 1}
          kv: {meaning: Shared key and value head, size: KVHeads, tile: 1}
        worktile:
          invariant:
            Qgroup: {meaning: Queries for the twelve local heads, shape: [head: 12, d: D], dtype: bf16}
            Ggroup: {meaning: Gate logits for the output group, shape: [head: 12, d: D], dtype: bf16}
          iteration:
            Ksel: {meaning: Selected keys shared by local heads, shape: [selected: 32, d: D], dtype: bf16, direction: ascending}
            Vsel: {meaning: Selected values shared by local heads, shape: [selected: 32, d: D], dtype: bf16, direction: ascending}
        target:
          single_tile:
            Ogroup: {meaning: Final gated output group, shape: [head: 12, d: D], dtype: bf16}
      mapping:
        worker: {unit: CTA, unit_count: 1}
        worker_count: 8192
      schedule:
        worktile_order:
          - {by: row, direction: ascending}
          - {by: kv, direction: ascending}
        worker_assignment:
          preassigned: {start: worker_id, count: 1}
          refill: none
          exit: assigned_done
  flow:
    launch: [{step: [MAIN]}]
```

Ggroup 在工作中不变并不意味着进入 worktile 就立即加载；末尾加载更短暂地占用寄存器。第一次我把 invariant 与常驻存储混在一起，读 DOC 4 的“局部迭代中保持不变”后，倾向于将加载时点单独交给 pipe。

容量粗估：K/V 单槽合计 32 KiB SMEM；12×256 的 FP32 U 为 12 KiB 寄存器总值；Q 若存 BF16 为 6 KiB，若加宽常驻为 12 KiB；每块 12×32 FP32 scores/P 各 1.5 KiB；再有统计量、gate/输出、地址临时量与协作开销。不是把 CTA 总寄存器值当作每线程用量。初稿只有单槽，暂不声称与计算重叠。缺少设备容量和工具链资料，不能从这个账本直接推出驻留数。

完整映射 cache 重排候选若被选中，可局部写交接：

```yaml
flow:
  launch:
    - step: [PackCache]
    - step: [MAIN]
      start_after: PackCache.kernel_done
  handoff:
    Kpacked: {producer: PackCache, consumer: [MAIN], storage: global}
    Vpacked: {producer: PackCache, consumer: [MAIN], storage: global}
```

PackCache 必须覆盖 logical token 0–131071、两个 KV heads、全部 D，只读已映射页；不能拷贝整个物理 cache allocation 的任意 bits 并在后面做“乘零消毒”。Kpacked/Vpacked 在 math 定义成紧凑视图的保留值、由 PackCache target 定义生产域、MAIN iteration 指明消费域。这里尚未完整写出 PackCache 的分解和 pipeline，片段不是另一份完整 schema。`start_after` 使整个 PackCache kernel 完成后再提交 MAIN，无需另造 ready 事件；不能靠 launch 列表顺序推断屏障。

## 5. pipe / using_verb / waiting 的实际试写

先使用一个 MAIN role、四个 warps，warp w 计算本组 head `3*w+a`（`0<=a<3`），对应全局 head `12*kv+3*w+a`。全部四个 warps 合作搬 K/V 到同一 CTA 槽，计算阶段每 warp 只更新自己三个 heads 的统计量和 U。这里 matmul 表示数学矩阵运算，当前局部表达按普通 FP32 运算实现；尚未绑定异步 MMA。

若只写“一个 role 串行”仍不够：serial 约束同一成员实例的动作；四个 warp 之间共享槽，所以必须等待全部生产者/读者，或者绑定覆盖全部参与者的同步原语。此稿选择显式边，不用长 describe 暗含跨成员等待。

```yaml
index:
  i: {meaning: Selected-token block ordinal for this row, definition: [0, "N(row)"]}
role:
  MAIN:
    info:
      describe: Cooperatively gather keys and values; warp w computes local heads 3*w+a for 0<=a<3.
      member: warp[0:4]
      using_verb:
        load:
          Q:
            execution_mode: serial
            describe: Read the three heads owned by this warp from Qgroup into its query registers.
          KV:
            execution_mode: serial
            describe: Read this warp's disjoint portion of Ksel(i) and Vsel(i) into the shared K and V slots. Decode quad bases and valid tail indices from logical_indices. No source read is issued for invalid slots; write zero into their staging positions. Done makes this member's writes available to the declared CTA readers.
          Gate:
            execution_mode: serial
            describe: Read this warp's three heads from Ggroup into local gate registers.
        compute:
          Init:
            execution_mode: serial
            describe: Initialize this warp's OnlineAttention states without reading previous storage contents.
          Stats:
            execution_mode: serial
            describe: Consume this warp's Scores(i) and prior M and L to form P(i), alpha and the updated row statistics by OnlineAttention. Invalid slots contribute no exponential weight.
          Gate:
            execution_mode: serial
            describe: Consume final U and L and local Gate values to form A with the required BF16 rounding and then Ogroup by the contract gate formula.
        matmul:
          Scores:
            execution_mode: serial
            describe: Read this warp's Q registers and the shared K slot for block i and produce its three rows of Scores(i) by the selected FP32 dot path.
          Update:
            execution_mode: serial
            describe: Consume local P(i), alpha and prior U together with the shared V slot, completing OnlineAttention's U update for this warp's three heads. Done includes all of this member's reads from the V slot.
        store:
          O:
            execution_mode: serial
            describe: Write this warp's three heads of final Ogroup to out without reading its previous contents; done provides output visibility at the invocation boundary.
    pipe:
      - load:Q
      - compute:Init
      - for(i):
          - load:KV(i)
          - matmul:Scores(i)
          - compute:Stats(i)
          - matmul:Update(i)
      - load:Gate
      - compute:Gate
      - store:O
    waiting:
      prepared:
        - "{issue matmul:Scores(i)} ← [{done MAIN[*]::load:KV(i)}]"
      released:
        - "{issue load:KV(i+1)} ← [{done MAIN[*]::matmul:Update(i)}]"
```

`prepared` 保证 K/V 写完再读。`released` 等全部 Update，保守地同时保护 K/V 槽覆盖；Scores 的 K 读取先于该成员 Update，所以不需要再单列 K 释放。首轮没有上一轮槽内容；最后一轮没有下一次 load，移位规则按已声明 i 域截断。所有 N 均大于零。没有预取领先，所以这组等待没有首轮资源循环。

局部 `storage` 尚需登记 K/V 的 SMEM 单槽，以及 Q、U、Scores/P 寄存器区域。例子中的 noun 文本已说明源/目标和成员，但是数学值的正式 symbol、槽绑定和 CuTe 协作加载的保证没有全部落地；因此没有把这段当成合法完整 tile_pipeline。若后续使用异步 load 或 MMA，应逐一选择 execution_mode 并新增所需 completion 等待，不能因都在同一 warp 就写成共用 async_ordered 队列。

## 6. 分支应怎样保存

按 DOC 0 的规则，我当前会这样判断：

- B=32 改成 B=64、单槽改双槽：若主角色与数学接口不变，目录参数变化，写清容量和依赖距离联动，不新建完整 schema。
- 只把四元组解码方式改成每 slot 完整读取：局部实现/局部变化，取决于是否改变已选的访存成本决定，不是独立算法。
- 映射 cache 的全局 repack：目前倾向目录“局部”变化，因为只增加既定 cache 视图生产者、提交/物化与 MAIN 的 source 绑定，主要 attention 工作归属、递推与 gate 不变。但增加全局预处理是否已算“主要计算重贯通”，我拿不太准，见 M03-04。
- split-KV + 跨 kernel merge：独立 schema。必须重新定义部分状态、最终归一化/gate 的输出生产者及跨工作归并；不能只写 split_factor 参数。
- Tensor Core 的 QK/PV 路线：仅选择等价的 QK 原语可能是实现选择；若把 P 改成 BF16，须回 math 记录舍入/误差。若仍是同一 CTA 的局部动作替换，可目录记录；若引入专职 loader/softmax/MMA、TMEM、整条主循环交接重构，我会保留为独立候选。尚未把其硬件表达能力当成已验证事实。
- 持久 CTA 跨 worktile 预取/保留状态：若真的重构主领取、退出及主循环，不用“多一个队列”压成参数；它可能是独立 schema。本任务工作量很均衡，暂未投入这条方向。

本次停在工作组织主路线、候选边界和局部动作协议。没有完整数值保证、资源驻留/原语绑定、合法全量 YAML、性能验收，也没有以缩小 Contract 域来遮掩缺口。
