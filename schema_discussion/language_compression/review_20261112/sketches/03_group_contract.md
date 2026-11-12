# 03 组内分工：局部表达对照

这些是**现行字段与语法的局部研究草图，不是新语法提案，也不是可直接替换的完整 Plan**。
未迁移旧样本的全部 using_verb/storage/index，未运行完整 schema 检查或 GPU；计数只比较明确列出的 width-8 局部。
固定资料见 [数据](../data/03_group_contract.json)；两套写法必须共享相同数学、存储、外部入口/出口，不能以删接口换取更短文字。

## 1. 整个 Inverse 保留一个短契约

现有事实：一个 CTA，Inverse 使用 `warp[0:4]`，register=224；A/Tinv 共用 Tring，A 的两块均形成后才进入 inverse；Tinv 与 Bmat 的外部发布和原 reuse 边保留。
`c0=n[t,2*p]`、`c1=n[t,2*p+1]` 只是本段缩写，均指上游同一 worktile 中的 chunk，不成为新的可独立分配工作。
相应 noun.describe 可写为：

> Invert A(c0) and A(c1) in Tring per inverse. At widths 8/32/64, warp[0:2] owns c0 and warp[2:4] owns c1. At width 16, warp r (0..3) handles diagonal block r of c0, then of c1, independently of the other warps. All four rendezvous before width 8 and after each level; retain the merge primitive's internal completion and visibility guarantees.

这里 `r` 是说明中的 warp 编号，**不能**进入 `info.member` 随 pipe 变化；所有四 warp 一直参与整体动作。
未再抄 BF16 舍入公式、8×8 消元、shuffle lane 和 MMA fragment 布局；它们分别由上游数学与原语实现承担。
这段比旧 invert 的 156 字符更明确，并不是该旧句的节字候选。
因为外部只使用后续 Tinv/Bmat 发布，width 8/16/32/64 暂无新的外部部分完成需求；内部层次与局部顺序可由契约保留。

## 2. 只比较 width 8：一个动作与两个固定 role

共同上下文使用当前 `index` 写法，引用同一组既有参数：

```yaml
index:
  t: {meaning: "Local worktile sequence.", definition: worktile}
  c: {meaning: "Chunk within the current worktile.", definition: [0, Np]}
  p: {meaning: "Pair of consecutive chunks.", definition: [0, Np/2]}
  n: {meaning: "Continuous chunk ordinal.", definition: {flatten: [t, c]}}
```

共同入口是已定义的 `Inverse::compute:A(n[t,2*p+1])` done；其 serial 前驱形成第一个 A。
共同出口是下一层已定义矩阵计算 `Inverse::matmul:Merge16(t,p)` 的 issue。
这里给入口/出口采用当前基础动词的名称，只用于比较 width-8 边界；不宣称迁移了 Inverse 的其余全部动作。

保留在 Inverse 内时，为该局部建立一条 noun 声明：

```yaml
Pair8:
  execution_mode: serial
  describe: "Invert all eight diagonal 8x8 blocks of each A in Tring per inverse, using FP32 elimination and BF16 writeback; warp[0:2] handles n[t,2*p] and warp[2:4] handles n[t,2*p+1]. All four rendezvous before and after this level."
```

这段放在 `Inverse.info.using_verb.compute`，相应 pipe 在两次 A 形成之后、Merge16 之前增加 `compute:Pair8(t,p)`。
计数为 1 个 noun 条目、1 个调用，0 个新 role、0 条新跨 role waiting；若保留整个 inverse 原语，连这个内层切点也无需登记。

静态拆分时，删除该 Pair8 条目/调用，增加下列两个 role，保留既有 Inverse 的其余动作：

```yaml
Pair0:
  info:
    describe: "Perform width-8 inversion for the first chunk in each pair."
    member: "warp[0:2]"
    register: 224
    using_verb:
      compute:
        Diag8:
          execution_mode: serial
          describe: "Read the first chunk A from Inverse's completed A action in Tring, invert its eight diagonal 8x8 blocks per inverse, and write BF16 results in place for Merge16."
  pipe:
    - "for(t)":
        - "for(p)":
            - "compute:Diag8(n[t,2*p])"
  waiting:
    prepared:
      - "{issue compute:Diag8(n[t,2*p])} ← [{done Inverse::compute:A(n[t,2*p+1])}]"
Pair1:
  info:
    describe: "Perform width-8 inversion for the second chunk in each pair."
    member: "warp[2:4]"
    register: 224
    using_verb:
      compute:
        Diag8:
          execution_mode: serial
          describe: "Read the second chunk A from Inverse's completed A action in Tring, invert its eight diagonal 8x8 blocks per inverse, and write BF16 results in place for Merge16."
  pipe:
    - "for(t)":
        - "for(p)":
            - "compute:Diag8(n[t,2*p+1])"
  waiting:
    prepared:
      - "{issue compute:Diag8(n[t,2*p+1])} ← [{done Inverse::compute:A(n[t,2*p+1])}]"
```

同时在 Inverse 的 `waiting.prepared` 增加一条出口规则：

```yaml
- "{issue matmul:Merge16(t,p)} ← [{done Pair0::compute:Diag8(n[t,2*p])}, {done Pair1::compute:Diag8(n[t,2*p+1])}]"
```

入口保证两个 A 均已形成；出口等待两个 chunk 的 width-8 结果，以源码同一次四 warp 汇合兑现跨 warp 可见性。
共计 2 个新增 role、2 个 noun/pipe 调用、2 条入口规则和 1 条含 2 前驱的出口规则；上游数学/存储、外部数据和复用边不变。
物理成员并集仍是 warp[0:4]，不是 8 个 warp；Inverse 与 Pair0/Pair1 在这段中分时执行，入口/出口关系必须落成可实现的代码。
两个计算 role 之间没有 chunk0→chunk1 的边；也没有允许 width16 提前越过另一个 chunk 的 width8。
它增加了可寻址进度而没有证明新的有用重叠；若未来需要部分消费者，可再检验是否值得付这份表达成本。
若全层静态展开，width16 必须由四个固定 warp 的各自顺序表达，不能再拼成全组 chunk0→全组 chunk1 的 serial 队列。

## 3. QSA 条件等待必须可寻址

下列仅表示某个静态 A-warp 的相关 pipe 和边，前提是按现行规则另行完整登记 noun 输入、寄存器副本、存储区域和整个生命周期；**不能原样塞回旧 SoftmaxA**。
例如固定 `member: warp[0:1]` 的 SoftmaxA0，负责 packed rows [0,32)；Vote(n) 是该 warp 的既有 move_ref 结果，并由 Weight0(n) 动作产生且保留。
Weight0 已完成概率与 vote 计算；Rescale0 只在 vote 时改这 32 行的全 D 列 Y；随后纯写入 P_A0 的 TMEM 切片。

```yaml
pipe:
  - "compute:Weight0(n)"
  - "if(i > 0 and Vote(n))":
      - "await:PriorPV(n)"
  - "if(Vote(n))":
      - "compute:Rescale0(n)"
  - "store:P_A0(n)"
waiting:
  prepared:
    - "{done await:PriorPV(n)} ← [{done MMA::matmul:P(n-1)@Vblk(n-1)}]"
```

`await` 与本 role 的计算采用 serial；所有控制名、i/n 绑定与输入就绪必须由外层循环和动作声明提供。这里不是新增隐式变量机制。
PriorPV 的 describe 应明确它等待现有 pvdone 通知，为后续 Y 的读改写提供完成保证；不能只是把前驱字符串写入说明。
其他三组复制真实静态成员与行范围，不能靠 `SoftmaxA[r]` 指代 warp。MMA 对本轮 P 需等四片 A 与 B 半区的必要发布。
还须重接 HalfMax rendezvous、11-warp ack、初始/下一 Q、末轮 drain 等原边；所需寄存器副本跨新动作存活时以现有 storage 登记。
源码 HalfMax 读退役与 padding vote 的未证明事实仍保留；本局部没有用新 barrier 修复它们。
普通不触发的 warp 不经过 PriorPV，也不经 Rescale0；无须把它们拖入别的 warp 的等待。

## 4. final-mini 别名与安全短写的核源算式

提取 GDN 的 `stKQ` 与 final `stg` 均以 stages 为基址；末 mini 的 stage 选择为 0 时：

- warp1、gid0、half0 的 Kc row0 key=16，源偏移 `((16>>6)*8192)+(((16&63)>>3)*16)+((16&7)*2)=32` 字节。
- warp0、gid0、half0、nt1、tig0 的 FinalState key0/value8，写偏移 `4*(0*(64+1)+8)=32` 字节。
- DU0 的 CTA join 在 K 读取之前；随后 mini `continue` 退出最后一轮，final stage 写入之前没有下一轮 CTA join；2839 的 join 在写入之后。

这些事实足以表明不能声称这段动作已有跨 warp 读退役保证；没有据静态地址重叠宣布每次运行都会错误。

可安全压缩的是另一个、位于 staging 发布之后的 final_store 描述：

> Per thread, stream VTT/8 float4 blocks with one-block lookahead, skipping lookahead on the last round.

它保留初始预取后每轮“下一块加载→本块存储”的局部重叠、每线程工作量和末轮无下一读；删掉末轮未使用 `cur` 赋值的解释。
原条件、转置、无后续工作与 kernel completion 退役没有变化；这项压缩不能用于删除前面的 final-mini 来源限制。
