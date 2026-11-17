# GDN 选型

固定来源为 `KernelSchema@3d64ad35900a0e6a66d6f93da34108fbceba0daf:kernel_wiki/aka-four-problems-20261019/gdn_prefill/aka_solution`。`solution.py:Model.forward` 强制 `value_rows_override=64`；`build_solution.py` 说明单文件由 scheduler、kernel、adapter 拼接再压缩得到。

只保留 `root.yaml`：B>1、每 CTA 64 value 行、64 token chunk、成对 inverse 准备、双 MMA issuer、片上 FP32 状态递推和同 chunk 输出。以下按 DOC0 局部闭合测试降级，删除旧 `m64_single_sequence.yaml/png`：

- B=1：工作仍是 `(sequence, head, value-half)`，仍使用同一 inverse、KS/QS/U/Y/S 主计算和 Recurrence 输出生产者；改变静态 descriptor、ScalarOutput 的双 raw TMA 槽和五槽 ready-only 交接，以及 Recurrence→MMA1 的三次单 barrier 通知。这些变化的入口、发布值及消费者仍对应 root 的已有对象，可完整记录局部协议而不重写整体计算。
- M64/M128 指 value/state 行数，token chunk 都是 64。当前公开 Model 没有 M128 dispatch。可读内部 adapter 的 override=128 保留原主机制，联动工作数、K 槽深、状态转换/搬运形态，记录参数变化与适用边界；不把历史 forced-M128 测量当本候选运行路径。
- B=1 时 `T%128==0` 仅关闭尾块标量 predicate；`T%64==0` 不足以关闭，奇数真实 chunk 仍有一个全 padding chunk。zero/continuation 都是显式 FP32 initial_state；不能采用内部 None 分支的首块省略。

root 的 Contract 保持 B=1..16、总 T≤16384 明示域，root Design 的 B>1 限制不缩小合同。旧两份稿只用于线索，核心源码和 ABI 已重新核对。
