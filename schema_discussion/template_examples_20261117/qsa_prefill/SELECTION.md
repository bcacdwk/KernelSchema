# QSA 选型

保留一份 `root.yaml`：BF16 FIT v3 的完整预处理与持久 attention 方案。固定来源为 `KernelSchema@3d64ad35900a0e6a66d6f93da34108fbceba0daf:kernel_wiki/aka-four-problems-20261019/qsa_prefill/aka_solution`，包版本 `fc30f74`；不采用旁边 README 中的 v7。

host 唯一实际提交是 union 预处理和主 kernel。评分域固定 group=12，br=10；主 kernel 使用 M128/N64、两槽 KV、三槽 S/P、两个列半 softmax 组、动态 ticket 分配。必要的完整知识是 union 掩码共享、两级地址预取、跨工作延迟发布、TMEM S/P 别名、稀疏修正、Q refill 与输出 drain 重叠及 ticket 生命周期；无需另一套完整 YAML。

构造器的 `s_stages=2` 会把 QK 领先距离从 2 改为 1，是明确局部配套；host 没有选择它。其他组大小的 br fallback 没有扩展评分域，且主 kernel 将工作固定解码为两 KV 头；不能把 fallback 写成已支持的完整方案。混合 group quad 的逐 lane hash fallback 是同一预处理的内部条件，不另立设计。`use_prefill_config` 被忽略。

源限制：Q padding 不写；HalfMax 仅有写后 rendezvous，未发现 peer 读完后再覆盖的保证；缓存 workspace/ticket 没有跨 stream 隔离或重放时 host base 更新。保留这些事实，root 限定串行 eager 调用，不新增修复协议。
