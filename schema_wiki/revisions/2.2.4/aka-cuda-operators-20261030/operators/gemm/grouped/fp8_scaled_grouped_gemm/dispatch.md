# 路径索引

| 节点 | 直接父 | workload | 相对父差异 | 选择依据 |
| --- | --- | --- | --- | --- |
| [root](root.yaml) | 无 | UMMA 默认路径：未被宽列门槛选中的分组 GEMM。 | 基准：128×128 输出块、成对 K scale-block 的持久 UMMA；dual 作为节点内参数。 | Model.forward 保留 UMMA engine，fp8umma_run 的 sched_policy 实际返回 engine=0；正常构建调用 umma_gemm<0>。 |
| [mma_sync](mma_sync.yaml) | root | 初始化未采用 UMMA，且 mma.sync 模块构建/加载成功的同一契约工作负载。 | 保留持久完整输出块与本地 K 归约，改为 64×128、三槽 cp.async、8 warp 协作的 mma.sync 及寄存器 partial；BF16 epilogue 复用 operand SMEM 后向量写回。 | 初始化查询 CUDA device 0；major≠10 或 UMMA 构建/加载抛异常时尝试此模块。保留 engine='sync' 后，Model.forward→fp8gg_run→fp8gg_bm64；不是 UMMA forward 运行报错后的回退。 |
| [umma_wide](umma_wide.yaml) | root | 命中 sched_policy 宽列门槛的 UMMA 工作负载，含 w1f/w1g。 | 保留持久 TMA/TMEM 组织，改为 128×256、每轮一个 K scale-block、行×列组 epilogue 和寄存器 scale 预取。w1f 的四槽环、32KiB staging、两次半块提交，与 w1g 的三槽环、64KiB staging、一次整块提交，作为同节点参数。 | 保留 UMMA engine 且 sched_policy 实际返回 engine=1 或 2。门槛使用 kb_count=k/128、非空组数 ne、mtiles=Σceil_div(组行数,128)、items256=mtiles*(n/256) 与保存的 nSM；完整门槛见 sched_policy 及 root.yaml 的 meta_data.dispatch.when。engine=2 遇 g_w1g_off 或 w1g 直接 launch 即时错误转 w1f，仍属本节点。 |

## 覆盖缺口

parent 仅表示设计复制来源；各节点条件在 root 的完整共享 Contract 域内独立求值，不叠加父节点的 when。精确条件和版本以各 YAML 为准；表中派生文件为计划内链接。

正常构建的计算入口均已列入。V0 的 dual、A-box 与 B 缓存策略，以及 C memo、普通 binding、ctypes、UMMA graph/直接 launch，留在所属节点内；诊断强制宏不属于本次正常构建。C binding 失败保留 UMMA ctypes 路径。

选择使用参数 memo 后实际传入的地址和尺寸。弱引用身份命中不重验 tensor 元数据；sched_policy 的单槽键为 (indptr,G,m,n,k)，命中不重读 indptr 内容或检查 nSM。不能按当前内容重算命中决策，也不能补入元数据不变保证。

路径覆盖不保证执行成功。两个 host 入口均拒绝 N/K 非 128 倍数，UMMA 另拒绝 G>116，mma.sync 拒绝 G>4096；当前共享 Contract 的维度不触发这些限制，不能外推未对齐 fallback。UMMA 首次配置检查所有 UMMA 实例的 SMEM 容量；tensor-map、资源配置、模块构建/加载或 launch 仍可能失败。除已列的 w1g→w1f 分支外，UMMA 运行错误不转入 mma.sync，forward 对非零返回报错；graph 与直接 launch 也不构成执行成功或额外完成等待的保证。

共享 Contract 的 tolerance 原样沿用，继续携带 P1 的外部验收 source_gap；缺失的比较公式、相对基准、比较精度、非有限值处理和返回别名/可写 out 检查路由不用于设计保证，不阻塞本次路径提取。
