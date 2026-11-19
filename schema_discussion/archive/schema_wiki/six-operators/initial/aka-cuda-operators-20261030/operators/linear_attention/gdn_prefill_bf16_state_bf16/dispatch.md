# 路径索引

| 节点 | 直接父 | workload | 相对父差异 | 选择依据 |
| --- | --- | --- | --- | --- |
| [root](root.yaml) | 无 | host 未选分段的 packed GDN 请求；VTT=16/32/64。 | 设计起点：64-token 系数预处理后，每个请求、值头和值 tile 用一次 MODE=0 遍历完整状态链并写输出。 | Model.forward → gdn_forward 最终 Gseg==1（gdn_kernel.cu:3163、3220–3250、3462–3479）；已审 root.when 为分段选择的补集。 |
| [segmented](segmented.yaml) | root | host 按元数据估计选中的低占用长链；Gseg=2..8，VTT=16 或 64，单个请求仍可能有空段。 | 保留 root 的 chunk 系数与状态更新；改为 MODE=1 按段传播并物化 FP32 段末态/衰减，再由 MODE=2 合成段输入、重跑并写输出。第一遍不做读出；弱衰减时第二遍仅首段重跑完整请求。prep 改为 slot-fastest；KQTMA、PRE0 均关闭。 | 同一 host 的 vt16 门内以 T、N、HV、运行时 SM 算 avg_chunks、Gc/Gw 并选择最终 Gseg>1（gdn_kernel.cu:3220–3250、3450–3460）；宽/窄 tile 和第二遍弱衰减回退均留在本节点。 |

## 覆盖缺口

直接父仅表示设计复制来源；各节点条件在共享 Contract 全域独立求值，不叠加父条件。精确选择条件和版本以各 YAML 为准，segmented.yaml 由 P5 生成。avg_chunks 是 T、N 的估计，不是逐请求长度保证。

VTT、流水深度、头共享、请求排序及可选归一化/输出留在节点内部；root 的直接 Q/K TMA 加载和首块预取也不另拆节点。分段路径沿用同一模板，以两遍执行、段状态物化和跨段合成为必要派生差异。

- 两节点覆盖 host 的 Gseg 路由，不表示实现已覆盖 Contract 的所有合法调用。仅 cu_seqlens=None 时 Python 补齐全部 chunk 元数据；边界已提供而 chunk_indices/chunk_offsets 缺省时，C++ 张量入口没有补全路径。
- core_attn_out 非连续时 host 写入 contiguous() 结果且不回拷，不能保证原调用者缓冲的副作用。静态 workspace/归一化缓存只依靠同流顺序复用，源码没有跨流重入同步。TMA 描述符编码失败直接报错，没有备用算法。
- 保留已审 root 的末态限制：VTT=64、末尾单 token chunk 位于 stage 0 且请求 FinalState 时，stage-0 写入前缺少跨 warp 的 K 读取结束保证。segmented 的同一模板后缀须按段内 stage 编号保留这一事实，不补造等待。
- 分段输入合成按源码的标量衰减近似；amax>1e-3 时虽回退到完整链，仍不形成共享 Contract 全域的误差保证。外部验收 source_gap 沿用 root 的 tolerance：已知 atol=0.01、rtol=0.05、max_mismatch_fraction=0.0；误差公式、相对基准、比较精度、非有限值策略及 CoreWritten 检查路由未提供，不作默认解释。本缺口不阻塞路径定稿。
