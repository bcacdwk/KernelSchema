template_fa4_decode 已核对 7 节点、19 个精确 pipe 组、48 个角色上下文。当前模板与冻结输入同 hash；规则依据归档的 `5.design.tile_pipeline.md`。A_decl（源声明）、B_local（本地展开）、C_effective（整区继承后）均为 48 roles／96 blocks／48 nested／282 actions；7 节点均本地提供 pipeline，无继承增量。以下逐节点列 B，root 单列；动作数均为静态文本数。

| 节点 | roles | blocks | nested | actions | 当前工作组织 |
| --- | ---: | ---: | ---: | ---: | --- |
| root | 7 | 17 | 10 | 48 | q4，两查询一 tile，partial＋combine |
| q1_global | 8 | 18 | 10 | 52 | q1，一查询一 tile，partial＋combine |
| q4_compact | 7 | 17 | 10 | 48 | q4，名义 split 与活跃 split 分离 |
| q1_cluster | 7 | 16 | 9 | 45 | q1，cluster 成员各占一 split |
| q4_cluster | 6 | 15 | 9 | 41 | q4，cluster 成员各占一 split |
| q1_direct | 7 | 7 | 0 | 24 | 单 KV 块、单 CTA |
| q4_direct | 6 | 6 | 0 | 24 | 单 KV 块、单 CTA |

96 块中 48 个嵌套块全是 `repeat+steady`，最深 3 层；70 块带 repeat，其中 67 次引用域名、3 次为 combine 局部 r 域。10 个列表含相邻循环，共 15 对；15 个列表混合动作与循环。19 个精确组的逐组证据、全部成员契约和上下文见 [review JSON](../reviews/template_fa4_decode.json)。指纹不含域、契约与调度，18 个形状组也不表示语义等价。

多块节点的 i 是一个 split 的 256-key 块序号，访问逻辑块 `r+i*WorkSplit(b)`；每个 worker 预分配一份工作，不是持久 worktile 循环。q1 的两个 Softmax 分别按 `2*j`、`2*k+1` 处理偶、奇块；q4 一个 Softmax 按 i 前进。两组 q1 的最大值仍经 Correction 交替握手，分组不表示递推独立。[root:102](../../../schema_template/examples/fa4_decode/root.yaml#L102)、[q1_global:320](../../../schema_template/examples/fa4_decode/q1_global.yaml#L320)。

一个块还包含两半键区间、两段归约维度的搬运与乘积，以及主概率分量和残差分量；这些工作封装在动作契约里，并没有逐项成为嵌套块。最后一块的因果与长度掩码在分数读入后应用，块域存在不意味着全部填充位置有效。查询打包宽度也随节点改变：单查询路径是十六列，两查询路径是三十二列；同一动作名对应的逻辑值容量不同。

Correction 的 head 先读最多两个最大值，corr_prefix 处理除最后一次外的双奇偶校正；corr_final 将末次读数与校正分置 FinalMaximum 两侧，使最大值先交出；两段 tail 分别读最后至多两块的和、捕获两个 Y 槽。`+2` 对应同奇偶前驱，`+4` 对应 P 环；它们是值版本及复用距离。初始化和收尾既在 prologue/epilogue，也在 steady 的固定动作及子块中，不能只按区名识别阶段。[root:408](../../../schema_template/examples/fa4_decode/root.yaml#L408)。

active 的 0/1 次重复表示有数据时执行一次初始化及 Q 装入，active=0 不是 Y 奇偶槽。N_r=0 时块循环及 Y 初始化不存在，统计初始化、FinalMaximum/FinalSum、Ostage 和输出动作仍在；后两者按契约无 payload。N_r=1 还需信号及排空初始化的空奇偶槽。direct 虽无 repeat，仍有四切片读数、异步发射、空奇偶排空、FinalSum 会合及 TMEM 归还；q1_direct 的 SoftmaxOdd 只发 EmptySum。[q1_direct:177](../../../schema_template/examples/fa4_decode/q1_direct.yaml#L177)。

K/V 与 Score/Value 各有独立轮转放行，允许 `K0..KLead,V0,…`；放行发生在最终操作数获取后，传输或结果仍可在途。data 控制可读，reuse 保护 Score、P、ExponentialSum 及 KV→Ostage 覆盖，sync 控制轮转和会合；加上角色执行模式才构成顺序。P 读完、Y 结果就绪、尾部捕获和 TMEM 释放是不同边界。[root:324](../../../schema_template/examples/fa4_decode/root.yaml#L324)。

发射角色整体采用异步推进，但启动、放行、会合和资源释放等动作有单独的串行完成契约。计算角色中的相邻动作则依其串行模式推进。因此同样一行列表可能约束发射顺序，也可能要求前项完成后才进入后项；列表数量本身无法区分这两类关系。完成尾部捕获以后，分子已经保留在寄存器中，后续输出归一化与暂存不再要求原来的张量存储继续存在。

cluster 在 Correction 中由成员 0 清零输出，Score 收尾交换最大值和分母，各活跃成员发布 BF16 贡献并 reduce-add；空成员仍参与统计。global 的 Score 则写 FP32 统计、发 PDL 放行，combine 先等前驱完成，再读统计、置零分子和计算权重；r 循环按 split 分段读分子并归约，最后归一化。root/q1_global 读取所有名义槽，包括零权重的有限旧 scratch；同一 pipe 组的 compact 在 payload 读取前跳过空槽，并逐线程重算权重、更新分母。其 N_r 又受 WorkSplit(b) 影响，名义域长度不能当有效读量。[root:434](../../../schema_template/examples/fa4_decode/root.yaml#L434)、[compact:353](../../../schema_template/examples/fa4_decode/q4_compact.yaml#L353)。

现有未决点：5 个多块节点的 LoadK 契约明确记录 Lead=2、i+4<N_r 时 PageIndex 旧 V 捕获先于同槽预取覆盖尚未证明；2 个 cluster 的成员 0 清零→输出累加等待只在动作契约描述，pipe/waits_for 未显式列出该跨成员前驱。以上是文本现状，未运行 GPU、展开事件或做消融，也未确认实际竞争发生率。[root:294](../../../schema_template/examples/fa4_decode/root.yaml#L294)、[q1_cluster:217](../../../schema_template/examples/fa4_decode/q1_cluster.yaml#L217)。
