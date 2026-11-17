# FA4 decode 实施与终审交接

最终四份设计为 root Q4/S9/global、q4_cluster Q4/S16、q4_direct Q4/S1、q1_global Q1/S16；root 是唯一 Contract，q1_global 整区共享参数化数学，其余三区本地。compact 请求分片及配套 serial combine 闭合在局部变化；Q1 cluster/direct 是源码已有组合，均保留在 dispatch。具体独立结论、修正与命令见 [REVIEW.md](REVIEW.md)。

固定来源为 SOURCES 的 `3d64ad35900a0e6a66d6f93da34108fbceba0daf` 四题快照 v55。公开 Contract 为 B1..17、Q{1,4}、Q≤L_b≤131072，完整 None ABI、out alias 与实际 atol/rtol 均保留。L20D/SM103 和匹配的软件版本有既有证据；Wiki 仍为规范占位。本轮没有 GPU/Runner 或 Git commit。

每份 when 固定一套实际模式，并限定 native launcher 已建立、同对象/签名/线程和当前设备、非 capture 的稳态命中。首次调用正常执行后仍可能 clone out、重复 native 调用、同步比较、失败恢复；dispatch 已记录这个前缀，不把 probe 阈值当 Contract tolerance。

独立终审修正了最大值发布与捕获的次序：SOFTMAX.compute:maximum 发布 rendezvous，新增 load:maximum 随后捕获；Q1 mutex 在捕获后交接。SCORE.sum_ready 改用 await；global 分母归约/写回改用 compute；direct 仅等待 final-max rendezvous，不伪造 disabled LSE/sink 路径中的最大值读取。cluster 的 vslot 公式使用本成员 count(i)。版本为 root11、其余9。

两个源边界保留：Page4 预取在等待 V0 turn 之前，没有确认 V0 capture 必先发生；普通 global combine 的零权仍读取空 split 未覆盖的 O，跨 workspace 布局重用不保证这些位有限。旧 SELECTION 的 PageIndex 容量证明已撤回，详见 [pageindex_order.md](pageindex_order.md)。有限图无环不证明这些地址访问安全，也不证明数值全域正确。

成员工具已修复并冻结，前轮 [MEMBER_GRAPH.md](../MEMBER_GRAPH.md) 记录135项工具测试通过；本 reviewer 未改公共工具、未重复其全套测试。最终 cluster T_b70 按真实5/4轮展开并通过，T_b1 的1活跃/15空成员也通过。所有16成员均保留单点 Mapping.Member，未复制角色、统一假 N 或改变 splitcount。

当前四份同名 PNG 均成功重绘：root N5/S9、q1_global N5/S16、q4_direct N1/S1、q4_cluster T_b16/S16，worktiles 均1。cluster PNG 对应 when 内 B1/Q4/L4096/P64、完整16成员各一轮；原 T_b1 的跨 when 图已被替换。T_b70/T_b17 的历史 PNG 超时仍留日志，不计为成功，也不再追求不必要的大图。

默认检查、最终只读和各份代表/边界图检查均通过，硬件未跳过。最终 PNG 来源摘要与当前 YAML/共享 root 已逐项核对；数字与日志索引见 REVIEW。早期 cluster_graph.log 的 split 未绑定、旧 root_graph/q1_graph 的大图超时以及各 member_* 日志为历史记录，不能覆盖这里的最终状态。
