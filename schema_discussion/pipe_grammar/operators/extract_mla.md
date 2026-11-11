# extract_mla pipe 现状

核对冻结 root、bootstrap 的全部 9 组、9 个角色位置及两个 main 上下文。A/B/C 相等：16 块、7 嵌套块、52 静态动作，最大深度 2；root 为 5 角色、9 块、30 动作，bootstrap 为 4、7、22。无 pipeline 继承；bootstrap 的 math_reshape 整区引用 root。精确指纹不包含域、契约、数学和调度。

| 角色 | root 组后缀／动作数 | bootstrap 组后缀／动作数 |
| --- | --- | --- |
| Softmax | 4b3c65803b2384bf／12 | 2d50c3cd0636fbb1／10 |
| Correction | a32a21825679dd9e／6 | a58b91d400a9b5d3／3 |
| MMA | eeda5f116cfbe0c1／7 | 2ef98021137320f4／6 |
| Loader | 2c8e7787bacc82c1／4 | 9ca5d516f2b5cccd／3 |
| Queue | 61f24e94d3c10737／1 | 无 |

root 五角色均以 `repeat: w` 包围本份工作。`w` 是 worker 的工作序号；`t=[0,N(w))`、`n={flatten:[w,t]}` 将不等长局部遍历连续编号，连接 K/V、S/P 和 Alpha 环。bootstrap 每 CTA 只做一份工作，域为 `t=[0,N)`；Loader 顶层 repeat t，另外三角色顶层无 repeat。7 个子块全是 repeat+steady，5 张列表混合动作与循环，无相邻循环。flatten、symbolic_bounds、worktile 标志分别为 4、8、9，是重叠线索，不是执行次数。

数学中的一份工作对应一个请求、一个头和一个查询块，因果掩码决定可见键范围，进而决定本份键块数；输出不做跨工作的合并。同一个工作域不表示五角色锁步推进。领取元数据、准备输入与归还输出状态分别按各自依赖前进，终端描述符结束持续工作序列，不能把它算成一次零数据工作。

两份 Softmax 都先 `latch:Item` 重置 M/D，再遍历非末项；末项单列，使 `produce:Final` 在末次 Weights 后、P 发布前发生。root 在末次 `load:Scores` 前插入 `request:Next(w)`，只通知服务器，不等领取结果；末尾 `wait:Next(w)` 才消费发布。bootstrap 用 prologue 初始化；root 的首尾动作都位于 w.steady，每份工作执行。

MMA 先首 QK，再按“本项 QK、前项 PV”推进，最后补末项 PV。Correction 只遍历非首项，首 PV 直接初始化 Y；Loader 遍历全部 KV。root 分别使用 `n[w,0:N(w)-1]`、两处 `n[w,1:]`、`n[w]`；bootstrap 对应局部 `[0,N-1)`、两处 `[1,N)` 和共享 t。N 为正；N=1 时内层非首／非末循环为空，显列首尾仍存在。`n[w,t+1]` 不跨工作，`n+3`、`n+KVSlots` 的复用边可以跨工作。

Queue 独占一个角色：等 Softmax 请求后执行原子领取、复制描述符并发布 `Next(w)`。`Item(w)` 在 w%2 槽，Next 在 (w+1)%2；`publish:Next(w+1)` 等四个消费者 `latch:Item(w)`，保护回绕覆盖。失败领取仍发布负 tile-count 描述符并有完成事件，但不产生下一份工作。策略另说明总计 WorkCount 次 pop、末次计数复位及终止后读写完成边界。

Loader 先发 Q 后 latch，首 Q 可与 CTA 初始化重叠，后续 Q 使用前次 Next。`load:QTile(w+1)` 等本份末 QK，允许与此前 PV、输出接续。root Correction 依次 `store:OutputPrefix(w)`、`release:OutputState(w)`、`store:OutputTail(w)`：release 已读出末 32 列，归还共享状态，尾部转换与写回随后完成；下一份首 PV 和 Final 等该 release。bootstrap 的 `store:Output` 在 epilogue 完成全部输出，没有后继工作消费信用。

Softmax/Correction 串行，MMA 为 async_ordered_completion，Loader 为 async；MMA/Loader 的串行 latch、wait 只约束自身前驱，并不自动排空在途操作。默认 P 就绪边、显式 score/rescale/复用边与有序 tensor 契约共同连接数据。tiny 仍执行主 P/PV，scale 全一时 rescale 仍完成；动作内部的分段传输、四组 PV、输出分块和 tiny 归约未计作 pipe 循环。

资源边界同时依赖块外定义：查询与输出状态各一个共享槽，分数和概率共用三槽，键值环按局部性条件取三槽或四槽，缩放系数使用两槽。缓冲深度限制生产者领先，角色数量和静态动作数量本身不表示并发规模。首次工作没有旧读者；跨工作却要等待真正的读取结束，输出已经释放也不等于目标写回已经完成。

两份 schedule 共用 host 复制元数据、等待、建表排序、上传及 main stream 等待；这些文字解释启动边界。dispatch 以 `2*WorkCount` 与 `3*LaunchSMCount` 分流：root 提交 LaunchSMCount 个 CTA 并 refill，bootstrap 提交 WorkCount 个且无 refill。提交数不承诺驻留；外部记录保留选择绑定，表中实际数据仍参与执行。

现有问题来自冻结文本的声明：两份均保留源码 TMEM 行地址差异，root 另有过早描述符复用差异，同时 YAML 保留完整行分工和两槽寿命约束；这里只能确认声明，未复查实现触发条件。近似计算缺少匹配误差界，硬件规格多项为空；无法据此确认实际重叠或全域正确性。全部证据、9 个组及角色关联见[审计记录](../reviews/extract_mla.json)，未执行 GPU。
