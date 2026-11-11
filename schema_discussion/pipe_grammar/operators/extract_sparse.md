8 个冻结节点、21 个精确 pipe 组及 51 个角色上下文均已核对，均为本地 pipeline。source 声明口径 A 为 44 roles、105 blocks、212 静态 actions；alias 展开 B 与 effective C 均为 51/125/247，其中 74 个嵌套块，最大深度 3。差额来自 probe_hidden_packed 的 remainder 引用 prefix：增加 7 roles、20 blocks、35 actions；其他 alias 只复用契约等内容。root 单独仅 2 roles、3 blocks、13 actions。全部 repeat 都引用具名域；symbolic_bounds 是重叠线索，不能计作运行轮数。完整逐成员证据见[审阅记录](../reviews/extract_sparse.json)。

root 的 attention 按 token/head/KV split/latent 切片分配 CTA，pipe 只遍历该 split 的 c∈[c0,c1)，`value:Acc(c-c0)` 对接局部递推；DQ 切片重复 gather、QK 和 normalizer。combine 经 flow 的 kernel_done 与三个 global handoff 接收 partials，四个动作内部折叠全部 split，没有 split repeat。其首批四向预取、后续四个一批及单个尾项都留在动作契约。空 split 仍执行 metadata 并写 sentinel，prepare/finish 保留 no-op 事件，PartO 不写；combine 的 padded head 四动作也保留 no-op。[root:284、428、519](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/root.yaml#L284)

mma_direct 用一个 CTA 处理 token/head/latent 切片的完整 c 域并直接归一化写出；Stages/QKS/Early 决定首块预发、Q 缓存和环轮转，pipe 形状不随配置改变。Stages=1 的 `enqueue:Future(c)` 仍存在但无操作；尾部只余空 copy group，finish 不额外 wait-all。probe_serial_mma 的 attention 与 probe_hidden_mma 的 remainder 固定 cfg8、HB16、KC128、DQ1，各有五动作 chunk 循环而无 enqueue；后者使用完整 T 预先规划的余段参数。[mma_direct:220](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/mma_direct.yaml#L220)

tc_heads 的七角色以 c 域分工：Setup 只有 prologue，Finish 只有 epilogue，Loader 和四个 Soft 在循环后收尾。两位 QK issuer 分别处理 latent 半轴，四个 Soft 分摊行，Soft0 发两半 PV；这些半轴、行组和乘加内部循环不增加 pipe 深度。Mask/valid 发布与 Gather 落地分离；同名唯一 load/publish 提供默认等待，显式边承担 c+2 槽复用、前轮 PV→rescale 和末轮 PV→stage，最终 store 同时等四个 stage 与 Loader 尾提交。[tc_heads:290](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/tc_heads.yaml#L290)

五个 packed phase 共用七组 pipe：tc_packed 的 attention 为 GT2；probe_serial_packed 的 attention 及两个 hidden 的 prefix、probe_hidden_packed 的 remainder 为 GT4/8。Setup 先 build:Union，再相邻执行 v、e 两域；v=[0,min(1,C)]、e=[0,1-min(1,C)] 控制非空计算与空 union 写零。Loader 与四个 Soft 在 v 内遍历 c，故出现第三层；Finish 只在 v 内写出。各 worker 预分配一份工作，无持久领取循环。空 union 不产生 v/c 动作，而非空 union 中某个 query 全无有效项仍走递推。c=0 的 rescale 保留事件但不读写 Acc；末轮 last(c) 受非空 v 保护。[tc_packed:237、404](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/tc_packed.yaml#L237)

打包组的共同语法之外，数学定义保留两种缓存池的身份，即使指针或行号相同也分别计数。跨查询共享一个带成员位的条目，同一查询的重复出现另建私有条目；池内顺序取实际构建次序。块数量由这个实际合并结果决定，不能由输入槽宽直接代替。加载动作内部再按跨池边界选择搬运方式，填充位置依然读取指定池的零号行。哈希表、完整索引行、键值环及最终输出暂存的先后复用，依赖契约中的发布与读者退休点；这些生命周期不会自动变成新的语法块。

四个 Soft 的概率发布各只覆盖自己的行，联合就绪才允许值乘积发射；两个 QK 动作的完成依靠共同通知，PV 则使用独立、可覆盖此前同类操作的完成组。最后一轮值乘积完成后才能读取累加器并暂存结果；加载方的空尾提交尚未结束也不阻止这一步，最终统一写出才汇合它。该区别解释了收尾动作的位置：列表顺序、默认同名依赖、显式边和原语承诺共同限定可继续执行的条件。单凭七角色或三层结构，无法得到硬件实际重叠时间。

四个 probe 节点每次提交八个 sample CTA；三处 pipe 为单个 produce，probe_serial_mma 则拆 inspect/build/publish。选槽遍历、next-index 预取、hash 线性探测与一次计数 flush 均在契约内部；无效、跨 request 或尾部样本仍写 -1。串行两节点由 flow 规定 probe→attention；hidden 两节点提交 probe/prefix/remainder，remainder 只等 probe。接受 R 时余段 packed，拒绝时余段 MMA，prefix 均保留。alias 相同不意味着输出范围、全局 group 偏移或 worker 存储相同。[probe_hidden_packed:307](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/probe_hidden_packed.yaml#L307)

探测计数还保留“创建键”与“首次设置成员位”可能来自不同并发次序的事实；真正构建注意力合并表时，竞争者等待完整位置发布。两类动作都涉及哈希和预取，但协议不同。串行提交处理整批输入，隐藏路径的前缀仅覆盖完整组，余段单独计算全局偏移并可能包含尾组。全批次指针保持原绑定，因此重复出现的同一个动作字符串不能独立说明它处理哪些输出。

pipe 的层级不能穷尽提交协议：hidden 的 caller-prior fork、最终 side-to-caller join、有界 host poll、跨调用 gate 身份及 graph 录制/回放边界位于 schedule/realization 文字。poll 退出不等于 probe 完成，ProbeRatio 没有 GPU handoff。来源还明确记载 root 的 DQ 非原子重复统计写、TC allocation/union 与活跃 barrier 的错误地址重叠、gate 重置竞争，以及 root 硬件 null 骨架；这些分别是来源限制或资料缺项，不是新增 pipe 层级的事实。判断依据为冻结旧规范的块执行、空域/no-op、默认依赖与动作完成规则；本次未执行 GPU，也未展开运行计数。
