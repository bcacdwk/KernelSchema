# M04：实际使用问题记录

以下包含初读误解、查清后的说明负担、输入缺口及未决分类；不要求每项已经证明文档错误。行号沿用 DRAFT 中固定来源。没有阅读其他角色报告。

## M04-01：文件头形成了错误的第一张机制图

- 场景：从 candidate/solution.py 入口向下读，先看 9–28 行的“single fused kernel / one CTA per / device-side ragged mapping / 2 reserved warps”，初步以为只有静态 tile 分配和普通 warp-specialized attention。
- 相关内容：P2 要追踪主要分配/协作，不能按 launch 数量计设计；实际 1648–1681 是 host DtoH 几何准备，2008 是动态 grid，1450–1486 的 warp10 是 pop server。文件头的 FP32 row sum、stage1 K overlay 也与当前 995–1063 的 f16x2 和 1081 的 SMEM_TINY 不完全相符。
- 两种读法：把头注释当当前权威设计概览，正文只是实现细节；或把它当历史总结，必须重新追实际 host/kernel 路径。
- 尝试与结果：用 rg 找函数、MODE、tiny、queue、launch，再读 host selector 和 warp 分工，改用实际代码，保留最初误读。没有把注释中的旧模式拆成候选。
- 影响：若只浅读头部会错选 root、漏完整持久方案，且错记数值链和 scratch 寿命。现行“事实回查源码”原则正确，但对一份千行内嵌 CUDA 文件，入口注释会先把读者带偏。
- 暂分类：输入说明陈旧；已自行纠正，不阻塞当前结构判断。

## M04-02：任务域的三种材料不一致，tiny 注释又引入第四种说法

- 场景：P1 需要区分完整任务、输入构造和候选实际支持；开始读 metadata 后，我暂把 H12/16/24/32/48/64/96/128 和 q≤kv 当完整域。
- 相关材料：metadata.description 的头数与“ragged首次causal prefill”；shape_range 的 head tuple 仅12/24/32/96/128、causal允许false/true、最后约束 query_lengths==kv_lengths；input.py:70 只断言0<q≤k；candidate:1074–1077 注释称 frozen shape_range allows q<k。
- 两种读法：输入构造与实现更通用，代表真实有效域；或 metadata 已指定 shape_range 为动态域权威，其他描述只是概括/旧注释。
- 尝试与结果：完整读 shape_range 的交叉约束，采用其具体域作本次 P1 摘要，不因更宽的函数签名或作者注释扩大域。此前 q≤kv 的理解登记为误读。
- 影响：影响 tiny warp-uniform 条件是否可能混合、head geometry、causal 分支、哪些实例是正式任务或仅 legacy。正式 Contract 仍应明确来源差异；不能用 sample 或候选 assert 代替域确认。
- 暂分类：输入说明不一致；当前按材料明确指定的 shape_range 暂定，可继续独立部分。

## M04-03：candidate 给出了看似完整阈值公式，metadata 却只给参数

- 场景：P1 逐输出验收。candidate 36–53 行直接写 `|a-r| <= .01+.05|a|`、rel-L2；metadata correctness_tolerances 只列 atol/rtol/mismatch/max_rel_l2。
- 相关规则：P1 第2段禁止只凭 relative_to 推公式，REVIEW 来源缺项允许保留已知判据继续；reference.py 的文件头称带 all scoring helpers，但实际通读只有 reference 计算/归一化/返回包装。
- 两种读法：candidate 作者总结足够视作验收来源；或候选实现自己的说明不能补成外部 evaluator 定义，尤其 mutated_inputs.out 与返回位置路由未确认。
- 尝试与结果：读 reference 全文、metadata 前部并 rg relative/tolerance/source，未找到比较器实现。保留 metadata 的已知阈值，candidate 公式只能附来源说明，不升级为已核实 comparator。
- 影响：可以继续确定数学和设计；不能承诺 relative_to=candidate、NaN/Inf 或归一化政策。没有无限追未提供的外部验收器，也不把 GPU 验证列成任务。
- 暂分类：输入不足；不阻塞当前提取事实，阻塞完整验收保证。

## M04-04：tiny 路径改变输出来源，局部闭合与“主要输出形成”两条判断都适用

- 场景：P2 为 tiny 候选选去处。读到精度路径与 epilogue 分支时，第一反应是新增完整设计；继续看 issuer 发现普通 P/PV 仍执行，又倾向局部。
- 相关规则：common:116–122 允许围绕已有值/role/存储闭合的局部新决定，但“需要重新贯通解释主要计算、输出形成”是新整体设计；REVIEW:45 要说明为什么主体加差额不足。源码698–738、1078–1087、1182–1217：SOFTMAX 额外形成 FP32 PV/denom 的 SMEM 结果，CORRECTION 读它，普通 MMA O 对实际输出失去作用但仍产生。
- 两种读法：主要 PV 从 tensor-core TMEM 换成 scalar FP32 SMEM，输出形成已变，应完整；或仅在已知 SOFTMAX/CORRECTION 和 fin/scratch 对象边界内改变值与消费者，工作组织和原 PV 主体都继续，因此可局部。
- 尝试与结果：对照 P3“被丢弃或没有消费者的值若仍发生求值……应如实保留”，不删无用 PV；暂把 root 定为 non-tiny，tiny 倾向局部但未放行。不能用 YAML 内 `if tiny_mode` 偷带另一个数值配置。
- 影响：直接影响完整代表数量、root 范围以及是否应把 tiny 当当前机制随输入分支。这里的难点不是“不知道源码做什么”，而是知道后仍难判断记录深度。
- 暂分类：说明/流程与自身尚不能确定；希望有一个类似“额外输出覆盖、原主计算仍执行”的现行实例说明如何衡量闭合，而非硬按涉及分区数判断。

## M04-05：K/V 3→4 槽看似参数变化，却同时换 release 协议

- 场景：从 host `<0,3>/<0,4>` selector 初步想降为 depth 参数，读 loader 后发现不同周期 barrier。
- 相关规则：common 的参数联动与局部新决定；DOC5 的存储复用/依赖要求；candidate 1329–1330、1361–1362、1393–1414：3槽复用直接等 pv_done；4槽另有 kv_free[4] 的 commit/wait，而 S/PV ring 仍3槽。
- 两种读法：同一 ring 深度变化及其必要同步联动，属于参数记录；或增加独立 release 通道是新的局部协议决定，至少应局部记录甚至可能主协作变化。
- 尝试与结果：追了初始化、PV commit、loader wait，暂降为“局部”，明确容量/索引/完成关联，而非只写 depth=4。因为 QK/PV/softmax/领取主体不变，目前没有强理由建第三份完整 YAML。
- 影响：若草率参数化，会漏 kv_free，甚至把3周期PV barrier用作4周期槽循环的实现依据。跨工作相位证明尚未做完，保留后续复核，不说已安全。
- 暂分类：说明/流程；已形成有据临时判断，但参数与局部标题界线仍有解释空间。

## M04-06：LPT 是运行时排列键，worktile_order.by 却只接受 partition 维度

- 场景：试写工作分配，想直接表达按 kv_tiles 下降、稳定保持 head/batch/querytile 原序。
- 相关规则：DOC4:193 by 为 partition 维度或列表；schedule.strategy 可补充运行时排列键。DOC4:164 的 worktile_count 是各 partition 工作数之积；本算子是 ragged 非矩形工作集合。
- 两种读法：需发明 `by: kv_tiles` 或 sort_key 字段，让长度成为排序维；或把真正独立工作编成 ItemGeo 表项 item 轴，by:item 升序，再在 strategy 唯一解释排序和原始坐标。
- 尝试与结果：未发明新字段。DRAFT 选择 item 轴，bootstrap 用已有 head/batch/tile 表序；LPT 拟采用同样表项身份配固定排序 strategy。查阅过程发现 strategy 表已明确允许运行时排列键，初始“by无法排序”疑问部分解决。
- 影响：能写，但关键工作映射大量落在字符串，读者必须确认 item 是真实独立worktile编号，而不是隐藏配置选路或把ragged洞作为工作。没有完整例子时较容易机械拆成batch×max_tiles而虚造padding CTAs。
- 暂分类：字段表达疑问/说明；当前不认为必须改固定字段，已找到保守写法。

## M04-07：side stream 输入就绪与静态表复用缺少可确认完成边

- 场景：按 P2 第3项检查缓存/调用前缀。host 每次读当前设备元数据，在静态 pinned/device staging 上复用，没有因“只是准备”跳过。
- 相关代码：1648–1681 nonblocking side_stream、DtoH、side_stream synchronize；1908–1912 HtoD d_table 与 record event；2011 主 stream wait event。静态 d_table、h_items、table_event、d_queue 跨调用存续。未见在 side_stream 的输入复制前 record/wait 当前stream事件，未见下一次表覆盖前等前一次 kernel 完成。
- 相关规则：DOC4:259–265 调用边界说本次入口提交在调用者此前设备工作完成后开始，外部供给/共享存储复用需真实前提；EXTRACT/REVIEW 说“依赖必须靠猜测时阻塞具体范围”，不能补造等待。
- 两种读法：固定 harness 每次调用前输入已同步、每次调用后已完全完成，因此现有side stream成立；或普通stream有序调用已经是接口承诺，nonblocking side stream可绕过输入生产者，也可能在上次kernel仍读表时覆写。
- 尝试与结果：定向读 launch_impl 全部 host 准备、上传、提交路径，rg capture/graph 未找到针对图录制的分流。没有追未提供调用 harness，也没有断言发生了实测竞态。只确认上行 upload→kernel 边，其他边标未知。
- 影响：不能用 `schedule.strategy: Geo is ready` 创造实际没有依据的保证；也不能把 shared d_table 保留当作普通 allocator 细节。所有拟完整代表的跨调用完整性受影响；仍可独立写内部工作/计算事实。
- 暂分类：输入不足/来源限制。最重要卡点。需要固定调用上下文给出确有的输入就绪与跨调用排空保证，或进一步据源码判定缺陷如何按规范忠实保留；本实验不修源码。

## M04-08：P2 正式记录要求定位完整基线，但此时只允许未来计划

- 场景：把 tiny、K/V ring、LPT 的降级判断写到候选表时，想按 common 的唯一“基线/差异与联动/条件与依据”格式直接写局部记录。
- 相关规则：common:172–180 要最终每条记录直接定位当前完整 YAML 及已有对象；P2:可写范围只有 dispatch 草案与root代表元数据，Design不能改，未来文件不能链接、共享待正文落实；P4才落实正式记录。
- 两种读法：P2也应严格使用唯一记录格式，否则草案不完整；或P2只需保留可核对线索与暂定对象，在P4正文存在后才要求正式基线定位。
- 尝试与结果：回读 P2 第6项“轻量记录线索”和 EXTRACT 的阶段表，采用第二种。DRAFT 的候选表明确不是正式 dispatch，未来文件用普通文字，不伪造链接，不创建只有provider作用的空YAML。
- 影响：这不是工具阻塞，但初读时我会担心“未完成基线不能降级”而提前写P3或虚构对象。本次通过回读阶段职责自行解决；保留这次查阅成本。
- 暂分类：说明/流程；已解决。可在P2靠近记录要求处直接提醒“正式局部基线定位在P4复核”，减少来回读。

## 未单列为问题的已知边界

- 工具/starter 尚未对齐已有明确说明。没有把旧字段复制进片段，也没有将本实验停在检查前；本实验本来就不运行工具。没有新发现需要重复登记这一已知问题。
- DOC4 的 CTA 单元一般来自硬件区，而本次硬件是占位；P1和模板明确豁免硬件检查，按源码写CTA事实即可，不把它记成硬件缺口。
- 最小等待语法通过DOC5例子已能读懂：quoted string、显式issue/done、箭头左侧是受约束事件、released按前驱条件分类。未运行checker，不能说工具验证通过。
