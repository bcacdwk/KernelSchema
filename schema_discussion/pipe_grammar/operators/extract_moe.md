# extract_moe pipe 现状

审阅冻结首批 106 个有稿节点、109 个精确组，以及全部 1,629 个有效角色位置、747 个 phase 上下文。状态为 approved 96、review_candidate 9、needs_authoring 1；154 个计划节点中另有 5 个失败骨架、43 个未开始，不作结构简洁证据。root 虽 authored=false，仍有材料并纳入。历史语义采用归档文档，未读取工作区正在改写的规则。

| 口径 | 角色 | pipe 块／嵌套块 | 静态动作 |
| --- | ---: | ---: | ---: |
| A：文件内源节点去 alias | 1,431 | 1,889／745 | 5,104 |
| B：本地 alias 展开 | 1,455 | 2,247／792 | 7,282 |
| C：每节点有效区域 | 1,629 | 2,517／888 | 8,186 |

三者最大深度均为 5；C 不展开运行迭代。A→B 增加 358 块、2,178 动作，B→C 增加 270 块、904 动作；13 个 no_scan 节点继承 pipeline，93 个节点本地提供。相同段落在分支、FC1/FC2 和四个 epilogue warp 间复用，体积不能直接当作独立机制数。root 单独为 B 19 角色、25 块、96 动作。

真实主体包括：root 的 count→scan→scatter→gather→FC1→FC2→finalize；small 把路由准备合到一个 CTA；split 把 FC1 分成两份归约切片，经 Part 物化后另行合并、激活和量化；direct 用一个十六 warp 协作角色处理连续 unit，再由最后到达者合成 token 输出。FC2 又分别采用七角色分工、四 warp mono/cp 或单阶段 flat。verify、probe、初始化叠加在这些主体之前；verify_only 只有一个角色、一个块、四个动作。它们不构成完整笛卡尔组合。

747 个有效 phase 中，600 个只有 Main，147 个使用两个生产者、MMA 和四个 epilogue 角色。角色数量没有展开到线程，也没有把动作内部的协作重新登记成角色。例如小路由准备的单动作包括暂存全部路由、分组直方图、前缀与散布，期间共享表从计数改作排位游标；root 则在多个 phase 间通过全局结果交接。这是已有工作划分在 pipe 中的呈现，不能只比较动作条数。

C 的字段组合为四字段 1,086、prologue+repeat+steady 133、prologue+steady 372、repeat+steady 838、steady 88。四字段中 1,070 个的首尾承担 worker 级协议进入、终结等待与 TMEM 退出，主体遍历分配的工作；另 16 个、5 个精确组的 flat FC2 块显式预填与排出末项（B 对应 946 和 14）。普通 mono/cp 的两次预置属于 t.steady，每份工作重新执行，不能与 worker prologue 合计成同一种填充。

t 通常是静态步进分配的本地 worktile 序号；i 为 H/BK、H/(SK·BK) 或 I/BK 的局部归约域，g=flatten(t,i) 为跨工作的连续阶段号。BProducer 可直接遍历 g，AProducer、MMA 按 t 后遍历 g[t]。cp 的槽号在每份工作归零，mono 的 g 槽号继续，完成 parity 又可跨工作保留。flat 在 I=BK 的选择域内令 t 同时充当工作号和阶段号。C 的 worktile／flatten／symbolic_bounds 标志为 1,539／461／938，是重叠句法线索，不是用途分区或次数。

最深只涉及组 `0c93510b22400ed9`：`double_verify_fused_big_flat`、`probe_verify_fused_big_flat`、`verify_fused_big_flat` 的 fc2.Main。五层依次是无 repeat 的 guard 根、run=[0,1−Clean)、t、前项／末项 b、c=[0,BN2/32)。每节点在 steady 和 epilogue 各有一个第五层 c，共 6 块；前项 b=[max(0,t−1),t)，末项 b=[max(0,count(t)−1),count(t))。这是条件、工作与分片读出叠加。small_cp 的 active=[0,Dirty) 包裹工作和局部阶段，最深为 4；split_flat 的 active=[0,min(1,count(t))) 保护分配与 last(t)。

Clean 对事件的影响各不相同。big_flat 的 run、small_cp 的 active 为空时，内部动作不存在；big_cp 等简单 count/finalize 则把已列动作定义为 no-op，保留 issue/done；部分工作域也由上游置空。无 payload 的 worker 仍可能执行 entry/retire；J>0 但本 worker 无工作时，普通七 warp 版本仍分配、会合、退出，flat 可不分配。不能把“零数据”“空工作域”“无动作”混为一谈。

flat 预填的范围分别受 min(1/2/3,count(t)) 截断，稳态交错前项读出、未来 payload、scale 和 metadata。Pending 延后写回三槽 Metadata，保护前项输出坐标；末项单列排出。mono/cp 的固定 p=[0,2) 则连短归约也发两次预置，越界 stage 提交空 copy group、保留事件，维持“两组更新”的等待协议。两种尾部处理不是同一种空域规则。

C 有 835 个纯 repeat+steady 子块、55 个 prologue 直属循环、15 个 epilogue 直属循环；220 张列表有相邻循环，305 张混合动作与循环。直接路径的 ValidUnit、Last 把有效性和末到者选举写成嵌套条件域；FC2 的 c 读出有的显列、有的封在 emit／produce 契约中。验证中的独立扫描、CTA OR、原子到达协议，以及单个 MMA 内四次 64 通道更新，也未计为额外 pipe 循环。

首尾含义还取决于发布边界。fused epilogue 在最后 Fragment 已读出后先 release，再做末段转换／写回；split FC1 先提交两个 Part 片段再 release。FC2 bulk 版本把 Sink 的源读取结束和全局写入完成分开，finish 只在整个序列末尾执行。相同 pipe 共有 46 组关联多种契约；ActQ epilogue 组有 304 个有效位置、32 种契约。BProducer 同一组既有 g+3 payload／g+2 scale 复用，也有 split_ws 的 g+2 联合等待。指纹相等不表示成员、模式、依赖或外围调度相等；fused_serial 的同形 entry 在普通 stream 下协议指令可无实际调度作用。

30 个初始化前缀生成格式表；其中 FormatTables 的单动作组跨 24 个节点，却有 24 份契约。重建时 probe 的临时 dirty 标志由 host 读取，每次 verify 则产出持久 verdict 并清输出；double_verify 保留两次扫描／清零，只以第二次判决控制后续。host mirror 的观察、超时、图录制都不替代 kernel_done，重建前缀也不由后续 replay 自动重做。13 个继承节点分别保留自己的 dispatch 和区域来源链，包括 schedule/math 继续跨父节点引用的情形。

外围绑定还决定这些遍历读取哪份数据：录制固定 ABI 与张量绑定，设备执行读取当时内容；构造时的路由直方图只参与选择，不能替代运行时路由图。小路由、分片归约和验证的工作分配并不相同，甚至 verify_split_ws 把多个 CTA 登记为同一个 worker。覆盖清单保留每个使用节点的工作域、分配方式、数学来源及父引用链，没有用代表组的坐标覆盖其他节点。

现有问题与证据边界：冻结文本明确保留 probe 并发普通标志写入、未检查 readback 状态，以及近似激活缺少全域误差界；split 路径还读取初值未规定的尾部 backing bytes，不能把掩码解释为完全无读取。root 硬件条目和外部比较器细节有缺项。census 未发现无效 pipe item，只能证明已统计结构可解析，不能证明源码全域正确、事件实际重叠或候选有效。全部节点绑定、成员自己的块坐标和分组判断见[审计清单](../reviews/extract_moe.json)；本轮未执行 GPU、提取或修改正式材料。
