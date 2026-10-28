# 撰写与优化 schema

本入口供 Planner（Designer）从零设计或修订已有 schema。目标是在已确认的 Contract 内，针对实际 workload 和目标形成可供 Coder 实现的设计。交付物是 schema 任务目录及必要的未决判断说明；实现与运行验证由后续实现任务承担。

## 1 确定目标与起点

先读任务要求，确认算法与验收标准、目标设备、已选实现栈、实际 workload 和优化目标。workload 与目标从任务说明取得；其字段格式尚未确定，不自行增加 schema 字段。Contract 的转录、资料缺项与参考入口问题按[模板准备说明](schema_template/README.md#1-确认资料与范围)处理，修改权限遵循 [README](README.md#2-schema-是什么)。

- **从零设计：** 用已确认要求建立 Contract，按[模板说明](schema_template/README.md#2-创建文件或进入已有任务)创建 root。首次填写时读 common 的[身份](schema_docs/0.common.md#单份-plan-与元数据)与[通用写法](schema_docs/0.common.md#通用写法)；转录 Contract 时用 [DOC 1](schema_docs/1.contract.algorithm_requirement.md#字段定义)与 [DOC 2](schema_docs/2.contract.hardware_specification.md#字段定义)核对表达和完整性。骨架中的占位不提供默认设计。
- **修改已有设计：** 先读 root 的共享 Contract、目标节点、其他节点的选择条件及已有反馈，确认当前版本和本次要解决的问题。再沿受影响的数学值、工作、资源与交接关系深入，其他节点只在影响范围涉及它们时展开。

区分本次目标与实现的既有选择。例如，改善某组 workload 的延迟，需要先确认计时边界、比较对象和允许的取舍；不能从当前源码支持范围推定算法有效域，也不能因本设计不用某项能力而删改硬件事实。

## 2 为当前判断查找依据

先确定主要成本或已有反馈指向的问题，再查能支持这一判断的材料：

| 当前问题 | 读取什么、核实什么 |
| --- | --- |
| 能用什么能力，容量和执行边界是什么？ | 从 [DOC 2 的资料绑定](schema_docs/2.contract.hardware_specification.md#字段定义)进入任务的 `wiki.hardware`；涉及具体原语或编译表达时查 `wiki.programming`。 |
| 哪类资源稀缺，某种交换是否值得？ | 查 `wiki.microbench` 的相对代价；核对设备、软件、运行条件与测量口径。没有适用测量时区分估算、设计假设与实测结论。 |
| 有哪些可借鉴的设计或实现？ | 查任务绑定的 `wiki.sota`，或本仓库的 [schema_wiki](schema_wiki/) 与 [kernel_wiki](kernel_wiki/)。先核对任务域、精度路径、环境和案例状态，再采用适用的决定及其依据。 |
| 如何理解已有性能反馈，或需要什么新证据？ | 查原实验记录，必要时进入 `wiki.profile` 定位采集和解释方法；提出具体待验证问题，不以调参替代尚未确定的设计。 |

学习现行表达时，先从[示例索引](schema_template/examples/README.md)选取相关案例并查看未决项。案例是材料，不是默认方案；设计有实际来源时，按[来源规则](schema_docs/0.common.md#来源与证据)记录，不为独立设计编造出处。

## 3 在数学与执行之间迭代

从目标所对应的成本入手，说明准备降低什么、付出什么，再逐步形成具体选择。以下是相互核对的设计问题，不要求按文件顺序一次填完：

| 设计问题 | 推进方法 |
| --- | --- |
| 实际计算哪些值，怎样减少主要成本？ | 按 [math_reshape 的设计指导](schema_docs/3.design.math_reshape.md#设计着眼点)确定改写、数值路径和逻辑寿命；未改写的计算引用 Contract，有改写时对照明确 baseline，按适用条件记录可计量的收益与代价。 |
| 一份工作切多大，由谁承担，是否有足够并行度？ | 按 [work_schedule 的设计指导](schema_docs/4.design.work_schedule.md#设计着眼点)权衡复用、工作数量、分配和交接；局部归约或递推难以并行时，返回数学区考虑改写。 |
| worker 内能否按预期推进，资源和依赖是否支持它？ | 按 [tile_pipeline 的设计指导](schema_docs/5.design.tile_pipeline.md#设计着眼点)核对负载、角色、缓冲、发布与复用。资源容量和驻留情况会反过来影响分块与工作组织。 |

保留决定的唯一定义，下游引用它；需要下游选值的参数按[绑定规则](schema_docs/3.design.math_reshape.md#引用与缺项)落实，不能留下让 Coder 自行决定的关键空位。数学、工作组织或局部流水线改变后，沿受影响关系重新核对。尚未选择方案时继续设计，缺少依据时明确待验证判断，不把这两种情况归结为 schema 缺字段。

设计收束后，用 [realization 的判断方法](schema_docs/6.realization.md#选择一条实现约束)补充上游尚未表达的必要实现条件与有理由的优先选择；没有新增交接信息时省略。

已有设计的修订按以下边界处理，具体文件操作复用[模板说明](schema_template/README.md#2-创建文件或进入已有任务)：

- 同一路径的改进，或能由同一份 Design 的参数、公式覆盖的变化，修订原节点。
- 不同 workload 确实需要并存设计时，再创建派生节点；完整给出该节点的 Design，核对所有选择条件并维护 dispatch 索引。`parent` 不提供 Design 继承。
- 需要改变任务语义、验收判据、设备或已选环境时，向任务方提出 Contract 问题，不能通过设计修订绕过。

收到 Coder 反馈后，先判断它暴露的是实现问题、设计问题还是 Contract 问题。设计问题回到对应决定处理；实现替代是否仍在自由范围内按 [DOC 6](schema_docs/6.realization.md#实现自由的边界)核对。修订时重新判断来源、realization 与旧反馈的适用性，版本和实验追溯按[模板规则](schema_template/README.md#3-填写与修订)维护。

## 4 检查与交付

按[模板入口检查](schema_template/README.md#4-检查与处理反馈)，并审阅数学与 Contract 的关系、参数绑定、资源可行性、依赖与交接是否闭合；多节点任务还需核对 dispatch 的互斥和覆盖。结构通过不代替这些判断，也不证明性能目标已经达成。

按[交付规则](schema_template/README.md#5-交付)提供完整任务目录，指明本次目标节点、设计修订、改动范围及未决判断。Coder 应能据此确定要实现哪条路径、必须保持哪些决定、还缺什么依据。影响实现的决定尚未完成时，明确交付的是草稿及可继续工作的范围；需要新实验时说明待验证的判断与适用条件，不要求 Planner 先完成生产 kernel。
