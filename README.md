# KernelSchema

本仓库设计一套描述 GPU kernel **流水线设计**的 schema：用结构化的 YAML 帮助 Agent 思考、表达和交流一个 kernel 要算什么、数学上怎么改写、工作怎么分配到硬件、微流水线怎样推进计算与复用资源，以及实现时必须遵守的条件。schema 位于所有 GPU 编程语言之上，由 Planner 填写、Coder 据此实现、人类专家审阅；YAML 最终供 Agent 和人阅读，不会被编译。

本 README 是全局导引。任务入口指导工作，`schema_docs` 定义字段语义与设计表达，`schema_template` 提供共用的 YAML 操作与检查。检查器落实文档中可直接判断的结构与关联要求，不反向增加设计义务；示例也不代替规范。

## 任务入口

执行具体任务时，先选择对应入口，再按当前问题进入规范、源码或资料：

| 当前任务 | 入口 | 交付什么 |
| --- | --- | --- |
| 离线从已有 SOTA kernel 及相关材料提取 schema，或修正已有提取结果 | [提取 schema](EXTRACT_SCHEMA.md) | 忠实于选定实现、可追溯的 schema 与必要缺口说明 |
| Planner 根据已完成的 Contract 设计新算子的 schema | [新建设计](DESIGN_SCHEMA.md) | 逐层完成数学改写、工作调度与微流水线的首版设计，以及未决判断 |
| Planner 根据已有 schema 与 Coder 反馈迭代优化 | [优化设计](OPTIMIZE_SCHEMA.md) | 有依据的设计修订、影响范围及下一轮验证要求 |
| Coder 根据 schema 从零实现，或按设计修改、优化 kernel | [实现 schema](IMPLEMENT_SCHEMA.md) | 对应设计版本的代码、验证结果及实现反馈 |

一次任务选择当前工作对应的入口，通过产物与必要说明交接。分区与权限见[分区总览](#2-schema-是什么)；需要系统学习或选择案例时，再看[阅读入口](#阅读入口)。

Schema Extract `2.0.4` 已完成真实算子的单命令验收，提供六阶段提取、依赖并发、暂停/取消、崩溃恢复、旧稿重审与原子发布。Supervisor 从手册发起 `python3 -m schema_extract extract --id <算子> --output <目录>`，运行始终使用自己的冻结执行器。算子问题随任务记录保存，规范、检查工具、资料或流程问题按稳定 ID 提报。当前硬件规格只保留模板原样占位，检查使用 `--skip-hardware`。[Schema Extract 手册](schema_extract/README.md) · [状态与恢复](schema_extract/README.md#恢复与交付) · [验收记录](schema_extract/README.md#验收记录) · [框架问题维护入口与历史决定](schema_extract/issues/README.md)

新建与优化均由 Planner 承担，按数学改写、工作调度、微流水线的顺序逐层推导和写入。先把上层方向收束，再向下投入优化精力；下层发现前提问题时，回到对应上层修正并复查受影响部分。

## 术语

- **Planner**：做设计的 Agent 或人，唯一能修改 Design 区。
- **Coder**：按 Contract 已选的实现栈把设计实现成代码的 Agent（如 AKA）。
- **schema**：既指这套描述语言（分区与字段规则），也指针对某个计算任务填好的设计实例；后者也叫 **plan**。Agent 直接编写和修改 YAML，供审阅与实现。一份 plan 对应一个 **dispatch**，描述该路径的流水线设计，可以包含多次计算 kernel 提交。
- **template**：`schema_template` 提供待填写骨架与统一检查入口；内部机器规则检查字段、必填条件和可校验关系。
- **wiki / microbench**：完整的硬件资料、指令手册，以及吞吐与延时的实测结果；schema 只引用，不抄录。

## 1 背景与动机

Kernel Agent 有两个普遍问题：**难以优化**——无法判断结果是否接近最优，长程任务高度依赖 LLM 自身的能力，很难突破平台期；**难以维护**——全部交给 Agent 后，人类专家无法观察、判断和介入，结果是黑盒，难以沉淀成可学习的 wiki。

根源是把**策略意图**和**代码实现**混在了一起。策略意图本质是数学问题：在给定的算法要求下，怎样最好地安排硬件资源、吞吐、内存与计算顺序；它是高层的、与具体 GPU 语言无关的，用结构化文本就能表达，也正是人类专家最擅长判断、最容易介入的 “taste”。代码实现则是 CuTe、CUDA、Triton 等语言的工程细节。两者混在一起时，每换一个策略就要推倒代码重来，Agent 因此过早收敛到局部最优。而好的流水线是 SOTA kernel 的必要条件：代码只能逼近一个设计的性能上限，不能凭空创造数据复用、并行度与流水重叠。

把两者分开，就要回答一个核心问题：

> 是否存在一个比源码明显更紧凑、更可解释，同时又保留绝大多数一阶性能因果的 **Pipeline IR**？

也就是说，这个中间表示能否成为性能的 “近似充分统计量”：同一流水线下，不同合格实现的性能差异相对可控；不同流水线之间的差异显著更大；据此就能判断候选方案的相对优劣；代码则可被验证为该流水线的一个 refinement。本仓库的 schema 就是这个 Pipeline IR 的候选。阶段性目标是：完备且唯一地表达主流 kernel（FA3/4、MoE、GEMM、MLA、GDN、KDA 等），人类能读懂并据此确定流水线，且词汇尽可能精简，不膨胀成又一门 DSL。

## 2 schema 是什么

**角色。** 两份 Contract 定义算法任务、硬件能力与任务已选环境，不随 Design 的选择增删。Planner 在此基础上做设计，只有它能改 Design 区；Coder 在固定的硬件、算法与设计下逐代实现，只能在 `realization.prefer[].feedback` 中附带有实验证据的替代反馈；人类专家随时介入、把握方向，只审阅 YAML 形式的 plan，不介入代码。协作方式介于 vibe coding 与全自动 loop 之间。

**分区。** 一份 plan 分为 Contract、Design、Realization 三部分、共六个子区，依次对应 `schema_docs/1–6`；此外开头有一段 `meta_data`（任务名、设计出处、dispatch 路径与修订号），不属于 Contract。

| 区 | 子区 | 记录什么 | 修改权限 |
| --- | --- | --- | --- |
| Contract | [contract.algorithm_requirement](schema_docs/1.contract.algorithm_requirement.md) | 算法要求：从外部看，算子必须满足的输入、输出与数学关系 | Planner 与 Coder 都不能改 |
| Contract | [contract.hardware_specification](schema_docs/2.contract.hardware_specification.md) | 硬件规格：目标设备的执行单元、核心能力与边界，已选的实现栈，以及完整规格与 microbench 的引用 | 同上 |
| Design | [design.math_reshape](schema_docs/3.design.math_reshape.md) | 数学改写：为了在该硬件上实现，对计算本身做的改写 | 只有 Planner 能改，Coder 遵守 |
| Design | [design.work_schedule](schema_docs/4.design.work_schedule.md) | 工作调度：计算如何分解、由谁执行、怎样分配与接续 | 同上 |
| Design | [design.tile_pipeline](schema_docs/5.design.tile_pipeline.md) | 微流水线：worker 内执行角色的动作、依赖、片上资源及跨工作接续 | 同上 |
| Realization | [realization](schema_docs/6.realization.md) | 实现交接：设计所依赖、但上面没有表达的实现条件，以及优先尝试的实现选择 | Planner 填写；Coder 仅可更新 `prefer[].feedback` |

Design 三区按计算方式、工作组织和局部执行排列：`math_reshape` 先定下有哪些量、计算依赖、哪些量跨迭代携带、哪些中间量物化，`work_schedule` 与 `tile_pipeline` 再分别表达宏观工作组织和局部执行约束。优化收益取决于具体任务与证据，不由所在层级决定。

通用 YAML 写法、元数据与任务组织见[公共约定](schema_docs/0.common.md)。

## 3 编写原则

1. **奥卡姆剃刀，less is more。** 删掉不影响设计理解的一切；判断标准是 “删掉它，读者会不会读错或漏掉设计”。
2. **一个决定只写一次，写在唯一合适的位置。** 下游直接引用，不重述、不改义；同一个对象在各区用同一个名字。
3. **定量、可核对。** 收益与代价写成能核对的事实（计算量、访存量、误差界等），不写 “更快”“优化访存” 这类口号。
4. **抓大放小。** Planner 必需的内容不能漏、不能错；设计决定在 Design 中表达，普通实现细节留给 Coder。
5. **有约束力的是具体决策，不是名称。** `persistent`、`CLC`、`online softmax` 这类名字只帮助理解，不能代替具体规则。

## 4 仓库结构与阅读顺序

```text
.
├── README.md            全局导引与任务分流
├── EXTRACT_SCHEMA.md    从固定源码与材料提取设计
├── DESIGN_SCHEMA.md     从已完成的 Contract 逐层设计新算子
├── OPTIMIZE_SCHEMA.md   判断实现反馈，逐层迭代已有设计
├── IMPLEMENT_SCHEMA.md  将 schema 落实为 kernel，验证并反馈
├── schema_extract/      新版 P1–P6 Runner、依赖并发、恢复、旧稿重审与原子发布
├── schema_extract_old/  旧实现与台账参考；旧运行仍使用各自冻结快照
├── schema_template/     共用的 YAML 操作与检查
│   ├── README.md        创建、修订、检查与交付
│   ├── starter.yaml     待填写骨架，复制为新任务的 root.yaml
│   ├── check_yaml.py    唯一日常检查入口，接收一个 YAML 路径
│   ├── tools/           检查器、有限流水线图工具、维护说明与测试
│   └── examples/        按现行规范维护的案例；README 说明状态与缺口
├── schema_docs/         公共约定与六个区的字段规则、设计指导及局部示例
│   ├── 0.common.md
│   ├── 1.contract.algorithm_requirement.md
│   ├── 2.contract.hardware_specification.md
│   ├── 3.design.math_reshape.md
│   ├── 4.design.work_schedule.md
│   ├── 5.design.tile_pipeline.md
│   ├── 6.realization.md
│   └── archived/        旧版草稿与旧 README（只读，不再维护）
├── kernel_wiki/         算子实现与证据库；README、分类索引与本地检索入口
│   └── <batch>/operators/<family>/[<subcategory>/]<task>/
│                       按批次、功能和稳定算子 ID 组织源码、题目与性能证据
└── schema_wiki/         schema 实例库（含尚未按现行规范验收的设计）
    └── <batch>/<task>/  root.yaml / <id>.yaml + dispatch.md
```

- `kernel_wiki` 保存任务与实现，`schema_wiki` 保存提取或撰写的设计；对应关系以 `meta_data.source` 等实际引用为准，不靠目录同名或批次号推定。
- 任务目录中的文件身份、共享 Contract 与派生关系见[公共约定](schema_docs/0.common.md#派生节点与-dispatch)。
- 基础 Harness 设施尚未纳入仓库，实际实现使用任务提供的构建、测试与测量入口。

### 阅读入口

**执行任务：** 从上方[任务入口](#任务入口)开始，不需要先读完全部规范。创建、修订或检查 YAML 时进入[模板说明](schema_template/README.md)的对应小节；字段与设计问题由流程引向相应 DOC。

**系统学习：** 希望先理解整套表达时，按以下顺序阅读：

1. [公共约定](schema_docs/0.common.md)：共用写法、身份、引用与共享 Contract。
2. 按上方分区表依次读第 1–6 篇：先 Contract，再 Design，最后 Realization。各篇的“核心思想”与设计指导解释如何思考，字段表和结构轮廓用于查阅，局部片段解释具体用法，不组成完整算子。
3. 选择一个任务入口，把规范用于具体工作；用[模板说明](schema_template/README.md)完成文件操作与检查。

**查阅案例与来源：** 学习现行写法时先看[示例索引及状态](schema_template/examples/README.md)；追溯或修改某份设计时读 `schema_wiki` 的任务目录，核对源码时进入对应 `kernel_wiki` 批次与任务说明。[Kernel Wiki](kernel_wiki/README.md)现收录 L20D 的 44 个算子实现，提供[分类总索引](kernel_wiki/CATALOG.md)、[精度／阶段／来源检索](kernel_wiki/FACETS.md)与[验证状态说明](kernel_wiki/EVIDENCE.md)。案例与实验的适用范围需要核实，完整实例不是开始工作的前提。

旧规范仅在追溯时查阅 `schema_docs/archived`；维护规范或检查器时，再读 [tools/README.md](schema_template/tools/README.md)。

## 5 现状与 TODO

**已有：** 提取、新建设计、优化设计和实现四个任务入口，公共约定与六个分区文档，YAML 填写骨架、统一 Python 检查入口及内部正反例测试；`kernel_wiki` 与尚未逐一按现行规范验收的 `schema_wiki`。使用方法与校验边界见[模板说明](schema_template/README.md#校验边界)。文档中的局部示例用于解释 schema，不承担完整算子设计或性能验收。

独立的[微流水线图工具](schema_template/README.md#微流水线图)按明确给出的展示规模展开各 phase 的动作和依赖，检查重复动作、悬空引用、成员匹配与依赖环，并在每份 YAML 旁生成同名 PNG。它不解析文字契约，也不证明跨 phase 活性或全部存储复用安全。

**TODO：**

- **案例核对与验收**：按实际任务逐一核对已有 YAML，补齐资料与设计缺口；已规范化的案例及其状态见[示例索引](schema_template/examples/README.md)，数值及性能验收另行进行。
- **更多转写与精简**：转写更多 SOTA kernel，收敛到最小交集原语。
- **检查器与评估器**：在现有结构、具名关系及有限 phase 内依赖检查之上，继续完善资源记账、跨 phase 与更大规模的合法性和活性分析，以及尽量保序的性能估算。
- **提取范围**：算子已经真实跑过 GPU；当前算子和 schema 都不需要 GPU 正确性验收。[Schema Extract](schema_extract/README.md)完全忽略 GPU 验收与实测。
