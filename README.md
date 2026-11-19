# KernelSchema

本仓库设计一套描述 GPU kernel **流水线设计**的 schema：用结构化的 YAML 帮助 Agent 思考、表达和交流一个 kernel 要算什么、数学上怎么改写、工作怎么分配到硬件、微流水线怎样推进计算与复用资源，以及实现时必须遵守的条件。schema 位于所有 GPU 编程语言之上，由 Planner 填写、Coder 据此实现、人类专家审阅；YAML 最终供 Agent 和人阅读，不会被编译。

Kernel Schema 的核心出发点在于：压缩是智能的基础，对语义、数学与算法的压缩本身就对应着新的知识、关系和信息。正如真随机数以不可压缩性衡量最大熵，schema 旨在用精简凝练、可组合表达的结构，最大限度抽象一个 Kernel 的核心信息——即算子的压缩。

本页是全局导引：任务入口指导工作推进，`schema_docs` 定义字段语义与设计表达，`schema_template` 提供待填写骨架与统一检查入口 `check_yaml.py`。填写后运行检查，工具自动规范化可确定的写法并以 warning 报告修改，超长正文或需判断的问题按诊断处理。检查器只落实可机械判定的结构与关联要求，不反向增加设计义务；案例也不代替规范。

## 任务入口

执行具体任务时，先按当前输入选择入口，再沿问题查阅规范、源码或资料：

| 当前任务 | 入口 | 交付什么 |
| --- | --- | --- |
| 从固定 SOTA kernel 及相关材料离线提取 schema，或维护未交付提取 | [提取 schema](EXTRACT_SCHEMA.md) | 忠实于选定实现、可追溯的具体完整设计、变体目录与必要缺口说明 |
| Planner 按给定或已确认的 Contract 设计新算子 schema | [新建设计](DESIGN_SCHEMA.md) | 逐层完成的首版具体设计、必要目录记录及待验证判断 |
| Planner 根据已有 schema 与实现反馈迭代优化 | [优化设计](OPTIMIZE_SCHEMA.md) | 有依据的设计修订（或保留原设计的结论）、影响范围及下一轮验证要求 |
| Coder 根据选定 schema 从零实现，或修改、调优 kernel | [实现 schema](IMPLEMENT_SCHEMA.md) | 对应设计版本与来源身份的代码、验证结果及有据反馈 |

`CONTRACT` 由任务方给定或确认（提取按固定来源建立）。新算子只有需求、尚无 `root.yaml` 时，先按[模板说明](schema_template/README.md#1-确认资料与范围)将已确认的算法要求、固定参考入口与硬件环境如实填入 `root.yaml`（缺少现成参考入口时先按 `Compute` 补齐并固定版本），再进入[新建设计](DESIGN_SCHEMA.md)；前提未定或冲突时向任务方确认，Planner 与 Coder 不自行设定任务或放宽验收。代表性 workload、优化目标与计量范围随原任务材料或本轮交接说明保留，不在 YAML 中自增字段。分区职责见[分区总览](#2-schema-是什么)，系统学习或查阅案例见[阅读入口](#阅读入口)。

新建与优化均由 Planner 按“数学改写 → 工作调度 → 微流水线”逐层收束并写入；下层发现前提问题时，返回最早受影响的上层修正并复查受影响部分。

Schema Extract 通过唯一入口 `schema_extract/extract_yaml.py` 执行六阶段提取、独立审阅、按实际来源返回与 Runner 原子发布（默认 Claude Code `qwen3.8-max[1m]` / `max`，各阶段与角色独立会话；Codex 可显式指定）。Linux 运行环境与命令见 [Supervisor 手册](schema_extract/README.md#开始一次试用)，角色与工具权限见[执行协议](schema_extract/tools/README.md#运行恢复与维护)，正式输出目录见 [Schema Wiki](schema_wiki/README.md)。交付须核对目标状态与凭据：`ready` 仅表示当前 P6 终审通过，登记输出目录以 `published: true` 为准，另目录 `export` 以对应发布记录与目标 `.extraction/receipt.json` 为准；待发布或中断运行的排查与接续见[恢复与交付](schema_extract/README.md#恢复与交付)。

## 术语

- **Planner**：做设计的 Agent 或人，唯一能修改 Design 区，并填写 `REALIZATION` 的 `Require / Prefer / Forbid`。
- **Coder**：按 Contract 已选的实现栈把设计实现成代码的 Agent（如 AKA）。
- **schema**：既指这套描述语言（分区与字段规则），也指针对某个计算任务填好的设计实例；一份完整 YAML 也叫 **plan**。每份 plan 表达一套具体已选设计，可以包含多次计算 kernel 提交；任务的 `dispatch.md` 组织完整设计和变体记录，身份与数量规则见 [common](schema_docs/0.common.md#派生节点与-dispatch)。
- **template**：`schema_template` 提供待填写骨架与统一检查入口；内部机器规则检查字段、必填条件和可校验关系。
- **wiki / microbench**：完整的硬件资料、指令手册，以及吞吐与延时的实测结果；schema 只引用，不抄录。

## 1 背景与动机

Kernel Agent 有两个普遍问题：**难以优化**——无法判断结果是否接近最优，长程任务高度依赖 LLM 自身的能力，很难突破平台期；**难以维护**——全部交给 Agent 后，人类专家无法观察、判断和介入，结果是黑盒，难以沉淀成可学习的 wiki。

根源是把**策略意图**和**代码实现**混在了一起。策略意图本质是数学问题：在给定的算法要求下，怎样最好地安排硬件资源、吞吐、内存与计算顺序；它是高层的、与具体 GPU 语言无关的，用结构化文本就能表达，也正是人类专家最擅长判断、最容易介入的 “taste”。代码实现则是 CuTe、CUDA、Triton 等语言的工程细节。两者混在一起时，每换一个策略就要推倒代码重来，Agent 因此过早收敛到局部最优。而好的流水线是 SOTA kernel 的必要条件：代码只能逼近一个设计的性能上限，不能凭空创造数据复用、并行度与流水重叠。

把两者分开，就要回答一个核心问题：

> 是否存在一个比源码明显更紧凑、更可解释，同时又保留绝大多数一阶性能因果的 **Pipeline IR**？

也就是说，这个中间表示能否成为性能的 “近似充分统计量”：同一流水线下，不同合格实现的性能差异相对可控；不同流水线之间的差异显著更大；据此就能判断候选方案的相对优劣；代码则可被验证为该流水线的一个 refinement。本仓库的 schema 就是这个 Pipeline IR 的候选。阶段性目标是：完备且唯一地表达主流 kernel（FA3/4、MoE、GEMM、MLA、GDN、KDA 等），人类能读懂并据此确定流水线，且词汇尽可能精简，不膨胀成又一门 DSL。

## 2 schema 是什么

**角色。** 两份 Contract 由任务方给定或确认（提取按固定来源建立），定义算法任务、硬件能力与已选实现环境，不随 Design 选择增删。Planner 在此基础上设计，唯一能改 Design 三区并填写 `REALIZATION` 的 `Require / Prefer / Forbid`；Coder 在固定的硬件、算法与设计下实现，仅可在 `REALIZATION.Feedback[]` 中追加针对 `Prefer` 且带真实实验证据的替代选择；人类专家随时介入把握方向，审阅 YAML 及目录中的设计记录，不介入代码。协作方式介于 vibe coding 与全自动 loop 之间。

**分区。** 一份完整有效 plan 分为 Contract、Design、Realization 三部分、共六个子区，依次对应 `schema_docs/1–6`；此外开头有一段 `META_DATA`（任务名、设计出处、完整设计身份与范围、修订号），不属于 Contract。共享 Contract 与 Design 的本地定义或整区引用遵循 [common](schema_docs/0.common.md#文件与共享-contract)。

| 区 | 子区 | 记录什么 | 修改权限 |
| --- | --- | --- | --- |
| Contract | [CONTRACT.Algorithm_requirement](schema_docs/1.contract.algorithm_requirement.md) | 算法要求：从外部看，算子必须满足的输入、输出、数学关系与验收判据 | 任务方给定或确认（提取按固定来源）；Planner 与 Coder 不可改 |
| Contract | [CONTRACT.Hardware_specification](schema_docs/2.contract.hardware_specification.md) | 硬件规格：目标设备的执行单元、核心能力与边界、已选实现栈及资料入口引用 | 同上 |
| Design | [DESIGN.Math_reshape](schema_docs/3.design.math_reshape.md) | 数学改写：结合硬件相对代价，对实际求值、精度路径与物化状态做的改写 | 仅 Planner 可改，Coder 遵守 |
| Design | [DESIGN.Work_schedule](schema_docs/4.design.work_schedule.md) | 工作调度：计算如何切成独立工作、映射到执行单元并安排排列、分配与交接 | 同上 |
| Design | [DESIGN.Tile_pipeline](schema_docs/5.design.tile_pipeline.md) | 微流水线：worker 内执行角色的动作、依赖、片上存储复用及跨工作接续 | 同上 |
| Realization | [REALIZATION](schema_docs/6.realization.md) | 实现交接：上游尚未表达的必要原语、预算兼容条件与有成本依据的优先选择 | Planner 填 `Require / Prefer / Forbid`；Coder 仅可追加 `Feedback[]` |

Design 三区按计算方式、工作组织和局部执行排列：`Math_reshape` 先定下有哪些量、计算依赖、哪些量跨迭代携带、哪些中间量物化，`Work_schedule` 与 `Tile_pipeline` 再分别表达宏观工作组织和局部执行约束。优化收益取决于具体任务与证据，不由所在层级决定。

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
├── DESIGN_SCHEMA.md     按给定或已确认的 Contract 逐层设计新算子
├── OPTIMIZE_SCHEMA.md   判断实现反馈，逐层迭代已有设计
├── IMPLEMENT_SCHEMA.md  将 schema 落实为 kernel，验证并反馈
├── schema_extract/      六阶段与审阅提示、Runner 执行协议、恢复维护与发布
├── schema_template/     共用的 YAML 操作与检查
│   ├── README.md        创建、修订、检查与交付
│   ├── starter.yaml     当前字段的待填写骨架
│   ├── check_yaml.py    统一规范化、检查与按需绘图入口
│   ├── tools/           只读规则、有限图模块、维护说明与测试
│   └── examples/        四个案例、九份完整设计、变体目录与有限事件图
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
│   ├── aka-cuda-operators-20261030/operators/  44 题主批次，按功能与稳定 ID 组织
│   └── aka-four-problems-20261019/            Template 四题固定来源快照
└── schema_wiki/         按 catalog 镜像的提取输出目录与导航
    └── <batch>/operators/<family>/[<subcategory>/]<task>/
                        未提取目录以 .gitkeep 占位；已发布目录含 root.yaml / <id>.yaml + dispatch.md
```

- `kernel_wiki` 保存任务与实现，`schema_wiki` 接收提取或撰写的设计；输出路径按 `kernel_wiki/catalog.json` 的 `paths.readme` 父目录镜像，具体操作见[输出目录导航](schema_wiki/README.md)。设计来源仍以 `META_DATA.Source` 等实际引用为准。
- 任务目录中的文件身份、共享 Contract 与派生关系见[公共约定](schema_docs/0.common.md#派生节点与-dispatch)。
- 基础 Harness 设施尚未纳入仓库，实际实现使用任务提供的构建、测试与测量入口。

### 阅读入口

**执行任务：** 从上方[任务入口](#任务入口)开始，不需要先读完全部规范。创建、修订或检查 YAML 时进入[模板说明](schema_template/README.md)的对应小节；字段与设计问题由流程引向相应 DOC。

**系统学习：** 希望先理解整套表达时，按以下顺序阅读：

1. [公共约定](schema_docs/0.common.md)：共用写法、身份、引用与共享 Contract。
2. 按上方分区表依次读第 1–6 篇：先 Contract，再 Design，最后 Realization。各篇的“核心思想”与设计指导解释如何思考，字段表和结构轮廓用于查阅，局部片段解释具体用法，不组成完整算子。
3. 选择一个任务入口开展具体工作，按模板说明完成创建、检查与交付。

**查阅案例与来源：** [Template 示例索引](schema_template/examples/README.md)提供四个案例的九份完整设计，沿各案例 README、dispatch 和 YAML 阅读；源码与证据来自独立的[四题固定快照](kernel_wiki/aka-four-problems-20261019/README.md)，入口见[来源索引](schema_discussion/template_examples_20261117/SOURCES.md)。[Kernel Wiki](kernel_wiki/README.md)另有 44 题主批次，提供[分类总索引](kernel_wiki/CATALOG.md)、[精度／阶段／来源检索](kernel_wiki/FACETS.md)与[验证状态说明](kernel_wiki/EVIDENCE.md)，其提取输出导航在 [schema_wiki](schema_wiki/README.md)。核对实际源码身份和实验范围后再复用。

维护规范或检查器时，再读 [tools/README.md](schema_template/tools/README.md)。

## 5 现状与 TODO

**已有：** 提取、新建设计、优化设计和实现四个任务入口，公共约定与六个分区文档，YAML 填写骨架、统一 Python 检查入口及检查器正反例测试；四个 Template 案例的九份完整设计与有限图；`kernel_wiki` 的 44 个算子与对应的 `schema_wiki` 输出目录。使用方法与校验边界见[模板说明](schema_template/README.md#校验边界)。文档中的局部示例用于解释 schema，不承担完整算子设计或性能验收。

统一入口的[微流水线图选项](schema_template/README.md#微流水线图)在其支持的协议内按展示规模展开各工作阶段的动作和依赖，检查重复动作、悬空引用、成员匹配与依赖环，并在请求的 YAML 旁生成同名 PNG。它不解析文字契约，也不证明跨工作阶段的活性或全部存储复用安全。

**TODO：**

- **执行器验证**：目标 Linux 运行环境见[开始试用](schema_extract/README.md#开始一次试用)，离线测试与 Reviewer 能力回归方法见[离线验证与能力回归](schema_extract/tools/README.md#离线验证与能力回归)；继续补充真实 CLI 与受控全链路的运行验证记录（与离线测试、正式算子提取分别记录）。
- **案例资料与来源缺口**：继续补齐 Wiki 绑定与源码中未获保证的关系，具体边界见[示例索引](schema_template/examples/README.md)；数值及性能验收按相应任务执行。
- **更多转写与精简**：转写更多 SOTA kernel，收敛到最小交集原语。
- **检查器与评估器**：在现有结构、具名关系及有限工作阶段内依赖检查之上，继续完善资源记账、跨工作阶段与更大规模的合法性和活性分析，以及尽量保序的性能估算。
