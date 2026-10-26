# KernelSchema

本仓库设计一套描述 GPU kernel **流水线设计**的 schema：用结构化的 YAML 记录一个 kernel 要算什么、数学上怎么改写、工作怎么分配到硬件、tile 内部怎么流水，以及实现时必须遵守的条件。schema 位于所有 GPU 编程语言之上，由 Designer 填写、Coder 实现、人类专家审阅。

本 README 是全局性的导引：交代背景、分区总览、编写原则、仓库结构与现状。语言规范由 [schema_docs/0.common.md](schema_docs/0.common.md)、六个分区文档与 `schema_template` 中 `kernel` 包的格式约束共同定义，两者必须一致。文档片段与 `schema_wiki` 都用于举例；发现差异时先判断规则是否合理，不以兼容示例为由放宽模板。

需要了解分区与权限时看[分区总览](#2-schema-是什么)，查找具体规则时用[阅读入口](#阅读入口)。各章按概念依赖组织，可从当前要处理的区域进入，再沿引用补充所需前提。

## 术语

- **Designer**（也称 planner）：做设计的 Agent 或人，唯一能修改 Design 区。
- **Coder**：按 Contract 已选的实现栈把设计实现成代码的 Agent（如 AKA）。
- **schema**：既指这套描述语言（分区与字段规则），也指针对某个计算任务填好的设计实例；后者也叫 **plan**。Agent 直接编写和修改 YAML，供审阅与实现。一份 plan 对应一个 **dispatch**，描述该路径的流水线设计，可以包含多次计算 kernel 提交。
- **template**：用 CUE 编写的母版，定义字段、必填条件、开放词汇与可校验关系；入口为 `schema_template/schema.cue` 的 `#Task`。
- **wiki / microbench**：完整的硬件资料、指令手册，以及吞吐与延时的实测结果；schema 只引用，不抄录。

## 1 背景与动机

Kernel Agent 有两个普遍问题：**难以优化**——无法判断结果是否接近最优，长程任务高度依赖 LLM 自身的能力，很难突破平台期；**难以维护**——全部交给 Agent 后，人类专家无法观察、判断和介入，结果是黑盒，难以沉淀成可学习的 wiki。

根源是把**策略意图**和**代码实现**混在了一起。策略意图本质是数学问题：在给定的算法要求下，怎样最好地安排硬件资源、吞吐、内存与计算顺序；它是高层的、与具体 GPU 语言无关的，用结构化文本就能表达，也正是人类专家最擅长判断、最容易介入的 “taste”。代码实现则是 CuTe、CUDA、Triton 等语言的工程细节。两者混在一起时，每换一个策略就要推倒代码重来，Agent 因此过早收敛到局部最优。而好的流水线是 SOTA kernel 的必要条件：代码只能逼近一个设计的性能上限，不能凭空创造数据复用、并行度与流水重叠。

把两者分开，就要回答一个核心问题：

> 是否存在一个比源码明显更紧凑、更可解释，同时又保留绝大多数一阶性能因果的 **Pipeline IR**？

也就是说，这个中间表示能否成为性能的 “近似充分统计量”：同一流水线下，不同合格实现的性能差异相对可控；不同流水线之间的差异显著更大；据此就能判断候选方案的相对优劣；代码则可被验证为该流水线的一个 refinement。本仓库的 schema 就是这个 Pipeline IR 的候选。阶段性目标是：完备且唯一地表达主流 kernel（FA3/4、MoE、GEMM、MLA、GDN、KDA 等），人类能读懂并据此确定流水线，且词汇尽可能精简，不膨胀成又一门 DSL。

## 2 schema 是什么

**角色。** Designer 在两份 Contract 的基础上做设计，只有它能改 Design 区；Coder 在固定的硬件、算法与设计下逐代实现，只能在 `realization` 中附带有实验证据的反馈；人类专家随时介入、把握方向，只审阅 YAML 形式的 plan，不介入代码。协作方式介于 vibe coding 与全自动 loop 之间。

**分区。** 一份 plan 分为 Contract、Design、Realization 三部分、共六个子区，依次对应 `schema_docs/1–6`；此外开头有一段 `meta_data`（任务名、设计出处、dispatch 路径与修订号），不属于 Contract。

| 区 | 子区 | 记录什么 | 修改权限 |
| --- | --- | --- | --- |
| Contract | [contract.algorithm_requirement](schema_docs/1.contract.algorithm_requirement.md) | 算法要求：从外部看，算子必须满足的输入、输出与数学关系 | Designer 与 Coder 都不能改 |
| Contract | [contract.hardware_specification](schema_docs/2.contract.hardware_specification.md) | 硬件规格：目标设备的执行单元、核心能力与边界，已选的实现栈，以及完整规格与 microbench 的引用 | 同上 |
| Design | [design.math_reshape](schema_docs/3.design.math_reshape.md) | 数学改写：为了在该硬件上实现，对计算本身做的改写 | 只有 Designer 能改，Coder 遵守 |
| Design | [design.work_schedule](schema_docs/4.design.work_schedule.md) | 工作调度：计算如何分解、由谁执行、怎样分配与接续 | 同上 |
| Design | [design.tile_pipeline](schema_docs/5.design.tile_pipeline.md) | tile 流水线：worker 内执行角色的动作、依赖、片上资源及跨工作接续 | 同上 |
| Realization | [realization](schema_docs/6.realization.md) | 实现交接：设计所依赖、但上面没有表达的实现条件，以及优先尝试的实现选择 | Designer 填写；Coder 只能附带实验反馈 |

Design 三区从上到下排列，越靠上越难优化、收益越大、在源码里越难看出来：`math_reshape` 是前置（定下有哪些量、计算顺序、哪些量跨迭代携带、哪些中间量物化），之后 `work_schedule`（macro）与 `tile_pipeline`（micro）交替协同。

**文件格式。** YAML 是 plan 的编辑源；设计、修订号和实验反馈都直接在 YAML 中修改。CUE 提供明确的结构模板，统一校验入口读取整个任务目录，检查结构及必要的关联，不改写原文件。

通用 YAML 写法、元数据与任务组织见[公共约定](schema_docs/0.common.md)。

## 3 预期使用

围绕 schema 有三条工作流，各对应未来一份专门的流程或规则（见 §6）：

- **提取**（离线，SOTA kernel → schema）：从 `kernel_wiki` 的 SOTA kernel 提取出对应的 schema。
- **撰写 / 优化**（在线，planner）：依据 `schema_wiki` 的案例、CUE template 与 `schema_docs` 规则，从零撰写或逐步优化一条流水线设计。
- **实现**（coder，schema → kernel）：按某个 dispatch 的设计从零实现，或逐步修改优化对应的 kernel。

三条流都遵守上述编写方向。YAML 编写与校验入口见 [模板使用说明](schema_template/README.md)，三份完整的操作指导仍待补充。

## 4 编写原则

1. **奥卡姆剃刀，less is more。** 删掉不影响设计理解的一切；判断标准是 “删掉它，读者会不会读错或漏掉设计”。没有内容的字段直接省略，不填空壳。
2. **一个决定只写一次，写在唯一合适的位置。** 下游直接引用，不重述、不改义；同一个对象在各区用同一个名字。
3. **定量、可核对。** 收益与代价写成能核对的事实（计算量、访存量、误差界等），不写 “更快”“优化访存” 这类口号。
4. **抓大放小。** Designer 必需的内容不能漏、不能错；普通实现细节留给 Coder。`realization.prefer` 只记录不改变 Contract 和 Design、且有具体成本理由的优先选择；尚未明确的设计返回对应 Design 区补齐。
5. **有约束力的是具体决策，不是名称。** `persistent`、`CLC`、`online softmax` 这类名字只帮助理解，不能代替具体规则。

## 5 仓库结构与阅读顺序

```text
.
├── README.md            本文件：背景、分区总览、原则与现状
├── schema_template/     模板统一入口：schema.cue、填写骨架、校验工具与测试
│   ├── schema.cue       全部核心约束，按职责分章
│   ├── starter.yaml     Agent 待填写骨架，由 init 复制为 root.yaml
│   ├── tools/           CUE 安装、YAML 任务加载与关联校验工具
│   ├── tests/           CUE 与校验工具的回归测试，仅维护时使用
│   └── examples/        纳入检查的 YAML 编写示例
├── cue.mod/             仓库级 CUE 模块配置
├── schema_docs/         公共约定与六个区的字段规则、设计指导及局部示例
│   ├── 0.common.md
│   ├── 1.contract.algorithm_requirement.md
│   ├── 2.contract.hardware_specification.md
│   ├── 3.design.math_reshape.md
│   ├── 4.design.work_schedule.md
│   ├── 5.design.tile_pipeline.md
│   ├── 6.realization.md
│   └── archived/        旧版草稿与旧 README（只读，不再维护）
├── kernel_wiki/         转换前：SOTA kernel 源库（参考实现、方案、evidence、验证报告）
│   └── <batch>/<task>/  按批次、按任务组织
└── schema_wiki/         转换后：schema 实例库（plan）
    └── <batch>/<task>/  root.yaml / <id>.yaml + dispatch.md
```

- `kernel_wiki`（转换前）与 `schema_wiki`（转换后）一一呼应：前者是作为数据来源的 SOTA kernel，后者是从中提取、或据此撰写的 schema 设计。
- 任务目录中的文件身份、共享 Contract 与派生关系见[公共约定](schema_docs/0.common.md#派生节点与-dispatch)。
- 工具的使用、检查范围与规范维护要求见[模板说明](schema_template/README.md)。
- 模板配套内容均在 `schema_template` 内；`cue.mod`、工具版本、Makefile 和 CI 等项目配置保留在根目录。
- 基础 Harness 设施尚未纳入仓库，见 §6。

### 阅读入口

按当前问题选择入口，各区的语义与对应 CUE 定义共同构成规范。“核心思想”提供本区的概念前提，字段表和结构轮廓用于查阅，相关规则旁的短片段示范容易写错的用法与取舍；已有上下文时可直接定位具体主题。

| 当前关心的问题 | 入口 | 进一步需要时 |
| --- | --- | --- |
| 元数据、YAML 与共用记法 | [公共约定](schema_docs/0.common.md#单份-plan-与元数据) | 多条并存设计再看 [dispatch](schema_docs/0.common.md#派生节点与-dispatch) |
| 算法或硬件前提 | [算法要求](schema_docs/1.contract.algorithm_requirement.md)、[硬件规格](schema_docs/2.contract.hardware_specification.md) | 数值语义、资料绑定与省略条件各在所属章节 |
| 数学改写的选择与表达 | [math_reshape](schema_docs/3.design.math_reshape.md#核心思想) | 符号、寿命、精度与账本规则 |
| 一份工作怎样分配 | [基本对象](schema_docs/4.design.work_schedule.md#基本对象)与[单个 phase](schema_docs/4.design.work_schedule.md#单个-phase) | [提交与接续](schema_docs/4.design.work_schedule.md#phase-的提交与接续)、[数据交接](schema_docs/4.design.work_schedule.md#跨-phase-数据交接) |
| 一个角色怎样执行 | [lane 的含义](schema_docs/5.design.tile_pipeline.md#核心思想)与[单个 lane](schema_docs/5.design.tile_pipeline.md#单个-lane) | [资源与协作](schema_docs/5.design.tile_pipeline.md#worker-内的资源与协作)、跨工作与跨 phase 关系 |
| 哪些实现选择需要交接 | [realization](schema_docs/6.realization.md#选择一条实现约束) | 实现自由的边界、有证据的替代反馈 |
| 初始化、校验与错误定位 | [模板说明](schema_template/README.md#填写与检查)与 [starter.yaml](schema_template/starter.yaml) | [校验边界](schema_template/README.md#校验边界) |

从零编写时，以 [starter.yaml](schema_template/starter.yaml) 为起点，按各章规则填写。文档中的片段各自说明一个 schema 用法，不组成某个算子的完整设计。需要核对完整文件组织时，可参考现有的 [GDN 示例](schema_template/examples/gdn/root.yaml)及其[状态说明](schema_template/README.md#校验边界)。`schema_wiki` 保存更多历史设计，`kernel_wiki` 保存源实现；旧写法仅在需要追溯时查阅 `schema_docs/archived`。

修改语言规范时再查[维护要求](schema_template/README.md#维护)，同步核对 Markdown、CUE 与对应测试。

## 6 现状与 TODO

**已有：** 公共约定与六个分区文档、统一 CUE 规范、YAML 初始化与任务校验工具、正反例测试及 CI 检查；`kernel_wiki` 与尚未逐一按现行规范验收的 `schema_wiki`。使用方法与校验边界见 [模板说明](schema_template/README.md#校验边界)。文档中的局部示例用于解释 schema，不承担完整算子设计或性能验收。

**TODO：**

- **案例核对与验收**：按实际任务逐一核对已有 YAML，补齐资料与设计缺口；完整编写示例见 `schema_template/examples/gdn/root.yaml`，数值及性能验收另行进行。
- **更多转写与精简**：转写更多 SOTA kernel，收敛到最小交集原语。
- **检查器与评估器**：在现有结构和具名关系校验之外，继续实现资源记账、展开事件图的合法性与活性分析，以及尽量保序的性能估算。
- **基础 Harness 设施**：支撑离线提取与在线撰写迭代的轻量平台，尚未放入仓库。
- **三份操作指导（流程或规则）**：① 离线从 SOTA kernel 提取 schema；② planner 在线依据 `schema_wiki` + CUE + `schema_docs` 从零撰写或逐步优化 schema；③ coder 依据 schema 设计从零实现或逐步优化 kernel。
