# 从已有 kernel 提取 schema

一次提取以一个完整算子为单位，将固定实现压缩为可追溯的 schema，保留数学与数值含义、工作组织、依赖、复用及适用范围。先建立共享 Contract，再审定路径划分、完成 root、定稿派生树、提取全部派生节点，最后独立终审。原实现的设计、代价与缺口按事实保留；提取过程中不优化或修复 kernel。

本页是流程总览，[Supervisor 手册](schema_extract/README.md)说明输入、输出、操作及命令，六张阶段任务页给出具体工作。catalog 算子的输出路径见 [schema_wiki 导航](schema_wiki/README.md)，检查显式使用 `--skip-hardware`。

## 提取顺序与交付

| 阶段 | 工作与产物 | 放行与下一步 |
| --- | --- | --- |
| [P1 共享 Contract](schema_extract/phases/01_contract.md) | 从 starter 建立 `root.yaml`，填写元数据和算法要求，保留硬件占位。 | reviewer 验收可独立理解的外部计算、有效域、输出与误差依据；Design 可留作草稿，随后进入 P2。 |
| [P2 dispatch 草案](schema_extract/phases/02_dispatch_draft.md) | 对照 host 路由，选代表主要 workload 的真实路径作 root，写 `dispatch.md` 草案及必要的 root `when`。 | reviewer 先验 root 选择、设计划分与已知覆盖缺口，再放行 P3。 |
| [P3 root 设计](schema_extract/phases/03_root.md) | 逐段完成 root 的数学改写、工作调度、微流水线和必要的实现交接。 | 完整 root 通过 `--skip-hardware` 检查及源码审阅，交给 P4。 |
| [P4 dispatch 定稿](schema_extract/phases/04_dispatch_plan.md) | 据已审 root 定稿索引，明确每个节点的 ID、直接父节点、设计差异和选择依据。 | reviewer 批准整棵树，Runner 核验结构化节点清单，再按依赖派发 P5。 |
| [P5 派生节点](schema_extract/phases/05_branch.md) | 从已审的直接父节点复制，为每个节点写完整 Design 和适用的 Realization。 | 各节点检查、独立审阅并合入后解锁子节点；没有分支时跳过 P5。 |
| [P6 全算子终审](schema_extract/phases/06_final.md) | 新 reviewer 先核对全树、条件、链接、引用和未决项，必要时再请 worker 修改。 | 当前计划全部节点已有效合入，整个目录检查与终审通过后才达到 `ready`，再向指定目录交付。 |

交付目录包含 `root.yaml`、所有计划内的 `<id>.yaml` 和中文 `dispatch.md`；只有 root 的任务也保留索引。共享 Contract 只在 root 中定义，各节点的 Design 均完整。精确选择条件以各 YAML 为准，`parent` 只表示设计派生关系。[节点与索引规则](schema_docs/0.common.md#派生节点与-dispatch)

Supervisor 指定来源和输出位置，并按[长期自主职责](schema_extract/README.md#角色与会话)处理预算、恢复与修复，记录决策并持续推进。Runner 管理依赖、调用、回执、checkpoint 与发布，是运行状态的唯一写入者。worker 只改本阶段允许的产物，reviewer 独立对照固定 kernel 判断设计语义与压缩是否忠实，正确且必要信息齐备时直接批准。工具代办调用身份、文件 SHA-256 和精确锚点；模型填写非空的语义 `summary`、选择证据并说明结论，不为机械字段或可选工具建议增加篇幅。会话、退修、裁决与恢复见[Supervisor 手册](schema_extract/README.md#角色与会话)，通用协议见[工具手册](schema_extract/tools/README.md#回执与短交接)。

## 七段内容的阅读导航

六个阶段负责推进，七段 schema 内容负责表达。每段的提取方法只在下表对应的任务页定义；P5 复用 P3 的逐段方法。字段语义仍由 `schema_docs/` 定义，创建、修订和检查操作见[模板说明](schema_template/README.md)。

| 段 | schema 内容 | 提取方法与规范 |
| --- | --- | --- |
| 1 | `meta_data` | [P1 元数据](schema_extract/phases/01_contract.md#第-1-段元数据) · [公共约定](schema_docs/0.common.md) |
| 2 | `contract.algorithm_requirement` | [P1 算法要求](schema_extract/phases/01_contract.md#第-2-段算法要求) · [算法规范](schema_docs/1.contract.algorithm_requirement.md) |
| 3 | `contract.hardware_specification` | [P1 硬件占位](schema_extract/phases/01_contract.md#第-3-段硬件占位)；提取只保留 starter 原样占位，不读取硬件资料 |
| 4 | `design.math_reshape` | [P3 数学改写](schema_extract/phases/03_root.md#第-4-段数学改写) · [数学规范](schema_docs/3.design.math_reshape.md) |
| 5 | `design.work_schedule` | [P3 工作调度](schema_extract/phases/03_root.md#第-5-段工作调度) · [调度规范](schema_docs/4.design.work_schedule.md) |
| 6 | `design.tile_pipeline` | [P3 微流水线](schema_extract/phases/03_root.md#第-6-段微流水线) · [流水线规范](schema_docs/5.design.tile_pipeline.md) |
| 7 | `realization` | [P3 实现交接](schema_extract/phases/03_root.md#第-7-段实现交接) · [实现交接规范](schema_docs/6.realization.md) |

**Less is more。** 每段复查“删掉这个细节，读者会不会读错或漏掉设计”。保留影响计算、写入边界、调度和同步的事实；已有决定直接引用，不重复叙述。源码从 host 接口、IO、路由和调用关系向内展开，读到能写出本段就先写、保存，再按问题回查。可以跨函数、跨层读取或分多轮完成复杂段，不一次通读源码树或一次生成整份 YAML。下层发现前提有误时，回到唯一定义处修正并复查受影响下游；跨阶段修改经 Runner 重开对应阶段。

worker 和 reviewer 始终以本次冻结的 `schema_docs` 为写法依据：先读当前段的规则，优先使用已有合法表达，不自创固定字段、封闭取值或记法，也不以正文改写结构含义。规范允许的具名对象、开放取值和自定义动作按其规定填写；旧稿或检查器通过不授予扩展权限。发现规则不适配、缺失或冲突时，先按下述问题分流指出具体条款、反例和现有写法为何不足，交 Supervisor 处理；经正式接续采用维护后的规则前，不将自造语义作为正式产物批准。

所有阶段采用[当前文档独立验收](schema_extract/REVIEW.md)：reviewer 先通读本阶段已完成的全部内容，包括未修改的段落，再核对固定源码；初始提示不提供前序批准、交接和旧稿。批准另须给出独立理解、分层、全文一致与内容必要性四项判断，P6 覆盖全部节点，不能只汇总既有结论。需要物理单文件阅读时，使用[正式视图生成](schema_extract/README.md#单文件阅读视图)展开共享 Contract，创作侧继续维护唯一 Contract。

## 来源、缺项与验收边界

输入构造、reference、baseline 和选定实现分别核实。来源要固定到实际版本；设计决定的证据保留文件、行号和短引文，长文件按相关范围续读。已有 schema 可帮助阅读，但事实回源码核实、表达按当前规范。[来源与证据](schema_docs/0.common.md#来源与证据) · [输入冻结与回执](schema_extract/README.md#输入与输出)

算子已经真实跑过 GPU；当前算子和 schema 都不需要 GPU 正确性验收。GPU 验收与实测不列为待办、缺口或交付条件。硬件区在撰写、搜索、检查、审阅及收尾中跳过，原样保留 starter 占位；完成 YAML 的检查显式使用 `--skip-hardware`。检查通过仅说明结构和已实现的关联通过，数学、调度、存储复用、选择条件和来源仍须独立审阅。[检查边界](schema_template/README.md#校验边界)

实现覆盖不足要说明实际范围，不缩小共享 Contract 或虚构 fallback。源实现缺陷按[设计分流](schema_extract/REVIEW.md#源实现缺陷与设计分流)保留事实与证据；不要求复现坏布局，也不借修复无依据地新增逻辑依赖或资源策略。确实不同的外部任务要求才拆成两个算子。来源明确的转录错误可纠正，实际任务要求变更仍须任务方确认。

缺项是否阻塞，以它是否迫使当前提取猜测计算或设计为准，P1 到 P6 使用同一边界。判断只覆盖当前阶段，不要求 P1 提前完成 Design：

- 任务数学、IO、reference 和实际数值路径可确定，外部验收器的公式、相对基准或检查路由未提供：在对应 `tolerance` 项保留[来源记录](schema_docs/1.contract.algorithm_requirement.md#固定实现提取中的验收记录)，回执说明证据与影响，独立审阅后可继续并发布忠实提取。
- 计算、输出副作用、关键精度转换、实现路径或依赖无法确定：指出缺失事实、受影响的具体关系，以及为何现有事实和明确限制不足以继续，阻塞受影响的范围。
- 已知事实被写错、遗漏或补造：退修，不能归为来源缺口。

可用材料中的定义仍须读取和保留；不要求无限追索未提供的外部验收器。“会改变外部测试通过/失败边界”本身不足以阻塞固定实现提取。后续阶段须核对未用缺口支撑设计或数值保证；若实际影响了某项提取，再处理该具体影响。缺项按[模板规则](schema_template/README.md#1-确认资料与范围)表达，不编造 reference、阈值或源码关系。批准表示设计忠实于固定来源及其限制，不证明源 kernel 已正确、复现原验收器或通过 GPU 验收；用于新建设计和实现验收时仍需完整判据。

## 框架问题分流

当前算子的实现差异、证据缺项和待修转录写入任务回执。规范、检查器、资料或流程本身造成错误、遗漏或重复劳动时，先核实已有规则，再在 `framework_candidates` 中提供可复查反例、最小改动及不改的影响；没有合格发现时返回 `[]`，正常完成本阶段。worker、reviewer 与 Supervisor 都可提报；worker/reviewer 不在算子调用中修改框架，Supervisor 按手册组织维护、验证和恢复，并将决定归档。

由主 agent 或维护者通过[维护入口](schema_extract/issues/README.md)按稳定 ID 和根因查重，再登记问题与决定。任务回执和必要交接保存在运行记录中，不为提报增设 schema 字段或固定报告文件。
