# 从已有 kernel 提取 schema

一次提取以一个完整算子为单位，将固定实现压缩为可追溯的 schema，保留数学与数值含义、工作组织、依赖、复用和适用范围。先建立共享 Contract，再确定 root 与派生路径，完成全部节点并独立终审。原实现的设计、代价及缺口按事实保留；提取不优化或修复 kernel。

本页说明常规流程，[Supervisor 手册](schema_extract/README.md)说明输入、操作及维护接续，阶段页定义具体工作，[REVIEW.md](schema_extract/REVIEW.md)统一审阅方法。catalog 输出位置见 [schema_wiki 导航](schema_wiki/README.md)。

## 提取顺序与交付

| 阶段 | 工作与产物 | 放行与下一步 |
| --- | --- | --- |
| [P1 共享 Contract](schema_extract/phases/01_contract.md) | 从 starter 建立 root.yaml，填写元数据和算法要求，保留硬件占位 | 验收外部计算、有效域、输出与误差依据；Design 可留草稿，进入 P2 |
| [P2 dispatch 草案](schema_extract/phases/02_dispatch_draft.md) | 核对 host 路由，选代表主要 workload 的真实路径作 root，写 dispatch.md 草案和必要的 root when | 审定代表路径、设计划分与覆盖缺口，进入 P3 |
| [P3 root 设计](schema_extract/phases/03_root.md) | 逐段完成数学改写、工作调度、微流水线和必要实现交接 | 完整 root 通过 `--skip-hardware` 检查及独立审阅，进入 P4 |
| [P4 dispatch 定稿](schema_extract/phases/04_dispatch_plan.md) | 据已审 root 定稿 ID、直接父、设计差异和选择依据 | reviewer 批准整树计划，Runner 核验结构化清单，按依赖派发 P5 |
| [P5 派生节点](schema_extract/phases/05_branch.md) | 首次从已审直接父复制，写本节点完整 Design 和适用 Realization；续写保留自己的稿件 | 各节点检查、独立审阅并有效合入后解锁子节点；无分支则跳过 |
| [P6 全算子终审](schema_extract/phases/06_final.md) | 全新 reviewer 核对全树、条件、链接、引用和未决项，必要时交 worker 修改 | 当前计划全部节点有效合入，全目录检查及终审通过才达到 ready，再交付指定目录 |

交付含 root.yaml、全部计划节点 YAML 和中文 dispatch.md；只有 root 时也保留索引。共享 Contract 只在 root 定义，每个节点的 Design 完整，精确条件以各 YAML 为准，parent 表示设计派生关系。[节点与索引规则](schema_docs/0.common.md#派生节点与-dispatch)

Supervisor 决定来源、输出、预算、恢复与工程维护；Runner 唯一写运行状态、管理依赖、调用、checkpoint 和发布。worker 只写许可产物，reviewer 独立审阅，两者不启动子代理。Supervisor 可委派工程维护，并按[角色与会话](schema_extract/README.md#角色与会话)验收和归档。常规 P1–P6 与连续修订的有限维护分别调度，维护完成后继续实际剩余工作，最终仍由当前 P6 交付。

## 七段内容的阅读导航

六阶段负责推进，七段 schema 内容负责表达。提取方法在对应阶段页定义，P5 复用 P3 的逐段方法；字段语义由 schema_docs 定义，操作见[模板说明](schema_template/README.md)。

| 段 | schema 内容 | 方法与规范 |
| --- | --- | --- |
| 1 | meta_data | [P1 元数据](schema_extract/phases/01_contract.md#第-1-段元数据) · [公共约定](schema_docs/0.common.md) |
| 2 | contract.algorithm_requirement | [P1 算法要求](schema_extract/phases/01_contract.md#第-2-段算法要求) · [算法规范](schema_docs/1.contract.algorithm_requirement.md) |
| 3 | contract.hardware_specification | [P1 硬件占位](schema_extract/phases/01_contract.md#第-3-段硬件占位)；原样保留 starter 占位 |
| 4 | design.math_reshape | [P3 数学改写](schema_extract/phases/03_root.md#第-4-段数学改写) · [数学规范](schema_docs/3.design.math_reshape.md) |
| 5 | design.work_schedule | [P3 工作调度](schema_extract/phases/03_root.md#第-5-段工作调度) · [调度规范](schema_docs/4.design.work_schedule.md) |
| 6 | design.tile_pipeline | [P3 微流水线](schema_extract/phases/03_root.md#第-6-段微流水线) · [流水线规范](schema_docs/5.design.tile_pipeline.md) |
| 7 | realization | [P3 实现交接](schema_extract/phases/03_root.md#第-7-段实现交接) · [实现交接规范](schema_docs/6.realization.md) |

**Less is more。** 每段保留会影响读者理解计算、写入边界、调度和同步的事实，已有决定直接引用。源码从 host、IO、路由和调用关系向内展开，读到能写当前段就保存，再按问题回查；复杂段可跨函数、跨层、分轮完成。下层发现前提有误时修正唯一定义处并复查下游，跨阶段修改由 Runner 正式重开。

写作和审阅均使用本次冻结规则，按既有字段及合法扩展表达；规范不适配时提供具体条款、反例和影响，经 Supervisor 正式处理后采用。合法扩展与规范一致性的判断集中在 [REVIEW.md](schema_extract/REVIEW.md)，旧稿及检查通过不能替代规则依据。

reviewer 独立阅读本阶段已完成的全文，包括复制和未改内容，再核对固定源码；初始提示不提供前序批准、旧稿或交接，维护另给本轮问题与范围。批准覆盖独立理解、分层、全文一致和内容必要性，P6 覆盖全树。身份、摘要与精确锚点由工具归一化，模型负责简短非空 summary 和有据的语义判断，见[回执协议](schema_extract/tools/README.md#回执与短交接)。单份 YAML 阅读需要完整 Contract 时生成[正式视图](schema_extract/README.md#单文件阅读视图)，创作侧仍保留唯一 Contract。

## 来源、缺项与验收边界

分别核实输入构造、reference、baseline 和选定实现，来源固定到实际字节；设计证据保留文件、行号和短引文。已有 schema 只辅助阅读，事实回查源码，表达依据冻结规范。[来源与证据](schema_docs/0.common.md#来源与证据) · [三种冻结身份](schema_extract/README.md#输入与输出)

算子已经真实跑过 GPU；当前提取不要求 GPU 正确性验收或实测，也不把它们列为待办。硬件资料在写作、检查和审阅中跳过，starter 硬件占位原样保留，YAML 检查显式用 `--skip-hardware`。检查只证明已实现的结构与关联门禁通过，源码语义仍须独立审阅。[检查边界](schema_template/README.md#校验边界)

缺项是否阻塞，取决于当前阶段是否必须猜测计算、数值或设计。仅缺外部验收器定义而当前提取可确定时，在对应 tolerance 保留[来源记录](schema_docs/1.contract.algorithm_requirement.md#固定实现提取中的验收记录)，说明限制后可继续；关键事实不明则阻塞具体范围，已知事实写错或遗漏则退修。可用材料仍须读取，不无限追索未提供的外部验收器；P6 复核缺口未被用作保证。具体判定统一见 [来源缺项](schema_extract/REVIEW.md#来源缺项)。

实现覆盖不足如实注明，不缩小共享 Contract 或虚构 fallback。有据的低层源缺陷可记录、归档并继续忠实提取，批准仍要求本阶段内容完整、符合规范；缺陷不能为残稿、错误或臆造设计开脱。[源实现缺陷与设计分流](schema_extract/REVIEW.md#源实现缺陷与设计分流)

批准不证明源 kernel 已正确或复现原验收器。确实不同的外部任务才拆成不同算子；来源明确的转录错误可改正，任务要求或真实来源改变按 Supervisor 手册新建提取。

## 框架问题分流

当前算子的实现差异、材料缺项和待修转录写入任务回执。规范、检查器、资料或流程造成错误、遗漏或重复劳动时，先核实已有规则，再用 framework_candidates 提供稳定 ID、可复查反例、最小改动及不改的影响；无合格发现时返回 `[]` 并继续本阶段。

Supervisor 通过[维护入口](schema_extract/issues/README.md)按 ID 和根因查重，记录决定、组织维护和验证，再选普通恢复、兼容运行层升级或连续修订接续任务。worker/reviewer 在算子调用中提报问题，由 Supervisor 处理框架；原回执和必要交接留在运行记录，不增设 schema 字段或固定报告文件。
