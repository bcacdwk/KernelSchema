# 从已有 kernel 提取 schema

一次提取以一个完整算子为单位，将固定实现整理为少量具体完整设计和规范的变体记录，保留数学与数值含义、工作组织、依赖、复用和适用范围。先建立共享 Contract，识别主要机制并选择真实代表，再完成正文、目录和独立终审。原实现的设计、代价与缺口按事实保留；提取不优化或修复 kernel。

本页定义提取方法，[阶段页](schema_extract/phases/01_contract.md)说明具体职责，[REVIEW.md](schema_extract/REVIEW.md)统一审阅方法。当前工具能力及[手工推演与阶段结论](schema_extract/EXECUTOR_ALIGNMENT.md#手工推演与阶段结论)见执行器对齐说明；真实命令、冻结运行和维护操作见 [Supervisor 手册](schema_extract/README.md)。catalog 输出位置见 [Schema Wiki](schema_wiki/README.md)。

## 提取顺序与交付

| 阶段 | 工作与产物 | 放行与下一步 |
| --- | --- | --- |
| [P1 共享 Contract](schema_extract/phases/01_contract.md) | 建立 root 的任务元数据与共享算法 Contract，最小登记有据的执行单元，其余硬件保持占位 | 独立验收外部计算、有效域、全部输出与验收依据；尚不批准 root Design 选择 |
| [P2 机制识别与代表选择](schema_extract/phases/02_dispatch_draft.md) | 有界调查主要机制，选定具体 root 和拟完整代表，记录差异线索与未知边界 | 主要机制、具体 root、拟完整代表及记录去处经独立审定，重要选型疑点已解决；拟完整集合含 root 不超过十份，进入 P3 |
| [P3 root 设计](schema_extract/phases/03_root.md) | 完成一套具体的数学、调度、流水线与必要 Realization，写实自身范围 | 完整 root 经适用结构检查与独立全文审阅，进入 P4 |
| [P4 设计目录与变体记录](schema_extract/phases/04_dispatch_plan.md) | 沿 P2 已定取舍与真实 root 落实目录，为最多九个非 root 明确具体实例、差异、拟共享与待落实位置 | 独立核对完整 root、记录及写作安排是否支持已定判断；未来内容仅获计划认可，按实际依赖进入 P5 |
| [P5 其他完整设计](schema_extract/phases/05_branch.md) | 每次只写自身 YAML；相同整区引用已存在已审的实际来源，其他新事实通过既有交接汇集 | 独立审阅本节点有效全文和全部使用上下文；影响取舍或依赖的变化先返回对应阶段，无非 root 则跳过 |
| [P6 全算子终审](schema_extract/phases/06_final.md) | 新 reviewer 审全部完整设计、目录与新事实去向；按需 worker 集中补记录或限域修稿 | 最终全目录检查和独立终审通过，才可正式交付 |

交付含 raw `root.yaml`、当前选定的其他完整 YAML 和中文 `dispatch.md`；含 root 最多十份完整设计，在保留关键知识前提下越少越好，每份须有独特完整价值。只有 root 也保留目录。共享 Contract 只在 root 定义，root Design 三区全本地，其他设计通过本地定义与实际整区 `from` 形成完整有效内容。参数、局部变化与已知组合按 [common 的唯一记录格式](schema_docs/0.common.md#选择条件与索引)整理，不因此逐条取得完整 YAML 身份。

Supervisor 在授权目标内决定来源、输出、预算、恢复和必要维护；Runner 唯一写运行状态、调用、checkpoint、合入与发布。worker 只写许可产物，reviewer 独立审阅，两者不启动子代理。Supervisor 的委派与维护职责见[角色与会话](schema_extract/README.md#角色与会话)；有限维护不替代尚未完成的阶段和最终 P6。

## 七段内容的阅读导航

六阶段负责推进，七段 schema 内容负责表达。P5 复用 P3 的逐段方法，字段语义由 schema_docs 定义，文件操作见[模板说明](schema_template/README.md)。

| 段 | schema 内容 | 方法与规范 |
| --- | --- | --- |
| 1 | meta_data | [P1 元数据](schema_extract/phases/01_contract.md#第-1-段元数据) · [公共约定](schema_docs/0.common.md) |
| 2 | contract.algorithm_requirement | [P1 算法要求](schema_extract/phases/01_contract.md#第-2-段算法要求) · [算法规范](schema_docs/1.contract.algorithm_requirement.md) |
| 3 | contract.hardware_specification | [P1 执行单元与硬件占位](schema_extract/phases/01_contract.md#第-3-段硬件占位)；执行单元按已有字段登记，其余保留 starter 占位 |
| 4 | design.math_reshape | [P3 数学改写](schema_extract/phases/03_root.md#第-4-段数学改写) · [数学规范](schema_docs/3.design.math_reshape.md) |
| 5 | design.work_schedule | [P3 工作调度](schema_extract/phases/03_root.md#第-5-段工作调度) · [调度规范](schema_docs/4.design.work_schedule.md) |
| 6 | design.tile_pipeline | [P3 微流水线](schema_extract/phases/03_root.md#第-6-段微流水线) · [流水线规范](schema_docs/5.design.tile_pipeline.md) |
| 7 | realization | [P3 实现交接](schema_extract/phases/03_root.md#第-7-段实现交接) · [实现交接规范](schema_docs/6.realization.md) |

每段保留影响计算、写入边界、调度与同步的事实，已有定义直接引用。P2 先建立机制概览；写正文时围绕当前问题深入源码，能写清本段就保存，再按具体疑点回查。复杂段可跨函数、跨层、分轮完成。前提有误时修正唯一定义并复核真实使用者，返回分工见 [REVIEW](schema_extract/REVIEW.md#发现问题与批准)。

写作和审阅依据本任务适用规则，按已有字段与合法扩展表达；规范不适配时提供具体条款、反例和影响，由 Supervisor 处理。规则一致性的判断见 [REVIEW.md](schema_extract/REVIEW.md)，旧稿和检查器接受不能替代依据。

reviewer 独立阅读本阶段已完成的有效全文，包括共享和未改内容，再核对固定源码。同一当前来源定义可读一次，每个使用节点的范围、绑定、数值、接口、寿命与 Realization 仍全覆盖。P4 读完整 root，P5 审本节点完整设计，P6 审全部最终设计及记录；历史批准和交接不补足正文。单文件[阅读视图](schema_docs/0.common.md#单文件阅读视图)与 raw 创作目录分开，工具操作见[手册](schema_extract/README.md#单文件阅读视图)。

## 来源、缺项与验收边界

分别核实输入构造、reference、baseline 与固定实现，来源固定到实际字节；设计证据保留文件、行号和短引文。已有 schema 只辅助阅读，事实回查源码，表达依据本任务规则。[来源与证据](schema_docs/0.common.md#来源与证据) · [三种冻结身份](schema_extract/README.md#输入与输出)

固定实现提取不要求 GPU 正确性验收或实测，也不把它们列为待办；已有测试和测量只按实际覆盖范围保留。硬件区按 [P1](schema_extract/phases/01_contract.md#第-3-段硬件占位)仅登记给定材料明确的执行单元，其余保持占位，不开展硬件资料、能力或容量调查。适用检查沿用 `--skip-hardware` 政策，有据登记由 worker/reviewer 对照来源核实。检查只证明所用工具实际覆盖的结构与关联门禁通过，源码语义仍须独立审阅。这些豁免不改变普通设计、实现或优化任务自己的验收要求。[检查边界](schema_template/README.md#校验边界)

缺项是否阻塞，取决于当前阶段是否必须猜测计算、数值或设计。仅缺外部验收器定义而当前提取可确定时，在对应 tolerance 保留[来源记录](schema_docs/1.contract.algorithm_requirement.md#固定实现提取中的验收记录)，说明限制后可继续；关键事实不明则阻塞具体范围，已知事实写错或遗漏则退修。可用材料仍须读取，不无限追索未提供的外部验收器；P6 复核缺口未被用作保证。具体判定见[来源缺项](schema_extract/REVIEW.md#来源缺项)。

实现覆盖不足如实注明，不缩小共享 Contract 或虚构 fallback。有据的低层源缺陷可记录、归档并继续忠实提取，批准仍要求本阶段内容完整且符合规范；缺陷不能为残稿、错误或臆造设计开脱。[源实现缺陷与设计分流](schema_extract/REVIEW.md#源实现缺陷与设计分流)

批准不证明源 kernel 已正确或复现原验收器。确实不同的外部任务才拆成不同算子；来源明确的转录错误可改正，任务要求或真实来源改变按 Supervisor 手册新建提取。

## 框架问题分流

当前算子的实现差异、材料缺项和待修转录写入既有回执与问题记录。规范、检查器、资料或流程造成错误、遗漏或重复劳动时，先核实已有规则，再按[回执协议](schema_extract/tools/README.md#回执与短交接)提供稳定标识、可复查反例、最小改动及不改的影响；无合格发现则继续当前工作。

Supervisor 通过[维护入口](schema_extract/issues/README.md)按标识和根因查重，记录决定、组织维护与验证，再依据实际工具能力选用适当接续。worker/reviewer 在算子调用中提报问题，原回执与必要交接留在运行记录，不增设 schema 字段或固定报告文件。
