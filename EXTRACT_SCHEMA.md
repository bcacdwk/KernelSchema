# 从已有 kernel 提取 schema

一次提取以一个完整算子为单位，将固定实现整理为少量具体完整设计和规范的变体记录，保留数学与数值含义、工作组织、依赖、复用和适用范围。先建立共享 Contract，识别主要机制并选择真实代表，再完成正文、目录和独立终审。原实现的设计、代价与缺口按事实保留；提取不优化或修复 kernel。

本页定义提取方法，[阶段页](schema_extract/prompt/phases/01_contract.md)说明各阶段职责，[REVIEW.md](schema_extract/REVIEW.md)统一独立审阅方法与[阶段结论边界](schema_extract/REVIEW.md#各阶段覆盖范围)。唯一工具入口为 `schema_extract/extract_yaml.py`（默认 Claude Code `qwen3.8-max[1m]` / `max`，Codex 可显式指定；每阶段、角色与 P5 节点建立独立会话）。Linux 运行环境、命令、冻结身份与维护操作见 [Supervisor 手册](schema_extract/README.md)，角色工具权限与执行边界见[执行协议](schema_extract/tools/README.md#运行恢复与维护)，catalog 输出位置见 [Schema Wiki](schema_wiki/README.md)。

## 提取顺序与交付

| 阶段 | 工作与产物 | 放行与下一步 |
| --- | --- | --- |
| [P1 共享 Contract](schema_extract/prompt/phases/01_contract.md) | 建立 root 的任务元数据与共享算法 Contract，最小登记有据的执行单元，其余硬件保持占位 | 独立验收外部计算、有效域、全部输出与验收依据；尚不批准 root Design 选择 |
| [P2 机制识别与代表选择](schema_extract/prompt/phases/02_dispatch_draft.md) | 有界调查主要机制，选定具体 root 和拟完整代表，记录差异线索与未知边界 | 主要机制、具体 root、拟完整代表及记录去处经独立审定，重要选型疑点已解决；拟完整集合含 root 不超过十份，进入 P3 |
| [P3 root 设计](schema_extract/prompt/phases/03_root.md) | 完成一套具体的数学、调度、流水线与必要 Realization，写实自身范围 | 完整 root 经适用结构检查与独立全文审阅，进入 P4 |
| [P4 设计目录与变体记录](schema_extract/prompt/phases/04_dispatch_plan.md) | 沿 P2 已定取舍与真实 root 落实目录，为最多九个非 root 明确具体实例、差异、拟共享与待落实位置 | 独立核对完整 root、记录及写作安排是否支持已定判断；未来内容仅获计划认可，按实际依赖进入 P5 |
| [P5 其他完整设计](schema_extract/prompt/phases/05_branch.md) | 每次只写自身 YAML；相同整区引用已存在已审的实际来源，其他新事实通过既有交接汇集 | 独立审阅本节点有效全文和全部使用上下文；影响取舍或依赖的变化先返回对应阶段，无非 root 则跳过 |
| [P6 全算子终审](schema_extract/prompt/phases/06_final.md) | 新 reviewer 审全部完整设计、目录与新事实去向；按需 worker 集中补记录或限域修稿 | 最终全目录检查和独立终审通过后达到 `ready`；无屏障时由 Runner 原子发布 |

交付含 raw `root.yaml`、当前选定的其他完整 YAML、中文 `dispatch.md` 和 `.extraction/receipt.json`；含 root 最多十份完整设计，在保留关键知识前提下越少越好，每份须有独特完整价值。只有 root 也保留目录。共享 Contract 只在 root 定义，root Design 三区全本地，其他设计通过本地定义与实际整区 `from` 形成完整有效内容。参数、局部变化与已知组合按 [common 的唯一记录格式](schema_docs/0.common.md#选择条件与索引)整理，不因此逐条取得完整 YAML 身份。

Supervisor 在授权目标内决定来源、输出、调用次数、恢复和限域维护请求；Runner 唯一写运行状态、调用结算、checkpoint、合入与原子发布。worker 只写本次许可产物，reviewer 始终只读，阶段调用均不委派子代理（见[角色与会话](schema_extract/README.md#角色与会话)）。中断恢复先用 `status RUN_UUID --full` 核实停止原因与在途调用结清（终止未确认保持 fence），沿用原冻结身份与已用次数接续；未交付固定 run 的 `maintain` 仅针对已选 P1–P5 且须显式给出 `fields` 上限（`fields: []` 的 `review` 仅做只读复核，进入退修 `patch` 须非空），不替代未完成阶段与最终新的 P6。

`ready` 仅表示当前 P6 终审通过，不等于已发布；无控制或维护屏障时由 Runner 自动核验并原子发布（`published: true`）。若接手时 `ready` 但 `published: false`，先用 `inspect --run RUN_UUID --json` 核对 `publication`、`control`、`global_hold` 与 `maintenance_request`（权威运行状态及已应用的 `maintenance` 详情查 `status RUN_UUID --full`），再按[恢复与交付](schema_extract/README.md#恢复与交付)和[发布协议](schema_extract/tools/README.md#发布与独立视图)释放屏障、`resume` 重放发布，或在目标冲突时保留双方内容并用 `export` 另交新目录。

## 七段内容的阅读导航

六阶段负责推进，七段 schema 内容负责表达。P5 复用 P3 的逐段方法，字段语义由 schema_docs 定义，文件操作见[模板说明](schema_template/README.md)。

| 段 | schema 内容 | 方法与规范 |
| --- | --- | --- |
| 1 | META_DATA | [P1 元数据](schema_extract/prompt/phases/01_contract.md#第-1-段元数据) · [公共约定](schema_docs/0.common.md) |
| 2 | CONTRACT.Algorithm_requirement | [P1 算法要求](schema_extract/prompt/phases/01_contract.md#第-2-段算法要求) · [算法规范](schema_docs/1.contract.algorithm_requirement.md) |
| 3 | CONTRACT.Hardware_specification | [P1 执行单元与硬件占位](schema_extract/prompt/phases/01_contract.md#第-3-段硬件占位)；执行单元按已有字段登记，其余保留 starter 占位 |
| 4 | DESIGN.Math_reshape | [P3 数学改写](schema_extract/prompt/phases/03_root.md#第-4-段数学改写) · [数学规范](schema_docs/3.design.math_reshape.md) |
| 5 | DESIGN.Work_schedule | [P3 工作调度](schema_extract/prompt/phases/03_root.md#第-5-段工作调度) · [调度规范](schema_docs/4.design.work_schedule.md) |
| 6 | DESIGN.Tile_pipeline | [P3 微流水线](schema_extract/prompt/phases/03_root.md#第-6-段微流水线) · [流水线规范](schema_docs/5.design.tile_pipeline.md) |
| 7 | REALIZATION | [P3 实现交接](schema_extract/prompt/phases/03_root.md#第-7-段实现交接) · [实现交接规范](schema_docs/6.realization.md) |

每段保留影响计算、写入边界、调度与同步的事实，已有定义直接引用。P2 先建立机制概览；写正文时围绕当前问题深入源码，能写清本段就保存，再按具体疑点回查。复杂段可跨函数、跨层、分轮完成。前提有误时修正唯一定义并复核真实使用者，返回分工见 [REVIEW](schema_extract/REVIEW.md#发现问题与批准)。

写作和审阅依据本任务冻结规则，按已有字段与合法扩展表达。模型自检使用本次固定只读 helper 预览规范化反馈、精简超长正文并处理诊断（手动调用[统一检查入口](schema_template/README.md#4-检查与处理反馈)时显式加 `--read-only --skip-hardware`）；Runner 保存原稿后整体规范化候选，核对权限与来源并严格重读落地 raw，再交独立 reviewer（见[规范化与检查](schema_extract/tools/README.md#规范化与检查)）。规范不适配时提供具体条款、反例和影响交 Supervisor 处理；规则一致性见 [REVIEW.md](schema_extract/REVIEW.md)，旧稿和检查器接受不能替代依据。

reviewer 独立阅读本阶段已完成的有效全文（含共享和未改内容），再核对固定源码；同一当前来源定义可读一次，每个使用节点的范围、绑定、数值、接口、寿命与本地 `REALIZATION` 仍全覆盖，历史批准和交接不补足正文。单文件[阅读视图](schema_docs/0.common.md#单文件阅读视图)与 raw 创作目录分开，导出操作见[手册](schema_extract/README.md#单文件阅读视图)（`standalone` 完整检查来源目录与导出单文件，不接收运行 `selected_designs`，也不产生新的阶段批准）。

## 来源、缺项与验收边界

分别核实输入构造、reference、baseline 与固定实现，来源固定到实际字节；设计证据保留文件、行号和短引文。已有 schema 只辅助阅读，事实回查源码，表达依据本任务规则。[来源与证据](schema_docs/0.common.md#来源与证据) · [三种冻结身份](schema_extract/README.md#输入与输出)

固定实现提取不要求 GPU 正确性验收或实测，也不把它们列为待办；已有测试和测量只按实际覆盖范围保留。硬件区按 [P1](schema_extract/prompt/phases/01_contract.md#第-3-段硬件占位)仅登记给定材料明确的执行单元，其余保持占位，不开展硬件资料、能力或容量调查。适用检查显式使用 `--read-only --skip-hardware`，有据登记由 worker/reviewer 对照来源核实。检查只证明所用工具实际覆盖的结构与关联门禁通过，源码语义仍须独立审阅。这些豁免不改变普通设计、实现或优化任务自己的验收要求。[检查边界](schema_template/README.md#校验边界)

缺项是否阻塞，取决于当前阶段是否必须猜测计算、数值或设计。来源明确给出完整比较公式时如实写出；若任务数学、IO、副作用和参考入口可确定，而外部验收器未提供完整公式、相对误差基准、归一化或检查路由（如仅给出 `atol/rtol` 或 `relative_to` 等局部标签），则在对应 `Golden_reference.Tolerance` 保留[验收来源记录](schema_docs/1.contract.algorithm_requirement.md#固定实现提取中的验收记录)，写清已知项与缺失定义并注明不套用默认简写解释（不从局部标签自推组合公式，也不挪入 `Gap`），说明限制后可继续。关键事实不明则阻塞具体范围，已知事实写错或遗漏则退修；可用材料仍须读取，P6 复核缺口未被用作保证。具体判定见[来源缺项](schema_extract/REVIEW.md#来源缺项)。

实现覆盖不足如实注明，不缩小共享 Contract 或虚构 fallback。有据的低层源缺陷可记录、归档并继续忠实提取，批准仍要求本阶段内容完整且符合规范；缺陷不能为残稿、错误或臆造设计开脱。[源实现缺陷与设计分流](schema_extract/REVIEW.md#源实现缺陷与设计分流)

批准不证明源 kernel 已正确或已复现原验收器；提取产物后续若用于新建设计、优化误差预算或实现验收，仍须由任务方给出或确认完整判据。确实不同的外部任务才拆成不同算子；来源明确的转录错误可改正，真实外部任务或固定来源改变时按[定点返回](schema_extract/prompt/operations/reopen.md)交 Supervisor 另建运行。

## 框架问题分流

当前算子的实现差异、材料缺项和待修转录写入既有回执与问题记录。规范、检查器、资料或流程造成错误、遗漏或重复劳动时，先核实已有规则，再按[回执协议](schema_extract/tools/README.md#回执与短交接)提供稳定标识、可复查反例、最小改动及不改的影响；无合格发现则继续当前工作。

Supervisor 通过[维护入口](schema_extract/issues/README.md)按稳定标识与根因查重，记录决定并组织修复与验证，未交付固定 run 按明确文件/字段范围维护；任务、来源或固定规则改变时另建运行。worker/reviewer 在算子调用中提报候选，原回执与交接留在运行记录，不增设 schema 字段或固定报告文件。
