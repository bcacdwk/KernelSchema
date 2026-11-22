# 四流程与 AKA 接入：初步架构和推进大纲

日期：2026-11-20。状态：三轮六份研究后的独立整合，供主代理汇报与用户判断；**不是已批准的实施计划**。
本文区分已核实事实、推荐方向和待验证假设；接口、支持范围、阶段安排均为拟议，不代表已有功能或用户已作决定。

后续状态：用户已回复“按推荐开始”，确认 D1–D3；详见[方向决策](03_decision_questions.md)与历史 H010。本文保留为当时的初步大纲，不逐段改写其待决措辞；后两轮及最终整合现已完成，执行入口为[完整计划](04_implementation_plan.md)与[工作包清单](05_work_packages.md)。

## 1. 先回答核心纠结

**建议 KernelSchema 拥有四个真实能力入口与 Schema 语义，AKA 提供可选的增强执行和编排，AKA_experiment 消费固定版本并做实验。**
KernelSchema 的 implement 可先是一个受限但能真正生成、验证、保存和续写代码的参考后端；它不需要承担 AKA 的全部长程运行设施。
design/optimize 在 KS 中成为可调用的 planner 能力，extract 保留现有 Runner；AKA 开启 Schema 模式时调用这些能力，并把 coder 任务改为实现固定设计。
这里的关键不是多一个规划 prompt，而是让固定设计、代码起点、验收证据、预算和下一步行动有明确归属。

用户已明确要求：先研究与初纲再决策、保留 extract、模块轻量可选、KS 有独立公开价值，以及区分设计迭代与 coder 迭代。
用户尚未决定：独立价值是否必须包含无需 AKA 后端的真实执行、首发支持多宽，以及 planner 按实现阶段还是 episode 周期介入。
推荐采用“受限独立路径 + 后续 AKA 增强”的分期方向；本轮不据此承诺同时维护两套全功能执行系统。

## 2. 三仓分工与单向依赖

| 仓库 | 推荐拥有的职责 | 对外可观察价值 |
| --- | --- | --- |
| KernelSchema（KS） | 语言/Template、四流程入口、共享写作与交接语义、参考 implement、有限 demo | 下载者能设计、提取、优化、实现声明范围内的任务，看到真实结果和未完成项 |
| AKA 适配分支/fork | 可选 Schema 模块、固定请求 backend、coder 材料/工具接线、原生执行与恢复 | 复用 Agent、源码工作区、Evaluation/Experiment/Execution、episode 和晋升设施 |
| AKA_experiment（EXP） | 固定依赖、数据集/矩阵、排程、公平预算、实验资格和结果分析 | 比较无 Schema、固定 Schema、动态修订的效果与成本，保留可复现来源 |

建议的源码依赖关系如下；fork、submodule 和消费方式仍待方向确认后落实：

```text
AKA_experiment ──固定版本──> AKA 适配 fork ──可选 submodule──> KernelSchema
KernelSchema 核心/参考 implement ──> Python、模型运行时、任务 evaluator
```

KS 不反向嵌入 AKA，不以 AKA 为基础安装/导入的必需依赖。
KS 用户选择 AKA backend 时，可以向外部已安装 adapter 发固定实现请求；这是可选运行调用，不产生相互 submodule 或递归启动。
每次运行只有一个循环所有者：KS demo 调固定 implement，或 AKA Schema 模块调 KS planner；二者不互相启动顶层循环。
EXP 当前是普通文件 vendor + overlay，不能把图中的关系写成现状；其投影会排除 gitlink 与 `.gitmodules`，未来必须显式物化 KS 或改用固定 checkout。
迁移应在独立工作树中核对现有成果与补丁归属，保留旧捕获及来源台账；上游升级与功能迁移分开，不用清理当前脏树换取“干净基线”。

## 3. 当前事实：已有基础足够复用，缺口也很具体

| 研究核实的事实 | 对本次规划的约束 |
| --- | --- |
| KS 的 extract 已有 P1–P6、审阅、恢复、发布；其他三项仍是角色方法文档 | 建设三个可调用能力，外层接收 extract 交付；不重做提取内部 |
| Template 已有只读检查、规范化规划、来源闭包和 effective 视图 | 直接复用；不另造 parser、展开器或语义等价 hash |
| Wiki/catalog 有真实资料与过滤工具，案例并非统一 GPU demo | 可以支撑参考选择；不能把目录检索说成现成近邻规划，也不能跳过任务材料资格 |
| AKA 有成熟的局部执行事务，但没有 Schema revision 生命周期 | 动态 planner、固定设计 prompt、partial 续接、fidelity 和结果映射仍需接入 |
| EXP 已有 Claude 的固定材料注入，依赖本地 role/context/session 补丁 | 可选择性借鉴；不是四流程、跨 revision 编排或真正 schema-off 的完成证据 |

KS 基线为 `4849028`，独立 AKA 为 `bf5557a`；EXP 的 HEAD 与活动工作树差异见 [baseline.json](baseline.json)。既有提取和实验变更不属于本任务。
R1 使用既有环境完成了有限 Template 只读验证；没有重演模型/GPU 流程。EXP 的 vendor verify 被缓存拒绝，辅助比对也不等于官方验证通过。
本整合只读材料并写初纲，没有新增模型、GPU、测试、远端或实现层验证结论。

依据：[R1 KS 现状](research/r1_schema_inventory.md)、[R1 AKA/EXP 现状](research/r1_aka_inventory.md)；[R2 独立契约](research/r2_standalone_contracts.md)、[R2 AKA 生命周期](research/r2_aka_lifecycle.md)；[R3 交付/实验审查](research/r3_delivery_evaluation.md)、[R3 反例审查](research/r3_architecture_challenge.md)。
关键现有代码：[Template 身份/有效视图](../../schema_template/tools/design_resolution.py#L29)、[extract 发布](../../schema_extract/tools/publication.py#L71)、[AKA 新 episode 起点](../../../atrex-kernel-agent/src/aka/workflow/light/long_horizon/campaign.py#L1785)、[EXP 固定 SchemaAgent](../../../AKA_experiment/tools/python/aka_experiment/schema.py#L158)。

## 4. 四流程怎样真正落在 KS

先有轻量的任务准备：固定 ABI、输入域、全部输出/副作用、reference 与完整误差公式、硬件/栈资料、要求实现范围、workload、目标和计时边界。
Task 可来自已授权的 benchmark manifest 或项目配置，不必每次人工确认；缺关键条件时返回缺项，只停止依赖它的工作。
任务提供构建/验收/测量入口；模型不能通过修改 evaluator 换取通过，Agate 也不能代替任务语义与判据。

| 流程 | 拟议输入 | 拟议输出与完成含义 |
| --- | --- | --- |
| design | 已确认 Task、规则、选定参考及可选 Contract 起点 | 完整 raw/dispatch、来源与有效视图、实现交接和待实测假设；必要设计决定与规定检查完成，可交 coder |
| extract | 现行 Runner 支持的固定来源和入口/验收绑定 | 保留原 raw、dispatch、run/receipt；完成仍按原提取政策，ready 与是否成功 published 分开 |
| optimize | 固定 Task、当前完整 Schema、目标及可选代码/实测/普通反馈 | `revise` 给新快照及变更影响，`keep` 给保留理由；两者均可完成，不强迫改 YAML |
| implement | 固定 Task、Schema/规则/范围、可空代码起点、后端与预算 | 精确代码、覆盖进度、构建结果、三条证据、反馈及可选续接；只有所选范围与规定验收完成才报完成 |

共同结果可先用 `completed / partial / blocked / failed` 表达交付摘要，再分别报告检查、停止原因和消耗；它不替代 extract/AKA 原生生命周期。
预算结束而留下代码通常是 partial；代码可交但 GPU 未执行时仍是 partial，相关检查为 not_run，不能由退出码 0 推出实现成功。
性能目标单报 `met / unmet / unknown`：正确、符合设计且已按要求测量，但比比较对象慢的代码仍可成为合格实现；项目优化目标继续未达。
任务明确规定为硬门槛的条件仍须满足；“允许交付较慢代码”不授权降低 Contract 或验收要求。

extract 的硬件占位、skip-hardware 和允许缺失的外部验收公式必须保留为提取语义；**转入实现前重新核对任务条件**，保留原件/回执并形成实现输入快照。
手写、design、人工审修和 Runner 产物都可作为来源；P6 receipt 不是统一入场证。既有复杂 Schema 的材料预审应提前进行。

## 5. 独立 implement 到哪里为止

“独立前端”只要求从 KS 调到真实能力；“无需 AKA 运行后端”还要求 KS 在支持环境中自己完成生成与验证。托管 AKA 服务也属于后端依赖。
推荐参考后端先支持单任务、单栈、一种模型运行时与一种任务 evaluator，串行调用 coder/验证，保存精确源码和累计预算。
原生模型运行时负责会话内工具循环；KS 薄控制器负责固定输入、有限继续、结果解释和交付，不新增通用 agent 平台。
最小续接承诺是从完整 C1 字节开新会话继续，包括文件删除与来源；跨机器/provider 搬运模型隐藏上下文不作为首发承诺。
若执行适配器无法结算未知远端作业，就返回受阻，不盲目重提；本地还是 Agate、具体栈和环境按后续选题确定，本初纲不预先删除 Agate 需求。

| 可接受替代 | 得到什么 | 代价与适用条件 |
| --- | --- | --- |
| 仅 KS 前端 + 外部 AKA backend | 少维护一个参考执行器，已有 AKA 用户更快接入 | 必须公开 AKA 运行依赖；固定设计、续写、验收适配仍不能省 |
| 先受限参考 implement，再加 AKA | 先兑现无需 AKA 的独立能力，后续增强长程执行 | 多维护一条窄路径；若迫使复制复杂隔离/调度/晋升，应缩小范围或重新选择前者 |

KS 不承担 ABBA/incumbent 晋升、多任务排程、多 trajectory、自动方向扩张、通用远端恢复或所有编程栈。
AKA 也不维护第二份 Schema 规则、planner 或解析器；共享边界只涵盖请求、产物、证据和续接含义。

## 6. 固定设计可以实现很多轮，设计修改是另一条轴

下面 R 表示固定设计及本次范围，C 表示精确代码，G 表示 AKA 保留的全局 incumbent；它们不是要求新建的数据库对象。

```text
任务条件就绪 + [已有 Schema / design / extract 交付后补齐条件]
  → 固定 R 与实现请求 → coder 实现/验证 → 保存 C 与绑定证据
      ├─ partial / 可修代码缺陷 → 同 R 从 C 继续（可跨多个 episode）↺
      ├─ 覆盖与验收合格 → 交付当前 R 的实现；性能目标与 G 晋升另报
      │    ├─ implement-only / 目标已达 → 停止
      │    └─ 已授权动态循环且有依据/预算 → optimize
      ├─ 可核对的设计冲突 → optimize 或报告缺项/未决
      └─ 外部故障 / 暂停 / 预算结束 → 结算并返回已有产物
optimize → revise：发布并准入 R'、明确代码起点 → 新固定实现请求
         → keep：完成优化操作；有具体行动才续写/补证，否则停止
```

同一 R 的第一次实现可以比后续修订获得更大总预算；不要求每个 episode 后调用一次 planner，也不把无晋升当作实现无进展。
同 episode 优先沿原生会话恢复；终端 episode 后的 partial 需显式作为下个 episode 起点，不能默认从 G 重来。
分别保留最新稳定代码、当前 R 的合格代码与 G；它们可以不同。新 R 可显式复用旧代码，重新判定受影响证据，不清空 G。
只需核验已有代码时走 implement 内部的无改码验收路径；原生 single-file 会拒绝无源码变化的 seal，不能靠加注释制造成功 episode。
编译错误、覆盖缺口、fidelity 失败通常先交 coder；服务超时或一次“写不出来”不等于设计不可实现。
暂停与全局预算约束 planner、新请求和新 episode 的准入；续接、keep、恢复和换 R 都不能退款或重置总额度。

## 7. 最小共享内容、权限与三条证据

拟议交接可小到 `run(固定请求、代码起点、限额) → 产物、证据、停止原因、可选续接`；这里只示意含义，不定 API 名或序列化字段。
共享内容为：Task/Schema 快照及范围、资料组装、现有 Template 检查、角色权限核对、普通反馈和用量含义；后端声明实际支持能力。
design/optimize 共用写作内核：读规则和有限参考→草稿→规范化候选→权限/全任务检查→受影响语义审阅→交付。现有角色文档继续作为方法真源。
参考先用 catalog/明确目录筛选并说明适用性；暂不建设 embedding 平台，不复制 extract 的 P1–P6/维护系统，也不先抽出万能 SDK。

身份至少同时绑定 Task、规则/工具版本、raw 来源闭包、dispatch、选定范围、精确 C 和 evaluator/环境/计时口径。
现有 effective_identity 不含全部这些内容，且会受 raw 排版/Feedback 字节影响；consumer Version 也不随 provider 改动自动增加。
因此新字节快照不自动等于一次设计优化；旧证据保留原绑定，经明确核对可关联复用，不能改标签冒充新测量。
非 root 的自包含交接使用现有 effective_document/effective_yaml，包含 root Contract；不把缺 Contract 的内部映射直接交给 coder。
root 是阅读主干，不是通用 fallback；dispatch 是设计目录，不是执行路由。先贯通适用路径，再补本次明确选择的必要节点/变化，返回总任务未覆盖项。
FA4 decode 的 root 只覆盖共享 Contract 的子域，正好说明“root 完成”不能推出任务完成。

| 证据 | 应回答的问题 | 初期接受的边界 |
| --- | --- | --- |
| 正确性 | 精确 C 是否按固定判据满足输出/副作用与规定输入 | 真实运行，明确有限覆盖；QSA 的历史与正式误差口径差异须核对 |
| fidelity（设计符合性） | 数值路径、工作组织、同步/寿命及 Require/Forbid 是否落实 | 可机械核对项 + 独立有限审阅，逐项 pass/fail/unknown；未知不能代签完成 |
| 性能 | 同一环境/输入/计时范围下，这份 C 的实际表现是什么 | 绑定比较源码和测量波动；更快不授权改设计，更慢不抹去合格产物 |

Planner 只修改已授权的 Design/实现约束；Coder 固定实现 R，可只读查询 docs、raw/effective、来源和参考，不能修改 Contract/Design/Require/Forbid。
AKA coder 的主任务正文、工具、report 修复和所有续会话路径都须遵守该边界；追加材料无法抵消原有自由换算法/自动换栈的指令。
一般 bug、阻碍、预算、覆盖缺口、瓶颈和未实测建议留在外部结果；仅符合现行 Prefer 替代规则且有 `repository@ref:path` 不可变实测证据的条目，才可作为 Feedback 提案核对后写入。
只读投递、checker 通过和 coder 自评都不构成 fidelity。测过 C1 后改成 C2，C2 不能继承那次 pass；同字节也须核对 Task/evaluator 是否改变。

## 8. AKA 接入：保留模块边界，先验证驱动假设

可选 Workflow provider 是开关和职责边界；其内部用单 episode façade，还是给原循环增加有限准入/终态接口，尚不能排序。
两者都须处理固定任务/只读材料、source seed、准确候选资格、终态投影和恢复；不能把静态 decorator 当作完整接入。
优先复用原生 Task、episode/session、draft、测量收据和 promotion audit；新增不可变 Schema 发布与固定请求关联即可，不把 R2 的所有概念各建一套权威账本。

R3 对 R2 的修正收束为以下技术验证；当前均未运行，失败时应调整或缩小方案，而不是降低交付语义：

| 待验证的边界 | 已发现的具体约束 | 失败退路 |
| --- | --- | --- |
| 指定 episode 与累计预算 | `max_version/max_episodes` 的 if/elif 不同时约束；`episode_limit=1` 在恢复结算后可能再开 e+1，Search 先例未证明这些语义 | 保留 provider，改用原 loop 的窄准入/终态接口；始终核对指定 e、全局预算与暂停 |
| partial 导入与源码身份 | exploring 可先落盘、seed 尚未装齐；异常回滚不证明宿主崩溃安全。public-candidate 的 single-file 产物含 solution.json，不能全量当可导入 source | 以导入意图/摘要和完整安装事实控制首次准入，复用 source codec；必要时首发只支持一种格式，已准入 draft 不重装 |
| 初次实现的准备与收尾 | `never` 还跳过 V1 的较慢首版接纳、pin/metadata 等职责；production 收尾、V0 reviewer、既有栈约束也不能忽略 | 按支持任务保留必要 helper/门禁，分别交付 C 与 G；不足时接 V1 或缩小任务范围 |
| fidelity 与晋升恢复 | 普通/未提交恢复可共用 seal 后、ABBA 前检查；已提交晋升走 audit，不会重进该点。任意 active 字段不会自动进入晋升证据 | 资格收据必须绑定准确 C 并持久进入晋升 audit；缺证据保留已提交事实、报告受阻，未知资格不准入晋升 |
| 动态 R 的准入与证据 | 冻结配置不能热改；“R' 已发布”不同于“下一实现请求已准入”。旧事务、Task/dispatch/来源变化不可按最新路径猜测 | 先稳定固定 R backend；动态修订暂作为下一次显式请求，拒绝不相容 continuation |
| 可选分发与原生关闭路径 | EXP vendor 不带 KS gitlink；现有 control 仍运行适配包装，不能证明 off | 显式物化固定资源、延迟加载；无 KS 时原路径仍可启动，不引入 Schema 工具/调用/费用 |

代码锚点：[限额/恢复循环](../../../atrex-kernel-agent/src/aka/workflow/light/long_horizon/campaign.py#L1698)、[source codec/导入](../../../atrex-kernel-agent/src/aka/task/candidate_workspace/source.py#L147)、[终态与 audit](../../../atrex-kernel-agent/src/aka/workflow/light/long_horizon/campaign.py#L1162)。反例展开见 R3 反例报告 §§3–8。
还须区分 Git HEAD 与 G 的源码身份：未晋升 episode 也可能提交 vN 元数据并推进 canonical HEAD；下一 candidate 取当时 HEAD，比较源码仍可不变。
原生 production 恢复前置可能重新调用 reviewer；只能对同一收据重放去重，真实再次调用必须记新增成本。新增 fidelity 的恢复去重须单独验证，不能承诺整个原生恢复零评审。
KS/AKA 都分别记录 planner、coder、reviewer/fidelity、extract 和执行的实际成本；未知用量不填零，原生 token 限额也不冒称调用内硬停。

## 9. 建议分期：三个门槛分开推进

首个工程切片证明状态/接缝，公开 demo 证明真实产品能力，受控实验才支持收益结论。以下为阶段依赖与可观察价值，不是文件级任务清单或工期承诺。

| 阶段 | 依赖与重点 | 用户能看到的价值/进入后续的依据 |
| --- | --- | --- |
| A：材料与最小交接 | 用户选定独立程度/支持面；小任务材料闭合，同时预审一项现有复杂 Schema | 明确能做什么、缺什么；避免 toy 协议长期脱离真实资产 |
| B：固定 R 实现/续写 | A；参考后端先跑窄路径，AKA 接缝可独立验证 | 准确 C1→C2、有限循环、真实验收、合格慢代码交付；先分清 coder 与 planner 问题 |
| C：尽早接公开四流程 demo | B 的独立路径；接真实 design、一次 optimize，并展示 extract 交付转入 | 用户调用入口可重跑；keep/revise 与失败返回真实，demo 不等待 AKA 动态编排 |
| D：真实既有 Schema 与必要 dispatch | A 的复杂材料预审，随后跟进 B/C；按资料资格选题 | 检验真实设计消费、范围与隐含前提；复杂 fidelity 的 unknown 如实保留 |
| E：AKA 固定 backend→动态编排 | 通过 §8 的适用验证，再接 KS planner 与 revision 准入 | 已有 AKA 用户复用长程设施；不以动态循环阻塞独立交付 |
| F：EXP 消费与收益研究 | 固定发行配对、独立/AKA 能力证据及统一任务/预算政策 | 形成范围明确、可解释的效果与成本比较；不拿旧静态注入结果代替新实验 |

A 的真实题预审应提前；D 的真实实现紧随独立小路径，不必等 E 全部完成。具体题目、设备、模型、预算和里程碑由后续计划确定。
EXP 核心对照可先无 Schema/固定 Schema/动态 Schema，统一总预算、资料、执行策略和验收；设计/提取成本分别报告边际与端到端口径。
extract 场景须区分原码复验、原码 seed 续改、仅凭 Schema 从共同骨架重建；前两种有产品价值，但不证明 Schema 足以传递全部设计。
小任务成功不证明复杂 Schema 有效，也不证明相对自由编码的普遍收益；真实题与失败样本、资料可见性、重复和测量波动须保留。

## 10. 七项验收缩略

下列按所发布能力适用；伪后端/故障模拟验证工程语义，真实模型/GPU 验证能力与数值性能，两类证据分别报告。

| 验收 | 最小可观察要求 |
| --- | --- |
| 运行正确性 | 固定完整比较器、输出/副作用和规定输入；没有错用旧 harness 的判据 |
| 设计符合性 | 准确 R/C 的有限独立证据；更快但违背设计不得视为成功，unknown 不冒充通过 |
| 性能 | 准确源码、比较对象和计时环境；合格慢实现仍返回，性能目标另报 |
| 代码续写 | 未晋升 C1 的完整内容/删除成为同 R 下一起点，覆盖和预算不重置 |
| 失败恢复 | 自身承诺边界内，身份/代码/消耗不丢；已完成事实不重复执行，未知作业先结算 |
| schema-off | 无 KS 也可启动原路径；归一化 prompt/工具/预算/输出策略保持，新增 Schema 模型调用为零 |
| 公开可用性 | 干净环境取得必要材料并运行真实入口；依赖/支持条件明确，无 AKA 或作者绝对路径暗依赖 |

公开材料需可实际取得，只有 receipt/SHA 不足够；首发可只闭合选中 demo 的代码、Task、资料和来源/许可说明，不要求先整理整个 Wiki。

## 11. 留给用户的三个产品判断

三项与主代理的 [方向决策页](03_decision_questions.md) 对齐，当前全部待答：

1. **D1 独立程度：是否要求不依赖本地或远端 AKA 后端，也能生成并验证支持范围内的 kernel？** 推荐“是，先一条受限真实路径”；替代为仅独立前端、强制 AKA backend，减少自有执行器维护。
2. **D2 首发覆盖：先小任务/单栈公开，现有复杂 Schema 紧随，还是先集中现有复杂算子？** 推荐工程 fixed-first、尽早接 design/一次 optimize demo，并在最早阶段预审真实材料；替代会更早检验复杂性，但公开 demo 和 fidelity 成本更难控制。
3. **D3 planner 介入：按固定设计实现阶段交接，还是按 episode 周期检查？** 推荐实现与验收完成后、或出现有证据的设计阻碍时接管；替代为每个或每隔若干 episode 调 planner，可更早纠偏，但增加成本和设计变动，仍允许 keep 并保留 partial/累计预算。

固定 implement 内不自行改设计；动态 demo/AKA 循环只在该请求已包含 optimize 且有依据与余额时推进。具体支持条件、预算与优化停止目标随后定清。
三仓职责、合格慢实现的交付，以及正确性/性能实测、有限独立 fidelity 审阅和 unknown 不冒充完成，是建议知会项；本轮不追加验收强度问卷。
provider、单步/loop 接点、V1/never、导入 phase 和内部目录由技术验证决定，不列为用户必须设计的选项。

## 12. 本阶段交付与后续约定

**首阶段仅交付六份研究和本初纲，由主代理汇报关键方向，再由用户判断。** 本文没有批准实现、测试/模型/GPU 运行、远端 fork、提交或发布。
用户决策后，按原约定再启动两轮全新的 max 子代理从不同角度撰写落地计划，最后由新的 xhigh 子代理整合完整计划；**这两轮及最终整合当前尚未启动**。
下一阶段应以用户决定修订假设，再确定技术验证、正式支持面、实现步骤和验收安排；本初纲保留被 R3 修正的前提及失败退路，不把候选推荐写成既定事实。
完整计划还应安排根 README 的定位与使用说明更新：清楚区分当前可用能力、四个独立功能入口、有限 demo，以及长期可组合的 extract/design→implement→optimize 自动循环。本阶段只在 discussion 保存讨论，未提前修改根 README 的现状承诺。
