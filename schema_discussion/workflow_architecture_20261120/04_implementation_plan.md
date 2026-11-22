# 四流程与 AKA 可选 Schema：双仓实施计划

日期：2026-11-20。状态：**实施已启动，按已签收前置滚动推进**。AKA 本地与远端 `feat/schema-workflows-v1` 分支已建立，KS 隔离工作树已恢复并验证干净。实际进度与资格见[实施记录](implementation/README.md)；本计划本身不构成实现或真实运行证据。

本次须交付两条可直接运行的路径：

| 用户路径 | 可运行交付与验收归属 |
| --- | --- |
| 在当前 AKA 中打开 Schema switch | 同一 AKA 安装提供明确的 on/off switch 和可复制命令；on 使用固定可选 KS，按实现阶段交 planner 并消费后续行动；off 保持原流程。A3/A6/A7、V09/V12/V14 负责 |
| 在 KernelSchema 中直接运行迭代 | 独立有限 driver 将 design 或已完成 extract 的显式消费接到 implement→至多一次 optimize；有合法 revise、同 R implement 或 verify 行动时，在原预算内实际执行后续固定请求。keep 无行动正常结束。K6、V09/V13 负责 |

AKA fixed 是独立里程碑；本次完整目标还包括 A6 的动态迭代和两条路径的实际使用说明。KS driver 只有一个顶层 owner，不建设通用调度平台或无限循环。
已确认方向见[决定页](03_decision_questions.md)。当前规范由下列四份正文分工承担；研究、R4 和历史快照仅供追溯。

| 正文 | 唯一职责 |
| --- | --- |
| 本文（P） | 方向、产品/仓库边界、阶段交接、发布门和维护原则 |
| [交付单元（W）](05_work_packages.md) | 15 个单元、改动区域、交付、唯一 owner/依赖、首批顺序及验收归属 |
| [公共合同（C）](planning/r5_contract_execution_review.md) | 唯一 API/请求、身份与选定闭包、源码、恢复/预算、结果及 fidelity |
| [验收与交付规范（V）](planning/r5_validation_delivery_plan.md) | 唯一 Task/案例参数、V01–V15、公开材料与消费模式 |

<a id="scope"></a>
## 1. 已确认方向与首发边界

D1：KS 提供不依赖 AKA 后端的受限真实 implement，AKA 提供增强能力。D2：工程先固定设计实现与续写，尽早公开 design/一次 optimize，同时提前预审复杂题。D3：同一设计可以跨多个 coder episode，实现阶段合格或出现有据设计阻碍后才交 planner。
首发支持 Linux x86_64、Python 3.11 源码 checkout/editable、Claude CLI、Triton、单一 kernel.py、一个完整 Task evaluator；实际 CLI/模型/依赖版本由资格记录冻结。参考 implement 限单任务、单栈、串行、有预算上限，不复制通用 Agent 调度平台。
首题为新增 row_softmax_fp32_v1，精确判据由[V 的 Task 规范](planning/r5_validation_delivery_plan.md#task)定义。Agate 优先核验完整执行能力，退路是运行同 evaluator 的本地 GPU；H20 仅是首先检查的候选。没有合格通道时相应真实发布门未通过，材料和 CPU 工程仍可进行。
F1 为既有 FA4 decode 的真实子请求，材料预审与首题同步，真实交付紧随小 demo；原 CuTe/实际 SM103、完整 evaluator、host 冷启动/稳态及有限 fidelity 均需自己的资格。F1 所需源格式/adapter 在案例内实现，后续扩域由真实结果与下一请求决定。
四个能力以真实入口和证据兑现：design 写完整 Schema，extract 保持原流程，optimize 写 keep/revise 与行动，implement 交准确代码及独立判据。仅方法 Markdown、CPU fixture 或历史性能不能称真实实现完成。

<a id="ownership"></a>
## 2. 仓库、循环与代码边界

| 仓库 | 唯一职责 | 边界 |
| --- | --- | --- |
| KernelSchema（KS） | 语言/Template、四流程合同与入口、写作内核、有限参考 implement 与迭代 driver、案例 | 核心安装/导入无 AKA；不复制 ABBA、多任务调度或长期远端维护 |
| AKA 适配 fork | 可选 Schema 模块、on/off switch、固定 backend 与按阶段迭代、原生 coder/执行/恢复/晋升接点 | 消费固定 KS 子模块的规则/planner；安装与资源路径由 AKA 自有交付负责，不复制 Schema parser/写作方法或父总账 |

```text
AKA 适配 fork ──显式开启的固定可选 submodule──> KS
KS 参考 implement ──> Claude CLI + Task evaluator + 合格执行通道
KS 选择 AKA backend ──同步调用──> 外部已安装的固定 adapter
```

这是目标依赖，不是现状。每个运行只有一个顶层循环 owner：KS 有限 demo 调固定 backend，或 AKA 动态模块调 KS 单次 planner；不得互相启动顶层循环。单 owner 可以跨进程调用外部 AKA，额度、暂停和原始收据须贯穿，细则由[C 预算](planning/r5_contract_execution_review.md#budget)统一定义。

| 未来区域 | 实际内容 |
| --- | --- |
| kernelschema/ | 唯一 CLI/Python API、contracts/materials/resources/template_api/artifacts、writing/design/optimize、fidelity/extract_bridge、runtime/claude、implement/reference、evaluation/task/agate |
| schema_design/、schema_optimize/、schema_implement/ | 各一份简短 README，链接角色方法、唯一命令与同组 examples；不复制配置或参数说明 |
| schema_extract/ | 现有命令、P1–P6 和原恢复/发布语义 |
| examples/workflows/、docs/workflows/ | Task/请求/evaluator/案例与按真实资格更新的使用说明；具体文件清单归 V |
| AKA:src/aka/workflow/schema/、packages/atrex-aka-schema/、optional/KernelSchema/ | 可选 provider/backend/bridge、包 ownership/metadata、固定 KS gitlink 与规则/资料、最少关联状态及资格投影；默认 off 走原 provider |

根角色文档仍是方法真源；Template 使用既有工具，非 root 阅读含 root Contract。基础导入不加载 torch/CUDA；资料与固定 assignment 的唯一权限规则在[C 输入合同](planning/r5_contract_execution_review.md#api)。

<a id="stages"></a>
## 3. 固定实现阶段与 planner 交接

一次 implement 固定 Task/Schema/规则/资料/scope、mode、准确 seed、backend 和预算。coder 可经历多个 episode；每次交付稳定 C，评价准确 C，再根据已结算反馈、下一行动和余额决定同 R 继续。完整公共状态只在[C 结果与阶段状态](planning/r5_contract_execution_review.md#state)定义。
一般代码缺陷、缺 dispatch 或未写完留在同 R；正确且符合设计的慢代码可以交付，性能目标另报。实现阶段完整合格且有已授权优化目标，或有独立可核对的设计阻碍，才交外壳决定 planner。超时、普通编译错误、无晋升、unknown 检查不能自动归因为设计失败。
optimize keep 无行动即结束；明确补证进入 verify_only，明确同 R 改码继续固定实现。revise 发布不可变新 R，旧事务结算/作用域关闭后另准入。KS driver 与 AKA 动态外壳都消费这些合法行动；有 fence、额度不足或材料缺项则照实停止。发布与准入分离，固定 backend 始终零 planner；暂停、耗尽、未结算 fence 的具体后果遵循合同。
当轮目标由 Materials 的 workflow_objective 承载，不能放宽 Task/Scope。方法冻结后每次 fidelity 解释当前完整 Schema，覆盖与设计符合分开判定；新 R 不能套用 R0 的常量或旧 C 的资格。
原码复验、显式原码 seed 续改、共同骨架下的纯 Schema 重建分别报告；材料边界与结论限制见[V 的消费模式](planning/r5_validation_delivery_plan.md#consumption)。普通反馈不自动写回 Schema，合法 REALIZATION.Feedback 的边界见 C。

<a id="release"></a>
## 4. 阶段交付与发布门

工程按接口、CPU 集成、真实资格分层。W 是唯一依赖表；消费稳定接口不等待整个单元的 GPU 或发行收尾。独立 KS 小 demo 不等 AKA 动态或复杂题，AKA fixed 不等 KS 真实 demo；跨后端声明另汇合双方证据。
下表定义支持声明的门，具体 owner/子 gate 见[W](05_work_packages.md)，测试内容和公开文件见[V 的验收](planning/r5_validation_delivery_plan.md#validation)及[公开材料](planning/r5_validation_delivery_plan.md#public)。复用已适用记录，不为同一声明重跑同一资格。

| 发布门 | 必须具备的证据与声明范围 |
| --- | --- |
| E0 工程预览 | 适用材料/接口和 CPU fixture 可审查；不声明真实独立 implement |
| KS-P1 独立产品 | K6.release：无 AKA 的真实 fixed、准确续写工程及真实 partial-seed 消费、真实 design/一次 optimize 及合法后续行动的消费、完整 Task/有限 fidelity/性能证据、K2 消费能力与源码资源；真实 keep 照实发布 |
| KS-P2 复杂子域 | K8.real：K7 预审后真实 F1 及实际所需栈/格式资格；仅声明通过的子范围，不冒充全 FA4 |
| AKA-P1 fixed | A7.fixed：A1–A5 的适用工程/真实证据和独立固定源码配对；不等待 A7.pair、K6 或 A6 |
| AKA dynamic | A6.real：通过同一 Schema-on 入口按阶段调 planner、执行合法后续固定请求，并保留发布/准入/恢复的真实动态证据；不反向阻塞 fixed，但属于本次完整目标 |
| 跨后端支持 | A7.pair：同 Task/主体/判据的对接 fixture 和两后端各自适用真实记录；不以一边 pass 代签另一边 |

整体收尾须能从两仓 quickstart 的实际命令分别走通开头的两条路径，保留阶段、后续行动、结果/费用和恢复证据；fixed 通过不能代替 AKA dynamic。F1 和跨后端支持仍按各自门签收，未通过项明确保留。
README/支持表只声明实际通过的门，逐能力引用 evidence/index；安装、真实重跑、无 GPU 历史回放分清。源码公开闭包、许可/固定获取方式及模型/GPU 依赖按 V 准备，不把作者缓存路径当公共交付。
长期无人值守、多 provider、复杂远端故障维护等继续作为方向；没有真实证据不写入当前支持面。实际选中请求需要的 dispatch、完整快照和删除语义属于该请求，不能以有限首发为由省略。

<a id="maintenance"></a>
## 5. 实施基线、固定版本与维护

实施时重新读取 KS/AKA 的 AGENTS/commit/branch/status，使用已建 AKA 分支与已完成的 KS 隔离工作树，保护既有修改、有意删除、untracked 和冻结运行产物；不 reset/clean/stash 当前树。[开工准备基线](implementation/baseline.json)与[历史](01_history.md)仅供追溯。基线检查是开工记录，不是额外产品包。
AKA 从记录的上游 SHA 开始，必要 role/context/session 接点逐文件记录来源；A3 归口来源，A1/A4/A5 各自拥有实现，A7 发行汇合。产品仅在 KS/AKA 各自拥有的源文件中开发，不把脏 KS 目录当固定 submodule commit。
A3 明确可选 KS gitlink、Schema 包 ownership/metadata、实际安装与规则/Template/角色/选中资料的定位；schema-on 使用文档化且实测的 AKA 自有路径，只取得所需固定子模块和资源。默认安装及 off 不要求 KS 存在，不扫描、初始化、导入或下载 KS；on 缺项明确诊断。A7 在独立目录和非原 cwd 验证固定配对、包与资源以及 fresh-child 实际来源，只承诺已验证的源码 checkout/editable 路径，不要求全仓递归或通用下载平台。
维护保留四条原则：上游更新独立进行；冻结新的 upstream/fork/KS 配对；保留旧提交/资源/运行产物；先验证 off，再补受影响的固定实现、动态交接、身份/续写/恢复与 Task 资格。活动 run 不换版本，不承诺新实现恢复旧版本冻结状态，回退只影响下一新 run。不存在独立例行更新工程包。

<a id="external-consumers"></a>
EXP 仅是范围外消费者：本次不修改其实现、配置或捕获，不运行也不验收它；必要旧代码至多只读追源参考，不构成迁移前置。当前 EXP 采用排除 gitlink 的普通文件 vendor，并有固定安装、launcher 和快照假设，因此不能承诺只更新 AKA SHA 即自动兼容；这一消费缺项不阻止 KS/AKA 开工或独立发行。事实依据见[本轮消费审查](revision_no_exp_20261120/r1_consumption_review.md)。

下一阶段首批和后续依赖只见[W 开工顺序](05_work_packages.md#start)。D1–D3 已确认，不再把环境资格未知转成新的产品方向问卷。
此前 34→17 的复审依据见[R3 裁决](review_20261120/r3_adjudication.md)和[原变更报告](review_20261120/r4_change_report.md)；本轮 17→15 及两条运行路径的落点见[本轮变更报告](revision_no_exp_20261120/r2_change_report.md)。[本轮修改前快照](revision_no_exp_20261120/before/04_implementation_plan.md.txt)保留历史文本，不参与当前规范。
