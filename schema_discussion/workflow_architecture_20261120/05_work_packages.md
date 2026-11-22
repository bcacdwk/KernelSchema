# 执行索引：15 个交付单元

日期：2026-11-20。**实施已启动，按已签收前置滚动派工**。下述内容定义交付要求；完成状态与实际证据见[实施记录](implementation/README.md)，不能以任务派工代替 gate 通过。
[主计划](04_implementation_plan.md)定义方向/阶段/发布门，[合同](planning/r5_contract_execution_review.md)定义唯一 API、身份、恢复/预算/资格，[验收规范](planning/r5_validation_delivery_plan.md)定义唯一 Task、V01–V15、公开材料与消费模式；本文只持有 owner、交付和依赖，不抄判据。
仓库前缀仅为 KS、AKA 适配 fork。共同 fixture 位于 `KS:tests/workflows/fixtures/v1/row_softmax/`，由 K2 管合同字节、K1 管 Task 原件、K4 管方法，A2 消费固定配对；不手改两份口径。

<a id="dependencies"></a>
## 1. 子 gate 与唯一依赖表

`.interface/.materials` 是可供下游开工的稳定子交付；`.cpu/.engineering/.docs` 证明适用工程，`.real/.demo/.fixed/.pair/.release` 才汇合对应真实/发行/消费事实。后缀不是新增工作包；没有“必须整包完成”的隐含边。
实施基线检查为全体开工前记录：重读 KS/AKA 状态/边界，核对已建 AKA 分支和 KS 隔离工作树，保留有意删除/untracked/冻结运行产物，固定拟用上游及可追溯源码。没有模型/GPU 完成前置。
表内仅列直接前置；真实 profile 资格是所在单元的实际交付，不以“目录存在”代替。证据允许按合同复用，`.real` 再被发行消费时不要求重复跑相同记录。

| 子 gate | 直接前置 | 交付边界 |
| --- | --- | --- |
| K1.materials | 实施基线 | Task/R0/骨架/材料、evaluator 输入输出接口与缺项表 |
| K1.evaluator | K1.materials、K2.interface | 唯一完整 evaluator、执行/收据接口及 CPU 错绑/结算路径 |
| K1.real | K1.evaluator | 所选执行通道的真实 Task/计时/身份资格 |
| K2.interface | K1.materials | 唯一合同/资源/codec/恢复锚点及消费桥接口 |
| K2.cpu | K2.interface | CLI/API、选定闭包/字节/模板/资源/提取消费工程 |
| K3.interface | K2.interface | 单调用、角色投递、终态/用量接口 |
| K3.cpu | K3.interface | fake CLI 与真实运行时调用代码的协议/权限验证 |
| K3.real | K3.cpu | 锁定 Claude profile 的真实单调用/结算资格 |
| K4.interface | K2.interface | 固定实现、当前 R 方法、合报告与结果/预算接口 |
| K4.cpu | K4.interface、K2.cpu、K3.cpu、K1.evaluator | 固定 loop/续写/unknown/预算/恢复和资格投影 |
| K4.real | K4.cpu、K3.real、K1.real | 参考实现真实生成/完整评测/独立 fidelity |
| K5.interface | K1.materials、K2.interface、K3.interface、K4.interface | 共享写作/目标/发布/交接接口；固定反馈 fixture 可用 |
| K5.cpu | K5.interface、K2.cpu、K3.cpu | keep/revise、权限/目标、发布恢复与新 R 交接 |
| K5.real | K5.cpu、K3.real；真实 optimize 另需 K4.real 或 A5.real 的已结算反馈 | 真实 design 与一次 optimize；反馈可由已合格参考或 AKA 路径供给，不以 KS GPU 为代码开发前置 |
| K6.docs | K2.cpu、K1.materials | 源码资源/公开文件/使用说明与有限 demo driver |
| K6.demo | K6.docs、K4.real、K5.real（采用 K4.real 反馈、无 AKA） | 无 AKA 的真实迭代 demo、合法后续固定行动、partial-seed 消费和 evidence |
| K6.release | K6.demo、K2.cpu | KS-P1 源码公开闭包/许可/任意 cwd 交付 |
| K7.materials | 实施基线、固定 FA4 来源 | F0/F1 预审及原栈/范围/公开材料缺项 |
| K8.engineering | K7.materials、K4.interface、K2.interface | F1 的 Task/evaluator、实际所需 source 格式和执行适配；选 AKA 时另消费 A2.interface/A5.interface |
| K8.real | K8.engineering、K6.demo；按所选路径取 K4.real 或 A5.real，另有本案例 CuTe/实际 SM103 资格 | 紧随小 demo 的真实 F1；不等 K6.release 的说明收尾 |
| A1.cpu | 实施基线 | 指定 e、生命周期/预算接点 CPU spike 与 driver 结论 |
| A2.interface | K1.materials、K2.interface | Task/source/native 角色映射和共同 fixture 接口 |
| A2.cpu | A2.interface、K2.cpu、K1.evaluator | 完整 Task hook/codec/证据导入的工程映射 |
| A2.real | A2.cpu | 当前 native/profile 的实际 hook/Task/计时资格 |
| A3.interface | A1.cpu、A2.interface、K2.interface、已审必要接点来源 | on/off switch、可选 KS gitlink/包安装、角色正文/资料与固定源码接口 |
| A3.cpu | A3.interface、K2.cpu | AKA 自有安装/资源闭包、无 KS off、实际资料投递及来源归口工程 |
| A4.cpu | A1.cpu、A2.cpu、A3.interface | baseline V0、C0 首调与跨 e 终态 C/HEAD 准备、seed 持久边界 |
| A5.interface | A1.cpu、A2.interface、K4.interface | 准入/预约/资格/audit/外部定位/完整结果接口，可与 A4 并进 |
| A5.cpu | A5.interface、A4.cpu、A3.cpu | 固定 backend、三 audit、用量/暂停/恢复工程汇合 |
| A5.real | A5.cpu、A2.real；所选 native coder/reviewer 的真实 profile 资格 | 实际 fixed 的完整 Task/有限 fidelity/性能及所宣告恢复证据 |
| A6.cpu | A5.cpu、K5.cpu | Schema-on 按阶段调 planner、消费合法行动、keep/新 R 与累计账户的有限动态工程 |
| A6.real | A6.cpu、A5.real、K3.real；当次 K5 写作检查在本 gate 验证 | 真实动态请求及发布/准入记录，不要求某次一定 revise |
| A7.fixed | A5.real、A3.cpu、K2.cpu | 固定 fork/KS 配对、独立源码安装/资源与 fresh-child 来源、AKA-P1 证据发行 |
| A7.pair | A7.fixed、K4.real、A2.cpu | 汇合两后端适用真实记录与差异索引，仅控制跨后端声明 |

K1 先交材料、K2 再交接口、K1 evaluator 再消费该接口；A3 接口先供 A4，A5 接口同期做，CPU 才汇合。这些都是子 gate 单向边，无整包互等。

<a id="k1"></a>
## 2. KernelSchema：K1–K8

### K1 — 首题材料、唯一 evaluator 与执行资格｜材料/evaluator 已交付；真实资格待验证

- Owner：KS Task/evaluation；改动 `examples/workflows/row_softmax/{task,schema,starter,references,evaluator}/`、`kernelschema/evaluation/{task,agate}.py`。
- 开工输入：V 的 row-softmax Task/材料规范。先交 R0/自由度、不可执行骨架、来源/许可/版本/缺项；随后实现 golden/baseline/input/check/timing/entry、固定 staging 与真实来源摘要、逻辑 job 结算。
- 完成证据：V01/V03/V06/V07 的适用材料/CPU/真实记录；Agate 不能跑完整判据则查同 evaluator 自定义执行或合格本地 GPU，真实能力缺失保留缺项，不能换比较器。K1 拥有原 out storage/view 和非零 offset 的硬检查。

### K2 — 唯一包/API、合同资源与提取消费｜接口/CPU 已交付

- Owner：KS core；改动 pyproject.toml、kernelschema 的 CLI/api/contracts/resources/template_api/materials/artifacts/extract_bridge 与共同 fixture。
- 开工输入：K1.materials 的固定 Task 形状；交严格 v1/JCS/Ref/source codec、能力检查、稳定 prefix/选定闭包、run 锚点接口、源码资源及原 extract 透传/消费桥。
- 完成证据：V06/V13 的字节/路径/主体定位/真依赖缺失与无关历史缺失反例、任意 cwd/无 AKA 核心安装、原 extract 语义与多来源消费；三新操作恢复共用 K2 锚点，由 K4/K5 实现流程。

### K3 — Claude 单调用与角色资料｜接口/CPU 与历史 CLI profile 资格已交付

- Owner：KS runtime；改动 kernelschema/runtime/claude.py 与角色投递。
- 开工输入：K2.interface；交独立 planner/coder/reviewer scope、固定 assignment/allowlist、原始事件、请求/报告模型、终态/用量与稳定 C 新会话续写。
- 完成证据：V02/V08/V13 的 fake 非零/缺终态/缺用量/角色伪造与真实锁定 profile；公开参数无法满足权限/终态/限额时该 profile 不准入，不照搬 extract worker。

### K4 — 固定参考实现、恢复与有限资格｜接口/CPU 已交付；真实整合待验证

- Owner：KS implement/fidelity；改动 kernelschema/implement/reference.py、fidelity.py、私有单 owner 状态及结果投影，消费 K2 artifacts API。
- 开工输入：合同方法/结果接口和 K1 材料；交固定 Q 的 coder→准确 C→评价/审阅→反馈 loop、verify_only、latest/qualified、当前 Schema 方法、coverage/fidelity 合报告、完整预算/结算/恢复。
- 完成证据：V02–V08 的适用轨迹及真实 fixed；包括同 C 对 R0/Rd 不同资格、检查 unknown 可补证、未知操作禁重提、首 result 前恢复、同 run/新 run/同父新 R 费用区分。方法未覆盖的真实设计义务保持 unknown。

### K5 — design/optimize 共享写作与发布｜接口/CPU 已交付；真实整合待验证

- Owner：KS writing；改动 kernelschema/{writing,design,optimize}.py 及写作请求生成。
- 开工输入：K5.interface 所列接口与固定反馈，不等 GPU。交 workflow_objective 唯一目标、参考筛选、草稿/规范化、权限/全任务/有限语义检查、keep/revise/行动、修订计数和当前 R 交接。
- 完成证据：V09 目标变化/越权、发布无哈希环、三操作恢复中的写作分支、keep 停止、新 R 重判及跨请求原件共存；真实 design/optimize 分别保留实际结果，不强迫 revise。

### K6 — 小 demo、源码公开发行与说明｜工程独立审阅完成；真实/发行门待验证

- Owner：KS demo/docs/release；改动 row-softmax requests/run_demo.py/evidence、根 README、三能力 README、docs/workflows 及发行资源索引。
- 开工输入：K2 稳定源码消费能力即可准备 V 公开文件。交无 AKA 的 fixed 与有限单 owner driver：design 或已有 extract 消费→implement→至多一次 optimize；对合法 revise、同 R implement 或 verify 行动，在原账户/限额内实际执行后续固定请求。另交标明人工来源的 partial seed 续改、可复制命令、重跑/历史回放、支持/许可/获取/校验说明。
- 完成证据：V01–V09/V13 中适用既有记录组成可追溯索引，任意 cwd 使用唯一 CLI；只导出已授权选定闭包，脱敏说明与可导入产物分清。driver 的后续行动、keep 停止、暂停/fence 和恢复使用 V09 的 CPU 分支与实际发生的真实记录；真实首轮即完成不伪造续写，真实 keep 无行动正常公开。

### K7 — FA4 来源、范围与原栈预审｜材料已交付（不代表真实资格）

- Owner：KS complex-task materials；改动 docs/workflows/cases/fa4_decode-admission.md 与新案例材料清单，原 YAML 保持。
- 开工输入：固定四题来源。交 F0 源包/许可/13 参数/完整比较器、F1 精确子请求、root/dispatch/when 覆盖和 CuTe/SM103/source 格式缺项。
- 完成证据：V15 材料对照与不覆盖/异步风险边界；报告 available/missing/incompatible 和公开取得方式。预审不等 GPU，也不签复杂运行支持。

### K8 — 真实 F1 与实际所需格式/适配｜所选 KS reference 工程审阅完成；真实门未签

- Owner：KS complex-task delivery；选 AKA 时本单元同时负责该 F1 的 AKA adapter 子交付，通用接点仍归 A 系列。
- 开工输入：K7 材料与 K4/K2 接口；交 F1 Task/profile/evaluator、root Contract+选中 Design、所需完整 source 格式/删除/安装及执行适配，不等 README 收尾。
- 完成证据：V05 必要格式扩展、V15 真实 CuTe/实际 SM103 的构建/正确性/副作用/有限 fidelity/host 冷启动与稳态性能；历史 30/30、kernel-sum 或 H20/Triton 不能代签。后续域按结果和下一请求另定，当前请求必要 dispatch 必须完成。

<a id="a1"></a>
## 3. AKA：A1–A7

### A1 — 指定 episode 与生命周期接点｜CPU 已交付

- Owner：AKA driver seams；改动 workflow/schema/bridge.py、必要原 loop 接点和指定 e fixture。
- 开工输入：实施基线；用真实 Store/CandidateWorkspace 配 fake Agent/Evaluation，交 setup/finish 单 owner、预算接点、HEAD/G 关联和 driver 结论，实际接缝实现留本单元。
- 完成证据：V08/V11；façade 若复制生命周期则采用原 loop 窄 admit/settled 接点，均不闭合则不发布多 e 能力，KS 继续。必要接点代码逐文件记来源供 A3 汇总。

### A2 — Task/source 映射与实际 hook 资格｜接口/CPU 已交付；实际资格待验证

- Owner：AKA task adapter/共同对接；改动 Atrex-Bench Task/profile、source codec/映射与共同 fixture 消费。
- 开工输入：K1/K2 接口；交 public/initial/evaluation 映射，V0 baseline 与 C0 分列，真实 runtime 来源/版本、完整 golden/副作用/计时 hook、稳定闭包导入和 capabilities。
- 完成证据：V01/V03/V06/V13/V14 的工程及实际 hook 记录，正确候选和四个关键坏候选使用统一字节；目录标记/普通 correctness 不代签。无合格 hook 则该任务 unsupported，配对汇总留 A7.pair。

### A3 — Schema switch、可选安装/off、角色正文与来源归口｜接口/CPU 已交付

- Owner：AKA optional integration；改动 workflow/schema/plugin/provider、packages/atrex-aka-schema、构建 owner/metadata、optional/KernelSchema 固定 gitlink、role/context/session/资源安装及来源台账。
- 开工输入：A1 结论、A2/K2 接口、已审必要接点来源；交明确的 on/off switch 与成对可复制命令、固定 assignment/allowlist/read/effective/check 工具。Schema-on 的准备步骤由 AKA 自有文档化路径取得所需固定 KS/资料并安装 KS 核心和 Schema 包；明确包 ownership、安装方式与规则/Template/角色的定位，不靠作者 cwd/缓存。默认安装及 off 不要求 KS 存在，不扫描、初始化、导入或下载；on 缺项给明确安装诊断。
- 完成证据：V02/V12/V13/V14 的实际安装/资料投递与无 KS off。工程可用固定接口/显式开发来源，正式 gitlink 必须指评审过的可取得 KS 提交，A7 冻结发行配对；只初始化所需子模块/资源，不新增通用下载平台、wheel 或全仓递归承诺。A1/A4/A5 各改自己的接点并留来源，A3 不承包全部代码，也不要求它们先完成才能交安装/正文接口。

### A4 — V0/C0 与跨 e 准备｜CPU 已交付

- Owner：AKA preparation/source；改动 main_adapter、orchestration helpers、CandidateWorkspace/EpisodeWorkspace 的 preparing/link/register/install。
- 开工输入：A1/A2/A3 的接口；交可测 baseline 的 native V0、never+helper 或受限 V1 结论、首次 coder 的准确 C0、后续 e 的终态 C/当前 HEAD，以及 intent/ready/admitted 持久关联。
- 完成证据：V03/V04/V05/V11。不可执行骨架和非空 partial seed 各证明 baseline 与首调来源分离；恢复不重做已结算 V0、不覆盖 draft、不退 G；必要 V1 仍受同 R/额度/提交前资格和 ABBA 约束。

### A5 — 固定 backend、三 audit、预算与恢复｜工程最终审阅完成；真实门未签

- Owner：AKA fixed backend/state/projection；改动 seal/attempt/promotion audit、workflow/schema/backend/projection、native 用量/准入/恢复关联。
- 开工输入：接口从 A1/K4 开始与 A4 并进；交准确 C 的资格绑定与三 audit、code/verify_only 完整结果、单 owner 外部预约/原生定位/暂停/收据、首 result 前及结算恢复。
- 完成证据：V02–V11 适用工程和实际 fixed 记录；资格先 durable、原生终态独立、未知费用不归零，V0/预取 reviewer/production/收尾均计账。固定工程在本单元闭合，不把恢复留给 dynamic。

### A6 — 按实现阶段交 planner 的有限动态链｜工程最终审阅完成；真实门待验证

- Owner：AKA dynamic provider；改动 workflow/schema/provider/state 和 KS 单次写作调用。
- 开工输入：A5.cpu 与 K5.cpu；交同一 Schema-on 入口中的已授权阶段边界、keep 行动、完整新 R 发布/准入及后续固定请求执行、选定闭包、同父费用/历史 C/G 保留。
- 完成证据：V08/V09 的 CPU 轨迹和真实写作/固定资格；无行动即止，未知操作不能用历史 qualified 遮盖。未闭合自动恢复只提供显式下一固定请求，不能宣称 dynamic 门通过；只交 fixed 也不构成本次 AKA 迭代目标完成。

### A7 — AKA 独立 fixed 发行与配对索引｜fixed 工程审阅完成；真实/配对门待验证

- Owner：AKA release/evidence；改动固定 upstream/fork/KS 配对清单、profile/evidence、README/quickstart/optional-module 说明。
- 开工输入：A7.fixed 仅汇合 A1–A5 的适用工程/真实证据和 K2 源码资源，不等 KS 真实链、A6 或配对子 gate。交固定 upstream/fork/KS 来源、源码 checkout/editable 安装、包 metadata/规则/Template/角色/所选资料闭包、许可与完整结果/用量；在独立目录、非原 cwd 验证 on/off 入口、fresh-child 实际模块/资源来源及 AKA 自身固定运行产物的恢复。实际发生的续写才标真实。
- 完成证据：V01–V14 的适用记录按当前主体复用；A7.pair 在双方真实记录都有时形成同 Task/判据差异索引，单独控制跨后端声明。dynamic 可复制运行命令与行动/恢复说明在 A6.real 后追加；A7 交付 AKA on/off 使用入口并链接 K6 的 KS 独立迭代说明，不将 fixed 发行写成全部目标完成。

<a id="verification"></a>
## 4. 验收 owner 与公开材料交付

V 是唯一测试判据表。下表给主责及必须消费的路径；同一记录可复用，CPU 不代签 MODEL/GPU/服务资格。

| V ID | 主责与适用消费 |
| --- | --- |
| V01 | K1 统一 Task/evaluator；A2 实际 adapter，K8 采用其复杂 Task 规则 |
| V02 | K4 当前 R 方法；K3/A3 权限；A5 原生资格，K5 消费新 R 交接 |
| V03 | K1 计时；K4 结果，A2/A4/A5 原生映射/慢 C |
| V04 / V05 | K4 参考续写，A4 源码准备、A5 返回；K8 仅补实际格式扩展 |
| V06 | K2 身份/选定闭包，K1 执行来源，K4/A2/A5 适用性与 verify |
| V07 | K1 作业结算，K4/A5 固定恢复 |
| V08 | K4/A5 owner/准入；K2 锚点、K3 调用、K5 写作、A1/A4 准备、A6 跨 R 消费 |
| V09 | K5 写作/发布，K2 闭包，K4 当前 R；K6 driver 与 A6 动态消费行动/恢复/资格归属 |
| V10 / V11 | A5 三 audit；A1 指定 e、A4 V0/C0，A5 用量汇合 |
| V12 | A3 off；A7 固定发行复用 |
| V13 | K2 源码/提取消费，K6 公开 demo；A2/A3 资料/映射，A7 独立发行 |
| V14 | A3 安装/资源/switch 接口、A7.fixed/pair；A6 动态入口证据 |
| V15 | K7 预审、K8 F1 真实交付 |

V 公开文件清单的 owner：Task/R0/骨架/完整 evaluator 为 K1；人工 partial seed 与方法 fixture 为 K4；合同/源码资源/提取消费为 K2；目标/写作请求为 K5；demo driver、公开 evidence、README/说明/许可汇总为 K6；FA4 为 K7/K8；AKA 发行与配对为 A7。

<a id="start"></a>
## 5. 首批开工交接

1. 恢复实施前先核对 KS/AKA 基线/隔离记录；首批就绪任务为 K1.materials、K7.materials 和 A1.cpu，按下节两槽派工。材料、预审与指定 e CPU spike 均不等模型/GPU。
2. K2.interface 消费 K1 材料后，K1.evaluator、K3、K4.interface、K5.interface、A2 分轨推进；真实资格只限制对应真实 gate。
3. 后续只按上表消费子 gate。KS-P1、F1、AKA fixed/dynamic/配对各按[发布门](04_implementation_plan.md#release)签收，不重问 D1–D3。

[此前变更报告](review_20261120/r4_change_report.md)保留 34→17 的历史映射；[本轮变更报告](revision_no_exp_20261120/r2_change_report.md)记录 17→15 及运行路径闭合。[本轮修改前索引](revision_no_exp_20261120/before/05_work_packages.md.txt)为历史，不是另一套当前依赖。

<a id="agent-dispatch"></a>
## 6. 开发代理派工与实施批次

用户本轮指定开发子代理默认 `gpt-6-astra / xhigh`，覆盖本任务此前的默认 effort。这里安排的是开发仓库的代理；产品运行时的 Claude CLI、planner/coder profile 仍按 C/V 和真实资格确定。
主代理负责基线/隔离、选取已就绪子交付、明确写入边界、衔接结果和更新状态；同时最多两位子代理。每个有界的新任务使用新代理，不设一位长期包办全部 KS 或 AKA 的常驻代理；同一任务内的小修和必要测试可自行完成。

下面是按现有依赖作出的 **11 个基准派工批次**，不是新增技术工作包、工期承诺或固定总轮数。15 个单元不等于 15 位代理/15 轮；未通过的接口、真实环境资格或新增缺陷可能产生修复批次。已有前置就绪即可占用空闲槽位，不因另一条线路等待 GPU 而整体等轮次结束。
批次中的 `.real` 和发行工作只有在既有实际前置、环境与运行配置就绪时才执行；没有条件时准确交付已完成的接口/CPU成果，不把批次结束写成真实门通过。

| 基准批次 | 子代理一：有界交付 | 子代理二：有界交付 | 本批主要交接 |
| --- | --- | --- | --- |
| B01 | K1.materials：首题 Task、手审 R0、骨架和 evaluator 接口材料 | K7.materials：FA4 F0/F1 来源、范围与原栈预审 | 固定第一批材料及实际缺项；主代理先完成实施基线/隔离记录 |
| B02 | K2.interface/cpu：唯一核心、CLI/API、Ref/source、run锚点、选定闭包及extract消费 | A1.cpu：原生指定episode/准备恢复的接缝代码与CPU验证 | 稳定公共接口和AKA driver结论，允许两仓独立继续 |
| B03 | K1.evaluator，条件具备时K1.real：完整判据/计时/执行及作业结算 | K3.interface/cpu，条件具备时K3.real：Claude调用、权限、终态/用量 | evaluator与runtime的可调用接口、测试及实际profile资格分开交付 |
| B04 | K4.interface/cpu：固定实现、准确续写、有限方法、结果/预算/恢复 | A2.interface/cpu，条件具备时A2.real：Task/source/Atrex映射与共同fixture | KS固定链工程与AKA任务适配；不互相代签真实资格 |
| B05 | K5.interface/cpu：design/optimize写作、目标、keep/revise、发布/恢复 | A3.interface/cpu与A5.interface：可选模块/角色资料/来源，以及后续固定事务接口 | planner工程可用；A4所需资料/准入/资格接口先稳定 |
| B06 | K4.real、K5.real、K6.docs/demo：独立链集成、真实fixed/写作与消费合法后续行动的有限demo，修复实际暴露的问题 | A4.cpu：正确V0、首调C0、后续seed、准备/恢复 | 首个独立demo的真实证据，以及AKA启动与续写工程 |
| B07 | 新代理审查KS独立链、修复明确缺陷并核对K6.release源码/说明/证据 | A5.cpu：固定backend资格/三audit、外部定位、用量/暂停与公共返回 | 首个独立产品的审查交付；AKA固定事务工程闭合 |
| B08 | K8.engineering及具备原栈条件后的K8.real：真实F1与所需格式/适配 | 新代理审查AKA fixed，完成A5.real/A7.fixed的适用真实验证、修复与发行配对 | 复杂子域按实际资格交付；AKA fixed按独立里程碑签收 |
| B09 | A6.cpu/real：Schema-on 按固定实现阶段调 planner、执行后续行动、新 R 准入及动态恢复 | 默认空闲；可整理 A7 已有职责内的 optional-module 固定安装/消费说明，避免与 A6 编辑同文件 | 动态链证据及 on/off 可运行命令，不能只交下一请求 JSON 就称迭代闭合 |
| B10 | 新代理审查 A6 及 KS↔AKA 对接，修复后汇合 A7.pair 的适用既有证据 | 整理 K6/A7 已有支持说明、来源/证据索引及两条路径的使用材料，修复后固定最终配对 | 动态/跨后端证据与支持说明一致；不新增技术包 |
| B11 | 新代理核对最终源码、版本、证据、公开材料与未完成项；核实 AKA switch 迭代/off 和 KS 独立迭代两条使用路径；必要时定点修复 | 默认空闲，仅在实际发现可独立修复的问题时派工 | 按 K/A gate 及两条可运行路径收尾，AKA fixed 不代替 A6；不无差别重跑已适用记录 |

首个独立可运行demo的基准落点是B06，独立审查/源码交付是B07；后续批次覆盖 AKA 动态、真实复杂题、双仓配对与两条运行路径的最终交付。这是任务分配基线，不是对真实模型/GPU成功率、运行时间或全部门在 11 批内通过的保证。主代理按已完成证据滚动更新下一批，不为维持表格顺序而阻塞已就绪工作。

每次派工只需一份简短任务交接，包含：所消费接口/材料的固定版本、当前子交付与目标、唯一可写目录/文件、应返回的代码/测试/实际证据、停止边界和未决事项。代理负责在自己的边界内完成实现和必要验证，不以新一份设计报告代替交付。
同轮避免两位代理写同一文件；核心合同/共享fixture/安装配置有明确单一写入者，变更先交接再供消费者使用。KS/AKA 消费者读取已验收的固定提交/快照，不从另一代理仍在修改的工作树安装；例如 B10 的最终配对在审查修复后固定，材料整理先用已有稳定来源并由各自 owner 交接。
主代理按diff范围、实际命令/结果及产物签收子交付，更新本表之外的实际执行记录和单元状态。普通批次不另设一轮纯总结代理；独立审查集中在KS独立交付、AKA fixed、动态/跨仓对接和最终交付，发现问题才增加有界修复任务。默认等待代理主动回报，不频繁追问进度。
本节补齐开发组织，不改变前文唯一依赖/验收规则；当前已按前置滚动推进 xhigh 实现代理；按实施记录签收实际产物，分支准备和派工不算技术单元完成。
