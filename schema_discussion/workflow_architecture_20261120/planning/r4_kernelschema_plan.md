# R4：KernelSchema 独立四流程的可执行开发计划

日期：2026-11-20。状态：用户已确认 D1–D3 后的分项计划，供 R5 审查和最终整合；本文所有新路径、字段、命令及验证均为拟议，尚未实现或执行。
本轮只读材料并写本文；未改代码、既有 Schema/提取产物，未运行测试、模型或 GPU，未安装、提交、操作远端或读取凭证。
已确认：无需 AKA 后端的受限真实 implement；工程 fixed-first、尽早 design/optimize demo、提前真实题预审；同 Schema 多 coder episode，按实现阶段交 planner。
本文不重新打开上述决策。技术默认值先用于开发，服务、版本与材料未知由下述资格门槛解决，不追加同样的产品问卷。
文中“观察”是已有材料/本轮源码阅读支持的事实；“计划/建议”不是现状。计划中的“通过”均指未来完成标准。

## 1. 首期交付、默认值与资格门槛

首期产品交付是一条可在没有 AKA 的环境中重跑的真实路径：固定 R0 → 生成 kernel.py → 真实验收 → 从准确代码继续 → 输出三类证据。
紧接同题 design → implement → 一次 optimize → keep 或 revise 后 implement；不要求跑赢参考，也不把小题当作复杂任务或科研收益证据。
复杂题材料预审与第一包同时启动；真正多节点实现放在独立 demo 后，并由材料资格决定是否需要新增实现栈。

| 维度 | 开发默认建议 | 首发边界 |
| --- | --- | --- |
| 宿主/安装 | Linux x86_64、Python 3.11、源码 checkout + editable 安装 | 根包唯一命名空间 kernelschema；非 editable wheel 单独过资源闭包门槛 |
| 公共入口 | 一个 kernelschema CLI，design/optimize/implement/extract 四子命令 | 原 extract 命令及其管理语义继续可用；不建立后台服务 |
| 模型运行时 | Claude CLI；planner/coder/reviewer 使用独立会话与角色权限 | 先支持一个实测版本；model 显式配置并记录请求值/报告值，不继承私有 qwen 路由默认 |
| 代码格式/栈 | 单一 kernel.py、Triton、PyTorch 调用 ABI | 首发不接受 helper 树、原生 CUDA/CuTe 或自动换栈；AKA 首发 single_file 可直接映射 |
| 设备 | H20 是首选资格候选；最终固定一个有真实通道且支持所选 Triton 的 NVIDIA profile | 此处没有 H20 服务可用证据；不同 GPU 要另建并验证 Task 硬件绑定 |
| 执行 | 一个任务 evaluator，拥有 reference、完整判据、输入与计时 | Agate 优先查资格；同 evaluator 的本地 GPU 通道是明确退路，二者不各自定义验收 |
| 编排 | 单任务、串行、固定设计、显式 coder/evaluator/总时长上限 | 不含 ABBA 晋升、多任务排程、自动 provider 搜索、跨机器会话搬迁 |
| 验收 | 数值、有限 fidelity、性能独立；机器项 + 独立只读审阅 | unknown 不代签；正确且符合设计但较慢可 completed，性能目标另报 unmet |

**小 demo 候选：新增 row_softmax_fp32_v1。**这是 R2 的教学建议，不是仓库已有 kernel/Schema。
计划固定连续 FP32 x[M,N]，M∈{1,32}、N∈{128,1024}，有限输入位于 [-20,20]；写满预分配 out，返回对象与 out 相同，x 不变。
reference 用 FP64 稳定 softmax；比较转 FP64 后逐元素 abs(y-g) ≤ 2e-6 + 2e-5·abs(g)，另验 shape/dtype、非有限值、别名和全部输出写入。
输入包含均匀、常值、极端有限值与固定随机种子；这些是有限测试分布，不是整个连续域的证明。
计时固定为完整 forward 调用，输出预分配、编译/预热不计入稳态，测量采样与同步方式写入 Task；记录每 shape 样本及离散程度。
比较源码另固定为 PyTorch FP32 softmax 后 copy 到 out 的同 ABI baseline；它与 FP64 golden 分开。目标可设全 workload 几何均值不慢于此 baseline，未达仍可完成实现。
R0 计划采用逐行单 program、稳定 max/sum 归约、无跨 program 同步；固定哪些 tile/warp 参数属设计或允许实现自由度，避免 optimize 只改未受约束调参。
首个 design 的参考材料以固定 Triton softmax 教材、CUDA/设备资料及本题 Contract 为主；可用现有 catalog 筛选相邻机制，但不假装已有高度近邻的小题 Schema。

**真实复杂 Schema 预审首选：schema_template/examples/fa4_decode。**
观察：现有四节点和 dispatch 来自独立 20261019 源包；root 仅覆盖 Q4、B∈{15,16} 等子域，Contract 范围更大，q1_global 复用 root 数学区。
观察：材料标记 L20D/SM103、CuTe DSL 4.6.1，Wiki 仍有占位；普通 global combine 空 split 与 PageIndex 顺序已有未证明限制。
因此它适合提前检验来源、范围和必要 dispatch，不适合声称已经属于 Triton 首发实现支持面；禁止混用 44 题主批次的相近源码/验收。
预审交付 Task/来源/硬件/判据/公开获取/未决表；随后实际实现优先选一条完整的 q4_direct 子请求，再按资格扩展 root 与必要节点/变化。
若机器、文档或原设计安全前提无法闭合，保留具体 blocked 项；可另审 curated mla_expanded_fp8，但不能通过换题抹去 FA4 失败与局限。

**G0 材料/执行资格门槛，KS-01 完成时产出一份可审查清单。**
每项标 available / missing / incompatible，并附实际位置、版本/摘要和取得方法；未知资源不能记 available。
必须覆盖：Task ABI/完整比较器；硬件与编程资料；准确栈/镜像；模型 CLI 参数与实际可用模型；evaluator 源码；源包与许可；一个真实执行通道。
有缺项时 CPU 材料、协议和 planner 开发可继续；真实 GPU 发布门槛保持未通过，不把“需装 AKA”作为 D1 的替代交付。
版本锁定由后续资格运行记录生成；此处不猜测 torch/triton/驱动的兼容组合，也不把旧 L20D 证据当 H20 资格。

### 1.1 Agate 必须在首期查清的接法

观察：AKA 随附 Agate 文档描述 --reference-dir 自定义 reference.py/input.py/shapes.json，支持 manifest、多文件摘要与 job 查询。
观察：其 jobs 文档描述同一逻辑提交的幂等键、job_id、等待/取消、unknown/quarantine；文档存在不等于当前客户端、租户和 GPU 已具备能力。
观察：所读普通 eval 文档不足以证明可表达本题完整误差公式和 out/输入副作用；不能仅把其 correctness.passed 翻译为 Task pass。
计划：task evaluator 固定任务判据；Agate adapter 只承担打包、提交、结算、取回原始证据。它直接依赖可取得的 Agate CLI/SDK，不导入 AKA。
G0 优先核对客户端/API 版本、授权 GPU/命令、镜像依赖、上传/返回限制、稳定提交键、终态查询以及精确源码/evaluator 摘要的回传。
若原生 eval 能证明实现全部 Task 判据，直接适配；否则评估在获准的 Agate 执行命令中运行同一固定 evaluator 包并返回结构化报告。
后一条要证明任务脚本可执行、材料与结果可闭合取得、稳定作业身份可结算；不自动假设 dev/OSS/自定义命令已开放。
若两种 Agate 方式均不能闭合，首个真实运行使用同一 evaluator 的本地 GPU 通道，记录 Agate 的具体缺项并保留 adapter 包任务。
若两种通道均不可用，只交 CPU/模型产物与 partial/blocked，真实 implement 发布门槛仍未完成；不得偷偷替换误差口径或复用旧报告。
不在运行到一半时无条件改用另一通道重测：先结算原 operation，确有剩余预算后明确记录新 operation/环境与新证据。

## 2. 目录、公共入口与共享边界

以下是拟新增布局；现有语言、Template 与 extract 主体保持原责任。先做本地模块，不先建立插件注册市场或通用 agent SDK。

| 拟议路径 | 责任与复用点 |
| --- | --- |
| pyproject.toml；kernelschema/__main__.py、api.py | 唯一包名/CLI；公开 design(request)、optimize(request)、implement(request)，协议版本检查 |
| kernelschema/contracts.py、materials.py | 请求/结果、Task/Schema/规则/范围冻结、材料资格与来源清单；不定义第二套 YAML 语言 |
| kernelschema/template_api.py、resources.py | 唯一 Template 加载适配、资源定位、检查版本；直接调用现有实现 |
| kernelschema/writing.py、design.py、optimize.py | 两个薄入口共用写作内核、参考筛选、权限 diff、候选与审阅；mode 决定任务与输出 |
| kernelschema/runtime/claude.py、prompts/ | 新角色适配、一次原生调用、实际输入/结果/用量；固定 DESIGN/OPTIMIZE/IMPLEMENT 与规则材料 |
| kernelschema/implement/reference.py、artifacts.py | 固定设计有限 loop、准确代码快照、串行准入、少量持久状态与 continuation |
| kernelschema/evaluation/task.py、agate.py | 固定 evaluator 调用与证据解释；可选 Agate 通道；无远端调度/资源池管理 |
| kernelschema/fidelity.py、extract_bridge.py | 有限符合性检查/独立审阅；extract 透传及提取产物准入，不修改 P1–P6 |
| examples/workflows/row_softmax/；docs/workflows/ | Task、R0、骨架、evaluator、资料索引、示例产物及用户文档 |
| tests/workflows/ | 必要身份/权限/续写/未知执行反例；与真实 demo 证据分开 |

Template 直接复用 normalize_directory/normalize_documents、validate_documents、validate_dispatch、validate_local_repositories、validate_resolved_file。
有效阅读视图用 ResolvedTask.effective_document(node) 或 effective_yaml，来源用 source_manifest/origins/effective_identity；非 root 不直接导出缺 Contract 的内部 effective_task 映射。
模块加载借鉴 extract 的独立 import 名方案，登记在 kernelschema 私有命名空间；禁止把两个 tools 路径同时塞进 sys.path 或导入裸 tools。
资源定位依包/已声明 checkout 根，不依赖 cwd、作者 home 或隐含 /opt/venv；规则冻结包含三份新流程角色文档，extract 的冻结清单不被顺便扩大。
基础依赖仅选 Python/PyYAML/jsonschema；Graphviz 只供绘图。torch/triton 和可选 Agate 依赖延迟加载，按执行 profile 锁定，不因读取 Schema 就导入 CUDA。
AgentCLI 只借“固定 prompt + 动态事实、精确角色 scope、核验原生终态、用量 unknown、不可覆盖产物”等模式。
它现有 worker/reviewer、YAML-only inventory、Bash allowlist、MCP 空集及 extract.session 绑定不能直接承担 coder；不 import AgentCLI 当新公共运行器。
必须独立适配的部分是新角色/代码权限、CLI 版本可用参数、原生结果解析、退出/超时结算与新会话资料投递；先只适配 Claude 一种。
首期 coder 只读 Task/Schema/规则，写 kernel.py；宿主在调用边界执行 evaluator，再把固定报告送回下一 coder 调用，避免给 coder 自行换判据或提交无限 GPU 作业。
不复用 extract 的 UUID、私有 home、冻结 run 或维护系统；CLI 权限不宣称完整宿主隔离，公开使用只运行所选可信 demo 的固定材料。

## 3. 拟议 v1 交接与产物

三项新能力使用同一版本化请求/结果壳，operation 为 design/optimize/implement；extract 仍走原生输入、退出码、状态与回执。
所有路径以下均指 bundle 内相对路径；“sha256:...”是文档占位。真实实现必须填实际摘要并验证，不接受占位入场。
request_id 是逻辑请求标识，不单独充当身份；request.json 的规范化摘要另存并绑定所有输入清单。输出位置由 CLI 指定，不纳入内容身份。
输入清单包括完整 Task/判据/evaluator、规则与工具版本、所有 raw/dispatch/真实引用来源、选择范围及资料可见性；不只保存一条 root 路径。
最小实现请求示意（计划）：

```json
{
  "protocol": "ks.workflow.v1",
  "operation": "implement",
  "request_id": "softmax-r0-impl",
  "task": {"manifest": "inputs/task/manifest.json", "sha256": "sha256:..."},
  "schema": {"manifest": "inputs/schema/manifest.json", "sha256": "sha256:..."},
  "rules": {"manifest": "inputs/rules/manifest.json", "sha256": "sha256:..."},
  "scope": {"nodes": ["root"], "coverage": "inputs/task/scope.json", "sha256": "sha256:..."},
  "seed": {"manifest": "sources/c0/manifest.json", "sha256": "sha256:..."},
  "budget": {"coder_calls": 3, "reviewer_calls": 2, "evaluator_jobs": 3, "wall_seconds": 1800},
  "backend": {"name": "reference", "runtime": "claude", "model": "<explicit-model>"}
}
```

以上限额只是调用例子，不是工期或已批准的实际运行预算；实际运行前由任务配置给出。evaluator 配置及通道能力在 Task 清单中固定。
design 省略 schema/seed，输入已确认的 Contract 与参考清单，budget 使用 planner_calls；optimize 必须给 schema，另给目标及可选 evidence/普通反馈引用。
公共必需项到此为止：请求/输入/范围/预算/后端、结果/证据/消耗；原生 session、episode、V0/V1、candidate branch、promotion audit 不进入公共字段。
可选扩展仅放 extensions.<backend>，默认消费者不得靠该私有内容判定 completed；首期不定义全部后端生命周期方法。
计划目录示意：

```text
runs/softmax-r0-impl/
  request.json                 # 固定交接，来源均可校验
  inputs/{task,schema,rules}/   # 各自 manifest；schema 内 raw/ 与 dispatch
  views/effective/             # 生成的阅读副本，不能再当 raw 节点读回
  sources/c0/, sources/c1/     # manifest + files/kernel.py；每份完整快照
  evidence/E0/                # intent、原始执行回执、规范化报告、环境/计时
  reviews/F0.json              # 有限独立 fidelity，准确绑定 R/C
  result.json                 # 原子发布的本次结果
  .runtime/                   # 私有单写者状态、调用/作业关联、消耗索引
```

每个 source manifest 列 entrypoint、完整可编辑文件清单、每文件 UTF-8 字节摘要与父快照摘要；首期只允许 kernel.py。
Task、不可改 helper/evaluator、依赖与包装元数据另存；绝不能把 AKA public-candidate 的 solution.json 自动当作可导入源码。
结果示意（计划，数字与摘要只示结构）：

```json
{
  "protocol": "ks.workflow.v1",
  "request_id": "softmax-r0-impl",
  "request_sha256": "sha256:...",
  "status": "completed",
  "stop_reason": "scope_accepted",
  "code": {"manifest": "sources/c1/manifest.json", "sha256": "sha256:..."},
  "coverage": {"requested": "pass", "task_uncovered": []},
  "evidence": {
    "correctness": {"path": "evidence/E0/correctness.json", "sha256": "sha256:..."},
    "fidelity": {"path": "reviews/F0.json", "sha256": "sha256:..."},
    "performance": {"path": "evidence/E0/performance.json", "sha256": "sha256:..."}
  },
  "performance_goal": "unmet",
  "usage": {"coder_calls": 1, "evaluator_jobs": 1, "tokens": null},
  "feedback": [],
  "continuation": null
}
```

evidence 引用必须可取得且有摘要；每份报告标 pass/fail/unknown/not_run、检查方法/有限范围、诊断与主体身份，build 信息可在执行报告内。
completed：本次明确范围的交付条件齐全；implement 要求正确性、必需 fidelity 通过、要求的测量已取得；不由进程退出 0、checker 或性能提升单独决定。
partial：已有草稿/代码但未完成或预算结束；blocked：缺外部资料/运行能力、未结算作业或有证据设计冲突；failed：协议损坏、不可解释执行错误等。
失败后仍保存已有可验证产物；普通编译/数值缺陷优先交 coder，耗尽后可 partial，不能自动归因于设计不可实现。
optimize 的 completed 还必须给 decision=keep/revise 与理由；keep 可直接结束，不强制写新 YAML。design 的 completed 只表示已完成所承诺的写作检查与交接。
continuation 仅为 {backend, token}，由后端绑定原请求、最新准确 C、规则/资料与累计预算；不可跨后端解读，也不能当作新余额。
无改码验收用 implement --verify-only：不启动 coder，但执行所需验收/审阅并计费；不得加注释伪造实现进展。
未来 CLI 调用示意（计划，非现有命令）：

```sh
kernelschema implement --request demo/fixed-r0.json --out runs/i0
kernelschema implement --continue runs/i0/result.json
kernelschema design --request demo/design.json --out runs/d0
kernelschema optimize --request demo/optimize.json --out runs/o0
kernelschema implement --request demo/verify.json --verify-only --out runs/v0
kernelschema extract -- <原 extract 子命令参数>
```

extract 包装只转入现有 extract 子命令并保留原始输出/管理入口；新结果如引用它，只引用原生 run/publication/receipt，不把 ready 改称已发布。
公开最小 Python API 与 CLI 使用相同请求验证和结果解释；无需分别写一套 prompt、Schema 解析或状态机。

## 4. 实现必须保持的规则

**身份与来源。**effective_identity 不含完整 Task、dispatch、规则/工具与评测环境，也会受 raw 排版/Feedback 字节影响；只能作为来源分量。
同 consumer Version 的 provider 变化也要冻结实际引用闭包；所有入场目录先按字节快照，不相信路径相同、mtime 或单个 Version。
反馈/排版变化保留新字节出处，但不自动计为新设计；结构/内容 diff 加受影响审阅决定是表示变化、证据更新还是设计修订，不建“语义等价 hash”。
改变 Task、验收器、dispatch 选择或规则后不得套旧 continuation；明确生成新固定请求，允许显式使用旧 C 作 seed，旧证据保留旧主体。
测量身份至少绑定 Task/判据、C、evaluator、输入/种子、实际设备/栈与计时；fidelity 另绑定 raw/有效来源闭包、规则和范围。
C1 测过后变成 C2，C2 是 unknown/not_run；同字节换 Schema 可经明确适用性核对引用旧数值测量，但重新判断 fidelity，不改旧报告标签。

**原始/有效 Schema 与权限。**原始来源、规范化候选、批准 raw、有效阅读视图分开；阅读展开不是批准。
任务方固定 Contract；planner 只改授权的 Design 和 Require/Prefer/Forbid；metadata、节点增删与 dispatch 更新须与实质设计变化相符并经规则检查。
Coder 不改 Contract/Design/Require/Forbid，也不直接写原始 Schema；允许提出外部 Prefer 替代提案，宿主核对后再由授权写作流程处理。
普通反馈记录 bug、覆盖缺口、资源故障、设计阻碍、瓶颈及未实测想法。只有针对 Prefer 且有 repository@ref:path 不可变实测证据的提案才可进入 REALIZATION.Feedback。
本地临时结果路径或 SHA 本身不等于合规 repository 引用；不足时提案留外部，不让 Feedback 成为绕过硬约束的入口。
extract 的硬件占位、skip-hardware 和容差来源豁免只适用于提取；实施准备须生成任务方已确认的完整实现输入副本，并保留原件/receipt。
手写、design、curated current.json、新 Runner receipt 均可准入；不能要求所有 Schema 有 P6 receipt，也不能继承 P6 作为实现验收。
材料清单还记录被提取原码是否对 coder 可见；原码复验、原码 seed 续写、仅凭 Schema 从共同骨架重建分别命名。重建需明确资料 allowlist/会话边界，前两者不能充当重建证据。

**候选与发布。**写作流程：冻结输入 → 草稿 → Template 规范化计划 → 冻结 Contract/角色 diff → 全任务检查 → 受影响语义审阅 → 不可覆盖发布。
normalize 返回 source_hashes 与候选；对候选调用 validate_documents，并在候选暂存目录核对 dispatch/本地引用，最终重读验证，不只展开或用 skip-hardware。
规范化只写新候选，保存原稿与改动说明；在真正发布前比较原来源摘要，遇并行变更拒绝覆盖，不碰活动提取目录。
发布先写齐 manifest/文件并 fsync，再原子 rename 到摘要目录，最后原子写 result；同 manifest 幂等、冲突拒绝，崩溃留下的暂存件不是已发布成果。
“Schema 已发布”和“下一实现请求已准入”是两件事；只有带新固定输入身份的请求才能使 coder 使用新设计，不能读 latest 目录猜测。

**代码续写、删除与执行结算。**每次已确认退出的 coder 调用后保存准确文件字节，分别保留 latest 与本 R 已验收 C；更慢的合格 C 不会被丢弃。
首期只有 kernel.py；删除该必需文件保留损坏草稿事实并拒绝执行准入，不能拿旧文件补回后声称保存了新代码。
后续源树能力使用完整文件清单：旧树中不在新清单的可编辑文件必须删除，不采用 overlay 拷贝；固定材料不能随删除变化。
导入新 seed 先完整暂存/核对，再激活工作区；intent 与 installed/admitted 事实分开，任何中断都不得让 coder 读半装源树或重装覆盖已开始的 draft。
单写者锁及小型状态记录足够；可从稳定 C 开新会话续写，不承诺搬运 CLI 隐藏上下文。规则/材料改变后不能复用冻结会话。
执行前持久化 operation/稳定提交键、输入与预算；known completed 取回原收据，known running 查询同 job，unknown 保留身份并停止新提交。
404、取消请求成功、CLI 超时、主进程消失均不能证明 GPU 未执行；没有可核对的终止/结算事实时返回 blocked，禁止换身份盲重提。
本地也须确认所属子进程结束；故障恢复只覆盖本工具已承诺的本地持久化和选中通道，不扩张为 AKA 级后台维护。

**预算与 planner 时机。**持久统计真实 planner/coder/reviewer/evaluator 用量，未知 token/GPU 用量保留 null；回放同收据不重计，真实新调用必须加计。
调用次数/任务额度在准入处强约束，总时长同时约束准入与子进程等待；模型 token 限额若只能调用后结算，文档明确其软边界。
wall_seconds 定义为累计活跃时长，含模型、排队、执行与审阅等待；显式暂停后的空闲间隔不扣款，但已用时长不重置。若任务另有绝对截止时间，两者同时约束准入。
evaluator_jobs 限制逻辑提交数，不冒称能硬限远端内部重试的物理 GPU 次数；实际 attempts/执行消耗从回执如实记账，无法取得保留 unknown。
暂停意图在私有状态中持久化，阻止新 coder/evaluator/planner；仍可吸收已有终态事实。显式 continue 只在身份、剩余额度和未结算作业检查通过后恢复准入。
新会话、新 episode、keep、恢复或新 Schema 均不能退款；demo 外壳持有一份总预算并给各子调用分配余额，不重复加算 AKA 原生账单。
固定 implement 内不调 optimize；partial/一般代码缺陷在同 R 继续，有预算/暂停/未知任务阻止时停止准入。
只有已授权 demo/AKA 外壳能在实现阶段完成，或出现独立可核对的设计冲突证据后交 planner；一次超时、一次“写不出”或无晋升不是触发理由。
首个 demo 只允许一次 optimize；keep 且无新增可执行补证行动则结束，revise 才发布新设计并明确旧代码是否作为新 seed。
发布范围可以是 Task 合法子集；结果必须列 task_uncovered。root 是阅读主干，dispatch 是设计目录，二者均不是自动路由/fallback。

**有限 fidelity。**首个 R0 建一张有限条目表：稳定 max-subtract 数值路径、FP32 归约/输出、逐行 work ownership、必要读写/寿命与 Require/Forbid。
机器项核对源码/运行 ABI、禁止调用参考/外部 softmax、固定范围/摘要及可机械识别结构；源码 AST/静态规则不能证明实际算法或同步语义。
独立只读 reviewer 读取准确 R/C、条目表与执行证据，逐项给 pass/fail/unknown、源码位置/理由；用不同会话，不能直接接受 coder 自评。
简单同步路径可给有范围的通过；复杂异步项无法论证则 unknown，必要时转人工审阅或 partial，不要求首期建设通用形式证明系统。
fidelity 收据在结果发布前持久化并绑定准确 C；其失败通常先修代码，若需改设计必须形成有据阻碍交 planner。性能再好也不能覆盖失败。

## 5. 可排序工作包

下列均为后续开发任务，无工期估计。每包只保留会推翻其承诺的必要验证；实现授权与本轮文档撰写分开。
主线：KS-01 → KS-02/03/04 → KS-05 → KS-06 → KS-07 → KS-08 → KS-09；KS-10 早开，KS-11/12 随后，KS-13 不阻塞独立发布。

### KS-01：锁定小任务及双通道资格（首个可立即执行包）
前置：实施阶段获准；读取最新仓库边界并在隔离开发副本工作，保留当前提取工作树。
拟改：examples/workflows/row_softmax/task/{task.json,contract.yaml,inputs.json,acceptance.md,materials.json}、docs/workflows/support.md；复用 R2 教学定义。
交付：上述精确 ABI/数值/副作用/计时合同、一个手工审查的 R0 草案、G0 表、Claude/H20/Triton 候选版本记录；明确 evaluator 应实现的每项检查。
必要验证：后续在干净环境查依赖/公开材料，按 G0 核实 Agate 或本地能力；只选一个实际通道/环境先生成资格证据，保留 Agate 差距。
完成标准：Task 无关键歧义，材料与资源缺项都有负责人/替代路径；GPU 能力未核实时明确禁止宣称运行资格已通过。
失败退路：仍推进 CPU 工程包，任务域保持既定小题；通道均不可用则真实发布阻塞，不以 private AKA 服务替代 D1。

### KS-02：唯一包、Template 适配与资源寻址
前置：KS-01 的任务形状；拟改 pyproject.toml、kernelschema/{resources,template_api,contracts}.py，复用现有 Template API。
交付：源码安装/CLI 骨架、协议读写与严格拒绝未知版本；角色/规则/Template 清单，运行不导入 AKA 或裸 tools。
必要验证：两个 tools 同时存在时无遮蔽；任意 cwd 与移动 checkout 可定位材料；少量现有 root/from 示例能走同一只读 API。
完成标准：静态准备能独立安装使用，结果标清 validation 范围；不改变 extract API/冻结协议。
失败退路：暂限源码 checkout 安装并说明，不先迁移 Template/extract 包或许诺 wheel。

### KS-03：一次 Claude 调用与新角色权限
前置：KS-02；拟改 runtime/claude.py、prompts/，借 AgentCLI 的终态/会话模式，独立实现适用参数与解析。
交付：planner/coder/reviewer 各自新 scope；单调用输入、原始事件、请求/报告模型、终止事实与用量；首期从稳定产物新会话继续。
必要验证：伪 CLI 覆盖非零退出、缺终态/错误 nonce、用量缺失；真实最小调用核实所锁 CLI 参数、code-only 写权限与材料只读。
完成标准：CLI 退出不等于领域成功，coder 不能修改固定材料，未确认结束时无后续准入；extract 会话不被复用。
失败退路：更换为该 CLI 已公开支持的等价参数/固定版本；不能仅加 prompt 代替权限或把旧 worker 当 coder。

### KS-04：Task evaluator 及 Agate 接线
前置：KS-01/02；拟改 examples/workflows/row_softmax/evaluator/、evaluation/task.py、agate.py。
交付：固定 evaluator JSON 协议、reference/输入/比较器/计时与完整源码摘要；一个通过 G0 的实际通道，另一通道有明确能力/缺项记录。
必要验证：正确候选、故意漏写 out/改 x/超容差候选分别判定；C1→C2 不错绑；Agate 若启用，实证稳定键/查询/取回且不偷换判据。
完成标准：原始执行证据与解释结果均可获取，数值/性能分别回传；无 GPU/未知远端终态时明确 not_run/unknown。
失败退路：Agate 不具备完整任务执行时同 evaluator 走本地；若不具备去重结算，只开放已结算新运行、拒绝未知作业自动恢复。

### KS-05：固定 Schema 有限 loop、保存与续接
前置：KS-02/03/04；拟改 implement/reference.py、artifacts.py；使用单 kernel.py 与不可变输入。
交付：准入→一次 coder→快照→evaluator→反馈→同 R 续写，有限预算、单写者、result/opaque continuation；verify-only 分支。
必要验证：C1 partial→新会话准确 C2，预算累计；已测 C1 后未测 C2 不继承 pass；保存/发布边界中断、未知作业禁止重提、删 kernel.py 明确拒绝。
完成标准：真实单题可生成并评价代码，预算结束保留产物；无改码验收不伪造注释；参考后端无 AKA 运行依赖。
失败退路：暂不支持原生 session resume，只从准确快照新会话继续；对损坏状态拒绝准入，不当新 run 重启。

### KS-06：有限 fidelity 与合格慢实现
前置：KS-05；拟改 fidelity.py、row_softmax/fidelity.json、结果聚合。
交付：有限机器条目与独立 reviewer 收据；requested coverage、Task 未覆盖项、性能目标独立投影。
必要验证：数值正确但调用参考/偏离 R 的实现不合格；未知项不 completed；正确、符合且已测的慢实现仍完整返回。
完成标准：首个真实固定 R/C 三类证据齐全，报告每类方法及有限覆盖；测量不由 reviewer 自评产生。
失败退路：机器难判条目保留审阅或 unknown；无法收敛则 partial，不放宽 Require/Forbid。

### KS-07：design/optimize 共享写作内核
前置：KS-02/03/06；拟改 writing.py、design.py、optimize.py，复用三份角色文档和 Template。
交付：固定 Task 的参考筛选、草稿/规范化/权限/全任务检查/有限审阅/发布；optimize keep/revise 与影响说明。
必要验证：Contract 越权被拒；from provider 改动触发 consumer 复核；排版/Feedback 不虚增设计；真实 design 与真实反馈的一次 optimize 均能返回合法结果。
完成标准：两入口实际调用模型并生成可交接材料，keep 可以完成；原始来源不被覆盖，所有必须决定已落实或显式 partial。
失败退路：先只支持单节点写作；多节点规划转人工/后续扩展，不复制 extract P1–P6 取得表面完整。

### KS-08：extract 消费桥与来源兼容
前置：KS-02/07；拟改 extract_bridge.py 与 docs/workflows/extract-to-implement.md，仅透传/读取。
交付：原 extract 子命令包装、已交付 raw/dispatch 准入；缺硬件/判据报告；任务方材料补齐后的独立输入副本。
必要验证：手写、design、curated current.json、新 receipt 四来源；skip-hardware/P6 不会绕过 implement 的任务资格；publication conflict 原样可见。
完成标准：旧 extract 命令与恢复不变，新消费不要求私有 run 或统一 P6 receipt；不对并行新提取结果做迁移。
失败退路：包装只保留命令透传，用户可直接用原入口；消费的缺项结果不得隐式修补原件。

### KS-09：有限公开 demo 与 README 第一次能力更新
前置：KS-05/06/07/08；拟改 examples/workflows/row_softmax/run_demo.py、README.md、docs/workflows/quickstart.md。
交付：先固定 R，再真实 design→implement→一次 optimize；keep 正常结束，revise 使用显式 seed/新验收；提供离线阅读产物与真实重跑入口。
必要验证：没有 AKA 的干净环境能按文档完成一条真实模型/GPU 路径；全部任务材料与证据可取得；有限预算/暂停不再准入 planner。
完成标准：README 分明标现有/首发/计划，说明限制、安装、未测与慢结果；不强制 optimize 制造修订，也不声称相对无 Schema 的收益。
失败退路：有产物则发布真实 partial 与缺项说明，但不得把回放/伪后端演示称为独立实现已交付。

### KS-10：FA4 decode 材料、范围与公开闭包预审（与 KS-01 并行开工）
前置：只读取得固定 20261019 来源；拟改 docs/workflows/cases/fa4_decode-admission.md 及新案例材料清单，不改既有 YAML。
交付：Contract/when/dispatch 覆盖表、完整比较器、源码/许可/依赖/设备映射，空 split/PageIndex/host 稳态限制和独立审阅需求。
必要验证：人工对照四节点与源码路由，q4_direct 子请求不冒充全 Task；能实际取得必要源字节，不把 receipt SHA 当下载地址。
完成标准：作出“可进入哪一子范围实现/还缺什么”的可审查结论；这是预审完成，不是 CuTe/SM103 实现或 GPU 验收完成。
失败退路：保留阻碍、继续小 demo；若改选候选，另存资格依据与原题失败记录，不使用更宽 Contract 的成功措辞。

### KS-11：明确新增源树保存/删除能力
前置：KS-05 稳定且真实题需要；拟改 artifacts.py、source capability 声明，AKA 对应支持由 adapter 另宣告。
交付：完整文件清单/删除、固定文件隔离、原子 seed 安装；辅助元数据与代码格式分别处理。
必要验证：两文件 C1→删旧文件的 C2、安装每个边界中断后只见完整 seed、已准入 draft 不被重装、非法路径/固定文件拒绝。
完成标准：本地后端完成声明源树的精确续写；AKA single_file 仍对超范围明确拒绝，不能因公共 manifest 存在就默认支持。
失败退路：维持首发单文件支持，复杂题后端按明确能力准入；不得 overlay 导入后谎称删除已生效。

### KS-12：真实复杂实现与必要 dispatch
前置：KS-09/10；若需多文件则 KS-11，另需可用 CuTe/SM103 profile 与任务 evaluator 资格；拟改新案例适配/交付目录。
交付：先完成一个 q4_direct 子请求，再按 Task 表逐项实现 root、q1_global、cluster 或选定 dispatch 联动；真实代码/覆盖/数值/有限 fidelity/性能。
必要验证：root 子域反例、来源复用闭包、host 准入/稳态前提、未覆盖输入明确拒绝；异步 unknown 不能被局部事件图代签。
完成标准：仅对实际实现并验收的范围声明支持；新栈资格和旧 Triton 支持面分别记录；原始提取限制不被静默修正。
失败退路：已完成子范围可交付，剩余 partial/blocked；有据设计冲突交 optimize，新设计不得冒充忠实重建原设计。

### KS-13：AKA 最小固定请求交接验收
前置：KS-05/06 与 AKA 固定 backend 的窄接缝通过；拟改 docs/workflows/aka-backend.md、协议 fixture，AKA 适配代码由 AKA 包拥有。
交付：同一 Task/R/scope/kernel.py/预算由两后端消费，AKA 回传精确源码、独立完成状态/三类证据/原生用量引用与 opaque continuation。
必要验证：未晋升 partial 能导出；慢但合格 C 不丢；AKA 无 fidelity 时 KS 不代签；未知/不兼容 continuation 明确拒绝。
完成标准：双方通过一个 fixture 映射与真实固定请求；公共合同不引入 native 私有字段，schema-off 不加载 KS。
失败退路：只发布稳定固定 backend，动态设计仍为下一显式请求；独立 demo 不等待 AKA 驱动/迁移完成。

### KS-14：分发闭包与 README 后续更新
前置：KS-09；复杂/AKA 能力只在其各自验收后加入；拟改打包资源规则、安装/能力表与根 README。
交付：固定规则/Template/角色 prompt/选中 demo 材料可随发行取得；CPU、模型、GPU/Agate 三层依赖分别说明，保留许可证与来源。
必要验证：干净 checkout 安装及无作者路径运行；若发布 wheel，再离开源码树验证全部必要资源；不上传模型 home、凭证、私有会话或整批实验目录。
完成标准：extract 原入口、三新入口、有限 demo、支持格式/栈/恢复界限可查；长期自动壳子列为未来 extract/design→implement→optimize，不冒称无人值守完成。
失败退路：先支持可复现源码发行，wheel/复杂栈/动态 AKA 分别保持未支持；不为“统一安装”整体重构 extract。
README 可按门槛增量更新：KS-02 标现有 extract/Template 与开发中入口，KS-05/06 标固定实现预览，KS-09 标真实有限 demo，KS-12/13 再列实际复杂范围与可选 AKA。

## 6. 与 AKA 的最小交接及不可越权事项

KS 交付固定 Task/判据/evaluator 清单、Schema raw/effective/来源、规则/范围、准确 seed、额度及公共结果语义；AKA 声明自身真正支持的格式/运行能力。
AKA 保留 native Task 权威，但其 evaluator 必须证明满足该完整判据；不同比较器或缺副作用验收时返回 unsupported_evaluator，不能用同名 Task/旧 numerical pass 代替。
AKA 拥有 native Task/Evaluation/Experiment/Execution、episode/session/draft、incumbent/晋升/恢复与私有计费；KS 不以公共状态替代这些权威事实。
AKA coder 必须在正文任务、工具、report 修复和续会话中固定设计；静态追加 YAML 不能抵消原有自由改算法/自动换栈指令。
KS 的 completed 与 AKA promotion 分开；正确慢实现可以前者成立、后者不成立。Git HEAD 也不是 incumbent 源码身份。
AKA 的源码 codec、seed 导入 intent/完成准入、normal/recovery fidelity/audit、V1 准备收尾与指定 episode/全局额度是 AKA 工作包，KS 不越界复制。
接口对齐先用单 kernel.py；public-candidate 编码、solution.json 与原生 checkpoint 均由 adapter 解释，不进入 KS 源树或公共 continuation。
依赖保持 AKA 可选固定 KS 版本；KS 基础安装/导入不依赖 AKA，KS 选择 AKA backend 仅向外部 adapter 发固定实现请求。
每次运行一个循环所有者：KS 有限 demo 调固定 backend，或 AKA 调 KS planner；AKA 不再调用会启动 AKA 循环的 KS 顶层 demo。
EXP 只消费固定发行配对、材料、预算/信息边界与产物；fork/submodule 物化、旧 overlay 迁移和远端操作由另一计划处理，本轮不执行。
需要对齐的最后技术项是 envelope 版本/摘要编码、Task evaluator 映射、source codec、unknown 结算投影与用量收据去重；这些是技术验证，不重问 D1–D3。

## 7. 依据与事实边界

主要依据：[用户已确认方向](../03_decision_questions.md)、[R1 KS 盘点](../research/r1_schema_inventory.md)、[R2 独立契约](../research/r2_standalone_contracts.md)、[R3 反例](../research/r3_architecture_challenge.md)、[R3 交付审查](../research/r3_delivery_evaluation.md)。
本轮只读核对：[Template 真 API](../../../schema_template/tools/README.md)、schema_template/tools/schema.py 与 design_resolution.py；extract 的 artifacts.py:46、agent.py:27、backends.py:171。
案例证据：[FA4 decode](../../../schema_template/examples/fa4_decode/README.md)及其 root/dispatch；[QSA 比较器反例](../../../schema_template/examples/qsa_prefill/README.md)。
Agate 观察来自 AKA 随附 reference-dir.md、eval-manifest.md、jobs-and-troubleshooting.md 及 gateway/submission.py；未查询真实 endpoint，也未核实当前服务/SDK/租户能力。
本轮未新增任何运行通过结论。所有 G0、正反例、真实模型/GPU、安装及后端映射验证都属于以上未来工作包。

