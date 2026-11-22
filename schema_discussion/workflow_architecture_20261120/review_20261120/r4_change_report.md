# R4：按独立裁决修改当前计划的实际变更

日期：2026-11-20。执行者：用户指定的最后一轮全新 xhigh 修改代理。修订基准为[R3 独立裁决](r3_adjudication.md)，已确认 D1–D3 不变；主代理只负责导航/历史与最终一致性。
本轮实际改写四份当前正文并新增本文，未改实现、原 Schema、决定页、before 快照或前轮报告；README/历史更新由主代理另做。全部实施仍未开始，没有模型/GPU/服务运行、安装、提交、远端或凭证操作，未派子代理。

| 简称 | 当前唯一职责 | 实际处理 |
| --- | --- | --- |
| [P](../04_implementation_plan.md) | 方向/仓库与循环边界/阶段/发布门/维护原则 | 删除全量合同、Task 数值、状态表、第二 DAG 与重复公开清单，直链 C/V/W |
| [W](../05_work_packages.md) | 17 单元/owner/改动区域/交付/唯一依赖与首批 | 从旧 34 单元改写为 17 单元和明确子 gate；不保留旧 ID 墓碑 |
| [C](../planning/r5_contract_execution_review.md) | 唯一 API/身份/闭包/source/恢复/预算/结果/fidelity | 就地改为当前合同，撤销旧恢复/全历史导入/预算重复义务；只留一个有真实字节摘要的 source JSON 例 |
| [V](../planning/r5_validation_delivery_plan.md) | 唯一 Task/F1/V01–15/公开材料/pilot | 修完整正确性、当前 R 方法、V0/C0、恢复反例和候选选择；删旧 owner/DAG/发行时序副本 |

C/V 的既有路径保留，标题/导语已成为当前规范；原 R5 推导直接链接 before 原字节快照，不再由“主计划优先”覆盖错误附录。研究/R4/审查文本仅为历史依据。

## 1. 逐项裁决落实

下表“完成”均指文档修改已落地；反例的工程/真实执行仍是未来验收。

| 已落实裁决 / 原审计标识 | 保留、修正或删除的实际内容 | 当前规范与交付/验收落点 |
| --- | --- | --- |
| [x] R3-C01 / C-01 | 三个新操作统一按 run 目录恢复，首笔费用前存锚点，首 result 前仍可结算/重建；写作无行动不再付费 | C §1/7/8；K2/K4/K5/A5；V08 的 implement、V09 的 design/optimize |
| [x] R3-C02 / C-02 | Q 与 run_id 分开；同 run 收据幂等、显式新 run 独立计费、同父新 Q/R/out 不重置；删除跨任意目录全局去重暗示 | C §2/6/7；K2/K4/A5/X2；V08 三轨迹 |
| [x] R3-C03 / C-03 | 已终止统计 null 不一律锁死；操作/结算或限额余额未明才 fence；检查 unknown 有据可补证但不 qualified；历史 qualified 不掩盖未结算 | C §6–8 唯一状态；P 仅引用；K4/A5；V02/V07/V08 |
| [x] R3-C04 / C-04 | 按有私有真实依赖的条件冲突修正规范，公开说明/可导入包/私有归档分清；脱敏不替旧 Ref | C §3、V §6；K2/K6/A2；V06/V13 |
| [x] R3-C05 / C-05 | Materials 的唯一 workflow_objective 承载当轮目标，planner/reviewer 可见，变化进入 Q、不得放宽 Task/scope | C §1、V §2；K2/K5/K6/X2；V09 缺失/重复/变化/越权 |
| [x] R3-FID / C-06、E-03 | 固定有限方法解释当前完整 R/scope；R0 清单只作 fixture，逐义务/Schema/源码定位及理由；新不明义务 unknown、新 R 重判 | C §5、V §2/5；K4/K5/K6/A5/A6；V02/V09 配对 |
| [x] R3-E01 / E-01 | 原 out 对象/storage 活引用/view/输出区与外围硬检查；原 72 次内每 shape seed23 设 x offset3/out offset5；加同对象换 storage 坏例 | V §1/3/V01 唯一 Task；K1/A2；两实际 adapter 正反例，未运行 |
| [x] R3-E02 / E-02 | native V0 使用 Task.baseline_source；不可执行骨架/partial 是首次 coder C0；V0/C0/G/HEAD 分列，恢复不重做 V0/盖 draft/退 G；准备及 review 真费用保留 | V §3/V11、C §6/7；A2/A4/A5；KS 独立链不等待此门 |
| [x] R3-E04 / E-04 | 删除最快观测选终态；最后登记合格候选，计时仅描述性观测，不承诺无偏/独立 | V §7/V09；X2；不增加独立复测 |
| [x] R3-S01 / S-01 | 删除完整 artifacts 历史复制；按选定对象真实语义递归闭包，保留 parent/raw/source 实体/原生依赖、稳定 prefix/原字节/模式/整次冲突拒绝；证据裸摘要主体须可定位 | C §2/3、V §6；K2/A2/K6；V06/V09/V13 |
| [x] R3-S02 / S-02 | 删除公共 continuation 核心字段、result 路径/token 门票与兼容别名；保留 recovery 提示、私有 native path/版本/key/句柄和收费前宿主→adapter 关联 | C §1/7/8；K2/K4/K5/A5；V08/V09 外部首返回丢失 |
| [x] R3-S03 / S-03 | 删除三套 tools/run.py 与能力目录配置副本；保留唯一 CLI/Python API、原 extract 透传、三份简短能力 README | P §2、C §1、V §6；K2/K6；V13 任意 cwd |
| [x] R3-S04 / S-04 | 四正文按唯一职责就地收敛，删除 C/V 历史整合指令/重复 DAG/spike/发行表；旧 inputs/sources/evidence 示例退出 | P 导航、W/C/V 当前导语；本轮离线文档检查 |
| [x] R3-S05 / S-05 | 34→17，材料/接口/CPU/真实子 gate 分开；无额外总补丁包/整包互等 | W §1–4；38 子 gate/87 条含条件依赖离线无环 |
| [x] R3-S06 / S-06 | 独立跨后端包拆入 A2 fixture 与 A7.pair；A7.fixed 不等待 KS 真实链/配对/EXP；同主体记录复用、两边真实资格分别取得 | C §4、W A2/A7、V03/V14；原三关键坏例加换 storage |
| [x] R3-S07 / S-07 | 保留外部已安装 AKA；单 owner 预约/暂停/真实收据贯穿，包含 native 预取并发；单活动时钟投影当前 Q；删除通用账户树/区间引擎及逐行强制 active_ms | C §6；K4/A5；V08/V11 |
| [x] R3-S08 / S-08 | 固定方法读取当前 R，删清单生成链/审批包意图，不让 planner 自选合格标准 | 与 R3-FID 同一 C §5/V02/V09，不增第二规则 |
| [x] R3-S09 / S-09 | 候选冻结时首次登记 (R,C)，迟到/重测/重放不编号；新 R 合格能交付，未完新 R 展示历史所属；删 best-of-budget、额外 P/C 臂与预排 F1 第二批 | V §7/V09；X2；全样本/9-run 建议及资格后冻结预算保留 |
| [x] R3-S10 / S-10 | 一份 fidelity 报告含 coverage 映射/未覆盖及设计义务，两个判定独立；删除独立 coverage 文件强制义务 | C §5/8；K4/A5；V02/V06/V15 |
| [x] R3-S11 / S-11 | 删除 wheel/离树 gate、独立例行更新包和抽象扩栈包；F1 真实现/必要格式纳 K8，选 AKA 才含其案例适配；F2/F3 不预排，当前请求必要 dispatch 不删 | P §1/4/5；W K7/K8；V §4/V15；源码任意 cwd/许可/固定版本保留 |
| [x] R3-J01 / J01 | change_report 先冻结并由 Schema 引用，终态 writing_review 后绑定冻结 Schema/scope，Schema 不反向引用；复用审查事实避免两套完整审阅 | C §3、K5、V09 无哈希环 |
| [x] R3-DOC | P/W/C/V 各有唯一职责与直链，决定页未改、R4/before 不再兼作规则 | 四导语、P 导航与本报告 |
| [x] R3-WORK | 最终 17 单元全部重写，旧 34 去向仅留本报告/裁决；保留每单元可开工输入、交付、必要证据 | W §1–6；下节逐旧 ID 对照 |
| [x] R3-GATE | 首批材料/复杂预审/基线→A1；接口/CPU/真实分开，AKA-P1 仅 A7.fixed；X2.protocol→X1→X2.run | W 唯一依赖表及 P 发布门 |
| [x] R3-CHECK | 有限检查项均有明确合同/验收/owner，撤销义务从有效文本删除 | 本报告 §4，均为离线规范核对，不冒称反例运行 |

## 2. 旧 34 单元完整去向

此表仅作历史迁移索引，当前开工只使用 W 的 17 单元。

| 旧 ID | 新归属/删并结果 |
| --- | --- |
| KS-01 | K1.materials：首题 Task/R0/材料 |
| KS-02 | K2：唯一包/合同/资源/codec/锚点 |
| KS-03 | K3：Claude 单调用/权限与真实 profile |
| KS-04 | K1.evaluator/real：完整 evaluator 与所选通道 |
| KS-05 | K4：固定实现、准确 C、预算/恢复 |
| KS-06 | K4：当前 R 方法/合报告/qualified |
| KS-07 | K5：共享写作与新 R 发布/交接 |
| KS-08 | K2：原 extract 透传/多来源消费桥 |
| KS-09 | K6：真实小 demo 与 evidence |
| KS-10 | K7：FA4 来源/F0/F1 预审 |
| KS-11 | K8：仅 F1 实际需要的 source 格式/删除/安装，不留抽象扩展包 |
| KS-12 | K8：真实 F1；后续域按结果与实际请求决定 |
| KS-13 | A2：对接 fixture/语义映射；A7.pair：双方证据索引 |
| KS-14 | K2：包资源建设；K6：当前源码发行/公开说明；wheel 义务删除 |
| AKA-01 | A1：指定 e/driver/生命周期接点 |
| AKA-02 | A2：Task/source 映射与真实 hook |
| AKA-03 | A3：可选模块/安装/off |
| AKA-04 | A3：角色正文/资料及来源归口 |
| AKA-05 | A4：V0/V1/helper/慢 C 收尾职责 |
| AKA-06 | A4：C0/跨 e seed 与准备持久边界 |
| AKA-07 | A5：资格/三 audit/verify_only |
| AKA-08 | A5：完整公共结果/准确 C/native 扩展 |
| AKA-09 | A5：预算/暂停/外部定位/固定恢复门 |
| AKA-10 | A6：有限动态 planner/R 准入 |
| AKA-11 | A7.fixed：独立 fixed 源码发行 |
| AKA-12 | K8 仅在 F1 选择 AKA 时的实际 adapter 子交付；独立抽象扩栈包删除 |
| MIG-01 | 实施前各仓基线/隔离检查记录，不再是产品单元 |
| MIG-02 | A3 归口必要迁入/来源；A1/A4/A5 各自实现接点并留来源，A7 发行汇合 |
| MIG-03 | A3 安装/固定源码接口；A7 独立发行配对 |
| MIG-04 | X1：生成式 vendor/KS 单独物化/新 capture |
| MIG-05 | X1：唯一产品来源/旧 capture/回退与能力消费 |
| MIG-06 | 工程包删除；P 保留独立更新、新配对、旧 capture、先 off 后受影响资格四原则 |
| EXP-01 | X2.protocol：B/F/D 预登记/CPU 配置 |
| EXP-02 | X2.run：资格后有限运行与全样本分析；额外臂/预排 F1 第二批删除 |

最终单元：K1 材料/evaluator、K2 核心/消费桥、K3 runtime、K4 fixed/fidelity、K5 writing、K6 demo/源码公开、K7 FA4 预审、K8 真实 F1；A1 driver、A2 Task/source、A3 可选/角色/来源、A4 V0/C0/续写准备、A5 fixed/恢复/预算、A6 dynamic、A7 发行/配对；X1 EXP 消费、X2 pilot。
首批仍为实施基线记录后并进 K1.materials、K7.materials、A1.cpu；其后 K2.interface 供 K1 evaluator/K3/K4/K5/A2/X2 分轨。K8 工程不等 README 收尾，AKA-P1 不等 A7.pair/KS 真实链/EXP。

## 3. 编辑时的最小澄清

- 将 durable run_id 显式列入公共结果以便与恢复锚点关联；它不进入 Q、不成为全局去重服务，未新增用户恢复门票。
- C 的合报告明确 coverage.mappings/task_uncovered 与 fidelity.items 各自状态，避免“合一份报告”被误读成合一个通过标签；结果两个 Check 可指同 Ref。
- 写作变更报告与终态审阅只共享已完成审查事实，结果 Check 引用终态证据，Schema 不回指它；终态失败不改冻件而另发修正，避免编辑时隐性哈希环。
- W 将 K5.real 的 optimize 反馈写成 K4.real 或 A5.real；K6 无 AKA demo 必须消费 K4 的真实反馈。A6.real 依模型资格并在当次验证写作，不要求先完成独立 KS GPU demo。它细化裁决的条件依赖，没有修改产品门。
- V 分清手审 fixture R0 与 pilot 默认选自真实 design 的固定起点 R0；命名相同不意味着设计相同，资格始终绑具体 Schema。
- 删除原大块占位 JSON，只保留一个真实 `pass\n` 字节摘要的 source 结构例，明确它不是可执行 Task；正式请求模板仍由 K2/K6 交付完整真实 Ref/摘要。没有以主计划覆盖旧错误路径。

以上均为裁决内的表述收口，没有推翻 R3 或 D1–D3，没有新增工程包、服务、实验臂或运行授权。

## 4. R3-CHECK 的规范归属与实际离线核对

| 必核反例组 | 已存在的唯一规范/owner/验收落点 | 本轮结论 |
| --- | --- | --- |
| 三入口首 result 前、外部 AKA 定位、三类费用、历史 qualified 与 fence | C §2/6–8；K2/K4/K5/A5；V08/V09 | 文本覆盖完整，未来运行未开始 |
| settled reviewer unknown 对 unknown job、外部额度/V0/预取/暂停时钟 | C §6/8；K4/A4/A5；V02/V07/V08/V11 | 两类 unknown 已分清，单 owner 不丢并发事实 |
| Schema parent/raw、选中证据主体、无关日志、脱敏、冲突、无环写作 | C §3；K2/K5/K6/A2；V06/V09/V13 | 真依赖与无关历史区分明确，稳定路径/字节保留 |
| V0 baseline、不可执行/partial C0、draft/G/HEAD、out storage/offset | V §1/3；K1/A2/A4/A5；V01/V04/V05/V11 | Task 硬项/来源分工留全，未用 native baseline 代签 R |
| 当前 R0/Rd 相反结论、新义务 unknown、scope/when/dispatch、合报告分判定 | C §1/5、V §2/4；K4/K5/A5/K8；V02/V09/V15 | 方法/fixture/当前义务分清，不以数值正确替符合 |
| 新 R 入终态、未完新 R 历史所属、旧收据晚到、描述性全样本 | V §7；X2；V09 | 三短轨迹及耗时变化不影响选择的检查已写入，未承诺无偏 |
| 不可误删内容 | P D1–D3/阶段/门；C 准确 C/latest/qualified/G/HEAD/固定 assignment/预算/三 audit；W 独立 owner；V off/公开源码/F1/旧 capture/必要 dispatch | 均有当前归属，不依赖历史报告补规则 |

实际执行的是文档/离线一致性核验：

- 四正文共 17 个唯一单元标题，全部标未开始；W 提取 38 个子 gate、87 条直接及条件引用边，无未定义 gate、无环；将所有“或”分支均视为可能边仍无环。A7.fixed 不依赖 A7.pair、K4.real、K6 或 X1/X2；K1/K2、A3/A4/A5、X1/X2 没有整包回边。
- V01–V15 各一行，W 覆盖全部验收 ID；C 为唯一字段/状态/预算规则，V 为唯一 Task 数值/测试规则；P 只链接规则，W 不重复判据。
- 对四正文检索旧工作包编号、continuation、旧 result 恢复命令、三套 tools/run.py、wheel、best-of-budget、完整 artifacts 历史复制、活动区间并集义务和旧整合导语，均无命中。旧 ID 只在本报告与历史裁决/快照中保留。
- 五份修改文档的 52 条本地 Markdown 文件/显式锚点链接已离线检查，无失效项；C 唯一 JSON 块可解析，size=5、UTF-8 `pass`+LF 的 SHA256 与示例一致。它是文档结构例检查，不是运行代码或 JCS 实现测试。
- review baseline.json 的五份 before 快照逐字节摘要、长度、行数均匹配，03 决定页摘要仍与基线相同；交回时记录的 14 份受保护文件（含前轮报告/基线/导航）摘要均未改。本代理只写授权的四正文和本文。
- R3-CHECK 每组均已落表核对；没有执行任何 V 的实现/模型/GPU/服务资格测试，因此不能称代码、GPU 或产品发布门通过。

## 5. 规模变化

以 baseline.json 的原始字节/行数对比修改后文件；行数只作维护规模记录，不作删减目标。

| 文件 | 修改前行数 / 字节 | 修改后行数 / 字节 |
| --- | ---: | ---: |
| P / 04_implementation_plan.md | 196 / 28,729 | 88 / 11,004 |
| W / 05_work_packages.md | 232 / 24,969 | 198 / 19,941 |
| C / r5_contract_execution_review.md | 378 / 47,507 | 175 / 27,293 |
| V / r5_validation_delivery_plan.md | 321 / 41,446 | 227 / 33,192 |
| 合计 | 1,127 / 142,651 | 688 / 91,430 |

缩减主要来自撤销的义务与重复规则/依赖/发行表；必要反例、真实 F1、独立真实门、源码公开与预算恢复边界均留在唯一当前正文中。计划修订已完成，17 单元实施均未开始。
