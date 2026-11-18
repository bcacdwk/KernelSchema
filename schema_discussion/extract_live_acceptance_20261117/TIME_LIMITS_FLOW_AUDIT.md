# 当前 extract 流程时间限制审计

审计日期：2026-11-18。对象为当前工作树 `schema_extract`，不是试跑各 run 的冻结执行器。只读检查代码、说明及已有耗时报告；未修改产品、配置或历史记录，未启动提取或真实模型调用。本文件覆盖流程主预算；原生 CLI、进程回收及 monitor 内部时间参数由同轮运行时审计覆盖。

**当前默认首个 worker 的可用调用时间约 1200 秒（20 分钟），不是界面参数的 1800 秒，也不是 r01 的 900 秒。** 默认整算子只有 7200 秒（2 小时），每阶段/P5 节点 3600 秒（1 小时）。这些数没有按此次完整提取的实测重新定稿。

## 1. 参数、起算与作用

| 限制 | 当前值/公式 | CLI 可配置性 | 起算、扣减及实际效果 | 代码依据 |
| --- | --- | --- | --- | --- |
| 整算子 `operator_seconds` | 默认 **7200 秒** | 新建 `extract --operator-seconds N`；正整数，无代码规定的数值上界，无 unlimited 选项 | 公开 CLI 从注册 run 的 `created_at` 起算；冻结准备、排队、暂停、维护、恢复和重试等待均消耗绝对墙钟。影响派发额度、拒收完成结果、孤儿恢复；不是一个独立常驻精确定时 kill 闸门 | [runner.py:43](../../schema_extract/tools/runner.py#L43)、[monitor.py:540](../../schema_extract/tools/monitor.py#L540)、[monitor.py:554](../../schema_extract/tools/monitor.py#L554)、[store.py:312](../../schema_extract/tools/store.py#L312) |
| 阶段/P5 节点 `stage_seconds` | 默认 **3600 秒** | 新建 `--stage-seconds N`；同为正整数，无数值上界 | 在该 scope 首次 `prepare_attempt` 时设 `started_at + S`，早于真实 CLI 启动。P1/P2/P3/P4/P6 各一份；P5 每个非 root 节点一份，P5 汇总阶段没有另一份实际派发预算。首次调用前的依赖等待不扣尚未开始的 scope，却扣整算子；开始后的等待一直扣 | [store.py:856](../../schema_extract/tools/store.py#L856)、[stages.py:61](../../schema_extract/tools/stages.py#L61)、[stages.py:74](../../schema_extract/tools/stages.py#L74) |
| 普通 worker/reviewer 单调用 `call_seconds` | 默认名义 **1800 秒**，实际还要取剩余时间减预留 | 新建 `--call-seconds N`；同为正整数，无数值上界；普通角色共用，不能分别设置 | `runner` 派发前计算一次，交给 AgentCLI 的单调用硬 timeout。包括该调用准备/原生 CLI 执行等时间的实际起算细节见运行时审计。可能在 1800 秒之前被流程剩余预算缩短 | [budget.py:49](../../schema_extract/tools/budget.py#L49)、[runner.py:1425](../../schema_extract/tools/runner.py#L1425) |
| 裁决/签收各自单调用 | `A=min(C,S/4,O/12)`；默认 **600 秒**各一次 | 没有独立 CLI 参数；从三个配置值导出后冻结 | 裁决时还为签收留 A；签收无后续预留。裁决/签收已用一次就不能重试，超时或检查预算耗尽也会成为一次性阻塞 | [runner.py:73](../../schema_extract/tools/runner.py#L73)、[budget.py:31](../../schema_extract/tools/budget.py#L31)、[store.py:868](../../schema_extract/tools/store.py#L868)、[runner.py:332](../../schema_extract/tools/runner.py#L332) |
| 普通独审预留 | `R=min(C,max(0,(min(S,O)-2A)/2))`；默认 **1200 秒** | 没有独立 CLI 参数 | 只在普通 worker 当前额度里额外扣 R；这不是 reviewer 的独立固定上限。reviewer 实际仍按 `min(C,剩余-未用裁决/签收预留)` 算 | [budget.py:26](../../schema_extract/tools/budget.py#L26)、[budget.py:39](../../schema_extract/tools/budget.py#L39) |
| 完成回执机器处理预算 | `max(0,min(120,剩余-当前角色预留))`；最多 **120 秒** | 硬编码，无 CLI 参数 | `accept_result` 入口取额度，覆盖原始证据保存、规范化、检查、合并、批准刷新等。同步处理不会到 120 秒立即中断；最终 `complete/approve` 处理超额度或余额≤0 会保稿并拒绝本轮放行，即使检查本身已通过。不是单独再给 120 秒 | [budget.py:61](../../schema_extract/tools/budget.py#L61)、[runner.py:551](../../schema_extract/tools/runner.py#L551)、[runner.py:737](../../schema_extract/tools/runner.py#L737)、[runner.py:268](../../schema_extract/tools/runner.py#L268) |
| 裁决候选准备 | 同一个 `check_timeout` 公式，最多 **120 秒** | 硬编码，无 CLI 参数 | 在裁决调用准备中遍历候选；每个候选开始前检查 elapsed，耗尽就停止增加候选。单个候选的规范化/检查不会被中途打断，因此可超额。这是另一个计时窗口，不与上行接收检查共用计时器 | [store.py:752](../../schema_extract/tools/store.py#L752) |
| 回执修复/普通执行失败退避 | `min(60,5×2^min(4,max(0,failures-1)))` 秒 | 无 CLI 参数 | 第一次失败等 **5 秒**，第二次 **10 秒**；更多累计失败数学上为20/40/60秒，但默认自动重试计数门通常已停止。等待消耗既有 operator/scope deadline；是最短等待，不是任务超时 | [recovery.py:32](../../schema_extract/tools/recovery.py#L32)、[runner.py:786](../../schema_extract/tools/runner.py#L786)、[runner.py:1403](../../schema_extract/tools/runner.py#L1403) |
| 孤儿调用超时接管 | 任一 `operator_deadline`、`scope_deadline`、`attempt.created_at + timeout_seconds` 到期 | 无独立 CLI 参数 | 原 Runner 丢失、但兄弟调用还活着时，每约 1 秒检查一次，恢复/结清已到期调用。这里单调用起点采用 attempt 创建时间，与 AgentCLI 自身起点略有差异 | [runner.py:1248](../../schema_extract/tools/runner.py#L1248)、[runner.py:1387](../../schema_extract/tools/runner.py#L1387) |
| 独立 `review-eval` | 默认每 case **240 秒**，允许 **1..1800 秒** | `review-eval --call-seconds N` | 每 case 仅调用一次 reviewer；顺序评估，没有整套 operator/stage deadline、上述阶段预留或120秒接收门。前置生成/检查不在该单调用 timeout 内；累计总耗时可以超过 `case数×call_seconds`，还含准备和结清 | [cli.py:153](../../schema_extract/tools/cli.py#L153)、[review_eval.py:120](../../schema_extract/tools/review_eval.py#L120)、[review_eval.py:164](../../schema_extract/tools/review_eval.py#L164) |

### 1.1 层层扣减的准确公式

记 `O=operator_seconds`、`S=stage_seconds`、`C=call_seconds`；`A`、`R` 如上。当前时刻 `t` 的可用总窗口：

```text
T = min(operator_deadline, scope_deadline 或 (t + S)) - t

未用的裁决次数 a = 1 或 0；未用的签收次数 s = 1 或 0
普通 worker 预留 = (a+s) × A + R
普通 reviewer 预留 = (a+s) × A
裁决调用预留 = s × A
签收调用预留 = 0

普通调用 timeout = min(C, T - 预留)
裁决/签收 timeout = min(A, T - 预留)
机器处理 allowance = max(0, min(120, T - 预留))
```

`T-预留 <= 0` 就不派发。普通模式进入 `budget_exhausted`，裁决/签收进入 `blocked`。保留的时间不是自动可用的兜底池：普通额度耗尽不会自动切到裁决，现有裁决入口是退修计数到限。因此可能留下大量预留，同时 worker 因时间不足停止。[budget.py:49](../../schema_extract/tools/budget.py#L49)、[runner.py:1426](../../schema_extract/tools/runner.py#L1426)、[stages.py:578](../../schema_extract/tools/stages.py#L578)

实际 `timeout` 在 `prepare_attempt` **之前**算好，准备 workspace、候选、材料和 prompt 后没有重新计算。准备期间绝对 deadline 继续流逝，因此正常进程主要靠单调用 timeout 终止，整算子/阶段并非精确的实时独立 kill 边界。接收时会再次检查余额，形成“模型已做完，流程仍因时间拒收”的路径。[runner.py:1425](../../schema_extract/tools/runner.py#L1425)

### 1.2 不能把 r01 的 900 秒直接当当前默认

以下假定新 run 立即派发第一阶段、整算子未因准备/排队先接近到期；数值为派发上限，不是预期完成时间。

| 配置 | O 整算子 | S 阶段 | C 单调用 | A 裁决/签收各自 | R 独审预留 | 首 worker 上限 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **当前产品默认** | 7200 | 3600 | 1800 | **600** | **1200** | **1200秒 / 20分钟** |
| 试跑 r01 | 21600 | 3600 | 1800 | 900 | 900 | **900秒 / 15分钟** |
| 试跑 r02 | 86400 | 28800 | 5400 | 5400 | 5400 | 5400秒 / 90分钟 |
| 试跑后续新 run | 86400 | 36000 | 7200 | 7200 | 7200 | 7200秒 / 120分钟 |

当前默认计算：`A=min(1800,900,600)=600`，`R=min(1800,(3600-1200)/2)=1200`，worker 得到 `min(1800,3600-600-600-1200)=1200`。若 worker 真用了1200秒，接下来 reviewer 通常也只剩约1200秒，还要扣准备、机器检查及角色交接耗时。若 worker 很快完成，reviewer 可以得到最多1800秒，故不能把 R 误称为 reviewer 固定timeout。

反直觉现象：**只把整算子上限从7200加到21600，首 worker 反而由1200降到900秒**，因为公式同时增加裁决/签收预留。只提高 `--call-seconds`、不改 O/S，也不能突破当前首 worker 的1200秒。这是现行公式产生的行为，不是报告中的估算误差。

## 2. 追加、暂停、维护及恢复

| 操作/状态 | 时间规则 | 次数/决定规则 | 依据 |
| --- | --- | --- | --- |
| `resume RUN --add-seconds N` | 默认 N=0；正追加把整算子和**每个已启动**scope的 deadline 改成 `max(now,旧deadline)+N`。到期后的等待不会要求先补足。未启动 scope 仍在首次调用时只拿原 S；C/A/R 均不变 | 不清除用量、修复数、裁决或签收决定；按授权 identity 只追加一次 | [budget.py:65](../../schema_extract/tools/budget.py#L65)、[cli.py:212](../../schema_extract/tools/cli.py#L212) |
| `--add-calls N` | 不加时间 | 对每阶段/节点有效上限加 N，已开始和未来 scope 都适用；不是全算子共享 N 次 | [budget.py:22](../../schema_extract/tools/budget.py#L22)、[cli.py:215](../../schema_extract/tools/cli.py#L215) |
| `resume` 无追加 | 不移动任何 deadline；若可用余额仍不足，不会重新派发 | 保留原计数；不能重授一次性裁决/签收 | [runner.py:1338](../../schema_extract/tools/runner.py#L1338)、[runner.py:1097](../../schema_extract/tools/runner.py#L1097) |
| `--scope` 恢复、`resume --all --hold ...` | 不追加、不退还时间；CLI禁止同时给 add-seconds/add-calls | 原次数/决定保留；单scope恢复只处理其绑定失败，不改健康兄弟 | [cli.py:370](../../schema_extract/tools/cli.py#L370)、[cli.py:385](../../schema_extract/tools/cli.py#L385) |
| queue / pause / hold / 等待确认 | 时间照走；`--queue` 帮助中的“resume starts”指启动执行，不代表resume才启动operator计时。pause在调用边界停止派发，cancel才请求终止在途 | 不清除任何已用计数 | [cli.py:100](../../schema_extract/tools/cli.py#L100)、[cli.py:118](../../schema_extract/tools/cli.py#L118)、[README.md:88](../../schema_extract/README.md#L88) |
| 维护 review/patch/verify | 没有独立维护时限，继续使用所选原阶段/P5节点与整算子的旧deadline及同样的角色预留。维护drain不自动加时，维护任务顺序调度 | 共用scope调用计数。维护review→patch→verify不是一套新阶段额度；被复用scope的裁决/签收计数也不清除 | [maintenance.py:158](../../schema_extract/tools/maintenance.py#L158)、[maintenance.py:208](../../schema_extract/tools/maintenance.py#L208)、[stages.py:123](../../schema_extract/tools/stages.py#L123) |
| reopen / 依赖变动 / 重新终审 | 保留原deadline，已批准很久后再回到旧阶段可能立刻预算不足 | 旧计数和一次性决定保留；新generation不是重新授额 | [stages.py:329](../../schema_extract/tools/stages.py#L329)、[stages.py:344](../../schema_extract/tools/stages.py#L344)、[stages.py:395](../../schema_extract/tools/stages.py#L395) |
| 调用或检查崩溃后的账目 | 有result/event/finished则读真实记录；缺失时保守用 `now-attempt.created_at`，未结检查区间也估计计入 | `elapsed_seconds`按attempt最高水位去重；**它是统计，不是再从deadline扣第二次**。预算派发使用绝对deadline，而非elapsed累计数 | [recovery.py:288](../../schema_extract/tools/recovery.py#L288)、[recovery.py:371](../../schema_extract/tools/recovery.py#L371) |

P5 可并发就绪兄弟节点；每节点保有自己的 S，所有节点共用同一 operator deadline。并发调用时长之和可以超过算子墙钟，不表示预算被重复扣除。当前没有 `scheduler.py`，调度实际位于 [stages.py:238](../../schema_extract/tools/stages.py#L238) 与 [runner.py:1400](../../schema_extract/tools/runner.py#L1400)。

## 3. 看起来像预算、但不是时间的限制

| 规则 | 当前值及作用 | 配置性/依据 |
| --- | --- | --- |
| 每scope调用上限 | 默认32，至少4；normal worker需留独审+未用裁决+未用签收次数，普通reviewer留后两者，裁决留签收 | 新建 `--max-calls`，恢复 `--add-calls`；[runner.py:68](../../schema_extract/tools/runner.py#L68)、[budget.py:31](../../schema_extract/tools/budget.py#L31) |
| 实质退修上限 | 普通流程 reviewer `revise`达到默认5次后切裁决；不是第5次时限。worker `continue`无单独次数上限，仍受总调用和时间门控制 | `--revision-limit`；[stages.py:564](../../schema_extract/tools/stages.py#L564)、[stages.py:578](../../schema_extract/tools/stages.py#L578) |
| 自动修复/执行失败次数 | `retry_limit=2`，按scope累计 `receipt_repairs` 与 `cli_failures` 两个计数分别检查；第三次相应失败停止自动继续。手动恢复不清零，故不是总共只允许两次人工恢复 | 硬编码；[runner.py:75](../../schema_extract/tools/runner.py#L75)、[runner.py:780](../../schema_extract/tools/runner.py#L780) |
| 裁决/签收一次性 | 各一次，准备时登记；追加预算、重开、普通恢复都不重置。时间失败也会消耗这次决定机会 | 无CLI开关；[budget.py:51](../../schema_extract/tools/budget.py#L51)、[runner.py:914](../../schema_extract/tools/runner.py#L914) |
| 维护次数边界 | 一份请求包含1..14个唯一scope任务；review可进入patch再verify，verify若不批准会阻塞；并非一个独立“14次调用”时间池 | [maintenance.py:36](../../schema_extract/tools/maintenance.py#L36)、[maintenance.py:392](../../schema_extract/tools/maintenance.py#L392) |
| 独立review-eval次数 | 每case一次，不自动重试 | [review_eval.py:154](../../schema_extract/tools/review_eval.py#L154) |

`check_failures`、`tool_failures`、`continues`、`p6_starts`等字段还用于状态/统计；不能把看到的计数名都当成独立时间或次数阈值。模型output token上限也不是秒数，见同轮运行时审计。

## 4. 文档与实现边界

README正确列出7200/3600/1800及暂停仍计时，但未展示 A/R 公式，单看“单次调用1800秒”不足以获知首 worker 只有1200秒。[README.md:68](../../schema_extract/README.md#L68)

角色prompt的动态预算提供scope deadline、**本次实际算出的** `call_seconds`、调用与修订计数；没有把operator deadline、A/R预留及120秒事后检查门完整显示出来。普通worker/重试/回执修复说明均要求沿用原预算，没有另藏一组文字规定的分钟数。[prompts.py:280](../../schema_extract/tools/prompts.py#L280)、[normal_worker.md](../../schema_extract/prompt/operations/normal_worker.md)、[receipt_repair.md](../../schema_extract/prompt/operations/receipt_repair.md)

机器检查实际是同步Python函数，`check_stage_snapshot`/`check_complete`本身没有timeout参数或subprocess watchdog；120秒是runner外层决定是否采纳的门槛，不能确保检查卡住120秒后退出。[artifacts.py:677](../../schema_extract/tools/artifacts.py#L677)、[artifacts.py:732](../../schema_extract/tools/artifacts.py#L732)

runner中还有非预算的短等待：孤儿观察约1秒；事件队列等待0.1秒；重试/在途空转等待0.01..0.25秒；退出时对每个线程join最多3秒。它们影响响应粒度或收尾，不给模型另一个几分钟限制，也不延长operator/scope deadline。[runner.py:1387](../../schema_extract/tools/runner.py#L1387)、[runner.py:1488](../../schema_extract/tools/runner.py#L1488)、[runner.py:1546](../../schema_extract/tools/runner.py#L1546)

## 5. 与本轮真实耗时对照

下表只从既有 [TIMING_STATUS.md](TIMING_STATUS.md) 和 [ATTEMPT_REPORT.md](ATTEMPT_REPORT.md) 抽取，不把旧冻结run的限制当作当前默认。

| 真实观察 | 与当前默认的关系 |
| --- | --- |
| FA4/QSA r01各先900秒timeout，追加后又1800秒timeout；共4次 | 证明该档实际调用预算不够；当前默认首worker1200秒虽比900多5分钟，仍不能据此认为足够 |
| QSA r02 首P1调用5400秒timeout；FA4 r04 首P3调用7200秒timeout | 单纯把上限从30分钟改为90/120分钟不能保证一次完成；也需减少机械返工、校准模型/effort |
| FA4 r04 P1 worker31.06分钟，另5.69分钟修证据；reviewer22.41分钟 | 首worker超过当前20分钟；仅该有效独审也超过假设worker用完1200秒后的独审余量 |
| QSA r05 P1 worker34.79分钟，reviewer39.46分钟，阶段合计74.25分钟 | 两个基本角色本身已超过当前stage60分钟；不是主要由大量流程重试造成 |
| FA4 r04 P2 首worker61.11分钟；新独审50.89分钟 | 关键实质调用明显超过30分钟名义值；随后格式修复等另增成本 |
| FA4 r04 P3 首worker120.22分钟timeout后保稿；续写85.89分钟complete | 两次写作账目约206.12分钟，已超过当前operator默认120分钟；不能把后续新run的7200秒调用当产品默认 |
| 当前run结清P1/P2：FA4 59.16/217.00分钟；QSA74.25/178.12分钟 | 累计账目包含重试、修订、独审及检查。FA4 P2其中77.30分钟为孤儿至正式结清保守计时，不能算成已知模型推理时长 |
| 共39个attempt、6次timeout、7个run实例；最终两条仍在P3，被用户取消 | 没有完整成功的P1→P6样本，不能从这轮推导一个有数据保证的全算子合理默认；旧5个run账目约9.88小时也不能加到并行墙钟上 |

记录中可以明确归因的6次timeout是4×r01、1×QSA r02 P1、1×FA4 r04 P3。本审计未发现报告声称本轮曾实际触发120秒检查预算，故该门的返工风险来自代码路径，不能把已有13小时耗时都归到它上面。当前后期配置、模型/effort和旧协议故障混在样本中，不能用某个最长观察值再随意乘系数设新默认。

## 6. 建议只决定几件事

1. **保留一个明确可配置的单调用硬timeout，用于终止卡住的CLI及限制无人值守成本。** 实际预算直接显示，不再由隐藏预留把30分钟变为20分钟。现有数据说明默认值需要重估，但不支持本轮就拍定一个新的统一秒数；正常长耗时也不能简单视为死锁。
2. **取消普通写作前对独审/裁决/签收的时间预扣；operator/stage优先改成统计和到点提醒。** 这样减少三层互相压缩的硬限制，避免“加整算子预算反降首worker额度”、暂停耗光旧scope、返回旧阶段立刻被拒。用户若明确给整轮总时限，再把它作为显式控制约束。调用/退修等已有次数边界可以保留用于发现循环，无须设计新预算系统。
3. **取消“检查已完成且通过，但处理超过120秒就拒收”的规则。** 保留源码身份、产物、schema、独审等真实验收；检查缓慢先记录和提醒。若检查本身确有卡死证据，单独修检查器的中断/恢复，直接从已落盘结果重新检查，不再逼模型为了机器时间门重写或重审。
4. **不把真正执行失败视为用掉一次语义裁决。** 现有一次性计数与超时耦合，时间不够也可能永远失去裁决/签收机会。可先保留“有效决定仅一次”的约束；其具体重试语义独立于时间上限讨论，不需要同时扩张所有计数。

这些都是待用户选择的方向，本轮没有实施。有限单调用、真实终止确认、真实内容验收各有用处；阶段预留和事后时间拒收对本任务的收益尚无实测支持，却已有明确制造重复工作的路径。先减掉这些叠加语义，再用完整成功run的阶段/角色耗时决定默认值，比逐层堆更大的常数更直接。
