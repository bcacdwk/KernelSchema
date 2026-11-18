# 前轮试跑修复正确性审计与最小后续方案

2026-11-18；基线 `fa853fafd32a78ac496fbf71e615e34200eeeae3`。独立审阅当前全部产品 diff、对应测试及 LIVE01–15/ATTEMPT_REPORT；结合 [输出默认方案](TOKEN_DEFAULT_PLAN.md) 和 [无 timeout 方案](NO_TIMEOUT_SUPERVISION_PLAN.md)。仅新增本文，未改产品、历史运行、冻结快照或原稿；未调用模型、运行完整提取或启动监督投递。

**结论：未发现前轮修改通过伪造回执、继承旧批准、放宽真实文件/来源/权限或免除独审来制造通过。多数修复应保留；LIVE05 的逐层输出参数扩散应撤掉；LIVE13 有一个已复现的维护 worker 保稿后无法重提缺口，需要补齐。** LIVE10/12 是使用已有正式恢复入口的操作验证，不能计作新产品修复；完整 P3–P6/发布及提醒接管仍未完成真实验收。

## 逐项结论

| 项目 | 独立判断 | 新政策下的确定处理 |
| --- | --- | --- |
| LIVE01 删除失效执行器文档入口 | 真实修复；移除清单中的不存在文件并改链，没有“缺文件就跳过”旁路。`snapshots.py/prompts.py` 与 README 系列改动相符。 | 保留。继续要求实际必需规则存在；不要恢复旧文档或自动跳过缺项。 |
| LIVE02 首稿 900 秒 | 当时是既定时间预留策略的诊断及正式加时，不是算法修复。 | 删除整套时间额度、预留、截止及追加入口；保留次数/一次性决定规则和用量。历史报告不重写。 |
| LIVE03 可选示例导航 | 只确认可选示例未冻结，没有证明必需规则缺失或工具矛盾。 | 无产品补丁可撤回；不借此扩包或放宽读取权限。 |
| LIVE04 输出上限 | 默认 32000 的协议诊断，FA4 有输出上限续写事件；不能推及 QSA。 | 保留诊断事实；默认 128000 的能力边界以 TOKEN_DEFAULT_PLAN 为准。 |
| LIVE05 显式输出参数 | 参数传递、环境隔离与冻结一致性本身成立；但默认仍为 null，未实现默认放大。它是可选调优能力，非原生调用故障修复。 | 改成 Claude backend 一处默认 128000；删 CLI/config/attempt/session/receipt/audit/monitor 的专用参数扩散，不加累计 token 门。保留环境配置指纹及真实 usage。 |
| LIVE05 null receipt/schema | `receipt_schema` 原把新引入的 `max_output_tokens=None` 标成 string，修成 null 是当时必要的配套修复；不是正式 YAML nullable IO 缺陷。当前 exact enum 仍约束值，没有放宽身份。 | 删除输出身份字段后，现有合法生产 attempt 的其余标量身份为字符串/整数，未找到独立 null 需求；建议连这个专用触发分支及预算断言一并精简，不为它保留字段或兼容层。`reopen` 的 object/null 与正式 nullable IO 校验是其他契约，必须保留。 |
| LIVE06 配置登记前置 | 真实修复：只在新登记前复用 `normalize_config`，旧请求复用仍以原请求身份为准。未释放任何真实冻结 run 的输出登记。 | 保留集中校验及其顺序；删时间/输出参数校验后仍要测无效配置不占 output。范围只覆盖配置；不是全部来源/IO 故障的原子回滚，不应据此新增通用清理系统。 |
| LIVE07 worker preview | 真实契约修复：P1–P5 预览通过返回 0；原稿不变、pending corrections/warnings/raw 状态保留。Runner 仍受控规范化、严格重读、权限/来源检查和独审。 | 保留；reviewer 与 P6 原判定不变。不能把预览 0 当批准。 |
| LIVE08 JSON vs code | 真实修复：按 `.json` 类型解析验收材料，不再由 Python/Markdown 中的同名文本推断整个文件是 JSON。坏 JSON、具名输出绑定、relative_to 与阈值/公式矛盾仍拒绝。 | 保留及精确键名提示；代码/说明仍由实际证据和独审核实，不新增程序解释器或通用文本语义解析。 |
| LIVE09 UUID 与启动前结清 | 真实生命周期修复：当前 attempt 适用性、原生成功、终止及 scope/UUID 校验后先保存会话；批准仍走原完整门禁。ExitStack 将 session 预检纳入未启动记录，旧 session/fence 不被覆盖。 | 保留。删除 timeout 形参时不要把预检重新移出结清范围；仍不能从“找不到 PID”推断未启动。 |
| LIVE10 新维护 reviewer | 对旧冻结状态使用 pause→maintain→resume 的操作方案，不是产品修复；创建新独立 scope，旧坏批准未继承。 | 作为历史处置保留，不复制成新恢复分支或特殊 API。 |
| LIVE11 blocked maintenance | 旧冻结版本确实无重提入口；后续 LIVE13 只修当前产品，未迁移旧 run。报告中的首个范围错误归因已由 LIVE13/15纠正。 | 旧现场保留；不补 UUID、不热改快照，不宣传旧 run 已恢复。 |
| LIVE12 孤儿结清 | 复用已存在的正向 gate/后代结清证据；无原生 result 按中断处理，未补造成功。Runner 消失原因未定位，nohup/setsid 只是运行方式调整。 | 保留现有中断结清、原稿及退休会话逻辑；删按年龄/超时处置孤儿，不删真实终止证明。 |
| LIVE13 维护回执重提与定位 | 对候选未变的 normal 维护 reviewer 修复成立；精确数组索引/路径/实际行数诊断有效。对保存新稿的 maintenance worker 未闭环，见下文。 | 保留恢复入口和诊断，修复自身保稿连续性；只移除时间条件。不能把所有 blocked 或一次性决定都改成 pending。 |
| LIVE14 证据跨度评估 | 只读方案分析，无产品改动；样本区分纯长度和真实越界，结论有明确范围。 | 保留历史分析，不再制定替代常数。 |
| LIVE15 删除 12 行门槛 | 实际只删两处任意跨度条件；真实末行、offset、320 字符精确摘录、manifest、来源身份和完整接收门禁还在。 | 保留。宽范围更易满足 P5 区域行覆盖是已披露取舍，不是理解/相关性的证明；不把它误称为零校验损失，也不再加任意局部上限。 |

## 唯一需要补齐的产品问题：维护 worker 保存合法新稿后，resume 把自身新稿判为漂移

**已证实触发**：当前 P2 完整候选进入 normal maintenance `action=patch`，授权只修改 `dispatch.md`。worker 在允许范围内修改一句说明，原生 CLI 成功，但 proposal 缺 `summary`。`runner._recover_call` 将其作为 `receipt_error`，按 continue 规范化并通过 `_draft` 保存新 checkpoint；任务正确停为 `invalid_receipt`。随后正式 resume 的 `_resume_maintenance_receipts` 调用 `maintenance.validate_attempt(status="invalid_receipt")`，该函数仍要求旧 attempt 的 `base_checkpoint == task.draft_checkpoint`，因此抛 `StaleResult: maintenance task, step or draft changed`。

关键路径：`runner.py:247` 的 `_draft` 更新 task 与 `attempt_usage.result_checkpoint`；`runner.py:763–770` 保存 worker 新稿；`runner.py:1057–1094` 的重提入口；`maintenance.py:292–300` 的候选等值判定。现有 `test_maintenance_receipts.py` 的重提成功路径只覆盖只读 reviewer，没有此情况。删时间限制不会修复它。

本轮短离线探针复用现有临时 scalar fixture 和 fake native CLI，只执行一个维护 patch 调用，没有真实模型、全流程或旧 run 修改。实测：

```text
calls=1; native_outcome=receipt_error
run_status=blocked; task_status=invalid_receipt; reason=receipt_repair
settled=true; active/unconfirmed/unsettled 均空
task.draft_checkpoint != attempt.base_checkpoint
task.draft_checkpoint == attempt_usage[attempt_id].result_checkpoint
resume -> StaleResult: maintenance task, step or draft changed
```

**最小修复**：让重提入口识别“同一次已结清调用由 Runner 合法保存的新稿”，不要求它退回旧输入。复用现有账目/checkpoint/repair 证据，不新增迁移、强制恢复或种稿机制：

1. 保持目前最新 task/step attempt、normal role、generation、输入身份、未接受旧回执、空 fence、settled usage、原生成功/真实终止和精确 session 条件。旧 attempt/result 与已结账摘要必须仍相同；原 base checkpoint 的文件仍须匹配 `dependency_digests`。
2. 候选未变沿原路径；只有 worker 的已登记保稿路径，才允许当前 task draft 等于**该 attempt** 的 `attempt_usage.result_checkpoint`。通过 `verify_checkpoint` 核其实际字节、run/input 身份，核 `attempt_id/stage/node` 以及 `parent == 原 base_checkpoint`。对应既有 `repair.json` 必须匹配账目中的摘要，保存的 validation/artifact_digests 与该 checkpoint 一致。所需记录现在都存在，无需新专用字段。
3. 正常接收时的 `maintenance.validate_attempt` 继续严格核当前候选；仅重提资格区分上述两种有证据的连续性，不全局跳过 base 检查、不篡改旧 attempt。新 attempt 从当前合法 draft 出发、沿同一 task/role/step/session；新的合法 complete 后仍必须独立审阅，旧提案不转成通过。

最低补验：本例缺 summary→保稿→同 UUID 重提成功，并确认新稿保留、旧 proposal 未采纳、后续 reviewer 独立；把 task draft 换成其他 attempt 的 checkpoint、改 parent/字节或破坏 repair 摘要时继续拒绝。现有 reviewer、语义 blocked、一次性 adjudication、候选漂移和真实 fence 测试继续保留。

## 删除预算时的测试与字段边界

| 现有位置 | 处理 |
| --- | --- |
| `test_backends_runtime.test_output_budget_is_explicit_fingerprinted_and_bound_to_the_session` | 删除显式 65536/128000 切换、正整数/Codex 参数拒绝和专用 session 字段断言；改为无参数时 Claude env=128000、主机同名值不改变默认、固定 thinking 不被继承、Codex 不获 Claude 设置。保留既有环境指纹稳定性。 |
| `test_cli_recovery.test_output_budget_is_frozen_passed_to_cli_and_rejected_for_codex` | 拆出 LIVE06 的实质行为。无效请求改用仍有效的 `--max-calls 3`、`--revision-limit 0` 或后端/effort 不匹配；同一 output 连续拒绝不产生登记，随后合法请求能创建，真实已登记请求改变 effort 仍拒绝。不能连带删掉配置前置验证。 |
| `test_review_eval`、`test_stage_flow` 的 max_output_tokens 参数/fixture/断言 | 删除专用值传播；保留两 backend、独立会话、每 case 一次、只读材料、实际 scope 与六阶段语义。预算 None 专用身份不再需要。 |
| `test_recovery_budget.test_writing_and_checking_preserve_review_adjudication_and_signoff`；`test_current_returns.test_checker_time_after_normalization_cannot_spend_reserved_review_time` | 纯时间预留/超时拒收预期过时，删除或替换为虚拟大 elapsed 下仍按实际检查结果接收。不得把正常慢检查改成 continue/失败。 |
| `test_cli_recovery.test_resume_adds_finite_credit_after_expiry_without_refunding_calls`；recovery_budget 的混合加时/加 calls 测试 | 删除 expiry、deadline、seconds 扩展；保留必要 calls-only 授权幂等、原计数/用量不重复、一次性决定与 fence 不被恢复抹掉的断言。 |
| `test_backends_runtime.test_cancel_ordinary_child` / `test_timeout_and_unconfirmed_termination_keep_the_session_fence` | 前者名称虽是 cancel，当前实际靠 0.8 秒 timeout 触发；改显式取消事件。后者删超时预期，保留后代退出、retired/fenced、预检拒绝不改旧 session。不要整条删掉生命周期覆盖。 |
| `test_scoped_recovery.test_orphan_timeout_keeps_healthy_sibling_and_original_budget` | 改成很老但仍健康的 orphan 不会被年龄终止；实际中断/明确取消只影响对应 scope，兄弟继续、账目保留。 |
| `test_maintenance_receipts`、LIVE09 stage-flow 新用例 | 去掉 deadline/时间退款断言和 timeout API 参数；保留真正越文件尾、同 UUID、未批准旧回执、原计数、候选/来源/step/role/generation/fence 检查。其 max_calls 反例仍有效，不能当累计 token 限额删掉。 |
| 新生产字段/记录 | 删除 `max_output_tokens` 的 config/attempt/session/receipt/汇总扩散及时间专用 `*_seconds` 配额、deadline、reserved_seconds、timeout_seconds、时间 capability/追加记录；真实 elapsed、started/completed、usage、processing、attempt evidence、result_checkpoint、repair 和终止记录继续需要。详细时间清单以 NO_TIMEOUT_SUPERVISION_PLAN 为准。 |

**容易误删的一行**：本轮 `recovery.verified_result` 除输出字段外还新增了 `configuration_digest` 比较（`recovery.py:109`）。它校验既有 provider/environment 配置身份，与移除输出参数无冲突，应保留；backend 默认仍进入该指纹。不能把该处整个 diff 回退。

**P5 与来源校验不可随预算清理删除**：`_region_anchored` 只保证当前 raw 文件/摘要及区域某行被范围包含；`_validate_derivation_anchors` 仍逐个检查实际 from 链中的当前引用、中间来源和最终定义。实际源解析、当前 provider 批准/字节同一性、消费者用途、stage 权限、P1 具体声明出现在 quote 中、最终 P6 批准与发布门仍有独立作用。`320` 字符摘录、approval handoff 的 `1200` 字符/`20` 非空行和正式 schema 文本规范均不是模型时间/token 配额，本次不动。

## 验证范围与实施交接

本轮另运行 7 个现有短离线用例，**7 tests / 5.565s / OK**：三项 multiline/真实边界与 offset/允许来源及 authentic quote；两项 Python/Markdown 普通证据与坏 JSON；P1 worker 只读预览；session 预检未启动证据。没有重复全套回归，也没有把这些用例称为完整提取验收。上述 worker 缺口用临时 fixture 实证，相关临时目录已清理。

可按两轮四个新执行代理完成：第一轮落地 backend 默认与无时间限制/可取消监督基础；第二轮一位在最终接口上补本文件唯一恢复缺口及相关回归，另一位统一旧字段、提示、文档和测试。`runner.py/maintenance.py/recovery.py/agent.py` 需明确单一文件 owner，避免同时修改接收及结清逻辑。无需新增第五项调查或重做 schema 设计。

本审计没有新增需用户决定的取舍；P5 宽范围的既有代价明确保留，cancel 信号/投递待定语义由 NO_TIMEOUT_SUPERVISION_PLAN 的既有讨论统一决定。其余按本文确定方案实施；不改旧运行，不继承旧批准，不扩张成新的恢复框架。
