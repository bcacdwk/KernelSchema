# B29：完整写作合同下的新真实 design

2026-11-21 UTC。固定 KS `5341470ba5ac03aea857b6480cc7a813c0e3682f`，用公开 `writing.request` / `api.design` 发起一个明确修正、完成 B24 原失败提案的独立 K5 请求。**完整合同已确认进入真实 planner 输入，但 planner 在既定 600 秒窗口超时终止，没有完整提案，因此新 design 为 failed/new_request，Schema=null，未到 precheck 或独立 reviewer。** 本次没有验证出成功 Schema，也没有证明模型能遵守新合同；原生超时不能解释为合同或 Template 已通过。

新增真实模型 **planner1 / reviewer0 / coder0**，GPU jobs **0**；tokens、CLI reported USD、native rounds、physical retries 均 **unknown**。request active **603727 ms**，精确 parent active 增量 **603727515775 ns**。原进程/operation 均已终止，原 parent 无 active/fence，收费 slot 已交回根。首明确失败后停止，没有新 run 重试、延时续额、拼接输出、宿主修稿或宽松 parser。

本项未改生产源码、提交、推送、派生代理、调用 GPU、触碰平台 protected targets、访问禁止的实验仓或安装依赖。原 main 仅写本报告；其余材料在 [/tmp/schema-b29-design-contract-validation-20261121](/tmp/schema-b29-design-contract-validation-20261121)。入口为 [preflight.json](/tmp/schema-b29-design-contract-validation-20261121/preflight.json)、[audit.json](/tmp/schema-b29-design-contract-validation-20261121/audit.json)、[native-launch-audit.json](/tmp/schema-b29-design-contract-validation-20261121/native-launch-audit.json) 和 [selected-export.json](/tmp/schema-b29-design-contract-validation-20261121/selected-export.json)。

## 来源、请求与授权材料

在自己的 `fixed-ks` 中使用 Git archive 的 534 公开快照，对相关 **177 个 Git blob** 核 SHA/长度，B26 五个 frozen files 全部匹配。实际 api/budget/writing/policy/runtime/Task adapter/Template 模块从此快照加载；没有安装或消费 B28 正在修改的活跃 worktree。解释器复用 `.worktrees/.tooling/k1-evaluation/bin/python`，既有 PyYAML 6.0.3、jsonschema 4.26.0、SDK 0.14.37；本次未调用 SDK。源码清单见 [fixed-source-manifest.json](/tmp/schema-b29-design-contract-validation-20261121/fixed-source-manifest.json)。

选择直接公开 `api.design`，用于独立 K5，不声称 standalone Demo 完整闭环。新 request_id 为 `b29-row-softmax-complete-failed-b24-design`。精确保留 **B24 原 Task 与 draft Scope 的 Ref 和递归字节闭包**，同一个 B14 parent，另行冻结新 Rules、Materials、Q 与输出目录；没有编辑旧 Q。

| 对象 | SHA-256 |
| --- | --- |
| 新 Q | `16003fc9bf26f8d23d89f82fa889be1154ac1e7dc3d73e266260f942b18ff6f8` |
| 原 B24 Task | `43f22ed08f6b4b948250af97e303c15471dc03716094e130ba685e45bd193ebd` |
| 原 draft Scope | `6c1fb17db31ef97c9d94c4520453574211124de4e0e29c4db8092a3736884812` |
| 新 Rules | `79532ac1081ff20bf5125476dd6e013b9580d92b7c88745a42507122823e2e86` |
| 新 Materials | `8b4824e2d206a3acb47e0aab9978fefa071f55b91c3e52dfa6cb4cb65ca0eb76` |
| 原 public parent | `85f19bfe92d717c94950442d3fbab16e63bccf52a0f4cb8594a71c6fd02ca3a7` |

新 `workflow_objective` 明确要求模型完成/修正随附失败设计，独立审视原稿所有技术主张，保持原 Task 数学、ABI、domain、数值/副作用硬判据、四形状等权及完整 forward timing 软目标，并声明本请求只写设计、GPU 尚未获得资格。原业务 objective 完整附于其中，未替换成 transport-only 目标。

公开 Materials 保留原 **20 个 planner grants**，额外授予 planner/reviewer 两项：`b24_failed_draft` 是逐字原 retained proposal，SHA `66f919efa8c676cd8f241cf82563c34c41c3d8e020f5c0c1b5c9f3e5596e2892`；`b24_failure_diagnosis` 是原静态诊断。二者在 origin 和 objective 中明确标为 **failed native-success proposal / 静态诊断**，不冒称 accepted Schema、独立语义检查、正确性或测量证据。B24 原失败、原费用、原稿与诊断均原样保留。没有人工改 ID、缩短 raw 或构造修正稿。

目标 environment SHA `d4b9c90c350b4db10664ac9ee66ec118c852a03f5c9ffd955204ea89e31ddb0f` 逐字一致：runtime name 为 `NVIDIA L20D` 加末尾 **8 个空格**，SM10.3、Torch2.9.0+cu130、Triton3.7.1、CUDA13.0。本次保留已提供的目标声明/provenance，不重新探测设备，也不把设计当 GPU 资格。没有更改 acceptance 为 allclose。

## 零调用与实际输入核对

[prepare.py](/tmp/schema-b29-design-contract-validation-20261121/prepare.py) 通过当前公开 writing/Materials 机制建新 Q。预览使用未准入的独立目录，没有第二 owner、register/reserve、operation 或计费。准备前后原 authority 字节完全相同；当时原 9 个 operation 全 terminated、无 active/fence，累计 model4/jobs3/active1039144ms，剩余 model20/jobs13/active9760856ms，roles5/6/15/0。

| 冻结参数 | 值 |
| --- | --- |
| request limits | active1250000ms；model_calls2；planner1/reviewer1/coder0/extract0；jobs0；tokens不限 |
| writing minimum | planner600000ms / reviewer600000ms |
| writing_finish_floor | 原完整 `cost()` 向量全零 |
| 原 parent limits | 24 models / 16 jobs / 10800000ms；roles6/8/16/0；tokens不限 |
| exact Claude profile | SHA `3e1766d7fad58da4e76b77c28a296b0adcb225c04e7e56704dac45a848efc530` |

Profile 原对象取自 B21 公开业务 profile 的 `parameters.claude`，CLI2.1.316、qwen3.8-max[1m]、xhigh、2.00 USD threshold、600000ms、existing_cli_auth/no-tools。官方 inspect/version/help 与 auth status 均通过，只保留 loggedIn/退出码，未读取、复制或导出凭据。原 profile 没有被改成更高美元/时间限额；2.00 是 CLI threshold，不是硬账单上限。

新 Rules 使用 B26 方法 SHA `bc90e3c710b867acc22ee3c9f248ed79e8e19e352aee7483cd51d8ec8bc76349`，保留原 docs/Template 与 standalone fidelity METHOD 的语义正文。严格 parser 与 Template 未改。

完整真实 planner input **463176 bytes**，含 **71 entries：26 正文 / 16 alias / 29 host-only**。实际 SHA 为 `8c532a540121c3e0b6db7fdac5b7ee641692d3b3660b6cb92a5760b363010053`。与零调用预览逐项核对：完整语义正文和固定 assignment 相同，仅写作输入的 publication Ref 路径不同。详见 [actual-planner-input-audit.json](/tmp/schema-b29-design-contract-validation-20261121/actual-planner-input-audit.json)。

实际投递验证了22个 exact bare selection ID/Ref/delivery ID/action ID 映射、原 Template 的80条 BUDGETS、Unicode字符/逻辑行计数规则、完整11字段 content、bindings list、design revise+none/null/空动作字段，以及 optimize 才可提动作的分支。原20 grants完整保留。text policy source SHA仍为 `3bb21e99a3872f94a00c296bbe52b18395c2f9a9fc05c13a00221ccb95ee69c0`。正文/alias未截断，私有 evaluator、目标手写R0、优化kernel、current_source、parent历史/authority均未进入模型输入；失败模型稿只是明确授权参考材料。

## 唯一真实调用与结果

唯一业务命令为：

```sh
/home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k1-evaluation/bin/python -I -B \
  /tmp/schema-b29-design-contract-validation-20261121/run_design.py
```

[runner](/tmp/schema-b29-design-contract-validation-20261121/run_design.py) 核定固定文件与原 authority 后写唯一 execution intent，再以原 `use_authority` 调用公开 `api.design`。没有替换 backend 或串联私有执行步骤。

| 原件事实 | 值 |
| --- | --- |
| run_id | `67394b5473fa4ad086a51b01fea55a2e` |
| operation | `67394b5473fa4ad086a51b01fea55a2e_planner` |
| native session | `d8c65dc0-0e61-45bd-aca1-49220a367e2c` |
| reservation | planner minimum600000ms；effective allowance649634ms；另保留reviewer1/600000ms |
| launch | timeout600000ms、原绝对UTC/monotonic截止、`--max-budget-usd 2.00`、xhigh |
| 权限 init | qwen3.8-max[1m]；dontAsk；tools/MCP/skills为空；两个原built-in plugin身份通过parser |
| 原 exit | `timed_out=true`，returncode=-9，spawn_error=null，active599614ms |
| 原 RoleReceipt | failed；terminated=true；accounting_complete=false |
| 原 errors | missing_terminal / nonzero_exit / timeout_terminated |
| 原 assistant / terminal | 完整assistant messages0；native result terminal0 |
| tokens / CLI reported USD | null / null；非零费用或零费用均不能由本原件推定 |
| native rounds / physical retries | null / null |

原事件文件保留1,398,765 bytes，共7,147行，其中只有 system init/ui事件与 `thinking_tokens` 事件，没有完整assistant或result。**事件数不是 token 用量**；本项未拼接或利用中间生成内容追认一个提案。原 child PID1181923/start_ticks163329453 与 supervisor PID1181922/start_ticks163329411 均核定已终止。

RoleReceipt 为 `artifacts/de7434b359e849a49a50cc2d33c5ef91/claude-receipt.json`，SHA `d8c547f68e7ff2ec004cacc2a34cd5112ea246a0d14cebd8fb430e4568e750c5`。原 profile/input/command/events/exit/process文件 Ref 与 SHA 完整列于 audit.roles[0].raw；receipt original base64 与各原字节一致，原 runtime parser 重解析一致。

K5 真实结果为 `failed / recovery=new_request`，stop_reason=`planner returned no complete valid writing proposal`；Schema/Scope/precheck/review 均 null，structure/permissions/semantics/publication 全 unknown。retained draft SHA `d3db95c5314407b3d9d4ae41c3d5cbff0c68695fa531e0ce604e1a91baf74dcd`，内容为 `ok=false, content=null`；它不是待修正的新 raw 稿。result SHA为 `e75bbf04be4cc57148b02a4b8b77f26a213ed99654c64dc41791dc783d239bfa`。

因此本轮只建立“补全合同在真实请求中完整投递”的证据；**独立K5成功Schema及独立review的验证目标仍未完成**。没有成功设计可供后续implement消费，没有启动implement/optimize/GPU或新增请求，后续策略由根另行决定。

## 结算、恢复与导出

[audit_finished.py](/tmp/schema-b29-design-contract-validation-20261121/audit_finished.py) 核原 native evidence、权限/输入边界、原进程终态及同 parent 原账。B29结算快照为：10个operations全终，累计model5/jobs3/active1642872ms，roles planner2/coder2/reviewer1/extract0；剩余 **19 models / 13 jobs / 9157128ms，roles4/6/15/0**。全账tokens仍unknown，370133仅已知下界；本次也为unknown。原descriptor、旧9个operations和旧requests全保持不变。精确active增量603727515775ns；父累计毫秒与独立request毫秒取整可相差1ms，未手工改账。

原 `api.design(continue_run=同run)` 终态恢复返回完全同一结果，新增model/GPU/operation/active_ns均0；证据见 [recovery-check.json](/tmp/schema-b29-design-contract-validation-20261121/recovery-check.json)。没有create第二owner或改limits。

[design-selected-export](/tmp/schema-b29-design-contract-validation-20261121/design-selected-export) 导出本失败请求的Task/Rules/Materials/Q/result和原RoleReceipt/输入/事件闭包，原auth/private authority不在导出中。该包用于追溯失败，**不是可消费的成功Schema**；descriptor中schema/scope/review均null。没有用宽松检查或CPU结果重新签署成功。
