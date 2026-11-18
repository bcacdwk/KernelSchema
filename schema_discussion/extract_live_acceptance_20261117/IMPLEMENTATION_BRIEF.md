# Extract 实施约定

2026-11-18。用户已同意无外加预算方案及 supervisor 自主处理取消/恢复，随后要求先合并 origin。合并已完成，当前 HEAD 为 `f48823bdea07aaadd2ccff7678589d20701cbf82`；详见 `ORIGIN_MERGE.md`。以下为两轮、四名新的 xhigh 执行代理共用约定。

## 必须遵守

- 以当前工作树为起点；HEAD之后的26个tracked WIP是已保留的真实工具修复，不能整文件回退。已有未跟踪测试也要保留其有效行为。
- **用户远端文字优先。** 当前精修MD/prompt是正文基线，只改新行为必需事实和删除失效规则，不能重写段落来展示“优化”。保留P1局部标签不能推出组合公式、维护fields边界、ready/发布区分、standalone范围及ack后scan语义。每位代理记录必要文字事实变化供最终告知用户。
- 不改schema_docs固定字段或YAML文本规范；不改案例YAML/PNG、Kernel、Schema Wiki、Atrex、主机全局配置或旧`.schema_extract`现场。不改写历史报告。不提交、不推送、不再操作stash。
- 删除自设时间上限而不是改为大常数/None兼容分支。CLI、阶段、算子、裁决/签收预留、检查120秒、补时、执行依赖短timeout都清掉；耗时只统计。轮询/心跳/提醒频率继续有值。Graphviz独立绘图不属此次Extract路径。
- Claude backend默认请求128000；删除显式输出选项和跨层身份字段。保留真实usage、环境configuration_digest和冻结源码身份。Codex可选backend继续可用。
- 使用统一Linux `/proc`身份，真实owned进程/后代与gate结清不可跳过。cancel显式触发KILL并报告pending/confirmed；pause按正常边界。观察错误、日志静默、运行年龄均不触发杀停。
- monitor只观察及提醒，删除额外auto_restart；Runner普通有限重试、人工resume及scope恢复保留。queue一份在途投递，未终结不重发，不能阻塞clock心跳/观察。锁忙与控制未提交须如实返回。
- 保留既有独审/来源/权限/真实证据检查与次数规则。维护worker自身保稿后的回执重提缺口由第二轮runner owner补齐，复用已有checkpoint/repair证据，不造第二套恢复框架。
- 不写历史/patch式注释或兼容旧预算参数、旧身份格式、旧frozen run的迁移。中间接口尚未整合无需伪装兼容；最终应完整可用。

## 分工与交接

每轮同一文件只有一个owner；跨文件接口由双方先约定并通过代理消息交流。需要改对方文件时发具体要求，由owner实施。不要起更多子代理。

| 轮次 | 代理 | 文件所有权 |
| --- | --- | --- |
| 1A | `execute_runtime_xhigh` | `agent.py`、`processes.py`、`backends.py`、`observation.py`、`review_eval.py`；Template `tools/rules/values.py`；相应backend/usage/review-eval测试 |
| 1B | `execute_supervision_xhigh` | `wakeup.py`、`monitor.py`、`common.py`；本轮`store.py`锁/监督记录、`cli.py`monitor/cancel控制；wakeup/monitor测试 |
| 2C | `execute_flow_xhigh` | `runner.py`、`store.py`、`stages.py`、`recovery.py`、删除`budget.py`；阶段/恢复/旧预算测试；必要时接管第一轮基础文件集成修正 |
| 2D | `execute_finalize_xhigh` | `cli.py`、`runtime.py`、`audit.py`、`receipts.py`、`reading.py`、`comparators.py`、`maintenance.py`、`snapshots.py`、`prompts.py`；活跃MD/prompt；对应CLI/来源/回执/提示测试；最终独立复核 |

第二轮D对runner/store/recovery及第一轮基础文件只提意见，第二轮C负责修改；`maintenance.py`的接收/恢复接口由C、D明确约定。第一轮结束后再启动第二轮，第二轮最终验收须在集成完成后进行。

用户追加：上述实现与验收全部结束后，再起一名新的 **high** 代理，专门精简本轮未提交的活跃MD/README/prompt新增或改写句子，统一到远端精修风格，删除重复和非必要指引，保留功能事实。不得与前述代理并行抢改文字；既有历史试跑/审计证据不作为重写对象。

再追加：high文字精简完成后，起一名新的 **xhigh** 代理，整体检查本轮代码、工具与文本并直接落实必要调整，验证文字精简没有损失功能边界。仍不跑提取全流程，不重写用户远端文风或既定schema规则。

## 验证与记录

只做受影响的短定向检查，使用虚拟时钟、fake transport和合成小进程，不跑算子提取全流程或大套回归。优先改已有行为测试，删除纯旧时间公式/输出透传测试，保住进程归属/fence、独审、合法来源和真实恢复反例。

当前轮代理不要自行发起真实Qwen或queue/monitor；最终集成完成后由supervisor决定安排一次128000短JSON请求及一次小范围真实提醒闭环，不填满tokens、不提取算子，完成即关闭临时监督。离线通过不能写成真实闭环通过。

各代理只新增自己的短实施记录：`IMPLEMENT_R1_RUNTIME.md`、`IMPLEMENT_R1_SUPERVISION.md`、`IMPLEMENT_R2_FLOW.md`、`IMPLEMENT_R2_FINAL.md`。记录实际改动、最终接口、验证、未决及必要文字事实变化，不重复粘贴旧调研。

方案依据：`UNLIMITED_EXECUTION_PLAN.md`为综合决定；`SIMPLIFICATION_IMPLEMENTATION_PLAN.md`为分工/删改清单；`LIVE_FIX_REVIEW_PLAN.md`为已复现维护worker缺口及不可误删的真实修复。其他初评只在需要查证细节时读取，不重复做大规模调查。
