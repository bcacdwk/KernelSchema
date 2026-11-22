# paged_indexed_latent_fp8：P3 host 入口证据普通恢复

本次只处理原 run `fd1d34e0-80f0-4a48-a64f-e3952b68d456`。北京时间 2026-11-20 06:12:26，经标准 CLI 执行一次普通 `resume`，不追加调用额度。P3 从 6/32 增至 7/32，原 reviewer 会话、checkpoint、P1/P2 批准和历史账目保留。06:13:27 的核验显示新 reviewer 调用正在运行，**尚未取得 P3 批准**。

## 停点与根因

恢复前为 `P3 / reviewer / normal / invalid_receipt`；失败 attempt `445c090d-590d-4a3d-b832-5040b464c56e`，checkpoint `263d3d4027d14ad39991fad39746a4847e42a051906eaa6b0c836c31caed5f7c`。Runner 的最后诊断为 `receipt_repair_limit_reached: missing bound host entry anchor`，更新时间为北京时间 05:40:13。

冻结绑定的 `entrypoint` 与 `host` 都是 `candidate/solution.py::Model.forward`（完整相对路径见 `diagnosis.json`）。最新未接纳的 approve 共 57 条源码证据：第 0 条覆盖 `solution.py:63–101`，但类别只有 `implementation`；第 1–3 条 `host` 证据分别为下游 `candidate/launch.cu:128–154`、`:177–187`、`:39–63`。源码、路径与坐标均真实，问题是**没有 `category=host` 且路径等于绑定入口文件的证据**。

这符合冻结 `schema_extract/tools/receipts.py:401–406` 的每类别绑定入口要求，以及冻结 `prompt/operations/response.md:7` 的“每类先覆盖对应绑定入口，再补必要调用链”。不是源文件缺失、坐标错误、解析失败或验证器误判，也不能由 Supervisor 代替 reviewer 改旧回执补引用。源码 `solution.py:65–99` 明确建立 `_call=s4_tc_forward`、检查维度/heads、调用扩展并返回 out；下游 launcher 的证据不能代替这一绑定角色。

P3 的调用时间线见 `p3_attempts.json`：

| 次数 | 角色 | 结果 |
| --- | --- | --- |
| 1 | worker | evidence 索引越界，进入补正 |
| 2 | worker | 缺绑定 implementation 入口，进入补正 |
| 3 | worker | complete 被接纳，转 reviewer |
| 4 | reviewer | 有效 revise，形成 1 次实质退修 |
| 5 | worker | complete 被接纳，转 reviewer |
| 6 | reviewer | 缺绑定 host 入口，停机 |

因此原 P3 仅 **6/32** 次调用；`receipt_repairs=3`、`invalid_receipts=3`、`retry_limit=2`，实质退修 1，裁决/签收均 0，无 `call_additions`。自动补正计数按阶段共享，本次 reviewer 首次出现 host 缺项，却因两次 worker 补正已用完额度而停止。P1 已完成的入口补正及 P2 批准均保持，不重提已通过阶段。

## 另行发现的后续门禁

同一旧回执的 `handoff` 为 **2282 字符、1 个非空行**，超过冻结 `receipts.py:461–464` 的 1200 字符上限（行数上限 20）。此检查位于 host 入口检查之后，因而不是 Runner 当前登记的失败原因；如果新提交原样保留它，会产生另一失败。

冻结 `schema_extract/prompt/operations/response.md:17` 只写“handoff 简短保留已定事项、理由、入口、新事实去向和缺口，不复制正文”，没有数值阈值。按 Supervisor 要求，也检索了本阶段首次 reviewer `231aa82a-b538-4382-9080-e8bef879911e` 和最新失败 reviewer `445c090d-590d-4a3d-b832-5040b464c56e` 的**完整动态 prompt**，两者均未出现字符串 `1200`；具体位置与范围记在 `handoff_prompt_observation.json`。这是两份实际提示的观察，不泛化为全框架结论。

普通 `resume` 没有追加反馈参数。本次保留这一后续风险记录，仅由标准 Runner 向新调用传递已经登记的真实 host 诊断；不编辑 state、prompt、规则或旧回执。Supervisor 已知该风险并确认执行一次恢复。若新结果再失败，先分析变化，不自行循环 resume。

## 恢复前条件与执行

失败调用 exit=0，原生唯一成功终结，schema/nonce 有效、无解析错误；`termination.confirmed=true`、`survivors=[]`。正式 inspect 显示 active/unconfirmed/unsettled 全空、旧 Runner 已退出且锁空闲、launch finished，无 pause/cancel、global hold 或维护请求。精确 reviewer session `193dfa9a-f2da-4a01-9caa-9a24a67cd238` 为 ready，未退休。

正式 `audit --attempt 445c090d-590d-4a3d-b832-5040b464c56e --verify` 返回 exit 0、verification=partial、errors=0、unavailable=1；这不产生批准。输出见 `before_audit.txt`。为保存输出而首次本地记录脚本误按 JSON 读取 CLI 文本，随后仅将该文本记录更名并继续，不重复审计、未改运行状态。

对本次相关的 receipts/runner/prompts Python 及 response/receipt_repair 文档逐字节比较，当前与原冻结文件一致。未发现本次必须修复的 Extract 工具故障，未走 `repair-runtime`。执行器仍为 2.5.1 / `b309055c36af2d9450efbfd52515b713d257972c79de3b097e3d66d7cda19c43`。

授权为本会话用户“普通问题按推荐直接处理”的持续指令与 Supervisor 明确委派、确认。正式命令和启动记录在 `resume_launch.json`，stdout/stderr 分别为 `resume.stdout`、`resume.stderr`。CLI 通过原有锁和冻结入口执行；未绕锁、未重开 run、未修改 kernel、未运行 GPU、未操作其他算子或 monitor 时钟。

## 新调用与保存验证

新 attempt `261c47b7-a755-4cb9-b9c7-490727e437ea` 为 `P3 / reviewer / normal`，精确恢复 session `193dfa9a-f2da-4a01-9caa-9a24a67cd238`；原生 system/init 已确认同一 session，随后产生 thinking_tokens 事件。新完整 prompt 保留原失败 attempt 的 host 诊断、当前 nonce 与 7/32 账目。

`recovery_verification.json` 的 11 项本次相关检查全部通过：原 run/source/rules/executor/config 身份、输入锁与 checkpoint 字节、选定实例、P1/P2 批准、accepted_receipts、旧 attempt/usage、旧失败证据均保留；只新增 1 次 P3 调用，补正/退修/裁决/签收计数不重置。旧失败 approve 仍未接纳，P3 approval 仍为空。

正式恢复后 inspect 显示 Runner alive/lock held，无 hold/maintenance；`unsettled=dispatched` 仅指新在途调用，并非未确认终止。当前恢复操作已完成，接下来由原 Runner 正常处理新回执及阶段门禁，不以在运行代替获批。

## 证据索引

- `before_state.json`、`before_inspect.json`、`before_audit.txt`：原停点及结清观察。
- `failed_*.json`、`p3_attempts.json`：失败调用原证据副本与本阶段调用时间线。
- `diagnosis.json`、`bound_host_source_excerpt.txt`、`handoff_prompt_observation.json`：类别/绑定/坐标、次级门禁及限定的提示观察。
- `resume_launch.json`、`resume.stdout`、`resume.stderr`：唯一一次普通恢复。
- `after_state.json`、`after_inspect.json`、`new_attempt.json`、`new_invocation_materials.json`、`recovery_verification.json`：新调用与保存不变量。
