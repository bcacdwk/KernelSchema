# 巡检 59：MLA 第 7 次 P2 调用后的恢复判断

本目录是上次恢复的后续证据，不覆盖上次记录。原 run 为 `af01989c-c712-4fcf-8ecc-3a88f10541eb`，本次失败调用为 `5ff0859f-db49-4c58-baef-ecbdc511ddd0`。

## 新证据

第 7 次调用结束后，仍为 P2 / reviewer / normal / invalid_receipt。调用 **7/32**，receipt_repairs=4、invalid_receipts=4、revisions=1，未追加预算。active/unconfirmed/unsettled 全空，exit=0、原生 success、schema/nonce 正确，termination confirmed 且 survivors=[]。精确 reviewer session 仍为 ready。

本次 handoff 从 **3011 字符 / 1 行** 实质缩短为 **1224 字符 / 5 行**，减少 1787 字符，只超出既有上限 24 字符。新 prompt 确实包含旧失败的 `<=1200 characters, <=20 nonempty lines` 反馈。模型确已响应补正，但 summary 声称“压缩到限制内”与实际长度不符；原生调用只执行 Read、固定检查 helper 和 StructuredOutput，没有调用字符计数器。根因是模型对自己输出长度的估计失准，不能归为 feedback 丢失、工具误解析或来源问题。

summary 负责简短语义结论，handoff 负责已定事项、理由、入口、新事实去向和缺口；详细依据保留在 conclusions/source_evidence/artifact_evidence 与既有问题记录。Supervisor 不替模型删改旧 handoff，不删除重要审计事实，后续仍由原角色选择适当摘要并重交完整回执。

## 入口比较与推荐

现有 `resume` 只提供 run、scope、add-calls 等参数，没有 `--feedback`、`--reason` 或附件指令入口。因此无法在不改代码或凭据的情况下直接追加“<=800 字符”等说明；不伪造这种能力，也不改冻结 prompt。

`maintain` 的 tasks.reason 虽可携带明确说明，但当前 P2 scope.status=invalid_receipt、approval=null。`maintenance.mark_drained` 对该状态判 `complete_candidate=false`，只读 review 任务会先 waiting，继而 `resume_progress` 回到普通 P2 worker，而非启动新的独立只读 reviewer。它会增加无必要的写作和再审，且普通调用不受维护 fields=[] 写权限限制；不为传一句长度要求启用这个路径。未提交维护请求，也不通过控制时序制造 reviewing 状态。

推荐再允许 **一次原 scope 普通 resume**：这是第 7 次已有明显收敛后的有限继续，原会话可以看到自己刚提交的 1224 字符回执及其新的拒绝，知道上次“已达标”的判断不成立。它不同于重复同一 3011 字符结果；但仍可能再次不合格式，因此本次只启动一次，不承诺成功、不循环刷次数。没有足够证据将模型的计数失准称为工具 bug，也不为此修改 validator、自动补正预算或工具代码。

所有来源、checkpoint、已获批准的 P1、既有证据与用量保持原样。旧两次 approve 均继续未接纳；新回执仍须过所有原门禁，不以摘要压缩替代 P2 语义审阅。

## 证据

- `diagnosis.json`、`handoff_diff.txt`：长度、差异、反馈、原生结果及维护路径判断。
- `before_state.json`、`failed_*.json`：恢复前和最新失败调用原证据副本。
- `before_audit.json`、`before_inspect.json`：正式只读审计和运行观察。
- `resume_launch.json`、`resume.stdout`、`resume.stderr`：本次唯一恢复命令及日志。
- `after_inspect.json`、`after_state.json`、`recovery_verification.json`：真实新调用及 35 项不变量核验。

## 已执行结果

Supervisor 在读到上述推荐后明确同意一次普通 resume，要求若再次失败先汇报而不再自动恢复。北京时间 **2026-11-20 03:10:29** 启动，03:10:37 登记新 attempt **`648ebf51-e1f2-48cc-ac64-23a873b8234f`**。

原 run 已恢复为 **P2 / reviewer / running，8/32 次调用**。仍精确使用 reviewer session `d769db00-a363-4a1b-94a1-7f413befc72d`；新 prompt 包含两次超长 handoff 的真实拒绝，尤其是第 7 次的新拒绝，nonce 和 checkpoint 均正确。原生 init 已报告同一 session；核验时已有 149 条原生事件，result 尚未返回，正式 inspect 显示 Runner 活跃且锁持有。没有因推理时长进行打断。

35 项不变量全部通过：原来源/规则/执行器/配置/预算、checkpoint、原批准、旧调用账目、旧失败证据和选择草案均保持；调用仅由 7 增至 8，receipt_repairs=4、invalid_receipts=4、revisions=1、裁决/签收计数不变。未提交 maintenance，未修改工具或回执，未增加预算。

本次操作到此完成；**P2 approval 仍为空，不能宣称 handoff 已通过或已进入 P3**。新调用正常运行，结果交后续巡检；若再次同类失败，先记录准确长度/差异和证据，不自行再次 resume。
