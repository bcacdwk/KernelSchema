LIVE13 — 原维护 scope 的无效回执恢复与精确证据诊断

结论：当前产品已补上手动 `resume` 的维护回执恢复入口。已结清、身份与候选未变、会话已正确登记的 normal 维护调用，可在原 task / role / step / session scope 中重新提交当前回执；原拟 approve 不被接受，计数和预算不退还。8 项定向测试通过。QSA r03 的旧冻结运行没有升级或恢复，仍按 LIVE11 保留现场。

本次还纠正了 LIVE11 / CURRENT_SUPERVISION 对实际拒绝原因的不完整判断。对 QSA r03 `f1a26608-6adf-463a-acbe-144e3a2bc69d`、maintenance reviewer `aac442e7-0dab-46ec-a9ad-776aee3eb859` 的原 proposal 原样重验，首个错误实际为：

```text
artifact_evidence[13] dispatch.md:71-76: invalid evidence range (maximum 12 lines, inclusive; file has 75 lines)
```

逐条只读验证恰有两条无效证据，全部 source_evidence 通过原范围、摘要和 excerpt 校验：

| 原证据 | 范围问题 |
| --- | --- |
| artifact_evidence[13]，dispatch.md:71–76 | 范围共 6 行，但文件只有 75 行，越过文件尾 |
| artifact_evidence[29]，root.yaml:181–194 | 共 14 行，且文件只有 193 行；同时超出 12 行上限及文件尾 |

原 `response.json` 与 `result.json.response` 完全相等。旧错误统一为 `invalid evidence range (maximum 12 lines)`，不能据此断言首个失败仅是长度超限。以上均未修改、截短或替换原证据。

本次具体变动：

1. [runner.py](../../schema_extract/tools/runner.py) 增加一个小的 `_resume_maintenance_receipts` 校验门，由原手动 resume 分支调用，随后继续既有 `_resume_failed_scopes` / `settle`。仅选择 normal / invalid_receipt 且原因为 proposal_rejected 或既有 receipt_repair 的任务。要求三类调用队列全空，失败 attempt 是原 task 当前 step 的最新调用、尚未接受、usage 已 settled；从既有 usage.evidence 重验原 attempt/result 摘要，复用 Store.context、维护身份校验和 checkpoint 校验，要求原生成功、正向进程结清证据及原 scope 已登记的精确 UUID。还有剩余预算才恢复 pending，并把原拒绝原因保留在反馈里。没有新维护请求、新 operation、新 generation、计数退款、预算清零或旧决定采纳。
2. [maintenance.py](../../schema_extract/tools/maintenance.py) 的既有 `validate_attempt` 接受待校验的 task status；默认仍为 pending，恢复门用 invalid_receipt 重验相同 stage/node/role/normal mode/step/draft/generation。没有增加状态、持久化字段或维护控制系统。
3. [receipts.py](../../schema_extract/tools/receipts.py) 在原 anchors 错误上下文中加入数组索引及 `path:line_start-line_end`；[reading.py](../../schema_extract/tools/reading.py) 同时注明含首尾的 12 行上限和实际文件行数。原比较条件、固定摘要检查及普通 ExtractError 分类保持不变，不把范围错误改成自动 receipt_repair，也不替模型选择证据。
4. [response.md](../../schema_extract/prompt/operations/response.md) 只在共用的“提交与证据”说明中明确 `source_evidence` 和 `artifact_evidence` 都按 `line_end - line_start + 1 ≤ 12` 计算；长事实分段并更新引用索引。没有在各阶段重复散布提示或把提示塞回 Python。
5. 新增 [test_maintenance_receipts.py](../../schema_extract/tools/tests/test_maintenance_receipts.py) 和本报告。保留工作区已有 LIVE01/05/06/07/08/09 等未提交修改；本次未改 agent.py、store.py、schema_docs 或固定 schema 字段。

LIVE09 的 UUID 修复已经位于统一 `_receive`：先核验当前 invocation，再登记成功原生会话，随后才校验回执。该入口同样覆盖 maintenance、adjudicate 和 signoff；本次没有发现需再补的登记路径。新增真实 fake CLI 测试分别证明维护和 adjudication 的原生 UUID 在证据拒绝后保留；一次性 adjudication 仍 blocked，正式 resume 不再调用。已有 signoff 中断边界测试也通过。

定向验证采用临时 Store、固定合成来源、真实 Runner / AgentCLI 与本地 fake Claude CLI，不调用真实模型。P1 先通过正常独审，P2 worker 提交完整候选，再安装原维护 review：

| 验证 | 结果 |
| --- | --- |
| artifact 14 行拒绝 → 正式 Runner resume → 缺 summary 的形状错误 → 再次 resume 提交两类合法 12 行证据 | 三次均为同 task、reviewer、review step、checkpoint、只读 workspace 和 session scope；后两次 CLI 精确检查 `--resume` 为首个 UUID；仅第三次的新合法 approve 被接受，正常游标进入 P3/worker |
| source 13 行证据 | 仍拒绝，明确定位 `source_evidence[0] kernel.py:1-13`；artifact 超限明确定位第 29 项 |
| 用量及审批 | P2 calls 为原 worker 的 1 次加 3 次维护调用；invalid_receipts=2、receipt_repairs=1，旧 attempt_usage 与 deadline 不变，budget_extensions 为空，旧拟 approve 无批准效力 |
| 三类 fence、usage 未结算、缺精确登记 UUID、预算耗尽、task step 改变 | 恢复门保持原状态；不派新模型调用、不清 fence |
| draft、role、generation 或固定输入身份改变 | 校验拒绝，任务仍 invalid_receipt |
| 已结束 fake 调用的 cli-exit 正向结清证据缺失 | 即使三类队列为空也不恢复 |
| 新合法 semantic blocked 维护回执 | 被正常接受为 blocked；再次正式 resume 零新增调用、无批准、用量不变 |
| 原生 adjudication 超长证据拒绝 | 原 UUID 登记保留，一次裁决仍已用且 blocked，resume 零新增调用 |
| 既有正常 scope UUID 恢复、跨 scope UUID 拒绝、P2 当前维护候选及 signoff 中断约束 | 均通过 |

最终测试命令：

```sh
/opt/venv/python3.13.12/bin/python3 -B -m unittest \
  schema_extract.tools.tests.test_maintenance_receipts \
  schema_extract.tools.tests.test_stage_flow.StageFlowTests.test_native_session_survives_rejected_source_range_and_resumes_exactly \
  schema_extract.tools.tests.test_stage_flow.StageFlowTests.test_native_completion_cannot_adopt_another_scopes_session \
  schema_extract.tools.tests.test_current_returns.CurrentReturnsTests.test_maintenance_reviews_current_p2_candidate \
  schema_extract.tools.tests.test_current_returns.CurrentReturnsTests.test_p4_signoff_interruption_retires_session_without_new_decision_credit
```

结果：8 tests，58.654s，OK；日志 `/tmp/live13_tests.log`。未运行全套测试、GPU 或真实模型。实际 QSA 原提案检查脚本和结果为 `/tmp/live13_real_probe.py`、`/tmp/live13_real_probe_result.json`；87 个原 state/lock/checkpoint/公开 attempt/session 登记及冻结关键文件的前后 SHA-256 一致，列表为 `/tmp/live13_real_before.json`。原 state SHA-256 为 `8fa5514b32fa056e9448fccdba43b2619e528ff35a982dbe26efa9d2cbfd5057`，原 proposal 为 `add9ed019255e9082c42e6847a1b67bfec25edec8cea63b1e5de038699b14409`。

未决范围：真实 QSA r03 仍使用 LIVE09 之前的冻结执行器，维护会话 UUID 未登记到 state；本次修复不迁移旧状态、不补记旧 UUID，不构成旧 run 的续审授权或成功证明。后续新冻结运行可使用本次恢复行为，但仍须由独立角色提交全新合法回执并完成正常 P1–P6 及交付验收。语义 blocked、CLI 故障、终止未明和一次性决定未扩权。本次未操作任何真实 run、冻结快照、稿件、回执或私有会话，未读取模型隐藏推理，未输出认证，未提交 Git。实现、定向验证和报告完成后停止写入，真实运行继续由主代理控制。
