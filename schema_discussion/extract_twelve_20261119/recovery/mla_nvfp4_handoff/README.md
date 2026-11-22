# mla_nvfp4：P2 handoff 普通恢复

原 run：`af01989c-c712-4fcf-8ecc-3a88f10541eb`。仅此 run 属本次操作范围；未操作其他算子或 monitor 时钟。

## 诊断与恢复决定

恢复前停在 `P2 / reviewer / normal / invalid_receipt`，失败 attempt 为 `07642a2c-7436-45d1-8eaa-55140639453b`，checkpoint 为 `a8ca1b1e26c6f31dcc172163c52c572110c5652a14127acd9ed97aedc11a8ced`。

真实原因是 reviewer 返回的 `approve.handoff` 长达 **3011 个字符、1 个非空行**，超过已有门禁的 1200 字符上限；不是缺字段、原生解析误诊、32 次调用耗尽或来源故障。`receipt.json` 已取证不表示该批准已获接纳；该 attempt 不在 `accepted_receipts` 中，P2 approval 仍为空。

P2 已用调用 **6/32**，实质退修 1 次，`receipt_repairs=3`、`invalid_receipts=3`；冻结 `retry_limit=2`，无 `call_additions`。此前两次补正分别由 worker 的 `root.yaml:163-201` 越界（实际 200 行）和 `dispatch.md:16` 的无效 `quote_offset` 引起，并均在随后调用修正。本次是 reviewer 首次因 handoff 过长遭拒，却因 **worker/reviewer 共用同阶段累计补正计数** 而立即达到自动补正上限。记录这一事实供之后审计，不修改既有预算策略。

最新调用 exit=0，原生结果成功、schema/nonce 正确、无解析错误，精确 reviewer session `d769db00-a363-4a1b-94a1-7f413befc72d` 仍为 ready；终止已确认、无存活后代，active/unconfirmed/unsettled 全空。正式 `inspect` 确认无 hold、控制或维护请求，旧 Runner 已退出且锁空闲。正式 `audit --attempt ... --verify` 返回无 errors；verification=partial 仅因为该未接纳回执尚未完成产物验证，不能宣称阶段批准。

已逐字节比较冻结与当前的 runner/recovery/prompts/receipts/cli/stages 及相关 response/receipt_repair 文档，全部相同。执行器仍为 2.5.1 / `b309055c36af2d9450efbfd52515b713d257972c79de3b097e3d66d7cda19c43`，无需修复工具或 repair-runtime。

规范入口为**原 run 普通 `resume` 一次，不带 `--add-calls`**。依据 `schema_extract/tools/runner.py::_resume_failed_scopes`，normal invalid_receipt 可由显式恢复重新派发，保留 role、mode、session、全部计数、checkpoint 及既有批准；当前拒绝诊断保留在 feedback，经 `prompts.py::render` 注入新调用。下一次 reviewer 将首次收到明确的 `<=1200 characters, <=20 nonempty lines` 诊断，使用新 attempt nonce 自行核实并提交；旧的 3011 字符回执不会被修改或视为有效。新结果仍完整通过原回执、证据、plan、权限和阶段门禁。即使总预算尚有余额，也不无限重复本次恢复。

恢复授权来自本会话持续指令及 Supervisor 本轮明确委派：普通问题按推荐直接处理并记录，允许在原 run、原预算内用正式 CLI 恢复。不创建替代 run、不改 kernel/正式 YAML/语义规则/prompt、不直接编辑 runtime 状态或旧回执。

## 证据

- `before_state.json`：恢复前只读副本。
- `diagnosis.json`：真实计数、全部 attempt 时间线、原生结果及冻结/当前文件摘要。
- `failed_response.json`、`failed_repair.json`、`failed_exit.json`、`failed_cli-exit.json`、`failed_finished.json`：失败调用原证据副本。
- `before_audit.json`：正式审计；`before_inspect.json`：正式观察。
- `resume_launch.json`、`resume.stdout`、`resume.stderr`：唯一一次正式 CLI 恢复命令、进程及日志。
- `after_inspect.json`、`after_state.json`、`recovery_verification.json`：恢复后的正式观察、状态副本和逐项不变量验证。

## 已执行结果

北京时间 2026-11-20 02:40:02 启动一次正式 `resume`，02:40:11 登记新 attempt `5ff0859f-db49-4c58-baef-ecbdc511ddd0`。原 run 已恢复为 **P2 / reviewer / normal / running，7/32 次调用**；`receipt_repairs=3`、`invalid_receipts=3`、实质退修 1 次不变。

已确认新 prompt 原样携带失败 attempt 的真实 handoff 超长诊断，新 nonce 正确，恢复精确 session `d769db00-a363-4a1b-94a1-7f413befc72d`。原生 `system/init` 已报告同一 session；验证时已产生 228 条真实原生事件，正式 inspect 显示 Runner 活跃且锁持有。当前 `unsettled_attempts` 只表示这一正常在途调用，没有未确认终止 fence。

验证中的全部 35 项不变量通过：原输入、来源、规则执行器、配置/预算、checkpoint、原批准、历史调用账目、原失败证据和选择草案均保持不变；P2 调用只增加 1，补正/实质退修/裁决/签收计数均未重置。旧无效 approve 仍未接纳、P2 approval 仍为空。

本次恢复操作完成，不等待整轮结果，也不再发起相同恢复。残余风险是 reviewer 必须自行提交合规 handoff，并通过其余全部原门禁；**尚未宣称 P2 获批或进入 P3**。其他算子和巡检时钟未被控制。
