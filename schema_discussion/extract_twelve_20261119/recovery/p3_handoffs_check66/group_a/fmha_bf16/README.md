# FMHA BF16：明显收敛的 P3 handoff 的有限续接

原 run：`e9af332e-dbf4-476b-8283-a88e3cd3cd98`。失败 attempt：`c616e3dd-9da2-499b-8d04-197f5a9cf027`。checkpoint：`778bc962de123aee8ad3164801d415b842ff3071e00702af1cc908e385bae667`。

恢复前实际 P3 **7/32** 次，`receipt_repairs=3`、`invalid_receipts=3`、`revisions=1`，无追加调用、裁决或签收。P1/P2 已批准，P3 approval 为空。停止来自累计回执补正上限，不是 32 次预算用尽。

最新两次 reviewer 均触发真实长度门禁。handoff 从 **2270 字符、1 非空行** 缩为 **1235 字符、4 非空行**，减少 1035 字符，只超出 1200 上限 35 字符。上次准确长度诊断已在最新失败调用 prompt 中送达（第 333 行，绑定 `c4124b46-2549-4f2e-b682-882cd3a037bf`）。模型已作实质压缩，仍未达到要求；这不是新诊断或反馈丢失，更不是解析误判。`handoff_diff.txt` 保存真实差异，没有修改旧回执。

前置 implementation 入口缺项已在 worker 补正后接纳；最新 reviewer 仍以 implementation 引用绑定 `Model.forward` 第 1450–1463 行。一次实质退修后的最新 reviewer 明确记录两项修订已修：三个 role Pipes 用有效工作条件保护，以及 Math 对完全 masked 阶段的 clamp、FTZ 条件作精确区分。当前连续两个失败响应均已到后置 handoff 门禁，相关产物未变；本轮没有重做独立语义审查，也不把未接纳的 approve 当成正式批准。

21 项恢复前置检查通过：全部 14 条历史调用真实退出并结清，原 reviewer session ready，无 active/unconfirmed/unsettled 调用或控制/hold/维护。正式 audit errors=[]，但保留 `partial / receipt_not_accepted; artifact validation not completed`。Supervisor 在获知 2270 → 1235 的新事实后明确同意一次有限普通 resume，保留严格门禁，不循环恢复。

北京时间 **2026-11-20 06:43:29** 的第一条 resume 命令因控制入口竞争返回 `busy / committed=false`。该 CLI 已退出；state 与恢复前副本字节完全相同，calls 仍为 7，新 attempt 为空。证据见 `resume.stderr`、`uncommitted_launch_verification.json` 和 `first_launch_after_*`。这条控制事务没有提交、没有发起模型调用。

Supervisor 明确授权重放同一未提交请求后，先通过正式 inspect 再查状态与屏障，并与 B 组确认其已结束控制提交。**06:44:41** 单独执行标准命令，记录在 `resume_replay1_launch.json`，这次成功启动唯一的新模型调用。

新 attempt 为 **`5b5ce7b1-c13d-46ab-919f-78be956a7eab`**，P3 **8/32**，仍使用精确 reviewer session **`0552eaee-18ef-4618-b172-6b8cb9ef3521`**。native init 确认同一 session，真实最新长度拒绝及失败 attempt 已在新 prompt 中送达。正式 inspect 显示调用 running、Runner 活跃、锁 held。

37 项保存检查全部通过；原 checkpoint、P1/P2 批准、选择/节点/计划、旧 accepted receipts、失败证据、逐调用用量和各补正/退修计数均保留。总计两条 CLI 命令、一次明确未提交请求、**一次已提交恢复和一个新增模型调用**。当前只确认运行恢复成功，验证时新 result 尚未返回，**P3 门禁尚未通过**；后续如再失败，先报告，不自行再次恢复。
