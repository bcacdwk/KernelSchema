# MLA NVFP4：P3 首次长度诊断后的普通恢复

原 run：`af01989c-c712-4fcf-8ecc-3a88f10541eb`。失败 attempt：`b9e65704-c504-4318-bb3d-cedf2576a66d`。checkpoint：`95bfe47ebdb83d115b6d978228f534e86dad96c565e08c1e155d62f8cfbf7f7b`。

恢复前实际 P3 **4/32** 次，`receipt_repairs=3`、`invalid_receipts=3`、`revisions=0`，无追加额度、裁决或签收。P1、P2 均已批准；本次是 P3 故障，不是此前的 P2 handoff 停点。

P3 调用链为：worker 因缺 implementation 绑定入口被补正；第二次 worker 完成并被接纳；首次 reviewer 同样缺 implementation 入口；第二次 reviewer 补上入口后，首次触发后置 handoff 长度门禁。最新 reviewer 新增 `source_evidence[15]`，以 implementation 类引用固定 `candidate/solution.py::Model.forward` 第 55–56 行、quote_offset=4；旧 reviewer 只有该文件的 host 引用。对应固定源码与验证器摘录已保存在 `diagnosis.json` 和 `frozen_validator_excerpt.txt`。

最新 handoff 为 **3241 字符、5 非空行**，前次为 3156 字符、5 行；此前先被入口缺项挡住，长度诊断尚未到达。真实字符计数超过 1200，没有解析误判。最新原生响应为 approve，但未被 accepted_receipts 接纳，P3 approval 仍为空；正式审计只有 evidence-only 的 partial 结论。

21 项前置核验通过，包括全部 16 条历史调用真实退出并结清、原 reviewer session ready、无活动控制/hold/维护。Supervisor 原委派允许此新诊断在原预算内一次普通恢复。因此在北京时间 **2026-11-20 06:43:29** 执行标准 `resume`，完整 argv 见 `resume_launch.json`；没有附加指令、维护、修复工具或追加调用。

新 attempt 为 **`d9bca411-63f2-4658-adf4-9659ea81152f`**，P3 **5/32**，仍使用精确 reviewer session **`40b8605d-1b09-42b5-a17c-4e1f21c273b3`**。native init 已确认同一 session，新 prompt 含最新失败的长度诊断及其 attempt id。新调用 running，Runner 活跃且锁 held。

37 项保存检查全部通过，原 checkpoint、P1/P2 批准、逐调用用量和旧失败证据等均保留，repair/revision 计数未重置。当前仅确认**运行恢复成功**；验证时新 result 尚未返回，**P3 门禁尚未通过**。不再次自动 resume，结果交后续正常巡检。

`first_launch_after_*` 是首次启动观察，`after_*` 是最终保存核验所用观察；原始证据和准确时间均在 JSON 内。核心文件为 `diagnosis.json`、`before_history.json`、`before_audit.json`、`resume_launch.json`、`resumed_prompt_diagnostic_excerpt.txt`、`resumed_native_init.json` 和 `recovery_verification.json`。
