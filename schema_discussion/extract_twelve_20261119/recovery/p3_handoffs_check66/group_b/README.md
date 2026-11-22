# scan66：两个入口已修正后的 P3 handoff 补正

北京时间 2026-11-20 **06:43:53**，对下面两个原 run 各执行了一次普通 `resume`。06:45:08 / 06:45:13 的核验显示，两者均实际进入原 reviewer 的第 **8/32** 次 P3 调用，精确会话由原生 `system/init` 确认。新 prompt 已携带此次真实 handoff 长度诊断及对应失败 attempt。

**这次完成的是受控恢复与保存验证；观察时两者均在运行，尚未获得 P3 批准。** 若新结果再次失败，先分析变化，不连续循环 resume。

| 算子 | 原 run | 最新失败回执的 handoff | 新调用 |
| --- | --- | --- | --- |
| fp8_mxfp4_grouped_gemm_masked | `c8dd5678-2935-4efa-943a-02e848d03611` | 2815 字符，6 非空行 | `364e4641-2fb2-43aa-bcc7-ff7e584b2ed4` |
| paged_indexed_latent_fp8 | `fd1d34e0-80f0-4a48-a64f-e3952b68d456` | 2326 字符，1 非空行 | `f478aab1-86cd-4716-a04a-b3dca5be7e90` |

## 新诊断与入口变化

masked GEMM 最新失败 attempt 为 `7a388176-110f-48b7-8275-a0796adc2e83`。相对上轮失败 `73af655b-77bb-4b4c-9bdd-af53d68ef4e3`，source_evidence 从 73 项增至 74 项；新增 `[0] category=implementation`，指向绑定的 `candidate/solution.py::Model.forward` 第 116–135 行，原有 host 入口证据保留。绑定文件摘要、范围和引文均有效。因此原 implementation 类别缺项确实已补上，当前 handoff 门禁不是重复入口诊断。handoff 从 2684 字符、6 行变为 2815 字符、6 行。

indexed FP8 最新失败 attempt 为 `261c47b7-a755-4cb9-b9c7-490727e437ea`。相对上轮失败 `445c090d-590d-4a3d-b832-5040b464c56e`，source_evidence 从 57 项增至 58 项；新增 `[1] category=host`，指向绑定 `candidate/solution.py` 第 83–99 行的入口守卫、参数传递及返回逻辑，原 implementation 入口证据保留。绑定文件摘要、范围和引文均有效。handoff 从 2282 字符、1 行变为 2326 字符、1 行。

两者冻结 `schema_extract/tools/receipts.py:401–406` 先检查各类别的绑定路径；后面的第 461–464 行要求 P3 approve 的 handoff 非空、至多 1200 字符、至多 20 个非空行。两份新回执均只在字符数上违反此项，Runner 最新真实 reason 均为 `approval needs a short handoff (<=1200 characters, <=20 nonempty lines)`。相关冻结文件的摘要与 inputs.lock 一致；未发现本次需要工具补丁的依据，也没有放宽门禁。

两份第 7 次调用的完整 prompt 均没有 `1200` 或此次长度诊断；第 8 次调用的新 prompt 均首次包含该数值门禁、实际拒绝摘要和第 7 次失败 attempt ID。此观察仅限这些实际提示，没有开展额外全框架审计。普通 resume 没有自由反馈参数，本次完全由标准 Runner 携带已登记诊断，未人工改提示或旧回执。

## 账目与恢复条件

恢复前两者均为 `P3 / root / reviewer / normal / invalid_receipt`，真实 P3 调用数 **7/32**，`receipt_repairs=4`、`invalid_receipts=4`、实质退修 1 次、裁决/签收均 0，冻结 `retry_limit=2`，没有 `call_additions`。P3 自动补正按阶段累计；本次长度拒绝是各自调用链中首次出现，停止来自补正累计条件，并非 32 次调用预算用完。

两份失败回执都提出 approve，但从未进入 accepted_receipts，P3 approval 为空。原生结果 completed / success、真实 CLI exit=0，schema/nonce 有效、无解析错误；终止 confirmed、survivors 为空。masked 的 15 条、indexed 的 17 条历史调用账目均已 settled。正式 inspect 显示 active/unconfirmed/unsettled 全空，旧 Runner 退出、锁 free、launch finished，无 pause/cancel/hold/maintenance 屏障。原 reviewer session 均 ready、没有退休：

- masked GEMM：`bca9e483-b406-44e1-8dae-114a217c5cbd`。
- indexed FP8：`193dfa9a-f2da-4a01-9caa-9a24a67cd238`。

授权依据是本会话用户“普通问题按推荐直接处理并记录”的持续指令，以及 Supervisor 对此次新 handoff 诊断明确授权每个原 run 各一次普通恢复。两个标准 CLI 命令均显式指定 `/home/shaohanyong.shy/KernelSchema` 和 `.schema_extract/v2`，已成功登记新调用；没有重发成功命令。

## 保存验证

每个 run 的 14 项针对性验证均通过：原 run/source/rules/executor/config 身份与输入锁保留，checkpoint、选择、计划及草稿不变，P1/P2 批准及 accepted_receipts 不变，所有历史 attempt 与 usage 保留。masked 的 340 个、indexed 的 394 个已记录旧证据/原稿/工作稿/checkpoint 文件摘要均不变。两个旧 checkpoint 分别继续为：

- masked：`c00e34f2e7cbd89d1e3e0a24002812d49a15c52c36b80ef56df952a4d5d5a384`。
- indexed：`263d3d4027d14ad39991fad39746a4847e42a051906eaa6b0c836c31caed5f7c`。

两者仅各新增 1 次 P3 调用，补正/退修/裁决/签收计数未重置，预算仍为 32。新 attempt、invocation.resume_session_id 和原生 init 一致指向各自原 reviewer 会话。观察时已有 424 / 675 条原生事件，Runner alive/lock held、对应新 attempt active，unconfirmed 为空；未产生新 result 或 P3 approval。

本次未新建 run、未增加额度、未直接修改运行 state/旧 receipt/批准/冻结 source/rules/prompt，未调用 repair-runtime 或 maintain，未修改 kernel、未进行 GPU 测试。没有控制另两个 A 组 run，也没有 ack/scan 或修改 monitor clock。

## 证据

`masked_gemm/` 与 `indexed_fp8/` 分别保存：

- `before_state.json`、`before_inputs.lock.json`、`before_inspect.json`、`before_session.json`：恢复前原状态与结清观察。
- `diagnosis.json`、`p3_attempts.json`、`failed_*.json`、`frozen_gate.json`：前后入口/长度变化、真实调用链、失败原证据及冻结门禁。
- `resume_launch.json`、`resume.stdout`、`resume.stderr`：唯一一次普通恢复的完整 argv、时间与日志。
- `after_state.json`、`after_inspect.json`、`resumed_attempt.json`、`resumed_invocation.json`、`resumed_native_init.json`：新调用与原会话。
- `preserved_files_before.json`、`recovery_verification.json`、`prompt_transition.json`、`resumed_prompt_diagnostic_excerpt.txt`：保存验证及新真实诊断的传递。

`resumed_runs.json` 汇总两项新调用。旧的 `masked_gemm_p3_anchor/` 和 `indexed_sparse_p3_anchor/` 报告全部保留。
