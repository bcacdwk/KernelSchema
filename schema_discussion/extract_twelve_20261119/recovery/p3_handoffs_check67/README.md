# 巡检 67：MLA / FMHA FP8 的 P3 handoff 有限恢复

本轮对两个原 run **各提交一次普通 resume**，控制按 MLA → FMHA 串行完成。恢复后均已真实启动原 P3 reviewer，分别进入 **6/32** 和 **9/32** 次调用。核验时均为 **P3 / running，P3 approval 为空，新 result 尚未返回**；本报告确认恢复与证据保存，不代表 P3 已通过。

| 算子 | 原 run | handoff 实际变化 | 本轮调用 | 新 attempt |
| --- | --- | --- | --- | --- |
| mla_nvfp4 | `af01989c-c712-4fcf-8ecc-3a88f10541eb` | 3241 → 1418 字符，5 → 6 非空行 | 5 → 6/32 | `deb5079a-db4e-4f79-a089-32041dd05357` |
| fmha_fp8_per_token_head | `1a493588-28a0-4097-849f-85cac55e5b16` | 2357 → 1396 字符，1 → 8 非空行 | 8 → 9/32 | `07b5b76d-49b4-415c-8d2e-8141c7a53212` |

## 恢复依据

MLA 的失败 attempt `d9bca411-63f2-4658-adf4-9659ea81152f` 已收到上次标准恢复送达的长度诊断，handoff 实际缩短 56.2%，尚超 218 字符。与原文的逐行差异保存在 `mla_nvfp4/handoff_diff.txt`；它不是原样重复失败。

FMHA 此前没有 Supervisor 手工恢复，但也不是首次出现长度诊断：第 7 次 P3 调用 `04373ca2-567e-40f0-957f-3bc8a915b625` 提交 2357 字符后被自动补正；第 8 次 `8d47deca-fd99-4b1f-a9c8-4bae12528939` 已收到诊断并缩至 1396 字符，缩短 40.8%，尚超 196 字符。其 P3 第 1 次调用的入口绑定补正及两次实质退修均保留在完整调用链中。

冻结门禁对 P2/P3/P4 的 approve 要求 handoff 非空、字符数不超过 1200、非空行数不超过 20。直接按冻结实现计数，最新 response、result 内 response、receipt 三处 handoff 一致。最新原生结果 success、真实 CLI exit=0，schema/nonce/native_output_matches 有效、parse_error_lines 为空，未发现解析或反馈送达故障。两份 approve 均被严格拒绝，没有进入 accepted_receipts，P3 没有批准。

实际调用余量充足。MLA 恢复前为 5/32、receipt_repairs=4、invalid_receipts=4、revisions=0；FMHA 为 8/32、receipt_repairs=3、invalid_receipts=3、revisions=2。冻结 retry_limit=2，补正次数按本 scope 累计；`call_limit_or_single_decision` 观察分类不表示 32 次额度耗尽。两者均无 call_additions、adjudications=0、signoffs=0。本轮保留上述失败、补正和退修计数。

两者恢复前分别 17/16 条历史调用均已 settled，实际 exit 证据确认终止且 survivors 为空。官方 inspect 显示 active/unconfirmed/unsettled 全空、Runner 不存活、lock free、launch finished；无 pause/cancel/hold/maintenance 或恢复屏障。原 reviewer session ready、未退休。基于实质压缩、正确反馈和充分余量，Supervisor 在本轮明确同意各一次有限普通恢复。

## 实际执行与保存结果

- MLA 于北京时间 **2026-11-20 07:13:27** 提交，**2026-11-20 07:14:29** 完成启动核验；原 reviewer session 为 `40b8605d-1b09-42b5-a17c-4e1f21c273b3`。
- FMHA 在 MLA 新 attempt 已登记后，于 **2026-11-20 07:14:16** 提交，**2026-11-20 07:14:37** 完成核验；原 reviewer session 为 `6e3a0d28-229e-437b-a31b-03f57cdff193`。

两条命令均显式指定 repo/state-root，没有 busy 重放或额外恢复。新 invocation 的 resume_session_id 与原生 system/init 完全匹配原 reviewer；新 prompt 摘要匹配、包含各自最新失败 attempt 和真实 `<=1200 characters, <=20 nonempty lines` 诊断。两者官方 inspect 均确认新 attempt active、Runner alive、lock held，未确认终止列表为空。

各 7 组必要保存检查全部通过：原 run/source/rules/executor/config/输入锁与 32 次上限保留；checkpoint、当前稿件、选择和计划保留；P1/P2 批准和 accepted_receipts 保留；历史 attempt 顺序、逐调用 usage 与旧证据保留；仅各新增一次 P3 调用且所有原失败/补正/退修计数未重置；会话与最新诊断送达正确。所选历史回执/结算证据及当前原稿/工作稿/checkpoint 摘要未变。没有展开全框架审计或重复测试。

- MLA 原 checkpoint：`95bfe47ebdb83d115b6d978228f534e86dad96c565e08c1e155d62f8cfbf7f7b`。
- FMHA 原 checkpoint：`247868b2872f727b81b42f03c44bb53131803a4632e49546b70acfe3202a5158`。

本轮未直接改 state、旧 receipt、批准、冻结 prompt/source/rules；未增额、重开 run、执行 maintain/repair-runtime、修工具或 kernel、运行 GPU。未 ack/scan、未修改 monitor 时钟。若新结果再次失败，保留实际变化给后续正常巡检判断，本轮不循环 resume。

## 后续流程观察与证据

本轮仅记录一个后续可评估点：在初始回执材料中更早明确长度上限，并提供本地计数提示，可能减少 approve 后才发现超长的补正成本；这不是本次工具故障结论，也未据此修改任何冻结材料。

两个算子子目录均有 `diagnosis.json`、`p3_history.json`、`handoff_diff.txt`、`failed_prompt_feedback_excerpt.txt`、`frozen_handoff_gate.txt`，记录真实失败、长度差异和前置条件；`before_state.json`、`before_inputs.lock.json`、`before_inspect.json`、`before_session.json` 与 `failed_*.json` 保存恢复前事实。`resume_launch.json` 记录完整 argv/PID/时间，`resumed_native_init.json`、`resumed_invocation.json`、`resumed_prompt_feedback_excerpt.txt` 证明新调用和诊断；`after_inspect.json`、`after_state.json`、`preserved_files_before.json`、`recovery_verification.json` 记录必要保存结果。`current_status.json` 是收尾时直接读取的真实状态。

`probe.py` 仅在本报告目录写证据、只读运行目录；旧 scan66 和更早恢复报告全部保留。
