# 混合精度 Sparse：P4 首次 handoff 长度诊断恢复

对 `paged_latent_mxfp8_nvfp4` 原 run `d138394a-20a7-4397-896e-ebac43aa10bb` 完成 **一次普通 resume**。北京时间 **2026-11-20 07:43:42** 发出，**07:44:04** 完成启动核验：P4 reviewer 已真实启动，调用从 **6/32 → 7/32**，新 attempt 为 `1a7c4502-7e41-447d-ab69-2b7ee2147525`。核验时为 **P4 / running，P4 approval 为空，新 result 尚未返回**；本报告确认恢复与证据保存，不代表 P4 通过。

## 诊断与恢复依据

最新失败 attempt `dbca9743-0c86-4698-9dcb-01916ac01e02` 的 approve handoff 为 **1368 字符、3 非空行**，超出 168 字符；response、result 内 response、receipt 三处一致。冻结门禁对 P2/P3/P4 的 approve 要求非空、字符数不超过 1200、非空行数不超过 20，拒绝有真实依据。该回执未进入 accepted_receipts，也未产生 P4 批准。

这是本 P4 scope 的**首次 handoff 长度诊断**。6 次既有 P4 调用的动态 prompt 均未含长度诊断，前两次回执补正均为 `conclusion/issue evidence index is outside source_evidence`。累计 receipt_repairs=3、invalid_receipts=3，冻结 retry_limit=2，因而首次长度失败直接停止。实际调用只用 6/32，revisions=1、adjudications=0、signoffs=0，无 call_additions；观察分类 `call_limit_or_single_decision` 不表示调用额度耗尽。

前一个 reviewer 回执为 `4bcb12e2-b1cf-4fa3-a71f-804db0152a2a`，decision=revise，handoff 为 1299 字符、5 非空行。其后 worker 修正目录中验收材料清点，最新 reviewer 改为 approve，handoff 也相应变化。逐行差异已保存；本次恢复依据是新诊断和充足余量，不把这段变化描述为重复长度失败后的压缩收敛。

最新原生结果 success、真实 CLI exit=0，schema/nonce/native_output_matches 有效，parse_error_lines 为空；原 reviewer session `d417b8ae-042c-46fa-aff3-e60fb0f1a201` ready 且无退休记录。19 个历史调用全部 settled，实际 exit 与 CLI exit 一致，termination confirmed、survivors 为空、descendants_reaped=true。官方 inspect 确认 active/unconfirmed/unsettled 全空、Runner 不存活、lock free、launch finished；无 pause/cancel/hold/maintenance 或恢复屏障。提交前再次核对无屏障且原 state 未变。

Supervisor 转达用户对本轮十二算子自动提取的授权：普通问题按推荐处理并记录；在新 handoff 诊断、余量充足和全部结清无屏障时允许一次普通恢复。此次据此续接原 reviewer，由正常动态材料传递真实诊断。

## 恢复核验

命令显式指定 `/opt/venv/python3.13.12/bin/python3 -B`、repo 与 state-root，无额外参数。GDN 的另一项控制提交完成后才发出本命令，无 busy 或重放。

- invocation.resume_session_id 与原生 system/init 均精确匹配原 P4 reviewer session；阶段、角色保持 P4/reviewer。
- 新 prompt 包含最新失败 attempt 和完整 `approval needs a short handoff (<=1200 characters, <=20 nonempty lines)` 诊断，prompt 摘要匹配 invocation。
- 原 run/source/rules/executor/config/输入锁与 32 次上限保留；checkpoint、稿件、选择、写作计划草案保留。
- P1/P2/P3 批准与 accepted_receipts 保留，P4 仍未批准；旧 attempt 顺序、逐调用 usage 和原失败/补正/退修计数未重置。仅新增一次 P4 调用。
- 298 个必要历史证据、当前原稿/工作稿、checkpoint 文件及冻结门禁摘要全部未变。官方 inspect 确认新 attempt active、Runner alive、lock held，unconfirmed 为空。

7 组必要保存检查全部通过。原 checkpoint 为 `2485ba254ac242ce90a9714c9c1a8cdaf15faea77fb15d02d2ea67679031caa0`。

未直接改 state、旧 receipt、批准、prompt/source/rules；未放宽校验、增加额度、重开 run、执行 maintain/repair-runtime、修工具或 kernel、运行 GPU。未 ack/scan 或修改 monitor clock。源包既有性能用户例外、机制级 A/B 原始回执及逐输出验证回执缺口保持原事实，本次格式诊断不构成源码需重新提交的结论。

若新结果再次失败，保留真实变化交由后续正常巡检判断；本次不循环 resume，也不等待整个 P4 完成。

## 证据

`diagnosis.json`、`p4_history.json`、`historical_settlements.json`、`handoff_diff.txt` 和 `frozen_handoff_gate.txt` 记录长度、调用链、实际结清及门禁。`before_state.json`、`before_inputs.lock.json`、`before_inspect.json`、`before_session.json` 和 `failed_*.json` 保存恢复前事实；`prelaunch_inspect.json` 保存提交前复核。

`resume_launch.json` 记录真实 argv/PID/时间；`resumed_native_init.json`、`resumed_invocation.json`、`resumed_prompt_feedback_excerpt.txt` 证明原会话续接与诊断送达。`preserved_files_before.json`、`after_inspect.json`、`after_state.json`、`recovery_verification.json` 保存核验依据；`current_status.json` 为收尾时直接读取的状态。`probe.py` 只在本报告目录写证据，唯一运行控制通过普通 resume CLI 执行。
