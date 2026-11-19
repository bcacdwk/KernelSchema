# MLA 维护 verify 的公开恢复路径

只读核对结论：冻结执行器 `b8a1d5e368bf0e423081110ffc748945097a503b7eb2fd11fac6355650a58d8a` 已支持当前维护回执拒绝的恢复。执行以下公开 whole-run 命令，可续跑原 verify reviewer：

```bash
python3 /home/shaohanyong.shy/KernelSchema/schema_extract/extract_yaml.py resume 0d78eac5-1e45-4ac0-9725-ab885b5c52dc --repo /home/shaohanyong.shy/KernelSchema --state-root /home/shaohanyong.shy/KernelSchema/.schema_extract/v2
```

本核对未执行该命令，未启动模型，也未修改 state、回执、checkpoint、产物或源码。

- 当前任务为 `31a684d9-bfba-484d-9692-6d5178e2a79f:P1`，`step=verify`、`role=reviewer`、`mode=normal`、`prepared=true`；review 与 patch 的两条已接受历史保留。
- 原 verify 会话为 `1e62b333-0ec3-4671-9f2c-68146d0ce1a4`。result、session.json 与 state.sessions 一致，状态均支持原会话恢复。
- 冻结 `runner.py:1017–1065` 的 `_resume_maintenance_receipts` 校验已结清失败调用及固定 draft，将 task.reason 追加到 task.feedback，只将 task.status 改为 pending 并清除 reason。它保留 verify、reviewer、已接受历史、调用计数及 draft `051131a1aecc072e02a5c460a53acdbabcf57b0fea6e83d07b2daf0177fdd1ec`。
- 冻结 `maintenance.py:208–220` 以同一维护步骤和角色选取已有 session_id；`prepare_workspace:316–317` 因 prepared=true 保留 draft。`stages.py:163–177` 与 `prompts.py:267` 将维护 feedback 放入下一次 prompt，其中包含 `artifact_evidence[14] root.yaml:146-184: invalid evidence range (file has 183 lines)`。没有自动调整证据边界。

已核对的恢复先决条件：

- 失败 attempt `1a4395b9-e10a-4514-897a-835c5e8257b0` 为本 verify 最新调用，未被接受，usage.settled=true；attempt.json/result.json 的框架 canonical digest 与已结清证据一致。
- result.outcome=completed、exit_code=0、native_status.success=true、session_status=ready；exit 的 termination.confirmed=true、survivors=[]，cli-exit 的 descendants_reaped=true；冻结 `termination_observed` 只读结果为 true。
- draft 与失败 attempt.base_checkpoint 一致，checkpoint.files 与 dependency_digests 一致，实际产物摘要匹配；root.yaml 实际为 183 行。
- active_attempts、unconfirmed_attempts、unsettled_attempts 均为空；runner.lock 与 verify session.lock 均 free；旧 runner 及记录的子进程 PID 均不存在。
- supervisor 中旧 launch=finished、active_hold=null，当前 run 无 control。P1 已用调用 5、上限 32；正常 reviewer 预留 2 次后仍有额度，故无需 `--add-calls`。下一次 verify 计入第 6 次；历史计数不清零、不退款。

启动时公开 CLI 仍会重新校验冻结输入/执行器、checkpoint、调用证据、控制状态与会话身份；若这些状态在核对后变化，应以启动检查为准。工作树新源码修复不会进入本次执行：公开入口经 `_frozen_entry`/`frozen_run_invocation` 切入该 run 的 snapshot，继续执行 b8a1d5e；本恢复分支已存在于该冻结版本。

不要替换成 `resume --scope P1`：冻结 scoped recovery 要求 scope.stage 等于外层 state.stage，而当前外层游标是 P2，因此该路径无法定位此维护 P1。也不需重提 maintain 请求。上述命令是 whole-run 恢复，verify 获批后会继续正常抽取进度，并非执行一次 verify 后自动停止。

上述源码行号均相对于 `.schema_extract/v2/snapshots/ea6a9a93e0a27a072e7fd06a8f18fe89046186aea865171dced4a3afc3ceb484/rules/schema_extract/tools/`。
