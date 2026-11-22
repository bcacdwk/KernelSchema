# GDN P5 首稿版本错误：单范围恢复记录

本次已通过正式入口恢复 `P5:warp_row_reduction` 一次，新增 worker 调用已经真实启动；尚未取得新候选的检查或独立审阅结果。

- Run：`e0311f19-50e5-4760-8331-5805dbb23a09`
- 算子：`gdn_conv_mtp_bf16_state_bf16`
- 原失败 attempt：`82c7d1ab-50fd-43fc-a564-f9a2c5a515fa`
- 恢复请求：`17dd09f1-042b-4158-9593-c8eca059d266`，已由原 Runner 消费。
- 新 attempt：`06b543bd-bc28-45c3-9763-15874fb5c232`
- 正式命令返回时间：北京时间 2026-11-20 07:43:18。

## 根因与恢复准入

这是 worker 的首稿版本错误。原调用为 `P5 / warp_row_reduction / worker / normal`，是该节点唯一一次调用。当前节点 `authored=false`，没有接受稿或批准，唯一候选 checkpoint 为 starter。基线 `META_DATA.Version=1`，worker 原稿和工作目录均为 `2`。冻结 P5 指令明确要求“新建草稿从版本 1 开始”；冻结 Runner 因尚未 authored 而传入 `first_draft=True`，冻结 `_versions` 对实际原稿复核得到同一个拒绝原因。因此 `framework_or_protocol` 分类并不表示工具故障，本次无需修改工具。

失败阶段账目为 `calls=1/32`、`invalid_receipts=1`、`receipt_repairs=0`、`revisions=0`；无次数追加。真实 CLI 返回 0，原生结果成功，schema、nonce、精确 session 及保存响应复核通过，但候选被版本门禁拒绝。账目 `settled=true`，`cli-exit.json` 为 `descendants_reaped=true`，终止确认且无 survivors。通过冻结恢复函数与当前 Linux 进程身份复核，原 gate 及全部已登记后代均已退出，目标 scope 没有活动、未确认或未结清调用。

当前实例仍为已选的非 FK、`vec==1`、`gdn_v1_kernel<true>` 分支，`generation=0`、`plan_revision=1`、`wait_for=[]`。原稿及 starter 均无 Design `from`；恢复核验的实际内容依赖只有已批准 root，SHA 为 `f80d48b83ed0fbb2784187e1d2e631873e49c0dba8df366ad54ed6e4b538de38`，来源 identity 为 `4a3a9cd6b79d8947ae823bd0824a40bcabd5e9732471677d53cc88ec48f7dcf7`。P1–P4 批准有效。

冻结执行器具有 `scoped_recovery.v2`；正式只读 `scoped_restart_evidence` 成功，`settlement_required=false`。无 pause、cancel、hold 或维护屏障。正式恢复入口接受 `proposal_rejected / invalid_receipt / normal` 的此类普通失败，未跳过版本或权限门禁。

## 已执行操作与保持项

唯一控制命令为：

```sh
/opt/venv/python3.13.12/bin/python3 -B schema_extract/extract_yaml.py resume e0311f19-50e5-4760-8331-5805dbb23a09 --scope P5:warp_row_reduction --repo /home/shaohanyong.shy/KernelSchema --state-root /home/shaohanyong.shy/KernelSchema/.schema_extract/v2
```

请求一次成功，无 busy 重放。原 Runner 消费请求后仅新增上述 worker attempt，scope 已用数变为 `2/32`。该调用恢复原精确 session `f8e8ff0f-384c-499c-a6a4-a6e6d5003258`，新 prompt 第 232 行包含原版本拒绝诊断，gate 已 released，进程存活，原生 init 确认相同 session 和 `qwen3.8-max[1m]`。失败原因由 Runner 转入当前反馈，未人为改写回执或候选。

恢复后的定向核对均通过：

- 原兄弟 reviewer `ab5f8af2-b7d9-4b38-aca8-ec9142e4d180` 仍存活；gate PID `1474264` 及 start token/归属不变，调用数仍为 2。
- 输入锁、source/config、冻结执行器、所选实例、generation、写作安排、原有效批准及 accepted receipts 均保持。
- run 与 merged checkpoint 仍为 `d8e823a3799933a918298257cca37022bbdfc832db43aac83214131c2ccdf3e6`；目标 starter checkpoint 仍为 `9045907a4a6e833a6bf7ef93861523cd240d9294efdedbfd8f2695ec13c2a993`。
- 全部原有调用账目、失败原稿、回执、结果和记录字节保持；未退款、追加额度或生成批准。

本次未做无 scope 恢复、暂停或取消兄弟、runtime state/旧证据编辑、工具修复、kernel/GPU 操作、新建 run 或监控 ack/scan/clock 操作。后续仍由原流程检查新候选并独立审阅；发起恢复不代表节点修复已经通过。

## 证据索引

- `preflight.json`：冻结函数的恢复目标、实际依赖、版本复核及真实进程终止观察。
- `native_result_check.json`：原失败调用的真实 exit、原生结果、schema/nonce、响应与 session 复核。
- `failed_attempt/`：原调用关键证据的只读副本，含 starter 和 worker 原稿。
- `state.before.json`、`state.pre_submit.json`、`state.after.json`：定向恢复前后权威状态快照。
- `status.before.json`：正式 `status --full` 读数；`inputs.lock.before.json`、`supervisor.before.json`：输入与控制基线。
- `resume.command.json`、`resume.execution.json`、`resume.stdout.json`、`resume.stderr.txt`：唯一控制命令与真实返回。
- `postflight.json`：请求 consumed、新调用身份/进程、scope 计数及保持项逐项核对。
- `new_attempt/`：新调用登记、启动和诊断材料，`prompt.evidence.json` 固定完整 prompt 摘要与原位置。
- `sibling_before/`：健康兄弟的原登记与进程身份；不以其自然推进为异常。

另有一次只读 `status` 参数放在子命令前而被 argparse 拒绝，未提交控制或模型调用；随后已按正式子命令参数位置成功读取。证据整理脚本曾因把 `accepted_receipts` 的摘要值当对象而提前退出，随后按真实数据类型完成保存，不涉及运行状态写入。
