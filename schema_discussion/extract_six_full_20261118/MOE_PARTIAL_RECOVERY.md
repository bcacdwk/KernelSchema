# MoE P5 部分失败恢复判断

核对时间：2026-11-18 19:55–19:57 UTC。Run：`74bcac91-808a-41b2-aab4-85143935ab40`。

**当前不执行恢复。** 冻结执行器没有能在本案中可靠携带拒因的公开局部恢复路径；保持健康 `mega` 自然推进，待其后续审阅也结束、所有在途和未结算记录清空且 Runner 退出后，再核对并通过公开 whole-run `resume` 接续。没有执行 scoped/whole-run resume、pause、cancel 或 maintain，也没有新增模型调用；因此没有向 `RECOVERIES.jsonl` 追加执行记录。

## 实际状态与失败证据

- state revision `172`，`running/P5`，共 27 次 attempts；P1–P4 和 root 已批准。root SHA-256 为 `b007b14a5f1ed883c276ed2e516b6e42bc3de76412a9e8efcf766cf81e09ba11`。
- `split_k`：`invalid_receipt / worker / normal`，calls `1/32`、invalid_receipts `1`、receipt_repairs `0`、feedback `[]`。唯一拒因见 [rejected.json](/home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/74bcac91-808a-41b2-aab4-85143935ab40/attempts/a4ac10eb-cd4b-4c86-b789-4e1856fa9ded/rejected.json)：`proposal_rejected / missing bound implementation entry anchor`。
- 失败 attempt `a4ac10eb-cd4b-4c86-b789-4e1856fa9ded` 的 native init、terminal、result、state 和 session 元数据一致绑定 `d56a4868-6b0e-41a2-860c-7894d4564531`；session `ready`，scope `P5:split_k / worker / normal`，模型 `qwen3.8-max[1m] / max`。native success，exit `0`，outcome `completed`，usage `settled=true`，termination confirmed、survivors 空、descendants_reaped 为 true；15 个已登记进程现场均已退出。attempt canonical digest `71d1b8c4848ef3e23d9be2dfd7bc0a0067935c464b13ed4f52c00189e97ab254` 与 invocation、result、launch、exit、cli-exit 一致。
- [失败原稿](/home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/74bcac91-808a-41b2-aab4-85143935ab40/attempts/a4ac10eb-cd4b-4c86-b789-4e1856fa9ded/workspace/split_k.yaml) 和 `original/split_k.yaml` 均完整保留：707 行，SHA-256 `0bc632684dd6e0f7bbc3ea3140db165ab9459c24358a70b1aa782c16bce9b7b1`。登记 baseline 为 63 行骨架，SHA-256 `58ec6fbc76e7e0cd4700c629473bcb2faccc09e423ac47438df18456fa90c907`，draft checkpoint `866999fb4e09902aabcca0a3197a05959790693420cafa28c107474f2fe45aab`；未将未审稿合入 checkpoint。

## 仍在运行的兄弟节点

`active_attempts` 仅有 `P5:mega → ee3a59f1-de76-43b2-b2c2-1989b2c11bd5`；unsettled 仅为该 attempt 的 `dispatched`，unconfirmed 空。mega calls `1/32`、reason 空，尚无 result/finished。19:57:02 UTC 现场核验 Runner PID `1958956`、mega gate PID `1999479` 和 CLI PID `1999483` 均存活且 Linux start token 精确匹配；mega events 已增长至 11,005,866 字节，最后写入 `19:56:58.345319 UTC`。Runner 命令指向本 run 原 snapshot，launch `74f3abe3-46ca-418d-8d5d-4a80acbf011b` 仍为 claimed。

19:57:20 UTC 再读，revision 仍为 `172`，上述 active、split_k reason/feedback 和 27 次 attempts 不变。call_additions 空，control 无待消费动作/请求，supervisor 中本 run 没有 scoped retry，active_hold 空。健康进程未被触碰。

## 冻结路径结论与后续条件

本 run executor 为 `b8a1d5e368bf0e423081110ffc748945097a503b7eb2fd11fac6355650a58d8a`，snapshot 为 `adcbd97155004fe0f7ad81f8bc907c0fb57e5c0d4ab26b059d304c727aff1bcc`。

公开 `resume --scope P5:split_k` 会转入原冻结执行器并把请求送给现有 Runner（[cli.py:391](/home/shaohanyong.shy/KernelSchema/.schema_extract/v2/snapshots/adcbd97155004fe0f7ad81f8bc907c0fb57e5c0d4ab26b059d304c727aff1bcc/rules/schema_extract/tools/cli.py:391)、[monitor.py:737](/home/shaohanyong.shy/KernelSchema/.schema_extract/v2/snapshots/adcbd97155004fe0f7ad81f8bc907c0fb57e5c0d4ab26b059d304c727aff1bcc/rules/schema_extract/tools/monitor.py:737)）。但本失败已经 settled，不经过额外中断交接；冻结普通 scoped 消费在 [runner.py:1178](/home/shaohanyong.shy/KernelSchema/.schema_extract/v2/snapshots/adcbd97155004fe0f7ad81f8bc907c0fb57e5c0d4ab26b059d304c727aff1bcc/rules/schema_extract/tools/runner.py:1178) 直接清除 owner.reason，没有先写入 feedback。新 prompt 只读取 owner.feedback（[prompts.py:267](/home/shaohanyong.shy/KernelSchema/.schema_extract/v2/snapshots/adcbd97155004fe0f7ad81f8bc907c0fb57e5c0d4ab26b059d304c727aff1bcc/rules/schema_extract/tools/prompts.py:267)），材料记录循环仅纳入 accepted_receipts；本失败回执不会通过该循环补回拒因。公开 resume 无可附加反馈文本的参数。已完成的 maintenance 不适用于此普通 P5 拒收；不新建 maintenance 任务来替代原恢复。

冻结 whole-run 的 [runner.py:1068](/home/shaohanyong.shy/KernelSchema/.schema_extract/v2/snapshots/adcbd97155004fe0f7ad81f8bc907c0fb57e5c0d4ab26b059d304c727aff1bcc/rules/schema_extract/tools/runner.py:1068) 会把失败 reason 追加到 feedback，并保留 session 和原计数。**以下命令仅为后续可用入口，本次未执行：**

```sh
/opt/venv/python3.13.12/bin/python3 -B /home/shaohanyong.shy/KernelSchema/schema_extract/extract_yaml.py resume 74bcac91-808a-41b2-aab4-85143935ab40 --repo /home/shaohanyong.shy/KernelSchema --state-root /home/shaohanyong.shy/KernelSchema/.schema_extract/v2
```

执行前必须重新确认：mega 已自然完成而非仅 worker 返回后转入 reviewer；active/unsettled/unconfirmed 全空，原 Runner 身份已退出且 writer 可用，upstream/root 批准仍有效，split_k 原拒因、精确 ready session 和完整原稿仍保留，计数未加额，无新 pause/cancel/hold/maintenance。若 mega 也出现失败，重新检查实际恢复范围；若仅 split_k 失败，whole-run 实际只续接它。启动后核实新 prompt 含原拒因、resume_session_id 精确一致、计数仅按真实调用增加，再停止恢复操作。

本次只做只读核验和写此报告；未改源码、frozen、state、feedback、产物或 receipt，未放宽绑定入口证据，未运行测试、提交或 push。当前源码的 scoped feedback 修复未热应用于本批。
