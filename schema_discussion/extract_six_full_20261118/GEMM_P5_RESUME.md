# GEMM P5 恢复结果

确认时间：2026-11-18T16:56:34.096663+00:00。Run：`06f3ed2c-daa0-46f0-a0dd-a7e1f4f4df31`。已通过公开 whole-run `resume` 续接原 `P5:wide` worker，发现真实在途后停止恢复操作，没有等待模型完成。

- active attempt：`c23a0d66-d4e5-439e-98c9-f1ed95a23a7b`；run `running/P5`，state_revision `123`。
- 原 worker session：`7482beab-e027-40b0-b42f-cd7a6ace1507`，native invocation 明确记录相同 `resume_session_id`，session `running`；scope 仍为 `P5:wide / worker / normal`。
- 模型/effort：`qwen3.8-max[1m] / max`；calls `1 → 2`、attempts `18 → 19`，max_calls `32`，未加额度。
- Runner PID `1946324`、launch `fa92d7f1-089e-4627-a8a0-3e816a43471d`；attempt process PID `1946463`、launch `9b3c35f7-b7ed-4860-b3c4-501f75a216d0`。两个 Linux start token 均与在途进程匹配，Runner 命令指向原冻结执行器 `b8a1d5e368bf0e423081110ffc748945097a503b7eb2fd11fac6355650a58d8a`。
- P1–P4 与 root 的原批准保持；原拒因完整进入新调用 prompt 的 feedback。

## 公开命令及重试

```sh
/opt/venv/python3.13.12/bin/python3 -B /home/shaohanyong.shy/KernelSchema/schema_extract/extract_yaml.py resume 06f3ed2c-daa0-46f0-a0dd-a7e1f4f4df31 --repo /home/shaohanyong.shy/KernelSchema --state-root /home/shaohanyong.shy/KernelSchema/.schema_extract/v2
```

| 次数 | 请求时间 UTC | PID | 结果 | 日志 |
| --- | --- | --- | --- | --- |
| 1 | 2026-11-18T16:53:43.924181+00:00 | 1945429 | busy，未提交 | [log](/home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runtime/supervisor-resume/06f3ed2c-daa0-46f0-a0dd-a7e1f4f4df31-20261118T165343924181Z.log) |
| 2 | 2026-11-18T16:54:08.278547+00:00 | 1945623 | busy，未提交 | [log](/home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runtime/supervisor-resume/06f3ed2c-daa0-46f0-a0dd-a7e1f4f4df31-20261118T165408278547Z.log) |
| 3 | 2026-11-18T16:54:43.259480+00:00 | 1945886 | busy，未提交 | [log](/home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runtime/supervisor-resume/06f3ed2c-daa0-46f0-a0dd-a7e1f4f4df31-20261118T165443259480Z.log) |
| 4 | 2026-11-18T16:55:21.170290+00:00 | 1946324 | running，真实 P5:wide 已启动 | [log](/home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runtime/supervisor-resume/06f3ed2c-daa0-46f0-a0dd-a7e1f4f4df31-20261118T165521170290Z.log) |

每次 busy 后均确认上一进程退出、`committed=false`，run revision `118`、control revision `2`、calls `1`、attempts `18` 不变后才串行重试。第 4 次控制 revision 变为 `3` 并启动原范围调用。全过程记录于 [RECOVERIES.jsonl](/home/shaohanyong.shy/KernelSchema/schema_discussion/extract_six_full_20261118/RECOVERIES.jsonl)。

## 恢复前核对与现稿交接

原拒收 attempt `5dc00c38-8caf-4237-bd19-61f1e177eb77` 是 native success、exit `0`、descendants_reaped、termination confirmed、survivors 空、usage settled；原 Runner PID `1776561` 与记录的 17 个 descendant 均已退出。active、unsettled、unconfirmed 全空。原 worker session 为 ready，只有 wide 失败。

拒因是 `artifact_evidence[10] root.yaml:291-329: invalid evidence range (file has 328 lines)`。root 实际 328 行，原 SHA-256 `4e28944a053c0338038e828db46034fd032a73b6fb918c221ced20d3095a60ba` 与批准一致。

完整的 265 行 [失败 raw 稿](/home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/06f3ed2c-daa0-46f0-a0dd-a7e1f4f4df31/attempts/5dc00c38-8caf-4237-bd19-61f1e177eb77/workspace/wide.yaml) 保留在原 workspace，SHA-256 `9ba4026d93789ffc32ebf1d99bdff5650c0c74a96329dedd47dba05b29f3882a`。新调用的 [baseline](/home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/06f3ed2c-daa0-46f0-a0dd-a7e1f4f4df31/attempts/c23a0d66-d4e5-439e-98c9-f1ed95a23a7b/materials/baseline/wide.yaml) 从登记的 63 行骨架 checkpoint `6b364c8848071458e5a0547709268cca6a0160e397ae1ccdf270df522c75113b` 建立，SHA-256 `5795300161b25f4b9884e16712bdc8c6302a764ca0a9c3b6ce8095a61d270ef3`。没有把未审 raw 自动合入 checkpoint；按补充监督决定，保留完整旧稿、精确 native session 和拒因，由原 worker 在授权 workspace 续交，通过原检查并接受独立 reviewer。

选择 whole-run 是因为冻结 [恢复逻辑](/home/shaohanyong.shy/KernelSchema/.schema_extract/v2/snapshots/39e60da3738fc70943dd784d96dfa8bf4af127c85c971565aa336070a2a41fb6/rules/schema_extract/tools/runner.py:1068) 会把 wide.reason 追加到 feedback；冻结 scoped 路径在 [消费恢复请求](/home/shaohanyong.shy/KernelSchema/.schema_extract/v2/snapshots/39e60da3738fc70943dd784d96dfa8bf4af127c85c971565aa336070a2a41fb6/rules/schema_extract/tools/runner.py:1178) 中直接清 reason。本 run 仅有这一个失败的非 root 节点，因此 whole-run 实际只续接 `P5:wide`。

未修改 frozen、状态、产物或回执，未代改证据、加额度或重开任务，未操作其他算子，未使用私有控制函数或复制 checkpoint，未运行额外真实模型或测试。当前记录只证明续接已启动，不宣称 P5 已通过。
