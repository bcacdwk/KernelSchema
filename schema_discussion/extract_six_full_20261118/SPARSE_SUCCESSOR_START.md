# Sparse successor 启动记录

> 后续状态：本次新调用已按用户要求取消，进程终止已确认。见 [取消证据](recovery_reassessment/SUCCESSOR_CANCELLED.md) 与 [当前建议](recovery_reassessment/RECOMMENDATION.md)。下面保留当时启动事实，不表示当前仍在运行或允许恢复。

2026-11-19 04:23:34 UTC 核验：新 run **`a362b85a-dce6-4ad4-b38e-3a186dd87c1c` 已进入 `running / P1 / worker`**，首个 active attempt 为 `88f07fb8-23d7-4c16-a368-69399bfffcd2`，unsettled 为 `dispatched`，reason 为 null，尚未发布。

按 [SPARSE_SUCCESSOR_PLAN.md](SPARSE_SUCCESSOR_PLAN.md) 使用现有公开接口完成准备和启动。旧 run `8f0844f9-f279-4781-a19d-0216b7fc4ef5` 保留 `P6 / invalid_receipt`，checkpoint 仍为 `66ad31f314d7ca7e894527d9641bbf3daad59f4ba35278e71e7841dd5e3901da`，三类在途为空且未发布。

## 输入与来源证明

[source.json](sparse_successor/source.json) 指向独立 `sparse_successor/inputs/`，原 entrypoint、host、io、input、reference、acceptance 及两个 shape scope 绑定逐项保持一致，只追加 `prior_work/README.md` scope。来源目录为 `.`，未指定旧 revision。

- 从旧锁对应 snapshot 普通复制 31 个 source 文件，保留路径与原字节，共 983,940 B；逐文件 SHA 与旧锁及新冻结 source 全部一致。
- 从指定 checkpoint 普通复制五份原稿，共 170,644 B；checkpoint 身份摘要及五份 SHA 均核验一致，原稿未改写。
- 另加入历史材料 README、原字节的 REGISTER_CLAIM_TRIAGE 报告及 origin.json，共 39 个输入文件、1,192,837 B，均进入新 source manifest。全部是普通文件，无 symlink。
- [历史材料说明](sparse_successor/inputs/prior_work/README.md) 明确旧稿不具当前批准效力、旧四项选择仅供参考、原任务数学/shape/验收不变；P3 按实际源码核实并收窄 208 的声明，不能将三个主数组的 32 位槽小计写成实际总寄存器数或无 spilling 保证。说明中的材料链接均在本包内有效。
- [origin.json](sparse_successor/inputs/prior_work/origin.json) 记录旧 run、snapshot、source、checkpoint 与逐文件 SHA、新包路径；没有导入 state、session、批准或回执。

新 source 身份由公开 extract 生成：

| 字段 | 值 |
| --- | --- |
| source id | `3edbe063587d41c1a6dff41a77f363efc6fcb32f6a4c5d08c06734623b1cc54a` |
| repository id | `SchemaExtractSource_6d336aa35b52` |
| ref | `6ba57ee61972a19e4cd3f3c75d2dede83118ac9d` |
| ref kind | `private_snapshot_commit` |
| input fingerprint | `e451c3950c283c01e4d92aeb6c7e46b542280d4c0cb338afcdb7777b9a889a52` |
| executor digest | `b6c8be73135ba726c351fda8b351aea7c37a2a3a50d8ca9f96253c9e684be71f` |

新冻结 `schema_extract/tools/artifacts.py` 与当前已修复源码 SHA 均为 `462c34a5b69d1fd4549470f299b6ecdb92bc5eb99092bcdb78820bb425686157`。只读 AST 核验确认 `_validate_scope` 的 P6 `validate_revision_scope` 使用 `new_files` YAML inventory。

## 启动和交接

首次启动前输出不存在且未在 starts 登记。公开 `extract --background` 仅调用一次，返回码 0、`reused=false`、`authorized=true`、`result=start_requested`；随后公开 `status --full` 确认上述 P1 在途状态。

```sh
python3 schema_extract/extract_yaml.py extract \
  --source schema_discussion/extract_six_full_20261118/sparse_successor/source.json \
  --output schema_discussion/extract_six_full_20261118/sparse_successor_delivery \
  --repo /home/shaohanyong.shy/KernelSchema \
  --state-root /home/shaohanyong.shy/KernelSchema/.schema_extract/v2 \
  --backend claude \
  --agent-binary /usr/local/lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe \
  --model 'qwen3.8-max[1m]' --worker-effort max --reviewer-effort max \
  --revision-limit 5 --max-calls 32 --background
```

配置与旧 run 一致：Claude、`qwen3.8-max[1m]`、worker/reviewer effort `max`、revision limit 5、max calls 32、retry limit 2。

完整命令与原始 stdout/stderr 见 [SPARSE_SUCCESSOR_EXTRACT_RESULT.json](SPARSE_SUCCESSOR_EXTRACT_RESULT.json)，启动后身份、manifest 和 SHA 核验摘要见 [SPARSE_SUCCESSOR_SOURCE_VERIFY.json](SPARSE_SUCCESSOR_SOURCE_VERIFY.json)。本次只准备资料包、公开启动一次并核验实际在途；未改原 kernel_wiki、旧 run、冻结文件或原五稿，未改工具/提示/规则，未操作 monitor、其他 run 或 RUNS/PROGRESS，未提交或推送。

已将新 UUID 和 P1 在途证明交主 Supervisor，由其更新同一 monitor 跟踪范围及主台账。后续正式交付须等本 run 当前 P6 和发布门禁通过，再由 Supervisor 使用公开 export 到标准 schema_wiki 目标；本次没有提前复制或导出。
