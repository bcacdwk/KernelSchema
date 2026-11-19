# 写回目的地维护登记

核对时间：2026-11-18T15:33:46.473Z。仅通过公开 `inspect`、`status`、`maintain` 操作两个既有 run；任务文件内容与 issue-id 均保持原样。

| 对象 | Request ID | 最终公开观察 | Runner / 调用 |
| --- | --- | --- | --- |
| MoE | `4d304fe6-bfdf-49ab-9b1f-5c861b5c6f8c` | `applied` | 维护 launch `d890b12c-93e5-4b57-8571-afc103548d76` 已 `claimed`，PID 1904715 存活；P1 reviewer `bef32774-2a1a-4fee-84fb-e704dd3174a6` 正在运行 |
| FMHA | `1170387d-57bb-483e-9ffa-1e960c39ab2c` | `draining` | 已交给原 Runner PID 1877032，等待当前 P3 reviewer `c2e23177-068d-473a-a2bc-26504999a072` 收尾 |

两项 issue-id 均为 `inplace-output-input-registration`。MoE 复用已有同载荷请求（`reused=true`）；FMHA 本次新登记（`reused=false`）。`applied` 仅表示维护请求已应用到调度，不表示内容维护已经完成。

执行的公开登记命令：

```bash
python3 schema_extract/extract_yaml.py maintain 74bcac91-808a-41b2-aab4-85143935ab40 --issue-id inplace-output-input-registration --tasks /home/shaohanyong.shy/KernelSchema/schema_discussion/extract_six_full_20261118/MOE_OUTPUT_BUFFER_MAINTENANCE.json --repo /home/shaohanyong.shy/KernelSchema --state-root /home/shaohanyong.shy/KernelSchema/.schema_extract/v2
python3 schema_extract/extract_yaml.py maintain fb8b3758-92d7-4093-aa18-c2e5028f0a43 --issue-id inplace-output-input-registration --tasks /home/shaohanyong.shy/KernelSchema/schema_discussion/extract_six_full_20261118/FMHA_OUTPUT_BUFFER_MAINTENANCE.json --repo /home/shaohanyong.shy/KernelSchema --state-root /home/shaohanyong.shy/KernelSchema/.schema_extract/v2
```

本子代理实际执行顺序与结果：

1. MoE 第一次：exit 1；`{"status":"busy","committed":false,"error":"supervisor is busy; control transaction was not committed"}`。
2. FMHA 第一次：exit 0；请求 `requested`，`reused=false`、`delivered=false`；delivery `active_runner`。
3. MoE 第二次：exit 1；同一次短锁忙错误，`committed=false`。
4. MoE 第三次：exit 0；复用原 request，`requested`、`reused=true`、`delivered=false`；delivery `launching`，mode `maintenance`。
5. 最后一次公开 `inspect --run 74bcac91-808a-41b2-aab4-85143935ab40 fb8b3758-92d7-4093-aa18-c2e5028f0a43 --json` 确认上表状态，随后停止对 run 的操作。

初次读状态误带 `run` 前缀，被只读 CLI 拒绝为 `run ID must be its exact UUID`；随即使用准确 UUID 核实，没有产生控制事务。

MoE 接管前公开状态为 P4 `invalid_receipt`，active/unsettled/unconfirmed 均为空，旧 Runner 已退出、writer lock free。只读核实旧 attempt `010d2757-c179-4dad-854b-bdc0b90c0af4` 的 `exit.json` 与 `cli-exit.json` 均 exit 0，termination confirmed、survivors=[]、descendants_reaped=true。拒收原因仍为 `dispatch.md:126-163` 超出实际 162 行；本次未改回执或重跑 P4。新 P1 维护 reviewer 由公开 maintain 自动启动，原 P4 scope 仍保留 `invalid_receipt`。

MoE attempt count 15→16（P1 calls 2→3），FMHA 仍为 14；两项调用上限仍为 32、additions=[]，执行身份均 `qwen3.8-max[1m]` / `max`。没有新增维护范围、公开 resume、手写 state/回执/产物、热改冻结工具、操作锁或停止其他 run。

