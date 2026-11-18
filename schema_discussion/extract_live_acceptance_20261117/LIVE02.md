# LIVE-02：P1 首次调用的实际预算为 900 秒

结论：这是当前配置触发的阶段内预留，不是 CLI/API 意外退出，也未发现预留作用域错误。保留现有机制；本次不改执行器、产品文字或 schema 字段规则。可以用正式预算追加恢复同一冻结 run。

## 原始证据

| 任务 | run / attempt | 实际结果 |
| --- | --- | --- |
| FA4 Prefill | `8794f043-7c53-4991-85bf-886afd417fed` / `ace72e69-3eb2-4973-aabd-7d66ab1ee2db` | `timeout_seconds=900.0`，`outcome=timeout`，CLI 耗时 900.444 秒 |
| QSA Prefill | `7c31f6e1-e912-43d2-ada3-7b181f8245e5` / `2674c80c-5c59-4e15-a198-24a3069b2a9c` | `timeout_seconds=900.0`，`outcome=timeout`，CLI 耗时 900.430 秒 |

证据位于 `.schema_extract/live_acceptance_20261117/runs/RUN/attempts/ATTEMPT/` 的 `attempt.json`、`result.json`、`cli-exit.json`、`exit.json`、`events.jsonl`。两次均由 Runner 超时后发送 SIGTERM，exit 143，`termination.confirmed=true`、`survivors=[]`。事件持续有思考与阅读，没有 API error；最终 response 为空，workspace 的 `root.yaml` 与 baseline 字节相同。恢复生成的规范化 checkpoint 不代表模型已写出新稿。

## 预算边界

[`runner.create_run`](../../schema_extract/tools/runner.py) 与 [`budget.py`](../../schema_extract/tools/budget.py) 使用：

```text
裁决额度 A = min(call_seconds, stage_seconds / 4, operator_seconds / 12)
审阅预留 R = min(call_seconds, max(0, (min(stage_seconds, operator_seconds) - 2A) / 2))
普通 worker 可用时间 = min(算子 deadline, 本 scope deadline) - 当前时间 - (R + 2A)
实际单次 timeout = min(call_seconds, 当前角色可用时间)
```

本次 `operator=21600, stage=3600, call=1800` 导出 `A=900, R=900`，所以首次 worker 获得 `3600 - 900 - 900 - 900 = 900` 秒。普通 reviewer 只预留裁决与签收；裁决只预留签收；签收不再预留。检查处理也不能动用后续角色预留。

预留作用于本阶段，P5 则是本节点 scope；不会把 P6 的预留重复扣到 P1。每个 scope 的裁决补丁都需要新 reviewer 签收，因此所有阶段保留签收是现有协议要求。单次上限不是承诺额度；只增大 `call_seconds` 无法解除本次阶段限制。

## 同一冻结 run 的恢复

建议分别对两个 run 执行正式 `resume RUN --add-seconds 7200`，沿用原 state-root。现有冻结执行器具备 `budget_extension.v1`；本轮诊断未改变其身份。追加给当前 P1 留出续写、审阅和退修余量，下一次普通调用可获原上限 1800 秒。主代理已按此建议执行，实测结果见下文。

`budget.extend` 追加算子 deadline 和所有 **已经启动、deadline 非空** 的阶段/节点 deadline；过期者从追加时刻起算。它不改 config、用量、已用次数、裁决或签收决定，不重置预算。未启动 P2–P6 的 deadline 仍为空，首次启动仍按 `stage_seconds=3600` 分配，因此它们首次 worker 仍最多 900 秒。后续若再次预算停止，需按实际范围和余量显式追加；运行中的重复 resume 不会追加额度。

追加不能使单次调用超过冻结的 1800 秒。若真实观测证明 30 分钟仍不足，需要另建 run 改配置。以当前 21600 秒算子总预算为例，`stage_seconds=7200, call_seconds=1800` 仅刚好容纳首稿、审阅、裁决、签收各 1800 秒；`stage_seconds=10800` 才额外留出检查和退修时间。不能把较大的阶段上限当作整个六阶段都拥有独立的算子总预算。

## 验证与移交

- 两个冻结快照中的 `budget.py`、`stages.py`、`runner.py` 与当前文件字节相同。
- 对两份真实 state 的内存副本调用追加与恢复路径：追加 7200 秒后当前 P1 为 `running`、调用额度 `(1800, None)`；未来 P2 仍为 `(900, None)`；唯一追加的 scope 为 P1，config、用量、调用次数均未改变。没有写回运行目录。
- 现有 `schema_extract.tools.tests.test_recovery_budget.BudgetTests` 五项通过，覆盖写作/检查保留后续额度、调用次数边界、追加幂等与旧用量、一次决定和独立节点。未新增测试，未运行全量测试。
- 主代理报告两次原始失败调用的 `audit --verify` 均为 verified，`errors=[]`、`unavailable=[]`，终止已确认且无幸存进程。
- 主代理正式追加后，本代理只读复核：FA4 新 attempt `5fedff12-8b1d-4bb0-ae94-05384a590a6f`，QSA 新 attempt `7f12371a-9db5-4cf1-bcc1-acc7cbaef751`；两者均为 P1 worker、`timeout_seconds=1800`、Claude `qwen3.8-max[1m]` / `max`。两份 ledger 各有一次 `origin=resume, seconds=7200, calls=0`，scope 仅包含 P1。它们尚在运行；恢复成功不代表阶段批准或完整验收完成。

2026-11-17：诊断与恢复额度验证完成。本代理仅新增本记录，未修改 `.schema_extract` 原始运行、快照或回执，未调用模型或启动恢复，未提交 git；真实恢复由主代理正式执行。本诊断代理停止写入，后续验收交主代理。
