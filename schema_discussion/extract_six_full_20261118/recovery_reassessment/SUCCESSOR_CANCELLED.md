# Sparse 新调用已取消，回到旧进度的续接研究

2026-11-19 04:59 UTC 核对。用户明确要求关闭新开的 Sparse 调用，并分别研究旧进度接续与 Supervisor 为何选择重跑。

- 新 run：`a362b85a-dce6-4ad4-b38e-3a186dd87c1c`。公开 `cancel` 复用了原请求 `fd8e232b-256a-41ae-a022-d9e2ef8caf09`，control revision 仍为 `2`，交付已 `confirmed: true`。
- Runner、gate、CLI 和全部已登记子进程均已确认退出；真实 `cli-exit.json` 为 `exit_code: -9`、`descendants_reaped: true`。没有为结算而执行 resume，也没有删除原稿或调用记录。
- 公开 cancel 直接终止进程；run state 中历史 `running / active_attempts / unsettled_attempts` 尚未由 Runner 重新结算。这些历史字段不代表模型仍在运行，不能据此恢复。取消屏障继续有效，已消耗的一次调用记录保留。
- 旧 run `8f0844f9-f279-4781-a19d-0216b7fc4ef5` 仍为 `invalid_receipt / P6 reviewer`，checkpoint 为 `66ad31f314d7ca7e894527d9641bbf3daad59f4ba35278e71e7841dd5e3901da`，三类在途/未确认/未结清集合均为空。当前只研究恢复方案，未改旧状态、批准、稿件或冻结执行器。
- FMHA 原任务继续，观察时为 `P6 worker`；其余四项已交付。现有 monitor 仍会观察已取消 successor，Supervisor 必须尊重取消及当前讨论，不按陈旧 running 字段启动它。

原 `SPARSE_SUCCESSOR_PLAN.md` / `SPARSE_SUCCESSOR_START.md` 保留为历史，不再代表当前执行计划。`RUNS.json` 的六个逻辑任务重新以旧 Sparse 为研究对象，新 run 单列 `cancelled_runs`；这只是监督台账更新，不是运行身份或状态迁移。

机器证据见 [SUCCESSOR_CANCEL_CONFIRMED.json](SUCCESSOR_CANCEL_CONFIRMED.json)。两份独立调查为 [TECHNICAL_RECOVERY.md](TECHNICAL_RECOVERY.md) 与 [SUPERVISOR_DECISION.md](SUPERVISOR_DECISION.md)；调查结论不冒充已实现的续接能力。
