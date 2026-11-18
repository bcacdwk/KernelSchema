# 现有 monitor 定时监督

> 本轮已由用户终止；2026-11-18 02:45 UTC 两条当前run已正式取消、巡检已关闭。完整结论与终止现场以 [ATTEMPT_REPORT.md](ATTEMPT_REPORT.md) 为准，不自动恢复。


记录时间：2026-11-18 UTC。目的：将模型持续轮询改为仓库已有的后台定时观察及当前线程唤醒；两条提取任务继续按原配置执行。该切换本身不代表提取完成。

## 范围和约束

- 当前用户线程：`01a1c900-a487-76a1-92ee-990cbff08276`。
- FA4：`ebe745a0-e758-43c4-9679-2e8aa6e2a842`。
- QSA：`dee63d48-09d3-40ec-a694-4ba7f259cdcf`。
- State root：`.schema_extract/live_acceptance_20261117`。
- `auto_restart.enabled=false`、`limit=0`、`used={}`，选择范围仅上述两条 run。
- 未暂停、重启或修改两条提取 run；未改变 schema、模型、effort 或预算；未添加 cron 或新守护系统；未向其他线程、Slack 或邮件投递。
- 后续目标仍为两条 run 最终 `ready` 且 `published`，完成 P6、Template 只读及 audit 核验。

## 现有能力与限制

依据 `schema_extract/README.md` 的“同线程持久提醒”、`schema_extract/tools/wakeup.py` 和 `schema_extract/tools/cli.py`：

- 现有 clock 以独立 Python 后台进程运行，每 30 秒只读观察，15 秒更新心跳；不需要模型持续值守。
- 初始 scan #0 后前三个提醒间隔为 300 秒，第 3 次提醒完成后改为每 1800 秒。间隔从上一轮 committed scan 完成起算。
- CLI 没有自定义 5–10 分钟周期或“仅状态变化/异常时唤醒”的选项。本次使用既有默认，不修改产品来调整周期。
- 定时提醒无论状态是否变化都会入队；主线程应在无变化时完成 ack/scan，避免向用户重复刷正常状态。后台发现异常不会提前触发定时提醒。
- 单轮最多 3 次投递尝试；未确认入队的重试间隔为 120 秒，已入队但未 ack 的补送间隔为 1800 秒。
- `codex queue --resume` 帮助明确说明：resume 失败也可能 exit 0。clock 的 queue 回执只能证明入队 ID，不能证明线程接收或执行。接管必须有本线程实际收到提醒，以及正式 ack、scan 的 committed/reused 证据。
- daemon 或主机退出后，需要显式 `monitor start`；这不是主机级自启服务。

## 启动记录

正确的 CLI 形式是单个 `--run` 后接两个 ID（该选项为 `nargs="+"`）：

```sh
/opt/venv/python3.13.12/bin/python3 -B schema_extract/extract_yaml.py monitor start \
  --thread 01a1c900-a487-76a1-92ee-990cbff08276 \
  --run ebe745a0-e758-43c4-9679-2e8aa6e2a842 dee63d48-09d3-40ec-a694-4ba7f259cdcf \
  --auto-restart-limit 0 \
  --state-root .schema_extract/live_acceptance_20261117
```

初次重复 `--run` 导致只选中最后的 QSA，发现后立即仅 stop/start clock 修正；两条提取 run 未停止或重启，纠正前没有投递提醒。该操作使 generation 为 3；保留的 initial-1 scan 只观察过 QSA，后续后台观察和正式线程 scan 使用修正后的两条 run 范围。

2026-11-18 02:41 UTC 启动核验：

| 项 | 事实 |
| --- | --- |
| enabled / generation | `true` / `3` |
| daemon launch ID | `d2cedf6a-7c42-474a-a465-2f36af72499f` |
| PID | `1415499` |
| process identity | boot `304858cb-b04e-4fee-9b88-16e0fb8eab46`, start ticks `136456930` |
| process | 存活，lock `held`，daemon `running` |
| 锁 | `.schema_extract/live_acceptance_20261117/monitor-clock.lock` |
| 日志 | `.schema_extract/live_acceptance_20261117/monitor-clock.log`（启动后为空，未见错误） |
| 权威记录 | `.schema_extract/live_acceptance_20261117/supervisor.json` 的 clock/auto_restart/observations |
| 第一轮 next_due_at | `1794969967.0143855` = `2026-11-18T02:46:07.014385+00:00` |
| 第一轮 wake ID | `check-3-1-2a507b058c1364aadf1231e0` |
| run_ids | 已核对为上表两条 UUID |

## 同线程接管验证

当前状态：**后台 clock 已启动，第一轮投递和本线程 ack/scan 尚待核验；不得据此宣称已接管。旧监督保留到验证成功。**

收到真实提醒后，由绑定的主线程执行提醒内命令。第一轮对应：

```sh
/opt/venv/python3.13.12/bin/python3 -B schema_extract/extract_yaml.py monitor ack \
  check-3-1-2a507b058c1364aadf1231e0 \
  --thread 01a1c900-a487-76a1-92ee-990cbff08276 \
  --state-root .schema_extract/live_acceptance_20261117

/opt/venv/python3.13.12/bin/python3 -B schema_extract/extract_yaml.py monitor scan \
  --wake check-3-1-2a507b058c1364aadf1231e0 \
  --thread 01a1c900-a487-76a1-92ee-990cbff08276 \
  --state-root .schema_extract/live_acceptance_20261117 --json
```

成功标准：主线程实际接收；ack `accepted=true`（中断续办可能已 acknowledged）；scan `clock.committed=true` 或 `reused=true`；check 观察 2 条 run；持久记录新 `next_due_at`。子代理不替未送达消息造 ack。

正式投递的 queue message ID、ack/scan 及最终接管结果将在实际证据产生后补充。
