# B47：原生查询 EXIT147 的有限诊断

2026-11-21 UTC。**已复现 SessionControl 暂停/恢复与 Bash 作业控制的兼容问题；会改变 shell 控制流，但没有证据证明它独立阻断 B41 的标准 Schema 实现。简单逆转 SIGCONT 顺序未通过验证，不应直接应用。**

## 证据与边界

复用 B41 的 [公开工具事件摘要](/tmp/schema-b41-standard-native-20261121/owned-tools-summary.json)，只保留本题必要字段到 [b41-public-evidence.json](/tmp/schema-b47-native-tool-exit-20261121/b41-public-evidence.json)：`env`、`kernel-records`、`kernel-pareto-frontier` 均有正常数据、`EXIT=147` 和 `Stopped`；前两者另见 `Done`。这些 Bash 工具事件的 `is_error=false`，随后 `record-read` 为 `EXIT=0`。`147=128+SIGSTOP` 是 Bash 观察到的命令状态，不能据此认定 Python 最终以 147 终止，亦不能把记录内的测量 `exit_code` 当作本地 CLI 的退出码。

固定源为 AKA `88d773b3acdd60ed13334dc46404e2d0caefe49f`。相关位置：

- [execution/control.py:55](/tmp/schema-b47-native-tool-exit-20261121/aka-source/src/aka/agent/execution/control.py:55)：`AgentProcessScope.signal` 对所有信号都先 root、后其余 owned PID；`request` 发两轮 STOP、处理请求、再 CONT。
- [tools/host.py:128](/tmp/schema-b47-native-tool-exit-20261121/aka-source/src/aka/agent/tools/host.py:128)：先退出暂停区，再发送客户端回复。
- [tools/sandbox.py:163](/tmp/schema-b47-native-tool-exit-20261121/aka-source/src/aka/resources/tools/sandbox.py:163)、[light/measurements.py:442](/tmp/schema-b47-native-tool-exit-20261121/aka-source/src/aka/workflow/light/measurements.py:442)：普通查询打印返回数据并采用 host 的 `exit_code`；上述两种记录查询成功时返回 0，没有主动构造 147 的查询分支。

## 最小 CPU 复现

使用固定原 `SessionControl`、原 sandbox `main()` 和原本地通信协议；fake coordinator 只在新临时 Unix socket 返回一份成功的 frontier JSON，并复用“恢复后发回复”的顺序。未接真实 SDK/server。`process_snapshot` 在诊断进程内替换为显式记录的自有 Bash/client 两个 PID，避免原函数枚举整机 `/proc`；产品文件未修改。Bash 为 `4.4.20(1)-release`。

| 条件 | Bash 命令状态 | 原 CLI `main()` | 数据与后续动作 |
| --- | ---: | ---: | --- |
| monitor on，无注入窗口 | 3 次中 1 次 147 | 均 0 | 147 那次有正常 JSON、Stopped/Done |
| monitor on，root CONT 后留 20ms 窗口 | 147 | 0 | 正常 JSON、Stopped/Done |
| monitor off，相同 20ms 窗口 | 0 | 0 | 正常 JSON |
| monitor on，20ms 窗口，`CLI && FOLLOWUP` | 147 | 0 | JSON 成功，FOLLOWUP 被跳过 |
| monitor on，停用 suspension，`CLI && FOLLOWUP` | 0 | 0 | FOLLOWUP 执行 |

自然复现见 [cpu-results-initial-wait-probe.json](/tmp/schema-b47-native-tool-exit-20261121/cpu-results-initial-wait-probe.json) 的 `monitor_on_2`；固定窗口对照见 [cpu-results-before-fix.json](/tmp/schema-b47-native-tool-exit-20261121/cpu-results-before-fix.json)。前者后来进行的事后 `wait PID` 返回 127，因为 Bash 已回收前台作业；这不是原 CLI 的终止状态。可靠边界为原 `main()` 返回 0、正常输出，以及 shell 已观测的 147/作业通知；没有取得独立的 Python `waitpid` 终止回执。所有记录的自有 PID 均已消失。

可重跑 [reproduce.py](/tmp/schema-b47-native-tool-exit-20261121/reproduce.py)；该脚本默认运行表中四个固定窗口/控制组，不运行模型或 GPU：

```bash
PYTHONPATH=/tmp/schema-b47-native-tool-exit-20261121/aka-source/src PYTHONDONTWRITEBYTECODE=1 /tmp/schema-b18-model-budget-usage-20261121/venv/bin/python /tmp/schema-b47-native-tool-exit-20261121/reproduce.py
```

## 候选修复验证与建议

按根代理追加授权，只在临时快照尝试了 [resume-descendants-first.patch](/tmp/schema-b47-native-tool-exit-20261121/resume-descendants-first.patch)：SIGCONT 按 owned 父子深度降序恢复，保留原 SIGSTOP 顺序。这样也覆盖 Native CLI 下更深的 Bash，而不只把最外层 root 移到最后。

**此 patch 被诊断否决。** 四个固定窗口对照恢复正常，但第一个无注入延迟的验证仍出现 147。日志明确为 child CONT → root CONT，原 CLI `main()` 仍为 0。见 [候选失败回执](/tmp/schema-b47-native-tool-exit-20261121/cpu-results-after-resume-order-candidate.json)；断言在该失败处停止，未运行剩余两例。临时源码已恢复固定原版；patch 仅作被否决方案留存，不能作为修复交付。停止/恢复的异步处理及 Bash 已缓存的作业状态尚未进一步分解。

实际影响：B41 使用分号串接并读到了数据，所以不能把 EXIT147 本身写成查询失败或候选资格的必然阻断；但已验证 `&&` 会短路，故也不能把它降格为纯显示问题。下一次受控修复应以“原查询数据成功、shell 状态 0、后续命令继续”为验收条件，同时覆盖自然调度和固定窗口。monitor off 在本 fixture 有效，可作为后续小修复候选；尚未验证它在真实 Native CLI 的 shell 初始化/恢复层是否持续生效，不在此直接改保护规则或建议运行中切换。

本次模型/GPU/外部网络调用均为 0。只读固定 AKA 与指定 B41 公开证据；未访问 B45 现场、账户、套件、旧控制 RPC、AKA_experiment 或 archive，未修改公共源码；环境 identity 已解决的问题没有重新调查，可选 bundle 也未改动。B47 到此结束。
