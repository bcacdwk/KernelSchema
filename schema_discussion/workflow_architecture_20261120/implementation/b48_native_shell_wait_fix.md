# B48：原生 Python shell 等待的最小兼容修复

2026-11-21 UTC。**固定版本的临时快照中，局部关闭 Bash monitor 的补丁通过 45 项 CPU 回归和 10 次无恢复延迟的自然调度验证；仅建议应用到后续新会话，尚未修改公共源码。** 修复范围是已有 `python` / `python3` / `python3.10` shell 包装函数中的前台等待。

固定源：AKA `88d773b3acdd60ed13334dc46404e2d0caefe49f`。产物目录：[schema-b48-native-shell-wait-20261121](/tmp/schema-b48-native-shell-wait-20261121)。复用 [B47 报告](b47_native_tool_exit_diagnosis.md) 和其 fake coordinator 脚本；B47 原脚本原样保存为 [reproduce-b47-original.py](/tmp/schema-b48-native-shell-wait-20261121/reproduce-b47-original.py)。

## 修复与原因

[生产补丁](/tmp/schema-b48-native-shell-wait-20261121/native-shell-wait.patch) 只改 `src/aka/execution/runtime_tools/session_shell_guard.sh`，净增 19 行：原 `atrex_guarded_python` 的策略与执行体完整移为公开 helper，新同名包装层保存 monitor 状态、`set +m`、执行原实现、保存实际返回码，再恢复 monitor 并返回该码。新 helper 同样 readonly，能随原 CLI 的公开函数快照恢复。

monitor 开启时，Bash 的前台等待可在子进程 stopped 状态提前结束；即使 CLI 随后成功退出，已决定的 `&&` 仍会短路。局部关闭 monitor 使这段等待持续到真正终止，不依赖 SIGCONT 顺序或 sleep。生产补丁没有延迟、重试或退出码映射。

`SessionControl` 两轮 STOP、原 root-first CONT、工具发送回复的顺序、活动计时、原 sandbox CLI/protocol 均未改。原 host kernel-import guard 和保护路径检查仍在原执行体内。没有新增 RPC、账本、恢复平台、profile 或 bundle。

## CPU 证据

Linux Bash `4.4.20(1)-release`，只读复用指定 Python `3.13.12` 环境。全部请求使用新临时 Unix socket 和 fake coordinator，调用固定原 sandbox `main()`、原 transport 与 `SessionControl`；没有旧 workspace RPC。

| 同一 fixture 条件 | 原 guard | 补丁 guard |
| --- | --- | --- |
| monitor on，自然恢复顺序 | 3/3 次 shell 147；native main 为 0；跳过 `&&` | 10/10 次 shell 0；native main 为 0；执行 `&&`；恢复 monitor on |
| monitor on，root CONT 后注入 20ms 测试窗口 | shell 147；跳过 `&&` | shell 0；执行 `&&`；无 Stopped/Done |
| monitor off，相同注入窗口 | shell 0；执行 `&&` | shell 0；执行 `&&`；仍为 monitor off |
| monitor on，禁用 suspension 的控制组 | shell 0；执行 `&&` | 不用于生产修复 |
| 真正 CPU 子命令退出 7 或 147，经原 CLI 传回 | — | native main 与 shell 都原样返回 7 / 147；`&&` 正确跳过 |

自然调度证据：[bounded-results.json](/tmp/schema-b48-native-shell-wait-20261121/bounded-results.json)；首轮原版/补丁直接对照：[smoke-results.json](/tmp/schema-b48-native-shell-wait-20261121/smoke-results.json)。20ms 只在测试注入与模拟 host 工作中使用，补丁不含 sleep。

[最终 pytest 回执](/tmp/schema-b48-native-shell-wait-20261121/pytest-results.xml)：**45 passed，21.17s**。

- 36 个原 CLI 用例：BASH_ENV 脚本入口、公开函数快照脚本入口、快照 `python -c` 入口 × monitor on/off × 自然/注入 × 真实 exit 0/7/147。均检查 JSON、stderr、返回码、`&&`、monitor 恢复、所有自有进程在 request 内为 `T`、活动计时冻结和退出清理。详见 [完整回执](/tmp/schema-b48-native-shell-wait-20261121/cpu-final-receipts.json)。
- 9 个 shell 快照用例：原 2 项 host kernel-import / gateway 路径保护回归；新增 6 项 `set -u` 下状态/monitor 保留；1 项脚本 CLI 在 `set -e` 下真实退出 7 且不执行后续。
- `bash -n` 通过；生产与测试 patch 都对固定版本拷贝 `git apply --check` 通过。

测试沿用 B47 的 PID 白名单约束；`python -c` 原实现中的管道会多一层 shell，仅从本次自有 client 的 PPID 确认并纳入该直接子树，没有全机进程枚举。最终从本次保存证据汇总的 357 个 PID 均为 gone，见 [清理回执](/tmp/schema-b48-native-shell-wait-20261121/owned-pid-cleanup.json)。

初测的 fixture 两类问题已处理并留存回执：`-c` 的额外自有父 shell 超过最初两 PID 假设；自有进程退出时读 `/proc/PID/stat` 可抛 ESRCH。它们均在 fixture 中修复，产品补丁未因此扩展。另确认当前 Bash 下旧 guard 已存在 `set -e` 配合失败的 `python -c` 管道返回 1 的行为，两版一致；本任务不扩修它。真实原 CLI 脚本的 `set -e` / exit 7 另有回归。

## 交付与应用边界

- [native-shell-wait.patch](/tmp/schema-b48-native-shell-wait-20261121/native-shell-wait.patch)：单个生产文件的最小补丁。
- [native-shell-wait-tests.patch](/tmp/schema-b48-native-shell-wait-20261121/native-shell-wait-tests.patch)：原 shell 快照回归的增补和新 CPU suspension fixture。
- [manifest.json](/tmp/schema-b48-native-shell-wait-20261121/manifest.json)：固定 commit、实际变更文件、验证结果和所有交付物 SHA-256。
- [run_evidence.py](/tmp/schema-b48-native-shell-wait-20261121/run_evidence.py)：有限原版/补丁对照脚本。

重跑回归使用新的临时测试目录：

```bash
PYTHONPATH=/tmp/schema-b48-native-shell-wait-20261121/aka-source/src \
PYTHONDONTWRITEBYTECODE=1 \
/tmp/schema-b18-model-budget-usage-20261121/venv/bin/python -m pytest -q -p no:cacheprovider \
  /tmp/schema-b48-native-shell-wait-20261121/aka-source/tests/test_shell_wait.py \
  /tmp/schema-b48-native-shell-wait-20261121/aka-source/tests/test_shell_guard_snapshot.py
```

根确认后可应用生产与测试 patch 到上述固定版本，再在**新会话的新 shell / 新公开函数快照**中生效。已运行会话保留旧函数，不能靠修改文件保证即时生效；本次没有热修 B45。覆盖的是已有包装函数执行的前台 Python 调用；显式 `command python3`、绝对路径解释器、绕过包装函数的调用不在此修复范围，外层另行管理的 job-control pipeline 也未作全局兼容承诺。未验证其他 Bash 版本，未运行真实模型会话，不能据此宣称 Schema 的独立阻断原因已经消除。

本次模型/GPU/外部网络调用均为 0。未访问 B45 现场/进程/账户/包，未操作旧进程或旧 RPC，未访问 AKA_experiment（包括 archive），未改 KS 原 main 提取、公共 AKA 源码或共享 Python 环境；未 stage/commit/push。
