# 移除自建进程隔离：实施记录

本次实现及取消竞态修复已完成，并经另一位新 XHIGH 独立审阅和复验签收，见 [REVIEW.md](REVIEW.md)。未提交 Git commit。当前 CLI 直接以调用用户身份运行，Extract 的自建 namespace/proc/pivot/capability/seccomp、隔离证明与虚拟路径均已删除；未增加关闭开关、旧产物兼容分支或替代沙箱。

## 本次基线

- Git HEAD：`27f36e5`；工作区已包含上一轮完整未提交重构，本次没有 reset/checkout 或覆盖其范围外成果。
- 删减前快照：`/tmp/extract-isolation-removal-baseline-49sshq5c`。`MANIFEST.json` 记录 87 个活动源码、prompt、测试与相关 Markdown 的相对路径、SHA256 和大小，`BASELINE.txt` 记录 HEAD 与原工作区状态；未复制旧 run、凭据或 archive。
- 本次最终文件差异清单见 `DELTA_MANIFEST.json`；独立审阅应将这些文件与上述快照比较，避免把 HEAD 到当前的上一轮成果混入本次差异。便捷 unified diff：`/tmp/extract-isolation-removal-current.diff`。

## 实际删除与保留

- 删除 345 行 `tools/isolation.py`，以及 agent 的准备/握手/proof、read_result/Runner 的隔离成功条件、恢复与审计的隔离文件、原生用量的逻辑 home 映射。冻结执行器按实际 Python 文件枚举，因此不再冻结该模块。
- `prompts.py`、阅读 helper、review-eval 全部提供真实路径。材料与规则记录实际来源摘要，正式 Source 仍为固定 `repository_id@ref:path::symbol`；没有旧逻辑地址映射。
- 保留默认 Claude Code `qwen3.8-max[1m]` / `max` 与显式 Codex 选择，独立 scope、精确 UUID resume、禁止内部委派、schema/nonce/native terminal/output 身份核验、预算、原稿/规范化/独立 review、文件字段白名单、固定来源与当前 run 关系。
- artifact_inventory 的读前许可集合、软硬链接/类型检查、不可变 checkpoint/approval、跨代知识交接、P1 边界与选型职责投影未削弱。根文档、活动说明、CHANGELOG 与当前 FINAL/IMPLEMENTATION 顶部指向直接运行；旧审阅和实测正文保留历史含义。未修改 Template 字段规则、Schema Wiki、kernel 或用户配置。

## 小 API 与会话变化

- attempt 的隔离 specification 改为实际 `read_directories` 与可选固定 `check_command`；backend.command 接收实际 cwd、workspace、schema_path、output_path 和 writable_files。
- invocation/result 去掉 isolation_digest、isolation_verified、logical_workspace，记录真实 `cwd`、`workspace` 及 `cwd_before/cwd_after`；正常成功与恢复都核 cwd 身份和始末空目录。Codex usage observation 只索引该 scope 私有真实 home 下的精确 UUID。
- 每 scope 使用 `session_directory/cwd` 作为固定、应保持空的实际启动目录；候选始终在各自 `attempt/workspace`，开始基线仍由已登记 checkpoint/依赖摘要定位，结束原稿仍由现有独立 original 目录封存。没有镜像、回拷、可变历史候选或 HOME/PWD 伪装。
- Claude 继续 safe-mode/restricted/dontAsk，工具按角色裁剪，Edit/Write 许可精确到本次文件，Bash 仅许可本次固定 helper。helper 放在材料目录，记录摘要并拒绝额外参数。
- Codex worker 采用原生 workspace-write 加本次 workspace；reviewer 为原生 read-only，approval=never。增加已通过本机正反诊断的 `project_doc_max_bytes=0`，避免真实 cwd 父目录的 AGENTS.md 混入冻结任务。未静默放宽权限或引入未核实的特性开关。
- 这些是普通用户进程与 CLI 工具权限；正式采纳由 Runner 的文件/字段/来源检查决定，不提供 Extract 对宿主文件系统的 OS 保护承诺。

## 进程结束

现有 launch gate 增加本次调用期间的 Linux child-subreaper 职责，不提权、不建 namespace、不常驻为服务。快速 double-fork/setsid 后代仍被本次 gate 收养；主 CLI 的实际 exit 与 gate 实际 wait/reap 结清分别登记在 `cli-exit.json`。

正常结束、取消和超时沿用进程身份、父子和进程组登记。先终止后代，保留 gate 直到后代清空；不再以 killpg 兜底杀掉无法确认的归属锚点。gate 外部消失而没有实际回收闭合记录时保持 fence；旧 result/exit/termination 中的 confirmed 和曾强杀 gate 的动作均不能替代闭合记录。

实施阶段曾修复 gate Python 收尾期间收到 TERM、把已成功 CLI 误记为 -15 的竞态；其后独立审阅发现的另一处真实 fork/扫描竞态及最终修复如下。

## 独立审阅后的取消竞态修复

独立审阅者用真实 fork/setsid 复现：ps 已列父进程、读取其 /proc/stat 前父进程 fork 并退出，新后代可能不在该次快照内。旧 terminate_owned 因扫描只见 gate 而 SIGKILL 它，再由 agent/recovery 的“本次强杀 gate”例外误记全部终止；实际仍有后代存活。

修复只涉及这一结束条件：terminate_owned 不再向 gate 发送 TERM/KILL，在有界 TERM/KILL 阶段中继续观察并终止新发现的后代；只有 gate 自身 wait/reap 到 ECHILD 后写出闭合并退出。持续漏观测时保留 gate 和 fence，不猜测已清空。

agent 删除强杀 gate 替代闭合的例外；read_result 对已启动调用重新核实际 gate 闭合。recovery 的有/无 process.json 路径均不再从旧 confirmed 循环推导结束；verified_result 拒绝缺闭合记录的结果，audit 沿用这个核验。无需新增进程平台或修改其他提取功能。

- 默认 grace=1 秒的真实反例现保留 gate，返回 confirmed=false；解除复现的强制调度后，同一调用可以正常取消，gate_reaped=true 且后代无存活。结果见 [GATE_RACE_RECHECK.json](GATE_RACE_RECHECK.json)，脚本见 [gate_race_recheck.py](gate_race_recheck.py)。该脚本只把原复现的 current 交接改成原子 replace，防协调文件瞬间空读，并在 fence 后验证恢复正常观察可收敛；没有伪造 ps/proc 行。最初直接重放原脚本遇到其协调文件空读，finally 已清理。
- 最终只选 9 项取消/恢复正反例，6.8 秒全部通过，见 [gate-repair-tests.txt](gate-repair-tests.txt)。覆盖普通成功、成功后脱离后代、普通子进程、double-fork 取消、timeout、未启动、恢复、保留旧 confirmed/强杀动作仍拒绝、无 process.json 的同类拒绝和选定审计拒绝。
- 原 3 次真实 CLI 证据未改写；其 cli-exit 尚无 descendants_reaped，只证明当时的直接路径和 CLI 操作，不代表最终 gate 闭合或当前审计已真实通过。本次修复没有追加模型调用、全 suite 或提取流程。
- 修复前局部源码快照：`/tmp/extract-gate-repair-baseline-i7wmy1r5`；便捷修复 diff：`/tmp/extract-gate-repair-final.diff`。实际反例后代由脚本 finally 清理/reap；9 项测试进程全部结束。

## 短离线验证

未重跑旧 125 项全 suite，也未执行真实 P1–P6 提取。

- 独立审阅前运行/恢复组：`test_backends_runtime` + `test_current_recovery`，18 项通过，28.3 秒。涵盖两后端/两角色、实际 cwd、独立 attempt、精确 resume、原子编辑、native/schema/nonce/output 门禁、普通子进程、double-fork/setsid 后代、成功后残留子进程、超时/取消、异常归属丢失 fence、恢复与选定审计。日志：`runtime-tests.txt`。
- 最终路径/用量组：`test_usage_observation` + 单项真实路径 helper 测试，6 项通过，3.1 秒。直接 home 的新会话/续接计数、索引越界/身份冲突拒绝、读前源身份、只读自检和额外参数拒绝。日志：`path-tests.txt`。测试 SQLite fixture 有非失败的 ResourceWarning；产品索引读取使用显式关闭连接。
- 材料、prompt、review protocol、review-eval 接线与当时运行组共 24 项通过，36.7 秒；其中 2 处旧 fixture/API 调用错误已修正。最后的运行修改由上述 18 项重验覆盖。
- 另选 8 项通过，113.9 秒：正常发布/冲突/重放/P6/source identity、单调用审计归档与原稿名称预检、原生 proposal/raw 门禁、reviewer 变更拒绝、原子替换与多余文件/链接拒绝、正常/失败/中断候选读前许可、跨代事实交接，以及 helper。均为离线 fixture，不调用模型。
- 两 CLI --help/--version、入口 --help、compileall、git diff --check 通过。Codex debug prompt-input 确认 worker/read-only 的实际角色配置，并以临时 Git 仓库的 sentinel 对照验证项目文档读取关闭。

## 三次极小真实 CLI

真实结果与摘要见 [REAL_CLI_RESULTS.json](REAL_CLI_RESULTS.json)，原始受限本机证据位于 `/home/shaohanyong.shy/.cache/extract-direct-smoke-1o0v77cx`。该目录的私有认证/home 未复制进仓库或归档。

| 调用 | 结果 | 用时 |
| --- | --- | --- |
| Claude worker 初次调用，指定文件 Read/Edit/Write 与读回 | completed，exit 0，实际 turn1 内容 | 16.5 秒 |
| Claude worker 同 UUID resume，在第二个 attempt 文件写入与读回 | completed，exit 0，保留前次 token 与独立 turn2 内容 | 16.7 秒 |
| Codex reviewer，原生 read-only 单命令读取指定文件 | completed，exit 0，原稿未变；原生日志确认 read-only/never | 18.2 秒 |

每次硬上限 65 秒；三次均有唯一原生成功终结、schema/nonce、精确 session 与真实进程结清，cwd 始末为空。Codex 真实 native counter 的 root_status=exact，完整子会话用量仍不推断。其 shell 的宿主 profile 产生 basename 警告，指定 cat 仍成功；该警告未伪造成工具内容成功之外的能力结论。

三次后没有追加模型调用。它们验证直接路径与原生角色启动；最后的 gate 结束竞态修复、失去归属的 fence、项目文档禁读和 helper 材料目录接线分别由离线测试/CLI 诊断覆盖。真实 Codex worker/resume、真实 Claude reviewer/helper 与完整真实提取链没有本次通过结论。

排查记录：官方 OpenAI 文档站在本机返回 403；本机 CLI help 与实际配置诊断用于核对接口。独立 sandbox 子命令的临时 profile 诊断曾报自身二进制 ENOENT，未据此引入绕过；产品原生 `exec -s read-only` 实际调用成功，不需要 proc/sudo 系统准备。

## 进程结清与独立签收

三次真实调用均保存 processes_closed=true；所有离线测试/短诊断进程已结束，最终 ps 检查没有本次 smoke、fake CLI 或启动脚本残留。未运行 GPU、完整算子提取、queue 消息或外部发布。

独立审阅者已核对本次快照增量、角色权限、真实路径与 attempt 证据，以及既有产物、批准和交接门禁。取消竞态经独立重放确认修复，8 项针对性复验通过；最终 39 文件清单及 87 文件基线 SHA 核验一致，没有其他未闭合的实质问题。未运行真实 P1–P6 全流程，既有真实调用证据保留其实际验证范围。
