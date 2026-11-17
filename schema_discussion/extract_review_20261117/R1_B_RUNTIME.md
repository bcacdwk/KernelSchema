R1 B：调用、会话、恢复与进程审阅。基线 `be4fc48`；本轮只使用离线事件、fake CLI 和临时目录。

已确认并修复的问题：

1. **确证未执行的调用仍会退休前次有效会话。** 最小反例为先成功建立 UUID，再在 `on_process` 登记启动门后取消：`launch_not_started=True`、终止已确认，Agent 的私有会话仍为 `ready`，但 `handoff_session` 将其改为 `retired` 并清空 Runner UUID。放行后 `exec_failed` 也确证 CLI 未执行，原实时结算却仅按“是否放行”决定退休。现统一使用绑定本 attempt 的未执行证明；恢复时只恢复该调用拥有的 `ready/empty` 状态，或保留尚未接管的既有可用会话。没有结果文件时也可以恢复；已执行中断仍退休，终止不明仍 fence，没有新增状态。
2. **正常 exec 失败逃逸到未处理异常，可能污染固定空 cwd。** 删除 fake 可执行文件触发 `exec_failed` 后，本机 Python 异常钩子在 cwd 创建 `.excp`，后续精确 resume 因 cwd 不为空失败。启动门现在登记 `exec_failed`、向 stderr 写诊断并返回 125；已有 child-subreaper 和后代结清规则不变。
3. **显式 Codex 的 `max` 在冻结 run 后才被拒绝。** `runner.create_run` 原通用 effort 集合允许它，`CodexBackend.command` 不支持它。B 将两后端已有集合提为 `validate_effort` 方法，command 继续复用；A 在创建 run 时复用该接口提前拒绝。没有扩大提取入口的 effort 选项。

验证结果：

- 最终限定运行 `test_backends_runtime`、`test_current_recovery.CurrentRecoveryTests`、`test_usage_observation`：**23 项全部通过，13.140 秒**。测试进程 PATH 前置阻断真实 `claude` / `codex`；调用均使用明确的 fake 可执行文件。
- 新增回归覆盖启动门前取消、放行后 exec 失败、结果存在/丢失、会话接管前/后的恢复，最后实际 fake resume 核对前次 UUID 与历史仍保留。
- 既有反向测试确认真正中断会退休、错误 nonce/schema/native result 不成功、缺少后代结清证据仍 fence。双后端、双角色、空 cwd、独立 attempt 路径、精确 resume、普通及脱离后代清理、Claude 增量/累计分开与 Codex 精确会话观测均通过相关回归。
- 按本次测试临时目录检查进程，剩余 **0**；本代理所改文件 `git diff --check` 通过。没有真实模型调用、queue、GPU、提取或发布批次。

保留的关键行为：默认 Claude Code `qwen3.8-max[1m]`、worker/reviewer 均为 `max`；显式 Codex 使用自身协议；原生成功、真实退出、schema、nonce、精确 session 同时约束结果；已用账目不重置；普通路径、CLI 角色权限与既有进程收尾保持。`runtime.py`、`observation.py` 无须修改，没有恢复隔离或增加历史兼容。

跨模块与文案：A 已接入创建时 effort 校验。`review_eval.evaluate` 原本就在创建输出前调用后端 command 校验，CLI 共用 choices 无须因此改写。当前文档关于确证未启动保留前次会话的承诺已由本次修复落实；无本轮已确认但未修的问题，无必要文案变更。真实 CLI 和完整流程未在本轮验证。

本代理实际修改：`schema_extract/tools/agent.py`、`backends.py`、`processes.py`、`recovery.py`、`tests/test_current_recovery.py`，以及本报告。没有写 A 所有的产品或测试文件，没有提交 commit；报告完成后停写。
