# R1 B：双后端、结果、进程与隔离

本轮局部出口已实现并通过 11 项离线测试；未执行正式提取或真实模型调用。保留原 `processes.py` 的启动门、Linux boot/start 身份、pidfd 信号和终止确认；本轮没有修改其正确性逻辑。新增 `backends.py`、`isolation.py`，重写 `agent.py` 共用生命周期，更新 `audit.py / observation.py` 的后端口径及相关测试。没有修改 A 的产物/回执文件，也没有为旧调用方保留 `CodexCLI` 别名。

## 调用接口

```python
from schema_extract.tools.agent import AgentCLI

cli = AgentCLI(backend=config['backend'], executable=config['agent_binary'], model=config['model'])
result = cli.run(attempt=attempt, prompt=prompt, schema=proposal_schema,
                 timeout=seconds, on_process=record_process, cancel_event=cancel_event)
```

`backend/model` 构造参数必填。`backends.DEFAULT_BACKEND / DEFAULT_MODEL / DEFAULT_EFFORT` 分别是 `claude / qwen3.8-max[1m] / max`，仅供新建 run 的入口使用。恢复和已登记调用不得用这些默认值补字段。Codex 保留独立模型、effort 和本机选定 `model_provider` 配置，必须选择它支持的 effort，不能把 Qwen/max 自动套过去。

attempt 必需：`backend, model, effort, run_id, attempt_id, stage, node, role, directory, workspace, session_directory, session_scope, session_id, writable_files, isolation`。`run_id / attempt_id / 非空 session_id` 要求精确 UUID。proposal 顶层 `attempt_id` 就是 nonce，不新增另一字段。

- `session_scope` 恰含 `{stage, node, role, operation}`，前三项必须与 attempt 相同。normal、adjudicate、signoff 用不同 operation；P5 按 node 分开，P1/P2 不共会话。
- `session_directory` 是 attempt 之外的稳定、独占目录，每个 scope 各一份。Runner 只保存 role UUID；Agent 另以 `session.json` 和锁核对 backend/model/effort/scope、可执行文件/版本及连接配置指纹。
- 首次 `session_id=None`；续接必须等于登记 UUID。新 attempt 的实际 workspace 可以变化，CLI 始终在 `/workspace` 中执行。每次动态信息通过文件 stdin 提供。
- `writable_files` 只允许 workspace 直属文件名。worker 的每个授权文件须由 Runner **启动前创建**（例如 P5 的 starter YAML、P2 的空 dispatch）；不支持替换挂载目录或以 rename 替换挂载点。reviewer 必须给空列表。
- 可选 `executable / cli_version` 若已登记，会再次核对。`executable` 应是解析后的绝对路径，不能直接填未解析的 `claude` 名称。

## 材料和自检映射

```python
attempt['isolation'] = {
    'read_only': [{'source': str(frozen_source), 'target': str(frozen_source)},
                  {'source': str(frozen_rules), 'target': str(frozen_rules)}],
    'check_command': [python_executable, frozen_phase_helper, '--read-only'],
    'python_path': [str(frozen_rules)],
}
```

`read_only` 必须显式给出，允许空列表；每项恰有 source/target。target 为规范绝对路径，不能覆盖另一材料或隔离内部目录。可保持冻结材料原宿主路径不变：只有指定叶子被挂载，父目录为空，其余 Runner 状态不会因路径前缀相同而暴露。也可选择 `/source` 等逻辑路径，但 prompt、helper 和来源定位须采用同一映射。

| 宿主材料 | CLI 内位置 | 权限 |
| --- | --- | --- |
| attempt/workspace | `/workspace` | 目录及非授权文件只读；worker 授权文件分别可写 |
| 显式冻结源码、规则、证据 | 每项 target | 只读 |
| session_directory/home | `/home/agent` | CLI 专用，可写；外层 session.json/锁不可见 |
| attempt/cli-tmp | `/tmp` | 本次 CLI 专用，可写 |
| 实际 prompt、schema | `/run/extract/prompt.txt`、`response.schema.json` | 只读 |
| 固定检查包装器 | `/run/extract/check` | 只读、可执行、不接受额外参数 |

`check_command / python_path` 可省略。它们是 Runner 指定的固定 argv 与冻结 Python 导入路径；本模块不决定阶段检查内容。R2 应按阶段复用 A 的 `check_stage_snapshot`，P1/P2/P4 不运行完整目录硬门禁；确需原 Template CLI 时显式传 `--read-only --skip-hardware`。Claude 只为该包装器开放精确的 `Bash(/run/extract/check)`，其他 Bash 命令由 dontAsk 拒绝。实际文件边界由 namespace 和挂载保证，工具白名单不承担 OS 隔离职责。

## 成功结果与会话结算

`result.outcome == 'completed'` 已同时检查：退出 0、终止确认、隔离证明、唯一最终原生成功、会话一致、结构化对象的 schema 和独立 nonce。Claude 只采用 `structured_output`；普通 result 文本不会冒充对象。Codex 核对原生末条 agent_message JSON 与 CLI 的 output-last-message 文件。原始 JSONL 和 stderr 保留，不伪造另一家事件。

常用结果键：`backend, outcome, exit_code, termination, native_status, session_id, session_status, response, response_validation, isolation_verified, requested_model, requested_effort, reported_model, reported_models, reported_effort, cli_reported_model, cli_version, trace, usage, raw_logs, elapsed_seconds`。

`model / effort` 仍表示请求值；reported 字段仅来自实际事件，未观察到就为 None。Claude init 中 model 单列为 `cli_reported_model`，不当作服务实际模型证明。

`session_status == 'ready'` 才能用返回 UUID 续接。正常成功和单纯格式错误可续接；超时/取消/原生失败在确认终止后退休。未确认终止保持 fenced，既不退休，也不准启动同 scope 后续调用。

恢复接口：

```python
read_result(attempt, exit_code=known_exit_code, termination=termination, outcome='completed')
settle_session(attempt, result=verified_result)
retire_session(attempt, termination=confirmed_termination, reason=reason)
```

`read_result` 重用同一 parser，核对 invocation 与 schema 绑定，不猜退出码。新增 `exit.json` 绑定 launch/attempt 并记录真实退出和终止结果；没有可靠退出证据时不可传 0。`settle_session` 用于 result 已落盘而 session 结算尚未落盘的崩溃窗口。取消/超时结果应保留其 outcome，不能拿完整日志重新解释成成功。session 退休不能绕过 `termination.confirmed`。

## 隔离事实与限制

在 Linux x86_64 上使用 user、mount、PID、IPC、UTS namespace；先建立私有根、只读绑定与逐文件写授权，再 pivot/detach 宿主根，清空全部 capability/bounding/ambient、设置 no_new_privileges，安装禁止重新挂载、namespace 创建、ptrace、文件句柄打开等操作的 seccomp。模型执行前由专用管道给 Runner 发送隔离证明，随后关闭该 FD。stdin 重新打开为内部只读挂载；stdout/stderr 使用管道由 Runner 保存，避免 CLI 借宿主日志 FD 改 Runner 文件。

PID namespace 的 init 退出会由内核清理全部子进程，离线测试覆盖 setsid + 双重 fork 后台写循环。进程身份与启动门保留原协议，不能用“进程似乎没了”清除 fence。

本机新挂载 procfs 返回 EPERM。实现没有绑定宿主 /proc，也没有制造固定的 `/proc/self`；只暴露真实只读 cpuinfo/meminfo。降权状态由启动器预开的自身 status FD 读取，执行 CLI 前关闭。真实 Claude 2.1.316 和 Codex 0.166.0 在此视图内 `--version` 均退出 0；**Codex 报告缺少 `/proc/self/exe`，无法自动创建 PATH aliases**。实际工具调用是否需要补充受控别名/运行依赖，明确留 R4 有界验证；目前不能声称该警告下全部原生 Codex 工具已实测。

外层边界禁止再次创建 namespace；因此 Codex 明确用 `-s danger-full-access` 关闭重复的内部 sandbox，真正写入权限仍由外层同一套挂载决定。不存在隔离失败后裸跑路径。

API 所需宿主网络保留，**没有声称网络隔离**。私有配置只继承必要认证、连接与选定 provider；不读取全部环境到审计，不复制个人 hooks/plugins/MCP 配置。Claude effort 参数和环境一致，不继承固定 thinking budget；两后端禁止内部委派，Reviewer 的产物一直只读。

## 用量口径

`parse_events(path, backend=...)` 必须指定后端，返回原生完成信息、trace 与 usage。`usage_summary` 供 attempt ledger；`usage_totals` 按 `(run_id/origin_run_id, attempt_id)` 去重，冲突记录不计已知小计，并分别按 backend/counter_scope 展示。

Claude 2.1.316 内置 schema 明确：`modelUsage / total_cost_usd` 可随 resume 承接累计值；`usage` 仅为当前 turn 主循环；`model_usage_since_last_result` 才是本进程已报告增量，缺失不等于 `{}`。实现保留全部原件，优先有 result_index=0 的原生增量；无此字段仅报主循环小计和未知费用，不相加累计。缓存读和写分列，Claude 不采用 Codex 的缓存输入子集公式。费用是 CLI 报告估计，账单仍未知。

Codex 的有界 native counter 观测仅在明确 codex 后端启用，读取该 scope 的私有 home、精确 session UUID；保留主会话累计差额和未观察到子会话的限制。未知缓存/成本不补 0，失败与恢复不重置原调用预算。

## 后轮接线与证据

- R2：Store 填齐上述身份、独立 session 目录、预建授权文件；Runner/所有特殊调用/review-eval 共用 AgentCLI，检查统一 outcome/session_status。prompt 和来源材料用相同 target；冻结文件列表加入新增 backends/isolation。
- R3：recovery 接 `read_result / settle_session / retire_session` 与 exit/fence；ledger 保留 backend、原口径 usage；audit 的状态格式/旧继承分支清理、完整隔离证据打包仍归 R3。本轮只改后端 parser 与用量汇总，没有抢写恢复/预算/运行状态主链。
- Codex 若需要恢复其原生末消息交叉验证，保留具体 `attempt/cli-tmp/response.json`。不要为取这一个证据而归档整个 CLI home、配置或临时目录。
- R4：完整动态 proposal schema、真实 Claude 工具读写/精确 resume、Codex 正常工具能力与无 procfs 的实际影响、主链联合验收。当前没有声明这些已完成。

已通过的 11 项测试及命令见 [R1_B_checks.txt](R1_B_checks.txt)，本机 CLI schema 文义、启动及 procfs 限制见 [R1_B_cli_semantics.txt](R1_B_cli_semantics.txt)。测试本身保留在 `test_backends_isolation.py / test_usage_observation.py`，可重跑核对。没有修改 wiki、旧 run、Template 字段规则或个人配置，没有 commit。
