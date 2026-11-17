# 提取方法与执行器对齐

当前 Markdown 定义少量具体完整设计、规范的 `dispatch.md` 与整区 `from`。Template 的 checker、starter、只读 API 和统一作者 CLI 已对齐当前字段；Extract 的 Python Runner、计划、权限和回执仍待对接，**不能用当前 Extract CLI 端到端运行这套新方法**。本页记录两者的实际边界，Template 检查通过不等于提取执行器已经可用。

## 方法规范

文件与记录规则以 [common](../schema_docs/0.common.md#派生节点与-dispatch) 为准，提取职责见 [P1–P6 总览](../EXTRACT_SCHEMA.md)与 [REVIEW](REVIEW.md)。目标是含 root 至多十份各有独特价值的具体完整设计，其他参数、局部变化和已知组合在目录中记录；内容共享按实际整区来源判断。

## 手工推演与阶段结论

在本次授权范围内，源码阅读、数学推导、调度片段和候选评议可以按现行文档继续，不以启动 Runner 或通过旧 checker 为前提。记录实际使用的材料、已核实关系和未完成部分；片段按其说明的关系核对，候选按选择依据核对。

阶段准入、完整性和独立审阅要求用于正式放行。缺少前置阶段批准或仅核对片段时，结论止于实际覆盖对象，不记为阶段完成。正式运行与检查须使用匹配本次规则的执行器；尚无匹配工具时如实保留检查未完成，不为通过旧工具改写为旧语法。手工工作不改变阶段写权限、Worker/Reviewer 分工或冻结 run 的状态与批准。

## 当前执行器

Template 已完成当前格式、工具和[四个案例](../schema_template/examples/README.md)的更新。下列 Extract 差异仍未对接；Runner、算子源码、冻结运行或批准状态未因此改变。

| 范围 | 当前实现事实 | 待对接的方法职责 |
| --- | --- | --- |
| Template 解析、检查与创建 | [resolver](../schema_template/tools/design_resolution.py)、[字段检查](../schema_template/tools/rules/structure.py)和 [starter](../schema_template/starter.yaml)已采用当前格式；覆盖整区 `from` DAG、实际来源链、使用节点绑定、raw 文本预算和 dispatch 最小检查。默认作者 CLI 会统一规范化整个任务及其引用；规范化库可只返回内存候选。 | Extract 使用匹配的严格只读校验 API，或显式调用 `check_yaml.py FILE --read-only --skip-hardware` 预览并检查规范化候选；适配候选/实际来源的区别、诊断与来源身份。 |
| P1 执行单元登记 | [artifacts.py](tools/artifacts.py) 的 `validate_changes` 仍要求整个硬件区与冻结 starter 相同，`_without_p1_fields` 也保留硬件区作受保护内容。 | P1 在现有 `execution` 中最小登记有据执行单元，其余硬件保持占位；当前门禁尚不接受这项流程权限。 |
| 目录与计划 | [artifacts.py](tools/artifacts.py)的 `validate_plan/index_prefix/validate_index` 要求七字段 plan、唯一根单父树和确定性的父表及覆盖缺口节。 | 有限完整集合、common 的记录格式、拟议内容与实际正文的区分。 |
| 调用、权限与证据 | [stages.py](tools/stages.py)按直接父批准解锁 P5；[artifacts.py](tools/artifacts.py)限制旧阶段写范围；[receipts.py](tools/receipts.py)按旧结论与依赖收集证据。 | P2/P3/P4 的批准对象与写权限、P5 实际来源依赖、普通记录的集中收尾和按影响返回。 |
| 修订、视图与交付 | 当前合入和复核沿旧父关系与代际推进；视图、回执、恢复和发布依赖登记的规则及执行能力。实现入口见[提取工具协议](tools/README.md)与[模板工具说明](../schema_template/tools/README.md)。 | 按实际定义和使用关系判断影响，保持 raw、完整来源身份、独立审阅、预算和最终文件集合一致。 |

## 冻结运行

旧 run 继续按其登记的 `source/config`、`rules` 和 `executor` 解释，身份含义见 [Supervisor 手册](README.md#输入与输出)。工作树的新文字不会自动改写旧状态、稿件、批准、回执或预算，也不能作为恢复旧 run 时替换冻结阶段页的依据。

现有命令及冻结协议仍由[操作手册](README.md)和[工具协议](tools/README.md)说明。语法、依赖或批准含义的改变需要实际工具支持和独立适用性验证；不能仅因现有 CLI 提供修订或升级入口，就把本次文字变更当作兼容运行层升级。

## 后续对接

后续代码工作须分别完成并验证：

1. 接入 Template 已提供的当前字段、`from`、raw/effective 来源身份、诊断和独立视图接口；将外部检查调用改为 `--read-only --skip-hardware`，避免默认排版越过阶段写权限。冻结运行仍使用其登记的规则与执行器。
2. 解除七字段计划和直接父表的耦合，支持有限完整集合与正式记录；实现数量、文件和链接等可检查项。独特设计价值、局部差异是否闭合仍由人工审阅。
3. 对接阶段批准对象和写权限，包括 P1 的执行单元最小登记、实际内容依赖、P5 新事实的目录收尾及定点返回，保持 Worker/Reviewer 分离、有效全文审阅和受限写入。
4. 按实际定义、使用者及共享 Contract 处理合入、修改影响与复核；保留 raw、历史批准和原预算。普通记录增补不要求全部重开，重大选择变化也不能降格为局部编辑。
5. 对齐冻结能力、旧 run 可恢复性、语义采用和发布回执，再用结构正反例、旧运行回归及独立语义案例验证。

本页记录已有 Template 接口及 Extract 待对接事实，不改变 Runner 命令、JSON 字段、状态或冻结规则。上述对接完成前，Template checker 通过不能证明提取方法已可端到端执行。

## Claude Code / Qwen 子任务调用与待接入边界

这里的“子代理”沿用当前执行器的含义：由 Supervisor/Runner 启动一个独立 CLI 进程，作为 worker 或 reviewer；不要求主代理本身使用 Claude，也不使用 Claude 内部的 Agent 工具再次委派。单独派发这种子任务只需调整调用命令。**接入现有 Extract Runner 还需要适配会话、回执、事件与权限，不能直接把 `--codex-bin` 改成 `claude`。** 当前 [agent.py](tools/agent.py)明确构造 Codex 参数，[runner.py](tools/runner.py)、[audit.py](tools/audit.py)和 [observation.py](tools/observation.py)也依赖 Codex 的完成事件及会话记录。

本入口固定模型 `qwen3.8-max[1m]`，只使用 `high / xhigh / max` 三档 effort；worker/reviewer 分别按本次登记值执行，默认可沿用当前 `xhigh`。接口地址与凭据来自已配置的 `~/.claude/settings.json` 或进程环境。effort 通过 `--effort` 设置；若环境中已有 `CLAUDE_CODE_EFFORT_LEVEL`，其优先级更高，因此派发器将该变量与本次参数设为同一值。这里只采用 effort 模式，不叠加固定 thinking budget。

可直接给主代理以下派发约定：

> 用 Claude Code 的非交互模式 `claude -p` 启动独立 Qwen worker/reviewer。每次调用明确模型、effort、角色、工作目录、允许修改的文件、输入材料和提案 schema。子进程不会继承主代理的对话，不要假设 Claude 自动读取 AGENTS.md；把相关规则、阶段要求和本次身份显式写入 prompt。worker 与 reviewer 使用独立工作区和角色会话，禁止在任务内继续委派。读取结构化结果并核对实际产物后，再由外层 Runner 决定是否采纳。

### 单次调用示例

下例为 worker；路径替换为本次已准备的工作区与 attempt 目录，`prompt.txt` 包含完整任务上下文，`response.schema.json` 是本次提案 schema。外层调用器负责超时、取消和进程结算。

```sh
(
  cd /absolute/worker-workspace || exit 1
  CLAUDE_CODE_EFFORT_LEVEL=xhigh claude -p \
    --model 'qwen3.8-max[1m]' --effort xhigh \
    --tools 'Read,Glob,Grep,Edit,Write' \
    --allowedTools 'Read,Glob,Grep,Edit,Write' \
    --permission-mode dontAsk \
    --strict-mcp-config --disable-slash-commands \
    --output-format stream-json --verbose \
    --json-schema "$(cat /absolute/attempt/response.schema.json)" \
    < /absolute/attempt/prompt.txt \
    > /absolute/attempt/events.jsonl \
    2> /absolute/attempt/stderr.txt
)
```

改为 `high` 或 `max` 时，同时替换环境变量和 `--effort` 的值。Reviewer 将两处工具列表都改为 `Read,Glob,Grep`，返回审阅提案，由 Runner 应用裁决补丁。上述列表适用于文件读写示例；执行 `reading.py` 等命令时，还须按实际任务补充受限命令权限和外部隔离，不能因此向 reviewer 开放任意可写 shell。

`--tools` 控制工具集合，`--allowedTools` 允许所列工具免交互审批，`dontAsk` 拒绝仍需审批的调用。它们不提供与 Codex `workspace-write/read-only` 相同的操作系统沙箱，也不自动限制 worker 只能改阶段许可文件。正式 Runner 接入仍须保留工作区隔离、只读来源、阶段写范围及实际差异检查；本示例不提供这部分隔离实现。

Python 派发器使用参数数组和 `Popen(cwd=workspace, ...)`，从 `schema_path.read_text()` 取得 `--json-schema` 参数，并将 prompt 文件作为 stdin；不必通过 shell 拼接上述命令。当前启动门、进程身份、单次超时和取消结算逻辑应继续由外层控制。

### 会话与结构化结果

首次调用不传 `--resume`，从 Claude 的初始化/最终结果事件保存实际 `session_id`。同一角色需要续写时，在同样的调用参数中增加：

```sh
--resume CLAUDE_SESSION_UUID
```

保留同一用户或 `CLAUDE_CONFIG_DIR` 下的会话数据及相应工作目录，每次续接仍显式传模型、effort、工具权限、本次 prompt 和提案 schema。需要恢复的调用不能使用 `--no-session-persistence`；不以 `--continue` 猜测最近会话，不把 Codex UUID 传给 Claude，也不让 reviewer 续接 worker 的会话。新建与续用时机仍遵循[角色与会话](tools/README.md#角色与会话)。

`--json-schema` 接收 JSON 内容，CLI 的最终 `type=result` 事件通过 `structured_output` 返回对象。成功结果示意：

```json
{
  "type": "result",
  "subtype": "success",
  "is_error": false,
  "session_id": "实际 Claude 会话 UUID",
  "structured_output": {"本次提案字段": "按 response.schema.json 返回"}
}
```

外层须同时检查进程退出 0、最终成功事件、无调用错误、会话身份一致，以及 `structured_output` 通过本次提案 schema/nonce 校验；随后把该对象写入原协议要求的 `response.json` 并进入已有规范化与产物检查。`result` 文本仅作辅助，不能把整份 CLI 结果封装或普通 prose 当成提案。`--output-format json` 也可返回单个结果对象，但正式接入宜保留上述 JSONL 事件记录。

### Runner 需要适配的部分

| 范围 | Claude/Qwen 接入要求 |
| --- | --- |
| 命令构造 | 新增明确的 Claude transport/backend 选择；使用 `-p`、`--model`、`--effort`、stdin 和 `Popen(cwd=...)`，不沿用 Codex 的 `exec`、`-C`、`-s`、`-c model_reasoning_effort` 参数。当前尚无这个 backend 选择入口。 |
| 提案返回 | 将 Codex 的 `--output-schema/--output-last-message` 对应到 `--json-schema` 与 `structured_output` 的提取落盘；保留原 proposal/receipt 的身份、规范化和校验职责。 |
| 完成事件 | 解析 Claude 的 `system/init`、`assistant/user`、`result` 等事件。把 Runner 当前直接依赖 `turn.completed` 的判断改为明确的 transport 完成状态；保留原始事件，不伪造成 Codex 事件。 |
| 会话与冻结身份 | 登记实际 CLI 类型、版本、模型、effort 和 Claude session ID；按角色精确续接。切换提供者时新建对应会话并按既有规则交接，不把已冻结的 Codex 调用解释成 Claude 调用。 |
| 用量与审计 | 适配 Claude 结果的 `usage/modelUsage` 及其缓存口径；旧 Codex session/SQLite 观测器不能原样复用。缺失项标记未知，不以估算费用替代实际账单。 |
| 权限与终止 | 实现与阶段职责一致的外部隔离，保留只读 reviewer、许可差异检查、启动登记、超时、取消、终止确认及预算；CLI 成功不替代这些门禁。 |

2026-11-17 已在 Claude Code `2.1.316` 上用隔离临时目录实测：`high` 读取文件并返回符合小型 schema 的 `structured_output`；`max` 用精确 UUID 续接，保留上一轮 nonce 且未重读文件；`xhigh` 新建独立会话、写入指定文件并返回结构化结果。三次均成功，续接 UUID 一致，worker 文件内容已核对。这验证了工具调用、结构化输出和会话续接的基本路径；完整动态提案 schema、正式隔离及 P1–P6 流程仍须在 transport 接入时验证。本节仅追加调用约定与待办，不表示现有 Runner 已支持 Claude/Qwen。
