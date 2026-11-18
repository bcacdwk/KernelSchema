# Claude 默认输出空间与 Atrex 对照

2026-11-18；第一轮只读评估。建议新执行器默认请求 **128000 tokens/原生模型响应**，不要求逐次传参，不增加累计 token 或时间预算。本轮只新增本文；没有真实模型调用、产品修改、全局设置修改或外部仓库写入，没有读取认证文件或输出模型隐藏推理。

## 已证实的边界

| 层次 | 已证实事实 | 不能据此声称 |
| --- | --- | --- |
| Extract 当前默认 | `runner.DEFAULT_CONFIG.max_output_tokens=None`；backend 不继承主机同名变量；只有显式参数才设置 `CLAUDE_CODE_MAX_OUTPUT_TOKENS`。 | 新增参数已经让默认输出空间变大。 |
| 本机 CLI 默认 | `qwen3.8-max[1m]`、`--effort max`、隔离配置下，实际请求 `max_tokens=32000`。 | 服务端只支持 32000。 |
| 本机 CLI 当前路径上限 | 输入 128000 原样发出；131072、256000、262144 均发出 **128000**。这是当前未识别模型路径的封顶。 | 设置 256K 就能向服务端请求 256K；所有模型/所有版本都只有这一上限。 |
| 真实服务已记录证据 | [LIVE05](LIVE05.md) 的一次 65536 短 JSON 探针成功。原探针临时文件已删除，长期证据是 LIVE05 的记录；本轮没有重跑。 | 服务能实际生成满 65536，更不能推导其支持满 128000/256000。 |
| 128K/256K 服务能力 | 本轮没有真实服务验证。128000 的请求接受能力、真实截断点、thinking/effort 的服务端映射均未知；256K 先受当前 CLI 路径封顶。 | 离线假 API 的 success 是 Qwen 服务端支持承诺。 |
| `[1m]` | 声明上下文窗口；捕获中模型 ID 发为 `qwen3.8-max`。 | 单响应输出自动变成 1M；provider 已确认实际 1M 能力。 |
| 累计用量 | 一次 CLI 调用可包含多次模型响应；累计输出可超过单响应 `max_tokens`。Extract 现有 usage/审计统计与该字段分开。 | `max_output_tokens` 是整个 run、阶段或调用的累计 token 额度。 |

当前可执行文件为 `/usr/local/lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe`，版本 **2.1.316**，SHA-256 为 `fd13c70000fec3f2116dfbcbb6121210b4a29328bf0e46210a59dd594a67d824`，与 LIVE04 相同。静态路径 `age` → `tRn` → `K3t` 先取模型 default/upper，再对环境请求值封顶；通用常量为 `32000/128000`。已识别模型、模型目录/能力数据等可走其他上限路径，因此不把 128000 描述成所有 Claude 请求的绝对上限。

官方 [环境变量文档](https://code.claude.com/docs/en/env-vars.md) 明确：`CLAUDE_CODE_MAX_OUTPUT_TOKENS` 对多数请求设置输出上限；超过模型 cap 会下调；无法解析为已知模型时默认 32000、cap 128000。官方 [模型配置文档](https://code.claude.com/docs/en/model-config.md#extended-context) 将 `[1m]` 定义为上下文窗口。文档查询与安装版本的离线捕获相互印证；不把 Anthropic 原生模型规格套用于 Qwen 网关。官方模型总览访问被转到区域不可用页，未据此补造服务端数字。

## 本轮离线协议捕获

使用 Python 标准库回环假 API、全新临时 HOME/config/cwd、清空继承环境、仅回环 endpoint 和固定测试认证占位值。保留 Extract 的 `--model qwen3.8-max[1m] --effort max`、安全/隔离选项；工具为空、无 JSON schema，仅在此诊断使用 `--max-turns 1`。假 API 返回固定 `{"ok":true}` SSE，不运行模型。只保留协议字段及终结状态；不记录 headers、完整请求或会话内容。诊断的短进程清理等待不是产品执行政策。

| 环境请求值 | 捕获 `max_tokens` | 耗时秒 | CLI 终结 |
| --- | --- | --- | --- |
| 未设置 | 32000 | 0.538 | exit 0 / success / end_turn / 1 turn |
| 128000 | 128000 | 0.533 | 同上 |
| 131072 | 128000 | 0.628 | 同上 |
| 256000 | 128000 | 0.595 | 同上 |
| 262144 | 128000 | 0.534 | 同上 |

五次均为 `/v1/messages?beta=true`，`model=qwen3.8-max`、`thinking={"type":"adaptive","display":"updates"}`、`output_config={"effort":"max"}`、`stream=true`；没有 `MAX_THINKING_TOKENS`。临时目录与进程已清理。当前实际观察到的最大请求值是 **128000（十进制，不是 131072）**。

## LIVE04 原始记录复核

只解析原 `invocation.json`、`result.json` 和 `events.jsonl` 的身份、事件类型、工具名、公开恢复提示与计数，未读取/展示隐藏推理正文。

| 任务 | 原 attempt | 本轮复核 |
| --- | --- | --- |
| FA4 | `8794f043-7c53-4991-85bf-886afd417fed/5fedff12-8b1d-4bb0-ae94-05384a590a6f` | `qwen3.8-max[1m]` / max；第 13806 行为 CLI 的 user 续写提示 `Output token limit hit.`；此前估算最大 30813，随后继续至 7137；最终 timeout / exit 143 / termination confirmed / survivors=[]。 |
| QSA | `7c31f6e1-e912-43d2-ada3-7b181f8245e5/7f12371a-9db5-4cf1-bcc1-acc7cbaef751` | 同模型/effort；没有该提示；估算最大 16140、最后 9480；同样 timeout / exit 143 / termination confirmed / survivors=[]。 |

两者均无原生 terminal `result/error`，公开工具名没有 Write/Edit。FA4 输出上限恢复与执行器 1800 秒结束是两件事；QSA 的未写稿不能归因于已观察到的输出上限事件。`estimated_tokens` 不是 provider 权威计数，assistant 初始 usage=0 也不是最终消费。当前 backend 对 Claude 保留 `result.usage`，并在可用时采用 `model_usage_since_last_result` 的 invocation delta；`modelUsage` 的累计口径不可随意重复加总。默认输出空间调整不应改动这些统计语义。

## Atrex 代码能支持的结论

检查的是 `/home/shaohanyong.shy/atrex-kernel-agent` 当前工作树（HEAD `dc388550ab6b32503e9657191e83673411bfd535`），主路径是 Light/native；不读取其私有登录设置或历史隐藏会话。

| 项目 | Atrex 实际代码路径 | 与 Extract 的关系 |
| --- | --- | --- |
| 调用方式 | `light/agent/provider.py` → `light/agent/native.py` / `runtime/runtime.py` → 共享 `legacy/agent/adapter.py:309`，直接拼 `claude --print --verbose --dangerously-skip-permissions --output-format stream-json --session-id … --effort …`。 | 主路径是原生 CLI 子进程，不是 Claude Agent SDK。仓库没有在该命令固定 CLI 版本/绝对路径；当前 shell 解析到同一安装，不证明历史运行完全同版。 |
| 模型/输出参数 | 共享 adapter **不传 `--model`、`--max-turns`、`--max-budget-usd`，没有显式输出 token 参数**；`ATREX_CLAUDE_SESSION_SETTINGS` 或通用 `ATREX_SESSION_SETTINGS` 可传给 `--settings`。 | 不能仅凭“都用 Claude”断言 Atrex 历史调用也固定使用 Extract 的 Qwen 模型路径。 |
| 环境/持久设置 | `legacy/agent/runtime.py:91` 从 `os.environ.copy()` 开始，只清理少数 Atrex 私有变量；未过滤输出、thinking 或 API timeout 变量。原生模式保留主机 Home；`light/agent/home.py` 的 sandbox 路径会种入 `.claude/settings.json` 并在恢复时保留已有文件。 | Atrex 可继承用户配置的更大/更小输出空间、固定 thinking、模型或请求超时；代码没有主动设置更大的输出默认值。Extract 的白名单隔离会排除这些隐式设置。 |
| 输出上限恢复 | `light/agent/runtime/adapter.py:136` 针对 root session 的匹配 API error + terminal error 分类；`light/workflow/session.py:298` 保留同一 Episode/draft、新开 provider session，默认最多 2 次。 | 说明系统专门设计了此类错误路径，不能说它不会遇到问题；也不证明当前 CLI 的普通 `max_tokens` 自动续写一定进入这条恢复分支。 |
| 时间额度 | `light/agent/provider.py:25` 默认单调用 28800 秒；`light/workflow/provider.py:21` 默认 Episode 14400 秒；`runtime/process.py:506` 取调用上限与 Episode 剩余额度较小者。受管工具等待可从 active time 排除。 | Atrex 并非无时间限制，只是额度与计时方式不同；不能照搬到用户要求无外加时限的 Extract。 |
| 累计 tokens | `light/workflow/session.py:143/309` 在供应 `remaining_token_budget` 时，检查累计值后决定是否启动后续调用；恢复不会重置 Episode 统计。 | Atrex 另有累计预算接口；不应将其引入 Extract。 |

**因此，“Atrex 看起来没遇到”没有现成的代码因果证明。** 可证实的差别是 env/settings 继承、未显式固定模型、更长的 active-time 额度以及专用错误恢复；不能把这些可能差异当成历史运行事实。如果同一 CLI、同一未识别模型且没有输出覆盖，Atrex 也会落到同一 32000 默认路径。Wiki 的独立模型/effort 配置不是主优化 agent 的模型证据。

## 最小长期方案

1. **默认只定义一次。** 在 `schema_extract/tools/backends.py` 定义 Claude 专用常量 `128000`，由 `ClaudeBackend.environment()` 每次无条件设置 `CLAUDE_CODE_MAX_OUTPUT_TOKENS`。worker、reviewer、maintenance、`review-eval` 复用此路径。保持当前模型与 max effort；不设置固定 thinking budget。不要把默认放在用户的 `~/.claude/settings.json`：当前隔离路径不会采用同名值，而且该文件会影响其他任务。
2. **删除本轮新增的逐次额度入口与重复身份字段。** 可撤掉两个 CLI 的 `--max-output-tokens`、config 默认/校验、逐层参数和 attempt/session/模型 receipt 身份中的该字段。常量已随 frozen executor 源码进入 manifest、executor digest 和 inputs fingerprint；保留输出环境值进入既有 `configuration_digest`，会话仍有配置一致性检查。若需要直观诊断，只在 invocation 留一处由实际环境导出的请求值，明确它不是已验证服务端上限，不再要求模型在回执中回填。
3. **累计统计照常，不能变成停止条件。** 不引入总 token/cost 额度、按累计用量暂停或 SDK `max_turns/max_budget`；离线捕获的 `--max-turns 1` 不进入产品。时间删除与巡检交由相应方案，不能为了放大输出另外加入超时或重试预算。
4. **服务端未知能力保持简单。** 实施验证时，同一路径做一次请求值为 128000、只要求短固定 JSON 的真实小探针；成功仅证明配置被该服务接受，不能证明填满额度或完整提取成功。请求拒绝则保留原生错误并具体处理服务能力，不能静默降到 32K/64K。无需每次运行探测、自动二分、模型能力数据库或降级框架。256000/262144 在当前路径会先变成 128000，因此同样的真实 CLI 短探针不能验证 256K；需要先取得可发送该值的受支持 CLI/服务配置证据，再谈真实验证，不伪装模型或修改 CLI 绕过。
5. **旧运行保持原冻结含义。** 新默认用于新 executor/run；不修改历史 `.schema_extract` 的 config、snapshot、attempt、回执来“升级”旧证据。冻结源码已经包含默认策略，无需再复制成一套新的预算身份协议。

涉及文件：`backends.py`（默认/env/digest）；`cli.py`、`runner.py`、`review_eval.py`（删除输入传递）；`agent.py`、`store.py`、`runtime.py`、`recovery.py`、`receipts.py`、`audit.py`、`monitor.py`（删除该值的重复身份/展示依赖）；对应 backend/CLI/review-eval/stage-flow 测试与 `schema_extract/README.md`。保留冻结执行器、session/process 身份与终结确认本身。receipt 的通用 null 类型修复是否还有独立需要，由源码审计分工判断，不因删掉额度字段直接回退。

最低验证：两个入口不传额度都实际产生 Claude env=128000；Codex 不受该默认影响；无主机输出/固定 thinking 配置漂移；冻结恢复仍由自己的 executor/env digest 决定；一组本地原生请求捕获证明线上同形参数路径。真实短探针留实施阶段，不在本轮执行，不填满 tokens，不跑全流程。

需用户决策：没有新的逐次参数决策。按已定方向推荐默认 **128000**；256K 在当前 CLI 路径没有额外效果。是否执行真实 128000 短探针归实施授权范围，本文不预先声称已经通过。
