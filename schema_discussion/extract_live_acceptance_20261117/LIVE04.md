# LIVE-04：Claude Code / Qwen 的单响应输出预算

结论：确认 FA4 的一次模型响应触发了 Claude Code 的输出上限恢复；本机 CLI 在当前参数下实际发送 `max_tokens=32000`、adaptive thinking 和 `effort=max`。没有发现本次丢失既有预算配置、CLI 忽略 effort 或固定 thinking budget 与输出预算冲突的证据。这属于 CLI 默认输出预算下的模型长思考行为，不能据此认定 Extract 执行器有缺陷。QSA 没有出现同一上限事件，不能用 FA4 的现象解释 QSA 的未写稿。

## 真实运行证据

本次只读核查以下 P1 worker；证据目录为 `.schema_extract/live_acceptance_20261117/runs/RUN/attempts/ATTEMPT/`。

| 任务 | run | attempt |
| --- | --- | --- |
| FA4 Prefill | `8794f043-7c53-4991-85bf-886afd417fed` | `5fedff12-8b1d-4bb0-ae94-05384a590a6f` |
| QSA Prefill | `7c31f6e1-e912-43d2-ada3-7b181f8245e5` | `7f12371a-9db5-4cf1-bcc1-acc7cbaef751` |

截至 2026-11-17 14:26 UTC，两次调用都已结束。`result.json` 分别记录 `outcome=timeout`、CLI 耗时 1800.649 / 1800.700 秒、exit 143；终止均已确认且 `survivors=[]`。两份 workspace `root.yaml` 都仍为 1536 字节，与各自 baseline 字节相同。公开工具调用中均没有 Write/Edit。

FA4 `events.jsonl` 第 13804 行的 `system/thinking_tokens.estimated_tokens=30813`；第 13805 行仅检查 content block 类型，为 `thinking`；第 13806 行是 **`type=user`** 的 CLI 续写提示：

> Output token limit hit. Resume directly — no apology, no recap of what you were doing. Pick up mid-thought if that is where the cut happened. Break remaining work into smaller pieces.

因此它不是 assistant 自己声称出错。第 13807 行的思考估算重新从 1 开始，之后继续到最后第 15753 行的 7137，说明 CLI 已自动发起后续响应，随后才到执行器的 1800 秒截止。

QSA 共 12558 行事件，没有这条输出上限提示；其单段思考估算最大值为第 9558 行的 16140，最后第 12558 行为 9480。它最后仍在响应过程中，不能将未写稿归因于 32000 的输出上限。

两份原生流都没有最终 `result` 或 `error` 事件。已发出的 assistant 消息的 `message.stop_reason` 均为 null、`usage.output_tokens` 均为 0；启动命令没有 `--include-partial-messages`，因而没有原始 `message_delta` 停止帧可用于回溯精确计数。这些 assistant 帧的 0 不是已完成响应的累计用量，`estimated_tokens` 也不是 provider 的权威 token 计数。不能将 30813 宣称为精确的 32000 消耗，或据此指控 provider 错报用量。

## 启动配置与本机 CLI

两份 `invocation.json` 均明确请求 `qwen3.8-max[1m]` / `max`，实际 argv 同时包含 `--model qwen3.8-max[1m]` 和 `--effort max`。在进程退出前，仅对白名单键读取 `/proc/PID/environ`：`CLAUDE_CODE_EFFORT_LEVEL=max`，`CLAUDE_CODE_MAX_OUTPUT_TOKENS`、`MAX_THINKING_TOKENS`、`CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING` 都不存在。

主机 Claude settings 的 env 和诊断进程 env 中也没有上述输出/思考预算设置；主机存在的模型别名设置与本次显式模型一致。现有 [ClaudeBackend.environment](../../schema_extract/tools/backends.py) 仅转交连接/代理配置，并显式设置 effort；它确实没有输出预算入口，但本次没有一个已配置的输出预算被它丢掉。其排除固定 `MAX_THINKING_TOKENS` 是已有测试明确要求的行为，不应在本次诊断中顺手恢复。

本机 `/usr/local/bin/claude --help` 确认 `--effort` 支持 max；真实可执行文件为：

```text
/usr/local/lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe
version: 2.1.316 (Claude Code)
sha256: fd13c70000fec3f2116dfbcbb6121210b4a29328bf0e46210a59dd594a67d824
```

只读检查该已安装二进制中的 CLI JavaScript，可复核以下静态依据；偏移为文件字节偏移，适用于上述 SHA：

| 偏移 / 函数 | 含义 |
| --- | --- |
| 276979007 附近 | 通用常量 `Qw=32000,jG=128000`。 |
| 276982736 `age`、276983178 `tRn` | 未识别模型使用通用输出默认值；`CLAUDE_CODE_MAX_OUTPUT_TOKENS` 可覆盖，按模型允许上限封顶。已识别模型/显式能力配置可改变此路径。 |
| 276850120 `K3t` | 未设置环境值时取默认；无效或非正值退回默认；超上限时封顶。 |
| 278275495 `iS`、278275946 `mae`、278280751 `dv` | effort 能力判断与 max 环境值解析独立于输出 token 上限。 |
| 275511370 附近、292164673 附近 | 上述原文是内置输出上限续写提示，属于 CLI 的恢复分支。 |

`[1m]` 选择上下文窗口，不将输出预算自动提升为 1M。128000 是此 CLI 的通用配置边界，不是已核实的 Qwen 服务端支持承诺。

## 不调用真实模型的本地请求捕获

为避免只凭静态代码推断，使用 Python 标准库 `http.server.ThreadingHTTPServer` 在 `127.0.0.1` 的临时端口提供假 `/v1/messages`；它仅返回固定 `OK` 的 SSE 响应，没有模型推理。以 `tempfile.TemporaryDirectory(prefix="live04-cli-budget-")` 创建空 HOME / Claude config，在清空继承环境后把 endpoint 指向该回环地址，开启安全模式、禁用非必要流量和更新。没有读取或使用真实认证信息。

调用方式是 `subprocess.Popen(..., cwd=临时目录, env=上述隔离环境, start_new_session=True)`，stdin 为固定协议测试文本，最长等待 25 秒。关键 argv 如下；这是协议预算复核，工具列表为空，不读取仓库材料：

```text
/usr/local/bin/claude -p
  --model qwen3.8-max[1m] --effort max
  --output-format stream-json --verbose
  --safe-mode --restricted --disable-slash-commands --no-chrome
  --setting-sources "" --strict-mcp-config --mcp-config '{"mcpServers":{}}'
  --settings '{"disableAllHooks":true}'
  --permission-mode dontAsk --permission-prompts none --tools ""
```

服务端只提取请求体中下列非敏感字段，得到一次请求；未输出 headers 或请求正文：

```json
{
  "path": "/v1/messages?beta=true",
  "model": "qwen3.8-max",
  "max_tokens": 32000,
  "thinking": {"type": "adaptive", "display": "updates"},
  "output_config": {"effort": "max"},
  "stream": true
}
```

假服务返回后 CLI 正常 exit 0，原生 terminal 为 `subtype=success, stop_reason=end_turn`，最终 `usage.output_tokens=1`。这也验证 assistant 消息的初始 0 与最终用量是不同观察点。

捕获仅保存在该 Python 进程的内存中并输出为上述字段；临时 `/tmp/live04-cli-budget-*` 目录已自动清理，没有另存真实请求、认证或调试日志。可长期引用的捕获记录就是本节，执行代码在本次诊断工具调用记录中。

这项离线复核证明本机 CLI 的请求参数路径：max effort 已送入协议，adaptive thinking 与 32000 单响应输出上限同时存在；不存在额外固定 thinking budget。它不能证明真实 provider 在服务端如何映射 adaptive / effort，也不能证明增加输出预算就一定写出稿件。

## 最小后续方案与冻结边界

本次不修改产品。保持固定模型和 max effort；FA4 的输出上限与两次执行器超时是两个不同的限制。增加时间能容纳更多后续工作，不能改变单响应 32000 的上限；增加输出预算也不能解释或直接解决 QSA 已观察到的阅读/长思考行为。

主代理已决定由新的 xhigh 代理处理下一轮显式输出预算配置。最小实现方向是给 Claude 单独提供明确的 `max_output_tokens` 输入，冻结到 run 配置并进入调用配置指纹，由 backend 设置 CLI 已支持的 `CLAUDE_CODE_MAX_OUTPUT_TOKENS`。不要依赖任意主机 env 的隐式变化，不放开整个 settings/env，不恢复固定 thinking budget，不降低 effort 或替换模型。值的合法性应在启动前检查，并区分本机 CLI 上限与服务端能力；下一轮实测仍需检查是否发生截断、是否写出稿件以及最终验收结果。

该变化应通过新冻结配置/运行边界实施，不能改写本次 `.schema_extract` 快照、attempt、回执或 session 配置指纹。后续如何续接材料、分配时间及监督真实模型，由主代理安排；本诊断没有执行恢复或额外真实模型调用。

2026-11-17：仅新增本记录。两个快照的 `backends.py` 与当前产品文件字节相同；没有修改产品、运行目录或认证配置，没有读取/复述隐藏推理正文，没有运行全量或新增仓库测试，没有 git 提交。诊断完成，停止写入。
