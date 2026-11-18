# LIVE-05：冻结 Claude 单响应输出预算

新增 `extract --max-output-tokens N` 和 `review-eval --max-output-tokens N`。本轮新 run 建议使用 **`--max-output-tokens 65536`**，保持 `--backend claude --model 'qwen3.8-max[1m]' --worker-effort max --reviewer-effort max`。本机 Claude Code 2.1.316 的回环请求捕获确认实际发送 `max_tokens=65536`；按主代理追加授权，只执行的一次真实 Qwen 短 JSON 探针成功。

## 参数与冻结边界

- 参数为可选正整数，只支持 Claude。Codex 显式收到该参数会报 `max_output_tokens is supported only by the Claude backend`；非法类型、零、负数被拒绝。
- 不指定时配置为 `null`，不传 `CLAUDE_CODE_MAX_OUTPUT_TOKENS`，保留原 CLI 默认行为。主机环境或 Claude settings 中的同名变量不会被隐式采用。
- `config.max_output_tokens` 与整个配置一同进入 inputs lock 和输入指纹。每个角色/阶段的 attempt 从冻结配置取值；执行边界、session binding、invocation、机器回执身份、恢复校验和审计均保留并核对该值。
- Claude 环境显式设置 `CLAUDE_CODE_MAX_OUTPUT_TOKENS=str(N)`，该环境值加入 `configuration_digest`。未指定参数时保持原连接配置指纹的计算内容。同一个 role session 不能更换预算。
- 参数请求的是单个原生模型响应的输出预算，不改变 call/stage/operator 时间，不增加重试，不设置固定 thinking budget。实际可用上限仍取决于 CLI 和服务端；本轮没有验证 128K。

实现复用现有 backend、冻结配置和 attempt 管线，仅增加一个共享的值校验函数。当前仓库没有 `add_agent_options` 公用函数，因此两个现有入口各声明该参数；没有引入额外适配层。README 只补参数含义。

## 本机 CLI 请求捕获

使用临时目录、空 Claude 配置、清空继承环境和仅指向 `127.0.0.1` 的假 API，调用新 `ClaudeBackend.environment` 与 `ClaudeBackend.command`。为了只验证请求协议，将命令的工具列表设为空、移除 `--json-schema`，加 `--max-turns 1`；临时 workspace/cwd 均为空，没有仓库材料。假 API 返回固定 `{"ok":true}` 的 SSE，不调用模型。

服务端只保留路径及下表中的非敏感字段，没有记录 headers、认证或完整请求正文。两次请求都为 `/v1/messages?beta=true`，模型名被 CLI 发为 `qwen3.8-max`，`thinking={"type":"adaptive","display":"updates"}`，`output_config={"effort":"max"}`，`stream=true`。

| 输入配置 | CLI 原生环境值 | 捕获的 `max_tokens` | CLI 结果 |
| --- | --- | --- | --- |
| 未指定 | 不存在 | 32000 | exit 0、success/end_turn、1 turn |
| 65536 | `65536` | 65536 | exit 0、success/end_turn、1 turn |

两次固定 JSON 均匹配，耗时分别为 0.596 / 0.638 秒。`MAX_THINKING_TOKENS` 均不存在。它们证明默认行为保留及显式值真实进入本机 CLI 请求，而不只是在 Python 配置中出现。

## 一次真实服务探针

主代理明确授权后，沿同一新 backend 环境路径使用现有认证来源，独立临时 home/config/cwd/workspace；没有修改主机 Claude settings 或认证。仍是空工具列表、没有 JSON schema 内部工具、`--max-turns 1`、180 秒超时。固定输入仅要求返回 `{"ok":true}`，不含仓库内容。

结果：

```json
{
  "cli_version": "2.1.316 (Claude Code)",
  "requested_model": "qwen3.8-max[1m]",
  "requested_effort": "max",
  "max_output_tokens": 65536,
  "native_output_env": "65536",
  "fixed_thinking_budget_present": false,
  "exit_code": 0,
  "timed_out": false,
  "elapsed_seconds": 2.126,
  "terminal": {
    "subtype": "success",
    "is_error": false,
    "stop_reason": "end_turn",
    "num_turns": 1,
    "duration_api_ms": 1576,
    "fixed_json_match": true
  }
}
```

只有一次真实 CLI 调用，没有再次尝试。捕获流只读取事件类型及终结状态/固定输出校验，没有读取或复述模型隐藏推理正文。临时 home 和原生会话日志均自动清理，进程已退出。

这证明当前服务接受该预算配置下的一次短响应；没有生成 65536 tokens，也没有证明服务端一定允许填满该额度，不能代替正式 proposal、worker/reviewer 或 P1–P6 成功验收。FA4 的已观察输出上限和 QSA 的未写稿原因仍保持 LIVE-04 的区分。

## 定向验证

新增两个用例覆盖显式传参/外部环境隔离/配置指纹/session 不可漂移，以及 CLI 参数拒绝/配置冻结/恢复后实际进程环境/audit。扩展既有 review-eval 用例检查预算配置和 attempt，扩展既有六阶段单 root 用例为 Claude 65536 / Codex 默认。

首次相关回归发现本次新增的 `null` 默认值进入机器回执 identity 后，既有标量 schema 生成器将其当成 string；已改为按实际值生成 `null` 类型。该修复作用于执行器回执身份，没有改 schema_docs 字段规则或示例 YAML。

最终回归命令：

```sh
python3 -B -m unittest \
  schema_extract.tools.tests.test_backends_runtime \
  schema_extract.tools.tests.test_cli_recovery \
  schema_extract.tools.tests.test_review_eval \
  schema_extract.tools.tests.test_current_recovery \
  schema_extract.tools.tests.test_stage_flow
```

最终结果：**40 tests，310.336 秒，OK**。覆盖 Claude 65536 和 Codex 默认配置的完整六阶段成功路径，以及未指定预算的既有默认流程。`extract --help`、`review-eval --help` 均展示新选项；`git diff --check` 已通过。

## 独立发现：无效参数保留 output 预登记

该行为没有在本任务中修复，已交主代理单独处理。以任意合法小型 source descriptor、全新临时 state-root、尚不存在的 OUT 复现：

```sh
python3 schema_extract/extract_yaml.py extract --source SOURCE.json --output OUT --state-root STATE --agent-binary /usr/bin/true --queue --max-output-tokens 0
python3 schema_extract/extract_yaml.py extract --source SOURCE.json --output OUT --state-root STATE --agent-binary /usr/bin/true --queue --max-output-tokens 65536
```

第一次返回 exit 2、`max_output_tokens must be a positive integer`。第二次仍返回 exit 2、`output is already registered ... with different request parameters`。实际检查发现 OUT 不存在，run 的 `state.json` 也不存在，但 supervisor 的 reservation 已处于 `registered`。因此用户纠正参数后不能在同一 state-root 复用这个从未生成的输出位置；错误提示中的 status/resume 也没有一个已创建 run 可操作。

原因是 `monitor.registered_extract` 先持久预登记 output，再调用 `runner.create_run` 做配置校验；创建前校验失败没有撤销或区分该预登记。该顺序为原有行为，本次新参数验证使它显现。定向测试将无效请求分配不同临时输出，只为了独立验证各自参数错误，不能视为该注册问题已修复。

## 变更范围

本任务只修改 Extract 执行配置/环境/身份/审计及相关测试和 `schema_extract/README.md` 参数说明，并新增本记录。未改 schema_docs、实际示例 YAML、旧 run/snapshot、主机认证或全局 settings；未提交 Git。已有的其他工作区修改保持原样。

2026-11-17：功能、定向回归、本机请求捕获和一次真实服务探针均完成。测试/探针临时目录已清理，临时诊断脚本已删除；本任务停止写入。后续新 run 的时间预算及实际阶段验收由主代理安排。
