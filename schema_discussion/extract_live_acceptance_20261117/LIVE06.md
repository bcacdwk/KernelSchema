# LIVE-06：无效参数提前占用输出登记

## 只读确认

`monitor.registered_extract` 在 `supervisor` 事务中写入 `starts[run_id]`，其中 `status=registered`、`request_key` 包含本次原始配置；事务结束后才调用 `runner.create_run`。`create_run` 此时才校验配置。由此，`--max-output-tokens 0` 等确定无效的参数在输出目录和 `state.json` 都不存在时，已经保留了输出路径；改为合法参数重试会被 `different request parameters` 拒绝。

此问题不限于新增输出预算：未知配置字段、无效后端/模型/effort、非正时间或次数预算、未满足最少调用预留、Codex 缺少显式配置，都走同一先登记后拒绝的顺序。不是已冻结输入的合理冲突。

2026-11-17 只读检查实际验收 store `.schema_extract/live_acceptance_20261117`：QSA `7c31f6e1-e912-43d2-ada3-7b181f8245e5` 与 FA4 `8794f043-7c53-4991-85bf-886afd417fed` 均为 `starts.status=created`，有 `state.json` 而尚无交付 output；这是正常未发布状态，不属于幽灵登记。本轮不修改该 store。LIVE05 所述非法值复现位于其临时验证 store。

## 现有边界与最小方案

当前新建顺序为：配置校验及默认值 → 定位可执行文件 → 检查 output/repo/store 布局 → 解析并校验来源绑定 → 检查来源/store 布局 → 初始化 store → 冻结 source/rules/config → 创建 run state。登记写在这串动作之前。

正常重复请求通过原始 `request_key` 认出既有 run，直接读取其固定 state；不应要求已经改变或移走的现时源码重新通过校验。原始请求参数不同仍须拒绝，不得用空 output 作为释放真实 run 身份的依据。真正进入冻结/创建后的故障仍保留登记，支持同参数复用 run ID 与并发排他；本轮不做任意异常回滚或历史登记迁移。

已把 `create_run` 已有的完整配置规范化、校验和可执行文件 `which/resolve` 集中为 `runner.normalize_config`，继续复用 `backends.validate_max_output_tokens`。新登记分支先调用该 helper，成功后才写 `starts`；`create_run` 使用同一 helper，避免重复定义配置规则。仅校验新登记分支，保留已有身份的冲突优先级与重用行为；不在 supervisor 短事务里引入 Git、来源遍历或快照工作。

产品只修改 `runner.py`、`monitor.py`，回归修改既有 `test_cli_recovery.py`。保持原始请求摘要不变，不对已有登记重做配置/来源校验，不新增字段、兼容分支、回滚或清理系统。来源描述、目录布局、冻结时文件变化与其他 IO 错误仍按原有流程处理；本次修复范围是新运行的配置拒绝提前落入持久登记。

## 定向验证

收到主代理确认 LIVE05 全部停写后，先修改现有输出预算测试，使全部非法请求使用同一 `budget-delivery` 路径，并断言拒绝后没有新增 `starts` 或 run 目录。在产品改动前运行该测试：**1 test，7.024 秒，预期失败**；非法 `max_output_tokens=0` 后 ledger 多出 `8b3f7273-7c9f-4e0c-8923-2476adf5bfa8`，直接确认旧行为的幽灵占位。所有材料在自动清理的临时目录中。

修复后运行：

```text
/opt/venv/python3.13.12/bin/python3 -B -m unittest \
  schema_extract.tools.tests.test_cli_recovery.CLIRecoveryTests.test_output_budget_is_frozen_passed_to_cli_and_rejected_for_codex \
  schema_extract.tools.tests.test_cli_recovery.CLIRecoveryTests.test_repeated_commands_report_saved_state_and_frozen_source -v
```

结果：**2 tests，49.245 秒，OK**。第一项在同一路径依次拒绝输出预算 0、原有 `max_calls=3`、Codex 不支持的输出预算，均无新增登记/run 目录；随后 Claude 65536 正常创建并冻结，再改 128000 仍拒绝真实已登记身份冲突。既有后续断言继续检查 65536 进入 stub 原生环境、attempt/invocation/result 与审计。第二项确认现时源码已改变时，前台/后台/排队的重复请求仍复用原 run/config/fingerprint，不启动额外调用；改变 effort 仍拒绝。

`git diff --check` 通过。本轮未新增测试文件、未重复 LIVE05 全量回归、未调用真实模型或 GPU、未改真实 r01 的 snapshot/run/session，也未提交 Git。

2026-11-17：修复与定向验证完成，本任务停止写入。
