# R6 B：命令、审计与监控精简

先读本目录约束及 R2 B、R3 A、R5 A/B 结论，检查限定的七个模块；先报告等价依据，再修改。以下差异均相对只读快照 `/tmp/schema-extract-pre-prune-ego_8hhb`，不把前轮修复算作本轮成果。

## 实际删改及等价理由

- `cli.inspect_runs` 没有独立行为，`repo` 形参未使用；删除该转发，由 CLI 直接调用现有 `monitor.inspect_runs`。`registry` 只处理一个固定根目录，删除单项外层循环，原目录、ID、证据和决定核验保持。
- `cli._frozen_entry` 的全部调用都传入 Store 和 state；删除无调用者的 `frozen_invocation` 备选分支、`lock` 参数及对应闲置读文件。仍在进入时即时 `store.load`，经 `frozen_run_invocation` 核对当前运行和冻结输入，切换到该运行目录，必要时 exec 冻结入口。控制登记或启动预留之前已有的 fingerprint 检查保留，没有将不同时间点的核验合成一次。
- `monitor._alive` 只是 `process_alive` 的转发；四个调用直接使用原函数，同一进程表和 True/False/None 语义保持。
- `audit._verify_attempt` 只有一个调用者，始终传真实 Store；Store 参数仅用于恒真开关。删除参数和开关，原生结果、后续 termination 证明、模型/effort、schema 及 Codex 原稿核验仍全部执行。
- `audit._checks` 的两处 `last_reminder` 没有当前写入者；现行 clock 按 kind 写 `last_check`/`last_probe`，同时写 `history`，完成扫描另写 `last_scan`。删除两个空候选项，保留有界历史、在途提醒、probe、最新扫描及迟到消息 ID。
- `prompts.prepare_attempt_materials` 删除未读取的 `scope` 局部计算；材料集合、导航、正文与动态上下文不变。

四个产品文件合计净减 17 行。没有增加抽象、迁移、兼容入口或新状态。`monitor.restart_evidence` 有 Runner 等跨模块消费者，本轮保留；没有为消灭一个转发扩大修改范围。

## 保留能力

默认 Claude Code `qwen3.8-max[1m]` 与双角色 max、显式 Codex、独立 reviewer、review-eval、standalone、当前维护及 issue 确认材料均保留。固定语义提示继续在 Markdown；机械 schema、动态事实和错误诊断继续在 Python。正式指南导航和新上下文必要指引未删。

未 accepted 的拒绝/repair 回执仍只能形成 partial、evidence-only 审计；accepted 仍必须有匹配 validation。后台 launching 的真实早退仍标失败，已领取/完成或不同 launch 不被降级。派发、消费恢复授权、终止证明及预算检查的各自时点保留；pause/cancel/hold/维护和有限重试权限不变。clock 的发送、ack、提交 scan 和自动恢复授权仍分开，未知用量不填零、会话累计值不相加。

## 验证

使用 `PATH=/usr/bin:/bin /opt/venv/python3.13.12/bin/python3 -B`，执行前断言真实 claude/codex 不在 PATH；原生调用仅使用绝对路径 fake CLI。

首批以下既有用例共 **7 项通过，44.129 秒**：

1. `CLIRecoveryTests.test_resume_adds_finite_credit_after_expiry_without_refunding_calls`：实际冻结入口 exec、EX_TEMPFAIL、预算追加不退款及只读审计。
2. `test_monitor_evidence` 全部 4 项：并行失败的证据归属、真实 `/usr/bin/false` 子进程早退、失败启动不复用活 launcher。
3. `AuditReceiptRecoveryTests`：拒绝和 receipt-repair 可保留部分证据，accepted 缺 validation 仍拒绝归档，state 字节不变。
4. `PromptMaterialsTests.test_p6_receives_worker_facts_after_final_content_and_only_real_documents_are_provided`：P2/P4 worker 事实、冻结正式指南、当前材料与阅读次序。

测试临时根 `/tmp/schema-extract-r6-b-checks-4ok0dr2y` 关联进程剩余 **0**，临时目录已清理。

删除 `last_reminder` 空候选后，仅复用既有 `WakeupRetryTests.test_ack_scan_and_stop_win_before_send_and_before_late_receipts`，并读取其 scan 现场调用 `_checks`：**通过，0.346 秒**；两个消息 ID、ack 线程和两次 queued 的时间记录均保留。该检查采用现成 fake clock/fake sender，真实 `_send` 被设为直接失败；没有启动后台 clock 或投递 queue。未重跑前批。

共执行 **8 项既有用例**，另有同一提醒现场的审计输出核对。限定模块 `ast.parse`、限定 `git diff --check` 通过。没有新增或删除测试文件，没有真实模型、GPU、完整提取、发布或 Git commit。

## 文件与未决项

- `schema_extract/tools/cli.py`
- `schema_extract/tools/audit.py`
- `schema_extract/tools/monitor.py`
- `schema_extract/tools/prompts.py`
- 本报告

`wakeup.py`、`review_eval.py`、`common.py`、产品说明及固定 prompt 均未修改。没有本轮已确认而未处理的问题；检查限于上述离线行为，不代表真实模型审阅质量或完整流程已验证。未改 Schema 字段、Template、案例、Kernel、Wiki、archive、历史报告或精简前快照。报告完成后停写。
