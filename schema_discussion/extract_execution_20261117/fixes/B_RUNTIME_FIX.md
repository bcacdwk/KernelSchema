# MAX 修复 B：执行器三项与审计名单接线

三项修复及 A-F1 的 `audit.py` 接线已完成，代码停止写入，交原 XHIGH 独立复验。本轮只用离线 fake CLI、临时 namespace 和现存日志，没有启动模型、完整阶段链、GPU 或 queue 调用，没有 commit、修改旧 run 或个人配置。

## B-F1：可信启动前失败可深审计与归档

`recovery.never_launched` 成为恢复与审计共用的未启动判定：本 attempt 的 invocation/result 绑定、明确 setup error、无退出码、终止确认且无 survivor、无 launch/process/exit/events 痕迹。路径存在性使用 `lexists`，断开的事件链接也不能冒充不存在。`verified_result` 同时复核保存结果的 run/attempt、scope、私有会话目录、executable/version 等 invocation 身份。

审计仅在此判定成立时接受 provider 流确实不存在，并保留 `not_started; no provider stream; usage unknown`。它继续核原始 invocation/result/schema/prompt/终止证据，未创建空 `events.jsonl`、零用量或退出记录。公开 `audit_run(..., verify=True, archive=...)` 对此返回 `partial` 且成功归档；`partial` 说明没有 provider stream/usage，不是证据错误。

新增反例覆盖发射痕迹、断链、错误 invocation/结果 scope、缺 setup error、未确认终止及矛盾 survivor。先前有效会话保留的既有用例继续通过。

## B-F2：格式修复也必须重验执行基础

Agent 的 `read_result` 对 `completed` 与 `receipt_error` 都重新从原始文件核隔离证明、实际退出码、唯一原生成功终结及精确 session。恢复与审计继续复用此入口，没有依赖缓存的 `isolation_verified` 或另写一套成功判定。

真正 namespace 中的 fake CLI 自然产生缺字段但 nonce 正确的 `receipt_error`；合法恢复与同 UUID 续接通过。随后只改临时隔离证明的 `no_new_privileges=false`，恢复结果变为 `isolation_error`，私有 session 退休且 UUID 清空，再次续接被拒绝；审计也拒绝已失效的原结果。原生失败、错 session、错误 nonce/schema 分别核对，timeout/cancelled/termination_unconfirmed 不因完整日志而恢复成功。

## B-F3：当前私有 Codex home 的精确映射

`begin_usage_observation` 必须显式收到实际 private home 和本次环境中的 logical home；它核 attempt 摘要、run/attempt、session scope/directory，实际目录只能是登记的 `session_directory/home/.codex`，logical home 只能是后端的 `/home/agent/.codex`。Agent 传未改写的 attempt，逻辑 cwd 取 invocation 的实际登记值。

索引中的 `/home/agent/.codex/sessions/...` 精确映射至本 scope 的 private home；原本位于同一 private sessions 目录的路径仍可读取。其他宿主路径、`..`、错 UUID、错 scope、符号链接和元数据冲突均拒绝。原有 resume 前采样、turn、累计差额、重复计数和 stream 交叉核验保持；没有通用 alias/path 框架，也没有宿主个人 home 回退。

仅只读重算既有 `codex-reviewer-fixed` 原始 SQLite、rollout 与 events，得到 root `exact`、整体 `partial`，input=35374、cached input=23376、output=1132。缓存输入属于输入子集；child coverage、整调用总量及账单仍未知。七份所读原始证据的 SHA-256 前后相同；历史 `usage-observation.json` 保留原来的 unavailable，不回写账目。

证据：[只读脚本](/tmp/extract-runtime-fix-y56oeowt/repro_usage_current.py)、[当前重算结果](/tmp/extract-runtime-fix-y56oeowt/after_codex_usage_mapping.json)。历史 usage SHA-256 为 `447d979b2ea5a68d407a03016433e72fe187d5310eba391afd5e0da0f336d1c6`。

## A-F1 的共享审计接线

原稿 hash/read/archive 前，审计复用 A 的 `receipts._artifact_baseline` 与 `artifact_manifest(expected=..., additions=...)`。普通调用名单来自 attempt 的固定 dependency digests；P2 worker 非 select 仅可增加 `dispatch.md`；adjudicate/select 必须匹配登记候选并采用其 `files`。不把 validation 自报的原稿名单作为放行依据；选定只读候选的字节也按其实际输入比较。

只运行一条 P1 fake CLI 调用验证公开审计/归档；增加普通 `unexpected.yaml` 后，对全部原稿 `read_bytes` 设置拒读断言，审计先报名单错误且不写归档。未运行 P1–P6 链。

## 验证与复查

原三个短复现首先按审阅报告重放，失败与报告一致：[setup](/tmp/extract-runtime-fix-y56oeowt/before_setup_audit.json)、[receipt recovery](/tmp/extract-runtime-fix-y56oeowt/before_receipt_recovery.json)、[usage mapping](/tmp/extract-runtime-fix-y56oeowt/before_codex_usage_mapping.json)。

本轮共覆盖 13 个不同局部用例：用量 5 项、未启动审计 1 项、原生恢复/精确续接/完成基础 6 项，以及单调用公开审计 1 项。首次测试有两处夹具错误（误要求结果保存 private home 字段；单调用夹具缺显式 retry_limit），已修正并局部补验；原始失败日志保留，不把它们记成产品通过。最终补验：[未启动审计 1 项通过](/tmp/extract-runtime-fix-y56oeowt/final_setup_tests.txt)、[双后端恢复与公开审计 2 项通过](/tmp/extract-runtime-fix-y56oeowt/closed_native_tests.txt)；其他五项原生用例的通过结果在 [首轮日志](/tmp/extract-runtime-fix-y56oeowt/native_tests.txt)。用量模块 5 项通过，0.113 秒。

稳定副本为 `/tmp/extract-runtime-fix-y56oeowt/repo`；[完整复制清单](/tmp/extract-runtime-fix-y56oeowt/snapshot.json) 与 [B 最终 9 个代码/测试文件摘要](/tmp/extract-runtime-fix-y56oeowt/runtime_snapshot.json) 已记录。九份文件的 AST 解析、与工作区字节核对和 `git diff --check` 通过；没有重跑 125 项整套测试。

以下命令可复查相应局部用例。`blockers/claude` 与 `blockers/codex` 均固定 exit 98；真实 namespace 测试直接使用临时 fake executable：

```sh
cd /tmp/extract-runtime-fix-y56oeowt/repo
PATH=/tmp/extract-runtime-fix-y56oeowt/blockers:/usr/bin:/bin \
  /opt/venv/python3.13.12/bin/python -B -m unittest \
  schema_extract.tools.tests.test_usage_observation \
  schema_extract.tools.tests.test_current_recovery.CurrentRecoveryTests.test_setup_failure_audit_preserves_missing_stream_and_rejects_forged_proof -v

sudo -n -E --preserve-env=HOME unshare --mount --mount-proc \
  setpriv --reuid 1000 --regid 100 --clear-groups --no-new-privs \
  /usr/bin/env PATH=/tmp/extract-runtime-fix-y56oeowt/blockers:/usr/bin:/bin \
  /opt/venv/python3.13.12/bin/python -B -m unittest \
  schema_extract.tools.tests.test_current_recovery.CurrentRecoveryTests.test_native_result_recovery_requires_real_exit_nonce_schema_and_isolation \
  schema_extract.tools.tests.test_current_recovery.CurrentRecoveryTests.test_receipt_repair_resumes_only_while_native_execution_remains_verified \
  schema_extract.tools.tests.test_current_delivery.CurrentDeliveryTests.test_one_call_audit_archives_setup_failure_and_preflights_original_names -v

/opt/venv/python3.13.12/bin/python -B /tmp/extract-runtime-fix-y56oeowt/repro_usage_current.py
```

本机仍需显式外部 bootstrap 挂真实 proc，且在运行任何仓库代码前永久降回 UID 1000/GID 100；产品没有 sudo、假 proc 或宿主整 proc 绕过。没有改变私有可写暂存目录和其他材料只读的隔离设计。

两次已停止的真实全链仍是零阶段批准、未发布；36 个不同登记进程身份 live=0/unknown=0 的既有证据未改。本次没有增加真实 Codex worker/resume 或完整语义链验证，也不把已取消的全链视为待补跑项。待原 XHIGH 对最终源代码独立复验。
