# 独立 XHIGH 审阅 B：最终复验

**B-F1、B-F2、B-F3 均已闭合，执行器审阅范围独立签收。** 已核最终生产差异、重放本人原来的三个反例，并补验必要正反例和共享产物名单接线；未发现新的实质遗漏。这不代替另一审阅者的流程签收，也不扩大真实模型验证范围。

1. **可信未启动记录可审计、可归档。** 原 `repro_setup_audit.py` 在最终副本输出 `audit=accepted`，仍为 `cli_error/not_started`、没有 `events.jsonl`。恢复与审计共用 `never_launched`，核本 attempt 的 invocation、明确准备错误、终止确认、空 survivors 和无发射痕迹。独立局测确认断链、发射痕迹、错身份、缺错误记录、矛盾终止证据均拒绝；公开 audit/archive 正例通过，未补空事件或零用量。

2. **receipt_error 续接必须保有执行可信性。** 原 namespace 反例现在得到 `isolation_error/isolation_verified=false/session retired`。`read_result` 对 `completed/receipt_error` 共用原生成功、退出码、隔离、精确 session 基础门；超时、取消、终止未确认不会被重解释为成功。独立原生离线局测确认合法 receipt_error 能按同 UUID 续接，证明失效后 UUID 清空、再次续接拒绝，深审计同样拒绝失效的原结果。

3. **Codex 私有会话路径正确映射。** 最终代码只将已登记 logical home 的 sessions 路径映射至本 scope 的 private home，并继续核 attempt 摘要、scope/directory、UUID、链接、元数据、resume 基线、turn、累计差额和 stream。本人直接读取既有 `codex-reviewer-fixed`，不再替换 locator，得到 root `exact`、整体 `partial`：input=35374、cached input=23376、output=1132；child coverage 与账单仍未知。七份原始证据摘要前后相同，历史 `usage-observation.json` 仍为 `unavailable`，没有回写。独立局测覆盖新会话、resume、其他 scope/home、错 UUID、`..` 与链接拒绝。

4. **共享名单门禁和正常交付目录没有混用。** 已核最终 `artifact_inventory` 先枚举完整名字/类型/单链接、随后 `artifact_manifest` 才读取内容；`_yaml_files` 已改为无内容读取的 inventory。审计从 `_artifact_baseline` 取得登记的 dependency/candidate 文件集合，不使用 validation 自报名单扩大许可；P2 worker 仅有明确的 `dispatch.md` 增量。本人运行的公开 audit/archive 用例证明额外 YAML 在整个 original 目录读取前被拒绝。补充目录形状检查证明含 `.extraction/receipt.json` 的交付目录可正常生成 standalone，并在 `validated_directory_files` 中保留回执；导出侧真实 `manifest/verify_manifest` 与不覆盖原子 rename 可处理此目录，再生成 standalone 同样成功。作者 workspace 的同名 `.extraction` 在拒读观察器下先被拒绝。publication 的目标侧仍使用包含交付回执的完整 manifest；audit 读取 run 原稿/checkpoint/账目，不把交付目标当作者目录枚举。本项未重跑完整阶段批准链。

5. **复验方式与证据。** 独立稳定副本为 `/tmp/extract-runtime-recheck-zlxlxhyt/repo`；A、B 全部停写后，`snapshot_recheck.json` 核其 200 份复制文件与当前树逐字节一致。三个反例结果为 `setup_replayed.json`、`receipt_replayed.json`、`usage_replayed.json`。另独立执行 5 项局部用例：`local_recheck.log` 的 3 项通过（0.157 秒），`native_recheck.log` 的 2 项通过（5.072 秒）；交付目录补查为 `check_delivery_marker.py`、`delivery_marker.json`。真实 namespace 的离线 fake CLI 始终使用显式系统 proc bootstrap，并在任何仓库 Python 之前永久降至 UID 1000/GID 100、设 NNP；PATH 中 claude/codex 均为 exit 98 阻断脚本。未运行新模型、真实全链、GPU、queue、整套 125 项测试或 commit。`git diff --check` 通过。

仍保留初审明确的边界：本机需要外部显式真实 proc bootstrap，产品不加入 sudo、假 proc 或宿主 proc 绕过；真实 Codex worker 是先前临时实现的局部证据，最终原子目录由离线 native 覆盖，真实精确 resume 仅验证 Claude；两次真实全链已停止、零阶段批准、未发布，完整真实 P1–P6 不作通过结论，也不是待补跑任务。B 域没有剩余必须修复项，现停止写入。
