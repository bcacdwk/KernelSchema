# 独立 XHIGH 审阅 B：执行器、隔离、恢复与交付

初审结论：**发现三项必须修复的 P2，当时不作最终签收**。修复后已由本审阅者独立复验闭合，最终范围和证据见 [B_RUNTIME_RECHECK.md](B_RUNTIME_RECHECK.md)。以下保留原始发现和复现，不改写初审事实。本审阅独立完成，未参与实现、未与另一审阅者互相代签。

## 1. P2：可信启动前失败无法通过深审计或归档

位置：`schema_extract/tools/audit.py:472`，相关可信判定在 `recovery.py:67–81`、`84–113`。

触发：Agent 的版本探测或其他发射前准备失败。产品正常保存 `outcome=cli_error`、`termination.confirmed=true/reason=not_started` 与绑定本 attempt 的 invocation，但没有 `launch.json/process.json/exit.json/events.jsonl`。恢复对此有明确、可核验的未启动分支，且保留此前有效会话。

问题：`_verify_attempt` 无条件经 `_Evidence.file(..., events=True)` 要求事件文件存在。这把“确证没有启动 provider”误报为证据缺失，使 `audit --verify` 标记 `incomplete`，`--archive` 拒写整个选择范围。现有 `test_setup_failure_keeps_the_previous_ready_session_and_usage_is_read_only` 只审计前次成功调用，没有审计本次失败调用。

稳定副本复现：

```sh
/opt/venv/python3.13.12/bin/python -B /tmp/extract-audit-runtime-q8a7cfpc/repro_setup_audit.py
```

结果见 `/tmp/extract-audit-runtime-q8a7cfpc/repro_setup_audit.json`：`verified_result` 正常返回 `cli_error`，`events_exist=false`，审计抛 `ExtractError: missing or changing events`。这次复现将版本探测替为确定性失败，没有启动 CLI actor 或模型。

最小修法：审计复用已有严格未启动判定，核 invocation/result/attempt 身份、无发射痕迹和已确认终止后，记录“无 provider 事件/用量”的事实；不生成空事件文件、不补零用量。已发射、身份冲突或未知的记录仍须严格拒绝。补测本次失败可深审计并归档，以及伪造 not_started、存在发射痕迹时仍拒绝。

## 2. P2：receipt_error 恢复绕过当前隔离基础门，仍可续接会话

位置：`schema_extract/tools/agent.py:174–184`、`99–110`，`recovery.py:98–123`；实际接收分支为 `runner.py:820–841`。

触发：原生调用成功、nonce 正确，但提案缺字段，因此保存为 `receipt_error`；恢复时其隔离证明已经不成立。

问题：`verified_result` 把已保存 outcome 传给 `read_result`；后者仅在 `outcome == "completed"` 时重新执行隔离、原生成功、session 等基础门。已有 `receipt_error` 保持原分类，即使重读得出 `isolation_verified=false`；`_settle_session` 仍将该会话置为 `ready`，Runner 会沿提案修复路径精确 resume。深审计能发现证明变化，但实际恢复没有相同约束。

短离线复现使用真正 namespace 中的 fake CLI：人为要求一个缺失字段，使产品自然返回 `receipt_error`；然后仅将本次临时 `isolation-proof.json` 的 `no_new_privileges` 改为 false，再调用 `recover_result`。原始结果为 `native_success=true/isolation_verified=true/nonce_valid=true`，恢复后为 `receipt_error/isolation_verified=false/session_status=ready`。没有修改真实调用证据。

```sh
sudo -n -E --preserve-env=HOME unshare --mount --mount-proc \
  setpriv --reuid 1000 --regid 100 --clear-groups --no-new-privs \
  /opt/venv/python3.13.12/bin/python -B \
  /tmp/extract-audit-runtime-q8a7cfpc/repro_receipt_recovery.py
```

结果见同目录 `repro_receipt_recovery.json`。系统 bootstrap 先永久降回 UID 1000，才运行仓库副本 Python；没有真实模型调用。

最小修法：允许续接的 `completed/receipt_error` 均重验成功基础门；隔离、原生成功或精确 session 不成立时，阻止续接并按已确认终止状态退休会话。保留 `timeout/cancelled/termination_unconfirmed` 的原语义，不能将它们改判成功。回归同时覆盖合法 receipt_error 可以 resume、证明失效不可 resume。

## 3. P2：Codex 原生用量索引没有映射隔离内的会话路径

位置：`schema_extract/tools/observation.py:73–90`，接线为 `agent.py:264–265` 与 `backends.py:212` 的 `CODEX_HOME=/home/agent/.codex`。

触发：Codex 在当前真实隔离里将 SQLite 的 `rollout_path` 记录为 `/home/agent/.codex/sessions/...`。观察器收到的是宿主本 scope 的私有 home，却直接把索引路径当宿主路径，因前缀不同抛 `native_index_identity_conflict`。

现有两次真实 Codex 的 `usage-observation.json` 均是这一错误，因此保留的原生增量观测在实际隔离接线下失效。当前报未知没有伪造数字，但这是路径解释错误，不是计数证据本身不可证明。

只读复现使用既有 `codex-reviewer-fixed` 的 SQLite、rollout 和事件流，没有启动 CLI、修改原始证据或发请求：

```sh
/opt/venv/python3.13.12/bin/python -B /tmp/extract-audit-runtime-q8a7cfpc/repro_codex_usage_mapping.py
```

结果见同目录 `repro_codex_usage_mapping.json`。仅在复现进程内把精确逻辑 sessions 路径映回当前私有 home，随后继续运行原有全部 turn、counter、stream 与元数据门禁，即得到 `root_status=exact`：input 35374、cached input 23376、output 1132。整体仍为 `partial`，因为 child coverage 未观测；此处不等于账单，也不回写历史账目。

最小修法：明确映射本 scope 已登记的逻辑 `CODEX_HOME/sessions` 到实际私有 home，并继续核精确 UUID、会话元数据、链接/路径边界、resume 基线和原生计数。不得允许任意宿主绝对路径，也不需要旧版本 alias/fallback。补测当前逻辑路径的新建和续接观测，以及逃逸/错 UUID/错 scope 拒绝。

## 4. 独立核对范围与已成立的边界

- 从基线 `27f36e5` 的变更清单和当前树核对执行器、后端、隔离、processes、runtime、recovery、audit、observation、publication、standalone、monitor、wakeup、CLI、输入/冻结、review_eval，以及关联的 Runner、Store、维护、预算和提示接线；包括未跟踪的 `backends.py/isolation.py`、新入口、prompt 和测试。旧迁移五模块及旧入口确已删除，保留的冻结执行、当前维护排空、预算、限次决定与发布屏障没有依赖这些已删模块。
- 双后端使用明确参数数组、受控认证/连接配置、私有 session home、稳定逻辑 cwd 和精确 scope/UUID；Qwen 默认及 Codex 显式选择分开，未见静默 fallback。正常完成同时核原生唯一终结、退出 0、session、schema/nonce、Codex 末响应交叉一致性及隔离证明；本报告第 2 项针对恢复的 receipt_error 分支。
- 当前隔离是非特权产品内的真实 user/mount/PID namespace、private readonly proc、pivot/detach、cap 清零、NNP 与 seccomp。worker 使用可原子编辑的私有可写目录，保护成员只读 bind；reviewer 整目录只读。源码、规则、Runner 状态不成为可写材料。已核对继承 FD、stdin 重开、proc root/PID、mount/namespace、setsid/double-fork 清理的代码和既有原生离线证据，没有把工具名单当 OS 边界。
- 候选类型/链接 preflight、阶段文件与字段门禁、当前来源身份、P6 精确集合与批准、未结调用和 control/maintenance 屏障均实际接入；交付保留原子不覆盖 rename、冲突和 crash replay。standalone 使用当前 Template 的源目录/有效文档/生成结果核验，并输出到创作目录外，未见旧 exporter fallback。
- 用量仍区分 provider 原生口径、调用增量与会话累计，按 attempt 去重，未知保留未知，报告费用不等于账单；Claude 不进入 Codex native store。第 3 项是当前 Codex 观测接线缺陷。当前 reviewer 正反例是能力回归，不是运行批准迁移。

## 5. 已复查的真实证据与明确不覆盖项

`/tmp/extract-audit-runtime-q8a7cfpc/check_existing.py` 只读重算既有结果，摘要在 `check_existing.json`：

- Claude worker、同 UUID resume、独立 reviewer，Codex 独立 reviewer，以及两项当前 review_eval，共六次原始事件/退出/schema/nonce/证明重读结果与保存记录一致。
- Codex worker 是报告已说明的真实 proc 临时实现，保留代码确有真正 `subset=pid` proc 挂载，原生 file_change/read/helper/完成响应证据成立。其早期 proof 没有当前新要求的 `procfs` 成员，不能通过当前完整 Agent 证明重放；最终原子目录接线仍依靠既有 native 离线覆盖，**不声称完成最终 Codex worker 的真实整体验证或真实 Codex resume**。
- 两次真实 P1–P6 试跑分别为 `budget_exhausted/P1` 与 `cancelled/P1`，零阶段批准、未发布；termination confirmed、survivors 空，会话 retired，active/unconfirmed/unsettled 全空。独立重查 51 次登记观察、36 个进程身份：live=0、unknown=0。
- 本机 rootless 新 procfs mount 的 EPERM 是已告知的环境前提；使用显式系统 bootstrap，再在任何仓库 Python/Runner/模型前降权。产品不含 sudo、不改 sudoers 或个人配置，不采用假 proc 或宿主 proc 绕过。
- 已阅读完整 125 项无 skip、最终 25 项及 6 项 Template API、B 域 44 项的日志与范围；没有重复整套测试。审阅只新增上述三个必要短复现，其中一个使用真实 namespace 的离线 fake CLI；没有模型、GPU、queue 消息、重启/恢复真实链或 commit。

审阅稳定副本：`/tmp/extract-audit-runtime-q8a7cfpc/repo`；200 个复制文件的摘要在 `snapshot.json`，收口时与仓库对应文件逐字节一致（`snapshot_recheck.json`）。`git diff --check` 通过。只写本报告及 `/tmp` 复现证据。等待新 MAX 修复这三项后，由本审阅者复验；完整真实模型链已被用户停止，**不是待补跑验收项**。
