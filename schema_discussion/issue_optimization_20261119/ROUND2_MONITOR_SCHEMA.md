# 第二轮实现：monitor 与 Schema 表达

2026-11-19，北京时间。实现代理：`/root/issue_r2_monitor_schema_xhigh`，`gpt-6-astra / xhigh`。依据 [第一轮审计](ROUND1_MONITOR_SCHEMA.md) 完成分配的 10 个稳定 ID；正式 issue 状态由第四轮据最终结果更新。

本轮修正采样提交顺序、前台等待反馈和人读时区，补齐非 Tensor 与存储名称说明，并限域修订 MLA/GEMM root。原有 normalizer、重名诊断、Version 权限提示及双角色 I/O 规则保留，没有扩展类型系统、恢复队列、公共 CLI 参数或硬件调查。

## 逐项落实

| 稳定 ID | 本轮落实与边界 |
| --- | --- |
| `monitor-registration-lock-retry` | `Clock(on_wait=...)` 转发 ack、scan_claim、scan_commit、scan_cleanup 四阶段的锁外等待。运行组实现 Store 可选回调及 CLI 节流 stderr。保留只重试进入锁、不重放正文和业务拒绝的规则；真实 SIGINT 后同 wake 可核对续接。 |
| `monitor-wait-stale-observation` | 初次、绑定及 daemon scan 在锁外保存观察完成时刻；逐 run 的真实 `observation.observed_at` 用于 latest 新旧比较、unchanged_since 与停滞判据，完成时间单独决定下一次提醒。较旧行不修改 latest/changes，并在 `superseded_run_ids` 明示。claim 仅按所属 PID 的 boot/start 身份重新观察；活 owner 和无法确认的 owner 均不抢占。 |
| `monitor-beijing-time-display` | 提醒消息及 CLI 人读观察用固定 UTC+8 formatter。monitor JSON 只在副本中增白名单 `_display`；epoch、None、ID、间隔与调度保持原意，停用后的到期显示标明保留记录。 |
| `normalizer-diff-not-visible` | 保留既有首差异 Unicode 位置、局部上下文与省略号实现。未改转换规则，也未扩为完整 diff。相关 Template 回归通过。 |
| `inplace-output-input-name-diagnostic` | 保留冲突双方路径及条件式双角色提示；统一命名空间、alias 证据要求不变。相关 Template 回归通过。 |
| `maintenance-version-bump-outside-field-cap` | 保留两处既有 worker 递增 Version、Runner 校验的提示。没有新增 Version 写入口或扩大 scope；运行组负责相邻权限回归。 |
| `tuple-abi-input-typing` | 不新增 tuple 类型机制。MLA 的 `sage_attn_sfs` 改为整体 tuple 的 `shape: []`；相邻整数 tuple 保留元素轴及 dtype，仅澄清 meaning。原 decline 所针对的新增机制不因实例修正而被改写。 |
| `inplace-output-input-registration` | 保留现有 Input/Output 双角色义务与已修实例，不重复添加同一规则。FMHA/Sparse 的完整语义复核留给指定专项轮。 |
| `non-tensor-parameter-registration` | DOC1 用三个现有字段样例区分逻辑元素、整体对象、torch.dtype；明确 None 成员、整个对象 nullable 与省略参数 default 各自职责，不从默认值或未使用推定固定 range。 |
| `location-storage-level-registration` | DOC2、DOC5、P1 对齐：所属执行单元摘要登记材料明确的存储名及范围，Storage.location 沿用；纯存储不必新增 Execution 键，不成为 worker/role。GEMM 既有 thread 摘要补 thread-local register；不改 wide、不新增容量或无 spill 保证。 |

## 观察与显示实现

`monitor.record_scan` 给 latest 的每个 run 保存 `observed_at`，比较时不把这个采样元数据算成状态变化。最新采样仍可反映同 revision 下的 Runner 存活/写锁变化；分批观察互不覆盖未采样的 run。顶层观察时间不倒退，change 事件按对应 run 的采样时间登记。旧样本仍可作为这次扫描结果返回，但显式列入 `superseded_run_ids`；CLI 默认文本也提示这些 ID。

锁等待不重新调用 `inspect_runs`，也不在控制锁内扫描整个 `/proc`。绑定扫描的 `completed_at` 使用成功提交时刻，`next_due_at` 仍为完成时刻加 300/1800 秒；queue 的 120/1800 秒重送条件和次数没有改动。中断不新增可能再次无限等锁的清理，原 claim 保留到其 owner 真正退出后由同 wake 回收。

共享接口按计划交付：

```python
Store.control_transaction(*, on_wait=None)  # 运行组单一写者；elapsed_seconds
Clock(store, repo, *, now=None, sender=None, on_wait=None)  # phase, elapsed_seconds
format_beijing_time(timestamp) -> str
clock_display(result: dict) -> dict
```

`clock_display` 深拷贝后只补以下已有绝对时间：根层与 `result.clock` 的 `observed_at/next_due_at`；其 `check/last_scan` 中 `started_at/due_at/ack_at/observed_at/completed_at`；`reminder` 中 `due_at/next_attempt_at/first_queued_at/last_queued_at` 与 `ack.at`；根层 `ack.at`；`daemon.heartbeat_at`。缺字段不制造原始值，None 显示“未记录”。不遍历任意 `_at`、run 证据、历史、PID 身份或 duration/age。未设置进程 TZ，未把 epoch 加八小时回存。

本组只编辑 `wakeup.py`、`monitor.py` 的 `record_scan`、`test_wakeup_retry.py`、DOC1/DOC2/DOC5、P1 与 wakeup prompt，以及两份 root 和本轮证据。`store.py`、`cli.py`、两个 Extract README 由运行组统一写入；monitor 恢复控制段未动。既有 normalizer/symbols/Version prompt 不再修改。

## 正式实例修订与证据

修稿前，逐项验证原 receipt 的 manifest 与正式目录字节相符，保存原 root 与 receipt；使用 `git show 24d2b3d3ca9d4907c4e7efd9607cf78cfb1a52b2:<path>` 与当前依据文件逐字比较。MLA 的 5 个、GEMM 的 6 个依据文件均与固定 commit 相同。完整摘要见下列 before 记录。

| 实例 | 限域变化 | 人工修订记录 | before / after |
| --- | --- | --- | --- |
| MLA | root Version 4→5；sage_attn_sfs 的 shape 与 meaning；num_elts_per_sage_attn_blk 的 meaning | [issue-review-20261119.json](../../schema_wiki/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/.extraction/issue-review-20261119.json) | [before](instance_revisions/mla_expanded_fp8/before.json) / [after](instance_revisions/mla_expanded_fp8/after.json) |
| GEMM | root Version 3→4；Execution.thread 补 register 名称及 thread-local 范围 | [issue-review-20261119.json](../../schema_wiki/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/.extraction/issue-review-20261119.json) | [before](instance_revisions/fp8_scaled_grouped_gemm/before.json) / [after](instance_revisions/fp8_scaled_grouped_gemm/after.json) |

原 root/receipt 字节分别保存在以上实例目录的 `before/`。原 `.extraction/receipt.json` 原样保留；原 P6 只批准原 manifest 字节。本记录是作者实施的人工限域修订，不是新 P6，也不是对新字节的独立终审。第一轮 max 独立判断的是修订方案；后续指定 high/xhigh 轮负责审阅当前新字节。

- MLA root：`4fec3eed4d4f2a2d7cf39aaa3b537e4b8485cecf10387c652b2a89acb01d583c` → `367704f500ce6078cc488b6e0ee809ddf61a4b862024e549f156557a01d061d8`；原 receipt SHA-256 为 `86378717dbe8c118243a0ea9307751ad26e32117b448e5be9979d1c38d305612`。
- GEMM root：`4e28944a053c0338038e828db46034fd032a73b6fb918c221ced20d3095a60ba` → `cba4ea9d28e94f3541ed0a60e255ff9b188bcaa5635d897d68475b4d42c3f92d`；原 receipt SHA-256 为 `80b4d4eef02cd9bddfed0c50507c58339df31f5d6a669da0383ef6ace2020319`。wide 与 dispatch 字节不变。

来源/语义核对由本轮作者实际完成：MLA 的候选 forward、reference forward、normalized/returned 及输入构造、域/metadata 对齐，AST 核对两个默认 tuple，确认未读取成员且没有固定 range 声明。当前字段表达一个整体 tuple 与四个整数元素；default 值保持原样，未加 nullable/range。GEMM 核对 root 的 `fin/sb_c/sb_n` 与 wide 的 `fin/sa_c/sa_n/sb_c/sb_n`，及两节点 Storage.location 的所属范围；局部数组和明确 register prefetch 意图不被解释为编译后无 spill。

深层结构差异断言确认只变更上表列出的 4/2 个字段路径，两个 root Version 均仅增加 1，其他交付文件及原 receipt 均逐字未变。新 manifest 和修订记录入口交根代理用于当前 wiki/Supervisor 索引；历史 batch 索引、运行证据、旧 P6、checkpoint 与调用数没有重写。

## 实际验证

离线环境统一使用 `PATH=/opt/venv/python3.13.12/bin:/usr/bin:/bin`、`PYTHONDONTWRITEBYTECODE=1`；未调用真实 Extract/Resume/Monitor CLI、模型或 GPU。

| 命令/核对 | 结果 |
| --- | --- |
| `python3 -B -m unittest schema_extract.tools.tests.test_wakeup_retry schema_extract.tools.tests.test_monitor_evidence -q` | 最终 **27 tests / 2.358s / OK**；[日志](ROUND2_MONITOR_TESTS.log)。 |
| `PYTHONPATH=schema_template python3 -B -m unittest tools.tests.test_yaml tools.tests.test_normalization tools.tests.test_schema -q` | **76 tests / 20.630s / OK**；包含保留的 normalizer、符号、nullable 与普通硬件/只读检查边界。 |
| `python3 -B schema_template/check_yaml.py schema_wiki/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/root.yaml --read-only --skip-hardware` | 最终退出 **0**，全目录 1 YAML、dispatch 通过、规范化预览 0 改动；[日志](instance_revisions/mla_expanded_fp8/check.log)。首次新增 meaning 超预算后已压缩，未改预算或 checker。 |
| `python3 -B schema_template/check_yaml.py schema_wiki/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/root.yaml --read-only --skip-hardware` | 退出 **0**，全目录 2 YAML、dispatch 通过、规范化预览 0 改动；[日志](instance_revisions/fp8_scaled_grouped_gemm/check.log)。 |
| DOC1 三条样例嵌入最小完整 Algorithm，经 `tools.rules.validate_algorithm` | 无诊断；没有增加逐句镜像测试或通用 Python 类型解析器。 |
| 固定源码字节、AST 默认值/未读取路径、正式 root 深层差异、前后 manifest/receipt | 全部断言通过；具体依据与范围记录于每算子人工修订记录。 |
| `git diff --check`（本组代码/规则/prompt 范围） | 退出 **0**。 |

监控回归的实质覆盖：旧采样等提交锁时插入更新采样，旧扫描检查仅调用一次，采样与完成相差 120 秒但最新投影不回退；同 revision 存活变化、分批 run 时间、只变采样时间不增加 changes、停滞提示不借提交等待提前出现。真实临时 flock 与子进程 SIGINT 覆盖 ack 等锁不改 ledger、scan 等提交锁保留 ack/claim、owner 退出后同 wake 一次完成，以及提交后输出前中断后的 reused 幂等。既有 stop/代际/重送和错误清理测试继续通过。

显示回归覆盖 UTC 日界与年界、None、原 epoch/提醒 ID、输出深拷贝与白名单范围。Store/CLI 的锁外回调与 stderr 节流由运行组负责相应组合回归；本组已经核对集成位置，并要求等待提示不未经对账宣称整个 wake“尚未提交”。

## 后续范围

本组实现无待补项。根代理统一执行最终全量离线检查；第三轮精简、第四轮复核归档及当前索引更新按既定职责继续。尚未对新字节完成独立终审，人工修订记录明确保留此边界。FMHA/Sparse 专项、原 continuation_audit 的待审计事实不由本组代替完成。真实 `.schema_extract/v2`、停用 monitor、失败 FMHA 原运行和取消的 Sparse successor 均未操作；未改产品版本，未 commit/push。
