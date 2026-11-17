# R4 B：执行器真实调用与文档收口

本轮完成，代码与文档停止写入，交后续两位新 XHIGH 独立验收。没有 GPU 调用、正式算子批量提取、真实 queue 投递、工作区 Git commit、旧 run 改写或个人配置修改。用户已明确停止完整真实流程试跑；相关状态与草稿保留，未宣称通过，也不再恢复或另建模型试跑。

## 实装与接口

- `isolation.py` 使用本次 PID namespace 的真正只读 `subset=pid` procfs。证明核 self=1、read-only 和 subset；宿主进程树与原根不可见。Claude 原生 realpath 需要 `/proc/self/fd`，Codex 配置/工具需要 `/proc/self/exe`，真实调用确认仅提供 cpuinfo/meminfo 不足。
- worker 的本次私有暂存目录支持原生临时文件与原子 rename；非许可现存条目逐项只读 bind，reviewer 整体只读。Claude Edit/Write 的实际临时文件语义已验证。R4 A 将既有采纳链收口到 `artifact_manifest`：目录、链接、非普通文件、非单链接及额外名字在读取/hash/copy 前拒绝；最终字段权限保持原阶段门禁。
- `agent.py` 增加真实 proc 证明核验，版本探测捕获 stderr；`AgentCLI.run/read_result` 调用签名保持。`audit.py` 的 preserved original 同样先过 `artifact_manifest`，新增同字节硬链在读取前拒绝的审计反例。恢复、进程 fence、会话、账目与原子发布接口沿 R3。
- 当前 Reviewer 小套件位于 `schema_extract/tools/tests/fixtures/review_eval/`；`review-eval` 明确指定 suite，不参与 run 批准。`test_review_eval` 直接使用这组当前材料。
- 活动 README、REVIEW、tools/README、EXECUTOR_ALIGNMENT、EXTRACT_SCHEMA、根导航及 issues 入口已精简为当前用法。Template README 与 DOC2 只修阶段链接和非规则状态说明；CHANGELOG 只追加本次工具事实。未改字段/格式规则或 Template 实现。

本机 rootless 新 procfs 挂载返回 EPERM，普通和 subset=pid 均如此。已验证的显式系统准备为：

```sh
sudo -n -E --preserve-env=HOME unshare --mount --mount-proc \
  setpriv --reuid 1000 --regid 100 --clear-groups --no-new-privs \
  /opt/venv/python3.13.12/bin/python ...
```

任何仓库 Python、Runner 或模型开始前已经永久降回原调用者；每次模型调用仍执行原 user/mount/PID 隔离、pivot、能力清零、seccomp 和终止协议。产品不内置 sudo、不改 sudoers、不将宿主 proc 给模型、不模拟 self/PID。具体要求写入 [运行环境](../../../schema_extract/EXECUTOR_ALIGNMENT.md#运行环境)。

## 真实 CLI

使用当前生成的完整 P1 proposal（13 个顶层字段），材料为临时标记复制任务。这验证工具/协议，不作为 schema 阶段批准。

| 调用 | 实际覆盖 | 结果 |
| --- | --- | --- |
| Claude worker | Read、原生 Edit、固定 helper、完整 schema/nonce | complete，79.9 秒 |
| Claude 同 scope resume | 精确 UUID、新 attempt/工作区、本次完整提案 | complete，55.1 秒；UUID 一致 |
| Claude 独立 reviewer | 新会话、整体只读、Read/helper、完整提案 | approve，75.3 秒 |
| Codex worker | 原生 command_execution、file_change、helper、末响应交叉验证 | complete，43.9 秒 |
| Codex 独立 reviewer | 只读会话、原生读取/helper、完整提案 | approve，42.7 秒 |

五次均有原生成功终结、退出 0、有效隔离/schema/nonce，保护文件不变，工具错误为 0。Claude 严格请求 `qwen3.8-max[1m]` / `max`；服务报告模型为 `qwen3.8-max`，未报告 effort。Codex 显式请求 `gpt-6-astra` / `high`；原生事件未报告模型/effort，保持未知。没有替换模型或裸跑。

Codex worker 的真实证据来自先验证真实 proc 的临时一致副本，采用当时的逐文件写挂载；其原生 file_change 已成功。worker 原子目录方案落地后，三次 Claude 和 Codex reviewer 使用产品接线；新目录下的 Codex worker 权限/结果协议由离线 native 用例覆盖，未追加真实 Codex worker 或 resume 调用。

会话、schema 摘要、工具计数、用量与日志路径见 [R4_B_cli_results.json](R4_B_cli_results.json)。Claude 本次原生增量与 resume 累计用量/费用分列；报告费用不是账单。Codex 保留原生计数，无法证明可加口径时不填写已知总量。

私有原始证据在 `/tmp/extract-r4-real-nhw9trua`。初次无 proc 的启动错误、真实 proc 下逐文件 bind 导致的 Edit/Write EROFS 与 blocked 提案均保留；公开摘要不复制认证、设置或私有 CLI home。

## Reviewer 正反例

当前默认 Claude/Qwen/max、每例独立只读会话、每例最多一次调用且上限 600 秒：

| 当前例 | 结果 | 耗时 |
| --- | --- | ---: |
| 正确标量复制文档 | approve，四项判断齐全，无缺陷 | 285.2 秒 |
| 调度 Thesis 将复制写成加一 | revise，定位 `DESIGN.Work_schedule.Thesis` 的全文不一致/分层问题 | 174.5 秒 |

两例均通过，见 [R4_B_review_eval_results.json](R4_B_review_eval_results.json)。初始来源说明误将正例域扩大到 NaN/Inf，而文档采用默认 finite；reviewer 正确要求修正。原报告的机械 `false_rejection` 标签和原始响应保留，材料校准后重新独立评当前两例，未改期望判断或规则来掩盖问题。

## 离线验证

测试使用临时目录、fake CLI、确定性回执和真实 namespace/Template，不执行真实模型。测试进程 PATH 中的 claude/codex blocker 均 exit 98。

- B 的 44 项执行器、材料、恢复、原子交付、审计、视图、监督和用量测试通过（195.476 秒）。其中 9 项实际 namespace/native 用例覆盖原子编辑、保护材料、动态 self/child-self、宿主 PID 不可见、stdin 重开、proc root/内存路径、mount/namespace 逃逸拒绝、double-fork 取消、精确会话与 fence。
- audit original 接 `artifact_manifest` 后，原生离线全链/归档及同字节硬链读前拒绝的增强用例通过（18.657 秒）。这些是同名用例补强，不累加为不同测试数。命令与输出见 [R4_B_checks.txt](R4_B_checks.txt)。
- 联合一致副本 `/tmp/extract-r4-integrated-vuw3zx8t` 完整 Extract suite 125 项通过（574.513 秒）。最后的 helper/绝对导航、P5 first_draft 与非法候选拒绝调整，另在最终副本补验 25 项（60.500 秒）及相关 Template API 6 项（0.966 秒），全部通过。最终生产字节、命令和范围见 [R4_A](R4_A.md)。
- 活动本地链接、Python 编译和 `git diff --check` 已核验。固定模型指令仍在 prompt 目录；A 已将真正可调用的 bare `/run/extract/check` 前置醒目，并加入 `/rules/EXTRACT_SCHEMA.md` 的明确导航，不放宽 shell。

## 已停止的完整模型试跑

| run | 最终状态 | 实际覆盖 |
| --- | --- | --- |
| `9dccdf30-5901-44ae-a1b1-ab9033e1c3f2` | budget_exhausted/P1，唯一调用 timeout=600.27 秒 | 原稿保留为合法 checkpoint；未返回提案，0 阶段批准，未发布 |
| `4b4c0ee7-ea39-43b6-9e43-d5e7c9d510b2` | 用户指示后以冻结 CLI 正式 cancel，唯一调用 cancelled=111.73 秒 | 0 阶段批准，未发布；未改原预算/身份 |

第一轮实际工具误用是模型将正确相对规则路径少算一级，以及给 `/run/extract/check` 拼 `echo` 后被拒；该次没有 StructuredOutput。第二轮在新导航和明确的新固定预算下启动，用户停止后立即取消，未继续执行。

两 run 的 termination 均 confirmed=true、survivors=[]，session 已退休，active/unconfirmed/unsettled 为空。本轮 51 次进程登记观察对应 36 个不同进程身份，复核 live=0、unknown=0。状态、调用、checkpoint 与终止证据见 [R4_B_stopped_trials.json](R4_B_stopped_trials.json)。未删除记录、隐式延长或换绑运行。

## 交独立验收

本轮没有待继续的模型调用或已知未修复的执行器故障。需要明确保留的边界：本机真实 proc 的环境前置条件；完整真实 P1–P6 模型链未通过验证；Codex 的真实覆盖为 worker/reviewer，精确 resume 的真实覆盖为 Claude。GPU、正式算子批量提取和实际账单均不在本轮结论内。后续 XHIGH 负责独立审阅当前完整改动，不应恢复用户已停止的试跑。
