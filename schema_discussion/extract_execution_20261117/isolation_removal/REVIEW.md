# 移除 Extract 自建隔离：独立审阅

审阅者：全新 XHIGH，独立于实施者。最终结论：独立签收。本次发现的一项进程结算竞态已由实施者修复，停写后独立复验通过；当前审阅范围内没有未修复的实质问题。

## 范围与方法

- 以 `/tmp/extract-isolation-removal-baseline-49sshq5c` 的 87 个文件为本次基线，逐项核 SHA；初始 `DELTA_MANIFEST.json` 的 36 个文件与实际增量一致，没有把 HEAD 之后上一轮的未提交重构当成本次改动。
- 读取本次 BRIEF、IMPLEMENTATION、差异清单和真实调用摘要，再独立核 agent/backends/processes、Runner/recovery/audit/observation、材料/helper/冻结执行器以及活动文档。
- 只读复核既有三次真实 CLI 的原始证据摘要和原生日志；本审阅不调用模型，不运行真实 P1–P6、GPU、queue 或 sudo，不改产品、测试、规则和用户配置，不提交 Git。

## 发现的问题与修复复验

初审发现进程表采样竞态：`terminate_owned` 在扫描只见 gate 时可 SIGKILL gate；agent/recovery 又把该强杀动作视为后代结清的替代证据。若子进程在 `ps` 枚举之后、读取其 `/proc/<pid>/stat` 之前 fork 并退出，新后代会落在本次采样之外。杀掉 gate 后，该后代失去可观察的归属锚点，旧逻辑仍返回 `confirmed=true`。

独立复现脚本 `/tmp/extract-gate-race-review-default-grace.py` 使用真实 gate、真实 fork/setsid/父退出和默认 `grace=1.0`；只控制采样间的调度，不伪造 `ps` 或 `/proc` 内容。1.27 秒复现得到 `confirmed=true`、`survivors=[]`、gate 被 SIGKILL、`gate_reaped=false`，同时仍有一个真实后代存活。现场为 `/tmp/extract-gate-race-review-g4746z7b`；复现后全部测试进程已清理并 reap。

要求的修法：后代由 gate 实际 wait/reap 结清；不以扫描为空、主 PID 消失、曾强杀 gate 或旧 `confirmed` 字段代替闭合记录。正常运行、恢复、无 process 记录分支和审计使用一致的核验，缺少闭合证据保持 fence。

已逐行核修复差异：`terminate_owned` 不再向 gate 发信号，有界 TERM/KILL 阶段继续处理新发现的后代；agent、read_result、recovery 的有/无 process 分支及 verified_result→audit 均删除旧确认的替代或循环推导。gate 仍是一次调用内的父进程，没有增加常驻服务或隔离平台。

停写后，审阅者以自己的 `/tmp/extract-gate-race-review-settle.py` 独立重放原反例：默认 grace=1.0、31 次真实 fork/setsid 交接时返回 `confirmed=false`，gate 和后代均仍在，`gate_reaped=false`；解除强制调度后，对同一调用再次正常终止，得到 `confirmed=true`、`gate_reaped=true`、`descendant_alive=false`。现场为 `/tmp/extract-gate-race-review-i1zga9yr`。协调文件改用原子 replace，未伪造进程表；测试进程均已清理/reap。

## 已核范围

- 自建 user/mount/PID namespace、proc 挂载、pivot/capability/seccomp、隔离握手及证明、虚拟路径映射已从活动实现删除；没有关闭开关、假成功字段或替代 sandbox 平台。
- 每 attempt 的真实 workspace 与每 scope 的稳定空 cwd、私有 home 分开；scope/角色/精确 UUID 绑定保留，材料与 helper 使用真实路径，正式 Source 仍为固定来源 locator。
- Claude 原生 safe-mode/restricted、角色工具清单、精确 Edit/Write 与固定 helper 保留；本机 help 明确支持这些参数。Codex worker 采用原生 workspace-write 加本次 workspace，reviewer 为 read-only，approval=never，项目文档读取关闭。工具权限不被描述为 Extract 的宿主 OS 保护。
- `artifact_inventory` 在读取前核直属许可集合、类型及软硬链接；原稿封存、规范化、阶段字段范围、独立批准、当前 raw 来源、事实交接与发布检查仍在。此次对 Runner 的成功门槛只删除旧隔离项。
- 活动入口及文档没有额外的私有 proc 挂载或 sudo 准备要求。正常 Linux 的 `ps`、`/proc` 身份读取和无提权 child-subreaper 仍用于进程结算；它们不是新建进程隔离。

## 独立验证与限制

- 基线 87 文件 SHA 一致，初审 36 文件清单 SHA 一致，无未报告的基线文件改动；修复停写后的最终清单为 39 文件，SHA 全部一致；`git diff --check` 通过。本审阅报告独立于实施者清单。
- 三项短离线测试通过，1.615 秒：原子替换与残留文件/链接拒绝、阶段字段范围、P6 限域修改。
- 修复后独立运行五项短测试，4.010 秒全部通过：double-fork 取消后退休 session、主 CLI 成功后清理脱离后代、timeout 与未确认 fence、两后端原生结果的 exit/schema/nonce 恢复、缺 gate 闭合时旧 confirmed/强杀动作/无 process 分支与审计均拒绝。测试取自 `test_backends_runtime` 和 `test_current_recovery`，未重复完整 suite。
- 三次既有真实调用的文件 SHA 全部一致，原生日志均有唯一成功终结、正确 UUID 与 schema/nonce；两个 Claude attempt 的 workspace 内容仍独立保存。初审时由原有 parser 重读通过。
- 这三次真实调用发生在最终 gate 闭合核验修复之前，其 `cli-exit.json` 不含 `descendants_reaped`。它们只支持直接路径与原生 CLI 操作的历史结论；最终 gate 实现的验证以本次后续短离线复验为准，不补写旧证据或宣称最终实现已真实全链通过。
- 本次没有真实 Codex worker/resume、真实 Claude reviewer/helper 或完整真实提取链的通过结论。

当前启动不再需要额外私有 proc 或 sudo 环境准备。删除范围完整，真实路径与原生角色权限的安排合理，材料、会话和产物职责保持分离；必要的进程收尾修复规模有限。签收不扩大上述真实 CLI 覆盖范围，也不承诺 Extract 对宿主文件系统提供 OS 隔离。
