# R3 B：当前运行恢复、真实交付与监督

本轮完成并停止写入。真实发布、恢复和监督出口已闭合；没有模型请求、GPU、正式提取、真实 queue 提醒、Git commit，也没有修改 Schema Wiki、Template 字段规则、旧运行数据或个人配置。测试使用临时源码、独立状态目录和离线 CLI。

## 实际修改

- `publication.py` 先核当前冻结身份、全部结算状态、P1–P4 与节点当前批准、P6 全文及最终 checkpoint，再用 `check_complete` 调当前 Template 完整目录 API。`selected_designs` 必须精确等于直属完整 YAML 集合；目录、唯一 root Contract、完整跨区来源图、固定源码引用和当前来源批准同时成立。发布不规范化已审字节，不排除未来文件或失效草稿，不增加 PNG 硬门禁。
- 交付仍使用已有原子不覆盖 rename 与 `.gitkeep` 支架协议。副本和源 checkpoint 复核后，在 Supervisor 短锁内再次核状态、维护、hold 和未消费的 pause/cancel 请求。目标冲突保留双方字节；目录可见而状态提交中断时，以完整 manifest 幂等重放。普通文件不改写；`.extraction/receipt.json` 记录实际 source、executor、选择身份、当前完整来源信息与 P6 receipt/attempt 摘要。
- `standalone.py` 直接使用当前 `validate_resolved_file`、`effective_document` 和 `validate_standalone_file`，保留节点元数据、when、Version、Realization，展开实际 Design 三区与 root Contract。生成结果必须与已验证有效文档相等，输出在 raw 目录外。`.source.json` 同时记录实际内容 `source_files/origins/effective_identity` 和完整目录 `validated_directory_files`、Template 文件身份。完整目录在源检查、生成检查和提交前后复核；来源漂移或不同目标内容均拒绝。
- `runtime.py` 只保留当前冻结输入、来源、配置、代码、能力和 attempt 后端/作用域核验。`EXECUTOR_CAPABILITIES` 采用 `frozen_execution.v1/current_maintenance.v1` 等当前能力；不再升级 execution binding、采用跨版本语义或迁移历史批准。`executor=True` 必须匹配实际执行的冻结 Python。默认 Claude/Qwen/max 仍只在创建新 run 时使用，恢复缺身份字段直接拒绝。
- `recovery.py` 将原生结果核验收口到 Agent 的 `read_result`，并核 `exit.json` 与 attempt/launch/真实退出码。已有成功结果须继续满足原生终结、隔离、schema、nonce、session 与原始事件身份；取消/超时不被完整日志改判成功。原来 `termination_unconfirmed` 的调用后续确认终止后按 `interrupted` 结算，不恢复批准。
- result 缺失的真正中断先核真实终止，再调用 `retire_session` 退休私有 session 绑定，最后清 Runner 的 UUID 并保存交接；不是仅改 state。严格未启动证明，包括没有调用记录的 prepared 状态，保留前次有效会话。未知进程或相反发射证据继续保留 fence。
- scope 恢复改用当前选定实例、generation、checkpoint、真实采用来源和冻结 executor/input 身份；不用父树或跨运行 lineage。自动恢复还须重读原生失败证据。相邻节点完成不改变失败授权，重放不重置调用、耗时、deadline、失败、裁决或签收次数。
- `audit.py` 只审本 run。原生 usage 保留调用增量、会话累计模型用量和累计费用的区别，按 attempt 去重，未知保持 `None`；累计会话费用不相加为调用费用。深审计复核退出/隔离/schema/nonce、保存的原稿、规范化后的 checkpoint 和实际来源锚点。归档只收具体证据文件；Codex 仅额外收其末条响应文件，不收整个私有 CLI home。后续终止证明单列，不改原始 result。
- `monitor.py` 观察当前显式 backend 的摘要与在途身份；不会将旧运行投影解释成当前状态。活跃项列 requested/reported model/effort、session scope 和本次 usage。汇总由 A 的 Store 从账本投影，包含未记账调用的未知项。Claude 不读取 Codex SQLite/native store；Codex 可选原生观测保留。`wakeup.py` 的 `queue_backend=codex` 是独立 Supervisor 职责，不随阶段后端替换。
- `cli.py` 接当前 `maintain RUN --issue-id ID --tasks FILE`；FILE 是 `{stage,node,action,fields,reason}` 数组，action 仅 review/patch。请求使用 A 的封存接口，运行仍由同一 Runner 排空和执行。冻结入口、后台 launcher 身份和控制代际全部保留，未重建旧 revise/continue 命令。
- 已核实并保留 `review-eval`：它评已知正反例的 Reviewer 语义能力，返回能力回归成绩，不参与 run 批准。原模块未依赖跨版本迁移；不是旧 adoption 工具。R4 只需同步其当前用途与显式 suite 文档。
- 经主代理分配，另删 `snapshots.py/inputs.py` 中失用的 `source_aliases/aliases` 参数和匹配分支；来源只匹配本次唯一 repository/ref 与文件清单。隔离材料路径映射、annotated 阅读和分页 cursor 保留。

## 活接口

| 接口 | 当前调用约定 |
| --- | --- |
| `runtime.verify_execution(store,state,*,executor=False,attempt=None,attempts=None,evidence=None)` | 返回本 run 唯一 lock；不升级、不猜身份。`execution_boundary` 在嵌套检查内复用深验。 |
| `runtime.frozen_run_invocation(store,state,arguments)` | 返回冻结新入口的 argv、run cwd 与环境，供 CLI/后台路由。 |
| `recovery.verified_result(attempt,result,termination)` | 只读复核真实 exit、原始 native 事件、schema/nonce/隔离与 session；不写会话。 |
| `recovery.recover_result(attempt,result,termination)` | 上述核验后用 Agent `settle_session` 结算；确证未启动时保留前次 session。Runner 已接。 |
| `recovery.handoff_session(state,attempt,result,reason,*,termination=None)` | 确认终止后先退休私有绑定，再登记 checkpoint/UUID 交接；幂等。Runner 中断调用显式传 termination。 |
| `recovery.scoped_restart_evidence(...)` | 绑定当前 scope 与真实来源；target 使用 `input_fingerprint/executor_digest`，没有执行升级区间。 |
| `publication.publish(store,state,output,*,export=False)` | 当前 ready、完整最终目录、有效 P6 批准、结算及控制屏障全部成立后原子交付。 |
| `standalone.create_view(source,output,repo)` | 当前 Template API 的严格独立视图；无旧 exporter/layout fallback。 |
| `maintenance.submit_request/barrier/publication_barrier/consume_requests/mark_drained` | A 定义与实现；CLI、monitor、publication 已接。节点当前来源用 `Store.verify_node/source_identity`，不把旧 approval identity 当作可直接改写字段。 |

## 删除与测试承接

A 确认当前维护、排空、调度和发布职责已迁出后，物理删除 `legacy.py`、`compatibility.py`、`continuation.py`、`revisions.py`、`revision_control.py`。当前 CLI 导入与命令集合已核验，五个模块不可导入，旧命令没有注册或别名。

删除跨版本专属 `test_semantic_revision.py`，以及依赖该夹具的旧父树 `test_design_derivation.py`。其中有用的不变量由当前 `test_artifact_transactions/test_stage_flow` 以及 A 的 `test_current_returns` 承接：实际 provider 权限、修复后新批准、完整旧稿保留、候选与旧 consumer 分离检查、等价身份与真实内容影响。旧 reparent、旧 exporter、执行绑定切换用例没有保留。

`test_cli_recovery` 只删 compatibility/continuation 专属用例，其余四项已改当前 Claude/新入口，并用真正隔离的离线 EX_TEMPFAIL/native 坏 nonce 验证。保留混合测试中的有效权限和来源检查，未使用 skip。

## 验证

28 项不同测试通过；补强后的原生审计与观察又做了针对性复验。机器验证的语义提案由确定性夹具提供，不代表真实模型识别了设计知识或语义缺陷。

| 模块 | 数量 | 实际覆盖 |
| --- | ---: | --- |
| `test_current_delivery` | 3 | 五节点真实六阶段发布、完整选择集合、实际/解析来源区别、P6/未结/未签收拒绝、冲突、原子 rename 后崩溃重放、pause 与 rename 竞争；冻结身份和 maintain CLI；原生离线 CLI 单 root 全链、完整 audit verify/archive/reuse 与 state 只读。 |
| `test_cli_recovery` | 4 | 新 CLI 固定输入重用、单写者、明确追加预算不退还原调用、坏 nonce 归因真实 attempt。 |
| `test_current_recovery` | 4 | 两后端真实离线 namespace 的 native/exit/schema/隔离检查；缺 result 私有 session 退休；未启动保留原 session；耗时高水位与 once 计数；实际 EX_TEMPFAIL 的 P5 限域恢复、旁支完成后的授权稳定与手动控制优先。 |
| `test_standalone` | 5 | 当前来源图、精确 Decimal、元数据与 Realization、完整目录漂移、源与结果严格检查、冲突/复用和新 CLI。 |
| `test_inputs` | 4 | annotated 来源仍独立保留，路径边界与冻结来源不被迁移精简破坏。 |
| `test_usage_observation` | 3 | 原生用量口径、Claude session 累计与调用增量、Codex 可选观测、冲突和 attempt 去重。 |
| `test_monitor_evidence` | 3 | 失败证据指向实际 scope，不猜其他节点最新 attempt。 |
| `test_wakeup_retry` | 2 | 本地 fake queue 的重复提醒、ack/scan/stop 竞争及持久上限；真实发送入口在测试中被拒绝。 |

真实交付与 CLI 六项在字节核对过的稳定副本通过（130.356 秒）；本地 21 项同样在稳定副本通过（27.037 秒）；追加的整链原生归档用例在第二个稳定副本通过（19.405 秒）。随后三项 native audit 补强检查（2.727 秒）和八项观察检查（0.450 秒）通过。稳定副本使用 `/opt/venv/python3.13.12/bin/python`，避免 `/tmp` 默认解释器缺 PyYAML。

所有当前 Python 编译、实际入口 help、CLI 命令注册/导入检查和 `git diff --check` 通过。简明测试输出见 [R3_B_checks.txt](R3_B_checks.txt)。

## R4 交接

1. 全面同步活动手册：旧迁移/父表、只限 P6 签收、旧 Codex 默认、旧 exporter fallback、旧 source aliases 与旧 README 评测 archive 默认均需清理；源码活路径已移除。`review-eval` 文档说明当前已知正反例评测，不参与 run 批准。
2. 尚留的有效旧测试夹具必须转换，不能以删有效断言或 skip 宣称全 suite 通过：`test_proposals/test_issue_evidence` 仍引用旧 `index_prefix`/P4 父表；`test_review_protocol` 有旧字段、continuation 和执行绑定夹具；`test_scoped_recovery` 用旧 plan/parent；`test_recovery_budget` 的末个调度夹具用旧 plan/parent；`test_source_gaps` 用小写旧字段。新当前测试覆盖核心不变量，但本轮未声称 unittest discover 全绿。
3. 按 R1/R2 交接完成完整动态 proposal schema 下的真实 Claude 工具读写、精确 resume、独立 reviewer，以及 Codex 无 `/proc/self/exe` 时的实际工具能力。当前测试中的 CLI 都是离线替身；没有真实模型、GPU 或 queue 投递。
4. A 将在其最终稳定副本联合重跑真实发布、原稿审计归档和限域恢复。若 A 后续接口有变化，沿本表核对；B 已停写，后续修改归新轮 owner。
