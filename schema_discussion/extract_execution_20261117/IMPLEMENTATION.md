# Extract 实施进度

当前执行方式已按用户决定移除 Extract 自建进程隔离：直接使用真实路径与 CLI 角色权限，不再需要额外 proc/sudo 准备。本次独立增量及最新验证见 [isolation_removal/IMPLEMENTATION.md](isolation_removal/IMPLEMENTATION.md)。下文保留此前重构与审阅的历史结论；其中隔离环境说明不再适用于当前代码。

用户已批准 [PLAN.md](PLAN.md) 并要求完成实现、独立验收和修复。规划页的“本轮只形成计划”描述规划时点，不限制本次实施。

用户随后明确要求停止全流程试跑：不再启动或继续真实 P1–P6 模型全链，已有在途试跑按正规取消流程结清。保留已完成的真实 CLI 局部验证和离线覆盖，继续代码收尾、独立审阅与修复，不将未完成全链记为通过。

基线：`27f36e5`。开始时仅本规划目录未跟踪；不修改 Schema Wiki、旧运行数据或 Template 规则。

## 分轮状态

| 轮次 | 范围 | 状态 |
| --- | --- | --- |
| 1 | 当前字段与产物接口；双后端与隔离 | 完成：A 12 项、B 11 项局部验证通过 |
| 2 | 六阶段主链；提示外置与入口 | 完成：A 7 项主链验证，B 7 项局部验证及入口冻结检查 |
| 3 | 定点返回、预算与裁决签收；恢复和交付精简 | 完成：A 33 项、B 28 项验证；A 留一项晚期共享元数据同步给 R4 |
| 4 | 集成、有限真实调用、文档同步 | A、B 全部完成停写；报告与实测边界已收口 |
| 独立验收 | 两位新 XHIGH 分别审阅完整改动 | 完成，A 4 项/B 3 项必须修复，均有独立短复现 |
| 修复与复验 | 新 MAX 修复，原审阅者复验 | 完成：原两位 XHIGH 独立签收，7 项发现全部闭合 |

## 协作约定

- 同时最多两个实施代理，每轮使用新的 MAX；每个共享文件有唯一负责人
- 各代理在 `implementation/` 保存简短的本轮结果、对外接口、检查及后续接线要求
- 不提交 Git commit；保留实际变化及可复查证据
- 默认阶段模型为 Claude Code `qwen3.8-max[1m]` / `max`，各阶段和角色独立会话；Codex 可选
- 自动规范化先于最终独立批准；裁决改稿同样自动处理，并保留一次有界签收
- 实测使用临时材料和独立状态目录，不运行 GPU 或批量转换 Schema Wiki
- 已有来源、凭据与个人配置不写入报告；工具缺口修复后才声明对应能力可用

## 当前交接

第一轮 A 独占 `artifacts.py / receipts.py / comparators.py`，必要的共用规则常量由 A 维护。
第一轮 B 独占 `agent.py / backends.py / isolation.py / processes.py / observation.py / audit.py`。
其余主链、提示、入口、恢复与发布文件留给后续轮次；跨文件需求通过接口报告传递。

第一轮 A 已完成并停止写入，交接见 [R1_A](implementation/R1_A.md)。
第一轮 B 已完成并停止写入，交接见 [R1_B](implementation/R1_B.md)。
第二轮已完成并停写，见 [R2_A](implementation/R2_A.md)、[R2_B](implementation/R2_B.md)。
第三轮已完成并停写，见 [R3_A](implementation/R3_A.md)、[R3_B](implementation/R3_B.md)。第四轮已完成，见 [R4_A](implementation/R4_A.md)、[R4_B](implementation/R4_B.md)。独立审阅：[A_FLOW](audit/A_FLOW.md)、[B_RUNTIME](audit/B_RUNTIME.md)；修复：[A_FLOW_FIX](fixes/A_FLOW_FIX.md)、[B_RUNTIME_FIX](fixes/B_RUNTIME_FIX.md)；最终独立签收：[A_FLOW_RECHECK](audit/A_FLOW_RECHECK.md)、[B_RUNTIME_RECHECK](audit/B_RUNTIME_RECHECK.md)。所有代理已停写，见[最终交付](FINAL.md)。
双方已协调 proposal 使用现有 `attempt_id` 作 nonce，不新增重复字段。
产物接口区分 P2 selection、P4 writing plan；执行器接口统一为 `AgentCLI.run`，由下一轮主链接入。

第一轮 B 的 11 项离线行为验证已通过，包括权限、续接、坏事件、用量与取消；两种本机 CLI 的隔离内版本启动通过。完整动态提案和实际工具调用留集成轮核验。
本机新挂载 procfs 返回 EPERM，隔离使用真实只读内核信息的窄视图，不模拟 self/PID；Codex 的 PATH alias 提示需在 R4 实测工具调用时闭合。
R2_A 独占 `runner.py` 内的 `DEFAULT_CONFIG/create_run`；R2_B 负责 CLI 参数映射、提示和冻结/helper。
R2_A 的 `Store.context` 只读取当前冻结 lock；`create_run/Store.create` 删除 seed/import_state，R2_B 的 CLI 需同步去掉入口。
提示接线约定：`prepare_attempt_materials(state, lock, attempt, *, rules_lock=None, executor_lock=None)` 在保存 attempt 前准备隔离材料/helper 并返回需合入的字段；`build_prompt` 沿用调用签名，生成本次 stdin。

第三轮 A 独占 `runner/stages/store/maintenance/budget`，按需接线 `artifacts/receipts` 及阶段返回/签收提示。
第三轮 B 独占 `publication/standalone/runtime/recovery/audit/monitor/wakeup/cli`，A 断开调用后由 B 删除兼容和迁移模块。
正常主链的测试已使用真实内容门禁，但发布此前为替身；第三轮必须接通真实发布与恢复。当前 source 身份漂移采取拒绝，机械等价对齐、精确失效及非 P6 签收预算仍需第三轮完成。
R3 已核实 `review_eval` 是已有的 Reviewer 已知正反例能力评测，不参与 run 批准，也非跨版本迁移入口；保留其当前用途，R4 清理旧 archive 默认说明。

R4 必须更新保留的当前行为用例：`test_proposals`、`test_issue_evidence`、`test_review_protocol`、`test_scoped_recovery`、`test_recovery_budget` 末用例、`test_source_gaps`。原迁移/换父专属用例已删；不得 skip 或删除有效行为来取得全绿。
第四轮 A 独占 `runner/stages/store/maintenance/budget/artifacts/receipts/prompts/reading` 与流程测试；B 负责执行器、交付、CLI、活动文档和真实模型验证。

## 第四轮具体补缺

- P1 晚期更正共享 `META_DATA.Name/Describe` 尚缺受控同步出口，R3 已保存复现；由 R4_A 修复，不能放开 P5 对这两字段的写权限
- 真实 Claude/Codex 均需要动态 `/proc/self`，R1 窄 proc 视图不支持正常调用；产品改用真实私有 proc
- 本机 rootless 新 proc 挂载被内核拒绝；显式系统 bootstrap 在私有 mount namespace 挂真实 proc，再永久降回原 UID/GID 后启动 Runner，可使后续原有隔离正常工作。产品不内置 sudo，也不改 sudoers；这是最终需告知用户的环境前置条件
- Claude 原生 Edit/Write 使用同目录临时文件/rename；worker 改用私有可写暂存目录，非许可现存文件只读挂载，最终严格按文件/字段集合采纳。reviewer 仍整体只读，源码/规则/宿主状态不可写
- R4_B 已验证显式 `gpt-6-astra/high` 的真实原生读写、自检、只读 reviewer 与完整 13 字段 proposal；Claude `qwen3.8-max[1m]/max` 的真实原子编辑、自检、完整 proposal 和同 UUID 续接已通过
- 真实 proc 与暂存目录下 9 项 namespace 检查通过，含动态 self/child-self、host PID 不可见、路径/内存/mount 越界拒绝与取消清理；Reviewer 正反例局部验证已开展
- 首个真实全链在 P1 单次 600 秒超时，保留合法草稿但未有阶段批准；后续全链试跑按用户最新要求停止，不再重启
- 停止已确认：首 run 为 `budget_exhausted/P1`；第二 run 经正规 cancel 为 `cancelled/P1`，termination confirmed、survivors 为空；两者均零阶段批准、未发布，active/unconfirmed/unsettled 为空。最终 51 次登记观察、36 个不同进程身份 live=0/unknown=0。证据：`implementation/R4_B_stopped_trials.json`
- 修正来源域后的两项真实 Reviewer 正反例均通过；后续不再启动模型调用
- 第四轮稳定副本全 Extract suite：125 项通过、无 skip；最后少量生产/提示改动仅做相关局部补验，完成后交独立 XHIGH 审阅
- 第四轮最终补验 25 项、Template API 6 项通过；35 份活动 Markdown 链接、116 个 Python 编译及 diff-check 通过。Codex 真实 worker 的证据来自真实 proc 临时副本，最终原子暂存目录由离线 native 覆盖；真实 resume 只验证 Claude，按实际范围报告

## 独立审阅发现与闭合

- A-F1：manifest 尚未按当前许可文件集合在读/hash/copy 之前拒绝额外 YAML；短复现最终拒绝发生在原稿已被读取/复制之后
- A-F2：P1 仅改共享 Name/Describe 后 P5 重审升代际，P6 材料仅取当前 generation，导致原 worker 尚未落实的普通变体事实不可读；需保留有来源/实例标识的事实交接，不将旧回执误作当前批准
- B-F1：可信 `not_started` 启动失败没有 provider events，恢复已正确识别但 audit 无条件要求 events 导致 verify/archive 错误失败；审计需复用相同可信判定并准确标记不可用信息，不生成假事件或用量
- B-F2：恢复 `receipt_error` 未像 completed 一样重验隔离/原生/会话成功基础，已失效的 proof 仍能使 session 重新 ready；可续接的格式错误同样需要可信运行证明，不成立时退休/阻止续接
- B-F3：Codex 原生 SQLite 使用沙箱内 logical CODEX_HOME rollout 路径，observer 仅认宿主私有 home 导致原生增量观测失效；需受限的本 scope 路径映射，保留 UUID/链接/计数校验、unknown/partial 和账单边界，不重写历史观测
- A-F3：consumer 因退休 provider 等待 P5 修稿时，P1 合法修共享 Contract 被规范化扩展到未闭合 Design 阻塞，P5 又必须等 P1，产生循环等待；P1 应可按本阶段边界完成修订，不能借此恢复旧 provider 的当前资格
- A-F4：P1 合法更正 Source 后未复核受影响的 P2 职责投影，P2 状态仍 approved 却让后续 P3 准备失败；需按真实投影重新开启必要的当前 P2 独立复审，保留未受影响的选择/回执/预算

以上七项由两位新 MAX 修复，原两位 XHIGH 已逐项独立复验签收。

B 最终复验：原三项反例全部纠正，5 项短正反例通过；200 份复制文件与当前树一致，既有 7 份原模型证据 hash 不变。原审阅者未发现新的实质遗漏，结论见其最终复验报告。
A 最终复验：原四项反例全部纠正，5 项相关回归通过；169 份复制文件与最终树一致。P2 独立复审后实际继续至离线 P6 ready，未新增真实模型调用。
