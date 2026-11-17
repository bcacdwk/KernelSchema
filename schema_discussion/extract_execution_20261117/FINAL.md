# Extract 实施交付

当前执行方式已按用户决定移除 Extract 自建进程隔离：直接使用真实路径与 CLI 角色权限，不再需要额外 proc/sudo 准备。本次独立增量及最新验证见 [isolation_removal/IMPLEMENTATION.md](isolation_removal/IMPLEMENTATION.md)。下文保留此前重构与审阅的历史结论；其中隔离环境说明不再适用于当前代码。

本轮实现、独立审阅和修复已完成。四轮各两位新 MAX 实施，两位新 XHIGH 独立审阅，两位新 MAX 修复，原两位 XHIGH 复验签收。工作区未提交 Git commit。

## 当前结果

- 统一入口为 `schema_extract/extract_yaml.py`
- 默认阶段 worker/reviewer 为 Claude Code `qwen3.8-max[1m]` / `max`，Codex 显式可选；阶段、角色和操作会话独立
- P2 选择少量完整设计，P4 安排写作，P5 按实际整区 `from` 处理来源；参数、局部变化与组合留在目录
- 固定提示集中在 `schema_extract/prompt/`，包括阶段、角色、审阅、恢复和交接；文本与命名提示聚焦核心信息，排版和长度交 Template
- 当前字段、规范化、权限、来源、预算、恢复、裁决签收、发布及阅读视图已接通；兼容和跨版本迁移模块已删除
- 未修改 Schema Wiki，未执行 GPU 或批量提取

## 独立审阅

流程侧四项问题和执行器侧三项问题均已修复并独立复验：

- [流程签收](audit/A_FLOW_RECHECK.md)：读前许可集合、跨代事实交接、P1 与待修 Design 的阶段边界、P2 职责投影复审
- [执行器签收](audit/B_RUNTIME_RECHECK.md)：未启动调用审计、格式错误恢复的可信基础、Codex 私有路径用量观测

当前无本轮已发现但尚未闭合的产品问题。

## 验证范围

- 稳定副本完整离线 Extract suite：125 项通过、无 skip；第四轮最终局部补验 25 项、Template API 6 项通过
- 审阅后修复另有针对性复现与相关回归，原两位 XHIGH 独立重放并核对最终文件；没有把旧整套结果当作最终修复验证
- 真实 Claude 的读取、原子写入、自检、完整 proposal、同 UUID resume、只读 reviewer，以及当前 Reviewer 正反例通过
- Codex 真实 worker 证据来自已挂真实 proc 的临时副本；最终暂存目录接线另有离线 native 覆盖，真实 reviewer 已通过，真实 resume 只验证 Claude
- 真实 P1–P6 模型全链按用户要求停止，未取得阶段批准或发布；所有本轮登记模型进程已确认结束，不计为完整真实链验收通过

详见 [R4 A](implementation/R4_A.md)、[R4 B](implementation/R4_B.md)、[流程修复](fixes/A_FLOW_FIX.md)、[执行器修复](fixes/B_RUNTIME_FIX.md)及对应日志。

## 需要知会的运行条件

本机限制普通用户创建新的 procfs，实际 CLI 调用需使用[运行环境说明](../../schema_extract/EXECUTOR_ALIGNMENT.md#运行环境)中的显式系统准备方式。特权阶段只运行系统挂载及降权命令；仓库 Python、Runner 和模型均在永久降回调用者后运行。产品不自动调用 sudo，不修改 sudoers 或个人配置。

完整真实模型链尚未验收是用户指定的验证范围；后续正式提取时据实际运行继续观察。
