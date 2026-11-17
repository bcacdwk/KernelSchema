# Extract 工具落实：规划任务

基线：`27f36e5`，规划开始时工作区干净。当前轮只提交实施计划，供用户审阅。

## 目标与范围

- 以当前 schema_docs、schema_template 和 Extract 六阶段方法为依据，实现提取工具；Schema Wiki 重提取另行安排
- 保留有价值的自动调度、独立审阅、恢复与交付能力，直接更新当前实现，不建立旧版本兼容层
- 根入口统一为 `extract_yaml.py`，工具保持简洁、模块清楚
- 默认阶段 Worker、Reviewer 使用 Claude Code 的 `qwen3.8-max[1m]`，effort 为 `max`；保留 Codex 可选后端，Supervisor 不随之切换
- 各阶段、各角色使用新会话，P5 另按节点分开；同一作用域内按需精确续写，交接材料显式传递
- 固定模型提示词集中在 `schema_extract/prompt/` 的 Markdown，现有阶段页一并归入；Python 负责动态上下文与组合
- 每次调用简短提醒文本与命名的表达重点，字符及排版规则交由 Template 工具处理
- 利用既有回执完成问题候选、独立判断与 Supervisor 处理的交接

## 必须落实的关系

- P2 独立审定主要机制与代表，含 root 最多十份完整设计，且每份有独特价值
- 参数、局部变化和已有组合集中记录于 `dispatch.md`，不展开为完整 YAML 队列
- P4 落实已定选择；真正影响选择的新事实才返回对应阶段
- P5 按实际整区 `from` 来源安排工作，共享 Contract 与设计借鉴关系不等于单父树
- 阶段写权限、规范化候选、实际产物、有效全文审阅和批准身份一致
- 进程退出、提供者完成事件、结构化提案和合法产物各自经过对应核验
- Claude 工具清单不视为操作系统隔离；实际读写能力与角色权限需在计划中说明

## 规划分工与交付

1. 两位新的 MAX 独立预走全流程，分别重点核对方法与产物、执行器与提示词
2. 一位新的 XHIGH 读取两份报告，核实分歧与关键接口，整合成 `PLAN.md`
3. 主代理核对范围、顺序和待决事项，向用户简短汇报

实施计划应给出三至四轮 MAX 的分工、两位 XHIGH 验收、一轮 MAX 修复及原审阅者复验；明确哪些保留、删减、改写、新增，以及必要的验证。真实模型调用和正式提取不在当前规划轮执行。

## 入口材料

- `EXTRACT_SCHEMA.md`、`schema_extract/README.md`、`REVIEW.md`、`phases/`
- `schema_extract/EXECUTOR_ALIGNMENT.md`、`tools/README.md` 与实际 Python 调用链
- `schema_docs/0.common.md`、`CHANGELOG.md` 与 Template 对外接口
- Extract archive 中与分支膨胀、恢复、成本直接相关的记录，按具体问题取证

执行器对齐记录中的旧运行迁移建议不覆盖本次无需兼容的要求。已有 Claude 基础调用证据可用，完整流程能力仍须在实施时验证。
