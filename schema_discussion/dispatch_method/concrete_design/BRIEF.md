# 具体设计、变体目录与提取方法

本轮依据用户最新决定重新审视规则。此前 dispatch 研究与已改 common 均不是不可改变的结论。

## 确定的目标

- 每个算子的完整设计 YAML 含 root 最多 10 份，越少越好；每份必须有独立的设计价值
- 每份 YAML 表达一套具体、闭合的设计，不充当多个配置的选路容器
- 不同配置的参数变化、局部改动、已有决定的组合，统一在 `dispatch.md` 规范记录，不展开所有配对
- 当前具体方案仍需定义自己的参数，以及随问题输入变化的尺寸、索引和工作量；不得借此偷偷加入配置选择语法
- 相同的完整 Design 区域使用 `from` 共享，非 root 主要撰写真正不同的区域；不借此次改动引入片段 patch 或组件语言
- common 面向所有 schema 作者，只定义文件职责、设计/变体的记录规则和必要理解；源码分析方法属于 extract
- extract 同时考虑前期识别核心设计、选择 root、编写代表、整理变体、审阅与维护；worker/reviewer 分工保持，阶段数量与具体职责需要论证
- 本轮只修改 Markdown/README，不修改 Python、检查器、运行状态、源码或现有算子 YAML，不启动提取

## 待研究的核心问题

1. 什么变化值得完整设计身份，什么应记录为参数、局部或组合变体；超过候选上限时如何精简而不伪装完整覆盖
2. `dispatch.md` 的最小规范：如何定位基线、说明差异/条件/组合关系、保存设计出处而不重复 YAML
3. `when`、整区 `from`、独立设计与设计派生如何配合，避免隐式 patch 或新配置语言
4. 提取在写 root 前需要完成哪些全局识别；六阶段是否保留，P2/P4 怎样避免重复和过早锁定分组
5. common、EXTRACT_SCHEMA、Supervisor README、阶段提示、REVIEW 与通用作者入口各自承担什么，消除重复与冲突
6. 纯文字规范改动与现有执行器实现的界线：具体代码迁移留 TODO，不宣称本轮已经实现新的运行语义

## 四轮安排

| 轮次 | 独立任务 A | 独立任务 B |
| --- | --- | --- |
| 1 | schema 作者规则与 dispatch 格式 | 提取方法、阶段及文件职责 |
| 2 | 规则反例与实际记录试写 | 工作流审计与统一修改方案 |
| 3 | 修改 schema 和作者侧文档 | 修改 extract 顶层及阶段文档 |
| 4 | 作者视角独立验收 | 提取 worker/reviewer 视角独立验收 |

每个任务使用新的 `gpt-6-astra`、`effort=max` 代理，每轮最多两个。主代理阅读关键结论、决定范围、整合方案并验收，不复刻各代理的全量探索。

## 本轮起点

工作树已有上一轮的 schema_docs 修改与 dispatch 研究，继续保留其有效内容。开始时的文本快照位于 `/tmp/dispatch_concrete_text_before_eeln7upz`，用于区分本轮增量，不用于回滚用户改动。

主要入口：当前 `schema_docs/0.common.md`、`EXTRACT_SCHEMA.md`、`schema_extract/README.md`、`schema_extract/phases/01_contract.md` 至 `06_final.md`、`schema_extract/REVIEW.md`。十算子证据及 MoE 的初始/默认范围差异见上一轮研究；规则建议须按用户本轮决定重新判断。
