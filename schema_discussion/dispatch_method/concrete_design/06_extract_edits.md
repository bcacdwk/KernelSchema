# 第 3 轮：提取方法文档实施

已按审定的 [PLAN](PLAN.md) 完成实施者 B 的 11 个独占文档。只修改 Markdown，保留现有未提交内容；没有修改 A 的文件、Python、checker、starter、算子 YAML、PNG、运行状态、inputs 或归档，没有启动提取、硬件流程、子代理或 commit。

## 关键改动

- [提取总览](../../../EXTRACT_SCHEMA.md)、六阶段及 [REVIEW](../../../schema_extract/REVIEW.md)统一为“具体完整代表＋目录记录”。dispatch 正式格式只引用 common；P1/P3 的逐段导航、Worker/Reviewer 分离和有效全文审阅保留。
- [P2](../../../schema_extract/phases/02_dispatch_draft.md)先识别主要机制，再选择真实具体 root。发现候选可以超过十个，拟完整交付集合含 root 不超过十个；继续深入需要实际源码线索，普通未读细节不构成无限调查理由。
- [P4](../../../schema_extract/phases/04_dispatch_plan.md)基于真实 root 落实正式记录，未来文件与对象明确仅为计划，不使用假链接或提前批准共享同一性；最多安排九个非 root，不重跑全 selector 分组。
- [P5](../../../schema_extract/phases/05_branch.md)只写自身 YAML。引用来源须实际存在且已独立审阅；仅影响本节点的完整本地化或改用已审相同整区，在当前 P5 审阅。集合、名额、未来依赖或其他节点前提/安排改变才先回 P4，动摇选型回 P2，Contract 回 P1，provider 本地错误回实际定义处。
- [P6](../../../schema_extract/phases/06_final.md)由新 reviewer 先独立理解最终全部产物，再核对 P2/P4 机制及 P5 新事实的去向。普通记录增补由按需 worker 集中更新 dispatch，再独立终审；重大改变不能降为收尾。完整集合、每份价值、真实基线、共享使用上下文和范围均须验收。
- [Supervisor 手册](../../../schema_extract/README.md)保留真实 CLI、冻结身份、预算、恢复、维护、ready/published 和交付操作。已由工具手册定义的旧引用迁移、provider 修复和合入细节改为链接，阅读视图按实际规则/工具版本解释。

逐输出验收、来源缺项、原实现缺陷、硬件占位及 `--skip-hardware` 边界保留；没有新增 GPU 验收或实测待办，这些豁免不扩展到普通设计与实现任务。

## 真实工具边界

[EXECUTOR_ALIGNMENT](../../../schema_extract/EXECUTOR_ALIGNMENT.md)为唯一实施状态入口：现有 Runner、checker、starter、计划和回执仍待对接，不能用当前工作树 CLI 端到端运行目标方法；旧 run 继续按登记的 source/rules/executor 解释。

只读核对了 `artifacts.py`、`stages.py`、`receipts.py`、`comparators.py`、当前 resolver/结构检查及 starter。实际七字段 plan、单父树与确定性索引、直接父批准解锁、阶段字段权限和旧语法没有因本文档修改而获得新能力。

[提取工具协议](../../../schema_extract/tools/README.md)保留这些真实协议，并补齐原主要位于阶段页的三处必要说明：P1 comparisons 的输出映射和声明锚点，P4 确定性表格/排序/转义/唯一覆盖缺口节，P6 revision_scope/affected_nodes 的实际字段授权。没有复制整套旧阶段，也没有新增命令、回执字段或状态。

## 已查范围

- 阅读本组全部 diff，并对照本轮前快照 `/tmp/dispatch_concrete_text_before_eeln7upz`；`git diff --check` 通过。
- 11 个文档的 194 个本地 Markdown 链接与目标锚点检查通过；EXTRACT、Supervisor README、REVIEW、tools README、P1、P3 的 57 个原有导航锚点全部保留。
- 已移除新方法正文中的旧 parent/from_parent、七字段、父树覆盖与旧回执字段要求。旧实现用语保留在实际操作、工具协议和对齐说明中；Supervisor 的“互斥”仍仅指 CLI 参数。
- 已将删除的 common“plan 与计算执行实例”链接改为 DOC4 的提交身份/调用边界；另修正 REVIEW 的旧 DOC5 动作锚点为现有动作声明入口。
- 定向检查确认 Python、YAML/YML、PNG、inputs 与 archive 没有本轮 diff。没有新增测试，也没有运行 extract、checker、绘图、GPU 或代码测试流程；链接核对使用临时只读脚本。

本组文字实施没有已知未完成项。第 4 轮独立语义验收由主代理继续安排；代码对接、旧运行回归及真实算子采用均留在集中页的后续工作中，本报告不宣称执行器已实现新方法。
