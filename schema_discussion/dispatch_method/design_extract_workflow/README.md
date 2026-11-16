# Design 与 Extract 流程改进结果

本轮完成两条流程的文字改进。八名新的 `gpt-6-astra` 代理按 **max → medium → max → xhigh** 分四轮工作，每轮两名；主代理审定修改方案、检查实际差异并收尾。没有新增字段议题。

## 正式改动

| 流程 | 结果 | 正式位置 |
| --- | --- | --- |
| Design | 先围绕主要 workload 逐层形成具体 root；主要组织清楚后，再判断其他方案的独特完整价值与并存需要，展开选定方案。 | [设计入口](../../../DESIGN_SCHEMA.md)第 1、5 节 |
| Extract | P2 审定机制与代表取舍；P4 落实已定集合、目录、对象与写作依赖；P3/P5 按已选实例写实。 | [提取总览](../../../EXTRACT_SCHEMA.md)、[P3](../../../schema_extract/phases/03_root.md)、[P4](../../../schema_extract/phases/04_dispatch_plan.md)、[P5](../../../schema_extract/phases/05_branch.md)、[P6](../../../schema_extract/phases/06_final.md) |
| 审阅与纠错 | 后续仍审当前有效全文与来源。选型的具体事实错误或关键关系缺口回 P2；已确认机制的正文遗漏、引文与普通记录按实际归属修正。 | [REVIEW](../../../schema_extract/REVIEW.md#发现问题与批准) |

P2 原有调查步骤与止点保持，不要求先把首读路径选为 root。普通 Design 保留有限比较、三层自检与回退，不增加 Extract 的调查或审批。完整设计上限、变体与共享的唯一规则仍在 [common](../../../schema_docs/0.common.md#派生节点与-dispatch)。

主代理验收时进一步明确：未展开的独立候选可以用相关实例交接，局部变化定位实际基线；root 定义错误仍回 P3，P6 保留原有授权范围内的修稿。没有扩大写权限或要求额外报告。

## 试读与取舍

两位 medium 分别用 FA4 真实参考材料及给定 workload、MoE 的真实限域路径作文字推演。两者均能推进方案与文件安排，没有报告必须新增规则才能继续的问题；这不构成完整设计、全算子分类或阶段批准。

八项反馈逐项处置见 [PLAN](PLAN.md#试读反馈的处置)。资料缺项、未完成推导、旧案例状态沿已有规则处理；MD-02 所涉 reference 可从本地 Git 固定版本读取，未将未读材料判成来源缺失。第一轮建议中，提前固定 Extract root 的强制重排未采用；第二轮 max 仅保留对正常推进有帮助的职责澄清。

共修改八份正式 Markdown，包含 [CHANGELOG](../../../schema_docs/CHANGELOG.md)。Optimize、Implement、common、DOC1–6、P1/P2、模板、算子实例及脚本保持不变；没有启动提取、schema checker、构建或 GPU 验证。既有 [F1/F2](../../role_prompt_trial/FIELD_ISSUES.md) 维持原状态。

主代理已检查完整差异与阶段分工。21 份新增或修改 Markdown 的 222 处本地链接、其中 105 处标题锚点均有效；排版检查与 `git diff --check` 通过。前六份原始评审与试读报告的摘要保持一致，修改范围核对通过。上述检查只证明文档一致性，不代替真实提取或设计验收。

## 过程记录

| 轮次 | Design | Extract |
| --- | --- | --- |
| 1：max 评审 | [01](01_max_design.md) | [02](02_max_extract.md) |
| 2：medium 角色推演 | [03](03_medium_design.md) | [04](04_medium_extract.md) |
| 3：max 裁定 | [05](05_max_decision_design.md) | [06](06_max_decision_extract.md) |
| 4：xhigh 修改 | [07](07_xhigh_design.md) | [08](08_xhigh_extract.md) |

[任务边界](BRIEF.md) · [审定方案](PLAN.md) · [Planner 输入](inputs/design.md) · [Extract 输入](inputs/extract.md)

原始评审与试读报告保留；实施结果以正式文档及本页为准。当前工具能力继续由 [EXECUTOR_ALIGNMENT](../../../schema_extract/EXECUTOR_ALIGNMENT.md)说明。
