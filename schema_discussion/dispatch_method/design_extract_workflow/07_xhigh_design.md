# 第四轮 Design 实施记录

依据[审定方案](PLAN.md)和[第三轮 Design 决策](05_max_decision_design.md)，采用两段替换稿，仅修改 [DESIGN_SCHEMA.md](../../../DESIGN_SCHEMA.md) 的第 1、5 节相关段落。

- 第 1 节明确优化目标、优先 workload、任务要求实现的范围和计量边界，围绕主要 workload 逐层形成具体 root，并链接 common 的 root 定位。
- 第 5 节在 root 的主要计算、工作归属和协作关系清楚后，先判断其他方案的独特价值与并存需要，再逐层完成选定方案；其他变化和未完整展开的独立方案沿 common 记录，交付范围对照任务要求核对。

已通读改后全文并检查差异，正式文档仅有上述两段替换。有限候选比较、三层推导、数值与资源自检、资料缺项处理和最早受影响层回退保持有效；root 无须先实测或覆盖整个 Contract。未增加字段、审批、固定报告或 Extract 的源码调查要求。

已核对全文 21 个本地 Markdown 链接的文件与锚点，并对照 common 的 root、完整设计和范围规则；`git diff --check -- DESIGN_SCHEMA.md` 通过。

本次实施只写入 Design 正式页和本记录，未修改评审、试读报告或 inputs。未运行 schema checker、真实提取、构建或 GPU 实验，未提交。未发现需偏离审定方案的问题；Changelog 与综合验收交由主代理完成。
