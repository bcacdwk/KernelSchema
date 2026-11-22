# 2026-11-21 用户修订后的实施要求

用户明确：能跑即可，参考 atrex-kernel-agent；自定义机制能减少就减少。design/optimize只需写Schema，不需要效果验收；implement参考AKA。此决定取代旧默认流程里的严格K1自定义判据、sandbox/probe资格、写作独立review及效果门槛，不再把它们作为本轮默认路径完成的前置。

## 当前两个目标

1. KernelSchema可以直接design/optimize产出Schema，并在显式配置后通过简单有限driver调用标准implement、把真实结果反馈给optimize继续迭代。
2. AKA用switch开启Schema指导的原标准实现流程，关闭保留原有行为。Schema是其可选子模块，不修改或依赖AKA_experiment。

## 默认职责

| 操作 | 输入与输出 | 完成含义 |
| --- | --- | --- |
| extract | 既有自动工具与kernel→Schema | 原流程保留 |
| design | 任务描述、有限近邻参考、相关docs/template→Schema文件 | 文件写出；无强制review/效果/性能门 |
| optimize | 当前Schema、真实实现反馈或优化目标→新Schema/keep说明 | 修改提案写出，不冒称优化有效 |
| implement | Schema路径、原AKA标准problem、工作目录/预算→solution及原标准评测结果 | 复用原AKA Agent/Light/Executor/Evaluator与Agate正确性/性能，不先跑自定义六job |

写作只保留产物安全路径、可读取闭包和必要解析；格式/模板问题可保留为诊断，不因为无关字段长度丢弃原稿。公开入口优先直接task/schema/reference/feedback路径，Ref/request旧接口留兼容内部。旧strict流程可作为显式legacy工具保留，不能阻塞默认路径。不会将旧失败原件或未通过的严格Task改成成功。

标准AKA路径默认使用原生预算与结果记录，额外OwnerAccount只能是显式可选薄适配。若适配能复用现有有限组/submit回调则保留原账，否则不能为旧自定义账本重新加默认阻塞。本轮旧parent已发生费用全部保留，新标准任务预算须明确记录，不以新任务抹去旧失败。

当前新agents：B30负责KS write-only/API/CLI/driver与必要共享Ref接点；B31负责AKA standard_plugin、原Light流程复用与standard:implement、标准problem准备。两者先CPU/零费用准备，根固定后安排小规模真实示例。默认模型/强制完整资料投递也应按实际需要减少，不再为验收框架扩大工作。

原工作包/合同/全部失败与用量记录留在历史。本次新默认路径按可用行为检验，原额外权限/副作用/精确采样保证明确列为legacy未覆盖，不能从标准Agate成功推定它们也通过。
