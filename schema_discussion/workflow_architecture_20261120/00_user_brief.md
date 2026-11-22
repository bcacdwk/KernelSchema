# 用户需求与讨论边界

记录日期：2026-11-20。以下为主代理对本轮用户需求的归纳，不将推测记作已批准决定。

## 当前问题

- 使用场景有四个：design（借助近邻参考从零设计）、extract（从 kernel 提取）、optimize（基于 schema 与反馈迭代）、implement（由 schema 实现 kernel）。
- extract 已有自动工具流程，用户本次不要求重新设计其内部。其余三项需要超越通用 Markdown 流程，形成具体可调用能力。
- 三个相关仓库是 `KernelSchema`、外源最新的 `atrex-kernel-agent`、用于多个 AKA 集合实验的 `AKA_experiment`。需核实各自现状，避免把功能开发与实验适配混在一起。

## 用户提出的方向（需要研究及确认，不能视为最终架构）

- KernelSchema 应可独立公开使用，有真实 demo 和四个功能入口；不希望最后只剩语法规则、extract 和 wiki。
- design/optimize 是 planner 的 schema 写作与迭代，依赖 docs/template/wiki；AKA 更适合作为 coder 完成 implement。
- 固定 schema 的实现可能跨多个 coder episode；design 出来的初版往往比后续 optimize 版本需要更长实现时间。需要区分 schema 版本迭代与 coder 内部迭代。
- implement 要以 root 为主干逐步完成必要 dispatch 与其他细节，并能够查阅 schema 文档。需要考虑 prompt 注入、工具调用、GPU/Agate 验证与反馈。
- KernelSchema 可以提供功能有限、简洁的独立 implement 工具；也可研究复用 AKA，但不能未经分析就复制一个完整 AKA。
- 长期可有 KernelSchema 自己的演示编排：extract 或 design 初始化 → implement → optimize → implement 循环。
- AKA 接入要轻量、模块化、易维护，最终是可开关的可选功能；倾向在外源 AKA 上开适配分支/fork，由 AKA_experiment 消费。当前不授权执行远端写操作。
- 倾向 KernelSchema 以 submodule 进入 AKA；也提出过反向依赖 AKA 的备选，需分析单向依赖与循环依赖问题。
- 开启 schema 时，AKA 需串行调用 planner/design/optimize，同时把 coder 的任务改成实现指定 schema。不能只插入规划步骤而忽视 coder prompt、工具、反馈和续跑语义。

## 本阶段的明确要求

- 先规划、研究、探索、推理，先厘清现状，再给初步大纲与需用户决策的问题。
- 与用户对齐方向后，才进入完整落地计划的撰写；现在不实现代码。
- 计划及其历史全部保存在 `KernelSchema/schema_discussion/` 下。
- 主代理担任 supervisor，负责全局、衔接、判断和面向用户的凝练汇报；复杂调查由子代理完成。
- 先三轮、共六位新子代理，模型 `gpt-6-astra`、effort `max`，每轮最多同时两位；再由新 `xhigh` 子代理整合初步大纲。
- 用户决策后，再两轮新的 `max` 子代理从不同角度撰写落地计划，最后由新的 `xhigh` 子代理整合完整计划。
- 优先使用新子代理保持上下文清晰；默认等待主动回报，只在阻塞、必要交接或决策时介入。

## 调研应回答的核心问题

1. 现有规则、工具、示例、实验适配到底实现到了哪一步？哪些仅在文档中，哪些代码路径已可用？
2. 四流程的最小公共输入输出、状态、错误与验收应是什么？需要多少共享基础设施才足够？
3. standalone implement 与 AKA implement 应共享什么、分别拥有什么，怎样避免复刻第二个完整 agent 框架？
4. schema revision、coder episode、GPU evaluation 和 optimize 的触发条件怎样对应？实现未完成、时间耗尽和设计不可实现怎样反馈？
5. planner/coder 的权限与 prompt 边界如何保持，schema fidelity 与正确性/性能怎样分别验证？
6. 三仓职责、依赖、fork/submodule 版本固定、上游同步和 switch-off 行为怎样安排？
7. 哪些先做，哪些后做；第一条端到端可验证路径是什么；哪些确实需要用户选择？
