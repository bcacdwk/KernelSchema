# 实现 schema 对应的 kernel

本入口供 Coder 根据既有 schema 从零实现，或按设计修改、优化已有 kernel。目标是在任务已选环境中落实 Contract 与 Design，交付对应版本的代码、任务要求的验证结果，以及有依据的实现反馈。

开始前理解 [DOC 6 的实现自由边界](schema_docs/6.realization.md#实现自由的边界)；字段语义与权限仍以 schema_docs 和 [README](README.md#2-schema-是什么)为准。实现方便、更快或一次测试通过，都不授予修改上游决定的权限。

## 1 接收设计并定位实现

读取任务要求、root 的共享 Contract、目标节点的完整 Design 和 realization。多节点任务先用 `dispatch.md` 定位路径，再以 YAML 的 `when` 确认适用范围；文件关系查 [common](schema_docs/0.common.md#文件与共享-contract)。派生节点使用同目录 root 的 Contract，不能只拿派生文件或从 parent 拼接 Design。

确认节点与 root 的版本、实际 workload、验收要求、已选设备和实现栈，以及任务提供的构建、测试和测量入口。通过 Contract 的资料绑定核对所需原语和工具链支持；资料占位的处理见 [DOC 2](schema_docs/2.contract.hardware_specification.md#省略与占位)。发现环境与设计不匹配时，指出具体缺口，不自行更换实现栈或降级设计。

- **从零实现：** 从任务 ABI 和所选栈的编程资料进入，确定外部入口、构建方式、依赖及代码交付范围，再落实该路径的执行设计。
- **修改已有实现：** 先确认代码原来对应的节点和共享 Contract 版本，区分本次是落实设计修订，还是在同一设计内优化实现，再定位受影响的 wrapper、dispatch、辅助工作与主 kernel。共享代码或路径选择受影响时，再检查相关其他节点及测试。

无法确认当前 schema 版本的检查结果时，使用[模板检查入口](schema_template/README.md#4-检查与处理反馈)，将发现的 schema 问题反馈 Planner。结构通过后仍需辨别资料和设计缺口；遇到阻碍按下文反馈，继续不受影响的部分。

旧实现、案例和实验只用于回答具体问题；从 [kernel_wiki](kernel_wiki/) 查阅时先核对来源和版本，已有记录不自动验证本次代码。

## 2 将设计约束落实到代码

沿数学值、工作实例和执行关系，找到每项设计在代码中的落实位置。先核对其语义，再选择实现细节：

| 要落实的内容 | 关注点与按需查阅 |
| --- | --- |
| 外部计算与数值路径 | 保持输入域、全部输出与副作用、初末状态、精度转换、归约与近似选择；默认精度和参考判据查 [DOC 1](schema_docs/1.contract.algorithm_requirement.md#数学表达与参考精度)，改写的精度语义查 [DOC 3](schema_docs/3.design.math_reshape.md#精度与辅助运算)。 |
| 工作组织与提交 | 按 work_schedule 实现分块、worker 组成、工作归属、领取与退出，覆盖需要的预处理和合并；有多个 phase 时查 [提交与接续](schema_docs/4.design.work_schedule.md#phase-的提交与接续)及[数据交接](schema_docs/4.design.work_schedule.md#跨-phase-数据交接)，不能靠源码排列推定完成依赖。 |
| 局部执行与资源 | 保持 lane 职责、动作契约、完成与可见性保证、资源容量和复用关系；异步推进查 [lane 顺序](schema_docs/5.design.tile_pipeline.md#lane-内的顺序)，缓冲共用查[资源与依赖](schema_docs/5.design.tile_pipeline.md#worker-内的资源与协作)，持久 worker 再查[跨工作接续](schema_docs/5.design.tile_pipeline.md#跨工作接续与索引边界)。 |
| 额外实现条件 | 同时满足 realization 中适用的 require 和 forbid，按理由优先尝试 prefer；具体原语保证查已选栈的 programming 资料。 |

普通地址计算、代码组织、barrier 编号和不改变既定语义的线程 layout 等细节由 Coder 选择。判断替代是否自由时，按 DOC 6 核对实际影响，不能只比较 API 或指令名称。需要改变数学、精度、工作组织、资源协议或交接方式时，先反馈对应设计问题。

## 3 实现与验证迭代

按依赖关系实现或修改代码，逐步核对受影响的计算和执行路径。利用任务现有的测试、构建与测量设施；本仓库尚未提供统一 GPU Harness。验证范围由任务要求和改动影响确定，修复问题后重查相关约束与结果。

| 验证问题 | 如何判断 |
| --- | --- |
| 正确性是否满足 Contract？ | 按固定 golden reference、gap 处理和 tolerance 检查全部契约输出及要求的原地写回，覆盖任务指定 workload 和受改动影响的边界。不能用另一种误差指标替换验收要求。 |
| 代码是否遵守设计？ | 核对数值路径、工作分配、提交、依赖、资源和 realization。对编译结果或异步行为有疑问时，按任务的 programming/profile 资料查生成代码、运行轨迹或相关工具证据；数值通过不能单独证明设计符合性。 |
| 是否达到要求的性能？ | 任务要求性能验证时，固定比较实现、实际 workload、环境与计时边界，计入要求范围内的辅助工作；记录适用范围和回退情况。不要由孤立原语或主 kernel 的加速推断完整调用收益。 |

代码修复、实现细节比较和验证可以在本角色内迭代。若性能问题来自既定设计，或必须违反上游条件才能继续，记录观察与依据并反馈 Planner。缺少运行环境时，说明未执行的验证及其影响，不能用结构检查或旧结果替代。

## 4 交付与反馈

交付可定位的代码、入口和必要依赖，说明对应的 schema 节点及共享 root 版本、修改范围、验证结果与尚未解决的问题。任务消息和已有实验记录即可承载这些信息，无需固定报告文件。引用和实验追溯见 [common](schema_docs/0.common.md#来源与证据)与[模板说明](schema_template/README.md#3-填写与修订)；不要将 `meta_data.source` 改作实现进度或实验记录。

按问题所属位置反馈：

- **有证据的 prefer 替代：** 按 [DOC 6](schema_docs/6.realization.md#有证据的替代反馈)追加 `prefer[].feedback`，保留原 choice 与 reason，引用真实实验记录；修改 YAML 后按模板维护版本并[检查任务](schema_template/README.md#4-检查与处理反馈)。
- **设计不完整、冲突或不可实现：** 向 Planner 指出相关节点、字段或动作、触发条件、具体阻碍和已有证据，由[设计优化流程](OPTIMIZE_SCHEMA.md)判断与处理；无法实现 require 时也按此处理，不能自行降为 prefer。
- **Contract 或环境问题：** 向任务方指出受影响的算法要求、验收、设备、实现栈或资料绑定，取得明确结论后再推进依赖部分。

未形成实验依据的建议和一般实现问题保留在任务沟通中，不塞入替代反馈。上游设计修订后，重新确认版本与受影响代码、测试和旧证据的适用范围，再继续实现。
