# 编写和检查 schema YAML

本篇提供创建、修订、检查与交付 schema YAML 的共用操作。任务方法分别见[提取](../EXTRACT_SCHEMA.md)、[规划](../PLAN_SCHEMA.md)和[实现](../IMPLEMENT_KERNEL.md)；按当前需要进入下面的小节，字段语义与设计指导仍以 schema_docs 为准。

## 填写与检查

### 1 确认资料与范围

开始前需要任务要求和目标硬件条件：输入输出与数学定义、参考入口和误差判据、目标设备、已选实现栈及相应资料。还需从任务说明取得实际 workload 和优化目标；当前规范尚未确定它们的字段格式，不自行增加字段。

Contract 使用任务给定或经任务方确认的内容；转录已确认的要求不等于获得修改要求的权限。需要改变任务边界、验收判据、设备或实现栈时，先向任务方指出问题，遵守[修改权限](../README.md#2-schema-是什么)。参考入口尚未形成时，按 [golden_reference 规则](../schema_docs/1.contract.algorithm_requirement.md#golden_reference)处理；不能用设计源码冒充参考入口。

必要信息不足时，在任务沟通中说明缺什么、影响哪个字段或设计判断、需要哪些资料或确认；可以继续不受影响的部分，但不能把草稿当成完整设计交付。不要编造参考入口、误差阈值、硬件能力或实验结果，也不要用 `null` 掩盖未完成的设计。只有[硬件资料入口](../schema_docs/2.contract.hardware_specification.md#省略与占位)和[数学区允许的位置](../schema_docs/3.design.math_reshape.md#引用与缺项)可按规定标记缺口；这些标记不表示相应判断已经成立。

### 2 创建文件或进入已有任务

新任务使用独立目录，按[任务身份规则](../schema_docs/0.common.md#单份-plan-与元数据)命名。下面的 `path/to/task` 替换为实际目录，例如 `schema_wiki/<batch>/<task>`；任务也可以位于仓库之外。从仓库根目录复制骨架，不覆盖已有文件：

```sh
mkdir -p path/to/task
cp -n schema_template/starter.yaml path/to/task/root.yaml
```

[starter.yaml](starter.yaml) 是编辑起点，不是完整字段清单或默认设计。按任务重命名 `Input / Output / Dimension / main / Main` 等名称并维护引用，将 `null` 换成真实内容；步骤、phase、launch、lane 和对象条目均按设计增删。骨架中的 `version: 1` 是新文件的起始修订号，不是已有文件的重置值。

不能只替换占位值就认为完成：按各篇“填写条件”补齐适用字段，没有内容的可选字段直接省略。例如提取设计需要登记真实的 `meta_data.source`；选择 `refill: none` 时还需补上骨架未列出的 `preassigned`。`Dimension` 表示待确定的独立工作维度；整个任务只有一份工作时，使用 `partition: {}` 并省略 `worktile_order`，见[工作分解](../schema_docs/4.design.work_schedule.md#工作分解)与[工作分配](../schema_docs/4.design.work_schedule.md#工作排列与分配)。

已有任务先读 root 的共享 Contract、目标节点和其他节点的选择条件，再确定影响范围。同一路径的设计改进在原文件修订，不重新复制 starter。遇到新的 workload，先考虑能否完善已有设计；能由同一份 Design 的参数或公式表达的变化留在原节点，例如同一套公式覆盖 cluster 成员数 4、8、16 时，不按成员数拆分。

确实需要并存的不同设计时，从最相关节点复制为新的 `<id>.yaml`，按[节点规则](../schema_docs/0.common.md#派生节点与-dispatch)设置身份、父节点和起始版本，删除复制来的 `contract`，完整填写自己的 Design，重新核对复制来的 `realization` 及反馈是否适用。同步核对设计出处、各节点的选择条件，并维护 `dispatch.md` 的派生树和链接。拆分应有设计理由，每多一条路径都增加维护和验收工作。

### 3 填写与修订

新文件先落实 `meta_data`；root 转录已确认的两份 Contract，派生节点使用共享 Contract。Design 的形成与往返核对按[提取入口](../EXTRACT_SCHEMA.md#3-从实现事实形成设计表达)或[规划入口](../PLAN_SCHEMA.md#3-在数学与执行之间迭代)推进，不把 YAML 的排列顺序当成只填写一遍的步骤。

修改时先改决定的唯一定义，再沿引用检查受影响的下游：数学值、shape、dtype 或寿命的变化可能影响分块、资源和实现约束；phase、lane、动作或资源改名时核对 flow、handoff、milestone、edge 与 realization 的引用。共享 Contract 经任务方确认发生变化后，要审阅所有节点是否仍符合它，不能只修当前节点。Coder 追加替代反馈时按[反馈规则](../schema_docs/6.realization.md#有证据的替代反馈)保留原选择与理由；涉及 Design 的变化交给 Designer。

按[版本规则](../schema_docs/0.common.md#单份-plan-与元数据)更新实际改动文件的修订号，结合 diff 核对历史递增；检查器只检查它是否为正整数。改动内容由 diff 给出，设计意图仍写在所属 `thesis` 或实现选择的 `reason` 中。实验记录用 `repository@ref:path` 固定它验证的 plan，同一 ref 同时固定该节点与 root；改过设计后不能把旧实验自动视为新版本的证据。

### 4 检查与处理反馈

需要 Python 3.10+ 和 PyYAML。从仓库根目录运行唯一的日常检查命令，填写中和每次修订后都使用同一入口：

```sh
python3 schema_template/check_yaml.py path/to/task/root.yaml
```

也可传入派生节点。检查范围始终是该文件所在目录直属的全部 `.yaml / .yml`，不递归子目录；因此错误可能来自其他节点，目录内的备份 YAML 也会被当作任务节点。输入路径相对当前目录解析，在其他目录调用时使用检查脚本的绝对路径。

缺少 PyYAML 时，检查器会给出一条安装命令；安装后重跑同一检查命令。检查本身不下载或安装依赖。

成功时输出任务目录和节点数；失败时按文件、字段路径和修改提示处理，已有字段同时显示 YAML 行列位置，缺字段可能没有行号。先修结构问题，再处理具名关系；修改后重跑同一命令，前一轮错误可能阻止了后续检查。退出状态为：`0` 通过，`1` 输入未通过检查，`2` 参数或工具环境问题导致检查未完成。

检查不补字段、不重排、不改写 YAML，也不改变文件权限或修改时间。若诊断与文档允许的表达冲突，核对具体规则并报告冲突，不为通过检查删除必要设计信息。人工审阅或实现反馈指出问题时，回到上一步确定修改范围，再检查整个任务。

### 5 交付

交付任务目录：`root.yaml`、存在的派生节点，以及多节点任务的 `dispatch.md`；只有 root 时索引可选。派生文件依赖共享 Contract，不作为脱离 root 的完整 plan 交付。源码、硬件资料和实验记录保留在各自实际位置，通过规定的引用定位，不复制进 YAML。

任务冻结时，将准备阶段用 tag 或 branch 定位的硬件与编程等 wiki 资料解析到 commit，按 Contract 修改权限确认后填回原 `ref`；引用更新也按版本规则修订 root，并重跑同一检查命令。核对设计出处、参考入口及实验记录均指向实际对象和所声明的版本，而不只是字符串格式合法。

交付时简要说明改动范围、检查结果及仍未解决的资料、设计或运行验证问题，可以写在任务消息中，无需新增固定报告文件或 YAML 状态字段。若必需信息或设计尚未确定，明确交付的是草稿及其缺口；检查通过不能改变这一状态。

## 校验边界

当前关联检查包括节点与 phase 身份、显式 Contract 属性引用、launch/handoff 的 phase、事件及局部资源引用，以及动作族匹配。`realization.item`、数学区的其他语义绑定，以及公式、strategy 和动作契约中的引用仍需人工解析；改名后应沿这些位置核对。

通过检查表示结构及已实现的关联检查通过，不证明数学等价、表达式正确、数值正确、资源可行、存储复用安全、活性或性能。当前检查器不展开完整动作实例，动作引用通过也不证明每个索引处的动作存在。dispatch 条件的互斥与覆盖、外部资料、修改权限、修订历史和实验依据仍需审阅；实际实现是否满足数值、执行约束及性能目标，需要相应运行证据。

wiki 占位和 `unknown:` 仍是待解决的缺口，不能靠猜测补齐，也不能把结构通过称为资料或设计已验收。[examples](examples/README.md) 仅作可选参考，不是学习规范的前提。维护检查器时再读 [tools/README.md](tools/README.md)。
