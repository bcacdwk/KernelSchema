# P2：dispatch 草案

本页先确定代表路径和拆分依据，再让 root 进入 Design。阶段调用与检查约定见[手册](../README.md)。

以下准入、推进与重开用于普通提取；`maintenance` 只执行本次任务，写入还受调用的文件/字段授权限制，不能据本页扩大范围。

## 准入与应读材料

P1 的元数据和共享 Contract 已通过，Runner 提供对应 root checkpoint、固定 host 入口、适用 workload、调用身份和可写范围。worker 续用 P1 会话，reviewer 使用新的阶段会话，明确读取本页；P2 独立验收和计退修。

worker 从当前 root 草稿接续；reviewer 按[当前文档验收](../REVIEW.md)先读当前产物，再沿固定 host 选择条件、参数归一和实际调用关系核对路径。两角色均依[派生与索引规则](../../schema_docs/0.common.md#派生节点与-dispatch)工作，只深入到足以区分设计，不先完成整棵树的 Design。需要任务边界依据时回查固定材料。

## worker：可写范围与操作

可写 `dispatch.md` 草案及 root 的 `meta_data.dispatch`，内容修订时按规则更新 root 版本。root 的共享 Contract、来源事实和 Design 保持已知内容；发现需要改变它们时经 Runner 转回对应阶段。不得创建派生 YAML、改源码或运行状态。

1. 在共享 Contract 的完整有效域内列出真实 host 路径及选择依据，按[计算执行实例](../../schema_docs/0.common.md#plan-与计算执行实例)区分仅录制的结构与实际执行。区分数学改写、工作拆分、中间结果物化和局部执行组织的变化；合并路径前，先依[参数与结构的分组规则](../../schema_docs/0.common.md#参数变化与结构变化)核实实际提交集合、跨 phase 生产者和完成关系。同一套合法参数或公式能表达的变化留在一个节点，不把需要正文删改结构的路径提前合并。函数、文件、kernel 或 shape 的数量均不等于节点数。
2. 从真实路径中选择覆盖主要 workload、较通用且便于派生的 root。root 保留全任务 Contract，自身 Design 只负责选中范围；不另造覆盖全域的 baseline。
3. 用中文 `dispatch.md` 简述 root 的 workload、选择理由和预计需要的其他设计及差异。链接 root；尚未生成的派生节点明确标为拟议，不假装已有 YAML。选择证据、覆盖缺口和后续需要核实的问题写进回执，精确条件不在索引中重复维护。
4. 有多路径时，按公共约定填写 root 的独立 `when`。单 root 通常省略 dispatch；显式保留时说明覆盖全契约域。`parent` 只表示设计派生来源，各节点条件在全域独立求值，不从祖先叠加。派生节点的精确条件待其 YAML 创建时写入。

已知实现覆盖不足要准确记录，不能缩小 Contract 或虚构 fallback。硬件占位与 GPU 边界沿用 [P1](01_contract.md#第-3-段硬件占位)；非硬件缺项按[总览规则](../../EXTRACT_SCHEMA.md#来源缺项与验收边界)定位。

## reviewer：放行门槛

按[当前文档验收](../REVIEW.md)覆盖 P1 已完成内容、当前选择条件及索引草案。独立追到真实 host 路由，判断 root 是否代表主要 workload，划分是否用最少必要节点保留数学、工作组织或局部执行的实质差异。核对主要路径及 root `when`、全契约域内的选择依据和覆盖缺口。

本阶段批准 root 选择和划分草案，允许只有拟议分支、尚无分支 YAML。检查草案可读、root 链接有效、改动未越界；不因 Design 未完成要求完整 YAML 检查退出 `0`，也不能把本阶段尚未完成的 partial 草案通过维护复核当作完成。普通调用若有证据推翻 P1，按本轮 `reopen_targets` 交 Runner 回 P1 核实。

## 回执与下一位

按[回执约定](../tools/README.md#回执与短交接)提交 root 选择、主要路径与实质差异、host 依据及覆盖缺口。reviewer 的 P2→P3 短交接只保留已定理由和待核实入口，由 Runner 保存。

本阶段放行后，Runner 启动 [P3](03_root.md) 的全新 worker/reviewer。worker 读取已审 root 草稿、dispatch 草案、短交接和选定路径源码，开始逐段设计；reviewer 从当前产物独立审阅。
