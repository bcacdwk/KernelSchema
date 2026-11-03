# P2：dispatch 草案

本页先确定代表路径和拆分依据，再让 root 进入 Design。阶段调用与检查约定见[手册](../README.md)。

## 准入与应读材料

P1 的元数据和共享 Contract 已通过，Runner 提供对应 root checkpoint、固定 host 入口、适用 workload、调用身份和可写范围。worker/reviewer 分别续用 P1 的原角色会话，明确读取本页；P2 独立验收和计退修。

先读已审 `root.yaml`、P1 的范围结论，再读[派生与索引规则](../../schema_docs/0.common.md#派生节点与-dispatch)。沿固定 host 选择条件、参数归一和实际调用关系找到各路径入口；只深入到足以区分设计，不先完成整棵树的 Design。需要任务边界依据时回查 P1 材料。

## worker：可写范围与操作

可写 `dispatch.md` 草案及 root 的 `meta_data.dispatch`，内容修订时按规则更新 root 版本。root 的共享 Contract、来源事实和 Design 保持已知内容；发现需要改变它们时经 Runner 转回对应阶段。不得创建派生 YAML、改源码或运行状态。

1. 在共享 Contract 的完整有效域内列出真实 host 路径及选择依据。区分数学改写、工作拆分、中间结果物化和局部执行组织的变化；同一套参数或公式能表达的变化留在一个节点。函数、文件、kernel 或 shape 的数量均不等于节点数。
2. 从真实路径中选择覆盖主要 workload、较通用且便于派生的 root。root 保留全任务 Contract，自身 Design 只负责选中范围；不另造覆盖全域的 baseline。
3. 用中文 `dispatch.md` 简述 root 的 workload、选择理由和预计需要的其他设计及差异。链接 root；尚未生成的派生节点明确标为拟议，不假装已有 YAML。选择证据、覆盖缺口和后续需要核实的问题写进回执，精确条件不在索引中重复维护。
4. 有多路径时，按公共约定填写 root 的独立 `when`。单 root 通常省略 dispatch；显式保留时说明覆盖全契约域。`parent` 只表示设计派生来源，各节点条件在全域独立求值，不从祖先叠加。派生节点的精确条件待其 YAML 创建时写入。

已知实现覆盖不足要准确记录，不能缩小 Contract 或虚构 fallback。硬件占位与 GPU 边界沿用 [P1](01_contract.md#第-3-段硬件占位)；非硬件缺项按[总览规则](../../EXTRACT_SCHEMA.md#来源缺项与验收边界)定位。

## reviewer：放行门槛

对照 P1 结论、host 路由和调用证据，确认 root 来自真实实现且能代表主要 workload；主要路径均已识别，拆分确有设计理由，参数变化未被过度拆分。核对 root `when` 与已知路由一致、没有收窄共享 Contract；实现覆盖缺口和仍待核实的路径已明确。

本阶段批准 root 选择和划分草案，允许只有拟议分支、尚无分支 YAML。检查草案可读、root 链接有效、改动未越界；不因 Design 仍未完成要求完整 YAML 检查退出 `0`。若新证据推翻 P1 的外部要求，交 Runner 回 P1 核实。

## 回执与下一位

按[回执约定](../README.md#回执与短交接)返回 root 选择及理由、主要路径与实质差异、host 文件行号和短引文、已知覆盖缺口及仍需核实的入口。reviewer 同时给出 P2→P3 短交接，由 Runner 保存；不要把计划历史或退修记录塞进索引。框架候选放 `framework_candidates`，退修与裁决遵循[统一规则](../README.md#退修与裁决)。

本阶段放行后，Runner 启动 [P3](03_root.md) 的全新 worker/reviewer。接手者读取已审 root 草稿、dispatch 草案、短交接和选定路径源码，开始逐段设计。
