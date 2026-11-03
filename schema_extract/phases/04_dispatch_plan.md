# P4：dispatch 定稿

本页将路径草案变成可据以派发节点的计划。

## 准入与应读材料

P3 完整 root 已通过检查与独立审阅，P1、P2 的放行仍有效。Runner 提供已审 root、dispatch 草案、P3 短交接、固定 host 选择入口、当前计划修订与可写范围。首次 P4 新开一对 worker/reviewer；因新证据重开时读取受影响节点、现有清单和重开原因，继续原计数与预算。

先读上述产物和[节点与索引规则](../../schema_docs/0.common.md#派生节点与-dispatch)，再核对 host 路由和各路径实际入口；按差异回查 P3 设计及源码。可选参考 [FA4 索引](../../schema_template/examples/fa4_prefill/dispatch.md)，不照抄其树或节点数。

## worker：可写范围与操作

可修改 `dispatch.md`，并通过回执提交结构化节点清单。root 和已审派生 YAML 只读；不提前创建 P5 节点，不把运行计划历史、退修或完成数写进索引。不允许直接修改 Runner 状态。

1. 据真实 root 设计和 host 路由重审分组；同一套参数能覆盖的设计保留在一个节点，实质设计差异才派生。每个节点确定稳定 ID、预期文件名、直接父节点、workload、相对父节点的变化和选择依据。
2. 明确 `parent` 是设计复制来源，条件在共享契约全域独立求值；父子并存路径的 `when` 不能被理解为条件嵌套。派生精确条件由 P5 写入各 YAML，选择依据在清单中足够定位对应 host 分支及必要派生量。
3. 用中文定稿 `dispatch.md`，只保留当前树、各节点相对链接、workload 与设计差异。精确条件和版本以 YAML 为准；尚未生成的计划内文件允许暂缺，P5 完成后由 P6 核对实有链接。只有 root 的任务也保留简短索引。
4. 回执提供逐节点清单：`id/file/parent/workload/difference/selection/evidence`，root 的 `parent` 为空。该清单供 Runner 校验和调度，不新建另一份项目配置或由 worker 写运行状态。`evidence` 是本回执 `source_evidence` 的索引，至少包括 host 依据。

重开 P4 时列明新增、删除、换父或受条件变化影响的节点及证据，交 Runner 更新计划修订和有效结论。要改选代表路径、root Design/条件或共享 Contract 时，按[计划修订入口](../README.md#并行合入与计划修订)回对应阶段，不能在 P4 顺手改 root。已知覆盖不足如实保留，不虚构 fallback；其他缺项和硬件边界沿用[总览](../../EXTRACT_SCHEMA.md#来源缺项与验收边界)。

## reviewer：放行门槛

独立核对整棵树的划分理由、直接父节点选择、主要路径、选择证据和已知覆盖缺口。确认清单与索引相同，节点差异能指导新 worker，从 root 到任意节点都有明确来源；root 文件未被越权改写。

批准时返回与 worker 相同的节点语义清单；自己的 `source_evidence` 顺序和 `evidence` 编号独立，不必复制 worker 的局部编号。重开时可从调用中列出的 `reviewed_artifacts` 读取此前已审 YAML，判断其是否仍适用。

Runner 接到批准后仍须检查唯一 root、ID 与文件身份唯一、父节点存在、无环且所有节点可达，再从清单得出分支数和依赖。P4 不要求尚不存在的分支 YAML 通过检查；已有 root 沿用有效 P3 审阅，本阶段门槛是索引与计划。若结构或关键选择证据不满足门槛，返回可执行退修或具体阻塞条件。

## 回执与下一位

按[回执约定](../README.md#回执与短交接)返回经审阅的结构化节点清单、树与来源核对、覆盖缺口及影响范围。reviewer 给出 P4→P5 短交接，保留选父理由、路径入口及分支需要核实的差异，由 Runner 保存。框架候选放 `framework_candidates`，退修与裁决遵循[统一规则](../README.md#退修与裁决)。

Runner 验证后保存当前计划，立即为直接父节点已审并合入的节点启动 [P5](05_branch.md)，每节点全新一对 worker/reviewer。没有派生节点时跳过 P5，仍由 [P6](06_final.md) 的新 reviewer 终审。
