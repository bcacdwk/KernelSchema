# P4：dispatch 定稿

本页将路径草案变成可据以派发节点的计划。

## 准入与应读材料

P3 完整 root 已通过检查与独立审阅，P1、P2 的放行仍有效。Runner 提供已审 root、dispatch 草案、P3 短交接、固定 host 选择入口、当前计划修订与可写范围。首次 P4 新开一对 worker/reviewer；因新证据重开时读取受影响节点、现有清单和重开原因，继续原计数与预算。

先读上述产物和[节点与索引规则](../../schema_docs/0.common.md#派生节点与-dispatch)，再核对 host 路由和各路径实际入口；按差异回查 P3 设计及源码。索引格式见本页最小示例，树和节点数由固定源码决定。

## worker：可写范围与操作

可修改 `dispatch.md`，并通过回执提交结构化节点清单。root 和已审派生 YAML 只读；不提前创建 P5 节点，不把运行计划历史、退修或完成数写进索引。不允许直接修改 Runner 状态。

1. 据真实 root 设计和 host 路由重审分组；同一套参数能覆盖的设计保留在一个节点，实质设计差异才派生。每个节点确定稳定 ID、预期文件名、直接父节点、workload、相对父节点的变化和选择依据。
2. 明确 `parent` 是设计复制来源，条件在共享契约全域独立求值；父子并存路径的 `when` 不能被理解为条件嵌套。派生精确条件由 P5 写入各 YAML，选择依据在清单中足够定位对应 host 分支及必要派生量。
3. 用中文定稿 `dispatch.md`，按下面的固定表格保存当前树的直接父关系、各节点相对链接、workload、设计差异和选择理由；表格逐项取自本次 plan。精确条件和版本以 YAML 为准；尚未生成的计划内文件允许暂缺，P5 完成后由 P6 核对实有链接。只有 root 的任务也保留选择理由和实际覆盖缺口，不重复流水线正文或通用规范。
4. 回执提供逐节点清单：`id/file/parent/workload/difference/selection/evidence`，root 的 `parent` 为空。该清单供 Runner 校验和调度，不新建另一份项目配置或由 worker 写运行状态。`evidence` 是本回执 `source_evidence` 的索引，至少包括 host 依据。

索引从以下标题和表头开始，root 在首行，其余行按 ID 排序；单元格中的 `\` 和 `|` 分别转义为 `\\` 和 `\|`，换行写成 `<br>`。直接父列的 root 使用 `无`。Runner 确定性核对整张表格；不另写一份可能与它矛盾的树。表后用 `## 覆盖缺口` 记录实际缺口，没有已知缺口也明确写明。比如 host 对未对齐输入的拒绝必须保留，不能据一条路径推断 fallback。

```markdown
# 路径索引

| 节点 | 直接父 | workload | 相对父差异 | 选择依据 |
| --- | --- | --- | --- | --- |
| [root](root.yaml) | 无 | plan.root.workload | plan.root.difference | plan.root.selection |

## 覆盖缺口

据本次源码核实的具体缺口和影响范围。
```

重开 P4 时列明新增、删除、换父或受条件变化影响的节点及证据，交 Runner 更新计划修订和有效结论。要改选代表路径、root Design/条件或共享 Contract 时，按[计划修订入口](../tools/README.md#并行合入与计划修订)回对应阶段，不能在 P4 顺手改 root。已知覆盖不足如实保留，不虚构 fallback；其他缺项和硬件边界沿用[总览](../../EXTRACT_SCHEMA.md#来源缺项与验收边界)。

## reviewer：放行门槛

独立对照 host 路由和已审 root，确认整棵树以最少必要节点保留真实设计差异，直接父节点能提供合理的复制起点。每个条目的 workload、差异和选择依据足以让新 worker 精确接手，覆盖缺口保持真实；清单与索引相同，root 文件未被越权改写。

批准时返回与 worker 相同的节点语义清单；自己的 `source_evidence` 顺序和 `evidence` 编号独立，不必复制 worker 的局部编号。重开时可从调用中列出的 `reviewed_artifacts` 读取此前已审 YAML，判断其是否仍适用。

Runner 对 complete/approve 检查唯一 root、ID 与文件身份唯一、父节点存在、无环且所有节点可达，并核对已审 root 的身份。多节点计划要求已有 root 的 `meta_data.dispatch.when` 非空，不能等 P5 创建节点后才发现。P4 不要求尚不存在的分支 YAML 通过检查；已有 root 沿用有效 P3 审阅。root 身份错误回 P2，现有 root 的 Design/when 缺失或错误按既有协议请求重开 P3；变更代表路径回 P2。P4/P5 仍无权改 root；不能获得上游修复时在 P4 明确阻塞，不派发子节点。其余结构或关键选择证据不满足门槛时，返回可执行退修或具体阻塞条件。

## 回执与下一位

按[回执约定](../tools/README.md#回执与短交接)提交经审阅的节点清单、划分与选父依据、覆盖缺口及影响范围。reviewer 的 P4→P5 短交接保留路径入口和各分支待核实的差异。

Runner 验证后保存当前计划，立即为直接父节点已审并合入的节点启动 [P5](05_branch.md)，每节点全新一对 worker/reviewer。没有派生节点时跳过 P5，仍由 [P6](06_final.md) 的新 reviewer 终审。
