# 本轮字段决定

本表记录已确认的唯一字段拼写及小写属性边界，与 DOC0–6、[FIELD_MAP](../FIELD_MAP.md)、`tools/rules/structure.py`、`hardware.py` 一致。结构检查、文本预算、源恢复、格式化、starter 与最小测试样例使用同一套决定。

路径缩写：`A = CONTRACT.Algorithm_requirement`，`H = CONTRACT.Hardware_specification`，`M = DESIGN.Math_reshape`，`W = DESIGN.Work_schedule`，`P = DESIGN.Tile_pipeline`。`[]` 是列表位置，`<name>` 是具名映射键；`<STEP>`、`<ROLE>` 是执行身份。单元格中的 `/` 分隔同一父路径下的独立字段。

## 明确已定

| 父路径 | 唯一字段拼写 |
| --- | --- |
| `META_DATA` | `Name / Describe / Source / Version / Dispatch` |
| `META_DATA.Dispatch` | `id / when` |
| `A.Compute[]` | `Name / Math / Symbol` |
| `A.Golden_reference` | `Source / Tolerance / Gap` |
| `H` | `Device / Architecture / Execution / Feature / Stack / Wiki` |
| `H.Wiki.Programming` | `Language / Template / Virtual_ISA / Machine_ISA` |
| `M` 本地定义 | `Thesis / Rewrite` |
| `M.Rewrite[]` | `Name / Math / Symbol / Ledger / Arithmetic` |
| `M.Rewrite[].Arithmetic` | `Approx / Accumulate` |
| `M.Rewrite[].Ledger` | `Baseline / Gain / Cost` |
| `W` 本地定义 | `Thesis / Step / Flow` |
| `W.Step.<STEP>` | `Describe / Decomposition / Mapping / Schedule` |
| `W.Step.<STEP>.Decomposition` | `Strategy / Partition / Worktile / Target` |
| `W.Step.<STEP>.Mapping` | `Strategy / Worker / Placement / Worker_count / Member` |
| `W.Step.<STEP>.Mapping.Worker`、同层 `Placement` | `unit / unit_count` |
| `W.Step.<STEP>.Schedule` | `Strategy / Worktile_order / Worker_assignment` |
| `W.Step.<STEP>.Schedule.Worktile_order[]` | `by / direction / group` |
| `W.Step.<STEP>.Schedule.Worker_assignment` | `Preassigned / Refill / Exit` |
| `W.Flow.Launch[]` | `step / order / start_after` |
| `P` 本地定义 | `Thesis / Step` |
| `P.Step.<STEP>.Role.<ROLE>.Info` | `Describe / Member / Register / Using_verb` |
| `REALIZATION.Require[] / Prefer[] / Forbid[]` | `Item / Choice / Reason` |
| `REALIZATION.Feedback[]` | `Item / Choice / Evidence` |

`Name` 的现存固定路径只有 `META_DATA.Name`、`A.Compute[].Name`、`M.Rewrite[].Name`，三处均已定；不另设“entry 一律使用 name”的规则。`Flow.Launch[]` 当前只有表中三个字段，没有未列出的“等等”。四个顶层键以及表中未改动的现行结构分区、登记表、资料类别保持原拼写。

以下既有数据属性保持当前拼写；不把上表的 `Member / Describe` 扩散到所有同词属性。

| 父路径 | 保留的小写属性 |
| --- | --- |
| `A.Constant.<name>` | `value / meaning` |
| `A.Input.<name>` | `meaning / shape / dtype / nullable / range / default` |
| `A.Output.<name>` | `meaning / shape / dtype / nullable` |
| `A.Compute[].Symbol.<name>` | `meaning / shape / dtype` |
| `M.Rewrite[].Symbol.<name>` | `meaning / shape / dtype / retention / binding / value` |
| `W.Step.<STEP>.Decomposition.Partition.<name>` | `meaning / size / tile / range` |
| `W.Step.<STEP>.Decomposition.Worktile.Invariant.<name>` | `meaning / shape / dtype / offset` |
| `W.Step.<STEP>.Decomposition.Worktile.Iteration.<name>` | `meaning / shape / dtype / direction` |
| `W.Step.<STEP>.Decomposition.Target.Single_tile.<name>` | `meaning / shape / dtype` |
| `W.Step.<STEP>.Decomposition.Target.Tile_reduction` | `dimension / input / output / method` |
| `W.Step.<STEP>.Schedule.Worker_assignment.Preassigned` | `start / stride / count` |
| `W.Flow.Handoff.<name>` | `producer / consumer / storage / initialize / ready / wait_at` |
| `P.Step.<STEP>.Index.<name>` | `meaning / definition` |
| `P.Step.<STEP>.Storage.<name>` | `meaning / depth / location / shape / dtype / total_size / member / view` |
| `P.Step.<STEP>.Storage.<name>.view.<value>` | `slot_index / region` |
| `P.Step.<STEP>.Role.<ROLE>.Info.Using_verb.<verb>.<noun>` | `execution_mode / describe` |
| `DESIGN.Math_reshape`、`DESIGN.Work_schedule`、`DESIGN.Tile_pipeline` 的整区引用映射 | `from` |

名称与引用按下表同步，不由任意字符串内容猜测身份。

| 声明位置或名称类别 | 规范目标与同步范围 |
| --- | --- |
| `W.Step.<STEP>`、`P.Step.<STEP>`、`P.Step.<STEP>.Role.<ROLE>` | 身份全大写；单一身份使用 `MAIN`；同步 Launch、Handoff、事件、Waiting、存储路径及限定引用中的对应身份 |
| `A.Constant/Input/Output`、两处 `Symbol`、`Partition`、`Invariant`、`Iteration`、`Single_tile`、`Handoff` 的具名键 | 名称全小写；沿同一已定义身份同步数学与引用，`Handoff` 键仍引用上游数学值 |
| `A.Golden_reference.Tolerance.<output>` | 沿用小写输出名，不生成另一套输出身份 |
| `P` 的 `Index / Storage / Event / view` 具名键；shape、region、成员区域轴 | 自取名称全小写；Event/view 的基名、形参按各自语法同步 |
| `Info.Using_verb.<verb>.<noun>`、数学变量／参数／函数、`Refill.<mechanism>` | 自取名称全小写；复合 noun 分别处理操作数基名，调用索引、作用域与表达式语义保留 |
| `META_DATA.Dispatch.id`、整区引用的节点 ID | 小写，与当前任务文件身份一致；`root` 规则保留 |
| `H.Execution.<unit>`、`H.Feature.<name>` | 登记名称全部小写；点、斜杠及合法空格仍属于完整名称；同步 `Worker.unit`、`Placement.unit`、`Info.Member` 的单元段、`Storage.location` 及其他已知身份引用 |
| 动作、控制、域和封闭值 | `compute / matmul / load / store / await / signal / acquire / release`、`for / if`、`flatten / worktile`、`issue / done` 及既有枚举保留当前语法 |
| 其他普通填写值 | `Name / Describe / Math / Choice` 的正文不按字段名套标识符规则；实际设备型号、架构值、类型、仓库定位符与外部函数／指令拼写保留 |

规范字段只发布一个拼写；工具可规范化该拼写的大小写，不建立不同单词或层级的兼容别名，例如不把 `arch` 作为 `Architecture` 的运行时别名。身份规范化须保持声明与使用一致；重名碰撞、歧义或无法可靠对应的引用不能猜改。

## 文档与机器定义

各章结构轮廓、字段表、正文路径、内嵌片段与跨章链接使用上述拼写。文档定义字段含义和引用语义；字符预算与排版分支由工具维护，示例展示规范化后的布局。

机器规范化从 [`tools/rules/structure.py`](../../../schema_template/tools/rules/structure.py) 导入 `CANONICAL_FIELDS`：每项为 `(父路径 tuple, 固定键 tuple)`，`None` 表示具名映射成员、`int` 表示列表位置，路径不含任务 ID。它直接来自现行 `record / named / sequence` 检查器，不包含别名，也不规定排版顺序。`STRING_FIELDS` 由相同定义提供非 null 标量只能是文本的叶位置；数值、shape 和域不据此猜成字符串。账本的可空文本叶可参与非 null 原文恢复，YAML null 始终保留。

`Mapping.Worker_count` 更名后，已有派生数学变量 `worker_count` 仍保持小写，并从该字段取值。

| 文档 | 当前内容 |
| --- | --- |
| DOC0 | 元数据与整区引用的唯一拼写；通用写法集中交代工具职责 |
| DOC1 | `Compute[].Name / Math`、`Golden_reference.Source / Gap`；保留数学、类型、引用与验收语义 |
| DOC2 | `Device / Architecture`、两处 `ISA`；登记身份及引用使用小写，实际设备、架构与外部指令保持原名 |
| DOC3 | `Thesis`、Rewrite、`Arithmetic.Approx / Accumulate`、`Ledger.Baseline / Gain / Cost`；Ledger 空值语义与数学换行含义 |
| DOC4 | Step、三个 `Strategy`、`Worker_count / Member`、`Exit`、Launch；Tile_reduction、Preassigned、Handoff 的小写属性及执行语义 |
| DOC5 | `Thesis`、`Info.Describe / Member / Register`、`Storage.view`；Storage 两行示例，Index 与 noun 的小写属性，事件、索引、范围与等待语法 |
| DOC6 | `Item / Choice / Reason / Evidence`；同步对 `Architecture`、其他上游对象的引用 |

公共工具说明只需集中表述一次：`check_yaml` 在检查过程中自动规范化可确定的大小写、必要引号、末句号与布局，并以 warning 列出修改供复核；超长度等无法自动保全语义的情况报 error。字段、引用、类型和合法性检查继续执行。字符预算与具体排版分支由工具规则维护，不在作者规范各处重复。
