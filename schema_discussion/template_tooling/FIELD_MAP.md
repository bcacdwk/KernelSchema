# 当前字段与名称

本表按路径记录 DOC0–6 与 Template 的唯一现行拼写，不按单词、值类型或任意同名键推断字段。机器表由 [`rules/structure.py`](../../schema_template/tools/rules/structure.py) 的现行检查器导出为 `CANONICAL_FIELDS`；`STRING_FIELDS` 同源列出非 null 标量只能是文本的叶位置，供原文规范化使用；YAML null 始终保留。两表不定义旧字段别名，也不放宽字段、类型或关系检查。

`<name>` 表示实例命名，`<STEP>`、`<ROLE>` 表示执行身份，`[]` 表示列表条目。机器表中的 `None` 对应具名映射成员，`int` 对应列表位置；表中的字段集合不规定展示顺序。必填、可省及合法值见对应 DOC。命名边界见[字段决定](auto_normalization/FIELD_DECISIONS.md)。

## 固定键

| 父路径 | 唯一固定字段 |
| --- | --- |
| 文档根 | `META_DATA / CONTRACT / DESIGN / REALIZATION` |
| `META_DATA` | `Name / Describe / Source / Version / Dispatch` |
| `META_DATA.Dispatch` | `id / when` |
| `CONTRACT` | `Algorithm_requirement / Hardware_specification` |
| `CONTRACT.Algorithm_requirement` | `Constant / Input / Output / Compute / Golden_reference` |
| `CONTRACT.Algorithm_requirement.Constant.<name>` | `value / meaning` |
| `CONTRACT.Algorithm_requirement.Input.<name>` | `meaning / shape / dtype / nullable / range / default` |
| `CONTRACT.Algorithm_requirement.Output.<name>` | `meaning / shape / dtype / nullable` |
| `CONTRACT.Algorithm_requirement.Compute[]` | `Name / Math / Symbol` |
| `CONTRACT.Algorithm_requirement.Compute[].Symbol.<name>` | `meaning / shape / dtype` |
| `CONTRACT.Algorithm_requirement.Golden_reference` | `Source / Tolerance / Gap` |
| `CONTRACT.Hardware_specification` | `Device / Architecture / Execution / Feature / Stack / Wiki` |
| `CONTRACT.Hardware_specification.Stack` | `Language / Template / Toolchain` |
| `CONTRACT.Hardware_specification.Wiki` | `Hardware / Programming / Microbench / Sota / Profile` |
| `CONTRACT.Hardware_specification.Wiki.Programming` | `Language / Template / Virtual_ISA / Machine_ISA` |
| `CONTRACT.Hardware_specification.Wiki.Sota` | `Kernel / Schema` |
| `DESIGN` | `Math_reshape / Work_schedule / Tile_pipeline` |
| `DESIGN.Math_reshape`（本地区域） | `Thesis / Rewrite` |
| `DESIGN.Math_reshape.Rewrite[]` | `Name / Math / Symbol / Arithmetic / Ledger` |
| `DESIGN.Math_reshape.Rewrite[].Symbol.<name>` | `meaning / shape / dtype / retention / binding / value` |
| `DESIGN.Math_reshape.Rewrite[].Arithmetic` | `Approx / Accumulate` |
| `DESIGN.Math_reshape.Rewrite[].Ledger` | `Baseline / Gain / Cost` |
| `DESIGN.Work_schedule`（本地区域） | `Thesis / Step / Flow` |
| `DESIGN.Work_schedule.Step.<STEP>` | `Describe / Decomposition / Mapping / Schedule` |
| `DESIGN.Work_schedule.Step.<STEP>.Decomposition` | `Strategy / Partition / Worktile / Target` |
| `DESIGN.Work_schedule.Step.<STEP>.Decomposition.Partition.<name>` | `meaning / size / tile / range` |
| `DESIGN.Work_schedule.Step.<STEP>.Decomposition.Worktile` | `Invariant / Iteration` |
| `DESIGN.Work_schedule.Step.<STEP>.Decomposition.Worktile.Invariant.<name>` | `meaning / shape / dtype / offset` |
| `DESIGN.Work_schedule.Step.<STEP>.Decomposition.Worktile.Iteration.<name>` | `meaning / shape / dtype / direction` |
| `DESIGN.Work_schedule.Step.<STEP>.Decomposition.Target` | `Single_tile / Tile_reduction` |
| `DESIGN.Work_schedule.Step.<STEP>.Decomposition.Target.Single_tile.<name>` | `meaning / shape / dtype` |
| `DESIGN.Work_schedule.Step.<STEP>.Decomposition.Target.Tile_reduction` | `dimension / input / output / method` |
| `DESIGN.Work_schedule.Step.<STEP>.Mapping` | `Strategy / Worker / Placement / Worker_count / Member` |
| `DESIGN.Work_schedule.Step.<STEP>.Mapping.Worker`、同层 `Placement` | `unit / unit_count` |
| `DESIGN.Work_schedule.Step.<STEP>.Schedule` | `Strategy / Worktile_order / Worker_assignment` |
| `DESIGN.Work_schedule.Step.<STEP>.Schedule.Worktile_order[]` | `by / direction / group` |
| `DESIGN.Work_schedule.Step.<STEP>.Schedule.Worker_assignment` | `Preassigned / Refill / Exit` |
| `DESIGN.Work_schedule.Step.<STEP>.Schedule.Worker_assignment.Preassigned` | `start / stride / count` |
| `DESIGN.Work_schedule.Flow` | `Launch / Handoff` |
| `DESIGN.Work_schedule.Flow.Launch[]` | `step / order / start_after` |
| `DESIGN.Work_schedule.Flow.Handoff.<name>` | `producer / consumer / storage / initialize / ready / wait_at` |
| `DESIGN.Tile_pipeline`（本地区域） | `Thesis / Step` |
| `DESIGN.Tile_pipeline.Step.<STEP>` | `Index / Event / Storage / Role` |
| `DESIGN.Tile_pipeline.Step.<STEP>.Index.<name>` | `meaning / definition` |
| `DESIGN.Tile_pipeline.Step.<STEP>.Storage.<name>` | `meaning / depth / location / shape / dtype / total_size / member / view` |
| `DESIGN.Tile_pipeline.Step.<STEP>.Storage.<name>.view.<value>` | `slot_index / region` |
| `DESIGN.Tile_pipeline.Step.<STEP>.Role.<ROLE>` | `Info / Pipe / Waiting` |
| `DESIGN.Tile_pipeline.Step.<STEP>.Role.<ROLE>.Info` | `Describe / Member / Register / Using_verb` |
| `DESIGN.Tile_pipeline.Step.<STEP>.Role.<ROLE>.Info.Using_verb.<verb>.<noun>` | `execution_mode / describe` |
| `DESIGN.Tile_pipeline.Step.<STEP>.Role.<ROLE>.Waiting` | `Prepared / Released / Allowed` |
| `DESIGN.<region>`（整区引用映射） | `from` |
| `REALIZATION` | `Require / Prefer / Forbid / Feedback` |
| `REALIZATION.Require[]`、`REALIZATION.Prefer[]`、`REALIZATION.Forbid[]` | `Item / Choice / Reason` |
| `REALIZATION.Feedback[]` | `Item / Choice / Evidence` |

同名词按精确路径解释：`Mapping.Member` 与 `Info.Member` 是固定字段，`Storage.<name>.member` 仍是存储成员属性；`Info.Describe` 与 noun 的 `describe` 分开定义。`view` 是存储条目中的具名映射，`region` 是每个视图的坐标属性。`Waiting` 的三个分类是字段，`Using_verb` 的八个动作族键和 Index 域中的 `flatten` 保留原语法。

## 名称、语法与引用

| 位置 | 当前规则 |
| --- | --- |
| 两区的 `Step.<STEP>`、`Role.<ROLE>` | 英文字母、数字、下划线，全大写且不以数字开头；单个身份使用 `MAIN`；全部引用一致 |
| Constant、Input、Output、Symbol、Partition、Invariant、Iteration、Single_tile、Handoff、Index、Storage、view、Event 的具名键；shape/region/member 的轴；Refill 机制名 | 自取标识全小写且不以数字开头；索引、视图与事件形参按各自语法；沿同一已定义身份同步数学、动作与引用 |
| 数学规模、参数、局部变量、函数与动作 noun | 自取单名全小写；复合矩阵乘按 `a@b` 登记，调用按 `matmul:a(i)@b(i)`；外部属性链与真实函数不凭大小写猜改 |
| `META_DATA.Dispatch.id` 与整区 `from` 的值 | 任务目录内的小写节点 ID，与文件名一致；root 规则不变，不由文本规范化改动文件或来源身份 |
| `Execution.<unit>`、`Feature.<name>` | 登记键全部小写，合法空格、点和斜杠仍属于完整名称；同步 Worker/Placement.unit、Info.Member 的单元段、Storage.location 与已知限定引用 |
| `Using_verb` 的动作族键与 `Pipe` 动作 | `compute / matmul / load / store / await / signal / acquire / release` |
| `Pipe` 控制及 Index 域语法 | `for(...) / if(...) / {flatten: [...]} / worktile` 保持当前语法 |
| 等待时点与封闭枚举 | `issue / done`、三种 execution_mode、order、direction、`none`、`assigned_done` 等保持原取值 |
| 普通填写值 | `Name / Describe / Math / Choice` 的正文不按字段名改名；真实设备型号、架构值、dtype、仓库、路径、指令和外部函数保持实际拼写 |

字段 `Mapping.Worker_count` 的改名不改变已有派生数学量 `worker_count`；`worker_id / worktile_count` 等派生量同样保持定义。`ref:`、`bind:`、`unknown:` 与 `::`、事件、动作、成员选择记法不变。例如 `ref:CONTRACT.Algorithm_requirement::s0.shape`、`bind:DESIGN.Work_schedule::bk`、`DESIGN.Tile_pipeline::MAIN.MMA`、`Tile_pipeline.Step.MAIN.Storage.accumulator`。

引用先匹配已有完整名称或路径；未结构化目标只使用已有标识、索引或限定引用，解释放在说明字段中。retention 只写方式、invocation 或非空括号实例及已有迭代范围。仓库定位符保留真实相对路径中的合法空格和括号，每条只定位一个目标，不带首尾空白或附加说明。

## 内容与规范化

`Ledger` 可省略；出现时保留 `Baseline` 键，`Baseline / Gain / Cost` 各允许 YAML null；空值不表示零变化。数值表达式保留 Unicode 数学和原字段特殊记法，拒绝纯空白、全标点及全运算符。`range / default` 的已定义表达式继续合法。

填写设计后运行 `check_yaml`；工具统一可确定的文本形式并报告修改，需要精简或判断的内容给出反馈。字符与逻辑行预算由工具规则维护，原有预算不因改名改变。映射布局不改变字段与值，列表顺序和正文逻辑换行保留其含义。
