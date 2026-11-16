# 拟议文本规则变更

仅限常见扩写入口，既有 `300/2` 保持。下表是字符串的 Unicode 字符／逻辑行预算，含前缀与空格；按值分支择一适用，允许合法 YAML 物理折行。

| 修改对象 | 新增预算与粒度 | 自然扩写风险 |
| --- | --- | --- |
| Schema 自取对象、数学变量／参数／函数、轴、noun、索引、View/Event 基名、step/role 名 | **50/1，每个单名**；可解析的直接查，其余作为命名约定 | 把描述塞进名字 |
| 普通 dtype；Storage.location；Hardware_specification.Execution/Feature 登记键 | **100/1**，按字段位置统一选限额，不叠加单名 50，不靠自称 external | 名称后夹说明 |
| Algorithm_requirement.Constant.value；Math_reshape 的 Symbol.value | **100/1，每个字符串叶子**，数组递归逐叶子计 | 数值后写推导 |
| 结构化数值字段的表达式：shape 大小、size/tile/depth/total_size、计数、offset/slot_index、Partition/Index/View 区间端点 | **100/1，每个表达式或端点** | 表达式混入解释 |
| Hardware_specification.Stack.Language/Template/Toolchain | **150/1，每个叶子** | 环境摘要变教程 |
| Math_reshape 的 Symbol.retention | **150/1，整条** | 寿命说明变长段 |
| Work_schedule 的 Worker_assignment.exit、Tile_reduction.input/output/method、Handoff.initialize 字符串；Math_reshape 的 Symbol.binding 内部语义引用 | **300/2，每条字符串**，允许列表处逐元素；引用本身沿用现有语法 | 引用后夹解释 |
| META_DATA.source 的自由材料摘要文本（非纯定位符） | **300/2，整个摘要** | 来源扩成综述 |

| 修改对象 | 拟改形式／布局 |
| --- | --- |
| 数值表达式、range 端点、固定值字符串 | 拒绝空白、全标点或全运算符串，如 `，`、`...`、`++`；保留 Unicode 表达式、数值/range 中有符号或无符号的 `∞`、值语境的 `∅`、事件 `*` 和省略切片端点；整数不接受无穷 |
| Symbol.retention | 在 carry_forward/keep_value 后检查 `invocation` 或括号实例，可跟现有迭代范围；只放方式、实例与范围 |
| 引用、类型／硬件身份 | 引用优先匹配已有完整名称／路径，未结构化目标只用现有标识、索引或限定引用；身份只写名称。可确定的游离说明报错，保留原名合法空格 |
| 仓库定位符 | 每条一个目标，不附说明；拒绝空路径与路径首尾空白，使用仓库相对路径，保留内部合法空格／括号。有本地原件则核对路径／入口，不新增联网验证 |
| 新限额与形式检查的报错 | 超限及确定形式错误报 error；真实外部标识与路径不截断、不为凑上限改名，确实超限时依据原材料调整对应字段规则 |
| Work_schedule 的 Mapping.Worker/Placement | 各小记录压为一行，如 `Worker: {unit: CTA, unit_count: 2}`；Mapping 展开，strategy 单独一行 |
| Work_schedule 的 Schedule.Worktile_order | 每 entry 行内；单 entry 与父字段合行，多 entry 各列一行；by 列表行内 |
| Worker_assignment.Preassigned | start/stride 或 start/count 整记录行内 |
| Worker_assignment.Refill | 一机制时整映射行内；≥2 机制时父映射展开，每机制一行 |
| Worker_assignment.exit 列表 | **1–3 项且每项 ≤50 字符、无逻辑换行时行内，否则展开列表逐项书写** |
