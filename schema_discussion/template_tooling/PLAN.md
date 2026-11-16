# Template 工具更新计划

推荐顺序：**先固定字段、命名与文本规则 → 当前字段与共享语法 → `from` 与来源 → effective 静态关联 → 局部有限图；预算与排版在基础稳定后独立推进 → 统一 `check_yaml.py`、模板与说明 → 最后整理四个案例并验收。**

本轮已实施阶段 0–5，交付与验证见 [IMPLEMENTATION](IMPLEMENTATION.md)；阶段 6 的四案例整理单独延期，`schema_template/examples/` 保持未改。实施目标依据当前 [DOC0](../../schema_docs/0.common.md) 至 [DOC6](../../schema_docs/6.realization.md)，并纳入用户在 [CHANGELOG 末尾](../../schema_docs/CHANGELOG.md#字符限制预设prompt的草稿初版)给出的大小写、引号与句尾、字符数和行数要求，以及已确定的必要引号、简单 entry 单行与 `range/default` 简短表达式规则。新增要求优先；实施时同步受影响的正式规范，不能因旧 DOC 不同而搁置。

[最新文本补充表](TEXT_RULES_PROPOSAL.md)已获批准，继续在阶段 4B/5 内落实名称、数值形式、引用和紧凑调度布局，随后由新的 xhigh 独立审阅并修正。`300/2` 保持；此前的 `text_limit_demo` 是上轮满额快照，本次不修改或重新填满，也不用它证明新规则已验收。
字段拼写与小写属性边界已全部确定，见[字段决定](auto_normalization/FIELD_DECISIONS.md)。Storage 的具名 entry 采用两行布局，完整任务规范化和只读预览使用同一候选；其他记录与正文预算保持各自规则。
直接实现当前格式，删除不再需要的规则；不做旧格式兼容、迁移器、旧版检测、新框架，也不为旧测试维持已经取消的限制。细节与源码依据分别见[规则报告](01_MAX_RULES.md)和[流水线报告](02_MAX_PIPELINE.md)，本计划确定实施取舍。

## 实施阶段与最小验收

相对工作量只表示范围，不承诺时长。四案例更新前用少量当前规范的合成任务验证，不以旧案例或旧图片作为正确性标准。

| 阶段 | 工作与依赖 | 最小验收 | 工作量 |
| --- | --- | --- | --- |
| 0. 固定目标 | 按 DOC0–6 与新增要求建立逐路径字段/属性/具名条目分类、规范拼写、预算与布局表；同步正式 DOC 的结构、字段路径、命名/空值规则及例子 | 大小写、双向引号检查、全部预算与初始单行路径均有明确落点，按本计划直接实施 | 小到中 |
| 1. raw 与语法 | 按规范拼写更新结构与命名规则、共享 pipe/event 解析；在 `yaml_source` 补格式所需源节点信息和针对性的缺引号诊断；同批替换最小 fixture | 最小当前任务通过结构检查；字段与属性的大小写、step/role 大写、自取名称小写、ledger 可空及关键语法边界正确；缺引号造成的语法/类型错误可定位；精确数值和 alias 读取不变 | 大 |
| 2. 整区引用 | 依赖阶段 1 的 raw 包络，重写三区合并的 `from` DAG、effective 合成、来源定位与 standalone 上下文 | 三个不同来源、多跳、跨区成环、使用节点绑定、来源变动与副本隔离均正确；root 本地与 10/11 份边界有效 | 中到大 |
| 3. 静态关联 | 依赖阶段 1–2，更新 Contract、符号、step/launch/handoff、事件签名、成员与存储关系 | 当前合法开放表达不因不能求值被拒绝；提交归属、启动前提、默认/局部交接和确定的容量冲突有正反例 | 大 |
| 4A. 有限图 | 依赖阶段 3 的 effective 内容及共享解析，重写控制展开、模式推进、显式等待；更新图对象和图例 | 兄弟循环保序、真假条件、空工作、四种时点、缺失前驱/移位边界、成员选择、ordered 跨工作均正确；同名不生数据边 | 大 |
| 4B. 预算与格式 | 依赖阶段 1–2，可与 3/4A 独立推进；实现下文预算、双向引号规则、源样式 warning 和初始路径布局表 | 字符/逻辑行上限含 `range/default` 的 150/1 全部生效；必要引号保留、可安全去引号、简单记录单行与结构展开正确；仅允许已核准的普通说明末句号差异，其余类型和值逐节点不变，from/alias/merge 保留，格式幂等；不截断超限正文 | 中到大 |
| 5. 入口与交接 | 整合 3/4A/4B；补轻量 dispatch 检查，统一 CLI；更新两份 README、starter 及 Extract 接口说明 | 一次调用完成所选检查、排版和绘图；只读无写入；退出分类、来源诊断和 PNG 发布正确；starter 仍是待填写骨架 | 中 |
| 6. 四案例收尾 | 工具关键验收通过后，先审定完整设计及目录记录，再编辑 YAML/from，最后更新 PNG 与索引 | 每任务含 root 至多 10 份且各有保留理由；dispatch 与实际集合一致；保留必要事实/缺口；通过新入口及代表性有限样例 | 大 |

数字、计数口径、大小写及最新补充均已确定。两位实施代理先检查各自模块，整合后运行一轮完整合成套件；没有新变化或失败，不重复扩大测试范围。阶段 6 仍单独延期。

## 模块取舍

保留 **raw 读取 → 结构检查 → effective 合成 → 关联检查** 分层。库 API 只读，字段与关联规则不读写文件、不打印、不决定进程退出；CLI 负责请求与副作用编排，复用一次实际来源快照。可确定的本地仓库定位核对由显式文件层入口执行。

| 模块 | 决定 |
| --- | --- |
| `check_yaml.py`、`tools/schema.py` | 保留唯一作者入口和只读检查 API；抽出可复用已读文档的内部校验路径，避免格式、检查、图各读一版内容 |
| `design_resolution.py` | 保留 `ResolvedTask`、`RegionOrigin`、raw/effective 与来源 API，重写引用算法；删除 parent 树、直接父限制及旧字段分支 |
| `yaml_source.py`、`diagnostics.py`、`errors.py` | 保留权威标量解析、真实位置、错误分类和容量保护；补源节点/anchor 信息及必要的预算、格式诊断 |
| `pipeline_syntax.py` | 原地重写为动作、`for/if`、显式事件和 waiting 的轻量共享解析；结构、关联与图使用同一结果 |
| `rules/structure.py`、`declarations.py`、`schedule.py`、`pipeline.py` | 按当前字段和关系修改；不另建 schema 模型或注册体系 |
| `rules/values.py`、`symbols.py`、`hardware.py`、`common.py` | 保留有用规则和开放表达边界，更新路径、枚举与条件；删除旧字段检查 |
| `pipeline/graph.py`、`actions.py`、`model.py` | 保留模块职责，重写实例化与建边；当前对象统一为 step/event，删除旧 phase/default-data 等语义及别名类型 |
| `pipeline/expressions.py`、`windows.py`、`coordinates.py` | 保留有界精确计算、声明窗口和坐标恢复；适配新控制作用域，不扩成通用求解器或运行模拟器 |
| `pipeline/render.py`、`output.py` | 保留成员折叠保护、环展示、PNG 所有权、来源核对与原子写入；不为减少文件硬合并 |
| `pipeline_graph.py`、`check_policy.py` | 删除独立图 CLI、转发壳和未启用的目录硬件策略；有用的 `configuration/analyze` 移入现有 `pipeline` 包入口 |
| 新增模块 | 仅 `text_policy.py`（路径预算表、文本与源样式诊断）、`yaml_format.py`（纯格式候选）、`rules/dispatch.py`（固定目录结构/链接的小检查）；命名规则留在现有结构/声明检查中，不新增状态库、备份体系或第二套 YAML 依赖 |

## 命名与规范同步

按完整路径区分结构字段、条目属性、语法项、具名标识和普通填写值，唯一拼写见 [FIELD_MAP](FIELD_MAP.md)；同名键不能一律转换。输入大小写通过现行字段的 casefold 匹配规范化，保留单词和下划线；例如 `meta_data` 对应 `META_DATA`，不改成 `METADATA`。不建立旧词别名或版本分支。

| 类型 | 落实规则 |
| --- | --- |
| 四个顶层字段 | `META_DATA / CONTRACT / DESIGN / REALIZATION` |
| Schema 结构字段 | 如 `Algorithm_requirement / Math_reshape / Step / Role / Pipe / Using_verb`；三处 `Thesis`、`Arithmetic.Approx / Accumulate`、`Ledger.Baseline / Gain / Cost`；缩写按 `Virtual_ISA / Machine_ISA` |
| 条目属性 | 保持各路径约定的小写，如 `meaning / value / dtype / shape / execution_mode`，以及 Tile_reduction、Preassigned、Handoff、Index、noun 的属性；Launch 固定使用 `step / order / start_after` |
| step/role 标识 | 全大写；单个时仍为 `MAIN`，声明与引用一致 |
| 自取具名标识 | 常量、输入输出、symbol、轴、noun、index、storage、event 等自取名称全小写；关联公式、动作及引用同步使用该名称 |
| 语法与填写值 | `compute / for / if` 等动作和控制记法沿用既定语法；普通字符串、数值、真实路径与外部标识不作通用大小写转换；硬件登记名小写，设备型号与源码符号保留实际拼写 |

固定字段的拼写由规则直接检查；自取标识的合法性由对应声明检查。普通 `Name` 文本不因字段名而整段转小写，外部标识也不能因出现在键位置就擅自改名。正式 DOC 的结构轮廓、字段路径、引用示例和 starter 使用同一拼写；数字预算集中在工具配置，不在 DOC 中重复铺设长度条款。
`normalize_documents` 从结构元数据与全任务绑定生成同步编辑，`format_document` 只应用可证的编辑并保真重排；声明、公式和引用一起规范化，重名与歧义明确拒绝。四案例的内容整理仍单独进行。

## 规则、引用与图的落实范围

字段改动按[规则报告第 2 节](01_MAX_RULES.md)逐项落地：dispatch 去 parent；`rewrite` 去 step、ledger 下沉至条目；两区及 launch 改 step、预分配使用 stride/count；补 meaning；删 launch.protocol；index 用 meaning/definition；storage/view 用当前字段；noun 声明 `execution_mode/describe`；pipe 改有序列表；waiting 改三类字符串列表；feedback 与 require/prefer/forbid 平级。
DOC1/2 中 nullable、合法 null、`python_none`、具名 shape、精确数值、开放 dtype/硬件名称及合法 wiki 占位继续有效。Contract 输入的 `range/default` 只在传达有效域或默认值信息时填写，保留现有合法类型及简短、已定义的表达式；字符串限 150 字符/1 逻辑行，具体计数见[工具说明](../../schema_template/tools/README.md#文字与形式)。`gap` 仍是单字符串，直接采用 100 字符/1 行，不增加每输出 gap 字段。
`Ledger.Baseline / Gain / Cost` 允许 YAML null，非空文本各限 150 字符/1 行；`Ledger` 可省略，出现时保留 `Baseline` 键，`Gain / Cost` 无内容可省略。空值与省略均不表示零变化。

`from` 必须一次做对：

- 三个区域各选同任务同名整区，合并全部引用边检查节点 DAG；跨区互引也成环。root 三区本地，引用映射恰为 `{from: id}`。
- 每区保存完整 consumer → 中间转发者 → provider 链。Contract 来自 root；metadata、when、version、realization 只取使用节点。
- 引入定义在使用节点的完整上下文绑定；例如引用区中的 `bind:design.work_schedule::Bk` 取当前节点 Bk，不带入 provider 的调度环境。
- 来源集合是 root、自身和三条实际内容链的并集，不扩大为所有可达节点；实际来源变动影响身份，无关兄弟文件变动不影响它。
- raw/effective 及不同 effective 节点之间不共享可变对象。诊断同时保留使用上下文、真实定义位置和 `.from` 使用位置，不使用展开后的虚拟行号。

静态关联必须覆盖提交、事件和所属关系，细节见[流水线报告第 3.4 节](02_MAX_PIPELINE.md)：

- 两区 step 一一对应，每个 step 恰好提交一次，多 step 的 order、共享 worker 和 per_worktile 配对按 [DOC4](../../schema_docs/4.design.work_schedule.md)检查；不从映射排列推执行顺序。
- `start_after` 核对事件基名、形参数量、来源提交与参与范围；内建事件无索引，不可被本地 event 重名。检查本次启动等待自身及可确定无初始事件的启动环，不能只认 `kernel_done`。
- `ready` 属于 producer，`wait_at` 属于点名 consumer；别名展开保留动作、成员与 issue/done。省略 ready 使用默认就绪，未被 wait_at 点名的消费者仍在默认入口等待；等待动作可提前 issue。
- 片上交接按 storage 所属 worker、副本及实际存续核对；拒绝跨 kernel 沿用局部资源身份和可确定的冲突。正式文档没有给出的“一切 concurrent 片上交接非法”不从旧代码继承，也不因此新增跨 worker 通信语言。
- storage/view 的成员、槽、轴及可确定数值边界要检查；total_size 只在所需量与 dtype 字节数已知时核算。未知表达式不判错，不以同名同 shape 证明数学值、版本、区域覆盖或共享资源。
- Contract 属性引用、tolerance 覆盖和可确定的符号作用域继续检查；Realization 核对字段结构、引用记法与 evidence 形式。公式含义、动作保证、引用目标的语义对应、证据真实性与正文定义的复杂绑定仍由审阅核实。

pipe 和有限图按 [DOC5](../../schema_docs/5.design.tile_pipeline.md)重写，关键见两报告的 pipe/图章节：

- 八个基础 verb、noun 级三模式；普通与复合 matmul 分别核对固定签名。空循环/假分支仍做静态声明检查；索引只在自身词法域绑定，内层不重绑活跃索引，flatten 恢复坐标不能覆盖冲突值。
- 按列表依次展开 for/if；兄弟循环不交错，两个 if 不隐含 else；空工作保留循环外动作。动态 worktile 的 count/last 限制不能被有限样本绕过。
- 图只加动作 issue→done、pipe 发起/serial 推进、async_ordered 累积完成和显式 waiting；删除同名 noun 数据边。ordered 跨 noun/verb/循环/worktile 接续，serial 不排空前面的异步动作。
- waiting 支持 issue←issue、issue←done、done←issue、done←done，所有分类合并检查环。省略成员始终指当前 worker 成员；不存在即报错，只有显式编号或 `[*]` 才跨成员，warp 数不当实例数。
- 按目标发生域实例化等待：目标不发生则无该等待；目标存在而窗口内前驱缺失必须报错；仅变量移位越过声明窗口的项可跳过。固定引用错误不能被两端事件求交或实际 min/max 隐藏。
- 普通检查不执行开放数学表达式。有限图只用显式样本与受限精确整数/布尔求值；缺值、未知条件或无法恢复坐标返回未完成，不猜取值、不产出假成功的部分图。
- 本轮证明范围是所选样本、所列 step、代表 worker 的局部事件关系；flow 做上述静态核对。图不证明整个算子的活性、跨 step 区域覆盖、存储安全或性能，不建设复杂跨 step 执行模拟器。

## 一个日常入口

当前唯一作者入口采用以下行为：

- `python3 schema_template/check_yaml.py FILE`：检查同目录全部当前 YAML、dispatch 和文本预算，并整理所选 FILE 的格式；不因 from 自动改写 provider。
- `--all`：检查范围不变，把排版及请求的绘图对象扩展为全部当前 raw 节点。
- `--read-only`：做同样检查并报告格式差异，不写 YAML/PNG；格式差异标为“需要排版”，与 schema 错误区分。
- `--graph png --set NAME=INT --worktiles N`：同一次调用完成排版、最终检查、有限展开和 PNG；`--graph check` 只增加有限展开检查，不需 Graphviz。需要全程只读时合用 `--read-only --graph check`；`--read-only --graph png` 参数冲突直接报错。
- 未请求 graph 时不要求样例参数或 Graphviz。保留 worktiles 默认 1 并在输出标明代表性样本；显式取值及实际展开范围随图记录，不从旧 PNG、算子名或 dispatch 猜配置。
- 保留 `--skip-hardware`：显式跳过整个硬件检查视图、对应预算及依赖硬件的关系；原数据保留，其余成员、动作、索引和引用检查照常进行。
- 保留 `--show-effective` 的只读草稿预览，明确 `validation: not_run`；保留 `--standalone` 的只读单文件校验，不读取兄弟文件或 dispatch。这两个专用请求不执行自动排版/绘图；正式视图导出仍先完整检查创作任务，再验证独立视图。

处理次序：参数/目标预检 → raw 读取与结构核对 → 选中文件的格式候选及保真核对 → 安全写回并重读 → 最终 raw 结构/预算 → from/effective 关联与目录检查 → 请求的有限图及发布。解析或结构失败先报错且不写；预算或关联失败可以伴随已完成的格式清理，最终仍失败，图须等前置检查通过。
退出码沿用 0/1/2：0 为请求全部完成且通过；1 为结构/关系/预算、只读格式差异或有限样本反例；2 为参数、环境、容量、写入或展开未完成。混合结果逐项报告，存在未完成时总码为 2；输出明确哪些检查已做、格式是否变化、图的实际范围。
PNG 先检查目标归属，新图完成且来源复核通过才原子替换；失败保留旧图并说明未更新，普通检查和 `--all` 不顺带清扫图片。完整样本发现环可输出标红诊断图并返回 1；来源漂移或展开不完整不发布新图。

## 字符预算、文风与排版

[最新文本补充表](TEXT_RULES_PROPOSAL.md)已经批准实施，覆盖常见的名称、数值表达式、身份、内部引用与自由摘要入口。既定的 `300/2` 保留；原有类型与字段不变。当前完整预算和计数粒度集中列在[工具说明](../../schema_template/tools/README.md#文字与形式)，实现由 `text_policy.py` 的路径预算与既有字段规则共用计数函数，不在 DOC0–6 复制整张表。

每个可解析的自取单名按名称预算检查；纯公式中未能机器识别的变量、参数与函数遵循同一命名约定。普通 dtype、Storage.location、硬件 Execution/Feature 登记键按字段身份预算择一，不叠加单名限额。数值文本拒绝空白、全标点及全运算符串，保留 Unicode 数学、原数值语境中的无穷、值语境中的空集及原切片/事件记法。整数仍须有限。

retention 只放方式、invocation 或括号实例及既有迭代范围。内部引用优先匹配已有完整名称或路径，未结构化目标使用既有标识、索引或限定引用；身份只放名称，合法原名空格保留。仓库定位符每条一个目标，使用真实仓库相对路径，保留内部合法空格、括号，拒绝空路径和首尾空白；超限或确定形式错误报 error，真实名称和路径依据原材料处理，不由格式器截断或改名。

字符按解码后的 Unicode 码点计数，含前缀、空格与语义换行。逻辑行保留内部空行，末尾换行不额外生成一行；允许的 null 仍按空值规则处理。YAML 物理折行与逻辑行分开，不 trim 内容。列表按规则逐元素或逐字符串叶计，不累计虚构的列表/全文预算。raw 定义只计一次，from 展开不重复计数，alias 的各使用路径按各自语境检查。

文件 API 与 CLI 在已有本地上下文能准确识别 repo/ref 时核对对应提交中的路径；未知仓库或修订不推断，入口语义由原件审阅。`validate_documents` 继续只校验给定最终快照，本地资料核对通过显式文件层 hook 完成，不增加 repo 配置、目录搜索或联网步骤。

布局按下表实施。行内记录是物理排版，必须保留字段、类型、列表顺序及字符串中的实际换行。

| 路径或形状 | 排版规则 |
| --- | --- |
| Contract 的 Constant/Input/Output、Compute.Symbol 和 Math_reshape.Symbol 条目 | 固定属性记录行内；shape、range、既有数值数组等属性保持其类型 |
| Work_schedule 的 Partition、Invariant/Iteration、Single_tile、Tile_reduction、Launch 和 Handoff 记录 | 简单记录行内；有合法多行正文或其他结构子项时保持必要层级 |
| Mapping.Worker、Mapping.Placement | 各记录一行；Mapping 展开，strategy 独立一行，worker_count/member 原行内行为保留 |
| Schedule.Worktile_order | 每个 entry 一个行内记录；单 entry 与父字段合行，多 entry 每项一行；by 列表与顺序保留 |
| Worker_assignment.Preassigned | start/stride 或 start/count 的整个记录行内，两种形式仍互斥 |
| Worker_assignment.Refill | none 照常；一机制时整个映射行内，至少两机制时父映射展开、每个机制规则独立一行；保留必要的 flow 引号 |
| Worker_assignment.exit 列表 | 1–3 项且每项不超过 50 字符、无逻辑换行时行内；否则展开逐项，标量保留自身逻辑行 |
| Tile_pipeline 的 Index、Using_verb noun 与 view 叶子记录 | 简单记录行内；已有属性结构保持类型与值 |
| Storage 具名 entry | 固定两行：名字及 meaning/depth/location/shape 为第一行，dtype/total_size/member/view 为第二行；可选字段缺失时略去，不改变分组；真实块文本及共享节点保留必要合法表示 |
| 区域、完整 step/role、Info/Pipe、Storage 登记表、控制列表、Compute/Rewrite | 结构展开；数学等多行正文保持必要层次 |

单条简单具名记录表或单项字符串列表可与父字段合行；上述结构容器与 exit 专门规则按各自路径处理。排版不设置全局行宽，不把其他层级递归压成单行。

必要引号由同一标量 resolver 及实际 block/flow、key/value 语境确定。缺引号导致的解析或类型错误报 error，能安全裸写的多余引号报 warning；改变为 flow 后重新保留必要引号。合法块标量和文本内部引号保留，不自动修复无法确定原意的坏 YAML。

除显式授权的大小写与单一身份 `MAIN` 同步外，只删除普通说明的确定终末一个 `.` 或 `。`，实际写回后给修正 warning；其余字符、空格与换行精确保留。缩写、引文、数学、路径和引用等不确定文本不改，alias/merge 的全部受影响用途都安全才修改；非说明或被跳过硬件的用途会阻止共享值修改。只读入口不改原文，其尚未应用的候选差异仍可检测。

格式候选由 PyYAML 源节点/事件与同一 resolver 生成，逐节点回读核对精确类型、值与 alias/merge 图，并验证幂等；Decimal 不经 float。原 Document 不变，真正的 YAML 注释报错，不能静默删除。写回前复核来源，原子替换并保留权限；未变化不写、不改 mtime。写后使用实际位置和 hash 进行预算、关联与图检查，版本不因格式清理递增。

## 四案例最后处理

现有[案例索引](../../schema_template/examples/README.md)为 FA4 prefill 11、decode 7、QSA 1、GDN 2，共 21 个 YAML；prefill 已超上限。**本计划不预定最终节点数，也不先把 21 份全面改名或逐份迁完。**

1. 先人工核对来源和完整端到端方案，形成少量完整设计的保留理由；把同规则参数、可闭合的局部变化、已知组合及未选中整体候选按 common 分清，再决定具体 root、when 与当前参数。
2. prefill 重点看持久化、split-KV 等整体组织与 joint/dual load、LPT、alpha 共享的边界；decode 看输出归并和 q1/q4 的实际分工，不能机械保留参数全排列。
3. QSA 保留 ticket、延迟发布、空 worker drain 和已知缺口；GDN 只依据本批 L20D 设计，保留描述符问题及真实执行但未被最终主 kernel 消费的物化/转换成本，不混入另一批 B200 方案。
4. 审定后编辑当前 YAML：落实规范字段拼写、自取名称与 step/role 大小写，并同步公式、动作、引用和 dispatch 定位；精简超限文字、句尾及 realization 条目。各份方案闭合，同一完整区才用 from，变更区完整本地定义；保留尾块、空工作、队列/终止及数值分支，不为图通过编造等待、隔离或保证。数据改动递增本文件版本。
5. 同步中文 dispatch 的“范围”“完整设计”和有内容才有的“变体与组合”；每条记录直接定位当前完整基线及既有对象，不形成 patch 链或新增全排列节点。目录不能补齐 YAML 缺失定义。
6. 最后更新保留集合的 PNG、复现命令、README 数量和链接；删除节点时同步处理其 from 引用与已确认生成图，不把额外完整设计藏到历史/子目录绕过上限。

工具只核对 dispatch 的固定最小结构、当前 YAML 链接覆盖/存在、基线链接和数量；独特设计价值、对象的语义定位、组合相容性及范围仍由审阅判断，不新增 Markdown 框架。starter 使用当前字段和 MAIN，保持 version: 1 与未填写骨架性质，不编造来源或参数使它通过。
测试只保留有意义的边界：精确读取与格式保真、路径分类与命名、Unicode/逻辑行上限、`range/default` 字符串及端点预算、block/flow 与 key/value 语境的双向引号检查、简单 entry 单行与多行 math、ledger 空值、from 来源/绑定、关键结构与 flow 关系、条件/窗口/模式事件反例、CLI 只读及 PNG 归属/来源漂移。缺引号分别覆盖解析失败与解析成功但类型错误；预算用代表性边界覆盖，不逐条镜像配置表。旧字段、旧默认边、旧 CLI、千级父链及固定案例数量的测试删除或重写；不在测试中重画正式案例。

## Extract 交接与必要输入

本轮只稳定 Template 侧接口：`yaml_source.load/loads` 与 `Document` 的值/位置/hash；`rules.validate_algorithm` 的 Contract-only 检查；`schema.resolve_directory`、`validate_file/directory/resolved_file/standalone_file`；`ResolvedTask` 的 raw/effective、origins、source_manifest、effective_identity 与 effective 导出原语。现有 Python 调用保持只读，新增预算诊断不引入文件写入。
默认 CLI 已自动排版；下一阶段 Extract 的外部检查调用使用 `--read-only --skip-hardware`，或直接使用只读 API。正式阅读视图必须由同一已验证来源合成，带完整 provenance，再独立验证。
本轮不改 Runner、阶段权限、审批/回执、冻结 run、恢复/发布或 schema wiki。若维护[执行器对齐说明](../../schema_extract/EXECUTOR_ALIGNMENT.md)，只事实性更新“Template 已完成哪些项、Extract 仍未完成”，不能据此改变运行状态或声称端到端可用。

## 已确定的三项边界

| 项目 | 实施规则 |
| --- | --- |
| 必要引号 | 语法/类型必需的引号保留并检查缺失；可安全裸写的普通文本去包裹引号。缺失引号导致的语法/类型错误报 error，多余引号报 warning；合法块标量与文字内部引号保留 |
| 简单 entry 单行 | 按上面的初始路径与形状表执行，input/output/constant/symbol 等记录单行；结构容器、控制列表和合法多行正文保留展开，不以任意宽度或递归压扁决定 |
| `range/default` 字符串 | 保留简短、已定义的表达式，字符串各限 150 字符/1 逻辑行；range 列表的合法字符串端点逐值检查，不改变其他类型与空值规则 |

以上均已定案。案例保留集合与具体内容仍由最后阶段研究和编辑，不把常规路径分类、模块细节或旧 DOC 差异转成新的用户问题。
