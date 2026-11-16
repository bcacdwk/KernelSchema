# Template 更新独立规划：规则、解析与数据流

本报告只规划，不实施。依据为当前 `schema_docs/0.common.md` 至第 6 篇；`CHANGELOG.md` 用于核对变化，末尾新出现的“字符限制预设（prompt 的草稿初版）”单独视作草稿输入。已读 `AGENTS.md`、本目录 `BRIEF.md`、正式 DOC0–6、Template 两份 README、下列源码与关键测试；未读取另一份独立报告，未运行提取、测试或绘图，未修改正式代码、文档和 YAML。

建议保留现有“读取 raw → 字段检查 → 展开 effective → 关联检查”的层次，以现有小模块直接承载新规范。主要工作集中在 `from` 引用图、pipe/事件解析和有限展开；新增说明字段本身不需要新数据模型。文本预算与格式整理各增加一个小模块即可，不引入 schema 编译器、通用插件体系、版本识别或兼容解析。

## 1. 推荐实施顺序

相对工作量：S 为局部调整，M 为单个职责的系统修改，L 为跨模块联动，XL 为本轮主要复杂工作。这里只比较规模，不承诺时长。

| 阶段 | 实施内容与依赖 | 阶段验收 | 规模 |
| --- | --- | --- | --- |
| 0 | 固定 DOC0–6 的目标字段及引用语义；整理本报告第 7 节的草稿冲突。预算数值和排版表待定不阻塞阶段 1–4 | 有一份按正式字段路径映射的变更清单；没有将草稿缩写、旧数值、大小写建议作为语言规则 | S |
| 1 | 先改 `pipeline_syntax.py` 的动作、事件、`for/if` 与词法作用域表示，再改 `rules/structure.py`、共用枚举与值检查；同批替换最小 fixture 和测试构造器 | 一个当前格式的最小完整任务能通过字段检查；各新字段、条件必填、单 step/role 的 `MAIN` 有关键正反例；没有旧字段分派逻辑 | L |
| 2 | 以阶段 1 的 raw 包络改 `design_resolution.py`，同步 `diagnostics.py`、`schema.py` 的完整检查与 standalone 接口 | 三个区域分别引用、跨区引用成环、实际 provider/chain、使用节点错误定位、完整来源摘要均正确；raw/effective 不互相污染 | L |
| 3 | 接通 `declarations / symbols / schedule / pipeline` 的当前关联规则。管线关联复用阶段 1 的解析结果，不另写一套字符串解析 | 在 effective 使用上下文中检查 Contract、step/launch/handoff、动作声明与签名、成员、索引和存储；当前合法开放表达不会因不能求值而被拒绝 | L |
| 4 | 改有限图：有序列表与条件展开、目标发生域、显式事件时点、当前成员规则、noun 模式；删除默认同名数据边。调整图报告与渲染标注 | 小型见证覆盖条件真假、空工作、移位边界、缺失前驱、成员选择及累积完成；缺值/工具不支持与 schema 错误分开；普通检查不需要 Graphviz | XL |
| 5 | 增加集中预算表及无损格式器；先做小规模保真原型，再填最终规则表。布局只使用已确认偏好；预算数值待用户最终确认 | 字符/逻辑行边界检查准确；格式前后值和类型逐节点一致，alias/from 不变，重复格式化无 diff；长文本只报错 | L |
| 6 | 将格式、检查、有效视图和图功能接入唯一作者入口 `check_yaml.py`；加任务目录的少量 dispatch 结构检查；同步两份 README、starter 与关键 CLI 测试 | 从任一当前节点完成同目录检查；显式只读模式不写文件；功能失败状态清楚；输出追溯同一次实际来源；starter 明确只是待填写骨架 | M |
| 7 | 最后处理四个案例：先审定具体完整设计集合及目录记录，再逐个修订当前 YAML、共享关系和引用，最后整理图片与索引 | 每个任务含 root 至多 10 个完整节点、每份有保留理由；dispatch 完整；关键设计事实和原有缺口不被删改；通过当前入口与必要有限样本检查 | XL |

各阶段只保留相应的关键检查；最终再跑一次维护检查。阶段 1 会使旧 fixture、旧字段断言和旧示例暂不适配，应在同一工作分支按阶段更新，不为使旧套件暂时全绿保留双轨规则。阶段 7 前不把四案例作为新 checker 的唯一正确性基准。

Extract 仅在阶段 6 记录稳定的 Template 接口边界，见第 9 节；不与本轮联动修改 Runner、审批状态或冻结运行。

## 2. 由新规范要求的修改

### 2.1 字段与数据模型

`rules/structure.py` 目前已经用 `record / named / sequence / predicate` 区分固定结构、具名映射与开放值（`values.py:28`、`:48`、`:61`）。这套小组合足够承载下表，不需要再引入完整 JSON Schema、Pydantic 模型或字段注册框架。

| 正式目标与依据 | 现有代码落点 | 推荐修改 |
| --- | --- | --- |
| `meta_data.dispatch` 只有 `id? / when`；root 独占 Contract；当前完整集合含 root 最多 10 份（DOC0:47、:136、:140） | `structure.py:213`；`declarations.py:31`；`rules/__init__.py:27`；`design_resolution.py:159` | 删除 `parent` 的字段、缺省、存在性和父树检查；保留任务名称/描述一致及文件名身份检查；在完整任务校验加 10 份硬上限 |
| 三个 Design 区本地完整定义或恰为 `{from: id}`；root 只能本地（DOC0:192） | `structure.py:202`；`schema.py:27` | 将分支改成当前 `from`，混合本地字段直接报普通结构错误；不识别或转换旧格式 |
| `rewrite[]` 为 `name / math / symbol? / arithmetic? / ledger?`；ledger 为 `baseline / gain? / cost?`（DOC3:35、:66） | `structure.py:54`；`declarations.py:44` | 删 `rewrite.step`、区域级资源账本及 `by` 关联；ledger 下沉至条目，baseline 必填且各实填值非空；不另造数学步骤 ID |
| work/tile 两区均为 `step`；launch 为 `step[]`；分配为 `start + stride` 或 `start + count`（DOC4:56、:211、:245） | `structure.py:79`、`:95`、`:105`、`:119`；全部关联模块 | 同步字段及定位路径；保留互斥、一次提交、worker 接续与工作排列检查 |
| partition、invariant、iteration、single_tile 新增 `meaning`（DOC4:99、:366） | `structure.py:63`–`:74`；`symbols.py:31` | 在现有条目规则加必填文本；不把 meaning 纳入工作坐标的数值等价判断 |
| launch 不再有 `protocol`；启动机制在动作契约与 Realization 绑定（DOC4:270、:398） | `structure.py:107`；`schedule.py:39` | 删除字段及“使用 launch_ready 必须带 protocol”的旧检查；不替换成新字段或固定句式 |
| pipeline 公共字段为 `index / event / storage`；index 条目含 `meaning / definition`（DOC5:40、:350） | `structure.py:122`、`:189`；`rules/pipeline.py:20`；图构造入口 | 沿用现有三种域的值检查，显式取 `.definition` 后进入域算法；说明文本不参与域求值 |
| storage 必填 `meaning / depth / location / shape / dtype / total_size`；view 仅有 `slot_index / region`（DOC5:377、:402） | `structure.py:172`；`declarations.py:115`；`pipeline.py:107`；`symbols.py:38` | 删除旧字段及 view.dtype；保留空 view 条目合法的特例；补字节占用和成员关系的可判定检查 |
| `info.describe / member / using_verb` 必填；noun 级 `execution_mode / describe`；八个封闭动词与三模式（DOC5:80、:174、:228） | `structure.py:178`；`common.py:9`；`pipeline.py:13` | 声明为 `using_verb.<verb>.<noun>`；不保留 role/verb 模式继承，也不容许自定义基础动词 |
| pipe 是非空有序列表，只有 `for / if`；waiting 三类各为非空字符串列表，时点显式（DOC5:262、:454、:474） | `structure.py:131`–`:150`、`:187`；`pipeline_syntax.py:75` | 重写 pipe/依赖解析模型；不把旧的段落映射包成新语法的内部默认形态 |
| 顶层 `realization.feedback[]` 为 `item / choice / evidence`（DOC6:23、:54） | `structure.py:195` | 删除 prefer 内嵌 feedback；新 feedback 与 require/prefer/forbid 平级，保持证据引用约束 |

DOC1、DOC2 多数结构可直接保留。特别保留 `nullable`、`python_none`、具名 shape、精确数值、默认值表达式、开放 dtype、硬件开放单元名称及 wiki 占位范围，不借“精简”新增封闭类型表。`algorithm_requirement.golden_reference.gap` 仍是一个可选字符串，不是按输出划分的映射。

`MAIN` 是单 step 与单 role 的条件规则（DOC4:358、DOC5:598），不是“所有 step/role 都必须大写”。其它数学值、维度、硬件标识保留现有名称语义。目录记录按唯一 rewrite 名称定位时，应检查该定位是否唯一，不能凭去掉旧 `rewrite.step` 就发明一套新 ID。

### 2.2 关联检查应做到的范围

建议继续由声明模块先保证后续遍历安全，再由所属模块检查关系。`rules/__init__.py:25` 的分派方式足够；删除其中重复的父树检查即可。

- **Contract 与数学接口**：保留输出 tolerance 的精确覆盖、Contract 对象唯一性、属性引用目标与无环、shape 轴名、nullable 明显结构冲突。`symbols.py:23` 有按 descriptor 位置遍历的正确边界；同步当前字段并去掉 view.dtype 后，不改成“搜索所有名叫 shape 的键”。结构化 `ref:contract.algorithm_requirement::X.dtype` 始终指 root。对于只在 math、strategy、动作契约中定义的 `bind:` 目标，不用扫描英文公式制造伪符号表；DOC3:145、:155 明确这部分仍需审阅。
- **工作提交**：保留 step 与两区一一对应、每个 step 恰好提交一次、多 step order、共享 worker、per_worktile 配对、preassigned 互斥等条件。比较的是坐标、排列和分配规则，不是新增的 meaning 文字或排版。
- **启动关系**：`schedule.py:37` 目前只有 kernel_done 引用进入 launch 环图。按 DOC4:390–:404，应统一解析已声明事件、它所属的提交及参数个数，检查本次提交等待自身和无初始事件的启动环；不能因为事件改叫 launch_ready 或本地 alias 就完全漏查。动作中的可见性、触发保证与实现机制仍由审阅核实，不靠 protocol 替代。
- **handoff**：保留 producer/consumer、存储路径、同 launch 同 worker 等可判定条件。event 别名解析要核对基名、形参数量、所属参与者及保留的 issue/done；wait_at 是各消费者自己的局部事件引用，不能当成一次统一替换所有消费者的开关。没有被 wait_at 点名的 consumer 仍采用默认入口等待。数学值/版本/区域覆盖、初始化必要性和跨 worker 进展保证不从相同 shape 推断。
- **storage**：depth > 1 时需要 view 和逐 view 的 slot_index；不推导槽映射。region 必须引用父 shape 的轴，检查字面量端点与容量界；整槽空 view 合法。total_size 仅在 depth、容量、成员数及 dtype 字节数均已知时做数值等式检查，按 worker.unit 成员计数，不按 warp。开放/打包 dtype 或符号表达式不能因为 checker 不知道 bytesize 就判非法，不建立庞大 dtype/硬件知识库，也不将多 CTA 总量直接比较单 SM 容量。
- **Realization**：feedback 结构和 evidence 检查是确定必改；对于可唯一解析的结构化 item，可检查它与被替代 prefer 指向同一对象。不能要求裸名与限定名逐字相同，也不能用本文本检查证明 evidence 真实或选择合法。通用 item 解析仍沿用上游已发布对象范围，不增加全局对象身份框架。

硬件普通检查仍默认开启。`--skip-hardware` 保留“跳过整个硬件检查视图及依赖硬件的关联检查，raw 与返回内容保留”的含义（DOC2:13；`schema.py:44`）；新增检查不得通过这个参数跳过非硬件的成员数、pipe、存储形状或引用规则。

## 3. `from`、有效视图与来源身份

### 3.1 必须替换的算法

`design_resolution.py:151`–`:221` 目前收集 `parents`，强制 `from_parent == direct parent`，再沿父树排序。这不是改两个字段名即可修复的地方。

推荐在现有函数内替换为以下小算法：

1. 从每个 raw 节点的三个区域收集 `(consumer, region) → target`，只接受本任务内实际文件 ID。检查 root 本地、目标存在、单字段 from、文件与 dispatch 身份；不创建隐式 parent。
2. 合并三区域的所有引用为节点有向图，对该图做无环检查，错误定位到具体 `.design.<region>.from`。例如 A 的 math 来自 B、B 的 schedule 来自 A，虽然单独各区域都能落地，仍必须拒绝（DOC0:201）。
3. 按图的依赖顺序，对每个 `(node, region)` 找到同名区的实际 provider，保留完整 chain。沿用 `RegionOrigin(provider, chain)` 和现有拷贝保护。
4. 合成当前节点的三个有效区域，保留本节点 metadata/when/version/realization。关联检查在此节点的 root Contract 与当前另外两区中执行，不沿 provider 的外部参数环境先绑定。

关键验收例：A 的数学区来自 B，其中有 `bind:design.work_schedule::Bk`；A、B 调度各选不同 Bk。不能要求两者 Bk 相同，也不能把 B 的调度带到 A；应在 A 的接口环境检查它是否成立（DOC0:205）。某个引用使 A 的 step 集合不匹配时，错误既指出 A 的组合上下文，也定位 B 的真实定义及 A 的 from 使用处。

### 3.2 应保留的结构

`ResolvedTask`（`:30`）、`RegionOrigin`（`:21`）、`effective_document`（`:53`）、`source_manifest`（`:40`）、`effective_identity`（`:47`）与 `diagnostic_source`（`:62`）已有合适职责。建议保留这些 API 的形状，修改其内容语义；这不是旧格式兼容。

source manifest 应是“root + 自身 + 三条实际区域链”的并集，不是节点 DAG 的全部可达节点。例如 A.math 来自 B，B.math 本地而 B.schedule 来自 C；若 A 没有使用 B.schedule 或 C 的区域，则 C 不属于 A 的内容来源。全任务的图仍要合法，但 A 的身份不应因未使用区域变化而无故扩大。现有 `:223`–`:259` 按 region chain 累计来源的思路值得保留。

使用节点 raw 未变而 provider 或中间转发者变动，manifest/effective identity 应变化；未引用的兄弟文件变动不应改变该节点身份。raw/effective、不同 effective 节点之间保持不共享可变对象。不要让格式器写回 `effective_task`，否则会把共享内容固化成副本。

### 3.3 草稿预览和正式阅读视图

当前 `check_yaml.py:33` 的 `--show-effective` 调用不做完整验证的 resolver，并输出 `validation: not_run`。当前正式规范要求阅读视图生成前核对完整有效 Design、另存并带全部内容来源（DOC0:213–:215）。

推荐保留两个用途的明确界线，不新增复杂模式：现有只读草稿预览可以保留，继续明确 not_run；用于导出/交付的阅读视图必须调用 `validate_resolved_file`，从同一 ResolvedTask 生成，并再过 `validate_standalone_file`。草稿解析成功不能成为正式视图的批准。若统一入口本轮不新增落盘导出，保留上述 Python 原语即可；Extract 的导出流程下一阶段接入。

`schema.py:114`–`:119` standalone 的 parent 分支应删除。standalone 自带 Contract 与本地三区，保留原节点身份；检查时用独立上下文，不把文件所在目录重新解释为创作任务，也不从相邻文件补缺。

`diagnostics.py:65` 应定位 `.from` 并保留 provider 与 consumer 两端。`effective_yaml` 当前将所有字符串加双引号以保护类型（`design_resolution.py:286`）；这比禁引号安全，但最终显示可复用第 6 节的标量策略，不能直接拿它给 raw 文件做格式化。

## 4. Pipe 与图：必须改的语义

`pipeline_syntax.py` 已被结构检查、关联检查与图共同使用，是正确的收口位置。推荐只增加表达当前语法所需的轻量节点（动作、for、if、显式事件、依赖规则）和原文位置；表达式正文仍是字符串，不把 math/strategy 编译成程序。

### 4.1 一个共享的语法解析层

- 替换 `ScopedPipe` 的 `repeat + regions` 模型（`:75`–`:117）为有序列表的控制树；记录局部域、绑定变量与条件位置。for 支持现有规范的域、切片、flatten 工作窗口、局部半开范围；if 只有当前判断与一个非空 body，没有 else 或隐式配对。
- `action_parts` 当前对旧 `mma` 强制两个操作数（`:166`）。当前 matmul 同时允许普通结果 noun 和 `A@B`；复合键只在 matmul 下成立。声明基名、各操作数实参数量与每次调用必须一致，包括空域/假分支中的静态调用。
- `parse_event` 当前识别 `Role/action@done` 并去掉后缀（`:193`）。改为完整的 `{issue|done Role[member]::verb:noun(...)}`，由同一解析器识别 waiting 的单目标、箭头与非空前驱列表，不能用简单按逗号切分破坏多参数和 flatten 坐标。
- 本地 waiting 可以省略当前 role/member；event 别名没有此上下文，必须显式 role/member/time。event 的形参属于别名自己的参数，不能捕获某个 role 的当前循环变量。
- 索引在循环 body 内绑定，兄弟循环不泄漏，仍活跃的外层索引不能被内层重新绑定。flatten 恢复出的外层坐标须一致；现有 `graph.py:70` 的 `dict(env, **item)` 会覆盖已有值，不能照搬。既定公共 worktile/flatten 的 count/last 限制继续保留。

### 4.2 默认边与成员选择

这两处直接改变正确性，不能作为可选重构：

1. 删除 `pipeline/graph.py:253` 的 `default_edges()` 及 `build():299` 调用。当前它从同名 load/publish 推数据边，DOC5:522 明确已取消。以后的局部图只建立动作 issue→done、pipe 发起/serial 推进、async_ordered 累积完成及显式 waiting；不能从 noun、storage 或 view 推数据边。
2. 替换 `graph.py:122` 的省略成员规则。当前不同 role 布局不同时可能自动选对方所有成员；DOC5:611 要求省略始终指当前 worker 成员，没有该实例就报错，跨成员需显式编号或 `[*]`。Role[*] 枚举 worker.unit 成员上的 role 实例，不枚举同 CTA 内的 warp。
3. 执行模式从 `using_verb.<verb>.<noun>` 取得。async_ordered 的 done 累积跨 noun/verb/循环/worktile；serial 不重置这个序列，也不排空先前 async_unordered。当前保存一个 `ordered` 指针的思路可用（`graph.py:95`–`:120`），但模式读取与枚举必须更新。

### 4.3 条件、边界与有限展开

当前 `graph.py:215` 的显式边在域组合上枚举两端；旧 pipe 没有 if，不能直接沿用来处理目标不存在。新流程须先解析两端的角色、动作族及签名，再依据左侧动作的发生域实例化：目标因 if/空域不存在时无等待；目标存在而窗口内前驱不存在时是错误；只有规范定义的变量移位域外项可以跳过。不能以两端实际事件集合求交来消除悬空前驱（DOC5:666）。

`windows.py`、`coordinates.py` 已包含从声明域判断窗口和精确恢复坐标的有用逻辑，建议在新控制上下文下保留可用部分。将条件发生域与声明索引窗口分开，避免把“假分支没有生成动作”解释成越界。不能为了缩短实现仅记录事件实际出现的 min/max 范围。

`expressions.py` 的 Fraction 整数求值、容量限制及显式 sample 输入可保留；为 if 增加必要的布尔值、比较及 and/or/not 求值，不使用 eval，不把其支持子集反向变成 schema 语言。动态工作领取或动作产出的条件并不是静态常量：只有足够的显式样本才能展开，不能从 describe 执行模拟、猜测真假，或把两个 if 自动当一对分支。

图仍可按 step 独立展示，并明确不证明跨 step 区域覆盖、同步原语保证、存储安全或性能。`pipeline_graph.analyze:40` 及 `render.py:65` 目前就是独立分区，不应因为改名就声称实现了完整跨 step 调度模拟。结构化 launch/event/handoff 关联在 checker 中接好即可，不在本轮扩建全局执行引擎。

若采用默认绘图，缺少动态条件/规模或不支持某表达式时，输出“schema/文本已检查；图未完成及缺少哪些样本”，请求的图阶段返回工具未完成状态。不要生成看似成功的部分图、默认为真、从旧 PNG 读参数或增加 symbolic/opaque/兼容旁路。建议保留已有显式 `--set`、`--worktiles` 一类工具参数；它们不是新增 YAML 字段。

## 5. 文本预算：集中而且只计正文

新增一个 `tools/text_policy.py` 即可：用正式字段路径及有限的 `* / []` 匹配描述文本类别、字符上限与逻辑行上限，再遍历 raw 的实际定义。不要在每个 `text()` 调用点硬编码数值，也不要仅凭最后一段键名就把同名自定义对象套入规则。

建议区分以下对象，不建立另一套数据模型：

| 类别 | 典型位置 | 处理 |
| --- | --- | --- |
| 描述正文 | meta 的 name/describe/when；compute/rewrite 的 name/math；meaning/thesis；各 strategy、refill 机制说明；hardware execution/feature；noun describe；choice/reason | 按明确路径预算 |
| 条件文本或说明 | tolerance 每输出字符串、单个 gap、arithmetic、ledger、unknown 理由 | 显式登记预算，保留各自字段形状；unknown 仅在规范允许处使用 |
| 语义字符串 | ref/bind/from、source/wiki/evidence、dtype、retention、成员路径、动作、waiting/event、整数表达式及条件 | 继续各自语法/引用规则；不套“所有字符串 150 字符”之类兜底限制，最终表若专门列项再加入 |
| 具名键 | 数学值名、轴名、节点 ID、noun 等 | 既有标识符或位置专属规则；不能把名叫 name 的自定义对象当成 `*.name` 字段 |

推荐按解析后的 Unicode 字符与逻辑换行计数，而非 UTF-8 字节、英文单词或 YAML 缩进行数；采用何种末尾换行口径需要最终表一次敲定。可采用 `len(value)` 和明确的逻辑行算法，说明 `|` 常见末尾换行是否计字符、是否产生额外空行；不偷偷 trim 所有空白以绕过预算。物理记录是否单行属于格式器，与多行字符串自己的逻辑行数分开，不能将字符串改成转义 `\n` 就宣称行预算已满足。

相同来源、相同规则的预算诊断去重，不因 from 展开或 alias 使用次数重复罚同一份文字；alias 被用于不同字段时仍适用各字段自己的上限，不能只检查 anchor 的原位置。错误定位真实 provider/anchor，并标出字段、实际字符/行数及上限。继承文本在多个使用上下文仍须检查关系，但不新增“展开后全任务字符总预算”。硬件被显式跳过时，需维持跳过该区检查的既有承诺，不让新预算暗中恢复硬件完整性门槛。

预算失败不自动删除文字、句号、标点、字段或条件，不把 strategy/math 移到目录来避限，不改变 from 选择。最终数值未确认前可先完成遍历与覆盖核对，不启用一套猜测值。必要验证只需表驱动的限值/越界、逻辑换行、路径例外、alias/from 来源定位；不为每个文案写测试。

## 6. 无损格式化与统一入口

### 6.1 格式器的最小实现路线

事实：`yaml_source.Document` 只留解析值、位置、值索引与 raw SHA256（`:26`）；`convert` 会展开 alias/merge（`:94`–`:134`），不保存 anchor 名、alias 使用点的完整语法结构或原排版。因此不能对 `Document.value` 做一次 dump 当作作者文件的格式器。`effective_yaml` 可以展开 alias 成阅读内容，raw 格式器却应保留作者的复用意图。

推荐新增 `tools/yaml_format.py`，优先用已有 PyYAML 的事件/节点与 emitter，在源表示上调整 flow/block 风格、缩进、必要引号和有限的字段顺序。保留 anchor/alias/merge 关系及标量原值；让 YAML emitter 处理合法引用与转义，不写一个完整 YAML 语法器。先以一小组难例验证这条路线；若事件适配的代码反而大于成熟 round-trip 库方案，可以选择一个私有后端，但不要因此替换权威语义 reader 或让两套 scalar 规则各自决定正确性。

格式策略是一张有限的路径表：短 descriptor/partition 条目、shape 与 from 映射可紧凑；math 和控制列表维持层次；info/using_verb、storage/view 的布局依最终确认表处理。不要按“所有实例键都压成一行”递归，因为 step/role 也是实例键，里面包含整段计算；也不要靠字符串长度猜哪个对象是 schema 结构。固定字段次序可参考正式轮廓；若移动内容会使 alias 先于 anchor，保留合法定义顺序，不能为外观破坏文档。

格式器必须满足以下不变量：

- 只处理 raw 源文本，不先展开 from；不新增或消除共享、不重命名对象、不中转成旧格式。
- 格式前后都用 `yaml_source.loads` 解析，并递归检查每个节点的类型与值相同，不能只用 Python 松散相等（例如 `1 == True`）。Decimal 不经过 float。表达式、依赖字符串、源引用、nullable/default/null 的含义及列表顺序逐项不变。
- 保留显式 anchor/alias/merge 的复用关系；scalar 相同不代表可自行创造 alias。多行标量的折叠/换行/尾换行不能被自动重排改值。
- 引号能安全省略才省略；`"true"`、数值外观字符串、以 `{` 开头的 waiting 字符串、含 `: ` 的文本等必须保持字符串类型和合法 YAML。不得全面报“去引号”warning。
- 文本超长仍报预算问题，不自动删句号、改术语或改大小写。DOC0:59 不允许实例 YAML 注释；如果发现真实注释，应报告并由作者处理，不让 formatter 静默删除文字，也不把字符串里的 `#` 误报成注释。
- 幂等；未变化不写文件；仅在显式格式操作中写入。写前复核实际 source hash、分文件临时写入再原子替换并保持权限；发生来源漂移不覆盖用户的新内容。修改后重新取得实际位置和 hash 供后续诊断、有效身份及绘图使用。纯排版不递增 version（DOC0:43）。

建议不把格式布局做成 schema 合法性的另一道门。普通 compact/block 表达仍按解析结构有效（DOC0:65）；格式检查若提供，应报告“需要整理”，不得混同字段非法。

### 6.2 统一入口如何编排

`check_yaml.py` 保留参数、结果输出与退出状态，`schema.py` 保留可复用的只读检查；格式与 PNG 的副作用由显式操作协调，不塞进 `rules` 或 `validate_algorithm`。内部复用一个 ResolvedTask，避免格式、引用视图与图分别读到不同版本。

推荐执行次序为：读取源 → 按请求产生格式候选并做无损回读 → 如需写入则提交并刷新 source snapshot → raw 结构及文本预算 → from/身份 → effective 关联 → 按请求做有限展开与输出。文本预算可与安全可遍历的 raw 结构诊断一起收集；图依赖完整语义检查通过。格式不能让预算错误消失，格式已执行也不能隐藏随后仍失败的检查。

**入口默认行为是整合选择，不是规范变更。** 我的推荐是默认只读检查，显式选格式化、有限图检查或 PNG；使用者仍只需记一个脚本。若主计划采用“一次调用默认格式并绘图”，也应通过同样的内部阶段完成，保留一个明确的只读选项，动态样本不足按第 4 节处理，不新增替代引擎。格式/图写入当前节点还是全部节点要一次说明；建议两者共用同一明确选择（当前文件，显式 all 扩展），而合法性检查继续覆盖同目录全部当前 YAML。

退出含义保留 `0 已请求操作全部完成`、`1 输入/约束错误`、`2 参数/依赖/容量或请求图无法完成`；输出分别报告实际完成的检查与缺失的图样本。不要把图未完成写成 YAML 非法，也不要只留下笼统 Valid 让用户以为所有阶段都完成。

`pipeline_graph.py` 的 `configuration/analyze` 可以移入现有 `pipeline/__init__.py` 或小型服务函数；删除其独立作者 CLI 和 README 中的第二条日常路径。无必要保留旧命令 wrapper；保留内部分析函数供脚本使用不等于保留旧格式。

### 6.3 与语义迁移分开的精简建议

下面不是遵守 DOC0–6 的前提，不应抢占 pipe/from 的实现优先级：

| 模块 | 建议 |
| --- | --- |
| `check_policy.py` | 当前只有 52 行，目录豁免未启用。可以保留以免本轮额外改已有选项；如果整合目标确定只保留显式 skip，可删除目录策略与 missing-hardware 分支及其专属测试。不要把它误称为必须存在的版本兼容器，也不建设新的配置系统 |
| `pipeline/output.py` | 保留 PNG 归属、manifest 检查与原子替换。当前 `GraphOutput.begin:68` 会在验证/渲染前删除旧图（`:82`），而 CLI 在验证前调用它（`pipeline_graph.py:64`）。若绘图成为默认行为，建议把写入/删除副作用延后，失败保留可由旧 manifest 识别的旧图；不增加 stale 状态库。是否仍需 all 时清理孤立图应独立决定 |
| `pipeline/model.py`、`render.py` | 当前 PhaseGraph/report 字段与图例应改成 step/event/新依赖类；其它成员折叠、issue/done 区分、环可视化可保留。内部 ID 格式无需为外观重新设计 |
| `yaml_source.py`、`errors.py` | 保留精确 reader、容量错误与错误分类；格式器所需源表示可只在格式操作读取。不要为了减少模块数混入 CLI、领域判断与写文件 |

## 7. 新增字符限制草稿：待确认输入与实际冲突

本轮只读 `git diff -- schema_docs/CHANGELOG.md` 已确认新增第 187–252 行。以下是草稿的输入地位，不是已冻结配置；本报告未修改该文件。

### 7.1 可直接转为预算配置、但数值尚未确认的条目

| 草稿位置 | 候选内容 | 需要落实到的当前字段 |
| --- | --- | --- |
| CHANGELOG:198–:202 | describe 150/1 行；name 50/1；math 1000/10；meaning 50/1；thesis 300/2 | 正式属性路径；不包括同名具名映射键 |
| :219 | dispatch.when 500/5 | `meta_data.dispatch.when` |
| :224 | tolerance 与 gap 100/1 | tolerance 是输出名到字符串；gap 是单字符串，需要确认草稿“每个 output”是否仅修饰 tolerance |
| :227–:229 | device/arch 50/1；execution/feature 100/1；stack/wiki 叶子单行 | 预算项与仅布局项分开；外部引用不能被改写 |
| :233–:239 | arithmetic、ledger、unknown、strategy、refill 150/1 | 仅保留候选限制；非空性和 unknown 允许位置仍按正式规范 |
| :246–:248 | item/choice/reason 300/2；feedback.choice 150/1 | item 本身为引用或引用列表，且 :252 又排除定位符，需明确 item 是否真的纳入预算 |

决定最终字符/行数及计数口径后，这些只改集中表，不重写结构校验器。是否需要额外检查尚未列出的 retention、来源或语义字符串，应由完整字段覆盖表显式说明；不能假定 pipeline 只剩 noun.describe 一个文本位置，因为 info.describe、index.meaning、storage.meaning 已存在。

### 7.2 不能作为普通格式化实施的条目

| 草稿要求 | 与现行规则的具体关系 | 本轮建议 |
| --- | --- | --- |
| :191–:192 全面不出现 YAML 引号，报 warning 要求去掉 | DOC0:65 为类型/歧义允许引号；DOC5:473 要求 `{...} ← [...]` 作为字符串使用引号或合规块标量。去掉可能变映射、布尔、数值或无效 YAML | 只安全去掉装饰性引号；必要引号不 warning。若用户坚持全禁，必须先解决合法表达和规范冲突，不能偷偷改 parser |
| :212 顶层全大写；:213 固定字段首字母大写 | 正式字段是 `meta_data / contract / design / realization` 等，固定结构区分键名 | 这是字段改名，超出纯排版；要有逐路径新字段决议并同步正式文档，当前不实施 |
| :212 step/role 全大写；:213 自定义名称全小写 | DOC4:358 推荐其它 step 小写、单 step MAIN；DOC5:598 只规定单 role MAIN；数学名保留大小写。改符号名还需修改公式/ref/bind/event 的引用，可能引发大小写碰撞 | 保留当前名字；不自动大小写转换。若要改约定，先明确范围及对语义字符串的影响 |
| :223 range/default 不得是字符串，可不传信息时删掉 | DOC1:199 明许 `0 <= route_id < E` 等约束字符串；:203 允许默认表达式；:233 使用 `default: None` 字符串表达真实缺省。删除 default 会把可省略参数变成必传 | 继续当前类型及含义；这不是格式器可以自动执行的删减 |
| :234 ledger 字段“可以为空” | DOC3:67–:69 要求实填非空，gain/cost 无内容时省略；DOC0:69 禁空值占位 | 若“为空”只是省略可选 gain/cost，无冲突；若意指空串/null/空 ledger，则需重新决定规范，不直接放宽 |
| :204–:207 所有新 entry 都单行、单项列表并到父字段 | DOC0:65 接受 flow/block；step/role 本身也是具名 entry；复杂记录可能有合法多行 math。举例中的 resource/prologue/steady 已不是当前字段 | 只形成有限路径排版表，不递归一刀切；单项列表仍是列表，不改成标量。草稿旧名字仅当说明，不新增同义字段 |
| :191 句尾不要点号、:247 删除非必要 require/prefer | 自动删标点或交接条目会改文字/设计，且用户已明确格式不删文字 | 保留为人工写作/审阅偏好，formatter 不执行 |
| :235 `unknown:<reason></reason>`、段落中的 metadata/algorithm/reshape 缩写 | DOC3:151 是 `unknown:<reason>` 前缀文本，其它段落使用正式路径 | 不新增 XML 标签、别名字段或新语法，把草稿意图映射到当前路径即可 |

以上冲突不阻塞按当前 DOC0–6 实施字段、from 和 pipe。真正需用户最终确定的是预算表与排版表；如果想采纳大小写/类型/空值变更，就先作为规范变更单独确认，不能在“formatter”提交里偷带。草稿 :250 的“不用体现在 docs”可以适用于纯工具预算偏好，不能用于掩盖字段语法与正式 DOC 不一致。

## 8. 目录、starter、四案例与关键验证

### 8.1 任务目录的可检查部分

DOC0:150 要求每个任务都有中文 dispatch，包括只有 root 的任务；当前 checker 只读取 YAML，没有任何 dispatch 解析。建议用一个小的纯校验模块（例如 `rules/dispatch.py`）接受 Markdown 文本和已发现的文件集合，由 `schema.py` 读取并编排，不建设 Markdown 编辑器。

机器检查限于：必需的“范围”“完整设计”节、完整设计表的当前 YAML 链接覆盖、链接目标存在、明确变体记录的完整基线链接存在、当前完整数量。表格应接受正常 Markdown 写法，不因空白或短条目合句而报错。参数/局部/组合是否分类正确、独特完整价值、语义定位对象与相容性仍由审阅核对；dispatch 不作为 patch、隐式 parent 或补齐 YAML 的来源，也不加入 effective 内容继承。

保持目录只扫描直属 `.yaml/.yml`、文件名 ID、重复扩展名检查。不得把 11 份里的一个改放子目录、藏为历史或 provider-only 文件来绕上限。

### 8.2 Starter 和 README

`starter.yaml:30` 仍有 rewrite.step，`:34`–`:55` 为旧 phase/role 模式/pipe.steady。阶段 6 同步改成当前字段与 MAIN，补新增 meaning、role describe 和 using_verb 外壳。它仍是编辑骨架，不能伪造 source/golden_reference 或设计参数来让它自动通过；明确 null 占位是未填写，正式 plan 不因此允许 null。必要可选结构在 README 用小片段说明，不把所有可选字段或配置菜单塞进 starter。

作者 README 保留一个入口、三个结果层次（结构关联与预算、有限样本图、人工语义审阅）、from/共享来源和 explicit skip 的说明。工具 README 记录模块职责、当前有限图边界、无损格式约束及少量维护命令；删除旧 parent/phase/default-data 的实际支持说明，不能将仍未完成的 Extract 宣称已可运行。

### 8.3 四案例只在最后阶段处理

现有索引明确共 21 份 YAML：FA4 prefill 11、decode 7、GDN 2、QSA 1（`examples/README.md:9`–`:18`）。四份 dispatch 仍是父树形式；FA4 prefill 的 11 份已超当前上限。另一个具体证据是 `fa4_decode/root.yaml:222`–`:225` 将 SplitCount 的多组 workload 选择留在 YAML 中；`:312` 等缓冲配置也须随所选具体方案核对，不能仅做字段替换就称为当前具体方案。

最后阶段要做的工作，不在本报告展开新一轮算子调查：

1. 对照已有材料审定每个任务的实际具体 root、必须保留的独立完整设计、参数/局部差额与已知组合。不预先按旧父树、文件数或名称定结论，也不为了恰好 10 份只删一份。
2. 按所选具体参数修订三 Design 区与 when；保留当前执行中的尺寸、尾块、空工作、队列/终止及数值分支。其它配置只写目录，不把它们藏成 min/查表/自然语言 selector。
3. 同一完整区按真实来源使用 from；修改过的区完整本地化。局部记录直接指向保留的当前完整基线及既有对象，不形成多层 patch，也不扩成参数笛卡尔积。
4. 保留来源已知的数值、原语与状态缺口；例如索引列出的 decode PageIndex、QSA padding/HalfMax、GDN 描述符在途读取问题（`examples/README.md:60`–`:70`），不能靠加等待或删动作“修成可通过”。GDN 的真实转换提交及未被主 kernel 消费的结果也不因整理而消失。
5. 列出全部当前完整设计的保留理由，补 canonical dispatch 范围/完整设计/有内容才有变体；同步 name/describe、版本、引用和 README 数量。schema 内容改动才递增对应文件版本。
6. 完成新工具关键验证后，实际实施阶段才最后更新保留节点的 PNG 与复现命令，处理被删除节点的已确认生成图。本轮不做任何重渲染或覆盖。

### 8.4 必要的关键验证集合

这些是实施时的验收建议，本轮没有运行：

| 核心对象 | 保留或重写的少量关键检查 |
| --- | --- |
| Reader/格式 | Decimal、科学计数外观字符串、布尔外观字符串、nullable/default/合法 null、流式/块式、带逗号 waiting、多行标量、anchor/alias/merge；格式后类型和值一致、幂等、不修改引用；只读模式字节/权限/mtime 不变 |
| from/来源 | 独立来源与多跳链；跨区成环；root 禁引用、目标不存在/混合字段；使用节点接口不匹配；中间来源漂移影响身份、无关来源不影响；standalone 自带完整内容且不读兄弟文件 |
| 结构/关联 | 当前字段最小正例与条件缺失反例；MAIN；step/launch 一次性、stride/count；Contract ref/tolerance；storage 字面量占用与 view；feedback 平级；合法符号表达/开放类型正例 |
| 有限图 | 两个相邻 for 顺序；if 真/假与空 worktile 仍保留入口/收尾；目标不存在无边、目标存在缺前驱报错；变量移位边界与固定缺失；noun 签名；当前成员和 [*]；三模式、ordered 穿过 serial；不再凭同名加数据边 |
| CLI/输出 | 任一节点检查全任务；显式格式/图选择；图缺样本返回未完成；Graphviz 缺失不影响纯检查；来源漂移不发布新图；输出冲突保护；dispatch 覆盖及超过 10 个完整节点 |

旧测试按所保护的含义分类：reader、精确标量、来源定位、只读性、声明窗口、成员与输出归属保留；父树/直系 parent 限制、角色默认模式、自定义基础动词、旧 ledger、旧默认数据边、旧命令及旧例子文件数量断言删除或重写。`test_design_resolution.py:203` 的超长父链、旧容量规模不应成为维持超过 10 个当前完整节点的理由；容量保护保留针对实际数据/alias/展开的关键检查即可。不为旧版输入建立一套专门识别测试。

## 9. Extract 的下一阶段边界

只保留少量被实际调用的 Template 能力，后续由 Extract 自己对接：

| Template 能力 | 只读确认的调用点 | 本轮处理 |
| --- | --- | --- |
| `yaml_source.load/loads`、Document.value/positions/source_sha256、错误类型 | `schema_extract/tools/artifacts.py:28`、`:126`；冻结包动态加载 | 保留同一权威 scalar/alias 语义与实际源位置；格式模块不得改变只读 parser 的含义 |
| `schema.resolve_directory` 与 `ResolvedTask` 的 raw/effective、origins、source_manifest、effective_identity | `artifacts.py:45`–`:87` | 实现当前 from 图并保持有用接口形状；Extract 的旧 parent 批准/计划逻辑不在此伪装兼容 |
| `rules.validate_algorithm` | `artifacts.py:513` | 保留 Contract-only 只读结构/关系入口；不强迫它读取 Design 或 dispatch |
| `validate_resolved_file / effective_document / validate_standalone_file` | `schema_extract/tools/standalone.py:67`–`:89` | Template 侧保证当前内容、来源和验证边界；不改 Extract 的导出回执或运行流程 |
| 普通检查 CLI 的显式 skip、错误状态 | 现有 Extract 外部调用协议 | 保留或明确记录变化，交给下一阶段；不在 Template 中增加旧格式旁路来使 Runner 继续可运行 |

`artifacts.py:94`–`:117` 仍按旧 parent 找可重开 provider，`EXECUTOR_ALIGNMENT.md:19`–`:25` 也明确 Runner 未对齐。这些是下一阶段的接口使用方改造，不是本轮应该顺便修改的源码。完整/局部方案选择、P1 最小 execution 写权限、批准/回执、历史运行恢复和 schema wiki 均不进入本轮实施范围。

## 10. 交给整合者的取舍

必须落实的语义只有一套：当前字段、三区合并无环 from 图、使用节点绑定、显式 pipe/事件、正确成员语义、无默认同名数据边、raw 与视图分离，以及最后按 common 整理四案例。上述不需要再次讨论是否兼容旧版。

整合者需要统一的实现选择是：默认命令是否只读或顺带格式/绘图；是否保留明确标识 not_run 的草稿预览；格式器采用 PyYAML 源事件还是一个私有 round-trip 后端；是否顺手删除未启用目录硬件策略；图失败时保留旧 PNG 还是沿用失效删除。建议以最小实现为准，不为这些选项新增模式矩阵。

真正缺少的用户输入是最终的字符/行数与排版表，以及是否打算将新增草稿中改变字段大小写、range/default 类型或空值语义的提议升级为正式规范。默认继续以 DOC0–6 为准；这些输入只阻塞相应预算/布局决策，不阻塞字段、from、pipe 的实施。
