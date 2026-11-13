# schema_docs 修改记录

## 2026-11-13

### 文本字段的填写重点

- `unknown` 点明具体信息缺口及受影响的判断；三类 `strategy` 只补结构与上游未表达的关键关系或条件。
- `refill` 聚焦工作来源、领取粒度、分配状态与失败条件，沿用已有定义；计数器示例压为一句，保留初态、原子领取及停止领取条件。
- `choice` 聚焦具体选择、作用位置及必要条件，理由与实验依据分别由 `reason`、`evidence` 承担。

## 2026-11-12

### 说明文字与示例精简

- 公共约定明确说明文字只补对象、作用与设计依据，沿用已有字段、公式和引用，使用常见术语与明确名称。
- 区分 `meaning / describe / thesis / strategy` 的职责；`math` 聚焦公式，`range / default` 保留有效域与默认值表达，`prefer` 以具体成本为依据。
- 动作输入仍须完整、可唯一确定；`describe` 只补上游及已声明关系未确定的部分，消除逐项重复列写的要求。
- 精简各章示例，去掉句末句点和装饰性引号，保留 YAML 必需引号。移除 ledger 的字符数与行数限制、storage 的强制单行要求；数量限制与统一排版留给后续工具。

### 对象说明：meaning 与角色 describe

- 第 4 篇的 `partition.<dimension>`、`worktile.invariant.<value>`、`worktile.iteration.<value>`、`target.single_tile.<value>` 增加必填 `meaning`，分别说明本层工作坐标、输入块及结果的含义，沿用上游数学定义。
- 第 5 篇的 `storage.<storage_id>` 增加必填 `meaning`，说明存储区域用途；`role.<role_id>.info` 增加必填 `describe`，概括角色职责。noun 的 `execution_mode / describe` 保持不变。
- `index.<index_name>` 统一为 `meaning / definition` 映射，两项必填；`definition` 沿用半开范围、`worktile` 或 `{flatten: [...]}` 三种域定义。遍历、切片、连续编号与索引作用域语义不变。
- 新增说明均为简短、非空的英文字符串，放在各条目属性首位；同步结构、字段表、条目顺序、正文与示例，包括第 3 篇的调度数据块示例。

### Flow：启动绑定、调用边界与交接默认

- 删除 `flow.launch[].protocol`，同步第 4 篇的结构、字段表与开放取值。保留 `launch_ready`，由第 5 篇的动作契约说明触发条件与参与范围，第 6 篇绑定具体启动机制及所需保证。
- 明确计算请求的入口与输出交付约定，不增加内部独立提交之间的顺序，也不要求 host 阻塞。外部供给和调用前提由相关 phase 的 `schedule.strategy` 简明记录；外部准备及共享状态不从计算入口推导同步。
- 片上交接先按 worker 匹配对应存储副本；`wait_at` 可引用空工作集合下仍存在的入口事件，循环外消费范围由动作 `describe` 给出。第 4、5 篇的交接说明保持一致。
- 收紧 `initialize`：只补本次交接版本尚缺的消费初值，沿用入口已给定或数学定义保留的初态，区分首次初始化与本次补值。保留显式 `storage` 和已有接续粒度。
- 明确各层依赖共同成立，已有前驱覆盖的 `ready/wait_at` 可省略，相同 handoff 属性可用现有 YAML alias 复用。通用正文移除录图与失败恢复的案例展开，保留实际提交原则。

### View：part 更名为 region

- `view.<value>.part` 更名为 `region`，表示槽内的位置与覆盖范围；同步第 5 篇的结构、字段说明及示例，区间形式、坐标单位和省略语义不变。

### View：区域绑定与数值定义

- `view` 仅保留 `slot_index / part`，取消 `dtype` 及省略时继承父类型的数值解释。父 `storage.dtype` 仍必填，用于容量和 `part` 坐标；数值类型与转换沿用当前 Design 的值或输入输出定义，必要本地格式在已有动作或 realization 绑定中定义一次。
- `part` 省略表示整槽的保守占用与冲突范围，不表示填满或访问全部字节。已有交接支持整槽先后复用时可省精确位置；子区并存、部分数据仍存活和部分覆盖时保留区域约束，不以额外全组等待改变重叠。
- 同步第 5 篇的结构、字段、绑定说明及两个 S/P 示例，保留逐 view 槽映射和容量一次计账。算子实例、模板与检查工具后续另行迁移。

### Phase 公共定义：storage / index / event

- `tile_pipeline.phase` 下的 `resource / domain / milestone` 更名为 `storage / index / event`，分别记录片上存储区域、有序索引和供跨 phase 引用的已有事件集合。同步修改第 5 篇的结构、字段表、正文及示例，第 4 篇的存储路径和事件引用说明；对外仍使用 `<phase_id>.<event_name>`。
- 存储按设计区域列账，互不重叠的区域可分列，API 如何合并申请由实现决定；共用字节通过同一 storage 的 view 表达并只计一次。保留 `slot_index / part / dtype` 与必填 `total_size`，容量包含设计所依赖的分配粒度；影响驻留的额外预留或实际申请量由 realization 说明。
- index 允许单 role 登记 worktile/flatten 序列；纯局部范围和条件使用 for/if。view 形参或局部实参不再单独触发公共索引登记，动态端点在实际遍历前须已定义并就绪。
- event 定义为已有事件的引用集合，明确形参代入、role/member 和 worker 实例身份，以及跨 phase 的值、版本和区域匹配。事件存在性沿用 pipe，空工作序列不删除循环外入口动作；默认交接或已有关系足够时不重复登记别名。
- 事件集合用于 ready/start_after 时要求全部前驱满足；用于 wait_at 时，就绪条件分别约束每个消费事件。wait_at 按消费者覆盖，未被点名的消费者保留默认入口等待。
- 上述语义直接融入字段定义，正文不保留改名历史或研究过程；调查依据见[六算子研究记录](../schema_discussion/phase_grammar/README.md)。本次只修改规范文档，算子实例、模板与检查工具后续另行迁移。

## 2026-11-11

第 1–4 节记录本日首轮仅涉及 `schema_docs/` 的改动；后续决定追加记录，保留首轮决策的历史背景。

### 1. `3.design.math_reshape.md`：去掉 `rewrite[].step`，`ledger` 下沉

- 删除 `rewrite[].step` 字段。`rewrite[]` 现在是 `name / math / symbol? / arithmetic? / ledger?`，与 Contract 的 `compute[]` 保持同一风格；列表顺序改为“按先定义、后使用排列，只表示定义关系”，不再有步骤 ID。
- `ledger` 从区域级下沉到每条 `rewrite`，与 `math`、`symbol`、`arithmetic` 平级，仍然可选。
- `ledger` 只保留 `baseline / gain / cost` 三个字段，各写一行、不超过 150 字符；原来的 `<resource>[].item / change / by` 全部删除。`ledger` 出现时 `baseline` 必填，`gain`、`cost` 按需填写，某一侧没有值得记录的变化就省略。
- 删除“账本资源键”一节（`matmul / vector / transcendental / traffic / footprint / depth / pass / comm / error` 封闭集合）。`gain`、`cost` 改为自由文本，计量约定（MAC、字节、元素、物化量、遍历趟数等）保留在“账本”一节；新增误差来源改记在 `cost`。
- 同步更新：核心思想的区域说明、设计着眼点表的“落在哪里”列、固定结构与扩展边界表、`unknown:` 的适用范围，以及账本示例（改为单条 rewrite 的 `baseline / gain / cost`）。

### 2. `4.design.work_schedule.md` / `5.design.tile_pipeline.md`：单 phase、单 role 统一叫 `MAIN`

- `4` 的「名称与顺序」加一句：整个区域只有一个 phase 时，phase ID 统一写 `MAIN`，不另起算子名。
- `5` 的「成员与名称」加一句：一个 phase 内只有一个 role 时，role ID 统一写 `MAIN`；phase ID 沿用 `work_schedule`，单 phase 时同样是 `MAIN`。
- 同步改掉文档里受影响的示例：`4` 的最小 `flow`（`attention` → `MAIN`）、`5` 的单 role 示例与跨 phase 交接示例（`Main` → `MAIN`）、`6` 的 `design.tile_pipeline::main.MMA` → `MAIN.MMA`。

### 3. `6.realization.md`：`feedback` 提为与 `require` 平级的 Coder 专区

- `prefer[].feedback` 取消，改为顶层分组 `realization.feedback[]`，排在 `forbid` 之后。
- 条目字段为 `item / choice / evidence`：`item` 指向被替代的那条 `prefer` 所约束的同一个上游对象，`evidence` 仍是单个固定版本的实验记录引用。
- `require / prefer / forbid` 的字段和写法不变，仍由 Designer（Planner）填写；`feedback` 是本区唯一由 Coder 填写的分组，只针对 `prefer` 提替代，不为 `require` / `forbid` 提供豁免，也不改变任何一组的约束强度。
- 新增一个 `feedback` 条目的 YAML 示例；`0.common.md` 的字段索引同步改为 `realization.feedback[].evidence`。

### 4. `5.design.tile_pipeline.md`：动词表、pipe 语法、依赖记法与资源字段的整体改写

本轮只改 `schema_docs/`。`schema_template/`、模板检查脚本、`kernel_wiki/`、`schema_wiki/`、`schema_extract/` 及各目录 README 尚未跟进，需要另行迁移。

#### 4.1 基础动词：八词封闭表，`info.verb` → `info.using_verb`

- 动作词收敛为四类二分的八个基础动词：`compute` / `matmul`（数值计算）、`load` / `store`（搬运访存）、`await` / `signal`（顺序同步）、`acquire` / `release`（资源占用）。原通用表的 `publish` / `produce` / `zero` / `mma` / `wait` 全部取消，并在文中写明各自的归宿。
- `info.verb` 改名 `info.using_verb`，**必填**，结构为 `using_verb.<verb>.<noun>: {execution_mode, describe}`。它是白名单：pipe 中每个动作必须唯一匹配一个已声明的 `(verb, noun)`。原「`info.verb` 是补充，不是白名单」一句删除。
- `meaning` 改名 `describe`。noun 声明键只写基名，矩阵乘的 `A@B` 整体作一个键；同一 `(role, verb, noun)` 只允许一种固定参数个数。
- 新增易错点清单：发布类动作按内容归入 `compute` 或 `store`；搬运的完成协议（事务字节计数预告一类）纳入搬运动作的完成保证，不单列为 `signal`；「已发射且源已读完」不是 `store` 的 done，需要时拆成 `store` + `release`。

#### 4.2 执行模式：三值改名并下沉到 noun

- `serial` / `async` / `async_ordered_completion` 改名为 `serial` / `async_unordered` / `async_ordered`。
- `info.execution_mode` 从 role 删除，`verb` 级覆盖写法也删除。模式只声明在 noun 条目上，无默认、无继承。
- 明确 `async_ordered` 中的 ordered **只指 `done` 的累积覆盖**，不指发起顺序；累积完成跨 noun、跨 verb、跨循环与 worktile 接续，中间的 `serial` 不重置该序列。新增一条约束：两个引擎各自有序不等于共同有序，没有共同保证用 `async_unordered`。

#### 4.3 pipe：有序列表 + `for` / `if` / `while`

- 取消 `repeat` / `prologue` / `steady` / `epilogue`。`pipe` 改为非空有序列表，每项是动作字符串或只有单个键的控制节点。
- 控制结构只有三种：`for`（域名、切片、`n[t]` 窗口、局部范围 `j=0:2`）、`if`（当前位置判断一次，两个 `if` 无隐式配对）、`while(r, condition)`（局部轮次编号，先判断后执行）。
- 首轮曾将 `while` 写入规范（后续已按第 5 节移除），同时写明验证代价：有限展开的检查工具无法展开动态 `while`，使用它的设计必须另附有限执行见证说明轮次上界，否则按「未验证」处理。
- 补入索引作用域规则：循环索引不泄漏到兄弟项，内层不得重绑外层索引，遍历 flatten 域须与外层坐标一致，`count/last` 不适用于动态域与 `while` 轮次。
- 原「空域不删除 prologue/epilogue」「动作不存在 ≠ 动作无数据可处理」等语义按新语法改写保留。

#### 4.4 依赖：`waits_for` → `waiting`，新事件记法

- 字段改名 `waiting`；三类分类键 `data` / `reuse` / `sync` 改名 `prepared` / `released` / `allowed`，值改为**非空字符串列表**。
- 每条规则写成 `"{issue B} ← [{done A}, {done C}]"`：两端时点必须显式写出，无位置默认值；单前驱也保留 `[]`。
- 动作引用改为 `Role[member]::verb:noun(indices)`。`::` 分隔角色，`@` 只保留在 `matmul:A@B` 的动作内部。
- 新增四种 `issue`/`done` 组合的含义表。
- **成员省略规则变更**：省略成员下标一律指当前 worker 成员，不再按双方 role 的布局自动取「对方全部实例」；跨成员必须写明编号或 `[*]`。
- `milestone` 没有当前 role / member 上下文，所列事件必须写成完整形式并显式给出 `Role[selector]` 与时点。

#### 4.5 默认依赖：取消同名数据边推导

- 删除「消费动作的对象名与某个 `load`/`publish` 完全相同且来源唯一时默认等待其 done」这条规则。**唯一的默认依赖是 role 内按各动作的 `execution_mode` 推进。**
- 理由：noun 是 role 局部名称，不具备全局对象身份；按名字推导会连错对象，误加的边还可能成环或等待当前分支下并不存在的事件。
- 配套要求：每个 noun 的 `describe` 必须逐项列明跨动作输入的值、版本／区域／副本及来源动作与成员；Reviewer 按「输入完整性 → 从就绪到实际读取的安全保证」两步核对，允许由执行模式、`waiting` 传递路径或明确绑定的原语保证覆盖。
- 对十个算子 baseline 的实测：真正需要补写的跨 role 边共约 9 条（MLA 6、FA4 decode 1、GDN 1、QSA 1），其中 8 条左侧是原 `wait:` 动作（新词 `await`），显式写本就符合该动词的定义。

#### 4.6 资源：字段改名、新增 `total_size`、`view` 保留

- 字段改名：`slot_count` → `depth`，`space` → `location`，`member_slot_shape` → `shape`。五个必填字段按 `depth / location / shape / dtype / total_size` 的顺序书写，没有 `view` 时整条写一行。
- 新增 `total_size`：这块分配在一个 worker 内的总占用，等于 `depth × 单槽字节数 × 持有成员数`，**统一以字节记**（整数或整数表达式）。省略 `member` 时持有成员数是 worker 的全部成员数，不是 warp 数。它是派生校验值，须与其余字段构成的等式一致；`view` 条目不重复计入；多 CTA 的合计不能直接对照单 SM 容量。
- `view` **保留** `slot_index? / part? / dtype?` 三个子字段。原计划删除 `view`、由 `depth` 加 `released` 边表达槽轮转，经语料核对后撤回：1510 个带索引的 view 条目中 350 个是复合或条件表达式（`(c-c0)%Stages`、`(g-cLo)%(D==128 ? 2 : 3)`），434 个（29%）的轮转索引不是键的最后一个实参（`S(i,d) → i%2`、`Kc(i,d) → (2*i+d)%4`），当前 FA4 模板同一文件内即有三种不同映射。版本到槽的映射是不可推导的设计选择。
- 文中明确 **`slot_index` 不设任何默认推导**，`depth > 1` 时必填，`depth: 1` 时省略。`slot_index` 与 `part` 的端点可引用 view 键上的形参及已定义量，但不创建循环、不改变索引作用域。
- 删除「多槽资源必须登记视图并明确槽位」的强制要求（改由 `slot_index` 的填写条件表达），并补入「一块分配只登记一个 resource，`total_size` 只计一次，视图不新增分配」。
- 保留两条原有语义并加强措辞：共用存储本身不产生数据边、发布或同步，覆盖安全只能由 `released` 边或写明的原语顺序保证；新分配不代表内容已清零。FA4 的 S/P 共用示例补充说明选中的是每行后半段，不是整块连续的后 32 KiB。

#### 4.7 其他同步改动

- 核心思想的四类信息表：「索引与重复」→「索引与控制」，「依赖与发布」→「依赖与同步」。
- 设计着眼点表的「落在哪里」列全部换成新字段名。
- 结构轮廓、主体字段表、role 字段表、固定结构与扩展边界表按新结构重写；扩展边界表新增「封闭取值」一行，覆盖八个动词、三种模式、三个分类键和三种控制结构。
- 「与上下游的职责边界」一节为 `total_size` 开明例外，说明它是「能推出的数量不另设字段」这条原则的唯一例外及其理由。
- 全部示例改写：单 role 最小示例、两段独立循环、递推 chunk、FA4 的 S/P 共用、多槽 `ScoreProbability`、两槽缓冲的 `prepared`/`released` 边、QSA 的 `done ← issue` 边、跨 phase 交接的 milestone、条件等待示例（原 `q: [0, "min(1,t)*(1-Hoist)"]` 的 0/1 假域改用 `if`）。
- `3.design.math_reshape.md`：`pipe.repeat` 的层数 → `tile_pipeline` 的循环结构，不再引用已删除的字段名，也不绑定具体 pipe 语法。

### 5. Pipe 控制裁剪：只保留 for / if

- 从正式规范移除 `while` 的定义、示例及专属作用域／验证条款；结构轮廓、正文与封闭取值统一为有序列表加 `for`、`if`。
- 增加兜底提示：未来 Agent 如确需 while，须先声明具体需求并提出 issue，讨论是否纳入；在此之前只使用 for 和 if。
- 同步讨论页的当前控制规则，旧 `pipe-control-v0.2` 试写与检查结果标为历史记录；while 的研究与证明单独保留，不作为另一套可用语法。本次不迁移旧检查器、算子或 Kernel 源码。
