# 规则覆盖与审阅边界

规范依据是 schema_docs/1–6 的正文规则与 CUE 定义，案例不覆盖规则。以下按规则族对应实现位置；六份文档的全部字段结构分别由同编号 CUE 文件给出，不另维护一套 Python 字段 schema。

“CUE”表示 kernel 包直接约束；“关联检查”表示本目录的 [tools/validate_relations.py](tools/validate_relations.py) 对已经通过 CUE 的具体数据进行检查；“审阅”表示需要数学、资料、实现或运行证据，生成成功不会把它当作已证明。starter.cue 仅为编写骨架，不参与规则定义。

## 通用及任务组织

| 规则 | 实现与验证 |
| --- | --- |
| 固定对象的字段、类型、必填/可选、非空要求 | CUE；本目录 [tests/test_schema.py](tests/test_schema.py) 的结构反例覆盖错误类型、缺项、多余字段、空壳 |
| 自定义名称、phase/节点 ID、开放硬件名称 | CUE；硬件名称保留空格、点和斜杠，不检查 execution/feature 跨表重名 |
| 英文内容、单一职责、删除测试、量化理由、篇幅锚点 | 审阅；不以 ASCII 限制误伤数学符号，也不把“通常十条”或“约 50–150 行”变成硬上限 |
| shape 为有序单项轴映射，标量 []，大小非负 | CUE；关联检查核对轴名唯一和结构化表达式语法 |
| YAML 行内映射/列表与展开写法 | 同一 CUE 定义校验解析后的数据；回归测试验证两种排版等价、同样拒绝错误类型及多余字段，排版不成为约束 |
| 表达式在契约域内取整数、变量有定义、整除条件成立 | 审阅；工具不执行表达式或从正文建立一个新 DSL |
| 省略、null 与空容器 | CUE；仅允许文档列出的有意义位置，包括单槽整槽空视图 |
| root 独占 Contract，派生节点的完整 Design、身份和相同任务描述 | CUE #Task；派生 ID 与键一致，root 不设 parent |
| parent 存在、无环、最终到 root | 关联检查；无 parent 时按 root 解释 |
| 单 root 的显式 when | 关联检查固定全域写法 |
| 多节点 when 互斥、共同覆盖契约域，parent 不附加选择条件 | 审阅；when 保留自然语言和数学文本，不进行布尔求解 |
| 当前生成流程的源与产物一致性 | 经精确 JSON 值紧凑排版，再由 CUE 回读核对；check 逐字节比对。当前生成入口仍不采用旧手写产物；这不限制 CUE 本身校验 YAML |
| 修订号为正整数、内容更新递增 | CUE 类型与生成器相对上次清单检查；历史责任由版本库审阅 |
| Designer、Coder 与 Contract 的修改权限 | 协作规则与版本库审阅；模板不是身份认证或权限系统 |

## 1. Algorithm requirement

| 规则 | 实现与验证 |
| --- | --- |
| constant/input/output/compute/symbol/golden_reference 全部字段 | 1_contract_algorithm_requirement.cue |
| tolerance 键恰好覆盖所有输出 | CUE 两向必填约束，缺项和多项均拒绝 |
| 数值格式为开放名称，Contract 不接受 bind/unknown 属性 | CUE；格式是否有真实定义需审阅 |
| shape/dtype 属性引用只在 Contract 内、属性匹配 | CUE 引用形式；关联检查存在性、先定义后使用及循环 |
| 常量、输入输出、登记 symbol 不重名；输入 shape 引入的规模不重复为对象 | 关联检查；公式中的局部规模/下标按作用域审阅 |
| 输入 range 的数值字面量端点顺序 | 关联检查；符号范围、dtype 范围和 default 有效性需审阅 |
| 常量不依赖运行时输入；重要对象必须登记 | 审阅 |
| 完整数学关系、边界、转换、FP32/RNE 默认、数值路径、标量/非张量区别 | 审阅；类型约束不证明公式含义 |
| golden source 带函数入口，版本引用非空且不为 pending | CUE；真实存在、固定版本、独立 oracle、gap 与误差判据含义需核验 |
| 尚未定义格式的 workload/优化目标字段 | 固定结构拒绝自行扩展，等待语言规范明确 |

## 2. Hardware specification

| 规则 | 实现与验证 |
| --- | --- |
| 固定结构、非空 execution/feature、字符串摘要 | 2_contract_hardware_specification.cue |
| stack.template 与 wiki.programming.template 同时出现 | CUE 存在性约束及正反例 |
| 每个 wiki 叶子单个引用，可显式占位 | CUE 引用形式；工具不把占位当作绑定完成 |
| 设备、架构、栈和摘要不能用 wiki 占位代填 | CUE #HardwareFact |
| virtual_isa/machine_isa 的条件必填 | 需根据实际栈审阅；没有硬编码某厂商或语言的层次表 |
| 具体型号、版本匹配、资料存在性、hardware 指向型号文件、冻结 commit | 外部资料核验 |
| 同设备事实一致、feature 删除测试、物理层级与能力的真实性 | 审阅；不从案例推导封闭的设备/指令/存储枚举 |

## 3. Math reshape

| 规则 | 实现与验证 |
| --- | --- |
| thesis/rewrite/symbol/arithmetic/ledger 的字段与条件 | 3_design_math_reshape.cue |
| step 唯一、ledger.by 指向步骤、九类资源、非空变化账本 | CUE；回归测试含错误 ID 与 baseline-only 账本 |
| symbol.value 与 binding 的定义权互斥 | CUE |
| life 为 carried/stored，携带或物化值具有 shape/dtype | CUE；生命周期范围与公式一致性需审阅 |
| 本区登记符号唯一、不重新登记 Contract 常量和输入输出 | 关联检查；沿用 Contract 中间量时的数学同义性需审阅 |
| 下游数据块取用的登记 symbol 必须有 shape/dtype | 关联检查直接的具名数据块接口；仅在文本里出现的接口需审阅 |
| ref/bind/unknown 的形式及所在字段 | CUE；跨区目标若只定义在 strategy、math 或动作契约中，人工解析其实际定义 |
| 公式、精度分叉、初值、物化/重算、改写等价性、账本数值与误差界 | 审阅与数值证据；不从字符串臆测执行或证明结果 |

## 4. Work schedule

| 规则 | 实现与验证 |
| --- | --- |
| flow/phase/decomposition/mapping/schedule 的字段 | 4_design_work_schedule.cue |
| phase 恰好引用一次、多 phase launch 必填 order | CUE |
| partition 为空表示一份工作；非空必须登记完整 worktile_order | CUE 检查维度覆盖和引用；group 的实际排列含义需审阅 |
| invariant offset 与 shape 同秩，iteration 必填 direction | CUE |
| preassigned 两种写法互斥，正步长/非负数量，refill 与 exit 形式 | CUE；关联检查字面量和表达式语法，none 时必须有预分配序列 |
| worker/placement 的 unit 来自 execution，字面量成员数量一致 | CUE；成员区域与端点语法由关联检查核对 |
| 同 launch 顺序 phase 共享 worker、per_worktile 配对 | 关联检查明确的结构/数值冲突；符号表达式等价、实际归属和协议进展需审阅 |
| handoff 参与者、storage 资源、ready/wait_at 的 phase 和里程碑 | CUE + 关联检查 |
| 局部资源只在同一次提交的同一 worker 实例中交接 | 关联检查；物理可见性和实际存续需审阅 |
| start_after 引用、自等待、kernel_done 启动依赖环、launch_ready 协议 | 关联检查；含局部事件的协议活性需事件图分析 |
| 生产/消费区域覆盖、tile_reduction 方法正确、初始化必要性、工作覆盖与均衡 | 审阅；不按 shape 相同猜测对象同一，也不执行 strategy |

## 5. Tile pipeline

| 规则 | 实现与验证 |
| --- | --- |
| phase 与 work_schedule 一一对应 | CUE #Design |
| lane/resource/view/domain/milestone/pipe/edge 的全部结构 | 5_design_tile_pipeline.cue；pipe 真正递归，不预设深度 |
| pipe 至少有动作区域、repeat 有 steady，列表和边非空 | CUE |
| resource 槽数、已知字面量槽位上下界、多槽视图、成员编号 | CUE + 关联检查；符号槽数与符号索引的范围需审阅 |
| part 引用父资源轴、字面量区域不越界 | 关联检查；类型重解释容量及逻辑值覆盖需审阅 |
| 同 worker 顺序 phase 的资源只定义一次；不同 worker 可同名 | 关联检查 |
| 执行成员的登记单元、半开选择区间和已知 worker 边界 | 关联检查；硬件层级合法性和同时活跃的重叠成员需审阅 |
| 共享域、flatten 引用及循环、repeat 域与局部作用域 | 关联检查；不会把循环域展开成具体时间线 |
| 当前循环索引仅在 steady 绑定，域查询不是变量读取 | 关联检查常用记法；数学值的负初始索引仍合法 |
| 动作、MMA 双操作数、事件后缀、lane/成员、动作族与自定义动词契约 | 关联检查；仅核对命名身份与索引元数，不证明每个索引实例存在 |
| 动态 refill 的 worktile 域不使用 count/last | 关联检查共享域；局部或数学上的间接依赖需审阅 |
| 展开后动作唯一、移位窗口、固定索引事件存在、默认 data 边、mode 顺序 | 语义约定与后续事件图分析；当前检查不宣称这些已证明 |
| 发布/复用、安全回收、别名读写保护、里程碑的真实通知、死锁、overlap | 实现/硬件审阅与后续分析 |
| space 名称沿用硬件资料、寄存器与片上容量记账 | 审阅；不把 feature 的键误当作完整的存储层级表 |

## 6. Realization

| 规则 | 实现与验证 |
| --- | --- |
| require/prefer/forbid 的固定结构，item 单值/列表，choice/reason 必填 | 6_realization.cue |
| feedback 只允许在 prefer，保留原始选择与理由 | CUE；无法通过 feedback 覆盖 require 或 forbid |
| evidence 单个版本记录引用，不使用占位 | CUE；真实性与实验支持程度需核验 |
| 重复 item 字面引用 | CUE list.UniqueItems；裸名与限定名是否同对象需语义解析 |
| item 上游语义定位，未登记的数学对象、相互冲突的约束、是否侵入 Design | 审阅；不建立另一套会迫使用户重复声明全部符号的对象注册表 |

## 验收方式

`make check` 检查 CUE 格式、测试及已迁移任务的生成一致性。测试使用明确标识的合成 fixture 验证边界；其测试引用不充当实际论文、硬件或实验来源。GDN 编写示例的用途与证据状态在它的 README 中说明。

需要审阅的规则仍属于规范，不是可忽略的建议。自动校验范围与审阅责任明确分开，避免为了让所有规则“机器可判定”而强迫公式、策略和动作契约变成另一门编程语言。
