# Template 更新独立规划：流水线、格式与统一入口

本报告以当前 DOC0–6 为目标，以 CHANGELOG 核对历史差异。只读研究了正式文档、Template 两份 README、starter、四案例索引、解析/规则/绘图实现与关键测试；没有阅读另一份独立报告，没有运行 Extract，没有修改正式代码、规范、YAML 或 PNG。做过两组只读内存解析试验，具体结果见下文。本文是供下一位整合代理判断的计划，不是实施结果。

## 1. 主要判断

1. 本轮不是一次字段替换。raw/effective 的引用图、pipe 的控制模型、动作声明及事件语法同时变化，应先更新共享解析与结构/关联规则，再更新有限事件展开，最后统一入口和整理案例。不能把旧图成功当作新规范的验收。
2. 图构建中必须删除按同名 load/publish 推导数据边，以及省略成员时自动取对方全部实例。新图的局部边来自动作自身 issue→done、role 的实际 pipe 顺序、noun 级 execution_mode 和显式 waiting。flow 的默认交接是另一层规则，不能因为局部默认边减少而一并删除。
3. 静态合法性和有限样例能否展开必须分开。普通检查不要求公式是 Python，不求执行结果；图可以支持一个有边界的整数/条件子集，但缺参数、未知谓词、不可恢复坐标或容量限制应报告“未完成”，不能说 schema 非法，也不能当成空分支通过。
4. 格式化应使用 raw 的 YAML 表示；不能把 effective 展开结果写回创作文件。当前 loader 的普通字典已失去 anchor、alias、merge 和排版信息，当前 effective_yaml 又主动把所有字符串加双引号，都不能直接充当原地格式器。
5. check_yaml.py 可以统一提供检查、预算、格式和绘图，但默认保留只读检查，格式写回和 PNG 各显式选择。独立的 pipeline_graph.py CLI 应退出日常使用，保留其有用的 Python 分析接口即可；不再留旧 CLI 转发壳或旧语法路径。
6. 四案例最后按 common 重新审定当前完整设计集合，再写所选 YAML 与 dispatch。FA4 prefill 现有 11 份已经超过硬上限；其余目录也不能因为未超过 10 就免于独特价值审查。不能先把 21 份全迁完再讨论是否需要。

推荐实施链是：目标/预算边界 → raw 解析与 from 来源图 → 全部结构和静态关系 → 有限图与预算/格式两条实现支线 → 统一入口、starter/README → 四案例。主要工作量在静态语义、有限图和案例；最终预算表仅阻塞预算规则的定稿。第 7 节给出逐步验收，其他各节是可按需回查的源码依据。

## 2. 目标差异与源码落点

| 范围 | 现行目标与必须处理的差异 | 主要代码落点 |
| --- | --- | --- |
| 公共身份、引用 | dispatch 只有 id?/when；非 root 不复制 Contract；三区独立 from 同任务同名整区；整个节点引用图无环，包括跨区的互引；在使用节点上下文绑定；完整集合含 root 至多 10 | design_resolution.py:151–262、rules/structure.py:201–231、rules/declarations.py:31–43、rules/__init__.py:27–37、schema.py:112–120、diagnostics.py:62–70 |
| Contract | 保留现有 nullable、精确数值、输出 tolerance 配对与属性引用检查；range/default 的字符串形式仍合法；硬件名称是开放完整字符串，不能自动改大小写 | rules/values.py、rules/hardware.py、rules/symbols.py；DOC1:58–67、197–235，DOC2:105–139 |
| 数学改写 | 删除 rewrite[].step 和区域级旧 ledger；每条 rewrite 自己有可选 baseline/gain?/cost? 账本；名字用于阅读/定位，不新增步骤 ID | rules/structure.py:54–59、rules/declarations.py:44–54；DOC3:35–69、133–157 |
| 工作调度 | phase→step、preassigned.step→stride；补 partition/数据块 meaning；删除 launch.protocol；保留 launch/order/handoff 的当前语义并补事件签名检查 | rules/structure.py:61–119、rules/declarations.py:63–109、rules/schedule.py；DOC4:54–77、243–324、388–412 |
| 流水线字段 | step/index/event/storage；index 是 meaning/definition；storage 新字段 meaning/depth/location/shape/dtype/total_size/member?/view?，view 只保留 slot_index/region | rules/structure.py:121–192、rules/declarations.py:110–126、rules/pipeline.py:16–25、107–122、rules/symbols.py:38–43 |
| 动作和 pipe | info.describe 必填；using_verb 的八个封闭动词；每个 noun 必填 execution_mode/describe；有序列表和 for/if；matmul 支持普通结果 noun 和 A@B 复合 noun，不能仅把 mma 字符串换名 | pipeline_syntax.py:12、75–117、166–198，rules/pipeline.py，pipeline/actions.py，pipeline/graph.py:41–120 |
| 等待和事件 | waiting 的三个分类都是非空字符串列表；显式 issue/done、Role[member]:: 引用；event 必须完整写 role、member、时点，形参不捕获 role 局部变量 | pipeline_syntax.py:193–198、rules/pipeline.py:57–106、pipeline/graph.py:122–251、275–301 |
| Realization | feedback 成为平级列表，字段 item/choice/evidence；不随 from 继承；不把旧 prefer.feedback 搬成兼容分支 | rules/structure.py:194–198；DOC6:21–64 |
| 日常工具 | 加入集中预算和纯格式化；原图命令能力汇入 check_yaml.py；统一诊断、退出码、检查与写入范围 | check_yaml.py、schema.py、yaml_source.py、pipeline_graph.py、pipeline/output.py、两份 README |

以上是替换当前定义与使用位置，不是为旧数据增加转换器。保留“旧字段不接受”的少量门禁即可，不需要逐个旧版本的行为回归。

### 引用解析先行

当前 resolver 把 parent 树当作展开顺序，并要求 from_parent 等于直接父；当前 origins、raw/effective 副本、source_manifest 和 effective_identity 已有可复用基础（design_resolution.py:20–67、94–114）。建议：

- 保留 ResolvedTask/RegionOrigin 和文件发现的基本职责，重写为实际 from 边的节点 DAG。检测跨区互引，不能只分别检查三个区域的环。
- 每区记录自己的 consumer→中间转发者→provider 链；来源闭包包括 root、自身、全部实际引用链，排除没有引用的“关系节点”。不从 dispatch.md 读取继承或 patch。
- 保留 raw 与各 effective 节点间不共享可变对象的性质。Contract 仍统一来自 root；元数据、when、version、realization 不从 provider 带入。
- 诊断同时给使用节点的字段上下文、provider 真实位置和 from 使用位置。更新 diagnostics.py 的硬编码 from_parent，不能引用展开视图的虚拟行号。
- resolve_directory 仍可作为有明确“未验证”状态的草稿阅读接口；完整 validate_* 先校验 raw，再校验展开后的当前组合。不要为显示草稿而放宽正式验证，也不要让显示草稿必须通过全部设计检查。
- 10 份上限使千级父树的专用压力测试失去任务价值。沿用简短的 DAG 遍历和通用容量保护即可，不继续扩展继承框架。

## 3. 静态检查与有限展开的分工

### 3.1 共享一套语法和声明模型

建议在 pipeline_syntax.py 中保留现有整数字面量、顶层分隔和成员路径解析的有用部分，重写 ScopedPipe；仅增加表达当前语言所需的 action、for、if 节点和词法作用域，不建设编译器框架。

新模型需要保留源字段路径、动作族签名、索引定义、当前绑定和控制条件。列表顺序是执行信息，mapping 顺序不是。所有分支中的动作都参与声明核对，即使样例里条件为假或循环为空：

- using_verb 只接受八个基础 verb，各动作唯一匹配已声明的 verb/noun；普通 noun 与复合 matmul noun 分别核对参数个数。
- 同一 role/verb/noun 只有一种固定参数个数；P 与 P(i) 不能重载。不同 role、不同 verb 的同名 noun 不自动指同一数学值。
- 内层 for 不可重绑仍在作用域内的外层索引；局部索引不泄漏到兄弟项。不能用“全文件禁止同名”代替词法作用域。
- flatten 恢复的坐标必须与已绑定外层坐标一致。当前 graph.py:70 使用 dict 合并覆盖，expressions.py:277 又会静默筛除不一致项；新实现必须在对应控制语境明确报告冲突。
- 动态 refill 下，worktile 域不可 count/last；查询限制不能因为给了有限 --worktiles 样例就失效。
- event 的引用只使用自身形参与已定义参数，不从某个 role 的当前循环环境捕获变量。
- 成员选择的表达式检查沿用 DOC0 的开放数学记法；只有已知数值关系可直接拒绝，不为静态检查调用有限求值器。

当前 free_action_indices 已在 AST 无法解析时交回审阅（pipeline_syntax.py:201–228），parse_members 也主动避免限定表达式语言（134–163），这两项边界应保留。if 条件也遵循同样原则：普通检查识别控制外壳、名称和能够确定的作用域问题，不把“有限 evaluator 不认识”当语法错误。

### 3.2 有限展开的最小正确模型

展开输入是通过静态检查的 effective plan 加显式样例参数。不得从旧 PNG、算子名、dispatch 的其他方案或源码正文补配置。

1. 按列表解释 for/if。兄弟 for 先完成第一段的发起序列，再展开第二段，不能合成交替循环；每个 if 在自身位置单独判断，没有隐式 else。
2. 空循环不删除循环外动作；no-op payload 不删除已经存在的动作。条件未知时，返回“该样例未能展开”，不默认真/假，也不把分支两侧合并成一个已验证图。
3. 动作仍保留 issue 与 done 两个事件。所有实际动作按 pipe 顺序发起；前一动作是 serial 时由其 done 约束下一动作 issue，其余只按 issue 顺序推进。
4. async_ordered 完成序列在同一 step、worker、成员、role 内跨 noun、verb、循环、worktile 接续；穿插 serial 不重置，async_unordered 不加入。serial 不排空先前异步动作。
5. 实例身份至少包含 step、代表 worker、worker 成员、role、verb/noun 及实参；位置/控制块不能用来让重复实例“变唯一”。当前仅每 step 展示代表 worker 时，应在报告中明确这一范围。
6. 将 waiting 全部分类、role 顺序和完成边一起做环检查。prepared/released/allowed 只是原因分类；相同目标的多条规则取前驱并集。
7. 删除 graph.py:253–273 的 default_edges；不因 noun 相同、view 同槽或 storage 同名补边。按 DOC5:522–529 审阅输入就绪与覆盖安全，工具不从 describe 推导未写事件。
8. Role:: 的省略成员永远是当前 worker 成员；不存在就报错。只有明确 [*] 才选全部实例。warp[0:4] 是该 worker 成员内四个协作 warp，不会生成四份 Role 事件。

现有 Fraction 整数运算、flatten、容量上限可以继续使用，但应补有限布尔值、比较及 and/or/not 的受限求值；不得 eval，也不得执行 describe 里的动作或查询。对 Full(t) 等参数化查询，可沿用明确样例取值的能力，具体采用参数名还是经实参替换后的查询键由小实现决定；必须记录实际使用的取值。不能把一次样例选择说成运行时状态演化的模拟。裸参数或查询取值不足时保持 ExpansionError。

### 3.3 waiting 必须按目标发生域处理

DOC5:664–673 是重写时最容易丢失的语义：

- 先检查两端 role、动作族、签名和明确成员是否合法。
- 目标因声明的 if 为假或循环无该项而不存在时，没有这一等待实例。
- 目标实际存在，且声明窗口内的前驱没有产生时，必须报悬空事件；不能用两端已生成事件集合求交删掉等待。
- 变量移位越过声明窗口才跳过该前驱；固定索引错误和窗口内缺失不能伪装成边界。一个前驱越界不应吞掉同条规则的其他有效前驱。
- 窗口依据控制声明和已定义数学域，不依据日志或生成事件的 min/max。n[t,i+1] 不进位到下一个工作，n+1 可以跨工作。

旧 IndexWindows/coordinates 已区分声明窗口和实际覆盖，并精确恢复仿射坐标，值得保留有实际价值的核心；但当前 explicit_edges 从两端变量域遍历，resolve_edge 对两端对称找实例（graph.py:170–251），不能原样套到带 if 的新语义上。

建议围绕“目标声明的发生条件、窗口内应存在的前驱、实际事件表”重新设计调用关系；保留有界坐标恢复作为辅助，不新增通用符号逆解器。目标匹配或窗口恢复无法唯一确定时报告未完成。测试重点采用 DOC5 的 Drain→ScaleTM、release→首 matmul 和跨工作双槽例子；不要为了保住旧测试而要求不存在的目标也产生等待。

手写固定动作列表与重复控制的旧测试须逐条回到当前规则审查，不能批量换字段后继续沿用预期。“固定引用指向不存在动作”与“目标因自己的控制条件不发生”不是同一类问题。

### 3.4 flow 静态检查是本轮必做项

完整跨 step 实例图不应成为静态规则更新的前置借口。以下项目无需解释任意数学正文，就能在现有 rules/schedule.py、declarations.py、pipeline.py 中对齐：

| 必做检查 | 检查边界 |
| --- | --- |
| work_schedule.step 与 tile_pipeline.step 一一对应、launch 每个 step 恰好提交一次、多个 step 时 order 必填、单 step/role 使用 MAIN | 明确结构与身份，不从 mapping 顺序推出调度 |
| 同一次 launch 的 kernel_done 是同一提交完成；禁止带索引的内建事件及本地同名 event；启动不能等待自身尚未发生的事件 | 内建事件不等同局部 step done |
| start_after 的 step、event 名、形参数量、完整局部事件引用、参与范围 | 现有 schedule.py:25–27 只比事件基名，需补签名；启动等待覆盖所有引用项 |
| 启动依赖环 | 当前仅向环图加入 kernel_done（schedule.py:37–38）；对能确定必须由来源提交产生的 launch_ready/本地 event，也检查其启动前提，不因删除 protocol 而漏掉 |
| 同 launch 非 concurrent step 的 worker/worker_count/必要 placement 一致；per_worktile 的坐标、排列与预分配一致 | 只拒绝确定冲突，不以字符串不相同推断数学不等价；per_worktile 不等于 per_worker |
| 同一 worker 的 storage 合账、同名重复登记、storage 引用目标与成员范围 | 不把不同 worker 同名存储认作共享；不从 shape 猜出物理副本 |
| ready 属于 producer，wait_at 属于列出的 consumer；别名展开后的时点、动作与签名合法 | 不能只检查别名存在；wait_at 必须指向实际等待完成或读取发起，具体动作是否承担该职责仍看契约 |
| 省略 ready/wait_at 的语义，以及 wait_at 只覆盖点名 consumer | 在内存解析结果或明确的辅助函数中算出覆盖关系，不写回默认字段；未点名 consumer 仍在 worktile 入口等默认就绪 |
| handoff 的数学值、producer/consumer、storage 和已知区域关系 | 查可明确解析的定义与确定冲突；正文定义的数学视图、变长区域覆盖、外部状态初值继续留给审阅 |

特别注意三个默认：ready 缺省为相关生产 worktile 完成并可见；wait_at 缺省为对应消费者 worktile 入口；局部别名只覆盖它点名的消费者，不会取消 order/start_after 或其他等待。等待动作可以提前 issue，只把就绪约束施加到其 done。工作集合为空时，循环外入口事件仍可能存在。这些都不能从“每个 step 图无环”推导出来。

完整跨 step 事件实例化还需要工作版本/区域映射、worker 配对及可能只写在数学/describe 中的关系。DOC4:319–325 明确允许一块消费区域覆盖多份生产 worktile，不能用 worktile 序号相等或 shape 相同代替匹配。**本轮推荐交付更新后的局部有限图、完整可做的 flow 静态检查与清楚的覆盖说明；不把任意 flow 展开成全局图作为必要能力。** 若显示 flow 总览，应直接标注 launch/order/handoff 与默认等待，不能把它作为已经验证的实例边加入局部 DAG 结论。

一个需整合代理复核的具体边界：旧 schedule.py:75–76 一律要求片上 handoff 的参与者处于同 launch 且非 concurrent。现行 DOC4:317–324、342 规定存储所属 worker 的副本匹配和跨 kernel 不保留局部身份，DOC5:537 规定非 concurrent 共享 worker，但没有一句直接重述“所有 concurrent 片上交接一概非法”。跨 kernel 的局部资源接续、确定的 worker 身份矛盾应拒绝；是否保留旧代码的一刀切条件，需按正文共同解释核定，不能因为旧工具这样写就当成已定规范，也不要借此擅自实现新的跨 worker 通信语言。

## 4. 文本预算与排版

### 4.1 预算单独集中，先核对覆盖，再填最终数值

当前 rules/values.py:22 只要求非空，尚无字符/行数检查。建议新增一份小的集中预算表和规则遍历，例如 rules/text.py；不要把历史数值散入 text()，也不要让格式器替作者缩写。

- 预算定位按 schema 的字段路径/角色，使用实际路径段区分固定字段、具名条目与列表。不能递归遇到键名 describe/name 就套规则；instance 本身也可能叫 shape、name 或 describe，硬件键还可包含点或斜杠。
- 单独列清正文类：meta_data 的 name/describe/dispatch.when；compute/rewrite 的 name/math/symbol.meaning；三类 thesis；hardware 的 device/arch、execution/feature 摘要、stack；schedule 的 describe/meaning/strategy/refill 说明；index/storage/role/noun 的说明；ledger/arithmetic；realization 的 choice/reason。
- 再列引用/表达式类：source、wiki、evidence、item、retention、range/default、shape/整数表达式、ref/bind/unknown 内容、事件/依赖字符串和开放 dtype。按用户最终表决定哪些计、计整体还是说明部分；不要默默套用 describe 的上限，也不要为了绕开限制改成映射或新字段。
- 检查 raw 的真实定义，诊断定位 raw；同一 from provider 的相同文字不必对每个消费者重复刷屏。effective 关联问题仍保留 consumer 上下文。alias 多处使用有不同字段预算时，分别验证使用语境，输出可合并成同一来源问题。
- 建议字符数按解码后的 Unicode 字符计，非字节/单词；行数按字符串内容行计，块标量的末尾换行不额外制造一行。换行是否计入字符数、空行与末尾换行如何计，要随最终表一起明确。物理 YAML 换行和文本内容行数不是同一概念。
- 报错包含文件、实际字段、字符/行实测与上限；只建议人工精简。不删除标点、折叠语句、换词、改缩写、删字段、改 version 或移动必要设计事实。
- YAML 结构正确但预算超限，可以完成纯格式整理，最终仍返回检查不通过。格式变化不使超长正文自动“合格”。

最终数值表尚未提供；不能把“草稿初版”的数值当默认，也不能把工具实现容量上限当文本预算。

### 4.2 对 CHANGELOG 追加草稿的逐项核对

新增输入位于 CHANGELOG.md:187–252，标题自称“prompt 的草稿初版”。应作为待整合意见保存；它不是覆盖 DOC0–6 的最终声明。

| 草稿主张 | 当前规范/实际解析事实 | 本轮建议 |
| --- | --- | --- |
| 全面禁止单/双引号，见 190–192 | DOC0:65 要求保持字符串类型和避免 YAML 歧义；DOC5:473 明确 waiting 要引号或合规块标量。事件别名列表里的花括号也必须是字符串 | 必要引号保留；无歧义普通字符串可自动用 plain。块标量虽能避开外层引号，但与所有 entry 单行冲突，不能假装两者可同时普遍满足 |
| 句尾去句号 | 这是改变字符串内容，超出纯排版；正文句点、文件名、外部标识不能靠通用正则区分 | 若最终要求，作为人工文本检查/提示，格式器不删除 |
| 全部具名 entry 单行、单项子条目并到字段所在行，见 204–207 | DOC0:65 允许 flow/block 两种并不强制换行；step/role、带 math 的 rewrite、带 view 的 storage 等有必要下级结构。矩阵/列表/表达式在 flow 中有额外引号要求 | 只对事先列明的短叶子记录压紧，容器和数学正文保留层级。按路径制定少量样式，不按“映射只有一项”一刀切 |
| meta_data/contract/design/realization 全大写、固定字段首字母大写、step/role 全大写、自定义名称全小写，见 212–214 | 固定字段本来区分拼写；DOC4:358 推荐多 step 小写、单 step MAIN；DOC5:598 只强制单 role MAIN；DOC0/1/3 要保留数学名与外部标识大小写 | 不自动改大小写。这会改 schema 语言及公式、引用身份，须先形成新的正式规范决定；本轮默认现行名字 |
| range/default 不得字符串，见 223 | DOC1:60–61、199–211 明确允许约束/默认表达式字符串；例如范围开端点不能普遍改成两元素闭区间 | 不能归入格式改写或预算；当前合法字符串继续接受 |
| tolerance 和 gap 按 output 对象限长，见 224 | tolerance 是输出名映射，gap 是一个字符串（DOC1:73–77） | 最终预算表必须按两个实际路径分别列，不新增 gap 输出映射 |
| ledger 可为空，见 234 | DOC3:66–69、231 要求出现时 baseline 非空、无意义 gain/cost 省略；DOC0 不允许用空值代替可选内容 | 不允许空内容或自动删账本；沿用现行结构，数字限制另定 |
| item 一律当自由文本限制，见 246/252 | DOC6:44、54、97–115 中 item 是一个引用或引用列表；列表语义不能被压成描述句 | 引用的预算口径单列；不按文本压缩改写引用 |
| “schema docs 的例子也应遵守”，见 216 | 本轮不改正式规范；若要新大小写/类型/强制行数，会影响 0–6 定义本身 | 不顺带改 docs。纯展示风格可在计划批准后的适当文档维护中统一，实质规则需另行确定 |

草稿中的 describe/name/meaning/math/thesis、dispatch.when、strategy/refill、unknown、arithmetic/ledger 和 realization 数字都暂不落为最终限制。不能只摘其中一些值先上线，遗漏的字符串路径也不能默认为无限制或继承最邻近的上限。

### 4.3 已核实的 YAML 风险与轻量实现路径

yaml_source.py:26–31 的 Document 只有 value、positions、values、source_sha256；convert 在 94–134 展开 alias/merge，并按每次用途构造普通值。它正确保证 Decimal、类型、显式重复键、循环 alias 和真实源位置，但不保存 scalar style、flow style、原始 anchor 名或注释。

只读内存试验的实际结果：

| 输入特征 | 当前结果 |
| --- | --- |
| waiting 字符串不加引号直接放入 flow 列表 | YAML 解析失败 |
| shape: [d: max(1,N)] | 得到 [{'d': 'max(1'}, 'N)']，已经改变列表结构；给表达式必要引号后才是单一具名维度 |
| 字符串 “1e3” 与数值 1e3；字符串 “false” 与布尔 false | 分别保持 str/Decimal、str/bool，不能去引号合并类型 |
| on、080、长小数 | 分别读为字符串、十进制整数 80、精确 Decimal；不得换回 PyYAML 默认 YAML 1.1 解释 |
| 同一 anchor 的两个 alias 用途 | 返回相等但不是同一个 Python 对象，字典本身已无法还原原来的 alias |
| 直接 safe_dump Decimal | RepresenterError，不能无设置 dump 普通字典 |

进一步试验在同一只读进程中捕获当前 Loader 的 raw node 图，使用同一组 implicit resolver 序列化，只调整节点样式。样例同时含 alias、merge、带逗号的 flow 表达式、数值形字符串、精确 Decimal、waiting 和多行 math。重新经 yaml_source.loads 读取后，递归检查类型和值均相同；alias/merge 仍保留，必要引号由 emitter 保留。PyYAML 把 anchor 名规范成 id001，未保留人工 anchor 的拼写；当前 common 赋予 anchor 的是内容复用语义，这不改变设计身份。

这个试验支持以下小改法，但不能宣称已验证所有 YAML：

1. 让 yaml_source 保留原始文本与 raw node 图/所需源范围，并把现有 core scalar resolver 提成共用定义；校验用的 value 和 positions 继续保留。无需先引入第二套 YAML 依赖。
2. 新增一个 format_yaml.py，按少量明确路径选择 flow/block、固定字段推荐顺序和必要 scalar style；共享 raw node 的 alias/merge 不展开，所有列表及 merge 列表保序，from 原样保留。
3. 只调整排版，不改字符串解码值、键名、表达式、依赖、引用、数字类型或方案。多行字符串和块标量 chomping 要保留精确值；从 block 改 flow 时重新处理表达式中的逗号、括号等 YAML 歧义。
4. 生成文本后用同一个 loader 回读，逐层验证 type/value 与列表顺序一致，再验证再次格式化不再变化。不能只用 Python 的宽松相等判断，因为 True、1、Decimal(1) 可能比较相等。
5. 含注释的输入须先明确处理：DOC0 禁止实例注释，而 raw node 默认不保留注释。不能把可能承载设计事实的注释静默删掉；初版检测后报告，保持原文件，交作者处理。该限制不是 schema 通过的替代判断。
6. 写回只针对明确选定 raw 文件；先在内存构造并验证，再原子替换，未变化文件不更新 mtime，保留文件权限。当前工具接受文件 symlink（test_yaml.py:179–189），格式写回不能把 symlink 静默替成普通文件或越界改目标；初版保持只读并报告该写入动作无法完成即可。
7. 写后从实际字节重新建立源位置/hash。纯排版不增 version，但 raw SHA256 和依赖它的 effective_identity 会变化，已有图的旧指纹不能沿用。

不建议为这件事把 record()/named()/sequence() 的闭包体系整体改成可反射元 schema；结构规则继续负责合法性，格式和预算用清楚的路径表，避免一项排版需求牵出新的框架。

## 5. check_yaml.py 的推荐边界

### 5.1 一个入口，少量正交选项

下表是建议的 CLI 设计，不是已经存在的命令：

| 调用意图 | 建议形式 | 默认范围与写入 |
| --- | --- | --- |
| 日常检查 | check_yaml.py FILE | 检查同目录全部当前 YAML 的结构、引用、关联和已配置文本预算；只读 |
| 检查统一排版 | check_yaml.py FILE --format check | 只读报告格式差异，格式差异与 schema 错误分开 |
| 一键排版并检查 | check_yaml.py FILE --format write | 只写所选 FILE 的排版，再给最终检查结果；加 --all 才处理同目录全部 raw |
| 检查有限样例 | check_yaml.py FILE --graph check --set NAME=INT --worktiles N | 不需要 Graphviz，不写 YAML/PNG；加 --all 才展开所有节点 |
| 生成微流水线图 | check_yaml.py FILE --graph png --set NAME=INT --worktiles N | 通过前置检查后，为所选节点写同名 PNG；加 --all 处理全目录节点 |
| 常用合并动作 | check_yaml.py FILE --format write --graph png … | 先格式化并重读/检查，再用同一份最终来源快照展开和绘图 |
| 查看草稿 effective | 保留 --show-effective | 只读 JSON envelope，明确 validation:not_run；与格式/图动作互斥 |
| 检查独立阅读视图 | 保留 --standalone | 只读指定文件，必须自带 Contract 和本地 Design；不要求旁边有创作目录/dispatch |

--skip-hardware 的当前显式含义保留：只排除整个硬件区及依赖硬件的检查，不丢弃 returned/raw 文件里的内容，不关闭动作、索引或其他关系。--set/--worktiles 只用于显式 graph 模式；不用历史图片补参数，不通过算子名选择样例，也不写入 YAML。--all 不改变基础检查一贯的全目录范围，只扩大明确请求的格式/绘图对象。

现有 --worktiles 默认 1 可以保留为清楚标注的代表性样例默认，而不是工作调度推断；若保留该默认，输出必须列出它。并不需要新增 YAML 配置、任务工作流文件、图场景注册表或自动迁移命令。

### 5.2 返回结果与内部边界

- 保留 0/1/2：0 为所请求检查完成且通过；1 为结构、关联、预算、所选格式检查或已展开依赖反例；2 为参数、环境、无法求值/展开、写入或容量导致未完成。图的 1 明确标为“该有限样例发现问题”，不要把所有 sample 错误统一叫 schema 语法错误。
- 摘要区分“schema/文本通过”“格式是否变化”“哪些 step 有限展开”“flow 已做何种静态核对”“图是否生成”。未知谓词或缺值不得打印 Acyclic sample。若多节点有的失败、有的未完成，应分别列出，不能只凭总退出码抹掉已发现的错误。
- 所有结构/预算/格式/图诊断尽量共用 Diagnostic、真实字段路径和 provider 定位；只增确有用途的种类/严重度信息，不新建一套问题/状态协议。
- schema.validate_file、validate_directory、validate_resolved_file 和 validate_standalone_file 保持只读 Python API，不导入 Graphviz、不执行排版。普通检查不因为图能力加入就开始要求所有参数可求值。
- check_yaml.py 只做参数与过程编排；让 schema.py 复用一次 raw 文档快照的校验/展开函数，formatter、graph 分别消费需要的 raw/effective 视图。规则模块不读写文件、不打印、不决定退出。
- --show-effective 是明确的阅读/定位帮助；正式单文件交付仍先检查完整创作任务，再验证生成视图。不要为统一入口把草稿预览悄悄改成验收。

### 5.3 PNG 管理与显示

保留 render.py 的 Graphviz 渲染、成员折叠安全检查、serial 必要时拆成 issue/done 的显示策略；重写旧 phase/边类型/图例文字。当前成员折叠和同步动作折叠已有语义反例测试（test_pipeline_graph.py:403–443），比像素快照更有价值。图必须保留“无时间尺度、显式样例、局部检查范围”，不暗示硬件重叠或全局存储安全。

保留 output.py 的所有权判断、PNG 内 manifest/origins/effective_identity、发布前后的来源核对和原子替换（85–120）。有一项建议供整合取舍：当前 begin 在结构检查前就删除旧 PNG，并会在 --all 时清退孤立图片（68–83）。统一入口后更推荐先做冲突预检，完成新图才原子替换；失败明确说旧图未更新、仅是历史图，不让普通检查/参数错误触发图片删除。所选案例的删除 PNG 在最后整理阶段明示处理，不把“检查”兼作目录清扫。若保留现有失效策略，也至少把它限定到显式 graph png 动作，format/check 不能触发它。

无需新增旁置场景 JSON 或持久图数据库。若因条件/参数无法完整展开一个所选样例，返回未完成并保留原因；不发布看起来成功的部分图。依赖环仍可以生成清楚标红的诊断图并返回 1。

## 6. 模块保留、删减和重写建议

| 模块 | 建议 |
| --- | --- |
| check_yaml.py、schema.py | 保留入口和只读 API；合并参数编排，减少重复读取；不把业务规则堆进 CLI |
| design_resolution.py | 保留结果/来源数据结构，重写 from DAG；删 parent 树和直接父限制，不加迁移或版本路径 |
| yaml_source.py、diagnostics.py、errors.py | 保留核心安全/精确数值/位置/容量边界；补 raw 表示、共享 resolver、预算/格式诊断的必要信息 |
| pipeline_syntax.py | 保留顶层分隔、成员记法、整数字面量；重写控制树、动作/显式事件/依赖解析 |
| rules/structure.py、declarations.py、schedule.py、pipeline.py | 原地按现行 DOC0–6 重写相应部分；删除旧 ledger、protocol、旧 verb/模式/pipe/等待定义 |
| rules/values.py、symbols.py、hardware.py、common.py | 大部分保留；更新路径/当前模式、删 view.dtype 及旧字段分支；开放表达式和硬件名边界不变 |
| pipeline/graph.py | 主要重写；删默认 noun 数据边、成员自动 fan-out、旧控制分区与事件默认时点 |
| pipeline/expressions.py、windows.py、coordinates.py | 保留有界精确计算/声明窗口/必要坐标恢复；适配新控制语义，不发展通用代数求解或 runtime 模拟 |
| pipeline/actions.py、model.py | 保留小模块职责，统一 parser；PhaseGraph/milestones 更名为 step/event 的当前对象，不保留别名类型 |
| pipeline/render.py、output.py | 保留正确显示与来源追溯，改图例/路径并明确失效策略 |
| pipeline_graph.py | 去掉独立 CLI 和 sys.path 入口修补；可保留 configuration/analyze 这个小 Python facade，避免为了“精简”把所有编排塞进 graph.py |
| check_policy.py | 当前仓库没有 check_policy.json，README 明说兼容读取且未启用，Extract 显式 --skip-hardware。倾向删除目录兼容读取及专用测试，只留显式参数；整合时同步维护说明，不改变本轮之外的 Extract 运行状态 |
| 新增模块 | 至多先增加预算规则表/检查模块和纯格式器两个明确用途模块；不新增通用 pipeline 引擎、插件注册、版本门禁或第三方 YAML 依赖 |

图的动作规模/域深度/边数保护属于真实处理容量，应保留。不要把这些内部数字与用户尚未确定的正文预算合为一个配置系统。

## 7. 实施顺序、相对工作量与最小验收

相对量级用于安排先后，不是工期承诺；正文/案例研究的不确定性比字段更名大。

| 步骤 | 工作与依赖 | 相对工作量 | 最小验收 |
| --- | --- | --- | --- |
| A. 冻结目标清单 | 以 DOC0–6 建立字段/语义差异表、预算路径覆盖与草稿冲突清单；确定只采用当前规范的边界 | 小 | 待定数字未进入代码；没有把 while、旧父链或草稿大小写列为目标；明确图/静态范围 |
| B. raw 解析和引用 | 补 source 表示与共享 resolver；from 节点 DAG、来源闭包、诊断、standalone 去 parent | 中 | 三个不同 provider、多跳、跨区环、使用节点上下文错误；精确数值/alias 不变；来源漂移改变 identity；数量 10/11 边界 |
| C. 全部结构和静态关系 | 按第 2、3 节更新 DOC0–6；建立新最小 fixture；更新 starter 的现行结构；在此核对 concurrent 片上 handoff 的文档边界 | 大 | 合法最小当前 plan 通过；旧字段直接拒绝；合法非 Python 表达式通过；flow 可直接检查项、noun 签名/MAIN/成员范围有正反例 |
| D. 有限图重写 | 依赖 B/C 的有效视图、共享语法与声明；for/if、noun 模式、显式 waiting/event、声明窗口 | 大 | 两段兄弟循环保序；条件假不产生等待，目标存在缺前驱失败；跨 noun/verb ordered、serial 不 drain；无隐式同名边；当前成员/[*]、空工作入口、移位窗口和真实环 |
| E. 预算和格式 | 依赖 B/C 与用户最终预算表；可在 D 期间独立实施 | 中到大 | 实际值/上限诊断；文本行与物理行区分；必要引号/Decimal/list/merge/from 全保真；格式幂等；超限不自动改文本 |
| F. 统一入口、输出与文档 | 整合 C/D/E；只留 check_yaml 日常命令；更新两份 README 的当前能力和边界 | 中 | 默认/graph check 全只读；format write/graph png 明确范围；0/1/2、provider 位置、Graphviz 可选、来源变化停止发布；从任意节点/外部 cwd 正常调用 |
| G. 四案例与 dispatch | 最后执行，先按 common 审定完整集合，再整理选中设计、引用、少量目录记录与图 | 大，需语义研究 | 每份具体已选方案、从输入到输出闭合；独特完整价值、总数、目录链接与引用准确；无全排列；有限图样例命令可复现、已知缺口保留 |
| H. 收尾验收 | 针对最终改动运行必要维护检查，补齐入口/案例状态说明 | 小到中 | 现行 fixture 和选定案例均通过对应检查；没有旧语法双轨、临时迁移器、意外 YAML/PNG；下游接口交接清楚，未声称 Extract 已更新 |

starter 仍是故意未填写的骨架，不应用猜测值把它做成可通过的假设计；保留 version:1 起始值，补现行 MAIN、step、meaning、using_verb/noun、列表 pipe 的形状，删除旧字段。copy starter 后不填内容仍应失败。可选结构不为“模板完整”全部塞入，真实合法的最小实例由测试 fixture 提供。

建议只保留下列具有设计意义的验证组；现有测试大部分直接绑定旧字段，需要重写/删减，不值得维护完整旧测试矩阵：

- 一个新最小任务与一个多来源任务，覆盖共同 Contract、from 的正确与错误、来源定位、无共享可变对象。
- 一个声明/作用域组，覆盖空域/假分支也检查动作族、普通和复合 matmul、内层重绑/兄弟局部域、event 形参和 member。
- 一个有限事件组，覆盖混合三模式、显式等待四时点、两个 if 独立、目标发生域、固定缺失与移位边界、跨工作 flatten。
- 一个 flow 静态组，覆盖提交归属、启动前提、ready/wait_at 默认与部分覆盖、同 worker 资源、空工作入口；不写伪造“任意跨 step 正确”的测试。
- 一个格式/预算组，使用真正会改变 YAML 类型或数据的例子验证往返、别名/merge、Decimal、公式/依赖字符串和保序，再检验幂等与预算报错。不要用“输出符合自己的实现”作唯一验收。
- 少量 CLI/图输出组验证只读、退出分类、来源漂移、图折叠不掩盖环；保留合成临时目录，不在测试里重画正式案例图片。
- 最后对四案例所选集合做一次通路验收和代表性样例图，不为旧节点全组合建立回归矩阵。

## 8. 四案例最后整理的判据

现有索引给出的规模是 FA4 prefill 11、decode 7、QSA 1、GDN 2，共 21；dispatch 仍是旧 parent 树，并明确把一些 split/cluster 参数变化留在 YAML（各 dispatch 末尾）。这些都不是新方法下可以直接沿用的交付集合。

建议先为每个任务列出“保留的完整端到端方案／直接依附现有基线的参数、局部与组合记录／确属独立但未选中完整展开的候选”，随后才编辑：

- FA4 prefill：先核对 static、persistent 的跨工作寿命与领取组织、split-KV 的部分结果/合并这些主线是否有独特整体价值；joint/dual load、LPT 与 alpha 共享等必须逐项检查能否局部闭合，不按主线配对复制节点。索引不足以在本轮宣布最终保留数量。
- FA4 decode：检查全局归并、cluster 归并、direct 的输出形成和数值路径；q1/q4 变化不能只按 token 数认定为参数，也不能机械保留 q×合并方式的全排列，需看 softmax 分工、接口与完成协议。
- QSA：只有 root 仍须改成 common 的范围/完整设计入口。ticket、延迟发布、空 worker drain 及已知事实缺口不因“模板精简”丢失。
- GDN：依据本批 L20D 设计核对单/多序列的主协作、描述符更新和标量预取是否值得两份完整表达；不能移植另一批 B200 多阶段设计，也不能删除实际执行却未被消费的物化/转换开销。

工具只检查数量、节点/链接/固定结构等可确定事实；独特价值、局部变化边界、when 范围和组合相容性由审阅决定。若把 dispatch 的基本结构与 YAML 链接纳入 check_yaml，使用 common 的固定最小章节/完整设计表做轻量核对即可，不引入通用 Markdown 解释器或要求所有语义都可机读。目录不能补齐 YAML 缺失定义，记录也不会自动应用到基线。

所有 YAML 的内容更新由对照规范和来源的编辑完成；格式器不能把旧动作硬改成八词、推导等待或自动选择参数。已知缺口按真实状态保留，不能为了图无环编造 release、可见性或隔离保证。删除不保留的 YAML 时同步处理其 PNG、目录链接和 from 引用；不另藏“历史完整设计”来绕过上限。

## 9. Extract 的必要接口交接

这里只交接调用面，不更新 Extract 的阶段、计划、权限、冻结状态或发布流程。

| 现有消费点 | 本轮应交接的稳定能力/变化 |
| --- | --- |
| artifacts.py:28–62 从指定规则目录动态加载 yaml_source/schema | 保留 load/loads、Document 的 value/positions/source_sha256、SchemaError/ToolError；新增 raw 表示不改变旧字段含义，不依赖主进程全局 tools 包 |
| artifacts.py:68–87 读取实际行号、origins、source_manifest、effective_identity | from 图的实际来源闭包与 provider/chain；真实 raw 位置；删除 parent 含义，不让下游自己重造 resolver |
| artifacts.py:528–539 调用 check_yaml.py --skip-hardware FILE | 默认入口继续只读、保留 0/1/2 和显式硬件跳过；新增预算可能产生新诊断，不能默认格式写回或自动绘图 |
| standalone.py:58–91 使用 validate_resolved_file/effective_document/validate_standalone_file | 保留精确值、完整 Contract 和当前节点本地元数据/realization；严格视图不残留 from；来源/正文分开保存 |
| snapshots.py:25–32 收集 Template 的 Python 文件、starter、README | 新预算/格式模块应留在现有 tools 层次中，正常可被收集；本轮不修改冻结器或兼容加载流程 |

阅读视图/图的 raw hash 因格式变化而变，版本号可不变；下游须按已有 manifest 与 effective_identity 识别来源，不能仅看 node 字节或 version。待 Extract 更新时按当前接口接入，不能以增加 Template 旧语法兼容来维持旧 Runner 的假通过。

## 10. 真正缺少的输入与整合判断

**需要用户补齐或确认的输入：**

1. 最终字符/行数表及覆盖路径、计数口径；尤其区分自由正文、引用/表达式、schema 的 name 属性与实例键名。没有它只能完成机制，不能宣称预算规则已完成。
2. CHANGELOG 草稿中与现行规范直接冲突的主张是否撤回/修改：全面禁引号、全局大小写、所有 entry 单行、range/default 禁字符串、ledger 可空。推荐继续采用现行字段和类型、保留必要引号，只对短叶子记录统一排版；若要实质更名或类型变化，应先另行确定正式规范，不能夹在格式化中实施。

**应由整合代理给出明确推荐、无需额外向用户索取普通实现细节的事项：**

- 局部有限图加 flow 静态检查是本轮必要交付，是否另外显示 flow 摘要；不应承诺任意跨 step 实例图。
- CLI 的 --format/--graph 枚举形式与默认只读、--all 的写入范围。
- 新图失败时保留带旧指纹的历史图还是删除工具所有的旧图；两者都要清楚，普通检查保持只读。
- 删除未启用的目录硬件豁免兼容读取，还是暂留该小模块；不能扩展新的策略体系。

案例最终保留哪几份应在最后的独立语义整理中依据已有材料决定；这是待完成工作，不是现在要求用户逐个挑选 21 个旧节点的理由。其他流程和代码实施可按上面的依赖推进，预算缺表仅阻塞最终预算规则与相关完整验收。

## 主要证据位置

- 规范：DOC0:59–77、132–215、221–225；DOC3:35–69、139–157；DOC4:260–324、388–412；DOC5:148–183、222–344、348–450、452–673；DOC6:21–64。
- 当前入口/解析：check_yaml.py:12–58；schema.py:14–41、85–127；yaml_source.py:26–31、53–146；design_resolution.py:151–303。
- 当前图语义：pipeline_syntax.py:75–117、166–198；pipeline/graph.py:64–150、170–301；pipeline/expressions.py:151–228、260–306；pipeline/windows.py:58–77、164–201；pipeline/render.py:31–65、91–103；pipeline/output.py:68–120。
- 回归依据：test_yaml.py:13–28、63–91、167–225；test_pipeline_graph.py:41–94、307–443；test_pipeline_domains.py:102–194；test_design_resolution.py:126–249；test_cli.py:19–95。
- 追加草稿：schema_docs/CHANGELOG.md:187–252。它保持原文件内容，本报告未修改。

