# 示例、排版与命名审阅

本轮宜落实文字职责和示例内容；不把字数、行数、大小写迁移或检查器要求写进正式规则。现有 YAML 语法足以表达简洁正文，无须另造记法。以下判断只针对 `schema_docs/0–6` 和本轮变更记录。

## 1. 去引号必须保持 YAML 类型与事件语义

对现有 47 个 fenced YAML 做只读解析，并尝试删除双引号。结果不是所有引号都可直接删：

| 当前真实例子 | 直接删引号的结果 | 最小处理 |
| --- | --- | --- |
| DOC3 `retention: "carry_forward (b,h); c=-1..Nc-1"`、`"keep_value (b,h)"`，位于流式记录 | **静默改变结构**：逗号把 `h)...` 拆成额外映射键 | 保留必要引号；若完全不用引号，展开条目后直接写 retention 字符串 |
| DOC3 ledger 的 `"State chain: T -> Nc ..."`、`"Saved snapshots: 0 -> ..."` | 冒号加空格触发映射解析错误 | 改成自然名词短语，例如 `State transitions per sequence T -> Nc`，无须保留该冒号 |
| DOC5 waiting 的 `"{issue B} ← [{done A}, {done C}]"` | 开头花括号被识别为 YAML 映射，解析失败 | 保留引号，或用 `- >-` 后接同一事件字符串 |
| DOC5 event 的 `["{done MAIN[0]::store:X}"]` | 去引号后，事件不再是字符串，解析失败 | 保留引号，或展开列表并使用块标量 |
| DOC5 `"S(i,d)"` 与 `"warp[0:4]"` 等 | 当前块式位置可去引号；若再折成流式集合，逗号和方括号会成为 YAML 分隔符 | 排版器必须结合所在位置判断，不能只按字符串内容统一删引号 |
| DOC1 `ref:S0.shape`、DOC3 `bind:design.work_schedule::Bk`、DOC2 wiki 引用 | 当前示例均能直接使用 plain scalar，类型不变 | 去除不必要引号；DOC2 删除“带引号”这一显式要求 |

布尔与 null 形式是类型边界，但当前 fenced 示例没有把 `true / false / null` 当作带引号的文字值。现有 `nullable: true` 应继续是布尔值；`default: "None"` 中的 `None` 在 PyYAML 下本就解析为字符串，不是 null。不能为了演示风险新增无关字段或改变这些语义。未来工具仍要处理 `true / false / null`、纯数字和日期等会自动变型的字符串；PyYAML 还会把 `on / off / yes / no` 解析成布尔值。

只需对齐一句通用原则，不需新语言：

- 若允许必要引号：普通字符串直接书写，仅在保持 YAML 语法和字符串类型所需时使用引号或块标量
- 若绝对不用引号：普通字符串直接书写，无法保持字符串类型或语法时使用块标量，并允许相关条目展开

后一种与“任何单元素集合都同行”不能同时无条件成立。本轮按主代理已告知用户的默认方案处理：普通字符串去引号，保留 YAML 必需引号，不为完全去引号大面积展开事件字符串；这不是用户已经确认未来格式规范的声明。

## 2. 大小写不是当前可直接运行的格式化规则

有三种不同的变化，不能混成一次文本大小写替换：

1. **正式字段改名**：当前是 `meta_data / contract / design / realization`，固定路径还包含 `algorithm_requirement / using_verb / single_tile` 等。只说首字母大写，不能唯一决定多词写成 `Math_reshape`、`Math_Reshape` 还是 `MathReshape`，也不能据用户自然语言中的 metadata、constants 推导删除下划线或改复数
2. **实例身份及引用迁移**：phase/role 的大小写要同步 `flow`、事件、`waiting`、realization 引用；noun 和 storage 要同步 pipe、view 与描述绑定。改 YAML 键却不改字符串中的目标会破坏引用
3. **数学及外部名称**：当前 DOC3 明确符号区分大小写并沿用 Contract 原名，DOC5 明确数学值保留上游大小写。`Q / K / S / Bk / Nc` 是公式和引用的身份；`CTA / SM / L1/SMEM` 与真实路径、指令名称还有外部命名规则。全变小写不是排版，且需先检查大小写折叠冲突

本轮保持正式字段名与已有数学身份。示例可用现有规则允许的清楚名称，但不先生成一套尚未唯一确定的新拼写。单 phase、单 role 的 `MAIN` 继续遵守现行规则。

## 3. 一行条目的边界

当前通用规则已经允许紧凑条目并要求较大结构保持层次。无需本轮另写硬限制：

- 适合一行：constant、input/output、symbol、partition、worktile/target 值、launch、handoff、index、无 view 的 storage，以及内容短的 noun 属性记录
- 保持层次：整个 phase、role、using_verb 和各 verb 下的 noun 集合；有 view 的 storage；pipe 的 for/if；需要块标量的文本
- 单个孩子不改变容器性质。只有一个 role 不能据此把整个 role 连同 info、using_verb、pipe 压进一行；只有一个 waiting/event 字符串也不能牺牲它的字符串类型

正文中的逗号在块式标量里安全，在流式记录里可能拆项。排版不能逼作者换索引语法、创造缩写或删除关键限定。

## 4. 最小示例改动清单

| 文件 | 本轮值得改的内容 |
| --- | --- |
| DOC0 | 若用户确认，补一句直接写字符串的原则；不新增大小写、字符数或统一换行规则 |
| DOC1 | 数学示例去句尾句点并保留范围、空集和 None 分支；gap 删掉已经写在 compute 的解释。tolerance 来源例保留三阈值、公式缺失、相对基准缺失和不采用默认判据，不能为未来限字删事实 |
| DOC2 | wiki 引用不再要求引号，示例移除无必要包裹；execution/feature 保留设备机制与边界，去句尾句点 |
| DOC3 | worktile meaning 改为对象短语，如 `Reduction block across all features`；strategy 保留 `Bk = 64`，删 iteration 已表达的普通逐块读取复述；ledger 用自然名词短语去除冒号空格，去句尾句点。现有两处 150 字符限制应移出正式 docs |
| DOC4 | partition meaning 可写 `Independent chunk`、`Value-column block`；worktile U 可写 `Chunk of U for this worktile's value columns`。refill 保留初始化时机、原子领取、返回值边界及停止条件，删逐 API 步骤口吻；不改 flow/handoff 默认或调用同步语义 |
| DOC5 | noun 示例可紧凑，但必须说明源对象/来源与本动作承诺，不能只写 `Form Y from the inputs defined upstream`。保留多槽、view、提前释放及条件等待示例中的关键事实；去无必要引号和句尾句点。保持 for/if、storage、released，不恢复旧词 |
| DOC6 | 示例 choice/reason 去句尾句点和不必要引号；优先理由写具体成本，require 保留 TMEM 路径与操作数读取/覆盖顺序这类必要兼容条件，不把它改成泛泛的“提高性能” |

不需要新增测试、变更旧实例或要求所有局部示例成为完整 plan。各示例继续明确自己的片段范围；完成后解析全部 YAML fence，并人工核对变动字符串是否保留关键设计事实即可。

初始验证使用 PyYAML 6.0.3；去引号实验发现 1 处静默结构改变与 9 处解析失败，其余变动示例数据等价。

## 5. 已实施及验证

按主代理分工，只修改 DOC4–6：分开说明各文字字段职责，精简 meaning 与动作示例，消除 noun 输入必须在 describe 中重复列写的冲突，并保留全部输入的完整性与读取安全核对。移除 storage 必须单行的排版条款；普通字符串去引号，waiting/event 保留必要引号。

31 个 YAML 示例全部解析通过。与修改前逐树比较，键、层次、列表长度、叶值类型、数值、引用、pipe 控制和动作、waiting/event 字符串全部保持，仅 30 个说明文本叶值变化；`git diff --check` 通过。初始化时机、原子领取规则、工作范围、存储绑定、成员释放条件及必要实现兼容保证均已人工复核。未修改工具、旧实例、模板、归档或 CHANGELOG。
