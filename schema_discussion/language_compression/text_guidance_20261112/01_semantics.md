# 文本职责审阅

本轮依据当前 `schema_docs/0–6`，只审阅字符串职责与示例。已读根 `AGENTS.md`、语言精简建议和本日已完成复核，不重复调查旧实例。行号指修改前文件。

结论：已有规则基本齐全，值得改的是公共短提示、三个 thesis 的职责、个别重复义务及示例。字段、类型、引用身份、默认语义和计算保证不变。字符数、行数、统一排版、大小写和诊断等级属于后续工具；DOC3 现有两处 150 字符限制应移出正式 docs。

## 字符串覆盖盘点

| 区域 | 用户清单已覆盖的说明文本 | 尚需区分的字符串 |
| --- | --- | --- |
| DOC0 | `meta_data.name / describe`、`dispatch.when` | `source` 是版本化引用；`dispatch.id / parent` 是身份。`when` 是选择谓词，可定义必要的派生量，不是设计摘要。 |
| DOC1 | 常量、输入、输出、compute symbol 的 `meaning`；`compute.name / math`；逐输出 `tolerance`；`gap`；`range / default` | `constant.value` 的固定表达式、shape 维度表达式、dtype、属性引用、`golden_reference.source`。`gap` 当前是整个参考入口的一条说明，**不是逐输出映射**。条目紧凑不意味着每个属性都是自由文字。 |
| DOC2 | `device / arch`、全部 stack 叶项、execution/feature 的值、wiki 叶项 | device、arch、stack 是准确身份与版本；wiki 是准确定位符。它们不应按普通叙述改写或截短。execution/feature 的键是官方或明确名称。 |
| DOC3 | `thesis`、`rewrite.name / math`、symbol.meaning、approx/accumulate、ledger 三项 | `retention` 是方式、实例及迭代范围；`binding` 和属性 `ref:/bind:` 是引用；`unknown:<reason>` 含一条尚缺的事实；`value` 可含已知字符串字面量。应保留既有语法，不把它们作为摘要统一压缩。 |
| DOC4 | `thesis`、phase.describe、partition/数据块.meaning，以及 strategy | strategy 实际有 **decomposition、mapping、schedule 三处**。清单遗漏 `refill.<mechanism>` 的规则文本。`exit` 是条件引用或引用列表，规则必要时由 refill/strategy 定义。其余为上游值、维度、事件、storage 引用或参数表达式。 |
| DOC5 | `thesis`、role.info.describe、noun.describe、index/storage.meaning | index.definition、member、slot_index、region 端点、pipe 动作/for/if、waiting、event 都有专门语法；location/dtype 是已定义名称。noun/事件/view 键可以带复合操作数或形参，不能按一般短句处理。 |
| DOC6 | Planner `item / choice / reason` | `item` 已是引用或引用列表，不能改写成解释段；`feedback.choice` 同样适用简洁选择表达，清单未单独提到；`feedback.item / evidence` 分别是对象引用与版本化记录引用。 |

全区共有的具名键、shape 轴名、封闭枚举及引用路径不是自由叙述。整数表达式可能出现于 shape、partition.size/tile/range、worker/placement 数量、member 区间、worktile_order.group、preassigned、offset、index 端点、storage.depth/total_size、slot_index/region。这些位置需要数学完整性，不能用“短描述”代替。

## 各字段最小职责

| 字段 | 应写什么 | 不应复述什么 |
| --- | --- | --- |
| name | 任务通称或本步数学决定的短名称 | 实现分支、优化介绍、整句算法 |
| meaning | 对象身份、索引含义、本地数学片段或存储用途 | 已有 shape/dtype、公式、登记它的字段类型 |
| meta/phase/role describe | 任务、工作或角色的职责 | 下层动作清单、整段流程 |
| noun.describe | 动作与上游对象的对应，及尚不明确的来源、版本、区域、存储绑定或保证 | 已定义的公式、成员切分、方向和通用 verb 保证 |
| thesis | 本区主要设计依据与取舍 | 下层配置、字段清单、泛化性能口号 |
| strategy | 该层结构未给出的必要分解、映射或分配规则 | 结构已经表达的顺序、计数和方向 |
| math | 核心定义、范围、递推、分支、初始化、边界输出和精度转换 | 代码过程、正确性证明、优化说明 |
| realization | 对象引用、具体选择及其必要兼容条件或具体成本依据 | 上游完整约束、源码采用事实、代码组织偏好 |

“一句读懂”是写作目标，不能把不同版本、初始化或完成范围省成同一含糊句。有关外部等待、组内分工和跨调用状态，沿用已完成复核：去过程，保留实际启动/消费位置、观察值、局部先后、必要读退休及初态。

## 最小修改位置与建议

### DOC0

- `0.common.md:58` 后集中一小段：`说明文字直接写对象、作用或设计依据，尽量一句读懂。使用常见术语和明确名称，不自造缩写；已有字段、公式或引用表达的内容不再复述。设计成立所需的映射、版本、初态和保证写在对应字段中。`
- `:32–33` 的 name/describe 已简洁准确，无需增加逐字段限额。`:144` 已规定 when 不抄 root 约束，保持职责，仅去示例句末点。
- 不把通用提示解释成要求 name/meaning 携带完整契约，也不迁移字段大小写。

### DOC1

- `1.contract.algorithm_requirement.md:62`：`用短名称标识这一步的数学关系`；`:63`：`用公式给出定义、范围、归约、分支、边界和必要转换`。
- `:152` 直接收窄原句：`math 用核心公式、递推和分情形定义计算；文字只补公式所需的记号、范围或分支含义，不写证明、源码过程或优化理由`。后续 `:179` 的范围、初值、边界义务完整保留。
- `:199–203`：range 只填有效域约束，default 只填实际缺省值或已定义的表达式。保留 `[a,b]`、`0 <= route_id < E`、`finite or -inf`、`default: 0`、`default: "None"` 的原有解释，排除附带说明性散文；不能机械禁止 YAML 字符串。
- `:185–190 / :229–230` 的数学示例去句末点，不改公式与分支。`:255–258` 来源验收记录可缩成 `Source record only; atol=0.01, rtol=0.05, max_mismatch_fraction=0; formula and relative-error basis unspecified; default criterion not adopted`，保留阈值、缺口及不采用默认判据三类事实。此例无法在强行 100 字符时仍保留同等清楚的信息，本轮不迁就未来限额。
- `gap` 例子 `:86` 可改为 `Disable power-of-two scale rounding in the reference to preserve the unrounded scales in compute`。

### DOC2

- `2.contract.hardware_specification.md:55–60` 已给每种文本明确职责，不需叠加新规则；`:117` 的强制引号可由格式代理判断后处理。
- `:82` 可写 `CTAs co-scheduled on SMs of one GPU processing cluster; size fixed at launch`；`:84` 可写 `Peer SMEM in a cluster supports loads, stores and atomics while that SMEM exists`。保留共调度范围和 SMEM 生命周期。
- 不缩写软件版本、引用路径和设备身份；`:173` 保留会改变机制理解的限定词已足够，不补一段重复清单。

### DOC3

- `3.design.math_reshape.md:11 / :58` 把 thesis 从“计算方式、收益代价摘要”改为 `简述采用或保留该计算方式的主要依据与取舍，不复述公式和账本`，两处对应一致。
- `:66 / :231` 移除现有 150 字符硬限，保留 baseline/gain/cost 的对照、单位、范围及误差来源职责。`:229` 的入选判据可压成 `账本只记录从公式和形状中不易直接看出的收益或代价，并给出比较基线`。
- `:169` 的 strategy 只需 `Bk = 64`，后半句已由 iteration 的结构表达。`:172` 的 meaning 可写 `Reduction block of K`；相关 shape、dtype、direction 均保留。
- `:114–117` 的 symbol 示例可在保持结构的前提下缩说明；`:245–247` 去句末点，其他账本内容已紧凑。
- `:179` 已明确 math 只写计算，本轮不再加同义段落。`:75` 的未消费结果保留义务涉及提取忠实范围，不借本轮文字精简改变。

### DOC4

- `4.design.work_schedule.md:88` thesis 改为主要工作组织的依据与取舍；`:89` describe 保留计算职责。
- `:97 / :151 / :191` 分别已有分解、映射、分配职责，只把含糊的“可以写”改为主动说明缺少的规则即可。
- `:394` 不再将 describe/thesis/strategy 混为一类：`meaning 说明条目的本地数学含义；describe 说明工作职责；strategy 补充本层结构未给出的规则；thesis 说明整体选择的依据与取舍`。
- `:124 / :131 / :134 / :368` 例子的 `Each work coordinate selects...`、`Each local iteration reads...` 可分别缩为 `Independent chunk`、`Value-column block`、`U chunk for this worktile's value columns`、`Expert work item`。后者的 `range: [0,null]` 不动。
- `:219–222` refill 示例可缩为 `Counter starts at 0 before launch; each worker atomically increments it by 1 and uses the old value if below worktile_count, otherwise stops fetching`。不能删计数器初态、原子性、返回旧值或终止条件。

### DOC5

- `5.design.tile_pipeline.md:67` thesis 改为局部执行选择的依据、主要限制及取舍，避免重述动作序列。
- `:187–194` 保留现有“无法唯一确定才补明”原则；`:506` 却要求所有跨动作输入必须逐项再列，与前节形成重复义务。建议直接改为 `按动作说明规则补明尚未确定的输入来源、版本、区域和副本，并对照上游核对跨动作输入完整性`。必要输入核对仍保留，已明确的上游事实不再强制重写。
- `:99 / :105` 例子可简化为角色 `Compute and write Y`，动作 `Compute Y from its upstream inputs and write the materialized result`；`:160–166` 删 `following the softmax formula`，保留未窄化语义。
- `:395 / :412 / :471` meaning 写存储用途，不复述 shape/depth。`:625–627 / :659–661` 保留 worker 本地序号、局部迭代、连续序号区别，可去多余定语。
- 保留等待/事件/成员/版本和 view 语法；本轮不把这些字符串当自由段落压缩。

### DOC6

- `6.realization.md:40–46 / :121–125 / :159–166` 已完整覆盖 require/prefer/forbid 与 item/choice/reason 的职责，不增加新检查清单。
- `:132–133` 的 choice 可简化为 `tcgen05.mma for this role's products`；reason 写它提供上游所需 TMEM 写入与先读后覆写保证，不复述整条角色流程。
- `:141–142` 的 choice 保留既定逐元素运算和舍入，reason 保留 FMA 指令数减少这个具体成本；`:79` feedback 仅写实际替代，不添实验摘要。
- `:168` 可压成 `必要兼容条件保留；已由上游表达的约束删除，属于设计的决定归回 Design`。原文的篇幅防御说明不必保留。

## 后续工具事项

1. 字符数和行数按解析后类型分类，不能把 `item`、引用、表达式和公式视为普通说明；明确 gap 是全局字符串，strategy 有三种，补上 refill 与 feedback.choice。
2. 可选字段“可空”需区分省略与显式空值；当前 ledger 可整体省略，出现时 baseline 必填，gain/cost 按需省略，不能写空字符串替代。
3. 无引号偏好必须保留 YAML 语义；`None` 是字符串哨兵，依赖记法可能需要引号或块标量。统一大小写会触及身份与引用，不是本轮文字修改。

## 已实施与验证

按主代理分工完成正式 DOC0–3 修改，DOC4–6 的建议交另一位代理处理：

- DOC0 集中说明文字职责和 YAML 普通字符串写法；DOC1 收窄 math/range/default 的文字职责；DOC3 thesis 改为依据与取舍，删除 ledger 字符数与行数限制。
- 示例去冗词、句末点和装饰性引号，保留必要的流式标量引号；引用值、版本、数学边界、精度、验收阈值、初态和存续关系不变。
- `git diff --check` 通过；DOC0–3 的 16 个 fenced YAML 全部解析成功。与 `/tmp/schema_text_guidance_d47ij189` 基线逐树比较，映射键及顺序、类型、集合长度、非文本值全部不变，仅有 10 个预期说明文本值变化；装饰性引号变化的解析值全部相同。
- 未新建测试，未修改工具、模板、现行实例或归档，未提交 commit。

当前可直接实施的是普通字符串省略装饰性引号；需要维持 YAML 字符串语义的引号仍保留，统一无引号风格及身份大小写留待后续处理。
