# 说明文字与示例精简

两位新的 `gpt-6-astra / xhigh` 子代理独立审阅后分工修改 DOC0–3、DOC4–6，主代理整合并验收。正式文档只明确文本职责并精简示例，数量限制与统一排版留给后续工具。

## 已落实

| 位置 | 改动 |
| --- | --- |
| [公共约定](../../../schema_docs/0.common.md#通用写法) | 用常见术语说明对象、作用或设计依据，尽量一句读懂；已有字段、公式和引用不再复述 |
| [算法要求](../../../schema_docs/1.contract.algorithm_requirement.md#数学表达与参考精度) | math 聚焦公式；range 只写有效域，default 只写默认值或有定义的表达式 |
| [硬件规格](../../../schema_docs/2.contract.hardware_specification.md#引用格式) | 引用直接书写，机制说明保留必要边界，精简示例 |
| [数学改写](../../../schema_docs/3.design.math_reshape.md#账本) | thesis 说明依据与取舍；ledger 只补不易直接看出的收益与代价，移除字符数及行数限制 |
| [工作调度](../../../schema_docs/4.design.work_schedule.md#省略与说明字符串) | meaning 解释本地对象，describe 概括职责；三类 strategy 各补其分解、映射或分配规则 |
| [微流水线](../../../schema_docs/5.design.tile_pipeline.md#describe-要说明什么) | noun.describe 说明对象对应与职责，只补尚未明确的输入及保证；完整核对全部输入，不要求重复列写 |
| [实现交接](../../../schema_docs/6.realization.md#choice-与-reason) | 沿用必要兼容条件 require、具体成本 prefer 的分工，精简 choice/reason 示例 |

示例删除冗词、句末句点和装饰性引号，保留初始化、版本、范围、精度与完成条件。引号采用可解析的默认方案：普通字符串直接写，YAML 语法或字符串类型需要时保留引号。移除 storage 无 view 时强制单行的排版要求。改动记入 [CHANGELOG](../../../schema_docs/CHANGELOG.md)。

## 后续检查清单需要补齐的地方

- `strategy` 有三处：decomposition、mapping、schedule；原清单漏列 mapping
- `refill.<mechanism>` 也承载规则文本；`feedback.choice` 也是说明文字
- `golden_reference.gap` 是整个参考入口的一条字符串，不是逐输出映射
- `item` 是引用或引用列表；source、binding、evidence、wiki 是定位符，retention、pipe、waiting、index 等使用各自的表达语法，不能按普通短文改写
- `range/default` 仍需支持开区间、符号条件和 None 等有定义的表达式；约束的是说明性散文，不是一律禁止 YAML 字符串
- 可选字段省略与显式空值不同；ledger 出现时 baseline 必填，gain/cost 按需省略

字符数、行数、warning/error、强制同行、字段与身份大小写迁移尚未实现。大小写需要同步名称及全部引用；彻底去引号需允许块文本，与无条件单行要求存在冲突。验收来源记录即使精简后仍可能超过拟议限额，后续工具需保留阈值、资料缺口和判据解释所需的信息。

## 验收

- 47 个 YAML 代码块及 2 个修改过的行内声明解析通过，检查重复键
- 与修改前逐树比较，映射键及顺序、集合长度、类型、非文本值不变；42 个说明文本值改写，引用、pipe 和 waiting/event 表达保持一致
- 人工逐项审阅文本变化，保留数学边界、初始化、成员范围、存储与实现兼容条件
- 正式文档链接与差异格式检查通过；未修改模板、工具、算子实例或归档

独立报告：[文字职责与字符串盘点](01_semantics.md)、[示例与格式判断](02_format.md)。报告中的初始行号及建议对应修改前版本，以最终正式文档为准。
