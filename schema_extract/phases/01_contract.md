# P1：共享 Contract

本页完成七段内容中的第 1–3 段，为整个算子建立共同前提。检查器的适用范围见[手册](../README.md#命令与当前可用性)。

## 准入与应读材料

Runner 提供算子身份、输出位置、固定来源入口、输入构造、reference、数值验收材料、适用范围和本次可写路径；续修时还提供当前 root 草稿、checkpoint 与退修意见。缺少必需材料时，先定位缺项及影响，可以继续不受影响的内容，不能假定来源已核实。

开始时读取[公共约定](../../schema_docs/0.common.md)、[算法要求规范](../../schema_docs/1.contract.algorithm_requirement.md)、[starter.yaml](../../schema_template/starter.yaml)及[模板的资料与创建规则](../../schema_template/README.md#1-确认资料与范围)。沿任务说明、IO、输入构造、reference 与验收入口读取固定源码，只追到本段所需范围；此时不展开各路径的 Design，不读硬件资料。

## worker：可写范围与操作

仅可写本次任务的 `root.yaml`：新建时复制 starter，完成 `meta_data` 与 `contract.algorithm_requirement`，硬件区原样保留，Design 暂留草稿。续修读取现有草稿，不覆盖为新 starter。不得写 dispatch 索引、派生 YAML、源码、规范、检查器或 Runner 状态。

### 第 1 段：元数据

从任务要求和外部 IO 识别矩阵收缩、归约、状态递推及其组合，初步区分独立轴和依赖关系。分清 reference、baseline 与选定实现，核实任务允许的输入和实现实际支持的范围。真正外部要求无法共用时才提出拆为不同算子；实现覆盖不足本身不改变任务身份。

按公共约定填写 `meta_data.name`、`describe`、实际设计依据 `source` 与 `version`。任务名称描述计算，不含 tile、架构或分支；源码引用能回到固定版本和真实入口。代码来源按 `repository@ref:path::function` 定位需要的入口，不伪造 ref；来源归一或入口缺失交回 Runner/Supervisor 补齐。新文件从 `version: 1` 开始，后续内容修订按[版本规则](../../schema_template/README.md#3-填写与修订)递增。root 的代表路径与必要的 `dispatch.when` 留到 P2 确定。

### 第 2 段：算法要求

`contract.algorithm_requirement` 只在 root 保存一份，供所有路径共用。从任务的输入构造、reference 和验收入口写清输入输出、有效域、数学关系、边界、精度转换和副作用。核对实际调用的 shape、布局、可选参数、读写及初始化区域；明确外部输出与内部 workspace 的区别，避免默认值域排除合法输入。

验收判据逐输出核对比较对象、相对误差基准（包括材料中的 `relative_to`）、计算精度、阈值及非有限值处理。阅读数值验收依据不等于执行 GPU 验收；本流程不安排 GPU 测试或实测。

材料已明确参考入口、比较对象、误差基准和阈值时，未声明的外部比较器计算精度或 NaN/Inf 行为可保留为未知备注。忠实填写已知判据，在回执说明未获支持的保证及其影响；不得用 reference 内部算术或 schema 默认值替外部比较器补证，也不能仅因这些细节未声明就将明确阈值判为缺失。reviewer 独立判断备注是否影响本阶段提取；真正缺少标准答案、比较对象或必要阈值仍须给出阻塞条件。

写完后，读者应能不依赖 Design 理解标准答案。生成参数、函数签名和候选实现限制不能单独决定任务要求；分块、映射、缓冲、执行顺序和候选数学改写留给 Design。公式、dtype、range、default、symbol 和 tolerance 按算法规范填写，不自行增加字段。

`golden_reference.source` 必须定位实际、固定的参考入口，不能用设计源码冒充。已知参考与契约差异按 `golden_reference.gap` 记录差异及处理方式，不把测试进度写入该字段；没有参考入口时按[参考入口规则](../../schema_docs/1.contract.algorithm_requirement.md#golden_reference)取得并固定参考实现。写参考代码不属于本页权限，所缺入口交 Supervisor 补齐；不编造来源或阈值，不以 `null` 通过验收。

### 第 3 段：硬件占位

`contract.hardware_specification` 原样保留本次 [starter](../../schema_template/starter.yaml) 的占位；撰写、搜索、检查、审阅与收尾均跳过该区。不填设备或能力事实，不提出硬件缺口，也不因此阻塞。完成 YAML 的检查显式使用 `--skip-hardware`。后续 Design 仍须如实描述源码使用的运算、执行者和同步关系，不由源码推断硬件保证。

每段写完保存，再按问题回查。非硬件资料不足或源码与要求不一致时，按[缺项边界](../../EXTRACT_SCHEMA.md#来源缺项与验收边界)保留事实、定位字段和恢复条件。来源明确的转录错误可纠正；实际外部要求变更交任务方确认，不能为当前实现缩小 Contract。

## reviewer：放行门槛

读取独立 checkpoint，对照相同固定材料核对：

- 元数据身份和来源准确；YAML 可解析，硬件占位保持原样。
- 输入有效域、全部输出、边界、精度转换与副作用可独立理解；没有把实现限制或 Design 选择转成算法前提。
- reference 可实际定位，误差依据覆盖全部输出，差异与缺项已按各自用途记录。
- 实现覆盖不足和已知源码缺陷按事实说明；关键算法要求或验收依据仍未知时，给出具体补充条件。

P1 按元数据和共享 Contract 验收，Design 可以保留 starter 草稿。完整检查器因 Design 未完成产生的失败不能作为 P1 退修理由；不声称完整 YAML 已检查通过。既有实现缺陷不自动否决源码忠实的提取，真正缺少关键外部事实则不能放行。

## 回执与下一位

按[回执约定](../README.md#回执与短交接)返回根文件改动、IO/reference/验收依据的文件行号与短引文、共享 Contract 结论、实现覆盖范围和未决项；框架问题按[分流规则](../../EXTRACT_SCHEMA.md#框架问题分流)填 `framework_candidates`。退修与裁决遵循[统一规则](../README.md#退修与裁决)。

Runner 验证并保存本阶段放行后进入 [P2](02_dispatch_draft.md)。P1 的 worker 与 reviewer 各自独立，P2 分别续用这两个会话并加载新任务页；本页不生成额外短交接文件。
