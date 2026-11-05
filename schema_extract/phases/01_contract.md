# P1：共享 Contract

本页完成七段内容中的第 1–3 段，为整个算子建立共同前提。检查器的适用范围见[手册](../README.md#验证边界)。

以下准入与推进用于普通提取；`maintenance` 只执行本次任务，写入还受调用的文件/字段授权限制，不能据本页扩大范围。

## 准入与应读材料

Runner 提供算子身份、输出位置、固定来源入口、输入构造、reference、数值验收材料、适用范围和本次可写路径；续修时还提供当前 root 草稿、checkpoint 与退修意见。缺少必需材料时，先定位缺项及影响，可以继续不受影响的内容，不能假定来源已核实。

开始时读取[公共约定](../../schema_docs/0.common.md)、[算法要求规范](../../schema_docs/1.contract.algorithm_requirement.md)、[starter.yaml](../../schema_template/starter.yaml)及[模板的资料与创建规则](../../schema_template/README.md#1-确认资料与范围)。沿任务说明、IO、输入构造、reference 与验收入口读取固定源码，只追到本段所需范围；此时不展开各路径的 Design，不读硬件资料。

## worker：可写范围与操作

仅可写本次任务的 `root.yaml`。Runner 用 starter 准备首次稿件，worker 核对后完成 `meta_data` 与 `contract.algorithm_requirement`；硬件区原样保留，Design 可暂留草稿。已有稿件继续使用，不能覆盖为新 starter。不得写 dispatch 索引、派生 YAML、源码、规范、检查器或 Runner 状态。

### 第 1 段：元数据

从任务要求和外部 IO 识别矩阵收缩、归约、状态递推及其组合，初步区分独立轴和依赖关系。分清 reference、baseline 与选定实现，核实任务允许的输入和实现实际支持的范围。真正外部要求无法共用时才提出拆为不同算子；实现覆盖不足本身不改变任务身份。

按公共约定填写 `meta_data.name`、`describe`、实际设计依据 `source` 与 `version`。任务名称描述计算，不含 tile、架构或分支；源码引用能回到固定版本和真实入口。代码来源按 `repository@ref:path::function` 定位需要的入口，不伪造 ref；来源归一或入口缺失交回 Runner/Supervisor 补齐。新文件从 `version: 1` 开始，后续内容修订按[版本规则](../../schema_template/README.md#3-填写与修订)递增。root 的代表路径与必要的 `dispatch.when` 留到 P2 确定。

### 第 2 段：算法要求

`contract.algorithm_requirement` 只在 root 保存一份，供所有路径共用。从任务的输入构造、reference 和验收入口写清输入输出、有效域、数学关系、边界、精度转换和副作用。核对实际调用的 shape、布局、可选参数、读写及初始化区域；明确外部输出与内部 workspace 的区别，避免默认值域排除合法输入。

外部对象可为张量或 `None` 时，按[nullable 与 None](../../schema_docs/1.contract.algorithm_requirement.md#nullable-与-none)填写并审阅共享 Contract；依据实际来源核对缺席与省略参数的条件。

验收材料逐输出核对比较对象、相对误差基准（包括 `relative_to`）、计算精度、阈值及非有限值处理。来源已给出的内容必须保留，未提供的定义按[验收来源记录](../../schema_docs/1.contract.algorithm_requirement.md#固定实现提取中的验收记录)处理。阅读这些材料不等于执行 GPU 验收；本流程不安排 GPU 测试或实测。

绑定的验收 JSON 若在具名输出的 `tolerance` 中明确写了 `relative_to`，P1 提案用 `comparisons: [{output, source_output, evidence}]` 将它绑定到实际 YAML 的 `golden_reference.tolerance.<output>`。`source_output` 是验收记录的 `name` 或 `outputs` 的键；例如 `out → Y`、`input.state[*] → State`，名称相同则绑定同名输出。仅 `input.state[0]`、`input.state[1]` 这类已知状态族可用 `input.state[*]` 汇总，每种不同的比较基准或阈值仍要有声明行的独立锚点。返回位置（如 `output[0]`、`output[1]`）及其他下标保留各自完整名称，即使基准与阈值相同也不合并，必须分别引用对应的声明行。`evidence` 引用本次 `source_evidence`；所选行范围与 `quote_offset` 生成的引文必须包含该输出的 `"relative_to": "candidate"` 等实际声明，不能只取 JSON 开头。worker 和 reviewer 各自核对映射并选择自己的锚点；没有这类明确声明时用 `comparisons: []`。

对应 YAML 判据明确保留 `relative_to=candidate` 或来源给出的其他值。例如材料已给出 elementwise、`atol=0.01`、`rtol=0.05` 时，可写 `abs(candidate-reference) <= 0.01 + 0.05*abs(candidate); relative_to=candidate`，并在后面说明未声明的比较器精度或 NaN/Inf 行为。Runner 核对这个显式基准、已有标准公式中的比较对象与可直接核对的阈值；自由公式及输出映射的语义仍由独立 reviewer 核实。不因单独一个 `relative_to` 字段推断 elementwise/L2 公式、运算精度或非有限值政策。

材料未声明 `relative_to` 或比较器实现时，不能把“没有 override”、reference 内部算术或 schema 默认值当作依据。按[来源缺项](../REVIEW.md#来源缺项)保留 `tolerance` 来源记录和 `source_gap`：已知判据完整保留，未知公式、基准、归一化或检查路由不自行补齐；当前计算与设计有据时可完成并批准。真正阻碍当前提取的缺项须指出具体关系，来源已明而草稿不符则退修。

写完后，读者应能不依赖 Design 理解标准答案。生成参数、函数签名和候选实现限制不能单独决定任务要求；分块、映射、缓冲、执行顺序和候选数学改写留给 Design。公式、dtype、range、default、symbol 和 tolerance 按算法规范填写，不自行增加字段。

`golden_reference.source` 必须定位实际、固定的参考入口，不能用设计源码冒充。已知参考与契约差异按 `golden_reference.gap` 记录差异及处理方式，不把测试进度写入该字段；没有参考入口时按[参考入口规则](../../schema_docs/1.contract.algorithm_requirement.md#golden_reference)取得并固定参考实现。写参考代码不属于本页权限，所缺入口交 Supervisor 补齐；不编造来源或阈值，不以 `null` 通过验收。

### 第 3 段：硬件占位

`contract.hardware_specification` 原样保留本次 [starter](../../schema_template/starter.yaml) 的占位；撰写、搜索、检查、审阅与收尾均跳过该区。不填设备或能力事实，不提出硬件缺口，也不因此阻塞。完成 YAML 的检查显式使用 `--skip-hardware`。后续 Design 仍须如实描述源码使用的运算、执行者和同步关系，不由源码推断硬件保证。

每段写完保存，再按问题回查。非硬件资料不足或源码与要求不一致时，按[缺项边界](../../EXTRACT_SCHEMA.md#来源缺项与验收边界)保留事实、定位字段和恢复条件。来源明确的转录错误可纠正；实际外部要求变更交任务方确认，不能为当前实现缩小 Contract。

## reviewer：放行门槛

先按[当前文档验收](../REVIEW.md)通读当前 root 的元数据与算法 Contract，再核对来源，给出四项文档判断。只审本阶段已完成内容，Design 草稿不在本阶段批准范围；后续阶段仍须复核这份 Contract 与新内容是否一致。

从独立 checkpoint 和固定材料确认：这份紧凑的 Contract 已说清外部任务的数学含义、有效域及全部输出，读者无需展开 Design 才能理解标准答案。重点核对：

- 元数据身份和来源准确；YAML 可解析，硬件占位保持原样。
- 输入有效域、全部输出、边界、精度转换与副作用可独立理解；没有把实现限制或 Design 选择转成算法前提。
- reference 可实际定位，每个输出的已知判据或验收来源记录均已覆盖；已知阈值、基准和检查没有被遗漏、改写或用默认公式替代。
- 来源缺口及不影响当前提取的理由有证据，未被用来支持数值保证；实现覆盖不足和源码缺陷按事实保留。

Runner 对 complete/approve 检查算法区字段、输出与 tolerance 对应、符号/属性引用、shape 轴与范围，这些要求不能留到 P3。完整检查器因 Design 未完成产生的失败不作为 P1 退修理由；但元数据或算法 Contract 尚未完成的 partial 草稿仍须继续原作者工作，不能借维护复核获批。

## 回执与下一位

按[回执约定](../tools/README.md#回执与短交接)提交语义提案：共享 Contract 结论、IO/reference/验收依据、实现覆盖范围和未决项。

Runner 验证并保存本阶段放行后进入 [P2](02_dispatch_draft.md)。P2 worker 续用本角色会话，reviewer 使用新的阶段会话，从当前文档独立建立理解；本页不生成额外短交接文件。
