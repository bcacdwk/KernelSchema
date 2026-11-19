# P1：共享 Contract

本页完成七段内容中的第 1–3 段，为整个算子建立任务元数据和共同 Contract。建立 `root.yaml` 不代表已经选定 root Design；具体代表在 P2 选择。

## 准入与应读材料

准备算子身份、输出位置、固定实现入口、输入构造、reference、数值验收材料、适用范围和本次可写路径；续修时读取当前 root 草稿与具体反馈。缺少必需材料时，先定位缺项及影响，继续不受影响的内容，不假定来源已核实。

读取[公共约定](../../../schema_docs/0.common.md)、[算法要求规范](../../../schema_docs/1.contract.algorithm_requirement.md)、[执行单元的既有字段规则](../../../schema_docs/2.contract.hardware_specification.md#具名映射)、[starter 的硬件占位](../../../schema_template/starter.yaml)及[模板的资料与创建规则](../../../schema_template/README.md#1-确认资料与范围)。沿任务说明、IO、输入构造、reference 与验收入口读取固定源码，只追到本阶段所需范围，不展开各实现的 Design，不读硬件资料。

## worker：可写范围与操作

仅可写 `root.yaml` 的任务元数据、`CONTRACT.Algorithm_requirement`，以及下述 `CONTRACT.Hardware_specification.Execution` 的有据登记；硬件区其余字段原样保留，Design 可暂留草稿。按规范建立首次稿件；已有稿件继续使用，不覆盖为新模板。不得写 dispatch 目录、其他完整 YAML、源码、规范、检查器或运行状态。

### 第 1 段：元数据

从任务要求和外部 IO 识别矩阵收缩、归约、状态递推及其组合，初步区分独立轴和依赖关系。分清 reference、baseline 与固定实现，核实任务允许的输入和实现实际支持的范围。真正外部要求无法共用时才提出拆为不同算子；实现限制和设计数量不改变任务身份。

按公共约定填写 `META_DATA.Name`、`Describe`、实际固定实现依据 `Source` 与 `Version`。任务名称描述计算，不含 tile、架构或分支；代码来源按 `repository@ref:path::function` 定位真实入口，不伪造 ref。来源或入口缺失交 Supervisor 补齐。新文件从版本 1 开始，后续内容修订按[版本规则](../../../schema_template/README.md#3-填写与修订)递增。P2 选择具体 root 时再核对相应来源与承担范围。

已有非 root 后更正同一固定任务的 `Name` 或 `Describe`，worker 仍只改 root。Runner 在本次候选事务内将规范化后的这两个共享字段同步到当前已有草稿，逐文件递增 Version，保留原稿、正文和历史来源。reviewer 核对最终快照中全部 YAML 的共享字段并引用各文件；这只批准 P1 的共同前提，不将尚未完成的 Design 当作已批准。已有正文随后按实际受影响的使用上下文复核，P5 不得自行改这两个字段。

### 第 2 段：算法要求

`CONTRACT.Algorithm_requirement` 只在 root 保存一份，供所有设计共用。从任务说明、输入构造、reference 和验收入口写清输入输出、有效域、数学关系、边界、精度转换和副作用。核对实际调用的 shape、布局、可选参数、读写及初始化区域；明确外部输出与内部 workspace 的区别，避免默认值域排除合法输入。

先核对各材料声明的作用和范围。任务明确指定的范围材料按其声明使用；签名更宽、生成器只构造部分调用或候选作者另有概括，都不自动改变任务域。材料不一致时，定位到具体对象、缺省条件或约束，写出已确认事实及尚不能确定的部分；缺少裁定依据时保留该缺口，继续不依赖它的内容。不能用候选实现的支持范围补齐外部要求。

外部对象可为张量或 `None` 时，按[nullable 与 None](../../../schema_docs/1.contract.algorithm_requirement.md#nullable-与-none)填写并审阅，依据实际来源核对缺席与省略参数的条件。

逐输出核对验收比较对象、相对误差基准（包括 `relative_to`）、计算精度、阈值及非有限值处理，并将验收材料中的输出对应到当前 Contract 的全部输出。返回位置和具有不同判据的状态输出分别核实；名称转换不能丢掉任何已有声明。worker 和 reviewer 各自读取声明并选择实际依据，阅读材料不等于执行 GPU 验收。

提案的 `comparisons` 覆盖验收 `.json` 中每个具名输出的 `tolerance.relative_to` 声明：`output` 填当前 `Golden_reference.Tolerance` 的精确键名，`source_output` 填验收输出名，`evidence` 索引覆盖该声明行的本次 `source_evidence`。无此类 JSON 声明时填 `[]`。比较器代码和 Markdown 按普通来源证据核实；其他 Contract 事实写入正文与 conclusions。

对应 `Golden_reference.Tolerance` 保留来源给出的判据。来源明确给出完整公式（例如逐元素 `abs(candidate-reference) <= 0.01 + 0.05*abs(candidate)`）时写出该判据，并说明未声明的计算精度或 NaN/Inf 行为；若仅给出 `elementwise`、`atol=0.01`、`rtol=0.05`、`relative_to=candidate` 等标签而未声明组合公式，只保留已知项并注明公式缺失，不能推成加和公式或推断计算精度、非有限值政策。

材料未声明的比较器事实按[验收来源记录](../../../schema_docs/1.contract.algorithm_requirement.md#固定实现提取中的验收记录)和[来源缺项](../roles/reviewer.md#来源缺项)处理。已有判据完整保留；未知公式、基准、归一化或检查路由不自行补齐，也不能用“没有 override”、reference 内部算术或 schema 默认值作依据。当前计算与设计可确定时，说明限制后继续；必须猜测当前关系时阻塞具体范围，已知事实写错或遗漏则退修。

写完后，读者应能不依赖 Design 理解标准答案。生成参数、函数签名和候选实现限制不能单独决定任务要求；分块、映射、缓冲、执行顺序和候选数学改写留给 Design。公式、dtype、range、default、symbol 和 tolerance 按算法规范填写，不自行增加字段。

`Golden_reference.Source` 必须定位实际固定参考入口，不能用设计源码冒充。已知参考与契约差异在 `Golden_reference.Gap` 记录差异和处理，不写测试进度；没有参考入口时按[参考入口规则](../../../schema_docs/1.contract.algorithm_requirement.md#golden_reference)交 Supervisor 取得并固定，写参考代码不属于本页权限。不编造来源或阈值，不以 `null` 通过验收。

### 第 3 段：硬件占位

`CONTRACT.Hardware_specification.Execution` 使用 DOC2 已有的具名映射，最小登记给定材料或固定源码已明确使用的执行单元，作为后续 worker 与 role 的名称来源。在所属单元摘要中同时登记后续 `Storage.location` 需要引用的存储名称及范围，例如 thread 的 `register`、CTA 的 `smem`；纯存储不因此成为执行者。只填写有依据的执行层级、组成及所属资源作用域，不把某次 launch 的线程数或某个实现配置写成设备上限；保留每项的来源依据。后续需要补充单元或存储名时回 P1 更新唯一定义，不在 Design 中另造登记位置。

硬件区其余字段原样保留本次 [starter](../../../schema_template/starter.yaml) 的占位，不开展硬件资料搜索、能力或容量调查，不把这些资料缺口列为提取待办或阻塞理由。适用的 YAML 检查使用 `--read-only --skip-hardware`；worker/reviewer 人工核实执行单元登记的来源，不把跳过检查当作已获硬件保证。后续 Design 如实描述源码的运算、执行者和同步关系，登记单元或存储名称不证明设备规模、容量或原语能力，也不要求调查编译后的寄存器分配或取得无 spill 证明。下游引用与来源规则保持不变；已有源码事实因此无法合规表达时，按[框架问题](../roles/reviewer.md#发现问题与批准)定位受影响范围。

每段写完保存，再按问题回查。非硬件资料不足或源码与要求不一致时，按[缺项边界](../../../EXTRACT_SCHEMA.md#来源缺项与验收边界)保留事实、定位字段和恢复条件。来源明确的转录错误可纠正；真实外部要求变更沿任务边界处理，不能为当前实现缩小 Contract。硬件与 GPU 豁免只适用于本次固定实现提取。

## reviewer：放行门槛

按[当前文档验收](../roles/reviewer.md)先通读当前任务元数据、算法 Contract 与最小执行单元登记，再核对固定材料，完成四项判断。Design 草稿和 root 代表选择不在本阶段批准范围；后续仍须复核共享 Contract 与新内容是否一致。

- 任务身份与来源准确，YAML 可解析；执行单元按已有字段登记且有来源依据，硬件区其余占位保持原样。
- 外部数学、有效域、全部输出、边界、精度转换及副作用可独立理解，没有将实现限制或 Design 选择变成算法前提。
- reference 可实际定位，每个输出的判据或验收来源记录均已覆盖，已知阈值、基准和检查未被遗漏或替换。
- 来源缺口及不影响当前提取的理由有据，未被用于支持数值保证；实现覆盖不足与源码缺陷按事实保留。

算法区字段、输出与 tolerance 对应、符号/属性引用、shape 轴及范围在 P1 完成，不留到 P3。Design 尚未完成不构成 P1 的完整检查失败理由；本阶段自身尚未完成的草稿仍须继续撰写，不能通过维护复核变成已批准内容。

## 回执与下一位

在既有回执中保留共享 Contract 结论、逐输出验收对应、IO/reference/验收依据、实现范围和未决项。具体编码见[回执协议](../operations/response.md)。不另生成固定交接文件。

放行后进入 [P2](02_dispatch_draft.md)。P2 的 worker 与 reviewer 均新建独立会话，从本次材料和当前文档建立理解。
