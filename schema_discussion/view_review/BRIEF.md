# View 的必要信息与表达层级

本轮重新评估 storage 下的 view。用户关心两件事：不要让 Agent 提前选择无关的底层布局；也不要删去分析容量、分时复用、别名与流水线所需的信息。此前保留 view 的结论不作为本轮前提。

主要问题是：dtype 是否重复定义，slot_index/part 是否过细或过早，以及 per-value 信息应留在 view、提升到 storage，还是采用更简单的单一结构。需要同时判断语义、层级和填写成本，不能只比较缩进层数。

## 输入

inputs.json 记录本轮当前 docs 的固定副本。当前正式字段已经是 storage/index/event；前一轮 phase_grammar/sources/rules 是改名前快照，不作为本轮现行规范。

复用 ../corpus.json 的六个固定 root、分支与十八份源代码；这些语料的 hash 已与原 working checkpoint / inputs.lock 核对。旧 YAML 仍用 resource 等旧字段，应按语义分析，不把改名当作缺口。前一轮报告可用作证据定位，但本轮要重新判断哪些内容应早定、可推导或应留给实现。

至少覆盖：同类型分时复用；FP32/FP16/FP8 混合表示的 S/P；局部区域与仍存活值并存；多槽中不同版本的映射；只存一个值的普通存储。候选资料包括六 root，以及 MoE/direct、Sparse/tc_packed、GEMM/mma_sync、FMHA xpose 等已冻结分支。具体选择由各代理独立决定，不要求把整树重新提取。

## 比较要求

1. 把上游数学值 dtype、存储容量的计量 dtype、实际存放/打包格式分开核对；不得只因拼写相同判重复，也不得把已由上游唯一确定的类型算成独立设计自由度。
2. 区分容量/别名关系、版本到槽、部分区域共用，与任意地址偏移/线程 layout。需要证明某个精确位置会改变哪些设计约束，或为何可以晚定。
3. 尝试用户建议：删除若干 view 属性；将属性提升至 storage 条目；扁平的逐值条目加共享关系；或保留精简 view。用同一批真实片段比较，不能把缺失事实悄悄移进长 describe。
4. 没有内容、单值、整槽、同类型等简单情况应尽量简洁；复杂案例仍要能识别异地副本、重叠区域、谁先用完以及字节数只计一次。
5. 提出删字段时明确其替代来源、成立条件与反例；提出多种写法时最终推荐一个一致的规则，不拼成随意任选的语法集合。

## 交付和分工

两位全新 gpt-6-astra/max 代理独立调查，不读取对方的新报告。只写分配的 reports/01_semantics.md、data/01_semantics.json 或 reports/02_forms.md、data/02_forms.json，及各自需要的 scripts/01_* / 02_*。不修改本 BRIEF、inputs.json、综合 README、正式 docs、算子、模板或旧研究材料，不再派子代理。

每份报告应有：明确建议；六算子相关覆盖；少量强反例；同一实例的现写法与候选写法；哪些是设计必要信息、上游可推导信息、可晚定布局或纯实现细节。不要泛论所有 GPU 或再次讨论整个 phase/while/verb 设计。

JSON 顶层使用 coverage/findings/proposals。每个关键发现有 id、claim、recommendation、evidence；证据含仓库相对 path、start_line、end_line、原文 quote。区别全量结构扫描与实读范围，保持 root/分支及批准/候选状态，精确引用冻结 YAML 与源码。

本轮只研究规则，尚不修改现行规范或实际算子，不执行 GPU。
