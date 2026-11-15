# 第 3 轮：schema 与普通作者入口实施

已按 [PLAN](PLAN.md) 完成实施者 A 的 10 个独占 Markdown/README。以本轮开始时的工作树及 `/tmp/dispatch_concrete_text_before_eeln7upz` 快照核对增量，保留此前有效修改；未修改实施者 B 的文件、其他 DOC、代码、starter、实例 YAML、图片、归档、输入或运行状态，未提交 commit。

## 主要改动

| 文件 | 完成内容 |
| --- | --- |
| [schema_docs/0.common.md](../../../schema_docs/0.common.md) | 明确一份具体已选设计、含 root 最多 10 份且越少越好、完整身份的独特价值与独立未展开候选的诚实记录。其他配置进入目录，当前输入工作量、尾块、空工作、Clean 与队列等必要条件留在正文。删除源码调查步骤，支持没有源码的原创设计。 |
| [common 的索引与共享规则](../../../schema_docs/0.common.md#选择条件与索引) | 只在此处定义 dispatch 最小格式：范围与完整设计表必需，变体节有内容才出现；标题类型三选一，三个信息项允许短句合写。最终记录直接定位完整 YAML 与已有对象，支持多基线及原始条目组合，不形成层层基线或隐式 patch。相同完整区直接 from，按使用上下文绑定，整个引用图无环，不为共享造配置容器或无独特价值 provider。 |
| [schema_docs/5.design.tile_pipeline.md](../../../schema_docs/5.design.tile_pipeline.md) | 只改成员配置与条件等待示例及必要解释：EPI=4，先 Drain 再 ScaleTM；保留 for/if、t>0、N>0、移位窗口、全部 release→Drain done 和 release→首 matmul。提前准备 scale 的另一种等待位置仅说明为局部变化并链接 common。 |
| [schema_docs/CHANGELOG.md](../../../schema_docs/CHANGELOG.md) | 在 2026-11-13 的既有记录之前增加本轮具体设计、目录、职责和工具状态说明；此前当日调整及历史日期记录保留。 |
| [README.md](../../../README.md) | plan 改为一份具体完整设计，交付包含必要目录记录；专家审阅范围包含目录中的设计信息。链接 common 与执行器对齐页，按冻结协议限定现有自动提取、解析和发布能力。角色权限保留，提取豁免不扩散到普通实现任务。 |
| [DESIGN_SCHEMA.md](../../../DESIGN_SCHEMA.md) | 以具体选定方案逐层推导；相同 workload 的独特完整方案也可并列，其他参数、局部与组合记录进入目录。删除参数统一留同节点和互斥覆盖要求，保留 Contract、逐层自检及原任务验收职责。 |
| [OPTIMIZE_SCHEMA.md](../../../OPTIMIZE_SCHEMA.md) | 按 from 的实际来源读取、绑定和修订；替换前核对使用者、目录差异与实验适用性。同设计原位演进，其他配置探索记录一次，确有并存价值才新增完整身份。 |
| [IMPLEMENT_SCHEMA.md](../../../IMPLEMENT_SCHEMA.md) | 目录定位任务明确选择的设计，读取实际 from 闭包；目录不授予新权限。已授权局部实验按任务范围实现与验证，关键设计未定仍返回 Planner，不普遍要求先补完整算子。保留构建、GPU、正确性、性能及反馈职责；修正原有动作顺序失效锚点并同步 waiting 名称。 |
| [schema_template/README.md](../../../schema_template/README.md) | 创建、修订、共享和交付按 common；删除 cluster 多配置并入同 YAML、直接父创作和 Runner seed/换父细节。保留操作节锚点，明确 starter 未迁移，检查/视图/绘图命令受当前工具协议约束。 |
| [schema_template/tools/README.md](../../../schema_template/tools/README.md) | 保留真实解析、视图、诊断、图工具和回归协议；顶部说明 parent/from_parent 及其他旧结构字段的实际支持范围，链接集中对齐页与 extract tools，不创造新 API。 |
| [schema_template/examples/README.md](../../../schema_template/examples/README.md) | 说明实例与图片未迁移、未按新方法验收；保留 11/7/1/2 共 21 份的真实数量、来源、历史核对范围及已知缺口。明确 FA4 prefill 的 11 份不符合现行上限，旧检查成功不等于新规范合规。 |

## 验证

- 已逐项回读本轮 diff，并用开始快照复核 common、DOC 5 和 CHANGELOG 的增量。DOC 1、3、4、6 与开始快照逐字节一致，保留其原有未提交修改。
- 临时只读脚本检查这 10 个文件中的 203 个本地 Markdown 链接与锚点，最终 0 个失效；外部网页没有重新抓取。common 的既有主锚点和模板到 P3“节点逐段方法”的链接保留。
- 使用 PyYAML 对文档中的 22 个 YAML fenced 片段做语法解析，全部通过。这只是片段可解析性检查，没有运行 schema checker 或声称语义验收。
- 只读核实示例文件数量为 FA4 prefill 11、FA4 decode 7、QSA 1、GDN 2；读取当前 resolver、structure 与 starter 以核对工具页事实。
- `git diff --check` 通过；当前 diff 中没有 Python、YAML 或 PNG 改动。未运行提取、检查器、绘图、GPU 验证或仓库测试，也未新增仓库测试。

## 交界与剩余工作

common 主定义和稳定锚点已在完成后同步给主代理与实施者 B。根导航、模板创建/检查/绘图入口、模板工具页、示例说明及 CHANGELOG 均可到达 B 创建的 [EXECUTOR_ALIGNMENT.md](../../../schema_extract/EXECUTOR_ALIGNMENT.md)。已只读核对集中页与本组工具范围说明一致；没有待 B 代改的作者文件，也没有发现需越界修改其他 DOC 的直接矛盾。

实施者 A 的文档任务无未完成项。执行器、starter、实例及图的对接仍按 PLAN 留作后续代码工作，本报告不声称新提取方法已经可运行；最终独立语义验收由后续轮次完成。
