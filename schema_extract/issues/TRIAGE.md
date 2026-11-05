# 六算子试用后的 issue 评估

2026-11-04，依据[六算子结果初验](../OPERATOR_REVIEW.md)、实际 host 路由和现行规范评估全部 16 项。评估开始时登记为 2 项待决、3 项部分采纳、11 项已解决；下表已同步本轮 Supervisor 验收结果。归档状态说明采取了什么措施，不保证未来实现没有遗漏，也不表示用户逐项审阅过。

优先处理会改变读者对实际计算、提交和参与成员理解的问题；对已经有效的小修保持原有模块边界。当前决定是沿用已有 `dispatch` 表达不同执行结构，补齐成员端点的参数表达，保留 Reviewer 的全文验收；另简化 P4 排版门禁并补齐校注输入收集。不扩成新的流程配置系统，也不以放宽检查或补写不存在的执行行为换取通过。

2.2.6 完成三项代码修复：成员端点的共用解析与参数展开、P4 覆盖节排版检查简化、catalog 可选校注文件的输入收集。最终 47 项提取回归和 152 项模板测试通过；两份 P4 历史失败稿只读重放通过，MLA 当前输入收集包含校注文件。该时点条件流程仍为待决，已确定下文的初步方案，尚未重提取历史产物。

后续 2.2.8 仅澄清现有 plan 的实际执行绑定和跨 role 等待表达，[capture-invocation-boundary](archive/schema_docs/capture-invocation-boundary.json) 归档的是绑定说明，该澄清本身不表示条件流程产物已修订。冻结 2.2.7 下合法的 GDN/FMHA 任务无需因此统一重跑；MoE 采用新冻结须正式 `continue` 并独立重审，原批准不继承。

2026-11-04，GDN 的 [2.2.7 六节点修订版](../../schema_wiki/revisions/2.2.7/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/dispatch.md)已完成 P6、正式 export 和 Supervisor 验收，该侧表达问题已收口。FMHA、MoE 和 Sparse 的实际产物仍待各自收口，`conditional-flow-topology` 整体仍为待决；Sparse 源布局缺陷现按 2.2.9 边界记录归档，不再作为提取阻塞，symbolic 及两个旧 kernel issue 的决定不变。

## 分类口径

2.3.0 完成以下稳定性工具改进，最终 10 个选定场景通过；全部使用隔离状态、固定计数与本地替身，真实旧 run 未升级。

| Issue | 本次决定 | 保留边界 |
| --- | --- | --- |
| [progress-preserving-continuation](archive/extract_workflow/progress-preserving-continuation.json) | 2.4.0 采纳：未完成 run 的连续语义修订、保稿维护与同 run 执行器登记 | 仅隔离验证，真实旧 run 和已完成产物未迁移 |
| [failed-node-recovery](archive/extract_tools/failed-node-recovery.json) | 已解决：单范围结算、恢复及原超时接管 | 未知终止保留 fence，预算和控制优先级不变 |
| [reminder-ack-delay](archive/extract_tools/reminder-ack-delay.json) | 2.4.1 已解决：配置原因已修，入队无 ack 时同 ID 有界补送，后台观察及已授权恢复独立继续 | 用户确认历史 HTTP 403 因主/子 Agent key 不同步，后续已同步修正；有限补送作通用兜底，不保证绝对投递或处理完成 |
| [core-dump-in-snapshot](archive/extract_tools/core-dump-in-snapshot.json) | 部分采纳：运行和诊断 cwd 移到 run/attempt | 未证明宿主 core handler 的历史原因或落盘行为 |
| [budget-and-usage-accounting](archive/extract_workflow/budget-and-usage-accounting.json) | 部分采纳；2.4.1 已收口本轮统计契约及委派证据缺口 | 历史缺采样、子会话 token 与实付未知是外部边界；墙钟预算保留 |
| [pipeline-graph-skip-hardware](archive/schema_template/pipeline-graph-skip-hardware.json) | 已解决：图入口显式透传硬件豁免 | 默认完整检查；旧冻结版本不自动获得能力 |

[conditional-flow-topology](extract_workflow/conditional-flow-topology.json) 继续待决；[sparse-tc-control-layout-overlap](archive/kernel_wiki/sparse-tc-control-layout-overlap.json) 已按部分采纳归档，提取侧证据记录与分流完成，剩余仅为上游未修事实，不是 Schema 待办。已有来源支持同一完整抽象 Design，实际 Design/P6 仍须补齐并独立审阅；上述工具验证与本次台账整理均不改变真实产物或源 kernel 的验收状态。

第一类是核心结果、表达和 Review；第二类是既有功能的局部质量修复；第三类是可选新增能力。类别、是否需要用户讨论与解决状态分别判断，第二类多数已修，无需重构。本次索引文案勘误属于第二类。

## 逐项判断

| Issue（登记状态） | 实际影响与根因 | 长期方案与是否返工 | 本轮及剩余工作 |
| --- | --- | --- | --- |
| [conditional-flow-topology](extract_workflow/conditional-flow-topology.json)（待决） | GDN、sparse、FMHA、MoE 的原产物用正文取消实际提交或交接，读者需要自行改图。 | **优先处理。** 先依现有 `dispatch` 拆分不能合法参数化的实际结构，不增加逐条 guard 或隐式删边。 | GDN 2.2.7 的 6/6、P6 与独立 export 已验收，原两节点保留；FMHA、MoE、Sparse 的实际产物继续独立修订，整体仍待决，Sparse 已登记的低层槽位偏差不阻塞提取。 |
| [symbolic-member-bounds](archive/schema_template/symbolic-member-bounds.json)（部分采纳） | MoE 的真实成员随 EPI、Nt 改变；三处数字解析限制诱发最大成员范围加正文修正。 | **优先处理。** 既有成员路径允许已定义整数端点表达式；共用 `parse_members`，普通检查与有限展开各守边界。只改变成员数量时保留参数化。 | 工具修复已验收，模板正式入口 152 项测试通过；普通检查不证明未绑定表达式，有限图拒绝缺参、非法整数/区间及 worker 越界。旧 MoE 的虚构成员仍需独立产物修订。 |
| [reviewer-document-integrity](archive/extract_workflow/reviewer-document-integrity.json)（已解决） | 原审阅沿用历史批准、只看差异，漏掉正文与结构冲突。 | **重点保留，无需返工架构。** 共用 REVIEW.md、累计内容范围和四项结论，复用现有回执与重开通道；不堆叠额外报告字段。 | 机制已有回归与真实样例证据；本轮补充分组规则引用。它不能代替旧产物修复，也不能由文件锚点机械证明模型确已通读。 |
| [p4-index-layout-hard-stop](archive/extract_tools/p4-index-layout-hard-stop.json)（已解决） | 准确计划表后的说明段曾触发停止；旧补丁改为诊断后再花调用修复排版。 | **已返工简化。** 保留固定表格投影及 plan、父关系、链接、覆盖事实检查，仅放开覆盖节与说明段的位置；移除 `IndexLayoutError` 及特殊修复分支。 | 已验收：最终 47 项提取回归及两份历史失败稿的只读重放通过；不自动改写模型文稿。 |
| [mla-fp8-source-comment-drift](archive/kernel_wiki/mla-fp8-source-comment-drift.json)（已解决） | 注释与执行代码不一致；原六算子批次的 MLA 快照实际未包含已有校注，原归档对输入链路的判断不完整。 | **保留事实原则，已补修输入链路。** `resolve_request` 复用 `_locator` 收集可选 `paths.annotated_entrypoint`，其余绑定保持；不另建证据系统。 | 4 项输入收集测试及最终回归通过；当前 MLA 收集 31 个文件并包含校注，仅对新快照生效。当前校注等价另由已有 `annotation_program` 只读核对，不代表提取新增了等价验证。 |
| [contract-comparator-evidence](archive/extract_workflow/contract-comparator-evidence.json)（已解决） | 已知相对误差基准丢失会改变通过条件。 | **保留。** `comparisons` 绑定输出和明确声明，机械核对可识别事实，Reviewer 判断其余数值语义；不建设通用公式证明器。 | 无本轮返工；继续拒绝丢失或改写已有基准。 |
| [p1-external-comparator-gate](archive/extract_workflow/p1-external-comparator-gate.json)（已解决） | 六项都曾因外部比较器定义不全止于 P1，重复调用无法补齐缺失事实。 | **保留。** 忠实提取与完整复刻外部验收器分开；保留已知判据和明确 source_gap，不把未知用于设计保证。 | 无本轮返工；计算或设计本身必须靠猜测时仍应阻塞。 |
| [zero-work-partition](archive/schema_template/zero-work-partition.json)（已解决） | 检查器曾把合法零工作量当成非法大小。 | **保留。** `size` 非负、`tile` 为正，文档和现有数值规则一致；无需另一种空任务表示。 | 无本轮返工；零工作量不授予删除已声明提交的权限。 |
| [proposal-summary-ownership](archive/extract_workflow/proposal-summary-ownership.json)（已解决） | 语义 summary 与文件哈希都称摘要，造成多次空值修复。 | **保留。** 模型负责非空语义结论，Runner 负责字节哈希；明确命名即可，不由工具代写审阅结论。 | 无本轮返工；实际返工率不能仅由提示修订推断。 |
| [p5-handoff-attempt-locator](archive/extract_tools/p5-handoff-attempt-locator.json)（已解决） | 并发失败节点曾引用兄弟节点最近一次调用，导致诊断读错证据。 | **保留。** 失败记录携带自身 attempt ID；旧记录定位不可靠时只报已知范围，不猜调用。 | 无本轮返工；保持当前轻量定位，不另建监控关联层。 |
| [p6-tree-artifact-evidence](archive/extract_workflow/p6-tree-artifact-evidence.json)（已解决） | 纯产物树事实被要求附上不相干的源码引文。 | **保留。** 树事实用完整产物证据及全树校验；选择、覆盖等源代码语义继续使用来源证据。 | 无本轮返工；与四项文档判断共用既有回执，不合并为模糊的通用 evidence。 |
| [mla-catalog-dynamic-domain](archive/kernel_wiki/mla-catalog-dynamic-domain.json)（已解决） | 导航摘要可能误纳入头数或排除合法输入。 | **保留。** Contract 以本题绑定的 metadata、shape_range 和输入构造为依据；导航不覆盖来源事实。 | 无本轮返工；不为单个目录摘要建立全局域推断系统。 |
| [scaled-gemm-output-abi](archive/kernel_wiki/scaled-gemm-output-abi.json)（已解决） | 通用描述可诱发不存在的 C 输入或尾行掩码。 | **保留。** 按实际接口、reference 和输入构造表达别名与写区，针对来源歧义定位修正。 | 无本轮返工；与源域、注释问题共用明确来源原则，不扩张 schema 字段。 |
| [gdn-slot-padding-coverage](archive/kernel_wiki/gdn-slot-padding-coverage.json)（部分采纳） | 旧 FP32 状态槽候选拒绝部分合法 padding。 | **范围外的上游正确性事项。** 区分完整任务域与候选对齐限制；不能由提取缩小 Contract。 | 提取侧已披露；剩余是上游未对齐槽支持，非本轮六算子的框架修复。 |
| [masked-gemm-tail-write-coverage](archive/kernel_wiki/masked-gemm-tail-write-coverage.json)（部分采纳） | 旧 masked GEMM 的尾块实际写区与合同保护区不同。 | **范围外的上游正确性事项。** 分清数值比较区域、调用方保护区和实际写区，不能用有效前缀冒充写掩码。 | 提取侧已披露；尾块写回修正属于上游实现，非本轮框架工作。 |
| [published-dispatch-draft-status](archive/extract_workflow/published-dispatch-draft-status.json)（已解决） | 已交付索引仍写待 P5 生成，误导读者判断完成状态。 | **保留规则修复，独立文案勘误。** P4/P6 将过程状态留在回执，当前阅读使用独立勘误页；无需重跑整树。 | 两份勘误页已保留 63 个节点及覆盖事实并修正当前入口；原 dispatch、YAML 和回执原样保留，未重审 Design。Supervisor 已验收，文案问题已收口；条件流程与成员表达仍独立待修。 |

## 条件流程的具体方案

公共依据是[参数变化与结构变化](../../schema_docs/0.common.md#参数变化与结构变化)。`when` 可以描述当前调用的运行时事实，且每个节点独立给出定义；它不是把实际探测前移的执行程序。先从真实 host 路由列出不同情形，再剪去不可达组合；不要凭 kernel 数、节点数或少量固定 shape 决定拆分。

### GDN

[固定 host 入口](../../kernel_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/candidate/gdn_kernel.cu) 在 `l2norm` 分支决定 normalize 提交，`kqtma` 决定输入直读或 Q/K 图像交接，`perm_ptr` 决定请求排列，`Gseg` 决定单链或传播/重放。2.2.6 的初步讨论曾从这个较宽接口列出以下五类，再按真实 Perm 存在性核对；下表保留讨论时点，不代表本任务的最终有效域：

| 选择情形 | 实际提交 | 关键交接 |
| --- | --- | --- |
| 非分段、归一化 | normalize、prep、chain | normalize 产生 Qb/Kb；prep 产生 Qc/Kc。 |
| 非分段、不归一化、bulk | prep、chain | Qb/Kb 对应输入；Qc/Kc 来自 prep，没有 normalize 完成边。 |
| 非分段、不归一化、KQTMA | prep、chain | chain 直接读输入；无 Qc/Kc 全局交接。 |
| 分段、归一化 | normalize、prep、propagate、replay | 保留 Qb/Kb、Qc/Kc 及 propagate 产生的 segA/segB。 |
| 分段、不归一化 | prep、propagate、replay | 无 normalize；保留 Qc/Kc 和 segA/segB。 |

本表枚举提交，不用排列顺序代替依赖。归一化完成后 prep 才开始；chain/propagate/replay 的 PDL 启动许可与数据就绪分别保留源代码依据。Perm 存在时登记 prep 的生产及实际消费者；不存在时移除对应交接和读取，不能保留字段后用正文取消。

五类乘以 Perm 两种情况是最多 10 个候选组合，必须按实际硬件与源码条件剪除不可达项，不是已证明的最小节点数。VTT、环深及能用现有参数/索引规则表达的局部数量变化继续合并。修订还须同步数学保留要求、worktile 读写区域和流水线动作，不能只删 flow 条目后宣布修复。

2.2.7 的独立 P1 依据实际动态生成器确认 `norm=false`，`cu_seqlens`、`chunk_indices`、`chunk_offsets` 必有，实际 nullable 只有 `core_attn_out`、`FinalState`、`CoreWritten`；因此 normalize 和 reference 的缺省元数据接口均不属于本任务域。最终采用 `root`、`cached_ordered`、`direct_qk`、`direct_qk_ordered`、`segmented`、`segmented_ordered` 六个完整节点，分段与 direct-QK 不并存。缓存/直接 QK 和 Perm 存在性在各节点的数学、工作组织与流水线中保持一致；最终 P6 还消除了排序与搬运间额外的 CTA 串行等待。

修订 run `45f9bb51-77f3-4285-b7a7-251cf2744fa1` 的最终 checkpoint 为 `349f0fdbad54effd79e2557888dffcc88de418c142438fd2989db2b4937d0a2d`，P6 reviewer 为 `1283b632-a802-4625-b666-a0eb3dce449d`，身份见[交付回执](../../schema_wiki/revisions/2.2.7/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/.extraction/receipt.json)。最终单次 P6 审计为 `verified`，新目录 8 文件一致、原目录 4 文件 SHA 不变，完整证据入口见[试用记录](../TRIAL_REPORT.md)。GDN 的表达修订已完成；源码、原发布与快照不变，core copyback、mini-chunk 读取退休、跨 stream 和外部比较器限制继续保留，未做 GPU 验证。

### Sparse probe

[固定 dual 调度](../../kernel_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/candidate/dsv4_kernel.cu) 的 `probe_begin`、`probe_end`、`dual_dispatch_hidden` 和 `dsv4_run` 给出一次普通内部 dual 路由的五种结构：

| 选择情形 | 实际提交与完成关系 |
| --- | --- |
| hidden，ratio 接受 packed | probe 与 packed prefix 可并行；packed remainder 在 probe 后，不等待 prefix。 |
| hidden，ratio 拒绝 packed | probe 与 packed prefix 可并行；mma remainder 在 probe 后，不等待 prefix。 |
| serial，ratio 接受 packed | probe 后提交全范围 packed。 |
| serial，ratio 拒绝 packed | probe 后提交全范围 mma。 |
| probe 缓冲或映射不可用 | 仅提交 mma，没有 probe 完成事件。 |

hidden 的 caller stream 最后通过 join 覆盖 prefix 和 remainder；不能新增二者之间的完成屏障。ratio 必须依据实际返回值定义：`probe_end` 到达 2,000,000 次轮询上限后仍对当时的正数样本求均值，部分样本也可能使 packed 被接受。超时不表示探测未提交，也不能一概写成 mma 路径。

这里的五类不是整个 `Model.forward` 所有异常轨迹的完整枚举。提交后的 join 错误与 replay 的异常出口可能再次进入完整路径，应保留对应来源限制，不能将已经发生的提交抹掉或称为干净 fallback；无须为此假造一套错误恢复算法。

### 代价与验收边界

MoE 的低 fill 跨 role 等待可用[现有条件等待示例](../../schema_docs/5.design.tile_pipeline.md#条件等待示例)表达：`Hoist=0` 的非首 worktile 以 0/1 repeat 生成带 q 身份的 serial wait，四个 epilogue release 约束 wait.done，随后才 prepare；wait 可以先 issue，完成仍须等齐 release。`Hoist=1` 仅保留 release→首个 MMA，允许 prepare 提前。`nvfp4_moe.cu:4843–4915` 与已验证的默认 97/81 选择提供来源，隔离 fc1/fc2 在 `R=3200/4800,E=256`、每 worker 两个 worktile 的图通过依赖断言且无环。此例没有新增 guards，也未改真实 root；跨 role 前驱不能藏在 prepare 契约中，本 role 内部计算及原语固有保证仍可封装。证据和剩余产物验收沿用 [conditional-flow-topology](extract_workflow/conditional-flow-topology.json)。

沿用 dispatch 会增加完整 Design 的重复和审阅节点，但复用了现有字段、检查器、独立审阅与发布路径。先对真实可达组合去重，只保留结构上的必要拆分；不要把较少节点当成正确性的优先条件。只有独立修订证明确实出现难以维护的重复，才凭具体案例重新评估 flow 内部分支；本轮没有新增该语法，也没有新增 guard 的条件蕴含、事件存在性或替代生产者求解器。

2.2.6 的框架复核没有重新提取历史全树，原六算子的发布 YAML、快照与批准回执保持原身份。随后 GDN 已独立完成 2.2.7 修订并交付；条件流程 issue 的其余算子及旧 MoE 成员表达仍待各自收口。索引文案已通过独立勘误修正当前阅读入口，历史索引原样保留。工具测试与文档检查只验证对应改动，不把检查通过或新的规则说明当作旧产物已修复。
