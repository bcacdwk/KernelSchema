# 六个算子的结果初验

2026-11-04 初验时，对照 schema_template 及其引用的 schema_docs 阅读当时推荐产物，基线为 `6e4ca91bd91d998cac99e8dadd83838e153c679d`；GEMM 使用 2.2.4 校正版。初验范围是六份 root 的各分区主干、六份 dispatch，以及代表分支的数学主张、工作组织和流水线，包括 GEMM 两个分支、FMHA 七个分支、稀疏注意力两个分支、GDN 分段分支和 MoE 各主要执行族。初验仅阅读；随后按用户追问，单独核对了检查器执行证据并做隔离副本对照，见下文。没有逐项核对源码数值或进行 GPU 验证，也不声称逐行审完当时的 77 份 YAML。

本页保留各次复核的时点。GDN 后续的 2.2.7 修订验收见文末；初验表和两节点对照仍对应原始发布，不回填为旧产物已经修复。

**核心内容的分层基本成立，但本轮不能给六项全部无保留通过。** 两项未发现阻止初验的表达问题，四项用未经规范定义的方式表达条件流程，MoE 另有成员范围问题。首先应纠正对 schema_docs 的遵循；这些输出不足以直接证明语言需要扩展。已有 P6 批准和证据审计说明原流程完成，本次新增的表达质量结论独立记录，原发布历史保留。

| 算子 | 本轮判断 | 算子侧依据 |
| --- | --- | --- |
| [fp8_scaled_grouped_gemm](../schema_wiki/revisions/2.2.4/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/root.yaml) | 基本通过 | Contract 定义分组缩放乘积与原地输出；数学层表达按 scale block 求积再提升精度；调度和流水线分别表达持久工作与两种指令路径。3 个节点的差异可以理解。 |
| [nvfp4_fused_moe](../schema_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/root.yaml) | 保留意见 | 准备权重、路由、激活、量化、两次投影及 BF16 route 后加权都有清楚归属。分支用正文取消已列出的提交和成员，需处理下述两个表达 issue。 |
| [fmha_mxfp8_qkv](../schema_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/root.yaml) | 保留意见 | Contract 保留 MXFP8 解码及概率舍入的数值组；融合、分片、前缀重放的差异落在数学/调度/流水线。split_tcg 的可选 segment 提交依赖正文改变 flow。 |
| [mla_expanded_fp8](../schema_wiki/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/root.yaml) | 基本通过 | 准备好的 Q/K/V 与本算子边界明确；在线状态、概率精度及 tiny 额外重算均能读出，工作队列和跨 worktile 复用放在下游。单 root 的组织在本轮范围内合理。 |
| [dual_cache_latent_bf16](../schema_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/root.yaml) | 保留意见 | 两池共享 KV、联合 softmax 和重复索引的含义明确；普通、packed、probe 调度有实际差异。但 combine 的存在性及 probe 后互斥路径靠正文决定。 |
| [gdn_prefill_bf16_state_bf16](../schema_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/root.yaml) | 保留意见 | 逐 token Contract 与分块求解、状态链、分段两遍的 Design 分开；初末状态及输出副作用有说明。关闭归一化时，正文取消仍被 flow 引用的 normalize 及其完成边。 |

## 需要解决的表达问题

1. [条件流程绕过现行规范](issues/extract_workflow/conditional-flow-topology.json)。GDN 的 `normalize`、FMHA 的 `segment`、MoE 的大小准备链，以及 sparse 的 `packed/mma` 都是实际提交集合的变化。现行 flow 规则给出固定提交和依赖，产物另外在正文中要求删去条目或改变前驱。这是未经定义的语义，不能按已提交动作的 no-op 解释。优先以 GDN 的开关和 sparse 的 probe 后选择核对现行 dispatch/phase/launch/handoff 规则能否合法表达；确有不适配、缺失或冲突时，再给出具体条款和反例提报维护。
2. [角色成员范围的工具限制与绕行写法](issues/archive/schema_template/symbolic-member-bounds.json)。MoE `ws_fused_ws` 将 Epilogue 登记为 `warp[3:11]`，正文却说实际只实例化 `warp[3:3+EPI]`；gather 同样需要 `Nt/32`。这违反 member 表达真实参与集合的要求。原 P5 worker 已提报数字端点限制，初验沿用其稳定 ID 登记；提报本身并不授权采用最大范围加正文取消的办法。后续 2.2.6 已统一成员端点的规范、关联检查和有限展开，旧产物尚待修订，当前为部分采纳。

初验时两项均保持待决；后续处理见文末。没有把产物中的绕行写法追加为既成规范，也没有把未修订的旧产物记录为已解决。

## 索引勘误

原 MoE dispatch 末尾仍写“表内计划文件由后续 P5 生成，当前缺文件不代表已有完整 Design”，原稀疏注意力索引写“派生文件待 P5 生成”。这是过期的阶段文字：两目录实际分别已发布 60 和 3 个节点。已补充 P4/P6 的索引交付规则；当前阅读入口已改为 [MoE 索引勘误页](../schema_wiki/errata/nvfp4_fused_moe.md)和[稀疏注意力索引勘误页](../schema_wiki/errata/dual_cache_latent_bf16.md)，沿用 [published-dispatch-draft-status](issues/archive/extract_workflow/published-dispatch-draft-status.json) 记录。

两页保留原表格与覆盖事实，只移除过期阶段措辞并将节点链接指向原 YAML。原 dispatch、YAML、发布回执及审阅身份均保留，历史交付链接不变。本次文案勘误不处理上述条件流程与成员表达保留意见，也不授予旧产物新的语义批准。

## 对 workflow 的判断

六类算子的主要计算都找到了现有分区中的合理位置，本轮没有发现必须改写任务语义才能表达的整类算子。短板集中在参数化设计改变提交集合或参与成员时，Agent 自行用正文改变结构解释，而原审阅没有拦下。先按现行规范复核和改正产物；只有能指出实际不适配、缺失或冲突，才提出语言或工具修订，不能从“Agent 这样写了”倒推出规范应当接受。

MoE 的 60 个节点包含独立 FC1/FC2 执行族组合、融合与 split-K/last-arriver 的物化差异、BF16 路径及零验证路径；tile、环深等已有不少参数化。节点数本身不足以证明过度拆分，本轮不据此要求合并，也不宣称 60 已是最少数量。合法表达确定后再评估重复组合和 root/通用 WS 路径的组织。

本轮还读取了六个原始交付 run、GEMM 校正版及六个接续前任的 `candidates`，共 13 次查询：MoE 有上述成员范围候选，另外 12 次为空。已归档的比较器、P4 布局、P5 定位、summary 与 P6 tree 证据问题继续沿用原决定。产物披露的源码覆盖/同步缺陷保留为来源限制，本轮未独立验证，不把它们误报成 schema 语言无法表达。

维护入口补充了逐 run（含接续前任）核对候选的收口职责。根据用户强调的长期原则，worker/reviewer 的实际共用提示、提取总览和 Supervisor 角色说明均明确了“先遵循 schema_docs，发现实际冲突或缺项先提报，不自行发明字段或语义”；reviewer 的批准同时要求源码忠实与规范一致。上述改动涉及初验、issue、提示和提取文档，没有新增语言字段或修改检查器，尚无修复后真实提取结果。

初验阶段验证：`python3 -B -m unittest discover -s schema_extract/tools/tests` 的当时 29 项提取回归通过；正式 `issues` 入口确认本轮登记的两个待决项和一项部分采纳决定，111 个文档本地链接均可定位。回归验证提示相关代码仍可运行，不证明模型后续必然遵规，也不改变当前产物的初验保留意见。

## check_yaml 是否实际执行

六个原始交付的最终 P6 均有 Runner 生成的 `check.json`，记录调用冻结 `schema_template/check_yaml.py --skip-hardware`、实际 `returncode=0` 和 `exit_code=0`，标准输出分别确认 3/60/8/1/3/2 个 YAML。各记录与对应 `validated.json.check` 相同；批准回执摘要匹配；P6 所查 workspace 与发布文件的逐文件 SHA-256 一致。

| 原始交付 | P6 节点数 | 退出码 | Runner 原始记录 |
| --- | ---: | ---: | --- |
| fp8_scaled_grouped_gemm | 3 | 0 | [check.json](../.schema_extract/v2/runs/4aac1ea4-94fa-4b4c-93ca-d48a8eb08996/attempts/63c946a5-a587-4759-8086-cb22c37715b6/check.json) |
| nvfp4_fused_moe | 60 | 0 | [check.json](../.schema_extract/v2/runs/3563c3b7-2b7c-4967-9d41-9c2042e53d2c/attempts/afccb5c7-c1be-4cb4-9081-01941bf81c26/check.json) |
| fmha_mxfp8_qkv | 8 | 0 | [check.json](../.schema_extract/v2/runs/acb68909-aaaa-4bc2-8b16-94904752af8e/attempts/e6471950-d227-4081-82d1-164d77998c9f/check.json) |
| mla_expanded_fp8 | 1 | 0 | [check.json](../.schema_extract/v2/runs/e3f7704f-061f-48ff-8131-cc317afe6003/attempts/3f0c08ba-ac6f-4535-90b8-5dfe5540bab0/check.json) |
| dual_cache_latent_bf16 | 3 | 0 | [check.json](../.schema_extract/v2/runs/d5a73c01-7c3a-45b2-a3ce-0d5eec634808/attempts/eb05f840-45f0-48b3-b7dc-6ae0dde84998/check.json) |
| gdn_prefill_bf16_state_bf16 | 2 | 0 | [check.json](../.schema_extract/v2/runs/97902388-eda0-4cbd-b8a6-ba71440e86ab/attempts/327bac93-0aa8-410f-b32b-fd32b199b1a1/check.json) |

进一步读取原批次 P3/P5/P6 的已接受 `complete/approve` 回执，共 162 条，每条都有成功的冻结检查记录，未发现漏记或非零退出。GEMM 2.2.4 校正版另有 8 条，最终 3 节点也检查成功。P1/P2/P4 的未完成设计使用阶段门禁，不能把这些阶段没有完整 YAML 检查视为漏跑。

原批次执行器是 2.2.2，校正版是 2.2.4；调查时分别比较快照中的 27 个检查器 Python 文件和全部现行 schema_docs，均与当时工作树一致。这一比较发生在后续 2.2.5 新增单文件视图之前，说明所指出的既有规则在原检查时已经存在。MoE 等 reviewer 回执的 `checks: []` 表示模型没有另外亲自执行命令；实际 Runner 调用记录不在该列表内。

上述链接属于保留在本地、Git 忽略的运行证据，公开仓库不包含这些目录；算子文件及发布回执仍可从已有交付导航定位。

## 为什么仍然通过

本次发现的主要问题发生在已有字段的文字内容里，尚未发现这些案例靠新增未知固定字段通过检查。`rules/values.py::record` 拒绝未知字段，`::text` 仅检查非空文本；`rules/schedule.py` 根据结构中已声明的 phase 和事件检查引用，不解释 describe 中声称它们不存在的条件。MoE 的数字成员范围本身合法，检查器也不会从 thesis 中求出 EPI 后纠正成员集合。

初验时以原始字节建立临时副本，用当时检查器做以下对照，早于 2.2.6 的成员端点支持；GDN 保留两个节点，成员对照只取 MoE 的 root 与 ws_fused_ws 两节点。所有命令均显式使用 `--skip-hardware`，原发布文件未改写。

| 副本中的内容 | 实际结果 |
| --- | --- |
| GDN 原文件，含正文中的条件取消说明 | 0，通过 |
| 只给 normalize 的 launch 新增 `when` 字段 | 1，`field not allowed` |
| 只将 prep 的前驱改为 `missing_phase.kernel_done` | 1，`unknown event phase` |
| 结构不变，只把 normalize.describe 改成“此阶段及完成事件对任何输入都不存在” | 0，仍通过 |
| MoE 两节点的原成员声明与最大范围说明 | 0，通过 |
| 只将一处 `warp[3:11]` 改为 `warp[3:3+EPI]` | 1，成员语法不接受该端点 |

这些对照说明固定字段和已实现的引用门禁实际生效；它们并未覆盖文本与结构的语义一致性。`--skip-hardware` 没有豁免上述字段和引用规则。因而当前结论是：机械检查执行了，未由工具解释的规范要求在原语义审阅中漏检；不能由退出 0 推出已经遵循全部 docs。

模板的校验边界和 Supervisor 的证据阅读说明已补充这一具体区别，当时两个待决 issue 追加了对照证据。没有通过放宽字段白名单、给文字增加隐式执行语义或关键词拦截来处理问题。

## Reviewer 机制修复

按用户选定的方法，2.2.5 已将[当前文档验收](REVIEW.md)接入真实提示、各阶段范围和批准门槛：先审当前累计完成产物，移除前序批准与交接输入，要求独立理解、分层、全文一致和内容必要性四项结论。规范表达问题可凭当前产物证据重开只读上游；单文件阅读由正式工具展开共享 Contract，创作侧仍保留唯一 Contract。

[reviewer-document-integrity](issues/archive/extract_workflow/reviewer-document-integrity.json) 已解决归档。验证包含 42 项提取测试、148 项模板测试、10 个真实语义回归样例，以及使用生产提示对原 GDN 两节点执行的隔离 P6；详见[验证记录](review_cases/RESULTS.md)。GDN 再审要求重开 P3，未给旧产物新的批准。

本节记录 2.2.5 的机制修复；前述四类产物的表达保留意见继续有效。原始发布、冻结快照和回执保持可审计的历史身份。

## 2.2.6 的长期复核与处理

[逐项分类](issues/TRIAGE.md)保留有效的 Reviewer、来源缺口和证据职责修复，优先处理会改变产物含义或造成重复劳动的问题。P4 已将计划表内容与说明段排版解耦，移除专用异常与修复分支；成员路径已共用整数端点解析，保留实际成员与事件语义。MLA 的可选校注文件也已纳入新快照的既有收集入口，纠正旧批次曾包含校注的错误归档判断；原绑定及旧快照不变。对应验收为最终 47 项提取回归、两份原失败稿的只读重放、当前 MLA 输入收集核验和 152 项模板回归。

条件流程的方案采用已有 dispatch：实际提交集合、交接对象或完成关系不同才分节点，数量变化仍参数化；GDN 与 sparse probe 的具体情形及来源写在分类文档。conditional-flow-topology 继续待决，symbolic-member-bounds 因旧 MoE 未改稿而部分采纳。此次没有重新审阅或发布历史四类算子，初验保留意见不因框架修复撤销。

## 2.2.7 的 GDN 修订验收

2026-11-04，GDN 在冻结 2.2.7 下完成 6/6 节点及 P6 独立终审，正式 export 后由 Supervisor 验收；[当前六节点索引](../schema_wiki/revisions/2.2.7/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/dispatch.md)成为推荐入口。节点为 `root`、`cached_ordered`、`direct_qk`、`direct_qk_ordered`、`segmented`、`segmented_ordered`。该修订版的表达问题已收口，原始两节点及其初验结论仍保留历史身份。

独立 P1 依据动态输入构造确认 `norm=false`，`cu_seqlens`、`chunk_indices`、`chunk_offsets` 均实际提供；原稿中的 normalize 提交不属于本任务域。实际 nullable 仅为 `core_attn_out`、`FinalState`、`CoreWritten`，reference 的额外可选接口不扩张本任务。新 Design 按缓存/直接 QK、分段与 Perm 存在性完整分开提交和交接；图像读者数 `(HV/H)*(128/VTT)`、系数读者数 `128/VTT` 已修正，`KS`、`QS`、`DU0` 已登记。最终 P6 将 `segmented_ordered` 的排序合入 `stage_inputs`，消除了源码没有的额外 CTA 串行等待。

交付 run 为 `45f9bb51-77f3-4285-b7a7-251cf2744fa1`，最终 checkpoint 为 `349f0fdbad54effd79e2557888dffcc88de418c142438fd2989db2b4937d0a2d`，P6 reviewer 为 `1283b632-a802-4625-b666-a0eb3dce449d`；[发布回执](../schema_wiki/revisions/2.2.7/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/.extraction/receipt.json)绑定该身份。现存[最终一次 P6 调用审计](../.schema_extract/trials/six-families/post-batch/artifact-repairs-2.2.7/gdn-quota-restored-final-audit-20261104T125856480549Z.json)为 `verified`，`errors`、`unavailable` 均为空；[交付核验事件](../.schema_extract/trials/six-families/post-batch/artifact-repairs-2.2.7/quota-restored-recovery.jsonl)中的 `gdn_export_delivery_verified` 确认新目录 8 个文件一致、原目录 4 个文件 SHA 未变。审计范围是最终这一次调用，完整身份及 export 状态见[试用记录](TRIAL_REPORT.md)。

源码与输入角色绑定不变，非连续 core 缺少 copyback、mini-chunk 读取退休、跨 stream/重入及外部比较器缺口继续披露；本次没有修 kernel 或进行 GPU 验证。FMHA、MoE 的表达修订继续推进，Sparse 保留上游源缺陷阻塞，因此 `conditional-flow-topology` 整体仍待决，其他旧产物的保留意见不变。

以上保留该验收时点；后续 2.2.9 已澄清源缺陷抽象边界，Sparse 的低层槽位问题现已[记录归档](issues/archive/kernel_wiki/sparse-tc-control-layout-overlap.json)，不再作为提取阻塞，实际 Design/P6 仍须独立完成和审阅。
