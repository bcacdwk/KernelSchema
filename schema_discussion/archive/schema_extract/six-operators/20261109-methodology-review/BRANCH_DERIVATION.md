# 分支判断、派生、继承与复用研究（2026-11-09）

**应把节点划分、公共定义复用和修订传播作为同一个研究专题，但目前没有证据证明它们具有同一根因。** 现行方法要求保留真实执行差异，已经允许数量参数化和 Design 整区共享；费用同时受范围扩大、反复创作/审阅、root 漏项及保守的祖先依赖传播影响。减少 YAML 数量、减少重复正文、减少模型输入和减少重新批准，是四个不同目标。

本文为后续讨论和实现提供依据，不判定 220 个节点已经最少，也不据成本直接建议合并。与 [SCHEMA_VERBOSITY.md](SCHEMA_VERBOSITY.md) 的交叉处是：哪些语义必须落在结构中、哪些定义可以共享、哪些局部说明会随修订传播；字段长度、文件内 alias、逐稿文字增减由该报告展开。root 独立的数值/来源/寿命漏项不能自动归因于节点多。

## 1. 范围、进度与费用

费用沿用[总复盘](../20261109-usage-review/README.md)与[逐调用 CSV](../20261109-usage-review/attempts.csv)：25 条正式 run，按 `(origin_run_id, attempt_id)` 排除 269 条继承重复记录，得到 1,041 次唯一尝试；其中 1,040 次有 invocation，1 次仅准备未派发。它们覆盖现存可追溯正式运行，**不含**开发、Supervisor、独立子代理、review-eval 及未报告用量。价格为用户给定的 gpt-6-astra、30 分钟缓存写入情景折算，不是账单；失败而没有用量记录不能证明没有提供者费用。

| 算子 | 初始交付节点 | 暂停时批准 / 当前计划 | 暂停成果 | 全历史记录 USD | 采用 2.5.0 系列的记录 USD |
| --- | ---: | ---: | --- | ---: | ---: |
| GDN | 2 | 6 / 6 | 2.2.7 已交付 | 363.76 | 0.00 |
| MLA | 1 | 2 / 2 | 2.5.0 已交付 | 122.99 | 84.06 |
| GEMM | 3 | 7 / 7 | 2.5.0 已交付 | 376.42 | 254.78 |
| Sparse | 3 | 8 / 8 | 2.5.0 已交付 | 483.66 | 228.85 |
| FMHA | 8 | 26 / 43 | 本轮 P6 / 发布未完成 | 1,838.83 | 28.53 |
| MoE | 60 | 96 / 154 | 本轮 P6 / 发布未完成 | 3,208.42 | 1,902.20 |
| 合计 | **77** | **145 / 220** | 四项交付、两项暂停 | **6,394.08** | **2,498.42** |

初始数量由[结果归档](../../../../schema_wiki/archive/six-operators/README.md)复核；当前数量来自[暂停清单](../20261109-pause/nodes.csv)。CSV 共 233 行，须排除 Sparse 的 6 个及 FMHA 的 7 个退役历史节点，不能把 233 当作当前计划。FMHA 尚有 4 个完整待审候选、13 个需继续写作/返工；MoE 尚有 9 个完整候选、1 个需继续写作/返工、5 个失败但无已记录创作结果、43 个未开始。原有八节点/六十节点交付不能代替这次新计划的终审。

| 费用范围 | P5 尝试 | P5 USD | 占该范围总费用 | 解释 |
| --- | ---: | ---: | ---: | --- |
| 全部现存正式运行 | 781 | 5,189.02 | 81.2% | 含不同版本的创作、维护、复审、失败和返工 |
| 2.5.0 系列 | 299 | 2,187.41 | 87.6% | 164 Worker、135 Reviewer；不是六算子从零重跑 |
| 其中 MoE 2.5.0 P5 | 248 | 1,863.86 | — | 约占本版本系列 P5 费用的 85.2% |

FMHA 的最新版 $28.53 仅是 **5 次有限维护尝试**，不是完成整个 FMHA 的价格。版本按每次 attempt 的冻结输入/semantic revision 判定，不能按 run 当前版本追溯套用。77→220 是不同交付时点和范围的净变化，既包含覆盖修订，也包含重新划分；它不是“同一正确设计被无理由复制了 143 份”的证据。

当前已批准的 139 个非 root raw 中，MLA 有 1 个节点引用 1 区，MoE 有 24 个节点引用 45 区，另四项没有显式 `from_parent`；合计 **25 个节点、46 个区域**。这是按 YAML 结构解析的快照，不是字符串计数，不含待审或未开始节点。无跨节点引用不等于无文件内 anchor/alias 复用，更不证明本地三区域可无条件共享。统计和精确样本 ID 保存在唯一支撑文件 [BRANCH_EVIDENCE.json](BRANCH_EVIDENCE.json)。

## 2. 现行判断方法：先辨真实差异，再选父与共享

以下是**规范要求**，不是声称现有模型已经完全遵循。

| 判断 | 现行要求与定位 |
| --- | --- |
| 这只是参数变化吗？ | 同一套合法参数、公式和索引能完整表达时保留一个节点；不能按每个 shape、tile、成员数复制设计。[0.common.md:160–164](../../../../schema_docs/0.common.md#参数变化与结构变化) |
| 需要拆节点吗？ | 实际提交集合、producer、物化或事件关系不同，且既有规则不能表达时，拆成各自完整的 dispatch；不能先写结构并集再以正文删边。函数、文件、kernel 的数量也不等于节点数。[P2:17–22](../../../phases/02_dispatch_draft.md#worker可写范围与操作) |
| 哪条做 root？ | 从真实路径选覆盖主要 workload、较通用且便于派生的实际 plan；共享 Contract 覆盖任务域，root Design 只负责自己范围；不另造全域 baseline。[P2:18](../../../phases/02_dispatch_draft.md) |
| 如何选 parent？ | 直接设计来源，须设计接近且有证据；所有节点最终连到 root。它不是路由条件包含关系，也不会自动继承内容；不为压缩自动换父。[P4:17–20](../../../phases/04_dispatch_plan.md) |
| 如何写 when？ | 每个节点，包括 root/中间节点，都在完整 Contract 域独立给出充要条件，互斥并覆盖实际范围；不叠加祖先条件。运行时 probe 结果可作为事实，但实际已提交的 probe 必须保留。[0.common.md:154–164](../../../../schema_docs/0.common.md#选择条件与索引) |
| 什么能继承？ | 三个 Design 区域各自选完整 local 或恰好 `{from_parent: 直接父ID}`；沿直接父同名区域追到实际 provider。禁止局部 patch、混合覆盖、跳祖先、兄弟/跨目录引用；root 三区域均 local。[0.common.md:125–144](../../../../schema_docs/0.common.md#design-整区引用) |
| 什么须本地确定？ | metadata、when、version、realization 不随引用继承；共享定义仍在子节点 Contract、参数、名称、phase/worker、资源和事件环境下绑定。realization 仅保留 Design/Contract 尚未表达而必要的实现交接，必要条目不能因“与父相同”省略。[P5:29–32](../../../phases/05_branch.md)、[6.realization.md:7–32](../../../../schema_docs/6.realization.md) |

关键规则短片段是“**同一套合法的参数、公式与索引规则能完整表达实际设计**”与“**引入的定义在使用节点的完整上下文中绑定**”。因此，“源码有 if”不能单独证明要分支，“文字一样”也不能单独证明可以共享。

**实现做了什么、没有做什么。** [artifacts.validate_plan():274](../../../tools/artifacts.py#L274) 核对七字段清单、唯一 root、文件身份、父存在、无环、索引一致；[rules.structure.validate_task():213](../../../../schema_template/tools/rules/structure.py#L213) 把 `when` 交给 [values.text:22](../../../../schema_template/tools/rules/values.py#L22) 检查非空。它们不自动证明自然语言条件互斥、全域覆盖、实际路由可达或节点数最小。这些仍由 P2/P4/P5/P6 的来源审阅承担。`from_parent` 的形状、直接父与环由 [design_resolution.resolve_documents():151](../../../../schema_template/tools/design_resolution.py#L151) 解析；成功获得 effective 视图仍不等于完整检查或语义批准。

## 3. 真实案例能支持怎样的分组

### MLA：节点增加与共享可以同时发生

[两节点 dispatch](../../../../schema_wiki/archive/six-operators/revisions/2.5.0/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/dispatch.md) 区分持久补领与一次 bootstrap：root 有 Queue 服务、终止发布、跨工作 Q 预取；bootstrap 每 CTA 一份工作，取消这些接续。不是只因为 CTA 数量不同就拆分。host 原句是 `grid_items = (items * 2 <= nsm_l * 3) ? items : nsm_l`，见 [solution.py:2007–2009](../../../../kernel_wiki/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/candidate/solution.py#L2007)，产物固定来源 ref 为 `228934a2253bde4d6f0c4463dd77aeada1bf07a8`。

[root.yaml:14](../../../../schema_wiki/archive/six-operators/revisions/2.5.0/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/root.yaml#L14) 是 `2 * WorkCount > 3 * LaunchSMCount`；[bootstrap.yaml:17–22](../../../../schema_wiki/archive/six-operators/revisions/2.5.0/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/bootstrap.yaml#L17) 是 `<=`，并引用 root 的 `math_reshape`。父子条件互斥；若把父条件当子前提，子将变成空域。bootstrap 的 schedule/pipeline 为 local，realization 在第 310 行本地保留。排序、K/V 三/四槽、causal/sink/LSE 等变化仍在节点内参数化。这是“保留两个真实组织，同时复用一个完整区域”的已交付实例。

### Sparse：已提交但结果不可用，不能合成“未提交”

[八节点 dispatch](../../../../schema_wiki/archive/six-operators/revisions/2.5.0/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/dispatch.md) 中，root 是 split attention→FP32 combine；`mma_direct` 不物化 split、在 attention 内归一化；`probe_serial_mma` 比 `mma_direct` 多一次真实 probe；hidden 两路保留 packed 前缀、side-stream probe 和 join，再按实际统计选择余段。

索引明确写道：“**已提交 probe 的工作和接续不因此消失**。”`mma_direct.when` 的 gate 不可用分支在提交前返回；`probe_serial_mma.when` 则要求 `probe_begin returns true and submits a probe`，即使有界轮询后的结果不可用，也不能抹去探测。可对照 [dsv4_kernel.cu:6140](../../../../kernel_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/candidate/dsv4_kernel.cu#L6140) 的 `dual_dispatch_hidden` 与第 6435 行调用；产物来源 ref 为 `6e4ca91bd91d998cac99e8dadd83838e153c679d`。split 数、head 分块、DQ、ring 深度和 combine 线程数仍参数化；纯录图也没有被单独造为空计算节点。

### FMHA / MoE：组合维度解释规模，不能证明当前数量最优

[FMHA 当前计划][fmha-plan] 的 43 个节点可按实际 ID 核对为：7 个融合族节点、4 个 mma.sync split 节点、24 个 D128 tcgen05 split 节点、8 个 D256 tcgen05 split 节点。D128 的 24 来自三种 scan × 两种搬运 × 两种部分和写回 × 两种归并；D256 为两种 scan × 两种搬运 × 两种归并。`tc_fused` 却将 D128/D256 参数化在同一节点，说明它并非简单按维度逐一复制。

计划的 `segmented` 相对 `flat` 明确“增加 segmax 物化及一次提交”；`cp` 相对 `tma` 改搬运等待与退休组织；`wide` 改归并工作组和部分和共享。它们提供了需要区分的依据，但尚未逐一证明每个组合都不能由已有参数规则统一，也没有最小节点数证明。43 节点的最大 parent 深度为 5；仅 26 个获批，不能把计划行当作完整 Design 的验收结论。

[MoE 当前计划][moe-plan] 将 ABI 复用/重建、初始化/零扫描、独立验证/再次验证、Small/大准备、FC1/FC2 组织和描述符回落组合起来。`fused_small` 用一个 `k_prepare_small` 替换大准备链；`double_verify_root` 保留独立验证图后完整主干中的第二次验证，两次提交不能因计算相似而删除。与此同时，`fused_big_cp` 已把 cfg2=23 和 44 的描述符失败汇合到同一有效 cp.async 组织，并未每个原配置保留一份。154 节点最大 parent 深度为 8，深度本身不是质量错误，但会增加父先子后调度和祖先验证的路径长度。

MoE 的 [fused_small_flat.yaml:38–39][moe-flat] 引用 `fused_small_mono` 的 math/schedule，保留 local pipeline 与第 470 行的 realization；`no_scan_fused_small_mono` 则共享 math/pipeline、local schedule。三个区域可以独立共享，但一个 parent 未必同时是三区域最适合的公共来源；现规则选择直接设计来源，尚无证据支持为短文本重新挂树。

### 不能把所有条件差异都推向新节点

[conditional-flow-topology 历史 issue](../../../issues/archive/extract_workflow/conditional-flow-topology.json) 记录旧 GDN normalize、Sparse probe/combine、FMHA TwoPass、MoE Small/gather 用正文取消 launch/handoff/依赖的问题。它证明部分旧表达越过既有语言语义，**不证明语言必然需要新字段，也不证明所有增长都由拆分规则造成**。

同一 issue 的 MoE hoist 反例已有现行表达：[tile_pipeline:416–441](../../../../schema_docs/5.design.tile_pipeline.md#条件等待示例) 用 `repeat: {q: [0, "min(1,t)*(1-Hoist)"]}` 与显式 `waits_for`，在低 fill 时增加 release→wait.done→prepare，在高 fill 时不增加该等待，两者保留 release→首个 MMA。无需另拆两个 fill 节点。另一方面，本 role 原语的数值、完成和可见性保证可以留在动作契约；不能把必要契约一律视为“正文膨胀”，也不能在自然语言里悄悄改另一 role 的命名事件前驱。相关文字演变由 [SCHEMA_VERBOSITY.md](SCHEMA_VERBOSITY.md) 追踪。

## 4. root 漏项和分支来回修订：已有数据

[root 复盘](../20261109-usage-review/ROOT_ANALYSIS.md) 的近期 P3 首次批准闭环通常为 $13–16；GDN 含一次语义退修为 $24.79。该样本排除了 P4/P5/P6，足以说明单 root 也会产生独立成本。首审后重开至少存在以下四例；精确 run、attempt 见本节末索引和支撑 JSON。

| 触发发现 | root 问题及局部回修 | 实际传播证据 |
| --- | --- | --- |
| GDN，P5 `cached_ordered` → P3 | Q/K 图像 reader 数；P3 回修 $13.81 | 已有六节点计划重新定稿，`invalidated_nodes` 包含全 6 节点；不能把列表中每个节点都称为“原已批准后重审” |
| MLA，P4 → P3 | 首轮 reuse 边界：`Next(w+2)/Item(w+1)` 改为 `Next(w+1)/Item(w)`；P3 $12.97 | `previous_plan` 为空，随后安装 root/bootstrap；这次不是已完成分支树的全树返工 |
| GEMM，**P6** → P3 | TMEM 行组地址的来源限制漏记；增加三行说明及版本；P3 $11.72 | 请求 `affected_nodes` 为 6 项，后续计划实际失效 7 项，包含 `mma_sync`；其后 6 个既有获批分支再走 P5 |
| FMHA，P5 `tcg_d128_fscan_cp_xpose_wide` → P1 | O 比较的 actual 相对基准/误差比漏记；P1 $10.44，后继 P3 未改字节仍花 $15.03 | 43→43 的计划重新定稿，全 43 项失效；P1–P4 共 8 次、$57.66，再继续分支 |

GEMM 发现阶段经原 [attempt][gemm-trigger-attempt] 与 [receipt][gemm-trigger] 双重核对为 **P6**；MLA 才是 P4。请求影响集合和 Runner 后续失效集合也不同，二者不能混写。GEMM receipt 的短理由是“**P3仅补齐root流水线中的必要说明，随后复核相关派生稿**”。实际费用如下，排除发现问题的首次 P6 调用：

| GEMM 本次重开之后直到最终批准 | 尝试 | USD | CLI 分钟 |
| --- | ---: | ---: | ---: |
| P3 修 root | 2 | 11.72 | 10.30 |
| P4 再定稿 | 2 | 13.92 | 13.71 |
| P5 六个旧分支 | 14 | 92.90 | 70.76 |
| 最终 P6 | 1 | 15.31 | 8.17 |
| 合计 | **19** | **133.85** | **102.94** |

P5 是 10 次 Reviewer、4 次 Worker：`v0_recovery`、`wide`、`wide_recovery`、`wide_recovery_chain` 各经历 revise→worker→approve；`mma_sync` 和 `wide_recovery_full` 直接再审。前四者的本地 pipeline 没有自动得到父补充的来源说明。[v0_recovery 回执][gemm-child-revise] 明言“**其三区域本地定义不会继承 root.yaml:261-263 的说明**”。所以这里同时存在首次来源漏项、全祖先传播和重复 local 内容维护，不能把 $133.85 全算成可删除的无效审阅，也不能宣称改用引用就能省掉它。

FMHA 的事件还有更大的后续观察集合：在触发前，当前计划内有 40 个节点曾取得 accepted P5 approve；事件后其中 36 个节点在 generation≥1、非 maintenance 的普通 P5 中出现 **110 次尝试、$735.67、9.11 CLI 小时**，全部仍采用 **2.2.7**。它包括 74 Worker、36 Reviewer，以及 49 条无 response decision 的尝试，不能把 110 次理解为 110 次有效批准。这个按时间/节点选出的集合说明旧节点确实被反复处理，但混有失败、重写和继续工作，**不是隔离出来的“Contract 一处修正的纯增量价格”**，更不是当前 2.5.0 保稿机制的性能实验。

事件身份：

| 算子 | 原 run | 触发 attempt / 后续关键 attempt |
| --- | --- | --- |
| GDN | `45f9bb51-77f3-4285-b7a7-251cf2744fa1` | `27c46219-0d41-4bfd-9bf3-bf6ac11feb34`；P3 `30816cbf-5f8c-486e-8964-0ccaac177270`、`d54a4fb8-b516-4f10-becf-597ef3b05f14` |
| MLA | `531567f1-bcae-45ad-acf6-fd674c055fdc` | `a1fcaf4d-ce92-464f-8284-fc3c866d4128`；P3 `1da6e4cd-b792-4f35-9d71-a5d1048b1c34`、`226f1b78-cd69-4c04-87fa-d6ac512a9a46` |
| GEMM | `d4e41124-e29c-458c-a1ce-2a9bf87ee0a8` | `7599acb3-e338-4c7f-9352-7d19ead4aba6`；子退修 `0e9a4654-8610-4c39-a662-133412f0ec3a`；终审 `29065f5c-c275-4f77-8cdb-ad022b331cb7` |
| FMHA | `45ea6608-66b2-4915-b47d-59d2c115602e` | `379c63c8-496c-49ea-ac24-3e66368021dd`；P3 未改字节 Worker `e856ac77-91b2-43ed-a598-3e5c19442da4`、Reviewer `8724fd8b-a13e-406e-83f7-e6ea128e1295` |

不能只按 `node=root` 求 root 价格，因为 P2/P4/P6 也用此标签。也不能把所有 revise/reopen 标签费用相加当作返工总额：一个 reopen 后的大部分费用会出现在后续 complete/approve 上。

## 5. 生命周期、成本触发与不可破坏的约束

下面把**当前实现**与上面的历史运行区分开。主要代码按工作树 `3ced7d32d7d7dc4af172f05d0d9a673fae177902` 核对；2.5.1 只裁剪 P1/P3 提示中的调度历史并标明已嵌入文档，未改变分支状态机，未自动升级旧 run。完整设计依据见 [2.5.0 整区派生设计](../../designs/2.5.0-design-region-derivation.md)。

| 触发 | 当前行为与成本含义 | 关键实现入口 |
| --- | --- | --- |
| P2 草案 / P4 定稿 | P2 选择真实 root、列划分依据；P4 固定七字段计划。每个新节点仍形成独立调度/审阅单位，不能仅以引用代替有效差异。计划语义签名排除回执局部 evidence 编号，其余字段变化可触发影响。 | [artifacts.validate_plan():274](../../../tools/artifacts.py#L274)、[stages.node_signature():139](../../../tools/stages.py#L139)、[plan_impact():296](../../../tools/stages.py#L296) |
| 首次 P5 / 既有稿恢复 | 直接父有效批准后才调度。确无己稿时建立版本1、空when、三区域引用模板，删除父Contract/realization；只要存在草稿/旧批准/可恢复稿，保留自己的raw，仅刷新只读祖先。完整候选可交Reviewer。 | [stages.next_calls():182](../../../tools/stages.py#L182)、[Store.start_node():620](../../../tools/store.py#L620)、[P5:9–27](../../../phases/05_branch.md) |
| 独立创作 / 审阅 / 合入 | P5仅写本节点；审阅须有 differences、inherited_design、walkthrough。父通用推导可按当前定义读一次，但每个子选择域、绑定、跨区接口、资源与realization仍审。接收再次核对身份、权限、证据、依赖并检查。 | [receipts.required_artifact_evidence():53](../../../tools/receipts.py#L53)、[_validate_derivation_anchors():290](../../../tools/receipts.py#L290)、[runner.accept_result():567](../../../tools/runner.py#L567) |
| 父内容变化 / 上游重开 | 原始子版本不因父变化伪增；generation/有效批准失效，完整旧稿通常进入reviewer，待创作稿进入worker。P1/P2/P3重开还失效后继阶段，重新定稿按root变化计算后代。raw保留与批准有效是两件事。 | [stages.invalidate_node():274](../../../tools/stages.py#L274)、[runner._begin_reopen():420](../../../tools/runner.py#L420)、[_plan_workspace():392](../../../tools/runner.py#L392)、[maintenance._reconsider_dependents():372](../../../tools/maintenance.py#L372) |
| 实际非root provider定义错误 | 仅普通normal P5可请求当前批准严格祖先中的实际provider，提交时重新解析来源。先保留请求者合法稿；provider新generation必须worker→独立reviewer，后代保守失效，兄弟/P1–P4保留。仅在子上下文不适用时应在子节点local修订。 | [artifacts.design_context():75](../../../tools/artifacts.py#L75)、[receipts.provider_request_details():263](../../../tools/receipts.py#L263)、[runner._begin_provider_reopen():482](../../../tools/runner.py#L482) |
| P4换父 | 影响沿旧、新树后代闭包传播；保存old/new parent、旧checkpoint、计划修订和原因。worker显式选择引用新父或从旧effective落为完整local；解析器不自动retarget。即使旧稿完整，也须新generation独立批准。 | [stages.install_plan():310](../../../tools/stages.py#L310)、[needs_authoring():269](../../../tools/stages.py#L269)、[_approve():347](../../../tools/stages.py#L347)、[P4:38](../../../phases/04_dispatch_plan.md) |
| 在途结果 / 串行合入 | generation、节点签名、规则、草稿或祖先摘要漂移使旧结果stale；不自动采纳晚到稿。仅替换候选自己的raw。检查集为当前有效节点＋候选＋祖先；父候选字节改变时先排除旧后代，记录superseded_dependents，避免旧稿卡死无关候选。 | [Store.validate_attempt():490](../../../tools/store.py#L490)、[merged_workspace():670](../../../tools/store.py#L670)、[runner._merge_check():538](../../../tools/runner.py#L538) |
| P6 / 规则接续 / 发布 | P6和发布要求整棵当前计划完整有效；子集检查不批准遗漏节点。旧规则按冻结context执行；跨语义版本的keep仅在独立证据满足内容、依赖、要求覆盖后采纳，不能靠字节相同沿用批准。maintenance字段范围不自动扩大。 | [stages.final_eligible():159](../../../tools/stages.py#L159)、[revisions.resolve_context():147](../../../tools/revisions.py#L147)、[_adoptable():403](../../../tools/revisions.py#L403)、[verify_effective_approval():548](../../../tools/revisions.py#L548)、[maintenance._patch_fields():347](../../../tools/maintenance.py#L347)、[P6](../../../phases/06_final.md) |

有两个容易混淆的依赖集合：

- **内容来源集合**：root、自身、实际引用链中的中间节点与provider；[ResolvedTask.source_manifest()/effective_identity():40–51](../../../../schema_template/tools/design_resolution.py#L40) 用原始文件SHA和来源链构成身份，即使展开文字一样，来源链改变也可改变有效身份。
- **审批依赖集合**：仍保守绑定**全部祖先文件**，包括没有被显式引用的祖先。[Store.verify_node():457–486](../../../tools/store.py#L457) 逐祖先检查当前批准和摘要。因此“本区没有from_parent”或“child字节未变”不能作为保留旧批准的充分理由。

来源、权限和批准的不变量必须保留：当前consumer与直接父的manifest并集、每个引用链的raw引用行/provider定义行都可追溯；worker不修改祖先、Contract或计划；回执不能脱离冻结source/rules/executor、当前generation、计划签名和依赖摘要；同一provider＋稳定issue＋固定源码证据身份不能靠改文案/升版本重买重开机会；原预算、计数、退修、历史批准和已用决定不重置。P6也没有普通P5的provider重开入口。[evidence_identity():47](../../../tools/receipts.py#L47) 明确固定证据只取 source ID/path/SHA/行范围。

这些机制的存在不是重构理由。它们也有成本：逐节点新会话、完整适用性审阅、祖先验证、raw/effective展开、引用证据、父先子后等待和最终整树检查。全流程复盘记录 2.5.0 Runner处理14.67小时，其中P5约14.35小时；但处理桶不能分解为各函数CPU/I/O耗时。相反，全部127条P1/P3 root尝试的处理仅7.18分钟。不能用root小样本否定P5检查成本，也不能未经profile把14小时归给resolver或深树。

## 6. 待决问题、候选方向与可衡量验证

不提前设计新schema语法，不以“减少到某个节点数”为先验目标。优先检验以下四个问题：

| 待决问题 | 可先做的候选方向 | 后续验证标准 |
| --- | --- | --- |
| 哪些组合真有必要独立？ | 从FMHA相邻scan/copy/combine组合、MoE Small/大准备及重复验证路径选少量对照；用现有P4七字段附带来源证据说明共同规则、不可合并的最小结构差异和可达见证。 | 同一冻结Contract/源码下，覆盖原选择范围；无正文删launch/producer/事件；同参数规则可统一的才减少节点。记录节点数、独立设计类数、遗漏/重叠发现数；没有见证的路径先标未知，不能直接删除。 |
| 哪些root问题可在传播前发现？ | 把已发生的首轮/稳态reuse、reader数、TMEM源限制、验收实际比较基准变为当前root审阅的具体边界样本；核实P2/P4新发现是否能更早回查。 | 同范围P3首次批准后的重开率、漏项类别、受影响既有节点数与后继调用/费用；保留独立审阅。用旧案例的检出率验证，不能只看提示变短。 |
| 整区共享能否降低实际维护？ | 选已证明同区等价的父子，使用现有from_parent；比较local与引用在相同子上下文下的有效内容、来源和必要local realization。优先评估共同推导只读一次的既有审阅组织。 | 分别记录raw重复量、模型输入/工具往返、W/R次数、返工、校验时间与总费用；用父改动和子参数改变验证不误用旧批准。无引用的节点不自动批量转换。 |
| 传播和检查的粒度是否过粗？ | 先测量GEMM式局部root修订、provider修复和换父的实际闭包与校验耗时；比较“请求影响、失效、实际改稿、纯重审”四集合。 | 明确每次失效的依赖原因；旧generation拒收、兄弟保持进度、权限不扩张、P6完整性全部成立。只有确认特定重复计算主导耗时后，才讨论更窄的失效或缓存方案；任何批准复用须有现有机制无法满足的独立依据。 |

仍未知：220节点中的最小必要数量及完整可达性；77→220中范围扩展、规范纠错和分组偏好的各自贡献；更多整区共享对真实模型费用的净收益；每种来源变化需要的最小语义重审范围；长处理时间的函数级分解；减小节点/提示后是否维持漏项检出率。上述六个当前run的state里未见provider reopen记录或identity_migration历史，故本节这两类生命周期的依据是现行代码和[2.5.0既有隔离验证](../../designs/2.5.0-design-region-derivation.md#10-实现范围与贯通验证)，**没有本次六算子生产成本对照**。

## 7. 证据位置与复核边界

可携带的主要证据为[初始/修订结果归档](../../../../schema_wiki/archive/six-operators/README.md)、[旧新位置与SHA映射](../../../../schema_wiki/archive/six-operators/LOCATIONS.json)、[费用汇总](../20261109-usage-review/summary.json)、[root样本](../20261109-usage-review/root_summary.json)、[root逐调用](../20261109-usage-review/root_attempts.csv)、[暂停节点清单](../20261109-pause/nodes.csv)和本报告支撑JSON。旧issue内的原路径属于当时事实，查现文件须使用LOCATIONS映射；归档搬迁没有改写run绑定或新增批准。

FMHA案例来自 run `45ea6608-66b2-4915-b47d-59d2c115602e` 的暂停工作checkpoint `e2cb480fdbadab88d555c766adc019cca0681c59889a63929553d81fa7789938`；MoE案例来自 run `feb0c655-4e13-43ac-817d-1f457f31c029` 的 `92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb`。这些是研究来源，不能当作已发布新结果。其他节点的selected checkpoint、文件SHA和原始路径见暂停CSV；支撑JSON保留本次输入CSV的SHA、选择口径及传播样本全部attempt ID。

本次只读解析JSON/CSV/YAML、核对代码/规范/回执并验证报告链接与算术，没有启动模型、Runner或监控，没有修改schema、测试、运行状态、冻结快照、产物或预算，也没有重新执行GPU验证。数据可以复核记录与机制，不能据此宣称表达问题、最小分支划分或提取费用已经解决。

[fmha-plan]: ../../../../.schema_extract/v2/runs/45ea6608-66b2-4915-b47d-59d2c115602e/checkpoints/e2cb480fdbadab88d555c766adc019cca0681c59889a63929553d81fa7789938/artifacts/dispatch.md
[moe-plan]: ../../../../.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/dispatch.md
[moe-flat]: ../../../../.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/fused_small_flat.yaml#L38
[gemm-trigger]: ../../../../.schema_extract/v2/runs/d4e41124-e29c-458c-a1ce-2a9bf87ee0a8/attempts/7599acb3-e338-4c7f-9352-7d19ead4aba6/receipt.json
[gemm-trigger-attempt]: ../../../../.schema_extract/v2/runs/d4e41124-e29c-458c-a1ce-2a9bf87ee0a8/attempts/7599acb3-e338-4c7f-9352-7d19ead4aba6/attempt.json
[gemm-child-revise]: ../../../../.schema_extract/v2/runs/d4e41124-e29c-458c-a1ce-2a9bf87ee0a8/attempts/0e9a4654-8610-4c39-a662-133412f0ec3a/receipt.json
