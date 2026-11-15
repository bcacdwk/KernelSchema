可以据此开始分析并记录 MoE dispatch。本次演练没有发现必须增补字段、模块或流程才能保存核心设计的缺口；必改建议为零，也不建议为这批材料再增加可选规则。最重要的判断是：数值变化可以是局部记录；初始化与失败前缀按其对所选设计的实际影响取舍；完整方案之间只在接口确实相容时组合。

本报告先读现行 DOC0/1/3–6 形成工作方式，再核对候选源码与指定产物，没有读取旧研究报告、旧文档 diff 或同轮审阅稿。源码简称 C 为 [nvfp4_moe.cu][cuda]，P 为 [solution.py][python]。[initial 归档][initial]涉及显式配置族；[较新批准批次][approved]覆盖默认选路及 ABI 重建、验证等调用状态。以下用前者考察额外数学/执行方案，用后者考察准备与验证组合，不合并两批的节点数量、Contract 或提取范围。

**按规则开始工作时，我会这样确定边界与 root。**

先以 [reference.py:69][reference] 和题目的输入边界确定预路由 MLP 的共同 Contract：解码、FC1、激活、block16 NVFP4 QDQ、FC2 的逐 route BF16 舍入、路由加权和、最终 BF16 输出。不能从某个候选函数签名扩充可选功能，也不能因某条路径方便而改误差要求。DOC0:110、130–138 和 DOC1:5–9 已经给出这一步的归属。

随后沿这个数据流比较“值怎样求出、保存在哪里、谁产生输出、谁等谁”，先找 split-K、融合归并、验证产生输出等差异，再检查相关容量、局部流水与包装。不会以 cfg、FC1×FC2 或 cache 状态作为最初目录轴。经过比较，选择无零验证的常规 packed-FP4 主链作为实际 root：路由准备 → gathered A → 融合 FC1/QDQ → FC2 路由结果 → finalize；cfg97/cfg81 是它的来源定位，不是设计身份。root 不承担尚未记录的 BF16、MEGA 或错误恢复路径。

这套起步方式直接来自 [DOC0:110–126][d0_class]，没有要求先枚举所有组合或求出最少节点数。将本次未保留的实现族写进 dispatch.md 的范围段即可；后续发现重要差异再补相应记录。

**八项实际演练。** 表中的局部名称只是 dispatch.md 中可引用的稳定名称，不是新增 schema 字段或 plan ID。

| 真实差异与证据 | 我的记录选择 | 必须保存的设计信息与理由 |
| --- | --- | --- |
| 同一单体 packed-FP4 主体的双、三、四级环；[C:7298][c_ring] 给出统一 SMEM 公式，[C:8748][c_ring_launch] 仅以模板深度调用同一主体。 | **同规则参数**。在代表该单体主体的完整 YAML 中定义深度及合法范围。 | 槽索引、复用距离、SMEM 容量随深度变化；参数关系必须写全，不能只保留 cfg7/8/9。改变等待位置、角色或 accumulator 组织则另判断，不能借这个深度参数包住其他工作组织。对应 DOC0:114、122。 |
| folded 与 non-folded 激活参数，及量化除法改成乘倒数；[C:446][c_numeric]、[C:491][c_fold]、[C:537][c_nonfold]、[C:1437][c_quant]。 | **局部记录**，如 `activation_argument`、`quantize_reciprocal`；相对已有 FC1 激活与 ActQ/ActS 定义。若完整 split-K 代表已经表达 non-folded 路径，直接链接它。 | SwiGLU 的 `fp32(g*fp32(alpha*L))` 与 `fp32(fp32(alpha*g)*L)` 不是同一个舍入链，L 为源码的 FP32 常量 -1.44269504；乘倒数也不能写成精确除法。保留 E4M3 scale、E2M1 cast、零 scale 处理以及近似依据/缺口，不能用“packed FP32”或一次 bit-exact 测试概括。数学/协议其他部分沿用基线，无须为每个 FC2 重写整套方案。对应 DOC0:150–154、DOC3:198–223、256–258、DOC6:178–187。 |
| FC1 从融合未分片求值变成 split-K：写 `[SK,Rmax,2I]` FP32 partials，再独立合并、激活和 QDQ；[C:3731][c_partial]、[C:1351][c_act]、[C:9287][c_act_launch]。较新批次的 split_store 是此类证据。 | **完整设计**，选择一套实际 FC1 split-K 与 FC2 配对作为代表；SK2/SK4 在同一分片规则内再参数化。 | 新增分片工作归属、全局 partials、合并提交及固定 slice 累加顺序，需要贯通到 ActQ/ActS 与最终输出。`k_act_packed` 的实际 Rmax 遍历和 pexp 检查也要保留，不能假定 padding 从未被访问或已经清零。后续换一个已知 FC2 不自动再建完整节点。对应 DOC0:116、120、154 与 DOC4:302–323。 |
| FC2 从写 BF16 RouteResult 后 finalize，变成 REDF 直接原子汇总并由 convert 收尾；[C:3948][c_redf]、[C:9164][c_zero_branch]、[C:9330][c_redf_join]。initial 的 ws_fused_redf 等保存了这一方案。 | **完整设计**，保存一个 REDF 代表，其他 FC1 配对记组合关系。 | 必须保留“先逐 route BF16 舍入，再 FP32 乘权重，再 FP32 原子和，最后 BF16”及原子和次序；不能因 RouteResult 不物化而删除那次舍入。`acc32[T,H]` 的每次清零、侧流 fork/join、convert 等待与存续共同改变输出形成。这里不能只将 RouteResult 的 dtype 改成 fp32。对应 DOC0:116、154，DOC3 精度规则和 DOC4 初始化/交接规则。 |
| Model 首用时执行 k_init_tables 与 device synchronize；[P:1005][p_tables]、[C:6618][c_tables]。但所选 cfg97/cfg81 packed 主干不读取它写入的三个 g_e2m1/g_e4m3 表；[C:136][c_e4m3] 的 e4m3_val 直接用位运算与算术解码。 | **对所选 root 属于可省实现细节**，不建立 conversion_tables 局部设计，不添加表就绪前提。讨论冷启动成本时，可以只留一句来源事实。 | “源码执行过初始化”不证明它是当前设计的依赖。表的实际读者是 decode_hidden、BF16 GEMM、MEGA 及诊断路径；应在保留这些消费者且该表影响设计时再记录。若首用同步本身影响本次所选范围的成本或协议，保留该影响，不因此将格式表提升为 packed 主干的数学/资源依赖。对应 DOC0:118–120。 |
| 完整主链前增加设备零验证；[C:1706][c_verify]、[C:9151][c_verify_launch]。当前数据 dirty 时继续原计算，clean 时结果由验证写出；ZERO2 又可先单独提交验证图，[C:9471][c_replay]。 | **主链的零验证是局部新决定**，如 `zero_verify`；独立 verify-only 若纳入目录，则保留一份以验证直接完成全部输出的完整代表。双验证与各 FC1/FC2 主链是已知决定的组合。 | 局部记录写清当前字节检查、无条件零写、dirty/arrival 初态与重置、ZGUARD 的输入来源、就绪边界和输出生产者。clean 时主链已提交的 kernel 仍存在，只是早退。ZERO2 的 clean 返回、dirty/timeout 后的第二次验证及完整主链，分别保留实际工作；不能把两次验证写成一次。无需复制较新批次每个 verify_*、double_verify_* 配对。对应 DOC0:115–117、154–156，DOC4:245–247。 |
| FC2 描述符拒绝时 cfg23→12→8，与 cfg44 的同类回退；[C:7226][c_fallback]、[C:8741][c_ring_launch]。[C:8722][c_refuse] 的其他拒绝分支约束的是当次 launch_gemm 的选择，不是整条主线。 | **已有决定的组合**：以已经说明的单体 cp.async FC2 为来源，分别说明它和融合/分片 FC1、验证记录的相容关系。搬运方式若尚未记录，先补一条相对单体 TMA FC2 的局部变化。 | REDF 的那次 FC2 不能回落为只写 rout 的主体，因为后继 convert 不读 rout；split-K 的那次 FC1 不能回落为不产出 partials 的主体。这不妨碍已经成功产生 partials 并经 ACT 形成 ActQ/ActS 的 split-K FC1 接普通 cp.async FC2；较新 [split_cp][split_cp] 正是 cfg99 与 cfg2=23/44→fc8 的组合。相容关系仍核对 ActQ/ActS → BF16 RouteResult 及就绪/退休，不为 selector 汇合点或验证包装重建节点。对应 DOC0:117、122、146。 |
| LP 的站点错误数字、graph_status 的数组下标等；[C:9082][c_status]。 | **可省实现细节**，必要时只概括一次来源行为。 | 错误码 740/733 的数值本身不解释设计。类似地，包装名称不产生新方案；但 [P:1576][p_wrapper] 的指针/对象存续条件、实际转调成本和失败前已发生的访问，应按是否影响所选设计保留，不能与名称一起删除。对应 DOC0:118、154，DOC6:13。 |

**两个容易误写但不需要追加规则的地方。**

格式表的补核修正了我将“首用函数执行过”误当成“所选 root 实际消费”的源码依赖判断。DOC0:118–120 已经要求按结果、性能取舍与必要保证判断保留价值；DOC0:154 保护的是相关状态与在途依赖，不要求逐项登记无关初始化。最小修改是修正本报告，**不需 common 增补规则**。

第一处是捕获失败的在途前缀。[C:9533][c_capture] 在 cudaStreamBeginCapture 失败时仍会调用独立流上的 run_pipeline，之后 [C:9577][c_capture_end] 还可能在调用流再次执行。不同写法会造成两种实质错误：把前一次调用当成“失败所以无工作”，或在正常主链上凭空补一个源码没有的跨流等待。最小处理是一次来源限制记录，说明实际可能已提交的访问、共用 scratch/output 与缺少的顺序保证；正常代表只主张已经说明的成立条件，不声称该错误前缀已经被修复。[DOC0:138、154][d0_scope] 和 [DOC4:261–263、312][d4_call] 已经明确约束这一做法。**不需规则修改。**

第二处是把历史 selector 当成持续输入前提。P 的 ZERO 接受结果只决定是否建立验证状态，C 每次读取当前字节；MEGA 的 cfg92 也可能在建立 ABI 时为空、后来读到非空 ids，[C:9096][c_mega] 明确保留实际计算。若把 when 写成“发布时为空所以本设计只清零”，会漏掉后续非空计算；若把 when 写成“当前 clean”而删掉验证，又把内部证明工作偷换成了调用者保证。最小处理是按设计的容量/接口与必要调用状态写范围，把发布时选路概括留在来源说明，验证和非空分支按实际设计保留。若共享 Contract 已保证某个值有限，也不额外捏造“权重永不变化”的限制。DOC0:136–146 足够区分这些事实。**不需规则修改。**

**共享、局部记录与 DOC1–6 的接合。**

root 保存 Contract 不要求其他设计有 parent；BF16 或 MEGA 可以与 root 并列。DOC1 的数学要求仍属于整个任务，不能用窄 when 修改量化、参考或误差。[DOC0:169–177][d0_from] 明确规定 from 只引入同名整区、在消费节点绑定；因而 FC2 工作参数变化后，共享公式中的 BN2 取当前节点值，但 ActQ/ActS 的数值意义、有效区域和产生顺序仍须核对。不能只想复用 FC2 就发明 per-step from；如果整个数学区包含已删除的 gather，那个区域必须本地改写。

同样，借用 math_reshape 或 tile_pipeline 不会自动带入来源的 when 或 realization。[DOC6:7–11][d6] 明确要求本地判断 tcgen05 完成、TMEM 读取退休、bulk-store 可见性等交接条件。局部 MD 只说明差异；完整 YAML 是否包含验证、清零、两次提交，仍由其实际三个 Design 区决定。DOC3:5、141、153，DOC4:5、245–247 与 DOC5:5、200 与这个读取方式一致，未发现本次涉及的分区冲突。

实际书写只需要现有 rewrite/symbol/arithmetic、schedule.strategy/flow、动作 describe/storage/waiting 和必要 realization。稳定的局部名称与来源链接即可组织差异，不需要新增“可应用 patch 模块”或隐式执行顺序。

最终文本 DOC0:150 已明确局部记录自身须以适用条件或所指已核实例界定范围。这与演练相容：zero_verify 限定于具有所需 ZGUARD 与就绪关系的主链；cp.async 回退限定于已核对 ActQ/ActS → RouteResult 接口的 FC2；数值记录限定其已声明的输入域和近似依据。它们不能只借一个较宽基线的 when 隐含扩域，也无需增加局部 when 字段或列出所有组合。

未审范围：没有重提取整个算子，没有逐份审完两批 YAML，也没有验证各个近似的完整数值域、所有底层 TMA/PDL 原语或所有备用 solution_*.py/.cu。没有运行提取、检查器、GPU 或测试；grouped GEMM 恢复重算未展开，MoE 自身的失败前缀已经足以检验本轮规则。上述结论是规范可操作性判断，不是对所有来源实现正确性的背书，也不主张演练中的完整设计数最少。

[cuda]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu
[python]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/solution.py
[reference]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/problem/reference.py:69
[initial]: /home/shaohanyong.shy/KernelSchema/schema_wiki/archive/six-operators/initial/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/dispatch.md
[approved]: /home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/dispatch.md
[d0_class]: /home/shaohanyong.shy/KernelSchema/schema_docs/0.common.md:110
[d0_scope]: /home/shaohanyong.shy/KernelSchema/schema_docs/0.common.md:136
[d0_from]: /home/shaohanyong.shy/KernelSchema/schema_docs/0.common.md:169
[d4_call]: /home/shaohanyong.shy/KernelSchema/schema_docs/4.design.work_schedule.md:261
[d6]: /home/shaohanyong.shy/KernelSchema/schema_docs/6.realization.md:7
[c_ring]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:7298
[c_ring_launch]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:8741
[c_numeric]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:446
[c_fold]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:491
[c_nonfold]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:537
[c_quant]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:1437
[c_partial]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:3731
[c_act]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:1351
[c_act_launch]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:9287
[c_redf]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:3948
[c_zero_branch]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:9164
[c_redf_join]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:9330
[p_tables]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/solution.py:1005
[c_tables]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:6618
[c_e4m3]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:136
[split_cp]: /home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/split_cp.yaml:19
[c_verify]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:1706
[c_verify_launch]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:9151
[c_replay]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:9471
[c_fallback]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:7226
[c_refuse]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:8722
[c_status]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:9082
[p_wrapper]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/solution.py:1576
[c_capture]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:9533
[c_capture_end]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:9577
[c_mega]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:9096
