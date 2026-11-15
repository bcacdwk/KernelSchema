# 第三轮独立反例审计

**结论：有条件接受“少量完整主线 + 一次局部记录”，接受 `when` 只表示记录范围、删除 `parent`、同任务整区 `from`。最强反对理由不是丢失全执行路径，而是作者可能用范围和来源链接，把尚未解释的数值接口、资源退休或已知缺陷藏起来。** 目前反例支持收紧这些表达边界，没有支持恢复组合矩阵、为每条短记建小 schema，或增加新的继承结构。

本轮先读 AGENTS、[问题定义](00_problem.md)、[审阅记录](REVIEW_NOTES.md)，独立审查 [03](03_design_identity.md)、[04](04_sharing_and_records.md)并形成下文判据，再读 [README](README.md)和两份 [case_trials](06_case_trials.md)初稿。直接核对了相关 YAML、MoE/GDN/FMHA/Sparse 源码，以及固定提交中的 FA4、GDN 模板源码；未运行提取或 GPU。以下“必须修”针对候选最终表述及试写事实，不是要求修复被研究的原实现。

## 1. 简记的边界：局部决定可以重要，但不能让读者补发明

**问题。** “改了多少区”“增加多少 kernel”不能决定记录深度；“把一个变化起名”也不能证明它有清楚边界。另一方面，完整 YAML 的语法本身不会使缺失事实变得正确。

**证据与反证。** 我采用的门槛是：短记连同明确基线，是否已经说出新增决定及它破坏或依赖的关系；读者是否还必须自行选择新的数值含义、工作归属或完成边界。这里检查解释是否闭合，不要求短记可机械展开。

| 案例 | 最小反例或容易遗漏的事实 | 本轮判断 |
| --- | --- | --- |
| FA4 `shared_alpha` | 仅把 Stats 改成一列，Correction 若仍把它当 `(old maximum, base)` 计算，值与读取范围都会错。源码把同一个 FP32 alpha 交给分母递推及 Correction；捕获 Stats 后可以归还槽位，但修正 Y 仍要等前次 PV。首块只确认 Stats、不读载荷的基线协议也不能丢。 | **短记足够。** 04 的生产者、载荷、两个消费者与基线握手说明已经切中边界。无需复写整条 attention、固定物理列号或逐调度再记一遍。[生产/使用][alpha-source]、[捕获后归还][alpha-release]。 |
| FA4 decode `compact` | 普通 global 读取空 split 的 Opartial；`0*NaN` 仍为 NaN。compact 只读有效集合，且每个输出线程串行重算 max/denominator，改变归约组织。仅写“跳过空 split”仍少一半决定。 | **短记可以足够。** 说明有效集合、未写 payload 的读取差异、归约顺序，并指回不变的 partial producer。它不据此声称源码取消了所有 scratch 初始化；改变的是结果对空 payload 的依赖。[普通公式][decode-global]、[compact 公式][decode-compact]。 |
| GDN `ready-only` | 五槽只能推迟覆盖，不能证明旧读者已结束。真正关系是同一 ScalarOutput 顺序中，`prepare(c+5)` 在 `store:O(c)` 完成后；已处理 G/Gamma/Beta 共同发布 ready，不逐读者返回 empty。若连 scalar TMA 一起记录，还要区分 RawScalar 两槽的读完后重发和已处理五槽。 | **可以围绕标量供给协议短记。** 不能只用 `slot_count=5` 或“去掉 empty”概括。输出进度已经证明退休时，无需再补一套消费者回执。现有两个模板也已经很少，没有为了减少一份而强制合并的理由。[模板协议][gdn-ready]、[实际循环][gdn-loop]、[RawScalar 复用][gdn-raw]。 |
| GDN 请求排序 | 若在 prep 完成前沿旧 `perm` 预取 chunk 0，就可能取错请求。源码允许 metadata warming，但丢弃其值；真实 `n=perm[n_idx]` 在等待后重新读取，PRE0 只在无 perm 时成立。 | **短记足够。** 04 的“真实请求身份出现时机 + PRE0 限制”已完整表达新增联动。无需记录排序网络，也不能扩大为 chain 一律不得提前启动。[源码][gdn-sort]。 |
| FMHA FSCAN | PartA/PartB 改成未加权 partial，PartM 与 MTail 也换了生产者；旧 combine 即使同名同 shape 仍接错语义。末 split 的 MTail 只能在 split kernel 完成后供 combine 使用。 | **选一个完整代表合理。** 新的 split-local scan 和最终归并共同构成主干。已有一个完整 FSCAN 代表后，不再为 copy/store/combine 的每种配对新建主线。若只记录局部知识，必须把整个联动说清，不能据此宣称另一个完整 FSCAN 方案已交付。[epilogue][fmha-fscan]、[权重函数][fmha-weight]。 |
| Sparse hidden_mma | host 拒绝 packing 前，packed 前缀已经提交。只引用 mma_direct 会丢失前缀输出；只引用两个算法会漏掉输出区间、输入 fork 和 caller join。余段在 side stream 等 probe，**不等 packed 前缀**。 | **目录若保留这种整体编排，应有一个完整代表。** 这是把两种后端组成同一次调用的新工作归属。来源说明可以很短，不能让它代替实际混合设计。[源码][sparse-hidden]。 |

FSCAN 的接口错误可以不用 GPU 展示：两份 partial 的 numerator 为 `1,0`、denominator 为 `1,1`，PartM 为 `0,ln(2)`，MTail 为 `ln(2)`。正确权重是 `1/2,1`，归一结果为 `1/3`；旧无权重 combine 得到 `1/2`。这是接口代数反例，不声称已经构造并执行完整算子 workload。

**严重性：必须保留的解释门槛；无须新增字段。** 03/04 大体已满足。不能把“出现新 producer/跨区联动”自动升级为完整设计，否则 alpha、ready-only 又会反向制造节点；也不能把任意整条调用临时命名成一个“局部模块”。

**最小修正。** 把上述门槛用于审稿：未说清的关系补在该项说明；需要重新组织整次调用时选一个完整代表。不要给每条记录加统一输入输出、事件、存储长表。FSCAN/Sparse 的完整代表是基于所选学习对象的判断，不是“必须再为所有组合补全”的新义务。

## 2. MoE 验证：短记不能把来源缺陷变成正确性保证

**问题。** 零验证不是普通包装。`when` 改成记录范围后，尤其容易把“来源 host 选到了此路径”悄悄改成“当前零结果已经被证明”。

**证据。** `k_zero_verify` 无条件写完整零输出，同时计算当前 device Clean。带 guard 的主干仍有实际提交，Clean 时跳过 payload，Dirty 时重算并覆盖零输出。设备 dirty OR、arrival 计数初态和重置、guard 读取前的验证完成关系仍是设计事实。[设备代码][moe-verify]与旧 [verify_only 限制][moe-only]均明确可核。

`graph_replay` 的 host word 无代次，CPU 写 `-1` 不等待旧 GPU writer。以下反例甚至不需要跨 stream 或并发 host 调用：

1. A 的有限轮询超时，host 将带自身 verify 的 full 图继续排入同一 stream；A 的设备工作仍在途。
2. host 随后为 B 把 mirror 重置为 `-1`，排入 B 的 clean 图。B 的输入更新可以按 stream 顺序排在 A 之后、B 之前，仍是合法的当次输入。
3. A 尚未结束的某次 verify 晚发 `1`。B 的 host 观察到它，因此不提交 B 的 full 图。
4. B 自己的当前输入实际为 Dirty，其 clean 图仍只写零输出；缺少 full 图覆盖。

这里区分的是 **HostObserved、当前 Clean、kernel_done**。same-stream 保证设备提交顺序，不能给 CPU 观察补出请求代次。来源注释“reset 后不可能陈旧”不能代替这个交错分析。[回放源码][moe-replay]、[晚发布位置][moe-publish]。

**严重性：必须保留。** 03 已正确拒绝补造调用隔离；试写选择不经 verify-only 前缀的 `verified_packed` 完整代表，再集中记录双图和镜像限制，是可接受的取舍。被批准过的历史 `verify_only` 不是当前调用正确性的证据。

**最小修正。** 目录只需保存一次：验证确实生产输出；双图有两次验证/输出版本；host 观察不证明当前 Clean 或完成；无代次是来源限制。设备完整主干保持其真实 guard/完成协议。无需逐 FC2 复制，也不新增一个“安全隔离”参数去美化来源。README 的最终规则应保留“范围声明不补造来源保证”，不能只把它留在待研究清单。

### 被否决的假问题：是否必须新增“权重 scale 不可变”

建 ABI 的 `k_zero_scan` 检查权重 scale，执行时 `k_zero_verify` 不重查；源码注释称权重内容稳定，但这不是自动提供的保护。**然而当前共同 Contract 已将两组权重 scale 限制在 `2^(-9)..448` 的有限范围，E2M1 payload 也没有非有限码。** 从一个合法有限 scale 改到另一个合法有限值，不破坏零激活的零输出推理。[Contract][moe-contract]、[probe 调用][moe-probe-call]。

**严重性：无需增加规则。** “旧 probe 后权重 scale 变 NaN”不是当前 Contract 内反例。可以照实说明 eligibility 多检查哪些对象、执行时不重复哪些检查，但不能据此强加整个图寿命内权重内容不可变。若将来扩大 Contract，才重新检查该结论。本轮不把源码注释升格为已实现的 immutability 保证。

## 3. 失败可以不穷举，但在途访问不会随着记录范围消失

**问题。** 排除某次失败调用，不等于排除它留给随后成功调用的状态。只保留“失败则走正常主干”会错过真正需要理解的异步访问。

**证据。** `moe_run` 创建 nonblocking capture stream 后，调用 `cudaStreamBeginCapture`，但不论 BeginCapture 成败都调用该流上的 `run_pipeline`。失败分支再回 caller stream 调用 pipeline，源码没有在两者之间提供 fork/join。[源码][moe-capture]。在确有私有流提交仍未结束的情形下，后来的调用即使本次只走 caller-stream eager、返回成功、没有新增私有流工作，也不能凭这些事实证明共享 workspace 无旧访问。

这和“一次 GEMM 在同一 stream 顺序重算几遍”不同。后者通常可概括重复次数上界、后次覆盖、无前次结果依赖和存续关系；前者留下跨流未接续的读写，不能当作同一串行主线的重复。这里没有验证具体 CUDA 错误组合的实际可达性，不把静态控制流风险写成已复现故障。

**严重性：必须保存相关来源限制；不要求完整失败路径目录。** 03 的“已经发生的异步访问和占用不能因失败消失”正确，但应用到 `when` 时还要考虑**先前调用**留下的访问。源码没有 drain 时，不能在新范围里声称服务边界已经提供 drain。

**最小修正。** 一次说明成功代表覆盖什么工作、哪些来源错误可能留下未接续访问；完整设计中的存续/复用条件对所有在途使用成立。错误码数字、异常语法、每种 capture/重建/FC2 配对均可省。若不研究这些异常轨迹，明确未研究，不宣称“整个来源族均已保证安全”。

## 4. `when` 的单一语义够用，字段名本身不是决定因素

**问题。** 现有名字 `dispatch.when` 强烈暗示 selector；但新增 `scope/selector/validated` 三个字段会把本轮不再承担的职责又结构化回来。

**证据与区分。** 下列三句话不能互换：

| 事实 | 一句话的落点 |
| --- | --- |
| 本记录范围 | `when`：本记录承担哪些输入和必要调用状态；完整 Design 在这些条件下解释全部输出与所需关系。 |
| 原实现选路 | 索引来源说明：某版本按何条件选此实现；可以只链接固定源码的长阈值式。 |
| 已有验证 | 索引实验链接：哪份实现、环境、样本和检查实际通过。 |

设计 R 可以保守地取原选择集合的一部分，但选择进入该路径不证明设计满足 Contract；也不能将旧 `HostObserved=1` 机械改写为“当前 Clean”。声明“只记录已准备调用”可以缩小记录范围，却不能制造准备值的初态、可见性、数据版本或排他保证。

**严重性：必须改旧语义；保留名字可讨论。** 目前没有必须改名或新增字段的反例。03 的单 root 缺省为全 Contract、额外限制必须显式写出，语义足够。范围之间重叠无须排序，空缺不表示 fallback、非法调用或原实现不支持。

**最小修正。** 正式规则用一句话固定“仅为本记录范围，非原选路、非验证证明”；迁移时重读旧 `Select exactly when`，不能只替换开头几个字。目录用一处范围说明交代主要未记录对象。旧来源中的已知缺陷仍须标注；即使选取健康子域，也不得声称来源会自动把调用隔离到该子域。

## 5. 删除 `parent` 与整区 `from`：更简单，但无环不证明兼容

**问题。** `parent` 能提供一个比较入口；删除后会失去机器可读单父谱系。任意整区来源又可能把多个完整方案拼成一个语义错误的消费者。

**证据。** FA4 的两份 alpha 数学区相同而主要来源不同，是解除复用/谱系绑定的具体理由；MoE 当前 96 稿扩大引用额外省 0 区，所以不能把这项语法变化说成主要压缩机制。Sparse 的 packed 前缀和 mma 余段则说明一个 parent 不能表达全部贡献。当前候选让这些贡献只在 MD 写一次，内容依赖由 `from` 唯一表达，没有需要恢复 parent 的信息空洞。

节点级 DAG **比展开终止所需更严格**。最小例子：A 的 math 引用 B、本地 pipeline；B 的 pipeline 引用 A、本地 math。每区都能有限展开，节点图仍有环。该例说明 node-DAG 是保守组织规则，不是数学必然；也没有必要为它引入跨区例外。可将公共正文放到同一个完整 provider，再让另一个引用，避免此交叉。

反过来，完全无环也不能使 FSCAN 的未加权 PartA 与旧 combine 自动兼容。消费上下文绑定必须继续检查值含义、版本、有效索引、worker/资源/事件和必要 Realization。provider 的 `when` 不继承，因此消费者不能顺手取得 provider 的前提或正确性结论。

**严重性：接受当前最小机制；无需增加规则。** 同任务唯一 ID、同名整区、本地或引用二选一、root 全本地、节点 DAG、消费者重新绑定已经构成清楚的展开边界。root 本地是阅读入口的选择，不是所有 DAG 都必须如此的技术定理。未见值得为抽象 provider、片段覆盖或跨区环增加复杂度的实际收益。

**最小修正。** 保留现规则即可。维护沿实际 `from` 和短记的解释依赖进行；纯历史来源链接不自动触发内容变更。provider 或消费条件改变时复核相关接口，不要求为每个未研究组合重新建审阅任务。审批/缓存从祖先关系迁移到有效内容依赖仍是工具 TODO，不能说删除 parent 后旧审批身份自然仍有效。

## 6. 试写确有筛选价值，但还要删掉“不值得单独记”的内容

**问题。** 如果每个旧分支都被改名为一个 Markdown 小节，成本只是搬家。只检查“短记够不够”还不够，必须判断其是否包含要保留的新决定。

**证据与最小修正。** 独立判据形成后读取的两份初稿中，有以下具体事项：

| 初稿事项 | 判断、严重性与最小修正 |
| --- | --- |
| FA4 额外完整条目 `folded_static` | **可撤出完整主线。** 已有 folded 映射和 root 单工作生命周期，不足以单独证明又有一套整体设计。但 [small_static_lpt][fa-static] 确有 elected-thread 的 WorkCoord 发布及首次 Q/K/V 等待，不能称 pipeline 原样等于 root。把这项坐标供给联动补在映射说明、链接已有具体组合即可。 |
| MoE `format_tables` 独立具名决定 | **无需保留为独立局部设计。** 在所选 packed 实例中后继不消费这些表；放弃逐调用复刻后，一次冷启动来源/范围说明已足够。若目录不讨论冷启动成本，也不必因为旧稿有一个 init 节点而登记同名笔记。其“实际提交过”只在声称复现该调用时必须展开。 |
| MoE `serial_handoff` 说要移除入口 griddep 协议 | **必须修正事实。** [源码][moe-pdl]明确 wait/trigger 无条件留在 kernel body，普通 launch 下惰性返回或不产生作用；[fused_serial][moe-serial]也保留 `enter:Protocol`。应写“不再用 early-launch 边接续，由普通 stream 保证前驱完成；入口 helper 仍执行但不起早启动作用”，不能写成来源删掉了它们。 |
| MoE `fc2_cp_fallback` 把“WS、split、REDF 不许回退”并列 | **必须明确作用域。** 拒绝规则针对被拒绝的那次 GEMM 及其输出接口。FC1 split-K 不能回落成不生产 Part 的普通 cp；这不禁止已产生匹配 Act 的 split-K 主干接 FC2 cp。[源码][moe-refusal]按 cfg 拒绝，`run_pipeline` 分别以 cfg/cfg2 调用两次 `launch_gemm`；[split_cp][moe-split-cp]正是后者的具体代表。无需展开 cfg 表，但不能把某个生产者的回落禁令写成整条主线的组合禁令。 |
| MoE `eligibility_scan` 的源码名 | **直接修正。** 函数是 `moe_zero_probe`，不是 `moe_probe_zero`；不要为追踪这项更正新增记录。 |
| 异常轨迹列为未研究 | **允许。** 不要求补回全失败矩阵；但“尚未研究全部轨迹”和“已知会留下未接续访问的来源限制”是两回事。后者可用第 3 节的一小段保存。 |
| `fc2_flat` 等通过局部记录再指基线 | **允许有解释依赖。** 不把这些名称注册为新的 dispatch ID；不默认说明顺序就是 patch 次序。只在所指对象改变时复核相关笔记，不给每条加审批状态。 |

收尾复读时，试写已撤去 `folded_static` 的独立完整入口和 `format_tables` 的独立标题，保存了 WorkCoord；已修正 serial helper、probe 函数名和回落主语，并补上先前在途访问及有限 scale 的区别。上表保留的是具体反例与修正依据，不表示这些初稿错误仍未解决。

这两份试写仍链接历史完整 YAML，没有迁移一套新语义的完整 YAML，所以目前能说明**目录可以减少重复身份和重复介绍**，不能说明所有设计细节已压成这些短段、实施者不再需要原代表，或已经测得净维护成本下降。

03 中“实际实施某组合时才生成完整目标”还有产生额外流程义务的风险。**README 已改为“需要交付完整 schema 时，再整理实际选定的具体方案”——这是应采用的最终措辞。** 实施者可以据设计做实现，不因为读了一条短记就自动欠下一份 schema；本轮研究也不以补成所有组合为完成条件。

## 7. 第三轮试写的验收门槛

以下是审稿人的反证测试，不是每条短记必须填写的小模板：

1. **删除测试。** 去掉该记录，具体会误解什么新决定；若只丢一个不被消费的历史调用名，可降为来源说明或省略。
2. **歧义测试。** 能否构造两个都符合短记、却在值含义、输出归属或复用时机上不同的解释；若可以，补造成分歧的那句话，或选一个完整代表。
3. **事实测试。** `HostObserved`、“成功返回”“缓存命中”等观察是否被偷换为当前值、完成或排他保证；若是，退回真实来源行为。
4. **组合测试。** 记录是否又按每条主线/后端复制；已核的共同决定只写一次，未研究组合不冒充冗余，也不因此自动新增待办节点。
5. **交付测试。** 完整 YAML 是否在自身范围内闭合，短记是否保持解释职责；没有承诺直接交付的组合不强制生成完整 schema。

**精简结论规则：** 完整 YAML 记录选定范围内的一套端到端设计；同规则参数合并，重要局部决定只补新增内容及必要联动，既有决定的组合不另生身份。`when` 只限本记录范围，不补造来源保证，也不缩 Contract；来源贡献由索引说明，同名整区 `from` 只共享内容并在消费者重新绑定。可以省掉原选路与失败组合的穷举，不能省掉被记录设计所依赖的真实值、初始化、在途访问和退休关系。确需完整 schema 交付时才整理所选具体方案。

本轮**不能声称**：全部 MoE/FMHA 组合已验证、镜像竞态或 CUDA 失败路径已做 GPU 复现、已有精确最少设计数、历史 60/154/96 稿已无损迁移、局部记录已经能自动展开、删除 parent 已兼容原审批机制，或实际费用/时间已按某比例下降。

[alpha-source]: https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_prefill/aka_solution/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py#L3523
[alpha-release]: https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_prefill/aka_solution/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py#L3753
[decode-global]: ../../schema_template/examples/fa4_decode/root.yaml#L161
[decode-compact]: ../../schema_template/examples/fa4_decode/q4_compact.yaml#L85
[gdn-ready]: ../../schema_template/examples/gdn_prefill/m64_single_sequence.yaml#L222
[gdn-loop]: https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/gdn_prefill/aka_solution/gated_delta_net_chunked.py#L2155
[gdn-raw]: https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/gdn_prefill/aka_solution/gated_delta_net_chunked.py#L2395
[gdn-sort]: ../../kernel_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/candidate/gdn_kernel.cu#L1998
[fmha-fscan]: ../../kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_tcg.cuh#L5181
[fmha-weight]: ../../kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_common.cuh#L292
[sparse-hidden]: ../../kernel_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/candidate/dsv4_kernel.cu#L6140
[moe-verify]: ../../kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu#L1707
[moe-replay]: ../../kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu#L9471
[moe-publish]: ../../kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu#L1764
[moe-capture]: ../../kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu#L9533
[moe-pdl]: ../../kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu#L180
[moe-refusal]: ../../kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu#L8722
[moe-probe-call]: ../../kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/solution.py#L1478
[moe-contract]: ../../.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/root.yaml#L39
[moe-only]: ../../.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/verify_only.yaml#L145
[moe-serial]: ../../.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/fused_serial.yaml#L247
[moe-split-cp]: ../../.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/split_cp.yaml
[fa-static]: ../../schema_template/examples/fa4_prefill/small_static_lpt.yaml#L143
