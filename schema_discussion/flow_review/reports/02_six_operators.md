# 六个提取算子的 flow：核源评估

**结论：保留 launch、显式启动依赖和按值 handoff，先收紧使用边界，不扩成 host 控制语言。** 实例支持把启动许可、数据就绪、消费等待点分开，也支持同 kernel 的 per_worker 接续、多 consumer 和存储延续。已查反例不要求新增 producer 列表、stream 表、retry 控制或一般 lifetime 字段。轻量方案可删除 protocol 字段，让已有 realization 绑定兑现启动机制；这与删除早启动能力是两回事。另可删除被启动链蕴含的 wait_at 及 API 过程说明。

现行 flow 无法直接寻址外部 Geo 上传、caller fork/join 或跨次共享状态；这些事实有设计意义，但不必因此把 host 步骤变成 phase。建议先规定简短的调用前置、输出可见性和复用约定，保留具体反例，再决定是否需要可选的外部端点语法。源码没有提供的顺序和代际保证不能由约定补成“已实现”。

## 阅读范围与证据身份

遵守本轮 METHOD，仅讨论 flow；没有改正式规则、算子或工具，没有 GPU 验证。六 root 的 **flow 全部直接读过**，并定向读其工作定义、数学值和管线片段；这不是六 root 全文正确性审计。规则依据为本轮捕获的 [work_schedule](../rules/4.design.work_schedule.md:234) 与 [tile_pipeline](../rules/5.design.tile_pipeline.md:503)，不是旧稿的 resource/milestone 拼写。

使用 corpus.json 指定的暂停工作输入。FMHA 的工作 checkpoint 为 e2cb480fdbadab88d555c766adc019cca0681c59889a63929553d81fa7789938，MoE 为 92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb；二者整树仍 paused。以下 approved 表示捕获时的历史状态，不表示符合今天规则。

| 算子 | 直接阅读的 root / 分支 | 阅读深度与状态 |
|---|---|---|
| GDN | root、cached_ordered、direct_qk、segmented、segmented_ordered | flow；root 和 segmented 的有关工作/事件；均 approved |
| GEMM | root、v0_recovery、wide_recovery、wide_recovery_chain、wide_recovery_full | flow；root 工作定义；full 的选择条件、数学版本与重复工作；均 approved |
| MLA | root、bootstrap | flow、工作定义及外部表/queue 说明；均 approved |
| Sparse | root、probe_serial_mma、probe_hidden_mma | flow；root partial 消费；hidden 的 fork/join、分区、host probe；均 approved |
| FMHA | root、tc_split_flat_narrow、tc_split_segmented_narrow、tcg_d128_flat_cp_direct_narrow、tcg_d128_fscan_cp_direct_narrow、tcg_d128_segmented_cp_direct_narrow、tcg_d256_flat_cp_narrow、tcg_d128_flat_tma_xpose_narrow | flow；TC 缺覆盖区域；fscan 提交变化；xpose 源限制；均 approved |
| FMHA | tcg_d256_segmented_cp_narrow | 仅看 flow；needs_authoring，不计为独立成立的设计证据 |
| MoE | root、direct、fused_small、fused_serial、split_store、init_no_scan_direct、verify_root、double_verify_root、verify_only、probe_fused_small | flow；direct/verify/初始化的相关工作和管线定义；均 approved |

另外对固定输入做过 flow 字段定位/去重扫描，用于选择代表，未把扫描到的其他节点算作语义覆盖。18 份固定源码及以上选读输入的哈希均与 corpus.json 相符；“已核哈希”不等于读过每份源码。

直接回查的源码是固定副本中的 GDN gdn_kernel.cu；GEMM candidate_solution.py 的绑定背景及 fp8gg_umma.cu 的重试路径；MLA solution.py 的表构造/上传/queue；Sparse dsv4_kernel.cu 的 partial merge、probe 与 graph 路径；FMHA fmha_mxfp8_qkv.cu、fmha_tc.cuh、fmha_tcg.cuh、fmha_common.cuh 的选中提交/等待/缺陷片段；MoE source_solution.py 和 source_nvfp4_moe.cu 的初始化、direct、verify 与提交路径。具体文件和主要行段见同目录的 [数据](../data/02_six_operators.json)。

复用旧调查只作定位：[SCHEMA_DISCUSSION](../../../schema_extract/archive/20261109-ten-operator-text-audit/SCHEMA_DISCUSSION.md:61) 的外部边界与共享 scratch；[SCHEMA_VERBOSITY](../../../schema_extract/archive/six-operators/20261109-methodology-review/SCHEMA_VERBOSITY.md:198) 的结构/来源区分；[phase_grammar](../../phase_grammar/README.md:42) 的事件及跨次边界。下列关键判断均回看了本轮输入和相关源码，没有阅读本轮其他代理报告。

## 计算组合：哪些差别必须留在 flow

| 算子 | root 与代表变体 | 对设计有意义的差别 |
|---|---|---|
| GDN | root：prep → chain；segmented：prep → pass_a → pass_c | 两次/三次提交、分段传播结果 SegA/SegB 的全局物化、多个消费者和更短的 recurrence 工作 |
| GEMM | root 一次；recovery 两次、三次或四次 | 多次完整重算和同一输出的有序覆盖；没有中间值交接 |
| MLA | root / bootstrap 都一次 | queue refill 与一 tile 一 CTA 改变工作分配，未改变 kernel 边界；Geo 外部供给仍需就绪 |
| Sparse | root attention → combine；serial probe → attention；hidden 为 probe、prefix、remainder 三次 | split partial 全局交接；或 prefix 与 remainder 无相互完成依赖、共同覆盖输出 |
| FMHA | root 一次 fused；TC flat 四次、segmented 五次；TCG flat 四次、segmented 五次、fscan 三次 | scan 融入 split 会取消一段全局中间结果和整个独立扫描链；producer 随所选计划改变 |
| MoE | root 七次；fused_small 五次；split_store 也是五次但阶段不同；direct 一次含两 phase | routing 融合、局部 FC1 activation 融合或拆分、全局 Part 交接、片上 Census 延续不是同一个“fused”标签 |

### E1. GDN：启动早不等于可读早，wait_at 不是可删的装饰

[root flow:326–338](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml:326) 已分别写出 prep.launch_ready、prep.kernel_done 和 chain.prep_ready。管线 [601–609](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml:601) 在 wait 前读 caller 的 initial_state，并可先发第一块输入 Q/K/V；系数拷贝随后才补齐同一 stage。

源码 [gdn_kernel.cu:410–421](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu:410) 在 prep 计算/发布前发启动提示；[1998–2025](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu:1998) 发起 input-only 预取；[2211–2224](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu:2211) 把状态加载放在 dependency wait 前。launch 属性在 [3431–3448](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu:3431) 绑定。

若把 start_after 改成 prep.kernel_done，会取消已有的独立准备机会；若把 handoff.ready 改成 prep.launch_ready，会放行尚未发布的系数；若删 wait_at 并采用消费者 worktile 入口等待，会把已有的输入工作推迟。三者不能合并成一条模糊的“after prep”。这里现行能力够用。

segmented 的 [223–232](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/segmented.yaml:223) 让 Ti、Kc 等被 pass_a/pass_c 都消费，同时 pass_a 生产 SegA/SegB。源 [2028–2037](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu:2028)、[3450–3460](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu:3450) 支持这三次提交与重跑结构。共享 prep 数据须存活至两个消费者的有关读都结束，不能在 pass_a 完成时统一释放。

### E2. MoE direct：per_worker 与片上 handoff 都有不可替代的实证

[direct:184–199](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/direct.yaml:184) 是一个 kernel、两个 phase、per_worker 接续。prelude 对每个 CTA 做一份复制 census；direct 则让该 CTA 处理其连续 unit 区间，[204–242](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/direct.yaml:204) 不是相同 worktile 坐标的一一配对。

源码 [1946–1976](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:1946) 在每个 CTA 内清无路由输出、复制扫描并发布 s_nvalid，随后 [1978–1985](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:1978) 遍历该 CTA 的多个 unit。没有全 grid census 屏障。因此：

- per_worktile 会错误要求两 phase 的工作配对；all 会增加源中没有的全体等待。
- 拆成两个 launch 会失去同一个 CTA 的 SMEM Census 身份。
- 仅把 storage 写 global 会改变实际物化路径。当前片上区域引用足够；旧稿 resource 应按现规则改名为 storage，而非新增 owner 或 scope。

最后到达的 unit 完成 token 归约并清回 RouteAccum/Arrivals，见源 [2205–2256](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:2205)。这是 direct phase 的跨 worktile 合并和终止职责，不是又一独立 finalizer phase。两 phase 共同写 Y 的不相交区域也不构成一个内部 handoff。

### E3. 融合/拆分应改变实际工作与交接，不靠说明字符串删 phase

FMHA TCG flat 的 [208–219](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/tcg_d128_flat_cp_direct_narrow.yaml:208) 有 group_max、flat_scan、split、combine；fscan 的 [220–229](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/tcg_d128_fscan_cp_direct_narrow.yaml:220) 只剩 group_max、split、combine，MTail 改由 split 产生，并增加 PartM 交给 combine。

源码 [1643–1677](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_fmha/source/candidate/fmha_mxfp8_qkv.cu:1643) 明确让 fused scan 跳过整个扫描链；不能同时保留 scan 的 MTail 写入再说它“不执行”。现有不同 plan + 单 producer 已能表达。也不能为了少一条 launch，把不同的 producer/consumer 偷藏进同一 flow 的条件文字。

MoE [fused_small:192–220](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/fused_small.yaml:192) 把三段 routing 和其 memset 改成一 CTA prepare；[split_store:246–271](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/split_store.yaml:246) 则独立物化 FC1 的 Part，由 activate 合并并量化。源码 routing 两种分支在 [9222–9241](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:9222)，独立 activation 在 [9315–9327](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:9315)。这些是实际计算/流量决定，应该留在 flow。

### E4. 重复提交不必引入 repeat/retry 语言，也不能被 graph 名称吞掉

GEMM [wide_recovery_full:34–50](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gemm/wide_recovery_full.yaml:34) 条件化到实际提交 L=[G,G,F,F]；其 [150–170](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gemm/wide_recovery_full.yaml:150) 为四个有序实例，anchor 共用工作定义。每次重算不读上次 out，故没有 handoff，但 WAW 顺序与四份计算量仍在。两次/三次代表见 [v0_recovery:125–134](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gemm/v0_recovery.yaml:125)、[wide_recovery_chain:150–159](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gemm/wide_recovery_chain.yaml:150)。

源 [fp8gg_umma.cu:1723–1762](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gemm/fp8gg_umma.cu:1723)、[1780–1816](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gemm/fp8gg_umma.cu:1780) 确实有 graph 调用报错后的 direct 重试和换 engine 路径；但静态源码不能证明某个报错 API 已提交或成功完成计算。应保留稿件“只计实际提交”的前提，不能把每次 API 尝试直接等同于一个完成 kernel，也不能声称已复现四次执行。

MoE [double_verify_root:359–404](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/double_verify_root.yaml:359) 的两次 verify 是更直接的重复实例：外层 capture 只记录两个 graph，之后 replay 才执行两次。源 [9471–9488](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:9471) 区分 outer capture、轮询和 full graph。保持两个 phase ID 即可，不能重复计算 capture 本身，也不需要把 host while 放进 flow。

## 字段是否够用，以及可实际缩掉什么

| 字段 | 建议 | 证据与限制 |
|---|---|---|
| launch / phase 列表 | 保留 | 单/多 kernel、同 kernel 多工作定义和实际多次提交都存在。phase ID 作为本次实例，anchor 共享字段已能减重复 |
| order | 保留 | E2 正向证明 per_worker；本组六算子未给另外三种取值足够反证，不能按未出现就删 |
| start_after | 保留，继续无隐式列表顺序 | GDN 早启动、GEMM 无 handoff 的 WAW、Sparse remainder 只等 probe，不能统一为默认顺序 |
| protocol | 轻量方案可删字段，机制保证留在已有 realization 绑定 | E1 证明早启能力必要，没有单独证明这个字段必要；GDN 已在 realization 完整绑定 PDL |
| handoff 的数学值键 / producer | 保留单 producer | 多 worktile 不等于多 phase producer；不同计划、不同版本、不相交输出已有去处。未见必须加 producer[] 的已查内部交接 |
| consumer[] | 保留 | GDN segmented 的系数、MoE Counts/Offs 有多个实际消费者，决定最后使用时点 |
| storage | 保留 global / 片上引用 | E2 Census 和 E1 全局系数给出真实区别；不用列分配器、cache key、每个 byte 的全局 arena |
| initialize | 保留窄用途，勿由 zeros API 推导 | 下节区分未写消费区域、归约初值、跨调用 self-clean 和源缺陷 |
| ready | 保留默认与覆盖 | PDL 的实际就绪是前驱全 kernel 完成；默认的相关生产 worktile 完成不应被误读成 launch_ready |
| wait_at | 保留按消费者覆盖，但删除可推导条目 | GDN 需要；MoE 完整 verify 启动链能蕴含七个 guarded 时点，无须重复别名 |

**protocol 的字段取舍必须与早启能力分开。** E1 要求三个时点有不同含义，但 [GDN root:684–686](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml:684) 已由 realization 绑定 programmatic serialization、launch_dependents 和 dependency wait。保留 start_after: prep.launch_ready、ready: prep.kernel_done、wait_at: chain.prep_ready 及这份绑定，删去 protocol: pdl 没有删掉内部依赖或实际机制。这是字段重复的证据；下面的滥填实例只是辅助证据。

因此按本轮轻量目标，我倾向删除 protocol 字段，并把 launch_ready 的定义改为“已由实现绑定兑现的前驱 grid 启动许可”，仍须说明有效范围和放行保证，不能让事件名字自行创造同步。若希望读 flow 时立刻看见协议名，可保留为可选的阅读提示；这是查阅成本的取舍，本组没有证明它具有不可替代的表达能力。无论保留还是删除字段，都不能合并 launch_ready、kernel_done 和消费等待位置，也不能把未测得的重叠说成性能事实。

**明确可删实例 A：MoE 的 Clean.wait_at。** [verify_root:260–290](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/verify_root.yaml:260) 和 [double_verify_root:374–404](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/double_verify_root.yaml:374) 均已有 verify.kernel_done → count 启动，后续每段从前驱的 launch_ready 才能启动。每个 launch_ready 不早于所属 kernel 启动，因此七个消费者的任何局部 guard 事件都晚于 verify.kernel_done。删除这条 wait_at 及仅供它引用的 guarded 别名，不删除实际 guard、Clean handoff 或它们的局部 pipe。此结论不能推广到 verify.launch_ready 直接放行 prepare 的其他节点。

**明确可删实例 B：即使保留字段，也不应填没有本次启动差别的 protocol。** [verify_only:141–144](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/verify_only.yaml:141) 自己说明 shape_gated_pdl 记录的是属性开关，而且本请求没有后继 kernel。按现规则 [protocol 条件](../rules/4.design.work_schedule.md:258)，其最小 flow 就是一次 verify；具体属性若仍需约束可留 realization。不要在 protocol 字符串里发明一个同时代表条件、属性和调度的复合机制。

FMHA [TC flat:270–280](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/tc_split_flat_narrow.yaml:270) 给 scan/combine 标 pdl，但同稿 [512–515](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/tc_split_flat_narrow.yaml:512)、[677–682](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/tc_split_flat_narrow.yaml:677) 说普通 MMA 前驱不早触发、普通 split 不消费 scan 的早许可。源 [1988–2008](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_fmha/source/candidate/fmha_mxfp8_qkv.cu:1988) 相符。因此这里 flow 可只保留 kernel_done 链；实现侧的 PDL 属性/entry helper 不需在 flow 重述。TC segmented 的 segmax → scan 确实有早许可，[313–319](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/tc_split_segmented_narrow.yaml:313) 与源 [1368–1376](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_fmha/source/candidate/fmha_tc.cuh:1368) 支持，不能照删。

串行完整 kernel_done 启动链已经保证数据就绪时，重复的 handoff.ready: producer.kernel_done 也可省略；保留值/存储交接。PDL 链中则需重新检查，不能批量删除。

## 初始化与存续：不要把不同问题都塞入 handoff

**Sparse root 是“不应见未写就初始化”的真实例子。** [finalize/merge:239–261](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/root.yaml:239) 中空 split 只写 PartM/PartL；首批 vector 预取可读到合法分配内的旧 PartO，但零权重分支丢弃该值。源 [491–492](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_sparse/dsv4_kernel.cu:491)、[571–579](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_sparse/dsv4_kernel.cu:571) 支持。后续零权重读取改指 split 0，[590–608](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_sparse/dsv4_kernel.cu:590)；其范围非空由稿中 Klen(t)>=1 的域保证，见 [344–349](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/root.yaml:344)。不能为首批被丢弃的字节给整个 PartO 添加 initialize: 0；也不能把后续实际参与乘法的替代读取一概叫“无影响预取”。非有限数传播属于另一个数值问题。

**MoE Counts/Fill 是归约/分配计数初值。** root 的 [400–410](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/root.yaml:400)、[435–440](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/root.yaml:435) 保留 count 前零初值和 scatter 的 Fill 零基准；源 [9230–9241](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:9230) 有据。这不是简单“生产者漏写消费者的一块区域”，而是原子累加和地址分配成立的前提。现规则 [142](../rules/4.design.work_schedule.md:142) 已给 decomposition.strategy 去处；可缩成“Counts/Fill 每次归零，分别先于 count/scatter 使用”，删 cudaMemsetAsync API 名和相邻分配叙述，无需虚构一个计算 phase。

**MoE direct 的中性状态跨调用存续。** [direct:239–242](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/direct.yaml:239) 的 RouteAccum/Arrivals 首次为零，以后由最后消费者清回零。源 [source_solution.py:938–974](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_solution.py:938) 与 direct 源 [2241–2254](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:2241) 相符。改成“每次 allocate/initialize”会改变工作和复用设计；完全删去则丢掉正确性前提。保留一条调用前置/退出约定即可，不建立一般 process cache schema。

**真实 table 初始化 kernel 必须计。** [init_no_scan_direct:210–227](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/init_no_scan_direct.yaml:210) 有 tables → 同 kernel prelude/direct，并交接 E2Table/ScaleTable。源 [ensure_tables:1005–1016](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_solution.py:1005) 调 moe_init 后同步。table 计算与单纯分配不同；稳态复用计划不应再算一次它。

**普通版本寿命已由规则覆盖。** GDN [381–383](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml:381) 的每次重写、顺序复用，对应源 [3254–3272](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu:3254)；保留到所有声明消费者及其在途读完成，正是现规则 [309–324](../rules/4.design.work_schedule.md:309)。不需每个 handoff 再列 release/lifetime；grow-only 容量缓存、泄漏式静态分配避免析构等过程可移出设计正文。

## 外部关系确实存在，但不应扩成执行 host 程序

### E5. Sparse hidden：两个输出 producer 与 caller join

[probe_hidden_mma:343–367](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/probe_hidden_mma.yaml:343) 的三条 launch 只表达 remainder 等 probe；caller 输入就绪后的 fork、两个不相交输出的 join 在 thesis/describe。prefix 与 remainder 的分区及回接见 [399–446](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/probe_hidden_mma.yaml:399)。

源 [6153–6163](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_sparse/dsv4_kernel.cu:6153) 先 fork 到 side，再在 caller 提交 prefix；[6180–6190](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_sparse/dsv4_kernel.cu:6180) 在 side 提交 remainder，再回接 caller。两段都必须完成后，依赖整个输出的 caller 工作才能读；host 本身不等两段结束。probe 的 host 观测不是 probe.kernel_done，Side 顺序仍给 remainder 完成依赖。ProbeRatio 被 host 用作选择，不是 attention 的 tensor 输入，不应编造 probe → attention 数学 handoff。

当前 phase.event 无法引用 caller 的外部事件，这是一条真实能力边界。但这三条 launch 已完整表达内部计算图；可用一句设计层的调用交接约定保持 fork/join，没必要增加空的 caller phase、host worker、event API 或 stream 程序。realization 只能绑定这条已确定关系的机制，不能在那里首次决定调度顺序；将文字移进 require 本身并未修复结构化外部端点缺口。若后续要求自动检查外部约束，需单独设计外部端点引用；不能把 caller.prior_done 当成现行合法引用。

### E6. MLA：Geo 先于启动，排序仍是设计，side-stream 代码不是

[root:321–338](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/root.yaml:321) 混合了数据上传、排序、查询缓存、capture 和可选输出。应保留的三件事是：本次 Geo 来自当前 metadata；具体排序及 Scatter 判据决定工作排列；Geo 完整就绪后 main 才启动。用量、排序谓词不能随 API 过程一起删。host allocation capacity/descriptor cache 的刷新步骤和 event 名称不是 flow 工作。

源 [1895–1898](../../phase_grammar/sources/kernel/extract_mla/solution.py:1895)、[2010–2024](../../phase_grammar/sources/kernel/extract_mla/solution.py:2010) 确立上传 → main；bootstrap [28–29](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/bootstrap.yaml:28) 与 root 都只有一次 kernel。不能新增假的 table kernel。

同时不能过度概括“same-stream 复用安全”：该源在 [1646–1681](../../phase_grammar/sources/kernel/extract_mla/solution.py:1646) 创建 nonblocking side 并直接 D2H，没有 caller → side 的输入 fork；main 等 upload 不等于下一次 side upload 等前次 main。所读源只有前向 table_event。外部 metadata 已就绪、前次 table 读者已退场或禁止重叠调用，均不能在没有依据时说成源已保证。本轮不证明整个运行环境是否另有同步，仅将它标为调用/来源限制。

### E7. 缓存/capture 的保留边界

FMHA root [394–403](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/root.yaml:394) 和 Sparse root [556–575](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/root.yaml:556) 的 graph 创建、失效、第二次命中、实例化和 fallback 过程可离开 flow；留下实际 kernel 集合、必要先后、当前输入的读取与绑定版本、输出完成以及 scratch 的复用前提。capture 成功只记录、replay 执行一次的区分用公共约定说明。

异常也不能反向删事实：MoE 源 [9533–9536](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:9533) 在 BeginCapture 报错后仍可调用 private-stream run_pipeline，再于 [9575–9577](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:9575) caller fallback。源码不自动提供两路额外 join。当前 [MoE root:18–24](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/root.yaml:18) 已明确排除这种额外 uncaptured 提交，不能把该例冒充 root 漏 launch。若要描述该异常的实际执行，应另给受限实例/来源说明，仍不需要一般错误处理语言。

## 源缺陷与填写错误，不能用新字段混为一谈

| 实证 | 分类与处理 |
|---|---|
| FMHA TC maximum grid 只覆盖部分 query，却 scan 全部 row | **source gap，不是 initialize 的缺口。** [tc_split_flat:104–125](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/tc_split_flat_narrow.yaml:104) 明说未写 PriorMax 无初值。源 [gm_grid:1979](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_fmha/source/candidate/fmha_mxfp8_qkv.cu:1979) 用 qtiles128，而 TC 源 [200–207](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_fmha/source/candidate/fmha_tc.cuh:200) 以 qph 推进。填 0 或 -inf 不能补出缺失的 scores maximum；新增持久未定值机制没有帮助 |
| FMHA 空 split 的 xpose stage 缺前置 join | **局部管线/source gap。** [xpose:604–612](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/tcg_d128_flat_tma_xpose_narrow.yaml:604) 已把需要的复用顺序与源缺陷分开；源 [5412–5424](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_fmha/source/candidate/fmha_tcg.cuh:5412) 在 staging 写后才有 barrier。handoff.ready: split.kernel_done 不能倒过来修复 kernel 内早覆盖 |
| FMHA 空 split 的 entrance 读取没有 T 上界 | **访问合法性/source gap。** [xpose:583–587](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/tcg_d128_flat_tma_xpose_narrow.yaml:583) 与源 [3480–3488](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_fmha/source/candidate/fmha_tcg.cuh:3480)。丢弃返回值不使非法读取合法，也不是 handoff.initialize 能解决 |
| MoE host mirror 的旧 verdict 可被新调用观察 | **跨次来源限制。** [verify_only:145–157](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/verify_only.yaml:145) 已记载；源 [9480–9488](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:9480) 的 CPU reset/poll 没有代号，前次 GPU [1770–1786](../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:1770) 仍可能随后写旧 verdict。不能把 HostObserved=1 命名成当前 verify 的 ready 或 kernel_done |
| MoE 重复 guarded 别名、无后继 shape_gated_pdl；FMHA TC flat 的属性型 pdl | **实例可缩减/与现行填写条件不符。** 不需要新能力，见上文删除证明 |
| 旧 resource/milestone 拼写、单 phase 名称 main | **历史语法迁移。** 不作为本轮表达缺口，也不改变前述设计事实 |

## 最小建议

1. **保留核心字段及现行默认。** launch 只枚举实际计算提交；顺序靠 start_after；同 kernel 接续靠 order；值、物化、ready/wait_at 靠 handoff。保留多 consumer 的全部读者寿命，不再叠加同义 lifetime。
2. **优先缩掉重复字段和使用。** protocol 名与已有 realization 启动绑定重复，可删除字段但保留 launch_ready 语义及机制保证；若仍保留字段，也应只用于有意义的启动协议。已有启动链覆盖的 ready/wait_at 可省；phase/launch 身份不等于 graph/cache/API 身份；少量重复提交用已有 anchor 共享内容。
3. **短调用约定保住真实约束。** 外部数据何时可用、后续 GPU 工作何时可消费完整输出、共享 neutral state 何时可复用必须先在设计/共同调用约定中确定，realization 只绑定其实现。区别源已保证与调用前置；不逐项登记分配、cache key、stream/event API、轮询次数和异常控制。文字约定仍不等于结构化外部端点。
4. **本轮不建议新增字段。** 外部端点若要可自动审查，是值得另议的真实需求；现材料不足以要求把它扩成 host 语言。所查“多生产者”无需 producer[]；缺失数据、无界读取和旧 host verdict 应保留来源限制，不能由 schema 美化。

可直接对照的当前语法片段、protocol 删除提案与边界说明草图见 [02_minimal_flows.md](../sketches/02_minimal_flows.md)。它们是讨论片段，没有修改输入，也没有声称完成整份旧稿迁移。
