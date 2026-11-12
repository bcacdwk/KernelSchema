**文本中的关系与可删实现细节：独立调查 04**

三个概念合理。在本次读过的六份 root 和所选分支中，**没有找到必须由 docs5 承担、而 resource/domain/milestone 及其现有配套结构无法清楚表达的最小反例**。当前最需要的是用好现有动作、视图、索引和依赖，并收紧说明的职责。确实只能靠正文保留的 host 上传、caller fork/join、跨调用共享存储关系，应在 docs4/调用边界继续讨论；不能据此扩张片上资源账或局部事件表。

自然语言仍有必要。`view` 给静态存储位置，动作的 `describe` 绑定“本次从哪一个生产者、哪份副本读取、写到哪里”，原语保证说明不能由事件先后直接推导的细部顺序。这些内容可以很短；没有理由把所有读写组成再变成一套字段。另一方面，`describe` 不能吞掉已命名动作之间的必要依赖，也不能让基础动作的 `done` 降格为“只是发起”。

**实际覆盖与复用依据。** 先读 METHOD、corpus、指定的 [SCHEMA_VERBOSITY.md](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/six-operators/20261109-methodology-review/SCHEMA_VERBOSITY.md) 与 [SCHEMA_DISCUSSION.md](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/SCHEMA_DISCUSSION.md)，再复用十算子审计的 findings_catalog、六份 operator 报告、相关 annotations/findings、root 注记与删改候选定位。本轮没有重做全文字数审计，也没有读取本轮其他调查报告。

六个固定 root **全文已读**；MoE `verify_only` 全文已读；另外只读 GEMM `mma_sync` 的 130–245、`wide_recovery_full` 的 145–255/450–484、Sparse `probe_hidden_packed` 的 285–450、FMHA `tcg_d128_flat_tma_xpose_narrow` 的 559–650。这五个分支的冻结状态均为 approved。没有声称全 172 份有设计材料的节点均已语义审阅；全语料仍区分 145 approved、13 review_candidate、14 needs_authoring，以及 5 scaffold、43 未开始。

源码回查涉及固定副本中的八份文件：GDN `gdn_kernel.cu`，GEMM `fp8gg_umma.cu`/`fp8gg.cu`，MLA `solution.py`，Sparse `dsv4_kernel.cu`，FMHA `fmha_tcg.cuh`/`fmha_mxfp8_qkv.cu`，MoE `source_nvfp4_moe.cu`。具体阅读范围在 [JSON 的 coverage](/home/shaohanyong.shy/KernelSchema/schema_discussion/phase_grammar/data/04_prose_gaps.json)。另外机械核对全部 18 份 corpus 源码与六 root 的 SHA256；均相符，未新增源码副本。读源码验证发生了什么，并不等于完成 ISA 保证、GPU 正确性或所有失败路径的验证。

| 算子、根稿状态 | resource / domain / milestone 条目 | 旧统计最长动作契约 | 本次关键判断 |
| --- | ---: | ---: | --- |
| [GDN root](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml)，approved | 14 / 3 / 1 | 1,300 字符 | 两处别名已有视图；扫描与 Q/K 搬运的独立推进要留；final-mini 的来源缺保证不能靠格式修复。 |
| [GEMM root](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gemm/root.yaml)，approved | 9 / 5 / 0 | 648 | 源槽归还与全局写完已有 release/store 表达；跨 tail 的历史读者集合可短写。 |
| [MLA root](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/root.yaml)，approved | 15 / 3 / 0 | 573 | TMEM/寄存器副本必须绑定；连续域和几何双槽有用；host 建表/上传在调用边界。 |
| [Sparse root](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/root.yaml)，approved | 15 / 1 / 0 | 1,266 | prepare 里有真实别名退休和异步完成区别；分支的外部 fork/join 不能由内部 launch 推出。 |
| [FMHA root](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/root.yaml)，approved | 9 / 2 / 0 | 619 | PV 的 enqueue-only 契约可用当前模式与等待改写；额外动态 SMEM 请求归实现预算。 |
| [MoE root](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/root.yaml)，approved | 20 / 9 / 0 | 680 | 条件等待已能表达；四个独立 epilogue 读者不能被 alias/通配符抹掉；旧 host mirror 是来源问题。 |

条目数来自共同 `root_fields.json`，按各 phase 的语义使用位置相加，仅作定位；不是作者重复输入量，也不能跨 kernel 累加成同时驻留量。最长值直接复用旧报告的归一化字符串统计，不是本次测出的纯自然语言字数。根稿批准状态只标识固定材料身份。

**按当前规则判断，而非按旧词名判断。** 本轮依据 sources/rules 的当前快照。八个基础 verb、逐 noun 的 `execution_mode/describe`、列表式 `for/if`、`waiting`、明确的 resource/view 绑定已经改变了旧长文的一部分成因。[docs5 132–192](/home/shaohanyong.shy/KernelSchema/schema_discussion/phase_grammar/sources/rules/5.design.tile_pipeline.md:132) 不再需要一个自定义 verb 为许多 noun 写说明书；[479–486](/home/shaohanyong.shy/KernelSchema/schema_discussion/phase_grammar/sources/rules/5.design.tile_pipeline.md:479) 也不从同名自动生成数据依赖。迁移旧稿时必须重新核对输入和前驱，不能只机械改名。

| 长动作中的事实 | 当前应放在哪里 |
| --- | --- |
| 存放层级、容量、槽数、共享区域 | `resource` 的 location/shape/depth/total_size 与 `view`；不靠正文再定义另一份区域。 |
| 读写来源、版本、独立副本 | noun 的短 `describe`，点名生产动作和 resource/view；noun 同名不建立绑定。 |
| 哪些 warp 负责、是否独立推进 | role.member；确实影响工作量/交接的组内分工短写；需要分别引用边界时再拆 role/动作。 |
| 共享索引、跨工作连续序号 | `domain`；单 role 且视图不用的局部遍历写在 `for`。 |
| 空工作、条件改变等待位置 | `for/if`；已存在的无 payload 动作与不存在的动作区分。 |
| 完成、数据可读、全部旧读者结束 | 基础 verb 的 done、逐 noun 的 mode、`waiting`；有依据的原语内部顺序仍可短写。 |
| 跨 phase 引用局部时点 | `milestone` 配合 `handoff.ready/wait_at`；整 kernel 的完成直接引用内建事件。 |
| CPU/上传、caller 前后工作、跨调用分配 | docs4/调用契约或必要 realization；不伪造局部资源和局部事件。 |

以下十项按 METHOD 给出一个主类。混合段落的局部删减并不意味着整字段可删；所有原文、源码和规范引文均在 JSON 中逐条保留。

**PG04-01 — 能删的是线程 lowering，不能删掉独立推进（implementation_detail）。**

GDN [root 498–508](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml:498) 写：

> One lane issues four Q/K TMA boxes. Threads 0..63 issue … Warps 0..1 wait for their own copy groups and join warps 4..5 … Warps 4..5 can then compute b/rho without waiting for Q/K.

原文省略部分见 JSON。关键不是具体 lane 和 barrier 编号，而是扫描组在 g/beta 写者发布后即可开始，不等 Q/K；Gram 前再汇合整个 CTA。源码 [338–389](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu:338) 确实把扫描分支与 `mbar_wait` 分开。保留参与组、独立等待和最后汇合即可，不必抄 `bar.sync 1/2`、线程地址或 shuffle 实现。ratio=8 的多遍历会改变任务推进，不能与编号一起随手删除。

同一删法也适用于 [final_store 661–669](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml:661)：可保留“从 stage 0 转置写出，并把下一片读取与当前片写出重叠”，删除 float4 临时变量交替、末轮把未用值赋回寄存器的讲解。源码 [2855–2864](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu:2855) 的最后 `cur = nxt` 不产生额外输出或下一片访问。

反过来，Sparse [prepare 444–460](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/root.yaml:444) 中的 Q_ring 先读完再让给 KV、Early 决定首块发起位置、空 copy-group 仍参与计数、prepare 不保证 KV 落地，都应保留。源码 [1150–1173、1187–1206](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_sparse/dsv4_kernel.cu:1150) 逐项支持。这些会改变安全或重叠，不能用“原语细节”一概删掉；可用现有动作和短契约解释。

**PG04-02 — 最长 finish_chunk 中的一部分是来源缺保证（source_or_draft_issue）。**

GDN [root 655–660](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml:655)：

> In a final mini chunk with (c_n-1)%NSTAGE=0, stage-0 writes can race other warps' outstanding Kc row-0 reads; the source supplies no read-retirement guarantee.

`RhsOutput` 的 Hrhs/Out/DU0、`Stages` 的 chunk 视图和 FinalState 已在 [565–586](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml:565) 明确共址。源码 mini 路径在 [2499–2511](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu:2499) 的 DU0 join **之后** 才读 K，随后 [2550](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu:2550) 直接 continue；最终 [2828–2839](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu:2828) 先写 stage 0 后 join。非末轮有下一 loop-top join，full 路径另有 K preload 后的 join；不能把这些依据移植到 final-mini。

应留该条件下缺保证的一句限制，去掉“合并为一动作也不能自动排序”的重复教学句。不要用一个 `release` 名称或 serial 外壳冒充全部 warp 的读者退休。对照 FMHA 已批准 xpose 分支：[604–612、642–649](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/tcg_d128_flat_tma_xpose_narrow.yaml:604) 已把需要的 `prepare → stage_partials` 复用边与源缺 join 分开。源码 [5412–5424](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_fmha/source/candidate/fmha_tcg.cuh:5412) 的屏障确在 staging 写之后。现有语言能表达需要的边，不能替实现兑现它。

**PG04-03 — 全部历史 store 的源读取，一句话足够（concise_semantics）。**

GEMM [root 442–446](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gemm/root.yaml:442)：

> Staging(t,f) instead waits, through the bulk source-read completion primitive, for every outstanding earlier full-tile store to stop reading OutputStaging, then rendezvous all epilogue members.

这一句说明了不能从 shape 推出的读者集合，值得保留。源码 [1017–1018、1033–1055](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gemm/fp8gg_umma.cu:1017) 在重填前等 `wait_group.read 0`，末尾才等 `wait_group 0`；tail 分支不清 `prev_store`。full→tail→full 时不能只等上一 worktile 的 store，也不能改等目标写完，否则会丢掉重叠。

`release:DotPair` 与 `release:Staging` 的不同含义现在各有 noun 条目，不必挤进一个 verb 段落。另一个反例是 [mma_sync 158–168、201–209](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gemm/mma_sync.yaml:158)：Output 已是 InputRing 的 slot 0 BF16 view；源码 [263–296](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gemm/fp8gg.cu:263) 先 drain，再 overlay，再汇合读者。可删“first input slot”的重复解释，不能另添独立 Output 分配。这里不需要新 `read_done` 事件。

**PG04-04 — 绑定动作的输入副本是合理的短语义（concise_semantics）。**

MLA [root 479](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/root.yaml:479)：

> Scores(n) reads both halves of S(n) into ScoreRegisters; done releases the score read for subsequent QK slot reuse. The second-half transfer overlaps first-half masking/max work.

`S(n)` 同时出现在 ScoreProbability 与 ScoreRegisters，`P(n)` 同时有 TMEM view 与 PackedRegisters，[387–412](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/root.yaml:387) 记录的是实际不同存储。源码 [918–945](/home/shaohanyong.shy/KernelSchema/schema_discussion/phase_grammar/sources/kernel/extract_mla/solution.py:918) 确实读进独立 `sc` 寄存器；[1090–1100](/home/shaohanyong.shy/KernelSchema/schema_discussion/phase_grammar/sources/kernel/extract_mla/solution.py:1090) 则把 packed P 写回 TMEM 后发布。不能因逻辑值同名就合并两块资源；也不能仅用 `load:Scores` 让读者猜源。

当前 `describe` 已明确负责这一绑定。保留“从 ScoreProbability.S(n) 到 ScoreRegisters.S(n)”及两半加载的必要重叠即可。另留 [569–571](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/root.yaml:569) 的先读旧 P 再写新 QK 的原语保证；同一 issuer 的实际调用见 [1273–1330](/home/shaohanyong.shy/KernelSchema/schema_discussion/phase_grammar/sources/kernel/extract_mla/solution.py:1273)。这条保证不能只从 done 的累积顺序推出。无需把全部输入组成再变成一个新的公共表。

**PG04-05 — enqueue-only 的旧完成定义已有结构解（existing_structure）。**

FMHA [root 614–621](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/root.yaml:614)：

> done denotes enqueue only; retire/drain own completion.

当前 `matmul.done` 不能如此降格。可以将 PV 写成 async matmul，在原 `retire:Previous` 处用 await 等上一 PV，在循环后等最后 PV。源码先做当前两半 QK/readout，[4899–4921](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_fmha/source/candidate/fmha_tcg.cuh:4899) 才等待旧 PV 并复用 V 槽/ValueScale；[5094、5121–5128、5173–5178](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_fmha/source/candidate/fmha_tcg.cuh:5094) 证明发起、中部退休、最终退休的位置。不能在 PV 发起处插入同步等待，也不能因为 QK/PV 同属 Main 就断言它们有共同的累积完成队列。

这是当前模式与 `waiting` 能处理的旧稿迁移问题。公共域中的 `z: [0,2]` 仅服务一个 role 的 half 遍历，资源 view 不使用 z；可以改为 `for(z=0:2)`。g 则仍负责 Key/Value/atom 的轮转视图，应继续公共登记。

**PG04-06 — 实际容量不能和缓存 bookkeeping 一起删（other_layer）。**

FMHA [root 688–698](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/root.yaml:688)：

> launch_tcg requests TcgSmem<128,PASS_FUSED,1>::bytes=102912 before tmem_safe_smem applies its runtime occupancy-based request adjustment.

本 root 登记的 live SMEM 是 `16384 + 2*16384 + 2*16384 + 2*512 + 2*512 = 83968` 字节；额外预留使初始动态请求达到 102912。这是按 [466–494](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/root.yaml:466) 的资源形状计算的单例对照，并非全语料容量审计。源码 [TcgLayout 1822–1852](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_fmha/source/candidate/fmha_tcg.cuh:1822) 和 [helper 817–838](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_fmha/source/candidate/fmha_mxfp8_qkv.cu:817) 证实多余区域与占用调整；[1183–1195](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_fmha/source/candidate/fmha_mxfp8_qkv.cu:1183) 把返回请求用于实际 launch。

可删缓存成员、error clearing 和逐次重试讲解，不能删实际每 CTA 请求与 256 TMEM columns 的约束原因，也不能把 helper 的返回值写成驻留保证。`resource.total_size` 明确只算登记的设计存储；不要登记虚构的 UnusedP 值来凑实际请求。短 realization 绑定实际请求已够用；若以后需要机器化预算，这是 launch/realization 接口问题。

**PG04-07 — 条件等待和四个读者已经能写清（existing_structure）。**

MoE [root 761–763](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/root.yaml:761)：

> Hoist=1 when R>=16*E, otherwise Hoist=0. Drain(t,q) completes after its explicit predecessors; q identifies the optional wait before the first scale copy of a worktile.

旧稿随后已经有 q 的 0/1 循环和四个 release 前驱；当前直接使用 if 即可。源码 [4855–4906](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:4855) 确实在 hoist 分支先拷 scale、再等待旧 epilogue；关闭 hoist 则先等。等待位置决定跨工作重叠，不能仅把“等待上一块”留在一句总述中。

Epilogue、Epilogue0、Epilogue1、Epilogue2 是一个 CTA 中的四个 warp role。[814–848](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/root.yaml:814) 与源码 [4431–4435、5143–5155](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:4431) 分别给出四次 release 和最后读完后的 arrival。不能用 `Epilogue[*]` 替换四项：当前 member 选择的是 worker 的 CTA 成员，不是 warp 编号。

alias 已经节省了这些合同的重复输入，但没有减少四个读者。类似地，GEMM [wide_recovery_full 150–169、252–254](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gemm/wide_recovery_full.yaml:150) 用 alias 共享 phase 正文，仍明确选择四次实际计算/覆盖输出的提交条件；源码 [1757–1762、1802–1816](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gemm/fp8gg_umma.cu:1757) 保留 fallback 提交。这里不证明所有 API 失败组合可达，只说明不能用字段内容共享隐去该分支选择的实际工作。跨 kernel 的同名资源也不是同一片上分配。

**PG04-08 — 上传与跨调用 workspace 应保留在正确层（other_layer，边界待定）。**

MLA [root 332–336](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/root.yaml:332)：

> Upload the complete Geo table on the side stream and record table_event; the main/current stream waits for that upload before the one main kernel.

源码 [1669–1681](/home/shaohanyong.shy/KernelSchema/schema_discussion/phase_grammar/sources/kernel/extract_mla/solution.py:1669)、[1895–1898](/home/shaohanyong.shy/KernelSchema/schema_discussion/phase_grammar/sources/kernel/extract_mla/solution.py:1895)、[2011–2024](/home/shaohanyong.shy/KernelSchema/schema_discussion/phase_grammar/sources/kernel/extract_mla/solution.py:2011) 给出 D2H/host 等待、H2D/event、主流 wait 后 launch。不能删成“kernel 读取 Geo”：先启动占用资源、到首次读表才等，是另一种安排。应在 docs4/调用前提保留主核的启动下界及元数据版本；`milestone` 只能引用本 phase 的局部动作，不能把 `table_event` 填进去就视为已有端点。

GDN [root 380–383](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml:380) 已经能用三句说明同一 workspace、消费槽每次重写、只靠同流复用且无跨流排他。源码 [3254–3274](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu:3254) 有 process-static 缓存、设备/容量条件和当前流。无需把分配器、L2 查询缓存和 host buffer 管理都结构化。需要另议的是：调用级哪些前提/共享身份必须显式导出；不是扩大 docs5 的资源 location。

**PG04-09 — 外部 fork/join 的关系存在，局部 milestone 不负责创造它（other_layer，边界待定）。**

Sparse [probe_hidden_packed 300–305](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/probe_hidden_packed.yaml:300)：

> The remainder need not wait for the prefix. Queue a side-to-caller join after the remainder and behind the prefix, so subsequent caller work observes all output rows.

内部 flow 只有 probe、prefix、remainder，后者等 probe。源码 [6153–6163](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_sparse/dsv4_kernel.cu:6153) 先用事件把 side 接在 caller 已排队的输入工作后；[6188–6191](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_sparse/dsv4_kernel.cu:6188) 则把 caller 后续工作接在 remainder 后。caller 自身的流序另覆盖 prefix。该 join 不阻塞 host 等输出，也不要求 prefix→remainder 串行。

这构成需要在 docs4/调用层讨论的清楚反例，但不是一个新的 phase 计算、数学合并或片上 milestone。正文可以压成上述必要关系。root [557–571](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/root.yaml:557) 的十个指针缓存键、失效/计数等 bookkeeping 可交 Coder；首次 capture 的 caller-stream completion 不能一起删，源码 [6384](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_sparse/dsv4_kernel.cu:6384) 确实调用了 stream synchronize。是否为这些外部端点加结构，应在正确层另定，当前保留短策略句。

**PG04-10 — 旧 host mirror 不能改名成当前结果（source_or_draft_issue）。**

MoE [verify_only 149–153](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/verify_only.yaml:149)：

> Source limitation: the mirror has no request-generation tag and its CPU reset does not await prior GPU writers, so HostObserved may come from an earlier verification …

源码 [9480–9488](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:9480) 由 CPU 写 -1、发起 clean graph、轮询；此前 GPU 仍可能在 [1770–1772 或 1784–1786](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:1770) 写同一 mirror。CPU 写先于本次 launch，不证明晚于上次 GPU 写。

应保留一条来源限制，区分当前 Clean、host 实际观察与 kernel_done。不能只增加 `generation` 或 `host_ready` 字段就把来源说成安全。也不能把同流的 device 访问顺序推广到 CPU sentinel reset。这里有真实共享存储问题，但没有可从语言中补出的源码正确性。

**三个最小改写草图。** 均为建议片段，不是已提交的实例，不是整段截断后的合格 plan；JSON 保留保留/删除事实清单。

1. GEMM 将 `release:Staging` 的必要差异与 `store:Output` 分开。原 pipe 中“release→填入 Output→store”、tail 分支和 worker 末尾 drain 保留。DotPair 的成员时序不由本片段改写。

```yaml
using_verb:
  release:
    Staging:
      execution_mode: serial
      describe: "Return OutputStaging after all earlier full-tile stores have finished reading it, then rendezvous all epilogue writers."
  store:
    Output:
      execution_mode: async_unordered
      describe: "Copy OutputStaging.Output(t,f) to this worktile's owned out region; done guarantees the global writes are visible."
```

保留的是全部历史源读取、所有写者、目标区域和较晚的全局完成；删除的是“第一块没有历史”“下一块可能重叠”等能由已有结构/该句推导的重复解释。没有增添源读完事件种类。

2. FMHA 只替换 PV、上一组等待、最终等待三个声明及相应边。PreviousPV 仍位于本组两个 QK/readout 之后、V refill/ValueScale/rescale/Prob8 写之前。`compute:StateProb(g)` 是旧 advance 的对应动作，其他输入绑定与就绪边在完整迁移稿保留。

```yaml
using_verb:
  matmul:
    PV:
      execution_mode: async_unordered
      describe: "Apply value_output for group g: read Prob8(g), UnitScale and ValueScale(g) from their TensorMemory views, and Value(g) from its SMEM view; update TensorMemory.NumState from the rescaled state prepared by compute:StateProb(g), except that the first group initializes it without reading the old value. Done covers NumState and all of this PV's operand reads."
  await:
    PreviousPV:
      execution_mode: serial
      describe: "Wait for PV(g-1); at g=cLo there is no previous PV and this action has no data effect."
    FinalPV:
      execution_mode: serial
      describe: "Wait for PV(cHi) before final NumState readout."
waiting:
  released:
    - "{done await:PreviousPV(g)} ← [{done matmul:PV(g-1)}]"
  prepared:
    - "{done await:FinalPV} ← [{done matmul:PV(cHi)}]"
```

首轮移位边由已声明 g 窗口排除，首轮等待动作仍为无数据效应；末轮在循环外明确引用 cHi，不借用退出后的隐含索引。保留旧重叠和全部 PV 读者；删 enqueue-only 的特殊 done 定义、parity 操作讲解。公共 z 改局部 `for(z=0:2)`，g 的资源视图映射保留。

3. MoE 的同一条件直接用 if。以下只展示工作循环与复用边；entry/retire、完整输入就绪边和 noun 模式仍需保留。旧纯复制 prepare 对应 `load:ScaleTM`，不是新基础动词。

```yaml
pipe:
  - "for(t)":
      - "if(t > 0 and Hoist == 0)":
          - "await:Drain(t)"
      - "for(g[t])":
          - "load:ScaleTM(g)"
          - "matmul:PayloadA(g)@PayloadB(g)"
waiting:
  released:
    - "{done await:Drain(t)} ← [{done Epilogue::release:Accumulator(t-1)}, {done Epilogue0::release:Accumulator(t-1)}, {done Epilogue1::release:Accumulator(t-1)}, {done Epilogue2::release:Accumulator(t-1)}]"
    - "{issue matmul:PayloadA(g[t,0])@PayloadB(g[t,0])} ← [{done Epilogue::release:Accumulator(t-1)}, {done Epilogue0::release:Accumulator(t-1)}, {done Epilogue1::release:Accumulator(t-1)}, {done Epilogue2::release:Accumulator(t-1)}]"
```

保留 Hoist=0 时 copy 前等待、Hoist=1 时只在首 MMA 前等，以及四个独立 warp 的 release。删除 q 的假循环、q 实参与解释。这里列表看起来长，恰好因为必须保留四个真实参与者；不能用错 member 通配符求短。

**登记是否过多。** 不能因 MLA 的 S/P 有两份副本、GEMM 有多次独立 kernel、MoE 有四个 epilogue，就合并其资源或事件。重点核对的片上缓冲未出现可有据整体删除的核心分配；这不是证明全 172 稿所有登记都必要。可收缩的明确位置是 GDN prep 的单 role 头组局部遍历、Active 的 0/1 控制，FMHA 的局部 z，以及 MoE 公共 c 在当前展开 pipe/view 中没有使用的登记。当前 docs5 已允许这种收缩，属于使用规则的问题。

milestone 也不需要普遍出现。GDN 的唯一 `prep_ready` 由八条 handoff 的 `wait_at` 共用，[root 331–338、552–553](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml:331) 对应 source [2216–2224](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu:2216) 中 state0 读取之后的 PDL wait：这是有意义的晚等待点，不能用 prep.kernel_done 作为启动屏障替代。其他五 root 没有自定义 milestone 不说明概念多余；内建 kernel_done 和默认交接已满足它们当前选定的跨 phase 关系。不应为每个 value 重复登记同一个事件组。

**英文名称独立建议。** 三个名字不追求形式整齐，分别优先说明内容。

| 当前名 | 推荐名 | 原因与反对的备选 |
| --- | --- | --- |
| resource | **buffer** | 与 depth/location/shape/view 配合，直观表示装值的片上存储，可包括累加器/寄存器状态。resource 过宽；allocation 易被当成分配动作；storage 是合理备选，但更像介质总称，和 location 较接近。 |
| domain | **index** | 键就是被循环/视图共同解释的索引，值给边界或序列。range 容易漏掉 worktile/flatten；loop 错把登记当执行；shared_index 较长，且视图所需公共索引未必被多 role 使用。 |
| milestone | **event_group** | 直接表示已有事件的 all-of 集合。event 更短但易误读为新建 CUDA event；checkpoint 暗示快照/停点；signal 与实际通知动作冲突。 |

改名不改变范围：buffer 不纳入 host/global 分配和 barrier 存储；index 不让 role 锁步；event_group 不制造通知、可见性或外部事件，也不重定义 kernel_done/launch_ready。

**最后判断与优先级。** 本轮十项主类为 implementation_detail 1、concise_semantics 2、existing_structure 2、other_layer 3、source_or_draft_issue 2；没有独立成立的 docs5 missing_capability 或 underspecified_rule。未提 issue 不是无缺口的证明，本判断来自当前规则与具体反例的重新对照。

优先落实三件事：让 describe 每次只承担唯一动作绑定与必要原语保证；把真完成、全部读者退休、条件等待位置落实到现有结构；删掉线程地址、barrier 编号、寄存器临时量和缓存维护说明。保留会改变并行、工作量、容量、安全或版本来源的少数事实。最值得另列需求的是 docs4/调用边界如何引用外部输入就绪、输出 join 和跨调用共享条件；实际 launch reservation 的机器化表示则另议预算接口。没有证据要求现在把这些都扩进 phase 的三个公共定义。

机器交付见 [04_prose_gaps.json](/home/shaohanyong.shy/KernelSchema/schema_discussion/phase_grammar/data/04_prose_gaps.json)。[04_build_report.py](/home/shaohanyong.shy/KernelSchema/schema_discussion/phase_grammar/scripts/04_build_report.py) 生成该 JSON，并检查引用是对应固定行段的原文子串、六 root/十八源码/所选五分支的摘要；草图只做 YAML 解析，不声称完整 schema 验证或语义等价证明。未修改正式文档、算子、模板、运行状态，也未恢复提取或运行 GPU。
