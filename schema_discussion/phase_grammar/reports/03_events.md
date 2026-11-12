# Phase 事件别名调查

建议保留**可选的对外事件别名**，字段名首选 `event`，备选保留 `milestone`。它不是新的同步原语，也不需要把全部内部动作登记一遍。GDN 的八个交接值共用一次延后等待，已经说明这个引用层有用；MoE 的部分入口事件组又确实可以删去，不能把“有事件”自动当成“需要别名”。

在本次核对范围内，没有找到必须给 milestone 增加内部字段才能表达的真实最小反例。应先补清事件实例范围、消费侧事件组展开和空工作入口的规则。MLA 的表上传、Sparse 的调用者 fork/join 则是实在的外部端点需求；当前内部事件别名没有覆盖它们，能写进 strategy 并不证明结构已经足够。

本报告只形成建议，未修改正式规范、算子、模板或提取状态。判断依据是本轮 `sources/rules` 快照，包含新八动词、noun 两字段、for/if、waiting 四端点。归档 YAML 使用历史语法；下面引用它们的执行事实，不把历史批准当成符合今日语法。

**读取范围与身份**

六 root 的完整 work_schedule、tile_pipeline 与 realization 均已读取；下表给出实际阅读范围。这里的“完整”限于这三个相关区域，不声称完成 Contract/数学等价审阅。

| 算子 | root 与状态 | root 覆盖 | 事件事实与额外取样 |
| --- | --- | --- | --- |
| GDN | [root](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml)，approved，v11 | 320–690；2 phase，1 alias | 八值共用 `chain.prep_ready`；另读 [segmented](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/segmented.yaml)，approved v3，检查 pass A/C 的两个等待点和多生产块覆盖 |
| GEMM | [root](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gemm/root.yaml)，approved，v4 | 194–546；1 phase，0 alias | 三 role 在同一 phase 内协作；源槽可复用与目标全局完成分别表达，不需要跨 phase 别名 |
| MLA | [root](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/root.yaml)，approved，v4 | 286–652；1 phase，0 alias | 五 role 的本地队列/数据关系；host 建表上传是计算启动前提，不能伪装成本 phase 的角色事件 |
| Sparse | [root](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/root.yaml)，approved，v13 | 277–575；2 phase，0 alias | root 用 `attention.kernel_done` 启动 combine；另读 [probe_hidden_packed](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/probe_hidden_packed.yaml)，approved v2，检查输入 fork、两路输出与外部 join |
| FMHA | [root](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/root.yaml)，approved，v9 | 383–702；1 phase，0 alias | root 为单 kernel；另读 [tcg_d128_segmented_tma_direct_narrow](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/tcg_d128_segmented_tma_direct_narrow.yaml)，approved v2，五次提交、四个等待别名 |
| MoE | [root](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/root.yaml)，approved，v7 | 365–1026；7 phase，0 alias | 入口 PDL 等待及四 epilogue 条件复用；另读 [double_verify_root](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/double_verify_root.yaml)，approved v1，七个 guard 组，其中两个含七个角色事件 |

分支只审阅事件相关上下文，没有声称逐字审阅整棵分支树。精确阅读行段、冻结状态、文件哈希、证据原文见 [03_events.json](../data/03_events.json)。源码核对涉及九份固定副本：GDN `gdn_kernel.cu`，GEMM `fp8gg_umma.cu`，MLA `sources/kernel/extract_mla/solution.py`，Sparse `dsv4_kernel.cu`，FMHA `fmha_common.cuh`、`fmha_tc.cuh`、`fmha_tcg.cuh`、`fmha_mxfp8_qkv.cu`，MoE `source_nvfp4_moe.cu`；其仓库相对路径、原 logical_path/ref 与实际 SHA256 均在 JSON 中，全部匹配共同 corpus。未新建源码副本。

已完整读两份指定旧报告：[SCHEMA_VERBOSITY](../../../schema_extract/archive/six-operators/20261109-methodology-review/SCHEMA_VERBOSITY.md) 与 [SCHEMA_DISCUSSION](../../../schema_extract/archive/20261109-ten-operator-text-audit/SCHEMA_DISCUSSION.md)。其中的外部边界和条件等待只是定位线索，以下判断均回查 YAML 与源码；未读取本轮其他新代理报告。

机械扫描单列：177 份实存文件中 172 份有设计，排除 5 scaffold；43 未开始节点不计入。这 172 份包含 approved 145、review_candidate 13、needs_authoring 14，不能合称已通过。扫描只数 raw 本地 milestone 字段，不展开 from_parent，不按运行迭代展开，YAML alias 按字段位置计数，不代表作者重复输入量。

| 算子 | 有设计文件 | 含本地 milestone 的文件 | alias 条目数 | 事件组长度 |
| --- | ---: | ---: | ---: | --- |
| GDN | 6 | 6 | 8 | 全部为 1 |
| GEMM | 7 | 0 | 0 | — |
| MLA | 2 | 0 | 0 | — |
| Sparse | 8 | 0 | 0 | — |
| FMHA | 43 | 34 | 102 | 101 个为 1；1 个为 2 |
| MoE | 106 | 32 | 184 | 136 个为 1；48 个为 7 |

294 个条目中没有带索引的 alias 键。这只是覆盖缺口，不能支持删除索引能力；含候选/待创作设计的扫描数也不是成功案例数。可用 [03_events.py](../scripts/03_events.py) 重现此统计和证据切片。

**八个证据点**

**EV01 · 保留别名，但不要合并三个时点（concise_semantics）。** GDN root 331–338 的 Ti、Qd、b、bc、Ai、Dl、Qc、Kc 全部引用 `chain.prep_ready`；552–553 将它绑定到一次 `join_prep:Prepared` 完成。601–609 明确，chain 在等待之前可以读取输入 state，PRE0 路径还先发出原始 Q/K/V 搬运。源码 `gdn_kernel.cu` 2211–2224 实际先 `sreg_load()` 再 `griddepcontrol.wait`，3433–3436 配置程序化启动。这里一个名字隔开上游八条数学交接与局部 role/action 路径，直接引用就要在八处放该内部路径。

FMHA 所选分支 659–668 与源码 `fmha_tcg.cuh` 3463–3474 同样把独立初始化、Q 准备放在 scan wait 前。`fmha_tc.cuh` 1375–1376 更有力：segment_max 先 trigger，再 wait。把 launch_ready 当数据完成会提前读取；把 wait_at 改成整个启动等待会缩掉合法准备和后继调度空间。

| 字段/事件 | 实际约束 | 有无可省部分 |
| --- | --- | --- |
| `start_after` | 整次提交何时允许开始 | launch 列表次序不隐含它；普通跨提交串行仍要写 |
| `ready` | 交接值在哪个生产事件后具备所需可见性 | 默认是相关生产 worktile 完成；不是默认整 grid 完成。PDL 示例中的 `kernel_done` 不宜机械删除 |
| `wait_at` | 消费者的数据必须在哪个实际等待完成/读取发起点前就绪 | 无独立前置工作且入口默认/其他已声明边足够时可省；GDN/FMHA 延后等待不可照删 |
| `kernel_done` | 所属计算 kernel 完成并满足后继可见性 | 同一次 launch 中各 phase 共享它；不用再定义同义本地 alias |
| `launch_ready` | 协议定义的整个前驱 grid 启动许可 | 不承担 payload 就绪；即使实际常靠 kernel 完成触发，也不能一般地与 kernel_done 合并 |

依据为 docs4 257–260、290–295、369–379；docs5 498–504 已明确内建事件不可重定义。单次引用的别名确有间接层成本，但当前材料支持保留可复用名字，尚不支持为了省一个声明再维护一套正式跨 phase role 寻址。

**EV02 · phase 级登记的是事件族，不是一枚全局事件（underspecified_rule）。** 当前 docs5 177 已把动作身份定义为 phase、worker、成员、role、verb、noun 与实参的组合；563–566 也明确 `Role[*]` 只选该 worker 内的成员，成员不是 warp。问题是 milestone 502–504 只说索引同名对应、必须完整写 role/member，docs4 373 再要求能确定作用域，仍缺一段直接说明别名如何随调用和 worker 实例化的规则。

真实数量并不一一对应：FMHA 分支 295–336 中 segment_max 有 `nr*Nseg` 个 worker、segmented_scan 只有 `nr` 个；一个 scan 消费同一行块的多段结果。GDN segmented 284–340 还有本地 `cl` 到全局 `c=lo_z+cl`、所有 G 个 SegA 与部分 SegB 的映射。不能把两端相同索引字母、worker 编号或 shape 当成配对依据。docs4 301–305 已给数学值、版本、区域覆盖这一正确来源，需要把它与事件实例接起来，而不是新增另一张映射表。

另一个边界来自 MoE：double_verify_root 608–626 使 Active=0 时工作集合为空，但 1187–1196 明确七个 role 的入口 guard 仍执行，随后才决定分配和 payload。不能先按消费 worktile 数量为空就把其入口等待删掉。当前 for/if 对动作存在性的规则足以承担它，milestone 需明确沿用。多 consumer 的 `wait_at` 列表若只覆盖部分 consumer，其余 consumer 采用默认入口还是报缺项，目前也应说清，才能安全局部删减。

建议只补三件事：键索引是形参、引用处代入，不捕获某个 role 的当前循环；按 handoff 的值/版本/区域定位实际 worker 与事件；包括 prologue 在内的事件是否存在由 pipe 决定。`start_after` 引用自定义别名时还必须有能约束整个提交的明确覆盖范围。六算子没有提供需要通用 worker 选择语言或任意事件索引变换字段的证据。

**EV03 · 消费侧事件组必须逐事件受约束（underspecified_rule）。** MoE double_verify_root 1366–1368 与 1601–1603 的 `guarded` 各列七个 role 的入口完成，398–404 由 Clean.wait_at 引用。生产侧“所列事件全部发生”可以作为一个 all-of 前驱；消费侧不能只把“这组事件最后全部完成”当成唯一目标。

用该真实组中的两名角色即可说明区别。令就绪为 r，两个等待完成为 c0、c1。若只要求 r 早于 `max(c0,c1)`，仍允许 `c0 < r < c1`；本意要求 r 分别不晚于 c0 和 c1。七成员组同理。最短补充是：**ready/start_after 对前驱集合取全部；wait_at 将就绪约束施加到所列集合中的每个事件。** 成员选择与条件发生域仍使用现有规则，不加 any/all 运算、通知对象或 barrier 字段。

这不是声称当前 MoE 源码存在此竞态。`source_nvfp4_moe.cu` 208–214 的 ZGUARD 每次真实执行 `pdl_wait()` 再读取 flag，YAML 的入口契约也写了该保证；而此节点还有更强的启动边。待补的是对“事件组”的共同解释，避免图或读者仅检查一个组完成点。

**EV04 · 一处真实可删的间接层（existing_structure）。** 同一个 double_verify_root 的 Clean.wait_at 和七个 guarded 定义可以省。374–397 已要求 count 整次提交等待 `verify.kernel_done`，其余提交又沿 count、scan、scatter、gather、fc1、fc2 的 launch_ready 链接。每次 launch_ready 都晚于其所属提交开始，因此 verify 完成传递地早于全部七个 consumer 的入口。Clean 的这个 ready 已被启动路径保证，不需要再用十九个局部事件引用重复施加一次。

源码 9230–9241 在 verify 后的同一 stream 先做 memset，再提交 count/scan/scatter，与该强边一致。删减时保留 Clean 的生产者、消费者、全局存储、`ready: verify.kernel_done`，保留全部启动边和实际 guard 动作。尤其保留 Active=0 仍读 verdict、无 payload 但 grid 已提交的事实。不能删除真实等待原语，也不能把这个删法套到 EV01 的延后等待。

这也限定了“七事件组证明别名必要”的说法：它证明组有真实对象可指，不证明每个现存组都增加了不可替代的约束。若另一个分支确有七个独立消费等待点，保留组名仍能隔离上游与角色划分；本节点先作可推导删除即可。

**EV05 · 本地等待不应塞进 milestone（existing_structure，涉及 resource/domain）。** MoE root 758–795 已把四 epilogue 的完成作为两个明确前驱位置：非 hoist 时的 Drain，以及首个 MMA 的复用。源码 4855–4907 确有 scale-copy hoist 和后置 drain 两种顺序。当前 for/if 能替换旧 `q:[0,min(1,t)*(1-Hoist)]` 的伪循环，不需要增加“条件 milestone”。当前 waiting 右侧本来就是事件列表，多前驱也不是新能力。

GEMM root 439–467 区分源 staging 可复用和目标全局完成；root 30–38 选 `umma_gemm<0>`，对应源码 `fp8gg_umma.cu` 1017 用 `bulk_wait_group_read0`，1055 才做终端 `bulk_wait_group0`。它是 `release`、`store` 和本地依赖的职责。barrier 地址、编号和奇偶位编码本身可以交给实现，但读完源和写完目标的区别不能删。FMHA root 577–593 的旧 `qk.done=enqueue` 与 read_score 内等待，应按今天的动作完成保证、执行模式和 waiting 迁移；它也不是缺一个跨 phase 名字。不能把历史 verb 的语义机械套进新八动词后，再用 milestone 遮住不一致。

**EV06 · MLA 的上传事件确在本层之外（other_layer）。** MLA root 322–336 写清：side stream 把 metadata 搬到 pinned host，host 等待、构造 Geo、上传，并让 main/current stream 等上传完成后才 launch。源码固定副本 1669–1681 实行 D2H 与 side-stream host 同步，1895–1898 上传并 record `table_event`，2011 等待该 event，2017–2024 才提交主 kernel。

当前 `flow.launch` 只登记 main；docs4 27–29 的 launch 是计算 kernel，367–373 的事件地址只来自 phase/内建事件/本地 milestone。上传既不是 main 内的一个 role，也不在已登记的计算 phase 中。给 main 新建 `table_ready` 不能产生该事件；把 await 放进 main 则会把“上传前不能启动”改成“先占 worker 再等”，不是同一设计。

应在调用/调度边界另行决定采用共同前提约定还是外部事件引用。在此之前保留一句精确的“Geo 上传完成后 main 才启动”及必要源码绑定。这里承认的是外边界结构能力尚未闭合，不能以允许 strategy 就宣称已经结构化；也不需要把 host 的分配容量、event 缓存过程逐项加入 milestone。源码侧流 metadata 读取是否已受调用者输入就绪保护，是另外的调用约定核对问题，新名字不能提供这个保证。

**EV07 · Sparse 的 fork/join 不能改写成多一个 phase（other_layer）。** probe_hidden_packed 299–326 与 427–444 需要三条关系：caller 既有工作在 probe/prefix 前；remainder 在 probe 后；caller 后续工作在 prefix 和 remainder 两路后。当前 flow 只表达中间那条。源码 `dsv4_kernel.cu` 6155–6156 record caller fork、side wait，6162 先提交 caller-stream prefix，6188 提交 side remainder，6189–6190 再把 side join 排到 caller stream。

prefix 与 remainder 写不相交输出，没有相互数据交接，也无需前后串行；两者都不等于调用者的后续任务。局部事件别名无法引用未登记的 caller 前驱/后继。应保留外边界需求并单独讨论共同调用约定，不能用虚构 merge、第四个 kernel 或 host 阻塞来满足现有寻址。probe 的 host 观察可早于 probe.kernel_done（YAML 321、532–535），也不能改叫“完成”。这是旧统计报告指出的真实问题，不是 milestone 的事件组缺一个字段。

**EV08 · 名字不能补保证（source_or_draft_issue）。** FMHA 分支 667–673 已刻意区分 scan 数据可见与额外读取合法：PDL wait 不初始化、不扩大 MState 区域；源码 `fmha_tcg.cuh` 3480–3482 在 `cLo>0` 时读取入口状态，没有以非空 split 保护。此处不作“所有实际输入都越界”的新断言；结论是现有来源限制不能靠一个 ready 别名消失。

同样，MoE 最后到达者发布 pinned-host verdict（源码 1760–1789），不是所有 CTA 的普通 output store 都完成。double_verify_root 1007–1018 已区分每 CTA 发布、最终 kernel 完成和 host hint。源码/旧稿有保证边界时，应按来源问题处理；新事件名不得把局部完成、host 观察或源码注释提升成不存在的全局保证。

docs5 500 本身方向正确，可以压成一条通用句：**事件别名只引用已有动作事件；它不创建动作、通知、可见性或初始化，保证由对应动作及绑定原语提供。** 原语参与范围、release/acquire 保证和进展前提必须留在设计/绑定中；barrier 编号、event 分配 API 和普通缓存细节不属于该别名表。无需为此再添 `visible`、`read`、`host_done` 等事件种类。

**三个局部草图**

以下仅供讨论。归档片段使用历史语法；候选使用今天的事件/控制写法，`event` 是建议更名。都不是完整 plan，也未提交正式规范。

1. **GDN：保留复用别名，名称点明消费等待。** root 552–553 的现存写法：

```yaml
milestone:
  prep_ready: ["Main/join_prep:Prepared@done"]
```

建议片段；八个交接值仍各自登记，这里只展示其中两个：

```yaml
event:
  prep_wait: ["{done MAIN[0]::await:Prepared}"]

Ti: {producer: prep, consumer: [chain], storage: global,
     ready: prep.kernel_done, wait_at: chain.prep_wait}
Qc: {producer: prep, consumer: [chain], storage: global,
     ready: prep.kernel_done, wait_at: chain.prep_wait}
```

直接引用的备选需要另加跨 phase 寻址，例如新的 `"{done chain::MAIN[0]::await:Prepared}"`。它可以省一个 alias 声明，却要在此例八处暴露这个长路径；对七角色组则要在上游展开整组。YAML anchor 可以省输入字数，但不提供独立的事件组身份。现阶段不建议同时引入两套正式写法。

2. **MoE：删重复引用，不删 guard。** double_verify_root 398–404 的 Clean 当前有一行七项 `wait_at`；七个别名里又有两个七角色列表。候选保留：

```yaml
Clean:
  producer: verify
  consumer: [count, scan, scatter, gather, fc1, fc2, finalize]
  storage: global
  ready: verify.kernel_done
```

同时删除这七个仅被 Clean.wait_at 使用的 guarded 定义。`count.start_after: verify.kernel_done`、后续启动链和每个实际入口 guard 全部保留。理由是 EV04 的支配关系，不是“别名看起来太多”。若某一节点没有该强边，这个改写就不适用。

3. **MoE：四方条件复用采用现有 if/waiting。** root 785–786 的现存写法：

```yaml
- repeat: {q: [0, "min(1,t)*(1-Hoist)"]}
  steady: ["wait:Drain(t,q)"]
```

在既有 `for(t)` 的内容中可改成以下局部片段，`await:Drain` 声明为 serial：

```yaml
- "if(t > 0 and Hoist == 0)":
    - "await:Drain(t)"
```

它的完成仍受四个原前驱约束：

```yaml
waiting:
  released:
    - "{done await:Drain(t)} ← [{done Epilogue[0]::release:Accumulator(t-1)}, {done Epilogue0[0]::release:Accumulator(t-1)}, {done Epilogue1[0]::release:Accumulator(t-1)}, {done Epilogue2[0]::release:Accumulator(t-1)}]"
```

首个 MMA 的四前驱复用边、prepare 的数据就绪边、初始化与回收仍须保留；JSON 给出包含首个 MMA 边的草图。这里去掉的是人工 q 轮次，没有取消多方等待，也没有创建本地 milestone。两处相同列表仍可用 YAML alias 复用。

**命名建议**

| 当前字段 | 首选 | 备选及取舍 |
| --- | --- | --- |
| `resource` | `storage` | 当前只记片上存储，能覆盖 SMEM/TMEM/寄存器状态；`buffer` 更熟但容易让累加器/持久状态显得只是搬运暂存；`resource` 太宽；`allocation` 偏实现且更长 |
| `domain` | `index` | 条目为共享索引确定身份和有序域，较易猜用途；保留 `domain` 也准确；`range` 容易排除 worktile/flatten，`loop` 错把声明读成执行 |
| `milestone` | `event` | 简短惯用，但定义第一句须说明只是别名；`event_group` 最直白却更长；`milestone` 易让人误读成 phase 只有一次的全局进度；`mark` 暗示插入动作，`sync` 错暗示创建同步 |

实例名也应帮助快速区分：生产者 `x_ready`，等待动作完成 `x_wait`，读取发起 `x_read`。名称不替代显式 issue/done，更不能靠改名解决成员/实例范围。

本次优先级是保留可选别名、补清短语义、按可推导关系删重复，然后另列外部边界需求。没有证据要求把 milestone 扩成事件协议语言，也没有证据足以证明现有表达对未来所有索引、worker 组合完备。
