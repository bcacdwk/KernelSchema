# 外部等待：删除反例与最小写法复核

结论：**正式 docs 不改；修改原表及旧实例，零新增字段。** caller 边界和外部供给的 strategy 出口已落实；“删除 host 操作过程”须收窄为删除不改变设计的过程叙述，保留观察方式、必要阻塞范围及等待位置。
不能把“只有文字可写”直接判为缺字段，也不能把“可写进 strategy”当成原长段已证明必要。

## 阅读范围与判断口径

- 权威是本轮 `rules/0–6`，重点逐段核对 DOC4、DOC5；DOC0 通用身份、DOC1 接口、DOC2 执行单元、DOC3 值与绑定、DOC6 实现交接作为边界依据。
- 主入口为 [SCHEMA_VERBOSITY][V] L80–89、198–221；辅核十算子 [README][A]、[SCHEMA_DISCUSSION][D]、[SCHEMA_LANGUAGE_GAPS][G] 与 [boundary_review][B] 对应段落。
- 定点核四份 YAML：MLA root、Sparse probe_hidden_packed、GDN root、MoE init_no_scan_fused_small_flat；源码均按 `inputs.json`→`flow_review/corpus.json` 定位并复核哈希。
- 四份均是暂停时 approved working 输入；MoE 所在提取仍 paused，不能据此称整树完成或源算子正确。未把 FMHA/MoE 旧 wiki 稿作现行证据。
- 这不是十算子全量语义重审；未运行 GPU，未改正式规则、实例或提取流程，未读取同轮搭档结论。

## 现行规则已经承担什么

| 当前位置 | 已有能力 | 本轮判断 |
| --- | --- | --- |
| [DOC4][R4] L262–266 | 无内部启动前驱的入口排在 caller 前序后；输出各生产区域完成可见后交给后继；不增加内部串行或 host 输出等待 | Sparse 原外部端点反例已被统一边界覆盖 |
| DOC4 L191、266 | 外部供给及调用前提进入相关 phase.schedule.strategy；已有 Geo 上传完成后启动 MAIN 的例子 | MLA 不再需要证明新增 external/service 字段 |
| DOC4 L296–301、374–388 | start_after、ready、wait_at 分开启动许可、数据就绪和实际等待点 | 内部晚等待已有结构，不能统一移到启动前 |
| [DOC5][R5] L185–200、500–507 | noun.describe 绑定输入、版本、来源与原语保证；已命名动作的必要依赖必须进 waiting | 不允许借“短契约”藏掉已有图边 |
| [DOC1][R1] L43–56；[DOC3][R3] L73–108、145–157 | 外部输出身份、值版本、保留与提供者各有位置 | binding 或 retention 不自动产生外部完成事件 |
| [DOC6][R6] L5–13、40–46 | 绑定实现已声明边界的原语，不能在这里重新决定依赖 | 不把整段 host 流程挪进 choice 来冒充精简 |

旧报告的 `protocol`、`milestone/domain/info.verb` 不作为今天漏洞；下文保留历史字段定位，现行表达使用 `event/index/using_verb`。

## 四种“等待”必须分开

| 类别 | 本轮实证 | 必须留下的事实 |
| --- | --- | --- |
| 整个提交的启动等待 | MLA 完整 Geo 上传→main 开始 | 指定提交、完整数据及可见性；不能换成 kernel 内读前等待 |
| 动作消费等待 | GDN chain 独立准备后才等 prep | 前面可执行的输入工作、实际 await 点、后续才能读的准备结果 |
| host 阻塞 | MoE 冷建表后的 device synchronize | CPU 等待范围含其他 stream，成功后才发布 ready 并继续 |
| host 观察/数据就绪 | Sparse 有界 poll 后最终 scan；remainder 仍等 probe.kernel_done | 观察不保证 GPU 完成；部分观察也能决定选择 |

## A. MLA 外部上传：现行已够，启动边不能删

[MLA YAML][MY] L321–338 是主报告点名的 1,174 字符 strategy；[源码][MS] L1669–1681 做侧流 D2H 并等待该侧流，L1895–1898 上传并记事件，L2010–2024 在 main 流等待后提交主核。
L1638–1645 的源码注释说明侧流供给可与 caller 已有 drain 重叠；本轮核实提交位置，不采用其中测量数字作新性能结论。
删除反例：只留下“读到当前 Geo”，允许先驻留 main 再等表；数值可以相同，驻留与重叠窗口已变。因此“完整上传并可见→main 开始”不能删。
反过来，当前 metadata 的数值要求已在 YAML L114–115，Geo 坐标在 L298–299；不能把“防止旧表”当成 L332–333 独占的信息。
**可安全缩短的真实子句**（仅替换 YAML L332–333，其他 strategy 内容不动）：

```text
原：Upload the complete Geo table on the side stream and record table_event;
    the main/current stream waits for that upload before the one main kernel.
短：Queue a device wait for the complete side-stream Geo upload before the main kernel starts.
```

归一化长度 146→90 字符。保留了侧流、整表、设备等待、主核启动位置；只删具体事件名与记录过程，不新增 host 阻塞。
其余保留去处：metadata→Geo 的 host 供给、D2H 范围及传输规模仍在 strategy；Geo 内容与排列仍按已有定义；本轮不删排序条件或缓存参数来源。
外部录图时 host 前缀执行、回放用已录绑定及实际读到的表内容（YAML L334–336）也不能改为“每次 replay 重建”。DOC4 L248 只统一说明录图不等于计算，具体绑定事实仍需保留。
准备过程的输入就绪另核：源码侧流 D2H 前没有 caller fork。MAIN 的入口排序不能倒推该 D2H 已排在 caller metadata 写后；适用域须有 metadata 已就绪的依据，不能凭新字段补同步。

## B. Sparse caller 回接：已解决，不再逐例复述默认边界

[Sparse YAML][SY] L298–311 的三个提交，与 [源码][SS] L6155–6163、6188–6191 完整对应：caller fork→probe；prefix 在 caller 流；probe 后提交 remainder；最后回接 caller。
最小关系为 caller-prior→{probe,prefix}，probe.kernel_done→remainder，{prefix,remainder 的相关输出完成}→caller 后续消费；没有 prefix→remainder 的完成边。
当前 DOC4 L264 已提供两端，flow 提供中间边，输出区域由 decomposition/target 给出；加入空 host phase、caller 事件类型或新 boundary 表没有必要。
**另一处可缩短的真实文字**：YAML L300–305 的 thesis（464 字符）可改为：

```text
Overlap the full-batch probe with the packed prefix; choose the disjoint remainder from R and run it after probe completion.
```

短句 124 字符。三个实际提交、probe→remainder 边和两段输出范围仍保留，caller 接入与回接依赖由现行共享规则承担。
非阻塞侧流避免 legacy stream 引入额外串行的必要机制仍保留于 realization（L1036–1046）；不是把全部事件/stream 条件删除。
错误短写反例：把 prefix.kernel_done 加给 remainder，会损失尾部接续与重叠；“返回前等待全部输出”则增加 CPU 阻塞。两者都不等价于原回接。

## C. GDN 的真实晚等待：现结构能保留，但它是内部依赖

[GDN YAML][GY] L326–338 已有 prep.launch_ready→chain 启动、prep.kernel_done→chain.prep_ready 消费等待；L601–609、670–674 把独立输入工作放在 join_prep 前。
[源码][GS] L1983–1987 初始化 barrier，L2011–2024 在 PRE0 条件下直接发起输入 Q/K/V 搬运，L2211–2224 先装入初态再执行 griddepcontrol.wait；后者才保护 prep 产物。
prep 的提前启动许可有真实触发（源码 L421），launch 能力由 L3431–3438 绑定，不能把许可视为 payload 就绪。
最小保留：start_after: prep.launch_ready；各交接值 ready: prep.kernel_done；wait_at 指到 chain 中 await 的 done；await 前的独立输入动作与条件保留。
现行 event 可把 `{done MAIN[0]::await:Prepared}` 命名为 prep_ready，pipe 使用有序动作；无需恢复 protocol 字段，也无需新增“晚等待”字段。
如果删 wait_at，DOC4 L299 的默认 worktile 入口等待会前移；如果改 start_after 为 prep.kernel_done，连启动及前置准备也被阻塞。两者都改变重叠而不必改变数值。
此例**不能证明外部 producer 需要一个新可引用端点**：prep 是已有计算 phase。已读案例未给出一个“kernel 内等待外部上传”的实际同源样本，不将这个内部例子冒充该证明。

## D. Sparse host 观察：不能按“host 过程”整段删

源码 L5582–5614：同一 mapped gate 先写 -2，提交 probe，最多 2,000,000 次检查，随后另做普通 FP32 scan；仅对实际读到的正值用 FP64 累加并转回 FP32。
YAML L124–148 已定义 observed 与 R；L315–326 补充提交时点，源码 L6162–6163 证明 **prefix 已提交，才开始 probe_end 的 poll/scan**。
不能删除的两项：观察可以不完整；观察放在 prefix 提交后。第一项改变 R，第二项改变 host poll 与 prefix 的重叠机会。
具体反例：最终 scan 只见一个正值 1，其余仍为 -2，可得到 R=1 并接受；若强制等齐且八个最终样本为 [1,4,4,4,4,4,4,4]，均值 3.625 会拒绝。这里是静态允许的观察对照，不是运行所得样本。
即使选择同一实现，把 poll 移到所有入口提交之前，也会把 prefix 的执行窗口推迟；因此不能用“外部条件都就绪后再启动全部工作”替代。
最小事实去处：observed/R 的来源、正值过滤与精度留在 math；“prefix 提交后观察，结果只约束 remainder 的选择/提交”留在相关 schedule.strategy；probe.kernel_done→remainder 保留在 flow。
可删 API 名及逐条调用叙述，但须保留“每轮重新观察映射值、退出后另作普通 scan”的区别；不得删 -2 初态、有界退出与最终非共同快照，或把超时自动写成 R=-1。
gate 是 process 共享存储、poll 退出不保证旧 GPU writer 结束（L1027–1033）；保留一句源限制即可，不能添加 generation 或每调用私有 gate 来假装原源已安全。
本例不需要把 host scan 伪造成 GPU role；R 被 host 消费，不构成 probe→attention 的 tensor handoff。源观察风险也不等于语言必须建立通用 runtime。

## E. MoE 全设备 host 阻塞：反对一刀切删除，不支持新模块

[MoE YAML][OY] L871–881 保存真正的全设备 host 等待；[Python 源码][OS] L1005–1016 在 ready-set miss 时提交 moe_init，执行 torch.cuda.synchronize(index)，成功后才 add(index)。
[CUDA 源码][OC] L6618–6621 确认为一个 k_init_tables 提交；Python L1166 表明该前缀先于后续构建。选中路径虽不消费这些格式表，实际建表工作仍存在（YAML L239–255）。
反例：另一 stream 有长任务，init_tables 先结束。仅保留 prepare.start_after: init_tables.kernel_done 会允许 host 提前继续；原实现仍等待整个设备。这是等待范围与 CPU 接续的差异。
最小写法置于 init_tables.schedule.strategy：

```text
After init_tables is submitted, the host waits for device-wide completion; only success permits readiness publication and continued request setup. This prefix is outside private pipeline graph replay.
```

保留 init_tables 的实际提交、全设备范围、host continuation、成功发布与私有 replay 边界；具体 synchronize API 只在需要绑定机制时留于 require，删重复过程与理由。
这条短约束足以排除“只等本核”“只排设备边”“提前 ready”三种替代。新增 host_completion/event 表可以便于未来组合检查，但当前例子未排除现有短说明方案。
按最新 DOC6 L11，边界事实不宜只在 choice 才被定义；这是实例归位，不是新增字段的理由，也不把历史 YAML 的旧合法写法算作当时错误。

## 可直接整合的分类与建议

| 分类 | 清单 | 建议 |
| --- | --- | --- |
| 现行已够 | Sparse caller 接入/输出回接；MLA Geo→主核；GDN 内部晚消费 | 采用现行默认、strategy、ready/wait_at，零新模块 |
| 实例错误/归位 | 未确认新的计算错误；MoE 全设备 host 边界的详细定义集中于 require；MLA 外部 D2H 输入就绪不能由 kernel 入口推得 | 前者按当前分层写一次短前提；后者核适用域，不补造同步 |
| 需规则澄清 | 无必要增量；点名作用点或 host 接续只是可选阅读提示 | 现行调用前提与说明职责已足够；正式 docs 不补条款，只修原表的删除口径和旧实例 |
| 真缺字段 | 本轮五类案例均未证明 | 新增字段清单为空；不把可组合 host 图的未来诉求当当前必要性 |
| 来源缺陷/范围 | side D2H 未建 caller 同步；共享 gate 的旧 writer 竞争；未运行核 | 保留精确限制；不推导授权域外失败，也不制造安全保证 |

建议替换原表为：**正式 docs 不改，不加外部执行模块。用现行调用边界与相关 strategy，保留外部对象/条件、实际启动或消费点及必要 host 接续；内部消费等待仍用 ready/wait_at。删 API 步骤与重复解释，不删改变观察、版本、初始化、参与者、等待范围或重叠的事实。**
必要规则增量为零。前述“补一句作用点或 host 接续”没有排除新的歧义，仅重复 DOC4 L191、266、394 的调用前提与说明职责，因此撤回该 docs 建议；实例只保留少量明确关系，不复述共享 caller 边界，也不搬运整段长文。
上述两处短写合计局部声明 610→214 字符；这是已核事实的局部候选，不是全算子压缩比例，也未宣布整份迁移后通过独立等价审阅。

[V]: ../../../../schema_extract/archive/six-operators/20261109-methodology-review/SCHEMA_VERBOSITY.md
[A]: ../../../../schema_extract/archive/20261109-ten-operator-text-audit/README.md
[D]: ../../../../schema_extract/archive/20261109-ten-operator-text-audit/SCHEMA_DISCUSSION.md
[G]: ../../../../schema_extract/archive/20261109-ten-operator-text-audit/SCHEMA_LANGUAGE_GAPS.md
[B]: ../../../../schema_extract/archive/20261109-ten-operator-text-audit/gap_review/boundary_review.md
[R1]: ../rules/1.contract.algorithm_requirement.md
[R3]: ../rules/3.design.math_reshape.md
[R4]: ../rules/4.design.work_schedule.md
[R5]: ../rules/5.design.tile_pipeline.md
[R6]: ../rules/6.realization.md
[MY]: ../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/root.yaml
[MS]: ../../../phase_grammar/sources/kernel/extract_mla/solution.py
[SY]: ../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/probe_hidden_packed.yaml
[SS]: ../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_sparse/dsv4_kernel.cu
[GY]: ../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml
[GS]: ../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu
[OY]: ../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/init_no_scan_fused_small_flat.yaml
[OS]: ../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_solution.py
[OC]: ../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu
