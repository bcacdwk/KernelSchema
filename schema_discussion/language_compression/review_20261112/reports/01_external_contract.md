**结论：修改原建议，保留“不加新模块”；现行规则已能表达本批关系，正式 docs 无需修改；新增字段 0，删除字段 0。**
DOC4 已有 caller 入口、输出可见性和 Geo 外部供给的共同约定，旧 MLA F18 / Sparse F05 不再证明缺少必要声明位置。
应把“只写 strategy”限定为**共同约定和既有结构尚未覆盖的外部关系**；既有内部依赖不搬进文字，排列规则也不随 host 过程删除。
但外部 Geo、caller、device-wide drain 仍不是 `start_after` / `event` 可解析的端点：人读契约可完整，外部依赖不能据此成为机器可展开的图。

本报告只核必要性与可表达性，不修改正式规则和算子，不评估分支复用、继承或 root/dispatch 组织。
规则以本轮 `rules/` 为准；YAML 与源码均取 `inputs.json` 指向的固定 corpus。旧审计只定位问题。
MLA root v4、Sparse root v13 / hidden packed v2、GDN root v11 均为 approved；MoE 首次建表实例为 paused working corpus 内的 approved v1。
FMHA/MoE 暂停不等于其全部节点获批；未把候选、未完成或骨架当成已成立设计。

**现行能力与旧实例处理。**

| 路径或规则位置 | 判断与最小处理 |
| --- | --- |
| DOC4 调用边界 [264–266](../rules/4.design.work_schedule.md:264) | 已定义入口等 caller 前序、输出区域完成且可见、无需 host 等输出；不在独立提交之间加边。 |
| `work_schedule.phase.<p>.schedule.strategy` [191、266](../rules/4.design.work_schedule.md:191) | 保留外部供给的条件、作用提交/消费点、必要版本及可见范围；共有入口/出口不逐例复述。 |
| `flow.launch[].start_after` [256–260](../rules/4.design.work_schedule.md:256) | 保留内部启动关系；不塞入未定义的 `Geo.ready`、`caller.done` 或虚构 host phase。 |
| `flow.handoff.<v>.ready / wait_at` [295–301](../rules/4.design.work_schedule.md:295) | 保留跨 phase 的就绪点和真实消费点；不能用统一启动等待代替。 |
| `phase.<p>.decomposition.strategy` [143](../rules/4.design.work_schedule.md:143) | 本 phase 原子合并的初值、建立者与时机可在此短写；不因其由 memset 建立就删除。 |
| `tile_pipeline.phase.<p>.role.<r>.info.using_verb.await.<n>` 与 `pipe` | 已有设备内等待才登记实际动作；`describe` 写保证，`execution_mode` 与 pipe 固定局部位置；不把 host event 伪装成动作。 |
| `realization.require[].choice` [5–13、40–46](../rules/6.realization.md:5) | 只绑定上述关系所需的机制，例如独立队列和不阻塞 host 的 join；不在此首次决定依赖。 |

去重复核后，原拟增补的“点明外部条件作用点、范围，保留晚等待”仅是可选阅读提示，没有增加必要语义。
外部供给/调用前提已有 [191、266](../rules/4.design.work_schedule.md:191)，说明字符串职责已有 [394](../rules/4.design.work_schedule.md:394)，启动/就绪/消费区别已有 [388](../rules/4.design.work_schedule.md:388)。
**因此正式 docs 无需修改**；只修原建议表的范围措辞，并按既有字段缩短旧实例，不再添加防御性说明。
原建议表应把“删除 host 操作过程”限定为不丢失有设计影响的初始化、观察版本、等待范围和录制/执行区别；这些事实已有声明位置。
DOC4 [374–388](../rules/4.design.work_schedule.md:374) 的现行事件语法仍无外部命名空间；本次去重没有消除这一机械引用边界，也未证明应扩展它。

**MLA：Geo 上传是整 grid 的启动前提，准备过程不应被统一入口屏障吞掉。**

真实 [YAML 321–338](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/root.yaml:321) 混写 D2H、host 等待、排序、缓存查询、H2D、录图与可选输出。
其中第 332–333 行是：“Upload the complete Geo table ... record table_event; the main/current stream waits ... before the one main kernel.”
源码分别在 [1669–1681](../../../phase_grammar/sources/kernel/extract_mla/solution.py:1669) 拷 metadata 并等 side、[1895–1898](../../../phase_grammar/sources/kernel/extract_mla/solution.py:1895) 上传并记录事件。
[2010–2024](../../../phase_grammar/sources/kernel/extract_mla/solution.py:2010) 先给 main stream 加等待，再提交主核；这不是核内读 Geo 时才等待。
反事实：保留 Geo 的值定义却允许主核先启动、读表前再等，可能保持数值，却改变 CTA 驻留、初始化和可重叠窗口；不能判等价。
相反，只保留 “Geo ready” 不说明整表、版本和启动点，也不能排除读取未完全上传的表。
可缩短为以下**现行合法字段片段**，它只替换外部等待说明，其他调度内容保留：

```yaml
schedule:
  strategy: >-
    Geo preparation requires its metadata version to be visible to the supplier
    and may overlap unrelated caller work. The entire main grid starts only
    after the complete Geo table for this execution is visible.
```

这里的 metadata 就绪是必要调用前提，不能写成源代码已经建立的保证。
源码 [1646–1681](../../../phase_grammar/sources/kernel/extract_mla/solution.py:1646) 建 nonblocking side 后直接 D2H，没有 caller→side 输入 fork；主核等上传不能修复这个方向的缺口。
因此需保留一条来源限制，或在有外部证据时明确已就绪的调用域；不能因 DOC4 有默认 caller 边界就替外部准备补同步。
第 325–331 行的表顺序与 Scatter 条件不是可随 host 删除的过程：它们决定任务排列与局部性，仍属于 `schedule.strategy`。
源码 [1785–1819、1867–1892](../../../phase_grammar/sources/kernel/extract_mla/solution.py:1785) 证明先按 head/batch/reverse tile 构表，再稳定按负载重排；无需保留计数排序循环。
第 335–336 行的 recording/replay 版本区别也须保留一次；DOC4 [248](../rules/4.design.work_schedule.md:248) 已说明 capture 本身不执行计算。
缓存查询刷新与 table/queue 的跨次复用另交第三轮；本报告不把它们整段判删。

**Sparse：共同调用边界已覆盖 fork/join，不能靠串行化补出口。**

真实 [hidden packed YAML 299–311](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/probe_hidden_packed.yaml:299) 的内部图只有：

```yaml
flow:
  launch:
    - phase: [probe]
    - phase: [prefix]
    - phase: [remainder]
      start_after: probe.kernel_done
```

这也是现行合法的 flow 片段；旧稿其他区域使用旧动作语法，不据此声称整份 YAML 已通过当前规则。
prefix 写 `[0,K*GT)`，remainder 写 `[K*GT,T)`，两者不合并、不共享局部状态，见 [375–382、437–444](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/probe_hidden_packed.yaml:375)。
源码 [6153–6163](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_sparse/dsv4_kernel.cu:6153) 在 caller 前序后 fork，side 提交 probe，caller 提交 prefix，然后 host 才观察 probe。
[6180–6190](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_sparse/dsv4_kernel.cu:6180) 将 remainder 排在 side 的 probe 后，随后 join 回 caller。
据当前 DOC4：入口为 probe/prefix；后继使用完整 out 等两段生产区域完成且可见；不存在 prefix→remainder 的必要边。
反事实一：删掉输出 join 且不适用共同边界，caller 可在 remainder 写完前读 out；错在输出可见性。
反事实二：加 `start_after: prefix.kernel_done`，虽然输出可正确，却丢掉慢 prefix 尾部与 remainder 的重叠机会。
现行共同边界排除第一种，现有 flow 不引入第二种额外边；无需新增第四个 join phase。
可删短真实 [428–434](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/probe_hidden_packed.yaml:428) 的 record/wait API 叙述和重复完成关系。
实例只须保留有区别的事实：“No host wait is added for output completion.” 必要机制在 realization 绑定，不复写全部 event 过程。
若实施要求禁止用 host 同步牺牲并发，这句仍有意义：DOC4 的“不要求 host 阻塞”本身并不等于逐实例禁止阻塞。

**Sparse host 观察：轮询结束不能冒充 probe 完成。**

真实 [315–326](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/probe_hidden_packed.yaml:315) 明确 host 才消费 ProbeRatio；attention 不读它，没有数学 handoff。
源码 [5582–5614](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_sparse/dsv4_kernel.cu:5582) 初始化 sentinel、有界轮询，最终逐槽读取正值形成观察结果。
即使轮询提前退出，remainder 仍受 side 队列的 probe 完成约束；不可将该边替换成“host 已选好参数”。
必要保留的是观察结果版本、它不保证 GPU 已完成、prefix 先提交再观察；具体轮询计数是否控制观察结果不能在未限定成功域时直接删。
旧 probe 写入可能与后次 reset 竞争是来源限制，不能通过新字段杜撰代际保证；完整观察/跨次状态由第三轮判断。

**晚等待及初始化：已有字段够用，但“host 过程全部删除”太宽。**

GDN [root 326–338](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml:326) 已把 `prep.launch_ready`、`prep.kernel_done`、`chain.prep_ready` 分开。
真实动作 [601–609、670–675](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml:601) 在 join 前读取 initial_state，并可预取首块输入；之后才消费 prep 系数。
源码 [2178–2224](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_gdn/gdn_kernel.cu:2178) 的 `sreg_load()` 位于 `griddepcontrol.wait` 前，提供可核的晚等待反例。
删 `wait_at` 或把启动改等 `prep.kernel_done` 会提前屏障、减少重叠；这是**内部依赖已有能力**，不是必须在外部 strategy 再写一遍。
当前 DOC4 [388](../rules/4.design.work_schedule.md:388) 和 DOC5 [519–551](../rules/5.design.tile_pipeline.md:519) 已明确定义三个时点及等待动作的 done。
MoE [root 400–403](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/root.yaml:400) 的 Counts/Fill 清零仍须保留，每次执行重建；可以去掉 `cudaMemsetAsync` 名称。
源码 [9230–9234、582–600](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:9230) 是先清零、后原子累加；删除初值会把上次残留计数累入当前结果。
补充源码级反证：同一固定文件 [9164–9191](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:9164) 的 cfg2=46/47 在 side 清零 Acc32，仅于 [9329–9331](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_nvfp4_moe.cu:9329) 的 GEMM2 前 join。
这一额外分支**没有对应已创作 YAML 证据**，不算 root 缺口；只证明“外部清零完成→相关后段启动”不能提升成全调用开头等待。
现有相关 phase 的 `schedule.strategy` 可保留这个作用点；Acc32 初值仍归其合并/初始化定义，不需要一个 memset phase。

**MoE 首次建表：设备范围的 host 等待不是 `tables.kernel_done` 的同义词。**

已批准 [init_no_scan_direct 205–237](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/init_no_scan_direct.yaml:205) 同时有 `tables.kernel_done→direct` 和全设备同步文字。
源码 [ensure_tables 1005–1016](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_solution.py:1005) 提交表核后调用 `torch.cuda.synchronize(index)`，返回后才标 ready；[1166](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_moe/source_solution.py:1166) 之后继续配置计算。
反事实：另一个 stream 有长任务 U、表核先结束；仅保留 tables 完成允许 direct 与 U 重叠，源路径的全设备 host wait 则不允许。
因此在保留这一初始化路径的启动/重叠设计时，至少短写：“Before the kernel containing prelude and direct, the host completes a device-wide wait after tables.”
放在 `direct.schedule.strategy`，ready 状态发布位置另留一次；不需列完整 Python 调用顺序，也不把全设备同步提升为所有路径的要求。
若只保留纯计算依赖，可将此作为明确舍弃的包装层行为；但不能宣称它与单 kernel 完成等价而无损删去。

**删除测试与总表示成本。**

| 处理对象 | 保留的设计事实 | 删除/保留的结果 |
| --- | --- | --- |
| MLA 上传的 event 名、record/wait 调用句 | 整个 Geo 版本可见→整 grid 启动，准备可与无关 caller 工作重叠 | 删除机制叙述不改数值、任务集、必要先后或并发机会。 |
| Sparse fork/join 多处复述 | 共有入口/输出边界、内部 probe→remainder、异步输出交接 | 省掉边界重抄，不增加 prefix→remainder；机制只绑定一次。 |
| Geo 排序、MoE Counts/Fill 初值 | 任务排列/局部性、每次执行的零初态 | 不能因在 host 过程附近就删除，否则改变调度或数值。 |
| GDN 晚等待、MoE device-wide 等待 | 消费点、等待作用范围与启动窗口 | 不可压成泛泛“before compute”；必须保持区别。 |
| Sparse root graph 失败/回退过程 [557–575](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/root.yaml:557) | capture 不执行、实际只执行该 flow；若保留录图准备的 host 等待则点名一次 | [源码 6384–6398、6492–6507](../../../../schema_extract/archive/20261109-ten-operator-text-audit/work/extract_sparse/dsv4_kernel.cu:6384) 支持删掉重复机制步骤，不支持凭空增减实际提交。 |

已有共同边界即可消除 Sparse 每个实例的 caller 端点重复；Geo 只需一条实例供给关系，而外部事件表还需身份、范围、版本和连接规则。
因此现范围选择零新字段成本更低；并非把同一长文换个自由字段，也不声称 checker 能校验外部关系或得到全量节省比例。
若未来要求机器组合任意 caller/外部供给图，才有真实的外部引用扩展需求；本轮实例并未证明这种自动化是必要目标。

| 分类 | 本轮结论 |
| --- | --- |
| 当前规则已有能力 | 共同 caller 边界、Geo 前提、内部启动/就绪/消费三点、初值和原语绑定。 |
| 实例没用好 | Sparse 在 thesis/describe 重复 fork/join，MLA 把机制过程与必要调度规则混写。 |
| 仍需短定义（实例） | 用现行 strategy 点名外部条件作用点、范围和位置；默认覆盖的关系不重复，无需增补共同规则。 |
| 真实结构边界 | 无法结构化引用外部事件；没有据此证明人读契约还缺必要字段或必须新增模块。 |
| 来源缺陷 | MLA 缺 caller→side 输入顺序；Sparse 的 host 观察可能先于 probe 完成，存在后次重置竞争。 |
| 实现细节 | event 名、record/wait API 步骤、无设计影响的分配与捕获流程可删；机制所提供的保证留下。 |

实际深读限于上述 YAML 片段与绑定源码；另读取十 root 的 flow 和相关 schedule/describe 投影来找反例，没有全量审计全部节点。
主要入口 [SCHEMA_VERBOSITY 80–89、198–221](../../../../schema_extract/archive/six-operators/20261109-methodology-review/SCHEMA_VERBOSITY.md:80) 的“必要文字/既有结构/来源缺陷”分流经当前规则复核。
完整读取范围、分类、证据行号与来源状态在 [data/01_external_contract.json](../data/01_external_contract.json)；没有运行 GPU，也没有把静态阅读或旧批准当成源码正确性证明。
与组内分工的交叉只到“若设备真有独立等待，则现有 await/pipe 要保留”；不全审 member 或角色拆分。
与跨调用状态的交叉是 Geo/table/queue 复用、probe host 观察和 table-ready 发布；本报告保留必要前提与来源限制，不全审缓存容量、替换和复用协议。
