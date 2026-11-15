# M06 实际卡点

本轮从 AGENTS、EXTRACT_SCHEMA、P1/P2 阶段页起步，已知 EXECUTOR_ALIGNMENT 所列工具未对齐；没有运行任何 Python/checker/构建/GPU，没有改正式产物。只读源范围和精确位置见同目录 DRAFT.md；本页不代替独立可读草稿，也未读取其他人的报告。下面按实际经历记录，既有未决，也有回读后解决的困惑。

## M06-01：ZERO2 的host条件提交缺少已找到的表达方式

分类：文字/字段适配，可能需要字段变化；也可能是我没有找到合规的既有用法，尚不建议直接新增字段。

已读：common:108–124、172–182，DOC4:232–259、369–395，DOC5的if/事件/跨step规则，P2第3项；C:1707–1835、9450–9618，S:1454–1555。

走到：能区分一次实际clean replay、一次条件full replay、full里的第二次verify，以及outer capture只记录、未来才执行。capture不算执行这个原始疑惑已经由DOC4解决。

剩余反例：`graph_replay`实际先提交clean，host读取pinned verdict；若verdict=1则返回，dirty/timeout才提交full。固定 `flow.launch` 列两个verify就多报了clean路径的实际提交；只列一个又漏dirty/timeout。现行DOC4：“每个 step ID 在 flow.launch[].step 中恰好出现一次”，“说明字符串也不提供条件删边或替换生产者的语义”。common却明确举例允许“根据本次产生的Clean决定必要动作”。我理解后者可落worker内if，尚不知host提交如何落地。

试过但未采用：固定列表+describe写条件；把后续kernel空工作视为未提交；把host当CUDA role；新增launch条件字段；拆成clean/dirty/timeout多个完整设计。前四种违背已读文字，最后一种尚不能解释同一host机制的timeout选择，也可能人为增加完整身份。

影响：ZERO2可作为独立机制候选，但目前无法写出忠实的完整schedule。普通root不受此阻碍；不能把ZERO2降为参数或称P2全范围已通过。需要规范方说明怎样表达，或指出现有合法模式。最小改动暂不能给定，先保留这个具体反例。

## M06-02：复杂入口的P2止点与局部/整体边界仍有实际未决

分类：自身不明/文字应用，需要补源码；不是已确认规范缺陷。

初读：S很长，cfg编号达百项，我一度担心需要枚举每个cfg和pair才能定集合。P2“主要机制”“不先展开所有selector”“不能因未读代码推定另一套设计”让我转向实际默认入口及输出责任。按该方法找到了97/81、99/44、MEGA92和ZERO2，避免以kernel或cfg数作为设计数。

但实际遇到RED F：C:9163起side-stream初始化acc32，G2把route归并移到原子输出，k_convert取代finalizer。common的“输出形成/工作归属需重新贯通”为保留独立候选提供理由；另一方面可把差额写成“root.G2 epilogue + acc32初始化 + finalize替换”，看起来又能用局部对象描述。仅“能列出改哪些对象”显然不够判断局部闭合，我目前也还没读到足以证明全部数值和完成关系的epilogue。

走到：保留RED F为待判断，倾向独立但不作批准；为下一步点名读该epilogue的route BF16边界和完成关系。单体FP4/BF16族也有实际入口但未完整研究，没有直接降级。暂定3个具体代表+受阻ZERO2，未凑数。

影响：本次试走没有找到能声称整个S范围P2完成的止点；仍可推进root。这里部分原因是源码规模和我理解未完成，不能把全部归咎于规范。若继续，先补具体疑点，不跑全cfg笛卡尔积。

## M06-03：验收声明能保留，精确比较路由仍不足

分类：输入不足；按现行规则不阻碍当前设计调查。

已读：problem/reference.py全文、metadata benchmark_contract和description、input.py的writable_views/observable_inputs/replay_tensors，DOC1:237–261和P1逐输出要求。

事实：metadata写allclose(reference_float,candidate_float,.01,.05)；default有max_mismatch_fraction=0，mutated_inputs.output却有.001并有max_rel_l2=.01。同一output既被原位修改，又以列表返回。未提供完整runner检查路由/rel_l2定义。这不是只有“未声明tolerance”，不能用默认简写把所有已知声明压成一个常见公式。

初读风险：容易把reference自然当rtol基准，或只给alias output登记一个判据。回读发现allclose实参顺序反而提示其第二实参candidate作为相对基准，仍须保留确切来源，不能把这一条当成其他override比较器的定义。

处理：DRAFT逐层列明阈值和缺口，未来在相应tolerance作来源记录，说明未采用schema默认解释；不写入golden_reference.gap，不追索未提供外部runner为本次硬前置。此问题不阻塞P2机制，但P1正式字段稿还没完成。

## M06-04：源码历史注释与实际实例不一致，已局部解决

分类：来源阅读陷阱/自身初读误判，不是框架字段问题。

已读：S的_pick_cfg与cfg2实际分支，C:7730–7820、8330–8359、1899–1990及2160–2268。

遇到两类具体偏差：C仍称99或81为“override-only”，而S已经默认return99/选择81；MEGA头注释称unit按I/64，实际body `const int S=p.I>>5` 是I/32，s_ug处理up/gate共64值。直接以长段英文设计注释分类，会取到错误默认范围或工作粒度。

处理：以实际调用/template实参/body算式固定实例；DRAFT已说明99/81的可达性和MEGA32-wide。历史性能数字只作线索，没有据它宣称实测、正确或bit-exact。

影响：需要多读实际绑定，不能仅靠注释降低阅读量。本次已解决上述两点；未宣称其余长注释均已审完。

## M06-05：运行时条件不全属于当前机制，回读后已修正

分类：文字应用/初读自身不明，当前有可用解释。

已读：C:4846–4920的CPHOIST、common:108–124；DOC5跨工作flatten/释放依赖。

初始想法：CPHOIST在kernel内用 `Rmax>=16E` 判断，似乎应像Clean一样留if。回读后理解common按“决定对象”区分：这是搬运相对accumulator退役的策略选择，可以root固定hoist-on子域、另记hoist-off局部变体；Clean是该验证机制本次产生的状态。没有把这个条件硬塞回单份root。

另一个已解决尝试：DEEP的等待最初想用每item局部i±3表达；源码g跨item不重置，DOC5 flatten可保留环接续。DRAFT的两个released边采用g+3/g+2，未新增事件种类。

影响：普通WS依赖有现成表达。CPHOIST的语义判断尚可供独立reviewer反驳，但不是本次硬阻塞；不能用这次解决去掩盖M06-01仍未解决的host提交问题。
