import json,re,hashlib,subprocess
from pathlib import Path
B=Path(__file__).resolve().parents[2]; W=B/'work/extract_sparse'; U=json.load(open(W/'indexed_units.json'))
F=[]; A={}
def ids(ns):return [U[n]['id'] for n in ns]
def ev(n):return list(dict.fromkeys('inputs/extract_sparse/'+o['node']+('.yaml' if o['node']!='dispatch.md' else '')+':'+str(o['line']) for o in U[n]['occurrences']))
def find(k,kind,ns,claim,core,counter,candidate,gap='none',evidence=None,confidence='high'):
 F.append(dict(id='extract_sparse-F'+k,kind=kind,unit_ids=ids(ns),claim=claim,core_fact=core,deletion_counterexample=counter,existing_structural_candidate=candidate,gap_kind=gap,evidence=evidence or ev(ns[0]),confidence=confidence))
def mark(ns,cat,core,structure,reason,fs=()):
 for n in ns:
  assert n not in A,n
  A[n]=dict(id=U[n]['id'],category=cat,core=core,structure=structure,reason=reason,evidence=ev(n),finding_ids=['extract_sparse-F'+x for x in fs],contexts_checked=True,occurrence_overrides=[])
# These are explicit human-selected groups after reading all indexed units and each effective structure.
mark([0,1,2],'MIXED','some','mixed','数学/有效域是核心；同时混有 ABI、副作用和重复的标量约束，逐句看 F01/F15，不能整段删。',['01','15'])
mark([3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,21,22,23,24,25,26],'EXPLANATION','none','already_present','输入或中间值的用途摘要；相应 compute 的定义、输入 shape/dtype/default/range 已给出约束。meaning 是必填解释，非整字段删除许可。',['02'])
mark([7,20],'ESSENTIAL_TEXT','all','open_semantics','调用返回对象与 output_buffer 的同一身份属于契约副作用，不能从等 shape/dtype 推出；当前以 meaning 正文承载。',['15'])
mark([19],'NONCORE','none','not_applicable','enable_pdl 在本固定调用者为 None，无数学或实际启动协议影响；保留接口标识即可。',['16'])
mark(range(27,35),'FORMAL','all','formula_or_predicate','输入允许值域/布局约束；英文谓词与数学不等式一样属于正式关系，不是冗余叙述。',['01'])
mark([35,36,37],'EXPLANATION','none','already_present','compute 标题导航，主体 math 承担定义；名称必填而非额外执行动作。',['02'])
mark([38,39],'SOURCE_LIMIT','some','not_applicable','保留比较器未知、来源实现偏差与树外路径范围；源码记录不能替代完整验收或自行补出保证。',['13','14'])
mark([40,41,42,43,44,45,46,47,48,49,51,52,53],'EXPLANATION','none','already_present','中文 dispatch 导航；精确选择与计算提交以本节点 YAML when/flow 为唯一来源，索引不追加约束。',['02'])
mark([50],'MIXED','some','mixed','capture 不执行和回放绑定是公共规则重复；私有 capture 前的 caller 完成边界与失败路径属有源事实。',['05','13'])
mark(range(54,62),'MIXED','some','mixed','独立充要选择谓词、实际 probe 观察和参数来源不可丢；夹带公共 capture 语义、派生说明和已在下层的提交摘要。',['01','05','06','16'])
mark(range(62,72),'MIXED','some','mixed','fast-math/融合允许范围是数值路径，unknown 及不承诺逐位等价是来源限制；不能把误差未知当允许任意误差。',['01','14'])
formal_ledger=[72,73,74,78,80,82,85,90,93,94,97,98,99,102,103,105,108,109,112,113,114,115,117,118,119,124,126,127,129,130]
mark(formal_ledger,'FORMAL','all','formula_or_predicate','定量成本差额或 ledger.by 正式步骤引用；已核对每一使用节点的 by、baseline 与局部参数，不能按英语字符视为可删。',['01'])
mark([75,77,79,83,86,88,91,92,95,96,100,104,106,107,110,111,116,120,121,122,123,125,128],'EXPLANATION','none','already_present','账本对象标签或比较基线；它解释量化公式的口径，不另定义计算/执行；必填标签保留短写。',['02'])
mark([76,81,84,87,89,101],'MIXED','some','mixed','精度改变已在 online/gather 公式和 arithmetic 明确；该账本兼有这些摘要与不可删的来源误差边界。',['01','14'])
mark([131,132,135,136,137,138,139,140,141,142,144,145,147,148,149,151,152,154,155,156,158],'FORMAL','all','formula_or_predicate','实际求值、离散选择、分组舍入及空域/填充访问的正式定义；保留公式不是自由叙述结构缺口。',['01','17'])
mark([133,153],'MIXED','some','mixed','multiplicity-preserving union、pool-major permutation、padding pool 选择是核心公式；同时说明运行时顺序与 prefix 物化作用域。',['01','07','11'])
mark([134,143,146,150],'MIXED','some','mixed','两半 QK、one-unit maximum lag、BF16/FP32 分叉是数学；尾部另含 RG 同组缩放及重复 m/l 的执行分工绑定。',['01','10','12'])
mark([157],'MIXED','some','mixed','R 的 FP64 求和/FP32 输出是公式，host reset、非共同快照和 replay 不重选是跨执行生命周期事实。',['06','17'])
mark(range(159,243),'EXPLANATION','none','already_present','本地 symbol 的短释义；已逐个核对所属 rewrite.math、shape/dtype、retention 或 binding，未发现释义独占而其定义缺失的计算事实。',['02'])
mark(range(243,259),'EXPLANATION','none','already_present','rewrite 标题，不作为动作/依赖；同一步 math 定义该决定。必填 name 可以短写，不能省空。',['02'])
mark(range(259,269),'FORMAL','all','formula_or_predicate','rewrite.step 是 ledger.by 使用的正式身份；不能当自然语言说明删去。',['01'])
mark(range(269,278),'EXPLANATION','none','already_present','任务或数学设计摘要；实际 union/recurrence/precision/cost 由 compute、rewrite、ledger 承载，摘要必填且有导航价值。',['02','04'])
mark(range(278,286),'EXPLANATION','none','already_present','pipeline 摘要中的分工、槽深、回收边界均在本节点 resource/role/pipe/waits_for/verb 中展开。',['02','04','08'])
# Every custom action is individually read; the following mechanism clusters preserve its exclusive contract.
mark([286,326],'MIXED','some','mixed','empty union 的零输出和 v/c 不产生动作已有 math+domain；不获取 TMEM/不初始化传输协议以及已完成 Q 拷贝仍是路径特有动作契约。',['08','09'])
mark([287,288,293],'ESSENTIAL_TEXT','all','open_semantics','serial role 的动作完成本身不表示 CTA 所有读者到达；retire 契约定义读者汇合后复用，不能仅凭动作名。',['09'])
mark([289,296,298,302,306,320,321,323,327,333,340,343,346],'MIXED','some','mixed','公式引用与已登记资源有重复；仍独占 warp/行/列归属、同步可见性或动作内部跨阶段推进保证，属开放动作语义。',['09','10'])
mark([290,292,294,297,309,324,325,332,341,345,348],'ESSENTIAL_TEXT','all','open_semantics','定义特定配置参数、异步 priming、Q/索引 publication 与准备完成保证；只看 serial 外壳不能还原何时 KV 可读。长文含可局部压缩片段但尚无整项纯结构等价改写。',['09','10'])
mark([291],'MIXED','some','mixed','c0 的 copy group 由 prepare 发出和 wait-all/CTA publication 是核心；previous retire 的就绪关系也由 serial 顺序保留。',['09'])
mark([295,305],'MIXED','some','mixed','final PV 前驱及 OutputStage alias 已结构化；TMEM 读取结束、实数行转换与 padded-row convergent reads 的动作保证需留。',['07','08','12'])
mark([299,312],'ESSENTIAL_TEXT','all','open_semantics','valid/Mask publication 与 Gather 数据落地分开；done 对应已发射后的成员汇合与代理可见性，不是传输 drain。',['12'])
mark([300],'MIXED','some','mixed','前一 PV 等待已有 reuse 边；c=0 不读写 Acc、全一仍保留等待、uniform 行组缩放和 TMEM 完成是动作契约。',['08','12'])
mark([301,307,331,334,338,344,347],'MIXED','some','mixed','跨 worktile 无状态、kernel_done 可见性有公共/现有结构依据；stage/copy/TMEM 退役、实际输出区域仍需动作绑定。',['07','08','09'])
mark([303,311,316],'ESSENTIAL_TEXT','all','open_semantics','原语完成组确定 QK 两个 issuer 共同 done 与 PV 独立累积完成，并规定全部矩阵读者已退役；不可从 API 名或普通 async 自动推出。',['12'])
mark([304,322,349],'MIXED','some','mixed','probe 数学贡献规则已有公式；hash 原子发布与 membership 的先后、预取窗口、本地 count 汇合和实际一次写出是核心协议。',['11','17'])
mark([308,314,319,337],'MIXED','some','mixed','TMEM 初始化/保留区域及状态广播是核心；末句披露固定源码错误地址别名，不能变成设计复用规则或无证修复。',['07','12','13'])
mark([310,336],'MIXED','some','mixed','finish 的身份是发行结束而非 drain，必须保留；真实 transfer 已由 PV/QK 链退役的解释可缩。',['08','12'])
mark([313,339],'MIXED','some','mixed','Gather 数学映射和实现 API 名可缩；TMA/mixed chunk 选择、落地 done 与发射独立和两种路径填同槽是必要绑定。',['07','12','16'])
mark([315,342],'ESSENTIAL_TEXT','all','open_semantics','enqueue done=commit 而非落地、Stages=1 no-op 仍有事件、尾部 empty commit 与距离定义没有被当前 resource/pipe 完整表达；必要开放契约。',['09'])
mark([318],'MIXED','some','mixed','最后 retire 后转 FP32 staging 和 global publication 是核心；DQ 非原子重复写无 ordering 是来源缺陷；不跨 worker 保留已由模型默认。',['07','09','13'])
mark([328],'MIXED','some','mixed','done 只发布本 warp 行并归还其 score 读取是必要契约；四 arrivals 共同授权的关系已经在显式前驱集合。',['08','12'])
mark([329],'MIXED','some','mixed','union 原子安装/位置发布/重复私有项/两端分配与读取完成定义必保留；Hash/IndexStage/ring 容量和复用已有 resource 视图。',['07','11'])
mark([330],'MIXED','some','mixed','预取次数/访问空 split 的行为已在 merge，异步寄存器 ready 和 padded head 四个 no-op 事件则需保留。',['09','17'])
mark([335],'MIXED','some','mixed','源 probe publication 只写一次且可能先于 kernel_done 被观察；probe→attention 已有 start_after，不能删掉 host 不等待事件的限制。',['06','08','17'])
mark([350,351,352,354,357,358,360,366],'MIXED','some','mixed','提供现有数值/完成/可见性契约的实现绑定需要保留；精确包装和 layout 不能仅因源码出现就称核心，按 F16 区分。',['12','16'])
mark([353,361,365],'MIXED','some','mixed','workspace/graph key 有源定义与 caller capture 前完成边界并存；capture/replay 公共语义与可探索 API 细节不是都不可压缩。',['05','13','16'])
mark([355,359],'MIXED','some','mixed','mapped gate reset/观察与 pending 写生命周期影响选择；源码缺少跨调用隔离必须保留；存储分配 API 本身非核心反例。',['06','13','17'])
mark([356,362],'MIXED','some','mixed','外部 fork/join 及 nonblocking 保持无 prefix→remainder 屏障是核心；event API 和 cached 资源名不独自证明扩语言。',['05','06','16'])
mark([363,364],'MIXED','some','mixed','初始 dual 判定、hidden 提前返回与录制后绑定是来源事实；已有 flow completion 和公共 replay 规则可缩写。',['05','16'])
mark(range(367,385),'EXPLANATION','none','already_present','本地 require 的理由指向已读上游动作/flow/choice 的完成保证或绑定；无独占新设计事实，仍是每条实现交接的必填依据。',['02','16'])
mark([385,386,388,389,392,393,394,395],'EXPLANATION','none','already_present','phase 的计算摘要；decomposition.target、数学定义及本节点 flow 已登记实际工作。describe 必填，非空短写。',['02'])
mark([387,396,397],'MIXED','some','mixed','bounded poll、mapped reset、实际观察与 gate race 属核心/来源限制；probe kernel 数量和内部 start_after 等已有结构。hidden 另有外部 fork。',['05','06','13','17'])
mark([390,391],'ESSENTIAL_TEXT','all','new_structure_candidate','remainder 完成后排 caller join，与 prefix 和外部后继的关系无法用本调用内部 start_after 指向一个已存在 consumer。',['05'])
mark([398,399,400,401,403,405,406,408,409,411,412,413,418],'MIXED','some','mixed','固定几何/参数来源及 token 范围绑定是正式核心；partition/iteration/target/default 部分重复，host gate 或空工作边界须分别保留。',['03','06','10','17'])
mark([402,404,410,414,415,416,417],'DUPLICATE','none','already_present','线性或反向 CTA 次序已由本节点 worktile_order 与 preassigned 完整定义；remainder 全局偏移还由其 decomposition.strategy 定义。无共驻保证是公共调度规则。',['03'])
mark([407],'FORMAL','all','formula_or_predicate','余段局部 worker 到全局 token 的偏移映射；partition 只有长度，保留正式映射或引用 decomposition 中同一来源。',['03'])
mark([419,423],'MIXED','some','mixed','launch 集合及 probe→remainder 已有字段；caller prior/final join 外部端点仅由正文定义，不是普通摘要。',['05'])
mark([420,421,422,424,425,426],'EXPLANATION','none','already_present','工作组织/代价摘要由本节点 partition、flow、strategy 与 ledger 展开，保留必要的短说明即可。',['02'])
mark([317],'ESSENTIAL_TEXT','all','open_semantics','定义probe元数据由GT参与者解析并CTA发布；无样本时不读metadata，资源表和inspect名称不能独自给出这些保证。',['09','17'])
assert len(A)==427, sorted(set(range(427))-A.keys())
# Findings include every observed mechanism; no new field is presented as existing syntax.
find('01','style',list(range(0,3))+list(range(27,35))+list(range(62,72))+formal_ledger+list(range(131,159))+list(range(259,269)),
'数学公式、输入谓词、成本公式与正式 step/by 身份应与纯叙述分开统计。',
'稀疏 slot 多重性、sink 只加分母、空集合、base-2/FP32/BF16 舍入、分组次序、padding 读取等决定实际计算。',
'删去 union 私有重复项可把同 query 两次相同 ID 合成一次；删去 l 使用未窄化 E 可让 BF16 Prob 决定分母；二者改变输出。',
'rewrite.math: 保留集合/递推/显式 cast；arithmetic.approx 保留 unknown；symbol.retention 保留实例范围。公式字符串本来就是当前正式数学载体。',
'open_semantics',['rules/schema_docs/3.design.math_reshape.md:52','work/extract_sparse/problem_reference.py:53','work/extract_sparse/dsv4_kernel.cu:3104','work/extract_sparse/dsv4_kernel.cu:3198'])
find('02','style',[n for n,a in A.items() if 'extract_sparse-F02' in a['finding_ids']],
'标题、meaning、thesis、中文导航和理由有阅读价值，但非独占设计定义；多数是必填短字段。',
'实际定义在本节点 compute/rewrite/strategy/verb 与结构中；不能因短字段重复就删成空字符串。','仅删解释不产生不同核心设计；若删必填字段则违反结构，故候选只采用合法短替换。',
'name: Joint softmax; meaning: Query latents; thesis 保留一两句主要选择与代价。dispatch.md 按 DOC0 保留节点导航，不复制精确充要条件。',
'none',['rules/schema_docs/0.common.md:155','rules/schema_docs/3.design.math_reshape.md:52','rules/schema_docs/6.realization.md:54'])
find('03','redundancy',[398,399,400,401,402,403,404,405,406,407,408,409,410,411,413,414,415,416,417,418],
'7 个 schedule.strategy 完全复述已有工作排列；其余分解策略混有参数及局部→全局索引，不可一并删。',
'反向遍历由 worktile_order.direction 表示；hidden remainder 偏移需要 decomposition 中 t=K*GT+u/grp=K+j。',
'删7个纯次序策略不改变排列；若把 remainder 偏移也删掉，局部 u 可被解释成前缀 token，导致重复/缺失输出。',
'worktile_order: [{by: token, direction: descending}]\nworker_assignment: {preassigned: {start: worker_id, count: 1}, refill: none, exit: assigned_done}\n# 保留原 remainder decomposition 的全局偏移；这不是新增 offset 字段。',
'none',['inputs/extract_sparse/tc_packed.yaml:219','inputs/extract_sparse/probe_hidden_packed.yaml:436','rules/schema_docs/4.design.work_schedule.md:193'])
find('04','structural_gap',[133,134,143,146,150,153,271,274,275,276,282,283,284,300,305,311,328,329,343],
'跨文件局部相同定义和四个 softmax 角色带来复制成本；本地 alias 已使用，但 from_parent 仅整区复用，不能 patch 局部 phase。',
'相同 packed 计算在 GT=2、GT=4/8、prefix 与 remainder 绑定不同选择域及工作范围。',
'共享完整串行父 work_schedule 会丢 hidden 的三提交关系；只因词句相同便复用其选择域，会错误排除 GT=2 或整个余段。',
'同文档可用 &soft_verb/*soft_verb、&packed_phase/*packed_phase；probe_hidden_packed 已这样复用 remainder。跨文档仅 design.tile_pipeline: {from_parent: direct_parent}，当前多数三区存在真实差异，不能宣称可直接替换。',
'representation_cost',['inputs/extract_sparse/probe_hidden_packed.yaml:986','inputs/extract_sparse/tc_heads.yaml:375','rules/schema_docs/0.common.md:130'])
find('05','structural_gap',[50,54,55,56,57,58,59,60,61,353,356,361,362,363,364,365,378,384,390,391,396,397,419,423],
'调用外部的 caller-prior→side fork、prefix/remainder→caller-subsequent join，以及 capture 前 host completion 边界没有现有结构端点。',
'内部图只写 probe.kernel_done→remainder；完整请求还必须保证 side 输入可见及后继 caller 看到两段 output，且不添加 prefix→remainder 等待。',
'去掉 fork/join 正文后，两个实现都满足三条 launch：A 正确等待 caller 先前写入并在返回后 caller 排序；B side 抢读旧输入或 caller 后继在 remainder 完成前消费输出。差异是可见性/依赖而非 event API 名。',
'flow.launch: [{phase: [probe]}, {phase: [prefix]}, {phase: [remainder], start_after: probe.kernel_done}]\n这是现有内部结构；start_after 只能引用已登记 phase 的 milestone/kernel_done。没有合法 external input-ready 或 invocation output-join 端点；造第四个无计算 phase、虚构 wait action/host worker 不是现有语法答案。',
'missing_field_or_event',['inputs/extract_sparse/probe_hidden_packed.yaml:305','inputs/extract_sparse/probe_hidden_packed.yaml:428','work/extract_sparse/dsv4_kernel.cu:6153','work/extract_sparse/dsv4_kernel.cu:6187','work/extract_sparse/dsv4_kernel.cu:6384','rules/schema_docs/4.design.work_schedule.md:269'])
find('06','structural_gap',[55,56,57,58,135,145,157,158,335,355,356,359,362,387,396,397,412],
'有界 host 观察、录制时一次选择、pending device writes 与 process gate 重用缺少紧凑的跨调用生命周期表示。',
'R 来自最终一次普通 FP32 scan 的正值均值，可能局部可见；poll exit 不是 probe 完成，后续 gate reset 可能与先前写竞争。',
'若将 poll exit 当作全局发布或复用完成，会错误排除部分样本接受 packed 的路径；若按 invocation 新建私有 gate，则改变原实现跨调用状态与竞争。',
'可以在 dispatch.when/rewrite.math 保留 R 的正式定义，flow 内保留 start_after: probe.kernel_done；handoff.consumer 必须是 GPU phase，tile resource 属于 worker，retention: keep_value invocation 不表示 process storage。当前没有 host-observation/poll-exit/cross-invocation storage 的独立结构；不得补虚构完成保证。',
'missing_field_or_event',['inputs/extract_sparse/probe_serial_mma.yaml:88','inputs/extract_sparse/probe_serial_mma.yaml:172','work/extract_sparse/dsv4_kernel.cu:5561','work/extract_sparse/dsv4_kernel.cu:5582','work/extract_sparse/dsv4_kernel.cu:5593','rules/schema_docs/4.design.work_schedule.md:323'])
find('07','existing_structure',[133,153,292,295,305,307,308,313,314,318,319,329,331,334,337,338,339,344,345,347],
'ring 被 hash/index、KV、output 分阶段复用，TMEM partial 双槽和 Acc 区域均已由一个 resource+view 表达；局部句子可缩。union 两段逻辑→物理映射可进一步落到现有 view.part。',
'共享存储需保留容量和读者退役；同名或 anchor 不制造存储身份。',
'把三个 ring 用途改成独立分配会提高 SMEM 需求；保留 view 但删 build.done/PV.done 到覆盖的路径会容许未读完就覆盖。',
'math: "Upos(k)=k if k<Ns, otherwise 2560-Ng+k-Ns, for 0<=k<Nu(grp)."\nresource:\n  UnionStore:\n    space: SMEM\n    member_slot_shape: [entry: 2560]\n    dtype: int32\n    slot_count: 1\n    view: {"UnionKey(k)": {part: {entry: ["Upos(k)", "Upos(k)+1"]}}}\nring 原有 HashTable/IndexStage/Gather(c)/OutputStage views 保留；此候选只结构化映射，不代替 atomic publication。',
'none',['inputs/extract_sparse/tc_packed.yaml:239','inputs/extract_sparse/tc_packed.yaml:363','rules/schema_docs/5.design.tile_pipeline.md:275','work/extract_sparse/dsv4_kernel.cu:2455'])
find('08','redundancy',[278,279,280,281,282,283,284,285,286,291,295,300,301,305,307,310,328,331,334,335,336,338,344,347],
'多读者结束、final PV 前驱、empty-union 条件动作和内部 launch 完成已经有结构；重复叙述只能局部省略，动作本身的完成契约仍保留。',
'四个 publish 必须全部满足；c=0 rescale 是存在的 no-op，empty union 的 v/c 主体才不存在；stage 不等待 loader empty-tail，Finish 才汇合。',
'把 publish 列表缩成任一一个会容许 score 提前覆盖；把 stage 也接 Loader/finish 会加强顺序；删 c0 no-op 动作会破坏动作身份。',
'domain: {v: [0, "min(1,C(grp))"], e: [0, "1-min(1,C(grp))"], c: [0, "C(grp)"]}\nreuse: {"qk:Partial(c+2,0)": ["Soft0/publish:Prob(c)", "Soft1/publish:Prob(c)", "Soft2/publish:Prob(c)", "Soft3/publish:Prob(c)"]}\ndata: {"stage:Result(v)": ["Soft0/pv:Acc(last(c))"]}\n这些都是现有实际结构，不能用未定义 done 语义替代。',
'none',['inputs/extract_sparse/tc_packed.yaml:437','inputs/extract_sparse/tc_packed.yaml:507','inputs/extract_sparse/tc_packed.yaml:600','rules/schema_docs/5.design.tile_pipeline.md:176','rules/schema_docs/5.design.tile_pipeline.md:306'])
find('09','structural_gap',[287,288,289,290,291,292,293,294,296,297,298,302,306,309,315,318,320,323,324,325,327,330,331,332,333,338,340,341,342,345,346,347,348],
'单 CTA Main 内包含多阶段 cp.async、metadata、Q 缓存、softmax 和 CTA rendezvous，现有外层 serial 不能推出动作内部协议。',
'prepare.done 不等于 KV landed；enqueue.done 只提交 copy group；retire.done 才汇合所有 PV readers；Stages=1/多槽、Early、QKS 影响发射位置。',
'相同外层 Main pipe 可把 prepare 改成等待所有 KV 或提前覆盖 Q_ring；前者损失已设计的 copy/compute overlap，后者违反存储生命周期。',
'现有 info.verb.<verb>: {execution_mode, meaning} 可合法定义这些保证；也可拆成已定义 load/wait/publish/release 及 waits_for，但必须明确每段真实事件与条件域，不能只新造动作名。本次未给整个 prepare 的短结构等价稿，故不标 STRUCTURABLE。',
'open_semantics',['inputs/extract_sparse/root.yaml:428','inputs/extract_sparse/root.yaml:444','work/extract_sparse/dsv4_kernel.cu:1125','work/extract_sparse/dsv4_kernel.cu:1201','work/extract_sparse/dsv4_kernel.cu:1424','rules/schema_docs/5.design.tile_pipeline.md:20'])
find('10','existing_structure',[134,143,146,150,294,296,298,302,306,321,325,333,341,343,346,398,399,400,401,405,406,408,409,411,418],
'参数表和 warp 数学分工目前多写成叙述；其中不依赖物理线程 layout 的固定数学参数已有 symbol.value 和公式可紧凑表达，分工契约仍需开放内容。',
'固定 MR/KC、HB/KM、PS 与 RG，以及 softmax role 对 16 行、QK 对 latent half 的归属。',
'删 binding 的唯一赋值会留下自由参数；删 role 对行的映射可产生重复/漏算。不能用同名 role 猜出物理成员坐标。',
'symbol:\n  MR: {meaning: Matrix rows including padding., value: 64}\n  KC: {meaning: Union entries per local chunk., value: 64}\n固定值采用后须移除该 symbol 原 bind 并使下游引用同一值。动作契约可短写 "Softj owns rows [j*RG,(j+1)*RG), j=0..3; RG=16."；公式位置仍是现有开放语义，未声称新增 role-mapping 字段。',
'none',['inputs/extract_sparse/tc_packed.yaml:68','inputs/extract_sparse/tc_packed.yaml:191','inputs/extract_sparse/tc_packed.yaml:454','rules/schema_docs/3.design.math_reshape.md:83','rules/schema_docs/5.design.tile_pipeline.md:98'])
find('11','structural_gap',[133,153,304,322,329,349,351,358,371,377],
'probe hash 与 attention union 的原子操作序列承担必要并发语义，不可用 shared key/membership 资源表或通用 produce 名代替。',
'attention 先发布 canonical position 再允许相同 key 加入；probe entry creation 与 membership 首次置位可交错，计数未必等于最终 attention union 数。',
'把 probe 简写成 distinct union size 会改 R；把 attention position publication 等待去掉，会读取尚未发布位置并错误共享重复项。',
'当前字段只有资源、动作及依赖；同一 CTA 内 hash 重试/atomic 返回值契约适合 info.verb.build/produce 与 math 中 Creator/First。按每次哈希冲突拆事件既缺运行时展开域又成本高；本审计不要求新增 atomic 字段。',
'open_semantics',['work/extract_sparse/dsv4_kernel.cu:2346','work/extract_sparse/dsv4_kernel.cu:2360','work/extract_sparse/dsv4_kernel.cu:3972','inputs/extract_sparse/probe_hidden_mma.yaml:97','inputs/extract_sparse/tc_packed.yaml:363'])
find('12','structural_gap',[295,299,300,303,305,308,311,312,313,314,316,319,321,328,337,339,343,350,351,352,354,357,358,360,366,379],
'异步完成/CTA publication/row ownership 等原语保证需契约说明；现有事件结构能接这些保证，但不能自行定义它们。',
'QK 两 issuer 共同 notification，PV 自己累积完成；valid/Mask 先 publication，数据后 landing；PV done 包含所有 operand reads，convergent TMEM access 即使无缩放也需满足。',
'将 load.done 解释成发行完成会使 QK 读取未到数据；将 publish.done 泛化成全体四 warp 完成会遗漏前驱。单纯更换 barrier API 而保持这些保证，不构成反例。',
'info.verb.load: {execution_mode: async, meaning: "done guarantees the complete Gather tile has landed."}\ninfo.verb.pv: {execution_mode: async_ordered_completion, meaning: "done covers prior PVs and ends their operand reads."}\n剩余保证在现有 meaning/realization 中绑定，属于有意开放原语语义。',
'open_semantics',['inputs/extract_sparse/tc_heads.yaml:269','inputs/extract_sparse/tc_heads.yaml:324','work/extract_sparse/dsv4_kernel.cu:2493','work/extract_sparse/dsv4_kernel.cu:3063','work/extract_sparse/dsv4_kernel.cu:3465','rules/schema_docs/5.design.tile_pipeline.md:115'])
find('13','source_limitation',[38,39,50,308,314,318,319,337,353,355,359,361,365,387,396,397,412],
'必须保留三种源偏差：TMEM allocation word/union keys 与活跃 barrier 地址重叠、DQ 非原子 m/l 重写、gate 超时跨调用竞争。不能把 approved 当执行正确保证。',
'当前 Design 按独立预算对象与依赖描述，不依赖错误 metadata alias；这属于规范允许的边界外实现偏差记录，不是证明修复后的执行已通过。',
'删除限制会让读者把 source 原字节执行等同无冲突抽象设计，或者无据添加跨 CTA/call 序列化。',
'保持 resource/view 与真实完成链；YAML 可用一句源限制。错误地址细节留审计证据；不增加 forbid 坏地址，也不增加虚构同步。',
'none',['work/extract_sparse/dsv4_kernel.cu:2104','work/extract_sparse/dsv4_kernel.cu:2497','work/extract_sparse/dsv4_kernel.cu:2510','work/extract_sparse/dsv4_kernel.cu:1460','work/extract_sparse/dsv4_kernel.cu:5599','rules/schema_extract/REVIEW.md:34'])
find('14','source_limitation',[38,39,62,63,64,65,66,67,68,69,70,71,76,81,84,87,89,101],
'atol/rtol 数值来自 metadata；comparator 公式、精度、NaN/Inf、relative basis/routing 未给出，fast-math 也无全域界。',
'保留必要未知，不把 raw tolerance 数字擅自转成默认绝对/相对公式。',
'删 unknown 可被误读成可证明已满足整个 Contract；以默认 atol+rtol 解释将添加源材料未承诺的判据。',
'保留 tolerance.out 来源记录与 arithmetic.approx 的 unknown；未提供 comparator 时没有合法可等价迁移的阈值结构。',
'none',['inputs/extract_sparse/root.yaml:143','work/extract_sparse/problem_metadata.json:38','work/extract_sparse/solution.py:77','rules/schema_docs/1.contract.algorithm_requirement.md:215'])
find('15','structural_gap',[0,1,2,7,13,20,38],
'ABI 的同一返回对象、允许写集和 scratch 无最终数值属于核心外部语义，shape/dtype/default 无法独自表达。',
'out 与 output_buffer 同一对象，其他 input storage 保持；空 J 直接写零、pool alias 不去重。',
'新建同值 BF16 tensor 返回而不覆盖 supplied output，满足单纯 out 公式却违反调用者观察；把相同池指针当同一个集合去重也改变 softmax multiplicity。',
'现有 contract.input/output.meaning 与 compute.math 合法承载副作用/对象身份；当前没有正式 alias/mutates 字段。本例可留一句关系，不由此证明需要扩展语言。',
'open_semantics',['inputs/extract_sparse/root.yaml:42','inputs/extract_sparse/root.yaml:60','work/extract_sparse/problem_reference.py:81','work/extract_sparse/problem_metadata.json:17','rules/schema_docs/1.contract.algorithm_requirement.md:31'])
find('16','noncore',[19,54,55,56,57,58,59,60,61,313,339,350,351,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366]+list(range(367,385)),
'实现绑定中存在普通 API 包装、allocator 与 layout 拼写；保留提供必要保证的部分，不用换 API 就叫核心不同。',
'性能反例只采用有因果依据的重叠/等待/访存：nonblocking 避免额外 prefix 屏障，mapped poll 不等 kernel completion；精确 API 字符串的必要性未逐项证明。',
'以语义相同 wrapper 替代 cudaEventRecord、不同 barrier 编号或等价 tensor-map 构造，不自动改变核心。若改为 host synchronize/增 prefix→remainder 屏障才改变已给依赖。',
'realization.require.choice 仅短写提供原语契约的实现兼容条件；纯布局/包装细节默认留 Coder。本审计未将全部 require 标为可删。',
'none',['rules/schema_docs/6.realization.md:15','rules/schema_docs/6.realization.md:36','work/extract_sparse/dsv4_kernel.cu:6178','work/extract_sparse/dsv4_kernel.cu:5555'])
find('17','correctness',[131,132,135,136,137,138,141,142,145,147,148,149,151,152,154,155,156,157,158,304,318,330,335,341,349,387,396,397,399,400,401,403,405,406,408,409,411,412,413,418],
'边界分支须区分无工作、no-op 事件、无有效数据、已提交但未观察全量；当前文本中的这些区别不可按长度截去。',
'空 split 留 PartO stale；combine 前四向量可读取但 predication 丢弃，后段换读 split0仍计算零乘；all-invalid MMA 仍算 recurrence，空 packed union 才直接零；partial-tail/cross-request probe 写-1，attention 仍处理这些 token。',
'将空 split 的 PartO 清零增加未声明初始化；将 all-invalid MMA 当 empty union 删除循环改变负无穷/sentinel/NaN 路径；把 host timeout 当未提交 probe 将漏掉实际工作。',
'已有 domain v/e/c 与 root c0/c1、math 分情形、flow/start_after 和 custom no-op 契约合起来足够描述当前边界；无需新增 empty flag 或自动删动作规则。',
'none',['inputs/extract_sparse/root.yaml:238','inputs/extract_sparse/root.yaml:250','inputs/extract_sparse/tc_packed.yaml:133','inputs/extract_sparse/probe_serial_mma.yaml:61','work/extract_sparse/dsv4_kernel.cu:491','work/extract_sparse/dsv4_kernel.cu:576','work/extract_sparse/dsv4_kernel.cu:598','work/extract_sparse/dsv4_kernel.cu:2466'])
find('18','source_limitation',[39,277],
'root 硬件区仍为 null 骨架，execution.Unit 未定义 CTA/warp；机器状态8个 approved不能替代当前硬件绑定完整性。',
'本次文本目录没有遗漏硬件正文：根本没有正文；资源容量/执行成员的硬件可实现性缺显式 Contract 依据。',
'若将 null 视为已验证设备/栈，会把 source 的 SM103a 与 fallback 构建行为错误当 Contract 固定保证。',
'现有 hardware_specification.device/arch/execution/feature/stack/wiki 字段足够填写；本审计只报告缺项，不补写规范或输入。',
'none',['inputs/extract_sparse/root.yaml:153','work/extract_sparse/solution.py:34','data/extract_sparse/summary.json:1'])
# These long contracts combine exclusive guarantees with repeated geometry/math or pipe order.
for n in [290,292,294,303,309,311,316,324,325,341,345,299,312,315,342]:
 A[n]['category']='MIXED'; A[n]['core']='some'; A[n]['structure']='mixed'
 A[n]['reason'] += ' 同段还重述已有数学/几何或pipe顺序，故整段按MIXED，核心局部不可删。'
A[7].update(category='DUPLICATE', core='none', structure='already_present', reason='input.output_buffer 的必需性由无 default 表示；旧值忽略和返回同一对象由 output.out.meaning 与 compute 明确，属于契约内重复，不可把必填 meaning 留空。')
# Narrow findings to the units actually discussing host observation/external boundaries.
relevant05={50,55,56,353,356,361,362,363,364,365,378,384,390,391,396,397,419,423}
relevant06={55,56,57,58,135,145,157,158,335,355,356,359,362,387,396,397,412}
for key,ns in [('05',relevant05),('06',relevant06)]:
 fid='extract_sparse-F'+key
 for n,ann in A.items():
  if n not in ns and fid in ann['finding_ids']:ann['finding_ids'].remove(fid)
 next(f for f in F if f['id']==fid)['unit_ids']=ids(sorted(ns))
# Link every substantive annotation to its findings; evidence includes its exact declaration(s).
for n,a in A.items():
 for fid in a['finding_ids']:
  f=next(x for x in F if x['id']==fid)
  if U[n]['id'] not in f['unit_ids']:f['unit_ids'].append(U[n]['id'])
# Local candidates selected after deletion tests. No global default shortening.
C=[]
def cut(n,quote,replacement='',reason='',kind='redundancy',extra=()):
 assert U[n]['text'].count(quote)==1,(n,quote)
 C.append(dict(unit_id=U[n]['id'],quote=quote,replacement=replacement,kind=kind,reason=reason,evidence=ev(n)+list(extra),applies_to_occurrences='all'))
for n in [402,404,410,414,415,416,417]:cut(n,U[n]['text'],'','整个可选 schedule.strategy 省略；本节点 worktile_order/preassigned 及余段 decomposition 已唯一给出相同次序。DOC4 允许省略已有结构足够的 strategy。',extra=['rules/schema_docs/4.design.work_schedule.md:193'])
cut(328,'Four such arrivals jointly authorize PV and score-slot reuse.','','全部4个 publish 前驱已由 PV data 与 QK c+2 reuse 列表给出；保留本warp发布/score读取结束的前句。',extra=['inputs/extract_sparse/tc_packed.yaml:536'])
cut(300,'For c>0 wait for the preceding PV even if every scale is one; then conditionally rescale', 'For c>0 conditionally rescale','c+1 rescale 的 reuse 前驱无条件指向上一PV，保留c0 no-op与完整TMEM完成契约。',extra=['inputs/extract_sparse/tc_packed.yaml:491'])
cut(329,'The 65536-byte table plus 4*GT*IndexWidth index bytes fits the 131072-byte ring by the host\'s GT*IndexWidth<=2560 gate.','','容量由 ring.member_slot_shape、HashTable/IndexStage.part 与 dispatch GT*IndexWidth<=2560 直接计算；保留复用完成句。',extra=['inputs/extract_sparse/tc_packed.yaml:243'])
for n in [310,336]:cut(n,'; real transfers are retired by the QK and PV completion chain.','.', '保留 finish 不是 drain 的身份；实际读取结束由 PV done 与 reuse 及 final stage 前驱保留。')
cut(301,'No scratch is reset or retained as numerical state for a later invocation. This CTA has no next worktile; final visibility is kernel_done.','','store 输出范围保留；scratch 非输出、keep_value(t)、preassigned count1及公共kernel_done语义已有唯一来源。',extra=['rules/schema_docs/4.design.work_schedule.md:307'])
for n in [307,334,344]:cut(n,'No worker takes another worktile.','','各phase preassigned.count=1 / refill:none / assigned_done 唯一确定。')
cut(338,'This worker takes no second worktile and retains no numerical state for another call.','','preassigned.count=1和carry_forward实例范围已限定，非新增省略语义。')
cut(347,'There are no scratch statistics, and the CTA takes no second worktile or numerical state into another invocation.','','没有split物化/target，单worktile结构及retention已经完整约束。')
cut(318,'This worker takes no second worktile and retains no Q, recurrence state or ring contents for another CTA or call.','','不同worker/kernel资源身份与单worktile分配已写清；保留源DQ重复写限制和发布边界。')
cut(54,'Pure capture only records work; replay uses the recorded host choice, scalars and bindings, with tensor contents read at execution.','','DOC0已定义计划执行实例及capture/replay绑定；前句保留此节点具体适用关系。',extra=['rules/schema_docs/0.common.md:38'])
cut(61,'For replay, use the recorded plan and bindings, with tensor contents read at execution.','','DOC0共同语义已给定；保留前句指出direct及recorded路径共用选择。',extra=['rules/schema_docs/0.common.md:38'])
cut(39,'- root 的 tolerance 保留外部验收器公式、相对误差基准、精度及检查路由未提供的来源记录；这些缺口不被当作数值保证，也不阻止当前设计提取。','- 验收来源缺口见 root.tolerance。','正文仍有root的必要来源限制；中文索引只导航，不去掉限制本身。',extra=['inputs/extract_sparse/root.yaml:143'])
cut(269,U[269]['text'],'Stream compacted slots through base-2 online softmax, sharing KV across heads and QK/PV. BF16 probabilities enable PV; KV splits add FP32 partials and merge, while DQ repeats QK.','必填thesis短替换；保留复用、精度、split代价、DQ代价，细节由rewrite及ledger保留。')
cut(270,U[270]['text'],'Use one base-2 online recurrence, sharing KV across heads and QK/PV. BF16 probabilities enable PV; normalize directly to BF16 output, with DQ repeating QK.','必填thesis短替换，保留全部主选择与代价。')
cut(280,U[280]['text'],'Eight warps gather and compute with the configured KV ring; multi-stage copies can overlap computation. Combine overlaps vector prefetch with statistics reduction.','保留thesis主要重叠；single-stage/CTA publication/count1在verb、pipe、schedule中完整保留。')
cut(285,U[285]['text'],'Eight warps gather and compute with the configured KV ring; multi-stage copies can overlap computation. Retired KV storage stages the BF16 output.','主要局部安排与复用保留，详细publication边界由verb保留。')
# Shorten the source-alias limitation without losing it.
for n in [308,314,319]:cut(n,'The source aliases allocation metadata and union keys with live transfer barriers; this design retains the separately budgeted objects and their lifetimes without depending on those erroneous implementation aliases.','Source limitation: allocation metadata and union keys overlap live barriers; the design retains the separately budgeted objects and lifetimes.','精确地址缺陷详证留F13；替换仍明确当前设计未依赖错误alias。',kind='style')
cut(337,"The source's allocation-return word overlaps a live transfer barrier; the resource and dependency design here does not rely on that erroneous implementation address alias.",'Source limitation: the allocation-return word overlaps a live transfer barrier; this design does not depend on that alias.','来源限制保留，不混入新布局要求。',kind='style')
notes={
'root':'v13；四个数学rewrite贯通Contract→split partials→combine；S/DQ/Stages/QKS/Early全域、空split与padded combine no-op、所有PV读者退役、临时ring及原m/l重写限制均核对。',
'mma_direct':'v3；直接unsplit与gate不可用未提交probe两种入口，P_run不同绑定；DQ/Stages/QKS/Early、Q-ring/output alias、单launch均核对。',
'probe_serial_mma':'v2；8 probe CTA和有界host观察后完整batch cfg8；两个launch串行，probe_count/Creator/First、Gate replay与单槽attention路径逐条核对。',
'probe_serial_packed':'v2；独立probe后GT4/8 packed；attention pipeline与tc_packed文本相同但参数和R接受域不同，Usable/empty union/跨request和tail均核对。',
'probe_hidden_packed':'v2；probe/prefix/remainder三个launch，余段仅等probe；prefix/remainder alias绑定独立worker资源，GT/全局group偏移/空union和external join均核对。',
'probe_hidden_mma':'v3；固定packed prefix保留，R拒绝只切余段为完整T预先规划的cfg8；packed与mma命名/资源/数值路径、nonempty remainder和fork/join均核对。',
'tc_heads':'v2；GT1/HB64全为真实head，QIS2双issuer/单PV、两个score轮转槽、全部4读者和loader tail汇合；局部alias展开核对。',
'tc_packed':'v1；GT2直接packed、probe免除；每query重复保持、pool-major运行时排列、hash/index/KV/output共ring及v/e条件执行核对。'}
limits=[
'只审冻结current-plan 8节点和dispatch；8个approved是库存状态，不代表GPU或完整当前规范验证通过；root.hardware_specification为null骨架。',
'427组全部人工语义分类；257种范围外字符串逐项随结构视图检查，只有标识、引用、retention、成员和数学大小表达式，没有发现应补录的正文。alias声明与展开实例分别计算。',
'固定source@6e4ca91bd91d998cac99e8dadd83838e153c679d的solution.py及dsv4_kernel.cu与当前文件相同；已读相关计算、dispatch、probe、图缓存、原子与完成路径。未完整阅读7053行中的强制/诊断/实验分支，不据其作全域证据。',
'参考reference.py全文及input.py相关构造、metadata/shape范围已取得固定版本；比较器实现、实测原始GPU实验未取得/未重新验证，性能只按结构与源码因果分析。',
'外部fork/join与process gate是字段/事件候选，不是立即扩语言的裁决；现有meaning可以合法承载但没有独立结构端点。API包装和纯hint差异不计已证核心反例。',
'短写候选仅局部审阅，未生成独立短稿并由第二审阅者等价验证；字符面积不是可删除字数，唯一可缩量来自非重叠quote→replacement。',
'未启动Runner/GPU/模型工作流，未修改输入、原schema、规范或运行状态。']
out=dict(operator='extract_sparse',prompt_version='v1.1',unit_annotations=[A[i] for i in range(427)],findings=F,removal_candidates=C,node_review=[dict(node=n,reviewed=True,effective_binding_checked=True,notes=s) for n,s in notes.items()],limits=limits)
(B/'annotations/extract_sparse.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print('annotations',len(A),'findings',len(F),'candidates',len(C))
