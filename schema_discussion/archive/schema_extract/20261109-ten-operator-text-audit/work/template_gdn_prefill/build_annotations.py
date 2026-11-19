import json
from pathlib import Path

B = Path(__file__).resolve().parents[2]
OP = 'template_gdn_prefill'
units = [json.loads(x) for x in (B / 'data' / OP / 'units.jsonl').read_text().splitlines()]
# Ordinals below were assigned by reading the entire v1 inventory, both complete
# local Designs and shared Contract, and each distinct occurrence context.
# This is an explicit human classification table, not a keyword/default classifier.
BASE_IDS = '''09eb3c457258c03d 328abd3390325a8a 02228a924ff40f4e 238bbb50b1f21d01 33bf0b815560f8e4 448cee0c44d458bc 48e86097184ce227 4c9d80558e36cf51 56a2488c70b8ff61 69db0dca0637b672 6ae617e7af900e94 7897b9c05e6a6412 7f64c16fca084b63 9106e71a77a1e6f2 b5c49c3a57a8ea60 c9c4342f956bf0f5 dfe8ee431d8b0f73 e49b2328ca2a64bc ec06ba8ce47a66c9 fecf230a6f4d2abd 044937ae45a516d3 0588c07a723593e9 54366254e437de83 5485d7ce1226bf07 b63ee7c06b98f0ed b958fd734bd04fee c32d83714413cf4b 5c985cbceafd5785 6164115ce2fe99a1 0f7376c7514cf6c5 3f7cb032ce65ad8b 73df9924b3c14109 779345bc7e6772ef 848563f239f8b32a 886460849560fe48 8c547aad00c18067 9bf940e6b6fec2d5 89bb4b6e5937114b 973d554715a852a3 263d5b133f5e61d6 6c67314fe3205270 bf65cfd621166b87 fbe55b4f6cea5249 4f201fb09921413f 7e4016ec26766b60 18bdcc1dac780fe8 51298e81989315e3 e4c62488f606729c 0298fa2c8bae14f7 05c7b177f0175ad8 157dab9ce93a9558 19c80c00542ebb48 1e1d496686c459aa 25be23c7ed3f3c89 2bd79a3681ec0911 3d4bb15128e038dd 4645f39032fa9869 50f410388ae5159e 8280173fa2d36e48 942574c9c058fc3c c9cc86f85a733d23 d47420ff649e9ff6 03e7f003d1a1e456 268eeef78f301be6 40df5dc7792535c7 83944359c0217afb be86032434361a08 06b74a4b00421eb0 0aaef773eb6fa619 0f986bb23bb1428a 12cc71bc981ebf96 13953c5deb7bde1c 3964705ab301e71b 4404ed6cfde10939 490580c9f75b1c57 5777bfa1fc55f7b5 57cacaaa34edef2b 5cf4671fbaf99a2f 649331bc4009f6a8 6db7812837549460 6e0582597a0fa3dc 7cf3756998bf8ddb 7df165f0a63b19f7 883ca107f3905f83 8a1eba8e4d9d0f23 8afdbb706b6006ef 8b92db6188a380a0 91c93ad2e2a72ccf 97709533776d8965 9b8a0c46171a2d65 ab649368db08e1c1 c04aea241f61cff9 d0491b410260f8d3 d44624e5ca4f3ac3 fa4216613f3e4884 fcf3e639e4c70fc0 77748160509f3f78 6f05304359c0b22c 9ad391a797c48070 d870b1a3c36b1cb6 0f07081619eca457 106cd544b6cbc270 1760e93dd2d618d9 18aca5f842ce6acc 1c4ecd3132b3b929 2d944261e9874f56 304cb8b6f5a7d0fd 373e3471edbe11bf 3f802932ce3ebc5c 43bffef6cb85ae17 47ae33007ef4ae5b 4908bdeb7a81806c 513dd7d932964669 64d091036abe724f 7a8b041439bedb57 83652c75fb992cb9 883da689cc69177a 94171abdbbd93873 983106a340af33ce b030da1c7751ec8f c145f25b39585b2a c47de18ba3ae684c cb27bc7b5078acf8 cec860337c98f00f d61b4c2a7ef13a4f dd4a2966a273836b dd6e50e80e7b8620 fd3ce3eba9be55e9 77f0d01467df493b 7dfe0e091a8ad1f2 b1a00a9ae185d899 c8e7b4b4b6ebbe7e fdd0d988c0872744 6390584ff8db3df0 731675cc45401616 768b71c4eecdfa3b 9457c507c163ecd7 fafd588dd07dab16 1a5ed071695eca1f 71e6999a09164fa8 b5977bf32348dc3a 02b7c3cf9d41a509 2a0f6a7113cc1467 3b33887778ec93dc 56041319279124d3 5b334a0189d7d8e5 cfdbf59620953e19 e65d1c79fa3e54f1'''.split()
assert len(BASE_IDS) == 148
byid = {u['id']: u for u in units}
U = {i: byid['t_' + x] for i, x in enumerate(BASE_IDS, 1)}
R = {}
def add(i, cat, core, structure, reason, *findings):
    assert i not in R
    u = U[i]
    R[i] = dict(id=u['id'], category=cat, core=core, structure=structure,
                reason=reason, evidence=[f"inputs/{OP}/{o['node'] + '.yaml' if o['node'] != 'dispatch.md' else 'dispatch.md'}:{o['line']}" for o in u['occurrences']],
                finding_ids=[f'{OP}-F{x:02}' for x in findings], contexts_checked=True, occurrence_overrides=[])

# D/E/F/M/T/S encode the manually chosen primary judgement, not text heuristics.
def D(i, reason, *fs): add(i,'DUPLICATE','none','already_present',reason,*(fs or (1,)))
def E(i, reason, *fs): add(i,'EXPLANATION','none','already_present',reason,*(fs or (1,)))
def F(i, reason, core='all', fs=(2,)): add(i,'FORMAL',core,'formula_or_predicate',reason,*fs)
def M(i, reason, *fs, core='some'): add(i,'MIXED',core,'mixed',reason,*fs)
def T(i, reason, *fs, structure='open_semantics'): add(i,'ESSENTIAL_TEXT','all',structure,reason,*fs)

add(1,'SOURCE_LIMIT','some','not_applicable','定义实际oracle求值路径与实数递推的区别；固定参考实现确为FP32、block_size=32。',5)
M(2,'递推是正式核心定义；连续CUDA布局、生成分布、仅两输出可变和计时域混在同一字段，不能整段删。',2,3,4)
E(3,'DV=128的语义标签；V/O/state的dv轴与公式已确定其用途。')
D(4,'mandatory由无default决定，Python由scale.range给出，乘数用途由compute给出。')
E(5,'State(t)更新后的时点由compute递推和inclusive readout确定。')
E(6,'cu_seqlens的边界含义由range和L_b/x公式确定。')
E(7,'v的简短对象标签；delta公式定义数值用途。')
E(8,'初值关系State(-1)=initial_state与dv/dk轴已明确。')
E(9,'DK标签；k/q的dk轴与求和轴已经明确。')
E(10,'HV标签；v/state的head轴已用HV登记。')
E(11,'output已是调用者必须提供的输入对象，返回别名由O.meaning唯一约定。',1,3)
T(12,'返回顺序及O与输入output的对象别名没有结构字段；shape相同和数学值相同不足以保证写回同一对象。',3,structure='new_structure_candidate')
D(13,'归一化域已由k.range精确定义。')
E(14,'output_state已登记为必传输入，返回别名由S_final.meaning定义。',1,3)
T(15,'第二返回值与output_state的对象别名没有结构字段；仅公式S_final不能表达此副作用。',3,structure='new_structure_candidate')
E(16,'beta在delta公式中的乘数用途已有唯一来源。')
E(17,'HQ标签；q/k的head轴及hq映射已有定义。')
E(18,'alpha的线性衰减用途由D公式确定。')
E(19,'q的生成及直接参与readout由compute定义；标签无需承担第二份输入域。')
E(20,'epsilon的加性稳定用途在D和chunk公式中明确。')
F(21,'beta的半开有效域是正式谓词，不能改为闭区间[0,1]。')
F(22,'zero/独立32-token参考前缀是初态的正式有效域；不是缺省初态或None分支。')
F(23,'alpha的半开有效域是正式谓词。')
F(24,'输入域由BF16生成后再normalize的操作关系确定；精度顺序必须保留。')
F(25,'边界严格增长、B/T上限与首尾值构成正式域，排除空序列。')
M(26,'允许NaN是非默认输入域；previous contents ignored被完整覆盖及compute不读取两个buffer所覆盖。',3,4)
F(27,'精确Python binary64标量值及其算法定义，不是可自由舍入的文案。')
F(28,'atol/rtol与返回及mutated_inputs的比较路由构成正式验收谓词。',fs=(2,3))
F(29,'state使用另一组阈值和相同双路比较要求，不能按O阈值合并。',fs=(2,3))
D(30,'共享Contract、parent、when和版本的权威位置由DOC0规定，索引此句仅重复导航。')
E(31,'表头用于导航，无设计定义。')
E(32,'single的固定界与四chunk标量前瞻由其Load/ScalarOutput及pipe确定。')
E(33,'M64由partition.dv.tile=64确定；对齐与初态均未改变when，只是索引提示。')
E(34,'派生树重述single.parent=root；保留其导航价值。')
E(35,'节点差异与共享Contract的入口摘要，无独占定义。')
E(36,'索引标题，无执行语义。')
E(37,'root的可变界和共用warp在两个对应role中明示。')
F(38,'B>1是root选择谓词，与B=1互斥并覆盖1<=B<=16。')
F(39,'B=1为single的充要条件；不是在父B>1上叠加条件。')
T(40,'硬件层没有register所有权的结构字段或隐式默认；线程私有性由此定义。',16)
T(41,'TMEM显式分配、与SMEM独立是硬件机制定义；仅space名称不提供这些语义。',16)
E(42,'DOC5已要求primitive-specific done/visibility；该句没有新增哪一原语的保证。',1,16)
M(43,'CTA存续/peer可达性是开放硬件定义；当前单CTA未使用peer能力，不能以未使用推断硬件条目无效。',16)
T(44,'实现栈名称和版本是有意开放的身份绑定，不能由Device或Design推导。',16)
M(45,'选定torch/cuda-python/编译链身份必要，末句为未记录编译器/驱动的来源限制；不能整体删。',5,16)
F(46,'规定32-token分组及FP32合并路径；sum32本身不规定分组，不属冗余精度。')
M(47,'主动fast近似路径是核心数值限定，误差未建立是必要来源限制。',2,5)
F(48,'把recurrence近似显式关联到chunk定义；DOC3不允许跨步骤默默继承arithmetic。')
E(49,'账本计量对象标签Token recurrence。')
E(50,'账本计量对象标签State dependency。')
F(51,'定量对照由Contract token递推与rewrite的Np递推推导，非新的执行决定。',core='none')
E(52,'baseline选择用于理解差额，直接递推与无全局中间state不约束现有Design。')
F(53,'按C/shape推导局部额外元素量级；不是安全可删字数或实测峰值。',core='none')
F(54,'ledger.by中的rewrite step引用，不是自然语言正文。',core='none')
F(55,'ledger.by的chunk正式引用，所有depth/footprint/error作用域已核对。',core='none')
E(56,'账本误差条目标签。')
E(57,'fast及BF16误差来源均有公式cast/arithmetic定义；作为账本摘要有阅读价值。')
E(58,'账本矩阵乘条目标签。')
F(59,'ledger.by的inverse正式引用；五处账本引用均对应已定义step。',core='none')
E(60,'账本footprint的对象标签。')
F(61,'3*L_b*DV*DK及替代量级由递推和block公式推导；保留为量化账本，不是新算法。',core='none')
F(62,'KK/QK/KS/QS/U/Y/S七个dense product加hierarchical inverse的计数。',core='none')
F(63,'Gauss-Jordan 8x8、分层8→64及中间BF16舍入共同定义正式算法，不能因英文句式当成解释。')
F(64,'single本地chunk定义用b=0,L_b=T，不读取Cu32，但保留已提交转换。')
F(65,'root的完整精度分叉、初态/末态及packed输出索引；Sdec和Udec来源不能删。')
F(66,'single的完整递推及直接输出索引，已核对与共享Contract的一致绑定。')
F(67,'root的Cu32边界、偶数padding、neutral gates及前缀公式全部正式必要。')
E(68,'Beta=0的padding由chunk.math定义。')
E(69,'Gamma相对初态的作用由chunk与recurrence公式确定。')
E(70,'U的FP32结果由mm32与dtype确定。')
E(71,'Ubar=bf16(U)及Y公式确定读出用途。')
E(72,'直接从U衰减并窄化由Udec=bf16(U⊙Decay)确定，不能反向把本标签当定义来源。')
E(73,'K的padding由chunk索引和边界分支给出。')
E(74,'W的共享由keep_value、A/Bmat公式及Wpair资源给出。')
E(75,'Bmat的衰减、scale与因果性来自inverse.math。')
E(76,'G的inclusive范围与log2在chunk.math定义。')
E(77,'Tinv的列beta缩放和舍入来自inverse.math。')
E(78,'KS的投影由mm32(Sbar,Kᵀ)定义。')
D(79,'single的Cu32无handoff、无gdn invariant/消费者且公式用T，故未消费已由完整结构确定；不能删index_cast本身。',7)
E(80,'两次BF16舍入已显式写入R公式。')
E(81,'Sbar=bf16(S(c-1))和Sdec并列公式已定义前衰减快照。')
E(82,'C是token chunk轴、value=64，标签保留可读性。')
E(83,'KK=mm32(K,Kᵀ)唯一确定Gram量。')
E(84,'V的padding由chunk.math给出。')
D(85,'root flow.handoff.Cu32与gdn invariant及chunk公式明确core消费。',7)
E(86,'Y由mma32得到，最后O=bf16(Y)；标签不新增舍入。')
E(87,'Sdec的Gamma[C-1]公式定义末端缩放。')
E(88,'S的fp32、dv/dk及carry_forward范围已有结构/公式。')
E(89,'QS由mm32(Sbar,Qᵀ)定义。')
E(90,'QK由mm32(Q,Kᵀ)定义。')
D(91,'A数值由inverse.math，原地逆workspace由Tring的A/Tinv同槽view与invert定义承载。',1,9)
E(92,'Alpha=1 padding在chunk.math定义。')
E(93,'QSscaled的scale/Gamma乘法由recurrence.math定义。')
E(94,'Decay由G末项减G[i]正式定义。')
E(95,'Q padding在chunk.math定义。')
E(96,'Np=2*ceil(L_b/(2*C))已给出偶数和当前序列域。')
E(97,'thesis总结已定义的chunk/inverse/FP32 carry与ledger代价。')
E(98,'任务摘要；外部buffer覆盖约束以Contract为权威，跨文件相同不单独构成可删理由。')
E(99,'两个compute组、两个issuer、rings和early return分别由role/resource/verb/边定义；是摘要。')
E(100,'four-lookahead/five-stage及ready-only已由single ScalarOutput定义；thesis未独占。')
D(101,'Beta的zero pad在chunk.math，独立5槽BetaRing与旧Tinv读完的reuse已表达acquire条件。',7)
M(102,'Gamma的数学和compute在publish前由上游/pipe覆盖；Alpha/GateRegs的寄存器捕获与合成对象绑定仍靠此句。',10)
D(103,'Tinv/Bmat公式及beta-stage的reuse前驱Inverse/publish:Tinv已覆盖计算和归还时点。',7)
T(104,'Gate是G/Gamma的合成发布对象；资源同槽不自动定义其成员或共同ready事件。',10)
D(105,'S(-1)=initial_state及每t的load和carry_forward重置规则已明确，包括零值输入。',7)
M(106,'公式、数据边与别名重复；独次S读取和合成StateOperand/UpdateOperand仍含核心动作语义。共用ready流是已核实源码事实，但其流身份的独立核心重要性未证。',9,10,11)
D(107,'零累加器由mm32定义，累积done由execution_mode及realization保证；自定义matmul仍需保留简短上游绑定。',8)
M(108,'两chunk Pair→W的动作绑定与捕获必要；gate归还边、Wpair存储及Bmat用途已明示。',9,10)
T(109,'store.done覆盖bulk-store读完O的原语保证，直接支撑Recurrence下轮O覆盖；不能只保留store.issue。',15)
M(110,'Cu32界更新与fence/旧请求在途是不可省的描述符协议；偏移和zero-fill已在math中定义。',5,15)
add(111,'STRUCTURABLE','all','existing_structure','raw相邻四head数学视图加两次异步load的prologue可完整表达本句；无需创造新事件种类。',13)
D(112,'KS/QS/U零初值、Y/QSscaled和S/Sdec由mm32/mma32确定，completion由mode/原语契约确定。',8)
D(113,'Cu32转换由math，合法B+1区域由target/input，PyTorch copy由describe；serial单produce已无设计级异步stage。',7)
M(114,'row Beta捕获与column Beta持续读取是不同寿命；整体BetaRing及Tinv reuse仅承载后者，不能删捕获语义。',9)
M(115,'Gamma/Decay寄存器捕获是必要动作绑定；StateOperand前后顺序、gate归还与可重叠边已在pipe/reuse中。',9)
M(116,'O descriptor更新/fence为开放原语契约；抑制padding由math有效输出域重述。',15)
T(117,'read:U把共享TMEM中的U捕获进寄存器后归还，resource.view只说明别名而不说明捕获复制。',9)
M(118,'单读S与fused写出含必要动作语义；数值舍入、已有数据链和别名部分重复。root独立四通知流属源码事实，不能只凭流数量差异证明核心设计缺字段。',9,10,11)
M(119,'固定T边界已在math/realization，bulk-store读完才能归还O的完成含义须保留。',15)
M(120,'双raw完成、h%4捕获和c+2先发射是内部协议核心；c%2/c%5及store(c)→prepare(c+5)已由resource/serial pipe确定。',11,13)
T(121,'KK/QK从TMEM提前捕获到register是核心存储/重叠事实；边只证明read完成，不能替代动作读何处写何处。',9)
T(122,'每级动态改变warp分工且组间rendezvous是核心执行分配；静态member无逐动作切片，现有拆role/动作表示成本较高。',12,structure='new_structure_candidate')
T(123,'Scalar合成对象对应G/Gamma/Beta全写完的联合ready；名称和ScalarRing容量不能自己定义成员。',10)
M(124,'捕获Gamma/Decay必要；StateOperand后和潜在KS/QS重叠由serial顺序与独立MMA role决定。',9)
D(125,'Tinv/Bmat的列beta/权重来自inverse.math，W保留来自Wpair与prepare定义；可短写绑定。',1)
add(126,'STRUCTURABLE','all','existing_structure','可把rescale:QS换为既有compute:QSscaled，并同步事件引用；公式与同址Arena view完整定义其计算和原地目标。',14)
M(127,'Pair到W/row Beta的捕获以及column Beta持续读取是核心；Wpair及Tinv/Bmat公式已覆盖保留/用途的一部分。',9,10)
D(128,'embedded descriptor由本节点realization绑定，T界与synthetic zero-fill由chunk.math定义。',15)
M(129,'源码确用四独立PipelineAsyncUmma并省略producer acquire；读者退役已由现有依赖链保证，未证该API或通知流数量有独立一阶影响。保留来源事实，其独立核心重要性未决。',11,15,core='uncertain')
T(130,'warp mma.sync与两个异步tcgen05 issuer的引擎分离不能仅由BF16公式或warp成员推导。',15)
M(131,'tcgen05绑定必要；single-CTA、TMEM和cumulative completion由mapping/resource/mode重复。',8,15)
M(132,'TensorMapManager/PipelineCpAsync/PipelineAsync提供具体保证；可变界与分离阶段的设计决定已在Design。',15)
M(133,'raw事务完成、warp rendezvous和async-shared fence提供必要可见性保证；ready-only/compact是动作契约重复的来源协议，其通知存储/API选择的独立性能重要性未证。',11,13,15)
E(134,'解释tcgen05为何支撑已有alias/reuse，新增保证应以matmul契约为准。',8)
E(135,'解释inverse独立引擎选择的理由，相关绑定在choice中。',15)
M(136,'首半为已有设计理由，末半是未建立旧descriptor在途读取顺序的必要限制，不能省成已证正确。',5,15)
E(137,'recurrent复用已在数据链中；full/empty握手可能只落实传递上已满足的条件。本句声称影响overlap，但缺少新增非传递等待或一阶成本证据。',11)
E(138,'transfer visibility/ready-only的理由，具体fence与原语已由choice给出。',13,15)
E(139,'root多序列descriptor和独立流的摘要，详细定义在pipeline/realization。')
E(140,'single固定界、TMA标量及compact事件的摘要。')
E(141,'PyTorch copy/版本在Convert动作与stack，timing在Contract，前后提交由flow定义；摘要保留阅读用途。',7)
E(142,'one-worktile由空partition/target/mapping，512容量只解释库lowering容纳B+1；本次未独立取得PyTorch该kernel源码。',7)
D(143,'统一preassigned来自work_schedule，flatten n定义跨t游标，DOC5默认不排空且math/pipe每t重置S。',7)
M(144,'P公式是核心但可内联worker_count；runtime num_sm绑定必要，线性id已由order定义，148-SM/64CTA是实例解释。',6)
D(145,'Np域/pipe n[t]、Cu32 invariant和single_tile双半输出已明确无跨半合并。',7)
M(146,'核心P数量可写进worker_count；runtime查询仍是开放绑定，linear id由order确定，148仅来源实例。',6)
M(147,'Np/双半独立与Cu32不消费已有结构；passed是调用ABI实现说明，不新增计算依赖。',7,core='none')
E(148,'dv.tile=64、full recurrent worktile和重复Q/K/inverse可推出；保留thesis作主要成本解释。')

findings=[]
def finding(n,kind,idx,claim,core,counter,candidate,gap='none',evidence=(),confidence='high'):
    findings.append(dict(id=f'{OP}-F{n:02}',kind=kind,unit_ids=[U[i]['id'] for i in idx],claim=claim,
        core_fact=core,deletion_counterexample=counter,existing_structural_candidate=candidate,
        gap_kind=gap,evidence=list(evidence),confidence=confidence))

finding(1,'redundancy',[3,4,5,6,7,8,9,10,11,13,14,16,17,18,19,20,30,31,32,33,34,35,36,37,49,50,52,56,57,58,60,68,69,70,71,72,73,74,75,76,77,78,80,81,82,83,84,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,125,139,140,148],
 '对象标签、thesis、dispatch和部分账本为有用摘要；可缩的依据是当前公式/结构，而非另一文件文本相同。',
 '核心数值唯一来源为rewrite/Contract，布局和角色唯一来源为resource/mapping/pipe。',
 '去掉标签不会改变已定数值或执行；但必填meaning/thesis以及custom matmul等仍须合法短写，不能删除所有字段。',
 '保留简短meaning/必填thesis；例如Udec.meaning: End-decayed update. 其直接从FP32 U窄化仍由Udec=bf16(U⊙Decay)定义。',
 evidence=['inputs/template_gdn_prefill/root.yaml:120','inputs/template_gdn_prefill/root.yaml:195','rules/schema_docs/3.design.math_reshape.md:85','rules/README.md:80'])
finding(2,'style',[2,21,22,23,24,25,27,28,29,38,39,46,47,48,51,53,54,55,59,61,62,63,64,65,66,67],
 '正式关系包括英文算法步骤、范围、精度分组、ledger计数及step引用，不能计为自由叙述漏洞。',
 '偶数chunk padding、BF16多次窄化、FP32 carry、8→64 inverse、32-token分组均影响既定数值。',
 '删除R的内层bf16或把Udec由Ubar计算将允许不同舍入路径；删除prefix分组允许另一sum32归约顺序。',
 '既有math及arithmetic足够：R=bf16(Vᵀ-bf16(KS⊙Gamma[None,:])); Udec=bf16(U⊙Decay[None,:])。无需新增字段。',
 evidence=['inputs/template_gdn_prefill/root.yaml:101','inputs/template_gdn_prefill/root.yaml:105','inputs/template_gdn_prefill/root.yaml:120','rules/schema_docs/3.design.math_reshape.md:223'])
finding(3,'structural_gap',[2,4,11,12,14,15,26,28,29],
 '外部返回顺序、对象别名和仅指定输入可变依赖Contract文字，input/output shape或ref属性不能表示对象身份。',
 'O is output，S_final is output_state；两个原buffer完整覆盖，其他输入不变。',
 '删掉别名/副作用文字后，返回新分配同值张量且不写原buffer与原地写回都满足剩余值公式。NaN初值也可能被错误累加。',
 '已查output.meaning、compute.math、shape/dtype ref、flow.storage及resource.view：后两者分别是内部物化与局部资源别名，不能合法表示外部Python对象别名。现有方式只能保留meaning/compute中的关系；外部alias/mutation字段是候选，未自行采用。',
 'missing_field_or_event',evidence=['inputs/template_gdn_prefill/root.yaml:26','inputs/template_gdn_prefill/root.yaml:29','inputs/template_gdn_prefill/root.yaml:43','rules/schema_docs/1.contract.algorithm_requirement.md:38','work/template_gdn_prefill/source/aka_solution/solution.py:812'])
finding(4,'structural_gap',[2,22,24,25,26,27],
 '输入分布、normalization、初态来源可写正式关系；连续CUDA布局与计时域则仍为Contract中的开放任务环境语义。',
 '输入生成/normalize/前缀评估不在计时中，实际forward含index_cast；有效域允许zero/continuation、排除空序列。',
 '删除计时域会允许把生成或prefix计入成本，删除contiguous会允许不同地址步长输入；删除初态域会把任意FP32 state误当等价测试域。',
 '可将生成关系压缩为q/k/v的range谓词，并继续用compute给出联合分布；现有Contract没有contiguity/benchmark_scope字段。公式化分布不是扩语言，计时字段缺少只是任务环境表达候选。',
 'open_semantics',evidence=['inputs/template_gdn_prefill/root.yaml:33','rules/schema_docs/1.contract.algorithm_requirement.md:45','work/template_gdn_prefill/source/input.py:24','work/template_gdn_prefill/source/input.py:47'])
finding(5,'source_limitation',[1,45,47,110,136],
 '三类限制须保留：oracle blocked FP32比较值、fast log2/exp2无可靠误差界、多序列旧descriptor在途读与更新顺序未建立；编译器/驱动记录不全也不能伪造。',
 '不能把数值误差边界或descriptor更新安全当成已获证明；已有schema不是GPU验证基线。',
 '删限制后读者可能把实数递推当逐位oracle，或认为fence证明旧请求读完；后二者都不是当前证据所支持。',
 '已有golden_reference.gap、arithmetic.approx和realization.reason可承载短限制；新增reuse边会改变已抽取设计，不能以修词自动补入。',
 'unresolved',evidence=['inputs/template_gdn_prefill/root.yaml:53','inputs/template_gdn_prefill/root.yaml:101','inputs/template_gdn_prefill/root.yaml:322','work/template_gdn_prefill/source/reference.py:269','work/template_gdn_prefill/source/aka_solution/solution.py:222'])
finding(6,'existing_structure',[144,146],
 'P数量公式目前仅在mapping.strategy，而worker_count填P；可以用现有数值表达式承载数量，无须新字段。',
 'worker_count=min(worktile_count,num_sm)，工作按既有batch/head/dv顺序展平并步长分配；num_sm来自运行时设备属性。',
 '只删strategy将使P未绑定，任意CTA数量均可能满足worker_count:P；固定148也会把runtime查询改成常量设计。',
 'mapping: {strategy: "num_sm is the runtime device SM count.", worker: {unit: CTA, unit_count: 1}, worker_count: "min(worktile_count,num_sm)"}\nschedule.worker_assignment: {preassigned: {start: worker_id, step: worker_count}, refill: none, exit: assigned_done}\n保留现有worktile_order，线性id由DOC4推导。',
 evidence=['inputs/template_gdn_prefill/root.yaml:178','inputs/template_gdn_prefill/m64_single_sequence.yaml:112','rules/schema_docs/4.design.work_schedule.md:167','work/template_gdn_prefill/source/aka_solution/solution.py:35','work/template_gdn_prefill/source/aka_solution/solution.py:794'])
finding(7,'redundancy',[79,85,101,103,105,113,141,142,143,145,147],
 '已有flow、域、资源、边及共享默认覆盖一批重复叙述：转换提交、零初态加载、Beta归还、跨worktile游标和不排空。',
 'single虽无Cu32消费者仍保留实际index_cast；root flattened n跨工作，状态按每t重载；Beta复用等Tinv。',
 '这些结构保留时缩句不改变设计；若连index_cast一起删则丢一次实际kernel，或把n重置则改变ring复用，均不在删减候选内。',
 '当前已写：flow.launch: [{phase: [index_cast]}, {phase: [gdn], start_after: index_cast.kernel_done}]；domain: {t: worktile, c: [0,Np], n: {flatten: [t,c]}}；每t先load:initial_state(t)。DOC5默认资源跨工作接续，不额外加drain。',
 evidence=['inputs/template_gdn_prefill/root.yaml:151','inputs/template_gdn_prefill/root.yaml:182','inputs/template_gdn_prefill/root.yaml:194','inputs/template_gdn_prefill/root.yaml:277','inputs/template_gdn_prefill/m64_single_sequence.yaml:85','rules/schema_docs/5.design.tile_pipeline.md:383','work/template_gdn_prefill/source/aka_solution/solution.py:812'])
finding(8,'redundancy',[107,112,131,134],
 'matmul零初值/累加目标与cumulative completion重复了mm32/mma32、execution_mode及原语绑定。',
 '数学产物名绑定仍需一句；operand-read完成保证不能在没有原语证据时凭mode猜测。',
 '保留math、async_ordered_completion及tcgen05契约时删重复不会丢数值；若省略operand-read语义的最后一份定义，Sbar/R与累加器alias可能过早覆盖。',
 'verb.matmul: "Compute the named upstream matrix product; done covers operand reads."\ninfo.execution_mode: async_ordered_completion\n保留realization对tcgen05的绑定；零初始及C参数由mm32/mma32确定。',
 evidence=['inputs/template_gdn_prefill/root.yaml:228','inputs/template_gdn_prefill/root.yaml:253','inputs/template_gdn_prefill/root.yaml:315','rules/schema_docs/5.design.tile_pipeline.md:112','rules/schema_docs/3.design.math_reshape.md:231'])
finding(9,'structural_gap',[91,106,108,114,115,117,118,121,124,127],
 '资源别名已结构化，但从别名槽捕获到寄存器、提前归还以及row/column Beta不同寿命仍依赖动作契约。',
 'KK/QK/U读到register后允许TMEM覆盖；Wpair保留到Bmat；row Beta可先捕获但column Beta读到Tinv。',
 '只保留read名和reuse边，可把read实现为晚读取TMEM的view，或者提前归还Beta而Tinv继续读SMEM；会破坏存储安全或失去提前归还的重叠。',
 'resource.space/register、view.part、waits_for.reuse可记容量/同址/读完成；没有通用read/write access-set或capture目的值绑定。可拆出已定义捕获值与load/release动作，但必须另给捕获数学/存储关系，不能只造Captured名当已解决。当前短动作契约属有意开放语义。',
 'open_semantics',evidence=['inputs/template_gdn_prefill/root.yaml:204','inputs/template_gdn_prefill/root.yaml:211','inputs/template_gdn_prefill/root.yaml:240','inputs/template_gdn_prefill/root.yaml:274','rules/schema_docs/5.design.tile_pipeline.md:298','work/template_gdn_prefill/source/aka_solution/solution.py:478','work/template_gdn_prefill/source/aka_solution/solution.py:724'])
finding(10,'structural_gap',[102,104,106,108,118,123,127],
 'StateOperand/Gate/Scalar/UpdateOperand/Pair是合成接口，名称未在上游定义；组成和一次读取复用只能从verb正文确定。',
 'StateOperand一次读S同时产Sbar并原地Sdec；Gate={G,Gamma}、Scalar={G,Gamma,Beta}、UpdateOperand={Ubar,Udec}在各自done后共同可读。',
 '删合成定义后，一个实现只发布Sbar或Ubar，另一个完成全部组合，都能使用相同动作名和剩余边；多次重读S也不违反现有资源容量。',
 '已查resource.view、math.symbol、milestone：view只声明存储，milestone只给已有事件组命名，均不能声明单动作包含哪些写入或单读复用。拆成多个publish并显式同步可表达顺序但改变原子发布粒度/性能；目前保留短合成契约，候选为结构化动作读写/发布成员，未引入新字段。',
 'open_semantics',evidence=['inputs/template_gdn_prefill/root.yaml:269','inputs/template_gdn_prefill/root.yaml:303','inputs/template_gdn_prefill/m64_single_sequence.yaml:196','inputs/template_gdn_prefill/m64_single_sequence.yaml:233','rules/schema_docs/5.design.tile_pipeline.md:81','work/template_gdn_prefill/source/aka_solution/solution.py:679'])
finding(11,'structural_gap',[106,118,120,129,133,137],
 '已核实源码用独立/共用通知流及ready-only协议，但未证明通知流身份需要专用核心字段；保留为开放的来源协议绑定，撤回missing_field_or_event结论。',
 '已证核心关系是数据发布及全部旧读者退役。root四条通知流、single共用一流、single Scalar五个ready-only槽是来源事实；它们的物理barrier数量/API身份是否另有一阶影响未决。single把Ubar/Udec合为UpdateOperand的发布关系已在pipe/data体现，不应再算流身份缺口。',
 '未找到只改变流身份/full-empty形式就必然新增不可推导等待的反例：StateOperand(n)已等S(n-1).done，MMA1累积完成覆盖旧KS/QS；Ubar覆盖R前已等U(n).done；R(n)和Udec(n)所需旧读者也由上一S.done传递保证。single prepare(c+5)在store:O(c).done之后，后者传递覆盖旧scalar读者。若full-empty在这些已有退役点归还，acquire只等待已满足条件；增加barrier存储/指令不等于增加图依赖，现有证据也未量化一阶性能影响。',
 '保留现有pipe、data/reuse和operand-read completion契约即可表达已证发布/退役关系；require/verb可以简短保存源码通知API及ready-only选择。没有专用流身份字段本身不证明核心表达不足；若未来证据显示不同协议引入新的等待/发布位置，应先用现有动作和边表达，再评估紧凑表示需求。此次不提出新增通知流字段。',
 'open_semantics',evidence=['inputs/template_gdn_prefill/root.yaml:253','inputs/template_gdn_prefill/root.yaml:269','inputs/template_gdn_prefill/root.yaml:286','inputs/template_gdn_prefill/root.yaml:324','inputs/template_gdn_prefill/m64_single_sequence.yaml:196','inputs/template_gdn_prefill/m64_single_sequence.yaml:229','inputs/template_gdn_prefill/m64_single_sequence.yaml:235','rules/schema_docs/5.design.tile_pipeline.md:112','work/template_gdn_prefill/source/aka_solution/solution.py:154','work/template_gdn_prefill/source/aka_solution/solution.py:162','work/template_gdn_prefill/source/aka_solution/solution.py:349'],confidence='medium')
finding(12,'structural_gap',[122],
 'Inverse在8/32/64层每chunk两个warp，而16层四warp依次处理两个chunk；固定role.member不能表达逐层分工。',
 '组内rendezvous和每级不同成员工作分配决定inverse critical path及两chunk并行度。',
 '删这句后，用全四warp串行做每个chunk和两warp同时处理两个chunk都满足member=warp[0:4]与invert:Pair，但吞吐/同步不同。',
 '现有语法可把固定warp子集拆role，按stage/level列pipe并用sync边rendezvous；需显式展开16层与其他层不同工作区域且为各动作定义数学绑定，不能让info.member依赖repeat变量。没有已审核的紧凑等价替稿，因此归为表示成本候选，保留原句。',
 'representation_cost',evidence=['inputs/template_gdn_prefill/root.yaml:242','inputs/template_gdn_prefill/m64_single_sequence.yaml:174','rules/schema_docs/5.design.tile_pipeline.md:194','work/template_gdn_prefill/source/aka_solution/solution.py:518','work/template_gdn_prefill/source/aka_solution/solution.py:523'])
finding(13,'existing_structure',[111,120,133],
 'raw标量相邻四head、首两次load和c+2条件预取可以迁入现有公式/repeat/事件；当前单个prepare隐藏内部异步工作，部分原语可见性仍需开放契约。',
 'raw读取h所在四head组；capture h%4且读完才覆盖c%2；c+2预取先于本chunk的G/Gamma计算；processed c%5由输出进度保护。',
 '删内部顺序可先算G再预取，失去重叠；删四head绑定可取错列；把五槽误作完整acquire/release会改变协议。',
 '数学视图：RawScalar(c)[0,i,r]=alpha[c*C+i,4*floor(h/4)+r], RawScalar(c)[1,i,r]=beta[c*C+i,4*floor(h/4)+r], 0<=r<4；域外raw采用TMA零填充，随后逻辑Alpha用1替换域外行。\n首发可用：prologue: [{repeat: {j: [0,2]}, steady: ["load:RawScalar(n[t,j])"]}]，load采用execution_mode: async。\n对内部次序可拆现有read契约为捕获column h%4的serial动作，再用repeat: {r: ["min(c+2,Np)","min(c+3,Np)"]}发射load:RawScalar(n[t,r])，然后compute:G/Gamma；read等待对应load.done。此局部片段说明可表达性，未宣称完整短稿已等价审阅；warp fence/事务合并仍保留realization。',
 evidence=['inputs/template_gdn_prefill/m64_single_sequence.yaml:136','inputs/template_gdn_prefill/m64_single_sequence.yaml:228','inputs/template_gdn_prefill/m64_single_sequence.yaml:235','rules/schema_docs/5.design.tile_pipeline.md:145','work/template_gdn_prefill/source/aka_solution/solution.py:271','work/template_gdn_prefill/source/aka_solution/solution.py:280'])
finding(14,'existing_structure',[126],
 'rescale是自定义动作，但其全部语义已可由compute:QSscaled、上游公式和Arena别名表示。',
 'QSscaled原地覆盖QS，等QS完成后执行，MMA1/Y等缩放完成。',
 '只删verb而保留rescale名会留下未定义动作；连同动作与引用改成compute:QSscaled则数值和先后保持。',
 'pipe中"rescale:QS(n)"→"compute:QSscaled(n)"；data: {"compute:QSscaled(n)@issue": ["MMA1/matmul:QS(n)@done"]}；MMA1的Y前驱改"Recurrence/compute:QSscaled(n)@done"。保留Arena中QS/QSscaled同part，以及math的QSscaled=(QS⊙Gamma)*fp32(scale)。不新增动作词或事件种类。',
 evidence=['inputs/template_gdn_prefill/root.yaml:125','inputs/template_gdn_prefill/root.yaml:212','inputs/template_gdn_prefill/root.yaml:275','inputs/template_gdn_prefill/root.yaml:259','rules/schema_docs/5.design.tile_pipeline.md:87'])
finding(15,'structural_gap',[109,110,116,119,128,129,130,131,132,133,135,136,138],
 '原语身份和原语自身可见性/descriptor更新行为是有意开放绑定；物理DMA完成与通用done并非由字符串动作名自行证明。',
 'tcgen05读完成/累积顺序、warp MMA独立引擎、mutable或embedded TMA界、O bulk-store完成、raw fence均需具体保证。',
 '换成不覆盖operand-read的通知或store.issue即归还O，仍可保留同图名称但资源被过早覆盖；descriptor更新不等待旧访问亦可能越序列界。',
 '已用realization.require.item关联Design对象，choice绑定tcgen05/mma.sync/TMA/管线API；未找到闭合的原语语义表可完全去除必要短契约。已有约束图承载设计先后，choice承担实现兼容性，不能据允许自由文本推定图本身足够。',
 'open_semantics',evidence=['inputs/template_gdn_prefill/root.yaml:220','inputs/template_gdn_prefill/root.yaml:305','inputs/template_gdn_prefill/root.yaml:315','inputs/template_gdn_prefill/m64_single_sequence.yaml:245','rules/schema_docs/6.realization.md:9','work/template_gdn_prefill/source/aka_solution/solution.py:737'])
finding(16,'structural_gap',[40,41,42,43,44,45],
 '硬件execution/feature及软件栈是有意开放的设备事实；冻结wiki仍为pending，不能假设已提供完整共享语义。',
 '存储层级所有权、TMEM独立分配、执行组成及版本身份须在本任务定义或真实资料绑定中可读。',
 '仅留SMEM/TMEM/register键名会允许不同存储所有权/可见性解释；仅留L20D不确定CuTe/torch版本。',
 'DOC2的execution/feature值限定为摘要字符串，禁止任意嵌套；wiki引用是既有复用入口，但当前pending不是真实替代。设备统一事实可在将来已固定wiki中共享，此处不擅自新增硬件结构。',
 'open_semantics',evidence=['inputs/template_gdn_prefill/root.yaml:58','inputs/template_gdn_prefill/root.yaml:62','inputs/template_gdn_prefill/root.yaml:67','inputs/template_gdn_prefill/root.yaml:70','rules/schema_docs/2.contract.hardware_specification.md:79'])
finding(17,'source_limitation',[110,132,136],
 '跨调用边界：源码按编译配置/device缓存TensorMap workspace，root的可变descriptor需额外核对并发stream调用是否允许；当前文本仅描述一次worker执行。',
 '一次调用的phase/worker身份不能自动为跨调用复用的同一物理descriptor存储建立退役关系。',
 '若同配置同device允许两stream并发调用，共享workspace可能在旧TMA读取期间被另一次调用更新；若调用串行则无此反例。本轮未取得并发调用约束，故不宣称已证race。',
 'DOC0为每次真实执行建立新事件实例，不能靠重命名调用消除共享存储；flow.kernel_done可表达同一次plan内顺序，现有节点没有跨调用workspace所有权契约。需核对调用域/库的并发保证或保留来源限制，不能擅自新增跨stream同步。single使用embedded descriptors，不依赖这份mutable workspace。',
 'unresolved',evidence=['inputs/template_gdn_prefill/root.yaml:321','rules/schema_docs/0.common.md:41','work/template_gdn_prefill/source/aka_solution/solution.py:803','work/template_gdn_prefill/source/aka_solution/solution.py:805'],confidence='medium')
for i in (110,132,136): R[i]['finding_ids'].append(f'{OP}-F17')

rem=[]
def cut(i,quote,replacement,reason,kind='redundancy'):
    assert U[i]['text'].count(quote)==1,(i,quote)
    assert len(replacement)<len(quote)
    rem.append(dict(unit_id=U[i]['id'],quote=quote,replacement=replacement,kind=kind,reason=reason,
                    evidence=R[i]['evidence'],applies_to_occurrences='all'))

# Only these reviewed exact substrings contribute to conservative savings.
cut(30,U[30]['text'],'精确选择条件与版本以各节点 YAML 为准。','DOC0已确定共享Contract、parent和version，保留索引指引。')
cut(72,U[72]['text'],'End-decayed update.','直接从U窄化在两个recurrence.math中明确。')
cut(80,U[80]['text'],'Residual.','R公式保留双bf16。')
cut(81,U[81]['text'],'State snapshot.','Sbar公式在衰减前从S(c-1)窄化。')
cut(91,U[91]['text'],'Feedback workspace.','bf16由公式/dtype，A/Tinv同址由Tring view和invert定义。')
cut(101,U[101]['text'],'Copy Beta into its stage.','Beta zero-pad在math，独立槽与acquire旧读者边在BetaRing/reuse。')
cut(105,U[105]['text'],'Load S from initial_state for this worktile.','保留load动作的S绑定；无default且公式S(-1)包含零值输入。')
cut(107,U[107]['text'],'Compute the named upstream matrix product; done covers operand reads.','mm32给零FP32累加；mode给此前完成，保留custom product绑定和read completion。')
cut(112,U[112]['text'],'Compute the named upstream matrix product; done covers operand reads.','mm32/mma32已规定零、QSscaled、Sdec；保留必要动作/原语绑定。')
cut(113,U[113]['text'],'Perform the upstream Cu32 conversion with the contiguous PyTorch copy kernel.','B+1有效域由shape/target，serial单动作无设计级异步stage。')
cut(115,' after StateOperand publication','', 'serial pipe已将compute:Decay放在publish:StateOperand之后。')
cut(115,', and overlap with KS/QS','', 'MMA1独立推进且仅受声明边约束；不把潜在overlap写成额外保证。')
cut(120,'Raw stages use c%2; write G/Gamma/Beta into processed stage c%5. ','','RawScalar/ScalarRing view已完全给出槽映射；保留接下来的ready-only限定及读者保护解释。')
cut(121,' before subsequent weighting arithmetic','', 'Inverse的serial pipe将read放在compute:A或publish:Bmat之前。')
cut(124,' after StateOperand publication, permitting overlap with KS/QS','', 'same-role serial及独立MMA1的已声明边共同确定。')
cut(143,U[143]['text'],'','DOC5默认跨工作不排空、flatten n跨t、每t load和carry_forward重置已全部存在。')
cut(144,' For B=1, linear id=2*h+row_half; the recorded 148-SM run gives 64 CTAs, one worktile each.','','保留P/runtime定义；linear id和64个worktile由B=1、HV/DV和worktile_order推导，148仅运行例。')
cut(145,U[145]['text'],'','Cu32 invariant、Np domain、n[t] pipe和single_tile输出已表达全部含义。')
cut(146,'; linear id=b*(2*HV)+2*h+row_half. The source L20D run records num_sm=148.','.','保留runtime P；worktile_order决定id，148并非设计常量。')

assert set(R)==set(range(1,149))
annotations=dict(operator=OP,prompt_version='v1.1',unit_annotations=list(R.values()),findings=findings,
 removal_candidates=rem,node_review=[
 dict(node='root',reviewed=True,effective_binding_checked=True,notes='template v7；Contract完整共享；三区本地，B>1。index_cast→gdn及Cu32 handoff；12 warp、双inverse/recurrent组；变长偶数padding、跨t flatten环、双gate读者与K/Q双issuer复用均逐一核对；descriptor旧请求顺序未证。'),
 dict(node='m64_single_sequence',reviewed=True,effective_binding_checked=True,notes='template v6；B=1独立于父when；共享root Contract，三区均本地。保留无消费者Cu32转换；固定T、raw双槽/processed五槽/四chunk lookahead，共用Recurrence ready流；首末chunk、Np=2/4及长序列、零/continuation初态已覆盖。')],
 limits=[
 '审计对象为规范化template，非GPU/语义/性能已验证基线；没有运行GPU、修改原schema或Runner。',
 '固定source commit 01fff90d714abc6e84904ea9c44523ba66d34a0e；亲读selected aka_solution/solution.py的scheduler、dispatch、各本例活跃role/原语调用路径，reference.py和input.py；未重新审计全部低层inverse汇编或安装版PyTorch copy内核的512元素容量。',
 'fast log2/exp2误差界、TensorMapManager旧descriptor读取与更新顺序、pending wiki及独立编译器/driver版本仍未补齐。',
 '未独立验证并发调用共享tensormap workspace的stream安全；source缓存按device复用workspace，跨调用/外层capture回放只能按DOC0区分实际执行，当前源码没有自有capture/fallback计算分支。',
 '逐字段面积不是可删量；removal_candidates只是保持原结构的保守短写候选，未生成或批准完整等价短稿。结构化片段是可审查建议而非已应用修改。',
 '现有from_parent只支持整区；本例math有Cu32/输出索引差异、schedule和pipeline有实际差异，不能仅凭跨文件相同文字删除子节点定义。'
 ])

# v1.1 calibrated additions, all independently read at their occurrence sites.
extra = [
 ('t_cd916003912abd86','EXPLANATION','none','already_present','compute标题；更新后inclusive readout已经由State/O的公式确定。',1),
 ('t_680d6642a8764eac','ESSENTIAL_TEXT','all','open_semantics','CTA成员共驻SM并共享SMEM/barrier是硬件层级定义，DOC2不给隐式默认；下游单CTA依赖该组成。',16),
 ('t_6f5eff329130bc7c','EXPLANATION','none','not_applicable','cluster的设备空间摘要；本例worker.unit_count=1且无cluster placement/peer访问，因此对既定两节点不增加核心约束；不能据此修改共用设备事实。',16),
 ('t_707ed1a5b6536df5','ESSENTIAL_TEXT','all','open_semantics','warp=32及CTA内从属是role规模和寄存器/线程预算的必要硬件定义；无独立数值组成字段。',16),
 ('t_432c9e0b148109b9','EXPLANATION','none','already_present','inverse标题，BF16 merge rounding全部保留在inverse.math。',1),
 ('t_de20a419c9fd7490','EXPLANATION','none','already_present','chunk标题，even pairs和prefix由Np/G公式定义。',1),
 ('t_f20bc386256985f1','EXPLANATION','none','already_present','recurrence标题，值修正和state更新已有公式。',1),
 ('t_48857c9a607fe610','FORMAL','all','formula_or_predicate','rewrite正式step身份，ledger.by与算术跨步引用需要该名字；不是自然语言正文。',2),
 ('t_4e32c8e46d5b4f61','FORMAL','all','formula_or_predicate','recurrence正式step身份；与ledger.by中相同文本分属不同字段族。',2),
 ('t_7b767d78cd5bd935','FORMAL','all','formula_or_predicate','chunk正式step身份，recurrence.arithmetic明确指回本步近似定义。',2),
]
for uid,cat,core,structure,reason,fid in extra:
    u=byid[uid]
    annotations['unit_annotations'].append(dict(id=uid,category=cat,core=core,structure=structure,reason=reason,
      evidence=[f"inputs/{OP}/{o['node']}.yaml:{o['line']}" for o in u['occurrences']],
      finding_ids=[f'{OP}-F{fid:02}'],contexts_checked=True,occurrence_overrides=[]))
    findings[fid-1]['unit_ids'].append(uid)
assert {a['id'] for a in annotations['unit_annotations']} == set(byid)

if __name__=='__main__':
    out=B/'annotations'/f'{OP}.json'; out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(annotations,ensure_ascii=False,indent=2)+'\n')
    print(out)
