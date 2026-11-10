import json,pathlib,collections
R=pathlib.Path('schema_extract/archive/20261109-ten-operator-text-audit'); O='template_fa4_prefill'; U=[json.loads(x) for x in (R/'data'/O/'units.jsonl').read_text().splitlines()]; A={}; F=[]; C=[]
def evidence(nums):
 return list(dict.fromkeys(f'inputs/{O}/{o["node"]}.yaml:{o["line"]}' if o['node']!='dispatch.md' else f'inputs/{O}/dispatch.md:{o["line"]}' for n in nums for o in U[n-1]['occurrences']))
def ann(nums,cat,core,structure,reason,fs=()):
 for n in nums:
  assert n not in A,n
  u=U[n-1];A[n]={'id':u['id'],'category':cat,'core':core,'structure':structure,'reason':reason,'evidence':evidence([n]),'finding_ids':[O+'-F'+str(f).zfill(2) for f in fs],'contexts_checked':True,'occurrence_overrides':[]}
def finding(n,kind,nums,claim,core,counter,candidate,gap='none',extra=(),confidence='high'):
 F.append({'id':O+'-F'+str(n).zfill(2),'kind':kind,'unit_ids':[U[i-1]['id'] for i in nums],'claim':claim,'core_fact':core,'deletion_counterexample':counter,'existing_structural_candidate':candidate,'gap_kind':gap,'evidence':evidence(nums)+list(extra),'confidence':confidence})
def removal(n,quote,replacement,reason,kind='redundancy',extra=()):
 assert U[n-1]['text'].count(quote)==1,(n,quote)
 C.append({'unit_id':U[n-1]['id'],'quote':quote,'replacement':replacement,'kind':kind,'reason':reason,'evidence':evidence([n])+list(extra),'applies_to_occurrences':'all'})
# Explicitly selected groups after reading every text and every local scope; no fallback label.
ann([1],'MIXED','some','mixed','ABI/连续布局/计时及可观察副作用与页索引正式关系混合；完整字段不能删。',[1,21])
ann([2],'FORMAL','all','formula_or_predicate','完整可见集、缩放softmax、BF16输出及返回对象同一性关系；是正式契约。',[1])
ann([3,4,5,7,8,10,11,12,13,14,16,18,20,21,22,23,24,25,26,27,28],'EXPLANATION','none','already_present','该对象的短释义；相应compute页映射/长度/因果输出公式、dtype或range已给出具体关系，仍是必填meaning。',[1,3])
ann([6],'NONCORE','none','not_applicable','两固定入口均删除workspace_buffer参数而不读取；首次清零不影响既定计算、执行或主要性能。保留scratch身份短写。',[2])
ann([9,17,19,29],'FORMAL','all','formula_or_predicate','常量的参数定义，区分头数、通道数和页粒度；不按自然语言冗余统计。')
ann([15],'ESSENTIAL_TEXT','all','open_semantics','out_after作为输入out原地内容的身份在meaning中定义；同shape/dtype与名字不建立存储/副作用身份。',[1])
ann([30,31],'FORMAL','all','formula_or_predicate','分别定义允许非有限旧out和页表有效/无效项，不能用默认有限值或普通int32范围替代。',[1])
ann([32,33],'DUPLICATE','none','already_present','Positive由0<Qlen(b)<=L(b)推出；None由已定义python_none单值dtype推出。range可合法省略。',[4])
ann([34,35],'EXPLANATION','none','already_present','compute.name是必填步骤标题；实际关系在紧随math，不是额外计算或执行标识。',[3])
ann([36],'FORMAL','all','formula_or_predicate','atol/rtol为正式验收阈值；两个输出各有声明，不代表两次独立执行。',[21])
ann(list(range(37,54)),'EXPLANATION','none','already_present','逐项核对dispatch树、链接及父差异，均是YAML when/parent和Design的导航摘要；中文索引按DOC0保留。',[3,5])
ann(list(range(54,65)),'FORMAL','all','formula_or_predicate','独立完整选择谓词含默认环境前提、Split及必要的PersistSmall派生量；跨文件重复不等于可省略定义。',[5])
ann([65,66,67,68,69,71],'ESSENTIAL_TEXT','all','open_semantics','执行层级或存储机制是硬件词汇的正式开放摘要；无当前可用wiki实体自动提供组成/生命周期定义。',[6])
ann([70],'DUPLICATE','none','already_present','异步动作须满足各自完成和可见性保证，是DOC5 issue/done与动作契约的通用要求；不提供特有SM103能力。',[6])
ann([72],'MIXED','some','mixed','含工具链/依赖版本身份及未记录compiler/driver的必要来源限制；不能把cu130升级成已核实nvcc。',[6,21])
ann([73],'FORMAL','all','formula_or_predicate','已选CuTe DSL与发行版本身份，不是自然语言论证。',[6])
ann([74],'FORMAL','all','formula_or_predicate','FP32双分量累加种子及四组归约树的正式求值次序；sum32本身不规定这些关系。',[7,8])
ann([75],'SOURCE_LIMIT','uncertain','not_applicable','combine显式引用online的fast-exp2/division未知误差；不得当作误差已验收。',[7,21])
ann([76],'FORMAL','all','formula_or_predicate','规定warp内权重归约及正权项按split升序FP32累加；公式和此精度例外共同约束。',[7,8])
ann([77,78],'MIXED','some','mixed','native fast-math方法/不替换多项式的数值路径与独立误差界未知同句，不能整段归为解释。',[7,21])
ann([79,83,86,125,126,127],'FORMAL','all','formula_or_predicate','rewrite.step与ledger.by是被引用的正式ID；逐节点确认stream/online/combine存在且绑定正确。',[8])
ann([80,82],'FORMAL','none','formula_or_predicate','有范围和单位的占用差额公式；由baseline、状态shape和保留关系推出，仍有对比价值。',[7,8])
ann([81],'MIXED','none','mixed','部分结果流量公式加空split未写与LSE重读限定；这些执行事实也在math/combine正文，保留计量范围。',[7,18])
ann([84,87,88,89],'EXPLANATION','none','not_applicable','账本基线或条目标题帮助解释定量差额，不新增当前Design计算；必填子字段不能单独删空。',[3,7])
ann([85],'DUPLICATE','none','already_present','P=fp8_e4m3fn(E)、ell=sum32(E)、Y使用P已完整表达这一误差来源；账本可读性不等于独占核心。',[7])
ann([90,91,92,94],'FORMAL','all','formula_or_predicate','分支求值、舍入点、递推初值/终值和输出关系均为正式数学；N=0与零权不求值部分须保留。',[7,8,11])
ann([93],'MIXED','some','mixed','页回退和QK分段收缩是正式关系，block-view/no-full-gather是物化约束；不能按长公式删去访问边界。',[7,8,15])
ann(list(range(95,125)),'EXPLANATION','none','already_present','已逐符号对照所在math、shape、retention或bind目标；短释义/步骤name未独占其数值或状态定义，字段仍必填。',[3,8])
ann(list(range(128,139)),'EXPLANATION','none','already_present','数学/元数据/流水线摘要；对应公式、角色、环容量及依赖另有主定义。各分支摘要与实际局部Design吻合。',[3,8,15])
ann([139,147],'MIXED','some','mixed','alpha求取来源与避免整块Y重写的投票是独有执行/成本规则；previous-PV依赖与握手事件已有边和pipe。',[14])
ann([140],'MIXED','some','mixed','BF16转换及有效片已由combine math给出，coalescent store是实现访问意图；未给定线程layout，不把它当新增同步语法。',[17])
ann([141],'MIXED','some','mixed','LSE等待与槽共用已有结构；CTA内部转置发布、LastValidSplit和所有线程可消费的边界仍在动作契约。',[18])
ann([142],'MIXED','some','mixed','Stats/P及重置由online+retention给出；仅FinalSum=ell(N-1)缺正式对象定义，可用既有math/symbol补齐，再采用通用publish。',[12])
ann([143,150,157,165],'MIXED','some','mixed','RequestMap/WorkCoord组合内容与一次CTA发布、FinalSum绑定混合；状态重置和Stats内容另由math/pipe给出。',[9,12])
ann([144],'DUPLICATE','none','already_present','通用release即本role所有读取结束；stage半块→release→末次store及跨工作首PV等待已完整给出提前复用。',[16])
ann([145,148,154,160],'MIXED','some','mixed','自定义QK/PV或token动作仍需计算、首半初始化与读者完成契约；部分累积完成、依赖/配对范围由结构或realization重复。',[13,15])
ann([146,153,155,156],'MIXED','some','mixed','stage自定义动作的normalize与group完成、store的读完边界和访问合并意图需要定义；其中数值/顺序/空split分支已有上游与repeat。',[11,17])
ann([149,151],'MIXED','some','mixed','thread-private stage readiness、前3次可预读未定义partial、后续裁剪及row归属是当前正文唯一访问/执行规则；显式data/reuse边已承载其外围顺序。',[18])
ann([152],'ESSENTIAL_TEXT','some','open_semantics','load-reduced max用于完整prefix、causal/tail suffix重新归约是主计算负载分摊；capture是自定义动作，公式只给数值max不足以确定此成本。',[14])
ann([158],'DUPLICATE','none','already_present','通用produce执行并物化LSEpart，math定义目标PartialLSE；pipe在FinalSum之后且与Correction无额外等待，整项可省略。',[17])
ann([159],'MIXED','some','mixed','KVStage到K/V的映射、每成员页数仍需开放定义；issue次序、四token限制由e域、pipe、e+4复用边支持。',[15])
ann([161,162],'MIXED','some','mixed','首次Stats只确认不读、后续读入寄存器再退役是capture特有契约；root同CTA来源句与显式成员/发布边重复。',[14,16])
ann([163,164],'MIXED','some','mixed','每成员K一页/V两页规定事务粒度与分摊；pair-wide前驱已由MMA唯一成员对双Load实例依赖承载。',[10,15])
ann([166,167,168],'ESSENTIAL_TEXT','some','open_semantics','所需TMA/tcgen05/cp.async机制绑定按DOC6有意开放；依赖图不是可替换成任意低成本原语的证明，API名称本身也不证明新核心结构缺口。',[20])
ann([169],'MIXED','some','mixed','384线程/空闲warp的原语启动条件与同一512列TMEM分配混合；后者可由单resource多view表达，前者不凭活跃role并集推出。',[19,20])
ann([170,171,172],'EXPLANATION','none','already_present','理由引用已有双CTA交接、别名存储或三次预取与私有消费；不额外定义顺序，reason仍为必填。',[3,20])
ann([173],'MIXED','uncertain','mixed','分配生命周期和idle warp让寄存器解释实现绑定；未量化384相对活跃warp数量的独立一阶性能影响，不据此宣称缺核心字段。',[19,20])
ann(list(range(174,181)),'EXPLANATION','none','already_present','phase计算概述；single_tile、tile_reduction、公式及schedule给出正式定义。必填describe可短写。',[3])
ann([181],'MIXED','some','mixed','首句成员列表位置规则由DOC4提供；K/V对不同轴的分摊不由row成员切片或容量推导。',[10])
ann([183,186],'ESSENTIAL_TEXT','some','open_semantics','每CTA对K/V不同逻辑轴的分摊仅在strategy给出；SMCount身份或省略成员默认不代替这些映射。',[10])
ann([182,189,191,193,194,195],'MIXED','some','mixed','请求排序/折叠映射的正式规则与map一次构建、查找算法、发布混合；排序名称和ascending键无法替代映射。',[9])
ann([184,192],'MIXED','some','mixed','折叠m(x)是必须保留的正式映射；query最快的维度次序已有worktile_order。',[9])
ann([185,187,188],'MIXED','some','mixed','M/Bk/N/sigma及split分配为正式参数关系；空/无效工作过滤与结构重复的块说明混合。',[9,11])
ann([190],'MIXED','some','mixed','combine头序/请求索引和LSE转置定义与Mc/Dc/Smax、padding、无效块规则混合；shape不证明转置值身份。',[11,18])
ann(list(range(196,201)),'EXPLANATION','none','already_present','调度摘要由partition/mapping/preassigned/launch和strategy主定义支持，不新增独占规则。',[3,9])
assert set(A)==set(range(1,201)),sorted(set(range(1,201))-set(A))
D0='rules/schema_docs/0.common.md';D1='rules/schema_docs/1.contract.algorithm_requirement.md';D2='rules/schema_docs/2.contract.hardware_specification.md';D3='rules/schema_docs/3.design.math_reshape.md';D4='rules/schema_docs/4.design.work_schedule.md';D5='rules/schema_docs/5.design.tile_pipeline.md';D6='rules/schema_docs/6.realization.md'; W=f'work/{O}'
finding(1,'structural_gap',[1,2,3,15,30,31], 'ABI、原地输出身份和连续布局有意由Contract开放正文承载，不能从同shape/dtype或out_after名字推断。','output与out是同一对象；out_after是out更新后的内容；只out可观察变异。','删掉out_after meaning的原地身份，仍可把out_after解释为独立新结果并另行返回out；删完整ABI关系允许非连续或不同页单位。单删output meaning仍有compute身份句，所以它不是独占核心。','已排查input/output的meaning/shape/dtype/nullable/range/default及ref属性；没有storage_alias字段。现有合法写法是output.out_after.meaning: Updated contents of input out. 这是有意开放语义，不是结构化解决或扩语言必要性证明。','open_semantics',extra=[D1+':39',D1+':135',W+'/solution.py:56',W+'/reference.py:112'])
finding(2,'noncore',[6],'workspace初次清零在本既定实现不产生工作或读取。','仅scratch ABI对象身份需保留。','非零workspace与零workspace均被solution/reference删除且不参与计算，数值、提交、资源与主要性能不变。','input.workspace_buffer: {meaning: Caller scratch., shape: [byte: WorkspaceBytes], dtype: uint8}；不删除参数。',extra=[W+'/solution.py:57',W+'/reference.py:31'])
finding(3,'style',list(range(34,36))+list(range(37,54))+list(range(122,125))+list(range(128,139))+list(range(170,181))+list(range(196,201)), '必填摘要/标题/理由及中文导航大多没有独占核心，但不意味着全部应删除。','核心关系在compute/rewrite、具体schedule、role/pipe/resource/waits_for；导航另有DOC0用途。','删短标题不会改变设计，删整个必填字段则非法；故仅给仍合法短替换。','phase.attention.describe: Compute final BF16 attention.；meta_data.describe与math.name仍保留非空短句；dispatch保留链接和树。',extra=[D0+':156',D3+':69',D4+':86',D5+':284',D6+':66'])
finding(4,'redundancy',[32,33],'seq_lens正值与单值None的range是同一当前契约中重复约束。','0<Qlen<=L；python_none为单值类型。','删除range并不开放L<=0或非None：compute与dtype仍约束。','删除两个可选range字段；不得添加nullable或default。',extra=[D1+':203',D1+':223'])
finding(5,'style',list(range(54,65)),'11个when都是正式谓词；共同派生式跨文件重复，但父when不继承且当前无独立dispatch宏。','Split优先；B=1分persistent及98304阈值；B=2..4分PersistSmall及共享alpha特例；B=5..16/更大batch覆盖。','删任一节点Split定义并仅依赖parent条件，独立求值失去所需派生量；把全部当自然语言删掉会失去互斥/覆盖。','可以保留紧凑数学式；from_parent只适用于Design，不能引用父when。共享选择表达式库属于复用提案，非已证核心事件缺口。',extra=[D0+':114',D0+':156',W+'/interface.py:562',W+'/interface.py:951',W+'/interface.py:1212'])
finding(6,'structural_gap',[65,66,67,68,69,70,71,72,73],'硬件execution/feature是有意开放的组成与机制定义；当前wiki占位不补定义。','32线程warp；CTA/cluster的层级与共享/协同；SMEM owner lifetime、TMEM独立存储。','去掉SMEM寿命边界会容许owner退出后peer继续访问；去掉warp组成会失去线程预算换算。通用异步提醒例外，DOC5已覆盖。','DOC2 execution.<unit>/feature.<name>值必须为字符串，不能加children/count结构。实际保留精炼摘要；不要发明硬件嵌套对象。','open_semantics',extra=[D2+':64',D2+':117',D5+':110','rules/schema_template/examples/README.md:36'])
finding(7,'source_limitation',[74,75,76,77,78,80,81,82,84,85,90,91,92,93,94],'长数学段的分支、FP8舍入/FP32分母、归约分组与快速数学未知界限应分别保留。','mhat/ell/Y初值；c在Python binary64计算再fp32；tau/offset=4；P窄化但ell保留E；空split不写PartialO；zero-weight不参与算术。','将partial无条件乘0仍允许0*NaN污染；把sum32当固定树会抹掉双分量种子和归约顺序；删除unknown会暗示已知误差界。','现有rewrite.math、arithmetic.approx/accumulate、symbol.retention已经是适用语法；公式字符串不是语言缺口。短写只能保持所有分支和舍入点。',extra=[D3+':243',D3+':281',W+'/sm100_hd256_2cta_fmha_forward.py:3508',W+'/sm100_hd256_2cta_fmha_forward.py:3660',W+'/flash_fwd_combine.py:542'])
finding(8,'existing_structure',list(range(74,130)),'7个直接父子对的完整math_reshape相同，可用现有整区引用降低raw复制；局部不同区不能用patch。','三种完整math版本：普通、共享alpha、split/combine；retention实例在split增加s，shared改变Stats列数。','当前跨文件同文不是冗余定义；直接删除子数学区不合法。引用父整个math且在子节点重新解析M/Bk/N/sigma才保持语义。','rect_joint_load、packed_persistent、small_static_lpt: design.math_reshape: {from_parent: root}\nsingle_persistent_dual_load、small_persistent_lpt: {from_parent: packed_persistent}\nsingle_persistent_joint_load: {from_parent: single_persistent_dual_load}\nsplitkv_joint_load: {from_parent: splitkv_dual_load}\n共享alpha两个节点继续本地math。',extra=[D0+':127',D0+':140'])
finding(9,'structural_gap',[182,184,185,187,188,189,191,192,193,194,195],'非单调query折叠、LPT具体排列和request-prefix解析仍由strategy中的正式关系/开放算法表达；仅worktile_order ascending并不定义它们。','m=2*x或2*(C-1-x)+1；小batch严格比较交换顺序及不稳定tie；persistent每CTA一次map和linear/4步binary查找。','只留ascending会把m顺排；只留LPT名字允许不同tie网络和不同早晚工作；每工作重建map与一次建表会改变主要重复开销。','已用preassigned:{start:worker_id,step:worker_count}表示静态persistent，worktile_order表示外层维次；m(x)和排序网络需要具体数学/算法定义。DOC4允许strategy来源键定义，没有可替代任意排列的封闭by函数。当前合法短写可保留m(x)分段式；不能换一个未定义fold/LPT名字冒充结构化。','open_semantics',extra=[D4+':207',D4+':251',W+'/sm100_hd256_2cta_fmha_forward.py:949',W+'/sm100_hd256_2cta_fmha_forward.py:1074',W+'/sm100_hd256_2cta_fmha_forward.py:1216'])
finding(10,'structural_gap',[181,183,186,163,164],'输出row的member切分未表达K按token、V按channel-half的不同输入供应映射。','CTA a供K[64a:64(a+1)]及每V half的64列。','交换两个CTA的K/V逻辑区而保留每成员容量与output row切片，剩余结构无法定位正确输入；错把K/V均按row分摊会改变搬运与结果。','已排查mapping.member（全工作维度区间，不能按不同值限定）、resource.member_slot_shape/member/view（容量/槽内区域，不定义全局输入坐标）、invariant.offset（工作内偏移）。这些可表达大小及物理分布，但不能消除该映射定义；现有策略中的两条切片关系已很紧凑。','open_semantics',extra=[D4+':174',D5+':298',W+'/sm100_hd256_2cta_fmha_forward.py:900'])
finding(11,'existing_structure',[185,187,188,190,92,155],'N=0 split已有active/empty域结构；无效网格工作拒绝仍是schedule允许的空工作规则，不能将其当作所有pipe无条件存在。','N=0仅EmptyLSE，其他roles不产生有效payload/等待；有效短域N=1保留首次QK/PV与最终输出；无效query工作被整体过滤。','删EmptyLSE契约会失去写PartialLSE=-inf；删域guard会生成last(empty)与不存在的Stats等待。仅删“no pipeline waits”说明，现有active/empty仍保留该事实。','已存在：domain: {active:[0,"min(1,N)"], empty:[0,"1-min(1,N)"]}; Load/MMA/Softmax pipe.repeat:active；Correction steady包含active子块和{repeat:empty,steady:["store:EmptyLSE"]}。更显式的无效work guard可用局部0/1 repeat，但需连同t/n及prologue/epilogue一起改写，不能仅删strategy拒绝句。',extra=[D4+':129',D5+':182'])
finding(12,'existing_structure',[142,143,150,157,165],'FinalSum是局部名字，ell最终版本仅在publish契约中绑定；可用既有数学定义+symbol消除这部分开放对象定义。','FinalSum为每有效worktile的最终ell，持久分支按t区分。','删掉FinalSum=ell(N-1)而只留同shape SMEM资源，会允许发布其他行统计量；其他Stats/P/状态重置句分别由公式与retention覆盖。','在online.math非空分支补 FinalSum = ell(N-1).；symbol.FinalSum: {meaning: Final denominator., shape:[row:M], dtype:fp32}。保持现有publish:FinalSum(t)/FinalSum(active)、pipe位置与所有data/reuse边。不得给N=0生成FinalSum。',extra=[D3+':101',D5+':89'])
finding(13,'structural_gap',[145,148,154,160,166],'自定义QK/PV的计算/初始化绑定和原语内部读写顺序不能从done有序完全推导；主要不是缺一般barrier事件。','QK d=0初始化S，PV i=0初始化Y；PV先读取P才容许后发QK写别名S槽；done覆盖所需操作数读完。','仅保留async_ordered_completion允许后发QK先写S但延迟done到旧PV之后，仍满足done顺序而旧P被破坏。该保证当前还在realization重复：单删一处不必丢事实；删除测试针对两处同义保证同时去掉。','已排查serial/async_ordered_completion、data/reuse、resource.view和release。增加PV.done→QK.issue会强化等待、抹掉原语内允许重叠，非等价。保留简短读先于写契约并以tcgen05绑定；无需据此新增@read事件。','open_semantics',extra=[D5+':108',D5+':146',D6+':101'])
finding(14,'structural_gap',[139,147,152,161,162],'capture和rescale包含独有数据驻留/负载决定，外围依赖已结构化。','capture先把Stats/ell保留到寄存器再退槽；首Stats只确认；prefix采用load-reduced max、suffix重算；无alpha<1时避免Y整块重写而保留握手。','去掉skip规则可以每轮无条件读写128×256 FP32 Y；去掉prefix/suffix规则可总做单独max归约；去掉capture寄存器副本可在归还槽后读取覆盖值。','rescale前的PV等待已有waits_for.data；Stats来源已有Softmax/capture边。math定义alpha/ell，不决定计算由哪个role完成或是否省略TMEM重写。自定义capture/rescale可用短契约表达，无一般数值计算/物化字段能无损替代全部局部动作含义。','open_semantics',extra=[W+'/sm100_hd256_2cta_fmha_forward.py:3500',W+'/sm100_hd256_2cta_fmha_forward.py:3688',D5+':98'])
finding(15,'structural_gap',[93,159,163,164,154],'joint-load四token推进已经有e域、pipe和e+4边；KVStage的载荷种类/计算与每成员事务数量仍由开放契约建立。','token的K/V双向映射；K一页/V两页完成合并；N=1为K0,V0；独立payload数组共用一个受限token序列。','删除token含义会允许contract:KVStage只做搬运或错做PV，而外围边名字仍能匹配；删除V的“两页均完成”允许done后读尚未完成的第二页。','已存在p:[0,2*N],e:{flatten:[p,d]}（persistent加t），view槽索引e[...]%4及load:KVStage(e+4)←MMA/contract:KVStage(e)。它们足以替代发射顺序/四token的重复解释，不替代K/V→token关系和原语完成。拆成K/PV具名动作需重写边与first/last序列，仍须初始化/完成契约，不能只更名。','open_semantics',extra=[D5+':77',D5+':324'])
finding(16,'redundancy',[144,161,145],'已有作用域、累积完成模式与release位置支持三处明确冗余压缩。','跨工作Y释放早于最后半块store，下一worktile首PV等待release；Stats/FinalSum绑定同CTA；累积done由execution_mode。','仅删这些解释句仍保留全部事件、边和模式。不能把release改成等待store.done，因为会丢提前复用。','保留pipe: [stage:Oblock(t,1), release:Y(t), store:Oblock(t,1)]及首PV(t)←Correction/release:Y(t-1)。通用release无需自定义info.verb.release。root去掉capture同CTA前缀，保留后面的读/退役契约。',extra=[D5+':90',D5+':114',D5+':390',W+'/sm100_hd256_2cta_fmha_forward.py:3968'])
finding(17,'structural_gap',[140,146,153,155,156,158],'stage是有意义的数学结果→OutputStage动作；group完成与coalescent访问意图仍需开放契约，但LSE produce说明可省略。','一半Y正规化并写staging；store结束源槽读取，group可复用；split保留FP32，普通支窄BF16。','去掉stage定义只剩一个未知动词无法确定操作和目标；若store.done不覆盖该group读完，下一stage可覆盖仍被其他线程读的数据。仅换barrier API而同样保证group done不会改变设计，不作为缺口证明。','上游math已有Opart/Oblock，OutputStage.view和serial pipe明确槽及顺序。可短写stage: Stage the defined output half; done covers Correction.；已有通用produce:LSEpart无需重复“after FinalSum/correction can already stage”。','open_semantics',extra=[D5+':81',D5+':98',W+'/sm100_hd256_2cta_fmha_forward.py:3966'])
finding(18,'structural_gap',[141,149,151,190,81,167,172],'combine的已定义partial与投机读取字节不同；per-row/每线程的实际访问裁剪和局部就绪仍在契约，不能误添全CTA逐stage等待。','前3次预读可读已分配但未写empty payload；LSE=-inf使权重零且不参加算术；后续copy按thread-owned row-group的last有效split裁剪，私有读保证无stage CTA barrier。','把“权重零”简化成无条件0*PartialO会传播NaN；让一个线程消费其他线程尚在途stage会失去现有私有就绪；删除后续copy裁剪则多做empty-split流量。','现有s repeat、min(s+3,Ssplit)预取域、四槽view、data/reuse边已表达role级次序。LastValidSplit可正式定义为max({s|LSE(s,row)!=-inf}∪{-1})，但role.member只到warp，不能从warp[0:8]推出每线程私有copy分区。展开8个warp角色仍不能无代价表达每线程独立裁剪且可能加强等待；因此这是开放原语内部语义/表达成本，不是已证明须新增barrier事件。LSE零拷贝transpose可用math等式LSEview[s,a,h]=PartialLSE[s,h,a]，不可仅同shape认同值。','representation_cost',extra=[W+'/flash_fwd_combine.py:488',W+'/flash_fwd_combine.py:538',W+'/flash_fwd_combine.py:599',W+'/flash_fwd_combine.py:619',D4+':391',D5+':98'])
finding(19,'existing_structure',[169,173],'两score槽与Y同一512列分配可用一个TMEM资源多view表达；384线程含闲warp不能由活跃role并集自动推出。','score的256列与Y的256列共一资源寿命，P/Stats别名保持原字节区；idle warps寄存器归还为原实现绑定。','现有两个resource允许独立分配/寿命，删require会丢共同分配事实；但没有证据量化384与352/320线程方案的一阶成本，不能由此证明缺核心线程字段。','在普通static节点用Accumulator: {space:TMEM, member_slot_shape:[row:128,column:512], dtype:fp32, slot_count:1, view:{"S(i,d)":{part:{column:["128*(i%2)","128*(i%2)+128"]}}, "P(i)":{part:{column:["128*(i%2)","128*(i%2)+32"]},dtype:fp8_e4m3fn}, "Stats(i)":{part:{column:["128*(i%2)+32","128*(i%2)+34"]}}, Y:{part:{column:[256,512]}}}}。persistent用n，shared-alpha的Stats上端+33，其他事件/名称保持。启动/idle实现条件仍留realization，不能造空role。',extra=[D5+':297',D5+':329',W+'/sm100_hd256_2cta_fmha_forward.py:215',W+'/sm100_hd256_2cta_fmha_forward.py:1440'])
finding(20,'style',[166,167,168,169,170,171,172,173],'Realization为有意开放的原语/实现条件绑定；重复上游保证可精简，不将API差异自动记为核心表达缺口。','tcgen05提供双CTA/累积/别名顺序保证；TMA页映射与cp.async私有stage兼容。','换API包装保持同计算/完成/成本不是反例；换成同步标量搬运则可能不满足既定异步角色及流量节奏，但这不能说明缺字段。','require[0].choice: Two-CTA tcgen05 MMA. 保留reason和Design中的读先于写及execution_mode。cp.async的四stage由resource给出，仍可保留绑定。DOC6没有choice枚举，未定义原语名不能替代已说明保证。',extra=[D6+':23',D6+':101',D6+':140'])
finding(21,'source_limitation',[1,36,72,75,77,78]+list(range(54,65)),'样本是规范化模板；原实现/参考可定位，但不把文字审计当数值、误差或完整硬件规格验收。','默认环境前提；缺快速数学独立界及compiler/driver/wiki绑定；计时/ABI源路径固定。','删unknown或默认环境限定会把未经审阅的近似/override路线纳入已证设计。','保留现有source、unknown、toolchain限制和when默认条件；不新增验证成功或GPU性能字段。该Model.forward直接调私有_flash_attn_fwd；公有fastpath缓存入口不在调用链，未据此虚构额外跨调用计算。',extra=[W+'/solution.py:34',W+'/solution.py:94',W+'/interface.py:1740',W+'/interface.py:3748','rules/schema_template/examples/README.md:69'])
# Conservative exact quote edits. Required fields always receive nonempty replacements.
removal(6,U[5]['text'],'Caller scratch.','workspace在两个入口均未使用；只缩meaning不删输入。','noncore')
removal(32,'Positive.','','可选range；root compute已要求0<Qlen<=L。')
removal(33,'Python None.','','可选range；python_none已定义为单值类型且default仍缺省，保留必传。')
removal(181,'Member list entry a is CTA rank a, for a=0,1. ','','DOC4规定member列表位置即成员序号；后续K/V分摊保留。')
removal(192,'Enumerate request outside query_cluster, so query_cluster changes fastest. ','','worktile_order以request为外层、query_cluster为内层。')
removal(145,"Products may overlap; a product's done covers all earlier products issued by this role. ",'','async_ordered_completion已给出发射重叠和累积完成；保留初始化、退役和读写别名保证。')
removal(161,"Stats and FinalSum refer to this CTA's Softmax publications. ",'','已有同CTA Softmax/capture显式引用及DOC5成员默认，保留首次确认/寄存器capture内容。')
removal(144,U[143]['text'],'','可选info.verb.release整个条目省略；通用release含读完语义，pipe位置及跨work首PV边保持。')
removal(158,U[157]['text'],'','可选info.verb.produce条目省略；通用produce、LSEpart公式/物化位置及serial pipe已有全部含义。')
removal(166,U[165]['text'],'Two-CTA tcgen05 MMA.','require.choice非空；全部11节点Design的QK/contract保证和async_ordered_completion保留。')
for n in [174,176,178,179,180]: removal(n,U[n-1]['text'],'Compute final BF16 attention.','describe必填保留短句；调度策略、分配结构和公式保存全部详细设计。')
removal(177,U[176]['text'],'Produce FP32 partial outputs and LSE.','describe保留产物；rectangular static与split分配由正式schedule给出。')
removal(175,U[174]['text'],'Combine splits into BF16 output.','describe非空；tile_reduction/math/target给出同一计算。')
# Exhaustive node notes are contextual, not a claim of GPU validation.
notes={
'root':'v6；共享Contract，普通math，rectangular static双loader；i/d环、S/P/Stats别名、首/末MMA、Correction同CTA发布与全部读者退役已核对。',
'rect_joint_load':'v4；parent root；普通math绑定不变；e=flatten(p,d)，N=1 token次序、e+4约束、K/V映射和单Load realization已核对。',
'packed_persistent':'v5；parent root；Cpad原请求序+4步binary；t/n跨工作环、Q与Y退役、FinalSum复用、RequestMap一次发布已核对。',
'single_persistent_dual_load':'v5；parent packed_persistent；B=1选择及identity/linear mapping；双loader和t/n资源绑定适用于子选择域。',
'single_persistent_joint_load':'v5；parent single_persistent_dual_load；e=flatten(t,p,d)与n=flatten(t,i)互相坐标绑定，跨工作Query与Y等待、单Load已核对。',
'small_persistent_lpt':'v5；parent packed_persistent；B=2..4排除共享alpha特例；LPT网络、linear map、固定步长、t/n各资源与原math已核对。',
'small_persistent_shared_rescale':'v5；parent small_persistent_lpt；B=4/T=16384；Stats数学shape及物理view均1列，Softmax单次alpha与Correction直接读取一致。',
'small_static_lpt':'v4；parent root；Cpad/static一工作，WorkCoord两int一次发布与两loader入口依赖；无t域或跨workY释放。',
'small_static_shared_rescale':'v4；parent small_static_lpt；共享alpha的公式/Stats view/Correction读取同步改变，WorkCoord与static边保留。',
'splitkv_dual_load':'v4；parent root；split数学/两launch和global handoff；active/empty=0/1分支、FP32 partial/LSE、combine四槽/三预取/尾域/未写payload已核对。',
'splitkv_joint_load':'v4；parent splitkv_dual_load；split数学及combine相同；attention改e token单Load，N=0各guard与N=1首/末映射均已核对。'}
limits=[
'本轮完整审阅200组、全部occurrences和11个有效Design/Contract/realization；未运行GPU或声称数值/性能通过。所有节点为template，非approved提取样本。',
'完整源入口solution.py与reference.py亲读；interface.py、sm100_hd256_2cta_fmha_forward.py、flash_fwd_combine.py只按本次机制核对相关段，不声称逐行审阅整库或独立证明ISA读写顺序。source均固定01fff90d714abc6e84904ea9c44523ba66d34a0e。',
'2736个字符串scalar按全量非开放项去重核对，范围外是引用/表达式/类型/动作身份/retention/硬件占位，未发现漏掉的开放正文。',
'读取原默认环境选择及内部计算，不覆盖FA_HD256 override；B/Q/K主机元数据与设备读取长度必须满足Contract。在该私有调用链未发现录图专属初始化/回放分支；每次执行工作身份沿用DOC0，未额外断言所有外部capture组合已验证。',
'无跨文件YAML alias，无from_parent，raw与effective Design当前相同；相同文字符组不等于定义共享。整区复用仅作为经绑定核对的候选，未改变输入。',
'开放契约反例区分删除唯一事实与只删一个重复位置；硬件/ABI/原语开放语义不等于已证明必须扩语言。combine每线程执行表示成本未被有限role结构彻底解决，但未证明新增事件是必要条件。',
'保守删减仅为精确quote→replacement的字段级候选，尚无独立短稿等价复审；from_parent和TMEM重组等较大结构建议未计入候选字符节省。'
]
out={'operator':O,'prompt_version':'v1.1','unit_annotations':[A[n] for n in sorted(A)],'findings':F,'removal_candidates':C,'node_review':[{'node':n,'reviewed':True,'effective_binding_checked':True,'notes':notes[n]} for n in sorted(notes)],'limits':limits}
(R/'annotations'/f'{O}.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print('wrote',len(A),'annotations',len(F),'findings',len(C),'removal candidates')
