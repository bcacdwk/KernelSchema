import json,pathlib,collections,re
BASE=pathlib.Path(__file__).resolve().parents[2]
U=[json.loads(x) for x in (BASE/'data/extract_gemm/units.jsonl').read_text().splitlines()]
A={};F=[];C=[]
def ids(ns):return [U[n-1]['id'] for n in ns]
def ev(n):return [f'inputs/extract_gemm/{o["node"]}.yaml:{o["line"]}' if o['node']!='dispatch.md' else f'inputs/extract_gemm/dispatch.md:{o["line"]}' for o in U[n-1]['occurrences']]
def ann(ns,cat,core,st,reason,fs=()):
 for n in ns:
  assert n not in A,n
  A[n]={'id':U[n-1]['id'],'category':cat,'core':core,'structure':st,'reason':reason,'evidence':ev(n),'finding_ids':[f'extract_gemm-F{x:02}' for x in fs],'contexts_checked':True,'occurrence_overrides':[]}
def finding(i,kind,ns,claim,core,counter,candidate,gap,evidence,confidence='high'):
 F.append(dict(id=f'extract_gemm-F{i:02}',kind=kind,unit_ids=ids(ns),claim=claim,core_fact=core,deletion_counterexample=counter,existing_structural_candidate=candidate,gap_kind=gap,evidence=evidence,confidence=confidence))
def cut(n,quote,replacement,reason,kind='redundancy'):
 assert U[n-1]['text'].count(quote)==1,(n,quote)
 assert len(replacement)<=len(quote),(n,len(quote),len(replacement))
 C.append(dict(unit_id=U[n-1]['id'],quote=quote,replacement=replacement,kind=kind,reason=reason,evidence=ev(n),applies_to_occurrences='all'))
# Every listed numerical index is manually reviewed; no fallback/default category.
ann([1,2,3],'MIXED','some','mixed','公式/有效域是正式关系；对象同一性、写入边界及无重构等外部语义由开放 compute 文字定义，不能整段删。',(1,))
ann([4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20],'EXPLANATION','none','already_present','对象说明由相应 shape/dtype/range 和 root compute 的尺度、边界及写回定义落实；短 meaning 仍是必填导航。',(14,))
ann([15],'ESSENTIAL_TEXT','all','open_semantics','Python 三元组的三个位置分别代表 M/N/K 尺度；scalar shape 与 python_tuple 类型本身不表达位置含义。',(1,))
ann([21,22,23,24,25,26],'FORMAL','all','formula_or_predicate','精确值/逐元素域或 BF16 非有限值许可，属于正式 Contract 条件。',(1,))
ann([27,28,29],'EXPLANATION','none','already_present','compute 名称是必填的人读标题；数学关系在同项 math，不能把标题长度当核心叙述。',(14,))
ann([30],'SOURCE_LIMIT','some','open_semantics','保留阈值且明确比较器公式、精度和非有限策略未知；改成裸 atol/rtol 会错误采用规范默认。',(2,))
ann([31,43,45,47],'SOURCE_LIMIT','some','open_semantics','索引保留源码实现偏差/失败与实际提交边界/验收缺口，不能由路径覆盖推成正确性或完成保证。',(2,3,15))
ann([32,33,37],'NONCORE','none','not_applicable','Markdown 标题和表头，仅组织索引，不决定任何节点数值、执行或资源。',(14,))
ann([34,35,36,38,39,40,41,42,44,46],'EXPLANATION','none','already_present','dispatch 导航摘要；精确选择、F/G 参数、提交数和依赖均在所链接 YAML 中独立定义。',(3,14))
ann([48,49,50,51,52,53,54],'MIXED','some','mixed','选择谓词含正式阈值，也依赖缓存命中、永久禁用状态、录制参数及错误前实际提交等开放运行时事实；各节点需独立闭合。',(3,4))
ann([55,56],'FORMAL','all','formula_or_predicate','四个 K=32 切片次序及首片初始化/零初始化约束 mm32 归约路径，属于正式算术契约。',(1,))
ann([57,60,61,62,63,64,65,66,67,69,70,71,75,76,85,87,88],'FORMAL','all','formula_or_predicate','有单位和评估次数的资源差额公式；计数需结合 Mp、Bn/ScaleN 与本节点实际提交数，不是可直接删除的自然语言。',(1,))
ann([58,59,78,79,81,84],'EXPLANATION','none','already_present','ledger.item 是必填计量对象标题，数量和依据由同条 change/by 给出。',(14,))
ann([68,77],'FORMAL','all','formula_or_predicate','ledger.by 的步骤引用；精确文本分组恰与 step 同字，不是重复正文，也不能删断关联。',(1,))
ann([72,74,80,82,86,89],'MIXED','some','mixed','baseline 的完整 FP32 operand 物化是比较口径的独有选择；提交数/含 padding 是对本节点 math 与账本范围的说明。',(1,14))
ann([73,83],'SOURCE_LIMIT','some','mixed','重结合误差来源已由 math 给出，但无最终参考误差界是必要限制，不能据源认证推断全域保证。',(2,))
ann([90,92,93,94,95,96,100,102],'MIXED','some','mixed','正式求值与独立 evaluation 索引绑定必要；完整重算、覆盖及互不消费关系部分又由 launch/retention/compute 定义。',(1,4))
ann([91,97,98,99,101,103],'FORMAL','all','formula_or_predicate','分段操作数定义、padding、Scale/F 递推和最终 cast 组成正式数学关系；字符串形式不构成缺口。',(1,6))
ann(list(range(104,120)),'EXPLANATION','none','already_present','symbol.meaning/name 对相应公式、shape、retention 或 binding 作简短说明；没有独占的新增核心规则。',(14,))
ann([120,121],'FORMAL','all','formula_or_predicate','rewrite.step 是正式引用身份，被 ledger.by 与下游计算引用；不是解释性长文。',(1,))
ann(list(range(122,129)),'EXPLANATION','none','already_present','必填摘要的数学路径、物化收益、padding 与多次恢复成本均在 rewrite/ledger/Contract 已给出。',(14,))
ann([129],'EXPLANATION','none','already_present','八 warp、三输入槽、拷贝与计算重叠、drain 后 alias 输出均由 Main 管线、模式和 InputRing 视图给出。',(12,14))
ann(list(range(130,136)),'MIXED','some','mixed','缓冲/角色/早释放摘要大部重复结构；源 TMEM 行组寻址遗漏的限制仍必要，不能删掉整段。',(2,14,15))
ann([136],'MIXED','some','mixed','InputRing slot0 的 BF16 alias 与 drain 顺序已结构化；CTA 发布 rendezvous 与实现 padding 的容量兼容是开放动作/实现条件。',(8,12))
ann([137],'MIXED','some','mixed','坐标和 F 初值重复上游；Full 条件及 epilogue 行归属属于本动作的正式参数/开放协作分工。',(7,8))
ann([138,150],'ESSENTIAL_TEXT','all','open_semantics','所有 TMEM 读者退役与 bulk 源读取完成是两个不同 release 保证；首批无旧 store、尾块保留 store 历史不能由 store.done 替代。',(10,))
ann([139,174],'STRUCTURABLE','all','existing_structure','await 仅等显式 fragment done；现有通用 wait 加原 @done 前驱可等价承担，保留等待完成位置。',(11,))
ann([140],'MIXED','some','mixed','drain 前驱已列最终两个空提交，ordered group 的全部完成及 CTA join 仍是原语保证；释放 alias 的结论由 pipe/resource 可推出。',(9,12))
ann([141,151,156,164],'MIXED','some','mixed','J/R 的 F/G 绑定与协作初始化必要；两 Product 槽、新 launch 独立资源和 generations 的说明可由结构/公共规则推出。',(4,7,8))
ann([142,143],'MIXED','some','mixed','实际搬运与 masked/空提交语义必要；两次预热、每轮替换、最终空提交的存在及数量已在 repeat 中。',(6,9))
ann([144],'ESSENTIAL_TEXT','some','open_semantics','warp 的 2x4 数学分块和协作 copy/store 是 role 内分工，mapping.member 只切 worker CTA，不能拿 CTA 成员结构冒充 warp 映射。',(8,))
ann([145],'MIXED','some','mixed','NI 谓词、单/双 warp 产品分工以及组合 done 的所有完成保证必要；Dot 数值和复用边在数学与 waits_for 已表达。',(7,8))
ann([146],'MIXED','some','mixed','scale 复制、越界置零和 p>=KB 的即时无读取完成必要；两槽深度、预热与 kb+2替换及捕获后覆盖部分可由资源/pipe推出。',(6,9))
ann([147,159],'ESSENTIAL_TEXT','some','open_semantics','role 标为 Producer/Main 不能自行推出 CTA 级边界复制、leader prefix 与两次 entry rendezvous；这是开放协作动作定义。',(8,))
ann([148,155,160,173,176],'MIXED','some','mixed','计算公式已在上游；动作隐含读取的 Scale/AS/b_scale、读完保证、可见性或跨 warp join 仍要定义，不能只留 compute 名称。',(8,9,11))
ann([149,170],'MIXED','some','mixed','bulk group 的盒数、写入坐标与 destination done 是实现原语/分批绑定；允许后续工作推进已由 async 模式表达。',(7,10,13))
ann([152],'ESSENTIAL_TEXT','some','open_semantics','普通 vector store 的 done 结束所有源 staging 读取而目的写可见性由 kernel_done 覆盖；trailing CTA rendezvous 保护下一 worktile alias。',(8,12))
ann([153,167],'ESSENTIAL_TEXT','all','open_semantics','Fragment 在 Product/DotPair 中的精确列区域及覆盖所有行/列组、done 含寄存器交付没有在动作名或资源 shape 中编码。',(8,11))
ann([154,161],'ESSENTIAL_TEXT','some','open_semantics','bulk 队列最终 destination 完成以及无 worktile 时也参与全 CTA teardown；普通 release 不提供目的写完成。',(8,10))
ann([157,177],'MIXED','some','mixed','A/B别名、按H实际字节完成与 Bstream 缓存/录图绑定必要；padding 定义和跨工作 ring position 已在数学/flatten/slot中。',(3,6,13))
ann([158,168],'ESSENTIAL_TEXT','some','open_semantics','输出分批转换/发布对异步访问域的可见性和所有 writers rendezvous 不由 publish 名称或 serial 模式自动提供。',(7,8,10))
ann([162,175],'MIXED','some','mixed','有效行边界已由 math和target给出；尾写不碰 staging/store history 以及 F 读完是局部完成契约。',(6,10))
ann([163],'MIXED','some','mixed','F 初值与全部 Bm 行计算已由 math给出；两组 warp 各拥有128列及组内分行、Full谓词由文字确定。',(7,8))
ann([165,166],'ESSENTIAL_TEXT','some','open_semantics','weight零预取、捕获的 scale 存活、wait_group1 窗口及读结束点由动作定义，资源形状与显式数据边不能独立确定这些行为。',(9,))
ann([169],'MIXED','some','mixed','四段初始化重复 math；本 MMA done 兼任产品可读与全部operand读完是开放完成保证。',(8,11))
ann([171,172],'ESSENTIAL_TEXT','some','open_semantics','acquire 的 TMEM 可见性或 per-thread group-window+CTA join 是抽象 done 的具体保证；已有 data边未定义这些原语语义。',(8,9,11))
ann([178,179,180,181,183,184,185,186,187,189],'ESSENTIAL_TEXT','uncertain','open_semantics','记录已读源码原语/布局兼容绑定；必要依赖图在 Design 已声明，尚无更换等价 API 必然改变核心的充分证据，核心重要性不计为已证。',(13,))
ann([182],'ESSENTIAL_TEXT','some','open_semantics','Bstream 对 L2 evict_first 的选择是有具体 reuse 原因的优先策略，通用 load 与依赖不表达 cache hint。',(13,))
ann([188],'MIXED','some','mixed','输出 alias 兼容与 17408<=24832 容量检查有意义；272-byte pitch和stmatrix/vector实现绑定并非已证唯一核心 layout。',(12,13))
ann([190,191,192,193,194,196,197,198,199,200,202,203],'EXPLANATION','none','already_present','reason 解释同项 choice 为什么符合已给 Design；源读取/可见性/全部读者结束等约束在对应 verb/pipe/resource 已定义。',(13,14))
ann([195,201],'MIXED','some','mixed','策略由 copied one-tile-per-group 几何产生的 cache优先理由；缓存可过期解释重复 Bstream 契约，但性能原因未由结构独立承载。',(13,))
ann([204,206,207],'EXPLANATION','none','already_present','phase describe 是计算对象导航；其 worktile/target/数学值和 evaluation 对应已另行定义。',(14,))
ann([205],'MIXED','some','mixed','r=0/1 到 graph/direct 的绑定只在本 describe 明确；完整重算、新 workers 和无 handoff 已由 math/flow 支持。',(4,))
ann([208,210,218,220],'MIXED','some','mixed','单调 group scan、actual Params/maps 对应和失败来源边界必要；启动完成、新 worker、无输出handoff、提交数等已结构化而重复。',(3,4,5))
ann([209,213,215],'MIXED','some','mixed','prefix/scan 的实际获取策略必要；空worker、每次调用重算与 graph 参数/当前内容等有部分公共语义和本节点 when重述。',(3,4,5))
ann([211,212,214,217,221,223],'MIXED','some','mixed','Bm/Bn/H和不规则 w↔gxy 映射/phase q 绑定仍在文字正式定义；局部K遍历、完整结果与无跨tile归约大部由数学和iteration/target承载。',(4,5,6))
ann([216],'MIXED','some','mixed','Bm/Bn、prefix 与 ragged工作编号逆映射是必要正式绑定；局部KB遍历与无splitK已由math/iteration/target给出。',(5,))
ann([219,222,224],'MIXED','some','mixed','worker_count 可直接写 C 的已有数值表达式；S来源、Wub用于NI、CTA总线程及说明混合，不能整字段判可删除。',(5,16))
ann([225,226,227,228,229,230],'EXPLANATION','none','already_present','thesis的提交数/持久分配/完成链与成本原因已由flow、mapping、assignment、策略和ledger表达；保留简短必填摘要即可。',(4,14))
assert set(A)==set(range(1,231)),set(range(1,231))-set(A)
finding(1,'style',[1,2,3,15,21,22,23,24,25,26,55,56,57,60,61,62,63,64,65,66,67,68,69,70,71,72,74,75,76,77,80,82,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,120,121],
 '数学公式、归约次序、baseline和实例范围不能按字符串长度判冗余；Contract对象同一性与Python tuple位置是有意开放语义。',
 '先FP8块乘再逐块Scale乘积/FMA；F(-1)=0；全物理行覆盖、输出对象/存储同一性、守卫不写；恢复r/q相互独立。',
 '删除对象同一性可返回新BF16对象；删除有序Scale/FMA路径可改为先缩放操作数，舍入点不同。形式表达本身保留则不发生这种损失。',
 '现有 rewrite.math / arithmetic.accumulate / symbol.retention 已承载正式关系，例如 F: {retention: "carry_forward (q,g,x,y); kb=-1..KB-1"}。没有需扩展的公式字符串问题。',
 'open_semantics',['inputs/extract_gemm/root.yaml:92','inputs/extract_gemm/root.yaml:136','inputs/extract_gemm/wide_recovery_full.yaml:60','rules/schema_docs/1.contract.algorithm_requirement.md:160','rules/schema_docs/3.design.math_reshape.md:67'])
finding(2,'source_limitation',[30,31,43,73,83,130,131,132,133,134,135],
 '必须保留比较器缺项与重结合误差界未知；TMEM源偏差限制可缩成一句但不能消失。',
 'atol=.01,rtol=.05,零失配比例有记录，比较公式/基准/精度/NaN规则未给出；认证不是该YAML全域正确性证明。',
 '裸 atol/rtol 会按 DOC1 自动采用 abs(x-g)<=a+r*abs(g)，原记录没有授权该解释；删除TMEM限制会把完整行设计当作源码已正确实现。',
 '继续使用现有 tolerance 来源记录和 thesis 的一句 source limit，无需新字段；完整诊断留本报告。',
 'none',['inputs/extract_gemm/root.yaml:106','work/extract_gemm/problem_metadata.json:1','work/extract_gemm/performance_CERTIFIED_BEST.json:1','rules/schema_docs/1.contract.algorithm_requirement.md:236','rules/schema_extract/REVIEW.md:57'])
finding(3,'structural_gap',[34,35,36,38,39,40,41,42,44,45,46,47,48,49,50,51,52,53,54,157,177,208,209,210,213,215,218,220],
 '缓存选择、永久fallback flag、实际提交前缀与录制参数需文字/谓词定义；缺的是紧凑可共享的host状态定义，不是已证明缺计算事件。',
 '策略缓存按地址和E/M/N/K命中可与当前设备group内容不同；图保存by-value Params与maps；错误API可能带已提交前缀，录图本身不计算。',
 '删缓存规则会用新m_indptr内容重选P；删actual-L约束会将失败图调用一律计0或计1次。两者改变tile路径或重复全量计算次数。',
 'dispatch.when 是现有开放谓词位置，可短写完整局部P状态规则和 L∈{[G,G,F],[G,F,F]}；不能仅写一个无定义的P或把parent.when隐式继承。尚无固定host-state/include字段；保留必要局部定义，不建议新增字段作为本轮结论。',
 'representation_cost',['inputs/extract_gemm/root.yaml:7','inputs/extract_gemm/wide_recovery_chain.yaml:10','work/extract_gemm/fp8gg_umma.cu:1539','work/extract_gemm/fp8gg_umma.cu:1707','work/extract_gemm/candidate_solution.py:275','rules/schema_docs/0.common.md:151'])
finding(4,'redundancy',[90,92,93,94,95,96,100,102,130,131,132,134,151,156,164,205,208,210,211,212,213,214,218,225,227,228,229],
 '恢复路径的提交数、逐次kernel_done与独立worker已有正式结构；长文重复不证明跨调用/提交语言缺口。',
 'v0两次，wide恢复2/3/4次；每次从原输入重算覆盖，无输出handoff；q/r绑定仍要保留。',
 '删 start_after 会容许重叠覆盖；仅删文字“new workers”不会，因为DOC4跨kernel新worker规则与实际launch已唯一确定。',
 'flow: {launch: [{phase: [first]}, {phase: [retry], start_after: first.kernel_done}, {phase: [final], start_after: retry.kernel_done}]}。phase.first:&body 与 retry:*body 可复用定义；math中保留q到phase的局部绑定。',
 'none',['inputs/extract_gemm/v0_recovery.yaml:130','inputs/extract_gemm/wide_recovery_chain.yaml:155','inputs/extract_gemm/wide_recovery_full.yaml:156','rules/schema_docs/4.design.work_schedule.md:266','rules/schema_docs/4.design.work_schedule.md:312'])
finding(5,'existing_structure',[208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224],
 '已有partition/worker_count/assignment承载工作数和静态步长；作者把派生公式及重复遍历说明放在strategy。ragged逆映射和单调scan仍属开放策略。',
 'W=prefix(E)*Tn；w到非空g、列y、行x的映射；worker_id+j*C归属；UMMA保留group hint，sync每次从0搜索。',
 '删除w逆映射会允许row-major次序；删除hint允许每个tile重新扫描，source显示这是实际调度成本。删除“C不是驻留”不改变worker_count规范语义。',
 'partition: {work: {size: "prefix(E)*Tn", tile: 1}}\nmapping: {worker: {unit: CTA, unit_count: 1}, worker_count: "max(1,min((ceil_div(M,Bm)+E)*Tn,S))"}\nschedule: {worktile_order: [{by: work, direction: ascending}], worker_assignment: {preassigned: {start: worker_id, step: worker_count}, refill: none, exit: assigned_done}}\n保留定义prefix、S和w↔gxy的strategy；sync上限用2*S，root的Wub仍供NI谓词引用。',
 'none',['inputs/extract_gemm/root.yaml:206','inputs/extract_gemm/mma_sync.yaml:98','work/extract_gemm/fp8gg_umma.cu:366','work/extract_gemm/fp8gg.cu:131','rules/schema_docs/4.design.work_schedule.md:163'])
finding(6,'redundancy',[91,97,98,99,101,103,142,143,146,157,162,175,177,211,212,214,217,221,223],
 'H载入范围、保留未初始化行与有效V行写回已在piecewise math；pipeline仍需实际字节完成/空读取语义，不能整段删。',
 'UMMA只读H行但计算Bm行，可读取下一物理group的padding；sync对u>=V置零；最终只写u<V。',
 '只保留“tail masked”可能把UMMA解释为所有padding清零，增加写入和完成字节；尾部直接store若写Bm行会破坏邻组。',
 '保留现有 math A_tile 的 u<H/全局M/retained 三分支、target.single_tile.out.shape:[row:V,column:Bn]；load短契约只需补实际箱字节完成、其联合别名和Bstream来源。',
 'none',['inputs/extract_gemm/root.yaml:136','inputs/extract_gemm/mma_sync.yaml:33','inputs/extract_gemm/root.yaml:344','work/extract_gemm/fp8gg_umma.cu:379','work/extract_gemm/fp8gg.cu:173'])
finding(7,'existing_structure',[137,141,145,149,151,156,163,164,168,170],
 'NI以及F/G的J、R、Full参数已有成员表达式/重复域/容量字段；参数值仍须在相应动作绑定一次。',
 'NI=1/2固定于worker生存期；F:J=1,R=4,两批32KiB；G:J=2,R=3,一批64KiB；tail没有bulk动作。',
 '删除J定义会令store批数、staging大小和ring深度都自由；把Full=0当bulk动作no-op会保留源码不存在的事件。',
 'info.member: "warp[1:1+NI]"\nresource.OperandRing.slot_count: R\nresource.OutputStaging.member_slot_shape: [row: Bm, column: "J*ScaleN"]\npipe.steady: [{repeat: {f: [0, "Full(t)*2/J"]}, steady: ["release:Staging(t,f)", "publish:Output(t,f)", "store:Output(t,f)"]}]\n这些字段当前已经存在；只保留J/R/NI/Full的短定义，不能删其定义。',
 'none',['inputs/extract_gemm/root.yaml:366','inputs/extract_gemm/wide.yaml:264','inputs/extract_gemm/wide.yaml:349','work/extract_gemm/fp8gg_umma.cu:1317','rules/schema_docs/5.design.tile_pipeline.md:204'])
finding(8,'structural_gap',[136,137,144,145,147,148,151,152,153,154,155,156,158,159,160,161,163,164,167,168,169,171,172,173,176],
 'role内部warp分工、CTA rendezvous与组合done依赖有意开放协作动作；若全转成细粒度结构成本很高。当前不能用CTA member索引替代warp事件。',
 'NI所有发射成员完成；epilogue所有行及两列组读完；entry/exit为CTA协作而主体角色子集；Fragment的数学区域和Scale隐含读取。',
 '删除join契约时只等一个warp读取完成也符合剩余整体动作名称，可能提前覆盖TMEM；删除warp区域分工可能复制计算而遗漏某行。差异是读者集合与工作分配，非仅barrier API名称。',
 'DOC5允许 info.verb 把集体完成定义在一个动作，现有 waits_for 对这个done建边即可；mapping.member切的是worker成员CTA，不是role的warp。可拆成明确多个role再列全部done，但NI可变和shared读写使其不紧凑。未证明必须新增warp事件字段；保留开放集体契约。',
 'representation_cost',['inputs/extract_gemm/root.yaml:368','inputs/extract_gemm/wide.yaml:295','inputs/extract_gemm/mma_sync.yaml:186','work/extract_gemm/fp8gg_umma.cu:336','work/extract_gemm/fp8gg_umma.cu:1303','rules/schema_docs/5.design.tile_pipeline.md:90','rules/schema_docs/5.design.tile_pipeline.md:359'])
finding(9,'structural_gap',[140,142,143,146,148,155,160,165,166,172,185,183],
 'copy group的空提交、wait_group1窗口与捕获值存活是开放原语语义；repeat与data边已记录事件，缺口候选是群组语义紧凑表示成本。',
 '两次预热、每轮替换、末两次空commit仍有事件；V0跨work不显式reset，sync在alias输出前drain；捕获scale在prefetch寄存器覆盖后继续有效。',
 '省掉空commit会让wait_group1在最后真实组仍未完成时通过；把source数据边等同于group窗口会丢现有原语依赖。不能因此断言换等价API破坏核心。',
 '现有 pipe: {repeat: p[0:2], steady:["prefetch_a:AS(t,p)"]} 与每轮 prefetch_a:AS(t,q+2) 保留全部事件；capture.waits_for.data 引用AS(t,q)。组提交/完成的物理含义仍由verb和realization绑定，不能增加一个未定义commit动作就声称解决。',
 'open_semantics',['inputs/extract_gemm/root.yaml:400','inputs/extract_gemm/mma_sync.yaml:189','work/extract_gemm/fp8gg_umma.cu:822','work/extract_gemm/fp8gg_umma.cu:1000','work/extract_gemm/fp8gg.cu:198','rules/schema_docs/5.design.tile_pipeline.md:188'])
finding(10,'structural_gap',[138,149,150,154,158,161,162,168,170,175,178,181],
 'release已经表达源缓冲归还先于目的写完成；跨tail的全部历史store与所有writer join仍需短原语契约，不是缺@read事件的证明。',
 'TMEM最后读完成后、最终FMA前归还；Staging重用等所有先前bulk源读，dest仍在途；首full/首batch无旧store，tail不清history，worker末drain destination。',
 '把release换成store.done会加强等待丢重叠；只等上一worktile store则中间tail没有该动作且可能忘记更早full store。只等issue会过早覆盖。',
 '现有序列 release:Staging(t,f)→publish:Output(t,f)→store:Output(t,f)，store为async，release为serial，exit drain保留；release契约明确“all previously issued bulk source reads”。若需跨tail统一编号，可令 f:[0,Full(t)*2/J], s:{flatten:[t,f]}，但release原语含义仍不可由编号本身推出；不提议未定义新事件。',
 'open_semantics',['inputs/extract_gemm/root.yaml:439','inputs/extract_gemm/wide.yaml:317','work/extract_gemm/fp8gg_umma.cu:1015','work/extract_gemm/fp8gg_umma.cu:1321','work/extract_gemm/fp8gg_umma.cu:1391','rules/schema_docs/5.design.tile_pipeline.md:145','rules/schema_docs/5.design.tile_pipeline.md:375'])
finding(11,'existing_structure',[139,148,153,155,167,169,171,173,174,176],
 '纯await定义可使用现有通用wait与@done边；Fragment读取/Scale隐含operand与MMA组合done则不能只靠换动作名消除。',
 'await只保证已命名read的寄存器交付；release放在最后await之后、最终compute之前。',
 '若把等待约束从wait.done移到wait.issue，会禁止等待提前开始；若删read→wait.done则可能读未交付寄存器。',
 '在原pipe位置把 await:Fragment(n,d) 改为 wait:Fragment(n,d)，保持role serial及 waits_for: {data: {"wait:Fragment(n,d)@done": ["read:Fragment(n,d)"]}}。删去info.verb.await（可选映射项），其余read/compute/release契约不改。',
 'none',['inputs/extract_gemm/root.yaml:433','inputs/extract_gemm/root.yaml:477','inputs/extract_gemm/root.yaml:504','inputs/extract_gemm/wide.yaml:314','rules/schema_docs/5.design.tile_pipeline.md:80'])
finding(12,'redundancy',[129,136,140,152,188],
 'sync的输入/输出别名已由一个InputRing资源和BF16 view表达，drain/发布/读完由同一serial role的有序pipe及契约保护。',
 '三槽输入；Output在slot0，17408B实际staging≤24832B单槽，Schedule在ring外始终保留。',
 '新增独立Output资源会改变容量而仍符合纯shape描述；删除已有view才会失去alias。只删文字“firstInputRing slot”不会，slot_index:0已唯一确定。',
 'InputRing.view: {"Output(t)": {slot_index: 0, dtype: bf16}}；现有pipe drain:Copies(t)→publish:Output(t)→store:Output(t)；保留drain/store的跨warp读完保证。',
 'none',['inputs/extract_gemm/mma_sync.yaml:158','inputs/extract_gemm/mma_sync.yaml:201','inputs/extract_gemm/mma_sync.yaml:220','work/extract_gemm/fp8gg.cu:29','work/extract_gemm/fp8gg.cu:266','rules/schema_docs/5.design.tile_pipeline.md:302'])
finding(13,'style',list(range(178,204))+[149,157,170,177],
 '原语绑定与布局兼容是Realization的有意开放内容；等价API差异不能单独证明核心字段缺失。Bstream cache hint有明确reuse原因，其余绑定核心唯一性多为uncertain。',
 'TMA字节完成、swizzle兼容、TMEM路径、cp.async组、bulk read/dest完成等保证已有Design约束；具体指令绑定来自已读源码。',
 '不保持swizzle/fragment兼容可能错读值；换保持同样数据路径/依赖/容量的封装不构成反例。B hint移到所有load会在当前高reuse场景影响主要流量，但没有本轮性能测量。',
 '现有 realization.require/prefer {item,choice,reason} 足够，reason可短引已有保证；不把整段choice搬到strategy。保留如 choice:"Use the L2 evict_first hint for B loads exactly when Bstream is set."，不提升为已证最优。',
 'open_semantics',['inputs/extract_gemm/root.yaml:505','inputs/extract_gemm/wide.yaml:363','inputs/extract_gemm/mma_sync.yaml:228','work/extract_gemm/fp8gg_umma.cu:205','work/extract_gemm/fp8gg_umma.cu:1637','rules/schema_docs/6.realization.md:27'])
finding(14,'redundancy',[4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,27,28,29,32,33,34,35,36,37,38,39,40,41,42,44,46,58,59,72,74,78,79,80,81,82,84,86,89]+list(range(104,120))+list(range(122,136))+[190,191,192,193,194,196,197,198,199,200,202,203,204,206,207,225,226,227,228,229,230],
 '必填摘要/meaning/name、reason与dispatch索引有阅读用途，通常不独占核心；应短写而非删除必填字段。',
 '需保持简洁可导航的任务、步骤、phase及原语理由；独有source limit和baseline物化选择另保留。',
 '删必填thesis/meaning/name会令文档不合法；下面只替换成非空短句，核心由同节点公式、flow、角色/资源、verb或选择条件保存。',
 'thesis: "Apply block scales after FP8 dots, avoiding scaled operands at the cost of ordered FMA and padded work."；保留已有rewrite/ledger。dispatch表仅保留链接、workload和有效差异，精确选择回YAML。',
 'none',['inputs/extract_gemm/root.yaml:128','inputs/extract_gemm/root.yaml:195','inputs/extract_gemm/root.yaml:537','inputs/extract_gemm/dispatch.md:3','rules/README.md:89','rules/schema_docs/0.common.md:155'])
finding(15,'source_limitation',[31,45,47,130,131,132,133,134,135],
 '当前YAML完整行所有权与源码TMEM读取地址存在已记录实现偏差；失败恢复链是条件化实际提交账，不宣称这些错误链一定可达或成功。',
 '源码row=(warp&3)*32+lane，TMEM ta只有tbase+slot列+coloff；完整product容量/row-specific scales/stores支持计划的行分工。',
 '删除限制会把抽象设计当源读地址已满足；另一方面不得为修源擅加role/slot/依赖或推断未观测API错误一定提交。',
 '不改Design；保留一句"Source TMEM drain omits the row-group address contribution; complete row ownership follows product generation, capacity, scales and stores."。实际提交用现有L条件和start_after；不新设失败空kernel。',
 'none',['inputs/extract_gemm/root.yaml:256','inputs/extract_gemm/wide.yaml:192','work/extract_gemm/fp8gg_umma.cu:787','work/extract_gemm/fp8gg_umma.cu:864','work/extract_gemm/fp8gg_umma.cu:1218','work/extract_gemm/fp8gg_umma.cu:1262','rules/schema_extract/REVIEW.md:59'],'medium')
finding(16,'source_limitation',[145,179,180,186,189,219,222,224],
 '共享hardware全null，来自冻结DOC2明示提取豁免；不能把零硬件正文当作硬件设计已简洁完备。',
 'CTA/warp/SMEM/TMEM与具体原语在Design/Realization使用，契约没有登记执行层级、设备容量或工具链资料。',
 '同YAML无法独立约束设备支持tcgen05或ring是否可驻留；approved仅表示本次提取阶段批准。',
 '已有 contract.hardware_specification.device/arch/execution/feature/stack/wiki 可补有据事实，不需新增字段。此次只审计，不补猜测硬件值；仅列出这些已有字段，不把未取得hardware证据伪造成填好方案。',
 'none',['inputs/extract_gemm/root.yaml:114','rules/schema_docs/2.contract.hardware_specification.md:13','rules/schema_extract/REVIEW.md:39'])
# Conservative, exact quote-to-replacement candidates. Required fields stay nonempty.
for n in [48,49,50,51,52,53,54]:
 text=U[n-1]['text']
 # Each locally defines graph recording already; select only redundant final recording clause.
 quotes={48:'A pure recording describes this future execution, with no extra invocation at recording.',49:'A pure recording describes this future execution, with no additional invocation at recording.',50:'Pure graph recording describes a future execution; it does not execute the recorded work.',51:'A pure recording describes the future execution and adds no invocation.',52:'A pure recording describes this future execution, with no extra invocation at recording.',53:'A pure recording describes that future execution and is not an additional invocation.',54:'A pure recording describes the selected future execution and is not an additional invocation.'}
 # sync has no earlier explicit non-computation sentence: keep its wording unchanged.
 if n!=50:cut(n,quotes[n],'','同字段先前已定义actual compute submissions及graph construction/instantiation不增L；DOC0计算执行实例已规定纯录图不执行。')
cut(1,'Multiplication and accumulation use FP32.','','Contract默认FP32，公式bf16显式窄化；保留非有限转换、不clamp、alias及守卫边界。')
cut(2,'All listed inputs are supplied; out_buffer and out_dtype are non-None.','','所有input省略default且无nullable，已要求显式非None；不删contiguous/CUDA或物理行规则。')
cut(2,'The scale widths and selectors have exactly the values specified in input.','','input.range已逐项固定；不动tuple位置含义。')
cut(3,'Products use FP32; stored payloads and scales are the numerical inputs, with no re-quantization or reconstruction of pre-quantization data.','Stored payloads and scales are used without re-quantization.','FP32由Contract默认；保留无再量化语义，公式直接使用存储值已排除重建。')
for n in [219,222]:cut(n,'C is a submission count, not a claim that all workers are resident simultaneously.','','DOC4明确worker_count是提交量且不隐含驻留。')
cut(224,'C is not a simultaneous residency claim.','','DOC4 worker_count唯一语义已足够。')
for n in [211,212,214,217,221,223]:
 cut(n,'All scales and all K contributions are applied inside this worktile; no cross-worktile reduction or intermediate global matrix is used.','','worktile.iteration保留完整K域，target为最终out；math与retention未生成跨tile中间量，无tile_reduction。')
cut(216,'No split-K work, cross-tile reduction or global intermediate is used.','','partition只有完整输出work，KB是iteration，math/target已排除split-K。')
for n in [151,156,164]:
 q={151:'Each launch starts its own ring positions, generations, scale state and store history; no local resource or event instance is shared between first and retry.',156:'Each launch starts its own ring positions, generations, scale state and store history; no local resource or event instance is shared among first, retry and final.',164:'Each launch starts its own ring positions, generations, scale state and store history; no local resource or event instance is shared among first, retry, fallback and final.'}[n]
 cut(n,q,'','flow的不同kernel建立不同worker；每phase有initialize、局部资源和drain，DOC4/5不跨kernel继承资源；保留q/J绑定。')
for n in [141,151,156,164]:cut(n,'ProductRing holds two independent Dot blocks.','','ProductRing shape:[row:Bm,column:Bn],slot_count:2,view:n%2和math已定义。')
cut(157,'Ring position and completion generations continue across worktiles.','','domain n=flatten[t,q]及n%3/n%2、n+3/n+2边已跨work接续，DOC5默认资源跨work。')
cut(177,'Ring positions and generations continue across worktiles.','','n=flatten[t,kb]及n%R和n+R边已表达连续ring，不重置。')
for n in [149,170]:
 quote={149:'The next worktile may start while these writes remain in flight.',170:'The next batch or worktile may advance while those writes remain in flight.'}[n]
 cut(n,quote,'','store execution_mode:async，release仅等source读取，最终drain才等destination；不新增任何等待。')
cut(136,'Convert the entire final F to BF16 in Output(t), using the first InputRing slot after drain.','Convert final F to BF16 in Output(t).','InputRing.Output(t).slot_index=0/dtype=bf16和pipe drain→publish已定义；其余capacity/rendezvous/存活信息保留。')
# Short legal summaries, retaining all independent limits elsewhere in the same node.
for n in [122,123,124,125,126,127]:
 count={122:'three times',123:'twice',124:'once',125:'once',126:'four times',127:'twice'}[n]
 cut(n,U[n-1]['text'],f'Apply block scales after FP8 dots {count}, avoiding scaled operands at the cost of ordered FMA and padded work.','必填thesis保留路径/收益/代价；实际evaluation数、writeback和padding详由rewrite/ledger支持。')
for n in [225,226,227,228,229,230]:
 rep={225:'Run two sequential persistent grids, each recomputing the output; bounded grid sizing can assign no work to some CTAs.',226:'Use persistent CTAs with local K reduction; group-major work offers weight reuse, while bounded grid sizing permits idle CTAs.',227:'Run four sequential persistent wide grids; each recomputes and overwrites the output.',228:'Run two sequential persistent wide grids; each recomputes and overwrites the output.',229:'Run three sequential persistent wide grids; each recomputes and overwrites the output.',230:'Use persistent CTAs with local K reduction and group-major weight reuse; the grid is capped at twice the supplied SM count.'}[n]
 cut(n,U[n-1]['text'],rep,'非空thesis保留组织要点；精确flow/start_after、数学独立性和工作范围保留原定义。')
cut(129,U[128]['text'],'Overlap copies with block computation in one eight-warp role; drain before staging output in the input ring.','role/三槽资源/pipe及alias完整保留；必填摘要只缩写。')
for n in [130,131,132,133,134,135]:
 prefix='Each kernel overlaps operand loads, products and epilogue work; product slots return after reads and staging waits only for source reads.'
 limit=' Source TMEM drain omits the row-group address contribution; complete row ownership follows product generation, capacity, scales and stores.'
 cut(n,U[n-1]['text'],prefix+limit,'缓冲/分批/所有读者和完成顺序在resource/verb/pipe中保留，源偏差限制仍明确；不声明源码正确。')
cut(30,U[29]['text'],'Source record: atol=0.01, rtol=0.05, max_mismatch_fraction=0.0 for the full output view; nonwrite inputs and storage outside it are unchanged. The bound metadata and certification omit the comparison formula, relative-error basis, comparator precision and NaN/Inf policy; default atol/rtol semantics are not adopted.','保留所有已知判据/比较范围/非写存储要求、每项未知和默认解释排除；仅去来源叙述铺垫。','source_limitation')
for n,rep in {190:'Bind source-read completion separately from destination completion for the shared staging buffer.',191:'The shared J-group buffer becomes reusable before destination writes complete.',192:'Bind copy-group capture and completed fragment reads required for early TMEM release.',193:'Cover actual H-row A/full-B bytes while preserving untouched rows and the R-slot ring.',194:'Bind TMEM completion for product visibility and operand reuse.',196:'Stream order implements graph.kernel_done without a blocking host wait.',197:'Rendezvous publishes output across warps and ends staging reads before ring reuse.',198:'Bind shared TMEM product/operand completion, including the NI-member join.',199:'Bind the ordered warp-cooperative FP8 register product path.',200:'Cover actual H-row A/full-B bytes while preserving untouched rows and the three-slot ring.',202:'The group window and CTA join protect stage reuse; drain protects BF16 alias staging.',203:'Early ProductRing release requires completed TMEM reads.'}.items():
 cut(n,U[n-1]['text'],rep,'同一choice和上游verb已经给出完整保证；保留非空reason及具体依赖，未把实现API差异算新增核心。')
node_notes={
'root':'完整共享Contract；128x128、paired K、NI=1/2、3-slot operand/2-slot pairTMEM；逐项查空worker、H/V、末空scale commits、所有readers、tail保持store history。',
'mma_sync':'完整本地Design；64x128，8 warp，3-slot输入，u>=V零填充；末空commit仍有事件，drain后slot0 BF16 alias，store末CTA join保护下一tile。',
'v0_recovery':'graph/direct两个实际launch，direct.start_after=graph.kernel_done；完整同一V0 body通过alias实例化两次，r绑定0/1且局部状态独立，require列表覆盖两phase。',
'wide':'128x256；F/G以J=1/2,R=5-J和2/J批次参数化；两列epilogue组、2 Product槽、scale预取；single actual submission，空full域无bulk事件。',
'wide_recovery':'first/retry alias各绑定q=0/1与L[q]，FF/GG/GF；每launch独立J/R/maps/Params且完成后下一次，无输出handoff。',
'wide_recovery_chain':'first/retry/final的三份alias分别q=0/1/2；GGF/GFF，全部Realization item列覆盖三phase，start_after逐一闭合。',
'wide_recovery_full':'first/retry/fallback/final四份alias分别q=0/1/2/3，GGFF；四次完整overwrite，逐次kernel_done，新worker及所有资源/事件独立；Realization逐phase全覆盖。'}
for f in F:
 f['nodes']=sorted({o['node'] for uid in f['unit_ids'] for u in U if u['id']==uid for o in u['occurrences']})
report={'operator':'extract_gemm','prompt_version':'v1.1','unit_annotations':[A[n] for n in sorted(A)],'findings':F,'removal_candidates':C,
'node_review':[{'node':n,'reviewed':True,'effective_binding_checked':True,'notes':v} for n,v in node_notes.items()],
'limits':['8个亲读实现/参考材料文件与私有snapshot inputs.lock SHA一致；详work/extract_gemm/source_identity.json。无GPU运行或性能复测。','硬件Contract为冻结提取流程允许的全null占位，硬件正文0不能解释为完备或简洁；未补猜测硬件/工具链。','源TMEM缺陷已核对源码地址与row生成，但未独立展开目标PTX ISA lane映射或运行修复验证；仅保留现有必要来源限制。','错误API后实际compute提交序列按L条件覆盖；没有错误注入证据证明每个多次恢复序列实际可达、完成或正确。','所有组及其全部declaration/alias使用范围均已核对；scalars额外193种非开放字符串为引用、标识、shape/retention/pipe/event正式表达，未发现漏收重要叙述。','建议短写是有据保守候选，未另行生成全套替代YAML并独立等价审阅；类别面积不是可删除字数。','不把等价barrier/通知API差异当核心结构缺口：多数require原语绑定core=uncertain；未证明需要新增字段或事件。']}
(BASE/'annotations/extract_gemm.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('annotations',len(A),'findings',len(F),'candidates',len(C))
