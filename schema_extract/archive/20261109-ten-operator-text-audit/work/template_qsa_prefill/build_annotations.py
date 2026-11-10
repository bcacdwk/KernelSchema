import json
from pathlib import Path
R=Path(__file__).resolve().parents[2]
OP='template_qsa_prefill'
U=[json.loads(l) for l in (R/'data'/OP/'units.jsonl').read_text().splitlines()]
A={}; F=[]; C=[]
def uid(n): return U[n-1]['id']
def e(n):
 o=U[n-1]['occurrences'][0]; return f"inputs/{OP}/root.yaml:{o['line']}" if o['node']=='root' else f"inputs/{OP}/dispatch.md:{o['line']}"
def ann(n,cat,core,structure,reason,fs=[]):
 assert n not in A
 A[n]=dict(id=uid(n),category=cat,core=core,structure=structure,reason=reason,evidence=[e(n)],finding_ids=[OP+'-'+x for x in fs],contexts_checked=True,occurrence_overrides=[])
def many(ns,cat,core,structure,reason,fs=[]):
 for n in ns: ann(n,cat,core,structure,reason,fs)
def finding(k,kind,ns,claim,fact,counter,candidate,gap='none',ev=[],confidence='high'):
 F.append(dict(id=OP+'-'+k,kind=kind,unit_ids=[uid(n) for n in ns],claim=claim,core_fact=fact,deletion_counterexample=counter,existing_structural_candidate=candidate,gap_kind=gap,evidence=ev or [e(ns[0])],confidence=confidence))
def candidate(n,quote,replacement,reason,kind='redundancy',ev=[]):
 assert U[n-1]['text'].count(quote)==1,(n,quote)
 C.append(dict(unit_id=uid(n),quote=quote,replacement=replacement,kind=kind,reason=reason,evidence=ev or [e(n)],applies_to_occurrences='all'))
ann(1,'MIXED','some','mixed','数学关系与两次 BF16 舍入是正式内容；K/V 存储别名、外部副作用和计时边界仍由正文限定。',['F01','F04'])
ann(2,'ESSENTIAL_TEXT','all','open_semantics','配置布尔值被忽略是外部接口约定，dtype/shape 不说明其行为。',['F01'])
ann(3,'DUPLICATE','none','already_present','A 的 bf16 舍入先于 gate 已由 compute.math 的两个公式唯一给定。',['F02'])
for n,r in {4:'固定 GroupSize 的分组含义；value 给出 QueryHeads/KVHeads。',5:'token_to_req 的请求归属接口定义，range=[0,0] 限制本域。',6:'output_gate 是 sigmoid 的 logit 参数而不是权重。',7:'KVHeads 的参数定义。',8:'D 同时用作 query/value 维度的参数定义。',9:'J 是 valid-count 截断后的槽位索引集合。',10:'PageSize 的单位与物理页参数含义。',11:'TableWidth 的逻辑页宽度参数含义。',12:'scale 的注意力缩放参数含义。',13:'q 的 query 值接口定义。',14:'Rows 是 CP-local 行数，不能把它当全局 token 数。',16:'SelectedWidth 不包含末尾 valid-count word。',18:'CachePages 是物理缓存页数。',19:'k_cache 的 key 输入接口定义。',21:'logical_indices 的 token ID 与末尾 count 编码定义。',22:'QueryHeads 的参数定义。',23:'block_table 请求到物理页的函数接口定义。'}.items(): ann(n,'FORMAL','all','formula_or_predicate',r,['F01'])
ann(15,'DUPLICATE','none','already_present','K/V 共享分配、stride 的独占定义在 compute.math 首句；这里是其摘要。',['F01','F02'])
ann(17,'DUPLICATE','none','already_present','可省略由 default 给出；compute 全量覆盖 out 且公式不读取其旧值，range 也允许任意位。必填 meaning 仍需短写。',['F01','F02'])
ann(20,'ESSENTIAL_TEXT','all','open_semantics','out 与返回 O 是同一 tensor 的对象身份仅由此定义；shape/ref 不表达存储别名。',['F01'])
for n,r in {24:'fitted-step15 选择生成域及 CP rank 是正式谓词，不能当作冗长说明删掉。',25:'页映射分段函数是输入域的精确关系。',26:'映射页有限、其他页可 NaN/inf 的值域，两次声明分别核对 K 与 V。',27:'out 初值任意 BF16 位的有效域，允许 NaN/inf 不能省成默认有限值。'}.items(): ann(n,'FORMAL','all','formula_or_predicate',r,['F01'])
ann(28,'EXPLANATION','none','already_present','compute 标题，计算由 math 定义；字段必填，只宜短写。',['F02'])
ann(29,'MIXED','some','mixed','atol/rtol 为正式判据；返回对象与原地目标同时检查的验收路由不可只保留阈值。',['F01'])
many([30,31,33,34],'EXPLANATION','none','already_present','单 root 的导航/摘要；拓扑和默认覆盖由 root 文件及 DOC0 确定，dispatch 入口仍须保留。',['F02'])
ann(32,'MIXED','some','mixed','硬件/输入导航重复 root，但 serialized eager 限制影响持久 ticket/base 的适用范围，不能删掉限制后宣称覆盖回放。',['F10'])
ann(35,'ESSENTIAL_TEXT','all','open_semantics','CTA 共驻 SM、共享 SMEM/barrier 是执行单元的开放组成定义，下游 worker.unit 只引用名称。',['F03'])
ann(36,'NONCORE','none','not_applicable','当前 worker 是单 CTA，无 cluster 协作或跨 CTA 通信；属于设备能力背景。DOC2 要求同设备统一硬件事实，故不据此提出删条目。',['F03'])
ann(37,'ESSENTIAL_TEXT','all','open_semantics','32 SIMT threads/warp 及 CTA 层级承载 role 成员和同步规模的含义；不存在嵌套组成字段。',['F03'])
ann(38,'ESSENTIAL_TEXT','all','open_semantics','register 私有作用域用于解释 L、Reference 和暂存值，space 名本身不定义该硬件事实。',['F03'])
ann(39,'ESSENTIAL_TEXT','all','open_semantics','TMEM 与 SMEM 分离及显式分配是资源账成立的硬件条件。',['F03'])
ann(40,'DUPLICATE','none','already_present','DOC5 已要求逐原语完成/可见性保证，MMA/load 动作与 realization 已进一步给定；此句无独占算子事实。',['F03'])
ann(41,'MIXED','some','mixed','CTA 生存期共享存储为本设计提供作用域；cluster peer 可访问性在本单 CTA 路径未使用。',['F03'])
many([42,43],'FORMAL','all','formula_or_predicate','已选软件组件与版本的正式身份值，不是应删除的自然语言解释；工具链与语言的字段职责不同。',['F03'])
many([44,45],'MIXED','some','mixed','近似原语的 FP32 路径/适用域是核心，unknown 误差界是必要来源限制；跨步骤引用保留。',['F04','F12'])
many([46,48,51,56,59,62,63],'EXPLANATION','none','already_present','ledger 的 item 只标识该条计量对象，定量关系由相邻 change 给出；item 必填，不整体删除。',['F04'])
many([47,55,60,88,89,90],'FORMAL','all','formula_or_predicate','rewrite.step 与 ledger.by 是正式 ID/引用；逐 occurrence 核对 union/online/gate 的实际因果归属。',['F04'])
for n,r in {49:'实际计算的 padded/掩码位置计入 MAC，不能只按有效 selection 估算。',52:'baseline 固定每 query/KV-head gather 和全局 FP32 score/probability，决定差额的比较对象。',53:'消除 score/probability 全局往返的定量差额。',54:'Union/Count 的逻辑 live workspace 及两个 KV-head 共享范围。',58:'KV gather 的前后逻辑 payload，包括 tail padding。',61:'物化变携带状态的字节差额，排除临时量及复制 m 的限定不可去掉。'}.items(): ann(n,'FORMAL','all','formula_or_predicate',r,['F04'])
ann(50,'DUPLICATE','none','already_present','增加一次预处理遍历已由 union rewrite 和两次 launch 及 start_after 表达；账本仍有量化解释价值。',['F02','F04'])
ann(57,'EXPLANATION','none','already_present','列出公式和 approx 已明确的误差来源，无新误差界；账本的诊断用途仍有价值。',['F04'])
ann(64,'MIXED','some','mixed','union/packing/tail 公式与源中未初始化 Q padding 的限制混合；padding 参与 vote 未证实，不能整体按公式或冗余分类。',['F04','F12'])
ann(65,'FORMAL','all','formula_or_predicate','零分母、两次 bf16 舍入、exp2 gate 与 reciprocal 关系是数值定义。',['F04'])
ann(66,'FORMAL','all','formula_or_predicate','未掩码 vmax、共享 vote、可下降 lagged reference、两半 L 和 Y 递推均为正式数值定义；文本公式不构成语言漏洞。',['F04'])
for n,r in {67:'Br 的工作分块含义由绑定和 union 数学共同给出。',68:'Union 条目内容由 union.math 给定。',69:'Y 的分子定义由 online 递推给定。',70:'L 两半 FP32 由 shape/dtype 和 online 递推给定。',71:'Bk 的 token 轴含义由绑定与 Kblk/Vblk shape 给定。',72:'Kblk gathered 值由 union.math 给定。',73:'Vblk gathered 值由 union.math 给定。',74:'P_A 的第一列半区由在线公式切片给定。',75:'m 的复制关系由在线公式及 Reference 视图给定。',76:'P 的舍入由 P=bf16(E) 给定。',77:'Count 由 union 长度定义给定。',78:'Qblk packing 已在 union.math 给定。',79:'tau 的比较作用和基二单位由 online 公式给定。',80:'rescale_peer 的正式绑定与动作契约给出 vote 区间。',81:'Etile 由 work_schedule 绑定和 Union iteration shape 给定。',82:'M 由工作参数和 Qblk shape 给定。',83:'P_B 的第二列半区由在线公式切片给定。',84:'S 的未掩码 QK 由在线公式给定。'}.items(): ann(n,'EXPLANATION','none','already_present',r,['F02'])
many([85,86,87],'EXPLANATION','none','already_present','数学步骤标题概括其 math，没有独占数值定义；必填 name 可以简化。',['F02'])
ann(91,'EXPLANATION','none','already_present','收益/代价摘要由 union/online、ledger 的 gather/物化/MAC/pass 项支撑。',['F02'])
ann(92,'EXPLANATION','none','already_present','任务摘要由 Contract gated attention 及两次 bf16 公式支撑，必填 describe 保留简写。',['F02'])
ann(93,'EXPLANATION','none','already_present','三前导 QK、跨工作双槽环和 Y 双半释放由 MMA pipe、n flatten、reuse 和 A/B drain 明确给定。',['F02'])
ann(94,'DUPLICATE','none','already_present','wait:Qblk0 位于 prologue，默认匹配唯一 publish:Qblk0；空 worktile 不删除 prologue。保留长形式 execution_mode=serial。',['F07'])
ann(95,'ESSENTIAL_TEXT','all','open_semantics','Follow GatherK 把地址前视契约应用于 V；ValueRing 和 V delta 绑定须保留，已有短引用未机械复写长正文。',['F05'])
ann(96,'ESSENTIAL_TEXT','all','open_semantics','自定义 combine 的含义通过引用 A 的交换/捕获契约定义，通用动作表没有这一动作。',['F08'])
ann(97,'MIXED','some','mixed','QK/首次 PV 初始化可从 mm32 推出，但 operand-read completion 与 alias 读先于覆盖的原语保证不能由 execution_mode 推出。',['F06'])
ann(98,'ESSENTIAL_TEXT','all','open_semantics','accept 是自定义 ticket 复制/ack 动作；5 次声明各覆盖 GK/GV/MMA/A/B，terminal 不进入工作但仍需 ack。',['F07'])
ann(99,'MIXED','some','mixed','异步 Q refill 动作定义/valid-ticket 条件必须保留；最后 QK 退役的先后可由此前 HalfMax/P 的依赖传递推出。',['F08'])
ann(100,'ESSENTIAL_TEXT','all','open_semantics','store 在此还负责最后寄存器片段的 gate 计算，纯通用 store 不说明延迟计算的位置；A/B occurrence 对应不同半区。',['F08'])
ann(101,'MIXED','some','mixed','ticket 的数值与 Ticket0/Next 终止身份需定义；ack 全体等待已写在 Scheduler 的 sync 和 serial pipe，部分正文重复。',['F07'])
ann(102,'ESSENTIAL_TEXT','all','open_semantics','32 列分片，最后片留寄存器，done 只代表 Y 半区读取结束，是早释放与下一工作重叠的独占动作定义。',['F08'])
ann(103,'ESSENTIAL_TEXT','all','open_semantics','两份 L 经 HalfSum 交换并捕获 normalization；256-thread 数量可推导，但内部协作及捕获完成由此动作定义。',['F08'])
ann(104,'MIXED','some','mixed','混有初始 terminal Q、score 捕获、rescale_peer 定义、条件 PV 等待、mask 前视、未决 HalfMax 安全及 Qnext 发布；有重复边，也有不可删核心。',['F08','F09','F12'])
ann(105,'ESSENTIAL_TEXT','all','open_semantics','B 明确消费 Y 的第二半并继承 A 的早释放合同；不能仅从 Y_B 名猜切片。',['F08'])
ann(106,'ESSENTIAL_TEXT','all','open_semantics','warp prefix scan/atomic reservation 和 Count 在写完后发布决定预处理工作、并集顺序及访问安全，资源形状不定义算法。',['F05'])
ann(107,'MIXED','some','mixed','两级地址前视、同 entry 共享地址是主要成本机制；load.done 延后到下一 issue 的一部分已用 sync 边表示，但最终 drain/完成发布合同仍必需。',['F05'])
ann(108,'MIXED','some','mixed','B 的 score 捕获、32-row vote/mask 继承、L/P 半区与不修改 Y 为必要动作语义；rendezvous 的部分先后已由对等 HalfMax 边给定。',['F08','F09'])
ann(109,'ESSENTIAL_TEXT','all','open_semantics','Hash/CompactCount 初始化、缓存、quad 前视、CAS 线性探测和每个 inserter 的 OR 共同定义原子去重计算及发布安全。',['F05'])
ann(110,'DUPLICATE','none','already_present','Scheduler.wait 的全部前驱在 waits_for.sync 给定，serial pipe 保证下次 publish 在其后；没有额外动作含义。',['F07'])
ann(111,'ESSENTIAL_TEXT','all','open_semantics','drain 的最终发布与无请求 no-op 完成条件是自定义动作的必要合同，两 gather occurrence 均已核对。',['F05','F07'])
many([112,113],'MIXED','uncertain','mixed','保留提供抽象完成/可见性保证的源码原语绑定；上游已约束大部分依赖，未以更换 API 本身证明核心性能差异或缺字段。',['F11'])
ann(114,'ESSENTIAL_TEXT','all','open_semantics','索引连续对齐及 gate/out stride 是向量访问兼容条件，现有输入 shape/dtype 不表达物理 stride/align。',['F01','F11'])
many([115,116,117],'EXPLANATION','none','already_present','将 require 的选择关联到上游 alias、vector transfer 或跨工作重叠；没有新增角色/依赖，reason 必填宜短写。',['F11'])
many([118,119],'EXPLANATION','none','already_present','phase 摘要由 partition、invariant/iteration、target 和数学定义给定，describe 必填。',['F02'])
ann(120,'MIXED','some','mixed','退出是正式条件；已获工作完成/在途排空属标准职责，terminal ack 与 Scheduler 当前结构一致。保留退出谓词而非删除。',['F07'])
ann(121,'ESSENTIAL_TEXT','all','open_semantics','refill 键 atomic_counter 不定义领取数、base 减法及 terminal 停止规则；需明确协议。',['F07','F10'])
ann(122,'ESSENTIAL_TEXT','all','open_semantics','SMCount 的外部设备查询来源只在此绑定，不能删成未定义符号。',['F03'])
ann(123,'FORMAL','all','formula_or_predicate','Br=10 是绑定数值的唯一正式来源，不因存于 strategy 字符串就当作正文漏洞。',['F04'])
ann(124,'ESSENTIAL_TEXT','all','open_semantics','跨调用 counter/base 初值、每次增量、序列化及 int32 wrap 是稳定分配的必要状态协议；单次工作字段不自动推出。',['F10'])
ann(125,'MIXED','some','mixed','M/Etile/Bk 是参数定义；末块六行可由 Rows/Br 算出，无跨工作归约由 single_tile/完整 KV 迭代给出。',['F02','F04'])
ann(126,'EXPLANATION','none','already_present','两 kernel 次序、动态 worker 数、atomic refill 和 row-descending 排列已经结构化。',['F02'])

finding('F01','structural_gap',[1,2,15,17,20,29,114],
'外部对象身份、物理 K/V 布局、hint 无作用和计时边界依赖开放语义；公式主体仍是正式关系。',
'O/out 同对象、K/V 共 allocation 与 stride/base delta、out 完整覆盖、use_prefill_config 忽略。',
'只保留 shape/dtype/输出数值时，可返回新 tensor 而未按要求返回 out；独立 K/V allocations 同 shape 也满足余下结构。',
'现有 output.meaning/compute.math 与 realization.require 可承载这些合同，但没有 alias/stride 固定字段。ref:O.shape 只共享属性，resource.view 只适用于 worker 片上存储，不能替代外部 tensor 身份。保留紧凑 O meaning: "Return out after writing O."；不把移动到 choice 当结构化。',
'open_semantics',['inputs/template_qsa_prefill/root.yaml:20','inputs/template_qsa_prefill/root.yaml:37','rules/schema_docs/1.contract.algorithm_requirement.md:47','rules/schema_docs/0.common.md:70','work/template_qsa_prefill/operator.py:49'])
finding('F02','redundancy',[3,15,17,28,30,31,33,34,50,*range(67,88),91,92,93,118,119,125,126],
'摘要/标题、若干 meaning、末块算术与已声明顺序存在可追溯重复；不是所有这些字段都允许省略。',
'数值公式、tile 选择和 pipeline 图保留；mandatory name/meaning/describe/thesis 仍给非空短写。',
'精确短写不删除公式、发布、角色或资源；不适用核心反例。',
'保持 partition.row={size:Rows,tile:Br}、target.single_tile.O、MMA.pipe、n.flatten 和 reuse；例如 attention.strategy 仅写 "M=128; Etile=16; Bk=4*Etile."，无须重写 tail=6。',
'none',['inputs/template_qsa_prefill/root.yaml:175','inputs/template_qsa_prefill/root.yaml:216','inputs/template_qsa_prefill/root.yaml:298','rules/schema_docs/4.design.work_schedule.md:154'])
finding('F03','structural_gap',[35,36,37,38,39,40,41,42,43,122],
'硬件层级/可见性是开放组成语义，软件版本为正式身份；cluster 背景对本路径无核心影响。',
'warp=32 threads、CTA scope、TMEM/SMEM/register 分离与 SMCount 的提供者。',
'单有 worker.unit=CTA 和 resource.space=TMEM 时，无法推出组成、共享范围或 TMEM 是否与 SMEM 同容量池。',
'已核对 execution/feature/stack 固定边界：execution.<unit> 的值只能是摘要，不能写 threads/parent 嵌套映射。保留短定义；设备级 wiki 可集中复用事实，但 pending 入口没有提供真实结构替代。',
'open_semantics',['inputs/template_qsa_prefill/root.yaml:59','inputs/template_qsa_prefill/root.yaml:187','rules/schema_docs/2.contract.hardware_specification.md:76','rules/schema_docs/2.contract.hardware_specification.md:174'])
finding('F04','style',[1,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,88,89,90,123,125],
'正式公式、符号/步骤身份、参数与成本模型占有文本面积，但不是已证结构漏洞。',
'union 按四 token packing、unmasked max、lagged reference、BF16 P、两半 L、最终两次舍入及非空 N>=32。',
'省去 m 的触发规则可变成每 tile 更新；省去第一次 bf16 可融合 gate 舍入；二者都改变数值路径。',
'现有 rewrite.math、symbol.value/binding/retention、arithmetic.approx、ledger.by 已是合适结构。参数 Br/M/Etile/Bk 按 DOC3 下游绑定定义，不需仅为去字符串增字段。',
'none',['inputs/template_qsa_prefill/root.yaml:83','inputs/template_qsa_prefill/root.yaml:104','inputs/template_qsa_prefill/root.yaml:129','rules/schema_docs/3.design.math_reshape.md:245'])
finding('F05','structural_gap',[95,106,107,109,111],
'预处理哈希/scan 与 gather 前视仍由动作契约定义；延迟发布的部分顺序已经显式结构化。',
'one-CAS/group、每个 inserter OR、one atomic reservation/warp；Union 两 tile 和 PageTable 一 tile 前视，同 entry 地址复用；跨 worktile 延迟完成和末次 drain。',
'删去 build/compact 定义后，资源/两 action 名允许逐 token 原子插入或每 entry reservation，主要工作量不同；删去 load 前视允许同步串行地址链。删去 drain 合同则末块和空 worker 完成条件不确定。',
'已经存在 waits_for.sync: {"load:Kblk(n)@done": ["load:Kblk(n+1)@issue"]}、n={flatten:[t,i]} 和 KeyRing.slot_count=2，可删正文中重复这条边的子句；最终 drain 保留非空发布/no-op 合同。地址解析本身及原子算法没有封闭动作字段，不能只新增 resolve 名称宣称已表达。',
'open_semantics',['inputs/template_qsa_prefill/root.yaml:209','inputs/template_qsa_prefill/root.yaml:256','inputs/template_qsa_prefill/root.yaml:269','work/template_qsa_prefill/qsa_preprocess.py:219','work/template_qsa_prefill/qsa_preprocess.py:281','work/template_qsa_prefill/qsa_kernel.py:792'])
finding('F06','structural_gap',[97,112,115],
'S/P alias 已由 resource.view 表达；操作数读完成与后续写覆盖的原语保证仍需独立定义。',
'PV(n) 先读 P，随后 QK 才可写复用的 S/P；done 包括操作数读取。',
'只保留 async_ordered_completion 可允许后续写覆盖先前尚未读完的操作数；完成事件有序不自动保证读写冲突解决。',
'view.P={part:{column:[Bk*(n%3),Bk*(n%3)+Bk/2]},dtype:bf16} 已表达物理共用；MMA.execution_mode 已表达累积完成。保留 "Completion includes operand reads; earlier PV reads precede later writes to its aliased S/P stage."；若拆 read-release 事件须有源码原语依据，不能额外插入强等待。',
'open_semantics',['inputs/template_qsa_prefill/root.yaml:228','inputs/template_qsa_prefill/root.yaml:296','rules/schema_docs/5.design.tile_pipeline.md:145','work/template_qsa_prefill/qsa_kernel.py:684'])
finding('F07','redundancy',[94,98,101,110,111,120,121],
'请求/ack 与可工作 ticket 分开后，Scheduler 的全体确认等待已是现有图的事实，不能把协议名当缺结构。',
'accept.done 是复制和 ack；terminal 仍 accept；wait 等五 role 共十一 warp；只在下一票被所有角色领取后继续。',
'删所有文字会使 accept 和 terminal 意义丢失；仅删 wait 重述不丢边。不能以当前工作完成代替 ack，否则丢一工作 lookahead。',
'现有 Scheduler.pipe={prologue:[publish:Ticket0,wait:Ticket0],steady:[publish:Next(t),wait:Next(t)],repeat:t} 与 wait@issue 的五个 accept@done 前驱已足够表达 ack 顺序。MMA prologue wait:Qblk0 默认匹配 A.publish:Qblk0；空 repeat 仍执行 prologue/epilogue。',
'none',['inputs/template_qsa_prefill/root.yaml:241','inputs/template_qsa_prefill/root.yaml:245','inputs/template_qsa_prefill/root.yaml:248','rules/schema_docs/5.design.tile_pipeline.md:175','work/template_qsa_prefill/qsa_kernel.py:318'])
finding('F08','existing_structure',[96,99,100,102,103,104,105,108],
'Q refill 的已退役顺序、HalfMax 后的 peer capture 和 Y 早释放已有图支撑；切片/内部 capture/normalization 合同仍必要。',
'末片先读入寄存器即释放 Y，gate/store 可继续；Qnext 在 A drain 后发布；两半都 release 后才允许下一工作 first PV。',
'将 drain.done 改成整个输出完成会加强顺序；仅有 Y_A 名不能推出列区间；删 score register capture 可能在 peer 读完之前覆盖 aliased S。',
'已有 MMA.reuse["mma:P(n[t+1,0])@Vblk(n[t+1,0])@issue"]=[SoftmaxA/drain:Y_A(t)@done,SoftmaxB/drain:Y_B(t)@done]。A.serial pipe: drain:Y_A(t), publish:Qnext(t), store:O_A_tail(t)。HalfMax→P 对等边加 serial 顺序保证 capture 后才进入 P；可用 Accumulator.view 添加 Y_A:{part:{column:[3*Bk,3*Bk+D/2]}}、Y_B:{part:{column:[3*Bk+D/2,3*Bk+D]}} 显式登记区域，但内部寄存器尾片与提前释放仍须契约；不把视图当同步。',
'none',['inputs/template_qsa_prefill/root.yaml:313','inputs/template_qsa_prefill/root.yaml:320','inputs/template_qsa_prefill/root.yaml:338','rules/schema_docs/5.design.tile_pipeline.md:282','work/template_qsa_prefill/qsa_kernel.py:615'])
finding('F09','existing_structure',[104,108],
'条件 PV 退役等待尚未写在图内。现有 conditional repeat/wait 可表达，但按四个静态 warp role 展开有显著成本；不是已经证明要扩语言。',
'只有 A 的触发 vote warp 在 i>0 时等待 PV(n-1)，然后 rescale 自己32行的全 D；B 只改自身 L/P，不等待该 PV 也不修改 Y。',
'当前 P_A 的 HalfMax/QK 前驱最多保证较早 PV 已完成，不能推出 PV(n-1)；删该句允许 rescale 与 PV 对 Y 并写。全 A 无条件等 PV(n-1)又消除未触发 warp 的提前推进。',
'对 w=0..3 分别登记静态 SoftmaxAw，member=warp[w:w+1]（实例中用具体数字）。定义 prepare:WeightsAw(n)：捕获已交换 maxima，依 online 公式计算该32行的 move_ref/alpha/weights，保留 weights 与 current mask；定义 wait 的 serial 契约为只等显式前驱；定义 rescale:YAw(n,q)：对行[32*w,32*(w+1))的全D执行 alpha*Y，done 保证写完。每轮 pipe 先 prepare，再 repeat:{q:[0,"1 if i>0 and move_ref(i,32*w) else 0"]},steady:["wait:PriorPV(n,q)","rescale:YAw(n,q)"]，最后 publish:PAw(n)（存该32行 BF16 P 并更新 L）；waits_for.reuse["wait:PriorPV(n,q)@done"]=["MMA/mma:P(n-1)@Vblk(n-1)@done"]。PAw 发布后让 MMA 同时等待四份 PAw 和 B 的 P_B。须同步拆分 HalfMax/ack/Q/refill/drain 为各 warp 的实际 slice 并保留原有256线程 rendezvous；这是一套可表达方向，不是已审阅的整份等价短稿，不能据此删原句。',
'representation_cost',['inputs/template_qsa_prefill/root.yaml:324','rules/schema_docs/5.design.tile_pipeline.md:101','rules/schema_docs/5.design.tile_pipeline.md:409','work/template_qsa_prefill/qsa_kernel.py:546'],confidence='medium')
finding('F10','structural_gap',[32,121,124],
'跨调用共享 ticket/base 的状态与序列化约束不由单次 worker assignment 自动表达；capture/replay 没有被本样本证明覆盖。',
'设备 ticket 和 host base 起始0；每次 eager 消耗 W+C 个票、base 加同数、int32 wrap-safe；禁止把每调用重置 counter 当成同设计。',
'若相同捕获的按值 tbase 第二次回放而 ticket 已增加，所有 worker 可拿 terminal 并不写本次 O；剩余单次 partition/order 字段无法排除。',
'可把增量正式缩写成 Delta=worktile_count+worker_count; base(k+1)=base(k)+Delta; id=int32(ticket-base(k))，但仍需开放合同说明设备/host 两份跨调用状态及 eager 序列化。resource 只登记 worker 局部资源，handoff 只管一次实例的 phase，不是真正跨调用状态字段。现有 dispatch.when 能明确 eager 适用域；它不能创造安全的 replay 实现或允许覆盖缺口消失。',
'open_semantics',['inputs/template_qsa_prefill/root.yaml:189','inputs/template_qsa_prefill/dispatch.md:12','rules/schema_docs/0.common.md:39','work/template_qsa_prefill/operator.py:108','work/template_qsa_prefill/operator.py:153'],confidence='high')
finding('F11','style',[112,113,114,115,116,117],
'Realization 绑定来源原语/布局兼容条件，不以 API 差异本身统计已证核心结构缺口。',
'TMEM 操作数 handoff 和 copy 完成/可见性已在上游说明，require 提供具体实现来源；alignment/stride 则是额外 ABI 条件。',
'换一个保持全部 read-complete/publication 保证的 API 未证明改变核心设计；alignment 不满足却仍发128-bit访问则可违反已选数据路径。',
'现有 require:item/choice/reason 是所需开放绑定。可缩 choice 为 "tcgen05.mma providing this role\'s TMEM and operand-read guarantees." 或保留 cp.async 的两种 wait_group 语义；没有 measured cost 支持删绑定后仍完全等价，故不提交整条删除。',
'none',['inputs/template_qsa_prefill/root.yaml:382','rules/schema_docs/6.realization.md:25','rules/schema_docs/6.realization.md:135','work/template_qsa_prefill/operator.py:64'])
finding('F12','source_limitation',[44,45,64,104],
'Q padding 对32行 vote 的影响、HalfMax 后继覆盖安全和近似误差界均未完成证明，应保留限制。',
'非真实 packed rows 未写 Q；vmax/vote 含 padding；HalfMax peer-load 后没有额外 read-complete barrier；exp2/rcp 误差界 unknown。',
'把 padding 置零或额外同步可以改变数值/执行与成本，不能借审计修复；删除限制会将未验证事项伪装成已证。',
'现有 approx 保留 unknown；resource.view 只给别名范围，不能自动证明所有读者退役。若以后源码证实额外读退役顺序可使用 reuse 边，现在不能虚构。',
'none',['inputs/template_qsa_prefill/root.yaml:88','inputs/template_qsa_prefill/root.yaml:126','inputs/template_qsa_prefill/root.yaml:327','work/template_qsa_prefill/qsa_kernel.py:510','work/template_qsa_prefill/qsa_kernel.py:874'])
finding('F13','source_limitation',[17,20,32],
'源码接口接受显式 out=None，而当前 Contract 只通过 default 表示可省略，未填 nullable；此处是源到选择域的边界待确认，不是语言缺字段。',
'缺省分配和显式 None 是不同接口维度，DOC1 已有 nullable 表达。',
'按当前 Contract 严格解读，显式 None 不在有效域；源码会分配并正常返回。不能因 optional 一词自动扩张已冻结 Contract。',
'如原任务域确含显式 None，可用 input.out.nullable:true, default:"None" 并在 compute.math 定义 "If out is None, allocate an output tensor; return the destination tensor after writing every element."。本审计不改 Contract；原始任务对显式 None 的验收范围未独立获取。',
'none',['inputs/template_qsa_prefill/root.yaml:35','inputs/template_qsa_prefill/dispatch.md:12','rules/schema_docs/1.contract.algorithm_requirement.md:257','work/template_qsa_prefill/operator.py:51'],confidence='medium')

candidate(110,U[109]['text'],'','verb.wait 可整体省略：Scheduler 自身 serial，wait 通用语义及 waits_for.sync 已完整给前驱；保留 pipe 的 wait 动作和全部事件。')
candidate(94,U[93]['text'],'Wait for Qblk0.','仅缩 meaning，不删除 long-form wait 的 execution_mode:serial；prologue 和默认 publish 匹配保留 terminal 路径的等待。')
candidate(101,"Before claiming the following ticket, await all eleven consumer warps' copies and acknowledgements of the current ticket. Acknowledgement permits one-worktile lookahead while that work is still executing.",'','Scheduler 两条 sync 边列出五 role=11 warp，serial publish/wait pipe 限制下一 claim；接受确认与工作完成仍由 accept 的合同区分。')
candidate(107,"load:Kblk(n) becomes done only after issuing tile n+1 and publishing n's completed copy.","load:Kblk(n) done publishes its completed copy.",'现有 sync 明确 load(n+1).issue -> load(n).done；保留完成copy发布保证，不动 cross-worktile/drain 句。')
candidate(104,"Both score halves have been captured before either group overwrites the aliased stage with P; HalfMax_A done precedes peer-maximum capture.",'','已保留 A/B HalfMax 将各自完整 score 半区读入寄存器的合同、各自 serial 顺序，以及对方 HalfMax.done -> P.issue 边；不省 HalfMax 未验证覆盖限制。')
candidate(125,'The final row block contains six rows; no cross-worktile reduction is needed.','','Rows=4096、Br=10 与 partition 边界推导最后6行；target.single_tile.O 与全部 union 迭代在同工作内，未引入 tile_reduction。')
candidate(17,U[16]['text'],'Output destination.','保留 default 和完整覆盖 out 的 compute；meaning 仍是合法非空对象摘要。')
candidate(15,U[14]['text'],'Paged values.','精确 K/V alias 和 stride 的唯一事实仍在 compute.math 首句；meaning 保留必填对象摘要。')
candidate(99,U[98]['text'],"Issue the next valid ticket's Q gather into Qblk.",'SoftmaxA 每工作最后 HalfMax 等最后 QK done，随后的 serial P/accept 再到 async prefetch.issue；退休先后已传递保证，保留动作目标及valid条件。')
candidate(92,U[91]['text'],'Selected paged attention with BF16 rounding before the sigmoid gate.','describe 必填保留短写；数学公式仍完整给出最终输出舍入。')
for c in C:
 n=next(i for i in range(1,127) if uid(i)==c['unit_id'])
 support={110:[245,248,249],94:[300,336],101:[245,248,249],107:[258,269],104:[338,348,368,376],125:[161,176,186],17:[35,40,47],15:[40],99:[338,340,347],92:[40]}[n]
 c['evidence'] += [f"inputs/{OP}/root.yaml:{line}" for line in support]
assert set(A)==set(range(1,127)), set(range(1,127))-set(A)
# Dispatch occurrences use node=dispatch.md in census; fix evidence by actual snapshot identity rather than node naming.
for n,a in A.items():
 o=U[n-1]['occurrences'][0]
 a['evidence']=[f"inputs/{OP}/{'dispatch.md' if U[n-1]['family']=='dispatch.md' else 'root.yaml'}:{o['line']}"]
report=dict(operator=OP,prompt_version='v1.1',unit_annotations=[A[i] for i in range(1,127)],findings=F,removal_candidates=C,node_review=[dict(node='root',reviewed=True,effective_binding_checked=True,notes='root v9/template；Contract/全部3个rewrite、preprocess/attention两个phase、7个role、资源/别名/所有本地事件和realization全部亲读；无alias/from_parent/分支；5个dispatch组和459个scalar均检查，未发现开放正文漏项。')],limits=[
'只审计冻结模板 root v9；template 不是 GPU/完整语义通过基线。本轮不运行 GPU，也不修改原 schema、规范或 Runner。',
'源码以 01fff90d714abc6e84904ea9c44523ba66d34a0e 定位，完整亲读 solution.py、reference.py、operator.py、qsa_kernel.py、qsa_preprocess.py。工作目录副本保留原字节；未独立复查外部 PTX/CUDA 手册与环境记录，hardware wiki 仍 pending。',
'Q padding、HalfMax 覆盖安全、近似误差界保持未验证；未用新增同步或初始化伪造正确性。',
'条件warp等待的现有结构方案尚未生成并独立审阅整份替代 YAML，标记 representation_cost 而不是已证缺失语言字段；不能把该方案算删减量。',
'候选仅为逐 quote 的保守文字净缩减，未完成独立短稿语义等价审阅；全字段分类面积绝非可删字符。',
'captured tbase 与设备 counter 生命周期提示 replay 覆盖风险，本样本只明确 serialized eager，未执行捕获/回放测试。显式 out=None 的原始任务有效域尚未独立确认。'])
(R/'annotations'/f'{OP}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('wrote',len(A),'annotations',len(F),'findings',len(C),'candidates')
