import json,pathlib,collections,re
R=pathlib.Path(__file__).resolve().parents[2]
U=[json.loads(l) for l in (R/'work/extract_gdn/reviewed_316_units.jsonl').read_text().splitlines()]
assert len(U)==316
A={}; F=[]; C=[]
RULE='rules/schema_docs/'
def ids(ns):return [U[i-1]['id'] for i in ns]
def ev(ns):return list(dict.fromkeys(f"inputs/extract_gdn/{o['node']+'.yaml' if o['node']!='dispatch.md' else 'dispatch.md'}:{o['line']}" for i in ns for o in U[i-1]['occurrences']))
def put(ns,cat,core,structure,reason,finds):
 for n in ns:
  assert n not in A,n
  A[n]={'id':U[n-1]['id'],'category':cat,'core':core,'structure':structure,'reason':reason,'evidence':ev([n]),'finding_ids':['extract_gdn-F'+str(f).zfill(2) for f in finds],'contexts_checked':True,'occurrence_overrides':[]}
def finding(n,kind,ns,claim,core,counter,candidate,gap='none',evidence=None,confidence='high'):
 F.append(dict(id=f'extract_gdn-F{n:02}',kind=kind,unit_ids=ids(ns),claim=claim,core_fact=core,deletion_counterexample=counter,existing_structural_candidate=candidate,gap_kind=gap,evidence=(evidence or ev(ns)),confidence=confidence))
# Explicit manual groups, after reading all 316 exact texts and every node's structural diff.
put([1,3,4],'MIXED','some','mixed','外部副作用/有效域与正式关系不可丢；同时夹有默认FP32、有限输入或接口摘要等已覆盖说明。F01/F02逐项区分，不能整段删。',[1,2])
put([2],'MIXED','some','mixed','递推、分组head映射、先衰减后残差及更新后读出由公式定义；后附的FP32/RNE与读出解释部分可由公式和共享默认推出。',[2])
put(list(range(5,28)),'EXPLANATION','none','already_present','简短对象释义；对应Contract compute中的packing、query_key_values、delta_recurrence或output_and_side_effect已唯一给出对象关系，shape/dtype/nullable登记属性。必填meaning有导航价值，不等于可删整个字段。',[4])
put(list(range(28,33)),'FORMAL','all','formula_or_predicate','输入值域谓词或对packing中正式关系的引用，约束值而不是优化解释。',[2])
put([33,34,35],'SOURCE_LIMIT','some','open_semantics','保留来源阈值、比较对象、None分支和外部比较器未定义项；既不能套用默认atol/rtol解释，也不能推为误差保证。',[3])
put([36,37,38,39,40,41,42,43,45,46,47,48],'EXPLANATION','none','already_present','中文dispatch导航/路径差异摘要；逐项核对六份YAML的when、parent、flow和局部资源，精确定义仍以YAML为准。',[4,5])
put([44],'SOURCE_LIMIT','some','mixed','dispatch中的来源限制汇总涉及tensor-map失败、core副作用偏差、最终mini读取退休缺口、跨stream复用与误差未知；均有YAML对应，不可当作已解决。',[3,12,15,16])
put([49,50,51,52],'MIXED','all','mixed','选择公式需要保留；LC绑定的是进程静态缓存而非当前设备属性，requested bytes不是已分配容量。这些host状态来源不能仅由shape谓词推出。',[5,16])
put([53,54],'MIXED','some','formula_or_predicate','保留等价选择谓词及SM来源；G/VTT、排序条件和范围推导属正式关系，证明性解释可缩，不能删除N/长度约束。',[5])
put([55],'MIXED','some','formula_or_predicate','八个16-key块的累加分组是必要数值事实；BF16输入/FP32结果以及已定义scan/rho分组部分重复mm32与上文。',[2])
put([56,57,60,61],'MIXED','some','mixed','保留相对于tokenwise参考的实际累加及compiler contraction绑定；mm32精度或上文已写的循环方向可缩。编译器/架构未固定到足以推出逐位数值，见F03限制。',[2,3])
put([58,59,83,86,87],'SOURCE_LIMIT','some','open_semantics','主动近似/数值偏差和缺少全域误差界是必要来源限制；不能从阈值、源码注释或历史批准推导精度保证。',[3])
put([62,63,64,67,68,71,72,76,78,85,88,91,100],'EXPLANATION','none','already_present','账本对象标签或明确对照基线；量与计算已由math定义，条目帮助读者审查代价，不另外定义设计。',[4])
put([65,66,69,70,73,74,75,77,79,80,81,82,84,89,90,92,93,94,95,96,97,98,99,101],'FORMAL','none','formula_or_predicate','正式成本关系或ledger.by步骤引用；属于计量/引用字符，不能算纯自然语言或直接可删量。计量必须保留适用范围。',[2])
put([102,103,109,117],'MIXED','some','mixed','chunk/mini数值路径、窄化分叉与输出终点不可丢；尾部混有Contract接口重复及非连续core写临时副本的必要源码偏差。',[2,15])
put([104,108],'FORMAL','all','formula_or_predicate','稳定降序排列、同计数按原请求顺序、每调用物化及原坐标写回构成正式映射定义。',[2,6])
put([105,107,112,115],'MIXED','some','mixed','全B行foreign/past-T处理与g/beta独立零填充是核心边界；物化选择也已由retention/handoff承载，首次PRE0无读者须保留。',[2,6,8])
put([106,110,111,113,114,116],'MIXED','all','formula_or_predicate','数学求值与边界正式定义，包括空segment、非有限零权项、不同merge括号次序或pad求值；夹有保留/传输说明，但不能按长段截断或简化成理想逆/精确递推。',[2,3,6,11])
put(list(range(118,159)),'EXPLANATION','none','already_present','符号短释义已与本节点math、binding/value、shape/dtype/retention逐项核对；正文定义给出精确值和用途。该必填接口说明并非独占核心事实。',[4])
put(list(range(159,170)),'EXPLANATION','none','already_present','thesis/任务摘要概括已在本节点math、schedule、pipe和源码限制中说明的方案及代价；未发现仅此处定义的执行关系。',[4])
# Fine-grained action judgments: internal cooperative semantics are not supplied by verb names.
manual={
170:('ESSENTIAL_TEXT','all','open_semantics','原始perm字节的投机暖读不代表有效Perm读取，丢弃值且不驱动地址/控制是关键；普通load:Perm会错误要求初始化和就绪。',[13]),
171:('MIXED','some','mixed','4×2 warp结果分工与QS跨solve存活仍在文字；KS/QS数学及b不用的说明已有上游。',[10,12]),
172:('MIXED','some','open_semantics','Rh发布CTA join和每warp 16-key状态归属没有结构化成员切分；prefetch/seed先后已有pipe。',[10,12]),
173:('ESSENTIAL_TEXT','all','open_semantics','暖读perm允许旧/未初始化字节，但不能驱动地址/控制；预等待不能提前读取状态或chunk。',[13]),
174:('MIXED','some','open_semantics','inverse/system初始化数值见math；全部scratch初值发布所需CTA join及范围在动作中。',[10,11]),
175:('ESSENTIAL_TEXT','all','open_semantics','PRE0先发Q/K/V、一次全StageBytes expectation后补系数，以及无PRE0仅预取状态，决定可重叠边界；pipe顺序不能推出分量/交易范围。',[8,9]),
176:('MIXED','some','open_semantics','Y计算见inverse公式，放MergeScratch并CTA发布到所有merge读者是额外核心保证。',[10,12]),
177:('MIXED','some','open_semantics','状态更新和最终Rh见math；无尾CTA join、下一advance退役及末尾仅读本warp寄存器说明真实协作边界。',[12]),
178:('MIXED','some','open_semantics','width-16四次merge及舍入见math，Z留寄存器、Ti写后CTA join仍是动作契约。',[10]),
179:('MIXED','some','mixed','fresh resources可由分立launch推出，A/C无局部身份延续；barrier初始化/发布及warp推进限制仍需契约。',[9,10]),
180:('ESSENTIAL_TEXT','all','open_semantics','prime一次发出NSTAGE-1轮的三种搬运及统一字节期望，done仅发射而非到达；资源slot数不定义该协议。',[8]),
181:('MIXED','some','mixed','完整分支的per-warp输出/状态/末尾overlay顺序不可丢，VTT=16域排除mini；部分资源与数学、下一轮生命周期已有定义。',[12]),
182:('MIXED','some','open_semantics','prime后读状态的宏观位置由pipe给出，Rh CTA发布和warp状态归属仍在契约。',[10,12]),
183:('MIXED','some','open_semantics','稳定排序结果已有schedule定义，但thread0排序与独立warps门控复制并发、128/64线程joins和分ratio扫描顺序未由单Main结构承载。',[10,6]),
184:('MIXED','some','open_semantics','input-order解析及初態预取的先后由pipe支持，但仅输入读且无chunk发射的动作效果需明确。',[9,12]),
185:('MIXED','some','mixed','slot/modulo已有view；parity acquire、CTA全读者join后才发未来轮以及传输在途是核心独有协议。',[8,12]),
186:('NONCORE','none','not_applicable','直接QK分支没有提交image bulk组；源码空wait无数据依赖，只是低阶实现指令，不定义主要性能因果。',[17]),
187:('MIXED','some','mixed','同一自定义动作混合暖读、PDL wait和fresh perm解析；后两者有既有结构候选，暖读允许未初始化原始字节仍需契约。',[9,13]),
188:('MIXED','some','open_semantics','输出对象/ready已有handoff，组间join退休全部shared读者、末组省join和done只承诺读完/发射是独有边界。',[10,12]),
189:('ESSENTIAL_TEXT','all','open_semantics','C prime的完整Q/K+系数+V发射与entry composition重叠、done非arrival未由现存pipe事件分离。',[8]),
190:('MIXED','some','open_semantics','wait后重新读取Perm而非沿用暖读值是核心；延迟seed位置可由pipe推出。',[13]),
191:('MIXED','some','open_semantics','FP32列代换与BF16写回已由math定义，寄存器局部求解和发布Ti的CTA join是额外动作保证。',[10]),
192:('MIXED','some','mixed','Qd数值已定义；异步直接global存储及其不阻塞solve8由async结构部分承载，来源done/完成语义需与kernel_done保留一致。',[9,10]),
193:('MIXED','some','open_semantics','G=2 SeedC公式已定义；R寄存器形成、Rh发布及每warp状态归属仍由动作补足。',[10,12]),
194:('ESSENTIAL_TEXT','all','open_semantics','入口metadata并行读取/安全clamp、barrier初始化、inactive CTA触发launch及GroupWidth来源是必要动作和参数定义。空a域只覆盖后续计算。',[10,11]),
195:('MIXED','some','open_semantics','6 warp分工、共享K load的Gram/F组合及F存活是核心；masked未写Gm读取的无影响限制不能从shape推断。',[10,11]),
196:('MIXED','some','mixed','Pitch/StageBytes/Partials是正式参数，barrier发布和launch trigger为动作效果；warp同步边界需保留。',[8,9,10]),
197:('MIXED','some','mixed','参数公式及barrier初始化是核心；MODE0无调用内后继可由flow读出，但不能据serial角色默认所有warp同步。',[8,10]),
198:('ESSENTIAL_TEXT','all','open_semantics','门控复制与QK TMA独立完成、scan的128/64线程subset joins及ratio8分pass顺序，是单Main动作内部的主要重叠因果。',[10]),
199:('ESSENTIAL_TEXT','all','open_semantics','入口metadata读、inactive触发和GroupWidth=HeadsPerPrep为本域具体契约；G=2域不能借用通用2-slot假设。',[10,11]),
200:('ESSENTIAL_TEXT','all','open_semantics','PRE0的首stage只补系数且不二次arrival是必要协议；计数和一次性字节expectation不由slot_index得出。',[8]),
201:('MIXED','some','mixed','初始化数值见math，所有scratch只做一次skeleton及组间不重置由pipe部分承载；CTA发布和覆盖集合仍需说明。',[10,11]),
203:('MIXED','some','mixed','SegB/SegA公式及空segment已定义，发射/全局可见性区分由kernel_done部分承载；一线程写aprod和末尾无额外drain需明确。',[9,11]),
204:('MIXED','some','open_semantics','foreign/past-T边界重复math；四个8192B输入boxes与系数/V的统一交易、done仅issue是核心。',[8]),
205:('MIXED','some','mixed','cl槽/parity与下一轮公式有结构成分；CTA全读者退休后才发未来轮及arrival保证仍在契约。',[8,12]),
206:('MIXED','some','mixed','与A相同环协议在C中保护RhsOutput；不是仅凭相同字面继承A资源。',[8,12]),
207:('MIXED','some','mixed','现存waits_for给出save_image.done到drain.done；实际bulk等待原语与源buffer未覆盖是完成依据，后半无每head wait可由pipe推出。',[9,12]),
208:('MIXED','some','open_semantics','perm数学已有定义；唯一producer、sort与门控copy并发、subset joins及扫描分组需动作契约。',[6,10]),
209:('MIXED','some','open_semantics','Hrhs数学已知；Ti预载与CTA发布是独占同步事实，A无输出亦由上游target可见。',[10,12]),
210:('MIXED','some','open_semantics','SeedC公式/方向已知，prefetched z0与寄存器composition、Rh CTA发布/warp ownership仍需契约。',[10,12]),
211:('MIXED','some','open_semantics','最终转置数学已知，float4流水在发当前store前读下一块是性能选择；无后续worktile/future chunk由work分配和prime域可推出。',[12,17]),
212:('MIXED','some','open_semantics','ratio2/3共享K的融合及其余独立loop、6warp与F跨head存活是核心，数学dtype和跳过upper tile已给出。',[10,11]),
213:('MIXED','some','open_semantics','Y/对角Ti读取和lower-left结果见math；CTA join是唯一发布保证。',[10]),
214:('MIXED','some','mixed','U/Uh/DU数学重复，但Qd/K预载及CTA全部Hrhs读者退休不能删；mini无读写保留事件符合共享no-op规则。',[11,12]),
215:('MIXED','some','mixed','launch_ready和payload ready不同已由flow结构分离，实际trigger位置/行为仍需原语绑定。',[9]),
216:('MIXED','some','open_semantics','prime先于seed已有pipe；state读取、Rh发布和每warp 16-key所有权仍在动词契约。',[10,12]),
217:('MIXED','some','mixed','VTT=32/64参数定义、barrier初始化及warp同步限制需保留；MODE0无后继只是flow解释。',[8,10]),
221:('MIXED','some','mixed','KS数值/split/b不用已由math；4×2 warp分工及rhs处合并仍定义实际存活/执行。',[10,12]),
222:('MIXED','some','mixed','float4提前读取流水需保留；FinalRows和no-op是否执行由pipe/math界定，末尾kernel退休共享规则可缩。',[11,12,17]),
223:('MIXED','some','mixed','VTT=32条件已核对；warp结果划分和QS跨solve保留仍独有，数学及unused b说明重复。',[10,12]),
225:('MIXED','some','mixed','perm/coeff全局就绪由handoff.kernel_done给出，actual trigger在input join后是不可从名称推出的效果。',[9]),
226:('MIXED','some','open_semantics','Hrhs公式已有，Ti preload和CTA join为独有发布/重叠事实。',[10,12]),
227:('MIXED','some','mixed','U/DU公式已有；Ti/K预载、全部Hrhs退休与mini无访问契约必要，no-op事件存在性来自共享规范。',[11,12]),
228:('MIXED','some','mixed','C完整/mini suffix包含每warp先后与全读者退休；final mini stage0来源缺口必须保留，不能用单动作假造跨warp屏障。',[12]),
229:('MIXED','some','open_semantics','Sys公式已给出；one/two head协同发布CTA join需动词补足。',[10]),
230:('MIXED','some','open_semantics','U窄化分叉已定义；Qd/K预载和CTA退休全部Hrhs读者保护alias是核心。',[12]),
231:('MIXED','some','mixed','数学split重复；KS/QS不同消费时点与warp分工使存活期不同，资源Projections不能单独推出。',[10,12]),
232:('MIXED','some','open_semantics','mini数学已定义；row0 owners直接输出及DU0在alias区域的发布与full preload分支需保留。',[11,12]),
234:('MIXED','some','mixed','参数公式、barrier发布、A触发C提前dispatch与非CTA锁步规则共同决定执行；flow已有宏观边。',[8,9,10]),
235:('ESSENTIAL_TEXT','all','open_semantics','A省略Q/Qd但保留完整stage分配；22528+128*VTT是实际交易字节不是资源容量，空m_z不发传输。',[8,11]),
236:('MIXED','some','open_semantics','Sys数学和group循环已给出；共享scratch复用需等待上组publish读完，不能只看顺序名称。',[10,12]),
237:('MIXED','some','mixed','image owner由SaveRows/producer确定，async模式已有；done=global completion、完整inverse loop保留KQImage及非owner无交易仍需契约。',[9,11,12]),
238:('MIXED','some','mixed','最长verb含不可丢的warp局部完成/输出alias/mini K读竞争；数学、stage0视图和后续事件部分重复，不能整段称冗余。',[12]),
239:('MIXED','some','open_semantics','初值/窄化及empty段已有math，Rh CTA发布与warp状态归属为核心补充。',[10,11]),
240:('MIXED','some','mixed','state/aprod/Rh公式重复；mini/full不同K读取时点、无末尾CTA join及仅寄存器summary写决定退休条件。',[12]),
241:('MIXED','some','open_semantics','数学branch已知；DU0 row0 shared发布和Ti预载/CTA join仍是局部核心。',[11,12]),
243:('ESSENTIAL_TEXT','all','open_semantics','wait后fresh perm重读及memory clobber阻止沿用暖读值，不能由数学Perm或普通依赖推出。',[13]),
244:('MIXED','some','mixed','full/mini数值已有math；row0 owners直接输出与DU0发布/alias是动作核心，后续solve no-op事件解释可缩。',[11,12]),
245:('MIXED','some','mixed','唯一group在本域由GroupWidth/HeadsPerPrep和pipe可算出；shared读结束、store仅发射的done界限仍需保留。',[10,12]),
246:('MIXED','some','open_semantics','fresh Perm读与aprod初始化必需，state延后seed已有pipe。',[11,13]),
247:('MIXED','some','mixed','U/Uh/DU数学重复，Qd/K预载及全部Hrhs退休保护alias；mini无访问及事件存在分别属于本地契约和共享语义。',[11,12]),
248:('MIXED','some','mixed','flow已将启动许可和数据就绪分开；实际trigger位于input join后需保留。',[9]),
249:('MIXED','some','open_semantics','C只对z0预取input state且不提前发chunk是本地动作效果；z>0数学composition另有定义。',[9,12]),
250:('MIXED','some','open_semantics','A初值/零值和prime先后已有上游；Rh CTA发布、empty仍seed和每warp归属须明确。',[10,11]),
251:('MIXED','some','open_semantics','Perm排序数学与唯一owner已有上游；thread0排序与门控copy并发及两种subset join是文字内关键执行设计。',[6,10]),
252:('MIXED','some','open_semantics','A初值/aprod见math；input-order前置预取的实际效果与无chunk发射仍需契约。',[9,11]),
254:('MIXED','some','open_semantics','width32 merge数值已给出，Z寄存器与Ti发布CTA join是独有执行条件。',[10]),
255:('MIXED','some','open_semantics','U/DU数值已有，K预载与CTA发布/全部Hrhs读取退休是独占条件。',[12])
}
for n,(cat,core,st,reason,fs) in manual.items():put([n],cat,core,st,reason,fs)
put([202,218,219,220,224,233,253],'STRUCTURABLE','all','existing_structure','自定义join的等待对象、完成点和前后位置可由现有通用wait动作、milestone、handoff.ready/wait_at及pipe表达；PDL原语绑定已在realization，不需发明新事件。F09给出真实短写。',[9])
put([242],'STRUCTURABLE','all','existing_structure','weak的扫描/判定在segments math、Run/Count在schedule已定义；用通用compute:weak和现有空run域即可保留返回前真实setup/wait/classify事件，见F11。',[11])
put([256,259,260,263,264,265,266],'ESSENTIAL_TEXT','all','open_semantics','实现原语或swizzle/基址同余的绑定保证不能从资源shape、容量、protocol名字推出；这是有意开放实现语义，尚不证明需新增DSL字段。',[14])
put([257,258,261],'UNRESOLVED','uncertain','uncertain','已核对source中的contiguous分支和map异常，但共享Contract允许只读非连续视图，flow未枚举可能的输入复制提交；缺匹配PyTorch lowering/版本来确定这些提交、线程分配及失败前缀，不能宣称全路径结构闭合。',[15])
put([262],'EXPLANATION','none','open_semantics','此prefer说明普通packed tile lowering的优先尝试方式，值、十tile容量、依赖均已固定；不是新的强制核心设计。可保留作性能交接。',[14])
put(list(range(267,275)),'EXPLANATION','none','already_present','实现选择的具体理由，未增加角色/搬运/数值定义；对应require/prefer和Design提供约束。bank-conflict及重叠理由有审阅价值而非独占设计事实。',[4,14])
put(list(range(275,283)),'EXPLANATION','none','already_present','phase计算摘要已由target、handoff、math与pipe具体列明；保留必填describe的简短导航功能。',[4])
put([283,285,288,291,293,297,299,301,303,310],'DUPLICATE','none','already_present','已由本节点partition/worktile_order/preassigned、唯一perm映射和分立launch给出逻辑工作归属；物理grid轴名不另定义执行先后。F07列唯一承载字段。',[7])
put([284,287,289,290,295,296,298,300,302,304,306,307,308],'MIXED','some','mixed','必要参数/有效范围/producer所有权定义与已登记的partition、worktile、target、retention/handoff/pipe重复说明混写。已逐节点核对PRE0、排序、A/C空段及reader组合，不能整段删；见F06/F11。',[6,11])
put([286,292,294,305,309],'MIXED','some','mixed','grow-only进程workspace/跨调用复用和缺跨stream协议不可由invocation内global handoff表示；同时有每调用写入及存续共享规则的重复解释。',[16])
put(list(range(311,317)),'EXPLANATION','none','already_present','整体工作组织和主要性能因果的摘要，其提交数、partition、permutation、retention/数据交接与弱衰减rerun已在结构及数学中给出。',[4])
assert set(A)==set(range(1,317)), sorted(set(range(1,317))-set(A))
# v1.1 additions were separately read with every occurrence context.
U.extend(json.loads(l) for l in (R/'data/extract_gdn/added_units_v1_1.jsonl').read_text().splitlines())
assert len(U)==336
put(list(range(317,329)),'EXPLANATION','none','already_present','v1.1补审：Contract步骤名或rewrite标题，仅导航到相邻完整math；padded inputs、merge次序、窄化分叉、segment/perm均已在公式定义，无独占核心事实。',[4])
put(list(range(329,337)),'FORMAL','all','formula_or_predicate','v1.1补审：rewrite.step是正式引用身份，已核对ledger.by及segments方法引用；不是可删除自然语言。',[2])
S='work/extract_gdn/gdn_kernel.cu'
finding(1,'structural_gap',[1,3,4,7,9,11,19,23],
 'nullable/shape/dtype不能表达外部写入前缀、只读背存储、fresh返回与不别名的完整约束；当前必须靠Contract math/meaning。',
 'CoreWritten是副作用而非第三个返回值；Out不alias core，输入初态不改，prefix外字节保持。',
 '删掉output_and_side_effect的存储条件后，返回core的视图或覆盖其整个capacity均可满足剩余shape/dtype/nullable及Out数值公式。',
 '已有output.CoreWritten:{shape:[element:"T*HV*ValueDim"],dtype:bf16,nullable:true}只声明逻辑输出；compute.math可紧写CoreWritten=core_attn_out[:T*HV*ValueDim]=flatten(Out)，但仍需readonly/fresh/no-alias文字。排查DOC1 output/nullable、DOC5 resource.view：后者仅worker局部存储，不能绑定外部背存储。',
 'open_semantics',ev([1,3])+[RULE+'1.contract.algorithm_requirement.md:40','work/extract_gdn/reference.py:41','work/extract_gdn/input.py:248'])
finding(2,'style',[2,28,29,30,31,32,55,56,57,60,61,65,66,69,70,73,74,75,77,79,80,81,82,84,89,90,92,93,94,95,96,97,98,99,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117]+list(range(329,337)),
 '公式、求值次序、正式引用和计量关系是有意开放的数学表示；不可按字符串面积判为语言缺口或冗余。',
 '保留FP32 carried state/BF16 operands、U到Uh和DU各自窄化、交错KS/QS reduction、近似exp、16/32/64层次inverse不同乘法括号、实际foreign/padded行。',
 '把DU改成bf16(Dl*Uh)、把层次rounded inverse改成理想inv(Sys)、或令Q/K按请求末尾零填，均改变原始数值路径。',
 '现有rewrite.math中的U=mm32(Ti,Hrhs); Uh=bf16(U); DU=bf16(Dl*U)及arithmetic.accumulate已足以表达；step/by应保留引用身份。仅去掉共享默认和相同公式的解释。',
 'none',[RULE+'3.design.math_reshape.md:229',RULE+'1.contract.algorithm_requirement.md:175',S+':2600',S+':2640'])
finding(3,'source_limitation',[33,34,35,44,56,57,58,59,60,61,83,86,87,102,103,106,109,110,111,113,114,116,117],
 '已知阈值与未知验收器/全域误差界必须分开；source编译表达式也不等于已固定逐位FP32 contraction。',
 'atol=.01/rtol=.05/max_mismatch_fraction=0、对应输出及未定义比较器；__expf、正rho溢出/非有限零权项和G>=3 scalar composition没有可用全域界。',
 '只留阈值简写会允许套用默认相对基准/NaN策略；删近似限制会把G>2标为精确segment scan，或忽略0*nonfinite的数值影响。',
 '继续用golden_reference.tolerance各输出的Source record、arithmetic.approx中的unknown:；不新增测试标准，不把误差缺口扩成新的设计。',
 'unresolved',ev([33,58,59])+[RULE+'1.contract.algorithm_requirement.md:281',S+':2234',S+':2269','work/extract_gdn/source.py:30'],confidence='high')
finding(4,'style',list(range(5,28))+[36,37,38,39,40,41,42,43,45,46,47,48,62,63,64,67,68,71,72,76,78,85,88,91,100]+list(range(118,170))+list(range(267,283))+list(range(311,329)),
 '摘要、释义、标题和导航没有独占核心；必填字段可短写，不能把全类面积算作可删量。',
 '设计事实唯一来源是相邻math、partition/flow及resource/pipe；reason保留具体代价解释。',
 '删除非必填导航不新增不同设计，但删除mandatory name/meaning会破坏文档合法性，因此候选是局部短写而非自动清空。',
 'rewrite.name可将Hierarchical rounded inverse with different merge order at the final level短写为Hierarchical rounded inverse；实际w=32/w=64求值留在math。',
 'none',[RULE+'0.common.md:29',RULE+'3.design.math_reshape.md:61',RULE+'6.realization.md:84'])
finding(5,'structural_gap',[40,49,50,51,52,53,54],
 '选择谓词可以紧写，但六节点独立when复写共享派生式；LC的process-static来源超出纯输入shape，且没有跨节点predicate共享入口。',
 '四个非segmented节点区分direct_qk/ordered，两个segmented节点区分排序；segmented_ordered严格G=2,VTT=16。',
 '按父when隐式约束子when会使分支域错误；用当前设备L2替代LC缓存可能选择另一条路径；删除G/VTT定义使分支绑定失效。',
 '现有when可保留Select exactly when ...及本地Let定义；DOC0的from_parent仅覆盖三个完整Design区，不能共享when或将任意兄弟谓词引入。没有现成跨节点公共predicate模板。',
 'representation_cost',ev([49,53,54])+[RULE+'0.common.md:156',S+':3162',S+':3220',S+':3316'])
finding(6,'existing_structure',[104,108,119,137,151,284,287,289,290,295,296,298,300,302,304,306,307,308],
 '调度长strategy混合真正的参数/owner定义和已登记的工作划分；后者可引用现有partition、iteration、target和retention/handoff，前者可以公式短写，不能全删。',
 'SU/Active/Shared/HeadsPerPrep、VTT/NSTAGE/PRE0、唯一Perm/image生产者、有效输出行数和rank→request绑定。',
 '删Active与HeadRows/SaveRows定义则空slot产物范围未定；删PRE0条件会允许首chunk错误等待cache；仅删No chunk is independent不改变partition。',
 '保留现有partition:{request:{size:N,tile:1},value_head:{size:HV,tile:1},value:{size:ValueDim,tile:VTT}}及worktile.iteration.Kc.direction:ascending。短strategy仅定义VTT=V; NSTAGE=...; PRE0=(VTT=16); OutputRows=Length[n]; CoreRows=...; FinalRows=...。cached_ordered可用已有rewrite.math定义perm[r]=stable_descending(c_n)[r]，symbol.perm:{shape:[request:N],dtype:int32,retention:"keep_value invocation"}，不再用binding把排列数学下放strategy。',
 'none',ev([284,287,289,300])+[RULE+'4.design.work_schedule.md:89',RULE+'3.design.math_reshape.md:183'])
finding(7,'redundancy',[283,285,288,291,293,297,299,301,303,310],
 '物理grid轴翻译和静态分配解释不另定义工作归属；逻辑排列和预分配已唯一写在结构里，排序节点另有perm数学和pipe等待。',
 '原请求/排名及segment/head/value的逻辑编号；没有dynamic refill，没有CTA时间先后保证。',
 '这些段落删除后，现有worktile_order/preassigned及perm映射仍确定同一逻辑工作序列；改CUDA grid轴表达式只要保持这个映射属于lowering。',
 '非segmented: worktile_order:[{by:request,direction:ascending},{by:value_head,direction:ascending},{by:value,direction:ascending}]; worker_assignment:{preassigned:{start:worker_id,count:1},refill:none,exit:assigned_done}。排序节点request改rank/request_rank，segmented加入segment；各节点已经具备这些字段。',
 'none',ev([283,291,293,297,299,301,303,310])+[RULE+'4.design.work_schedule.md:202',RULE+'4.design.work_schedule.md:308'])
finding(8,'structural_gap',[105,107,112,115,175,180,185,189,196,197,200,204,205,206,217,234,235],
 'Stages.view/slot_count表达容量和alias，但prime/advance的发射、到达、parity和累计byte expectation仍是opaque动作协议；PRE0分两时段填同一交易尤为明显。',
 '领先NSTAGE-1轮；先acquire并CTA退役全部旧读者再发未来轮；A仅22528+128*VTT交易字节但保留完整StageBytes空间；PRE0不二次arrival。',
 '保留同样slot/view和pipe而将prime.done解释为全部到达会丢重叠；把PRE0后补系数再arrival则变更barrier代际，可能错误完成或死锁。',
 '已有domain:c:[0,c_n]、Stages:{slot_count:NSTAGE,view:{"Kc(c)":{slot_index:"c%NSTAGE",...}}}和execution_mode/issue/done可显式拆load/acquire。可是现有prime/advance把批量发射、barrier等待、CTA join合一，完整等价拆分还需定义负载集合/字节协议；只造load:Stages名称不足。判断为表示成本及有意开放原语契约，不声称缺少第四事件种类。',
 'representation_cost',ev([175,185,200,235])+[RULE+'5.design.tile_pipeline.md:100',RULE+'5.design.tile_pipeline.md:250',S+':2011',S+':2094',S+':2248',S+':2331'])
finding(9,'existing_structure',[175,179,184,187,192,196,202,203,207,215,218,219,220,224,225,233,237,248,249,252,253,256,259,264],
 '宏观PDL启动许可/数据可见性已用flow与milestone表达；七个纯join动作可使用现有通用wait，原语绑定保留在Realization。',
 'pre-wait input/metadata与post-wait prepared/perm不可混；A启动C可早于A完成，SegA/SegB必须等kernel_done。',
 '若只删join而不换wait+milestone会提前用系数；若把start_after改kernel_done会消除既定重叠。',
 '以root为例：flow.handoff.Ti:{producer:prep,consumer:[chain],storage:global,ready:prep.kernel_done,wait_at:chain.prep_ready}; chain.milestone.prep_ready:["Main/wait:Ti@done"]; Main.pipe.prologue:["setup:Resources","early_inputs:Inputs","wait:Ti","prime:Stages","seed:R"]。其余既有handoff仍指同一prep_ready，删除join_prep自定义verb；现有realization.require将prepared-data wait绑定griddepcontrol.wait。A/C同理wait:Ti或wait:SegA并保留所有原handoff。',
 'none',ev([202,218,220,224,233,253])+[RULE+'5.design.tile_pipeline.md:89',RULE+'4.design.work_schedule.md:308',RULE+'4.design.work_schedule.md:330',S+':1984',S+':2224'])
finding(10,'structural_gap',[171,172,174,176,178,179,182,183,191,192,193,194,195,196,197,198,199,201,208,209,210,212,213,216,217,221,223,226,229,231,234,236,239,245,250,251,254],
 '单个Main(warp[0:8])把scan/copy子集协作与数学归约执行方式放进verb；同一粗动作之间只有role级issue/done，不能读出其内部128/64线程joins和重叠。',
 '门控copy warps0/1与scan warps4/5先join，其他warps独立等QK；ratio8分三pass扫描；F跨head保留；各merge CTA发布。',
 '用一次全CTA wait替代subset join仍满足当前Main顺序/资源结构，却把scan等待Q/K；把F在首head后释放会重复Gram或者丢后续Qd。',
 '现有info.member和waits_for能拆真实独立角色，理论上无需新事件；但一个source warp在copy、Gram与merge阶段改变作用，完整拆分要协调相同物理warp的复用并定义真实发布/退休动作。未得到更短且完整等价的仅结构写法，不能把新增未定义动作名当解决方案。现有结构候选保留Main及精确契约，只删其重复math部分。',
 'representation_cost',ev([183,198,212,251])+[RULE+'5.design.tile_pipeline.md:44',RULE+'5.design.tile_pipeline.md:98',RULE+'5.design.tile_pipeline.md:447',S+':268',S+':327',S+':389',S+':672'])
finding(11,'existing_structure',[106,111,114,194,199,201,203,209,214,222,227,232,235,237,239,241,242,244,246,247,250,252,284,287,289,290,295,296,298,300,302,304,306,307,308],
 '空slot、空segment、weak提前返回及无payload但动作存在已有repeat域/公式与通用no-op规则；不需要用文本删除提交或变更角色。',
 'inactive prep仍enter/trigger；空A仍seed并发布EntryA/1；weak C保留setup/wait/test，run=0仅移除后续主体；mini solve保留事件但无访问。',
 '把m_z=0等同于不提交A会漏SegA=1并改变weak；把weak z>0当整个C kernel不存在会漏已执行setup/wait。',
 '保留现有pass_c.domain:{run:[0,Run],cl:[0,Count]}与Run=0 if weak and z>0 else 1；将自定义classify:weak替为通用compute:weak（segments math已定义扫描），prologue保留setup/early_inputs/join_a/compute，steady仍repeat:run。pass_a的publish_transfer仍在cl循环外epilogue。generic no-op动作仍有issue/done见DOC5，不另写取消事件。',
 'none',ev([106,114,242,300,304])+[RULE+'5.design.tile_pipeline.md:161',S+':233',S+':2234',S+':2809'])
finding(12,'structural_gap',[171,172,177,181,182,185,188,193,203,205,206,207,209,210,211,214,216,221,222,223,226,227,228,230,231,232,236,237,238,239,240,241,244,245,247,249,255],
 'RhsOutput/Stages.FinalState的alias已结构化；真正的全读者退休及warp内部先后仍由契约承载。合并finish_chunk既不能制造CTA锁步，也不能修复source末尾mini竞态。',
 'full solve先预载Qd/K后CTA join，读完Hrhs才overlay输出；nonfinal loop-top join退役；final mini stage0写可能早于其他warp K行读取。VTT16的segmented_ordered不含该mini缺口。',
 '保留相同view与Main.pipe，把final staging写视作同一动作完成之后跨warp统一开始，会无依据补出barrier并掩盖源码竞态；只等某warp读完也不能覆盖所有读者。',
 '现有resource.RhsOutput.view:{Hrhs:{},Out:{},DU0:{part:{token:[0,1]}}}与Stages.view.FinalState:{slot_index:0,part:{element:[0,"4*128*(VTT+1)"]},dtype:fp32}已表达alias。通用release+waits_for.reuse可表达有来源的全部读者退休，但Main[member]的member是CTA成员，不是warp0..7；完整warp级事件需拆roles并保留原语契约。source缺少的final-mini边不能作为压缩时新增的release。',
 'representation_cost',ev([181,228,238])+[RULE+'5.design.tile_pipeline.md:298',RULE+'5.design.tile_pipeline.md:355',S+':2499',S+':2659',S+':2754',S+':2820'],confidence='high')
finding(13,'structural_gap',[170,173,187,190,243,246],
 '投机读取旧/未初始化perm字节用于cache warming，与读取本调用数学Perm必须区分；现有load默认值就绪与数学命名不直接表达这种纯地址合法的读取。',
 '暖读地址在界内，读值仅丢弃，post-wait重新读取并用memory clobber避免CSE；该前置工作只在排序分支。',
 '若把暖读当load:Perm，会无据要求初始化或等prep；若用暖读结果作为n，会读错request或越界。',
 '可继续用warm:Indices的现有自定义契约明确raw bytes/no control/no address use，再用join/resolve读取Perm；通用resource没有global raw-allocation身份。本地原语开放语义可以承载，不等于已由结构表达；若要求结构独立解析需区分raw storage与数学值，尚未证明应扩语言。',
 'open_semantics',ev([170,173,187,243])+[RULE+'4.design.work_schedule.md:362',S+':2041'])
finding(14,'structural_gap',[256,259,260,262,263,264,265,266,267,268,269,270,271,272,273,274],
 'PDL/TMA/mbarrier原语保证、swizzle相位同余属于有意开放的实现绑定；资源形状与协议名不足以推出地址解释和完成保证。',
 '四个8192B SWIZZLE_128B图像、prep/chain基址mod1024一致、VTT64按实际目标相位解码；mbarrier每generation一次期望及parity acquire。',
 '相同shape/dtype/字节数采用不匹配swizzle相位会改变读出逻辑值；把griddepcontrol.wait换为仅launch顺序会改变重叠/可见性。',
 '现有realization.require:{item:["design.tile_pipeline::prep.KQImage","design.tile_pipeline::chain.Stages"],choice:"...",reason:"..."}是正确承载位置，item连接已有对象，但choice仍需文字语义。preference half-column layout可留给Coder比较，不新增布局DSL。',
 'open_semantics',ev([256,259,260,263,265,266])+[RULE+'6.realization.md:76',S+':1918',S+':2094',S+':2564'])
finding(15,'source_limitation',[1,3,33,34,35,44,102,103,109,117,257,258,261,268],
 '非连续core复制到临时张量不回写是已核对的来源偏差；非连续只读输入materialization在Realization被允许，但实际前置复制提交未在六节点flow中展开，尚不能确定完整launch覆盖。',
 'dense输入条件、保留contiguous storage offset、不重归一化，tensor-map编码失败无fallback；caller-visible CoreWritten契约不能靠临时副本满足。',
 '忽略core分支会把未回写称符合Contract；忽略readonly contiguous化会把有前置复制成本的调用算作仅2/3个提交。',
 '既有flow.launch/phase/handoff可登记实际输入转换（不是新字段需求），但本次源码没有固定PyTorch lowering与执行映射证据，不能凭空写出其kernel数或CTA分配。暂将三项dense require字段UNRESOLVED；保留known core偏差，未修改schema或源码。',
 'unresolved',ev([44,103,257,258,261])+[S+':3091',S+':3117',S+':3066','work/extract_gdn/reference.py:41','work/extract_gdn/input.py:248'],confidence='high')
finding(16,'structural_gap',[44,49,50,51,52,286,292,294,305,309],
 'process-static grow-only workspace与L2缓存跨调用持有身份，现有invocation内handoff和worker局部resource不能表达分配容量、设备键、跨调用复用顺序。',
 'workspace不足或设备改变才分配；direct-QK仍保留未用图像容量；perm在尾pad，seg summaries随后；仅同stream顺序可支持复用，无跨stream锁/事件；L2缓存无设备键。',
 '删掉这些文字后，每调用重分配、跨stream复用同一workspace、或查询本设备L2均满足相同raw flow/shape，但有不同主要开销、并发冲突或dispatch。',
 '排查DOC3 retention:keep_value invocation、DOC4 handoff.storage:global/initialize及DOC5 resource生命周期：都不足以命名process跨调用存储/host cache。文字能保留限制，若目标要求结构承担该核心，需要跨调用资源与复用身份候选；不把global resource的任意space字符串当合法跨调用作用域。',
 'missing_field_or_event',ev([286,292,294,305,309])+[RULE+'0.common.md:43',RULE+'3.design.math_reshape.md:116',RULE+'4.design.work_schedule.md:344',S+':3258',S+':3267',S+':3316'],confidence='high')
finding(17,'noncore',[186,211,222],
 'direct-QK的空bulk-group wait和若干末尾实现解释没有独占核心；可做保守删减，不能连同真实float4预载流水一起删除。',
 'final_store前读下一float4再发当前store仍是主要延迟隐藏选择，保留；空group wait无数据含义。',
 '删空wait说明不允许跳过真实image drain；删末尾无next-worktile说明后，preassigned.count=1、finite prime域与kernel退休规则仍保留。',
 '保留已有final_store动作/原语行为，仅缩写说明；该低阶动作是否从源码层移除不属于本审计。',
 'none',ev([186,211,222])+[RULE+'5.design.tile_pipeline.md:381',S+':2248',S+':2840'])
# All repeated local definitions relevant to reuse were manually reviewed above;
# this list records the selected repeated mechanism, not a keyword-based category rule.
repeat_ns=[55,56,58,60,103,105,110,112,116,117,171,172,176,178,179,183,185,188,191,192,194,197,198,201,203,204,205,206,207,211,212,213,214,217,221,222,223,225,227,231,235,236,237,238,242,244,247,248,254,256,257,258,259,260,261,262,263,264,266,268,269,270,272,273,274,275,276,277,281,289,291,292,293,297,303]
finding(18,'structural_gap',repeat_ns,
 '重复精确文本大多为跨文件独立声明；现有整区from_parent不能共享一个rewrite、一个verb或资源recipe，而六节点在每个Design区均有本地差异。',
 '所有分支都要保持相同数值与原语契约，但PRE0、perm、direct-QK、segmentation及G=2缩域改变区内部分内容。',
 '直接令cached_ordered整区继承root会丢perm或带入PRE0；仅因文本相同删除子定义也没有语义引用可解析。',
 'DOC0唯一已有跨文件候选是design.math_reshape:{from_parent:root}等完整区；本树对应区并非相同。YAML anchor只能文档内复用：segmented的mapping及segmented_ordered的mapping/schedule/warm已有alias。进一步局部跨节点复用属于表示成本候选，未作为当前可删量。',
 'representation_cost',[RULE+'0.common.md:138','inputs/extract_gdn/root.yaml:172','inputs/extract_gdn/cached_ordered.yaml:55','inputs/extract_gdn/segmented_ordered.yaml:324'])
for n in repeat_ns:
 A[n]['finding_ids'].append('extract_gdn-F18')
# Make finding membership exhaustive and reciprocal for all already-reviewed annotations.
for f in F:
 linked=[a['id'] for a in A.values() if f['id'] in a['finding_ids']]
 f['unit_ids']=list(dict.fromkeys(f['unit_ids']+linked))
# Conservative local exact replacements; no structural rewrites counted as savings.
def cut(n,quote,replacement,reason,occ='all'):
 assert U[n-1]['text'].count(quote)==1,(n,quote)
 assert len(replacement)<=len(quote)
 C.append(dict(unit_id=U[n-1]['id'],quote=quote,replacement=replacement,kind='redundancy',reason=reason,evidence=ev([n]),applies_to_occurrences=occ))
cut(2,'The residual uses the decayed incoming state; readout uses the updated state.','','同字段delta含a*S，Out读取S(l+1)，精确公式已保留这一关系。')
cut(2,'No state checkpoint is narrowed to BF16.','','S递推及输出转换已精确定义；Contract默认不引入未写cast。')
cut(3,'Inputs are finite; floating ranges additionally constrain g and beta.','','DOC1浮点输入默认finite，g/beta.range仍在。')
cut(4,'Floating arithmetic is FP32; no intermediate BF16 narrowing is part of the reference requirement.','','DOC1默认FP32/RNE及显式cast规则，参考公式没有中间BF16 cast。')
cut(53,'In the shared Contract domain these conditions are equivalent to segmented execution with a nonnull perm: the host average floor((T+63*N)/(64*N)) is 32, giving G=2 and VTT=16. Segmentation requires the pre-segmentation 16-wide grid to fit SM; the final chain_blocks=(ValueDim/VTT)*HV*N also fits, so sorting requires N>=8. With T<=16384, this excludes N>8 and the wide segmented arm\'s average>=192 condition.','In this Contract domain the selected path has G=2 and VTT=16.','精确选择谓词和G/VTT仍在；删去等价性证明，不删用于schedule绑定的取值。')
cut(54,"A is a metadata estimate, not a lower bound on an individual request's chunk count.",'','A公式仅依赖T/N，不对每请求长度作保证；各c_n继续来自Contract。')
for n in [283,285,288,291,293,297,299,301,303,310]:
 cut(n,U[n-1]['text'],'','可省略strategy的逻辑归属由本节点partition/worktile_order/preassigned、perm公式与flow定义，详见F07。')
for n in [287,295,302,308]:cut(n,'No chunk is an independent chain worktile.','','partition不含chunk，worktile.iteration与math c循环已给出局部串行遍历。')
cut(238,"Keeping this suffix in one action does not order those reads before other warps' writes.",'','保留第一句无CTA join及紧前一句source无read-retirement保证，已排除合并动作制造同步。')
cut(228,'The source supplies no retirement guarantee; combining this suffix in one action does not add one.','The source supplies no retirement guarantee.','保留无读取退休保证和精确final-mini竞争条件；删解释性重复。')
cut(211,'No worktile follows in this CTA, and no speculative future chunk remains.','','每CTA preassigned.count=1及prime/advance的c<c_n条件已约束；保留float4流水和无store-drain事实。')
cut(222,'No next worktile or speculative future chunk remains; retirement follows kernel completion.','','count=1、未来chunk域及DOC5 worker结束安全回收给出同义规则；保留条件无读写和float4顺序。')
cut(272,'; it does not change the declared values or add an accuracy guarantee.','.','Realization不能改变上游值/误差标准是DOC6共享规则；保留具体bank-conflict理由。')
cut(327,U[326]['text'],'Hierarchical rounded inverse','标题可以短写；w32/w64不同乘法括号与舍入的正式定义仍在math。')
notes={
'root':'v11，Contract共享提供者；PRE0仅VTT16，2/3stage按VTT/A；cached图像first chunk仍写但可无chain读者；读完所有math/flow/resources/verbs/realization。',
'cached_ordered':'v2，perm在prep唯一owner生成、暖读旧字节后wait重读；无PRE0，state延至prime后；全部Design本地。',
'direct_qk':'v2，无Q/K keep_value/handoff/save_image，保留global allocation区域；VTT32/64，prewait只load输入state；全部Design本地。',
'direct_qk_ordered':'v2，独立perm math/ledger，wait与warm/resolve合并动作；state/prime均postwait；VTT32/64；全部Design本地。',
'segmented':'v3，prep→A→C，G/VTT两臂，A只K/Ti/vector/V交易；空A发布summary，weak C run=0仍执行前缀；最终mini race局限保留。',
'segmented_ordered':'v3，G=2,VTT16,N=8缩域；HeadsPerPrep<=2；Perm两pass消费，SegB1无人读，C无tile_reduction，warm alias读两个phase fresh资源；无mini分支。'}
limits=[
 '本文完成全部336精确组和6节点有效Design/共享Contract/Realization/dispatch全文；来源复核围绕关键机制，不宣称逐条审计整个3502行CUDA或进行GPU验证。',
 '来源均固定meta_data.source commit 6e4ca91bd91d998cac99e8dadd83838e153c679d，工作副本由git show取得；source.py是solution.py的逐字副本。所有代码行号指该版本，未借用当前tree行号。',
 'root hardware_specification按该extract范围保留null占位；没有锁定GPU/compiler/PyTorch lowering实证，本研究不将approved解释为GPU或全域数值正确。',
 'UNRESOLVED三组覆盖六节点的dense-input materialization：源码明确contiguous调用，实际前置计算提交/线程划分及map异常前缀需匹配库版本进一步核对。',
 'SOURCE_LIMIT保留外部比较器未给定义、__expf/combined error无全域界、非连续core不回写和final-mini stage0读取退休缺口；没有添加源码不存在的同步修复。',
 '当前实现入口没有显式capture/instantiate/replay分支。按DOC0录图固定host选择/按值绑定、设备读取使用本次内容；本轮没有独立图回放/跨stream运行证据，不能从同stream调用推并发或replay安全。',
 'v1.1 scalars全量核对后，新增compute/rewrite.name和step已覆盖；剩余字符串为源路径、引用、标识、dtype/space/member、shape/域/retention表达式，未发现遗漏的独占正文。',
 '候选仅逐条局部等价理由，没有独立短稿全量复审或应用后schema检查；保守候选量不等于已认证可删字数，MIXED字段面积不能当作冗余率。',
 'F10/F12的role拆分尚未提供完整更短等价稿，故是representation_cost而非证明必须新增成员事件；F16为缺作用域/身份的语言候选，也未正式扩语法。']
D=dict(operator='extract_gdn',prompt_version='v1.1',unit_annotations=[A[n] for n in sorted(A)],findings=F,removal_candidates=C,node_review=[dict(node=n,reviewed=True,effective_binding_checked=True,notes=v) for n,v in notes.items()],limits=limits)
(R/'annotations/extract_gdn.json').write_text(json.dumps(D,ensure_ascii=False,indent=2)+'\n')
print('units',len(A),'findings',len(F),'candidates',len(C));print(collections.Counter(a['category'] for a in A.values()))
