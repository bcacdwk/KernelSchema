import json,pathlib,collections,re
B=pathlib.Path('schema_extract/archive/20261109-ten-operator-text-audit'); OP='template_fa4_decode'; W=B/'work'/OP
U=[json.loads(x) for x in (B/'data'/OP/'units.jsonl').read_text().splitlines()]
A={}; F=[]; C=[]
def fid(n):return f'{OP}-F{n:02d}'
def ev(i):return list(dict.fromkeys(f"inputs/{OP}/{o['node']+'.yaml' if o['node']!='dispatch.md' else 'dispatch.md'}:{o['line']}" for o in U[i]['occurrences']))
def put(ids,cat,core,structure,reason,fs):
 for i in ids:
  assert i not in A,i
  A[i]={'id':U[i]['id'],'category':cat,'core':core,'structure':structure,'reason':reason,'evidence':ev(i),'finding_ids':[fid(f) for f in fs],'contexts_checked':True,'occurrence_overrides':[]}
def find(n,kind,ids,claim,core,counter,candidate,gap='none',extra=[],confidence='high'):
 F.append({'id':fid(n),'kind':kind,'unit_ids':[U[i]['id'] for i in sorted(set(ids))],'claim':claim,'core_fact':core,'deletion_counterexample':counter,'existing_structural_candidate':candidate,'gap_kind':gap,'evidence':list(dict.fromkeys([e for i in ids for e in ev(i)]+extra)),'confidence':confidence,'nodes':sorted({o['node'] for i in ids for o in U[i]['occurrences']})})
def cand(i,quote,replacement,reason,kind='redundancy',nodes=None,extra=[]):
 assert U[i]['text'].count(quote)==1,(i,quote)
 occ='all' if nodes is None else [o['occurrence_id'] for o in U[i]['occurrences'] if o['node'] in nodes]
 C.append({'unit_id':U[i]['id'],'quote':quote,'replacement':replacement,'kind':kind,'reason':reason,'evidence':ev(i)+extra,'applies_to_occurrences':occ})
# All classifications below are manually selected after reading the exact text and each local binding.
put([0,1],'MIXED','some','mixed','公式定义注意力、页面解释和标量类型；还含外部别名/计时边界叙述。公式不是语言缺口，别名和调用边界不能当纯解释删掉。',[13,14])
put([2,4,5,6,7,9,11,12,13,16,17,18,19,20,21,23,25,26],'EXPLANATION','none','already_present','必填对象短释义；结合 root shape/dtype、范围及 compute 的索引/公式读取，没有独占的计算或执行决定。字段不能留空。',[1])
put([3],'ESSENTIAL_TEXT','all','open_semantics','首次隔离使用前清零是 workspace 生命周期前提；shape/uint8 不保证初值，当前 handoff 也未定义跨调用旧内容。',[14])
put([8,10],'EXPLANATION','none','already_present','必传 None 的单值类型、无 default、compute 中禁用条件共同确定接口；短释义说明禁用功能，不能据此扩出可省略参数。',[13])
put([14],'EXPLANATION','none','already_present','固定 0.0625 与 compute.score 已完整给出缩放数值；captured quantization scales 是参数来历的接口解释，不定义额外量化计算。',[13])
put([15,22,24],'EXPLANATION','none','already_present','输入输出别名/覆盖在 root compute 与 out.range 有唯一明确约束；本处是必填接口说明，不以同 shape 推断别名。',[13])
put([27],'DUPLICATE','none','already_present','python_none 已在 root compute 定义为仅含 None 的单值类型；range 不增加有效值，两个使用输入均核对无 default。',[13])
put([28],'MIXED','some','mixed','Any BF16 bit pattern 显式放宽默认有限域，必须保留；不读旧值与 out.meaning 和输出公式重叠。',[13])
put([29,30,31,34],'FORMAL','all','formula_or_predicate','分别为离散输入有效域、合法页索引界、长度界和逐输出验收谓词；保留其正式关系，不以字符串载体判成叙述缺口。',[13])
put([32,33],'EXPLANATION','none','already_present','必填 compute 标题；该步骤正文承担定义。captured 标题本身不能证明所有 eager 初始化/重试都在计划内。',[1,14])
put(list(range(35,48)),'EXPLANATION','none','already_present','中文导航/树/节点摘要；已逐一对照 parent、when、有效数学与 pipeline，精确参数以 YAML 为准。',[1])
put([48,52],'FORMAL','all','formula_or_predicate','q 长度与单 KV block 的充要选择谓词；分别独立在共享契约域求值。',[2])
put([49,50,51,53,54],'MIXED','some','mixed','完整选择谓词是核心正式关系，末尾同设计归类句仅为解释；没有把 parent 条件叠加进 child。',[2])
put([55,56,57,58,59,60,61],'ESSENTIAL_TEXT','all','open_semantics','硬件执行层级、存储作用域和异步完成保证的开放定义；名称/space 字符串不单独定义这些能力，wiki 尚未完成绑定。',[12])
put([62],'MIXED','some','mixed','保留已选工具链版本及编译器/驱动未记录的边界；版本身份不可改成泛称，也不能据 PyTorch cu130 推断驱动。',[12])
put([63],'ESSENTIAL_TEXT','all','open_semantics','已选 CuTe DSL 及版本身份，不是自然语言优化理由；硬件 arch 并不决定实现栈。',[12])
put([64,69],'MIXED','some','mixed','归约分组/遍历含核心求值与计算分配事实，同时保留 FP32 contraction 未证实限制；与 combine 动作说明部分重复。',[8,9])
put([65],'MIXED','some','mixed','每个 V slice 的 hi 后 lo 累加及四组 denominator 分组细化数学求值；FP32 与上游公式重复，但不能整段删除。',[9])
put([72],'MIXED','some','mixed','每个 V slice 的 hi→lo 累加与 Lbase 两项分别缩放后相加是求值事实；FP32 contraction 未确认限制必须保留。',[9])
put([66],'MIXED','some','mixed','XOR 分组与 BF16 分贡献后无固定加法次序是核心；内部舍入/subnormal 尚未验证，不能以 bf16 dtype 推定。',[7,9])
put([67,68,71,73],'MIXED','some','open_semantics','给出 fast-exp2/近似倒数路径来源并明确局部误差界未知；公式使用该名称不等于定义其近似误差。',[9])
put([70],'FORMAL','all','formula_or_predicate','显式引用 step exponent 的同一近似定义，避免把跨步骤精度例外当隐式全局默认。',[9])
put([74,78,83,85,86,91],'FORMAL','none','formula_or_predicate','ledger.by 是 rewrite.step 的正式引用；同名文字在不同节点解析到各自本地步骤，不能按普通名词删除。',[1])
put([75,76,80,87,88,90],'FORMAL','none','formula_or_predicate','有单位和作用域的计算量、流量或深度关系；设计决策由 rewrite 决定，账本固定比较结论，非纯散文。',[1])
put([77,82,84],'EXPLANATION','none','already_present','误差账本概括本地 cast、近似和归约重组，独占误差界未在这里给出；保留作为量化/误差导航。',[1,9])
put([79,81,89,92,93,94,95],'EXPLANATION','none','already_present','账本对象标题或必要 baseline 说明，不定义新计算；并不因没有独占核心而能把必填字段删空。',[1])
put([96],'MIXED','some','mixed','正式 Q/K/V 索引及 mm32 关系与 speculative 首页访问说明并存；后者说明真实额外访问域，不能当掩码计算的同义句删除。',[3,13])
put([97,99,100,101,102,103,104,105],'FORMAL','all','formula_or_predicate','定义求值、索引、FP8/BF16 舍入点、递推与空分支的正式关系；英文条件和 empty 分情形属于数学正文。',[9,10,13])
put([98,106],'MIXED','some','mixed','正式物化/归约关系之外明确 inactive Opartial 保留且必须有限；乘零不能掩掉 NaN/Inf，跨调用旧值约束须保留。',[14])
put(list(range(107,144)),'EXPLANATION','none','already_present','必填数学接口短释义。已核对各处公式、shape/dtype、retention/binding；唯一决定在相应定义，不通过同文字跨节点隐式继承。',[1])
put(list(range(144,153)),'EXPLANATION','none','already_present','必填 rewrite.name 概括相应数学步骤；核心关系、舍入和保留方式在其正文与 symbol。',[1])
put(list(range(153,159)),'FORMAL','all','formula_or_predicate','rewrite.step 是正式步骤身份，被 ledger/normalize 引用；不是可删标题。',[1])
put([159,160,161,162,164,165],'EXPLANATION','none','already_present','必填 thesis/任务摘要，所述分块、FP8 分解、角色、寄存器或空 parity 在相应本地结构及动作定义保留。',[1])
put([163],'MIXED','uncertain','mixed','大部分是已登记角色/复用的摘要；q1 无寄存器再分配是来源实现条件，其独立主要性能影响未证实；不把省略 register 当作性能证明。',[1,11])
put([166,188,192,199,201,223],'MIXED','some','mixed','Startup 合并首表预取/初始化可见性、库容量及 stage/slot 公式；q4 还含条件寄存器路径、compact 的设备 WorkSplit。含冗余结构解释，但长段有核心定义。',[3,4,11,12])
put([167],'MIXED','some','mixed','BF16 reduce-add 与 async-proxy acquire 绑定必须保留；对 member 0 zero_done 的跨 role 等待目前在文字中，可用现有条件等待显式化。',[6,11])
put([168,200],'ESSENTIAL_TEXT','all','open_semantics','retire 动作定义真实 TMEM 提前释放；reuse 边给退役前驱，但不替代自定义 retire 的资源副作用/空 split 含义。',[3,10])
put([169,216,218,227,228],'MIXED','some','mixed','每 slice 独立等待、操作数读取后归还及 hi/lo 子积映射是核心；块级 pipe 只表达部分边，文字还重复已有数学/turn/空域关系。',[3,4,9,10])
put([170],'MIXED','some','mixed','M/Lpartial 写出由上游 target 与显式前驱说明，四项 FinalSum 局部归约仍是 store:Lpartial 的复合动作含义。',[3])
put([171,189,191,208],'ESSENTIAL_TEXT','all','open_semantics','read/capture 自定义契约规定 parity 读取顺序、源槽返还和寄存器捕获；N_r=1 的空 parity 仍需消费，单个动作名无法替代。',[3,10])
put([172,180],'DUPLICATE','none','already_present','FinalSum 发布→Score wait→Correction capture 的全部跨 role 关系已有显式边和 serial wait，rendezvous API 本身未证额外核心约束。',[1])
put([173,174,194],'MIXED','some','mixed','包含操作数/页面捕获及 slice 协议引用，部分 Q 先于 K/等待关系已有 pipe；不能省掉 per-slice 原语接口。',[3,4])
put([175,177,181,210],'MIXED','some','mixed','复合发布中有 FinalSum 四项、Ostage 的转换/组合及空 payload no-op；提前 FinalMaximum 与存储别名位置部分已由 pipe/view 表达。',[3,10])
put([176,217,226,230,232],'MIXED','some','mixed','必须定义初始 Statistic/Y/empty sum/Lbase 与 fence；空域不产生 zero 动作由 domain/repeat 已定，可局部压缩该重复说明。',[3,10])
put([178,197,215],'ESSENTIAL_TEXT','all','open_semantics','操作数读完的 release 与目标结果 commit 是不同保证，已有 @done 名称本身不定义源读取退休；绑定实际原语，不能用 issue 或结果完成替换。',[3,4])
put([179],'DUPLICATE','none','already_present','直接路径仅一次 m 更新；Correction/read:m 与 FinalSum→Normalization 顺序保证 Maximum 使用先后，masked maximum 本身在 exponent 定义。',[1])
put([182,204],'MIXED','some','mixed','active FP32 Opartial 写出已由物化公式/target 规定；N_r=0 仍有 store 事件但不读 Ostage、不写 payload 是必要契约。',[3,10,14])
put([183],'MIXED','some','mixed','update 的寄存器状态计算引用 parity；Y 槽归还后再计算 Lbase 的顺序已在 pipe，不能以笼统数学 compute 合并掉中间释放。',[3,9])
put([184],'MIXED','some','mixed','ScoreBlock 捕获/立即返回与 mask 条件同段；末两逻辑块才掩码的成本选择未由上游逐元素正确性公式唯一规定。',[3,8])
put([185],'MIXED','some','mixed','Y 初始化与空域解释混合；member 0 输出清零/fence/通知独占跨成员协议事实，且 store 对它等待尚未显式化。',[6,10])
put([186],'ESSENTIAL_TEXT','all','open_semantics','PDL release 动作把本 CTA 贡献绑定 grid launch_ready；发射 statistic store 后即可放行，不能偷换为 kernel_done。',[3,11])
put([187,202,213],'ESSENTIAL_TEXT','all','open_semantics','初始化空 sum parity 的信号/单块 no-op 分支是动作语义；不能删无 payload 的事件，q1 和 q4 的发信 role 已分别核对。',[3,10])
put([190],'MIXED','some','uncertain','PageIndex 完成/提前预取定义与环复用顺序未证实限制并存；同槽 LoadV 读取完成不能从 turn 边推出，保留缺口。',[5])
put([193,195,209,219,221,222],'ESSENTIAL_TEXT','all','open_semantics','yield_turn 的 done 表示最后 slice/operand acquire 后的通知，而非传输/矩阵结果 done；已有 sync 边需这个不同完成含义才能保持早放行。',[3,4])
put([196,220],'MIXED','some','mixed','P_pair 的两个 FP8 值与 ExponentialSum 范围/发布含义是接口定义；先 pair 后 sum 的顺序与 serial pipe 重复。',[3])
put([198],'DUPLICATE','none','already_present','produce:output 的通用含义沿用数学 normalize 和 target.output，已显式规定一次 BF16 cast/覆盖；可省略整个 generic verb 覆写。',[1])
put([203,207],'MIXED','some','mixed','四组 FinalSum 合并、Normalization 覆盖 Maximum 的动作计算/副作用存在；cluster 额外包含跨成员 XOR 分母算法。',[3,7])
put([205],'MIXED','some','mixed','PDL wait 的原语绑定和完成条件保留；任何 partial read 前等待已由 serial prologue 与 handoff.wait_at@done 固定。',[1,11])
put([206,224],'MIXED','some','open_semantics','两种 combine 的 warp 合作与每线程重新扫描成本不同；当前 member:warp[0:4] 与相同 pipe 不唯一确定计算分配。',[8])
put([211],'ESSENTIAL_TEXT','all','open_semantics','XOR partner 及逐轮 transfer 完成定义 cluster 算法/拓扑；仅 reduce:ClusterMax 名称和本地事件不能还原它。',[7])
put([212,231],'MIXED','some','mixed','maximum/P-ring 的关键前驱已在 sync/reuse；动作更新/捕获 m 的实现语义仍需定义，q1 的 alternation 不等于可丢事件。',[1,3])
put([214],'MIXED','some','mixed','rescale 定义读取/修改/fence/归还 Y，依赖与后续 Lbase 的顺序部分已显式；不能删除资源副作用。',[3])
put([225,229],'MIXED','some','mixed','统计量异步搬运后 CTA 可见与 scratch payload 读域是核心动作事实；compact 跳过 inactive payload，global 读取 finite 旧值。',[3,8,14])
put([233],'MIXED','some','open_semantics','输出搬运是已定义 store，但异步完成/退出保证不可从 async role 的末尾 issue 推导；保留操作完成条件。',[3])
put([234],'FORMAL','all','formula_or_predicate','workspace 最小字节量与四字节整除的正式兼容性谓词，含额外预留区；不是可删除实现杂项。',[14])
put([235,236],'MIXED','some','mixed','setmaxnreg 与 launch168→角色64/168/256 的实现兼容性；稳态数值部分已有 info.register，空 CTA 不再分配不能从静态预算推出。',[11])
put([237,239,240,241,242],'ESSENTIAL_TEXT','all','open_semantics','选定原语为现有抽象动作提供 TMEM、完成/发布或集体归约保证；这里只证明绑定需要交接，不证明某个 API 名是唯一合法实现。',[11])
put([238],'MIXED','uncertain','open_semantics','named-barrier 是已知来源绑定；独立 turn/提前放行已在图和动作契约，尚无同图替代的主要成本证据证明该 API 必须固定。',[11])
put([243],'MIXED','some','mixed','FP32 workspace 视图和未消费预留区解释容量额外项；不能以无消费者为由缩小接口所需字节。其完整地址片段仍可在源码。',[14])
put([244,245,246,247,248,249,250],'EXPLANATION','none','already_present','必填实现选择理由，指回已经保留的数值/流水线/寄存器或别名条件；不是另一套独占设计定义。',[1,11])
put([251,252,253],'EXPLANATION','none','already_present','必填 phase 工作摘要；worktile/target/合并公式与实际 pipeline 定义工作。compact 的 nominal 是统计槽枚举，不是读每个 payload。',[1])
put([254,259],'DUPLICATE','none','already_present','optional schedule.strategy 完全复述同节点紧邻 worktile_order 的外到内排列；可以合法省略字段。',[1])
put([255,256,257,258,260,261,262],'MIXED','some','mixed','分块/选择预算与 WorkSplit 是正式绑定定义；遍历/空 split/清零描述部分复述数学和 pipeline，compact 末句是选择域外说明。',[2,6,16])
put([263],'ESSENTIAL_TEXT','all','open_semantics','CTA rank→split 一一归属和每成员各载 Q 是核心。mapping.unit_count 和集体 iteration shape 本身不规定分工；member 列表对变长 S 的紧凑复用有成本。',[7])
put([264,265,266,267,268,269],'EXPLANATION','none','already_present','整体工作主张摘要；launch/handoff/partition 与本地 combine/cluster 契约完整保留核心差异，不能只按大树重复计可删量。',[1])
assert set(A)==set(range(270)), sorted(set(range(270))-set(A))
# Context override: same sentence has a distinct nominal/payload reading in compact.
for o in U[252]['occurrences']:
 if o['node']=='q4_compact':A[252]['occurrence_overrides'].append({'occurrence_id':o['occurrence_id'],'category':'EXPLANATION','core':'none','structure':'already_present','reason':'all nominal slots 指统计槽；此节点 normalize 与 load 明确只读 active Opartial，不可按 root 的同句解释为读全部 payload。','evidence':[f'inputs/{OP}/q4_compact.yaml:100',f'inputs/{OP}/q4_compact.yaml:354']})
# Findings: mechanisms, not one finding per duplicate occurrence.
find(1,'redundancy',[32,33,*range(35,48),*range(74,96),*range(107,153),*range(153,166),172,179,180,198,205,212,231,*range(244,255),259,*range(264,270)],
 '摘要/短释义与完整定义、事件及 worktile_order 分工明确；root 两项 schedule.strategy 完全重复结构。必填标题/meaning/thesis/reason 只能短写，不能空值或任意删字段。',
 '删减候选保留所有公式、容量定义、动作身份及默认/显式前驱。',
 '不适用：本 finding 不声称这些字段携带独占核心。FinalSum→Score wait→Correction capture 的边不随压缩改变。',
 "schedule:\n  worktile_order: [{by: batch, direction: ascending}, {by: query_tail, direction: ascending}, {by: split, direction: ascending}]\n  worker_assignment: {preassigned: {start: worker_id, count: 1}, refill: none, exit: assigned_done}\n# optional strategy is omitted; all existing waits_for remain.",extra=['rules/schema_docs/4.design.work_schedule.md:197','rules/schema_docs/5.design.tile_pipeline.md:112','rules/schema_docs/6.realization.md:27'])
find(2,'style',[48,49,50,51,52,53,54,255,256,257,258,260,261,262],
 'when 的谓词和 decomposition 的参数关系是正式定义，不能把 Let/otherwise 句式当成自由叙述漏洞；调度层没有独立 parameter 表但 bind 可定位策略中定义。',
 '节点选择、SplitCount/WorkSplit、QueryTile 及容量关系。',
 '删 SplitCount 或 WorkSplit 定义会使同一 partition/domain 允许不同独立工作数和链长。',
 '现有写法已成立：symbol.SplitCount.binding: bind:design.work_schedule::SplitCount；partition.split.size: SplitCount；strategy 中保留完整分情形公式。把同文搬到另一 strategy 不算结构化。',extra=['rules/schema_docs/0.common.md:154','rules/schema_docs/3.design.math_reshape.md:189'])
find(3,'structural_gap',[166,168,169,170,171,173,174,175,176,177,178,181,182,183,184,186,187,188,189,191,192,193,194,195,196,197,199,200,201,202,203,204,207,208,209,210,212,213,214,215,216,217,218,219,220,221,222,223,225,226,227,228,229,230,231,232,233],
 '现有依赖图仍依赖动作契约定义对象组合、初始化/捕获/归还副作用、空 payload 与 primitive done。这里是有意开放的动作语义，未证明要扩充字段或事件。',
 'yield_turn 完成于最后 acquire 后、release 完成于源读取退休、capture 返回 TMEM 槽而保留 RF、no-op 仍有事件。',
 '只留动作名和图时，可把 yield_turn.done 理解为整块传输完成（丢失重叠），或把 retire 理解为不释放 TMEM；也可错误给空 split 读取未生成 Y。这些不是 barrier API 拼写差异。',
 '已有 @issue/@done、info.verb.{meaning,execution_mode}、resource.view、waits_for 和 release 可承载；本样本已用它们，动作语义仍必须定义。没有提出未定义的 acquire/release 新事件来冒充解法。',gap='open_semantics',extra=['rules/schema_docs/5.design.tile_pipeline.md:98','rules/schema_docs/5.design.tile_pipeline.md:137','work/template_fa4_decode/source_gqa_decode_paged.py:1230','work/template_fa4_decode/source_gqa_decode_paged.py:1520','work/template_fa4_decode/source_gqa_decode_paged.py:2024'])
find(4,'structural_gap',[166,169,173,174,178,188,192,193,194,195,197,199,201,209,215,216,218,219,221,222,223,227,228],
 '四 slice 共用 KV 环的独立到达/读取退役被压进块级 load/score/accumulate 契约。现有域、视图和 release 可以展开，但要增加 slice 级有定义动作与边，属表示成本而非已证缺事件。',
 '后续 slice 搬运可与早期 slice GEMM 重叠；KV 槽等待真正旧操作数读完，P 归还独立于 Y commit。',
 '只保留 load:Kblk.done→score:ScoreBlock.done 不能排除先等全块再开始所有 GEMM的实现，也不能说明任何具体槽的旧读者是谁；删除正文会改变允许的执行/复用。',
 "可行方向使用现有语法而需完整展开验证：domain: {u: [0,4]}；resource.KV.view 已有 Kslice(i,u)/Vslice(i,u)；LoadK pipe 可按 u 执行 load:Kslice(i,u)，Score 在明确读完该 slice 的现有原语保证下执行 release:Kslice(i,u)。复用边指向该 release.done，不能指 load.issue 或直接假设 score.done 等于 operand release。为了保留块外接口须同时重写所有引用及最后-slice turn，未提供为全段等价已审短稿。",gap='representation_cost',extra=['rules/schema_docs/5.design.tile_pipeline.md:135','rules/schema_docs/5.design.tile_pipeline.md:260','work/template_fa4_decode/source_gqa_decode_paged.py:1255','work/template_fa4_decode/source_gqa_decode_paged.py:1420'])
find(5,'source_limitation',[190],
 '五个多块节点明确记录 PageIndex 四槽环复用未证；这不是语言无法写 reuse 边，而是不能凭审计自行造来源顺序。',
 'LoadK 在轮 i 预取 i+1，LoadV 捕获旧 PageIndex 后才可覆盖同槽。',
 'Lead=2 时，K 可在获下一 turn 前发出下一表项的预取；仅 K/V issue-turn 的存在不足以证明 V 已捕获同槽旧值。去掉 unknown 句会虚假宣称安全。',
 '若来源存在明确 V-capture 完成事件，可用 waits_for.reuse 指向该事件；当前 load:Vblk.done 是整块完成，会过度串行化且不能作为源事实新增。尚未证明所需现有 primitive 顺序。',gap='unresolved',extra=['work/template_fa4_decode/source_gqa_decode_paged.py:1230','work/template_fa4_decode/source_gqa_decode_paged.py:1336','rules/schema_docs/5.design.tile_pipeline.md:298'],confidence='high')
find(6,'existing_structure',[167,185,260,262],
 'cluster LoadV 对 member0 输出清零的等待是另一已命名 role 动作的前驱，当前只在 store/zero/strategy 文字中，现有条件 wait 与成员引用可以显式承载。',
 '只有活跃成员的 reduce-add 需先看到 member0 GPU-scope 清零；不得把清零提前变成所有 K/V 的启动屏障，空 store 事件仍保留。片段拆出原 store 内已有 acquire，未增加源实现不存在的等待；它不是已审等价的全文替换，也不纳入删减候选。',
 '删除这几处文字、保留当前图，会容许活跃成员在 member0 清零前 reduce-add；最后清零覆盖贡献。启动 receiver rendezvous 不能推出输出已清零。',
 "# LoadV: keep existing loop, store contract, and data/sync edges.\ninfo:\n  verb:\n    wait: {execution_mode: serial, meaning: 'Acquire member 0 output-zero visibility; the existing zero_done protocol provides this guarantee.'}\npipe:\n  repeat: i\n  steady: ['load:Vblk(i)', 'yield_turn:V(i)']\n  epilogue:\n    - repeat: active\n      steady: ['wait:OutputZero(active)']\n    - 'store:output_tile'\nwaits_for:\n  data:\n    'wait:OutputZero(active)@issue': ['Correction/publish:Ostage@done']\n    'wait:OutputZero(active)@done': ['Correction[0]/zero:output@done']\n# The existing zero:output contract includes the GPU fence/notification; member0 is always active because seq_lens>=q_len_per_req>0.\n# N_r=0 creates no wait and keeps the existing no-op store event.",extra=['rules/schema_docs/5.design.tile_pipeline.md:98','rules/schema_docs/5.design.tile_pipeline.md:353','rules/schema_docs/5.design.tile_pipeline.md:416','work/template_fa4_decode/source_gqa_decode_paged.py:1392','work/template_fa4_decode/source_gqa_decode_paged.py:1848'])
find(7,'structural_gap',[66,203,211,263],
 'cluster rank→split 与 XOR-partner 逐轮 max/sum 属当前开放成员/集体语义。固定 S 可用 mapping.member 明示分工；变长 S 和对等成员事件需要枚举/角色展开，缺紧凑表达的成本大于本例新增字段的证据。',
 '每成员处理自己的 r、独立载 Q；交换 r xor 2^j、每轮完成后再下一轮。',
 '仅 unit_count:S 与 reduce:ClusterMax 不能区分每个 CTA 重算全部 split 或一一分片，也不能区分 XOR 轮次与全串行合并；这会改变通信/并行度。',
 "固定 SplitCount=4 的现有示意：mapping.member: [{split: [0,1]}, {split: [1,2]}, {split: [2,3]}, {split: [3,4]}]。S=2/8/16 必须有对应完整列表或继续保留准确 strategy；不虚构 list-comprehension/Role[r xor...] 语法。集体可维持现有 reduce 契约与 realization DSMEM 绑定，或完整展开有定义 round 动作；增加空 action 名本身不解决数据匹配。",gap='representation_cost',extra=['rules/schema_docs/4.design.work_schedule.md:157','rules/schema_docs/5.design.tile_pipeline.md:353','work/template_fa4_decode/source_gqa_decode.py:272','work/template_fa4_decode/source_gqa_decode.py:302'])
find(8,'structural_gap',[64,69,184,206,224,225,229],
 'global 与 compact 的同形 combine role/pipe 并不确定 warp 分担权重或每线程重复计算，尾部 mask 避免内区计算也需正文；这是主要计算分配/成本因果的开放语义，不是任意线程 layout。',
 'warp 路径合作求 max/Z、广播并读取所有 nominal O；compact 每线程扫描/重算且在 payload read 前排除 inactive；内区不做 mask。',
 '保持 member:warp[0:4]、compute:split_weight/compute:Numerator 与输出数学关系，可以实现 128 线程重复 exp2 或每 warp 合作计算；计算次数和通信不同。',
 '计算次数/归约分组可继续用 arithmetic 与 math 公式明示，thread 映射不是当前独立固定字段。若拆成更多角色/数学接口需保持实际 warp 协作和依赖，未找到比现有短契约更紧凑的纯结构替换。',gap='open_semantics',extra=['work/template_fa4_decode/source_gqa_decode.py:467','work/template_fa4_decode/source_gqa_decode.py:531','rules/schema_docs/5.design.tile_pipeline.md:23'])
find(9,'source_limitation',[64,65,66,67,68,69,70,71,72,73,77,82,84,97,99,100,102,103,104,105,169,218,227],
 'FP8 hi/residual、parity regrouping 与 BF16 split-rounding 是公式/算术的核心路径；fast-exp2/rcp、contraction、BF16 reduce-add 内部行为仍缺独立误差或舍入依据，不能删除 unknown 或补成保证。',
 '每 V reduction slice 的 hi→lo 累加、unrounded FP32 denominator、cluster 先各自 BF16 再 unordered add。',
 '把两次 FP8 合成一次窄化或把每 split BF16 改为全局 FP32 合并后只 cast 一次，数值路径不同且仍可能满足简略 dtype 标注。',
 '已有 rewrite.math 的显式 cast/mm32/mma32 与 arithmetic.{approx,accumulate} 合适；未知误差用现有 unknown: 限定，不能靠需要字符串就证明语言缺口。',extra=['rules/schema_docs/3.design.math_reshape.md:231','work/template_fa4_decode/source_gqa_decode.py:43','work/template_fa4_decode/source_gqa_decode.py:337','work/template_fa4_decode/source_gqa_decode_paged.py:1527'])
find(10,'redundancy',[101,103,104,168,175,176,177,181,182,185,187,189,191,200,202,204,208,210,213,217,226,227,230,232],
 '空 repeat 不产生动作而空 payload 的既有动作仍保留：本例 active/head/corr/tail 已清楚区分。可压缩解释空域的重复句，不能删 Ostage/store/EmptySum/FinalSum 事件。',
 'N_r=0 保留 Startup/neutral stats/FinalSum/retire 和 no-op Ostage/store；N_r=1 初始化并捕获空 parity1，先归还后做剩余 normalization。',
 '用 active repeat 包围整个 epilogue 会删除其他 role 仍等待的事件；把 N_r=1 当成只有一个物理 Y 槽会令 tail 捕获读取未初始化存储。',
 "domain: {active: [0, 'min(1,N_r)'], tail: ['max(0,N_r-2)', N_r]}；现有 zero:Y(active) 位于 active 块，而 publish:FinalMaximum/publish:FinalSum/publish:Ostage 在其外。保持这些结构；no-op payload 条件仍由动作契约说明。",extra=['rules/schema_docs/5.design.tile_pipeline.md:173','work/template_fa4_decode/source_gqa_decode_paged.py:1138','work/template_fa4_decode/source_gqa_decode_paged.py:1549'])
find(11,'structural_gap',[163,186,192,199,223,235,236,237,238,239,240,241,242,244,245,246,247,248,249,250],
 'Realization 的 primitive/寄存器兼容条件是有意开放绑定。静态 info.register 不定义 setmaxnreg 的初始池与空 CTA 分支；named-barrier 名称本身未被证明不可替换，不据此计新核心事件缺口。',
 'launch168 的寄存器池和非空 CTA 的再分配；PDL launch permission 与 predecessor completion 分离；TMEM operand release 与 commit 分离。',
 '删除 setmaxnreg 池条件可能容许 Softmax 请求192并使分配超过 CTA pool；把 PDL permission 当成数据就绪则可早读 partial。仅换 barrier API、保持图和保证不构成反例。',
 '已有 info.register 保留稳态64/168/256，realization.require 绑定 providing primitive；现有 launch.protocol:pdl、start_after:partial.launch_ready 和 handoff ready:partial.kernel_done/wait_at:combine.read_partial 已分开事件。更换 API 的可行性/成本另需证据。',gap='open_semantics',extra=['rules/schema_docs/6.realization.md:9','work/template_fa4_decode/source_gqa_decode_paged.py:658','work/template_fa4_decode/source_gqa_decode_paged.py:1138','work/template_fa4_decode/source_gqa_decode.py:442'])
find(12,'structural_gap',[55,56,57,58,59,60,61,62,63,166,188,192,199,201,223],
 'execution/feature/stack 的定义开放是明确设计取舍；库返回 T=512/Csm=232448 的身份与容量公式不能从下游使用反推。wiki pending 仍是资料绑定限制，不能叫已验证硬件基线。',
 'warp/CTA/cluster 作用域、SMEM owner 存活、TMEM 独立分配、所选栈版本和阶段容量来源。',
 '只保留 space:SMEM/placement:cluster 时不能知道跨成员可见性需要 owner 存活；仅保留 score slot 数不足以恢复整个 TMEM 分配与 KVStageCount 预算。',
 '当前硬件区只允许 execution/feature 句子及 wiki 引用；本例 stage/slot_count 和 member_slot_shape 已结构化，公式定义属正式表达而不需虚构 capacity 字段。',gap='open_semantics',extra=['rules/schema_docs/2.contract.hardware_specification.md:53','rules/schema_template/examples/README.md:54','work/template_fa4_decode/source_gqa_decode_paged.py:270'])
find(13,'structural_gap',[0,1,8,10,14,15,22,24,27,28,29,30,31,34,96,102,103],
 '外部 output/out_after/out 的对象同一性、contiguous ABI、非数值 None 及掩码/访问域依赖 contract math/meaning；已有公式/范围字段可表达，不把缺专用 alias 字段自动判为缺语言。',
 '返回 out 原对象、覆盖而不读旧内容；sentinel 必传；首轮访问 unused 合法表项，score 可见性独立。',
 '仅同 shape/dtype 容许分配另一个输出、只写 out_after 而不返 out，或将 None 视为可省略参数；这些都会改变 ABI。',
 '保留 compute.math 的 out_after=output 与 same tensor object as out；python_none 定义配 dtype，无 default。其 range: Python None only 可合法省略；out.range 保留 Any BF16 bit pattern。',gap='open_semantics',extra=['rules/schema_docs/1.contract.algorithm_requirement.md:56','rules/schema_docs/1.contract.algorithm_requirement.md:240','work/template_fa4_decode/source_reference.py:72','work/template_fa4_decode/source_operator.py:599'])
find(14,'source_limitation',[1,3,33,98,106,119,120,182,204,225,229,234,243],
 'global inactive scratch 读取依赖首次清零及后续有限旧值，compact 不读其 payload。handoff.initialize:0 每次清零会改变旧值/成本，不能当免费结构修复。来源 Model.forward 的 native cache 构建/失败另有实际执行，当前 captured 计算边界未覆盖这些路径。',
 '保留跨调用 scratch 值与额外预留 region 容量；capture 时 host 选择/标量固定、seq_lens 等设备内容按当次读取。',
 '去掉 finite 前提允许 NaN scratch 被零权重读取造成 NaN；改成每调用初始化增加工作。把首次 eager native 验证的 ffi 再执行当成仅编译，可漏算真实计算。',
 '已有 input.meaning/compute.math 可声明首用与旧值前提；handoff 保留当前 storage/ready/wait_at。若扩审整个入口，按已存在 flow.launch 为实际额外提交登记 phase 和边，而非新增 capture 事件。',gap='unresolved',extra=['rules/schema_docs/4.design.work_schedule.md:353','rules/schema_docs/0.common.md:39','work/template_fa4_decode/source_operator.py:227','work/template_fa4_decode/source_input.py:83','work/template_fa4_decode/source_operator.py:547','work/template_fa4_decode/source_operator.py:699','work/template_fa4_decode/source_operator.py:826'])
# Exactly unchanged math region in the direct parent is an existing reuse opportunity, not global duplicate deletion.
shared=[i for i,u in enumerate(U) if u['family'].startswith('math.') and any(o['node']=='q1_global' for o in u['occurrences'])]
find(15,'existing_structure',shared,
 'q1_global 的完整 math_reshape 与直接父 root 解析后完全相等；其 QueryTile/WorkSplit 经 bind 解析到 q1 本地调度，因此可以用现有整区引用。其他数学相等的 sibling 不可直接互引，局部相同也不能 patch。',
 '同一公式在 q1 参数下成立，三个工作/流水线区域保持本地 q1 行为。',
 '只因 parent 为 root 就删数学区不合法；必须显式 from_parent，且不能继承 root 的 when 或 realization。',
 'design:\n  math_reshape: {from_parent: root}\n# q1_global alone; work_schedule and tile_pipeline stay local.\n# raw math string declarations 6436 -> 4; effective math remains 6436. Not included in prose removal savings.',extra=['rules/schema_docs/0.common.md:125','inputs/template_fa4_decode/q1_global.yaml:16','inputs/template_fa4_decode/root.yaml:93'])
find(16,'noncore',[257],
 'compact 的 B=12/S=8 源分支在本节点 selector 下不可达；说明有源审计价值，但不参与既定有效设计。',
 'WorkSplit 的 S=8/B10、11 和 S=32 分情形保留。',
 '不适用：删除不可达分支说明不改变任何契约域内被该节点选择的动作或结果。',
 '保留现有 when 与全部 WorkSplit 公式，仅删除最后的不可达源代码备注。',extra=['work/template_fa4_decode/source_operator.py:730'])
# Preserve detailed occurrence coverage and connect every annotation to the relevant findings.
for f in F:
 for uid in f['unit_ids']:
  i=next(i for i,u in enumerate(U) if u['id']==uid)
  if f['id'] not in A[i]['finding_ids']:A[i]['finding_ids'].append(f['id'])
for f in F:
 if f['id']==fid(15):
  f['nodes']=['q1_global']
  f['applies_to_occurrences']=[o['occurrence_id'] for i in shared for o in U[i]['occurrences'] if o['node']=='q1_global']
  f['binding_evidence_nodes']=['root','q1_global']
# Conservative non-overlapping quote -> replacement proposals. Do not remove their supporting definitions.
for i in [254,259]:cand(i,U[i]['text'],'','同节点 worktile_order 完整保留外到内排列，strategy 为可省略字段。')
for i in [27]:cand(i,U[i]['text'],'','省略 optional range 字段；保留 dtype:python_none 及 root.compute 的单值类型定义。')
for i in [198]:cand(i,U[i]['text'],'','删除可省略的 generic produce 覆写；保留 produce:output 动作、上游 normalize 与 target.output，不删除其他 verb。')
for i,tail in [(49,'Changes in the global split count share one two-kernel design.'),(50,'Nominal S=8/32, near-uniform and proportional active counts share the serial-per-thread combine design.'),(51,'Short, small-batch and four-member policies share the cluster protocol.'),(53,'The nominal cluster sizes 4, 8 and 16 share this path.'),(54,'The wave-fit and occupancy split counts retain the same two-kernel design.')]:cand(i,tail,'','保留该 when 的全部 Let 定义与充要谓词；同节点完整 Design 表达其参数范围内机制。')
cand(257,"The host's B=12/S=8 compact clause is unreachable under the default selector.",'','当前 selector 排除该分支；保留 WorkSplit 的全部可达定义。','noncore')
cand(230,'For N_r=0, active is empty: no zero action or wait for it exists, and setup statistics remain in use.','','domain.active、active repeat 及 Startup initialize:Statistic 已保留；不删 no-op Ostage/store 或 neutral stats 契约。')
cand(232,'For N_r=0 active is empty, so no zero action or wait is created; setup statistics remain in use.','','同节点 active 域和 zero 所在 repeat 给出动作不存在；初始化中立统计仍由 initialize 定义。')
cand(185,'For N_r=0 active is empty, so no zero action or wait is created; setup statistics remain in use.','','保留随后 member0 输出清零/fence/通知全部文字，仅省略现有 active 域可推导的 zero 空域说明。')
cand(172,U[172]['text'],'Wait for FinalSum.','保留 Score wait 的 serial 与显式 FinalSum 前驱、Correction capture→Score wait 的 sync；只压缩重复关系，仍是合法非空 verb 定义。')
cand(180,U[180]['text'],'Wait for FinalSum.','保留已有 FinalSum 发布→wait→capture 边和 execution_mode；未改 API 绑定或完成事件。')
cand(179,U[179]['text'],'Publish m.','保留 exponent 对 m 的定义、Correction read:m 与最终 Normalization 的数据/串行顺序；此 direct 域只有一次更新。')
cand(184,'Capture raw ScoreBlock in registers and immediately return its TMEM slot.','Capture ScoreBlock in registers and return its TMEM slot immediately.','只合并句法，完整保留捕获/即时返还和所有末块 mask 条件。')
for i in [166,188,192,199,201,223]:
 q='T is in TMEM columns and Csm is in bytes; these are the values returned by the original sm_100 helper calls in CUTLASS DSL 4.6.1.'
 cand(i,q,'T and Csm are sm_100 helper limits in CUTLASS DSL 4.6.1: TMEM columns and SMEM bytes.','保留具体 512/232448、单位、库版本、所有容量公式、特殊分支及预取/初始化保证；没有把源码来源当设备实测。')
for i in [175,181,210]:
 cand(i,'FinalMaximum releases the final m before the last parity correction.','FinalMaximum publishes the final m.','Correction pipe 已明确 FinalMaximum 位于 corr_final rescale 前；保留发布对象/完成含义，其余 sum/no-op/Ostage 内容全部保留。')
# Required summary fields get valid short replacements; no backing formulas/contracts are removed.
cand(164,U[164]['text'],'Separate load, MMA, softmax and correction roles; overlap two numerator parities with bounded issue lead.','各本地 role/member/register、Lead 公式、slot_count、reuse/sync 均保留，提供每个摘要决定的唯一详细依据。')
cand(159,U[159]['text'],'Compute one key block with two FP8 probability components and an FP32 denominator.','保留 direct 的所有数学公式、Qpack 映射和非物化事实；thesis 仍非空。')
cand(160,U[160]['text'],'Stripe KV chains and use two FP32 numerator parities with leading/residual FP8 probabilities.','保留 rewrite 的分片/双 parity/残差积与 ledger 的额外 contraction/省全局中间流量；thesis 仍完整概括决定。')
notes={
'root':'v9；q4 global：QueryTile=2/R=32，partial+PDL combine，FP32 nominal scratch 全读；单 softmax、64/168/256 预算、active/corr/tail 与 Ostage 复用两方读者核对。',
'q1_global':'v7；q1 global：QueryTile=1/R=16，数学与 root 全等但参数绑定本地；even/odd softmax 分别 2*j/2*k+1，Correction warp12:16，PDL/kernel_done 及 all-nominal 读取核对。',
'q4_compact':'v7；q4 compact：名义 S8/32，设备 WorkSplit，所有 CTA 提交但不同 N_r；combine 每线程重算并跳过 inactive payload，保留 no-op load/store 事件。',
'q1_cluster':'v7；q1 cluster：S2/4/8/16 参数成员，QueryTile1、双 softmax；成员独立 N_r，empty 参加统计，rank0 清零，XOR max/sum、BF16 reduce-add 与每成员独立载 Q 核对。',
'q4_cluster':'v7；q4 cluster：QueryTile2、S2/4/8/16 与本地选择政策，单 softmax+寄存器重配；empty CTA setup、cluster receiver 初始化与 output zero 分时，rank0 依赖需显式化。',
'q1_direct':'v7；q1 单 KV block，七 role/16warp，SoftmaxOdd 只发 EmptySum；仍有双 Y 槽及 empty parity drain，独立 K/V slice、Normalization alias 和 BF16 out 核对。',
'q4_direct':'v7；q4 单 KV block，每请求两个 QueryTile；六 role/12warp，Softmax 自发 EmptySum，register64/168/256；direct 数学和 q1相同但 bind=QueryTile2。'}
limits=[
'这是冻结规范 v1.1 下的文字研究审计；7份均为 template，不是 GPU 数值/性能验证基线，没有运行 GPU、修改原 schema 或运行提取器。',
'270组、777 occurrence 均手工语义判断；完整 YAML 通过主定义+所有本地差异阅读，scalars 的其余2111字符串/285精确值逐类核对为表达式、引用、标识或显式 pending，没有发现遗漏重要散文。',
'源证据限定 git 01fff90 的 aka_solution，work/source_manifest.json 固定路径和 SHA256；亲读了入口、selector/ABI/native cache、reduction 文件及 paged 的容量/setup/关键流水线/收尾段，没有逐行复查全部线程布局和外部 CUTLASS helper 实现。',
'fast-exp2/rcp 误差、FP32 contraction、BF16 reduce-add 内部算术以及 PageIndex reuse 原语顺序仍未证明。没有凭 GPU 历史状态补充保证。',
'Contract 只说明 captured 计算及 cache 构建不在计时域；源 native builder 实际重复 ffi、copy/检查和异常恢复没有在当前七节点建模，不声称全 Model.forward 冷启动/失败路径均已覆盖；纯 capture 无 kernel_done，回放按执行实例重新产生事件。',
'删减字符为人工候选替换的归一化 quote 长度减 replacement 长度，不是获独立短稿等价审阅的可安全删除字数；整区 from_parent 的结构复用机会不并入 prose 节省。',
'表示成本和有意开放语义不等于已证需要新字段；named-barrier 单纯API替换没有主要性能证据，核心必要性列 uncertain；没有已证 missing_field_or_event。']
report={'operator':OP,'prompt_version':'v1.1','unit_annotations':[A[i] for i in range(270)],'findings':F,'removal_candidates':C,'node_review':[{'node':n,'reviewed':True,'effective_binding_checked':True,'notes':v} for n,v in notes.items()],'limits':limits}
(B/'annotations'/f'{OP}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('annotations',len(A),'findings',len(F),'candidates',len(C))
print(collections.Counter(a['category'] for a in A.values()))
