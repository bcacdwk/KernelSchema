import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[2];op='template_qsa_prefill'; S=json.loads((R/'data'/op/'audit_summary.json').read_text());A=json.loads((R/'annotations'/f'{op}.json').read_text())
lines=['# template_qsa_prefill — 全量文字审计 v1.1','',
'结论：当前 root 仍依赖开放文字确定若干核心设计，但不应据此断言需要扩展语言。ticket 确认、双槽 KV 环、三阶段 S/P 别名和 Y 两半归还已有结构；条件 warp 修正等待是明确的现有结构未用问题，细分静态角色可表达而成本较高。数学公式本身不是漏洞，API 名称不同也不是核心缺口证明。','',
'样本只有 `root`，状态为 **template**，版本9，384行、29759字节，SHA256 `9d36b006a339be7969dddfdcd40c1d94fc403d9f0153cc0624db8e2ec4015686`。来源是冻结 examples，不是 GPU/完整语义已验证基线。完整审阅 Contract、3个rewrite、两次launch、preprocess与attention两个phase、7个role、全部资源/事件与3条require，以及dispatch的5组文本。126/126精确文本组全部人工分类，140个声明上下文均检查；459个scalar复核未发现开放正文漏项，无需occurrence override。','',
'规则为本目录冻结 README、DOC0–6、REVIEW和examples README。按 `meta_data.source` 使用 `git show 01fff90d714abc6e84904ea9c44523ba66d34a0e:<path>`，完整亲读 solution.py、reference.py、operator.py、qsa_kernel.py、qsa_preprocess.py；原字节及SHA256记录在 [work](../work/template_qsa_prefill/SOURCE_IDENTITY.md)。核对以执行代码为准，未把源码旧注释的调度摘要当成额外事实。','',
'统计按统一 `scope_statistics`：字符均为归一化字符串长度；“核心”栏仅统计 `core=some/all` 且category非FORMAL的字段，不表示整段不可压缩。W/T/R在相同口径下另细分。','',
'|范围|精确组/去重字符|声明数/字符|含非正式核心的组/声明/字符面积|候选净缩减|','|---|---:|---:|---:|---:|']
for key,title in [('all','总计'),('yaml','英文 YAML'),('main_WTR','W+T+R'),('math','Design 数学'),('hardware','硬件 execution/feature/stack'),('dispatch_md','中文 dispatch.md')]:
 d=S['scope_statistics'][key];lines.append(f"|{title}|{d['unique_units']} / {d['unique_chars']}|{d['declarations']} / {d['declared_chars']}|{d.get('nonformal_core_bearing_units',0)} / {d.get('nonformal_core_bearing_declarations',0)} / {d.get('nonformal_core_bearing_declared_chars',0)}|{d.get('candidate_saved_declaration_chars',0)}|")
lines+=['','|分区|组/声明|去重/声明字符|非正式核心组/声明/面积|','|---|---:|---:|---:|','|W work_schedule|9 / 9|948 / 948|5 / 5 / 661|','|T tile_pipeline|19 / 25|4112 / 4683|16 / 22 / 4331|','|R realization|6 / 6|619 / 619|1 / 1 / 113|','',
'无 `dispatch.when`；单 root 的缺省全契约覆盖按DOC0理解。raw与有效Design相同，没有from_parent，没有YAML anchor/alias，也没有跨文件复制；14次额外声明来自同文分开写入，不能误称alias展开。root全部字符串声明量=实例量=17448，含公式、引用和标识符；有效Design全部字符串为数学4551、调度1324、流水线6852。开放字段声明面积与上述全字符串量分开。','',
'|全字段类别|组数|去重字符|声明数|声明字符面积|','|---|---:|---:|---:|---:|']
for cat in ['DUPLICATE','NONCORE','STRUCTURABLE','ESSENTIAL_TEXT','FORMAL','SOURCE_LIMIT','EXPLANATION','MIXED','UNRESOLVED']:
 d=S['category_statistics'].get(cat,{})
 lines.append(f"|{cat}|{d.get('unit_count',0)}|{d.get('unique_chars',0)}|{d.get('declarations',0)}|{d.get('declared_chars',0)}|")
lines+=['',
'34个YAML文本组/40个声明含非正式核心信息，均位于唯一root；其中WTR为22组/28声明。ESSENTIAL_TEXT的20组不是20项已证语言缺陷。没有整字段STRUCTURABLE，因可迁移的依赖常与独占动作含义混在同字段；所有部分迁移亦登记finding。两个Realization API字段的核心重要性保留uncertain，不纳入已证核心字段计数。来源限制嵌在MIXED字段中，所以SOURCE_LIMIT=0不表示限制不存在。','',
'13项去重机制均附完整unit IDs、反例、证据和替代表达于 [annotations](../annotations/template_qsa_prefill.json)：','',
'|Finding|覆盖及结论|','|---|---|',
'|F01：外部ABI，7组|O与out同对象、K/V存储与stride、忽略hint和验收路由需要开放合同。shape/ref只传属性，片上resource.view不能表达外部tensor别名；属于开放语义，未证明必须增字段。|',
'|F02：摘要/说明重复，36组|标题、meaning、tail=6及结构摘要有确定上游来源。必填name/meaning/describe/thesis不能直接删除。|',
'|F03：硬件，10组|执行层级、存储作用域及SMCount提供者由文字定义；stack版本是正式身份。cluster对该单CTA路径无核心作用，但DOC2要求同设备事实统一，未建议随算子删硬件条目。|',
'|F04：公式与账本，30组|保留union、unmasked max、lagged reference、两半L、BF16 P和输出两次舍入；Br/M/Etile/Bk是正式绑定。ledger量化的是指定baseline，不能当无关文字。|',
'|F05：预处理/gather，5组|CAS+OR、warp reservation、两级地址前视、末块发布/no-op drain为动作含义。load(n+1).issue→load(n).done、n.flatten、双槽reuse已有结构，相关正文可局部缩减。|',
'|F06：S/P alias，3组|视图已定义共存储，async_ordered_completion已定义累积完成；旧PV操作数读先于后续QK写和done含operand reads仍需原语合同，不能从事件有序推出。|',
'|F07：ticket/ack，7组|五角色共十一warp的ack等待已显式登记；accept复制/ack、terminal不工作但仍确认、初始Q等待及无工作drain仍需定义。|',
'|F08：capture/early release，8组|Q refill退役顺序和Y双半归还已在图内；可用现有view补Y_A/Y_B切片。32列读取、寄存器尾片、normalization捕获等内部动作不能只用新名字替代。|',
'|F09：条件warp等待，2组|源码仅A中vote=true且i>0的warp等待前一PV；当前正文新增跨role前驱，违反DOC5。四个静态warp role配条件repeat/wait可表达，完整展开仍需重新审阅，登记为representation_cost。|',
'|F10：跨调用状态，3组|ticket/base初值、W+C增量和int32 wrap由开放合同给定；单次phase/handoff并不表达跨调用持久状态。捕获按值tbase反复回放可能全部取得terminal，故不能把eager覆盖外推到replay。|',
'|F11：Realization，6组|保留提供上游保证的具体原语来源和align/stride兼容条件；未以更换barrier/copy API证明新核心缺口，未建议整体删除require。|',
'|F12：来源限制，4组|Q padding影响vote、HalfMax所有读者退役、exp2/rcp误差界未证实；不能删限制，也不能靠补零/额外barrier伪造现有设计。|',
'|F13：显式None边界，3组|源码接收out=None，当前Contract仅有default无nullable。DOC1已有nullable写法，但原始任务是否覆盖显式None未独立取得，作为来源/选择域待确认项。|','',
'F01/F03/F05/F06/F10共5项 `structural_gap/open_semantics` 表示本稿核心靠开放语义承载，不是已证“必须新字段”；缺失固定字段/事件类的已证finding为0。F08/F09为2项现有结构方向，其中F09同时带1项表示成本。','',
'关键现有写法（均保留动作契约，不用空动作名替代语义）：','',
'```yaml',
'# 以下仅为审计建议片段；原schema未改。',
'role:',
'  GatherK:',
'    waits_for:',
'      sync:',
'        "load:Kblk(n)@done": ["load:Kblk(n+1)@issue"]',
'  MMA:',
'    waits_for:',
'      reuse:',
'        "mma:P(n[t+1,0])@Vblk(n[t+1,0])@issue":',
'          - "SoftmaxA/drain:Y_A(t)@done"',
'          - "SoftmaxB/drain:Y_B(t)@done"',
'```','',
'条件等待的候选为每个静态A-warp先计算并保留weights/vote，随后用 `q:[0,1 if i>0 and move_ref(i,32*w) else 0]` 的局部repeat产生serial `wait:PriorPV(n,q)` 和已定义的rescale，wait.done显式依赖 `MMA/mma:P(n-1)@Vblk(n-1)@done`。发布每warp P slice后MMA等待全部四份slice及B半区。JSON给出动作契约与必须同步拆分的HalfMax/ack/Q/drain边界；这不是完整等价短稿，不能直接用于删除原文或将四warp统一加等待。','',
'保守短写共有10处，合并无重叠quote后净缩减 **772字符**：WTR653、其他YAML119、数学/硬件/dispatch均0；仅是候选，约占YAML开放声明面积5.78%。所有完整原quote与replacement在JSON，代表项如下：','',
'|位置|原quote/候选|净减|','|---|---|---:|',
'|root:244 Scheduler.wait|删除“Join all consumer acknowledgements before posting the following ticket or retiring the terminal ticket.”这个可选verb契约；保留wait动作与sync边。|103|',
'|root:295 MMA.wait.meaning|“Wait for the initial Q publication even when Ticket0 is terminal.” → “Wait for Qblk0.”；保留execution_mode:serial。|50|',
'|root:241 Scheduler.publish|只去掉已由五个accept前驱及serial pipe保证的ack/one-worktile lookahead重复句。|204|',
'|root:256 GatherK.load|“load:Kblk(n) becomes done only after issuing tile n+1 and publishing n\'s completed copy.” → “load:Kblk(n) done publishes its completed copy.”|41|',
'|root:175 attention.strategy|保留“M=128; Etile=16; Bk=4*Etile.”，去掉可推导的tail=6和无跨工作归约句。|77|','',
'其余为HalfMax重复先后139、Q预取重复退休关系39、v_cache meaning49、out meaning38、meta.describe32。MMA初始化句未计入候选，避免把数学零初值与动作边界的缩写等价性当成已完成审阅。','',
'边界核对：本域J非空、N(b)>=32，MMA两项prologue/两项epilogue均存在；末row block为6真实行，其余packed rows不输出但仍参与共享vote，其影响未证实，限制保留。worker可首票即terminal，此时五方ack、A初始Q gather、MMA初始Q等待和两个drain动作事件仍存在，drain无payload不等最后tile。KV的连续n跨worktile，不在工作边界强制排空。Y下一worktile首PV等待A/B两半读完，尾部gate/store仍可进行；HalfMax缺失的读退役保证没有擅自补边。外部out/KV别名不同于片上视图；跨调用ticket、串行eager、capture/replay及显式None边界分别保留。','',
'运行 `python3 -B scripts/summarize.py --operator template_qsa_prefill` 得到VALID，覆盖126/126组、1/1节点、UNRESOLVED类别0；统计见 [audit_summary](../data/template_qsa_prefill/audit_summary.json)。这只验证交付格式、范围、quote与证据定位，不证明全部语义已正确。未运行GPU、未做独立短稿等价验收，未独立复查外部硬件手册和环境记录；约束与未决事实详见JSON limits。','']
# Use actual finding counts rather than manually assuming list expansion counts.
counts={f['id'].rsplit('-',1)[1]:len(f['unit_ids']) for f in A['findings']}
import re
text='\n'.join(lines)
for k,v in counts.items(): text=re.sub(r'('+k+r'：[^(|]+，)\d+(组\|)',lambda m:m.group(1)+str(v)+m.group(2),text)
(R/'operators'/f'{op}.md').write_text(text)
w=R/'work'/op
src=['# 固定源码身份','','Commit: `01fff90d714abc6e84904ea9c44523ba66d34a0e`。全部副本通过git show取得，未经修改。','','|副本|原路径（均在kernel_wiki/aka-four-problems-20261019/qsa_prefill下）|SHA256|','|---|---|---|']
for fn,path in [('solution.py','aka_solution/solution.py'),('reference.py','reference.py'),('operator.py','aka_solution/qwen38_core_qsa_cutedsl/operator.py'),('qsa_kernel.py','aka_solution/qwen38_core_qsa_cutedsl/qsa_kernel.py'),('qsa_preprocess.py','aka_solution/qwen38_core_qsa_cutedsl/qsa_preprocess.py')]: src.append(f'|{fn}|{path}|{hashlib.sha256((w/fn).read_bytes()).hexdigest()}|')
(w/'SOURCE_IDENTITY.md').write_text('\n'.join(src)+'\n')
print('report written; findings counts',counts)
