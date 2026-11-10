import json,pathlib,collections
B=pathlib.Path(__file__).resolve().parents[2]
S=json.load(open(B/'data/extract_gemm/audit_summary.json'));N=json.load(open(B/'data/extract_gemm/nodes.json'));A=json.load(open(B/'annotations/extract_gemm.json'));U=[json.loads(x) for x in (B/'data/extract_gemm/units.jsonl').read_text().splitlines()]
rows=[]
for n in N:
 rows.append(f'| {n["node"]} | approved / v{n["version"]} | {sum(n["raw_string_declaration_chars"].values())} / {sum(n["raw_string_instance_chars"].values())} | {sum(n["open_text_declaration_chars"].values())} / {sum(n["open_text_instance_chars"].values())} | '+ ' / '.join(str(n['effective_design_string_instance_chars'][k]) for k in ['math_reshape','work_schedule','tile_pipeline'])+' |')
scopes=[]
for key,label in [('main_WTR','W/T/R 主口径'),('math','数学 Design'),('hardware','硬件'),('dispatch_when','YAML dispatch.when'),('dispatch_md','中文 dispatch.md'),('yaml','全部英文 YAML'),('all','全部开放文本')]:
 q=S['scope_statistics'].get(key,{})
 scopes.append('| '+label+' | '+' | '.join(str(q.get(k,0)) for k in ['unique_units','unique_chars','declarations','declared_chars','nonformal_core_bearing_units','nonformal_core_bearing_declarations','candidate_saved_declaration_chars'])+' |')
cats=[]
for k in ['DUPLICATE','NONCORE','STRUCTURABLE','ESSENTIAL_TEXT','FORMAL','SOURCE_LIMIT','EXPLANATION','MIXED','UNRESOLVED']:
 q=S['category_statistics'].get(k,{})
 cats.append(f'| {k} | {q.get("unit_count",0)} | {q.get("unique_chars",0)} | {q.get("declarations",0)} | {q.get("declared_chars",0)} |')
mechanisms=[]
for f in A['findings']:
 mechanisms.append(f'| {f["id"].split("-")[-1]} | {len(f["unit_ids"])} / {len(f["nodes"])} | {f["gap_kind"]} | {f["claim"]} |')
wtr=[]
for prefix,label in [('schedule.','W'),('pipeline.','T'),('realization.','R')]:
 u=[x for x in U if x['family'].startswith(prefix)]
 wtr.append(f'{label}：{len(u)}组/{sum(x["chars"] for x in u)}去重字符/{sum(len(x["occurrences"]) for x in u)}声明/{sum(x["chars"]*len(x["occurrences"]) for x in u)}声明字符')
text='''# extract_gemm 全量文字审计（v1.1）

7/7 个计划节点均为本次冻结 current-plan 的 **approved** 稿，230/230 精确文本组已人工分类，0 未决组；不是采样。16 个机制 finding 和 64 条精确 quote→replacement 候选见 [机器明细](../annotations/extract_gemm.json)，统一校验结果为 VALID。approved 不表示 GPU/源实现正确性或硬件完备。

本算子确实借助开放叙述确定一部分核心设计。统一主口径 W/T/R 中，70/102 组、141/210 声明含非 FORMAL 核心片段，涉及全部7节点；这包括 MIXED 中的正式参数、动作定义及必要来源限制，**不等于70项语言缺口或49008字符不可压缩**。已证明需要新增固定字段/事件的案例为0；主要依赖是有意开放的集体动作/原语完成保证，另有 host 状态定义复用和 role 内成员表达的成本。

## 身份、范围与统计

输入身份来自 manifest 的工作 checkpoint `6a1253c6db2be19c8849e0c2a2b76470a3247730a632a62da41255cb0c3f6d5c`；源码为 `SchemaExtractSource_2c7b5a74c0b3@497a8077ac54ed1431bc00772871337f9a309852` 的私有 snapshot。`solution.py`、`fp8gg_umma.cu`、`fp8gg.cu`、`fp8gg_bind.cpp`、reference/input/metadata/CERTIFIED_BEST 共8文件已核对 inputs.lock SHA，证据副本不改字节；见 [source_identity.json](../work/extract_gemm/source_identity.json)。核对了实际 UMMA V0 sareg=0、F/G、sync、host选择/缓存/恢复/绑定路径；未把诊断宏或未选择 SAREG=1 当当前设计。

规则使用本审计冻结 README、DOC0/1/2/3/4/5/6、REVIEW 与 examples README。特别遵守 DOC0 独立when/整区引用、DOC4新kernel独立worker、DOC5开放原语/空域/源读取归还、DOC6等价API可替换边界。硬件null为 DOC2:13 明示提取豁免，**不是本次作者漏填违规**，也不提供默认硬件能力。

字符按统一空白归一化后的 len，含公式、引用和标识符，不能称自然语言字数。下表“字符串”包含所有值；“开放文本”只取census范围。有效 M/W/T 是 Design 三个区域的字符串实例量，不与raw相加。

| 节点 | 状态/版本 | 全字符串声明/实例 | 开放文本声明/实例 | 有效 M/W/T 字符 |
| --- | --- | --- | --- | --- |
'''+ '\n'.join(rows)+'''

全部 YAML 字符串声明 '''+str(sum(sum(n['raw_string_declaration_chars'].values()) for n in N))+'''、实例 '''+str(sum(sum(n['raw_string_instance_chars'].values()) for n in N))+'''；开放文本声明101943、实例'''+str(sum(sum(n['open_text_instance_chars'].values()) for n in N))+'''。root单独为16942开放声明字符，其余分支合计85001；不能用树的复制总量说明root复杂程度。7节点均为全本地Design，没有from_parent；4恢复节点分别用2/2/3/4次YAML alias共用phase，声明不重复计而每个worker实例都检查了。跨文件相同字句则仍分别计声明。中文dispatch另列2776字符。

| 统一scope_statistics | 去重组 | 去重字符 | 声明数 | 声明字符 | 非FORMAL含核心组 | 对应声明 | 保守候选节省声明字符 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
'''+ '\n'.join(scopes)+'''

W/T/R细分：'''+ '；'.join(wtr)+'''。硬件0是整个Contract占位，不是已完成的简洁基线。自动开放目录外的193种不同字符串已检查，均为对象/引用/shape/retention/动作/事件正式记法，未发现漏收长正文；2533个scalar实例与alias结构均纳入核对。

## 全字段分类与全部机制

以下是字段面积，不是可以删除的字符数。ESSENTIAL_TEXT 28组中10组原语绑定的core为uncertain：有源码记录，但缺少“同一依赖图换API就改变核心”的证明，未算入确认的非FORMAL核心组。

| 主分类 | 去重组 | 去重字符 | 声明数 | 声明字符 |
| --- | ---: | ---: | ---: | ---: |
'''+ '\n'.join(cats)+'''

所有finding列出全部涉及ID、节点、最小删除反例、规则与源码行号以及候选现有写法；同组的每个本地声明及alias scope均已核对，没有需要改变主分类的occurrence override。“组/节点”包括解释/复述实例，不能当作独立缺陷发生率。

| Finding | 涉及组/节点 | gap_kind | 判断 |
| --- | ---: | --- | --- |
'''+ '\n'.join(mechanisms)+'''

F03/F08 是表示成本候选，F09/F10 是有意开放语义；共4个 structural_gap finding，均不声称需要扩语言。F01/F13亦记录有意开放数值/实现绑定。F05/F07/F11 是现有结构可表达的现象，只有F11的2个纯await字段可整体归 STRUCTURABLE；其余长字段仍混有必要定义，故标MIXED。

## 现有写法、边界和短写

纯 `await` 可合法改用通用 `wait`，不需要发明新动作。只在原位置重命名动作与左侧引用，role继续serial，仍把read完成接到wait.done；随后release与最终FMA相对位置完全保留：

```yaml
# root/v0: 对所有原 await 使用处作同一重命名；wide同理
steady: ["wait:Fragment(n,d)"]
waits_for:
  data:
    "wait:Fragment(n,d)@done": ["read:Fragment(n,d)"]
```

`info.verb.await` 是可选项，此时可以省略。该结构建议没有混入64个文字候选节省量；只删定义而不改pipe是不成立的。

多次恢复的完成链已经写成如下形式，不能通过正文取消一次提交，也不能把录图加入一次计算。phase body可用anchor复用，q/r绑定必须保留；不跨kernel继承局部资源。

```yaml
flow:
  launch:
    - {phase: [first]}
    - {phase: [retry], start_after: first.kernel_done}
    - {phase: [final], start_after: retry.kernel_done}
```

F/G 的批数、容量与输入深度也已有结构：`slot_count:R`，staging `[row:Bm,column:J*ScaleN]`，`repeat:{f:[0,Full(t)*2/J]}`。因此可以缩掉数量复述，但 `J=1(F)/2(G), R=5-J` 仍需定义。ragged工作逆映射和H绑定是正式数学规则，不能用一个未定义的mapping名字替代。

有据短写举例（64项完整原quote与replacement均在机器文件，合并区间不重叠）：

- `C is a submission count, not a claim that all workers are resident simultaneously.` → 删除；DOC4已给出worker_count的唯一含义，保留C/S/Wub公式。
- `Ring positions and generations continue across worktiles.` → 删除；已有 `n:{flatten:[t,kb]}`、`n%R` 和 `load(n+R)←mma(n)`；V0同理。
- `The next batch or worktile may advance while those writes remain in flight.` → 删除；store为async，release只等源读取，最终drain仍等目的完成，未加强顺序。
- `Convert the entire final F to BF16 in Output(t), using the first InputRing slot after drain.` → `Convert final F to BF16 in Output(t).`；slot0/BF16 view与drain→publish仍在结构中，CTA发布与容量语句仍保留。
- `The single staging buffer is shared by all row writers and its reuse permission intentionally precedes completion of the destination writes.` → `Bind source-read completion separately from destination completion for the shared staging buffer.`；合法非空reason，完整release保证仍在verb。

64项共保守候选节省9863声明字符：W/T/R 7487，数学1386，dispatch.when 518，其余Contract472；全部为英文YAML，dispatch.md未提出删除候选。该量约为YAML开放声明字符的9.7%，未经独立短稿等价审阅，不能称已安全删减量。

边界核对：Contract M>=1，完整请求不是全空，但允许空group及C>W的空worker；prologue/teardown仍发生，steady零次。KB在契约中正且偶，V0没有需保留的奇末块分支。copy prefetch末两次空commit仍有issue/done；sync drain后才把input slot0解释为BF16，store完成的CTA join保护下一tile。UMMA的H以下越全局M零填充，H以上保留旧内容；只写V行。TMEM归还须所有行、所有列组读完，且早于最后register FMA。full→tail→full时bulk-store历史不清，下一full仍等之前源读取，最终drain等目的写；不能用store.done代替release而丢重叠。graph capture仅记录，执行用录制host参数和当前tensor内容；缓存策略不随当前内容重算。恢复各phase只读原输入、独立初始化F、全部覆盖out，无前次output输入。

每节点覆盖注记详见annotations.node_review：root paired+NI1/2；sync单role+alias；V0双提交r0/1；wide F/G参数化；wide恢复FF/GG/GF、GGF/GFF、GGFF分别覆盖2/3/4实际提交，所有Realization item列表均核对到各phase。

研究限制：未运行GPU或误差/性能复测；没有API错误注入证明所有L序列可达或成功；source TMEM行地址遗漏已核对地址与row生成，但没有另行展开目标ISA lane规则或验证修复。硬件Contract无绑定，无法据YAML独立证明原语支持或容量驻留。误差公式/NaN策略未知照实保留。原schema、规范、AGENTS.md、Runner状态均未修改。
'''
(B/'operators/extract_gemm.md').write_text(text)
print('report chars',len(text))
