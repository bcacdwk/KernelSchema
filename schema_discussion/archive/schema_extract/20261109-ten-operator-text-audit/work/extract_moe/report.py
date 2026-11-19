import pathlib,json,collections
B=pathlib.Path('schema_extract/archive/20261109-ten-operator-text-audit');W=B/'work/extract_moe';A=json.loads((B/'annotations/extract_moe.json').read_text());S=json.loads((B/'data/extract_moe/audit_summary.json').read_text());I=json.loads((B/'data/extract_moe/summary.json').read_text());N=json.loads((B/'data/extract_moe/nodes.json').read_text());M=json.loads((B/'data/manifest.json').read_text())['operators']['extract_moe'];C=json.loads((W/'context_records.json').read_text());U=json.loads((W/'indexed.json').read_text());D={n['node']:n for n in N};cm={x['node']:x for x in C};root=D['root']
assert S['valid'] and S['finding_count']==25
lines=[];add=lines.append
add('# extract_moe 全量文字审计（v1.1）\n')
add('本算子的现有设计**依赖开放正文确定核心行为**，但不能把所有正文都当成语言缺口。W/T/R 中有 600/992 个精确文本组含已确认的非公式核心信息；另有实现原语绑定的核心重要性未获独立证明。分块、预分配、launch/handoff、环槽与大量 reuse 边已有结构。主要剩余文字负责复合动作完成承诺、空动作、跨调用存储与主机服务；具体全设备 host 完成事件是本轮提出的结构缺口候选。\n')
add('审计读完 2555/2555 精确文本组、106/106 创作 YAML 的有效组合及全部 dispatch；另读 5 份无创作结果骨架、列出 43 个未启动节点。没有抽样、未审默认标签或按关键词分类。人工分批记录在 [manual.json](../work/extract_moe/manual.json)，覆盖与 source 固定身份见 [coverage_complete.json](../work/extract_moe/coverage_complete.json) 和 [source_identity.json](../work/extract_moe/source_identity.json)。\n')
add('## 材料与口径\n')
add('固定范围为 pause inventory 当前计划：96 approved、9 review_candidate、1 needs_authoring，均为语义样本；其余 48 个不进入简洁/密度计算。106 个 meta_data.source 全部绑定 `SchemaExtractSource_c98aa54e58ac@ce1b7d5e8e7560b5c80abd87549bc2328c52df34` 的 `candidate/solution.py::Model.forward`，其 `_SRC` 指向同快照 `nvfp4_moe.cu`。源在 `.schema_extract/v2/snapshots/c496cab3…/source`，复制到 work 的 solution/CUDA/reference/metadata 字节均通过 inputs.lock alias 的 SHA256 核对。未使用当前树不匹配的源码行号。\n')
add('已读冻结 README、DOC0、DOC1、DOC3–6、REVIEW 和 examples README。硬件区为明确的 skip-hardware 空值骨架（root:197–208），没有 hardware 开放文本组；它不提供完整硬件能力证明。approved 是冻结状态，不是本轮 GPU/数值或动态全域验证。\n')
add('字符按统一空白归一化的 Python len 计；字符串总量含公式、引用和标识。英文 YAML 与中文 dispatch 分列。声明量每个源标量计一次，alias 实例展开量另列；from_parent 的有效视图不与 raw 相加。精确相同文字按清单去重读取，但另读 166 种局部结构、73 种 flow、289 种数学属性定义及全部使用节点。词级差分包可完整还原原组，不作为语义相等证明。\n')
add('| 范围 | 精确组 | 去重字符 | 声明数 | 声明字符 | 含非公式核心组 | 保守短写候选字符（声明量） |\n|---|---:|---:|---:|---:|---:|---:|')
for label,key in [('YAML 全字段','yaml'),('W/T/R 主研究范围','main_WTR'),('math','math'),('dispatch.when','dispatch_when'),('dispatch.md（中文）','dispatch_md')]:
 x=S['scope_statistics'][key];add(f"| {label} | {x['unique_units']:,} | {x['unique_chars']:,} | {x['declarations']:,} | {x['declared_chars']:,} | {x.get('nonformal_core_bearing_units',0):,} | {x.get('candidate_saved_declaration_chars',0):,} |")
add('| hardware | 0 | 0 | 0 | 0 | 0 | 0 |\n')
add('W/T/R 的 600 组对应 3642 次创作声明、1,550,479 字符的**字段面积**，不是不可删字符数；MIXED 中的重复片段仍可能可缩。全 YAML 含核心的非公式组为 851，dispatch.md 另有 7 组。93 组/494 次声明的核心重要性为 uncertain；UNRESOLVED 组、未决上下文和未决声明均为 0。这里的 0 只表示分类/绑定已完成，源缺陷和未知验收并未消失。\n')
add('| raw/effective 量 | 全 106 节点 | root |\n|---|---:|---:|')
for label,key in [('全部字符串声明字符','raw_string_declaration_chars'),('全部字符串 alias 实例字符','raw_string_instance_chars'),('开放字段声明字符','open_text_declaration_chars'),('开放字段 alias 实例字符','open_text_instance_chars'),('有效 Design 字符实例（独立口径）','effective_design_string_instance_chars')]:add(f"| {label} | {sum(sum(n.get(key,{}).values()) for n in N):,} | {sum(root.get(key,{}).values()):,} |")
add('\nraw pipeline.verb 声明 1,158,927 字符、alias 实例 2,162,721 字符；相差 1,003,794 是使用位置展开，不能说成又写了一百万字。实际本地 provider 数为 math 86、W 94、T 93。root 有 184 个开放文本组，其中 W/T/R 48 组、26 组含非公式核心；大树 W/T/R 的 600 组不能用重复字符压倒这个 root 基线。`no_scan_fused_small_cp` 的 raw W/T/R 没有独占核心组，但有效视图引入其父的动作定义；这正说明不能按空 raw 区断言设计更简单。\n')
add('## 全字段分类\n')
add('| 类别 | 精确组 | 去重字符 | 声明数 | 声明字符面积 |\n|---|---:|---:|---:|---:|')
for cat in ['DUPLICATE','NONCORE','STRUCTURABLE','ESSENTIAL_TEXT','FORMAL','SOURCE_LIMIT','EXPLANATION','MIXED','UNRESOLVED']:
 x=S['category_statistics'].get(cat,{});add(f"| {cat} | {x.get('unit_count',0):,} | {x.get('unique_chars',0):,} | {x.get('declarations',0):,} | {x.get('declared_chars',0):,} |")
add('\n每组的 reason、原始行号、finding 和 occurrence 清单位于 [annotations/extract_moe.json](../annotations/extract_moe.json)。NONCORE 为 0 不表示所有源码细节都核心：布局/API等内容大多与必要事实混在同一字段，归 MIXED，或列为核心重要性 uncertain；没有把混合字段整体判为可删。\n')
add('## 可复核判断\n')
add('| finding | 涉及组/实际 YAML 节点 | 判断（集合可重叠，不能相加） |\n|---|---:|---|')
for f in A['findings']:
 add(f"| {f['id']} | {f['unit_count']}/{len(f['nodes'])} | {f['claim']} |")
add('\nF05 的 when/索引集合包含中文说明，节点数排除了 `dispatch.md` 这个文件标签。F15 是广泛的主机服务相关正文集合；F25 才是具体全设备 host 等待候选（84 个相关字段、30 个初始化节点），并非这 84 个字段的每个字符都缺结构。所有完整 unit ID、节点、反例及已有字段写法在机器明细。\n')
add('已有结构足以承载的例子有两类：BN2 固定值可落在 `phase.fc2.decomposition.partition.hidden.tile: 128`，数学符号用现有 `binding: bind:design.work_schedule::fc2.decomposition.partition.hidden.tile`；6 个纯计算的自定义 scan/check 可改成通用 `compute:<已定义结果>`，保留 serial 与后续发布。两个独立副本 [bn2_candidate.yaml](../work/extract_moe/bn2_candidate.yaml) 和 [generic_compute_candidate.yaml](../work/extract_moe/generic_compute_candidate.yaml) 均通过 `check_yaml.py --standalone --skip-hardware`。这只是结构验证；若在 raw 上落实 BN2，需要同步受影响的所有 W provider，不能让子节点保留 `tile: BN2` 而形成循环。\n')
add('表达成本的例子是全局别名与生命周期：片上 TensorStorage/ScaleTM、GroupTable 的共址已经用 `resource.view/part/slot_index` 表达；R 容量/M 有效视图、Counts/Fill 一次清零、两个 verdict/output 版本和模块表跨调用存续仍有正文。已有 `retention` 能表达逻辑保留，不能把它误当成分配器或地址模型；全局 allocation/view 的紧凑复用是候选，不是已证明必须扩语言。\n')
add('具体缺事件候选是 `torch.cuda.synchronize(device)`：`tables.kernel_done` 只约束本请求中的 kernel；在其他 stream 有长任务时，它不表达 host 还要等整个 device 才发布 ready-set、继续配置与 probe。把该 API 换个封装不会改变核心，取消这条 host 完成边界却会改变主时序。这是 F25 的反例。当前仍可在 strategy/choice 合法保留，尚未设计或采用新字段。\n')
add('有意开放的核心包括原语组合的 done 承诺、可选输入/空动作、零状态初始化与复用、四个 epilogue 读者全部退役、bulk source-read 与 global-completion 分离，以及 last-arrival 的可见性范围。现有 pipe/repeat/边已经表达大部分关系；没有把换一种 barrier API 当作核心字段缺失。原语名与 ISA 兼容绑定单列 uncertain，不能用它们证明需要新语言。\n')
add('必须保留的边界：Clean=1 时有的 bounded repeat 不产生动作，有的已列动作保留 no-op issue/done；J=0 与 J>0 但 worker 无 item 的 TMEM 分配/回收不同；三槽 payload 与两槽 scale 跨 work 接续，flat FC2 另有 metadata ring 与 Pending；空 copy-group 仍参与 wait_group 选择；split activation 在 expert 检查前读 R-row partial 尾部，不能补零；未消费的 3072-byte 格式表仍是实际工作；纯录图不产生执行事件；BeginCapture 失败后的未捕获私有流前缀不能抹掉。\n')
add('源限制与设计缺口分开：固定源码确认 probe 的普通同址写、未检查 D2H copy，graph_replay 的无代际 host mirror，以及 BeginCapture 失败仍调用私有流 run_pipeline。verify_only 的 HostObserved=1 可来自旧 GPU writer，不证明当前 Clean 或零输出；F24 还标出 3 个上下文共用的一条 mirror 复用说法需限定 device 顺序。它们是来源/表述问题，不是新增 barrier 或 generation 字段已经被证明必要；本轮没有修源或补造恢复方案。\n')
add('## 保守短写候选\n')
add(f"27 个精确 quote→replacement 候选合并后，去重层面净减 5,950 字符，创作声明层面净减 **75,898** 字符；全部属于英文 YAML，约占开放声明的 {75898/3103483:.2%}。其中 W/T/R 为 72,097，math 765，when 111，其他为 metadata/Contract。approved 68,239、review_candidate 7,166、needs_authoring 493。没有计入 alias 展开或 from_parent 有效视图增益；不是独立认证的最终安全删除量。\n")
add('候选保留必填字段和执行事件。例如：\n')
add('- `Drain(t,q) completes after the four explicit prior-work accumulator releases. q identifies the one noninitial-work wait; FC1 performs this wait before its first scale copy.` → `Wait for the explicit predecessors.` 保留长形式的 `execution_mode: serial`、q 域、pipe 位置和四个 release 前驱；不删除整个 wait 定义。')
add('- `One worktile covers all 256 byte codes and writes every table entry.` → 删除这条可选 strategy；公式、partition/target、实际 table launch 保留。')
add('- 任务 describe → `MoE with precomputed routing, prepared NVFP4 weights, gated activation, activation QDQ and BF16 output.` 同时用于全部 106 节点，保持必填与同名描述要求。')
add('- tolerance 仅缩写重复措辞，保留 allclose 参数次序、两组 mismatch 阈值、max_rel_l2、输出别名检查身份、全部未提供定义及不采用 schema 默认公式的声明。完整短稿见 JSON。\n')
add('Supervisor末轮复核收紧了7组gather strategy（63次声明）：Offs输入只有prefix:1，无offset，相应gather动作也未另行明确Offs[E]。M=Offs[E]的数学等式不足以替代这次读取映射，因此撤回整段删除，保留读取句及各组的Clean/guard限定，只压缩flatten、128分块、空域和裁剪等重复说明。分类从DUPLICATE改为MIXED；声明级净减量较上一版减少3374字符，去重级减少504字符。核对记录见 [gather_offset_qa.json](../work/extract_moe/gather_offset_qa.json)。\n')
add('没有按长字段比例扣减。特别是 106 个 when、last-arrival/emit 合约和 host service 长段仍含必要信息；MIXED 面积不可整体算作可删或必需。\n')
add('## 每节点覆盖\n')
add('下表逐一列出全部 154 计划节点。`self` 表示本地 provider；M/W/T 为三个实际 Design provider；所有有创作内容节点都已核对当下 when、完整有效 Design 与本地 Realization。每个角色/资源/模式/索引/边和 flow 的结构明细在 [context_records.json](../work/extract_moe/context_records.json)、[phase_structures.json](../work/extract_moe/phase_structures.json) 和 [flow_groups.json](../work/extract_moe/flow_groups.json)。\n')
add('| 节点 | 状态 | 版本 | M / W / T provider | 开放声明字符 | 覆盖 |\n|---|---|---:|---|---:|---|')
for n in M['nodes']:
 node=n['node']; status=n['category']
 if node in D:
  d=D[node]; providers=' / '.join('self' if v==node else v for v in d['effective_provider'].values());count=sum(d['open_text_declaration_chars'].values());note=f"已读；{len(cm[node]['phase_groups'])} phases";version=d['version'];link=f'[{node}](../inputs/extract_moe/{node}.yaml)'
 else:
  providers='—';count='—';version=n.get('node_version') or '—';note='无创作结果骨架；空 when，三域 from_parent' if status=='attempted_without_authored_result' else '未启动；仅检查计划/dispatch覆盖';link=node
 add(f'| {link} | {status} | {version} | {providers} | {count} | {note} |')
add('\n## 验证与限制\n')
for limit in A['limits']:add('- '+limit)
add('\n统一汇总命令：`python3 -B scripts/summarize.py --operator extract_moe`（在审计根目录运行）。结果 VALID：2555/2555 组、106/106 语义节点，0 格式/覆盖错误、0 未决上下文；93 组核心重要性 uncertain。完整准确统计见 [audit_summary.json](../data/extract_moe/audit_summary.json)。输入 YAML、规范、Runner、AGENTS.md、kernel 与运行状态均未修改。')
(B/'operators').mkdir(exist_ok=True);(B/'operators/extract_moe.md').write_text('\n'.join(lines)+'\n')
print('report lines',len(lines),'bytes',(B/'operators/extract_moe.md').stat().st_size)
