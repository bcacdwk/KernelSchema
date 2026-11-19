# 审计范围与计量方法

本研究响应用户要求：四个现行 template 算子、第一批六算子现有结果及全部 dispatch，按每轮两个新的 high 代理、五轮完成十份独立报告。研究目标是识别必要自由文本、已有结构未使用、结构表达缺口和冗余，不是执行提取或设计新的精简 agent。

**固定样本。** 模板取 `schema_template/examples` 当前四任务，共21 YAML。六算子取2026-11-09暂停清单的当前计划与最新选定安全稿：GDN6、GEMM7、MLA2、Sparse8、FMHA43、MoE154个计划节点。六项中172份有创作内容，另5份只有失败调用留下的骨架、43份未开始；骨架与未开始项不参加语义密度统计。全部193份有内容YAML和10份dispatch按原字节放入 `inputs/`，五份骨架放 `unwritten_scaffolds/`。副本仅作研究证据，不是发布或新批准。

六项的145份approved、13份review_candidate、14份needs_authoring分列。approved root的`node.authored`可能为false，因为批准记在stage；范围按批准和实存内容保留root，不以该内部标记误删。旧2.2.2首次交付不混入当前队列，退役节点也不算当前节点。FMHA/MoE的未审或旧版本草稿不被视为符合当前规则的完整成品。

`data/manifest.json`保存原路径、原checkpoint、状态、节点版本、source文件摘要、分析副本与规范快照身份；当前规则包括用户在本轮前写入README的motivation修改，原README不由审计代理修改。旧稿的批准仍按其历史规则理解，本次“结构与共享规范是否足够”的研究标准另行应用。

**机械计量。** 使用 `len(re.sub(r"\s+", " ", value).strip())`。单位是Unicode字符，既不是token，也不是纯自然语言字数。数值型YAML值、键名、缩进和注释不计入字符串值总量；英文YAML与中文/混合dispatch分列，禁止据字符数同比折算模型费用。

| 口径 | 含义 | 不能解释成什么 |
| --- | --- | --- |
| raw字符串声明量 | 每文件的每个源字符串标量计一次；YAML alias不重复 | 不全是叙述，包含公式、引用、动作名 |
| raw字符串实例量 | 每个字段位置计一次，展开alias | 不等于作者实际重复写字量 |
| 开放文本位置 | 文本字段及公式、谓词、名称等开放字符串槽位；族名单由census代码给出 | 不等于所有字符都可删；FORMAL单独分类 |
| 精确文本组 | 算子内相同字段族＋完全相同归一化文字合并阅读 | 不证明不同节点上下文等价，也不证明跨文件副本可删 |
| effective Design | 根据当前选定provider展开三个Design区域，计算字段实例量 | 不与raw相加；只说明本次分析组合，不授予草稿批准 |
| 精确改写候选量 | 明确quote→replacement的净字符差，按适用声明位置计，区间不得重叠 | 未经完整短稿等价审阅，不能称已验证可安全删除量 |

开放文本范围覆盖metadata.describe/dispatch.when、Contract的meaning/range/compute/tolerance/gap、数学的thesis/rewrite/arithmetic/ledger/meaning、工作调度说明/refill/exit、流水线thesis/verb、realization.choice/reason/feedback，以及模板硬件的execution/feature/stack。`scalars.jsonl`保留全部字符串供漏项核查；最终残余非目标长字符串扫描未发现遗漏正文。结构标识和公式不能按语言长度自动判断重要性。

**人工分类。** 每一个精确文本组都有annotation，附理由、当前文件/规范/必要源码的行锚点，并检查各使用节点。完整类别与机器格式见 [共同提示](AUDIT_PROMPT.md)。类别为DUPLICATE、NONCORE、STRUCTURABLE、ESSENTIAL_TEXT、FORMAL、SOURCE_LIMIT、EXPLANATION、MIXED、UNRESOLVED；另记录core与structure轴，允许已理解原句但核心重要性仍uncertain。

分类统计衡量的是“被归为某类的字段数量和字符面积”，不直接分割句子中的信息量。特别是MIXED：一个1000字符字段可能既有关键句也有大量解释，不能将1000字符全算必要或全算冗余。`scope_statistics.main_WTR`统一比较work_schedule、tile_pipeline、realization开放文本，避免数学、硬件占位、不同语言dispatch或大树复制主导结论；`nonformal_core_bearing_*`只表示含有核心内容的非FORMAL字段，并不表示其每个字符不可压缩。

同一事实换名称或换措辞，仍可能分成不同精确文本组；本次不自动做语义等价归并。因此组数不是独立语义事实数，字符面积也不是信息熵。本研究不宣称找到了绝对最短表示。

一组文字在不同节点中可能有不同适用性。组主标签保持主要语义分类，逐声明统计应用occurrence override；另报`groups_with_unresolved_contexts`和`unresolved_declarations`，不能把“主标签只有一个UNRESOLVED组”误读成所有其它上下文都已确定。W/T/R核心组按至少一个实际上下文有核心事实计，声明量逐上下文计；核心重要性uncertain另列。

STRUCTURABLE须给出现有字段、事件或表达式的实际写法；把文字移到另一个strategy/choice不算。ESSENTIAL_TEXT须给出删除后的设计歧义；“规则允许自由文本”不证明结构足够。字段/事件缺口、现有表达过于繁重、有意开放原语语义分别登记。只改变barrier/API而无新增依赖或主要成本依据，不能证明核心设计不同。

单字段相对另一处文字冗余，也不意味着所有文字可同时清空。删改必须保留一个明确的规范定义位置；两处互相引用为删除依据会一起丢失事实。本研究的候选量没有经过完整短稿等价审阅，后续应用时须复核这种跨字段相互作用，不能仅累加局部节省便称设计仍完整。

SOURCE_LIMIT和UNRESOLVED分别保留已知来源边界及尚不能判断的问题；不能因为资料不足就武断删除，也不能把缺资料当成已证明无语法可表达。已有结构检查通过不证明数学或同步正确。逐节点读者退役、首末/空分支、别名、共享状态及录图边界由审计记录覆盖。

**首轮校准与跨轮一致性。** 初始v1清单遗漏了硬件execution与compute/rewrite名称/step；第一轮两代理补审后统一采用v1.1。已有ID不变，仅新增319组，十项最终共5875组；变更详见 `supervision/scope_calibration.json`。随后类别和样本保持不变，提示只增加质量边界澄清。每轮代理名、effort和实际提示摘要记录在 `supervision/rounds.json`。

首轮Supervisor还复核模板GDN的通知流finding：没有不可传递等待或一阶成本证据时，从“缺字段”收窄为开放语义及核心重要性未决。更正保留在对应work目录和报告中。这类复核用于避免“规范允许文字”与“实现有差异所以必须新字段”两个相反的循环论证。

**复核与再生成。** 在仓库根目录运行：

```sh
python3 -B schema_extract/archive/20261109-ten-operator-text-audit/scripts/census.py
python3 -B schema_extract/archive/20261109-ten-operator-text-audit/scripts/summarize.py
python3 -B schema_extract/archive/20261109-ten-operator-text-audit/scripts/aggregate.py
python3 -B schema_extract/archive/20261109-ten-operator-text-audit/scripts/integrity.py
```

第一条从冻结inputs重新生成机械清单，不重新读取运行中的原稿；`--freeze`只用于建立新研究样本，本次复算不使用。第二条检查每组/节点覆盖、枚举、证据文件与行、quote唯一匹配、候选适用范围和非重叠区间，并生成各算子及整体汇总。第三条生成CSV、finding目录和关键合计，第四条复核文件身份与报告链接。这些校验不代替语义审阅。

Supervisor额外将十个root的动词统计与前报告独立基线对比，全部一致；将193节点有效Design统计与生产`resolve_documents().effective_document()`逐项对比，全部一致。结果在 `supervision/identity_verification.json`、`effective_verification.json`。已复制材料的SHA256核对通过。最终交付另外检查报告链接、全局数字及原文件改动范围。
