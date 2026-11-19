# Ten-operator text audit — common task v1.1

你是本轮一个算子的独立研究审计者，effort=high。Supervisor 会指定 OPERATOR_ID。仅审计这个算子的全部现有 YAML 和 dispatch，不再启动子代理。每轮两个新代理，共五轮。你的任务不是证明“文本都可删”，也不是证明“规范没问题”，而是用完整清单和可复核证据区分事实。

## 目标与权限

判断：哪些文本是冗余/非核心，哪些核心事实可用现有结构表达但被作者写进文字，哪些必要核心语义没有合适的现有结构表达。特别区分“现有稿只在文本中定义”与“现有结构确实无法/很难表达”。不能以规范允许自由文本为由宣布结构已经完整，也不能把数学公式是字符串当成语言漏洞。

只写本审计目录下 `operators/OPERATOR_ID.md`、`annotations/OPERATOR_ID.json`，以及必要的 `work/OPERATOR_ID/` 临时脚本/证据。输入、原仓库规范、原 YAML、AGENTS.md 和运行状态只读。不得恢复 extract、修改 kernel、运行 GPU、发布或启动另外的模型工作流。本任务已由用户授权，不请求额外确认。根 README 的用户改动只读。

## 固定材料

审计根目录：`schema_extract/archive/20261109-ten-operator-text-audit/`。

- `data/manifest.json`：样本、状态、原路径、冻结副本及 SHA256。
- `inputs/OPERATOR_ID/*.yaml` 与 `dispatch.md`：本次选定的完整原字节。模板四项取当前 examples；六项取 2026-11-09 pause inventory 的 current-plan 最新选定稿，不能混入最初 2.2.2 全套历史稿。
- `data/OPERATOR_ID/units.jsonl`、`TEXT_INVENTORY.md`：全部开放文本位置的精确文本分组。分组键为字段族＋归一化文本，不是语义等价证明。
- `data/OPERATOR_ID/scalars.jsonl`：全部字符串值，包括表达式、引用、标识符，供检查自动文本范围有无漏项。
- `data/OPERATOR_ID/nodes.json`、`summary.json`：统一机器统计，直接复用；有疑问报 Supervisor，不自行改口径。
- `rules/README.md`、`rules/schema_docs/*.md`、`rules/schema_extract/REVIEW.md`、`rules/schema_template/examples/README.md`：本次冻结的目标与字段语义。优先读 README 目标/原则、DOC0、DOC4/5/6，数学/Contract 问题再读 DOC1/3。

根据 `meta_data.source` 定位实现；当前模板的旧源可用 `git show 01fff90d714abc6e84904ea9c44523ba66d34a0e:<path>` 只读获取。当前源码可能已删，不等于来源失效。六项源码须核对 source commit/snapshot；不能使用不匹配版本的行号。规则分析用本次冻结规范，历史获批不代表符合当前更强的“结构承载核心”研究目标。无法取得或没有亲读的证据要直说。

前轮 `SCHEMA_VERBOSITY.md` 与 `VERBOSITY_FACT_AUDIT.md` 仅为背景，不是待复述的结论，不代替本算子全量审计。

## 统计口径（不可自行变更）

1. 字符为 `len(re.sub(r"\s+", " ", value).strip())`，非 token。字符串总量包括公式/引用，不能叫纯自然语言字数。中英文 dispatch.md 与英文 YAML 分列。
2. **声明量**：每个 YAML 中同一源标量/alias 定义计一次。分开声明的相同文本仍计多次。
3. **实例量**：alias 每个使用位置都计。有效 Design 另展开 `from_parent`，只读分析，不与 raw 相加。
4. **精确文本组**：整个算子内同字段族＋完全相同归一化文字读一次主定义，但须检查每个使用节点/作用域。文字相同不证明上下文相同，存在不同判断时用 occurrence override。
5. 当前六项区分 approved、review_candidate、needs_authoring；not_started 和 attempted_without_authored_result 只列覆盖缺口/骨架，不能拿它们算简洁。模板仅表示规范化示例，不能叫 GPU/语义已验证基线。root、全节点、去重复组都分别报告。
6. 全字段分类的字符数是“处于该类字段中的字符面积”，不是可安全删除字数。MIXED 不能整体算可删或必要。可删量只来自逐条、有据、可定位的精确 quote→replacement 候选，按合并区间避免重叠；没有短稿等价审阅，最多称保守候选量。

v1.1 在第一轮完成前统一补入硬件 execution 定义及 compute/rewrite 的 name/step。既有文本 ID 和类别不变；这些可能是开放的组成语义、标题或正式标识，须逐项判断，不能全称为自然语言。两个首轮代理补审 added_units，随后八项从同一完整范围开始。

## 必须做的全量工作

1. 列出全部计划节点的状态；阅读全部现有有效 Design、Contract、Realization 和 dispatch。相同文本可去重阅读，但每个本地不同定义、选择条件、引用绑定、角色/资源/事件组合都要覆盖。检查 scalars 中开放文本目录是否漏掉重要正文，若有向 Supervisor 报告。
2. 对 `units.jsonl` **每一个 id** 提交人工语义分类。禁止只挑最长字段，禁止按关键词/长度自动分配“必要/冗余”。可用程序写出人工选定的 ID 分组，不能用未审阅的默认标签掩盖缺项。确实无法判定的留 UNRESOLVED 并说明缺什么。
3. 同一句兼有数学、设计事实、重复说明时标 MIXED，再用局部精确 quote 分析。不得把尚有核心定义的整段叫“可删”。
4. 对所有疑似结构缺口/改写结构/已有字段未用的现象建立 finding，而不只汇报最显眼几例。重复机制可合并 finding，并列全部涉及 unit IDs 和节点；比较覆盖范围与发生频次。
5. 对最有价值的删减给出原 quote、候选 replacement（删除为空串）、依据和适用 occurrence；对结构化建议给出实际已有字段/事件/表达式的短写片段。新增字段或一个未定义动作名不算“现有语法已解决”。
   把同一长文搬到另一个 strategy/choice 也不算结构化。必填字段不能直接删成空值；整段删除须说明该字段可合法省略，否则给出仍合法的短替换。候选不得因压缩取消原有事件或加强顺序。
6. 报告空分支、首次/末次、跨工作、所有读者退役、别名存储、跨调用与捕获/回放等边界。不是要求把这些审计过程重新抄进 schema，而是避免删掉真正的核心事实。

## 全字段类别

每组只有一个主 category；它是整个字段的分类，不是细句的分割。可用 findings 和 removal_candidates 表达局部。

| category | 定义 |
| --- | --- |
| DUPLICATE | 全部设计含义已由当前结构、上游定义或共享规范覆盖；需给唯一来源，不能仅因其他文件文字相同判可删 |
| NONCORE | 全段为不影响既定核心设计的实现/过程/当前选择域外内容；需说明为何无数值、执行或主要性能影响 |
| STRUCTURABLE | 核心含义当前在文字，但整项已有可行的现有结构/表达式写法；finding 必须给出写法 |
| ESSENTIAL_TEXT | 当前文字承担不可丢的核心事实，尚无合适现有结构解法；须说明是有意开放语义、表达成本问题还是缺字段/事件，不能自动等同“已证明需要扩语言” |
| FORMAL | 数学公式、参数定义、选择谓词等正式关系，不能把保留这些内容当成不压缩。夹带重要叙述时标 MIXED |
| SOURCE_LIMIT | 来源、误差/验收未知、实现偏差的必要限制；不许一律删，也不把缺陷详证扩写为新设计 |
| EXPLANATION | 摘要、理由、导航等解释用途，没有独占核心事实；有阅读价值不等于应删除整个字段 |
| MIXED | 同时含不同类别的重要片段；指出哪些独有、哪些重复/可迁移 |
| UNRESOLVED | 尚不能判断，说明未解决的具体证据或绑定，不伪装成已证明可删/不可删 |

`core` 为 `none / some / all / uncertain`：是否含删后会丢的核心事实。
`structure` 为 `already_present / existing_structure / open_semantics / new_structure_candidate / formula_or_predicate / mixed / not_applicable / uncertain`。

“核心”指既定数值语义、工作/计算分配、依赖与发布、状态/资源容量及复用，或主要性能因果。普通 Coder 可探索的 lowering/layout/API 细节不自动是核心。删句后若会容许在这些方面不同的设计，给出最小反例；如果变更只影响审计解释，不能借此声称核心丢失。

首轮校准：同一依赖图换成不同barrier/通知API，并不自动证明缺核心字段。须说明是否增加不可由已有关系推出的等待/发布，或提供有据的主要成本影响；否则保留协议来源事实，把其核心重要性标uncertain，不把实现差异直接统计为已证结构缺口。

## 机器交付：annotations/OPERATOR_ID.json

```json
{
  "operator": "OPERATOR_ID",
  "prompt_version": "v1.1",
  "unit_annotations": [
    {
      "id": "t_...",
      "category": "MIXED",
      "core": "some",
      "structure": "mixed",
      "reason": "简短但具体的判断，可引用同一 finding",
      "evidence": ["inputs/OPERATOR_ID/root.yaml:123", "rules/schema_docs/5.design.tile_pipeline.md:96"],
      "finding_ids": ["OPERATOR_ID-F01"],
      "contexts_checked": true,
      "occurrence_overrides": []
    }
  ],
  "findings": [
    {
      "id": "OPERATOR_ID-F01",
      "kind": "redundancy",
      "unit_ids": ["t_..."],
      "claim": "可审计的判断",
      "core_fact": "必要时填写",
      "deletion_counterexample": "删除后两个不同设计如何都满足剩余结构，或不适用原因",
      "existing_structural_candidate": "现有字段短写，或未找到及排查过什么",
      "gap_kind": "none",
      "evidence": ["inputs/OPERATOR_ID/root.yaml:123"],
      "confidence": "high"
    }
  ],
  "removal_candidates": [
    {
      "unit_id": "t_...",
      "quote": "必须是归一化 text 中恰好出现一次的完整子串",
      "replacement": "删掉时为空串；否则给英文短写",
      "kind": "redundancy",
      "reason": "说明含义由哪里保留",
      "evidence": ["inputs/OPERATOR_ID/root.yaml:123"],
      "applies_to_occurrences": "all"
    }
  ],
  "node_review": [
    {"node": "root", "reviewed": true, "effective_binding_checked": true, "notes": "简短的节点覆盖说明"}
  ],
  "limits": ["明确而具体的研究边界"]
}
```

finding.kind 为 `redundancy / noncore / existing_structure / structural_gap / source_limitation / correctness / style`。
gap_kind 为 `none / open_semantics / representation_cost / missing_field_or_event / unresolved`。
`occurrence_overrides` 必要时填 `{occurrence_id, category, core, structure, reason, evidence}`；默认判断不得掩盖上下文不适用。`applies_to_occurrences` 可填明确 occurrence_id 列表。

若 quote 太短多次出现，请扩大完整句子，不随意指定第一次。所有省略结论和段落改写必须有据，不以字符节省代替语义论证。

## 人读交付：operators/OPERATOR_ID.md

简洁说明样本/状态/源与规则身份，列统一统计、每个节点的覆盖和类型、全字段分类结果、全部去重机制的 finding、结构缺口候选及现有语法替代表达、若干有据短写、未决项。正文可以引用机器明细避免复制所有原文，但数字必须能回到 annotations 和 census。

特别回答：

- 本算子是否依赖自由叙述才能确定核心设计？具体依赖哪些信息，覆盖多少字段/节点？
- 哪些是已有字段未使用，哪些只是缺少紧凑复用/成员事件表示，哪些是字段/事件语义确实缺失，哪些是公式/实现原语绑定等有意开放的内容？
- 明显可删/可缩内容的保守数量；哪些长字段仍含必要内容，不能按长度截断？
- root 与分支、raw 与有效视图、alias 与跨文件复制有何差异？不能以大树的重复字数压倒其他算子。

初次汇报在完成 inventory 阅读/范围核对后给 Supervisor；完成后运行 `python3 -B scripts/summarize.py --operator OPERATOR_ID`（工具若尚未提供先通知 Supervisor），修好格式/覆盖问题。最终发 <=1200 中文字符结论、报告路径、准确数量和最重要的限制。任何中途上下文不足均保存本地审计进度后继续，不把抽样冒充全量。
