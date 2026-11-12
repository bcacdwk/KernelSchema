# Phase 公共定义：调查口径

本轮研究 docs5 的 resource、domain、milestone：片上存储占用、共享索引、供跨 phase 引用的事件组。目标是用六算子材料判断表达是否足够、是否有歧义、是否记录过细，并给出更直白的英文名称。这里只形成研究和建议，不修改正式规范、算子、模板或提取状态。

## 输入与版本

- 当前正式规则以本目录 sources/rules/ 的快照为准，来源是本轮开始时 schema_docs 的 0–6 篇，包含 v1/v2 及已接受的后续文字整理。
- 主语料为 2026-11-09 暂停时的六算子当前稿。复用 schema_extract/archive/20261109-ten-operator-text-audit/data/manifest.json 和其中 inputs/extract_* 的固定副本；根文件与原 working checkpoint 的字节一致性由本轮核对。不要拿 initial 的旧 FMHA/MoE 发布稿代替当前稿。
- corpus.json 给出全部节点、状态、路径和摘要；六个 root 是主样本，已有分支是反例与变体来源。草稿、空架子和未开始节点分开计数。旧语法样本不被视为当前规范的合格实例，也不因字段改名直接判作表达能力缺口。
- 对照资料：SCHEMA_VERBOSITY.md 和 SCHEMA_DISCUSSION.md，完整位置见 corpus.json。已有报告是调查线索；关键结论仍定位原 YAML 和相关源码。
- sources/kernel/ 只在需要补齐原研究未公开保存的固定源码时使用。其余复用 corpus.json 所列的历史源码副本。所有路径必须指向固定、可核对的内容。

## 问题分类

每个重要发现归入下列一种主类，并说明是否涉及另外两种公共定义：

1. existing_structure：现有字段或有序列表/for/if/waiting 可以表达，只是旧稿没有使用。
2. concise_semantics：属于设计级的必要语义，用一个短 describe/strategy 足够，不需要新增字段。
3. underspecified_rule：已有概念足够，但范围、引用、作用域或格式不清，需要补清规则。
4. missing_capability：应由本层承担的核心关系，无法用现有结构清楚表达；必须给出最小反例。
5. other_layer：真实需求属于数学、调度、外部调用约定或 realization，不由三类 phase 定义兜底。
6. implementation_detail：删去后不影响数学、工作量、存储量、交接、依赖、可见性或可重叠关系，可交给 Coder。
7. source_or_draft_issue：来源本身缺保证、材料未定或旧稿错误，不能靠语法补出不存在的正确性。

自然语言不等于缺口，字段缺席也不等于多余；不得把这些类别混为一谈。提出删除时说明保留了哪些有意义的事实；提出结构扩充时说明现有组合具体卡在哪里。

## 独立调查与交付

五个全新 gpt-6-astra、max effort 子代理按不同角度调查；依根 AGENTS.md 同时最多两个子代理，分批运行。每位先独立形成结论，不读取其他代理的报告。主代理负责共同语料清单、机械统计、证据复核与最终综合。

每位只写分配给自己的 reports/NN_*.md、data/NN_*.json 及必要的 scripts/NN_* 辅助文件。不要修改其他报告、README、METHOD、共同 corpus 或正式资料。

报告要求：

- 明确实际读过的六 root/分支/源码路径，不把词法扫描数声称为完整语义审阅数。
- 提供六算子小表；每个关键问题引用文件及行号，说明根稿/分支及其状态。
- 选少量真实片段做最小改写，展示字段保留、删减、改名或必要扩充的效果。草图明确标为建议，不能暗中当作现行规范。
- 给 resource/domain/milestone 各自独立的英文命名建议及理由；名称简洁、惯用、读后能猜到作用，不为了整齐造成误读。
- 不按篇幅推定好坏，不把 alias 或 from_parent 展开量当作作者重复输入量；对没有发现必要用例的字段只下有限范围结论。

数据文件使用顶层 coverage、findings、naming、sketches。每条 finding 至少含 id、operator、category、claim、recommendation、evidence；evidence 为 path、start_line、end_line、quote 的列表。路径相对于仓库根，quote 须为对应片段的原文子串。新增源码证据另记录 SHA256。若需内部统计，给出脚本或清楚算法与分母。

本研究不运行 GPU、不继续或发布已暂停的提取任务，不假定源码里所有实现细节都应该进入 schema。
