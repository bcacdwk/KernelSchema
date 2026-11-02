# Schema Extract 框架问题台账

这里留给维护者决策的是提取框架确有必要改进的问题。先核对现象，查现有规则能否处理，再说明哪种错误、遗漏或无效工作会重复，建议能解决它的最小改动。一次算子的实现差异、缺失证明、已定位的转录错误与暂时不便按原流程记入该任务。某一算子也可以提供足够的反例，例如目录导航和明确指定的任务域互相矛盾。

| 待决目录 | 适用问题 |
| --- | --- |
| `schema_docs/` | 规范缺少必要语义、规则矛盾或表述会导向不同的提取结果 |
| `schema_template/` | 检查器误报、漏报或诊断妨碍执行规范 |
| `kernel_wiki/` | 资料的来源、导航、组织或内容需要维护 |
| `extract_workflow/` | 提取步骤、角色交接、证据约定或协作规则需要调整 |
| `extract_tools/` | 提取执行器、读取、恢复、巡检等工具的具体缺陷 |

每个待决问题占一个 `<分类>/<id>.json`。用 `python3 -m schema_extract issues` 查看待决总数、问题与路径，也可直接数五个分类下的 JSON 文件。`archived` 只计入 `archive/` 中已有决议的 JSON。迁移前的原文保留当时的状态、验证与清理记录，其中的相对路径按原位置 `schema_extract/` 解读。

## 提报与入账

1. Worker 和 reviewer 在阶段回执、Supervisor 在只读巡检决定中使用同一 `framework_candidates` 数组，没有合格发现就返回 `[]`。只写已有工作中亲见的具体问题和可复查依据，继续按本职完成任务。保持 worker 的暂存写入边界和 reviewer／Supervisor 的只读边界。
2. 收尾时主 agent 或维护者在清理运行记录前执行 `python3 -m schema_extract candidates RUN_ID`。每个 ID 下，相同提报只展示一次，`sources` 保留全部来源，有新增证据或不同建议的提报分别列出。已登记项显示台账路径和状态，新项的 `registered` 为 `null`。多个 plan 可共享同一 task。Worker 和 reviewer 的来源指向检查点或调用，Supervisor 的来源指向本次巡检决定及输入摘要。
3. 核对材料、必要性和建议范围，在待决分类及 archive 里查相同 ID **和同一根因的其他 ID**。合格的新项直接用内置 `apply_patch` 单独建档，留待决策。同一问题补充原项。复现已归档问题时沿用原决定。确有新证据时在人工决策中明确引用旧决定。没有合格项时结束本轮。普通任务 issues、回执原文和接受的检查点照常留在运行区。

记录字段与回执候选相同。`id` 用简短稳定的小写 ASCII 字母数字 slug，可用 `-` 或 `_` 分词，不带 run、fingerprint 或序号。分类调整时保留身份。例如 `mla-catalog-dynamic-domain` 始终指向 MLA 导航摘要的任务域问题。`category` 为上表目录名，`problem` 描述观察到的具体问题，`evidence` 列出能复查的路径、行号或复现及观察结果。入账时补上已核对的原回执或检查点定位，并保留足以脱离临时目录理解的事实。`proposal` 写可执行的修改建议，`necessity` 说明不改会产生什么实际影响。

## 维护者决定

处理完毕后用一个补丁将原文件移动到 `archive/<id>.json`，保留问题字段，新增 `decision` 对象。待决目录不保留副本。

| `decision.outcome` | 决议内容 |
| --- | --- |
| `declined` 不解决 | `reason` 写决定和不解决的理由。 |
| `resolved` 完全解决 | `reason` 说明实际改了什么和核验依据。 |
| `partially_adopted` 部分采纳 | `reason` 说明取舍和核验依据，`adopted` 写已落实的范围，`remaining` 写未采纳或暂缓的范围及原因。 |

维护者决定是否改框架。提案不计入任务的 `issues`、检查、ready 或 export。
