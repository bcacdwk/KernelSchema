# Schema Extract 执行协议

正式入口为 [`../extract_yaml.py`](../extract_yaml.py)。[Supervisor 手册](../README.md)说明命令，[六阶段](../../EXTRACT_SCHEMA.md)定义工作职责，[REVIEW](../REVIEW.md)定义独立审阅。固定模型指令全部位于 [`prompt/`](../prompt/)；Python 负责动态事实、机械 schema、角色调用、检查、状态和发布。

## 角色与会话

新建 run 默认 `backend=claude`、`model=qwen3.8-max[1m]`，worker/reviewer 的 effort 均为 `max`。显式 Codex 须同时指定自己的模型和两角色 effort。已登记调用、恢复和后台入口核原身份，缺字段拒绝；不猜后端或最近会话。

会话 scope 为 `{stage, node, role, operation}`。每阶段和角色新建，P5 另按节点分开；normal、adjudicate、signoff 分开。仅同一 scope 可精确 resume 已登记 UUID。每 scope 使用固定的真实空 cwd 与私有 CLI home；每个 attempt 有独立 workspace、材料、原稿与完整 stdin，调用前后记录并核验 cwd 为空，上下文不依赖前阶段对话。P6 先审，worker 按需加入。

Supervisor 负责控制请求，Runner 是状态与合入的唯一写者。worker 编辑本次私有暂存目录内的授权候选；仅获本次写作许可，最终文件/字段差异仍须过门禁。reviewer 普通调用始终只读，可以提出退修或受控裁决补丁。两种后端均禁内部委派。

## 阶段决定与产物

| 阶段 | worker 的职责 | 独立放行对象 |
| --- | --- | --- |
| P1 | root 的任务元数据、算法 Contract、有据 Execution 最小登记 | 全部输出及验收依据、任务和执行单元；其余硬件保留 starter |
| P2 | root Source/Dispatch/Version 与目录草案，提交 `{id, instance}` | 机制、具体 root、少量完整代表及记录去处；含 root ≤10 |
| P3 | root 本地 Design 三区、必要 Realization、自身 when/Version | 完整 root 有效全文及当前目录范围 |
| P4 | 目录与 `{id, wait_for}`，落实已定非 root 写作安排 | 完整 root、相关基线与有限未来计划 |
| P5 | 每节点只写自身 YAML | 共享 Contract、实际 raw 来源链、本节点全部使用上下文 |
| P6 | 按 reviewer 明确的非 root 字段修稿或补目录 | 全部最终 YAML、中文目录、实际来源及知识去向 |

P2 和 P4 分别绑定真实 reviewer 回执。P4 不增删选定 ID 或更换实例；每个非 root 恰有一项安排，等待可零项或多项，拒绝自环和循环。P4 不创建占位 YAML，未来文件标明“仅计划”。

P5 首稿从 starter 建自身身份和待写区域；已有 raw 原样续写。内容共享来自实际已审同名整区 `from`。root 保留唯一 Contract 与全本地 Design，when/Version/Realization 为本节点内容。单 root 仍经 P4 和 P6，P5 明确 skipped。

## 提示与材料

`prompts.py` 从冻结 Markdown 装配共同说明、角色、阶段与操作要求，再附本次身份、范围、决定、材料和反馈。`provided_documents` 标记已嵌入全文；实际 stdin 保存在 `attempt/prompt.txt`。固定规则不会藏在 Python 字符串中。

调用材料全部使用真实绝对路径：`workspace` 指本次候选目录，`source_root` 指固定源码，`rules_root` 与 `executor_root` 指冻结规则和执行器，`materials` 指本次 baseline、回执/交接、候选和检查参数。CLI 私有 home 位于 scope 的 `home/`，临时文件位于 attempt 的 `cli-tmp/`。

正式 Source 始终用 `repository_id@ref:relative_path::symbol`。材料清单记录实际路径、固定身份和摘要，不做虚拟路径映射。当前调用只提供其所需材料；这些导航和 CLI 工具许可不是操作系统访问控制。

P6 先独立理解最终成果，再按必要 P2/P4/P5 回执和短交接反查新事实去向。过程记录不补足正文。框架问题沿 `framework_candidates` 提出，reviewer 独立确认，Supervisor 按稳定 ID/根因处理。

## 回执与短交接

模型只返回本次动态 schema 的 `schema_extract.proposal.v1`，含精确 `attempt_id` nonce。Runner 从实际 bytes 生成完整 receipt 的身份、摘要与差异。

- `decision`、`summary`、`conclusions` 表达本次判断；正常 worker 可 complete/continue/blocked，reviewer 可 approve/revise/blocked，适用阶段可 reopen。
- `source_evidence` 选择真实阅读的相对 path、category、行范围、quote_offset 和 note；每段最多 12 行。`artifact_evidence` 锚定本次实际审阅或提交的 raw（普通调用为 `workspace`，裁决 `select` 为所选候选 `materials/candidates/<checkpoint>`），不能用展开视图或规范化预览的位置。
- conclusions/issues/framework_candidates 的 evidence 索引本次 source_evidence；当前文档的四项判断可由 artifact_evidence 支持，纯文档退修/阻塞使用相应 kind 与文件锚点，具体编码见[提交与证据](../prompt/operations/response.md)。reviewer 仍独立核查实际源码。
- P1 comparisons 绑定逐输出验收；P2/P4 普通或维护 reviewer 独立认可后，approve 的 plan 保留 plan_draft；P6 revision_scope 限目录及明确非 root 字段，affected_nodes 登记受影响者并在批准时覆盖 pending_rechecks。
- checks 只记亲自执行的命令及退出码；未执行时为空并说明。handoff 简短记录已定事项、必要依据、新事实去处与缺项。

提案形状、身份、来源和产物问题分别记录。格式修复不授权改模型身份或伪造源码证据，缺失的语义判断仍须由本角色完成。

## 规范化与检查

候选按同一事务处理：原稿取证 → Template 整体规范化候选 → 文件/字段权限与来源比较 → 受控副本落地 → 严格重读与 checkpoint → 独立 review。

原稿 bytes、原回执和行号保留；validation 连接原稿、corrections、最终摘要及新 checkpoint。模型格式瑕疵由 Template 统一处理；纯格式变化不伪造 Version 增长。越权或关联修正不能完整落地时，整体拒绝并返回所属阶段。批准后的 raw 不再隐式规范化。

| 场景 | 机器检查范围 |
| --- | --- |
| P1 | 适用算法/结构、来源与阶段权限，不要求未来 Design |
| P2/P4 | 选择/安排、现有/未来文件、中文目录和合法链接 |
| P3/P5 | 受控完成快照及解析必需来源的严格 Template 检查；记录 checked/excluded |
| P6/发布 | 全部直属 YAML、`dispatch.md`、完整实际图，且直属 YAML 集合精确等于 `selected_designs` |
| standalone | 用完整目录 API 检查来源目录全部直属 YAML、`dispatch.md` 与完整实际图，并严格校验导出单文件；不接收运行 `selected_designs` |

`invocation.check_command` 指向本次固定 helper，不接受额外参数。worker 获得规范化预览，reviewer 检查当前 raw；输出区分 raw_passed、preview_passed 和 pending_corrections。模型手动用 Template CLI 时必须显式 `--read-only --skip-hardware`。预览通过与最终 raw 已通过分别记录。

## 并行合入与计划修订

P2 选择、P4 等待与 P5 实际来源分别保存。wait_for 只约束本实例、本等待集合的首次派发准入；同一安排下的续写、复审、退修和恢复沿用已通过的准入，实例或等待集合改变时重新核验。实际 `from`、origins、共享 Contract 和使用上下文决定内容依赖与修订影响，实际来源批准持续核验。

`source_manifest(node)` 包含 root、自身及同名区域实际链。为解析 provider 其他区域而补读的材料记录在解析闭包中，不自动成为该节点的采用来源。跨区环由完整图检查。

合入只采用目标 raw，再对最新 merge 重算实际图。真正采用的来源须在调用时已提供、已审且当前仍适用；相关失效结果退休，无关兄弟完成可以并存。P3/P5 的阶段批准核职责投影、实例、generation 和实际来源，不用早期全文 SHA 禁止后续合法写作。

来源完整 SHA 变化时核当前 identity。只有严格证明实际定义、Contract 和完整使用上下文不变，才在既有 validation 留下等价依据并保留适用性；原批准 identity 不改。否则保留草稿并独立复审真实使用者。

## 定点返回与一次裁决

普通记录由 P3 交 P4、P5 交 P6 汇集；当前节点本地化或换用已审整区沿 P5 全文审阅。已定代表的登记/等待改变回 P4，具体事实表明机制或取舍错误回 P2，共享 Contract 错误回 P1。provider 定义错误回实际定义的 P3/P5，consumer 绑定不适用由 consumer 修正。已有固定源码的具体误读同样可以返回。

稳定 issue、证据、原预算、草稿与未受影响批准保留。真实任务或固定来源改变另建运行。返回和恢复不重授已用额度。

普通退修达到有界裁决条件时，现有 reviewer 选择 select/patch/blocked。patch 是唯一匹配 `{path, old, new}`；Runner 应用后走同一规范化、权限与严格检查，再由全新 reviewer 对最终 raw 和本阶段有效全文签收。签收只有 approve/blocked，原 scope 预留一次调用和时间，没有新裁决循环。

select 只采用本次实际提供且已规范化的候选；版本、来源和必要调整在交给裁决前完成。直接选择所读候选后不再改字节。未签收的改稿不能解锁后续或发布。

## 运行、恢复与维护

CLI 直接使用当前用户真实文件系统。Claude 的工具清单按角色裁剪，worker 的 Edit/Write 精确到授权文件，Bash 只许可固定 helper；Codex worker 用原生 workspace-write 加本次 workspace，reviewer 用 read-only。worker 可以用同目录临时文件完成原子编辑；最终采纳要求精确许可集合、普通直属文件和受保护内容不变。Extract 不提供宿主文件系统隔离保证。

`agent.run(attempt, prompt, schema, timeout)` 返回真实 backend、退出/终止、原生完成、session、response、usage 和日志。成功同时要求退出 0、唯一明确成功终结、无调用错误、精确 session、有效 schema/nonce。Claude 读取 structured_output；Codex 核原生末条 agent_message 与 output-last-message。文本、截断事件或缺失结果不放行。

启动门先登记身份再放行，并作为本次调用的 child-subreaper 保持脱离后代的归属。CLI 主进程的实际退出单独登记；正常结束、取消或超时均由已有进程身份/父子/进程组检查确认后代结清。未确认终止保持 fence。恢复重用同一 parser 和真实 exit，确认中断后退休私有 session，以 checkpoint 和短交接新建；确证未启动保留前次会话。

当前维护只处理未交付的固定 run，由已有 Supervisor ledger、单写者排空与发布屏障执行。任务指定已选 P1–P5 的 `review` 或 `patch` 及 `fields`（`patch` 必填非空；`review` 无默认上限，填 `[]` 仅做只读复核，非空时作为退修进入 `patch` 的范围上限）；完整稿先审，未完成稿继续正常写作。维护不换来源、规则或执行器，不代替正常阶段和 P6。

## 发布与独立视图

发布核冻结身份、全部调用结清、当前 P1–P4/节点批准、P6、最终 checkpoint 和完整 Template 目录检查。最终直属 YAML 精确等于 selected_designs，中文目录、来源、唯一共享 Contract 与实际图齐全。检查不改已审 bytes。

原子交付保留 raw 与 `.extraction/receipt.json`；存在全局 hold、控制请求或维护屏障时不落盘，目标已有不同内容时记 `conflict` 并保留双方，原子可见后状态提交中断可由 `resume`/`export` 按相同完整 manifest 幂等重放。PNG 按需生成，不是统一发布硬门禁。

standalone 使用当前 Template 的完整目录校验与导出 API（不接收某次 run 的 `selected_designs`），展开实际 Design 和 root Contract，保留该节点元数据、when、Version、Realization。输出置于来源目录外，`.source.json` 分别绑定实际内容来源、完整目录检查清单及 Template 身份；目录或原件漂移均拒绝。

## usage、时间与证据封存

用量按 attempt 去重，原始 provider 口径留在 result 和 ledger。requested/reported model/effort 分列；未知项保留 null。失败、取消、恢复和明确预算追加都保留已用账目。

Claude 的 `model_usage_since_last_result` 是原生本进程已报告增量；`usage` 仅主循环当前 turn，`modelUsage/total_cost_usd` 可随 resume 累计。没有增量时不把累计 session 值相加成调用总量。cache read/write 分列，不套 Codex 的缓存输入子集公式。

Codex 原生 usage 与可选精确 UUID 的 native counter 观测分别保留。可证的根会话增量不自动代表全部模型调用，未观察到子会话也不记为零。CLI 报告成本不等于实际账单。

墙钟、CLI、检查处理和等待时间分别记录；等待可与兄弟工作重叠。进程内 Template 检查在提交前结算实际耗时，超额候选不获批准；它不提供硬实时抢占。

`audit --verify` 深验原生事件、exit、schema/nonce、原稿、最终 checkpoint 和来源证据。归档只收具体证据文件；Codex 额外收末条响应，私有认证及整个 CLI home 不进入包。包用于审计，不提供新的恢复或批准授权。

## 离线验证与能力回归

```sh
python3 -m unittest discover -s schema_extract/tools/tests -v
```

离线测试使用临时目录、fake CLI 与确定性回执；运行测试时将真实 claude/codex 入口从测试进程 PATH 阻断。不需要额外 proc 或 sudo 环境准备。Template 只重跑相关 API 回归。

`review-eval --suite FILE --output NEW_DIRECTORY` 则会真实调用独立 reviewer，在固定的合成标量复制任务（共用同一评估指令、相邻 `reference.py`/`source.md` 绑定与标量复制 `dispatch.md`）下评已知正确/错误用例的四项判断及具体缺陷；自定义 suite 用于增改该任务用例，不切换评估算子。当前小套件为 [`tests/fixtures/review_eval/suite.json`](tests/fixtures/review_eval/suite.json)，不设 archive 默认；能力回归不参与 run 批准。真实 CLI、真实六阶段和离线机器测试的覆盖分别记录。

### review-eval 用例约定

字段骨架参照 [`suite.json`](tests/fixtures/review_eval/suite.json)，增改用例时遵守以下装配与评分语义：

- **材料装配与 `Source` 覆写：**`base` 与同目录的 `reference.py`、`source.md` 相对 suite 所在目录读取。展开后的每个 case 写入 `root.yaml`，工具自动生成标量复制 `dispatch.md`，并将 `META_DATA.Source` 与 `CONTRACT.Algorithm_requirement.Golden_reference.Source` 覆写为本次冻结快照下的 `reference.py::kernel` 和 `reference.py::reference`（针对这两处的补丁会被覆盖）。送审前先完成规范化及 `skip-hardware` 结构与来源检查，错误例须为结构合法的语义缺陷。
- **`extends` 与 `patches`：**省略 `extends` 从 `base` 展开，填写则从同 suite 指定 case 的已展开文档开始，再顺序应用本 case 补丁；仅继承文档，`expected` 由各 case 自填。补丁 `path` 用非空分段数组沿已有父容器定位（字符串作映射键、整数作列表下标，如 `["DESIGN", "Work_schedule", "Thesis"]`），将目标整值替换为 `value`，不做深合并。
- **四项判断标识：**[REVIEW 四项判断](../REVIEW.md#四项判断)依次对应 `document_self_contained`（独立理解）、`document_layering`（分层正确）、`document_consistency`（全文一致）、`document_relevance`（内容必要）。
- **`expected` 命中语义：**通过例填 `decision: "approve"`、`criteria: []`、`path: ""`，要求 reviewer 返回 `approve` 且无 `issues`。错误例填 `decision: "revise"`：`criteria` 为可接受标识集合，`path` 为单个不带文件名的字段路径字符串（从 `META_DATA`、`CONTRACT`、`DESIGN` 或 `REALIZATION` 起始，如 `"DESIGN.Work_schedule.Thesis"`）；reviewer 返回 `revise`，且任一条位于这四个顶层区内的 `issue` 同时满足 `criterion` 属于 `criteria`、`detail` 非空、`issue.path` 与 `expected.path` 相同或以 `.`/`[` 为边界互为祖先/后代即判通过（不要求检出 `criteria` 中全部项，也不逐项验收多处缺陷）。
