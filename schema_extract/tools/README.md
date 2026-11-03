# Schema Extract：工具维护与回执协议

[Supervisor 手册](../README.md)保留入口、参数、状态与恢复；各阶段页定义写审方法和阶段专有字段。[stages.py](stages.py)定义可机械判断的阶段读写范围，[receipts.py](receipts.py)定义回执 schema 与门禁。本页和运行代码一同进入 `rules/` 冻结清单。

## 写入与完整性

[runner.py](runner.py)持有运行写锁并串行提交结果；[store.py](store.py)负责不可变 checkpoint、已审依赖和权威状态，[common.py](common.py)记录当前 Runner 线程实际写出的内容。`_RunGuard` 每次仍枚举完整运行目录并拒绝符号链接、特殊文件及未授权的增删改；只有设备、inode、模式、链接数、大小、纳秒 mtime/ctime 均未变的文件才复用内存中的 SHA-256。缓存不跨 Runner 生命周期，不能依靠恢复 mtime 隐藏同长度改写。

`refresh` 只接受本轮 Runner 登记的精确字节摘要、新 checkpoint 的已验证 manifest，以及刚确认结束调用原先排除的 workspace 和已知 transport 文件。每次刷新都核对其余历史文件和文件集合；登记消费后清空，不给同名路径永久写权限。`write_json`、文本回写、复制和派生目录重建分别登记实际内容或删除范围。Agent 线程和并发 `requests/` 提交者没有这份写入权限。

活跃 Runner 读取 `state.json` 时还会与自己最后写出的字节比较。完整性失败保存 `authority_changed`、破坏路径的预期摘要/不存在标记和最后可信状态，保留在途调用的 fence；历史修复前不能重新建立基线。`cancel` 仍可确认旧进程终止，保留 attempt 给修复后的正常恢复对账、计时和退休中断 UUID。源码/规则快照、批准回执、checkpoint 和最终合入继续使用内容门禁，stat 缓存不替代这些验收。

## 角色与会话

Supervisor 在主窗口发起任务、读状态、补充外部材料、处理已明确的任务要求变更和发布冲突。Runner 是运行状态的唯一写入者：选择阶段页、派发调用、验证回执、保存 checkpoint、合入和发布。worker 只修改调用中列明的产物；reviewer 通常只读对应 checkpoint，按阶段门槛给出结论，不通过修改状态自行放行。

| 阶段 | worker 与 reviewer 的上下文 |
| --- | --- |
| P1 | 分别新建，两个角色独立。 |
| P2 | 分别续用 P1 对应的会话，但明确注入 P2 页并单独验收和计退修。 |
| P3、P4 | 每阶段各开全新一对。 |
| P5 | 每个节点各开全新一对；同节点返工续用本节点原角色。 |
| P6 | 全新 reviewer 先审；确需修改才启动全新 worker。 |

Runner 直接启动 `codex exec`，正常续写用 `codex exec resume <会话 UUID>` 精确指向原角色，禁止 `--last` 或会话名查询。每次只传当前任务页和定位信息：角色、阶段、节点、固定源码入口、起始 checkpoint、计划修订、可写范围、应读产物和本轮退修意见。换页时不因会话已有历史而省略新任务页。P1 的两个会话必须不同，P2 分别复用；P3/P4、每个 P5 节点和 P6 全部换新，各角色不得复用别的阶段/节点 UUID。reviewer 使用独立产物副本和只读 sandbox；裁决时也只提交补丁提案，由 Runner 应用。未完成调用恢复时，先确认终止，再从安全 checkpoint 建新上下文；`session_handoffs` 记录旧 UUID、新尝试和新 UUID，已退休 UUID 不能复用。正常暂停后已完成的会话仍按上表续接。

## 回执与短交接

机器格式是 `schema_extract.receipt.v2`，每次调用的 `response.schema.json` 是完整字段清单，定义见 [receipts.py](receipts.py)。agent 通过 CLI 最后输出返回 JSON，由 Runner 保存，不要求 agent 创建额外报告文件。每页的“回执与下一位”补充阶段专有内容。回执包含：

- 调用给定的阶段、角色、节点、尝试及输入定位；不得自行生成 checkpoint 或计划身份。
- 本次完成的内容、实际改动文件、检查命令与退出码；未运行时说明原因，不能把手工核对写成脚本通过。
- 支持结论的来源版本、精确文件和行号、短引文，以及对应字段或决定；引用必须能在固定来源中回查。
- worker 的完成或 `continue` 请求，或 reviewer 的通过、可执行退修、已核实阻塞；缺口要说明影响范围、证据和恢复条件。
- `framework_candidates`，没有合格建议时为 `[]`；分流准则见[提取总览](../../EXTRACT_SCHEMA.md#框架问题分流)。

Runner 验证回执、许可的文件差异、检查结果与依赖内容，再将放行绑定到具体产物摘要。草稿 checkpoint 只用于恢复，不能解锁后续工作；阶段通过与阶段内续写分别记录。

`identity` 原样回传 `run_id/attempt_id/input_fingerprint/stage/role/node/plan_revision/base_checkpoint/dependency_digests/mode/generation/node_signature`。`generation` 绑定本阶段或节点的有效输入，`node_signature` 绑定 P5 已审条目；不要自行改写。`mode` 为 `normal/adjudicate/signoff`，角色仍为 worker 或 reviewer。`artifacts` 列出本次 workspace 全部交付文件的 `path/sha256`，包含未改的只读文件；P5 例外，只提交本节点 YAML，root/祖先仅作为只读证据。`changed_files` 按字典序列出真实变更。`source_evidence` 带 `source_id/category/path/sha256/line_start/line_end/quote/note`，`artifact_evidence` 使用同样的文件锚点字段。路径相对固定树，行号从 1 开始；引用最多 12 行、短引文最多 320 字符，必须精确命中固定内容。

`decision` 为 worker 的 `complete/continue/blocked` 或 reviewer 的 `approve/revise/blocked`。P1 完成/通过需引用 implementation、IO、input、reference、acceptance、scope 六类，并给出 `contract/coverage` 结论；P2 需 host 证据与 `root_choice/paths/coverage` 结论。P3 需 implementation/host 及 `design/walkthrough/dependencies`；P4 需 host 及 `tree/paths/coverage`；P5 需 implementation/host 及 `differences/inherited_design/walkthrough`；P6 需 host 及 `tree/selection/coverage/references/unresolved`。每类至少定位一次绑定入口，调用链其他文件可继续引用，语义关系由 reviewer 核实。reviewer 的 `artifact_evidence` 在 P1 引用 root；P2/P3/P4 引用 root 和 dispatch；P5 引用 root、直接父和本节点；P6 引用全部最终交付文件。每次调用的 `required_artifact_evidence` 由同一门禁规则给出具体文件名。`checks` 只记录实际命令及退出码；未执行时为 `[]` 并填写 `checks_note`，Runner 的结构门禁另记在 `validated.json`，不会采信模型声称的脚本通过作为门禁。

按范围阅读使用冻结的 [reading.py](reading.py)，例如 `python3 -B <rules>/schema_extract/tools/reading.py --root <source> --path input.py --start-line 1 --end-line 80`。返回源文件 SHA-256、原始行列片段及 `next_cursor`；续读时保留原范围并传 `--cursor`，避免长行被截断丢失。工具不写阅读状态。

P2→P3、P3→P4、P4→P5 换上下文时，由 reviewer 在回执中给出短交接，Runner 在对应运行记录生成 `handoff.md`。通常约 10 行，最多 20 个非空短行，正文建议不超过 1,200 字符。只写已定事项及理由、关键源码位置、待核实缺口和下一步入口；checkpoint 和来源标识由 Runner 自动附上。P5 读取 P4 交接、root 和已审直接父节点，完成时只留节点回执；子节点直接读取父产物，P6 汇总有效结论。

P2–P6 的普通回执另有必填 `reopen` 字段，无重开时填写 `null`。有新固定来源发现时可返回 `decision: "reopen"`，并给出 `reopen: {issue_id, target, reason, evidence}`；`evidence` 指向本轮 `source_evidence`，`issues` 仍须有具体范围、证据和恢复条件。`target` 只能回到上游：P2→P1，P3→P2/P1，P4→P3/P2/P1，P5/P6→P4/P3/P2/P1。外部任务要求变化用 `target: "external"`，保存需要任务方确认的明确条件。来源明确的转录错误可以引用已读过的源行：`issues.kind` 用 `transcription`，`scope` 定位错误产物，并以 `artifact_evidence` 引用当前错误字段。相同问题和固定源码范围只重开一次，换注释或短引文不能重新计次。

## 退修与裁决

普通 worker 的 `blocked` 是独立审阅请求。通过来源、版本和写入范围门禁的草稿仍交本阶段 reviewer，附上具体缺项；reviewer 独立判断通过、退修或真正基础阻塞。草稿检查退出 `1` 不妨碍 reviewer 返回 `revise`，该次意见照常计入实质退修并附检查诊断；检查退出 `2` 仍作为工具错误停止。完成和批准始终要求完整的阶段证据及必需检查退出 `0`。P1 的 `first_draft` 与初版门禁都按起始 checkpoint 是否仍为 starter 判断，初稿提交版本为 1；本轮先复制再填写不另算一次已接受修订。

每阶段单独累计实质退修，P5 按节点计数。reviewer 给出有证据、可执行的修改后，Runner 登记一次退修；低于上限时原 worker 修改、原 reviewer 再审。`continue` 段内续写、回执格式修复、CLI 暂时失败分别记账，不冒充实质退修，也都受范围内的调用与时间预算约束。JSON/字段格式错误和临时 CLI 错误各最多重试两次，保留原角色已建立的 UUID 及经过字段、来源门禁的合法草稿。陈旧身份、假摘要、非法证据路径或行范围、越权和来源篡改硬拒绝；未确认终止的尝试始终保留进程 fence。

若 `artifacts` 仅漏列不可写的文件，且已声明的路径、摘要和其余回执门禁全部正确，普通调用可按上述原预算修复完整清单，保留已保存草稿；必须重新提交完整回执才能推进。漏可写文件、错误摘要、重复项、额外或越界路径仍硬拒绝；裁决和签收不因此得到重试。

若证据路径、版本、实际摘要和不超过 12 行的范围均有效，非空且不超过 320 字符的 `quote` 文本却不命中，该提案仍被拒绝。Runner 先核验其余全部硬门禁，再一次列出不匹配的字段和行范围，让原角色重新读取、复核相关结论并提交完整回执；不补字、不改引用，也不默认维持批准。此类普通调用沿用上述最多两次修复、原 UUID、合法 checkpoint 和时间预算；后续回执必须重新通过精确引用及全部阶段验证。空引文、长度或范围越界仍硬拒绝，裁决与签收仍不重试。

达到上限（默认第 5 次退修）后，原 reviewer 接手一次裁决：选定已有写法，或提交限于当前产物的补丁，由 Runner 在隔离副本应用并检查。满足阶段门槛就记录裁决和修订后推进；关键事实仍缺、补丁无法应用或必需检查仍不通过，就保留草稿、阻塞原因和恢复条件。裁决只有推进或 `blocked` 两个出口。回执额外包含 `adjudication: {action, checkpoint, patches}`：`action` 为 `select/patch/blocked`；select 只能选本阶段/节点已保存的候选，patch 基于当前 checkpoint，条目为 `{path, old, new}`，`old` 必须非空且唯一匹配。回执的产物摘要仍绑定裁决者实际读取的只读副本；Runner 在隔离副本处理并另存结果摘要。恢复旧稿会按实际变化保留或提高当前版本，不回滚权威状态、预算和计数。补丁越界、应用失败、回执错误或检查失败即阻塞，裁决不重试。正常审阅已核实必须等待外部输入时，可直接 `blocked`，无需耗尽退修。

如果 P6 裁决者亲自改了最终交付，Runner 再开一位独立 reviewer 做一次签收，覆盖最后改动及受影响结论，仅判 `ready` 或 `blocked`。签收不再启动返工、裁决或传输重试；原裁决和签收分别保存身份与结论。选择已有 worker 写法不额外触发此签收。普通 P6 worker 修订仍走正常审阅回合。

退修次数、已用时间、截止条件和裁决所需预算持久化；超时保存最后可恢复 checkpoint。恢复、重开计划继续原计数和预算，同一问题及同一证据只触发一次重开。分支阻塞只让其后代等待，独立分支继续；所有可继续的工作结束后仍有必需节点阻塞，整个算子以 `blocked` 结束并交付进展及具体原因。

## 并行、合入与计划修订

Runner 立即派发所有满足依赖的节点，不设固定 jobs 数或 worker/reviewer 槽。每个节点只有一个在途调用，先写再审；兄弟节点可同时写、同时审。节点通过并合入后立即解锁自己的子节点，不等整层完成。阻塞节点仅使其后代等待，独立兄弟继续。

每个分支在私有目录工作，只读已审 root 和所需祖先，只写自己的 YAML。检查器会扫描同目录，因此兄弟草稿必须隔离。Runner 串行合入同一算子的结果：核对尝试、输入指纹、阶段、节点、计划修订、checkpoint 和实际依赖摘要；将许可差异合入最新已审目录，复查目录与依赖，先保存不可变 checkpoint，再原子更新权威指针。重复回执只提交一次。仅兄弟先完成时可保留既有审阅；父节点、Contract 或选择条件变化时重新核对受影响结果。

P5 发现树或直接父节点需要改变，以新证据申请重开 P4，由 P4 worker/reviewer 修订并验收。要改选代表路径先回 P2，再验 P3；现有 root 的 Design 或选择条件变化要复核 P3；涉及共享 Contract 先回 P1，实际外部要求变更交任务方确认。重开期间，旧 P5 调用可以结束，返回结果暂存。P4 通过后，Runner 比较条目和真实依赖内容，保留未受影响的已审节点及合法在途结果；失效回执记录在 `stale_results`。旧调用只有确认终止才能释放对应 `active_attempts` 身份，随后才允许同节点的新调用。

计划删除节点会移出当前产物集合，历史节点账本仍保留。重新加入相同 ID 时，沿用原退修、调用、已用时间、截止条件和一次裁决记录，版本继续原稿。重开与节点重加不产生新的预算，也不重新授予已用裁决。P4 的审阅计划与影响记录保存在 `plan_history`，不写入交付索引。

P6 仅在当前计划节点集合与当前有效、已合入节点集合相同，且 P1–P4 放行仍有效时启动。完成数从该集合推导，草稿和失效的旧审阅不计入。

## 离线验证

从仓库根目录检查 CLI 入口并运行保留的检查器测试：

```sh
python3 -B schema_extract/extract.py --version
python3 -B schema_extract/extract.py --help
python3 -B -m unittest discover -s schema_template/tools/tests -t schema_template
```

这些命令不会调用模型。发布、恢复或回执门禁的临时探针使用仓库外的合成输入与运行目录，需要调用替身时显式指定受控的 `--codex-bin`。已提取的 YAML 使用 `schema_template/check_yaml.py --skip-hardware` 检查。

生成回执 schema 用标量 enum 限定冻结来源的 ID、path/sha256 候选及 reviewer 的产物路径/摘要；worker 的动态产物摘要限定为 64 位小写十六进制。生成约束使用单独字段副本，接收端仍核验真实内容和路径/摘要配对，坏摘要硬拒绝；完整清单和空 `changed_files` 仍须按协议返回。
