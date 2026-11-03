# Schema Extract：Supervisor 手册

**当前版本 `2.0.4`：`extract` 完成 P1–P6、依赖并发、独立审阅及向 `--output` 原子发布。** 包含暂停、取消、进程终止确认、崩溃恢复、发布重试、显式旧 checkpoint 导入和只读巡检。实际运行与验证范围见[验收记录](#验收记录)。

每个任务始终使用创建时冻结的执行器。当前 CLI 在跳转前验证输入锁和整个快照；`extract` 创建后也进入刚冻结的入口。B/C/D 和 v1 记录保留各自冻结版本的能力，升级不会由新代码直接接管旧状态；旧 v1 产物通过显式导入建立新版任务。

人和 Supervisor 从[提取总览](../EXTRACT_SCHEMA.md)理解全流程；worker、reviewer 按下表只读当前阶段页和其中指定的材料，无需翻阅迁移计划。

| 阶段 | 任务页 | 当前产物 |
| --- | --- | --- |
| P1 | [共享 Contract](phases/01_contract.md) | 含元数据、共享算法要求和原样硬件占位的 root 草稿 |
| P2 | [dispatch 草案](phases/02_dispatch_draft.md) | root 选择、路径草案、必要的 root `when` |
| P3 | [root 设计](phases/03_root.md) | 完整且已审的 root |
| P4 | [dispatch 定稿](phases/04_dispatch_plan.md) | 索引、经审阅的结构化节点清单 |
| P5 | [派生节点](phases/05_branch.md) | 每次一个完整且已审的派生 YAML |
| P6 | [全算子终审](phases/06_final.md) | 可交付的整个任务目录及终审结论 |

## 输入与输出

Supervisor 提供算子 ID 或明确来源、任务输出目录及适用调用范围。catalog ID 从[算子索引](../kernel_wiki/CATALOG.md)定位，显式来源可以是本地目录或历史 Git 版本；两者都应解析到同一组输入：选定实现及 host 入口、任务 IO 和输入构造、reference、数值验收材料、适用范围。ID 本身、文件同名或候选函数签名都不能代替这些材料。

Runner 冻结选中来源目录的实际文件，将历史 Git 引用解析到实际 commit，并分别记录源码和本次规范、starter、检查器、六张阶段页、执行器的版本。源码与 `rules/` 各有逐文件 SHA-256 和整体身份；规范与执行器从当次工作树读取，包含未提交的改动。脏本地源码在快照内生成独立的 `source.git` 和真实 commit，保留原仓库、原 HEAD 与脏文件清单；不会提交或修改用户仓库。所有阶段读同一份固定来源；必需入口或引用缺失时返回具体缺项。

最终已审 checkpoint 包含 `root.yaml`、当前计划内全部 `<id>.yaml` 与 `dispatch.md`，另在运行目录保存 P2/P3/P4 短交接。每节点的 `draft_checkpoint` 与全算子的 `merged_checkpoint` 分开；只有通过审批并合入的节点才能解锁子节点。运行资料使用独立 `.schema_extract/v2/` 与显式 `schema_extract.run.v2` 状态格式；缺少并发调用及重开账本的旧记录会要求使用其冻结执行器。当前一次命令选择一个算子。

```text
.schema_extract/v2/
  format.json
  snapshots/<input-fingerprint>/
    inputs.lock.json             # source、rules 的独立身份与实际文件摘要
    source/                     # 固定源码、IO、reference、验收、范围材料
    source.git/                 # 仅脏本地源码需要的独立 Git 引用
    rules/                      # 当次规范、模板、检查器、六阶段页、执行器
  runs/<run-uuid>/
    state.json                  # Runner 唯一写入的权威状态
    inputs.lock.json
    attempts/<attempt-uuid>/    # 调用身份、prompt、CLI日志、结果、验证/拒绝记录
      workspace/                # 本次独立产物副本
    checkpoints/<digest>/       # 不可变产物、内容清单与来源身份
    stages/P2|P3|P4/handoff.md
    imported/                   # 可选：完整旧产物、checkpoint、task、lock及来源记录
    publications/               # 发布意图与发布后回执；可幂等重放
  requests/<run-uuid>/           # Supervisor只追加请求，Runner消费后改变state
```

catalog ID 使用索引中的明确角色绑定，本地 catalog 算子目录也可作为 `--source`。其他来源使用 `schema_extract.source.v2` JSON 描述文件，或在本地目录内放 `schema-inputs.json`。以下字段中的路径均相对 `directory`；`repository` 相对描述文件所在目录，省略为 `.`。省略 `revision` 使用本地实际字节，填写时只读相应 Git commit。`host` 省略时使用 `entrypoint`；同文件的不同、可定位函数可以分别作为实现与 reference。文件名本身不会被用来猜测输入角色。

```json
{
  "format": "schema_extract.source.v2",
  "id": "increment",
  "repository": "/path/to/kernel-repository",
  "directory": "operators/increment",
  "entrypoint": "kernel.py::kernel",
  "host": "kernel.py::kernel",
  "io": ["io.md"],
  "input": "input.py::make_inputs",
  "reference": "reference.py::reference",
  "acceptance": ["acceptance.md"],
  "scope": ["scope.md"]
}
```

历史 FA4 来源的可用定位示例：`repository` 指向本仓库，`revision` 为 `01fff90d714abc6e84904ea9c44523ba66d34a0e`，`directory` 为 `kernel_wiki/aka-four-problems-20261019/fa4_prefill`。选定实现用 `aka_solution/solution.py::Model.forward`，host 用 `aka_solution/flash_attn/cute/interface.py::_flash_attn_fwd`，IO 用 `metadata.json`，输入用 `input.py::_make_inputs`，reference 用 `reference.py::Model.forward`；验收材料为 `metadata.json` 和 `aka_solution/AGATE_VALIDATION.json`，范围材料为 `shapes.json`、`shape_train.json`、`shape_valid.json`。这些材料只作为固定来源；采样记录不替代完整数值语义的审阅。

## 命令与当前可用性

| 操作 | 当前状态与约定 |
| --- | --- |
| 新版 `extract` | 已实现 P1–P6、依赖就绪节点并发、增量合入、上游重开及最终发布。`--queue` 只冻结并排队。 |
| 新版 `status` | 已实现运行摘要、节点状态分类和 `--full` 权威状态读取；不会启动调用。 |
| 新版 `pause`、`resume`、`cancel` | 已实现调用边界暂停、取消请求与终止确认、按原预算恢复；具体命令见下文。 |
| `export`、`import-legacy` | 已实现终审 checkpoint 的另行导出、已确认终止的 v1 旧稿导入与独立重审。 |
| `inspect`、`candidates`、`issues` | 默认只读；`inspect --start` 可明确请求启动一个仍在排队的任务。候选按稳定 ID 接已有决定，台账仍由维护者编辑。 |
| `schema_template/check_yaml.py --skip-hardware` | 已实现；可检查现有或人工编写完成的任务 YAML，不能推进提取阶段。 |

从仓库根目录运行，需 Python 3.10+、PyYAML 和已配置 provider 的 Codex CLI：

```text
python3 -m schema_extract --version
python3 -m schema_extract extract --id bf16_gemm --output /path/to/delivery/bf16_gemm
python3 -m schema_extract extract --source /path/to/source.json --output /path/to/delivery/task
python3 -m schema_extract status <run-uuid>
python3 -m schema_extract status <run-uuid> --full
python3 -m schema_extract pause <run-uuid>
python3 -m schema_extract cancel <run-uuid>
python3 -m schema_extract resume <run-uuid>
python3 -m schema_extract export <run-uuid> --output /path/to/another/delivery
python3 -m schema_extract inspect
python3 -m schema_extract candidates <run-uuid>
python3 -m schema_extract issues
```

`--repo` 指定 catalog/Git 来源仓库，默认当前目录；`--state-root` 默认该仓库的 `.schema_extract/v2`。历史来源也支持 `--source 'git:<ref>:<directory>' --bindings /path/to/source.json`，这时 Git 仓库取 `--repo`，其余角色仍由描述文件提供。

`--version` 输出当前入口版本；已有运行的执行器版本应读其冻结 `rules/schema_extract/__init__.py`，完整内容身份以 `inputs.lock.json` 的 `rules.id` 为准。仅有相同版本号不能代替冻结摘要核验。

`--worker-effort`、`--reviewer-effort` 分别配置，默认 `xhigh`，同角色跨阶段不变；可选 `low/medium/high/xhigh/max`。`--model` 可覆盖模型，省略时沿用用户配置。Runner 保留 provider 配置，以 `--no-daemon` 直接调用 CLI；不会使用会丢失 provider 的 `--ignore-user-config`。`--codex-bin` 可选已安装 CLI 或可执行测试替身。

创建时固定 `--revision-limit 5`、`--operator-seconds 7200`、`--stage-seconds 3600`、`--call-seconds 1800`。`--max-calls 32` 是每阶段、每个 P5 节点各自的总调用预算，包含续写、修复和失败尝试；不限制整棵树的总调用数。阶段/节点从首次调用固定截止时间，算子截止时间从创建起固定。每个范围预留一次裁决；P6 另预留一次签收。各次预留时间在创建时确定为 `min(call_seconds, stage_seconds / 4, operator_seconds / 12)`，普通模型调用和检查器都不能动用；裁决时仍为 P6 签收保留时间。

`extract/resume` 输出运行 UUID、状态、checkpoint 和发布结果。退出码 `0` 表示终审已通过且已发布；`extract --queue` 的 `0` 表示已排队。`1` 表示暂停、取消、调用、回执、缺项、检查、预算停止或发布冲突；`2` 表示入口输入/工具错误。`pause/cancel` 的 `0` 只表示请求已提交，应从 `status` 核实停止位置与 `unconfirmed_attempts`；`status/inspect` 成功为 `0`。

以下检查命令已经实现。从仓库根目录运行，将 `path/to/task/root.yaml` 替换为实际文件；需要 Python 3.10+ 和 PyYAML：

```sh
python3 schema_template/check_yaml.py --skip-hardware path/to/task/root.yaml
```

检查器会扫描该文件所在目录直属的全部 YAML。完成的 P3、P5、P6 必须退出 `0`；`1` 表示输入未通过，诊断交当前修改者；`2` 表示参数、工具或环境问题，先解决检查无法完成的原因。两类失败分别记录。P1、P2、P4 按阶段产物验收，不以未完成的 Design 触发完整 YAML 检查门槛。硬件占位不改动、不审阅；GPU 验收与实测始终不在本流程内。

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

`inspect` 调用各自冻结执行器的 `status`，默认只读运行摘要。使用 `extract --queue` 建立排队任务后，可用 `inspect --start <run-uuid>` 请求启动；Runner 在独占写锁内再次确认任务仍为 `queued`，不会把竞争期间已暂停或取消的任务恢复。排队仍沿用创建时的截止条件。此入口不安装 launchd、cron 或 Codex automation；巡检缺席不影响主流程。

## 回执与短交接

机器格式是 `schema_extract.receipt.v2`，每次调用的 `response.schema.json` 是完整字段清单，定义见 [receipts.py](receipts.py)。agent 通过 CLI 最后输出返回 JSON，由 Runner 保存，不要求 agent 创建额外报告文件。每页的“回执与下一位”补充阶段专有内容。回执包含：

- 调用给定的阶段、角色、节点、尝试及输入定位；不得自行生成 checkpoint 或计划身份。
- 本次完成的内容、实际改动文件、检查命令与退出码；未运行时说明原因，不能把手工核对写成脚本通过。
- 支持结论的来源版本、精确文件和行号、短引文，以及对应字段或决定；引用必须能在固定来源中回查。
- worker 的完成或 `continue` 请求，或 reviewer 的通过、可执行退修、已核实阻塞；缺口要说明影响范围、证据和恢复条件。
- `framework_candidates`，没有合格建议时为 `[]`；分流准则见[提取总览](../EXTRACT_SCHEMA.md#框架问题分流)。

Runner 验证回执、许可的文件差异、检查结果与依赖内容，再将放行绑定到具体产物摘要。草稿 checkpoint 只用于恢复，不能解锁后续工作；阶段通过与阶段内续写分别记录。

`identity` 原样回传 `run_id/attempt_id/input_fingerprint/stage/role/node/plan_revision/base_checkpoint/dependency_digests/mode/generation/node_signature`。`generation` 绑定本阶段或节点的有效输入，`node_signature` 绑定 P5 已审条目；不要自行改写。`mode` 为 `normal/adjudicate/signoff`，角色仍为 worker 或 reviewer。`artifacts` 列出本次 workspace 全部交付文件的 `path/sha256`，包含未改的只读文件；P5 例外，只提交本节点 YAML，root/祖先仅作为只读证据。`changed_files` 按字典序列出真实变更。`source_evidence` 带 `source_id/category/path/sha256/line_start/line_end/quote/note`，`artifact_evidence` 使用同样的文件锚点字段。路径相对固定树，行号从 1 开始；引用最多 12 行、短引文最多 320 字符，必须精确命中固定内容。

`decision` 为 worker 的 `complete/continue/blocked` 或 reviewer 的 `approve/revise/blocked`。P1 完成/通过需引用 implementation、IO、input、reference、acceptance、scope 六类，并给出 `contract/coverage` 结论；P2 需 host 证据与 `root_choice/paths/coverage` 结论。P3 需 implementation/host 及 `design/walkthrough/dependencies`；P4 需 host 及 `tree/paths/coverage`；P5 需 implementation/host 及 `differences/inherited_design/walkthrough`；P6 需 host 及 `tree/selection/coverage/references/unresolved`。每类至少定位一次绑定入口，调用链其他文件可继续引用，语义关系由 reviewer 核实。reviewer 的 `artifact_evidence` 在 P1 引用 root；P2/P3/P4 引用 root 和 dispatch；P5 引用 root、直接父和本节点；P6 引用全部最终交付文件。每次调用的 `required_artifact_evidence` 由同一门禁规则给出具体文件名。`checks` 只记录实际命令及退出码；未执行时为 `[]` 并填写 `checks_note`，Runner 的结构门禁另记在 `validated.json`，不会采信模型声称的脚本通过作为门禁。

按范围阅读使用冻结的 [reading.py](reading.py)，例如 `python3 -B <rules>/schema_extract/reading.py --root <source> --path input.py --start-line 1 --end-line 80`。返回源文件 SHA-256、原始行列片段及 `next_cursor`；续读时保留原范围并传 `--cursor`，避免长行被截断丢失。工具不写阅读状态。

P2→P3、P3→P4、P4→P5 换上下文时，由 reviewer 在回执中给出短交接，Runner 在对应运行记录生成 `handoff.md`。通常约 10 行，最多 20 个非空短行，正文建议不超过 1,200 字符。只写已定事项及理由、关键源码位置、待核实缺口和下一步入口；checkpoint 和来源标识由 Runner 自动附上。P5 读取 P4 交接、root 和已审直接父节点，完成时只留节点回执；子节点直接读取父产物，P6 汇总有效结论。

P4 的 `plan` 是节点数组，各项包含 `id/file/parent/workload/difference/selection/evidence`；`evidence` 是本回执来源索引，须包含 host 依据。root 为 `id: root, file: root.yaml, parent: ""`；派生文件为 `<id>.yaml`。worker 与 reviewer 返回相同的节点语义清单，两角色的证据数组顺序和索引独立。Runner 校验唯一 root、ID/文件唯一、父存在、无环、全可达和索引链接集合。P5 的 `node_evidence` 是当前 P4 回执中已解析的本节点来源锚点，`plan_receipt` 给出该只读回执路径，避免把上游的局部索引当成本轮证据编号。

P6 退修通过 `revision_scope: [{path, fields}]` 指明文件和点分字段路径，另用 `affected_nodes` 列节点 ID。`dispatch.md` 使用 `fields: ["*"]`；派生 YAML 可限域修改 `design`、`realization`、`meta_data.source`、`meta_data.dispatch.when` 下的字段，版本随内容递增。Runner 将受影响的后代一起列入 `pending_rechecks`，后续批准必须覆盖全部。root、共享 Contract 或树的改动通过下述上游重开处理。P6 派发前和放行前都验证已审节点的回执、Runner 门禁、checkpoint 和实际内容一致。

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

## 状态阅读

`status <run-uuid>` 展示阶段、checkpoint、已审集合、计划节点数、完成数及停止原因。`nodes_by_status` 的 `ready/in_flight/revision/approved/waiting_parent/blocked` 列出具体节点，`node_details` 给出父节点、角色、模式、调用 ID 和原因；P5 的并发进展以这些节点项为准。`--full` 可读固定预算、会话 UUID、阶段与节点计数、`active_attempts` 和重开历史。

运行状态为 `queued/running/paused/cancelled`，终审通过为 `ready`；阶段可停在 `invalid_receipt`、`blocked`、`budget_exhausted`、`tool_error`、`cli_error` 或 `termination_unconfirmed`。节点错误保存在对应节点，独立分支继续后，算子汇总为 `blocked`。若仍有未确认终止的调用，保留其 `active_attempts` 和 `unconfirmed_attempts`，仅其范围不能重发；独立节点继续。取消后的 `cancelled` 不替代进程确认，应同时读取未确认列表。格式修复只保存合法草稿，不能授予审批。

下表列出状态阅读重点。`published` 和 `publication.status` 区分已审、发布成功与目标冲突；`elapsed_seconds` 是本次调用和检查的已记账时间，导入任务另保留 `legacy_usage`。

| 摘要 | Supervisor 如何判断下一步 |
| --- | --- |
| 阶段、计划修订、已审集合与总节点数 | 确认完成数来自当前有效合入；查看对应 checkpoint。 |
| 就绪、在途、退修、等待父节点的节点 | 由 Runner 按依赖继续；核对当前角色、调用与退修次数。 |
| 阻塞节点和后代 | 读具体缺项和恢复条件；独立分支仍可继续。 |
| 预算、截止条件、最后 checkpoint | 定位剩余工作和超时后可恢复位置。 |
| 暂停、取消及进程终止确认 | 确认停止到哪个调用边界，恢复前核实旧调用已终止。 |
| `ready` 与发布结果 | `ready` 只在 P6 通过后成立；另查输出目录是否成功发布或存在冲突。 |

## 恢复与交付

`pause <run-uuid>` 和 `cancel <run-uuid>` 只向 `requests/` 追加请求。已有 Runner 消费请求；若它已退出，命令进入同一 Runner 写锁完成对账和停止。暂停不再派发下一调用，已启动的调用可完成并保存结果。取消通知全部在途调用终止，并在各自 `result.json` 或 `termination.json` 保存确认；无法确认时保留该范围的身份。

`resume <run-uuid>` 先确认旧调用及其已记录子进程终止。调用已启动但进程身份缺失或无法核实时保留 fence；不能凭“没有看到进程文件”断言已终止。其他依赖就绪节点继续。旧调用留下合法草稿时保存到 checkpoint，否则继续使用上一个安全 checkpoint，并在 `recovery.json` 说明丢弃原因。已经使用的一次裁决或签收若没有可验证的完成结果，就阻塞，不重新授予。

恢复读取磁盘上的 attempt、result、receipt、验证记录及 checkpoint，包含上游重开时尚未合入的并发结果。已审成果仍须通过依赖校验；只有结果对应的 checkpoint 完整落盘后才更新任务指针。`accepted_receipts` 和按 attempt ID 保存的 `attempt_usage` 防止重复合入、退修和计时。检查期间崩溃而无法取得结束采样时，保守计入未闭合时段；原算子/阶段截止时间、调用上限及预留均不重置。`resume` 不自动批准外部任务要求变化。

发布把整个已审目录及 `.extraction/receipt.json` 一次性重命名到目标，原子操作拒绝覆盖已存在的空目录、文件或不同成果。若目标已经是同一完整结果，`resume` 可补写发布回执而无需再次调用模型。冲突时 `status` 给出原目标和已审 checkpoint；用下列命令另行导出，原目标和原 `--output` 保留：

```text
python3 -m schema_extract export <run-uuid> --output /path/to/new/delivery
```

旧 v1 状态继续从 `inputs.lock.json` 的 `snapshot` 内调用其真实旧包，使用原 `--state-root`。例如已包含旧 `status` 的快照可运行：

```text
cd /path/from/old/inputs.lock/snapshot
python3 -B -m schema_extract --repo "$PWD" --state-root /path/to/old/.schema_extract status <old-run-id> --full
```

具体语法以该冻结版本的 `--help` 为准。新版不从 `schema_extract_old` 导入执行代码。旧源码、进程确认和固定 checkpoint 都齐备时，另写一份 `schema_extract.source.v2` 描述文件，`repository` 指向旧 snapshot，绑定该版本实际实现、IO、input、reference、验收和范围，然后执行：

```text
python3 -m schema_extract import-legacy --from-task /old/.schema_extract/tasks/<task-key> --checkpoint <old-checkpoint-id> --source /path/to/legacy-source.json --output /new/delivery/task --queue
python3 -m schema_extract resume <new-run-uuid>
```

导入要求旧任务已停止、旧 Runner 锁空闲、全部已登记调用有终止证据，并通过旧冻结入口真实读回状态；它不会替旧任务杀进程或改状态。选定源码必须与旧锁的实际字节一致，新增证据使用新的 `extract`。`imported/` 保存完整旧产物、原 checkpoint、任务、输入锁、用量、旧问题与来源；旧引用只在文件内容已核实相同时绑定到本次冻结源。root 作为 P1 草稿，旧索引和派生节点供后续阅读，新阶段不继承旧批准。若旧硬件段不是当前占位，初始草稿恢复当前 starter 并提高版本，旧字节完整保留。原用量单独留存，本次独立重审受创建时固定的新阶段预算约束。

本阶段以真实历史 `69e3027` 的 v1 代码和保真受控运行验证旧入口与导入；当前主机没有原 Mac 环境中的五个旧任务目录，不能据此声称已在本机重读那五个实际任务。已核对本机没有对应 launchd、Codex automation 或 crontab 巡检配置，未安装新定时器。

清理运行记录前，核实任务缺项、交付位置和框架候选已留存。使用[新版维护入口](issues/README.md)按稳定 ID 查重；它直接读取[旧版台账和历史决定](../schema_extract_old/issues/README.md)的数据，保留已有结论。候选 ID 不合格式时仍能读到原报告和具体原因，不阻断其他合法候选。

## 验收记录

**`2.0.4` 已通过 F 端到端验收。** Supervisor 用下列单条命令完成 [GDN post-convolution MTP BF16 / FP32 state](../kernel_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_postconv_mtp_bf16_state_fp32/README.md) 的真实提取，原 `extract` 进程退出 `0`，最终状态为 `ready`、`published: true`。运行时审定 **1 个 root（version 2）**，P5 因无派生节点跳过。

```sh
python3 -m schema_extract extract --id gdn_postconv_mtp_bf16_state_fp32 --output /home/shaohanyong.shy/KernelSchema/schema_wiki/extract-v2-acceptance/gdn_postconv_mtp_bf16_state_fp32
```

已发布 [root.yaml](../schema_wiki/extract-v2-acceptance/gdn_postconv_mtp_bf16_state_fp32/root.yaml)、[dispatch.md](../schema_wiki/extract-v2-acceptance/gdn_postconv_mtp_bf16_state_fp32/dispatch.md) 和机器生成的[发布凭据](../schema_wiki/extract-v2-acceptance/gdn_postconv_mtp_bf16_state_fp32/.extraction/receipt.json)。重复提取应选择新的输出目录。

| 身份 | 实际记录 |
| --- | --- |
| run ID | `02be0098-eec6-4f8f-8f6e-05c743dc788b` |
| 源码 commit | `c3fc7a029476ba448d6378aad4d5323d24a476ee`；选定源码字节与该 commit 一致 |
| 冻结执行器 | `2.0.4`；`rules.id = 14cd1ae43b8a64cc777621e059ad6f32ee68110704a3b08a94811a12157daed2` |
| 输入指纹 | `56df9fa1bccdf4332d5f18f0288a5b0ec96401d165a6f464a4dcd713495b2813` |
| 最终已审 checkpoint | `50fd86191db4b879b39924f281bd7aef1dfb93b7402ac678740467e20e0dfaf5` |

本次为 **7 个真实 Codex 会话、9 次直接 CLI 调用**，CLI 均为 `0.166.0`，worker/reviewer 均为 `xhigh`，保留 provider 配置并使用 `--no-daemon`。P1/P2 分别精确续接各自角色；P3、P4、P6 新建会话。P6 只执行一次，运行计数为 0 次实质退修、回执修复、CLI 失败和上游重开。全部调用确认终止，最终没有在途或终止未确认的尝试。

| 阶段 | worker 会话 UUID | reviewer 会话 UUID | 结果 |
| --- | --- | --- | --- |
| P1 | `01a19d1e-b4e8-7af2-8661-66f00eff51f1` | `01a19d27-b10d-7c83-9233-c18717c5e729` | 共享 Contract 独立通过 |
| P2 | 精确续接 P1 worker | 精确续接 P1 reviewer | root 选择与路径草案独立通过 |
| P3 | `01a19d38-4207-7d33-ba05-b847d0161fbf` | `01a19d42-cf5f-7600-9d53-7e5196411b1f` | 完整 root 独立通过，Runner 检查退出 0 |
| P4 | `01a19d4c-cffb-7692-9963-e62c5819159d` | `01a19d53-3e46-71b3-ab26-ea37e3b1666c` | 索引及唯一 root 的结构化计划独立通过 |
| P5 | — | — | `skipped`，0 次调用 |
| P6 | 按需角色未启动 | `01a19d59-9a97-7e90-87ba-d2b672442ccc` | 终审通过，Runner 全目录检查退出 0，原子发布成功 |

完整证据保存在 `.schema_extract/v2/runs/02be0098-eec6-4f8f-8f6e-05c743dc788b/` 的输入锁、各 attempt 的 prompt/argv/事件/回执/检查记录、checkpoint 和 publications 中。独立复核了 181 个源码锚点、72 个产物锚点以及全部阶段审批链；P6 回执、已审 checkpoint、发布凭据和目标完整 manifest 一致。Supervisor 与独立 reviewer 均在发布目标上重跑 `check_yaml.py --skip-hardware`，退出 `0`。以下命令用于读取本次状态和复查交付：

```sh
python3 -m schema_extract status 02be0098-eec6-4f8f-8f6e-05c743dc788b
python3 schema_template/check_yaml.py --skip-hardware schema_wiki/extract-v2-acceptance/gdn_postconv_mtp_bf16_state_fp32/root.yaml
```

中断后的同版恢复使用 `python3 -m schema_extract resume <run-uuid>`；已发布运行的幂等行为、目标冲突和旧版本边界见[恢复与交付](#恢复与交付)。本次成功运行没有中断；恢复路径由自动回归覆盖。

**自动回归：** `2.0.4` 的 82 项提取回归全部通过（793.329 秒），共用检查器的 147 项测试全部通过（28.216 秒）。提取回归覆盖正常、返工、并行、重开、恢复、旧稿导入、发布及实际运行发现的回执边界，可用 `python3 -m unittest discover -s schema_extract/tests -p 'test_*.py' -v` 重跑。

本次仓库链接核对覆盖 147 个 Markdown 文件、1505 条本地路径与标题链接，以及 14 条本仓 Git 历史链接，均可解析；历史目标已逐个核对到真实 Git 对象。

**保留的来源限制：** `dispatch.md` 与 root 的调度说明披露了合法 padding 被候选对齐检查拒绝且没有 fallback；`contract.algorithm_requirement.golden_reference.tolerance` 保留外部比较器精度和 NaN/Inf 策略未声明这一事实，`design.math_reshape` 保留 fast-math 全域误差界未知及 BF16 中间舍入。P6 回执的 `issues` 定位了对齐覆盖和比较器未知两项问题及其恢复条件。可选图工具的 `pipeline-graph-skip-hardware` 候选保存在 P3 worker 回执，可用 `python3 -m schema_extract candidates 02be0098-eec6-4f8f-8f6e-05c743dc788b` 查阅；它不改变本次必需检查与终审结果。

此前 `2.0.0–2.0.3` 的四次未发布试跑分别暴露初稿版本定位、遗漏只读清单成员、只读摘要抄录和精确引文转录问题，促成了通用提示、生成约束与有界回执重交修复。旧运行和原始错误提案保持原样；正式验收以以上 `2.0.4` 成功运行及回归为准。

FA4 完整重新提取单列为后续长程任务。本机旧数据兼容性仅以真实 `69e3027` 冻结 v1 代码及保真受控运行验证；原 Mac 环境中的五个实际旧任务目录不在本机，不能据此声称已重读它们。
