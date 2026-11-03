# Schema Extract：Supervisor 手册

**当前版本 `2.1.3`：`extract` 完成 P1–P6、依赖并发、独立审阅及向 `--output` 原子发布。** 包含暂停、取消、进程终止确认、崩溃恢复、发布重试、显式旧 checkpoint 导入和只读巡检。发布状态与恢复方法见[恢复与交付](#恢复与交付)。

每个任务始终使用创建时冻结的执行器。当前 CLI 在跳转前验证输入锁和整个快照；`extract` 创建后也进入刚冻结的入口。旧任务保留各自冻结版本的能力，升级不会由新代码直接接管旧状态；旧 v1 产物通过显式导入建立新版任务。

统一从 [extract.py](extract.py) 调用 `tools/` 中的实现。冻结清单同时包含入口和工具代码；`status`、控制命令、巡检和 reader 都按已验证的清单定位，版本升级不改写旧快照。

人和 Supervisor 从[提取总览](../EXTRACT_SCHEMA.md)理解全流程；worker、reviewer 按下表只读当前阶段页和其中指定的材料。

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

最终已审 checkpoint 包含 `root.yaml`、当前计划内全部 `<id>.yaml` 与 `dispatch.md`，另在运行目录保存 P2/P3/P4 短交接。每节点的 `draft_checkpoint` 与全算子的 `merged_checkpoint` 分开；只有通过审批并合入的节点才能解锁子节点。运行资料使用独立 `.schema_extract/v2/` 与显式 `schema_extract.run.v2` 状态格式；缺少并发调用及重开账本的旧记录会要求使用其冻结执行器。当前一次命令选择一个算子。catalog 算子的输出目录按 `paths.readme` 的父路径镜像到 `schema_wiki/`，查找与命令示例见[输出目录导航](../schema_wiki/README.md)。

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
python3 schema_extract/extract.py --version
python3 schema_extract/extract.py extract --id bf16_gemm --output schema_wiki/aka-cuda-operators-20261030/operators/gemm/dense/bf16_gemm
python3 schema_extract/extract.py extract --source /path/to/source.json --output /path/to/delivery/task
python3 schema_extract/extract.py status <run-uuid>
python3 schema_extract/extract.py status <run-uuid> --full
python3 schema_extract/extract.py pause <run-uuid>
python3 schema_extract/extract.py cancel <run-uuid>
python3 schema_extract/extract.py resume <run-uuid>
python3 schema_extract/extract.py export <run-uuid> --output /path/to/another/delivery
python3 schema_extract/extract.py inspect
python3 schema_extract/extract.py candidates <run-uuid>
python3 schema_extract/extract.py issues
```

`--repo` 指定 catalog/Git 来源仓库，默认当前目录；`--state-root` 默认该仓库的 `.schema_extract/v2`。历史来源也支持 `--source 'git:<ref>:<directory>' --bindings /path/to/source.json`，这时 Git 仓库取 `--repo`，其余角色仍由描述文件提供。

从仓库外可用脚本绝对路径启动；`--help`、`--version` 不依赖当前目录。任务命令同时显式提供 `--repo /path/to/repository --state-root /path/to/runtime`。

`--version` 输出当前入口版本，单点定义位于 `tools/__init__.py`；已有运行从其冻结入口读取版本，完整内容身份以 `inputs.lock.json` 的 `rules.id` 为准。仅有相同版本号不能代替冻结摘要核验。

`--worker-effort`、`--reviewer-effort` 分别配置，默认 `xhigh`，同角色跨阶段不变；可选 `low/medium/high/xhigh/max`。`--model` 可覆盖模型，省略时沿用用户配置。Runner 保留 provider 配置，以 `--no-daemon` 直接调用 CLI；不会使用会丢失 provider 的 `--ignore-user-config`。`--codex-bin` 可选已安装 CLI 或可执行测试替身。

创建时固定 `--revision-limit 5`、`--operator-seconds 7200`、`--stage-seconds 3600`、`--call-seconds 1800`。`--max-calls 32` 是每阶段、每个 P5 节点各自的总调用预算，包含续写、修复和失败尝试；不限制整棵树的总调用数。阶段/节点从首次调用固定截止时间，算子截止时间从创建起固定。每个范围预留一次裁决；P6 另预留一次签收。各次预留时间在创建时确定为 `min(call_seconds, stage_seconds / 4, operator_seconds / 12)`，普通模型调用和检查器都不能动用；裁决时仍为 P6 签收保留时间。

`extract/resume` 输出运行 UUID、状态、checkpoint 和发布结果。退出码 `0` 表示终审已通过且已发布；`extract --queue` 的 `0` 表示已排队。`1` 表示暂停、取消、调用、回执、缺项、检查、预算停止或发布冲突；`2` 表示入口输入/工具错误。`pause/cancel` 的 `0` 只表示请求已提交，应从 `status` 核实停止位置与 `unconfirmed_attempts`；`status/inspect` 成功为 `0`。

以下检查命令已经实现。从仓库根目录运行，将 `path/to/task/root.yaml` 替换为实际文件；需要 Python 3.10+ 和 PyYAML：

```sh
python3 schema_template/check_yaml.py --skip-hardware path/to/task/root.yaml
```

检查器会扫描该文件所在目录直属的全部 YAML。完成的 P3、P5、P6 必须退出 `0`；`1` 表示输入未通过，诊断交当前修改者；`2` 表示参数、工具或环境问题，先解决检查无法完成的原因。两类失败分别记录。P1、P2、P4 按阶段产物验收，不以未完成的 Design 触发完整 YAML 检查门槛。硬件占位不改动、不审阅；GPU 验收与实测始终不在本流程内。

## 角色与会话

Supervisor 提供任务与外部材料、读状态、处理任务要求变更和发布冲突。Runner 独占写状态、派发、检查、合入及发布。worker 修改许可产物，reviewer 独立审阅；P1/P2 精确续接同角色，其他阶段及每个 P5 节点换新会话，P6 先审后修。详细会话边界见[工具协议](tools/README.md#角色与会话)。

`inspect` 调用各自冻结执行器的 `status`，默认只读运行摘要。使用 `extract --queue` 建立排队任务后，可用 `inspect --start <run-uuid>` 请求启动；Runner 在独占写锁内再次确认任务仍为 `queued`，不会把竞争期间已暂停或取消的任务恢复。排队仍沿用创建时的截止条件。此入口不安装 launchd、cron 或 Codex automation；巡检缺席不影响主流程。

## 回执与短交接

Runner 将批准绑定到具体输入、调用身份与产物字节；草稿 checkpoint 可恢复但不授予批准。机器字段、精确引文、短交接及修复边界统一见[回执协议](tools/README.md#回执与短交接)，阶段专有字段就近定义在任务页。

## 退修与裁决

退修按阶段、P5 按节点累计；达到创建时的上限后由原 reviewer 一次裁决，P6 裁决亲自改稿后另开一次独立签收。所有恢复沿用原预算和已用决定，具体边界见[退修与裁决协议](tools/README.md#退修与裁决)。

## 并行、合入与计划修订

依赖就绪的节点立即派发，不设固定 worker/reviewer 槽；单写者按内容合入，子节点只从已审直接父节点出发。阻塞只传给后代，独立兄弟继续。树、root 或 Contract 的变化经对应上游阶段重新审阅，旧结果按实际依赖保留或判迟到；详见[计划修订协议](tools/README.md#并行合入与计划修订)。

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

发布把整个已审目录及 `.extraction/receipt.json` 一次性重命名到目标。预建叶目录仅含一个零字节普通 `.gitkeep` 时，Runner 可移走占位并发布；占位身份写入发布意图，支持中断后重试。其他已有目录、文件、链接或不同成果均保留并报告冲突；普通空目录也不自动接管。若目标已经是同一完整结果，`resume` 可补写发布回执而无需再次调用模型。冲突时 `status` 给出原目标和已审 checkpoint；用下列命令另行导出，原目标和原 `--output` 保留：

```text
python3 schema_extract/extract.py export <run-uuid> --output /path/to/new/delivery
```

旧 v1 状态继续从 `inputs.lock.json` 的 `snapshot` 内调用其真实旧包，使用原 `--state-root`。例如已包含旧 `status` 的快照可运行：

```text
cd /path/from/old/inputs.lock/snapshot
python3 -B -m schema_extract --repo "$PWD" --state-root /path/to/old/.schema_extract status --help
python3 -B -m schema_extract --repo "$PWD" --state-root /path/to/old/.schema_extract status <old-run-id>
```

具体语法以该冻结版本的 `--help` 为准。早期 v1 的普通 `status` 已返回完整记录，支持 `--full` 的后期版本需添加该参数。导入器查询冻结 CLI 的能力并核对完整返回值，命令失败仍报错。旧源码、进程确认和固定 checkpoint 都齐备时，另写一份 `schema_extract.source.v2` 描述文件，`repository` 指向旧 snapshot，绑定该版本实际实现、IO、input、reference、验收和范围，然后执行：

```text
python3 schema_extract/extract.py import-legacy --from-task /old/.schema_extract/tasks/<task-key> --checkpoint <old-checkpoint-id> --source /path/to/legacy-source.json --output /new/delivery/task --queue
python3 schema_extract/extract.py resume <new-run-uuid>
```

导入要求旧任务已停止、旧 Runner 锁空闲、全部已登记调用有终止证据，并通过旧冻结入口真实读回状态；它不会替旧任务杀进程或改状态。选定源码必须与旧锁的实际字节一致，新增证据使用新的 `extract`。`imported/` 保存完整旧产物、原 checkpoint、任务、输入锁、用量、旧问题与来源；旧引用只在文件内容已核实相同时绑定到本次冻结源。root 作为 P1 草稿，旧索引和派生节点供后续阅读，新阶段不继承旧批准。若旧硬件段不是当前占位，初始草稿恢复当前 starter 并提高版本，旧字节完整保留。原用量单独留存，本次独立重审受创建时固定的新阶段预算约束。

清理运行记录前，核实任务缺项、交付位置和框架候选已留存。使用[维护入口](issues/README.md)按稳定 ID 查重；台账和历史决定集中在 `schema_extract/issues/`，迁移保留原内容、ID 和已有结论。候选 ID 不合格式时仍能读到原报告和具体原因，不阻断其他合法候选。

## 验证边界

当前 `schema_wiki` 仅保留 catalog 对应的 44 个输出目录占位，尚无提取成果。`2.1.3` 的真实多分支 P5/P6 与发布尚未完成验收；后续任务仍须经过全部阶段检查、独立终审和发布回执核对。硬件资料、GPU 正确性验收与实测不在提取流程内。
