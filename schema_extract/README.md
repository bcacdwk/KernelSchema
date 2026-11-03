# Schema Extract：Supervisor 手册

**当前版本 `2.2.0`：六阶段提取、独立语义审阅、依赖并发与原子发布。** 支持批量后台、轻量巡检、同线程持久提醒、全局 hold、显式维护接续和用量审计/证据封存；能力以各 run 的冻结摘要和 capability 为准。发布状态与恢复方法见[恢复与交付](#恢复与交付)。

每个任务始终使用创建时冻结的执行器。当前 CLI 在跳转前验证输入锁和整个快照；`extract` 创建后也进入刚冻结的入口。旧任务保留各自冻结版本的能力，升级不会由新代码直接接管旧状态；旧 v1 产物通过显式导入建立新版任务。

统一从 [extract.py](extract.py) 调用 `tools/` 中的实现。`status`、单 run 控制和 reader 进入经过核验的冻结执行器；`inspect`、`monitor scan` 直接读取轻量投影，不能授予批准。版本升级不改写旧快照。

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

最终已审 checkpoint 包含 `root.yaml`、当前计划内全部 `<id>.yaml` 与 `dispatch.md`，另在运行目录保存 P2/P3/P4 短交接。每节点的 `draft_checkpoint` 与全算子的 `merged_checkpoint` 分开；只有通过审批并合入的节点才能解锁子节点。运行资料使用独立 `.schema_extract/v2/` 与显式 `schema_extract.run.v2` 状态格式；缺少并发调用及重开账本的旧记录会要求使用其冻结执行器。批量 catalog ID 的输出按 `paths.readme` 父路径镜像到 `--output-root`；参见[输出目录导航](../schema_wiki/README.md)。

```text
.schema_extract/v2/
  format.json
  supervisor.json               # 唯一控制账本：启动、hold、观察、时钟、接续与授权重试
  supervisor.lock
  monitor-clock.lock            # 单个后台 Python 时钟持有的进程锁
  monitor-clock.log             # 仅时钟进程诊断
  snapshots/<input-fingerprint>/
    inputs.lock.json             # source、rules 的独立身份与实际文件摘要
    source/                     # 固定源码、IO、reference、验收、范围材料
    source.git/                 # 仅脏本地源码需要的独立 Git 引用
    rules/                      # 当次规范、模板、检查器、六阶段页、执行器
  runs/<run-uuid>/
    state.json                  # Runner 唯一写入的权威状态
    summary.json                # 原子观察投影，不替代权威 state 或批准
    inputs.lock.json
    attempts/<attempt-uuid>/    # 调用身份、prompt、CLI日志、结果、验证/拒绝记录
      workspace/                # 本次独立产物副本
    checkpoints/<digest>/       # 不可变产物、内容清单与来源身份
    stages/P2|P3|P4/handoff.md
    imported/                   # 可选：完整旧产物、checkpoint、task、lock及来源记录
    continued/                  # 可选：接续前的固定旧稿、state、lock和checkpoint
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
| 新版 `extract` | 单算子或 `--ids ... --output-root ...`；`--background` 登记后后台运行，`--queue` 仅排队，两者互斥。 |
| 新版 `status` | 已实现运行摘要、节点状态分类和 `--full` 权威状态读取；不会启动调用。 |
| 新版 `pause`、`resume`、`cancel` | 单任务保持调用边界与原预算；`pause --all` 返回 hold ID，`resume --all --hold ID` 只释放该次影响。 |
| `continue UUID --checkpoint DIGEST` | 以当前冻结执行器创建同源新排队 run，从 P1 独立重审，继承全部原预算。 |
| `export`、`import-legacy` | 已实现终审 checkpoint 的另行导出、已确认终止的 v1 旧稿导入与独立重审。 |
| `inspect`、`monitor scan` | 默认读投影和已登记活跃调用；`--run UUID...` 精确过滤，`--json` 输出完整观察。普通 scan 只保存有界观察；周期提交使用 `--wake ID --thread UUID`。 |
| `monitor start/status/ack/stop` | 显式绑定已有线程；后台仅计时及 `codex queue --resume`，收到提醒的同线程完成 ack 和一次 scan。 |
| `audit UUID` | 读取已记账 token/时间及会话定位；`--attempt UUID` 或 `--session UUID` 选范围，`--verify` 显式深验，`--archive PATH` 封存选中证据。 |
| `candidates`、`issues` | 只读候选和稳定 ID 台账。兼容的 `inspect --start` 只接受仍在排队的单个任务。 |
| `schema_template/check_yaml.py --skip-hardware` | 已实现；可检查现有或人工编写完成的任务 YAML，不能推进提取阶段。 |

从仓库根目录运行，需 Python 3.10+、PyYAML 和已配置 provider 的 Codex CLI：

```text
python3 schema_extract/extract.py --version
python3 schema_extract/extract.py extract --id bf16_gemm --output schema_wiki/aka-cuda-operators-20261030/operators/gemm/dense/bf16_gemm
python3 schema_extract/extract.py extract --source /path/to/source.json --output /path/to/delivery/task
python3 schema_extract/extract.py extract --ids bf16_gemm mxfp8_gemm --output-root schema_wiki --background
python3 schema_extract/extract.py extract --id bf16_gemm --output /path/to/delivery/task --background
python3 schema_extract/extract.py status <run-uuid>
python3 schema_extract/extract.py status <run-uuid> --full
python3 schema_extract/extract.py pause <run-uuid>
python3 schema_extract/extract.py cancel <run-uuid>
python3 schema_extract/extract.py resume <run-uuid>
python3 schema_extract/extract.py pause --all
python3 schema_extract/extract.py resume --all --hold <hold-uuid>
python3 schema_extract/extract.py continue <old-run-uuid> --checkpoint <digest>
python3 schema_extract/extract.py export <run-uuid> --output /path/to/another/delivery
python3 schema_extract/extract.py inspect
python3 schema_extract/extract.py inspect --run <run-uuid> <another-run-uuid> --json
python3 schema_extract/extract.py monitor scan --json
python3 schema_extract/extract.py monitor start --thread <supervisor-thread-uuid>
python3 schema_extract/extract.py monitor status
python3 schema_extract/extract.py monitor ack <reminder-id> --thread <supervisor-thread-uuid>
python3 schema_extract/extract.py monitor scan --wake <reminder-id> --thread <supervisor-thread-uuid>
python3 schema_extract/extract.py monitor stop
python3 schema_extract/extract.py audit <run-uuid>
python3 schema_extract/extract.py audit <run-uuid> --session <session-uuid> --json
python3 schema_extract/extract.py audit <run-uuid> --attempt <attempt-uuid> --verify --archive /path/to/evidence.tar.gz
python3 schema_extract/extract.py candidates <run-uuid>
python3 schema_extract/extract.py issues
```

`--repo` 指定 catalog/Git 来源仓库，默认当前目录；`--state-root` 默认该仓库的 `.schema_extract/v2`。历史来源也支持 `--source 'git:<ref>:<directory>' --bindings /path/to/source.json`，这时 Git 仓库取 `--repo`，其余角色仍由描述文件提供。

从仓库外可用脚本绝对路径启动；`--help`、`--version` 不依赖当前目录。任务命令同时显式提供 `--repo /path/to/repository --state-root /path/to/runtime`。

`--version` 输出当前入口版本，单点定义位于 `tools/__init__.py`；已有运行从其冻结入口读取版本，完整内容身份以 `inputs.lock.json` 的 `rules.id` 为准。仅有相同版本号不能代替冻结摘要核验。

`--worker-effort`、`--reviewer-effort` 分别配置，默认 `xhigh`，同角色跨阶段不变；可选 `low/medium/high/xhigh/max`。`--model` 可覆盖模型，省略时沿用用户配置。Runner 保留 provider 配置，以 `--no-daemon` 直接调用 CLI；不会使用会丢失 provider 的 `--ignore-user-config`。`--codex-bin` 可选已安装 CLI 或可执行测试替身。

创建时固定 `--revision-limit 5`、`--operator-seconds 7200`、`--stage-seconds 3600`、`--call-seconds 1800`。`--max-calls 32` 是每阶段、每个 P5 节点各自的总调用预算，包含续写、修复和失败尝试；不限制整棵树的总调用数。阶段/节点从首次调用固定截止时间，算子截止时间从创建起固定。每个范围预留一次裁决；P6 另预留一次签收。各次预留时间在创建时确定为 `min(call_seconds, stage_seconds / 4, operator_seconds / 12)`，普通模型调用和检查器都不能动用；裁决时仍为 P6 签收保留时间。

前台 `extract/resume` 的 `0` 表示已终审并发布；`extract --queue`、`continue` 的 `0` 表示排队，后台 `extract` 的 `0` 表示登记及启动请求成功，不表示提取完成。`1` 表示任务或单项启动失败、暂停、取消、预算停止、发布冲突；`2` 表示入口错误。`pause/cancel` 的 `0` 只表示请求已提交，应核实在途调用与终止证据。批量结果逐项返回身份、PID/启动身份或具体错误，独立项继续。

后台启动先持久登记 run UUID、目的地、原创建时间和启动意图，再由带令牌的子进程进入冻结 Runner；父命令退出不终止它。重复相同命令返回原身份，暂停、取消、阻塞及原 `--queue` 任务不会因此得到新预算或启动授权。启动窗口中断后用原命令对账；旧 run 无登记时先保留其目的地归属，显式 `resume` 才启动。

以下检查命令已经实现。从仓库根目录运行，将 `path/to/task/root.yaml` 替换为实际文件；需要 Python 3.10+ 和 PyYAML：

```sh
python3 schema_template/check_yaml.py --skip-hardware path/to/task/root.yaml
```

检查器会扫描该文件所在目录直属的全部 YAML。完成的 P3、P5、P6 必须退出 `0`；`1` 表示输入未通过，诊断交当前修改者；`2` 表示参数、工具或环境问题，先解决检查无法完成的原因。两类失败分别记录。P1、P2、P4 按阶段产物验收，不以未完成的 Design 触发完整 YAML 检查门槛。硬件占位不改动、不审阅；GPU 验收与实测始终不在本流程内。

## 角色与会话

Supervisor 提供任务与外部材料、读状态、处理任务要求变更和发布冲突。Runner 独占写状态、派发、检查、合入及发布。worker 修改许可产物，reviewer 独立审阅；P1/P2 精确续接同角色，其他阶段及每个 P5 节点换新会话，P6 先审后修。详细会话边界见[工具协议](tools/README.md#角色与会话)。

`inspect` 默认列举 state-root 的一级 run 名称并读 `summary.json`。投影缺失或陈旧至多回读一次 state，显示 `unknown/stale`，不调用冻结 CLI、检查器或全历史扫描。一次观察只采一次进程表；仅对已登记活跃调用读取身份、结果和日志大小/mtime。没有进程归属证据时显示 unknown；日志沉默或阶段未变不等于死锁。

`monitor scan` 保存最新短观察和最多 64 条变化；报告原 deadline 越界、Runner 已退出及活 Runner 无调用且持续无状态进展等诊断线索，不据此终止进程。默认范围动态包含全部 run，能看到新建 successor；显式 `--run` 仅观察所列 UUID，不自动扩展到接续链。前任显示 `superseded/continuing` 和 successor，不重复告警其历史失败。

## 同线程持久提醒

`monitor start --thread UUID [--run UUID...]` 绑定当前 Supervisor 的真实线程。首次 start 在前台完成检查 #0，再启动一个独立后台 Python 时钟；之后三个间隔分别为 300 秒，检查 #3 完成后改为每 1,800 秒。每次都从成功扫描提交时计算下一次时间；晚点只处理一轮，不补发积压。后台提取与时钟是两个独立生命周期。

收到提醒后执行消息中的 `monitor ack ID --thread UUID`。仅 `accepted=true` 才执行一次 `monitor scan --wake ID --thread UUID`；它使用 start 登记的范围，原子提交短观察、完成号和 `next_due_at`，然后结束本轮对话。`accepted=false` 时结束该提醒。普通 `monitor scan` 不消耗周期，也不调用自动恢复；不要用它代替到期提交，不做对话内 sleep 或另建闹钟。

每轮查看各算子的阶段、在途调用和异常锚点，正常时简短报告进展。单个分支 blocked 先定位其缺口，不因此全停。需要修复共享框架时，按影响范围显式 hold，核对 `pause --all` 返回的影响集合、已停和 pending 项，再按已有工程授权交接问题、证据与恢复条件；扫描器和时钟不自行改工具或派 Agent。尊重用户最新的暂停和讨论；完成本次提醒不意味着丢下正在处理的已授权事项。

ack 后中断会显示 `acknowledged_unfinished`。同线程显式执行原 `scan --wake ID --thread UUID`，或 `scan --thread UUID` 续完当前已接受轮次；重复 ack 仍返回 false。扫描尚在另一个存活进程中时返回 `scan_busy`；进程已退出可续做，已完成 ID 的重复提交不再推进计数。周期扫描不接受临时 `--run` 改范围。

`monitor status` 分别展示 PID/启动身份、实际进程锁、心跳、晚点、发送尝试数、消息 ID 和线程 ack。Linux 优先使用进程起始 tick 与 boot ID；其他 Unix 沿用现有 `ps` 的 PID/启动时间，读取失败显示 unknown。身份与锁不一致显示 `identity_lock_mismatch`；时钟处理异常显示 `clock_error` 和 `needs_attention`，成功 tick 后才清除。队列确认只表示已入队；`queued_unacknowledged` 尚未证明线程实际收到。发送前先记次数，失败或不确定送达沿用同 ID，最多 3 次且间隔至少 120 秒；成功入队后只等 ack。次数耗尽，或 1,800 秒仍无 ack/扫描完成时显示 `needs_attention`，不会无限重送。

时钟退出或主机重启后需显式 `monitor start --thread UUID` 恢复原进度；状态文件本身不表示进程仍运行，也不能证明 Codex app-server 存活。重复 start 复用实际持锁的同线程时钟。换线程或范围使用 stop/start；`monitor stop` 先使旧提醒失效，再停止时钟，不暂停、取消或恢复提取。完成号、原下一次时间、累计发送尝试数和自动恢复用量均保留，换代后须新的有效 ack 才重新标记该线程已验证。

显式 `--auto-restart-limit N` 只授权该次 start 选定的 run；省略 `--run` 时授权当时已登记的名单，新 run 不自动获得重试权。授权只在初始和已接受的周期扫描提交后应用，并重核失败范围、原预算、hold 和手动控制；默认关闭，重复 start 不退款。`--codex-bin` 可指定可执行队列替身。可选 `--probe-delay 1..300` 只增加同一代际的一次线程验证提醒，ack 后查看 status，不消耗正式检查号或改变生产间隔。

时钟/ack/崩溃/重试由假时间和可执行队列替身验收。30 秒真实 probe 已在 Supervisor 结束上一轮对话后唤醒同一线程，ack 被接受，停止前 `verified=true`；正式检查号与下一次时间保持不变。随后已 stop，进程退出、锁释放，旧 probe 再次 ack 被拒绝。该验证没有创建提取任务。

## 回执与短交接

模型返回带本次 `attempt_id` nonce 的语义提案，自行选择证据范围并解释结论；Runner 从固定调用和真实字节生成完整回执、摘要及精确引文。批准始终绑定 reviewer 实际读取的 checkpoint，原始 response/result 保持不变；草稿 checkpoint 可恢复但不授予批准。通用协议与修复边界见[回执协议](tools/README.md#回执与短交接)，阶段专有字段就近定义在任务页。

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

## 审计与封存

`audit UUID` 默认只读取该 run 的 `attempt_usage` 与现有 Supervisor 检查锚点，按阶段、角色、节点给出短摘要；`--json` 包含逐调用定位。`--attempt` 和 `--session` 互斥，继承调用按 `origin_run_id/attempt_id` 去重，分别列出本段与继承用量。会话记录区分请求和实际 UUID；继承会话交接指向原 run 的 state，显式深验时读取原文。旧记录缺字段就显示 unknown，旧 session 尚未索引时说明筛选可能不完整；不会为报表启动调用、恢复任务或补写旧记录。

`result.json` 保存提供者在 `turn.completed.usage` 报告的原值、字段来源及事件行/字节锚点。`input_tokens` 已含 `cached_input_tokens`，两者不能相加。reported 仅表示事件字段齐备，失败或截断为 partial，缺 usage 为 unknown。有界真实 bf16 P1 调用已保存一条完整 usage 及提供者扩展字段，两个超时调用保持 unknown。新会话与 resume 的计数范围仍未确认，`additive_tokens` 保持未知，逐次原值可查；不据此估算总 token、完整账单或美元费用。

CLI 时间包括配置查询、调用、终止和日志解析；检查/处理时间来自 Runner 的 `processing.json`，崩溃后的保守结算单独标记。二者可累计比较阶段成本，但并发之和不是算子墙钟。可核实发布提交时，墙钟从原算子创建计至该提交，包含排队和暂停；其他情况只给创建至本次观察的间隔，最终墙钟未知。暂停等待与纯模型执行时间没有可靠独立采样，保持未知；原 deadline 和预算高水位始终保留。Supervisor 检查时间为 state-root 共享的有界历史，不分摊到某次模型调用。

`--verify` 只检查选中调用及其输入/结果 checkpoint、来源/规则和引用证据，复用现有快照、checkpoint 与引文门禁；不授予批准或推进阶段。普通 audit 不读大日志。显式深验和封存允许对选中对象做门禁复验后再流式采集，避免另一套校验规则；不扫描其他调用或 provider 私有历史。

`--archive PATH` 同时深验，生成 `tar.gz`，内含 `manifest.json` 和按 SHA-256 去重的 `objects/`。清单记录范围、原位置、字节数及摘要；单调用包保存该调用的输入/已登记结果 checkpoint，全 run 包另保存当前/最终 checkpoint。源码、规则及工具自身的私有 `source.git` 来自冻结快照；同内容只保存一次。包用于搬运审计证据，不是可执行恢复目录。目标须在 runtime 外、父目录已存在，路径不能含 `..` 或链接；已有相同且可验证的清单返回 reused，不同或损坏目标保留并报错。缺对象或选中范围仍活动/未结时不写包，运行记录始终保留。旧记录缺原始摘要仍可封存当前可验证对象，但清单明确标为 partial。

audit 返回 `0` 表示读取/所选可得证据检查完成，旧字段的 partial 仍须看 `unavailable`；证据不符、不完整或封存未写出返回 `1`，入口错误返回 `2`。封存清单与摘要不能替代 Runner 的独立审批与发布回执。

## 恢复与交付

`pause <run-uuid>` 和 `cancel <run-uuid>` 只向 `requests/` 追加请求。已有 Runner 消费请求；若它已退出，命令进入同一 Runner 写锁完成对账和停止。暂停不再派发下一调用，已启动的调用可完成并保存结果。取消通知全部在途调用终止，并在各自 `result.json` 或 `termination.json` 保存确认；无法确认时保留该范围的身份。

`resume <run-uuid>` 先确认旧调用及其已记录子进程终止。调用已启动但进程身份缺失或无法核实时保留 fence；不能凭“没有看到进程文件”断言已终止。其他依赖就绪节点继续。旧调用留下合法草稿时保存到 checkpoint，否则继续使用上一个安全 checkpoint，并在 `recovery.json` 说明丢弃原因。已经使用的一次裁决或签收若没有可验证的完成结果，就阻塞，不重新授予。

明确的人工 `resume` 继续暂停、取消、排队及未结调用，也可在环境修复后重试普通调用/工具停止；它记录本次全任务恢复授权，但不退款失败计数、调用、时间或已用决定。预算、外部确认与一次性决定的硬停止仍保留。hold 释放和自动接续不因此解封其他错误范围。`resume` 始终使用旧冻结执行器；工具修复需要新执行器时，显式 `continue OLD --checkpoint DIGEST`：旧 run 必须静止，写锁可得，全部调用与计时已结算，旧启动进程也不能仍在认领或 exec 窗口。新 run 使用完全相同的源码字节和当前规则，从 P1 新会话独立重审，不继承有效批准。原截止、退修/调用/重开计数、已用时间、裁决/签收原文与预留、删除节点账本和自动恢复 lineage 上限都保留。余额不足则新 run 明确 blocked。

接续先登记唯一 successor，构建期间拒绝启动或从半成品再次接续；中断后相同命令补全同一个身份。完成后显式 `resume NEW` 才运行；统一入口拒绝旧 run 再派发或发布，保留其只读查看与停止操作。旧 `state`、快照、批准原文不改写；`continued/` 保存所选旧稿和溯源证据。相关阶段通过 `continued_draft.artifacts` 读取列明的旧稿；P5 仅给本节点与祖先线索，旧状态和批准不授予新权限。不能通过再次接续或恢复重置预算。

`pause --all` 先持久关闭派发和发布闸门，再登记每项影响并追加带 hold ID 的暂停请求。新 Runner 在派发授权及发布最终 rename 前与 hold 使用同一控制锁；已经提交的发布记为 hold 前完成。已启动调用可按原边界结束，尚未停止的项持续显示 pending。`resume --all --hold ID` 只恢复能核实由该 hold 暂停且没有后续手动控制的项；原阻塞、手动暂停、取消、未知终止均保留。仅阻止排队的项按原启动授权处理，`--queue` 仍留在队列。释放时仍在途的项先保留，之后重复同一释放命令可重新核对。

旧冻结执行器缺少 `hold.v1` 时明确显示不支持闸门，只能通过已有 pause 请求在边界停止；未停止前不承诺已拦住其发布，释放时要求手动确认恢复。单 run resume/export 也受活动 hold 约束。一个算子或分支 blocked 不自动全停；共享工具风险由 Supervisor 判断是否显式 hold 后交给工程修复。

自动接续默认关闭。当前只把已确认终止的明确进程退出码 `75`（EX_TEMPFAIL）识别为暂态；在原普通重试余额内自然停止该 scope，其他就绪分支继续。时钟 start 的显式 `--auto-restart-limit N` 通过控制账本授权所选 run；请求绑定失败 attempt、scope、真实 result 摘要和控制修订，Runner 再次核验。普通两次重试、原 deadline、一次性决定和持久 lineage 总上限都继续生效；重启观察/时钟不退款。超时、通用 CLI 错误、坏回执、未知终止或工具缺陷只诊断/交接，不自动接续。实际 provider 是否使用退出码 75 尚需真实调用验证。

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

受控验收已覆盖四节点 P1–P6、兄弟并发与父子解锁、独立终审/发布、五次退修后的单次裁决和独立签收，以及后台 hold、两次 resume 不重复记账、接续、轻量扫描和封存复用。这些调用使用可执行替身，产物通过实际冻结检查器。

有界真实 bf16 P1 共调用三次：worker 首次超时后保存合法草稿，同一 run 按原预算、原 UUID 恢复并提交通过 Runner 门禁的 complete 提案；独立 reviewer 使用新 UUID，在剩余普通调用预算（约 43 秒）到期后超时，裁决预留未动用。任务停在 `paused / P1 / reviewer`，三个调用均已确认终止并结算，活动、未确认和未结调用均为空；原 deadline 保留，audit 深验和证据封存通过。真实 P1 尚未获得 reviewer 批准，没有进入 P2/P6 或发布。同线程提醒已完成上述[真实 probe 验证](#同线程持久提醒)。

当前 `schema_wiki` 仅保留 catalog 对应的 44 个输出目录占位，尚无提取成果。完整真实多分支提取仍未验收；每个实际任务须经过全部阶段检查、独立终审和发布回执核对。硬件资料、GPU 正确性验收与实测不在提取流程内。
