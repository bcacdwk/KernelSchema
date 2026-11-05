# Schema Extract：Supervisor 手册

版本 `2.4.1`。入口 [extract.py](extract.py) 从固定 kernel 提取 schema，Runner 推进写作、独立审阅、返工与发布。本页说明操作；[提取总览](../EXTRACT_SCHEMA.md)定义 P1–P6，[工具协议](tools/README.md)说明回执与执行门禁。

## 开始一次试用

需要 Python 3.10+、PyYAML、Git 和已配置 provider 的 Codex CLI。从仓库根目录执行：

```sh
python3 schema_extract/extract.py extract --id bf16_gemm --output schema_wiki/aka-cuda-operators-20261030/operators/gemm/dense/bf16_gemm --background
python3 schema_extract/extract.py status RUN_UUID
```

第一条返回 run UUID、固定输入身份和后台启动结果；后续用该 UUID 查询和恢复。`status` 只读，`resume RUN_UUID` 在前台执行到交付或停止。**`ready` 表示终审通过，`published: true` 表示已发布到原输出目录。** 交付包含 `root.yaml`、计划内全部派生 YAML、中文 `dispatch.md` 和 `.extraction/receipt.json`。

批准表示忠实表达固定实现及其限制，来源缺项按[统一边界](../EXTRACT_SCHEMA.md#来源缺项与验收边界)处理。提取不承担 GPU 验收或复现未提供的外部验收器。

## 输入与输出

Supervisor 提供 catalog ID 或明确来源、输出目录及适用范围。材料须定位选定实现、host 入口、任务 IO、输入构造、reference、数值验收材料和适用范围。[算子索引](../kernel_wiki/CATALOG.md)提供 catalog ID；本地 catalog 算子目录也可直接作为 `--source`。

其他来源使用 `schema_extract.source.v2` JSON 描述文件，或在来源目录放 `schema-inputs.json`。角色路径相对 `directory`，`repository` 相对描述文件所在目录，省略为 `.`；`host` 省略时使用 `entrypoint`。同一文件中的不同函数可以分别充当实现和 reference。

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

省略 `revision` 使用本地实际字节，填写时读取解析后的 Git commit。也支持 `--source 'git:<ref>:<directory>' --bindings source.json`，Git 仓库由 `--repo` 指定。文件名和函数签名不能代替角色绑定与任务材料。

运行中区分三种冻结身份，每个调用都绑定自己的上下文：

| 身份 | 内容与变更入口 |
| --- | --- |
| source/config | 原源码字节、输入角色、任务目标、配置及检查政策；沿用原 `inputs.lock.json`。真实来源或任务改变须换输出目录重新 extract |
| rules | 规范、starter、检查器、阶段页与提示材料；初始来自原锁，连续修订通过 `semantic_revision` 追加目标规则快照 |
| executor | 实际运行的冻结代码与能力；兼容修复通过 `execution_binding` 登记，语义修订也记录采用的执行器 |

快照记录逐文件摘要和整体身份，包含冻结时未提交的规则与工具改动。脏本地源码用快照内独立 `source.git` 固定引用，原仓库不变。catalog 的可选 `paths.annotated_entrypoint` 作为阅读辅助一同收集，不替代原实现，也不自动证明二者等价。旧调用、回执和批准始终按原身份解释，工作树修改不会自动作用于已有 run。

资料默认位于 `.schema_extract/v2/`：`snapshots/` 保存固定输入，`runs/RUN_UUID/` 保存状态、调用证据和 checkpoint，`supervisor.json` 保存控制记录。合法草稿与有效合入分别记录；只有有效批准能解锁依赖和交付。批量输出映射见[输出目录导航](../schema_wiki/README.md)。

## 命令与参数

```sh
python3 schema_extract/extract.py extract --source /path/to/source.json --output /path/to/delivery
python3 schema_extract/extract.py extract --ids bf16_gemm mxfp8_gemm --output-root schema_wiki --background
python3 schema_extract/extract.py inspect --run RUN_UUID --json
```

单任务默认前台执行；`--queue` 只冻结并排队，与 `--background` 互斥。`--ids` 需要 `--output-root` 和 `--background` 或 `--queue`，各项独立返回身份、启动结果或错误。`--repo` 默认当前目录，`--state-root` 默认该仓库的 `.schema_extract/v2`；从仓库外启动时使用脚本绝对路径并显式指定两者。`--help`、`--version` 可独立使用。

| 参数 | 默认值与含义 |
| --- | --- |
| `--worker-effort`、`--reviewer-effort` | `xhigh`；可选 low/medium/high/xhigh/max，同角色跨阶段一致 |
| `--model`、`--codex-bin` | 模型沿用用户配置，CLI 默认 `codex`；保留 provider 设置 |
| `--revision-limit` | 5；每阶段、每个 P5 节点的实质退修上限 |
| `--operator-seconds` | 7200；从创建开始的算子墙钟预算 |
| `--stage-seconds` | 3600；阶段/节点从首次调用开始的墙钟预算 |
| `--call-seconds` | 1800；单次调用上限 |
| `--max-calls` | 32，最少 3；每阶段/节点总调用上限，含续写、修复、失败及预留调用 |

排队、暂停和维护等待均计入墙钟预算。派发为后续审阅、裁决及 P6 签收保留额度，余额不足则停止；公式见[预算协议](tools/README.md#退修与裁决)。

`resume RUN_UUID --add-seconds N` 将算子及所有已开始阶段/节点的截止时间分别延至 `max(当前时间, 原截止) + N`；未开始阶段仍用原 `stage_seconds`。`--add-calls N` 给每阶段/节点（含后来建立的节点）增加 N 次上限。追加记入 `budget_extensions`，保留初始配置、已用计数和单次决定。每次新的 `resume --add-*` 是一次追加授权，重放同一持久授权只记一次；不确定是否生效时先查 `status --full`。

普通恢复、范围恢复、重开、hold 释放和自动重试不增加预算。默认 `continue` 沿用原余额，不能附 `--add-*`；显式 `continue --full-restart` 可在继承原账目后追加一次。

### 重复命令与退出码

重复 extract 沿用登记时的固定源码、规则和参数，不重新读取工作树。结果中的 `result/run_status/published/next_step/frozen` 分别说明命令结果、保存状态、交付与身份；请求参数改变时报告原目的地占用。

| 命令 | 退出 0 的含义 |
| --- | --- |
| 前台 extract / 普通 resume | 已终审并发布 |
| `resume --scope` | 范围恢复已授权或消费，且未报告启动失败 |
| `extract --queue` | 已排队，或重复请求已发布的结果 |
| `extract --background` | 启动请求成功、已有任务运行、被 hold 保留或已发布 |
| `continue` / `revise` | 请求、排空、待评估计划或采纳状态已返回；`delivered:false` 仍未交付 |
| `continue --full-restart` | successor 已登记，尚未执行到交付 |
| `pause` / `cancel` / `resume --all --hold ID` | 控制请求已处理，仍须核实受影响任务的实际状态 |
| `compatibility-report` / `upgrade-runtime` | 报告生成/核验或执行绑定登记完成；未启动任务 |
| `status` / 普通 `inspect` / `monitor` | 查询或控制操作完成，不代表提取完成 |

前台复用正在运行的任务返回 `1 / already_running`。执行停止、预算耗尽、发布冲突及未获启动授权的重复请求也可返回 `1`；入口错误返回 `2`。audit 的证据完整性另见[审计与封存](#审计与封存)。

## 角色与会话

Supervisor 在既定任务语义和授权内主动决定预算追加、故障恢复、必要修复及合理取舍，记录依据、改动和验证结果，并处理归档 issue。只有确需用户决定且无法据现有授权解决的分歧才交回用户。工程维护可委派子代理实施，Supervisor 验收实际差异和证据。

Runner 唯一写状态、派发、检查、合入和发布。extract worker 只写本次许可产物，reviewer 按 [REVIEW.md](REVIEW.md)独立审阅；两者都不启动子代理，框架问题提交 `framework_candidates` 交 Supervisor 处理。先判断能否按既有规则修正产物，规则确有缺口才组织维护。

常规提取按 P1–P6 推进；已批准的依赖解锁下一阶段或子节点。连续修订临时执行有限维护待办，完成后回到实际剩余工作，最终仍由 P6 交付。会话创建、续用和退修裁决的规则集中在[工具协议](tools/README.md#角色与会话)。

## 状态阅读

`status RUN_UUID` 展示阶段、角色、checkpoint、有效完成数和停止原因；`--full` 展示权威状态。P5 以 `nodes_by_status` 和 `node_details` 为准，后者给出父节点、调用和具体原因。状态包括 queued/running/paused/cancelled/ready，以及 blocked/budget_exhausted/tool_error/cli_error/invalid_receipt/termination_unconfirmed。

先读 `reason` 和节点原因。仍有 `active_attempts`、`unconfirmed_attempts` 或 `unsettled_attempts` 时，停止状态不表示调用已结算。`inspect --run UUID...` 读取轻量投影；`monitor scan` 另保存短观察和最多 64 条变化，默认不恢复任务。陈旧投影或证据不足显示 stale/unknown；日志沉默不能证明死锁。前任已有 successor 时先查看接续身份。

## 恢复与交付

下表命令均接在 `python3 schema_extract/extract.py` 后：

| 场景 | 操作 |
| --- | --- |
| 在调用边界暂停 | `pause RUN_UUID`；已启动调用可完成并保存结果 |
| 请求终止在途调用 | `cancel RUN_UUID`；核对 termination 确认证据 |
| 恢复排队、暂停或已处理的普通异常 | `resume RUN_UUID`，使用已登记的冻结执行器 |
| 只恢复一个普通失败范围 | `resume RUN_UUID --scope P5:NODE_ID`；由唯一 Runner 结算并消费 |
| 余额不足 | `resume RUN_UUID --add-seconds 7200 --add-calls N`，按实际缺额选参数 |
| 采用兼容执行器修复 | `upgrade-runtime RUN_UUID --report REPORT.json --review REVIEW.json` |
| 未完成任务采用新规则或修复语义问题 | `continue RUN_UUID --reason '具体理由'`，独立评估后 `continue RUN_UUID --plan PLAN.json` |
| 从 P1 完整重提取 | `continue RUN_UUID --full-restart --checkpoint DIGEST --reason '具体理由'`，再 `resume SUCCESSOR_UUID` |
| 原输出目录冲突 | `export RUN_UUID --output /path/to/new/delivery` |

恢复先对账已登记调用的进程身份、终止和处理记录，再采用合法草稿或安全 checkpoint。发射后身份缺失保持 fence；计时按 attempt 高水位去重。范围恢复绑定失败 attempt/result、节点代际、父依赖、执行绑定及控制修订；健康兄弟可继续。原 Runner 已退出时由同一写锁接管，沿用原单次超时、阶段及算子截止时间。`--scope` 不接受预算追加；pause/cancel/hold 和已用单次决定继续生效。P1–P4/P6 的恢复 scope 直接使用阶段名，P5 使用 `P5:NODE_ID`；它与修订计划中的 `P3:root` 等 scope 名不同。

兼容升级先用 `compatibility-report RUN_UUID --checkpoint DIGEST` 生成完整差异报告，可加 `--target-repo PATH`。独立 Supervisor 审阅全部差异并保存原始 JSON，格式见[兼容审阅协议](tools/README.md#兼容差异与审阅记录)。`upgrade-runtime` 须等旧写者、启动窗口与全部调用结清，且无待处理语义修订或维护；登记保留 UUID、原输入、有效批准、checkpoint 和账目，追加执行绑定并退休旧角色会话。它不启动或解除控制；随后显式 `resume`。未知影响或语义变化使用连续修订。

连续修订限于**请求时未完成**的 run：ready、published、已有 export 或已提交发布 rename 的任务不接受。原来源、任务与 config/check_policy 固定，目标规则由 `--target-repo PATH` 选择，省略时使用当前执行器所在规则树。完整步骤也可分开调用：

```sh
python3 schema_extract/extract.py revise request RUN_UUID --reason '具体问题与采用目标' --target-repo /path/to/target
python3 schema_extract/extract.py revise prepare RUN_UUID
python3 schema_extract/extract.py revise prepare RUN_UUID --plan PLAN.json
python3 schema_extract/extract.py revise apply RUN_UUID
python3 schema_extract/extract.py revise status RUN_UUID
```

请求登记即拦新派发和发布；在途调用按原上下文和截止时间结算，安全排空后才把计划绑定到实际稿件与批准。无独立评估时返回 `awaiting_evaluation` 与计划路径。Supervisor 评估完整差异及问题，逐范围选择 `keep/review/patch/reextract`；旧批准的真实性与当前适用性分别核验，只有有据采纳才继续有效，依据不足先 review。计划格式与维护动作见[维护协议](tools/README.md#多任务控制与维护接续)。

完整候选可直接交 Reviewer；未完成草稿保留内容并继续原作者工作。P5 优先使用自己的稿件，只刷新只读祖先；父变化产生复核候选，不覆盖孩子的完整 Design。维护后的批准仍覆盖本阶段全文，最终交付仍需当前上下文的 P6。这里的 CLI `continue` 用于修订规则/适用性；worker 回执 `decision=continue` 仅保存本步骤草稿并续写。

采纳后仅在原运行意图仍有效且无新控制时接续后台任务；暂停任务只登记，之后由 Supervisor 显式恢复。活动维护结束或安全 paused/cancelled 且调用全部结清后，可通过 `revise request` 追加替代修订。相同目标、范围、问题证据和动作不因换请求 ID 获得新机会，原调用、时间、退修、P6 starts 和单次决定均保留。冻结执行器须支持 `semantic_revision.v1`；旧执行器先在安全边界采用已审兼容运行层。

`--full-restart` 是单独的显式选择：创建从 P1 重审的 successor，不继承批准，封存原稿和原账目并继承用量、预算追加与已用决定。同一命令可补全中断构建；返回 0 后使用返回的 successor UUID 启动。

`pause --all` 返回 hold ID，拦新派发与发布；`resume --all --hold ID` 只释放该次影响，保留后来的手动控制、原阻塞与未知终止。仍在途的项停止后可重复释放命令核对。旧冻结执行器缺少 hold 能力时，须按其原有 pause 边界处理。

发布原子重命名完整已审目录及回执。目标须不存在，或仅含一个零字节普通 `.gitkeep`；其他目录（包括空目录）、文件及链接均报告冲突。同一完整结果可幂等重放；export 另行交付，不改变原输出位置。v1 兼容导入用 `import-legacy --help`：原任务与调用须确认终止，来源绑定原快照；导入稿留作证据，从 P1 重审，旧用量单列，本次使用新建预算。

## 单文件阅读视图

需要把一个节点交给只拿到单份 YAML 的读者时：

```sh
python3 schema_extract/extract.py standalone path/to/task/branch.yaml --output path/to/views/branch
python3 schema_template/check_yaml.py --standalone --skip-hardware path/to/views/branch/branch.yaml
```

生成器检查创作任务，保留节点元数据、Design 和 Realization，展开当前 root 的共享 Contract；不继承 parent Design。任务外的输出目录保存原文件名 YAML 和 `.source.json` 摘要，相同输入可复用，冲突内容保留。视图保持原选择范围，不产生新的 P6 批准；发现语义问题须修订创作源并重审，再生成视图。

## 同线程持久提醒

```sh
python3 schema_extract/extract.py monitor start --thread THREAD_UUID
python3 schema_extract/extract.py monitor status
python3 schema_extract/extract.py monitor stop
```

start 绑定 Supervisor 的真实线程，可用 `--run UUID...` 限定范围。完成检查 #0 后，间隔依次为 300/300/300 秒，此后每 1800 秒一轮，从成功扫描提交时计时；逾期只处理一轮。后台另做有界观察和已授权恢复，通过 `codex queue --resume` 投递提醒。

收到提醒后执行其中绑定了 repo/state-root 的 `monitor ack ID --thread THREAD_UUID`；accepted=true 且 scan_required=true 时执行一次 `monitor scan --wake ID --thread THREAD_UUID`。ack 后中断可用原 scan 续完；若重复 ack 返回 already_acknowledged 且 unfinished=true，同样续做原 scan。其他拒绝不推进周期。scan 返回 `clock.committed=true` 或 `reused=true` 才表示本轮已登记；busy/invalidated 时查 monitor status。普通 scan 不消耗周期。可选 `--probe-delay 1..300` 只验证唤醒，不触发正式检查。

同 wake ID 最多发送 3 次：不确定发送后至少等 120 秒，已入队未 ack 则等 1800 秒，均从最近尝试结束（未结束则开始）计算。ack 后停止补送；耗尽仍等显式 ack/scan，不自动推进周期，后台观察继续。status 给出真实进程、发送次数、ack/完成进度与 needs_attention；入队确认不保证线程已处理。进程退出后显式 start，换线程或范围用 stop/start；stop 使旧提醒及恢复授权失效，不终止已有提取调用。

自动恢复默认关闭。`monitor start --auto-restart-limit N` 仅授权所选 run（未选范围则为当时已登记任务）中已确认终止、返回 EX_TEMPFAIL 75 且有余额的普通失败，按 lineage 累计，重启不退还次数。提醒补送与已修配置问题的历史依据见[归档记录](issues/archive/extract_tools/reminder-ack-delay.json)。

## 审计与封存

```sh
python3 schema_extract/extract.py audit RUN_UUID
python3 schema_extract/extract.py audit RUN_UUID --attempt ATTEMPT_UUID --verify --archive /path/to/evidence.tar.gz
```

audit 默认读已记账摘要，按阶段、角色、节点和来源展示用量；`--attempt` 与 `--session` 互斥，`--json` 给出定位。`--verify` 深验选定调用、checkpoint、来源和引文，不推进阶段。

`result.usage` 保留提供者原口径；`usage-observation.json` 另记本次根会话的有据增量，按来源/attempt 和会话区间去重。根范围分 exact/partial/unavailable，只有非空范围内全部已登记调用 exact 才称根统计完整；cached input 已包含在 input 中。委派观察的 observed/not_observed_in_range/unavailable 仅说明已读范围，禁止委派或未发现标记都不能证明子会话为零。完整调用 token 与外部实际费用仍未知，根小计不代表完整账单。旧采样不会自动补齐，运行层升级仅影响未来调用；细节见[用量协议](tools/README.md#usage时间与证据封存)。

审计分列墙钟、CLI、检查处理及已知等待区间；等待可与兄弟工作重叠，不能相减推断纯模型时间。archive 同时深验，生成 `manifest.json` 与按摘要去重的 `objects/`；目标须在 runtime 外，父目录已存在，路径无 `..` 或链接。同一有效包可复用，冲突、缺失或活动/未结算范围不写包。封存用于审计，不提供执行恢复授权，也不删除运行记录。

audit 退出 `0` 表示读取或可得证据检查完成，仍须看 unavailable；证据不完整、不符或封存失败为 `1`，入口错误为 `2`。

## 验证边界

P3/P5/P6 的完成、批准与退修由 Runner 自动执行冻结检查器，放行须 `--skip-hardware` 退出 0；1 是产物诊断，2 是工具/环境问题。P1/P2/P4 使用阶段门禁。结构通过仍须独立语义审阅。

实际检查证据在 `attempts/ATTEMPT_UUID/check.json` 和 `validated.json`，P5 合入另有 `merge_check.json`。回执 `checks` 只记录模型亲自执行的命令，空列表不代表 Runner 漏检。最终 P6 与交付回执绑定同一完整文件集合。

维护验证方法见[离线验证](tools/README.md#离线验证)。历史结果分别保留在[试用记录](TRIAL_REPORT.md)、[算子审阅](OPERATOR_REVIEW.md)、[Reviewer 验证](review_cases/RESULTS.md)、[问题与决定](issues/TRIAGE.md)及[已完成连续修订设计](IMPROVEMENT_PLAN.md)，不作为当前 run 已通过的证据。
