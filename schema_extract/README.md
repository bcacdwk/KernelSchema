# Schema Extract：Supervisor 手册

版本 `2.2.10`。从固定 kernel 提取可读、可追溯的 schema，自动完成写作、独立审阅、返工与发布。统一入口是 [extract.py](extract.py)，实现位于 `tools/`。

## 开始一次试用

需要 Python 3.10+、PyYAML、Git 和已配置 provider 的 Codex CLI。从仓库根目录执行：

```sh
python3 schema_extract/extract.py extract --id bf16_gemm --output schema_wiki/aka-cuda-operators-20261030/operators/gemm/dense/bf16_gemm --background
python3 schema_extract/extract.py status RUN_UUID
```

第一条返回 run UUID、固定输入身份和后台启动结果；后续以该 UUID 查看和恢复。后台进程自行推进，`status` 只读进度。暂停或处理异常后用 `resume RUN_UUID`，它在前台执行到交付或停止。巡检、同线程提醒和审计按需使用。

**`ready` 表示终审通过，`published: true` 表示已发布到原输出目录。** 交付包含 `root.yaml`、计划内全部派生 YAML、中文 `dispatch.md` 和 `.extraction/receipt.json`。

提取批准表示忠实表达固定实现及其来源缺口。外部验收器未提供的定义可按[统一边界](../EXTRACT_SCHEMA.md#来源缺项与验收边界)记录并继续；计算或设计必须靠猜测才能确定时仍阻塞。交付不宣称已复现原验收器，用于新设计或实现验收前仍需完整判据。

流程见[提取总览](../EXTRACT_SCHEMA.md)，各阶段的方法分别在：

| 阶段 | 任务页 |
| --- | --- |
| P1 | [共享 Contract](phases/01_contract.md) |
| P2 | [dispatch 草案](phases/02_dispatch_draft.md) |
| P3 | [root 设计](phases/03_root.md) |
| P4 | [dispatch 定稿](phases/04_dispatch_plan.md) |
| P5 | [派生节点](phases/05_branch.md) |
| P6 | [全算子终审](phases/06_final.md) |

## 输入与输出

Supervisor 提供 catalog ID 或明确来源、输出目录及适用范围。来源须能定位选定实现、host 入口、任务 IO、输入构造、reference、数值验收材料和适用范围。[算子索引](../kernel_wiki/CATALOG.md)提供 catalog ID；本地 catalog 算子目录也可直接作为 `--source`。

其他来源使用 `schema_extract.source.v2` JSON 描述文件，或在来源目录放 `schema-inputs.json`。角色路径相对 `directory`，`repository` 相对描述文件所在目录，省略为 `.`；`host` 省略时使用 `entrypoint`。同一文件中可定位的不同函数可以分别充当实现和 reference。

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

省略 `revision` 使用本地实际字节，填写时读取解析后的 Git commit。也支持 `--source 'git:<ref>:<directory>' --bindings source.json`，Git 仓库由 `--repo` 指定。文件名和函数签名不能代替输入角色与任务材料。

Runner 冻结源码及本次规范、starter、检查器、阶段页和执行器，分别保存逐文件摘要与整体身份；规范和工具包含当时未提交的改动。catalog 提供可选的 `paths.annotated_entrypoint` 时，新快照同时收集该校注文件作为阅读辅助，原实现与其他输入角色绑定保持不变；不追改旧快照，也不自动证明校注与原件等价。脏本地源码使用快照内的独立 `source.git` 固定引用，不修改用户仓库。所有阶段读取同一快照。

运行资料默认位于 `.schema_extract/v2/`：`snapshots/` 保存固定输入，`runs/RUN_UUID/` 保存状态、调用证据和 checkpoint，`supervisor.json` 保存控制记录。草稿与有效合入分别记录；草稿不能解锁子节点或作为正式交付。批量输出按 catalog 路径映射到 `--output-root`，见[输出目录导航](../schema_wiki/README.md)。

## 命令与参数

```sh
python3 schema_extract/extract.py extract --source /path/to/source.json --output /path/to/delivery
python3 schema_extract/extract.py extract --ids bf16_gemm mxfp8_gemm --output-root schema_wiki --background
python3 schema_extract/extract.py inspect --run RUN_UUID --json
```

单任务省略 `--background` 时前台执行；`--queue` 只冻结并排队，两者互斥。`--ids` 需要 `--output-root` 和 `--background` 或 `--queue`。批量结果逐项返回身份、启动状态或错误，独立项继续。

`--repo` 默认当前目录，`--state-root` 默认该仓库的 `.schema_extract/v2`。从仓库外启动时使用脚本绝对路径，并显式指定这两个参数。`--help`、`--version` 可独立使用。

| 参数 | 默认值与含义 |
| --- | --- |
| `--worker-effort`、`--reviewer-effort` | `xhigh`；可选 low/medium/high/xhigh/max，同角色跨阶段保持一致 |
| `--model` | 沿用用户配置；Runner 直接调用 Codex CLI，保留 provider 设置 |
| `--codex-bin` | `codex`；可指定已安装 CLI 或可执行测试替身 |
| `--revision-limit` | 5；每阶段、每个 P5 节点的实质退修上限 |
| `--operator-seconds` | 7200；从创建开始的算子墙钟预算 |
| `--stage-seconds` | 3600；阶段/节点从首次调用开始的墙钟预算 |
| `--call-seconds` | 1800；单次调用上限 |
| `--max-calls` | 32，最少 3；每阶段/节点的总调用上限，包含续写、修复、失败和预留调用 |

排队、暂停和维护等待不自动停表。派发时为随后审阅、一次裁决及 P6 签收预留时间和调用；余额不足即停止，预留计算见[预算协议](tools/README.md#退修与裁决)。

`resume/continue --add-seconds N` 将算子及所有已开始阶段/节点的截止时间分别延至 `max(当前时间, 原截止) + N`；未开始的阶段仍用原 `stage_seconds`。`--add-calls N` 为每阶段/节点（含后来建立的节点）增加 N 次调用上限。追加保存在 `budget_extensions`，不改输入锁、初始配置和已用计数，也不重新授予裁决/签收。

相同接续命令只创建一个 successor、追加一次；每次新的 `resume --add-*` 是新的追加授权。不确定上次是否生效时先读 `status --full`。普通恢复、重开、重复 extract、hold 释放和自动重试均不增加预算。

### 重复命令与退出码

重复 extract 沿用登记时的固定源码、规则和参数，不重新读取工作树。结果分别给出 `result`、保存的 `run_status`、`published`、`next_step` 和 `frozen` 身份。修改请求参数会报告原目的地占用；需要新来源或参数时换输出目录重新 extract，采用工具修复则用 continue。

| 命令 | 退出 0 的含义 |
| --- | --- |
| 前台 extract / resume | 已终审并发布 |
| extract --queue / continue | 已排队；重复 extract 也可返回已发布结果 |
| extract --background | 启动请求成功、已有任务运行中、被 hold 保留或已发布 |
| pause / cancel | 控制请求已提交，仍须核实在途调用与终止证据 |
| status / inspect | 只读查询完成 |

前台复用运行中任务返回 `1 / already_running`。旧任务暂停、取消、阻塞，或后台命令没有原排队任务的启动授权时，也返回 `1` 和下一步；不会创建新任务绕过停止。其他执行停止、预算耗尽及发布冲突返回 `1`，入口错误返回 `2`。各查询/审计命令的成功不等于提取完成。

## 角色与会话

Supervisor 的长期职责是主动推进提取：在既定任务语义内，自行决定预算追加、故障恢复、必要修复和合理取舍，处理并归档 issue。只有确实存在明显的漏洞且该漏洞的解决存在二分争议（必须由用户参与）才能继续的情况，才进行暂停。其余情况自行把握，期间的所有决策、改动和验证结果都须忠实记录，供后续审计。

需要工程修改时，Supervisor 确定方案与范围，交 subagent 实现并运行必要测试，再依据实际差异和验证结果验收；保持主线程集中于判断与监督。

Runner 唯一写状态、派发、检查、合入和发布。worker 修改许可产物，reviewer 独立对照当前文档和固定源码。P1/P2 的 worker 续用本角色；reviewer 按阶段使用独立会话。P3、P4、每个 P5 节点使用新上下文；P6 由全新 reviewer 先审，确需修改才启动 worker。

worker/reviewer 的共用提示始终要求对齐本次冻结的 `schema_docs`，不得自创字段、记法或用正文覆盖结构语义；确有不适配、缺失或冲突时，先指出具体条款和反例，由 Supervisor 按[问题分流](../EXTRACT_SCHEMA.md#框架问题分流)处理。Supervisor 先判断能否按既有规则改正产物，不因 Agent 写不顺就直接修改语言；reviewer 的批准须同时满足源码忠实性与规范一致性。

Reviewer 按[当前文档验收方法](REVIEW.md)通读本阶段已完成的全部内容，包含复制和未改段落，先形成自己的理解再核对固定源码。初始提示不提供前序批准、交接或旧稿；批准必须有独立理解、分层、全文一致和内容必要性四项文档结论及本阶段文件锚点。只读上游的规范表达错误可用 `schema_conformance` 及当前产物锚点请求正式重开，原预算和一次重开限制不变。

退修按阶段、P5 按节点累计，默认第五次退修后由原 reviewer 一次裁决；P6 裁决亲自改稿后须全新 reviewer 一次签收。不能完成则明确阻塞。依赖就绪的分支立即派发，子节点从已审直接父 YAML 复制；阻塞只影响后代。会话、回执与重开规则见[工具协议](tools/README.md)。

## 状态阅读

`status RUN_UUID` 展示阶段、角色、checkpoint、有效完成数和停止原因；`--full` 展示权威状态。`nodes_by_status` 区分 ready/in_flight/revision/approved/waiting_parent/blocked，`node_details` 给出父节点、调用和原因，P5 以这些节点项为准。

状态包括 queued/running/paused/cancelled、终审后的 ready，以及 blocked/budget_exhausted/tool_error/cli_error/invalid_receipt/termination_unconfirmed。先读 `reason` 和节点原因，再按恢复条件处理。仍有 `active_attempts`、`unconfirmed_attempts` 或 `unsettled_attempts` 时，不能把停止状态当作所有调用已结算。

`inspect` 查看多个任务的轻量投影；`monitor scan` 另保存短观察和最多 64 条变化。可用 `--run UUID...` 限定范围。投影陈旧或证据不足显示 stale/unknown，日志沉默不等于死锁；普通扫描不会自动恢复任务。已有 successor 的前任显示接续身份，避免重复处理。

## 恢复与交付

| 场景 | 操作 |
| --- | --- |
| 在调用边界暂停 | `pause RUN_UUID`；已启动调用可完成并保存结果 |
| 请求终止在途调用 | `cancel RUN_UUID`；检查 result/termination 的确认记录 |
| 恢复排队、暂停或已处理的普通异常 | `resume RUN_UUID`，使用原冻结执行器 |
| 时间或调用余额不足 | `resume RUN_UUID --add-seconds 7200` 或 `--add-calls N` |
| 采用当前工具/规则修复 | `continue OLD_UUID --checkpoint DIGEST`，随后 `resume NEW_UUID` |
| 原输出目录存在冲突 | `export RUN_UUID --output /path/to/new/delivery` |

恢复先确认已登记调用及其子进程终止，再采用合法草稿或上一个安全 checkpoint。发射后进程身份缺失时保留 fence，不能凭文件缺失判断退出。结算按 attempt 高水位去重；检查中断的开放时段保守计时。外部确认和已用单次决定不能靠 resume 或追加预算解封。

continue 要求原 run 静止、写锁空闲、调用和计时全部结算。它固定相同源码与当前规则，从 P1 新会话独立重审；旧批准不继承，原用量、预算追加、删除节点账目与单次决定都保留。可同时使用 `--add-seconds/--add-calls`。接续关系唯一，构建中断后用相同命令补全；接续建立后原 run 不再派发或发布。冻结执行器不支持某项能力时，入口会给出明确提示，不由新代码直接接管。

补充或更换源码材料需换输出目录重新 extract。不要改快照来补材料，也不要删除运行账目来获得新预算。

`pause --all` 返回 hold ID，关闭新派发与发布的闸门；`resume --all --hold ID` 只释放该次影响。手动暂停、取消、原阻塞和未知终止保留，仍在途的项可在停止后重复同一释放命令核对。单 run resume/export 也受 hold 约束。缺少 hold 能力的冻结执行器只支持原有 pause，停止前不能保证发布已被拦住。

发布将完整已审目录及回执原子重命名到目标。目标须不存在，或仅含一个零字节普通 `.gitkeep`；其他目录（包括空目录）、文件和链接均保留并报告冲突。同一完整结果可幂等重放；export 另行交付，不更改原输出位置。

`import-legacy` 是 v1 产物的兼容入口：原任务须已终止、锁空闲、调用终止证据完整；描述文件须绑定原快照源码。导入稿保存在 `imported/`，从 P1 独立重审，原用量单独保留，本次使用新建任务的预算。具体参数用 `import-legacy --help` 查看：

```sh
python3 schema_extract/extract.py import-legacy --from-task /old/runtime/tasks/TASK_KEY --checkpoint CHECKPOINT_ID --source /path/to/source.json --output /path/to/new/delivery --queue
```

## 单文件阅读视图

需要把某个节点交给只拿到一份 YAML 的读者时，生成包含完整共享 Contract 的独立视图：

```sh
python3 schema_extract/extract.py standalone path/to/task/branch.yaml --output path/to/views/branch
python3 schema_template/check_yaml.py --standalone --skip-hardware path/to/views/branch/branch.yaml
```

生成器先检查创作任务，保留节点自身的元数据、Design 和 Realization，展开当前 root 的 Contract；不继承 parent 的 Design。输出目录在原任务之外，保存原文件名的 YAML 和 `.source.json` 来源摘要。相同输入可幂等复用，冲突内容保留；原节点和运行状态不变。

视图保留原节点的选择范围，不作为整个算子所有路径的替代，也不产生新的 P6 批准。其语义仍接受 [Reviewer 方法](REVIEW.md)约束；原文件若含隐含依赖或规范问题，生成视图不会消除它。需要改动时修订创作源并重新审阅，随后生成新的视图目录。

## 同线程持久提醒


```sh
python3 schema_extract/extract.py monitor start --thread THREAD_UUID
python3 schema_extract/extract.py monitor status
python3 schema_extract/extract.py monitor stop
```

start 绑定 Supervisor 的真实线程，可用 `--run UUID...` 限定范围。首次完成检查 #0，随后间隔为 300/300/300 秒，再每 1800 秒检查；从成功扫描提交时计算下一次时间，逾期只处理一轮。后台时钟只计时和投递 `codex queue --resume` 消息。

收到提醒后执行消息中的 `monitor ack ID --thread THREAD_UUID`；仅 accepted=true 才执行一次 `monitor scan --wake ID --thread THREAD_UUID` 并结束该轮。accepted=false 时结束该提醒。ack 后中断可用同一 scan 命令续完；普通 scan 不消耗周期。不要用对话内 sleep 或另建闹钟维持此周期。

发送失败或不确定送达沿用同 ID，最多 3 次、至少间隔 120 秒；成功入队后等待 ack。耗尽重送次数，或 1800 秒仍无 ack/完成时显示 needs_attention。队列确认不代表线程实际收到，状态文件不代表时钟进程仍存活。

进程退出后显式 start 恢复进度；换线程或范围用 stop/start。stop 使旧提醒失效并停止时钟，不控制提取任务。状态显示真实进程身份、锁、心跳与提醒进度；可选 `--probe-delay 1..300` 验证线程唤醒，不改变正式周期。

自动恢复默认关闭。`monitor start --auto-restart-limit N` 只授权当次选定的 run；不指定范围时仅覆盖当时已登记的任务。当前只对确认终止、明确返回 EX_TEMPFAIL 75 且仍有预算的普通调用生效；按 lineage 累计上限，重启时钟不退款。其他错误按恢复条件显式处理。

## 审计与封存

```sh
python3 schema_extract/extract.py audit RUN_UUID
python3 schema_extract/extract.py audit RUN_UUID --attempt ATTEMPT_UUID --verify --archive /path/to/evidence.tar.gz
```

audit 默认读已记账摘要，按阶段、角色、节点和来源展示用量；`--attempt` 与 `--session` 互斥，`--json` 给出详细定位。`--verify` 深验选定调用、checkpoint、来源和引文，不授予批准或推进阶段。

usage 保留提供者原值。cached input 已包含于 input，不能再相加；失败或截断为 partial，缺字段为 unknown，不补作零。新会话与 resume 的计数范围仍未确认，`additive_tokens` 保持未知，不据此估算总 token 或账单。CLI/检查累计时间也不等于并发任务的墙钟；暂停等待和纯模型时间未单独可靠采样。

archive 同时深验，生成含 `manifest.json` 和按摘要去重的 `objects/` 的 tar.gz。目标须在 runtime 外、父目录已存在，路径不含 `..` 或链接；同一有效清单可复用，冲突目标保留。对象缺失或选中范围仍活动/未结算时不写包。封存包用于审计，不能直接恢复执行，也不删除运行记录。

audit 的 `0` 表示读取或可得证据检查完成，仍须看 unavailable；证据不完整、不符或封存失败为 `1`，入口错误为 `2`。

## 验证边界

P3/P5/P6 完成或批准时，Runner 自动用冻结检查器执行 `--skip-hardware`；退出 0 才放行，1 是产物诊断，2 是工具/环境问题。P1/P2/P4 使用阶段门禁。检查不代替语义审阅；硬件保留 starter 占位，GPU 验收与实测不在提取范围内。

核对是否实际检查时，查看对应调用的 `attempts/<attempt_id>/check.json` 及 `validated.json` 中的命令、退出码和产物绑定；P5 合入另有 `merge_check.json`。Agent 回执的 `checks` 只记录它亲自执行的命令，`checks: []` 不能据此判断 Runner 漏跑；最终还应核对 P6 所查文件与发布回执绑定的文件一致。

仓库保留回执、预算、依赖和 CLI 恢复回归，执行方法见[离线验证](tools/README.md#离线验证)。2.2.5 当时通过 42 项提取测试、148 项模板测试，以及 10 个真实 Reviewer 语义样例；对原 GDN 的隔离 P6 复核要求重开 P3，见[Reviewer 验证记录](review_cases/RESULTS.md)。这些结果验证审阅机制，历史产物的具体表达问题继续单独跟踪。

2.2.6 按[长期分类与处理方案](issues/TRIAGE.md)简化 P4 索引布局检查，统一参数化成员路径的规范、校验与有限展开，并补齐 catalog 校注文件的输入收集；最终 47 项提取回归、两份历史失败稿只读重放，以及 152 项模板回归通过。条件流程继续用既有 dispatch 表达不同实际执行关系，四类历史产物仍待独立修订；本轮未重复模型评估或重新提取六项算子。

2.2.7 为共享 Contract 输入输出增加 [nullable 与 None](../schema_docs/1.contract.algorithm_requirement.md#nullable-与-none) 的正式写法，区分对象缺席、参数省略和非 `None` 属性；[框架修复已验收](issues/archive/schema_docs/optional-tensor-contract.json)。157 项模板回归、49 项提取回归通过，包含 GDN/FMHA/Sparse 的最小 I/O 片段；这些结果只验证规范字段与门禁，不代表算子复审通过，也未运行 GPU 或外部模型评估。已有冻结 2.2.6 的调用与历史产物保持原状，使用新规则须正式接续。

2.2.8 澄清静态 plan 与[一次算子请求实际执行](../schema_docs/0.common.md#plan-与计算执行实例)的绑定，以及[条件等待](../schema_docs/5.design.tile_pipeline.md#条件等待示例)的显式跨 role 前驱；沿用现有字段、重复和索引规则，不增加执行系统。规范绑定问题见 [capture-invocation-boundary](issues/archive/schema_docs/capture-invocation-boundary.json)，实际产物表达仍由 [conditional-flow-topology](issues/extract_workflow/conditional-flow-topology.json) 跟踪。检查限于文档链接、示例和维护记录；隔离 MoE 示例通过既有检查器及有限事件图检查，不代表真实 root 已修复。现有 GDN/FMHA 在冻结 2.2.7 下的合法任务无需因本次澄清统一重跑；MoE 应通过正式 `continue` 采用新冻结并独立重审，原批准不继承。

2.2.9 将[源实现缺陷与设计分流](REVIEW.md#源实现缺陷与设计分流)统一到共用 Reviewer 方法和真实调用提示：有来源依据的完整设计可携带源问题继续，必须猜测或改变设计决定时仍阻塞，残稿仍须补齐并独立审阅。[流程改进已由 Supervisor 验收](issues/archive/extract_workflow/source-defect-abstraction-boundary.json)；未修的 Sparse 源布局问题由 [sparse-tc-control-layout-overlap](issues/kernel_wiki/sparse-tc-control-layout-overlap.json) 独立跟踪，实际 Design/P6 尚未完成。采用新规则须正式接续，不改旧回执或继承批准。

2.2.10 修复[当前产物问题的证据门禁](issues/archive/extract_workflow/artifact-only-issue-evidence.json)：仅 `revise/blocked` 的 `submission_format`、`schema_conformance` 可使用与问题 scope 中完整相对文件名绑定的有效 `artifact_evidence`；源码类问题、批准和上游重开保留原要求。最终 62 项提取回归通过，两份真实 MoE 退修提案只读重放及错误路径、源码类问题的负例通过，原状态与输入字节未改变。Supervisor 已验收这项协议修复；MoE 的失败接续设计和 P4 计划仍须独立审阅，采用新门禁须正式接续。

六类固定实现曾在冻结 2.2.2 下完成 77 节点发布与证据审计；2.2.4 另完成分组 GEMM 的三节点真实验证，见[试用记录](TRIAL_REPORT.md)。样本结果不代表其他实现的提取成功率，也不代替 GPU 或外部比较器验收。

框架候选与稳定 ID 决定通过[维护入口](issues/README.md)查看；算子自身缺项留在任务回执。
