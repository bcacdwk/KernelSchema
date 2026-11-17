# Schema Extract：Supervisor 手册

Schema Extract 将固定源码和任务材料整理为少量完整设计、中文 `dispatch.md` 和交付凭据。唯一入口是 [extract_yaml.py](extract_yaml.py)。六阶段职责见[提取总览](../EXTRACT_SCHEMA.md)，审阅方法见 [REVIEW](REVIEW.md)，执行协议见 [tools/README](tools/README.md)。

默认阶段后端为 Claude Code，模型固定 `qwen3.8-max[1m]`，worker/reviewer 均为 `max`。Codex 是显式可选后端，须同时指定模型和两种角色的 effort。Supervisor 的 Codex queue 提醒另行配置。

## 开始一次试用

运行需要 Linux、Python 3.11+、PyYAML、jsonschema、Git、`ps` 和已配置认证的 CLI。调用以当前用户身份直接使用真实文件系统；具体角色权限见[执行边界](EXECUTOR_ALIGNMENT.md)。

从仓库根目录运行：

```sh
python3 schema_extract/extract_yaml.py extract --source /path/to/source.json --output /path/to/delivery --background
python3 schema_extract/extract_yaml.py status RUN_UUID
```

单任务默认前台执行；`--background` 登记并启动冻结 Runner，`--queue` 只冻结和排队，两者互斥。`ready` 表示当前 P6 已通过，`published: true` 表示已发布到登记的输出目录。交付包含 `root.yaml`、所选非 root YAML、中文目录及 `.extraction/receipt.json`。

提取只表达固定实现及其设计边界，不运行 GPU 或修复 kernel。适用检查跳过硬件占位；结构检查和独立语义审阅分别留下证据。

## 输入与输出

可以使用[算子索引](../kernel_wiki/CATALOG.md)中的 catalog ID、本地 catalog 算子目录，或明确的来源描述：

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

角色路径相对 `directory`，`repository` 相对描述文件，省略为 `.`；`host` 省略时采用 entrypoint。来源目录中的 `schema-inputs.json` 也可用。未指定 `revision` 时冻结本地实际字节；指定时读取解析后的 Git commit。`--source 'git:<ref>:<directory>' --bindings FILE` 使用 `--repo` 指定的仓库。

一个 run 固定以下身份，恢复与维护均沿用它们：

| 身份 | 固定内容 |
| --- | --- |
| source/config | 源码字节、入口与角色绑定、任务、输出和调用预算 |
| rules | schema_docs、starter、Template、完整 prompt 树及其引用 |
| executor | 实际 Python 执行器、能力和后端配置 |

快照含逐文件摘要与整体身份；脏本地源码以快照内独立 Git 引用定位，原仓库不变。来源、任务、规则或执行器改变时另建 run。annotated 源码只作为已登记的阅读辅助，事实仍回查固定实现。

默认状态目录为 `.schema_extract/v2`；`--state-root` 可以独立指定。`snapshots/` 保存输入，`runs/RUN_UUID/` 保存状态、attempt、原稿、规范化记录和 checkpoint，`supervisor.json` 保存控制请求。正式输出位置见 [Schema Wiki](../schema_wiki/README.md)。

## 命令与参数

```sh
python3 schema_extract/extract_yaml.py extract --id bf16_gemm --output /path/to/delivery --queue
python3 schema_extract/extract_yaml.py extract --ids bf16_gemm mxfp8_gemm --output-root /path/to/deliveries --background
python3 schema_extract/extract_yaml.py inspect --run RUN_UUID --json
```

`--ids` 需要 `--output-root` 和 `--background` 或 `--queue`。`--repo` 默认当前目录；从仓库外启动时使用入口绝对路径，并明确 repo/state-root。各命令的完整选项见 `--help`。

| 参数 | 新建 run 的默认值 |
| --- | --- |
| `--backend`、`--agent-binary` | `claude`；可用 CLI 绝对路径 |
| `--model` | `qwen3.8-max[1m]` |
| `--worker-effort`、`--reviewer-effort` | `max` |
| `--revision-limit` | 每阶段或 P5 节点 5 次实质退修 |
| `--operator-seconds` | 从创建开始 7200 秒 |
| `--stage-seconds` | 阶段或节点从首次调用开始 3600 秒 |
| `--call-seconds` | 单次调用 1800 秒 |
| `--max-calls` | 每阶段或节点 32 次，最少 4 次 |

显式 Codex 示例：

```sh
python3 schema_extract/extract_yaml.py extract --source /path/to/source.json --output /path/to/delivery --backend codex --model YOUR_MODEL --worker-effort high --reviewer-effort high
```

预算同时覆盖续写、退修、失败、检查处理和一次裁决/签收的预留。排队、暂停和维护等待仍计墙钟。恢复、返回和重复 issue 不退还已用次数；需要追加时由 Supervisor 明确指定。

## 角色与会话

Supervisor 确定来源、输出、预算和具体维护请求；Runner 唯一提交运行状态、checkpoint、合入和发布。worker 通过 CLI 角色工具编辑本次目录中的授权产物；reviewer 使用只读角色。正式采纳由 Runner 核验授权文件集合、字段权限、完整文件类型及固定来源。CLI 权限与差异核验不保证宿主文件系统不可写。

每 scope 使用固定的真实空 cwd 和私有 CLI home；候选、材料与原稿始终保存在各自 attempt 目录。每阶段、角色和操作建立独立新会话，P5 另按节点分开。P1 与 P2 也独立。只有同一 scope 内的续写或退修可用已登记的精确 UUID resume；新调用始终获得完整动态材料。P6 先启动新 reviewer，存在明确限域修改时才启动 worker。所有阶段调用禁止内部委派。

P2 提交 `{id, instance}` 的有限选择，P4 提交 `{id, wait_for}` 的写作安排，P5 按实际 raw `from` 采用已审来源。单 root 仍完成 P4、明确跳过 P5，并经过 P6。

## 状态阅读

`status` 展示阶段、角色、checkpoint、节点和停止原因；`--full` 读取权威状态。`inspect` 是轻量只读观察，`monitor scan` 另保存短观察，默认不恢复任务。日志沉默不能证明死锁。

先看 `reason`、节点原因及 `active_attempts / unconfirmed_attempts / unsettled_attempts`。停止状态与调用结清分别核对；终止未确认时保持 fence，不能启动同一范围的新调用。

## 恢复与交付

下表命令接在 `python3 schema_extract/extract_yaml.py` 后：

| 操作 | 命令与边界 |
| --- | --- |
| 调用边界暂停 | `pause RUN_UUID`，已启动调用可结算 |
| 取消在途调用 | `cancel RUN_UUID`，核对实际终止证明 |
| 恢复当前冻结运行 | `resume RUN_UUID` |
| 恢复单个普通失败范围 | `resume RUN_UUID --scope P5:NODE_ID` |
| 明确增加预算 | `resume RUN_UUID --add-seconds N --add-calls N` |
| 全局暂停与释放 | `pause --all` 返回 hold；`resume --all --hold ID` 只释放该 hold |
| 交付到另一新目录 | `export RUN_UUID --output /path/to/new/delivery` |

恢复先核进程、真实 exit、原生结果、schema、nonce 和精确会话。真正中断的会话在确认终止后退休，保留 checkpoint 和短交接；确证未启动时可以保留前次有效会话。范围恢复绑定实际失败 attempt、当前实例和实际来源；健康兄弟继续。`--scope` 不接受预算追加。

当前未交付 run 的有限维护：

```sh
python3 schema_extract/extract_yaml.py maintain RUN_UUID --issue-id copy-definition --tasks /path/to/tasks.json
```

```json
[{"stage":"P3","node":"root","action":"patch","fields":[{"path":"root.yaml","fields":["DESIGN.Math_reshape"]}],"reason":"修正已定位的计算定义"}]
```

任务仅针对当前 P1–P5，action 为 `review` 或 `patch`；patch 须给明确文件/字段。请求先挡新派发和发布，由同一 Runner 排空后执行。完整稿可以先审，未完成稿继续普通 worker；维护结束仍需正常阶段和新的 P6。已交付运行不能再通过维护改稿。

发布核当前批准、来源、完整目录、最终 checkpoint 和控制屏障，然后原子交付，保留已审字节。目标须不存在或只含零字节普通 `.gitkeep`。冲突保留双方内容；相同完整结果可幂等重放。另行 export 以 `exports[].status=published` 和目标回执为准。

## 单文件阅读视图

```sh
python3 schema_extract/extract_yaml.py standalone /path/to/task/node.yaml --output /path/to/views/node
python3 schema_template/check_yaml.py /path/to/views/node/node.yaml --standalone --read-only --skip-hardware
```

工具先用当前 Template 完整检查 raw 目录，再展开该节点的有效 Design 与 root Contract，保留自身元数据、when、Version 和 Realization，并严格检查生成文件。视图写在 raw 目录之外，`.source.json` 同时记录实际内容来源和完整目录检查身份。相同结果可复用，冲突拒绝；视图不产生新的阶段批准。

## 审计与用量

```sh
python3 schema_extract/extract_yaml.py audit RUN_UUID
python3 schema_extract/extract_yaml.py audit RUN_UUID --attempt ATTEMPT_UUID --verify --archive /path/to/evidence.tar.gz
```

默认读取已登记账目；`--verify` 复核选定调用、原始事件、原稿、规范化后的 checkpoint、来源和证据。archive 归档具体证据文件，不收整个私有 CLI home。目标须在 runtime 外，父目录已存在；冲突或未结清范围不封存。

requested model/effort 是调用要求，reported 只取原生报告，缺项保持未知。Claude 的本次调用增量、主循环 usage、累计 session modelUsage/费用分别展示；累计值不能相加成调用量，缓存读/写分别保留。Codex 的原生计数及可得的精确会话观测单列。各项按 attempt 去重，失败和恢复不重置账目。CLI 费用报告不是账单，未知项不填零。详见[用量协议](tools/README.md#usage时间与证据封存)。

## Reviewer 能力回归

```sh
python3 schema_extract/extract_yaml.py review-eval --suite schema_extract/tools/tests/fixtures/review_eval/suite.json --output /path/to/new/evaluation --call-seconds 360
```

`review-eval` 对明确提供的已知正确/错误文档各启动独立只读 reviewer，检查四项判断及缺陷定位。它会真实调用模型，用于 Reviewer 能力回归，不参与 run 批准；suite 和新输出目录必须显式指定。

## 同线程持久提醒

只有需要 Supervisor 提醒并明确授权投递时使用：

```sh
python3 schema_extract/extract_yaml.py monitor start --thread THREAD_UUID --run RUN_UUID
python3 schema_extract/extract_yaml.py monitor status
python3 schema_extract/extract_yaml.py monitor stop
```

提醒使用独立的 `codex queue --resume`。收到后按消息执行绑定线程的 `monitor ack`，accepted 且 scan_required 时执行一次 `monitor scan --wake ID --thread THREAD_UUID`；以 committed/reused 为本轮完成依据。stop 只停提醒，不取消已启动提取。

自动恢复默认关闭；`--auto-restart-limit N` 只授权所选 run 内已确认终止、EX_TEMPFAIL 75、有剩余预算的普通失败，持久累计次数。普通 pause/cancel/hold 保持优先，提醒补送有固定上限。

框架候选由 `candidates RUN_UUID` 汇总，reviewer 确认项按稳定 ID 交 [issues](issues/README.md) 处理。实际工具与模型验证范围见[执行器对齐说明](EXECUTOR_ALIGNMENT.md)；历史记录从 [archive](archive/README.md) 查阅。
