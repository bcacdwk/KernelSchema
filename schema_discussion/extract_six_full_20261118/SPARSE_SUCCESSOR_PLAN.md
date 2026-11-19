# Sparse：用现有接口另建单项 run

> 历史方案，已撤回。2026-11-19 用户要求关闭新调用并研究旧进度续接；新调用已确认取消。当前结论见 [主代理复核](recovery_reassessment/RECOMMENDATION.md)，原论证保留以便追溯，不能再据此启动或恢复 successor。

2026-11-19。**推荐只为 Sparse 新建使用当前修复工具的 run，以旧稿作为明示只读参考，从 P1 完成新的真实审核；无需新增产品能力，也无需用户再次决定。** 原 run 保留真实阻塞记录，其他五项继续原流程。用户已有完整提取与修工具授权；只有额外要求“必须保留原 run UUID/批准”才会超出现有能力，当前不推荐执行器迁移方案。

本次只读核对代码和必要记录，仅新增本文；未准备输入包、启动模型、改状态/快照/产物、运行测试或操作 monitor。

## 当前事实与接口边界

- 旧 run：`8f0844f9-f279-4781-a19d-0216b7fc4ef5`；`P6/invalid_receipt`、未发布，三类在途均为空；checkpoint：`66ad31f314d7ca7e894527d9641bbf3daad59f4ba35278e71e7841dd5e3901da`。scope 工具故障及旧 run 不能公开续接的依据见 [P6_APPROVE_SCOPE.md](P6_APPROVE_SCOPE.md)。不得改旧 state、冻结执行器或批准。
- 旧 `source.files` 共 31 个文件、983,940 B；本次逐文件 SHA 核对，当前仓库对应文件仍与旧快照一致。建议仍从旧快照复制，避免后续工作区变化影响选材。
- 公共来源接口只有 `entrypoint/host/io/input/reference/acceptance/scope`；`scope` 可绑定 Markdown。每个描述文件的 `directory` 整棵普通文件树及绑定材料都会冻结。[inputs.py](../../schema_extract/tools/inputs.py:142)、[inputs.py](../../schema_extract/tools/inputs.py:245)
- `annotated_entrypoint` 仅在 catalog 选择分支登记额外阅读辅助；自定义 descriptor 不消费这个键。没有 `--seed`、`--from-run`、外部 `materials` 或批准导入选项。运行内部 `historical_definitions`、`materials/candidates` 仅来自本 run 的受控记录，不能由 Supervisor 冒充注入。[inputs.py](../../schema_extract/tools/inputs.py:208)、[prompts.py](../../schema_extract/tools/prompts.py:94)
- 新 run 总是 starter → P1 worker，阶段批准和节点批准全部重新建立。[store.py](../../schema_extract/tools/store.py:334) 每阶段提示均包含 `inputs.scope` 与固定 `source_root`，整个固定源码目录是已登记只读材料；所以 `scope` 中的导航可合法指向同一包内的历史稿。[prompts.py](../../schema_extract/tools/prompts.py:218)

## 最小参考材料包

在讨论目录下创建独立资料目录，不往原 `kernel_wiki`、旧 runtime 或 snapshot 加文件；不要把主仓库 `directory: "."` 用作打包根，否则会冻结大量无关材料。建议布局：

```text
schema_discussion/extract_six_full_20261118/sparse_successor/
├── source.json
└── inputs/
    ├── kernel_wiki/...       # 原 source manifest 的 31 个文件，路径/字节不变
    └── prior_work/
        ├── README.md        # 角色声明、资料导航、已知缺陷和使用边界
        ├── origin.json      # 旧 run/snapshot/source/checkpoint 身份与逐文件 SHA
        ├── REGISTER_CLAIM_TRIAGE.md
        └── artifacts/       # 原 checkpoint 五份文件，原字节不改
            ├── root.yaml
            ├── split_kv.yaml
            ├── tcgen05.yaml
            ├── tcgen05_packed.yaml
            └── dispatch.md
```

准备时核对旧 `inputs.lock.json` 的 source manifest 和 checkpoint 的 `checkpoint.json`/五文件 SHA，再普通复制，不用 symlink。`origin.json` 记录：旧 run UUID、原 snapshot/source 身份、原 checkpoint、原 source 文件 SHA、原五稿 SHA，以及这些文件在新包中的路径；不复制 state、session、审批字段或批准回执作为当前运行记录。这是被新 snapshot 哈希保护的来源说明，不是工具提供的批准迁移协议。

`prior_work/README.md` 应明确：

1. 这是同一任务的历史候选和审计导航，不新增数学要求、shape、验收阈值、实现保证或强制选择计划。任务事实仍取原 `kernel_wiki` 材料。
2. 五稿仅供当前阶段阅读、核对和转写；其旧 `Source`、Version、选择、`from`、旧判断没有当前批准效力。源实现/比较语义证据仍锚定实际实现与原角色材料，不能用旧 YAML 或此说明替代源码证据。
3. 保留 `REGISTER_CLAIM_TRIAGE.md` 的限制：208 是三个主数组的每线程 32 位槽小计，不能写成编译后总寄存器数或无 spilling 保证。P3 独立核实并收窄两处表述及相关 Reason；其他内容仍按正常阶段全文核查。

下面是**生成 descriptor 的完整结构**，供准备包时执行；只从旧锁提取已登记角色，避免手写路径/符号出错。实际复制与来源摘要应先按上述清单完成。

```python
from copy import deepcopy
import json
from pathlib import Path

repo = Path('/home/shaohanyong.shy/KernelSchema')
old_run = repo / '.schema_extract/v2/runs/8f0844f9-f279-4781-a19d-0216b7fc4ef5'
bundle = repo / 'schema_discussion/extract_six_full_20261118/sparse_successor'
lock = json.loads((old_run / 'inputs.lock.json').read_text())
bindings = lock['source']['bindings']
spec = {
    'format': 'schema_extract.source.v2',
    'id': bindings['id'],
    'repository': str(bundle / 'inputs'),
    'directory': '.',
    **{key: deepcopy(bindings[key]) for key in
       ('entrypoint', 'host', 'io', 'input', 'reference', 'acceptance', 'scope')},
}
spec['scope'].append({'path': 'prior_work/README.md', 'symbol': ''})
(bundle / 'source.json').write_text(json.dumps(spec, ensure_ascii=False, indent=2) + '\n')
```

原角色路径已经相对 repository 根，故这里 `directory: "."` 正确；原两个 shape scope 保留。不要给此 descriptor 填旧 `revision`：新增材料不在旧 Git commit 中。新包会生成真实的 `private_snapshot_commit` 和新 repository/source 身份，原实现字节及任务绑定不变；**不能声称新 source_id/ref 与旧值相同**。Worker 按当前 invocation 的真实新引用写 `META_DATA.Source`/reference，旧稿中的历史引用保持只读。[snapshots.py](../../schema_extract/tools/snapshots.py:81)

## 启动、阶段复用与正式交付

在主仓库根目录执行，使用同一个 state-root，首次新输出必须不存在且未登记：

```sh
python3 schema_extract/extract_yaml.py extract \
  --source schema_discussion/extract_six_full_20261118/sparse_successor/source.json \
  --output schema_discussion/extract_six_full_20261118/sparse_successor_delivery \
  --backend claude \
  --agent-binary /usr/local/lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe \
  --model 'qwen3.8-max[1m]' --worker-effort max --reviewer-effort max \
  --revision-limit 5 --max-calls 32 --background
```

保存返回的新 UUID 后核对其 `status --full`/`inputs.lock.json`：是新 run、P1、当前修复 executor 已冻结、31 个原 source SHA 保持一致，参考资料已入新 source manifest。当前执行入口冻结当前主仓库的 rules/executor，不会因为 `repository` 指向资料包而采用旧工具。[runner.py](../../schema_extract/tools/runner.py:75)、[snapshots.py](../../schema_extract/tools/snapshots.py:124)

| 阶段 | 可参考的旧工作 | 必须重新形成的当前工作 |
| --- | --- | --- |
| P1 | root 的任务、IO、参考/验收表达与来源导航 | 从 starter 写本阶段许可字段、真实来源核查、当前独立批准；不整份灌入 root Design |
| P2 | dispatch 的代表候选、取舍与源码线索 | 独立选择当前 root/非 root、plan 和目录；四项旧选择不是硬编码批准 |
| P3 | root 三个 Design 区与 Realization 的完整草稿 | 在当前 Contract/代表上写作、纠正寄存器保证、独立全文审核 |
| P4 | 旧写作顺序、共享说明 | 当前计划/依赖/目录的批准；不能导入旧 P4 receipt |
| P5 | 三份非 root 草稿、设计和绑定线索 | 各当前节点建立身份并写作；`from` 只能采用本 run 当前已审整区，逐节点独立审核 |
| P6 | 历史缺陷清单可作核查导航 | 当前完整五文件/实际图/全部有效批准/新 reviewer 终审及正常发布 |

新稿 Version 从当前正常写作规则建立，不继承旧版本号、调用次数或 session。旧 run 的成本与失败记录保留；新 run 有自己的调用账目。收益是减少重新探索与转录，不能保证免除任何阶段或模型调用。当前工具没有自动跨 run 进度/批准复用；以上资料谱系足以完成此次续做，不需要为了省审核新造迁移系统。

旧标准目标虽然不存在，仍由旧 `starts` 注册占用；用相同参数会复用旧 run，改参数则拒绝。不得删除注册或改旧 output。新 run 先正常发布至上述新目录；达到 `ready` 且当前 P6/所有发布门禁通过后，使用公开 export：

```sh
python3 schema_extract/extract_yaml.py export "$SPARSE_SUCCESSOR_RUN" \
  --output schema_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16
```

`SPARSE_SUCCESSOR_RUN` 必须赋为实际返回的新 UUID。`export` 会重新核冻结身份、当前批准、完整 checkpoint、Template 和发布屏障，原子不覆盖落盘，并记新 run 的 `exports[].status=published` 与目标 `.extraction/receipt.json`；它不调用新 run 的 output 注册，也不转移旧注册。[monitor.py](../../schema_extract/tools/monitor.py:669)、[runner.py](../../schema_extract/tools/runner.py:1557)、[publication.py](../../schema_extract/tools/publication.py:41) 标准目录的交付归属以**新 run 的 export/receipt**为准；旧 run 仍处于原 `invalid_receipt` 阻塞、未发布。若目标被别的内容占据，export 记 conflict，不得手工搬文件绕过。

## 同线程更新既有 monitor

当前 clock 绑定线程 `01a1c900-a487-76a1-92ee-990cbff08276` 和明确的六 UUID；不会自动发现新 run。现有 start 明确禁止在线改 scope，需对**同一 `.schema_extract/v2` 单例**执行 stop/start。[wakeup.py](../../schema_extract/tools/wakeup.py:237)

先结清已接受而未完成的提醒；取得新 UUID 后记录当前 `monitor status`，再执行 `monitor stop`，确认 `stop_status=confirmed`（若 pending，按 `monitor status` 的真实进程/投递状态结清后再 start）。随后用同一线程、原五个非 Sparse UUID 加新 UUID 重启同一 clock：

```sh
python3 schema_extract/extract_yaml.py monitor status
python3 schema_extract/extract_yaml.py monitor stop
# 确认旧 clock/queue 已结清后：
python3 schema_extract/extract_yaml.py monitor start \
  --thread 01a1c900-a487-76a1-92ee-990cbff08276 \
  --run 06f3ed2c-daa0-46f0-a0dd-a7e1f4f4df31 \
        0d78eac5-1e45-4ac0-9725-ab885b5c52dc \
        68edadab-4371-402c-8acf-f73eb5ec9f92 \
        74bcac91-808a-41b2-aab4-85143935ab40 \
        fb8b3758-92d7-4093-aa18-c2e5028f0a43 \
        "$SPARSE_SUCCESSOR_RUN"
```

stop/start 使 generation 更新并退休旧 reminder，**保留 last_check_number、next_due_at、历史**，不重新执行 scan #0，不暂停/取消五个 Runner；仍使用同一个 singleton lock 和同一个 Codex thread。[wakeup.py](../../schema_extract/tools/wakeup.py:186)、[wakeup.py](../../schema_extract/tools/wakeup.py:640) 不新建另一套 timer/线程，不改 supervisor.json。若需要继续自动展示旧阻塞记录，也可把旧 UUID 保留为第七项；推荐六项目标列表替换 Sparse，并在讨论台账单独保留旧→新谱系，避免反复恢复已知无解旧 run。

新建成功后由主 Supervisor 更新现有 RUNS/PROGRESS/恢复记录中的 Sparse 当前执行 UUID及旧→新关系；新 run 正式交付时记录 standard target 来自 export。本文不把计划写成已执行恢复或已批准交付。
