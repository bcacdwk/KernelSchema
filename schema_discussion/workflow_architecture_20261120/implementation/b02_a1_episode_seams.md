# B02 / A1.cpu：指定 episode、生命周期与预算窄接点

日期：2026-11-20。结论：**A1 的实际原生接缝与 CPU spike 已完成，可供 A3/A4/A5 消费**。18 项 synthetic CPU 测试通过；原 upstream 与 fork 默认 off 的归一化 trace 相同。本记录不宣告完整 V08/V11/V12、Schema switch、AKA fixed、真实 Task/profile 或跨进程恢复资格通过。

## 1. 工作边界与实际文件

唯一源码树：`/home/shaohanyong.shy/atrex-kernel-agent`；分支 `feat/schema-workflows-v1`，基线及当前 HEAD `bf5557ac04f6ff9988901d511cf25e1cbe61a1bc`。开工树干净；未创建/切换分支、提交、更改 Git 配置或远端，未安装 KS。没有访问、修改或运行 AKA_experiment，也没有迁入其补丁。

已读当前 P/W/C/V 及适用 AGENTS。AKA 路径上无适用 AGENTS；本报告遵守 KS 根 AGENTS，未派生代理。

| 实际 AKA 文件 | 改动 |
| --- | --- |
| `src/aka/workflow/schema/bridge.py` | 小型 `run_episode`、`EpisodeLifecycle`、`EpisodeResult`；只调用现有 engine，无 planner、父账本或第二调度器 |
| `src/aka/workflow/schema/__init__.py` | 最小可选 namespace |
| `src/aka/workflow/light/long_horizon/campaign.py` | 原 `run` 的指定 e、prepare/recover/episode 准入、已结算通知、首 draft 前准备接点；恢复 preparing 时保留其阶段，防止暂停后漏 seed 准备 |
| `src/aka/workflow/light/long_horizon/store.py` | 仅显式 e 使用的只读严格 checkpoint；原默认容错 reader 未改 |
| `tests/schema_cpu_fixture.py` | 真实 ManagedCampaign/Store/GitCandidateWorkspace/EpisodeWorkspace/LongSessionRunner 配 fake Agent/Evaluation/辅助 reviewer |
| `tests/test_schema_episode_bridge.py` | 指定 e、状态/限额、V0/seed/HEAD、原生恢复与通知集成测试 |
| `tests/schema_off_probe.py`、`tests/test_schema_off.py` | 默认原生入口 off probe、禁可选导入/KS 路径 I/O/网络及归一化 trace |
| `docs/schema-episode-seams.md` | 接点、消费责任、限制与逐文件来源说明 |

上述均为原生新代码或基于当前 upstream 的窄修改，非迁入。上游 `.gitignore` 忽略整个 `tests/`；仅给本次四个测试文件执行 `git add -N -f` 以便 diff 可见，没有暂存内容，`git diff --cached --stat` 为空；`.gitignore` 未改。最后 `git diff --check` 通过。

## 2. Driver 与生命周期结论

选定**原 loop 的指定 e + 窄准入/已结算接点**。没有复制 V0/setup/收尾。

实际调用链仍是 `LongHorizonCampaign.run → main_adapter.prepare_campaign → ManagedCampaign.setup_baseline / resume / ensure_framework_baseline`。外层 `orchestration.Campaign.run` 原本拥有 `_finish`；bridge 不调用它。未来外层 owner 在本次 workflow 真正结束时收尾一次，不应每 e 重启完整 `Campaign.run`，因为生产收尾可能再调用 reviewer。

新入口：

```python
from aka.workflow.schema.bridge import run_episode

result = run_episode(engine, e, lifecycle=owner, prepare_episode=prepare)
# result: episode, reason, native_root, attempt（没有完整终态时为 None）
```

`engine` 是原 `LongHorizonCampaign`，携带冻结的 native `max_version/token_budget`；原 `max_episodes` 模式也可消费。bridge 不覆盖这些配置，不创建自己的 allowance。native `token_budget=0` 的既有语义仍是未设 native token cap，外部零余额必须由 owner 拒绝。

指定 e 的行为：

- 新 e 只能是下一 e；若另一个 e 活跃，先拒绝，不能恢复错误 e 或跳号。
- 已终 e 在任何准备/恢复工作前返回原 attempt；即便外部首 result 尚未生成，也不打开 e+1。
- `_recover_interrupted` 在当前调用结算了 e，同样立即停止；这修复了仅使用 `episode_limit=1` 时恢复后继续开新 e 的问题。
- 显式请求先检查现有 native version/token 限额；已耗尽时不启动准备费用或新 coder。首轮 `max_version=0` 也不会为了不可能准入的 e 再跑 V0。
- 原生严格状态读取拒绝损坏 JSON、失去既有 episode accounting、缺失终态序列、非法 active 身份和低于 retained receipts 的 token 数，不能默认空状态退款。

`max_version` 与 `max_episodes` 是原二选一模式；实际 native 双限额是 version/episode 上限与累计 token cap，不把三个上限重新定义成另一套调度规则。

## 3. 预算、定位、准备和已结算通知

| 接点 | 真实位置 | 下游责任 |
| --- | --- | --- |
| `admit(phase="prepare", episode, workspace, active)` | 原生准备之前，含首 V0 之前 | A5 先持久化 owner/原 run/operation/version/native-root/预约关联；一次 preparation 可能含 V0/V1/并行预取 reviewer/production，不是一次模型调用 |
| `admit(phase="recover", …)` | 原生 active 恢复之前 | 恢复可能新评测/审阅，须贯彻原授权/fence，不能默认免费 |
| `admit(phase="episode", …)` | 原 native 限额检查后、新 active 创建或 coder 恢复之前 | 同一 owner 再准入；false 不启动 coder，不创建 fresh active e |
| `prepare_episode(worktree, active)` | 原 native materialize/link/clean 后、首次 draft registration 前 | A4 安装/验证准确完整 seed；自持 intent/ready/admitted、源码摘要、恢复绑定。回调异常保留 preparing；恢复 preparing 后即使被准入拒绝也不丢失该阶段 |
| `settled(episode, workspace, attempt)` | native attempt/state durable 且 active 已清除后；终态重放也通知 | A5 以原 operation/receipt 幂等吸收；通知失败后可重放，不重开 e |

这些是生命周期预约接点，**不构成完整逐 Agent/Evaluation 费用执行器**。A5 仍需在实际模型/GPU 调用、失败重试、V0/V1/helper、prefetch、fidelity、production 上接同一账户，保留最低收尾额度、unknown receipts、暂停/fence 与 owner active clock。这里没有复制父总账，也没有把 native tokens 当全部费用。

已准入后抛异常/KeyboardInterrupt，不调用伪造的 settled，不推断取消成功，不释放未知预约。native e 的终态也不代表所有远端 job/未知统计已结算。`settled` 可重复投递，owner 必须幂等；它不是新收费授权。

`workspace` 在首费用/首 result 之前可用，实际定位 `.atrex_long_horizon`、active、state、e 归档和 usage receipts。CPU 用 synthetic owner 在 V0 前落 native locator，模拟 V0 提交后未知且没有 result，再通过原 owner 拒绝下一 preparation，观察零重提。**没有实现或宣称外部进程原操作查询协议、未知 V0 job 自动回收或通用跨进程恢复**；这仍是 A5 资格的一部分。

## 4. 真实 V0、C0、G 与 HEAD

CPU 走真实 `ManagedCampaign.setup_baseline`、原 `_run_v0_evaluator_with_correctness_prefetch`、V0 source/measurement commit、format pin、worktree materialize、Git-free draft、原 runner 和原终态归档。只有 Agent/Evaluation/辅助 reviewer 服务是 fake。

fake evaluator 执行一份可执行的小型 CPU softmax+copy 基线，检查返回同一个 out 并写入正确值；计时为 synthetic。它不是首题 GPU evaluator 或 baseline/native-profile 的真实资格。

已观察：

- `framework_baseline="never"` 仍执行 V0，V0 先于 coder；已结算 V0 在后续 e/原 e 恢复时不重测。
- 配置 correctness reviewer 时，原 prefetch scheduler 在 `never` 下仍会调用它；测试保留实际调度，只 fake reviewer 服务。因此 A3/A5 不得把 `never` 当“准备零模型费用”。
- 不可执行 starter 与非空 partial 两例均在 V0 后、首次 draft 前安装；首次 Agent 收到准确 seed，V0 始终测 baseline，未拿 starter 冒充 V0。
- V0 source commit、V0 measurement/format metadata 后的 HEAD、episode outcome 后的 HEAD 可以不同。未晋升时 canonical 源码 G 保持 baseline，已归档 C 含新 seed/partial 字节。
- 第二 e 从当时实际 HEAD 创建 ancestry，再安装上次真实 `public-candidate.json` 中的终态完整源码；Agent 收到 C1，不退回 G。两 e 的原 native token receipts 累计为 14（各 synthetic 7），未重置。
- exploring draft 中断恢复不会重新执行 prepare callback。准备阶段中断可重入；中间暂停不把它错误改成已准备完毕。

当前 A1 没有生产用 seed codec/安装事务；测试里的安装只证明接点。A4 仍需冻结同一 callback/seed 语义，补完整 source/format/删除、intent/ready/admitted 和必要 V1 策略。原 native 在 materialize 尚未成功等中断点可能将 e 记录为 interrupted，A4 不应伪装其为 coder 成功。

原 native post-commit recovery 的 attempt 可能较稀疏，例如不再直接携带本次 token/candidate 字段；原 native usage receipts、public candidate 和归档仍可定位。A5 必须用这些原件重建投影，缺字段不是零费用/无候选，bridge 没有补造字段。

## 5. 实际测试与 off 回归

环境：Linux，系统/隔离解释器均实际为 **Python 3.13.12**。依赖仅安装在 `/tmp/schema-b02-a1-venv`，使用 `uv venv` / `uv pip`，`pytest==9.2.0`、`packaging==26.3`。没有修改系统 Python、KS 活动环境或计划。首发 Python 3.11 尚未在本 A1 回合验证。

最终命令：

```sh
cd /home/shaohanyong.shy/atrex-kernel-agent
PYTHONPATH=src /tmp/schema-b02-a1-venv/bin/python -m pytest -q tests
git diff --check
```

结果：**18 passed in 37.32s**；diff whitespace 检查通过。

覆盖首次指定 e/已终重放、两种 C0、原 version/token/max-episodes 限额、prepare/episode 外部拒绝、首轮 native 零 version 上限、错误 active e、缺失/损坏/退款式状态、原 Git outcome commit 后返回丢失、settled 通知失败重放、Agent 中断/暂停/保留 draft、preparing 中断/暂停/重入、V0 未知后的 owner fence，以及 never 的实际 prefetch 行为。

上游 checkout 没有已跟踪的 native Workflow 单元测试目录（`tests/` 被整体忽略）；找到的已跟踪测试是 Wiki 工具、CUDA 示例及 evaluator script，不适用于此接点。因此另以**基线源码运行相同原生 CPU 场景**作直接回归，而没有调用无关 GPU/Wiki 测试。

用 `git archive bf5557ac04f6ff9988901d511cf25e1cbe61a1bc src` 生成只读比较用源码快照到 `/tmp/schema-b02-a1-upstream-ox6edpiw`，不是另一个 Git 工作树；运行：

```sh
PYTHONPATH=/tmp/schema-b02-a1-upstream-ox6edpiw/src /tmp/schema-b02-a1-venv/bin/python tests/schema_off_probe.py /tmp/schema-b02-a1-off-upstream-final /tmp/schema-b02-a1-upstream-trace.json
PYTHONPATH=src /tmp/schema-b02-a1-venv/bin/python tests/schema_off_probe.py /tmp/schema-b02-a1-off-fork-final /tmp/schema-b02-a1-fork-trace.json
```

两者均走原完整 `campaign.run`，V0 一次、两次 coder/e、setup 一次、原 `_finish` 一次；停止于 native max-iters，tokens=14，零晋升。probe 在导入前禁止 KS/Schema 模块、KS/optional 路径访问及 socket 连接，未触发任何禁项。

归一化保留 prompts、源码、native evaluator argv、limit/outcome、最终源码/导出主体与 setup/finish 计数，去掉运行根路径；JSON 对象及文件字节完全一致。两份 trace SHA-256：

`898cd2ad20ad46e8230c61498f2a7022c5e986df5c1f9915e436267b594adf99`

这证明本 CPU 场景默认 off 与基线一致，不声称随机模型输出逐字相同。没有 KS 安装/switch/package 发行检查，完整 V12/V14 仍由 A3/A7 汇合。

## 6. 给后续 owner 的明确交接

- **A3**：消费 optional bridge/namespace 与原 loop kwargs；从原构造路径复用一份 native engine，保留默认 off；完成 provider/switch、固定源码/KS 安装和角色权限。源码来源按本记录归口，无 EXP 迁入依赖。
- **A4**：消费首 draft 前 `prepare_episode` 及严格 preparing/exploring 区别；完成 baseline/C0、准确终态 C/当前 HEAD、完整 source/删除与 seed 恢复协议；不要把 G 当下一轮 seed，不要用不可执行 starter 作 V0。
- **A5**：消费三阶段 `admit`、重复安全的 `settled`、`EpisodeResult.native_root`、真实 Store/active/archive；把外部 run/operation/version/owner 绑定在首费用前落盘，并贯穿逐收费操作、暂停/未知/最低收尾、三 audit、实际外部定位与完整结果投影。native terminal 不替代 Schema qualified 或远端费用结算。

仅承诺在保留 native 状态、原配置和 outer owner 锁的串行调用中的 CPU 接点。未实现完整 provider、固定 backend、资格 audit、跨进程账户恢复、实际 GPU/模型/Agate、动态 planner 或发布；这些保留在已分配的后续单元。无需为本 A1 改 EXP 或阻塞 KS 独立链。
