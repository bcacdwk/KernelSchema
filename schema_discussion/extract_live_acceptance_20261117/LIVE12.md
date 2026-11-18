LIVE12 — FA4 r04 孤儿 reviewer 的正式中断结清与恢复

结论：本现场可以直接用一次正式 `resume` 恢复。冻结 Runner 会凭原启动门的后代回收证明确认终止，将缺少 `result.json` 的旧调用正式记为已结清的中断，退休旧会话并重新准备 normal P2 reviewer。它不会补造成功结果或批准，不需要 pause、maintenance、手工改 UUID、补 exit、清 fence 或修改快照。本轮只验证并给出步骤，没有操作任何真实 run。

核查起点为 `CURRENT_SUPERVISION.md` 最新 21:24 交接；只补读必要 README、冻结恢复代码及所需原记录，没有通读历史。适用 run 为 FA4 r04 `ebe745a0-e758-43c4-9679-2e8aa6e2a842`，旧 reviewer 为 `0ef292dd-4d41-4557-b88c-814170e7b1ae`。

现场事实及成功边界：

- 原 state 仍为 running / P2 / reviewer；active.P2 和 unsettled 指向旧 attempt，unconfirmed 为空。Runner、启动门及 process.json 中所有登记身份本次只读核查均 `process_alive=false`；没有向任何真实进程发信号。
- `cli-exit.json` 的 launch_id 为 `e417498f-6aba-4cc2-b338-2773e90000e0`，与 invocation/launch/process 一致；protocol 与 attempt 摘要绑定成立，exit_code=0、descendants_reaped=true。冻结 `gate_reaped` 对原 attempt 返回 true。
- 原始事件经冻结解析器确认：唯一 init，session UUID `5ffd0d95-cfb4-455c-93c6-b9758b0c3af0`；`native_status.success=false`、terminals=[]、issue=`missing_or_nonfinal_unique_terminal`。`exit.json`、`result.json`、`response.json` 均缺失，invocation 没有 cwd_after。启动门 exit0 不能成为原生成功、有效回执或 P2 批准。
- 当前合法 checkpoint 为 `9e63214d1d37144892ef4fd4c7d2ece431328f9d35522e879b9e4b23af6fdb4e`。plan_draft 保留 root（shape14 单 launch persistent）与 split_kv（shape0 num_splits=5 及 combine）的两个选择；selected_designs 和 P2 approval 仍为 null。P1 批准有效。
- session8274 已不可查询，Runner 消失的原因仍未确定；本报告没有把前台工具会话生命周期、日志沉默或模型行为当作已证根因。

核查执行器是该 run 原冻结实现：snapshot `29223addb27262692e15bae48ba6013e29879ad2d569c2676487074a110ed993`，rules id `5038818c033a8521bbf77ffa859f298dd1e29535da728850073c471ac0630809`，executor version 2.5.1、digest `7ccd2c0e6e458bd981f88796ebc9519977548d3ec5f9b40ce28d0cda0e151a5b`。以下行号均相对该 snapshot 的 `rules/schema_extract/tools/`。

正式恢复为什么可行：

1. `cli.py:437–448` 为显式 resume 登记正式启动身份，再切入冻结入口；`runner.py:1267–1327` 校验原 inputs/config/executor、消费控制，再调用 `_reconcile`。旧 Runner 身份确认已退出时，`monitor.py:360–368` 允许登记新 launch；未知或仍活着的 owner 不会被当作死进程覆盖。
2. `runner.py:999–1016` 先按原 attempt 计账，再调用 `confirm_termination`。`recovery.py:250–286` 核对 process/launch/invocation 绑定和进程身份；即使进程全退出，也必须由 `_closed_ownership` 确认启动门已回收全部后代。`processes.py:259–271` 对 protocol、launch_id、attempt_id、attempt 摘要及 descendants_reaped 逐项绑定。
3. 原 `result.json` 缺失，因此 `runner.py:1024–1034` 不走结果成功重放，而直接调用 `_interrupted`。`runner.py:875–931` 保留 reviewer 的当前候选，正式释放旧 active、生成 recovery 记录；任何中断决定均不能批准阶段。
4. `recovery.py:394–434` 从原事件读取旧原生 UUID，调用 `retire_session`，令私有 session.status=retired、session_id=null，并写入 handoff 的 previous_session_id。旧私有 session 原来未保存 UUID，因而 retired_sessions 列表仍可为空；完整旧 UUID 由 handoff 和原事件保留，不需要手工补 UUID。新调用将获得完整材料和这份短交接。
5. 同一恢复操作把 P2 scope 恢复为 reviewing。`runner.py:1359–1397` 按正常预算和 next_calls 准备新的只读 reviewer；新 attempt 的 session_id=null，Claude 命令不带 `--resume`。新 reviewer 必须自己产生当前 nonce 的有效提案，通过原生成功、来源/产物证据、schema、候选/plan 一致性和独立批准门禁，才可能使 P2 通过。P3–P6 未被跳过。

临时副本的实际验证：

验证脚本 `/tmp/live12_probe.py`；通过结果 `/tmp/live12_probe_7sttrvn5/result.json`，原文件摘要清单 `/tmp/live12_probe_7sttrvn5/real_before.json`。复制完整冻结 source/rules 与公开 run 文件，排除私有 CLI home。副本 state.snapshot、run inputs.lock 和 snapshot inputs.lock 全部指向复制快照；attempt 路径及受路径迁移影响的绑定摘要在副本内一致重映射。源码、规则、候选字节、模型配置、预算、UUID 和决策内容没有改动。冻结 `verify_run_inputs` 和执行器校验实际通过。

Python audit hook 将所有写入限制在本次 `/tmp/live12_probe_7sttrvn5`；子进程只允许只读 `ps`。正式 Runner 使用禁止调用的 transport，信号函数亦被强制阻断。没有启动模型、CLI 会话或 GPU，没有运行全量测试。

| 检查 | 实际结果 |
| --- | --- |
| 正式 `reserve_launch(mode=resume)` + `Runner.run(resume=True)` | 原调用先结清，随后实际进入首个新 attempt 的正式 prepare；在 dispatch 前截停 |
| 旧调用结清 | termination.confirmed=true、actions=[]；新增 termination/recovery/handoff/finished；旧 attempt_usage.settled=true |
| 不提升缺失结果 | 旧 exit/result/response/receipt 文件仍缺失，旧 attempt 没有进入 accepted_receipts |
| 新准备调用 | normal / P2 / root / reviewer，purpose 无额外维护身份，writable_files=[]，session_id=null；原 checkpoint 和 plan_draft |
| 新会话入口 | 冻结 `_session` 接受 retired/null；正常 Claude 命令无 `--resume`；完整新 prompt 47026 字节，材料路径均在临时副本 |
| 计数与预算 | 新 prepare 前旧 calls=3，新 prepare 后 calls=4；既有 calls、revisions、deadline、operator deadline 不清零或补时；此前其他 attempt 的 elapsed 不变 |
| 候选与批准 | checkpoint、候选字节、plan_draft、P1 approval、既有 accepted_receipts 在迁移后的基线内保持；selected_designs/P2 approval 仍 null |
| 重放 | 对已结清旧 attempt 再 reconcile，不重复记账、不重派它 |
| 反例 | 只移除另一临时副本的 cli-exit.json，其余原进程身份仍全部退出；正式 resume 返回 termination_unconfirmed，active/unconfirmed/unsettled 保留，旧私有 session 仍 running，零新 prepare |

旧调用没有最终 result，账目保留 `usage.status=unknown`、`usage_not_recorded` 和空 receipt；没有捏造 token/cost 或 outcome。实际中断依据在 recovery/handoff/finished，不是新增 `result.outcome=cli_error`。`elapsed_on_disk` 在最终计时样本缺失时按创建至恢复的墙钟保守记账，本次副本为约 4185.719 秒；实际恢复时可能更长，原 deadline 始终不变。这是保留失败成本，不是预算退款。

副本验证前后，739 个原公开 run/冻结文件及共享 supervisor、QSA state 的摘要逐项一致。没有读取或输出模型隐藏推理、认证内容；原事件只由协议解析器提取终端状态及会话身份。

| 原文件 | SHA-256 |
| --- | --- |
| FA4 state.json | `88bca25b49025fcf72385d3247d44a6e8e24bd5f3c72e3405e38b2a32f198622` |
| FA4 inputs.lock.json | `c89f52a9769b9f255f820ae01e6681e188fdac53b457dd647965622622621533` |
| 旧 reviewer events.jsonl | `38b4be9fc9119d1df11d7c00a86b40eccbb229244cac93e23c9ef460b3554d06` |
| 旧 reviewer cli-exit.json | `cbb928ed50da121ce5f59528c7382495aa08c875dc7fa43b34c781bc93bf0aac` |

给主代理的最小正式步骤（以下均未执行）：

1. 只读复核仍是上述旧 attempt、有效回收绑定、无新 Runner 与冲突控制/hold，并且原预算尚有余量。不要根据静默时长判断终止；以启动门及精确进程身份为准。
2. 一次正式 resume 即可，不增加预算、不改模型、不补结果、不先做 maintenance：

```sh
/opt/venv/python3.13.12/bin/python3 -B /home/shaohanyong.shy/KernelSchema/schema_extract/extract_yaml.py resume ebe745a0-e758-43c4-9679-2e8aa6e2a842 --repo /home/shaohanyong.shy/KernelSchema --state-root /home/shaohanyong.shy/KernelSchema/.schema_extract/live_acceptance_20261117
```

3. 确认旧 attempt 的 recovered/settled、termination 与 handoff，首个新 attempt 应为上表的 normal P2 reviewer，并绑定 `9e63214d…` checkpoint。新 UUID 必须来自新的正式 CLI；不得自行指定。若证据改变或仍 termination_unconfirmed，保留 fence 和新原因重新判断。此步骤验证恢复起点，不预先宣称 P2 或后续阶段通过。

后台恢复入口的轻量核实：

- 冻结 CLI `resume` 没有 `--background`（`cli.py:123–142`）；当前产品入口同样没有该选项。`extract --background` 调用 `monitor.start_background` 时默认 mode=queued（`cli.py:512–515`）。对于这个非 queued run，`monitor.py:369–370` 返回 preserved。临时副本实际调用得到 `preserved / run_status=running`，没有启动后续进程。因此不能重复同一 extract 请求代替恢复。
- 内部正式 launcher 已支持 `start_background(..., mode="resume")`；其 reserve → detached Popen → `_background` claim → 冻结 Runner 路径仍使用相同身份和控制屏障。临时副本已验证 mode=resume 的后台 reservation 为 launching；本轮未启动该后台进程。若产品以后需要公开入口，可让 `resume --background` 最小复用该 launcher，并沿用现有预算追加/控制语义；本轮不实现。
- 现在可将上面的原正式 resume 用已安装的 `/usr/bin/nohup` 与 `/usr/bin/setsid` 脱离前台会话运行。所有标准流明确重定向，使用独立日志，不依赖 session8274 或新的临时 exec session 持续存活。主代理只应从前台命令和以下命令中选择一种启动一次：

```sh
/usr/bin/nohup /usr/bin/setsid /opt/venv/python3.13.12/bin/python3 -B /home/shaohanyong.shy/KernelSchema/schema_extract/extract_yaml.py resume ebe745a0-e758-43c4-9679-2e8aa6e2a842 --repo /home/shaohanyong.shy/KernelSchema --state-root /home/shaohanyong.shy/KernelSchema/.schema_extract/live_acceptance_20261117 </dev/null >/tmp/live12-fa4-r04-resume.log 2>&1 &
```

这只改变宿主启动方式，仍执行完整正式 CLI、冻结验证、launch 登记和 Runner。启动成功应以新的 supervisor launch、state.runner 身份、runner.lock 与新 attempt 为准，不能把 shell 返回或日志为空当作接管成功。本轮仅确认程序存在并审阅启动语义，没有实际执行这条命令；不声称它解释或消除尚未查明的上次 Runner 消失原因。

启动后可用下面的只读观测记录实际 Runner PID/PGID/start_token、正式 launch 身份、锁状态及日志位置；记录写在 `/tmp`，不写运行状态。它只读取现有登记，不调用内部 reserve/claim，也不把 shell 的 `$!` 当作实际 Runner 身份：

```sh
/opt/venv/python3.13.12/bin/python3 -B - <<'PY' >/tmp/live12-fa4-r04-launch-observation.json
import json, pathlib, sys, time
root = pathlib.Path('/home/shaohanyong.shy/KernelSchema/.schema_extract/live_acceptance_20261117')
rid = 'ebe745a0-e758-43c4-9679-2e8aa6e2a842'
state = json.loads((root / 'runs' / rid / 'state.json').read_text())
ledger = json.loads((root / 'supervisor.json').read_text())
sys.path.insert(0, str(pathlib.Path(state['snapshot']) / 'rules'))
from schema_extract.tools.processes import process_alive, process_table
from schema_extract.tools.store import Store
table = process_table()
runner = state.get('runner', {}).get('identity')
launch = ledger['starts'][rid].get('launch', {})
owner = launch.get('identity')
print(json.dumps({
    'observed_at': time.time(), 'run_id': rid,
    'runner': runner, 'runner_alive': process_alive(runner, table),
    'launch_id': launch.get('id'), 'launch_status': launch.get('status'),
    'launch_owner': owner, 'launch_owner_alive': process_alive(owner, table),
    'same_registered_runner': bool(runner and owner and
        (runner['pid'], runner.get('start_token')) == (owner['pid'], owner.get('start_token'))),
    'writer_lock': Store(root).writer_status(rid),
    'active_attempts': state['active_attempts'],
    'log': '/tmp/live12-fa4-r04-resume.log'
}, ensure_ascii=False, indent=2))
PY
```

接管确认条件是新 launch 为 claimed、登记 owner 与 state.runner 精确同身份且都 alive=true、writer_lock=held；旧 Runner1115979身份仍应不存在。正式 Runner 的排他 writer.lock 防止同 run 双写。若启动尚未完成，稍后重复只读观测；若条件不符先查日志及登记，不立即叠加第二次 resume。确认后以观测 JSON 保存的新 PID 为准继续监督，无需新增后台或监控模块。

本次仅新增本报告并写 `/tmp` 诊断副本；FA4/QSA 真实运行、原稿、批准、源文件、规则快照与产品代码均未修改。完整 FA4/QSA P1–P6、最终发布和验收审计仍由主代理继续执行。本报告完成，停止写入。
