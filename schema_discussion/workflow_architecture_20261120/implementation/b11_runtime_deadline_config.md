# B11：业务 runtime 的显式期限配置

2026-11-20。**业务 `ClaudeProfile.timeout_ms` 已允许显式配置超过 120 秒，默认仍为 120000 ms。新 launch 与重复等待共同使用原持久化截止；未进行任何真实长调用资格试验。** 本项依据 [B10 的实际 design 超时](b10_k5_design_trial.md)作有界修复，没有重试那个已关闭的 scope，没有重新规划业务。

生产修改仅在 `/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1`，起点为干净的 `76ef9b1427e4d24174557054d049e7b134c9958d`。七文件已向 supervisor 提交最终 SHA 并冻结，由 supervisor 精确提交；本代理没有提交、推送、派生代理或修改 AKA。原 main 仅新增本报告。没有读取、枚举或运行 `AKA_experiment`，所有 shell 工作目录均为明确的 KS 或本项 `/tmp`。

Supervisor 已生成交付提交 `1c9e592c6e0ec021f91e17af5a8c39956ce39178`。本代理最终只读核验工作树干净，提交后的七文件 SHA 与下表完全一致。

本批真实 MODEL calls=0、GPU jobs=0。真实全会话 6 calls / GPU 0 的原账目不因本项改变。B10 原 planner 的失败、tokens=null、USD=null 与闭批决定原样保留；未新增真实账户、请求或预约。测试中的 OwnerAccount 和 subprocess 都是独立 CPU fixture，不是业务额度或资格证据。

## 合同依据与实际修复

[C 合同 §6](../planning/r5_contract_execution_review.md#budget)要求所有收费接点取 parent/request/native 最紧限额、保留最低收尾预约、同 owner 不重置额度、未知消耗不退款；没有把 120 秒定义成不可更改的全产品上限。旧 runtime INTERFACE 的 “Timeout is at most 120 seconds” 是本次明确修订的产品限制。公开 profile、历史资格总结、B10 原记录都没有改成更长调用已经通过。

实际发现的第二个缺口是 `invoke(wait=True)` 原先每次按 `now + profile.timeout_ms + 5s` 等待。仅放开 profile 后，已很短的原请求/预约截止或者已过期的恢复仍可能让宿主重新等待数分钟。因此同时修复宿主等待窗口，而非只改配置校验。

| 文件/环节 | 最终行为 |
| --- | --- |
| `runtime/claude.py` / `ClaudeProfile` | timeout 接受严格正 safe integer，范围 `1..2**53-1` ms；bool、float、string、null、零、负值和不安全整数拒绝。默认 120000、profile 格式/版本/字段不变，timeout 改变仍改变 profile/后端身份。 |
| `runtime/claude.py` / 最终 launch | 保留 `min(profile.timeout_ms, fixed_request.active_ms, original_reservation.active_ms)`；原 owner `authorize` 仍在收费前核验原预约、暂停/fence、实际剩余额度。 |
| `runtime/_local.py` / 冻结窗口 | 每个新 launch 都保存有效 UTC、monotonic、boot 与 timeout。只有 duration 的调用也从最终 launch 固定窗口，排队不能再取得一整段 duration；没有 owner 绝对字段时，launch 之前的输入准备仍不能靠 duration 推算扣除。极大数值的 UTC 截止取紧到协议 safe-integer 上界，不发生 JSON 溢出。 |
| 原 `_runner.py` / supervisor、child | 未改源码。supervisor 生成 child 前、child hash/fsync 后且 exec 前、实际等待/kill 都沿用 `remaining_timeout_ms` 对原截止取紧。过去的 UTC-only 预约仍会冻结有效 monotonic 截止，抵抗后续 wall-clock rollback。 |
| `runtime/claude.py` / 宿主等待 | 读取原 `launch.json`，取两个原截止的剩余最小值，再仅允许固定 5 秒终态收集。此余量由 supervisor 明确认可，只收集 kill/exit 事实，不延长 CLI 执行；多次 invoke/continue 不累加、不重新从 now 起算。实际等待继续进入原 owner active clock。已完成终态的回放不会因为后来 boot 改变被等待检查误拒绝。 |
| 原 `budget.py`、writing、reference implement | 未改账户、预约、调用吸收、最低 reviewer/finish reserve 或新 Q 绑定逻辑。原请求与父账户的 active、UTC 截止和原预约的 UTC/monotonic 截止仍从同一授权收紧；没有第二预算、通用调度或迁移机制。 |
| 原 `runtime/qualify.py` | 未改源码。明确保留 request.active_ms=120000、reservation.active_ms=120000、单次调用上限；USD native threshold 仍 ≤0.50。传入长 profile 时也保留其准确身份，只在资格请求/预约层取紧，不以短 profile capture 代签长 profile。 |

固定 5 秒结束后仍未确认终止，只能保留原操作及 fence 并继续 settle，不能据宿主停止等待推断零调用、退款或再次准入。原 supervisor 仍负责原进程组终止与 exit 收集。

## 使用边界

公开 [runtime INTERFACE](/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1/kernelschema/runtime/INTERFACE.md)、[WRITING](/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1/kernelschema/WRITING.md)、[protocol](/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1/docs/workflows/protocol.md) 和 [profile README](/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1/kernelschema/runtime/profiles/README.md)已一致说明：

```python
from dataclasses import replace
from kernelschema import writing

new_claude_profile = replace(fixed_claude_profile, timeout_ms=600000)
new_business_profile = writing.backend_profile(new_claude_profile, environment_ref)
```

这只创建不同身份的业务配置。应先冻结新 profile，再构造和发布尚未准入的新 Q；仍使用原 `parent_ref` 和原私有 authority，不能换 owner 消除历史消耗或未知费用。根据实际需求，在新 profile 冻结前选择 reviewer/finish minima；加长 timeout 不自动增加默认 reviewer minimum，也不扩充 active/call 余额。已准入的 Q/profile/reservation、旧 run 与旧 capture 均不可改。

旧 120 秒 profile 的真实资格不自动传给新 profile。资格 probe 即使绑定新长 profile，也只证明其实际受限试验中的行为，不证明 >120 秒真实调用或完整 design 成功。本项没有安排新的模型调用；B10 的失败 scope 绝不重跑。

## 定点 CPU 验证

解释器为原 `/home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k2-cpu/bin/python`，Python 3.13.12；没有安装、下载或切换解释器。测试使用 `PYTHONDONTWRITEBYTECODE=1` 和 `-p no:cacheprovider`，未运行全套 CPU 测试。

1. 首轮选定 runtime 测试：**34 passed、4 failed、47 deselected，53.79s**。四个失败全在新 fixture 的最后 usage 发布：误将原 receipt Ref 发布到空的另一 bundle，因闭包缺原 receipt 正确拒绝。此前短限终止、双截止、一次调用、未知 tokens、原操作回放、不重复吸收的断言均已通过。修正测试在原 run 发布 usage；没有为测试改变生产账本/闭包规则。
2. 最终源码上只复验受影响分支：**9 passed、76 deselected，21.99s**。包括四种实际 `OwnerAccount` 场景（parent active、request active、parent UTC deadline、request UTC deadline），600000 ms profile 被约 2 秒原窗口与 reviewer reserve 收紧；fake CLI 被原 supervisor 终止，tokens 仍为 null；相同 handle 重开、跨 boot 的已完成终态回放、一次吸收与 coder 额度不退款。UTC 或 monotonic 任一截止已超过原 5 秒收集余量时，重复 invoke 不 sleep、不 spawn、不再 authorize，原 launch 字节不变。
3. 既有 writing minimum reserve / 同 parent 新 Q 的必要兼容：**4 passed、48 deselected，30.03s**。覆盖最低 reviewer 不足时零调用、外部 finish floor 不足时零调用、reviewer 后同 owner 仍保留合法后续预约，以及新 Q 不重置 parent 消耗。
4. `git diff --check` 通过。默认120秒、长profile不同身份/不能塞回已冻结旧Q、无效类型/不安全整数、最大合法profile保持安全UTC、原公开profile逐字节载入、supervisor排队过期、child exec前拒绝、剩余窗口、monotonic抵抗wall rollback与原caller死亡恢复等在首轮定点集合通过。

准确命令与日志：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k2-cpu/bin/python -m pytest -p no:cacheprovider -q tests/workflows/test_claude_runtime.py -k 'profile_default or profile_timeout or long_profile or maximum_profile_timeout or deadline or timeout_kills_original or unknown_original_process or actual_caller_death or public_profile'

PYTHONDONTWRITEBYTECODE=1 /home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k2-cpu/bin/python -m pytest -p no:cacheprovider -q --basetemp=/tmp/ks-b11-runtime-deadline-20261120-YjaFav/recheck-runtime tests/workflows/test_claude_runtime.py -k 'long_profile or deadline_boot_change or timeout_kills_original or unknown_original_process'

PYTHONDONTWRITEBYTECODE=1 /home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k2-cpu/bin/python -m pytest -p no:cacheprovider -q --basetemp=/tmp/ks-b11-runtime-deadline-20261120-YjaFav/writing-reserve tests/workflows/test_writing.py -k 'finish_floor_blocks_planner or writing_finish_floor_short_budget or writing_finish_floor_survives_reviewer or same_parent_across_new_q'
```

日志分别为 [首轮](/tmp/ks-b11-runtime-deadline-20261120-YjaFav/targeted-runtime-tests.txt)、[最终受影响复验](/tmp/ks-b11-runtime-deadline-20261120-YjaFav/recheck-runtime-tests.txt)、[writing兼容](/tmp/ks-b11-runtime-deadline-20261120-YjaFav/writing-reserve-tests.txt)。首轮成功数与复验成功数存在重叠，不相加伪装成不同测试数量。

## 原 76 源码、安装与 B10 身份保全

编辑源码前，从固定 Git commit 的公开路径归档 **183 个文件**到只读目录：

`/tmp/ks-b11-runtime-deadline-20261120-YjaFav/original-76ef9b`

范围为 `kernelschema`、`schema_template`、`schema_docs`、`examples/workflows`、`docs/workflows`、三个写作/实现规范和 `pyproject.toml`；只取 Git 固定文件，没有复制私有 capture 或认证状态。归档 [original-76ef9b.tar](/tmp/ks-b11-runtime-deadline-20261120-YjaFav/original-76ef9b.tar) SHA-256 为 `ea7fa59f86af9d8c5d650e477d472465b5845ad6134899e585b938cfac4dceee`。最终逐文件复核原 manifest 且均无写权限。

原 `k2-cpu` 安装、Python executable、真实 CLI、原 run/request/owner/capture 目录都没有改。**该环境默认 editable 导入仍指向会更新的工作树，因此恢复旧 B10 时必须显式选择原快照，不能拿新源码冒充原实现。** 在本项明确 `/tmp` 工作目录下，以 `PYTHONPATH=/tmp/ks-b11-runtime-deadline-20261120-YjaFav/original-76ef9b` 搭配原 `k2-cpu/bin/python -B`，已只读核验：

- `WritingBackend.identity` 与 B10 原 `run.json` implementation 完全相等。
- `runtime.reopen` 可读取原 operation `50c816956e0f453fbc01334fbf048175_planner`；没有 invoke、settle 或 API continue，没有收费动作或修改原事实。
- 原 profile SHA-256 仍为 `76fd98e0094f4a3e07b849c8f7a9207752689dd236105bd82919d904b405603e`。
- B10 原 result SHA-256 仍为 `6976c24012c70db0b5ef8b5000182e34092854070f85936d3d71ba108f0542b8`，原失败 receipt SHA-256 仍为 `2808b8778e2b0c737b56db4aab8d592133a48333d9ecaf241ed71142441ae608`，与 B10 报告一致。

证据：[原实现身份核验](/tmp/ks-b11-runtime-deadline-20261120-YjaFav/original-identity-check.json)、[公开源码清单](/tmp/ks-b11-runtime-deadline-20261120-YjaFav/original-source-manifest.json)、[最终保全核验](/tmp/ks-b11-runtime-deadline-20261120-YjaFav/preservation-check.json)。此保全不移动原 private runtime，不修改原身份，也不声称支持任意搬迁恢复。

## 最终冻结清单

以下 SHA-256 已通知 supervisor，可精确提交七文件；不包含本报告或任何 `/tmp`。

| 相对固定 worktree 的路径 | SHA-256 |
| --- | --- |
| `docs/workflows/protocol.md` | `47537248f40a00176d2e1a998f55201b60cdcd884819358fe40fac5e2366f146` |
| `kernelschema/WRITING.md` | `aa7a45a5ba57f9f7f035b1c0f4a552a40ed8482f109cebe8b7ff1be3c27c0915` |
| `kernelschema/runtime/INTERFACE.md` | `0e52c4b8844733046278f732b6a5668ac79488e97c0630bade9b26d854811304` |
| `kernelschema/runtime/_local.py` | `7a60dee10f6048dae32eb8a1f34b575442a015b7f9058d500c54e9f5e969a6d5` |
| `kernelschema/runtime/claude.py` | `6df1e6522f0f6947ea84beacbbbdf95b51fb3744420c7d53894a8d085fab20f3` |
| `kernelschema/runtime/profiles/README.md` | `80a53327f3f1326d29bdb38a244aa80c2172dd5c86bdd1a34e12a3fc7f034e7d` |
| `tests/workflows/test_claude_runtime.py` | `8463526f83661f362b15ac808ae3bd65a943a251352226d41ba29d22704d1e94` |

[完整 patch](/tmp/ks-b11-runtime-deadline-20261120-YjaFav/final.patch) SHA-256：`4c88c2fe244124a9aa8194b9e5a9c0f00b06222fec6639205bb6a5e4c0c6beda`。

仍未验证：>120 秒真实模型调用成功、长profile真实资格、完整K5.real design/reviewer发布、optimize实际业务与GPU任务资格。没有等待数分钟模拟成功，也没有借原120秒profile的资格补齐这些结论。
