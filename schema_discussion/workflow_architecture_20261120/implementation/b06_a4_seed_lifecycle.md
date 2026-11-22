# B06 / A4.cpu — AKA V0、C0 与准确源码续写

日期：2026-11-20 UTC。状态：A4 的 CPU 源码/恢复接口已实现并通过限定验证；A5 固定业务循环、真实模型/GPU、native 三审计及 production 交付仍未验证。

## 1. 固定输入和改动范围

- AKA 工作树：`/home/shaohanyong.shy/atrex-kernel-agent`，`feat/schema-workflows-v1`，本批前后 HEAD 均为 `03a3f109fe0a89dd4a8fc77bb0070ebe771eba7f`。本批未提交、推送、切分支或改 Git 配置。
- 只消费 AKA 内固定 KS 安装：`optional/KernelSchema`，detached `189a54153a74ac659d26652a9426bc0cc638a59a`。未更新 gitlink、KS 安装字节或 release 文件，未读写 KS 开发源码，未访问 EXP。
- 从 `/tmp` 使用 Python 3.13 隔离环境 `/tmp/schema-b05-a3-venv/bin/python -I` 调用 `verify_release()` 成功：`ks_commit=189a54153a74ac659d26652a9426bc0cc638a59a`，manifest digest `sha256:e819b9b6bc053aef20092060b638aa75a0bdaf68a6c291a41b8b3a1623bebab4`，实际 KS root 为 AKA 子模块。
- 未下载解释器/依赖栈，未调用模型、GPU 或 Agate。仅复用已安装 CPU Torch 执行真实 baseline 的小规模逻辑检查。
- 阅读了 P/C/V、A1/A2/A3 说明和实际 native/KS 接口；本批遵守只交此报告的 KS main 写入范围。没有派生子代理。

实际产品文件：

| 文件 | 变化 |
| --- | --- |
| `src/aka/workflow/schema/preparation.py` | 新 `SeedLifecycle`，负责源码状态、seed 准备、admitted 恢复、一次结果安装及终态 C 捕获 |
| `src/aka/workflow/schema/source.py` | 将 A2 原完整校验抽为 `read_native_source`；原 `freeze_native_source` 委托它；没有复制或修改 KS Source codec |
| `src/aka/workflow/light/long_horizon/campaign.py` | 11 行增加、2 行替换：可选 `before_draft` / `draft_ready` 接点；没有新业务循环或 Schema import |
| `tests/test_schema_seed_lifecycle.py` | 当前 35 个 A4 CPU case；使用原 native Git/Store/EpisodeWorkspace/A1 与实际 A3 接口 |
| `docs/schema-seed-lifecycle.md` | 接口、状态、V0/V1/helper/费用和未验证边界 |
| `docs/schema-episode-seams.md` / `schema-optional-module.md` | 记录新增窄接点及 A4 的具体安装接口 |

新测试以精确 `git add -N -f tests/test_schema_seed_lifecycle.py` 保持可见；`git diff --cached --name-only` 为空，未暂存内容。新 preparation 和说明文件是未跟踪文件，须由后续统一审查/提交收录。

## 2. 可调用 API 和原 native 路径

```python
from aka.workflow.schema.preparation import SeedLifecycle

sources = SeedLifecycle.create(original_run_context, original_engine, lifecycle_owner)
outcome = sources.run_episode(1)
c1 = sources.current_source()  # SourceBinding，终态准确 C，独立于 G

sources = SeedLifecycle.reopen(original_run_context, original_engine, lifecycle_owner)
outcome = sources.run_episode(2)

# A5 从原 RoleSession/NativeCallAuthority 得到一次已经授权的返回结果后：
binding = sources.install_result(e, native_role_result, authority=original_authority)
```

`create` 只允许无 native Git/active/work 的初始状态。恢复调用 `reopen`，缺失/损坏状态不会创建新 seed。私有记录通过 `kernelschema.api.RunContext.save_state/load_state` 持久保存；沿用原 run_id/Q/owner，并绑定 Task、R、rules、scope、profile、C0、baseline、native 位置、framework/conversion、V0 reviewers 和 native 上限。恢复不得替换这些身份。

具体代码入口：`preparation.py:120` create、`:135` reopen、`:242` prepare callable、`:291` before_draft、`:302` draft_ready、`:319` current_source、`:325` install_result、`:392` settled、`:422` run_episode。A2 的只读校验在 `source.py:42`，原 freeze 在 `:88`。

实际执行仍是 A1 `bridge.run_episode` → `LongHorizonCampaign.run` → `main_adapter.prepare_campaign` → 原 `ManagedCampaign.setup_baseline`/V0 → 原 `GitCandidateWorkspace.plan/materialize` → A4 prepare callback → 原 `register_runtime_episode`/`EpisodeWorkspace.prepare` → 原 runner。A4 没有直接调用完整 `Campaign.run`，也不调用 `_finish`。最终收尾仍由外层原 Campaign 执行一次。

新增 native hooks 位于 `long_horizon/campaign.py:1880`、`:1897`、`:1904`：支持 draft_ready 的准备对象在 hook 持久完成前保持 native preparing；before_draft 在原 EpisodeWorkspace 有机会自动替换源码前核对已准入状态。普通无 hook 的 A1 callback 和 off 路径维持既有行为。

## 3. V0 / C0 / G / HEAD 事实

V0 的 `InitialSource` 必须逐字节等于 `Task.baseline_source`。测试通过实际 `SchemaProblemProvider` 和 `LegacySourceProvider` 加载真实 Task baseline：FP32 `torch.softmax(x, dim=-1)` 加 `out.copy_`、返回原 out。测试 Evaluation 执行该实际源码并检查 out 身份和值，随后提供明确标为 synthetic 的 native timing，使 CPU 测试可以进入原 native 后续 lifecycle。它不是把 A2 的 CPU 状态改成 CUDA PASS。

两种首调分别使用：

1. `seed=null` → 原 `Task.starter_source`，该骨架不可执行；
2. 用户显式非空 partial Source，包含 Unicode、CRLF、无末尾 LF。

二者均在 V0 完成之后、第一次 draft/coder 之前安装，首调收到准确 C0。显式 seed 不可读、摘要/模式不符、缺 `kernel.py`、额外 `extra.py` 或 `solution.json` 的 public Source payload 均在收费入口前拒绝，不退 starter 或 G。合法 null/null empty_start 在当前 AKA profile 不支持；有显式合法 seed 时不受 Task starter 为 null 影响。

测试的 e1 故意以 native `invalid_handoff`、accepted=false 结束，保留完整但未晋升的 C1。e2 取当时的新 canonical HEAD 作 ancestry，然后安装准确 C1；实际下一次 Agent 收到 C1，G 仍是 baseline。V0 source revision、metadata 后 HEAD、e1 outcome 后 HEAD 与 C 源码身份明确分列。旧终态 e1 重放不会把 latest 从 e2 回滚。

这个证据覆盖 V04/V11 的未晋升准确续写，不把 invalid_handoff 冒称为“慢但已资格合格”。V03 的 qualified/completed/performance_goal 与 native 三审计分流仍归 A5；本批没有制造假 qualified C 或 GPU 慢实现。

## 4. 持久安装与故障恢复

| 状态/故障点 | 实际行为及证据 |
| --- | --- |
| Seed intent 后、安装前 | 先保存准确 source/native/context/e/HEAD/G 绑定，再调用原 installer；重启从同一 seed 继续 |
| 原 installer 中途 | 未 admitted 的 preparing 可完整重装，不使用半装源码；测试人为留下截断 kernel 后中断，恢复首调仍只收到完整 C0 |
| Ready 后 | 重新核对同一 seed/context，完成原 draft 注册；episode admission 的暂停保持 preparing |
| Draft admitted 后但 native phase 尚 preparing | A4 先落 admitted，native exploring 随后落盘；恢复识别已有 admitted，不重装 seed；用户新增源码修改保留 |
| Exploring 中断/暂停 | 原 EpisodeWorkspace/draft 继续；不重装 seed。控制 worktree 漂移若会触发原自动覆盖，则在注册前失败 |
| 删除 kernel/改 mode/增加源码/混入 metadata/缺 draft state | 收费/重装前失败，保留原 draft 和最近完整 C |
| 一次 A3 结果出版/安装前中后 | 固定原 operation、intent digest、publication ID、target C。完整 before/after 状态可重放同一安装；未知 partial 写入或用户改动明确 fence，保留诊断和最近完整 C |
| Pending result 尚未完成 | 拒绝下一 invocation，要求先对原结果安装做 reconciliation；不会借重启重新授权模型 |
| Native outcome 已 Git commit，返回丢失 | 走原 native post-commit 恢复，准确捕获持久 draft C，不再调用 coder/V0、不打开 e+1 |
| C 捕获 intent/已发布/terminal save 后丢失 | 重放原 publication ID；终态先保存 C/latest，再通知 owner；失去 owner 返回不产生新 episode |
| 已绑定 V0 record 删除/变化 | 在 native setup 前拒绝，测试没有第二次 V0 evaluation |
| 未知 V0 job | 外部 owner 可持续 deny prepare，原 locator 留存。A4 不把未知 job 当作零费用或可重提 |

安装使用原 `CandidateWorkspace.install_source`，随后 fsync 实际 kernel/目录并使用 A2 完整校验复读。源码格式仍只有 Linux single_file/UTF-8/kernel.py/0644；旧辅助源码拒绝，不用 overlay 凑成新 C。模式、size、kernel byte hash、public manifest hash、native source identity 分开保存。CRLF/Unicode/无尾 LF 实际往返通过。

`current_source(e)` 返回该 e 最近完整稳定 C，在 damaged draft fence 后仍可读；不声称它等于损坏草稿。无 e 的 `current_source()` 是最新终态 C，首终态前为 C0。A4 不复制预算账本，也不把异常回滚冒称机器断电原子性：尚未 admitted 的 seed 可重装，已 admitted 的未知写入必须保留并 fence。

## 5. A5 的具体结果/费用接点

`install_result(e, result, authority=...)` 要求实际 A3 `NativeRoleResult` 和原 `NativeCallAuthority`，核对原 run、reservation、operation intent、e、coder、session、assignment、profile 和输入 C。调用实际 `result.native_source()`、`result.freeze_source(publication, candidate_workspace)`，向原 draft 安装完整源码并捕获 SourceBinding。重复已完成 operation 只返回原 binding，不覆盖后续用户修改。模型只提供逻辑 kernel 字节，不提供宿主目的路径。

A5 后续仍需做：

- 对真实 Agent、V0/Evaluation、reviewer、retry、各 native audit，在同一个已有 KS `OwnerAccount` 中持久 reserve/authorize；A4 的 lifecycle admit 只是 coarse prelude boundary。
- 在调用前保存原 operation_id/reservation、e、输入 C、角色/purpose、native session/job locator；原 `NativeCallAuthority.record_locator` 已为模型路径提供具体连接。
- 保留并吸收 `NativeRoleResult.invocation` 的真实 stdout/stderr/usage/events/unknown，即使 output 或安装失败也不能漏账。A4 不消费或编造费用收据。
- 用返回的 public C、native identity、kernel hash/size/mode 接原 `seal_candidate`、A2 Evaluation、fidelity 和三 audit 的准确 subject；原 ABBA/audit 仍是独立真实回调，factory 替换不覆盖它们。
- 为 pause/fence、active clock、finishing floor、未知远端工作、原 receipt 去重和最终 `_finish` 负责。恢复不自动 release reservation，也不分配新账本。

实际 A3 测试使用 `OwnerAccount.for_context`、原 reserve、`NativeCallAuthority.bind` 和 `RoleSession.from_authority`。只有被测试的模型进程返回和费用业务是 synthetic；原 Agent/NativeSession/authority/result/installer 是真实代码。该调用保留 9 个实际 synthetic token 计量值；A4 未再次 reserve/authorize/absorb。

## 6. never / helper / 受限 V1 决定

本 profile 固定 `framework_baseline=never`、Triton、无 native stack conversion；`always`/`auto` 明确 unsupported，没有回退成原生任意自由优化。这里不宣称已完成受限 V1。

| 职责 | 本批结论 |
| --- | --- |
| V0 Task/baseline/evaluator | 保留原入口与 baseline 字节；实际 A2 CPU 仍不可给 native PASS |
| native 格式/measurement metadata | 保留原 helper；它们可能前进 HEAD，e 使用更新后的 ancestry |
| solution metadata | 实际 direct Task evaluator 不依赖 solution.json；本 fixture 合法 absent，记录 `present=false, identity=null, required=false`。若外层另有 metadata helper，它必须在准备 e 前落 native record，再读新 HEAD |
| V1 baseline pin | never 明确无 pin，不伪造 speedup baseline；需要 pin 的 profile 要另行实现/验证受限 V1 |
| V0 reviewer prefetch | 原并行 scheduler 仍执行已配置 reviewer。测试观察到真实 scheduler 调 synthetic reviewer；实际费用、缓存重试与 reviewer context gate 交 A5 |
| production/source/dependency | 原 `_finish` 检查/导出 G。交付 C 必须由 A5 在准确 C staging 上单独走真实检查；本批不把 G 的合规当成 C 的资格 |
| 更慢但合格 C | 已保留未晋升 C 的接口；完整慢合格 completion/goal/native promotion 分流待 A5 的实证 |

## 7. 验证命令与实际结果

所有 AKA 测试均从 AKA cwd 使用 `PYTHONPATH=src` 和 `/tmp/schema-b05-a3-venv/bin/python`，因此执行当前工作树 native 窄补丁，而非旧 installed Light 文件。

| 命令 | 结果 |
| --- | --- |
| `python -m pytest -q tests/test_schema_seed_lifecycle.py -x`（新增最后 8 项前） | 26 passed，100.23 s |
| `python -m pytest -q tests/test_schema_seed_lifecycle.py -k 'actual_a3 or terminal' -x`（结果 fault 4 项、终态 fault 4 项加入后） | 9 passed / 25 deselected，47.58 s；含 1 项既有 normal-result 重叠，共覆盖当时 34 个不同 A4 case |
| `python -m pytest --collect-only -q tests/test_schema_seed_lifecycle.py`（最后 Task/scope 拒绝 case 加入前） | 34 tests collected |
| `python -m pytest -q tests/test_schema_episode_bridge.py tests/test_schema_off.py` | 18 passed，35.61 s；原 default/off 禁 optional import、KS checkout I/O 和网络的 native probe 通过 |
| `python -m pytest -q tests/test_schema_task_adapter.py::test_exact_source_bytes_native_identity tests/test_schema_task_adapter.py::test_bad_seed_missing_multifile_mode_and_encoding tests/test_schema_task_adapter.py::test_public_source_extra_file_and_expected_C_rejected 'tests/test_schema_task_adapter.py::test_full_task_through_real_aka_interfaces[None-None]'` | 6 passed，32.93 s |
| `git diff --check`；`git diff --cached --name-only` | 无 whitespace 错误；无已暂存内容 |

A2 实际 CPU regression 使用原 LocalExecutor/controller、72 basic calls、两次 state reuse 和 80 timing batches；其 native gate 仍为 `UNAVAILABLE`，command projection `all_pass=false`、`comparison=null`。这与 A4 synthetic V0 fixture 分开，未修改生产投影。

共享 pytest 临时目录会被并行任务的 retention 清理，因此另将两种 C0→e2 与一次实际 A3 安装代表路径重跑到固定 `/tmp/schema-b06-a4-retained`，完整输出日志保留 `/tmp/schema-b06-a4-retained-tests.log`；这次只补留证据，不用于扩大资格结论。

该保留证据命令为 `python -m pytest -q --basetemp=/tmp/schema-b06-a4-retained tests/test_schema_seed_lifecycle.py::test_actual_v0_then_exact_c0_and_unpromoted_c1_to_e2 'tests/test_schema_seed_lifecycle.py::test_actual_a3_result_installation_once_and_full_replacement[None]'`，结果 **3 passed / 22.21 s**。索引 `/tmp/schema-b06-a4-evidence.json` 保留三例原 state 路径、Q/run、V0/C0/C1/C2/G/HEAD、result 原 operation 与产品文件 SHA-256；生成索引时再次用 KS Bundle 验证保留 Source bytes/hash/size，并断言两种 C0 的 C1=e2 seed、e1 terminal HEAD=e2 base、C1≠G。

交付审查又增加 native `EvaluationInput` 中实际 Task/scope 必须等于原 Q 的检查（复用 A2 `evaluation_bundle`），防止相同 baseline 遮住 evaluator 绑定变化。新增第 35 项 `test_native_evaluator_binding_cannot_change_under_the_same_seed`。最终限定回归命令为 `python -m pytest -q --basetemp=/tmp/schema-b06-a4-final-check tests/test_schema_seed_lifecycle.py::test_native_evaluator_binding_cannot_change_under_the_same_seed tests/test_schema_seed_lifecycle.py::test_actual_v0_then_exact_c0_and_unpromoted_c1_to_e2 'tests/test_schema_seed_lifecycle.py::test_actual_a3_result_installation_once_and_full_replacement[None]'`，**4 passed / 24.89 s**，日志 `/tmp/schema-b06-a4-final-check.log`。上述 evidence index 已更新为本次最终代码下的三例正向原状态和当前产品字节 hash；篡改 evaluator 的负例只验证先拒绝、无 V0/coder 调用。

## 8. 未验证/交接边界

没有真实模型/GPU/Agate、完整 A5 backend/planner loop、跨机器恢复、并发 writer、完整 restricted V1、production C package、native ABBA 或三审计资格。默认 A3 `--schema` 的 missing-backend 行为未改为偷偷运行普通优化。A4 是上述真实 native 接点上的源码生命周期实现；后续 A5 必须接实际收费/资格/最终结果路径，不能仅凭这些 CPU 通过宣布整条工作流可生产使用。
