# Sparse 续接候选实现与交接

> 后续状态：用户已在 [USER_DECISION.md](USER_DECISION.md) 授权仅为工具故障保留进度完成，再由用户决定额外审计。以下记录保留讨论期方案与验证历史；其中“等待讨论”和前置P3/P5维护已经被新决定取代。实际已换绑并正常续P6，当前交接见 [SPARSE_CONTINUATION_STATUS.md](SPARSE_CONTINUATION_STATUS.md)。

2026-11-19。**当前是已完成离线定向验证、等待讨论及交叉审查的候选实现。真实旧 run 没有 prepare/apply、没有恢复或新增模型调用；不把实现准备视为产品政策已经通过。**

用户最新要求先讨论本轮 issues，续接必须讨论怎么做。主代理据此设置 live 应用边界；此前授权的 Sparse 持续监管职责仍由 `sparse_progress_recovery_research_xhigh` 承担，讨论后再接着执行。issue/TRIAGE/RUNS 由主代理单写，本代理没有改它们。

## 实际范围

采用技术报告 B 的窄实现，没有恢复旧语义 revision、跨 run import、来源变更或完整迁移系统：

- 新增 `repair-runtime prepare/apply/probe`。prepare 冻结显式代码树并封存 before state、差异、历史 attempt 身份和目标执行器的只读 preflight；无调用权限。apply 必须在目标冻结 Python 中执行，接受绑定该完整差异的兼容性审查记录，封存 binding 后才原子提交 state 指针。probe 只读。
- 原 `inputs.lock.json`、source、schema_docs、Template、prompt 全部保持原字节。独立 executor capsule 只有 `rules/` 和自己的 `executor.lock.json`；仅允许 `schema_extract` Python 变化，所有非执行语义文件必须与原快照逐字节一致。文件类型限制不能代替对 Python 行为变化的审查。
- 保持当前 `schema_extract.run.v2`，使用旧执行器明确拒绝的 `execution_binding/execution_bindings` 作为能力边界。`state.executor` 是当前真实执行器，原执行器封存在 before/binding；原输入指纹继续表示原输入，不伪称它已涵盖新代码。
- 历史 attempt 由封存清单解析其原执行器，新 attempt/receipt/validation/approval 带 binding；新 checkpoint 记其创建时 binding。旧 receipt、checkpoint、失败 response 均不重写。普通 origin/effective approval 与最终检查仍生效。
- 新 binding 放进会话 operation/group，补丁后的调用使用新的私有会话目录；原 UUID/session 文件保持历史原貌，无需在 apply 时修改它们或伪造退出记录。
- 保留原合法 P6 revision_scope，不通过清空 scope 避开 bug。纳入现有 `new_files` inventory 修复。已接受 P6 候选按真实 producer/receipt/checkpoint 校验后，只把 packed 的 draft 从 `71ec…` 定位到 `66ad…`，不改它的旧 approval/approved_sha256。以后 P6 worker 保存候选时也同步实际改动节点的 draft，避免再次遗失新稿。
- apply 拒绝未结清调用、活动/未知 Runner、控制/维护/恢复屏障、已交付或已 ready 的 run；不会解除 pause/cancel/hold，不启动模型。复用当前 writer 锁、state 原子提交、executor_digest 启动围栏。原 calls、revisions、单次决定、P6 starts 和所有47次历史调用保持。
- 原状态与封存 before 必须精确匹配；控制/启动依据在提交前再次比较。封存后未提交的记录没有执行权；提交后异常由同一 plan/review 幂等确认。不同 review 不能覆盖已绑定记录。

当前代码的兼容性记录要求 `plan_id / decision=compatible_executor_only / reviewer / summary / checked_files`，是供讨论的候选协议；尚未替真实旧 run 生成这样的批准。

## 可审查材料

- [完整目标补丁](EXECUTOR_PATCH_DIFF.patch)：从真实旧冻结 rules 树到候选执行器的精确差异。
- [逐文件 manifest](EXECUTOR_PATCH_MANIFEST.json)：11个 Python 文件，原执行器/输入/checkpoint，以及明确排除的当前其他工具改动。
- [真实旧候选的只读 probe](EXECUTOR_PATCH_READONLY_PROBE.json)：目标代码检查四节点，保留 scope，定位 packed 最新稿；记录真实 state 前后字节摘要。
- 临时完整目标树：`/tmp/sparse-executor-candidate-i1pc4pp_`。可用补丁从原冻结 rules 副本重建，不能把这个可变临时路径当成正式冻结身份。
- 本次编辑前的工具基线：`/tmp/sparse-executor-patch-baseline-zelhmrbn`。目标通过“本次基线→当前代码”的精确补丁应用到原 frozen 树，避免暗带已有未提交修复。

目标涉及新 `executor_repairs.py`，以及 runtime/snapshots/store/stages/receipts/runner/cli/audit/publication 和既有 artifacts 修复。没有把当前 maintenance/reading/wakeup 变化、receipt evidence repair、相关 prompt/Template 变化全部带入；目标 manifest 列出当前与目标仍不同的 Python 文件。当前其他算子仍使用各自原冻结执行器。

## 已完成验证

```text
python -B -m unittest -f \
  schema_extract.tools.tests.test_executor_repairs.ExecutorRepairTests \
  schema_extract.tools.tests.test_artifact_transactions.ArtifactTransactions.test_p6_retained_scope_covers_unchanged_artifacts \
  schema_extract.tools.tests.test_artifact_transactions.ArtifactTransactions.test_p6_changes_are_limited_to_the_reviewed_fields

Ran 4 tests in 92.715s — OK
```

两项新增组合场景使用临时 Store、真实 Template 检查和本地 deterministic backend，没有调用真实模型：

1. 原 P6 已接受候选、故意模拟旧 draft 指针 → prepare → 在冻结目标代码中分别模拟 state 提交前/后的故障 → 幂等 apply。保留原 stage/approval/scope/计数/候选和历史会话，旧 attempt 仍按原身份核验，未换代码的进程不能假冒目标执行器。
2. 接着实际执行 P3 patch+verify、分支 readonly review；原 P1/P2/P4/无关分支批准保留，分支复审输入字节等于接续候选；再追加一次 P6 reviewer revise→worker→reviewer，最终通过正常 publication。只有一个 binding，说明后续同类局部修改无需再次维修；交付凭据记录真实执行器历史。
3. 语义材料变化、暂停控制拒绝；原有 gate 回归继续确认未知文件、root/metadata 越权拒绝。

真实旧 Sparse 只做目标代码的 readonly probe：checkpoint `66ad31f3…`，`checked_nodes=[root,split_kv,tcgen05,tcgen05_packed]`、`diagnostics=[]`，packed 的候选 producer 为已接受 worker `d46bb723…`、SHA-256 `fb9427b6…`。机械成功不批准其中的 root 寄存器保证，P3 限域修订仍然必要。

首次手动 probe 从 `/tmp` 使用了未配置 PyYAML 的系统 Python，已改用仓库既有 `/opt/venv/python3.13.12/bin/python` 成功检查；未安装或修改环境。第一次组合测试达到 P6 ready 时碰到测试自己的60秒期限，随后复用同调用内的绑定校验结果，并给离线组合场景合理期限；最终上列场景全部通过。测试里的文本换行使字面子串断言不稳，最终采用候选/复审原稿的实际字节摘要验证保稿。

## 讨论边界和下一步

需要讨论的是产品/责任规则，而不是让用户批准每个函数或测试：同 run 是否允许追加执行器身份；哪些 Python 差异能归类为纯工具修复及由谁审查；同语义输入下，原批准通过来源真实性与当前适用性校验后如何保留。当前候选采用保守的“显式冻结+完整差异审查+原证据身份+正常内容复审”方向，但不把代码存在当作规则已经定案。

本次已验证范围是当前格式、同 source/config/语义规则、未交付、调用结清、满足 preflight 的执行器修复接续。来源/任务/语义规则变化、已发布产物修订、跨 run 批准承接、允许在活动调用或暂停状态下换绑等不在这次实现与验证范围。即使 Sparse 以后交付，也不能据此把广义 `progress-preserving-continuation` 直接标 resolved。

讨论及主代理交叉审查通过后，由本代理继续：

1. 必要调整候选与接口说明，形成真实旧 run 的公开 prepare 及具体差异/证据，供独立兼容性代码审查；主代理处理既有 monitor 的代码刷新。
2. 通过公开 apply 换绑及保存最新 packed 待审稿。原失败 approve 不采纳、不重放。
3. 原 Qwen/max 后端通过公开 maintain 执行 root P3 最小 patch+独立 verify，以及 packed readonly P5 review；新正常 P6与正式交付。顺利路径估计4次新增调用，不预先强制审核结论。
4. 核对四份 YAML、dispatch、最终 checkpoint/P6/交付回执字节一致，Template read-only 检查、调用结清与来源；审计 partial 项如实说明。

**当前监督交接点**：旧 run `8f0844f9-f279-4781-a19d-0216b7fc4ef5`，P6 reviewer / invalid_receipt，checkpoint `66ad31f3…`，47次历史调用，无新 binding、无新 attempt、无在途调用。等待续接方案讨论与交叉审查；不要恢复已取消 successor `a362b85a…`，不要回 P1。没有提交、推送、GPU、monitor 或其他算子控制操作。
