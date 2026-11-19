# Sparse P6 approve 的 scope 误拒收

> 后续更新：本文保留最初调查和当时的接口边界。末尾曾提到的另起run恢复方向已由 [保进度复盘](recovery_reassessment/RECOMMENDATION.md)替代，新Sparse调用已取消。2026-11-19 06:05 UTC，FMHA出现同根P6误拒；临时副本的原gate失败、修复gate通过证据见 [FMHA_P6_SCOPE_REPRO.json](FMHA_P6_SCOPE_REPRO.json)。两项真实run均尚未采用新执行器。

2026-11-19。run `8f0844f9-f279-4781-a19d-0216b7fc4ef5`，attempt `aa4176a8-3767-4ee4-b045-cb95b15c06fa`，固定执行器 `b8a1d5e368bf0e423081110ffc748945097a503b7eb2fd11fac6355650a58d8a`。这是工具校验缺陷，与此前显式声明 `META_DATA.Version` 的回执表达问题不同。当前源码已最小修复；旧 run 仍阻塞，未重交、未批准、未修改冻结文件或运行状态。

## 实际拒收点

真实 `response.json` 是 `decision=approve`、`revision_scope=[]`、`affected_nodes=[tcgen05_packed]`。`result.json` 是 `outcome=completed`、`exit_code=0`、`session_status=ready`，原生唯一 success，无 native issues；`event.json.error=null`，`processing.json` 已结束，`rejected.json` 记录 `proposal_rejected: P6 revision scope requires unique selected files and explicit fields`。退出记录确认进程终止且 survivors 为空；state 的 active/unsettled/unconfirmed 三类在途为空。

state 仍合法保留原 reviewer 的两条 worker 修改授权：

- `dispatch.md`：`['*']`。
- `tcgen05_packed.yaml`：两条 `DESIGN.Tile_pipeline.Step.MAIN.Role.{SOFTMAX,LOADER}.Info.Using_verb.store.zero_rows.describe`，以及 `REALIZATION.Prefer`。

实际 selected 集合为 `root/split_kv/tcgen05/tcgen05_packed`，当前 checkpoint 为 `66ad31f314d7ca7e894527d9641bbf3daad59f4ba35278e71e7841dd5e3901da`。reviewer workspace 与该 checkpoint 的完整五文件 manifest 相同。

冻结源码的实际调用链是：

1. `runner.accept_result` 首先对普通 reviewer 调用 `_artifact_gate(..., decision='continue')`，早于 `normalize_response` 和 `receipt.json` 写出。
2. `runner._artifact_gate`（冻结 `runner.py:196`）把 **state 保留的原授权** 交给 `validate_artifacts`。
3. `artifacts.validate_artifacts`（冻结 `artifacts.py:640`）只解析 `root`、改动过的 YAML 和当前 node。reviewer 无写入，当前 node 是 root，因此 `new` 只有 root。
4. `artifacts._validate_scope`（冻结 `artifacts.py:597`）错误地用这个局部 `new` 集合构造 selected，再检查原授权；`tcgen05_packed.yaml` 因不在局部集合，被 `validate_revision_scope`（冻结 `artifacts.py:486`）误拒。

因此拒收不是空 `revision_scope` 非法，也不是原两条授权非法。回执内容尚未进入正式回执校验；该 attempt 没有 `receipt.json` 或正式批准是符合此调用顺序的结果。

## 最小修复和验证

[当前 artifacts.py](../../schema_extract/tools/artifacts.py) 只改 `_validate_scope` 的 P6 inventory 来源：由局部已解析文档改为已经过文件清单门禁的完整 `new_files` YAML 集合。`validate_revision_scope`、`validate_artifacts`、文件和字段写权限、Version 责任、reviewer/root 只读、独立审核及 `_final_inputs` 最终 selected/manifest 门禁均未改。上述关键原有函数与本批 frozen 的 AST 仍一致。

[回归测试](../../schema_extract/tools/tests/test_artifact_transactions.py) 新增一个方法，覆盖非 root scope 保留时 reviewer 无写入、worker 仅更新目录，以及 reviewer 写入、未知文件、root scope、受保护 metadata 仍被拒绝。连同已有 P6 字段越权测试，共两个定向测试通过：

```text
python -B -m unittest \
  schema_extract.tools.tests.test_artifact_transactions.ArtifactTransactions.test_p6_retained_scope_covers_unchanged_artifacts \
  schema_extract.tools.tests.test_artifact_transactions.ArtifactTransactions.test_p6_changes_are_limited_to_the_reviewed_fields
Ran 2 tests — OK
```

还在临时目录复制真实 checkpoint 和 reviewer workspace，仅调用无写入的 gate/校验函数，未调用 `accept_result`：

```text
frozen module: snapshots/e5db05dc281797bc4a7350cc1d7eb35b2d1331efb10f304bd0ab3b1955ff6388/rules/schema_extract/tools/artifacts.py
receipt empty scope against actual selected: valid
retained state scope against actual selected: valid
before/after manifest equal: true
frozen _artifact_gate scope paths: [dispatch.md, tcgen05_packed.yaml]
frozen _validate_scope selected ids: [root]
frozen result: exact recorded ExtractError
current fixed _artifact_gate: changed_files=[], exit_code=0
in-memory normalize_response → validate_receipt: approve, revision_scope=[], affected_nodes=[tcgen05_packed]
fixed-material strict P6 snapshot: exit_code=0, four checked nodes, diagnostics=[]
same inventory with one selected node omitted: rejected
```

完整回执校验只在深拷贝 state 中恢复该 attempt 的模拟 active fence，以重现提交时前提；没有持久化模拟 state、receipt 或 approval。此机械验证不等于正式采纳，也不替代来源真实性审阅。

本次源码与测试差异通过 `git diff --check`。收尾只读复核：live 仍为原 `invalid_receipt`，P6 calls=5、approval=null，checkpoint 不变，三类在途仍为空，原 attempt 仍无 `receipt.json`。

## 旧 run 的公开恢复边界

已只读解析并调用 `frozen_run_invocation` 生成命令，确认公开 whole-run `resume` 经 `cli._frozen_entry` 路由至上述 snapshot 的 `schema_extract/extract_yaml.py`，不会采用当前源码修复。冻结 `_resume_failed_scopes` 在深拷贝上只将 stop 恢复为 running/reviewer、补反馈；原 revision_scope、checkpoint 和 calls 全部不变。因此下一次 reviewer 无论返回 approve、revise 或 reopen，都会先经过同一个错误 gate。未启动一次已知无效的模型重交，未增加 RECOVERIES 执行记录。

真实 P3 问题可以按其本身必要性走 `maintain` 的 review/patch/verify 和后续正常复核，但这条路径也不解除本次工具阻塞：冻结 `maintenance.mark_drained → stages.invalidate_final` 不重建 revision_scope；`maintenance._resume_normal → stages.resume_progress → stages.settle` 重返 P6 时同样保留它。对整个冻结 `schema_extract/tools` 的赋值点做 AST 检查，run 初始化后的唯一 state revision_scope 替换是 `stages.advance` 的 **已接受 P6 reviewer revise** 分支（冻结 `stages.py:625`）；而该 reviewer 当前到不了此分支。不能为绕过故障制造维护请求。

现有公开接口没有替换旧 run 冻结执行器或迁移批准的入口；直接用当前 `accept_result` 也违反固定执行器校验。严格保持当前约束时，旧 run 须留在现有阻塞点。可讨论的后续选项是用修复版本新建独立 run 并完成真实审核；若必须保留旧 run 身份，则需要另行设计和审查显式执行器迁移协议，现有代码不支持，不能冒充现成恢复。此处未启动新 run、未 fork/reset、未热改快照/状态/产物/回执/批准，也未提交或推送。

## 来源风险与正式问题记录

真实 reviewer summary 明确承认：root 的 `~208 registers/thread` 在冻结材料中无对应记录，却因 root 只读而留待后续修订。当前 checkpoint 的 `root.yaml:218` 写 `Costs ~208 registers/thread`，`:341` 写 `~208 registers per thread without spilling`。这是独立的产物来源风险，不能因为 summary 自称 approve 或机械检查通过就忽略。已交主代理，由正常来源核查和责任阶段审核/维护处理；本次未改 root，也未扩大语义审计。

正式工具问题建议稳定 ID：`p6-retained-scope-inventory`，category `extract_tools`。根因是把局部解析集合当作完整 P6 inventory；最小修复、冻结函数位置、真实无写入复现和旧 run 恢复限制如上，供主代理登记台账。源码缺陷修复与旧运行恢复须分别记账，不能将 current-source resolved 写成该 run 已恢复。
