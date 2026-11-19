# Sparse 保留进度接续：技术复核

2026-11-19。仅调查和临时副本验证；未实施恢复、修改源码/原稿/状态/冻结文件、运行模型、提交或推送。

**建议保留旧 run，从现有五文件 checkpoint 接续；推荐候选 B：追加有记录的小范围冻结执行器补丁绑定，保留原合法 scope 和历史批准，再复用 P3/P5 maintenance 和正常 P6。** 不需要回 P1/P2，不需要重写四份设计，也不需要先建设通用迁移系统。候选 A“公开关闭已完成 P6 写窗口”代码更少，但它只是有记录地避开旧 gate 的触发条件，下一轮同类 P6 修改仍会复发，故只作备用。

这两个选项均是待实现的工具协议，不是建议人工修改 `state.json`。此前“现成公开接口没有入口”的结论成立，但由此推导“内容必须重新提取”不成立。

## 1. 已核实的进度和精确断点

| 对象 | 当前身份/事实 |
| --- | --- |
| 旧 run | `8f0844f9-f279-4781-a19d-0216b7fc4ef5` |
| 固定输入 | `e5db05dc281797bc4a7350cc1d7eb35b2d1331efb10f304bd0ab3b1955ff6388` |
| 原执行器 | `b8a1d5e368bf0e423081110ffc748945097a503b7eb2fd11fac6355650a58d8a`，2.5.1 |
| 必须保留的最新候选 | `66ad31f314d7ca7e894527d9641bbf3daad59f4ba35278e71e7841dd5e3901da`，包含 root/split_kv/tcgen05/tcgen05_packed/dispatch |
| 已合并设计基线 | `71ec255351ca0a11d68a38db310dfa981ba5f04967fe0b97fa7b13dbfd7d0425`；比上述候选早，不能拿它代替最新候选 |
| 产生最新候选的真实 worker | `d46bb723-7666-4046-b6d9-049dabca272d`，已接受的 P6 `complete`，修改 dispatch 和 packed |
| 失败 reviewer | `aa4176a8-3767-4ee4-b045-cb95b15c06fa`，P6 response 为 approve，但未正式接受、没有 receipt；保持这个事实 |
| 待复核 | `pending_rechecks=[tcgen05_packed]` |
| 用量 | 47 次历史调用；P3 calls=6、packed calls=5、P6 calls=5；每范围上限32，无追加调用记录 |

本次用原冻结模块实际验证了 P1/P2/P3/P4 及三个 P5 节点的 `verify_approval_origin`，全部通过。四份设计已有真实独立批准；packed 的**后来 P6 修改**仍是待审候选，不能把旧 P5 批准解释成已批准它的新字节。

现有程序已经支持保进度的大部分工作：

- `Store.verify_checkpoint` 校验不可变 checkpoint、run/input 身份和完整 manifest。
- `verify_approval_origin` 保留真实原批准；`verify_effective_approval`、`refresh_approvals`、`align_source_identity` 判断批准对当前内容是否仍有效，并记录 `retained_approvals/applicability`。
- `maintenance.mark_drained` 从 **`state.checkpoint`** 建立 `working_checkpoint`；P1–P4 的 `merge_upstream` 合回完整树。`review → patch → verify` 或 `patch → verify` 复用普通 receipt/checker/独立 reviewer；结束后 `resume_progress` 只返回真正未完成的责任。
- `invalidate_stage/invalidate_node` 保留旧批准和稿件、计数；`pending_rechecks` 与正常 P6 要求新的最终审核。
- 旧 Sparse 自己已用这些机制完成过 P1 输入登记维护，并保留后续进度。

当前缺口有两个，第二个此前未被充分考虑：

1. **原 gate 错误拒绝 `state.revision_scope` 中合法保留的 P6 写授权。** 这个 state 本身没有坏；错误是 gate 用局部解析 `root` 当全 selected 集合。普通 resume 只会再撞相同 gate。详见 [P6_APPROVE_SCOPE](../P6_APPROVE_SCOPE.md)。
2. **P6 接受的新稿没有同步到 packed 的节点 draft 指针。** `state.checkpoint=66ad…`，但 `nodes.tcgen05_packed.draft_checkpoint=71ec…`。`Store._node_draft` 优先旧节点 draft，`start_node` 会把它盖到最新 merged 阅读材料上。若 P3 维护后 packed 被判需复审，必须先保证它拿到 P6 已完成修改后的原稿，而不是悄悄退回较早 P5 稿。

当前源码与冻结版本之间不止 artifacts 一个文件不同，另有 maintenance/reading/receipts/runner/wakeup 的改动。不能把“直接采用当前整套目录”称为仅接入这次 gate 修复。本次核对的 `mark_drained/prepare_workspace/merge_upstream/advance`、`_node_draft/start_node/refresh_approvals`、`accept_result` 和批准投影/等价比较函数的 AST 在两者中相同；上述接续分析适用于真实冻结 run。

## 2. 备用候选 A：公开、受控地关闭已完成 P6 写窗口

这是一项**新增的工作流转换能力**：只有已接受 P6 worker 完成、候选未再变化、接下来是只读 reviewer 时，才能结束旧写授权并保存该候选作为后续复审原稿。它不接收旧失败 approve、不授予任何内容批准、不扩大下一次写权限。原 run 继续使用原冻结执行器；当前源码的 inventory 修复继续用于今后的新冻结执行器。

必须如实说明：现有协议故意保留原 scope，并没有“worker complete 后必须清空 scope”的不变量；因此不能把清空说成恢复损坏状态。可以另行设计“关闭写窗口”这个有用的合法动作，但若专门为本 bug 引入并反复调用，就是受控规避，不是修复旧 gate。它虽然安全可设计，却不如 B 符合用户“修工具后续接”的要求。

### 接口及最小代码面

建议新增一个专用入口，例如：

```text
repair-p6-handoff RUN --issue-id p6-retained-scope-inventory
  --checkpoint CHECKPOINT --worker-attempt WORKER --rejected-attempt REVIEWER
```

实际可拆成 prepare/apply，便于代码审查与本地验证；这是工具事务步骤，不要求用户逐步批准。不能提供任意 JSON patch、任意 state 字段赋值、跳过 checker 或接受旧 receipt 的参数。

- 新增一个专用模块，负责前提核验、定向机械重校验、封存修复记录、幂等事务。
- `cli.py` 在普通 `_frozen_entry` 前分派这个**明确的恢复控制入口**；普通 run/resume/maintain 继续原冻结路由。该入口是协议新增的单写者维修职责，不是关闭普通 `executor=True` 检查。
- 复用 `Store.writer`、短 `control_transaction`、`Store.save` 和已有封存/fsync 工具。新增只读 audit 分支，导出/核验修复链；无需改正常 receipt schema、P1–P6 调度、批准算法或 source/rules 身份。
- 对本例的 draft 指针保存使用已有字段；增加一个专用追加历史指针（例如 `tool_repairs`）引用封存记录。旧冻结执行器目前不会因这种额外历史键拒绝状态，但它也不会校验此新历史；**新公开 audit 必须校验，文档必须明确这个兼容边界**。

### 必须逐项核验的前提

1. run 未发布、无 export、无发布 rename commit；没有活动维护/维护请求。持有同一个 run writer 锁；三类在途集合为空，还要核实 supervisor 启动项、runner 身份和旧进程终止，不能只看空列表。pause/cancel/hold 不得被该操作解除，也不启动模型。
2. `verify_execution(executor=False)` 和 `verify_run_inputs` 验原锁、源码、规则、原执行器身份；核对输入/config/selected/plan_signature 与参数。这里只验证旧固定身份，不谎称恢复控制程序就是旧执行器。
3. 当前确为普通 P6 reviewer 停点，失败 attempt 未被接受，拒收点精确匹配本 issue；它的 baseline、完整 workspace manifest 等于登记候选，原生结果/退出/终止均可核实。
4. 参数 worker 是该候选 checkpoint 的真实 producer；其原 attempt/receipt/validated/check、accepted_receipts 对应，decision=complete，实际产物摘要就是当前候选。原 reviewer revise 授权、worker 使用授权和现有 `revision_scope` 一致，且 worker 修改只在其范围内。不能凭 checkpoint 文件存在推断已接受。
5. 以原材料重现原拒收，再以修复后的只读校验器验证原授权、完整 selected inventory、实际 worker diff、最终候选结构。保留机械验证为新的 `postcheck` 证据，不能写成模型批准或回填旧 attempt。
6. 确认所有待保留批准的 origin 真实；明确记录当前候选相对已批准基线的差异，尤其 packed 尚待复审。关闭的是**已经消费完的写授权**，不是把候选当成已批准版本。

### 允许的精确状态变化

| 字段 | 本例动作 | 含义 |
| --- | --- | --- |
| `revision_scope` | 原列表 → `[]` | 结束已完成 worker 的写授权；后续 reviewer 若 revise，必须重新形成真实授权 |
| `nodes.tcgen05_packed.last_draft_checkpoint` | 记录其原 draft 指针 `71ec…` | 保留前稿定位 |
| `nodes.tcgen05_packed.draft_checkpoint` | → `66ad…` | 将已接受 P6 worker 稿登记为待复审原稿，不改变 approval/status/approved_sha256 |
| `tool_repairs` | 追加封存记录 ID | 有来源的维修历史；不能覆盖前记录 |
| 状态落盘字段 | `state_revision/updated_at` 正常更新 | 使用已有原子提交 |

一般化时只能推进“已接受 P6 worker 实际修改且需复核”的非 root 节点，逐节点证明候选身份；不得任意改选定集合或把整个 merged 指针推进到未批准候选。

`checkpoint`、`merged_checkpoint`、全部 approval/accepted_receipts、pending_rechecks、原47次 attempt/usage、source/config/executor/输入锁、plan_revision、calls/revisions/单次决定、P6 starts、控制状态均保持原值。status/reason 暂留原停点，接续由现有 maintain/resume 改变。

P3/P5 maintenance 的 task/step 自带独立新 session。最终 P6 必须是新的真实 attempt；为避免旧 approve 会话形成锚定，建议用现有 `handoff_session/retire_session` 的可审计方式退休旧 P6 reviewer 会话，记录真实退出证明、旧 UUID、checkpoint、原因和后继 attempt。若纳入本操作，应把 `sessions.P6.reviewer`、`session_handoffs` 和可变 session 控制文件列入明确事务清单，封存其 before 值；不改旧 response/receipt/event。本例真实退出证明在 `exit.json.termination`，不是不存在的 `termination.json`。

### 新旧身份和证据怎么保存

封存 `repairs/ID/`：before state、旧输入锁摘要、原执行器身份、实际修复控制程序/校验器的代码 manifest 与摘要、稳定 issue、完整允许字段差异、旧 scope、producer 和 rejected attempt 的证据摘要、checkpoint/manifest、进程终止证明、postcheck、原批准 origin 核验结果。记录验证程序的真实字节，禁止随运行变化的 current 路径冒充冻结身份。

明确区分三者：历史调用由 b8a1… 执行；恢复 postcheck 由登记的新校验器执行；后续模型调用仍由 b8a1… 执行。新 postcheck 只证明“这份合法 worker 候选满足修复后机械门禁及维修前提”，不证明旧失败 approve 已被接受。

记录先封存，state 指针后提交；在同一短控制事务重查 before state 和 launch/control 身份，再提交允许字段。`(run, issue, worker, checkpoint)` 幂等，同 ID 不同内容拒绝。已提交而控制回执未确认时按 state 指针补确认；封存但未被 state 引用的记录无执行权。会话退休如采用多文件步骤，须能从封存 before/after 与 state 指针重放，不回滚已用调用。

### 操作顺序

1. 实现并独立审查上述工具操作，做第5节定向验证；仅此后在真实旧 run 执行一次交接修复。**不要用脚本先行手改状态。**
2. 通过已有 `maintain` 一次提交两个明确任务：P3 root patch，以及 P5 packed review。P3 使用 [REGISTER_CLAIM_TRIAGE](../REGISTER_CLAIM_TRIAGE.md) 的字段和来源证据；P5 只审来自 `66ad…` 的最新 packed 及当前 root 前提。
3. P3 worker 小修，独立 verify 审 P3 全文；P1/P2/P4 批准保留，另两个分支由已有严格等价验证保留。P5 packed 用自己的最新原稿只读审核；如符合，形成其新批准，不先发重写 worker。
4. 正常 P6 重新审完整四设计+dispatch，形成新 receipt 和批准，按既有最终门禁/发布流程完成。失败 aa417… 永远保留为历史失败调用。

推荐任务形状如下。`REALIZATION.Require` 是列表，reason 必须将语义授权收窄到第三项 Choice/Reason；不能借字段上限改其他条目。

```json
[
  {
    "stage": "P3", "node": "root", "action": "patch",
    "fields": [{"path":"root.yaml","fields":["DESIGN.Tile_pipeline.Thesis","REALIZATION.Require"]}],
    "reason": "按REGISTER_CLAIM_TRIAGE所列固定源码证据，只删除实际总寄存器约208和without spilling保证；Require仅第三项Choice/Reason收窄为共享存储申请及device opt-in检查，其他设计/Storage/Contract/when/Source不变，Version按既有规则更新。独立verify仍审本阶段全文。"
  },
  {
    "stage": "P5", "node": "tcgen05_packed", "action": "review",
    "fields": [],
    "reason": "独立审阅由已接受P6 worker d46bb723-7666-4046-b6d9-049dabca272d形成的66ad31f3候选，保留其zero_rows与Prefer修订；核对当前root前提，原P5批准不覆盖这些新字节，旧P6失败approve不算本次批准。"
  }
]
```

理想成功路径需要 P3 patch+verify、packed review、P6 review 共 **4 次真实调用**。若选择 P3 reviewer-first，则多一次 review。现余额足够，不因此追加预算。`fields:[]` 的 packed review 发现真实问题会停止，由 Supervisor 按新证据处理；不能把它伪装为通过或自动开启无界修订。

方案 A 的限制：旧冻结 gate 代码仍有 bug；将来若 P6 再批准一轮非 root 修改，完成后的交接可能再次需要同类显式操作。这不影响本次接续真实性，但不能把该 run 报告为已经换成修复执行器。若不希望保留这个限制，采用下面的 B。

## 3. 推荐候选 B：同 run 显式冻结执行器补丁绑定

旧 run 后续调用应直接使用修复 gate，因此推荐恢复**仅纯执行器补丁**能力，不恢复整个语义 revision/跨 run import 系统。这保留合法 revision_scope 的原语义，不靠清授权消除报错。

[2.4.0 连续修订设计](../../../schema_extract/archive/designs/2.4.0-continuous-revision.md) 的原则正是“默认保留稿件和有效进度；gate变更定向重校验”。Git `059275d` 的真实 `runtime.py` 已实现过 `upgrade_runtime`、sealed before/report/review/target lock、单写者/quiescent、旧 attempt 身份解析和追加 binding。其代码还耦合旧语义格式和 budget，不能整段复制到2.5.1；可复用协议结构。历史 `continuation._inherit` 的全量 reset 到 P1 不适合这里，也不能因为它存在就把它当唯一继承方式。

最小必要改变：

1. `runtime/snapshots` 增加冻结补丁记录和活动执行器选择。以**原冻结规则树**为基础，明确加入 gate 修复和接续支持代码；语义文件、Template/checker、prompt、source/config 不变。实际新代码 manifest 全量列明，不能顺手引入当前其他五个模块的无关改动。
2. `Store.context` 继续返回原 `source_lock/rules_lock`，单独返回新 `executor_lock`。原 `inputs.lock.json/input_fingerprint/snapshot` 始终保留其原意。`rules.id` 当前混含执行器代码，不能用目标 rules.id 替换原输入指纹再宣称“输入没变”。
3. state 使用明确新能力/格式及 `execution_binding` 指针、追加历史；旧 frozen b8 必须拒绝新状态，新的 CLI/monitor 将 resume 路由到登记的新冻结执行器。`state.executor` 可表示活动执行器，原值必须封存在 binding 的 before state；不要只覆盖原字段而无历史。
4. `verify_execution/_attempt_identity` 按 attempt 所属 binding 验证，原 attempt 无 binding 只在封存的原 attempt 列表内解释为原执行器。新 attempt/receipt identity、validation、checkpoint 增加 binding ID；旧对象不回写、不重哈希。
5. `cli/monitor` 启动凭据与边界加入 binding，`audit/publication` 展示并核验新旧执行证据链。保留来源门禁和最终批准逻辑。
6. quiescent 单写者事务封存 before/目标代码/差异审阅/postcheck/旧证据身份，然后仅切换执行身份及必要会话；不清批准、计数或阶段。**同时登记第1节缺失的 packed 最新 draft 指针**，否则只换 gate 仍会在 P3 后回到旧稿。此登记同样只是候选保存，不是批准。

建议具体改动边界如下，不必把语义revision或跨run adoption一并复活：

| 模块 | 必要改动 |
| --- | --- |
| 新增窄 `executor_repairs.py`（也可收于runtime） | 只接受同source/config/语义规则的修复；冻结代码、验证差异/前提、封存before与binding、登记缺失候选指针 |
| `runtime.py` | 双版本状态准入；当前/历史binding解析；`_key`加入binding；保留旧attempt真实执行身份，验证当前实际Python字节 |
| `snapshots.py` | 冻结并核验独立执行器目标；不修改原输入锁；提供目标invocation |
| `store.py` | context分离；新attempt/checkpoint带binding；节点draft仍保本人最新稿；老格式对象照原身份校验 |
| `receipts.py/stages.py/runner.py` | 新call identity透传binding到receipt/validation/approval；不更改内容批准条件；P6候选按实际changed_files保存相应节点draft指针 |
| `cli.py/monitor.py/audit.py/publication.py` | 公开prepare/apply及新冻结路由/启动身份；审计封存链和最终执行来源；控制/最终门禁仍有效 |
| `artifacts.py` | 只纳入已定位的`new_files`完整inventory修复，其他字段权限和检查规则不放宽 |

多数现有模块只是追加binding字段或调用同一个resolver，不各建一套迁移/批准逻辑；主要新逻辑集中在封存事务和身份解析。更新README/协议中“执行器变化另建run”的说法为这个明确例外，登记新能力；没有binding的其他旧run继续原路线，不能因本次Sparse修复改变它们。

这里“独立执行器目标”可以使用封存的原规则树副本加明确代码补丁；非执行语义材料逐字节等于原快照。`verify_execution` 对原输入的返回契约不能被悄悄改成目标输入锁：旧调用位置仍要拿到原 source/rules，`frozen_run_invocation` 明确选择独立 executor context。新 binding 是执行变化记录，不能伪称原 `input_fingerprint` 已经涵盖新代码。

原子apply的状态差异应限定为：新format/capability、活动binding及追加历史、活动executor身份、必要session handoff、packed draft/last_draft候选定位、正常state_revision/updated_at；scope、checkpoint/merged_checkpoint、批准、计数和控制均保持。封存记录至少包括A列出的候选/producer/拒收证据，以及目标代码manifest、原/新执行器、全部原attempt身份摘要、兼容性独立审查与实际postcheck。旧批准的有效性仍由当前内容检查决定，不能只凭兼容标签接受。

**本次执行顺序是 B prepare/独立审查/定向测试 → B apply（不运行模型、不采纳aa417旧approve）→ 第2节所列P3和packed维护任务 → 新P6。** 第2节的scope关闭步骤不执行；其候选保存、证据和维护顺序由B事务承担。P6若再返回同类非root revise，新的gate应完整处理reviewer→worker→reviewer，不需要第二次维修命令。这是推荐B而非A的决定性稳定性差别。

P1/P2/P4 和未变分支批准可以作为原执行器的真实历史证据保留；机械重校验和精确语义等价证明其继续适用，不把它们伪造为新执行器签发。root 与 packed 仍走上面两项维护和新的 P6。aa417… 不重放采纳，因另有必须修正的 root 事实范围问题，其旧完整候选即使通过新 gate 也不能直接交付。

此方案改动面大于 A，但仍围绕一个 context/identity 入口、已有维护和检查链；无需重建任务图、重新选择设计或跨 run 批准迁移。只新增一个 `upgrade-runtime/repair-runtime` 类型公开操作即可，不必引入通用语义迁移产品。

## 4. 哪些批准可保留，以及为什么

本次在临时目录复制 `66ad…`，按 triage 仅收窄 root 两处声明和第三项 Reason，并递增 Version。用**原冻结 checker**做无模型验证，结果如下；这份临时文本不是正式修改产物。

| 范围 | 实测结果 | 后续责任 |
| --- | --- | --- |
| P1 Contract | subject projection 不变 | 保留原批准，不重走 P1 |
| P2 selection/Source/id | projection 不变 | 保留原批准，不重走 P2 |
| P3 root Design/Realization | projection 改变；strict check 成功 | 原 P3 批准保留为历史，由 P3 patch+独立 verify 给当前批准 |
| P4 selected/writing plan | projection 不变 | 保留原批准；最终 P6仍检查目录、覆盖和当前一致性 |
| split_kv | strict check成功；与旧批准 `align_source_identity.equivalent=true`；仅来源 root 原始字节摘要改变 | 现有 `retained_approvals/applicability` 记录新严格上下文，无新 P5模型调用 |
| tcgen05 | 同上 | 同上 |
| tcgen05_packed | strict check成功；equivalent=false，raw/Tile/use内容已被P6修改 | 从当前新稿做一个 P5只读review；不能仅借root等价或旧P5批准放行 |
| P6 | 无正式当前批准 | 新真实完整终审 |

三个非 root 的 Math/Work/Tile `origins` 均为本节点，没有 `from: root` 继承 root Tile。它们共享的是 root Contract，此次不变。208 是64+16+128个主数组32位槽的小计，收窄其描述不改变任务、代表路径、设计分支、Storage或计算。没有把此局部问题扩成 P1/P2 或全部 P5 重提取的依据。

packed增加一次P5复审是沿用当前维护合并机制的结果：P3批准把含packed新稿的完整working checkpoint合入后，`refresh_approvals`不能继续让旧P5批准覆盖packed新字节，必须暂挂它；`final_eligible`也要求节点当前批准齐备。理论上可新增“只合入root、另保留P6待审分支候选”的双frontier方案，令最终P6直接审packed，但那会扩大本次恢复实现面。复用现有一次readonly P5 review更小，也仍然保留全部设计进度。

## 5. 已做验证、实施后必要验证与边界

本调查实际完成：

- 原批准 origin：P1–P4及三个非root节点均通过原冻结 `Store.verify_approval_origin`。
- 原冻结 gate：仅在深拷贝 state 把已完成写授权置空，同一个实际候选/readonly workspace 得到 `changed_files=[]` 且 manifest 完全一致；未调用 `accept_result`、未伪造 active fence、未写 receipt/approval。
- 临时 root 修订的各阶段 projection、四节点 strict snapshot 与来源等价比较如上。
- 两个 checkpoint 的真实 producer、manifest、原 revision_scope、pending_rechecks、节点 draft 优先级和源码函数 AST 已交叉核对。
- 临时验证结束，真实 `state.json` 与读取前深相等；无模型、无真实工具控制、无原稿/冻结改写。

实施后只需少量聚合定向场景，不必跑全算子或完整真实模型矩阵：

1. **真实字节隔离复现**：修复前 gate精确失败；repair计划只含允许字段；B在保留原合法scope时通过修复后gate；未知文件/root越权/metadata越权仍被拒绝。再覆盖一轮新的P6非root revise→worker complete→reviewer，确认不需再次关闭scope或重新维修。若验证A，则如实展示关闭scope可通过原gate、再次同类修订会复发。
2. **保稿与有效性组合**：临时Store按实际frontier执行 P3 patch/verify + packed review + P6本地替身；packed输入必须是66ad的新字节，不能是71ec旧稿；split/tc原批准+新applicability有效，P1/P2/P4不被清掉，P6没批准前不能ready/publish。
3. **身份与原子性**：错误worker/checkpoint、已变化scope、活动/未确认进程、发布commit、假旧receipt均拒绝；重复同请求幂等；封存后/提交前后故障重放不重复计数或授予批准。若B，再验证旧执行器拒绝新格式、旧attempt仍按原binding核验、新attempt必须绑定新代码。
4. **实际接续验收**：新的模型审阅按固定来源形成真实receipt/checkpoint；调用数仅增加真实调用，旧47次和失败原貌保留，最终完整manifest/独立批准/源身份/发布门禁正常。

不要在调查阶段直接清scope、换executor、把失败response补成receipt，或把scope修复等同内容审核。上述操作只能由已实现、审查并验证的公开工具事务完成。当前只读证明的是可实施性，未声称真实run已恢复。

## 6. 授权和决策

用户已授权修工具并要求保留旧 Sparse 进度；选择 A 或 B、实现最小接口、封存恢复证据、限域维护、独立复审与正常接续属于主代理可据此推进的工程工作，不需要再让用户批准常规字段/接口/测试取舍。本调查明确推荐 B：一次修复执行器接续能力，保留当前合法状态语义并覆盖后续同类修订；A只作为有记录但会复发的备用转换，不以“README规定另建run”结束讨论。

只有发现必须改变源字节、规则/目标、模型/输出位置、追加实际不足的预算，或本次修复范围外的实质需求时，才形成新的具体用户决定。当前证据没有这些需要，也没有从 P1 重跑的技术必要性。
