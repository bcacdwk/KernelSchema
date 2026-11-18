LIVE10 — QSA r03 通过暂停屏障登记新的 P2 独立维护审阅

结论：可行，但必须依次使用正式 `pause → maintain P2 action=review → resume`。直接 `maintain` 不会重新审阅当前候选；直接 `resume` 会重现旧 normal reviewer 的 UUID 问题。三步路径已用真实冻结代码、临时状态副本验证到新 attempt 准备和原生会话预检通过；没有调用模型，没有构造回执或批准，没有操作真实运行。

本结论针对 QSA r03 `f1a26608-6adf-463a-acbe-144e3a2bc69d` 的本次现场：P1 独立批准仍有效；P2 worker `eb61bbe9-0cd2-4ca8-a4cf-da570314a4db` 已合法提交完整候选，checkpoint 为 `0e558ae6ef20d1bf02a247be4ef9228f07c155a766f013c4d4a7efda0f8a54dd`，`plan_draft` 是单 root 的端到端 QSAPreprocessUnion + QSAForward 选择。`selected_designs=null`，P2 `approval=null`，状态与 P2 scope 均为 `invalid_receipt`，原 role 仍为 reviewer；active/unconfirmed/unsettled 全空，旧 Runner 已结束。

旧 reviewer `0b7794c2-1885-4753-b000-0de0257c21a7` 原生 completed / exit 0 / success，termination confirmed、survivors 空；其 `source_evidence[0]`、`[14]`、`[96]` 分别是 1–14、109–122、1–14，共 14 行，合法超过 12 行上限而被拒绝。旧拟 approve 没有进入 accepted_receipts。私有登记仍是 ready UUID `3037bd77-40bd-4127-b6d5-493494ead218`，normal `sessions.P2.reviewer` 仍为 null。

核查的是该 run 的冻结快照，而非 LIVE09 后的当前产品代码：

- snapshot：`b9f3b9a0c9681ffbf097cd43d2d1e520f7b906ae7555bb48960f1cbf07162814`。
- rules id：`7c991dc2208f1c65f44d262334e77319fc05bde7b3e73ff1782c4448eef74185`。
- executor version `2.5.1`，digest `b7f2757135f3f6e3dda459ba08471072af65983a85839f3b2456fceea0ef602b`。
- runner、maintenance、cli、monitor、recovery、stages、store、agent、runtime、prompts、receipts 共 11 个冻结模块均与原 inputs.lock 的 SHA-256 相符。以下代码行号均指此快照 `rules/schema_extract/tools/`。

控制顺序及决定性边界：

1. `cli.py:455` 的普通控制入口先切回冻结执行器。正式 pause 在调用已结清的现场只消费暂停请求；`runner.py:1483` 保存 `resume_status=invalid_receipt`，不会调用 reviewer、改 UUID 或批准 P2。
2. 暂停后提交 maintain，`monitor.py:373` 明确返回 `preserved`，请求仍为 requested，既不启动后台 Runner，也不安装维护任务。这一步必须先完成，不能先 resume 再追赶提交。
3. 正式 resume 使用原冻结代码。`runner.py:1262` 先消费控制和维护请求，建立 draining 屏障；`1291–1297` 随后执行 `_resume_stopped` 和 `_resume_failed_scopes`。后者在 `1076–1082` 按原 role=reviewer 将 P2 scope 从 invalid_receipt 恢复成 reviewing，仅恢复调度状态，不采纳旧回执或补记 session。
4. 此时维护请求已经存在；`runner.py:1314–1320` 仅结清登记的调用，现场三类队列为空。`1346–1354` 在任何新调用准备之前执行 `mark_drained`，普通 next_calls 被屏障阻挡。故状态恢复发生在维护安装前，旧 normal reviewer 的 AgentCLI 预检不会发生。
5. `maintenance.py:162` 现在识别完整候选；创建 pending review，保持原 checkpoint，提升 P2 generation 以隔离旧结果。P1 不失效，旧计数、deadline、invalid_receipts 也不清零。`stages.py:86–98` 为新请求生成独立的 `maintenance:<request_id>:P2:review` session group/operation，初始 UUID 为 null 对应的是全新空私有 scope，不是旧 ready normal scope。
6. `maintenance.py:305–309` 从 normal_cursor 恢复原 P2 plan_draft，而不是从仍为 null 的 selected_designs 取值。`store.py:874–906` 在新 scope 准备只读 reviewer workspace，原候选摘要和来源不变。

直接 maintain 的反例也已验证：因为 `_reject` 在 `runner.py:1516–1518` 把 scope 设成 invalid_receipt，`maintenance.py:162` 判 `complete_candidate=false`，任务进入 waiting；`230–243` 立即把 maintenance 标为 complete，并由 resume_progress 回到普通 P2 worker。即便该 worker 随后重新 complete，`resume_deferred` 在 `251–253` 因 maintenance.status 已不是 waiting 而不接管；`stages.py:568–575` 仍会安排普通 P2 reviewer，UUID 问题依旧。不能靠“先普通 worker 再自动维护 review”恢复。

批准与后续阶段边界：

- 新 reviewer 必须自己重读固定来源、检查当前候选并产生当前 nonce 的新提案。`runner.py:563–579` 仍检查原生成功、session、nonce、只读产物及候选摘要；`593–645` 仍执行来源证据、完整回执和阶段检查，12 行上限不变。`receipts.py:467–470` 还要求 reviewer 的 plan 等于 worker 已提交的 plan。
- `prompts.py:151–172` 只提供 accepted_receipts 内的既有材料。被拒的旧 reviewer 不在该集合，因此不成为新的已接受决定。
- 只有新提案通过全部门禁且实际 decision=approve，`maintenance.py:421–438` 才调用通常的 `_approve`；`stages.py:442–468` 安装 selected_designs、selection_signature 和带新维护身份的 P2 approval，更新 merged checkpoint。随后 `maintenance.py:230–243` 清理已批准的 plan_draft，恢复正常游标；`resume_progress` 从第一个真正未完成阶段 P3/worker 继续。P3、P4、正常 P5 判定与最终 P6 仍须完成；单 root 的 P5 是否 skipped 由通常规则决定。
- `fields=[]` 意味着本次仅审阅，不授予产品稿件修改权限。若当前候选有实质问题，reviewer 应 blocked 并给证据，不能为了推进预定 approve。正式过程若再拒绝或阻塞，保留新现场后重新判断；本报告不是对尚未发生的新批准的保证。

给主代理/监督的最小执行步骤（本报告没有执行）：

1. 只读确认仍为上述现场、三类调用队列为空、旧 Runner 结束；没有新的 pause/cancel/global hold 与之冲突。先运行下列 pause，再确认状态为 paused、resume_status 为 invalid_receipt、P2 role 仍为 reviewer。

```sh
python3 -B schema_extract/extract_yaml.py pause f1a26608-6adf-463a-acbe-144e3a2bc69d --repo /home/shaohanyong.shy/KernelSchema --state-root /home/shaohanyong.shy/KernelSchema/.schema_extract/live_acceptance_20261117
```

2. 将下面原样任务数组保存为 `/tmp/live10-qsa-p2-review.tasks.json`，通过 maintain 登记。预期 delivery.status=preserved，maintenance request.status=requested，运行仍 paused；若不符合，先核现场，不执行下一步。

```json
[
  {
    "stage": "P2",
    "node": "root",
    "action": "review",
    "fields": [],
    "reason": "对本次固定来源下已合法提交的完整 P2 候选进行一次新的独立审阅。候选来自 worker eb61bbe9-0cd2-4ca8-a4cf-da570314a4db，checkpoint 0e558ae6ef20d1bf02a247be4ef9228f07c155a766f013c4d4a7efda0f8a54dd；请独立核对其单 root 端到端选择及 dispatch 草案与固定来源的一致性。旧 reviewer 0b7794c2-1885-4753-b000-0de0257c21a7 的拟 approve 因来源证据范围超限而被拒，不构成批准或本轮审阅结论。请自行重读来源，为当前 nonce、当前候选与全部必要结论建立全新证据，每条连续来源证据最多 12 行；不得截改旧回执或沿用旧批准。本次 fields 为空，只审阅不改稿。仅在独立检查及本轮全部门禁通过时 approve；若发现需要修改的实质问题，blocked 并给出当前来源与产物证据。后续仍按正常完整 P1–P6 流程继续。"
  }
]
```

```sh
python3 -B schema_extract/extract_yaml.py maintain f1a26608-6adf-463a-acbe-144e3a2bc69d --repo /home/shaohanyong.shy/KernelSchema --state-root /home/shaohanyong.shy/KernelSchema/.schema_extract/live_acceptance_20261117 --issue-id live10-qsa-p2-independent-review --tasks /tmp/live10-qsa-p2-review.tasks.json
```

3. 正式 resume，不增加预算、不改模型。原配置保持 Claude Code / `qwen3.8-max[1m]`，worker 与 reviewer 均 max。监督首个 attempt 必须为 `purpose=maintenance, stage=P2, node=root, role=reviewer, maintenance_step=review`，operation 为新维护身份，base_checkpoint 精确等于上面的完整候选；它不应是普通 P2 worker/reviewer。

```sh
python3 -B schema_extract/extract_yaml.py resume f1a26608-6adf-463a-acbe-144e3a2bc69d --repo /home/shaohanyong.shy/KernelSchema --state-root /home/shaohanyong.shy/KernelSchema/.schema_extract/live_acceptance_20261117
```

验证记录：`/tmp/live10_probe.py` 与 `/tmp/live10_probe_result.json`。在 `/tmp/live10_probe_gwqfqmp5` 内复制公开 state、inputs.lock、checkpoint、已接受 attempt/receipt/validation 和旧 session 登记；导入上述冻结代码。实际执行 Store/Runner 的暂停、维护登记、恢复、排空及新 attempt 准备；在首次准备后截停调度，另用冻结 AgentCLI 的真实 session 预检，在进入 `_run` 后立即返回探针结果，未执行 CLI。直接 maintain 首调用为普通 worker；三步路径首调用为新的 maintenance reviewer，预检读取 empty/null 的新 scope，旧 normal session 字节不变。候选和 plan_draft 保持，selected_designs/P2 approval 仍 null，accepted_receipts 不变，P1 approval 不变；P2 calls 从 2 到 3，原 deadline 和 invalid_receipts 保持。没有模拟或生成 approve/receipt，也没有运行全测试、GPU 或真实模型。新批准后的阶段回归结论来自上述冻结代码审阅。

真实现场只读保护：`/tmp/live10_real_before.json` 记录 46 个原 run / checkpoint / session 登记 / 冻结代码文件摘要，验证后逐项相同。关键摘要：

| 文件 | SHA-256 |
| --- | --- |
| state.json | `00717e7ee31f0671baf28c21d855326d55892d0a3a08370c4ccdfd662b0b77e3` |
| inputs.lock.json | `4ffd843181b9795d3a83dee3a705cc54f824d6c81e160b8fb104c76c1810e1f9` |
| 旧 P2 reviewer result.json | `1db2aa9618ea4ff0f562981b38a799dc597f7cf4aac538e63bc09a21748b77eb` |
| 合法 P2 worker receipt.json | `be096fc64dc331a4a4d80cf0d7d2230e32457cde4be987c63d73d496ec36fdcd` |

本次仅写本报告及 `/tmp` 诊断副本，未修改产品、真实状态、冻结快照、模型稿件、原回执或私有模型历史，未读取隐藏推理；未操作 FA4 r04。报告与验证完成，停止写入；真实三步操作由已获授权的主代理/监督执行。
