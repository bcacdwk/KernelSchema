LIVE11 — QSA r03 的 blocked maintenance 不能再登记新独立维护审阅

结论：在本 run 的真实冻结执行器中，不可通过新的 issue/request ID 再次独立审阅同一候选。`pause → maintain → resume` 本次不成立：pause 保留 blocked maintenance，新 maintain 在登记阶段即合法拒绝。普通 resume 保持 blocked；`resume --scope P2` 也拒绝。本结论已用真实状态的临时副本、正式冻结 CLI 和禁止调用模型的假 transport 验证。没有可提供的真实 run 最小续审命令。

与 LIVE10 的区别是决定性的：LIVE10 起点只有普通 P2 reviewer 的 invalid_receipt，maintenance 尚未 active；现在旧维护 request 已安装，`maintenance.status=blocked` 仍被定义为 active。不能把前次成功的三步路径复用于这个不同的控制边界。在不得改真实状态、冻结代码、UUID、原回执或证据规则的约束下，现有正式命令没有为这个 blocked 维护任务建立新 operation 的入口。若必须继续完成提取，新 run 是排除上述正式路径后的最后选择，由主代理安排完整流程；本报告不启动新 run，不引入 seed、迁移或特殊绕过。

现场与冻结身份：

- run：`f1a26608-6adf-463a-acbe-144e3a2bc69d`，state root `.schema_extract/live_acceptance_20261117`。
- maintenance request：`3c0c4669-1cd0-4981-bfd1-899c350a0df6`，issue `live10-qsa-p2-independent-review`；维护状态 blocked，唯一 P2 task 为 `invalid_receipt / review / reviewer / complete_candidate=true`。
- 最新 attempt：`aac442e7-0dab-46ec-a9ad-776aee3eb859`。原生 success、outcome completed、exit 0；termination confirmed、survivors 空。唯一超限是 `artifact_evidence[29]` 的 `root.yaml:181–194`，共 14 行；本次全部 source_evidence 均不超过 12 行。拟 approve 未进入 accepted_receipts。
- working checkpoint 与顶层 checkpoint 都为 `0e558ae6ef20d1bf02a247be4ef9228f07c155a766f013c4d4a7efda0f8a54dd`。P1 approval 保持，P2 approval=null，selected_designs=null；原单 root plan_draft 保持。三类 active/unconfirmed/unsettled 队列为空。
- 旧 operation：`maintenance:3c0c4669-1cd0-4981-bfd1-899c350a0df6:P2:review`。state 的 reviewer session=null，原生 ready session 为 `1b81629c-386a-48d6-8e6e-c9eba619cf55`；没有补记或改写它。
- snapshot：`b9f3b9a0c9681ffbf097cd43d2d1e520f7b906ae7555bb48960f1cbf07162814`；rules id `7c991dc2208f1c65f44d262334e77319fc05bde7b3e73ff1782c4448eef74185`；executor 2.5.1、digest `b7f2757135f3f6e3dda459ba08471072af65983a85839f3b2456fceea0ef602b`。
- cli、maintenance、monitor、receipts、runner、recovery、stages、store、agent、runtime、reading、prompts 共 12 个冻结模块已逐个与 inputs.lock 中的 SHA-256 核对一致。以下未特别标注的代码行号均指该 snapshot 的 `rules/schema_extract/tools/`。

决定性控制边界：

1. `maintenance.py:19–20`：`active(state)` 明确包含 `active` 和 `blocked`。`submit_request` 在 `97–98` 拒绝新的请求，只要旧请求 pending、maintenance active 或 waiting。不同 issue ID 不豁免此条件，request UUID 也尚未生成。相同 issue 与相同 tasks 在 `92–96` 只复用已有 identity；更改同一 issue 的具体任务则拒绝。
2. `runner.py:1483–1489` 的 pause 只记录顶层暂停屏障和 resume_status；不会把 blocked maintenance 关闭。实测 pause 后顶层 paused、resume_status=blocked，maintenance 仍 blocked，新 maintain 仍被同一条规则拒绝。
3. `runner.py:1052–1066` 的 `_resume_failed_scopes` 对 active maintenance 只考察 `budget_exhausted` task，并在 `settle` 后返回；本次 `invalid_receipt` 不会变成 pending。`maintenance.py:276–281` 再次把顶层与维护都保持 blocked。加预算也不改变本次错误的类别，故不构成恢复办法。
4. `stages.py:238–241` 在 maintenance active 时仅走维护调度；`maintenance.py:210–214` 要求顶层 running、task pending。旧 task 为 invalid_receipt，因此直接 resume 不会抢先派发旧 reviewer，也不会退回普通 P2 reviewer。新请求无法登记，所以不存在“先派新 request 还是旧 request”的竞争。
5. 当前 state 内 `maintenance_request.status=draining` 是旧投影；真正 supervisor ledger 中该 request 已是 applied。`maintenance.py:27–29,111–124` 的 pending barrier/消费依据 ledger。普通 resume 不会重新安装旧 request，也不会靠这个旧投影解除 blocked task。
6. `resume --scope P2` 首先在 `recovery.py:183–186` 排除顶层 blocked 的非普通恢复现场，实测错误为 `scope is not an ordinary recoverable failure`。即使先前证据门通过，`maintenance.py:32–33` 与 `monitor.py:613–615` 仍明确用 publication barrier 阻止当前维护期间的 scope recovery。CLI 无关闭、撤销或替换 blocked maintenance 的正式命令；cancel 与 pause 一样只改变运行控制状态。
7. 批准入口仍须经过当前调用的全部门禁，并由 `maintenance.advance` 的新 accepted receipt 驱动；`maintenance.py:421–425` 只有合法新 approve 才调用 `_approve`。控制命令不采纳旧拟 approve。本次验证中没有任何新调用能到达这个入口。

临时副本实测：

| 路径 | 正式行为 | 是否派发调用 |
| --- | --- | --- |
| 新 issue 的 maintain | exit 2：`finish the pending current-run maintenance before submitting another request` | 否；Runner 未启动 |
| pause → 新 issue 的 maintain | pause exit 0；maintain 同样 exit 2；maintenance 仍 blocked | 否 |
| 上述失败登记后 resume | exit 1；回到 blocked，task 仍 invalid_receipt | 否 |
| 直接 resume | exit 1；仍 blocked | 否 |
| resume --scope P2 | exit 2：`scope is not an ordinary recoverable failure` | 否 |
| 同一 issue、原 tasks 再提交 | reused=true，只返回旧 applied request | 不生成新 request/operation |

验证脚本 `/tmp/live11_probe.py`，结果 `/tmp/live11_probe_result.json`，最终通过的临时目录 `/tmp/live11_probe_3ia9ger3`。脚本复制公开 state、inputs.lock、checkpoint、attempt 的公开记录、session 登记、QSA 控制/维护请求以及 supervisor ledger；不复制或阅读私有模型对话历史。它导入冻结模块，实际调用 `cli.main` 的 pause、maintain、resume 和 scope resume，保留正式请求、冻结执行器核验、Runner 与调度控制流。唯一运行替身是给 Runner 注入禁止模型调用的 transport，并让任何意外 attempt 准备、后台启动或 re-exec 立即失败；所有预期结果均正常通过，没有触发这些替身。

每条路径都验证：checkpoint/产物字节、plan_draft、selected_designs、accepted_receipts、P1/P2 全部 stage 数据、旧 maintenance/task、session 字节、attempt 列表与 attempt_usage、budget_extensions、原 deadline 完全不变。P2 calls 始终 3、invalid_receipts 始终 2、receipt_repairs 始终 0；没有退款、清零、补记 UUID 或新 approval。pause/resume 只允许自身正式控制与 Runner 观察记录变化。零模型 transport、零新 attempt、零新 receipt、零批准。

关于现有 receipt_repair：

本次不在其自然入口内。脚本将原 proposal 原样交给冻结 `receipts.normalize_response`，仍得到精确的普通 `ExtractError: artifact_evidence: invalid evidence range (maximum 12 lines)`，没有更改 181–194 或任何其他字段。`reading.py:113–115` 对超范围抛 ValueError，`receipts.py:203–206` 包装为 ExtractError；`runner.py:853–864` 仅对 ReceiptFormatError 进入自动 receipt_repair，普通 ExtractError 走合法拒绝。

现有自动 receipt_repair 用于符合条件的 JSON 形状或指定产物格式错误，并非可手动调用的“仅补证据、锁定原结论”命令。它仍需当前作用域的模型提案与完整门禁；而 `runner.py:807–808` 对 purpose=maintenance 的 recover_call 还会明确把维护 task 置为 invalid_receipt/cli_error。当前产品的 LIVE09 修复在 `_receive` 前记录真实完成 UUID，但没有把这类证据错误改成 ReceiptFormatError，也没有增加 blocked maintenance 的关闭/替换入口。本 run 不热升级；本报告没有新增功能或扩大规则。

真实现场只读保护：`/tmp/live11_real_before.json` 记录 94 个 QSA run、checkpoint、session 登记、公开 attempt 记录、请求和冻结模块文件摘要，完成后全部一致；QSA 的 supervisor maintenance ledger 也未变。共享 supervisor 中其他 run 的合法观察不在本次修改范围。

| 文件 | SHA-256 |
| --- | --- |
| state.json | `8fa5514b32fa056e9448fccdba43b2619e528ff35a982dbe26efa9d2cbfd5057` |
| inputs.lock.json | `4ffd843181b9795d3a83dee3a705cc54f824d6c81e160b8fb104c76c1810e1f9` |
| 最新 reviewer response.json | `add9ed019255e9082c42e6847a1b67bfec25edec8cea63b1e5de038699b14409` |
| 最新 reviewer rejected.json | `816f15ea9c8c62eb01047ec6a64e927471f86603f5ac487d1e8a942375f276a2` |

本次只写本报告及 `/tmp` 临时验证资料，未修改产品、真实 run、快照、原稿、原回执、证据范围或私有会话；未调用真模型、未操作 FA4 r04、未提交 Git。报告与验证完成，停止写入。真实后续由已获授权的主代理/监督决定并执行。
