# LIVE15：删除证据选择范围的任意 12 行门槛

结论：按主代理接受的 [LIVE14](LIVE14.md) 评估，当前产品已取消 `source_evidence` / `artifact_evidence` 选择范围的固定 12 行上限。10 项定向测试通过；六份真实公开提案的 684 条证据原样补验，21 条纯长度拒绝消失，QSA 维护提案的两条真实越文件尾错误继续拒绝。此次补验没有接受旧提案、生成批准或恢复任何真实 run。

## 精简净变化

- [reading.py](../../schema_extract/tools/reading.py) 的 `read_evidence` 和 [receipts.py](../../schema_extract/tools/receipts.py) 的 `_anchor` 各删除一个跨度谓词。没有新数字上限、参数、配置、字段、迁移或摘录系统；保留原先 `selected[quote_offset:quote_offset + 320]` 的精确摘录。
- [response.md](../../schema_extract/prompt/operations/response.md) 删除计行和拆段要求，改为“source_evidence 与 artifact_evidence 选择支持当前判断的必要原文范围”；[工具 README](../../schema_extract/tools/README.md) 同步一句正向指导。没有在其他阶段增加重复提示。
- [LIVE13](LIVE13.md) 的 `source_evidence[i]` / `artifact_evidence[i]`、`path:start-end` 定位保持；范围错误现在报告 `invalid evidence range (file has N lines)`，不再显示已删除的上限。
- [test_proposals.py](../../schema_extract/tools/tests/test_proposals.py) 增加三项定向证据检查。[test_stage_flow.py](../../schema_extract/tools/tests/test_stage_flow.py) 和 [test_maintenance_receipts.py](../../schema_extract/tools/tests/test_maintenance_receipts.py) 中原以 13/14 行超限触发的恢复测试，改为按真实文件行数加一触发越界；原恢复断言保留。维护恢复最终的新合法提案使用 source 13 行、artifact 38 行。

本次改动四份产品代码/说明、三份测试，新增本报告；保留工作区此前未提交改动。正式 `schema_docs` 字段、Kernel、examples 和全部 `.schema_extract` 原 state、snapshot、稿件、回执均未由本次修改。没有替旧提案裁剪、拆分或补填证据。

## 定向验证

使用临时 fixture / Store、真实产品归一化与 Runner，以及本地 fake Claude CLI；没有真实模型或 GPU 调用。

| 验证 | 结果 |
| --- | --- |
| 两类证据分别选择真实 13 / 38 行，含非零 offset | 正规 `normalize_response` 与 `validate_receipt` 通过；原范围与提案不变，全文摘要/来源身份保留，quote 精确且不超过 320 字符 |
| 起始为 0、倒置范围、超真实文件尾 | 归一化和 receipt 再校验均拒绝；保留数组索引、路径与实际行数 |
| 负数、尾后、布尔或非整数 offset；非整数行号 | 拒绝；既有长单行中的晚出现声明仍可精确选择 |
| 不在允许清单的文件、相对越界、绝对路径、符号链接 | 拒绝；来源和 reviewer raw 字节变化仍拒绝 |
| 伪造、空白、超过 320 字符的 quote 或错误摘要 | receipt 校验拒绝 |
| normal 维护真实越界 → 原 scope resume → 缺 summary → 原 scope 再次合法提交 | 只有第三次新 approve 被接受；原 task、role、step、checkpoint 与精确 session UUID 延续，调用/用量保留，旧拟 approve 无效 |
| LIVE09 原生完成后证据越界，再次精确 resume 与丢 UUID 预检结清 | 原会话登记和恢复覆盖保留，旧提案仍未批准 |
| LIVE13 fence、未结算、候选/身份漂移、缺登记 UUID、耗尽预算、缺正向结清证据、语义 blocked、一次性 adjudication | 原恢复约束保持；不借证据规则变化获得额外决定或退款 |

实际命令分两组：

```sh
/opt/venv/python3.13.12/bin/python3 -B -m unittest \
  schema_extract.tools.tests.test_proposals.ProposalTests.test_multiline_evidence_preserves_ranges_and_exact_short_quotes \
  schema_extract.tools.tests.test_proposals.ProposalTests.test_evidence_requires_real_bounds_and_valid_offsets \
  schema_extract.tools.tests.test_proposals.ProposalTests.test_evidence_requires_allowed_sources_and_authentic_quotes \
  schema_extract.tools.tests.test_proposals.ProposalTests.test_long_line_evidence_can_select_a_late_declaration_losslessly \
  schema_extract.tools.tests.test_proposals.ProposalTests.test_source_and_reviewer_artifact_changes_are_rejected

/opt/venv/python3.13.12/bin/python3 -B -m unittest \
  schema_extract.tools.tests.test_maintenance_receipts \
  schema_extract.tools.tests.test_stage_flow.StageFlowTests.test_native_session_survives_rejected_source_range_and_resumes_exactly
```

结果分别为 5 tests / 1.983s / OK 和 5 tests / 36.420s / OK；日志 `/tmp/live15_evidence_tests.log`、`/tmp/live15_recovery_tests.log`。`git diff --check` 通过，未运行全套测试。

## 六份真实公开提案的只读补验

样本与路径沿用 [LIVE14](LIVE14.md)，不改变任何原范围、offset、note、索引或决定。用本次修改前保存的 reader 与当前 reader 对同一原选择逐条比较，再运行当前 `_anchor` 和 `normalize_response`；所有结果只在内存或 `/tmp` 中，不进入真实运行接收/审批路径。

| 原提案 | 修改前拒绝 | 当前拒绝 | 完整 proposal 归一化 |
| --- | --- | --- | --- |
| FA4 r02 worker `98ca1cea` | 1 | 0 | 通过 |
| QSA r03 worker `eb61bbe9` | 0 | 0 | 通过 |
| QSA r03 reviewer `0b7794c2` | 10 | 0 | 通过 |
| QSA r03 maintenance reviewer `aac442e7` | 2 | 2 | 在第 13 项 artifact 拒绝 |
| FA4 r04 worker `9d74ca7d` | 0 | 0 | 通过 |
| FA4 r04 reviewer `e6336933` | 10 | 0 | 通过 |

合计 607 条 source、77 条 artifact；范围拒绝从 23 降至 2，恰好消除 21 条仅超过旧行数门槛的拒绝。全部成功摘录仍不超过 320 字符。QSA 的 `root.yaml:156–193` 是原有真实 38 行范围，未拆分后通过选择和归一化检查。

QSA maintenance 的原证据仍有两条无效：

```text
artifact_evidence[13] dispatch.md:71-76: invalid evidence range (file has 75 lines)
artifact_evidence[29] root.yaml:181-194: invalid evidence range (file has 193 lines)
```

完整归一化按顺序停在第一项；逐条选择检查确认第二项也继续被拒。该提案并非仅因 12 行上限受阻。另五份通过的是当前工具的正规归一化与证据检查，不代表完整阶段语义审阅、其余验收或批准已通过。

脚本 `/tmp/live15_real_probe.py`，结果 `/tmp/live15_real_probe_result.json`；88 个涉及的固定来源、公开 attempt/proposal、锁、raw 产物及关键冻结文件的前后 SHA-256 一致，清单 `/tmp/live15_real_before.json`。没有对正在运行的 state 宣称静止，也没有运行 `resume`、维护或发布命令。

## 保留的边界与实际代价

合法固定来源和允许文件、普通文件/相对路径/无符号链接、整数行号及 `1 <= line_start <= line_end <= 实际行数`、有效 offset、非空真实 quote、全文摘要和身份校验均保留。阶段字段写权限、必需证据、P1 quote 中的具体比较声明、全文及使用上下文的独立审阅、P5 链上的真实文件和定义覆盖、未批准不能发布的规则均未改动；handoff 原有 1200 字符 / 20 行预算保持。

取消跨度限制会让较宽范围更容易满足 P5 `_region_anchored` 的“范围包含定义行”检查，机械局部定位约束确实减少。该检查仍要求真实文件和当前摘要；范围是否支持判断继续由独立角色审阅。本次没有另设防御阈值或扩大产品设计，也不把范围覆盖等同于理解定义。

本次仅影响当前工具补验及后续新冻结。FA4 r04、QSA r05 及所有已有冻结运行保持其原执行版本，不热升级、不迁移、不合成批准。未读取或复述模型隐藏推理，未输出认证信息，未提交 Git。实现、定向验证和报告完成，停止写入。
