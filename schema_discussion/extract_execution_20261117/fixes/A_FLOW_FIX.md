# MAX 修复 A：流程四项

A-F1–A-F4 已修复并完成相关离线验证，代码停止写入，交原 XHIGH 独立复验。没有启动模型、GPU、queue 或真实六阶段全流程；没有修改 Template 字段/格式规则、Wiki、旧 run、个人配置，也没有 commit。已取消的真实全链不作为待补跑项。

## A-F1：可信集合在内容读取之前核验

根因是 `artifact_manifest()` 只按扩展名接纳文件，阶段文件范围检查发生在原稿取证之后。

- `artifacts.artifact_inventory(directory, expected=..., additions=...)` 先完成全目录名字、文件类型、单链接、直属位置及必需文件检查，不读内容；`artifact_manifest` 随后只散列已列出的文件。内部 YAML 枚举直接使用 inventory，因此最终目录也能先拒绝额外 YAML。
- 普通调用名单来自已验证的 base checkpoint；P2 worker 仅允许新增 `dispatch.md`。既有只读文件仍读取、散列并比较，保护字段仍按规范化语义核验。计划中尚未准备的其他节点没有文件许可。
- 正常采纳、回执、规范化、失败/中断稿、只读 helper、裁决候选及 checkpoint 创建/复核均接同一门。`Store.checkpoint` 现在要求调用者传 `expected_files`，各调用点只从受信准备、已有检查结果及明确采用的文件推导。原稿仍正常完整保存，复制遍历已经核验的名单。
- select 回执使用 `_artifact_baseline()` 核登记的候选和 `chosen.files`。B 已在 `audit.py` 复用该接口，审计不能用 validation 自报的原稿名单扩大许可；其公开审计/归档反例见 [B_RUNTIME_FIX](B_RUNTIME_FIX.md)。

原 `repro_preflight.py` 修复前记录额外文件读取 **4 次**、已复制进 original；修复后为 **0 次**、未复制。新正常/失败/中断稿用例进一步把 `extra` 放在已选但尚未准备的计划中，对整个候选目录设置拒读观察器，三条路径均先拒绝；checkpoint 也拒绝。原子 rename 正例、剩余文件/目录/软硬链接/缺失文件反例及合法原稿证据仍通过。

## A-F2：事实阅读资格与批准代际分开

根因是 P6 只提供当前 generation 的记录，同 scope 又只保留最近一份，普通 worker 的未落实事实会在重新独立审阅后失去材料入口。

`prepare_attempt_materials()` 继续使用既有 accepted receipts：同 scope 及本阶段需要的上游交接，只要有 handoff/issues 就保留完整回执的只读指针；P6 还接收 P3→P4 的原始交接及 P5 事实，不要求中间 reviewer 转抄。每个指针绑定原回执摘要、固定 source_id、实例、原代际，并标明实例仍相同、已改变或已移除。回执与登记 invocation 的身份同时核对。prompt 只列材料指针，长回执放在 materials，保留先读当前成果、再读过程记录的顺序。共同提示明确历史事实不授予当前批准或 `from` 来源资格。

原 `repro_handoff.py` 中，升代际后的新 P5 与 P6 均重新能读到原 worker 的参数事实；原回执继续 accepted 且内容不变。闭环测试还实际从 P2 移除该代表，P6 仍收到标记 `removed` 的原事实，而该节点不在 workspace、dependency digests 或当前 source identities 中。没有增加知识数据库、长报告或 schema 历史字段。

## A-F3：P1 不把待修 Design 当 Contract 修订前提

根因是任意 Contract 语义差异都会令 `_normalization_consumers()` 扩到所有非 root，继而解析尚待 P5 本地化的退休来源。

P1 的初始规范化范围遵守本阶段完成边界；Contract 的语义修改由既有 `refresh_approvals()` 核实际使用并安排后续 P3/P5 修审。Template 标识为 `identifier+reference` 的关联规范化仍扩到完整消费者集合，超出权限仍整组拒绝；普通文案及末句号修正不再被当成必须解析所有 Design 的理由。其他阶段的来源闭合与最终全图检查没有放宽，也没有把历史 provider 恢复为当前来源。

原低层复现中 metadata 与 Contract 均 `landed`；实际阶段链的 P1 worker 原样复现也成功采纳。扩展的既有退休 provider 测试实际完成 P1 独立批准、P3/P4 复审/安排、P5 显式本地化及独立批准，同时核原历史 checkpoint 未变。该测试的输入说明带末句号，确实经过 Template 格式修正。另一反例证明真正的 Contract 名称规范化仍纳入完整集合，并拒绝对只读 provider 的关联写入。

## A-F4：合入后复核受影响的阶段投影

根因是合入后只检查 P3/P5 使用上下文，没有复核 P2 绑定的 root Source，导致状态仍 approved 而下一阶段准入才失败。

`refresh_approvals()` 在上游合入后比较已有 P2/P4 职责投影，实际变化时核原批准真实性，再复用 `invalidate_stage()` 与 `resume_progress()` 开启独立复审。既有 validation 保存旧/新职责摘要与原批准指针；原回执、实例、预算、计数和一次裁决/签收额度保留。进入 P2/P4 reviewer 时，从当前选择/安排准备相应 `plan_draft`。

原 A4 脚本现在输出 `stage_after_p1_approval: P2`、`p2_status: reviewing`。其最后一行硬编码寻找旧路径 P3，因此出现 `StopIteration`，该原始输出完整保留，未记作通过脚本。新增行为测试从同一 Source locator 更正实际完成 **P2 reviewer → P3 reviewer → P6 ready**，并核当前实例与原预算、旧批准均保留。既有 Name/Describe 同步正例确认 P2/P4 投影未变时继续沿用原批准。

## 验证与复放

完整命令、原始输出及源码核对见 [A_FLOW_checks.txt](A_FLOW_checks.txt)，各副本见 [A_FLOW_paths.json](A_FLOW_paths.json)。所有普通复现使用确定性 `FixtureBackend` 和真实 Runner/Store/Template 门；PATH 的 `claude/codex` 固定 exit 97，未调用模型。

| 范围 | 实际结果 |
| --- | --- |
| 审阅报告五个原脚本（四项问题，A3 两种复现） | 修复前全部重现；修复后 A1–A3 输出纠正，A4 按上文进入 P2 |
| 四项修复及相邻边界 | 6 项通过，98.616 秒 |
| 相关既有门及最终 P2/名单路径 | 14 项通过，150.177 秒 |
| 最后一次改动：只枚举名字与最终目录读前拒绝 | 4 项通过，7.343 秒 |
| 最终源码核对 | 169 个快照文件与当前工作区逐字节一致；所改 11 个 Python 文件编译、`git diff --check` 通过 |

本轮覆盖 **18 个不同用例**，上述局部复验不重复计数，没有重跑 125 项整套。测试包括合法原稿/最终证据分离、P5 保护字段、P6 限域写、实际跨区闭包、独立来源批准、旁支合入、维护以及裁决 patch/select/signoff。

最终一致副本为 `/tmp/extract-flow-fix-sealed-cqay990h`，其中 `snapshot-manifest.json` 保存实际文件摘要。可在该目录使用以下方式重放普通反例及对应完整闭环（其余原脚本仅替换脚本名）：

```sh
PATH=/tmp/extract-flow-fix-sealed-cqay990h/audit-bin:/usr/bin:/bin \
  /opt/venv/python3.13.12/bin/python -B repro_preflight.py

PATH=/tmp/extract-flow-fix-sealed-cqay990h/audit-bin:/usr/bin:/bin \
  /opt/venv/python3.13.12/bin/python -B -m unittest -v \
  schema_extract.tools.tests.test_current_returns.CurrentReturnsTests.test_process_facts_survive_metadata_return_and_removed_instances \
  schema_extract.tools.tests.test_current_returns.CurrentReturnsTests.test_removed_provider_stays_historical_and_consumer_must_change_from \
  schema_extract.tools.tests.test_current_returns.CurrentReturnsTests.test_p1_source_change_opens_only_the_invalidated_selection_projection
```

真实隔离中的 helper 使用临时 fake CLI，按既有显式系统 bootstrap 先挂真实私有 proc，再通过 `setpriv` 永久降回 UID 1000/GID 100 并设置 NNP，最后才启动仓库 Python；确切命令已留日志。产品没有加入 sudo、假 proc 或隔离绕过。

没有本轮尚未通过的已执行适用检查；待原 XHIGH 对四项修法和最终源代码独立复验。真实语义全链仍未验收，已有停止证据与原始日志未改。
