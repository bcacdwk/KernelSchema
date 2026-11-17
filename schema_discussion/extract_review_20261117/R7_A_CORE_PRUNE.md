# R7 A：按实际用途裁剪核心工具

本轮删除三项没有独立使用价值的机制；没有调整正式字段规则、六阶段语义或预算策略。

## 删除内容与理由

1. **不再接管仅含 `.gitkeep` 的输出目录。** 删除 publication 的 marker 备份、inode/mtime 身份、恢复与清理协议，以及只服务该恢复协议的 intent 文件。新提取直接要求目标不存在；已有目标只可能是相同完整结果的重放或内容冲突。`create_run` 在冻结输入前拒绝已有目录；B 同步注册入口和 README。原子不覆盖 rename、已审字节、P6 批准与控制屏障保持。
2. **删除通用 JSON 分页阅读器。** 当前调用已提供原生 Read/Glob/Grep；删除仅测试使用的 `read_range`、base64 游标、逐字符二分装页及其 CLI 参数。保留 `--evidence` 的文件校验、行号、长行字符偏移和短引文，以及阶段只读自检。删除测试中专门重组分页的片段，继续验证长行后部证据。
3. **交接只保存在已接受 receipt。** 删除 `Store.handoff` 与 Runner 写入的 `stages/.../handoff.md`；B 删除 prompt 中再次复制该文件的分支。当前批准回执仍直接登记为 `current_decision`，已有跨阶段 `knowledge_destination` 收集保持，回执哈希与调用身份继续验证。中断恢复交接不变。测试改为从 P6 材料定位当前 consumer 回执并读取其 `handoff`，不再断言多一份文件。

A 主责四个产品文件合计净减 201 行，三个测试文件净减 9 行；相对本轮开始计算，不重复计算 R5/R6。B 的协同文件另计。

## 验证

以下四项既有离线用例全部通过，耗时 **117.912 秒**：

- `CurrentDeliveryTests.test_real_complete_publication_conflicts_replay_p6_and_source_identity`：临时新目录发布、`.gitkeep` 冲突保留、相同结果复用、rename 后写状态中断重放、当前 P6/来源与暂停屏障。
- `StageFlowTests.test_multiple_sources_and_late_unrelated_sibling_do_not_expire_review`：实际整区来源、无关节点不使审阅过期、P6 当前批准回执交接可达。
- `ProposalTests.test_long_line_evidence_can_select_a_late_declaration_losslessly`：长行后部精确短引文和偏移边界。
- `PromptMaterialsTests.test_p1_preview_is_read_only_and_reports_pending_normalization`：自检保持 workspace 原字节并报告待规范化内容。

另以临时目录直接检查：`create_run` 对已有 `.gitkeep` 目录在创建 runtime 前报错；`reading.main --evidence` 输出所选 Unicode 行的准确引文。七个改动 Python 文件 AST 与限定 `git diff --check` 通过，删除接口没有现行引用。

测试使用 `PATH=/usr/bin:/bin /opt/venv/python3.13.12/bin/python3 -B`，确认 PATH 中无 Claude/Codex。流程用确定性内存回执和临时合成材料；没有调用真实模型、queue、GPU 或正式发布。

## 文件与未决

A 修改：

- `schema_extract/tools/publication.py`
- `schema_extract/tools/reading.py`
- `schema_extract/tools/store.py`
- `schema_extract/tools/runner.py`
- `schema_extract/tools/tests/test_current_delivery.py`
- `schema_extract/tools/tests/test_stage_flow.py`
- `schema_extract/tools/tests/test_proposals.py`
- 本报告。

B 协同修改：`schema_extract/tools/prompts.py`、`schema_extract/tools/monitor.py`、`schema_extract/README.md`。没有本轮已确认未处理的问题；未改 Template、案例、Kernel、archive、旧证据或精简前快照，未提交 commit。报告完成后停止写入。
