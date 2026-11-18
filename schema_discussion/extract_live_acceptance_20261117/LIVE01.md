# LIVE-01：冻结清单引用已删除文档

根因：`schema_extract/EXECUTOR_ALIGNMENT.md` 已不存在，但 `rule_files()` 仍将它列为必读冻结输入，导致新 run 冻结失败；材料导航白名单、现有测试和活跃文档也保留该入口。

修复：从 `snapshots.py`、`prompts.py` 及 `test_prompt_materials.py` 删除该入口；更新根 README、EXTRACT_SCHEMA、Extract README/REVIEW、Template README/案例索引，将导航指向现有执行协议、阶段覆盖和验证章节。未恢复历史文档，未添加缺失文件跳过逻辑；schema_docs 字段规则保持不变。

验证：

- 当前 `rule_files()` 的 97 个文件全部存在；活跃目录中该旧入口引用为零，替代链接目标及章节存在。
- 在系统临时目录创建单文件来源，`freeze_inputs()` 与 `verify_snapshot()` 均通过。
- 现有 `test_p6_receives_worker_facts_after_final_content_and_only_real_documents_are_provided` 单测通过（1 项，9.622 秒）。
- `git diff --check` 通过；未调用真实模型、未运行全量测试、未修改 `.schema_extract` 运行快照，未提交 commit。

2026-11-17：修复完成，可启动 FA4 Prefill 与 QSA 的真实运行；本修复代理停止写入。
