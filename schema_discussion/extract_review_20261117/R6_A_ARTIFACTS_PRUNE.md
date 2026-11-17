# R6 A：产物、输入与来源精简

完整阅读限定的七个产品模块及 R2 A、R4 A/B、R5 A/B 结论，以 `/tmp/schema-extract-pre-prune-ego_8hhb` 只读快照区分前轮修复与本轮精简。先向主代理报告具体冗余，再修改；没有发现值得大改的重复职责，保留现有产物与来源流程。

## 实际删改

1. **删除范围检查未使用的 `decision` 参数。** `_validate_scope` 从未读取此参数，`validate_artifacts` 也只向它转发。两处定义、内部调用和唯一产品调用同步删除；`runner.py` 的一行调用变更已获得主代理额外授权。`normalize_candidate` 和 `validate_artifact_sources` 仍读取各自的 `decision`，保留完成/批准时的严格检查、来源要求及 Runner 改稿授权。测试仅同步十个调用，没有增加、删除或弱化断言。
2. **删除 `_same_inode` 无消费者的目录选项。** 全部四处调用都只检查备份 marker 的设备号与 inode，从未传入 `directory=True`。函数保留原有 `lstat`、缺失路径处理及设备号/inode 比较；目录身份仍由 `_same_scaffold` 单独核对类型、设备号、inode、mtime 和 ctime，marker 内容仍由 `_empty_marker` 检查。原子不覆盖 rename、marker 恢复和并发冲突处理均未改。
3. **删除两个未用导入。** `inputs.py` 的 `json` 与 `reading.stage_check` 的 `manifest` 没有读者；其余读取、冻结与 preview 行为不变。

相对精简前快照，四个主责产品文件净减两行，另同步 `runner.py` 的一个调用关键字。`standalone.py`、`snapshots.py`、`comparators.py` 保持原样；没有新增包装或兼容分支。

## 保留必要行为的依据

- `_file_scope` 在读入候选前的目录/写入边界，与 `_validate_scope` 对规范化后语义及字段权限的核对，针对不同输入；原稿摘要、候选摘要、严格重读、落地前输入漂移检查也针对不同时间点。本轮不合并这些门禁。
- P1/P2 继续复用 Template 的算法与适用文本预算，warning 仍进 `notices`；未来 Design 不要求提前完成。冻结清单保留执行对齐与 Template 工具两份正式指南，相对路径排除 tests 的修复保留。
- `_synchronize_metadata(nodes=...)` 不是闲置选项：Store 恢复指定节点草稿时实际使用。本轮保留此参数和共享元数据记录；唯一 Contract、任务 Name/Describe、版本、阶段权限及 P6 明确范围仍由原检查约束。
- 当前 `from` 来源、实际内容 manifest 与跨区解析闭包承担不同职责；来源当前批准、P6 独立批准和发布精确集合继续分别核对，没有删除来源或批准身份字段。
- standalone 的 Template digest 同时绑定解析缓存和回执，生成后仍用同一包严格校验，并在提交前重核代码 manifest。`source_files` 与 `validated_directory_files`、生成/复用各时点来源漂移检查、节点自身上下文和原子目标冲突保护均保持。冻结根与当前代码摘要两种 parser 缓存调用没有合并。

## 实际验证

仅执行以下四项已有离线检查，**4 项通过，3.508 秒**：

1. `ArtifactTransactions.test_current_field_scopes_and_root_when`：P2/P3 字段权限、root 范围及 reviewer 只读。
2. `ArtifactTransactions.test_p1_shared_metadata_is_a_recorded_derivation_without_worker_node_permission`：P1 规范化事务、派生共享元数据、版本及 P1/P5 写入边界。
3. `ArtifactTransactions.test_p6_changes_are_limited_to_the_reviewed_fields`：P6 明确审阅字段范围。
4. `SourceGapTests.test_source_record_does_not_hide_missing_core_facts_or_false_reference`：缺失核心事实或错误 reference 仍被拒绝。

测试命令使用 `PATH=/usr/bin:/bin /opt/venv/python3.13.12/bin/python3 -B`，先断言 `claude` 和 `codex` 不在 PATH。上述测试只读当前规则并使用临时合成材料，没有调用模型/CLI 子进程；测试解释器已退出。没有运行真实模型、queue、GPU、完整提取或正式发布，也没有为无效参数/导入新增测试。

七个变动 Python 文件 `ast.parse` 与限定 `git diff --check` 通过；对当前 Extract Python 调用进行 AST 检查，确认 `validate_artifacts` / `_validate_scope` 不再接收 `decision`。逐文件快照差异确认主责产品范围只有上述删改；三份保留模块与快照完全相同。测试同步仅为调用参数删除。

## 文件与未决项

- `schema_extract/tools/artifacts.py`
- `schema_extract/tools/publication.py`
- `schema_extract/tools/inputs.py`
- `schema_extract/tools/reading.py`
- `schema_extract/tools/runner.py`：仅授权的一个调用关键字。
- `schema_extract/tools/tests/test_artifact_transactions.py`
- `schema_extract/tools/tests/test_source_gaps.py`
- 本报告。

没有本轮已确认而未处理的问题。未改 common、固定 prompt、正式字段规则、Template、案例、Kernel、Wiki、archive 或既有报告，未提交 commit。验证限于列出的离线行为，不代表完整提取或真实模型质量已验收。报告完成后停止写入。
