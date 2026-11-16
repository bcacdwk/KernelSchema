# R3 集成交接

本轮已将新的全任务规范化库接入唯一作者 CLI 和显式只读计划入口，完成当前字段的维护测试与作者文案调整。代码未提交；既有未提交工作完整保留。

## 接手位置

- `schema_template/check_yaml.py`：默认整任务规范化；`--all` 仅扩大图范围。读取/恢复后先获得验证过的整套候选，再暂存、核对原始摘要、替换并重读实际文件，之后执行结构、预算、from、关系、dispatch 与请求图。
- `tools/schema.py`：新增 `normalize_directory` 并导出规范化库入口；`validate_*` 保持严格只读，`validate_documents` 零 I/O。候选的源摘要和位置属于候选，原始 `source_hashes` 专用于写前比较。
- warning 按五类汇总，给出字段位置与声明/引用变化；预览显示 `planned` 和 `normalization preview; not written`，真正写回后才显示 `applied`。无变化不 warning、不重写。
- 全部变化文件先暂存，逐次替换前复核任务来源；失败时只撤回仍等于本次候选的已写文件，并恢复其原 mtime，保留外部新修改。写后真实文档供诊断和图来源身份使用。
- `tools/tests` 更新为现行字段；新 `test_normalization.py` 覆盖全任务/from 的 MAIN 和名称/引用同步、API 原始摘要与零 I/O、只读候选定位、混合缺引号、碰撞/null/旧词错误、保留 id/from 值、skip_hardware、alias/精确值与写入失败恢复。
- `schema_template/README.md`、`tools/README.md` 及根导航/Design/Extract 的检查说明集中表述工具职责；删去逐处大小写、引号、行数和预算表的作者负担。`EXECUTOR_ALIGNMENT` 只更新接口事实；CHANGELOG 增加简短条目，末尾用户草稿原样保留。

正常的缺引号恢复、规范化与共享作用域细节由上一轮核心代理在 `normalization.py / yaml_source.py / yaml_format.py` 完成。本轮未另建规划框架，未改正式 examples、wiki、Extract 执行代码或运行状态，未运行真实提取或 GPU。

## 验证证据

规范化核心冻结后只运行一轮完整维护 suite：

```sh
/opt/venv/python3.13.12/bin/python3 -B -m unittest discover -s schema_template/tools/tests -t schema_template -v
```

93 项中 92 项通过。唯一失败是新碰撞测试的正则仅接受 `collision/collid/ambiguous`，实际诊断正确指出 `fixed-field normalization would merge ...`，所有源均保持不变。只放宽这一句诊断断言，随后运行：

```sh
PYTHONPATH=schema_template /opt/venv/python3.13.12/bin/python3 -B -m unittest tools.tests.test_normalization.NormalizationTests.test_collisions_and_unprovable_damage_leave_every_source_untouched -v
```

定向 1 项通过。日志：`/tmp/kernel-schema-r3-suite.log`、`/tmp/kernel-schema-r3-targeted.log`。该失败未导致实现改动，未重复全套。另已完成 12 个入口/测试 Python 文件 AST 检查与 `git diff --check`。

## 后续审修

由新的 xhigh 审阅者检查完整当前实现，优先复核定义/消费者闭包、共享 scalar 的组合授权、恢复前真实摘要、只读候选位置、写回失败和图发布边界。审阅者给出具体复现，再由新的 MAX 修复并做针对性验证，交回同一 xhigh 复核。之后再由专门代理制作新的展示 demo。

[FIELD_DECISIONS](FIELD_DECISIONS.md) 中 A/B/C 是额外建议，尚未采纳；当前字段继续唯一现行拼写，等待用户明确回复，不阻本轮已授权集成验收。此处没有将它们当作已批准更名，也没有旧词运行时兼容。
