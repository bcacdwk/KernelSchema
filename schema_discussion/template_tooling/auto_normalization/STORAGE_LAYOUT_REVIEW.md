# Storage 固定两行与最后命名：独立验收

结论：通过，可交回原 demo 作者更新 `display_demo/root.yaml`。

独立读取源码并复现后修正两处过度保守的展开：文档中的无关／独立 anchor 不再禁用 Storage 固定排序；view 内无逻辑换行的块标量进入固定第二行。只在排序会迁移 anchor 定义所属字段时保留原顺序，真实换行、整 entry alias、merge 和超长显式键继续保真。新增两项定向回归；alias 检查逐路径和使用次数核对，允许属性规范排序。

布局仍只有人工选定的两组字段，无长度估计、断点候选或双模式；其他记录没有扩大压缩范围。新 `Thesis`、`Approx / Accumulate`、`Baseline / Gain / Cost` 已沿结构元数据进入规范化、文本叶恢复和原预算路径；明确的小写属性及 `from` 保持。

验证结果：

- 完整套件 **113/113 通过（21.654 秒）**，日志 `/tmp/kernel-schema-storage-review-suite.log`。
- 40 个 Python 文件通过 Python 3.10 grammar AST 检查；实际执行环境为 Python 3.13.12。`git diff --check` 通过。
- 独立进程 CLI 从引用节点预览与默认写回使用相同候选；核对 warning、Version、权限、from、最终 source hash，以及重复执行的 bytes / mtime 不变。合成任务与日志位于 `/tmp/kernel-storage-review-6xlyh6w2/`。
- DOC5 三个 Storage 示例与格式器输出逐字一致；真实 view 换行及超过简单键上限的显式键保真、幂等复现通过。

本次验收只改 `yaml_format.py`、`test_yaml.py` 和当前交接文档；未写正式 examples、两个展示目录、wiki、Extract 或用户 CHANGELOG 草稿。源 Document、列表顺序、标量类型／词面和共享图仍由现有候选验证保护；没有改变预算数字或执行依赖语义。
