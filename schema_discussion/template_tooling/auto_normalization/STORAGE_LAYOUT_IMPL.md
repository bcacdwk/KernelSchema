# Storage 固定两行布局：实施交接

`yaml_format.py` 只为 `DESIGN.Tile_pipeline.Step.<STEP>.Storage.<name>` 设置两组固定字段，内容长度不参与布局选择：

| 行 | 字段顺序 |
| --- | --- |
| 第一行 | entry 名字、`meaning / depth / location / shape` |
| 第二行 | `dtype / total_size / member / view` |

第二行比名字缩进两格。可选字段缺失时略去，view 的多个条目仍在第二行内。例如：

```yaml
Storage:
  buffer: {meaning: Two scalar slots, depth: 2, location: register, shape: [],
    dtype: fp32, total_size: 8, view: {x(i): {slot_index: i%2}}}
```

Storage 登记表保持展开；只有明确的 Storage 属性与 view 记录可合入这两行。其他容器、记录、列表顺序与 `300/2` 正文规则不变。格式化经原有精确值、Decimal lexeme、tag、anchor、alias/merge 图与幂等检查；原 Document 不变。

必要边界：真实解码换行保留块布局；整个 entry 的 alias 保留 alias；merge 不展开或复制；固定排序会将 anchor 定义迁移到原 alias 字段时保留源顺序，若因此交错两个字段组则保留展开。无关或独立的 anchor 不阻止 Storage 固定排序。超出 YAML 简单键条件的名字采用必要的显式键表示。仅含一个字段组的未完成记录或 merge-only 记录不生成空行凑成两行。Storage 及 view 的块标量若解码后没有换行，可按同一固定布局输出。

已确认拼写从结构元数据进入字段、文本叶、源恢复和格式化；预算路径同步使用三个 `Thesis`、`Arithmetic.Approx / Accumulate`、`Ledger.Baseline / Gain / Cost`。Tile_reduction、Preassigned、Handoff、Index、noun 属性与整区 `from` 继续小写，正文中的 `ex2.approx` 等外部名称不变。

DOC5 的三个 Storage 文内示例已对应上述静态分组。正式 examples、两个展示目录、wiki、Extract 执行器与状态、CHANGELOG 末尾用户原稿均未修改。

验证入口：

```sh
/opt/venv/python3.13.12/bin/python3 -B -m unittest discover -s schema_template/tools/tests -t schema_template -v
```

集中回归在 `test_yaml.py`、`test_normalization.py`、`test_cli.py`，覆盖无 view、单／多 view、固定断点、长正文与必要引号、alias/merge/tag/Decimal、必要展开、原文与列表顺序保真、完整任务从引用节点预览／写回、Version 与重复运行不变。新增部分已通过独立验收，见 [STORAGE_LAYOUT_REVIEW](STORAGE_LAYOUT_REVIEW.md)；可由原 demo 作者更新展示。

实施验证：相关 81 项通过；完整套件 **111/111 通过（21.069 秒）**，日志 `/tmp/kernel-schema-storage-suite.log`；40 个 Python 文件通过 Python 3.10 grammar AST 检查，实际运行环境为 Python 3.13.12；`git diff --check` 通过。DOC5 三个 Storage 示例与格式器输出逐字一致，超过 YAML 简单键长度的显式键保真、幂等复现通过。
