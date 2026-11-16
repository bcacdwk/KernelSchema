# 自动规范化：实现与验证

`check_yaml.py FILE` 对同目录整套任务统一生成并应用规范化候选，再检查字段、预算、引用、关系和 dispatch。输入节点只决定默认图对象，`--all` 扩展请求的图范围。普通作者填写设计后运行同一个入口，复核修改 warning，精简超长正文并处理需要判断的问题。

## 实现边界

- `normalization.load_for_normalization` 显式恢复可证明的缺引号与标量引号类型，保存恢复前真实源摘要；普通 `yaml_source.load/loads` 保持严格。纯 int/bool 叶由结构规则的 `LITERAL_TYPES` 标记，去引号必须符合相同 YAML core 规则、单行原 lexeme 和全部 alias/merge 用途；null、显式 tag、空白/转义内容及含字符串/表达式的联合类型不转换，取值范围继续独立检查。
- `normalize_documents` 以结构检查同源的当前字段定义规划整个任务，包括共享区域的实际消费者。它只返回全任务 `documents`、变化 `candidates`、原始 `source_hashes` 和修改 `corrections`，没有文件写入。
- Storage entry 使用固定两行：名字及 `meaning / depth / location / shape` 在首行，`dtype / total_size / member / view` 在第二行；缺少可选字段时略去，内容长度不改变分组。view 的多个具名条目及合法嵌套属性保持在记录内，真实多行正文和锚定义顺序保留必要表示。
- CLI 先核对全部规范化候选，再暂存变化文件，核对整套来源后逐个原子替换。写入、严格重读和最终摘要核对共享恢复边界；失败时用原始 bytes 恢复仍为本次候选的已写文件及 mtime，保留外部新修改并报告恢复结果。最终诊断、有效身份和 PNG 使用实际字节与位置；后续字段、预算和关系检查独立进行。
- 数学形参沿可证明的普通运算符续行同步，下一定义和独立量词保持各自作用域；需要改名但无法界定的尾部使用会明确拒绝。已绑定属性链根随声明改名，成员、外部命名空间与字符串保留拼写。
- 只读 CLI 预览并校验候选，修改及其位置明确标为未写入；`validate_*` 继续检查规范快照，不自动恢复或规范化。`validate_documents` 零 I/O，dispatch 和本地仓库检查分开。
- 不改文件名、`Dispatch.id` 或 `from` 来源值；不存在旧词别名、迁移器或历史版本分支。碰撞、歧义和损坏不会产生可写的半套结果；超预算不阻止已经可证的独立格式修正，也不自动删正文。

warning 按 `field / identifier+reference / quote / period / layout` 汇总有效修改，声明与引用使用 before→after；无变化不 warning、不重写。保留原文件权限，未变化文件的 mtime 保持不变。特殊 `--standalone` 与 `--show-effective` 请求仍严格只读。

字段决定见 [FIELD_DECISIONS](FIELD_DECISIONS.md)。三个 Design 区域统一 `Thesis`，Arithmetic 使用 `Approx / Accumulate`，Ledger 使用 `Baseline / Gain / Cost`；Tile_reduction、Preassigned、Handoff、Index、noun 的既定属性及整区 `from` 继续小写。命名沿结构元数据传播到源恢复、预算与格式化，只发布一套拼写。

## 验证

维护测试只使用 `schema_template/tools/tests` 的合成输入，覆盖结构与预算、来源/独立视图、作用域和事件图、全任务同步、源恢复、只读与文件发布安全。命令为：

```sh
/opt/venv/python3.13.12/bin/python3 -B -m unittest discover -s schema_template/tools/tests -t schema_template -v
```

Storage 与最终字段拼写新增八项集中回归，覆盖固定分组、无 view 与多 view、必要引号、长正文、精确小数与共享节点、无关 anchor、view 单逻辑行块标量、原文不变、完整任务的只读与写回候选一致及版本不变。实施边界见 [STORAGE_LAYOUT_IMPL](STORAGE_LAYOUT_IMPL.md)，独立验收见 [STORAGE_LAYOUT_REVIEW](STORAGE_LAYOUT_REVIEW.md)。

完整套件 **113/113 通过（21.654 秒）**；40 个 Python 文件通过 Python 3.10 grammar AST 检查，`git diff --check` 通过。实际测试环境为 Python 3.13.12，完整日志 `/tmp/kernel-schema-storage-review-suite.log`。

已有 R1–R4 审修结果见 [REVIEW](REVIEW.md) 与 [FIXES](FIXES.md)。本次布局与命名改动在此基础上独立验收，不以已有审修结论代替新增部分的复核。

正式 examples、旧展示与统计、新版 display_demo、wiki、Extract 执行器和运行状态均未作为此次实施的更新对象。

## 后续交接

Storage 与最终命名已通过新的 xhigh 独立验收，可由原 demo 作者更新现有新版展示。展示允许 placeholder，按字段覆盖和排版验收，不宣称算法或引用正确。
