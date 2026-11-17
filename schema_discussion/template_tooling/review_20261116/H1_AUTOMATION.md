# H1：自动化与入口精简

基线 `0778456`，精简前对照 `/tmp/schema-template-before-prune-tp4svoxx`。阅读 R1–R3 自动化报告与本轮分工后，检查规范化、源恢复、排版、文本策略、schema、CLI 和诊断模块。本轮实际代码只修改 `tools/text_policy.py`：删除 39 行、加入 13 行，净减 26 行。没有修改测试、作者 README、规则/图文件或受保护材料，没有提交 git。

## 实际删改与等价依据

1. 删除无调用的 `expects_text`，同时移除只供它使用的 `Budget.string_only` 和预算构造参数。仓库检索确认没有代码或当前文档 API 消费者；缺引号与标量类型恢复早已由 `rules.structure` 的类型元数据决定。对照快照逐项比较 **80 条预算**的路径、字符数、行数、prose 与 prefix，全部相同；预算检查继续只处理实际字符串。
2. `sentence_period_changes` 复用 `yaml_source.scalar_contexts`，删除自己的节点/alias/merge 递归与字段路径重建。仍按同一源节点分组、按完整用途去重；只有全部用途属于可处理的说明字段才改句尾。key、非说明字段、被跳过的硬件用途和被覆盖的 merge 默认值仍阻止不安全修改。诊断直接使用已有 `ScalarContext.path`，保持特殊具名键的精确路径。

两个 fixture 与三个合成样本，分别按两种硬件模式，对照旧实现的句号计划及全部文本诊断：**5 × 2 组全部一致**。合成样本包含 scalar 跨说明/非说明字段复用、硬件跨区复用和被显式值覆盖的 merge 默认值。

## 验证

使用 `/opt/venv/python3.13.12/bin/python3 -B`；未重跑完整套件、未新增镜像测试。

- **12 个受影响现有测试通过，5.048 秒**：普通/不确定句尾、scalar/collection 共享用途、skip_hardware 共享用途、Unicode/逻辑行预算、range/Item/Feedback 预算、Storage 的 tag/alias/merge/Decimal 保真、纯快照零 I/O、整任务共享 alias、类型恢复混用保护、真实 CLI 句尾先于预算，以及默认 CLI 的权限/版本/无变化 mtime。
- 上述快照预算/诊断行为对照通过；`git diff --check` 通过。

## 有实际用途的保留项

- **不直接跳过 LoadedDocument 的再次标量恢复。** 包装不记录读入时的 `skip_hardware`。真实探针：以 `skip_hardware=True` 读入 `Hardware_specification.Device: 123`，其值仍为 int；随后默认 `normalize_documents` 必须恢复为字符串 `'123'`，并继续携带原文摘要。该路径已验证。为省一次扫描而添加状态/API，超出本轮有价值的精简；已与主代理同步保留。
- `_math_tokens` 的 AST、开放文本平衡片段和 exact 回退处理不同的绑定证据；分别保护熟悉量词、开放记法及无法确定局部范围的文本。没有发现可直接删除的分支，本轮不重写它、不扩大数学解析语言。下一轮若进一步收拢，应继续以 R3 多行冲突/安全开放量词对照为门槛。
- 排版的值比较与源节点图比较不互相替代：后者保护标签、lexeme、anchor/alias 复用，前者还核对 merge 取值及重命名后的结果。CLI 写回前后的摘要检查、失败恢复与外部修改保护也各有现有回归证据，保留。

本次删除的是内部死元数据和重复遍历；现有公开入口、当前作者说明及固定字段/预算/Storage 布局约定无需调整。
