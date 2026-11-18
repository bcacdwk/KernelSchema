# Template 工具

日常编写与检查使用 [check_yaml.py](../check_yaml.py)，操作步骤见[模板说明](../README.md#4-检查与处理反馈)。本目录提供实现当前 DOC0–6 的只读校验与规范化规划库；文件写回与 PNG 发布仅由 CLI 执行。正式案例、Extract Runner 和 GPU 任务不纳入本包维护回归。

## 回归与修改

需要 Python 3.10+ 和 PyYAML，生成 PNG 另需 Graphviz `dot`。从仓库根目录运行：

```sh
python3 -m pip install -r schema_template/tools/requirements.txt
python3 -B -m unittest discover -s schema_template/tools/tests -t schema_template -v
```

合成回归覆盖源码解析、规范化与引用同步、结构与文本预算、静态关系、事件图、只读接口及文件发布安全。校验规则只表达可机械核对的设计约束，不把有限图求值器的范围变成额外的 schema 语法限制。

| 模块 | 职责 |
| --- | --- |
| `yaml_source.py` | 精确值、源节点与位置、anchor/alias/merge、原文摘要；按需启用的窄范围缺引号及文本、int/bool 类型恢复 |
| `rules/structure.py` | 当前字段结构；同源导出 `CANONICAL_FIELDS / STRING_FIELDS / CANONICAL_VALUES` 与纯 int/bool 叶的 `LITERAL_TYPES`，不维护旧字段别名 |
| `normalization.py` | 规划全任务定义、消费者作用域及字段/名称/引用同步，返回经源保真核对的候选 |
| `yaml_format.py` | 应用授权源编辑与保守句末句号清理，统一引号和布局，核对源图、精确值与幂等性 |
| `text_policy.py`、`rules/` | 按 raw 定义检查文本预算、字段、声明、符号、硬件、调度和流水线关系 |
| `design_resolution.py` | 解析整区 `from` DAG 与使用节点绑定，构造独立 raw/effective 视图与来源身份 |
| `schema.py` | 组合只读规范化与校验入口；将 `dispatch.md` 和本地仓库定位检查与纯内存快照 API 分离 |
| `pipeline_syntax.py`、`pipeline/` | 动作、控制流与事件语法；有限样本展开、依赖图、渲染与 PNG 发布 |

字段规则接收 `(value, path, errors)`，关联规则接收 `CheckContext`。路径由 `field_path` 拼接以保留特殊具名键，缺项传 `missing=True`。规则函数不读文件、不打印、不决定退出码；字符数与行数预算只在代码中维护，不在作者文档重抄整表。

`Storage` 的具名条目采用固定两行 flow mapping：首行写条目名及 `meaning / depth / location / shape`，次行缩进两格写 `dtype / total_size / member / view`；缺失的可选键直接略去，不随内容长短改变分组。多行正文、整条 alias 及超出 YAML 简单键范围的名称保留必要写法，并优先保持锚点先定义后使用；其他映射与正文布局不受此限。

## 只读 Python API

以 `schema_template` 为导入根调用 `tools`。`validate_*` 只检查严格读取或已传入的规范快照，不自动修复或改写内容；需要规范化时先显式调用规划入口，再决定校验候选或交给 CLI 写回。

| API | 输入与结果 |
| --- | --- |
| `yaml_source.load(path, name)`、`loads(text, name, *, path=...)` | 严格返回 `Document`，不自动修复、不执行自定义标签 |
| `normalization.load_for_normalization(path, name, *, skip_hardware=False)`、`loads_for_normalization(text, name, *, path=..., skip_hardware=False)` | 显式源恢复；返回 `LoadedDocument(document, source_sha256, corrections)`，摘要始终对应恢复前的原文 |
| `normalization.normalize_documents(files, documents, *, canonical_fields=None, skip_hardware=False)` | 输入完整任务的 `Document` 或 `LoadedDocument`；零 I/O 返回 `NormalizedTask` |
| `schema.normalize_directory(directory, *, skip_hardware=False)` | 读取整套任务并返回同一规范化计划，不写文件 |
| `schema.validate_documents(files, documents, *, notices=None, source_hashes=None, skip_hardware=False)` | 零 I/O 校验结构、raw 预算、`from` 与有效关联，返回 `ResolvedTask` |
| `schema.validate_file(path, ...)`、`validate_directory(directory, ...)`、`validate_resolved_file(path, ...)` | 严格读取完整任务并执行 dispatch 与本地仓库检查；分别返回 `(directory, effective_task)`、`effective_task`、`(directory, ResolvedTask)` |
| `schema.validate_dispatch(files)`、`validate_local_repositories(files, documents, *, skip_hardware=False)` | 独立文件检查，不重读 YAML；普通 CLI 与完整文件 API 自动调用 |
| `schema.validate_standalone_file(path, ...)` | 只读校验自带 Contract 与全部本地 Design 的单文件，保留节点 ID，不读兄弟 YAML 或 dispatch |
| `schema.resolve_directory(directory)`、`effective_yaml(resolved, node)` | 展开草稿与独立阅读视图；展开成功不代表通过完整校验 |
| `rules.validate_algorithm(algorithm)` | 零 I/O 检查算法区的结构、关系与值预算，返回诊断列表 |
| `text_policy.check_document(document, *, skip_hardware=False)`、`check_values(value, path='root', *, scope='document', skip_hardware=False)` | raw 源/值预算与样式诊断；勿传入展开副本重复计数，单独检查算法映射时传 `scope='algorithm'` |
| `yaml_format.format_document(document, *, edits=(), corrections=None, skip_hardware=False)` | 仅返回保真候选字符串，不规划跨文件同步，也不写文件 |
| `pipeline.configuration(parameters, worktiles=1)`、`pipeline.analyze(plan, config)` | 按显式整数样本分析已校验的 effective 设计；零 I/O 返回各 step 的 `StepGraph`，完整环记入 `cycle`，未完成展开或确定引用错误抛出异常 |

`NormalizedTask` 包含 `files / documents / candidates / source_hashes / corrections`：`documents` 为整套最终候选，`candidates` 仅含发生变化的文本；`source_hashes` 记录磁盘原始字节摘要，落盘前比较不得用候选 `Document.source_sha256` 替代。候选行号对应未落盘文本，展示时须注明预览状态；`corrections` 在写回成功后才能报为已应用。

`Document.source_sha256` 与 `positions` 对应实际解析的 UTF-8 文本，其余源节点信息用于保真核对。小数解析为 `Decimal`，日期和 `on/off/yes` 保持字符串。类型恢复仅在所有用途都要求同一已知类型时调整引号：纯文本字段保留非 null 原始字面量，纯 int/bool 叶去掉可直接解析的包裹引号；union、null、显式 tag、带转义的引号字符串及混用 alias 保持原类型，重复键、递归 alias、非法标签和多文档输入仍直接拒绝。

`ResolvedTask` 同时保留 raw 与 effective 副本、provider 及完整引用链；来源身份涵盖 root、本节点与三区实际来源文件。共享区在每个使用节点的有效上下文中分别校验，诊断回指真实 provider 与引用位置。独立阅读视图保存在创作目录之外并携带 provenance，不作为 raw 节点读回。

单 step 构图入口为 `pipeline.build_step(step, definition, worker, parameters, worktiles=1, *, dynamic_worktiles=False, event_calls=(), member_regions=None)`。`event_calls` 元素为 `(本 step 的 Event 调用字符串, 来源字段路径)`，如 `('ready(2)', 'DESIGN.Work_schedule.Flow.Launch[1].start_after')`。`analyze` 自动收集 `Flow.Launch[].start_after` 与 `Flow.Handoff.*.ready/wait_at`，`StepGraph.events` 只记录实际调用解析出的事件集合；未调用的 Event 仅做静态核对。缺少样本参数或表达式无法求值时抛出 `ExpansionError`，调用落到缺失动作时抛出 `GraphError`。事件图只包含各 step 内部的局部边，不跨 step 连边。

`analyze` 自动传入 `Mapping.Member`（直接调 `build_step` 时可通过 `member_regions` 传入）；只有可证明宽度为 1 的半开区域才绑定成员数学坐标，多元素或未知区域不猜代表值。局部循环索引按词法作用域遮蔽同名坐标，其范围端点仍取外层值。动作展开、等待窗口和 Event 解析均保留各成员上下文：`Waiting` 综合两端声明域枚举索引上下文，在当前目标成员求值地址表达式，并对每个显式前驱及其所选成员独立检查窗口与动作存在性（移位越界只跳过对应前驱实例，其余有效前驱照常向目标累加边）；Event 调用中显式传入的实参优先。

有限求值器支持整数异或 `^` 与左移 `<<`（位移量限 `0..40`，求值结果绝对值上限 `10^12`）；这些求值器限制不收窄 schema 本身的开放整数表达式。

## CLI 与发布边界

CLI 依次执行：读取与可证明的源恢复、整套任务规范化规划、候选保真核对、安全写回与严格重读，随后检查字段、预算、`from`、关联、`dispatch.md` 及所请求的有限图。预算和普通语义错误不阻止独立的安全格式修正；名称碰撞、YAML 损坏或无法证明安全的共享引用改动则拒绝生成可写候选。节点文件名、`Dispatch.id` 与 `from` 目标名保持不变。

有变化的文件先在同目录暂存并保留原权限，每次原子替换前核对全部 `source_hashes`；中途失败时只回滚仍由本次写入持有的文件，不覆盖外部并发新改动；无变化时不重写文件。`--read-only` 校验规范化候选而不落盘；`--standalone` 与 `--show-effective` 独立执行严格只读处理，不触发规范化。

warning 通过 `Diagnostic.severity` 或 `notices` 返回，规范化改动按 `field / identifier+reference / quote / period / layout` 五类汇总；`--read-only` 下存在未应用的规范化差异时返回退出码 `1`。请求的所有节点全部完成图分析且来源摘要未变后，才写入同名 PNG；中途失败或来源漂移均保留旧图。

## 硬件占位豁免

传入 `skip_hardware=True` 或 CLI `--skip-hardware` 时，显式跳过 `CONTRACT.Hardware_specification` 的结构、预算及依赖硬件的关联检查。规范化原样保留该区的值，不恢复其文本类型，也不改写硬件登记名称，仅可统一源布局。其余分区、成员语法、索引与动作照常校验；文件所在目录不自动触发豁免，需避免写回时应同时加 `--read-only`。
