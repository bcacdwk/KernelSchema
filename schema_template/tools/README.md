# Template 工具

日常作者使用 [check_yaml.py](../check_yaml.py)，填写设计和处理反馈见[模板说明](../README.md#4-检查与处理反馈)。默认对整套任务同步规范化并检查；`--read-only` 预览候选，`--all` 仅扩展请求的图范围。

工具实现当前 DOC0–6 的唯一字段和语法。检查 API 保持只读；规范化库只生成候选，CLI 负责安全写回和 PNG 发布。正式案例、Extract Runner 和 GPU 任务不属于维护回归输入。

## 回归与修改

需要 Python 3.10+ 和 PyYAML，PNG 另需 Graphviz 的 `dot`。从仓库根目录运行：

```sh
python3 -m pip install -r schema_template/tools/requirements.txt
python3 -B -m unittest discover -s schema_template/tools/tests -t schema_template -v
```

合成回归覆盖来源解析、规范化与引用同步、结构和预算、关键静态关系、事件图、只读接口以及文件发布安全。规则只表达可检查的设计要求，不能把有限图的求值范围变成额外 schema 语法限制。

| 模块 | 职责 |
| --- | --- |
| `yaml_source.py` | 精确值、源节点与位置、anchor/alias/merge、摘要；显式启用的窄范围缺引号及文本、int/bool 类型恢复 |
| `rules/structure.py` | 当前结构；同源导出 `CANONICAL_FIELDS / STRING_FIELDS / CANONICAL_VALUES` 及纯 int/bool 叶的 `LITERAL_TYPES`，不维护旧名别名 |
| `normalization.py` | 全任务定义和消费者作用域、字段与名称/引用的同步计划；返回经过源保真核对的候选 |
| `yaml_format.py` | 应用授权源编辑和保守句号清理，统一引号与布局，核对源图、精确值和幂等 |
| `text_policy.py`、`rules/` | 按 raw 定义检查预算、字段、声明、符号、硬件、调度和流水线关系 |
| `design_resolution.py` | 整区 `from` DAG、使用节点绑定、独立 raw/effective 视图和来源身份 |
| `schema.py` | 组合只读规范化/校验入口；任务索引和本地仓库定位检查与纯快照 API 分开 |
| `pipeline_syntax.py`、`pipeline/` | 动作、控制与事件语法；有限展开、依赖图、渲染与 PNG 发布 |

字段规则接收 `(value, path, errors)`，关联规则使用 `CheckContext`。路径通过 `field_path` 保留特殊具名键，缺项使用 `missing=True`；规则不读取文件、不打印、不决定退出码。预算和排版分支只在机器定义中维护，作者文档不复制维护表。

Storage 的具名 entry 使用固定两行 flow mapping：第一行是名字及 `meaning / depth / location / shape`，第二行是 `dtype / total_size / member / view`，比 entry 名缩进两格。可选字段缺失时直接略去，内容长短不改变分组。真实多行正文、整条 alias 及超出 YAML 简单键条件的名字保留必要表示；锚定义与使用的顺序优先。其他记录与正文布局不受该规则影响。

## 只读 Python API

以 `schema_template` 为导入根使用 `tools`。`validate_*` 检查传入或严格读取的规范快照，不自动改写或恢复内容。需要自动规范化时显式调用下列计划入口，再决定检查候选或交给 CLI 写回。

| API | 输入与结果 |
| --- | --- |
| `yaml_source.load(path, name)`、`loads(text, name, *, path=...)` | 严格返回 `Document`，不自动修复、不执行自定义标签 |
| `normalization.load_for_normalization(path, name, *, skip_hardware=False)`、`loads_for_normalization(text, name, *, path=..., skip_hardware=False)` | 显式源恢复；返回 `LoadedDocument(document, source_sha256, corrections)`，摘要始终对应恢复前原文 |
| `normalization.normalize_documents(files, documents, *, canonical_fields=None, skip_hardware=False)` | 输入完整任务的 `Document` 或 `LoadedDocument`；零 I/O 返回 `NormalizedTask` |
| `schema.normalize_directory(directory, *, skip_hardware=False)` | 读取整套任务并返回同一规范化计划，不写文件 |
| `schema.validate_documents(files, documents, *, notices=None, source_hashes=None, skip_hardware=False)` | 零 I/O 校验结构、raw 预算、`from` 与有效关联，返回 `ResolvedTask` |
| `schema.validate_file(path, ...)`、`validate_directory(directory, ...)`、`validate_resolved_file(path, ...)` | 严格读取完整任务，包含 dispatch 与可确定的本地仓库检查；分别返回 `(directory, effective_task)`、`effective_task`、`(directory, ResolvedTask)` |
| `schema.validate_dispatch(files)`、`validate_local_repositories(files, documents, *, skip_hardware=False)` | 独立文件检查，不重读 YAML；普通 CLI 和完整文件 API 自动调用 |
| `schema.validate_standalone_file(path, ...)` | 检查自带 Contract 和全部本地 Design 的单文件，保留节点 ID，不读兄弟 YAML 或 dispatch |
| `schema.resolve_directory(directory)`、`effective_yaml(resolved, node)` | 草稿展开与独立阅读视图；展开成功不代表完整校验通过 |
| `rules.validate_algorithm(algorithm)` | 零 I/O 检查算法区的结构、关系与值预算，返回诊断列表 |
| `text_policy.check_document(document, *, skip_hardware=False)`、`check_values(value, path='root', *, scope='document', skip_hardware=False)` | raw 源/值预算和样式诊断；不要传有效展开副本重复计数，算法映射使用 `scope='algorithm'` |
| `yaml_format.format_document(document, *, edits=(), corrections=None, skip_hardware=False)` | 只返回保真候选字符串；不承担跨任务绑定规划或写回 |
| `pipeline.configuration(parameters, worktiles=1)`、`pipeline.analyze(plan, config)` | 用显式整数样本分析已校验的 effective 设计；零 I/O 返回各 step 的 `StepGraph`，完整环记录在 `cycle`，未完成或确定的引用错误抛出异常 |

`NormalizedTask` 含 `files / documents / candidates / source_hashes / corrections`。`documents` 是全任务最终候选，`candidates` 只含变化文件；`source_hashes` 是原始字节摘要，不能用候选 `Document.source_sha256` 代替写回比较依据。候选中的位置属于候选文本，展示时必须标明尚未写入。`corrections` 是计划修改，写回成功后才能报告为已应用。

`Document.source_sha256` 对应其实际解析的 UTF-8 文本，`positions` 是该文本的位置；其余源节点资料服务于保真核对。小数直接构造 `Decimal`，日期与 `on/off/yes` 保持字符串。类型恢复仅在全部用途要求同一已知类型时调整引号：纯文本字段保留非 null 原始 lexeme，纯 int/bool 叶去除可直接解析的字面量引号。union、null、显式 tag、含转义的引号文本及混用 alias 保持原类型；重复显式键、递归 alias、非法标签和多文档输入仍拒绝。

`ResolvedTask` 保留 raw/effective 副本、provider 与完整来源链；来源身份含 root、自身和各区实际来源文件。共享区在每个 consumer 的有效上下文校验，错误回指真实 provider 和引用位置。独立阅读视图另存，保留 provenance，不作为 raw 节点读回创作目录。

单 step 入口是 `pipeline.build_step(step, definition, worker, parameters, worktiles=1, *, dynamic_worktiles=False, event_calls=())`。`event_calls` 每项为 `(本 step 的 Event 调用字符串, 来源字段路径)`，例如 `('ready(2)', 'DESIGN.Work_schedule.Flow.Launch[1].start_after')`。`analyze` 自动收集 `Flow.Launch[].start_after` 和 `Flow.Handoff.*.ready/wait_at`；`StepGraph.events` 只记录这些实际调用得到的具体事件集合。未调用的 Event 只做静态检查；缺少样本取值或表达式无法求值时返回 `ExpansionError`，具体调用指向缺失动作时返回 `GraphError`。图仍限于各 step 的局部事件，不增加跨 step 边。

## CLI 与发布边界

CLI 顺序为读取/可证源恢复、整任务规范化计划、候选保真检查、安全写回并严格重读，随后执行字段、预算、引用、关联、dispatch 和所请求的图。预算与普通语义错误不阻止独立的安全修正；碰撞、损坏或无法证明的共享引用变化不给出可写候选。节点文件名、`Dispatch.id` 和 `from` 来源值不改名。

变化文件先全部暂存并保留权限，每次替换前核对完整来源摘要；写入失败时只恢复仍由本次写入持有的文件，保留外部新修改。无变化不重写。`--read-only` 校验候选且不落盘；`--standalone` 与 `--show-effective` 仍为独立的严格只读请求，不暗中规范化。

warning 使用 `Diagnostic.severity` 或 `notices` 返回，规范化修改按 `field / identifier+reference / quote / period / layout` 汇总；未应用的只读差异返回 `1`。图操作遵循模板说明的退出码和有限样本边界，全部请求分析完成后才发布 PNG，来源漂移或失败保留旧图。

## 硬件占位豁免

`skip_hardware=True` 或 CLI 的 `--skip-hardware` 显式跳过 `CONTRACT.Hardware_specification` 的结构、预算、硬件关系和依赖硬件的校验。规范化保留该区的值，不恢复其文本类型或改写登记身份；源布局仍可统一。其他字段、成员、索引和动作继续检查，文件所在目录不自动产生豁免；需要不写入时同时使用 `--read-only`。
