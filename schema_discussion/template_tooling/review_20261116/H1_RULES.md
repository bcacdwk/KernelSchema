# H1 · 规则与图代码精简

2026-11-16；基线 HEAD `0778456`。本轮对照前三轮完成后的快照 `/tmp/schema-template-before-prune-tp4svoxx`，只精简现行实现，不追加规则或功能。

## 删改与等价依据

| 文件 | 删改 | 为什么保持现行行为 |
| --- | --- | --- |
| `tools/pipeline_syntax.py` | 删除 `common_domains` | 全仓未发现运行消费者；仍保留 Waiting 使用的 `domain_choices`、`same_domain`。 |
| `tools/pipeline/actions.py` | 删除 `operands` 薄包装 | 现行 Template、Extract、测试及文档 API 均未消费。唯一调用在历史 `pipe_grammar/tools/check_trial_roots.py:25,173`；对应 `trial_roots/README.md:3` 明确该旧字段、检查器与通过记录不作为现行规范验收。历史文件原样保留，不为旧语法试验维持正式模块的兼容入口。共享 `Action.operands` 属性及当前 `arguments`/`instantiate` 均保留。 |
| `tools/pipeline/windows.py` | 删除 `missing` 中只写不读的 `checked` 集合 | 坐标处理由 `pending`、`outside`、`unavailable` 控制，删除该集合没有改变分支或坐标恢复。 |
| `tools/rules/pipeline.py` | 同一个 Pipe 只解析一次；`_pipe` 直接检查所得语法树 | 原先为收集局部索引解析一次，随后检查时再解析一次。现在缓存解析树和语法错误；仍先收集全部角色的局部索引，再按原顺序做 member、作用域、action family 等检查。语法错误仍在对应角色原有检查位置报告，没有省略空分支的声明检查。 |
| 同上 | `_domain` 去掉未使用的 `name` 形参，合并同一 list 定义的两处分支 | 区间与表达式检查的顺序、参数和诊断不变；flatten 检查保留。 |
| `tools/design_resolution.py` | `effective_document` 去掉一次性的 `_Budget().charge(...)` 包装 | `_units(plan)` 已在遍历时执行相同 `MAX_RESOLVED_UNITS` 上限；随后新建、初始计数为零的 `_Budget` 再检查同一数值没有作用。复制前的容量检查、错误类型和消息保留；整项解析的累计预算对象仍保留。 |

## 验证

以 `schema_template` 为导入根，使用 `/opt/venv/python3.13.12/bin/python3 -B`：

```text
python3 -B -m unittest tools.tests.test_schema tools.tests.test_dispatch \
  tools.tests.test_design_resolution tools.tests.test_pipeline_graph \
  tools.tests.test_pipeline_output
Ran 58 tests in 3.461s — OK
```

- 全仓调用检索已区分现行工具与明确归档的 pipe 试写检查器；现行 Template/Extract/DOC 中没有所删 helper 的调用或公开 API 承诺。
- 与精简前快照逐文件对照确认，本轮只修改上表五个代码文件；没有新增或改写测试。`git diff --check` 通过，未重复全套回归。

## 明确保留

- Item 完整名称优先、裸/限定身份一致、实际引用处的重名歧义和未引用重名边界原样保留。Math 属性的存在性/环检查、定义顺序与 binding 冲突属于不同职责，没有再次合并或恢复重复 `_target`。
- Waiting 的目标/前驱、固定/移位/空目标分支和精确坐标恢复仍有实际用途；Event 仅按 Flow 实际调用展开，`build_step(..., *, event_calls=())` 保持。没有恢复 `alias_contexts`、同名数据边或旧语法。
- 结构检查与图入口检查服务于不同调用层；不因检查内容近似而删除。`rules.validate_structure` 是当前导入消费者使用的重导出，并非无用 import。
- dispatch 三列登记与变体类型、仅本地明确 Hardware 目录拒绝、四时点、三模式、图未完成/错误分类及发布安全均沿用前轮实现。所有规则预算不变。

未修改正式文档、展示 demo、`schema_template/examples/`、wiki、Extract 脚本与状态、旧 text_limit_demo、既有研究报告或 CHANGELOG；未执行 commit/reset。
