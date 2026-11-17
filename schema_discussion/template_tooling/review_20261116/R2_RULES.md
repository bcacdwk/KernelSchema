# R2 规则与引用修复

基线：`0778456`；2026-11-16。依据当前 DOC0、DOC1、DOC2、DOC3、DOC6 落实 R1-C1–C4，未增加字段、取值预算或历史兼容规则。

## 修复

| 项目 | 具体行为 | 文件 |
| --- | --- | --- |
| C1 · Item 的已知路径与身份 | 裸名称和限定名称先匹配完整登记名，再沿已知数学对象检查属性。`x.no_such_property` 与其限定形式均报错；`x.dtype` 与对应限定属性不能重复。已知属性不再进入开放目标集合，Feedback 因而不能借该分支对应到另一个属性。内部身份区分操作名、值及属性、发布路径和开放语义目标；完整操作名 `x.dtype` 与显式输入属性仍可分别引用。 | `tools/rules/symbols.py`、`tools/tests/test_schema.py` |
| C2 · 被引用的重复名称 | Compute/Rewrite 名称保留候选数量；Item、结构化弱引用及 Math 的操作绑定在实际命中重名时报告歧义。未引用的重复标题保持原边界，不新增全局唯一性要求。 | 同上 |
| C3 · dispatch 最小结构 | 识别 DOC0 的「设计 / 具体机制与保留理由 / 关系」三列表格，全部当前 YAML 必须在数据行的设计列登记。表外、关系列和后续标题中的链接不补登记。变体必须有名称及「参数 / 局部 / 组合」类型，继续检查直接完整基线链接。 | `tools/rules/dispatch.py`、`tools/tests/test_dispatch.py` |
| C4 · 硬件资料文件 | 本地修订查询按路径缓存 Git 对象类型；明确为 tree 的 `Wiki.Hardware` 报错，包括仓库根目录。普通 Wiki 入口仍可引用目录；以提交树为准，不以当前 checkout 或文件扩展名猜测。 | `tools/rules/values.py`、`tools/tests/test_schema.py` |

Math 的 shape/dtype 目标继续由 `descriptor_properties` 检查存在性及属性依赖，删除 `math` 中同一目标的第二次解析；定义顺序仍在 `math` 检查，binding 仍检查整体对象、环和已知属性冲突。对应缺失引用现在只产生一条该属性的诊断，没有增加通用去重器。

## 验证

使用 `/opt/venv/python3.13.12/bin/python3 -B`，以 `schema_template` 为导入根：

- `python3 -B -m unittest tools.tests.test_schema tools.tests.test_dispatch`：最终 29 项通过，1.643 秒。回归覆盖裸/限定属性、同一目标重复、完整名称优先、Feedback、仅被引用重名、Math binding、表格内登记、标题类型及合句正文；已有 reference link、行内 code 标签、围栏与外链测试继续通过。
- `python3 -B -m unittest tools.tests.test_design_resolution`：8 项通过，0.608 秒；来源展开与消费者绑定保持通过。
- `minimal` fixture 和 `auto_normalization/display_demo` 的 dispatch 与严格 `validate_directory` 只读检查均通过；没有改动 fixture 或展示 demo。
- 精确复现 `Wiki.Hardware: KernelSchema@0778456:schema_docs`：纯快照校验保持不读仓库，独立 `validate_local_repositories` 在该字段报告目录错误。已有提交中的资料文件、普通 Programming 目录、占位与 `skip_hardware` 回归通过。
- 与本轮 normalization 改动的交界探针：`Item: X.SHAPE` 一次成为 `x.shape`，随后 `validate_documents` 通过。
- `git diff --check` 通过。未重复全套测试或扩大图验证。

## 边界与交接

- 公式和 Strategy 中未结构化发布的合法目标继续保留人工解析；没有从正文提取符号表。变体正文可合成一句，不强制三个标签，也不判断机制、条件或保留理由是否充分。
- Wiki 文件的内容是否对应具体型号仍需审阅；只检查能够在本地明确确定的修订、路径和对象类型，不联网。
- 四份正式案例、`schema_wiki/kernel_wiki`、Extract 脚本与状态、旧 demo、历史研究及 CHANGELOG 均未修改。保护案例已有旧 dispatch 表式，按本轮范围延期，不以当前检查结果宣称其完成迁移。
- 未修改规范文档或 tools README；无需本轮展示 demo 调整。后续独立复核继续由新代理进行。
