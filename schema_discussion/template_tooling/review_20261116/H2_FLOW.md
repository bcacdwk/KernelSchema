# H2：自动化链路精简复核

基线 `0778456`；精简前快照 `/tmp/schema-template-before-prune-tp4svoxx`。先读 AGENTS 与 H1、R2/R3 自动化结论，以当前共享工作区继续复核 normalization → format → schema → CLI，以及源读取、恢复、诊断与两份 README。本轮没有重跑全套、添加测试或修改作者规则。

## 实际删改与等价依据

仅修改 `tools/normalization.py` 的 `_Scope._table`：删除没有调用者的 `target` 回调参数，以及 `target(key) if target is not None else None` 分支。仓库检索确认六处调用均未传入 `target`；它们原本总向 `_Names.add` 传 `target=None`，现在使用该方法已有的同值默认参数。声明大小写、重复声明处理与诊断完全不变。

`_Names.add`、`_Names.bind` 的 `target` 有真实调用：分别处理 step/role 的 MAIN 身份与事件局部作用域，因此保留。没有为减少行数继续改造其它辅助方法。

## 保留的职责与流程

- `yaml_source` 负责严格源解析和明确启用的窄恢复；`normalization` 负责整任务及各 consumer 的绑定；`yaml_format` 负责授权编辑与保真候选；`schema` 组合校验；CLI 负责落盘、最终重读和反馈。纯快照 API 与 dispatch、本地仓库等 I/O 检查的分离清楚，未发现应归并的重复入口。
- `LoadedDocument` 的再次类型恢复仍有必要：读入与计划阶段可以使用不同的 `skip_hardware`。不为跳过扫描新增记录状态或参数。
- normalization 的宽松 `from` 来源预解析与 `design_resolution` 的严格解析处于不同阶段；前者须在未知字段和普通语义错误之前完成安全名称计划，不能直接替换成后者。
- 三处路径匹配器分别服务规范化的大小写匹配、布局表的模式占位匹配、文本预算的严格字段匹配，规则并不完全相同。不引入通用匹配框架。
- 源图同构、精确值比较、二次排版幂等检查保护不同性质；类型恢复的图核对还允许唯一授权的 tag 变化，不与排版图检查合并。Math 的已识别作用域、开放文本切片与 exact 回退保持原实现。
- CLI 的原文摘要、暂存、逐次替换检查、最终磁盘重读与失败恢复有真实职责；只读差异与任务校验失败的状态也不能合并，否则会错误阻止候选图检查。保留当前无变化不写、mtime/权限恢复和外部修改保护。
- 两份 README 分别面向作者流程和库接口，当前规范化顺序、候选保真与语义校验的边界、窄类型恢复、只读模式和固定 Storage 两行说明一致。没有发现值得删除的重复要求或矛盾，保持不改。

## 验证与未决

使用 `/opt/venv/python3.13.12/bin/python3 -B`，两个受影响现有测试通过（1.228 秒）：

- `test_default_updates_provider_and_from_consumer_names_references_and_main`
- `test_skip_hardware_preserves_its_unfinished_values_through_cli_normalization`

第一项覆盖全任务声明、来源消费者、引用与 MAIN 改名，并检查再次执行不写；第二项覆盖被跳过硬件与其它规范化继续工作。`git diff --check` 通过。无新的实质缺陷或未决项。

本轮只删除私有辅助方法的死参数分支；H1 的 text_policy 精简未重做。没有修改 tests、README、rules、pipeline、design_resolution、正式 examples、wiki、Extract 状态、旧 demo 或历史要求，没有提交 git。
