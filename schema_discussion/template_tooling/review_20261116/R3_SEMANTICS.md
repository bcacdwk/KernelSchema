# R3 语义复核与 Event 调用修复

基线 `0778456`；2026-11-16。独立复核 R1 契约与规范化报告、R2 规则修复，对照 DOC0–6 的相关接口，重点走读静态 schedule/pipeline 规则、共享语法与有限事件图。本轮只修改 `tools/pipeline/__init__.py`、`tools/pipeline/graph.py`、`tools/tests/test_pipeline_graph.py`；README 的同步说明由本轮 B 负责。

## 修复：Event 按实际调用代入，不从 Pipe 猜形参域

DOC5「跨 step 交接」规定：调用 `snapshot_ready(k)` 时把形参代入事件列表，不捕获某个 role 的当前循环变量。原图实现却主动枚举所有 Event，并在形参与循环索引同名时借用该索引域；其他形参则从列表首项的动作窗口求逆。

最小反例：Pipe 为 `for(i=0:2): compute:x(i+1)`。`ready(i): ['{done MAIN[0]::compute:x(i)}']` 被错误展开到不存在的 `ready(0)`；仅将形参改成 `k` 就变成 `ready(1)/ready(2)` 并通过。固定事件放在列表首项时还会造成不必要的未完成；多个事件窗口不同时，列表顺序会决定猜出的调用域。这些结果均不能作为额外 schema 限制。

当前实现：

- `analyze` 从全部正式 Event 消费入口收集调用：`Flow.Launch[].start_after`、`Flow.Handoff.<value>.ready`、`.wait_at`，兼顾单值与列表。
- 每次调用直接求出实参，以独立的形参环境解析列表中的每个事件。不生成跨 step 边，不模拟工作区域匹配、可见性或跨 worker 活性。
- 删除整个 `alias_contexts` 推域/求逆分支及其 `recover` 导入。未调用的 Event 保留静态声明、签名、角色与成员检查，不为报告标签主动求值或生成实例。
- 调用缺少显式样本值或表达式不在有限求值范围内，返回 `ExpansionError`（未完成）。已知实参确实指向缺失动作，返回 `GraphError`（有限样本反例）。别名列表中的事件仍须全部存在，不能按实际事件交集删项。
- 内建 `kernel_done / launch_ready` 继续由 Flow 静态检查，不伪造本地动作节点。

公开低层 API 增加一个可选 keyword-only 参数，原有位置参数不变：

```python
build_step(step, definition, worker, parameters, worktiles=1,
           *, dynamic_worktiles=False, event_calls=())
```

`event_calls` 每项为 `(本 step 的调用字符串, 来源字段路径)`，例如 `('ready(2)', 'DESIGN.Work_schedule.Flow.Launch[1].start_after')`。默认空；直接调用 `build_step` 时不再主动枚举 Event。`analyze(plan, config)` 自动提供这些调用；`configuration` 不变。`StepGraph.events` 只报告本次实际调用的具体别名及其事件列表。纯快照/图分析仍不读写源文件。

## 已复核的重要边界

- R2 Item 修复仍先匹配完整登记名，再解析确定的对象属性；裸/限定同一对象具有同一身份。完整操作名 `x.dtype` 可以与显式 Input 属性分别指认；未被引用的重名 Compute/Rewrite 不因本轮而全局禁用。Feedback 不再把确定属性误当开放目标。
- dispatch 核对最低三列表格、设计列中的实际登记与变体类型；合句正文仍接受，不判断机制与理由质量。Hardware 核对本地指定提交中能确定的路径类型，普通资料目录与占位边界保持。
- 整区引用仍按节点 DAG、实际 provider/chain 及 consumer 当前定义绑定；没有增加第二套来源或引用身份。
- 四种显式 issue/done 组合、三种执行模式、跨 noun/verb/worktile 的 ordered 完成、serial 不排空其他异步及同名 noun 不生数据边保持。
- 跨局部循环小复现：生产者 `i=[0,3)`，消费者 `j=[0,4)`，引用 `x(j-1)` 得到 3 条合法前驱边及 1 次边界跳过；改为 `x(j)` 或 `x(99)` 都是确定的 `GraphError`。没有给开放表达式添加 Python 语法要求。
- 空工作小复现：每工作迭代数依次为 `(0,2)`、`(2,0)`、`(0,0)`，固定首项目标 `g[t,0]` 同时等待移位 `release(t-1)` 与固定 `release(0)`；分别保留 1/1/0 条去重后的合法边，边界跳过为 0/1/0。目标为空不生成等待，单个越界前驱不抹掉另一个合法前驱；循环外动作保持存在。
- Flow 仍检查归属、签名、启动环、共享 worker 的确定冲突、handoff 参与者和合法等待时点。未点名的消费者保持默认入口。实际 Event 调用展开不把局部无环图升级为跨 step 执行证明。

## 验证

使用 `/opt/venv/python3.13.12/bin/python3 -B`，导入根为 `schema_template`：

- `python3 -B -m unittest tools.tests.test_pipeline_graph tools.tests.test_pipeline_output`：21 项通过，0.957 秒。新增 3 项回归覆盖形参改名、固定首项及事件列表重排、不同声明窗口下的确定调用、三处正式调用入口、未用开放表达式、缺样本与已知缺失事件的分类；输出失败保旧测试继续通过。
- R2 独立定向复核 6 项通过，1.243 秒：`test_item_resolves_known_bare_properties_and_full_names_first`、`test_operation_names_must_be_unique_only_when_referenced`、`test_local_repository_uses_available_commit_not_the_checkout`、dispatch 最低表格及合句变体两项、`test_binding_uses_consumer_definition_and_diagnostics_use_provider_marks`。
- 上述跨局部循环、空工作组合采用进程内合成数据小复现，没有改动 fixture 或展示 demo。`git diff --check` 通过。未重复全套测试。

## 留给 high 与最终验收

- 不恢复为所有 Event 主动猜域的旧分支，不把事件列表的首项、交集或并集当作隐含调用域。复杂实际调用无法求值应保持未完成；跨 step 的工作匹配与活性继续在既定范围外。
- 本轮没有新增用户待决的 schema 规则。DOC5 已足以决定按调用代入；规范无需为工具增加 Event 域字段。新的可选低层 API 与 `StepGraph.events` 输出范围应纳入最终验收。
- R1 已指出的无消费者 `pipeline_syntax.common_domains`、`pipeline.actions.operands` 仍留给 high 在全仓检索后精简。本轮删除求域分支后不要另建兼容层；其余 Waiting 坐标恢复仍有真实用途。
- 四份正式案例、wiki、Extract 脚本与状态、旧 demo、历史报告及 CHANGELOG 均未改动。没有 git commit/reset，也未编辑本轮 B 负责的 normalization/source/formatter/CLI 文件。
