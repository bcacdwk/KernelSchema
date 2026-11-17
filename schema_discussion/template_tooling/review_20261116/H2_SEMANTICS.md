# H2 · 规则与图实现的第二轮精简

2026-11-16；基线 HEAD `0778456`。先读取 H1_RULES、R2_RULES、R3_SEMANTICS，再按现行调用链检查共享语法、静态规则、有限图和设计展开。仅修改下列三个代码文件；本轮直接前态保存在 `/tmp/schema-h2-semantics-eo3y54v9`。

## 实际删改与等价依据

| 文件 | 精简 | 调用链依据 |
| --- | --- | --- |
| `tools/pipeline_syntax.py` | 删除 `IndexScope.parent` | `extend` 已将全部可见定义复制到新 `definitions`；消费者只读完整快照，没有向 parent 查找定义的路径。各节点的外层 scope 与正文扩展 scope 原样保留。 |
| 同上 | 删除 `ActionDeclaration.conditions`、`ScopedPipe.conditions` 及递归传递的条件元组 | 全仓没有字段读取。`IfBlock.condition` 与有序正文树已保存实际控制关系；静态规则逐节点检查，图展开递归计算当前 `IfBlock`，窗口投影按约定忽略控制条件。 |
| 同上 | 删除 `ForLoop.restored` | 全仓没有字段读取。恢复坐标仍由 `inner.coordinates` 求出并加入 `body.bound`；实际有限扩展仍通过 `Domains` 与 `merge_coordinates` 检查恢复冲突。仅同步模块开头的接口说明，没有删除恢复行为。 |
| `tools/pipeline/coordinates.py`、`tools/pipeline/windows.py` | `recover` 收敛为 `recover(arguments, domains)`；删除未使用的完整求解模式、外来 env 继承分支及相应参数传递 | R3 删除 Event 猜域后，唯一消费者是 `IndexWindows.missing`，固定传入 `env={}` 和 `partial=True`。所删完整恢复分支永不运行；Windows 继续在声明窗口内判断未恢复坐标与边界。`affine` 的常量求值仍使用同一空环境及 sample parameters。 |

最后一项没有直接删除 `represented` 聚合中的所有调用：其中 `domains.coordinates(name)` 会检查重复恢复坐标、递归和深度上限，因此保留同序调用，只去掉无人使用的集合聚合结果。没有把带校验作用的调用误判为无效计算。

## 验证

使用 `/opt/venv/python3.13.12/bin/python3 -B`，以 `schema_template` 为导入根：

```text
python3 -B -m unittest tools.tests.test_schema tools.tests.test_pipeline_graph
Ran 41 tests in 1.612s — OK
```

这些现有测试覆盖共享语法在 schema 与图中的实际消费，包括词法作用域、活跃索引不可重绑、四时点、三模式、Waiting 固定/移位/空目标、flatten 冲突和实际 Event 调用。没有添加机械测试或重复全套验收。

另以本轮前态 `coordinates.py` 加载为独立模块，按原唯一调用方式与新接口逐项比较返回值或异常类型及消息，9 个进程内小对照全部一致：线性移位、部分可解、欠定、非整数精确解、不一致方程、flatten 坐标恢复、重复恢复坐标、递归域及 128 层深度上限。没有改写 fixture 或展示 demo。

`git diff --check` 通过；逐文件对照本轮前态确认仅上述删改。

## 保留与边界

- R2 的完整名称优先、规范身份、实际引用处的重名歧义、dispatch 三列登记/变体类型和本地 Hardware 文件检查均保留。本轮没有找到值得进一步收拢且能证明等价的规则改动。
- R3 的三处 Flow 实际 Event 调用、未用别名不求值、未知表达式未完成、确定缺失报错以及低层 `event_calls=()` 接口均保留；未恢复 Event 的域猜测或来源坐标捕获。
- `ForLoop.scope/bound`、正文绑定集、作用域完整定义快照和所有实际域恢复检查仍有消费者，未删除。普通 for 端点依旧使用外层作用域。
- 结构入口、静态关系入口与公开低层图入口服务于不同调用方，没有仅凭检查相似就合并。设计展开的累计容量、独立交付复制容量、来源身份与 PNG 发布保护未改。
- 未修改正式规范、README、四份正式 examples、wiki、Extract 脚本/状态、旧 demo、原有历史报告或 CHANGELOG；未执行 commit/reset。未增加字段、规则、预算或功能。

本轮完成，停止写入，交最终 xhigh 统一验收。
