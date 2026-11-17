# R2：规范化修复与交接

基线 `0778456`，依据 `R1_NORMALIZATION.md` 独立复核 N1/N2/N3。只修改 `schema_template/tools/normalization.py`、`tools/README.md`、`tools/tests/test_normalization.py`；没有改动 `check_yaml.py`、源恢复或排版器，也没有提交 git。共享工作区中的 rules 与对应测试由另一实施代理负责。

## 已修复

### N1：局部改名不能捕获已知外层名字

`_Names.bind` 在进入局部作用域时保留被遮蔽声明的原始拼写和路径。只有识别到原本属于不同外层声明的使用，才在产生可写候选前报 `name normalization would capture outer name`；没有一概拒绝 casefold 相同的局部声明。

已独立复现并修复的路径如下，外层均有 Contract 输入 `t`：

| 路径 | 原表达式的关键部分 | 修复前结果 |
| --- | --- | --- |
| Math 形参 | `f(T) = T + t` | `f(t) = t + t`，候选校验通过 |
| Math 量词 | `sum(T + t for T in [0])` | `sum(t + t for t in [0])`，候选校验通过 |
| Pipe 局部 for | `for(T=0:2)` / `compute:y(T + t)` | 参数与外层输入均变为 `t`，候选校验通过 |
| Event 形参 | `finished(T)` / `{done MAIN[0]::compute:y(T + t)}` | 两个不同使用均变为 `t`，候选校验通过 |
| Storage view 形参 | `y(T): {slot_index: T + t}` | `y(t): {slot_index: t + t}`，候选校验通过 |
| Waiting 成员选择 | Pipe 绑定 `T`，Waiting 使用 `MAIN[t]::compute:y(T)` | `MAIN[t]::compute:y(t)`，候选校验通过 |

Math 的函数名与形参/RHS 分开处理，避免把函数声明名误当作局部使用。原有已识别的方括号形参中，`[T,t]` 也不再合并为同一个坐标；精确重复拼写仍可保留。

量词的熟悉表达式使用 AST 只确定绑定与源码位置，不求值、不重排正文。元素、条件和迭代对象分别使用正确的作用域，避免把 `sum(T for T in t)` 的外层迭代对象误判为捕获。开放数学正文仍不要求符合 Python 语法，无法解析时保留既有的行内量词处理与捕获保护。

Waiting 的成员选择现在与动作参数一起在使用节点的局部作用域确定后规范化，修正了原先成员选择过早改写的遗漏。

安全对照通过：`f(T)=T+1`、原始精确同名的 `same(t)`、函数名与形参分别为 `Self(self)`、公式作用域外继续使用输入 `t`、`sum(T for T in t)`、并列局部循环，以及 Event/view 的安全局部改名。普通字符串和属性成员继续受既有保护。

### N2：确定引用归属后一次完成对象与属性规范化

`_tree_reference` / `_area_target` 用 `None` 明确表示没有找到结构化目标，字符串是否改变不再充当命中标记。短引用会先确定 Math 或 Contract 中的实际目标，再统一对象名和属性路径；只在没有找到结构化目标时做 token 回退。限定引用也使用同一查找结果，没有循环重跑规范化阶段。

`ref:X.SHAPE` 首次即得到 `ref:x.shape`。已知引用槽中的裸 `X.SHAPE` 同样处理；共享 `from`、同一源 scalar 的 anchor/alias 使用都得到一致结果。候选再规范化没有变化。

### N3：README 同步现有窄类型恢复

模块职责表和 `Document` 说明现在同时描述纯文本字段补引号、纯 int/bool 叶去引号，以及 union/null/显式 tag/转义文本/混用 alias 的保留条件。恢复规则没有扩展。

## 验证

所有执行使用 `/opt/venv/python3.13.12/bin/python3 -B`。没有重复运行全套。

- `tools.tests.test_normalization tools.tests.test_yaml tools.tests.test_cli`：65 tests，OK。覆盖已有精确值、引号、类型恢复、alias、固定 Storage 两行、文件写回/恢复和只读行为。
- 最后 Waiting 顺序调整后，补跑捕获拒绝、安全局部作用域、provider/from/Waiting 同步的 3 项相关测试：OK。
- 新增 3 个测试方法，复用既有任务 fixture 和入口；另在原碰撞测试中补充方括号坐标碰撞。没有新增测试框架。
- 真实子进程 CLI：N1 默认与 `--read-only --graph check` 均返回 1，不打印候选 `Schema checked` 或已应用 warning；带独立可修正项的兄弟文件也保持字节、权限、mtime 不变。
- 真实子进程 CLI：N2 的 `from + anchor/alias + 裸属性` 任务首次返回 0，写成最终 `ref:x.shape`；随后默认和只读均返回 0，无 warning，字节、权限和 mtime 均不变。候选纯 API 校验及再规范化也通过。
- 真实子进程 CLI：安全 Math 对照默认及随后只读均返回 0，只读不改动文件。
- 复制当前 `auto_normalization/display_demo` 到临时目录，真实 `--read-only` 返回 0、`0 would change`，副本字节与 mtime 不变；真实展示目录未写入。
- `git diff --check`：通过。

四正式 examples、wiki、Extract 脚本、旧 text_limit_demo、历史研究、CHANGELOG 原始要求均未修改。字段/预算未新增或调整，Launch 小写字段和 Storage 固定分组保持不变。

## 遗留与第三轮重点

1. **retention 的绑定域需先核实，尚不登记为已确认捕获缺陷。** 现有 `_retention` 会把整个字段放入局部迭代名字表。输入 `t`、Symbol `z` 具有 `shape: []` 与 `dtype: fp32` 时，`carry_forward (t); T=0..2` 会变成 `carry_forward (t); t=0..2`，候选校验通过。DOC3 `schema_docs/3.design.math_reshape.md`“值的保留”只说明实例与 `i=起点..终点` 格式，没有明确实例和端点是否受该迭代变量绑定。若直接对整个字段套捕获检查，还可能误拒本应使用外层参数的 `carry_forward invocation; T=0..t`。本轮撤掉了这项尚无足够语义依据的推断，保留原实现，交第三轮依据真实语义决定是否需要修复；不要据此新增全局同名禁令。
2. 下一轮独立核对 Math 量词的嵌套/续行与作用域边界，尤其开放数学语法回退的范围；目标仍是证明改名保留绑定，不能把 AST 可解析性变成作者语法要求。Lambda 等其它数学记法未在本轮扩大识别或宣称已覆盖。
3. R1 提到的重复 `recover_scalar_quotes`、未使用 `text_policy.expects_text` 留给后续精简；本轮不碰其它代理的模块，不为 N1/N2 新增兼容层、重试阶段或旧名字迁移。
