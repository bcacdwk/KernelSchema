# 工具独立复核与精简

2026-11-17，独立 xhigh 复核。直接阅读 `7a8030f` 至当前的工具 diff、DOC4/5、工具 README 和原始最小复现；没有以先前报告代替代码检查。工具已冻结，最终完整测试为 **137 项通过**。

## 独立结论

本轮原有修复的抽象层正确：动态 worktile 的完整域与当前工作内有界域分开；完整 flatten 地址的返回值受连续序号域约束；固定参数检查只求所需列；If 仅用于证明目标不存在，不过滤前驱的名义窗口。`Mapping.Member` 的单点区域是已有数学坐标，按成员展开及分开缓存有规则依据，无需读取 Strategy、引入保留字或新增 schema 字段。多元素和未知区域仍不猜代表值。

复核发现并修正两处遗漏：

1. **成员数学坐标与局部索引的作用域。** 原实现把成员值直接放入循环绑定环境，合法的 `for(split=0:split+2)` 会与外层成员 `split` 冲突，Waiting 也可能把同名域误当成已经绑定。现在成员值是外层数学环境，局部域按词法作用域遮蔽它；范围端点使用外层值，正文及 Waiting 完整枚举局部索引，退出循环后恢复成员值。新增回归覆盖两成员的不同端点、前后动作和全部等待；既有禁止重绑定活跃循环、flatten 外层坐标冲突的检查仍通过。
2. **固定参数的联合投影。** 原检查逐列取值，声明只有 `(0,1)`、`(1,0)` 时，引用 `(0,0)` 也可能被变量越界掩盖。现在投影保留完整参数元组，错误组合报 `GraphError`。新增正反例确认合法组合仍产生边界跳过，非法组合不能放行。

## 实现精简

- 完整窗口和固定列共用 `IndexWindows.values` 的投影、缓存与枚举预算，删除固定列的第二套遍历。缓存键保留窗口（含成员）及投影位置；不再保存无人使用的动作到环境列表。
- `resolve` 只查一个成员，选中成员的遍历统一留在 Waiting 调用处；删除重复目标选择及中间成员列表。`[*]` 中某成员的边界不会删除其他成员的有效边。
- 控制节点先遍历一次，再建立各成员的词法 Domains，减少重复 AST 遍历。删除解释首项修复历史的注释，保留名义窗口、固定参数和坐标恢复的不变量。
- 工具 README 说明成员坐标遮蔽与精确整数位运算；未增加历史说明、CLI 开关、兼容路径或求解框架。

位运算不需要额外代码修复：`^`、`<<` 使用已有精确求值结果，整数值的分式表达式和负整数可用；布尔、非整数、负位移报确定错误，过大位移或结果报 `ExpansionError`。在原测试中补充了上述正向边界，不限于整数字面量。

## 验证

最终代码完成后执行一次完整测试：

```sh
/opt/venv/python3.13.12/bin/python3 -B -m unittest discover -s schema_template/tools/tests -t schema_template -v
```

结果：`Ran 137 tests in 30.294s; OK`，见 [完整日志](tools_polish_tests.log)。原 135 项加两项实际漏洞回归。`git diff --check -- schema_template/tools` 无输出。开发过程另执行必要的定向图测试；首次新增成员测试遗漏 role 的多成员选择，修正测试夹具后通过，没有把该次失败计入最终结果。

两份原始脚本均完成，见 [最小复现日志](tools_polish_repros.log)。FA4 脚本保留旧成功提示 `unexpectedly completed`；两个分支实际成功。QSA 的 `last(i)` 与等价端点都为 3 actions、无环。回归继续覆盖动态 `count/last(t/n)` 禁止、未绑定外层坐标、未知条件、固定索引错误、窗口内缺失前驱以及非线性候选不能被另一个已知边界掩盖。

另外直接核对参数相关性：声明 `f(i,i%2)`，同域引用 `f(i,0)` 在 i=1 报 `GraphError`；`f(i+2,0)` 也报错。声明 `matmul:a(i)@b(i)`，引用 `a(i)@b(i+1)` 时联立恢复证明不一致并报 `GraphError`，没有成功跳过或增加求解器。

按各案例 README 的参数复用最终检查命令，九份主图样本与七项边界共 **16 次只读检查**全部退出 0、零规范化变化、无环；没有使用 `--all` 套用参数或跳过硬件。精确命令和结果见 [案例日志](tools_polish_cases.log)。九份主图结果如下（actions / edges / boundary skips）：

| 节点 | 各 step 结果 |
| --- | --- |
| FA4 prefill root | MAIN 194 / 700 / 29 |
| packed_persistent | MAIN 217 / 782 / 40 |
| split_kv | ATTENTION 194 / 705 / 18；COMBINE 20 / 58 / 1 |
| FA4 decode root | PARTIAL 351 / 1168 / 20；COMBINE 24 / 70 / 0 |
| q1_global | PARTIAL 359 / 1196 / 25；COMBINE 38 / 112 / 0 |
| q4_direct | MAIN 88 / 284 / 8 |
| q4_cluster（T_b16） | MAIN 1730 / 5782 / 128 |
| QSA root | PREPROCESS 2 / 4 / 0；ATTENTION 158 / 586 / 12 |
| GDN root | INDEX_CAST 1 / 1 / 0；GDN 249 / 979 / 30 |

其余检查覆盖 FA4 prefill 单轮、persistent 有效工作接 padding、空 split、cluster T_b70 与 T_b1、QSA 空 worker、GDN 最小合法两 chunk。已有 15 条命令的各项计数与 [整合者冻结前日志](final_checks.log) 完全一致；附加 T_b1 为 590 / 1972 / 8。

对当前 cluster 再查实际 `Operation.context`，见 [成员结果](tools_polish_members.json)：T_b70 的成员 0..5 各五轮、6..15 各四轮；T_b1 为一活跃成员和十五空成员。两样本每成员均有八次蝶形发送，zero_done 仅来自成员 0，异构成员没有折叠。

## 冻结与交接

源码只读 API、零 I/O 分析入口、错误分类及既有 64 成员/工作样本、20000 动作/域/窗口、100000 枚举、表达式与数值容量保护保留。复杂或不唯一的求逆、不支持的数学表达式和缺少有限样本仍可 incomplete；没有一般条件求解器，不将这些能力边界改成 schema 禁令。

本复核没有修改案例 YAML/PNG、固定 kernel、schema_docs/schema_wiki、Extract 状态或执行器；没有提交 Git 或运行 GPU/Runner。九份现有主图的动作及依赖结果不受这次精简影响，**无需因本次工具调整重绘 PNG**；最终来源摘要与发布状态由整合者签收。没有新增需用户决定的语法、字段或工具事项。案例中已记录的源码风险仍由各自审阅结论承担。
