# R3：自动化独立复核与必要补缺

基线 `0778456`；2026-11-16。先读仓库 AGENTS、R1/R2 normalization 报告、R2 rules 报告和当前 DOC3，独立检查名称绑定、候选生成、只读/写回与文本边界。本轮没有重跑全套，没有提交 git。

本轮实际修改 `normalization.py`、`tests/test_normalization.py`、两处当前 README；另经主代理明确协调，在 A 结束写入后修改 `rules/pipeline.py` 的一处范围端点检查。未修改其他 rules、pipeline 实现、正式 DOC、字段或预算。A 的事件图改动见 `R3_SEMANTICS.md`。

## 已确认并修复

### N4：开放数学的逐行量词回退既能漏掉捕获，也能误拒安全改名

R2 的 AST 分支能够区分量词元素、条件与迭代对象；但 AST 不能解析完整开放数学时，回退重新把每个物理行放入一个局部名字表。以 Contract 输入 `t` 为例，定向调用原 `_math_tokens` 得到：

| 原文 | R2 结果 |
| --- | --- |
| `y = sum(T for T in t) ⊙ t` | 错报 `capture outer name 't'`，将外层迭代对象误判为局部使用 |
| `y = sum(T for T in [0]) ⊙ t` | 同样误拒，量词外的 `t` 被整行局部域覆盖 |
| `y = sum(` / `T + t` / `for T in [0]` / `) ⊙ t` | 输出局部 `t` 与外层 `t` 合并后的 `t + t`，没有报捕获 |

修复仍只识别已有熟悉量词，没有建立新的数学语法：

- 在完整表达式不能用现有 AST 路径识别时，用已有 `_group_end` 提取包含量词的平衡源码片段，包括原始换行，再由同一个量词处理入口绑定。字符串内容先遮蔽给括号扫描器，实际正文不改。
- 无法确定局部范围时，不再逐行绑定或仅凭 casefold 认定变量相同。允许不与量词冲突的确切声明改名；局部改名或可能合并已知引用的变更报明确歧义。已有规范名字的开放表达式照常保留，不要求 Python 可解析。
- `_formula_scopes` 的已有简单定义识别同时提供下一定义的边界，保证裸 `T for T in x` 与其后的 `y = ...` 各自处理，函数名继续属于外层域。
- `_Names.find` / `_tokens` 增加仅供这条开放回退使用的 `exact` 选项；其余已经识别的作用域与引用逻辑未换成另一套名字表。

回归明确覆盖：跨行捕获拒写；开放数学中的单行和多行安全量词；普通形参与操作符续行；独立裸量词；字符串保护；已声明 `N` 在开放正文中改为 `n`，而不能证明绑定到输入 `x` 的 `X` 保留原文。未知局部改名例如 `sum(X ⊙ x for X in [0])` 在计划阶段停止，不发布半次改名。

### N5：普通 for 的新局部名不能反向污染自己的外层端点

加入本轮要求的安全对照后，发现 `for(T=0:t)` / `compute:y(T)` 正确改成 `for(t=0:t)` / `compute:y(t)`，但随后静态检查报 `index t is not bound in this lexical scope`。原因是 `all_indices` 收集了整个 step 的局部循环名，范围端点虽然使用 `ForLoop.scope` 外层域，仍被这个全集判成局部索引使用。

`rules/pipeline.py` 现在仅在检查新局部范围的端点时，去掉尚未进入外层 `IndexScope` 的本次局部名。已有外层 Index 域继续要求绑定，正文仍使用局部域，禁止重绑活跃循环索引的解析规则未改。没有新增全局同名禁令，也没有要求增加 Contract 字段。

真实 CLI 用 `--graph check --set t=2` 验证该安全循环得到 2 个动作；它与上述多行开放量词、共享 `from` 一起首次返回 0。

## retention 的独立结论

DOC3「值的保留」把实例解释为一份逻辑值/查询行，分号后的范围描述 Math 中的递推进度；初值、重置、有效范围与输出关系均由 Math 给出。因此这里是递推轴的描述性引用，不应按 Pipe 中新声明一个可执行循环来套整字段局部绑定。

`carry_forward (t); T=0..2` 变成 `(t); t=0..2` 时实例表达式本身未变，单凭结果同名不能证明发生了函数式变量捕获。本轮没有据此新增规则或拒绝安全例。另实测 `carry_forward invocation; T=0..t` 一次成为 `carry_forward invocation; t=0..t`，默认、后续默认和只读均通过；外层端点的含义得以保留。

本轮未证成需要另外修复的 retention 缺陷，保留原实现。retention 与实际 Math 递推坐标是否一致仍按现有职责审阅，不把尚未形式化的任意数学绑定关系升级为新的作者语言要求。

## 其他边界与文档

- R2 的 `ref:X.SHAPE` 归属修复保持一次完成；现有 provider/from、anchor/alias 与裸属性回归通过，候选再次规范化无变化。没有增加重复规范化阶段。
- 字段大小写、必要引号、纯文本与 int/bool 恢复、句尾处理、Unicode/逻辑行预算和固定 Storage 两行仍由原模块拥有。读取对应源保真检查后，未发现本轮需要修改的缺陷；相关定向回归通过。
- `normalize_documents` 和 `validate_documents` 的纯快照入口保持零 I/O。碰撞或绑定歧义在返回候选前失败；普通预算/字段错误仍允许独立安全规范化写回，但不会打印 `Schema checked`，符合既定入口顺序。
- tools README 把「已验证候选」明确为「经过源保真核对的候选」，避免误读成已经通过全部语义校验。
- 已按 A 的最终实现同步有限图 API：`build_step(..., *, dynamic_worktiles=False, event_calls=())` 接收 `(本 step 调用字符串, 来源字段路径)`；`analyze` 收集 Flow 的三处真实调用，`StepGraph.events` 仅记录实际调用。未调用 Event 不要求有限取值，完整环仍通过 `cycle` 返回。template README 同步作者可见的调用边界。

## 验证

全部使用 `/opt/venv/python3.13.12/bin/python3 -B`，以 `schema_template` 为导入根。只做必要定向验证：

- 共 16 个不同现有/新增测试方法通过：名称捕获与安全作用域、开放量词、形参续行、独立裸量词、模糊多行拒写、属性根、provider/from/alias 引用、纯快照零 I/O、固定字段/类型恢复、碰撞拒写、普通语义错误与预算、句号预算边界、Storage tag/alias/merge/Decimal、Unicode/逻辑行，以及既有活跃索引/flatten 作用域测试。
- 本轮只新增 1 个 meaningful 测试方法，并扩展已有捕获拒写、安全作用域两项。最后开放回退收拢后，补跑开放量词、捕获拒写、安全作用域 3 项，OK。
- 真实子进程 CLI：冲突的多行开放量词在默认与 `--read-only --graph check` 均返回 1，不报告 `Schema checked` 或已应用 warning；整套任务含兄弟文件的字节、权限、mtime 均未变。
- 真实子进程 CLI：安全开放量词 + `for(T=0:t)` + retention 描述 + from 任务，首次 `--graph check --set t=2` 返回 0，图有 2 个动作；随后默认与只读图检查均返回 0、无 warning，任务字节、权限、mtime 不变。
- `git diff --check` 通过。没有使用真实案例或展示 demo 作为写入/回归输入。

## 给后续 high / xhigh

已证成的两项问题均已修复；本轮不再拓展新的数学记法或假设。high 可重点看 `_math_tokens` 的源码切片与 `exact` 回退是否还能收拢，但应保留多行安全/冲突两个对照及普通 for 的外层端点。AST 仅服务熟悉量词的绑定与源码位置，不能成为 Math 的合法语法要求。

前两轮列出的 `recover_scalar_quotes` 重复调用和未使用 `text_policy.expects_text` 本轮未动，仍可作为纯精简候选。最终全套与两轮 fresh high、fresh xhigh 由主代理统一执行。

四正式 examples、wiki、Extract 脚本/状态、旧 text_limit_demo、历史记录与 CHANGELOG 原始要求均未修改。
