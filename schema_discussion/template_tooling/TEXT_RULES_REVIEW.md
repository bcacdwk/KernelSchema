# 文本规则最终审阅

结论：按当前 [批准表](TEXT_RULES_PROPOSAL.md) 完成独立源码审阅、必要修正与验收，无阻塞。未扩充字段、值类型、动作或等待语义。

修复一处实质误拒：`REALIZATION.item` 原来在查登记对象之前就拒绝任何方括号，导致 `Copy [scalar]` 这样的 Rewrite 名和 `Feature.L1/SMEM [shared]` 这样的硬件路径无法引用。现在先匹配完整登记名称和映射路径，再对未解析目标检查下标形式；新增回归同时确认未登记的 `formula[0]` 仍不能作为 item 的列表下标身份。

除现有测试外，独立复现确认：

- 完整快照及 `from` 展开接受恰好 300 字符、2 逻辑行的引用；Input 的 150 字符 range/default 保留，混合数学区间括号可用。屏蔽文件读取、文件状态/路径解析与 subprocess 后，`validate_documents` 仍通过。
- 100 字符的真实 dtype、Execution 登记键及其成员引用可用，不叠加单名 50；纯数学正文不建立新符号表。
- 文件 API 核对当前明确仓库的本地提交树：含空格、圆括号和方括号的目录可定位；工作树删除不影响固定版本，确定缺失路径报错，缺本地历史保留未解析。仅路径存在性通过不表示源码入口或实验依据已获证明。
- Worker/Placement、Worktile_order/by、Preassigned、单/多机制 Refill 和 exit 格式化保留值、顺序、显式 tag、字符串类型、anchor/alias 及幂等；既有套件覆盖 Decimal、必要引号和安全句号修正。

最终完整套件：`/opt/venv/python3.13.12/bin/python3 -B -m unittest discover -s schema_template/tools/tests -t schema_template`，**82 项通过，5.328 秒**。38 个 Python 文件通过 3.10 语法检查；`git diff --check` 通过。

本次未改 examples、wiki、Extract 执行器或运行状态、starter、CHANGELOG 用户草稿及旧 `text_limit_demo` 与统计附件；审阅前后受保护的 2297 个已跟踪/未忽略文件内容一致。旧 demo 是前轮压力快照，不作为新规则通过样例，也未为它放宽规则。
