# Mapping.Member 的有限图上下文

2026-11-17，独立 high 核对并修复 FA4 decode cluster 暴露的成员展开缺口。基于本轮已有 [TOOL_FIXES.md](TOOL_FIXES.md) 的 130 项通过状态继续；没有提交 Git，没有运行 GPU/Runner，没有改源 kernel、案例 YAML、其他案例或 schema_docs。

## 规则依据与取舍

[DOC4 Mapping](../../schema_docs/4.design.work_schedule.md#mapping) 已规定 `Member` 列表位置对应成员序号，每维登记左闭右开的局部区域；[DOC5](../../schema_docs/5.design.tile_pipeline.md) 接收上游数学分工，并以成员作为动作实例身份的一部分。因此 `[r,r+1)` 可以把该数学维度绑定为本成员的已知坐标，无需 `member` 保留字、新的 Index 域、新 schema 字段或对 Strategy 英文的推断。

工具只绑定能由给定整数样本确定、宽度恰为 1 的区域。多元素或端点未知的区域不选代表值；需要该坐标的表达式继续返回 incomplete。`analyze` 从当前 effective plan 的 `Work_schedule.Step.*.Mapping.Member` 自动取值；零 I/O 的 `build_step` 可显式传同一列表为 `member_regions`。没有新增 CLI 开关，也没有改变快照/来源身份或 PNG 发布边界。

Waiting 的地址表达式仍在当前目标成员上下文求值，例如 `split`、`count(i)` 和循环 `i` 属于该目标。显式 `A[peer]` 选择对方的事件身份；对方是否实际存在、固定参数是否有效以及变量位移是否越界，使用对方自己的完整声明窗口。默认成员仍是当前成员，`[*]` 逐一要求所有选中前驱；某个成员的合法边界跳过不会抹掉另一个成员的有效边。Event 调用先由显式实参决定 selector，动作在各选中成员上下文求值；显式实参优先，不能从 loop 或成员坐标捕获同名形参。没有跨 step 执行模拟。

现有有限解析器使用 Python AST，`^` 没有其他已实现的固定含义。本轮只补真实蝶形需要的整数 XOR `^` 和左移 `<<`；拒绝布尔/分数操作数、负位移，位移超过 40 返回 incomplete，结果继续受既有 `10^12` 幅度上限限制。案例已经写作 `split ^ (1 << j)`，没有创建 64 项 peer 参数表。schema 的开放数学表达式不因此收窄。

## 最小代码变化

- `pipeline/expressions.py`：Domains 接收已知成员坐标，供域端点、局部范围、条件和表达式查询使用；补整数 XOR/左移。
- `pipeline/graph.py`：每成员独立展开，作用域缓存按成员分开；Waiting 的活跃家族及求值上下文按成员区分；显式 Event 逐成员查找。
- `pipeline/windows.py`：窗口按 `(role, member, family)` 登记；缓存不混用其他成员的长度，仍共享原来的总 100000 次窗口枚举上限。
- `pipeline/__init__.py`：从既有 Mapping.Member 传值。`tools/README.md` 补低层参数和上述求值边界。
- `tests/test_pipeline_graph.py` 新增 5 项合成回归：2 成员不等长/局部 if/inline 域/不折叠；peer 正反例与 `[*]`；成员空 flatten；有界位运算；analyze 与 Event selector/形参。

本轮没有改 `rules/` 或 renderer；这些路径中已有的未提交修改属于上一轮窗口修复。现有图折叠以实际动作和邻接关系为依据，真实异构样本均没有折叠角色，未用 16 份同构 Role 或统一假 N 绕过问题。

## 实际验证

使用 `/opt/venv/python3.13.12/bin/python3`。最终全量命令为：

```sh
/opt/venv/python3.13.12/bin/python3 -B -m unittest discover -s schema_template/tools/tests -t schema_template -v
```

结果 `Ran 135 tests in 29.526s; OK`，见 [member_tool_tests.log](fa4_decode/member_tool_tests.log)。`git diff --check -- schema_template/tools` 无输出。之前完成的一轮 135 项后，自查将窗口统一回原总枚举上限，再跑了上述最终完整轮。

真实案例只读检查及当前源码上下文审计得到：

| T_b / S | 每成员真实迭代数 | actions | edges | boundary skips | 结果 |
| --- | --- | ---: | ---: | ---: | --- |
| 70 / 16 | 成员 0..5 各 5；6..15 各 4 | 5198 | 17474 | 320 | 无环 |
| 17 / 16 | 成员 0 为 2；1..15 各 1 | 1775 | 5937 | 134 | 无环 |
| 1 / 16 | 成员 0 为 1；1..15 为 0 | 589 | 1969 | 8 | 无环 |
| 0 / 16 | 全部为 0 | 512 | 1712 | 0 | 无环；仅空工作机制诊断，非合法输入 |

从生成的 Operation.context 实查各成员 LOADK key 的 i 集合得到上述长度，不只核对总动作数。全部四个样本中每个成员都有 8 个蝶形发送动作（4 max + 4 sum）；前三个样本的 `signal:zero_done` 仅存在于成员 0，T_b0 无此动作。T_b70、17、1 的 `display_members` 都返回空集合，异构轨迹完整保留。N1 活跃路径与 N0 中性统计路径已经实际展开。

最终 T_b70 命令与结果见 [member_cluster_t70_check.log](fa4_decode/member_cluster_t70_check.log)：

```sh
/opt/venv/python3.13.12/bin/python3 -B schema_template/check_yaml.py schema_template/examples/fa4_decode/q4_cluster.yaml \
  --read-only --graph check --set T_b=70 --set splitcount=16 --worktiles 1
```

普通结构、文本、from、关系与 dispatch 均通过，硬件未跳过，`4 YAML; 0 would change`。

## PNG 与样本范围

T_b70 主输入（B1/Q4/L_b17844/P279）和合法较小的 T_b17 输入（B1/Q4/L_b4097/P65）都完成了图检查，但 Graphviz 均超过 60 秒，退出码 2，未发布伪成功图。日志分别为 [T70](fa4_decode/member_cluster_t70_png.log) 和 [T17](fa4_decode/member_cluster_t17_png.log)。

最终 [q4_cluster.png](../../schema_template/examples/fa4_decode/q4_cluster.png) 成功使用 T_b1/S16/worktiles1，见 [日志](fa4_decode/member_cluster_t1_png.log)。它代表同机制合法输入 `Q4/B2/seq_lens=[64,17844]/P280` 的 request0，**不在 q4_cluster.yaml 当前 B1 的 when 内**。核对 host `_pick_tiles/_select_policy`：grid_mass=4、cap=70，仍返回 S16 atomic；request0 的 T_b1 正好得到 1 活跃成员和 15 个空成员。没有改文件 when，也没有把该图称作 T_b70 主输入的图。

PNG 为 25019×5347、约 4.6 MB，保留全部 16 成员，缩览很宽，查看细节需放大。图内 `samples` 正确记录 T_b1；`source_sha256` 与 `source_manifest` 已逐项实算核对当前文件：

- q4_cluster.yaml：`e75c90cb71425268c16c4f2ed314cdec138b4fe2d239063b6055cb1e93dd1c73`
- root.yaml：`6a3f04f2465ee29991cce51b43ca732ca6d391081dc42a3b29fd8afc820cac63`

## 交给 fresh case reviewer

工具和本记录已完成，案例语义仍须独立全文复核。请依据 [WRITE](fa4_decode/WRITE.md)、[SELECTION](fa4_decode/SELECTION.md) 继续核对 16 成员不等长工作、空成员中性统计、zero_done、BF16 reduce-add 与资源退休；本轮图不证明 kernel 数值或全域正确性。

案例 README/WRITE 中“split 未绑定、cluster PNG 缺席、预期失败”的旧状态需要 reviewer 更新为本记录的实际结果，并明确主输入检查与小样本 PNG 的范围。若最终修改任一来源 YAML，应重查并重画 PNG；当前 hash 仅对应这里记录的版本。不要把 T_b1 小样本改写为 B1 when 内输入，也不要为更小图片删掉空成员或蝶形协作。

保留局限：复杂/不唯一坐标求逆、未知端点和不支持表达式仍可能 incomplete；名义窗口仍不是按任意 If 过滤后的动作集合，缺失窗口内前驱继续是错误。没有一般条件求解器、源码解释器或隐式数学/数据依赖推断。PageIndex 等原始源协议未决也不会由此图检查自动消除。
