# 本轮有限图工具修复

2026-11-17，独立 high 复现；工具起点为 `7a8030f`。本轮只修改 `schema_template/tools/pipeline/{expressions,graph,windows}.py`、`tools/rules/pipeline.py` 与两份现有测试文件。未修改案例 YAML、PNG、kernel 源码、schema_docs、Extract 执行器、schema_wiki 或公共 README，未提交 commit。

修复前已把两个原始复现脚本备份到 `/tmp/template-tool-fixes.BuXXs9/{qsa_repro,fa4_repro}.py`。原件与历史失败记录保留；以下成功结果来自修复后的独立重跑。

## 1. 动态工作内有界局部域的末项

来源：[QSA TOOLS](qsa_prefill/TOOLS.md)、[最小复现](qsa_prefill/last_local_domain_repro.py)。修复前 `last(i)` 报 `count/last on a dynamically refilled worktile domain is undefined`，等价端点 `tile_count(t)-1` 可展开。

源因：静态 `_dynamic` 与有限求值器 `Domains.has_worktiles` 把范围端点中使用已绑定 `t`，与枚举动态 `t` 的完整域混为一类。依照 [DOC5 索引](../../schema_docs/5.design.tile_pipeline.md#索引)，当前工作的有界局部域可查询，动态工作序列总长仍不可查询。

最小变动：动态总长检查只沿 flatten 轴与显式 `count/last` 查询传播；普通端点使用工作坐标不传播此限制。查询前仍要求域依赖的外层坐标已绑定；缺绑定时有限展开报 `ExpansionError`，普通静态检查报告词法作用域错误。`count/last(t)` 及包含动态 t 的完整 flatten 域仍拒绝，零工作样本不能绕过限制。

验证：原始最小复现的两种表达均为 `3 actions, acyclic=True`。新增静态及图回归覆盖局部 `count/last(i)`、外层未绑定、动态 t/n 查询仍禁止。另将 QSA 案例复制到 `/tmp/template-tool-fixes.BuXXs9/qsa_last/`，只在副本把四处 `tile_count(t)-1` 替换为 `last(i)`；使用原 README 的全部参数执行 `--read-only --graph check`，普通检查与图均通过。PREPROCESS 为 2 actions / 4 edges / 0 skips；ATTENTION 为 158 / 586 / 12，均无环。案例原件未改。

## 2. 首项与循环混合声明的 flatten 越界

来源：[FA4 TOOLS](fa4_prefill/TOOLS.md)、[最小复现](fa4_prefill/tools_repro.py) 的 `flatten_boundary`。修复前等待目标 `compute:k(n+2,0)` 在末端触发 `cannot uniquely recover action-index coordinates: t`。

源因：完整声明窗口已经没有该动作；循环声明能够证明连续序号越界，但首项声明 `n[t,0]` 的子坐标求逆不受支持，造成不确定候选盖过同一可判定边界。

最小变动：在候选既有求逆/常量一致性检查之后，直接完整 flatten 地址的返回值可与它独立声明的连续序号域比较。域外序号无需反求内部 t/i。此判定仅适用于当前可独立枚举的完整 flatten 域；没有更改 `coordinates.py`，没有新增求逆器，也没有降低不确定候选的优先级。完整名义窗口、实际事件查找与窗口内缺失前驱检查保持原义。

验证：原始最小复现成功完成；回归验证保留 2 条合法 Released 边与 2 项边界跳过，错误固定参数 99 和窗口内缺失前驱仍报 `GraphError`。增加一个独立的非线性候选后仍为 `ExpansionError`，没有用已知越界盖掉尚无法判定的候选。

## 3. 无效 padding 的假分支与固定首项

来源：同一 FA4 复现的 `padding_window`。修复前只为验证第二参数常量 `0`，却枚举整个动作表达式，求值无效工作中的 `n[t,tile_count(t)-1]`，从而报 `i=-1`。

最小变动分两处：

- `validate_fixed` 只投影并求值正在核对的参数，仍按完整声明域枚举，不顺带计算其他参数，也不按已生成动作取交集。
- 真实 persistent 命令随后暴露同根的固定首项 `n[t,0]`：空域检查原先要求每个声明都绑定 i，但首项声明只绑定外层 t。现在窗口登记复用现有 If 节点及其词法 Domains，仅供 `empty_target` 在该上下文证明外层条件为假；已证明不发生的目标不生成等待。没有新增语法字段或恢复未使用元数据；未知条件不吞掉，前驱缺失处理不使用此判定。

验证：最小复现成功完成；固定参数 99 仍报错，未知条件仍不能展开，目标实际发生时缺失前驱仍报错。真实命令成功：

```sh
/opt/venv/python3.13.12/bin/python3 -B schema_template/check_yaml.py schema_template/examples/fa4_prefill/packed_persistent.yaml --read-only --graph check \
  --set 'tile_count(0)=1' --set 'tile_count(1)=0' --set 'valid(0)=1' --set 'valid(1)=0' --set 'rescale(n)=0' --worktiles 2
```

结果为 `67 actions, 220 edges, 8 boundary skips; acyclic`；3 YAML 的规范化预览零变化。

## 验证汇总

两份原始复现的运行命令保持不变。FA4 脚本的成功分支原本写成 `unexpectedly completed`，本轮两个分支均进入该成功分支；它不是新的错误消息。

按 [FA4 README](../../schema_template/examples/fa4_prefill/README.md) 的原三份主图参数，把 `--graph png` 改为 `--read-only --graph check`，结果均无环：

| 样本 | actions | edges | boundary skips |
| --- | ---: | ---: | ---: |
| root / MAIN | 168 | 622 | 3 |
| packed_persistent / MAIN | 191 | 704 | 13 |
| split_kv / ATTENTION | 175 | 648 | 2 |
| split_kv / COMBINE | 20 | 58 | 1 |

相关图与普通 schema 测试：47 项通过。最终完整工具测试在全部代码/测试修改后执行一次：

```sh
/opt/venv/python3.13.12/bin/python3 -B -m unittest discover -s schema_template/tools/tests -t schema_template -v
```

实际结果：`Ran 130 tests in 31.501s`，`OK`。原 125 项加本轮 5 项；`git diff --check -- schema_template/tools` 无输出。最初一次以仓库根模块名直接运行单文件测试因 `support.py` 的 `check_yaml` 导入路径失败，随后使用项目支持的顶层路径运行；该调用错误不计作通过。

## 交给案例 reviewers

工具现已冻结。QSA reviewer 可根据源码语义决定是否把等价端点简化为局部 `last(i)`；FA4 reviewer 可复查临时 guards 是否仅为工具绕行、是否应保留为协议表达。工具作者未改这些表达，也未重画 PNG。reviewer 如改 YAML，应按各案例 README 的参数重新检查并最终重画，更新案例中仍称工具问题未解决的说明。

保留边界：复杂/不唯一坐标求逆与未提供样本值仍为 incomplete；本轮不建设一般条件求解器。名义窗口没有统一按 If 过滤，因此本修复不宣称所有任意假分支数学式都可展开。没有运行 GPU，也没有由有限图推出源码数值或全部输入正确性。
