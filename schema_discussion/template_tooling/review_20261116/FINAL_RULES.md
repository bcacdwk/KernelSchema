# xhigh 最终验收 A · 规则、语义与图

2026-11-16；基线 `0778456`。独立对照当前 DOC0–6、规则与图实现及精简前快照 `/tmp/schema-template-before-prune-tp4svoxx`。未发现 high 两轮删减造成的行为回归；另补齐两处精简前已存在的确定缺口。代码已冻结并通知 B 统一运行最终全套。

## 最终纠正

- **局部 for 端点的外层作用域。** 公共域 `i=[0,3]` 下，`for(i=0:count(i))` 原先通过静态检查，却在有限展开时查询覆盖后的新域，误报递归；`for(i=0:i)` 使用同名数学参数时，Waiting 投影也会误判自依赖。现有 `Domains` 复用 `ForLoop.scope/bound` 保存端点的实际外层求值上下文，动作展开、域依赖、动态域追踪和 Waiting 声明窗口共同使用它。Waiting 仍按实际引用声明选择域，不再把合法局部域与被遮蔽的公共域误判为两个候选。未恢复 `IndexScope.parent`，未新增 schema 字段、域语法或预算。真正递归域、活跃索引重绑和 flatten 坐标冲突仍报错。
- **Hardware 的子模块目录。** 独立临时 Git 仓库复现：`ls-tree` 中 `160000 commit` 的 gitlink 被仅拒绝 `tree` 的分支放行。现将这两种确定目录类型都用于既有 `Wiki.Hardware` 文件检查；普通 Wiki 目录、资料文件、未绑定占位和 `skip_hardware` 保持原行为。不追踪符号链接，不联网或判断资料内容。

本轮代码仅修改 `tools/pipeline/expressions.py`、`tools/pipeline/graph.py`、`tools/rules/values.py`，新增一个聚焦图回归，并扩展已有本地资料测试。

## 独立核对的保留行为

- 精简删除的 `conditions/restored/parent` 是无人读取的派生信息；有序 `IfBlock` 树、循环头外层 scope、正文绑定与 flatten 坐标仍完整保留。当前消费者没有调用所删 helper；历史 trial helper 不承担现行 API 兼容要求。
- Waiting 保留目标控制条件、固定/移位/空目标的区别、每个前驱独立生效和精确坐标恢复。`recover` 仅留下当前 Windows 所用的部分求解，重复坐标、递归、深度与坐标数量检查均在。
- 四种显式 issue/done 边、三种执行模式、跨 noun/verb/worktile 的 ordered 完成、serial 不排空其他异步及同名 noun 不生数据边均通过相关现有回归。
- Item 优先完整登记名，裸/限定属性身份一致；被实际引用的重名报告歧义，未引用的重名不新增全局禁止。Math 属性存在性、定义顺序、环与 binding 冲突职责仍分开。dispatch 核对最低三列表格的设计列登记、变体名称/类型和直接完整基线。
- Event 经 `start_after / ready / wait_at` 三处实际调用代入；未用别名不猜域、不求值。缺显式样本或超出有限求值范围为 `ExpansionError`，确定缺失事件为 `GraphError`，事件列表不能取交集删项。不构造跨 step 执行证明。
- `from` 保留整区 DAG、实际 provider/chain、consumer 当前定义绑定和独立交付副本。删去一次性 `_Budget` 包装后，交付复制前容量与整项累计容量仍各自有效。

## 验证与遗留

使用 `/opt/venv/python3.13.12/bin/python3 -B`，导入根为 `schema_template`：

```text
python3 -B -m unittest tools.tests.test_schema tools.tests.test_pipeline_graph \
  tools.tests.test_dispatch tools.tests.test_design_resolution
Ran 56 tests in 2.439s — OK
```

独立进程内探针另核对：新旧 `recover` 的 9 项返回值/异常完全一致（精确、部分、非整数、不一致、flatten、重复坐标、递归、128 层深度及 64 个坐标上限）；跨局部循环的三条合法边与一次边界跳过；三组 `(0,2)/(2,0)/(0,0)` 空工作组合保留其他合法前驱；复制前及累计容量；公开 `analyze`、嵌套局部窗口和同名动态域查询拒绝。`git diff --check` 通过。未重复运行完整测试集或 PNG，交 B 统一验收。

没有新增待决规范。开放数学、正文目标、硬件内容及跨 step 工作匹配继续按既定工具边界审阅。正式 examples、wiki、Extract 执行器/状态、旧 demo 与历史材料未修改；未执行 commit/reset。
