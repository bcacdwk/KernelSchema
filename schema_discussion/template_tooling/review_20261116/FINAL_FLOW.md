# 最终验收 B：规范化、文本与完整入口

基线 `0778456`；精简前快照 `/tmp/schema-template-before-prune-tp4svoxx`。独立读取 R2/R3 自动化结论、两轮 high 记录和实际实现，沿 `yaml_source → normalization → yaml_format → schema → check_yaml` 核对。B 未发现需要追加代码修复的遗漏，没有新增测试；本轮只收尾验收记录及当前 CHANGELOG。规则和有限图的最终补缺见 [A 的验收记录](FINAL_RULES.md)。

## 精简与行为核对

- 逐项对照精简前后的 **80 条预算**，路径、字符数、逻辑行数、prose 与 prefix 完全相同。8 份源、两种 hardware 模式共 **16 组**句号计划及全部文本诊断一致，包含 scalar/collection alias、键用途、特殊具名路径、被显式值覆盖的 merge 默认值及跳过硬件的共享用途。新的用途遍历仍须全部用途都授权才删除句号。
- `_Scope._table` 删除的回调没有调用者；`_Names.add/bind` 的目标与遮蔽信息仍有实际用途。公式形参、熟悉量词和开放正文分开保留绑定证据；Math 不要求符合 Python 语法，未知局部域不依 casefold 猜测。`ref:X.SHAPE` 先确定对象归属，首次即生成 `ref:x.shape`，共享 `from` 和源 alias 使用同一计划。
- 严格读取、显式窄类型恢复、授权源编辑、源图同构和精确值比较仍各自承担职责。纯 int/bool 恢复只在全部用途一致时去引号，显式 tag、null、union、转义与混用 alias 仍保留；Decimal lexeme、anchor/alias/merge 和真实多行正文的保护未删减。Storage 的两个固定字段组及 `Launch.step/order/start_after` 小写表述与实现一致。
- 纯快照规范化与检查保持零 I/O；CLI 先生成整任务候选，安全写回并严格重读，再作预算、语义及所请求的图。冲突不产生可写候选；普通字段或预算错误仍允许独立安全修正。原始摘要、权限、无变化 mtime、失败恢复及保留外部新修改的保护都保留，已有故障回归继续覆盖。

## 真实入口与图

使用 `/opt/venv/python3.13.12/bin/python3 -B` 和临时合成任务，产物与逐次 stdout/stderr 位于 `/tmp/schema-final-flow-x5jnzo22`；不以占位内容证明算子正确性。

- 从 `branch.yaml` 运行真实默认 CLI，整套 root/branch 首次成功，按字段、声明/引用、句尾和布局发出已应用 warning。任务同时包含开放跨行量词、安全形参、`for(T=0:t)`、`ref:X.SHAPE` 的共享 scalar alias、整区 `from`、裸属性引用和 Storage view；首次就达到最终小写引用、固定两行 Storage 与规范 Launch 字段。纯计划再规范化为空。
- 后续真实 `--read-only --graph check --set t=2` 和默认请求均返回 0、无 warning，源文件 bytes、权限和 mtime 不变。另用缺引号说明文本验证必要引号与句尾自动修复；151 字符说明仍返回 1，保留原内容并指出 `150 / 1` 预算，安全字段修正仍正常报告。
- `--all --graph png --set t=2` 经真实 Graphviz **2.40.1** 生成 root/branch 两张 **862×1020** PNG，并实际查看成图。每张包含 MAIN 的两个动作及 NEXT 的一个动作；`StepGraph.events` 精确记录 `ready(1)` 指向 `compute:y(1)@done`，未调用的开放 Event 不要求样本值。PNG 的最终原文摘要、完整来源 manifest、origins、effective identity 和两个 step 的样本均与最终磁盘解析一致。
- 对同一已生成任务省略 `t` 请求 PNG，返回 2、明确图未完成且不报告 PNG 已更新，既有图片和全部源文件的 bytes/权限/mtime 不变。独立捕获冲突任务在默认与只读图请求均返回 1，无候选成功或已应用 warning，含可修兄弟文件的整套来源未写入。
- 对真实 `auto_normalization/display_demo/root.yaml` 只读执行，返回 0、`0 would change`；全目录文件 bytes、权限和 mtime 与执行前一致。

上述图验证仅覆盖给定样本和 step 内局部事件关系，不构成算子数学正确性、整体存储安全、跨 step 活性或性能证明。

## 最终回归与范围

收到 A 明确“规则侧已冻结”、且 B 无代码编辑后，执行约定的完整回归：

```text
/opt/venv/python3.13.12/bin/python3 -B -m unittest discover -s schema_template/tools/tests -t schema_template
Ran 125 tests in 33.534s — OK
```

随后仅对原临时 `safe/branch.yaml` 重跑 `--all --graph png --set t=2`，返回 0、无 warning、两份源 `0 changed`；两张真实 PNG 均重新发布。最终代码的 `ready(1)` 实际调用、两 step 动作数、来源及样本元数据再次核对一致，源 bytes/权限/mtime 不变。没有重建任务或重新排列已有故障用例。

最终 `git diff --check` 和修改范围核对通过。A 额外修复的是精简前已存在的普通 for 图端点及 Hardware gitlink 目录两处缺口；high 删减未见回归，两份当前作者 README 与最终接口一致。

四个正式 examples、wiki、Extract 脚本和状态、旧 text_limit_demo、历史审阅记录及 CHANGELOG 原始要求均保留。本轮没有新增字段、预算、兼容层、数学语法或待用户决策的规则。验收完成，停止写入。
