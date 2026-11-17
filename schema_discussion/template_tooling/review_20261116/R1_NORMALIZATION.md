# R1：规范化、校验入口与文件落地独立审阅

基线 `0778456`。只读审查 `check_yaml.py`、`schema.py`、`yaml_source.py`、`normalization.py`、`yaml_format.py`、`diagnostics.py`、`text_policy.py` 及当前 DOC0–6、作者 README 和展示 demo；正式代码与真实 demo 均未改动。复现使用 `/opt/venv/python3.13.12/bin/python3 -B`、合成 minimal fixture 和临时目录。主代理提供的基线结果为 113 项 unittest 全过；本审阅未重跑全套。

## 必须修复

### N1 · P1：局部形参改小写会捕获原本独立的自由变量，错误公式被成功写回

位置：`schema_template/tools/normalization.py:169` 的 `_formula_scopes`，以及 `:460` 的 `_Scope.formula`，特别是 `:466` 的 `local.entries.update(parameters.entries)`。

从 `tools/tests/fixtures/minimal/root.yaml` 复制任务到临时目录，将 Input 与对应 Worktile.Invariant 的 `x` 改为 `t`，Contract 的 Math 改为 `y = t`。Math_reshape 的 Math 填：

```text
f(T) = T + t
y = f(0)
```

`T` 是形参，`t` 是已声明的 Contract 输入。当前规范化结果为：

```text
f(t) = t + t
y = f(0)
```

输入 `t=3` 时，原设计得到 `y=3`，改后得到 `y=0`。这是工具改写引入的绑定变化，不是要求检查器验证任意数学正确性。

实际入口结果：

| 调用 | 当前结果 |
| --- | --- |
| `check_yaml.py <copy>/root.yaml --read-only --graph check` | 返回 1，仅因存在未落盘差异；打印 `Schema checked`、`Graph checked ... acyclic`，计划中的公式已错误合并；原文件字节和 mtime 保持不变 |
| `check_yaml.py <copy>/root.yaml` | 返回 **0**，写回上述错误公式，并以普通 `identifier+reference` warning 报告 |
| `schema.validate_documents(plan.files, plan.documents)` | PASS |

原因：局部形参表以 casefold 为键覆盖外层表，随后 `_tokens` 将精确拼写不同的 `T` 和 `t` 当作同一个局部绑定。已有检查只覆盖形参之间的碰撞及部分多行边界，没有检查改名造成的自由变量捕获。

建议：利用已有局部/外层名字表、原始拼写和已识别的公式作用域，在生成候选前检查真正的名称捕获；该例应报歧义并停止整任务写回。不引入公式求解器，也不一概拒绝合法局部同名或遮蔽。

验收：默认、只读入口均在计划阶段拒绝此例，不报告候选校验成功、不落盘；兄弟文件即使存在独立可修正项也保持字节及 mtime 不变。保留安全用例，例如 `f(T) = T + 1` 的局部改名，以及作用域外继续使用输入 `t`；已使用同一精确小写拼写的合法局部遮蔽不能被误拒。

### N2 · P2：属性引用的对象名与属性名同时改大小写时，需要运行两次

位置：`schema_template/tools/normalization.py:650` 的 `_area_target`、`:698` 的 `_reference`。

在 minimal fixture 中将 Contract Input `x` 及其 Math/Invariant 使用改为 `X`，为 Rewrite 加入：

```yaml
Symbol:
  z: {meaning: Copied view, shape: 'ref:X.SHAPE', dtype: fp32}
```

当前第一次默认命令写回 `ref:x.SHAPE`，退出 **1**，报 `reference the same .shape property`；第二次命令才写成 `ref:x.shape` 并退出 **0**。对第一份候选再次调用 `normalize_documents` 也得到非空 `candidates`，说明整任务规范化本身不幂等。相邻对照 `ref:X.shape`、`ref:x.SHAPE` 和 `ref:CONTRACT.Algorithm_requirement::X.SHAPE` 均能一次完成。

原因：未限定引用先尝试 Math 区；该区没有目标对象时，`_area_target` 仍回退到 token 改名，得到 `x.SHAPE`。调用方把“字符串发生变化”当成“已找到目标”，提前停止，不再查找真正拥有 `X` 的 Contract 区。

建议：区分已找到结构化目标和仅做 token 回退；确定所属对象后一次完成对象名与属性路径修正，再处理真正未结构化的引用。不要靠循环重复整个规范化阶段掩盖绑定问题，也不要要求作者把可解析的短引用全部改成完整限定名。

验收：第一次普通命令得到 `ref:x.shape` 并通过；随后普通/只读入口均 0 项修改、无规范化 warning、字节与 mtime 不变。补充共享 `from` 使用节点和必要 alias 情况，使同一源引用只产生一个一致结果。

相关但归属另一份报告：裸 `REALIZATION.Item` 的已知对象属性路径被当成未解析文本，导致 `x.no_such_property` 及裸/限定同一属性的重复引用漏检；已由审查者 A 确认并纳入 `R1_CONTRACT.md`，此处不重复登记。

### N3 · P3：当前工具 README 仍只描述文本类型恢复

`schema_template/tools/README.md:54` 声称“类型恢复只处理……全部用途均为已知文本字段”。实际 `yaml_source.recover_scalar_quotes`（`:531` 起）也会对结构声明为纯 int/bool 的叶子去除引号，例如 `Version: '1'`、`nullable: 'true'`；现有 `test_quoted_integer_and_boolean_fields_normalize_in_preview_and_default_cli` 已覆盖它们。

建议只更新这段当前作者/API 说明，描述已存在的双向窄范围恢复及 union/null/显式 tag/混用 alias 的保留条件，不改历史研究记录，不扩大恢复规则。文档应与模块职责表中的 `LITERAL_TYPES` 保持一致。

## 可删冗余

- `text_policy.expects_text`（`:110`）没有 Template 内调用者，未列为文档 API；缺引号判断现已从 `rules.structure` 的字段类型表推导。可以删除这段旧的预算表推断入口，避免以后有人误以为“受预算约束”就等于“字段只接受字符串”。删除前再做一次仓库调用者检索即可。
- `loads_for_normalization` 已执行一次 `recover_scalar_quotes`，`normalize_documents` 对同一个 `LoadedDocument` 又执行一次。纯 `Document` 入口仍需要恢复；如下一轮精简，可仅跳过明确已经恢复的包装结果，不改变原文摘要或 API 含义。此项是重复工作，不影响本轮两项缺陷的修复。

## 本轮证据边界与后续验收

当前展示 `auto_normalization/display_demo/root.yaml --read-only` 实测退出 **0**、`0 would change`，字段/文本预算/关联/dispatch 均通过；它仍只是占位展示，不能为数学正确性背书。本文复现已经过候选、严格结构与文本检查、实际 CLI 写回、只读有限图这条主链。既有写回中断/恢复测试已阅读，不额外重复故障注入全排列。

后续新审阅应优先检查 N1 相邻的明确作用域：Pipe 局部 `for`、Event/view 形参和 Math 行内量词是否同样存在改名捕获；本轮没有把这些尚未独立复现的路径写成已确认缺陷。N1/N2 修复后应由独立审阅者核对一次完整规范化与下一次无变化，特别覆盖 `from` 的 provider/consumer。

不需要新的字段、预算或用户规则决策；上述修复都在已有“仅安全自动修正、歧义停止写回”的授权内。
