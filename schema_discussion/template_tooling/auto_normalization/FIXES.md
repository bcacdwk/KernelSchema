# R1/R2/R3/R4 审阅修正

- **R1**：`normalization.py` 复用原有逐行名称作用域，形参覆盖相邻的普通运算符续行（行末或下一行开头均可），下一定义和独立量词保持各自作用域。需要改名但无法界定的括号多行、悬空运算符或裸尾部使用会给出字段、函数名与 Math 行号，拒绝产生半改公式。没有引入通用数学解析器。
- **R2**：词法替换只保护字符串和点号后的成员，根标识继续查当前作用域名称表。因此登记输入和形参的 `X.item()` 可同步为 `x.item()`；未知外部 `Torch.Op`、成员拼写和字符串保留，实际参数仍按绑定同步。
- **R3**：写入、严格重读、文档摘要比较及最后的磁盘核对统一在原有恢复边界中，成功后才返回真实文档并报告 applied。读取/解析/摘要失败时按 written 列表恢复仍等于本次候选的文件，恢复前原始 bytes（含缺引号、原 lexeme、CRLF）和 mtime 均保留；外部新内容不覆盖，并报告实际恢复数量及无法安全恢复的文件。后续独立字段、预算、关系检查不纳入此边界。
- **R4**：结构规则附带最小 `LITERAL_TYPES` 元数据，覆盖 Version、Input/Output.nullable、Info.Register 和 Storage.member 的纯整数/布尔叶。原源修复阶段统一为 `recover_scalar_quotes`，只对无显式 tag、无首尾空白/转义/多行的普通 quoted lexeme 去引号；同一 YAML core resolver 和严格读取须确认其唯一类型。全部 alias/merge 用途必须同型，原 lexeme、节点复用图、摘要及诊断路径继续核对。null、联合表达式字段、`!!str`、`"1.0"`、`"yes"` 等保留；整数 0 等非法取值不会被改成合法值，去引号后仍由原范围规则报错。`ScalarEdit` 的字符串类型限制未放宽。

R1/R2/R3 新增 7 项维护测试覆盖成功同步、独立定义/量词、歧义拒绝、局部属性根与外部保护、写后读取/摘要失败、读取前后外部编辑和原文恢复。修正前的定向复现失败记录为 `/tmp/kernel-schema-max-fix-red.log`；修正后规范化定向 **17/17** 通过，日志 `/tmp/kernel-schema-max-fix-targeted.log`。原 xhigh 的两个临时复现脚本也确认续行和属性问题已消失，写后失败恢复全部原文，中途失败保留外部编辑和未写文件。

R1/R2/R3 修正后的完整套件只运行一轮，**100/100 通过，21.578 秒**：

```sh
/opt/venv/python3.13.12/bin/python3 -B -m unittest discover -s schema_template/tools/tests -t schema_template
```

日志 `/tmp/kernel-schema-max-fix-suite.log`。40 个 Python 文件通过 Python 3.10 grammar AST 检查（检查器运行于 Python 3.13.12，未运行 Python 3.10）；`git diff --check` 通过。

R4 新增 5 项测试覆盖普通/只读入口、strict API 和 formatter 仍不改类型、原始摘要与精确 lexeme、同型 alias/merge、混合用途与被覆盖 merge 默认值、tag/null/联合类型/非字面文本以及非法数值保留。修正前复现日志 `/tmp/kernel-schema-max-r4-red.log`。

R4 完成后的 normalization/yaml/schema 相关测试 **60/60 通过，14.678 秒**，日志 `/tmp/kernel-schema-max-r4-targeted.log`。随后按上面的完整命令只运行一轮，**105/105 通过，20.979 秒**，日志 `/tmp/kernel-schema-max-r4-suite.log`；40 个 Python 文件的 Python 3.10 grammar AST 与 `git diff --check` 均通过。实际测试环境仍为 Python 3.13.12，未运行 Python 3.10。

本次修正限于规范化/源恢复、同源类型元数据、对应测试及说明。字段集合与待决拼写、examples、wiki、Extract 执行/状态、旧 demo 和 CHANGELOG 草稿均未修改；交回原 xhigh 复核。
