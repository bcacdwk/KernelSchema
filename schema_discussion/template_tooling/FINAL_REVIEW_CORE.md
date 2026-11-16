# 最终核心验收

本轮独立阅读当前字段表、DOC0–6、结构/符号/调度规则、整区引用、只读 API 与唯一 CLI，未以正式案例作为输入。重点核对了 root Contract、三区合并 DAG、consumer 绑定、实际来源链与身份、副本隔离、十份上限、standalone、格式写入门禁和 PNG 发布编排。

修复两类可复现问题：

- `Constant` 依赖曾复用流水线索引的提示函数：直接 `x[0]` 漏掉输入对象，固定求和的局部 `x` 却被误当运行时输入。现在只识别简单且可确定的值依赖；外部限定名、局部绑定及其他数学记法保留审阅边界。直接运行时依赖与确定的常量环仍报错，依据 DOC1 的固定值条件及 DOC0 的开放数学表达规则。
- `REALIZATION` 的同一角色可写成 `MAIN.MAIN` 或 `MAIN.Role.MAIN`，Contract 对象也可按区域或具名登记表定位；此前 Feedback 按未统一的字符串误报不属于 Prefer。现在统一这些已解析路径。仅在公式或 strategy 中定义、无法由登记表唯一解析的目标保留语义审阅；已知不同对象及没有 Prefer 的反馈仍报错，依据 DOC6 的同一上游对象与引用等价规则。

配合流水线验收，在 DOC5「依赖的索引窗口」末条明确原有目标发生域优先语义：某份工作目标声明域为空时，固定首项不生成等待；非空域的固定越界引用仍须报错。未新增字段或执行规则。

验证：

```sh
PYTHONPATH=schema_template python3 -B -m unittest tools.tests.test_schema tools.tests.test_design_resolution tools.tests.test_cli tools.tests.test_dispatch -v
```

上述核心、引用、入口与目录模块共 38 项通过。随后扩充同一 Feedback 回归的等价 Contract 路径和公式目标子例，两个本轮新增测试定点复测通过。另以合成任务独立核对 `MAIN.ready(0)` 通过，成员 1 越界、缺失形参和 `kernel_done(0)` 均被准确拒绝。

当前 38 个 Python 工具/测试文件通过 Python 3.10 语法解析；13 份现行入口与 DOC 文档的相对文件链接（排除 fenced 示例）无失效目标；`git diff --check` 通过。最终全套计数由主代理在另一验收完成后统一记录。

本轮只修改 `rules/symbols.py`、`tests/test_schema.py`、上述 DOC5 说明和本记录。examples 全目录、wiki、Extract 脚本与状态、用户 CHANGELOG 末尾草稿均未触碰。未发现待修的核心交付阻塞；公式含义、未登记语义目标、跨 step 覆盖/活性及证据真实性仍是规范明确保留的审阅边界。
