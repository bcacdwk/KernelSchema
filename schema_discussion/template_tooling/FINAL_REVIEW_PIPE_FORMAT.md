# 最终独立复核：pipe、有限图与 YAML

本轮以 AGENTS、PLAN、FIELD_MAP、IMPLEMENTATION 和 DOC0/1/5 为依据，独立检查共享动作/控制语法、静态索引及成员检查、有限事件图、PNG 发布，以及 YAML 权威读取、预算与格式器。未改 examples、wiki、Extract 脚本或状态，也未增加语言、兼容层或依赖。

## 已修正

- YAML merge 的列表包装此前绕过 tag 检查，`<<: !bad [*base]` 和 `<<: !!map [*base]` 会被接受。现在要求普通序列 tag，保留正常 alias/merge 语义；两种复现加入现有读取回归。
- 整个 Waiting 目标动作族不发生时，曾在确定无等待前解析未使用的开放数学实参。现在只继续检查成员寻址，不执行这些实参；目标实际发生时，无法有限求值仍返回 ExpansionError。回归同时保留缺席目标上的错误成员检查。
- 在 `g=flatten[t,i]` 且某份工作的 `i` 域为空时，固定首项目标 `y(g[t,0])` 曾错误地要求该工作非空。现在只有动作声明确实绑定该空域时才跳过该目标等待；证明使用声明域及所引用工作的坐标，不以缺失事件日志推断。回归验证 t0 空、t1 非空时仍保留 t1 的 Released 边，并验证非空域中的固定 99 继续报错。DOC5 的对应澄清由核心审阅代理同步。

## 验证

```sh
PYTHONPATH=schema_template python3 -B -m unittest tools.tests.test_pipeline_graph tools.tests.test_pipeline_output tools.tests.test_yaml -v
```

本次负责的 27 项全部通过；14 个实现模块通过 Python 3.10 语法检查，负责文件的 `git diff --check` 通过。完整合并套件留给主代理统一运行。

另在 `/tmp` 合成独立任务，实际调用 `/usr/bin/dot` 并读图核验：PNG 为 870×778、78,410 字节，两个动作、四个明确时点，完成与后续 issue 之间的环用红色显示。PNG 的 owner、最终 source_manifest、effective_identity、样本及 `acyclic=false` 与所检查来源一致；analyze 前后的 effective 数据相同。临时目录已清理，没有向仓库或正式案例保存图片。

既有测试还核对四种时点、兄弟 for 顺序、ordered 跨动作族/工作接续、serial 不排空先前 async、缺失前驱/移位窗口、成员、动态 worktile 查询，以及来源漂移/无效 PNG 保旧、Decimal/anchor/alias/merge 保真、格式幂等、注释/引号、Unicode/逻辑行和逐路径预算。本轮没有发现必须保留为未解决项的额外实现缺陷；有限求值的显式样本和局部图证明范围维持既定边界。
