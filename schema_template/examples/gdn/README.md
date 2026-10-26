# GDN CUE 编写示例

`task.cue` 把 schema_docs 中 GDN 的 Contract、数学改写、五个 phase 和局部流水串成一个可生成的任务。示例固定 HeadCount=8、SequenceLength=4096；数学 chunk 参数仍由 math_reshape 定义。没有新增实现交接时省略 realization，这是完整 plan 的合法形式。

数学与组织的主要来源是文档引用的 FLA 固定提交。这里为说明如何绑定环境，选用文档中的 B200 + CUDA C++/CUTLASS 栈；这不声称 FLA 的原始 Triton 实现已移植到这个栈。wiki 中的显式占位仍待补齐。本例经过结构与关联校验，没有另行进行 GPU 数值、活性或性能验收。

修改只发生在 CUE：

```sh
python3 schema_template/tools/schema.py validate schema_template/examples/gdn
python3 schema_template/tools/schema.py build schema_template/examples/gdn
python3 schema_template/tools/schema.py check schema_template/examples/gdn
```

`root.yaml` 与 `.schema-generated.json` 是只读派生产物。旧的 schema_wiki 案例不因本例通过而被视为已迁移或已符合规范。

当前 YAML 示例对短记录使用行内映射，对 shape、范围和短引用使用行内列表；公式保留多行。排版前后数据经过 CUE 回读比较，未改变数学内容、属性、精度或修订号。CUE 的结构约束同样接受展开写法。
