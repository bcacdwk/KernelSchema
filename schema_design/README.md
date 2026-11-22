# 独立 Design

从 Task 自动寻找互补 Schema 近邻，写出完整新 Schema。没有适用近邻也可直接设计。方法见 [DESIGN_SCHEMA.md](../DESIGN_SCHEMA.md)，字段规范只读使用现有 [DOC](../schema_docs/0.common.md) 与 [Template](../schema_template/README.md)。

```sh
python3 schema_design/design_schema.py --task path/to/task.md --out path/to/new-design
```

默认 Claude CLI + `qwen3.8-max[1m]` + effort `max`；`--reference` 可重复，`--model/--effort` 可覆盖，`--max-calls` 默认 3，`--timeout-s` 默认 900。脚本按自身位置定位 checkout，可用绝对脚本路径从其他 cwd 运行。需要 Python 3.10+、PyYAML 6、jsonschema 4 和已配置的 Claude CLI；help/import 不读认证或启动模型。

host 读取 catalog 并发现真正的 root/dispatch，通常一调用选材、一调用写作、最后一调用修具体诊断。固定 Task/绑定资料/原位置与摘要，保留完整 raw/effective/from；只读模型实际查阅 DOC/Template，host 收取完整文件，规范化并严格重读检查。这里只消费已有验收材料，不自动创建 oracle，不运行 kernel。

输出必须是全新目录，不覆盖输入或保护资料。`input/` 保存固定材料与来源清单，`drafts/NN.json` 保存真正回答、实际读取及诊断，`schema/` 保存新候选，`notes.md` 说明边界；只有 `result.json.schema` 非空才是可交接 Schema。坏稿保留不冒充成功。材料过大明确失败，不截断。

初始化后 stdout 仅一行绝对 `result.json` 路径；退出码 `0/1/2=complete/incomplete/interrupted`。结果为 `status, stop_reason, calls_used, schema, notes`，路径相对结果。无 resume；每个调用在启动前计数，超时回收进程组，未知进程不启动替代。

保留原 HOME、Claude 配置目录及登录来源；只将配置中的连接 env 补给子进程，进程环境优先。hooks/MCP/插件禁用，凭据不进入输入、prompt、摘要或产物。CLI 工具限制不是 OS 隔离承诺。

CPU 检查：`python3 -m unittest discover -s schema_design/tools/tests -p 'test_*.py'`。
