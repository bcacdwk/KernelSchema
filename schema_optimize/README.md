# 独立 Optimize

读取完整 Task、Schema、kernel 和真实反馈，决定保持设计、修复代码落实或产生新设计。方法见 [OPTIMIZE_SCHEMA.md](../OPTIMIZE_SCHEMA.md)，直接只读使用 [Template](../schema_template/README.md)。

```sh
python3 schema_optimize/optimize_schema.py --task path/to/task.md --schema path/to/schema --kernel path/to/kernel --feedback path/to/feedback.json --out path/to/new-optimize
```

`--schema` 接目录或任一 YAML，始终读取完整任务及全部共享使用者。`--kernel` 必须含完整 wrapper/dispatcher/helper/构建源码，单文件须自足。`--feedback` 可重复；result 入口会追读其源码、说明、反馈和原 request/response，按原位置解析相对引用并固定实物。反馈的源码/Schema 摘要如有声明必须匹配。只有真实无源码 Implement `design_gap` 和具体诊断才允许省略 kernel。

默认 Claude CLI + `qwen3.8-max[1m]` + effort `max`；可覆盖 `--model/--effort`，`--max-calls` 默认 2，`--timeout-s` 默认 900。需要 Python 3.10+、PyYAML 6、jsonschema 4 和已配置 Claude CLI。绝对脚本路径可从其他 cwd 运行；help/import 不读认证或启动模型。连接沿用现有配置/登录，禁用 hooks/MCP/插件，不落盘凭据。

| 决定 | Schema | required_fix |
| --- | --- | --- |
| keep | 引用未改输入快照 | false；不代表旧 kernel 通过本次评测 |
| repair | 引用未改输入快照，notes 给具体代码修复 | 恒为 true |
| revise | 完整有效新稿，有实质解析变化 | 必须明确；仅具体必需问题为 true |

Task/完整 root CONTRACT 解析值、kernel、oracle 和原反馈不变。模型只读，host 收稿并在规范化前后核对 Contract，严格重读校验。原稿、真实读取和诊断保留；非法稿/Contract 漂移/关键缺项返回 `decision=null`，不以 keep 隐藏失败。旧成绩不变成本次测量。

输出仅 `input/`、`drafts/NN.json`、新稿 `schema/`、`notes.md`、`result.json`，目录必须全新且不覆盖输入/保护资料。结果含 `status, stop_reason, calls_used, schema, notes, decision, required_fix`；路径相对结果，未有可信诊断的 required_fix 为 null。stdout 一行绝对结果路径，`0/1/2=complete/incomplete/interrupted`。有限调用无 resume，不另外执行 Reviewer 或性能验收。

CPU 检查：`python3 -m unittest discover -s schema_optimize/tools/tests -p 'test_*.py'`。
