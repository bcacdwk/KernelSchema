# YAML 编写与校验

Agent 参考 [schema.cue](schema.cue) 和 [schema_docs/1–6](../schema_docs/)，直接编辑 YAML。CUE 定义结构规范，Python 补充复杂关联检查；字段语义与角色权限由六份 MD 说明，案例不覆盖规范。

## 填写与检查

需要 Python 3.10+ 标准库和固定版本 CUE。以下命令从仓库根目录运行：

```sh
make setup
python3 schema_template/tools/schema.py init schema_wiki/my_batch/my_task
# 直接填写 root.yaml，再校验整个任务。
python3 schema_template/tools/schema.py validate schema_wiki/my_batch/my_task
```

`init` 将唯一的 [starter.yaml](starter.yaml) 复制为可编辑的 `root.yaml`，不覆盖已有 plan。按任务重命名骨架中的对象，把 `null` 待填项替换为真实内容；这些待填项不能通过校验，不是合法的缺省值。

一个任务一个目录，每个节点一个 YAML 映射文档。`validate` 读取目录直属的全部 `.yaml` / `.yml`，以文件名去掉扩展名作为节点 ID，同一 ID 不能同时使用两种扩展名。root 保存唯一的 Contract；派生节点写完整的元数据与 Design，不复制 Contract、不继承父节点的 Design。文件组织、身份和修订规则详见 [文档 1](../schema_docs/1.contract.algorithm_requirement.md)。

短记录推荐行内映射，shape、范围和短引用列表推荐行内列表，数学正文保留多行；紧凑与展开写法同样合法。数据内容变化时递增该节点的 `meta_data.version`，纯排版变化无需递增。

`validate` 不改写内容、权限或修改时间，不补字段、不重排 YAML。错误标出文件和字段路径，CUE 提供源位置时一并保留。多个目录可一次传入，任何任务失败都返回非零状态；任务也可以位于仓库之外。

```sh
python3 schema_template/tools/schema.py validate path/to/task_a path/to/task_b
```

## 文件组织

| 文件 | 用途 |
| --- | --- |
| `schema.cue` | 全部核心规则，按任务、共用值、Contract、Design、Realization 分章；`package kernel`，任务入口 `#Task` |
| `starter.yaml` | 待填写骨架 |
| `tools/schema.py` | init / validate、YAML 加载与错误定位 |
| `tools/validate_task.cue` | 将按文件名组织的 YAML 接入 `#Task`，保留 CUE 源位置 |
| `tools/validate_relations.py` | 具名引用、图关系和必要记法解析 |
| `tools/json_value.py` | 精确十进制 JSON 传递及测试数据编码，避免 binary64 舍入 |
| `tools/install_cue.py` | 官方固定 release 的 SHA-256 校验与本地安装 |
| `tests/` | 正反例与最小合成 YAML fixture |
| `examples/gdn/root.yaml` | 完整的紧凑 YAML 示例 |

`#Task` 的映射键来自文件名，容器只用于校验，不写进 YAML。适配层不另定义字段规则。安装器支持 macOS/Linux 的 amd64、arm64，二进制保存在忽略提交的 `.tools/`；CUE 模块、版本、Makefile 与 CI 配置位于仓库根目录。

## 校验边界

- **CUE**：固定结构、类型、必填和条件字段、封闭词汇、清晰的局部约束，以及节点身份、共享任务描述、phase 对应等直接可表达的关系。可选字段的语义缺省值不自动写入数据。
- **Python**：Contract 与数学符号引用、父节点和启动依赖图、共享 worker 与交接关系、资源区域、域及动作/事件的记法和部分索引作用域。所有派生节点使用 root 的共享 Contract；数值解析和比较保留精确十进制值，不执行公式。
- **审阅与运行证据**：dispatch 条件的互斥与全覆盖、文本中的数学和策略引用、外部资料及实验依据、修订号相对历史递增、修改权限。通过校验不证明数学等价、数值正确、资源可行或性能；检查器不展开事件图，因而不证明具体动作实例存在、存储复用安全或活性。wiki 占位和 `unknown:` 仍表示尚未补齐的证据。

[GDN 示例](examples/gdn/root.yaml) 连接完整 Contract、数学改写和五个 phase，固定 HeadCount=8、SequenceLength=4096。数学与组织引用 FLA 固定提交，示例环境选用 B200 + CUDA C++/CUTLASS，不表示原始 Triton 实现已完成移植。wiki 占位尚待补齐；示例通过结构与关联检查，未进行 GPU 数值、活性或性能验收。测试 fixture 中的 `testing.invalid` 引用仅标识合成数据。

## 维护

```sh
make fmt
make check
# 只检查指定任务时，也可使用：
make validate TASKS="path/to/task"
```

`make check` 检查核心 CUE 与适配层格式、正反例测试及维护中的 YAML 示例。历史 `schema_wiki` 不在默认验收范围内；显式传给 `validate` 时仍使用相同规则，不自动迁移或放宽约束。修改规范时同步更新 `schema.cue`、对应 MD 和正反例；只改组织方式时保持原有规则与公开定义。
