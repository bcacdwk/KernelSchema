# 检查器维护

日常创建、提取、修改和检查 schema 见[作者入口](../README.md)；本篇只说明检查器实现、依赖、测试和规则维护，规范以 [schema_docs](../../schema_docs/0.common.md) 为准。检查器服务于填写和阅读，不把 plan 当作待编译的程序。

新增拒绝条件前，先说明它防止了什么阅读歧义、遗漏或引用冲突。不能只因实现方便而限制设计表达，也不能仅因检查器已有某个限制，就把它补成规范。设计取舍和无法直接判定的数学事实留给审阅。

## 目录与职责

```text
tools/
├── schema.py              任务文件发现、检查编排、失败状态
├── yaml_source.py         YAML 解析、精确数值与源位置
├── diagnostics.py         统一诊断、字段路径、值显示
├── requirements.txt       解析依赖
├── rules/
│   ├── __init__.py        明确列出的检查顺序
│   ├── values.py          可复用的字段类型、值形式检查
│   ├── structure.py       按 DOC 0–6 分组的字段结构
│   ├── common.py          错误收集、图与区间辅助检查
│   ├── declarations.py    任务身份、声明配对与关联遍历前提
│   ├── symbols.py         Contract、数学符号与属性引用
│   ├── schedule.py        launch、共享 worker、handoff
│   └── pipeline.py        lane、资源、动作/事件与索引作用域
└── tests/
    ├── support.py         合成任务、变异与只读断言
    ├── test_structure.py  字段、条件必填和开放表达
    ├── test_relations.py  声明、引用、图与跨文件关系
    ├── test_yaml.py       解析、精确数值、文件组织和只读性
    ├── test_cli.py        单文件入口、诊断和退出状态
    ├── test_examples.py   现有示例通过作者入口接受检查
    └── fixtures/minimal/  最小合成测试底稿
```

依赖方向是 `check_yaml.py → schema.py → yaml_source / rules → diagnostics`。解析器和诊断不导入规则；规则只检查已解析的数据，不读写 plan、不打印、不决定进程退出。字段结构通过后，先检查声明是否完整，再按符号、工作调度、微流水线的顺序检查关联。

输入文件名决定节点 ID，所在目录决定任务范围；文件符号链接不使检查切换到目标文件的任务目录。无论传入 root 还是派生节点，都会检查同目录直属的全部 `.yaml / .yml`。共享 Contract 只用于解析引用，不写进派生节点，也不继承父节点 Design。

## 回归与修改

需要 Python 3.10+ 和 PyYAML。在所用 Python 环境中安装依赖，从仓库根目录运行：

```sh
python3 -m pip install -r schema_template/tools/requirements.txt
```

运行全部维护检查：

```sh
python3 -B -m unittest discover -s schema_template/tools/tests -t schema_template -v
```

这条命令同时运行规则、解析、入口和现有示例测试。`-t schema_template` 指定测试包的导入根，`-B` 避免生成字节码缓存。测试底稿中的 `testing.invalid` 引用仅标识合成数据；历史 `schema_wiki` 不在默认验收范围内。

修改时按职责定位：

- 字段、必填项、值形式：修改 `rules/structure.py`，共用值检查放进 `rules/values.py`，补 `test_structure.py`。
- 引用或一致性条件：修改对应关联模块；只有影响后续遍历安全的声明前提才放进 `declarations.py`，补 `test_relations.py`。
- YAML 解析或源位置：修改 `yaml_source.py`，补 `test_yaml.py`；诊断显示改 `diagnostics.py`，补 `test_cli.py`。
- 新工具保持独立用途；需要现有校验能力时复用 `schema.py`，日常作者入口保持单文件参数。

字段规则统一接收 `(value, path, errors)`；关联规则通过 `CheckContext.require` 收集诊断。路径统一使用 `field_path` 表达特殊映射键，缺失字段标记 `missing=True`。公共入口和解析器不承载新的领域规则。

同时保留“确有歧义必须拒绝”和“表达合理应当接受”的测试。测试不为规则提供独立正当性；撤销无必要的规则时，应将相应反例改为正例，而不是继续迁就旧实现。

## 检查范围与诊断

保留固定字段、非空必填、shape 轴名、共享 Contract、节点与 phase 身份、声明引用和明显数值范围冲突等检查；静态工作归属、多槽视图和模板资料配对等条件防止设计信息遗漏。内建事件名不能兼作局部里程碑，避免作用域歧义。

dtype、外部入口名称和数学表达式保持开放；表达式不编译、不执行、不限定函数白名单。能直接识别的数字仍检查取值范围，简单索引检查已声明变量的作用域；其他数学写法交给审阅。自然语言不按固定句式判定，`repeat` 不强制创建 `steady`，但 lane 必须有实际动作。

解析只接受普通 YAML 数据，不执行自定义标签；拒绝重复显式键和递归 alias。anchor、alias 和本地 merge 仅复用字段值。`on / off / yes` 和日期保持字符串；小数直接从原文构造 Decimal，不经过 binary64。

错误报告文件、字段路径和真实源位置，列表下标从 0 开始；alias 子字段可能定位到锚点定义。缺失字段只报告文件和字段，不编造行号。退出状态为 `0` 通过、`1` 输入存在问题、`2` 参数或工具环境使检查未完成。检查不下载依赖，不修改文件内容、权限或修改时间。

作者需要了解的检查能力及人工审阅边界见[校验边界](../README.md#校验边界)。

## 示例状态

维护的任务、规范化依据和未决项见[示例索引](../examples/README.md)。现有示例测试自动遍历各任务目录，通过作者入口检查全部节点并验证文件未被改写。结构与关联检查通过不代表已完成 GPU 数值、活性或性能验收。
