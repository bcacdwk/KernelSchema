# 检查器维护

日常创建、提取、修改和检查 schema 见[作者入口](../README.md)；本篇只说明检查器实现、依赖、测试和规则维护，规范以 [schema_docs](../../schema_docs/0.common.md) 为准。检查器服务于填写和阅读，不把 plan 当作待编译的程序。

新增拒绝条件前，先说明它防止了什么阅读歧义、遗漏或引用冲突。不能只因实现方便而限制设计表达，也不能仅因检查器已有某个限制，就把它补成规范。设计取舍和无法直接判定的数学事实留给审阅。

字段规则只接受现行名称和枚举值，不设置旧字段别名、版本回退或自动改写。更名时同时更新规范、模板、现行实例和测试；测试应明确拒绝旧名，即使同一条目也填写了新名。输入中的自定义对象名不受这项字段限制。

## 目录与职责

```text
tools/
├── schema.py              任务文件发现、检查编排、失败状态
├── check_policy.py        未归档 schema_wiki 的显式硬件占位豁免
├── yaml_source.py         YAML 解析、精确数值与源位置
├── diagnostics.py         统一诊断、字段路径、值显示
├── errors.py              工具环境与处理容量错误
├── pipeline_syntax.py     共用动作记法、pipe 声明与词法索引作用域
├── requirements.txt       解析依赖
├── pipeline_graph.py      可选的有限展开、检查与绘图入口
├── pipeline/              动作解析、表达式、索引范围、事件图、展示及产物管理
├── rules/
│   ├── __init__.py        明确列出的检查顺序
│   ├── values.py          可复用的字段类型、值形式检查
│   ├── structure.py       按 DOC 0–6 分组的字段结构
│   ├── hardware.py        同一套硬件字段及缺失信息的限定处理
│   ├── common.py          错误收集、图与区间辅助检查
│   ├── declarations.py    任务身份、声明配对与关联遍历前提
│   ├── symbols.py         Contract、数学符号与属性引用
│   ├── schedule.py        launch、共享 worker、handoff
│   └── pipeline.py        role、资源、动作/事件与索引作用域
└── tests/
    ├── support.py         合成任务、变异与只读断言
    ├── test_structure.py  字段、条件必填和开放表达
    ├── test_relations.py  声明、引用、图与跨文件关系
    ├── test_yaml.py       解析、精确数值、文件组织和只读性
    ├── test_cli.py        单文件入口、诊断和退出状态
    ├── test_examples.py   现有示例通过作者入口接受检查
    ├── test_hardware_policy.py  硬件豁免的范围、诊断和只读性
    ├── test_pipeline_graph.py   有限展开、反例和输出检查
    ├── test_pipeline_domains.py  作用域、坐标等价写法与默认依赖
    ├── test_pipeline_output.py   PNG 来源、失败清理与输出冲突
    ├── test_tool_limits.py       两个公共入口的容量与错误分类
    └── fixtures/minimal/  最小合成测试底稿
```

依赖方向是 `check_yaml.py → schema.py → check_policy / yaml_source / rules → diagnostics`。解析器和诊断不导入规则；规则只检查已解析的数据，不读写 plan、不打印、不决定进程退出。字段结构通过后，先检查声明是否完整，再按符号、工作调度、微流水线的顺序检查关联。

输入文件名决定节点 ID，所在目录决定任务范围；文件符号链接不使检查切换到目标文件的任务目录。无论传入 root 还是派生节点，都会检查同目录直属的全部 `.yaml / .yml`。共享 Contract 只用于解析引用，不写进派生节点，也不继承父节点 Design。

## 回归与修改

需要 Python 3.10+ 和 PyYAML；生成 PNG 另需 Graphviz 的 `dot`。在所用 Python 环境中安装解析依赖，从仓库根目录运行：

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

dtype、外部入口名称和数学表达式保持开放；日常入口 `check_yaml.py` 不编译或执行表达式，不限定函数白名单。能直接识别的数字仍检查取值范围，简单索引检查已声明变量的作用域；其他数学写法交给审阅。自然语言不按固定句式判定，`repeat` 不强制创建 `steady`，但 role 必须有实际动作。

解析只接受普通 YAML 数据，不执行自定义标签；拒绝重复显式键和递归 alias。anchor、alias 和本地 merge 仅复用字段值。`on / off / yes` 和日期保持字符串；小数直接从原文构造 Decimal，不经过 binary64。

错误报告文件、字段路径和真实源位置，列表下标从 0 开始；alias 子字段可能定位到锚点定义。缺失字段只报告文件和字段，不编造行号。退出状态为 `0` 通过、`1` 输入存在问题、`2` 参数或工具环境使检查未完成。检查不下载依赖，不修改文件内容、权限或修改时间。

作者需要了解的检查能力及人工审阅边界见[校验边界](../README.md#校验边界)。

## 硬件占位豁免

仓库内的 [schema_wiki/check_policy.json](../../schema_wiki/check_policy.json) 显式设置 `allow_missing_hardware: true`。只对该仓库未归档 `schema_wiki` 内的任务生效；`archived`、现行示例和仓库外的任务仍使用完整检查。删除配置或改为 `false` 即恢复完整检查，不接受其他豁免开关或旧版字段。

豁免允许 `contract.hardware_specification` 整段或其中字段省略、写 `null`、空字符串、空映射，或保留 `unknown:` / `<...@pending:...>`。检查器保留原文，不代填设备、容量或能力；依赖硬件资料才能判断的执行单元登记、模板资料配对暂缓检查。已填写的非空内容仍检查当前字段名与类型。

算法、工作分配、动作、索引、成员数量和引用继续检查。其他区域的资料占位不会因这个设置自动通过。两种命令都会显示豁免说明，生成的 PNG 标题与图片元数据也保留这条说明；配置格式错误返回 `2`，不伪装为 YAML 设计错误。

## 微流水线图的维护边界

用法见[微流水线图](../README.md#微流水线图)。`pipeline_graph.py` 复用 `schema.py`。规则与图工具共用 `pipeline_syntax` 的动作记法和词法域声明；`actions` 实例化动作，`expressions` 求有限整数与域，`coordinates` 恢复目标坐标，`windows` 判断声明窗口，`graph/model` 绑定事件并构建依赖，`render` 把内存中的 DOT 交给 Graphviz 生成 PNG，`output` 管理图片来源与原子写入。规模只由命令行显式给出；[示例总览](../examples/README.md#生成-png)保存复现命令，生成器不读取它，也不按算子名称分支。

默认运行只写 YAML 同目录的同名 PNG。图片内保存来源文件名、内容指纹、展示取值和检查状态；不记录本机绝对路径，也不把旧图片当作输入配置。生成前确认同名图片归属，并使旧图失效；无法确认归属的同名 PNG 会保留并报告冲突。`--all` 同时清理已确认由工具生成、但 YAML 已删除的孤立图片。图片完整生成后原子替换目标文件，失败不留下旧成功图或半张图。`--check-only` 完全只读，不需要 Graphviz，不清理或写入图片。两条路径共用相同的展开和依赖检查。

能直接确定的错误由日常检查器尽早报告，例如引用未实例化的 role 成员、同一成员上的事件等待自身或自身稍后的完成。索引展开后才能判断的重复动作、悬空引用和环由图工具检查。跨成员依赖、`done` 等待自身的 `issue` 等合理写法应当保留。

`GraphError` 表示具体展开存在问题；`ExpansionError` 表示样例取值不足或工具不能完成展开，不能把绘图器支持的表达式子集变成新的 schema 语法限制。修改这些行为时，在 `test_pipeline_graph.py` 中同时保留错误用例、合理边界和只读性回归。

索引窗口来自动作声明时的域及其表达式，不从已生成动作的最小值、最大值倒推。局部声明遮蔽外层同名域；事件引用只取被引用动作族实际使用的域，不混入无关动作的局部定义。需要绑定的同名域在这些声明中不一致时，返回 `ExpansionError`。手写动作列表可通过同一动作族中的未移位引用（例如 `X(i)`）关联到已声明的 `i` 域；缺少这种依据时也不能反推一个较小范围。

窗口计算先从完整参数关系恢复目标的外层坐标，再用目标坐标求变长内层域。可唯一恢复的仿射参数使用精确有理数求解；flatten 仍使用其声明的坐标映射。未出现在动作参数中的外层域按完整声明范围有限枚举；需要恢复却无法唯一确定的坐标关系返回 `2`，不能猜用源事件的外层坐标。一个候选窗口无法确定时，不能仅因另一个候选不含该点就把整个动作族当作越界。固定引用、域内缺失和拼错动作仍是输入错误。

展开对动作与边数量、索引依赖深度、坐标求解规模和窗口求值总量设置工具上限。达到上限或无法唯一恢复外层坐标时返回 `2`，不要求作者为工具删改合法的数学表达。前置规则的环检查使用迭代遍历；YAML 读取或规则检查遇到实现容量限制，也按工具无法完成报告。已确认的依赖环或递归 alias 仍按输入错误报告。

## 示例状态

维护的任务、规范化依据和未决项见[示例索引](../examples/README.md)。现有示例测试自动遍历各任务目录，通过作者入口检查全部节点并验证文件未被改写。结构与关联检查通过不代表已完成 GPU 数值、活性或性能验收。
