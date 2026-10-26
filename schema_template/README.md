# CUE 编写与 YAML 生成

本目录与 `schema_docs/1–6` 共同定义 KernelSchema。Markdown 解释设计含义与审阅要求，CUE 定义允许的数据形态；变更规则时同时修改两者。案例用于解释和发现规则问题，不作为兼容性约束。

规则、填写骨架、工具、测试和示例统一放在本目录。仓库根目录保留 CUE 模块、版本、Makefile、Agent 导航和 CI 等项目配置；以下命令均从仓库根目录执行。

Agent 编写 CUE，生成器输出 YAML，人类和 Coder 阅读 YAML。修改计划、补充反馈或调整版本都回到 CUE。生成 YAML 不是第二份可以独立维护的源文件。

## 文件与入口

| 文件 | 负责内容 |
| --- | --- |
| `common.cue` | 文本、名称、shape、数值表达式、引用等共用形式 |
| `meta_data.cue` | 任务身份、修订号与 dispatch |
| `1_contract_algorithm_requirement.cue` | 输入输出、计算定义、参考入口与误差判据 |
| `2_contract_hardware_specification.cue` | 设备事实、执行单元、实现栈与 wiki 入口 |
| `3_design_math_reshape.cue` | 数学改写、符号属性、精度例外与账本 |
| `4_design_work_schedule.cue` | phase、工作分解、映射、调度与交接 |
| `5_design_tile_pipeline.cue` | lane、资源、视图、递归 pipe 与依赖 |
| `6_realization.cue` | require、prefer、forbid 与有证据的反馈 |
| `plan.cue` | `#Task`、`#RootPlan`、`#DerivedPlan`、`#Design` |
| `starter.cue` | 待填写骨架，由 init 复制到新任务，不参与 kernel 包的规则定义 |
| `tools/` | 初始化、校验、生成、工具安装与精确数值传递 |
| `tests/` | 结构、关联、生成一致性的正反例与测试数据 |
| `examples/` | 展示完整编写与生成流程 |

正常编写使用 `#Task`，以整个任务作为验证单位。单区定义可以单独用于编辑器提示或规则测试，但单区通过不能代替完整任务校验。

任务目录内的 CUE 文件统一声明 `package plan`，通过本仓库模块导入模板：

```cue
package plan

import kernel "kernelschema.local/schema_template:kernel"

plans: kernel.#Task
plans: root: {
    // 在这里填写 root 的完整 meta_data、Contract 和 Design。
}
```

上面仅展示入口；未填写的必填内容会使导出失败。实际起点是 [starter.cue](starter.cue)，完整编写示例是 [GDN](examples/gdn/task.cue)。

本目录中的规则使用 `package kernel`，骨架单独使用 `package starter`。初始化命令复制骨架时将包名改为 `plan`；这样骨架中的待填项不会被下层示例或测试任务继承。导入和直接校验规则时明确选择 `schema_template:kernel`。

`plans` 是 CUE 编写容器，不进入每份 YAML。其键是输出身份：`root` 生成 `root.yaml`，`small` 生成 `small.yaml`。其他公开顶层字段会被生成器拒绝；编写辅助量用 `let` 或 `#` 定义。可以拆成多个 CUE 文件，文件名不决定输出身份。

## 编写流程

运行环境是 Python 3.10+ 标准库与 CUE v0.17.1。安装脚本支持 macOS/Linux 的 amd64 与 arm64，从官方固定 release 下载，校验仓库记录的 SHA-256，只写入 `schema_template/.tools` 本地缓存。

```sh
make setup
python3 schema_template/tools/schema.py init schema_wiki/my_batch/my_task
```

然后阅读相关 Markdown 与 CUE 定义，填写新建的 `task.cue`。把 `string`、`int` 等尚未具体化的字段替换为实际决定；不填写虚构来源或为了通过校验而补默认值。标量 shape 明确写 `[]`，整任务只有一份工作时明确写 `partition: {}`。

```sh
schema_template/.tools/cue/v0.17.1/cue fmt ./schema_wiki/my_batch/my_task
python3 schema_template/tools/schema.py validate schema_wiki/my_batch/my_task
python3 schema_template/tools/schema.py build schema_wiki/my_batch/my_task
python3 schema_template/tools/schema.py check schema_wiki/my_batch/my_task
```

- `validate`：具体化 CUE，强制套用 `#Task`，运行关联检查，不写产物。漏写模板 import 也不能绕过这一入口的校验。
- `build`：先完整验证所有节点、渲染所有输出，再写 YAML 和 `.schema-generated.json`。失败时不会用未通过的设计覆盖已有 YAML。
- `check`：从 CUE 重新生成期望结果，与现有 YAML、生成清单逐字节比较。缺失、过期、手改或多余的 YAML 都导致失败。
- `--all`：处理 `schema_wiki` 与 `schema_template/examples` 中的 CUE 任务；归档和未迁移的手写案例不冒充已通过验证的任务。

生成的 YAML 与清单应一起提交。清单记录 CUE 版本、本目录 CUE 源文件、模板和生成工具的摘要，以及每份产物的内容摘要与修订号。它不进入 Plan，也不是实验记录。

具体化数值以精确十进制值在工具之间传递，不先转换成 Python binary64 浮点数。输出层将短记录、shape、范围和短引用列表排成 `{...}` / `[...]`，数学正文保留多行；写入前由 CUE 回读，核对数据未发生变化，包括系数和文本空白。

行内与展开 YAML 都能通过相同的 CUE 约束；紧凑格式是可读性选择，不是新增字段或放松校验。以下命令可直接检查现有 YAML 的 root 结构，不会重排文件：

```sh
schema_template/.tools/cue/v0.17.1/cue vet -c ./schema_template:kernel -d '#RootPlan' schema_template/examples/gdn/root.yaml
```

这条原生 CUE 命令检查结构与 CUE 中的约束；任务级加载及额外的关联检查仍由工具承担。

生成器把 YAML 设置为本地只读；Git 不保存写权限，重新克隆后由 `build` 恢复。真正的仓库一致性检查是 `check`，CI 执行 `make check`。拥有文件系统权限的人仍能改文件，因此这里的“不可修改”是编写契约加上可检测的生成一致性，不是文件系统访问控制。

发现生成文件被手改时，普通 `build` 拒绝覆盖。先把真正需要的修改写入 CUE；如需丢弃手改并恢复派生结果，使用：

```sh
python3 schema_template/tools/schema.py build schema_wiki/my_batch/my_task --repair
```

`--repair` 只处理生成清单已拥有的产物，不采用手写 YAML，不放松模板、不跳过修订检查。首次生成遇到已有手写 YAML 会拒绝覆盖；旧库迁移是独立工作。

## root、派生节点与复用

root 定义唯一的 Contract。派生节点完整声明自己的 Design，不能出现 `contract`，也没有运行时“继承父节点设计”的机制。`parent` 只表达设计来源；其缺省值是 root。

在同一任务 CUE package 中增加节点时，可使用另一个文件：

```cue
package plan

import kernel "kernelschema.local/schema_template:kernel"

plans: small: {
    meta_data: {
        version: 1
        dispatch: {
            id: "small"
            when: string
        }
    }
    design: kernel.#Design
}
```

这仍是待填写的片段。`#Task` 从 root 给派生节点统一 `name / describe`，校验节点键与 `dispatch.id` 一致；有多个节点时 root 也必须填写 `dispatch.when`。各节点的条件在完整契约域上独立求值；互斥与全覆盖需要审阅，父节点条件不会自动叠加。

CUE 可以用字段引用或局部定义复用完全相同的内容，作用类似文档中的 YAML anchor。导出后每个节点拥有完整 Design。不要用未审阅的父节点覆盖规则、数组下标或源码路径代替稳定的 Plan 对象身份。

每份 plan 的数据内容发生变化时，递增它自己的 `meta_data.version`；生成器相对已有清单拒绝内容变化但版本未递增的更新。CUE 注释或格式改变只更新来源摘要；没有改变 plan 数据时不要求递增。只改 root 的 Contract 不要求给数据未变的派生节点递增版本。此检查不替代版本库历史审阅。

## 严谨与开放的边界

固定结构均为 CUE definitions，拒绝额外字段；具名映射允许增加实例名，但条目结构固定。资源类别、方向、lane mode 等封闭词汇严格检查；dtype、设备机制、动作契约、数学公式与设计理由保持文档规定的开放空间。

只在引用需要消歧的作用域要求名称唯一。Contract 对象与步骤 ID 需要唯一；硬件的 `execution` 与 `feature` 已由路径区分，同名不会产生歧义，不施加跨表唯一性。不同 worker 的同名资源也不因此合并。

省略保留文档原有语义：例如资源默认一槽、lane 默认 sync、派生节点的 parent 默认 root。模板不会为了“填满”输出而插入这些可选字段。标量 `shape: []`、单份工作 `partition: {}`、单槽整槽视图 `view: {X: {}}` 和 partition range 的无界上端点是有语义的例外。

文档允许的 wiki 占位与 math_reshape 的 `unknown:` 被保留；Contract 其他事实不能借这套记号变成待填项。CUE 已具体化只说明值能够导出，不说明资料已就绪、误差界已知或引用已经核验。

数学公式和运行时表达式仍为文本。辅助检查器只解析结构化整数表达式与已约定的动作/事件记法，不执行 Python、数学公式或 GPU 代码。它检查已登记的 phase、lane、动作族与部分索引作用域；不会靠关键词搜索从自然语言“证明”绑定已完成。

完整范围见 [规则覆盖与审阅边界](coverage.md)。通过生成入口不能替代数学与数值验证、外部资料核查、展开后的事件图检查、资源可行性或性能测量。

## 维护与验收

```sh
make fmt
make build
make check
```

测试包含合法最小结构、开放硬件名称、空值例外、深层嵌套 pipe、root/派生关系、条件字段、错误引用、索引作用域，以及生成失败保护、只读产物、手改检测、修订检查和重复生成。

修改规范时先确认规则的设计必要性；同步修改 CUE、对应 Markdown、覆盖说明和正反例，再重新生成受影响的 CUE 案例。不要通过全局开放字段、任意字符串兜底或固定几层的 pipe 展开掩盖缺口。
