# 编写和检查 schema YAML

本目录提供当前 DOC0–6 的待填写骨架、统一作者入口和只读 Python API。任务方法见[提取](../EXTRACT_SCHEMA.md)、[新建设计](../DESIGN_SCHEMA.md)、[优化设计](../OPTIMIZE_SCHEMA.md)和[实现 schema](../IMPLEMENT_SCHEMA.md)，字段语义以 [schema_docs](../schema_docs/0.common.md) 为准。[四个案例](examples/README.md)提供九份经独立审阅的完整设计、变体目录及同名 PNG，各自注明来源和验证边界。

## 填写与检查

### 1 确认资料与范围

先取得任务给定的输入输出、数学定义、参考入口、误差判据、目标设备与已选实现栈，以及实际 workload 和优化目标。新建设计和优化使用既定 Contract；提取按固定材料建立共享 Contract。修改前提遵守[修改权限](../README.md#2-schema-是什么)，不能为当前方案缩小任务或放宽验收。

资料不足时说明缺项及其影响，继续能够确定的部分。不要编造参考入口、硬件保证、实验结果或设计定义。缺口按[算法验收记录](../schema_docs/1.contract.algorithm_requirement.md#固定实现提取中的验收记录)、[硬件占位](../schema_docs/2.contract.hardware_specification.md#省略与占位)和[数学缺项](../schema_docs/3.design.math_reshape.md#引用与缺项)表达。

固定源码提取按 [P1](../schema_extract/prompt/phases/01_contract.md#第-3-段硬件占位)最小登记有据的执行单元，其余硬件保留占位，显式使用 `--read-only --skip-hardware` 检查。Extract 的阶段检查、执行边界与运行环境见[执行器对齐说明](../schema_extract/EXECUTOR_ALIGNMENT.md)。

### 2 创建文件或进入已有任务

新任务使用独立目录，复制 [starter.yaml](starter.yaml) 到尚不存在的 `root.yaml`，按 [common](../schema_docs/0.common.md#选择条件与索引)建立中文 `dispatch.md`。starter 使用当前字段，`Version: 1` 是新文件的起点；其中 null 是待填写位置，骨架不能通过完整检查。

```sh
mkdir -p path/to/task
cp -n schema_template/starter.yaml path/to/task/root.yaml
```

参照骨架和对应字段说明，命名输入输出、轴、对象和动作，并用引用连接它们。普通正文、设备名称和实际路径表达真实设计与来源。

完整设计集合含 root 最多十份。root 保存唯一 Contract；其他 `<id>.yaml` 填本地元数据、`Dispatch.id/when`、版本与必要 Realization。三个 Design 区域分别完整本地定义或写 `{from: id}` 引用同任务同名整区，整个引用图须无环。参数、局部变化及已知组合按 common 记录，目录不产生隐式 patch。

### 3 填写与修订

按所属任务方法形成数学改写、工作调度、微流水线与必要实现条件；当前定义须涵盖其范围内的全部输出、尾块、空工作和数值分支。每个决定表达一次，说明写在已有说明字段中，数据类型、定义关系与引用范围按对应 DOC 填写。

修改唯一定义后沿实际引用复核受影响的节点。引入定义在使用节点的完整上下文绑定；provider 的元数据、选择条件、版本和反馈不随区域传播。只改共享区域的一部分时，将该整区完整本地化再编辑，不能在 `from` 旁增加覆盖字段。重命名时同步属性引用、动作、事件、交接及 Realization 定位。

设计内容变化按[版本规则](../schema_docs/0.common.md#单份-plan-与元数据)递增实际改动文件的版本；工具的规范化不递增版本。来源身份由 root、自身和三区实际内容链的 raw 文件摘要及 provider/chain 组成；版本相同不证明来源未变。

### 4 检查与处理反馈

需要 Python 3.10+ 和 PyYAML；安装方式及库接口见 [tools](tools/README.md)。从仓库根目录运行：

```sh
python3 -B schema_template/check_yaml.py path/to/task/root.yaml
python3 -B schema_template/check_yaml.py path/to/task/root.yaml --read-only
```

填写设计后运行同一个命令。工具先为同目录直属的全部 `.yaml / .yml` 生成规范化候选，自动处理可确定的大小写、必要引号、普通说明末句号与布局，并同步定义和引用；核对候选后写回，再检查字段、文本预算、`from`、关联与 dispatch。warning 按修改类别列出字段和例子，核对修改即可；超长正文按反馈精简，语义、字段、类型或引用问题按具体诊断修改。

传入任意节点时，规范化和检查都覆盖整个任务，避免只改来源而留下旧引用。文件名、`Dispatch.id`、`from` 来源值保持不变；目录内不要放备份 YAML。`--all` 只用于下方图操作的范围。

仓库定位符写成一个完整目标，保留真实相对路径中的空格与括号。输入文件所在本地仓库与引用身份、修订可准确匹配时，工具核对对应提交中的路径；源码入口及资料内容按原件审阅。

`--read-only` 预览同一规范化计划并检查候选，不写 YAML 或 PNG；修改和候选行号明确标为未写入。`--skip-hardware` 跳过整个硬件区的规则、预算与依赖硬件的关系，保留该区的值；其他结构、成员、索引与动作仍检查。它本身不启用只读。

不能可靠恢复的 YAML 损坏、名称碰撞或歧义会说明原因并停止写回；不会合并重名内容或猜测语义。预算和后续校验错误可以伴随已完成的安全修正。必要引号、合法文本换行、精确数值、anchor/alias/merge 都受保真核对；真实 YAML 注释仍需移入相应字段。没有变化时不写、不改变 mtime；写回保留权限并核对原文摘要，需改写的文件 symlink 会明确拒绝并保留链接。

退出码 `0` 表示所请求操作全部完成并通过；`1` 表示结构、关系、预算、尚未应用的只读规范化差异或有限样本反例；`2` 表示参数、环境、容量、写入或展开未完成。warning 本身不失败。写回后诊断使用实际文件位置；引用区错误同时保留 consumer 路径、provider 原文、引用位置和完整链。

只查看尚未完成的有效草稿，或校验独立阅读视图时使用专用只读模式：

```sh
python3 -B schema_template/check_yaml.py path/to/task/node.yaml --show-effective
python3 -B schema_template/check_yaml.py path/to/view/node.yaml --standalone
```

`--show-effective` 输出含 `validation: not_run` 的 JSON、精确 effective YAML 和来源身份，成功只表示可以展开。`--standalone` 要求单文件自带完整 Contract 和本地 Design，只读校验该文件及可确定的本地仓库定位，不读取兄弟 YAML 或 dispatch。这两种模式不执行格式写回或图操作。正式导出须先完整检查创作任务，再独立验证生成的阅读视图。

### 5 交付

交付当前 raw YAML 集合与 `dispatch.md`，核对每份完整设计的范围、独特价值、必要定义及目录记录的真实基线。源码、资料和实验保留在实际来源，以规定引用定位。独立阅读视图另存，并记录完整 provenance；不能用草稿展开替代验收。

按任务要求固定资料与实验来源，说明检查命令、实际结果及尚未解决的缺口。固定实现提取保留其硬件占位和已有外部验收缺口，依所属流程独立审阅；其他任务仍执行自己的验收要求。

## 微流水线图

有限展开与绘图通过同一入口按需请求：

```sh
python3 -B schema_template/check_yaml.py path/to/task/root.yaml --graph png --set n=4 --worktiles 2
python3 -B schema_template/check_yaml.py path/to/task/root.yaml --read-only --graph check --set n=4 --worktiles 2
```

按实际 plan 填显式样本参数；`n` 只是命令示意，`--worktiles` 默认 1。`--graph check` 增加有限检查，不要求 Graphviz；只有 `--graph png` 需要 `dot` 并生成同目录同名 PNG。`--read-only` 与 `--graph png` 不能合用。默认分析传入节点，加 `--all` 分析所有当前节点。

所有所选图先完成分析再发布。缺少取值、未知条件或容量限制返回 `2`，保留旧图；前置检查、来源复核或渲染失败也不覆盖该图。完整样本发现环时可生成标红诊断图并返回 `1`。PNG 内记录最终 raw 来源摘要、实际来源链、effective identity 和样本；普通检查与 `--all` 不清扫旧图片。

图按有序 `for/if` 展开动作，建立 issue→done、pipe 发起/serial 推进、`async_ordered` 累积完成和显式 Waiting 的四种时点边，同名 noun 不产生数据边。它只检查所选样本、所列 step 内代表 worker 的局部事件关系。Flow 先做静态核对，再按 `start_after`、`ready`、`wait_at` 的实际参数检查所调用的 Event；未调用的 Event 不需要有限取值。缺前驱、成员错误和越过声明窗口是不同情况，不能从已生成动作倒推更小窗口。

## 校验边界

检查器覆盖当前字段与具名标识、共享 Contract、整区引用、可确定的符号/属性关系、提交与交接、事件签名、成员及已知存储数量冲突。开放表达式在普通检查中不执行；有限图遇到无法求值的表达式会明确报告未完成。

数学正确性、公式和正文的语义、设计范围与独特价值、组合相容性、证据真实性以及整体存储安全、活性和性能仍需审阅或相应运行证据。检查通过不把 `unknown:` 或资料占位变成保证。维护测试使用当前格式的合成任务；真实案例的检查参数与覆盖范围见[示例索引](examples/README.md)。
