# 编写和检查 schema YAML

先读[公共约定](../schema_docs/0.common.md)，再按[第 1–6 篇](../README.md#阅读入口)核对 Contract、填写 Design 和必要的实现交接。角色权限见仓库 README；Contract 使用任务给定的内容。

## 填写与检查

从仓库根目录为新任务复制骨架，不覆盖已有文件：

```sh
mkdir -p path/to/task
cp -n schema_template/starter.yaml path/to/task/root.yaml
```

按任务重命名骨架中的对象，将 `null` 替换为真实内容。骨架不是完整字段清单：还需按各篇“填写条件”补齐适用字段，没有内容的可选字段直接省略。骨架中的 `Dimension` 表示待确定的独立工作维度；整个任务只有一份工作时，使用 `partition: {}` 并省略 `worktile_order`，见[工作分解](../schema_docs/4.design.work_schedule.md#工作分解)。

需要 Python 3.10+ 和 PyYAML。填写后运行唯一的日常检查命令：

```sh
python3 schema_template/check_yaml.py path/to/task/root.yaml
```

也可传入派生节点。检查范围始终是该文件所在目录直属的全部 `.yaml / .yml`；派生节点共用同目录 root 的 Contract，文件组织见[公共约定](../schema_docs/0.common.md#派生节点与-dispatch)。任务可以位于仓库之外；输入路径相对当前目录解析，在其他目录调用时使用检查脚本的绝对路径。

缺少 PyYAML 时，检查器会给出一条安装命令；安装后重跑同一检查命令。检查本身不下载或安装依赖。

成功时输出任务目录和节点数；失败时按文件、字段路径和修改提示处理，已有字段同时显示 YAML 行列位置。缺字段可能没有行号；修改后重跑同一命令。退出状态为：`0` 通过，`1` 输入未通过检查，`2` 参数或工具环境问题导致检查未完成。

检查不补字段、不重排、不改写 YAML，也不改变文件权限或修改时间。填写与修订仍遵守[通用写法](../schema_docs/0.common.md#通用写法)和[版本规则](../schema_docs/0.common.md#单份-plan-与元数据)。

## 校验边界

通过检查表示结构及已实现的关联检查通过，不证明数学等价、数值正确、资源可行、存储复用安全、活性或性能。dispatch 条件的互斥与覆盖、外部资料和实验依据仍需核实；wiki 占位和 `unknown:` 仍是待解决的缺口，不能靠猜测补齐。

examples 仅作可选参考，不是学习规范的前提。维护检查器时再读 [tools/README.md](tools/README.md)。
