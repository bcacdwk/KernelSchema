# Template 案例

四个案例按 DOC0–6 从固定实现提取，共九份完整 YAML，各配同名 PNG。先看各案例 README 了解范围与图样本，再由 `dispatch.md` 查阅完整设计、参数、局部变化与已知组合；每份 YAML 表达一套具体方案，`root.yaml` 保存该算子的唯一共享 Contract。

| 案例入口 | 完整设计 | 主要区别 |
| --- | --- | --- |
| [FA4 prefill](fa4_prefill/README.md) · [目录](fa4_prefill/dispatch.md) | [root](fa4_prefill/root.yaml)、[packed_persistent](fa4_prefill/packed_persistent.yaml)、[split_kv](fa4_prefill/split_kv.yaml) | 静态每 CTA pair 单工作、跨工作常驻与提前输出释放、五路 KV partial 与独立 combine |
| [FA4 decode](fa4_decode/README.md) · [目录](fa4_decode/dispatch.md) | [root](fa4_decode/root.yaml)、[q4_cluster](fa4_decode/q4_cluster.yaml)、[q4_direct](fa4_decode/q4_direct.yaml)、[q1_global](fa4_decode/q1_global.yaml) | 独立 CTA/global combine、cluster 蝶形统计与 BF16 reduce-add、CTA 内直接输出，以及 Q1 双 softmax WG |
| [QSA prefill](qsa_prefill/README.md) · [目录](qsa_prefill/dispatch.md) | [root](qsa_prefill/root.yaml) | 行块 hash union 预处理、ticket 动态分配与跨工作 attention 流水 |
| [GDN prefill](gdn_prefill/README.md) · [目录](gdn_prefill/dispatch.md) | [root](gdn_prefill/root.yaml) | 多序列、64 行 value/state、64-token 成对 chunk 变换与片上递推 |

九份设计中，仅 FA4 prefill 的 `packed_persistent` 与 FA4 decode 的 `q1_global` 通过 `{from: root}` 整区引用共享数学改写，并在本节点绑定各自的工作参数；其余 Design 区域均为本地完整定义。GDN 单序列、decode compact 及 Q1 组合记录在各自 `dispatch.md`；FA4 prefill 默认关闭的 CLC 作为已研究但未展开的独立候选说明。

## 来源

固定源码为 [Kernel Wiki 四题快照](../../kernel_wiki/aka-four-problems-20261019/README.md)（`KernelSchema@3d64ad35900a0e6a66d6f93da34108fbceba0daf:kernel_wiki/aka-four-problems-20261019`），各题的合同、host、核心 kernel、参考实现、输入与证据入口见 [SOURCES.md](../../schema_discussion/template_examples_20261117/SOURCES.md)。该快照独立于 Kernel Wiki 44 题主批次，实现与实验证据不互换。硬件与实现栈记录 L20D/SM103、CuTe DSL 4.6.1 等环境事实，`Wiki` 字段保留尖括号待绑定占位。

## 检查与绘图

从仓库根目录运行（需 Python 3.10+ 与 PyYAML；生成 PNG 另需 Graphviz `dot`）。传入任一节点即可规范化或只读检查同目录全部 YAML：

```sh
python3 -B schema_template/check_yaml.py schema_template/examples/fa4_prefill/root.yaml
python3 -B schema_template/check_yaml.py schema_template/examples/fa4_prefill/root.yaml --read-only
```

绘图或有限图检查须按**各案例 README 给出的节点文件与确切 `--set / --worktiles` 参数**逐份运行；将 `--graph png` 换为 `--read-only --graph check` 可在不写盘、不依赖 Graphviz 的情况下检查事件图。不同节点的样本参数不同，不能用同一组 `--all` 参数统跑整套目录。CLI 与库接口说明见 [Template 入口](../README.md#微流水线图)和 [tools](../tools/README.md)。

图像仅展示各 step 代表 worker 在给定样本下的局部事件依赖，不含实测时间比例，也不替代全输入域数值或 GPU 运行验证；具体调用前提与源码限制见各案例 README 和 `dispatch.md`。
