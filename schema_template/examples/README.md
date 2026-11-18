# Template 案例

四个案例按当前 DOC0–6 从固定实现人工提取并经独立全文审阅，共九份完整 YAML，各配同名 PNG。阅读时先看各案例 README 了解范围与图样本，再由 `dispatch.md` 查阅完整设计、参数、局部变化与已知组合；每份 YAML 表达一套具体方案，root 保存该算子的唯一共享 Contract。

| 案例与检查入口 | 完整设计 | 主要区别 |
| --- | --- | --- |
| [FA4 prefill](fa4_prefill/README.md) · [目录](fa4_prefill/dispatch.md) | [root](fa4_prefill/root.yaml)、[packed_persistent](fa4_prefill/packed_persistent.yaml)、[split_kv](fa4_prefill/split_kv.yaml) | 静态每 CTA pair 单工作、跨工作常驻与提前输出释放、五路 KV partial 与独立 combine |
| [FA4 decode](fa4_decode/README.md) · [目录](fa4_decode/dispatch.md) | [root](fa4_decode/root.yaml)、[q4_cluster](fa4_decode/q4_cluster.yaml)、[q4_direct](fa4_decode/q4_direct.yaml)、[q1_global](fa4_decode/q1_global.yaml) | 独立 CTA/global combine、cluster 蝶形统计与 BF16 reduce-add、CTA 内直接输出，以及 Q1 双 softmax WG |
| [QSA prefill](qsa_prefill/README.md) · [目录](qsa_prefill/dispatch.md) | [root](qsa_prefill/root.yaml) | 行块 hash union 预处理、ticket 动态分配与跨工作 attention 流水 |
| [GDN prefill](gdn_prefill/README.md) · [目录](gdn_prefill/dispatch.md) | [root](gdn_prefill/root.yaml) | 多序列、64 行 value/state、64-token 成对 chunk 变换与片上递推 |

九份设计中，仅 FA4 prefill 的 `packed_persistent` 与 FA4 decode 的 `q1_global` 通过 `{from: root}` 整区引用共享数学改写，并在本节点绑定各自的工作参数；其余 Design 区域均为本地完整定义。GDN 单序列、decode compact 及 Q1 组合记录在各自 `dispatch.md`；FA4 prefill 默认关闭的 CLC 作为已研究但未展开的独立候选说明。

## 来源

固定源码为 [Kernel Wiki 四题快照](../../kernel_wiki/aka-four-problems-20261019/README.md)（`KernelSchema@3d64ad35900a0e6a66d6f93da34108fbceba0daf:kernel_wiki/aka-four-problems-20261019`），各题的合同、host、核心 kernel、参考实现、输入与既有证据入口汇总见 [SOURCES.md](../../schema_discussion/template_examples_20261117/SOURCES.md)。该快照独立于 Kernel Wiki 44 题主批次，题名相近不代表实现或实验证据可互换。

硬件与实现栈保留已核实的 L20D/SM103、CuTe DSL 4.6.1 等环境事实。`Wiki` 字段中的尖括号占位表示资料尚未完成绑定；检查通过不会将占位或 `unknown:` 变成已知保证。

## 检查与绘图

从仓库根目录使用 Python 3.10+ 与 PyYAML（生成 PNG 另需 Graphviz `dot`）。默认检查覆盖传入文件所在案例的全部 YAML，四例均已填齐硬件规格，无需 `--skip-hardware`：

```sh
python3 -B schema_template/check_yaml.py schema_template/examples/fa4_prefill/root.yaml
python3 -B schema_template/check_yaml.py schema_template/examples/fa4_prefill/root.yaml --read-only
```

绘图或做有限图检查时，按**各案例 README 给出的节点文件与确切 `--set / --worktiles` 参数**逐份运行：将 `--graph png` 换为 `--read-only --graph check` 可在不写盘、不依赖 Graphviz 的情况下检查事件图；不同节点的样本参数不同，不能用同一组 `--all` 参数统跑整套目录。CLI 与库接口说明见 [Template 入口](../README.md#微流水线图)和 [tools](../tools/README.md)。

九张 PNG 的嵌入元数据均匹配对应 raw YAML 摘要、provider/chain、effective identity 与样本参数。图像只展示各 step 代表 worker 在给定样本下的局部事件依赖，不含实测时间比例：例如 FA4 decode 的 `q4_cluster` 主图采用 `when` 内的 `T_b=16`（16 成员各一轮），5/4 轮不等长循环另由 `T_b=70` 图检查覆盖；QSA 主图将每份工作缩至 4 片（低于真实域至少 32 片的下界）以示意跨工作协议；FA4 prefill 的 `rescale` 取值也只是分支示意而非实测数值轨迹。

## 验证边界

四个案例均已完成默认/只读检查、逐份有限图检查、PNG 来源核对与独立全文审阅（未运行 GPU 或 Extract Runner；自动提取流程见 [Extract 手册](../../schema_extract/README.md)）。工具检查不证明全输入域的数值正确性、存储安全、跨 step 活性或实际性能，历史 GPU 记录仅覆盖原测试样本。

| 案例 | 仍保留的来源限制 |
| --- | --- |
| FA4 prefill | native exp2/log/division 的独立误差界；空 split 未写 PartialO 而 combine 仍可预取，保持源码的正权重累加条件 |
| FA4 decode | PageIndex 同槽覆盖晚于 V 捕获未获证明；普通 global combine 读取空 split 未写的 `opartial` 槽，跨布局复用 workspace 不保证其有限；近似数值路径不构成全域误差界 |
| QSA | 未写 `qblk` padding 参与修正投票；单槽 `halfmax` 仅在写后同步，覆写前无 peer 读完保证；缓存无跨 stream 隔离且录图重放不更新 host `base`；fast exp2/rcp 无来源误差界 |
| GDN prefill | descriptor 更新相对旧在途读取的顺序、缓存 workspace 跨调用互斥缺少依据；fast log2/exp2 无可靠误差界 |

各案例均如实记录上述源码行为，不补造同步、清零或实验结论。案例整理时的独立验收与变更记录见 [FINAL.md](../../schema_discussion/template_examples_20261117/FINAL.md) 和 [CHANGELOG.md](../../schema_discussion/template_examples_20261117/CHANGELOG.md)。
