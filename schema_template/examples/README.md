# Template 案例

四个案例按当前 DOC0–6 人工提取并经独立全文审阅，共九份完整 YAML，各有同名 PNG。先读各案例 README 了解范围与图样本，再用 `dispatch.md` 查完整设计、参数、局部变化和已知组合；每份 YAML 表达一套具体方案，root 保存该案例唯一 Contract。

| 案例与检查入口 | 完整设计 | 主要区别 |
| --- | --- | --- |
| [FA4 prefill](fa4_prefill/README.md) · [目录](fa4_prefill/dispatch.md) | [root](fa4_prefill/root.yaml)、[packed_persistent](fa4_prefill/packed_persistent.yaml)、[split_kv](fa4_prefill/split_kv.yaml) | 静态每 CTA pair 单工作、跨工作常驻与提前输出释放、五路 KV partial 与独立 combine |
| [FA4 decode](fa4_decode/README.md) · [目录](fa4_decode/dispatch.md) | [root](fa4_decode/root.yaml)、[q4_cluster](fa4_decode/q4_cluster.yaml)、[q4_direct](fa4_decode/q4_direct.yaml)、[q1_global](fa4_decode/q1_global.yaml) | 独立 CTA/global combine、cluster 蝶形统计与 BF16 reduce-add、CTA 内直接输出，以及 Q1 双 softmax WG |
| [QSA prefill](qsa_prefill/README.md) · [目录](qsa_prefill/dispatch.md) | [root](qsa_prefill/root.yaml) | 行块 hash union 预处理、ticket 动态分配与跨工作 attention 流水 |
| [GDN prefill](gdn_prefill/README.md) · [目录](gdn_prefill/dispatch.md) | [root](gdn_prefill/root.yaml) | 多序列、64 行 value/state、64-token 成对 chunk 变换与片上递推 |

只有 prefill 的 `packed_persistent` 和 decode 的 `q1_global` 整区引用各自 root 的 `Math_reshape`；使用节点各自绑定工作参数。其余 Design 区域完整本地定义。GDN 单序列、decode compact/Q1 组合等差异定位在各自目录；prefill 默认关闭的 CLC 保留为未展开独立候选。

## 来源

固定材料为 [Kernel Wiki 四题快照](../../kernel_wiki/aka-four-problems-20261019/README.md)，引用 `KernelSchema@3d64ad35900a0e6a66d6f93da34108fbceba0daf:kernel_wiki/aka-four-problems-20261019`。合同、host、核心 kernel、reference、输入与既有证据入口见 [SOURCES.md](../../schema_discussion/template_examples_20261117/SOURCES.md)。该快照独立于 Kernel Wiki 的 44 题主批次，相近题名不代表实现或实验可互换。

硬件和实现栈保留与源码匹配的 L20D/SM103、CuTe DSL 4.6.1 等已核实事实。YAML 中的正式 Wiki 尖括号占位仍表示资料绑定未完成；检查通过不会补齐未知信息。

## 检查与绘图

从仓库根目录使用 Python 3.10+ 与 PyYAML；生成 PNG 另需 Graphviz `dot`。默认检查覆盖目标所在案例的全部 YAML，当前四例均无需 `--skip-hardware`：

```sh
python3 -B schema_template/check_yaml.py schema_template/examples/fa4_prefill/root.yaml
python3 -B schema_template/check_yaml.py schema_template/examples/fa4_prefill/root.yaml --read-only
```

逐份绘图或有限图检查时，采用**各案例 README 的文件和确切参数**。用 `--graph png` 生成同名图；将其换成 `--read-only --graph check` 可只检查事件图，不需 Graphviz。不同设计使用不同样本，不用同一组 `--all` 参数覆盖整个集合。工具与 API 见 [Template 入口](../README.md#微流水线图)及 [tools](../tools/README.md)。

九张 PNG 已核对最终 raw YAML 摘要、provider/chain、effective identity 和样本。图只描述每 step 代表 worker 的有限局部事件关系，无实测时间比例：decode cluster 当前图是 when 内的 `T_b=16`、16 成员各一轮，不等长 5/4 轮另以 `T_b=70` 检查；QSA 的两份工作各四片低于真实至少 32 片的下界，仅是协议示意；prefill 的 rescale 分支也不是输入数值轨迹。各样本的实际输入、空工作及环复用边界以案例 README 为准。

## 验证边界

本轮完成默认/只读检查、逐份有限图检查、PNG 来源核对和独立语义审阅；未运行 GPU 或 Extract Runner。工具检查不证明全部输入的数值正确性、存储安全、跨 step 活性或性能；历史 GPU 结果只保留原实验范围。Extract 自动流程仍见[执行器对齐说明](../../schema_extract/EXECUTOR_ALIGNMENT.md)。

| 案例 | 仍保留的来源限制 |
| --- | --- |
| FA4 prefill | native exp2/log/division 的独立误差界；空 split 未写 PartialO 而 combine 仍可预取，保持源码的正权重累加条件 |
| FA4 decode | PageIndex 覆盖与 V 捕获的顺序未获保证；普通 global combine 读取空 split 未覆盖 O 槽，跨布局 workspace 复用不保证其有限；近似数值路径不构成全域误差保证 |
| QSA | 未写 Q padding 参与投票、HalfMax 缺少 peer 读完后覆盖保证；跨 stream 缓存与捕获重放的 base 更新限制；fast exp2/reciprocal 无来源误差界 |
| GDN prefill | descriptor 更新相对旧在途读取的顺序、缓存 workspace 跨调用互斥缺少依据；fast log2/exp2 无可靠误差界 |

设计如实保留这些行为，未补造同步、清零或实验结论。最终验收与本轮简短记录见 [FINAL.md](../../schema_discussion/template_examples_20261117/FINAL.md) 和 [CHANGELOG.md](../../schema_discussion/template_examples_20261117/CHANGELOG.md)。
