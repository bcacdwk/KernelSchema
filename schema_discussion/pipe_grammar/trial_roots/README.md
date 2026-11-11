# 十个 root 的简单控制流试写

本轮使用唯一规则 **pipe-control-v0.2**：有序列表加 `for(...)`、`if(...)`、`while(...)`，每个控制节点的内容直接是列表。旧试写和旧试行规则已替换；十个 root 分别由十个全新、max effort 的子代理对照原始 YAML 与固定 Kernel 源码重新编写。

[Pipe 规则](../README.md#pipe-规则) · [输入与源码清单](manifest.json) · [统一验证结果](validation.json)

先看 YAML，再看同名说明中的源码对应与边界核对。每份覆盖原 root 的全部 phase 和 role；数学、工作调度、资源、成员、执行模式、动作契约和里程碑沿 `experiment.baseline` 继承。需要调整依赖时，文件给出对应 role 的完整 `waits_for` 替换。

| 算子 | 新 pipe | 核对说明 | for / if / while |
| --- | --- | --- | ---: |
| 模板 FA4 prefill | [YAML](template_fa4_prefill.yaml) | [说明](template_fa4_prefill.md) | 5 / 0 / 0 |
| 模板 FA4 decode | [YAML](template_fa4_decode.yaml) | [说明](template_fa4_decode.md) | 12 / 2 / 0 |
| 模板 GDN | [YAML](template_gdn_prefill.yaml) | [说明](template_gdn_prefill.md) | 17 / 0 / 0 |
| 模板 QSA | [YAML](template_qsa_prefill.yaml) | [说明](template_qsa_prefill.md) | 11 / 0 / 0 |
| 提取 GDN | [YAML](extract_gdn.yaml) | [说明](extract_gdn.md) | 2 / 1 / 0 |
| 提取 GEMM | [YAML](extract_gemm.yaml) | [说明](extract_gemm.md) | 6 / 2 / 0 |
| 提取 MLA | [YAML](extract_mla.yaml) | [说明](extract_mla.md) | 9 / 0 / 0 |
| 提取 Sparse | [YAML](extract_sparse.yaml) | [说明](extract_sparse.md) | 1 / 0 / 0 |
| 提取 FMHA | [YAML](extract_fmha.yaml) | [说明](extract_fmha.md) | 2 / 0 / 0 |
| 提取 MoE | [YAML](extract_moe.yaml) | [说明](extract_moe.md) | 19 / 2 / 0 |

**十个 root 均可按现有动作粒度表达，共 84 个 for、7 个 if。** 没有使用 while：已定义的有序工作序列与局部遍历已经足够，底层等待轮询和既有动作内部实现未被人为提升。while 的语义保留在规则中，并有独立控制检查，不以这十个 root 冒充对所有反馈循环的验证。

建议先看三个例子：FA4 prefill 用五个 for 和直接首尾动作呈现各角色的职责；GEMM 保留工作／产品／分片的真实层次，把末片读取、提前归还、最后计算直接排在一起；MoE 用两个 if 表达可选等待，没有为它们再登记 q 的门控域。GEMM 的两个互斥 if 使用整份工作期间保持不变的 Full(t)，不依赖隐式分支配对。

条件相关的依赖随目标动作的发生域实例化。FA4 decode 删除 active 域，替换 Value、Softmax、Correction 的依赖；提取 GDN 删除 a 域，替换 prep.Main 的依赖；MoE 替换两个 MMA 的依赖。消去的控制索引使用实际值 0 保留原事件身份。目标不存在时没有等待实例；窗口内目标存在却缺前驱仍报错，拼错角色、名称和签名也不被掩盖。

统一检查通过：

- 21 个 phase、58 个 role 完整覆盖；输入、规则和源码散列保持一致。
- 53 组局部参数、323 条角色动作轨迹与原 root 同序、同身份。
- 对域或依赖有修改的 phase，21 次有限依赖图比较一致。
- 控制检查覆盖索引重绑定、flatten 坐标冲突、while 零轮与反馈继续，以及非法语法和条件依赖负例；完整项目见验证记录。

参数见 [cases.json](cases.json)。其中有用于语法展开的抽象范围，说明中明确区分局部分片几何与完整 host dispatch 的可达性。验证不执行动作实现或 GPU，不证明一般终止性、全部运行时并发或数值正确性；原有源码与旧契约的差异也保留在各说明中。

从仓库根目录重跑：

```bash
python schema_discussion/pipe_grammar/tools/check_trial_roots.py --write-report
```

`sources/` 是核对用的固定源码输入。四模板及三个提取算子来自所引 Git 修订；GEMM、MLA、MoE 的源码逐文件匹配所引私有修订的输入锁散列。完整来源在 manifest 中。
