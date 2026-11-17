# FA4 Decode 模板案例

入口为 [root.yaml](root.yaml)，选型和其余 host 配置见 [dispatch.md](dispatch.md)。目录保留四份设计：Q4/S9/global、Q4/S16/cluster、Q4/S1/direct、Q1/S16/global。root 定义唯一 Contract；Q1 global 的整个数学区引用 root，其工作参数和双 softmax 流水线在本节点定义。

固定源码为 `KernelSchema@3d64ad35900a0e6a66d6f93da34108fbceba0daf:kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution` 的 v55。公开域是 B=1..17、Q∈{1,4}、Q≤L_b≤131072；参考要求对返回值和原位 out 都做 FP32 比较，双方有限且 `abs(actual-reference)≤0.06+0.04*abs(reference)`，并检查 shape/dtype、out alias、未声明输入不变。2026-10-21 harness 在运行前把 out 填 NaN。历史30/30记录只覆盖那些输入，本轮未运行 GPU。

数值链包括 leading/residual 两次 FP8 PV、未量化分母、两 numerator/sum parity。N=1 仍初始化和读取空 parity；N=0 仍准备 CTA/TMEM/page lookup 并发布中性统计。cluster 用蝶形 max/sum、zero_done 和先转 BF16 的 reduce-add，不能等同于唯一 FP32 汇总后一次转换。

独立全文审阅仍保留两项源码边界：普通 global combine 读取空 split 未覆盖的 O 槽，workspace 跨 `(S,B,Q)` 布局复用不保证该槽有限；PageIndex 的下一组预取早于 V-turn 等待，尚未证明同槽覆盖晚于 V 捕获。YAML 保留源码行为，没有补造清零、guard 或同步。详见 [写作交接](../../../schema_discussion/template_examples_20261117/fa4_decode/WRITE.md) 与 [PageIndex 顺序复核](../../../schema_discussion/template_examples_20261117/fa4_decode/pageindex_order.md)。

每图展开一个代表 worker；cluster 的一个 worker 包含全部16个 CTA 成员；图中相同轨迹可合并显示并列出成员编号。global 的 COMBINE 面板表示该 worker 所消费的全局 partial 接口，其余 producer worker 未全部画出。图是依赖示意，不证明 GPU 正确性、竞态自由或延迟比例。有限样本保留四个真实 operand slices，两个 issue-turn 协议及资源退休。

| 节点 | 主图参数与实际输入 | 额外检查 |
| --- | --- | --- |
| root | `N=5, splitcount=9, worktiles=1`；shape_valid[14]，B15/Lmax31420/P4067，request0 L_b9430/T_b37/r0 | 最长请求 L_b31420/T_b123/r0/N14；N0/N1 |
| q1_global | `N=5, splitcount=16, worktiles=1`；shape_valid[1]，B16/Lmax31413/P4344，request0 L_b17857/T_b70/r0 | 最长请求 L_b31413/T_b123/r0/N8；N0/N1 |
| q4_direct | `N=1, splitcount=1, worktiles=1`；域内 B1/Q4/L_b64/P1/r0 | 空 sum/numerator parity 的通知、读取和 TMEM 释放 |
| q4_cluster | `T_b=16, splitcount=16, worktiles=1`；when 内 B1/Q4/L_b4096/P64，完整16成员各一轮 | 主输入 B1/L_b17844/T_b70：成员0..5各五轮、其余四轮；另检查 T_b1 的1活跃/15空成员 |

从仓库根目录使用 `/opt/venv/python3.13.12/bin/python3 -B schema_template/check_yaml.py schema_template/examples/fa4_decode/root.yaml` 正常化并检查整个目录；加 `--read-only` 可验证无待写变化。生成单图用 `--graph png --set N=5 --set splitcount=9 --worktiles 1`；将节点和参数替换为上表对应项。`--read-only --graph check` 只检查有限图；`--read-only` 不能与 `--graph png` 同用。cluster 生成命令使用 `q4_cluster.yaml --graph png --set T_b=16 --set splitcount=16 --worktiles 1`。四份同名 PNG 均已按最终 YAML 重绘；准确命令、计数和历史渲染限制见 [独立审阅](../../../schema_discussion/template_examples_20261117/fa4_decode/REVIEW.md)。

cluster 的 T_b16 图仅展示一轮与 N1 空 parity；不等长循环和环复用由 T_b70 的完整图检查覆盖。T_b1 空成员诊断来自同机制合法输入 B2/Q4/seq_lens=[64,17844]/P280 的 request0，位于 dispatch 的参数范围，**不在本文件 B1 的 when 内**。它只作额外检查，未用作当前 PNG。此前 T_b70/T_b17 的 Graphviz 超时记录保留，没有为绘图修改 when、splitcount 或成员长度。
