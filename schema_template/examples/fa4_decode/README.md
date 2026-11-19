# FA4 Decode 模板案例

默认阅读入口为 [root.yaml](root.yaml)，选型条件、局部变化与组合关系见 [dispatch.md](dispatch.md)。本目录收录四份完整设计：`root`（Q4/S9/global）、[q4_cluster.yaml](q4_cluster.yaml)（Q4/S16/cluster）、[q4_direct.yaml](q4_direct.yaml)（Q4/S1/direct）与 [q1_global.yaml](q1_global.yaml)（Q1/S16/global）。`root` 保存唯一共享 Contract；`q1_global` 通过 `{from: root}` 复用数学改写，在本节点定义工作参数与双 softmax WG 流水线。

固定源码为 `KernelSchema@3d64ad35900a0e6a66d6f93da34108fbceba0daf:kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution`（v55），公开域为 `B=1..17`、`Q in {1,4}`、`Q <= L_b <= 131072`。参考判据在 FP32 下检查返回值与原位 `out` 均有限且满足 `abs(actual-reference) <= 0.06 + 0.04*abs(reference)`，同时核验 `shape/dtype`、`out` 别名及未声明修改的输入保持不变。

主计算采用 leading/residual 两次 FP8 PV、未量化分母及双奇偶分子与指数和：`N=1` 仍初始化并读取空奇偶槽，`N=0` 仍完成 CTA/TMEM/page lookup 准备并发布中性统计。`q4_cluster` 通过蝶形交换汇总 max/sum、配合 `zero_done` 并对先转 BF16 的分片贡献做 reduce-add，不等同于单次 FP32 全局求和后再窄化。

源码限制：普通 global combine 会读取空 split 未覆写的 `opartial` 槽，跨 `(S,B,Q)` 布局复用 workspace 时不保证旧值有限；PageIndex 下一组预取早于 V-turn 等待，同槽覆写晚于 V 捕获尚未获证明（见 [PageIndex 顺序分析](../../../schema_discussion/template_examples_20261117/fa4_decode/pageindex_order.md)）。

每张 PNG 展示单个代表 worker 的局部事件依赖：`q4_cluster` 的单个 worker 含全部 16 个 CTA 成员（相同轨迹合并列出成员号），global 方案的 `COMBINE` 面板只画该代表 worker 消费的全局 partial 接口。

| 节点 | 主图参数与实际输入 | 可选边界样本 |
| --- | --- | --- |
| root | `N=5, splitcount=9, worktiles=1`；shape_valid[14]，B15/Lmax31420/P4067，request0 L_b9430/T_b37/r0 | 最长请求 L_b31420/T_b123/r0/N14；N0/N1 |
| q1_global | `N=5, splitcount=16, worktiles=1`；shape_valid[1]，B16/Lmax31413/P4344，request0 L_b17857/T_b70/r0 | 最长请求 L_b31413/T_b123/r0/N8；N0/N1 |
| q4_direct | `N=1, splitcount=1, worktiles=1`；域内 B1/Q4/L_b64/P1/r0 | 空 sum/numerator parity 的通知、读取和 TMEM 释放 |
| q4_cluster | `T_b=16, splitcount=16, worktiles=1`；when 内 B1/Q4/L_b4096/P64，完整16成员各一轮 | 主输入 B1/L_b17844/T_b70：成员0..5各五轮、其余四轮；另可测 T_b1（1活跃/15空成员） |

从仓库根目录运行规范化检查与单图生成（将节点与 `--set` 参数替换为上表对应项；只做有限图检查时将 `--graph png` 换为 `--read-only --graph check`）：

```sh
python3 -B schema_template/check_yaml.py schema_template/examples/fa4_decode/root.yaml --read-only
python3 -B schema_template/check_yaml.py schema_template/examples/fa4_decode/root.yaml --graph png --set N=5 --set splitcount=9 --worktiles 1
python3 -B schema_template/check_yaml.py schema_template/examples/fa4_decode/q4_cluster.yaml --graph png --set T_b=16 --set splitcount=16 --worktiles 1
```

`q4_cluster` 主图取 `when` 内的 `T_b=16`（16 成员各一轮，含 `N=1` 空奇偶）；不等长循环与环复用可用 `--read-only --graph check --set T_b=70 --set splitcount=16 --worktiles 1` 检查。`T_b=1`（1 活跃 / 15 空成员）对应多请求输入 `B=2, Q=4, seq_lens=[64,17844], P=280` 的 request 0，属于 `dispatch.md` 参数范围而非 `q4_cluster` 自身 `B=1` 的 `when`。
