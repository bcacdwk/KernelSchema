# 分页稀疏分组查询注意力

## 范围

收录固定 BF16 FIT v3 `fc30f74` 的预处理与持久主 kernel，采用串行 eager 调用。Contract 来自 step-15 的两个 CP rank、四个拟合索引样本；空 union 不在题目域内，动态分配仍可能让某个已提交 CTA 没有工作。缓存不提供跨 stream 隔离，录图重放没有 host ticket base 更新保证。

## 完整设计

| 设计 | 具体机制与保留理由 | 关系 |
| --- | --- | --- |
| [root](root.yaml) | 行块 hash union、双列半 softmax、TMEM S/P 别名、ticket 动态分配及跨工作 KV/Q 流水；保留完整输入到输出与缓存生命周期 | 阅读入口，保存唯一 Contract |

## 变体与组合

### 两槽分数环（局部）

- 基线：[root](root.yaml) 的 `ATTENTION.MMA` 与 `accumulator`。
- 差异与联动：源码构造器允许 `s_stages=2`，QK 领先 PV 从两片改为一片，同时调整 S/P 环槽、Y 起始位置与阶段索引；KV 双槽、softmax 分工和 ticket 协议不变。
- 条件与依据：`QSAForward._mma_role` 明确选择 `ahead=1`；当前 host 仅实例化默认三槽，不是运行时 selector。来源：`KernelSchema@3d64ad35900a0e6a66d6f93da34108fbceba0daf:kernel_wiki/aka-four-problems-20261019/qsa_prefill/aka_solution/qwen38_core_qsa_cutedsl/qsa_kernel.py`。
