# 分页稀疏分组查询注意力

## 范围

收录固定 BF16 FIT v3（`fc30f74`）的行块预处理与持久主 kernel，限定单一当前 stream 上的串行 eager 调用。共享 Contract 取自 step-15 的两个 CP rank 拟合选择规律；空 union 不在输入有效域内，但动态取票仍可能使部分已启动 CTA 领到初始终止票而无实际工作。缓存 workspace 不提供跨 stream 隔离，CUDA Graph 捕获重放不推进 host ticket `base`。

## 完整设计

| 设计 | 具体机制与保留理由 | 关系 |
| --- | --- | --- |
| [root](root.yaml) | 行块 hash union、双列半 softmax、TMEM S/P 别名、ticket 动态分配及跨工作 KV/Q 流水；保留完整输入到输出与缓存生命周期 | 阅读入口，保存唯一 Contract |

## 变体与组合

### 两槽分数环（局部）

- 基线：[root](root.yaml) 的 `ATTENTION.MMA` 与 `accumulator`。
- 差异与联动：构造器参数设为 `s_stages=2` 时，`ATTENTION.MMA` 中 QK 相对 PV 的领先距离由两片降为一片（`ahead=1`），`accumulator` 内 `s(n)/p(n)` 别名槽由三槽改为两槽且 `y` 起始列同步前移；双槽 KV 环、双 softmax WG 分工与 ticket 领取协议不变。
- 条件与依据：`QSAForward._mma_role` 在 `s_stages == 2` 时切换 `ahead=1`；当前 host 固定实例化默认三槽（`s_stages=3`），无运行时分流。来源：`KernelSchema@3d64ad35900a0e6a66d6f93da34108fbceba0daf:kernel_wiki/aka-four-problems-20261019/qsa_prefill/aka_solution/qwen38_core_qsa_cutedsl/qsa_kernel.py`。
