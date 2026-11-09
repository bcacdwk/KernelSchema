# 路径索引

| 节点 | 直接父 | workload | 相对父差异 | 选择依据 |
| --- | --- | --- | --- | --- |
| [root](root.yaml) | 无 | 共享 Contract 全域的三维 packed ragged FP8 展开 MLA；Q/KV 等长、等 head，causal、sink 和 LSE 可选。 | 根设计：每个 CTA 处理完整 query tile 的 KV 归约，融合 QK、softmax、PV 与 BF16 写回；排序、队列、K/V 环深度及 tiny 重算在同一框架内表达。 | Model.forward 固定 MODE=0；长度/scatter 排序、Items/SM 网格、KVST 选择及 kernel 内 tiny 条件已纳入 root，无需另设节点（solution.py:1078–1087、1682–1898、2007–2024、2143–2151）。 |

## 覆盖缺口

单 root 负责共享 Contract 全域的设计选择；实际执行仍受下列限制。精确条件、版本和流水线细节以 root.yaml 为准。

- host 拒绝 Q/K/V/out 基址未按 16 字节对齐的输入；共享 Contract 的连续性要求不保证该对齐，没有未对齐 fallback。首次设备检查还要求 SM10x，且 opt-in 共享内存同时满足 Cfg<0,4> 与 Cfg<1,3>，即使固定 MODE=0 也保留此检查。
- 元数据生产流到 side_stream 下载、旧 kernel 读表到缓存表覆写之间没有显式等待；共享表、事件和队列也没有重入协议。不能从本计划推定跨流或跨调用安全。
- 固定入口与共享 Contract 不选择 MODE=1、四维 legacy 输入或缺省 out 分支；这些分支不是本计划的运行时 fallback。
- 验收沿用 root 的 tolerance 来源记录；外部比较器的未定公式、相对误差基准、归一化和检查路由继续记为 source_gap，不据此增加数值或验收通过保证。
