# 路径索引

| 节点 | 直接父 | workload | 相对父差异 | 选择依据 |
| --- | --- | --- | --- | --- |
| [root](root.yaml) | 无 | 工作数较大的 ragged FP8 attention；每份工作对应一个 head 的一个 128 行 query 块。 | 代表设计：首份工作预分配，随后原子补领；缓冲和环索引跨工作接续，下一 Q 可与旧 PV/输出尾段重叠。 | Model.forward → launch_impl 中 grid_items 取 nsm_l 的分支；工作数由当前 CuQ 的 query 块数乘 H 得出，阈值使用缓存的设备 SM 数，非正时取源码 fallback 148。 |
| [bootstrap](bootstrap.yaml) | root | 同一契约下工作数较小的 attention；每个 CTA 只处理一个 head/query 块。 | 保持单工作块数学与缓冲深度规则；CTA 数改为工作数，每 CTA 仅预分配一份；取消补领、Queue 服务、终止发布和跨工作 Q 预取，保留本工作内依赖并完成后退出。 | 同一 launch_impl 中 grid_items 取 items 的分支，包含阈值等号；使用当前 CuQ 派生的工作数与 n_sm_launch（非正时取 148）判定。 |

bootstrap 直接取 root 为设计起点：两者的单工作块计算相同，差异在工作分配、终止和跨工作接续。各节点条件在共享 Contract 全域独立求值；精确条件和版本以各 YAML 为准。

两路径的 Model.forward 前缀均读取当前长度与前缀、在 host 构造工作表并侧流上传；主流等待上传完成后提交一个注意力核。固定入口选择 FP8 PV，FP16 V 预转换及其备用主核不另列路径。

以下变化在各节点内参数化：

- 几何表采用 head-major 重块优先排列，或按 KV 块数稳定降序排列；排序门控只重排同一组独立工作。
- K/V 环按 scatter 门控取三槽或四槽，S/P 环仍为三槽。三槽以对应的 `pv_done` 归还 K/V，四槽保留按 K/V 周期通知的 `kv_free`；两者都等待上次占用同槽的 PV 完成。
- causal、sink 与 LSE 分别控制可见性、归一化和写回。短请求追加 FP32 tiny-row 修正，原有 QK/PV 主路径仍执行。

## 覆盖缺口

两节点覆盖固定三维 THD 契约下的 host grid 两臂。`launch_impl` 对 Q/K/V/out 基址要求 16 字节对齐，未对齐输入会被拒绝；这项实现限制未被写成共享契约的数学约束，固定入口没有替代路径。实现还检查 SM10x 与足够的 opt-in shared memory，检查失败即终止，没有其他硬件分支。

路径覆盖不构成全域数值通过保证。root 保留 TMEM 行寻址和持久描述符复用的源实现限制，以及外部比较器定义缺失的验收记录。
