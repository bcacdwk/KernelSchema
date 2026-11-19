# 路径索引

| 节点 | 直接父 | workload | 相对父差异 | 选择依据 |
| --- | --- | --- | --- | --- |
| [root](root.yaml) | 无 | 自动策略中需要 split-KV 的请求。 | 代表设计：mma.sync 在线注意力写 FP32 分块结果，再由独立 combine 归一化并写回 BF16。 | 初始 P 为非 dual 的 mma 配置且 splits>1；由 make_plan 的 CTA 并行度和工作区容量规则选择。 |
| [mma_direct](mma_direct.yaml) | root | 不分 KV 的全批次 mma.sync 请求，包括未提交 probe 的回落。 | 去掉 split 物化与 combine；在 attention 内加 sink、归一化并写回 BF16，保留分块及 DQ 参数。 | 初始 P 非 dual、非 cfg90 且 splits=1；或 dual 隐藏路径未进入、串行 probe_begin 因 gate 不可用而未提交 probe，改用 P.alt_*。 |
| [probe_hidden_mma](probe_hidden_mma.yaml) | probe_hidden_packed | 隐藏 probe 后选择 mma 余段的请求。 | 仅把余段换成 P.alt_* 的不分 KV mma.sync；已提交的 packed 前缀、side-stream probe 和最终 join 均保留。 | P.dual=1，满足 hidden 的资源、整批次、双波及 fork/probe_begin 门槛；实际 R 不满足 0<R<=2.6。 |
| [probe_hidden_packed](probe_hidden_packed.yaml) | probe_serial_packed | 隐藏 probe 后继续 packed 余段的请求。 | 将 probe 放到 side stream；全批次注意力拆成 caller stream 的 packed 前缀和 side stream 的 packed 余段，保留 fork/join，余段不等待前缀完成。 | P.dual=1；hidden 前置检查及 fork/probe_begin 成功，K=N-8>=1、ceil_div(T,P.gt)>=2*K，且 0<R<=2.6；前缀与余段以 K*P.gt 为界。 |
| [probe_serial_mma](probe_serial_mma.yaml) | mma_direct | 串行 probe 后选择 mma 的全批次请求。 | 增加独立 probe 与 host 轮询；全批次 mma 在同一 caller stream 上接续 probe，不因统计不可用而删除 probe。 | P.dual=1，hidden 在提交 probe 前返回 false；串行 probe_begin 已提交 probe，但实际 R 不满足 0<R<=2.6，选择 P.alt_*。 |
| [probe_serial_packed](probe_serial_packed.yaml) | tc_packed | 串行 probe 后接受 packed 的全批次请求。 | 增加独立 probe、映射 host gate 和轮询选择；全批次 packed 在 caller stream 上等待 probe 完成后执行。 | P.dual=1，hidden 在提交 probe 前返回 false；串行 probe_begin 成功，实际 R 满足 0<R<=2.6。 |
| [tc_heads](tc_heads.yaml) | mma_direct | 单 query 的 head block 已能填满设备的 tcgen05 请求。 | 改用 cfg90 的 TMA gather4、TMEM 和 QIS2 双发射者 QK 分和；每 CTA 处理单 query 的 64 行 head block，直接归一化写回。 | 初始 P 为非 dual 的 cfg90 且 gt=1，对应 make_plan 的 tc_eligible 分支。 |
| [tc_packed](tc_packed.yaml) | tc_heads | 低 head 数、可直接选择多 query 打包的请求。 | 多个 query 合用 64 行；新增按池组织的索引并集及成员标记，保留同 query 内重复次数，并在 gather 前复用 KV ring 构建索引。 | 初始 P 为非 dual 的 cfg90 且 gt>1；make_plan 在备选 mma 不分 KV 且 gt<=2.6 时免 probe，出厂策略下即 gt=2。 |

本索引对应 Model.forward 的出厂自动策略，包含未编入 tcgen05 时的 mma 回落。各节点条件在共享 Contract 全域独立求值，直接父仅表示设计派生来源；精确条件与版本以各 YAML 为准。

probe 的选择依据是本次有界轮询后实际读到的统计结果，轮询未齐仍可能接受 packed；已提交 probe 的工作和接续不因此消失。head 分块、KV chunk/split 数、输出维分片 DQ、ring 深度、combine 线程数及 packed 家族内的每 CTA query 数变化保留为节点参数；节点划分保留物化、生产者和计算提交集合的差异。

非 dual 的直接提交和成功图回放使用同一计算设计。内部录图先同步 caller stream；BeginCapture 失败在 dispatch 前返回，capture/实例化失败不执行录制的 kernel，图失败禁用缓存后接续同一 chosen plan 的直接 dispatch。纯录图不增加计算实例；外层图回放采用录制或显式更新后的选择、标量和绑定，张量内容在执行时读取，录制的 probe 不会在回放时重新选择后段类型。

## 覆盖缺口

- 强制配置、注册 fast-call、诊断构建和修改扩展开关不在这棵出厂策略设计树内；共享 Contract 不据此缩小。
- 路由选中不保证运行前提满足：共享内存预算、opt-in 属性或 TMA tensor-map 编码失败会拒绝执行；源码没有为这些拒绝提供另一条算法 fallback。编译时未启用 tcgen05 的回落由现有 mma 节点覆盖。
- 保留源实现限制：DQ>1 对同一 m/l 地址存在非原子重复写；probe 超时可能与后续调用复用同一 host gate 缓冲发生竞争。计划不据此新增跨 CTA 或跨调用序列化，也不宣称这些源路径已获正确性保证。probe 已提交但轮询未齐的计算及 stream 接续仍属于所选节点。
- root 的 tolerance 保留外部验收器公式、相对误差基准、精度及检查路由未提供的来源记录；这些缺口不被当作数值保证，也不阻止当前设计提取。
