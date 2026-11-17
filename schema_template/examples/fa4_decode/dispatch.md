# 分页分组查询 Decode 注意力

## 范围

固定来源：`KernelSchema@3d64ad35900a0e6a66d6f93da34108fbceba0daf:kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution`，版本 v55。下文 host 指 `qoder_greenfield_decode_attention/operator.py`，paged 指 `gqa_decode_paged.py`，combine 指 `gqa_decode.py`。不采用 44 题批次的相近 FMHA。

唯一 Contract 在 root：B=1..17，Q∈{1,4}，Q≤L_b≤131072，16/1 heads、D256、page64、E4M3FN 输入、BF16 原位输出；完整 ABI、None 参数和原误差判据见 YAML。输入构造的连续独占页、随机幅度和具体 shape 列表不作为新增公共限制。实现接受的 E5M2、其他页/头/tile、sink、LSE、BLASST 不在本题范围。

四份 YAML 各固定一套配置，并限定已建立 native launcher、相同对象/签名/线程及当前设备、非 stream capture 的稳态调用。它们是机制代表，不以 root 覆盖所有 host 条件。现有证据仅证明记录的30个样例；未运行 GPU。普通 global combine 读取空 split 未写 O 槽；初始零和同布局有限旧值不证明跨布局 workspace 复用安全。PageIndex同槽覆盖晚于V捕获的顺序也未获证明，详见案例README；保留源码预取位置，不补同步。

## 完整设计

| 设计 | 具体机制与保留理由 | 关系 |
| --- | --- | --- |
| [root](root.yaml) | Q4、S9、单 softmax WG；独立 CTA 物化 FP32 O/M/L，独立 query-head combine | 唯一 Contract；三区本地；shape_valid[14] 的 B15/L31420/P4067 |
| [q4_cluster](q4_cluster.yaml) | Q4、S16；cluster 蝶形 max/sum，zero_done，BF16 contribution reduce-add | 三区本地；shape_valid[0] 的 B1/L17844/P279 |
| [q4_direct](q4_direct.yaml) | Q4、S1；CTA 内归一化并直接输出，保留 N1 空 parity | 三区本地；域内 B1/L64/P1，未在30个历史样例中运行 |
| [q1_global](q1_global.yaml) | Q1、S16；两个 softmax WG 交替 maximum mutex，分别生产 L parity | Math_reshape 整区来自 root，参数绑定到本节点；其余本地；shape_valid[1] 的 B16/L31413/P4344 |

## 变体与组合

### split 数与 host 条件（参数）

- 基线：[root](root.yaml)、[q1_global](q1_global.yaml) 的 `splitcount/worksplit`、`PARTIAL`、partial workspace 与 `COMBINE`；[q4_cluster](q4_cluster.yaml) 的成员数和 `ClusterReceive`。
- 差异与联动：S 改变 grid、KV 条纹步长、workspace 和 combine 遍历；cluster S 同时改变成员和蝶形轮数。global 最大64，cluster 仅2/4/8/16；global 的 unroll16 仅用于 S%16=0，其余 unroll8。Q4 长序列 wave 表为 mass 10/12/14/16/24/26/28/30/32 对应 S14/12/10/9/6/5/5/9/9；不是新整体设计。
- 条件与依据：host `_select_policy:78–224`。C=ceil(L/256)，mass=B*ceil(Q/2)，E=64P，skew=(2E<B*L)；C≤1 为 direct。Q1 按长序列小批、非 skew 平均长度、skew 长链、B9..17 中长带选择 global 或 cluster；Q4 按 L≤2048、2048<L<24576、长序列 wave/skew 带选择。原函数是完整门控依据；代表的 when 只选其简单真实子域。combine `:495–555`。

### compact Q4 请求内 split（局部）

- 基线：[root](root.yaml) 的 `worksplit/stripe`、`PARTIAL.Decomposition`、`mpartial/lpartial/opartial` 与 `COMBINE.MAIN`。
- 差异与联动：名义 S 和 worker/workspace 不变；每 CTA 读取 batch min/max。S32 用 max(4,ceil(L_b*S/Lmax))；S8 在 min≥3max/4 时 B10/B11 取7/6、B12按比例，其他 B10也按比例，其余保持S。KV 步长改为请求 worksplit；r≥worksplit 走现有中性空 split。combine 同时改 serial 分支，仅 Mpartial>-inf 时读取 O。主要 KQ/PV、parity correction、预分配工作和归并拥有者均保持，局部边界闭合。
- 条件与依据：host `:730–747`：Q4/kernel、L≥24576，S8且B10..12，或S32且B10..17且P>ceil(L/64)+16(B−1)；另须 B<12 或 P<B*ceil(L/64)。paged `:1078–1103`；combine `:531–555`。旧 q4_compact 已删除。

### Q1 的 cluster 与 direct（组合）

- 基线：[q1_global](q1_global.yaml) 的 `SOFTMAXEVEN/SOFTMAXODD`、`maximum`、`p_hi/p_lo`、`exponentialsum`，以及 [q4_cluster](q4_cluster.yaml) 的 cluster 统计/zero_done/output 或 [q4_direct](q4_direct.yaml) 的单块 normalization/output。
- 差异与联动：已知 Q1 双 WG 主循环接上对应归并组织；querytile=1、packedcolumns=16、512线程、无 Q4 register redistribution。统计接口仍是每列 FP32 max/sum 和两 numerator parity。Q1/N1 的 odd WG 只发空 L parity 通知，不计算第二个 softmax。
- 条件与依据：host `_compile_kernel:269–290` 和 `_select_policy` 实际选择上述组合；cluster 例 shape_valid[2] B1/L17845/S16，direct 域内 B1/L64/S1。paged `:621–681,1568–1579,1726–1790`。旧 q1_cluster/q1_direct 已删除；不推广成任意开发开关组合。

### 首次调用与 native launcher（局部）

- 基线：[root](root.yaml)、[q1_global](q1_global.yaml)、[q4_cluster](q4_cluster.yaml)、[q4_direct](q4_direct.yaml) 的提交、out 和 workspace；global 同时包含 `PARTIAL/COMBINE`。
- 差异与联动：cache miss 先正常执行 fmha；允许建立 native launcher 时 clone 当前 out，再执行同一 fmha、设备同步，比较 FP32 相对范数≤1e-2。失败恢复 out；成功缓存 weakref、签名、thread 和 keepalive。命中才是 YAML 的一次稳态 ffi 调用；首次额外计算、比较和可能恢复不能从计时边界消失。
- 条件与依据：host `:474–583,652–843`。stream capture 不建立 native launcher；此时沿用同 kernel 组织。probe 判据不是 Contract 的 atol/rtol，不据此声称首次与稳态调用成本相同。
