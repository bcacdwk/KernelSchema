# 分页分组查询 Decode 注意力

## 范围

固定来源：`KernelSchema@3d64ad35900a0e6a66d6f93da34108fbceba0daf:kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution`（v55）。下文 host 指 `qoder_greenfield_decode_attention/operator.py`，paged 指 `gqa_decode_paged.py`，combine 指 `gqa_decode.py`；不混用 44 题主批次的相近实现。

唯一共享 Contract 保存在 `root.yaml`：`B=1..17`，`Q in {1,4}`，`Q <= L_b <= 131072`，16/1 heads、`D=256`、`page=64`、FP8 E4M3FN 输入与 BF16 原位输出；完整 ABI、`None` 参数和误差判据以 YAML 为准。源码支持但题目未启用的 E5M2、其他页/头/tile、sink、LSE 与 BLASST 不在本目录范围。

四份完整 YAML 均限定已建立 native launcher、相同张量对象/签名/线程与当前设备、非 stream capture 的稳态调用，`root` 不默认覆盖其余 host 分支。源码限制：普通（非 compact）global combine 会读取空 split 未覆写的 `opartial` 槽，跨布局复用 workspace 时不保证旧值有限；PageIndex 下一组预取早于 V-turn 等待，同槽覆写晚于 V 捕获未获证明（见[案例 README](README.md)）。

## 完整设计

| 设计 | 具体机制与保留理由 | 关系 |
| --- | --- | --- |
| [root](root.yaml) | Q4、S9、单 softmax WG；独立 CTA 物化 FP32 O/M/L，独立 query-head combine | 阅读入口，保存唯一 Contract |
| [q4_cluster](q4_cluster.yaml) | Q4、S16；cluster 蝶形 max/sum，zero_done，BF16 contribution reduce-add | 独立完整定义 |
| [q4_direct](q4_direct.yaml) | Q4、S1；CTA 内归一化并直接输出，保留 N1 空 parity | 独立完整定义 |
| [q1_global](q1_global.yaml) | Q1、S16；两个 softmax WG 交替 maximum mutex，分别生产 L parity | Math_reshape 整区引用 root，其余本地定义 |

## 变体与组合

### split 数与 host 条件（参数）

- 基线：[root](root.yaml)、[q1_global](q1_global.yaml) 的 `splitcount/worksplit`、`PARTIAL`、partial workspace 与 `COMBINE`；[q4_cluster](q4_cluster.yaml) 的成员数与 `clusterreceive`。
- 差异与联动：`S` 联动改变 grid、KV 条纹步长、workspace 切片和 combine 遍历；cluster 模式下还联动改变 cluster 成员数与蝶形交换轮数。global 最大支持 `S=64`（`S%16==0` 用 unroll 16，其余用 unroll 8），cluster 仅取 `2/4/8/16`。Q4 长序列 wave 查表为 `mass` `10/12/14/16/24/26/28/30/32` 依次取 `S` `14/12/10/9/6/5/5/9/9`。
- 条件与依据：host `_select_policy:78–224`，combine `:495–555`。记 `C=ceil(L/256)`、`mass=B*ceil(Q/2)`、`E=64*P`、`skew=(2*E<B*L)`：`C<=1` 选 direct；Q1 按长序列小批、非 skew 平均长度、skew 长链、`B=9..17` 中长带在 global 与 cluster 间选择；Q4 按 `L<=2048`、`2048<L<24576`、长序列 wave/skew 带选择。各代表 YAML 的 `when` 仅取上述门控的简洁真实子域。

### compact Q4 请求内 split（局部）

- 基线：[root](root.yaml) 的 `worksplit/stripe`、`PARTIAL.Decomposition`、`mpartial/lpartial/opartial` 与 `COMBINE.MAIN`。
- 差异与联动：名义 `S`、worker 数和 workspace 布局不变；每 CTA 读取 batch 的最小/最大序列长计算请求有效 `worksplit(b)`：`S=32` 取 `max(4, ceil(L_b*S/Lmax))`；`S=8` 在 `min >= 3*max/4` 时对 `B=10/11` 取 `7/6`、`B=12` 按比例缩放，其他情况下的 `B=10` 也按比例缩放，其余保持 `S`。KV 步长改为 `worksplit(b)`，`r >= worksplit(b)` 走既有中性空 split；`COMBINE.MAIN` 切到 serial 分支，仅在 `mpartial > -inf` 时读取 `opartial`。主循环 KQ/PV、奇偶修正、预分配工作与归并归属均不变。
- 条件与依据：host `:730–747`（Q4 kernel、`L >= 24576`，满足 `S=8, B=10..12` 或 `S=32, B=10..17, P > ceil(L/64)+16*(B-1)`，且 `B < 12` 或 `P < B*ceil(L/64)`）；paged `:1078–1103`；combine `:531–555`。

### Q1 的 cluster 与 direct（组合）

- 基线：[q1_global](q1_global.yaml) 的 `SOFTMAXEVEN/SOFTMAXODD`、`maximum`、`p_hi/p_lo`、`exponentialsum`，以及 [q4_cluster](q4_cluster.yaml) 的 cluster 统计/zero_done/output 或 [q4_direct](q4_direct.yaml) 的单块 normalization/output。
- 差异与联动：将 `q1_global` 的双 softmax WG 主循环（`querytile=1`、`packedcolumns=16`、512 线程、无 Q4 寄存器重分布）接入 `q4_cluster` 的蝶形归并或 `q4_direct` 的 CTA 内直接输出；交接接口仍为每列 FP32 max/sum 与双奇偶分子，`N=1` 时 `SOFTMAXODD` 仅发送空指数和奇偶通知而不执行第二次 softmax。
- 条件与依据：host `_compile_kernel:269–290` 与 `_select_policy` 实际启用这两组搭配（如 cluster 对应 `B=1, L=17845, S=16`，direct 对应 `B=1, L=64, S=1`）；paged `:621–681,1568–1579,1726–1790`。

### 首次调用与 native launcher（局部）

- 基线：[root](root.yaml)、[q1_global](q1_global.yaml)、[q4_cluster](q4_cluster.yaml)、[q4_direct](q4_direct.yaml) 的提交、out 和 workspace；global 同时包含 `PARTIAL/COMBINE`。
- 差异与联动：cache miss 时先正常执行一次 fmha；若允许建立 native launcher，则克隆当前 `out`，通过 TVM FFI 再执行同一 fmha 并设备同步，在 FP32 下计算差范数 `delta` 与首次输出范数 `denominator`，要求分段误差（`denominator > 1e-6` 时取 `delta / denominator`，否则取 `delta`）`<= 1e-2`。未通过或抛异常时拷回原 `out`；通过则缓存 weakref、签名、线程与 keepalive。仅后续命中对应 YAML 描述的单次稳态 FFI 调用。
- 条件与依据：host `:474–583,652–843`（分段探测见 `:556–562`）。stream capture 下不建立 native launcher，沿用原 kernel 路径运行；`1e-2` 仅为 launcher 探测阈值，不是 Contract 容差。
