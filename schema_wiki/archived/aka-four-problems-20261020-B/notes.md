# 四个 SOTA 算子的 dispatch schema

## 版本与交付范围

固定来源：[01fff90d714abc6e84904ea9c44523ba66d34a0e](https://github.com/bcacdwk/KernelSchema/commit/01fff90d714abc6e84904ea9c44523ba66d34a0e)，撰写开始时复核的仓库 HEAD；提交名 `261021_0000_doc6`，提交时间 `2026-10-20T16:01:59Z`。本目录可以整体放入 `schema_wiki/aka-four-problems-20261020/`。

本包包含 **35 份完整、独立的 YAML plan 和本说明文件**。每份均包含 `meta_data`、`contract.algorithm_requirement`、`contract.hardware_specification`、`design.math_reshape`、`design.work_schedule`、`design.tile_pipeline` 和 `realization`，无 YAML 注释、anchor、include 或共享 contract 外部依赖。

| 算子 | MANIFEST 选定版本 | 独立 plan | 当前计分 shape 触发的 plan | 计分 shape |
| --- | --- | ---: | ---: | ---: |
| FA4 Prefill | v37，package label v43 | 11 | 6 | 30 |
| FA4 Decode | v55 | 21 | 10 | 30 |
| QSA Prefill | BF16 FIT v3 / fc30f74 | 1 | 1 | 4 |
| GDN Prefill | Expert BF16/M64 with tail-padding fix | 2 | 2 | 30 |
| 合计 | | **35** | **19** | **94** |

版本选择依据 [MANIFEST.json](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/MANIFEST.json)。以下 shape ID 均指相应算子的 `shapes.json` 键，保留原来的 0 起始编号。表中的“—”表示当前没有计分 shape 命中；这 16 条路径均在固定任务 ABI 的默认分派中可达，不代表新增 GPU 验收。

### metadata 与独立迭代

`meta_data.dispatch.id` 是稳定的特化标识，格式为 `<operator>.<path>`；`meta_data.dispatch.when` 自含所需符号定义与选择条件；`meta_data.version` 是该 ID 对应 plan 的独立版本。文件名为 `<operator>__<path>.yaml`。本次独立复审后，11 份 FA4 Prefill、21 份 FA4 Decode 和 1 份 QSA 升为 `version: 2`；两份 GDN 经复核无需修改，保留 `version: 1`。35 个 dispatch ID 与选择条件均不变。版本升级只改变对应文件的 `version`，不以源码版本号替代 schema 版本，也不靠文件名中的数值选择路径。

本次采用已确认的 `dispatch` metadata 扩展；其余分区遵循固定提交的 [README](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/README.md) 和 [schema_docs/1–6](https://github.com/bcacdwk/KernelSchema/tree/01fff90d714abc6e84904ea9c44523ba66d34a0e/schema_docs)。本说明的简写用于浏览，单份 YAML 已具备独立理解该路径所需的完整 contract、分派条件和设计。

### 覆盖边界

FA4 两个算子覆盖本题 FP8 E4M3FN、16:1 heads、D=256、page=64、score scale=0.0625、output scale=1 的默认分派，允许所写 contract 内的 batch 和正序列长度变化；不扩展到任意 dtype、scale 或环境变量实验配置。GDN 保留强制 M64、HQ=4/HV=32、D=128、存在初始/最终 state 的入口。QSA 保留固定 FIT v3 的 shape、stride、selection 数据域和 eager 调用方式。

硬件身份统一为用户指定的 NVIDIA L20D / SM103。Prefill 的 shape 映射使用源码运行记录中的 `SMCount=148`，但 YAML 保留真正的 runtime SMCount 公式；Decode 的 148 是 selector 源码硬编码的调度预算。两者都没有把该数字写成未经核定的硬件规格。依据：[GDN L20D 运行记录](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/gdn_prefill/aka_solution/README.md#L96-L108)、[Decode selector](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution/qoder_greenfield_decode_attention/operator.py)。

## 1. FA4 Prefill：11 条路径

每个 attention worker 是两个 CTA 的 cluster，M=256 packed query rows、Bk=128、D=256。区别来自 split-KV 与 combine、矩形/flattened 工作空间、static/persistent 领取、请求顺序，以及 joint/dual loader 和 shared-rescale 消息。`J` 表示 joint K/V producer 共享 token 环；`D` 表示 K、V producer 与 token 环独立。两者的 K/V 数据缓冲始终物理分开。

| 文件 | 设计区别 | 计分 shape ID |
| --- | --- | --- |
| [fa4_prefill__splitkv_joint_load.yaml](fa4_prefill__splitkv_joint_load.yaml) | 矩形 static split-KV，J；第二 kernel 做 FP32 partial/LSE combine | 0 |
| [fa4_prefill__single_persistent_joint_load.yaml](fa4_prefill__single_persistent_joint_load.yaml) | 单请求 flattened persistent，折叠 query 顺序，J | 1, 5, 8, 9, 11, 17, 22, 26 |
| [fa4_prefill__single_persistent_dual_load.yaml](fa4_prefill__single_persistent_dual_load.yaml) | 单请求 flattened persistent，折叠 query 顺序，D | 16, 24 |
| [fa4_prefill__small_persistent_lpt.yaml](fa4_prefill__small_persistent_lpt.yaml) | 2–4 请求按 KV 长度排序，flattened persistent，D | 10, 14, 15, 18, 19, 27, 29 |
| [fa4_prefill__small_static_lpt.yaml](fa4_prefill__small_static_lpt.yaml) | 2–4 请求按 KV 长度排序，flattened static，一 cluster 一 tile，D | 2, 4, 6, 7, 13, 21, 23, 28 |
| [fa4_prefill__packed_persistent.yaml](fa4_prefill__packed_persistent.yaml) | 5–16 请求按原请求顺序，flattened persistent，D | 3, 12, 20, 25 |
| [fa4_prefill__splitkv_dual_load.yaml](fa4_prefill__splitkv_dual_load.yaml) | 矩形 static split-KV，D；第二 kernel 做 combine | — |
| [fa4_prefill__rect_joint_load.yaml](fa4_prefill__rect_joint_load.yaml) | 矩形 static query/request grid，J，attention 内完成最终输出 | — |
| [fa4_prefill__rect_dual_load.yaml](fa4_prefill__rect_dual_load.yaml) | 矩形 static query/request grid，D，attention 内完成最终输出 | — |
| [fa4_prefill__small_persistent_shared_rescale.yaml](fa4_prefill__small_persistent_shared_rescale.yaml) | 四请求且 TotalQ=16384，LPT persistent；Softmax 发送共享 alpha | — |
| [fa4_prefill__small_static_shared_rescale.yaml](fa4_prefill__small_static_shared_rescale.yaml) | 同一 shared-alpha 消息，flattened static | — |

### 精确选择条件


Let `B=BatchCount`, `T=TotalQ`, `Q=max_q_len`, `K=64*ceil(max_kv_len/64)`, and `SM=runtime multiprocessor count`. The rounded K is essential: the adapter passes `max_pages*64`, not the unrounded maximum, into the library.

```text
C = ceil(T/16)*B
G = max(1,floor(SM/2))
Split = 2*C<=G and ceil(K/128)>=16 and 2*T*16*256*4<=24000000
SinglePersistent = B=1 and ceil(Q/8)>SM+floor(SM/2)
Dual = B>1 or K>=98304
SharedRescale = B=4 and T=16384

nc = ceil((T+15*(B-1))/16)
g = max(1,min(floor(SM/2),nc))
w = T*T/(4096*B)
qlo = min(max(1,T-(B-1)*Q),Q)
A = w+T/256
F = max(0,min(qlo*(K-63-qlo),Q*(K-63-Q)))/2048
PersistSmall = ceil(nc/g)>=4 and
  (ceil(K/128)*g/w<=0.30 or (B*K-T)*8<=K or
   (K<=8*Q and ceil((K+15)/128)*g/(A+F)<=0.60))
```

| Suffix | Predicate |
| --- | --- |
| `splitkv_joint_load` | `Split and not Dual` |
| `single_persistent_joint_load` | `not Split and SinglePersistent and not Dual` |
| `single_persistent_dual_load` | `not Split and SinglePersistent and Dual` |
| `small_persistent_lpt` | `not Split and 2<=B<=4 and PersistSmall and not SharedRescale` |
| `small_static_lpt` | `not Split and 2<=B<=4 and not PersistSmall and not SharedRescale` |
| `packed_persistent` | `not Split and 5<=B<=16` |
| `splitkv_dual_load` | `Split and Dual` |
| `rect_joint_load` | `not Split and B=1 and not SinglePersistent and not Dual` |
| `rect_dual_load` | `not Split and (B>16 or (B=1 and not SinglePersistent and Dual))` |
| `small_persistent_shared_rescale` | `not Split and SharedRescale and PersistSmall` |
| `small_static_shared_rescale` | `not Split and SharedRescale and not PersistSmall` |

Under the fixed contract these predicates are mutually exclusive and exhaustive. The small-batch persistence expression is an algebraic restatement of the source, including its tightened dominance bound. At 148 SMs, single-request persistence starts at `Q >= 1777`; the nearby source comment's q1792 is an example, not the exact boundary.


### 设计边界与未计分路径

split 数保留 runtime 公式，不按每个结果值另建文件；计分 shape 0 的实际 split 数是 5。同一策略内 split 数变化不改变分区、producer 分工、partial 格式或 combine 算法。J/D、persistent、LPT 和 shared alpha 会改变这些设计关系，因而独立拆分。B=2/3/4 的排序网络实现、较大 batch 的 prefix 查找方式及实际页地址仍留在同一设计内。源码 `flatten_varlen_lpt` 属性没有控制实际排序逻辑；小 batch 排序由其映射代码执行，相关 selector 对 shared-rescale 的影响已单独体现。

五条未计分路径在 148 SM 下的可达例子如下；这里只证明路由可达：

| 文件后缀 | Query lengths | KV lengths |
| --- | --- | --- |
| `splitkv_dual_load` | `[200]` | `[131072]` |
| `rect_joint_load` | `[512]` | `[1024]` |
| `rect_dual_load` | `[1500]` | `[100000]` |
| `small_persistent_shared_rescale` | `[4096,4096,4096,4096]` | `[4096,4096,4096,4096]` |
| `small_static_shared_rescale` | `[4096,4096,4096,4096]` | `[131072,131072,131072,131072]` |

schema 保留两段 QK/PV contraction、lagged-max 与 exponent offset、FP32 denominator/FP8 P 的精度区别、QK0/QK1/PV0 的 issue 顺序、S/P/Stats 别名、persistent 跨 tile 的 Y 复用与输出 drain。首块 Stats 只 wait/release 事件，不读取 payload；后续 Stats 的消费通过 correction→PV→后续 QK 的传递依赖保护复用。没有添加源码不存在的等待。

split combine 的 grid 为 `(ceil(TotalQ*Hq/8), ceil(D/128), B)`，按请求拒绝超出本请求行数的 CTA。combine 使用 head-major 行号；主 attention 使用 head-fast packed 行。每个 combine CTA 负责八行、128 channels，用四槽 cp.async 环、三槽预取领先流过 partials。空 split 只写 `PartialLSE=-inf`，不初始化 PartialO；combine 排除其权重运算。来源：[interface](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_prefill/aka_solution/flash_attn/cute/interface.py)、[two-CTA kernel](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_prefill/aka_solution/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py)、[scheduler](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_prefill/aka_solution/flash_attn/cute/tile_scheduler.py)、[combine](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_prefill/aka_solution/flash_attn/cute/flash_fwd_combine.py)。

## 2. FA4 Decode：21 条路径

Q1 包含 8 条、Q4 包含 13 条。Q1 每 tile 一个 tail token、16 packed columns、16 warps、两组交替 softmax；Q4 每 tile 两个 tail token、32 packed columns、12 warps、一组 softmax，并采用独立的 issuer/softmax/correction register 配额。数值上均保留 leading/residual 两份 FP8 P、FP32 denominator 和双 parity 的 Y/分母递推。

| 输出模式 | 设计区别 |
| --- | --- |
| direct | 单 CTA 完成局部归一化与最终 BF16 输出 |
| cluster | 多 CTA 通过 cluster max/sum 归约对齐；每个 split 先舍入 BF16，再 reduce-add 到已清零的 BF16 输出 |
| ordinary global | 物化 FP32 split partials，第二 kernel 并行归约统计量和 numerator，最后舍入 BF16 |
| compact global | 保持 nominal launch/workspace；改变 active split 或 combine 协作，combine 由各线程串行重算统计量 |

下表 ID 对应完整文件名 `fa4_decode__<ID>.yaml`，metadata ID 为 `fa4_decode.<ID>`。


The following symbols make the mapping table compact. Every YAML repeats only the definitions needed by its own `dispatch.when`; selecting a file does not require reading this note.

```text
Let B=BatchCount, Q=q_len_per_req, L=max_seq_len, P=PageCount,
C=ceil(L/256), E=64*P, A=floor(E/B), and skew=(2*E < B*L).
U=(B>=5 and L>=32768 and A>=(65536 if B=8 else 32768) and not skew).
G=(B<=4 and L>=24576) or (U and B>=8)
  or (skew and 5<=B<=8 and L>=65536)
  or (not skew and L>=24576 and 16<=B<=17).
X16=(B<=4) or (skew and L>=32768)
  or (not skew and 5<=B<=7 and
      (12288<=L<=20480 or (L>=24576 and A>=24576))).
X8=(B<=8) or skew or (L>=24576 and 9<=B<=15).
M=2*B; near=(4*min(131072,min_b seq_lens[b]) >= 3*L).
compact32=(10<=B<=17 and P>ceil(L/64)+16*(B-1)
           and (B<12 or P<B*ceil(L/64))).
pow2_floor(n) = largest power of two <= max(1,n).
```

The conditions below include Q=1 for `q1_*` and Q=4 for `q4_*`.

| 文件 / dispatch 后缀 | 选择条件 | Split / active-work 规则 | 计分 shape ID |
| --- | --- | --- | --- |
| [fa4_decode__q1_direct.yaml](fa4_decode__q1_direct.yaml) | `C=1` | `S=1; WorkSplit(b)=SplitCount` | — |
| [fa4_decode__q1_cluster_nominal4.yaml](fa4_decode__q1_cluster_nominal4.yaml) | `C>1 and not G and not X16 and not X8` | `S=pow2_floor(min(4,C)); WorkSplit(b)=SplitCount` | — |
| [fa4_decode__q1_cluster_nominal8.yaml](fa4_decode__q1_cluster_nominal8.yaml) | `C>1 and not G and not X16 and X8` | `S=pow2_floor(min(8,C)); WorkSplit(b)=SplitCount` | 10, 13, 15, 16 |
| [fa4_decode__q1_cluster_nominal16.yaml](fa4_decode__q1_cluster_nominal16.yaml) | `C>1 and not G and X16` | `S=pow2_floor(min(16,C)); WorkSplit(b)=SplitCount` | 2, 8, 11, 19 |
| [fa4_decode__q1_global_small_batch.yaml](fa4_decode__q1_global_small_batch.yaml) | `B<=4 and L>=24576` | `S=min(C,max(16,pow2_floor(floor(64/B)))); WorkSplit(b)=SplitCount` | 18, 20, 22, 23, 25, 26, 28, 29 |
| [fa4_decode__q1_global_occupancy.yaml](fa4_decode__q1_global_occupancy.yaml) | `(U and B>=8) or (not U and not skew and L>=24576 and 16<=B<=17 and 5*B*L<9*E)` | `S=min(C,max(8,pow2_floor(floor(148/B)))); WorkSplit(b)=SplitCount` | 6 |
| [fa4_decode__q1_global_long_dominated16.yaml](fa4_decode__q1_global_long_dominated16.yaml) | `not U and not skew and L>=24576 and 16<=B<=17 and 5*B*L>=9*E` | `S=16; WorkSplit(b)=SplitCount` | 1, 12 |
| [fa4_decode__q1_global_skew32.yaml](fa4_decode__q1_global_skew32.yaml) | `skew and 5<=B<=8 and L>=65536` | `S=32; WorkSplit(b)=SplitCount` | — |
| [fa4_decode__q4_direct.yaml](fa4_decode__q4_direct.yaml) | `C=1` | `S=1; WorkSplit(b)=SplitCount` | — |
| [fa4_decode__q4_cluster_short.yaml](fa4_decode__q4_cluster_short.yaml) | `C>1 and L<=2048` | `S=pow2_floor(min(C,min(8,max(2,pow2_floor(ceil(96/(2*BatchCount))))))); WorkSplit(b)=SplitCount` | — |
| [fa4_decode__q4_cluster_mid_small_batch.yaml](fa4_decode__q4_cluster_mid_small_batch.yaml) | `2048<L<24576 and B<=4` | `S=pow2_floor(min(C,min(16,max(4,pow2_floor(ceil(96/(2*BatchCount))))))); WorkSplit(b)=SplitCount` | 0, 9 |
| [fa4_decode__q4_cluster_mid4.yaml](fa4_decode__q4_cluster_mid4.yaml) | `2048<L<24576 and B>=10 and (B<17 or skew)` | `S=4; WorkSplit(b)=SplitCount` | — |
| [fa4_decode__q4_global_mid.yaml](fa4_decode__q4_global_mid.yaml) | `2048<L<24576 and ((5<=B<=9) or (B>=17 and not skew))` | `S=min(C,max(4,pow2_floor(floor(148/(2*BatchCount))))); WorkSplit(b)=SplitCount` | — |
| [fa4_decode__q4_global_one_wave.yaml](fa4_decode__q4_global_one_wave.yaml) | `L>=24576 and not skew and M in {10,12,14,16,24,26,28}` | `S=floor(148/(2*BatchCount)); WorkSplit(b)=SplitCount` | 5 |
| [fa4_decode__q4_global_two_waves.yaml](fa4_decode__q4_global_two_waves.yaml) | `L>=24576 and ((not skew and M in {30,32}) or (skew and M in {16,18}))` | `S=floor(296/(2*BatchCount)); WorkSplit(b)=SplitCount` | 3, 7, 14 |
| [fa4_decode__q4_global_balanced.yaml](fa4_decode__q4_global_balanced.yaml) | `L>=24576 and not skew and M not in {10,12,14,16,24,26,28,30,32} and B not in {10,11}` | `S=min(C,max(8,pow2_floor(floor(148/(2*BatchCount))))); WorkSplit(b)=SplitCount` | 17, 21, 24, 27 |
| [fa4_decode__q4_global_skew32.yaml](fa4_decode__q4_global_skew32.yaml) | `L>=24576 and skew and M not in {16,18} and not compact32` | `S=min(C,max(32,pow2_floor(floor(148/(2*B))))); WorkSplit(b)=SplitCount` | — |
| [fa4_decode__q4_compact8_one_wave.yaml](fa4_decode__q4_compact8_one_wave.yaml) | `L>=24576 and not skew and B in {10,11} and near` | `S=8; WorkSplit(b)=floor(148/(2*BatchCount))` | 4 |
| [fa4_decode__q4_compact8_proportional.yaml](fa4_decode__q4_compact8_proportional.yaml) | `L>=24576 and not skew and B=10 and not near` | `S=8; WorkSplit(b)=ceil(seq_lens[b]*SplitCount/max_seq_len)` | — |
| [fa4_decode__q4_compact8_serial.yaml](fa4_decode__q4_compact8_serial.yaml) | `L>=24576 and not skew and B=11 and not near` | `S=8; WorkSplit(b)=SplitCount` | — |
| [fa4_decode__q4_compact32_proportional.yaml](fa4_decode__q4_compact32_proportional.yaml) | `L>=24576 and skew and compact32` | `S=32; WorkSplit(b)=max(4,ceil(seq_lens[b]*SplitCount/max_seq_len))` | — |

### 拆分依据与边界

Direct, cluster and global-combine policies have different materialization and output arithmetic: direct normalizes once inside the CTA; cluster exchanges maxima/sums and atomically adds individually BF16-rounded contributions; global mode saves FP32 split results and casts only after combining them. Q1 packs 16 columns, uses 16 warps and alternating softmax warpgroups; Q4 packs two tail tokens into 32 columns, uses 12 warps, one softmax warpgroup and the 64/168/256 issuer/softmax/correction register allocation.

Nominal 4/8/16 cluster policies are intentional workload choices, while the available-key-tile cap remains a formula inside each policy. Thus shape 11's nominal-16 policy capped to 8 is not reassigned to `q1_cluster_nominal8`; the analogous Q4 cap on shape 9 stays in `q4_cluster_mid_small_batch`. Q1 skew32 is fixed at 32 on its B=5…8 domain. Q4 skew32 retains the source occupancy base: `S=min(C,max(32,pow2_floor(floor(148/(2*B)))))`. For example B=1, Q=4, L=24576 and PageCount=128 selects 64 splits when logical pages reuse physical pages. The stable dispatch ID is retained; `skew32` does not override this formula. Compact32 remains nominally 32 on its separate B=10…17 domain.

The long-Q4 table is represented by two exact formulas on its original lookup domains: the one-wave entries equal floor(148/M), and the two-wave entries equal floor(296/M). Those formulas do not extend selection to other M. The constants describe source scheduling budgets; they do not claim verified hardware SM counts.

Compact policies preserve the nominal launch/workspace layout while changing active stripes and the combine design. The near-uniform 8-slot policy uses 7 working splits at B=10 and 6 at B=11; proportional policies use ceil(L_b*S/L), with a minimum of 4 only for the 32-slot policy. `q4_compact8_serial` keeps 8 working splits but still switches to serial per-thread statistic reduction, so it remains distinct. The source's B=12/S=8 compact subcase is not reached by the default selector and therefore has no independent file. The near-uniform gate intentionally uses `min(131072,min(seq_lens))`, including its finite initializer, rather than replacing it with the unrestricted minimum.


### 短循环、空 split 与尾部

所有路径都包含 N=0、N=1 与一般尾块的合法行为。空 split 写 M=-inf、L=0，不写 Opartial。ordinary combine 仍读取该 Opartial 并乘零，因此保留本题 scratch 初次使用前清零、后续 payload 有限的前提；compact combine 对空 split 跳过该读取。没有增加每次调用的 workspace 清零。N=1 仍初始化并通知空 parity，使共同 epilogue 可排空两个槽。

K/Score issue lead 的尾部由显式无 payload 的 Ktail/ScoreTail turn 表达，不靠越界删边虚构生产者。Q4 的 FP32 输出暂存占两个 KV slice，已按字节范围登记 alias；PDL 的提前 launch permission 与读取 partial 前的 predecessor-completion wait 也分别表达。来源：[paged kernel](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution/qoder_greenfield_decode_attention/gqa_decode_paged.py)、[reduction implementation](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution/qoder_greenfield_decode_attention/gqa_decode.py)。

### 十一条未计分路径的可达例子

下面均保持对应 Q1/Q4 固定 ABI；`[x]×n` 表示 n 个相同长度，`+` 表示拼接。用于确认默认 selector 可达，不是 GPU 测试记录。

| 文件后缀 | KV lengths |
| --- | --- |
| `q1_direct` | `[128]` |
| `q1_cluster_nominal4` | `[4096]×9` |
| `q1_global_skew32` | `[65536] + [1024]×4` |
| `q4_direct` | `[128]` |
| `q4_cluster_short` | `[1024]` |
| `q4_cluster_mid4` | `[4096]×10` |
| `q4_global_mid` | `[4096]×5` |
| `q4_global_skew32` | `[32768,512,512]` |
| `q4_compact8_proportional` | `[32768] + [20000]×9` |
| `q4_compact8_serial` | `[32768] + [20000]×10` |
| `q4_compact32_proportional` | `[32768] + [2048]×9` |

## 3. GDN Prefill：2 条路径

入口强制 M64；两条路径共用 64-token chunk、8→16→32→64 分块逆、七类 GEMM 和 `(batch, value_head, 64-value slice)` 工作划分。每份 worktile 顺序处理 `2*ceil(L/128)` 个配对 chunk，保留 FP32 state、BF16 snapshot、残差两次 BF16 舍入，以及从同一 FP32 U 分别生成 Ubar/Udec。

| 文件 | 选择条件与设计区别 | 计分 shape ID |
| --- | --- | --- |
| [gdn_prefill__m64_single_sequence.yaml](gdn_prefill__m64_single_sequence.yaml) | B=1；固定 descriptor；raw scalar 双槽、五槽联合 ready、CG1 单一 phase 事件流 | 3, 17, 19, 20, 21, 22, 23 |
| [gdn_prefill__m64_packed_sequences.yaml](gdn_prefill__m64_packed_sequences.yaml) | 2≤B≤16；每序列更新 descriptor；Alpha/Beta 独立生产、独立 consumer-return 与 CG1 input streams | 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 24, 25, 26, 27, 28, 29 |

单序列 raw scalar 环使用本 worktile 的 `c%2`，处理后 scalar 使用 `c%5`，没有误写为跨 worktile 连续序号；B=1 在记录的 148 SM 环境下为 64 个 CTA，各一份工作。两份都保留计时内 PyTorch int64→int32 转换：packed 读取 Cu32 并登记 handoff，B=1 的核心直接使用 `[0,T)`，因此不虚构消费 Cu32 的关系。来源：[M64 dispatch](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/gdn_prefill/aka_solution/gdn_prefill.py#L142-L167)、[chunked kernel](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/gdn_prefill/aka_solution/gated_delta_net_chunked.py)。

`T%128==0` 仅消除 paired-tail 中性填充，不新拆 schema；`T%64==0` 不足以消除最后一个配对 padding chunk。零初始 state 与 continuation 都使用显式 initial_state，不改变本设计。M128、无 initial_state、checkpoint、state-pool、`B=2 && HQ==HV` 特例均不在本题入口/ABI 内。对应修复依据：[tail-padding fix](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/gdn_prefill/aka_solution/README.md#L29-L40)。

## 4. QSA Prefill：1 条路径

| 文件 | 适用路径 | 计分 shape ID |
| --- | --- | --- |
| [qsa_prefill__union_persistent.yaml](qsa_prefill__union_persistent.yaml) | 固定 single-request FIT v3，eager，十行共享 union 后 persistent attention | 0, 1, 2, 3 |

四个计分 shape 只改变 CP rank 和 seed。wrapper 忽略 `use_prefill_config`；传入 out 与 callable 内分配 out 也共用同一流水。实际 logical_indices 为 `[4096,2052]`，包含输入构造附加的 valid_count。每行有 512 个不同四-token 组及至多 3 个 causal tail token，logical pages `[0,1024)` 都映射到已初始化的有限 K/V，故每行至少 2048 个有效 key、任意 rowblock union Count≥512，attention 至少 32 次 KV 迭代。没有把 empty union、无效页或重复 token 的 multiset 泛化进该 contract。依据：[input.py](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/qsa_prefill/input.py#L286-L447)。

410 个 512-thread preprocessing CTA 构造四-token union 和 40-bit membership；820 个 attention worktile 按 rowblock 降序/KV head 升序，由 persistent CTA 动态领取。M=128 容纳至多 120 个真实 query/head 行，Bk=64；两个 softmax warpgroup 处理 KV 列半区。保留三 QK 领先、K/V 双槽、S/P 三槽 alias、conditional correction、BF16 attention 后再 gate 舍入，以及跨 eager 调用 ticket/tbase 的生命周期。来源：[operator](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/qsa_prefill/aka_solution/qwen38_core_qsa_cutedsl/operator.py)、[preprocess](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/qsa_prefill/aka_solution/qwen38_core_qsa_cutedsl/qsa_preprocess.py)、[attention](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/qsa_prefill/aka_solution/qwen38_core_qsa_cutedsl/qsa_kernel.py)。

union 长度、末尾不足十行/不足十六个 union entry 仍属于同一设计；其他 group/tile/stage 与 mixed-group fallback 不在固定任务域。源码只记录 eager 验收；CUDA Graph replay 对 host tbase/device ticket 的适配不在本 plan 中。两项源码局部证明缺口保留为下节明确的 pending。

## 5. 占位、未确认事实与运行前提

以下是本包全部非硬件类 pending 的分类，以及共用硬件占位。pending 没有代替本次必须写出的宏观工作分配与主流水。

| 范围 / schema 位置 | 占位或未核定事实 |
| --- | --- |
| 全部 `contract.hardware_specification.execution/feature/wiki` | 按要求以 `<hardware-...@pending:...>`、`<wiki-...@pending:...>` 保留；四个算子的 device/arch/execution/feature 一致，不凭别的 GPU 案例填容量和吞吐。 |
| 全部 `hardware_specification.stack` | 只采用各自 requirements：FA4 为 `torch>=2.9; cuda-python; nvidia-cutlass-dsl==4.6.1`；QSA 为 `torch>=2.9; cuda-python>=13.0; nvidia-cutlass-dsl>=4.6.1`；GDN 为 `torch>=2.9; cuda-python>=13.0; nvidia-cutlass-dsl[cu13]>=4.5.0`。未锁定的实际 CUDA/compiler/package 版本继续 pending。 |
| FA4 Prefill `math_reshape.*.arithmetic.approx` | `<numeric-bound@pending:SM103/fast-exp2>`；split 路径还含 `fast-log`。源码没有独立原语误差界。 |
| FA4 Decode `math_reshape.*.arithmetic.approx` | `<error-bound@pending:fa4_decode/fast-exp2>` 与 `reciprocal`。保留 benchmark 的 atol/rtol，不把它当成原语误差证明。 |
| FA4 Decode `tile_pipeline.*.lane.LoadK.info.verb.configure` | CuTe 4.6.1 的 `get_max_tmem_alloc_cols(sm_100)` 与 `get_smem_capacity_in_bytes(sm_100)` 返回值没有随仓库提供。stage 数、Lead、slot 映射保留其源码公式；不借 SM100 示例猜测 SM103 容量。 |
| FA4 Decode `tile_pipeline.*.lane.LoadK.info.verb.advance` | `<sync-proof@pending:fa4_decode/PageIndex-reuse>`：四槽 PageIndex 环中，尚未从源码证明 V 对第 i 份索引的寄存器捕获完成先于第 i+4 份异步覆盖可见；保留实际预取/issue-turn 顺序，不添加虚构 reuse edge。 |
| GDN `math_reshape.rewrite[inverse].arithmetic.approx` | `<error-bound@pending:gdn_prefill/log2-exp2>`。 |
| GDN `work_schedule.phase.index_cast` 与 `tile_pipeline.phase.index_cast` | `<cast-tile/grid/grid-stride/lane/lowering@pending:PyTorch/to-int32>`：PyTorch cast 内部 grid、tile、warp 与 lowering 不在选定仓库内。转换语义、计时边界和真实 handoff 已写清。 |
| QSA `math_reshape.*.arithmetic.approx` | `<numerical-bound@pending:qsa/fast-exp2>` 与 `reciprocal`。 |
| QSA `math_reshape.rewrite[union].math` | `<padding-proof@pending:qsa/unwritten-Q-rows>`：Q gather 只写真实 packed 行，而 warp correction vote 包含未写的 padding 行；未从源码独立证明对有效行 correction 时机的影响。 |
| QSA `tile_pipeline.phase.attention.lane.SoftmaxA.info.verb.publish` | `<sync-proof@pending:qsa/HalfMax-reuse>`：两个半区各写 HalfMax 后有一次 256-thread rendezvous，未见显式“对方已读完”回执；复用安全性的完整证明仍待补。 |

Decode 的新增 PageIndex pending 也只是尚未闭合的源码次序证明，不是已证实的竞态：next PT prefetch 在等待 V 的 issue turn 之前发出，现有 turn 本身没有表达 V 已读完该槽。定位：[PT prefetch 与 K turn](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution/qoder_greenfield_decode_attention/gqa_decode_paged.py#L1228-L1250)、[V 捕获](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution/qoder_greenfield_decode_attention/gqa_decode_paged.py#L1341-L1345)。

QSA 最后两项是如实保留的源码证明缺口，不据此判定 kernel 错误，也没有给 plan 添加源码没有的 padding 清零或第二道 barrier。定位：[Q gather](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/qsa_prefill/aka_solution/qwen38_core_qsa_cutedsl/qsa_kernel.py#L854-L883)、[HalfMax/vote/correction](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/qsa_prefill/aka_solution/qwen38_core_qsa_cutedsl/qsa_kernel.py#L460-L565)。

FA4 的 `sinks=None`、`o_sf_scale=None` 是必传的非数值 ABI sentinel，直接在 contract compute 声明，不使用虚构 dtype。Doc1 一方面要求全部调用者对象进入 input，另一方面将 input.dtype 限定为数值格式；当前规则尚缺这类必传非数值 sentinel 的登记方式，因此保留此明确的格式能力缺口。QSA 的可选 `out` 则已有数值 tensor 类型，默认值使用表达式字符串 `"None"`，由 compute 说明缺省分配，不再以 pending 代替已知事实。Python 整数实参用 `python_int`，并在每份 contract 明确定义为任意精度有符号整数；Python float 实参用 fp64，进入实现的 FP32 处理另在 design 中表达，避免把 host 标量误写成 int32/FP32 张量。初次 workspace 清零、out 原地修改与返回别名、GDN 最终 state 都按对应 input/reference/metadata 表达。Decode ordinary combine 的空 payload 读取前提、QSA eager ticket 生命周期、GDN 单序列 local scalar 环的目标设备适用前提，均在相关 plan 和上文保留。

## 6. 检查结果与边界

本次在上一版交付之上又做了两轮独立审阅，每轮按 Prefill、Decode、GDN/QSA 分为三组，共六个新 agent 会话。第二轮使用新的上下文，不继承第一轮的推理和报告。每组通读 README 与 schema_docs/1–6，覆盖分配到的全部文件，并对照固定源码检查；主审逐项核对语义 diff、文档规则和源码证据后合并。同一审阅组内，共同主体可以复用核验结论，但每个变体的完整差异、分派条件和独有流水都单独检查。

本次合并的主要修订：

- **FA4 Prefill：** 明确不足一个 128-key block 时的真实页复用；修正 persistent Q 复用边的工作作用域；明确空 split 只写 LSE、combine 的零权重跳过与 reciprocal-then-multiply；补充分母 packed-pair 的缩放分组。删除 realization 中已由上游表达的重复数量，不展开不改变设计的排序网络细节。
- **FA4 Decode：** 补齐空/单块 split 的初始化、奇偶槽与短循环语义，保留 FinalMaximum 的提前发布和 TMEM 的真实归还时机；修正 `q4_global_skew32` 的完整 split 公式；区分统计 store 发射与全局可见，补齐 FinalSum rendezvous 到最终 Y 捕获的关系；记录 global/compact 路径的 caller workspace 容量与四字节整除要求。PageIndex 环的未闭合次序证明按第 5 节保留 pending。
- **QSA：** 将已知 `out=None` 默认表达式直接写出，统一逐实例标量 `Count(b)`，补齐首次 ticket 初始化、跨 eager 调用的 base 生命周期和终止票据行为；明确 union 列先求指数再做 membership mask，以及从未发出 gather 的 worker 的 drain 为 no-op。
- **GDN：** 两轮都复核了 M64 数值转换、七个矩阵乘、单序列/packed scalar 协议、paired-tail 与最终 state；没有需要修改的证据，保留原文。

Decode 的 workspace 条件来自 [入口切片与容量检查](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution/qoder_greenfield_decode_attention/operator.py#L227-L254)，包括入口仍保留但 kernel 不使用的末尾 m 区；它作为实现兼容条件记录，不新增数学中间张量。FinalSum 等待来自 [Correction 的最终归约与 Y 捕获](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution/qoder_greenfield_decode_attention/gqa_decode_paged.py#L1991-L2032) 及 [reduction 角色](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution/qoder_greenfield_decode_attention/gqa_decode.py#L137-L275)；没有为 schema 添加源码不存在的 barrier。

| 检查 | 结果 |
| --- | --- |
| 唯一键 YAML 加载、顶层顺序、已实现的闭字段规则、phase/动作引用、metadata 唯一性 | 35/35 通过；无 YAML 注释、anchor、alias、merge 或 include。检查脚本是部分静态检查器，不替代文档和源码审阅 |
| 分派与计分 shape 映射 | 所有 dispatch ID/选择条件不变；保留原有 94/94 唯一映射，共触发 19 个 dispatch，逐文件对应关系见上表 |
| 本次针对 skew32 的检查 | 从实际 YAML 取出选择条件与 SplitCount 公式，独立改变物理 PageCount，与固定源码纯 host selector 比较 1,971 个命中用例，覆盖 32 和 64 两种 split 结果 |
| 原有 Prefill 路由证据 | 10,000 组有限输入与源码纯 host 规则一致；本次未改变其策略 |
| 原有 Decode 路由证据 | 7,380 组边界/长度组合，以及另一次实际 metadata 谓词的 7,720 组有限检查；本次补充了此前未覆盖的独立物理页数变化，不把旧检查当作整个输入域的穷尽证明 |
| 流水边界审阅 | 检查 N=0/1/2、parity/tail、persistent 工作长度变化、资源全部读者与最终归还；明确区分数据发布、槽位可复用与 kernel 完成 |
| 未计分路径 | 5 个 Prefill、11 个 Decode 均有合法输入域内的路由 witness |
| GPU 验收 | 本轮未运行数值、竞态或性能测试；已有源码测量不作为本次 schema 的新测量 |

这些静态检查验证转写、结构和路由，不是 GPU 正确性或性能证明。未知 SDK 容量、Decode PageIndex 与 QSA 局部证明缺口保留为 pending；其他可由源码确定的设计已直接填写。

## 7. 固定提交的 contract 与实现来源

上文源码链接与每份 `meta_data.source` 都固定到同一提交。完整 contract 依据如下；实现只取 `aka_solution`，不使用基线 `solution/`、seed 或 `production_ncu.py` 的设计。

| 算子 | Contract / 计分依据 | 入口与依赖 |
| --- | --- | --- |
| fa4_prefill | [reference.py](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_prefill/reference.py) · [input.py](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_prefill/input.py) · [metadata.json](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_prefill/metadata.json) · [shapes.json](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_prefill/shapes.json) | [solution.py](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_prefill/aka_solution/solution.py) · [requirements.txt](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_prefill/aka_solution/requirements.txt) · [README.md](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_prefill/aka_solution/README.md) |
| fa4_decode | [reference.py](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_decode/reference.py) · [input.py](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_decode/input.py) · [metadata.json](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_decode/metadata.json) · [shapes.json](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_decode/shapes.json) | [solution.py](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution/solution.py) · [requirements.txt](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution/requirements.txt) · [README.md](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution/README.md) |
| qsa_prefill | [reference.py](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/qsa_prefill/reference.py) · [input.py](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/qsa_prefill/input.py) · [metadata.json](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/qsa_prefill/metadata.json) · [shapes.json](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/qsa_prefill/shapes.json) | [solution.py](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/qsa_prefill/aka_solution/solution.py) · [requirements.txt](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/qsa_prefill/aka_solution/requirements.txt) · [README.md](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/qsa_prefill/aka_solution/README.md) |
| gdn_prefill | [reference.py](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/gdn_prefill/reference.py) · [input.py](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/gdn_prefill/input.py) · [metadata.json](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/gdn_prefill/metadata.json) · [shapes.json](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/gdn_prefill/shapes.json) | [solution.py](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/gdn_prefill/aka_solution/solution.py) · [requirements.txt](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/gdn_prefill/aka_solution/requirements.txt) · [README.md](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/gdn_prefill/aka_solution/README.md) |
