# AKA 四算子：按设计分派的完整 plans

## 固定来源与文件身份

全部实现与契约固定到 [`bcacdwk/KernelSchema@01fff90d714abc6e84904ea9c44523ba66d34a0e`](https://github.com/bcacdwk/KernelSchema/tree/01fff90d714abc6e84904ea9c44523ba66d34a0e)。字段与写法依据同一提交的 [README](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/README.md) 和 [schema_docs/1–6](https://github.com/bcacdwk/KernelSchema/tree/01fff90d714abc6e84904ea9c44523ba66d34a0e/schema_docs)，仅采用本次讨论明确批准的扩展 `meta_data.dispatch`；本包不采用 README 的行数锚点。

每个文件都是独立完整 plan：`meta_data`、`contract.algorithm_requirement`、`contract.hardware_specification`、`design.math_reshape`、`design.work_schedule`、`design.tile_pipeline`、`realization` 均在本文件内。`dispatch` 是稳定路径标识，与文件名双下划线后的部分一致；`version` 是该路径自己的 plan 修订号，不是源码版本号，也不要求其他路径同步递增。本轮有修改的 16 份 FA4 与 1 份 QSA 递增为 `version: 2`；两份 GDN 保持 `version: 1`。后续同一路径的修订只递增自己的 version；新增路径使用新的 dispatch。源码身份始终由带完整 commit 的 `source` 记录。

本包按约定交付 **19 份 YAML：FA4 prefill 5、FA4 decode 11、QSA prefill 1、GDN prefill 2**。默认入口下，全部 **94 个计分 shape ID**（30、30、4、30）各归入恰好一份 plan。下文 ID 均指各算子 `shapes.json` 的键，不把 seed、初态或重复验证算成新 shape。源码还存在未被本次计分域触发的路径；它们在下文列为扩展候选，不声称已被这 19 份覆盖。

输入根目录为 `kernel_wiki/aka-four-problems-20261019/`。每份设计沿 `aka_solution/solution.py::Model.forward` 的真实调用链转写；契约来自同目录 `reference.py`、`input.py`、`metadata.json`、`shapes.json`，栈来自 `aka_solution/requirements.txt`。所选包版本以输入根目录的 [README](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/README.md) 和 [MANIFEST](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/MANIFEST.json) 为准。基线、seed kernel 与 schema_docs 中其他版本的示例不作为本次设计事实。

## 拆分原则与路由读法

会改变数学分解、工作归属、工作遍历、warp 职责、数据交接、缓冲复用或合并协议的特化独立成文件。FA4 decode 的 split 数虽然可以是运行时参数，但它是 host 针对 workload 明确选择的并行分解；本次也按已约定的 split 数独立维护。循环长度、尾块 mask、N=0/1 边界以及 seed 内容不另建文件，其行为在所属 plan 内闭合。

contract 保留外部算法任务与 ABI，design 明确所选路径的条件和配置。这里的映射先按源码 host policy 得到配置，再按配置选择文件；不能仅凭文件名把阈值外推到未覆盖 workload。路由以选定入口的默认环境为准。FA4 prefill 使用运行时 SM 数，核对计分域时采用仓库 L20D 环境记录的 148；FA4 decode 的策略自身固定 `_SM_COUNT=148`。改变环境覆盖、硬件可见 SM 数或入口参数后，应重新计算映射。

## FA4 prefill：5 份

| 文件 / dispatch | 计分 shape ID | 划分依据 |
| --- | --- | --- |
| `fa4_prefill__splitkv_s5.yaml` / `splitkv_s5` | 0 | 短 query 的矩形网格，KV 连续分成 5 段；每段由非持久 CTA pair 写 FP32 归一化 partial O 和自然对数 LSE，再运行独立 combine kernel。 |
| `fa4_prefill__persistent_shared_load.yaml` / `persistent_shared_load` | 1、5、8、9、11、17、22、26 | 单 request、单 split、flattened persistent；warp 9 同时加载 K/V，两者共用一个四阶段握手序列，warp 10 没有加载角色。 |
| `fa4_prefill__nonpersistent_split_load.yaml` / `nonpersistent_split_load` | 2、4、6、7、13、21、23、28 | 2–3 个 request 的 flattened 非持久路径；每对 CTA 只处理一份工作，一次发布工作坐标，warp 9/10 分别推进 K/V 独立四阶段流水。 |
| `fa4_prefill__persistent_split_load_lpt.yaml` / `persistent_split_load_lpt` | 10、14、15、16、18、19、24、27、29 | 单 split、flattened persistent、独立 K/V loader；当前计分子集为 1–2 个 request，按 KV 长度降序排列 request，再线性查找缓存的前缀表。 |
| `fa4_prefill__persistent_split_load_original_order.yaml` / `persistent_split_load_original_order` | 3、12、20、25 | 7–9 个 request 的 persistent 路径；保留 request 输入次序，查找一次建立的前缀表时采用固定四步二分，独立 K/V loader。 |

### 路由边界与未新增文件的配置

`BatchCount`、`TotalQ`、`max_q_len` 和 page-rounded KV 上界驱动 host dispatch。包装器传入的 KV 上界为 `64*ceil(max_kv_len/64)`；loader 分工的阈值按这个值判断，不能直接以未向 page 对齐的 `max_kv_len` 代替。默认独立 loader 条件是 `BatchCount>1` 或 KV 上界至少 `96*1024`，否则共用 loader。单 request 持久化门槛使用 `ceil(max_q_len*16/128) > SMCount + floor(SMCount/2)`；2–4 个 request 的持久化还受 `interface.py` 中 wave 数、dominance bound、near-pinned 和 tightened-bound 条件共同控制，并不等于一个简单的 query 长度阈值。

当前环境取 `SMCount=148`，persistent 网格最多 74 对 CTA。这个设备事实来自同一批 kernel 的 `gdn_prefill/aka_solution/README.md` 中 `agate env L20D` 的 `sm_count` 记录；prefill 源码实际查询设备 SM 数。覆盖表是在这个目标环境下对 30 个 shape 的结果，不能把它当成任意 GPU 的固定分派表。

源码中还存在未被这些 shape 触发的 `BatchCount=4 && TotalQ=16384` 的 flat-LPT/shared-rescale 配置，以及可由环境开关选择的 CLC、其他 persistent mode、非 TMA page load、exp2 emulation 等实验路径；本轮不为未命中的配置新增文件。`_hd256_fastpath` 缓存和重放已选中的 kernel 序列，不改变设备数学、分工或流水，故不单独 dispatch。tile 尾部 masking、需要 rescale 的逐行判定，以及 split 内的五个 KV 段是该文件内部算法/工作域，不拆成额外 schema。

### 需要注意的设计事实

- Shared-load 的 Kbuf 与 Vbuf 是两个独立分配的数组，但四个 stage 的生产/消费游标共用；加载与 MMA 顺序均为 `K0; Ki,V(i-1), i=1..N-1; V(N-1)`。schema 用连续 operand-stage 域明确这个耦合关系，不能仅合并两个 loader 的名字后仍保留独立推进语义。
- Persistent 文件的 score/probability、K/V 输入 ring 和统计量跨工作接续；下一份工作的首次 PV 必须等上一份工作的 Y 所有权归还。非持久文件每个 worker 只处理一份工作，删除了这些跨工作边；其 4 个 Int32 的 scheduler allocation 只有前 2 个元素作为 `WorkCoord` 视图。
- S5 按连续的 128-token KV block 切分，初始区间为 `[0,14)`、`[14,28)`、`[28,42)`、`[42,56)`、`[56,68)`，再与工作块的 causal 范围相交。shape 0 的每个有效 query tile 在每个 split 内都非空，这份 schema 按该非空域展开；源码在其他情形会写 `-inf` LSE 并跳过该 split 的计算。
- S5 partial O 在分段内已经归一化，保存为 FP32；combine 使用自然对数 LSE 计算权重，最后才保存 BF16。partial LSE 的真实存储为 `[split, head, TotalQ]`，接口转置只是视图；combine 的八行 tile 使用 head-major query 次序，和主 kernel 的 query-major GQA packing 不同。
- Combine 的 8 个 warp 共同执行加载、LSE reduction 和输出累加；没有虚构专用 loader warp。五个 split 的 LSE 填充为 32 槽，partial O 使用 4 槽 cp.async ring，先预取 3 段，在同步计算期间保留尚未完成的预取。

### 占位

硬件 execution/feature/wiki、未固定的 CUDA/driver 版本、Python None sentinel 的 schema dtype 名称，以及 fast exp2/log/division 的认证误差界保留 pending。tile、warp、精度、索引、分派和同步事实均由源码实写。

### 固定来源索引

以下路径均相对于 `kernel_wiki/aka-four-problems-20261019/`，均使用上面的固定提交：

| 事实 | 路径 / 入口 |
| --- | --- |
| 生产 ABI、HND cache 视图、page-rounded KV 上界、默认开关 | `fa4_prefill/aka_solution/solution.py::Model.forward` |
| 数值参考、误差、原地输出、输入域、计分 shape | `fa4_prefill/reference.py::Model.forward`、`fa4_prefill/input.py`、`fa4_prefill/metadata.json`、`fa4_prefill/shapes.json` |
| split heuristic、flatten/persist/loader 分派、combine 配置和调用 | `fa4_prefill/aka_solution/flash_attn/cute/interface.py::_hd256_split_heuristic`、`_flash_attn_fwd`、`_flash_attn_fwd_combine` |
| paired tile、warp 分工、RequestMap、共享/独立 K/V stage、score/P alias、online softmax、LSE/partial O | `fa4_prefill/aka_solution/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py` |
| static/persistent grid、分配跨度和退出 | `fa4_prefill/aka_solution/flash_attn/cute/tile_scheduler.py::Sm100FmhaStaticTileScheduler` |
| 8×128 combine tile、head-major mapping、LSE/weight alias、4-stage partial ring | `fa4_prefill/aka_solution/flash_attn/cute/flash_fwd_combine.py::FlashAttentionForwardCombine` |
| CuTe DSL / Torch / cuda-python 栈 | `fa4_prefill/aka_solution/requirements.txt` |
| L20D 148 SM 的环境实测记录 | `gdn_prefill/aka_solution/README.md`，`num_sm = 148` 段 |

## FA4 decode：11 份

### 文件与计分输入映射

下表中的文件名均加前缀 `fa4_decode__`、后缀 `.yaml`。`Q` 是 `q_len_per_req`，`S` 是 launch split 数，`W` 是 KV 访问步长和有效 split 数；普通路径 `W=S`。

| dispatch | Q | 合并方式 | S / W | shape IDs | 本计分域的局部迭代数 N |
|---|---:|---|---|---|---|
| `q1_kernel_s8` | 1 | 独立 FP32 partial combine | 8 / 8 | 6 | 55、56、57、58、60、61 |
| `q1_kernel_s16` | 1 | 独立 FP32 partial combine | 16 / 16 | 1、12 | 1–8 |
| `q1_kernel_s64` | 1 | 独立 FP32 partial combine | 64 / 64 | 18、20、22、23、25、26、28、29 | 3、4 |
| `q1_atomic_s8` | 1 | 8 CTA cluster 内统计合并，BF16 输出 reduce-add | 8 / 8 | 10、11、13、15、16 | 0–16 |
| `q1_atomic_s16` | 1 | 16 CTA cluster 内统计合并，BF16 输出 reduce-add | 16 / 16 | 2、8、19 | 4、5、27、28、29、30、31 |
| `q4_kernel_s9` | 4 | 独立 FP32 partial combine | 9 / 9 | 3、7、14 | 2–14 |
| `q4_kernel_s14` | 4 | 独立 FP32 partial combine | 14 / 14 | 5 | 31–36 |
| `q4_kernel_s64` | 4 | 独立 FP32 partial combine | 64 / 64 | 17、21、24、27 | 3、4 |
| `q4_compact_s8_active6` | 4 | compact 标量统计合并、仅读取活跃 numerator | 8 / 6 | 4 | 活跃流为 73、74、75、76、77、80、81、82；另有两个 N=0 槽 |
| `q4_atomic_s8` | 4 | 8 CTA cluster 内统计合并，BF16 输出 reduce-add | 8 / 8 | 9 | 0、1、2 |
| `q4_atomic_s16` | 4 | 16 CTA cluster 内统计合并，BF16 输出 reduce-add | 16 / 16 | 0 | 4、5 |

这里有五种宏观骨架：Q1/kernel、Q1/atomic、Q4/kernel、Q4/compact、Q4/atomic；按已同意的特化粒度，进一步独立维护 11 份实际选中的配置。kernel 路径的 S 是 runtime 参数，不在编译 cache key 中，但被保留为独立调度策略，分别决定 stream 数、局部链长度、partial 工作量和 combine 范围；atomic 的 S=8/16 还改变 resident cluster 成员数及 3/4 轮 XOR 交换。combine 根据 `S%16` 选择循环展开因子，只是实现展开，没有再分文件。

### 路由条件与文件边界

令 `B=BatchCount`、`L=max_seq_len`、`P=PageCount`、`E=64*P`。输入构造保证 `P=sum_b ceil(seq_lens[b]/64)`。下面说明这些实际输入为何落到对应配置；精确覆盖仍以表中的 shape IDs 为准。

- `q1_kernel_s8`：shape 6 的 B=11、L=124815；非 skew 的超长上下文 batch 路径用 `max(8,floor_po2(148/B))=8` 个流。
- `q1_kernel_s16`：shapes 1、12 的 B=16、L=31413/31414；长上下文分支满足 `5*B*L>=9*E`，把最长请求划为 16 流。
- `q1_kernel_s64`：单请求 L=54528–54541 的小 batch、长上下文分支选择 64 流。
- `q1_atomic_s8`：B=15、L=31417–31422 的长上下文分支，以及 B=2、L=2117 的短上下文 tile cap 分支，均选择 8 CTA cluster。
- `q1_atomic_s16`：B=1、L=17845/17846 的小 batch 分支，以及 B=5、L=126921 的非 skew 超长上下文分支，均选择 16 CTA cluster。
- `q4_kernel_s9`：B=15/16 的非 skew 长上下文路径，wave table 将两个 query tile 形成的 grid mass 30/32 映射到 S=9。
- `q4_kernel_s14`：shape 5 的 B=5、L=126920；同一 wave table 把 grid mass 10 映射到 S=14。
- `q4_kernel_s64`：B=1、L=54527–54539；每个请求的两个 query tile 各自暴露 64 流。
- `q4_compact_s8_active6`：shape 4 的 B=11、L=124814，host 选择 S=8 compact；`min(seq_lens)=113103` 且 `4*min>=3*max`，device 分支令 W=6。这不是按每条请求长度比例分配活跃 split。槽 6、7 不生产 numerator，但发布 `M=-inf,L=0`；combine 避免读取其未写的 O_partial。
- `q4_atomic_s8`：shape 9 的 B=2、L=2116；小 batch 初选受 KV tile cap 限制到 8 CTA cluster。
- `q4_atomic_s16`：shape 0 的 B=1、L=17844；小 batch 分支选择 16 CTA cluster。

Q1 每个 CTA 用 16 个 warp，两个 softmax warpgroup 交替处理偶数/奇数 tile；Q4 每个 CTA 用 12 个 warp，一个 softmax warpgroup 处理所有 tile，并保留两个交替的 numerator/denominator parity 状态。Q4 的 MMA/TMA、softmax、correction 组分别使用 64、168、256 的寄存器配置。Q4/kernel 的 FP32 输出 tile 覆盖两个 KV fragment 的复用空间；不是 Q1 的单 fragment 输出视图。

Atomic 文件把一个完整 query tile 定义为 worktile，S 个 CTA 组成一个 cluster worker，split 是 worker member 分工。因此统计合并属于 worktile 内协作。Kernel 文件以单 split 为 worktile，独立 combine phase 显式填写 `target.tile_reduction`。这两种 worktile 边界与其真实通信域一致。

在所属计分域可出现空流或单 tile 的 plan 中，这些边界均显式保留。N=0 的 Correction 只发布独立 `Empty_statistics` 事件，正常 L-final/Y-drain 的 `final` 域为空；N=1 的尾部边通过有界 `tail`/parity 域只引用实际存在的生产者。N=0 仍完成 neutral M/L 与所需统计归约；VP 早释放 TMEM，其他流式加载、turn relay 和 numerator 输出不执行。N=1 只读取一次最大值，另一条 parity 为零，并用 completion-only 的 L/Y 信号补齐排出；最后 `beta_tail=1`。正常排出中，Correction 读回并归还最后两份 Y 后，MMA_VP 即可释放 TMEM；Correction 随后进行最终归一化、转换和 SMEM 输出发布，这些操作与 TMEM 释放之间没有额外的完成等待。Atomic 的 split 0 在启动阶段归零 output，并通过 GPU fence、peer zero-done 通知与 TMA 侧 acquire/async-proxy fence 保证所有 reduce-add 在归零后；各 split 的 contribution 先转 BF16，再由到达次序未规定的 BF16 TMA reduce-add 合入 output。

### 固定版本的源码依据

固定提交：`01fff90d714abc6e84904ea9c44523ba66d34a0e`。

- [默认入口](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution/solution.py)、[输入构造](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_decode/input.py)、[参考实现](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_decode/reference.py)、[benchmark 元数据](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_decode/metadata.json)、[30 个计分输入](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_decode/shapes.json)：contract 和覆盖域。
- [operator.py](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution/qoder_greenfield_decode_attention/operator.py#L59-L224)：wave table、tile packing 与完整默认 dispatch；268–291 行为编译 key/runtime splits；727–747 行为 compact host 条件。
- [gqa_decode_paged.py](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution/qoder_greenfield_decode_attention/gqa_decode_paged.py#L239-L326)：tile/stage 和分配公式；464–466 行为 grid/cluster；623–688 行为角色/寄存器；1077–1145 行为 compact work_splits 和空流；1180–1785 行为 fragment 流水、概率拆分及单 tile 信号；1828–2120 行为 correction、overlapped zero、排出与 reduction dispatch。
- [gqa_decode.py](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution/qoder_greenfield_decode_attention/gqa_decode.py#L164-L325)：FP32 partial epilogue、cluster max/denominator XOR tree 与 normalization；329–553 行为独立 combine、PDL wait、warp-parallel / compact scalar 两种合并。

### 占位与未确定事实

- hardware execution/feature/wiki 按任务要求保留 pending；device=NVIDIA L20D、arch=SM103，不推断未给出的硬件容量。
- stack 采用仓库 requirements 指定的 `nvidia-cutlass-dsl==4.6.1`、`torch>=2.9`、`cuda-python`；未钉死的 CUDA/包版本保持 pending。
- Python `None` 是 sinks 和 o_sf_scale 的真实 ABI 值；schema 没有定稿的 None dtype，使用 `<schema-abi@pending:Python-None-sentinel>`，不虚构张量。
- `Score.slot` 和 KV fragment ring 容量来自未随快照提供的 CUTLASS capacity helper，故保持 pending。其源码关系为：`Rows=16/32`，`SStages=min(floor((C_tmem-6*Rows)/(2*Rows)),4)`，TMEM allocation 为 `next_power_of_two(6*Rows+2*Rows*SStages)`；kernel 的 `B0=180+2324*Rows+16*SStages` 字节，atomic 再加 `32*Rows+88`；`A=128-(B0%128)`，`K0=floor((C_smem-B0-A)/16384)`，`KVStages=K0-(1 if 16*K0>A else 0)`。PT=4、P=4、Y=2、L=2 已由本地源码确定，未占位。不能用假定 SMEM/TMEM 容量填出具体 stage 数。
- fast exp2 的误差界、reciprocal 的误差界、FP32 标量 multiply-add contraction，以及原生 BF16 TMA reduce-add 的内部算术/非正规数语义，没有足够仓库资料，分别保留 numerics pending。明确写出的两次 FP8 概率舍入、FP32 parity 状态、BF16 contribution 舍入和不确定到达次序不依赖这些未知值。

本轮增加条件性占位 `<source-order@pending:FA4-decode/PT-four-slot-reuse/F2>`。当 `F=2` 且 `i+4<N`，K 在下一轮等待 V turn 之前已经预取 PT(i+4)，当前可见依赖没有证明该同槽写入晚于 V 捕获 PT(i)。最短情形需 N≥5；N=4 的最后一轮不发出 PT(4)。源依据为 [gqa_decode_paged.py:1229–1267](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution/qoder_greenfield_decode_attention/gqa_decode_paged.py#L1229-L1267) 与 [1314–1353](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution/qoder_greenfield_decode_attention/gqa_decode_paged.py#L1314-L1353)。F≤1 时已有 turn 链可保护四槽；F=2 的实际配置仍取决于前述未展开的 capacity helper。此项记录缺少的源码顺序保证，**不声称已复现 GPU 错误**，也没有给 plan 补上源码不存在的等待。

该条件只在 8 份文件中保留：`q1_kernel_s8`、`q1_kernel_s16`、`q1_atomic_s8`、`q1_atomic_s16`、`q4_kernel_s9`、`q4_kernel_s14`、`q4_compact_s8_active6`、`q4_atomic_s16`。`q1_kernel_s64`、`q4_kernel_s64` 的 N≤4，`q4_atomic_s8` 的 N≤2，当前计分域无此覆写，未加该占位。

默认入口在本次计分集合外还可能选择 S=1/no-reduce、其他 split 数以及其他 compact 活跃流分配；这 11 份文件未声称覆盖那些未计分的配置。内部库的 sinks、window、BLASST、其他 dtype/布局等开关也未当成默认 benchmark 的 dispatch。

## QSA prefill：1 份

| 文件 | 路径 | Shape IDs |
| --- | --- | --- |
| `qsa_prefill__group12_union_br10_s3.yaml` | Group=12、Br=10、M128/N64、KV 两槽、S/P 三槽；union 预处理后动态持久化 attention | 0, 1, 2, 3 |

四项均为 `q=[4096,24,256]`、两个 KV heads。CP rank 和 fitted index draw 改变选择内容，不改变设计，因此共用一份 plan。`use_prefill_config` 在该入口没有引出另一套设计。其他 Group 的 Br 路由未被计分域触发；构造器的 S/P 默认值仍为三槽，不能将其他 Group 自动写成两槽。

本 plan 保留十行 union 的去重与 membership、两组 softmax 对列的划分、跨 worktile 的动态 ticket/base 生命周期、三槽 S/P 别名、两槽 gather 以及 `QK0,QK1,QK2,PV0,...` 的启动次序。输入范围明确限定完整四元组及 causal tail 的构造：实际 `logical_indices` ABI 是追加 valid_count 后的 `[4096,2052]`，不是 shapes 中构造前的 2051 宽；每个有效块至少有 512 entries，故当前计分域 N≥32。attention 先 BF16 舍入，再乘 sigmoid gate 并再次 BF16 舍入。

账本统一采用每行有效前缀 `J(r)` 的逻辑 sparse baseline；score/probability traffic 与 footprint 使用 `valid_count(r)`，与 KV/MAC 的比较口径一致。这不是声称参考实现的矩形 einsum 会跳过无效位置。

占位包括通用硬件/wiki、未锁定栈版本、近似 exp2/reciprocal 的认证局部误差界，以及外部 TMEM 容量 helper 的返回值；源码确定的 TMEM 视图 `[0,448)` 已实写，未把视图占用当成已核定申请容量。

本轮新增 `<synchronization@pending:QSA_HalfMax_peer_read_before_next_store>`。源码 [qsa_kernel.py:506–512](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/qsa_prefill/aka_solution/qwen38_core_qsa_cutedsl/qsa_kernel.py#L506-L512) 的次序是写本侧 HalfMax、256-thread rendezvous、读取对侧 HalfMax；目前没有找到保证“对侧旧读取完成后，本侧才可写下一轮”的完整依赖，初始 QK0/QK1/QK2 的领先发射也不能补上该保证。plan 因而把 `half_max_*@done` 限定为写入及 rendezvous 完成，把 peer capture 放在 P 动作中，未虚构后置 barrier 或复用边。这是尚待证明的同步保证，**不是已复现的 GPU race 结论**。HalfSum 跨工作覆盖前必经下一工作的首次 HalfMax rendezvous，能保护旧 peer-load，因此没有机械添加相同占位。

来源：[operator](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/qsa_prefill/aka_solution/qwen38_core_qsa_cutedsl/operator.py)、[preprocess](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/qsa_prefill/aka_solution/qwen38_core_qsa_cutedsl/qsa_preprocess.py)、[kernel](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/qsa_prefill/aka_solution/qwen38_core_qsa_cutedsl/qsa_kernel.py)。

## GDN prefill：2 份

| 文件 | 路径 | Shape IDs |
| --- | --- | --- |
| `gdn_prefill__packed_m64_dynamic.yaml` | B>1；M64 value-row halves；静态步长持久化；动态 sequence-end 描述符；gate 与 beta 独立流水 | 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 24, 25, 26, 27, 28, 29 |
| `gdn_prefill__single_m64_static.yaml` | B=1；M64；直接 virtual-head 调度；固定描述符；两槽 raw scalar TMA；合并 scalar ready 与紧凑 CG1 事件流 | 3, 17, 19, 20, 21, 22, 23 |

入口固定 `value_rows_override=64`，HQ=4、HV=32、DK=DV=128，初态与末态均为显式 FP32 tensor。两条路径共用成对的 64-token chunks、BF16 层次逆、双 MMA issuer 和局部 FP32 状态递推；B=1 的差异改变搬运与同步，值得独立维护。两个 value-row halves 各自保留完整序列递推，没有跨 half 的状态归并。

单请求路径在读取 raw chunk c 后，先向同一两槽 raw ring 发出 c+2，再进行当前 gate scan。准备好的 G/gamma/beta 领先输出四个 chunk，共用五槽与一次 ready 发布；写 c+5 之前已完成 O(c)，由传递依赖保护旧标量消费者，无 producer empty acquisition 或 consumer return。CG1 把 Sbar、VKS、NVbar/DecayV 的通知串入一条交替 phase 事件流；最后一次通知同时覆盖 output 与 state-update 两个产品。多请求路径保留独立通知与 descriptor 重绑。

以下边界在各自 plan 内处理，不再拆文件：

- `static_tma_full_chunks` 的精确条件是 **T%128==0**，只消除 scalar capture 的尾部 predicate。七个计分 B=1 长度为 14784、2258、6、61、134、525、1377，均未命中；T=14784 虽能被 64 整除，仍需一个 synthetic chunk。统一使用 `Nc=2*ceil(L/128)`，padding 为 Q/K/V=0、gate=1、beta=0。
- 零初态和 continuation 都传入非 None 的 FP32 state，因此均保留 initial-state load；不能把数值全零当成 `use_initial_state=False`。
- B=1 wrapper 仍在计时内执行 `cu_seqlens.to(int32)` 并传入结果，但所选 kernel 使用 start=0、length=T，不读取转换后的数组。plan 保留实际 timed cast 和 stream 顺序，未虚构 offset32 到主核的数据 handoff。

**纠正旧版说明：固定描述符不代表命中静态 scale。** `input.py` 传入的 Python binary64 `1.0/math.sqrt(128)` 为 `0.08838834764831843`；host 与 `128**-0.5`（`0.08838834764831845`）严格比较，结果为假。因此全部计分 shape 都使用 runtime scale。后续 FP32 cast 相同，不会改变此前 host dispatch 的结果。

通用库的 M128、无初态、其他 dtype、checkpoint、indexed state 等能力未被当前入口/输入域选中；B=2/HQ=HV 的 compact-event 特化也不适用，因为实际 HQ≠HV。

两份共有的占位包括通用硬件/wiki、未锁定栈版本、PyTorch cast 的实际 launch 展开与工作/微流水、fast log2/exp2 的认证误差界，以及外部 TMEM helper 的返回容量。已知 TMEM 视图 `[0,512)` 与别名完整保留。只有多请求 plan 另外保留 `TensorMapManager_inflight_descriptor_update_semantics`：源码有 descriptor 更新和 fence，但库对旧在途访问的精确保障未随仓库提供；没有添加源码不存在的每工作全量 drain。`cast_offset` 是 opaque library flow 槽，不能据它断言转换恰好一个 GPU launch。

来源：[实际入口](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/gdn_prefill/aka_solution/solution.py)、[dispatch 与 scale/full-pair 谓词](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/gdn_prefill/aka_solution/gdn_prefill.py)、[可读 kernel](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/gdn_prefill/aka_solution/gated_delta_net_chunked.py)、[scheduler](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/gdn_prefill/aka_solution/gated_delta_net_tile_scheduler.py)、[input](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/gdn_prefill/input.py)。

## 共通占位与核验范围

| 项目 | 处理 |
| --- | --- |
| 硬件资料 | device=NVIDIA L20D、arch=SM103；execution、feature、wiki 按要求使用 `<...@pending:...>`。148 SM 的路由核对依据是仓库 [GDN README 的 L20D 环境记录](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/gdn_prefill/aka_solution/README.md)，没有由 B300 注释反推。 |
| FA4 实现栈 | torch>=2.9、cuda-python、nvidia-cutlass-dsl==4.6.1；requirements 未锁定的实际版本与 CUDA/driver 保持 pending。 |
| QSA 实现栈 | torch>=2.9、cuda-python>=13.0、nvidia-cutlass-dsl>=4.6.1；未锁定实际版本 pending。 |
| GDN 实现栈 | torch>=2.9、cuda-python>=13.0、nvidia-cutlass-dsl[cu13]>=4.5.0；入口要求 CUDA major>=13，未锁定实际版本 pending。 |
| Python None ABI | FA4 的 sinks、o_sf_scale 明确必传 Python None；`<schema-abi@pending:Python-None-sentinel>` 只表示现行 schema 缺少这一 dtype 写法，真实取值并不未知。 |
| 数值原语 | 实际 casts、矩阵累加精度和近似路径均实写；仓库没有给出的认证局部误差界保留 pending，不用最终输出容差代替。 |
| 外部实现 | 未随仓库提供的容量 helper、library lowering 或 descriptor 保证在相应 plan 及上文逐项说明，未凭经验补数值。 |

本轮另起六个全新上下文，分两轮审阅全部 19 份：每轮分别负责 prefill 5 份、decode 11 份、GDN/QSA 3 份；先读 README/六区文档和 plan、记录无源码重建，再核对实际入口实现。第二轮没有复用第一轮 reviewer；另一个全新上下文仅交叉核查 Decode PT 复用保证。主审逐项对照 docs/源码检查候选 diff，再决定合入；没有按行数裁剪核心设计。

| 本轮修订 | 范围与理由 |
| --- | --- |
| Prefill S5 的 partial 生命周期 | `Opartial`、`LSEpartial` 使用完整张量 shape，`life` 对齐为 `stored invocation`；不再与逐行实例 `(z,a,h)` 混用。 |
| Decode 尾部事件 | `q1_kernel_s16`、`q1_atomic_s8`、`q4_atomic_s8`、`q4_compact_s8_active6` 四份的并集：消除可达 N=0/1 下不存在的固定索引生产者，区分空统计通知与正常 L/Y 排出；保留统计提前推进、TMEM 早释放。 |
| Decode 最大值与 P 槽协议 | 11 份分别表达共享 Maximum 的写入和 m 的可读发布：首次写入等初始化 credit，后续写入等旧读者，m 发布等 P 槽归还；保留写入可先于 P 等待的重叠。 |
| Decode PT 复用 | 仅 8 份存在 N≥5 的计分域记录 `F=2` 时尚未闭合的四槽复用保证；另外 3 份不添加此项。未发明保护边。 |
| QSA 账本与完成边界 | 统一有效前缀 baseline；区分 HalfMax 发布完成与 peer capture，并明确上文未闭合的复用保证。 |
| FA4 wiki 引用 | 16 份按 docs2 的明确要求为 wiki 叶子补引号，引用值保持不变；其中 4 份只有此项格式修改。 |
| GDN | 两份经复核保持原样，未为产生 diff 添加字段或重复解释。 |

交付前使用未改动的六区结构校验器检查全部 YAML 的解析、重复键、意外 null、固定结构与顶层顺序、输出容差覆盖、phase/launch/handoff、符号与生命周期、resource/view、lane/pipe/edge 及 realization 引用；19 份均无 YAML 注释、anchor 或 alias。另核对源码 host 策略与 94 个计分 shape 的唯一映射、dispatch/文件名和逐文件 version。针对本轮实际修改的 Decode 尾部，对全部实际可达的 75 组“文件/N”展开相关动作与前驱，未发现不存在的相关生产者；并确认统计没有增加最终 VP 依赖，TMEM 释放没有增加 normalization 依赖。新增的初始 credit、旧最大值读者到写入、P 槽归还到 m 发布三类边另作 300 个本地实例的有界展开：SStages=1/2/3/4 只作为符号参数，N=0 按明确的源码分支处理，检查局部 pipe、默认/显式边及事件环；这不核定实际容量，也不证明完整 cluster 协议；更新动作名使用 `update:Maximum`，避免与 `publish:m` 的默认同名依赖形成伪环。该定向检查不解释完整事件图，也不替代跨 CTA 原语或外部库保证的证明。首尾迭代、缓冲别名和跨工作复用另经源码审阅，静态检查不是完整的 GPU race 或活性证明。

**本次没有执行 GPU 编译、数值 benchmark 或性能复测。** 仓库已有的测试记录不等于本次重新运行；plan 中未增加新的性能或数值通过声明。

压缩包根目录只有 19 份 YAML 和本 notes.md，已核对 ZIP CRC 与逐文件字节一致性。解压目标为 `schema_wiki/aka-four-problems-20261020/`；如果目录里已有前版四个不带 dispatch 后缀的 YAML，它们属于旧的代表路径版本，不应继续作为这一组新文件的默认入口。
