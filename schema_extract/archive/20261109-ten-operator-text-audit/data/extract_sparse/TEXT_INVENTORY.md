# extract_sparse: full open-text inventory

These are exact-text groups, not proof of semantic equivalence across contexts.

## t_53637bf71aea4102 | algorithm_contract.math | 839 chars | 1 declarations

root:101 contract.algorithm_requirement.compute[1].math

Independently for each token t, find b with cum_seq_lens_q[b] <= t < cum_seq_lens_q[b+1]. Let visible = seq_lens[b] - (cum_seq_lens_q[b+1] - cum_seq_lens_q[b]) + t - cum_seq_lens_q[b] + 1. J(t) contains the slot indices j satisfying sparse_indices[t,j] >= 0 and either 0 <= j < min(visible, SW) or SW <= j < sparse_topk_lens[t]. In particular, unused SWA slots before column SW do not shift the start of the compressed segment; -1 invalidates a slot independently of lengths. For j in J(t), V[t,j,:] is the row sparse_indices[t,j] of swa_kv_cache when j < SW, otherwise the row of compressed_kv_cache. Set V[t,j,:] = 0 for other slots. Each slot contributes independently: repeated IDs retain their multiplicity, and SWA/compressed states are never deduplicated by logical token position. The same row supplies both K and V for every head.

## t_7487228c6567ee34 | algorithm_contract.math | 1018 chars | 1 declarations

root:119 contract.algorithm_requirement.compute[2].math

Token t and head h are independent, with 0 <= t < T and 0 <= h < H. Reference dot products, exponentials, reductions and normalization use FP32 tensors. Arithmetic and the final BF16 conversion are unclamped; overflow and invalid operations may produce infinities or NaNs. If J(t) is empty, out[t,h,d] = bf16(0) for all 0 <= d < D, without evaluating a softmax. Otherwise, for j in J(t), L[t,h,j] = bmm1_scale * sum_{d=0}^{D-1} query[t,h,d] * V[t,j,d]. Let m = max(max_{j in J(t)} L[t,h,j], sinks[h]). Let p_j = exp(L[t,h,j] - m) for j in J(t), and z = sum_{j in J(t)} p_j + exp(sinks[h]-m). For 0 <= d < D, out[t,h,d] = bf16((sum_{j in J(t)} p_j * V[t,j,d]) * (bmm2_scale / z)). There is one denominator over both caches. A sink contributes no value term. L is used only on J(t); its other slots are defined as -inf. Only output_buffer and workspace_buffer may be written. The entire output is overwritten on every call; all other inputs and their storage are preserved. Scratch contents are not returned or compared.

## t_909fcefde83dd6f6 | algorithm_contract.math | 2258 chars | 1 declarations

root:63 contract.algorithm_requirement.compute[0].math

1 <= B <= 32; H in {8,16,32,64,128}; 1 <= T <= 16384. For 0 <= b < B, let q_b = cum_seq_lens_q[b+1] - cum_seq_lens_q[b]. 1 <= q_b <= min(4096, seq_lens[b]); sum_b q_b = T; sum_b seq_lens[b] <= 1048576; max_q_len = max_b q_b. Both caches are contiguous HND tensors with one shared latent KV head, physically stored as [page, entry, latent]. Flatten the first two axes to address row i as [floor(i/page_size), i mod page_size, :], where page_size is SwaPageSize for SWA and CompPageSize for the compressed pool. Query, output and index/length tensors in the supplied caller are contiguous; tensor arguments reside on the same CUDA device. Physical page placement may be contiguous or scattered, independently of logical selection. A prepared call has a caller-side compression ratio r in {0,1,2,4,128} and sparse selection budget k in {512,1024}; these are not forward arguments. SwaPageSize in {128,256}; SwaPageSize is divisible by max(1,r); CompPageSize = SwaPageSize / max(1,r). If r = 0, compressed_kv_cache aliases swa_kv_cache and IndexWidth = SW. If r in {1,2,4}, IndexWidth = SW + k. If r = 128, IndexWidth = SW + max(SW, pow2ceil(max(1,floor(max_b(seq_lens[b])/r)))), where pow2ceil(x) is the smallest power of two at least x. Thus SW <= IndexWidth <= 8320 and IndexWidth is divisible by 4. The first SW slots encode the causal sliding window; compressed slots encode an already selected causal subset (all compressed positions for r=128). Score-related, local, strided, shared-prefix and random selections are supported; this core consumes the supplied IDs without reranking or reconstructing history. RoPE, selection, cache publication and output projection are outside this task. All numeric data inputs are finite; the old output_buffer contents are unrestricted. The valid task domain requires bmm1_scale = D^(-0.5) and bmm2_scale = 1.0; D^(-0.5) denotes the Python result of D ** (-0.5). Both scales and sinks must be explicitly supplied; sinks cannot be None. Scales are Python floats, not tensor scales. None denotes Python None. The supplied constructor uses a NaN-filled output_buffer and WorkspaceSize = 134217728 zero-initialized bytes. The bound caller requires output_buffer even though the reference also supports allocating an output.

## t_044ae9bfd90b57c3 | algorithm_contract.meaning | 37 chars | 1 declarations

root:54 contract.algorithm_requirement.input.max_q_len.meaning

Maximum query length across requests.

## t_1eb37f2023e5b50c | algorithm_contract.meaning | 127 chars | 1 declarations

root:34 contract.algorithm_requirement.input.sparse_indices.meaning

Physical row IDs in two separate pools; the first SW columns address SWA and the remaining columns address the compressed pool.

## t_386083a2942fce5a | algorithm_contract.meaning | 81 chars | 1 declarations

root:38 contract.algorithm_requirement.input.compressed_kv_cache.meaning

Prepared compressed shared key/value pool; supplied even in the single-pool case.

## t_3e7b439782d7a6a2 | algorithm_contract.meaning | 63 chars | 1 declarations

root:53 contract.algorithm_requirement.input.cum_seq_lens_q.meaning

Prefix offsets delimiting each request in the query token axis.

## t_3fdcbbce665d64ae | algorithm_contract.meaning | 127 chars | 1 declarations

root:42 contract.algorithm_requirement.input.output_buffer.meaning

Caller argument out; required destination whose previous contents are ignored and whose object identity is preserved on return.

## t_42845676767027a5 | algorithm_contract.meaning | 62 chars | 1 declarations

root:47 contract.algorithm_requirement.input.bmm2_scale.meaning

Python scalar multiplier on the normalized weighted value sum.

## t_5203e99c344b0e32 | algorithm_contract.meaning | 78 chars | 1 declarations

root:139 contract.algorithm_requirement.compute[2].symbol.L.meaning

Scaled FP32 attention logits, with inactive slots masked by negative infinity.

## t_59e25fa9bb6c3733 | algorithm_contract.meaning | 38 chars | 1 declarations

root:56 contract.algorithm_requirement.input.swa_topk_lens.meaning

Disabled separate SWA length override.

## t_68f2c7d2d7088299 | algorithm_contract.meaning | 38 chars | 1 declarations

root:27 contract.algorithm_requirement.constant.D.meaning

Shared key and value latent dimension.

## t_6c0a412e77cbf05e | algorithm_contract.meaning | 102 chars | 1 declarations

root:117 contract.algorithm_requirement.compute[1].symbol.V.meaning

Selected shared key/value latents indexed by query and physical slot, zero outside the valid slot set.

## t_742dbe9f5cec26ca | algorithm_contract.meaning | 73 chars | 1 declarations

root:32 contract.algorithm_requirement.input.workspace_buffer.meaning

Writable scratch with no logical output value or required final contents.

## t_75e5bea22ed93a36 | algorithm_contract.meaning | 38 chars | 1 declarations

root:58 contract.algorithm_requirement.input.extra_sparse_topk_lens.meaning

Disabled extra sparse length override.

## t_7edb9f9411c8ed9a | algorithm_contract.meaning | 51 chars | 1 declarations

root:46 contract.algorithm_requirement.input.bmm1_scale.meaning

Python scalar multiplier on query-key dot products.

## t_9c77fe8b72c5ccf0 | algorithm_contract.meaning | 46 chars | 1 declarations

root:31 contract.algorithm_requirement.input.swa_kv_cache.meaning

Prepared sliding-window shared key/value pool.

## t_b071ee06da1e2fe9 | algorithm_contract.meaning | 40 chars | 1 declarations

root:28 contract.algorithm_requirement.constant.SW.meaning

Reserved sliding-window slots per query.

## t_b9fb77e7032c817f | algorithm_contract.meaning | 89 chars | 1 declarations

root:39 contract.algorithm_requirement.input.sparse_topk_lens.meaning

Per-query exclusive end of the compressed slot interval, including the SW reserved slots.

## t_bc87bf251fde3d9f | algorithm_contract.meaning | 82 chars | 1 declarations

root:55 contract.algorithm_requirement.input.enable_pdl.meaning

Unspecified launch preference in the selected caller, with no mathematical effect.

## t_c672715739dd73c6 | algorithm_contract.meaning | 101 chars | 1 declarations

root:60 contract.algorithm_requirement.output.out.meaning

Attention result overwriting every element of output_buffer; the returned tensor is that same object.

## t_cbd8d8e88d2a6c9e | algorithm_contract.meaning | 40 chars | 1 declarations

root:52 contract.algorithm_requirement.input.kv_layout.meaning

Cache layout selector for this callable.

## t_da0619ee0511ad51 | algorithm_contract.meaning | 68 chars | 1 declarations

root:49 contract.algorithm_requirement.input.sinks.meaning

Per-head extra softmax logit with zero value, shared by all queries.

## t_db334a5855ca27f0 | algorithm_contract.meaning | 76 chars | 1 declarations

root:40 contract.algorithm_requirement.input.seq_lens.meaning

Final context lengths of the requests, including their current query tokens.

## t_e1513a31f3b7b807 | algorithm_contract.meaning | 33 chars | 1 declarations

root:57 contract.algorithm_requirement.input.extra_sparse_indices.meaning

Disabled extra sparse index pool.

## t_e4939edf260d9421 | algorithm_contract.meaning | 62 chars | 1 declarations

root:116 contract.algorithm_requirement.compute[1].symbol.J.meaning

Function giving the valid selected slot set for a query token.

## t_ec744c7aab5ffb51 | algorithm_contract.meaning | 47 chars | 1 declarations

root:30 contract.algorithm_requirement.input.query.meaning

Query latents for concatenated ragged requests.

## t_17509f4ab36ba0af | algorithm_contract.range | 51 chars | 1 declarations

root:53 contract.algorithm_requirement.input.cum_seq_lens_q.range

Strictly increasing, starting at 0 and ending at T.

## t_31f68b66c211ffa8 | algorithm_contract.range | 22 chars | 1 declarations

root:54 contract.algorithm_requirement.input.max_q_len.range

1 <= max_q_len <= 4096

## t_5086d11b0be9c290 | algorithm_contract.range | 27 chars | 1 declarations

root:40 contract.algorithm_requirement.input.seq_lens.range

1 <= seq_lens[b] <= 1048576

## t_57242d69630f9169 | algorithm_contract.range | 21 chars | 1 declarations

root:46 contract.algorithm_requirement.input.bmm1_scale.range

bmm1_scale = D^(-0.5)

## t_a06d554ae02fcfc1 | algorithm_contract.range | 126 chars | 1 declarations

root:37 contract.algorithm_requirement.input.sparse_indices.range

Consumed IDs are -1 (invalid) or nonnegative in-bounds rows of the corresponding flattened pool; unconsumed slots are ignored.

## t_dfb1cae99b01fe9f | algorithm_contract.range | 39 chars | 1 declarations

root:39 contract.algorithm_requirement.input.sparse_topk_lens.range

SW <= sparse_topk_lens[t] <= IndexWidth

## t_e8267df39f2c1ae4 | algorithm_contract.range | 11 chars | 1 declarations

root:52 contract.algorithm_requirement.input.kv_layout.range

Only 'HND'.

## t_fef2c3b6ba5cf44f | algorithm_contract.range | 51 chars | 1 declarations

root:45 contract.algorithm_requirement.input.output_buffer.range

Any BF16 bit pattern, including NaN and infinities.

## t_958cf0ceb9f5c4ee | contract.compute_name | 24 chars | 1 declarations

root:118 contract.algorithm_requirement.compute[2].name

Joint softmax and output

## t_aa55d60113a80863 | contract.compute_name | 35 chars | 1 declarations

root:62 contract.algorithm_requirement.compute[0].name

Domain and prepared cache interface

## t_c5012a0f862e529b | contract.compute_name | 24 chars | 1 declarations

root:100 contract.algorithm_requirement.compute[1].name

Visible selected latents

## t_f00c971fe1c390e2 | contract.tolerance | 727 chars | 1 declarations

root:143 contract.algorithm_requirement.golden_reference.tolerance.out

Source record for out compared with Model.forward: metadata.json declares default atol=0.01, rtol=0.05 and max_mismatch_fraction=0.0. CERTIFIED_BEST.json states that comparator/tolerances are unchanged but provides no comparator definition. The formula, relative-error basis, comparator arithmetic precision and NaN/Inf handling are unspecified. input.py declares out as the observable writeback; external comparator routing between that writeback and the return value is not provided. No relative_to value is inferred and the schema's default atol/rtol interpretation is not adopted. The returned tensor must be the supplied output_buffer object; return and writeback denote the same BF16 result, with input storage preserved.

## t_0388393e299f901d | dispatch.md | 402 chars | 1 declarations

dispatch.md:22 dispatch.md

- 强制配置、注册 fast-call、诊断构建和修改扩展开关不在这棵出厂策略设计树内；共享 Contract 不据此缩小。 - 路由选中不保证运行前提满足：共享内存预算、opt-in 属性或 TMA tensor-map 编码失败会拒绝执行；源码没有为这些拒绝提供另一条算法 fallback。编译时未启用 tcgen05 的回落由现有 mma 节点覆盖。 - 保留源实现限制：DQ>1 对同一 m/l 地址存在非原子重复写；probe 超时可能与后续调用复用同一 host gate 缓冲发生竞争。计划不据此新增跨 CTA 或跨调用序列化，也不宣称这些源路径已获正确性保证。probe 已提交但轮询未齐的计算及 stream 接续仍属于所选节点。 - root 的 tolerance 保留外部验收器公式、相对误差基准、精度及检查路由未提供的来源记录；这些缺口不被当作数值保证，也不阻止当前设计提取。

## t_1ae21abd8030ffa6 | dispatch.md | 243 chars | 1 declarations

dispatch.md:10 dispatch.md

| [probe_serial_packed](probe_serial_packed.yaml) | tc_packed | 串行 probe 后接受 packed 的全批次请求。 | 增加独立 probe、映射 host gate 和轮询选择；全批次 packed 在 caller stream 上等待 probe 完成后执行。 | P.dual=1，hidden 在提交 probe 前返回 false；串行 probe_begin 成功，实际 R 满足 0<R<=2.6。 |

## t_2304aaf0197a9461 | dispatch.md | 7 chars | 1 declarations

dispatch.md:20 dispatch.md

## 覆盖缺口

## t_32b044faa5d8348b | dispatch.md | 38 chars | 1 declarations

dispatch.md:3 dispatch.md

| 节点 | 直接父 | workload | 相对父差异 | 选择依据 |

## t_422f8aef3863fda5 | dispatch.md | 315 chars | 1 declarations

dispatch.md:8 dispatch.md

| [probe_hidden_packed](probe_hidden_packed.yaml) | probe_serial_packed | 隐藏 probe 后继续 packed 余段的请求。 | 将 probe 放到 side stream；全批次注意力拆成 caller stream 的 packed 前缀和 side stream 的 packed 余段，保留 fork/join，余段不等待前缀完成。 | P.dual=1；hidden 前置检查及 fork/probe_begin 成功，K=N-8>=1、ceil_div(T,P.gt)>=2*K，且 0<R<=2.6；前缀与余段以 K*P.gt 为界。 |

## t_4cae608ab09d4aa1 | dispatch.md | 237 chars | 1 declarations

dispatch.md:11 dispatch.md

| [tc_heads](tc_heads.yaml) | mma_direct | 单 query 的 head block 已能填满设备的 tcgen05 请求。 | 改用 cfg90 的 TMA gather4、TMEM 和 QIS2 双发射者 QK 分和；每 CTA 处理单 query 的 64 行 head block，直接归一化写回。 | 初始 P 为非 dual 的 cfg90 且 gt=1，对应 make_plan 的 tc_eligible 分支。 |

## t_4e68f7cf72a00d05 | dispatch.md | 175 chars | 1 declarations

dispatch.md:5 dispatch.md

| [root](root.yaml) | 无 | 自动策略中需要 split-KV 的请求。 | 代表设计：mma.sync 在线注意力写 FP32 分块结果，再由独立 combine 归一化并写回 BF16。 | 初始 P 为非 dual 的 mma 配置且 splits>1；由 make_plan 的 CTA 并行度和工作区容量规则选择。 |

## t_4fe9e5689d7c577a | dispatch.md | 256 chars | 1 declarations

dispatch.md:9 dispatch.md

| [probe_serial_mma](probe_serial_mma.yaml) | mma_direct | 串行 probe 后选择 mma 的全批次请求。 | 增加独立 probe 与 host 轮询；全批次 mma 在同一 caller stream 上接续 probe，不因统计不可用而删除 probe。 | P.dual=1，hidden 在提交 probe 前返回 false；串行 probe_begin 已提交 probe，但实际 R 不满足 0<R<=2.6，选择 P.alt_*。 |

## t_7875020cff23f0cd | dispatch.md | 6 chars | 1 declarations

dispatch.md:1 dispatch.md

# 路径索引

## t_7f9df6b7d9ff8767 | dispatch.md | 180 chars | 1 declarations

dispatch.md:16 dispatch.md

probe 的选择依据是本次有界轮询后实际读到的统计结果，轮询未齐仍可能接受 packed；已提交 probe 的工作和接续不因此消失。head 分块、KV chunk/split 数、输出维分片 DQ、ring 深度、combine 线程数及 packed 家族内的每 CTA query 数变化保留为节点参数；节点划分保留物化、生产者和计算提交集合的差异。

## t_922f33c0c656e3f2 | dispatch.md | 246 chars | 1 declarations

dispatch.md:6 dispatch.md

| [mma_direct](mma_direct.yaml) | root | 不分 KV 的全批次 mma.sync 请求，包括未提交 probe 的回落。 | 去掉 split 物化与 combine；在 attention 内加 sink、归一化并写回 BF16，保留分块及 DQ 参数。 | 初始 P 非 dual、非 cfg90 且 splits=1；或 dual 隐藏路径未进入、串行 probe_begin 因 gate 不可用而未提交 probe，改用 P.alt_*。 |

## t_9374b8371e3558ae | dispatch.md | 216 chars | 1 declarations

dispatch.md:18 dispatch.md

非 dual 的直接提交和成功图回放使用同一计算设计。内部录图先同步 caller stream；BeginCapture 失败在 dispatch 前返回，capture/实例化失败不执行录制的 kernel，图失败禁用缓存后接续同一 chosen plan 的直接 dispatch。纯录图不增加计算实例；外层图回放采用录制或显式更新后的选择、标量和绑定，张量内容在执行时读取，录制的 probe 不会在回放时重新选择后段类型。

## t_a0ab965714ea3665 | dispatch.md | 239 chars | 1 declarations

dispatch.md:12 dispatch.md

| [tc_packed](tc_packed.yaml) | tc_heads | 低 head 数、可直接选择多 query 打包的请求。 | 多个 query 合用 64 行；新增按池组织的索引并集及成员标记，保留同 query 内重复次数，并在 gather 前复用 KV ring 构建索引。 | 初始 P 为非 dual 的 cfg90 且 gt>1；make_plan 在备选 mma 不分 KV 且 gt<=2.6 时免 probe，出厂策略下即 gt=2。 |

## t_bb3390056beb0a24 | dispatch.md | 108 chars | 1 declarations

dispatch.md:14 dispatch.md

本索引对应 Model.forward 的出厂自动策略，包含未编入 tcgen05 时的 mma 回落。各节点条件在共享 Contract 全域独立求值，直接父仅表示设计派生来源；精确条件与版本以各 YAML 为准。

## t_ce720dae23a43e52 | dispatch.md | 245 chars | 1 declarations

dispatch.md:7 dispatch.md

| [probe_hidden_mma](probe_hidden_mma.yaml) | probe_hidden_packed | 隐藏 probe 后选择 mma 余段的请求。 | 仅把余段换成 P.alt_* 的不分 KV mma.sync；已提交的 packed 前缀、side-stream probe 和最终 join 均保留。 | P.dual=1，满足 hidden 的资源、整批次、双波及 fork/probe_begin 门槛；实际 R 不满足 0<R<=2.6。 |

## t_05176cf5bcdd9159 | dispatch.when | 1539 chars | 1 declarations

mma_direct:9 meta_data.dispatch.when

Let W be the byte capacity of the actual workspace passed by Model.forward after its workspace identity cache or private scratch selection, and let N be dsv4::nsm_cached(). Let P be dsv4::make_plan(H,T,IndexWidth,W,0,0,N,true) in the fixed candidate/dsv4_kernel.cu, including its non-tcgen05 build fallback. P is the initial plan: cfg selects the configuration, splits the KV partitions, and dual indicates probe-gated packing; alt_* holds its unsplit mma alternative. Select exactly when the shipped automatic policy is active (force_cfg=0, force_s=0, Model.fast is false, normal non-diagnostic build and unchanged default extension switches), and either: (a) P.dual=0, P.cfg!=90 (TC_CFG_QIS2), and P.splits=1; or (b) P.dual=1, dual_dispatch_hidden returns false before any probe or packed attention is submitted, and the subsequent caller-stream probe_begin returns false because gate_pinned() or gate_mapped() is null. This returns ratio=-1 without submitting a probe and selects P.alt_* for the whole batch. In (b), the cached gate pointers are unavailable, so hidden dispatch returns before recording a fork event or enqueuing a side-stream wait. Let P_run=P in (a); in (b), P_run uses P.alt_cfg/alt_hb/alt_nb/alt_splits/alt_kc, with gt=1, dual=0 and ws_bytes=0. Both cases submit one unsplit attention kernel. Direct execution and execution of the corresponding recorded graph use this same selection. Pure capture only records work; replay uses the recorded host choice, scalars and bindings, with tensor contents read at execution.

## t_6cd3fa590b2a6b23 | dispatch.when | 1899 chars | 1 declarations

probe_hidden_packed:10 meta_data.dispatch.when

Let W be the byte capacity of the workspace actually passed by Model.forward following its identity cache or private scratch selection, and N=dsv4::nsm_cached(). Let P=dsv4::make_plan(H,T,IndexWidth,W,0,0,N,true) in the fixed source, including its compile-time fallback when tcgen05 is not built. P is the initial plan; dual denotes probe-gated packing, cfg the kernel, and gt queries per CTA. Select exactly when the shipped automatic policy is active (force_cfg=0, force_s=0, Model.fast is false, normal non-diagnostic build and unchanged default extension switches), P.dual=1, dual_dispatch_hidden passes its prechecks and successfully records the caller fork, queues the side-stream wait and submits probe_begin, and the actual R returned by probe_end satisfies 0<R<=fp32(2.6). The hidden prechecks require P.gt>1, P.splits=P.alt_splits=1, the original full input token range, available nonblocking side stream, fork/join events and mapped gate, and K=N-8>=1 with ceil_div(T,P.gt)>=2*K. The prefix owns [0,K*P.gt); the accepted packed remainder owns [K*P.gt,T). Both dispatches clear dual. P.gt starts at 64/H and halves until P.gt*IndexWidth<=2560 and ceil_div(T,P.gt)>=N, returning one if the result exceeds T. Dual packing requires the alternative allow_tc=false plan to have splits=1 and P.gt>2.6, so GT is 4 or 8. R is the FP32 mean, accumulated in FP64, of the positive values actually read from the eight mapped sample slots after the bounded poll. A timeout can still produce an accepted partial mean; selection uses observed R, not final samples. The extension returns directly after the hidden join is queued, without its private graph cache. External capture, if performed, fixes host selection at recording; replay executes the recorded kernels with recorded bindings and current device tensor contents, without rerunning host polling. Pure capture produces no GPU completion events.

## t_96e978e7c8dfbbf3 | dispatch.when | 2111 chars | 1 declarations

probe_hidden_mma:10 meta_data.dispatch.when

Let W be the byte capacity of the workspace actually passed by Model.forward following its identity cache or private scratch selection, and N=dsv4::nsm_cached(). Let P=dsv4::make_plan(H,T,IndexWidth,W,0,0,N,true) in the fixed source, including its compile-time fallback when tcgen05 is not built. P is the initial plan; dual denotes probe-gated packing, cfg the kernel, and gt queries per CTA. Select exactly when the shipped automatic policy is active (force_cfg=0, force_s=0, Model.fast is false, normal non-diagnostic build and unchanged default extension switches), P.dual=1, dual_dispatch_hidden passes its prechecks and successfully records the caller fork, queues the side-stream wait and submits probe_begin, and the actual R returned by probe_end does not satisfy 0<R<=fp32(2.6). The hidden prechecks require P.gt>1, P.splits=P.alt_splits=1, the original full input token range, available nonblocking side stream, fork/join events and mapped gate, and K=N-8>=1 with ceil_div(T,P.gt)>=2*K. The prefix owns [0,K*P.gt); the mma remainder owns [K*P.gt,T). Both dispatches clear dual. The remainder uses P.alt_cfg/alt_hb/alt_nb/alt_splits/alt_kc with gt=1 and ws_bytes=0; this does not re-plan the shorter range or change the packed prefix. P.gt starts at 64/H and halves until P.gt*IndexWidth<=2560 and ceil_div(T,P.gt)>=N, returning one if the result exceeds T. Dual packing requires the alternative allow_tc=false plan to have splits=1 and P.gt>2.6, so GT is 4 or 8. R is the FP32 mean, accumulated in FP64, of the positive values actually read from the eight mapped sample slots after the bounded poll. A timeout can still produce a partial mean; rejection uses observed R, not final samples. R=-1 (no positive observations) and R>fp32(2.6) both select this node. The extension returns directly after the hidden join is queued, without its private graph cache. External capture, if performed, fixes host selection at recording; replay executes the recorded kernels with recorded bindings and current device tensor contents, without rerunning host polling. Pure capture produces no GPU completion events.

## t_a5b629321d9553cf | dispatch.when | 1868 chars | 1 declarations

probe_serial_packed:10 meta_data.dispatch.when

Let W be the byte capacity of the workspace actually passed by Model.forward after its identity cache or private scratch selection, and N=dsv4::nsm_cached(). Let P=dsv4::make_plan(H,T,IndexWidth,W,0,0,N,true) in the fixed source, including its compile-time fallback when tcgen05 is not built. P is the initial plan; dual denotes probe-gated packing, cfg the kernel, and gt queries per CTA. Select exactly when the shipped automatic policy is active (force_cfg=0, force_s=0, Model.fast is false, normal non-diagnostic build and unchanged default extension switches), P.dual=1, dual_dispatch_hidden returns false before submitting any probe, the subsequent probe_begin on the caller stream succeeds, and the actual R returned by probe_end satisfies 0<R<=fp32(2.6). The hidden path can decline for unavailable side-stream/event resources, failed fork setup, or insufficient groups: with K=N-8 it requires K>=1 and ceil_div(T,P.gt)>=2*K. Gate unavailability also declines hidden execution, but this node requires the subsequent serial gate acquisition to succeed. P.gt starts at 64/H and halves until P.gt*IndexWidth<=2560 and ceil_div(T,P.gt)>=N, returning one if the result exceeds T. Dual packing requires the alternative allow_tc=false plan to have splits=1 and P.gt>2.6; hence GT is 4 or 8 here. R is the FP32 mean, accumulated in FP64, of the positive values actually read from the eight mapped sample slots after the bounded polling loop. A timeout can still produce an accepted partial mean. This selection uses that observed R, not a prediction from all final samples. The extension's private graph replay/capture is ineligible because the initial P.dual remains one. External capture, if performed, fixes the host selection at recording; execution uses the recorded probe and attention bindings and current device input contents, without rerunning the host poll.

## t_b4c761c91c878391 | dispatch.when | 1321 chars | 1 declarations

probe_serial_mma:9 meta_data.dispatch.when

Let W be the actual workspace byte capacity after Model.forward's identity cache or private scratch selection, and N be dsv4::nsm_cached(). Let P be dsv4::make_plan(H,T,IndexWidth,W,0,0,N,true) in candidate/dsv4_kernel.cu, including its compile-time fallback when tcgen05 is not built. P.dual denotes probe-gated packing; P.gt is its candidate query grouping, and P.alt_* is the unsplit mma alternative resolved with allow_tc=false. Select exactly when the shipped automatic policy is active (force_cfg=0, force_s=0, Model.fast is false, normal non-diagnostic build and unchanged default extension switches), P.dual=1, dual_dispatch_hidden returns false before submitting any probe or packed attention, the subsequent caller-stream probe_begin returns true and submits a probe, and the actual probe_end result R does not satisfy 0<R<=fp32(2.6). R is defined by probe_select below, including partial or unavailable observations after bounded polling. The whole batch then uses P.alt_cfg/alt_hb/alt_nb/alt_splits/alt_kc, with gt=1, dual=0 and ws_bytes=0. These resolve to cfg8 in this scope. For execution of an externally recorded graph, use the recorded host choice, scalars and bindings; device reads use current tensor contents. Pure capture records the two kernels and does not itself execute them or complete a probe.

## t_ca0ce85467e62a74 | dispatch.when | 1154 chars | 1 declarations

tc_packed:10 meta_data.dispatch.when

Let W be the byte capacity of the workspace actually passed by Model.forward following its identity cache or private scratch selection, and N=dsv4::nsm_cached(). Let P=dsv4::make_plan(H,T,IndexWidth,W,0,0,N,true) in the fixed source, including its compile-time fallback when tcgen05 is not built. P is the initial plan; cfg selects the kernel, gt the tokens per CTA, and dual denotes probe-gated packing. Select exactly when the shipped automatic policy is active (force_cfg=0, force_s=0, Model.fast is false, normal non-diagnostic build and unchanged default extension switches), P.dual=0, P.cfg=90, and P.gt>1. Here pack_gt starts at 64/H and halves until gt*IndexWidth<=2560 and ceil_div(T,gt)>=N; it returns one if gt exceeds T. Packing is selected only when the alternative make_plan with allow_tc=false has splits=1. The default threshold 2.6 admits gt=2 without a probe; larger factors use a different, dual path. Direct execution and execution of its recorded graph use this same selection. Pure capture records work; replay uses recorded host choices, scalars, tensor bindings and tensor maps, while device reads use the current tensor contents.

## t_d84d78313c6e2529 | dispatch.when | 924 chars | 1 declarations

tc_heads:9 meta_data.dispatch.when

Let W be the byte capacity of the workspace actually passed by Model.forward after its identity cache or private scratch selection, and N=dsv4::nsm_cached(). Let P=dsv4::make_plan(H,T,IndexWidth,W,0,0,N,true) in the fixed source, including its compile-time fallback when tcgen05 is not built. P is the initial plan; cfg selects the kernel, gt the tokens per CTA, and dual denotes probe-gated packing. Select exactly when the shipped automatic policy is active (force_cfg=0, force_s=0, Model.fast is false, normal non-diagnostic build and unchanged default extension switches), P.dual=0, P.cfg=90, and P.gt=1. This is the tc_eligible branch: H>=64 and T*ceil_div(H,64)>=N in a tcgen05 build. Direct execution and execution of its recorded graph use this same selection. Pure capture records work; replay uses recorded host choices, scalars, tensor bindings and tensor maps, while device reads use the current tensor contents.

## t_fd8bc943a1f428bb | dispatch.when | 1114 chars | 1 declarations

root:7 meta_data.dispatch.when

Let W be the byte capacity of the actual workspace passed by Model.forward to the extension, including its workspace identity cache or private scratch selection. Let N be the actual value returned by dsv4::nsm_cached(). Let P be dsv4::make_plan(H, T, IndexWidth, W, 0, 0, N, true) in the fixed candidate/dsv4_kernel.cu, including its compile-time fallback when tcgen05 is not built. P is the initial plan, before overlap probing, graph replay, or token-range launches. P.dual denotes probe-gated packing, P.cfg selects the kernel configuration, and P.splits is the number of independent KV splits. Select exactly when the shipped automatic policy is active (force_cfg = 0, force_s = 0, Model.fast is false, normal non-diagnostic build and unchanged default extension switches), P.dual = 0, P.cfg != 90 (TC_CFG_QIS2), and P.splits > 1. This is mma.sync attention followed by a separate FP32 combine. Direct execution and execution of the corresponding recorded graph use this same selection; pure capture only records the plan. For replay, use the recorded plan and bindings, with tensor contents read at execution.

## t_039e3cd562ad5488 | math.arithmetic | 131 chars | 1 declarations

probe_serial_mma:136 design.math_reshape.rewrite[3].arithmetic.accumulate

Scalar multiply/add expressions may contract under fast-math compilation; no bitwise equivalence to the FP32 reference is asserted.

## t_054c925266ced2e4 | math.arithmetic | 173 chars | 3 declarations

probe_hidden_mma:135 design.math_reshape.rewrite[1].arithmetic.approx; probe_hidden_packed:150 design.math_reshape.rewrite[1].arithmetic.approx; probe_serial_packed:149 design.math_reshape.rewrite[1].arithmetic.approx

ProbeRatio division uses the shipped --use_fast_math build; unknown: no source-provided error bound. R follows the explicit FP64 host accumulation and final FP32 conversion.

## t_1f2d1d93c0fd9bd7 | math.arithmetic | 138 chars | 1 declarations

probe_hidden_mma:272 design.math_reshape.rewrite[5].arithmetic.accumulate

Scalar multiply/add may contract under fast-math compilation; the specified reductions do not assert bitwise equivalence to the reference.

## t_256335b092324da7 | math.arithmetic | 143 chars | 2 declarations

mma_direct:95 design.math_reshape.rewrite[1].arithmetic.accumulate; root:235 design.math_reshape.rewrite[1].arithmetic.accumulate

Scalar multiply/add expressions may contract under the shipped fast-math compilation; no bitwise equivalence to the FP32 reference is asserted.

## t_395ded0bd074dd2b | math.arithmetic | 211 chars | 5 declarations

probe_hidden_mma:187 design.math_reshape.rewrite[2].arithmetic.approx; probe_hidden_packed:228 design.math_reshape.rewrite[2].arithmetic.approx; probe_serial_packed:227 design.math_reshape.rewrite[2].arithmetic.approx; tc_heads:99 design.math_reshape.rewrite[1].arithmetic.approx; tc_packed:127 design.math_reshape.rewrite[1].arithmetic.approx

E2 and division use the shipped --use_fast_math build; unknown: no source-provided error bound. The finite sentinel, clamp and maximum lag are retained without a magnitude guarantee for the full Contract domain.

## t_82afdfcba8b5051f | math.arithmetic | 176 chars | 1 declarations

probe_serial_mma:135 design.math_reshape.rewrite[3].arithmetic.approx

E2 and FP32 division use the --use_fast_math build; unknown: no source-provided error bound. The finite maximum sentinel and clamp have no additional input-magnitude guarantee.

## t_c457a00396cc9a94 | math.arithmetic | 92 chars | 1 declarations

probe_serial_mma:85 design.math_reshape.rewrite[1].arithmetic.approx

The device quotient uses the --use_fast_math build; unknown: no source-provided error bound.

## t_cbba0514636641aa | math.arithmetic | 193 chars | 2 declarations

mma_direct:94 design.math_reshape.rewrite[1].arithmetic.approx; root:234 design.math_reshape.rewrite[1].arithmetic.approx

E2 and FP32 division use the --use_fast_math build; unknown: no source-provided error bound. The finite maximum sentinel and clamp are preserved without an additional input-magnitude guarantee.

## t_d1d2abbbd3c54a77 | math.arithmetic | 180 chars | 5 declarations

probe_hidden_mma:190 design.math_reshape.rewrite[2].arithmetic.accumulate; probe_hidden_packed:231 design.math_reshape.rewrite[2].arithmetic.accumulate; probe_serial_packed:229 design.math_reshape.rewrite[2].arithmetic.accumulate; tc_heads:100 design.math_reshape.rewrite[1].arithmetic.accumulate; tc_packed:129 design.math_reshape.rewrite[1].arithmetic.accumulate

Scalar multiply/add expressions may contract under fast-math compilation. The half-latent addition and the specified sum grouping do not imply bitwise equivalence to the reference.

## t_fd5594591d140aef | math.arithmetic | 171 chars | 1 declarations

probe_hidden_mma:270 design.math_reshape.rewrite[5].arithmetic.approx

E2 and FP32 division use the same fast-math build as online; unknown: no source-provided error bound. The finite sentinel and clamp have no additional magnitude guarantee.

## t_0381cddafb86f89c | math.ledger | 136 chars | 2 declarations

mma_direct:113 design.math_reshape.ledger.matmul[0].change; root:272 design.math_reshape.ledger.matmul[0].change

2*H*count(J(t))*D -> (DQ+1)*Hp*C(t)*KC*D MACs per token, including invalid slots, padded heads and repeated QK across output partitions.

## t_06311df5f243ef15 | math.ledger | 6 chars | 29 declarations

mma_direct:111 design.math_reshape.ledger.footprint[0].by[0]; mma_direct:113 design.math_reshape.ledger.matmul[0].by[1]; mma_direct:115 design.math_reshape.ledger.error[0].by[0]; probe_hidden_mma:310 design.math_reshape.ledger.footprint[0].by[1]; probe_hidden_mma:322 design.math_reshape.ledger.matmul[0].by[1]; probe_hidden_mma:327 design.math_reshape.ledger.vector[0].by[0]; probe_hidden_mma:337 design.math_reshape.ledger.error[0].by[1]; probe_hidden_packed:275 design.math_reshape.ledger.footprint[0].by[1]; probe_hidden_packed:283 design.math_reshape.ledger.matmul[0].by[1]; probe_hidden_packed:287 design.math_reshape.ledger.vector[0].by[0]; probe_hidden_packed:297 design.math_reshape.ledger.error[0].by[1]; probe_serial_mma:152 design.math_reshape.ledger.footprint[0].by[0]; probe_serial_mma:154 design.math_reshape.ledger.matmul[0].by[1]; probe_serial_mma:158 design.math_reshape.ledger.error[0].by[0]; probe_serial_packed:271 design.math_reshape.ledger.footprint[0].by[1]; probe_serial_packed:279 design.math_reshape.ledger.matmul[0].by[1]; probe_serial_packed:283 design.math_reshape.ledger.vector[0].by[0]; probe_serial_packed:293 design.math_reshape.ledger.error[0].by[1]; root:269 design.math_reshape.ledger.footprint[0].by[0]; root:272 design.math_reshape.ledger.matmul[0].by[1]; root:276 design.math_reshape.ledger.error[0].by[0]; tc_heads:116 design.math_reshape.ledger.footprint[0].by[0]; tc_heads:118 design.math_reshape.ledger.matmul[0].by[1]; tc_heads:120 design.math_reshape.ledger.vector[0].by[0]; tc_heads:122 design.math_reshape.ledger.error[0].by[0]; tc_packed:163 design.math_reshape.ledger.footprint[0].by[1]; tc_packed:167 design.math_reshape.ledger.matmul[0].by[1]; tc_packed:171 design.math_reshape.ledger.vector[0].by[0]; tc_packed:177 design.math_reshape.ledger.error[0].by[1]

online

## t_063a65b9a799c22b | math.ledger | 118 chars | 3 declarations

probe_hidden_mma:312 design.math_reshape.ledger.footprint[1].change; probe_hidden_packed:277 design.math_reshape.ledger.footprint[1].change; probe_serial_packed:273 design.math_reshape.ledger.footprint[1].change

+4*PS retained control bytes per invocation; the host consumes observations without materializing an attention matrix.

## t_0b300ad3b2dd108c | math.ledger | 32 chars | 2 declarations

mma_direct:113 design.math_reshape.ledger.matmul[0].item; root:272 design.math_reshape.ledger.matmul[0].item

Padded and repeated contractions

## t_0c3601c5fc13bb89 | math.ledger | 211 chars | 1 declarations

probe_serial_mma:158 design.math_reshape.ledger.error[0].change

HB*KC probabilities per chunk narrow FP32 -> BF16 for PV while the denominator retains FP32; base-2 fast exponentials, sentinel clamp, reduction grouping and FP32 scale casts add unbounded-in-source differences.

## t_0d1f749575e41fc7 | math.ledger | 13 chars | 4 declarations

probe_hidden_mma:293 design.math_reshape.ledger.traffic[0].item; probe_hidden_packed:261 design.math_reshape.ledger.traffic[0].item; probe_serial_packed:258 design.math_reshape.ledger.traffic[0].item; tc_packed:154 design.math_reshape.ledger.traffic[0].item

Cache payload

## t_1676c09a43020223 | math.ledger | 203 chars | 3 declarations

probe_hidden_mma:298 design.math_reshape.ledger.traffic[1].change; probe_hidden_packed:266 design.math_reshape.ledger.traffic[1].change; probe_serial_packed:263 design.math_reshape.ledger.traffic[1].change

+4*sum_b Items(b) index bytes for usable samples and 4*PS published bytes per invocation; host polling adds at most 2000000*PS scalar reads and a PS-element final scan. These do not reuse attention data.

## t_16a7d704259601ec | math.ledger | 16 chars | 5 declarations

probe_hidden_mma:327 design.math_reshape.ledger.vector[0].item; probe_hidden_packed:285 design.math_reshape.ledger.vector[0].item; probe_serial_packed:281 design.math_reshape.ledger.vector[0].item; tc_heads:120 design.math_reshape.ledger.vector[0].item; tc_packed:169 design.math_reshape.ledger.vector[0].item

QK partial merge

## t_19af906808997bac | math.ledger | 93 chars | 1 declarations

tc_heads:118 design.math_reshape.ledger.matmul[0].change

2*H*count(J(t))*D -> 2*Hp*C(t)*KC*D MACs per token; the QK split does not duplicate products.

## t_2699bdee77db94eb | math.ledger | 220 chars | 1 declarations

root:276 design.math_reshape.ledger.error[0].change

KC probabilities per padded head per chunk narrow FP32 -> BF16 for PV while the denominator retains FP32; base-2 fast exponentials, sentinel clamp, split grouping and FP32 scale casts add unbounded-in-source differences.

## t_2b3b335f7092d7c5 | math.ledger | 193 chars | 1 declarations

root:274 design.math_reshape.ledger.traffic[0].change

0 -> 4*min(S,C(t))*Hp*D + 8*S*Hp*DQ bytes stored and 4*S*H*(D+2) bytes loaded per token, including discarded vector prefetches and duplicate statistic stores; logical payload, not DRAM traffic.

## t_2b4deee8d0b2c5b2 | math.ledger | 247 chars | 2 declarations

mma_direct:109 design.math_reshape.ledger.baseline; root:267 design.math_reshape.ledger.baseline

Direct FP32 evaluation of the Contract over valid slots, materializing full FP32 logits and probabilities before PV; no split partials. Design-side counts include the padded and repeated work explicitly evaluated by gather and online for every dq.

## t_2e8ff6d1770a6fed | math.ledger | 177 chars | 1 declarations

probe_hidden_mma:339 design.math_reshape.ledger.error[1].change

CM(t)*HB*KM exponentials narrow to BF16 for PV while the denominator retains unrounded FP32 values; KM chunking, grouped sums, fast exponentials and scale casts change rounding.

## t_3177286368eade55 | math.ledger | 268 chars | 1 declarations

probe_serial_mma:156 design.math_reshape.ledger.vector[0].change

Add sum_s Items(s) index positions over usable samples, bounded by 8*2560 per execution, plus eight ratio writes. Host selection adds up to 2000000 scans of at most eight flags during eager submission or graph recording, not graph replay; no attention work is removed.

## t_32bec29c311030e2 | math.ledger | 37 chars | 3 declarations

mma_direct:111 design.math_reshape.ledger.footprint[0].item; probe_serial_mma:152 design.math_reshape.ledger.footprint[0].item; root:269 design.math_reshape.ledger.footprint[0].item

Score and probability materialization

## t_33162ce0941c3c6e | math.ledger | 254 chars | 1 declarations

tc_heads:122 design.math_reshape.ledger.error[0].change

Each chunk narrows HB*KC exponentials to BF16 only for PV; two half-latent sums, one FP32 score addition, grouped normalizer sums, a one-unit base-2 maximum-update threshold and fast exponentials change rounding. No global output error bound is supplied.

## t_33f21f4252f12b41 | math.ledger | 18 chars | 1 declarations

probe_hidden_mma:338 design.math_reshape.ledger.error[1].item

Remainder rounding

## t_39cddda889a3a95e | math.ledger | 230 chars | 1 declarations

mma_direct:115 design.math_reshape.ledger.error[0].change

KC probabilities per padded head per chunk narrow FP32 -> BF16 for PV while the denominator retains FP32; base-2 fast exponentials, sentinel clamp, chunk reduction grouping and FP32 scale casts add unbounded-in-source differences.

## t_3b3417b953503daf | math.ledger | 6 chars | 20 declarations

mma_direct:113 design.math_reshape.ledger.matmul[0].by[0]; probe_hidden_mma:296 design.math_reshape.ledger.traffic[0].by[0]; probe_hidden_mma:310 design.math_reshape.ledger.footprint[0].by[0]; probe_hidden_mma:322 design.math_reshape.ledger.matmul[0].by[0]; probe_hidden_mma:337 design.math_reshape.ledger.error[0].by[0]; probe_hidden_packed:264 design.math_reshape.ledger.traffic[0].by[0]; probe_hidden_packed:275 design.math_reshape.ledger.footprint[0].by[0]; probe_hidden_packed:283 design.math_reshape.ledger.matmul[0].by[0]; probe_hidden_packed:297 design.math_reshape.ledger.error[0].by[0]; probe_serial_mma:154 design.math_reshape.ledger.matmul[0].by[0]; probe_serial_packed:261 design.math_reshape.ledger.traffic[0].by[0]; probe_serial_packed:271 design.math_reshape.ledger.footprint[0].by[0]; probe_serial_packed:279 design.math_reshape.ledger.matmul[0].by[0]; probe_serial_packed:293 design.math_reshape.ledger.error[0].by[0]; root:272 design.math_reshape.ledger.matmul[0].by[0]; tc_heads:118 design.math_reshape.ledger.matmul[0].by[0]; tc_packed:157 design.math_reshape.ledger.traffic[0].by[0]; tc_packed:163 design.math_reshape.ledger.footprint[0].by[0]; tc_packed:167 design.math_reshape.ledger.matmul[0].by[0]; tc_packed:177 design.math_reshape.ledger.error[0].by[0]

gather

## t_3cae718cae614f73 | math.ledger | 11 chars | 1 declarations

root:270 design.math_reshape.ledger.footprint[1].item

Split state

## t_3d5159f9de5fac2b | math.ledger | 13 chars | 1 declarations

root:274 design.math_reshape.ledger.traffic[0].item

Split handoff

## t_3d6e5e3eae079504 | math.ledger | 186 chars | 1 declarations

probe_serial_mma:152 design.math_reshape.ledger.footprint[0].change

8*H*count(J(t)) bytes per token -> no full-row materialization; each active attention worktile holds HB*KC FP32 scores, HB*KC BF16 probabilities and 4*HB*(D+2) bytes of recurrence state.

## t_42c0d4ff57b64398 | math.ledger | 177 chars | 2 declarations

mma_direct:111 design.math_reshape.ledger.footprint[0].change; root:269 design.math_reshape.ledger.footprint[0].change

8*H*count(J(t)) bytes per token -> no full-row materialization; per active worktile HB*KC FP32 scores and HB*KC BF16 probabilities, plus 4*HB*(D/DQ+2) bytes of recurrence state.

## t_466998addc13f4be | math.ledger | 23 chars | 1 declarations

probe_hidden_mma:315 design.math_reshape.ledger.footprint[2].item

Remainder intermediates

## t_4debd329bdd01bcb | math.ledger | 370 chars | 1 declarations

probe_hidden_mma:288 design.math_reshape.ledger.baseline

Direct FP32 evaluation per query over valid slots, gathering each BF16 latent once for both QK and PV and materializing full logits and probabilities. Counts below include every padded product actually evaluated. Packed group counts below apply only to grp in [0,K); remainder counts apply per token t in [K*GT,T). Probe costs cover the full sampled batch independently.

## t_5187ac0a76de1fe0 | math.ledger | 96 chars | 1 declarations

probe_hidden_mma:324 design.math_reshape.ledger.matmul[1].change

2*H*count(J(t))*D -> 2*HB*CM(t)*KM*D MACs per remainder token, including padded heads and slots.

## t_56a175c1c2f34773 | math.ledger | 39 chars | 4 declarations

probe_hidden_mma:327 design.math_reshape.ledger.vector[0].change; probe_hidden_packed:286 design.math_reshape.ledger.vector[0].change; probe_serial_packed:282 design.math_reshape.ledger.vector[0].change; tc_packed:170 design.math_reshape.ledger.vector[0].change

+MR*C(grp)*KC FP32 additions per group.

## t_58ed309a70c950de | math.ledger | 10 chars | 2 declarations

probe_hidden_mma:304 design.math_reshape.ledger.traffic[2].by[0]; probe_hidden_mma:325 design.math_reshape.ledger.matmul[1].by[0]

mma_gather

## t_59bff04cfcf09547 | math.ledger | 13 chars | 3 declarations

probe_hidden_mma:311 design.math_reshape.ledger.footprint[1].item; probe_hidden_packed:276 design.math_reshape.ledger.footprint[1].item; probe_serial_packed:272 design.math_reshape.ledger.footprint[1].item

Probe results

## t_5a93183344f2b508 | math.ledger | 269 chars | 4 declarations

probe_hidden_mma:334 design.math_reshape.ledger.error[0].change; probe_hidden_packed:294 design.math_reshape.ledger.error[0].change; probe_serial_packed:290 design.math_reshape.ledger.error[0].change; tc_packed:174 design.math_reshape.ledger.error[0].change

The realized union permutation reorders reductions. Each chunk narrows MR*KC exponentials to BF16 only for PV; split QK, grouped normalizer sums, a one-unit base-2 maximum-update threshold and fast exponentials change rounding. No global output error bound is supplied.

## t_5b18c3ba6e9d9e9b | math.ledger | 8 chars | 10 declarations

mma_direct:115 design.math_reshape.ledger.error[0].by[1]; probe_hidden_mma:337 design.math_reshape.ledger.error[0].by[2]; probe_hidden_packed:297 design.math_reshape.ledger.error[0].by[2]; probe_serial_mma:158 design.math_reshape.ledger.error[0].by[1]; probe_serial_packed:293 design.math_reshape.ledger.error[0].by[2]; root:270 design.math_reshape.ledger.footprint[1].by[0]; root:274 design.math_reshape.ledger.traffic[0].by[0]; root:276 design.math_reshape.ledger.error[0].by[1]; tc_heads:122 design.math_reshape.ledger.error[0].by[1]; tc_packed:177 design.math_reshape.ledger.error[0].by[2]

finalize

## t_5b86fe1a0799d67d | math.ledger | 180 chars | 1 declarations

probe_hidden_mma:316 design.math_reshape.ledger.footprint[2].change

8*H*count(J(t)) full-row logit/probability bytes -> no full-row materialization; HB*KM FP32 scores, HB*KM BF16 probabilities and 4*HB*(D+2) recurrence bytes per remainder worktile.

## t_5e4cbc4d5a84c104 | math.ledger | 19 chars | 5 declarations

probe_hidden_mma:306 design.math_reshape.ledger.footprint[0].item; probe_hidden_packed:271 design.math_reshape.ledger.footprint[0].item; probe_serial_packed:267 design.math_reshape.ledger.footprint[0].item; tc_heads:116 design.math_reshape.ledger.footprint[0].item; tc_packed:159 design.math_reshape.ledger.footprint[0].item

Intermediate values

## t_5f7b1e096787dc92 | math.ledger | 11 chars | 1 declarations

probe_serial_mma:156 design.math_reshape.ledger.vector[0].by[0]

probe_count

## t_622dcf4a1623f410 | math.ledger | 18 chars | 3 declarations

probe_hidden_mma:328 design.math_reshape.ledger.vector[1].item; probe_hidden_packed:288 design.math_reshape.ledger.vector[1].item; probe_serial_packed:284 design.math_reshape.ledger.vector[1].item

Probe update count

## t_67175161edad0507 | math.ledger | 198 chars | 1 declarations

tc_heads:114 design.math_reshape.ledger.baseline

Direct FP32 evaluation over valid slots, materializing full logits and probabilities before PV, with no split partials. Design counts include the padded and invalid slots explicitly evaluated above.

## t_72266bb05f8b64e0 | math.ledger | 37 chars | 1 declarations

tc_heads:120 design.math_reshape.ledger.vector[0].change

+Hp*C(t)*KC FP32 additions per token.

## t_7e4a708495c659f1 | math.ledger | 12 chars | 1 declarations

probe_serial_mma:156 design.math_reshape.ledger.vector[0].by[1]

probe_select

## t_81d2ab2138991e56 | math.ledger | 19 chars | 4 declarations

probe_hidden_mma:320 design.math_reshape.ledger.matmul[0].item; probe_hidden_packed:281 design.math_reshape.ledger.matmul[0].item; probe_serial_packed:277 design.math_reshape.ledger.matmul[0].item; tc_packed:165 design.math_reshape.ledger.matmul[0].item

Packed contractions

## t_81ded792f9ad600a | math.ledger | 22 chars | 1 declarations

probe_serial_mma:156 design.math_reshape.ledger.vector[0].item

Rejected overlap probe

## t_8485e3ae4c58daf2 | math.ledger | 5 chars | 9 declarations

probe_hidden_mma:301 design.math_reshape.ledger.traffic[1].by[0]; probe_hidden_mma:314 design.math_reshape.ledger.footprint[1].by[0]; probe_hidden_mma:331 design.math_reshape.ledger.vector[1].by[0]; probe_hidden_packed:269 design.math_reshape.ledger.traffic[1].by[0]; probe_hidden_packed:279 design.math_reshape.ledger.footprint[1].by[0]; probe_hidden_packed:291 design.math_reshape.ledger.vector[1].by[0]; probe_serial_packed:265 design.math_reshape.ledger.traffic[1].by[0]; probe_serial_packed:275 design.math_reshape.ledger.footprint[1].by[0]; probe_serial_packed:287 design.math_reshape.ledger.vector[1].by[0]

probe

## t_858592c9bbeb8803 | math.ledger | 105 chars | 1 declarations

probe_hidden_mma:303 design.math_reshape.ledger.traffic[2].change

2*D*count(J(t)) -> 2*D*CM(t)*KM bytes per remainder token; invalid and padded slots still fetch row zero.

## t_85c8c41d6be6c7f5 | math.ledger | 167 chars | 3 declarations

probe_hidden_mma:329 design.math_reshape.ledger.vector[1].change; probe_hidden_packed:289 design.math_reshape.ledger.vector[1].change; probe_serial_packed:285 design.math_reshape.ledger.vector[1].change

+sum_b Items(b) selected-index inspections over usable samples per invocation, with runtime-dependent hash-collision work and at most PS*GT*IndexWidth inspected slots.

## t_8a47afaaf8f6840a | math.ledger | 244 chars | 4 declarations

probe_hidden_mma:307 design.math_reshape.ledger.footprint[0].change; probe_hidden_packed:272 design.math_reshape.ledger.footprint[0].change; probe_serial_packed:268 design.math_reshape.ledger.footprint[0].change; tc_packed:160 design.math_reshape.ledger.footprint[0].change

8*H*sum_g count(J(t_g)) full-row logit/probability bytes -> 5*Nu(grp) retained union bytes and no full attention matrices; chunk operands include 2*MR*KC FP32 QK partials and MR*KC BF16 probabilities, with 4*MR*(D+2) recurrence bytes per group.

## t_8f1b8fa7b59115a1 | math.ledger | 22 chars | 1 declarations

probe_hidden_mma:323 design.math_reshape.ledger.matmul[1].item

Remainder contractions

## t_9118dd5f2651de57 | math.ledger | 153 chars | 4 declarations

probe_hidden_mma:294 design.math_reshape.ledger.traffic[0].change; probe_hidden_packed:262 design.math_reshape.ledger.traffic[0].change; probe_serial_packed:259 design.math_reshape.ledger.traffic[0].change; tc_packed:155 design.math_reshape.ledger.traffic[0].change

2*D*sum_g count(J(t_g)) -> 2*D*C(grp)*KC bytes per group, for the gather shared by QK and PV; sharing saves rows only when overlap exceeds chunk padding.

## t_9d971dbb12f8b924 | math.ledger | 92 chars | 4 declarations

probe_hidden_mma:321 design.math_reshape.ledger.matmul[0].change; probe_hidden_packed:282 design.math_reshape.ledger.matmul[0].change; probe_serial_packed:278 design.math_reshape.ledger.matmul[0].change; tc_packed:166 design.math_reshape.ledger.matmul[0].change

2*H*sum_g count(J(t_g))*D -> 2*MR*C(grp)*KC*D MACs per group; C(grp)=0 executes no products.

## t_a073dbdb6ed1145d | math.ledger | 5 chars | 3 declarations

root:270 design.math_reshape.ledger.footprint[1].by[1]; root:274 design.math_reshape.ledger.traffic[0].by[1]; root:276 design.math_reshape.ledger.error[0].by[2]

merge

## t_a9bb291fa330addd | math.ledger | 17 chars | 8 declarations

mma_direct:115 design.math_reshape.ledger.error[0].item; probe_hidden_mma:333 design.math_reshape.ledger.error[0].item; probe_hidden_packed:293 design.math_reshape.ledger.error[0].item; probe_serial_mma:158 design.math_reshape.ledger.error[0].item; probe_serial_packed:289 design.math_reshape.ledger.error[0].item; root:276 design.math_reshape.ledger.error[0].item; tc_heads:122 design.math_reshape.ledger.error[0].item; tc_packed:173 design.math_reshape.ledger.error[0].item

Numerical changes

## t_aa0473e982331e23 | math.ledger | 21 chars | 3 declarations

probe_hidden_mma:297 design.math_reshape.ledger.traffic[1].item; probe_hidden_packed:265 design.math_reshape.ledger.traffic[1].item; probe_serial_packed:262 design.math_reshape.ledger.traffic[1].item

Probe control payload

## t_afac367b0d53c314 | math.ledger | 207 chars | 1 declarations

probe_serial_mma:150 design.math_reshape.ledger.baseline

Direct FP32 evaluation of the Contract over valid slots, materializing full FP32 logits and probabilities before PV, without a probe. Counts include the padded work explicitly evaluated by gather and online.

## t_ba957b6197076c86 | math.ledger | 23 chars | 1 declarations

probe_hidden_mma:302 design.math_reshape.ledger.traffic[2].item

Remainder cache payload

## t_bb27bfeb0f206974 | math.ledger | 10 chars | 3 declarations

probe_hidden_mma:318 design.math_reshape.ledger.footprint[2].by[0]; probe_hidden_mma:325 design.math_reshape.ledger.matmul[1].by[1]; probe_hidden_mma:341 design.math_reshape.ledger.error[1].by[0]

mma_online

## t_c0a5ea111f15688d | math.ledger | 212 chars | 3 declarations

probe_hidden_packed:257 design.math_reshape.ledger.baseline; probe_serial_packed:255 design.math_reshape.ledger.baseline; tc_packed:151 design.math_reshape.ledger.baseline

Direct FP32 evaluation per query over valid slots, gathering each BF16 latent once for both QK and PV and materializing full logits and probabilities. Counts below include every padded product actually evaluated.

## t_c4c38f4f06c95918 | math.ledger | 85 chars | 1 declarations

root:270 design.math_reshape.ledger.footprint[1].change

0 -> 4*S*Hp*(D+2) bytes per token; empty split vectors are allocated but not written.

## t_c9e9082bf60efbb3 | math.ledger | 93 chars | 1 declarations

probe_serial_mma:154 design.math_reshape.ledger.matmul[0].change

2*H*count(J(t))*D -> 2*HB*C(t)*KC*D MACs per token, including invalid slots and padded heads.

## t_cbed84057bf46127 | math.ledger | 19 chars | 2 declarations

probe_serial_mma:154 design.math_reshape.ledger.matmul[0].item; tc_heads:118 design.math_reshape.ledger.matmul[0].item

Padded contractions

## t_db118e707268e29c | math.ledger | 12 chars | 1 declarations

probe_hidden_mma:341 design.math_reshape.ledger.error[1].by[1]

mma_finalize

## t_df658b802e9c62c7 | math.ledger | 219 chars | 1 declarations

tc_heads:116 design.math_reshape.ledger.footprint[0].change

8*H*count(J(t)) bytes per token -> no full-row score or probability materialization; each chunk uses 2*HB*KC FP32 QK partials and HB*KC BF16 probabilities, with 4*HB*(D+2) bytes of logical recurrence state per worktile.

## t_14bc85462608409b | math.math | 733 chars | 1 declarations

probe_hidden_mma:194 design.math_reshape.rewrite[3].math

For each prefix grp in [0,K): If C(grp)=0, overwrite out[t0+floor(r/H),r mod H,d]=bf16(0) for r in [0,nrows), d in [0,D), without evaluating online or final normalization. Otherwise, last=C(grp)-1 and for r in [0,MR): z[r]=l(last)[r]+E2(sink2[r]-m(last)[r]); inv[r]=a2/z[r] if r<nrows and z[r]>0, otherwise 0. For r<nrows and d in [0,D), Result[r,d]=bf16(Acc(last)[r,d]*inv[r]); overwrite out[t0+floor(r/H),r mod H,d]=Result[r,d]. No Result value is formed for padded rows. All matrix products and online updates still include those rows. A query with J(t_g) empty inside a nonempty union follows online and normalization rather than the empty-union shortcut. E2, division and scalar contraction follow online's arithmetic exception.

## t_18e722beea657cf4 | math.math | 703 chars | 3 declarations

probe_hidden_packed:235 design.math_reshape.rewrite[3].math; probe_serial_packed:233 design.math_reshape.rewrite[3].math; tc_packed:133 design.math_reshape.rewrite[2].math

If C(grp)=0, overwrite out[t0+floor(r/H),r mod H,d]=bf16(0) for r in [0,nrows), d in [0,D), without evaluating online or final normalization. Otherwise, last=C(grp)-1 and for r in [0,MR): z[r]=l(last)[r]+E2(sink2[r]-m(last)[r]); inv[r]=a2/z[r] if r<nrows and z[r]>0, otherwise 0. For r<nrows and d in [0,D), Result[r,d]=bf16(Acc(last)[r,d]*inv[r]); overwrite out[t0+floor(r/H),r mod H,d]=Result[r,d]. No Result value is formed for padded rows. All matrix products and online updates still include those rows. A query with J(t_g) empty inside a nonempty union follows online and normalization rather than the empty-union shortcut. E2, division and scalar contraction follow online's arithmetic exception.

## t_2604f1438586c429 | math.math | 2188 chars | 1 declarations

probe_hidden_mma:49 design.math_reshape.rewrite[0].math

GT, MR, KC and K are bound below. For group grp in [0,ceil_div(T,GT)), let t0=grp*GT, G=min(GT,T-t0), nrows=G*H, and t_g=t0+g for g in [0,G). For each real token use the Contract's request lookup and visible length: A(t)=min(max(visible,0),SW); Klen(t)=A(t)+max(min(sparse_topk_lens[t],IndexWidth)-SW,0). For each distinct (pool,id) among the group's valid slots J(t_g), let m_g(pool,id) count its occurrences in query g. Pools remain distinct even when their pointers or row numbers coincide; pool 0 is SWA, pool 1 compressed. Define a union with one shared entry carrying all bits g with m_g>0, then m_g-1 private entries carrying only bit g for each g with m_g>1. For prefix groups grp in [0,K), materialize each entry as UnionKey[k]=int32(pool*16777216+id) and UnionMask[k]=uint8(sum_g bit_g*2^g). Nu denotes the attention union count; evaluate it only for prefix groups. Only prefix forms the union arrays, Q, Mask and Gather defined here. Let Ns and Ng be the entry counts for pools 0 and 1, Nu(grp)=Ns+Ng, and C(grp)=ceil_div(Nu(grp),KC). Nu(grp)<=sum_g count(J(t_g))<=GT*IndexWidth. Entries are ordered with all SWA entries first and all compressed entries second. Within each pool preserve the runtime union-construction permutation; no stable order across executions is assumed. For each (pool,id) its number of entries is 1+sum_g max(m_g(pool,id)-1,0), not max_g m_g(pool,id). For r in [0,MR), Q[r,:]=query[t0+floor(r/H),r mod H,:] when r<nrows, otherwise bf16(0). For c in [0,C(grp)) and u in [0,KC), let k=c*KC+u. Mask(c)[u]=UnionMask[k] if k<Nu(grp), otherwise uint8(0). Define pool(k)=floor(UnionKey[k]/16777216) for k<Nu(grp), otherwise 0; id(k)=UnionKey[k] mod 16777216 for k<Nu(grp), otherwise 0. Let mixed(c) mean 0<Ns-c*KC<KC and (Ns-c*KC) mod 4 != 0. The gathered pool is pool(k) if mixed(c), otherwise pool(c*KC+4*floor(u/4)); Gather(c)[u,:] is row id(k) of that pool. Thus padding still fetches row zero; a padded tail sharing a four-row group with compressed entries uses compressed row zero, while wholly padded groups and mixed-chunk padding use SWA row zero. a1=fp32(bmm1_scale); a2=fp32(bmm2_scale); L2=fp32(1.4426950408889634); neg=fp32(-3.0e38); scale=a1*L2.

## t_328376e8863e3f0f | math.math | 1781 chars | 2 declarations

probe_hidden_packed:154 design.math_reshape.rewrite[2].math; probe_serial_packed:153 design.math_reshape.rewrite[2].math

E2(x) is the compiled fast exp2f(x), with E2(-inf)=0; maxnum is FP32 fmaxf, where a single NaN loses to the numeric operand. For each grp with C(grp)>0, initialize sink2[r]=sinks[r mod H]*L2 for r<nrows and neg for padded rows; m(-1)[r]=sink2[r]. l(-1)[r]=0 and Acc(-1)[r,:]=0. For c=0..C(grp)-1: Partial(c,a)=mm32(Q[:,a*D/2:(a+1)*D/2], Gather(c)[:,a*D/2:(a+1)*D/2]^T), a in {0,1}. Each partial accumulates D/32 successive 16-wide products from zero. Z=Partial(c,0)+Partial(c,1), with one FP32 addition per score. Scores(c)[r,u]=Z[r,u]*scale if floor(Mask(c)[u]/2^floor(r/H)) mod 2 != 0, otherwise -inf. For b in {0,1} and q in {0,1,2,3}, reduce Scores[r,b*32+4*n+q] for n=0..7 by successive maxnum from -inf, obtaining M[b,q]. Mhalf[b]=maxnum(maxnum(M[b,0],M[b,1]),maxnum(M[b,2],M[b,3])); candidate=maxnum(m(c-1)[r],maxnum(Mhalf[0],Mhalf[1])); clamped=candidate if candidate>neg*0.5, otherwise neg; m(c)[r]=m(c-1)[r] if clamped-m(c-1)[r]<=1, otherwise clamped. alpha(c)[r]=E2(m(c-1)[r]-m(c)[r]); E[r,u]=E2(Scores(c)[r,u]-m(c)[r]); Prob(c)[r,u]=bf16(E[r,u]). For each b,q form SumPart[b,q] by adding E[r,b*32+4*n+q] from zero for n=0..7; Rhalf[b]=(SumPart[b,0]+SumPart[b,1])+(SumPart[b,2]+SumPart[b,3]); l(c)[r]=l(c-1)[r]*alpha(c)[r]+(Rhalf[0]+Rhalf[1]). Acc(0)=mm32(Prob(0),Gather(0)); Acc(c)=mma32(Prob(c),Gather(c),alpha(c)[:,None] ⊙ Acc(c-1)) for c>0. Each PV output half traverses KC/16 successive 16-wide products. Partition the rows into consecutive groups of RG rows, with RG bound below. Rescaling is omitted for a row group when every alpha is exactly 1; otherwise all its rows are multiplied, including those with alpha=1. The two key halves evaluate duplicate m,l updates; the lower half's l supplies the final denominator. Unrounded E supplies l, and Prob supplies PV.

## t_3a69a7007fcef504 | math.math | 1702 chars | 1 declarations

probe_hidden_packed:104 design.math_reshape.rewrite[1].math

PS is bound below and NG=ceil_div(T,GT). For sample b in [0,PS), if b>=NG set ProbeRatio[b]=fp32(-1). Otherwise let sg(b)=b if NG<=PS, else floor(b*NG/PS), st(b)=sg(b)*GT, pg(b)=min(GT,T-st(b)), and Items(b)=sum_{g=0}^{pg(b)-1} Klen(st(b)+g). A sample is usable when pg(b)=GT, all its tokens belong to the same Contract request, and 0<Items(b)<=2560. If unusable set ProbeRatio[b]=fp32(-1). For a usable sample, enumerate occurrences x=(g,j) with g in [0,GT) and j in J(st(b)+g). Let g(x)=g and e(x)=uint32(sparse_indices[st(b)+g,j]) bitwise-OR (0 if j<SW, otherwise 16777216). For each encoded key e, Creator(e) is its first entry-creating occurrence; for each (e,g), First(e,g) is its first membership-marking occurrence in the actual concurrent update order. These two orders need not agree. ProbeCount(b)=sum_x (1 if x=Creator(e(x)) else (0 if x=First(e(x),g(x)) else 1)). Invalid IDs are absent from J, but Items includes their selected slots. ProbeRatio[b]=(fp32(ProbeCount(b))*fp32(GT))/fp32(Items(b)). Only the count is retained locally; no attention union or KV is produced. Before this computation, the mapped sample slots contain fp32(-2). For each b let observed[b] be the actual FP32 value read by the host's final scan after its bounded poll; the scan need not observe all final sample values. Starting sum64=fp64(0), visit b=0..PS-1 and add fp64(observed[b]) in FP64 only when observed[b]>0. Let npos count these additions. R=fp32(sum64/fp64(npos)) if npos>0, otherwise fp32(-1). R selects only the remainder implementation; the prefix uses packing independently of R. This node accepts the same packing for the remainder. No probe value enters an attention score, probability or output.

## t_445dac2ab3596f27 | math.math | 993 chars | 1 declarations

root:177 design.math_reshape.rewrite[0].math

HB, KC, S, DQ and U are bound below; NB = ceil_div(H,HB), Hp = NB*HB. For token t use the Contract's request lookup and visible length. A(t) = min(max(visible,0),SW); Klen(t) = A(t) + max(min(sparse_topk_lens[t],IndexWidth)-SW,0). C(t) = ceil_div(Klen(t),KC). For split s in [0,S), c0(t,s) = s*floor(C(t)/S) + min(s,C(t) mod S), c1(t,s) = c0(t,s) + floor(C(t)/S) + (1 if s < C(t) mod S else 0). For chunk c and u in [0,KC), let k=c*KC+u and j=k if k<A(t), else SW+k-A(t). id = sparse_indices[t,j] if k<Klen(t), otherwise -1; valid(c)[u] = uint8(1 if id>=0 else 0). Gather(c)[u,:] is row max(id,0) of swa_kv_cache if k<A(t), otherwise compressed_kv_cache. Invalid and padded positions therefore still gather row zero, not a zero latent. For head block hb and r in [0,HB), Q[r,:]=query[t,hb*HB+r,:] for hb*HB+r<H, otherwise Q[r,:]=bf16(0). Both matrix products include these padded head rows. a1=fp32(bmm1_scale); a2=fp32(bmm2_scale); L2=fp32(1.4426950408889634); neg=fp32(-3.0e38); scale=a1*L2.

## t_4713460eec7c9865 | math.math | 828 chars | 1 declarations

tc_heads:32 design.math_reshape.rewrite[0].math

HB and KC are bound below; NB=ceil_div(H,HB), Hp=NB*HB. For token t use the Contract's request lookup and visible length. A(t)=min(max(visible,0),SW); Klen(t)=A(t)+max(min(sparse_topk_lens[t],IndexWidth)-SW,0); C(t)=ceil_div(Klen(t),KC). Klen(t)>=1 in this node's Contract domain. For c in [0,C(t)) and u in [0,KC), let k=c*KC+u and j=k if k<A(t), otherwise SW+k-A(t). id=sparse_indices[t,j] if k<Klen(t), otherwise -1; valid(c)[u]=uint8(1 if id>=0 else 0). Gather(c)[u,:] is row max(id,0) of swa_kv_cache when k<A(t), otherwise compressed_kv_cache. Invalid and padded slots still read row zero. For hb in [0,NB), Q[r,:]=query[t,hb*HB+r,:], r in [0,HB). H is a multiple of HB here; slots retain their original multiplicities. a1=fp32(bmm1_scale); a2=fp32(bmm2_scale); L2=fp32(1.4426950408889634); neg=fp32(-3.0e38); scale=a1*L2.

## t_47f2cd7fd637e8b0 | math.math | 325 chars | 1 declarations

root:238 design.math_reshape.rewrite[2].math

For each (t,hb,s,dq) with c0<c1, let last=c1-c0-1 and retain PartO[s,hb*HB+r,d0:d0+D/DQ]=Acc(last)[r,:], PartM[s,hb*HB+r]=m(last)[r], PartL[s,hb*HB+r]=l(last)[r], including padded heads. If c0=c1, write only PartM=neg and PartL=0; PartO keeps its prior scratch bits. All DQ partitions produce the same PartM and PartL values.

## t_4f943725affe8113 | math.math | 1254 chars | 1 declarations

root:203 design.math_reshape.rewrite[1].math

Define E2(x) as the compiled fast exp2f(x), with E2(-inf)=0. For each (t,hb,s,dq), dq in [0,DQ), let d0=dq*D/DQ; within this instance abbreviate c0(t,s),c1(t,s) as c0,c1. For c0<c1 initialize m(-1)[r]=sink2[r]; sink2[r]=sinks[hb*HB+r]*L2 for a real head, neg for a padded head. Initialize l(-1)[r]=0 and Acc(-1)[r,:]=0. At local iteration i=0..c1-c0-1, c=c0+i: Z=mm32(Q,Gather(c)^T), accumulating the D axis as 32 successive 16-wide products. Scores(c)[r,u]=Z[r,u]*scale; then replace Scores(c)[r,u] by -inf where valid(c)[u]=0. zmax=maxnum(m(i-1)[r],maxnum_u Scores(c)[r,u]); m(i)[r]=zmax if zmax>neg*0.5, otherwise neg. alpha[r]=E2(m(i-1)[r]-m(i)[r]); E[r,u]=E2(Scores(c)[r,u]-m(i)[r]); Prob(c)[r,u]=bf16(E[r,u]). maxnum is FP32 fmaxf (a single NaN loses to the numeric operand). The sum of E is evaluated in KC/U consecutive groups of U columns. In each group v, form four sums R[a] (a=0..3), each starting at zero and successively adding E[r,v*U+8*n+2*a]+E[r,v*U+8*n+2*a+1] for n=0..U/8-1. Starting rs at zero, add each group's ((R[0]+R[1])+(R[2]+R[3])) in ascending v order. l(i)[r]=l(i-1)[r]*alpha[r]+rs; Acc(i)=mma32(Prob(c),Gather(c)[:,d0:d0+D/DQ],alpha[:,None] ⊙ Acc(i-1)). The PV contraction advances through KC/16 consecutive 16-wide products.

## t_51015578893384a3 | math.math | 1169 chars | 1 declarations

mma_direct:64 design.math_reshape.rewrite[1].math

Define E2(x) as the compiled fast exp2f(x), with E2(-inf)=0. For each (t,hb,dq), dq in [0,DQ), let d0=dq*D/DQ. Initialize m(-1)[r]=sink2[r]; sink2[r]=sinks[hb*HB+r]*L2 for a real head, neg for a padded head. Initialize l(-1)[r]=0 and Acc(-1)[r,:]=0. At iteration c=0..C(t)-1: Z=mm32(Q,Gather(c)^T), accumulating the D axis as 32 successive 16-wide products. Scores(c)[r,u]=Z[r,u]*scale; then replace Scores(c)[r,u] by -inf where valid(c)[u]=0. zmax=maxnum(m(c-1)[r],maxnum_u Scores(c)[r,u]); m(c)[r]=zmax if zmax>neg*0.5, otherwise neg. alpha[r]=E2(m(c-1)[r]-m(c)[r]); E[r,u]=E2(Scores(c)[r,u]-m(c)[r]); Prob(c)[r,u]=bf16(E[r,u]). maxnum is FP32 fmaxf (a single NaN loses to the numeric operand). The sum of E is evaluated in KC/U consecutive groups of U columns. In each group v, form four sums R[a] (a=0..3), each starting at zero and successively adding E[r,v*U+8*n+2*a]+E[r,v*U+8*n+2*a+1] for n=0..U/8-1. Starting rs at zero, add each group's ((R[0]+R[1])+(R[2]+R[3])) in ascending v order. l(c)[r]=l(c-1)[r]*alpha[r]+rs; Acc(c)=mma32(Prob(c),Gather(c)[:,d0:d0+D/DQ],alpha[:,None] ⊙ Acc(c-1)). The PV contraction advances through KC/16 consecutive 16-wide products.

## t_58f8d83e02abf95d | math.math | 644 chars | 1 declarations

probe_hidden_mma:214 design.math_reshape.rewrite[4].math

HB and KM are bound below. For each t in [K*GT,T), CM(t)=ceil_div(Klen(t),KM). A(t), Klen(t), a1, a2, L2, neg and scale are defined by gather; CM(t)>=1 even when every consumed ID is invalid. For r in [0,HB), Qm[r,:]=query[t,r,:] if r<H, otherwise bf16(0). For c in [0,CM(t)), u in [0,KM), let k=c*KM+u, j=k if k<A(t), otherwise SW+k-A(t). id=sparse_indices[t,j] if k<Klen(t), otherwise -1; ValidM(c)[u]=uint8(1 if id>=0 else 0). GatherM(c)[u,:] is row max(id,0) of swa_kv_cache if k<A(t), otherwise compressed_kv_cache. Invalid and padded slots still gather row zero. No union is formed for these tokens and repeated IDs remain separate slots.

## t_6b01d666cfa1338a | math.math | 428 chars | 1 declarations

probe_hidden_mma:276 design.math_reshape.rewrite[6].math

For each remainder token t, last=CM(t)-1. For r in [0,HB), zM[r]=lM(last)[r]+E2(SinkM[r]-mM(last)[r]); invM[r]=a2/zM[r] if r<H and zM[r]>0, otherwise 0. ResultM[r,d]=bf16(AccM(last)[r,d]*invM[r]) for d in [0,D), including padded heads. Overwrite out[t,r,d]=ResultM[r,d] only for r<H. Evaluate the recurrence and normalization even when J(t) is empty. E2, division and scalar contraction follow mma_online's arithmetic exception.

## t_71a96eb88c1e120b | math.math | 1751 chars | 1 declarations

tc_packed:80 design.math_reshape.rewrite[1].math

E2(x) is the compiled fast exp2f(x), with E2(-inf)=0; maxnum is FP32 fmaxf, where a single NaN loses to the numeric operand. For each grp with C(grp)>0, initialize sink2[r]=sinks[r mod H]*L2 for r<nrows and neg for padded rows; m(-1)[r]=sink2[r]. l(-1)[r]=0 and Acc(-1)[r,:]=0. For c=0..C(grp)-1: Partial(c,a)=mm32(Q[:,a*D/2:(a+1)*D/2], Gather(c)[:,a*D/2:(a+1)*D/2]^T), a in {0,1}. Each partial accumulates D/32 successive 16-wide products from zero. Z=Partial(c,0)+Partial(c,1), with one FP32 addition per score. Scores(c)[r,u]=Z[r,u]*scale if floor(Mask(c)[u]/2^floor(r/H)) mod 2 != 0, otherwise -inf. For b in {0,1} and q in {0,1,2,3}, reduce Scores[r,b*32+4*n+q] for n=0..7 by successive maxnum from -inf, obtaining M[b,q]. Mhalf[b]=maxnum(maxnum(M[b,0],M[b,1]),maxnum(M[b,2],M[b,3])); candidate=maxnum(m(c-1)[r],maxnum(Mhalf[0],Mhalf[1])); clamped=candidate if candidate>neg*0.5, otherwise neg; m(c)[r]=m(c-1)[r] if clamped-m(c-1)[r]<=1, otherwise clamped. alpha(c)[r]=E2(m(c-1)[r]-m(c)[r]); E[r,u]=E2(Scores(c)[r,u]-m(c)[r]); Prob(c)[r,u]=bf16(E[r,u]). For each b,q form R[b,q] by adding E[r,b*32+4*n+q] from zero for n=0..7; Rhalf[b]=(R[b,0]+R[b,1])+(R[b,2]+R[b,3]); l(c)[r]=l(c-1)[r]*alpha(c)[r]+(Rhalf[0]+Rhalf[1]). Acc(0)=mm32(Prob(0),Gather(0)); Acc(c)=mma32(Prob(c),Gather(c),alpha(c)[:,None] ⊙ Acc(c-1)) for c>0. Each PV output half traverses KC/16 successive 16-wide products. Partition the rows into consecutive groups of RG rows, with RG bound below. Rescaling is omitted for a row group when every alpha is exactly 1; otherwise all its rows are multiplied, including those with alpha=1. The two key halves evaluate duplicate m,l updates; the lower half's l supplies the final denominator. Unrounded E supplies l, and Prob supplies PV.

## t_8ba3f9308e4a58a1 | math.math | 1104 chars | 1 declarations

probe_serial_mma:105 design.math_reshape.rewrite[3].math

Define E2(x) as the compiled fast exp2f(x), with E2(-inf)=0. Independently for each token t, initialize m(-1)[r]=sink2[r]; sink2[r]=sinks[r]*L2 for r<H, otherwise neg for a padded head. Initialize l(-1)[r]=0 and Acc(-1)[r,:]=0. At c=0..C(t)-1: Z=mm32(Q,Gather(c)^T), accumulating D as 32 successive 16-wide products. Scores(c)[r,u]=Z[r,u]*scale, replaced by -inf where valid(c)[u]=0. zmax=maxnum(m(c-1)[r],maxnum_u Scores(c)[r,u]); m(c)[r]=zmax if zmax>neg*0.5, otherwise neg. alpha[r]=E2(m(c-1)[r]-m(c)[r]); E[r,u]=E2(Scores(c)[r,u]-m(c)[r]); Prob(c)[r,u]=bf16(E[r,u]). maxnum is FP32 fmaxf (a single NaN loses to the numeric operand). Sum E in KC/U consecutive groups, with U bound below. For each group v, form four sums GroupSum[a], a=0..3, each starting at zero and successively adding E[r,v*U+8*n+2*a]+E[r,v*U+8*n+2*a+1] for n=0..U/8-1. Starting rs at zero, add ((GroupSum[0]+GroupSum[1])+(GroupSum[2]+GroupSum[3])) in ascending v order. l(c)[r]=l(c-1)[r]*alpha[r]+rs; Acc(c)=mma32(Prob(c),Gather(c),alpha[:,None] ⊙ Acc(c-1)). The PV contraction advances through KC/16 consecutive 16-wide products.

## t_9aae864f6055e971 | math.math | 1635 chars | 1 declarations

probe_serial_packed:104 design.math_reshape.rewrite[1].math

PS is bound below and NG=ceil_div(T,GT). For sample b in [0,PS), if b>=NG set ProbeRatio[b]=fp32(-1). Otherwise let sg(b)=b if NG<=PS, else floor(b*NG/PS), st(b)=sg(b)*GT, pg(b)=min(GT,T-st(b)), and Items(b)=sum_{g=0}^{pg(b)-1} Klen(st(b)+g). A sample is usable when pg(b)=GT, all its tokens belong to the same Contract request, and 0<Items(b)<=2560. If unusable set ProbeRatio[b]=fp32(-1). For a usable sample, enumerate occurrences x=(g,j) with g in [0,GT) and j in J(st(b)+g). Let g(x)=g and e(x)=uint32(sparse_indices[st(b)+g,j]) bitwise-OR (0 if j<SW, otherwise 16777216). For each encoded key e, Creator(e) is its first entry-creating occurrence; for each (e,g), First(e,g) is its first membership-marking occurrence in the actual concurrent update order. These two orders need not agree. ProbeCount(b)=sum_x (1 if x=Creator(e(x)) else (0 if x=First(e(x),g(x)) else 1)). Invalid IDs are absent from J, but Items includes their selected slots. ProbeRatio[b]=(fp32(ProbeCount(b))*fp32(GT))/fp32(Items(b)). Only the count is retained locally; no attention union or KV is produced. Before this computation, the mapped sample slots contain fp32(-2). For each b let observed[b] be the actual FP32 value read by the host's final scan after its bounded poll; the scan need not observe all final sample values. Starting sum64=fp64(0), visit b=0..PS-1 and add fp64(observed[b]) in FP64 only when observed[b]>0. Let npos count these additions. R=fp32(sum64/fp64(npos)) if npos>0, otherwise fp32(-1). R is used only to select the following full-batch attention implementation; no probe value enters an attention score, probability or output.

## t_9cd7a72787ab811d | math.math | 1719 chars | 1 declarations

tc_heads:56 design.math_reshape.rewrite[1].math

E2(x) is the compiled fast exp2f(x), with E2(-inf)=0; maxnum is FP32 fmaxf, where a single NaN loses to the numeric operand. For each (t,hb), initialize sink2[r]=sinks[hb*HB+r]*L2; m(-1)[r]=sink2[r]; every row maps to a real head in this node. l(-1)[r]=0 and Acc(-1)[r,:]=0. For c=0..C(t)-1: Partial(c,a)=mm32(Q[:,a*D/2:(a+1)*D/2], Gather(c)[:,a*D/2:(a+1)*D/2]^T), a in {0,1}. Each partial accumulates D/32 successive 16-wide products from zero. Z=Partial(c,0)+Partial(c,1), with one FP32 addition per score. Scores(c)[r,u]=Z[r,u]*scale if valid(c)[u]!=0, otherwise -inf. For b in {0,1} and q in {0,1,2,3}, reduce Scores[r,b*32+4*n+q] for n=0..7 by successive maxnum from -inf, obtaining M[b,q]. Mhalf[b]=maxnum(maxnum(M[b,0],M[b,1]),maxnum(M[b,2],M[b,3])); candidate=maxnum(m(c-1)[r],maxnum(Mhalf[0],Mhalf[1])); clamped=candidate if candidate>neg*0.5, otherwise neg; m(c)[r]=m(c-1)[r] if clamped-m(c-1)[r]<=1, otherwise clamped. alpha(c)[r]=E2(m(c-1)[r]-m(c)[r]); E[r,u]=E2(Scores(c)[r,u]-m(c)[r]); Prob(c)[r,u]=bf16(E[r,u]). For each b,q form R[b,q] by adding E[r,b*32+4*n+q] from zero for n=0..7; Rhalf[b]=(R[b,0]+R[b,1])+(R[b,2]+R[b,3]); l(c)[r]=l(c-1)[r]*alpha(c)[r]+(Rhalf[0]+Rhalf[1]). Acc(0)=mm32(Prob(0),Gather(0)); Acc(c)=mma32(Prob(c),Gather(c),alpha(c)[:,None] ⊙ Acc(c-1)) for c>0. Each PV output half traverses KC/16 successive 16-wide products. Partition the rows into consecutive groups of RG rows, with RG bound below. Rescaling is omitted for a row group when every alpha is exactly 1; otherwise all its rows are multiplied, including those with alpha=1. The two key halves evaluate duplicate m,l updates; the lower half's l supplies the final denominator. Unrounded E supplies l, and Prob supplies PV.

## t_a32c4b66a90d848a | math.math | 895 chars | 1 declarations

mma_direct:41 design.math_reshape.rewrite[0].math

HB, KC, DQ and U are bound below; NB = ceil_div(H,HB), Hp = NB*HB. For token t use the Contract's request lookup and visible length. A(t) = min(max(visible,0),SW); Klen(t) = A(t) + max(min(sparse_topk_lens[t],IndexWidth)-SW,0). C(t) = ceil_div(Klen(t),KC), with Klen(t)>=1 in the Contract domain. For chunk c and u in [0,KC), let k=c*KC+u and j=k if k<A(t), else SW+k-A(t). id = sparse_indices[t,j] if k<Klen(t), otherwise -1; valid(c)[u] = uint8(1 if id>=0 else 0). Gather(c)[u,:] is row max(id,0) of swa_kv_cache if k<A(t), otherwise compressed_kv_cache. Invalid and padded positions therefore still gather row zero, not a zero latent. For head block hb and r in [0,HB), Q[r,:]=query[t,hb*HB+r,:] for hb*HB+r<H, otherwise Q[r,:]=bf16(0). Both matrix products include these padded head rows. a1=fp32(bmm1_scale); a2=fp32(bmm2_scale); L2=fp32(1.4426950408889634); neg=fp32(-3.0e38); scale=a1*L2.

## t_a679871a6449e04f | math.math | 333 chars | 1 declarations

tc_heads:103 design.math_reshape.rewrite[2].math

For last=C(t)-1, z[r]=l(last)[r]+E2(sink2[r]-m(last)[r]); inv[r]=a2/z[r] if z[r]>0, otherwise 0; Result[r,d]=bf16(Acc(last)[r,d]*inv[r]); overwrite out[t,hb*HB+r,d]=Result[r,d] for r in [0,HB), d in [0,D). These formulas are evaluated even when J(t) is empty. E2, division and scalar contraction follow online's arithmetic exception.

## t_aa25845eb3946746 | math.math | 1124 chars | 1 declarations

probe_hidden_mma:236 design.math_reshape.rewrite[5].math

E2 and maxnum have the definitions in online. For each remainder token t, initialize SinkM[r]=sinks[r]*L2 if r<H, otherwise neg; mM(-1)[r]=SinkM[r], lM(-1)[r]=0 and AccM(-1)[r,:]=0. For c=0..CM(t)-1: Zm=mm32(Qm,GatherM(c)^T), accumulating 32 successive 16-wide products. ScoresM(c)[r,u]=Zm[r,u]*scale, then replace it by -inf where ValidM(c)[u]=0. candidate=maxnum(mM(c-1)[r],maxnum_u ScoresM(c)[r,u]); mM(c)[r]=candidate if candidate>neg*0.5, otherwise neg. AlphaM(c)[r]=E2(mM(c-1)[r]-mM(c)[r]); Em[r,u]=E2(ScoresM(c)[r,u]-mM(c)[r]); ProbM(c)[r,u]=bf16(Em[r,u]). For each consecutive UM-column group v, form four sums S[a], a=0..3, each starting at zero and successively adding Em[r,v*UM+8*n+2*a]+Em[r,v*UM+8*n+2*a+1] for n=0..UM/8-1. Starting rs=0, add ((S[0]+S[1])+(S[2]+S[3])) in ascending v=0..KM/UM-1. lM(c)[r]=lM(c-1)[r]*AlphaM(c)[r]+rs; AccM(c)=mma32(ProbM(c),GatherM(c),AlphaM(c)[:,None] ⊙ AccM(c-1)). The PV contraction advances through KM/16 consecutive 16-wide products. UM is bound below. Every chunk rescales all accumulator rows, including c=0. No one-unit lag or half-latent QK addition is used on this path.

## t_adfc51e6e3569085 | math.math | 1797 chars | 1 declarations

probe_hidden_mma:139 design.math_reshape.rewrite[2].math

E2(x) is the compiled fast exp2f(x), with E2(-inf)=0; maxnum is FP32 fmaxf, where a single NaN loses to the numeric operand. For each prefix grp in [0,K) with C(grp)>0, initialize sink2[r]=sinks[r mod H]*L2 for r<nrows and neg for padded rows; m(-1)[r]=sink2[r]. l(-1)[r]=0 and Acc(-1)[r,:]=0. For c=0..C(grp)-1: Partial(c,a)=mm32(Q[:,a*D/2:(a+1)*D/2], Gather(c)[:,a*D/2:(a+1)*D/2]^T), a in {0,1}. Each partial accumulates D/32 successive 16-wide products from zero. Z=Partial(c,0)+Partial(c,1), with one FP32 addition per score. Scores(c)[r,u]=Z[r,u]*scale if floor(Mask(c)[u]/2^floor(r/H)) mod 2 != 0, otherwise -inf. For b in {0,1} and q in {0,1,2,3}, reduce Scores[r,b*32+4*n+q] for n=0..7 by successive maxnum from -inf, obtaining M[b,q]. Mhalf[b]=maxnum(maxnum(M[b,0],M[b,1]),maxnum(M[b,2],M[b,3])); candidate=maxnum(m(c-1)[r],maxnum(Mhalf[0],Mhalf[1])); clamped=candidate if candidate>neg*0.5, otherwise neg; m(c)[r]=m(c-1)[r] if clamped-m(c-1)[r]<=1, otherwise clamped. alpha(c)[r]=E2(m(c-1)[r]-m(c)[r]); E[r,u]=E2(Scores(c)[r,u]-m(c)[r]); Prob(c)[r,u]=bf16(E[r,u]). For each b,q form SumPart[b,q] by adding E[r,b*32+4*n+q] from zero for n=0..7; Rhalf[b]=(SumPart[b,0]+SumPart[b,1])+(SumPart[b,2]+SumPart[b,3]); l(c)[r]=l(c-1)[r]*alpha(c)[r]+(Rhalf[0]+Rhalf[1]). Acc(0)=mm32(Prob(0),Gather(0)); Acc(c)=mma32(Prob(c),Gather(c),alpha(c)[:,None] ⊙ Acc(c-1)) for c>0. Each PV output half traverses KC/16 successive 16-wide products. Partition the rows into consecutive groups of RG rows, with RG bound below. Rescaling is omitted for a row group when every alpha is exactly 1; otherwise all its rows are multiplied, including those with alpha=1. The two key halves evaluate duplicate m,l updates; the lower half's l supplies the final denominator. Unrounded E supplies l, and Prob supplies PV.

## t_ae0488acee5bf41d | math.math | 733 chars | 1 declarations

probe_serial_mma:38 design.math_reshape.rewrite[0].math

HB and KC are bound below. For token t use the Contract's request lookup b(t) and visible length. A(t)=min(max(visible,0),SW); Klen(t)=A(t)+max(min(sparse_topk_lens[t],IndexWidth)-SW,0). C(t)=ceil_div(Klen(t),KC), with Klen(t)>=1 in the Contract domain. For chunk c and u in [0,KC), let k=c*KC+u and j=k if k<A(t), otherwise SW+k-A(t). id=sparse_indices[t,j] if k<Klen(t), otherwise -1; valid(c)[u]=uint8(1 if id>=0 else 0). Gather(c)[u,:] is row max(id,0) of swa_kv_cache if k<A(t), otherwise compressed_kv_cache. Invalid and padded positions still gather row zero. For r in [0,HB), Q[r,:]=query[t,r,:] if r<H, otherwise bf16(0). a1=fp32(bmm1_scale); a2=fp32(bmm2_scale); L2=fp32(1.4426950408889634); neg=fp32(-3.0e38); scale=a1*L2.

## t_b25d78cf367beb3d | math.math | 389 chars | 1 declarations

probe_serial_mma:139 design.math_reshape.rewrite[4].math

For r in [0,HB), let last=C(t)-1, z[r]=l(last)[r]+E2(sink2[r]-m(last)[r]), inv[r]=a2/z[r] if r<H and z[r]>0, otherwise 0. Result[r,d]=bf16(Acc(last)[r,d]*inv[r]) for d in [0,D), including padded heads. For real heads only, overwrite out[t,r,d]=Result[r,d]. Evaluate these formulas even when J(t) is empty. E2, division and contracted scalar arithmetic follow online's arithmetic exception.

## t_ba4b747e8b06d347 | math.math | 1968 chars | 3 declarations

probe_hidden_packed:46 design.math_reshape.rewrite[0].math; probe_serial_packed:46 design.math_reshape.rewrite[0].math; tc_packed:37 design.math_reshape.rewrite[0].math

GT, MR and KC are bound below. For group grp let t0=grp*GT, G=min(GT,T-t0), nrows=G*H, and t_g=t0+g for g in [0,G). For each real token use the Contract's request lookup and visible length: A(t)=min(max(visible,0),SW); Klen(t)=A(t)+max(min(sparse_topk_lens[t],IndexWidth)-SW,0). For each distinct (pool,id) among the group's valid slots J(t_g), let m_g(pool,id) count its occurrences in query g. Pools remain distinct even when their pointers or row numbers coincide; pool 0 is SWA, pool 1 compressed. Create one shared entry carrying all bits g with m_g>0, then m_g-1 private entries carrying only bit g for each g with m_g>1. Every entry encodes UnionKey[k]=int32(pool*16777216+id) and UnionMask[k]=uint8(sum_g bit_g*2^g). Let Ns and Ng be the entry counts for pools 0 and 1, Nu(grp)=Ns+Ng, and C(grp)=ceil_div(Nu(grp),KC). Nu(grp)<=sum_g count(J(t_g))<=GT*IndexWidth. Entries are ordered with all SWA entries first and all compressed entries second. Within each pool preserve the runtime union-construction permutation; no stable order across executions is assumed. For each (pool,id) its number of entries is 1+sum_g max(m_g(pool,id)-1,0), not max_g m_g(pool,id). For r in [0,MR), Q[r,:]=query[t0+floor(r/H),r mod H,:] when r<nrows, otherwise bf16(0). For c in [0,C(grp)) and u in [0,KC), let k=c*KC+u. Mask(c)[u]=UnionMask[k] if k<Nu(grp), otherwise uint8(0). Define pool(k)=floor(UnionKey[k]/16777216) for k<Nu(grp), otherwise 0; id(k)=UnionKey[k] mod 16777216 for k<Nu(grp), otherwise 0. Let mixed(c) mean 0<Ns-c*KC<KC and (Ns-c*KC) mod 4 != 0. The gathered pool is pool(k) if mixed(c), otherwise pool(c*KC+4*floor(u/4)); Gather(c)[u,:] is row id(k) of that pool. Thus padding still fetches row zero; a padded tail sharing a four-row group with compressed entries uses compressed row zero, while wholly padded groups and mixed-chunk padding use SWA row zero. a1=fp32(bmm1_scale); a2=fp32(bmm2_scale); L2=fp32(1.4426950408889634); neg=fp32(-3.0e38); scale=a1*L2.

## t_c3dbdac3aacc1966 | math.math | 429 chars | 1 declarations

mma_direct:98 design.math_reshape.rewrite[2].math

For each (t,hb,dq) and r in [0,HB), let last=C(t)-1 and z[r]=l(last)[r]+E2(sink2[r]-m(last)[r]); inv[r]=a2/z[r] if hb*HB+r<H and z[r]>0, otherwise 0. Result[r,v]=bf16(Acc(last)[r,v]*inv[r]) for v in [0,D/DQ), including padded heads. For real heads only, overwrite out[t,hb*HB+r,d0+v]=Result[r,v]. Evaluate these formulas even when J(t) is empty. E2, division and contracted scalar arithmetic follow online's arithmetic exception.

## t_ccf4b8cf8413248f | math.math | 1163 chars | 1 declarations

probe_serial_mma:61 design.math_reshape.rewrite[1].math

GT is bound below; NG=ceil_div(T,GT). For sample s in [0,8), g(s)=s if NG<=8, otherwise floor(s*NG/8), and t0(s)=g(s)*GT. For a in [0,GT), t=t0(s)+a; n_a=Klen(t) and request_a=b(t) when t<T, otherwise n_a=0 and request_a=-2. Let Items(s)=sum_a n_a. Usable(s) means s<NG, all request_a equal, and 0<Items(s)<=2560. If not Usable(s), ProbeRatio[s]=fp32(-1) without processing indices. Otherwise enumerate occurrences x=(a,k), a in [0,GT), k in [0,n_a). Use gather's compact-slot mapping to read id; discard id<0 from the set build, but retain its position in Items(s). For each retained occurrence define e(x)=uint32(id) bitwise-OR (0 for SWA, 16777216 for compressed). For each distinct encoded key e, let Creator(e) be its first entry-creating occurrence; for each (e,a), let First(e,a) be its first membership-marking occurrence in the actual concurrent update order. Count(s)=sum_{retained x} (1 if x=Creator(e(x)) else (0 if x=First(e(x),a(x)) else 1)). The entry-creation and membership-marking orders need not agree. ProbeRatio[s]=fp32(fp32(Count(s))*fp32(GT)/fp32(Items(s))). All eight ProbeRatio elements are eventually written, including unusable samples.

## t_d411c01b5929b7a2 | math.math | 882 chars | 1 declarations

root:250 design.math_reshape.rewrite[3].math

Independently for each real (t,h), use G groups, bound below. M=maxnum(neg,maxnum_s PartM[s,h]); w[s]=E2(PartM[s,h]-M) if PartL[s,h]>0, otherwise 0. For g=0..G-1, sum w[s]*PartL[s,h] in ascending s=g,g+G,... into z_g. z=sum_g z_g in ascending g, then add E2(sinks[h]*L2-M) once. inv=a2/z if z>0, otherwise 0. For each g and latent d, a_g starts at zero. The first four s=g+k*G (k=0..3) load PartO whenever s<S, even if unwritten; add w[s]*PartO[s,h,d] only when w[s]!=0. Later groups of four splits load PartO[s,h,d] when w[s]!=0, otherwise PartO[0,h,d], and add their four weighted values as one expression; finish remaining splits singly with the same address substitution. These later zero-weight products are still evaluated. Fold a_g in ascending g, and write out[t,h,d]=bf16((sum_g a_g)*inv). E2, division and contracted scalar arithmetic follow online's arithmetic exception.

## t_e45a09d0c9a8bae7 | math.math | 798 chars | 1 declarations

probe_serial_mma:88 design.math_reshape.rewrite[2].math

During eager host selection or external graph recording, set Gate[s]=fp32(-2) for s in [0,8) before submitting or recording the probe. The executed probe writes ProbeRatio[s] to Gate[s]. After bounded polling, let v_s be the actual FP32 value read for slot s during the final ascending scan. These reads need not observe every completed probe result or one common snapshot. Starting Sum=fp64(0) and n=0, for s=0..7 add fp64(v_s) to Sum and increment n only when v_s>0. R=fp32(Sum/fp64(n)) if n>0, otherwise fp32(-1). This node executes attention for every token when not (0<R<=fp32(2.6)). The eight device writes still occur even if none contributed to R. Execution of a recorded graph uses its recorded R and attention choice; it repeats the device writes but not host reset, polling or selection.

## t_e76a41927eca61bd | math.math | 1780 chars | 1 declarations

probe_hidden_mma:97 design.math_reshape.rewrite[1].math

PS is bound below and NG=ceil_div(T,GT). For sample b in [0,PS), if b>=NG set ProbeRatio[b]=fp32(-1). Otherwise let sg(b)=b if NG<=PS, else floor(b*NG/PS), st(b)=sg(b)*GT, pg(b)=min(GT,T-st(b)), and Items(b)=sum_{g=0}^{pg(b)-1} Klen(st(b)+g). A sample is usable when pg(b)=GT, all its tokens belong to the same Contract request, and 0<Items(b)<=2560. If unusable set ProbeRatio[b]=fp32(-1). For a usable sample, enumerate occurrences x=(g,j) with g in [0,GT) and j in J(st(b)+g). Let g(x)=g and e(x)=uint32(sparse_indices[st(b)+g,j]) bitwise-OR (0 if j<SW, otherwise 16777216). For each encoded key e, Creator(e) is its first entry-creating occurrence; for each (e,g), First(e,g) is its first membership-marking occurrence in the actual concurrent update order. These two orders need not agree. ProbeCount(b)=sum_x (1 if x=Creator(e(x)) else (0 if x=First(e(x),g(x)) else 1)). Invalid IDs are absent from J, but Items includes their selected slots. ProbeRatio[b]=(fp32(ProbeCount(b))*fp32(GT))/fp32(Items(b)). Only the count is retained locally; no attention union or KV is produced. Before this computation, the mapped sample slots contain fp32(-2). For each b let observed[b] be the actual FP32 value read by the host's final scan after its bounded poll; the scan need not observe all final sample values. Starting sum64=fp64(0), visit b=0..PS-1 and add fp64(observed[b]) in FP64 only when observed[b]>0. Let npos count these additions. R=fp32(sum64/fp64(npos)) if npos>0, otherwise fp32(-1). R selects only the remainder implementation; the prefix uses packing independently of R. This node rejects packing for the remainder, which uses mma_gather, mma_online and mma_finalize. The committed prefix is unchanged. No probe value enters an attention score, probability or output.

## t_0219b4ca52f919c6 | math.meaning | 74 chars | 5 declarations

probe_hidden_mma:174 design.math_reshape.rewrite[2].symbol.E.meaning; probe_hidden_packed:196 design.math_reshape.rewrite[2].symbol.E.meaning; probe_serial_packed:195 design.math_reshape.rewrite[2].symbol.E.meaning; tc_heads:91 design.math_reshape.rewrite[1].symbol.E.meaning; tc_packed:115 design.math_reshape.rewrite[1].symbol.E.meaning

Unrounded exponentials used for the normalizer and probability conversion.

## t_098617c88172e315 | math.meaning | 37 chars | 1 declarations

root:247 design.math_reshape.rewrite[2].symbol.PartL.meaning

Split normalizers excluding the sink.

## t_0c3579a6cec5926b | math.meaning | 47 chars | 5 declarations

probe_hidden_mma:172 design.math_reshape.rewrite[2].symbol.Partial.meaning; probe_hidden_packed:188 design.math_reshape.rewrite[2].symbol.Partial.meaning; probe_serial_packed:187 design.math_reshape.rewrite[2].symbol.Partial.meaning; tc_heads:89 design.math_reshape.rewrite[1].symbol.Partial.meaning; tc_packed:113 design.math_reshape.rewrite[1].symbol.Partial.meaning

FP32 QK sum for one latent half a of one chunk.

## t_15c82f2892df61c8 | math.meaning | 55 chars | 1 declarations

probe_hidden_mma:265 design.math_reshape.rewrite[5].symbol.lM.meaning

Remainder unrounded exponential sum excluding the sink.

## t_17507c51943ec033 | math.meaning | 51 chars | 1 declarations

tc_heads:94 design.math_reshape.rewrite[1].symbol.sink2.meaning

Base-2 logit of the required sink for head hb*HB+r.

## t_23b5d88a3a04a882 | math.meaning | 36 chars | 1 declarations

tc_heads:112 design.math_reshape.rewrite[2].symbol.Result.meaning

Complete normalized BF16 head block.

## t_369a9bed53a4ab61 | math.meaning | 67 chars | 1 declarations

probe_serial_mma:102 design.math_reshape.rewrite[2].symbol.R.meaning

Actual positive-value mean used once to choose the full-batch plan.

## t_38245acb216b71fe | math.meaning | 40 chars | 3 declarations

mma_direct:61 design.math_reshape.rewrite[0].symbol.Q.meaning; probe_serial_mma:58 design.math_reshape.rewrite[0].symbol.Q.meaning; root:200 design.math_reshape.rewrite[0].symbol.Q.meaning

Query head block with zero padded heads.

## t_3872f4a6a7eec705 | math.meaning | 57 chars | 1 declarations

probe_serial_mma:131 design.math_reshape.rewrite[3].symbol.l.meaning

Running sum of unrounded exponentials excluding the sink.

## t_3982108d57af9381 | math.meaning | 57 chars | 1 declarations

probe_serial_mma:129 design.math_reshape.rewrite[3].symbol.sink2.meaning

Base-2 sink logits with the defined padded-head sentinel.

## t_3aa7f2c78b794ed1 | math.meaning | 58 chars | 2 declarations

mma_direct:86 design.math_reshape.rewrite[1].symbol.Scores.meaning; root:226 design.math_reshape.rewrite[1].symbol.Scores.meaning

Scaled and masked FP32 score matrix for the current chunk.

## t_434f7c110b2263f1 | math.meaning | 48 chars | 1 declarations

probe_serial_mma:57 design.math_reshape.rewrite[0].symbol.valid.meaning

Validity of gathered slots before score masking.

## t_44301386fc164190 | math.meaning | 39 chars | 1 declarations

probe_hidden_mma:259 design.math_reshape.rewrite[5].symbol.ProbM.meaning

Rounded remainder probability operands.

## t_44dc16328d646eec | math.meaning | 55 chars | 1 declarations

probe_hidden_mma:86 design.math_reshape.rewrite[0].symbol.UnionKey.meaning

Pool-tagged row IDs in the realized prefix union order.

## t_4e0fe778d83b9e97 | math.meaning | 48 chars | 4 declarations

probe_hidden_mma:93 design.math_reshape.rewrite[0].symbol.Mask.meaning; probe_hidden_packed:99 design.math_reshape.rewrite[0].symbol.Mask.meaning; probe_serial_packed:99 design.math_reshape.rewrite[0].symbol.Mask.meaning; tc_packed:77 design.math_reshape.rewrite[0].symbol.Mask.meaning

Per-chunk query membership before score masking.

## t_4e1fb6a57f5415e3 | math.meaning | 66 chars | 1 declarations

tc_heads:111 design.math_reshape.rewrite[2].symbol.inv.meaning

Final per-head output multiplier published for both latent halves.

## t_4ec4c586ca67b842 | math.meaning | 68 chars | 1 declarations

root:246 design.math_reshape.rewrite[2].symbol.PartM.meaning

Split maxima including sink initialization on nonempty chunk ranges.

## t_50faad2fd397f560 | math.meaning | 36 chars | 1 declarations

probe_hidden_mma:227 design.math_reshape.rewrite[4].symbol.HB.meaning

Padded heads per remainder worktile.

## t_54169e8310ee3657 | math.meaning | 39 chars | 5 declarations

probe_hidden_mma:177 design.math_reshape.rewrite[2].symbol.alpha.meaning; probe_hidden_packed:204 design.math_reshape.rewrite[2].symbol.alpha.meaning; probe_serial_packed:203 design.math_reshape.rewrite[2].symbol.alpha.meaning; tc_heads:93 design.math_reshape.rewrite[1].symbol.alpha.meaning; tc_packed:118 design.math_reshape.rewrite[1].symbol.alpha.meaning

Rescaling factor for the current chunk.

## t_580abe2b5514df90 | math.meaning | 34 chars | 4 declarations

probe_hidden_mma:83 design.math_reshape.rewrite[0].symbol.GT.meaning; probe_hidden_packed:77 design.math_reshape.rewrite[0].symbol.GT.meaning; probe_serial_packed:77 design.math_reshape.rewrite[0].symbol.GT.meaning; tc_packed:68 design.math_reshape.rewrite[0].symbol.GT.meaning

Queries grouped into one worktile.

## t_582fe050ea80e343 | math.meaning | 40 chars | 5 declarations

probe_hidden_mma:182 design.math_reshape.rewrite[2].symbol.l.meaning; probe_hidden_packed:218 design.math_reshape.rewrite[2].symbol.l.meaning; probe_serial_packed:217 design.math_reshape.rewrite[2].symbol.l.meaning; tc_heads:96 design.math_reshape.rewrite[1].symbol.l.meaning; tc_packed:122 design.math_reshape.rewrite[1].symbol.l.meaning

FP32 exponential sum excluding the sink.

## t_5af5960fe2400614 | math.meaning | 45 chars | 1 declarations

root:265 design.math_reshape.rewrite[3].symbol.G.meaning

Number of interleaved split reduction groups.

## t_5b81b10d42c74280 | math.meaning | 29 chars | 3 declarations

mma_direct:55 design.math_reshape.rewrite[0].symbol.HB.meaning; probe_serial_mma:54 design.math_reshape.rewrite[0].symbol.HB.meaning; root:193 design.math_reshape.rewrite[0].symbol.HB.meaning

Heads per attention worktile.

## t_5c1d77225f4527f3 | math.meaning | 61 chars | 1 declarations

probe_serial_mma:128 design.math_reshape.rewrite[3].symbol.alpha.meaning

Row rescaling factors for normalizers and value accumulators.

## t_6142e2abb1b1df44 | math.meaning | 50 chars | 1 declarations

tc_heads:51 design.math_reshape.rewrite[0].symbol.Q.meaning

Query head block reused throughout the recurrence.

## t_6283f5694006e624 | math.meaning | 38 chars | 5 declarations

probe_hidden_mma:173 design.math_reshape.rewrite[2].symbol.Scores.meaning; probe_hidden_packed:192 design.math_reshape.rewrite[2].symbol.Scores.meaning; probe_serial_packed:191 design.math_reshape.rewrite[2].symbol.Scores.meaning; tc_heads:90 design.math_reshape.rewrite[1].symbol.Scores.meaning; tc_packed:114 design.math_reshape.rewrite[1].symbol.Scores.meaning

Added, scaled and masked chunk scores.

## t_62eccd7db70b9de3 | math.meaning | 61 chars | 3 declarations

probe_hidden_mma:130 design.math_reshape.rewrite[1].symbol.ProbeRatio.meaning; probe_hidden_packed:140 design.math_reshape.rewrite[1].symbol.ProbeRatio.meaning; probe_serial_packed:139 design.math_reshape.rewrite[1].symbol.ProbeRatio.meaning

Per-sample FP32 control results published to the mapped gate.

## t_68ff7b9e1f563b0c | math.meaning | 27 chars | 5 declarations

probe_hidden_mma:180 design.math_reshape.rewrite[2].symbol.m.meaning; probe_hidden_packed:213 design.math_reshape.rewrite[2].symbol.m.meaning; probe_serial_packed:212 design.math_reshape.rewrite[2].symbol.m.meaning; tc_heads:95 design.math_reshape.rewrite[1].symbol.m.meaning; tc_packed:121 design.math_reshape.rewrite[1].symbol.m.meaning

Lagging base-2 row maximum.

## t_6a83d0d31cc1c637 | math.meaning | 46 chars | 5 declarations

probe_hidden_mma:171 design.math_reshape.rewrite[2].symbol.RG.meaning; probe_hidden_packed:186 design.math_reshape.rewrite[2].symbol.RG.meaning; probe_serial_packed:185 design.math_reshape.rewrite[2].symbol.RG.meaning; tc_heads:88 design.math_reshape.rewrite[1].symbol.RG.meaning; tc_packed:112 design.math_reshape.rewrite[1].symbol.RG.meaning

Rows sharing the conditional rescale decision.

## t_6a8a2246588b065e | math.meaning | 48 chars | 4 declarations

probe_hidden_mma:91 design.math_reshape.rewrite[0].symbol.Gather.meaning; probe_hidden_packed:95 design.math_reshape.rewrite[0].symbol.Gather.meaning; probe_serial_packed:95 design.math_reshape.rewrite[0].symbol.Gather.meaning; tc_packed:76 design.math_reshape.rewrite[0].symbol.Gather.meaning

Shared cache rows including padding substitutes.

## t_6bdacdff4f3437fe | math.meaning | 61 chars | 2 declarations

mma_direct:91 design.math_reshape.rewrite[1].symbol.Acc.meaning; root:231 design.math_reshape.rewrite[1].symbol.Acc.meaning

Unnormalized weighted value accumulator for one output slice.

## t_6fdea369ce839eba | math.meaning | 62 chars | 1 declarations

probe_serial_mma:81 design.math_reshape.rewrite[1].symbol.GT.meaning

Candidate packing group size sampled before rejecting packing.

## t_738289cf503035d5 | math.meaning | 47 chars | 1 declarations

probe_hidden_mma:267 design.math_reshape.rewrite[5].symbol.AccM.meaning

Remainder unnormalized full-latent accumulator.

## t_772407e2b92c8f70 | math.meaning | 63 chars | 1 declarations

probe_serial_mma:148 design.math_reshape.rewrite[4].symbol.Result.meaning

Normalized BF16 output including locally computed padded heads.

## t_7796eb5b07cc462b | math.meaning | 65 chars | 3 declarations

probe_hidden_packed:86 design.math_reshape.rewrite[0].symbol.UnionMask.meaning; probe_serial_packed:86 design.math_reshape.rewrite[0].symbol.UnionMask.meaning; tc_packed:73 design.math_reshape.rewrite[0].symbol.UnionMask.meaning

Membership bits selecting which queries consume each union entry.

## t_79b351b9fce4f1c0 | math.meaning | 74 chars | 4 declarations

probe_hidden_mma:178 design.math_reshape.rewrite[2].symbol.sink2.meaning; probe_hidden_packed:208 design.math_reshape.rewrite[2].symbol.sink2.meaning; probe_serial_packed:207 design.math_reshape.rewrite[2].symbol.sink2.meaning; tc_packed:119 design.math_reshape.rewrite[1].symbol.sink2.meaning

Base-2 sink logit for each packed row, with a finite sentinel for padding.

## t_8566e906ab445b88 | math.meaning | 35 chars | 1 declarations

probe_hidden_mma:258 design.math_reshape.rewrite[5].symbol.ScoresM.meaning

Scaled and masked remainder scores.

## t_866d1f18f422f125 | math.meaning | 48 chars | 3 declarations

probe_hidden_packed:81 design.math_reshape.rewrite[0].symbol.UnionKey.meaning; probe_serial_packed:81 design.math_reshape.rewrite[0].symbol.UnionKey.meaning; tc_packed:71 design.math_reshape.rewrite[0].symbol.UnionKey.meaning

Pool-tagged row IDs in the realized union order.

## t_87a7a61e92e07274 | math.meaning | 61 chars | 1 declarations

probe_serial_mma:126 design.math_reshape.rewrite[3].symbol.U.meaning

Key group width used by the score and probability reductions.

## t_881ed3dafa0dabb2 | math.meaning | 72 chars | 4 declarations

probe_hidden_mma:208 design.math_reshape.rewrite[3].symbol.inv.meaning; probe_hidden_packed:249 design.math_reshape.rewrite[3].symbol.inv.meaning; probe_serial_packed:247 design.math_reshape.rewrite[3].symbol.inv.meaning; tc_packed:146 design.math_reshape.rewrite[2].symbol.inv.meaning

Final per-row multiplier shared by both output halves, zero for padding.

## t_8947c0023f8987d2 | math.meaning | 46 chars | 1 declarations

probe_hidden_mma:231 design.math_reshape.rewrite[4].symbol.ValidM.meaning

Remainder chunk validity before score masking.

## t_8a0b96d4d75a1673 | math.meaning | 30 chars | 1 declarations

tc_heads:53 design.math_reshape.rewrite[0].symbol.valid.meaning

Validity before score masking.

## t_8bb9b754b2ab6b94 | math.meaning | 30 chars | 1 declarations

probe_serial_mma:127 design.math_reshape.rewrite[3].symbol.Scores.meaning

Scaled and masked FP32 scores.

## t_8cadf4bd4fe7eaf2 | math.meaning | 54 chars | 2 declarations

mma_direct:59 design.math_reshape.rewrite[0].symbol.Gather.meaning; root:198 design.math_reshape.rewrite[0].symbol.Gather.meaning

Actual BF16 cache rows including row-zero substitutes.

## t_8d23eb926946c264 | math.meaning | 53 chars | 1 declarations

probe_hidden_mma:229 design.math_reshape.rewrite[4].symbol.Qm.meaning

Query rows for one remainder token including padding.

## t_8da414237e23c1a8 | math.meaning | 27 chars | 3 declarations

mma_direct:89 design.math_reshape.rewrite[1].symbol.m.meaning; probe_serial_mma:130 design.math_reshape.rewrite[3].symbol.m.meaning; root:229 design.math_reshape.rewrite[1].symbol.m.meaning

Running base-2 row maximum.

## t_8ecca04533507473 | math.meaning | 30 chars | 4 declarations

probe_hidden_mma:85 design.math_reshape.rewrite[0].symbol.KC.meaning; probe_hidden_packed:79 design.math_reshape.rewrite[0].symbol.KC.meaning; probe_serial_packed:79 design.math_reshape.rewrite[0].symbol.KC.meaning; tc_packed:70 design.math_reshape.rewrite[0].symbol.KC.meaning

Union entries per local chunk.

## t_908b67e0ed82bcc0 | math.meaning | 40 chars | 1 declarations

probe_serial_mma:132 design.math_reshape.rewrite[3].symbol.Acc.meaning

Unnormalized weighted value accumulator.

## t_90fbd45af4f878a3 | math.meaning | 47 chars | 1 declarations

probe_serial_mma:56 design.math_reshape.rewrite[0].symbol.Gather.meaning

BF16 cache rows including row-zero substitutes.

## t_9eae820021199c83 | math.meaning | 59 chars | 3 declarations

probe_hidden_mma:132 design.math_reshape.rewrite[1].symbol.R.meaning; probe_hidden_packed:145 design.math_reshape.rewrite[1].symbol.R.meaning; probe_serial_packed:144 design.math_reshape.rewrite[1].symbol.R.meaning

Actual positive-sample mean read by the host for selection.

## t_9f4caf45d8c04822 | math.meaning | 26 chars | 1 declarations

probe_serial_mma:55 design.math_reshape.rewrite[0].symbol.KC.meaning

Slots per attention chunk.

## t_a110d9a36b05cd9f | math.meaning | 72 chars | 2 declarations

mma_direct:58 design.math_reshape.rewrite[0].symbol.U.meaning; root:197 design.math_reshape.rewrite[0].symbol.U.meaning

Contiguous key group width used by the score and probability reductions.

## t_a5364b25d4745d55 | math.meaning | 69 chars | 1 declarations

mma_direct:107 design.math_reshape.rewrite[2].symbol.Result.meaning

Normalized BF16 output slice including locally computed padded heads.

## t_a7f2c14d2d443884 | math.meaning | 53 chars | 1 declarations

probe_hidden_mma:88 design.math_reshape.rewrite[0].symbol.UnionMask.meaning

Membership bits for each retained prefix union entry.

## t_b0ee21db3eab0cfa | math.meaning | 32 chars | 1 declarations

root:195 design.math_reshape.rewrite[0].symbol.S.meaning

Number of independent KV splits.

## t_b10f0464cf1a2b62 | math.meaning | 36 chars | 3 declarations

probe_hidden_mma:128 design.math_reshape.rewrite[1].symbol.PS.meaning; probe_hidden_packed:134 design.math_reshape.rewrite[1].symbol.PS.meaning; probe_serial_packed:133 design.math_reshape.rewrite[1].symbol.PS.meaning

Number of independent probe samples.

## t_b450d06af855e1ce | math.meaning | 53 chars | 2 declarations

mma_direct:92 design.math_reshape.rewrite[1].symbol.Prob.meaning; root:232 design.math_reshape.rewrite[1].symbol.Prob.meaning

Rounded per-chunk probability matrix used only by PV.

## t_b58c750bf99885de | math.meaning | 58 chars | 1 declarations

probe_serial_mma:82 design.math_reshape.rewrite[1].symbol.Count.meaning

Actual integer numerator accumulated by one sampled group.

## t_b6f46323d489fa5a | math.meaning | 56 chars | 1 declarations

probe_hidden_mma:257 design.math_reshape.rewrite[5].symbol.UM.meaning

Consecutive key group width in the remainder reductions.

## t_be3076dfd65915b3 | math.meaning | 58 chars | 1 declarations

tc_heads:52 design.math_reshape.rewrite[0].symbol.Gather.meaning

Cache rows including invalid-slot and padding substitutes.

## t_c4208be2a684180d | math.meaning | 56 chars | 2 declarations

mma_direct:88 design.math_reshape.rewrite[1].symbol.sink2.meaning; root:228 design.math_reshape.rewrite[1].symbol.sink2.meaning

Base-2 sink logit with the defined padded-head sentinel.

## t_c69c3ce2f4089311 | math.meaning | 35 chars | 5 declarations

probe_hidden_mma:184 design.math_reshape.rewrite[2].symbol.Acc.meaning; probe_hidden_packed:223 design.math_reshape.rewrite[2].symbol.Acc.meaning; probe_serial_packed:222 design.math_reshape.rewrite[2].symbol.Acc.meaning; tc_heads:97 design.math_reshape.rewrite[1].symbol.Acc.meaning; tc_packed:124 design.math_reshape.rewrite[1].symbol.Acc.meaning

Unnormalized full-latent value sum.

## t_cae7298a4f0fb4b6 | math.meaning | 26 chars | 1 declarations

probe_hidden_mma:228 design.math_reshape.rewrite[4].symbol.KM.meaning

Slots per remainder chunk.

## t_cb22f9a5a8a64d29 | math.meaning | 63 chars | 1 declarations

root:245 design.math_reshape.rewrite[2].symbol.PartO.meaning

Unnormalized split vectors; empty-split slots remain unwritten.

## t_cc9bd54c312dfc8a | math.meaning | 80 chars | 2 declarations

mma_direct:87 design.math_reshape.rewrite[1].symbol.alpha.meaning; root:227 design.math_reshape.rewrite[1].symbol.alpha.meaning

Per-chunk row rescaling factor for the running normalizer and value accumulator.

## t_cca0f45e807e72b8 | math.meaning | 22 chars | 3 declarations

mma_direct:56 design.math_reshape.rewrite[0].symbol.KC.meaning; root:194 design.math_reshape.rewrite[0].symbol.KC.meaning; tc_heads:50 design.math_reshape.rewrite[0].symbol.KC.meaning

Slots per local chunk.

## t_d7bbe5c1a2666cfc | math.meaning | 54 chars | 4 declarations

probe_hidden_mma:210 design.math_reshape.rewrite[3].symbol.Result.meaning; probe_hidden_packed:253 design.math_reshape.rewrite[3].symbol.Result.meaning; probe_serial_packed:251 design.math_reshape.rewrite[3].symbol.Result.meaning; tc_packed:148 design.math_reshape.rewrite[2].symbol.Result.meaning

Normalized BF16 rows for the real tokens of the group.

## t_d9abbe8ae08e8d2b | math.meaning | 60 chars | 1 declarations

probe_serial_mma:83 design.math_reshape.rewrite[1].symbol.ProbeRatio.meaning

Final per-sample statistics consumed by host plan selection.

## t_da46e7ac2b984c9b | math.meaning | 19 chars | 1 declarations

tc_heads:49 design.math_reshape.rewrite[0].symbol.HB.meaning

Heads per worktile.

## t_dbf63553fef93cef | math.meaning | 38 chars | 1 declarations

probe_serial_mma:133 design.math_reshape.rewrite[3].symbol.Prob.meaning

Rounded probabilities used only by PV.

## t_dd23cc0f7b51d479 | math.meaning | 60 chars | 1 declarations

probe_serial_mma:101 design.math_reshape.rewrite[2].symbol.Gate.meaning

Reused mapped statistics storage observed by host selection.

## t_dde39ad09adb46a5 | math.meaning | 26 chars | 1 declarations

probe_hidden_mma:263 design.math_reshape.rewrite[5].symbol.mM.meaning

Remainder running maximum.

## t_dfbb9de51298a613 | math.meaning | 40 chars | 1 declarations

probe_hidden_mma:94 design.math_reshape.rewrite[0].symbol.K.meaning

Number of complete packed prefix groups.

## t_e0237298e9474c5b | math.meaning | 58 chars | 2 declarations

mma_direct:60 design.math_reshape.rewrite[0].symbol.valid.meaning; root:199 design.math_reshape.rewrite[0].symbol.valid.meaning

Per-chunk validity of gathered slots before score masking.

## t_e4735a4c9f34549b | math.meaning | 62 chars | 1 declarations

probe_hidden_mma:285 design.math_reshape.rewrite[6].symbol.ResultM.meaning

BF16 remainder output including locally computed padded heads.

## t_e47db3acfb9536c0 | math.meaning | 31 chars | 1 declarations

probe_hidden_mma:260 design.math_reshape.rewrite[5].symbol.AlphaM.meaning

Remainder row rescaling factor.

## t_e81cf6ab367885e2 | math.meaning | 36 chars | 4 declarations

probe_hidden_mma:90 design.math_reshape.rewrite[0].symbol.Q.meaning; probe_hidden_packed:91 design.math_reshape.rewrite[0].symbol.Q.meaning; probe_serial_packed:91 design.math_reshape.rewrite[0].symbol.Q.meaning; tc_packed:75 design.math_reshape.rewrite[0].symbol.Q.meaning

Packed query rows with zero padding.

## t_f05fc5e35a4f2a26 | math.meaning | 52 chars | 1 declarations

probe_hidden_mma:261 design.math_reshape.rewrite[5].symbol.SinkM.meaning

Base-2 remainder sink logits including padded heads.

## t_f124ba8fe6fb6790 | math.meaning | 52 chars | 1 declarations

probe_hidden_mma:232 design.math_reshape.rewrite[4].symbol.GatherM.meaning

Remainder cache rows including row-zero substitutes.

## t_f229d9bda8cbb02d | math.meaning | 62 chars | 2 declarations

mma_direct:90 design.math_reshape.rewrite[1].symbol.l.meaning; root:230 design.math_reshape.rewrite[1].symbol.l.meaning

Running sum of unrounded exponentials excluding the sink term.

## t_f2323c92c50babd3 | math.meaning | 59 chars | 3 declarations

probe_hidden_mma:129 design.math_reshape.rewrite[1].symbol.ProbeCount.meaning; probe_hidden_packed:136 design.math_reshape.rewrite[1].symbol.ProbeCount.meaning; probe_serial_packed:135 design.math_reshape.rewrite[1].symbol.ProbeCount.meaning

Actual integer numerator accumulated for one usable sample.

## t_f3b109fd9b7b6f17 | math.meaning | 33 chars | 5 declarations

probe_hidden_mma:176 design.math_reshape.rewrite[2].symbol.Prob.meaning; probe_hidden_packed:200 design.math_reshape.rewrite[2].symbol.Prob.meaning; probe_serial_packed:199 design.math_reshape.rewrite[2].symbol.Prob.meaning; tc_heads:92 design.math_reshape.rewrite[1].symbol.Prob.meaning; tc_packed:117 design.math_reshape.rewrite[1].symbol.Prob.meaning

BF16 probability operands for PV.

## t_f4499a5162384e88 | math.meaning | 37 chars | 2 declarations

mma_direct:57 design.math_reshape.rewrite[0].symbol.DQ.meaning; root:196 design.math_reshape.rewrite[0].symbol.DQ.meaning

Independent output-column partitions.

## t_ffb4593b92fae1c6 | math.meaning | 30 chars | 4 declarations

probe_hidden_mma:84 design.math_reshape.rewrite[0].symbol.MR.meaning; probe_hidden_packed:78 design.math_reshape.rewrite[0].symbol.MR.meaning; probe_serial_packed:78 design.math_reshape.rewrite[0].symbol.MR.meaning; tc_packed:69 design.math_reshape.rewrite[0].symbol.MR.meaning

Matrix rows including padding.

## t_155fe48b13f9c2cd | math.name | 46 chars | 3 declarations

mma_direct:63 design.math_reshape.rewrite[1].name; probe_serial_mma:104 design.math_reshape.rewrite[3].name; root:202 design.math_reshape.rewrite[1].name

FP32 recurrence with BF16 probability operands

## t_20c1e7dfb450ce46 | math.name | 23 chars | 1 declarations

root:237 design.math_reshape.rewrite[2].name

Materialize split state

## t_2219dad35724ac17 | math.name | 52 chars | 1 declarations

tc_heads:102 design.math_reshape.rewrite[2].name

Add the sink and overwrite the complete output block

## t_2b544b3414cd33f4 | math.name | 59 chars | 3 declarations

probe_hidden_mma:96 design.math_reshape.rewrite[1].name; probe_hidden_packed:103 design.math_reshape.rewrite[1].name; probe_serial_packed:103 design.math_reshape.rewrite[1].name

Count sampled updates and retain the observed host decision

## t_559fc3929351c31d | math.name | 40 chars | 1 declarations

root:249 design.math_reshape.rewrite[3].name

Merge split states and add the sink once

## t_7559293ab1021ea3 | math.name | 53 chars | 1 declarations

probe_serial_mma:87 design.math_reshape.rewrite[2].name

Select from the actually observed positive statistics

## t_7bc7fa98922ebccb | math.name | 45 chars | 2 declarations

mma_direct:97 design.math_reshape.rewrite[2].name; probe_serial_mma:138 design.math_reshape.rewrite[4].name

Add the sink and normalize directly to output

## t_92023491df518550 | math.name | 46 chars | 1 declarations

probe_hidden_mma:213 design.math_reshape.rewrite[4].name

Compact each remaining token into wider chunks

## t_a5f512bd3389c67c | math.name | 59 chars | 4 declarations

probe_hidden_mma:193 design.math_reshape.rewrite[3].name; probe_hidden_packed:234 design.math_reshape.rewrite[3].name; probe_serial_packed:232 design.math_reshape.rewrite[3].name; tc_packed:132 design.math_reshape.rewrite[2].name

Write zero for an empty union or normalize real packed rows

## t_b03bded00506dae4 | math.name | 49 chars | 1 declarations

tc_heads:31 design.math_reshape.rewrite[0].name

Compact reserved slots without deduplicating them

## t_bb46912429104c80 | math.name | 45 chars | 5 declarations

probe_hidden_mma:138 design.math_reshape.rewrite[2].name; probe_hidden_packed:153 design.math_reshape.rewrite[2].name; probe_serial_packed:152 design.math_reshape.rewrite[2].name; tc_heads:55 design.math_reshape.rewrite[1].name; tc_packed:79 design.math_reshape.rewrite[1].name

Two QK partials and a lazy maximum recurrence

## t_dde4a9cec136bc37 | math.name | 53 chars | 1 declarations

probe_serial_mma:60 design.math_reshape.rewrite[1].name

Sample the encoded key union and repeated memberships

## t_ebfdbdaf1b2038ff | math.name | 52 chars | 3 declarations

mma_direct:40 design.math_reshape.rewrite[0].name; probe_serial_mma:37 design.math_reshape.rewrite[0].name; root:176 design.math_reshape.rewrite[0].name

Compact reserved slots and retain masked gather work

## t_f3195d1588fc1107 | math.name | 50 chars | 1 declarations

probe_hidden_mma:235 design.math_reshape.rewrite[5].name

One full-latent QK chain and eager maximum updates

## t_f4b6d688b1bf88c2 | math.name | 43 chars | 1 declarations

probe_hidden_mma:275 design.math_reshape.rewrite[6].name

Normalize and overwrite every remainder row

## t_f5cc481418c4f447 | math.name | 57 chars | 4 declarations

probe_hidden_mma:48 design.math_reshape.rewrite[0].name; probe_hidden_packed:45 design.math_reshape.rewrite[0].name; probe_serial_packed:45 design.math_reshape.rewrite[0].name; tc_packed:36 design.math_reshape.rewrite[0].name

Build a multiplicity-preserving union and pack query rows

## t_088c79537491c96e | math.step | 10 chars | 1 declarations

probe_hidden_mma:234 design.math_reshape.rewrite[5].step

mma_online

## t_0980757eba0effb0 | math.step | 5 chars | 1 declarations

root:248 design.math_reshape.rewrite[3].step

merge

## t_7d3629a62626eb0c | math.step | 5 chars | 3 declarations

probe_hidden_mma:95 design.math_reshape.rewrite[1].step; probe_hidden_packed:102 design.math_reshape.rewrite[1].step; probe_serial_packed:102 design.math_reshape.rewrite[1].step

probe

## t_7f0ea405f1315f82 | math.step | 6 chars | 8 declarations

mma_direct:39 design.math_reshape.rewrite[0].step; probe_hidden_mma:47 design.math_reshape.rewrite[0].step; probe_hidden_packed:44 design.math_reshape.rewrite[0].step; probe_serial_mma:36 design.math_reshape.rewrite[0].step; probe_serial_packed:44 design.math_reshape.rewrite[0].step; root:175 design.math_reshape.rewrite[0].step; tc_heads:30 design.math_reshape.rewrite[0].step; tc_packed:35 design.math_reshape.rewrite[0].step

gather

## t_88a04e4d62745807 | math.step | 8 chars | 8 declarations

mma_direct:96 design.math_reshape.rewrite[2].step; probe_hidden_mma:192 design.math_reshape.rewrite[3].step; probe_hidden_packed:233 design.math_reshape.rewrite[3].step; probe_serial_mma:137 design.math_reshape.rewrite[4].step; probe_serial_packed:231 design.math_reshape.rewrite[3].step; root:236 design.math_reshape.rewrite[2].step; tc_heads:101 design.math_reshape.rewrite[2].step; tc_packed:131 design.math_reshape.rewrite[2].step

finalize

## t_9a2e9bf114a39440 | math.step | 12 chars | 1 declarations

probe_hidden_mma:274 design.math_reshape.rewrite[6].step

mma_finalize

## t_a778bcbb097636eb | math.step | 11 chars | 1 declarations

probe_serial_mma:59 design.math_reshape.rewrite[1].step

probe_count

## t_b7afdd70bf5771f9 | math.step | 12 chars | 1 declarations

probe_serial_mma:86 design.math_reshape.rewrite[2].step

probe_select

## t_be34a28632a94b43 | math.step | 10 chars | 1 declarations

probe_hidden_mma:212 design.math_reshape.rewrite[4].step

mma_gather

## t_cd9bee15f13aa998 | math.step | 6 chars | 8 declarations

mma_direct:62 design.math_reshape.rewrite[1].step; probe_hidden_mma:137 design.math_reshape.rewrite[2].step; probe_hidden_packed:152 design.math_reshape.rewrite[2].step; probe_serial_mma:103 design.math_reshape.rewrite[3].step; probe_serial_packed:151 design.math_reshape.rewrite[2].step; root:201 design.math_reshape.rewrite[1].step; tc_heads:54 design.math_reshape.rewrite[1].step; tc_packed:78 design.math_reshape.rewrite[1].step

online

## t_0bd552f6526e9ce7 | math.thesis | 320 chars | 1 declarations

root:169 design.math_reshape.thesis

Stream compacted SWA/compressed slot chunks through an online base-2 softmax, sharing each gathered latent across heads and between QK and PV. BF16 probabilities enable the second matrix product; KV splits trade shorter recurrences for FP32 partial storage and a separate merge, while output-column partitions repeat QK.

## t_3a4bbaf747b6fccf | math.thesis | 314 chars | 1 declarations

mma_direct:33 design.math_reshape.thesis

Stream compacted SWA/compressed slot chunks through an online base-2 softmax, sharing each gathered latent across heads and between QK and PV. BF16 probabilities enable the second matrix product. One recurrence covers the entire key range and normalizes directly to BF16 output; output-column partitions repeat QK.

## t_500f162c13d6f799 | math.thesis | 378 chars | 1 declarations

tc_packed:28 design.math_reshape.thesis

Pack consecutive queries into one matrix and share a pool-major union of their selected latents, retaining each query's multiplicities through membership bytes. Chunked base-2 softmax avoids full attention matrices; two QK partials and BF16 probabilities retain the tensor-core path, at the cost of union construction, padded matrix rows and a runtime-dependent reduction order.

## t_89d656cb3f604593 | math.thesis | 304 chars | 1 declarations

probe_serial_mma:30 design.math_reshape.thesis

Sample index overlap before running unsplit online attention. The rejected probe adds index processing and eight published statistics; attention shares each gathered latent across padded heads and QK/PV, rounds probabilities to BF16, and normalizes directly into the output without split materialization.

## t_90614d29e7afff96 | math.thesis | 300 chars | 1 declarations

tc_heads:24 design.math_reshape.thesis

Stream slot chunks through a base-2 online softmax and normalize directly. Two independent half-latent QK sums shorten each accumulation chain but add a score addition; a lagging maximum can avoid value rescaling, while BF16 probabilities change PV precision and require two score partials per chunk.

## t_9cf633a4e4f9f67e | math.thesis | 304 chars | 1 declarations

probe_serial_packed:38 design.math_reshape.thesis

Sample the packing overhead before evaluating the full batch with a shared pool-major union, retaining each query's multiplicities. The probe adds index counting and host observations; attention retains two QK partials, BF16 probabilities and chunked base-2 softmax, with padding and runtime union order.

## t_a7bac05ffbd6987f | math.thesis | 325 chars | 1 declarations

probe_hidden_mma:41 design.math_reshape.thesis

The committed prefix uses a multiplicity-preserving packed union and two QK partials. After a rejected observed probe mean, each remaining token uses compact slot order and a single QK chain; both paths use BF16 probabilities and direct normalization. The probe counts only control data and never supplies attention operands.

## t_d9b40bae78a5ff7a | math.thesis | 300 chars | 1 declarations

probe_hidden_packed:38 design.math_reshape.thesis

Count sampled packing overhead and use the observed host mean to select the remainder, while the prefix already uses packing. Attention retains the pool-major multiplicity-preserving union, two QK partials, BF16 probabilities and chunked base-2 softmax, including padding and the runtime union order.

## t_6814f418bb5e2c79 | metadata.describe | 133 chars | 8 declarations

mma_direct:3 meta_data.describe; probe_hidden_mma:3 meta_data.describe; probe_hidden_packed:3 meta_data.describe; probe_serial_mma:3 meta_data.describe; probe_serial_packed:3 meta_data.describe; root:3 meta_data.describe; tc_heads:3 meta_data.describe; tc_packed:3 meta_data.describe

Joint sparse attention over prepared sliding-window and compressed shared key/value latents, with a per-head zero-value softmax sink.

## t_0a6194ea8b4a5bab | pipeline.thesis | 351 chars | 1 declarations

tc_heads:170 design.tile_pipeline.thesis

Four loader warps feed a two-slot KV ring while four softmax warps advance independently. Two softmax warps issue disjoint QK halves and another issues PV; separate completion groups permit QK of the next chunk during prior PV. KV reuse waits for PV, score reuse waits for all softmax publishers, and the final PV releases the ring for output staging.

## t_0cc3555971264192 | pipeline.thesis | 329 chars | 1 declarations

probe_hidden_mma:486 design.tile_pipeline.thesis

Probe CTAs count independently while the packed prefix may execute. Prefix uses separate loader and softmax roles, a two-slot KV ring and TMEM; each remainder CTA uses the same eight warps for a single-slot gather, QK, softmax and PV. Both paths reuse their own retired KV storage for output and retain no state across worktiles.

## t_14068ca706ff2fc1 | pipeline.thesis | 329 chars | 1 declarations

root:375 design.tile_pipeline.thesis

The same eight warps gather and compute. Multi-stage rings allow future KV transfers during QK, softmax and PV; a single-stage ring waits for each chunk before computing. Publication and reuse barriers limit advancement. Combine can overlap its first vector loads with the statistics reduction; every worker handles one worktile.

## t_179c9810a705b774 | pipeline.thesis | 241 chars | 1 declarations

probe_serial_mma:240 design.tile_pipeline.thesis

Probe workers overlap a one-item index prefetch with shared hash updates and publish once. Attention workers cache Q, then serialize each KV gather with QK/softmax/PV; CTA publication and reader-retirement barriers protect single-slot reuse.

## t_24739344cd69eed9 | pipeline.thesis | 367 chars | 1 declarations

probe_serial_packed:391 design.tile_pipeline.thesis

Each probe CTA cooperatively counts one sample and publishes one mapped scalar; its local state dies before the independent attention CTAs begin. Attention builds its union in the idle KV ring, then overlaps four loader warps with four softmax warps. PV releases KV slots and probability publication releases scores; the final PV releases the ring for output staging.

## t_37fc4ec6e795e0a4 | pipeline.thesis | 386 chars | 1 declarations

probe_hidden_packed:462 design.tile_pipeline.thesis

Probe CTAs own independent counting state while prefix CTAs may run concurrently. Both attention phases use the same per-CTA pipeline with separate resource instances: union construction reuses the idle KV ring, four loader warps overlap four softmax warps, and final PV completion releases the ring for output staging. The two phases share no local buffers or completion notifications.

## t_5a8fd86e4a9797a9 | pipeline.thesis | 382 chars | 1 declarations

tc_packed:228 design.tile_pipeline.thesis

Cooperatively build the union using the still-idle KV ring for a hash table and staged indices. Retain keys and membership separately, then feed the two-slot ring with four loader warps while four softmax warps advance QK/PV. KV reuse waits for PV, score reuse waits for all probability publishers, and final PV releases the ring for output staging; empty unions skip this pipeline.

## t_b933320c70fe66df | pipeline.thesis | 317 chars | 1 declarations

mma_direct:167 design.tile_pipeline.thesis

The same eight warps gather and compute. Multi-stage rings allow future KV transfers during QK, softmax and PV; a single-stage ring waits for each chunk before computing. Publication and reuse barriers limit advancement. After the last PV read, the ring becomes BF16 output staging; every worker handles one worktile.

## t_00ef85a6ad8ed33c | pipeline.verb | 305 chars | 2 declarations

probe_hidden_mma:671 design.tile_pipeline.phase.prefix.role.Setup.info.verb.zero; probe_hidden_packed:702 design.tile_pipeline.phase.prefix.role.Setup.info.verb.zero

For the empty union, all eight warps overwrite every real output row with finalize's zero result. Q copies have completed in build:Union. This path acquires no TMEM and initializes no transfer barriers; it creates none of the v- or c-indexed actions. Stores complete by the containing kernel's completion.

## t_0a3ffad86c7d6de0 | pipeline.verb | 137 chars | 1 declarations

probe_serial_mma:368 design.tile_pipeline.phase.attention.role.Main.info.verb.retire

A CTA barrier joins all PV readers before KVStage, valid, Prob and control may be overwritten. Q_cached and IndexShared remain unchanged.

## t_0db5a140db8dfb2d | pipeline.verb | 151 chars | 2 declarations

mma_direct:275 design.tile_pipeline.phase.attention.role.Main.info.verb.retire; root:486 design.tile_pipeline.phase.attention.role.Main.info.verb.retire

A CTA barrier joins all PV readers before rotating the ring. It protects the next writes to KV, validity, Prob and control; Q_cached remains unchanged.

## t_1178941171cfd1ae | pipeline.verb | 326 chars | 1 declarations

root:532 design.tile_pipeline.phase.combine.role.Merge.info.verb.fold

Consume VectorPrefetch with merge's value predication; then fetch remaining split vectors four at a time and finally singly, with merge's address substitution. Keep one float4 accumulator per thread. For G>1 publish all group vectors into GroupPartial and barrier before group zero folds them; G=1 uses its registers directly.

## t_1932e36125e849f0 | pipeline.verb | 751 chars | 1 declarations

probe_hidden_mma:998 design.tile_pipeline.phase.remainder.role.Main.info.verb.prepare

Issue the query copies into Qm, zero padded heads and commit the query group. Initialize control from mma_online and publish the prefetched indices into IndexShared, each store consuming its loaded register. Wait for query copies and rendezvous to publish Qm, control and indices, ending the metadata slab reads. Only then enqueue chunk zero into the single ring slot. Enqueueing resolves selected IDs in batches from IndexShared, records ValidM, issues 16-byte row copies and commits once, including the row-zero substitutes from mma_gather. Cache all 32 query fragments in registers, with one complete Q replica per key-group warp, and zero AccM. done guarantees Q is ready and chunk zero has been issued, while its KV copies may still be in flight.

## t_1fc0417994152166 | pipeline.verb | 211 chars | 1 declarations

probe_serial_mma:349 design.tile_pipeline.phase.attention.role.Main.info.verb.await

Enqueue Gather(c) unless c=0, whose copy group was issued in prepare. Wait for all copy groups and use a CTA barrier to publish Gather(c) and valid(c). Later chunks reuse storage released by the previous retire.

## t_219bb7cb7e1dd61e | pipeline.verb | 1252 chars | 1 declarations

mma_direct:233 design.tile_pipeline.phase.attention.role.Main.info.verb.prepare

Stage Q with zero padded heads and commit one copy group. Use Q_shared when QHeads>0, otherwise alias ring slot zero as Q_ring. If QPRE and Early=0, enqueue chunk zero in slot zero before waiting for Q. Initialize control state, then publish cached indices into IndexShared when present. Each index store depends on its register load result; query issuance need not wait for it. Wait leaving one group outstanding only in that pre-issued case, otherwise wait all; a CTA barrier publishes Q, control and index storage and ends metadata reads. If QPRE and Early=1, only now enqueue chunk zero, even when the width disables index caching. Cache the first QKS 16-wide Q slices in registers, duplicated across key groups. When Q occupies ring slot zero, a CTA barrier after all Q fragment reads frees it. Q_cached and any uncached Q_shared slices remain readable for the whole recurrence. Zero Acc. For j=(1 if QPRE else 0)..Stages-2, enqueue chunk j into slot j if j<C(t), otherwise commit an empty group. Enqueueing a chunk resolves slot IDs in batches before issuing 16-byte row copies, records validity and commits once; invalid slots still copy row zero of their pool. Completion of prepare means Q is ready and priming is issued, not that KV is ready.

## t_21aaf700b1405fad | pipeline.verb | 146 chars | 1 declarations

probe_hidden_mma:1027 design.tile_pipeline.phase.remainder.role.Main.info.verb.retire

Rendezvous all PV readers before the next gather overwrites the single KV slot, validity, probabilities or control. Query caches remain unchanged.

## t_24e7c639b2f6af9f | pipeline.verb | 566 chars | 1 declarations

probe_hidden_mma:989 design.tile_pipeline.phase.remainder.role.Main.info.verb.metadata

Work is the assigned batch token t. UM=KM/8=16: the eight warps each own HB heads by UM keys for QK and HB heads by D/8 columns for PV. Issue whole-row index loads into IndexPrefetch, including zero values for inactive lanes. The packed gate gives IndexWidth<=640, so index caching is always enabled here. Stage request boundaries and this token's topk in RequestBounds, and sequence lengths in RequestLengths; B<=32 fits the slabs. Rendezvous before resolving the request and CM(t). Index loads may remain in flight at done; metadata does not consume their results.

## t_2c322ee64aeceeef | pipeline.verb | 379 chars | 1 declarations

tc_heads:330 design.tile_pipeline.phase.attention.role.Soft0.info.verb.stage

After the final PV, compute finalize's inverse from the lower key-half state and sink2, share inv across the warp, read both output halves and stage this warp's BF16 Result rows. OutputStage aliases the retired KV ring with a D+8 BF16-element row pitch. done ends these TMEM reads and publishes the staged rows; there is no wait for the loader's empty tail commit before staging.

## t_2ecd0ace9150fb9d | pipeline.verb | 330 chars | 1 declarations

probe_serial_mma:353 design.tile_pipeline.phase.attention.role.Main.info.verb.score

U=16. Each of eight warps forms HB by U scores from Q_cached and its distinct key group in Gather(c), using 32 synchronous 16-wide QK products. Scale, mask and publish per-warp row maxima into RowMax. Barrier before row-owner lanes fold groups and update m and alpha; barrier again to publish the new control values to every warp.

## t_317d756928319ba1 | pipeline.verb | 220 chars | 1 declarations

root:461 design.tile_pipeline.phase.attention.role.Main.info.verb.await

For Stages=1 enqueue c unless c=c0 and QPRE, then wait all groups. For Stages>1 wait until all but the newest Stages-2 groups have completed. A CTA barrier then makes current Gather(c) and valid(c) readable by all warps.

## t_341eefb3473b5f10 | pipeline.verb | 215 chars | 1 declarations

probe_hidden_mma:1023 design.tile_pipeline.phase.remainder.role.Main.info.verb.value

Rescale all AccM rows and execute synchronous PV using ProbM and the same GatherM(c). Each warp owns all HB rows and D/8 output columns. All matrix reads finish by done; AccM remains in registers for the next chunk.

## t_3452d677df32fbfa | pipeline.verb | 259 chars | 1 declarations

tc_heads:280 design.tile_pipeline.phase.attention.role.Loader.info.verb.publish

Rendezvous all loader warps after the current gather instructions and validity stores have been issued; publish valid(c) with one CTA-level arrival and the required proxy ordering. This does not wait for Gather(c) to land. It precedes issuing the next gather.

## t_3510b50a4e27687a | pipeline.verb | 260 chars | 14 declarations

probe_hidden_mma:730 design.tile_pipeline.phase.prefix.role.Soft0.info.verb.rescale; probe_hidden_mma:793 design.tile_pipeline.phase.prefix.role.Soft1.info.verb.rescale; probe_hidden_mma:846 design.tile_pipeline.phase.prefix.role.Soft2.info.verb.rescale; probe_hidden_mma:890 design.tile_pipeline.phase.prefix.role.Soft3.info.verb.rescale; probe_hidden_packed:766 design.tile_pipeline.phase.prefix.role.Soft0.info.verb.rescale; probe_hidden_packed:830 design.tile_pipeline.phase.prefix.role.Soft1.info.verb.rescale; probe_hidden_packed:888 design.tile_pipeline.phase.prefix.role.Soft2.info.verb.rescale; probe_hidden_packed:937 design.tile_pipeline.phase.prefix.role.Soft3.info.verb.rescale; probe_serial_packed:695 design.tile_pipeline.phase.attention.role.Soft0.info.verb.rescale; probe_serial_packed:759 design.tile_pipeline.phase.attention.role.Soft1.info.verb.rescale; probe_serial_packed:816 design.tile_pipeline.phase.attention.role.Soft2.info.verb.rescale; probe_serial_packed:864 design.tile_pipeline.phase.attention.role.Soft3.info.verb.rescale; tc_heads:312 design.tile_pipeline.phase.attention.role.Soft0.info.verb.rescale; tc_packed:462 design.tile_pipeline.phase.attention.role.Soft0.info.verb.rescale

For c=0 do not read or write Acc. For c>0 wait for the preceding PV even if every scale is one; then conditionally rescale this warp's rows of both output halves. The group predicate is uniform across the warp, and all TMEM reads and writes finish before done.

## t_39938a619bb9d32e | pipeline.verb | 246 chars | 1 declarations

root:537 design.tile_pipeline.phase.combine.role.Merge.info.verb.store

The first 128 threads multiply by inv, convert paired values to BF16 and overwrite the complete output row. No scratch is reset or retained as numerical state for a later invocation. This CTA has no next worktile; final visibility is kernel_done.

## t_41f3c943236bf466 | pipeline.verb | 417 chars | 2 declarations

mma_direct:261 design.tile_pipeline.phase.attention.role.Main.info.verb.score; root:472 design.tile_pipeline.phase.attention.role.Main.info.verb.score

Eight warps form WR1 head groups by WC1 key groups. Each computes MT1 by U scores, reading Q_cached where present and the remaining Q from Q_shared, plus Gather(c). Complete the synchronous QK products, scale and mask scores, then publish per-warp row maxima into RowMax. A CTA barrier precedes row-owner threads folding the groups and updating m and alpha; another barrier makes those updates available to all warps.

## t_49acdab3b5bd6252 | pipeline.verb | 314 chars | 5 declarations

probe_hidden_mma:742 design.tile_pipeline.phase.prefix.role.Soft0.info.verb.pv.meaning; probe_hidden_packed:778 design.tile_pipeline.phase.prefix.role.Soft0.info.verb.pv.meaning; probe_serial_packed:707 design.tile_pipeline.phase.attention.role.Soft0.info.verb.pv.meaning; tc_heads:324 design.tile_pipeline.phase.attention.role.Soft0.info.verb.pv.meaning; tc_packed:474 design.tile_pipeline.phase.attention.role.Soft0.info.verb.pv.meaning

Lane zero issues both 256-column PV output halves. Each starts from zero on chunk zero; later chunks accumulate into the rescaled Acc. Its per-issuer completion group covers all prior PV operations, independently of both QK issuers. done guarantees Acc is written and all reads of Prob(c) and Gather(c) have ended.

## t_5af12f884380979d | pipeline.verb | 1373 chars | 1 declarations

probe_serial_packed:446 design.tile_pipeline.phase.probe.role.Main.info.verb.produce

For an in-range sample, the first GT participants resolve SelectedCount, SwaCount and RequestId into ProbeMeta; rendezvous before checking usability. An out-of-range or unusable sample writes the final -1 from one participant, without initializing or reading the hash state. A usable sample clears ProbeHash, ProbeMembership and ProbeCount, then rendezvous before inserting. All participants share a 4096-entry open-addressed table of pool-tagged keys. A slot has one membership byte for up to eight queries. Apply probe's Creator/First contribution rule to LocalCount: atomicCAS publishes a new entry before its creator sets the membership bit; matching participants use the previous bit returned by atomicOr, without waiting for the creator. Each nonzero local count is flushed into ProbeCount once. Each participant prefetches one next selected index into PrefetchedId before processing the current key; invalid IDs skip the hash operation. No index row, query or latent is staged for reuse by the attention phase. After the counting rendezvous, form probe's scalar ratio and write only this sample's mapped slot once from one participant. There is no device reduction across samples. This publication does not assert that every CTA has finished; all local reads and stores retire by probe.kernel_done. The static resources are per CTA, and no next worktile reuses them.

## t_5c6336f18a8d20d1 | pipeline.verb | 466 chars | 13 declarations

probe_hidden_mma:748 design.tile_pipeline.phase.prefix.role.Soft0.info.verb.stage; probe_hidden_mma:803 design.tile_pipeline.phase.prefix.role.Soft1.info.verb.stage; probe_hidden_mma:856 design.tile_pipeline.phase.prefix.role.Soft2.info.verb.stage; probe_hidden_mma:900 design.tile_pipeline.phase.prefix.role.Soft3.info.verb.stage; probe_hidden_packed:784 design.tile_pipeline.phase.prefix.role.Soft0.info.verb.stage; probe_hidden_packed:840 design.tile_pipeline.phase.prefix.role.Soft1.info.verb.stage; probe_hidden_packed:898 design.tile_pipeline.phase.prefix.role.Soft2.info.verb.stage; probe_hidden_packed:947 design.tile_pipeline.phase.prefix.role.Soft3.info.verb.stage; probe_serial_packed:713 design.tile_pipeline.phase.attention.role.Soft0.info.verb.stage; probe_serial_packed:769 design.tile_pipeline.phase.attention.role.Soft1.info.verb.stage; probe_serial_packed:826 design.tile_pipeline.phase.attention.role.Soft2.info.verb.stage; probe_serial_packed:874 design.tile_pipeline.phase.attention.role.Soft3.info.verb.stage; tc_packed:480 design.tile_pipeline.phase.attention.role.Soft0.info.verb.stage

After the final PV, compute finalize's inverse from the lower key-half state and sink2, share inv across the warp, read both output halves and stage only this warp's real BF16 Result rows. OutputStage aliases the retired KV ring with a D+8 BF16-element row pitch. Padded rows participate in convergent TMEM reads but perform no BF16 conversion or staging. done ends these reads and publishes the staged rows; staging does not wait for the loader's empty tail commit.

## t_6037be0b76f7e544 | pipeline.verb | 243 chars | 2 declarations

mma_direct:271 design.tile_pipeline.phase.attention.role.Main.info.verb.value; root:482 design.tile_pipeline.phase.attention.role.Main.info.verb.value

Rescale Acc, then synchronous PV reads Prob and the same Gather(c) as QK. Each warp owns all HB rows and D/(8*DQ) distinct output columns in this CTA's slice. Acc remains in registers across chunks; all probability/KV matrix reads finish here.

## t_62aa734bab98b548 | pipeline.verb | 369 chars | 1 declarations

tc_heads:432 design.tile_pipeline.phase.attention.role.Finish.info.verb.store

Rejoin the same eight warps at a CTA barrier after all Result rows are staged. Cooperatively copy complete contiguous output rows from OutputStage into out. TMEM may be deallocated by its owner after that barrier while other warps copy, since copy-out no longer reads TMEM. Global stores can remain pending until attention.kernel_done. No worker takes another worktile.

## t_7469bb278668802e | pipeline.verb | 860 chars | 2 declarations

probe_serial_packed:618 design.tile_pipeline.phase.attention.role.Setup.info.verb.prepare; tc_packed:385 design.tile_pipeline.phase.attention.role.Setup.info.verb.prepare

Work(v) is the nonempty group's execution setup; RG=16. Acquire 512 TMEM columns, initialize m_initial, l_initial, sink2 and alpha from online, and initialize completion protocols. Only m_initial is read into the replicated register state; alpha is first initialized to one. Q completion, a proxy fence and CTA rendezvous publish Q and control before done. Acc is initialized by the first PV, not by reading new TMEM. Acc's two 256-column output halves occupy disjoint lane halves of columns 0..255; each Partial uses its reserved score columns. All four score regions remain reserved throughout this nonempty worktile; unused lanes are unread. The source aliases allocation metadata and union keys with live transfer barriers; this design retains the separately budgeted objects and their lifetimes without depending on those erroneous implementation aliases.

## t_75f37a632e73d41d | pipeline.verb | 353 chars | 1 declarations

probe_serial_mma:332 design.tile_pipeline.phase.attention.role.Main.info.verb.metadata

Work is token t. Issue the entire index row into IndexPrefetch before staging request boundaries, lengths and topk in the future RowMax/RowSum slabs. A CTA barrier publishes metadata; every lane then resolves the request and C(t). B<=32 and IndexWidth<=640 here, so both caching paths are active. Index loads may remain in flight at metadata completion.

## t_7a10a10dce205cc9 | pipeline.verb | 190 chars | 1 declarations

tc_heads:285 design.tile_pipeline.phase.attention.role.Loader.info.verb.finish

Finish the loader issue loop, including its empty trailing copy commit. Transfers denotes this issue path, not a transfer drain; real transfers are retired by the QK and PV completion chain.

## t_885928614800b8e8 | pipeline.verb | 349 chars | 7 declarations

probe_hidden_mma:778 design.tile_pipeline.phase.prefix.role.Soft1.info.verb.qk.meaning; probe_hidden_mma:831 design.tile_pipeline.phase.prefix.role.Soft2.info.verb.qk.meaning; probe_hidden_packed:815 design.tile_pipeline.phase.prefix.role.Soft1.info.verb.qk.meaning; probe_hidden_packed:873 design.tile_pipeline.phase.prefix.role.Soft2.info.verb.qk.meaning; probe_serial_packed:744 design.tile_pipeline.phase.attention.role.Soft1.info.verb.qk.meaning; probe_serial_packed:801 design.tile_pipeline.phase.attention.role.Soft2.info.verb.qk.meaning; tc_packed:511 design.tile_pipeline.phase.attention.role.Soft1.info.verb.qk.meaning

Lane zero issues the half a named by Partial(c,a), covering all MR rows and KC keys. The first product overwrites its score region. Both issuers commit separately to a two-arrival QK completion notification; both done events use that shared notification, covering the partial writes and Q/Gather reads. Neither issuer shares the PV completion group.

## t_88a93b29d7b350a9 | pipeline.verb | 260 chars | 4 declarations

probe_hidden_mma:698 design.tile_pipeline.phase.prefix.role.Loader.info.verb.publish; probe_hidden_packed:731 design.tile_pipeline.phase.prefix.role.Loader.info.verb.publish; probe_serial_packed:660 design.tile_pipeline.phase.attention.role.Loader.info.verb.publish; tc_packed:427 design.tile_pipeline.phase.attention.role.Loader.info.verb.publish

Rendezvous all loader warps after the current gather instructions and membership stores have been issued; publish Mask(c) with one CTA-level arrival and the required proxy ordering. This does not wait for Gather(c) to land. It precedes issuing the next gather.

## t_88ea647797ffcd83 | pipeline.verb | 644 chars | 4 declarations

probe_hidden_mma:688 design.tile_pipeline.phase.prefix.role.Loader.info.verb.load.meaning; probe_hidden_packed:721 design.tile_pipeline.phase.prefix.role.Loader.info.verb.load.meaning; probe_serial_packed:650 design.tile_pipeline.phase.attention.role.Loader.info.verb.load.meaning; tc_packed:417 design.tile_pipeline.phase.attention.role.Loader.info.verb.load.meaning

Read UnionKey and UnionMask through the two-block logical-to-physical mapping, write Mask(c), and issue the complete Gather(c) tile. Unless mixed(c), each of 128 loaders issues one 512-byte TMA gather4 for four rows in one 64-element latent plane using the first entry's pool map. Each issuer arms its own transaction bytes before that request. The single possible mixed-pool chunk uses batched keys and 16-byte asynchronous copies with per-loader completion arrivals. Both forms transfer the padding rows specified by gather and fill the same swizzled tile. done guarantees all tile data has landed; issuing later actions need not wait for it.

## t_8c95a93d1f123a80 | pipeline.verb | 871 chars | 1 declarations

probe_hidden_mma:658 design.tile_pipeline.phase.prefix.role.Setup.info.verb.prepare

Work(v) is the nonempty group's execution setup; RG=16 for prefix. Acquire 512 TMEM columns, initialize m_initial, l_initial, sink2 and alpha from online, and initialize completion protocols. Only m_initial is read into the replicated register state; alpha is first initialized to one. Q completion, a proxy fence and CTA rendezvous publish Q and control before done. Acc is initialized by the first PV, not by reading new TMEM. Acc's two 256-column output halves occupy disjoint lane halves of columns 0..255; each Partial uses its reserved score columns. All four score regions remain reserved throughout this nonempty worktile; unused lanes are unread. The source aliases allocation metadata and union keys with live transfer barriers; this design retains the separately budgeted objects and their lifetimes without depending on those erroneous implementation aliases.

## t_904bec294caa375a | pipeline.verb | 455 chars | 1 declarations

root:465 design.tile_pipeline.phase.attention.role.Main.info.verb.enqueue

Future(c) is the prospective copy group for chunk c+Stages-1. For Stages>1 issue it into slot (c-c0+Stages-1)%Stages when below c1, otherwise commit an empty group. Stages=1 does nothing here. For Stages>1, done means the group was committed, not that the transfer landed; it can remain in flight through score, softmax and value. The target is initially unused or was released by the preceding iteration's retire barrier, so no current KV is overwritten.

## t_941e5cf716f74f3a | pipeline.verb | 349 chars | 1 declarations

tc_heads:357 design.tile_pipeline.phase.attention.role.Soft1.info.verb.qk.meaning

Lane zero issues the half a named by Partial(c,a), covering all HB rows and KC keys. The first product overwrites its score region. Both issuers commit separately to a two-arrival QK completion notification; both done events use that shared notification, covering the partial writes and Q/Gather reads. Neither issuer shares the PV completion group.

## t_94ba7630497e6a32 | pipeline.verb | 209 chars | 1 declarations

probe_serial_mma:259 design.tile_pipeline.phase.probe.role.Main.info.verb.inspect

Work is sample s. One lane per candidate token resolves n_a, its SWA count and request id into GroupMeta; a CTA barrier publishes these. All lanes determine Usable(s). An absent sample needs no metadata reads.

## t_9b2cab398347fde3 | pipeline.verb | 668 chars | 1 declarations

root:489 design.tile_pipeline.phase.attention.role.Main.info.verb.finish

After the last retire, every real KV copy has been waited for and consumed; only empty committed groups can remain. The source adds no final wait-all. Reinterpret the ring prefix as PartialStage. Stage the owned columns from Acc in FP32, barrier, then cooperatively copy that slice to global scratch. Identical m/l values are redundantly published by the dq CTAs; the source's non-atomic duplicate stores have no inter-CTA ordering. Global stores may still be pending when individual threads finish. Kernel completion provides the successor's visibility. This worker takes no second worktile and retains no Q, recurrence state or ring contents for another CTA or call.

## t_9c44e84b525e0a3f | pipeline.verb | 876 chars | 1 declarations

probe_hidden_packed:689 design.tile_pipeline.phase.prefix.role.Setup.info.verb.prepare

Work(v) is the nonempty group's execution setup; for both phases RG=16. Acquire 512 TMEM columns, initialize m_initial, l_initial, sink2 and alpha from online, and initialize completion protocols. Only m_initial is read into the replicated register state; alpha is first initialized to one. Q completion, a proxy fence and CTA rendezvous publish Q and control before done. Acc is initialized by the first PV, not by reading new TMEM. Acc's two 256-column output halves occupy disjoint lane halves of columns 0..255; each Partial uses its reserved score columns. All four score regions remain reserved throughout this nonempty worktile; unused lanes are unread. The source aliases allocation metadata and union keys with live transfer barriers; this design retains the separately budgeted objects and their lifetimes without depending on those erroneous implementation aliases.

## t_9cb95024f9e36cea | pipeline.verb | 214 chars | 1 declarations

probe_serial_mma:359 design.tile_pipeline.phase.attention.role.Main.info.verb.softmax

Compute online's E, retain unrounded sums and store BF16 Prob. Four-lane reductions publish RowSum; barrier makes Prob and RowSum visible. Row-owner lanes update l; another CTA barrier publishes it. Discard Scores.

## t_9df1cb2328329682 | pipeline.verb | 428 chars | 1 declarations

tc_heads:304 design.tile_pipeline.phase.attention.role.Soft0.info.verb.online

A softmax warp owns its consecutive RG head rows. Its two lane halves hold the two 32-key halves and duplicate m,l; read both Partial(c,a), add, scale, mask and compute online's E and state updates. Exchange the two key-half maxima and sums within the warp. Publish alpha once per row and rendezvous the warp. Scores are replaced in place by E; both score partials have been read when done. E remains live until publish:Prob(c).

## t_a3111e0f5352a330 | pipeline.verb | 1376 chars | 2 declarations

probe_hidden_mma:517 design.tile_pipeline.phase.probe.role.Main.info.verb.produce; probe_hidden_packed:517 design.tile_pipeline.phase.probe.role.Main.info.verb.produce

For an in-range sample, the first GT participants resolve SelectedCount, SwaCount and RequestId into ProbeMeta; rendezvous before checking usability. An out-of-range or unusable sample writes the final -1 from one participant, without initializing or reading the hash state. A usable sample clears ProbeHash, ProbeMembership and ProbeCount, then rendezvous before inserting. All participants share a 4096-entry open-addressed table of pool-tagged keys. A slot has one membership byte for up to eight queries. Apply probe's Creator/First contribution rule to LocalCount: atomicCAS publishes a new entry before its creator sets the membership bit; matching participants use the previous bit returned by atomicOr, without waiting for the creator. Each nonzero local count is flushed into ProbeCount once. Each participant prefetches one next selected index into PrefetchedId before processing the current key; invalid IDs skip the hash operation. No index row, query or latent is staged for reuse by either attention phase. After the counting rendezvous, form probe's scalar ratio and write only this sample's mapped slot once from one participant. There is no device reduction across samples. This publication does not assert that every CTA has finished; all local reads and stores retire by probe.kernel_done. The static resources are per CTA, and no next worktile reuses them.

## t_a3998489ce1480be | pipeline.verb | 275 chars | 1 declarations

probe_hidden_mma:1018 design.tile_pipeline.phase.remainder.role.Main.info.verb.softmax

Evaluate mma_online's unrounded exponentials, retain their sums and write BF16 ProbM. Four-lane reductions publish RowSum; rendezvous publishes both arrays. Row owners fold the eight sums and update lM; a second rendezvous completes the update. ScoresM can then be discarded.

## t_a4d8c5899712b143 | pipeline.verb | 686 chars | 1 declarations

probe_serial_mma:338 design.tile_pipeline.phase.attention.role.Main.info.verb.prepare

Stage Q, including zero padded heads, into Q_shared and commit one copy group. Initialize m, l and sink2; publish IndexPrefetch into IndexShared, each store depending on its own load result. Wait for Q and barrier to publish Q, indices and control, also ending the metadata slab reads. Now enqueue chunk zero into KVStage, recording valid and committing once. Cache all 32 query slices in Q_cached, one copy per key group, while that gather can remain in flight. Zero Acc. Enqueueing resolves slot IDs in batches before issuing 16-byte latent copies; invalid and padded slots still read row zero of their pool. Prepare completes after Q caching and gather issuance, before KV readiness.

## t_a6908f0f52de4401 | pipeline.verb | 793 chars | 1 declarations

mma_direct:220 design.tile_pipeline.phase.attention.role.Main.info.verb.metadata

Work denotes this CTA's assigned (t,hb,dq). For P_run.cfg 2/4/5/6/7/8, (Stages,QKS,Early) is respectively (2,16,0)/(3,32,0)/(3,32,1)/(1,0,0)/(1,32,0)/(1,32,1). The DQ IX variants retain the cfg5/cfg8 tuple. All use KSPLIT=1. WR1=HB/16 if QKS>0, otherwise max(HB/32,1); WC1=8/WR1; MT1=HB/WR1 and U=KC/WC1. QHeads=0 when QKS=32 and Stages>1, else HB. QPRE is true exactly for Stages=1 in this scope. Early=1 first issues whole-row index loads into IndexPrefetch if IndexWidth<=1152; wider rows use per-chunk global reads. It stages request boundaries, lengths and topk in the future RowMax/RowSum slabs, barriers, then resolves the request and C(t). Other configs scan metadata directly. Index loads may remain in flight when metadata is done; metadata resolution does not consume their results.

## t_ad8d4cbfd0586128 | pipeline.verb | 292 chars | 2 declarations

probe_serial_packed:631 design.tile_pipeline.phase.attention.role.Setup.info.verb.zero; tc_packed:398 design.tile_pipeline.phase.attention.role.Setup.info.verb.zero

For the empty union, all eight warps overwrite every real output row with finalize's zero result. Q copies have completed in build:Union. This path acquires no TMEM and initializes no transfer barriers; it creates none of the v- or c-indexed actions. Stores complete by attention.kernel_done.

## t_ae0252b0ed24072d | pipeline.verb | 266 chars | 1 declarations

probe_hidden_mma:1013 design.tile_pipeline.phase.remainder.role.Main.info.verb.score

Compute the synchronous QK products from Q_cached and GatherM(c), then scale and mask ScoresM. Publish per-warp maxima into RowMax and rendezvous. Row owners combine the eight groups and update mM and AlphaM, followed by a CTA rendezvous making both updates visible.

## t_ae5989245ab2cd71 | pipeline.verb | 258 chars | 14 declarations

probe_hidden_mma:735 design.tile_pipeline.phase.prefix.role.Soft0.info.verb.publish; probe_hidden_mma:798 design.tile_pipeline.phase.prefix.role.Soft1.info.verb.publish; probe_hidden_mma:851 design.tile_pipeline.phase.prefix.role.Soft2.info.verb.publish; probe_hidden_mma:895 design.tile_pipeline.phase.prefix.role.Soft3.info.verb.publish; probe_hidden_packed:771 design.tile_pipeline.phase.prefix.role.Soft0.info.verb.publish; probe_hidden_packed:835 design.tile_pipeline.phase.prefix.role.Soft1.info.verb.publish; probe_hidden_packed:893 design.tile_pipeline.phase.prefix.role.Soft2.info.verb.publish; probe_hidden_packed:942 design.tile_pipeline.phase.prefix.role.Soft3.info.verb.publish; probe_serial_packed:700 design.tile_pipeline.phase.attention.role.Soft0.info.verb.publish; probe_serial_packed:764 design.tile_pipeline.phase.attention.role.Soft1.info.verb.publish; probe_serial_packed:821 design.tile_pipeline.phase.attention.role.Soft2.info.verb.publish; probe_serial_packed:869 design.tile_pipeline.phase.attention.role.Soft3.info.verb.publish; tc_heads:317 design.tile_pipeline.phase.attention.role.Soft0.info.verb.publish; tc_packed:467 design.tile_pipeline.phase.attention.role.Soft0.info.verb.publish

Narrow this warp's E to its rows of Prob(c), fence the shared writes and TMEM rescale, then arrive once on probability readiness. done publishes only this warp's rows and releases its score reads. Four such arrivals jointly authorize PV and score-slot reuse.

## t_aee60cbbe354a819 | pipeline.verb | 1497 chars | 4 declarations

probe_hidden_mma:636 design.tile_pipeline.phase.prefix.role.Setup.info.verb.build; probe_hidden_packed:667 design.tile_pipeline.phase.prefix.role.Setup.info.verb.build; probe_serial_packed:596 design.tile_pipeline.phase.attention.role.Setup.info.verb.build; tc_packed:363 design.tile_pipeline.phase.attention.role.Setup.info.verb.build

Union denotes UnionKey, UnionMask and the two pool counts for this grp. All eight warps resolve each token's A and Klen into TokenExtent and issue Q copies, zeroing padded rows. Clear the 8192-word uint64 HashTable in the ring and the membership array. Stage the G complete sparse-index rows into IndexStage using asynchronous copies; wait for those copies and the earlier Q copies, then rendezvous before reading indices. Only the selected slot intervals feed the union; invalid markers contribute no entry. Each thread prefetches its next item before its current hash operation. Use open addressing: atomically install the pool-tagged key, allocate its canonical entry and publish the key/position pair in one atomic word. A same-key contender waits for that position publication. Atomic membership updates share the canonical entry across tokens; an already-set token bit allocates a private extra entry, as defined by gather. SWA entries allocate upward from slot zero and compressed entries downward from slot 2559. Logical entries [0,Ns) use their identical physical slots; entry k in [Ns,Nu(grp)) uses slot 2560-Ng+k-Ns. This reverses the compressed allocation order in the logical traversal; it does not move union bytes. A CTA rendezvous publishes keys, masks and counts and ends all HashTable and IndexStage reads. Their ring storage is now free for the first gather. The 65536-byte table plus 4*GT*IndexWidth index bytes fits the 131072-byte ring by the host's GT*IndexWidth<=2560 gate.

## t_b1e347dc95d76669 | pipeline.verb | 370 chars | 1 declarations

root:520 design.tile_pipeline.phase.combine.role.Merge.info.verb.prime.meaning

For a real head, each 128-thread group g issues four float4 vector loads from splits g+k*G before loading statistics, substituting register zero only for s>=S. VectorPrefetch keeps their results through the statistics reduction; done means these registers are ready. For padded heads all four pipe actions are no-ops with issue/done events and no memory reads or writes.

## t_b30065e495f0e573 | pipeline.verb | 404 chars | 1 declarations

probe_hidden_mma:1030 design.tile_pipeline.phase.remainder.role.Main.info.verb.finish

Every real KV copy has landed and its readers have retired. Compute mma_finalize's inverse, scale AccM and convert ResultM, including padded heads. Reuse the ring prefix as OutputM for the HB by D BF16 result. Rendezvous before cooperatively copying only real heads to out. Stores can remain pending until remainder.kernel_done. No scratch statistics are written and this worker takes no second worktile.

## t_b832db3785fb3a4a | pipeline.verb | 218 chars | 1 declarations

probe_hidden_mma:1009 design.tile_pipeline.phase.remainder.role.Main.info.verb.await

For c>0 enqueue the current GatherM and ValidM as prepare describes; c=0 uses the already committed group. Wait for all copy groups and rendezvous before any warp reads the current tile. No future chunk is issued here.

## t_b9c4cae8fed9c5ad | pipeline.verb | 397 chars | 1 declarations

root:526 design.tile_pipeline.phase.combine.role.Merge.info.verb.normalize

Load all S m/l pairs into SplitControl and barrier. Groups reduce maxima into GroupControl, barrier, fold M, publish weights in SplitControl, barrier. Groups reduce weighted normalizers, barrier; thread zero folds the group totals, adds the sink and publishes inv, barrier. G groups each contain 128 threads; thread offset v owns latent columns 4*v..4*v+3. Vector prefetch may overlap these steps.

## t_c5831293de6d8106 | pipeline.verb | 414 chars | 2 declarations

probe_serial_packed:902 design.tile_pipeline.phase.attention.role.Finish.info.verb.store; tc_packed:597 design.tile_pipeline.phase.attention.role.Finish.info.verb.store

Rejoin all eight warps at a CTA barrier after the real Result rows are staged. Cooperatively copy nrows contiguous rows from OutputStage into out, mapping row r to token t0+floor(r/H) and head r mod H. TMEM may be deallocated by its owner after that barrier while other warps copy, since copy-out no longer reads TMEM. Global stores can remain pending until attention.kernel_done. No worker takes another worktile.

## t_c700ff02b3e5d83c | pipeline.verb | 359 chars | 1 declarations

probe_serial_mma:284 design.tile_pipeline.phase.probe.role.Main.info.verb.publish

One lane writes the final ProbeRatio[s] directly to Gate[s], including -1 for unusable groups. No provisional ratio is published. The write may reach host polling before probe.kernel_done; host selection does not wait on this action's event. The kernel completion still precedes attention. Each CTA handles one sample and has no state carried to another call.

## t_c94a47792d86dcd4 | pipeline.verb | 193 chars | 4 declarations

probe_hidden_mma:703 design.tile_pipeline.phase.prefix.role.Loader.info.verb.finish; probe_hidden_packed:736 design.tile_pipeline.phase.prefix.role.Loader.info.verb.finish; probe_serial_packed:665 design.tile_pipeline.phase.attention.role.Loader.info.verb.finish; tc_packed:432 design.tile_pipeline.phase.attention.role.Loader.info.verb.finish

Finish the loader issue loop, including its empty trailing copy commit. Transfers(v) denotes this issue path, not a transfer drain; real transfers are retired by the QK and PV completion chain.

## t_cc36237c152cd712 | pipeline.verb | 939 chars | 1 declarations

tc_heads:245 design.tile_pipeline.phase.attention.role.Setup.info.verb.prepare

Work is the assigned (t,hb); RG=16. Resolve request metadata and C(t), cooperatively issue Q copies, initialize online's state and acquire 512 TMEM columns. m_initial and l_initial hold the initialization values; only m_initial is read into the replicated register state. Initialize the alpha slot to one; the first online update overwrites it. Q copies complete and a proxy fence plus CTA rendezvous publishes Q, control and initialized completion protocols before done. Acc is initialized by the first PV, not by reading new TMEM. Acc's two 256-column output halves occupy disjoint lane halves of columns 0..255; each Partial uses its reserved score columns with the unused lane positions left unread. All four score regions remain reserved throughout the worktile. The source's allocation-return word overlaps a live transfer barrier; the resource and dependency design here does not rely on that erroneous implementation address alias.

## t_d0a745644af7748c | pipeline.verb | 741 chars | 1 declarations

mma_direct:278 design.tile_pipeline.phase.attention.role.Main.info.verb.finish

After the last retire, every real KV copy has been waited for and consumed; only empty committed groups can remain. The source adds no final wait-all. Read m, l and sink2 to form finalize's inverse; each warp scales its Acc columns and converts to BF16 Result, including padded heads. Reuse the ring prefix as OutputStage, writing only this CTA's D/DQ columns in HB by D staging. A CTA barrier publishes Result, then cooperatively copy the owned columns of real heads into out. Global stores may remain pending when individual threads finish; attention.kernel_done provides output visibility. There are no scratch statistics or duplicate inter-CTA stores. This worker takes no second worktile and retains no numerical state for another call.

## t_d0d32037da9d47af | pipeline.verb | 624 chars | 1 declarations

tc_heads:269 design.tile_pipeline.phase.attention.role.Loader.info.verb.load.meaning

Resolve IDs, write valid(c) and issue the complete Gather(c) tile. Unless 0<A(t)-c*KC<KC and (A(t)-c*KC)%4!=0, each of the 128 loaders issues one 512-byte TMA gather4 for four rows in one 64-element latent plane, using the appropriate pool map. Each issuer arms its own transaction bytes before that request. The straddling chunk instead uses batched indices and 16-byte cp.async copies with per-loader completion arrivals. Both forms transfer row-zero substitutes and fill the same swizzled tile. done guarantees all tile data has landed through the transfer completion protocol; issuing later actions need not wait for it.

## t_d6e308ca7a7c7c5f | pipeline.verb | 239 chars | 2 declarations

mma_direct:267 design.tile_pipeline.phase.attention.role.Main.info.verb.softmax; root:478 design.tile_pipeline.phase.attention.role.Main.info.verb.softmax

Compute online's E, retain its unrounded sums, and store BF16 Prob. Four-lane reductions publish RowSum; a CTA barrier makes both Prob and RowSum visible. Row-owner threads update l, followed by a CTA barrier. Scores can then be discarded.

## t_d89b0e707d355934 | pipeline.verb | 1070 chars | 1 declarations

root:428 design.tile_pipeline.phase.attention.role.Main.info.verb.metadata

Work denotes this CTA's assigned (t,hb,s,dq); c0,c1 use online's abbreviation. For P.cfg 2/4/5/6/7/8, (Stages,QKS,Early) is respectively (2,16,0)/(3,32,0)/(3,32,1)/(1,0,0)/(1,32,0)/(1,32,1). The DQ IX variants retain the cfg5/cfg8 tuple. All use KSPLIT=1. WR1=HB/16 if QKS>0, otherwise max(HB/32,1); WC1=8/WR1; MT1=HB/WR1 and U=KC/WC1. QHeads=0 when QKS=32 and Stages>1, else HB. QPRE is true exactly for Stages=1 in this scope. Early=1 first issues whole-row index loads into IndexPrefetch if IndexWidth<=1152; wider rows use per-chunk global reads. It stages request boundaries, lengths and topk in the future RowMax/RowSum slabs, barriers, then resolves the request and c0,c1. Other configs scan metadata directly. Index loads may remain in flight when metadata is done; metadata resolution does not consume their results. Empty c0=c1 work publishes finalize's sentinel statistics and discards any prefetched indices without an explicit drain. No query or KV copy was issued; c is empty. prepare and finish still have issue/done events but perform no reads or writes.

## t_e1959cd30d2e909d | pipeline.verb | 454 chars | 1 declarations

mma_direct:254 design.tile_pipeline.phase.attention.role.Main.info.verb.enqueue

Future(c) is the prospective copy group for chunk c+Stages-1. For Stages>1 issue it into slot (c+Stages-1)%Stages when below C(t), otherwise commit an empty group. Stages=1 does nothing here. For Stages>1, done means the group was committed, not that the transfer landed; it can remain in flight through score, softmax and value. The target is initially unused or was released by the preceding iteration's retire barrier, so no current KV is overwritten.

## t_e1daf6a9c919f69b | pipeline.verb | 474 chars | 13 declarations

probe_hidden_mma:722 design.tile_pipeline.phase.prefix.role.Soft0.info.verb.online; probe_hidden_mma:785 design.tile_pipeline.phase.prefix.role.Soft1.info.verb.online; probe_hidden_mma:838 design.tile_pipeline.phase.prefix.role.Soft2.info.verb.online; probe_hidden_mma:882 design.tile_pipeline.phase.prefix.role.Soft3.info.verb.online; probe_hidden_packed:758 design.tile_pipeline.phase.prefix.role.Soft0.info.verb.online; probe_hidden_packed:822 design.tile_pipeline.phase.prefix.role.Soft1.info.verb.online; probe_hidden_packed:880 design.tile_pipeline.phase.prefix.role.Soft2.info.verb.online; probe_hidden_packed:929 design.tile_pipeline.phase.prefix.role.Soft3.info.verb.online; probe_serial_packed:687 design.tile_pipeline.phase.attention.role.Soft0.info.verb.online; probe_serial_packed:751 design.tile_pipeline.phase.attention.role.Soft1.info.verb.online; probe_serial_packed:808 design.tile_pipeline.phase.attention.role.Soft2.info.verb.online; probe_serial_packed:856 design.tile_pipeline.phase.attention.role.Soft3.info.verb.online; tc_packed:454 design.tile_pipeline.phase.attention.role.Soft0.info.verb.online

A softmax warp owns its consecutive RG matrix rows. Its two lane halves hold the two 32-key halves and duplicate m,l; read both Partial(c,a), add and scale, select the membership bit floor(r/H), then compute online's masked E and state updates. Exchange the two key-half maxima and sums within the warp. Publish alpha once per row and rendezvous the warp. Scores are replaced in place by E; both score partials have been read when done. E remains live until publish:Prob(c).

## t_e20674c2ad25e5bb | pipeline.verb | 427 chars | 2 declarations

probe_hidden_mma:924 design.tile_pipeline.phase.prefix.role.Finish.info.verb.store; probe_hidden_packed:975 design.tile_pipeline.phase.prefix.role.Finish.info.verb.store

Rejoin all eight warps at a CTA barrier after the real Result rows are staged. Cooperatively copy nrows contiguous rows from OutputStage into out, mapping row r to token t0+floor(r/H) and head r mod H. TMEM may be deallocated by its owner after that barrier while other warps copy, since copy-out no longer reads TMEM. Global stores can remain pending until the containing kernel's completion. No worker takes another worktile.

## t_e5928cf820bed032 | pipeline.verb | 1266 chars | 1 declarations

root:444 design.tile_pipeline.phase.attention.role.Main.info.verb.prepare

For nonempty Work, stage Q with zero padded heads and commit one copy group. Use Q_shared when QHeads>0, otherwise alias ring slot zero as Q_ring. If QPRE and Early=0, enqueue chunk c0 in slot zero before waiting for Q. Initialize control state, then publish cached indices into IndexShared when present. Each index store depends on its register load result; query issuance need not wait for it. Wait leaving one group outstanding only in that pre-issued case, otherwise wait all; a CTA barrier publishes Q, control and index storage and ends metadata reads. If QPRE and Early=1, only now enqueue c0, even when the width disables index caching. Cache the first QKS 16-wide Q slices in registers, duplicated across key groups. When Q occupies ring slot zero, a CTA barrier after all Q fragment reads frees it. Q_cached and any uncached Q_shared slices remain readable for the whole recurrence. Zero Acc. For j=(1 if QPRE else 0)..Stages-2, enqueue c0+j into slot j if it is below c1, otherwise commit an empty group. Enqueueing a chunk resolves slot IDs in batches before issuing 16-byte row copies, records validity and commits once; invalid slots still copy row zero of their pool. Completion of prepare means Q is ready and priming is issued, not that KV is ready.

## t_f0e5288e295640e5 | pipeline.verb | 235 chars | 1 declarations

probe_serial_mma:363 design.tile_pipeline.phase.attention.role.Main.info.verb.value

Rescale Acc, then read Prob and Gather(c) for synchronous PV. Each warp owns every HB row and D/8 distinct output columns. Acc stays in registers across chunks; all probability and KV reads by this warp finish before it reaches retire.

## t_f5729978632a5911 | pipeline.verb | 497 chars | 1 declarations

probe_serial_mma:371 design.tile_pipeline.phase.attention.role.Main.info.verb.finish

The last retire ends all reads of Gather; all copy groups have been waited. Form finalize's inverse, scale Acc and convert to BF16 Result, including padded heads. Reuse the KVStage prefix as OutputStage and barrier before cooperatively writing all columns of real heads to out. Stores may remain pending when individual lanes finish; attention.kernel_done provides output visibility. There are no scratch statistics, and the CTA takes no second worktile or numerical state into another invocation.

## t_f9c7952ce464f534 | pipeline.verb | 219 chars | 1 declarations

mma_direct:250 design.tile_pipeline.phase.attention.role.Main.info.verb.await

For Stages=1 enqueue c unless c=0 and QPRE, then wait all groups. For Stages>1 wait until all but the newest Stages-2 groups have completed. A CTA barrier then makes current Gather(c) and valid(c) readable by all warps.

## t_fe3ebcfec286747c | pipeline.verb | 1365 chars | 1 declarations

probe_serial_mma:263 design.tile_pipeline.phase.probe.role.Main.info.verb.build

For Usable(s), zero Hash, Membership and Count, then barrier. Hash holds a nonzero encoding of e+1 per occupied slot, zero for empty; insertion publishes the complete key with an atomic compare-and-swap. Collisions advance by linear probing until an empty or matching slot. Each slot has eight membership bits, four slots per Membership word. Enumerate occurrences in token-major compact-slot order, divided into 1024 strided lane sequences. A lane loads its first index, then in each iteration retains the current index and issues its next index into NextIndex before processing the current one. Consuming a prefetched index waits for that load result; the next load may overlap hash work. Negative IDs do no hash work. Each insertion adds one to LocalCount and sets its query bit; an existing key adds one only if the atomic bit-set returns that bit already set. Insertion does not inspect the previous membership bit and does not block other updaters until it is set. Initialize each lane's LocalCount to zero, accumulate across its sequence, then flush a nonzero count once to Count with an atomic add. A CTA barrier joins all flushes before Count is used for ProbeRatio. Hash and Membership have no consumers after this build; no next index is loaded beyond the lane's last occurrence. For an unusable sample, build completes without accessing these resources.

## t_05bb309f43079753 | realization.choice | 219 chars | 1 declarations

probe_serial_mma:388 realization.require[0].choice

Bind copies to cp.async.cg.shared.global 16-byte transfers, commit_group, wait_group 0 and CTA __syncthreads at the stated publication/reuse boundaries; use mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 for QK/PV.

## t_12266712f730210a | realization.choice | 286 chars | 4 declarations

probe_hidden_mma:1060 realization.require[2].choice; probe_hidden_packed:1010 realization.require[2].choice; probe_serial_packed:935 realization.require[2].choice; tc_packed:629 realization.require[2].choice

Bind hash installation/publication to shared-memory 64-bit atomicCAS/atomicExch and membership updates to 32-bit atomicOr on the word containing four mask bytes. Bind index staging and query prefetch to cp.async, retiring their groups with cp.async.wait_group before the CTA rendezvous.

## t_3435eb7183f19207 | realization.choice | 280 chars | 5 declarations

probe_hidden_mma:1047 realization.require[0].choice; probe_hidden_packed:997 realization.require[0].choice; probe_serial_packed:922 realization.require[0].choice; tc_heads:446 realization.require[0].choice; tc_packed:617 realization.require[0].choice

Bind the matrix products to tcgen05.mma.cta_group::1 with BF16 operands and FP32 accumulators, using per-issuer tcgen05.commit notifications. Use warp-convergent tcgen05.ld/st and their completion waits; the score load uses the 16x32bx2 form, while accumulator access uses 32x32b.

## t_466b6616ff8445ef | realization.choice | 837 chars | 2 declarations

tc_heads:467 realization.require[2].choice; tc_packed:637 realization.require[3].choice

Model.forward caches workspace selection by argument identity, using private scratch for a non-CUDA or smaller-than-32-MiB buffer. P.ws_bytes=0 here, so dsv4_run allocates no larger workspace. Submit on the current CUDA stream. The single-entry graph cache keys tensor pointers, dimensions, scales, the resolved plan and extension switches. A new eligible key dispatches directly; repeating it synchronizes the caller stream before private-stream capture, instantiation and replay. An executable-key hit replays without that synchronization. Capture records this one kernel; no numerical answers are cached. Begin-capture failure returns before dispatch; failed capture or instantiation executes no recorded work. Graph failure disables graph use and falls through to direct dispatch, without creating an additional completion guarantee.

## t_5388fdf258b1bff8 | realization.choice | 350 chars | 1 declarations

tc_heads:456 realization.require[1].choice

Build each pool's two-dimensional BF16 CUtensorMap with row stride 2*D, a 64-element-wide box and SWIZZLE_128B; pass maps by value to the kernel. Bind gather4 completion to mbarrier transaction bytes and the straddling chunk's cp.async completion to cp.async.mbarrier.arrive. Preserve the separate validity publication and data-landing notifications.

## t_7681fcbb3feb2e22 | realization.choice | 826 chars | 1 declarations

probe_serial_mma:396 realization.require[1].choice

Model.forward caches workspace selection by argument identity; a non-CUDA or smaller-than-32-MiB buffer selects cached private scratch. Initial P.ws_bytes=0, so no larger extension allocation is needed. Gate is the first eight words of process-cached cudaHostAllocMapped storage, with a cudaHostGetDevicePointer device alias. Reset uses volatile host stores, the readiness loop uses volatile host loads, and the final scan uses ordinary float loads. There is no cudaStreamSynchronize in this serial probe path. Submit both kernels on the current CUDA stream. Hidden setup may have attempted a fork event or side-stream wait before returning false, but submitted no side computation in this node. The internal graph cache requires initial P.dual=0; this node dispatches directly even after clearing the chosen plan's dual flag.

## t_78f8c0c686efb483 | realization.choice | 525 chars | 1 declarations

probe_hidden_packed:1037 realization.require[5].choice

Bind the fork and join to the cached cudaEventDisableTiming events and cudaEventRecord/cudaStreamWaitEvent. Use the cached cudaStreamNonBlocking side stream so the legacy default stream cannot introduce a prefix-to-probe or prefix-to-remainder barrier. Use the initial GT for both token ranges. The side stream, events and mapped gate persist across calls. Event waits queue device ordering, not a host completion wait; their reuse does not isolate gate contents between calls. Return out after successfully queuing the join.

## t_86e691845374eaeb | realization.choice | 243 chars | 1 declarations

probe_hidden_mma:1107 realization.require[7].choice

Bind query and KV copies to cp.async.cg.shared.global 16-byte transfers, commit_group and wait_group 0, with CTA __syncthreads at the declared publication and reuse boundaries. Use mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 for QK/PV.

## t_89c02d2d70ba236b | realization.choice | 337 chars | 3 declarations

probe_hidden_mma:1067 realization.require[3].choice; probe_hidden_packed:1017 realization.require[3].choice; probe_serial_packed:941 realization.require[3].choice

Bind hash insertion to shared-memory 64-bit atomicCAS, membership updates to 32-bit atomicOr on four packed mask bytes, and count flushing to integer atomicAdd. Keep probe storage static and separate from attention's 1024-byte-aligned dynamic shared storage; the probe fits the 48-KiB static limit without a dynamic shared-memory opt-in.

## t_ab5671cd5a6c173f | realization.choice | 486 chars | 3 declarations

probe_hidden_mma:1076 realization.require[4].choice; probe_hidden_packed:1026 realization.require[4].choice; probe_serial_packed:950 realization.require[4].choice

Use the process-lifetime 64-float cudaHostAllocMapped buffer and its cudaHostGetDevicePointer alias; only its first PS slots carry samples. Arm them through volatile host stores, submit the probe, poll with volatile host reads, then perform the ordinary FP32 final reads used for R. Retain storage through all pending probe writes even if polling stops early. The source reuses this one gate without per-invocation isolation; it provides no cross-call completion guarantee at poll exit.

## t_c4b29b7e56dd5caa | realization.choice | 222 chars | 2 declarations

mma_direct:298 realization.require[0].choice; root:549 realization.require[0].choice

Bind copy groups to cp.async.cg.shared.global 16-byte transfers, commit_group, wait_group and CTA __syncthreads at the stated publication/reuse boundaries; use mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 for QK/PV.

## t_d404c52ac9560689 | realization.choice | 1112 chars | 1 declarations

root:557 realization.require[1].choice

Model.forward caches workspace selection by argument identity; a non-CUDA or smaller-than-32-MiB workspace selects cached private scratch. dsv4_run allocates larger scratch if P.ws_bytes exceeds the selected capacity. The two-kernel flow uses the current CUDA stream, directly or through the process-wide single-entry CUDA Graph cache. Graph eligibility requires P.ws_bytes<=W in this node. Its key includes the ten tensor pointers, dimensions, scales, plan and extension switches. A changed key invalidates the old entry and submits directly. Repeating the last eligible key synchronizes the caller stream before capturing dispatch on a private stream, then instantiates and launches the graph on the caller stream. An executable key hit replays without that capture synchronization. Kernels read current input contents; the cache stores no numerical answers. Begin-capture failure returns before dispatch; failed capture or instantiation executes none of the recorded kernels. Graph failure disables graph use for the process and falls through to direct dispatch; it creates no additional completion guarantee.

## t_d6c6133c3657c555 | realization.choice | 483 chars | 1 declarations

probe_hidden_mma:1087 realization.require[5].choice

Bind the fork and join to the cached cudaEventDisableTiming events and cudaEventRecord/cudaStreamWaitEvent. Use the cached cudaStreamNonBlocking side stream so the legacy default stream cannot introduce a prefix-to-probe or prefix-to-remainder barrier. The side stream, events and mapped gate persist across calls. Event waits queue device ordering, not a host completion wait; their reuse does not isolate gate contents between calls. Return out after successfully queuing the join.

## t_e01cdc11a5d3e8b3 | realization.choice | 669 chars | 1 declarations

probe_serial_packed:961 realization.require[5].choice

Model.forward caches workspace selection by object identity, substituting private scratch for a non-CUDA or smaller-than-32-MiB buffer. P.ws_bytes=0 requires no larger allocation. Submit both kernels on the current CUDA stream; ordinary stream ordering implements probe.kernel_done before attention starts. Keep the accepted initial plan, including GT, when dispatching cfg90. The private graph cache tests initial P.dual, so it cannot capture or replay this path even though the host decision has completed. External graph capture records submissions without producing GPU completion events; an actual replay runs the recorded kernels and does not repeat host polling.

## t_e0b4c49e210a1f59 | realization.choice | 398 chars | 2 declarations

probe_hidden_mma:1098 realization.require[6].choice; probe_hidden_packed:1048 realization.require[6].choice

Model.forward caches workspace selection by object identity, substituting private scratch for a non-CUDA or smaller-than-32-MiB buffer. P.ws_bytes=0 needs no further allocation. The successful hidden path returns before the extension's private graph machinery. External capture records the three submissions and event relationships; actual replay uses those bindings without repeating host polling.

## t_ecd4e5bf7ba37c68 | realization.choice | 1064 chars | 1 declarations

mma_direct:306 realization.require[1].choice

Model.forward caches workspace selection by argument identity; a non-CUDA or smaller-than-32-MiB buffer selects cached private scratch. P.ws_bytes=0 here, so dsv4_run needs no larger allocation and attention does not consume scratch. Submit on the current CUDA stream. The process-wide single-entry graph cache is eligible only when the initial P.dual=0, even if P_run.dual=0 after fallback. Its key includes tensor pointers, dimensions, scales, resolved plan and extension switches. A new key invalidates the entry and dispatches directly. Repeating the last eligible key synchronizes the caller stream before capturing dispatch on a private stream, then instantiates and launches the graph on the caller stream. An executable-key hit replays without that capture synchronization. The cache stores no numerical answers. Begin-capture failure returns before dispatch; failed capture or instantiation executes none of the recorded kernels. Graph failure disables graph use for the process and falls through to direct dispatch, without adding a completion guarantee.

## t_f2bf364df458d219 | realization.choice | 352 chars | 4 declarations

probe_hidden_mma:1053 realization.require[1].choice; probe_hidden_packed:1003 realization.require[1].choice; probe_serial_packed:928 realization.require[1].choice; tc_packed:623 realization.require[1].choice

Build each pool's two-dimensional BF16 CUtensorMap with row stride 2*D, a 64-element-wide box and SWIZZLE_128B; pass maps by value to the kernel. Bind gather4 completion to mbarrier transaction bytes and the mixed-pool chunk's cp.async completion to cp.async.mbarrier.arrive. Preserve the separate membership publication and data-landing notifications.

## t_00353f1316336401 | realization.reason | 185 chars | 1 declarations

probe_hidden_mma:1057 realization.require[1].reason

Both loader forms feed the same shared operand layout while membership becomes visible independently of transfer completion. Maps are bound to the current pool pointers for each launch.

## t_01e313b7c33f6cdf | realization.reason | 133 chars | 2 declarations

probe_hidden_mma:1104 realization.require[6].reason; probe_hidden_packed:1054 realization.require[6].reason

Workspace identity participates in the initial plan, and capture must not be counted as an extra execution of the probe or attention.

## t_133d62b307465ac2 | realization.reason | 152 chars | 3 declarations

probe_hidden_mma:1084 realization.require[4].reason; probe_hidden_packed:1034 realization.require[4].reason; probe_serial_packed:958 realization.require[4].reason

Mapped storage permits the bounded host decision without cudaStreamSynchronize. The actual observations, including partial results, determine this node.

## t_1816a22bb157a3ce | realization.reason | 229 chars | 1 declarations

probe_serial_mma:408 realization.require[1].reason

The workspace binding feeds selection; mapped publication permits bounded host polling without a completion wait. Caller-stream ordering supplies the probe-to- attention dependency independently of which statistics host observed.

## t_3940e5ed49a4aaf6 | realization.reason | 228 chars | 4 declarations

probe_hidden_mma:1063 realization.require[2].reason; probe_hidden_packed:1013 realization.require[2].reason; probe_serial_packed:938 realization.require[2].reason; tc_packed:634 realization.require[2].reason

A competing inserter must observe a complete canonical position, and same-query duplicates must remain private entries. The staged indices and asynchronous query copies must be retired before ring reuse or an empty-group return.

## t_5765b1f198b9428b | realization.reason | 202 chars | 2 declarations

tc_heads:480 realization.require[2].reason; tc_packed:650 realization.require[3].reason

Workspace binding feeds selection despite unused device scratch. Graph capture and replay preserve the single-kernel computation and recorded tensor-map bindings while changing the host submission path.

## t_8d1da99d7223fe50 | realization.reason | 197 chars | 1 declarations

tc_packed:626 realization.require[1].reason

Both loaders must feed the same shared operand layout while membership becomes visible independently of transfer completion. Captured maps must remain bound to the pool pointers used by that graph.

## t_8ec687ef1050afed | realization.reason | 188 chars | 1 declarations

probe_hidden_mma:1110 realization.require[7].reason

The single ring slot cannot be read before copy completion or overwritten before all synchronous PV readers finish; shared publication also joins index, row-statistic and output producers.

## t_ae28cbf50b3ab244 | realization.reason | 146 chars | 1 declarations

probe_serial_packed:971 realization.require[5].reason

Workspace identity still participates in selection. The stream, independently of host polling, closes the probe-to-attention execution dependency.

## t_be03dc13b7dbc62e | realization.reason | 132 chars | 2 declarations

probe_hidden_mma:1095 realization.require[5].reason; probe_hidden_packed:1045 realization.require[5].reason

The nonblocking stream and event pair implement the declared fork/join while preserving overlap and caller-stream output visibility.

## t_c4c7c33d2848fd22 | realization.reason | 142 chars | 3 declarations

probe_hidden_mma:1073 realization.require[3].reason; probe_hidden_packed:1023 realization.require[3].reason; probe_serial_packed:947 realization.require[3].reason

The count uses returned atomic membership bits; static probe storage avoids weakening the alignment required by attention operand descriptors.

## t_cca429cd79a242f4 | realization.reason | 249 chars | 1 declarations

root:572 realization.require[1].reason

Scratch selection and graph eligibility determine the address used by both kernels. Capture preparation has an additional caller-stream completion boundary; successful replay retains the same producer-to-combine ordering over current input contents.

## t_ce933053be2ea18a | realization.reason | 183 chars | 5 declarations

probe_hidden_mma:1050 realization.require[0].reason; probe_hidden_packed:1000 realization.require[0].reason; probe_serial_packed:925 realization.require[0].reason; tc_heads:451 realization.require[0].reason; tc_packed:620 realization.require[0].reason

Independent issuer groups preserve QK/PV overlap; the score-load shape supplies both key halves, and convergent accumulator access is required even when only some rows need rescaling.

## t_e19c097658cbb728 | realization.reason | 154 chars | 1 declarations

probe_serial_mma:392 realization.require[0].reason

Copy completion and CTA visibility make gathered operands readable; synchronous matrix consumption lets the retirement barrier release the single KV slot.

## t_e813f57d59df0506 | realization.reason | 154 chars | 2 declarations

mma_direct:302 realization.require[0].reason; root:553 realization.require[0].reason

Group-count waits make the current ring slot readable while later transfers continue; synchronous matrix consumption lets the PV barrier release the slot.

## t_eab71beb07d5d419 | realization.reason | 195 chars | 1 declarations

tc_heads:462 realization.require[1].reason

Both loaders must feed the same shared operand layout while validity becomes visible independently of transfer completion. Captured maps must remain bound to the pool pointers used by that graph.

## t_ef2fe64021093250 | realization.reason | 180 chars | 2 declarations

probe_hidden_packed:1007 realization.require[1].reason; probe_serial_packed:932 realization.require[1].reason

Both loaders feed the same shared operand layout while membership becomes visible independently of transfer completion. Maps are bound to the current pool pointers for each launch.

## t_f11f1f543861e4e6 | realization.reason | 230 chars | 1 declarations

mma_direct:321 realization.require[1].reason

Workspace selection feeds the host plan despite unused device scratch. Graph eligibility and binding preserve the one-kernel flow; capture preparation adds a caller-stream completion boundary that ordinary replay does not require.

## t_07f0d27911d7fb77 | schedule.describe | 95 chars | 2 declarations

probe_serial_packed:350 design.work_schedule.phase.attention.describe; tc_packed:189 design.work_schedule.phase.attention.describe

Build the packed union, gather its chunks, run QK/softmax/PV, and write complete query outputs.

## t_396f7ecf5ca8d478 | schedule.describe | 78 chars | 1 declarations

root:293 design.work_schedule.phase.attention.describe

Gather, QK, online softmax and PV for one split; write its FP32 partial state.

## t_51ff7a659f823b52 | schedule.describe | 795 chars | 1 declarations

probe_serial_packed:307 design.work_schedule.phase.probe.describe

Produce all sample ratios for the host's R calculation. The host first arms the mapped slots with -2, submits this kernel, then performs at most 2000000 poll passes, stopping when a pass finds no -2. It subsequently reads each slot once and evaluates probe's positive-only mean. In this node that mean accepts packing, so the host submits attention with the initial P parameters. The host may finish reading before probe.kernel_done; this does not release the device completion edge. There is no GPU reduction or GPU consumer of ProbeRatio, hence no inter-phase tensor handoff. ProbeCount follows probe's actual update order and can differ from the attention union count. The source also permits pending probe writes to race a later invocation's gate reset; selection uses the actual observed R.

## t_5246602d5d2bb5a7 | schedule.describe | 77 chars | 1 declarations

tc_heads:132 design.work_schedule.phase.attention.describe

Gather, two-part QK, online softmax, PV and direct normalized BF16 writeback.

## t_7a4634c4fe0d369a | schedule.describe | 88 chars | 1 declarations

probe_serial_mma:208 design.work_schedule.phase.attention.describe

Compute all heads and output columns for one token through an unsplit online recurrence.

## t_9a1f2b5a1ae3d8d2 | schedule.describe | 460 chars | 1 declarations

probe_hidden_packed:428 design.work_schedule.phase.remainder.describe

Build packed unions and write outputs for the remaining groups on the side stream. After this kernel, record the join on the side stream and queue its wait on the caller stream behind prefix. The join covers remainder.kernel_done, which also follows probe.kernel_done; caller order covers prefix.kernel_done. Return does not block the host for either output producer. The join is an external completion boundary, not a fourth compute kernel or an output merge.

## t_a5ebf24bf1af7b8a | schedule.describe | 389 chars | 1 declarations

probe_hidden_mma:441 design.work_schedule.phase.remainder.describe

Compute each remaining token independently using mma_gather, mma_online and mma_finalize on the side stream. Record join after this kernel and queue its wait on the caller stream behind prefix. The join covers remainder.kernel_done, which follows probe.kernel_done; caller order covers prefix.kernel_done. Return queues this completion boundary without blocking the host or merging values.

## t_a91bb15eeabc0f0a | schedule.describe | 84 chars | 1 declarations

probe_serial_mma:170 design.work_schedule.phase.probe.describe

Compute and publish one overlap statistic per sampled group for host plan selection.

## t_ad2f1aabc67cf4ac | schedule.describe | 91 chars | 2 declarations

probe_hidden_mma:399 design.work_schedule.phase.prefix.describe; probe_hidden_packed:368 design.work_schedule.phase.prefix.describe

Build packed unions and write all outputs for the first K full groups on the caller stream.

## t_ba5795c46f82557f | schedule.describe | 63 chars | 1 declarations

root:339 design.work_schedule.phase.combine.describe

Merge all split vectors and statistics for one real output row.

## t_bd0b4eabb2b5c146 | schedule.describe | 109 chars | 1 declarations

mma_direct:125 design.work_schedule.phase.attention.describe

Gather, QK, online softmax and PV over the complete key range; normalize and overwrite the BF16 output slice.

## t_bfa6b2f72883142e | schedule.describe | 974 chars | 1 declarations

probe_hidden_mma:355 design.work_schedule.phase.probe.describe

The caller first records a fork after its prior work; the side stream waits for that fork before reading input tensors. The host arms the mapped slots with -2 and submits this full-batch probe to the side stream. It then submits prefix to the caller stream before polling at most 2000000 passes, stopping when a pass finds no -2. The final scan reads each slot once and computes probe's positive-only R. This node rejects packing and submits remainder using the stored P.alt_* parameters. Polling may finish before probe.kernel_done. Side-stream order still prevents remainder from starting before probe completion; prefix has no probe completion wait. ProbeRatio is consumed only by the host, not by either attention kernel, so no tensor handoff connects these phases. ProbeCount follows probe's actual update order and can differ from the attention union count. The source also permits pending probe writes to race a later gate reset; selection uses the actual observed R.

## t_e4b968e3612bdcdb | schedule.describe | 969 chars | 1 declarations

probe_hidden_packed:314 design.work_schedule.phase.probe.describe

The caller first records a fork after its prior work; the side stream waits for that fork before reading input tensors. The host arms the mapped slots with -2 and submits this full-batch probe to the side stream. It then submits prefix to the caller stream before polling at most 2000000 passes, stopping when a pass finds no -2. The final scan reads each slot once and computes probe's positive-only R. This node accepts packing and submits remainder using the initial P parameters. Polling may finish before probe.kernel_done. Side-stream order still prevents remainder from starting before probe completion; prefix has no probe completion wait. ProbeRatio is consumed only by the host, not by either attention kernel, so no tensor handoff connects these phases. ProbeCount follows probe's actual update order and can differ from the attention union count. The source also permits pending probe writes to race a later gate reset; selection uses the actual observed R.

## t_11a0fe7e8d5ab5f6 | schedule.strategy | 826 chars | 1 declarations

probe_hidden_mma:401 design.work_schedule.phase.prefix.decomposition.strategy

GT=P.gt in {4,8}, MR=64, KC=64, NG=ceil_div(T,GT) and K=N-PS, where N is the SM count used by the initial plan. The prefix retains P.cfg=90, splits=1, nb=1 and ws_bytes=0, and clears dual. H is 8 or 16 and GT*H<=MR. Prefix's local token coordinate equals the batch token coordinate and covers [0,K*GT), giving global groups grp in [0,K). Each group owns all H heads and D columns for GT tokens; this range ends on a complete group boundary. Resolve every token's request independently; prefix groups may cross request boundaries although probe samples exclude those groups. Nonempty unions traverse c=0..C(grp)-1; empty unions still write zeros. Padded matrix rows have no stores. The two attention phases share only read-only inputs, have disjoint outputs, no cross-phase KV/state reuse, no workspace partials and no combine.

## t_29d3c32fe209daff | schedule.strategy | 657 chars | 1 declarations

probe_hidden_mma:448 design.work_schedule.phase.remainder.decomposition.strategy

Bind HB=P.alt_hb=16, KM=P.alt_kc=128, P.alt_cfg=8, P.alt_nb=1, P.alt_splits=1 and DQ=1. The alternative was planned for the full T: GT>=4 and NG>=N imply T>N, so neither the underfilled-grid narrow-ring rule nor the one-wave DQ expansion can apply. No smaller-tail re-planning occurs. Set chosen.gt=1, dual=0 and ws_bytes=0. The range's local token u maps to t=K*GT+u in [K*GT,T); the original full-batch pointers and T are retained. Each worktile scans c=0..CM(t)-1 and owns all H heads and D output columns. Invalid slots and padded heads keep their gather and matrix work; CM(t)>=1. The complete prefix and the nonempty remainder cover every output once.

## t_2c723d01266648a8 | schedule.strategy | 549 chars | 1 declarations

tc_packed:191 design.work_schedule.phase.attention.decomposition.strategy

GT=P.gt=2, MR=64 and KC=64. The selected plan has splits=1, nb=1 and ws_bytes=0. Group grp owns tokens [grp*GT,min((grp+1)*GT,T)) and all H heads; H is 8, 16 or 32 here, so nrows<=MR. Groups may cross request boundaries; compute each token's metadata independently. Each nonempty group traverses c=0..C(grp)-1 over its realized union. A zero-entry union is still assigned work and writes zero output directly. All H sinks are shared across the group's queries. Padded matrix rows produce no output stores. Every real output element has one producer.

## t_36e80c95f1ca1869 | schedule.strategy | 904 chars | 1 declarations

probe_hidden_packed:370 design.work_schedule.phase.prefix.decomposition.strategy

For all phases, GT=P.gt in {4,8}, MR=64, KC=64, NG=ceil_div(T,GT), and K=N-PS, where N is the SM count used by the initial plan. Both range launches retain P.cfg=90, splits=1, nb=1 and ws_bytes=0, and clear dual. The alternative is never submitted. H is 8 or 16 and GT*H<=MR. Prefix's local token coordinate equals the batch token coordinate and covers [0,K*GT), giving global groups grp in [0,K). Each group owns all H heads and D columns for its GT tokens. This range ends on a complete group boundary. For both attention phases, resolve each real token's request independently; groups may cross request boundaries although probe samples exclude those groups. A nonempty union traverses c=0..C(grp)-1; an empty union still writes zeros. All H sinks are shared across queries, and padded matrix rows produce no stores. The two phases have disjoint outputs, no cross-phase KV/state reuse, and no combine.

## t_4355abb6b4f80dd1 | schedule.strategy | 165 chars | 1 declarations

root:331 design.work_schedule.phase.attention.schedule.strategy

Linear block index is (((t*S+s)*NB+hb)*DQ+dq). Adjacent dq CTAs request the same KV rows; this ordering creates reuse opportunities without a co-residency guarantee.

## t_46d8ed8936013cbb | schedule.strategy | 612 chars | 1 declarations

probe_hidden_packed:436 design.work_schedule.phase.remainder.decomposition.strategy

Use the shared GT, MR, KC, NG and K fixed by prefix. The remainder partition's local token coordinate u maps to batch token K*GT+u, covering [K*GT,T). Its local group coordinate j maps to global grp=K+j. Each group owns all H heads and D columns for G=min(GT,T-grp*GT) real tokens, with a possible partial group only at the batch tail. The full-batch pointers and T remain unchanged; only the launch range and its group offset change. Resolve request crossings, iterate nonempty unions and handle empty unions exactly as prefix defines. Together the two partitions produce every real output element exactly once.

## t_4ffd82837141e0e6 | schedule.strategy | 201 chars | 1 declarations

tc_heads:161 design.work_schedule.phase.attention.schedule.strategy

For block index worker_id, t*NB+hb=T*NB-1-worker_id. Reverse order can move longer causal rows earlier; it does not guarantee concurrent residency, balanced durations, or sharing between adjacent CTAs.

## t_5e1e3c8bb632f158 | schedule.strategy | 449 chars | 1 declarations

probe_serial_mma:210 design.work_schedule.phase.attention.decomposition.strategy

HB=16 and KC=128: P.alt_cfg=8, alt_hb=16, alt_nb=1, alt_splits=1, alt_kc=128. T>N prevents the narrow-ring and DQ rewrites of this MMA plan. There is one head block with base head zero; r is the actual head for r<H. Worktile t traverses c=0..C(t)-1, computes padded HB by KC scores and accumulates HB by D values. Read sinks[0:H]; padded heads use online's sentinel. Every real output element has one producer. Neither kernel uses workspace scratch.

## t_5f1200c1e81403fe | schedule.strategy | 727 chars | 1 declarations

mma_direct:127 design.work_schedule.phase.attention.decomposition.strategy

Use P_run defined in dispatch.when: HB=P_run.hb, KC=P_run.kc. DQ=2 for DQ2IX_CFG_5/DQ2IX_CFG_8, DQ=4 for DQ4IX_CFG_5/DQ4IX_CFG_8, and DQ=1 otherwise. P_run.splits=1 and P_run.gt=1 throughout this node. These are fixed host choices: make_plan resolves head/chunk geometry and scratch-capped splits from H,T,IndexWidth,W,N. Only cfg5/cfg8 gain DQ, using the largest power of two up to four whose CTA count fits N. Each worktile (t,hb,dq) traverses c=0..C(t)-1, computes HB by KC scores and row normalizers, but accumulates and writes only D/DQ output columns. Short rows still have C(t)>=1; invalid slots and padded heads retain gather and matrix work. Each real output element has one producer; scratch is unused by this kernel.

## t_5fbd461725dc3b45 | schedule.strategy | 123 chars | 1 declarations

probe_hidden_mma:477 design.work_schedule.phase.remainder.schedule.strategy

For block index worker_id, t=K*GT+worker_id. The launch offset is added before token decoding; no prefix token is repeated.

## t_6b4bd3d5ec17b343 | schedule.strategy | 808 chars | 1 declarations

root:295 design.work_schedule.phase.attention.decomposition.strategy

Use the initial plan P defined by meta_data.dispatch.when: HB=P.hb, KC=P.kc, S=P.splits, with 1<S<=128. DQ=2 or 4 for P.cfg DQ2IX_CFG_5/DQ2IX_CFG_8 or DQ4IX_CFG_5/DQ4IX_CFG_8 respectively, otherwise DQ=1. These are fixed host choices, not freely tunable parameters: make_plan resolves head/chunk geometry and scratch-capped splits from H,T,IndexWidth,W,N; only cfg5/cfg8 gain DQ, at the largest power of two up to four that still fits one wave. A worktile (t,hb,s,dq) traverses c0(t,s)..c1(t,s)-1. Shorter token rows can give empty splits despite the common launch S. It computes full HB by KC scores and normalizers but only D/DQ output columns; QK and gathers repeat across dq. Empty splits publish only sentinel statistics; their vector regions remain allocated with prior contents as defined by finalize.

## t_73f60841b5f84acd | schedule.strategy | 425 chars | 1 declarations

tc_heads:134 design.work_schedule.phase.attention.decomposition.strategy

HB=64 and KC=64. The selected plan has gt=1, splits=1 and ws_bytes=0. Each worktile (t,hb) traverses c=0..C(t)-1 and owns every output column of its HB heads. All heads are real since H is 64 or 128 in this node. Invalid slots remain in the chunk computation; all-invalid indices do not remove the recurrence. The sink input block is sinks[hb*HB:hb*HB+HB]. Each output element has one producer; workspace is unused on device.

## t_7a8dd94116cb204a | schedule.strategy | 155 chars | 1 declarations

mma_direct:160 design.work_schedule.phase.attention.schedule.strategy

Linear block index is ((t*NB+hb)*DQ+dq). Adjacent dq CTAs gather the same KV rows; this creates cache reuse opportunities without a co-residency guarantee.

## t_9417612bd771d2f3 | schedule.strategy | 692 chars | 1 declarations

probe_serial_packed:352 design.work_schedule.phase.attention.decomposition.strategy

GT=P.gt in {4,8}, MR=64 and KC=64. The accepted initial plan has cfg=90, splits=1, nb=1 and ws_bytes=0; its alternative is never submitted. H is 8 or 16 and GT*H<=MR. Group grp owns tokens [grp*GT,min((grp+1)*GT,T)) and all H heads and D columns. The full batch is assigned once; no packed prefix or token-range remainder is submitted. Groups may cross request boundaries; resolve each real token independently, even though the probe excludes such sample groups. Each nonempty group traverses c=0..C(grp)-1. Empty unions still write their zero outputs. All H sinks are shared across the group's queries. Padded matrix rows produce no output stores; every real output element has one producer.

## t_9b210d5cef4c8914 | schedule.strategy | 1206 chars | 1 declarations

probe_serial_mma:172 design.work_schedule.phase.probe.decomposition.strategy

GT=P.gt is 4 or 8 in this node; pack_gt starts at 64/H and halves until GT*IndexWidth<=2560 and ceil_div(T,GT)>=N. The threshold eliminates GT=2 from probe-gated plans. This implies H in {8,16} and T>N. Worktile s uses probe_count's sampled GT-token group. Every one of the eight submitted worktiles writes a ratio; absent, partial or cross-request groups publish -1. Index slots are visited only for Usable(s). During eager submission or external graph recording, host initializes Gate, then after submitting or recording the probe polls at most 2000000 times. Each ascending scan stops at its first -2; polling ends when a scan finds none across all eight volatile reads. It performs probe_select's final scan and chooses the MMA alternative. Host polling does not establish device completion and may end with pending slots. Graph replay executes both recorded kernels without repeating host selection. Gate is process-cached mapped host storage; only its first eight FP32 words are used. ProbeRatio is written directly there, with no extra device copy. The source permits an outstanding probe to race a later invocation's reset. Attention reads none of these statistics; there is no device data handoff.

## t_b2786c6c52097f61 | schedule.strategy | 577 chars | 1 declarations

root:341 design.work_schedule.phase.combine.decomposition.strategy

Launch T*Hp CTAs; the padded h>=H work is empty. CT=512 if T*Hp<N, otherwise CT=128, where N is the cached SM count in dispatch.when; G=CT/128. Each worktile folds all S splits of one (t,h). The four initial vector prefetches per group may include stale but allocated empty-split storage; zero weights discard those values. Later zero-weight addresses redirect to split zero, whose range is nonempty because Klen(t)>=1; their products still follow merge. Scratch is not cleared between calls; attention.kernel_done covers every producer of the vectors and statistics read here.

## t_bb640adf0daa9a86 | schedule.strategy | 145 chars | 2 declarations

probe_serial_packed:382 design.work_schedule.phase.attention.schedule.strategy; tc_packed:219 design.work_schedule.phase.attention.schedule.strategy

For block index worker_id, grp=ceil_div(T,GT)-1-worker_id. Reverse group order does not imply simultaneous residency or any sharing between CTAs.

## t_bd48dc1729c1c3b4 | schedule.strategy | 123 chars | 2 declarations

probe_hidden_mma:432 design.work_schedule.phase.prefix.schedule.strategy; probe_hidden_packed:419 design.work_schedule.phase.prefix.schedule.strategy

For block index worker_id, grp=K-1-worker_id. Reverse ordering is local to this range; each worker receives one full group.

## t_c28db41fb6a80014 | schedule.strategy | 129 chars | 1 declarations

probe_hidden_packed:453 design.work_schedule.phase.remainder.schedule.strategy

For block index worker_id, grp=NG-1-worker_id. Descend only within the remainder range; no prefix work is reassigned or repeated.

## t_ea1aca46a6d23885 | schedule.strategy | 99 chars | 1 declarations

probe_serial_mma:236 design.work_schedule.phase.attention.schedule.strategy

Each block index is its token index; each CTA traverses the whole selected key range independently.

## t_f0df11dfa8d23400 | schedule.strategy | 474 chars | 3 declarations

probe_hidden_mma:369 design.work_schedule.phase.probe.decomposition.strategy; probe_hidden_packed:328 design.work_schedule.phase.probe.decomposition.strategy; probe_serial_packed:320 design.work_schedule.phase.probe.decomposition.strategy

PS=8. Sample b is the ascending sample coordinate; sg(b), Items(b) and usability are defined by probe. Set PG=0 for b>=NG, otherwise PG=min(GT,T-sg(b)*GT). All PS samples receive a worker, including out-of-range, cross-request and partial-tail samples; those still publish -1. A usable sample examines every selected slot of its PG tokens and publishes one scalar, with no KV access. The probe hash/membership state is private to its worktile and is not reused by attention.

## t_8dde6f25e6466201 | schedule.thesis | 464 chars | 1 declarations

probe_hidden_packed:299 design.work_schedule.thesis

Fork after the caller's preceding work, run the full-batch probe on a side stream and a packed prefix on the caller stream, then submit the accepted packed remainder behind the probe. The remainder need not wait for the prefix. Queue a side-to-caller join after the remainder and behind the prefix, so subsequent caller work observes all output rows. One probe and two attention launches plus the event pair permit overlap without promising simultaneous residency.

## t_940ced24b2cfe413 | schedule.thesis | 312 chars | 1 declarations

root:278 design.work_schedule.thesis

One CTA owns a token, head block, KV split and output-column slice. It reuses Q while scanning balanced contiguous chunk ranges; smaller head blocks and KV/D partitions increase work count at the cost of repeated gathers or QK. Attention completes before combine consumes the global FP32 partials and writes out.

## t_a57165b0361320bc | schedule.thesis | 289 chars | 1 declarations

tc_packed:179 design.work_schedule.thesis

One CTA owns every head and output column of up to two consecutive queries. The group shares its union and gathered KV across matrix rows; reverse group order visits later tokens first. There is one direct-output kernel and no probe, split materialization, combine or device workspace use.

## t_aa009490bbfccb9b | schedule.thesis | 228 chars | 1 declarations

tc_heads:124 design.work_schedule.thesis

One CTA computes one query and one 64-head block over all selected slots and all output columns. Reverse grid order visits later queries first; no split materialization or combine is needed, and KV is shared across all 64 heads.

## t_ad3274fd12c8f8fe | schedule.thesis | 317 chars | 1 declarations

probe_hidden_mma:343 design.work_schedule.thesis

Fork after prior caller work, run the full-batch probe on the side stream and a packed prefix on the caller stream, then submit an unsplit mma remainder behind the probe. The remainder has no prefix-completion wait. A side-to-caller join queued behind the prefix makes later caller work observe both disjoint outputs.

## t_bb14eed70ccec08b | schedule.thesis | 229 chars | 1 declarations

mma_direct:117 design.work_schedule.thesis

One CTA owns a token, head block and output-column slice, scanning every chunk before normalizing and writing its slice. Larger head blocks reuse KV across more heads; DQ increases work count by repeating gathers, QK and softmax.

## t_d4b95f3bdd7db37f | schedule.thesis | 307 chars | 1 declarations

probe_serial_packed:295 design.work_schedule.thesis

Submit an independent sample-counting kernel, poll its mapped results on the host, then submit one full-batch packed attention kernel on the caller stream. The stream completion edge serializes their device work even if the poll times out early. Probe results select the plan but are not attention operands.

## t_efe1ce1a6860e0e2 | schedule.thesis | 217 chars | 1 declarations

probe_serial_mma:160 design.work_schedule.thesis

Eight CTAs sample candidate query groups, then host polling chooses one full-batch attention launch on the same caller stream. Each attention CTA owns one token; probe latency precedes all attention work in this path.
