# extract_mla: full open-text inventory

These are exact-text groups, not proof of semantic equivalence across contexts.

## t_114c35721a217016 | algorithm_contract.math | 496 chars | 1 declarations

root:89 contract.algorithm_requirement.compute[2].math

For 0 <= t < Tq and 0 <= h < H, let m = max_{u in J(t)} Score[t,h,u] and z = sum_{u in J(t)} exp(Score[t,h,u] - m). BaseLSE[t,h] = m + ln(z). BaseValue[t,h,v] = sum_{u in J(t)} (exp(Score[t,h,u] - m) / z) * DequantV[u,h,v], for 0 <= v < ValueDim. If Sinks is None, TotalLSE[t,h] = BaseLSE[t,h]. Otherwise, let a = max(BaseLSE[t,h], Sinks[h]); then TotalLSE[t,h] = a + ln(exp(BaseLSE[t,h]-a) + exp(Sinks[h]-a)). A sink adds exp(Sinks[h]) to the softmax denominator and contributes no value vector.

## t_1403e30a8da8525c | algorithm_contract.math | 595 chars | 1 declarations

root:74 contract.algorithm_requirement.compute[1].math

For each request b and 0 <= i < SeqLens[b], let t = CuQ[b] + i. J(t) = {CuKV[b]+j : 0 <= j < SeqLens[b] and (not Causal or j <= i)}. Equal Q/KV lengths make the causal alignment j <= i. WindowLeft = -1 imposes no additional window. J(t) is nonempty throughout the domain. For 0 <= t < Tq, 0 <= u < Tk and 0 <= h < H, Score[t,h,u] = fp32(ScoreScale) * sum_{d=0}^{QKDim-1} Q[t,h,d] * K[u,h,d]. Only scores for u in J(t) contribute to the result. DequantV[u,h,v] = V[u,h,v] * fp32(ValueScale), for 0 <= v < ValueDim. Floating-point arithmetic and accumulation use FP32; the reference disables TF32.

## t_6af5ea81a8b49cad | algorithm_contract.math | 1015 chars | 1 declarations

root:57 contract.algorithm_requirement.compute[0].math

Requests and heads are independent. All indices are zero-based. 1 <= B <= 32, H in {12, 24, 32, 96, 128}, and 1 <= Tq <= 16384. CuQ[0] = CuKV[0] = 0, CuQ[B] = Tq, CuKV[B] = Tk. For 0 <= b < B, let q_b = CuQ[b+1] - CuQ[b] and k_b = CuKV[b+1] - CuKV[b]. Require 1 <= q_b = k_b = SeqLens[b] <= 16384. Thus Tq = Tk; the source also bounds Tk <= 1048576. BatchSize = B, MaxQ = max_b q_b, and MaxKV = max_b k_b. All tensor arguments have contiguous storage in the listed axis order and reside on the same CUDA device. Q/K/V are the supplied stored-precision values; no unquantized payload is used. Page tables, projections, RoPE and cache writes are outside this prepared ragged interface. OutBuffer is always present. LseBuffer is present exactly when ReturnLSE is true. Sinks may be None independently of Causal and ReturnLSE; None disables sink attention. Every listed argument is supplied by the frozen input interface; no omitted-argument behavior is required. The fixed disabled options have no mathematical effect.

## t_ce814a346e335515 | algorithm_contract.math | 895 chars | 1 declarations

root:105 contract.algorithm_requirement.compute[3].math

For every query token t, head h and value channel v, O[t,h,v] = bf16(BaseValue[t,h,v] * exp(BaseLSE[t,h] - TotalLSE[t,h])). If ReturnLSE is true, LSE[t,h] = TotalLSE[t,h] / ln(2), stored in FP32; overwrite the full logical OutBuffer and LseBuffer and return (O, LSE). Otherwise LSE is None, overwrite the full logical OutBuffer and return O alone. O is OutBuffer after the write; non-None LSE is LseBuffer after the write. Initial destination contents do not contribute to either result. Workspace is permitted scratch with unspecified final contents and is not an output. Its prepared allocation is zero-initialized; each call computes from current Q/K/V and metadata, without retaining numerical state from a previous call. Preserve every other input and all storage outside the logical destination and scratch views, including the caller's guard regions. There are no other numerical outputs.

## t_027bb91482912c78 | algorithm_contract.meaning | 85 chars | 1 declarations

root:48 contract.algorithm_requirement.input.KVLengthsCPU.meaning

Unused host length argument (kv_seq_lens_cpu); python_none contains only Python None.

## t_075f16480eb250b2 | algorithm_contract.meaning | 51 chars | 1 declarations

root:33 contract.algorithm_requirement.input.ValueScale.meaning

Per-tensor value dequantization scale (bmm2_scale).

## t_09bca4da690053d2 | algorithm_contract.meaning | 55 chars | 1 declarations

root:51 contract.algorithm_requirement.input.SkipRowCheck.meaning

Supplied row-check option (skip_all_rows_active_check).

## t_0b1772892bf2bf6a | algorithm_contract.meaning | 29 chars | 1 declarations

root:25 contract.algorithm_requirement.input.V.meaning

Prepared value codes (value).

## t_10ed4bec25ec2956 | algorithm_contract.meaning | 33 chars | 1 declarations

root:29 contract.algorithm_requirement.input.MaxQ.meaning

Maximum query length (max_q_len).

## t_13eb211cfe84596d | algorithm_contract.meaning | 128 chars | 1 declarations

root:54 contract.algorithm_requirement.output.LSE.meaning

Base-2 log-sum-exp written over LseBuffer and returned with its object identity preserved when requested; otherwise Python None.

## t_19d3c4f638188196 | algorithm_contract.meaning | 34 chars | 1 declarations

root:26 contract.algorithm_requirement.input.SeqLens.meaning

Per-request KV lengths (seq_lens).

## t_2234e9bb4a49829a | algorithm_contract.meaning | 44 chars | 1 declarations

root:41 contract.algorithm_requirement.input.OutputScale.meaning

Disabled output scale argument (o_sf_scale).

## t_22d638b30761c951 | algorithm_contract.meaning | 58 chars | 1 declarations

root:32 contract.algorithm_requirement.input.ScoreScale.meaning

Composite per-tensor Q/K and attention scale (bmm1_scale).

## t_276ffe21219185fe | algorithm_contract.meaning | 29 chars | 1 declarations

root:23 contract.algorithm_requirement.input.Q.meaning

Prepared query codes (query).

## t_484e08bb26511c84 | algorithm_contract.meaning | 25 chars | 1 declarations

root:24 contract.algorithm_requirement.input.K.meaning

Prepared key codes (key).

## t_493de80627dd8e4e | algorithm_contract.meaning | 88 chars | 1 declarations

root:50 contract.algorithm_requirement.input.SPCompress.meaning

Unspecified compression option (uses_spcompress); python_none contains only Python None.

## t_542afe3a3a02d966 | algorithm_contract.meaning | 88 chars | 1 declarations

root:53 contract.algorithm_requirement.output.O.meaning

Attention result written over OutBuffer and returned with its object identity preserved.

## t_62e1173dc1dd79ee | algorithm_contract.meaning | 61 chars | 1 declarations

root:38 contract.algorithm_requirement.input.OutBuffer.meaning

Caller-owned destination (out); initial contents are ignored.

## t_66546ad79c9eb099 | algorithm_contract.meaning | 84 chars | 1 declarations

root:47 contract.algorithm_requirement.input.QLengthsCPU.meaning

Unused host length argument (q_seq_lens_cpu); python_none contains only Python None.

## t_6af09457217bccd9 | algorithm_contract.meaning | 105 chars | 1 declarations

root:43 contract.algorithm_requirement.input.SkipSoftmax.meaning

Disabled threshold argument (skip_softmax_threshold_scale_factor); python_none contains only Python None.

## t_77f4dfc49c0414b9 | algorithm_contract.meaning | 36 chars | 1 declarations

root:28 contract.algorithm_requirement.input.CuKV.meaning

KV prefix offsets (cum_seq_lens_kv).

## t_7bb03fe89e004e17 | algorithm_contract.meaning | 36 chars | 1 declarations

root:42 contract.algorithm_requirement.input.EnablePDL.meaning

Disabled launch option (enable_pdl).

## t_7c9546f64b2d57fa | algorithm_contract.meaning | 48 chars | 1 declarations

root:87 contract.algorithm_requirement.compute[1].symbol.DequantV.meaning

Value codes after the supplied per-tensor scale.

## t_8de145a38a244c03 | algorithm_contract.meaning | 28 chars | 1 declarations

root:20 contract.algorithm_requirement.constant.QKDim.meaning

Query and key channel count.

## t_9d3803088cd31d60 | algorithm_contract.meaning | 51 chars | 1 declarations

root:102 contract.algorithm_requirement.compute[2].symbol.BaseValue.meaning

Normalized value over visible keys before the sink.

## t_9eca3cdbfbb08c59 | algorithm_contract.meaning | 31 chars | 1 declarations

root:30 contract.algorithm_requirement.input.MaxKV.meaning

Maximum KV length (max_kv_len).

## t_9f3013d51dc65ffb | algorithm_contract.meaning | 53 chars | 1 declarations

root:37 contract.algorithm_requirement.input.ReturnLSE.meaning

Whether the call returns and writes LSE (return_lse).

## t_a1cab6153e3cc012 | algorithm_contract.meaning | 60 chars | 1 declarations

root:101 contract.algorithm_requirement.compute[2].symbol.BaseLSE.meaning

Natural-log normalization over visible keys before the sink.

## t_a7b6a127b4e8a918 | algorithm_contract.meaning | 38 chars | 1 declarations

root:27 contract.algorithm_requirement.input.CuQ.meaning

Query prefix offsets (cum_seq_lens_q).

## t_ad90b2bd7861a3dc | algorithm_contract.meaning | 48 chars | 1 declarations

root:36 contract.algorithm_requirement.input.Sinks.meaning

Optional per-head sink logits (attention_sinks).

## t_b23ea9260dcf6791 | algorithm_contract.meaning | 55 chars | 1 declarations

root:31 contract.algorithm_requirement.input.BatchSize.meaning

Redundant batch count supplied by the ABI (batch_size).

## t_bb5f9c39e2125b66 | algorithm_contract.meaning | 31 chars | 1 declarations

root:21 contract.algorithm_requirement.constant.ValueDim.meaning

Value and output channel count.

## t_bbf46611f69629da | algorithm_contract.meaning | 52 chars | 1 declarations

root:85 contract.algorithm_requirement.compute[1].symbol.J.meaning

Set of visible global key indices for query token t.

## t_c008d67ecd25aa35 | algorithm_contract.meaning | 51 chars | 1 declarations

root:34 contract.algorithm_requirement.input.Causal.meaning

Whether each query excludes later keys (is_causal).

## t_c63d9fec3d589d20 | algorithm_contract.meaning | 113 chars | 1 declarations

root:44 contract.algorithm_requirement.input.SageScales.meaning

Disabled SAGE scales (sage_attn_sfs); python_none_tuple4 contains only the Python tuple (None, None, None, None).

## t_cab28cf8ec93d00c | algorithm_contract.meaning | 54 chars | 1 declarations

root:103 contract.algorithm_requirement.compute[2].symbol.TotalLSE.meaning

Natural-log normalization including the optional sink.

## t_ce4ee30d3b4387a6 | algorithm_contract.meaning | 61 chars | 1 declarations

root:86 contract.algorithm_requirement.compute[1].symbol.Score.meaning

Scaled query-key dot products; visibility is determined by J.

## t_d0c0f47955f77247 | algorithm_contract.meaning | 47 chars | 1 declarations

root:35 contract.algorithm_requirement.input.WindowLeft.meaning

Disabled sliding-window argument (window_left).

## t_dd790bf2d22d33c9 | algorithm_contract.meaning | 59 chars | 1 declarations

root:46 contract.algorithm_requirement.input.Backend.meaning

Backend selector (backend); string denotes a Python string.

## t_dd96b54b63637107 | algorithm_contract.meaning | 76 chars | 1 declarations

root:40 contract.algorithm_requirement.input.Workspace.meaning

Caller-provided writable scratch (workspace_buffer); not a numerical result.

## t_e2bbfda43dfa4f15 | algorithm_contract.meaning | 70 chars | 1 declarations

root:39 contract.algorithm_requirement.input.LseBuffer.meaning

Optional caller-owned destination (lse); initial contents are ignored.

## t_e54e9aae1493084f | algorithm_contract.meaning | 85 chars | 1 declarations

root:49 contract.algorithm_requirement.input.FP16Softmax.meaning

Unspecified softmax option (use_fp16_softmax); python_none contains only Python None.

## t_fc86ce5d120899d0 | algorithm_contract.meaning | 119 chars | 1 declarations

root:45 contract.algorithm_requirement.input.SageBlock.meaning

Disabled SAGE block sizes (num_elts_per_sage_attn_blk); python_zero_tuple4 contains only the Python tuple (0, 0, 0, 0).

## t_11a4d87099297679 | algorithm_contract.range | 14 chars | 1 declarations

root:32 contract.algorithm_requirement.input.ScoreScale.range

ScoreScale > 0

## t_15524ee340dc64fc | algorithm_contract.range | 14 chars | 1 declarations

root:27 contract.algorithm_requirement.input.CuQ.range

0 <= CuQ <= Tq

## t_25fcae0e95545078 | algorithm_contract.range | 18 chars | 1 declarations

root:42 contract.algorithm_requirement.input.EnablePDL.range

EnablePDL is false

## t_974f39734dae819e | algorithm_contract.range | 23 chars | 1 declarations

root:46 contract.algorithm_requirement.input.Backend.range

Backend == 'trtllm-gen'

## t_afbf47591287186a | algorithm_contract.range | 22 chars | 2 declarations

root:38 contract.algorithm_requirement.input.OutBuffer.range; root:39 contract.algorithm_requirement.input.LseBuffer.range

finite, NaN, or +/-inf

## t_c9073dabd570c491 | algorithm_contract.range | 20 chars | 1 declarations

root:51 contract.algorithm_requirement.input.SkipRowCheck.range

SkipRowCheck is true

## t_d8e84b9ef1e90bb8 | algorithm_contract.range | 14 chars | 1 declarations

root:33 contract.algorithm_requirement.input.ValueScale.range

ValueScale > 0

## t_f637d6405d71bbd8 | algorithm_contract.range | 15 chars | 1 declarations

root:28 contract.algorithm_requirement.input.CuKV.range

0 <= CuKV <= Tk

## t_49c206d202b6eda7 | contract.compute_name | 28 chars | 1 declarations

root:56 contract.algorithm_requirement.compute[0].name

Request domain and interface

## t_8da4e81c583e73f1 | contract.compute_name | 30 chars | 1 declarations

root:73 contract.algorithm_requirement.compute[1].name

Scaled scores and visible keys

## t_bdca71b380f1c617 | contract.compute_name | 41 chars | 1 declarations

root:88 contract.algorithm_requirement.compute[2].name

Softmax with an optional zero-valued sink

## t_c975b0199c905741 | contract.compute_name | 43 chars | 1 declarations

root:104 contract.algorithm_requirement.compute[3].name

Output conversion and caller-visible writes

## t_bb3a3ddd84215e85 | contract.tolerance | 484 chars | 1 declarations

root:131 contract.algorithm_requirement.golden_reference.tolerance.LSE

None must match None and is not compared numerically; otherwise compare the written and returned LSE against the reference. Source record from problem/metadata.json: default has atol=0.01, rtol=0.05, max_mismatch_fraction=0.0, with no LSE-specific override. The source does not define the comparison formula, relative-error basis, comparison precision, NaN/Inf policy, or returned-versus-mutated check routing. This record does not adopt the schema's default atol/rtol interpretation.

## t_f6c6482d158f3f03 | contract.tolerance | 732 chars | 1 declarations

root:121 contract.algorithm_requirement.golden_reference.tolerance.O

Source record from problem/metadata.json benchmark_contract.correctness_tolerances: default has atol=0.01, rtol=0.05, max_mismatch_fraction=0.0; mutated_inputs.out has atol=0.01, rtol=0.05, max_mismatch_fraction=0.005, max_rel_l2=0.01. O is both the returned result and the out writeback. The source does not define comparison formulas, relative-error basis, L2 normalization, comparison precision, NaN/Inf policy, or how returned and mutated checks are routed or combined. These are source records, not the schema's default atol/rtol or relative-L2 interpretation. CERTIFIED_BEST.json records unchanged comparators and sampled full_pass results, without supplying these missing definitions or guaranteeing the whole dynamic domain.

## t_11d103e769a4800a | dispatch.md | 237 chars | 1 declarations

dispatch.md:14 dispatch.md

- 几何表采用 head-major 重块优先排列，或按 KV 块数稳定降序排列；排序门控只重排同一组独立工作。 - K/V 环按 scatter 门控取三槽或四槽，S/P 环仍为三槽。三槽以对应的 `pv_done` 归还 K/V，四槽保留按 K/V 周期通知的 `kv_free`；两者都等待上次占用同槽的 PV 完成。 - causal、sink 与 LSE 分别控制可见性、归一化和写回。短请求追加 FP32 tiny-row 修正，原有 QK/PV 主路径仍执行。

## t_2304aaf0197a9461 | dispatch.md | 7 chars | 1 declarations

dispatch.md:18 dispatch.md

## 覆盖缺口

## t_30be952880eeadb7 | dispatch.md | 97 chars | 1 declarations

dispatch.md:8 dispatch.md

bootstrap 直接取 root 为设计起点：两者的单工作块计算相同，差异在工作分配、终止和跨工作接续。各节点条件在共享 Contract 全域独立求值；精确条件和版本以各 YAML 为准。

## t_32b044faa5d8348b | dispatch.md | 38 chars | 1 declarations

dispatch.md:3 dispatch.md

| 节点 | 直接父 | workload | 相对父差异 | 选择依据 |

## t_33567800009ce304 | dispatch.md | 64 chars | 1 declarations

dispatch.md:22 dispatch.md

路径覆盖不构成全域数值通过保证。root 保留 TMEM 行寻址和持久描述符复用的源实现限制，以及外部比较器定义缺失的验收记录。

## t_7875020cff23f0cd | dispatch.md | 6 chars | 1 declarations

dispatch.md:1 dispatch.md

# 路径索引

## t_800e4e056f39471a | dispatch.md | 266 chars | 1 declarations

dispatch.md:6 dispatch.md

| [bootstrap](bootstrap.yaml) | root | 同一契约下工作数较小的 attention；每个 CTA 只处理一个 head/query 块。 | 保持单工作块数学与缓冲深度规则；CTA 数改为工作数，每 CTA 仅预分配一份；取消补领、Queue 服务、终止发布和跨工作 Q 预取，保留本工作内依赖并完成后退出。 | 同一 launch_impl 中 grid_items 取 items 的分支，包含阈值等号；使用当前 CuQ 派生的工作数与 n_sm_launch（非正时取 148）判定。 |

## t_8cf2c43702dc6331 | dispatch.md | 104 chars | 1 declarations

dispatch.md:10 dispatch.md

两路径的 Model.forward 前缀均读取当前长度与前缀、在 host 构造工作表并侧流上传；主流等待上传完成后提交一个注意力核。固定入口选择 FP8 PV，FP16 V 预转换及其备用主核不另列路径。

## t_901837e66794c412 | dispatch.md | 13 chars | 1 declarations

dispatch.md:12 dispatch.md

以下变化在各节点内参数化：

## t_eda49ba887d41049 | dispatch.md | 261 chars | 1 declarations

dispatch.md:5 dispatch.md

| [root](root.yaml) | 无 | 工作数较大的 ragged FP8 attention；每份工作对应一个 head 的一个 128 行 query 块。 | 代表设计：首份工作预分配，随后原子补领；缓冲和环索引跨工作接续，下一 Q 可与旧 PV/输出尾段重叠。 | Model.forward → launch_impl 中 grid_items 取 nsm_l 的分支；工作数由当前 CuQ 的 query 块数乘 H 得出，阈值使用缓存的设备 SM 数，非正时取源码 fallback 148。 |

## t_f4e357316cce3f7b | dispatch.md | 167 chars | 1 declarations

dispatch.md:20 dispatch.md

两节点覆盖固定三维 THD 契约下的 host grid 两臂。`launch_impl` 对 Q/K/V/out 基址要求 16 字节对齐，未对齐输入会被拒绝；这项实现限制未被写成共享契约的数学约束，固定入口没有替代路径。实现还检查 SM10x 与足够的 opt-in shared memory，检查失败即终止，没有其他硬件分支。

## t_d63e88f949a4fb98 | dispatch.when | 584 chars | 1 declarations

root:7 meta_data.dispatch.when

Let WorkCount = H * sum_{b=0}^{B-1} ceil_div(CuQ[b+1] - CuQ[b], 128). Let ReportedSMCount be launch_impl's cached n_sm_launch value: while that cache is zero, the source queries cudaDevAttrMultiProcessorCount for the current device with cudaDeviceGetAttribute. Let LaunchSMCount = ReportedSMCount if it is positive, otherwise 148, where 148 is the source's fallback value, not an asserted device property. Select exactly when 2 * WorkCount > 3 * LaunchSMCount. These are the host values used for the selected launch; a recorded execution retains the host selection bound at recording.

## t_fdf3a93b75fa98cb | dispatch.when | 585 chars | 1 declarations

bootstrap:10 meta_data.dispatch.when

Let WorkCount = H * sum_{b=0}^{B-1} ceil_div(CuQ[b+1] - CuQ[b], 128). Let ReportedSMCount be launch_impl's cached n_sm_launch value: while that cache is zero, the source queries cudaDevAttrMultiProcessorCount for the current device with cudaDeviceGetAttribute. Let LaunchSMCount = ReportedSMCount if it is positive, otherwise 148, where 148 is the source's fallback value, not an asserted device property. Select exactly when 2 * WorkCount <= 3 * LaunchSMCount. These are the host values used for the selected launch; a recorded execution retains the host selection bound at recording.

## t_047dda1f7bc27b12 | math.arithmetic | 219 chars | 1 declarations

root:272 design.math_reshape.rewrite[5].arithmetic.approx

ln, log1p, exp and division follow the source logf, log1pf, expf and scalar division compiled with --use_fast_math, including approximate division and flush-to-zero behavior. unknown: no matched error bound is supplied.

## t_88afce1b9b222ae7 | math.arithmetic | 183 chars | 1 declarations

root:249 design.math_reshape.rewrite[4].arithmetic.approx

exp2 follows the exp2f/build behavior specified in weights.arithmetic.approx; unknown: a matched error bound is not provided, so FP32 correction does not assert exact real arithmetic.

## t_9d73c921140d43d7 | math.arithmetic | 296 chars | 1 declarations

root:215 design.math_reshape.rewrite[2].arithmetic.approx

half_exp2 is the source ex2.approx.f16x2 operation on FP16 inputs, producing FP16 results. Alpha uses exp2f under the source --use_fast_math build, including its approximate/flush-to-zero scalar behavior. unknown: no matched instruction error bound or whole-domain output-error bound is supplied.

## t_a9be693f3a85423f | math.arithmetic | 160 chars | 1 declarations

root:234 design.math_reshape.rewrite[3].arithmetic.accumulate

Four consecutive FP8 matrix accumulation groups per key tile; the first group of the first tile initializes the output accumulator without reading old contents.

## t_ab6941ce24adb2a8 | math.arithmetic | 138 chars | 1 declarations

root:190 design.math_reshape.rewrite[1].arithmetic.accumulate

FP8 matrix instructions accumulate six consecutive channel groups into FP32; no claim of bitwise agreement with a scalar FP32 dot product.

## t_1938bd251ab465da | math.ledger | 6 chars | 2 declarations

root:279 design.math_reshape.ledger.footprint[0].by[0]; root:281 design.math_reshape.ledger.matmul[0].by[1]

scores

## t_35f4fe23f61a940e | math.ledger | 19 chars | 1 declarations

root:285 design.math_reshape.ledger.error[0].item

Narrow softmax path

## t_3973042d96219d10 | math.ledger | 201 chars | 1 declarations

root:281 design.math_reshape.ledger.matmul[0].change

H*A*(QKDim+ValueDim) -> H*Bq*Bk*C*(QKDim+ValueDim) MACs; causal tile pruning reduces C, while boundary padding remains fully evaluated. PV multiplication inputs change from FP32 weights/values to E4M3.

## t_546a20e5d765d4ff | math.ledger | 125 chars | 1 declarations

root:279 design.math_reshape.ledger.footprint[0].change

8*H*A globally materialized score/weight bytes -> 0; bounded tile and recurrence storage replaces request-wide intermediates.

## t_65970c1d5fa71157 | math.ledger | 29 chars | 1 declarations

root:283 design.math_reshape.ledger.vector[0].item

Rescaling and tiny correction

## t_7624d9fcb54a1aad | math.ledger | 233 chars | 1 declarations

root:285 design.math_reshape.ledger.error[0].change

Adds FP16 exponent-argument rounding, approximate FP16 exp2, four FP16 sum chains and E4M3 PV-weight rounding; tiny outputs bypass these weight errors but retain FP8 QK and compiled FP32 functions. No supplied end-to-end error bound.

## t_967c66b6735b0e99 | math.ledger | 30 chars | 1 declarations

root:279 design.math_reshape.ledger.footprint[0].item

Score and weight intermediates

## t_987921675ab7254d | math.ledger | 7 chars | 3 declarations

root:279 design.math_reshape.ledger.footprint[0].by[1]; root:281 design.math_reshape.ledger.matmul[0].by[2]; root:285 design.math_reshape.ledger.error[0].by[0]

weights

## t_ac4fc3bdac56544e | math.ledger | 156 chars | 1 declarations

root:283 design.math_reshape.ledger.vector[0].change

Up to H*Bq*ValueDim*sum_b sum_a (N-1) FP32 rescale multiplies; additionally at most H*U*32*32*ValueDim FP32 correction FMAs, with zero-weight terms skipped.

## t_aceac166e79d42a4 | math.ledger | 11 chars | 4 declarations

root:279 design.math_reshape.ledger.footprint[0].by[2]; root:281 design.math_reshape.ledger.matmul[0].by[3]; root:283 design.math_reshape.ledger.vector[0].by[0]; root:285 design.math_reshape.ledger.error[0].by[1]

value_state

## t_af2cde51a3267b4d | math.ledger | 142 chars | 1 declarations

root:277 design.math_reshape.ledger.baseline

Full per-request QK and FP32 normalized weights materialized, followed by FP32 PV over each dense q_b by k_b score matrix with causal masking.

## t_ced8953596d4aeee | math.ledger | 18 chars | 1 declarations

root:281 design.math_reshape.ledger.matmul[0].item

Dense contractions

## t_d193882ad9a99204 | math.ledger | 6 chars | 1 declarations

root:285 design.math_reshape.ledger.error[0].by[3]

output

## t_fadcdfe2ea45a1fb | math.ledger | 5 chars | 1 declarations

root:281 design.math_reshape.ledger.matmul[0].by[0]

tiles

## t_fc986f4fd3d6080c | math.ledger | 4 chars | 3 declarations

root:279 design.math_reshape.ledger.footprint[0].by[3]; root:283 design.math_reshape.ledger.vector[0].by[1]; root:285 design.math_reshape.ledger.error[0].by[2]

tiny

## t_159d001e7eb771d1 | math.math | 892 chars | 1 declarations

root:193 design.math_reshape.rewrite[2].math

For each row r, initialize M(-1)[r]=z and D(-1)[r]=0. For t = 0..N-1, let cand = max(M(t-1)[r],max_j X(t)[r,j]). M(t)[r] = cand if M(t-1)[r]=z or (cand-M(t-1)[r])*c > 0.75, otherwise M(t-1)[r]. Let safe = 0 if M(t)[r]=z, otherwise M(t)[r]. Alpha(t)[r] = 0 if M(t-1)[r]=z, otherwise exp2((M(t-1)[r]-safe)*c). Let off = fma32(-safe,c,8), and E[r,j] = half_exp2(fp16(fma32(X(t)[r,j],c,off))). P(t)[r,j] = e4m3_sat(E[r,j]). For u=0..3, initialize H_u=fp16(0), then for g=0..31 set H_u=fp16(H_u+E[r,4*g+u]). local = (fp32(H_0)+fp32(H_1))+(fp32(H_2)+fp32(H_3)). D(t)[r] = fma32(D(t-1)[r],Alpha(t)[r],local). e4m3_sat is round-to-nearest-even FP8 E4M3 conversion with finite saturation. The offset 8 scales weights by 256 before narrowing; the denominator uses the FP16 E values, not the rounded E4M3 P values. Masked entries also undergo the exponent and conversion path from the finite sentinel z.

## t_612a8d3ff71d170d | math.math | 492 chars | 1 declarations

root:222 design.math_reshape.rewrite[3].math

For t=0, initialize Yprev to the zero Bq by ValueDim matrix. For t>0, Yprev[r,v] = Alpha(t)[r]*Y(t-1)[r,v]. Multiplication is skipped for each consecutive group of 32 rows whose Alpha values are all exactly one; otherwise the whole group is multiplied. Starting with Yprev, for g=0..3 accumulate Yprev = mma32(P(t)[:,32*g:32*g+32],VTile(t)[32*g:32*g+32,:],Yprev). Set Y(t)=Yprev. This path executes for every worktile, including tiny rows whose final result instead uses the correction below.

## t_77180dba98dd7e44 | math.math | 483 chars | 1 declarations

root:237 design.math_reshape.rewrite[4].math

When q_b <= 32 (hence a=0 and N=1), for every 0 <= r < 32 recompute e[j] = exp2(fma32(S(0)[r,j],c,-M(0)[r]*c)) if visible(r,0,j), otherwise zero, for j=0..31. TinyD[r] is the FP32 sum in ascending j order. Initialize TinyY[r,v]=0, then in ascending j order, only when e[j] != 0, TinyY[r,v]=fma32(e[j],VTile(0)[j,v],TinyY[r,v]) for all v. The V codes are widened via FP16 to FP32 exactly; weights are not narrowed. The original E, P, D and Y calculations still execute for these rows.

## t_8baca609b7cf82c9 | math.math | 323 chars | 1 declarations

root:180 design.math_reshape.rewrite[1].math

In each tile t, S(t) = mm32(QTile[:,0:32],KTile(t)[:,0:32]ᵀ), then for g = 1..5 replace S(t) by mma32(QTile[:,32*g:32*g+32],KTile(t)[:,32*g:32*g+32]ᵀ,S(t)). Let z = -3.4028234663852885981e38, s = fp32(ScoreScale), vscale = fp32(ValueScale), c = s*1.4426950408889634074. X(t)[r,j] = S(t)[r,j] if visible(r,t,j), otherwise z.

## t_8ef95a88ca8197af | math.math | 769 chars | 1 declarations

root:160 design.math_reshape.rewrite[0].math

Let q_b = CuQ[b+1]-CuQ[b], k_b = SeqLens[b]. For each (b,h,a), 0 <= a < ceil_div(q_b,Bq), let q0 = a*Bq, qv = min(Bq,q_b-q0), vend = min(k_b,q0+qv) if Causal, otherwise k_b, and N = ceil_div(vend,Bk). For 0 <= r < Bq, QTile[r,d] is Q[CuQ[b]+q0+r,h,d] when its global token index is below Tq, otherwise zero. For 0 <= j < Bk, 0 <= t < N, KTile(t)[j,d] and VTile(t)[j,v] are K[CuKV[b]+t*Bk+j,h,d] and V[CuKV[b]+t*Bk+j,h,v] when that global token index is below Tk, otherwise zero. Thus request-tail transfers can include the next request's stored values. visible(r,t,j) means t*Bk+j < k_b and (not Causal or t*Bk+j <= q0+r). Full tiles are evaluated, including r >= qv; only r < qv is written out. Let A = sum_b q_b*k_b, C = sum_b sum_a N, and U = count({b | q_b <= 32}).

## t_a0577d8cacab57d9 | math.math | 738 chars | 1 declarations

root:255 design.math_reshape.rewrite[5].math

For every computed row, choose (Fsum,Fvalue,shift) = (TinyD[r],TinyY[r,:],0) when q_b<=32 and r<32, otherwise (D(N-1)[r],Y(N-1)[r,:],8). Fmax[r]=M(N-1)[r]; retain Fsum and Fmax until output normalization. norm = vscale/Fsum if Fsum>0, otherwise 0. base = fma32(Fmax[r],s,ln(Fsum)-0.6931471805599453*shift). If Sinks is present, total = max(base,Sinks[h])+log1p(exp(-abs(base-Sinks[h]))) and norm = norm*exp(base-total); otherwise total=base. Compute result[v]=bf16(Fvalue[v]*norm) and result_lse=total*1.4426950408889634074. For r<qv, write O[CuQ[b]+q0+r,h,:]=result, and, if ReturnLSE, LSE[CuQ[b]+q0+r,h]=result_lse. result_lse is evaluated even without an LSE store. Return the caller-owned buffers exactly as specified by the Contract.

## t_0049d7b384b364f3 | math.meaning | 25 chars | 1 declarations

root:174 design.math_reshape.rewrite[0].symbol.Bk.meaning

Keys per local iteration.

## t_1430500de0e00d7e | math.meaning | 42 chars | 1 declarations

root:175 design.math_reshape.rewrite[0].symbol.QTile.meaning

Full query transfer view for one worktile.

## t_1c3a4f6193138898 | math.meaning | 34 chars | 1 declarations

root:211 design.math_reshape.rewrite[2].symbol.D.meaning

Narrow-weight normalization state.

## t_1d9c3906aa9767c5 | math.meaning | 47 chars | 1 declarations

root:176 design.math_reshape.rewrite[0].symbol.KTile.meaning

Full key transfer view for one local iteration.

## t_262ec6f4ab02dade | math.meaning | 75 chars | 1 declarations

root:188 design.math_reshape.rewrite[1].symbol.S.meaning

Unscaled score tile retained until softmax and any tiny correction read it.

## t_2c3ac42bbd4267b2 | math.meaning | 49 chars | 1 declarations

root:177 design.math_reshape.rewrite[0].symbol.VTile.meaning

Full value transfer view for one local iteration.

## t_2d285c675ac48282 | math.meaning | 24 chars | 1 declarations

root:173 design.math_reshape.rewrite[0].symbol.Bq.meaning

Query rows per worktile.

## t_35147996e71a7f1f | math.meaning | 31 chars | 1 declarations

root:210 design.math_reshape.rewrite[2].symbol.M.meaning

Frozen raw-score maximum state.

## t_426cb8cf59bbd48a | math.meaning | 48 chars | 1 declarations

root:247 design.math_reshape.rewrite[4].symbol.TinyY.meaning

FP32 correction numerator for the first 32 rows.

## t_615582f88d70f1a2 | math.meaning | 64 chars | 1 declarations

root:232 design.math_reshape.rewrite[3].symbol.Y.meaning

FP32 main PV state retained through correction and output drain.

## t_78388ede9a406fd0 | math.meaning | 52 chars | 1 declarations

root:213 design.math_reshape.rewrite[2].symbol.P.meaning

Rounded scaled weights consumed by the main PV path.

## t_96c1a1cc890cfcad | math.meaning | 46 chars | 1 declarations

root:270 design.math_reshape.rewrite[5].symbol.Fmax.meaning

Final raw-score offset for every computed row.

## t_be2ff05c71f9ec62 | math.meaning | 62 chars | 1 declarations

root:212 design.math_reshape.rewrite[2].symbol.Alpha.meaning

Rescaling factors published before weight conversion finishes.

## t_d1078b36416b7e7e | math.meaning | 37 chars | 1 declarations

root:246 design.math_reshape.rewrite[4].symbol.TinyD.meaning

Unscaled FP32 correction denominator.

## t_fdac7b33786ef5a2 | math.meaning | 50 chars | 1 declarations

root:269 design.math_reshape.rewrite[5].symbol.Fsum.meaning

Selected final denominator for every computed row.

## t_004cac2d91598b3b | math.name | 45 chars | 1 declarations

root:236 design.math_reshape.rewrite[4].name

Additional FP32 correction for short requests

## t_482cd01d7d67c8e6 | math.name | 33 chars | 1 declarations

root:192 design.math_reshape.rewrite[2].name

Frozen maximum and narrow weights

## t_78716efa1d529970 | math.name | 32 chars | 1 declarations

root:221 design.math_reshape.rewrite[3].name

Rescaled FP32 value accumulation

## t_a9b99d5a1c0fec61 | math.name | 44 chars | 1 declarations

root:254 design.math_reshape.rewrite[5].name

Sink normalization and caller-visible writes

## t_b1bbb88ed4f631ea | math.name | 37 chars | 1 declarations

root:179 design.math_reshape.rewrite[1].name

Raw FP8 products with delayed scaling

## t_e32837a8d7e431d6 | math.name | 41 chars | 1 declarations

root:159 design.math_reshape.rewrite[0].name

Padded tiles and visible reduction extent

## t_75b1a8aaee7da481 | math.step | 11 chars | 1 declarations

root:220 design.math_reshape.rewrite[3].step

value_state

## t_9ee8788a627177f9 | math.step | 7 chars | 1 declarations

root:191 design.math_reshape.rewrite[2].step

weights

## t_bf6f4a873c7ec1db | math.step | 4 chars | 1 declarations

root:235 design.math_reshape.rewrite[4].step

tiny

## t_c680156a1c4cb17d | math.step | 6 chars | 1 declarations

root:253 design.math_reshape.rewrite[5].step

output

## t_d0f00f4d34ae776a | math.step | 6 chars | 1 declarations

root:178 design.math_reshape.rewrite[1].step

scores

## t_e863782023c80a19 | math.step | 5 chars | 1 declarations

root:158 design.math_reshape.rewrite[0].step

tiles

## t_580e9a44d35b5b25 | math.thesis | 312 chars | 1 declarations

root:152 design.math_reshape.thesis

Stream blocked QK and PV through an online normalization state instead of materializing request-wide scores. Frozen maxima reduce accumulator rescaling, while FP16 exponentiation/sums and E4M3 weights trade precision for narrow PV; short requests also compute an FP32 correction without removing the narrow path.

## t_da32246ee7b3f6ba | metadata.describe | 111 chars | 2 declarations

bootstrap:3 meta_data.describe; root:3 meta_data.describe

Ragged attention from prepared E4M3 Q/K/V to BF16 output, with optional attention sinks and base-2 log-sum-exp.

## t_1094b04067f5993b | pipeline.thesis | 329 chars | 1 declarations

bootstrap:82 design.tile_pipeline.thesis

Separate loading, matrix issue, softmax and output correction within one worktile. Publish scales ahead of weights; three score/probability slots and a three- or four-slot K/V ring bound the producer lead. The source's TMEM row-address discrepancy remains a limitation; the full row partition and its data lifetimes are retained.

## t_c811aac3eb444962 | pipeline.thesis | 475 chars | 1 declarations

root:355 design.tile_pipeline.thesis

Separate loading, matrix issue, softmax, output correction and queue service. Publish scales ahead of weights and refill during the last QK wait; retain continuous rings across worktiles so next-Q loading can overlap the previous PV and output drain. Score/probability aliasing and single-output reuse bound the lead. The source has TMEM row-address and early descriptor-reuse discrepancies; the complete row partition and declared two-slot lifetime constraints are retained.

## t_046dd2f01b70017b | pipeline.verb | 281 chars | 1 declarations

bootstrap:235 design.tile_pipeline.phase.main.role.Correction.info.verb.rescale

Y(t) reads Alpha(t), copies all 128 previous PV columns into CorrectionRegisters before multiplying and writing them back, and publishes completion for nonfirst tiles. A group with all scales equal to one skips TMEM data traffic but still completes and publishes its rescale event.

## t_07a7ef4342d56d48 | pipeline.verb | 97 chars | 1 declarations

root:606 design.tile_pipeline.phase.main.role.Loader.info.verb.latch.meaning

Latch Item(w) after issuing its Q transfer; retain its K/V base and head through all local fills.

## t_0d7c412105cc827b | pipeline.verb | 279 chars | 1 declarations

root:487 design.tile_pipeline.phase.main.role.Softmax.info.verb.produce

Final(w) captures Fsum,Fmax on the last tile. For short requests, the first row group rereads the first 32 S columns and VTile for the tiny rewrite, storing TinyY and TinyD into the selected final denominator. Other rows keep the main denominator. P publication remains required.

## t_0ea08103e91c4b4e | pipeline.verb | 147 chars | 1 declarations

root:492 design.tile_pipeline.phase.main.role.Softmax.info.verb.wait

Next(w) observes the published next descriptor, including a failed pop; terminal descriptors end the role only after its current work has finished.

## t_0ea25aab0a5f80fe | pipeline.verb | 281 chars | 1 declarations

root:523 design.tile_pipeline.phase.main.role.Correction.info.verb.rescale

Y(n) reads Alpha(n), copies all 128 previous PV columns into CorrectionRegisters before multiplying and writing them back, and publishes completion for nonfirst tiles. A group with all scales equal to one skips TMEM data traffic but still completes and publishes its rescale event.

## t_1687c96691453d71 | pipeline.verb | 259 chars | 1 declarations

bootstrap:195 design.tile_pipeline.phase.main.role.Softmax.info.verb.latch

Latch this CTA's preassigned Item geometry after the initialization join, and reset M,D. The first Geometry slot is initialized from Geo[worker_id]; the second allocated slot remains unused. Each warp handles its consecutive 32-row group throughout this role.

## t_1995f3c1d0ad3f64 | pipeline.verb | 125 chars | 1 declarations

root:478 design.tile_pipeline.phase.main.role.Softmax.info.verb.request

Next(w) requests one refill before the last score wait. This only signals the server and does not wait for the atomic result.

## t_28db977203000f8d | pipeline.verb | 178 chars | 1 declarations

root:479 design.tile_pipeline.phase.main.role.Softmax.info.verb.load

Scores(n) reads both halves of S(n) into ScoreRegisters; done releases the score read for subsequent QK slot reuse. The second-half transfer overlaps first-half masking/max work.

## t_3891fd185b1c3771 | pipeline.verb | 334 chars | 1 declarations

bootstrap:201 design.tile_pipeline.phase.main.role.Softmax.info.verb.publish

Alpha(t) updates M and publishes all row scales before exponentiation. P(t) copies PackedRegisters into its aliased TMEM view and completes its writes before publication. The two-slot scale relay is safe because QK(t+2) completion preceding its next write follows PV(t), which already required the applicable rescale to read Alpha(t).

## t_407ffe161a2a66b7 | pipeline.verb | 322 chars | 1 declarations

root:532 design.tile_pipeline.phase.main.role.Correction.info.verb.release

OutputState(w) reads the last 32 ordinary Y columns into Tail(w). For tiny rows the preceding prefix action has completed all scratch reads and stores. done returns the shared Y/Fsum/Fmax/TinyY lifetime after every row group's required reads; ordinary tail conversion, stores and LSE write may continue after this release.

## t_4d9787d680b8e9d6 | pipeline.verb | 573 chars | 1 declarations

bootstrap:267 design.tile_pipeline.phase.main.role.MMA.info.verb.mma

P(t)@VTile(t) performs the value_state rewrite into Y. First PV initializes Y; later PV follows correction. Products issue in one ordered tensor stream, and a completion notification covers all earlier products. Earlier PV reads of an aliased P slot finish before a later QK overwrites that slot. This ordering also protects the tiny reread before the same S slot can recur. PV completion releases its K/V slot. For KVSlots=3 the release notification cycles with the three S/P slots; for KVSlots=4 it additionally cycles with the four K/V slots, preventing parity aliasing.

## t_4f9a7c1e98a75ff7 | pipeline.verb | 446 chars | 1 declarations

root:524 design.tile_pipeline.phase.main.role.Correction.info.verb.store

OutputPrefix(w) evaluates the output rewrite using Fsum,Fmax and drains/stores the first 96 Y columns in three serial chunks. For tiny rows it instead reads and stores all TinyY columns. OutputTail(w) converts/stores the last 32 ordinary columns from Tail(w) registers and then conditionally writes LSE. Invalid query rows still execute collective reads and arithmetic; only logical destination stores are masked. No LSE store occurs when absent.

## t_503c7fc9355d4484 | pipeline.verb | 573 chars | 1 declarations

root:565 design.tile_pipeline.phase.main.role.MMA.info.verb.mma

P(n)@VTile(n) performs the value_state rewrite into Y. First PV initializes Y; later PV follows correction. Products issue in one ordered tensor stream, and a completion notification covers all earlier products. Earlier PV reads of an aliased P slot finish before a later QK overwrites that slot. This ordering also protects the tiny reread before the same S slot can recur. PV completion releases its K/V slot. For KVSlots=3 the release notification cycles with the three S/P slots; for KVSlots=4 it additionally cycles with the four K/V slots, preventing parity aliasing.

## t_5357a0d4a9ab79ee | pipeline.verb | 140 chars | 1 declarations

bootstrap:263 design.tile_pipeline.phase.main.role.MMA.info.verb.qk

S(t) performs the scores rewrite from QTile and KTile(t). It uses the same ordered tensor stream and cumulative completion guarantee as mma.

## t_5aca5a24c7550f70 | pipeline.verb | 98 chars | 1 declarations

root:538 design.tile_pipeline.phase.main.role.Correction.info.verb.wait

Next(w) consumes the next descriptor or terminal publication after this worktile's output actions.

## t_5ba6f28aa6da6ffe | pipeline.verb | 468 chars | 1 declarations

root:598 design.tile_pipeline.phase.main.role.Loader.info.verb.load

KVSlots = 3 if Scatter, otherwise 4. QTile(w) loads the complete query tile into the single Query buffer; for w=0 it reads the preassigned Geo entry directly and can start during CTA initialization. For later work it uses the preceding Next descriptor. KV(n) loads KTile(n) and VTile(n); done means all bytes of both are readable. Each Q/K transfer has a 128-channel segment and a 64-channel tail; V has one 128-channel segment, with the padded views defined in tiles.

## t_5c2a81998832110e | pipeline.verb | 172 chars | 1 declarations

root:560 design.tile_pipeline.phase.main.role.MMA.info.verb.wait.meaning

QTile(w) waits for the Q transfer; Next(w) consumes the next descriptor or sentinel. Neither wait drains previously issued matrix operations beyond its stated predecessors.

## t_65b788018df40c53 | pipeline.verb | 178 chars | 1 declarations

bootstrap:200 design.tile_pipeline.phase.main.role.Softmax.info.verb.load

Scores(t) reads both halves of S(t) into ScoreRegisters; done releases the score read for subsequent QK slot reuse. The second-half transfer overlaps first-half masking/max work.

## t_8683418b781b47a9 | pipeline.verb | 163 chars | 1 declarations

root:561 design.tile_pipeline.phase.main.role.MMA.info.verb.qk

S(n) performs the scores rewrite from the current worktile's QTile and KTile(n). It uses the same ordered tensor stream and cumulative completion guarantee as mma.

## t_8c3b8e4c38cd9c1b | pipeline.verb | 268 chars | 1 declarations

bootstrap:261 design.tile_pipeline.phase.main.role.MMA.info.verb.latch.meaning

Latch the preassigned Item after the initialization join. One elected issuer handles matrix operations; the warp allocates the full TMEM resource before use and frees it after the final CTA join, when all roles and their required reads and output writes have finished.

## t_90b47cd9d8e2e6c3 | pipeline.verb | 240 chars | 1 declarations

bootstrap:234 design.tile_pipeline.phase.main.role.Correction.info.verb.latch

Latch the preassigned Item geometry after the initialization join; its four warps handle the same four consecutive 32-row groups as Softmax. Initial output credit declares the new accumulator and final-statistic storage free of old readers.

## t_95be4c62f9d062f9 | pipeline.verb | 165 chars | 1 declarations

root:607 design.tile_pipeline.phase.main.role.Loader.info.verb.wait.meaning

Next(w) waits for the server publication. A terminal descriptor performs no next-Q transfer and no next worktile; outstanding earlier fills are not implicitly reset.

## t_965b351e892ebc31 | pipeline.verb | 152 chars | 1 declarations

root:559 design.tile_pipeline.phase.main.role.MMA.info.verb.latch.meaning

Latch Item(w). One elected issuer handles matrix operations; the warp allocates the full TMEM resource before use and frees it after the final CTA join.

## t_9ba6bc6e0d7f0ab9 | pipeline.verb | 120 chars | 1 declarations

bootstrap:302 design.tile_pipeline.phase.main.role.Loader.info.verb.latch.meaning

Latch Item after issuing Q and completing the initialization join; retain its K/V base and head through all local fills.

## t_9d8325bcf6b10685 | pipeline.verb | 380 chars | 1 declarations

bootstrap:295 design.tile_pipeline.phase.main.role.Loader.info.verb.load

KVSlots = 3 if Scatter, otherwise 4. QTile loads the complete query tile into Query directly from the preassigned Geo entry and can start during CTA initialization. KV(t) loads KTile(t) and VTile(t); done means all bytes of both are readable. Each Q/K transfer has a 128-channel segment and a 64-channel tail; V has one 128-channel segment, with the padded views defined in tiles.

## t_a115c8533f9166de | pipeline.verb | 110 chars | 1 declarations

root:486 design.tile_pipeline.phase.main.role.Softmax.info.verb.compute

Weights(n) performs the weights rewrite after Alpha publication, retains packed P in registers and advances D.

## t_a699af457e9f4c83 | pipeline.verb | 464 chars | 1 declarations

root:626 design.tile_pipeline.phase.main.role.Queue.info.verb.publish

Next(w) executes the atomic_queue rule, copies successful geometry or the negative tile-count terminal descriptor into Geometry, then release-publishes it for all other roles. The terminal publication still occurs and has a completion event; no worktile is created for it. All consumers latch their old descriptor before that slot is overwritten. Geometry and ring storage are allocated once per CTA; only the queue word persists between ordered kernel executions.

## t_b701f47e8ff168d2 | pipeline.verb | 276 chars | 1 declarations

bootstrap:208 design.tile_pipeline.phase.main.role.Softmax.info.verb.produce

Final captures Fsum,Fmax on the last tile. For short requests, the first row group rereads the first 32 S columns and VTile for the tiny rewrite, storing TinyY and TinyD into the selected final denominator. Other rows keep the main denominator. P publication remains required.

## t_c0acad3503d2b64b | pipeline.verb | 110 chars | 1 declarations

bootstrap:207 design.tile_pipeline.phase.main.role.Softmax.info.verb.compute

Weights(t) performs the weights rewrite after Alpha publication, retains packed P in registers and advances D.

## t_c454ed18330cc42b | pipeline.verb | 337 chars | 1 declarations

root:472 design.tile_pipeline.phase.main.role.Softmax.info.verb.latch

Item(w) is the assigned geometry latched into this role's registers; N(w) denotes its positive key-tile count. Item(0) comes from the preassigned table entry and the CTA initialization join. Reset M,D for this worktile, but do not reset any ring or notification epoch. Each warp handles its consecutive 32-row group throughout this role.

## t_c7c57c479a6dc5dd | pipeline.verb | 338 chars | 1 declarations

root:480 design.tile_pipeline.phase.main.role.Softmax.info.verb.publish

Alpha(n) updates M and publishes all row scales before exponentiation. P(n) copies PackedRegisters into its aliased TMEM view and completes its writes before publication. The two-slot scale relay is safe because the QK(n+2) completion preceding its next write follows PV(n), which already required the applicable rescale to read Alpha(n).

## t_d751710809390bcd | pipeline.verb | 111 chars | 1 declarations

root:522 design.tile_pipeline.phase.main.role.Correction.info.verb.latch

Latch Item(w) geometry for this role; its four warps handle the same four consecutive 32-row groups as Softmax.

## t_e78e8a3f1a5965f8 | pipeline.verb | 518 chars | 1 declarations

bootstrap:236 design.tile_pipeline.phase.main.role.Correction.info.verb.store

Output evaluates the output rewrite using Fsum,Fmax. Ordinary rows drain Y in four serial 32-column read/wait/convert/store chunks, using OutputChunk registers; tiny rows read and store all TinyY columns instead. The source returns output-state credit after its final scratch read; this CTA has no later worktile to consume it. Invalid query rows still execute collective reads and arithmetic; only destination stores are masked. Write LSE exactly when present. done covers all output writes and required source reads.

## t_f8bda7defe1b0895 | pipeline.verb | 79 chars | 1 declarations

bootstrap:262 design.tile_pipeline.phase.main.role.MMA.info.verb.wait.meaning

QTile waits for the initial Q transfer; it does not drain unrelated operations.

## t_23a8613ec5939b7f | realization.choice | 123 chars | 2 declarations

bootstrap:313 realization.require[0].choice; root:645 realization.require[0].choice

Use the source SM10x tcgen05 FP8 matrix path with TMEM accumulation and cumulative tcgen05.commit completion notifications.

## t_b5529df51df427e4 | realization.choice | 103 chars | 2 declarations

bootstrap:316 realization.require[1].choice; root:648 realization.require[1].choice

Bind Q and K/V loads to cp.async.bulk.tensor transfers with mbarrier transaction-byte completion waits.

## t_fff95173eb1b616e | realization.choice | 135 chars | 2 declarations

bootstrap:319 realization.require[2].choice; root:651 realization.require[2].choice

Implement TMEM reads and writes with converged full-warp tcgen05 collective accesses and the corresponding load/store completion waits.

## t_60bbecf887216568 | realization.reason | 143 chars | 2 declarations

bootstrap:320 realization.require[2].reason; root:652 realization.require[2].reason

Invalid output rows remain participants in collective accesses, and score/accumulator release must follow the actual completion of those reads.

## t_7a11ba095dc1f005 | realization.reason | 126 chars | 2 declarations

bootstrap:317 realization.require[1].reason; root:649 realization.require[1].reason

Issue alone does not make transferred bytes readable; the three-slot and four-slot reuse notifications have different periods.

## t_7ca293f6915afc3c | realization.reason | 149 chars | 2 declarations

bootstrap:314 realization.require[0].reason; root:646 realization.require[0].reason

The aliased S/P views rely on ordered earlier-operand reads before later tensor writes, and consumers rely on completion covering preceding products.

## t_192ed494b4e5bc3d | schedule.describe | 117 chars | 2 declarations

bootstrap:32 design.work_schedule.phase.main.describe; root:295 design.work_schedule.phase.main.describe

Compute all score, normalization, main PV, optional tiny correction and output work for one head and one query block.

## t_ae3b378621a2cf50 | schedule.exit | 117 chars | 1 declarations

root:353 design.work_schedule.phase.main.schedule.worker_assignment.exit

The terminal descriptor has been observed and all assigned work and in-flight reads and output writes have completed.

## t_f00e9ac4965d1d28 | schedule.refill | 597 chars | 1 declarations

root:343 design.work_schedule.phase.main.schedule.worker_assignment.refill.atomic_queue

A private global uint32 counter starts at zero when allocated. At the start of each assigned worktile's last key iteration, request one pop; a dedicated server atomically increments the counter and maps its old value p to Item=worker_count+p. Publish that worktile if Item<WorkCount, otherwise publish a terminal descriptor and stop popping. Exactly WorkCount pops occur: WorkCount-worker_count successful refills plus one failed pop per CTA. The pop returning WorkCount-1 resets the counter to zero for the next ordered launch. This queue is private scratch, separate from the supplied Workspace.

## t_3e8ea11d69476421 | schedule.strategy | 162 chars | 1 declarations

root:317 design.work_schedule.phase.main.mapping.strategy

LaunchSMCount is the host-derived count defined in dispatch.when; it fixes the submitted CTA count, not an asserted residency guarantee. Each CTA has 384 threads.

## t_5173ba54ca7e4ee7 | schedule.strategy | 1174 chars | 2 declarations

bootstrap:58 design.work_schedule.phase.main.schedule.strategy; root:321 design.work_schedule.phase.main.schedule.strategy

Before the main launch, a nonblocking side stream copies SeqLens, CuQ and CuKV to pinned host buffers (4*(3*B+2) bytes); the host waits for these copies and constructs WorkCount table entries of 48 bytes each. Start from h ascending, b ascending, a descending. Stable-sort by N descending exactly when every q_b>=Bq and Scatter is false. Let CachedL2 and CachedSM be launch_impl's cached device-query results, refreshed while CachedL2 is zero. Let ScatterBytes = min(H*B,CachedSM if positive else 148)*max_b(k_b)*(QKDim+ValueDim). Scatter = CachedL2>0 and 4*ScatterBytes>3*CachedL2. This is a host locality heuristic, not a measured cache-capacity guarantee. Upload the complete Geo table on the side stream and record table_event; the main/current stream waits for that upload before the one main kernel. Only allocation capacity and descriptor encodings are cached, not reused numerical metadata. During external recording this host prefix still runs; later execution uses the recorded binding and the table contents actually read. Sinks has a payload only when present. LSE is stored only when ReturnLSE; neither option changes the work set or the main kernel submission.

## t_a486b2db9c811fee | schedule.strategy | 365 chars | 2 declarations

bootstrap:34 design.work_schedule.phase.main.decomposition.strategy; root:297 design.work_schedule.phase.main.decomposition.strategy

Bq = 128 and Bk = 128. Item is the ordinal of a geometry-table entry Geo[Item]=(b,h,a,q0,qv,N,CuQ[b]+q0,CuKV[b]); its order is defined below. Each worktile consumes QTile once and traverses KTile(t),VTile(t) for t=0..N-1. All reductions and optional tiny correction remain inside it. Each valid output row/head belongs to exactly one entry; no cross-worktile merge.

## t_fdd997e281a17eed | schedule.strategy | 141 chars | 1 declarations

bootstrap:54 design.work_schedule.phase.main.mapping.strategy

WorkCount is defined in dispatch.when. Submit one 384-thread CTA per table entry; the submitted count is not an asserted residency guarantee.

## t_a2408993a8a76239 | schedule.thesis | 231 chars | 1 declarations

bootstrap:24 design.work_schedule.thesis

Assign one head/query-block worktile to each CTA and keep the full key recurrence local. Host table ordering retains its locality and heavy-first rules; submitting all worktiles avoids refill traffic but has no dynamic rebalancing.

## t_b0dd88dbf7949a92 | schedule.thesis | 238 chars | 1 declarations

root:287 design.work_schedule.thesis

Give each persistent CTA one head/query-block worktile and keep the entire key recurrence local. A host-built geometry table and early atomic refill balance unequal tile counts while retaining head locality when sorting would scatter K/V.
