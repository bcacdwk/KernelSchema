# extract_fmha: full open-text inventory

These are exact-text groups, not proof of semantic equivalence across contexts.

## t_53b70444903670e7 | algorithm_contract.math | 1614 chars | 1 declarations

root:101 contract.algorithm_requirement.compute[0].math

B, H, KH, Sq, Skv and D are positive integers. 1 <= B <= 16, 1 <= Sq <= 8192, 1 <= Skv <= 262144, Sq <= Skv, B*Sq <= 16384, B*Skv <= 1048576, and H % KH = 0. (H,KH,D) belongs to {(8,2,128), (8,2,256), (8,8,128), (8,8,256), (16,1,128), (16,1,256), (16,4,128), (32,8,128)}. These are dynamic ranges; the ten fixed shape cases are only samples. Q/K/V and O use logical batch-head-sequence-channel axes with dense batch-sequence-head-channel physical storage. LSE is contiguous B,H,Sq. Scale tensors have the packed storage shapes above; sequence padding affects scale storage only and does not extend the logical attention domain. None denotes Python None. The prepared scale is D^(-1/2). window_left=None or -1 disables the left bound; preparation maps -1 to None. seq_kv_lens is None for causal calls; noncausal calls require the tensor with every element equal to Skv. Omitting it is valid only for causal calls. lse_tensor may be None or omitted for any call: then LSE=None and there is no LSE storage write. Otherwise LSE is written into lse_tensor. No paged, ragged, sink, gate, per-tensor scaling or quantized-output mode is part of this contract. The normal/tiny/peaked/zero input-construction families do not supply any hidden unquantized operands to the answer. W is a nonnegative integer, allocated from the prepared implementation's scratch_workspace_bytes(); scratch initialization is unspecified. The call returns Python None. Only O, present LSE, and workspace may be written; preserve caller buffer identities and all other storage, including guards. O and LSE do not depend on their initial contents.

## t_6e38e323a2b8840d | algorithm_contract.math | 386 chars | 1 declarations

root:154 contract.algorithm_requirement.compute[2].math

Batch b and query head h are independent; h uses KV head k=floor(h/(H/KH)). For each query i, let center=i+Skv-Sq. A key j is visible iff 0 <= j < Skv, either is_causal=false or j <= center, and either window_left is None/-1 or j >= center-window_left. S[b,h,i,j] = D^(-1/2) * sum_{d=0}^{D-1} Qd[b,h,i,d]*Kd[b,k,j,d] for visible keys, and -inf otherwise. Both mask bounds are inclusive.

## t_aa745a634c60463d | algorithm_contract.math | 1587 chars | 1 declarations

root:164 contract.algorithm_requirement.compute[3].math

For each (b,h,i), process numerical groups t=0,...,ceil_div(Skv,KeyGroup)-1 with J_t={j | t*KeyGroup <= j < min((t+1)*KeyGroup,Skv)} in increasing order. This order defines rounding semantics, not a required execution schedule. Suppress the independent (b,h,i) indices on the row states M, Z and N. Initialize M(-1)=-inf, Z(-1)=0, N(-1)[d]=0 for 0 <= d < D. Let beta=16 when D=128, otherwise beta=1. For each group: a=max_{j in J_t} S[b,h,i,j]. M(t)=a if a-M(t-1) > 4*ln(2), otherwise M(t-1); the comparison is false for NaN, including -inf-(-inf). safe=M(t) if M(t) is finite, otherwise 0. alpha=exp(M(t-1)-safe) if M(t-1) is finite, otherwise 0. For j in J_t, P[b,h,i,j]=beta*exp(S[b,h,i,j]-safe) and Pq[b,h,i,j]=fp8_e4m3fn(clamp(P[b,h,i,j],0,ProbabilityMax)), using RNE. N(t)[d]=alpha*N(t-1)[d]+sum_{j in J_t} Pq[b,h,i,j]*Vd[b,k,j,d]. Z(t)=alpha*Z(t-1)+sum_{j in J_t} P[b,h,i,j]. Pq has unit scale; Z uses the unrounded probabilities. Fully masked groups have P=Pq=0 and zero new denominator contribution. Their numerator still follows the FP32 recurrence, including 0*Inf=NaN and 0*NaN=NaN for nonfinite decoded values or prior numerator states. After the last group T=ceil_div(Skv,KeyGroup)-1, O[b,h,i,d]=bf16(N(T)[d]/Z(T)) if Z(T)>0, otherwise bf16(0). Independently, when lse_tensor is present, LSE[b,h,i]=logsumexp_{j=0}^{Skv-1} S[b,h,i,j] in natural-log units, without probability rounding or beta scaling. An entirely masked row has O=0 and LSE=-inf. The valid rectangular mask domain always includes the center key for finite inputs, but the empty-row rule fixes the boundary.

## t_c448416b8b285a62 | algorithm_contract.math | 1157 chars | 1 declarations

root:127 contract.algorithm_requirement.compute[1].math

All following numerical arithmetic is FP32 between explicit casts; the reference disables TF32. Indices and byte addresses are integers. For a padded logical scale matrix whose column count C is a multiple of ScaleCols, define its byte offset f(r,c,C) = floor(r/ScaleRows)*(C/ScaleCols)*(ScaleRows*ScaleCols) + floor(c/ScaleCols)*(ScaleRows*ScaleCols) + (r % Block)*(ScaleRows/Block)*ScaleCols + floor((r % ScaleRows)/Block)*ScaleCols + c % ScaleCols. Byte addressing below starts at each scale tensor's first logical byte. Let Rq=ScaleRows*ceil_div(Sq,ScaleRows), Rk=ScaleRows*ceil_div(Skv,ScaleRows), Cq=D/Block, Cv=Rk/Block. For 0 <= b < B, 0 <= h < H, 0 <= k < KH, 0 <= i < Sq, 0 <= j < Skv, 0 <= d < D: Qd[b,h,i,d] = q_tensor[b,h,i,d] * 2^(sf_q.flat[(b*H+h)*Rq*Cq + f(i,floor(d/Block),Cq)]-ExponentBias). Kd[b,k,j,d] = k_tensor[b,k,j,d] * 2^(sf_k.flat[(b*KH+k)*Rk*Cq + f(j,floor(d/Block),Cq)]-ExponentBias). Vd[b,k,j,d] = v_tensor[b,k,j,d] * 2^(sf_v.flat[f(d,(b*KH+k)*Cv+floor(j/Block),B*KH*Cv)]-ExponentBias). Thus V's scale matrix packs batch/head groups across columns, while Q/K pack them across rows. E8M0 byte 0 means 2^(-ExponentBias), not zero.

## t_03fc9f2e712b44da | algorithm_contract.meaning | 40 chars | 1 declarations

root:95 contract.algorithm_requirement.input.sf_o.meaning

Python None disables block-scaled output

## t_0dddb777b7ec1d40 | algorithm_contract.meaning | 47 chars | 1 declarations

root:76 contract.algorithm_requirement.input.seq_kv_lens.meaning

Prepared call's optional rectangular KV lengths

## t_110470a37f173f06 | algorithm_contract.meaning | 54 chars | 1 declarations

root:90 contract.algorithm_requirement.input.descale_v.meaning

Python None disables a separate per-tensor value scale

## t_1f1d2e56899d2314 | algorithm_contract.meaning | 12 chars | 1 declarations

root:151 contract.algorithm_requirement.compute[1].symbol.Kd.meaning

Decoded keys

## t_22e0ad68d462b1b0 | algorithm_contract.meaning | 51 chars | 1 declarations

root:35 contract.algorithm_requirement.constant.ProbabilityMax.meaning

Saturation limit before the probability E4M3FN cast

## t_2bf8d0c4e1d59ca1 | algorithm_contract.meaning | 35 chars | 1 declarations

root:91 contract.algorithm_requirement.input.scale_o.meaning

Python None disables output scaling

## t_2e41beb302fd7bab | algorithm_contract.meaning | 42 chars | 1 declarations

root:192 contract.algorithm_requirement.compute[3].symbol.M.meaning

Delayed normalizer after numerical group t

## t_31da960d49afa5ed | algorithm_contract.meaning | 60 chars | 1 declarations

root:33 contract.algorithm_requirement.constant.KeyGroup.meaning

Consecutive keys in one reference probability-rounding group

## t_38f22c32eb84d27c | algorithm_contract.meaning | 36 chars | 1 declarations

root:83 contract.algorithm_requirement.input.sinks.meaning

Python None disables attention sinks

## t_475afecdfb7a0c06 | algorithm_contract.meaning | 59 chars | 1 declarations

root:194 contract.algorithm_requirement.compute[3].symbol.N.meaning

Rounded-probability value numerator after numerical group t

## t_49b95e43ff2bff64 | algorithm_contract.meaning | 32 chars | 1 declarations

root:30 contract.algorithm_requirement.constant.Block.meaning

Elements sharing one MXFP8 scale

## t_57b6d813f88c8fa0 | algorithm_contract.meaning | 39 chars | 1 declarations

root:93 contract.algorithm_requirement.input.block_table_v.meaning

Python None selects dense value storage

## t_5fa4a2b218c18aae | algorithm_contract.meaning | 29 chars | 1 declarations

root:31 contract.algorithm_requirement.constant.ScaleRows.meaning

Rows in one scale-layout atom

## t_63286582843722df | algorithm_contract.meaning | 20 chars | 1 declarations

root:37 contract.algorithm_requirement.input.q_tensor.meaning

Stored query payload

## t_63b79dc8ba4c0d00 | algorithm_contract.meaning | 79 chars | 1 declarations

root:51 contract.algorithm_requirement.input.sf_v.meaning

Packed F8_128x4 E8M0 bytes scaling values along the key axis in groups of Block

## t_67c6d505d69b5d21 | algorithm_contract.meaning | 37 chars | 1 declarations

root:92 contract.algorithm_requirement.input.block_table.meaning

Python None selects dense key storage

## t_6a6563508ea3edaf | algorithm_contract.meaning | 20 chars | 1 declarations

root:39 contract.algorithm_requirement.input.v_tensor.meaning

Stored value payload

## t_6ceb59f228a8bcdb | algorithm_contract.meaning | 39 chars | 1 declarations

root:162 contract.algorithm_requirement.compute[2].symbol.S.meaning

Scaled and masked FP32 attention scores

## t_6f2797375cc61350 | algorithm_contract.meaning | 87 chars | 1 declarations

root:98 contract.algorithm_requirement.output.LSE.meaning

Optional natural-log softmax normalizer written into the full logical lse_tensor region

## t_76299155712401cf | algorithm_contract.meaning | 48 chars | 1 declarations

root:67 contract.algorithm_requirement.input.is_causal.meaning

Prepared-model bottom-right causal-mask selector

## t_764566549cb4be99 | algorithm_contract.meaning | 70 chars | 1 declarations

root:56 contract.algorithm_requirement.input.o_tensor.meaning

Caller-owned output destination; its initial contents are not operands

## t_7c44885434ef425b | algorithm_contract.meaning | 18 chars | 1 declarations

root:38 contract.algorithm_requirement.input.k_tensor.meaning

Stored key payload

## t_7d7daa5d2e2f6a8c | algorithm_contract.meaning | 40 chars | 1 declarations

root:69 contract.algorithm_requirement.input.window_left.meaning

Prepared-model integer left-window width

## t_8b0fd58217ac7895 | algorithm_contract.meaning | 32 chars | 1 declarations

root:32 contract.algorithm_requirement.constant.ScaleCols.meaning

Columns in one scale-layout atom

## t_8ef763882864f87c | algorithm_contract.meaning | 14 chars | 1 declarations

root:152 contract.algorithm_requirement.compute[1].symbol.Vd.meaning

Decoded values

## t_8f07d34d889868f6 | algorithm_contract.meaning | 101 chars | 1 declarations

root:97 contract.algorithm_requirement.output.O.meaning

Attention result written into the full logical o_tensor region without replacing the caller's storage

## t_9196020d17de9531 | algorithm_contract.meaning | 45 chars | 1 declarations

root:84 contract.algorithm_requirement.input.seq_q_lens.meaning

Python None selects rectangular query lengths

## t_a0aa84b7020ac599 | algorithm_contract.meaning | 68 chars | 1 declarations

root:41 contract.algorithm_requirement.input.sf_q.meaning

Packed F8_128x4 E8M0 bytes scaling query channels in groups of Block

## t_a1058c5b3917ea97 | algorithm_contract.meaning | 54 chars | 1 declarations

root:88 contract.algorithm_requirement.input.descale_q.meaning

Python None disables a separate per-tensor query scale

## t_a19246ec8b6839d3 | algorithm_contract.meaning | 44 chars | 1 declarations

root:86 contract.algorithm_requirement.input.current_stream.meaning

Python None uses the caller's current stream

## t_a703366dcfb331e3 | algorithm_contract.meaning | 72 chars | 1 declarations

root:61 contract.algorithm_requirement.input.lse_tensor.meaning

Optional caller-owned LSE destination; initial contents are not operands

## t_b8588fc8d91f15a8 | algorithm_contract.meaning | 42 chars | 1 declarations

root:87 contract.algorithm_requirement.input.amax_o.meaning

Python None disables output-amax reporting

## t_c275147fe1917461 | algorithm_contract.meaning | 58 chars | 1 declarations

root:196 contract.algorithm_requirement.compute[3].symbol.Pq.meaning

Saturated E4M3FN probabilities used in the value numerator

## t_c8461c484075d61f | algorithm_contract.meaning | 54 chars | 1 declarations

root:85 contract.algorithm_requirement.input.scale_softmax.meaning

Python None uses the prepared scale defined in compute

## t_c9c64327cdb3872a | algorithm_contract.meaning | 52 chars | 1 declarations

root:89 contract.algorithm_requirement.input.descale_k.meaning

Python None disables a separate per-tensor key scale

## t_cc301ee076db6b6a | algorithm_contract.meaning | 65 chars | 1 declarations

root:82 contract.algorithm_requirement.input.workspace.meaning

Caller-provided mutable scratch with no mathematical output value

## t_cd4bb10d807c99b4 | algorithm_contract.meaning | 68 chars | 1 declarations

root:195 contract.algorithm_requirement.compute[3].symbol.P.meaning

Unrounded probabilities in each key's numerical group and normalizer

## t_d0cd17c2b2005e6f | algorithm_contract.meaning | 66 chars | 1 declarations

root:46 contract.algorithm_requirement.input.sf_k.meaning

Packed F8_128x4 E8M0 bytes scaling key channels in groups of Block

## t_df325a961322aef4 | algorithm_contract.meaning | 15 chars | 1 declarations

root:150 contract.algorithm_requirement.compute[1].symbol.Qd.meaning

Decoded queries

## t_ead493b42d780320 | algorithm_contract.meaning | 34 chars | 1 declarations

root:94 contract.algorithm_requirement.input.gate.meaning

Python None disables output gating

## t_f57b3f598c238be5 | algorithm_contract.meaning | 32 chars | 1 declarations

root:34 contract.algorithm_requirement.constant.ExponentBias.meaning

Bias of the stored E8M0 exponent

## t_f588b5afc7279771 | algorithm_contract.meaning | 53 chars | 1 declarations

root:193 contract.algorithm_requirement.compute[3].symbol.Z.meaning

Unrounded softmax denominator after numerical group t

## t_0b1ec3ba84f2230b | algorithm_contract.range | 85 chars | 1 declarations

root:65 contract.algorithm_requirement.input.lse_tensor.range

Arbitrary initial contents, including NaN and +/-inf; fully overwritten when present.

## t_5c259c3cde3ec891 | algorithm_contract.range | 79 chars | 1 declarations

root:44 contract.algorithm_requirement.input.sf_q.range

Active E8M0 codes are 0 through 254; padding bytes do not define extra queries.

## t_61d0eddc89060534 | algorithm_contract.range | 25 chars | 1 declarations

root:80 contract.algorithm_requirement.input.seq_kv_lens.range

Every element equals Skv.

## t_887aefd6c982bd42 | algorithm_contract.range | 80 chars | 1 declarations

root:59 contract.algorithm_requirement.input.o_tensor.range

Arbitrary initial contents, including nonfinite bit patterns; fully overwritten.

## t_a08d81062ce9c3db | algorithm_contract.range | 76 chars | 2 declarations

root:49 contract.algorithm_requirement.input.sf_k.range; root:54 contract.algorithm_requirement.input.sf_v.range

Active E8M0 codes are 0 through 254; padding bytes do not define extra keys.

## t_dd5a200793d7f1ca | algorithm_contract.range | 13 chars | 1 declarations

root:67 contract.algorithm_requirement.input.is_causal.range

{false, true}

## t_3ee80cc47d632f5e | contract.compute_name | 39 chars | 1 declarations

root:126 contract.algorithm_requirement.compute[1].name

Decode the stored block-scaled operands

## t_42e4805031ec9025 | contract.compute_name | 43 chars | 1 declarations

root:153 contract.algorithm_requirement.compute[2].name

Grouped-query scores and bottom-right masks

## t_5d77c27f83cf1e59 | contract.compute_name | 45 chars | 1 declarations

root:163 contract.algorithm_requirement.compute[3].name

Probability rounding and normalized value sum

## t_b8c8c9aa0cb77baa | contract.compute_name | 27 chars | 1 declarations

root:100 contract.algorithm_requirement.compute[0].name

Domain and external storage

## t_09c1b73adc4eb145 | contract.tolerance | 627 chars | 1 declarations

root:217 contract.algorithm_requirement.golden_reference.tolerance.LSE

Candidate and reference LSE must both be None or both be tensors; None has no numerical comparison. Source record for mutated lse_tensor when present: metadata.benchmark_contract.correctness_tolerances.default declares atol=0.01, rtol=0.05 and max_mismatch_fraction=0.0. CERTIFIED_BEST records unchanged comparator/tolerances but does not supply the comparator. The formula, relative-error basis, comparator precision and NaN/Inf handling are unspecified; no schema-default criterion is adopted. This record preserves the known settings without claiming numerical guarantees or reproduction of the external acceptance boundary.

## t_7ef318086245418b | contract.tolerance | 1118 chars | 1 declarations

root:200 contract.algorithm_requirement.golden_reference.tolerance.O

Source records for candidate versus reference mutated o_tensor: metadata.benchmark_contract.correctness_tolerances.default declares atol=0.01, rtol=0.05 and max_mismatch_fraction=0.0. The BF16-partial A/B margin experiment documented in source@6e4ca91bd91d998cac99e8dadd83838e153c679d:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_common.cuh reports ratio=max_i(abs(actual[i]-ref[i])/(atol+rtol*abs(actual[i]))) over all O elements. Here actual is the experiment's candidate O and ref its frozen-reference O; relative_to=candidate. With the stated tolerances and zero mismatch fraction, one elementwise ratio above 1 fails the shape. This records that BF16 margin experiment on legal gated geometries; its binding to the final external O/LSE checking routes is not supplied. CERTIFIED_BEST records unchanged comparator and tolerances but does not supply the comparator implementation. Comparator precision and NaN/Inf handling remain unspecified; no schema-default criterion is adopted. This scoped record does not establish a general acceptance formula or a numerical guarantee.

## t_04b09fd8021e66a0 | dispatch.md | 311 chars | 1 declarations

dispatch.md:29 dispatch.md

| [tcg_d128_fscan_tma_direct_wide](tcg_d128_fscan_tma_direct_wide.yaml) | tcg_d128_fscan_tma_direct_narrow | D128 split，融合扫描，TMA，PSK 直接写回，宽归并 | 归并改为四个 128 线程组分担 split；共享 numerator/warp 分母部分和，屏障后由第零组输出。 | launch_sequence 的 tcgen05、use_split、D128 分支，pl.scan_fused 为真、psk 为真；launch_tcg 选 TMA，launch_combine 选宽归并。 |

## t_07aa718300935030 | dispatch.md | 281 chars | 1 declarations

dispatch.md:43 dispatch.md

| [tcg_d256_flat_tma_wide](tcg_d256_flat_tma_wide.yaml) | tcg_d256_flat_tma_narrow | D256 split，平坦扫描，TMA，宽归并 | 归并改为四个 128 线程组分担 split；共享 numerator/warp 分母部分和，屏障后由第零组输出。 | launch_sequence 的 tcgen05、use_split、D256 分支，pl.scan_fused 为假、scan2p 为假；launch_tcg 选 TMA，launch_combine 选宽归并。 |

## t_086d4b5e1aef50b5 | dispatch.md | 332 chars | 1 declarations

dispatch.md:39 dispatch.md

| [tcg_d128_segmented_tma_xpose_wide](tcg_d128_segmented_tma_xpose_wide.yaml) | tcg_d128_segmented_tma_xpose_narrow | D128 split，分段扫描，TMA，非 PSK 转置写回，宽归并 | 归并改为四个 128 线程组分担 split；共享 numerator/warp 分母部分和，屏障后由第零组输出。 | launch_sequence 的 tcgen05、use_split、D128 分支，pl.scan_fused 为假、scan2p 为真、psk 为假；launch_tcg 选 TMA，launch_combine 选宽归并。 |

## t_0ccba735461caa9c | dispatch.md | 339 chars | 1 declarations

dispatch.md:35 dispatch.md

| [tcg_d128_segmented_cp_xpose_wide](tcg_d128_segmented_cp_xpose_wide.yaml) | tcg_d128_segmented_cp_xpose_narrow | D128 split，分段扫描，cp.async，非 PSK 转置写回，宽归并 | 归并改为四个 128 线程组分担 split；共享 numerator/warp 分母部分和，屏障后由第零组输出。 | launch_sequence 的 tcgen05、use_split、D128 分支，pl.scan_fused 为假、scan2p 为真、psk 为假；launch_tcg 选 cp.async，launch_combine 选宽归并。 |

## t_179003c7f84be775 | dispatch.md | 113 chars | 1 declarations

dispatch.md:59 dispatch.md

计划覆盖固定默认构建在共享 Contract 域内的实际 host 路由，不把十个固定样本当作全域。源码的明确输入检查失败仍会报错；描述符编码失败的 cp.async 路径是实际分支，不构成任意布局、硬件或发射失败的通用保证。

## t_1e1c132eb66733a1 | dispatch.md | 245 chars | 1 declarations

dispatch.md:14 dispatch.md

| [tc_split_segmented_narrow](tc_split_segmented_narrow.yaml) | tc_split_flat_narrow | D128/D256，mma.sync split，分段扫描，窄归并 | 扫描改为段最大值→分段重放，增加 segmax 物化及一次提交；其余 split 组织保持。 | launch_sequence 的 mma.sync、use_split 分支，scan2p 为真；launch_combine 选择窄归并。 |

## t_1e663b9f3f69fc0b | dispatch.md | 201 chars | 1 declarations

dispatch.md:8 dispatch.md

| [fused_d128_key](fused_d128_key.yaml) | root | D128，查询 CTA 较少、可见组较短的融合计算 | 双 warpgroup 分担同组 key；cp.async 四槽环、双 score tile、跨组 QK/PV 预取，分子驻 TMEM。 | launch_sequence 的非 split、key_split 为真且 gsplit 为假分支。 |

## t_2304aaf0197a9461 | dispatch.md | 7 chars | 1 declarations

dispatch.md:57 dispatch.md

## 覆盖缺口

## t_2db01478868aee72 | dispatch.md | 324 chars | 1 declarations

dispatch.md:19 dispatch.md

| [tcg_d128_flat_cp_xpose_wide](tcg_d128_flat_cp_xpose_wide.yaml) | tcg_d128_flat_cp_xpose_narrow | D128 split，平坦扫描，cp.async，非 PSK 转置写回，宽归并 | 归并改为四个 128 线程组分担 split；共享 numerator/warp 分母部分和，屏障后由第零组输出。 | launch_sequence 的 tcgen05、use_split、D128 分支，pl.scan_fused 为假、scan2p 为假、psk 为假；launch_tcg 选 cp.async，launch_combine 选宽归并。 |

## t_2dcb941dd5ff58e3 | dispatch.md | 334 chars | 1 declarations

dispatch.md:28 dispatch.md

| [tcg_d128_fscan_tma_direct_narrow](tcg_d128_fscan_tma_direct_narrow.yaml) | tcg_d128_flat_tma_direct_narrow | D128 split，融合扫描，TMA，PSK 直接写回，窄归并 | 改为 group-max→局部扫描融合 split→加权归并三次提交；保留各 split 的 part_m，末 split 发布 mtail，归并重缩放。 | launch_sequence 的 tcgen05、use_split、D128 分支，pl.scan_fused 为真、psk 为真；launch_tcg 选 TMA，launch_combine 选窄归并。 |

## t_32b044faa5d8348b | dispatch.md | 38 chars | 1 declarations

dispatch.md:3 dispatch.md

| 节点 | 直接父 | workload | 相对父差异 | 选择依据 |

## t_34ed6e6636c42644 | dispatch.md | 355 chars | 1 declarations

dispatch.md:16 dispatch.md

| [tcg_d128_flat_cp_direct_narrow](tcg_d128_flat_cp_direct_narrow.yaml) | tcg_d128_flat_tma_direct_narrow | D128 split，平坦扫描，cp.async，PSK 直接写回，窄归并 | group-max 与 split 搬运改为 cp.async，按 copy-group 提交/等待保护 payload 和 scale；保留扫描及部分和写回组织。 | launch_sequence 的 tcgen05、use_split、D128 分支，pl.scan_fused 为假、scan2p 为假、psk 为真；launch_tcg 选 cp.async，launch_combine 选窄归并。 |

## t_351ca9e12a625f1f | dispatch.md | 319 chars | 1 declarations

dispatch.md:42 dispatch.md

| [tcg_d256_flat_tma_narrow](tcg_d256_flat_tma_narrow.yaml) | fused_d256_tma | D256 split，平坦扫描，TMA，窄归并 | 增加 group-max→平坦扫描→split→窄归并四次提交及 FP32 全局中间量；两计算 pass 均双 warpgroup，split 无融合 MIDPIPE，部分和经 shared 转置写回。 | launch_sequence 的 tcgen05、use_split、D256 分支，pl.scan_fused 为假、scan2p 为假；launch_tcg 选 TMA，launch_combine 选窄归并。 |

## t_384bc8ef5bf8c6c7 | dispatch.md | 369 chars | 1 declarations

dispatch.md:34 dispatch.md

| [tcg_d128_segmented_cp_xpose_narrow](tcg_d128_segmented_cp_xpose_narrow.yaml) | tcg_d128_segmented_tma_xpose_narrow | D128 split，分段扫描，cp.async，非 PSK 转置写回，窄归并 | group-max 与 split 搬运改为 cp.async，按 copy-group 提交/等待保护 payload 和 scale；保留扫描及部分和写回组织。 | launch_sequence 的 tcgen05、use_split、D128 分支，pl.scan_fused 为假、scan2p 为真、psk 为假；launch_tcg 选 cp.async，launch_combine 选窄归并。 |

## t_42321f0e66ad5508 | dispatch.md | 333 chars | 1 declarations

dispatch.md:44 dispatch.md

| [tcg_d256_segmented_cp_narrow](tcg_d256_segmented_cp_narrow.yaml) | tcg_d256_segmented_tma_narrow | D256 split，分段扫描，cp.async，窄归并 | group-max 与 split 搬运改为 cp.async，按 copy-group 提交/等待保护 payload 和 scale；保留扫描及部分和写回组织。 | launch_sequence 的 tcgen05、use_split、D256 分支，pl.scan_fused 为假、scan2p 为真；launch_tcg 选 cp.async，launch_combine 选窄归并。 |

## t_478bea52514eabaf | dispatch.md | 296 chars | 1 declarations

dispatch.md:47 dispatch.md

| [tcg_d256_segmented_tma_wide](tcg_d256_segmented_tma_wide.yaml) | tcg_d256_segmented_tma_narrow | D256 split，分段扫描，TMA，宽归并 | 归并改为四个 128 线程组分担 split；共享 numerator/warp 分母部分和，屏障后由第零组输出。 | launch_sequence 的 tcgen05、use_split、D256 分支，pl.scan_fused 为假、scan2p 为真；launch_tcg 选 TMA，launch_combine 选宽归并。 |

## t_4a4eb6b9bf7e4a66 | dispatch.md | 312 chars | 1 declarations

dispatch.md:20 dispatch.md

| [tcg_d128_flat_tma_direct_narrow](tcg_d128_flat_tma_direct_narrow.yaml) | root | D128 split，平坦扫描，TMA，PSK 直接写回，窄归并 | 增加 group-max→平坦扫描→split→窄归并四次提交及 FP32 全局中间量；QK 为 128 列，分子驻寄存器，PSK 行直接写部分和。 | launch_sequence 的 tcgen05、use_split、D128 分支，pl.scan_fused 为假、scan2p 为假、psk 为真；launch_tcg 选 TMA，launch_combine 选窄归并。 |

## t_4c251980efd70ed0 | dispatch.md | 333 chars | 1 declarations

dispatch.md:30 dispatch.md

| [tcg_d128_fscan_tma_xpose_narrow](tcg_d128_fscan_tma_xpose_narrow.yaml) | tcg_d128_flat_tma_xpose_narrow | D128 split，融合扫描，TMA，非 PSK 转置写回，窄归并 | 改为 group-max→局部扫描融合 split→加权归并三次提交；保留各 split 的 part_m，末 split 发布 mtail，归并重缩放。 | launch_sequence 的 tcgen05、use_split、D128 分支，pl.scan_fused 为真、psk 为假；launch_tcg 选 TMA，launch_combine 选窄归并。 |

## t_4c6c598c04a9cde0 | dispatch.md | 318 chars | 1 declarations

dispatch.md:21 dispatch.md

| [tcg_d128_flat_tma_direct_wide](tcg_d128_flat_tma_direct_wide.yaml) | tcg_d128_flat_tma_direct_narrow | D128 split，平坦扫描，TMA，PSK 直接写回，宽归并 | 归并改为四个 128 线程组分担 split；共享 numerator/warp 分母部分和，屏障后由第零组输出。 | launch_sequence 的 tcgen05、use_split、D128 分支，pl.scan_fused 为假、scan2p 为假、psk 为真；launch_tcg 选 TMA，launch_combine 选宽归并。 |

## t_4fcd303decf40340 | dispatch.md | 317 chars | 1 declarations

dispatch.md:27 dispatch.md

| [tcg_d128_fscan_cp_xpose_wide](tcg_d128_fscan_cp_xpose_wide.yaml) | tcg_d128_fscan_cp_xpose_narrow | D128 split，融合扫描，cp.async，非 PSK 转置写回，宽归并 | 归并改为四个 128 线程组分担 split；共享 numerator/warp 分母部分和，屏障后由第零组输出。 | launch_sequence 的 tcgen05、use_split、D128 分支，pl.scan_fused 为真、psk 为假；launch_tcg 选 cp.async，launch_combine 选宽归并。 |

## t_504b87676575a12e | dispatch.md | 185 chars | 1 declarations

dispatch.md:55 dispatch.md

固定构建关闭 BF16 部分和、D256 融合扫描、B1/B2 扫描、四 warpgroup 融合和额外 combine 通道切片。合法头几何使 scalar 分支不可达；use_tcv 在当前 D128/D256 的 use_tcg 定义下不可达。首次 tcg 可用性 probe 及缓存查询保留实际行为；图重放、捕获后发射与直接发射使用同一计算序列，不另派设计节点。

## t_552378d78a198acb | dispatch.md | 178 chars | 1 declarations

dispatch.md:6 dispatch.md

| [fused_d128_cp](fused_d128_cp.yaml) | root | D128，QK64 融合，TMA 描述符失败 | 改为 cp.async；K/V 合并搬运等待，逐组退休 PV，取消 TMA 专用的延迟退休组织。 | 同 QK64 host 分支，maps.valid 为假；launch_tcg 发射 TMA=false。 |

## t_58747f9c887fdabc | dispatch.md | 325 chars | 1 declarations

dispatch.md:17 dispatch.md

| [tcg_d128_flat_cp_direct_wide](tcg_d128_flat_cp_direct_wide.yaml) | tcg_d128_flat_cp_direct_narrow | D128 split，平坦扫描，cp.async，PSK 直接写回，宽归并 | 归并改为四个 128 线程组分担 split；共享 numerator/warp 分母部分和，屏障后由第零组输出。 | launch_sequence 的 tcgen05、use_split、D128 分支，pl.scan_fused 为假、scan2p 为假、psk 为真；launch_tcg 选 cp.async，launch_combine 选宽归并。 |

## t_58b24eb627b77915 | dispatch.md | 320 chars | 1 declarations

dispatch.md:38 dispatch.md

| [tcg_d128_segmented_tma_xpose_narrow](tcg_d128_segmented_tma_xpose_narrow.yaml) | tcg_d128_flat_tma_xpose_narrow | D128 split，分段扫描，TMA，非 PSK 转置写回，窄归并 | 扫描改为段最大值→分段重放，增加 segmax 物化及一次提交；其余 split 组织保持。 | launch_sequence 的 tcgen05、use_split、D128 分支，pl.scan_fused 为假、scan2p 为真、psk 为假；launch_tcg 选 TMA，launch_combine 选窄归并。 |

## t_64b9a9091aaca6a3 | dispatch.md | 303 chars | 1 declarations

dispatch.md:45 dispatch.md

| [tcg_d256_segmented_cp_wide](tcg_d256_segmented_cp_wide.yaml) | tcg_d256_segmented_cp_narrow | D256 split，分段扫描，cp.async，宽归并 | 归并改为四个 128 线程组分担 split；共享 numerator/warp 分母部分和，屏障后由第零组输出。 | launch_sequence 的 tcgen05、use_split、D256 分支，pl.scan_fused 为假、scan2p 为真；launch_tcg 选 cp.async，launch_combine 选宽归并。 |

## t_64dbcf016f610dae | dispatch.md | 217 chars | 1 declarations

dispatch.md:11 dispatch.md

| [tc_fused](tc_fused.yaml) | root | D128/D256，tcgen05 不可用时的融合计算 | 64 行、四 warp 的 mma.sync QK/PV；FP32 分子驻寄存器，直接输出；D/32 与 D/128 参数化通道循环。 | launch_sequence 的 pl.use_tc 分支且未选 tcgen05、use_split 为假；launch_tc 的 PASS_FUSED。 |

## t_6525b98915869d20 | dispatch.md | 318 chars | 1 declarations

dispatch.md:40 dispatch.md

| [tcg_d256_flat_cp_narrow](tcg_d256_flat_cp_narrow.yaml) | tcg_d256_flat_tma_narrow | D256 split，平坦扫描，cp.async，窄归并 | group-max 与 split 搬运改为 cp.async，按 copy-group 提交/等待保护 payload 和 scale；保留扫描及部分和写回组织。 | launch_sequence 的 tcgen05、use_split、D256 分支，pl.scan_fused 为假、scan2p 为假；launch_tcg 选 cp.async，launch_combine 选窄归并。 |

## t_69ab8fdcac49832b | dispatch.md | 333 chars | 1 declarations

dispatch.md:37 dispatch.md

| [tcg_d128_segmented_tma_direct_wide](tcg_d128_segmented_tma_direct_wide.yaml) | tcg_d128_segmented_tma_direct_narrow | D128 split，分段扫描，TMA，PSK 直接写回，宽归并 | 归并改为四个 128 线程组分担 split；共享 numerator/warp 分母部分和，屏障后由第零组输出。 | launch_sequence 的 tcgen05、use_split、D128 分支，pl.scan_fused 为假、scan2p 为真、psk 为真；launch_tcg 选 TMA，launch_combine 选宽归并。 |

## t_6e47de5960864e9f | dispatch.md | 268 chars | 1 declarations

dispatch.md:53 dispatch.md

同一节点内参数化 split 数、group-max 自己的 gm_nsplit、扫描 block 大小、尾块及 CTA 重排。D256 的 PSK 只改变行映射和标量有效区，两个情形均经 shared 转置写部分和；D128 的 PSK 同时改变写回生产者和共享存储接续，故单列 direct/xpose。D128 group-max 的长短局部 span 可在同次提交中并存，节点须保留各 CTA 实际选择的顺序或提前 QK 组织；空 split 及仅一个 warpgroup 有工作的边界不意味着取消已发射 CTA 或其必要同步。

## t_73791f01606e700f | dispatch.md | 242 chars | 1 declarations

dispatch.md:13 dispatch.md

| [tc_split_flat_wide](tc_split_flat_wide.yaml) | tc_split_flat_narrow | D128/D256，mma.sync split，平坦扫描，宽归并 | 归并改为四个 128 线程组分担 split；共享 numerator/warp 分母部分和，屏障后由第零组输出。 | launch_sequence 的 mma.sync、use_split 分支，scan2p 为假；launch_combine 选择宽归并。 |

## t_738b0549baf6af42 | dispatch.md | 262 chars | 1 declarations

dispatch.md:61 dispatch.md

mma.sync split 存在源码既有的 group-max 覆盖缺口：launch_sequence 使用 qtiles128 个查询块发射 group-max，而非 TCG 的 fmha_tc_kernel 按 qph=64/(H/KH) 定位查询块。当 Sq>qtiles128*qph 时，后段查询的 bm 未由该 pass 写入，后续扫描仍覆盖全部 rows_total；输出可能依赖原 workspace 内容。四个 tc_split 节点均须保留这一事实，不补造扩大 grid、初始化或 fallback。

## t_7875020cff23f0cd | dispatch.md | 6 chars | 1 declarations

dispatch.md:1 dispatch.md

# 路径索引

## t_7a000e97397c5120 | dispatch.md | 354 chars | 1 declarations

dispatch.md:18 dispatch.md

| [tcg_d128_flat_cp_xpose_narrow](tcg_d128_flat_cp_xpose_narrow.yaml) | tcg_d128_flat_tma_xpose_narrow | D128 split，平坦扫描，cp.async，非 PSK 转置写回，窄归并 | group-max 与 split 搬运改为 cp.async，按 copy-group 提交/等待保护 payload 和 scale；保留扫描及部分和写回组织。 | launch_sequence 的 tcgen05、use_split、D128 分支，pl.scan_fused 为假、scan2p 为假、psk 为假；launch_tcg 选 cp.async，launch_combine 选窄归并。 |

## t_7b5081f0e0ccbc31 | dispatch.md | 321 chars | 1 declarations

dispatch.md:36 dispatch.md

| [tcg_d128_segmented_tma_direct_narrow](tcg_d128_segmented_tma_direct_narrow.yaml) | tcg_d128_flat_tma_direct_narrow | D128 split，分段扫描，TMA，PSK 直接写回，窄归并 | 扫描改为段最大值→分段重放，增加 segmax 物化及一次提交；其余 split 组织保持。 | launch_sequence 的 tcgen05、use_split、D128 分支，pl.scan_fused 为假、scan2p 为真、psk 为真；launch_tcg 选 TMA，launch_combine 选窄归并。 |

## t_83d56d77829c7fb6 | dispatch.md | 67 chars | 1 declarations

dispatch.md:49 dispatch.md

直接父是完整 Design 的复制起点；每个节点的选择条件在共享 Contract 全域独立求值，精确条件与版本以各 YAML 为准。

## t_8a124a71c0efb5c6 | dispatch.md | 318 chars | 1 declarations

dispatch.md:25 dispatch.md

| [tcg_d128_fscan_cp_direct_wide](tcg_d128_fscan_cp_direct_wide.yaml) | tcg_d128_fscan_cp_direct_narrow | D128 split，融合扫描，cp.async，PSK 直接写回，宽归并 | 归并改为四个 128 线程组分担 split；共享 numerator/warp 分母部分和，屏障后由第零组输出。 | launch_sequence 的 tcgen05、use_split、D128 分支，pl.scan_fused 为真、psk 为真；launch_tcg 选 cp.async，launch_combine 选宽归并。 |

## t_8cef4cce6d6e96f8 | dispatch.md | 320 chars | 1 declarations

dispatch.md:22 dispatch.md

| [tcg_d128_flat_tma_xpose_narrow](tcg_d128_flat_tma_xpose_narrow.yaml) | tcg_d128_flat_tma_direct_narrow | D128 split，平坦扫描，TMA，非 PSK 转置写回，窄归并 | PSK 关闭；部分分子先写复用的 shared 区，经 CTA 屏障由全体线程转置写出，替代每行线程直接写回。 | launch_sequence 的 tcgen05、use_split、D128 分支，pl.scan_fused 为假、scan2p 为假、psk 为假；launch_tcg 选 TMA，launch_combine 选窄归并。 |

## t_92d34c314d0334f1 | dispatch.md | 257 chars | 1 declarations

dispatch.md:15 dispatch.md

| [tc_split_segmented_wide](tc_split_segmented_wide.yaml) | tc_split_segmented_narrow | D128/D256，mma.sync split，分段扫描，宽归并 | 归并改为四个 128 线程组分担 split；共享 numerator/warp 分母部分和，屏障后由第零组输出。 | launch_sequence 的 mma.sync、use_split 分支，scan2p 为真；launch_combine 选择宽归并。 |

## t_945cf0c6f51ab850 | dispatch.md | 370 chars | 1 declarations

dispatch.md:32 dispatch.md

| [tcg_d128_segmented_cp_direct_narrow](tcg_d128_segmented_cp_direct_narrow.yaml) | tcg_d128_segmented_tma_direct_narrow | D128 split，分段扫描，cp.async，PSK 直接写回，窄归并 | group-max 与 split 搬运改为 cp.async，按 copy-group 提交/等待保护 payload 和 scale；保留扫描及部分和写回组织。 | launch_sequence 的 tcgen05、use_split、D128 分支，pl.scan_fused 为假、scan2p 为真、psk 为真；launch_tcg 选 cp.async，launch_combine 选窄归并。 |

## t_98f3ef8a80cab4a6 | dispatch.md | 253 chars | 1 declarations

dispatch.md:12 dispatch.md

| [tc_split_flat_narrow](tc_split_flat_narrow.yaml) | tc_fused | D128/D256，mma.sync split，平坦扫描，窄归并 | 增加 group-max→平坦扫描→split→窄归并四次提交及 FP32 全局中间量；保留源码 group-max 的 qtiles128 发射缺口。 | launch_sequence 的 mma.sync、use_split 分支，scan2p 为假；launch_combine 选择窄归并。 |

## t_996724ae36524d05 | dispatch.md | 340 chars | 1 declarations

dispatch.md:33 dispatch.md

| [tcg_d128_segmented_cp_direct_wide](tcg_d128_segmented_cp_direct_wide.yaml) | tcg_d128_segmented_cp_direct_narrow | D128 split，分段扫描，cp.async，PSK 直接写回，宽归并 | 归并改为四个 128 线程组分担 split；共享 numerator/warp 分母部分和，屏障后由第零组输出。 | launch_sequence 的 tcgen05、use_split、D128 分支，pl.scan_fused 为假、scan2p 为真、psk 为真；launch_tcg 选 cp.async，launch_combine 选宽归并。 |

## t_9b7ed42cd53a5b33 | dispatch.md | 348 chars | 1 declarations

dispatch.md:24 dispatch.md

| [tcg_d128_fscan_cp_direct_narrow](tcg_d128_fscan_cp_direct_narrow.yaml) | tcg_d128_fscan_tma_direct_narrow | D128 split，融合扫描，cp.async，PSK 直接写回，窄归并 | group-max 与 split 搬运改为 cp.async，按 copy-group 提交/等待保护 payload 和 scale；保留扫描及部分和写回组织。 | launch_sequence 的 tcgen05、use_split、D128 分支，pl.scan_fused 为真、psk 为真；launch_tcg 选 cp.async，launch_combine 选窄归并。 |

## t_a5975bb8cda664a6 | dispatch.md | 310 chars | 1 declarations

dispatch.md:31 dispatch.md

| [tcg_d128_fscan_tma_xpose_wide](tcg_d128_fscan_tma_xpose_wide.yaml) | tcg_d128_fscan_tma_xpose_narrow | D128 split，融合扫描，TMA，非 PSK 转置写回，宽归并 | 归并改为四个 128 线程组分担 split；共享 numerator/warp 分母部分和，屏障后由第零组输出。 | launch_sequence 的 tcgen05、use_split、D128 分支，pl.scan_fused 为真、psk 为假；launch_tcg 选 TMA，launch_combine 选宽归并。 |

## t_aae327c60eeed961 | dispatch.md | 203 chars | 1 declarations

dispatch.md:7 dispatch.md

| [fused_d128_group](fused_d128_group.yaml) | fused_d128_key | D128，查询 CTA 较少、可见组较长的融合计算 | 双 warpgroup 改为交替处理整组 key；各自 K/V 环与组间最大值/分母状态交接。 | launch_sequence 的非 split、gsplit 分支；useful_key_span 触发整组交替组织。 |

## t_ac1d61c4c9ba149c | dispatch.md | 153 chars | 1 declarations

dispatch.md:63 dispatch.md

root 的验收来源记录已保留 BF16-partial A/B margin 实验的 O 误差比及 candidate 相对基准，适用范围限于该实验。它与最终外部 O/LSE 检查路由的绑定、比较精度及 NaN/Inf 策略仍未提供，LSE 的外部公式和相对基准也未补齐；该记录不授予设计或全域数值保证。

## t_bed4696225b19085 | dispatch.md | 284 chars | 1 declarations

dispatch.md:46 dispatch.md

| [tcg_d256_segmented_tma_narrow](tcg_d256_segmented_tma_narrow.yaml) | tcg_d256_flat_tma_narrow | D256 split，分段扫描，TMA，窄归并 | 扫描改为段最大值→分段重放，增加 segmax 物化及一次提交；其余 split 组织保持。 | launch_sequence 的 tcgen05、use_split、D256 分支，pl.scan_fused 为假、scan2p 为真；launch_tcg 选 TMA，launch_combine 选窄归并。 |

## t_d66a515184cc018b | dispatch.md | 205 chars | 1 declarations

dispatch.md:10 dispatch.md

| [fused_d256_tma](fused_d256_tma.yaml) | root | D256，TMA 可用的融合计算 | 双 warpgroup 分担 key/输出通道；512 列 TMEM、提前 scale 搬运及分离 K/V barrier 的 MIDPIPE。 | launch_sequence 的 tcgen05 非 split、D256 分支；launch_tcg 选择 TMA。 |

## t_d8f4ee7937b75399 | dispatch.md | 330 chars | 1 declarations

dispatch.md:51 dispatch.md

选择入口均从 candidate/solution.py::Model.forward 到 candidate/fmha_mxfp8_qkv.cu 的 plan_geometry、launch_sequence、launch_tcg 和 launch_combine。pl 为实际 HostPlan；use_split 包含 workspace 容量门槛，maps.valid 为实际 K/V 描述符结果。scan2p 使用 T、rows_total 和有效 window，scan_fused 还使用 useful_key_span；psk 使用 Sq、H/KH 与 qtiles128；宽归并分支使用 rows_total、nsplit 与实际 SM 数。

## t_e0d8846ec2fdd374 | dispatch.md | 347 chars | 1 declarations

dispatch.md:26 dispatch.md

| [tcg_d128_fscan_cp_xpose_narrow](tcg_d128_fscan_cp_xpose_narrow.yaml) | tcg_d128_fscan_tma_xpose_narrow | D128 split，融合扫描，cp.async，非 PSK 转置写回，窄归并 | group-max 与 split 搬运改为 cp.async，按 copy-group 提交/等待保护 payload 和 scale；保留扫描及部分和写回组织。 | launch_sequence 的 tcgen05、use_split、D128 分支，pl.scan_fused 为真、psk 为假；launch_tcg 选 cp.async，launch_combine 选窄归并。 |

## t_e44b2bf8d9e5703c | dispatch.md | 317 chars | 1 declarations

dispatch.md:23 dispatch.md

| [tcg_d128_flat_tma_xpose_wide](tcg_d128_flat_tma_xpose_wide.yaml) | tcg_d128_flat_tma_xpose_narrow | D128 split，平坦扫描，TMA，非 PSK 转置写回，宽归并 | 归并改为四个 128 线程组分担 split；共享 numerator/warp 分母部分和，屏障后由第零组输出。 | launch_sequence 的 tcgen05、use_split、D128 分支，pl.scan_fused 为假、scan2p 为假、psk 为假；launch_tcg 选 TMA，launch_combine 选宽归并。 |

## t_ef5f70756155bfa6 | dispatch.md | 203 chars | 1 declarations

dispatch.md:9 dispatch.md

| [fused_d256_cp](fused_d256_cp.yaml) | fused_d256_tma | D256，TMA 描述符失败的融合计算 | 改为 cp.async，保留 SF_AHEAD；使用顺序组主体，不采用 TMA MIDPIPE 的中段提交与延迟 PV 退休。 | 同 D256 融合 host 分支，maps.valid 为假；launch_tcg 发射 TMA=false。 |

## t_ef72351d0c0fc429 | dispatch.md | 288 chars | 1 declarations

dispatch.md:41 dispatch.md

| [tcg_d256_flat_cp_wide](tcg_d256_flat_cp_wide.yaml) | tcg_d256_flat_cp_narrow | D256 split，平坦扫描，cp.async，宽归并 | 归并改为四个 128 线程组分担 split；共享 numerator/warp 分母部分和，屏障后由第零组输出。 | launch_sequence 的 tcgen05、use_split、D256 分支，pl.scan_fused 为假、scan2p 为假；launch_tcg 选 cp.async，launch_combine 选宽归并。 |

## t_fb1eac81f471246d | dispatch.md | 186 chars | 1 declarations

dispatch.md:5 dispatch.md

| [root](root.yaml) | 无 | D128，查询 CTA 较多，TMA 可用的融合计算 | 单 warpgroup QK64；完整组 scores 保留于寄存器，分子驻 TMEM，延迟 PV 退休，直接写 O/可选 LSE。 | launch_sequence 的 tcgen05 非 split、QK64 分支；launch_tcg 选择 TMA。 |

## t_02f93e7b0c8896c6 | dispatch.when | 1028 chars | 1 declarations

tc_split_flat_wide:9 meta_data.dispatch.when

Let active mean window_left is not None and window_left>=0, and let wl be window_left when active, otherwise 0. Let HostPlan be the actual Plan returned by fmha_mxfp8::plan_geometry(B,H,KH,Sq,Skv,D,is_causal,wl,active), from source@6e4ca91bd91d998cac99e8dadd83838e153c679d:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu::plan_geometry. It uses the prepared binary's cached tcg_available probe and device queries, including their source fallbacks. Let SmCount be device_sm_count() from the same source, R=B*H*Sq, T=ceil_div(Skv,KeyGroup), Ns=HostPlan.nsplit. Let split mean HostPlan.use_tc and Ns>1 and W>0 and W>=HostPlan.workspace. Let scan2p mean T>=128 and R<=512 and not active. Select exactly when HostPlan.use_tc and not HostPlan.use_tcg and not HostPlan.use_tcv and split and not scan2p and ((R<2*SmCount and Ns>=8) or Ns>128). The fixed build uses PVMODE=0 and the PASS_GROUP_MAX -> flat scan -> PASS_SPLIT -> FP32 wide combine chain, with four groups and no D slicing.

## t_1357e5d57f821de2 | dispatch.when | 1196 chars | 1 declarations

tcg_d128_flat_cp_direct_wide:9 meta_data.dispatch.when

Let active mean window_left is not None and window_left>=0, and wl be window_left when active, otherwise 0. Let HostPlan be the actual Plan returned by fmha_mxfp8::plan_geometry(B,H,KH,Sq,Skv,D,is_causal,wl,active) in source@6e4ca91bd91d998cac99e8dadd83838e153c679d:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu::fmha_mxfp8::plan_geometry. Let SmCount be that source's actual device_sm_count() result, including its fallback. Let MapsValid be the actual short-circuit conjunction of its K/V encode_tile_map results, including descriptor-cache hits. Let split mean HostPlan.use_tc and HostPlan.nsplit>1 and W>0 and W>=HostPlan.workspace. Let scan2p mean HostPlan.T>=128 and HostPlan.rows_total<=512 and not active. Let psk mean 2*Sq*(H/KH)<=128*HostPlan.qtiles128. Let wide mean (HostPlan.rows_total<2*SmCount and HostPlan.nsplit>=8) or HostPlan.nsplit>128. Select exactly when HostPlan.use_tcg and split and D=128 and not MapsValid and not HostPlan.scan_fused and not scan2p and psk and wide. The fixed defaults select QK128, one warpgroup per matrix CTA, FP32 partials, PSK direct stores, cp.async staging, separate flat scan and wide combine.

## t_15cfdcb1e67eea31 | dispatch.when | 1187 chars | 1 declarations

tcg_d128_flat_tma_direct_wide:10 meta_data.dispatch.when

Let active mean window_left is not None and window_left>=0, and wl be window_left when active, otherwise 0. Let HostPlan be the actual Plan returned by fmha_mxfp8::plan_geometry(B,H,KH,Sq,Skv,D,is_causal,wl,active) in source@6e4ca91bd91d998cac99e8dadd83838e153c679d:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu::fmha_mxfp8::plan_geometry. Let SmCount be that source's actual device_sm_count() result, including its fallback. Let MapsValid be the actual short-circuit conjunction of its K/V encode_tile_map results, including descriptor-cache hits. Let split mean HostPlan.use_tc and HostPlan.nsplit>1 and W>0 and W>=HostPlan.workspace. Let scan2p mean HostPlan.T>=128 and HostPlan.rows_total<=512 and not active. Let psk mean 2*Sq*(H/KH)<=128*HostPlan.qtiles128. Let wide mean (HostPlan.rows_total<2*SmCount and HostPlan.nsplit>=8) or HostPlan.nsplit>128. Select exactly when HostPlan.use_tcg and split and D=128 and MapsValid and not HostPlan.scan_fused and not scan2p and psk and wide. The fixed defaults select QK128, one warpgroup per matrix CTA, FP32 partials, PSK direct stores, TMA staging, separate flat scan and wide combine.

## t_1897975a41fbf6c3 | dispatch.when | 1015 chars | 1 declarations

tc_split_segmented_narrow:10 meta_data.dispatch.when

Let active mean window_left is not None and window_left>=0, and let wl be window_left when active, otherwise 0. Let HostPlan be the actual Plan returned by fmha_mxfp8::plan_geometry(B,H,KH,Sq,Skv,D,is_causal,wl,active), from source@6e4ca91bd91d998cac99e8dadd83838e153c679d:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu::plan_geometry. It uses the prepared binary's cached tcg_available probe and device queries, including their source fallbacks. Let SmCount be device_sm_count() from the same source, R=B*H*Sq, T=ceil_div(Skv,KeyGroup), Ns=HostPlan.nsplit. Let split mean HostPlan.use_tc and Ns>1 and W>0 and W>=HostPlan.workspace. Let scan2p mean T>=128 and R<=512 and not active. Select exactly when HostPlan.use_tc and not HostPlan.use_tcg and not HostPlan.use_tcv and split and scan2p and Ns<=128 and (R>=2*SmCount or Ns<8). The fixed build uses PVMODE=0 and the PASS_GROUP_MAX -> segment maxima -> segmented replay -> PASS_SPLIT -> FP32 narrow combine chain.

## t_18b9fb04fe0e4663 | dispatch.when | 1131 chars | 1 declarations

tcg_d256_flat_tma_narrow:9 meta_data.dispatch.when

Let active mean window_left is not None and window_left>=0, and wl be window_left when active, otherwise 0. HostPlan is the actual result of fmha_mxfp8::plan_geometry(B,H,KH,Sq,Skv,D,is_causal,wl,active), from source@6e4ca91bd91d998cac99e8dadd83838e153c679d:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu::fmha_mxfp8::plan_geometry. SmCount is device_sm_count() from that source, including its fallback. Let split mean HostPlan.use_tc and HostPlan.nsplit>1 and W>0 and W>=HostPlan.workspace. MapsValid is the actual short-circuit conjunction of K and V encode_tile_map results in fmha_mxfp8_forward, including cache hits; these probe/encoding results are runtime facts, not guarantees. Let scan2p=(HostPlan.T>=128 and HostPlan.rows_total<=512 and not active). Select exactly when HostPlan.use_tcg and split and D=256 and MapsValid and not HostPlan.scan_fused and not scan2p and not ((HostPlan.rows_total<2*SmCount and HostPlan.nsplit>=8) or HostPlan.nsplit>128). Fixed defaults select FP32 partials, QK64=false, GSPLIT=false, FSCAN=false and two warpgroups in both compute passes.

## t_238d5b2bad3383ce | dispatch.when | 1116 chars | 1 declarations

tcg_d128_fscan_cp_xpose_wide:10 meta_data.dispatch.when

Let active mean window_left is not None and window_left >= 0, and wl be window_left when active, otherwise 0. Let HostPlan be the actual cached Plan returned by fmha_mxfp8::plan_geometry(B,H,KH,Sq,Skv,D,is_causal,wl,active) in source@6e4ca91bd91d998cac99e8dadd83838e153c679d:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu::fmha_mxfp8::plan_geometry. Let SmCount be that source's actual device_sm_count() result, including its query fallback. Let MapsValid be the actual short-circuit conjunction of K and V encode_tile_map results in fmha_mxfp8_forward, including cache hits. Let split mean HostPlan.use_tc and HostPlan.nsplit>1 and W>0 and W>=HostPlan.workspace. Let psk mean 2*Sq*(H/KH)<=128*HostPlan.qtiles128. Let wide mean (HostPlan.rows_total<2*SmCount and HostPlan.nsplit>=8) or HostPlan.nsplit>128. Select exactly when HostPlan.use_tcg and split and D=128 and not MapsValid and HostPlan.scan_fused and not psk and wide. The fixed defaults select FP32 partials, QK128, non-PSK rows, cp.async and wide weighted combine. Both matrix kernels have one warpgroup.

## t_29db629734b82a4b | dispatch.when | 1194 chars | 1 declarations

tcg_d128_segmented_tma_direct_narrow:10 meta_data.dispatch.when

Let active mean window_left is not None and window_left>=0, and wl be window_left when active, otherwise 0. Let HostPlan be the actual Plan returned by fmha_mxfp8::plan_geometry(B,H,KH,Sq,Skv,D,is_causal,wl,active) in source@6e4ca91bd91d998cac99e8dadd83838e153c679d:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu::fmha_mxfp8::plan_geometry. Let SmCount be that source's actual device_sm_count() result, including its fallback. Let MapsValid be the actual short-circuit conjunction of its K/V encode_tile_map results, including descriptor-cache hits. Let split mean HostPlan.use_tc and HostPlan.nsplit>1 and W>0 and W>=HostPlan.workspace. Let scan2p mean HostPlan.T>=128 and HostPlan.rows_total<=512 and not active. Let psk mean 2*Sq*(H/KH)<=128*HostPlan.qtiles128. Let wide mean (HostPlan.rows_total<2*SmCount and HostPlan.nsplit>=8) or HostPlan.nsplit>128. Select exactly when HostPlan.use_tcg and split and D=128 and MapsValid and not HostPlan.scan_fused and scan2p and psk and not wide. The fixed defaults select QK128, one warpgroup per matrix CTA, FP32 partials, PSK direct stores, TMA staging, segment-max/replay scan and narrow combine.

## t_2a23f2b562852826 | dispatch.when | 1239 chars | 1 declarations

fused_d128_group:10 meta_data.dispatch.when

Let active mean window_left is not None and window_left >= 0, and let wl be window_left when active, otherwise 0. Let HostPlan be the actual Plan returned by fmha_mxfp8::plan_geometry(B,H,KH,Sq,Skv,D,is_causal,wl,active), from source@6e4ca91bd91d998cac99e8dadd83838e153c679d:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu::fmha_mxfp8::plan_geometry. Let SmCount be device_sm_count() from that source, including its query fallback. HostPlan includes the actual cached capability-probe results. Let CtaCount=B*KH*HostPlan.qtiles128 and split mean HostPlan.use_tc and HostPlan.nsplit>1 and W>0 and W>=HostPlan.workspace. Let qLast=(HostPlan.qtiles128-1)*HostPlan.qph128, loLast=max(0,qLast+Skv-Sq-wl) when active, otherwise 0, and Span=max(1,floor((Skv-1)/128)-floor(loLast/128)+1). Span is the actual useful_key_span result, evaluated on the last query tile; it need not bound the span of earlier windowed tiles. Select exactly when HostPlan.use_tcg and not split and D=128 and CtaCount<=SmCount and Span>=8. With the fixed build defaults this selects launch_tcg<128,PASS_FUSED,2,false,false,true> with TMA=false. Descriptor encoding still occurs, but its success does not select this path.

## t_350409118bfb630a | dispatch.when | 1128 chars | 1 declarations

tcg_d128_fscan_tma_xpose_narrow:10 meta_data.dispatch.when

Let active mean window_left is not None and window_left>=0, and wl be window_left when active, otherwise 0. Let HostPlan be the actual Plan returned by fmha_mxfp8::plan_geometry(B,H,KH,Sq,Skv,D,is_causal,wl,active) in source@6e4ca91bd91d998cac99e8dadd83838e153c679d:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu::fmha_mxfp8::plan_geometry. Let SmCount be that source's actual device_sm_count() result, including its fallback. Let MapsValid be the actual short-circuit conjunction of its K/V encode_tile_map results, including descriptor-cache hits. Let split mean HostPlan.use_tc and HostPlan.nsplit>1 and W>0 and W>=HostPlan.workspace. Let psk mean 2*Sq*(H/KH)<=128*HostPlan.qtiles128. Let wide mean (HostPlan.rows_total<2*SmCount and HostPlan.nsplit>=8) or HostPlan.nsplit>128. Select exactly when HostPlan.use_tcg and split and D=128 and MapsValid and HostPlan.scan_fused and not psk and not wide. The fixed defaults select QK128, one warpgroup per matrix CTA, FP32 partials, non-PSK rows, shared transpose stores, TMA staging, split-local scan and weighted narrow combine.

## t_3b2feba32f50994e | dispatch.when | 994 chars | 1 declarations

tc_split_flat_narrow:10 meta_data.dispatch.when

Let active mean window_left is not None and window_left>=0, and let wl be window_left when active, otherwise 0. Let HostPlan be the actual Plan returned by fmha_mxfp8::plan_geometry(B,H,KH,Sq,Skv,D,is_causal,wl,active), from source@6e4ca91bd91d998cac99e8dadd83838e153c679d:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu::plan_geometry. It uses the prepared binary's cached tcg_available probe and device queries, including their source fallbacks. Let SmCount be device_sm_count() from the same source, R=B*H*Sq, T=ceil_div(Skv,KeyGroup), Ns=HostPlan.nsplit. Let split mean HostPlan.use_tc and Ns>1 and W>0 and W>=HostPlan.workspace. Let scan2p mean T>=128 and R<=512 and not active. Select exactly when HostPlan.use_tc and not HostPlan.use_tcg and not HostPlan.use_tcv and split and not scan2p and Ns<=128 and (R>=2*SmCount or Ns<8). The fixed build uses PVMODE=0 and the PASS_GROUP_MAX -> flat scan -> PASS_SPLIT -> FP32 narrow combine chain.

## t_3b7a7fb7e303d407 | dispatch.when | 1077 chars | 1 declarations

fused_d256_cp:10 meta_data.dispatch.when

Let active mean window_left is not None and window_left >= 0, and let wl be window_left when active, otherwise 0. Let HostPlan be the actual Plan returned by fmha_mxfp8::plan_geometry(B,H,KH,Sq,Skv,D,is_causal,wl,active), from source@6e4ca91bd91d998cac99e8dadd83838e153c679d:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu::fmha_mxfp8::plan_geometry. Let SmCount be device_sm_count() from that source, including its query fallback, and CtaCount=B*KH*HostPlan.qtiles128. Let split mean HostPlan.use_tc and HostPlan.nsplit>1 and W>0 and W>=HostPlan.workspace, matching the actual forward workspace guard. Let MapsValid be the actual short-circuit conjunction of the K and V encode_tile_map results in fmha_mxfp8_forward, including descriptor-cache hits. Probe and encoding success are runtime facts, not guarantees. Select exactly when HostPlan.use_tcg and not split and D=256 and not MapsValid. The fixed defaults select launch_tcg<256,PASS_FUSED,2> with TMA=false, QK64=false, PSK=false, SF_AHEAD=true and kernel MIDPIPE=false.

## t_414db86a376fb998 | dispatch.when | 1208 chars | 1 declarations

tcg_d128_segmented_cp_direct_wide:9 meta_data.dispatch.when

Let active mean window_left is not None and window_left>=0, and wl be window_left when active, otherwise 0. Let HostPlan be the actual Plan returned by fmha_mxfp8::plan_geometry(B,H,KH,Sq,Skv,D,is_causal,wl,active) in source@6e4ca91bd91d998cac99e8dadd83838e153c679d:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu::fmha_mxfp8::plan_geometry. Let SmCount be that source's actual device_sm_count() result, including its fallback. Let MapsValid be the actual short-circuit conjunction of its K/V encode_tile_map results, including descriptor-cache hits. Let split mean HostPlan.use_tc and HostPlan.nsplit>1 and W>0 and W>=HostPlan.workspace. Let scan2p mean HostPlan.T>=128 and HostPlan.rows_total<=512 and not active. Let psk mean 2*Sq*(H/KH)<=128*HostPlan.qtiles128. Let wide mean (HostPlan.rows_total<2*SmCount and HostPlan.nsplit>=8) or HostPlan.nsplit>128. Select exactly when HostPlan.use_tcg and split and D=128 and not MapsValid and not HostPlan.scan_fused and scan2p and psk and wide. The fixed defaults select QK128, one warpgroup per matrix CTA, FP32 partials, PSK direct stores, cp.async staging, segment-max/replay scan and four-group wide combine.

## t_44ea791552f6e879 | dispatch.when | 1330 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:9 meta_data.dispatch.when

Let active mean window_left is not None and window_left >= 0, and wl be window_left when active, otherwise 0. Let HostPlan be the actual cached Plan returned by fmha_mxfp8::plan_geometry(B,H,KH,Sq,Skv,D,is_causal,wl,active) in source@6e4ca91bd91d998cac99e8dadd83838e153c679d:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu::fmha_mxfp8::plan_geometry. Let SmCount be that source's actual device_sm_count() result, including its query fallback. Let MapsValid be the actual short-circuit conjunction of K and V encode_tile_map results in fmha_mxfp8_forward, including cache hits. Let split mean HostPlan.use_tc and HostPlan.nsplit>1 and W>0 and W>=HostPlan.workspace. Let scan2p mean HostPlan.T>=128 and HostPlan.rows_total<=512 and not active. Let psk mean 2*Sq*(H/KH)<=128*HostPlan.qtiles128. Let wide mean (HostPlan.rows_total<2*SmCount and HostPlan.nsplit>=8) or HostPlan.nsplit>128. Select exactly when HostPlan.use_tcg and split and D=128 and not MapsValid and not HostPlan.scan_fused and scan2p and not psk and wide. The fixed defaults select FP32 partials, QK128, non-PSK rows, cp.async, segment-max then row replay, and wide combine. Both matrix kernels have one warpgroup; split stages partial numerators in reused shared operand storage for a cooperative transposed write.

## t_4590e4631e084faa | dispatch.when | 1271 chars | 1 declarations

tcg_d128_flat_tma_xpose_wide:9 meta_data.dispatch.when

Let active mean window_left is not None and window_left>=0, and wl be window_left when active, otherwise 0. Let HostPlan be the actual Plan returned by fmha_mxfp8::plan_geometry(B,H,KH,Sq,Skv,D,is_causal,wl,active) in source@6e4ca91bd91d998cac99e8dadd83838e153c679d:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu::fmha_mxfp8::plan_geometry. Let SmCount be that source's actual device_sm_count() result, including its fallback. Let MapsValid be the actual short-circuit conjunction of its K/V encode_tile_map results, including descriptor-cache hits. Let split mean HostPlan.use_tc and HostPlan.nsplit>1 and W>0 and W>=HostPlan.workspace. Let scan2p mean HostPlan.T>=128 and HostPlan.rows_total<=512 and not active. Let psk mean 2*Sq*(H/KH)<=128*HostPlan.qtiles128. Let wide mean (HostPlan.rows_total<2*SmCount and HostPlan.nsplit>=8) or HostPlan.nsplit>128. Select exactly when HostPlan.use_tcg and split and D=128 and MapsValid and not HostPlan.scan_fused and not scan2p and not psk and wide. The fixed defaults select QK128, one warpgroup per matrix CTA, FP32 partials, non-PSK rows, shared transpose stores, TMA staging and separate flat scan. Wide combine uses four 128-thread groups per row without channel slicing.

## t_57b8207304f76595 | dispatch.when | 1115 chars | 1 declarations

tcg_d128_fscan_cp_direct_narrow:10 meta_data.dispatch.when

Let active mean window_left is not None and window_left>=0, and wl be window_left when active, otherwise 0. Let HostPlan be the actual Plan returned by fmha_mxfp8::plan_geometry(B,H,KH,Sq,Skv,D,is_causal,wl,active) in source@6e4ca91bd91d998cac99e8dadd83838e153c679d:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu::fmha_mxfp8::plan_geometry. Let SmCount be that source's actual device_sm_count() result, including its fallback. Let MapsValid be the actual short-circuit conjunction of its K/V encode_tile_map results, including descriptor-cache hits. Let split mean HostPlan.use_tc and HostPlan.nsplit>1 and W>0 and W>=HostPlan.workspace. Let psk mean 2*Sq*(H/KH)<=128*HostPlan.qtiles128. Let wide mean (HostPlan.rows_total<2*SmCount and HostPlan.nsplit>=8) or HostPlan.nsplit>128. Select exactly when HostPlan.use_tcg and split and D=128 and not MapsValid and HostPlan.scan_fused and psk and not wide. The fixed defaults select QK128, one warpgroup per matrix CTA, FP32 partials, PSK direct stores, cp.async staging, locally fused scan and weighted narrow combine.

## t_61635d4b2e043734 | dispatch.when | 1049 chars | 1 declarations

tc_split_segmented_wide:9 meta_data.dispatch.when

Let active mean window_left is not None and window_left>=0, and let wl be window_left when active, otherwise 0. Let HostPlan be the actual Plan returned by fmha_mxfp8::plan_geometry(B,H,KH,Sq,Skv,D,is_causal,wl,active), from source@6e4ca91bd91d998cac99e8dadd83838e153c679d:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu::plan_geometry. It uses the prepared binary's cached tcg_available probe and device queries, including their source fallbacks. Let SmCount be device_sm_count() from the same source, R=B*H*Sq, T=ceil_div(Skv,KeyGroup), Ns=HostPlan.nsplit. Let split mean HostPlan.use_tc and Ns>1 and W>0 and W>=HostPlan.workspace. Let scan2p mean T>=128 and R<=512 and not active. Select exactly when HostPlan.use_tc and not HostPlan.use_tcg and not HostPlan.use_tcv and split and scan2p and ((R<2*SmCount and Ns>=8) or Ns>128). The fixed build uses PVMODE=0 and PASS_GROUP_MAX -> segment maxima -> segmented replay -> PASS_SPLIT -> FP32 wide combine, without combine-side weighting or D slicing.

## t_64abd62a2de951b2 | dispatch.when | 1064 chars | 1 declarations

fused_d256_tma:10 meta_data.dispatch.when

Let active mean window_left is not None and window_left >= 0, and let wl be window_left when active, otherwise 0. Let HostPlan be the actual Plan returned by fmha_mxfp8::plan_geometry(B,H,KH,Sq,Skv,D,is_causal,wl,active), from source@6e4ca91bd91d998cac99e8dadd83838e153c679d:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu::fmha_mxfp8::plan_geometry. Let SmCount be device_sm_count() from that source, including its query fallback, and CtaCount=B*KH*HostPlan.qtiles128. Let split mean HostPlan.use_tc and HostPlan.nsplit>1 and W>0 and W>=HostPlan.workspace, matching the actual forward workspace guard. Let MapsValid be the actual short-circuit conjunction of the K and V encode_tile_map results in fmha_mxfp8_forward, including descriptor-cache hits. Probe and encoding success are runtime facts, not guarantees. Select exactly when HostPlan.use_tcg and not split and D=256 and MapsValid. The fixed defaults select launch_tcg<256,PASS_FUSED,2> with TMA=true, QK64=false, PSK=false, SF_AHEAD=true and MIDPIPE=true.

## t_64abf5bf071f0e28 | dispatch.when | 1218 chars | 1 declarations

fused_d128_cp:10 meta_data.dispatch.when

Let active mean window_left is not None and window_left >= 0, and let wl be window_left when active, otherwise 0. Let HostPlan be the actual Plan returned by fmha_mxfp8::plan_geometry(B,H,KH,Sq,Skv,D,is_causal,wl,active), from source@6e4ca91bd91d998cac99e8dadd83838e153c679d:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu::fmha_mxfp8::plan_geometry. Let SmCount be device_sm_count() from the same source. These are actual cached probe and device-query results, including the source's query fallbacks. Let CtaCount=B*KH*HostPlan.qtiles128. Let split mean HostPlan.use_tc and HostPlan.nsplit>1 and W>0 and W>=HostPlan.workspace, matching the actual forward workspace guard. Let MapsValid be the actual short-circuit conjunction of the K and V encode_tile_map results in that source's fmha_mxfp8_forward, including descriptor-cache hits; it is not an assumed descriptor-success guarantee. Select exactly when HostPlan.use_tcg and not split and D=128 and CtaCount>SmCount and not MapsValid. With the fixed build defaults this selects launch_tcg<128,PASS_FUSED,1,true> with TMA=false: the single-warpgroup QK64 fused path with cp.async staging and per-group PV retirement.

## t_6c69fe64fd921ad7 | dispatch.when | 1223 chars | 1 declarations

tcg_d128_segmented_tma_xpose_narrow:9 meta_data.dispatch.when

Let active mean window_left is not None and window_left>=0, and wl be window_left when active, otherwise 0. Let HostPlan be the actual Plan returned by fmha_mxfp8::plan_geometry(B,H,KH,Sq,Skv,D,is_causal,wl,active) in source@6e4ca91bd91d998cac99e8dadd83838e153c679d:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu::fmha_mxfp8::plan_geometry. Let SmCount be that source's actual device_sm_count() result, including its fallback. Let MapsValid be the actual short-circuit conjunction of its K/V encode_tile_map results, including descriptor-cache hits. Let split mean HostPlan.use_tc and HostPlan.nsplit>1 and W>0 and W>=HostPlan.workspace. Let scan2p mean HostPlan.T>=128 and HostPlan.rows_total<=512 and not active. Let psk mean 2*Sq*(H/KH)<=128*HostPlan.qtiles128. Let wide mean (HostPlan.rows_total<2*SmCount and HostPlan.nsplit>=8) or HostPlan.nsplit>128. Select exactly when HostPlan.use_tcg and split and D=128 and MapsValid and not HostPlan.scan_fused and scan2p and not psk and not wide. The fixed defaults select QK128, one warpgroup per matrix CTA, FP32 partials, non-PSK rows, shared transpose stores, TMA staging, segment-max then row replay, and narrow combine.

## t_6ef94cea6d7321ce | dispatch.when | 1194 chars | 1 declarations

tcg_d256_flat_cp_wide:9 meta_data.dispatch.when

Let active mean window_left is not None and window_left>=0, and wl be window_left when active, otherwise 0. HostPlan is the actual result of fmha_mxfp8::plan_geometry(B,H,KH,Sq,Skv,D,is_causal,wl,active), from source@6e4ca91bd91d998cac99e8dadd83838e153c679d:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu::fmha_mxfp8::plan_geometry. SmCount is device_sm_count() from that source, including its fallback. Let split mean HostPlan.use_tc and HostPlan.nsplit>1 and W>0 and W>=HostPlan.workspace. MapsValid is the actual short-circuit conjunction of K and V encode_tile_map results in fmha_mxfp8_forward, including cache hits; these probe/encoding results are runtime facts, not guarantees. Let scan2p=(HostPlan.T>=128 and HostPlan.rows_total<=512 and not active). Select exactly when HostPlan.use_tcg and split and D=256 and not MapsValid and not HostPlan.scan_fused and not scan2p and ((HostPlan.rows_total<2*SmCount and HostPlan.nsplit>=8) or HostPlan.nsplit>128). Fixed defaults select TMA=false, FP32 partials, QK64=false, GSPLIT=false, FSCAN=false, two warpgroups in both compute passes, and a four-group wide combine without channel slicing.

## t_7136cab41d64c5b0 | dispatch.when | 1122 chars | 1 declarations

tcg_d128_fscan_cp_xpose_narrow:10 meta_data.dispatch.when

Let active mean window_left is not None and window_left >= 0, and wl be window_left when active, otherwise 0. Let HostPlan be the actual cached Plan returned by fmha_mxfp8::plan_geometry(B,H,KH,Sq,Skv,D,is_causal,wl,active) in source@6e4ca91bd91d998cac99e8dadd83838e153c679d:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu::fmha_mxfp8::plan_geometry. Let SmCount be that source's actual device_sm_count() result, including its query fallback. Let MapsValid be the actual short-circuit conjunction of K and V encode_tile_map results in fmha_mxfp8_forward, including cache hits. Let split mean HostPlan.use_tc and HostPlan.nsplit>1 and W>0 and W>=HostPlan.workspace. Let psk mean 2*Sq*(H/KH)<=128*HostPlan.qtiles128. Let wide mean (HostPlan.rows_total<2*SmCount and HostPlan.nsplit>=8) or HostPlan.nsplit>128. Select exactly when HostPlan.use_tcg and split and D=128 and not MapsValid and HostPlan.scan_fused and not psk and not wide. The fixed defaults select FP32 partials, QK128, non-PSK rows, cp.async and narrow weighted combine. Both matrix kernels have one warpgroup.

## t_763978ad86eebbba | dispatch.when | 1226 chars | 1 declarations

tcg_d128_flat_cp_xpose_narrow:10 meta_data.dispatch.when

Let active mean window_left is not None and window_left>=0, and wl be window_left when active, otherwise 0. Let HostPlan be the actual Plan returned by fmha_mxfp8::plan_geometry(B,H,KH,Sq,Skv,D,is_causal,wl,active) in source@6e4ca91bd91d998cac99e8dadd83838e153c679d:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu::fmha_mxfp8::plan_geometry. Let SmCount be that source's actual device_sm_count() result, including its fallback. Let MapsValid be the actual short-circuit conjunction of its K/V encode_tile_map results, including descriptor-cache hits. Let split mean HostPlan.use_tc and HostPlan.nsplit>1 and W>0 and W>=HostPlan.workspace. Let scan2p mean HostPlan.T>=128 and HostPlan.rows_total<=512 and not active. Let psk mean 2*Sq*(H/KH)<=128*HostPlan.qtiles128. Let wide mean (HostPlan.rows_total<2*SmCount and HostPlan.nsplit>=8) or HostPlan.nsplit>128. Select exactly when HostPlan.use_tcg and split and D=128 and not MapsValid and not HostPlan.scan_fused and not scan2p and not psk and not wide. The fixed defaults select QK128, one warpgroup per matrix CTA, FP32 partials, non-PSK rows, shared transpose stores, cp.async staging, separate flat scan and narrow combine.

## t_777b2c51b2eb2e6e | dispatch.when | 1217 chars | 1 declarations

tcg_d128_flat_tma_xpose_narrow:10 meta_data.dispatch.when

Let active mean window_left is not None and window_left>=0, and wl be window_left when active, otherwise 0. Let HostPlan be the actual Plan returned by fmha_mxfp8::plan_geometry(B,H,KH,Sq,Skv,D,is_causal,wl,active) in source@6e4ca91bd91d998cac99e8dadd83838e153c679d:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu::fmha_mxfp8::plan_geometry. Let SmCount be that source's actual device_sm_count() result, including its fallback. Let MapsValid be the actual short-circuit conjunction of its K/V encode_tile_map results, including descriptor-cache hits. Let split mean HostPlan.use_tc and HostPlan.nsplit>1 and W>0 and W>=HostPlan.workspace. Let scan2p mean HostPlan.T>=128 and HostPlan.rows_total<=512 and not active. Let psk mean 2*Sq*(H/KH)<=128*HostPlan.qtiles128. Let wide mean (HostPlan.rows_total<2*SmCount and HostPlan.nsplit>=8) or HostPlan.nsplit>128. Select exactly when HostPlan.use_tcg and split and D=128 and MapsValid and not HostPlan.scan_fused and not scan2p and not psk and not wide. The fixed defaults select QK128, one warpgroup per matrix CTA, FP32 partials, non-PSK rows, shared transpose stores, TMA staging, separate flat scan and narrow combine.

## t_784e85277e29a0d7 | dispatch.when | 1203 chars | 1 declarations

tcg_d128_segmented_cp_direct_narrow:10 meta_data.dispatch.when

Let active mean window_left is not None and window_left>=0, and wl be window_left when active, otherwise 0. Let HostPlan be the actual Plan returned by fmha_mxfp8::plan_geometry(B,H,KH,Sq,Skv,D,is_causal,wl,active) in source@6e4ca91bd91d998cac99e8dadd83838e153c679d:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu::fmha_mxfp8::plan_geometry. Let SmCount be that source's actual device_sm_count() result, including its fallback. Let MapsValid be the actual short-circuit conjunction of its K/V encode_tile_map results, including descriptor-cache hits. Let split mean HostPlan.use_tc and HostPlan.nsplit>1 and W>0 and W>=HostPlan.workspace. Let scan2p mean HostPlan.T>=128 and HostPlan.rows_total<=512 and not active. Let psk mean 2*Sq*(H/KH)<=128*HostPlan.qtiles128. Let wide mean (HostPlan.rows_total<2*SmCount and HostPlan.nsplit>=8) or HostPlan.nsplit>128. Select exactly when HostPlan.use_tcg and split and D=128 and not MapsValid and not HostPlan.scan_fused and scan2p and psk and not wide. The fixed defaults select QK128, one warpgroup per matrix CTA, FP32 partials, PSK direct stores, cp.async staging, segment-max/replay scan and narrow combine.

## t_7f18215723622c96 | dispatch.when | 1154 chars | 1 declarations

tcg_d256_flat_tma_wide:9 meta_data.dispatch.when

Let active mean window_left is not None and window_left>=0, and wl be window_left when active, otherwise 0. HostPlan is the actual result of fmha_mxfp8::plan_geometry(B,H,KH,Sq,Skv,D,is_causal,wl,active), from source@6e4ca91bd91d998cac99e8dadd83838e153c679d:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu::fmha_mxfp8::plan_geometry. SmCount is device_sm_count() from that source, including its fallback. Let split mean HostPlan.use_tc and HostPlan.nsplit>1 and W>0 and W>=HostPlan.workspace. MapsValid is the actual short-circuit conjunction of K and V encode_tile_map results in fmha_mxfp8_forward, including cache hits; these probe/encoding results are runtime facts, not guarantees. Let scan2p=(HostPlan.T>=128 and HostPlan.rows_total<=512 and not active). Select exactly when HostPlan.use_tcg and split and D=256 and MapsValid and not HostPlan.scan_fused and not scan2p and ((HostPlan.rows_total<2*SmCount and HostPlan.nsplit>=8) or HostPlan.nsplit>128). Fixed defaults select FP32 partials, QK64=false, GSPLIT=false, FSCAN=false, two warpgroups in both compute passes, and an unsliced wide combine.

## t_7fd13fa03e27dc66 | dispatch.when | 804 chars | 1 declarations

tc_fused:9 meta_data.dispatch.when

Let active mean window_left is not None and window_left >= 0, and let wl be window_left when active, otherwise 0. Let HostPlan be the actual Plan returned by fmha_mxfp8::plan_geometry(B,H,KH,Sq,Skv,D,is_causal,wl,active), from source@6e4ca91bd91d998cac99e8dadd83838e153c679d:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu::plan_geometry. It uses the prepared binary's cached tcg_available probe and device queries, including their source fallbacks. Let split mean HostPlan.use_tc and HostPlan.nsplit>1 and W>0 and W>=HostPlan.workspace, matching the actual forward workspace guard. Select exactly when HostPlan.use_tc and not HostPlan.use_tcg and not HostPlan.use_tcv and not split. This selects fmha_tc_kernel<D,4,PASS_FUSED> with its default PVMODE=0.

## t_8d1d19f35b4997d9 | dispatch.when | 1107 chars | 1 declarations

tcg_d128_fscan_tma_xpose_wide:10 meta_data.dispatch.when

Let active mean window_left is not None and window_left >= 0, and wl be window_left when active, otherwise 0. Let HostPlan be the actual cached Plan returned by fmha_mxfp8::plan_geometry(B,H,KH,Sq,Skv,D,is_causal,wl,active) in source@6e4ca91bd91d998cac99e8dadd83838e153c679d:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu::fmha_mxfp8::plan_geometry. Let SmCount be that source's actual device_sm_count() result, including its query fallback. Let MapsValid be the actual short-circuit conjunction of K and V encode_tile_map results in fmha_mxfp8_forward, including cache hits. Let split mean HostPlan.use_tc and HostPlan.nsplit>1 and W>0 and W>=HostPlan.workspace. Let psk mean 2*Sq*(H/KH)<=128*HostPlan.qtiles128. Let wide mean (HostPlan.rows_total<2*SmCount and HostPlan.nsplit>=8) or HostPlan.nsplit>128. Select exactly when HostPlan.use_tcg and split and D=128 and MapsValid and HostPlan.scan_fused and not psk and wide. The fixed defaults select FP32 partials, QK128, non-PSK rows, TMA and wide weighted combine. Both matrix kernels have one warpgroup.

## t_96bf886a50d7038f | dispatch.when | 1188 chars | 1 declarations

tcg_d128_segmented_tma_direct_wide:10 meta_data.dispatch.when

Let active mean window_left is not None and window_left>=0, and wl be window_left when active, otherwise 0. Let HostPlan be the actual Plan returned by fmha_mxfp8::plan_geometry(B,H,KH,Sq,Skv,D,is_causal,wl,active) in source@6e4ca91bd91d998cac99e8dadd83838e153c679d:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu::fmha_mxfp8::plan_geometry. Let SmCount be that source's actual device_sm_count() result, including its fallback. Let MapsValid be the actual short-circuit conjunction of its K/V encode_tile_map results, including descriptor-cache hits. Let split mean HostPlan.use_tc and HostPlan.nsplit>1 and W>0 and W>=HostPlan.workspace. Let scan2p mean HostPlan.T>=128 and HostPlan.rows_total<=512 and not active. Let psk mean 2*Sq*(H/KH)<=128*HostPlan.qtiles128. Let wide mean (HostPlan.rows_total<2*SmCount and HostPlan.nsplit>=8) or HostPlan.nsplit>128. Select exactly when HostPlan.use_tcg and split and D=128 and MapsValid and not HostPlan.scan_fused and scan2p and psk and wide. The fixed defaults select QK128, one warpgroup per matrix CTA, FP32 partials, PSK direct stores, TMA staging, segment-max/replay scan and wide combine.

## t_9d4574287131cb05 | dispatch.when | 1238 chars | 1 declarations

fused_d128_key:10 meta_data.dispatch.when

Let active mean window_left is not None and window_left >= 0, and let wl be window_left when active, otherwise 0. Let HostPlan be the actual Plan returned by fmha_mxfp8::plan_geometry(B,H,KH,Sq,Skv,D,is_causal,wl,active), from source@6e4ca91bd91d998cac99e8dadd83838e153c679d:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu::fmha_mxfp8::plan_geometry. Let SmCount be device_sm_count() from that source, including its query fallback. HostPlan includes the actual cached capability-probe results. Let CtaCount=B*KH*HostPlan.qtiles128 and split mean HostPlan.use_tc and HostPlan.nsplit>1 and W>0 and W>=HostPlan.workspace. Let qLast=(HostPlan.qtiles128-1)*HostPlan.qph128, loLast=max(0,qLast+Skv-Sq-wl) when active, otherwise 0, and Span=max(1,floor((Skv-1)/128)-floor(loLast/128)+1). Span is the actual useful_key_span result, evaluated on the last query tile; it need not bound the span of earlier windowed tiles. Select exactly when HostPlan.use_tcg and not split and D=128 and CtaCount<=SmCount and Span<8. With the fixed build defaults this selects launch_tcg<128,PASS_FUSED,2> with TMA=false and GSPLIT=false. Descriptor encoding still occurs, but its success does not select this path.

## t_a37f929fd073205d | dispatch.when | 1202 chars | 1 declarations

tcg_d128_flat_cp_direct_narrow:9 meta_data.dispatch.when

Let active mean window_left is not None and window_left>=0, and wl be window_left when active, otherwise 0. Let HostPlan be the actual Plan returned by fmha_mxfp8::plan_geometry(B,H,KH,Sq,Skv,D,is_causal,wl,active) in source@6e4ca91bd91d998cac99e8dadd83838e153c679d:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu::fmha_mxfp8::plan_geometry. Let SmCount be that source's actual device_sm_count() result, including its fallback. Let MapsValid be the actual short-circuit conjunction of its K/V encode_tile_map results, including descriptor-cache hits. Let split mean HostPlan.use_tc and HostPlan.nsplit>1 and W>0 and W>=HostPlan.workspace. Let scan2p mean HostPlan.T>=128 and HostPlan.rows_total<=512 and not active. Let psk mean 2*Sq*(H/KH)<=128*HostPlan.qtiles128. Let wide mean (HostPlan.rows_total<2*SmCount and HostPlan.nsplit>=8) or HostPlan.nsplit>128. Select exactly when HostPlan.use_tcg and split and D=128 and not MapsValid and not HostPlan.scan_fused and not scan2p and psk and not wide. The fixed defaults select QK128, one warpgroup per matrix CTA, FP32 partials, PSK direct stores, cp.async staging, separate flat scan and narrow combine.

## t_aa7d6d64c8179b0a | dispatch.when | 0 chars | 5 declarations

tcg_d128_flat_cp_xpose_wide:10 meta_data.dispatch.when; tcg_d128_segmented_cp_xpose_narrow:10 meta_data.dispatch.when; tcg_d128_segmented_tma_xpose_wide:10 meta_data.dispatch.when; tcg_d256_segmented_cp_narrow:10 meta_data.dispatch.when; tcg_d256_segmented_tma_wide:10 meta_data.dispatch.when



## t_ac63de782f5b8aa7 | dispatch.when | 1191 chars | 1 declarations

root:7 meta_data.dispatch.when

Let active mean window_left is not None and window_left >= 0, and let wl be window_left when active, otherwise 0. Let HostPlan be the actual Plan returned by fmha_mxfp8::plan_geometry(B,H,KH,Sq,Skv,D,is_causal,wl,active), from source@6e4ca91bd91d998cac99e8dadd83838e153c679d:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu::fmha_mxfp8::plan_geometry. Let SmCount be device_sm_count() from the same source. These are actual cached probe and device-query results, including the source's query fallbacks. Let CtaCount=B*KH*HostPlan.qtiles128. Let split mean HostPlan.use_tc and HostPlan.nsplit>1 and W>0 and W>=HostPlan.workspace, matching the actual forward workspace guard. Let MapsValid be the actual short-circuit conjunction of the K and V encode_tile_map results in that source's fmha_mxfp8_forward, including descriptor-cache hits; it is not an assumed descriptor-success guarantee. Select exactly when HostPlan.use_tcg and not split and D=128 and CtaCount>SmCount and MapsValid. With the fixed build defaults this selects launch_tcg<128,PASS_FUSED,1,true> with TMA=true: the single-warpgroup QK64 fused path with deferred PV retirement.

## t_c911dfff9357cec4 | dispatch.when | 1127 chars | 1 declarations

tcg_d256_segmented_tma_narrow:9 meta_data.dispatch.when

Let active mean window_left is not None and window_left>=0, and wl be window_left when active, otherwise 0. HostPlan is the actual result of fmha_mxfp8::plan_geometry(B,H,KH,Sq,Skv,D,is_causal,wl,active), from source@6e4ca91bd91d998cac99e8dadd83838e153c679d:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu::fmha_mxfp8::plan_geometry. SmCount is device_sm_count() from that source, including its fallback. Let split mean HostPlan.use_tc and HostPlan.nsplit>1 and W>0 and W>=HostPlan.workspace. MapsValid is the actual short-circuit conjunction of K and V encode_tile_map results in fmha_mxfp8_forward, including cache hits; these probe/encoding results are runtime facts, not guarantees. Let scan2p=(HostPlan.T>=128 and HostPlan.rows_total<=512 and not active). Select exactly when HostPlan.use_tcg and split and D=256 and MapsValid and not HostPlan.scan_fused and scan2p and not ((HostPlan.rows_total<2*SmCount and HostPlan.nsplit>=8) or HostPlan.nsplit>128). Fixed defaults select FP32 partials, QK64=false, GSPLIT=false, FSCAN=false and two warpgroups in both compute passes.

## t_d79066c2205c44db | dispatch.when | 1226 chars | 1 declarations

tcg_d256_segmented_cp_wide:9 meta_data.dispatch.when

Let active mean window_left is not None and window_left>=0, and wl be window_left when active, otherwise 0. Let HostPlan be the actual result of fmha_mxfp8::plan_geometry(B,H,KH,Sq,Skv,D,is_causal,wl,active), from source@6e4ca91bd91d998cac99e8dadd83838e153c679d:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu::fmha_mxfp8::plan_geometry. Let SmCount be device_sm_count() from that source, including its fallback. Let split mean HostPlan.use_tc and HostPlan.nsplit>1 and W>0 and W>=HostPlan.workspace. Let MapsValid be the actual short-circuit conjunction of the K and V encode_tile_map results in fmha_mxfp8_forward, including descriptor-cache hits. Probe and descriptor success are runtime facts. Let R=B*H*Sq, T=ceil_div(Skv,KeyGroup), Ns=HostPlan.nsplit, Nm=HostPlan.gm_nsplit and scan2p=(T>=128 and R<=512 and not active). Select exactly when HostPlan.use_tcg and split and D=256 and not MapsValid and not HostPlan.scan_fused and scan2p and ((R<2*SmCount and Ns>=8) or Ns>128). Fixed build defaults select cp.async payloads, FP32 partials, QK64=false, GSPLIT=false, FSCAN=false, two warpgroups in each compute pass, and the four-group wide combine without channel slicing.

## t_df599f026b8096b6 | dispatch.when | 1193 chars | 1 declarations

tcg_d128_flat_tma_direct_narrow:10 meta_data.dispatch.when

Let active mean window_left is not None and window_left>=0, and wl be window_left when active, otherwise 0. Let HostPlan be the actual Plan returned by fmha_mxfp8::plan_geometry(B,H,KH,Sq,Skv,D,is_causal,wl,active) in source@6e4ca91bd91d998cac99e8dadd83838e153c679d:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu::fmha_mxfp8::plan_geometry. Let SmCount be that source's actual device_sm_count() result, including its fallback. Let MapsValid be the actual short-circuit conjunction of its K/V encode_tile_map results, including descriptor-cache hits. Let split mean HostPlan.use_tc and HostPlan.nsplit>1 and W>0 and W>=HostPlan.workspace. Let scan2p mean HostPlan.T>=128 and HostPlan.rows_total<=512 and not active. Let psk mean 2*Sq*(H/KH)<=128*HostPlan.qtiles128. Let wide mean (HostPlan.rows_total<2*SmCount and HostPlan.nsplit>=8) or HostPlan.nsplit>128. Select exactly when HostPlan.use_tcg and split and D=128 and MapsValid and not HostPlan.scan_fused and not scan2p and psk and not wide. The fixed defaults select QK128, one warpgroup per matrix CTA, FP32 partials, PSK direct stores, TMA staging, separate flat scan and narrow combine.

## t_ea77104f68680e05 | dispatch.when | 1100 chars | 1 declarations

tcg_d128_fscan_tma_direct_wide:10 meta_data.dispatch.when

Let active mean window_left is not None and window_left>=0, and wl be window_left when active, otherwise 0. Let HostPlan be the actual Plan returned by fmha_mxfp8::plan_geometry(B,H,KH,Sq,Skv,D,is_causal,wl,active) in source@6e4ca91bd91d998cac99e8dadd83838e153c679d:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu::fmha_mxfp8::plan_geometry. Let SmCount be that source's actual device_sm_count() result, including its fallback. Let MapsValid be the actual short-circuit conjunction of its K/V encode_tile_map results, including descriptor-cache hits. Let split mean HostPlan.use_tc and HostPlan.nsplit>1 and W>0 and W>=HostPlan.workspace. Let psk mean 2*Sq*(H/KH)<=128*HostPlan.qtiles128. Let wide mean (HostPlan.rows_total<2*SmCount and HostPlan.nsplit>=8) or HostPlan.nsplit>128. Select exactly when HostPlan.use_tcg and split and D=128 and MapsValid and HostPlan.scan_fused and psk and wide. The fixed defaults select QK128, one warpgroup per matrix CTA, FP32 partials, PSK direct stores, TMA staging, locally fused scan and weighted wide combine.

## t_f53671150683e4f7 | dispatch.when | 1109 chars | 1 declarations

tcg_d128_fscan_cp_direct_wide:10 meta_data.dispatch.when

Let active mean window_left is not None and window_left>=0, and wl be window_left when active, otherwise 0. Let HostPlan be the actual Plan returned by fmha_mxfp8::plan_geometry(B,H,KH,Sq,Skv,D,is_causal,wl,active) in source@6e4ca91bd91d998cac99e8dadd83838e153c679d:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu::fmha_mxfp8::plan_geometry. Let SmCount be that source's actual device_sm_count() result, including its fallback. Let MapsValid be the actual short-circuit conjunction of its K/V encode_tile_map results, including descriptor-cache hits. Let split mean HostPlan.use_tc and HostPlan.nsplit>1 and W>0 and W>=HostPlan.workspace. Let psk mean 2*Sq*(H/KH)<=128*HostPlan.qtiles128. Let wide mean (HostPlan.rows_total<2*SmCount and HostPlan.nsplit>=8) or HostPlan.nsplit>128. Select exactly when HostPlan.use_tcg and split and D=128 and not MapsValid and HostPlan.scan_fused and psk and wide. The fixed defaults select QK128, one warpgroup per matrix CTA, FP32 partials, PSK direct stores, cp.async staging, locally fused scan and weighted wide combine.

## t_f6535cbaeee1904b | dispatch.when | 1146 chars | 1 declarations

tcg_d256_flat_cp_narrow:9 meta_data.dispatch.when

Let active mean window_left is not None and window_left>=0, and wl be window_left when active, otherwise 0. HostPlan is the actual result of fmha_mxfp8::plan_geometry(B,H,KH,Sq,Skv,D,is_causal,wl,active), from source@6e4ca91bd91d998cac99e8dadd83838e153c679d:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu::fmha_mxfp8::plan_geometry. SmCount is device_sm_count() from that source, including its fallback. Let split mean HostPlan.use_tc and HostPlan.nsplit>1 and W>0 and W>=HostPlan.workspace. MapsValid is the actual short-circuit conjunction of K and V encode_tile_map results in fmha_mxfp8_forward, including cache hits; these probe/encoding results are runtime facts, not guarantees. Let scan2p=(HostPlan.T>=128 and HostPlan.rows_total<=512 and not active). Select exactly when HostPlan.use_tcg and split and D=256 and not MapsValid and not HostPlan.scan_fused and not scan2p and not ((HostPlan.rows_total<2*SmCount and HostPlan.nsplit>=8) or HostPlan.nsplit>128). Fixed defaults select TMA=false, FP32 partials, QK64=false, GSPLIT=false, FSCAN=false and two warpgroups in both compute passes.

## t_fc5edc57ee3b47eb | dispatch.when | 1106 chars | 1 declarations

tcg_d128_fscan_tma_direct_narrow:10 meta_data.dispatch.when

Let active mean window_left is not None and window_left>=0, and wl be window_left when active, otherwise 0. Let HostPlan be the actual Plan returned by fmha_mxfp8::plan_geometry(B,H,KH,Sq,Skv,D,is_causal,wl,active) in source@6e4ca91bd91d998cac99e8dadd83838e153c679d:kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu::fmha_mxfp8::plan_geometry. Let SmCount be that source's actual device_sm_count() result, including its fallback. Let MapsValid be the actual short-circuit conjunction of its K/V encode_tile_map results, including descriptor-cache hits. Let split mean HostPlan.use_tc and HostPlan.nsplit>1 and W>0 and W>=HostPlan.workspace. Let psk mean 2*Sq*(H/KH)<=128*HostPlan.qtiles128. Let wide mean (HostPlan.rows_total<2*SmCount and HostPlan.nsplit>=8) or HostPlan.nsplit>128. Select exactly when HostPlan.use_tcg and split and D=128 and MapsValid and HostPlan.scan_fused and psk and not wide. The fixed defaults select QK128, one warpgroup per matrix CTA, FP32 partials, PSK direct stores, TMA staging, locally fused scan and weighted narrow combine.

## t_0a3b38e61b5a92ef | math.arithmetic | 186 chars | 1 declarations

tcg_d256_segmented_cp_wide:225 design.math_reshape.rewrite[6].arithmetic.approx

The final reciprocal and CUDA logf retain the source compilation path. unknown: no end-to-end bound is supplied; the shared external tolerance record is not a numerical design guarantee.

## t_129f4743ea49bdf8 | math.arithmetic | 156 chars | 3 declarations

tcg_d128_fscan_cp_xpose_narrow:200 design.math_reshape.rewrite[5].arithmetic.approx; tcg_d128_fscan_cp_xpose_wide:213 design.math_reshape.rewrite[5].arithmetic.approx; tcg_d128_fscan_tma_xpose_wide:215 design.math_reshape.rewrite[5].arithmetic.approx

Weight uses CUDA expf; reciprocal/division and CUDA logf retain the fixed compilation path. unknown: no matched numerical bound for final O/LSE is supplied.

## t_1d69550c2206dc5d | math.arithmetic | 196 chars | 6 declarations

fused_d128_cp:155 design.math_reshape.rewrite[3].arithmetic.approx; fused_d128_group:154 design.math_reshape.rewrite[3].arithmetic.approx; fused_d128_key:157 design.math_reshape.rewrite[3].arithmetic.approx; fused_d256_cp:151 design.math_reshape.rewrite[3].arithmetic.approx; fused_d256_tma:151 design.math_reshape.rewrite[3].arithmetic.approx; root:365 design.math_reshape.rewrite[3].arithmetic.approx

The final reciprocal/division and scalar CUDA logf retain the source compilation path. unknown: no matched numerical error bound for them or for final O/LSE is established by the supplied sources.

## t_1f07ca09d792212e | math.arithmetic | 479 chars | 2 declarations

fused_d128_group:79 design.math_reshape.rewrite[1].arithmetic.accumulate; fused_d128_key:81 design.math_reshape.rewrite[1].arithmetic.accumulate

The mma32 operands denote native block-scaled values of the stored E4M3 payloads and their E8M0 blocks, without separately rounded FP32 operand decoding. Actual products consume those narrow payloads and scales directly with native block-scaled tcgen05 MMA, four ordered 32-channel contractions for the full group; there is no FP32 decoded operand array or intervening BF16 cast. FP32 accumulation does not assert reference reduction order or a bound on native internal rounding.

## t_208350918a6c059c | math.arithmetic | 72 chars | 15 declarations

tcg_d128_flat_cp_direct_narrow:168 design.math_reshape.rewrite[4].arithmetic.approx; tcg_d128_flat_cp_direct_wide:168 design.math_reshape.rewrite[4].arithmetic.approx; tcg_d128_flat_cp_xpose_narrow:174 design.math_reshape.rewrite[4].arithmetic.approx; tcg_d128_flat_cp_xpose_wide:157 design.math_reshape.rewrite[4].arithmetic.approx; tcg_d128_flat_tma_direct_narrow:169 design.math_reshape.rewrite[4].arithmetic.approx; tcg_d128_flat_tma_direct_wide:169 design.math_reshape.rewrite[4].arithmetic.approx; tcg_d128_flat_tma_xpose_narrow:171 design.math_reshape.rewrite[4].arithmetic.approx; tcg_d128_flat_tma_xpose_wide:173 design.math_reshape.rewrite[4].arithmetic.approx; tcg_d128_segmented_cp_direct_narrow:190 design.math_reshape.rewrite[5].arithmetic.approx; tcg_d128_segmented_cp_direct_wide:189 design.math_reshape.rewrite[5].arithmetic.approx; tcg_d128_segmented_cp_xpose_narrow:179 design.math_reshape.rewrite[5].arithmetic.approx; tcg_d128_segmented_tma_direct_narrow:190 design.math_reshape.rewrite[5].arithmetic.approx; tcg_d128_segmented_tma_direct_wide:190 design.math_reshape.rewrite[5].arithmetic.approx; tcg_d128_segmented_tma_xpose_narrow:195 design.math_reshape.rewrite[5].arithmetic.approx; tcg_d128_segmented_tma_xpose_wide:179 design.math_reshape.rewrite[5].arithmetic.approx

weight uses CUDA expf; unknown: no matched end-to-end bound is supplied.

## t_213d0b7e26ddfc7b | math.arithmetic | 139 chars | 1 declarations

tcg_d128_fscan_tma_xpose_narrow:201 design.math_reshape.rewrite[5].arithmetic.approx

Weight uses CUDA expf; reciprocal/division and logf retain the source compilation path. unknown: no matched final-output bound is supplied.

## t_24909227b54cf85d | math.arithmetic | 469 chars | 2 declarations

fused_d128_cp:79 design.math_reshape.rewrite[1].arithmetic.accumulate; root:289 design.math_reshape.rewrite[1].arithmetic.accumulate

The mma32 operands denote native block-scaled values of the stored E4M3 payloads and their E8M0 blocks, without separately rounded FP32 operand decoding. Actual products consume those narrow payloads and scales directly with native block-scaled tcgen05 MMA, four ordered 32-channel contractions per half; there is no FP32 decoded operand array or intervening BF16 cast. FP32 accumulation does not assert reference reduction order or a bound on native internal rounding.

## t_2697af0059a8e6dd | math.arithmetic | 178 chars | 1 declarations

tcg_d256_segmented_cp_wide:166 design.math_reshape.rewrite[4].arithmetic.accumulate

The separate even/odd probability chains use FP32 adds. Scalar denominator multiply/add updates retain source compiler defaults; no additional contraction restriction is imposed.

## t_2d0ba00fe3062ab1 | math.arithmetic | 265 chars | 20 declarations

tcg_d128_flat_cp_direct_narrow:84 design.math_reshape.rewrite[1].arithmetic.accumulate; tcg_d128_flat_cp_direct_wide:84 design.math_reshape.rewrite[1].arithmetic.accumulate; tcg_d128_flat_cp_xpose_narrow:86 design.math_reshape.rewrite[1].arithmetic.accumulate; tcg_d128_flat_cp_xpose_wide:69 design.math_reshape.rewrite[1].arithmetic.accumulate; tcg_d128_flat_tma_direct_narrow:85 design.math_reshape.rewrite[1].arithmetic.accumulate; tcg_d128_flat_tma_direct_wide:85 design.math_reshape.rewrite[1].arithmetic.accumulate; tcg_d128_flat_tma_xpose_narrow:86 design.math_reshape.rewrite[1].arithmetic.accumulate; tcg_d128_flat_tma_xpose_wide:85 design.math_reshape.rewrite[1].arithmetic.accumulate; tcg_d128_fscan_cp_direct_narrow:84 design.math_reshape.rewrite[1].arithmetic.accumulate; tcg_d128_fscan_cp_direct_wide:84 design.math_reshape.rewrite[1].arithmetic.accumulate; tcg_d128_fscan_tma_direct_narrow:84 design.math_reshape.rewrite[1].arithmetic.accumulate; tcg_d128_fscan_tma_direct_wide:84 design.math_reshape.rewrite[1].arithmetic.accumulate; tcg_d128_fscan_tma_xpose_narrow:85 design.math_reshape.rewrite[1].arithmetic.accumulate; tcg_d128_segmented_cp_direct_narrow:85 design.math_reshape.rewrite[1].arithmetic.accumulate; tcg_d128_segmented_cp_direct_wide:84 design.math_reshape.rewrite[1].arithmetic.accumulate; tcg_d128_segmented_cp_xpose_narrow:69 design.math_reshape.rewrite[1].arithmetic.accumulate; tcg_d128_segmented_tma_direct_narrow:85 design.math_reshape.rewrite[1].arithmetic.accumulate; tcg_d128_segmented_tma_direct_wide:85 design.math_reshape.rewrite[1].arithmetic.accumulate; tcg_d128_segmented_tma_xpose_narrow:85 design.math_reshape.rewrite[1].arithmetic.accumulate; tcg_d128_segmented_tma_xpose_wide:69 design.math_reshape.rewrite[1].arithmetic.accumulate

mma32 consumes native E4M3 payloads and E8M0 block scales through four ordered 32-channel tcgen05 contractions. FP32 accumulation does not assert decoded FP32 operands, reference reduction order or a bound on native internal rounding; no BF16 conversion intervenes.

## t_2ff5177f3ac2b0f8 | math.arithmetic | 216 chars | 2 declarations

tcg_d128_fscan_cp_xpose_narrow:198 design.math_reshape.rewrite[5].arithmetic.accumulate; tcg_d128_fscan_cp_xpose_wide:211 design.math_reshape.rewrite[5].arithmetic.accumulate

Weighted multiply-add expressions retain the fixed CUDA -O3 compilation path and its contraction choices; no separate-rounding or bitwise equivalence to a separately rounded multiply followed by addition is asserted.

## t_41ff293639d19806 | math.arithmetic | 178 chars | 5 declarations

tc_fused:154 design.math_reshape.rewrite[3].arithmetic.accumulate; tc_split_flat_narrow:214 design.math_reshape.rewrite[4].arithmetic.accumulate; tc_split_flat_wide:214 design.math_reshape.rewrite[4].arithmetic.accumulate; tc_split_segmented_narrow:249 design.math_reshape.rewrite[6].arithmetic.accumulate; tc_split_segmented_wide:249 design.math_reshape.rewrite[6].arithmetic.accumulate

PV uses ordered 16-key BF16 products directly accumulating into the rescaled FP32 numerator. BF16 V scaling can round subnormal products differently from FP32 reference decoding.

## t_58e4513da1b3994b | math.arithmetic | 217 chars | 1 declarations

tcg_d256_segmented_cp_wide:161 design.math_reshape.rewrite[4].arithmetic.approx

E1 is expf2_bias<0> with FMHA_EXP_NO_KTINY=1 and FMHA_BIAS_MAGIC_FOLD=1; alpha uses scalar CUDA expf. unknown: no reliable bound is supplied for these approximations or their effect at probability-rounding boundaries.

## t_5ded117f1a238a68 | math.arithmetic | 338 chars | 2 declarations

fused_d128_group:123 design.math_reshape.rewrite[2].arithmetic.approx; fused_d128_key:126 design.math_reshape.rewrite[2].arithmetic.approx

E16 is expf2_bias<4> with the fixed defaults FMHA_BIAS_MAGIC_FOLD=1 and FMHA_EXP_NO_KTINY=1, including ex2.approx.ftz.f32. alpha uses scalar CUDA expf. unknown: a reliable error bound for these paths and their effect at probability-rounding boundaries is not supplied; the scoped acceptance records establish no bound for this fused path.

## t_5f5d7f340bac8f0a | math.arithmetic | 461 chars | 2 declarations

fused_d256_cp:76 design.math_reshape.rewrite[1].arithmetic.accumulate; fused_d256_tma:76 design.math_reshape.rewrite[1].arithmetic.accumulate

The mma32 operands denote native block-scaled values of the stored E4M3 payloads and their E8M0 blocks, without separately rounded FP32 operand decoding. Actual products consume those narrow payloads and scales directly with native block-scaled tcgen05 MMA, eight ordered 32-channel contractions; there is no FP32 decoded operand array or intervening BF16 cast. FP32 accumulation does not assert reference reduction order or a bound on native internal rounding.

## t_62d1bb4922201bb4 | math.arithmetic | 139 chars | 4 declarations

tcg_d128_fscan_cp_direct_narrow:197 design.math_reshape.rewrite[5].arithmetic.approx; tcg_d128_fscan_cp_direct_wide:212 design.math_reshape.rewrite[5].arithmetic.approx; tcg_d128_fscan_tma_direct_narrow:197 design.math_reshape.rewrite[5].arithmetic.approx; tcg_d128_fscan_tma_direct_wide:213 design.math_reshape.rewrite[5].arithmetic.approx

Weights use CUDA expf; reciprocal/division and logf retain the source compilation path. unknown: no matched final-output bound is supplied.

## t_6427596bde820cef | math.arithmetic | 175 chars | 4 declarations

tc_split_flat_narrow:237 design.math_reshape.rewrite[5].arithmetic.approx; tc_split_flat_wide:249 design.math_reshape.rewrite[5].arithmetic.approx; tc_split_segmented_narrow:272 design.math_reshape.rewrite[7].arithmetic.approx; tc_split_segmented_wide:297 design.math_reshape.rewrite[7].arithmetic.approx

Reciprocal and scalar CUDA logf follow the source build. unknown: no matched end-to-end O/LSE bound is supplied; shared scoped acceptance records are not numerical guarantees.

## t_726763f177f27e11 | math.arithmetic | 270 chars | 4 declarations

tcg_d128_fscan_cp_xpose_narrow:173 design.math_reshape.rewrite[4].arithmetic.accumulate; tcg_d128_fscan_cp_xpose_wide:173 design.math_reshape.rewrite[4].arithmetic.accumulate; tcg_d128_fscan_tma_xpose_wide:173 design.math_reshape.rewrite[4].arithmetic.accumulate; tcg_d128_segmented_cp_xpose_wide:196 design.math_reshape.rewrite[5].arithmetic.accumulate

PV uses the scores step's native block-scaled path, with E4M3 Prob8 and unit E8M0 scales, original E4M3 V and its key-axis E8M0 scales. No BF16 intermediate is introduced. The numerator rescale is a multiply when alpha differs from one; the post-PV addition is separate.

## t_7513ffed08c12a9f | math.arithmetic | 88 chars | 4 declarations

tc_split_flat_narrow:216 design.math_reshape.rewrite[4].arithmetic.approx; tc_split_flat_wide:216 design.math_reshape.rewrite[4].arithmetic.approx; tc_split_segmented_narrow:251 design.math_reshape.rewrite[6].arithmetic.approx; tc_split_segmented_wide:251 design.math_reshape.rewrite[6].arithmetic.approx

Weight uses scalar CUDA expf; unknown: no matched error bound is supplied for this path.

## t_8508f78ae6d71a30 | math.arithmetic | 309 chars | 7 declarations

tcg_d256_flat_cp_narrow:84 design.math_reshape.rewrite[1].arithmetic.accumulate; tcg_d256_flat_cp_wide:85 design.math_reshape.rewrite[1].arithmetic.accumulate; tcg_d256_flat_tma_narrow:84 design.math_reshape.rewrite[1].arithmetic.accumulate; tcg_d256_flat_tma_wide:84 design.math_reshape.rewrite[1].arithmetic.accumulate; tcg_d256_segmented_cp_narrow:69 design.math_reshape.rewrite[1].arithmetic.accumulate; tcg_d256_segmented_tma_narrow:84 design.math_reshape.rewrite[1].arithmetic.accumulate; tcg_d256_segmented_tma_wide:69 design.math_reshape.rewrite[1].arithmetic.accumulate

mma32 consumes original E4M3 payloads and E8M0 blocks through native block-scaled tcgen05 products: eight ordered 32-channel contractions with FP32 accumulation, without a separately rounded FP32 decoded array or BF16 intermediate. Native internal rounding is not equated with the reference contraction order.

## t_85a066342cead513 | math.arithmetic | 284 chars | 1 declarations

tc_fused:118 design.math_reshape.rewrite[2].arithmetic.approx

alpha and Prob use scalar CUDA expf under the source's -O3 build without --use_fast_math. unknown: no matched error bound for this fused path or its probability-rounding boundaries is supplied. The shared scoped BF16-partial margin record supplies no guarantee for these computations.

## t_899c681a99ad2c67 | math.arithmetic | 215 chars | 1 declarations

tcg_d128_fscan_cp_direct_wide:210 design.math_reshape.rewrite[5].arithmetic.accumulate

Weighted additions retain source scalar compiler defaults without an extra contraction restriction. Numerator split order is fixed within each group; denominator trees and final group-order sums follow the formulas.

## t_90eb4da75ae3b0a9 | math.arithmetic | 314 chars | 1 declarations

tcg_d256_segmented_cp_wide:84 design.math_reshape.rewrite[1].arithmetic.accumulate

mma32 here consumes stored E4M3 payloads and E8M0 blocks directly through native block-scaled tcgen05 products: eight ordered 32-channel contractions with FP32 accumulation, without a decoded FP32 array or intermediate BF16 cast. Native internal rounding and reference reduction-order equivalence are not asserted.

## t_977fe32425e77e01 | math.arithmetic | 182 chars | 7 declarations

tcg_d256_flat_cp_narrow:131 design.math_reshape.rewrite[3].arithmetic.approx; tcg_d256_flat_cp_wide:132 design.math_reshape.rewrite[3].arithmetic.approx; tcg_d256_flat_tma_narrow:131 design.math_reshape.rewrite[3].arithmetic.approx; tcg_d256_flat_tma_wide:131 design.math_reshape.rewrite[3].arithmetic.approx; tcg_d256_segmented_cp_narrow:138 design.math_reshape.rewrite[4].arithmetic.approx; tcg_d256_segmented_tma_narrow:153 design.math_reshape.rewrite[4].arithmetic.approx; tcg_d256_segmented_tma_wide:138 design.math_reshape.rewrite[4].arithmetic.approx

E1 is expf2_bias<0> with FMHA_EXP_NO_KTINY=1 and FMHA_BIAS_MAGIC_FOLD=1; alpha uses CUDA expf. unknown: no reliable bound for these approximations or probability-rounding boundaries.

## t_98481cb1b834f0db | math.arithmetic | 143 chars | 7 declarations

tcg_d256_flat_cp_narrow:157 design.math_reshape.rewrite[4].arithmetic.accumulate; tcg_d256_flat_cp_wide:158 design.math_reshape.rewrite[4].arithmetic.accumulate; tcg_d256_flat_tma_narrow:157 design.math_reshape.rewrite[4].arithmetic.accumulate; tcg_d256_flat_tma_wide:157 design.math_reshape.rewrite[4].arithmetic.accumulate; tcg_d256_segmented_cp_narrow:164 design.math_reshape.rewrite[5].arithmetic.accumulate; tcg_d256_segmented_tma_narrow:179 design.math_reshape.rewrite[5].arithmetic.accumulate; tcg_d256_segmented_tma_wide:164 design.math_reshape.rewrite[5].arithmetic.accumulate

PV uses the native path defined in scores, original V and key-axis scales, unit-scale E4M3 probabilities, and four ordered 32-key contractions.

## t_9dc7fb317592f8a2 | math.arithmetic | 186 chars | 1 declarations

tcg_d256_segmented_cp_wide:192 design.math_reshape.rewrite[5].arithmetic.accumulate

PV uses the native path defined in scores, with E4M3 probabilities and unit E8M0 scales, original E4M3 V and its key-axis scales, four ordered 32-key contractions, and FP32 accumulation.

## t_a00390d8c2507a93 | math.arithmetic | 122 chars | 7 declarations

tcg_d256_flat_cp_narrow:135 design.math_reshape.rewrite[3].arithmetic.accumulate; tcg_d256_flat_cp_wide:136 design.math_reshape.rewrite[3].arithmetic.accumulate; tcg_d256_flat_tma_narrow:135 design.math_reshape.rewrite[3].arithmetic.accumulate; tcg_d256_flat_tma_wide:135 design.math_reshape.rewrite[3].arithmetic.accumulate; tcg_d256_segmented_cp_narrow:142 design.math_reshape.rewrite[4].arithmetic.accumulate; tcg_d256_segmented_tma_narrow:157 design.math_reshape.rewrite[4].arithmetic.accumulate; tcg_d256_segmented_tma_wide:142 design.math_reshape.rewrite[4].arithmetic.accumulate

Scalar denominator multiply/add updates retain source compiler defaults; no additional contraction restriction is imposed.

## t_a3f5f8bed4ae4b97 | math.arithmetic | 252 chars | 8 declarations

tcg_d128_flat_cp_xpose_narrow:171 design.math_reshape.rewrite[4].arithmetic.accumulate; tcg_d128_flat_cp_xpose_wide:154 design.math_reshape.rewrite[4].arithmetic.accumulate; tcg_d128_flat_tma_xpose_narrow:168 design.math_reshape.rewrite[4].arithmetic.accumulate; tcg_d128_flat_tma_xpose_wide:170 design.math_reshape.rewrite[4].arithmetic.accumulate; tcg_d128_fscan_tma_xpose_narrow:173 design.math_reshape.rewrite[4].arithmetic.accumulate; tcg_d128_segmented_cp_xpose_narrow:176 design.math_reshape.rewrite[5].arithmetic.accumulate; tcg_d128_segmented_tma_xpose_narrow:192 design.math_reshape.rewrite[5].arithmetic.accumulate; tcg_d128_segmented_tma_xpose_wide:176 design.math_reshape.rewrite[5].arithmetic.accumulate

PV uses the scores step's native block-scaled contraction with E4M3 Prob8 and unit E8M0 scales, and original E4M3 V with its key-axis scales. Numerator rescaling is a multiply and the later PV addition is separate, as selected by FMHA_RESCALE_LAYOUT=0.

## t_acfd76d818ff5701 | math.arithmetic | 272 chars | 4 declarations

tcg_d128_fscan_cp_xpose_narrow:85 design.math_reshape.rewrite[1].arithmetic.accumulate; tcg_d128_fscan_cp_xpose_wide:85 design.math_reshape.rewrite[1].arithmetic.accumulate; tcg_d128_fscan_tma_xpose_wide:85 design.math_reshape.rewrite[1].arithmetic.accumulate; tcg_d128_segmented_cp_xpose_wide:87 design.math_reshape.rewrite[1].arithmetic.accumulate

mma32 here consumes native E4M3 payloads and E8M0 block scales through tcgen05 block-scaled contraction, four ordered 32-channel products. FP32 accumulation does not assert separately decoded FP32 products, reference reduction order or a bound on native internal rounding.

## t_b26fc75549d38805 | math.arithmetic | 68 chars | 8 declarations

tcg_d256_flat_cp_narrow:158 design.math_reshape.rewrite[4].arithmetic.approx; tcg_d256_flat_cp_wide:159 design.math_reshape.rewrite[4].arithmetic.approx; tcg_d256_flat_tma_narrow:158 design.math_reshape.rewrite[4].arithmetic.approx; tcg_d256_flat_tma_wide:158 design.math_reshape.rewrite[4].arithmetic.approx; tcg_d256_segmented_cp_narrow:165 design.math_reshape.rewrite[5].arithmetic.approx; tcg_d256_segmented_cp_wide:196 design.math_reshape.rewrite[5].arithmetic.approx; tcg_d256_segmented_tma_narrow:180 design.math_reshape.rewrite[5].arithmetic.approx; tcg_d256_segmented_tma_wide:165 design.math_reshape.rewrite[5].arithmetic.approx

weight uses CUDA expf; unknown: no reliable error bound is supplied.

## t_b46cf5b44fabf5fa | math.arithmetic | 231 chars | 5 declarations

tc_fused:85 design.math_reshape.rewrite[1].arithmetic.accumulate; tc_split_flat_narrow:98 design.math_reshape.rewrite[1].arithmetic.accumulate; tc_split_flat_wide:98 design.math_reshape.rewrite[1].arithmetic.accumulate; tc_split_segmented_narrow:99 design.math_reshape.rewrite[1].arithmetic.accumulate; tc_split_segmented_wide:99 design.math_reshape.rewrite[1].arithmetic.accumulate

Each 32-channel FP8 product starts at zero. The scale correction is ordered by channel block, without separately decoded FP32 Q/K operands. FP32 MMA accumulation does not assert the reference's internal reduction order or rounding.

## t_b4ea7466063a4e7b | math.arithmetic | 121 chars | 15 declarations

tcg_d128_flat_cp_direct_narrow:186 design.math_reshape.rewrite[5].arithmetic.approx; tcg_d128_flat_cp_direct_wide:198 design.math_reshape.rewrite[5].arithmetic.approx; tcg_d128_flat_cp_xpose_narrow:193 design.math_reshape.rewrite[5].arithmetic.approx; tcg_d128_flat_cp_xpose_wide:176 design.math_reshape.rewrite[5].arithmetic.approx; tcg_d128_flat_tma_direct_narrow:187 design.math_reshape.rewrite[5].arithmetic.approx; tcg_d128_flat_tma_direct_wide:198 design.math_reshape.rewrite[5].arithmetic.approx; tcg_d128_flat_tma_xpose_narrow:190 design.math_reshape.rewrite[5].arithmetic.approx; tcg_d128_flat_tma_xpose_wide:205 design.math_reshape.rewrite[5].arithmetic.approx; tcg_d128_segmented_cp_direct_narrow:208 design.math_reshape.rewrite[6].arithmetic.approx; tcg_d128_segmented_cp_direct_wide:221 design.math_reshape.rewrite[6].arithmetic.approx; tcg_d128_segmented_cp_xpose_narrow:198 design.math_reshape.rewrite[6].arithmetic.approx; tcg_d128_segmented_tma_direct_narrow:208 design.math_reshape.rewrite[6].arithmetic.approx; tcg_d128_segmented_tma_direct_wide:223 design.math_reshape.rewrite[6].arithmetic.approx; tcg_d128_segmented_tma_xpose_narrow:214 design.math_reshape.rewrite[6].arithmetic.approx; tcg_d128_segmented_tma_xpose_wide:198 design.math_reshape.rewrite[6].arithmetic.approx

Reciprocal/division and CUDA logf retain the source compilation path; unknown: no matched final-output bound is supplied.

## t_b50a722060248c5a | math.arithmetic | 337 chars | 2 declarations

fused_d256_cp:119 design.math_reshape.rewrite[2].arithmetic.approx; fused_d256_tma:119 design.math_reshape.rewrite[2].arithmetic.approx

E1 is expf2_bias<0> with the fixed defaults FMHA_BIAS_MAGIC_FOLD=1 and FMHA_EXP_NO_KTINY=1, including ex2.approx.ftz.f32. alpha uses scalar CUDA expf. unknown: a reliable error bound for these paths and their effect at probability-rounding boundaries is not supplied; the external tolerance record supplies no numerical design guarantee.

## t_b602e70ba25899cd | math.arithmetic | 276 chars | 4 declarations

tc_split_flat_narrow:171 design.math_reshape.rewrite[3].arithmetic.approx; tc_split_flat_wide:171 design.math_reshape.rewrite[3].arithmetic.approx; tc_split_segmented_narrow:206 design.math_reshape.rewrite[5].arithmetic.approx; tc_split_segmented_wide:206 design.math_reshape.rewrite[5].arithmetic.approx

alpha and Prob use scalar CUDA expf under the source's -O3 build without --use_fast_math. unknown: no matched error bound for these scalar computations or probability-rounding boundaries is supplied. The shared BF16-partial margin record does not bound this FP32-partial path.

## t_b7a5c7a9ffd27828 | math.arithmetic | 338 chars | 2 declarations

fused_d128_cp:124 design.math_reshape.rewrite[2].arithmetic.approx; root:334 design.math_reshape.rewrite[2].arithmetic.approx

E16 is expf2_bias<4> with the fixed defaults FMHA_BIAS_MAGIC_FOLD=1 and FMHA_EXP_NO_KTINY=1, including ex2.approx.ftz.f32. alpha uses scalar CUDA expf. unknown: a reliable error bound for these paths and their effect at probability-rounding boundaries is not supplied; the external tolerance record supplies no numerical design guarantee.

## t_bacb6847a134b177 | math.arithmetic | 128 chars | 1 declarations

tc_fused:158 design.math_reshape.rewrite[3].arithmetic.approx

The final reciprocal and scalar CUDA logf follow the source build. unknown: no matched end-to-end O/LSE error bound is supplied.

## t_bc1d5bba72546fb7 | math.arithmetic | 192 chars | 1 declarations

tcg_d128_fscan_tma_xpose_narrow:199 design.math_reshape.rewrite[5].arithmetic.accumulate

Weighted scalar multiply-add expressions use source compiler defaults without an additional contraction restriction; no bitwise equivalence to separately rounded weighted partials is asserted.

## t_c838c7304bbf87d7 | math.arithmetic | 133 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:231 design.math_reshape.rewrite[6].arithmetic.approx

Reciprocal/division and CUDA logf retain the fixed compilation path. unknown: no matched numerical bound for final O/LSE is supplied.

## t_caa1460265b7dd84 | math.arithmetic | 349 chars | 1 declarations

tcg_d128_fscan_tma_xpose_wide:212 design.math_reshape.rewrite[5].arithmetic.accumulate

Weighted numerator multiply-add expressions retain the fixed CUDA -O3 compilation path and its contraction choices. Denominator products precede the FP32 bin tree; the subsequent bin and group sums are separate FP32 additions. No bitwise equivalence to the narrow merge or to a separately rounded numerator multiply followed by addition is asserted.

## t_cc3b6621ed7b6166 | math.arithmetic | 272 chars | 4 declarations

tcg_d128_fscan_cp_xpose_narrow:138 design.math_reshape.rewrite[3].arithmetic.approx; tcg_d128_fscan_cp_xpose_wide:138 design.math_reshape.rewrite[3].arithmetic.approx; tcg_d128_fscan_tma_xpose_wide:138 design.math_reshape.rewrite[3].arithmetic.approx; tcg_d128_segmented_cp_xpose_wide:161 design.math_reshape.rewrite[4].arithmetic.approx

E16 is expf2_bias<4> with FMHA_BIAS_MAGIC_FOLD=1 and FMHA_EXP_NO_KTINY=1. alpha uses CUDA expf. unknown: a reliable error bound for this path and its probability-rounding effects is not supplied; the shared external tolerance record provides no numerical design guarantee.

## t_d1a4d1c2e1b292e8 | math.arithmetic | 99 chars | 1 declarations

tcg_d128_fscan_tma_direct_wide:212 design.math_reshape.rewrite[5].arithmetic.accumulate

Weighted additions retain source scalar compiler defaults without an extra contraction restriction.

## t_e22c22e056f1cded | math.arithmetic | 198 chars | 6 declarations

fused_d128_cp:151 design.math_reshape.rewrite[3].arithmetic.accumulate; fused_d128_group:150 design.math_reshape.rewrite[3].arithmetic.accumulate; fused_d128_key:153 design.math_reshape.rewrite[3].arithmetic.accumulate; fused_d256_cp:147 design.math_reshape.rewrite[3].arithmetic.accumulate; fused_d256_tma:147 design.math_reshape.rewrite[3].arithmetic.accumulate; root:361 design.math_reshape.rewrite[3].arithmetic.accumulate

PV uses the native block-scaled path described in scores: E4M3 probabilities with E8M0 unit scales and original E4M3 V with its key-axis E8M0 scales, with FP32 accumulation and no BF16 intermediate.

## t_e31de55daecb1205 | math.arithmetic | 154 chars | 2 declarations

tcg_d128_fscan_cp_direct_narrow:195 design.math_reshape.rewrite[5].arithmetic.accumulate; tcg_d128_fscan_tma_direct_narrow:195 design.math_reshape.rewrite[5].arithmetic.accumulate

Weighted additions retain source scalar compiler defaults without an extra contraction restriction; split order is fixed and no parallel sum tree is used.

## t_ec5636ba0c3dea20 | math.arithmetic | 260 chars | 20 declarations

tcg_d128_flat_cp_direct_narrow:137 design.math_reshape.rewrite[3].arithmetic.approx; tcg_d128_flat_cp_direct_wide:137 design.math_reshape.rewrite[3].arithmetic.approx; tcg_d128_flat_cp_xpose_narrow:139 design.math_reshape.rewrite[3].arithmetic.approx; tcg_d128_flat_cp_xpose_wide:122 design.math_reshape.rewrite[3].arithmetic.approx; tcg_d128_flat_tma_direct_narrow:138 design.math_reshape.rewrite[3].arithmetic.approx; tcg_d128_flat_tma_direct_wide:138 design.math_reshape.rewrite[3].arithmetic.approx; tcg_d128_flat_tma_xpose_narrow:139 design.math_reshape.rewrite[3].arithmetic.approx; tcg_d128_flat_tma_xpose_wide:138 design.math_reshape.rewrite[3].arithmetic.approx; tcg_d128_fscan_cp_direct_narrow:136 design.math_reshape.rewrite[3].arithmetic.approx; tcg_d128_fscan_cp_direct_wide:136 design.math_reshape.rewrite[3].arithmetic.approx; tcg_d128_fscan_tma_direct_narrow:136 design.math_reshape.rewrite[3].arithmetic.approx; tcg_d128_fscan_tma_direct_wide:136 design.math_reshape.rewrite[3].arithmetic.approx; tcg_d128_fscan_tma_xpose_narrow:142 design.math_reshape.rewrite[3].arithmetic.approx; tcg_d128_segmented_cp_direct_narrow:159 design.math_reshape.rewrite[4].arithmetic.approx; tcg_d128_segmented_cp_direct_wide:158 design.math_reshape.rewrite[4].arithmetic.approx; tcg_d128_segmented_cp_xpose_narrow:144 design.math_reshape.rewrite[4].arithmetic.approx; tcg_d128_segmented_tma_direct_narrow:159 design.math_reshape.rewrite[4].arithmetic.approx; tcg_d128_segmented_tma_direct_wide:159 design.math_reshape.rewrite[4].arithmetic.approx; tcg_d128_segmented_tma_xpose_narrow:160 design.math_reshape.rewrite[4].arithmetic.approx; tcg_d128_segmented_tma_xpose_wide:144 design.math_reshape.rewrite[4].arithmetic.approx

E16 is expf2_bias<4> with FMHA_BIAS_MAGIC_FOLD=1 and FMHA_EXP_NO_KTINY=1; alpha uses CUDA expf. unknown: a reliable bound for this path and its probability-rounding effects is not supplied. The shared tolerance source records do not provide a design guarantee.

## t_f415708457526d58 | math.arithmetic | 251 chars | 12 declarations

tcg_d128_flat_cp_direct_narrow:165 design.math_reshape.rewrite[4].arithmetic.accumulate; tcg_d128_flat_cp_direct_wide:165 design.math_reshape.rewrite[4].arithmetic.accumulate; tcg_d128_flat_tma_direct_narrow:166 design.math_reshape.rewrite[4].arithmetic.accumulate; tcg_d128_flat_tma_direct_wide:166 design.math_reshape.rewrite[4].arithmetic.accumulate; tcg_d128_fscan_cp_direct_narrow:169 design.math_reshape.rewrite[4].arithmetic.accumulate; tcg_d128_fscan_cp_direct_wide:169 design.math_reshape.rewrite[4].arithmetic.accumulate; tcg_d128_fscan_tma_direct_narrow:169 design.math_reshape.rewrite[4].arithmetic.accumulate; tcg_d128_fscan_tma_direct_wide:169 design.math_reshape.rewrite[4].arithmetic.accumulate; tcg_d128_segmented_cp_direct_narrow:187 design.math_reshape.rewrite[5].arithmetic.accumulate; tcg_d128_segmented_cp_direct_wide:186 design.math_reshape.rewrite[5].arithmetic.accumulate; tcg_d128_segmented_tma_direct_narrow:187 design.math_reshape.rewrite[5].arithmetic.accumulate; tcg_d128_segmented_tma_direct_wide:187 design.math_reshape.rewrite[5].arithmetic.accumulate

PV uses the scores step's native block-scaled contraction with E4M3 Prob8 and unit E8M0 scales, and original E4M3 V with its key-axis scales. Register rescaling is a multiply and the later PV addition is separate, as selected by FMHA_RESCALE_LAYOUT=0.

## t_f62047c18a1edf45 | math.arithmetic | 70 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:199 design.math_reshape.rewrite[5].arithmetic.approx

weight uses CUDA expf; unknown: no end-to-end error bound is supplied.

## t_f7d2beaf835978e7 | math.arithmetic | 184 chars | 7 declarations

tcg_d256_flat_cp_narrow:172 design.math_reshape.rewrite[5].arithmetic.approx; tcg_d256_flat_cp_wide:186 design.math_reshape.rewrite[5].arithmetic.approx; tcg_d256_flat_tma_narrow:172 design.math_reshape.rewrite[5].arithmetic.approx; tcg_d256_flat_tma_wide:184 design.math_reshape.rewrite[5].arithmetic.approx; tcg_d256_segmented_cp_narrow:179 design.math_reshape.rewrite[6].arithmetic.approx; tcg_d256_segmented_tma_narrow:194 design.math_reshape.rewrite[6].arithmetic.approx; tcg_d256_segmented_tma_wide:179 design.math_reshape.rewrite[6].arithmetic.approx

Reciprocal and CUDA logf retain the source compilation path. unknown: no end-to-end error bound is supplied. The shared scoped acceptance records provide no numerical design guarantee.

## t_0193fd719e618f47 | math.ledger | 198 chars | 1 declarations

tc_fused:162 design.math_reshape.ledger.baseline

Direct Contract evaluation retaining full FP32 score/probability and E4M3 probability arrays; logical payload bytes are counted. Worktile counts below include padded rows and the visited key groups.

## t_0884f270b020118b | math.ledger | 196 chars | 4 declarations

tc_split_flat_narrow:240 design.math_reshape.ledger.baseline; tc_split_flat_wide:252 design.math_reshape.ledger.baseline; tc_split_segmented_narrow:275 design.math_reshape.ledger.baseline; tc_split_segmented_wide:300 design.math_reshape.ledger.baseline

Direct Contract evaluation with full FP32 score/probability and E4M3 probability arrays. Counts include actual padded rows and visited groups; footprint counts values written, not peak allocation.

## t_0a35c234897b744c | math.ledger | 40 chars | 6 declarations

fused_d128_cp:166 design.math_reshape.ledger.footprint[0].item; fused_d128_group:165 design.math_reshape.ledger.footprint[0].item; fused_d128_key:168 design.math_reshape.ledger.footprint[0].item; fused_d256_cp:162 design.math_reshape.ledger.footprint[0].item; fused_d256_tma:162 design.math_reshape.ledger.footprint[0].item; root:376 design.math_reshape.ledger.footprint[0].item

Full score and probability intermediates

## t_0d3420b8e503f3b1 | math.ledger | 102 chars | 3 declarations

tcg_d128_fscan_cp_xpose_narrow:206 design.math_reshape.ledger.footprint[0].change; tcg_d128_fscan_cp_xpose_wide:219 design.math_reshape.ledger.footprint[0].change; tcg_d128_fscan_tma_xpose_wide:220 design.math_reshape.ledger.footprint[0].change

9*R*Skv -> 4*R*(T+1+Ns*(D+2)) bytes; excludes untouched alignment and reserved scan/segment-max arrays

## t_0e307e294f906ab9 | math.ledger | 180 chars | 1 declarations

tcg_d128_fscan_tma_xpose_wide:226 design.math_reshape.ledger.depth[0].change

T -> max_s Count(Ns,s) group steps per numerator, with min(L(Ns,s),T) preceding scalar scan steps per split and a merge over Cg intervals of at most ceil_div(Ns,Cg) entries per row

## t_1243fb1305d42129 | math.ledger | 70 chars | 1 declarations

tc_fused:169 design.math_reshape.ledger.matmul[0].change

Per (b,kh,a), 2*G*min(Qph,Sq-q0)*Skv*D MACs -> 2*Br*KeyGroup*Ng*D MACs

## t_1938bd251ab465da | math.ledger | 6 chars | 126 declarations

fused_d128_cp:166 design.math_reshape.ledger.footprint[0].by[0]; fused_d128_cp:168 design.math_reshape.ledger.matmul[0].by[1]; fused_d128_cp:172 design.math_reshape.ledger.error[0].by[0]; fused_d128_group:165 design.math_reshape.ledger.footprint[0].by[0]; fused_d128_group:167 design.math_reshape.ledger.matmul[0].by[1]; fused_d128_group:171 design.math_reshape.ledger.error[0].by[0]; fused_d128_key:168 design.math_reshape.ledger.footprint[0].by[0]; fused_d128_key:170 design.math_reshape.ledger.matmul[0].by[1]; fused_d128_key:174 design.math_reshape.ledger.error[0].by[0]; fused_d256_cp:162 design.math_reshape.ledger.footprint[0].by[0]; fused_d256_cp:164 design.math_reshape.ledger.matmul[0].by[1]; fused_d256_cp:168 design.math_reshape.ledger.error[0].by[0]; fused_d256_tma:162 design.math_reshape.ledger.footprint[0].by[0]; fused_d256_tma:164 design.math_reshape.ledger.matmul[0].by[1]; fused_d256_tma:168 design.math_reshape.ledger.error[0].by[0]; root:376 design.math_reshape.ledger.footprint[0].by[0]; root:378 design.math_reshape.ledger.matmul[0].by[1]; root:382 design.math_reshape.ledger.error[0].by[0]; tc_fused:167 design.math_reshape.ledger.footprint[0].by[0]; tc_fused:169 design.math_reshape.ledger.matmul[0].by[1]; tc_fused:171 design.math_reshape.ledger.vector[0].by[0]; tc_fused:175 design.math_reshape.ledger.error[0].by[0]; tc_split_flat_narrow:250 design.math_reshape.ledger.matmul[0].by[1]; tc_split_flat_narrow:265 design.math_reshape.ledger.error[0].by[0]; tc_split_flat_wide:262 design.math_reshape.ledger.matmul[0].by[1]; tc_split_flat_wide:278 design.math_reshape.ledger.error[0].by[0]; tc_split_segmented_narrow:285 design.math_reshape.ledger.matmul[0].by[1]; tc_split_segmented_narrow:306 design.math_reshape.ledger.error[0].by[0]; tc_split_segmented_wide:313 design.math_reshape.ledger.matmul[0].by[1]; tc_split_segmented_wide:335 design.math_reshape.ledger.error[0].by[0]; tcg_d128_flat_cp_direct_narrow:193 design.math_reshape.ledger.footprint[0].by[0]; tcg_d128_flat_cp_direct_narrow:196 design.math_reshape.ledger.matmul[0].by[1]; tcg_d128_flat_cp_direct_narrow:202 design.math_reshape.ledger.error[0].by[0]; tcg_d128_flat_cp_direct_wide:205 design.math_reshape.ledger.footprint[0].by[0]; tcg_d128_flat_cp_direct_wide:208 design.math_reshape.ledger.matmul[0].by[1]; tcg_d128_flat_cp_direct_wide:214 design.math_reshape.ledger.error[0].by[0]; tcg_d128_flat_cp_xpose_narrow:200 design.math_reshape.ledger.footprint[0].by[0]; tcg_d128_flat_cp_xpose_narrow:203 design.math_reshape.ledger.matmul[0].by[1]; tcg_d128_flat_cp_xpose_narrow:209 design.math_reshape.ledger.error[0].by[0]; tcg_d128_flat_cp_xpose_wide:183 design.math_reshape.ledger.footprint[0].by[0]; tcg_d128_flat_cp_xpose_wide:186 design.math_reshape.ledger.matmul[0].by[1]; tcg_d128_flat_cp_xpose_wide:192 design.math_reshape.ledger.error[0].by[0]; tcg_d128_flat_tma_direct_narrow:194 design.math_reshape.ledger.footprint[0].by[0]; tcg_d128_flat_tma_direct_narrow:197 design.math_reshape.ledger.matmul[0].by[1]; tcg_d128_flat_tma_direct_narrow:203 design.math_reshape.ledger.error[0].by[0]; tcg_d128_flat_tma_direct_wide:205 design.math_reshape.ledger.footprint[0].by[0]; tcg_d128_flat_tma_direct_wide:209 design.math_reshape.ledger.matmul[0].by[1]; tcg_d128_flat_tma_direct_wide:215 design.math_reshape.ledger.error[0].by[0]; tcg_d128_flat_tma_xpose_narrow:197 design.math_reshape.ledger.footprint[0].by[0]; tcg_d128_flat_tma_xpose_narrow:200 design.math_reshape.ledger.matmul[0].by[1]; tcg_d128_flat_tma_xpose_narrow:206 design.math_reshape.ledger.error[0].by[0]; tcg_d128_flat_tma_xpose_wide:212 design.math_reshape.ledger.footprint[0].by[0]; tcg_d128_flat_tma_xpose_wide:215 design.math_reshape.ledger.matmul[0].by[1]; tcg_d128_flat_tma_xpose_wide:221 design.math_reshape.ledger.error[0].by[0]; tcg_d128_fscan_cp_direct_narrow:204 design.math_reshape.ledger.footprint[0].by[0]; tcg_d128_fscan_cp_direct_narrow:207 design.math_reshape.ledger.matmul[0].by[1]; tcg_d128_fscan_cp_direct_narrow:213 design.math_reshape.ledger.error[0].by[0]; tcg_d128_fscan_cp_direct_wide:219 design.math_reshape.ledger.footprint[0].by[0]; tcg_d128_fscan_cp_direct_wide:222 design.math_reshape.ledger.matmul[0].by[1]; tcg_d128_fscan_cp_direct_wide:227 design.math_reshape.ledger.error[0].by[0]; tcg_d128_fscan_cp_xpose_narrow:207 design.math_reshape.ledger.footprint[0].by[0]; tcg_d128_fscan_cp_xpose_narrow:210 design.math_reshape.ledger.matmul[0].by[1]; tcg_d128_fscan_cp_xpose_narrow:219 design.math_reshape.ledger.error[0].by[0]; tcg_d128_fscan_cp_xpose_wide:220 design.math_reshape.ledger.footprint[0].by[0]; tcg_d128_fscan_cp_xpose_wide:224 design.math_reshape.ledger.matmul[0].by[1]; tcg_d128_fscan_cp_xpose_wide:233 design.math_reshape.ledger.error[0].by[0]; tcg_d128_fscan_tma_direct_narrow:204 design.math_reshape.ledger.footprint[0].by[0]; tcg_d128_fscan_tma_direct_narrow:207 design.math_reshape.ledger.matmul[0].by[1]; tcg_d128_fscan_tma_direct_narrow:213 design.math_reshape.ledger.error[0].by[0]; tcg_d128_fscan_tma_direct_wide:220 design.math_reshape.ledger.footprint[0].by[0]; tcg_d128_fscan_tma_direct_wide:223 design.math_reshape.ledger.matmul[0].by[1]; tcg_d128_fscan_tma_direct_wide:229 design.math_reshape.ledger.error[0].by[0]; tcg_d128_fscan_tma_xpose_narrow:207 design.math_reshape.ledger.footprint[0].by[0]; tcg_d128_fscan_tma_xpose_narrow:210 design.math_reshape.ledger.matmul[0].by[1]; tcg_d128_fscan_tma_xpose_narrow:219 design.math_reshape.ledger.error[0].by[0]; tcg_d128_fscan_tma_xpose_wide:221 design.math_reshape.ledger.footprint[0].by[0]; tcg_d128_fscan_tma_xpose_wide:224 design.math_reshape.ledger.matmul[0].by[1]; tcg_d128_fscan_tma_xpose_wide:233 design.math_reshape.ledger.error[0].by[0]; tcg_d128_segmented_cp_direct_narrow:215 design.math_reshape.ledger.footprint[0].by[0]; tcg_d128_segmented_cp_direct_narrow:218 design.math_reshape.ledger.matmul[0].by[1]; tcg_d128_segmented_cp_direct_narrow:227 design.math_reshape.ledger.error[0].by[0]; tcg_d128_segmented_cp_direct_wide:228 design.math_reshape.ledger.footprint[0].by[0]; tcg_d128_segmented_cp_direct_wide:231 design.math_reshape.ledger.matmul[0].by[1]; tcg_d128_segmented_cp_direct_wide:240 design.math_reshape.ledger.error[0].by[0]; tcg_d128_segmented_cp_xpose_narrow:205 design.math_reshape.ledger.footprint[0].by[0]; tcg_d128_segmented_cp_xpose_narrow:208 design.math_reshape.ledger.matmul[0].by[1]; tcg_d128_segmented_cp_xpose_narrow:216 design.math_reshape.ledger.error[0].by[0]; tcg_d128_segmented_cp_xpose_wide:238 design.math_reshape.ledger.footprint[0].by[0]; tcg_d128_segmented_cp_xpose_wide:241 design.math_reshape.ledger.matmul[0].by[1]; tcg_d128_segmented_cp_xpose_wide:247 design.math_reshape.ledger.error[0].by[0]; tcg_d128_segmented_tma_direct_narrow:215 design.math_reshape.ledger.footprint[0].by[0]; tcg_d128_segmented_tma_direct_narrow:218 design.math_reshape.ledger.matmul[0].by[1]; tcg_d128_segmented_tma_direct_narrow:227 design.math_reshape.ledger.error[0].by[0]; tcg_d128_segmented_tma_direct_wide:230 design.math_reshape.ledger.footprint[0].by[0]; tcg_d128_segmented_tma_direct_wide:234 design.math_reshape.ledger.matmul[0].by[1]; tcg_d128_segmented_tma_direct_wide:243 design.math_reshape.ledger.error[0].by[0]; tcg_d128_segmented_tma_xpose_narrow:221 design.math_reshape.ledger.footprint[0].by[0]; tcg_d128_segmented_tma_xpose_narrow:224 design.math_reshape.ledger.matmul[0].by[1]; tcg_d128_segmented_tma_xpose_narrow:232 design.math_reshape.ledger.error[0].by[0]; tcg_d128_segmented_tma_xpose_wide:205 design.math_reshape.ledger.footprint[0].by[0]; tcg_d128_segmented_tma_xpose_wide:208 design.math_reshape.ledger.matmul[0].by[1]; tcg_d128_segmented_tma_xpose_wide:216 design.math_reshape.ledger.error[0].by[0]; tcg_d256_flat_cp_narrow:182 design.math_reshape.ledger.footprint[0].by[0]; tcg_d256_flat_cp_narrow:184 design.math_reshape.ledger.matmul[0].by[1]; tcg_d256_flat_cp_narrow:190 design.math_reshape.ledger.error[0].by[0]; tcg_d256_flat_cp_wide:196 design.math_reshape.ledger.footprint[0].by[0]; tcg_d256_flat_cp_wide:198 design.math_reshape.ledger.matmul[0].by[1]; tcg_d256_flat_cp_wide:204 design.math_reshape.ledger.error[0].by[0]; tcg_d256_flat_tma_narrow:182 design.math_reshape.ledger.footprint[0].by[0]; tcg_d256_flat_tma_narrow:184 design.math_reshape.ledger.matmul[0].by[1]; tcg_d256_flat_tma_narrow:190 design.math_reshape.ledger.error[0].by[0]; tcg_d256_flat_tma_wide:194 design.math_reshape.ledger.footprint[0].by[0]; tcg_d256_flat_tma_wide:196 design.math_reshape.ledger.matmul[0].by[1]; tcg_d256_flat_tma_wide:202 design.math_reshape.ledger.error[0].by[0]; tcg_d256_segmented_cp_narrow:189 design.math_reshape.ledger.footprint[0].by[0]; tcg_d256_segmented_cp_narrow:191 design.math_reshape.ledger.matmul[0].by[1]; tcg_d256_segmented_cp_narrow:198 design.math_reshape.ledger.error[0].by[0]; tcg_d256_segmented_cp_wide:235 design.math_reshape.ledger.footprint[0].by[0]; tcg_d256_segmented_cp_wide:237 design.math_reshape.ledger.matmul[0].by[1]; tcg_d256_segmented_cp_wide:244 design.math_reshape.ledger.error[0].by[0]; tcg_d256_segmented_tma_narrow:204 design.math_reshape.ledger.footprint[0].by[0]; tcg_d256_segmented_tma_narrow:206 design.math_reshape.ledger.matmul[0].by[1]; tcg_d256_segmented_tma_narrow:213 design.math_reshape.ledger.error[0].by[0]; tcg_d256_segmented_tma_wide:189 design.math_reshape.ledger.footprint[0].by[0]; tcg_d256_segmented_tma_wide:191 design.math_reshape.ledger.matmul[0].by[1]; tcg_d256_segmented_tma_wide:198 design.math_reshape.ledger.error[0].by[0]

scores

## t_1a54e8c27d594a19 | math.ledger | 91 chars | 1 declarations

tcg_d128_fscan_tma_xpose_narrow:215 design.math_reshape.ledger.vector[0].change

Per (b,kh,a), U*T -> Br*sum_s(min(L(Ns,s),T)+Count(Ns,s)) scalar updates, including padding

## t_1b1b72094643a041 | math.ledger | 229 chars | 1 declarations

tcg_d256_segmented_cp_wide:239 design.math_reshape.ledger.depth[0].change

Ng -> at most ceil_div(Ng,Ns) transitions per split plus at most ceil_div(Ns,SP)+SP ordered numerator additions per output; a separate maximum scan has at most T group transitions and ceil_div(Nseg-1,SegChunk) chunk tests per row

## t_1ce79973cbce6926 | math.ledger | 238 chars | 2 declarations

tc_split_segmented_narrow:292 design.math_reshape.ledger.traffic[1].change; tc_split_segmented_wide:320 design.math_reshape.ledger.traffic[1].change

Add 4*R*T bytes of segment-pass maximum reads, 4*R*Nseg bytes of summary writes, 4*R*(Nseg-1) bytes of summary reads and 4*sum_{r=0}^{R-1} WalkCount(r) bytes of replay maximum reads; the last term ranges from 4*R*Seg to 4*R*T in this node

## t_1d3b14328965483b | math.ledger | 200 chars | 1 declarations

tcg_d256_segmented_cp_wide:230 design.math_reshape.ledger.baseline

Direct Contract evaluation materializing full FP32 scores and probabilities and E4M3 probabilities; logical payload bytes. Product counts explicitly include this design's Br-row and key-group padding.

## t_22fb703e45ce4949 | math.ledger | 155 chars | 1 declarations

tc_fused:171 design.math_reshape.ledger.vector[0].change

0 -> Br*KeyGroup*(D/Block)*Ng FP32 multiplications and the same number of FMAs per worktile; KeyGroup*D*Ng V scale multiplications use BF16 instead of FP32

## t_24a5cc69132d6185 | math.ledger | 70 chars | 1 declarations

tcg_d128_segmented_tma_direct_wide:231 design.math_reshape.ledger.footprint[1].change

0 -> 4*MergeGroups*(D+4) bytes of retained GroupA/GroupB per row merge

## t_24afe6fc8571024c | math.ledger | 83 chars | 6 declarations

tcg_d128_flat_cp_direct_narrow:198 design.math_reshape.ledger.depth[0].change; tcg_d128_flat_cp_direct_wide:210 design.math_reshape.ledger.depth[0].change; tcg_d128_flat_cp_xpose_narrow:205 design.math_reshape.ledger.depth[0].change; tcg_d128_flat_cp_xpose_wide:188 design.math_reshape.ledger.depth[0].change; tcg_d128_flat_tma_direct_narrow:199 design.math_reshape.ledger.depth[0].change; tcg_d128_flat_tma_xpose_narrow:202 design.math_reshape.ledger.depth[0].change

T -> max_s Count(Ns,s) group steps; a T-step scan and Ns-entry merge remain per row

## t_2900adf462699d89 | math.ledger | 98 chars | 20 declarations

tcg_d128_flat_cp_direct_narrow:195 design.math_reshape.ledger.matmul[0].change; tcg_d128_flat_cp_direct_wide:207 design.math_reshape.ledger.matmul[0].change; tcg_d128_flat_cp_xpose_narrow:202 design.math_reshape.ledger.matmul[0].change; tcg_d128_flat_cp_xpose_wide:185 design.math_reshape.ledger.matmul[0].change; tcg_d128_flat_tma_direct_narrow:196 design.math_reshape.ledger.matmul[0].change; tcg_d128_flat_tma_direct_wide:208 design.math_reshape.ledger.matmul[0].change; tcg_d128_flat_tma_xpose_narrow:199 design.math_reshape.ledger.matmul[0].change; tcg_d128_flat_tma_xpose_wide:214 design.math_reshape.ledger.matmul[0].change; tcg_d128_fscan_cp_direct_narrow:206 design.math_reshape.ledger.matmul[0].change; tcg_d128_fscan_cp_direct_wide:221 design.math_reshape.ledger.matmul[0].change; tcg_d128_fscan_tma_direct_narrow:206 design.math_reshape.ledger.matmul[0].change; tcg_d128_fscan_tma_direct_wide:222 design.math_reshape.ledger.matmul[0].change; tcg_d128_fscan_tma_xpose_narrow:209 design.math_reshape.ledger.matmul[0].change; tcg_d128_segmented_cp_direct_narrow:217 design.math_reshape.ledger.matmul[0].change; tcg_d128_segmented_cp_direct_wide:230 design.math_reshape.ledger.matmul[0].change; tcg_d128_segmented_cp_xpose_narrow:207 design.math_reshape.ledger.matmul[0].change; tcg_d128_segmented_tma_direct_narrow:217 design.math_reshape.ledger.matmul[0].change; tcg_d128_segmented_tma_direct_wide:233 design.math_reshape.ledger.matmul[0].change; tcg_d128_segmented_tma_xpose_narrow:223 design.math_reshape.ledger.matmul[0].change; tcg_d128_segmented_tma_xpose_wide:207 design.math_reshape.ledger.matmul[0].change

Per (b,kh,a), 2*U*Skv*D -> 3*Br*KeyGroup*Span*D MACs including padded and masked boundary products

## t_29df2647a80b2179 | math.ledger | 18 chars | 3 declarations

tcg_d256_segmented_cp_narrow:195 design.math_reshape.ledger.traffic[0].item; tcg_d256_segmented_tma_narrow:210 design.math_reshape.ledger.traffic[0].item; tcg_d256_segmented_tma_wide:195 design.math_reshape.ledger.traffic[0].item

Scan intermediates

## t_2bbeed15082e32a4 | math.ledger | 129 chars | 4 declarations

tc_split_flat_narrow:253 design.math_reshape.ledger.traffic[0].change; tc_split_flat_wide:265 design.math_reshape.ledger.traffic[0].change; tc_split_segmented_narrow:288 design.math_reshape.ledger.traffic[0].change; tc_split_segmented_wide:316 design.math_reshape.ledger.traffic[0].change

Add 8*Ns*R*(D+1) bytes for one write/read of partials and Nm*Rc*(2*D+4) temporary output bytes with LSE, or Nm*Rc*2*D without LSE

## t_2c95714adc386bb7 | math.ledger | 20 chars | 23 declarations

fused_d128_cp:172 design.math_reshape.ledger.error[0].item; fused_d128_group:171 design.math_reshape.ledger.error[0].item; fused_d128_key:174 design.math_reshape.ledger.error[0].item; fused_d256_cp:168 design.math_reshape.ledger.error[0].item; fused_d256_tma:168 design.math_reshape.ledger.error[0].item; root:382 design.math_reshape.ledger.error[0].item; tc_fused:175 design.math_reshape.ledger.error[0].item; tc_split_flat_narrow:262 design.math_reshape.ledger.error[0].item; tc_split_flat_wide:275 design.math_reshape.ledger.error[0].item; tc_split_segmented_narrow:303 design.math_reshape.ledger.error[0].item; tc_split_segmented_wide:332 design.math_reshape.ledger.error[0].item; tcg_d128_fscan_cp_xpose_narrow:218 design.math_reshape.ledger.error[0].item; tcg_d128_fscan_cp_xpose_wide:232 design.math_reshape.ledger.error[0].item; tcg_d128_fscan_tma_xpose_wide:232 design.math_reshape.ledger.error[0].item; tcg_d128_segmented_cp_xpose_wide:246 design.math_reshape.ledger.error[0].item; tcg_d256_flat_cp_narrow:190 design.math_reshape.ledger.error[0].item; tcg_d256_flat_cp_wide:204 design.math_reshape.ledger.error[0].item; tcg_d256_flat_tma_narrow:190 design.math_reshape.ledger.error[0].item; tcg_d256_flat_tma_wide:202 design.math_reshape.ledger.error[0].item; tcg_d256_segmented_cp_narrow:198 design.math_reshape.ledger.error[0].item; tcg_d256_segmented_cp_wide:244 design.math_reshape.ledger.error[0].item; tcg_d256_segmented_tma_narrow:213 design.math_reshape.ledger.error[0].item; tcg_d256_segmented_tma_wide:198 design.math_reshape.ledger.error[0].item

Numerical departures

## t_2d25c7481343f26c | math.ledger | 67 chars | 7 declarations

tcg_d256_flat_cp_narrow:188 design.math_reshape.ledger.traffic[0].change; tcg_d256_flat_cp_wide:202 design.math_reshape.ledger.traffic[0].change; tcg_d256_flat_tma_narrow:188 design.math_reshape.ledger.traffic[0].change; tcg_d256_flat_tma_wide:200 design.math_reshape.ledger.traffic[0].change; tcg_d256_segmented_cp_narrow:196 design.math_reshape.ledger.traffic[1].change; tcg_d256_segmented_tma_narrow:211 design.math_reshape.ledger.traffic[1].change; tcg_d256_segmented_tma_wide:196 design.math_reshape.ledger.traffic[1].change

0 -> 8*Ns*R*(D+1) bytes for one logical write/read of FP32 partials

## t_2f426dc7b4106141 | math.ledger | 179 chars | 3 declarations

tcg_d256_segmented_cp_narrow:195 design.math_reshape.ledger.traffic[0].change; tcg_d256_segmented_tma_narrow:210 design.math_reshape.ledger.traffic[0].change; tcg_d256_segmented_tma_wide:195 design.math_reshape.ledger.traffic[0].change

0 -> 4*R*(GCount+2*Nseg-1) bytes for segment input reads, SegMax writes and SegMax[1:] reads; replay adds SegLen*R..GCount*R GroupMax reads and (GCount+1)*R state writes, all FP32

## t_2fecec26900fc8f0 | math.ledger | 160 chars | 1 declarations

tcg_d256_flat_cp_wide:200 design.math_reshape.ledger.depth[0].change

Ng -> at most ceil_div(Ng,Ns) steps per split plus at most ceil_div(Ns,Cg)+Cg serial numerator additions per output; a separate GCount-step maximum scan remains

## t_307b6089566da9f0 | math.ledger | 122 chars | 2 declarations

tcg_d128_fscan_cp_direct_narrow:209 design.math_reshape.ledger.depth[0].change; tcg_d128_fscan_tma_direct_narrow:209 design.math_reshape.ledger.depth[0].change

T -> max_s Count(Ns,s) numerator steps; each split also walks Tw(s) prefix maxima, with an Ns-entry weighted merge per row

## t_31e37a99757ffcac | math.ledger | 199 chars | 1 declarations

tcg_d128_fscan_cp_xpose_wide:232 design.math_reshape.ledger.error[0].change

Native block-scaled contractions, E16, denominator trees, separate per-group PV addition, split weighting and hierarchical merge replace the reference evaluation choices; no end-to-end bound asserted

## t_32d21c4a8e556199 | math.ledger | 80 chars | 3 declarations

tcg_d128_segmented_cp_xpose_narrow:213 design.math_reshape.ledger.vector[0].change; tcg_d128_segmented_tma_xpose_narrow:229 design.math_reshape.ledger.vector[0].change; tcg_d128_segmented_tma_xpose_wide:213 design.math_reshape.ledger.vector[0].change

0 -> R*Nseg*SegLen additional fmaxf evaluations, including final-segment padding

## t_33d891e2bfae4945 | math.ledger | 9 chars | 6 declarations

tc_split_segmented_narrow:281 design.math_reshape.ledger.footprint[0].by[0]; tc_split_segmented_narrow:290 design.math_reshape.ledger.traffic[0].by[0]; tc_split_segmented_narrow:306 design.math_reshape.ledger.error[0].by[1]; tc_split_segmented_wide:306 design.math_reshape.ledger.footprint[0].by[0]; tc_split_segmented_wide:318 design.math_reshape.ledger.traffic[0].by[0]; tc_split_segmented_wide:335 design.math_reshape.ledger.error[0].by[1]

max_table

## t_3424cb1346c1dd1e | math.ledger | 24 chars | 1 declarations

tc_split_segmented_wide:307 design.math_reshape.ledger.footprint[1].item

Grouped combine partials

## t_35ea402c88d64f67 | math.ledger | 26 chars | 1 declarations

tc_fused:171 design.math_reshape.ledger.vector[0].item

Post-dot scale corrections

## t_381de67206f0464e | math.ledger | 142 chars | 2 declarations

tcg_d128_segmented_cp_direct_narrow:220 design.math_reshape.ledger.depth[0].change; tcg_d128_segmented_tma_direct_narrow:220 design.math_reshape.ledger.depth[0].change

T -> max_s Count(Ns,s) group steps; scan retains WalkCount(r) ordered group updates, at most T, and combine retains Ns ordered entries per row

## t_3d3952755d3f49d1 | math.ledger | 139 chars | 1 declarations

tcg_d128_fscan_cp_xpose_narrow:212 design.math_reshape.ledger.depth[0].change

T -> max_s Count(Ns,s) group steps per numerator, with min(L(Ns,s),T) preceding scalar scan steps per split and an Ns-entry combine per row

## t_40eceb989643d5c5 | math.ledger | 139 chars | 1 declarations

tcg_d128_fscan_tma_xpose_narrow:212 design.math_reshape.ledger.depth[0].change

T -> max_s Count(Ns,s) numerator group steps, preceded by min(L(Ns,s),T) scalar prefix updates per split; an Ns-entry merge remains per row

## t_42eeb039ae888c6c | math.ledger | 186 chars | 2 declarations

tc_split_segmented_narrow:279 design.math_reshape.ledger.footprint[0].change; tc_split_segmented_wide:304 design.math_reshape.ledger.footprint[0].change

9*R*Skv bytes -> 4*T*(Rc+R)+4*Nseg*R+4*R+4*Ns*R*(D+1) bytes of written maxima, summaries, states, tails and partials; another 4*T*(R-Rc) bytes of incoming scratch supply uncovered maxima

## t_43b4e56c642bbdb6 | math.ledger | 158 chars | 1 declarations

tcg_d256_segmented_cp_wide:244 design.math_reshape.ledger.error[0].change

Native contractions, E1, two carried half denominators, split weighting and grouped partial reduction change reference rounding; no final error bound asserted

## t_441fe5e701221e59 | math.ledger | 254 chars | 1 declarations

tc_split_segmented_narrow:298 design.math_reshape.ledger.depth[0].change

T coupled group updates -> Seg ordered segment fmax operations, then Nc chunk decisions and WalkCount(r) serial maximum tests per row, followed by at most ceil_div(Ng(a),Ns) PV updates per interval and PaddedNs combine additions; all T scan stores remain

## t_4619a5fd54f77cb8 | math.ledger | 150 chars | 2 declarations

fused_d128_cp:166 design.math_reshape.ledger.footprint[0].change; root:376 design.math_reshape.ledger.footprint[0].change

9*B*H*Sq*Skv bytes -> none globally; simultaneous per-worktile Score and Prob8 capacity is 5*Br*KeyGroup bytes, plus 4*Br*(D+2) bytes of carried state

## t_4a0561d903f21123 | math.ledger | 4 chars | 92 declarations

tc_split_flat_narrow:246 design.math_reshape.ledger.footprint[0].by[0]; tc_split_flat_narrow:255 design.math_reshape.ledger.traffic[0].by[0]; tc_split_flat_narrow:260 design.math_reshape.ledger.depth[0].by[0]; tc_split_flat_narrow:265 design.math_reshape.ledger.error[0].by[1]; tc_split_flat_wide:258 design.math_reshape.ledger.footprint[0].by[0]; tc_split_flat_wide:267 design.math_reshape.ledger.traffic[0].by[0]; tc_split_flat_wide:273 design.math_reshape.ledger.depth[0].by[0]; tc_split_flat_wide:278 design.math_reshape.ledger.error[0].by[1]; tc_split_segmented_narrow:281 design.math_reshape.ledger.footprint[0].by[2]; tc_split_segmented_narrow:295 design.math_reshape.ledger.traffic[1].by[1]; tc_split_segmented_narrow:301 design.math_reshape.ledger.depth[0].by[1]; tc_split_segmented_narrow:306 design.math_reshape.ledger.error[0].by[3]; tc_split_segmented_wide:306 design.math_reshape.ledger.footprint[0].by[2]; tc_split_segmented_wide:323 design.math_reshape.ledger.traffic[1].by[1]; tc_split_segmented_wide:330 design.math_reshape.ledger.depth[0].by[1]; tc_split_segmented_wide:335 design.math_reshape.ledger.error[0].by[3]; tcg_d128_flat_cp_direct_narrow:193 design.math_reshape.ledger.footprint[0].by[1]; tcg_d128_flat_cp_direct_narrow:199 design.math_reshape.ledger.depth[0].by[0]; tcg_d128_flat_cp_direct_wide:205 design.math_reshape.ledger.footprint[0].by[1]; tcg_d128_flat_cp_direct_wide:211 design.math_reshape.ledger.depth[0].by[0]; tcg_d128_flat_cp_xpose_narrow:200 design.math_reshape.ledger.footprint[0].by[1]; tcg_d128_flat_cp_xpose_narrow:206 design.math_reshape.ledger.depth[0].by[0]; tcg_d128_flat_cp_xpose_wide:183 design.math_reshape.ledger.footprint[0].by[1]; tcg_d128_flat_cp_xpose_wide:189 design.math_reshape.ledger.depth[0].by[0]; tcg_d128_flat_tma_direct_narrow:194 design.math_reshape.ledger.footprint[0].by[1]; tcg_d128_flat_tma_direct_narrow:200 design.math_reshape.ledger.depth[0].by[0]; tcg_d128_flat_tma_direct_wide:205 design.math_reshape.ledger.footprint[0].by[1]; tcg_d128_flat_tma_direct_wide:212 design.math_reshape.ledger.depth[0].by[0]; tcg_d128_flat_tma_xpose_narrow:197 design.math_reshape.ledger.footprint[0].by[1]; tcg_d128_flat_tma_xpose_narrow:203 design.math_reshape.ledger.depth[0].by[0]; tcg_d128_flat_tma_xpose_wide:212 design.math_reshape.ledger.footprint[0].by[1]; tcg_d128_flat_tma_xpose_wide:218 design.math_reshape.ledger.depth[0].by[0]; tcg_d128_fscan_cp_direct_narrow:204 design.math_reshape.ledger.footprint[0].by[1]; tcg_d128_fscan_cp_direct_narrow:210 design.math_reshape.ledger.depth[0].by[0]; tcg_d128_fscan_cp_direct_wide:219 design.math_reshape.ledger.footprint[0].by[1]; tcg_d128_fscan_cp_direct_wide:224 design.math_reshape.ledger.depth[0].by[0]; tcg_d128_fscan_cp_xpose_narrow:207 design.math_reshape.ledger.footprint[0].by[1]; tcg_d128_fscan_cp_xpose_narrow:213 design.math_reshape.ledger.depth[0].by[0]; tcg_d128_fscan_cp_xpose_narrow:216 design.math_reshape.ledger.vector[0].by[0]; tcg_d128_fscan_cp_xpose_wide:220 design.math_reshape.ledger.footprint[0].by[1]; tcg_d128_fscan_cp_xpose_wide:227 design.math_reshape.ledger.depth[0].by[0]; tcg_d128_fscan_cp_xpose_wide:230 design.math_reshape.ledger.vector[0].by[0]; tcg_d128_fscan_tma_direct_narrow:204 design.math_reshape.ledger.footprint[0].by[1]; tcg_d128_fscan_tma_direct_narrow:210 design.math_reshape.ledger.depth[0].by[0]; tcg_d128_fscan_tma_direct_wide:220 design.math_reshape.ledger.footprint[0].by[1]; tcg_d128_fscan_tma_direct_wide:226 design.math_reshape.ledger.depth[0].by[0]; tcg_d128_fscan_tma_xpose_narrow:207 design.math_reshape.ledger.footprint[0].by[1]; tcg_d128_fscan_tma_xpose_narrow:213 design.math_reshape.ledger.depth[0].by[0]; tcg_d128_fscan_tma_xpose_narrow:216 design.math_reshape.ledger.vector[0].by[0]; tcg_d128_fscan_tma_xpose_wide:221 design.math_reshape.ledger.footprint[0].by[1]; tcg_d128_fscan_tma_xpose_wide:227 design.math_reshape.ledger.depth[0].by[0]; tcg_d128_fscan_tma_xpose_wide:230 design.math_reshape.ledger.vector[0].by[0]; tcg_d128_segmented_cp_direct_narrow:215 design.math_reshape.ledger.footprint[0].by[2]; tcg_d128_segmented_cp_direct_narrow:221 design.math_reshape.ledger.depth[0].by[0]; tcg_d128_segmented_cp_direct_narrow:224 design.math_reshape.ledger.vector[0].by[1]; tcg_d128_segmented_cp_direct_wide:228 design.math_reshape.ledger.footprint[0].by[2]; tcg_d128_segmented_cp_direct_wide:234 design.math_reshape.ledger.depth[0].by[0]; tcg_d128_segmented_cp_direct_wide:237 design.math_reshape.ledger.vector[0].by[1]; tcg_d128_segmented_cp_xpose_narrow:205 design.math_reshape.ledger.footprint[0].by[2]; tcg_d128_segmented_cp_xpose_narrow:211 design.math_reshape.ledger.depth[0].by[0]; tcg_d128_segmented_cp_xpose_wide:238 design.math_reshape.ledger.footprint[0].by[2]; tcg_d128_segmented_cp_xpose_wide:244 design.math_reshape.ledger.depth[0].by[1]; tcg_d128_segmented_tma_direct_narrow:215 design.math_reshape.ledger.footprint[0].by[2]; tcg_d128_segmented_tma_direct_narrow:221 design.math_reshape.ledger.depth[0].by[0]; tcg_d128_segmented_tma_direct_narrow:224 design.math_reshape.ledger.vector[0].by[1]; tcg_d128_segmented_tma_direct_wide:230 design.math_reshape.ledger.footprint[0].by[2]; tcg_d128_segmented_tma_direct_wide:237 design.math_reshape.ledger.depth[0].by[0]; tcg_d128_segmented_tma_direct_wide:240 design.math_reshape.ledger.vector[0].by[1]; tcg_d128_segmented_tma_xpose_narrow:221 design.math_reshape.ledger.footprint[0].by[2]; tcg_d128_segmented_tma_xpose_narrow:227 design.math_reshape.ledger.depth[0].by[0]; tcg_d128_segmented_tma_xpose_wide:205 design.math_reshape.ledger.footprint[0].by[2]; tcg_d128_segmented_tma_xpose_wide:211 design.math_reshape.ledger.depth[0].by[0]; tcg_d256_flat_cp_narrow:182 design.math_reshape.ledger.footprint[0].by[1]; tcg_d256_flat_cp_narrow:186 design.math_reshape.ledger.depth[0].by[0]; tcg_d256_flat_cp_wide:196 design.math_reshape.ledger.footprint[0].by[1]; tcg_d256_flat_cp_wide:200 design.math_reshape.ledger.depth[0].by[0]; tcg_d256_flat_tma_narrow:182 design.math_reshape.ledger.footprint[0].by[1]; tcg_d256_flat_tma_narrow:186 design.math_reshape.ledger.depth[0].by[0]; tcg_d256_flat_tma_wide:194 design.math_reshape.ledger.footprint[0].by[1]; tcg_d256_flat_tma_wide:198 design.math_reshape.ledger.depth[0].by[0]; tcg_d256_segmented_cp_narrow:189 design.math_reshape.ledger.footprint[0].by[2]; tcg_d256_segmented_cp_narrow:193 design.math_reshape.ledger.depth[0].by[0]; tcg_d256_segmented_cp_narrow:195 design.math_reshape.ledger.traffic[0].by[1]; tcg_d256_segmented_cp_wide:235 design.math_reshape.ledger.footprint[0].by[2]; tcg_d256_segmented_cp_wide:239 design.math_reshape.ledger.depth[0].by[0]; tcg_d256_segmented_cp_wide:241 design.math_reshape.ledger.traffic[0].by[1]; tcg_d256_segmented_tma_narrow:204 design.math_reshape.ledger.footprint[0].by[2]; tcg_d256_segmented_tma_narrow:208 design.math_reshape.ledger.depth[0].by[0]; tcg_d256_segmented_tma_narrow:210 design.math_reshape.ledger.traffic[0].by[1]; tcg_d256_segmented_tma_wide:189 design.math_reshape.ledger.footprint[0].by[2]; tcg_d256_segmented_tma_wide:193 design.math_reshape.ledger.depth[0].by[0]; tcg_d256_segmented_tma_wide:195 design.math_reshape.ledger.traffic[0].by[1]

scan

## t_4d4955e5a942ba91 | math.ledger | 12 chars | 22 declarations

fused_d128_cp:166 design.math_reshape.ledger.footprint[0].by[2]; fused_d128_cp:168 design.math_reshape.ledger.matmul[0].by[2]; fused_d128_cp:172 design.math_reshape.ledger.error[0].by[2]; fused_d128_group:165 design.math_reshape.ledger.footprint[0].by[2]; fused_d128_group:167 design.math_reshape.ledger.matmul[0].by[2]; fused_d128_group:171 design.math_reshape.ledger.error[0].by[2]; fused_d128_key:168 design.math_reshape.ledger.footprint[0].by[2]; fused_d128_key:170 design.math_reshape.ledger.matmul[0].by[2]; fused_d128_key:174 design.math_reshape.ledger.error[0].by[2]; fused_d256_cp:162 design.math_reshape.ledger.footprint[0].by[2]; fused_d256_cp:164 design.math_reshape.ledger.matmul[0].by[2]; fused_d256_cp:168 design.math_reshape.ledger.error[0].by[2]; fused_d256_tma:162 design.math_reshape.ledger.footprint[0].by[2]; fused_d256_tma:164 design.math_reshape.ledger.matmul[0].by[2]; fused_d256_tma:168 design.math_reshape.ledger.error[0].by[2]; root:376 design.math_reshape.ledger.footprint[0].by[2]; root:378 design.math_reshape.ledger.matmul[0].by[2]; root:382 design.math_reshape.ledger.error[0].by[2]; tc_fused:167 design.math_reshape.ledger.footprint[0].by[2]; tc_fused:169 design.math_reshape.ledger.matmul[0].by[2]; tc_fused:171 design.math_reshape.ledger.vector[0].by[1]; tc_fused:175 design.math_reshape.ledger.error[0].by[2]

value_output

## t_4e3816d2d9cca91c | math.ledger | 23 chars | 4 declarations

tc_split_flat_narrow:248 design.math_reshape.ledger.matmul[0].item; tc_split_flat_wide:260 design.math_reshape.ledger.matmul[0].item; tc_split_segmented_narrow:283 design.math_reshape.ledger.matmul[0].item; tc_split_segmented_wide:311 design.math_reshape.ledger.matmul[0].item

QK recomputation and PV

## t_4ef65f6674620cff | math.ledger | 175 chars | 1 declarations

tcg_d128_flat_tma_direct_wide:211 design.math_reshape.ledger.depth[0].change

T -> max_s Count(Ns,s) group steps; a T-step scan and Ns-entry merge remain per row, with numerator merge chains of at most ceil_div(Ns,MergeGroups) then MergeGroups additions

## t_4faaff9c92641b8e | math.ledger | 15 chars | 1 declarations

tcg_d128_fscan_cp_xpose_wide:221 design.math_reshape.ledger.footprint[1].item

Merge subtotals

## t_500a4366e0ba54f9 | math.ledger | 61 chars | 1 declarations

tc_fused:173 design.math_reshape.ledger.depth[0].change

ceil_div(Skv,KeyGroup) -> Ng ordered transitions per worktile

## t_50c310c1907fc0c1 | math.ledger | 93 chars | 1 declarations

tcg_d256_segmented_cp_wide:237 design.math_reshape.ledger.matmul[0].change

Per (b,kh,a), 2*G*min(Qph,Sq-q0)*Skv*D -> 3*Br*KeyGroup*Ng*D MACs; QK executes twice, PV once

## t_5123003fe8f7df70 | math.ledger | 22 chars | 4 declarations

tcg_d128_fscan_cp_xpose_narrow:215 design.math_reshape.ledger.vector[0].item; tcg_d128_fscan_cp_xpose_wide:229 design.math_reshape.ledger.vector[0].item; tcg_d128_fscan_tma_xpose_narrow:215 design.math_reshape.ledger.vector[0].item; tcg_d128_fscan_tma_xpose_wide:229 design.math_reshape.ledger.vector[0].item

Delayed-max recurrence

## t_51689b843165bdaf | math.ledger | 97 chars | 1 declarations

tcg_d128_fscan_cp_xpose_wide:221 design.math_reshape.ledger.footprint[1].change

0 -> 4*MergeGroups*(D+4) bytes per row for NumGroup and DenGroup retained through final reduction

## t_5203d55aada741a2 | math.ledger | 11 chars | 24 declarations

tcg_d128_segmented_cp_direct_narrow:215 design.math_reshape.ledger.footprint[0].by[1]; tcg_d128_segmented_cp_direct_narrow:224 design.math_reshape.ledger.vector[0].by[0]; tcg_d128_segmented_cp_direct_wide:228 design.math_reshape.ledger.footprint[0].by[1]; tcg_d128_segmented_cp_direct_wide:237 design.math_reshape.ledger.vector[0].by[0]; tcg_d128_segmented_cp_xpose_narrow:205 design.math_reshape.ledger.footprint[0].by[1]; tcg_d128_segmented_cp_xpose_narrow:213 design.math_reshape.ledger.vector[0].by[0]; tcg_d128_segmented_cp_xpose_wide:238 design.math_reshape.ledger.footprint[0].by[1]; tcg_d128_segmented_cp_xpose_wide:244 design.math_reshape.ledger.depth[0].by[0]; tcg_d128_segmented_tma_direct_narrow:215 design.math_reshape.ledger.footprint[0].by[1]; tcg_d128_segmented_tma_direct_narrow:224 design.math_reshape.ledger.vector[0].by[0]; tcg_d128_segmented_tma_direct_wide:230 design.math_reshape.ledger.footprint[0].by[1]; tcg_d128_segmented_tma_direct_wide:240 design.math_reshape.ledger.vector[0].by[0]; tcg_d128_segmented_tma_xpose_narrow:221 design.math_reshape.ledger.footprint[0].by[1]; tcg_d128_segmented_tma_xpose_narrow:229 design.math_reshape.ledger.vector[0].by[0]; tcg_d128_segmented_tma_xpose_wide:205 design.math_reshape.ledger.footprint[0].by[1]; tcg_d128_segmented_tma_xpose_wide:213 design.math_reshape.ledger.vector[0].by[0]; tcg_d256_segmented_cp_narrow:189 design.math_reshape.ledger.footprint[0].by[1]; tcg_d256_segmented_cp_narrow:195 design.math_reshape.ledger.traffic[0].by[0]; tcg_d256_segmented_cp_wide:235 design.math_reshape.ledger.footprint[0].by[1]; tcg_d256_segmented_cp_wide:241 design.math_reshape.ledger.traffic[0].by[0]; tcg_d256_segmented_tma_narrow:204 design.math_reshape.ledger.footprint[0].by[1]; tcg_d256_segmented_tma_narrow:210 design.math_reshape.ledger.traffic[0].by[0]; tcg_d256_segmented_tma_wide:189 design.math_reshape.ledger.footprint[0].by[1]; tcg_d256_segmented_tma_wide:195 design.math_reshape.ledger.traffic[0].by[0]

segment_max

## t_54a444fa8ffe681f | math.ledger | 114 chars | 4 declarations

tcg_d128_fscan_cp_direct_narrow:203 design.math_reshape.ledger.footprint[0].change; tcg_d128_fscan_cp_direct_wide:218 design.math_reshape.ledger.footprint[0].change; tcg_d128_fscan_tma_direct_narrow:203 design.math_reshape.ledger.footprint[0].change; tcg_d128_fscan_tma_direct_wide:219 design.math_reshape.ledger.footprint[0].change

9*R*Skv -> 4*R*(T+1+Ns*(D+2)) bytes, excluding unused segment-max, per-group scan-state and alignment reservations

## t_557952c8c1c791cf | math.ledger | 163 chars | 7 declarations

tcg_d128_flat_cp_direct_wide:213 design.math_reshape.ledger.error[0].change; tcg_d128_flat_tma_direct_wide:214 design.math_reshape.ledger.error[0].change; tcg_d128_flat_tma_xpose_wide:220 design.math_reshape.ledger.error[0].change; tcg_d128_fscan_cp_direct_wide:226 design.math_reshape.ledger.error[0].change; tcg_d128_fscan_tma_direct_wide:228 design.math_reshape.ledger.error[0].change; tcg_d128_segmented_cp_direct_wide:239 design.math_reshape.ledger.error[0].change; tcg_d128_segmented_tma_direct_wide:242 design.math_reshape.ledger.error[0].change

Native contractions, E16, denominator trees, fresh per-group PV, split weights and grouped merge replace reference evaluation choices; no end-to-end bound asserted

## t_5a51d979567c3434 | math.ledger | 167 chars | 1 declarations

tcg_d256_flat_tma_wide:202 design.math_reshape.ledger.error[0].change

Native contractions, E1, separate half denominators, split weighting, grouped numerator addition and a denominator tree change reference rounding; no final error bound

## t_5aa1f52face6a38e | math.ledger | 196 chars | 1 declarations

tcg_d128_fscan_tma_direct_wide:225 design.math_reshape.ledger.depth[0].change

T -> max_s Count(Ns,s) numerator steps; each split also walks Tw(s) prefix maxima; the merge adds at most max(8,ceil_div(Ns,MergeGroups)) local numerator terms then MergeGroups final terms per row

## t_5e12483131f28834 | math.ledger | 82 chars | 3 declarations

tcg_d256_segmented_cp_narrow:189 design.math_reshape.ledger.footprint[0].change; tcg_d256_segmented_tma_narrow:204 design.math_reshape.ledger.footprint[0].change; tcg_d256_segmented_tma_wide:189 design.math_reshape.ledger.footprint[0].change

9*R*Skv -> 4*R*(2*GCount+Nseg+1+Ns*(D+1)) bytes; no global scores or probabilities

## t_5e20a27f5ea8c141 | math.ledger | 97 chars | 3 declarations

tcg_d128_fscan_cp_xpose_narrow:215 design.math_reshape.ledger.vector[0].change; tcg_d128_fscan_cp_xpose_wide:229 design.math_reshape.ledger.vector[0].change; tcg_d128_fscan_tma_xpose_wide:229 design.math_reshape.ledger.vector[0].change

Per (b,kh,a), U*T -> Br*sum_s (min(L(Ns,s),T)+Count(Ns,s)) scalar scan updates, including padding

## t_5e26ee6bb06a0552 | math.ledger | 14 chars | 2 declarations

tcg_d128_flat_tma_direct_wide:206 design.math_reshape.ledger.footprint[1].item; tcg_d128_segmented_tma_direct_wide:231 design.math_reshape.ledger.footprint[1].item

Merge partials

## t_62291961059fa77b | math.ledger | 87 chars | 7 declarations

tcg_d256_flat_cp_narrow:184 design.math_reshape.ledger.matmul[0].change; tcg_d256_flat_cp_wide:198 design.math_reshape.ledger.matmul[0].change; tcg_d256_flat_tma_narrow:184 design.math_reshape.ledger.matmul[0].change; tcg_d256_flat_tma_wide:196 design.math_reshape.ledger.matmul[0].change; tcg_d256_segmented_cp_narrow:191 design.math_reshape.ledger.matmul[0].change; tcg_d256_segmented_tma_narrow:206 design.math_reshape.ledger.matmul[0].change; tcg_d256_segmented_tma_wide:191 design.math_reshape.ledger.matmul[0].change

Per (b,kh,a), 2*G*min(Qph,Sq-q0)*Skv*D -> 3*Br*KeyGroup*Ng*D MACs; QK twice and PV once

## t_6404b8bd40cc8139 | math.ledger | 206 chars | 20 declarations

tcg_d128_flat_cp_direct_narrow:189 design.math_reshape.ledger.baseline; tcg_d128_flat_cp_direct_wide:201 design.math_reshape.ledger.baseline; tcg_d128_flat_cp_xpose_narrow:196 design.math_reshape.ledger.baseline; tcg_d128_flat_cp_xpose_wide:179 design.math_reshape.ledger.baseline; tcg_d128_flat_tma_direct_narrow:190 design.math_reshape.ledger.baseline; tcg_d128_flat_tma_direct_wide:201 design.math_reshape.ledger.baseline; tcg_d128_flat_tma_xpose_narrow:193 design.math_reshape.ledger.baseline; tcg_d128_flat_tma_xpose_wide:208 design.math_reshape.ledger.baseline; tcg_d128_fscan_cp_direct_narrow:200 design.math_reshape.ledger.baseline; tcg_d128_fscan_cp_direct_wide:215 design.math_reshape.ledger.baseline; tcg_d128_fscan_tma_direct_narrow:200 design.math_reshape.ledger.baseline; tcg_d128_fscan_tma_direct_wide:216 design.math_reshape.ledger.baseline; tcg_d128_fscan_tma_xpose_narrow:203 design.math_reshape.ledger.baseline; tcg_d128_segmented_cp_direct_narrow:211 design.math_reshape.ledger.baseline; tcg_d128_segmented_cp_direct_wide:224 design.math_reshape.ledger.baseline; tcg_d128_segmented_cp_xpose_narrow:201 design.math_reshape.ledger.baseline; tcg_d128_segmented_tma_direct_narrow:211 design.math_reshape.ledger.baseline; tcg_d128_segmented_tma_direct_wide:226 design.math_reshape.ledger.baseline; tcg_d128_segmented_tma_xpose_narrow:217 design.math_reshape.ledger.baseline; tcg_d128_segmented_tma_xpose_wide:201 design.math_reshape.ledger.baseline

Direct Contract evaluation with full FP32 scores/probabilities and E4M3 probabilities materialized; logical payload bytes, including the source padding and useful-group traversal in the worktile comparison.

## t_6ec51664043262cf | math.ledger | 76 chars | 7 declarations

tcg_d128_segmented_cp_direct_narrow:214 design.math_reshape.ledger.footprint[0].change; tcg_d128_segmented_cp_direct_wide:227 design.math_reshape.ledger.footprint[0].change; tcg_d128_segmented_cp_xpose_narrow:204 design.math_reshape.ledger.footprint[0].change; tcg_d128_segmented_tma_direct_narrow:214 design.math_reshape.ledger.footprint[0].change; tcg_d128_segmented_tma_direct_wide:229 design.math_reshape.ledger.footprint[0].change; tcg_d128_segmented_tma_xpose_narrow:220 design.math_reshape.ledger.footprint[0].change; tcg_d128_segmented_tma_xpose_wide:204 design.math_reshape.ledger.footprint[0].change

9*R*Skv -> 4*R*(2*T+Nseg+1+Ns*(D+1)) bytes, excluding alignment reservations

## t_6edf48098570bb81 | math.ledger | 9 chars | 61 declarations

fused_d128_cp:166 design.math_reshape.ledger.footprint[0].by[1]; fused_d128_cp:170 design.math_reshape.ledger.depth[0].by[1]; fused_d128_cp:172 design.math_reshape.ledger.error[0].by[1]; fused_d128_group:165 design.math_reshape.ledger.footprint[0].by[1]; fused_d128_group:169 design.math_reshape.ledger.depth[0].by[1]; fused_d128_group:171 design.math_reshape.ledger.error[0].by[1]; fused_d128_key:168 design.math_reshape.ledger.footprint[0].by[1]; fused_d128_key:172 design.math_reshape.ledger.depth[0].by[1]; fused_d128_key:174 design.math_reshape.ledger.error[0].by[1]; fused_d256_cp:162 design.math_reshape.ledger.footprint[0].by[1]; fused_d256_cp:166 design.math_reshape.ledger.depth[0].by[1]; fused_d256_cp:168 design.math_reshape.ledger.error[0].by[1]; fused_d256_tma:162 design.math_reshape.ledger.footprint[0].by[1]; fused_d256_tma:166 design.math_reshape.ledger.depth[0].by[1]; fused_d256_tma:168 design.math_reshape.ledger.error[0].by[1]; root:376 design.math_reshape.ledger.footprint[0].by[1]; root:380 design.math_reshape.ledger.depth[0].by[1]; root:382 design.math_reshape.ledger.error[0].by[1]; tc_fused:167 design.math_reshape.ledger.footprint[0].by[1]; tc_fused:173 design.math_reshape.ledger.depth[0].by[1]; tc_fused:175 design.math_reshape.ledger.error[0].by[1]; tc_split_flat_narrow:260 design.math_reshape.ledger.depth[0].by[1]; tc_split_flat_narrow:265 design.math_reshape.ledger.error[0].by[2]; tc_split_flat_wide:273 design.math_reshape.ledger.depth[0].by[1]; tc_split_flat_wide:278 design.math_reshape.ledger.error[0].by[2]; tc_split_segmented_narrow:301 design.math_reshape.ledger.depth[0].by[2]; tc_split_segmented_narrow:306 design.math_reshape.ledger.error[0].by[4]; tc_split_segmented_wide:330 design.math_reshape.ledger.depth[0].by[2]; tc_split_segmented_wide:335 design.math_reshape.ledger.error[0].by[4]; tcg_d128_flat_cp_direct_narrow:202 design.math_reshape.ledger.error[0].by[1]; tcg_d128_flat_cp_direct_wide:214 design.math_reshape.ledger.error[0].by[1]; tcg_d128_flat_cp_xpose_narrow:209 design.math_reshape.ledger.error[0].by[1]; tcg_d128_flat_cp_xpose_wide:192 design.math_reshape.ledger.error[0].by[1]; tcg_d128_flat_tma_direct_narrow:203 design.math_reshape.ledger.error[0].by[1]; tcg_d128_flat_tma_direct_wide:215 design.math_reshape.ledger.error[0].by[1]; tcg_d128_flat_tma_xpose_narrow:206 design.math_reshape.ledger.error[0].by[1]; tcg_d128_flat_tma_xpose_wide:221 design.math_reshape.ledger.error[0].by[1]; tcg_d128_fscan_cp_direct_narrow:213 design.math_reshape.ledger.error[0].by[1]; tcg_d128_fscan_cp_direct_wide:227 design.math_reshape.ledger.error[0].by[1]; tcg_d128_fscan_cp_xpose_narrow:219 design.math_reshape.ledger.error[0].by[1]; tcg_d128_fscan_cp_xpose_wide:233 design.math_reshape.ledger.error[0].by[1]; tcg_d128_fscan_tma_direct_narrow:213 design.math_reshape.ledger.error[0].by[1]; tcg_d128_fscan_tma_direct_wide:229 design.math_reshape.ledger.error[0].by[1]; tcg_d128_fscan_tma_xpose_narrow:219 design.math_reshape.ledger.error[0].by[1]; tcg_d128_fscan_tma_xpose_wide:233 design.math_reshape.ledger.error[0].by[1]; tcg_d128_segmented_cp_direct_narrow:227 design.math_reshape.ledger.error[0].by[1]; tcg_d128_segmented_cp_direct_wide:240 design.math_reshape.ledger.error[0].by[1]; tcg_d128_segmented_cp_xpose_narrow:216 design.math_reshape.ledger.error[0].by[1]; tcg_d128_segmented_cp_xpose_wide:247 design.math_reshape.ledger.error[0].by[1]; tcg_d128_segmented_tma_direct_narrow:227 design.math_reshape.ledger.error[0].by[1]; tcg_d128_segmented_tma_direct_wide:243 design.math_reshape.ledger.error[0].by[1]; tcg_d128_segmented_tma_xpose_narrow:232 design.math_reshape.ledger.error[0].by[1]; tcg_d128_segmented_tma_xpose_wide:216 design.math_reshape.ledger.error[0].by[1]; tcg_d256_flat_cp_narrow:190 design.math_reshape.ledger.error[0].by[1]; tcg_d256_flat_cp_wide:204 design.math_reshape.ledger.error[0].by[1]; tcg_d256_flat_tma_narrow:190 design.math_reshape.ledger.error[0].by[1]; tcg_d256_flat_tma_wide:202 design.math_reshape.ledger.error[0].by[1]; tcg_d256_segmented_cp_narrow:198 design.math_reshape.ledger.error[0].by[1]; tcg_d256_segmented_cp_wide:244 design.math_reshape.ledger.error[0].by[1]; tcg_d256_segmented_tma_narrow:213 design.math_reshape.ledger.error[0].by[1]; tcg_d256_segmented_tma_wide:198 design.math_reshape.ledger.error[0].by[1]

normalize

## t_7143201757974d20 | math.ledger | 91 chars | 4 declarations

tc_split_flat_narrow:249 design.math_reshape.ledger.matmul[0].change; tc_split_flat_wide:261 design.math_reshape.ledger.matmul[0].change; tc_split_segmented_narrow:284 design.math_reshape.ledger.matmul[0].change; tc_split_segmented_wide:312 design.math_reshape.ledger.matmul[0].change

2*R*Skv*D MACs -> B*KH*Br*KeyGroup*D*(sum_{a=0}^{Ag-1} Ng(a)+2*sum_{a=0}^{Aq-1} Ng(a)) MACs

## t_71e9cbe6dca4bd97 | math.ledger | 182 chars | 3 declarations

tcg_d256_segmented_cp_narrow:193 design.math_reshape.ledger.depth[0].change; tcg_d256_segmented_tma_narrow:208 design.math_reshape.ledger.depth[0].change; tcg_d256_segmented_tma_wide:193 design.math_reshape.ledger.depth[0].change

Ng -> at most ceil_div(Ng,Ns) steps per split plus Ns ordered combine additions; the separate row scan has at most GCount group transitions plus ceil_div(Nseg-1,SegChunk) chunk tests

## t_72680623bceefd0d | math.ledger | 8 chars | 136 declarations

tcg_d128_flat_cp_direct_narrow:193 design.math_reshape.ledger.footprint[0].by[2]; tcg_d128_flat_cp_direct_narrow:196 design.math_reshape.ledger.matmul[0].by[2]; tcg_d128_flat_cp_direct_narrow:199 design.math_reshape.ledger.depth[0].by[1]; tcg_d128_flat_cp_direct_narrow:202 design.math_reshape.ledger.error[0].by[2]; tcg_d128_flat_cp_direct_wide:205 design.math_reshape.ledger.footprint[0].by[2]; tcg_d128_flat_cp_direct_wide:208 design.math_reshape.ledger.matmul[0].by[2]; tcg_d128_flat_cp_direct_wide:211 design.math_reshape.ledger.depth[0].by[1]; tcg_d128_flat_cp_direct_wide:214 design.math_reshape.ledger.error[0].by[2]; tcg_d128_flat_cp_xpose_narrow:200 design.math_reshape.ledger.footprint[0].by[2]; tcg_d128_flat_cp_xpose_narrow:203 design.math_reshape.ledger.matmul[0].by[2]; tcg_d128_flat_cp_xpose_narrow:206 design.math_reshape.ledger.depth[0].by[1]; tcg_d128_flat_cp_xpose_narrow:209 design.math_reshape.ledger.error[0].by[2]; tcg_d128_flat_cp_xpose_wide:183 design.math_reshape.ledger.footprint[0].by[2]; tcg_d128_flat_cp_xpose_wide:186 design.math_reshape.ledger.matmul[0].by[2]; tcg_d128_flat_cp_xpose_wide:189 design.math_reshape.ledger.depth[0].by[1]; tcg_d128_flat_cp_xpose_wide:192 design.math_reshape.ledger.error[0].by[2]; tcg_d128_flat_tma_direct_narrow:194 design.math_reshape.ledger.footprint[0].by[2]; tcg_d128_flat_tma_direct_narrow:197 design.math_reshape.ledger.matmul[0].by[2]; tcg_d128_flat_tma_direct_narrow:200 design.math_reshape.ledger.depth[0].by[1]; tcg_d128_flat_tma_direct_narrow:203 design.math_reshape.ledger.error[0].by[2]; tcg_d128_flat_tma_direct_wide:205 design.math_reshape.ledger.footprint[0].by[2]; tcg_d128_flat_tma_direct_wide:209 design.math_reshape.ledger.matmul[0].by[2]; tcg_d128_flat_tma_direct_wide:212 design.math_reshape.ledger.depth[0].by[1]; tcg_d128_flat_tma_direct_wide:215 design.math_reshape.ledger.error[0].by[2]; tcg_d128_flat_tma_xpose_narrow:197 design.math_reshape.ledger.footprint[0].by[2]; tcg_d128_flat_tma_xpose_narrow:200 design.math_reshape.ledger.matmul[0].by[2]; tcg_d128_flat_tma_xpose_narrow:203 design.math_reshape.ledger.depth[0].by[1]; tcg_d128_flat_tma_xpose_narrow:206 design.math_reshape.ledger.error[0].by[2]; tcg_d128_flat_tma_xpose_wide:212 design.math_reshape.ledger.footprint[0].by[2]; tcg_d128_flat_tma_xpose_wide:215 design.math_reshape.ledger.matmul[0].by[2]; tcg_d128_flat_tma_xpose_wide:218 design.math_reshape.ledger.depth[0].by[1]; tcg_d128_flat_tma_xpose_wide:221 design.math_reshape.ledger.error[0].by[2]; tcg_d128_fscan_cp_direct_narrow:204 design.math_reshape.ledger.footprint[0].by[2]; tcg_d128_fscan_cp_direct_narrow:207 design.math_reshape.ledger.matmul[0].by[2]; tcg_d128_fscan_cp_direct_narrow:210 design.math_reshape.ledger.depth[0].by[1]; tcg_d128_fscan_cp_direct_narrow:213 design.math_reshape.ledger.error[0].by[2]; tcg_d128_fscan_cp_direct_wide:219 design.math_reshape.ledger.footprint[0].by[2]; tcg_d128_fscan_cp_direct_wide:222 design.math_reshape.ledger.matmul[0].by[2]; tcg_d128_fscan_cp_direct_wide:224 design.math_reshape.ledger.depth[0].by[1]; tcg_d128_fscan_cp_direct_wide:227 design.math_reshape.ledger.error[0].by[2]; tcg_d128_fscan_cp_xpose_narrow:207 design.math_reshape.ledger.footprint[0].by[2]; tcg_d128_fscan_cp_xpose_narrow:210 design.math_reshape.ledger.matmul[0].by[2]; tcg_d128_fscan_cp_xpose_narrow:213 design.math_reshape.ledger.depth[0].by[1]; tcg_d128_fscan_cp_xpose_narrow:219 design.math_reshape.ledger.error[0].by[2]; tcg_d128_fscan_cp_xpose_wide:220 design.math_reshape.ledger.footprint[0].by[2]; tcg_d128_fscan_cp_xpose_wide:224 design.math_reshape.ledger.matmul[0].by[2]; tcg_d128_fscan_cp_xpose_wide:227 design.math_reshape.ledger.depth[0].by[1]; tcg_d128_fscan_cp_xpose_wide:233 design.math_reshape.ledger.error[0].by[2]; tcg_d128_fscan_tma_direct_narrow:204 design.math_reshape.ledger.footprint[0].by[2]; tcg_d128_fscan_tma_direct_narrow:207 design.math_reshape.ledger.matmul[0].by[2]; tcg_d128_fscan_tma_direct_narrow:210 design.math_reshape.ledger.depth[0].by[1]; tcg_d128_fscan_tma_direct_narrow:213 design.math_reshape.ledger.error[0].by[2]; tcg_d128_fscan_tma_direct_wide:220 design.math_reshape.ledger.footprint[0].by[2]; tcg_d128_fscan_tma_direct_wide:223 design.math_reshape.ledger.matmul[0].by[2]; tcg_d128_fscan_tma_direct_wide:226 design.math_reshape.ledger.depth[0].by[1]; tcg_d128_fscan_tma_direct_wide:229 design.math_reshape.ledger.error[0].by[2]; tcg_d128_fscan_tma_xpose_narrow:207 design.math_reshape.ledger.footprint[0].by[2]; tcg_d128_fscan_tma_xpose_narrow:210 design.math_reshape.ledger.matmul[0].by[2]; tcg_d128_fscan_tma_xpose_narrow:213 design.math_reshape.ledger.depth[0].by[1]; tcg_d128_fscan_tma_xpose_narrow:219 design.math_reshape.ledger.error[0].by[2]; tcg_d128_fscan_tma_xpose_wide:221 design.math_reshape.ledger.footprint[0].by[2]; tcg_d128_fscan_tma_xpose_wide:224 design.math_reshape.ledger.matmul[0].by[2]; tcg_d128_fscan_tma_xpose_wide:227 design.math_reshape.ledger.depth[0].by[1]; tcg_d128_fscan_tma_xpose_wide:233 design.math_reshape.ledger.error[0].by[2]; tcg_d128_segmented_cp_direct_narrow:215 design.math_reshape.ledger.footprint[0].by[3]; tcg_d128_segmented_cp_direct_narrow:218 design.math_reshape.ledger.matmul[0].by[2]; tcg_d128_segmented_cp_direct_narrow:221 design.math_reshape.ledger.depth[0].by[1]; tcg_d128_segmented_cp_direct_narrow:227 design.math_reshape.ledger.error[0].by[2]; tcg_d128_segmented_cp_direct_wide:228 design.math_reshape.ledger.footprint[0].by[3]; tcg_d128_segmented_cp_direct_wide:231 design.math_reshape.ledger.matmul[0].by[2]; tcg_d128_segmented_cp_direct_wide:234 design.math_reshape.ledger.depth[0].by[1]; tcg_d128_segmented_cp_direct_wide:240 design.math_reshape.ledger.error[0].by[2]; tcg_d128_segmented_cp_xpose_narrow:205 design.math_reshape.ledger.footprint[0].by[3]; tcg_d128_segmented_cp_xpose_narrow:208 design.math_reshape.ledger.matmul[0].by[2]; tcg_d128_segmented_cp_xpose_narrow:211 design.math_reshape.ledger.depth[0].by[1]; tcg_d128_segmented_cp_xpose_narrow:216 design.math_reshape.ledger.error[0].by[2]; tcg_d128_segmented_cp_xpose_wide:238 design.math_reshape.ledger.footprint[0].by[3]; tcg_d128_segmented_cp_xpose_wide:241 design.math_reshape.ledger.matmul[0].by[2]; tcg_d128_segmented_cp_xpose_wide:244 design.math_reshape.ledger.depth[0].by[2]; tcg_d128_segmented_cp_xpose_wide:247 design.math_reshape.ledger.error[0].by[2]; tcg_d128_segmented_tma_direct_narrow:215 design.math_reshape.ledger.footprint[0].by[3]; tcg_d128_segmented_tma_direct_narrow:218 design.math_reshape.ledger.matmul[0].by[2]; tcg_d128_segmented_tma_direct_narrow:221 design.math_reshape.ledger.depth[0].by[1]; tcg_d128_segmented_tma_direct_narrow:227 design.math_reshape.ledger.error[0].by[2]; tcg_d128_segmented_tma_direct_wide:230 design.math_reshape.ledger.footprint[0].by[3]; tcg_d128_segmented_tma_direct_wide:234 design.math_reshape.ledger.matmul[0].by[2]; tcg_d128_segmented_tma_direct_wide:237 design.math_reshape.ledger.depth[0].by[1]; tcg_d128_segmented_tma_direct_wide:243 design.math_reshape.ledger.error[0].by[2]; tcg_d128_segmented_tma_xpose_narrow:221 design.math_reshape.ledger.footprint[0].by[3]; tcg_d128_segmented_tma_xpose_narrow:224 design.math_reshape.ledger.matmul[0].by[2]; tcg_d128_segmented_tma_xpose_narrow:227 design.math_reshape.ledger.depth[0].by[1]; tcg_d128_segmented_tma_xpose_narrow:232 design.math_reshape.ledger.error[0].by[2]; tcg_d128_segmented_tma_xpose_wide:205 design.math_reshape.ledger.footprint[0].by[3]; tcg_d128_segmented_tma_xpose_wide:208 design.math_reshape.ledger.matmul[0].by[2]; tcg_d128_segmented_tma_xpose_wide:211 design.math_reshape.ledger.depth[0].by[1]; tcg_d128_segmented_tma_xpose_wide:216 design.math_reshape.ledger.error[0].by[2]; tcg_d256_flat_cp_narrow:182 design.math_reshape.ledger.footprint[0].by[2]; tcg_d256_flat_cp_narrow:184 design.math_reshape.ledger.matmul[0].by[2]; tcg_d256_flat_cp_narrow:186 design.math_reshape.ledger.depth[0].by[1]; tcg_d256_flat_cp_narrow:188 design.math_reshape.ledger.traffic[0].by[0]; tcg_d256_flat_cp_narrow:190 design.math_reshape.ledger.error[0].by[2]; tcg_d256_flat_cp_wide:196 design.math_reshape.ledger.footprint[0].by[2]; tcg_d256_flat_cp_wide:198 design.math_reshape.ledger.matmul[0].by[2]; tcg_d256_flat_cp_wide:200 design.math_reshape.ledger.depth[0].by[1]; tcg_d256_flat_cp_wide:202 design.math_reshape.ledger.traffic[0].by[0]; tcg_d256_flat_cp_wide:204 design.math_reshape.ledger.error[0].by[2]; tcg_d256_flat_tma_narrow:182 design.math_reshape.ledger.footprint[0].by[2]; tcg_d256_flat_tma_narrow:184 design.math_reshape.ledger.matmul[0].by[2]; tcg_d256_flat_tma_narrow:186 design.math_reshape.ledger.depth[0].by[1]; tcg_d256_flat_tma_narrow:188 design.math_reshape.ledger.traffic[0].by[0]; tcg_d256_flat_tma_narrow:190 design.math_reshape.ledger.error[0].by[2]; tcg_d256_flat_tma_wide:194 design.math_reshape.ledger.footprint[0].by[2]; tcg_d256_flat_tma_wide:196 design.math_reshape.ledger.matmul[0].by[2]; tcg_d256_flat_tma_wide:198 design.math_reshape.ledger.depth[0].by[1]; tcg_d256_flat_tma_wide:200 design.math_reshape.ledger.traffic[0].by[0]; tcg_d256_flat_tma_wide:202 design.math_reshape.ledger.error[0].by[2]; tcg_d256_segmented_cp_narrow:189 design.math_reshape.ledger.footprint[0].by[3]; tcg_d256_segmented_cp_narrow:191 design.math_reshape.ledger.matmul[0].by[2]; tcg_d256_segmented_cp_narrow:193 design.math_reshape.ledger.depth[0].by[1]; tcg_d256_segmented_cp_narrow:196 design.math_reshape.ledger.traffic[1].by[0]; tcg_d256_segmented_cp_narrow:198 design.math_reshape.ledger.error[0].by[2]; tcg_d256_segmented_cp_wide:235 design.math_reshape.ledger.footprint[0].by[3]; tcg_d256_segmented_cp_wide:237 design.math_reshape.ledger.matmul[0].by[2]; tcg_d256_segmented_cp_wide:239 design.math_reshape.ledger.depth[0].by[1]; tcg_d256_segmented_cp_wide:242 design.math_reshape.ledger.traffic[1].by[0]; tcg_d256_segmented_cp_wide:244 design.math_reshape.ledger.error[0].by[2]; tcg_d256_segmented_tma_narrow:204 design.math_reshape.ledger.footprint[0].by[3]; tcg_d256_segmented_tma_narrow:206 design.math_reshape.ledger.matmul[0].by[2]; tcg_d256_segmented_tma_narrow:208 design.math_reshape.ledger.depth[0].by[1]; tcg_d256_segmented_tma_narrow:211 design.math_reshape.ledger.traffic[1].by[0]; tcg_d256_segmented_tma_narrow:213 design.math_reshape.ledger.error[0].by[2]; tcg_d256_segmented_tma_wide:189 design.math_reshape.ledger.footprint[0].by[3]; tcg_d256_segmented_tma_wide:191 design.math_reshape.ledger.matmul[0].by[2]; tcg_d256_segmented_tma_wide:193 design.math_reshape.ledger.depth[0].by[1]; tcg_d256_segmented_tma_wide:196 design.math_reshape.ledger.traffic[1].by[0]; tcg_d256_segmented_tma_wide:198 design.math_reshape.ledger.error[0].by[2]

partials

## t_73b6783a9e132575 | math.ledger | 84 chars | 1 declarations

tc_split_segmented_wide:308 design.math_reshape.ledger.footprint[1].change

Add 4*CombineGroups*(D+CombineBt/32) bytes of retained group partials per output row

## t_73dad17941e893d9 | math.ledger | 34 chars | 2 declarations

tc_split_segmented_narrow:291 design.math_reshape.ledger.traffic[1].item; tc_split_segmented_wide:319 design.math_reshape.ledger.traffic[1].item

Segment summaries and replay reads

## t_74851ed48a964722 | math.ledger | 148 chars | 1 declarations

fused_d128_key:168 design.math_reshape.ledger.footprint[0].change

9*B*H*Sq*Skv bytes -> none globally; per-worktile retained Score and Prob8 total 5*Br*KeyGroup bytes, plus 4*Br*(D+3) bytes of logical carried state

## t_765105f966b2c35b | math.ledger | 217 chars | 7 declarations

tcg_d256_flat_cp_narrow:177 design.math_reshape.ledger.baseline; tcg_d256_flat_cp_wide:191 design.math_reshape.ledger.baseline; tcg_d256_flat_tma_narrow:177 design.math_reshape.ledger.baseline; tcg_d256_flat_tma_wide:189 design.math_reshape.ledger.baseline; tcg_d256_segmented_cp_narrow:184 design.math_reshape.ledger.baseline; tcg_d256_segmented_tma_narrow:199 design.math_reshape.ledger.baseline; tcg_d256_segmented_tma_wide:184 design.math_reshape.ledger.baseline

Direct Contract evaluation materializing FP32 scores/probabilities and E4M3 probabilities. Product figures include Br-row and key-group padding; bytes count logical payload, not allocator capacity or measured traffic.

## t_793031b08801adc9 | math.ledger | 243 chars | 2 declarations

fused_d256_cp:168 design.math_reshape.ledger.error[0].change; fused_d256_tma:168 design.math_reshape.ledger.error[0].change

Native block-scaled contraction order, carried PV accumulation, separately carried half denominators, E1 without the low log2(e) correction, and output reciprocal/logf replace reference evaluation choices; no end-to-end error bound is asserted

## t_7a223ec12a4ac31f | math.ledger | 145 chars | 5 declarations

tcg_d256_flat_cp_narrow:190 design.math_reshape.ledger.error[0].change; tcg_d256_flat_tma_narrow:190 design.math_reshape.ledger.error[0].change; tcg_d256_segmented_cp_narrow:198 design.math_reshape.ledger.error[0].change; tcg_d256_segmented_tma_narrow:213 design.math_reshape.ledger.error[0].change; tcg_d256_segmented_tma_wide:198 design.math_reshape.ledger.error[0].change

Native contractions, E1, separate half denominators, split weighting and ordered partial addition change reference rounding; no final error bound

## t_7b4f04d08c6d911f | math.ledger | 281 chars | 1 declarations

fused_d128_group:171 design.math_reshape.ledger.error[0].change

Native block-scaled contraction order, carried PV accumulation, one whole-group denominator recurrence with even/odd-key summation chains, E16 without the low log2(e) correction, and output reciprocal/logf replace reference evaluation choices; no end-to-end error bound is asserted

## t_7bff916ad3016883 | math.ledger | 115 chars | 6 declarations

fused_d128_cp:170 design.math_reshape.ledger.depth[0].change; fused_d128_group:169 design.math_reshape.ledger.depth[0].change; fused_d128_key:172 design.math_reshape.ledger.depth[0].change; fused_d256_cp:166 design.math_reshape.ledger.depth[0].change; fused_d256_tma:166 design.math_reshape.ledger.depth[0].change; root:380 design.math_reshape.ledger.depth[0].change

ceil_div(Skv,KeyGroup) -> Ng ordered state transitions per worktile; no inter-group reassociation of delayed maxima

## t_7d281f4da5f14b73 | math.ledger | 36 chars | 4 declarations

tc_split_flat_narrow:252 design.math_reshape.ledger.traffic[0].item; tc_split_flat_wide:264 design.math_reshape.ledger.traffic[0].item; tc_split_segmented_narrow:287 design.math_reshape.ledger.traffic[0].item; tc_split_segmented_wide:315 design.math_reshape.ledger.traffic[0].item

Partials and temporary output stores

## t_7d79d9d4b6f4d782 | math.ledger | 170 chars | 1 declarations

tcg_d256_flat_cp_wide:204 design.math_reshape.ledger.error[0].change

Native contractions, E1, separate half denominators, split weighting, grouped numerator sums and tree-reduced denominators change reference rounding; no final error bound

## t_7e52b4948a1a7544 | math.ledger | 188 chars | 1 declarations

tcg_d128_fscan_cp_xpose_wide:226 design.math_reshape.ledger.depth[0].change

T -> max_s Count(Ns,s) group steps per numerator, with min(L(Ns,s),T) preceding scalar scan steps per split; merge numerator depth is at most max(8,Chunk)+MergeGroups additions per channel

## t_834eecc9b30a8bd3 | math.ledger | 235 chars | 2 declarations

fused_d128_cp:172 design.math_reshape.ledger.error[0].change; root:382 design.math_reshape.ledger.error[0].change

Native block-scaled contraction order, carried PV accumulation, four-pair denominator trees, E16 without the low log2(e) correction, and output reciprocal/logf replace reference evaluation choices; no end-to-end error bound is asserted

## t_8479e5cb634464ac | math.ledger | 12 chars | 1 declarations

tcg_d256_segmented_cp_wide:241 design.math_reshape.ledger.traffic[0].item

Scan staging

## t_862c7f171ea97ea1 | math.ledger | 136 chars | 1 declarations

tcg_d128_flat_tma_xpose_wide:217 design.math_reshape.ledger.depth[0].change

T -> max_s Count(Ns,s) group steps; a T-step scan and at most ceil_div(Ns,SP)+SP serial numerator-merge additions remain per row/channel

## t_8d0a3cd96d5833ca | math.ledger | 69 chars | 1 declarations

tcg_d128_flat_tma_direct_wide:206 design.math_reshape.ledger.footprint[1].change

0 -> 4*MergeGroups*(D+4) bytes retained per row until final summation

## t_8d1a25742302e8ef | math.ledger | 7 chars | 64 declarations

tc_split_flat_narrow:255 design.math_reshape.ledger.traffic[0].by[2]; tc_split_flat_narrow:260 design.math_reshape.ledger.depth[0].by[3]; tc_split_flat_narrow:265 design.math_reshape.ledger.error[0].by[4]; tc_split_flat_wide:267 design.math_reshape.ledger.traffic[0].by[2]; tc_split_flat_wide:273 design.math_reshape.ledger.depth[0].by[3]; tc_split_flat_wide:278 design.math_reshape.ledger.error[0].by[4]; tc_split_segmented_narrow:290 design.math_reshape.ledger.traffic[0].by[2]; tc_split_segmented_narrow:301 design.math_reshape.ledger.depth[0].by[4]; tc_split_segmented_narrow:306 design.math_reshape.ledger.error[0].by[6]; tc_split_segmented_wide:309 design.math_reshape.ledger.footprint[1].by[0]; tc_split_segmented_wide:318 design.math_reshape.ledger.traffic[0].by[2]; tc_split_segmented_wide:330 design.math_reshape.ledger.depth[0].by[4]; tc_split_segmented_wide:335 design.math_reshape.ledger.error[0].by[6]; tcg_d128_flat_cp_direct_narrow:199 design.math_reshape.ledger.depth[0].by[2]; tcg_d128_flat_cp_direct_narrow:203 design.math_reshape.ledger.error[0].by[3]; tcg_d128_flat_cp_direct_wide:211 design.math_reshape.ledger.depth[0].by[2]; tcg_d128_flat_cp_direct_wide:215 design.math_reshape.ledger.error[0].by[3]; tcg_d128_flat_cp_xpose_narrow:206 design.math_reshape.ledger.depth[0].by[2]; tcg_d128_flat_cp_xpose_narrow:210 design.math_reshape.ledger.error[0].by[3]; tcg_d128_flat_cp_xpose_wide:189 design.math_reshape.ledger.depth[0].by[2]; tcg_d128_flat_cp_xpose_wide:193 design.math_reshape.ledger.error[0].by[3]; tcg_d128_flat_tma_direct_narrow:200 design.math_reshape.ledger.depth[0].by[2]; tcg_d128_flat_tma_direct_narrow:204 design.math_reshape.ledger.error[0].by[3]; tcg_d128_flat_tma_direct_wide:206 design.math_reshape.ledger.footprint[1].by[0]; tcg_d128_flat_tma_direct_wide:212 design.math_reshape.ledger.depth[0].by[2]; tcg_d128_flat_tma_direct_wide:216 design.math_reshape.ledger.error[0].by[3]; tcg_d128_flat_tma_xpose_narrow:203 design.math_reshape.ledger.depth[0].by[2]; tcg_d128_flat_tma_xpose_narrow:207 design.math_reshape.ledger.error[0].by[3]; tcg_d128_flat_tma_xpose_wide:218 design.math_reshape.ledger.depth[0].by[2]; tcg_d128_flat_tma_xpose_wide:222 design.math_reshape.ledger.error[0].by[3]; tcg_d128_fscan_cp_direct_narrow:210 design.math_reshape.ledger.depth[0].by[2]; tcg_d128_fscan_cp_direct_narrow:214 design.math_reshape.ledger.error[0].by[3]; tcg_d128_fscan_cp_direct_wide:224 design.math_reshape.ledger.depth[0].by[2]; tcg_d128_fscan_cp_direct_wide:228 design.math_reshape.ledger.error[0].by[3]; tcg_d128_fscan_cp_xpose_narrow:213 design.math_reshape.ledger.depth[0].by[2]; tcg_d128_fscan_cp_xpose_narrow:220 design.math_reshape.ledger.error[0].by[3]; tcg_d128_fscan_cp_xpose_wide:221 design.math_reshape.ledger.footprint[1].by[0]; tcg_d128_fscan_cp_xpose_wide:227 design.math_reshape.ledger.depth[0].by[2]; tcg_d128_fscan_cp_xpose_wide:234 design.math_reshape.ledger.error[0].by[3]; tcg_d128_fscan_tma_direct_narrow:210 design.math_reshape.ledger.depth[0].by[2]; tcg_d128_fscan_tma_direct_narrow:214 design.math_reshape.ledger.error[0].by[3]; tcg_d128_fscan_tma_direct_wide:226 design.math_reshape.ledger.depth[0].by[2]; tcg_d128_fscan_tma_direct_wide:230 design.math_reshape.ledger.error[0].by[3]; tcg_d128_fscan_tma_xpose_narrow:213 design.math_reshape.ledger.depth[0].by[2]; tcg_d128_fscan_tma_xpose_narrow:220 design.math_reshape.ledger.error[0].by[3]; tcg_d128_fscan_tma_xpose_wide:227 design.math_reshape.ledger.depth[0].by[2]; tcg_d128_fscan_tma_xpose_wide:234 design.math_reshape.ledger.error[0].by[3]; tcg_d128_segmented_cp_direct_narrow:221 design.math_reshape.ledger.depth[0].by[2]; tcg_d128_segmented_cp_direct_narrow:228 design.math_reshape.ledger.error[0].by[3]; tcg_d128_segmented_cp_direct_wide:234 design.math_reshape.ledger.depth[0].by[2]; tcg_d128_segmented_cp_direct_wide:241 design.math_reshape.ledger.error[0].by[3]; tcg_d128_segmented_cp_xpose_narrow:211 design.math_reshape.ledger.depth[0].by[2]; tcg_d128_segmented_cp_xpose_narrow:217 design.math_reshape.ledger.error[0].by[3]; tcg_d128_segmented_cp_xpose_wide:244 design.math_reshape.ledger.depth[0].by[3]; tcg_d128_segmented_cp_xpose_wide:248 design.math_reshape.ledger.error[0].by[3]; tcg_d128_segmented_tma_direct_narrow:221 design.math_reshape.ledger.depth[0].by[2]; tcg_d128_segmented_tma_direct_narrow:228 design.math_reshape.ledger.error[0].by[3]; tcg_d128_segmented_tma_direct_wide:231 design.math_reshape.ledger.footprint[1].by[0]; tcg_d128_segmented_tma_direct_wide:237 design.math_reshape.ledger.depth[0].by[2]; tcg_d128_segmented_tma_direct_wide:244 design.math_reshape.ledger.error[0].by[3]; tcg_d128_segmented_tma_xpose_narrow:227 design.math_reshape.ledger.depth[0].by[2]; tcg_d128_segmented_tma_xpose_narrow:233 design.math_reshape.ledger.error[0].by[3]; tcg_d128_segmented_tma_xpose_wide:211 design.math_reshape.ledger.depth[0].by[2]; tcg_d128_segmented_tma_xpose_wide:217 design.math_reshape.ledger.error[0].by[3]

combine

## t_8d3c94d5f2b313d1 | math.ledger | 273 chars | 1 declarations

fused_d128_key:174 design.math_reshape.ledger.error[0].change

Native block-scaled contraction order, carried PV accumulation, two slice-local denominator recurrences followed by one final sum, E16 without the low log2(e) correction, and output reciprocal/logf replace reference evaluation choices; no end-to-end error bound is asserted

## t_8de17af0705b502d | math.ledger | 177 chars | 3 declarations

tcg_d128_segmented_cp_xpose_narrow:210 design.math_reshape.ledger.depth[0].change; tcg_d128_segmented_tma_xpose_narrow:226 design.math_reshape.ledger.depth[0].change; tcg_d128_segmented_tma_xpose_wide:210 design.math_reshape.ledger.depth[0].change

T -> max_s Count(Ns,s) group steps; scan walks WalkCount(r) groups and tests SegBatch*ceil_div(Nseg-1,SegBatch) bounds per row, retaining T snapshot stores and an Ns-entry merge

## t_8e85484856d38453 | math.ledger | 215 chars | 1 declarations

tc_split_flat_wide:276 design.math_reshape.ledger.error[0].change

Post-dot scale correction, BF16 V, four denominator partials and grouped weighted split sums change reference rounding; uncovered rows additionally depend on incoming scratch, without a general equivalence guarantee

## t_8f7d889ea1db1c26 | math.ledger | 194 chars | 1 declarations

tcg_d128_fscan_cp_xpose_narrow:218 design.math_reshape.ledger.error[0].change

Native block-scaled contractions, E16, denominator trees, separate per-group PV addition, split weighting and ordered merge replace the reference evaluation choices; no end-to-end bound asserted

## t_91a1e4fddd45b653 | math.ledger | 150 chars | 6 declarations

fused_d128_cp:168 design.math_reshape.ledger.matmul[0].change; fused_d128_group:167 design.math_reshape.ledger.matmul[0].change; fused_d128_key:170 design.math_reshape.ledger.matmul[0].change; fused_d256_cp:164 design.math_reshape.ledger.matmul[0].change; fused_d256_tma:164 design.math_reshape.ledger.matmul[0].change; root:378 design.math_reshape.ledger.matmul[0].change

Per (b,kh,a), 2*G*min(Qph,Sq-q0)*Skv*D MACs -> 2*Br*KeyGroup*Ng*D MACs; masked boundary groups still execute fully, wholly excluded groups are skipped

## t_951528be4fa74735 | math.ledger | 89 chars | 4 declarations

tcg_d128_segmented_cp_direct_narrow:223 design.math_reshape.ledger.vector[0].change; tcg_d128_segmented_cp_direct_wide:236 design.math_reshape.ledger.vector[0].change; tcg_d128_segmented_tma_direct_narrow:223 design.math_reshape.ledger.vector[0].change; tcg_d128_segmented_tma_direct_wide:239 design.math_reshape.ledger.vector[0].change

0 -> Seg*Nseg fmax evaluations and SegBatch*ceil_div(Nseg-1,SegBatch) comparisons per row

## t_98f802fcf595c781 | math.ledger | 18 chars | 20 declarations

tcg_d128_flat_cp_direct_narrow:201 design.math_reshape.ledger.error[0].item; tcg_d128_flat_cp_direct_wide:213 design.math_reshape.ledger.error[0].item; tcg_d128_flat_cp_xpose_narrow:208 design.math_reshape.ledger.error[0].item; tcg_d128_flat_cp_xpose_wide:191 design.math_reshape.ledger.error[0].item; tcg_d128_flat_tma_direct_narrow:202 design.math_reshape.ledger.error[0].item; tcg_d128_flat_tma_direct_wide:214 design.math_reshape.ledger.error[0].item; tcg_d128_flat_tma_xpose_narrow:205 design.math_reshape.ledger.error[0].item; tcg_d128_flat_tma_xpose_wide:220 design.math_reshape.ledger.error[0].item; tcg_d128_fscan_cp_direct_narrow:212 design.math_reshape.ledger.error[0].item; tcg_d128_fscan_cp_direct_wide:226 design.math_reshape.ledger.error[0].item; tcg_d128_fscan_tma_direct_narrow:212 design.math_reshape.ledger.error[0].item; tcg_d128_fscan_tma_direct_wide:228 design.math_reshape.ledger.error[0].item; tcg_d128_fscan_tma_xpose_narrow:218 design.math_reshape.ledger.error[0].item; tcg_d128_segmented_cp_direct_narrow:226 design.math_reshape.ledger.error[0].item; tcg_d128_segmented_cp_direct_wide:239 design.math_reshape.ledger.error[0].item; tcg_d128_segmented_cp_xpose_narrow:215 design.math_reshape.ledger.error[0].item; tcg_d128_segmented_tma_direct_narrow:226 design.math_reshape.ledger.error[0].item; tcg_d128_segmented_tma_direct_wide:242 design.math_reshape.ledger.error[0].item; tcg_d128_segmented_tma_xpose_narrow:231 design.math_reshape.ledger.error[0].item; tcg_d128_segmented_tma_xpose_wide:215 design.math_reshape.ledger.error[0].item

Evaluation changes

## t_9a10f0ab95588d85 | math.ledger | 207 chars | 2 declarations

tc_split_flat_narrow:263 design.math_reshape.ledger.error[0].change; tc_split_segmented_narrow:304 design.math_reshape.ledger.error[0].change

Post-dot scale correction, BF16 V, four denominator partials and weighted split sums change reference rounding; uncovered rows additionally depend on incoming scratch, without a general equivalence guarantee

## t_9ae37fa8d6dc3543 | math.ledger | 7 chars | 8 declarations

tc_split_segmented_narrow:281 design.math_reshape.ledger.footprint[0].by[1]; tc_split_segmented_narrow:295 design.math_reshape.ledger.traffic[1].by[0]; tc_split_segmented_narrow:301 design.math_reshape.ledger.depth[0].by[0]; tc_split_segmented_narrow:306 design.math_reshape.ledger.error[0].by[2]; tc_split_segmented_wide:306 design.math_reshape.ledger.footprint[0].by[1]; tc_split_segmented_wide:323 design.math_reshape.ledger.traffic[1].by[0]; tc_split_segmented_wide:330 design.math_reshape.ledger.depth[0].by[0]; tc_split_segmented_wide:335 design.math_reshape.ledger.error[0].by[2]

segment

## t_9b3e8394d08bea8f | math.ledger | 71 chars | 1 declarations

tcg_d256_segmented_cp_wide:242 design.math_reshape.ledger.traffic[1].change

0 -> 8*Ns*R*(D+1) bytes for one logical write and read of FP32 partials

## t_9c41a5404ebc3f01 | math.ledger | 196 chars | 1 declarations

tcg_d128_segmented_tma_direct_wide:236 design.math_reshape.ledger.depth[0].change

T -> max_s Count(Ns,s) group steps; scan retains WalkCount(r) ordered group updates, at most T, and the merge numerator chain has at most ceil_div(Ns,MergeGroups)+MergeGroups additions per channel

## t_a7229c97b28ed6a0 | math.ledger | 24 chars | 1 declarations

tc_fused:173 design.math_reshape.ledger.depth[0].item

Delayed state recurrence

## t_aca59607dffc14f6 | math.ledger | 189 chars | 1 declarations

tcg_d256_segmented_cp_wide:241 design.math_reshape.ledger.traffic[0].change

0 -> 4*R*(T+2*Nseg-1) bytes for segment input reads, SegMax writes and scan reads of SegMax[1:]; scan additionally reads between SegLen*R and T*R GroupMax elements and writes (T+1)*R states

## t_ad9ac6d6f1b08556 | math.ledger | 7 chars | 20 declarations

tc_split_flat_narrow:246 design.math_reshape.ledger.footprint[0].by[1]; tc_split_flat_narrow:250 design.math_reshape.ledger.matmul[0].by[2]; tc_split_flat_narrow:255 design.math_reshape.ledger.traffic[0].by[1]; tc_split_flat_narrow:260 design.math_reshape.ledger.depth[0].by[2]; tc_split_flat_narrow:265 design.math_reshape.ledger.error[0].by[3]; tc_split_flat_wide:258 design.math_reshape.ledger.footprint[0].by[1]; tc_split_flat_wide:262 design.math_reshape.ledger.matmul[0].by[2]; tc_split_flat_wide:267 design.math_reshape.ledger.traffic[0].by[1]; tc_split_flat_wide:273 design.math_reshape.ledger.depth[0].by[2]; tc_split_flat_wide:278 design.math_reshape.ledger.error[0].by[3]; tc_split_segmented_narrow:281 design.math_reshape.ledger.footprint[0].by[3]; tc_split_segmented_narrow:285 design.math_reshape.ledger.matmul[0].by[2]; tc_split_segmented_narrow:290 design.math_reshape.ledger.traffic[0].by[1]; tc_split_segmented_narrow:301 design.math_reshape.ledger.depth[0].by[3]; tc_split_segmented_narrow:306 design.math_reshape.ledger.error[0].by[5]; tc_split_segmented_wide:306 design.math_reshape.ledger.footprint[0].by[3]; tc_split_segmented_wide:313 design.math_reshape.ledger.matmul[0].by[2]; tc_split_segmented_wide:318 design.math_reshape.ledger.traffic[0].by[1]; tc_split_segmented_wide:330 design.math_reshape.ledger.depth[0].by[3]; tc_split_segmented_wide:335 design.math_reshape.ledger.error[0].by[5]

partial

## t_aea87339806bd59b | math.ledger | 194 chars | 2 declarations

tcg_d128_fscan_tma_xpose_wide:232 design.math_reshape.ledger.error[0].change; tcg_d128_segmented_cp_xpose_wide:246 design.math_reshape.ledger.error[0].change

Native block-scaled contractions, E16, denominator trees, separate per-group PV addition, split weighting and grouped merge replace the reference evaluation choices; no end-to-end bound asserted

## t_b0ba6cf00ffa43d5 | math.ledger | 143 chars | 1 declarations

tc_split_flat_narrow:258 design.math_reshape.ledger.depth[0].change

T coupled group updates -> T scan updates per row followed by at most ceil_div(Ng(a),Ns) PV updates per interval and PaddedNs combine additions

## t_b0d22b4036d12fec | math.ledger | 373 chars | 1 declarations

tc_split_segmented_wide:326 design.math_reshape.ledger.depth[0].change

T coupled group updates -> Seg ordered segment fmax operations, then Nc chunk decisions and WalkCount(r) serial maximum tests per row, followed by at most ceil_div(Ng(a),Ns) PV updates per interval and at most Chunk local numerator additions in combine, five denominator tree stages, then CombineGroups numerator and Npartial denominator additions; all T scan stores remain

## t_b2283dfb50418cd3 | math.ledger | 168 chars | 1 declarations

fused_d128_group:165 design.math_reshape.ledger.footprint[0].change

9*B*H*Sq*Skv bytes -> none globally; retained Score and Prob8 total 5*Br*KeyGroup bytes per evaluated group, plus 4*Br*(D+2) bytes of logical carried state per worktile

## t_b5388e8345f0dcc0 | math.ledger | 265 chars | 1 declarations

tc_split_flat_wide:270 design.math_reshape.ledger.depth[0].change

T coupled group updates -> T scan updates per row followed by at most ceil_div(Ng(a),Ns) PV updates per interval; combine adds at most Cb numerator terms plus four interval sums per channel, and uses five denominator tree stages plus sixteen final additions per row

## t_b7de17713b5612a8 | math.ledger | 163 chars | 13 declarations

tcg_d128_flat_cp_direct_narrow:201 design.math_reshape.ledger.error[0].change; tcg_d128_flat_cp_xpose_narrow:208 design.math_reshape.ledger.error[0].change; tcg_d128_flat_cp_xpose_wide:191 design.math_reshape.ledger.error[0].change; tcg_d128_flat_tma_direct_narrow:202 design.math_reshape.ledger.error[0].change; tcg_d128_flat_tma_xpose_narrow:205 design.math_reshape.ledger.error[0].change; tcg_d128_fscan_cp_direct_narrow:212 design.math_reshape.ledger.error[0].change; tcg_d128_fscan_tma_direct_narrow:212 design.math_reshape.ledger.error[0].change; tcg_d128_fscan_tma_xpose_narrow:218 design.math_reshape.ledger.error[0].change; tcg_d128_segmented_cp_direct_narrow:226 design.math_reshape.ledger.error[0].change; tcg_d128_segmented_cp_xpose_narrow:215 design.math_reshape.ledger.error[0].change; tcg_d128_segmented_tma_direct_narrow:226 design.math_reshape.ledger.error[0].change; tcg_d128_segmented_tma_xpose_narrow:231 design.math_reshape.ledger.error[0].change; tcg_d128_segmented_tma_xpose_wide:215 design.math_reshape.ledger.error[0].change

Native contractions, E16, denominator trees, fresh per-group PV, split weights and ordered merge replace reference evaluation choices; no end-to-end bound asserted

## t_b90d4750e2046f11 | math.ledger | 140 chars | 1 declarations

tcg_d256_flat_tma_wide:198 design.math_reshape.ledger.depth[0].change

Ng -> at most ceil_div(Ng,Ns) steps per split plus at most ceil_div(Ns,SP)+SP combine additions; a separate GCount-step maximum scan remains

## t_bafdff3a9ddea934 | math.ledger | 23 chars | 4 declarations

tc_split_flat_narrow:257 design.math_reshape.ledger.depth[0].item; tc_split_flat_wide:269 design.math_reshape.ledger.depth[0].item; tc_split_segmented_narrow:297 design.math_reshape.ledger.depth[0].item; tc_split_segmented_wide:325 design.math_reshape.ledger.depth[0].item

State and PV recurrence

## t_bbd5935391cd5f8a | math.ledger | 166 chars | 2 declarations

tc_split_flat_narrow:244 design.math_reshape.ledger.footprint[0].change; tc_split_flat_wide:256 design.math_reshape.ledger.footprint[0].change

9*R*Skv bytes -> 4*T*(Rc+R)+4*R+4*Ns*R*(D+1) bytes of written maxima, states, tails and partials; another 4*T*(R-Rc) bytes of incoming scratch supply uncovered maxima

## t_bf2847edaf08ec14 | math.ledger | 77 chars | 4 declarations

tcg_d256_flat_cp_narrow:182 design.math_reshape.ledger.footprint[0].change; tcg_d256_flat_cp_wide:196 design.math_reshape.ledger.footprint[0].change; tcg_d256_flat_tma_narrow:182 design.math_reshape.ledger.footprint[0].change; tcg_d256_flat_tma_wide:194 design.math_reshape.ledger.footprint[0].change

9*R*Skv -> 4*R*(2*GCount+1+Ns*(D+1)) bytes; no global scores or probabilities

## t_c67eae106242f3bd | math.ledger | 17 chars | 7 declarations

tcg_d128_segmented_cp_direct_narrow:223 design.math_reshape.ledger.vector[0].item; tcg_d128_segmented_cp_direct_wide:236 design.math_reshape.ledger.vector[0].item; tcg_d128_segmented_cp_xpose_narrow:213 design.math_reshape.ledger.vector[0].item; tcg_d128_segmented_tma_direct_narrow:223 design.math_reshape.ledger.vector[0].item; tcg_d128_segmented_tma_direct_wide:239 design.math_reshape.ledger.vector[0].item; tcg_d128_segmented_tma_xpose_narrow:229 design.math_reshape.ledger.vector[0].item; tcg_d128_segmented_tma_xpose_wide:213 design.math_reshape.ledger.vector[0].item

Segment screening

## t_c6dda4ef5b0d1da8 | math.ledger | 20 chars | 36 declarations

tc_split_flat_narrow:243 design.math_reshape.ledger.footprint[0].item; tc_split_flat_wide:255 design.math_reshape.ledger.footprint[0].item; tc_split_segmented_narrow:278 design.math_reshape.ledger.footprint[0].item; tc_split_segmented_wide:303 design.math_reshape.ledger.footprint[0].item; tcg_d128_flat_cp_direct_narrow:192 design.math_reshape.ledger.footprint[0].item; tcg_d128_flat_cp_direct_wide:204 design.math_reshape.ledger.footprint[0].item; tcg_d128_flat_cp_xpose_narrow:199 design.math_reshape.ledger.footprint[0].item; tcg_d128_flat_cp_xpose_wide:182 design.math_reshape.ledger.footprint[0].item; tcg_d128_flat_tma_direct_narrow:193 design.math_reshape.ledger.footprint[0].item; tcg_d128_flat_tma_direct_wide:204 design.math_reshape.ledger.footprint[0].item; tcg_d128_flat_tma_xpose_narrow:196 design.math_reshape.ledger.footprint[0].item; tcg_d128_flat_tma_xpose_wide:211 design.math_reshape.ledger.footprint[0].item; tcg_d128_fscan_cp_direct_narrow:203 design.math_reshape.ledger.footprint[0].item; tcg_d128_fscan_cp_direct_wide:218 design.math_reshape.ledger.footprint[0].item; tcg_d128_fscan_cp_xpose_narrow:206 design.math_reshape.ledger.footprint[0].item; tcg_d128_fscan_cp_xpose_wide:219 design.math_reshape.ledger.footprint[0].item; tcg_d128_fscan_tma_direct_narrow:203 design.math_reshape.ledger.footprint[0].item; tcg_d128_fscan_tma_direct_wide:219 design.math_reshape.ledger.footprint[0].item; tcg_d128_fscan_tma_xpose_narrow:206 design.math_reshape.ledger.footprint[0].item; tcg_d128_fscan_tma_xpose_wide:220 design.math_reshape.ledger.footprint[0].item; tcg_d128_segmented_cp_direct_narrow:214 design.math_reshape.ledger.footprint[0].item; tcg_d128_segmented_cp_direct_wide:227 design.math_reshape.ledger.footprint[0].item; tcg_d128_segmented_cp_xpose_narrow:204 design.math_reshape.ledger.footprint[0].item; tcg_d128_segmented_cp_xpose_wide:237 design.math_reshape.ledger.footprint[0].item; tcg_d128_segmented_tma_direct_narrow:214 design.math_reshape.ledger.footprint[0].item; tcg_d128_segmented_tma_direct_wide:229 design.math_reshape.ledger.footprint[0].item; tcg_d128_segmented_tma_xpose_narrow:220 design.math_reshape.ledger.footprint[0].item; tcg_d128_segmented_tma_xpose_wide:204 design.math_reshape.ledger.footprint[0].item; tcg_d256_flat_cp_narrow:182 design.math_reshape.ledger.footprint[0].item; tcg_d256_flat_cp_wide:196 design.math_reshape.ledger.footprint[0].item; tcg_d256_flat_tma_narrow:182 design.math_reshape.ledger.footprint[0].item; tcg_d256_flat_tma_wide:194 design.math_reshape.ledger.footprint[0].item; tcg_d256_segmented_cp_narrow:189 design.math_reshape.ledger.footprint[0].item; tcg_d256_segmented_cp_wide:235 design.math_reshape.ledger.footprint[0].item; tcg_d256_segmented_tma_narrow:204 design.math_reshape.ledger.footprint[0].item; tcg_d256_segmented_tma_wide:189 design.math_reshape.ledger.footprint[0].item

Global intermediates

## t_c94d5666534ba90a | math.ledger | 42 chars | 6 declarations

fused_d128_cp:168 design.math_reshape.ledger.matmul[0].item; fused_d128_group:167 design.math_reshape.ledger.matmul[0].item; fused_d128_key:170 design.math_reshape.ledger.matmul[0].item; fused_d256_cp:164 design.math_reshape.ledger.matmul[0].item; fused_d256_tma:164 design.math_reshape.ledger.matmul[0].item; root:378 design.math_reshape.ledger.matmul[0].item

Dense products including padding and masks

## t_c99cd113c748c987 | math.ledger | 165 chars | 1 declarations

tc_fused:167 design.math_reshape.ledger.footprint[0].change

9*B*H*Sq*Skv global bytes -> no global intermediate; per worktile retain 4*Br*KeyGroup Score bytes, 2*KeyGroup*D Vbf bytes and 4*Br*(D+5) logical carried-state bytes

## t_ca7c4abd1eeb20d3 | math.ledger | 72 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:237 design.math_reshape.ledger.footprint[0].change

9*R*Skv -> 4*R*(2*T+Nseg+1+Ns*(D+1)) bytes; excludes untouched alignment

## t_cbd132b5411a29c5 | math.ledger | 94 chars | 8 declarations

tcg_d128_flat_cp_direct_narrow:192 design.math_reshape.ledger.footprint[0].change; tcg_d128_flat_cp_direct_wide:204 design.math_reshape.ledger.footprint[0].change; tcg_d128_flat_cp_xpose_narrow:199 design.math_reshape.ledger.footprint[0].change; tcg_d128_flat_cp_xpose_wide:182 design.math_reshape.ledger.footprint[0].change; tcg_d128_flat_tma_direct_narrow:193 design.math_reshape.ledger.footprint[0].change; tcg_d128_flat_tma_direct_wide:204 design.math_reshape.ledger.footprint[0].change; tcg_d128_flat_tma_xpose_narrow:196 design.math_reshape.ledger.footprint[0].change; tcg_d128_flat_tma_xpose_wide:211 design.math_reshape.ledger.footprint[0].change

9*R*Skv -> 4*R*(2*T+1+Ns*(D+1)) bytes, excluding unused segment-max and alignment reservations

## t_ced8953596d4aeee | math.ledger | 18 chars | 8 declarations

tcg_d256_flat_cp_narrow:184 design.math_reshape.ledger.matmul[0].item; tcg_d256_flat_cp_wide:198 design.math_reshape.ledger.matmul[0].item; tcg_d256_flat_tma_narrow:184 design.math_reshape.ledger.matmul[0].item; tcg_d256_flat_tma_wide:196 design.math_reshape.ledger.matmul[0].item; tcg_d256_segmented_cp_narrow:191 design.math_reshape.ledger.matmul[0].item; tcg_d256_segmented_cp_wide:237 design.math_reshape.ledger.matmul[0].item; tcg_d256_segmented_tma_narrow:206 design.math_reshape.ledger.matmul[0].item; tcg_d256_segmented_tma_wide:191 design.math_reshape.ledger.matmul[0].item

Dense contractions

## t_d02ab2efa9d0bba0 | math.ledger | 6 chars | 50 declarations

fused_d128_cp:168 design.math_reshape.ledger.matmul[0].by[0]; fused_d128_cp:170 design.math_reshape.ledger.depth[0].by[0]; fused_d128_group:167 design.math_reshape.ledger.matmul[0].by[0]; fused_d128_group:169 design.math_reshape.ledger.depth[0].by[0]; fused_d128_key:170 design.math_reshape.ledger.matmul[0].by[0]; fused_d128_key:172 design.math_reshape.ledger.depth[0].by[0]; fused_d256_cp:164 design.math_reshape.ledger.matmul[0].by[0]; fused_d256_cp:166 design.math_reshape.ledger.depth[0].by[0]; fused_d256_tma:164 design.math_reshape.ledger.matmul[0].by[0]; fused_d256_tma:166 design.math_reshape.ledger.depth[0].by[0]; root:378 design.math_reshape.ledger.matmul[0].by[0]; root:380 design.math_reshape.ledger.depth[0].by[0]; tc_fused:169 design.math_reshape.ledger.matmul[0].by[0]; tc_fused:173 design.math_reshape.ledger.depth[0].by[0]; tc_split_flat_narrow:250 design.math_reshape.ledger.matmul[0].by[0]; tc_split_flat_wide:262 design.math_reshape.ledger.matmul[0].by[0]; tc_split_segmented_narrow:285 design.math_reshape.ledger.matmul[0].by[0]; tc_split_segmented_wide:313 design.math_reshape.ledger.matmul[0].by[0]; tcg_d128_flat_cp_direct_narrow:196 design.math_reshape.ledger.matmul[0].by[0]; tcg_d128_flat_cp_direct_wide:208 design.math_reshape.ledger.matmul[0].by[0]; tcg_d128_flat_cp_xpose_narrow:203 design.math_reshape.ledger.matmul[0].by[0]; tcg_d128_flat_cp_xpose_wide:186 design.math_reshape.ledger.matmul[0].by[0]; tcg_d128_flat_tma_direct_narrow:197 design.math_reshape.ledger.matmul[0].by[0]; tcg_d128_flat_tma_direct_wide:209 design.math_reshape.ledger.matmul[0].by[0]; tcg_d128_flat_tma_xpose_narrow:200 design.math_reshape.ledger.matmul[0].by[0]; tcg_d128_flat_tma_xpose_wide:215 design.math_reshape.ledger.matmul[0].by[0]; tcg_d128_fscan_cp_direct_narrow:207 design.math_reshape.ledger.matmul[0].by[0]; tcg_d128_fscan_cp_direct_wide:222 design.math_reshape.ledger.matmul[0].by[0]; tcg_d128_fscan_cp_xpose_narrow:210 design.math_reshape.ledger.matmul[0].by[0]; tcg_d128_fscan_cp_xpose_wide:224 design.math_reshape.ledger.matmul[0].by[0]; tcg_d128_fscan_tma_direct_narrow:207 design.math_reshape.ledger.matmul[0].by[0]; tcg_d128_fscan_tma_direct_wide:223 design.math_reshape.ledger.matmul[0].by[0]; tcg_d128_fscan_tma_xpose_narrow:210 design.math_reshape.ledger.matmul[0].by[0]; tcg_d128_fscan_tma_xpose_wide:224 design.math_reshape.ledger.matmul[0].by[0]; tcg_d128_segmented_cp_direct_narrow:218 design.math_reshape.ledger.matmul[0].by[0]; tcg_d128_segmented_cp_direct_wide:231 design.math_reshape.ledger.matmul[0].by[0]; tcg_d128_segmented_cp_xpose_narrow:208 design.math_reshape.ledger.matmul[0].by[0]; tcg_d128_segmented_cp_xpose_wide:241 design.math_reshape.ledger.matmul[0].by[0]; tcg_d128_segmented_tma_direct_narrow:218 design.math_reshape.ledger.matmul[0].by[0]; tcg_d128_segmented_tma_direct_wide:234 design.math_reshape.ledger.matmul[0].by[0]; tcg_d128_segmented_tma_xpose_narrow:224 design.math_reshape.ledger.matmul[0].by[0]; tcg_d128_segmented_tma_xpose_wide:208 design.math_reshape.ledger.matmul[0].by[0]; tcg_d256_flat_cp_narrow:184 design.math_reshape.ledger.matmul[0].by[0]; tcg_d256_flat_cp_wide:198 design.math_reshape.ledger.matmul[0].by[0]; tcg_d256_flat_tma_narrow:184 design.math_reshape.ledger.matmul[0].by[0]; tcg_d256_flat_tma_wide:196 design.math_reshape.ledger.matmul[0].by[0]; tcg_d256_segmented_cp_narrow:191 design.math_reshape.ledger.matmul[0].by[0]; tcg_d256_segmented_cp_wide:237 design.math_reshape.ledger.matmul[0].by[0]; tcg_d256_segmented_tma_narrow:206 design.math_reshape.ledger.matmul[0].by[0]; tcg_d256_segmented_tma_wide:191 design.math_reshape.ledger.matmul[0].by[0]

domain

## t_d17eff8de6d5f6a1 | math.ledger | 21 chars | 4 declarations

tcg_d128_fscan_cp_xpose_narrow:209 design.math_reshape.ledger.matmul[0].item; tcg_d128_fscan_cp_xpose_wide:223 design.math_reshape.ledger.matmul[0].item; tcg_d128_fscan_tma_xpose_wide:223 design.math_reshape.ledger.matmul[0].item; tcg_d128_segmented_cp_xpose_wide:240 design.math_reshape.ledger.matmul[0].item

Recomputed score pass

## t_d193882ad9a99204 | math.ledger | 6 chars | 24 declarations

tcg_d256_flat_cp_narrow:186 design.math_reshape.ledger.depth[0].by[2]; tcg_d256_flat_cp_narrow:188 design.math_reshape.ledger.traffic[0].by[1]; tcg_d256_flat_cp_narrow:190 design.math_reshape.ledger.error[0].by[3]; tcg_d256_flat_cp_wide:200 design.math_reshape.ledger.depth[0].by[2]; tcg_d256_flat_cp_wide:202 design.math_reshape.ledger.traffic[0].by[1]; tcg_d256_flat_cp_wide:204 design.math_reshape.ledger.error[0].by[3]; tcg_d256_flat_tma_narrow:186 design.math_reshape.ledger.depth[0].by[2]; tcg_d256_flat_tma_narrow:188 design.math_reshape.ledger.traffic[0].by[1]; tcg_d256_flat_tma_narrow:190 design.math_reshape.ledger.error[0].by[3]; tcg_d256_flat_tma_wide:198 design.math_reshape.ledger.depth[0].by[2]; tcg_d256_flat_tma_wide:200 design.math_reshape.ledger.traffic[0].by[1]; tcg_d256_flat_tma_wide:202 design.math_reshape.ledger.error[0].by[3]; tcg_d256_segmented_cp_narrow:193 design.math_reshape.ledger.depth[0].by[2]; tcg_d256_segmented_cp_narrow:196 design.math_reshape.ledger.traffic[1].by[1]; tcg_d256_segmented_cp_narrow:198 design.math_reshape.ledger.error[0].by[3]; tcg_d256_segmented_cp_wide:239 design.math_reshape.ledger.depth[0].by[2]; tcg_d256_segmented_cp_wide:242 design.math_reshape.ledger.traffic[1].by[1]; tcg_d256_segmented_cp_wide:244 design.math_reshape.ledger.error[0].by[3]; tcg_d256_segmented_tma_narrow:208 design.math_reshape.ledger.depth[0].by[2]; tcg_d256_segmented_tma_narrow:211 design.math_reshape.ledger.traffic[1].by[1]; tcg_d256_segmented_tma_narrow:213 design.math_reshape.ledger.error[0].by[3]; tcg_d256_segmented_tma_wide:193 design.math_reshape.ledger.depth[0].by[2]; tcg_d256_segmented_tma_wide:196 design.math_reshape.ledger.traffic[1].by[1]; tcg_d256_segmented_tma_wide:198 design.math_reshape.ledger.error[0].by[3]

output

## t_d235b7fecff07fdf | math.ledger | 274 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:243 design.math_reshape.ledger.depth[0].change

T -> max_s Count(Ns,s) group steps per numerator; per row the scalar scan walks WalkCount(r) groups, tests 16*ceil_div(Nseg-1,16) segment bounds and retains all T stores; combine requires at most ceil_div(Ns,MergeGroups)+MergeGroups dependent numerator additions per channel

## t_d42a6777ff655c16 | math.ledger | 255 chars | 6 declarations

fused_d128_cp:160 design.math_reshape.ledger.baseline; fused_d128_group:159 design.math_reshape.ledger.baseline; fused_d128_key:162 design.math_reshape.ledger.baseline; fused_d256_cp:156 design.math_reshape.ledger.baseline; fused_d256_tma:156 design.math_reshape.ledger.baseline; root:370 design.math_reshape.ledger.baseline

Direct Contract evaluation materializing the full FP32 scores and probabilities and E4M3 probabilities before the value product; count logical payload bytes. Worktile figures below explicitly include the source's Br-row padding and useful-group traversal.

## t_ddcadaae542e53f8 | math.ledger | 195 chars | 1 declarations

tcg_d128_segmented_cp_direct_wide:233 design.math_reshape.ledger.depth[0].change

T -> max_s Count(Ns,s) group steps; scan retains WalkCount(r) ordered group updates, at most T, and combine uses at most Cmerge local additions followed by MergeGroups final additions per channel

## t_df23cd532c5a9f91 | math.ledger | 15 chars | 8 declarations

tcg_d256_flat_cp_narrow:188 design.math_reshape.ledger.traffic[0].item; tcg_d256_flat_cp_wide:202 design.math_reshape.ledger.traffic[0].item; tcg_d256_flat_tma_narrow:188 design.math_reshape.ledger.traffic[0].item; tcg_d256_flat_tma_wide:200 design.math_reshape.ledger.traffic[0].item; tcg_d256_segmented_cp_narrow:196 design.math_reshape.ledger.traffic[1].item; tcg_d256_segmented_cp_wide:242 design.math_reshape.ledger.traffic[1].item; tcg_d256_segmented_tma_narrow:211 design.math_reshape.ledger.traffic[1].item; tcg_d256_segmented_tma_wide:196 design.math_reshape.ledger.traffic[1].item

Partial handoff

## t_e28652e221cdf892 | math.ledger | 184 chars | 4 declarations

tcg_d128_fscan_cp_xpose_narrow:203 design.math_reshape.ledger.baseline; tcg_d128_fscan_cp_xpose_wide:216 design.math_reshape.ledger.baseline; tcg_d128_fscan_tma_xpose_wide:217 design.math_reshape.ledger.baseline; tcg_d128_segmented_cp_xpose_wide:234 design.math_reshape.ledger.baseline

Direct Contract evaluation materializing FP32 scores and probabilities plus E4M3 probabilities before PV; logical payload bytes, with padding and useful-group traversal included below.

## t_e4a88749c688bbd7 | math.ledger | 185 chars | 1 declarations

tc_fused:175 design.math_reshape.ledger.error[0].change

Blockwise post-dot Q/K scaling, BF16 V scaling, direct carried PV accumulation and final-only denominator reduction replace reference evaluation choices; no end-to-end bound is asserted

## t_e826d9361ed8f359 | math.ledger | 24 chars | 1 declarations

tc_fused:167 design.math_reshape.ledger.footprint[0].item

Scores and probabilities

## t_e90da07d03e4075b | math.ledger | 150 chars | 2 declarations

fused_d256_cp:162 design.math_reshape.ledger.footprint[0].change; fused_d256_tma:162 design.math_reshape.ledger.footprint[0].change

9*B*H*Sq*Skv bytes -> none globally; simultaneous per-worktile Score and Prob8 capacity is 5*Br*KeyGroup bytes, plus 4*Br*(D+3) bytes of carried state

## t_eabe0f8e28fff5a3 | math.ledger | 20 chars | 32 declarations

tcg_d128_flat_cp_direct_narrow:198 design.math_reshape.ledger.depth[0].item; tcg_d128_flat_cp_direct_wide:210 design.math_reshape.ledger.depth[0].item; tcg_d128_flat_cp_xpose_narrow:205 design.math_reshape.ledger.depth[0].item; tcg_d128_flat_cp_xpose_wide:188 design.math_reshape.ledger.depth[0].item; tcg_d128_flat_tma_direct_narrow:199 design.math_reshape.ledger.depth[0].item; tcg_d128_flat_tma_direct_wide:211 design.math_reshape.ledger.depth[0].item; tcg_d128_flat_tma_xpose_narrow:202 design.math_reshape.ledger.depth[0].item; tcg_d128_flat_tma_xpose_wide:217 design.math_reshape.ledger.depth[0].item; tcg_d128_fscan_cp_direct_narrow:209 design.math_reshape.ledger.depth[0].item; tcg_d128_fscan_cp_direct_wide:224 design.math_reshape.ledger.depth[0].item; tcg_d128_fscan_cp_xpose_narrow:212 design.math_reshape.ledger.depth[0].item; tcg_d128_fscan_cp_xpose_wide:226 design.math_reshape.ledger.depth[0].item; tcg_d128_fscan_tma_direct_narrow:209 design.math_reshape.ledger.depth[0].item; tcg_d128_fscan_tma_direct_wide:225 design.math_reshape.ledger.depth[0].item; tcg_d128_fscan_tma_xpose_narrow:212 design.math_reshape.ledger.depth[0].item; tcg_d128_fscan_tma_xpose_wide:226 design.math_reshape.ledger.depth[0].item; tcg_d128_segmented_cp_direct_narrow:220 design.math_reshape.ledger.depth[0].item; tcg_d128_segmented_cp_direct_wide:233 design.math_reshape.ledger.depth[0].item; tcg_d128_segmented_cp_xpose_narrow:210 design.math_reshape.ledger.depth[0].item; tcg_d128_segmented_cp_xpose_wide:243 design.math_reshape.ledger.depth[0].item; tcg_d128_segmented_tma_direct_narrow:220 design.math_reshape.ledger.depth[0].item; tcg_d128_segmented_tma_direct_wide:236 design.math_reshape.ledger.depth[0].item; tcg_d128_segmented_tma_xpose_narrow:226 design.math_reshape.ledger.depth[0].item; tcg_d128_segmented_tma_xpose_wide:210 design.math_reshape.ledger.depth[0].item; tcg_d256_flat_cp_narrow:186 design.math_reshape.ledger.depth[0].item; tcg_d256_flat_cp_wide:200 design.math_reshape.ledger.depth[0].item; tcg_d256_flat_tma_narrow:186 design.math_reshape.ledger.depth[0].item; tcg_d256_flat_tma_wide:198 design.math_reshape.ledger.depth[0].item; tcg_d256_segmented_cp_narrow:193 design.math_reshape.ledger.depth[0].item; tcg_d256_segmented_cp_wide:239 design.math_reshape.ledger.depth[0].item; tcg_d256_segmented_tma_narrow:208 design.math_reshape.ledger.depth[0].item; tcg_d256_segmented_tma_wide:193 design.math_reshape.ledger.depth[0].item

Numerator recurrence

## t_ecaf560d12bddd69 | math.ledger | 68 chars | 1 declarations

tcg_d256_segmented_cp_wide:235 design.math_reshape.ledger.footprint[0].change

9*R*Skv -> 4*R*(2*T+Nseg+1+Ns*(D+1)) bytes; no global Score or Prob8

## t_efc1f2fef2bfda78 | math.ledger | 36 chars | 1 declarations

tc_fused:169 design.math_reshape.ledger.matmul[0].item

Products including padding and masks

## t_f09db95e1a8992f4 | math.ledger | 107 chars | 1 declarations

tcg_d128_fscan_tma_xpose_narrow:206 design.math_reshape.ledger.footprint[0].change

9*R*Skv -> 4*R*(T+1+Ns*(D+2)) bytes, excluding unused scan-snapshot, segment-max and alignment reservations

## t_f1ed99d81e3fdbed | math.ledger | 212 chars | 1 declarations

tcg_d128_fscan_cp_direct_wide:224 design.math_reshape.ledger.depth[0].change

T -> max_s Count(Ns,s) numerator steps; each split also walks Tw(s) prefix maxima, with a max(8,Cmerge)-term local numerator merge chain including first-batch padding, then MergeGroups final additions per channel

## t_f6ef1bd6275b5ca0 | math.ledger | 25 chars | 20 declarations

tcg_d128_flat_cp_direct_narrow:195 design.math_reshape.ledger.matmul[0].item; tcg_d128_flat_cp_direct_wide:207 design.math_reshape.ledger.matmul[0].item; tcg_d128_flat_cp_xpose_narrow:202 design.math_reshape.ledger.matmul[0].item; tcg_d128_flat_cp_xpose_wide:185 design.math_reshape.ledger.matmul[0].item; tcg_d128_flat_tma_direct_narrow:196 design.math_reshape.ledger.matmul[0].item; tcg_d128_flat_tma_direct_wide:208 design.math_reshape.ledger.matmul[0].item; tcg_d128_flat_tma_xpose_narrow:199 design.math_reshape.ledger.matmul[0].item; tcg_d128_flat_tma_xpose_wide:214 design.math_reshape.ledger.matmul[0].item; tcg_d128_fscan_cp_direct_narrow:206 design.math_reshape.ledger.matmul[0].item; tcg_d128_fscan_cp_direct_wide:221 design.math_reshape.ledger.matmul[0].item; tcg_d128_fscan_tma_direct_narrow:206 design.math_reshape.ledger.matmul[0].item; tcg_d128_fscan_tma_direct_wide:222 design.math_reshape.ledger.matmul[0].item; tcg_d128_fscan_tma_xpose_narrow:209 design.math_reshape.ledger.matmul[0].item; tcg_d128_segmented_cp_direct_narrow:217 design.math_reshape.ledger.matmul[0].item; tcg_d128_segmented_cp_direct_wide:230 design.math_reshape.ledger.matmul[0].item; tcg_d128_segmented_cp_xpose_narrow:207 design.math_reshape.ledger.matmul[0].item; tcg_d128_segmented_tma_direct_narrow:217 design.math_reshape.ledger.matmul[0].item; tcg_d128_segmented_tma_direct_wide:233 design.math_reshape.ledger.matmul[0].item; tcg_d128_segmented_tma_xpose_narrow:223 design.math_reshape.ledger.matmul[0].item; tcg_d128_segmented_tma_xpose_wide:207 design.math_reshape.ledger.matmul[0].item

Repeated score evaluation

## t_f71eb4b8b8c6e412 | math.ledger | 239 chars | 1 declarations

tc_split_segmented_wide:333 design.math_reshape.ledger.error[0].change

Post-dot scale correction, BF16 V, four denominator partials, split weighting and grouped numerator/denominator sums change reference rounding; uncovered rows additionally depend on incoming scratch, without a general equivalence guarantee

## t_f73b1521b2e208ce | math.ledger | 22 chars | 6 declarations

fused_d128_cp:170 design.math_reshape.ledger.depth[0].item; fused_d128_group:169 design.math_reshape.ledger.depth[0].item; fused_d128_key:172 design.math_reshape.ledger.depth[0].item; fused_d256_cp:166 design.math_reshape.ledger.depth[0].item; fused_d256_tma:166 design.math_reshape.ledger.depth[0].item; root:380 design.math_reshape.ledger.depth[0].item

Group state dependence

## t_fcc8263106283e79 | math.ledger | 100 chars | 4 declarations

tcg_d128_fscan_cp_xpose_narrow:209 design.math_reshape.ledger.matmul[0].change; tcg_d128_fscan_cp_xpose_wide:223 design.math_reshape.ledger.matmul[0].change; tcg_d128_fscan_tma_xpose_wide:223 design.math_reshape.ledger.matmul[0].change; tcg_d128_segmented_cp_xpose_wide:240 design.math_reshape.ledger.matmul[0].change

Per (b,kh,a), 2*U*Skv*D -> 3*Br*KeyGroup*Span*D MACs, including padding and masked boundary products

## t_fd278440d03451da | math.ledger | 124 chars | 2 declarations

tcg_d256_flat_cp_narrow:186 design.math_reshape.ledger.depth[0].change; tcg_d256_flat_tma_narrow:186 design.math_reshape.ledger.depth[0].change

Ng -> at most ceil_div(Ng,Ns) steps per split plus Ns ordered combine additions; a separate GCount-step maximum scan remains

## t_06a02daf9592d933 | math.math | 941 chars | 1 declarations

tcg_d128_fscan_cp_xpose_narrow:178 design.math_reshape.rewrite[5].math

For each real row r and 0<=s<Ns let Weight[s]= expf(PartM[s,r]-MTail[r]) when both maxima are finite, otherwise 0. Accumulate PartA[s,r,d]*Weight[s] and PartB[s,r]*Weight[s] from zero by increasing s in FP32 to obtain TotalA[d] and TotalB, retaining the source's scalar compiler path. All Ns entries are read, including empty splits even when their PartB is zero; zero weights do not suppress NaN or Inf in PartA. The eight-at-a-time loop adds zero for missing entries in its final batch. Set live=(TotalB>0), inv=1/TotalB when live, otherwise 0. Write O[b,h,i,d]=bf16(TotalA[d]*inv) when live, otherwise bf16(0), with RNE. If LSE is present, write MTail[r]+logf(TotalB/16) when live and MTail[r] is finite, otherwise -inf. Initial O and LSE bytes are never read; all valid destinations are written. The host returns None. Workspace retains intermediates without a mathematical output requirement; no scratch zero initialization is required.

## t_06fb4a04f432c9f4 | math.math | 1587 chars | 8 declarations

tcg_d128_flat_cp_direct_narrow:103 design.math_reshape.rewrite[3].math; tcg_d128_flat_cp_direct_wide:103 design.math_reshape.rewrite[3].math; tcg_d128_flat_tma_direct_narrow:104 design.math_reshape.rewrite[3].math; tcg_d128_flat_tma_direct_wide:104 design.math_reshape.rewrite[3].math; tcg_d128_segmented_cp_direct_narrow:125 design.math_reshape.rewrite[4].math; tcg_d128_segmented_cp_direct_wide:124 design.math_reshape.rewrite[4].math; tcg_d128_segmented_tma_direct_narrow:125 design.math_reshape.rewrite[4].math; tcg_d128_segmented_tma_direct_wide:125 design.math_reshape.rewrite[4].math

For a numerator split let cLo=L(Ns,s), cHi=Hlast(Ns,s), end=max(cLo-1,cHi). Initialize DenState(cLo-1)[u]=0 for u<C. In a nonempty split initialize MaxState(cLo-1)[u] from MState[cLo-1,r] when cLo>0, otherwise -inf. For g=cLo,...,cHi set MaxState(g)[u]=MState[g,r], safe[u]=MaxState(g)[u] if finite, otherwise 0, and alpha[u]=expf(MaxState(g-1)[u]-safe[u]) if the old state is finite, otherwise 0. Compute x=Score[u,v]*scale-safe[u] with separate FP32 multiply and subtract. The actual E16 approximation is t=sat01(fma32(x,fp32(0.0057249800302088261),0.5)), w=fma_rm32(t,252,12582917), e=fma32(x,fp32(1.4426950216293335),12583043-w), d=bitcast_f32((bitcast_u32(w)<<23) mod 2^32), E16(x)=d*ex2_ftz(e). fma_rm32 rounds toward negative infinity; sat01 saturates to [0,1] with NaN-to-zero; ex2_ftz is the source approximate base-two exponential with flush-to-zero. No low log2(e) correction is added. Prob[u,v]=E16(x), Prob8[u,v]=satfinite_e4m3_rne(Prob[u,v]), the source hardware saturating RNE cast with unit scale; for nonnegative finite values this rounds min(Prob,ProbabilityMax) to E4M3FN. For chunk c=0,...,3, partition z=0,...,3 and pair p=0,1, let F[c,z,p] be the left-to-right FP32 sum from zero of Prob[u,32*c+8*z+2*j+p], j=0,...,3. Sum (F[c,0,p]+F[c,1,p])+(F[c,2,p]+F[c,3,p]) over increasing c from zero to E[p]. Then DenState(g)[u]=DenState(g-1)[u]*alpha[u]+(E[0]+E[1]). This scalar update uses source compiler defaults without an additional contraction restriction; the denominator never uses Prob8. An empty split has no probability evaluation and leaves its denominator zero.

## t_070e8691f95c67b5 | math.math | 569 chars | 5 declarations

tcg_d256_flat_cp_narrow:161 design.math_reshape.rewrite[5].math; tcg_d256_flat_tma_narrow:161 design.math_reshape.rewrite[5].math; tcg_d256_segmented_cp_narrow:168 design.math_reshape.rewrite[6].math; tcg_d256_segmented_tma_narrow:183 design.math_reshape.rewrite[6].math; tcg_d256_segmented_tma_wide:168 design.math_reshape.rewrite[6].math

For each row/channel, initialize OutNum=0 and OutDen=0. Add PartA and PartB in ascending split order, in batches of eight, adding zero for missing positions in the final batch. Let live=OutDen>0 and inv=1/OutDen when live, otherwise 0. O[b,h,i,d]=bf16(OutNum*inv) when live, otherwise bf16(0), using RNE. When LSE is present, LSE[b,h,i]=Mtail[row]+logf(OutDen) if live and Mtail is finite, otherwise -inf. Initial O/LSE contents are never read. Workspace is overwritten with the intermediates; unrelated scratch bytes have no mathematical result. The host returns None.

## t_0b4c29b41facd2b5 | math.math | 1210 chars | 1 declarations

tcg_d128_flat_tma_direct_wide:172 design.math_reshape.rewrite[5].math

For each real row r let Cmerge=ceil_div(Ns,MergeGroups). For merge group z=0,...,MergeGroups-1 set sLo(z)=min(z*Cmerge,Ns), sHi(z)=min(sLo(z)+Cmerge,Ns). GroupA[z,d] is the left-to-right FP32 sum from zero of PartA[s,r,d] for s=sLo(z),...,sHi(z)-1, without extra tail zeros. For subgroup w=0,...,3 form x[l]=PartB[sLo(z)+32*w+l,r] when sLo(z)+32*w+l<sHi(z), otherwise zero, for l=0,...,31. Reduce by FP32 halving offsets 16,8,4,2,1: at each offset replace x[l] by x[l]+x[l+offset] for l=0,...,offset-1. DenChunk[z,w]=x[0]. TotalB is the left-to-right FP32 sum from zero of DenChunk[z,w], ordered first by increasing z, then increasing w. TotalA[d] is the left-to-right FP32 sum from zero of GroupA[z,d] by increasing z. Let live=(TotalB>0), inv=1/TotalB when live, otherwise 0. Write O[b,h,i,d]=bf16(TotalA[d]*inv) when live, otherwise bf16(0), using RNE. When LSE is present write MTail[r]+logf(TotalB/16) if live and MTail[r] is finite, otherwise -inf. Initial destination bytes are not read. All valid caller destinations are written and the host returns None. Workspace retains intermediates; no mathematical scratch output or scratch-zero initialization is required. The segment-max reservation is unused.

## t_0ebbcc3e4608c774 | math.math | 314 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:92 design.math_reshape.rewrite[2].math

Let Nseg=ceil_div(T,SegLen). For 0<=z<Nseg and each row r, initialize mx=-inf and fold fmaxf(mx,GroupMax[z*SegLen+k,r]) in increasing k=0,...,SegLen-1, using -inf when z*SegLen+k>=T. Store SegMax[z,r]=mx. fmaxf selects the non-NaN operand when only one operand is NaN. SegMax[0,r] is materialized but not consumed.

## t_0f7029fa5ce807be | math.math | 802 chars | 4 declarations

tcg_d128_fscan_cp_xpose_narrow:67 design.math_reshape.rewrite[1].math; tcg_d128_fscan_cp_xpose_wide:67 design.math_reshape.rewrite[1].math; tcg_d128_fscan_tma_xpose_wide:67 design.math_reshape.rewrite[1].math; tcg_d128_segmented_cp_xpose_wide:69 design.math_reshape.rewrite[1].math

On each evaluated group, initialize A[-1]=0 by overwrite and, for k=0,...,3, A[k]=mma32(Qb[:,32*k:32*k+32],Kb[:,32*k:32*k+32]^T,A[k-1]). DotTile=A[3]. For u<Br, Score[u,v]=DotTile[u,v] when lo(u)<=g*KeyGroup+v<=hi(u), otherwise -inf. Let scale=fp32(D^(-1/2)). GroupMax[g,r]=scale*fmax(Score[u,:]) for valid rows, reducing from -inf. The source uses fmaxf behavior for NaN: masked chunks use a 32-element halving tree then combine four chunk maxima; wholly unmasked readout can use native 32-value max reduction then fmaxf across chunks. GroupMax[g,r]=-inf outside gLoAll..gHiAll, filled by the first/last group-max split. Empty splits produce no interior entries. The numerator pass recomputes DotTile with the same contractions but does not recompute maxima; each score chunk is consumed by normalize.

## t_0fc9c8ac179872ad | math.math | 1418 chars | 4 declarations

tcg_d128_fscan_cp_direct_narrow:107 design.math_reshape.rewrite[3].math; tcg_d128_fscan_cp_direct_wide:107 design.math_reshape.rewrite[3].math; tcg_d128_fscan_tma_direct_narrow:107 design.math_reshape.rewrite[3].math; tcg_d128_fscan_tma_direct_wide:107 design.math_reshape.rewrite[3].math

Initialize DenState(cLo-1)[u]=0 for u<C. For g=cLo,...,cHi, use the scan step's MaxState and set safe[u]=MaxState(g)[u] if finite, otherwise 0, and alpha[u]=expf(MaxState(g-1)[u]-safe[u]) if the old state is finite, otherwise 0. Compute x=Score[u,v]*scale-safe[u] with separate FP32 multiply and subtract. The actual E16 approximation is t=sat01(fma32(x,fp32(0.0057249800302088261),0.5)), w=fma_rm32(t,252,12582917), e=fma32(x,fp32(1.4426950216293335),12583043-w), d=bitcast_f32((bitcast_u32(w)<<23) mod 2^32), E16(x)=d*ex2_ftz(e). fma_rm32 rounds toward negative infinity; sat01 saturates to [0,1] with NaN-to-zero; ex2_ftz is the source approximate base-two exponential with flush-to-zero. No low log2(e) correction is added. Prob[u,v]=E16(x), Prob8[u,v]=satfinite_e4m3_rne(Prob[u,v]), the source hardware saturating RNE cast with unit scale; for nonnegative finite values this rounds min(Prob,ProbabilityMax) to E4M3FN. For chunk c=0,...,3, partition z=0,...,3 and pair p=0,1, let F[c,z,p] be the left-to-right FP32 sum from zero of Prob[u,32*c+8*z+2*j+p], j=0,...,3. Sum (F[c,0,p]+F[c,1,p])+(F[c,2,p]+F[c,3,p]) over increasing c from zero to E[p]. Then DenState(g)[u]=DenState(g-1)[u]*alpha[u]+(E[0]+E[1]). This scalar update uses source compiler defaults without an additional contraction restriction; the denominator never uses Prob8. An empty split has no probability evaluation and leaves its denominator zero.

## t_10f9f68d17edfbfb | math.math | 1681 chars | 1 declarations

tcg_d128_fscan_tma_direct_wide:174 design.math_reshape.rewrite[5].math

For each real row r let Chunk=ceil_div(Ns,MergeGroups), Slo(p)=min(p*Chunk,Ns), Shi(p)=min(Slo(p)+Chunk,Ns), MergeSize(p)=Shi(p)-Slo(p), for p=0,...,MergeGroups-1. For 0<=x<MergeSize(p), Weight[p,x]=expf(PartM[Slo(p)+x,r]-MTail[r]) when both maxima are finite, otherwise 0. For MergeSize(p)<=x<128, set Weight[p,x]=0 without reading PartM. GroupA[p,d] accumulates PartA[s,r,d]*Weight[p,s-Slo(p)] from zero in increasing s over Slo(p)<=s<Shi(p), with scalar FP32 multiply-add evaluation. The first eight-entry batch adds zero for absent entries without reading weights or evaluating products. Later complete batches and the scalar remainder preserve increasing split order. For x=0,...,127, let B0[p,x]=PartB[Slo(p)+x,r]*Weight[p,x] when x<MergeSize(p), otherwise 0*Weight[p,x]. For w=0,...,3 initialize MergeTree[l]=B0[p,32*w+l], l=0,...,31; at offsets 16,8,4,2,1 simultaneously replace MergeTree[l] by MergeTree[l]+MergeTree[l+offset] wherever l+offset<32. GroupB[p,w]=MergeTree[0]. Empty groups produce zero GroupA and GroupB. TotalA[d] sums GroupA[p,d] from zero in increasing p. TotalB sums GroupB[p,w] from zero with p outermost and w innermost, both increasing. Let live=(TotalB>0), inv=1/TotalB when live, otherwise 0. Write O[b,h,i,d]=bf16(TotalA[d]*inv) when live, otherwise bf16(0), using RNE. When LSE is present write MTail[r]+logf(TotalB/16) if live and MTail[r] is finite, otherwise -inf. Initial destination bytes are not read. All valid caller destinations are written and the host returns None. Workspace retains intermediates; no mathematical scratch output or scratch-zero initialization is required. Segment-max and per-group scan-state reservations are unused.

## t_11c88ff8e9e6ad1f | math.math | 1632 chars | 1 declarations

tc_split_segmented_wide:254 design.math_reshape.rewrite[7].math

Let Chunk=ceil_div(Ns,CombineGroups). For sg=0,...,CombineGroups-1, sLo(sg)=min(sg*Chunk,Ns), sHi(sg)=min(sLo(sg)+Chunk,Ns), Count(sg)=sHi(sg)-sLo(sg). For every row r and channel d, initialize GroupA(-1)[sg,r,d]=0. For j=0,...,Count(sg)-1, GroupA(j)[sg,r,d]=GroupA(j-1)[sg,r,d]+PartialA[sLo(sg)+j,r,d]. These are ascending FP32 additions without a zero-padded numerator tail. Retain GroupFinalA[sg,r,d]=GroupA(Count(sg)-1)[sg,r,d]. For w=0,...,CombineBt/32-1 and l=0,...,31 let q=sLo(sg)+32*w+l. Initialize F(-1)[sg,r,w,l]=PartialB[q,r] if q<sHi(sg), otherwise 0. For t=0,...,4 let off=16/2^t and, simultaneously over l, set F(t)[sg,r,w,l]=F(t-1)[sg,r,w,l]+F(t-1)[sg,r,w,l2], where l2=l+off when l+off<32, otherwise l. Retain GroupB[sg,r,w]=F(4)[sg,r,w,0]; discard the other components. Let Npartial=CombineGroups*(CombineBt/32). Initialize SumB(-1)[r]=0. For v=0,...,Npartial-1, in ascending order, SumB(v)[r]=SumB(v-1)[r]+GroupB[floor(v/(CombineBt/32)),r,v mod (CombineBt/32)]. Initialize SumA(-1)[r,d]=0. For sg=0,...,CombineGroups-1, SumA(sg)[r,d]=SumA(sg-1)[r,d]+GroupFinalA[sg,r,d]. All additions above are FP32. Set total=SumB(Npartial-1)[r], live=(total>0), inv=1/total if live, otherwise 0. Write O[b,h,i,d]=bf16(SumA(CombineGroups-1)[r,d]*inv) if live, otherwise bf16(0), using RNE, where r=(b*H+h)*Sq+i. When LSE is present, write MaxTail[r]+logf(total/beta) if live and MaxTail[r] is finite, otherwise -inf. No initial O/LSE values are read. GroupFinalA and GroupB die after their final sums. PartialA, PartialB and MaxTail die after the last consumer. Workspace retains its written bytes; the Python call returns None.

## t_14bae231b0ad8c89 | math.math | 363 chars | 3 declarations

tcg_d128_segmented_cp_xpose_narrow:74 design.math_reshape.rewrite[2].math; tcg_d128_segmented_tma_xpose_narrow:90 design.math_reshape.rewrite[2].math; tcg_d128_segmented_tma_xpose_wide:74 design.math_reshape.rewrite[2].math

Let Nseg=ceil_div(T,SegLen). For each row r and segment z=0,...,Nseg-1, start mx=-inf and update mx=fmaxf(mx,SegmentInput[k]) for k=0,...,SegLen-1, where SegmentInput[k]=GroupMax[z*SegLen+k,r] when z*SegLen+k<T, otherwise -inf. Store SegMax[z,r]=mx. fmaxf selects the non-NaN operand when only one operand is NaN. SegMax[0,r] is stored but not consumed by replay.

## t_1d5bebc33bd51e20 | math.math | 993 chars | 4 declarations

tcg_d128_segmented_cp_direct_narrow:102 design.math_reshape.rewrite[3].math; tcg_d128_segmented_cp_direct_wide:101 design.math_reshape.rewrite[3].math; tcg_d128_segmented_tma_direct_narrow:102 design.math_reshape.rewrite[3].math; tcg_d128_segmented_tma_direct_wide:102 design.math_reshape.rewrite[3].math

Define update(x,m)=x if x-m>fp32(2.7725887222397812), otherwise m; the comparison is false for NaN. For each row r initialize ScanState(-1)=-inf. For g=0,...,min(Seg,T)-1 set ScanState(g)=update(GroupMax[g,r],ScanState(g-1)). Then process c=0,...,ceil_div(Nseg-1,SegBatch)-1 in order. Let z0=1+c*SegBatch, g0=Seg*z0, g1=min(Seg*(z0+SegBatch),T), incoming=ScanState(g0-1), and sx_batch[j]=SegMax[z0+j,r] when z0+j<Nseg, otherwise -inf, for j=0,...,SegBatch-1. Let Jump={j in integers | 0<=j<SegBatch and sx_batch[j]-incoming>fp32(2.7725887222397812)}. Cut(c,r)=g1 when Jump is empty, otherwise min(g0+Seg*min(Jump),g1). For g=g0,...,Cut(c,r)-1 set ScanState(g)=incoming without reading GroupMax. For g=Cut(c,r),...,g1-1 set ScanState(g)=update(GroupMax[g,r],ScanState(g-1)). Later segments in that suffix are not retested against updated states. Store MState[g,r]=ScanState(g) for every g=0,...,T-1 and MTail[r]=ScanState(T-1). Let WalkCount(r)=min(Seg,T)+sum_c(g1-Cut(c,r)), using each c's g1.

## t_1eba10aeea010950 | math.math | 922 chars | 2 declarations

tcg_d128_fscan_cp_direct_narrow:174 design.math_reshape.rewrite[5].math; tcg_d128_fscan_tma_direct_narrow:174 design.math_reshape.rewrite[5].math

For each real row r and s=0,...,Ns-1, Weight[s]=expf(PartM[s,r]-MTail[r]) when both maxima are finite, otherwise 0. Accumulate PartA[s,r,d]*Weight[s] and PartB[s,r]*Weight[s] from zero in increasing s with the source's scalar FP32 multiply-add evaluation to obtain TotalA[d] and TotalB. The eight-at-a-time loop adds zero for missing final-batch entries without reading their weights or evaluating a product. Let live=(TotalB>0), inv=1/TotalB when live, otherwise 0. Write O[b,h,i,d]=bf16(TotalA[d]*inv) when live, otherwise bf16(0), using RNE. When LSE is present write MTail[r]+logf(TotalB/16) if live and MTail[r] is finite, otherwise -inf. Initial destination bytes are not read. All valid caller destinations are written and the host returns None. Workspace retains intermediates; no mathematical scratch output or scratch-zero initialization is required. Segment-max and per-group scan-state reservations are unused.

## t_23ce222f5604dd61 | math.math | 287 chars | 8 declarations

tcg_d128_flat_cp_direct_narrow:89 design.math_reshape.rewrite[2].math; tcg_d128_flat_cp_direct_wide:89 design.math_reshape.rewrite[2].math; tcg_d128_flat_cp_xpose_narrow:91 design.math_reshape.rewrite[2].math; tcg_d128_flat_cp_xpose_wide:74 design.math_reshape.rewrite[2].math; tcg_d128_flat_tma_direct_narrow:90 design.math_reshape.rewrite[2].math; tcg_d128_flat_tma_direct_wide:90 design.math_reshape.rewrite[2].math; tcg_d128_flat_tma_xpose_narrow:91 design.math_reshape.rewrite[2].math; tcg_d128_flat_tma_xpose_wide:90 design.math_reshape.rewrite[2].math

For each row r, ScanState(-1)=-inf. For g=0,...,T-1, ScanState(g)=GroupMax[g,r] if GroupMax[g,r]-ScanState(g-1)> fp32(2.7725887222397812), otherwise ScanState(g-1). The comparison is false for NaN. Store MState[g,r]=ScanState(g) and MTail[r]=ScanState(T-1), without prefix reassociation.

## t_25741d61b6c6e420 | math.math | 763 chars | 2 declarations

tc_split_flat_narrow:219 design.math_reshape.rewrite[5].math; tc_split_segmented_narrow:254 design.math_reshape.rewrite[7].math

Let PaddedNs=8*ceil_div(Ns,8). For each row r and channel d, SumA(-1)[r,d]=0 and SumB(-1)[r]=0. For s=0,...,PaddedNs-1: SumA(s)[r,d]=SumA(s-1)[r,d]+(PartialA[s,r,d] if s<Ns, otherwise 0), SumB(s)[r]=SumB(s-1)[r]+(PartialB[s,r] if s<Ns, otherwise 0). Add in ascending order in FP32, including the zero tail. Set total=SumB(PaddedNs-1)[r], live=(total>0), inv=1/total if live, otherwise 0. Write O[b,h,i,d]=bf16(SumA(PaddedNs-1)[r,d]*inv) if live, otherwise bf16(0), using RNE, where r=(b*H+h)*Sq+i. When LSE is present, write MaxTail[r]+logf(total/beta) if live and MaxTail[r] is finite, otherwise -inf. No initial O/LSE values are read. PartialA, PartialB and MaxTail die after the last consumer. Workspace retains its written bytes; the Python call returns None.

## t_2c2ed6e872987d12 | math.math | 1215 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:166 design.math_reshape.rewrite[5].math

Initialize NumState(cLo-1)[u,d]=0. For each nonempty split group, C[-1]=0 and C[k]=mma32(Prob8[:,32*k:32*k+32], Vb[32*k:32*k+32,:],C[k-1]), k=0,...,3; GroupPV=C[3]. First rescale the numerator state by alpha unless alpha equals one, then add GroupPV in FP32: NumState(g)=NumState(g-1)*alpha+GroupPV. This is a fresh PV accumulation per group followed by a separate addition, not the fused path's carried native MMA accumulator. For each valid row in a nonempty split let weight= expf(MaxState(cHi)[u]-MTail[r]) when both maxima are finite, otherwise 0. Store PartA[s,r,d]=NumState(cHi)[u,d]*weight and PartB[s,r]=DenState(cHi)[u]*weight, both FP32, without BF16 narrowing. For an empty split, NumState=0, DenState=0 and weight=0 for every valid row, so the evaluated numerator and denominator to be stored are zero. PartB[s,r]=0. The materialized PartA slice is not guaranteed to be zero: an unsynchronized overwrite in the fixed source can corrupt overlapping entries. PartA denotes the actual stored FP32 values, including this empty-split case, rather than the evaluated zeros. MaxState, DenState and NumState are discarded after these stores. No multiplication by zero suppresses NaN or Inf in a nonempty split.

## t_2f06ba5de2da552a | math.math | 800 chars | 3 declarations

tcg_d128_flat_cp_xpose_narrow:177 design.math_reshape.rewrite[5].math; tcg_d128_flat_cp_xpose_wide:160 design.math_reshape.rewrite[5].math; tcg_d128_flat_tma_xpose_narrow:174 design.math_reshape.rewrite[5].math

For each real row r, accumulate PartA[s,r,d] and PartB[s,r] from zero by increasing s with left-to-right FP32 additions, obtaining TotalA[d] and TotalB. Read all Ns partials, including empty splits regardless of PartB being zero. The eight-at-a-time loop also adds zero for missing entries in its final batch. Let live=(TotalB>0), inv=1/TotalB when live, otherwise 0. Write O[b,h,i,d]=bf16(TotalA[d]*inv) when live, otherwise bf16(0), using RNE. When LSE is present write MTail[r]+logf(TotalB/16) if live and MTail[r] is finite, otherwise -inf. Initial destination bytes are not read. All valid caller destinations are written and the host returns None. Workspace retains intermediates; no mathematical scratch output or scratch-zero initialization is required. The segment-max reservation is unused.

## t_2f8302aa2b7ae28c | math.math | 1315 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:202 design.math_reshape.rewrite[6].math

For each real row r, let chunk=ceil_div(Ns,MergeGroups). For m=0,...,MergeGroups-1 let sLo(m)=min(m*chunk,Ns) and sHi(m)=min(sLo(m)+chunk,Ns). MergeA[m,d] is the left-to-right FP32 sum from zero of PartA[s,r,d], s=sLo(m),...,sHi(m)-1. An empty interval gives zero. No extra terms pad this numerator sum. For w=0,...,3 and j=0,...,31 set x[j]=PartB[sLo(m)+32*w+j,r] when sLo(m)+32*w+j<sHi(m), otherwise 0. Reduce x by offsets 16,8,4,2,1: at each offset simultaneously replace x[j] by x[j]+x[j+offset] for 0<=j<offset. Store MergeB[m,w]=x[0]. The host bounds Ns<=128*MergeGroups, so these four trees cover every denominator entry in each interval. TotalA[d] sums MergeA[m,d] from zero by increasing m. TotalB sums MergeB[m,w] from zero by increasing m, then w. All additions are FP32. All Ns partials are read, including empty splits even when their PartB is zero. Set live=(TotalB>0), inv=1/TotalB when live, otherwise 0. Write O[b,h,i,d]=bf16(TotalA[d]*inv) when live, otherwise bf16(0), with RNE. If LSE is present, write MTail[r]+logf(TotalB/16) when live and MTail[r] is finite, otherwise -inf. Initial O and LSE bytes are never read; all valid destinations are written. The host returns None. Workspace retains intermediates without a mathematical output requirement; no scratch zero initialization is required.

## t_36a3eee4f43ab8ca | math.math | 1405 chars | 8 declarations

tcg_d128_flat_cp_xpose_narrow:36 design.math_reshape.rewrite[0].math; tcg_d128_flat_cp_xpose_wide:19 design.math_reshape.rewrite[0].math; tcg_d128_flat_tma_xpose_narrow:36 design.math_reshape.rewrite[0].math; tcg_d128_flat_tma_xpose_wide:35 design.math_reshape.rewrite[0].math; tcg_d128_fscan_tma_xpose_narrow:35 design.math_reshape.rewrite[0].math; tcg_d128_segmented_cp_xpose_narrow:19 design.math_reshape.rewrite[0].math; tcg_d128_segmented_tma_xpose_narrow:35 design.math_reshape.rewrite[0].math; tcg_d128_segmented_tma_xpose_wide:19 design.math_reshape.rewrite[0].math

Let G=H/KH, Qph=Br/G, R=B*H*Sq, T=ceil_div(Skv,KeyGroup). For (b,kh,a), 0<=a<ceil_div(Sq,Qph), q0=a*Qph, U=G*min(Qph,Sq-q0). Row u=0,...,Br-1 maps to i=q0+(u mod Qph), h=kh*G+floor(u/Qph). Valid rows have i<Sq; there are U valid rows, not necessarily a prefix. For mask calculation set ic=min(i,Sq-1), lo(u)=max(0,ic+Skv-Sq-wl) when active, otherwise 0, and hi(u)=ic+Skv-Sq when is_causal, otherwise Skv-1. Use the Contract's operand scale addressing and inclusive masks. Let gLo=floor((active ? max(0,q0+Skv-Sq-wl) : 0)/KeyGroup), gHi=floor((is_causal ? min(q0+Qph-1,Sq-1)+Skv-Sq : Skv-1)/KeyGroup), Span=gHi-gLo+1. For split count n and split s=0,...,n-1 define per(n)=ceil_div(Span,n), L(n,s)=gLo+s*per(n), Hlast(n,s)=min(L(n,s)+per(n)-1,gHi), Count(n,s)=max(0,Hlast(n,s)-L(n,s)+1). The group-max evaluation uses n=Ngm; the numerator evaluation uses n=Ns. For an evaluated group g, j=g*KeyGroup+v, 0<=v<KeyGroup. Qb, Kb and Vb are original E4M3 payloads with their E8M0 block scales, without separately rounded FP32 decoding. Invalid Q rows have zero payload and unit scales; K/V payload is zero for j>=Skv, but allocated final-group scale padding bytes are still read. Matrix evaluations cover Br rows and KeyGroup keys. All Br rows evaluate scores, probabilities and PV readout, including padded rows. Only valid rows produce global results. Let r=(b*H+h)*Sq+i for valid rows, otherwise r=(b*H+h)*Sq.

## t_385b5e6c99e6df30 | math.math | 1070 chars | 2 declarations

tc_split_segmented_narrow:104 design.math_reshape.rewrite[2].math; tc_split_segmented_wide:104 design.math_reshape.rewrite[2].math

For a<Ag, maximum interval z computes scores on L(a,z,Nm)..U(a,z,Nm) and writes BmCovered[g,b,h,i]=GroupMax[u] for valid rows. Interval z=0 also writes -inf for 0<=g<gLo(a); z=Nm-1 writes -inf for gHi(a)<g<T, even when its computational interval is empty. BmCovered is defined only for i<Covered. For Covered<=i<Sq define PriorMax[g,b,h,i-Covered] as the native FP32 interpretation of the four incoming workspace bytes at 4*(g*R+(b*H+h)*Sq+i). These bytes are not written by the maximum computation and have no specified initial numerical value. Define OTemporary[u,d]=bf16(0), LseTemporary[u]=-inf. After every maximum interval, including empty ones, store these into valid rows of o_tensor and, when present, lse_tensor. All Nm intervals write the same locations. The temporary values have no later consumer; the final combine overwrites them. For each row r=(b*H+h)*Sq+i and g=0,...,T-1, let x(g,r)=BmCovered[g,b,h,i] if i<Covered, otherwise PriorMax[g,b,h,i-Covered]. The uncovered rows can make the result depend on incoming scratch; no missing maxima are supplied.

## t_38bbe5dd755df84f | math.math | 1657 chars | 1 declarations

fused_d128_key:91 design.math_reshape.rewrite[2].math

Initialize MaxState(cLo-1)[u]=-inf and DenSlice(cLo-1)[z,u]=0 for z=0,1. MaxState(g)[u]=GroupMax[u] if GroupMax[u]-MaxState(g-1)[u]> fp32(2.7725887222397812), otherwise MaxState(g-1)[u]. The comparison is false for NaN. safe[u]=MaxState(g)[u] if finite, otherwise 0. alpha[u]=expf(MaxState(g-1)[u]-safe[u]) if the old state is finite, otherwise 0. For each Score entry, x=(Score[u,v]*scale)-safe[u], with a separate FP32 multiply and subtract. Define the actual E16 approximation by: t=sat01(fma32(x,fp32(0.0057249800302088261),0.5)), w=fma_rm32(t,252,12582917), e=fma32(x,fp32(1.4426950216293335),12583043-w), d=bitcast_f32((bitcast_u32(w)<<23) mod 2^32), E16(x)=d*ex2_ftz(e). Here fma_rm32 rounds the fused result toward negative infinity; sat01 is the source saturating FP32 conversion to [0,1], including its NaN-to-zero behavior; ex2_ftz is the source approximate base-two exponential with flush-to-zero. No low log2(e) correction is added. Prob[u,v]=E16(x); Prob8[u,v]=satfinite_e4m3_rne(Prob[u,v]). The latter is the source hardware saturating RNE conversion, with unit probability scale; for nonnegative finite values it rounds min(Prob,ProbabilityMax) to E4M3FN. For z=0,1 and p=0,1, E[z,p] sums Prob[u,64*z+2*s+p] over s=0,...,31 by left-to-right FP32 adds from zero, continuing across the two 32-key chunks of that slice. DenSlice(g)[z,u]=DenSlice(g-1)[z,u]*alpha[u]+(E[z,0]+E[z,1]), using the source's scalar FP32 update under its compiler defaults; no additional contraction restriction is imposed. The denominator never uses Prob8. At the end DenFinal[u]=DenSlice(cHi)[0,u]+DenSlice(cHi)[1,u]. MaxState and DenSlice are discarded after output.

## t_39158494597ac670 | math.math | 1484 chars | 1 declarations

tc_fused:33 design.math_reshape.rewrite[0].math

Let G=H/KH, Qph=Br/G, q0=a*Qph and 0<=a<ceil_div(Sq,Qph). For each batch b and KV head kh, row u=0,...,Br-1 maps to h=kh*G+floor(u/Qph), i=q0+(u mod Qph). A row is valid iff i<Sq. Let active mean window_left is neither None nor -1. For valid rows, lo(u)=max(0,i+Skv-Sq-window_left) if active, otherwise 0; hi(u)=min(i+Skv-Sq,Skv-1) if is_causal, otherwise Skv-1. For invalid rows set lo(u)=0 and hi(u)=Skv-1. Let imin=q0, imax=min(q0+Qph-1,Sq-1), cLo=floor((active ? max(0,imin+Skv-Sq-window_left) : 0)/KeyGroup), cHi=floor((is_causal ? imax+Skv-Sq : Skv-1)/KeyGroup), Ng=cHi-cLo+1. Process this positive-length interval in increasing g. Within group g, v=0,...,KeyGroup-1 denotes key j=g*KeyGroup+v. Qraw[u,d], Kraw[v,d] and Vraw[v,d] are the stored E4M3 payloads at these coordinates. Zero-fill Qraw for invalid queries and Kraw/Vraw for j>=Skv. Use the Contract's packed scale-byte addressing. QueryScale[u,k] and KeyScale[v,k] are 2^(byte-ExponentBias) for k=0,...,D/Block-1, but zero for invalid queries and j>=Skv respectively. QueryScale is computed once per row tile. ValueScale[c,d], c=0,...,KeyGroup/Block-1, is the BF16 scale for this group's key-axis block c. Its byte decodes as 2^(byte-ExponentBias): byte 0 gives 2^-127 and padding byte 255 gives +inf. Allocated padding V-scale bytes are still used; padded V is not zeroed after scaling. All Br rows and KeyGroup keys in each visited group are evaluated. Wholly excluded groups are skipped; only valid rows are stored.

## t_3cf16fbaa5a6d9aa | math.math | 344 chars | 2 declarations

tc_split_segmented_narrow:134 design.math_reshape.rewrite[3].math; tc_split_segmented_wide:134 design.math_reshape.rewrite[3].math

Let Nseg=ceil_div(T,Seg). For every row r and segment 0<=s<Nseg, start mx=-inf. In order v=0,...,Seg-1 set mx=fmax(mx,x(s*Seg+v,r) if s*Seg+v<T, otherwise -inf). Store SegmentMax[s,r]=mx. This reads all T maxima for every row, including PriorMax. SegmentMax[0,r] is written but never consumed; the remaining summaries are retained for the scan.

## t_43fa42661fe7b327 | math.math | 766 chars | 7 declarations

tcg_d256_flat_cp_narrow:138 design.math_reshape.rewrite[4].math; tcg_d256_flat_cp_wide:139 design.math_reshape.rewrite[4].math; tcg_d256_flat_tma_narrow:138 design.math_reshape.rewrite[4].math; tcg_d256_flat_tma_wide:138 design.math_reshape.rewrite[4].math; tcg_d256_segmented_cp_narrow:145 design.math_reshape.rewrite[5].math; tcg_d256_segmented_tma_narrow:160 design.math_reshape.rewrite[5].math; tcg_d256_segmented_tma_wide:145 design.math_reshape.rewrite[5].math

For a nonempty split, NumState(l-1)=0 is the initial mathematical state. At g=l use PVAcc[-1]=0 by overwrite without reading storage. Later, PVAcc[-1]=NumState(g-1)*alpha rowwise on evaluated rows, retaining NumState(g-1) on other rows. An exact unit rescale may be skipped. For k=0,...,3, PVAcc[k]=mma32(Prob8[:,32*k:32*k+32], Vb[32*k:32*k+32,:],PVAcc[k-1]); NumState(g)=PVAcc[3]. DenFinal[u]=DenPart(e)[u,0]+DenPart(e)[u,1]. For valid rows, weight=expf(Mscan[e,row]-Mtail[row]) when both states are finite, otherwise 0. PartA[s,row,d]=NumState(e)[u,d]*weight; PartB[s,row]=DenFinal[u]*weight, retaining multiplication by zero even for nonfinite numerators. Empty splits write PartA=0, PartB=0 for every valid row without reading a numerator. Local states then die.

## t_474d01d934b0b46e | math.math | 723 chars | 2 declarations

tcg_d128_flat_cp_direct_narrow:171 design.math_reshape.rewrite[5].math; tcg_d128_flat_tma_direct_narrow:172 design.math_reshape.rewrite[5].math

For each real row r, accumulate PartA[s,r,d] and PartB[s,r] from zero by increasing s with left-to-right FP32 additions, obtaining TotalA[d] and TotalB. The eight-at-a-time loop also adds zero for missing entries in its final batch. Let live=(TotalB>0), inv=1/TotalB when live, otherwise 0. Write O[b,h,i,d]=bf16(TotalA[d]*inv) when live, otherwise bf16(0), using RNE. When LSE is present write MTail[r]+logf(TotalB/16) if live and MTail[r] is finite, otherwise -inf. Initial destination bytes are not read. All valid caller destinations are written and the host returns None. Workspace retains intermediates; no mathematical scratch output or scratch-zero initialization is required. The segment-max reservation is unused.

## t_4b354a9457c40560 | math.math | 933 chars | 3 declarations

tcg_d256_segmented_cp_narrow:89 design.math_reshape.rewrite[3].math; tcg_d256_segmented_tma_narrow:104 design.math_reshape.rewrite[3].math; tcg_d256_segmented_tma_wide:89 design.math_reshape.rewrite[3].math

Define advance(m,x)=x if x-m>fp32(2.7725887222397812), otherwise m; the comparison is false for NaN. For each row, Mscan(-1,row)=-inf. For t=0,...,min(SegLen,GCount)-1, unconditionally evaluate Mscan[t,row]=advance(Mscan(t-1,row),GroupMax[t,row]). For c=0,...,ceil_div(Nseg-1,SegChunk)-1, let f0=1+c*SegChunk, t0=f0*SegLen, t1=min((f0+SegChunk)*SegLen,GCount), incoming=Mscan[t0-1,row]. For z=0,...,SegChunk-1, Hseg[z]=SegMax[f0+z,row] when f0+z<Nseg, otherwise -inf. Jump={z | 0<=z<SegChunk and Hseg[z]-incoming>fp32(2.7725887222397812)}. Let tp=t1 when Jump is empty, otherwise min(t0+min(Jump)*SegLen,t1). For t0<=t<tp, write Mscan[t,row]=incoming without reading GroupMax. For tp<=t<t1, evaluate Mscan[t,row]=advance(Mscan(t-1,row),GroupMax[t,row]). Replay the entire suffix without retesting individual segments. Mtail[row]=Mscan[GCount-1,row]. Keep every Mscan for split replay and Mtail for partial weighting and optional LSE.

## t_513ab7924bb3b295 | math.math | 1456 chars | 1 declarations

tcg_d128_fscan_cp_direct_wide:174 design.math_reshape.rewrite[5].math

For each real row r and s=0,...,Ns-1, Weight[s]=expf(PartM[s,r]-MTail[r]) when both maxima are finite, otherwise 0. Let Cmerge=ceil_div(Ns,MergeGroups). For z=0,...,MergeGroups-1, let Slo(z)=min(z*Cmerge,Ns), Shi(z)=min(Slo(z)+Cmerge,Ns). GroupA[z,d] sums PartA[s,r,d]*Weight[s] from zero in increasing s in [Slo(z),Shi(z)), with source scalar FP32 multiply-add evaluation. The first eight entries include zero additions for absent entries without reading their weights or evaluating a product; later full batches and a scalar remainder cover only existing entries. For t=0,...,MergeWidth-1 and q=Slo(z)+t, set Vmerge[z,t]= PartB[q,r]*Weight[q] when q<Shi(z), otherwise zero. For w=0,...,MergeWidth/32-1, start X[l]=Vmerge[z,32*w+l], l=0,...,31; for o=16,8,4,2,1 update X[l]=X[l]+X[l+o] for 0<=l<o in FP32. GroupB[z,w]=X[0]. TotalA[d] sums GroupA[z,d] from zero in increasing z; TotalB sums GroupB[z,w] from zero in lexicographic (z,w) order. Let live=(TotalB>0), inv=1/TotalB when live, otherwise 0. Write O[b,h,i,d]=bf16(TotalA[d]*inv) when live, otherwise bf16(0), using RNE. When LSE is present write MTail[r]+logf(TotalB/16) if live and MTail[r] is finite, otherwise -inf. Initial destination bytes are not read. All valid caller destinations are written and the host returns None. Workspace retains intermediates; no mathematical scratch output or scratch-zero initialization is required. Segment-max and per-group scan-state reservations are unused.

## t_57410ed204628997 | math.math | 815 chars | 12 declarations

tcg_d128_flat_cp_direct_narrow:65 design.math_reshape.rewrite[1].math; tcg_d128_flat_cp_direct_wide:65 design.math_reshape.rewrite[1].math; tcg_d128_flat_tma_direct_narrow:66 design.math_reshape.rewrite[1].math; tcg_d128_flat_tma_direct_wide:66 design.math_reshape.rewrite[1].math; tcg_d128_fscan_cp_direct_narrow:65 design.math_reshape.rewrite[1].math; tcg_d128_fscan_cp_direct_wide:65 design.math_reshape.rewrite[1].math; tcg_d128_fscan_tma_direct_narrow:65 design.math_reshape.rewrite[1].math; tcg_d128_fscan_tma_direct_wide:65 design.math_reshape.rewrite[1].math; tcg_d128_segmented_cp_direct_narrow:66 design.math_reshape.rewrite[1].math; tcg_d128_segmented_cp_direct_wide:65 design.math_reshape.rewrite[1].math; tcg_d128_segmented_tma_direct_narrow:66 design.math_reshape.rewrite[1].math; tcg_d128_segmented_tma_direct_wide:66 design.math_reshape.rewrite[1].math

For each group g in a nonempty split, A(-1)=0 by overwrite and A(k)=mma32(Qb[:,32*k:32*k+32],Kb[:,32*k:32*k+32]^T,A(k-1)), k=0,...,3. DotTile=A(3). For u<C, Score[u,v]=DotTile[u,v] when lo(u)<=g*KeyGroup+v<=hi(u), otherwise -inf. Let scale=fp32(D^(-1/2)). In the group-max evaluation, GroupMax[g,r]=scale*fmax(Score[u,:]) for valid rows. Reduce from -inf with source fmaxf NaN behavior: four 32-value chunk maxima followed by fmaxf across chunks; masked chunks use a halving tree, wholly unmasked chunks can use the native max reduction. GroupMax[g,r]=-inf for g<gLo or g>gHi, explicitly filled by the first or last group-max split. No matrix product evaluates those groups. The numerator evaluation recomputes DotTile with the same contractions but consumes Score in probability chunks without a maximum reduction.

## t_576072428dd1fe1b | math.math | 996 chars | 1 declarations

tcg_d128_fscan_tma_xpose_narrow:178 design.math_reshape.rewrite[5].math

For each real row r and s=0,...,Ns-1 let Weight[s]= expf(PartM[s,r]-MTail[r]) when both maxima are finite, otherwise 0. Accumulate PartA[s,r,d]*Weight[s] and PartB[s,r]*Weight[s] from zero by increasing s in FP32 to obtain TotalA[d] and TotalB. All Ns entries are read, including empty splits regardless of PartB being zero; multiplication by zero does not suppress NaN/Inf in PartA. Missing entries in the final eight-entry batch add zero without evaluating a weighted product. Let live=(TotalB>0), inv=1/TotalB when live, otherwise 0. Write O[b,h,i,d]=bf16(TotalA[d]*inv) when live, otherwise bf16(0), using RNE. When LSE is present write MTail[r]+logf(TotalB/16) if live and MTail[r] is finite, otherwise -inf. Initial destination bytes are not read. All valid caller destinations are written and the host returns None. Workspace retains intermediates; no mathematical scratch output or scratch-zero initialization is required. The global scan-snapshot and segment-max reservations are unused.

## t_5c05b65e7f20540d | math.math | 1310 chars | 7 declarations

tcg_d256_flat_cp_narrow:37 design.math_reshape.rewrite[0].math; tcg_d256_flat_cp_wide:38 design.math_reshape.rewrite[0].math; tcg_d256_flat_tma_narrow:37 design.math_reshape.rewrite[0].math; tcg_d256_flat_tma_wide:37 design.math_reshape.rewrite[0].math; tcg_d256_segmented_cp_narrow:22 design.math_reshape.rewrite[0].math; tcg_d256_segmented_tma_narrow:37 design.math_reshape.rewrite[0].math; tcg_d256_segmented_tma_wide:22 design.math_reshape.rewrite[0].math

R=B*H*Sq, GCount=ceil_div(Skv,KeyGroup), G=H/KH, Qph=Br/G, q0=a*Qph, 0<=a<ceil_div(Sq,Qph), 0<=u<Br. With PSK, i=q0+floor(u/G), h=kh*G+(u mod G); otherwise i=q0+(u mod Qph), h=kh*G+floor(u/Qph). A row is valid iff i<Sq. Let ic=min(i,Sq-1), row=(b*H+h)*Sq+i, ReadRow=(b*H+h)*Sq+(i if valid, otherwise 0), and EvalRows=32*ceil_div(G*min(Qph,Sq-q0),32) with PSK, otherwise Br. Scalar score/probability evaluation covers u<EvalRows; dense products still evaluate all Br rows, while only valid rows have outputs. Use the Contract's scale addressing and inclusive mask: lo(u)=max(0,ic+Skv-Sq-wl) when active, otherwise 0; hi(u)=ic+Skv-Sq when is_causal, otherwise Skv-1. gLo=floor((active ? max(0,q0+Skv-Sq-wl) : 0)/KeyGroup), gHi=floor((is_causal ? min(q0+Qph-1,Sq-1)+Skv-Sq : Skv-1)/KeyGroup), Ng=gHi-gLo+1. For n in {Nm,Ns}, per(n)=ceil_div(Ng,n), L(n,s)=gLo+s*per(n), End(n,s)=min(L(n,s)+per(n)-1,gHi), 0<=s<n. A partition is empty iff L(n,s)>End(n,s). Qb denotes original E4M3 query and its E8M0 blocks, with zero query payload and unit scales for invalid rows. At j=g*KeyGroup+v, Kb[v,d] and Vb[v,d] denote original E4M3 operands and their scales, 0<=v<KeyGroup, 0<=d<D. Payloads are zero for j>=Skv; allocated padding scale bytes are still used. Only gLo..gHi execute products, including masked boundary keys.

## t_5ca51b463a3d5ec8 | math.math | 1298 chars | 4 declarations

tc_split_flat_narrow:178 design.math_reshape.rewrite[4].math; tc_split_flat_wide:178 design.math_reshape.rewrite[4].math; tc_split_segmented_narrow:213 design.math_reshape.rewrite[6].math; tc_split_segmented_wide:213 design.math_reshape.rewrite[6].math

For the current group, Vbf[v,d]=bf16(bf16(Vraw[v,d])*ValueScale[floor(v/Block),d]), using a BF16 multiplication and retaining its rounded result. Initialize NumState(cLo-1)[u,d]=0. For each g, start C(-1)[u,d]=NumState(g-1)[u,d]*alpha[u], with a multiply by one skipped when selected by the source. For t=0,...,KeyGroup/16-1, Pfrag[u,v]=bf16(Prob8[u,16*t+v]), 0<=v<16, and C(t)=mma32(Pfrag,Vbf[16*t:16*t+16,:],C(t-1)). E4M3-to-BF16 widening is exact for finite values and preserves NaN. NumState(g)=C(KeyGroup/16-1). Pfrag is consumed before the next chunk; no full rounded-probability tile is retained. For cLo<=cHi let endm[u]=ScanState[cHi,rho(u)] and weight[u]=expf(endm[u]-MaxTail[rho(u)]) if both are finite, otherwise 0. For an empty interval weight[u]=0 and NumState and DenPart stay zero; no MaxTail read is required for weighting. Set Zlocal[u]=(DenPart(gLast)[u,0]+DenPart(gLast)[u,1])+ (DenPart(gLast)[u,2]+DenPart(gLast)[u,3]). For valid r=(b*H+h)*Sq+i, store PartialA[z,r,d]=NumState(gLast)[u,d]*weight[u], PartialB[z,r]=Zlocal[u]*weight[u], both FP32. Empty intervals write zero partials. Nonempty intervals still evaluate products of nonfinite states with zero weight. Vbf dies after its group's PV; local carried values die after partial output. ScanState dies after all intervals.

## t_6806b40fee3a82f2 | math.math | 1430 chars | 12 declarations

tcg_d128_flat_cp_direct_narrow:34 design.math_reshape.rewrite[0].math; tcg_d128_flat_cp_direct_wide:34 design.math_reshape.rewrite[0].math; tcg_d128_flat_tma_direct_narrow:35 design.math_reshape.rewrite[0].math; tcg_d128_flat_tma_direct_wide:35 design.math_reshape.rewrite[0].math; tcg_d128_fscan_cp_direct_narrow:34 design.math_reshape.rewrite[0].math; tcg_d128_fscan_cp_direct_wide:34 design.math_reshape.rewrite[0].math; tcg_d128_fscan_tma_direct_narrow:34 design.math_reshape.rewrite[0].math; tcg_d128_fscan_tma_direct_wide:34 design.math_reshape.rewrite[0].math; tcg_d128_segmented_cp_direct_narrow:35 design.math_reshape.rewrite[0].math; tcg_d128_segmented_cp_direct_wide:34 design.math_reshape.rewrite[0].math; tcg_d128_segmented_tma_direct_narrow:35 design.math_reshape.rewrite[0].math; tcg_d128_segmented_tma_direct_wide:35 design.math_reshape.rewrite[0].math

Let G=H/KH, Qph=Br/G, R=B*H*Sq, T=ceil_div(Skv,KeyGroup). For (b,kh,a), 0<=a<ceil_div(Sq,Qph), q0=a*Qph, U=G*min(Qph,Sq-q0), and C=32*ceil_div(U,32). Row u=0,...,Br-1 maps to i=q0+floor(u/G), h=kh*G+(u mod G). Rows u<U are valid. For mask calculation set ic=min(i,Sq-1), lo(u)=max(0,ic+Skv-Sq-wl) when active, otherwise 0, and hi(u)=ic+Skv-Sq when is_causal, otherwise Skv-1. Use the Contract's operand scale addressing and inclusive masks. Let gLo=floor((active ? max(0,q0+Skv-Sq-wl) : 0)/KeyGroup), gHi=floor((is_causal ? min(q0+Qph-1,Sq-1)+Skv-Sq : Skv-1)/KeyGroup), Span=gHi-gLo+1. For split count n and split s=0,...,n-1 define per(n)=ceil_div(Span,n), L(n,s)=gLo+s*per(n), Hlast(n,s)=min(L(n,s)+per(n)-1,gHi), Count(n,s)=max(0,Hlast(n,s)-L(n,s)+1). The group-max evaluation uses n=Ngm; the numerator evaluation uses n=Ns. For an evaluated group g, j=g*KeyGroup+v, 0<=v<KeyGroup. Qb, Kb and Vb are original E4M3 payloads with their E8M0 block scales, without separately rounded FP32 decoding. Invalid Q rows have zero payload and unit scales; K/V payload is zero for j>=Skv, but allocated final-group scale padding bytes are still read. Matrix evaluations cover Br rows and KeyGroup keys. Scalar score/probability/PV readout covers u<C, including padded rows; the remaining rows have no probability writes or scalar PV accumulation. Only u<U produces global results. Let r=(b*H+h)*Sq+i for valid rows, otherwise r=(b*H+h)*Sq.

## t_6cb107cf0d038792 | math.math | 696 chars | 3 declarations

tcg_d128_fscan_cp_xpose_narrow:90 design.math_reshape.rewrite[2].math; tcg_d128_fscan_cp_xpose_wide:90 design.math_reshape.rewrite[2].math; tcg_d128_fscan_tma_xpose_wide:90 design.math_reshape.rewrite[2].math

For numerator split s let cLo=L(Ns,s), cHi=Hlast(Ns,s), tw=min(cLo,T). For u<Br, use r=(b*H+h)*Sq+i on valid rows, and r=(b*H+h)*Sq on padding. Initialize MaxState(-1)[u]=-inf. For g=0,...,tw-1, then g=cLo,...,cHi when nonempty, apply MaxState(g)[u]=GroupMax[g,r] if GroupMax[g,r]-MaxState(g-1)[u]>fp32(2.7725887222397812), otherwise MaxState(g-1)[u]. The comparison is false for NaN. Define end(s)=cHi for a nonempty split, otherwise tw-1. No reassociation or global per-group state snapshots are used. Store PartM[s,r]=MaxState(end(s))[u] for every valid row. The last split s=Ns-1 also stores MTail[r]=PartM[s,r], including when it is empty. Maxima beyond its scanned useful interval are -inf.

## t_72881d9e093aff65 | math.math | 1588 chars | 2 declarations

fused_d256_cp:86 design.math_reshape.rewrite[2].math; fused_d256_tma:86 design.math_reshape.rewrite[2].math

Initialize MaxState(cLo-1)[u]=-inf and DenPart(cLo-1)[u,z]=0 for z=0,1. MaxState(g)[u]=GroupMax[u] if GroupMax[u]-MaxState(g-1)[u]> fp32(2.7725887222397812), otherwise MaxState(g-1)[u]. The comparison is false for NaN. safe[u]=MaxState(g)[u] if finite, otherwise 0. alpha[u]=expf(MaxState(g-1)[u]-safe[u]) if the old state is finite, otherwise 0. For each Score entry, x=(Score[u,v]*scale)-safe[u], with a separate FP32 multiply and subtract. Define the actual E1 approximation by: t=sat01(fma32(x,fp32(0.0057249800302088261),0.5)), w=fma_rm32(t,252,12582913), e=fma32(x,fp32(1.4426950216293335),12583039-w), d=bitcast_f32((bitcast_u32(w)<<23) mod 2^32), E1(x)=d*ex2_ftz(e). Here fma_rm32 rounds the fused result toward negative infinity; sat01 is the source saturating FP32 conversion to [0,1], including its NaN-to-zero behavior; ex2_ftz is the source approximate base-two exponential with flush-to-zero. No low log2(e) correction is added. Prob[u,v]=E1(x); Prob8[u,v]=satfinite_e4m3_rne(Prob[u,v]). The latter is the source hardware saturating RNE conversion, with unit probability scale; for nonnegative finite values it rounds min(Prob,ProbabilityMax) to E4M3FN. For each half z and parity p=0,1, start E[z,p]=0 and add Prob[u,64*z+2*r+p] for r=0,...,31 in increasing order with FP32 adds. Scale DenPart(g-1)[u,z] by alpha[u], then add E[z,0]+E[z,1], obtaining DenPart(g)[u,z]. These are the source's scalar FP32 updates under its compiler defaults; no additional contraction restriction is imposed. DenPart never uses Prob8. MaxState and DenPart are discarded after the output step.

## t_72a679deafdd406e | math.math | 1207 chars | 1 declarations

tc_fused:126 design.math_reshape.rewrite[3].math

For the current group, Vbf[v,d]=bf16(bf16(Vraw[v,d])*ValueScale[floor(v/Block),d]), using a BF16 multiplication and retaining its rounded result. Initialize NumState(cLo-1)[u,d]=0. For each g, start C(-1)[u,d]=NumState(g-1)[u,d]*alpha[u], with a multiply by one skipped when selected by the source. For t=0,...,KeyGroup/16-1, Pfrag[u,v]=bf16(Prob8[u,16*t+v]), 0<=v<16, and C(t)=mma32(Pfrag,Vbf[16*t:16*t+16,:],C(t-1)). E4M3-to-BF16 widening is exact for finite values and preserves NaN. NumState(g)=C(KeyGroup/16-1). Pfrag is consumed before the next chunk; no full rounded-probability tile is retained. After g=cHi, let Z[u]=(DenPart(g)[u,0]+DenPart(g)[u,1])+ (DenPart(g)[u,2]+DenPart(g)[u,3]). Set live[u]=(Z[u]>0), inv[u]=1/Z[u] when live, otherwise 0. Write valid rows as O[b,h,i,d]=bf16(NumState(g)[u,d]*inv[u]) when live, otherwise bf16(0), using RNE. When LSE is present write MaxState(g)[u]+logf(Z[u]/beta) if live and MaxState is finite, otherwise -inf. Vbf is discarded after the group's last PV use; carried states are discarded after output. Zero probabilities do not suppress products with Inf/NaN Vbf. Workspace remains untouched; the host returns None without reading initial output contents.

## t_757582c9cea60e88 | math.math | 393 chars | 1 declarations

fused_d128_key:68 design.math_reshape.rewrite[1].math

Start A[-1]=0 and for k=0,...,3 form A[k]=mma32(Qb[:,32*k:32*k+32],Kb[:,32*k:32*k+32]^T,A[k-1]). Score[u,v]=A[3][u,v] if lo(u)<=g*KeyGroup+v<=hi(u), and -inf otherwise. Let scale=fp32(D^(-1/2)). For z=0,1, HalfMax[z,u] is the fmax reduction over Score[u,64*z:64*z+64], starting at -inf. RawMax[u]=fmax(HalfMax[0,u],HalfMax[1,u]); use the source fmaxf NaN behavior. GroupMax[u]=RawMax[u]*scale.

## t_778d764bd6afd49a | math.math | 1444 chars | 1 declarations

tcg_d128_fscan_tma_xpose_narrow:112 design.math_reshape.rewrite[3].math

Initialize DenState(cLo-1)[u]=0 for u<Br. For g=cLo,...,cHi in a nonempty split use scan's preceding and current MaxState, safe[u]=MaxState(g)[u] if finite, otherwise 0, and alpha[u]=expf(MaxState(g-1)[u]-safe[u]) if the old state is finite, otherwise 0. Compute x=Score[u,v]*scale-safe[u] with separate FP32 multiply and subtract. The actual E16 approximation is t=sat01(fma32(x,fp32(0.0057249800302088261),0.5)), w=fma_rm32(t,252,12582917), e=fma32(x,fp32(1.4426950216293335),12583043-w), d=bitcast_f32((bitcast_u32(w)<<23) mod 2^32), E16(x)=d*ex2_ftz(e). fma_rm32 rounds toward negative infinity; sat01 saturates to [0,1] with NaN-to-zero; ex2_ftz is the source approximate base-two exponential with flush-to-zero. No low log2(e) correction is added. Prob[u,v]=E16(x), Prob8[u,v]=satfinite_e4m3_rne(Prob[u,v]), the source hardware saturating RNE cast with unit scale; for nonnegative finite values this rounds min(Prob,ProbabilityMax) to E4M3FN. For chunk c=0,...,3, partition z=0,...,3 and pair p=0,1, let F[c,z,p] be the left-to-right FP32 sum from zero of Prob[u,32*c+8*z+2*j+p], j=0,...,3. Sum (F[c,0,p]+F[c,1,p])+(F[c,2,p]+F[c,3,p]) over increasing c from zero to E[p]. Then DenState(g)[u]=DenState(g-1)[u]*alpha[u]+(E[0]+E[1]). This scalar update uses source compiler defaults without an additional contraction restriction; the denominator never uses Prob8. An empty split has no probability evaluation and leaves its denominator zero.

## t_7817d7d7eed82a6d | math.math | 1142 chars | 1 declarations

tcg_d256_segmented_cp_wide:199 design.math_reshape.rewrite[6].math

Let Cp=ceil_div(Ns,SP). For 0<=p<SP define sLo(p)=min(p*Cp,Ns), sHi(p)=min(sLo(p)+Cp,Ns). For each row and channel d, initialize GroupNum[p,d]=0 and add PartA[s,row,d] in ascending s over sLo(p)<=s<sHi(p). Empty intervals retain zero; no zero tail terms are added. For 0<=w<4 and 0<=v<32, let s=sLo(p)+32*w+v and X[v]=PartB[s,row] when s<sHi(p), otherwise 0. Define reduce32(X) by successively replacing a length-n vector with its first n/2 entries plus its last n/2 entries, for n=32,16,8,4,2; each addition is FP32. GroupDen[p,w]=reduce32(X). Initialize OutDen=0 and add GroupDen[p,w] with p ascending, then w ascending within p. Independently for every channel, initialize OutNum=0 and add GroupNum[p,d] in ascending p. Let live=OutDen>0 and inv=1/OutDen when live, otherwise 0. O[b,h,i,d]=bf16(OutNum*inv) when live, otherwise bf16(0), using RNE. If LSE is present, LSE[b,h,i]=Mtail[row]+logf(OutDen) when live and Mtail is finite, otherwise -inf. GroupMax, SegMax, Mscan, Mtail, PartA and PartB occupy mutable workspace. Other scratch regions have no mathematical result. Initial O/LSE contents are never read, and the host returns None.

## t_7a52ecac92f2bb72 | math.math | 684 chars | 2 declarations

tcg_d128_segmented_cp_direct_narrow:193 design.math_reshape.rewrite[6].math; tcg_d128_segmented_tma_direct_narrow:193 design.math_reshape.rewrite[6].math

For each real row r, accumulate PartA[s,r,d] and PartB[s,r] from zero by increasing s with left-to-right FP32 additions, obtaining TotalA[d] and TotalB. The eight-at-a-time loop also adds zero for missing entries in its final batch. Let live=(TotalB>0), inv=1/TotalB when live, otherwise 0. Write O[b,h,i,d]=bf16(TotalA[d]*inv) when live, otherwise bf16(0), using RNE. When LSE is present write MTail[r]+logf(TotalB/16) if live and MTail[r] is finite, otherwise -inf. Initial destination bytes are not read. All valid caller destinations are written and the host returns None. Workspace retains intermediates; no mathematical scratch output or scratch-zero initialization is required.

## t_7b337ed4f6337af6 | math.math | 1494 chars | 4 declarations

tcg_d128_fscan_cp_xpose_narrow:34 design.math_reshape.rewrite[0].math; tcg_d128_fscan_cp_xpose_wide:34 design.math_reshape.rewrite[0].math; tcg_d128_fscan_tma_xpose_wide:34 design.math_reshape.rewrite[0].math; tcg_d128_segmented_cp_xpose_wide:36 design.math_reshape.rewrite[0].math

Let G=H/KH, Qph=Br/G, R=B*H*Sq, T=ceil_div(Skv,KeyGroup). For each (b,kh,a), 0<=a<ceil_div(Sq,Qph), let q0=a*Qph, U=G*min(Qph,Sq-q0). Rows u=0,...,Br-1 map to i=q0+(u mod Qph), h=kh*G+floor(u/Qph). Valid rows have i<Sq; their count is U and they need not form a prefix. Set ic=min(i,Sq-1) for masks. Use the Contract's stored operands, scale addressing and inclusive masks. Let lo(u)=max(0,ic+Skv-Sq-wl) when active, otherwise 0; hi(u)=ic+Skv-Sq when is_causal, otherwise Skv-1. Let gLoAll=floor((active ? max(0,q0+Skv-Sq-wl) : 0)/KeyGroup), gHiAll=floor((is_causal ? min(q0+Qph-1,Sq-1)+Skv-Sq : Skv-1)/KeyGroup), Span=gHiAll-gLoAll+1. For split count N and 0<=s<N, define per(N)=ceil_div(Span,N), L(N,s)=gLoAll+s*per(N), Hlast(N,s)=min(L(N,s)+per(N)-1,gHiAll), Count(N,s)=max(0,Hlast(N,s)-L(N,s)+1). Group-max uses N=Ngm; the numerator pass uses N=Ns. The pass label below distinguishes these two QK evaluations. For g in a nonempty split, j=g*KeyGroup+v, 0<=v<KeyGroup. Qb, Kb, Vb denote the stored E4M3 payloads with their original E8M0 block scales, not separately rounded decoded arrays. Invalid Q rows have zero payload and unit scales. K/V payload is zero for j>=Skv; the allocated padding scale bytes are still read. All Br rows and KeyGroup keys participate in the contractions, score evaluation and probabilities, including padded rows. Only valid rows contribute global mathematical results. Let r=(b*H+h)*Sq+i for valid rows. Groups outside gLoAll..gHiAll have no matrix evaluation.

## t_7db216690167f85f | math.math | 785 chars | 1 declarations

fused_d128_key:134 design.math_reshape.rewrite[3].math

Let NumState(cLo-1)=0 denote the initial mathematical numerator. For g=cLo, start C[-1]=0 using overwrite semantics, without reading uninitialized storage. For later groups, C[-1][u,d]= NumState(g-1)[u,d]*alpha[u]; a rescale by exactly one may be skipped. For k=0,...,3, C[k]=mma32(Prob8[:,32*k:32*k+32], Vb[32*k:32*k+32,:],C[k-1]). NumState(g)=C[3]. At g=cHi, let live[u]=(DenFinal[u]>0) and inv[u]=1/DenFinal[u] when live, otherwise 0. For each valid row write O[b,h,i,d]=bf16(NumState(g)[u,d]*inv[u]) when live, otherwise bf16(0), using RNE conversion. When LSE is present write MaxState(g)[u]+logf(DenFinal[u]/16) if live and MaxState is finite, otherwise -inf. NumState is discarded after output. This fused path leaves workspace untouched and returns None through the host entry.

## t_7f0000fa45e56651 | math.math | 732 chars | 1 declarations

tcg_d128_fscan_tma_xpose_narrow:90 design.math_reshape.rewrite[2].math

For each numerator split set cLo=L(Ns,s), cHi=Hlast(Ns,s), tw=min(cLo,T), end=max(cLo-1,cHi), and mEnd=cHi when nonempty, otherwise tw-1. Initialize MaxState(-1)[u]=-inf for u<Br. For g=0,...,tw-1 followed by g=cLo,...,cHi when nonempty, apply MaxState(g)[u]=GroupMax[g,r] if GroupMax[g,r]-MaxState(g-1)[u]>fp32(2.7725887222397812), otherwise MaxState(g-1)[u]. The comparison is false for NaN. No reassociation or global per-group state snapshots are used. Store PartM[s,r]=MaxState(mEnd)[u] for every valid row, including empty splits. The last split s=Ns-1 also stores MTail[r] from this state. Groups beyond its scanned useful interval have GroupMax=-inf. Retain MaxState through both state stores and the probability recurrence.

## t_8005006628b86c62 | math.math | 1582 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:128 design.math_reshape.rewrite[4].math

For numerator split s, let cLo=L(Ns,s), cHi=Hlast(Ns,s). For a nonempty split and u<Br, initialize MaxState(cLo-1)[u] from MState[cLo-1,r] when cLo>0, otherwise -inf, and DenState(cLo-1)=0. For padding use r=(b*H+h)*Sq, the first real row of that head. For each g=cLo,...,cHi, MaxState(g)[u]=MState[g,r]. safe[u]=MaxState(g)[u] if finite, otherwise 0; alpha[u]=expf(MaxState(g-1)[u]-safe[u]) if the old state is finite, otherwise 0. Compute x=Score[u,v]*scale-safe[u], with separate FP32 multiply and subtract. Define the actual E16 approximation by t=sat01(fma32(x,fp32(0.0057249800302088261),0.5)), w=fma_rm32(t,252,12582917), e=fma32(x,fp32(1.4426950216293335),12583043-w), d=bitcast_f32((bitcast_u32(w)<<23) mod 2^32), E16(x)=d*ex2_ftz(e). fma_rm32 rounds toward negative infinity; sat01 is the source saturating FP32 conversion to [0,1], including NaN-to-zero; ex2_ftz is its approximate base-two exponential with flush-to-zero. There is no low log2(e) correction. Prob[u,v]=E16(x); Prob8[u,v]=satfinite_e4m3_rne(Prob[u,v]), the hardware saturating RNE conversion with unit scale. For nonnegative finite values it rounds min(Prob,ProbabilityMax) to E4M3FN. For each chunk c=0,...,3, pair p=0,1 and partition z=0,...,3, F[c,z,p] is the left-to-right FP32 sum from zero of Prob[u,32*c+8*z+2*j+p], j=0,...,3. Sum (F[c,0,p]+F[c,1,p])+(F[c,2,p]+F[c,3,p]) over increasing c from zero to E[p]. DenState(g)[u]=DenState(g-1)[u]*alpha[u]+(E[0]+E[1]), with the source's scalar FP32 compiler path; no additional contraction restriction is imposed. The denominator never uses rounded Prob8.

## t_81a5d4741e3d281b | math.math | 387 chars | 2 declarations

fused_d256_cp:64 design.math_reshape.rewrite[1].math; fused_d256_tma:64 design.math_reshape.rewrite[1].math

Start A[-1]=0 and for k=0,...,7 form A[k]=mma32(Qb[:,32*k:32*k+32],Kb[:,32*k:32*k+32]^T,A[k-1]). Score[u,v]=A[7][u,v] if lo(u)<=g*KeyGroup+v<=hi(u), and -inf otherwise. Let scale=fp32(D^(-1/2)). For z=0,1, HalfMax[u,z]=fmax over Score[u,64*z:64*z+64], starting at -inf. RawMax[u]=fmax(HalfMax[u,0],HalfMax[u,1]). The maxima use the source fmaxf NaN behavior. GroupMax[u]=RawMax[u]*scale.

## t_857e76cdab2a771d | math.math | 1313 chars | 7 declarations

tcg_d256_flat_cp_narrow:104 design.math_reshape.rewrite[3].math; tcg_d256_flat_cp_wide:105 design.math_reshape.rewrite[3].math; tcg_d256_flat_tma_narrow:104 design.math_reshape.rewrite[3].math; tcg_d256_flat_tma_wide:104 design.math_reshape.rewrite[3].math; tcg_d256_segmented_cp_narrow:111 design.math_reshape.rewrite[4].math; tcg_d256_segmented_tma_narrow:126 design.math_reshape.rewrite[4].math; tcg_d256_segmented_tma_wide:111 design.math_reshape.rewrite[4].math

For nonempty split s, l=L(Ns,s), e=End(Ns,s), DenPart(l-1)[u,z]=0. At g=l,...,e and u<EvalRows, mp=Mscan[g-1,ReadRow] when g>0, otherwise -inf; mt=Mscan[g,ReadRow], safe=mt when finite, otherwise 0; alpha=expf(mp-safe) when mp is finite, otherwise 0. For each evaluated Score entry, x=Score[u,v]*scale-safe, with separate FP32 multiply and subtract. Define the actual E1 approximation: t=sat01(fma32(x,fp32(0.0057249800302088261),0.5)), w=fma_rm32(t,252,12582913), z1=fma32(x,fp32(1.4426950216293335),12583039-w), factor=bitcast_f32((bitcast_u32(w)<<23) mod 2^32), E1(x)=factor*ex2_ftz(z1). fma_rm32 rounds downward; sat01 saturates to [0,1], including NaN-to-zero; ex2_ftz is approximate base-two exponential with flush-to-zero. No low log2(e) correction is added. Prob[u,v]=E1(x), Prob8[u,v]=satfinite_e4m3_rne(Prob[u,v]), the hardware saturating RNE conversion with unit scale, rounding min(Prob,ProbabilityMax) for nonnegative finite inputs. For z=0,1 and p=0,1, start EvenOdd[z,p]=0 and add Prob[u,64*z+2*r+p] for r=0,...,31 in increasing FP32 order. DenPart(g)[u,z]=DenPart(g-1)[u,z]*alpha+(EvenOdd[z,0]+EvenOdd[z,1]). Rows u>=EvalRows keep DenPart=0 and perform no probability overwrite or numerator rescale; their unspecified retained operands still enter dense PV, but these rows have no logical outputs.

## t_8a3204c418d57a10 | math.math | 1388 chars | 1 declarations

tcg_d256_segmented_cp_wide:130 design.math_reshape.rewrite[4].math

For a nonempty split s, let l=L(Ns,s), e=Hlast(Ns,s). Initialize DenPart(l-1)[u,z]=0 for z=0,1. For g=l,...,e and u<EvalRows, use mp=Mscan[g-1,ReadRow] when g>0, otherwise -inf; mt=Mscan[g,ReadRow], safe=mt when finite, otherwise 0, alpha=expf(mp-safe) when mp is finite, otherwise 0. For each evaluated Score entry, x=Score[u,v]*scale-safe, using a separate FP32 multiply and subtract. Define t=sat01(fma32(x,fp32(0.0057249800302088261),0.5)), w=fma_rm32(t,252,12582913), e1=fma32(x,fp32(1.4426950216293335),12583039-w), factor=bitcast_f32((bitcast_u32(w)<<23) mod 2^32), E1(x)=factor*ex2_ftz(e1). fma_rm32 rounds toward negative infinity; sat01 is the saturating conversion to [0,1], including NaN-to-zero; ex2_ftz is approximate base-two exponential with flush-to-zero. No low log2(e) correction is added. Prob[u,v]=E1(x), Prob8[u,v]=satfinite_e4m3_rne(Prob[u,v]). This is saturating RNE E4M3 conversion with unit probability scale; nonnegative finite inputs round min(Prob,ProbabilityMax). For z=0,1 and p=0,1 initialize E[z,p]=0, then add Prob[u,64*z+2*r+p] for r=0,...,31 in increasing order. DenPart(g)[u,z]=DenPart(g-1)[u,z]*alpha+(E[z,0]+E[z,1]). DenPart uses unrounded Prob. Rows u>=EvalRows keep DenPart=0 and have no probability overwrite or numerator rescale; their unspecified retained operands still enter dense PV but have no logical outputs. All valid rows are evaluated.

## t_8cbc3cc7483b5801 | math.math | 1209 chars | 1 declarations

tcg_d128_segmented_tma_direct_wide:193 design.math_reshape.rewrite[6].math

For each real row r let MergeChunk=ceil_div(Ns,MergeGroups). For z=0,...,MergeGroups-1 set MergeLo(z)=min(z*MergeChunk,Ns), MergeHi(z)=min(MergeLo(z)+MergeChunk,Ns). For each channel d, accumulate PartA[s,r,d] from zero by increasing s in [MergeLo(z),MergeHi(z)) with left-to-right FP32 additions, retaining GroupA[z,d]. For w=0,...,3 and l=0,...,31 define X[l]=PartB[MergeLo(z)+32*w+l,r] when MergeLo(z)+32*w+l<MergeHi(z), otherwise 0. Reduce X by a halving tree, adding the upper half to the lower half at offsets 16,8,4,2,1, and retain the final scalar as GroupB[z,w]. Empty intervals retain zero GroupA and four zero GroupB values. Sum GroupB[z,w] from zero in lexicographic (z,w) order to TotalB; sum GroupA[z,d] from zero in increasing z to TotalA[d]. Let live=(TotalB>0), inv=1/TotalB when live, otherwise 0. Write O[b,h,i,d]=bf16(TotalA[d]*inv) when live, otherwise bf16(0), using RNE. When LSE is present write MTail[r]+logf(TotalB/16) if live and MTail[r] is finite, otherwise -inf. Initial destination bytes are not read. All valid caller destinations are written and the host returns None. Workspace retains intermediates; no mathematical scratch output or scratch-zero initialization is required.

## t_8d72b2b93e8c01d7 | math.math | 1160 chars | 2 declarations

tc_split_segmented_narrow:147 design.math_reshape.rewrite[4].math; tc_split_segmented_wide:147 design.math_reshape.rewrite[4].math

Let Nc=ceil_div(Nseg-1,Sc). For each row r initialize ScanCarry(-1)[r]=-inf. Define a walk on [l,u) to process g=l,...,u-1 in order: ScanCarry(g)[r]=x(g,r) if x(g,r)-ScanCarry(g-1)[r]> fp32(2.7725887222397812), otherwise ScanCarry(g-1)[r]; store ScanState[g,r]=ScanCarry(g)[r]. NaN comparisons are false. First walk [0,min(Seg,T)) unconditionally. For c=0,...,Nc-1 let sb=1+c*Sc, t0=sb*Seg, t1=min((sb+Sc)*Seg,T), incoming=ScanCarry(t0-1)[r]. For k=0,...,Sc-1 let summary[k]=SegmentMax[sb+k,r] if sb+k<Nseg, otherwise -inf. Test all Sc summaries against this same incoming state. Let j be the smallest k with summary[k]-incoming>fp32(2.7725887222397812), or Sc when none qualifies. Set tp=min(t0+j*Seg,t1). For g=t0,...,tp-1 set ScanCarry(g)[r]=incoming and store ScanState[g,r]=incoming without reading x(g,r). Walk [tp,t1), including all segments after the first candidate without retesting their summaries against updated state. Store MaxTail[r]=ScanCarry(T-1)[r]. All T ScanState entries are written. BmCovered, PriorMax and SegmentMax die after replay. Let WalkCount(r)=min(Seg,T)+sum_{c=0}^{Nc-1}(t1(c)-tp(c,r)) count the group maxima reread by the scan.

## t_8e4f3fbec10fc793 | math.math | 819 chars | 1 declarations

tcg_d256_segmented_cp_wide:172 design.math_reshape.rewrite[5].math

For each nonempty split, NumState(l-1)=0 denotes the initial mathematical state. At g=l initialize PVAcc[-1]=0 by overwrite, without reading storage; later PVAcc[-1]=NumState(g-1)*alpha on evaluated rows, and NumState(g-1) on other rows. An exact unit rescale may be skipped. For k=0,...,3, PVAcc[k]=mma32(Prob8[:,32*k:32*k+32],Vb[32*k:32*k+32,:],PVAcc[k-1]); NumState(g)=PVAcc[3]. DenFinal[u]=DenPart(e)[u,0]+DenPart(e)[u,1]. For each valid row, weight=expf(Mscan[e,row]-Mtail[row]) when both states are finite, otherwise 0. PartA[s,row,d]=NumState(e)[u,d]*weight; PartB[s,row]=DenFinal[u]*weight. These multiplications still occur for weight=0, including nonfinite numerator behavior. An empty split writes PartA=0 and PartB=0 for every valid row, without reading a numerator. Discard local states after these writes.

## t_918f942c04c6de89 | math.math | 1588 chars | 7 declarations

tcg_d128_flat_cp_xpose_narrow:105 design.math_reshape.rewrite[3].math; tcg_d128_flat_cp_xpose_wide:88 design.math_reshape.rewrite[3].math; tcg_d128_flat_tma_xpose_narrow:105 design.math_reshape.rewrite[3].math; tcg_d128_flat_tma_xpose_wide:104 design.math_reshape.rewrite[3].math; tcg_d128_segmented_cp_xpose_narrow:110 design.math_reshape.rewrite[4].math; tcg_d128_segmented_tma_xpose_narrow:126 design.math_reshape.rewrite[4].math; tcg_d128_segmented_tma_xpose_wide:110 design.math_reshape.rewrite[4].math

For a numerator split let cLo=L(Ns,s), cHi=Hlast(Ns,s), end=max(cLo-1,cHi). Initialize DenState(cLo-1)[u]=0 for u<Br. In a nonempty split initialize MaxState(cLo-1)[u] from MState[cLo-1,r] when cLo>0, otherwise -inf. For g=cLo,...,cHi set MaxState(g)[u]=MState[g,r], safe[u]=MaxState(g)[u] if finite, otherwise 0, and alpha[u]=expf(MaxState(g-1)[u]-safe[u]) if the old state is finite, otherwise 0. Compute x=Score[u,v]*scale-safe[u] with separate FP32 multiply and subtract. The actual E16 approximation is t=sat01(fma32(x,fp32(0.0057249800302088261),0.5)), w=fma_rm32(t,252,12582917), e=fma32(x,fp32(1.4426950216293335),12583043-w), d=bitcast_f32((bitcast_u32(w)<<23) mod 2^32), E16(x)=d*ex2_ftz(e). fma_rm32 rounds toward negative infinity; sat01 saturates to [0,1] with NaN-to-zero; ex2_ftz is the source approximate base-two exponential with flush-to-zero. No low log2(e) correction is added. Prob[u,v]=E16(x), Prob8[u,v]=satfinite_e4m3_rne(Prob[u,v]), the source hardware saturating RNE cast with unit scale; for nonnegative finite values this rounds min(Prob,ProbabilityMax) to E4M3FN. For chunk c=0,...,3, partition z=0,...,3 and pair p=0,1, let F[c,z,p] be the left-to-right FP32 sum from zero of Prob[u,32*c+8*z+2*j+p], j=0,...,3. Sum (F[c,0,p]+F[c,1,p])+(F[c,2,p]+F[c,3,p]) over increasing c from zero to E[p]. Then DenState(g)[u]=DenState(g-1)[u]*alpha[u]+(E[0]+E[1]). This scalar update uses source compiler defaults without an additional contraction restriction; the denominator never uses Prob8. An empty split has no probability evaluation and leaves its denominator zero.

## t_97f24fc61d797a7f | math.math | 1305 chars | 1 declarations

tcg_d256_segmented_cp_wide:39 design.math_reshape.rewrite[0].math

Let G=H/KH, Qph=Br/G, q0=a*Qph and 0<=a<ceil_div(Sq,Qph). Each (b,kh,a) instance has Br rows u. With PSK, map i=q0+floor(u/G), h=kh*G+(u mod G); otherwise map i=q0+(u mod Qph), h=kh*G+floor(u/Qph). A row is valid iff i<Sq; ic=min(i,Sq-1), row=(b*H+h)*Sq+i. Let ReadRow=(b*H+h)*Sq+(i if valid, otherwise 0). Let EvalRows=32*ceil_div(G*min(Qph,Sq-q0),32) with PSK, otherwise Br. Scalar score/probability evaluation covers u<EvalRows; dense products still evaluate every Br row. Only valid rows have results. Use the Contract's scale addressing and inclusive mask. Define lo(u)=max(0,ic+Skv-Sq-wl) when active, otherwise 0, hi(u)=ic+Skv-Sq when is_causal, otherwise Skv-1, gLo=floor((active ? max(0,q0+Skv-Sq-wl) : 0)/KeyGroup), gHi=floor((is_causal ? min(q0+Qph-1,Sq-1)+Skv-Sq : Skv-1)/KeyGroup), Ng=gHi-gLo+1>0. For partition count n in {Nm,Ns}, let per(n)=ceil_div(Ng,n), L(n,s)=gLo+s*per(n), Hlast(n,s)=min(L(n,s)+per(n)-1,gHi), 0<=s<n. A partition is empty when L(n,s)>Hlast(n,s). Qb is the original E4M3 query with its E8M0 block scales; invalid query payloads are zero with unit scales. For j=g*KeyGroup+v, Kb[v,d] and Vb[v,d] are the original E4M3 operands with their scales. Payloads are zero for j>=Skv, but allocated padding scale bytes are still used. Only groups gLo..gHi execute dense products.

## t_98ed5c0fa069efca | math.math | 319 chars | 3 declarations

tcg_d256_segmented_cp_narrow:77 design.math_reshape.rewrite[2].math; tcg_d256_segmented_tma_narrow:92 design.math_reshape.rewrite[2].math; tcg_d256_segmented_tma_wide:77 design.math_reshape.rewrite[2].math

Nseg=ceil_div(GCount,SegLen). For 0<=f<Nseg and each row, SegMax[f,row]=fmax over GroupMax[t,row] in ascending t for f*SegLen<=t<min((f+1)*SegLen,GCount), starting at -inf and padding the final segment to SegLen entries with -inf. Preserve fmaxf NaN behavior. Save every segment; SegMax[0,row] is produced but not read.

## t_9a4a6df5f50ad57f | math.math | 1545 chars | 1 declarations

tcg_d128_fscan_cp_xpose_wide:178 design.math_reshape.rewrite[5].math

For each real row r and 0<=s<Ns let Weight[s]= expf(PartM[s,r]-MTail[r]) when both maxima are finite, otherwise 0. Let Chunk=ceil_div(Ns,MergeGroups). For 0<=cg<MergeGroups set sLo(cg)=min(cg*Chunk,Ns), sHi(cg)=min(sLo(cg)+Chunk,Ns). NumGroup[cg,d] accumulates PartA[s,r,d]*Weight[s] from zero by increasing s in [sLo(cg),sHi(cg)), retaining the scalar compiler path. The first eight-entry batch adds zero for missing entries; subsequent full batches and the scalar tail visit only actual entries. For 0<=t<128, set BV[cg,t]=PartB[sLo(cg)+t,r]*Weight[sLo(cg)+t] when sLo(cg)+t<sHi(cg), otherwise 0*0. Define tree32(x) by y=x, then replace y by y_new[l]=y[l]+y[l+o] for 0<=l<o at o=16,8,4,2,1; tree32(x) is the final y[0]. For 0<=w<4, DenGroup[cg,w]=tree32(BV[cg,32*w:32*w+32]). TotalB accumulates DenGroup[cg,w] from zero in increasing cg then w; TotalA[d] accumulates NumGroup[cg,d] from zero in increasing cg. All Ns entries are read, including empty splits even when PartB is zero; zero weights do not suppress NaN or Inf in PartA. Empty merge groups still form zero NumGroup and DenGroup. Set live=(TotalB>0), inv=1/TotalB when live, otherwise 0. Write O[b,h,i,d]=bf16(TotalA[d]*inv) when live, otherwise bf16(0), with RNE. If LSE is present, write MTail[r]+logf(TotalB/16) when live and MTail[r] is finite, otherwise -inf. Initial O and LSE bytes are never read; all valid destinations are written. The host returns None. Workspace retains intermediates without a mathematical output requirement; no scratch zero initialization is required.

## t_9a9361184c4c814e | math.math | 761 chars | 3 declarations

tcg_d128_segmented_cp_xpose_narrow:182 design.math_reshape.rewrite[6].math; tcg_d128_segmented_tma_xpose_narrow:198 design.math_reshape.rewrite[6].math; tcg_d128_segmented_tma_xpose_wide:182 design.math_reshape.rewrite[6].math

For each real row r, accumulate PartA[s,r,d] and PartB[s,r] from zero by increasing s with left-to-right FP32 additions, obtaining TotalA[d] and TotalB. Read all Ns partials, including empty splits regardless of PartB being zero. The eight-at-a-time loop also adds zero for missing entries in its final batch. Let live=(TotalB>0), inv=1/TotalB when live, otherwise 0. Write O[b,h,i,d]=bf16(TotalA[d]*inv) when live, otherwise bf16(0), using RNE. When LSE is present write MTail[r]+logf(TotalB/16) if live and MTail[r] is finite, otherwise -inf. Initial destination bytes are not read. All valid caller destinations are written and the host returns None. Workspace retains intermediates; no mathematical scratch output or scratch-zero initialization is required.

## t_9afb2c3cca7a9547 | math.math | 794 chars | 3 declarations

fused_d128_cp:132 design.math_reshape.rewrite[3].math; fused_d128_group:131 design.math_reshape.rewrite[3].math; root:342 design.math_reshape.rewrite[3].math

Let NumState(cLo-1)=0 denote the initial mathematical numerator. For g=cLo, start C[-1]=0 using overwrite semantics, without reading uninitialized storage. For later groups, C[-1][u,d]= NumState(g-1)[u,d]*alpha[u]; a rescale by exactly one may be skipped. For k=0,...,3, C[k]=mma32(Prob8[:,32*k:32*k+32], Vb[32*k:32*k+32,:],C[k-1]). NumState(g)=C[3]. At g=cHi, let live[u]=(DenState(g)[u]>0) and inv[u]=1/DenState(g)[u] when live, otherwise 0. For each valid row write O[b,h,i,d]=bf16(NumState(g)[u,d]*inv[u]) when live, otherwise bf16(0), using RNE conversion. When LSE is present write MaxState(g)[u]+logf(DenState(g)[u]/16) if live and MaxState is finite, otherwise -inf. NumState is discarded after output. This fused path leaves workspace untouched and returns None through the host entry.

## t_a52004a377bf30d9 | math.math | 938 chars | 1 declarations

tcg_d256_segmented_cp_wide:105 design.math_reshape.rewrite[3].math

Define tau=fp32(2.7725887222397812) and advance(m,x)=x if x-m>tau, otherwise m; comparison is false for NaN. Independently for each row, Mscan(-1,row)=-inf. For t=0,...,min(SegLen,T)-1, unconditionally evaluate Mscan[t,row]=advance(Mscan(t-1,row),GroupMax[t,row]). For c=0,...,ceil_div(Nseg-1,SegChunk)-1, let f0=1+c*SegChunk, t0=f0*SegLen, t1=min((f0+SegChunk)*SegLen,T), incoming=Mscan[t0-1,row]. For z=0,...,SegChunk-1, let Hseg[z]=SegMax[f0+z,row] when f0+z<Nseg, otherwise -inf. Let Jump={z | 0<=z<SegChunk and Hseg[z]-incoming>tau}. Let tp=t1 if Jump is empty, otherwise min(t0+min(Jump)*SegLen,t1). For t0<=t<tp, write Mscan[t,row]=incoming without reading GroupMax. For tp<=t<t1, evaluate Mscan[t,row]=advance(Mscan(t-1,row),GroupMax[t,row]). The entire suffix is replayed without retesting individual segments. Mtail[row]=Mscan[T-1,row]. Retain every Mscan entry for split replay and Mtail for partial weighting and optional LSE.

## t_aa238c1e61749ec8 | math.math | 1189 chars | 1 declarations

tcg_d128_flat_cp_direct_wide:171 design.math_reshape.rewrite[5].math

For each real row r let MergeChunk=ceil_div(Ns,MergeGroups). For q=0,...,MergeGroups-1 define sLo(q)=min(q*MergeChunk,Ns) and sHi(q)=min(sLo(q)+MergeChunk,Ns). GroupA[q,d] is the left-to-right FP32 sum from zero of PartA[s,r,d] for s=sLo(q),...,sHi(q)-1; an empty interval yields zero. For each q and w=0,...,3, define Tree(0)[l]=PartB[sLo(q)+32*w+l,r] when that split is below sHi(q), otherwise 0, for l=0,...,31. For k=1,...,5 and 0<=l<2^(5-k), Tree(k)[l]=Tree(k-1)[l]+Tree(k-1)[l+2^(5-k)] in FP32. GroupB[q,w]=Tree(5)[0]. TotalA[d] sums GroupA[q,d] from zero in increasing q; TotalB sums GroupB[q,w] from zero in increasing q, then increasing w. Both final sums use left-to-right FP32 additions. Let live=(TotalB>0), inv=1/TotalB when live, otherwise 0. Write O[b,h,i,d]=bf16(TotalA[d]*inv) when live, otherwise bf16(0), using RNE. When LSE is present write MTail[r]+logf(TotalB/16) if live and MTail[r] is finite, otherwise -inf. Initial destination bytes are not read. All valid caller destinations are written and the host returns None. Workspace retains intermediates; no mathematical scratch output or scratch-zero initialization is required. The segment-max reservation is unused.

## t_ae0d2f0b5a8c71a4 | math.math | 348 chars | 1 declarations

tcg_d256_segmented_cp_wide:92 design.math_reshape.rewrite[2].math

Let Nseg=ceil_div(T,SegLen). For 0<=f<Nseg and each row, SegMax[f,row]=fmax over GroupMax[t,row] in ascending t for f*SegLen<=t<min((f+1)*SegLen,T), starting at -inf and padding the final segment with -inf to SegLen entries. Preserve fmaxf NaN behavior. Every segment is materialized, including segment 0, whose maximum is not consumed by the scan.

## t_afdf7d235c341c80 | math.math | 1355 chars | 1 declarations

tc_split_flat_wide:219 design.math_reshape.rewrite[5].math

For each row r=(b*H+h)*Sq+i, let Cb=ceil_div(Ns,4), aLo(f)=min(f*Cb,Ns), aHi(f)=min(aLo(f)+Cb,Ns), n(f)=aHi(f)-aLo(f), f=0,...,3. The host plan bounds Ns<=512. For each f and channel d initialize GroupAccum(-1)[d]=0. For s=0,...,n(f)-1, GroupAccum(s)[d]=GroupAccum(s-1)[d]+ PartialA[aLo(f)+s,r,d], in ascending FP32 order without tail padding. Save GroupA[f,d]=GroupAccum(n(f)-1)[d]; empty intervals save zero. For each f and w=0,...,3, initialize components l=0,...,31: Btree(0)[l]=PartialB[aLo(f)+32*w+l,r] if 32*w+l<n(f), otherwise 0. For t=1,...,5 set delta=2^(5-t) and, for l=0,...,31, Btree(t)[l]=Btree(t-1)[l]+Btree(t-1)[l+delta if l+delta<32, otherwise l]. Save GroupB[f,w]=Btree(5)[0]; discard other final components. Starting from zero, add GroupB[f,w] in lexicographic (f,w) order to obtain total, and GroupA[f,d] in ascending f order to obtain numerator[d]. These are FP32 sums including zero groups. Set live=(total>0), inv=1/total if live, otherwise 0. Write O[b,h,i,d]=bf16(numerator[d]*inv) if live, otherwise bf16(0), using RNE. When LSE is present, write MaxTail[r]+logf(total/beta) if live and MaxTail[r] is finite, otherwise -inf. No initial O/LSE values are read. GroupA and GroupB die after the final sums. PartialA, PartialB and MaxTail die after the last consumer. Workspace retains its written bytes; the Python call returns None.

## t_b1e8d0f9890eaebf | math.math | 427 chars | 1 declarations

fused_d128_group:67 design.math_reshape.rewrite[1].math

Start A[-1]=0 and for k=0,...,3 form A[k]=mma32(Qb[:,32*k:32*k+32],Kb[:,32*k:32*k+32]^T,A[k-1]). Score[u,v]=A[3][u,v] if lo(u)<=g*KeyGroup+v<=hi(u), and -inf otherwise. Let scale=fp32(D^(-1/2)). For p=0,...,3, M4[p,u] is the left-to-right fmax reduction of Score[u,4*s+p], s=0,...,31, starting at -inf. RawMax[u]=fmax(fmax(M4[0,u],M4[1,u]),fmax(M4[2,u],M4[3,u])); use the source fmaxf NaN behavior. GroupMax[u]=RawMax[u]*scale.

## t_b57ee59eacc7ddb9 | math.math | 1551 chars | 1 declarations

fused_d128_group:89 design.math_reshape.rewrite[2].math

Initialize MaxState(cLo-1)[u]=-inf and DenState(cLo-1)[u]=0. MaxState(g)[u]=GroupMax[u] if GroupMax[u]-MaxState(g-1)[u]> fp32(2.7725887222397812), otherwise MaxState(g-1)[u]. The comparison is false for NaN. safe[u]=MaxState(g)[u] if finite, otherwise 0. alpha[u]=expf(MaxState(g-1)[u]-safe[u]) if the old state is finite, otherwise 0. For each Score entry, x=(Score[u,v]*scale)-safe[u], with a separate FP32 multiply and subtract. Define the actual E16 approximation by: t=sat01(fma32(x,fp32(0.0057249800302088261),0.5)), w=fma_rm32(t,252,12582917), e=fma32(x,fp32(1.4426950216293335),12583043-w), d=bitcast_f32((bitcast_u32(w)<<23) mod 2^32), E16(x)=d*ex2_ftz(e). Here fma_rm32 rounds the fused result toward negative infinity; sat01 is the source saturating FP32 conversion to [0,1], including its NaN-to-zero behavior; ex2_ftz is the source approximate base-two exponential with flush-to-zero. No low log2(e) correction is added. Prob[u,v]=E16(x); Prob8[u,v]=satfinite_e4m3_rne(Prob[u,v]). The latter is the source hardware saturating RNE conversion, with unit probability scale; for nonnegative finite values it rounds min(Prob,ProbabilityMax) to E4M3FN. For p=0,1, E[p,u] sums Prob[u,2*s+p] over s=0,...,63 by left-to-right FP32 adds from zero, continuing across all four 32-key chunks. DenState(g)[u]=DenState(g-1)[u]*alpha[u]+(E[0,u]+E[1,u]), using the source's scalar FP32 update under its compiler defaults; no additional contraction restriction is imposed. The denominator never uses Prob8. MaxState and DenState are discarded after output.

## t_b59e015def6b6aff | math.math | 816 chars | 8 declarations

tcg_d128_flat_cp_xpose_narrow:67 design.math_reshape.rewrite[1].math; tcg_d128_flat_cp_xpose_wide:50 design.math_reshape.rewrite[1].math; tcg_d128_flat_tma_xpose_narrow:67 design.math_reshape.rewrite[1].math; tcg_d128_flat_tma_xpose_wide:66 design.math_reshape.rewrite[1].math; tcg_d128_fscan_tma_xpose_narrow:66 design.math_reshape.rewrite[1].math; tcg_d128_segmented_cp_xpose_narrow:50 design.math_reshape.rewrite[1].math; tcg_d128_segmented_tma_xpose_narrow:66 design.math_reshape.rewrite[1].math; tcg_d128_segmented_tma_xpose_wide:50 design.math_reshape.rewrite[1].math

For each group g in a nonempty split, A(-1)=0 by overwrite and A(k)=mma32(Qb[:,32*k:32*k+32],Kb[:,32*k:32*k+32]^T,A(k-1)), k=0,...,3. DotTile=A(3). For u<Br, Score[u,v]=DotTile[u,v] when lo(u)<=g*KeyGroup+v<=hi(u), otherwise -inf. Let scale=fp32(D^(-1/2)). In the group-max evaluation, GroupMax[g,r]=scale*fmax(Score[u,:]) for valid rows. Reduce from -inf with source fmaxf NaN behavior: four 32-value chunk maxima followed by fmaxf across chunks; masked chunks use a halving tree, wholly unmasked chunks can use the native max reduction. GroupMax[g,r]=-inf for g<gLo or g>gHi, explicitly filled by the first or last group-max split. No matrix product evaluates those groups. The numerator evaluation recomputes DotTile with the same contractions but consumes Score in probability chunks without a maximum reduction.

## t_b722793ecf1357c2 | math.math | 1161 chars | 1 declarations

tcg_d128_segmented_cp_direct_wide:192 design.math_reshape.rewrite[6].math

For each real row r let Cmerge=ceil_div(Ns,MergeGroups). For z=0,...,MergeGroups-1 set a_z=min(z*Cmerge,Ns), b_z=min(a_z+Cmerge,Ns). MergeA[z,d] is the left-to-right FP32 sum from zero of PartA[s,r,d] over increasing s=a_z,...,b_z-1. Empty intervals give zero; nonexistent entries cause no additions. For each z and w=0,...,3 define Y(32)[l]=PartB[a_z+32*w+l,r] when a_z+32*w+l<b_z, otherwise zero, for l=0,...,31. For k=16,8,4,2,1 set Y(k)[l]=Y(2*k)[l]+Y(2*k)[l+k], 0<=l<k, with FP32 additions, then MergeB[z,w]=Y(1)[0]. TotalB is the left-to-right FP32 sum from zero of MergeB[z,w], ordered by increasing z then increasing w. TotalA[d] is the left-to-right FP32 sum from zero of MergeA[z,d] in increasing z. Let live=(TotalB>0), inv=1/TotalB when live, otherwise 0. Write O[b,h,i,d]=bf16(TotalA[d]*inv) when live, otherwise bf16(0), using RNE. When LSE is present write MTail[r]+logf(TotalB/16) if live and MTail[r] is finite, otherwise -inf. Initial destination bytes are not read. All valid caller destinations are written and the host returns None. Workspace retains intermediates; no mathematical scratch output or scratch-zero initialization is required.

## t_bc6dba286778e3b0 | math.math | 1368 chars | 6 declarations

fused_d128_cp:41 design.math_reshape.rewrite[0].math; fused_d128_group:40 design.math_reshape.rewrite[0].math; fused_d128_key:41 design.math_reshape.rewrite[0].math; fused_d256_cp:37 design.math_reshape.rewrite[0].math; fused_d256_tma:37 design.math_reshape.rewrite[0].math; root:251 design.math_reshape.rewrite[0].math

Let G=H/KH, Qph=Br/G and q0=a*Qph, where 0 <= a < ceil_div(Sq,Qph). Each (b,kh,a) instance has rows u=0,...,Br-1 mapped to h=kh*G+floor(u/Qph), i=q0+(u mod Qph). A row is valid iff i<Sq; set ic=min(i,Sq-1) for mask calculation. Use the Contract's scale addressing and inclusive mask definitions. Qb denotes the original E4M3 query with its E8M0 block scales for a valid row; invalid query payloads are zero with unit scales. Let lo(u)=max(0,ic+Skv-Sq-window_left) when the window is active, otherwise 0; hi(u)=ic+Skv-Sq when causal, otherwise Skv-1. Let imin=q0, imax=min(q0+Qph-1,Sq-1), cLo=floor((active ? max(0,imin+Skv-Sq-window_left) : 0)/KeyGroup), cHi=floor((is_causal ? imax+Skv-Sq : Skv-1)/KeyGroup), Ng=cHi-cLo+1, which is positive in this node's valid domain. Process g=cLo,...,cHi in increasing order. Within each group j=g*KeyGroup+v, 0<=v<KeyGroup; Kb[v,d] and Vb[v,d] denote the original E4M3 operands and their scales at (b,kh,j,d), with zero payload for j>=Skv. Scale addressing extends through the allocated padded final group: stored padding atom bytes are still loaded, not replaced by unit scales. Groups outside this interval are not evaluated. All Br rows and all KeyGroup keys of an evaluated group take part in the products, including padded rows and masked keys. Only valid rows are written to caller storage; padding creates no extra output.

## t_bcdf72691cb8ad4b | math.math | 770 chars | 8 declarations

tcg_d128_flat_cp_direct_narrow:142 design.math_reshape.rewrite[4].math; tcg_d128_flat_cp_direct_wide:142 design.math_reshape.rewrite[4].math; tcg_d128_flat_tma_direct_narrow:143 design.math_reshape.rewrite[4].math; tcg_d128_flat_tma_direct_wide:143 design.math_reshape.rewrite[4].math; tcg_d128_segmented_cp_direct_narrow:164 design.math_reshape.rewrite[5].math; tcg_d128_segmented_cp_direct_wide:163 design.math_reshape.rewrite[5].math; tcg_d128_segmented_tma_direct_narrow:164 design.math_reshape.rewrite[5].math; tcg_d128_segmented_tma_direct_wide:164 design.math_reshape.rewrite[5].math

Initialize NumState(cLo-1)[u,d]=0 for u<C. For every nonempty-split group, VAcc(-1)=0 by overwrite and VAcc(k)=mma32(Prob8[:,32*k:32*k+32], Vb[32*k:32*k+32,:],VAcc(k-1)), k=0,...,3. Let GroupPV=VAcc(3). Rescale NumState(g-1) by alpha unless alpha=1, then separately add GroupPV in FP32 to obtain NumState(g). The native PV accumulator is fresh each group, not carried through groups. For each valid row of a nonempty split, weight=expf(MaxState(cHi)[u]-MTail[r]) when both maxima are finite, otherwise 0. Store PartA[s,r,d]=NumState(cHi)[u,d]*weight and PartB[s,r]=DenState(cHi)[u]*weight in FP32, without BF16 narrowing. A nonempty split's multiplication by zero does not suppress NaN/Inf. Empty splits store zero PartA and PartB. Discard local states after the stores.

## t_bd277a4b2e5fec7e | math.math | 851 chars | 2 declarations

fused_d256_cp:127 design.math_reshape.rewrite[3].math; fused_d256_tma:127 design.math_reshape.rewrite[3].math

Let NumState(cLo-1)=0 denote the initial mathematical numerator. For g=cLo, start C[-1]=0 using overwrite semantics, without reading uninitialized storage. For later groups, C[-1][u,d]= NumState(g-1)[u,d]*alpha[u]; a rescale by exactly one may be skipped. For k=0,...,3, C[k]=mma32(Prob8[:,32*k:32*k+32], Vb[32*k:32*k+32,:],C[k-1]). NumState(g)=C[3]. After the last group form DenFinal[u]=DenPart(cHi)[u,0]+DenPart(cHi)[u,1]. Let live[u]=(DenFinal[u]>0) and inv[u]=1/DenFinal[u] when live, otherwise 0. For each valid row write O[b,h,i,d]=bf16(NumState(cHi)[u,d]*inv[u]) when live, otherwise bf16(0), using RNE conversion. When LSE is present write MaxState(cHi)[u]+logf(DenFinal[u]) if live and MaxState is finite, otherwise -inf. NumState is discarded after output. This fused path leaves workspace untouched and returns None through the host entry.

## t_c11b1f9c85acb692 | math.math | 1416 chars | 1 declarations

tcg_d128_flat_tma_xpose_wide:176 design.math_reshape.rewrite[5].math

For each real row r let C=ceil_div(Ns,SP). For t=0,...,SP-1 define ALo(t)=min(t*C,Ns), AHi(t)=min(ALo(t)+C,Ns). MergeA[t,d] is the left-to-right FP32 sum from zero of PartA[s,r,d] for s=ALo(t),...,AHi(t)-1. Full eight-entry batches are followed by a scalar remainder, without adding nonexistent tail entries. For each t and segment z=0,...,3 initialize v[l], l=0,...,31, from PartB[ALo(t)+32*z+l,r] if that split is below AHi(t), otherwise zero. For o=16,8,4,2,1 in order, simultaneously replace v[l] by v[l]+v[l+o] for 0<=l<o; MergeB[t,z]=v[0]. The bound Ns<=128*SP ensures coverage. TotalA[d] sums MergeA[t,d] from zero in increasing t. TotalB sums MergeB[t,z] from zero with t outermost, z innermost, both increasing. These sums use left-to-right FP32 additions. Empty merge intervals produce zero partials. Every materialized split is read, including empty splits regardless of PartB being zero; no new weight is applied. Let live=(TotalB>0), inv=1/TotalB when live, otherwise 0. Write O[b,h,i,d]=bf16(TotalA[d]*inv) when live, otherwise bf16(0), using RNE. When LSE is present write MTail[r]+logf(TotalB/16) if live and MTail[r] is finite, otherwise -inf. Initial destination bytes are not read. All valid caller destinations are written and the host returns None. Workspace retains intermediates; no mathematical scratch output or scratch-zero initialization is required. The segment-max reservation is unused.

## t_c16d037db54b2135 | math.math | 1560 chars | 1 declarations

tcg_d128_fscan_tma_xpose_wide:178 design.math_reshape.rewrite[5].math

For each real row r and 0<=s<Ns let Weight[s]= expf(PartM[s,r]-MTail[r]) when both maxima are finite, otherwise 0. Let chunk=ceil_div(Ns,Cg), slo(z)=min(z*chunk,Ns), shi(z)=min(slo(z)+chunk,Ns), for z=0,...,Cg-1. The host bounds Ns<=128*Cg, so each interval has at most 128 entries. For each z and channel d, GroupA[z,d] is the FP32 running sum from zero of PartA[s,r,d]*Weight[s] in increasing s from slo(z) to shi(z)-1. Its first batch always has eight additions, adding zero without reads for missing entries. Subsequent full batches of eight and the scalar remainder include only present entries. All Ns stored partials are read, including empty splits; a zero weight does not suppress NaN/Inf. For w=0,...,3 define a 32-element vector x with x[l]=PartB[s,r]*Weight[s] for s=slo(z)+32*w+l<shi(z), otherwise 0. Reduce x in FP32 by successive offsets o=16,8,4,2,1, replacing x[l] by x[l]+x[l+o] for 0<=l<o. BinB[z,w]=x[0]. TotalB is a FP32 running sum from zero of BinB[z,w], with z increasing outermost and w increasing innermost. TotalA[d] is a FP32 running sum from zero of GroupA[z,d] in increasing z. Set live=(TotalB>0), inv=1/TotalB when live, otherwise 0. Write O[b,h,i,d]=bf16(TotalA[d]*inv) when live, otherwise bf16(0), with RNE. If LSE is present, write MTail[r]+logf(TotalB/16) when live and MTail[r] is finite, otherwise -inf. Initial O and LSE bytes are never read; all valid destinations are written. The host returns None. Workspace retains intermediates without a mathematical output requirement; no scratch zero initialization is required.

## t_c3850e17c8eb47ef | math.math | 1243 chars | 4 declarations

tc_split_flat_narrow:143 design.math_reshape.rewrite[3].math; tc_split_flat_wide:143 design.math_reshape.rewrite[3].math; tc_split_segmented_narrow:178 design.math_reshape.rewrite[5].math; tc_split_segmented_wide:178 design.math_reshape.rewrite[5].math

For each value interval z let cLo=L(a,z,Ns), cHi=U(a,z,Ns), gLast=max(cLo-1,cHi). Initialize DenPart(cLo-1)[u,r]=0, r=0,...,3. Let beta=16 when D=128, otherwise 1. For g=cLo,...,cHi recompute scores; the resulting GroupMax is unused. Read mt[u]=ScanState[g,rho(u)] and mp[u]=ScanState[g-1,rho(u)] when g>0, otherwise -inf. safe[u]=mt[u] if finite, otherwise 0. alpha[u]=expf(mp[u]-safe[u]) if mp[u] is finite, otherwise 0. Prob[u,v]=expf(Score[u,v]-safe[u])*beta, with separate FP32 subtraction, exponential and bias multiplication. Prob8[u,v]=satfinite_e4m3_rne(Prob[u,v]), the hardware saturating RNE conversion; finite nonnegative values saturate at ProbabilityMax. For each r=0,...,3 and p=0,1, compute E[u,r,p]=sum_{n=0}^{15} Prob[u,8*n+2*r+p] using increasing n, left-to-right FP32 additions starting from zero. DenPart(g)[u,r]=DenPart(g-1)[u,r]*alpha[u]+(E[u,r,0]+E[u,r,1]), following the source scalar FP32 update under compiler defaults; no additional contraction restriction is imposed on this update. Multiplication of carried state by exactly one may be skipped. The four partials are not merged between groups. Only unrounded Prob supplies DenPart. Fully masked entries have Prob=Prob8=0; their value products are still evaluated.

## t_c4e60f0f1d2783b0 | math.math | 1156 chars | 1 declarations

tcg_d256_flat_cp_wide:162 design.math_reshape.rewrite[5].math

For each row, chunk=ceil_div(Ns,Cg), cL(cg)=min(cg*chunk,Ns), cH(cg)=min(cL(cg)+chunk,Ns), 0<=cg<Cg. Each interval has at most 128 splits. GroupA[cg,d] is the FP32 sum of PartA[s,row,d], starting at zero and adding in ascending s over cL(cg)<=s<cH(cg). Empty intervals yield zero; no padded numerator terms are added. For z=0,...,3, form 32 values indexed by k=0,...,31: v[k]=PartB[cL(cg)+32*z+k,row] if that split is below cH(cg), else zero. At offsets 16,8,4,2,1, replace v[k] by v[k]+v[k+offset] for 0<=k<offset, retaining the lower half. GroupB[cg,z]=v[0] after this FP32 tree. OutNum starts at zero and adds GroupA[cg,d] in ascending cg. OutDen starts at zero and adds GroupB[cg,z] in ascending cg with ascending z inside each cg. GroupA and GroupB then die. Let live=OutDen>0 and inv=1/OutDen when live, otherwise 0. O[b,h,i,d]=bf16(OutNum*inv) when live, otherwise bf16(0), using RNE. When LSE is present, LSE[b,h,i]=Mtail[row]+logf(OutDen) if live and Mtail is finite, otherwise -inf. Initial O/LSE contents are never read. Workspace is overwritten with the intermediates; unrelated scratch bytes have no mathematical result. The host returns None.

## t_c802cd9a8211c64e | math.math | 999 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:104 design.math_reshape.rewrite[3].math

Let tau=fp32(2.7725887222397812). For each r initialize ScanState(-1)=-inf. Walking a group g means evaluating ScanState(g)=GroupMax[g,r] if GroupMax[g,r]-ScanState(g-1)>tau, otherwise ScanState(g-1), then storing MState[g,r]=ScanState(g). All comparisons are false for NaN. Walk g=0,...,min(SegLen,T)-1. For z0=1,1+SegBatch,...<Nseg, let t0=z0*SegLen, t1=min((z0+SegBatch)*SegLen,T), m=ScanState(t0-1). For each k=0,...,SegBatch-1 test v[k]-m>tau, where v[k]=SegMax[z0+k,r] if z0+k<Nseg, otherwise -inf. Let kfirst be the smallest passing k, when any exists. Set tp=t0+kfirst*SegLen when a test passes, otherwise tp=t1. For t0<=g<tp assign ScanState(g)=m and store MState[g,r]=m, without reading GroupMax[g,r]. Walk every g from tp through t1-1, including segments after the first passing segment regardless of their individual tests. Store MTail[r]=ScanState(T-1). Let WalkCount(r) count groups actually walked, including segment zero; min(SegLen,T)<=WalkCount(r)<=T. Every MState entry is stored.

## t_ccb91736e2e78d1b | math.math | 840 chars | 1 declarations

tcg_d128_flat_tma_xpose_narrow:144 design.math_reshape.rewrite[4].math

Initialize NumState(cLo-1)[u,d]=0 for u<Br. For every nonempty-split group, VAcc(-1)=0 by overwrite and VAcc(k)=mma32(Prob8[:,32*k:32*k+32], Vb[32*k:32*k+32,:],VAcc(k-1)), k=0,...,3. Let GroupPV=VAcc(3). Rescale NumState(g-1) by alpha unless alpha=1, then separately add GroupPV in FP32 to obtain NumState(g). The native PV accumulator is fresh each group, not carried through groups. For each valid row of a nonempty split, weight=expf(MaxState(cHi)[u]-MTail[r]) when both maxima are finite, otherwise 0. Store PartA[s,r,d]=NumState(cHi)[u,d]*weight and PartB[s,r]=DenState(cHi)[u]*weight in FP32, without BF16 narrowing. A nonempty split's multiplication by zero does not suppress NaN/Inf. An empty split has NumState=DenState=weight=0 and stores PartA[s,r,d]=0 and PartB[s,r]=0 for every valid row. Discard local states after the stores.

## t_ce3085dc3d3ad36b | math.math | 1240 chars | 3 declarations

tcg_d128_fscan_cp_xpose_narrow:143 design.math_reshape.rewrite[4].math; tcg_d128_fscan_cp_xpose_wide:143 design.math_reshape.rewrite[4].math; tcg_d128_fscan_tma_xpose_wide:143 design.math_reshape.rewrite[4].math

Initialize NumState(cLo-1)[u,d]=0. For each nonempty split group, C[-1]=0 and C[k]=mma32(Prob8[:,32*k:32*k+32], Vb[32*k:32*k+32,:],C[k-1]), k=0,...,3; GroupPV=C[3]. First rescale the numerator state by alpha unless alpha equals one, then add GroupPV in FP32: NumState(g)=NumState(g-1)*alpha+GroupPV. This is a fresh PV accumulation per group followed by a separate addition, not the fused path's carried native MMA accumulator. For each valid row set wgt=1 for a nonempty split, otherwise 0. Store PartA[s,r,d]=NumState(max(cLo-1,cHi))[u,d]*wgt and PartB[s,r]=DenState(max(cLo-1,cHi))[u]*wgt, both FP32, without BF16 narrowing. For an empty split, NumState=0, DenState=0 and wgt=0 for every valid row, so the evaluated numerator and denominator to be stored are zero. PartB[s,r]=0. The materialized PartA slice is not guaranteed to be zero: an unsynchronized overwrite in the fixed source can corrupt overlapping entries. PartA denotes the actual stored FP32 values, including this empty-split case, rather than the evaluated zeros. MaxState remains available through PartM and last-split MTail publication; DenState and NumState are discarded after their partial stores. No multiplication by zero suppresses NaN or Inf in a nonempty split.

## t_d02eb7d2f8a9de3a | math.math | 956 chars | 3 declarations

tcg_d128_segmented_cp_xpose_narrow:86 design.math_reshape.rewrite[3].math; tcg_d128_segmented_tma_xpose_narrow:102 design.math_reshape.rewrite[3].math; tcg_d128_segmented_tma_xpose_wide:86 design.math_reshape.rewrite[3].math

Let tau=fp32(2.7725887222397812). For each row r, ScanState(-1)=-inf. Walking group g evaluates ScanState(g)=GroupMax[g,r] when GroupMax[g,r]-ScanState(g-1)>tau, otherwise ScanState(g-1), then stores MState[g,r]=ScanState(g). Comparisons are false for NaN. Walk g=0,...,min(SegLen,T)-1 unconditionally. For z0=1,1+SegBatch,...<Nseg, let t0=z0*SegLen, t1=min((z0+SegBatch)*SegLen,T), m=ScanState(t0-1). Test Bound[k]-m>tau for k=0,...,SegBatch-1, where Bound[k]=SegMax[z0+k,r] when z0+k<Nseg, otherwise -inf. Let kfirst be the smallest passing k. Set tp=t0+kfirst*SegLen when any test passes, otherwise tp=t1. For t0<=g<tp assign ScanState(g)=m and store MState[g,r]=m without reading GroupMax[g,r]. Walk every group tp<=g<t1, including later segments regardless of their individual tests. Store MTail[r]=ScanState(T-1). Let WalkCount(r) count the actually walked groups, including segment zero; min(SegLen,T)<=WalkCount(r)<=T. Every MState entry is stored.

## t_d1a10042c64edf05 | math.math | 1086 chars | 1 declarations

tcg_d256_flat_tma_wide:161 design.math_reshape.rewrite[5].math

For each row, Chunk=ceil_div(Ns,SP), SLo(z)=min(z*Chunk,Ns), SHi(z)=min(SLo(z)+Chunk,Ns), 0<=z<SP. Initialize GroupNum[z,d]=0 and add PartA[s,row,d] in ascending s over [SLo(z),SHi(z)); an empty interval retains zero, and no tail zeros are added. For each z and w=0,...,3, initialize Dbuf[r]=PartB[SLo(z)+32*w+r,row] when SLo(z)+32*w+r<SHi(z), otherwise 0, for r=0,...,31. At offsets q=16,8,4,2,1 in order, simultaneously replace Dbuf[r] by Dbuf[r]+Dbuf[r+q] for 0<=r<q. GroupDen[z,w]=Dbuf[0]. Start OutDen=0 and add GroupDen[z,w] in increasing z, then w. For each channel start OutNum=0 and add GroupNum[z,d] in increasing z. All these additions are FP32; no split weighting is repeated here. Let live=OutDen>0 and inv=1/OutDen when live, otherwise 0. O[b,h,i,d]=bf16(OutNum*inv) when live, otherwise bf16(0), using RNE. When LSE is present, LSE[b,h,i]=Mtail[row]+logf(OutDen) if live and Mtail is finite, otherwise -inf. Initial O/LSE contents are never read. Workspace is overwritten with the intermediates; unrelated scratch bytes have no mathematical result. The host returns None.

## t_da59210ed81f3b84 | math.math | 395 chars | 2 declarations

fused_d128_cp:68 design.math_reshape.rewrite[1].math; root:278 design.math_reshape.rewrite[1].math

For z=0,1 and v=64*z,...,64*z+63, start A_z[-1]=0 and for k=0,...,3 form A_z[k]=mma32(Qb[:,32*k:32*k+32], Kb[64*z:64*z+64,32*k:32*k+32]^T,A_z[k-1]). Score[u,v]=A_z[3][u,v-64*z] if lo(u)<=g*KeyGroup+v<=hi(u), and -inf otherwise. Let scale=fp32(D^(-1/2)). RawMax[u]=fmax over both halves of Score[u,:], starting at -inf; the maximum uses the source fmaxf NaN behavior. GroupMax[u]=RawMax[u]*scale.

## t_dcac5db6146ba0a3 | math.math | 1063 chars | 6 declarations

tcg_d128_flat_cp_xpose_narrow:144 design.math_reshape.rewrite[4].math; tcg_d128_flat_cp_xpose_wide:127 design.math_reshape.rewrite[4].math; tcg_d128_flat_tma_xpose_wide:143 design.math_reshape.rewrite[4].math; tcg_d128_segmented_cp_xpose_narrow:149 design.math_reshape.rewrite[5].math; tcg_d128_segmented_tma_xpose_narrow:165 design.math_reshape.rewrite[5].math; tcg_d128_segmented_tma_xpose_wide:149 design.math_reshape.rewrite[5].math

Initialize NumState(cLo-1)[u,d]=0 for u<Br. For every nonempty-split group, VAcc(-1)=0 by overwrite and VAcc(k)=mma32(Prob8[:,32*k:32*k+32], Vb[32*k:32*k+32,:],VAcc(k-1)), k=0,...,3. Let GroupPV=VAcc(3). Rescale NumState(g-1) by alpha unless alpha=1, then separately add GroupPV in FP32 to obtain NumState(g). The native PV accumulator is fresh each group, not carried through groups. For each valid row of a nonempty split, weight=expf(MaxState(cHi)[u]-MTail[r]) when both maxima are finite, otherwise 0. Store PartA[s,r,d]=NumState(cHi)[u,d]*weight and PartB[s,r]=DenState(cHi)[u]*weight in FP32, without BF16 narrowing. A nonempty split's multiplication by zero does not suppress NaN/Inf. An empty split has NumState=DenState=weight=0; its evaluated values for storage are zero and PartB[s,r]=0. Its actual materialized PartA slice is not guaranteed zero: an unsynchronized source overwrite can corrupt overlapping entries. PartA denotes the actual stored FP32 values consumed by combine, including this empty-split case. Discard local states after the stores.

## t_e3d268609eea37a2 | math.math | 641 chars | 4 declarations

tcg_d128_fscan_cp_direct_narrow:89 design.math_reshape.rewrite[2].math; tcg_d128_fscan_cp_direct_wide:89 design.math_reshape.rewrite[2].math; tcg_d128_fscan_tma_direct_narrow:89 design.math_reshape.rewrite[2].math; tcg_d128_fscan_tma_direct_wide:89 design.math_reshape.rewrite[2].math

For numerator split s let cLo=L(Ns,s), cHi=Hlast(Ns,s), end=max(cLo-1,cHi), and Tw(s)=min(cLo,T). Define advance(m,x)=x if x-m>fp32(2.7725887222397812), otherwise m; the comparison is false for NaN. For all u<Br, PrefixState(-1)[u]=-inf. For t=0,...,Tw(s)-1, using domain's r for each row slot, PrefixState(t)[u]=advance(PrefixState(t-1)[u],GroupMax[t,r]). For u<C initialize MaxState(cLo-1)[u]=PrefixState(Tw(s)-1)[u]. For g=cLo,...,cHi, MaxState(g)[u]=advance(MaxState(g-1)[u],GroupMax[g,r]). Empty splits still evaluate the bounded prefix and retain its final state. No per-group global snapshots are stored and no prefix is reassociated.

## t_e6f6d31cf29d6d66 | math.math | 350 chars | 4 declarations

tcg_d128_segmented_cp_direct_narrow:90 design.math_reshape.rewrite[2].math; tcg_d128_segmented_cp_direct_wide:89 design.math_reshape.rewrite[2].math; tcg_d128_segmented_tma_direct_narrow:90 design.math_reshape.rewrite[2].math; tcg_d128_segmented_tma_direct_wide:90 design.math_reshape.rewrite[2].math

Let Nseg=ceil_div(T,Seg). For row r and segment z=0,...,Nseg-1, initialize SegAcc(-1)=-inf. For j=0,...,Seg-1 set sx[j]=GroupMax[Seg*z+j,r] when Seg*z+j<T, otherwise -inf, and SegAcc(j)=fmaxf(SegAcc(j-1),sx[j]). SegMax[z,r]=SegAcc(Seg-1), using the scores step's fmaxf NaN behavior. All segments are materialized; replay does not consume SegMax[0,r].

## t_e91275c1ae1de89a | math.math | 678 chars | 4 declarations

tc_split_flat_narrow:79 design.math_reshape.rewrite[1].math; tc_split_flat_wide:79 design.math_reshape.rewrite[1].math; tc_split_segmented_narrow:80 design.math_reshape.rewrite[1].math; tc_split_segmented_wide:80 design.math_reshape.rewrite[1].math

Initialize A(-1)[u,v]=0. For k=0,...,D/Block-1: T_k=mm32(Qraw[:,Block*k:Block*(k+1)], Kraw[:,Block*k:Block*(k+1)]^T). A(k)[u,v]=fma32(T_k[u,v]*QueryScale[u,k], KeyScale[v,k],A(k-1)[u,v]). The multiplication before fma32 rounds separately to FP32. Let scale=fp32(D^(-1/2)). Set Score[u,v]=A(D/Block-1)[u,v]*scale, then replace entries outside lo(u)<=g*KeyGroup+v<=hi(u) by -inf. For r=0,...,3, reduce the pairs Score[u,8*n+2*r:8*n+2*r+2] for n=0,...,15 with fmax starting at -inf, obtaining PairMax[r]. GroupMax[u]=fmax(fmax(PairMax[0],PairMax[1]),fmax(PairMax[2],PairMax[3])). fmax selects the non-NaN argument when only one argument is NaN, following the source fmaxf behavior.

## t_e92d8ffde4a73150 | math.math | 648 chars | 1 declarations

tc_fused:68 design.math_reshape.rewrite[1].math

Initialize A(-1)[u,v]=0. For k=0,...,D/Block-1: T_k=mm32(Qraw[:,Block*k:Block*(k+1)], Kraw[:,Block*k:Block*(k+1)]^T). A(k)[u,v]=fma32(T_k[u,v]*QueryScale[u,k], KeyScale[v,k],A(k-1)[u,v]). The multiplication before fma32 rounds separately to FP32. Let scale=fp32(D^(-1/2)). Set Score[u,v]=A(D/Block-1)[u,v]*scale, then replace entries outside lo(u)<=g*KeyGroup+v<=hi(u) by -inf. For r=0,...,3, reduce the pairs Score[u,8*n+2*r:8*n+2*r+2] for n=0,...,15 with fmax starting at -inf, obtaining R[r]. GroupMax[u]=fmax(fmax(R[0],R[1]),fmax(R[2],R[3])). fmax selects the non-NaN argument when only one argument is NaN, following the source fmaxf behavior.

## t_ecb7308925eda3b1 | math.math | 626 chars | 1 declarations

tcg_d256_segmented_cp_wide:68 design.math_reshape.rewrite[1].math

For each evaluated group initialize A[-1]=0; for k=0,...,7, A[k]=mma32(Qb[:,32*k:32*k+32],Kb[:,32*k:32*k+32]^T,A[k-1]). Score[u,v]=A[7][u,v] when lo(u)<=g*KeyGroup+v<=hi(u), otherwise -inf. Let scale=fp32(D^(-1/2)). For each valid row, HalfMax[z]=fmax over Score[u,64*z:64*z+64], starting at -inf, and GroupMax[g,row]=fmax(HalfMax[0],HalfMax[1])*scale. Maxima retain the source fmaxf/native max NaN behavior. GroupMax[g,row] is -inf outside gLo..gHi. The group-max pass covers all T groups by these products and explicit -inf tails. The split pass repeats the same products and masks, retaining Score but computing no maximum.

## t_ed18c73c23822e10 | math.math | 864 chars | 4 declarations

tcg_d128_fscan_cp_direct_narrow:141 design.math_reshape.rewrite[4].math; tcg_d128_fscan_cp_direct_wide:141 design.math_reshape.rewrite[4].math; tcg_d128_fscan_tma_direct_narrow:141 design.math_reshape.rewrite[4].math; tcg_d128_fscan_tma_direct_wide:141 design.math_reshape.rewrite[4].math

Initialize NumState(cLo-1)[u,d]=0 for u<C. For every nonempty-split group, VAcc(-1)=0 by overwrite and VAcc(k)=mma32(Prob8[:,32*k:32*k+32], Vb[32*k:32*k+32,:],VAcc(k-1)), k=0,...,3. Let GroupPV=VAcc(3). Rescale NumState(g-1) by alpha unless alpha=1, then separately add GroupPV in FP32 to obtain NumState(g). The native PV accumulator is fresh each group, not carried through groups. For each valid row of a nonempty split store PartA[s,r,d]=NumState(cHi)[u,d]*1 and PartB[s,r]=DenState(cHi)[u]*1 in FP32, without BF16 narrowing. Empty splits store zero PartA and PartB. Every split stores PartM[s,r]=MaxState(end)[u]. Only s=Ns-1 also writes MTail[r]=MaxState(end)[u] from that same local state, including for an empty last split. No split reads MTail. Discard local states after stores. A later zero combine weight does not suppress a nonempty partial's NaN/Inf.

## t_ee39543c08706f83 | math.math | 1407 chars | 3 declarations

tcg_d128_fscan_cp_xpose_narrow:109 design.math_reshape.rewrite[3].math; tcg_d128_fscan_cp_xpose_wide:109 design.math_reshape.rewrite[3].math; tcg_d128_fscan_tma_xpose_wide:109 design.math_reshape.rewrite[3].math

Initialize DenState(cLo-1)[u]=0. For each nonempty split group g=cLo,...,cHi, use the scan step's preceding and current MaxState. safe[u]=MaxState(g)[u] if finite, otherwise 0; alpha[u]=expf(MaxState(g-1)[u]-safe[u]) if the old state is finite, otherwise 0. Compute x=Score[u,v]*scale-safe[u], with separate FP32 multiply and subtract. Define the actual E16 approximation by t=sat01(fma32(x,fp32(0.0057249800302088261),0.5)), w=fma_rm32(t,252,12582917), e=fma32(x,fp32(1.4426950216293335),12583043-w), d=bitcast_f32((bitcast_u32(w)<<23) mod 2^32), E16(x)=d*ex2_ftz(e). fma_rm32 rounds toward negative infinity; sat01 is the source saturating FP32 conversion to [0,1], including NaN-to-zero; ex2_ftz is its approximate base-two exponential with flush-to-zero. There is no low log2(e) correction. Prob[u,v]=E16(x); Prob8[u,v]=satfinite_e4m3_rne(Prob[u,v]), the hardware saturating RNE conversion with unit scale. For nonnegative finite values it rounds min(Prob,ProbabilityMax) to E4M3FN. For each chunk c=0,...,3, pair p=0,1 and partition z=0,...,3, F[c,z,p] is the left-to-right FP32 sum from zero of Prob[u,32*c+8*z+2*j+p], j=0,...,3. Sum (F[c,0,p]+F[c,1,p])+(F[c,2,p]+F[c,3,p]) over increasing c from zero to E[p]. DenState(g)[u]=DenState(g-1)[u]*alpha[u]+(E[0]+E[1]), with the source's scalar FP32 compiler path; no additional contraction restriction is imposed. The denominator never uses rounded Prob8.

## t_f0a0e307c3341c79 | math.math | 1202 chars | 1 declarations

tc_fused:92 design.math_reshape.rewrite[2].math

Initialize MaxState(cLo-1)[u]=-inf and DenPart(cLo-1)[u,r]=0 for r=0,...,3. Let beta=16 when D=128, otherwise 1. MaxState(g)[u]=GroupMax[u] if GroupMax[u]-MaxState(g-1)[u]> fp32(2.7725887222397812), otherwise MaxState(g-1)[u]; NaN comparisons are false. safe[u]=MaxState(g)[u] if finite, otherwise 0. alpha[u]=expf(MaxState(g-1)[u]-safe[u]) if the old state is finite, otherwise 0. Prob[u,v]=expf(Score[u,v]-safe[u])*beta, with separate FP32 subtraction, exponential and bias multiplication. Prob8[u,v]=satfinite_e4m3_rne(Prob[u,v]), the hardware saturating RNE conversion; finite nonnegative values saturate at ProbabilityMax. For each r=0,...,3 and p=0,1, compute E[u,r,p]=sum_{n=0}^{15} Prob[u,8*n+2*r+p] using increasing n, left-to-right FP32 additions starting from zero. DenPart(g)[u,r]=DenPart(g-1)[u,r]*alpha[u]+(E[u,r,0]+E[u,r,1]), following the source scalar FP32 update under compiler defaults; no additional contraction restriction is imposed on this update. Multiplication of carried state by exactly one may be skipped. The four partials are not merged between groups. Only unrounded Prob supplies DenPart. Fully masked entries have Prob=Prob8=0; their value products are still evaluated.

## t_f27b17d3c2ca08e7 | math.math | 1741 chars | 2 declarations

fused_d128_cp:89 design.math_reshape.rewrite[2].math; root:299 design.math_reshape.rewrite[2].math

Initialize MaxState(cLo-1)[u]=-inf and DenState(cLo-1)[u]=0. MaxState(g)[u]=GroupMax[u] if GroupMax[u]-MaxState(g-1)[u]> fp32(2.7725887222397812), otherwise MaxState(g-1)[u]. The comparison is false for NaN. safe[u]=MaxState(g)[u] if finite, otherwise 0. alpha[u]=expf(MaxState(g-1)[u]-safe[u]) if the old state is finite, otherwise 0. For each Score entry, x=(Score[u,v]*scale)-safe[u], with a separate FP32 multiply and subtract. Define the actual E16 approximation by: t=sat01(fma32(x,fp32(0.0057249800302088261),0.5)), w=fma_rm32(t,252,12582917), e=fma32(x,fp32(1.4426950216293335),12583043-w), d=bitcast_f32((bitcast_u32(w)<<23) mod 2^32), E16(x)=d*ex2_ftz(e). Here fma_rm32 rounds the fused result toward negative infinity; sat01 is the source saturating FP32 conversion to [0,1], including its NaN-to-zero behavior; ex2_ftz is the source approximate base-two exponential with flush-to-zero. No low log2(e) correction is added. Prob[u,v]=E16(x); Prob8[u,v]=satfinite_e4m3_rne(Prob[u,v]). The latter is the source hardware saturating RNE conversion, with unit probability scale; for nonnegative finite values it rounds min(Prob,ProbabilityMax) to E4M3FN. For each 32-key chunk c=0,...,3, accumulate four independent pairs: for r=0,...,3 and p=0,1, sum Prob[u,32*c+8*r+2*s+p] over s=0,...,3 by left-to-right FP32 adds from zero, obtaining F[c,r,p]. Combine F[c,:,p] as (F[c,0,p]+F[c,1,p])+(F[c,2,p]+F[c,3,p]); add these four chunk results in increasing c from zero to E[p]. DenState(g)[u]=DenState(g-1)[u]*alpha[u]+(E[0]+E[1]), using the source's scalar FP32 update under its compiler defaults; no additional contraction restriction is imposed. The denominator never uses Prob8. MaxState and DenState are discarded after the output step.

## t_f2fc152b96c76a94 | math.math | 1788 chars | 4 declarations

tc_split_flat_narrow:33 design.math_reshape.rewrite[0].math; tc_split_flat_wide:33 design.math_reshape.rewrite[0].math; tc_split_segmented_narrow:34 design.math_reshape.rewrite[0].math; tc_split_segmented_wide:34 design.math_reshape.rewrite[0].math

Let G=H/KH, Qph=Br/G, Aq=ceil_div(Sq,Qph), Ag=ceil_div(Sq,2*Qph), Covered=min(Sq,Ag*Qph), and Rc=B*H*Covered. For each batch b, KV head kh and 0<=a<Aq, let q0=a*Qph. Row u=0,...,Br-1 maps to h=kh*G+floor(u/Qph), i=q0+(u mod Qph). A row is valid iff i<Sq. Its scratch row is rho(u)=(b*H+h)*Sq+(i if valid, otherwise 0). For valid rows, lo(u)=max(0,i+Skv-Sq-window_left) if active, otherwise 0; hi(u)=min(i+Skv-Sq,Skv-1) if is_causal, otherwise Skv-1. For invalid rows set lo(u)=0 and hi(u)=Skv-1. Let imin=q0, imax=min(q0+Qph-1,Sq-1), gLo(a)=floor((active ? max(0,imin+Skv-Sq-window_left) : 0)/KeyGroup), gHi(a)=floor((is_causal ? imax+Skv-Sq : Skv-1)/KeyGroup), Ng(a)=gHi(a)-gLo(a)+1. For n>=1 intervals and 0<=z<n define L(a,z,n)=gLo(a)+z*ceil_div(Ng(a),n), U(a,z,n)=min(L(a,z,n)+ceil_div(Ng(a),n)-1,gHi(a)). The interval is empty when L>U. Maximum computation uses a<Ag,n=Nm; value computation uses a<Aq,n=Ns. They have separate local instances. All Br rows and KeyGroup keys of each visited group are evaluated; wholly excluded groups are skipped. Only valid rows are stored. Within g, v=0,...,KeyGroup-1 denotes key j=g*KeyGroup+v. Qraw[u,d], Kraw[v,d] and Vraw[v,d] are the stored E4M3 payloads at these coordinates. Zero-fill Qraw for invalid queries and Kraw/Vraw for j>=Skv. Use the Contract's packed scale-byte addressing. QueryScale[u,k] and KeyScale[v,k] are 2^(byte-ExponentBias) for k=0,...,D/Block-1, but zero for invalid queries and j>=Skv respectively. QueryScale is computed once per row tile. ValueScale[c,d], c=0,...,KeyGroup/Block-1, is the BF16 scale for this group's key-axis block c. Its byte decodes as 2^(byte-ExponentBias): byte 0 gives 2^-127 and padding byte 255 gives +inf. Allocated padding V-scale bytes are still used; padded V is not zeroed after scaling.

## t_fa0448a682738041 | math.math | 950 chars | 1 declarations

tcg_d128_fscan_tma_xpose_narrow:147 design.math_reshape.rewrite[4].math

Initialize NumState(cLo-1)[u,d]=0 for u<Br. For every nonempty-split group, VAcc(-1)=0 by overwrite and VAcc(k)=mma32(Prob8[:,32*k:32*k+32], Vb[32*k:32*k+32,:],VAcc(k-1)), k=0,...,3. Let GroupPV=VAcc(3). Rescale NumState(g-1) by alpha unless alpha=1, then separately add GroupPV in FP32 to obtain NumState(g). The native PV accumulator is fresh each group, not carried through groups. For each valid row let wgt=1 for a nonempty split, otherwise 0. Store PartA[s,r,d]=NumState(end)[u,d]*wgt and PartB[s,r]=DenState(end)[u]*wgt in FP32 without BF16 narrowing. An empty split has NumState=DenState=0; its evaluated storage values are zero and PartB[s,r]=0. Its actual PartA slice is not guaranteed zero: an unsynchronized source overwrite can corrupt overlapping entries. PartA denotes the actual stored FP32 values consumed by combine. Discard NumState and DenState after partial stores; retain MaxState through PartM and last-split MTail publication.

## t_fbe11c41478f6f6e | math.math | 611 chars | 7 declarations

tcg_d256_flat_cp_narrow:68 design.math_reshape.rewrite[1].math; tcg_d256_flat_cp_wide:69 design.math_reshape.rewrite[1].math; tcg_d256_flat_tma_narrow:68 design.math_reshape.rewrite[1].math; tcg_d256_flat_tma_wide:68 design.math_reshape.rewrite[1].math; tcg_d256_segmented_cp_narrow:53 design.math_reshape.rewrite[1].math; tcg_d256_segmented_tma_narrow:68 design.math_reshape.rewrite[1].math; tcg_d256_segmented_tma_wide:53 design.math_reshape.rewrite[1].math

For each evaluated group, A[-1]=0 and for k=0,...,7, A[k]=mma32(Qb[:,32*k:32*k+32],Kb[:,32*k:32*k+32]ᵀ,A[k-1]). Score[u,v]=A[7][u,v] when lo(u)<=g*KeyGroup+v<=hi(u), otherwise -inf. scale=fp32(D^(-1/2)). For each valid row and z=0,1, HalfMax[u,z]=fmax over Score[u,64*z:64*z+64], starting at -inf. GroupMax[g,row]=fmax(HalfMax[u,0],HalfMax[u,1])*scale, retaining the source fmaxf/native max NaN behavior. GroupMax[g,row]=-inf outside gLo..gHi. The group-max pass forms all GCount entries per valid row. The split pass repeats these products and masks, retaining Score for probabilities but computing no maximum.

## t_fe193f35d5c846c2 | math.math | 307 chars | 4 declarations

tcg_d256_flat_cp_narrow:92 design.math_reshape.rewrite[2].math; tcg_d256_flat_cp_wide:93 design.math_reshape.rewrite[2].math; tcg_d256_flat_tma_narrow:92 design.math_reshape.rewrite[2].math; tcg_d256_flat_tma_wide:92 design.math_reshape.rewrite[2].math

For each row, Mscan(-1,row)=-inf. For t=0,...,GCount-1, Mscan[t,row]=GroupMax[t,row] if GroupMax[t,row]-Mscan(t-1,row)>fp32(2.7725887222397812), otherwise Mscan(t-1,row); the comparison is false for NaN. Mtail[row]=Mscan[GCount-1,row]. Keep Mscan for replay and Mtail for partial weighting and optional LSE.

## t_fecc7955bed3747f | math.math | 1377 chars | 2 declarations

tc_split_flat_narrow:103 design.math_reshape.rewrite[2].math; tc_split_flat_wide:103 design.math_reshape.rewrite[2].math

For a<Ag, maximum interval z computes scores on L(a,z,Nm)..U(a,z,Nm) and writes BmCovered[g,b,h,i]=GroupMax[u] for valid rows. Interval z=0 also writes -inf for 0<=g<gLo(a); z=Nm-1 writes -inf for gHi(a)<g<T, even when its computational interval is empty. BmCovered is defined only for i<Covered. For Covered<=i<Sq define PriorMax[g,b,h,i-Covered] as the native FP32 interpretation of the four incoming workspace bytes at 4*(g*R+(b*H+h)*Sq+i). These bytes are not written by the maximum computation and have no specified initial numerical value. Define OTemporary[u,d]=bf16(0), LseTemporary[u]=-inf. After every maximum interval, including empty ones, store these into valid rows of o_tensor and, when present, lse_tensor. All Nm intervals write the same locations. The temporary values have no later consumer; the final combine overwrites them. For each row r=(b*H+h)*Sq+i and g=0,...,T-1, let x(g,r)=BmCovered[g,b,h,i] if i<Covered, otherwise PriorMax[g,b,h,i-Covered]. Initialize ScanCarry(-1)[r]=-inf. Set ScanCarry(g)[r]=x(g,r) if x(g,r)-ScanCarry(g-1)[r]>fp32(2.7725887222397812), otherwise ScanCarry(g-1)[r]. NaN comparisons are false. Save ScanState[g,r]=ScanCarry(g)[r] and MaxTail[r]=ScanCarry(T-1)[r]. All T groups are read. BmCovered and PriorMax die after the scan. The uncovered rows can make the result depend on incoming scratch; no missing maxima are supplied.

## t_0176b535d9f08e66 | math.meaning | 34 chars | 12 declarations

tc_split_flat_narrow:211 design.math_reshape.rewrite[4].symbol.PartialB.meaning; tc_split_flat_wide:211 design.math_reshape.rewrite[4].symbol.PartialB.meaning; tc_split_segmented_narrow:246 design.math_reshape.rewrite[6].symbol.PartialB.meaning; tc_split_segmented_wide:246 design.math_reshape.rewrite[6].symbol.PartialB.meaning; tcg_d256_flat_cp_narrow:155 design.math_reshape.rewrite[4].symbol.PartB.meaning; tcg_d256_flat_cp_wide:156 design.math_reshape.rewrite[4].symbol.PartB.meaning; tcg_d256_flat_tma_narrow:155 design.math_reshape.rewrite[4].symbol.PartB.meaning; tcg_d256_flat_tma_wide:155 design.math_reshape.rewrite[4].symbol.PartB.meaning; tcg_d256_segmented_cp_narrow:162 design.math_reshape.rewrite[5].symbol.PartB.meaning; tcg_d256_segmented_cp_wide:190 design.math_reshape.rewrite[5].symbol.PartB.meaning; tcg_d256_segmented_tma_narrow:177 design.math_reshape.rewrite[5].symbol.PartB.meaning; tcg_d256_segmented_tma_wide:162 design.math_reshape.rewrite[5].symbol.PartB.meaning

Weighted FP32 denominator partials

## t_03eccc9cd9f013da | math.meaning | 28 chars | 8 declarations

tcg_d256_flat_cp_narrow:100 design.math_reshape.rewrite[2].symbol.Mscan.meaning; tcg_d256_flat_cp_wide:101 design.math_reshape.rewrite[2].symbol.Mscan.meaning; tcg_d256_flat_tma_narrow:100 design.math_reshape.rewrite[2].symbol.Mscan.meaning; tcg_d256_flat_tma_wide:100 design.math_reshape.rewrite[2].symbol.Mscan.meaning; tcg_d256_segmented_cp_narrow:107 design.math_reshape.rewrite[3].symbol.Mscan.meaning; tcg_d256_segmented_cp_wide:126 design.math_reshape.rewrite[3].symbol.Mscan.meaning; tcg_d256_segmented_tma_narrow:122 design.math_reshape.rewrite[3].symbol.Mscan.meaning; tcg_d256_segmented_tma_wide:107 design.math_reshape.rewrite[3].symbol.Mscan.meaning

Saved delayed-maximum states

## t_042f1956ca0f46dd | math.meaning | 24 chars | 5 declarations

tc_fused:64 design.math_reshape.rewrite[0].symbol.KeyScale.meaning; tc_split_flat_narrow:72 design.math_reshape.rewrite[0].symbol.KeyScale.meaning; tc_split_flat_wide:72 design.math_reshape.rewrite[0].symbol.KeyScale.meaning; tc_split_segmented_narrow:73 design.math_reshape.rewrite[0].symbol.KeyScale.meaning; tc_split_segmented_wide:73 design.math_reshape.rewrite[0].symbol.KeyScale.meaning

Decoded key block scales

## t_050eac37165bc8d7 | math.meaning | 33 chars | 1 declarations

tc_split_segmented_wide:287 design.math_reshape.rewrite[7].symbol.GroupFinalA.meaning

Retained group numerator partials

## t_060af5932b2849c8 | math.meaning | 53 chars | 3 declarations

tcg_d128_fscan_cp_xpose_narrow:105 design.math_reshape.rewrite[2].symbol.PartM.meaning; tcg_d128_fscan_cp_xpose_wide:105 design.math_reshape.rewrite[2].symbol.PartM.meaning; tcg_d128_fscan_tma_xpose_wide:105 design.math_reshape.rewrite[2].symbol.PartM.meaning

Saved end state for each split including empty splits

## t_088ccdd466b5e039 | math.meaning | 44 chars | 5 declarations

tc_fused:62 design.math_reshape.rewrite[0].symbol.Vraw.meaning; tc_split_flat_narrow:68 design.math_reshape.rewrite[0].symbol.Vraw.meaning; tc_split_flat_wide:68 design.math_reshape.rewrite[0].symbol.Vraw.meaning; tc_split_segmented_narrow:69 design.math_reshape.rewrite[0].symbol.Vraw.meaning; tc_split_segmented_wide:69 design.math_reshape.rewrite[0].symbol.Vraw.meaning

Value payload view for the current key group

## t_08930e3834b1fe2c | math.meaning | 39 chars | 1 declarations

tc_split_flat_wide:243 design.math_reshape.rewrite[5].symbol.GroupAccum.meaning

Numerator sum within one split interval

## t_098566483d861329 | math.meaning | 33 chars | 3 declarations

tc_split_flat_narrow:233 design.math_reshape.rewrite[5].symbol.SumA.meaning; tc_split_segmented_narrow:268 design.math_reshape.rewrite[7].symbol.SumA.meaning; tc_split_segmented_wide:293 design.math_reshape.rewrite[7].symbol.SumA.meaning

Numerator sum over split partials

## t_09fbc8db55edf5a4 | math.meaning | 64 chars | 1 declarations

tcg_d256_segmented_cp_wide:82 design.math_reshape.rewrite[1].symbol.Score.meaning

Masked unscaled scores retained for split probability evaluation

## t_0a611ae9b4def925 | math.meaning | 57 chars | 4 declarations

tcg_d128_fscan_cp_direct_narrow:164 design.math_reshape.rewrite[4].symbol.PartM.meaning; tcg_d128_fscan_cp_direct_wide:164 design.math_reshape.rewrite[4].symbol.PartM.meaning; tcg_d128_fscan_tma_direct_narrow:164 design.math_reshape.rewrite[4].symbol.PartM.meaning; tcg_d128_fscan_tma_direct_wide:164 design.math_reshape.rewrite[4].symbol.PartM.meaning

End delayed maximum of every split including empty splits

## t_0ac9cf58d9b193e8 | math.meaning | 52 chars | 1 declarations

tcg_d128_fscan_cp_xpose_wide:206 design.math_reshape.rewrite[5].symbol.NumGroup.meaning

Weighted numerator subtotal for each merge partition

## t_0afac2581f548ff9 | math.meaning | 62 chars | 3 declarations

tcg_d128_fscan_cp_xpose_narrow:194 design.math_reshape.rewrite[5].symbol.Weight.meaning; tcg_d128_fscan_cp_xpose_wide:205 design.math_reshape.rewrite[5].symbol.Weight.meaning; tcg_d128_fscan_tma_xpose_wide:206 design.math_reshape.rewrite[5].symbol.Weight.meaning

Split weights reused across numerator channels and denominator

## t_12228b8ac75e5155 | math.meaning | 53 chars | 1 declarations

tc_split_segmented_wide:289 design.math_reshape.rewrite[7].symbol.F.meaning

Denominator tree state including discarded components

## t_14bf5f8f8602945f | math.meaning | 46 chars | 1 declarations

tcg_d128_segmented_cp_direct_wide:213 design.math_reshape.rewrite[6].symbol.MergeGroups.meaning

Split intervals reduced before the final merge

## t_156abc804ab4e4b1 | math.meaning | 47 chars | 5 declarations

tcg_d128_segmented_cp_xpose_narrow:82 design.math_reshape.rewrite[2].symbol.SegBatch.meaning; tcg_d128_segmented_cp_xpose_wide:100 design.math_reshape.rewrite[2].symbol.SegBatch.meaning; tcg_d128_segmented_tma_xpose_narrow:98 design.math_reshape.rewrite[2].symbol.SegBatch.meaning; tcg_d128_segmented_tma_xpose_wide:82 design.math_reshape.rewrite[2].symbol.SegBatch.meaning; tcg_d256_segmented_cp_wide:101 design.math_reshape.rewrite[2].symbol.SegChunk.meaning

Segments tested against one incoming scan state

## t_1609b3106409898c | math.meaning | 50 chars | 1 declarations

tcg_d128_fscan_cp_xpose_wide:204 design.math_reshape.rewrite[5].symbol.MergeGroups.meaning

Number of contiguous split partitions in the merge

## t_1648bb72417de5a2 | math.meaning | 37 chars | 6 declarations

fused_d128_cp:120 design.math_reshape.rewrite[2].symbol.MaxState.meaning; fused_d128_group:118 design.math_reshape.rewrite[2].symbol.MaxState.meaning; fused_d128_key:121 design.math_reshape.rewrite[2].symbol.MaxState.meaning; fused_d256_cp:115 design.math_reshape.rewrite[2].symbol.MaxState.meaning; fused_d256_tma:115 design.math_reshape.rewrite[2].symbol.MaxState.meaning; root:330 design.math_reshape.rewrite[2].symbol.MaxState.meaning

Delayed maximum for the current group

## t_17a491b6bc99c40c | math.meaning | 30 chars | 7 declarations

tcg_d256_flat_cp_narrow:80 design.math_reshape.rewrite[1].symbol.HalfMax.meaning; tcg_d256_flat_cp_wide:81 design.math_reshape.rewrite[1].symbol.HalfMax.meaning; tcg_d256_flat_tma_narrow:80 design.math_reshape.rewrite[1].symbol.HalfMax.meaning; tcg_d256_flat_tma_wide:80 design.math_reshape.rewrite[1].symbol.HalfMax.meaning; tcg_d256_segmented_cp_narrow:65 design.math_reshape.rewrite[1].symbol.HalfMax.meaning; tcg_d256_segmented_tma_narrow:80 design.math_reshape.rewrite[1].symbol.HalfMax.meaning; tcg_d256_segmented_tma_wide:65 design.math_reshape.rewrite[1].symbol.HalfMax.meaning

Maximum of each row's key half

## t_18fef974cd90d02c | math.meaning | 51 chars | 6 declarations

tc_split_segmented_narrow:169 design.math_reshape.rewrite[4].symbol.Sc.meaning; tc_split_segmented_wide:169 design.math_reshape.rewrite[4].symbol.Sc.meaning; tcg_d128_segmented_cp_direct_narrow:119 design.math_reshape.rewrite[3].symbol.SegBatch.meaning; tcg_d128_segmented_cp_direct_wide:118 design.math_reshape.rewrite[3].symbol.SegBatch.meaning; tcg_d128_segmented_tma_direct_narrow:119 design.math_reshape.rewrite[3].symbol.SegBatch.meaning; tcg_d128_segmented_tma_direct_wide:119 design.math_reshape.rewrite[3].symbol.SegBatch.meaning

Segment summaries tested against one incoming state

## t_19e5fead96348887 | math.meaning | 31 chars | 3 declarations

tcg_d128_fscan_cp_xpose_narrow:104 design.math_reshape.rewrite[2].symbol.MaxState.meaning; tcg_d128_fscan_cp_xpose_wide:104 design.math_reshape.rewrite[2].symbol.MaxState.meaning; tcg_d128_fscan_tma_xpose_wide:104 design.math_reshape.rewrite[2].symbol.MaxState.meaning

Locally scanned delayed maximum

## t_1dcb310249d09903 | math.meaning | 42 chars | 1 declarations

tcg_d256_segmented_cp_wide:223 design.math_reshape.rewrite[6].symbol.GroupDen.meaning

Denominator tree results for each interval

## t_212b2ae27237800c | math.meaning | 39 chars | 4 declarations

tcg_d128_fscan_cp_xpose_narrow:133 design.math_reshape.rewrite[3].symbol.DenState.meaning; tcg_d128_fscan_cp_xpose_wide:133 design.math_reshape.rewrite[3].symbol.DenState.meaning; tcg_d128_fscan_tma_xpose_wide:133 design.math_reshape.rewrite[3].symbol.DenState.meaning; tcg_d128_segmented_cp_xpose_wide:156 design.math_reshape.rewrite[4].symbol.DenState.meaning

Local unrounded-probability denominator

## t_217e122337d633f3 | math.meaning | 38 chars | 8 declarations

tcg_d256_flat_cp_narrow:129 design.math_reshape.rewrite[3].symbol.Prob8.meaning; tcg_d256_flat_cp_wide:130 design.math_reshape.rewrite[3].symbol.Prob8.meaning; tcg_d256_flat_tma_narrow:129 design.math_reshape.rewrite[3].symbol.Prob8.meaning; tcg_d256_flat_tma_wide:129 design.math_reshape.rewrite[3].symbol.Prob8.meaning; tcg_d256_segmented_cp_narrow:136 design.math_reshape.rewrite[4].symbol.Prob8.meaning; tcg_d256_segmented_cp_wide:159 design.math_reshape.rewrite[4].symbol.Prob8.meaning; tcg_d256_segmented_tma_narrow:151 design.math_reshape.rewrite[4].symbol.Prob8.meaning; tcg_d256_segmented_tma_wide:136 design.math_reshape.rewrite[4].symbol.Prob8.meaning

Unit-scale rounded probability operand

## t_22503008580b6955 | math.meaning | 44 chars | 17 declarations

tcg_d128_flat_cp_direct_narrow:81 design.math_reshape.rewrite[1].symbol.GroupMax.meaning; tcg_d128_flat_cp_direct_wide:81 design.math_reshape.rewrite[1].symbol.GroupMax.meaning; tcg_d128_flat_cp_xpose_narrow:83 design.math_reshape.rewrite[1].symbol.GroupMax.meaning; tcg_d128_flat_cp_xpose_wide:66 design.math_reshape.rewrite[1].symbol.GroupMax.meaning; tcg_d128_flat_tma_direct_narrow:82 design.math_reshape.rewrite[1].symbol.GroupMax.meaning; tcg_d128_flat_tma_direct_wide:82 design.math_reshape.rewrite[1].symbol.GroupMax.meaning; tcg_d128_flat_tma_xpose_narrow:83 design.math_reshape.rewrite[1].symbol.GroupMax.meaning; tcg_d128_flat_tma_xpose_wide:82 design.math_reshape.rewrite[1].symbol.GroupMax.meaning; tcg_d128_fscan_tma_xpose_narrow:82 design.math_reshape.rewrite[1].symbol.GroupMax.meaning; tcg_d128_segmented_cp_direct_narrow:82 design.math_reshape.rewrite[1].symbol.GroupMax.meaning; tcg_d128_segmented_cp_direct_wide:81 design.math_reshape.rewrite[1].symbol.GroupMax.meaning; tcg_d128_segmented_cp_xpose_narrow:66 design.math_reshape.rewrite[1].symbol.GroupMax.meaning; tcg_d128_segmented_cp_xpose_wide:84 design.math_reshape.rewrite[1].symbol.GroupMax.meaning; tcg_d128_segmented_tma_direct_narrow:82 design.math_reshape.rewrite[1].symbol.GroupMax.meaning; tcg_d128_segmented_tma_direct_wide:82 design.math_reshape.rewrite[1].symbol.GroupMax.meaning; tcg_d128_segmented_tma_xpose_narrow:82 design.math_reshape.rewrite[1].symbol.GroupMax.meaning; tcg_d128_segmented_tma_xpose_wide:66 design.math_reshape.rewrite[1].symbol.GroupMax.meaning

Complete group-major maxima consumed by scan

## t_22f9005b2436f100 | math.meaning | 39 chars | 1 declarations

tcg_d256_segmented_cp_wide:187 design.math_reshape.rewrite[5].symbol.DenFinal.meaning

Sum of the two split-local denominators

## t_24e977bbfeb4302a | math.meaning | 18 chars | 4 declarations

tcg_d128_segmented_cp_xpose_narrow:81 design.math_reshape.rewrite[2].symbol.SegLen.meaning; tcg_d128_segmented_cp_xpose_wide:99 design.math_reshape.rewrite[2].symbol.SegLen.meaning; tcg_d128_segmented_tma_xpose_narrow:97 design.math_reshape.rewrite[2].symbol.SegLen.meaning; tcg_d128_segmented_tma_xpose_wide:81 design.math_reshape.rewrite[2].symbol.SegLen.meaning

Groups per segment

## t_2746f977c4133033 | math.meaning | 58 chars | 4 declarations

tc_split_flat_narrow:139 design.math_reshape.rewrite[2].symbol.LseTemporary.meaning; tc_split_flat_wide:139 design.math_reshape.rewrite[2].symbol.LseTemporary.meaning; tc_split_segmented_narrow:130 design.math_reshape.rewrite[2].symbol.LseTemporary.meaning; tc_split_segmented_wide:130 design.math_reshape.rewrite[2].symbol.LseTemporary.meaning

Temporary negative infinities stored in present caller LSE

## t_27770c5152bae5af | math.meaning | 57 chars | 3 declarations

tcg_d128_fscan_cp_xpose_narrow:82 design.math_reshape.rewrite[1].symbol.GroupMax.meaning; tcg_d128_fscan_cp_xpose_wide:82 design.math_reshape.rewrite[1].symbol.GroupMax.meaning; tcg_d128_fscan_tma_xpose_wide:82 design.math_reshape.rewrite[1].symbol.GroupMax.meaning

Complete group-major maxima consumed by split-local scans

## t_294aa98d4986a660 | math.meaning | 22 chars | 20 declarations

tcg_d128_flat_cp_direct_narrow:183 design.math_reshape.rewrite[5].symbol.TotalA.meaning; tcg_d128_flat_cp_direct_wide:195 design.math_reshape.rewrite[5].symbol.TotalA.meaning; tcg_d128_flat_cp_xpose_narrow:190 design.math_reshape.rewrite[5].symbol.TotalA.meaning; tcg_d128_flat_cp_xpose_wide:173 design.math_reshape.rewrite[5].symbol.TotalA.meaning; tcg_d128_flat_tma_direct_narrow:184 design.math_reshape.rewrite[5].symbol.TotalA.meaning; tcg_d128_flat_tma_direct_wide:195 design.math_reshape.rewrite[5].symbol.TotalA.meaning; tcg_d128_flat_tma_xpose_narrow:187 design.math_reshape.rewrite[5].symbol.TotalA.meaning; tcg_d128_flat_tma_xpose_wide:202 design.math_reshape.rewrite[5].symbol.TotalA.meaning; tcg_d128_fscan_cp_direct_narrow:192 design.math_reshape.rewrite[5].symbol.TotalA.meaning; tcg_d128_fscan_cp_direct_wide:207 design.math_reshape.rewrite[5].symbol.TotalA.meaning; tcg_d128_fscan_tma_direct_narrow:192 design.math_reshape.rewrite[5].symbol.TotalA.meaning; tcg_d128_fscan_tma_direct_wide:209 design.math_reshape.rewrite[5].symbol.TotalA.meaning; tcg_d128_fscan_tma_xpose_narrow:196 design.math_reshape.rewrite[5].symbol.TotalA.meaning; tcg_d128_segmented_cp_direct_narrow:205 design.math_reshape.rewrite[6].symbol.TotalA.meaning; tcg_d128_segmented_cp_direct_wide:218 design.math_reshape.rewrite[6].symbol.TotalA.meaning; tcg_d128_segmented_cp_xpose_narrow:195 design.math_reshape.rewrite[6].symbol.TotalA.meaning; tcg_d128_segmented_tma_direct_narrow:205 design.math_reshape.rewrite[6].symbol.TotalA.meaning; tcg_d128_segmented_tma_direct_wide:220 design.math_reshape.rewrite[6].symbol.TotalA.meaning; tcg_d128_segmented_tma_xpose_narrow:211 design.math_reshape.rewrite[6].symbol.TotalA.meaning; tcg_d128_segmented_tma_xpose_wide:195 design.math_reshape.rewrite[6].symbol.TotalA.meaning

Final merged numerator

## t_2a0eaa6fbb9f1e8f | math.meaning | 48 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:227 design.math_reshape.rewrite[6].symbol.MergeB.meaning

Four denominator tree results per split interval

## t_2b7949151de6a2f3 | math.meaning | 76 chars | 7 declarations

tcg_d128_flat_cp_xpose_narrow:166 design.math_reshape.rewrite[4].symbol.PartA.meaning; tcg_d128_flat_cp_xpose_wide:149 design.math_reshape.rewrite[4].symbol.PartA.meaning; tcg_d128_flat_tma_xpose_wide:165 design.math_reshape.rewrite[4].symbol.PartA.meaning; tcg_d128_fscan_tma_xpose_narrow:168 design.math_reshape.rewrite[4].symbol.PartA.meaning; tcg_d128_segmented_cp_xpose_narrow:171 design.math_reshape.rewrite[5].symbol.PartA.meaning; tcg_d128_segmented_tma_xpose_narrow:187 design.math_reshape.rewrite[5].symbol.PartA.meaning; tcg_d128_segmented_tma_xpose_wide:171 design.math_reshape.rewrite[5].symbol.PartA.meaning

Materialized split numerators including the empty-split overwrite limitation

## t_2bc41d62f2e1f225 | math.meaning | 42 chars | 5 declarations

tc_fused:61 design.math_reshape.rewrite[0].symbol.Kraw.meaning; tc_split_flat_narrow:67 design.math_reshape.rewrite[0].symbol.Kraw.meaning; tc_split_flat_wide:67 design.math_reshape.rewrite[0].symbol.Kraw.meaning; tc_split_segmented_narrow:68 design.math_reshape.rewrite[0].symbol.Kraw.meaning; tc_split_segmented_wide:68 design.math_reshape.rewrite[0].symbol.Kraw.meaning

Key payload view for the current key group

## t_2d1caa3b754e9a00 | math.meaning | 46 chars | 4 declarations

tcg_d128_segmented_cp_direct_narrow:98 design.math_reshape.rewrite[2].symbol.SegMax.meaning; tcg_d128_segmented_cp_direct_wide:97 design.math_reshape.rewrite[2].symbol.SegMax.meaning; tcg_d128_segmented_tma_direct_narrow:98 design.math_reshape.rewrite[2].symbol.SegMax.meaning; tcg_d128_segmented_tma_direct_wide:98 design.math_reshape.rewrite[2].symbol.SegMax.meaning

Segment-major maxima screening replay prefixes

## t_31fcf2978b09f2f8 | math.meaning | 32 chars | 12 declarations

tc_split_flat_narrow:209 design.math_reshape.rewrite[4].symbol.PartialA.meaning; tc_split_flat_wide:209 design.math_reshape.rewrite[4].symbol.PartialA.meaning; tc_split_segmented_narrow:244 design.math_reshape.rewrite[6].symbol.PartialA.meaning; tc_split_segmented_wide:244 design.math_reshape.rewrite[6].symbol.PartialA.meaning; tcg_d256_flat_cp_narrow:154 design.math_reshape.rewrite[4].symbol.PartA.meaning; tcg_d256_flat_cp_wide:155 design.math_reshape.rewrite[4].symbol.PartA.meaning; tcg_d256_flat_tma_narrow:154 design.math_reshape.rewrite[4].symbol.PartA.meaning; tcg_d256_flat_tma_wide:154 design.math_reshape.rewrite[4].symbol.PartA.meaning; tcg_d256_segmented_cp_narrow:161 design.math_reshape.rewrite[5].symbol.PartA.meaning; tcg_d256_segmented_cp_wide:189 design.math_reshape.rewrite[5].symbol.PartA.meaning; tcg_d256_segmented_tma_narrow:176 design.math_reshape.rewrite[5].symbol.PartA.meaning; tcg_d256_segmented_tma_wide:161 design.math_reshape.rewrite[5].symbol.PartA.meaning

Weighted FP32 numerator partials

## t_3376569565045cfa | math.meaning | 61 chars | 1 declarations

tcg_d128_fscan_tma_xpose_wide:208 design.math_reshape.rewrite[5].symbol.BinB.meaning

Denominator sums for 32-entry bins within each merge interval

## t_3566d8d89dc0de5c | math.meaning | 42 chars | 1 declarations

tcg_d128_flat_tma_direct_wide:194 design.math_reshape.rewrite[5].symbol.DenChunk.meaning

Denominator sum for each 32-entry subgroup

## t_39599f3c284c8237 | math.meaning | 34 chars | 7 declarations

tcg_d256_flat_cp_narrow:63 design.math_reshape.rewrite[0].symbol.Nm.meaning; tcg_d256_flat_cp_wide:64 design.math_reshape.rewrite[0].symbol.Nm.meaning; tcg_d256_flat_tma_narrow:63 design.math_reshape.rewrite[0].symbol.Nm.meaning; tcg_d256_flat_tma_wide:63 design.math_reshape.rewrite[0].symbol.Nm.meaning; tcg_d256_segmented_cp_narrow:48 design.math_reshape.rewrite[0].symbol.Nm.meaning; tcg_d256_segmented_tma_narrow:63 design.math_reshape.rewrite[0].symbol.Nm.meaning; tcg_d256_segmented_tma_wide:48 design.math_reshape.rewrite[0].symbol.Nm.meaning

Number of group-maximum partitions

## t_39bd66d0f15f8a43 | math.meaning | 46 chars | 1 declarations

tcg_d128_fscan_tma_xpose_narrow:106 design.math_reshape.rewrite[2].symbol.PartM.meaning

End state of each split including empty splits

## t_3ac9650e02188863 | math.meaning | 22 chars | 1 declarations

tcg_d128_flat_tma_xpose_wide:199 design.math_reshape.rewrite[5].symbol.SP.meaning

Number of merge groups

## t_3e0b0032deefc6e9 | math.meaning | 48 chars | 4 declarations

tcg_d128_fscan_cp_direct_narrow:101 design.math_reshape.rewrite[2].symbol.PrefixState.meaning; tcg_d128_fscan_cp_direct_wide:101 design.math_reshape.rewrite[2].symbol.PrefixState.meaning; tcg_d128_fscan_tma_direct_narrow:101 design.math_reshape.rewrite[2].symbol.PrefixState.meaning; tcg_d128_fscan_tma_direct_wide:101 design.math_reshape.rewrite[2].symbol.PrefixState.meaning

Per-split prefix maximum including padding slots

## t_3e4ec0b4ca6a8229 | math.meaning | 41 chars | 2 declarations

fused_d256_cp:145 design.math_reshape.rewrite[3].symbol.DenFinal.meaning; fused_d256_tma:145 design.math_reshape.rewrite[3].symbol.DenFinal.meaning

Final sum of the two carried denominators

## t_3eec245337a53361 | math.meaning | 42 chars | 1 declarations

tcg_d128_flat_tma_xpose_wide:200 design.math_reshape.rewrite[5].symbol.MergeA.meaning

Numerator sums of disjoint split intervals

## t_3fa53c887ac2221b | math.meaning | 30 chars | 5 declarations

tc_fused:150 design.math_reshape.rewrite[3].symbol.Vbf.meaning; tc_split_flat_narrow:203 design.math_reshape.rewrite[4].symbol.Vbf.meaning; tc_split_flat_wide:203 design.math_reshape.rewrite[4].symbol.Vbf.meaning; tc_split_segmented_narrow:238 design.math_reshape.rewrite[6].symbol.Vbf.meaning; tc_split_segmented_wide:238 design.math_reshape.rewrite[6].symbol.Vbf.meaning

Decoded and scaled BF16 values

## t_404eb45747004ba3 | math.meaning | 43 chars | 4 declarations

tc_split_flat_narrow:135 design.math_reshape.rewrite[2].symbol.MaxTail.meaning; tc_split_flat_wide:135 design.math_reshape.rewrite[2].symbol.MaxTail.meaning; tc_split_segmented_narrow:174 design.math_reshape.rewrite[4].symbol.MaxTail.meaning; tc_split_segmented_wide:174 design.math_reshape.rewrite[4].symbol.MaxTail.meaning

Final maximum for partial weighting and LSE

## t_41231396e7d276f5 | math.meaning | 35 chars | 1 declarations

tcg_d128_flat_cp_direct_wide:193 design.math_reshape.rewrite[5].symbol.GroupA.meaning

Numerator sum per local split group

## t_4176a193c1426a1d | math.meaning | 45 chars | 4 declarations

tcg_d128_segmented_cp_direct_narrow:97 design.math_reshape.rewrite[2].symbol.Seg.meaning; tcg_d128_segmented_cp_direct_wide:96 design.math_reshape.rewrite[2].symbol.Seg.meaning; tcg_d128_segmented_tma_direct_narrow:97 design.math_reshape.rewrite[2].symbol.Seg.meaning; tcg_d128_segmented_tma_direct_wide:97 design.math_reshape.rewrite[2].symbol.Seg.meaning

Number of group maxima summarized per segment

## t_4242e423aa02dd39 | math.meaning | 69 chars | 4 declarations

tcg_d128_fscan_cp_direct_narrow:103 design.math_reshape.rewrite[2].symbol.MaxState.meaning; tcg_d128_fscan_cp_direct_wide:103 design.math_reshape.rewrite[2].symbol.MaxState.meaning; tcg_d128_fscan_tma_direct_narrow:103 design.math_reshape.rewrite[2].symbol.MaxState.meaning; tcg_d128_fscan_tma_direct_wide:103 design.math_reshape.rewrite[2].symbol.MaxState.meaning

Locally advanced delayed maximum including empty-split entrance state

## t_444ca76b11789d42 | math.meaning | 38 chars | 4 declarations

tc_split_flat_narrow:133 design.math_reshape.rewrite[2].symbol.ScanState.meaning; tc_split_flat_wide:133 design.math_reshape.rewrite[2].symbol.ScanState.meaning; tc_split_segmented_narrow:172 design.math_reshape.rewrite[4].symbol.ScanState.meaning; tc_split_segmented_wide:172 design.math_reshape.rewrite[4].symbol.ScanState.meaning

Delayed maximum saved after each group

## t_45468632543c4943 | math.meaning | 26 chars | 5 declarations

tc_fused:65 design.math_reshape.rewrite[0].symbol.ValueScale.meaning; tc_split_flat_narrow:74 design.math_reshape.rewrite[0].symbol.ValueScale.meaning; tc_split_flat_wide:74 design.math_reshape.rewrite[0].symbol.ValueScale.meaning; tc_split_segmented_narrow:75 design.math_reshape.rewrite[0].symbol.ValueScale.meaning; tc_split_segmented_wide:75 design.math_reshape.rewrite[0].symbol.ValueScale.meaning

BF16 key-axis value scales

## t_45578276127b3975 | math.meaning | 56 chars | 5 declarations

tc_fused:151 design.math_reshape.rewrite[3].symbol.Pfrag.meaning; tc_split_flat_narrow:205 design.math_reshape.rewrite[4].symbol.Pfrag.meaning; tc_split_flat_wide:205 design.math_reshape.rewrite[4].symbol.Pfrag.meaning; tc_split_segmented_narrow:240 design.math_reshape.rewrite[6].symbol.Pfrag.meaning; tc_split_segmented_wide:240 design.math_reshape.rewrite[6].symbol.Pfrag.meaning

BF16-widened rounded probabilities for the current chunk

## t_462b5ee054f9ddcd | math.meaning | 32 chars | 2 declarations

fused_d128_cp:121 design.math_reshape.rewrite[2].symbol.DenState.meaning; root:331 design.math_reshape.rewrite[2].symbol.DenState.meaning

Unrounded-probability normalizer

## t_48847f26f6a2072e | math.meaning | 30 chars | 7 declarations

tcg_d256_flat_cp_narrow:62 design.math_reshape.rewrite[0].symbol.Ns.meaning; tcg_d256_flat_cp_wide:63 design.math_reshape.rewrite[0].symbol.Ns.meaning; tcg_d256_flat_tma_narrow:62 design.math_reshape.rewrite[0].symbol.Ns.meaning; tcg_d256_flat_tma_wide:62 design.math_reshape.rewrite[0].symbol.Ns.meaning; tcg_d256_segmented_cp_narrow:47 design.math_reshape.rewrite[0].symbol.Ns.meaning; tcg_d256_segmented_tma_narrow:62 design.math_reshape.rewrite[0].symbol.Ns.meaning; tcg_d256_segmented_tma_wide:47 design.math_reshape.rewrite[0].symbol.Ns.meaning

Number of numerator partitions

## t_49312626ef3088ec | math.meaning | 46 chars | 12 declarations

tcg_d128_flat_cp_direct_narrow:158 design.math_reshape.rewrite[4].symbol.GroupPV.meaning; tcg_d128_flat_cp_direct_wide:158 design.math_reshape.rewrite[4].symbol.GroupPV.meaning; tcg_d128_flat_tma_direct_narrow:159 design.math_reshape.rewrite[4].symbol.GroupPV.meaning; tcg_d128_flat_tma_direct_wide:159 design.math_reshape.rewrite[4].symbol.GroupPV.meaning; tcg_d128_fscan_cp_direct_narrow:158 design.math_reshape.rewrite[4].symbol.GroupPV.meaning; tcg_d128_fscan_cp_direct_wide:158 design.math_reshape.rewrite[4].symbol.GroupPV.meaning; tcg_d128_fscan_tma_direct_narrow:158 design.math_reshape.rewrite[4].symbol.GroupPV.meaning; tcg_d128_fscan_tma_direct_wide:158 design.math_reshape.rewrite[4].symbol.GroupPV.meaning; tcg_d128_segmented_cp_direct_narrow:180 design.math_reshape.rewrite[5].symbol.GroupPV.meaning; tcg_d128_segmented_cp_direct_wide:179 design.math_reshape.rewrite[5].symbol.GroupPV.meaning; tcg_d128_segmented_tma_direct_narrow:180 design.math_reshape.rewrite[5].symbol.GroupPV.meaning; tcg_d128_segmented_tma_direct_wide:180 design.math_reshape.rewrite[5].symbol.GroupPV.meaning

Fresh PV result before register-state addition

## t_4c5c023903961365 | math.meaning | 50 chars | 4 declarations

tc_split_flat_narrow:129 design.math_reshape.rewrite[2].symbol.PriorMax.meaning; tc_split_flat_wide:129 design.math_reshape.rewrite[2].symbol.PriorMax.meaning; tc_split_segmented_narrow:126 design.math_reshape.rewrite[2].symbol.PriorMax.meaning; tc_split_segmented_wide:126 design.math_reshape.rewrite[2].symbol.PriorMax.meaning

Unwritten maximum-table input bytes viewed as FP32

## t_4dad92095bab4869 | math.meaning | 49 chars | 2 declarations

tc_split_segmented_narrow:143 design.math_reshape.rewrite[3].symbol.SegmentMax.meaning; tc_split_segmented_wide:143 design.math_reshape.rewrite[3].symbol.SegmentMax.meaning

Segment maxima including the unused first segment

## t_4ee0393d6ef66b23 | math.meaning | 56 chars | 1 declarations

tcg_d128_segmented_cp_direct_wide:216 design.math_reshape.rewrite[6].symbol.MergeB.meaning

Denominator sums of each interval's four 32-entry blocks

## t_4f15185682f1f33f | math.meaning | 32 chars | 1 declarations

tcg_d256_segmented_cp_wide:80 design.math_reshape.rewrite[1].symbol.HalfMax.meaning

Two partial maxima per query row

## t_4f5df71babf33cb4 | math.meaning | 48 chars | 1 declarations

tc_split_flat_wide:247 design.math_reshape.rewrite[5].symbol.GroupB.meaning

Saved 32-entry denominator tree sums for one row

## t_506dad8d5ef66dc4 | math.meaning | 48 chars | 15 declarations

tcg_d128_flat_cp_direct_narrow:99 design.math_reshape.rewrite[2].symbol.MTail.meaning; tcg_d128_flat_cp_direct_wide:99 design.math_reshape.rewrite[2].symbol.MTail.meaning; tcg_d128_flat_cp_xpose_narrow:101 design.math_reshape.rewrite[2].symbol.MTail.meaning; tcg_d128_flat_cp_xpose_wide:84 design.math_reshape.rewrite[2].symbol.MTail.meaning; tcg_d128_flat_tma_direct_narrow:100 design.math_reshape.rewrite[2].symbol.MTail.meaning; tcg_d128_flat_tma_direct_wide:100 design.math_reshape.rewrite[2].symbol.MTail.meaning; tcg_d128_flat_tma_xpose_narrow:101 design.math_reshape.rewrite[2].symbol.MTail.meaning; tcg_d128_flat_tma_xpose_wide:100 design.math_reshape.rewrite[2].symbol.MTail.meaning; tcg_d128_segmented_cp_direct_narrow:122 design.math_reshape.rewrite[3].symbol.MTail.meaning; tcg_d128_segmented_cp_direct_wide:121 design.math_reshape.rewrite[3].symbol.MTail.meaning; tcg_d128_segmented_cp_xpose_narrow:106 design.math_reshape.rewrite[3].symbol.MTail.meaning; tcg_d128_segmented_tma_direct_narrow:122 design.math_reshape.rewrite[3].symbol.MTail.meaning; tcg_d128_segmented_tma_direct_wide:122 design.math_reshape.rewrite[3].symbol.MTail.meaning; tcg_d128_segmented_tma_xpose_narrow:122 design.math_reshape.rewrite[3].symbol.MTail.meaning; tcg_d128_segmented_tma_xpose_wide:106 design.math_reshape.rewrite[3].symbol.MTail.meaning

Final delayed maximum used for weighting and LSE

## t_51535a8d03bed10a | math.meaning | 61 chars | 5 declarations

tc_fused:116 design.math_reshape.rewrite[2].symbol.E.meaning; tc_split_flat_narrow:168 design.math_reshape.rewrite[3].symbol.E.meaning; tc_split_flat_wide:168 design.math_reshape.rewrite[3].symbol.E.meaning; tc_split_segmented_narrow:203 design.math_reshape.rewrite[5].symbol.E.meaning; tc_split_segmented_wide:203 design.math_reshape.rewrite[5].symbol.E.meaning

Two ordered probability sums per partial in the current group

## t_516b6373b6ef5afa | math.meaning | 53 chars | 1 declarations

tcg_d128_fscan_tma_direct_wide:205 design.math_reshape.rewrite[5].symbol.GroupA.meaning

Group numerator partials retained for the final merge

## t_529d99e201bb9009 | math.meaning | 19 chars | 1 declarations

tc_fused:114 design.math_reshape.rewrite[2].symbol.MaxState.meaning

Delayed row maximum

## t_535d930956599fda | math.meaning | 37 chars | 1 declarations

tcg_d128_segmented_cp_direct_wide:214 design.math_reshape.rewrite[6].symbol.MergeA.meaning

Numerator sums of the merge intervals

## t_5a1a06564ffdc893 | math.meaning | 34 chars | 1 declarations

tcg_d256_flat_tma_wide:180 design.math_reshape.rewrite[5].symbol.SP.meaning

Number of local combine partitions

## t_5a295ef4cd16c711 | math.meaning | 23 chars | 4 declarations

tcg_d256_segmented_cp_narrow:84 design.math_reshape.rewrite[2].symbol.SegLen.meaning; tcg_d256_segmented_cp_wide:100 design.math_reshape.rewrite[2].symbol.SegLen.meaning; tcg_d256_segmented_tma_narrow:99 design.math_reshape.rewrite[2].symbol.SegLen.meaning; tcg_d256_segmented_tma_wide:84 design.math_reshape.rewrite[2].symbol.SegLen.meaning

Groups per scan segment

## t_5b8154713191eed6 | math.meaning | 42 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:124 design.math_reshape.rewrite[3].symbol.MState.meaning

Per-group delayed state replayed by splits

## t_5cbb8f3ef8db277a | math.meaning | 47 chars | 8 declarations

tcg_d128_flat_cp_xpose_narrow:164 design.math_reshape.rewrite[4].symbol.GroupPV.meaning; tcg_d128_flat_cp_xpose_wide:147 design.math_reshape.rewrite[4].symbol.GroupPV.meaning; tcg_d128_flat_tma_xpose_narrow:161 design.math_reshape.rewrite[4].symbol.GroupPV.meaning; tcg_d128_flat_tma_xpose_wide:163 design.math_reshape.rewrite[4].symbol.GroupPV.meaning; tcg_d128_fscan_tma_xpose_narrow:166 design.math_reshape.rewrite[4].symbol.GroupPV.meaning; tcg_d128_segmented_cp_xpose_narrow:169 design.math_reshape.rewrite[5].symbol.GroupPV.meaning; tcg_d128_segmented_tma_xpose_narrow:185 design.math_reshape.rewrite[5].symbol.GroupPV.meaning; tcg_d128_segmented_tma_xpose_wide:169 design.math_reshape.rewrite[5].symbol.GroupPV.meaning

Fresh PV result before numerator-state addition

## t_5d14711786255b0d | math.meaning | 41 chars | 1 declarations

tc_split_flat_wide:245 design.math_reshape.rewrite[5].symbol.GroupA.meaning

Saved numerator interval sums for one row

## t_5d517e3f581d9675 | math.meaning | 55 chars | 1 declarations

tcg_d128_fscan_cp_xpose_wide:207 design.math_reshape.rewrite[5].symbol.DenGroup.meaning

Four 32-entry denominator subtotals per merge partition

## t_6536d61e48b5c7a9 | math.meaning | 41 chars | 6 declarations

fused_d128_cp:65 design.math_reshape.rewrite[0].symbol.Br.meaning; fused_d128_group:64 design.math_reshape.rewrite[0].symbol.Br.meaning; fused_d128_key:65 design.math_reshape.rewrite[0].symbol.Br.meaning; fused_d256_cp:61 design.math_reshape.rewrite[0].symbol.Br.meaning; fused_d256_tma:61 design.math_reshape.rewrite[0].symbol.Br.meaning; root:275 design.math_reshape.rewrite[0].symbol.Br.meaning

Number of grouped query rows per worktile

## t_66beec8e9df65759 | math.meaning | 51 chars | 1 declarations

tcg_d128_flat_cp_direct_wide:194 design.math_reshape.rewrite[5].symbol.GroupB.meaning

Four denominator tree results per local split group

## t_66e2fa534f248b36 | math.meaning | 41 chars | 8 declarations

tcg_d256_flat_cp_narrow:152 design.math_reshape.rewrite[4].symbol.NumState.meaning; tcg_d256_flat_cp_wide:153 design.math_reshape.rewrite[4].symbol.NumState.meaning; tcg_d256_flat_tma_narrow:152 design.math_reshape.rewrite[4].symbol.NumState.meaning; tcg_d256_flat_tma_wide:152 design.math_reshape.rewrite[4].symbol.NumState.meaning; tcg_d256_segmented_cp_narrow:159 design.math_reshape.rewrite[5].symbol.NumState.meaning; tcg_d256_segmented_cp_wide:188 design.math_reshape.rewrite[5].symbol.NumState.meaning; tcg_d256_segmented_tma_narrow:174 design.math_reshape.rewrite[5].symbol.NumState.meaning; tcg_d256_segmented_tma_wide:159 design.math_reshape.rewrite[5].symbol.NumState.meaning

Split-local rounded-probability numerator

## t_686428fbc010fdbc | math.meaning | 35 chars | 3 declarations

tc_split_flat_narrow:235 design.math_reshape.rewrite[5].symbol.SumB.meaning; tc_split_segmented_narrow:270 design.math_reshape.rewrite[7].symbol.SumB.meaning; tc_split_segmented_wide:295 design.math_reshape.rewrite[7].symbol.SumB.meaning

Denominator sum over split partials

## t_6f272d9640f9e3f3 | math.meaning | 43 chars | 5 declarations

tc_fused:60 design.math_reshape.rewrite[0].symbol.Qraw.meaning; tc_split_flat_narrow:66 design.math_reshape.rewrite[0].symbol.Qraw.meaning; tc_split_flat_wide:66 design.math_reshape.rewrite[0].symbol.Qraw.meaning; tc_split_segmented_narrow:67 design.math_reshape.rewrite[0].symbol.Qraw.meaning; tc_split_segmented_wide:67 design.math_reshape.rewrite[0].symbol.Qraw.meaning

Query payload view for the grouped row tile

## t_72a2655e73b515c5 | math.meaning | 36 chars | 1 declarations

tcg_d256_flat_cp_wide:184 design.math_reshape.rewrite[5].symbol.GroupB.meaning

Denominator sums for 32-split blocks

## t_74aa37b34aeb0832 | math.meaning | 32 chars | 2 declarations

tc_split_segmented_narrow:142 design.math_reshape.rewrite[3].symbol.Seg.meaning; tc_split_segmented_wide:142 design.math_reshape.rewrite[3].symbol.Seg.meaning

Groups summarized by one segment

## t_75c54862c252e073 | math.meaning | 39 chars | 15 declarations

tcg_d128_flat_cp_direct_narrow:96 design.math_reshape.rewrite[2].symbol.ScanState.meaning; tcg_d128_flat_cp_direct_wide:96 design.math_reshape.rewrite[2].symbol.ScanState.meaning; tcg_d128_flat_cp_xpose_narrow:98 design.math_reshape.rewrite[2].symbol.ScanState.meaning; tcg_d128_flat_cp_xpose_wide:81 design.math_reshape.rewrite[2].symbol.ScanState.meaning; tcg_d128_flat_tma_direct_narrow:97 design.math_reshape.rewrite[2].symbol.ScanState.meaning; tcg_d128_flat_tma_direct_wide:97 design.math_reshape.rewrite[2].symbol.ScanState.meaning; tcg_d128_flat_tma_xpose_narrow:98 design.math_reshape.rewrite[2].symbol.ScanState.meaning; tcg_d128_flat_tma_xpose_wide:97 design.math_reshape.rewrite[2].symbol.ScanState.meaning; tcg_d128_segmented_cp_direct_narrow:120 design.math_reshape.rewrite[3].symbol.ScanState.meaning; tcg_d128_segmented_cp_direct_wide:119 design.math_reshape.rewrite[3].symbol.ScanState.meaning; tcg_d128_segmented_cp_xpose_narrow:103 design.math_reshape.rewrite[3].symbol.ScanState.meaning; tcg_d128_segmented_tma_direct_narrow:120 design.math_reshape.rewrite[3].symbol.ScanState.meaning; tcg_d128_segmented_tma_direct_wide:120 design.math_reshape.rewrite[3].symbol.ScanState.meaning; tcg_d128_segmented_tma_xpose_narrow:119 design.math_reshape.rewrite[3].symbol.ScanState.meaning; tcg_d128_segmented_tma_xpose_wide:103 design.math_reshape.rewrite[3].symbol.ScanState.meaning

Delayed maximum carried through one row

## t_75d7ff99016131b7 | math.meaning | 53 chars | 3 declarations

tcg_d128_fscan_cp_xpose_narrow:170 design.math_reshape.rewrite[4].symbol.PartB.meaning; tcg_d128_fscan_cp_xpose_wide:170 design.math_reshape.rewrite[4].symbol.PartB.meaning; tcg_d128_fscan_tma_xpose_wide:170 design.math_reshape.rewrite[4].symbol.PartB.meaning

Unweighted denominator for every split and output row

## t_7705f429c5d1f49d | math.meaning | 39 chars | 4 declarations

tc_split_flat_narrow:76 design.math_reshape.rewrite[0].symbol.Nm.meaning; tc_split_flat_wide:76 design.math_reshape.rewrite[0].symbol.Nm.meaning; tc_split_segmented_narrow:77 design.math_reshape.rewrite[0].symbol.Nm.meaning; tc_split_segmented_wide:77 design.math_reshape.rewrite[0].symbol.Nm.meaning

Number of maximum-computation intervals

## t_7ab395b2e014eac6 | math.meaning | 39 chars | 3 declarations

tcg_d128_segmented_cp_xpose_narrow:83 design.math_reshape.rewrite[2].symbol.SegMax.meaning; tcg_d128_segmented_tma_xpose_narrow:99 design.math_reshape.rewrite[2].symbol.SegMax.meaning; tcg_d128_segmented_tma_xpose_wide:83 design.math_reshape.rewrite[2].symbol.SegMax.meaning

Segment maxima locating replay suffixes

## t_7ce93f232cf4964b | math.meaning | 40 chars | 6 declarations

fused_d128_cp:149 design.math_reshape.rewrite[3].symbol.NumState.meaning; fused_d128_group:148 design.math_reshape.rewrite[3].symbol.NumState.meaning; fused_d128_key:151 design.math_reshape.rewrite[3].symbol.NumState.meaning; fused_d256_cp:144 design.math_reshape.rewrite[3].symbol.NumState.meaning; fused_d256_tma:144 design.math_reshape.rewrite[3].symbol.NumState.meaning; root:359 design.math_reshape.rewrite[3].symbol.NumState.meaning

FP32 rounded-probability value numerator

## t_7e10386fae84f2c7 | math.meaning | 45 chars | 7 declarations

tcg_d256_flat_cp_narrow:128 design.math_reshape.rewrite[3].symbol.DenPart.meaning; tcg_d256_flat_cp_wide:129 design.math_reshape.rewrite[3].symbol.DenPart.meaning; tcg_d256_flat_tma_narrow:128 design.math_reshape.rewrite[3].symbol.DenPart.meaning; tcg_d256_flat_tma_wide:128 design.math_reshape.rewrite[3].symbol.DenPart.meaning; tcg_d256_segmented_cp_narrow:135 design.math_reshape.rewrite[4].symbol.DenPart.meaning; tcg_d256_segmented_tma_narrow:150 design.math_reshape.rewrite[4].symbol.DenPart.meaning; tcg_d256_segmented_tma_wide:135 design.math_reshape.rewrite[4].symbol.DenPart.meaning

Split-local unrounded denominator by key half

## t_7e2c79c76f78ad93 | math.meaning | 21 chars | 24 declarations

tcg_d128_flat_cp_direct_narrow:62 design.math_reshape.rewrite[0].symbol.Ns.meaning; tcg_d128_flat_cp_direct_wide:62 design.math_reshape.rewrite[0].symbol.Ns.meaning; tcg_d128_flat_cp_xpose_narrow:64 design.math_reshape.rewrite[0].symbol.Ns.meaning; tcg_d128_flat_cp_xpose_wide:47 design.math_reshape.rewrite[0].symbol.Ns.meaning; tcg_d128_flat_tma_direct_narrow:63 design.math_reshape.rewrite[0].symbol.Ns.meaning; tcg_d128_flat_tma_direct_wide:63 design.math_reshape.rewrite[0].symbol.Ns.meaning; tcg_d128_flat_tma_xpose_narrow:64 design.math_reshape.rewrite[0].symbol.Ns.meaning; tcg_d128_flat_tma_xpose_wide:63 design.math_reshape.rewrite[0].symbol.Ns.meaning; tcg_d128_fscan_cp_direct_narrow:62 design.math_reshape.rewrite[0].symbol.Ns.meaning; tcg_d128_fscan_cp_direct_wide:62 design.math_reshape.rewrite[0].symbol.Ns.meaning; tcg_d128_fscan_cp_xpose_narrow:63 design.math_reshape.rewrite[0].symbol.Ns.meaning; tcg_d128_fscan_cp_xpose_wide:63 design.math_reshape.rewrite[0].symbol.Ns.meaning; tcg_d128_fscan_tma_direct_narrow:62 design.math_reshape.rewrite[0].symbol.Ns.meaning; tcg_d128_fscan_tma_direct_wide:62 design.math_reshape.rewrite[0].symbol.Ns.meaning; tcg_d128_fscan_tma_xpose_narrow:63 design.math_reshape.rewrite[0].symbol.Ns.meaning; tcg_d128_fscan_tma_xpose_wide:63 design.math_reshape.rewrite[0].symbol.Ns.meaning; tcg_d128_segmented_cp_direct_narrow:63 design.math_reshape.rewrite[0].symbol.Ns.meaning; tcg_d128_segmented_cp_direct_wide:62 design.math_reshape.rewrite[0].symbol.Ns.meaning; tcg_d128_segmented_cp_xpose_narrow:47 design.math_reshape.rewrite[0].symbol.Ns.meaning; tcg_d128_segmented_cp_xpose_wide:65 design.math_reshape.rewrite[0].symbol.Ns.meaning; tcg_d128_segmented_tma_direct_narrow:63 design.math_reshape.rewrite[0].symbol.Ns.meaning; tcg_d128_segmented_tma_direct_wide:63 design.math_reshape.rewrite[0].symbol.Ns.meaning; tcg_d128_segmented_tma_xpose_narrow:63 design.math_reshape.rewrite[0].symbol.Ns.meaning; tcg_d128_segmented_tma_xpose_wide:47 design.math_reshape.rewrite[0].symbol.Ns.meaning

Numerator split count

## t_826546120103c125 | math.meaning | 25 chars | 4 declarations

tcg_d128_fscan_cp_xpose_narrow:64 design.math_reshape.rewrite[0].symbol.Ngm.meaning; tcg_d128_fscan_cp_xpose_wide:64 design.math_reshape.rewrite[0].symbol.Ngm.meaning; tcg_d128_fscan_tma_xpose_wide:64 design.math_reshape.rewrite[0].symbol.Ngm.meaning; tcg_d128_segmented_cp_xpose_wide:66 design.math_reshape.rewrite[0].symbol.Ngm.meaning

Group-maximum split count

## t_829de7e9d61e5aba | math.meaning | 36 chars | 1 declarations

tcg_d128_fscan_tma_direct_wide:202 design.math_reshape.rewrite[5].symbol.MergeGroups.meaning

Number of numerical split-sum groups

## t_84c1b8412af7ea53 | math.meaning | 26 chars | 5 declarations

tc_fused:63 design.math_reshape.rewrite[0].symbol.QueryScale.meaning; tc_split_flat_narrow:70 design.math_reshape.rewrite[0].symbol.QueryScale.meaning; tc_split_flat_wide:70 design.math_reshape.rewrite[0].symbol.QueryScale.meaning; tc_split_segmented_narrow:71 design.math_reshape.rewrite[0].symbol.QueryScale.meaning; tc_split_segmented_wide:71 design.math_reshape.rewrite[0].symbol.QueryScale.meaning

Decoded query block scales

## t_85a7e2718f18699f | math.meaning | 50 chars | 3 declarations

tcg_d256_segmented_cp_narrow:86 design.math_reshape.rewrite[2].symbol.SegMax.meaning; tcg_d256_segmented_tma_narrow:101 design.math_reshape.rewrite[2].symbol.SegMax.meaning; tcg_d256_segmented_tma_wide:86 design.math_reshape.rewrite[2].symbol.SegMax.meaning

Segment maxima identifying unchanged scan prefixes

## t_85e3a672ef5a4e66 | math.meaning | 34 chars | 1 declarations

tcg_d128_flat_tma_direct_wide:193 design.math_reshape.rewrite[5].symbol.GroupA.meaning

Numerator sum for each merge group

## t_8674b767ad99b3d4 | math.meaning | 50 chars | 1 declarations

tcg_d128_fscan_tma_direct_wide:203 design.math_reshape.rewrite[5].symbol.Weight.meaning

Guarded split factors with zero-padded group tails

## t_8892718a8c36a221 | math.meaning | 64 chars | 4 declarations

tcg_d128_fscan_cp_xpose_narrow:168 design.math_reshape.rewrite[4].symbol.PartA.meaning; tcg_d128_fscan_cp_xpose_wide:168 design.math_reshape.rewrite[4].symbol.PartA.meaning; tcg_d128_fscan_tma_xpose_wide:168 design.math_reshape.rewrite[4].symbol.PartA.meaning; tcg_d128_segmented_cp_xpose_wide:191 design.math_reshape.rewrite[5].symbol.PartA.meaning

Materialized numerator including unspecified empty-split entries

## t_8abf5eeb06554f40 | math.meaning | 41 chars | 4 declarations

tcg_d128_fscan_cp_xpose_narrow:135 design.math_reshape.rewrite[3].symbol.Prob8.meaning; tcg_d128_fscan_cp_xpose_wide:135 design.math_reshape.rewrite[3].symbol.Prob8.meaning; tcg_d128_fscan_tma_xpose_wide:135 design.math_reshape.rewrite[3].symbol.Prob8.meaning; tcg_d128_segmented_cp_xpose_wide:158 design.math_reshape.rewrite[4].symbol.Prob8.meaning

Generated probability rows consumed by PV

## t_8b80c80173737d9f | math.meaning | 56 chars | 1 declarations

fused_d128_key:79 design.math_reshape.rewrite[1].symbol.HalfMax.meaning

Unscaled maximum of each key slice for the current group

## t_8fd7e58b408e8695 | math.meaning | 47 chars | 15 declarations

tcg_d128_flat_cp_direct_narrow:162 design.math_reshape.rewrite[4].symbol.PartB.meaning; tcg_d128_flat_cp_direct_wide:162 design.math_reshape.rewrite[4].symbol.PartB.meaning; tcg_d128_flat_cp_xpose_narrow:168 design.math_reshape.rewrite[4].symbol.PartB.meaning; tcg_d128_flat_cp_xpose_wide:151 design.math_reshape.rewrite[4].symbol.PartB.meaning; tcg_d128_flat_tma_direct_narrow:163 design.math_reshape.rewrite[4].symbol.PartB.meaning; tcg_d128_flat_tma_direct_wide:163 design.math_reshape.rewrite[4].symbol.PartB.meaning; tcg_d128_flat_tma_xpose_narrow:165 design.math_reshape.rewrite[4].symbol.PartB.meaning; tcg_d128_flat_tma_xpose_wide:167 design.math_reshape.rewrite[4].symbol.PartB.meaning; tcg_d128_segmented_cp_direct_narrow:184 design.math_reshape.rewrite[5].symbol.PartB.meaning; tcg_d128_segmented_cp_direct_wide:183 design.math_reshape.rewrite[5].symbol.PartB.meaning; tcg_d128_segmented_cp_xpose_narrow:173 design.math_reshape.rewrite[5].symbol.PartB.meaning; tcg_d128_segmented_tma_direct_narrow:184 design.math_reshape.rewrite[5].symbol.PartB.meaning; tcg_d128_segmented_tma_direct_wide:184 design.math_reshape.rewrite[5].symbol.PartB.meaning; tcg_d128_segmented_tma_xpose_narrow:189 design.math_reshape.rewrite[5].symbol.PartB.meaning; tcg_d128_segmented_tma_xpose_wide:173 design.math_reshape.rewrite[5].symbol.PartB.meaning

Weighted split denominators consumed by combine

## t_912423a4fea567d7 | math.meaning | 33 chars | 7 declarations

tcg_d256_flat_cp_narrow:153 design.math_reshape.rewrite[4].symbol.DenFinal.meaning; tcg_d256_flat_cp_wide:154 design.math_reshape.rewrite[4].symbol.DenFinal.meaning; tcg_d256_flat_tma_narrow:153 design.math_reshape.rewrite[4].symbol.DenFinal.meaning; tcg_d256_flat_tma_wide:153 design.math_reshape.rewrite[4].symbol.DenFinal.meaning; tcg_d256_segmented_cp_narrow:160 design.math_reshape.rewrite[5].symbol.DenFinal.meaning; tcg_d256_segmented_tma_narrow:175 design.math_reshape.rewrite[5].symbol.DenFinal.meaning; tcg_d256_segmented_tma_wide:160 design.math_reshape.rewrite[5].symbol.DenFinal.meaning

Sum of the two local denominators

## t_913fb8d6a11bead8 | math.meaning | 52 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:125 design.math_reshape.rewrite[3].symbol.MTail.meaning

Final delayed maximum used for split weights and LSE

## t_9244ef482b0a2403 | math.meaning | 31 chars | 8 declarations

tcg_d256_flat_cp_narrow:65 design.math_reshape.rewrite[0].symbol.PSK.meaning; tcg_d256_flat_cp_wide:66 design.math_reshape.rewrite[0].symbol.PSK.meaning; tcg_d256_flat_tma_narrow:65 design.math_reshape.rewrite[0].symbol.PSK.meaning; tcg_d256_flat_tma_wide:65 design.math_reshape.rewrite[0].symbol.PSK.meaning; tcg_d256_segmented_cp_narrow:50 design.math_reshape.rewrite[0].symbol.PSK.meaning; tcg_d256_segmented_cp_wide:65 design.math_reshape.rewrite[0].symbol.PSK.meaning; tcg_d256_segmented_tma_narrow:65 design.math_reshape.rewrite[0].symbol.PSK.meaning; tcg_d256_segmented_tma_wide:50 design.math_reshape.rewrite[0].symbol.PSK.meaning

Query-major padding-skip choice

## t_96603a4a4172c2ee | math.meaning | 52 chars | 1 declarations

tcg_d128_fscan_tma_xpose_narrow:104 design.math_reshape.rewrite[2].symbol.MaxState.meaning

Locally scanned delayed maximum including the prefix

## t_9d9f4369b4bde452 | math.meaning | 48 chars | 7 declarations

tcg_d256_flat_cp_narrow:82 design.math_reshape.rewrite[1].symbol.Score.meaning; tcg_d256_flat_cp_wide:83 design.math_reshape.rewrite[1].symbol.Score.meaning; tcg_d256_flat_tma_narrow:82 design.math_reshape.rewrite[1].symbol.Score.meaning; tcg_d256_flat_tma_wide:82 design.math_reshape.rewrite[1].symbol.Score.meaning; tcg_d256_segmented_cp_narrow:67 design.math_reshape.rewrite[1].symbol.Score.meaning; tcg_d256_segmented_tma_narrow:82 design.math_reshape.rewrite[1].symbol.Score.meaning; tcg_d256_segmented_tma_wide:67 design.math_reshape.rewrite[1].symbol.Score.meaning

Split scores retained for probability evaluation

## t_9e0f7b7fa0994479 | math.meaning | 37 chars | 1 declarations

tcg_d256_segmented_cp_wide:81 design.math_reshape.rewrite[1].symbol.GroupMax.meaning

Scaled group maxima for the full scan

## t_9e3765c8157a26df | math.meaning | 26 chars | 20 declarations

tcg_d128_flat_cp_direct_narrow:60 design.math_reshape.rewrite[0].symbol.Br.meaning; tcg_d128_flat_cp_direct_wide:60 design.math_reshape.rewrite[0].symbol.Br.meaning; tcg_d128_flat_cp_xpose_narrow:62 design.math_reshape.rewrite[0].symbol.Br.meaning; tcg_d128_flat_cp_xpose_wide:45 design.math_reshape.rewrite[0].symbol.Br.meaning; tcg_d128_flat_tma_direct_narrow:61 design.math_reshape.rewrite[0].symbol.Br.meaning; tcg_d128_flat_tma_direct_wide:61 design.math_reshape.rewrite[0].symbol.Br.meaning; tcg_d128_flat_tma_xpose_narrow:62 design.math_reshape.rewrite[0].symbol.Br.meaning; tcg_d128_flat_tma_xpose_wide:61 design.math_reshape.rewrite[0].symbol.Br.meaning; tcg_d128_fscan_cp_direct_narrow:60 design.math_reshape.rewrite[0].symbol.Br.meaning; tcg_d128_fscan_cp_direct_wide:60 design.math_reshape.rewrite[0].symbol.Br.meaning; tcg_d128_fscan_tma_direct_narrow:60 design.math_reshape.rewrite[0].symbol.Br.meaning; tcg_d128_fscan_tma_direct_wide:60 design.math_reshape.rewrite[0].symbol.Br.meaning; tcg_d128_fscan_tma_xpose_narrow:61 design.math_reshape.rewrite[0].symbol.Br.meaning; tcg_d128_segmented_cp_direct_narrow:61 design.math_reshape.rewrite[0].symbol.Br.meaning; tcg_d128_segmented_cp_direct_wide:60 design.math_reshape.rewrite[0].symbol.Br.meaning; tcg_d128_segmented_cp_xpose_narrow:45 design.math_reshape.rewrite[0].symbol.Br.meaning; tcg_d128_segmented_tma_direct_narrow:61 design.math_reshape.rewrite[0].symbol.Br.meaning; tcg_d128_segmented_tma_direct_wide:61 design.math_reshape.rewrite[0].symbol.Br.meaning; tcg_d128_segmented_tma_xpose_narrow:61 design.math_reshape.rewrite[0].symbol.Br.meaning; tcg_d128_segmented_tma_xpose_wide:45 design.math_reshape.rewrite[0].symbol.Br.meaning

Grouped query-row capacity

## t_9e47dec2839bd284 | math.meaning | 31 chars | 7 declarations

tcg_d256_flat_cp_narrow:81 design.math_reshape.rewrite[1].symbol.GroupMax.meaning; tcg_d256_flat_cp_wide:82 design.math_reshape.rewrite[1].symbol.GroupMax.meaning; tcg_d256_flat_tma_narrow:81 design.math_reshape.rewrite[1].symbol.GroupMax.meaning; tcg_d256_flat_tma_wide:81 design.math_reshape.rewrite[1].symbol.GroupMax.meaning; tcg_d256_segmented_cp_narrow:66 design.math_reshape.rewrite[1].symbol.GroupMax.meaning; tcg_d256_segmented_tma_narrow:81 design.math_reshape.rewrite[1].symbol.GroupMax.meaning; tcg_d256_segmented_tma_wide:66 design.math_reshape.rewrite[1].symbol.GroupMax.meaning

Scaled maxima for the full scan

## t_9f8f5b7e2ff781e0 | math.meaning | 20 chars | 4 declarations

tcg_d128_fscan_cp_xpose_narrow:62 design.math_reshape.rewrite[0].symbol.Br.meaning; tcg_d128_fscan_cp_xpose_wide:62 design.math_reshape.rewrite[0].symbol.Br.meaning; tcg_d128_fscan_tma_xpose_wide:62 design.math_reshape.rewrite[0].symbol.Br.meaning; tcg_d128_segmented_cp_xpose_wide:64 design.math_reshape.rewrite[0].symbol.Br.meaning

Grouped row capacity

## t_a2375456436241fa | math.meaning | 53 chars | 4 declarations

tcg_d128_fscan_cp_xpose_narrow:195 design.math_reshape.rewrite[5].symbol.TotalA.meaning; tcg_d128_fscan_cp_xpose_wide:208 design.math_reshape.rewrite[5].symbol.TotalA.meaning; tcg_d128_fscan_tma_xpose_wide:209 design.math_reshape.rewrite[5].symbol.TotalA.meaning; tcg_d128_segmented_cp_xpose_wide:228 design.math_reshape.rewrite[6].symbol.TotalA.meaning

Merged numerator retained through final normalization

## t_a27d8e7925788928 | math.meaning | 28 chars | 4 declarations

tc_split_flat_narrow:127 design.math_reshape.rewrite[2].symbol.BmCovered.meaning; tc_split_flat_wide:127 design.math_reshape.rewrite[2].symbol.BmCovered.meaning; tc_split_segmented_narrow:124 design.math_reshape.rewrite[2].symbol.BmCovered.meaning; tc_split_segmented_wide:124 design.math_reshape.rewrite[2].symbol.BmCovered.meaning

Written maximum-table region

## t_a60ab0ac6b478403 | math.meaning | 40 chars | 4 declarations

tc_split_flat_narrow:137 design.math_reshape.rewrite[2].symbol.OTemporary.meaning; tc_split_flat_wide:137 design.math_reshape.rewrite[2].symbol.OTemporary.meaning; tc_split_segmented_narrow:128 design.math_reshape.rewrite[2].symbol.OTemporary.meaning; tc_split_segmented_wide:128 design.math_reshape.rewrite[2].symbol.OTemporary.meaning

Temporary zero values stored in caller O

## t_a67daafbfbb134e5 | math.meaning | 21 chars | 20 declarations

tcg_d128_flat_cp_direct_narrow:61 design.math_reshape.rewrite[0].symbol.Ngm.meaning; tcg_d128_flat_cp_direct_wide:61 design.math_reshape.rewrite[0].symbol.Ngm.meaning; tcg_d128_flat_cp_xpose_narrow:63 design.math_reshape.rewrite[0].symbol.Ngm.meaning; tcg_d128_flat_cp_xpose_wide:46 design.math_reshape.rewrite[0].symbol.Ngm.meaning; tcg_d128_flat_tma_direct_narrow:62 design.math_reshape.rewrite[0].symbol.Ngm.meaning; tcg_d128_flat_tma_direct_wide:62 design.math_reshape.rewrite[0].symbol.Ngm.meaning; tcg_d128_flat_tma_xpose_narrow:63 design.math_reshape.rewrite[0].symbol.Ngm.meaning; tcg_d128_flat_tma_xpose_wide:62 design.math_reshape.rewrite[0].symbol.Ngm.meaning; tcg_d128_fscan_cp_direct_narrow:61 design.math_reshape.rewrite[0].symbol.Ngm.meaning; tcg_d128_fscan_cp_direct_wide:61 design.math_reshape.rewrite[0].symbol.Ngm.meaning; tcg_d128_fscan_tma_direct_narrow:61 design.math_reshape.rewrite[0].symbol.Ngm.meaning; tcg_d128_fscan_tma_direct_wide:61 design.math_reshape.rewrite[0].symbol.Ngm.meaning; tcg_d128_fscan_tma_xpose_narrow:62 design.math_reshape.rewrite[0].symbol.Ngm.meaning; tcg_d128_segmented_cp_direct_narrow:62 design.math_reshape.rewrite[0].symbol.Ngm.meaning; tcg_d128_segmented_cp_direct_wide:61 design.math_reshape.rewrite[0].symbol.Ngm.meaning; tcg_d128_segmented_cp_xpose_narrow:46 design.math_reshape.rewrite[0].symbol.Ngm.meaning; tcg_d128_segmented_tma_direct_narrow:62 design.math_reshape.rewrite[0].symbol.Ngm.meaning; tcg_d128_segmented_tma_direct_wide:62 design.math_reshape.rewrite[0].symbol.Ngm.meaning; tcg_d128_segmented_tma_xpose_narrow:62 design.math_reshape.rewrite[0].symbol.Ngm.meaning; tcg_d128_segmented_tma_xpose_wide:46 design.math_reshape.rewrite[0].symbol.Ngm.meaning

Group-max split count

## t_a8be8658ab1faa92 | math.meaning | 24 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:155 design.math_reshape.rewrite[4].symbol.MaxState.meaning

Current replayed maximum

## t_aa7516baa88a6c31 | math.meaning | 39 chars | 1 declarations

tcg_d256_flat_tma_wide:181 design.math_reshape.rewrite[5].symbol.GroupNum.meaning

Numerator sum of each combine partition

## t_ab445420b6bbee0f | math.meaning | 76 chars | 3 declarations

tcg_d128_fscan_cp_direct_narrow:190 design.math_reshape.rewrite[5].symbol.Weight.meaning; tcg_d128_fscan_cp_direct_wide:201 design.math_reshape.rewrite[5].symbol.Weight.meaning; tcg_d128_fscan_tma_direct_narrow:190 design.math_reshape.rewrite[5].symbol.Weight.meaning

Guarded split rescaling factors shared across numerator and denominator sums

## t_abaacd5e2908f6b6 | math.meaning | 36 chars | 4 declarations

tcg_d128_fscan_cp_xpose_narrow:196 design.math_reshape.rewrite[5].symbol.TotalB.meaning; tcg_d128_fscan_cp_xpose_wide:209 design.math_reshape.rewrite[5].symbol.TotalB.meaning; tcg_d128_fscan_tma_xpose_wide:210 design.math_reshape.rewrite[5].symbol.TotalB.meaning; tcg_d128_segmented_cp_xpose_wide:229 design.math_reshape.rewrite[6].symbol.TotalB.meaning

Merged denominator used by O and LSE

## t_ac8fde9780d8aca6 | math.meaning | 39 chars | 4 declarations

tc_split_flat_narrow:131 design.math_reshape.rewrite[2].symbol.ScanCarry.meaning; tc_split_flat_wide:131 design.math_reshape.rewrite[2].symbol.ScanCarry.meaning; tc_split_segmented_narrow:170 design.math_reshape.rewrite[4].symbol.ScanCarry.meaning; tc_split_segmented_wide:170 design.math_reshape.rewrite[4].symbol.ScanCarry.meaning

Current delayed maximum during scanning

## t_ac9e8b29a8a0d4f1 | math.meaning | 50 chars | 1 declarations

tcg_d128_segmented_tma_direct_wide:216 design.math_reshape.rewrite[6].symbol.GroupA.meaning

Interval numerator sums retained for final merging

## t_adddc7dd7c73709f | math.meaning | 55 chars | 1 declarations

tcg_d256_segmented_cp_wide:102 design.math_reshape.rewrite[2].symbol.SegMax.meaning

Segment maxima used to identify unchanged scan prefixes

## t_afcf0d84ed29fb05 | math.meaning | 56 chars | 4 declarations

tcg_d128_fscan_cp_direct_narrow:166 design.math_reshape.rewrite[4].symbol.MTail.meaning; tcg_d128_fscan_cp_direct_wide:166 design.math_reshape.rewrite[4].symbol.MTail.meaning; tcg_d128_fscan_tma_direct_narrow:166 design.math_reshape.rewrite[4].symbol.MTail.meaning; tcg_d128_fscan_tma_direct_wide:166 design.math_reshape.rewrite[4].symbol.MTail.meaning

Last split's delayed maximum consumed by combine and LSE

## t_b2fc12dc2eb4f3fd | math.meaning | 37 chars | 20 declarations

tcg_d128_flat_cp_direct_narrow:79 design.math_reshape.rewrite[1].symbol.DotTile.meaning; tcg_d128_flat_cp_direct_wide:79 design.math_reshape.rewrite[1].symbol.DotTile.meaning; tcg_d128_flat_cp_xpose_narrow:81 design.math_reshape.rewrite[1].symbol.DotTile.meaning; tcg_d128_flat_cp_xpose_wide:64 design.math_reshape.rewrite[1].symbol.DotTile.meaning; tcg_d128_flat_tma_direct_narrow:80 design.math_reshape.rewrite[1].symbol.DotTile.meaning; tcg_d128_flat_tma_direct_wide:80 design.math_reshape.rewrite[1].symbol.DotTile.meaning; tcg_d128_flat_tma_xpose_narrow:81 design.math_reshape.rewrite[1].symbol.DotTile.meaning; tcg_d128_flat_tma_xpose_wide:80 design.math_reshape.rewrite[1].symbol.DotTile.meaning; tcg_d128_fscan_cp_direct_narrow:79 design.math_reshape.rewrite[1].symbol.DotTile.meaning; tcg_d128_fscan_cp_direct_wide:79 design.math_reshape.rewrite[1].symbol.DotTile.meaning; tcg_d128_fscan_tma_direct_narrow:79 design.math_reshape.rewrite[1].symbol.DotTile.meaning; tcg_d128_fscan_tma_direct_wide:79 design.math_reshape.rewrite[1].symbol.DotTile.meaning; tcg_d128_fscan_tma_xpose_narrow:80 design.math_reshape.rewrite[1].symbol.DotTile.meaning; tcg_d128_segmented_cp_direct_narrow:80 design.math_reshape.rewrite[1].symbol.DotTile.meaning; tcg_d128_segmented_cp_direct_wide:79 design.math_reshape.rewrite[1].symbol.DotTile.meaning; tcg_d128_segmented_cp_xpose_narrow:64 design.math_reshape.rewrite[1].symbol.DotTile.meaning; tcg_d128_segmented_tma_direct_narrow:80 design.math_reshape.rewrite[1].symbol.DotTile.meaning; tcg_d128_segmented_tma_direct_wide:80 design.math_reshape.rewrite[1].symbol.DotTile.meaning; tcg_d128_segmented_tma_xpose_narrow:80 design.math_reshape.rewrite[1].symbol.DotTile.meaning; tcg_d128_segmented_tma_xpose_wide:64 design.math_reshape.rewrite[1].symbol.DotTile.meaning

Unscaled QK tile in either evaluation

## t_b4c491b00de594a5 | math.meaning | 47 chars | 1 declarations

tcg_d256_flat_cp_wide:182 design.math_reshape.rewrite[5].symbol.Cg.meaning

Groups of split partials in the final reduction

## t_b50bc15e51488104 | math.meaning | 31 chars | 5 declarations

tc_fused:59 design.math_reshape.rewrite[0].symbol.Br.meaning; tc_split_flat_narrow:65 design.math_reshape.rewrite[0].symbol.Br.meaning; tc_split_flat_wide:65 design.math_reshape.rewrite[0].symbol.Br.meaning; tc_split_segmented_narrow:66 design.math_reshape.rewrite[0].symbol.Br.meaning; tc_split_segmented_wide:66 design.math_reshape.rewrite[0].symbol.Br.meaning

Grouped query rows per worktile

## t_bb15ed48f8e3c1ee | math.meaning | 51 chars | 1 declarations

fused_d128_key:122 design.math_reshape.rewrite[2].symbol.DenSlice.meaning

Unrounded-probability normalizer for each key slice

## t_bb4995adea7d81b2 | math.meaning | 33 chars | 1 declarations

tcg_d256_flat_cp_wide:183 design.math_reshape.rewrite[5].symbol.GroupA.meaning

Numerator sums for combine groups

## t_bc874ed643a31cc0 | math.meaning | 45 chars | 9 declarations

tcg_d128_flat_cp_direct_narrow:160 design.math_reshape.rewrite[4].symbol.PartA.meaning; tcg_d128_flat_cp_direct_wide:160 design.math_reshape.rewrite[4].symbol.PartA.meaning; tcg_d128_flat_tma_direct_narrow:161 design.math_reshape.rewrite[4].symbol.PartA.meaning; tcg_d128_flat_tma_direct_wide:161 design.math_reshape.rewrite[4].symbol.PartA.meaning; tcg_d128_flat_tma_xpose_narrow:163 design.math_reshape.rewrite[4].symbol.PartA.meaning; tcg_d128_segmented_cp_direct_narrow:182 design.math_reshape.rewrite[5].symbol.PartA.meaning; tcg_d128_segmented_cp_direct_wide:181 design.math_reshape.rewrite[5].symbol.PartA.meaning; tcg_d128_segmented_tma_direct_narrow:182 design.math_reshape.rewrite[5].symbol.PartA.meaning; tcg_d128_segmented_tma_direct_wide:182 design.math_reshape.rewrite[5].symbol.PartA.meaning

Weighted split numerators consumed by combine

## t_c07ff29e51541d41 | math.meaning | 38 chars | 1 declarations

tcg_d128_fscan_cp_direct_wide:199 design.math_reshape.rewrite[5].symbol.MergeGroups.meaning

Number of local split-reduction groups

## t_c19ef2f722f75cf0 | math.meaning | 35 chars | 1 declarations

tcg_d128_flat_tma_direct_wide:192 design.math_reshape.rewrite[5].symbol.MergeGroups.meaning

Cooperating split-merge group count

## t_c3939522fb321bc5 | math.meaning | 34 chars | 2 declarations

fused_d256_cp:74 design.math_reshape.rewrite[1].symbol.HalfMax.meaning; fused_d256_tma:74 design.math_reshape.rewrite[1].symbol.HalfMax.meaning

Maximum of each row's 64-key slice

## t_c3ccd9e4da62a723 | math.meaning | 49 chars | 5 declarations

tcg_d128_fscan_cp_direct_narrow:162 design.math_reshape.rewrite[4].symbol.PartB.meaning; tcg_d128_fscan_cp_direct_wide:162 design.math_reshape.rewrite[4].symbol.PartB.meaning; tcg_d128_fscan_tma_direct_narrow:162 design.math_reshape.rewrite[4].symbol.PartB.meaning; tcg_d128_fscan_tma_direct_wide:162 design.math_reshape.rewrite[4].symbol.PartB.meaning; tcg_d128_fscan_tma_xpose_narrow:170 design.math_reshape.rewrite[4].symbol.PartB.meaning

Unweighted split denominators consumed by combine

## t_c3d942037fc1488f | math.meaning | 47 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:225 design.math_reshape.rewrite[6].symbol.MergeGroups.meaning

Parallel intervals in the final split reduction

## t_c4b77cc43a70dbf3 | math.meaning | 21 chars | 8 declarations

tcg_d256_flat_cp_narrow:101 design.math_reshape.rewrite[2].symbol.Mtail.meaning; tcg_d256_flat_cp_wide:102 design.math_reshape.rewrite[2].symbol.Mtail.meaning; tcg_d256_flat_tma_narrow:101 design.math_reshape.rewrite[2].symbol.Mtail.meaning; tcg_d256_flat_tma_wide:101 design.math_reshape.rewrite[2].symbol.Mtail.meaning; tcg_d256_segmented_cp_narrow:108 design.math_reshape.rewrite[3].symbol.Mtail.meaning; tcg_d256_segmented_cp_wide:127 design.math_reshape.rewrite[3].symbol.Mtail.meaning; tcg_d256_segmented_tma_narrow:123 design.math_reshape.rewrite[3].symbol.Mtail.meaning; tcg_d256_segmented_tma_wide:108 design.math_reshape.rewrite[3].symbol.Mtail.meaning

Final delayed maximum

## t_c66f8d240aac7d80 | math.meaning | 38 chars | 1 declarations

tcg_d256_flat_tma_wide:182 design.math_reshape.rewrite[5].symbol.GroupDen.meaning

Denominator sums of 32 split positions

## t_c6bdc882af78a31e | math.meaning | 56 chars | 1 declarations

tcg_d128_fscan_tma_xpose_narrow:195 design.math_reshape.rewrite[5].symbol.Weight.meaning

Split weights reused by all channels and the denominator

## t_c8e9dc43d37dedae | math.meaning | 26 chars | 4 declarations

tcg_d128_fscan_cp_xpose_narrow:81 design.math_reshape.rewrite[1].symbol.DotTile.meaning; tcg_d128_fscan_cp_xpose_wide:81 design.math_reshape.rewrite[1].symbol.DotTile.meaning; tcg_d128_fscan_tma_xpose_wide:81 design.math_reshape.rewrite[1].symbol.DotTile.meaning; tcg_d128_segmented_cp_xpose_wide:83 design.math_reshape.rewrite[1].symbol.DotTile.meaning

Unscaled current QK result

## t_c97a872fc60c4581 | math.meaning | 38 chars | 15 declarations

tcg_d128_flat_cp_direct_narrow:97 design.math_reshape.rewrite[2].symbol.MState.meaning; tcg_d128_flat_cp_direct_wide:97 design.math_reshape.rewrite[2].symbol.MState.meaning; tcg_d128_flat_cp_xpose_narrow:99 design.math_reshape.rewrite[2].symbol.MState.meaning; tcg_d128_flat_cp_xpose_wide:82 design.math_reshape.rewrite[2].symbol.MState.meaning; tcg_d128_flat_tma_direct_narrow:98 design.math_reshape.rewrite[2].symbol.MState.meaning; tcg_d128_flat_tma_direct_wide:98 design.math_reshape.rewrite[2].symbol.MState.meaning; tcg_d128_flat_tma_xpose_narrow:99 design.math_reshape.rewrite[2].symbol.MState.meaning; tcg_d128_flat_tma_xpose_wide:98 design.math_reshape.rewrite[2].symbol.MState.meaning; tcg_d128_segmented_cp_direct_narrow:121 design.math_reshape.rewrite[3].symbol.MState.meaning; tcg_d128_segmented_cp_direct_wide:120 design.math_reshape.rewrite[3].symbol.MState.meaning; tcg_d128_segmented_cp_xpose_narrow:104 design.math_reshape.rewrite[3].symbol.MState.meaning; tcg_d128_segmented_tma_direct_narrow:121 design.math_reshape.rewrite[3].symbol.MState.meaning; tcg_d128_segmented_tma_direct_wide:121 design.math_reshape.rewrite[3].symbol.MState.meaning; tcg_d128_segmented_tma_xpose_narrow:120 design.math_reshape.rewrite[3].symbol.MState.meaning; tcg_d128_segmented_tma_xpose_wide:104 design.math_reshape.rewrite[3].symbol.MState.meaning

Snapshots replayed by numerator splits

## t_c99ab9b16f4d66e3 | math.meaning | 39 chars | 1 declarations

tcg_d128_flat_cp_direct_wide:192 design.math_reshape.rewrite[5].symbol.MergeGroups.meaning

Number of local split groups in combine

## t_ca759675265387a1 | math.meaning | 54 chars | 1 declarations

fused_d128_group:120 design.math_reshape.rewrite[2].symbol.alpha.meaning

Current group numerator and denominator rescale factor

## t_cba92034dc1b2292 | math.meaning | 50 chars | 5 declarations

tc_fused:115 design.math_reshape.rewrite[2].symbol.DenPart.meaning; tc_split_flat_narrow:166 design.math_reshape.rewrite[3].symbol.DenPart.meaning; tc_split_flat_wide:166 design.math_reshape.rewrite[3].symbol.DenPart.meaning; tc_split_segmented_narrow:201 design.math_reshape.rewrite[5].symbol.DenPart.meaning; tc_split_segmented_wide:201 design.math_reshape.rewrite[5].symbol.DenPart.meaning

Four separately carried unrounded-probability sums

## t_cd737d0c39ade756 | math.meaning | 45 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:101 design.math_reshape.rewrite[2].symbol.SegMax.meaning

Segment maxima used to locate replay suffixes

## t_ce8a63cbb132a1ee | math.meaning | 39 chars | 1 declarations

tcg_d256_segmented_cp_wide:64 design.math_reshape.rewrite[0].symbol.Br.meaning

Grouped query rows per compute worktile

## t_cebf964310a2ee15 | math.meaning | 57 chars | 5 declarations

tc_fused:152 design.math_reshape.rewrite[3].symbol.NumState.meaning; tc_split_flat_narrow:207 design.math_reshape.rewrite[4].symbol.NumState.meaning; tc_split_flat_wide:207 design.math_reshape.rewrite[4].symbol.NumState.meaning; tc_split_segmented_narrow:242 design.math_reshape.rewrite[6].symbol.NumState.meaning; tc_split_segmented_wide:242 design.math_reshape.rewrite[6].symbol.NumState.meaning

FP32 numerator from rounded probabilities and BF16 values

## t_ced00bbffb2282fd | math.meaning | 56 chars | 1 declarations

fused_d128_group:119 design.math_reshape.rewrite[2].symbol.DenState.meaning

Unrounded-probability normalizer after the current group

## t_cf7c5e4240a26926 | math.meaning | 59 chars | 1 declarations

tcg_d256_segmented_cp_wide:221 design.math_reshape.rewrite[6].symbol.SP.meaning

Number of contiguous split intervals in the final reduction

## t_d05e2ced9b122e1e | math.meaning | 48 chars | 1 declarations

tcg_d128_segmented_tma_direct_wide:218 design.math_reshape.rewrite[6].symbol.GroupB.meaning

Denominator sums of consecutive 32-split subsets

## t_d0887ebb3128acb7 | math.meaning | 42 chars | 1 declarations

tcg_d128_segmented_tma_direct_wide:215 design.math_reshape.rewrite[6].symbol.MergeGroups.meaning

Number of split intervals in one row merge

## t_d14da1fe3d60b698 | math.meaning | 46 chars | 1 declarations

tcg_d128_fscan_cp_direct_wide:200 design.math_reshape.rewrite[5].symbol.MergeWidth.meaning

Split capacity of each local denominator group

## t_d191c1acabaa3794 | math.meaning | 36 chars | 1 declarations

tc_split_segmented_wide:283 design.math_reshape.rewrite[7].symbol.CombineGroups.meaning

Independent groups of split partials

## t_d2842f4178f1d3b0 | math.meaning | 66 chars | 1 declarations

fused_d128_key:123 design.math_reshape.rewrite[2].symbol.DenFinal.meaning

Final unrounded-probability denominator after combining key slices

## t_d403b6fcf4fb28ad | math.meaning | 36 chars | 1 declarations

tcg_d128_fscan_cp_direct_wide:203 design.math_reshape.rewrite[5].symbol.GroupA.meaning

Local merge-group numerator partials

## t_d5f8367dfa234a8b | math.meaning | 46 chars | 6 declarations

fused_d128_cp:122 design.math_reshape.rewrite[2].symbol.Prob8.meaning; fused_d128_group:121 design.math_reshape.rewrite[2].symbol.Prob8.meaning; fused_d128_key:124 design.math_reshape.rewrite[2].symbol.Prob8.meaning; fused_d256_cp:117 design.math_reshape.rewrite[2].symbol.Prob8.meaning; fused_d256_tma:117 design.math_reshape.rewrite[2].symbol.Prob8.meaning; root:332 design.math_reshape.rewrite[2].symbol.Prob8.meaning

Current unit-scale rounded probability operand

## t_d76944a58c2e3727 | math.meaning | 50 chars | 2 declarations

fused_d256_cp:116 design.math_reshape.rewrite[2].symbol.DenPart.meaning; fused_d256_tma:116 design.math_reshape.rewrite[2].symbol.DenPart.meaning

Unrounded-probability normalizer for each key half

## t_da6da0e49ab73001 | math.meaning | 33 chars | 7 declarations

tcg_d256_flat_cp_narrow:64 design.math_reshape.rewrite[0].symbol.Br.meaning; tcg_d256_flat_cp_wide:65 design.math_reshape.rewrite[0].symbol.Br.meaning; tcg_d256_flat_tma_narrow:64 design.math_reshape.rewrite[0].symbol.Br.meaning; tcg_d256_flat_tma_wide:64 design.math_reshape.rewrite[0].symbol.Br.meaning; tcg_d256_segmented_cp_narrow:49 design.math_reshape.rewrite[0].symbol.Br.meaning; tcg_d256_segmented_tma_narrow:64 design.math_reshape.rewrite[0].symbol.Br.meaning; tcg_d256_segmented_tma_wide:49 design.math_reshape.rewrite[0].symbol.Br.meaning

Grouped rows per compute worktile

## t_dbb9051f05054769 | math.meaning | 47 chars | 1 declarations

tc_split_segmented_wide:285 design.math_reshape.rewrite[7].symbol.GroupA.meaning

Ordered numerator state within each split group

## t_dc67992c9be8d565 | math.meaning | 41 chars | 1 declarations

tcg_d256_segmented_cp_wide:158 design.math_reshape.rewrite[4].symbol.DenPart.meaning

Split-local denominator for each key half

## t_dce5e31e5362b517 | math.meaning | 48 chars | 1 declarations

tc_split_segmented_wide:284 design.math_reshape.rewrite[7].symbol.CombineBt.meaning

Per-group denominator capacity and channel width

## t_dcf7d147fce8b3e7 | math.meaning | 35 chars | 1 declarations

tcg_d128_fscan_cp_direct_wide:205 design.math_reshape.rewrite[5].symbol.GroupB.meaning

Local 32-entry denominator partials

## t_de38b9d397b6966c | math.meaning | 50 chars | 6 declarations

fused_d128_cp:77 design.math_reshape.rewrite[1].symbol.Score.meaning; fused_d128_group:77 design.math_reshape.rewrite[1].symbol.Score.meaning; fused_d128_key:78 design.math_reshape.rewrite[1].symbol.Score.meaning; fused_d256_cp:73 design.math_reshape.rewrite[1].symbol.Score.meaning; fused_d256_tma:73 design.math_reshape.rewrite[1].symbol.Score.meaning; root:287 design.math_reshape.rewrite[1].symbol.Score.meaning

Masked unscaled scores for the whole current group

## t_de706642723fdfa5 | math.meaning | 37 chars | 1 declarations

tcg_d128_flat_tma_xpose_wide:201 design.math_reshape.rewrite[5].symbol.MergeB.meaning

Denominator sums of 32-entry segments

## t_e2dd2fb12a15f097 | math.meaning | 39 chars | 1 declarations

tcg_d256_segmented_cp_wide:222 design.math_reshape.rewrite[6].symbol.GroupNum.meaning

Numerator sums of the combine intervals

## t_e344391ee433c551 | math.meaning | 24 chars | 20 declarations

tcg_d128_flat_cp_direct_narrow:184 design.math_reshape.rewrite[5].symbol.TotalB.meaning; tcg_d128_flat_cp_direct_wide:196 design.math_reshape.rewrite[5].symbol.TotalB.meaning; tcg_d128_flat_cp_xpose_narrow:191 design.math_reshape.rewrite[5].symbol.TotalB.meaning; tcg_d128_flat_cp_xpose_wide:174 design.math_reshape.rewrite[5].symbol.TotalB.meaning; tcg_d128_flat_tma_direct_narrow:185 design.math_reshape.rewrite[5].symbol.TotalB.meaning; tcg_d128_flat_tma_direct_wide:196 design.math_reshape.rewrite[5].symbol.TotalB.meaning; tcg_d128_flat_tma_xpose_narrow:188 design.math_reshape.rewrite[5].symbol.TotalB.meaning; tcg_d128_flat_tma_xpose_wide:203 design.math_reshape.rewrite[5].symbol.TotalB.meaning; tcg_d128_fscan_cp_direct_narrow:193 design.math_reshape.rewrite[5].symbol.TotalB.meaning; tcg_d128_fscan_cp_direct_wide:208 design.math_reshape.rewrite[5].symbol.TotalB.meaning; tcg_d128_fscan_tma_direct_narrow:193 design.math_reshape.rewrite[5].symbol.TotalB.meaning; tcg_d128_fscan_tma_direct_wide:210 design.math_reshape.rewrite[5].symbol.TotalB.meaning; tcg_d128_fscan_tma_xpose_narrow:197 design.math_reshape.rewrite[5].symbol.TotalB.meaning; tcg_d128_segmented_cp_direct_narrow:206 design.math_reshape.rewrite[6].symbol.TotalB.meaning; tcg_d128_segmented_cp_direct_wide:219 design.math_reshape.rewrite[6].symbol.TotalB.meaning; tcg_d128_segmented_cp_xpose_narrow:196 design.math_reshape.rewrite[6].symbol.TotalB.meaning; tcg_d128_segmented_tma_direct_narrow:206 design.math_reshape.rewrite[6].symbol.TotalB.meaning; tcg_d128_segmented_tma_direct_wide:221 design.math_reshape.rewrite[6].symbol.TotalB.meaning; tcg_d128_segmented_tma_xpose_narrow:212 design.math_reshape.rewrite[6].symbol.TotalB.meaning; tcg_d128_segmented_tma_xpose_wide:196 design.math_reshape.rewrite[6].symbol.TotalB.meaning

Final merged denominator

## t_e594374dd00392ac | math.meaning | 35 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:123 design.math_reshape.rewrite[3].symbol.ScanState.meaning

Running delayed maximum for one row

## t_e5b3f0097a548627 | math.meaning | 53 chars | 1 declarations

tcg_d128_fscan_tma_direct_wide:207 design.math_reshape.rewrite[5].symbol.GroupB.meaning

Denominator tree results retained for the final merge

## t_e5c011ce59637e06 | math.meaning | 45 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:226 design.math_reshape.rewrite[6].symbol.MergeA.meaning

Numerator sums for contiguous split intervals

## t_e656882c88462cc3 | math.meaning | 44 chars | 20 declarations

tcg_d128_flat_cp_direct_narrow:156 design.math_reshape.rewrite[4].symbol.NumState.meaning; tcg_d128_flat_cp_direct_wide:156 design.math_reshape.rewrite[4].symbol.NumState.meaning; tcg_d128_flat_cp_xpose_narrow:162 design.math_reshape.rewrite[4].symbol.NumState.meaning; tcg_d128_flat_cp_xpose_wide:145 design.math_reshape.rewrite[4].symbol.NumState.meaning; tcg_d128_flat_tma_direct_narrow:157 design.math_reshape.rewrite[4].symbol.NumState.meaning; tcg_d128_flat_tma_direct_wide:157 design.math_reshape.rewrite[4].symbol.NumState.meaning; tcg_d128_flat_tma_xpose_narrow:159 design.math_reshape.rewrite[4].symbol.NumState.meaning; tcg_d128_flat_tma_xpose_wide:161 design.math_reshape.rewrite[4].symbol.NumState.meaning; tcg_d128_fscan_cp_direct_narrow:156 design.math_reshape.rewrite[4].symbol.NumState.meaning; tcg_d128_fscan_cp_direct_wide:156 design.math_reshape.rewrite[4].symbol.NumState.meaning; tcg_d128_fscan_tma_direct_narrow:156 design.math_reshape.rewrite[4].symbol.NumState.meaning; tcg_d128_fscan_tma_direct_wide:156 design.math_reshape.rewrite[4].symbol.NumState.meaning; tcg_d128_fscan_tma_xpose_narrow:164 design.math_reshape.rewrite[4].symbol.NumState.meaning; tcg_d128_segmented_cp_direct_narrow:178 design.math_reshape.rewrite[5].symbol.NumState.meaning; tcg_d128_segmented_cp_direct_wide:177 design.math_reshape.rewrite[5].symbol.NumState.meaning; tcg_d128_segmented_cp_xpose_narrow:167 design.math_reshape.rewrite[5].symbol.NumState.meaning; tcg_d128_segmented_tma_direct_narrow:178 design.math_reshape.rewrite[5].symbol.NumState.meaning; tcg_d128_segmented_tma_direct_wide:178 design.math_reshape.rewrite[5].symbol.NumState.meaning; tcg_d128_segmented_tma_xpose_narrow:183 design.math_reshape.rewrite[5].symbol.NumState.meaning; tcg_d128_segmented_tma_xpose_wide:167 design.math_reshape.rewrite[5].symbol.NumState.meaning

Split-local numerator carried between groups

## t_e68eea18fb80263c | math.meaning | 46 chars | 5 declarations

tc_fused:83 design.math_reshape.rewrite[1].symbol.Score.meaning; tc_split_flat_narrow:94 design.math_reshape.rewrite[1].symbol.Score.meaning; tc_split_flat_wide:94 design.math_reshape.rewrite[1].symbol.Score.meaning; tc_split_segmented_narrow:95 design.math_reshape.rewrite[1].symbol.Score.meaning; tc_split_segmented_wide:95 design.math_reshape.rewrite[1].symbol.Score.meaning

Scaled and masked scores for the current group

## t_e6b3433fa6ebe605 | math.meaning | 38 chars | 4 declarations

tcg_d128_fscan_cp_xpose_narrow:164 design.math_reshape.rewrite[4].symbol.NumState.meaning; tcg_d128_fscan_cp_xpose_wide:164 design.math_reshape.rewrite[4].symbol.NumState.meaning; tcg_d128_fscan_tma_xpose_wide:164 design.math_reshape.rewrite[4].symbol.NumState.meaning; tcg_d128_segmented_cp_xpose_wide:187 design.math_reshape.rewrite[5].symbol.NumState.meaning

Local numerator carried between groups

## t_ec1e57e1c480421f | math.meaning | 47 chars | 4 declarations

tcg_d128_fscan_cp_direct_narrow:160 design.math_reshape.rewrite[4].symbol.PartA.meaning; tcg_d128_fscan_cp_direct_wide:160 design.math_reshape.rewrite[4].symbol.PartA.meaning; tcg_d128_fscan_tma_direct_narrow:160 design.math_reshape.rewrite[4].symbol.PartA.meaning; tcg_d128_fscan_tma_direct_wide:160 design.math_reshape.rewrite[4].symbol.PartA.meaning

Unweighted split numerators consumed by combine

## t_ed4bed564d2d2674 | math.meaning | 55 chars | 4 declarations

tcg_d128_fscan_cp_xpose_narrow:166 design.math_reshape.rewrite[4].symbol.GroupPV.meaning; tcg_d128_fscan_cp_xpose_wide:166 design.math_reshape.rewrite[4].symbol.GroupPV.meaning; tcg_d128_fscan_tma_xpose_wide:166 design.math_reshape.rewrite[4].symbol.GroupPV.meaning; tcg_d128_segmented_cp_xpose_wide:189 design.math_reshape.rewrite[5].symbol.GroupPV.meaning

Fresh native PV result before addition to the numerator

## t_ef6b0f92e462f9d4 | math.meaning | 55 chars | 1 declarations

tcg_d128_fscan_tma_xpose_wide:205 design.math_reshape.rewrite[5].symbol.Cg.meaning

Number of split intervals merged locally per output row

## t_f0acb9989dbf5b9b | math.meaning | 48 chars | 3 declarations

tcg_d256_segmented_cp_narrow:85 design.math_reshape.rewrite[2].symbol.SegChunk.meaning; tcg_d256_segmented_tma_narrow:100 design.math_reshape.rewrite[2].symbol.SegChunk.meaning; tcg_d256_segmented_tma_wide:85 design.math_reshape.rewrite[2].symbol.SegChunk.meaning

Segment maxima tested against one incoming state

## t_f0faada7163f102b | math.meaning | 36 chars | 15 declarations

tcg_d128_flat_cp_direct_narrow:130 design.math_reshape.rewrite[3].symbol.MaxState.meaning; tcg_d128_flat_cp_direct_wide:130 design.math_reshape.rewrite[3].symbol.MaxState.meaning; tcg_d128_flat_cp_xpose_narrow:132 design.math_reshape.rewrite[3].symbol.MaxState.meaning; tcg_d128_flat_cp_xpose_wide:115 design.math_reshape.rewrite[3].symbol.MaxState.meaning; tcg_d128_flat_tma_direct_narrow:131 design.math_reshape.rewrite[3].symbol.MaxState.meaning; tcg_d128_flat_tma_direct_wide:131 design.math_reshape.rewrite[3].symbol.MaxState.meaning; tcg_d128_flat_tma_xpose_narrow:132 design.math_reshape.rewrite[3].symbol.MaxState.meaning; tcg_d128_flat_tma_xpose_wide:131 design.math_reshape.rewrite[3].symbol.MaxState.meaning; tcg_d128_segmented_cp_direct_narrow:152 design.math_reshape.rewrite[4].symbol.MaxState.meaning; tcg_d128_segmented_cp_direct_wide:151 design.math_reshape.rewrite[4].symbol.MaxState.meaning; tcg_d128_segmented_cp_xpose_narrow:137 design.math_reshape.rewrite[4].symbol.MaxState.meaning; tcg_d128_segmented_tma_direct_narrow:152 design.math_reshape.rewrite[4].symbol.MaxState.meaning; tcg_d128_segmented_tma_direct_wide:152 design.math_reshape.rewrite[4].symbol.MaxState.meaning; tcg_d128_segmented_tma_xpose_narrow:153 design.math_reshape.rewrite[4].symbol.MaxState.meaning; tcg_d128_segmented_tma_xpose_wide:137 design.math_reshape.rewrite[4].symbol.MaxState.meaning

Replayed maximum in a nonempty split

## t_f1c9bf85423efe74 | math.meaning | 52 chars | 4 declarations

tcg_d128_fscan_cp_xpose_narrow:106 design.math_reshape.rewrite[2].symbol.MTail.meaning; tcg_d128_fscan_cp_xpose_wide:106 design.math_reshape.rewrite[2].symbol.MTail.meaning; tcg_d128_fscan_tma_xpose_narrow:108 design.math_reshape.rewrite[2].symbol.MTail.meaning; tcg_d128_fscan_tma_xpose_wide:106 design.math_reshape.rewrite[2].symbol.MTail.meaning

Last split's delayed maximum used by combine and LSE

## t_f3fff4e4625ef4d3 | math.meaning | 44 chars | 1 declarations

tc_split_segmented_wide:291 design.math_reshape.rewrite[7].symbol.GroupB.meaning

Retained sums of consecutive 32-split blocks

## t_f55a7525cb4995ba | math.meaning | 51 chars | 4 declarations

tcg_d128_fscan_cp_direct_narrow:81 design.math_reshape.rewrite[1].symbol.GroupMax.meaning; tcg_d128_fscan_cp_direct_wide:81 design.math_reshape.rewrite[1].symbol.GroupMax.meaning; tcg_d128_fscan_tma_direct_narrow:81 design.math_reshape.rewrite[1].symbol.GroupMax.meaning; tcg_d128_fscan_tma_direct_wide:81 design.math_reshape.rewrite[1].symbol.GroupMax.meaning

Complete group-major maxima consumed by local scans

## t_f82ef3a2c07142db | math.meaning | 51 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:193 design.math_reshape.rewrite[5].symbol.PartB.meaning

Weighted denominator for every split and output row

## t_fb047a77ba70a0f1 | math.meaning | 32 chars | 4 declarations

tc_split_flat_narrow:96 design.math_reshape.rewrite[1].symbol.GroupMax.meaning; tc_split_flat_wide:96 design.math_reshape.rewrite[1].symbol.GroupMax.meaning; tc_split_segmented_narrow:97 design.math_reshape.rewrite[1].symbol.GroupMax.meaning; tc_split_segmented_wide:97 design.math_reshape.rewrite[1].symbol.GroupMax.meaning

Current full-group score maximum

## t_fb925501bbf7c4ab | math.meaning | 33 chars | 20 declarations

tcg_d128_flat_cp_direct_narrow:132 design.math_reshape.rewrite[3].symbol.DenState.meaning; tcg_d128_flat_cp_direct_wide:132 design.math_reshape.rewrite[3].symbol.DenState.meaning; tcg_d128_flat_cp_xpose_narrow:134 design.math_reshape.rewrite[3].symbol.DenState.meaning; tcg_d128_flat_cp_xpose_wide:117 design.math_reshape.rewrite[3].symbol.DenState.meaning; tcg_d128_flat_tma_direct_narrow:133 design.math_reshape.rewrite[3].symbol.DenState.meaning; tcg_d128_flat_tma_direct_wide:133 design.math_reshape.rewrite[3].symbol.DenState.meaning; tcg_d128_flat_tma_xpose_narrow:134 design.math_reshape.rewrite[3].symbol.DenState.meaning; tcg_d128_flat_tma_xpose_wide:133 design.math_reshape.rewrite[3].symbol.DenState.meaning; tcg_d128_fscan_cp_direct_narrow:131 design.math_reshape.rewrite[3].symbol.DenState.meaning; tcg_d128_fscan_cp_direct_wide:131 design.math_reshape.rewrite[3].symbol.DenState.meaning; tcg_d128_fscan_tma_direct_narrow:131 design.math_reshape.rewrite[3].symbol.DenState.meaning; tcg_d128_fscan_tma_direct_wide:131 design.math_reshape.rewrite[3].symbol.DenState.meaning; tcg_d128_fscan_tma_xpose_narrow:137 design.math_reshape.rewrite[3].symbol.DenState.meaning; tcg_d128_segmented_cp_direct_narrow:154 design.math_reshape.rewrite[4].symbol.DenState.meaning; tcg_d128_segmented_cp_direct_wide:153 design.math_reshape.rewrite[4].symbol.DenState.meaning; tcg_d128_segmented_cp_xpose_narrow:139 design.math_reshape.rewrite[4].symbol.DenState.meaning; tcg_d128_segmented_tma_direct_narrow:154 design.math_reshape.rewrite[4].symbol.DenState.meaning; tcg_d128_segmented_tma_direct_wide:154 design.math_reshape.rewrite[4].symbol.DenState.meaning; tcg_d128_segmented_tma_xpose_narrow:155 design.math_reshape.rewrite[4].symbol.DenState.meaning; tcg_d128_segmented_tma_xpose_wide:139 design.math_reshape.rewrite[4].symbol.DenState.meaning

Split-local unrounded denominator

## t_fea266b223206a00 | math.meaning | 45 chars | 20 declarations

tcg_d128_flat_cp_direct_narrow:134 design.math_reshape.rewrite[3].symbol.Prob8.meaning; tcg_d128_flat_cp_direct_wide:134 design.math_reshape.rewrite[3].symbol.Prob8.meaning; tcg_d128_flat_cp_xpose_narrow:136 design.math_reshape.rewrite[3].symbol.Prob8.meaning; tcg_d128_flat_cp_xpose_wide:119 design.math_reshape.rewrite[3].symbol.Prob8.meaning; tcg_d128_flat_tma_direct_narrow:135 design.math_reshape.rewrite[3].symbol.Prob8.meaning; tcg_d128_flat_tma_direct_wide:135 design.math_reshape.rewrite[3].symbol.Prob8.meaning; tcg_d128_flat_tma_xpose_narrow:136 design.math_reshape.rewrite[3].symbol.Prob8.meaning; tcg_d128_flat_tma_xpose_wide:135 design.math_reshape.rewrite[3].symbol.Prob8.meaning; tcg_d128_fscan_cp_direct_narrow:133 design.math_reshape.rewrite[3].symbol.Prob8.meaning; tcg_d128_fscan_cp_direct_wide:133 design.math_reshape.rewrite[3].symbol.Prob8.meaning; tcg_d128_fscan_tma_direct_narrow:133 design.math_reshape.rewrite[3].symbol.Prob8.meaning; tcg_d128_fscan_tma_direct_wide:133 design.math_reshape.rewrite[3].symbol.Prob8.meaning; tcg_d128_fscan_tma_xpose_narrow:139 design.math_reshape.rewrite[3].symbol.Prob8.meaning; tcg_d128_segmented_cp_direct_narrow:156 design.math_reshape.rewrite[4].symbol.Prob8.meaning; tcg_d128_segmented_cp_direct_wide:155 design.math_reshape.rewrite[4].symbol.Prob8.meaning; tcg_d128_segmented_cp_xpose_narrow:141 design.math_reshape.rewrite[4].symbol.Prob8.meaning; tcg_d128_segmented_tma_direct_narrow:156 design.math_reshape.rewrite[4].symbol.Prob8.meaning; tcg_d128_segmented_tma_direct_wide:156 design.math_reshape.rewrite[4].symbol.Prob8.meaning; tcg_d128_segmented_tma_xpose_narrow:157 design.math_reshape.rewrite[4].symbol.Prob8.meaning; tcg_d128_segmented_tma_xpose_wide:141 design.math_reshape.rewrite[4].symbol.Prob8.meaning

Generated probability rows for the current PV

## t_ff6f48d4a24c702e | math.meaning | 37 chars | 1 declarations

tcg_d128_fscan_tma_xpose_wide:207 design.math_reshape.rewrite[5].symbol.GroupA.meaning

Numerator sum for each merge interval

## t_02b7b3c7282a6fff | math.name | 36 chars | 8 declarations

tcg_d128_flat_cp_direct_narrow:88 design.math_reshape.rewrite[2].name; tcg_d128_flat_cp_direct_wide:88 design.math_reshape.rewrite[2].name; tcg_d128_flat_cp_xpose_narrow:90 design.math_reshape.rewrite[2].name; tcg_d128_flat_cp_xpose_wide:73 design.math_reshape.rewrite[2].name; tcg_d128_flat_tma_direct_narrow:89 design.math_reshape.rewrite[2].name; tcg_d128_flat_tma_direct_wide:89 design.math_reshape.rewrite[2].name; tcg_d128_flat_tma_xpose_narrow:90 design.math_reshape.rewrite[2].name; tcg_d128_flat_tma_xpose_wide:89 design.math_reshape.rewrite[2].name

Ordered delayed-normalizer snapshots

## t_0afc0a281aa05901 | math.name | 38 chars | 1 declarations

tcg_d128_fscan_cp_xpose_narrow:177 design.math_reshape.rewrite[5].name

Narrow ordered merge and caller writes

## t_147e7e75101c4bb2 | math.name | 46 chars | 4 declarations

tc_split_flat_narrow:142 design.math_reshape.rewrite[3].name; tc_split_flat_wide:142 design.math_reshape.rewrite[3].name; tc_split_segmented_narrow:177 design.math_reshape.rewrite[5].name; tc_split_segmented_wide:177 design.math_reshape.rewrite[5].name

Replay maxima into independent value intervals

## t_1d7742430dee4f4f | math.name | 46 chars | 2 declarations

tc_split_segmented_narrow:103 design.math_reshape.rewrite[2].name; tc_split_segmented_wide:103 design.math_reshape.rewrite[2].name

Partially written maxima and temporary outputs

## t_1f088755656d2c2f | math.name | 50 chars | 6 declarations

fused_d128_cp:131 design.math_reshape.rewrite[3].name; fused_d128_group:130 design.math_reshape.rewrite[3].name; fused_d128_key:133 design.math_reshape.rewrite[3].name; fused_d256_cp:126 design.math_reshape.rewrite[3].name; fused_d256_tma:126 design.math_reshape.rewrite[3].name; root:341 design.math_reshape.rewrite[3].name

Carry the native PV accumulator through all groups

## t_1fce82a1462c41ba | math.name | 36 chars | 1 declarations

tc_split_segmented_wide:253 design.math_reshape.rewrite[7].name

Grouped split sums and final outputs

## t_274dbbbacaeddd76 | math.name | 36 chars | 4 declarations

tcg_d128_flat_cp_direct_wide:170 design.math_reshape.rewrite[5].name; tcg_d128_flat_tma_direct_wide:171 design.math_reshape.rewrite[5].name; tcg_d128_flat_tma_xpose_wide:175 design.math_reshape.rewrite[5].name; tcg_d128_segmented_cp_direct_wide:191 design.math_reshape.rewrite[6].name

Grouped FP32 merge and caller writes

## t_2b6477b9fa37e897 | math.name | 39 chars | 2 declarations

tc_split_segmented_narrow:133 design.math_reshape.rewrite[3].name; tc_split_segmented_wide:133 design.math_reshape.rewrite[3].name

Materialize consecutive-group summaries

## t_3279f2e909b21ca2 | math.name | 38 chars | 13 declarations

tcg_d128_flat_cp_direct_narrow:170 design.math_reshape.rewrite[5].name; tcg_d128_flat_cp_xpose_narrow:176 design.math_reshape.rewrite[5].name; tcg_d128_flat_cp_xpose_wide:159 design.math_reshape.rewrite[5].name; tcg_d128_flat_tma_direct_narrow:171 design.math_reshape.rewrite[5].name; tcg_d128_flat_tma_xpose_narrow:173 design.math_reshape.rewrite[5].name; tcg_d128_fscan_cp_direct_narrow:173 design.math_reshape.rewrite[5].name; tcg_d128_fscan_tma_direct_narrow:173 design.math_reshape.rewrite[5].name; tcg_d128_fscan_tma_xpose_narrow:177 design.math_reshape.rewrite[5].name; tcg_d128_segmented_cp_direct_narrow:192 design.math_reshape.rewrite[6].name; tcg_d128_segmented_cp_xpose_narrow:181 design.math_reshape.rewrite[6].name; tcg_d128_segmented_tma_direct_narrow:192 design.math_reshape.rewrite[6].name; tcg_d128_segmented_tma_xpose_narrow:197 design.math_reshape.rewrite[6].name; tcg_d128_segmented_tma_xpose_wide:181 design.math_reshape.rewrite[6].name

Ordered narrow merge and caller writes

## t_3bf9c953a2e376e3 | math.name | 39 chars | 4 declarations

tcg_d128_fscan_cp_xpose_narrow:66 design.math_reshape.rewrite[1].name; tcg_d128_fscan_cp_xpose_wide:66 design.math_reshape.rewrite[1].name; tcg_d128_fscan_tma_xpose_wide:66 design.math_reshape.rewrite[1].name; tcg_d128_segmented_cp_xpose_wide:68 design.math_reshape.rewrite[1].name

Native QK and materialized group maxima

## t_3d16b533c141c0f6 | math.name | 38 chars | 8 declarations

tcg_d256_flat_cp_narrow:36 design.math_reshape.rewrite[0].name; tcg_d256_flat_cp_wide:37 design.math_reshape.rewrite[0].name; tcg_d256_flat_tma_narrow:36 design.math_reshape.rewrite[0].name; tcg_d256_flat_tma_wide:36 design.math_reshape.rewrite[0].name; tcg_d256_segmented_cp_narrow:21 design.math_reshape.rewrite[0].name; tcg_d256_segmented_cp_wide:38 design.math_reshape.rewrite[0].name; tcg_d256_segmented_tma_narrow:36 design.math_reshape.rewrite[0].name; tcg_d256_segmented_tma_wide:21 design.math_reshape.rewrite[0].name

Grouped rows and useful key partitions

## t_3dc34b4bd6c201b8 | math.name | 41 chars | 5 declarations

tcg_d128_fscan_cp_direct_narrow:140 design.math_reshape.rewrite[4].name; tcg_d128_fscan_cp_direct_wide:140 design.math_reshape.rewrite[4].name; tcg_d128_fscan_tma_direct_narrow:140 design.math_reshape.rewrite[4].name; tcg_d128_fscan_tma_direct_wide:140 design.math_reshape.rewrite[4].name; tcg_d128_fscan_tma_xpose_narrow:146 design.math_reshape.rewrite[4].name

Per-group PV and unweighted FP32 partials

## t_47bce0bf60e9814c | math.name | 31 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:201 design.math_reshape.rewrite[6].name

Grouped merge and caller writes

## t_47d662ae52145a1f | math.name | 52 chars | 1 declarations

fused_d128_key:67 design.math_reshape.rewrite[1].name

Full-group product and maximum across two key slices

## t_4dc7f7c19bae89bd | math.name | 39 chars | 15 declarations

tcg_d128_flat_cp_direct_narrow:141 design.math_reshape.rewrite[4].name; tcg_d128_flat_cp_direct_wide:141 design.math_reshape.rewrite[4].name; tcg_d128_flat_cp_xpose_narrow:143 design.math_reshape.rewrite[4].name; tcg_d128_flat_cp_xpose_wide:126 design.math_reshape.rewrite[4].name; tcg_d128_flat_tma_direct_narrow:142 design.math_reshape.rewrite[4].name; tcg_d128_flat_tma_direct_wide:142 design.math_reshape.rewrite[4].name; tcg_d128_flat_tma_xpose_narrow:143 design.math_reshape.rewrite[4].name; tcg_d128_flat_tma_xpose_wide:142 design.math_reshape.rewrite[4].name; tcg_d128_segmented_cp_direct_narrow:163 design.math_reshape.rewrite[5].name; tcg_d128_segmented_cp_direct_wide:162 design.math_reshape.rewrite[5].name; tcg_d128_segmented_cp_xpose_narrow:148 design.math_reshape.rewrite[5].name; tcg_d128_segmented_tma_direct_narrow:163 design.math_reshape.rewrite[5].name; tcg_d128_segmented_tma_direct_wide:163 design.math_reshape.rewrite[5].name; tcg_d128_segmented_tma_xpose_narrow:164 design.math_reshape.rewrite[5].name; tcg_d128_segmented_tma_xpose_wide:148 design.math_reshape.rewrite[5].name

Per-group PV and FP32 weighted partials

## t_517dfcad6648fe2f | math.name | 40 chars | 5 declarations

tc_fused:32 design.math_reshape.rewrite[0].name; tc_split_flat_narrow:32 design.math_reshape.rewrite[0].name; tc_split_flat_wide:32 design.math_reshape.rewrite[0].name; tc_split_segmented_narrow:33 design.math_reshape.rewrite[0].name; tc_split_segmented_wide:33 design.math_reshape.rewrite[0].name

Grouped query rows and useful key groups

## t_53af4d5076694e04 | math.name | 37 chars | 1 declarations

tc_fused:125 design.math_reshape.rewrite[3].name

BF16 value product and direct outputs

## t_55e15ed752e5a9c6 | math.name | 55 chars | 1 declarations

tcg_d256_segmented_cp_wide:104 design.math_reshape.rewrite[3].name

Fill unchanged prefixes and replay the remaining groups

## t_595ae60e7d325bd0 | math.name | 41 chars | 3 declarations

tcg_d256_segmented_cp_narrow:76 design.math_reshape.rewrite[2].name; tcg_d256_segmented_tma_narrow:91 design.math_reshape.rewrite[2].name; tcg_d256_segmented_tma_wide:76 design.math_reshape.rewrite[2].name

Save maxima of consecutive group segments

## t_59d06b95cb433b2c | math.name | 55 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:165 design.math_reshape.rewrite[5].name

Per-group numerator and FP32 globally weighted partials

## t_59fe715a30ea2519 | math.name | 54 chars | 2 declarations

fused_d256_cp:63 design.math_reshape.rewrite[1].name; fused_d256_tma:63 design.math_reshape.rewrite[1].name

Full-width product and a maximum across two key halves

## t_5a2d926d04e35291 | math.name | 37 chars | 3 declarations

tcg_d128_fscan_cp_xpose_narrow:89 design.math_reshape.rewrite[2].name; tcg_d128_fscan_cp_xpose_wide:89 design.math_reshape.rewrite[2].name; tcg_d128_fscan_tma_xpose_wide:89 design.math_reshape.rewrite[2].name

Split-local ordered prefix recurrence

## t_5ede27e5b0834fd9 | math.name | 20 chars | 3 declarations

tcg_d128_segmented_cp_xpose_narrow:73 design.math_reshape.rewrite[2].name; tcg_d128_segmented_tma_xpose_narrow:89 design.math_reshape.rewrite[2].name; tcg_d128_segmented_tma_xpose_wide:73 design.math_reshape.rewrite[2].name

Saved segment maxima

## t_60cfc35b42b18ef3 | math.name | 17 chars | 4 declarations

tcg_d128_segmented_cp_direct_narrow:89 design.math_reshape.rewrite[2].name; tcg_d128_segmented_cp_direct_wide:88 design.math_reshape.rewrite[2].name; tcg_d128_segmented_tma_direct_narrow:89 design.math_reshape.rewrite[2].name; tcg_d128_segmented_tma_direct_wide:89 design.math_reshape.rewrite[2].name

Segment summaries

## t_60db4c1a0d1ea86f | math.name | 54 chars | 2 declarations

fused_d128_cp:67 design.math_reshape.rewrite[1].name; root:277 design.math_reshape.rewrite[1].name

Two narrow products followed by one full-group maximum

## t_6b82dcc3e83818d1 | math.name | 36 chars | 2 declarations

tcg_d128_fscan_cp_direct_wide:173 design.math_reshape.rewrite[5].name; tcg_d128_segmented_tma_direct_wide:192 design.math_reshape.rewrite[6].name

Grouped wide merge and caller writes

## t_6f8176a76fe797cd | math.name | 36 chars | 3 declarations

tcg_d128_segmented_cp_xpose_narrow:85 design.math_reshape.rewrite[3].name; tcg_d128_segmented_tma_xpose_narrow:101 design.math_reshape.rewrite[3].name; tcg_d128_segmented_tma_xpose_wide:85 design.math_reshape.rewrite[3].name

Segment screening and ordered replay

## t_714dc128cf7fef0a | math.name | 32 chars | 24 declarations

tcg_d128_flat_cp_direct_narrow:33 design.math_reshape.rewrite[0].name; tcg_d128_flat_cp_direct_wide:33 design.math_reshape.rewrite[0].name; tcg_d128_flat_cp_xpose_narrow:35 design.math_reshape.rewrite[0].name; tcg_d128_flat_cp_xpose_wide:18 design.math_reshape.rewrite[0].name; tcg_d128_flat_tma_direct_narrow:34 design.math_reshape.rewrite[0].name; tcg_d128_flat_tma_direct_wide:34 design.math_reshape.rewrite[0].name; tcg_d128_flat_tma_xpose_narrow:35 design.math_reshape.rewrite[0].name; tcg_d128_flat_tma_xpose_wide:34 design.math_reshape.rewrite[0].name; tcg_d128_fscan_cp_direct_narrow:33 design.math_reshape.rewrite[0].name; tcg_d128_fscan_cp_direct_wide:33 design.math_reshape.rewrite[0].name; tcg_d128_fscan_cp_xpose_narrow:33 design.math_reshape.rewrite[0].name; tcg_d128_fscan_cp_xpose_wide:33 design.math_reshape.rewrite[0].name; tcg_d128_fscan_tma_direct_narrow:33 design.math_reshape.rewrite[0].name; tcg_d128_fscan_tma_direct_wide:33 design.math_reshape.rewrite[0].name; tcg_d128_fscan_tma_xpose_narrow:34 design.math_reshape.rewrite[0].name; tcg_d128_fscan_tma_xpose_wide:33 design.math_reshape.rewrite[0].name; tcg_d128_segmented_cp_direct_narrow:34 design.math_reshape.rewrite[0].name; tcg_d128_segmented_cp_direct_wide:33 design.math_reshape.rewrite[0].name; tcg_d128_segmented_cp_xpose_narrow:18 design.math_reshape.rewrite[0].name; tcg_d128_segmented_cp_xpose_wide:35 design.math_reshape.rewrite[0].name; tcg_d128_segmented_tma_direct_narrow:34 design.math_reshape.rewrite[0].name; tcg_d128_segmented_tma_direct_wide:34 design.math_reshape.rewrite[0].name; tcg_d128_segmented_tma_xpose_narrow:34 design.math_reshape.rewrite[0].name; tcg_d128_segmented_tma_xpose_wide:18 design.math_reshape.rewrite[0].name

Grouped rows and split intervals

## t_7693d46063f73a1e | math.name | 36 chars | 7 declarations

tcg_d256_flat_cp_narrow:67 design.math_reshape.rewrite[1].name; tcg_d256_flat_cp_wide:68 design.math_reshape.rewrite[1].name; tcg_d256_flat_tma_narrow:67 design.math_reshape.rewrite[1].name; tcg_d256_flat_tma_wide:67 design.math_reshape.rewrite[1].name; tcg_d256_segmented_cp_narrow:52 design.math_reshape.rewrite[1].name; tcg_d256_segmented_tma_narrow:67 design.math_reshape.rewrite[1].name; tcg_d256_segmented_tma_wide:52 design.math_reshape.rewrite[1].name

Native scores and saved group maxima

## t_7bddc0d9c690812c | math.name | 37 chars | 1 declarations

tcg_d256_segmented_cp_wide:129 design.math_reshape.rewrite[4].name

Replay states and round probabilities

## t_8e00363fbff7d88d | math.name | 58 chars | 2 declarations

tc_split_flat_narrow:102 design.math_reshape.rewrite[2].name; tc_split_flat_wide:102 design.math_reshape.rewrite[2].name

Partially written maxima and complete delayed-state replay

## t_8e3682960cb01a81 | math.name | 40 chars | 2 declarations

tcg_d128_fscan_cp_xpose_wide:177 design.math_reshape.rewrite[5].name; tcg_d128_fscan_tma_xpose_wide:177 design.math_reshape.rewrite[5].name

Grouped weighted merge and caller writes

## t_9055fe40702c8260 | math.name | 38 chars | 2 declarations

tc_split_segmented_narrow:146 design.math_reshape.rewrite[4].name; tc_split_segmented_wide:146 design.math_reshape.rewrite[4].name

Constant-prefix fill and suffix replay

## t_92cfdf5be9dce8fd | math.name | 37 chars | 1 declarations

tcg_d256_segmented_cp_wide:171 design.math_reshape.rewrite[5].name

Carry and weight each split numerator

## t_933fa011b512f6eb | math.name | 39 chars | 22 declarations

tcg_d128_flat_cp_direct_narrow:102 design.math_reshape.rewrite[3].name; tcg_d128_flat_cp_direct_wide:102 design.math_reshape.rewrite[3].name; tcg_d128_flat_cp_xpose_narrow:104 design.math_reshape.rewrite[3].name; tcg_d128_flat_cp_xpose_wide:87 design.math_reshape.rewrite[3].name; tcg_d128_flat_tma_direct_narrow:103 design.math_reshape.rewrite[3].name; tcg_d128_flat_tma_direct_wide:103 design.math_reshape.rewrite[3].name; tcg_d128_flat_tma_xpose_narrow:104 design.math_reshape.rewrite[3].name; tcg_d128_flat_tma_xpose_wide:103 design.math_reshape.rewrite[3].name; tcg_d128_segmented_cp_direct_narrow:124 design.math_reshape.rewrite[4].name; tcg_d128_segmented_cp_direct_wide:123 design.math_reshape.rewrite[4].name; tcg_d128_segmented_cp_xpose_narrow:109 design.math_reshape.rewrite[4].name; tcg_d128_segmented_tma_direct_narrow:124 design.math_reshape.rewrite[4].name; tcg_d128_segmented_tma_direct_wide:124 design.math_reshape.rewrite[4].name; tcg_d128_segmented_tma_xpose_narrow:125 design.math_reshape.rewrite[4].name; tcg_d128_segmented_tma_xpose_wide:109 design.math_reshape.rewrite[4].name; tcg_d256_flat_cp_narrow:103 design.math_reshape.rewrite[3].name; tcg_d256_flat_cp_wide:104 design.math_reshape.rewrite[3].name; tcg_d256_flat_tma_narrow:103 design.math_reshape.rewrite[3].name; tcg_d256_flat_tma_wide:103 design.math_reshape.rewrite[3].name; tcg_d256_segmented_cp_narrow:110 design.math_reshape.rewrite[4].name; tcg_d256_segmented_tma_narrow:125 design.math_reshape.rewrite[4].name; tcg_d256_segmented_tma_wide:110 design.math_reshape.rewrite[4].name

Replay the scan and round probabilities

## t_9702ee3d36fb0b12 | math.name | 54 chars | 4 declarations

tcg_d128_fscan_cp_direct_narrow:88 design.math_reshape.rewrite[2].name; tcg_d128_fscan_cp_direct_wide:88 design.math_reshape.rewrite[2].name; tcg_d128_fscan_tma_direct_narrow:88 design.math_reshape.rewrite[2].name; tcg_d128_fscan_tma_direct_wide:88 design.math_reshape.rewrite[2].name

Ordered local prefix and delayed-normalizer recurrence

## t_a5d2288e45086ac1 | math.name | 41 chars | 20 declarations

tcg_d128_flat_cp_direct_narrow:64 design.math_reshape.rewrite[1].name; tcg_d128_flat_cp_direct_wide:64 design.math_reshape.rewrite[1].name; tcg_d128_flat_cp_xpose_narrow:66 design.math_reshape.rewrite[1].name; tcg_d128_flat_cp_xpose_wide:49 design.math_reshape.rewrite[1].name; tcg_d128_flat_tma_direct_narrow:65 design.math_reshape.rewrite[1].name; tcg_d128_flat_tma_direct_wide:65 design.math_reshape.rewrite[1].name; tcg_d128_flat_tma_xpose_narrow:66 design.math_reshape.rewrite[1].name; tcg_d128_flat_tma_xpose_wide:65 design.math_reshape.rewrite[1].name; tcg_d128_fscan_cp_direct_narrow:64 design.math_reshape.rewrite[1].name; tcg_d128_fscan_cp_direct_wide:64 design.math_reshape.rewrite[1].name; tcg_d128_fscan_tma_direct_narrow:64 design.math_reshape.rewrite[1].name; tcg_d128_fscan_tma_direct_wide:64 design.math_reshape.rewrite[1].name; tcg_d128_fscan_tma_xpose_narrow:65 design.math_reshape.rewrite[1].name; tcg_d128_segmented_cp_direct_narrow:65 design.math_reshape.rewrite[1].name; tcg_d128_segmented_cp_direct_wide:64 design.math_reshape.rewrite[1].name; tcg_d128_segmented_cp_xpose_narrow:49 design.math_reshape.rewrite[1].name; tcg_d128_segmented_tma_direct_narrow:65 design.math_reshape.rewrite[1].name; tcg_d128_segmented_tma_direct_wide:65 design.math_reshape.rewrite[1].name; tcg_d128_segmented_tma_xpose_narrow:65 design.math_reshape.rewrite[1].name; tcg_d128_segmented_tma_xpose_wide:49 design.math_reshape.rewrite[1].name

Repeated native QK and saved group maxima

## t_a7994f6590e3a0ba | math.name | 35 chars | 1 declarations

tcg_d128_fscan_tma_xpose_narrow:89 design.math_reshape.rewrite[2].name

Split-local delayed-normalizer scan

## t_a97eba76d1a8553d | math.name | 43 chars | 1 declarations

tcg_d256_segmented_cp_wide:67 design.math_reshape.rewrite[1].name

Native scores and materialized group maxima

## t_b10dfd36577df055 | math.name | 56 chars | 3 declarations

tcg_d256_segmented_cp_narrow:88 design.math_reshape.rewrite[3].name; tcg_d256_segmented_tma_narrow:103 design.math_reshape.rewrite[3].name; tcg_d256_segmented_tma_wide:88 design.math_reshape.rewrite[3].name

Fill unchanged prefixes and replay each remaining suffix

## t_b5453c194684831d | math.name | 42 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:127 design.math_reshape.rewrite[4].name

Replay scan states and round probabilities

## t_b579f02db093581f | math.name | 50 chars | 1 declarations

tcg_d128_fscan_tma_xpose_narrow:111 design.math_reshape.rewrite[3].name

Use locally scanned states and round probabilities

## t_b873a54e179023a8 | math.name | 46 chars | 6 declarations

fused_d128_cp:40 design.math_reshape.rewrite[0].name; fused_d128_group:39 design.math_reshape.rewrite[0].name; fused_d128_key:40 design.math_reshape.rewrite[0].name; fused_d256_cp:36 design.math_reshape.rewrite[0].name; fused_d256_tma:36 design.math_reshape.rewrite[0].name; root:250 design.math_reshape.rewrite[0].name

Grouped rows and the useful key-group interval

## t_b9abb9ee195a97fe | math.name | 27 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:91 design.math_reshape.rewrite[2].name

Materialized segment maxima

## t_b9c5a476d2aba9d3 | math.name | 48 chars | 3 declarations

tcg_d128_fscan_cp_xpose_narrow:142 design.math_reshape.rewrite[4].name; tcg_d128_fscan_cp_xpose_wide:142 design.math_reshape.rewrite[4].name; tcg_d128_fscan_tma_xpose_wide:142 design.math_reshape.rewrite[4].name

Per-group numerator and unweighted FP32 partials

## t_be346ade1528ed30 | math.name | 39 chars | 4 declarations

tcg_d256_flat_cp_narrow:91 design.math_reshape.rewrite[2].name; tcg_d256_flat_cp_wide:92 design.math_reshape.rewrite[2].name; tcg_d256_flat_tma_narrow:91 design.math_reshape.rewrite[2].name; tcg_d256_flat_tma_wide:91 design.math_reshape.rewrite[2].name

Materialize every delayed-maximum state

## t_c99437c020b5476d | math.name | 53 chars | 3 declarations

tcg_d128_fscan_cp_xpose_narrow:108 design.math_reshape.rewrite[3].name; tcg_d128_fscan_cp_xpose_wide:108 design.math_reshape.rewrite[3].name; tcg_d128_fscan_tma_xpose_wide:108 design.math_reshape.rewrite[3].name

Locally scanned normalizers and rounded probabilities

## t_d769d241ec7c5b5f | math.name | 43 chars | 4 declarations

tc_split_flat_narrow:177 design.math_reshape.rewrite[4].name; tc_split_flat_wide:177 design.math_reshape.rewrite[4].name; tc_split_segmented_narrow:212 design.math_reshape.rewrite[6].name; tc_split_segmented_wide:212 design.math_reshape.rewrite[6].name

BF16 PV and globally weighted FP32 partials

## t_dd03a7fa9e71b486 | math.name | 53 chars | 1 declarations

tc_fused:91 design.math_reshape.rewrite[2].name

Delayed maximum and four carried denominator partials

## t_dd315af3aebca30a | math.name | 45 chars | 4 declarations

tcg_d128_fscan_cp_direct_narrow:106 design.math_reshape.rewrite[3].name; tcg_d128_fscan_cp_direct_wide:106 design.math_reshape.rewrite[3].name; tcg_d128_fscan_tma_direct_narrow:106 design.math_reshape.rewrite[3].name; tcg_d128_fscan_tma_direct_wide:106 design.math_reshape.rewrite[3].name

Use local scan states and round probabilities

## t_e29342964d58ac0c | math.name | 37 chars | 1 declarations

tcg_d256_flat_tma_wide:160 design.math_reshape.rewrite[5].name

Grouped combination and caller writes

## t_e2a4a670abd70092 | math.name | 43 chars | 1 declarations

tcg_d128_fscan_tma_direct_wide:173 design.math_reshape.rewrite[5].name

Four-group weighted merge and caller writes

## t_e33acbafbaf75af3 | math.name | 37 chars | 5 declarations

tcg_d256_flat_cp_narrow:160 design.math_reshape.rewrite[5].name; tcg_d256_flat_tma_narrow:160 design.math_reshape.rewrite[5].name; tcg_d256_segmented_cp_narrow:167 design.math_reshape.rewrite[6].name; tcg_d256_segmented_tma_narrow:182 design.math_reshape.rewrite[6].name; tcg_d256_segmented_tma_wide:167 design.math_reshape.rewrite[6].name

Ordered combination and caller writes

## t_e599749778ab631e | math.name | 41 chars | 7 declarations

tcg_d256_flat_cp_narrow:137 design.math_reshape.rewrite[4].name; tcg_d256_flat_cp_wide:138 design.math_reshape.rewrite[4].name; tcg_d256_flat_tma_narrow:137 design.math_reshape.rewrite[4].name; tcg_d256_flat_tma_wide:137 design.math_reshape.rewrite[4].name; tcg_d256_segmented_cp_narrow:144 design.math_reshape.rewrite[5].name; tcg_d256_segmented_tma_narrow:159 design.math_reshape.rewrite[5].name; tcg_d256_segmented_tma_wide:144 design.math_reshape.rewrite[5].name

Carry and weight each partition numerator

## t_e7098113ec86ff8b | math.name | 54 chars | 6 declarations

fused_d128_cp:88 design.math_reshape.rewrite[2].name; fused_d128_group:88 design.math_reshape.rewrite[2].name; fused_d128_key:90 design.math_reshape.rewrite[2].name; fused_d256_cp:85 design.math_reshape.rewrite[2].name; fused_d256_tma:85 design.math_reshape.rewrite[2].name; root:298 design.math_reshape.rewrite[2].name

Delayed state and the actual probability approximation

## t_e79d6e3bcf71de85 | math.name | 42 chars | 1 declarations

tcg_d256_segmented_cp_wide:198 design.math_reshape.rewrite[6].name

Grouped FP32 combination and caller writes

## t_e91ddc60c312e6f6 | math.name | 36 chars | 2 declarations

tc_split_flat_narrow:218 design.math_reshape.rewrite[5].name; tc_split_segmented_narrow:253 design.math_reshape.rewrite[7].name

Ordered split sums and final outputs

## t_e96e9250fceb62b4 | math.name | 42 chars | 1 declarations

tcg_d256_flat_cp_wide:161 design.math_reshape.rewrite[5].name

Grouped wide combination and caller writes

## t_ebb7ddda9c91f1ea | math.name | 36 chars | 1 declarations

tc_split_flat_wide:218 design.math_reshape.rewrite[5].name

Four interval sums and final outputs

## t_f1fbd7f8c3d99224 | math.name | 54 chars | 5 declarations

tc_fused:67 design.math_reshape.rewrite[1].name; tc_split_flat_narrow:78 design.math_reshape.rewrite[1].name; tc_split_flat_wide:78 design.math_reshape.rewrite[1].name; tc_split_segmented_narrow:79 design.math_reshape.rewrite[1].name; tc_split_segmented_wide:79 design.math_reshape.rewrite[1].name

Raw FP8 block products with rank-one scale corrections

## t_f22a653f4371a893 | math.name | 30 chars | 1 declarations

fused_d128_group:66 design.math_reshape.rewrite[1].name

Full-group product and maximum

## t_f7c171fd0cec8e17 | math.name | 51 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:103 design.math_reshape.rewrite[3].name

Segment tests and ordered delayed-normalizer replay

## t_fb0e6efeddadb0f7 | math.name | 48 chars | 1 declarations

tcg_d256_segmented_cp_wide:91 design.math_reshape.rewrite[2].name

Materialize maxima of consecutive group segments

## t_fdb08a846734d160 | math.name | 43 chars | 4 declarations

tcg_d128_segmented_cp_direct_narrow:101 design.math_reshape.rewrite[3].name; tcg_d128_segmented_cp_direct_wide:100 design.math_reshape.rewrite[3].name; tcg_d128_segmented_tma_direct_narrow:101 design.math_reshape.rewrite[3].name; tcg_d128_segmented_tma_direct_wide:101 design.math_reshape.rewrite[3].name

Constant prefixes and ordered suffix replay

## t_09770bd4680cf55c | math.step | 4 chars | 36 declarations

tc_split_flat_narrow:101 design.math_reshape.rewrite[2].step; tc_split_flat_wide:101 design.math_reshape.rewrite[2].step; tc_split_segmented_narrow:145 design.math_reshape.rewrite[4].step; tc_split_segmented_wide:145 design.math_reshape.rewrite[4].step; tcg_d128_flat_cp_direct_narrow:87 design.math_reshape.rewrite[2].step; tcg_d128_flat_cp_direct_wide:87 design.math_reshape.rewrite[2].step; tcg_d128_flat_cp_xpose_narrow:89 design.math_reshape.rewrite[2].step; tcg_d128_flat_cp_xpose_wide:72 design.math_reshape.rewrite[2].step; tcg_d128_flat_tma_direct_narrow:88 design.math_reshape.rewrite[2].step; tcg_d128_flat_tma_direct_wide:88 design.math_reshape.rewrite[2].step; tcg_d128_flat_tma_xpose_narrow:89 design.math_reshape.rewrite[2].step; tcg_d128_flat_tma_xpose_wide:88 design.math_reshape.rewrite[2].step; tcg_d128_fscan_cp_direct_narrow:87 design.math_reshape.rewrite[2].step; tcg_d128_fscan_cp_direct_wide:87 design.math_reshape.rewrite[2].step; tcg_d128_fscan_cp_xpose_narrow:88 design.math_reshape.rewrite[2].step; tcg_d128_fscan_cp_xpose_wide:88 design.math_reshape.rewrite[2].step; tcg_d128_fscan_tma_direct_narrow:87 design.math_reshape.rewrite[2].step; tcg_d128_fscan_tma_direct_wide:87 design.math_reshape.rewrite[2].step; tcg_d128_fscan_tma_xpose_narrow:88 design.math_reshape.rewrite[2].step; tcg_d128_fscan_tma_xpose_wide:88 design.math_reshape.rewrite[2].step; tcg_d128_segmented_cp_direct_narrow:100 design.math_reshape.rewrite[3].step; tcg_d128_segmented_cp_direct_wide:99 design.math_reshape.rewrite[3].step; tcg_d128_segmented_cp_xpose_narrow:84 design.math_reshape.rewrite[3].step; tcg_d128_segmented_cp_xpose_wide:102 design.math_reshape.rewrite[3].step; tcg_d128_segmented_tma_direct_narrow:100 design.math_reshape.rewrite[3].step; tcg_d128_segmented_tma_direct_wide:100 design.math_reshape.rewrite[3].step; tcg_d128_segmented_tma_xpose_narrow:100 design.math_reshape.rewrite[3].step; tcg_d128_segmented_tma_xpose_wide:84 design.math_reshape.rewrite[3].step; tcg_d256_flat_cp_narrow:90 design.math_reshape.rewrite[2].step; tcg_d256_flat_cp_wide:91 design.math_reshape.rewrite[2].step; tcg_d256_flat_tma_narrow:90 design.math_reshape.rewrite[2].step; tcg_d256_flat_tma_wide:90 design.math_reshape.rewrite[2].step; tcg_d256_segmented_cp_narrow:87 design.math_reshape.rewrite[3].step; tcg_d256_segmented_cp_wide:103 design.math_reshape.rewrite[3].step; tcg_d256_segmented_tma_narrow:102 design.math_reshape.rewrite[3].step; tcg_d256_segmented_tma_wide:87 design.math_reshape.rewrite[3].step

scan

## t_1ec4fa86a724efc3 | math.step | 6 chars | 43 declarations

fused_d128_cp:39 design.math_reshape.rewrite[0].step; fused_d128_group:38 design.math_reshape.rewrite[0].step; fused_d128_key:39 design.math_reshape.rewrite[0].step; fused_d256_cp:35 design.math_reshape.rewrite[0].step; fused_d256_tma:35 design.math_reshape.rewrite[0].step; root:249 design.math_reshape.rewrite[0].step; tc_fused:31 design.math_reshape.rewrite[0].step; tc_split_flat_narrow:31 design.math_reshape.rewrite[0].step; tc_split_flat_wide:31 design.math_reshape.rewrite[0].step; tc_split_segmented_narrow:32 design.math_reshape.rewrite[0].step; tc_split_segmented_wide:32 design.math_reshape.rewrite[0].step; tcg_d128_flat_cp_direct_narrow:32 design.math_reshape.rewrite[0].step; tcg_d128_flat_cp_direct_wide:32 design.math_reshape.rewrite[0].step; tcg_d128_flat_cp_xpose_narrow:34 design.math_reshape.rewrite[0].step; tcg_d128_flat_cp_xpose_wide:17 design.math_reshape.rewrite[0].step; tcg_d128_flat_tma_direct_narrow:33 design.math_reshape.rewrite[0].step; tcg_d128_flat_tma_direct_wide:33 design.math_reshape.rewrite[0].step; tcg_d128_flat_tma_xpose_narrow:34 design.math_reshape.rewrite[0].step; tcg_d128_flat_tma_xpose_wide:33 design.math_reshape.rewrite[0].step; tcg_d128_fscan_cp_direct_narrow:32 design.math_reshape.rewrite[0].step; tcg_d128_fscan_cp_direct_wide:32 design.math_reshape.rewrite[0].step; tcg_d128_fscan_cp_xpose_narrow:32 design.math_reshape.rewrite[0].step; tcg_d128_fscan_cp_xpose_wide:32 design.math_reshape.rewrite[0].step; tcg_d128_fscan_tma_direct_narrow:32 design.math_reshape.rewrite[0].step; tcg_d128_fscan_tma_direct_wide:32 design.math_reshape.rewrite[0].step; tcg_d128_fscan_tma_xpose_narrow:33 design.math_reshape.rewrite[0].step; tcg_d128_fscan_tma_xpose_wide:32 design.math_reshape.rewrite[0].step; tcg_d128_segmented_cp_direct_narrow:33 design.math_reshape.rewrite[0].step; tcg_d128_segmented_cp_direct_wide:32 design.math_reshape.rewrite[0].step; tcg_d128_segmented_cp_xpose_narrow:17 design.math_reshape.rewrite[0].step; tcg_d128_segmented_cp_xpose_wide:34 design.math_reshape.rewrite[0].step; tcg_d128_segmented_tma_direct_narrow:33 design.math_reshape.rewrite[0].step; tcg_d128_segmented_tma_direct_wide:33 design.math_reshape.rewrite[0].step; tcg_d128_segmented_tma_xpose_narrow:33 design.math_reshape.rewrite[0].step; tcg_d128_segmented_tma_xpose_wide:17 design.math_reshape.rewrite[0].step; tcg_d256_flat_cp_narrow:35 design.math_reshape.rewrite[0].step; tcg_d256_flat_cp_wide:36 design.math_reshape.rewrite[0].step; tcg_d256_flat_tma_narrow:35 design.math_reshape.rewrite[0].step; tcg_d256_flat_tma_wide:35 design.math_reshape.rewrite[0].step; tcg_d256_segmented_cp_narrow:20 design.math_reshape.rewrite[0].step; tcg_d256_segmented_cp_wide:37 design.math_reshape.rewrite[0].step; tcg_d256_segmented_tma_narrow:35 design.math_reshape.rewrite[0].step; tcg_d256_segmented_tma_wide:20 design.math_reshape.rewrite[0].step

domain

## t_2b8c0a5bbb5aca32 | math.step | 11 chars | 12 declarations

tcg_d128_segmented_cp_direct_narrow:88 design.math_reshape.rewrite[2].step; tcg_d128_segmented_cp_direct_wide:87 design.math_reshape.rewrite[2].step; tcg_d128_segmented_cp_xpose_narrow:72 design.math_reshape.rewrite[2].step; tcg_d128_segmented_cp_xpose_wide:90 design.math_reshape.rewrite[2].step; tcg_d128_segmented_tma_direct_narrow:88 design.math_reshape.rewrite[2].step; tcg_d128_segmented_tma_direct_wide:88 design.math_reshape.rewrite[2].step; tcg_d128_segmented_tma_xpose_narrow:88 design.math_reshape.rewrite[2].step; tcg_d128_segmented_tma_xpose_wide:72 design.math_reshape.rewrite[2].step; tcg_d256_segmented_cp_narrow:75 design.math_reshape.rewrite[2].step; tcg_d256_segmented_cp_wide:90 design.math_reshape.rewrite[2].step; tcg_d256_segmented_tma_narrow:90 design.math_reshape.rewrite[2].step; tcg_d256_segmented_tma_wide:75 design.math_reshape.rewrite[2].step

segment_max

## t_314368b8c3409398 | math.step | 9 chars | 43 declarations

fused_d128_cp:87 design.math_reshape.rewrite[2].step; fused_d128_group:87 design.math_reshape.rewrite[2].step; fused_d128_key:89 design.math_reshape.rewrite[2].step; fused_d256_cp:84 design.math_reshape.rewrite[2].step; fused_d256_tma:84 design.math_reshape.rewrite[2].step; root:297 design.math_reshape.rewrite[2].step; tc_fused:90 design.math_reshape.rewrite[2].step; tc_split_flat_narrow:141 design.math_reshape.rewrite[3].step; tc_split_flat_wide:141 design.math_reshape.rewrite[3].step; tc_split_segmented_narrow:176 design.math_reshape.rewrite[5].step; tc_split_segmented_wide:176 design.math_reshape.rewrite[5].step; tcg_d128_flat_cp_direct_narrow:101 design.math_reshape.rewrite[3].step; tcg_d128_flat_cp_direct_wide:101 design.math_reshape.rewrite[3].step; tcg_d128_flat_cp_xpose_narrow:103 design.math_reshape.rewrite[3].step; tcg_d128_flat_cp_xpose_wide:86 design.math_reshape.rewrite[3].step; tcg_d128_flat_tma_direct_narrow:102 design.math_reshape.rewrite[3].step; tcg_d128_flat_tma_direct_wide:102 design.math_reshape.rewrite[3].step; tcg_d128_flat_tma_xpose_narrow:103 design.math_reshape.rewrite[3].step; tcg_d128_flat_tma_xpose_wide:102 design.math_reshape.rewrite[3].step; tcg_d128_fscan_cp_direct_narrow:105 design.math_reshape.rewrite[3].step; tcg_d128_fscan_cp_direct_wide:105 design.math_reshape.rewrite[3].step; tcg_d128_fscan_cp_xpose_narrow:107 design.math_reshape.rewrite[3].step; tcg_d128_fscan_cp_xpose_wide:107 design.math_reshape.rewrite[3].step; tcg_d128_fscan_tma_direct_narrow:105 design.math_reshape.rewrite[3].step; tcg_d128_fscan_tma_direct_wide:105 design.math_reshape.rewrite[3].step; tcg_d128_fscan_tma_xpose_narrow:110 design.math_reshape.rewrite[3].step; tcg_d128_fscan_tma_xpose_wide:107 design.math_reshape.rewrite[3].step; tcg_d128_segmented_cp_direct_narrow:123 design.math_reshape.rewrite[4].step; tcg_d128_segmented_cp_direct_wide:122 design.math_reshape.rewrite[4].step; tcg_d128_segmented_cp_xpose_narrow:108 design.math_reshape.rewrite[4].step; tcg_d128_segmented_cp_xpose_wide:126 design.math_reshape.rewrite[4].step; tcg_d128_segmented_tma_direct_narrow:123 design.math_reshape.rewrite[4].step; tcg_d128_segmented_tma_direct_wide:123 design.math_reshape.rewrite[4].step; tcg_d128_segmented_tma_xpose_narrow:124 design.math_reshape.rewrite[4].step; tcg_d128_segmented_tma_xpose_wide:108 design.math_reshape.rewrite[4].step; tcg_d256_flat_cp_narrow:102 design.math_reshape.rewrite[3].step; tcg_d256_flat_cp_wide:103 design.math_reshape.rewrite[3].step; tcg_d256_flat_tma_narrow:102 design.math_reshape.rewrite[3].step; tcg_d256_flat_tma_wide:102 design.math_reshape.rewrite[3].step; tcg_d256_segmented_cp_narrow:109 design.math_reshape.rewrite[4].step; tcg_d256_segmented_cp_wide:128 design.math_reshape.rewrite[4].step; tcg_d256_segmented_tma_narrow:124 design.math_reshape.rewrite[4].step; tcg_d256_segmented_tma_wide:109 design.math_reshape.rewrite[4].step

normalize

## t_3c856bc3ae258d94 | math.step | 7 chars | 2 declarations

tc_split_segmented_narrow:132 design.math_reshape.rewrite[3].step; tc_split_segmented_wide:132 design.math_reshape.rewrite[3].step

segment

## t_4097593e7e33b946 | math.step | 12 chars | 7 declarations

fused_d128_cp:130 design.math_reshape.rewrite[3].step; fused_d128_group:129 design.math_reshape.rewrite[3].step; fused_d128_key:132 design.math_reshape.rewrite[3].step; fused_d256_cp:125 design.math_reshape.rewrite[3].step; fused_d256_tma:125 design.math_reshape.rewrite[3].step; root:340 design.math_reshape.rewrite[3].step; tc_fused:124 design.math_reshape.rewrite[3].step

value_output

## t_5c4c7d0f9f3b971f | math.step | 9 chars | 2 declarations

tc_split_segmented_narrow:102 design.math_reshape.rewrite[2].step; tc_split_segmented_wide:102 design.math_reshape.rewrite[2].step

max_table

## t_8527484c0e94fa0d | math.step | 7 chars | 4 declarations

tc_split_flat_narrow:176 design.math_reshape.rewrite[4].step; tc_split_flat_wide:176 design.math_reshape.rewrite[4].step; tc_split_segmented_narrow:211 design.math_reshape.rewrite[6].step; tc_split_segmented_wide:211 design.math_reshape.rewrite[6].step

partial

## t_c680156a1c4cb17d | math.step | 6 chars | 8 declarations

tcg_d256_flat_cp_narrow:159 design.math_reshape.rewrite[5].step; tcg_d256_flat_cp_wide:160 design.math_reshape.rewrite[5].step; tcg_d256_flat_tma_narrow:159 design.math_reshape.rewrite[5].step; tcg_d256_flat_tma_wide:159 design.math_reshape.rewrite[5].step; tcg_d256_segmented_cp_narrow:166 design.math_reshape.rewrite[6].step; tcg_d256_segmented_cp_wide:197 design.math_reshape.rewrite[6].step; tcg_d256_segmented_tma_narrow:181 design.math_reshape.rewrite[6].step; tcg_d256_segmented_tma_wide:166 design.math_reshape.rewrite[6].step

output

## t_d0f00f4d34ae776a | math.step | 6 chars | 43 declarations

fused_d128_cp:66 design.math_reshape.rewrite[1].step; fused_d128_group:65 design.math_reshape.rewrite[1].step; fused_d128_key:66 design.math_reshape.rewrite[1].step; fused_d256_cp:62 design.math_reshape.rewrite[1].step; fused_d256_tma:62 design.math_reshape.rewrite[1].step; root:276 design.math_reshape.rewrite[1].step; tc_fused:66 design.math_reshape.rewrite[1].step; tc_split_flat_narrow:77 design.math_reshape.rewrite[1].step; tc_split_flat_wide:77 design.math_reshape.rewrite[1].step; tc_split_segmented_narrow:78 design.math_reshape.rewrite[1].step; tc_split_segmented_wide:78 design.math_reshape.rewrite[1].step; tcg_d128_flat_cp_direct_narrow:63 design.math_reshape.rewrite[1].step; tcg_d128_flat_cp_direct_wide:63 design.math_reshape.rewrite[1].step; tcg_d128_flat_cp_xpose_narrow:65 design.math_reshape.rewrite[1].step; tcg_d128_flat_cp_xpose_wide:48 design.math_reshape.rewrite[1].step; tcg_d128_flat_tma_direct_narrow:64 design.math_reshape.rewrite[1].step; tcg_d128_flat_tma_direct_wide:64 design.math_reshape.rewrite[1].step; tcg_d128_flat_tma_xpose_narrow:65 design.math_reshape.rewrite[1].step; tcg_d128_flat_tma_xpose_wide:64 design.math_reshape.rewrite[1].step; tcg_d128_fscan_cp_direct_narrow:63 design.math_reshape.rewrite[1].step; tcg_d128_fscan_cp_direct_wide:63 design.math_reshape.rewrite[1].step; tcg_d128_fscan_cp_xpose_narrow:65 design.math_reshape.rewrite[1].step; tcg_d128_fscan_cp_xpose_wide:65 design.math_reshape.rewrite[1].step; tcg_d128_fscan_tma_direct_narrow:63 design.math_reshape.rewrite[1].step; tcg_d128_fscan_tma_direct_wide:63 design.math_reshape.rewrite[1].step; tcg_d128_fscan_tma_xpose_narrow:64 design.math_reshape.rewrite[1].step; tcg_d128_fscan_tma_xpose_wide:65 design.math_reshape.rewrite[1].step; tcg_d128_segmented_cp_direct_narrow:64 design.math_reshape.rewrite[1].step; tcg_d128_segmented_cp_direct_wide:63 design.math_reshape.rewrite[1].step; tcg_d128_segmented_cp_xpose_narrow:48 design.math_reshape.rewrite[1].step; tcg_d128_segmented_cp_xpose_wide:67 design.math_reshape.rewrite[1].step; tcg_d128_segmented_tma_direct_narrow:64 design.math_reshape.rewrite[1].step; tcg_d128_segmented_tma_direct_wide:64 design.math_reshape.rewrite[1].step; tcg_d128_segmented_tma_xpose_narrow:64 design.math_reshape.rewrite[1].step; tcg_d128_segmented_tma_xpose_wide:48 design.math_reshape.rewrite[1].step; tcg_d256_flat_cp_narrow:66 design.math_reshape.rewrite[1].step; tcg_d256_flat_cp_wide:67 design.math_reshape.rewrite[1].step; tcg_d256_flat_tma_narrow:66 design.math_reshape.rewrite[1].step; tcg_d256_flat_tma_wide:66 design.math_reshape.rewrite[1].step; tcg_d256_segmented_cp_narrow:51 design.math_reshape.rewrite[1].step; tcg_d256_segmented_cp_wide:66 design.math_reshape.rewrite[1].step; tcg_d256_segmented_tma_narrow:66 design.math_reshape.rewrite[1].step; tcg_d256_segmented_tma_wide:51 design.math_reshape.rewrite[1].step

scores

## t_deb99bb16abaca20 | math.step | 7 chars | 28 declarations

tc_split_flat_narrow:217 design.math_reshape.rewrite[5].step; tc_split_flat_wide:217 design.math_reshape.rewrite[5].step; tc_split_segmented_narrow:252 design.math_reshape.rewrite[7].step; tc_split_segmented_wide:252 design.math_reshape.rewrite[7].step; tcg_d128_flat_cp_direct_narrow:169 design.math_reshape.rewrite[5].step; tcg_d128_flat_cp_direct_wide:169 design.math_reshape.rewrite[5].step; tcg_d128_flat_cp_xpose_narrow:175 design.math_reshape.rewrite[5].step; tcg_d128_flat_cp_xpose_wide:158 design.math_reshape.rewrite[5].step; tcg_d128_flat_tma_direct_narrow:170 design.math_reshape.rewrite[5].step; tcg_d128_flat_tma_direct_wide:170 design.math_reshape.rewrite[5].step; tcg_d128_flat_tma_xpose_narrow:172 design.math_reshape.rewrite[5].step; tcg_d128_flat_tma_xpose_wide:174 design.math_reshape.rewrite[5].step; tcg_d128_fscan_cp_direct_narrow:172 design.math_reshape.rewrite[5].step; tcg_d128_fscan_cp_direct_wide:172 design.math_reshape.rewrite[5].step; tcg_d128_fscan_cp_xpose_narrow:176 design.math_reshape.rewrite[5].step; tcg_d128_fscan_cp_xpose_wide:176 design.math_reshape.rewrite[5].step; tcg_d128_fscan_tma_direct_narrow:172 design.math_reshape.rewrite[5].step; tcg_d128_fscan_tma_direct_wide:172 design.math_reshape.rewrite[5].step; tcg_d128_fscan_tma_xpose_narrow:176 design.math_reshape.rewrite[5].step; tcg_d128_fscan_tma_xpose_wide:176 design.math_reshape.rewrite[5].step; tcg_d128_segmented_cp_direct_narrow:191 design.math_reshape.rewrite[6].step; tcg_d128_segmented_cp_direct_wide:190 design.math_reshape.rewrite[6].step; tcg_d128_segmented_cp_xpose_narrow:180 design.math_reshape.rewrite[6].step; tcg_d128_segmented_cp_xpose_wide:200 design.math_reshape.rewrite[6].step; tcg_d128_segmented_tma_direct_narrow:191 design.math_reshape.rewrite[6].step; tcg_d128_segmented_tma_direct_wide:191 design.math_reshape.rewrite[6].step; tcg_d128_segmented_tma_xpose_narrow:196 design.math_reshape.rewrite[6].step; tcg_d128_segmented_tma_xpose_wide:180 design.math_reshape.rewrite[6].step

combine

## t_efd7993c8ce15542 | math.step | 8 chars | 32 declarations

tcg_d128_flat_cp_direct_narrow:140 design.math_reshape.rewrite[4].step; tcg_d128_flat_cp_direct_wide:140 design.math_reshape.rewrite[4].step; tcg_d128_flat_cp_xpose_narrow:142 design.math_reshape.rewrite[4].step; tcg_d128_flat_cp_xpose_wide:125 design.math_reshape.rewrite[4].step; tcg_d128_flat_tma_direct_narrow:141 design.math_reshape.rewrite[4].step; tcg_d128_flat_tma_direct_wide:141 design.math_reshape.rewrite[4].step; tcg_d128_flat_tma_xpose_narrow:142 design.math_reshape.rewrite[4].step; tcg_d128_flat_tma_xpose_wide:141 design.math_reshape.rewrite[4].step; tcg_d128_fscan_cp_direct_narrow:139 design.math_reshape.rewrite[4].step; tcg_d128_fscan_cp_direct_wide:139 design.math_reshape.rewrite[4].step; tcg_d128_fscan_cp_xpose_narrow:141 design.math_reshape.rewrite[4].step; tcg_d128_fscan_cp_xpose_wide:141 design.math_reshape.rewrite[4].step; tcg_d128_fscan_tma_direct_narrow:139 design.math_reshape.rewrite[4].step; tcg_d128_fscan_tma_direct_wide:139 design.math_reshape.rewrite[4].step; tcg_d128_fscan_tma_xpose_narrow:145 design.math_reshape.rewrite[4].step; tcg_d128_fscan_tma_xpose_wide:141 design.math_reshape.rewrite[4].step; tcg_d128_segmented_cp_direct_narrow:162 design.math_reshape.rewrite[5].step; tcg_d128_segmented_cp_direct_wide:161 design.math_reshape.rewrite[5].step; tcg_d128_segmented_cp_xpose_narrow:147 design.math_reshape.rewrite[5].step; tcg_d128_segmented_cp_xpose_wide:164 design.math_reshape.rewrite[5].step; tcg_d128_segmented_tma_direct_narrow:162 design.math_reshape.rewrite[5].step; tcg_d128_segmented_tma_direct_wide:162 design.math_reshape.rewrite[5].step; tcg_d128_segmented_tma_xpose_narrow:163 design.math_reshape.rewrite[5].step; tcg_d128_segmented_tma_xpose_wide:147 design.math_reshape.rewrite[5].step; tcg_d256_flat_cp_narrow:136 design.math_reshape.rewrite[4].step; tcg_d256_flat_cp_wide:137 design.math_reshape.rewrite[4].step; tcg_d256_flat_tma_narrow:136 design.math_reshape.rewrite[4].step; tcg_d256_flat_tma_wide:136 design.math_reshape.rewrite[4].step; tcg_d256_segmented_cp_narrow:143 design.math_reshape.rewrite[5].step; tcg_d256_segmented_cp_wide:170 design.math_reshape.rewrite[5].step; tcg_d256_segmented_tma_narrow:158 design.math_reshape.rewrite[5].step; tcg_d256_segmented_tma_wide:143 design.math_reshape.rewrite[5].step

partials

## t_12b782de5f15ddf3 | math.thesis | 347 chars | 1 declarations

fused_d128_group:31 design.math_reshape.thesis

Fuse increasing 128-key groups with a carried numerator and delayed normalizer, avoiding full score/probability materialization. Retain each group's scores and advance one maximum/denominator recurrence in group order. Native MXFP8 products, paired denominator sums and the shortened exponential approximation preserve the source's numerical path.

## t_221bfaa6bbf09c40 | math.thesis | 317 chars | 1 declarations

tc_fused:24 design.math_reshape.thesis

Fuse the useful 128-key groups without global score or probability arrays. Raw FP8 block products receive separate scale corrections; rounded probabilities and BF16 V feed a carried FP32 numerator. Four denominator partials remain separate until output, changing summation order while eliminating split-state traffic.

## t_419cd73ebe6a05e4 | math.thesis | 379 chars | 3 declarations

tcg_d128_fscan_cp_xpose_narrow:28 design.math_reshape.thesis; tcg_d128_fscan_cp_xpose_wide:28 design.math_reshape.thesis; tcg_d128_fscan_tma_xpose_wide:28 design.math_reshape.thesis

Materialize group maxima, repeat their ordered prefix recurrence in each key split, and combine unweighted FP32 partials using saved split-end maxima. This removes global scan snapshots and their separate pass at the cost of repeated prefix work and one additional scalar per split and row; native MXFP8 products and probability rounding retain the fixed source's numerical path.

## t_42a5c84fb9ba1835 | math.thesis | 342 chars | 3 declarations

tcg_d128_segmented_cp_xpose_narrow:13 design.math_reshape.thesis; tcg_d128_segmented_tma_xpose_narrow:29 design.math_reshape.thesis; tcg_d128_segmented_tma_xpose_wide:13 design.math_reshape.thesis

Save group and segment maxima, then skip unchanged prefixes while replaying the ordered delayed-max scan. Independent key-split numerators and FP32 weighted partials shorten numerator recurrences at the cost of a second QK evaluation and global intermediate traffic; segment screening adds scalar work and storage and can reduce replay loads.

## t_575f6e32f95fb1d6 | math.thesis | 425 chars | 1 declarations

fused_d128_key:31 design.math_reshape.thesis

Fuse increasing 128-key groups with a carried numerator and delayed normalizer, avoiding full score/probability materialization. Retain each group's scores, share its maximum across two key slices, and combine their carried denominators only at output. Native MXFP8 products, a shortened exponential approximation and reordered denominator sums retain the source's numerical choices rather than bitwise reference equivalence.

## t_597f12ae4e455902 | math.thesis | 350 chars | 4 declarations

tcg_d128_segmented_cp_direct_narrow:29 design.math_reshape.thesis; tcg_d128_segmented_cp_direct_wide:28 design.math_reshape.thesis; tcg_d128_segmented_tma_direct_narrow:29 design.math_reshape.thesis; tcg_d128_segmented_tma_direct_wide:29 design.math_reshape.thesis

Save group and segment maxima, then skip constant prefixes while replaying the ordered delayed-max scan. Independent key splits shorten numerator recurrences at the cost of a second QK evaluation and global FP32 partials; segment summaries add storage and a pass to avoid some group-max reloads. Native MXFP8 and probability rounding remain explicit.

## t_63b723f2651a9855 | math.thesis | 412 chars | 1 declarations

tcg_d256_segmented_cp_wide:29 design.math_reshape.thesis

Materialize group and segment maxima, use segment tests to skip unchanged prefixes of the delayed-state scan, then recompute QK in independent key partitions and combine FP32 weighted partials. Shorter numerator recurrences and potentially fewer scan reads cost repeated QK and global intermediates; native MXFP8 products and approximate probabilities retain the source's numerical departures from the reference.

## t_73049312c14ac2e0 | math.thesis | 419 chars | 1 declarations

tc_split_segmented_wide:27 design.math_reshape.thesis

Materialize group and segment maxima, fill constant prefixes during delayed-state replay, then recompute QK over independent value intervals and merge weighted FP32 partials in groups. Segment summaries can reduce replay reads; grouping shortens local numerator chains but adds partial materialization and changes rounding. The extra maximum pass, scratch, source maximum-grid defect and temporary output writes remain.

## t_79fb10736ca20857 | math.thesis | 388 chars | 2 declarations

fused_d256_cp:28 design.math_reshape.thesis; fused_d256_tma:28 design.math_reshape.thesis

Fuse increasing 128-key groups with a carried numerator and delayed normalizer, avoiding full score/probability materialization. A full-width score product feeds one group maximum, while two key halves carry separate denominators until output; native MXFP8 products and a shortened exponential approximation retain the source's numerical choices rather than bitwise reference equivalence.

## t_86e76566cefdd060 | math.thesis | 345 chars | 1 declarations

tcg_d128_fscan_tma_direct_wide:28 design.math_reshape.thesis

Save group maxima, derive delayed-normalizer states locally in each split, and merge unweighted FP32 partials using their saved end states in four sum groups. This removes global scan-state materialization at the cost of repeated prefix walks; the second QK evaluation, native MXFP8 contractions and probability-rounding choices remain explicit.

## t_8d2222701e361965 | math.thesis | 349 chars | 1 declarations

tcg_d128_fscan_tma_xpose_narrow:29 design.math_reshape.thesis

Save group maxima, repeat their ordered prefix recurrence within each key split, and combine unweighted FP32 partials using saved split-end maxima. This removes global scan snapshots and a separate scan submission, at the cost of repeated prefixes and one more scalar per split and row; native MXFP8 and probability-rounding choices remain explicit.

## t_a13bea6a28e34048 | math.thesis | 335 chars | 4 declarations

tcg_d256_flat_cp_narrow:28 design.math_reshape.thesis; tcg_d256_flat_cp_wide:29 design.math_reshape.thesis; tcg_d256_flat_tma_narrow:28 design.math_reshape.thesis; tcg_d256_flat_tma_wide:28 design.math_reshape.thesis

Save group maxima and their delayed-state scan, then recompute QK in independent key partitions and combine weighted FP32 partials. Shorter numerator recurrences cost repeated QK and global state traffic; native MXFP8 products and the source probability approximation retain their numerical departures from direct reference evaluation.

## t_a96132cefd1eb2c1 | math.thesis | 326 chars | 3 declarations

tcg_d128_fscan_cp_direct_narrow:28 design.math_reshape.thesis; tcg_d128_fscan_cp_direct_wide:28 design.math_reshape.thesis; tcg_d128_fscan_tma_direct_narrow:28 design.math_reshape.thesis

Save group maxima, derive delayed-normalizer states locally in each split, and merge unweighted FP32 partials using their saved end states. This removes global scan-state materialization at the cost of repeated prefix walks; the second QK evaluation, native MXFP8 contractions and probability-rounding choices remain explicit.

## t_b0895b27c4d4c587 | math.thesis | 342 chars | 3 declarations

tcg_d256_segmented_cp_narrow:13 design.math_reshape.thesis; tcg_d256_segmented_tma_narrow:28 design.math_reshape.thesis; tcg_d256_segmented_tma_wide:13 design.math_reshape.thesis

Save group and segment maxima, fill unchanged scan prefixes, and replay the remaining groups before independent numerator partitions and weighted FP32 combination. Shorter dependent walks cost a segment reduction, repeated QK and global intermediates; native MXFP8 products and the probability approximation retain their numerical departures.

## t_d6afd45a94ea05a2 | math.thesis | 384 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:30 design.math_reshape.thesis

Materialize group and segment maxima, then use segment tests to skip unchanged portions of the ordered delayed-max scan. Independent key splits replay its snapshots and combine FP32 partials, shortening numerator recurrences at the cost of repeated QK products and global scalar/partial traffic; native MXFP8 products and probability rounding retain the fixed source's numerical path.

## t_eb4036b79e44c0bf | math.thesis | 254 chars | 2 declarations

tc_split_flat_narrow:27 design.math_reshape.thesis; tc_split_flat_wide:27 design.math_reshape.thesis

Materialize delayed maxima, recompute QK over independent value intervals, and merge weighted FP32 partials. Shorter PV recurrences cost another QK pass and global scratch; the source maximum-grid coverage defect and temporary output writes are retained.

## t_f36b197cde619a7b | math.thesis | 293 chars | 8 declarations

tcg_d128_flat_cp_direct_narrow:28 design.math_reshape.thesis; tcg_d128_flat_cp_direct_wide:28 design.math_reshape.thesis; tcg_d128_flat_cp_xpose_narrow:30 design.math_reshape.thesis; tcg_d128_flat_cp_xpose_wide:13 design.math_reshape.thesis; tcg_d128_flat_tma_direct_narrow:29 design.math_reshape.thesis; tcg_d128_flat_tma_direct_wide:29 design.math_reshape.thesis; tcg_d128_flat_tma_xpose_narrow:30 design.math_reshape.thesis; tcg_d128_flat_tma_xpose_wide:29 design.math_reshape.thesis

Save group maxima and their ordered delayed-max scan, then compute independent key-split numerators and combine FP32 weighted partials. Shorter numerator recurrences cost a second QK evaluation and global intermediate traffic; the native MXFP8 and probability-rounding choices remain explicit.

## t_f7d4766057d5bcf8 | math.thesis | 391 chars | 2 declarations

fused_d128_cp:32 design.math_reshape.thesis; root:242 design.math_reshape.thesis

Fuse increasing 128-key groups with a carried numerator and delayed normalizer, avoiding full score/probability materialization. Two 64-key score products preserve the full-group maximum while retaining all 128 scores; native MXFP8 products, a shortened exponential approximation and reordered denominator sums retain the source's numerical choices rather than bitwise reference equivalence.

## t_f89e6717133d8590 | math.thesis | 324 chars | 1 declarations

tc_split_segmented_narrow:28 design.math_reshape.thesis

Materialize group and segment maxima, fill constant prefixes during delayed-state replay, then recompute QK over independent value intervals and merge weighted FP32 partials. Segment summaries can reduce replay reads at the cost of another pass and scratch; the source maximum-grid defect and temporary output writes remain.

## t_5595fb0e66d9c3ae | metadata.describe | 134 chars | 43 declarations

fused_d128_cp:3 meta_data.describe; fused_d128_group:3 meta_data.describe; fused_d128_key:3 meta_data.describe; fused_d256_cp:3 meta_data.describe; fused_d256_tma:3 meta_data.describe; root:3 meta_data.describe; tc_fused:3 meta_data.describe; tc_split_flat_narrow:3 meta_data.describe; tc_split_flat_wide:3 meta_data.describe; tc_split_segmented_narrow:3 meta_data.describe; tc_split_segmented_wide:3 meta_data.describe; tcg_d128_flat_cp_direct_narrow:3 meta_data.describe; tcg_d128_flat_cp_direct_wide:3 meta_data.describe; tcg_d128_flat_cp_xpose_narrow:3 meta_data.describe; tcg_d128_flat_cp_xpose_wide:3 meta_data.describe; tcg_d128_flat_tma_direct_narrow:3 meta_data.describe; tcg_d128_flat_tma_direct_wide:3 meta_data.describe; tcg_d128_flat_tma_xpose_narrow:3 meta_data.describe; tcg_d128_flat_tma_xpose_wide:3 meta_data.describe; tcg_d128_fscan_cp_direct_narrow:3 meta_data.describe; tcg_d128_fscan_cp_direct_wide:3 meta_data.describe; tcg_d128_fscan_cp_xpose_narrow:3 meta_data.describe; tcg_d128_fscan_cp_xpose_wide:3 meta_data.describe; tcg_d128_fscan_tma_direct_narrow:3 meta_data.describe; tcg_d128_fscan_tma_direct_wide:3 meta_data.describe; tcg_d128_fscan_tma_xpose_narrow:3 meta_data.describe; tcg_d128_fscan_tma_xpose_wide:3 meta_data.describe; tcg_d128_segmented_cp_direct_narrow:3 meta_data.describe; tcg_d128_segmented_cp_direct_wide:3 meta_data.describe; tcg_d128_segmented_cp_xpose_narrow:3 meta_data.describe; tcg_d128_segmented_cp_xpose_wide:3 meta_data.describe; tcg_d128_segmented_tma_direct_narrow:3 meta_data.describe; tcg_d128_segmented_tma_direct_wide:3 meta_data.describe; tcg_d128_segmented_tma_xpose_narrow:3 meta_data.describe; tcg_d128_segmented_tma_xpose_wide:3 meta_data.describe; tcg_d256_flat_cp_narrow:3 meta_data.describe; tcg_d256_flat_cp_wide:3 meta_data.describe; tcg_d256_flat_tma_narrow:3 meta_data.describe; tcg_d256_flat_tma_wide:3 meta_data.describe; tcg_d256_segmented_cp_narrow:3 meta_data.describe; tcg_d256_segmented_cp_wide:3 meta_data.describe; tcg_d256_segmented_tma_narrow:3 meta_data.describe; tcg_d256_segmented_tma_wide:3 meta_data.describe

Dense grouped-query attention on block-scaled FP8 Q/K/V with FP8 probability rounding, BF16 output, and optional FP32 natural-log LSE.

## t_01c3035c8952d95f | pipeline.thesis | 403 chars | 1 declarations

tcg_d128_segmented_cp_direct_narrow:422 design.tile_pipeline.thesis

Cooperative warps retain Q and stream cp.async payload/scale rings. Copy-group waits and CTA joins publish operands; empty commits preserve lookahead and retire the last payload. Long group-max spans issue QK one group early. Split uses one overlapping score/PV TMEM region and a register numerator, waiting each PV before readback; PDL lets its independent prologue start before the scan data is ready.

## t_01f368ef82882473 | pipeline.thesis | 350 chars | 1 declarations

tcg_d128_flat_tma_direct_wide:381 design.tile_pipeline.thesis

Matrix CTAs retain Q and stream TMA payload/scale rings: long group-max spans issue QK one group early, while split aliases score/PV TMEM and waits each PV before register readback. PDL permits independent split preparation before scan readiness; combine's sixteen warps exchange partials at a CTA barrier before the first four warps merge and write.

## t_027d5e200aed4910 | pipeline.thesis | 352 chars | 1 declarations

tcg_d256_flat_tma_wide:363 design.tile_pipeline.thesis

Two cooperating warpgroups share QK/PV tiles. Group-max is sequential for every span; split overlaps one-ahead TMA and three-ahead atoms with computation but retires PV each group. PDL permits independent setup before data waits; the split epilogue reuses retired operand storage. Four combine groups shorten the split sum and exchange shared partials.

## t_0c639282d98b007b | pipeline.thesis | 496 chars | 1 declarations

tcg_d128_fscan_cp_direct_wide:372 design.tile_pipeline.thesis

Cooperative warps retain Q and fill payload/scale rings through cp.async, with per-thread copy-group waits and CTA publication. Long group-max spans issue QK one group early. Split overlaps its bounded prefix walk with primed K/V transfers, then aliases scores and fresh PV results in TMEM while carrying its numerator in registers. Wide combine publishes weights before use, then shared group partials before the first group writes outputs; PDL separates launch permission from input visibility.

## t_151ce7661dd87c8a | pipeline.thesis | 560 chars | 3 declarations

tcg_d128_segmented_cp_xpose_narrow:410 design.tile_pipeline.thesis; tcg_d128_segmented_tma_xpose_narrow:426 design.tile_pipeline.thesis; tcg_d128_segmented_tma_xpose_wide:410 design.tile_pipeline.thesis

Cooperative warps retain Q and stream TMA payload/scale rings. Long group-max spans issue QK one group early. Split uses one overlapping score/PV TMEM region and a register numerator, waiting each PV before readback; PDL lets its independent prologue start before the scan data is ready. The split epilogue reuses shared operands for cooperative FP32 transpose stores, adding a shared round trip and a post-staging CTA join. Segment screening consumes register batches before replay loads and can reduce dependent global reads without reducing snapshot stores.

## t_1736aad972518864 | pipeline.thesis | 407 chars | 1 declarations

tc_split_flat_wide:459 design.tile_pipeline.thesis

Four cooperative warps per MMA CTA use two raw-input slots and synchronous register-result products; one BF16 V slot is reused after group-end readers join. Scan batches independent loads before its recurrence, and wide combine publishes four groups of partial sums at one CTA barrier before the first group writes outputs. Ordinary MMA launches keep the chain completion-ordered despite helper PDL signals.

## t_246ee9358a7d87d0 | pipeline.thesis | 434 chars | 1 declarations

tcg_d128_fscan_tma_direct_wide:373 design.tile_pipeline.thesis

Cooperative warps retain Q and stream TMA payload/scale rings. Long group-max spans issue QK one group early. Split overlaps its bounded prefix walk with primed K/V transfers, then aliases scores and fresh PV results in TMEM while carrying its numerator in registers. Wide combine publishes weights before group-local sums, then publishes group partials for group zero to finish. PDL separates launch permission from input visibility.

## t_2ff8d2975fba9c5c | pipeline.thesis | 530 chars | 1 declarations

tcg_d128_fscan_cp_xpose_narrow:360 design.tile_pipeline.thesis

Each matrix CTA uses four cooperative warps, a persistent Q tile and cp.async payload/scale rings. Group-max can issue QK one group early; split instead reuses the score TMEM region for each fresh PV result and carries its numerator in registers. At split exit, the Q/K/V shared arena stages FP32 partial numerators for a CTA-wide cooperative write, adding a shared round trip and publication barrier. The split prefix scan can overlap its primed copy groups; weighted combine publishes shared split weights once for all channels.

## t_30ceda274d54e6f7 | pipeline.thesis | 386 chars | 1 declarations

tc_split_flat_narrow:442 design.tile_pipeline.thesis

Four cooperative warps per MMA CTA use two raw-input slots and synchronous register-result products; one BF16 V slot is reused after group-end readers join. Scan batches independent loads before its recurrence, and narrow combine postpones denominator publication until numerator accumulation finishes. Ordinary MMA launches keep the chain completion-ordered despite helper PDL signals.

## t_31b0b683f9013a12 | pipeline.thesis | 528 chars | 1 declarations

tcg_d128_segmented_cp_direct_wide:438 design.tile_pipeline.thesis

Cooperative warps retain Q and stream cp.async payload/scale rings. Copy-group waits and CTA joins publish operands; empty commits preserve lookahead and retire the last payload. Long group-max spans issue QK one group early. Split uses one overlapping score/PV TMEM region and a register numerator, waiting each PV before readback; PDL lets its independent prologue start before the scan data is ready. Wide merge exchanges four groups' numerator and warp-denominator partials in shared memory before group zero writes the row.

## t_384e0625e1166335 | pipeline.thesis | 583 chars | 1 declarations

tcg_d128_fscan_tma_xpose_wide:381 design.tile_pipeline.thesis

Each matrix CTA uses four cooperative warps, a persistent Q tile and a two-slot TMA payload/scale traversal. Split reuses the score TMEM region for each fresh PV result and carries its numerator in registers. At split exit, the Q/K/V shared arena stages FP32 partial numerators for a CTA-wide cooperative write, adding a shared round trip and publication barrier. The split prefix scan can overlap primed TMA transfers. Wide combine publishes weights after its first numerator prefetch, then publishes group numerators and warp denominator sums before the first group writes outputs.

## t_4035c6343a96c1f7 | pipeline.thesis | 325 chars | 3 declarations

tcg_d256_segmented_cp_narrow:387 design.tile_pipeline.thesis; tcg_d256_segmented_tma_narrow:402 design.tile_pipeline.thesis; tcg_d256_segmented_tma_wide:387 design.tile_pipeline.thesis

Two cooperating warpgroups share QK/PV tiles. Group-max is sequential for every span; split overlaps one-ahead TMA and three-ahead atoms with computation but retires PV each group. Segment scans batch register reads; PDL separates launch permission from data visibility, and the split epilogue reuses retired operand storage.

## t_4be142dc97db14cd | pipeline.thesis | 362 chars | 1 declarations

fused_d128_group:237 design.tile_pipeline.thesis

Two independent warpgroups alternate whole groups, each with three K slots, two V slots, one score tile and one probability tile. Early maximum and later denominator handoffs let their readout/softmax work overlap; the shared PV accumulator still advances in group order. The source's additional generic cold prefetch and its ordering limitation remain explicit.

## t_4cce9be67d97eb51 | pipeline.thesis | 427 chars | 1 declarations

tc_split_segmented_wide:551 design.tile_pipeline.thesis

Four cooperative warps per MMA CTA use two raw-input slots and synchronous register-result products. Segment summaries and group maxima are loaded in batches before dependent scalar work. The scan may launch at segmax's entry trigger but waits for completed stores; ordinary MMA launches keep the other boundaries completion-ordered. Wide combine shortens group sums at the cost of shared partials and one sixteen-warp barrier.

## t_51eea540a51a8364 | pipeline.thesis | 343 chars | 2 declarations

tcg_d256_flat_cp_narrow:350 design.tile_pipeline.thesis; tcg_d256_flat_cp_wide:364 design.tile_pipeline.thesis

Two cooperating warpgroups share QK/PV tiles. Group-max is sequential for every span; split overlaps one-ahead payloads and three-ahead atoms with computation but retires PV each group. PDL permits independent setup before data waits; cp.async groups complete payloads and atoms together, and the split epilogue reuses retired operand storage.

## t_6aa313bf34bdcaa2 | pipeline.thesis | 441 chars | 1 declarations

tcg_d128_fscan_cp_direct_narrow:356 design.tile_pipeline.thesis

Cooperative warps retain Q and fill payload/scale rings through cp.async, with per-thread copy-group waits and CTA publication. Long group-max spans issue QK one group early. Split overlaps its bounded prefix walk with primed K/V transfers, then aliases scores and fresh PV results in TMEM while carrying its numerator in registers. Weighted combine publishes shared weights before use; PDL separates launch permission from input visibility.

## t_6d649f05323e8cd7 | pipeline.thesis | 287 chars | 2 declarations

tcg_d128_flat_tma_direct_narrow:367 design.tile_pipeline.thesis; tcg_d128_segmented_tma_direct_narrow:422 design.tile_pipeline.thesis

Cooperative warps retain Q and stream TMA payload/scale rings. Long group-max spans issue QK one group early. Split uses one overlapping score/PV TMEM region and a register numerator, waiting each PV before readback; PDL lets its independent prologue start before the scan data is ready.

## t_6e48ceba22119ef2 | pipeline.thesis | 491 chars | 2 declarations

tcg_d128_flat_cp_xpose_narrow:373 design.tile_pipeline.thesis; tcg_d128_flat_cp_xpose_wide:356 design.tile_pipeline.thesis

Cooperative warps retain Q and stream cp.async payload/scale rings, with per-thread copy-group waits followed by CTA publication. Long group-max spans issue QK one group early. Split uses one overlapping score/PV TMEM region and a register numerator, waiting each PV before readback; PDL lets its independent prologue start before the scan data is ready. The split epilogue reuses shared operands for cooperative FP32 transpose stores, adding a shared round trip and a post-staging CTA join.

## t_7980f9c005c9e28b | pipeline.thesis | 344 chars | 1 declarations

fused_d256_tma:239 design.tile_pipeline.thesis

Two cooperative warpgroups split each row's keys and output channels. MIDPIPE issues QK(g+1) after the score-readout exchange and retires PV(g-1) mid-iteration, allowing matrix execution to overlap probability work. This costs three K slots, two V slots, three ahead-loaded atom slots and 512 TMEM columns, while retaining 64 scores per thread.

## t_7f62a120caae7bbe | pipeline.thesis | 463 chars | 1 declarations

tcg_d128_fscan_tma_xpose_narrow:361 design.tile_pipeline.thesis

Cooperative warps retain Q and stream TMA payload/scale rings. Long group-max spans issue QK one group early. Split uses one overlapping score/PV TMEM region and a register numerator, waiting each PV before readback; PDL lets its independent prologue start before group maxima are ready. The bounded prefix overlaps primed transfers. Shared operand storage then stages FP32 partial numerators, while combine publishes one shared weight per split for all channels.

## t_892be253fc1aa92f | pipeline.thesis | 420 chars | 1 declarations

tcg_d128_segmented_tma_direct_wide:440 design.tile_pipeline.thesis

Cooperative warps retain Q and stream TMA payload/scale rings. Long group-max spans issue QK one group early. Split uses one overlapping score/PV TMEM region and a register numerator, waiting each PV before readback; PDL lets its independent prologue start before the scan data is ready. Wide combine shortens the split-sum chains with four cooperative groups, paying for shared partials and one CTA publication barrier.

## t_8ca678ebc69728e0 | pipeline.thesis | 353 chars | 1 declarations

fused_d128_cp:244 design.tile_pipeline.thesis

One cooperative warpgroup retains Q and streams two K/V slots through a single 64-column score tile. Combined cp.async groups prefetch the next K/V payloads and atoms while the current group computes; each PV retires before the next iteration. Retaining 128 scores per row trades register lifetime for one score readout and a 256-column TMEM allocation.

## t_9ca1fb28bc82d4a7 | pipeline.thesis | 333 chars | 1 declarations

tc_split_segmented_narrow:518 design.tile_pipeline.thesis

Four cooperative warps per MMA CTA use two raw-input slots and synchronous register-result products. Segment summaries and group maxima are loaded in batches before dependent scalar work. The scan may launch at segmax's entry trigger but waits for completed stores; ordinary MMA launches keep the other boundaries completion-ordered.

## t_9f3e30b46c7ab68c | pipeline.thesis | 381 chars | 1 declarations

tcg_d128_flat_cp_direct_narrow:366 design.tile_pipeline.thesis

Cooperative warps retain Q and stream cp.async payload/scale rings, using per-thread copy-group waits and CTA joins while newer copies can remain in flight. Long group-max spans issue QK one group early. Split uses one overlapping score/PV TMEM region and a register numerator, waiting each PV before readback; PDL lets its independent prologue start before the scan data is ready.

## t_aadf290518fa1d1c | pipeline.thesis | 849 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:443 design.tile_pipeline.thesis

Each matrix CTA uses four cooperative warps, a persistent Q tile and cp.async payload/scale rings with per-thread copy-group waits followed by CTA publication. Group-max can issue QK one group early; split instead reuses the score TMEM region for each fresh PV result and carries its numerator in registers. At split exit, the Q/K/V shared arena stages FP32 partial numerators for a CTA-wide cooperative write, adding a shared round trip and publication barrier. Segment reduction, row replay and combine consume global intermediates through PDL waits. Row replay batches segment tests before loading only the required group-max suffix; register batches are consumed before reuse. Wide combine publishes four interval numerators and sixteen denominator tree results through shared memory, then its first thread group writes the row after a CTA join.

## t_acfa335bfca90828 | pipeline.thesis | 377 chars | 1 declarations

root:454 design.tile_pipeline.thesis

One cooperative warpgroup retains Q and streams two K/V slots through a single 64-column score tile. TMA permits PV(g-1) to remain in flight through both QK halves of g; its retirement protects the next probability, numerator rescale, V-scale write and V-slot refill. Retaining 128 scores per row trades register lifetime for one score readout and a 256-column TMEM allocation.

## t_b6b6b7b25ba424a2 | pipeline.thesis | 475 chars | 1 declarations

tcg_d128_flat_cp_direct_wide:380 design.tile_pipeline.thesis

Cooperative warps retain Q and stream cp.async payload/scale rings, using per-thread copy-group waits and CTA joins while newer copies can remain in flight. Long group-max spans issue QK one group early. Split uses one overlapping score/PV TMEM region and a register numerator, waiting each PV before readback; PDL lets its independent prologue start before the scan data is ready. Wide combine publishes group partials through shared memory before group zero writes the row.

## t_c2c10786c99c0e2a | pipeline.thesis | 515 chars | 1 declarations

tcg_d256_segmented_cp_wide:447 design.tile_pipeline.thesis

Both compute passes use two cooperating warpgroups and cp.async copy groups for payloads and scale atoms. Group-max stays sequential for every span; split preloads scales three groups ahead but retires PV each iteration. Segment scans batch register loads and serialize only each row's state. PDL separates launch permission from data visibility; split's cooperative partial write reuses retired operand storage. Four combine groups reduce split intervals into shared sums; a CTA barrier precedes group-zero output.

## t_c8a9c3ac1c14a804 | pipeline.thesis | 280 chars | 1 declarations

tcg_d256_flat_tma_narrow:350 design.tile_pipeline.thesis

Two cooperating warpgroups share QK/PV tiles. Group-max is sequential for every span; split overlaps one-ahead TMA and three-ahead atoms with computation but retires PV each group. PDL permits independent setup before data waits; the split epilogue reuses retired operand storage.

## t_d3995396001c6188 | pipeline.thesis | 406 chars | 1 declarations

fused_d256_cp:239 design.tile_pipeline.thesis

Two cooperative warpgroups split each row's keys and output channels. cp.async prefetches the next K/V group while the current group executes; ahead-loaded atoms let scale publication share the staging barrier. Each group completes QK, probability work and PV in sequence. K uses two of its three allocated slots; V uses two, atoms use three, and TMEM uses 512 columns, while each thread retains 64 scores.

## t_d7d5c57168a9270b | pipeline.thesis | 628 chars | 1 declarations

tcg_d128_fscan_cp_xpose_wide:376 design.tile_pipeline.thesis

Each matrix CTA uses four cooperative warps, a persistent Q tile and cp.async payload/scale rings. Group-max can issue QK one group early; split instead reuses the score TMEM region for each fresh PV result and carries its numerator in registers. At split exit, the Q/K/V shared arena stages FP32 partial numerators for a CTA-wide cooperative write, adding a shared round trip and publication barrier. The split prefix scan can overlap its primed copy groups. Wide combine uses four groups to shorten each split sum, paying for shared numerator/denominator subtotals and a second CTA barrier before group zero writes the output.

## t_d8d83645cc56a3c1 | pipeline.thesis | 309 chars | 1 declarations

fused_d128_key:240 design.tile_pipeline.thesis

Two cooperating warpgroups split each group's keys while one issuer drives full-width QK and PV. Four shared K/V slots, two score tiles and alternating scale slots permit QK(g+1) to overlap probability work for g and defer PV(g) retirement to g+1. This costs 512 TMEM columns and a per-group maximum exchange.

## t_e4b724b5d7244ff6 | pipeline.thesis | 272 chars | 1 declarations

tc_fused:236 design.tile_pipeline.thesis

Four cooperative warps retain Q in shared memory and full score/numerator fragments in registers. Two raw K/V slots overlap next-group cp.async transfers with current work. One decoded BF16 V slot is reused after the group-end CTA barrier joins all synchronous PV readers.

## t_efb4398c59f6c917 | pipeline.thesis | 376 chars | 1 declarations

tcg_d128_fscan_tma_direct_narrow:356 design.tile_pipeline.thesis

Cooperative warps retain Q and stream TMA payload/scale rings. Long group-max spans issue QK one group early. Split overlaps its bounded prefix walk with primed K/V transfers, then aliases scores and fresh PV results in TMEM while carrying its numerator in registers. Weighted combine publishes shared weights before use; PDL separates launch permission from input visibility.

## t_f655447a99bd4c7d | pipeline.thesis | 551 chars | 1 declarations

tcg_d128_flat_tma_xpose_wide:387 design.tile_pipeline.thesis

Cooperative warps retain Q and stream TMA payload/scale rings. Long group-max spans issue QK one group early. Split uses one overlapping score/PV TMEM region and a register numerator, waiting each PV before readback; PDL lets its independent prologue start before the scan data is ready. The split epilogue reuses shared operands for cooperative FP32 transpose stores, adding a shared round trip and a post-staging CTA join. Wide merge publishes four groups of numerator and warp denominator sums at one CTA barrier, then group zero completes the row.

## t_fcffaf223af89f89 | pipeline.thesis | 424 chars | 1 declarations

tcg_d128_flat_tma_xpose_narrow:370 design.tile_pipeline.thesis

Cooperative warps retain Q and stream TMA payload/scale rings. Long group-max spans issue QK one group early. Split uses one overlapping score/PV TMEM region and a register numerator, waiting each PV before readback; PDL lets its independent prologue start before the scan data is ready. The split epilogue reuses shared operands for cooperative FP32 transpose stores, adding a shared round trip and a post-staging CTA join.

## t_00c6334a6fcfe858 | pipeline.verb | 822 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:802 design.tile_pipeline.phase.combine.role.Main.info.verb.partials

Four consecutive 128-thread groups own combine's intervals m=0,...,MergeGroups-1. Each thread owns one channel within its group and initializes IntervalSum to zero. Load eight PartA entries at a time into NumeratorBatch, consume them in order before register reuse, then consume the scalar tail. Store IntervalSum to MergeA. Each thread also loads one PartB entry at its local split offset, substituting zero past sHi(m); each warp reduces its 32 values with the specified tree and its first lane writes MergeB. All groups, including empty intervals, publish their zero or computed partials. The CTA barrier at the end joins all MergeA/MergeB stores; done guarantees their visibility to the first group. There is no fused-scan weight array, weight computation or preceding weight-publication barrier in this instantiation.

## t_0194943fd139ab1d | pipeline.verb | 172 chars | 1 declarations

tcg_d256_segmented_cp_wide:855 design.tile_pipeline.phase.combine.role.Output.info.verb.await

Every role thread performs the predecessor-grid dependency wait at kernel entry, before any partial read. done proves all PartA/PartB writes and transitively Mtail visible.

## t_025cc30be3972c89 | pipeline.verb | 367 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:810 design.tile_pipeline.phase.combine.role.Main.info.verb.merge

The first 128-thread group reads all MergeB entries in combine's order, independently forming TotalB per thread, and each thread sums the MergeA entries for its channel into TotalA. Other groups have completed their shared writes and have no data effects in merge or write. Both shared arrays remain unchanged through all readers; no later local worktile reuses them.

## t_0331d24b1ceba829 | pipeline.verb | 339 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:686 design.tile_pipeline.phase.split.role.Main.info.verb.prime

For Ng>0 enqueue K(cLo), V(cLo) and both scale atoms in slot zero, then each thread commits its payload/atom copies in one group. Payload copies move 16 bytes with zero-fill for keys beyond Skv; scale atoms use unguarded 16-byte copies from allocated padding. For Ng=0 issue neither copies nor a commit. done means submission, not arrival.

## t_0503ee09e59ba653 | pipeline.verb | 416 chars | 1 declarations

tcg_d256_segmented_cp_wide:798 design.tile_pipeline.phase.split.role.Main.info.verb.pv

Thread zero fences after thread-sync and issues four ordered 32-key steps, each with both 128-channel products. The first step of the first group overwrites each numerator half; later products accumulate. Commit PV, then every thread waits for its completion, advances parity and fences. done retires Value, ValueScale and Prob8 reads and NumState updates. There is no deferred PV, next-QK issue or tail CTA barrier.

## t_059294c6571177d2 | pipeline.verb | 292 chars | 1 declarations

tcg_d128_fscan_tma_xpose_wide:442 design.tile_pipeline.phase.group_max.role.Main.info.verb.stage

Wait for current K(g) and its atom, fence shared async access and CTA-sync. Publish KeyScale, wait for its TMEM stores, fence and CTA-sync. TMA parity flips on the two-slot wrap. These barriers also publish Q, and the preceding QK completion protects replacement of the fixed KeyScale region.

## t_05a6945f2040273f | pipeline.verb | 273 chars | 2 declarations

tcg_d128_segmented_cp_direct_narrow:484 design.tile_pipeline.phase.group_max.role.Main.info.verb.prefetch; tcg_d128_segmented_cp_direct_wide:501 design.tile_pipeline.phase.group_max.role.Main.info.verb.prefetch

Enqueue K(g+2) and atom for Long, otherwise K(g+1) and atom, only when that group is <=cHi. Otherwise issue no transfer, but always commit a copy group, including an empty one at the tail. done means enqueue; previous QK completion and atom-reader joins protect ring reuse.

## t_05c68748a1783eb4 | pipeline.verb | 543 chars | 2 declarations

tcg_d128_flat_cp_direct_narrow:432 design.tile_pipeline.phase.group_max.role.Main.info.verb.stage; tcg_d128_flat_cp_direct_wide:446 design.tile_pipeline.phase.group_max.role.Main.info.verb.stage

Each thread waits until all but its newest copy group complete, then fences shared access and CTA-syncs. For Long this proves K(g+1) and atom when g<cHi; write the next physical atom slot into alternating KeyScale. At g=cHi the wait still executes after the empty commit and stale atom bytes are still copied; no later QK consumes them. With Ng>=8 that shared slot was previously filled. For non-Long the wait proves current K(g) and atom; write KeyScale at its fixed base. Wait for TMEM stores, fence and CTA-sync. These joins also publish Q.

## t_068b26f86363752c | pipeline.verb | 532 chars | 1 declarations

tcg_d128_fscan_tma_xpose_wide:430 design.tile_pipeline.phase.group_max.role.Main.info.verb.setup

Establish the scheduled coordinates, row masks and completion parities, initially zero. Only two of the three allocated shared Key and KeyAtom slots are traversed; KeyScale uses a fixed TMEM region. The q0>=Sq guard is unreachable under this grid. Valid rows of split zero/last issue the outside -inf stores. Thread zero initializes and publishes transaction barriers and issues all TMA/MMA; all warps cooperate on Q/scales and synchronization. All four warps read scores, including padded rows; only valid rows write global maxima.

## t_06a1d028357ad6cb | pipeline.verb | 468 chars | 3 declarations

tcg_d128_fscan_cp_xpose_narrow:571 design.tile_pipeline.phase.split.role.Main.info.verb.probability; tcg_d128_fscan_cp_xpose_wide:588 design.tile_pipeline.phase.split.role.Main.info.verb.probability; tcg_d128_fscan_tma_xpose_wide:576 design.tile_pipeline.phase.split.role.Main.info.verb.probability

All warps apply scan's update using CurrentMax, advance CurrentMax to NextMax, rescale DenState and NumState, then consume DotTile in 32-key chunks through ScoreChunk and normalize, including padded rows. Pack every Prob8 chunk into TMEM and wait for its stores. Fence and CTA-sync before PV; this join publishes all probability rows and retires every DotTile reader before PV overwrites the same columns. Padded rows are computed but never written to global partials.

## t_07c488b6e63f9d8a | pipeline.verb | 818 chars | 1 declarations

tcg_d128_flat_tma_xpose_narrow:604 design.tile_pipeline.phase.split.role.Main.info.verb.stage_partials

After preparation writes and all old OperandArena readers and asynchronous fills have finished, nonempty valid rows load MTail and form weighted FP32 numerators; empty splits stage zeros. Each valid row writes D channels into the PartA view of OperandArena. Its 81920-byte operand capacity holds the 67584-byte staging span including row spacing; padded rows are neither staged nor later read. For Ng>0, the loop's current-slot transfer waits and QK/PV completion retire all old arena uses; the last group enqueues no future transfer. For Ng=0 there are no K/V fills or matrix readers. A CTA barrier after staging publishes the staged rows and joins final TMEM readouts; done includes this barrier. Source limitation: the empty-split path lacks a pre-staging join for Query writers and can violate this reuse ordering.

## t_0866d24820e85e73 | pipeline.verb | 358 chars | 5 declarations

tc_fused:316 design.tile_pipeline.phase.main.role.Main.info.verb.prefetch; tc_split_flat_narrow:580 design.tile_pipeline.phase.split.role.Main.info.verb.prefetch; tc_split_flat_wide:597 design.tile_pipeline.phase.split.role.Main.info.verb.prefetch; tc_split_segmented_narrow:687 design.tile_pipeline.phase.split.role.Main.info.verb.prefetch; tc_split_segmented_wide:721 design.tile_pipeline.phase.split.role.Main.info.verb.prefetch

For g<cHi, enqueue next-group K/V payloads and atoms into slot (g+1-cLo)%2. Commit on every iteration, including an empty group at g=cHi. The last action reads/writes no payload but retains issue/done events; done means enqueue/commit only. The first target slot is unused. Later, the previous group-end barrier has joined all readers of its old g-1 version.

## t_0a390cc52b7404a4 | pipeline.verb | 493 chars | 7 declarations

tcg_d256_flat_cp_narrow:603 design.tile_pipeline.phase.split.role.Main.info.verb.advance; tcg_d256_flat_cp_wide:617 design.tile_pipeline.phase.split.role.Main.info.verb.advance; tcg_d256_flat_tma_narrow:598 design.tile_pipeline.phase.split.role.Main.info.verb.advance; tcg_d256_flat_tma_wide:612 design.tile_pipeline.phase.split.role.Main.info.verb.advance; tcg_d256_segmented_cp_narrow:671 design.tile_pipeline.phase.split.role.Main.info.verb.advance; tcg_d256_segmented_tma_narrow:686 design.tile_pipeline.phase.split.role.Main.info.verb.advance; tcg_d256_segmented_tma_wide:671 design.tile_pipeline.phase.split.role.Main.info.verb.advance

Apply normalize on evaluated rows and advance replay registers. After the first group, eligible warps with any alpha unequal to one rescale their 128 NumState channels in 32-channel TMEM round trips and wait for stores. Form Prob8 and DenPart. Wholly padded PSK warps do neither rescaling nor probability overwrite; their retained Prob8 rows still enter PV. All warps wait stores, fence before thread-sync and CTA-sync, publishing live probability/state writes and retiring Score/atom readers.

## t_0a400de157421488 | pipeline.verb | 311 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:565 design.tile_pipeline.phase.segment_max.role.Main.info.verb.reduce

Each valid thread loads its segment into GroupBatch, substituting -inf beyond T, before folding into SegmentState initialized to -inf. Lanes address consecutive rows in group-major storage. Reduction is within each thread, without a cross-thread tree or barrier. Out-of-range rows perform no loads or reduction.

## t_0b9f4ddc3040be2f | pipeline.verb | 425 chars | 1 declarations

tcg_d128_fscan_cp_direct_wide:654 design.tile_pipeline.phase.combine.role.Main.info.verb.write

Group zero alone consumes GroupA and GroupB in combine's specified order, with TotalB replicated per channel, then writes its BF16 channels. Its first thread reads MTail and writes LSE when present. Other groups execute this action as a no-op, without output reads or stores. Shared buffers are never overwritten afterward. Initial destinations are never read. Kernel completion provides visibility; no next tile is assigned.

## t_0df762598a72cfd0 | pipeline.verb | 448 chars | 1 declarations

tcg_d128_fscan_tma_xpose_wide:676 design.tile_pipeline.phase.combine.role.Main.info.verb.sum_a.meaning

Each group consumes the prefetched batch, then full eight-entry batches and the scalar remainder, in the combine step's order. GroupAccumulator carries its channel sums; batches are reused after consumption. Store the completed sums to GroupA. Even an empty interval stores zero for every channel. done covers all GroupA stores, but groups may proceed to their denominator work independently; no CTA barrier separates these stores from those loads.

## t_0f0bff88c5905400 | pipeline.verb | 421 chars | 1 declarations

tcg_d128_flat_tma_xpose_narrow:613 design.tile_pipeline.phase.split.role.Main.info.verb.write

All threads cooperatively read 16-byte chunks of staged FP32 PartA for valid rows and write global PartA through domain's head-major map, preserving FP32 bits. Each valid row owner separately writes weighted DenState to PartB; empty splits write PartB=0 from registers. done means stores issued with sources consumed; no later local reuse occurs and kernel completion provides global visibility. No O/LSE is written here.

## t_0f4bba15d06dda41 | pipeline.verb | 130 chars | 4 declarations

tcg_d256_flat_cp_narrow:470 design.tile_pipeline.phase.scan.role.Main.info.verb.batch16; tcg_d256_flat_cp_wide:484 design.tile_pipeline.phase.scan.role.Main.info.verb.batch16; tcg_d256_flat_tma_narrow:468 design.tile_pipeline.phase.scan.role.Main.info.verb.batch16; tcg_d256_flat_tma_wide:482 design.tile_pipeline.phase.scan.role.Main.info.verb.batch16

Apply the same load-then-replay procedure to 16 groups starting at 64*floor(GCount/64)+16*j, reusing the first 16 Batch positions.

## t_0f8dbb2514751024 | pipeline.verb | 278 chars | 1 declarations

tc_fused:310 design.tile_pipeline.phase.main.role.Main.info.verb.prepare

Load/store Qraw into Query and decode QueryScale, publishing both at a CTA barrier. Invalid query payloads and scales are zero. Initialize NumState, MaxState and DenPart as in math_reshape. Query and QueryScale survive all groups. Ordinary Q loads may overlap the primed copies.

## t_0fa658549e057156 | pipeline.verb | 388 chars | 2 declarations

tc_split_segmented_narrow:614 design.tile_pipeline.phase.scan.role.Main.info.verb.entry; tc_split_segmented_wide:648 design.tile_pipeline.phase.scan.role.Main.info.verb.entry

Wait for grid dependencies, then trigger scan-to-split launch completion, before the row>=R return. The wait requires every segmax thread to exit and its stores to be visible; its entry trigger supplies launch permission only. The ordinary split successor waits for scan.kernel_done and does not use the scan's early permission. Source compile-time no-op helpers do not add early signals.

## t_0ffee10f10ee9ff4 | pipeline.verb | 363 chars | 1 declarations

tcg_d256_segmented_cp_wide:806 design.tile_pipeline.phase.split.role.Main.info.verb.fold

Both halves write DenPart to Exchange, CTA-sync and read their sum, including C=0. For nonempty valid rows read Mtail and compute the partial weight; otherwise weight remains zero. This barrier joins all final QK/PV and atom readers before operand storage is reused. All nonempty copies have completed; only empty cp.async commits can remain, carrying no payload.

## t_10bff4c153cdfb83 | pipeline.verb | 368 chars | 1 declarations

tcg_d128_fscan_tma_xpose_narrow:562 design.tile_pipeline.phase.split.role.Main.info.verb.probability

All warps apply scan's update from CurrentMax, advance CurrentMax to NextMax, rescale DenState/NumState, consume DotTile in 32-key chunks and write Prob8, including padded rows. Wait for TMEM stores, fence and CTA-sync to publish every probability row and retire all DotTile readers before PV overwrites it. Padded rows are computed but do not produce global partials.

## t_112c886069ac6b1a | pipeline.verb | 163 chars | 4 declarations

tc_split_flat_narrow:479 design.tile_pipeline.phase.group_max.role.Main.info.verb.land; tc_split_flat_wide:496 design.tile_pipeline.phase.group_max.role.Main.info.verb.land; tc_split_segmented_narrow:554 design.tile_pipeline.phase.group_max.role.Main.info.verb.land; tc_split_segmented_wide:588 design.tile_pipeline.phase.group_max.role.Main.info.verb.land

Wait until at most the newest committed copy group remains pending, then join the CTA. Current K and scale atoms are readable while next-group copies may continue.

## t_1213d46d80df1a11 | pipeline.verb | 323 chars | 2 declarations

tcg_d128_flat_cp_xpose_narrow:576 design.tile_pipeline.phase.split.role.Main.info.verb.prime; tcg_d128_flat_cp_xpose_wide:559 design.tile_pipeline.phase.split.role.Main.info.verb.prime

For Ng>0 enqueue K/V(cLo) and their 512-byte atoms into slot zero using 16-byte cp.async copies, then each thread commits its group. Invalid payload copies zero-fill from a clamped source address; allocated scale padding bytes are copied unchanged. For Ng=0 issue no copy or commit. done is enqueue and commit, not arrival.

## t_1282c0a7950c9bd5 | pipeline.verb | 266 chars | 20 declarations

tcg_d128_flat_cp_direct_narrow:421 design.tile_pipeline.phase.group_max.role.Main.info.verb.prepare; tcg_d128_flat_cp_direct_wide:435 design.tile_pipeline.phase.group_max.role.Main.info.verb.prepare; tcg_d128_flat_cp_xpose_narrow:429 design.tile_pipeline.phase.group_max.role.Main.info.verb.prepare; tcg_d128_flat_cp_xpose_wide:412 design.tile_pipeline.phase.group_max.role.Main.info.verb.prepare; tcg_d128_flat_tma_direct_narrow:420 design.tile_pipeline.phase.group_max.role.Main.info.verb.prepare; tcg_d128_flat_tma_direct_wide:434 design.tile_pipeline.phase.group_max.role.Main.info.verb.prepare; tcg_d128_flat_tma_xpose_narrow:424 design.tile_pipeline.phase.group_max.role.Main.info.verb.prepare; tcg_d128_flat_tma_xpose_wide:442 design.tile_pipeline.phase.group_max.role.Main.info.verb.prepare; tcg_d128_fscan_cp_direct_narrow:411 design.tile_pipeline.phase.group_max.role.Main.info.verb.prepare; tcg_d128_fscan_cp_direct_wide:428 design.tile_pipeline.phase.group_max.role.Main.info.verb.prepare; tcg_d128_fscan_tma_direct_narrow:409 design.tile_pipeline.phase.group_max.role.Main.info.verb.prepare; tcg_d128_fscan_tma_direct_wide:427 design.tile_pipeline.phase.group_max.role.Main.info.verb.prepare; tcg_d128_fscan_tma_xpose_narrow:415 design.tile_pipeline.phase.group_max.role.Main.info.verb.prepare; tcg_d128_segmented_cp_direct_narrow:477 design.tile_pipeline.phase.group_max.role.Main.info.verb.prepare; tcg_d128_segmented_cp_direct_wide:494 design.tile_pipeline.phase.group_max.role.Main.info.verb.prepare; tcg_d128_segmented_cp_xpose_narrow:465 design.tile_pipeline.phase.group_max.role.Main.info.verb.prepare; tcg_d128_segmented_tma_direct_narrow:475 design.tile_pipeline.phase.group_max.role.Main.info.verb.prepare; tcg_d128_segmented_tma_direct_wide:494 design.tile_pipeline.phase.group_max.role.Main.info.verb.prepare; tcg_d128_segmented_tma_xpose_narrow:481 design.tile_pipeline.phase.group_max.role.Main.info.verb.prepare; tcg_d128_segmented_tma_xpose_wide:465 design.tile_pipeline.phase.group_max.role.Main.info.verb.prepare

Prefetch Q and scale words into QueryPrefetch, even for empty splits. First warp allocates TMEM; a CTA barrier publishes its base before that warp relinquishes the permit. Write Query/QueryScale, wait for TMEM stores and discard QueryPrefetch. Q survives all groups.

## t_1306dfb65c776889 | pipeline.verb | 306 chars | 2 declarations

tcg_d128_fscan_cp_xpose_narrow:536 design.tile_pipeline.phase.split.role.Main.info.verb.prime; tcg_d128_fscan_cp_xpose_wide:553 design.tile_pipeline.phase.split.role.Main.info.verb.prime

For Ng>0 all warps cooperatively enqueue K(cLo), V(cLo) and both 512-byte scale atoms in slot zero, and each thread commits its copies as one group. Payload copies zero-fill keys beyond Skv; atoms read allocated scale bytes. For Ng=0 issue neither copies nor a commit. done means enqueue, not data arrival.

## t_161b37a0390b05dd | pipeline.verb | 477 chars | 2 declarations

tcg_d128_segmented_cp_direct_narrow:469 design.tile_pipeline.phase.group_max.role.Main.info.verb.setup; tcg_d128_segmented_cp_direct_wide:486 design.tile_pipeline.phase.group_max.role.Main.info.verb.setup

Establish coordinates, masks and initially zero completion parities. RingUsed=3 for Long, otherwise 2; ScaleCol(g)=136+8*(g%2) for Long, otherwise 136. Three shared slots remain allocated in both cases. Valid rows of the first/last split issue the outside -inf fills. Thread zero initializes matrix completion barriers and issues MMA; all four warps stage Q, copy payloads/scales and synchronize. Only warps with a valid row perform scalar score readout. q0>=Sq is unreachable.

## t_17e2885bfa2cec1a | pipeline.verb | 312 chars | 7 declarations

tcg_d256_flat_cp_narrow:619 design.tile_pipeline.phase.split.role.Main.info.verb.fold; tcg_d256_flat_cp_wide:633 design.tile_pipeline.phase.split.role.Main.info.verb.fold; tcg_d256_flat_tma_narrow:614 design.tile_pipeline.phase.split.role.Main.info.verb.fold; tcg_d256_flat_tma_wide:628 design.tile_pipeline.phase.split.role.Main.info.verb.fold; tcg_d256_segmented_cp_narrow:687 design.tile_pipeline.phase.split.role.Main.info.verb.fold; tcg_d256_segmented_tma_narrow:702 design.tile_pipeline.phase.split.role.Main.info.verb.fold; tcg_d256_segmented_tma_wide:687 design.tile_pipeline.phase.split.role.Main.info.verb.fold

Both halves write DenPart to Exchange, CTA-sync and sum them, including C=0. For nonempty valid rows read Mtail and form weight; otherwise it remains zero. This barrier joins all final QK/PV and atom readers before OperandScratch is overwritten. Every nonempty copy has completed; empty commits carry no payload.

## t_18da37462dffbd25 | pipeline.verb | 169 chars | 2 declarations

tc_split_flat_narrow:527 design.tile_pipeline.phase.scan.role.Main.info.verb.store; tc_split_flat_wide:544 design.tile_pipeline.phase.scan.role.Main.info.verb.store

Write each valid MaxTail. No final trigger follows these stores; kernel_done supplies visibility. The unfilled maximum region retains its incoming contents across calls.

## t_18daf717bb16c03f | pipeline.verb | 614 chars | 1 declarations

fused_d256_tma:323 design.tile_pipeline.phase.main.role.Main.info.verb.setup

Establish work coordinates and completion protocols. The q0>=Sq guard exits before allocation or copying; the exact grid never takes it. Initialize three K and two V transaction barriers and QK/PV completion phases at parity zero, publishing initialization before bulk use. Warps 0..3 and 4..7 form halves z=0 and z=1: thread 128*z+u owns row u, keys [64*z,64*z+64) and output channels [128*z,128*z+128). Both halves execute every row, including padding. MaxState is replicated identically; DenPart is separate per half. Thread 0 issues all TMA and matrix products; the eight warps cooperate on the remaining work.

## t_1a753aee814b56e7 | pipeline.verb | 268 chars | 2 declarations

tcg_d128_segmented_cp_direct_narrow:663 design.tile_pipeline.phase.split.role.Main.info.verb.setup; tcg_d128_segmented_cp_direct_wide:680 design.tile_pipeline.phase.split.role.Main.info.verb.setup

Establish coordinates and PSK valid/live-warp predicates. Initialize QK/PV completion barriers at parity zero. All four warps copy payloads/scales, commit/wait copy groups and synchronize; one thread maps to each row, and thread zero issues MMA. q0>=Sq is unreachable.

## t_1b1d0112a4a5810c | pipeline.verb | 237 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:568 design.tile_pipeline.phase.segment_max.role.Main.info.verb.write

Valid row owners issue SegMax stores from SegmentState, consuming the register value. All segments are written, including segment zero. Out-of-range rows have no stores. Global visibility follows kernel completion and the successor wait.

## t_1b9c8b749bdec0b4 | pipeline.verb | 532 chars | 1 declarations

fused_d256_cp:324 design.tile_pipeline.phase.main.role.Main.info.verb.setup

Establish work coordinates and QK/PV completion phases at parity zero. The q0>=Sq guard exits before allocation or copying; the exact grid never takes it. prepare's CTA barrier publishes initialization. Warps 0..3 and 4..7 form halves z=0 and z=1: thread 128*z+u owns row u, keys [64*z,64*z+64) and output channels [128*z,128*z+128). Both halves execute every row, including padding. MaxState is replicated identically; DenPart is separate per half. Thread 0 issues all matrix products; all eight warps cooperate in payload staging.

## t_1bab2dbb9b4eec98 | pipeline.verb | 588 chars | 12 declarations

tcg_d128_flat_tma_direct_narrow:428 design.tile_pipeline.phase.group_max.role.Main.info.verb.stage; tcg_d128_flat_tma_direct_wide:442 design.tile_pipeline.phase.group_max.role.Main.info.verb.stage; tcg_d128_flat_tma_xpose_narrow:432 design.tile_pipeline.phase.group_max.role.Main.info.verb.stage; tcg_d128_flat_tma_xpose_wide:450 design.tile_pipeline.phase.group_max.role.Main.info.verb.stage; tcg_d128_fscan_tma_direct_narrow:417 design.tile_pipeline.phase.group_max.role.Main.info.verb.stage; tcg_d128_fscan_tma_direct_wide:435 design.tile_pipeline.phase.group_max.role.Main.info.verb.stage; tcg_d128_fscan_tma_xpose_narrow:423 design.tile_pipeline.phase.group_max.role.Main.info.verb.stage; tcg_d128_segmented_cp_xpose_narrow:473 design.tile_pipeline.phase.group_max.role.Main.info.verb.stage; tcg_d128_segmented_tma_direct_narrow:483 design.tile_pipeline.phase.group_max.role.Main.info.verb.stage; tcg_d128_segmented_tma_direct_wide:502 design.tile_pipeline.phase.group_max.role.Main.info.verb.stage; tcg_d128_segmented_tma_xpose_narrow:489 design.tile_pipeline.phase.group_max.role.Main.info.verb.stage; tcg_d128_segmented_tma_xpose_wide:473 design.tile_pipeline.phase.group_max.role.Main.info.verb.stage

For Long wait for K(g+1) and atom when g<cHi, fence shared access and CTA-sync, then write its alternating KeyScale, wait for TMEM stores, fence and CTA-sync. At g=cHi the source still copies the next physical atom slot's stale bytes to the alternate scale slot; no later QK consumes them and no new transfer/wait is performed. With Ng>=8 that shared slot has already been filled. For non-Long wait for current K(g) and atom, fence and CTA-sync, write KeyScale at its fixed base, wait for stores, fence and CTA-sync. Transaction parity follows the actual ring wrap. These joins publish Q.

## t_1c991e70792c2bdd | pipeline.verb | 319 chars | 2 declarations

tcg_d128_flat_cp_xpose_narrow:573 design.tile_pipeline.phase.split.role.Main.info.verb.setup; tcg_d128_flat_cp_xpose_wide:556 design.tile_pipeline.phase.split.role.Main.info.verb.setup

Establish head-major coordinates and row validity; rowsLive is always true. Initialize QK/PV completion barriers at parity zero. All four warps cooperate, with one thread per row. Thread zero issues MMA, all warps copy payloads and the first warp copies atoms. No TMA transaction barrier is used; q0>=Sq is unreachable.

## t_1ca35d1f5b3fbe9b | pipeline.verb | 238 chars | 1 declarations

tc_split_flat_wide:672 design.tile_pipeline.phase.combine.role.Main.info.verb.publish

One CTA barrier joins all sixteen warps and makes GroupSums readable. The two shared buffers are disjoint and remain unchanged through their final reads. No shared weight table or earlier publication barrier is used in this instantiation.

## t_1e3d396701e38e52 | pipeline.verb | 365 chars | 1 declarations

tcg_d128_fscan_tma_direct_wide:647 design.tile_pipeline.phase.combine.role.Main.info.verb.write

Only group zero forms TotalB and each channel's TotalA in combine's order and writes its BF16 channel. Its first thread rereads MTail and writes LSE when present. Other groups have no data access in this action. Initial destinations are never read; kernel completion provides visibility. No next tile is assigned and no additional launch-permission event is issued.

## t_1e4dcf9997c087fc | pipeline.verb | 76 chars | 7 declarations

tcg_d256_flat_cp_narrow:560 design.tile_pipeline.phase.split.role.Main.info.verb.await; tcg_d256_flat_cp_wide:574 design.tile_pipeline.phase.split.role.Main.info.verb.await; tcg_d256_flat_tma_narrow:558 design.tile_pipeline.phase.split.role.Main.info.verb.await; tcg_d256_flat_tma_wide:572 design.tile_pipeline.phase.split.role.Main.info.verb.await; tcg_d256_segmented_cp_narrow:631 design.tile_pipeline.phase.split.role.Main.info.verb.await; tcg_d256_segmented_tma_narrow:646 design.tile_pipeline.phase.split.role.Main.info.verb.await; tcg_d256_segmented_tma_wide:631 design.tile_pipeline.phase.split.role.Main.info.verb.await

Perform the PDL data wait after preparation and before any scan-output read.

## t_1f75c6a337b5551f | pipeline.verb | 492 chars | 1 declarations

fused_d128_cp:354 design.tile_pipeline.phase.main.role.Main.info.verb.land

Inputs(g) waits for each thread's committed cp.async groups preceding its most recent commit, proving the current Key, Value and both atoms complete while permitting the next group to remain in flight. A shared async-proxy fence and CTA barrier publish these operands and the initial Query stores. All four warps copy the current atoms into KeyScale and ValueScale, wait for their TMEM stores, then fence and CTA-sync before QK. Previous QK and PV completion protect these single scale slots.

## t_1fccdddc39dd910a | pipeline.verb | 772 chars | 1 declarations

fused_d256_cp:403 design.tile_pipeline.phase.main.role.Main.info.verb.pv

Thread 0 applies its after-thread-sync fence and enqueues four increasing 32-key steps, each containing both 128-channel output products. Read Prob8, Value(g), UnitScale and ValueScale(g); the first key step of the first group overwrites both output halves, later steps accumulate. Commit one PV completion phase. Every thread waits, advances parity once and applies the completion fence. done retires Value, ValueScale and Prob8 reads and NumState updates before the next refill, rescale or probability write. No tail CTA barrier or per-group numerator readback follows. At the final group all products and nonempty cp.async groups have completed; its newest commit is empty and unused tail scale stores already waited locally. No additional post-loop drain is performed.

## t_2059bc27ff8c42e8 | pipeline.verb | 298 chars | 2 declarations

tcg_d128_flat_cp_direct_narrow:429 design.tile_pipeline.phase.group_max.role.Main.info.verb.prefetch; tcg_d128_flat_cp_direct_wide:443 design.tile_pipeline.phase.group_max.role.Main.info.verb.prefetch

Enqueue K(g+2) and atom for Long, otherwise K(g+1) and atom, only when that group is <=cHi. Every thread then commits a group, including an empty group when no transfer remains. done means enqueue/commit; prior QK completion and reader joins protect reuse of the destination payload and atom slots.

## t_2109621ebf0f5bb8 | pipeline.verb | 453 chars | 2 declarations

tc_split_segmented_narrow:629 design.tile_pipeline.phase.scan.role.Main.info.verb.fill_walk; tc_split_segmented_wide:663 design.tile_pipeline.phase.scan.role.Main.info.verb.fill_walk

Consume JumpMask and ScanCarry to store the constant prefix and walk the entire selected suffix with the GroupBatch protocol of walk. Do not retest later summaries against updated state. A zero mask writes constants without maximum loads or a walk call; the action still completes. Rows can choose different suffixes without cross-row synchronization. done retires all batch reads and preserves ScanCarry before the next chunk reuses the register slots.

## t_2141beac120167bd | pipeline.verb | 168 chars | 1 declarations

tcg_d256_flat_tma_wide:675 design.tile_pipeline.phase.combine.role.Main.info.verb.prepare

Group z uses warps [4*z,4*z+4) and splits [SLo(z),SHi(z)). Its local thread d0=0..127 owns channels d0 and d0+128. Initialize Numerator to zero, including empty groups.

## t_2196a21c9e7f114c | pipeline.verb | 184 chars | 3 declarations

tcg_d128_segmented_cp_xpose_narrow:546 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.dependency; tcg_d128_segmented_tma_xpose_narrow:562 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.dependency; tcg_d128_segmented_tma_xpose_wide:546 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.dependency

Execute the PDL wait before any early return or input load. done guarantees segment_max completion and SegMax visibility, and transitively group_max completion and GroupMax visibility.

## t_21d3333c17c85f98 | pipeline.verb | 260 chars | 4 declarations

tc_split_flat_narrow:604 design.tile_pipeline.phase.split.role.Main.info.verb.close; tc_split_flat_wide:621 design.tile_pipeline.phase.split.role.Main.info.verb.close; tc_split_segmented_narrow:711 design.tile_pipeline.phase.split.role.Main.info.verb.close; tc_split_segmented_wide:745 design.tile_pipeline.phase.split.role.Main.info.verb.close

Add the E component pairs into DenPart. A CTA barrier joins all current raw-input, decoded-scale and Vbf readers before the next prefetch/decode. Score, E and PreviousMax are dead. NumState and DenPart carry forward; the last CurrentMax survives for weighting.

## t_2205a9a4e894fbb2 | pipeline.verb | 171 chars | 2 declarations

tc_split_flat_narrow:650 design.tile_pipeline.phase.combine.role.Main.info.verb.load; tc_split_segmented_narrow:757 design.tile_pipeline.phase.combine.role.Main.info.verb.load

For channel slice c and batch k, issue eight independent FP32 PartialA loads per thread, substituting zero past Ns. done makes NumBatch readable; no new weight is applied.

## t_2282210fd5f6f451 | pipeline.verb | 445 chars | 1 declarations

tcg_d128_fscan_tma_xpose_wide:451 design.tile_pipeline.phase.group_max.role.Main.info.verb.finish

All enqueued K transactions and QK products have been waited. A before-thread-sync fence and CTA barrier join final TMEM users; the first warp deallocates 256 columns. Each CTA signals PDL launch completion after its global stores and local teardown; launch_ready applies when the whole grid satisfies the protocol. Actual global result visibility is established by kernel completion and the successor dependency wait, not by enqueueing a store.

## t_2282f1b98f6219eb | pipeline.verb | 135 chars | 4 declarations

tc_split_flat_narrow:481 design.tile_pipeline.phase.group_max.role.Main.info.verb.decode; tc_split_flat_wide:498 design.tile_pipeline.phase.group_max.role.Main.info.verb.decode; tc_split_segmented_narrow:556 design.tile_pipeline.phase.group_max.role.Main.info.verb.decode; tc_split_segmented_wide:590 design.tile_pipeline.phase.group_max.role.Main.info.verb.decode

Decode KeyScale, zeroing j>=Skv, and publish at a CTA barrier. The preceding group-end barrier protects this single decoded-scale slot.

## t_22d0ab7daddf73db | pipeline.verb | 249 chars | 2 declarations

tc_split_flat_narrow:645 design.tile_pipeline.phase.combine.role.Main.info.verb.entry; tc_split_segmented_narrow:752 design.tile_pipeline.phase.combine.role.Main.info.verb.entry

Wait for grid dependencies before partial reads. The ordinary split predecessor emits no early trigger, so delivery requires its completion. Initialize SumA to zero. Thread tid owns channels tid+c*CombineBt; the exact grid needs no row bounds guard.

## t_22f6cd8f791ba583 | pipeline.verb | 242 chars | 4 declarations

tc_split_flat_narrow:483 design.tile_pipeline.phase.group_max.role.Main.info.verb.qk; tc_split_flat_wide:500 design.tile_pipeline.phase.group_max.role.Main.info.verb.qk; tc_split_segmented_narrow:558 design.tile_pipeline.phase.group_max.role.Main.info.verb.qk; tc_split_segmented_wide:592 design.tile_pipeline.phase.group_max.role.Main.info.verb.qk

Run scores with synchronous FP8 register-result products. Retain full Score through the maximum reduction; warp mask intersections skip element tests only for wholly visible groups. done includes Q/K reads and the four-lane maximum reduction.

## t_23502bd848f961d9 | pipeline.verb | 244 chars | 2 declarations

tc_split_segmented_narrow:626 design.tile_pipeline.phase.scan.role.Main.info.verb.select; tc_split_segmented_wide:660 design.tile_pipeline.phase.scan.role.Main.info.verb.select

Test every Summary(c) entry against the unchanged incoming ScanCarry and form JumpMask; bit k represents the scan step's candidate condition. All summary reads end here. The lowest set bit selects the suffix start; zero selects an empty suffix.

## t_23a06b9a680d1aa5 | pipeline.verb | 200 chars | 12 declarations

tcg_d128_flat_tma_direct_narrow:426 design.tile_pipeline.phase.group_max.role.Main.info.verb.prefetch; tcg_d128_flat_tma_direct_wide:440 design.tile_pipeline.phase.group_max.role.Main.info.verb.prefetch; tcg_d128_flat_tma_xpose_narrow:430 design.tile_pipeline.phase.group_max.role.Main.info.verb.prefetch; tcg_d128_flat_tma_xpose_wide:448 design.tile_pipeline.phase.group_max.role.Main.info.verb.prefetch; tcg_d128_fscan_tma_direct_narrow:415 design.tile_pipeline.phase.group_max.role.Main.info.verb.prefetch; tcg_d128_fscan_tma_direct_wide:433 design.tile_pipeline.phase.group_max.role.Main.info.verb.prefetch; tcg_d128_fscan_tma_xpose_narrow:421 design.tile_pipeline.phase.group_max.role.Main.info.verb.prefetch; tcg_d128_segmented_cp_xpose_narrow:471 design.tile_pipeline.phase.group_max.role.Main.info.verb.prefetch; tcg_d128_segmented_tma_direct_narrow:481 design.tile_pipeline.phase.group_max.role.Main.info.verb.prefetch; tcg_d128_segmented_tma_direct_wide:500 design.tile_pipeline.phase.group_max.role.Main.info.verb.prefetch; tcg_d128_segmented_tma_xpose_narrow:487 design.tile_pipeline.phase.group_max.role.Main.info.verb.prefetch; tcg_d128_segmented_tma_xpose_wide:471 design.tile_pipeline.phase.group_max.role.Main.info.verb.prefetch

Enqueue K(g+2) and atom for Long, otherwise K(g+1) and atom, only when that group is <=cHi. Otherwise issue no transfer. done means enqueue; previous QK completion and reader joins protect ring reuse.

## t_23abd94560526e1b | pipeline.verb | 969 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:731 design.tile_pipeline.phase.split.role.Main.info.verb.stage_partials

Nonempty valid rows load MTail and form weighted FP32 numerators; empty splits evaluate zeros. Each valid row writes its D channels into the PartA view of OperandArena; padded rows are neither staged nor later read. The 81920-byte operand capacity accommodates the 67584-byte staging span, including row spacing. For Ng>0, stage's CTA joins completed all Query writes, QK/PV waits retired operand reads, and the last group enqueued no future transfer, so the arena is free before these writes. For Ng=0, no matrix iteration or its CTA joins occur: a warp's zero staging writes can overlap another warp's unfinished Query writes in the same shared region. No pre-staging barrier orders those writes, so the staged bytes need not encode zero. The source's CTA barrier is after the staging writes. It joins all preparation/staging writes and any final TMEM readouts, making the resulting bytes readable; it does not repair the empty-split race. done includes this barrier.

## t_23e996cdc16d847f | pipeline.verb | 678 chars | 1 declarations

tcg_d128_flat_tma_xpose_wide:695 design.tile_pipeline.phase.combine.role.Main.info.verb.partials

Group t consists of warps [4*t,4*t+4); its 128 threads each own a channel d over [ALo(t),AHi(t)). Initialize LocalA[t,d]=0, load full eight-split batches into NumeratorBatch and consume each before reuse, then handle the scalar tail. Store the resulting MergeA[t,d] to shared memory. Each thread also reads PartB[ALo(t)+d,r], or uses zero beyond AHi(t); warp z within the group performs combine's five down-shuffle stages and its lane zero writes MergeB[t,z]. done includes the final CTA barrier, publishing all numerator and denominator partials. Before that barrier the groups progress independently; empty intervals still publish zeros. No additional shared weights are used.

## t_24ad43594c9e81c7 | pipeline.verb | 276 chars | 7 declarations

tcg_d256_flat_cp_narrow:625 design.tile_pipeline.phase.split.role.Main.info.verb.stage; tcg_d256_flat_cp_wide:639 design.tile_pipeline.phase.split.role.Main.info.verb.stage; tcg_d256_flat_tma_narrow:620 design.tile_pipeline.phase.split.role.Main.info.verb.stage; tcg_d256_flat_tma_wide:634 design.tile_pipeline.phase.split.role.Main.info.verb.stage; tcg_d256_segmented_cp_narrow:693 design.tile_pipeline.phase.split.role.Main.info.verb.stage; tcg_d256_segmented_tma_narrow:708 design.tile_pipeline.phase.split.role.Main.info.verb.stage; tcg_d256_segmented_tma_wide:693 design.tile_pipeline.phase.split.role.Main.info.verb.stage

Read each half's NumState in self-retiring 32-channel loads, or synthesize zero chunks for C=0. Valid rows multiply by weight and write PartAStaging at raw base with 1040-byte row pitch, including 16-byte row padding. Query/Key/Value are dead; padded rows do not stage output.

## t_25319008fa74b2c4 | pipeline.verb | 386 chars | 10 declarations

tcg_d128_flat_cp_direct_narrow:659 design.tile_pipeline.phase.combine.role.Main.info.verb.sum_a; tcg_d128_flat_cp_xpose_narrow:687 design.tile_pipeline.phase.combine.role.Main.info.verb.sum_a; tcg_d128_flat_cp_xpose_wide:670 design.tile_pipeline.phase.combine.role.Main.info.verb.sum_a; tcg_d128_flat_tma_direct_narrow:652 design.tile_pipeline.phase.combine.role.Main.info.verb.sum_a; tcg_d128_flat_tma_xpose_narrow:675 design.tile_pipeline.phase.combine.role.Main.info.verb.sum_a; tcg_d128_segmented_cp_direct_narrow:750 design.tile_pipeline.phase.combine.role.Main.info.verb.sum_a; tcg_d128_segmented_cp_xpose_narrow:746 design.tile_pipeline.phase.combine.role.Main.info.verb.sum_a; tcg_d128_segmented_tma_direct_narrow:742 design.tile_pipeline.phase.combine.role.Main.info.verb.sum_a; tcg_d128_segmented_tma_xpose_narrow:762 design.tile_pipeline.phase.combine.role.Main.info.verb.sum_a; tcg_d128_segmented_tma_xpose_wide:746 design.tile_pipeline.phase.combine.role.Main.info.verb.sum_a

First cooperatively load Ns PartB entries into DenParts, leaving unused capacity unwritten and unread. Each thread owns one channel, loads numerator batches of eight into NumeratorBatch and performs combine's ordered addition, consuming each batch before reuse. Missing final-batch entries are zero. The final CTA barrier joins all DenParts stores and publishes them before sum_b reads.

## t_25e57f90c4d341fc | pipeline.verb | 171 chars | 7 declarations

tcg_d256_flat_cp_narrow:423 design.tile_pipeline.phase.group_max.role.Main.info.verb.qk; tcg_d256_flat_cp_wide:437 design.tile_pipeline.phase.group_max.role.Main.info.verb.qk; tcg_d256_flat_tma_narrow:421 design.tile_pipeline.phase.group_max.role.Main.info.verb.qk; tcg_d256_flat_tma_wide:435 design.tile_pipeline.phase.group_max.role.Main.info.verb.qk; tcg_d256_segmented_cp_narrow:459 design.tile_pipeline.phase.group_max.role.Main.info.verb.qk; tcg_d256_segmented_tma_narrow:474 design.tile_pipeline.phase.group_max.role.Main.info.verb.qk; tcg_d256_segmented_tma_wide:459 design.tile_pipeline.phase.group_max.role.Main.info.verb.qk

Thread zero fences after thread-sync, issues eight score contractions and commits QK. Every thread waits for completion, advances QK parity and fences, retiring Q/K reads.

## t_25f33968a8be1ba5 | pipeline.verb | 150 chars | 3 declarations

tcg_d128_segmented_cp_xpose_narrow:521 design.tile_pipeline.phase.segment_max.role.Main.info.verb.dependency; tcg_d128_segmented_tma_xpose_narrow:537 design.tile_pipeline.phase.segment_max.role.Main.info.verb.dependency; tcg_d128_segmented_tma_xpose_wide:521 design.tile_pipeline.phase.segment_max.role.Main.info.verb.dependency

Every thread waits for group_max before the row-range guard or any GroupMax load. done guarantees full predecessor completion and GroupMax visibility.

## t_26629c72b979992c | pipeline.verb | 228 chars | 1 declarations

fused_d128_group:472 design.tile_pipeline.phase.main.role.Even.info.verb.stagev

After PV enqueue, issue V(g+2) and its atom into local slot (j+1)%2 when present, and always commit a copy group. The old g-2 reader was retired by Own(j); the fill may overlap the current PV, which reads the other local V slot.

## t_2705240850d78bdc | pipeline.verb | 567 chars | 1 declarations

tcg_d128_fscan_cp_xpose_wide:676 design.tile_pipeline.phase.combine.role.Main.info.verb.prime

Warps 4*cg through 4*cg+3 form merge group cg. Its 128 threads each own one channel d and the possible split sLo(cg)+d. Clear NumAccumulator, read MTail, compute each valid split's Weight once, and store it in shared position (cg,d); store zero for positions beyond sHi(cg). Preload the first eight PartA values per channel into NumeratorBatch, substituting zero beyond sHi(cg), before the CTA barrier. done includes that barrier, publishing all weights to every channel and making the first numerator batch ready. Empty groups participate and initialize their slots.

## t_276dd32637f73b29 | pipeline.verb | 540 chars | 1 declarations

fused_d128_cp:337 design.tile_pipeline.phase.main.role.Main.info.verb.prepare

Prefetch this tile's Q payload and scale words into QueryPrefetch registers before allocating TMEM. The first warp allocates; a CTA barrier publishes the base, then that warp relinquishes its allocation permit. Write Query and QueryScale/UnitScale, wait for these TMEM stores, then discard QueryPrefetch. Initialize MaxState and DenState as in math_reshape; do not clear NumState. Query and its scales survive all groups. Commit the source's empty cp.async group before the loop, preserving pending-group accounting after the hoisted prime.

## t_28cc6b5831d6145a | pipeline.verb | 116 chars | 3 declarations

tcg_d256_segmented_cp_narrow:524 design.tile_pipeline.phase.scan.role.Main.info.verb.await; tcg_d256_segmented_tma_narrow:539 design.tile_pipeline.phase.scan.role.Main.info.verb.await; tcg_d256_segmented_tma_wide:524 design.tile_pipeline.phase.scan.role.Main.info.verb.await

Every thread waits for segment_max at entry, proving SegMax and transitively GroupMax visible before any input read.

## t_29236c9307681835 | pipeline.verb | 148 chars | 4 declarations

tcg_d128_fscan_cp_direct_narrow:546 design.tile_pipeline.phase.split.role.Main.info.verb.dependency; tcg_d128_fscan_cp_direct_wide:563 design.tile_pipeline.phase.split.role.Main.info.verb.dependency; tcg_d128_fscan_tma_direct_narrow:540 design.tile_pipeline.phase.split.role.Main.info.verb.dependency; tcg_d128_fscan_tma_direct_wide:558 design.tile_pipeline.phase.split.role.Main.info.verb.dependency

Execute the PDL dependency wait before the first GroupMax read. done guarantees group_max completion and visibility for all prefix and local maxima.

## t_2952d8ed8fb1f7db | pipeline.verb | 306 chars | 2 declarations

tcg_d256_flat_cp_narrow:395 design.tile_pipeline.phase.group_max.role.Main.info.verb.setup; tcg_d256_flat_cp_wide:409 design.tile_pipeline.phase.group_max.role.Main.info.verb.setup

Establish coordinates and the domain row map. Warpgroups z=0,1 each cover Br rows and keys [64*z,64*z+64). Half zero writes its assigned -inf tails. Initialize QK/PV completion protocols at parity zero and publish initialization before use. The exact grid never takes q0>=Sq; empty spans still reach setup.

## t_29ac8a49cd1d97b9 | pipeline.verb | 288 chars | 1 declarations

fused_d128_key:379 design.tile_pipeline.phase.main.role.Main.info.verb.retire

Previous(g) waits for PV(g-1), advances PV parity once and applies the after-thread-sync fence. This completes its NumState update and all Value/ValueScale/Prob8 reads before the current rescale or probability overwrite. At g=cLo the action completes without a wait or accumulator access.

## t_2a4226b3aea4e086 | pipeline.verb | 324 chars | 8 declarations

tcg_d128_flat_cp_direct_narrow:579 design.tile_pipeline.phase.split.role.Main.info.verb.prepare; tcg_d128_flat_cp_direct_wide:593 design.tile_pipeline.phase.split.role.Main.info.verb.prepare; tcg_d128_flat_tma_direct_narrow:574 design.tile_pipeline.phase.split.role.Main.info.verb.prepare; tcg_d128_flat_tma_direct_wide:588 design.tile_pipeline.phase.split.role.Main.info.verb.prepare; tcg_d128_segmented_cp_direct_narrow:669 design.tile_pipeline.phase.split.role.Main.info.verb.prepare; tcg_d128_segmented_cp_direct_wide:686 design.tile_pipeline.phase.split.role.Main.info.verb.prepare; tcg_d128_segmented_tma_direct_narrow:664 design.tile_pipeline.phase.split.role.Main.info.verb.prepare; tcg_d128_segmented_tma_direct_wide:683 design.tile_pipeline.phase.split.role.Main.info.verb.prepare

Prefetch Q/scales, then first-warp allocate 256 TMEM columns, CTA-publish the base and relinquish the permit. Write Query, QueryScale and UnitScale, wait for TMEM stores, discard QueryPrefetch and clear NumState. This also occurs for empty splits and precedes the scan wait, allowing independent work before scan completion.

## t_2a5c2c5d216c10bf | pipeline.verb | 351 chars | 1 declarations

tcg_d128_fscan_tma_xpose_wide:541 design.tile_pipeline.phase.split.role.Main.info.verb.setup

Establish coordinates and the head-major row-valid predicate; rowsLive is always true. Initialize QK, PV and two combined K/V transaction barriers with phase zero and publish their initialization. All four warps remain cooperative participants; each thread maps to one row. Thread zero issues TMA and MMA. The exact grid never takes q0>=Sq early exit.

## t_2aaeee94a9be8919 | pipeline.verb | 664 chars | 1 declarations

tcg_d128_fscan_tma_xpose_narrow:547 design.tile_pipeline.phase.split.role.Main.info.verb.prefix.meaning

Initialize DenState=0 and MaxState=-inf, then perform scan's bounded prefix, loading 16 GroupMax values into PrefixBatch before ordered updates and finishing with a scalar remainder. Consume each batch before reuse. All threads participate using domain's r, including padding. Load CurrentMax=GroupMax[cLo,r] only for Ng>0, otherwise keep -inf. CurrentMax/NextMax cache group maxima, not delayed states; no global snapshot or MTail is read. done covers all prefix work, but threads can continue with their own initialized state before other warps finish. The next stage or stage_partials CTA join covers collective completion. Empty splits also perform the prefix.

## t_2b3c1d6a44880dc6 | pipeline.verb | 243 chars | 4 declarations

tcg_d128_flat_tma_direct_narrow:584 design.tile_pipeline.phase.split.role.Main.info.verb.prefetch; tcg_d128_flat_tma_direct_wide:598 design.tile_pipeline.phase.split.role.Main.info.verb.prefetch; tcg_d128_segmented_tma_direct_narrow:674 design.tile_pipeline.phase.split.role.Main.info.verb.prefetch; tcg_d128_segmented_tma_direct_wide:693 design.tile_pipeline.phase.split.role.Main.info.verb.prefetch

Load NextMax for g+1 when g<cHi, otherwise use -inf, including in dead warps. Enqueue next K/V and atoms in the alternate slot only when g<cHi. Prior QK/PV completion retired that slot's old operand reads; done does not wait for new transfers.

## t_2ba5fc558e031b68 | pipeline.verb | 331 chars | 2 declarations

tcg_d128_fscan_cp_xpose_narrow:533 design.tile_pipeline.phase.split.role.Main.info.verb.setup; tcg_d128_fscan_cp_xpose_wide:550 design.tile_pipeline.phase.split.role.Main.info.verb.setup

Establish coordinates and the head-major row-valid predicate; rowsLive is always true. Initialize QK and PV completion barriers with phase zero. All four warps remain cooperative participants; each thread maps to one row. Each thread owns its copy-group stream; thread zero issues MMA. The exact grid never takes q0>=Sq early exit.

## t_2c5c2f9e196d88bd | pipeline.verb | 160 chars | 3 declarations

tcg_d256_segmented_cp_narrow:498 design.tile_pipeline.phase.segment_max.role.Main.info.verb.await; tcg_d256_segmented_tma_narrow:513 design.tile_pipeline.phase.segment_max.role.Main.info.verb.await; tcg_d256_segmented_tma_wide:498 design.tile_pipeline.phase.segment_max.role.Main.info.verb.await

Every thread performs the PDL dependency wait, proving GroupMax visible. Thread tid owns row r0+tid; out-of-range threads return only after both protocol calls.

## t_2c98e82d11ae9ae0 | pipeline.verb | 194 chars | 4 declarations

tcg_d128_fscan_cp_direct_narrow:624 design.tile_pipeline.phase.combine.role.Main.info.verb.dependency; tcg_d128_fscan_cp_direct_wide:641 design.tile_pipeline.phase.combine.role.Main.info.verb.dependency; tcg_d128_fscan_tma_direct_narrow:617 design.tile_pipeline.phase.combine.role.Main.info.verb.dependency; tcg_d128_fscan_tma_direct_wide:635 design.tile_pipeline.phase.combine.role.Main.info.verb.dependency

All threads execute the PDL wait at entry. done guarantees every PartA, PartB, PartM and MTail producer in the split grid has completed and its stores are visible, including an empty last split.

## t_2d099ce87c6ee7e6 | pipeline.verb | 594 chars | 2 declarations

tcg_d128_segmented_cp_direct_narrow:674 design.tile_pipeline.phase.split.role.Main.info.verb.replay; tcg_d128_segmented_cp_direct_wide:691 design.tile_pipeline.phase.split.role.Main.info.verb.replay

Initialize DenState to zero and MaxState to -inf. Every thread loads MState at (cLo-1)*R+rowG when cLo>0, including empty splits; load CurrentMax at cLo only when Ng>0. rowG is domain's r. The source has no any_compute guard or T-bound clamp on the entrance load: a logically out-of-range empty-split read is not given an initialization or safety guarantee here. Its loaded value is discarded from empty-split results; dead-warps' loads likewise do not affect output or subsequent addresses. After these loads, Ng>0 commits the empty copy group left by the hoisted prime; Ng=0 issues no commit.

## t_2dd077715265fb9f | pipeline.verb | 330 chars | 1 declarations

tcg_d256_segmented_cp_wide:820 design.tile_pipeline.phase.split.role.Main.info.verb.write

CTA-sync publishes the complete PartAStaging. All eight warps cooperatively read valid rows in contiguous 16-byte chunks and issue PartA stores, reversing the PSK row map as needed. Half zero writes PartB for valid rows. No O/LSE write occurs. done consumes the shared/register store sources; global writes can still be in flight.

## t_2e0056904c552f0c | pipeline.verb | 339 chars | 7 declarations

tcg_d256_flat_cp_narrow:596 design.tile_pipeline.phase.split.role.Main.info.verb.qk; tcg_d256_flat_cp_wide:610 design.tile_pipeline.phase.split.role.Main.info.verb.qk; tcg_d256_flat_tma_narrow:591 design.tile_pipeline.phase.split.role.Main.info.verb.qk; tcg_d256_flat_tma_wide:605 design.tile_pipeline.phase.split.role.Main.info.verb.qk; tcg_d256_segmented_cp_narrow:664 design.tile_pipeline.phase.split.role.Main.info.verb.qk; tcg_d256_segmented_tma_narrow:679 design.tile_pipeline.phase.split.role.Main.info.verb.qk; tcg_d256_segmented_tma_wide:664 design.tile_pipeline.phase.split.role.Main.info.verb.qk

Thread zero fences after thread-sync, issues eight score contractions and commits. All threads wait QK, advance parity and fence. Each half reads/masks its 64 scores in two locally completed TMEM loads. Wholly padded PSK warps skip readout but still wait; without PSK all warps read. Retain Score without computing or exchanging a maximum.

## t_2f96ae73833e17b6 | pipeline.verb | 204 chars | 1 declarations

fused_d128_group:481 design.tile_pipeline.phase.main.role.Even.info.verb.broadcast

A CTA barrier publishes Exchange's FinalMax and FinalDen, then both roles load them into their local RowState registers. They denote MaxState(cHi) and DenState(cHi); no reduction or re-association occurs.

## t_31be26ac1fb14603 | pipeline.verb | 438 chars | 2 declarations

fused_d128_cp:414 design.tile_pipeline.phase.main.role.Main.info.verb.finish; root:636 design.tile_pipeline.phase.main.role.Main.info.verb.finish

A before-thread-sync fence and CTA barrier join all final TMEM readers, then the first warp deallocates the 256-column region. No next worktile is assigned to this CTA. Neighboring CTAs have independent resources and may proceed while these ordinary output writes complete; kernel completion gives caller-stream visibility. Each subsequent invocation initializes fresh CTA state; host descriptor/graph caches do not carry numerical state.

## t_32ef99140147fe3d | pipeline.verb | 569 chars | 1 declarations

fused_d128_group:363 design.tile_pipeline.phase.main.role.Even.info.verb.publish

Query applies async-shared and before-thread-sync fences, then a CTA barrier publishes Q and shared scales to both roles, including an Odd role with no group. Operands(j) writes ValueAtom(g) into this role's ValueScale; wait::st, a before fence and a 128-thread role barrier publish Prob8, value scales and the numerator rescale to its issuer. Its previous PV was retired before these overwrites. Final writes MaxState(cHi) and DenState(cHi) to Exchange's FinalMax and FinalDen only for w=(Ng-1)%2; the other role completes the branch check without writing these slots.

## t_3353a5b7b845618f | pipeline.verb | 152 chars | 1 declarations

tcg_d128_fscan_tma_xpose_narrow:641 design.tile_pipeline.phase.combine.role.Main.info.verb.dependency

All threads execute the PDL wait at entry. done guarantees every PartA/PartB/PartM and MTail producer in split has completed and its writes are visible.

## t_34684d5d0144d206 | pipeline.verb | 458 chars | 1 declarations

tcg_d128_fscan_tma_xpose_wide:570 design.tile_pipeline.phase.split.role.Main.info.verb.stage

Wait for the current combined K/V transaction, flipping parity after every two-slot wrap. Fence shared async access and CTA-sync; cooperatively write current K/V scales into their fixed TMEM slots, wait for TMEM stores, then fence and CTA-sync. The preceding group already waited for PV, so ValueScale and the refilled V slot have no remaining matrix readers. This CTA join also retires previous GroupPV readout before QK reuses its overlapping score region.

## t_347a98f2ccb7dbc3 | pipeline.verb | 399 chars | 1 declarations

fused_d128_key:394 design.tile_pipeline.phase.main.role.Main.info.verb.lookahead

Next(g) lets thread 0 enqueue four QK contractions for g+1 and commit one QK phase after its after fence. Key(g+1) and its scale are already published. RawScore(g+1) occupies the other score tile; any prior g-1 readout retired in g-1. This MMA may execute during current probability work and PV issue. At g=cHi the action completes after the bound check without issuing an MMA or committing a phase.

## t_34e1acff40a28575 | pipeline.verb | 218 chars | 4 declarations

tcg_d256_flat_cp_narrow:461 design.tile_pipeline.phase.scan.role.Main.info.verb.begin; tcg_d256_flat_cp_wide:475 design.tile_pipeline.phase.scan.role.Main.info.verb.begin; tcg_d256_flat_tma_narrow:459 design.tile_pipeline.phase.scan.role.Main.info.verb.begin; tcg_d256_flat_tma_wide:473 design.tile_pipeline.phase.scan.role.Main.info.verb.begin

Trigger dependent launch before the recurrence; this permits split setup but does not publish scan data. Thread tid owns r0+tid. Out-of-range threads return after triggering; valid threads initialize ScanState to -inf.

## t_356a8fe555164c3a | pipeline.verb | 431 chars | 7 declarations

tcg_d256_flat_cp_narrow:435 design.tile_pipeline.phase.group_max.role.Main.info.verb.finish; tcg_d256_flat_cp_wide:449 design.tile_pipeline.phase.group_max.role.Main.info.verb.finish; tcg_d256_flat_tma_narrow:433 design.tile_pipeline.phase.group_max.role.Main.info.verb.finish; tcg_d256_flat_tma_wide:447 design.tile_pipeline.phase.group_max.role.Main.info.verb.finish; tcg_d256_segmented_cp_narrow:471 design.tile_pipeline.phase.group_max.role.Main.info.verb.finish; tcg_d256_segmented_tma_narrow:486 design.tile_pipeline.phase.group_max.role.Main.info.verb.finish; tcg_d256_segmented_tma_wide:471 design.tile_pipeline.phase.group_max.role.Main.info.verb.finish

Both halves publish their unchanged zero denominators over Exchange, CTA-sync and read the unused sum, including C=0. Prior maximum readers have retired. No V/probability or output access occurs. Every nonempty copy was consumed; empty commits carry no payload. Fence before thread-sync and CTA-sync, then the first warp deallocates 256 TMEM columns. Trigger programmatic launch completion at exit. No further worktile is assigned.

## t_35f10b2deac66d4b | pipeline.verb | 960 chars | 1 declarations

fused_d128_group:324 design.tile_pipeline.phase.main.role.Even.info.verb.setup

Even is warpgroup w=0 and Odd is w=1. Within either role, j is its local loop index and Group(j) means g=cLo+w+2*j. Each role computes all rows, keys and channels of its groups and shares the ordered NumState. Each of its 128 threads owns row u=tid mod 128. Both roles remain present for the full CTA lifetime, including padded rows. A CTA may have Ng=1 despite Span>=8; Odd then owns no group but executes its prologue and epilogue. Thread 0 initializes separate QK/PV barriers for both roles with one arrival each, two maximum barriers and four denominator barriers with 128 arrivals each. Each thread follows the listed row operations in program order; the async maximum/probability/denominator actions allow other warps in that role to lag. Their aggregate done events do not add a producer-side barrier. Initialize parity counters at zero and row states as in normalize. The first PV overwrites NumState without an earlier TMEM accumulator initialization.

## t_369603dc806b6725 | pipeline.verb | 635 chars | 2 declarations

tcg_d128_segmented_cp_direct_narrow:487 design.tile_pipeline.phase.group_max.role.Main.info.verb.stage; tcg_d128_segmented_cp_direct_wide:504 design.tile_pipeline.phase.group_max.role.Main.info.verb.stage

Every thread waits for all but its newest copy group, then fences shared async access and CTA-syncs. For Long this proves K(g+1) and atom when g<cHi; write its alternating KeyScale. At g=cHi the wait still executes and the next physical atom slot's stale bytes are still copied to the alternate scale slot, with no later QK consumer. Ng>=8 ensures that shared slot was filled earlier. For non-Long the wait proves current K(g) and atom; write KeyScale at its fixed base. Wait for TMEM stores, fence and CTA-sync. These joins also publish Q. Tail empty commits make all actual copies complete without a separate wait-for-zero operation.

## t_36ea1f1e964951f7 | pipeline.verb | 421 chars | 1 declarations

fused_d128_group:355 design.tile_pipeline.phase.main.role.Even.info.verb.prepare

Prefetch this role's Q pieces into QueryPrefetch; Even also loads the query scales, using the domain step's padding values. The first warp allocates TensorMemory; a CTA barrier publishes its base, then that warp relinquishes the permit. Both roles store their Q pieces to Query; Even alone writes QueryScale and UnitScale and waits for TMEM stores. QueryPrefetch is now dead; Q and these shared scales survive all groups.

## t_3731431188ef46db | pipeline.verb | 526 chars | 2 declarations

tcg_d128_flat_cp_direct_narrow:424 design.tile_pipeline.phase.group_max.role.Main.info.verb.start; tcg_d128_flat_cp_direct_wide:438 design.tile_pipeline.phase.group_max.role.Main.info.verb.start

For Ng>0 each thread commits an empty group at the original prime position; the first copies were already committed by prime. For Long also enqueue K(cLo+1) and atom into slot one and commit, then each thread waits until all but its newest group complete. This proves slot zero's payload and atom. Fence shared async access and CTA-sync, publish the first KeyScale, wait for TMEM stores, fence and CTA-sync, then enqueue and commit QK(cLo). Non-Long performs no copy wait or QK here; Ng=0 has no copy, commit or matrix effect.

## t_3785c674678cc3e7 | pipeline.verb | 571 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:715 design.tile_pipeline.phase.split.role.Main.info.verb.stage

Each thread waits until at most one copy group remains pending, proving the current K/V payload and both atoms; the last iteration's empty commit also makes its current data older than the newest group. Fence shared async access and CTA-sync; cooperatively write current K/V scales into their fixed TMEM slots, wait for TMEM stores, then fence and CTA-sync. The preceding group already waited for PV, so ValueScale and the refilled V slot have no remaining matrix readers. This CTA join also retires previous GroupPV readout before QK reuses its overlapping score region.

## t_37a3008be538b6f6 | pipeline.verb | 343 chars | 1 declarations

tcg_d128_fscan_cp_xpose_narrow:662 design.tile_pipeline.phase.combine.role.Main.info.verb.sum_a

Consume the first batch, then load and consume further batches of eight in order. Multiply each valid split's PartA by its shared Weight and accumulate as in combine; missing entries add zero without reading Weight. Register batches are reused only after consumption. The shared weights remain live through all channel and denominator readers.

## t_38de8dfaf98cad12 | pipeline.verb | 321 chars | 4 declarations

tcg_d128_segmented_cp_direct_narrow:535 design.tile_pipeline.phase.segment_max.role.Main.info.verb.produce; tcg_d128_segmented_cp_direct_wide:552 design.tile_pipeline.phase.segment_max.role.Main.info.verb.produce; tcg_d128_segmented_tma_direct_narrow:531 design.tile_pipeline.phase.segment_max.role.Main.info.verb.produce; tcg_d128_segmented_tma_direct_wide:550 design.tile_pipeline.phase.segment_max.role.Main.info.verb.produce

Each valid thread loads its segment's Seg entries into GroupBatch, substituting -inf beyond T, reduces them as segment_max specifies and issues one SegMax store. Out-of-range rows do no arithmetic or memory access after the entry protocol. There is no cross-thread reduction; kernel completion supplies global visibility.

## t_392bdf310948dfa2 | pipeline.verb | 421 chars | 2 declarations

tcg_d128_segmented_cp_direct_narrow:684 design.tile_pipeline.phase.split.role.Main.info.verb.stage; tcg_d128_segmented_cp_direct_wide:701 design.tile_pipeline.phase.split.role.Main.info.verb.stage

Every thread waits for all but its newest copy group, proving current K/V plus atoms. The final empty commit makes this wait retire the last payload. Fence shared async access and CTA-sync, write current KeyScale and ValueScale, wait for TMEM stores, fence and CTA-sync. Prior PV was already waited, protecting the scale/operand slots. These joins retire the previous GroupPV readers before QK overwrites the same region.

## t_39cdc5915de6d1a5 | pipeline.verb | 739 chars | 2 declarations

tcg_d128_flat_cp_xpose_narrow:439 design.tile_pipeline.phase.group_max.role.Main.info.verb.stage; tcg_d128_flat_cp_xpose_wide:422 design.tile_pipeline.phase.group_max.role.Main.info.verb.stage

Every iteration executes wait_group<1>, fences shared access and CTA-syncs. For Long this proves K(g+1) and its atom when g<cHi; write its alternating KeyScale, wait for TMEM stores, fence and CTA-sync. At g=cHi the same copy wait still executes, and the source copies the next physical atom slot's stale bytes to the alternate scale slot; no later QK consumes them. With Ng>=8 that shared slot has already been filled. For non-Long the wait proves current K(g) and atom; write KeyScale at its fixed base, wait for stores, fence and CTA-sync. Empty tail commits put the last real copy group before the one allowed outstanding. The CTA joins publish all threads' copies and Q; a per-thread copy wait alone does not provide that publication.

## t_3a02d5c529cf638f | pipeline.verb | 244 chars | 1 declarations

tcg_d256_segmented_cp_wide:526 design.tile_pipeline.phase.group_max.role.Main.info.verb.qk

Thread zero applies the after-thread-sync fence, issues the scores step's eight contractions into ScoreTile and commits QK. All threads wait for its parity completion, advance parity and fence. done proves scores readable and Q/K reads retired.

## t_3a04def1b32e668d | pipeline.verb | 328 chars | 4 declarations

tcg_d128_fscan_cp_xpose_narrow:415 design.tile_pipeline.phase.group_max.role.Main.info.verb.prepare; tcg_d128_fscan_cp_xpose_wide:432 design.tile_pipeline.phase.group_max.role.Main.info.verb.prepare; tcg_d128_fscan_tma_xpose_wide:436 design.tile_pipeline.phase.group_max.role.Main.info.verb.prepare; tcg_d128_segmented_cp_xpose_wide:502 design.tile_pipeline.phase.group_max.role.Main.info.verb.prepare

Load Q payload and scale words into QueryPrefetch registers even for empty splits. First warp allocates TMEM; a CTA barrier publishes its base before that warp relinquishes the permit. Write Query and QueryScale, wait for TMEM stores, then discard QueryPrefetch. Query persists through all groups. No V or P operand is prepared.

## t_3a13fef255be476b | pipeline.verb | 332 chars | 4 declarations

tcg_d128_fscan_cp_direct_narrow:543 design.tile_pipeline.phase.split.role.Main.info.verb.prepare; tcg_d128_fscan_cp_direct_wide:560 design.tile_pipeline.phase.split.role.Main.info.verb.prepare; tcg_d128_fscan_tma_direct_narrow:537 design.tile_pipeline.phase.split.role.Main.info.verb.prepare; tcg_d128_fscan_tma_direct_wide:555 design.tile_pipeline.phase.split.role.Main.info.verb.prepare

Prefetch Q/scales, then first-warp allocate 256 TMEM columns, CTA-publish the base and relinquish the permit. Write Query, QueryScale and UnitScale, wait for TMEM stores, discard QueryPrefetch and clear NumState. This also occurs for empty splits and precedes the GroupMax wait, allowing independent work before group_max completes.

## t_3a9ce1980888a8f9 | pipeline.verb | 345 chars | 1 declarations

tc_split_flat_wide:658 design.tile_pipeline.phase.combine.role.Main.info.verb.entry

Wait for grid dependencies before reading partials. The ordinary split predecessor has no early trigger, so delivery requires its completion. Consecutive sets of four warps form group f=0,...,3. Its thread d0=0,...,127 owns channel d0 and also d0+128 when D=256. Initialize GroupAccum to zero. The exact row grid needs no row guard or D slicing.

## t_3aba100e028a7bb3 | pipeline.verb | 182 chars | 12 declarations

tcg_d128_flat_tma_direct_narrow:572 design.tile_pipeline.phase.split.role.Main.info.verb.prime; tcg_d128_flat_tma_direct_wide:586 design.tile_pipeline.phase.split.role.Main.info.verb.prime; tcg_d128_flat_tma_xpose_narrow:567 design.tile_pipeline.phase.split.role.Main.info.verb.prime; tcg_d128_flat_tma_xpose_wide:585 design.tile_pipeline.phase.split.role.Main.info.verb.prime; tcg_d128_fscan_tma_direct_narrow:535 design.tile_pipeline.phase.split.role.Main.info.verb.prime; tcg_d128_fscan_tma_direct_wide:553 design.tile_pipeline.phase.split.role.Main.info.verb.prime; tcg_d128_fscan_tma_xpose_narrow:531 design.tile_pipeline.phase.split.role.Main.info.verb.prime; tcg_d128_segmented_cp_xpose_narrow:638 design.tile_pipeline.phase.split.role.Main.info.verb.prime; tcg_d128_segmented_tma_direct_narrow:662 design.tile_pipeline.phase.split.role.Main.info.verb.prime; tcg_d128_segmented_tma_direct_wide:681 design.tile_pipeline.phase.split.role.Main.info.verb.prime; tcg_d128_segmented_tma_xpose_narrow:654 design.tile_pipeline.phase.split.role.Main.info.verb.prime; tcg_d128_segmented_tma_xpose_wide:638 design.tile_pipeline.phase.split.role.Main.info.verb.prime

For Ng>0 enqueue K/V(cLo) and their atoms into slot zero. One transaction barrier accounts for 2*(16384+512) bytes per slot. For Ng=0 issue no transfer. done is enqueue, not arrival.

## t_3b5561830de68997 | pipeline.verb | 425 chars | 7 declarations

tcg_d256_flat_cp_narrow:427 design.tile_pipeline.phase.group_max.role.Main.info.verb.maximum; tcg_d256_flat_cp_wide:441 design.tile_pipeline.phase.group_max.role.Main.info.verb.maximum; tcg_d256_flat_tma_narrow:425 design.tile_pipeline.phase.group_max.role.Main.info.verb.maximum; tcg_d256_flat_tma_wide:439 design.tile_pipeline.phase.group_max.role.Main.info.verb.maximum; tcg_d256_segmented_cp_narrow:463 design.tile_pipeline.phase.group_max.role.Main.info.verb.maximum; tcg_d256_segmented_tma_narrow:478 design.tile_pipeline.phase.group_max.role.Main.info.verb.maximum; tcg_d256_segmented_tma_wide:463 design.tile_pipeline.phase.group_max.role.Main.info.verb.maximum

Read each half's 64 scores in two 32-column chunks. An unmasked warp uses native load-reduce-max; a masked warp loads, waits, masks and reduces locally. Only wholly padded PSK warps skip readout; all warps still perform exchange and barriers. Write HalfMax, CTA-sync, read both halves, and have half zero store GroupMax for valid rows. A final CTA barrier retires all ScoreTile and Exchange readers before the next overwrite.

## t_3b7bea205b7e864f | pipeline.verb | 319 chars | 8 declarations

tcg_d128_flat_cp_xpose_narrow:449 design.tile_pipeline.phase.group_max.role.Main.info.verb.reduce; tcg_d128_flat_cp_xpose_wide:432 design.tile_pipeline.phase.group_max.role.Main.info.verb.reduce; tcg_d128_flat_tma_xpose_narrow:441 design.tile_pipeline.phase.group_max.role.Main.info.verb.reduce; tcg_d128_flat_tma_xpose_wide:459 design.tile_pipeline.phase.group_max.role.Main.info.verb.reduce; tcg_d128_fscan_tma_xpose_narrow:432 design.tile_pipeline.phase.group_max.role.Main.info.verb.reduce; tcg_d128_segmented_cp_xpose_narrow:482 design.tile_pipeline.phase.group_max.role.Main.info.verb.reduce; tcg_d128_segmented_tma_xpose_narrow:498 design.tile_pipeline.phase.group_max.role.Main.info.verb.reduce; tcg_d128_segmented_tma_xpose_wide:482 design.tile_pipeline.phase.group_max.role.Main.info.verb.reduce

All warps reduce scores in 32-key chunks, using masked readout or native unmasked max readout, each with its TMEM-load wait. Only valid rows issue GroupMax stores. A CTA barrier joins all score readers before the next QK overwrites DotTile; no full-group score register array, PV or cross-warpgroup reduction is needed.

## t_3b9f97d638e8f73a | pipeline.verb | 239 chars | 4 declarations

tcg_d128_fscan_cp_xpose_narrow:578 design.tile_pipeline.phase.split.role.Main.info.verb.accumulate; tcg_d128_fscan_cp_xpose_wide:595 design.tile_pipeline.phase.split.role.Main.info.verb.accumulate; tcg_d128_fscan_tma_xpose_wide:583 design.tile_pipeline.phase.split.role.Main.info.verb.accumulate; tcg_d128_segmented_cp_xpose_wide:729 design.tile_pipeline.phase.split.role.Main.info.verb.accumulate

All warps read GroupPV in 32-channel chunks with TMEM-load waits and add to register NumState. There is no tail CTA barrier: the next stage action's CTA joins protect the next TMEM overwrite; the last iteration joins during stage_partials.

## t_3c765620295c283b | pipeline.verb | 556 chars | 1 declarations

fused_d256_cp:377 design.tile_pipeline.phase.main.role.Main.info.verb.qk

Thread 0 applies its after-thread-sync fence, issues the scores step's eight contractions into ScoreTile, the first overwriting, and commits QK completion. All threads wait, advance parity once and apply the completion fence. Each half reads its 64 columns in two 32-column loads with load-completion waits, masks and forms HalfMax, retaining Score for probabilities. done retires Key/KeyScale reads and local ScoreTile reads. Previous-group ScoreTile readers finished before that group's probability barrier, so the current product can overwrite the tile.

## t_3c825558bdc34ebc | pipeline.verb | 466 chars | 2 declarations

tcg_d128_fscan_cp_direct_narrow:414 design.tile_pipeline.phase.group_max.role.Main.info.verb.start; tcg_d128_fscan_cp_direct_wide:431 design.tile_pipeline.phase.group_max.role.Main.info.verb.start

For Ng>0 every thread commits one empty copy group after Q preparation. For Long additionally enqueue K(cLo+1) and atom, commit, and wait for all but the newest committed group, proving slot zero's copies complete for each issuer. Fence shared access and CTA-sync to publish them and Q, write the first KeyScale, wait for its TMEM stores, fence and CTA-sync, then enqueue and commit QK(cLo). Non-Long performs no copy wait or QK here; Ng=0 performs no commit either.

## t_3d398f3ddcd3eee7 | pipeline.verb | 648 chars | 1 declarations

fused_d256_cp:342 design.tile_pipeline.phase.main.role.Main.info.verb.prime

After prepare, enqueue K(cLo) and V(cLo) into slot zero with predicated 16-byte cp.async copies, zero-filling padded keys. Each payload is 32768 bytes. Threads 0..63 also enqueue the 1024-byte K and V atoms for groups cLo through min(cLo+2,cHi) into their three-slot rings. Commit one group per thread, wait_group 0, fence the shared async proxy and CTA-sync. This completes all prime copies and publishes Query and the atoms. Warps 0..3 copy cLo's atoms to KeyScale/ValueScale and wait for TMEM stores. A before-thread-sync fence and CTA barrier publish these scales. No QK is issued here. The fixed default does not hoist the prime above prepare.

## t_3d75b872dca3e190 | pipeline.verb | 416 chars | 1 declarations

fused_d128_group:387 design.tile_pipeline.phase.main.role.Even.info.verb.stage

Ahead(j) issues K(g+4) and its atom into local slot (j+2)%3 when g+4 exists. Always commit, then wait_group 1, an async-shared fence and role barrier establish K(g+2) when present and V(g) readiness; the newly issued K may remain in flight. On reuse this K slot previously held g-2, whose QK and scale-atom reads ended in the prior local iteration. This dedicated-ring proof does not retire cold's cross-role writes.

## t_3d9a3c9b63ac6d1a | pipeline.verb | 167 chars | 11 declarations

tcg_d128_flat_cp_direct_narrow:486 design.tile_pipeline.phase.flat_scan.role.Main.info.verb.tail; tcg_d128_flat_cp_direct_wide:500 design.tile_pipeline.phase.flat_scan.role.Main.info.verb.tail; tcg_d128_flat_cp_xpose_narrow:495 design.tile_pipeline.phase.flat_scan.role.Main.info.verb.tail; tcg_d128_flat_cp_xpose_wide:478 design.tile_pipeline.phase.flat_scan.role.Main.info.verb.tail; tcg_d128_flat_tma_direct_narrow:482 design.tile_pipeline.phase.flat_scan.role.Main.info.verb.tail; tcg_d128_flat_tma_direct_wide:496 design.tile_pipeline.phase.flat_scan.role.Main.info.verb.tail; tcg_d128_flat_tma_xpose_narrow:486 design.tile_pipeline.phase.flat_scan.role.Main.info.verb.tail; tcg_d128_flat_tma_xpose_wide:504 design.tile_pipeline.phase.flat_scan.role.Main.info.verb.tail; tcg_d128_segmented_cp_xpose_narrow:557 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.tail; tcg_d128_segmented_tma_xpose_narrow:573 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.tail; tcg_d128_segmented_tma_xpose_wide:557 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.tail

Issue MTail stores for valid rows after their last snapshot. All R*T MState and R MTail entries are overwritten each invocation; kernel completion provides visibility.

## t_3e35cc99a20bf8b9 | pipeline.verb | 421 chars | 1 declarations

root:548 design.tile_pipeline.phase.main.role.Main.info.verb.prepare

Prefetch this tile's Q payload and scale words into QueryPrefetch registers before allocating TMEM. The first warp allocates; a CTA barrier publishes the base, then that warp relinquishes its allocation permit. Write Query and QueryScale/UnitScale, wait for these TMEM stores, then discard QueryPrefetch. Initialize MaxState and DenState as in math_reshape; do not clear NumState. Query and its scales survive all groups.

## t_3e87a784a4907d6f | pipeline.verb | 347 chars | 5 declarations

tc_fused:329 design.tile_pipeline.phase.main.role.Main.info.verb.decode; tc_split_flat_narrow:587 design.tile_pipeline.phase.split.role.Main.info.verb.decode; tc_split_flat_wide:604 design.tile_pipeline.phase.split.role.Main.info.verb.decode; tc_split_segmented_narrow:694 design.tile_pipeline.phase.split.role.Main.info.verb.decode; tc_split_segmented_wide:728 design.tile_pipeline.phase.split.role.Main.info.verb.decode

Decode KeyAtom to KeyScale, zeroing j>=Skv entries. Decode ValueAtom to packed BF16 ValueScale, including padding scales. Publish scales at a CTA barrier, then convert and scale Vraw into Vbf and publish it at another CTA barrier. The preceding group-end barrier protects the single Vbf and decoded-scale slots against overwriting earlier readers.

## t_3f279bcbf78bc520 | pipeline.verb | 256 chars | 1 declarations

fused_d256_cp:387 design.tile_pipeline.phase.main.role.Main.info.verb.exchange

Each half writes HalfMax into Exchange. A CTA barrier joins all score readers and publishes both halves; each thread reads them and forms the group maximum. advance's CTA barrier retires all Exchange readers before its next overwrite. No next QK is issued.

## t_3f86ee3058ee7b99 | pipeline.verb | 321 chars | 2 declarations

tcg_d128_flat_cp_xpose_narrow:600 design.tile_pipeline.phase.split.role.Main.info.verb.prefetch; tcg_d128_flat_cp_xpose_wide:583 design.tile_pipeline.phase.split.role.Main.info.verb.prefetch

Load NextMax for g+1 when g<cHi, otherwise use -inf, including for padded rows. Enqueue next K/V and atoms in the alternate slot only when g<cHi. Prior QK/PV completion retired that slot's old operand reads. Every thread commits once, including an empty commit on the last iteration; done does not wait for new transfers.

## t_3fee0ab944001c25 | pipeline.verb | 423 chars | 2 declarations

tcg_d128_flat_cp_direct_narrow:445 design.tile_pipeline.phase.group_max.role.Main.info.verb.finish; tcg_d128_flat_cp_direct_wide:459 design.tile_pipeline.phase.group_max.role.Main.info.verb.finish

All nonempty copy groups and QK products have been waited; final empty commits advance the wait without another payload. No extra wait-all is issued. Fence and CTA-sync join final TMEM users, then the first warp deallocates 256 columns. Each CTA signals PDL launch completion after its stores and teardown; launch_ready is the grid-wide protocol event. Global visibility comes from kernel completion and the successor wait.

## t_404508d47413cfae | pipeline.verb | 295 chars | 20 declarations

tcg_d128_flat_cp_direct_narrow:437 design.tile_pipeline.phase.group_max.role.Main.info.verb.score; tcg_d128_flat_cp_direct_wide:451 design.tile_pipeline.phase.group_max.role.Main.info.verb.score; tcg_d128_flat_cp_xpose_narrow:446 design.tile_pipeline.phase.group_max.role.Main.info.verb.score; tcg_d128_flat_cp_xpose_wide:429 design.tile_pipeline.phase.group_max.role.Main.info.verb.score; tcg_d128_flat_tma_direct_narrow:434 design.tile_pipeline.phase.group_max.role.Main.info.verb.score; tcg_d128_flat_tma_direct_wide:448 design.tile_pipeline.phase.group_max.role.Main.info.verb.score; tcg_d128_flat_tma_xpose_narrow:438 design.tile_pipeline.phase.group_max.role.Main.info.verb.score; tcg_d128_flat_tma_xpose_wide:456 design.tile_pipeline.phase.group_max.role.Main.info.verb.score; tcg_d128_fscan_cp_direct_narrow:428 design.tile_pipeline.phase.group_max.role.Main.info.verb.score; tcg_d128_fscan_cp_direct_wide:445 design.tile_pipeline.phase.group_max.role.Main.info.verb.score; tcg_d128_fscan_tma_direct_narrow:423 design.tile_pipeline.phase.group_max.role.Main.info.verb.score; tcg_d128_fscan_tma_direct_wide:441 design.tile_pipeline.phase.group_max.role.Main.info.verb.score; tcg_d128_fscan_tma_xpose_narrow:429 design.tile_pipeline.phase.group_max.role.Main.info.verb.score; tcg_d128_segmented_cp_direct_narrow:493 design.tile_pipeline.phase.group_max.role.Main.info.verb.score; tcg_d128_segmented_cp_direct_wide:510 design.tile_pipeline.phase.group_max.role.Main.info.verb.score; tcg_d128_segmented_cp_xpose_narrow:479 design.tile_pipeline.phase.group_max.role.Main.info.verb.score; tcg_d128_segmented_tma_direct_narrow:489 design.tile_pipeline.phase.group_max.role.Main.info.verb.score; tcg_d128_segmented_tma_direct_wide:508 design.tile_pipeline.phase.group_max.role.Main.info.verb.score; tcg_d128_segmented_tma_xpose_narrow:495 design.tile_pipeline.phase.group_max.role.Main.info.verb.score; tcg_d128_segmented_tma_xpose_wide:479 design.tile_pipeline.phase.group_max.role.Main.info.verb.score

Non-Long enqueues and commits QK(g) here; Long uses the product already issued by start/advance. Wait for its QK completion phase, toggle parity and apply the completion fence. K and KeyScale reads have retired and DotTile is readable; each group's first contraction overwrites the score region.

## t_4082a181fcb4304c | pipeline.verb | 191 chars | 2 declarations

tc_split_flat_narrow:524 design.tile_pipeline.phase.scan.role.Main.info.verb.replay; tc_split_flat_wide:541 design.tile_pipeline.phase.scan.role.Main.info.verb.replay

Consume MaxBatch in increasing group order through the scan step, updating ScanCarry and storing each ScanState entry. All batch reads finish before the next load overwrites this single slot.

## t_40b3d21b509f77cc | pipeline.verb | 330 chars | 1 declarations

fused_d128_group:436 design.tile_pipeline.phase.main.role.Even.info.verb.probability.meaning

Apply normalize to retained Score, store all 128 rounded probabilities per row into this role's Prob8 tile, and retain the two unrounded sum chains for the denominator update. Score is now dead; Operands(j) publishes the TMEM stores. The role's previous PV has retired; the other role's live PV reads a different probability tile.

## t_417f5003cbbbb477 | pipeline.verb | 243 chars | 4 declarations

tcg_d256_flat_cp_narrow:474 design.tile_pipeline.phase.scan.role.Main.info.verb.finish; tcg_d256_flat_cp_wide:488 design.tile_pipeline.phase.scan.role.Main.info.verb.finish; tcg_d256_flat_tma_narrow:472 design.tile_pipeline.phase.scan.role.Main.info.verb.finish; tcg_d256_flat_tma_wide:486 design.tile_pipeline.phase.scan.role.Main.info.verb.finish

Valid threads write Mtail and discard ScanState. No CTA barrier or exit trigger follows; begin already granted permission. Kernel completion, observed by split's data wait, publishes all results. No local resource identity crosses this launch.

## t_41b67de0006272fc | pipeline.verb | 262 chars | 5 declarations

tc_fused:323 design.tile_pipeline.phase.main.role.Main.info.verb.land; tc_split_flat_narrow:584 design.tile_pipeline.phase.split.role.Main.info.verb.land; tc_split_flat_wide:601 design.tile_pipeline.phase.split.role.Main.info.verb.land; tc_split_segmented_narrow:691 design.tile_pipeline.phase.split.role.Main.info.verb.land; tc_split_segmented_wide:725 design.tile_pipeline.phase.split.role.Main.info.verb.land

Wait until only the newest committed group may remain pending, then join the CTA. Current K/V payloads and atoms become readable to all warps while next-group copies may continue. On the last iteration the newest commit is empty; no final wait_group<0> is added.

## t_42120ca178dd8cc7 | pipeline.verb | 148 chars | 4 declarations

tcg_d128_segmented_cp_direct_narrow:557 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.dependency; tcg_d128_segmented_cp_direct_wide:574 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.dependency; tcg_d128_segmented_tma_direct_narrow:553 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.dependency; tcg_d128_segmented_tma_direct_wide:572 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.dependency

All threads wait for segment_max completion before signaling or testing the row guard. done guarantees SegMax and transitively GroupMax are visible.

## t_425a20db04aa9751 | pipeline.verb | 576 chars | 1 declarations

fused_d128_cp:403 design.tile_pipeline.phase.main.role.Main.info.verb.write

The final per-group retirement already completed PV, and every requested K/V payload and atom has been waited and consumed. Read NumState in 32-channel chunks, each with its TMEM-load wait. Valid rows issue scaled BF16 RNE stores directly to caller O and, when present, one FP32 LSE store using value_output. Initial destination bytes are never read. Padding still participates in computation/readout but issues no output store. done means these ordinary stores have been issued and their register sources consumed; it does not introduce a device-wide or host completion wait.

## t_4441d3f65710dc8f | pipeline.verb | 790 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:691 design.tile_pipeline.phase.split.role.Main.info.verb.prepare.meaning

Prefetch Q and scale words into QueryPrefetch, allocate 256 TMEM columns with the first warp, publish the base by CTA barrier and relinquish the permit. Each thread writes Query, QueryScale and UnitScale, waits for its TMEM stores, discards QueryPrefetch and clears its numerator before its dependency/replay work. These ordinary shared Q writes are not followed by a CTA join here. done guarantees completion of preparation across all warps; later actions may start before that collective completion. The first stage CTA join proves it for Ng>0; for Ng=0 only the barrier after the stage_partials writes proves it. The asynchronous completion expresses this cross-warp progression, not a different copy instruction. Preparation also occurs for empty splits and can precede scan completion.

## t_4481a674a22aab90 | pipeline.verb | 350 chars | 1 declarations

tcg_d128_fscan_tma_xpose_wide:662 design.tile_pipeline.phase.combine.role.Main.info.verb.dependency

All threads execute the PDL dependency wait at kernel entry. done guarantees every PartA, PartB, PartM and MTail producer in split has completed and its global writes are visible. Group z consists of four consecutive warps, covers all D channels and consumes the combine step's interval slo(z)..shi(z)-1. No D slicing or tensor-core work occurs here.

## t_450382bd4cc68d3e | pipeline.verb | 200 chars | 2 declarations

tc_split_flat_narrow:516 design.tile_pipeline.phase.scan.role.Main.info.verb.initialize; tc_split_flat_wide:533 design.tile_pipeline.phase.scan.role.Main.info.verb.initialize

Each thread owns one row and initializes ScanCarry to -inf when valid. Padding threads perform the entry protocol and have no later numerical effects; all listed role actions retain issue/done events.

## t_453b038c069aad83 | pipeline.verb | 608 chars | 1 declarations

fused_d128_key:368 design.tile_pipeline.phase.main.role.Main.info.verb.atoms

Upcoming(g) reads next slot (g+1-cLo)%4's K atom into KeyScale(g+1), and ValueAtom(g) into ValueScale(g). Only warps 0..3 perform these replicated writes; wait::st, a before fence and CTA barrier publish them. Alternating TMEM scale slots avoid QK(g) and PV(g-1), which may still read the other parity. Where prior users exist, the overwritten slots' last readers were QK(g-1) and PV(g-2), already retired. At g=cHi the next K-atom read/write still executes, using stale or never-initialized shared bytes; no QK consumes that terminal KeyScale value. No initialization or out-of-range global copy is implied.

## t_458411389b5d0f49 | pipeline.verb | 439 chars | 1 declarations

tcg_d128_segmented_cp_direct_wide:777 design.tile_pipeline.phase.combine.role.Main.info.verb.write

Group zero alone reads the shared partials and performs combine's final sums. Each thread retains TotalA and a replicated TotalB, then writes its BF16 channel; thread zero also reads MTail and writes LSE when present. Other groups perform no further data accesses. Shared partials remain unchanged until every group-zero reader finishes. Initial destinations are never read; kernel completion supplies visibility. No next tile is assigned.

## t_45be81ebc4d8f713 | pipeline.verb | 272 chars | 12 declarations

tcg_d128_flat_tma_direct_narrow:423 design.tile_pipeline.phase.group_max.role.Main.info.verb.start; tcg_d128_flat_tma_direct_wide:437 design.tile_pipeline.phase.group_max.role.Main.info.verb.start; tcg_d128_flat_tma_xpose_narrow:427 design.tile_pipeline.phase.group_max.role.Main.info.verb.start; tcg_d128_flat_tma_xpose_wide:445 design.tile_pipeline.phase.group_max.role.Main.info.verb.start; tcg_d128_fscan_tma_direct_narrow:412 design.tile_pipeline.phase.group_max.role.Main.info.verb.start; tcg_d128_fscan_tma_direct_wide:430 design.tile_pipeline.phase.group_max.role.Main.info.verb.start; tcg_d128_fscan_tma_xpose_narrow:418 design.tile_pipeline.phase.group_max.role.Main.info.verb.start; tcg_d128_segmented_cp_xpose_narrow:468 design.tile_pipeline.phase.group_max.role.Main.info.verb.start; tcg_d128_segmented_tma_direct_narrow:478 design.tile_pipeline.phase.group_max.role.Main.info.verb.start; tcg_d128_segmented_tma_direct_wide:497 design.tile_pipeline.phase.group_max.role.Main.info.verb.start; tcg_d128_segmented_tma_xpose_narrow:484 design.tile_pipeline.phase.group_max.role.Main.info.verb.start; tcg_d128_segmented_tma_xpose_wide:468 design.tile_pipeline.phase.group_max.role.Main.info.verb.start

For Long enqueue K(cLo+1) and atom, wait for slot zero, fence shared async access and CTA-sync. Publish the first KeyScale, wait for its TMEM stores, fence and CTA-sync, then enqueue and commit QK(cLo). For non-Long this action has no data effect, transfer or matrix wait.

## t_45e033297a079637 | pipeline.verb | 448 chars | 2 declarations

tcg_d128_fscan_tma_direct_narrow:542 design.tile_pipeline.phase.split.role.Main.info.verb.prefix; tcg_d128_fscan_tma_direct_wide:560 design.tile_pipeline.phase.split.role.Main.info.verb.prefix

Initialize DenState to zero and PrefixState to -inf. All threads, including dead warps, walk GroupMax[0:Tw(s),r] in ordered batches of 16 followed by a scalar remainder. Consume PrefixBatch before reuse. Empty splits perform the same bounded walk. MaxState continues in the same register as PrefixState; for Ng>0 load CurrentGroupMax at cLo, otherwise use -inf. This work can overlap primed K/V transfers. No global mstate array is read or written.

## t_466b565607c1c213 | pipeline.verb | 254 chars | 1 declarations

tc_split_segmented_wide:799 design.tile_pipeline.phase.combine.role.Main.info.verb.publish

Execute the sole CTA barrier, joining all sixteen warps and publishing GroupPartials, meaning all GroupFinalA and GroupB entries. These buffers are written once and remain valid until the final group-zero readers finish; no next worktile overwrites them.

## t_4686f0406f7b3140 | pipeline.verb | 181 chars | 8 declarations

tcg_d128_flat_cp_direct_narrow:582 design.tile_pipeline.phase.split.role.Main.info.verb.dependency; tcg_d128_flat_cp_direct_wide:596 design.tile_pipeline.phase.split.role.Main.info.verb.dependency; tcg_d128_flat_tma_direct_narrow:577 design.tile_pipeline.phase.split.role.Main.info.verb.dependency; tcg_d128_flat_tma_direct_wide:591 design.tile_pipeline.phase.split.role.Main.info.verb.dependency; tcg_d128_segmented_cp_direct_narrow:672 design.tile_pipeline.phase.split.role.Main.info.verb.dependency; tcg_d128_segmented_cp_direct_wide:689 design.tile_pipeline.phase.split.role.Main.info.verb.dependency; tcg_d128_segmented_tma_direct_narrow:667 design.tile_pipeline.phase.split.role.Main.info.verb.dependency; tcg_d128_segmented_tma_direct_wide:686 design.tile_pipeline.phase.split.role.Main.info.verb.dependency

Execute the PDL scan-dependency wait before the first snapshot load. done guarantees the scan's MState and MTail regions are visible; it does not initialize or extend either region.

## t_46d5d64a1beec8ca | pipeline.verb | 312 chars | 1 declarations

tcg_d256_segmented_cp_wide:636 design.tile_pipeline.phase.scan.role.Main.info.verb.classify

Valid threads load the chunk's SegChunk maxima into SegmentBatch before any threshold test; use -inf for absent segments. Compare all entries to the same incoming ScanState and capture Jump as a per-row mask, then tp. All loaded maxima are consumed before entering the replay helper; no state update occurs here.

## t_47ae7969cdbba4df | pipeline.verb | 185 chars | 10 declarations

tcg_d128_flat_cp_direct_narrow:663 design.tile_pipeline.phase.combine.role.Main.info.verb.sum_b; tcg_d128_flat_cp_xpose_narrow:691 design.tile_pipeline.phase.combine.role.Main.info.verb.sum_b; tcg_d128_flat_cp_xpose_wide:674 design.tile_pipeline.phase.combine.role.Main.info.verb.sum_b; tcg_d128_flat_tma_direct_narrow:656 design.tile_pipeline.phase.combine.role.Main.info.verb.sum_b; tcg_d128_flat_tma_xpose_narrow:679 design.tile_pipeline.phase.combine.role.Main.info.verb.sum_b; tcg_d128_segmented_cp_direct_narrow:754 design.tile_pipeline.phase.combine.role.Main.info.verb.sum_b; tcg_d128_segmented_cp_xpose_narrow:750 design.tile_pipeline.phase.combine.role.Main.info.verb.sum_b; tcg_d128_segmented_tma_direct_narrow:746 design.tile_pipeline.phase.combine.role.Main.info.verb.sum_b; tcg_d128_segmented_tma_xpose_narrow:766 design.tile_pipeline.phase.combine.role.Main.info.verb.sum_b; tcg_d128_segmented_tma_xpose_wide:750 design.tile_pipeline.phase.combine.role.Main.info.verb.sum_b

Each thread independently performs combine's ordered sum over DenParts, unchanged until all readers finish; there is no parallel reduction tree. TotalB is replicated per output channel.

## t_47e64510a96930a3 | pipeline.verb | 256 chars | 1 declarations

tcg_d128_fscan_cp_xpose_narrow:667 design.tile_pipeline.phase.combine.role.Main.info.verb.write

Write the owned BF16 output channel; thread zero additionally reads MTail and writes LSE when present. Caller destinations are not read. Kernel completion supplies visibility, and there is no subsequent local worktile or additional launch permission event.

## t_48bba37bd02e41df | pipeline.verb | 754 chars | 5 declarations

tcg_d256_flat_tma_narrow:545 design.tile_pipeline.phase.split.role.Main.info.verb.prepare; tcg_d256_flat_tma_wide:559 design.tile_pipeline.phase.split.role.Main.info.verb.prepare; tcg_d256_segmented_cp_narrow:618 design.tile_pipeline.phase.split.role.Main.info.verb.prepare; tcg_d256_segmented_tma_narrow:633 design.tile_pipeline.phase.split.role.Main.info.verb.prepare; tcg_d256_segmented_tma_wide:618 design.tile_pipeline.phase.split.role.Main.info.verb.prepare

Establish work coordinates. AlignOffset is the 0..1023-byte displacement from raw shared base to its next 1024-byte boundary; OperandScratch covers the union of operand and epilogue views. Initialize two combined K/V transaction stages and QK/PV protocols at parity zero, publishing initialization before use. Prefetch Q/scales before the first warp allocates 512 TMEM columns; CTA-sync publishes the base, then that warp relinquishes its permit. Write Query cooperatively and QueryScale/UnitScale with warps 0..3, waiting for stores; QueryPrefetch then dies. Half z owns all Br rows, keys [64*z,64*z+64) and output channels [128*z,128*z+128). Preparation reads no scan result and issues no K/V prime, even for nonempty work. The q0>=Sq guard is untaken.

## t_492e5d9225af4fce | pipeline.verb | 605 chars | 4 declarations

tc_split_flat_narrow:573 design.tile_pipeline.phase.split.role.Main.info.verb.prepare; tc_split_flat_wide:590 design.tile_pipeline.phase.split.role.Main.info.verb.prepare; tc_split_segmented_narrow:680 design.tile_pipeline.phase.split.role.Main.info.verb.prepare; tc_split_segmented_wide:714 design.tile_pipeline.phase.split.role.Main.info.verb.prepare

Stage Qraw and QueryScale, zero invalid rows and publish at a CTA barrier. Initialize NumState and DenPart to zero. This also occurs on empty intervals. Whenever cLo>0, the source reads (cLo-1)*R+rho relative to the ScanState base into an initial local maximum. The first replay overwrites that value, or an empty interval leaves it unused. If cLo-1>=T this discarded access reaches neighboring allocated scratch, not a defined ScanState element. ScanState is followed by aligned MaxTail and partial regions; no initialization or extra wait is added for this access. Q and QueryScale survive the interval.

## t_49c73818cfe9b697 | pipeline.verb | 229 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:813 design.tile_pipeline.phase.combine.role.Main.info.verb.write

The first group writes all D BF16 channels; its first thread additionally reads MTail and writes LSE when present. Caller destinations are not read. Kernel completion supplies visibility, with no further launch-permission signal.

## t_4a29b8aca0b21faa | pipeline.verb | 470 chars | 5 declarations

tcg_d256_flat_tma_narrow:412 design.tile_pipeline.phase.group_max.role.Main.info.verb.land; tcg_d256_flat_tma_wide:426 design.tile_pipeline.phase.group_max.role.Main.info.verb.land; tcg_d256_segmented_cp_narrow:450 design.tile_pipeline.phase.group_max.role.Main.info.verb.land; tcg_d256_segmented_tma_narrow:465 design.tile_pipeline.phase.group_max.role.Main.info.verb.land; tcg_d256_segmented_tma_wide:450 design.tile_pipeline.phase.group_max.role.Main.info.verb.land

Issue K(g+1) and its atom into the other slot when present. Commit a cp.async group even at the tail, then wait_group 1. Wait current TMA, slot (g-l)%2, parity floor((g-l)/2)%2; apply the shared async-proxy fence and CTA-sync. Warps 0..3 copy current atoms to KeyScale, wait for TMEM stores, then fence before thread-sync and CTA-sync again. Current operands are ready; previous QK and its final barrier retired old tile/atom/scale readers before conflicting slot reuse.

## t_4af9019b66807ae1 | pipeline.verb | 359 chars | 1 declarations

tcg_d128_fscan_cp_direct_wide:650 design.tile_pipeline.phase.combine.role.Main.info.verb.sum_b

Load one guarded PartB term per thread and multiply by its shared weight. Each warp performs combine's 32-entry tree; its first lane writes GroupB. A CTA barrier publishes every GroupA and GroupB partial. Ns<=512 ensures Cmerge<=MergeWidth, so one term per thread covers the denominator. Weight remains unchanged through all numerator and denominator readers.

## t_4b283af372768415 | pipeline.verb | 262 chars | 6 declarations

tcg_d256_flat_cp_narrow:669 design.tile_pipeline.phase.combine.role.Main.info.verb.numerator; tcg_d256_flat_cp_wide:683 design.tile_pipeline.phase.combine.role.Main.info.verb.numerator; tcg_d256_flat_tma_narrow:664 design.tile_pipeline.phase.combine.role.Main.info.verb.numerator; tcg_d256_segmented_cp_narrow:737 design.tile_pipeline.phase.combine.role.Main.info.verb.numerator; tcg_d256_segmented_tma_narrow:752 design.tile_pipeline.phase.combine.role.Main.info.verb.numerator; tcg_d256_segmented_tma_wide:737 design.tile_pipeline.phase.combine.role.Main.info.verb.numerator

For channel tid+128*c, issue eight independent loads for splits 8*j..8*j+7, using zero without a read beyond Ns. Add in ascending order; the batch dies before reuse. Both complete channel walks precede the denominator barrier, preserving load overlap with PartB.

## t_4bc2b0f5767a5b32 | pipeline.verb | 359 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:712 design.tile_pipeline.phase.split.role.Main.info.verb.prefetch

Load NextMax for g+1 when g<cHi, otherwise retain -inf; this load occurs for all threads, including padded rows. Enqueue K/V(g+1) and their atoms into the alternate slot when g<cHi. Every thread then commits a group, including an empty group at g=cHi. Prior QK/PV completion has retired that slot's operand reads; done does not wait for newly enqueued copies.

## t_4c5875ac72c74900 | pipeline.verb | 206 chars | 1 declarations

tcg_d256_segmented_cp_wide:650 design.tile_pipeline.phase.scan.role.Main.info.verb.finish

Valid threads write Mtail from ScanState and discard local state. There is no exit trigger or CTA barrier. Kernel completion, observed by split's late dependency wait, publishes every Mscan entry and Mtail.

## t_4c9ab24ba17367c6 | pipeline.verb | 429 chars | 2 declarations

tc_split_segmented_narrow:583 design.tile_pipeline.phase.segmax.role.Main.info.verb.entry; tc_split_segmented_wide:617 design.tile_pipeline.phase.segmax.role.Main.info.verb.entry

Trigger programmatic launch completion, then wait for grid dependencies, before the row>=R return. The grid-wide trigger condition supplies segmax.launch_ready before summary stores; it does not publish data. The ordinary maximum predecessor has no early trigger, so this wait requires its completion. Padding threads execute the entry protocol; later role actions retain their events without numerical effects for those threads.

## t_4cafea7b13237b34 | pipeline.verb | 584 chars | 1 declarations

tcg_d256_segmented_cp_wide:752 design.tile_pipeline.phase.split.role.Main.info.verb.prime

When C>0, all eight warps enqueue K(l) and V(l) in 16-byte cp.async pieces, 65536 bytes total, zero-filling keys beyond Skv. Threads 0..63 also enqueue both 1024-byte atoms for l through min(l+2,e) into three slots. Commit payloads and atoms together once; wait_group 0, shared async-proxy fence and CTA-sync publish both payloads, all primed atoms and Query. Warps 0..3 copy l's atoms to its TMEM parity slots and wait for stores; a before-thread-sync fence and CTA barrier publish them. Current K/V copies are already complete. At C=0 this action issues no copies, commits or waits.

## t_4d389b763ad1015c | pipeline.verb | 713 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:741 design.tile_pipeline.phase.split.role.Main.info.verb.write

All 128 threads cooperatively read 16-byte chunks of the actual shared bytes for valid staged rows and issue global PartA stores, remapping each row through the head-major map. FP32 bit patterns are preserved, including any empty-split corruption. Each valid row owner separately stores weighted DenState to PartB after its numerator stores; empty splits still write PartB=0 from registers. No O or LSE is written here. done means stores issued with their sources consumed; no later local reuse occurs, and global visibility still requires kernel completion. The successor reads these actual PartA values without an empty-split mask, so the race may affect TotalA and O; it has no PartB or MTail data path to LSE.

## t_4d733bff96c23721 | pipeline.verb | 678 chars | 1 declarations

fused_d128_group:374 design.tile_pipeline.phase.main.role.Even.info.verb.prime

If cLo+w<=cHi, issue that group's K/V and atoms into this role's first slots, commit, then issue K and its atom for cLo+w+2 into its second K slot when present and commit again, including an empty second commit. wait_group 1, the async-shared fence and role barrier prove this role's older copies, including its own cold contribution. Stage the first K atom into KeyScale; wait::st, before fence and role barrier publish it. After an after fence, the leading thread issues four QK contractions into its RawScore and commits its QK barrier. RawScore is A[3] from scores; done means QK enqueue. With no owned group the bound check completes without copying, committing or waiting.

## t_4d8d33cd3a3344af | pipeline.verb | 366 chars | 1 declarations

fused_d128_key:412 design.tile_pipeline.phase.main.role.Main.info.verb.pv

After thread 0's after fence, enqueue four ordered full-width PV contractions, reading Prob8, Value(g), UnitScale and ValueScale(g), then commit one PV phase. The first contraction at cLo overwrites NumState; all others accumulate. done denotes enqueue only. No numerator readback or tail CTA barrier follows; the next iteration can stage while this PV remains live.

## t_4e7b6b559a798fc1 | pipeline.verb | 714 chars | 2 declarations

tcg_d256_flat_cp_narrow:547 design.tile_pipeline.phase.split.role.Main.info.verb.prepare; tcg_d256_flat_cp_wide:561 design.tile_pipeline.phase.split.role.Main.info.verb.prepare

Establish work coordinates. AlignOffset is the 0..1023-byte displacement from raw shared base to its next 1024-byte boundary; OperandScratch covers the union of operand and epilogue views. Initialize QK/PV protocols at parity zero, publishing initialization before use. Prefetch Q/scales before the first warp allocates 512 TMEM columns; CTA-sync publishes the base, then that warp relinquishes its permit. Write Query cooperatively and QueryScale/UnitScale with warps 0..3, waiting for stores; QueryPrefetch then dies. Half z owns all Br rows, keys [64*z,64*z+64) and output channels [128*z,128*z+128). Preparation reads no scan result and issues no K/V prime, even for nonempty work. The q0>=Sq guard is untaken.

## t_4f1aee2b6b68abb6 | pipeline.verb | 366 chars | 1 declarations

tc_fused:298 design.tile_pipeline.phase.main.role.Main.info.verb.setup

Establish this CTA's coordinates and masks. The q0>=Sq guard precedes copying; the exact launch grid cannot take it. Each warp owns 16 rows; each four-lane subgroup shares two rows and partitions paired key/output columns across its lanes. MaxState has four replicas per row, while DenPart has four distinct partials. All four warps stage data and join CTA barriers.

## t_507c73ed1835f95f | pipeline.verb | 476 chars | 1 declarations

tcg_d128_fscan_tma_xpose_wide:684 design.tile_pipeline.phase.combine.role.Main.info.verb.write

Group zero alone reads all shared BinB entries to form TotalB and each channel's GroupA entries to form TotalA, following combine's orders, then writes its BF16 output channel. Its first thread additionally writes LSE when present. Other groups have no output payload for this action; all their required partials and barrier participation are complete. Caller destinations are not read. Kernel completion supplies visibility; there is no next tile or launch permission signal.

## t_51fa16e280d575c5 | pipeline.verb | 668 chars | 1 declarations

tcg_d128_fscan_tma_direct_wide:637 design.tile_pipeline.phase.combine.role.Main.info.verb.sum_a

Consecutive four-warp groups implement combine's MergeGroups intervals. In group p each thread owns one channel d and candidate split Slo(p)+d. Clear GroupAcc; every thread reads MTail and writes Weight[p,d], including zero for d>=MergeSize(p). Load the guarded first eight numerator partials per channel into NumeratorBatch, then CTA-sync to publish every weight before weighted additions. Consume the first batch, later complete eight-entry batches and scalar remainder in combine's order, reusing registers after their contents are consumed. Every thread stores its final GroupAcc into GroupA. done covers own stores, with cross-group publication deferred to sum_b.

## t_52ae9dd22f8432e6 | pipeline.verb | 848 chars | 6 declarations

tcg_d128_flat_cp_xpose_narrow:617 design.tile_pipeline.phase.split.role.Main.info.verb.stage_partials; tcg_d128_flat_cp_xpose_wide:600 design.tile_pipeline.phase.split.role.Main.info.verb.stage_partials; tcg_d128_flat_tma_xpose_wide:623 design.tile_pipeline.phase.split.role.Main.info.verb.stage_partials; tcg_d128_segmented_cp_xpose_narrow:676 design.tile_pipeline.phase.split.role.Main.info.verb.stage_partials; tcg_d128_segmented_tma_xpose_narrow:692 design.tile_pipeline.phase.split.role.Main.info.verb.stage_partials; tcg_d128_segmented_tma_xpose_wide:676 design.tile_pipeline.phase.split.role.Main.info.verb.stage_partials

Nonempty valid rows load MTail and form weighted FP32 numerators; empty splits evaluate zeros. Each valid row writes D channels into the PartA view of OperandArena. Its 81920-byte operand capacity holds the 67584-byte staging span including row spacing; padded rows are neither staged nor later read. For Ng>0, the loop's CTA joins completed all Query writes, QK/PV waits retired operand reads, and the last group enqueued no future transfer, so the arena is free before staging. For Ng=0 the loop and its joins are absent: zero staging writes can race with another warp's unfinished Query writes in the overlapping shared region. The source has no pre-staging barrier. Its barrier after staging joins preparation/staging writes and final TMEM readouts, making the resulting bytes readable without restoring zero. done includes this actual barrier.

## t_533841fd74436c1d | pipeline.verb | 234 chars | 5 declarations

tcg_d256_flat_tma_narrow:400 design.tile_pipeline.phase.group_max.role.Main.info.verb.prime; tcg_d256_flat_tma_wide:414 design.tile_pipeline.phase.group_max.role.Main.info.verb.prime; tcg_d256_segmented_cp_narrow:438 design.tile_pipeline.phase.group_max.role.Main.info.verb.prime; tcg_d256_segmented_tma_narrow:453 design.tile_pipeline.phase.group_max.role.Main.info.verb.prime; tcg_d256_segmented_tma_wide:438 design.tile_pipeline.phase.group_max.role.Main.info.verb.prime

If C>0, thread zero issues K(l) as two 16384-byte TMA requests on a 32768-byte transaction stage, while threads 0..63 issue its 1024-byte atom with cp.async; commit once. Otherwise issue no requests. done denotes enqueue, not arrival.

## t_534379a13643ee21 | pipeline.verb | 319 chars | 5 declarations

tcg_d256_flat_tma_narrow:394 design.tile_pipeline.phase.group_max.role.Main.info.verb.setup; tcg_d256_flat_tma_wide:408 design.tile_pipeline.phase.group_max.role.Main.info.verb.setup; tcg_d256_segmented_cp_narrow:432 design.tile_pipeline.phase.group_max.role.Main.info.verb.setup; tcg_d256_segmented_tma_narrow:447 design.tile_pipeline.phase.group_max.role.Main.info.verb.setup; tcg_d256_segmented_tma_wide:432 design.tile_pipeline.phase.group_max.role.Main.info.verb.setup

Establish coordinates and the domain row map. Warpgroups z=0,1 each cover Br rows and keys [64*z,64*z+64). Half zero writes its assigned -inf tails. Initialize two transaction stages and QK/PV completion protocols at parity zero; publish initialization. The exact grid never takes q0>=Sq; empty spans still reach setup.

## t_54d97868476ec692 | pipeline.verb | 423 chars | 1 declarations

fused_d128_key:360 design.tile_pipeline.phase.main.role.Main.info.verb.stage

Ahead(g) issues g+2 into shared slot (g+2-cLo)%4 when g+2<=cHi. It always commits a copy group, including empty groups at the tail, then waits until at most one group remains outstanding. The async-shared fence and CTA barrier publish g and, when it exists, g+1. The just-issued g+2 may remain in flight. On wrap, the destination previously held g-2: its QK and atom readers had finished by g-2, and PV(g-2) retired in g-1.

## t_55415be3c5f79c1f | pipeline.verb | 290 chars | 1 declarations

fused_d128_key:419 design.tile_pipeline.phase.main.role.Main.info.verb.drain

Wait for the last PV phase, advance its parity and apply the after fence and CTA barrier. The final empty copy commits and wait_group 1 have already completed all payload transfers; no extra wait_group 0 is inserted. QK has been retired in every iteration. NumState is now ready for output.

## t_56b515b6cdff545e | pipeline.verb | 199 chars | 1 declarations

tcg_d128_fscan_tma_xpose_wide:544 design.tile_pipeline.phase.split.role.Main.info.verb.prime

For Ng>0 enqueue K(cLo), V(cLo) and both scale atoms in slot zero. One transaction barrier accounts for 2*(16384+512) bytes per slot. For Ng=0 issue no transfer. done means enqueue, not data arrival.

## t_57a522db3133d444 | pipeline.verb | 791 chars | 3 declarations

tcg_d128_fscan_cp_xpose_narrow:596 design.tile_pipeline.phase.split.role.Main.info.verb.write; tcg_d128_fscan_cp_xpose_wide:613 design.tile_pipeline.phase.split.role.Main.info.verb.write; tcg_d128_fscan_tma_xpose_wide:601 design.tile_pipeline.phase.split.role.Main.info.verb.write

All 128 threads cooperatively read 16-byte chunks of the actual shared bytes for valid staged rows and issue global PartA stores, remapping each row through the head-major map. FP32 bit patterns are preserved, including any empty-split corruption. Each valid row owner separately stores unweighted DenState to PartB and its scanned end state to PartM after its numerator stores; empty splits still write PartB=0 and their prefix state to PartM from registers. No O or LSE is written here. done means stores issued with their sources consumed; no later local reuse occurs, and global visibility still requires kernel completion. The successor weights these actual PartA values without an empty-split mask, so the race may affect TotalA and O; it has no PartB, PartM or MTail data path to LSE.

## t_587ef5daa2d313af | pipeline.verb | 191 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:603 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.tail

Valid threads issue MTail stores after their last MState entry. Store completion and visibility are supplied by kernel completion. Every invocation rewrites all R*T snapshots and all R tails.

## t_58b7c5a855f4c4cb | pipeline.verb | 290 chars | 2 declarations

tc_split_flat_narrow:652 design.tile_pipeline.phase.combine.role.Main.info.verb.sum; tc_split_segmented_narrow:759 design.tile_pipeline.phase.combine.role.Main.info.verb.sum

For NumBatch(c,k), append eight values to the owned SumA channel in ascending order; reads finish before the next batch overwrites its slot. For Denominator, each thread initializes its SumB replica to zero and sums shared values in ascending eight-wide batches with the declared zero tail.

## t_5a002e32dc32f2de | pipeline.verb | 121 chars | 3 declarations

tcg_d256_segmented_cp_narrow:495 design.tile_pipeline.phase.segment_max.role.Main.info.verb.trigger; tcg_d256_segmented_tma_narrow:510 design.tile_pipeline.phase.segment_max.role.Main.info.verb.trigger; tcg_d256_segmented_tma_wide:495 design.tile_pipeline.phase.segment_max.role.Main.info.verb.trigger

Every thread grants scan's programmatic launch permission at entry, before waiting for group_max. This publishes no data.

## t_5a659530002c9feb | pipeline.verb | 605 chars | 1 declarations

root:565 design.tile_pipeline.phase.main.role.Main.info.verb.land

Key(g) waits for this slot's K payload and atom. The shared async-proxy fence and CTA barrier make the staged bytes readable to all warps. Copy KeyAtom into KeyScale; wait for the TMEM stores, then fence and CTA-sync before thread 0 can issue QK. This also publishes Query on the first group. Value(g) independently waits for current V and its atom, fences the shared operand, writes ValueScale and waits for those TMEM stores. It follows retirement of the previous PV's last reads of the single ValueScale slot. K/V ring parity changes only on a two-slot wrap, independently for the two TMA barrier sets.

## t_5b905a6275b8a956 | pipeline.verb | 243 chars | 1 declarations

tcg_d128_fscan_cp_xpose_narrow:660 design.tile_pipeline.phase.combine.role.Main.info.verb.prime

Load the first eight numerator partials into NumeratorBatch, using zero for s>=Ns, before the CTA barrier that joins all DenParts and Weight stores. done guarantees these shared values are readable by all channels and the first batch is ready.

## t_5ccd8df223430d65 | pipeline.verb | 402 chars | 1 declarations

fused_d256_tma:421 design.tile_pipeline.phase.main.role.Main.info.verb.pv

Thread 0 applies its after-thread-sync fence and enqueues four increasing 32-key steps, each containing both 128-channel output products. Read Prob8, Value(g), UnitScale and ValueScale(g); the first key step of the first group overwrites both output halves, later steps accumulate. Commit one PV completion phase. done denotes enqueue only; there is no tail CTA barrier or per-group numerator readback.

## t_5d51a22e6ea31e23 | pipeline.verb | 493 chars | 1 declarations

fused_d128_key:385 design.tile_pipeline.phase.main.role.Main.info.verb.read

Wait for the current QK phase, advance its parity once and apply the after fence. Each thread reads its 64 scores in two self-retiring 32-value TMEM loads, masks them and retains Score. It stores HalfMax in its Exchange slice; a CTA barrier publishes both halves and proves all current score readouts have retired. Both warpgroups read the two maxima. The helper has no trailing barrier; the later probability-publication barrier joins these scratch readers before a future Exchange overwrite.

## t_5d7007bf6e0d67a2 | pipeline.verb | 932 chars | 1 declarations

tc_split_flat_wide:662 design.tile_pipeline.phase.combine.role.Main.info.verb.reduce

Each group independently consumes its numerator interval. For each full batch, each thread loads eight FP32 PartialA values for its first channel, then appends them in order to GroupAccum; D=256 repeats for the second channel. All reads of NumeratorBatch finish before the next channel or batch reuses it. Remaining splits use scalar loads/adds without zero padding. Empty intervals leave zero accumulators. Store every GroupA channel into its group's shared row, including zeros. Each thread then loads PartialB at aLo(f)+d0 or uses zero beyond aHi(f). The host bound n(f)<=128 lets this single load round cover the interval. Each warp performs the five denominator tree stages; its first lane stores GroupB. All sixteen entries are written, even for empty subranges. GroupSums means GroupA and GroupB together. done covers these ordinary shared stores; there is no publication barrier or combine-side weighting inside this action.

## t_5d718e37fe8401cd | pipeline.verb | 374 chars | 1 declarations

tcg_d256_segmented_cp_wide:495 design.tile_pipeline.phase.group_max.role.Main.info.verb.setup

Establish work coordinates and the PSK row map. Warpgroups z=0,1 each cover all Br rows and keys [64*z,64*z+64). They cooperate in copies and CTA barriers even for padding or C=0. The exact grid never takes the q0>=Sq early exit. Half zero writes the -inf tails assigned by GMGroups. Initialize QK/PV completion protocols with parity zero; publish initialization before use.

## t_5e1a98ee9fa04b60 | pipeline.verb | 222 chars | 1 declarations

tcg_d128_fscan_cp_xpose_narrow:665 design.tile_pipeline.phase.combine.role.Main.info.verb.sum_b

Each thread reads DenParts and Weight in eight-entry batches and evaluates combine's ordered weighted denominator sum. Missing final entries add zero without shared reads. There is no cross-thread sum or further reduction.

## t_5e821d0956aa422e | pipeline.verb | 155 chars | 1 declarations

tcg_d256_segmented_cp_wide:619 design.tile_pipeline.phase.scan.role.Main.info.verb.await

At kernel entry every thread performs the PDL dependency wait for segment_max, proving SegMax and transitively GroupMax visible before any scan input read.

## t_5e91b53c607e1f75 | pipeline.verb | 479 chars | 5 declarations

tcg_d256_flat_tma_narrow:568 design.tile_pipeline.phase.split.role.Main.info.verb.prime; tcg_d256_flat_tma_wide:582 design.tile_pipeline.phase.split.role.Main.info.verb.prime; tcg_d256_segmented_cp_narrow:641 design.tile_pipeline.phase.split.role.Main.info.verb.prime; tcg_d256_segmented_tma_narrow:656 design.tile_pipeline.phase.split.role.Main.info.verb.prime; tcg_d256_segmented_tma_wide:641 design.tile_pipeline.phase.split.role.Main.info.verb.prime

For C>0, enqueue K(l)/V(l), two 16384-byte TMA requests each on one 65536-byte transaction stage, and separately enqueue both 1024-byte atoms for l..min(l+2,e) into three slots. Commit once, wait_group 0, fence the shared async proxy and CTA-sync. Warps 0..3 copy l's atoms to its TMEM parity slots, wait for stores, fence before thread-sync and CTA-sync. Current payload TMA can remain in flight. At C=0 this action issues no copies, commits or waits, while retaining its event.

## t_5f4281d22fdfd51d | pipeline.verb | 192 chars | 13 declarations

tcg_d128_flat_cp_direct_narrow:665 design.tile_pipeline.phase.combine.role.Main.info.verb.write; tcg_d128_flat_cp_xpose_narrow:693 design.tile_pipeline.phase.combine.role.Main.info.verb.write; tcg_d128_flat_cp_xpose_wide:676 design.tile_pipeline.phase.combine.role.Main.info.verb.write; tcg_d128_flat_tma_direct_narrow:658 design.tile_pipeline.phase.combine.role.Main.info.verb.write; tcg_d128_flat_tma_xpose_narrow:681 design.tile_pipeline.phase.combine.role.Main.info.verb.write; tcg_d128_fscan_cp_direct_narrow:633 design.tile_pipeline.phase.combine.role.Main.info.verb.write; tcg_d128_fscan_tma_direct_narrow:626 design.tile_pipeline.phase.combine.role.Main.info.verb.write; tcg_d128_fscan_tma_xpose_narrow:653 design.tile_pipeline.phase.combine.role.Main.info.verb.write; tcg_d128_segmented_cp_direct_narrow:756 design.tile_pipeline.phase.combine.role.Main.info.verb.write; tcg_d128_segmented_cp_xpose_narrow:752 design.tile_pipeline.phase.combine.role.Main.info.verb.write; tcg_d128_segmented_tma_direct_narrow:748 design.tile_pipeline.phase.combine.role.Main.info.verb.write; tcg_d128_segmented_tma_xpose_narrow:768 design.tile_pipeline.phase.combine.role.Main.info.verb.write; tcg_d128_segmented_tma_xpose_wide:752 design.tile_pipeline.phase.combine.role.Main.info.verb.write

Write each owned BF16 output channel; thread zero reads MTail and writes LSE when present. Initial destinations are never read. Kernel completion provides visibility; no next tile is assigned.

## t_61d480fac2f07bb7 | pipeline.verb | 210 chars | 12 declarations

tcg_d128_flat_cp_direct_narrow:606 design.tile_pipeline.phase.split.role.Main.info.verb.accumulate; tcg_d128_flat_cp_direct_wide:620 design.tile_pipeline.phase.split.role.Main.info.verb.accumulate; tcg_d128_flat_tma_direct_narrow:599 design.tile_pipeline.phase.split.role.Main.info.verb.accumulate; tcg_d128_flat_tma_direct_wide:613 design.tile_pipeline.phase.split.role.Main.info.verb.accumulate; tcg_d128_fscan_cp_direct_narrow:569 design.tile_pipeline.phase.split.role.Main.info.verb.accumulate; tcg_d128_fscan_cp_direct_wide:586 design.tile_pipeline.phase.split.role.Main.info.verb.accumulate; tcg_d128_fscan_tma_direct_narrow:562 design.tile_pipeline.phase.split.role.Main.info.verb.accumulate; tcg_d128_fscan_tma_direct_wide:580 design.tile_pipeline.phase.split.role.Main.info.verb.accumulate; tcg_d128_segmented_cp_direct_narrow:697 design.tile_pipeline.phase.split.role.Main.info.verb.accumulate; tcg_d128_segmented_cp_direct_wide:714 design.tile_pipeline.phase.split.role.Main.info.verb.accumulate; tcg_d128_segmented_tma_direct_narrow:689 design.tile_pipeline.phase.split.role.Main.info.verb.accumulate; tcg_d128_segmented_tma_direct_wide:708 design.tile_pipeline.phase.split.role.Main.info.verb.accumulate

Live warps read GroupPV in 32-channel chunks with TMEM-load waits and add into NumState; dead warps skip readback. No tail CTA barrier is added: next stage's joins protect reuse, or finish joins the last group.

## t_62522bc4153c7ee5 | pipeline.verb | 427 chars | 3 declarations

tcg_d128_segmented_cp_xpose_narrow:523 design.tile_pipeline.phase.segment_max.role.Main.info.verb.produce; tcg_d128_segmented_tma_xpose_narrow:539 design.tile_pipeline.phase.segment_max.role.Main.info.verb.produce; tcg_d128_segmented_tma_xpose_wide:523 design.tile_pipeline.phase.segment_max.role.Main.info.verb.produce

Each valid thread loads its segment into GroupBatch, substituting -inf for missing final-segment groups, then performs segment_max's ordered reduction and issues its SegMax store. Lanes address consecutive rows; no cross-thread reduction or CTA barrier occurs. Segment zero is also written. Out-of-range rows perform no loads, reduction or stores. done consumes the register batch; global visibility requires kernel completion.

## t_64730ab2af955121 | pipeline.verb | 209 chars | 8 declarations

tcg_d128_flat_cp_xpose_narrow:615 design.tile_pipeline.phase.split.role.Main.info.verb.accumulate; tcg_d128_flat_cp_xpose_wide:598 design.tile_pipeline.phase.split.role.Main.info.verb.accumulate; tcg_d128_flat_tma_xpose_narrow:602 design.tile_pipeline.phase.split.role.Main.info.verb.accumulate; tcg_d128_flat_tma_xpose_wide:621 design.tile_pipeline.phase.split.role.Main.info.verb.accumulate; tcg_d128_fscan_tma_xpose_narrow:568 design.tile_pipeline.phase.split.role.Main.info.verb.accumulate; tcg_d128_segmented_cp_xpose_narrow:674 design.tile_pipeline.phase.split.role.Main.info.verb.accumulate; tcg_d128_segmented_tma_xpose_narrow:690 design.tile_pipeline.phase.split.role.Main.info.verb.accumulate; tcg_d128_segmented_tma_xpose_wide:674 design.tile_pipeline.phase.split.role.Main.info.verb.accumulate

All warps read GroupPV in 32-channel chunks with TMEM-load waits and add into NumState. The next stage's joins protect subsequent TMEM reuse; stage_partials joins the final group. No tail CTA barrier is added.

## t_65970022a6c1b825 | pipeline.verb | 285 chars | 7 declarations

tcg_d128_flat_cp_xpose_narrow:589 design.tile_pipeline.phase.split.role.Main.info.verb.dependency.meaning; tcg_d128_flat_cp_xpose_wide:572 design.tile_pipeline.phase.split.role.Main.info.verb.dependency.meaning; tcg_d128_flat_tma_xpose_narrow:578 design.tile_pipeline.phase.split.role.Main.info.verb.dependency.meaning; tcg_d128_flat_tma_xpose_wide:597 design.tile_pipeline.phase.split.role.Main.info.verb.dependency.meaning; tcg_d128_segmented_cp_xpose_narrow:650 design.tile_pipeline.phase.split.role.Main.info.verb.dependency.meaning; tcg_d128_segmented_tma_xpose_narrow:666 design.tile_pipeline.phase.split.role.Main.info.verb.dependency.meaning; tcg_d128_segmented_tma_xpose_wide:650 design.tile_pipeline.phase.split.role.Main.info.verb.dependency.meaning

Each thread executes the PDL scan wait after its own preparation and before its first snapshot read. done guarantees the predecessor scan's MState/MTail visibility, without initializing or extending those regions. The wait does not join this CTA's warps or complete their Query writes.

## t_65f4613c5f9dcc33 | pipeline.verb | 382 chars | 2 declarations

tcg_d128_flat_cp_xpose_narrow:454 design.tile_pipeline.phase.group_max.role.Main.info.verb.finish; tcg_d128_flat_cp_xpose_wide:437 design.tile_pipeline.phase.group_max.role.Main.info.verb.finish

All issued K payload/atom copies and QK products have been waited; any final copy group contains no payload. Fence and CTA-sync join final TMEM users, then the first warp deallocates 256 columns. Each CTA signals PDL launch completion after its stores and teardown; launch_ready is the grid-wide protocol event. Global visibility comes from kernel completion and the successor wait.

## t_65fc27c4142c4b03 | pipeline.verb | 675 chars | 1 declarations

tcg_d256_segmented_cp_wide:726 design.tile_pipeline.phase.split.role.Main.info.verb.prepare

Establish coordinates and completion protocols; AlignOffset is the 0..1023-byte displacement from raw shared base to its next 1024-byte boundary. The exact grid never takes q0>=Sq. Initialize QK/PV completion protocols at parity zero. Prefetch Q and scale words before the first warp allocates 512 TMEM columns, CTA-sync publishes the base, and that warp relinquishes the permit. Write Query cooperatively; warps 0..3 write QueryScale and UnitScale and wait for TMEM stores. Discard QueryPrefetch. The two warpgroups cover identical Br rows; half z owns keys [64*z,64*z+64) and output channels [128*z,128*z+128). This preparation reads no scan output and issues no K/V prime.

## t_65fcc84bd47b295d | pipeline.verb | 527 chars | 1 declarations

tcg_d256_segmented_cp_wide:531 design.tile_pipeline.phase.group_max.role.Main.info.verb.maximum

Each warp reads its 64-key slice in two 32-column chunks, except wholly padded warps when PSK is active. A warp with no mask uses native load-reduce-max; a masked warp loads, waits, masks and reduces in registers. With PSK, a wholly padded warp skips readout, but remains in the exchange. Both halves write HalfMax, CTA-sync, then read both halves. Half zero stores GroupMax for valid rows, followed by another CTA barrier. This retires ScoreTile and Exchange readers before their next overwrite; no scores survive this action.

## t_683e8b76557801d3 | pipeline.verb | 616 chars | 1 declarations

fused_d128_key:349 design.tile_pipeline.phase.main.role.Main.info.verb.prime

Commit an empty copy group for the already-hoisted slot zero. Issue cLo+1 into slot one if it exists, then commit another copy group even if empty. wait_group 1 proves cLo's copies complete; an async-shared fence and CTA barrier publish Query and the landed inputs. Copy KeyAtom(cLo) into KeyScale(cLo), wait for TMEM stores, then apply a before-thread-sync fence and CTA barrier. After its after-thread-sync fence, thread 0 enqueues four QK contractions into RawScore(cLo) and commits one QK phase. RawScore denotes the unmasked A[3] of scores. done is QK enqueue only; the loop's read action establishes readiness.

## t_69387e2755510ea5 | pipeline.verb | 183 chars | 1 declarations

tcg_d128_fscan_tma_xpose_wide:434 design.tile_pipeline.phase.group_max.role.Main.info.verb.prime

For Ng>0 enqueue K(cLo) and its 512-byte scale atom into slot zero, sharing one transaction barrier for 16384+512 bytes. For Ng=0 issue no transfer. done denotes enqueue, not arrival.

## t_6c19e03dea3b4771 | pipeline.verb | 491 chars | 4 declarations

tcg_d128_flat_tma_direct_narrow:579 design.tile_pipeline.phase.split.role.Main.info.verb.replay; tcg_d128_flat_tma_direct_wide:593 design.tile_pipeline.phase.split.role.Main.info.verb.replay; tcg_d128_segmented_tma_direct_narrow:669 design.tile_pipeline.phase.split.role.Main.info.verb.replay; tcg_d128_segmented_tma_direct_wide:688 design.tile_pipeline.phase.split.role.Main.info.verb.replay

Initialize DenState to zero and MaxState to -inf. Every thread loads MState at (cLo-1)*R+rowG when cLo>0, including empty splits; load CurrentMax at cLo only when Ng>0. rowG is domain's r. The source has no any_compute guard or T-bound clamp on the entrance load: a logically out-of-range empty-split read is not given an initialization or safety guarantee here. Its loaded value is discarded from empty-split results; dead-warps' loads likewise do not affect output or subsequent addresses.

## t_6cd4d676c43f79f2 | pipeline.verb | 458 chars | 1 declarations

fused_d128_key:440 design.tile_pipeline.phase.main.role.Main.info.verb.finish

A before fence and CTA barrier join all final TMEM readers; the first warp deallocates the 512-column region. The source calls fmha_pdl_trigger at exit, but this ordinary fused launch has no dependent phase or PDL input wait. No new work is assigned to the CTA. Kernel completion supplies caller-stream visibility while other CTAs may proceed independently. A later invocation initializes fresh numerical state; host caches carry descriptors and graphs only.

## t_6d272863da77b502 | pipeline.verb | 215 chars | 2 declarations

tcg_d128_fscan_cp_direct_narrow:631 design.tile_pipeline.phase.combine.role.Main.info.verb.sum_b; tcg_d128_fscan_tma_direct_narrow:624 design.tile_pipeline.phase.combine.role.Main.info.verb.sum_b

Each thread independently performs combine's ordered weighted sum over DenParts and Weight. Both shared arrays remain unchanged through all numerator and denominator readers; TotalB is replicated per output channel.

## t_6d6ee50a6c5f0e0f | pipeline.verb | 414 chars | 1 declarations

tcg_d256_flat_tma_wide:685 design.tile_pipeline.phase.combine.role.Main.info.verb.publish

GroupPartials comprises GroupNum and GroupDen. Store Numerator to this group's GroupNum slice. Local thread d0 loads PartB[SLo(z)+d0,row] when below SHi(z), otherwise uses zero. Descending shuffle offsets implement output's 32-position reduction; each warp's lane zero writes GroupDen. One CTA barrier publishes all shared sums, including empty-group zeros. No weight array is consumed in this non-fused-scan form.

## t_6dace21cbf74fe53 | pipeline.verb | 304 chars | 2 declarations

fused_d256_cp:417 design.tile_pipeline.phase.main.role.Main.info.verb.fold; fused_d256_tma:435 design.tile_pipeline.phase.main.role.Main.info.verb.fold

Overwrite Exchange with each half's final DenPart, CTA-sync, read both halves in key order and overwrite each register copy with DenFinal. The final advance barrier already retired the old HalfMax readers. Both halves use this denominator for output; no denominator exchange occurs inside the group loop.

## t_6ee2099a03c04432 | pipeline.verb | 292 chars | 1 declarations

tcg_d256_segmented_cp_wide:503 design.tile_pipeline.phase.group_max.role.Main.info.verb.prime

When C>0, all eight warps enqueue the first 32768-byte K tile in 16-byte cp.async pieces, zero-filling keys beyond Skv; threads 0..63 also enqueue its 1024-byte KeyAtom. Commit these requests together. When C=0 there are no operand requests. done denotes enqueue, leaving transfers in flight.

## t_6ef475c3c3c7d507 | pipeline.verb | 408 chars | 1 declarations

tcg_d256_segmented_cp_wide:642 design.tile_pipeline.phase.scan.role.Main.info.verb.replay

Write the unchanged prefix and replay the selected suffix using scan and the helper defined in first. The fill may be the whole chunk, or empty; its complement may be empty. This action still completes in either case and writes every state of the chunk. Only the suffix reads GroupMax. A row's state is retained until the next chunk; no other thread reads its registers and no inter-thread barrier is needed.

## t_6f713b50ed73737a | pipeline.verb | 559 chars | 1 declarations

tcg_d256_segmented_cp_wide:541 design.tile_pipeline.phase.group_max.role.Main.info.verb.finish

Even at C=0, both halves publish their unchanged zero denominators over Exchange, CTA-sync and read the unused sum. The final maximum barrier retired its previous readers. No V, probability, numerator or caller-output access occurs. All nonempty K/atom requests have been consumed; empty commits carry no payload. A before-thread-sync fence and CTA barrier join TMEM readers before the first warp deallocates 256 columns. Trigger programmatic launch completion at exit. No next tile is assigned; kernel_done supplies global visibility to segment_max and scan.

## t_6f74b91b5be77499 | pipeline.verb | 162 chars | 2 declarations

tc_split_segmented_narrow:624 design.tile_pipeline.phase.scan.role.Main.info.verb.load; tc_split_segmented_wide:658 design.tile_pipeline.phase.scan.role.Main.info.verb.load

Load the Sc independent summaries for chunk c into Summary(c), using -inf beyond Nseg. done makes the full batch available before any candidate test or walk call.

## t_70934de73770789d | pipeline.verb | 283 chars | 1 declarations

tcg_d256_segmented_cp_wide:587 design.tile_pipeline.phase.segment_max.role.Main.info.verb.produce

Valid threads apply segment_max to GroupBatch and store one SegMax[f,row]; invalid rows perform no data access. GroupBatch is dead after the reduction. No CTA barrier, exit trigger or cross-row communication occurs; kernel completion publishes all segments to scan's dependency wait.

## t_70acbc6d808b5d7f | pipeline.verb | 662 chars | 1 declarations

tcg_d128_fscan_tma_xpose_narrow:535 design.tile_pipeline.phase.split.role.Main.info.verb.prepare.meaning

Prefetch Q/scales, first-warp allocate 256 TMEM columns, CTA-publish the base and relinquish the permit. Each thread writes Query/QueryScale/UnitScale, waits for its TMEM stores, discards QueryPrefetch and clears its NumState before entering its group-max wait. No CTA join follows these Query writes here. done covers preparation across all warps; later actions can start before that collective completion. The next stage CTA join proves completion for Ng>0; for Ng=0 only the barrier after stage_partials writes proves it. This models cross-warp progression of ordinary shared stores, without changing their instructions. Empty splits also perform preparation.

## t_71724834d6b19a8b | pipeline.verb | 212 chars | 4 declarations

tc_split_flat_narrow:473 design.tile_pipeline.phase.group_max.role.Main.info.verb.prepare; tc_split_flat_wide:490 design.tile_pipeline.phase.group_max.role.Main.info.verb.prepare; tc_split_segmented_narrow:548 design.tile_pipeline.phase.group_max.role.Main.info.verb.prepare; tc_split_segmented_wide:582 design.tile_pipeline.phase.group_max.role.Main.info.verb.prepare

Stage Qraw and decode QueryScale, explicitly zeroing invalid rows, then publish at a CTA barrier. Q staging occurs even on empty intervals and may overlap the primed K copy. Q and QueryScale survive the interval.

## t_723638ccf7a06612 | pipeline.verb | 619 chars | 2 declarations

fused_d128_cp:385 design.tile_pipeline.phase.main.role.Main.info.verb.advance; root:603 design.tile_pipeline.phase.main.role.Main.info.verb.advance

Apply normalize and the numerator rescale using the retained Score and RowState. After the first group, any warp with an alpha unequal to one reads/multiplies/stores its NumState rows in 32-channel chunks and waits for the stores; all lanes of that warp participate. The first group never reads NumState. Compute probabilities and denominator partials, pack Prob8 into TMEM, then wait for its stores. A before-thread-sync fence and CTA barrier publish all probability and scale writes before thread 0's next PV. Score and group temporaries are now dead; MaxState, DenState, Prob8 and NumState retain their defined uses.

## t_72b1dbc1d1d3047c | pipeline.verb | 829 chars | 2 declarations

tcg_d256_flat_cp_narrow:581 design.tile_pipeline.phase.split.role.Main.info.verb.ahead; tcg_d256_flat_cp_wide:595 design.tile_pipeline.phase.split.role.Main.info.verb.ahead

Prefetch Mscan[g+1] when present, otherwise ReplayNext=-inf. Enqueue K/V(g+1) into the other two-slot stage and atoms(g+3) into slot (g+3-l)%3 when each group exists. Commit their cp.async requests together even if empty; wait_group 1 proves current K/V and all older atom requests. Copy atoms(g+1) to the alternate TMEM scale parity with warps 0..3 and wait for stores. For g+1<=e, prime or an earlier iteration completed and CTA-published these bytes. Prior QK/PV retirement frees old tile/scale slots; prior head synchronization orders atom readers before refill. At g=e the scale copy still executes on stale or, for C=1 or C=2, uninitialized atom bytes, writing an in-bounds unused TMEM slot. Fence the shared proxy and before thread-sync, then CTA-sync to publish current K/V and ahead scales. No extra tail initialization.

## t_72de25734a920300 | pipeline.verb | 657 chars | 6 declarations

tcg_d128_flat_cp_xpose_narrow:581 design.tile_pipeline.phase.split.role.Main.info.verb.prepare.meaning; tcg_d128_flat_cp_xpose_wide:564 design.tile_pipeline.phase.split.role.Main.info.verb.prepare.meaning; tcg_d128_flat_tma_xpose_wide:589 design.tile_pipeline.phase.split.role.Main.info.verb.prepare.meaning; tcg_d128_segmented_cp_xpose_narrow:642 design.tile_pipeline.phase.split.role.Main.info.verb.prepare.meaning; tcg_d128_segmented_tma_xpose_narrow:658 design.tile_pipeline.phase.split.role.Main.info.verb.prepare.meaning; tcg_d128_segmented_tma_xpose_wide:642 design.tile_pipeline.phase.split.role.Main.info.verb.prepare.meaning

Prefetch Q/scales, first-warp allocate 256 TMEM columns, CTA-publish the base and relinquish the permit. Each thread writes Query/QueryScale/UnitScale, waits for its TMEM stores, discards QueryPrefetch and clears its NumState before entering its scan wait. No CTA join follows these Query writes here. done covers preparation across all warps; later actions can start before that collective completion. The next stage CTA join proves completion for Ng>0; for Ng=0 only the barrier after stage_partials writes proves it. This models cross-warp progression of ordinary shared stores, without changing their instructions. Empty splits also perform preparation.

## t_733f5b3f2b73de23 | pipeline.verb | 624 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:494 design.tile_pipeline.phase.group_max.role.Main.info.verb.setup

Establish the scheduled coordinates, row masks and completion parities, initially zero. RingUsed=3 for Long, otherwise 2; ScaleCol(g)=136+8*(g%2) for Long, otherwise 136. The allocated three shared slots are unchanged. The q0>=Sq guard is unreachable under this grid. Valid rows of split zero/last issue the outside -inf stores. Thread zero initializes the QK/PV completion barriers and issues MMA. All warps cooperatively copy payload; the first warp copies the 512-byte K scale atom. Each thread commits and waits its own copy groups. All four warps read scores, including padded rows; only valid rows write global maxima.

## t_73f04c976a141f66 | pipeline.verb | 213 chars | 1 declarations

tcg_d256_segmented_cp_wide:582 design.tile_pipeline.phase.segment_max.role.Main.info.verb.load

Valid threads load the segment's SegLen group maxima into GroupBatch before reducing any entry. Out-of-range groups use -inf without a global read. Group-major layout makes adjacent valid lanes read adjacent rows.

## t_743ca51c0b7f0a14 | pipeline.verb | 351 chars | 2 declarations

tcg_d128_fscan_cp_direct_narrow:553 design.tile_pipeline.phase.split.role.Main.info.verb.prefetch; tcg_d128_fscan_cp_direct_wide:570 design.tile_pipeline.phase.split.role.Main.info.verb.prefetch

Load NextGroupMax at g+1 when g<cHi, otherwise use -inf, including in dead warps. Enqueue next K/V and atoms in the alternate slot only when g<cHi. Every thread commits a copy group either way, including the empty final one. Prior QK/PV completion retired that slot's operand reads and prior CTA joins retired its atom readers. done means commit only.

## t_747a39e4e92b94cc | pipeline.verb | 248 chars | 1 declarations

fused_d256_tma:355 design.tile_pipeline.phase.main.role.Main.info.verb.start

Thread 0 applies the after-thread-sync fence, enqueues the scores step's eight contractions for cLo into ScoreTile, and commits QK completion. The first contraction overwrites. done means enqueue completion; read_score owns the readiness guarantee.

## t_750f6d19583b90e7 | pipeline.verb | 528 chars | 2 declarations

tcg_d128_flat_cp_direct_narrow:413 design.tile_pipeline.phase.group_max.role.Main.info.verb.setup; tcg_d128_flat_cp_direct_wide:427 design.tile_pipeline.phase.group_max.role.Main.info.verb.setup

Establish coordinates, masks and initially zero completion parities. RingUsed=3 for Long, otherwise 2; ScaleCol(g)=136+8*(g%2) for Long, otherwise 136. Three shared slots remain allocated in both cases. Valid rows of the first/last split issue the outside -inf fills. Thread zero initializes matrix-completion barriers and issues MMA; all four warps stage Q, copies and scales and synchronize. Each thread has an initially empty copy-group stream. Only warps with a valid row perform scalar score readout. q0>=Sq is unreachable.

## t_765f58a3acb39dd0 | pipeline.verb | 689 chars | 1 declarations

tcg_d128_flat_tma_direct_wide:668 design.tile_pipeline.phase.combine.role.Main.info.verb.partial

Four consecutive warps form each merge group z, with one channel d per thread. Clear GroupAccum and accumulate PartA over sLo(z)..sHi(z)-1 in full batches of eight through NumeratorBatch, consuming each batch before register reuse; process the remaining entries individually. Store the result to GroupA. Each thread also loads PartB at sLo(z)+d, or zero beyond sHi(z); Cmerge<=128 makes this cover the whole interval. Each warp performs combine's 32-entry tree and its first lane stores DenChunk. A CTA barrier joins all sixteen warps and publishes every GroupA/DenChunk entry; done is this publication. There is no weight-array publication because the split already weighted its partials.

## t_76bdc896ac533e25 | pipeline.verb | 358 chars | 1 declarations

fused_d128_cp:396 design.tile_pipeline.phase.main.role.Main.info.verb.pv

After the issuer's after-thread-sync fence, enqueue the four ordered PV contractions and commit one PV completion phase. done denotes enqueue only; retire owns completion. Read Prob8 from TMEM, Value from its current shared slot, and the unit/V scale atoms. The first contraction of the first group overwrites NumState, later contractions accumulate into it.

## t_76daf261a804f72f | pipeline.verb | 754 chars | 1 declarations

fused_d256_tma:342 design.tile_pipeline.phase.main.role.Main.info.verb.prime

Enqueue K(cLo), K(cLo+1) when it exists, and V(cLo) on separate transaction barriers. Each operand uses two 16384-byte TMA requests, accounting for 32768 payload bytes; atoms do not share these barriers. Threads 0..63 enqueue the 1024-byte K and V atoms for groups cLo through min(cLo+2,cHi) into their own three-slot rings using cp.async. Commit one group, wait_group 0, fence the shared async proxy, wait for K(cLo), fence again and CTA-sync. This publishes Query and the primed atoms; V(cLo) and K(cLo+1) may remain in flight. Warps 0..3 copy cLo's atoms to its KeyScale/ValueScale slots and wait for TMEM stores. A before- thread-sync fence and CTA barrier publish these scales before start. The fixed default does not hoist this prime above prepare.

## t_77f75d1472f4f851 | pipeline.verb | 161 chars | 4 declarations

tc_split_flat_narrow:486 design.tile_pipeline.phase.group_max.role.Main.info.verb.record; tc_split_flat_wide:503 design.tile_pipeline.phase.group_max.role.Main.info.verb.record; tc_split_segmented_narrow:561 design.tile_pipeline.phase.group_max.role.Main.info.verb.record; tc_split_segmented_wide:595 design.tile_pipeline.phase.group_max.role.Main.info.verb.record

Subgroup representatives store each valid BmCovered value. A CTA barrier joins current K, scale and Score readers before the next iteration reuses their storage.

## t_79ba699849964f87 | pipeline.verb | 165 chars | 2 declarations

tc_split_segmented_narrow:634 design.tile_pipeline.phase.scan.role.Main.info.verb.store; tc_split_segmented_wide:668 design.tile_pipeline.phase.scan.role.Main.info.verb.store

Write valid MaxTail entries. No final trigger follows the stores; kernel_done supplies visibility. This invocation leaves the unfilled maximum-table bytes unchanged.

## t_7a3e28949c1df119 | pipeline.verb | 257 chars | 2 declarations

tcg_d128_fscan_cp_direct_narrow:540 design.tile_pipeline.phase.split.role.Main.info.verb.prime; tcg_d128_fscan_cp_direct_wide:557 design.tile_pipeline.phase.split.role.Main.info.verb.prime

For Ng>0 enqueue K/V(cLo) and both 512-byte atoms into slot zero, then commit one copy group per thread. Payload copies zero-fill out-of-range keys while scale padding is read. Ng=0 performs no transfer or commit. done means enqueue and commit, not arrival.

## t_7a940fabbf9511cc | pipeline.verb | 191 chars | 6 declarations

tcg_d256_flat_cp_narrow:674 design.tile_pipeline.phase.combine.role.Main.info.verb.denominator; tcg_d256_flat_cp_wide:688 design.tile_pipeline.phase.combine.role.Main.info.verb.denominator; tcg_d256_flat_tma_narrow:669 design.tile_pipeline.phase.combine.role.Main.info.verb.denominator; tcg_d256_segmented_cp_narrow:742 design.tile_pipeline.phase.combine.role.Main.info.verb.denominator; tcg_d256_segmented_tma_narrow:757 design.tile_pipeline.phase.combine.role.Main.info.verb.denominator; tcg_d256_segmented_tma_wide:742 design.tile_pipeline.phase.combine.role.Main.info.verb.denominator

CTA-sync publishes SharedDen after numerator accumulation. Every thread sums it in the output step's order; no warp shuffle or cross-CTA reduction occurs. SharedDen lasts through all readers.

## t_7b4e8c4be72b4f62 | pipeline.verb | 238 chars | 3 declarations

tcg_d256_segmented_cp_narrow:538 design.tile_pipeline.phase.scan.role.Main.info.verb.classify; tcg_d256_segmented_tma_narrow:553 design.tile_pipeline.phase.scan.role.Main.info.verb.classify; tcg_d256_segmented_tma_wide:538 design.tile_pipeline.phase.scan.role.Main.info.verb.classify

Valid threads load all SegChunk maxima into SegmentBatch, substituting -inf for absent segments, before comparing any. Capture Jump and tp against the incoming ScanState. Every loaded maximum is consumed before entering the replay helper.

## t_7bf33cb7175e896f | pipeline.verb | 674 chars | 2 declarations

tcg_d128_flat_cp_xpose_narrow:594 design.tile_pipeline.phase.split.role.Main.info.verb.replay.meaning; tcg_d128_flat_cp_xpose_wide:577 design.tile_pipeline.phase.split.role.Main.info.verb.replay.meaning

After its scan wait, each thread initializes DenState=0 and MaxState=-inf; load the entrance snapshot at (cLo-1)*R+rowG whenever cLo>0, including empty splits, and CurrentMax at cLo only for Ng>0. rowG is domain's r. done covers all replay work; each thread can continue with its own registers before other warps finish. The next stage or stage_partials CTA join covers completion. For Ng>0 each thread then commits an empty copy group left by the hoisted prime; this does not wait or join Query writers. The entrance load has no any_compute guard or T-bound clamp; a logically out-of-range empty-split read has no initialization or safety guarantee. Its value is discarded.

## t_7c25346798791720 | pipeline.verb | 707 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:513 design.tile_pipeline.phase.group_max.role.Main.info.verb.stage

Every thread waits until at most one copy group remains pending. For Long this proves K(g+1) and its atom when g<cHi; for non-Long it proves current K(g) and its atom. Fence shared async access and CTA-sync before reading shared atoms. Publish the next group's alternating KeyScale for Long, or the current fixed-base KeyScale otherwise; wait for TMEM stores, fence and CTA-sync. At g=cHi Long still reads the previously filled next physical atom slot and writes stale bytes to an unused alternate TMEM scale slot. The final empty commits retire all payload copies; no following QK consumes the stale scale. These CTA joins also publish Q. Copy-group completion has no transaction-byte count or ring parity.

## t_7c6ccf13e0858bd1 | pipeline.verb | 372 chars | 1 declarations

tcg_d256_segmented_cp_wide:827 design.tile_pipeline.phase.split.role.Main.info.verb.finish

A before-thread-sync fence and CTA barrier join final readers, then the first warp deallocates all 512 TMEM columns. Trigger programmatic launch completion at exit; combine's dependency wait establishes visibility of every partial. No next worktile reuses this CTA's resources, and the next invocation starts fresh numerical state without clearing unrelated scratch bytes.

## t_7cb0bc05bb964ff3 | pipeline.verb | 231 chars | 1 declarations

tcg_d128_fscan_tma_xpose_narrow:543 design.tile_pipeline.phase.split.role.Main.info.verb.dependency.meaning

Each thread executes the PDL predecessor-grid wait after its own preparation and before its first GroupMax read. done guarantees group_max completion and visibility; it does not join this CTA's warps or complete their Query writes.

## t_7cd6bbaf9bbc55fd | pipeline.verb | 450 chars | 1 declarations

tcg_d128_fscan_tma_xpose_wide:667 design.tile_pipeline.phase.combine.role.Main.info.verb.weights.meaning

Clear GroupAccumulator. Each thread reads MTail and computes at most one split Weight from PartM within its group's interval. The 128 entries per group's shared Weight segment are all written, with zero for missing splits. A valid Weight[s] occupies that group's local position s-slo(z). Each lane can continue to its numerator loads after its own store; done covers all stores but does not introduce a CTA join. The next prime action publishes them.

## t_7d2308b161989eb7 | pipeline.verb | 286 chars | 4 declarations

tcg_d128_segmented_cp_direct_narrow:561 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.first; tcg_d128_segmented_cp_direct_wide:578 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.first; tcg_d128_segmented_tma_direct_narrow:557 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.first; tcg_d128_segmented_tma_direct_wide:576 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.first

Each valid thread initializes ScanState and replays segment zero without reading SegMax[0,r]. Replay loads full Seg-sized GroupMax batches into ReplayBatch before ordered updates/stores, then handles a partial tail with scalar loads and updates. ScanState persists through the worktile.

## t_7dbe66834d39ff7e | pipeline.verb | 207 chars | 11 declarations

tcg_d128_flat_cp_direct_narrow:481 design.tile_pipeline.phase.flat_scan.role.Main.info.verb.signal; tcg_d128_flat_cp_direct_wide:495 design.tile_pipeline.phase.flat_scan.role.Main.info.verb.signal; tcg_d128_flat_cp_xpose_narrow:490 design.tile_pipeline.phase.flat_scan.role.Main.info.verb.signal; tcg_d128_flat_cp_xpose_wide:473 design.tile_pipeline.phase.flat_scan.role.Main.info.verb.signal; tcg_d128_flat_tma_direct_narrow:477 design.tile_pipeline.phase.flat_scan.role.Main.info.verb.signal; tcg_d128_flat_tma_direct_wide:491 design.tile_pipeline.phase.flat_scan.role.Main.info.verb.signal; tcg_d128_flat_tma_xpose_narrow:481 design.tile_pipeline.phase.flat_scan.role.Main.info.verb.signal; tcg_d128_flat_tma_xpose_wide:499 design.tile_pipeline.phase.flat_scan.role.Main.info.verb.signal; tcg_d128_segmented_cp_xpose_narrow:548 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.signal; tcg_d128_segmented_tma_xpose_narrow:564 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.signal; tcg_d128_segmented_tma_xpose_wide:548 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.signal

Signal split launch permission immediately after the wait, before scan arithmetic. Out-of-range lanes also signal; their subsequent scan/tail actions have no data effects. Permission does not publish MState.

## t_7dd252fabad04cb2 | pipeline.verb | 438 chars | 2 declarations

tcg_d128_flat_cp_xpose_narrow:432 design.tile_pipeline.phase.group_max.role.Main.info.verb.start; tcg_d128_flat_cp_xpose_wide:415 design.tile_pipeline.phase.group_max.role.Main.info.verb.start

For Ng>0 each thread first commits an empty group left by the hoisted prime. For Long then enqueue K(cLo+1) and atom, commit, and wait_group<1> to prove slot-zero copies. Fence shared async access and CTA-sync; publish the first KeyScale, wait for its TMEM stores, fence and CTA-sync, then enqueue and commit QK(cLo). For non-Long only the empty commit occurs when Ng>0; no copy wait or QK occurs. For Ng=0 this action has no data effect.

## t_7e086d4e0d1a4fc5 | pipeline.verb | 611 chars | 1 declarations

tcg_d128_flat_cp_direct_wide:675 design.tile_pipeline.phase.combine.role.Main.info.verb.reduce

Each consecutive four-warp group q owns [sLo(q),sHi(q)); each thread owns one channel. Initialize LocalA to zero, load full eight-split batches into NumeratorBatch and consume each before reuse, then add the scalar tail. Store GroupA, including zero for an empty interval. Each thread loads at most one PartB entry at sLo plus its local thread index, or uses zero beyond sHi. Apply combine's five-level tree within each warp; its first lane writes one GroupB element. All threads join at a CTA barrier. done publishes GroupPartials, meaning both shared GroupA and GroupB arrays, before the output group's reads.

## t_7e5df3996d14d430 | pipeline.verb | 333 chars | 12 declarations

tcg_d128_flat_tma_direct_narrow:442 design.tile_pipeline.phase.group_max.role.Main.info.verb.finish; tcg_d128_flat_tma_direct_wide:456 design.tile_pipeline.phase.group_max.role.Main.info.verb.finish; tcg_d128_flat_tma_xpose_narrow:446 design.tile_pipeline.phase.group_max.role.Main.info.verb.finish; tcg_d128_flat_tma_xpose_wide:464 design.tile_pipeline.phase.group_max.role.Main.info.verb.finish; tcg_d128_fscan_tma_direct_narrow:431 design.tile_pipeline.phase.group_max.role.Main.info.verb.finish; tcg_d128_fscan_tma_direct_wide:449 design.tile_pipeline.phase.group_max.role.Main.info.verb.finish; tcg_d128_fscan_tma_xpose_narrow:437 design.tile_pipeline.phase.group_max.role.Main.info.verb.finish; tcg_d128_segmented_cp_xpose_narrow:487 design.tile_pipeline.phase.group_max.role.Main.info.verb.finish; tcg_d128_segmented_tma_direct_narrow:497 design.tile_pipeline.phase.group_max.role.Main.info.verb.finish; tcg_d128_segmented_tma_direct_wide:516 design.tile_pipeline.phase.group_max.role.Main.info.verb.finish; tcg_d128_segmented_tma_xpose_narrow:503 design.tile_pipeline.phase.group_max.role.Main.info.verb.finish; tcg_d128_segmented_tma_xpose_wide:487 design.tile_pipeline.phase.group_max.role.Main.info.verb.finish

All issued K transactions and QK products have been waited. Fence and CTA-sync join final TMEM users, then the first warp deallocates 256 columns. Each CTA signals PDL launch completion after its stores and teardown; launch_ready is the grid-wide protocol event. Global visibility comes from kernel completion and the successor wait.

## t_7e78f04d45cba2fe | pipeline.verb | 270 chars | 2 declarations

tc_split_flat_narrow:658 design.tile_pipeline.phase.combine.role.Main.info.verb.write; tc_split_segmented_narrow:765 design.tile_pipeline.phase.combine.role.Main.info.verb.write

Execute combine and write owned BF16 O channels. Only thread 0 reads MaxTail and writes present LSE. done means stores consumed their register sources; kernel_done supplies caller-stream visibility. No scratch consumers or local numerical state remain after this kernel.

## t_7f6cd2557b7adc93 | pipeline.verb | 306 chars | 1 declarations

tcg_d128_fscan_tma_xpose_wide:567 design.tile_pipeline.phase.split.role.Main.info.verb.prefetch

Load NextMax from GroupMax[g+1,r] when g<cHi, otherwise use -inf, for all rows including padding. Enqueue K/V(g+1) and their atoms into the alternate slot when g<cHi; otherwise issue no transfer. Prior QK/PV completion has retired that slot's operand reads; done does not wait for newly enqueued transfers.

## t_7f7a7e4c13d57f4b | pipeline.verb | 325 chars | 1 declarations

fused_d128_group:407 design.tile_pipeline.phase.main.role.Even.info.verb.read

Wait for this role's current QK phase, toggle QK parity and apply the after fence. Four self-retiring 32-score TMEM loads per row form masked Score and GroupMax. The warp vote skips mask operations only for entirely visible groups. Retain all 128 scores per row; a 128-thread barrier joins the readouts before RawScore reuse.

## t_810a6c7b9aad3d9d | pipeline.verb | 147 chars | 3 declarations

tcg_d128_segmented_cp_xpose_narrow:519 design.tile_pipeline.phase.segment_max.role.Main.info.verb.signal; tcg_d128_segmented_tma_xpose_narrow:535 design.tile_pipeline.phase.segment_max.role.Main.info.verb.signal; tcg_d128_segmented_tma_xpose_wide:519 design.tile_pipeline.phase.segment_max.role.Main.info.verb.signal

Every thread signals PDL permission for segmented_scan at entry, before its predecessor wait. This is launch permission, not publication of SegMax.

## t_82414da6ec9f0137 | pipeline.verb | 445 chars | 1 declarations

tcg_d128_fscan_tma_xpose_wide:680 design.tile_pipeline.phase.combine.role.Main.info.verb.sum_b

Each group loads one PartB per valid local split and zero for missing entries, multiplies by its shared Weight, and performs combine's 32-entry tree using warp shuffles. One lane per warp writes BinB, including zero bins. A CTA barrier across all 16 warps publishes every GroupA and BinB entry and joins all weight readers. These shared partials remain live until the first group finishes reading them; there is no further reuse within this CTA.

## t_83f9fca35408b0d4 | pipeline.verb | 443 chars | 1 declarations

tcg_d128_flat_tma_direct_wide:675 design.tile_pipeline.phase.combine.role.Main.info.verb.write

Group zero alone consumes GroupA and DenChunk, performs combine's ordered final sums, and writes its BF16 channels. TotalB is replicated per output channel; thread zero additionally reads MTail and writes LSE when present. Other groups perform no data accesses in this action. Shared partials stay unchanged until all readers finish; no next worktile reuses them. Initial destinations are never read, and kernel completion provides visibility.

## t_8421c7e857018f16 | pipeline.verb | 542 chars | 1 declarations

tcg_d256_segmented_cp_wide:880 design.tile_pipeline.phase.combine.role.Output.info.verb.write

Every Output thread reads all SP*4 shared denominator entries in the output step's order and forms its own denominator and reciprocal. For each of its two channels it then reads all SP GroupNum entries, sums and stores BF16 O. Thread zero alone reads Mtail and writes LSE when present. Shared sums are never overwritten and stay live through these reads, even after Partners exits. Ordinary stores consume their local sources; kernel completion provides caller-stream output visibility. No additional completion wait or launch trigger occurs.

## t_8424b7b4720a9aad | pipeline.verb | 424 chars | 5 declarations

tc_fused:336 design.tile_pipeline.phase.main.role.Main.info.verb.qk; tc_split_flat_narrow:591 design.tile_pipeline.phase.split.role.Main.info.verb.qk; tc_split_flat_wide:608 design.tile_pipeline.phase.split.role.Main.info.verb.qk; tc_split_segmented_narrow:698 design.tile_pipeline.phase.split.role.Main.info.verb.qk; tc_split_segmented_wide:732 design.tile_pipeline.phase.split.role.Main.info.verb.qk

Load Q/K fragments and perform scores with synchronous FP8 MMA, in D/Block channel blocks and 8-key result fragments. Retain the entire scaled Score in registers. A warp-wide mask intersection skips individual mask tests only when the whole group lies inside every row's bounds. Local pair maxima and two four-lane shuffle stages produce the full-group maximum. done includes current Q/K reads and the register score result.

## t_85587f55a8acce26 | pipeline.verb | 333 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:499 design.tile_pipeline.phase.group_max.role.Main.info.verb.prime

For Ng>0 cooperatively enqueue K(cLo) and its scale atom into slot zero, then each thread commits its copies as one group. Each payload copy moves 16 bytes, zero-filling keys beyond Skv; scale atoms use unguarded 16-byte copies from allocated padding. For Ng=0 issue neither copies nor a commit. done denotes submission, not arrival.

## t_858b2fb5a10bbd21 | pipeline.verb | 427 chars | 2 declarations

tcg_d128_flat_cp_direct_narrow:611 design.tile_pipeline.phase.split.role.Main.info.verb.finish; tcg_d128_flat_cp_direct_wide:625 design.tile_pipeline.phase.split.role.Main.info.verb.finish

Fence and CTA-sync join final TMEM users; first warp deallocates. All nonempty copy groups and products were already waited; no extra wait-all is issued. Signal PDL completion after partial stores and teardown. No CTA receives another tile; later kernels have fresh local resources. Global intermediates survive all declared readers, and caller-stream completion orders subsequent invocations without retaining numerical state.

## t_86c1ae9ede0f48df | pipeline.verb | 454 chars | 4 declarations

tcg_d128_fscan_cp_xpose_narrow:602 design.tile_pipeline.phase.split.role.Main.info.verb.finish; tcg_d128_fscan_cp_xpose_wide:619 design.tile_pipeline.phase.split.role.Main.info.verb.finish; tcg_d128_fscan_tma_xpose_wide:607 design.tile_pipeline.phase.split.role.Main.info.verb.finish; tcg_d128_segmented_cp_xpose_wide:747 design.tile_pipeline.phase.split.role.Main.info.verb.finish

A before-thread-sync fence and CTA barrier join all final TMEM accesses; the first warp deallocates. Every requested transfer and matrix product was already waited, including on the last group. Signal PDL launch completion after local teardown and partial stores. Each CTA ends after its one tile. Subsequent kernels have fresh local resources; scratch regions remain live through their declared consumers, and caller-stream completion orders later uses.

## t_876bce86dec2d3ef | pipeline.verb | 706 chars | 2 declarations

tcg_d128_fscan_cp_xpose_narrow:426 design.tile_pipeline.phase.group_max.role.Main.info.verb.stage; tcg_d128_fscan_cp_xpose_wide:443 design.tile_pipeline.phase.group_max.role.Main.info.verb.stage

Each thread waits until at most one committed copy group remains, then fences shared async access and CTA-syncs. For Long this proves K(g+1) and its atom when g<cHi; copy that atom into the alternating KeyScale slot, wait for TMEM stores, fence and CTA-sync. At g=cHi, still read the next physical atom slot and write its stale bytes into the alternate TMEM scale slot; no following QK consumes them. With Ng>=8 that slot was filled previously. The tail's empty commits preserve the copy-group accounting and leave no payload transfer outstanding. For non-Long, the wait proves current K(g) and atom; publish KeyScale at its fixed base, wait for TMEM stores, fence and CTA-sync. These joins also publish Q.

## t_8867c161999b15d1 | pipeline.verb | 295 chars | 1 declarations

fused_d256_tma:429 design.tile_pipeline.phase.main.role.Main.info.verb.drain

Wait for the final outstanding PV and apply the completion fence. All QK and TMA payloads have been waited and consumed; every nonempty atom cp.async group was completed by prime or wait_group 1. Remaining empty commits need no payload drain. The unused tail scale stores already waited locally.

## t_88902c04663b5167 | pipeline.verb | 786 chars | 1 declarations

fused_d128_group:341 design.tile_pipeline.phase.main.role.Even.info.verb.cold

All 256 threads issue the generic hoisted copy of cLo into global slot zero of Key, Value and their atoms, then each commits a cp.async group. Payload pieces use the 256-thread partition; the first warp copies atoms. Out-of-range payload pieces zero-fill; allocated padding atom bytes are loaded. done means enqueue only. This copy remains present in GSPLIT: the dedicated prime repeats cLo's writes with 128 threads. There is no cross-role completion join before that re-prime or the first QK/PV; the first K-slot reuse also lacks such a join with the other role's cold copies. Odd with Ng=1 never waits for its cold payloads. This missing wait is preserved without asserting duplicate-write ordering or a block-exit drain. Active roles eventually prove their own cold copies in prime.

## t_894cd89015cab230 | pipeline.verb | 660 chars | 1 declarations

tcg_d128_fscan_tma_xpose_narrow:580 design.tile_pipeline.phase.split.role.Main.info.verb.write

All threads cooperatively read 16-byte chunks of actual shared bytes for valid staged rows and write global PartA through domain's head-major map. FP32 bits are preserved, including possible empty-split corruption. Each valid row owner separately writes unweighted DenState to PartB and its end state to PartM; empty splits write PartB=0 and their prefix state to PartM from registers. done means stores issued with sources consumed; no later local reuse occurs and kernel completion provides global visibility. No O/LSE is written here. Combine reads and weights every PartA, so corruption can reach TotalA and O; there is no PartA data path to TotalB or LSE.

## t_8a3021811f2af342 | pipeline.verb | 102 chars | 4 declarations

tcg_d128_segmented_cp_direct_narrow:534 design.tile_pipeline.phase.segment_max.role.Main.info.verb.dependency; tcg_d128_segmented_cp_direct_wide:551 design.tile_pipeline.phase.segment_max.role.Main.info.verb.dependency; tcg_d128_segmented_tma_direct_narrow:530 design.tile_pipeline.phase.segment_max.role.Main.info.verb.dependency; tcg_d128_segmented_tma_direct_wide:549 design.tile_pipeline.phase.segment_max.role.Main.info.verb.dependency

All threads wait for group_max completion and global visibility before any row guard or GroupMax load.

## t_8b3762ccd3be7370 | pipeline.verb | 147 chars | 4 declarations

tcg_d128_segmented_cp_direct_narrow:532 design.tile_pipeline.phase.segment_max.role.Main.info.verb.signal; tcg_d128_segmented_cp_direct_wide:549 design.tile_pipeline.phase.segment_max.role.Main.info.verb.signal; tcg_d128_segmented_tma_direct_narrow:528 design.tile_pipeline.phase.segment_max.role.Main.info.verb.signal; tcg_d128_segmented_tma_direct_wide:547 design.tile_pipeline.phase.segment_max.role.Main.info.verb.signal

All threads signal segmented_scan launch permission at entry, before the wait and row guard. This is launch_ready only; it does not publish SegMax.

## t_8b5d864d55a52d14 | pipeline.verb | 423 chars | 1 declarations

tcg_d128_fscan_tma_direct_wide:643 design.tile_pipeline.phase.combine.role.Main.info.verb.sum_b

Each thread loads its candidate PartB, or uses zero beyond its interval, and multiplies by its own Weight. Each warp performs combine's halving tree, and its first lane writes one GroupB entry. Empty groups write zero GroupA/GroupB and participate in both CTA barriers. A CTA-sync publishes all GroupA and GroupB entries and joins all weight readers. These three shared arrays are not overwritten during the remaining work.

## t_8bc22240cc82c702 | pipeline.verb | 305 chars | 1 declarations

fused_d128_key:425 design.tile_pipeline.phase.main.role.Main.info.verb.combine

Store the two carried DenSlice values into the same Exchange storage used for HalfMax, then use a CTA barrier and ordered loads/add to form DenFinal in both warpgroups. The loop's later barriers already retired the last maximum exchange readers. DenFinal replaces each thread's local denominator register.

## t_8cfeadafbed02e02 | pipeline.verb | 453 chars | 1 declarations

fused_d128_group:463 design.tile_pipeline.phase.main.role.Even.info.verb.pv

The leading thread applies the after fence and enqueues four ordered full-width PV contractions, reading its Prob8, Value(g), ValueScale and shared UnitScale; commit its own PV barrier. The first contraction at cLo overwrites NumState; all later contractions accumulate. done means enqueue only. The other role waits for this PV to complete before rescaling or issuing the next PV, so at most one PV is in flight while the QK streams remain independent.

## t_8d4617fccffa4845 | pipeline.verb | 497 chars | 1 declarations

tcg_d128_segmented_tma_direct_wide:767 design.tile_pipeline.phase.combine.role.Main.info.verb.write

Only group zero performs combine's final sums and output stores. Each of its threads reads all GroupB entries and its channel's GroupA entries; TotalB is replicated per output channel. Thread zero additionally reads MTail and writes LSE when present. Other groups do no final arithmetic or stores. Caller destinations are not read. Shared storage survives group zero's reads; there is no post-output CTA barrier, next worktile or launch signal. Kernel completion supplies global output visibility.

## t_8ea9034ad4bb750e | pipeline.verb | 580 chars | 1 declarations

fused_d128_group:396 design.tile_pipeline.phase.main.role.Even.info.verb.retire

Own(j) waits for this role's previous PV, g-2, when j>0, advances its own completion counter and applies the after fence. It frees that PV's Prob8, V and value-scale readers; at j=0 it completes without a wait. The other role's PV(g-1) may remain in flight during current score/probability work. LastPV waits this role's own last PV if it owns any groups and applies the after fence in both cases. Active roles' final empty commits and wait_group 1 already retire dedicated payload copies; no extra copy wait is inserted. An idle Odd role leaves its cold-copy completion unproven.

## t_8ee0572f0dd52fb4 | pipeline.verb | 294 chars | 5 declarations

tc_fused:349 design.tile_pipeline.phase.main.role.Main.info.verb.probability_value; tc_split_flat_narrow:600 design.tile_pipeline.phase.split.role.Main.info.verb.probability_value; tc_split_flat_wide:617 design.tile_pipeline.phase.split.role.Main.info.verb.probability_value; tc_split_segmented_narrow:707 design.tile_pipeline.phase.split.role.Main.info.verb.probability_value; tc_split_segmented_wide:741 design.tile_pipeline.phase.split.role.Main.info.verb.probability_value

For chunk t, compute Prob and Pfrag from Score, append the two 8-key halves to E in the defined order, and consume Pfrag with synchronous BF16 PV MMA over all D output channels. done includes this fragment's Pfrag/Vbf reads and updated register numerator. Pfrag can then be overwritten for t+1.

## t_8f726c1a9bf878d2 | pipeline.verb | 415 chars | 1 declarations

tcg_d128_segmented_cp_direct_wide:769 design.tile_pipeline.phase.combine.role.Main.info.verb.sum_a

Four consecutive warps form group z, owning interval a_z..b_z-1; its thread d owns channel d. Initialize NumeratorState to zero, load full eight-split batches into NumeratorBatch and consume each batch before reuse. Read the terminal suffix one split at a time. Write MergeA[z,d] to shared memory, including zero for empty intervals. done guarantees each thread's own store; cross-thread publication occurs at join.

## t_90032ff2fa913fa9 | pipeline.verb | 436 chars | 1 declarations

fused_d128_key:341 design.tile_pipeline.phase.main.role.Main.info.verb.prepare

Load Q and query scales into QueryPrefetch before allocation, with the domain step's padding values. The first warp allocates TensorMemory; a CTA barrier publishes its base and that warp relinquishes the allocation permit. Store Q to Query, and let warps 0..3 store QueryScale and UnitScale, waiting for the TMEM stores. Initialize RowState as in normalize. QueryPrefetch is now dead; Query and these two scale atoms survive all groups.

## t_903635cf28334a0d | pipeline.verb | 296 chars | 1 declarations

fused_d256_tma:380 design.tile_pipeline.phase.main.role.Main.info.verb.read_score

Wait for QK(g), advance its parity once and apply the completion fence. Each half reads its 64 columns in two 32-column loads, each with its own load-completion wait, masks and reduces HalfMax, and retains Score for probabilities. This retires QK's Key/KeyScale reads but does not retire PV(g-1).

## t_90fb96eb3e8d5fac | pipeline.verb | 444 chars | 2 declarations

tcg_d128_fscan_cp_direct_narrow:556 design.tile_pipeline.phase.split.role.Main.info.verb.stage; tcg_d128_fscan_cp_direct_wide:573 design.tile_pipeline.phase.split.role.Main.info.verb.stage

Every thread waits for all but the newest committed copy group, proving its current K/V and atom copies complete, including on the final iteration after the empty commit. Fence shared access and CTA-sync to publish all issuers' copies. Write current KeyScale/ValueScale, wait for TMEM stores, fence and CTA-sync. Prior PV completion protects scale/operand reuse; these joins retire previous GroupPV readers before QK overwrites the same region.

## t_9116af9e01675697 | pipeline.verb | 470 chars | 6 declarations

tcg_d128_flat_tma_direct_narrow:413 design.tile_pipeline.phase.group_max.role.Main.info.verb.setup; tcg_d128_flat_tma_direct_wide:427 design.tile_pipeline.phase.group_max.role.Main.info.verb.setup; tcg_d128_fscan_tma_direct_narrow:402 design.tile_pipeline.phase.group_max.role.Main.info.verb.setup; tcg_d128_fscan_tma_direct_wide:420 design.tile_pipeline.phase.group_max.role.Main.info.verb.setup; tcg_d128_segmented_tma_direct_narrow:468 design.tile_pipeline.phase.group_max.role.Main.info.verb.setup; tcg_d128_segmented_tma_direct_wide:487 design.tile_pipeline.phase.group_max.role.Main.info.verb.setup

Establish coordinates, masks and initially zero completion parities. RingUsed=3 for Long, otherwise 2; ScaleCol(g)=136+8*(g%2) for Long, otherwise 136. Three shared slots remain allocated in both cases. Valid rows of the first/last split issue the outside -inf fills. Thread zero initializes/publishes transaction barriers and issues TMA/MMA; all four warps stage Q/scales and synchronize. Only warps with a valid row perform scalar score readout. q0>=Sq is unreachable.

## t_915b28584a012554 | pipeline.verb | 292 chars | 1 declarations

tcg_d256_segmented_cp_wide:623 design.tile_pipeline.phase.scan.role.Main.info.verb.begin

Trigger the split's programmatic launch immediately after await, before the scan. Each thread owns row r0+thread_id; out-of-range threads return after triggering, while valid threads initialize ScanState to -inf. The trigger permits independent split preparation but publishes no scan result.

## t_91793ffa53d96766 | pipeline.verb | 161 chars | 1 declarations

tcg_d256_segmented_cp_wide:574 design.tile_pipeline.phase.segment_max.role.Main.info.verb.trigger

Every thread grants the successor scan's programmatic launch permission at kernel entry, before waiting for group_max. This publishes no GroupMax or SegMax data.

## t_9185945212b9c8a8 | pipeline.verb | 431 chars | 2 declarations

tc_split_flat_narrow:519 design.tile_pipeline.phase.scan.role.Main.info.verb.load; tc_split_flat_wide:536 design.tile_pipeline.phase.scan.role.Main.info.verb.load

Let N64=floor(T/64), N16=floor((T mod 64)/16), N1=T mod 16, Nb=N64+N16+N1. For batch j, (Base(j),Len(j)) is (64*j,64) if j<N64, (64*N64+16*(j-N64),16) if N64<=j<N64+N16, otherwise (64*N64+16*N16+j-N64-N16,1). Issue all Len(j) independent global loads for each valid row before its dependent recurrence. BmCovered supplies Cblock; incoming PriorMax supplies Ublock. done makes the transposed [group,row] batch readable in registers.

## t_91ec26bf77eefb91 | pipeline.verb | 407 chars | 2 declarations

fused_d256_cp:423 design.tile_pipeline.phase.main.role.Main.info.verb.write; fused_d256_tma:441 design.tile_pipeline.phase.main.role.Main.info.verb.write

Each half reads its 128 NumState channels in 32-channel loads with load-completion waits, and issues value_output's BF16 RNE stores directly to caller O. Half zero alone stores optional FP32 LSE. Padded rows compute and read out but store neither output. Initial destination bytes are not read. done means ordinary stores were issued and register sources consumed, not a host or device-wide completion wait.

## t_91eebc124d8b1ef7 | pipeline.verb | 303 chars | 3 declarations

tcg_d256_segmented_cp_narrow:532 design.tile_pipeline.phase.scan.role.Main.info.verb.first; tcg_d256_segmented_tma_narrow:547 design.tile_pipeline.phase.scan.role.Main.info.verb.first; tcg_d256_segmented_tma_wide:532 design.tile_pipeline.phase.scan.role.Main.info.verb.first

Valid threads replay segment 0. The replay helper loads each full SegLen-group batch into GroupBatch before advancing state and writing its entries in order; an incomplete final batch uses scalar load/advance/store. GroupBatch reads finish before its next overwrite. The same helper serves replay below.

## t_925d70b4f41d2455 | pipeline.verb | 489 chars | 2 declarations

tcg_d128_fscan_cp_xpose_narrow:441 design.tile_pipeline.phase.group_max.role.Main.info.verb.finish; tcg_d128_fscan_cp_xpose_wide:458 design.tile_pipeline.phase.group_max.role.Main.info.verb.finish

All enqueued K/atom copies and QK products have been waited; only completed empty copy groups can remain. A before-thread-sync fence and CTA barrier join final TMEM users; the first warp deallocates 256 columns. Each CTA signals PDL launch completion after its global stores and local teardown; launch_ready applies when the whole grid satisfies the protocol. Actual global result visibility is established by kernel completion and the successor dependency wait, not by enqueueing a store.

## t_92d304d1e4f69d43 | pipeline.verb | 291 chars | 6 declarations

tcg_d128_flat_tma_xpose_narrow:564 design.tile_pipeline.phase.split.role.Main.info.verb.setup; tcg_d128_flat_tma_xpose_wide:582 design.tile_pipeline.phase.split.role.Main.info.verb.setup; tcg_d128_fscan_tma_xpose_narrow:528 design.tile_pipeline.phase.split.role.Main.info.verb.setup; tcg_d128_segmented_cp_xpose_narrow:635 design.tile_pipeline.phase.split.role.Main.info.verb.setup; tcg_d128_segmented_tma_xpose_narrow:651 design.tile_pipeline.phase.split.role.Main.info.verb.setup; tcg_d128_segmented_tma_xpose_wide:635 design.tile_pipeline.phase.split.role.Main.info.verb.setup

Establish head-major coordinates and row validity; rowsLive is always true. Initialize QK/PV and the two combined K/V transaction barriers at parity zero and publish initialization. All four warps cooperate; one thread maps to each row, and thread zero issues TMA/MMA. q0>=Sq is unreachable.

## t_930673c05697e17f | pipeline.verb | 321 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:510 design.tile_pipeline.phase.group_max.role.Main.info.verb.prefetch

Enqueue K(g+2) and its atom for Long, otherwise K(g+1) and its atom, only when that group is <=cHi. Each thread then commits a copy group even when it is empty. done means submission only. Ring reuse follows prior QK completion and joined score readers; payload and atoms for a group belong to the same per-thread commit.

## t_934a3ea788fdd868 | pipeline.verb | 416 chars | 2 declarations

tcg_d128_segmented_cp_direct_narrow:480 design.tile_pipeline.phase.group_max.role.Main.info.verb.start; tcg_d128_segmented_cp_direct_wide:497 design.tile_pipeline.phase.group_max.role.Main.info.verb.start

For Ng>0 commit the empty copy group left by the hoisted prime. For Long additionally enqueue K(cLo+1) and atom, commit, then wait for all but the newest copy group. Fence shared async access and CTA-sync to publish slot zero's payload and atom. Publish the first KeyScale, wait for its TMEM stores, fence and CTA-sync, then enqueue and commit QK(cLo). Non-Long issues no product or wait; Ng=0 also issues no commit.

## t_935e54bcdbfaedc6 | pipeline.verb | 483 chars | 7 declarations

tcg_d256_flat_cp_narrow:561 design.tile_pipeline.phase.split.role.Main.info.verb.replay; tcg_d256_flat_cp_wide:575 design.tile_pipeline.phase.split.role.Main.info.verb.replay; tcg_d256_flat_tma_narrow:559 design.tile_pipeline.phase.split.role.Main.info.verb.replay; tcg_d256_flat_tma_wide:573 design.tile_pipeline.phase.split.role.Main.info.verb.replay; tcg_d256_segmented_cp_narrow:632 design.tile_pipeline.phase.split.role.Main.info.verb.replay; tcg_d256_segmented_tma_narrow:647 design.tile_pipeline.phase.split.role.Main.info.verb.replay; tcg_d256_segmented_tma_wide:632 design.tile_pipeline.phase.split.role.Main.info.verb.replay

Initialize DenPart=0. Load ReplayPrevious at l-1 when l>0, otherwise use -inf; load ReplayCurrent at l only when C>0. All threads make these reads, including padding using ReadRow. The source l-1 load has no C>0 or l<=GCount guard: empty splits can read beyond logical Mscan into following workspace bytes. Its value is discarded and does not define their zero partials; no clamp, initialization or extra wait is supplied. NumState remains uninitialized until the first PV overwrite.

## t_9385add34b2ca293 | pipeline.verb | 532 chars | 4 declarations

tc_split_flat_narrow:607 design.tile_pipeline.phase.split.role.Main.info.verb.write; tc_split_flat_wide:624 design.tile_pipeline.phase.split.role.Main.info.verb.write; tc_split_segmented_narrow:714 design.tile_pipeline.phase.split.role.Main.info.verb.write; tc_split_segmented_wide:748 design.tile_pipeline.phase.split.role.Main.info.verb.write

Merge DenPart with two four-lane shuffle-add stages. Nonempty intervals read MaxTail and weight with retained CurrentMax. Empty intervals skip this weighting read and write zero partials. Store valid rows as FP32 PartialA and PartialB, without O/LSE writes. done consumes register sources; kernel_done supplies visibility. There is no programmatic trigger, TMEM state or next worktile. Every payload copy was waited before use; the final empty commit needs no payload drain. Later invocations initialize fresh local numerical state.

## t_938c1766b1f69341 | pipeline.verb | 823 chars | 2 declarations

tcg_d128_fscan_cp_xpose_narrow:554 design.tile_pipeline.phase.split.role.Main.info.verb.prefix.meaning; tcg_d128_fscan_cp_xpose_wide:571 design.tile_pipeline.phase.split.role.Main.info.verb.prefix.meaning

For Ng>0 each thread first commits the empty group left by hoisting the initial copies; Ng=0 commits nothing. Initialize MaxState=-inf and DenState=0 and perform scan's bounded prefix, loading batches of 16 GroupMax values into PrefixBatch before ordered updates, then a scalar remainder. The register batch is consumed before reuse. All rows participate; padding uses that head's first real row. Load CurrentMax from GroupMax[cLo,r] only for Ng>0, otherwise keep -inf. CurrentMax/NextMax cache GroupMax values, not delayed states. No global scan snapshot or MTail is read. done covers all prefix work; a warp can continue with its own registers before others finish. The following stage or stage_partials CTA join covers collective completion. This work can overlap primed K/V copy groups and also occurs for empty splits.

## t_9547933080d3d62e | pipeline.verb | 313 chars | 1 declarations

fused_d128_cp:379 design.tile_pipeline.phase.main.role.Main.info.verb.retire

NumState(g) waits for the current committed PV phase, advances its parity once and applies the after-thread-sync fence on all threads. This completes NumState updates and all PV reads of Value, ValueScale and Prob8 before the next iteration or output. There is no per-group numerator readback or tail CTA barrier.

## t_97a532fa6dd73b68 | pipeline.verb | 615 chars | 1 declarations

tcg_d128_fscan_cp_xpose_wide:681 design.tile_pipeline.phase.combine.role.Main.info.verb.partials

Each group consumes its first numerator batch, then full batches of eight and a scalar remainder, accumulating combine's NumGroup in NumAccumulator before storing it to shared memory. Reuse each register batch after consumption. Each thread also loads its one valid PartB or zero, multiplies by its own shared weight, and performs the five-step warp shuffle-down tree. Each warp's first lane stores DenGroup. The following CTA barrier publishes both subtotal arrays; done includes that barrier and all groups' weight reads. The three shared arrays use distinct storage and remain unmodified through their consumers.

## t_97cb38cf3522f2c8 | pipeline.verb | 795 chars | 3 declarations

tcg_d128_fscan_cp_xpose_narrow:541 design.tile_pipeline.phase.split.role.Main.info.verb.prepare.meaning; tcg_d128_fscan_cp_xpose_wide:558 design.tile_pipeline.phase.split.role.Main.info.verb.prepare.meaning; tcg_d128_fscan_tma_xpose_wide:548 design.tile_pipeline.phase.split.role.Main.info.verb.prepare.meaning

Prefetch Q and scale words into QueryPrefetch, allocate 256 TMEM columns with the first warp, publish the base by CTA barrier and relinquish the permit. Each thread writes Query, QueryScale and UnitScale, waits for its TMEM stores, discards QueryPrefetch and clears its numerator before its dependency/prefix work. These ordinary shared Q writes are not followed by a CTA join here. done guarantees completion of preparation across all warps; later actions may start before that collective completion. The first stage CTA join proves it for Ng>0; for Ng=0 only the barrier after the stage_partials writes proves it. The asynchronous completion expresses this cross-warp progression, not a different copy instruction. Preparation also occurs for empty splits and can precede group-max completion.

## t_97d6eb697a429479 | pipeline.verb | 551 chars | 2 declarations

tcg_d128_fscan_cp_xpose_narrow:418 design.tile_pipeline.phase.group_max.role.Main.info.verb.start; tcg_d128_fscan_cp_xpose_wide:435 design.tile_pipeline.phase.group_max.role.Main.info.verb.start

For Long each thread commits an empty group for the already-hoisted slot-zero prime, enqueues K(cLo+1) and its atom into slot one, commits, then waits until at most one copy group remains. This proves slot zero payload and atom complete. Fence shared async access and CTA-sync; publish the first KeyScale, wait for TMEM stores, fence and CTA-sync, then enqueue QK(cLo) and commit its completion phase. For non-Long and Ng>0 commit only the empty group replacing the hoisted prime; for Ng=0 do nothing. The non-Long cases issue no product or copy wait.

## t_9816702de144cffe | pipeline.verb | 394 chars | 2 declarations

tcg_d128_fscan_cp_xpose_narrow:561 design.tile_pipeline.phase.split.role.Main.info.verb.prefetch; tcg_d128_fscan_cp_xpose_wide:578 design.tile_pipeline.phase.split.role.Main.info.verb.prefetch

Load NextMax from GroupMax[g+1,r] when g<cHi, otherwise use -inf, for all rows including padding. Enqueue K/V(g+1) and their atoms into the alternate slot when g<cHi; otherwise issue no transfer. Every thread then commits a group, including an empty group on the last iteration. Prior QK/PV completion has retired the refilled slot's operand reads; done does not wait for newly enqueued copies.

## t_9820c14d98985a7d | pipeline.verb | 276 chars | 2 declarations

tcg_d128_flat_cp_xpose_narrow:436 design.tile_pipeline.phase.group_max.role.Main.info.verb.prefetch; tcg_d128_flat_cp_xpose_wide:419 design.tile_pipeline.phase.group_max.role.Main.info.verb.prefetch

Enqueue K(g+2) and atom for Long, otherwise K(g+1) and atom, only when that group is <=cHi. Otherwise issue no transfer. Each thread commits once even when the group contains no copies; done means enqueue and commit. Previous QK completion and reader joins protect ring reuse.

## t_983ab6e7b7c4b332 | pipeline.verb | 265 chars | 1 declarations

tcg_d128_fscan_tma_xpose_wide:439 design.tile_pipeline.phase.group_max.role.Main.info.verb.prefetch

Enqueue K(g+1) and its atom into the alternate slot only when g<cHi; otherwise issue no transfer. done is enqueue only. Prior QK completion and the joined score readers protect ring reuse. There is no cp.async group stream; the D128 atom shares the TMA transaction.

## t_988c5af9735a1020 | pipeline.verb | 147 chars | 1 declarations

fused_d128_group:477 design.tile_pipeline.phase.main.role.Even.info.verb.join

A CTA barrier after each role's LastPV joins both matrix drains; all QK was retired by read. This barrier alone is not a cold-copy completion wait.

## t_98f4a3a6c8a7768b | pipeline.verb | 475 chars | 6 declarations

tcg_d128_flat_tma_xpose_narrow:417 design.tile_pipeline.phase.group_max.role.Main.info.verb.setup; tcg_d128_flat_tma_xpose_wide:435 design.tile_pipeline.phase.group_max.role.Main.info.verb.setup; tcg_d128_fscan_tma_xpose_narrow:408 design.tile_pipeline.phase.group_max.role.Main.info.verb.setup; tcg_d128_segmented_cp_xpose_narrow:458 design.tile_pipeline.phase.group_max.role.Main.info.verb.setup; tcg_d128_segmented_tma_xpose_narrow:474 design.tile_pipeline.phase.group_max.role.Main.info.verb.setup; tcg_d128_segmented_tma_xpose_wide:458 design.tile_pipeline.phase.group_max.role.Main.info.verb.setup

Establish coordinates, masks and initially zero completion parities. RingUsed=3 for Long, otherwise 2; ScaleCol(g)=136+8*(g%2) for Long, otherwise 136. Three shared slots remain allocated in both cases. Valid rows of the first/last split issue the outside -inf fills. Thread zero initializes/publishes transaction barriers and issues TMA/MMA; all four warps stage Q/scales and synchronize. All warps perform scalar score readout, including padded rows. q0>=Sq is unreachable.

## t_990966e72208f128 | pipeline.verb | 591 chars | 2 declarations

tcg_d128_flat_cp_xpose_narrow:421 design.tile_pipeline.phase.group_max.role.Main.info.verb.setup; tcg_d128_flat_cp_xpose_wide:404 design.tile_pipeline.phase.group_max.role.Main.info.verb.setup

Establish coordinates, masks and initially zero completion parities. RingUsed=3 for Long, otherwise 2; ScaleCol(g)=136+8*(g%2) for Long, otherwise 136. Three shared slots remain allocated in both cases. Valid rows of the first/last split issue the outside -inf fills. Thread zero initializes matrix completion barriers and issues MMA. All four warps copy payloads and the first warp copies scale atoms; each thread commits/waits its own copy groups. All four warps stage Q/scales, synchronize and read scores, including padded rows. No TMA transaction barrier is used; q0>=Sq is unreachable.

## t_99198ea398b9851e | pipeline.verb | 294 chars | 2 declarations

tcg_d256_flat_cp_narrow:401 design.tile_pipeline.phase.group_max.role.Main.info.verb.prime; tcg_d256_flat_cp_wide:415 design.tile_pipeline.phase.group_max.role.Main.info.verb.prime

For C>0, all eight warps issue the 32768-byte K(l) tile as 16-byte cp.async copies; threads 0..63 also issue its 1024-byte atom. Each thread commits its payload and atom requests together. At C=0 there are no requests or commits; the action retains its event. done denotes enqueue, not arrival.

## t_995c9bc4ec6a9b48 | pipeline.verb | 311 chars | 2 declarations

tcg_d128_flat_cp_direct_narrow:576 design.tile_pipeline.phase.split.role.Main.info.verb.prime; tcg_d128_flat_cp_direct_wide:590 design.tile_pipeline.phase.split.role.Main.info.verb.prime

For Ng>0 cooperatively enqueue K/V(cLo) and their 512-byte atoms into slot zero using 16-byte asynchronous copies, then commit one group per thread. Payload copies zero-fill keys past Skv; scale copies read allocated padding bytes. For Ng=0 issue neither copies nor commits. done is enqueue/commit, not arrival.

## t_9a1a500a893e4125 | pipeline.verb | 650 chars | 2 declarations

tcg_d128_fscan_cp_direct_narrow:422 design.tile_pipeline.phase.group_max.role.Main.info.verb.stage; tcg_d128_fscan_cp_direct_wide:439 design.tile_pipeline.phase.group_max.role.Main.info.verb.stage

Every thread waits for all but the newest committed copy group, then fences shared access and CTA-syncs. For Long this proves K(g+1) and atom when g<cHi; write its alternating KeyScale. At g=cHi the same wait still executes, and the next physical atom slot's stale bytes are still copied to the alternate scale slot without a later QK consumer; Ng>=8 ensures this slot was previously filled. Non-Long instead publishes current K(g)'s atom at the fixed KeyScale base. Wait for TMEM stores, fence and CTA-sync in both cases. The last iteration's wait after its empty commit completes all copies. These joins also publish Q and protect scale-slot reuse.

## t_9adfbffb67206f28 | pipeline.verb | 363 chars | 2 declarations

tcg_d128_fscan_cp_direct_narrow:436 design.tile_pipeline.phase.group_max.role.Main.info.verb.finish; tcg_d128_fscan_cp_direct_wide:453 design.tile_pipeline.phase.group_max.role.Main.info.verb.finish

All issued K/atom copies and QK products have been waited; no extra copy drain is issued. Fence and CTA-sync join final TMEM users, then the first warp deallocates 256 columns. Each CTA signals PDL launch completion after its stores and teardown; launch_ready is the grid-wide protocol event. Global visibility comes from kernel completion and the successor wait.

## t_9b212d6309f9e7c4 | pipeline.verb | 997 chars | 1 declarations

tcg_d128_fscan_tma_xpose_narrow:570 design.tile_pipeline.phase.split.role.Main.info.verb.stage_partials

After each thread's own scan/accumulation, valid rows of the last split issue MTail stores from MaxState; other splits and padded rows issue none. No split reads MTail. Nonempty valid rows stage NumState times one; empty splits evaluate zeros. Each valid row writes D channels into the PartA view of OperandArena. Its 81920-byte operand capacity holds the 67584-byte staging span including row spacing; padded rows are neither staged nor later read. For Ng>0, the loop's CTA joins completed all Query writes, QK/PV waits retired operand reads, and the last group enqueued no future transfer, so the arena is free before staging. For Ng=0 the loop and its joins are absent: zero staging writes can race with another warp's unfinished Query writes in the overlapping shared region. The source has no pre-staging barrier. Its barrier after staging joins preparation/staging writes and final TMEM readouts, making the resulting bytes readable without restoring zero. done includes this actual barrier.

## t_9d6002f8e15b6305 | pipeline.verb | 245 chars | 5 declarations

tcg_d128_flat_tma_xpose_narrow:588 design.tile_pipeline.phase.split.role.Main.info.verb.prefetch; tcg_d128_flat_tma_xpose_wide:607 design.tile_pipeline.phase.split.role.Main.info.verb.prefetch; tcg_d128_segmented_cp_xpose_narrow:660 design.tile_pipeline.phase.split.role.Main.info.verb.prefetch; tcg_d128_segmented_tma_xpose_narrow:676 design.tile_pipeline.phase.split.role.Main.info.verb.prefetch; tcg_d128_segmented_tma_xpose_wide:660 design.tile_pipeline.phase.split.role.Main.info.verb.prefetch

Load NextMax for g+1 when g<cHi, otherwise use -inf, including for padded rows. Enqueue next K/V and atoms in the alternate slot only when g<cHi. Prior QK/PV completion retired that slot's old operand reads; done does not wait for new transfers.

## t_9e5883468d98c316 | pipeline.verb | 499 chars | 1 declarations

fused_d128_group:445 design.tile_pipeline.phase.main.role.Even.info.verb.denominator.meaning

Except at g=cLo, wait for the other role's previous-group denominator publication and read DenTransfer(g-1); use zero at the first group. Apply normalize and publish DenState(g) to this role's slot j%2 with 128 arrivals. For a consumed producer publication of local index k, select barrier k%2 and phase floor(k/2)%2. Two slots preserve the older value while the producer can advance its early maximum. Reuse at j+2 requires the intervening maximum chain, which follows the old denominator consumer.

## t_9e5eddba0e57d9be | pipeline.verb | 517 chars | 2 declarations

tcg_d256_flat_cp_narrow:570 design.tile_pipeline.phase.split.role.Main.info.verb.prime; tcg_d256_flat_cp_wide:584 design.tile_pipeline.phase.split.role.Main.info.verb.prime

For C>0, all eight warps enqueue K(l)/V(l), 65536 bytes total, as 16-byte cp.async copies. Threads 0..63 also enqueue both 1024-byte atoms for l..min(l+2,e) into three slots. Each thread commits these requests together; wait_group 0 completes every prime payload and atom. Fence the shared async proxy and CTA-sync to publish them. Warps 0..3 copy l's atoms to TMEM parity slots, wait for stores, fence before thread-sync and CTA-sync. At C=0 this action issues no copies, commits or waits, while retaining its event.

## t_9fd93fb3cd758c6f | pipeline.verb | 248 chars | 1 declarations

tcg_d128_fscan_tma_xpose_wide:445 design.tile_pipeline.phase.group_max.role.Main.info.verb.score

Enqueue this group's four 32-channel QK contractions and commit, then wait for QK completion, toggle parity and apply the completion fence. DotTile is now readable and K/KeyScale reads are retired. The first contraction overwrites the score region.

## t_a0741081f6279b7c | pipeline.verb | 317 chars | 2 declarations

tcg_d128_fscan_cp_direct_narrow:419 design.tile_pipeline.phase.group_max.role.Main.info.verb.prefetch; tcg_d128_fscan_cp_direct_wide:436 design.tile_pipeline.phase.group_max.role.Main.info.verb.prefetch

Enqueue K(g+2) and atom for Long, otherwise K(g+1) and atom, only when that group is <=cHi. Every thread then commits a copy group, including an empty one when there is no transfer. done means commit only. Prior QK completion retires the refilled payload slot's readers; prior CTA joins retire its scale-atom readers.

## t_a089e8dc7e918b5d | pipeline.verb | 314 chars | 2 declarations

tcg_d128_flat_cp_direct_narrow:418 design.tile_pipeline.phase.group_max.role.Main.info.verb.prime; tcg_d128_flat_cp_direct_wide:432 design.tile_pipeline.phase.group_max.role.Main.info.verb.prime

For Ng>0 cooperatively enqueue K(cLo) and its 512-byte scale atom into slot zero with 16-byte asynchronous copies, then commit one group per thread. Payload copies zero-fill keys past Skv; scale copies read allocated padding bytes. For Ng=0 issue neither copies nor commits. done means enqueue/commit, not arrival.

## t_a092e58bfea0ead5 | pipeline.verb | 523 chars | 1 declarations

fused_d128_group:424 design.tile_pipeline.phase.main.role.Even.info.verb.maximum.meaning

Except at g=cLo, wait for the other role's previous-group maximum publication and read MaxTransfer(g-1). Apply normalize's maximum and alpha, retaining alpha until rescale. Store MaxState(g) to MaxTransfer(g); all 128 row writers arrive on this role's maximum barrier. done is the complete row publication; producers do not wait on their own barrier before starting their row exponentials. Overwrite at g+2 follows the wait for g+1, hence the other role's read of g; one maximum slot per role suffices without an extra ack.

## t_a140ac345bd87e77 | pipeline.verb | 62 chars | 4 declarations

tcg_d256_flat_cp_narrow:473 design.tile_pipeline.phase.scan.role.Main.info.verb.tail; tcg_d256_flat_cp_wide:487 design.tile_pipeline.phase.scan.role.Main.info.verb.tail; tcg_d256_flat_tma_narrow:471 design.tile_pipeline.phase.scan.role.Main.info.verb.tail; tcg_d256_flat_tma_wide:485 design.tile_pipeline.phase.scan.role.Main.info.verb.tail

Load GroupMax[t,row], update ScanState and write Mscan[t,row].

## t_a1ba10541fd1a1a9 | pipeline.verb | 311 chars | 4 declarations

tc_split_flat_narrow:596 design.tile_pipeline.phase.split.role.Main.info.verb.advance; tc_split_flat_wide:613 design.tile_pipeline.phase.split.role.Main.info.verb.advance; tc_split_segmented_narrow:703 design.tile_pipeline.phase.split.role.Main.info.verb.advance; tc_split_segmented_wide:737 design.tile_pipeline.phase.split.role.Main.info.verb.advance

Read CurrentMax and PreviousMax from normalize, retaining CurrentMax for weighting. Apply alpha to NumState and DenPart if any lane in the owning warp needs a nonunit rescale. Initialize E. Score survives all probability chunks. There is no inserted PDL wait here; this ordinary launch follows scan.kernel_done.

## t_a1c3630a0aa09233 | pipeline.verb | 225 chars | 1 declarations

root:622 design.tile_pipeline.phase.main.role.Main.info.verb.drain

Wait for the one final outstanding PV and apply the completion fence before NumState readout. Every requested K/V payload and atom has already been consumed and waited; no next-group transfer was issued on the last iteration.

## t_a29044ff9b0c2277 | pipeline.verb | 588 chars | 2 declarations

tcg_d128_fscan_cp_xpose_narrow:408 design.tile_pipeline.phase.group_max.role.Main.info.verb.setup; tcg_d128_fscan_cp_xpose_wide:425 design.tile_pipeline.phase.group_max.role.Main.info.verb.setup

Establish the scheduled coordinates, row masks and completion parities, initially zero. RingUsed=3 for Long, otherwise 2; ScaleCol(g)=136+8*(g%2) for Long, otherwise 136. The allocated three shared slots are unchanged. The q0>=Sq guard is unreachable under this grid. Valid rows of split zero/last issue the outside -inf stores. Thread zero initializes QK/PV completion barriers and issues MMA; all warps cooperatively copy payloads, commit and wait on their own copy-group streams, and stage scales. All four warps read scores, including padded rows; only valid rows write global maxima.

## t_a2f93f9a98929e92 | pipeline.verb | 216 chars | 4 declarations

tcg_d128_fscan_cp_direct_narrow:571 design.tile_pipeline.phase.split.role.Main.info.verb.tail; tcg_d128_fscan_cp_direct_wide:588 design.tile_pipeline.phase.split.role.Main.info.verb.tail; tcg_d128_fscan_tma_direct_narrow:564 design.tile_pipeline.phase.split.role.Main.info.verb.tail; tcg_d128_fscan_tma_direct_wide:582 design.tile_pipeline.phase.split.role.Main.info.verb.tail

The last split's valid rows issue MTail stores from final MaxState, including for an empty split. Other split CTAs perform no tail store. done means stores issued with register sources consumed; no split reads MTail.

## t_a36696238e2fc021 | pipeline.verb | 277 chars | 4 declarations

tc_split_flat_narrow:470 design.tile_pipeline.phase.group_max.role.Main.info.verb.prime; tc_split_flat_wide:487 design.tile_pipeline.phase.group_max.role.Main.info.verb.prime; tc_split_segmented_narrow:545 design.tile_pipeline.phase.group_max.role.Main.info.verb.prime; tc_split_segmented_wide:579 design.tile_pipeline.phase.group_max.role.Main.info.verb.prime

For a nonempty interval enqueue Kraw and KeyAtom at gmLo into slot 0 and commit. done means enqueue/commit, not arrival. On an empty interval this action completes without copies or a commit. Payload padding is source-size zero-filled; allocated scale atoms are copied in full.

## t_a4553b72840688ec | pipeline.verb | 340 chars | 12 declarations

tcg_d128_flat_tma_direct_narrow:587 design.tile_pipeline.phase.split.role.Main.info.verb.stage; tcg_d128_flat_tma_direct_wide:601 design.tile_pipeline.phase.split.role.Main.info.verb.stage; tcg_d128_flat_tma_xpose_narrow:591 design.tile_pipeline.phase.split.role.Main.info.verb.stage; tcg_d128_flat_tma_xpose_wide:610 design.tile_pipeline.phase.split.role.Main.info.verb.stage; tcg_d128_fscan_tma_direct_narrow:549 design.tile_pipeline.phase.split.role.Main.info.verb.stage; tcg_d128_fscan_tma_direct_wide:567 design.tile_pipeline.phase.split.role.Main.info.verb.stage; tcg_d128_fscan_tma_xpose_narrow:556 design.tile_pipeline.phase.split.role.Main.info.verb.stage; tcg_d128_segmented_cp_xpose_narrow:663 design.tile_pipeline.phase.split.role.Main.info.verb.stage; tcg_d128_segmented_tma_direct_narrow:677 design.tile_pipeline.phase.split.role.Main.info.verb.stage; tcg_d128_segmented_tma_direct_wide:696 design.tile_pipeline.phase.split.role.Main.info.verb.stage; tcg_d128_segmented_tma_xpose_narrow:679 design.tile_pipeline.phase.split.role.Main.info.verb.stage; tcg_d128_segmented_tma_xpose_wide:663 design.tile_pipeline.phase.split.role.Main.info.verb.stage

Wait for current K/V plus atoms; parity flips after a two-slot wrap. Fence shared async access and CTA-sync, write current KeyScale and ValueScale, wait for TMEM stores, fence and CTA-sync. Prior PV was already waited, protecting the scale/operand slots. These joins retire the previous GroupPV readers before QK overwrites the same region.

## t_a47d72da4483cd2d | pipeline.verb | 348 chars | 2 declarations

tcg_d128_segmented_cp_direct_narrow:680 design.tile_pipeline.phase.split.role.Main.info.verb.prefetch; tcg_d128_segmented_cp_direct_wide:697 design.tile_pipeline.phase.split.role.Main.info.verb.prefetch

Load NextMax for g+1 when g<cHi, otherwise use -inf, including in dead warps. Enqueue next K/V and atoms in the alternate slot only when g<cHi, then always commit a copy group, including an empty one at the tail. Prior QK/PV completion retired that slot's old operand reads and CTA joins retired its atom readers; done does not wait for new copies.

## t_a47f19b5640e1803 | pipeline.verb | 383 chars | 2 declarations

fused_d128_cp:372 design.tile_pipeline.phase.main.role.Main.info.verb.read_score; root:587 design.tile_pipeline.phase.main.role.Main.info.verb.read_score

Wait for this half's QK completion phase, apply the after-thread- sync fence, then read its 64 columns into the corresponding half of Score. Each TMEM read waits for its own load completion. Apply row masks, reduce the raw maximum locally and retain the scores for advance. Both half maxima feed one full-group maximum. No cross-lane or cross-warpgroup numerical reduction is needed.

## t_a537e039e39c32f9 | pipeline.verb | 395 chars | 1 declarations

tcg_d256_segmented_cp_wide:780 design.tile_pipeline.phase.split.role.Main.info.verb.qk

Thread zero fences after thread synchronization, issues eight score contractions and commits QK. Every thread waits and advances QK parity, then fences. Each warp reads and masks its 64-key slice in two self-retiring TMEM loads, retaining Score, except wholly padded warps when PSK is active. Wholly padded PSK blocks skip readout but not the QK wait. No maximum or half-maximum exchange occurs.

## t_a79c631e5e1d75a1 | pipeline.verb | 360 chars | 7 declarations

tcg_d256_flat_cp_narrow:407 design.tile_pipeline.phase.group_max.role.Main.info.verb.prepare; tcg_d256_flat_cp_wide:421 design.tile_pipeline.phase.group_max.role.Main.info.verb.prepare; tcg_d256_flat_tma_narrow:405 design.tile_pipeline.phase.group_max.role.Main.info.verb.prepare; tcg_d256_flat_tma_wide:419 design.tile_pipeline.phase.group_max.role.Main.info.verb.prepare; tcg_d256_segmented_cp_narrow:443 design.tile_pipeline.phase.group_max.role.Main.info.verb.prepare; tcg_d256_segmented_tma_narrow:458 design.tile_pipeline.phase.group_max.role.Main.info.verb.prepare; tcg_d256_segmented_tma_wide:443 design.tile_pipeline.phase.group_max.role.Main.info.verb.prepare

Prefetch Q and its scale words into QueryPrefetch before the first warp allocates 256 TMEM columns. CTA-sync publishes the base, then that warp relinquishes its permit. All warps write Query; warps 0..3 write QueryScale and wait for TMEM stores. QueryPrefetch then dies. With C>0, commit an additional empty cp.async group to preserve hoisted-prime accounting.

## t_a7a95d90581a7d2b | pipeline.verb | 497 chars | 1 declarations

tcg_d256_segmented_cp_wide:517 design.tile_pipeline.phase.group_max.role.Main.info.verb.land

Enqueue K(g+1) and its atom into the other slot when g+1<=e. Commit one cp.async group even at the tail and wait_group 1, proving each thread's current K and atom copies; a shared async-proxy fence and CTA-sync publish Query and current K/atoms. Warps 0..3 copy current atoms to the single KeyScale, wait for TMEM stores, then a before-thread-sync fence and CTA barrier publish them. Previous QK and the preceding iteration's final barrier retire old K/atom/scale readers before conflicting reuse.

## t_a8834dc017142f60 | pipeline.verb | 372 chars | 7 declarations

tcg_d256_flat_cp_narrow:612 design.tile_pipeline.phase.split.role.Main.info.verb.pv; tcg_d256_flat_cp_wide:626 design.tile_pipeline.phase.split.role.Main.info.verb.pv; tcg_d256_flat_tma_narrow:607 design.tile_pipeline.phase.split.role.Main.info.verb.pv; tcg_d256_flat_tma_wide:621 design.tile_pipeline.phase.split.role.Main.info.verb.pv; tcg_d256_segmented_cp_narrow:680 design.tile_pipeline.phase.split.role.Main.info.verb.pv; tcg_d256_segmented_tma_narrow:695 design.tile_pipeline.phase.split.role.Main.info.verb.pv; tcg_d256_segmented_tma_wide:680 design.tile_pipeline.phase.split.role.Main.info.verb.pv

Thread zero fences after thread-sync and issues four 32-key steps, each with both 128-channel products. First step of the first group overwrites each numerator half; later steps accumulate. Commit PV, then all threads wait, advance parity and fence. done retires Value/ValueScale/Prob8 reads and NumState updates. No deferred PV, next-QK issue or tail CTA barrier follows.

## t_a89b44cedf6364bf | pipeline.verb | 250 chars | 1 declarations

tcg_d128_fscan_tma_xpose_narrow:650 design.tile_pipeline.phase.combine.role.Main.info.verb.sum_b

Each thread independently reads DenParts and Weight to perform combine's ordered weighted denominator sum in batches of eight. Missing entries add zero without shared reads; no parallel reduction tree is used. TotalB is replicated per output channel.

## t_a9248f7ef005a3bd | pipeline.verb | 495 chars | 1 declarations

tcg_d128_flat_tma_xpose_narrow:571 design.tile_pipeline.phase.split.role.Main.info.verb.prepare.meaning

Prefetch Q/scales, first-warp allocate 256 TMEM columns, CTA-publish the base and relinquish the permit. Each thread writes Query/QueryScale/UnitScale, waits for its TMEM stores, discards QueryPrefetch and clears its NumState before entering its scan wait. done guarantees that preparation, including all Query writes, has completed across all warps. Independent scan waits may begin before this collective completion; the PDL wait does not join the warps. Empty splits also perform preparation.

## t_a955e2f4c0fe36a6 | pipeline.verb | 283 chars | 2 declarations

tcg_d128_fscan_cp_direct_narrow:408 design.tile_pipeline.phase.group_max.role.Main.info.verb.prime; tcg_d128_fscan_cp_direct_wide:425 design.tile_pipeline.phase.group_max.role.Main.info.verb.prime

For Ng>0 cooperatively enqueue K(cLo) and its 512-byte scale atom into slot zero, then commit one copy group per thread. Payload copies zero-fill out-of-range keys while allocated scale padding is read. Ng=0 performs no transfer or commit. done means enqueue and commit, not arrival.

## t_a95d0d2e9feb9f88 | pipeline.verb | 527 chars | 2 declarations

fused_d256_cp:431 design.tile_pipeline.phase.main.role.Main.info.verb.finish; fused_d256_tma:449 design.tile_pipeline.phase.main.role.Main.info.verb.finish

A before-thread-sync fence and CTA barrier join final TMEM readers before the first warp deallocates all 512 columns. The source emits its programmatic-launch-completion trigger at exit; this ordinary single-kernel path has no dependent launch or PDL data wait. No next tile is assigned. Other CTAs retain independent resources while ordinary output writes complete; kernel completion gives caller-stream visibility. Subsequent invocations initialize fresh numerical state; descriptor/graph caches do not carry numerical state.

## t_a965814d2a6ff2c7 | pipeline.verb | 142 chars | 15 declarations

tcg_d128_flat_cp_direct_narrow:657 design.tile_pipeline.phase.combine.role.Main.info.verb.dependency; tcg_d128_flat_cp_direct_wide:673 design.tile_pipeline.phase.combine.role.Main.info.verb.dependency; tcg_d128_flat_cp_xpose_narrow:685 design.tile_pipeline.phase.combine.role.Main.info.verb.dependency; tcg_d128_flat_cp_xpose_wide:668 design.tile_pipeline.phase.combine.role.Main.info.verb.dependency; tcg_d128_flat_tma_direct_narrow:650 design.tile_pipeline.phase.combine.role.Main.info.verb.dependency; tcg_d128_flat_tma_direct_wide:666 design.tile_pipeline.phase.combine.role.Main.info.verb.dependency; tcg_d128_flat_tma_xpose_narrow:673 design.tile_pipeline.phase.combine.role.Main.info.verb.dependency; tcg_d128_flat_tma_xpose_wide:693 design.tile_pipeline.phase.combine.role.Main.info.verb.dependency; tcg_d128_segmented_cp_direct_narrow:748 design.tile_pipeline.phase.combine.role.Main.info.verb.dependency; tcg_d128_segmented_cp_direct_wide:767 design.tile_pipeline.phase.combine.role.Main.info.verb.dependency; tcg_d128_segmented_cp_xpose_narrow:744 design.tile_pipeline.phase.combine.role.Main.info.verb.dependency; tcg_d128_segmented_tma_direct_narrow:740 design.tile_pipeline.phase.combine.role.Main.info.verb.dependency; tcg_d128_segmented_tma_direct_wide:758 design.tile_pipeline.phase.combine.role.Main.info.verb.dependency; tcg_d128_segmented_tma_xpose_narrow:760 design.tile_pipeline.phase.combine.role.Main.info.verb.dependency; tcg_d128_segmented_tma_xpose_wide:744 design.tile_pipeline.phase.combine.role.Main.info.verb.dependency

All threads execute the PDL wait at entry. done guarantees every PartA/PartB producer has completed; scan and MTail are complete transitively.

## t_aa10c6a99ad71cee | pipeline.verb | 320 chars | 12 declarations

tcg_d128_flat_cp_direct_narrow:440 design.tile_pipeline.phase.group_max.role.Main.info.verb.reduce; tcg_d128_flat_cp_direct_wide:454 design.tile_pipeline.phase.group_max.role.Main.info.verb.reduce; tcg_d128_flat_tma_direct_narrow:437 design.tile_pipeline.phase.group_max.role.Main.info.verb.reduce; tcg_d128_flat_tma_direct_wide:451 design.tile_pipeline.phase.group_max.role.Main.info.verb.reduce; tcg_d128_fscan_cp_direct_narrow:431 design.tile_pipeline.phase.group_max.role.Main.info.verb.reduce; tcg_d128_fscan_cp_direct_wide:448 design.tile_pipeline.phase.group_max.role.Main.info.verb.reduce; tcg_d128_fscan_tma_direct_narrow:426 design.tile_pipeline.phase.group_max.role.Main.info.verb.reduce; tcg_d128_fscan_tma_direct_wide:444 design.tile_pipeline.phase.group_max.role.Main.info.verb.reduce; tcg_d128_segmented_cp_direct_narrow:496 design.tile_pipeline.phase.group_max.role.Main.info.verb.reduce; tcg_d128_segmented_cp_direct_wide:513 design.tile_pipeline.phase.group_max.role.Main.info.verb.reduce; tcg_d128_segmented_tma_direct_narrow:492 design.tile_pipeline.phase.group_max.role.Main.info.verb.reduce; tcg_d128_segmented_tma_direct_wide:511 design.tile_pipeline.phase.group_max.role.Main.info.verb.reduce

Live warps reduce scores in 32-key chunks, using masked readout or native unmasked max readout, each with its TMEM-load wait. Only valid rows issue GroupMax stores. A CTA barrier joins all score readers before the next QK overwrites DotTile; no full-group score register array, PV or cross-warpgroup reduction is needed.

## t_aa6b8862f80a5985 | pipeline.verb | 298 chars | 1 declarations

fused_d128_key:335 design.tile_pipeline.phase.main.role.Main.info.verb.cold

Issue cp.async for cLo's Key, Value and both atoms into slot zero, then commit one per-thread copy group. All threads partition payload copies; the first warp copies the atoms. Payloads past Skv zero-fill; allocated scale padding bytes are loaded as stored. done means enqueue, not copy completion.

## t_aad7b3c567f8f424 | pipeline.verb | 695 chars | 3 declarations

tcg_d128_segmented_cp_xpose_narrow:550 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.scan; tcg_d128_segmented_tma_xpose_narrow:566 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.scan; tcg_d128_segmented_tma_xpose_wide:550 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.scan

Each valid thread initializes ScanState and unconditionally walks segment zero. A walk loads 16 consecutive groups into WalkBatch before their ordered recurrence and stores, then handles a scalar remainder. For each later batch, load all SegmentBounds, using -inf for missing segments, and consume every bound to build the jump mask against one incoming state before calling the walk helper. A zero mask only fills constant snapshots; otherwise fill the unchanged prefix and walk the entire suffix from the lowest set bit. Consume each register batch before reuse; ScanState persists across walks and fills. Out-of-range rows have no data effects. There is no cross-thread reduction or CTA join.

## t_ab41597db328356b | pipeline.verb | 551 chars | 5 declarations

tcg_d128_flat_tma_xpose_narrow:583 design.tile_pipeline.phase.split.role.Main.info.verb.replay.meaning; tcg_d128_flat_tma_xpose_wide:602 design.tile_pipeline.phase.split.role.Main.info.verb.replay.meaning; tcg_d128_segmented_cp_xpose_narrow:655 design.tile_pipeline.phase.split.role.Main.info.verb.replay.meaning; tcg_d128_segmented_tma_xpose_narrow:671 design.tile_pipeline.phase.split.role.Main.info.verb.replay.meaning; tcg_d128_segmented_tma_xpose_wide:655 design.tile_pipeline.phase.split.role.Main.info.verb.replay.meaning

After its scan wait, each thread initializes DenState=0 and MaxState=-inf; load the entrance snapshot at (cLo-1)*R+rowG whenever cLo>0, including empty splits, and CurrentMax at cLo only for Ng>0. rowG is domain's r. done covers all replay work; each thread can continue with its own registers before other warps finish. The next stage or stage_partials CTA join covers completion. The entrance load has no any_compute guard or T-bound clamp; a logically out-of-range empty-split read has no initialization or safety guarantee. Its value is discarded.

## t_abd82c63d2af69e2 | pipeline.verb | 273 chars | 6 declarations

tcg_d128_flat_tma_direct_narrow:569 design.tile_pipeline.phase.split.role.Main.info.verb.setup; tcg_d128_flat_tma_direct_wide:583 design.tile_pipeline.phase.split.role.Main.info.verb.setup; tcg_d128_fscan_tma_direct_narrow:532 design.tile_pipeline.phase.split.role.Main.info.verb.setup; tcg_d128_fscan_tma_direct_wide:550 design.tile_pipeline.phase.split.role.Main.info.verb.setup; tcg_d128_segmented_tma_direct_narrow:659 design.tile_pipeline.phase.split.role.Main.info.verb.setup; tcg_d128_segmented_tma_direct_wide:678 design.tile_pipeline.phase.split.role.Main.info.verb.setup

Establish coordinates and PSK valid/live-warp predicates. Initialize QK/PV and the two combined K/V transaction barriers at parity zero and publish initialization. All four warps cooperate; one thread maps to each row, and thread zero issues TMA/MMA. q0>=Sq is unreachable.

## t_ac9094d1528c43a1 | pipeline.verb | 323 chars | 4 declarations

tcg_d128_fscan_cp_direct_narrow:573 design.tile_pipeline.phase.split.role.Main.info.verb.write; tcg_d128_fscan_cp_direct_wide:590 design.tile_pipeline.phase.split.role.Main.info.verb.write; tcg_d128_fscan_tma_direct_narrow:566 design.tile_pipeline.phase.split.role.Main.info.verb.write; tcg_d128_fscan_tma_direct_wide:584 design.tile_pipeline.phase.split.role.Main.info.verb.write

Valid rows directly issue 16-byte FP32 PartA stores from NumState times one and scalar PartB stores from DenState times one; empty splits instead store zeros. Every split also stores final MaxState to PartM. No shared transpose staging or O/LSE store occurs. done means ordinary stores issued and register sources consumed.

## t_ace350add1f71f3c | pipeline.verb | 309 chars | 2 declarations

tcg_d128_flat_cp_direct_narrow:590 design.tile_pipeline.phase.split.role.Main.info.verb.prefetch; tcg_d128_flat_cp_direct_wide:604 design.tile_pipeline.phase.split.role.Main.info.verb.prefetch

Load NextMax for g+1 when g<cHi, otherwise use -inf, including in dead warps. Enqueue next K/V and atoms in the alternate slot only when g<cHi, then every thread commits a group, empty at g=cHi. Prior QK/PV completion and CTA joins retired that slot's operand/atom reads; done does not wait for new transfers.

## t_ad05ad60c0edfdeb | pipeline.verb | 314 chars | 7 declarations

tcg_d256_flat_cp_narrow:637 design.tile_pipeline.phase.split.role.Main.info.verb.finish; tcg_d256_flat_cp_wide:651 design.tile_pipeline.phase.split.role.Main.info.verb.finish; tcg_d256_flat_tma_narrow:632 design.tile_pipeline.phase.split.role.Main.info.verb.finish; tcg_d256_flat_tma_wide:646 design.tile_pipeline.phase.split.role.Main.info.verb.finish; tcg_d256_segmented_cp_narrow:705 design.tile_pipeline.phase.split.role.Main.info.verb.finish; tcg_d256_segmented_tma_narrow:720 design.tile_pipeline.phase.split.role.Main.info.verb.finish; tcg_d256_segmented_tma_wide:705 design.tile_pipeline.phase.split.role.Main.info.verb.finish

Fence before thread-sync and CTA-sync joins final readers; the first warp deallocates 512 TMEM columns. Trigger dependent launch at exit. combine waits for kernel completion and visible partials. No next tile is assigned; later invocations initialize fresh numerical state without clearing unrelated scratch bytes.

## t_ad6a28a1e2975451 | pipeline.verb | 283 chars | 3 declarations

tcg_d128_fscan_cp_xpose_narrow:438 design.tile_pipeline.phase.group_max.role.Main.info.verb.advance; tcg_d128_fscan_cp_xpose_wide:455 design.tile_pipeline.phase.group_max.role.Main.info.verb.advance; tcg_d128_segmented_cp_xpose_wide:525 design.tile_pipeline.phase.group_max.role.Main.info.verb.advance

For Long and g<cHi, enqueue QK(g+1) using the staged next K and alternating scale slot and commit; done means enqueue only. Otherwise perform no product or wait. This lets QK(g+1) run underneath the following prefetch/stage work without changing the single-score-slot reader barrier.

## t_adfab06a1690c3f9 | pipeline.verb | 556 chars | 1 declarations

fused_d256_tma:400 design.tile_pipeline.phase.main.role.Main.info.verb.exchange

Each half writes HalfMax into Exchange. A CTA barrier joins all score readers and publishes ahead's scale stores and land's fences. Each thread reads both halves and forms the group maximum. Thread 0 then applies its after-thread-sync fence and, when g+1<=cHi, enqueues and commits QK(g+1) into the reusable ScoreTile. done promises the next QK's enqueue, not its completion; it may run through advance and PV issue. The final group still exchanges maxima but issues no next QK. advance's CTA barrier retires all Exchange readers before the next overwrite.

## t_ae68def446bd9d39 | pipeline.verb | 540 chars | 1 declarations

fused_d128_key:402 design.tile_pipeline.phase.main.role.Main.info.verb.advance

Apply normalize to retained Score and RowState. For g>cLo, each warp with any alpha unequal to one rescales its assigned NumState channels using two 32-channel load/multiply/store chunks per row, then waits for stores. The first group never reads NumState. Each warpgroup writes its 64 probability keys into the common Prob8 tile and advances its own DenSlice. wait::st, the before fence and CTA barrier publish Prob8 and numerator rescaling to the issuer. Score and group temporaries are now dead; the two MaxState copies remain identical.

## t_ae847771c9fc7ec0 | pipeline.verb | 250 chars | 1 declarations

tc_fused:305 design.tile_pipeline.phase.main.role.Main.info.verb.prime

Enqueue Kraw, Vraw, KeyAtom and ValueAtom for cLo into slot 0 and commit a copy group. done means enqueue and commit, not copy completion. Payloads zero-fill keys outside Skv through the source-size operand; atoms include all allocated padding bytes.

## t_af1f45cb88cacc73 | pipeline.verb | 302 chars | 1 declarations

fused_d256_tma:386 design.tile_pipeline.phase.main.role.Main.info.verb.retire

For g>cLo wait for PV(g-1), advance its parity once and apply the completion fence. This retires its Value, ValueScale and Prob8 reads and NumState updates before conflicting refill, probability stores or rescale. At g=cLo the action has no previous product to wait for and reads no previous numerator.

## t_afc9edcd2ce30d69 | pipeline.verb | 359 chars | 2 declarations

tc_split_segmented_narrow:620 design.tile_pipeline.phase.scan.role.Main.info.verb.walk; tc_split_segmented_wide:654 design.tile_pipeline.phase.scan.role.Main.info.verb.walk

Execute the scan step's unconditional first-segment walk. Load sixteen independent x values into GroupBatch before applying their ordered recurrence and stores; any partial tail uses scalar load-update-store steps. A batch's reads end before its registers are reused. done includes updated ScanCarry and issued ScanState stores; SegmentMax[0,r] is never read.

## t_b18e039995903cf8 | pipeline.verb | 269 chars | 1 declarations

root:542 design.tile_pipeline.phase.main.role.Main.info.verb.prime

Enqueue group cLo into slot zero. TMA issues K and V separately, each with its 512-byte scale atom on the same transaction barrier as its 16384-byte payload. done means requests have been issued, not that data has arrived; the land actions own the readiness guarantees.

## t_b1ad40e2550b479d | pipeline.verb | 559 chars | 1 declarations

tcg_d128_fscan_cp_xpose_wide:687 design.tile_pipeline.phase.combine.role.Main.info.verb.finalize

After the subtotal barrier, only merge group zero sums DenGroup and NumGroup in combine's stated order, forms TotalB, TotalA and inv, and writes the owned BF16 channels. Its first thread additionally reads MTail and writes LSE when present. Other groups have completed their work and perform no final reads or stores; this collective action still has issue/done events. Caller destinations are never read. Stores consume register sources before CTA exit; kernel completion supplies global visibility. Each CTA has one worktile and no additional launch signal.

## t_b29ea8bd06238b0f | pipeline.verb | 424 chars | 1 declarations

fused_d128_key:431 design.tile_pipeline.phase.main.role.Main.info.verb.write

Read the assigned 64 NumState channels in self-retiring 32-channel chunks. Valid rows issue value_output's BF16 RNE stores directly into caller O; only warpgroup zero writes present LSE. Padding executes readout but no output store. If LSE is absent the action has only O writes. Initial destination bytes are never read. done means stores issued and register sources consumed, without a host or device-wide completion wait.

## t_b3ac35e31954a527 | pipeline.verb | 557 chars | 2 declarations

tcg_d128_fscan_cp_xpose_narrow:565 design.tile_pipeline.phase.split.role.Main.info.verb.stage; tcg_d128_fscan_cp_xpose_wide:582 design.tile_pipeline.phase.split.role.Main.info.verb.stage

Each thread waits until at most one committed copy group remains, proving all current K/V payload and atom copies complete. The final empty commit ensures the last real copies are covered. Fence shared async access and CTA-sync; cooperatively write current K/V scales into their fixed TMEM slots, wait for TMEM stores, then fence and CTA-sync. The preceding group already waited for PV, so ValueScale and the refilled V slot have no remaining matrix readers. This CTA join also retires previous GroupPV readout before QK reuses its overlapping score region.

## t_b3c761611a700a78 | pipeline.verb | 261 chars | 4 declarations

tcg_d128_fscan_cp_xpose_narrow:575 design.tile_pipeline.phase.split.role.Main.info.verb.pv; tcg_d128_fscan_cp_xpose_wide:592 design.tile_pipeline.phase.split.role.Main.info.verb.pv; tcg_d128_fscan_tma_xpose_wide:580 design.tile_pipeline.phase.split.role.Main.info.verb.pv; tcg_d128_segmented_cp_xpose_wide:726 design.tile_pipeline.phase.split.role.Main.info.verb.pv

Enqueue four ordered native PV contractions into GroupPV, using overwrite for the first contraction of every group. Commit, wait for PV completion, toggle parity and apply the completion fence. All current Value, ValueScale and Prob8 operand reads have retired.

## t_b44d8ec845cfe69b | pipeline.verb | 286 chars | 4 declarations

tc_split_flat_narrow:567 design.tile_pipeline.phase.split.role.Main.info.verb.setup; tc_split_flat_wide:584 design.tile_pipeline.phase.split.role.Main.info.verb.setup; tc_split_segmented_narrow:674 design.tile_pipeline.phase.split.role.Main.info.verb.setup; tc_split_segmented_wide:708 design.tile_pipeline.phase.split.role.Main.info.verb.setup

Establish rows, cLo/cHi and rho. The grid cannot take q0>=Sq. Each warp owns 16 rows; four-lane subgroups share two rows and partition key/output pairs. Replayed maxima have four replicas per row and DenPart has four distinct partials. All four warps stage inputs and join CTA barriers.

## t_b480e78570718bfd | pipeline.verb | 307 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:700 design.tile_pipeline.phase.split.role.Main.info.verb.dependency.meaning

Each thread executes the PDL scan-dependency wait after its own preparation and before its first scan-state read. done guarantees all MState and MTail results, including entrance snapshots, are visible. This wait is for the predecessor grid; it does not join this CTA's warps or complete their Query writes.

## t_b489adee995291f5 | pipeline.verb | 276 chars | 2 declarations

tc_split_segmented_narrow:588 design.tile_pipeline.phase.segmax.role.Main.info.verb.load; tc_split_segmented_wide:622 design.tile_pipeline.phase.segmax.role.Main.info.verb.load

Each valid thread owns one row in segment s. Load Seg independent maximum values into SegmentBatch, substituting -inf beyond T. Cblock reads BmCovered and Ublock reads PriorMax; adjacent lanes read adjacent rows of the transposed table. done makes the complete batch readable.

## t_b4d0bf480ae122e0 | pipeline.verb | 578 chars | 2 declarations

tcg_d128_fscan_cp_direct_narrow:403 design.tile_pipeline.phase.group_max.role.Main.info.verb.setup; tcg_d128_fscan_cp_direct_wide:420 design.tile_pipeline.phase.group_max.role.Main.info.verb.setup

Establish coordinates, masks and initially zero matrix-completion parities. RingUsed=3 for Long, otherwise 2; ScaleCol(g)=136+8*(g%2) for Long, otherwise 136. Three shared slots remain allocated. Valid rows of the first/last split issue outside -inf fills. Thread zero initializes matrix barriers and issues MMA; prepare's CTA join publishes initialization. All four warps stage payloads, the first also copies scale atoms, and every thread commits/waits its own copy groups. Dead warps keep these duties and synchronization but skip scalar score readout. q0>=Sq is unreachable.

## t_b51fd2fb2a5c6d07 | pipeline.verb | 332 chars | 2 declarations

tcg_d128_segmented_cp_direct_narrow:501 design.tile_pipeline.phase.group_max.role.Main.info.verb.finish; tcg_d128_segmented_cp_direct_wide:518 design.tile_pipeline.phase.group_max.role.Main.info.verb.finish

All issued K/atom copies and QK products have been waited. Fence and CTA-sync join final TMEM users, then the first warp deallocates 256 columns. Each CTA signals PDL launch completion after its stores and teardown; launch_ready is the grid-wide protocol event. Global visibility comes from kernel completion and the successor wait.

## t_b64ca0e52614f30d | pipeline.verb | 188 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:561 design.tile_pipeline.phase.segment_max.role.Main.info.verb.signal

Every thread signals PDL launch permission for segmented_scan at entry, before waiting for group_max. This permits successor launch but does not publish SegMax or complete the predecessor.

## t_b8107b708505b7ad | pipeline.verb | 187 chars | 2 declarations

tc_split_segmented_narrow:618 design.tile_pipeline.phase.scan.role.Main.info.verb.initialize; tc_split_segmented_wide:652 design.tile_pipeline.phase.scan.role.Main.info.verb.initialize

One thread owns each row and initializes ScanCarry to -inf when valid. Padding threads execute entry and have no later numerical effects; all listed role actions retain issue/done events.

## t_b89838c3961db4f3 | pipeline.verb | 275 chars | 4 declarations

tc_split_flat_narrow:570 design.tile_pipeline.phase.split.role.Main.info.verb.prime; tc_split_flat_wide:587 design.tile_pipeline.phase.split.role.Main.info.verb.prime; tc_split_segmented_narrow:677 design.tile_pipeline.phase.split.role.Main.info.verb.prime; tc_split_segmented_wide:711 design.tile_pipeline.phase.split.role.Main.info.verb.prime

For a nonempty interval enqueue Kraw, Vraw, KeyAtom and ValueAtom at cLo into slot 0 and commit. done is enqueue/commit, not arrival. Empty intervals complete this action without copies or a commit. Payload copies zero-fill j>=Skv; full scale atoms include allocated padding.

## t_b8c376fd7aa5c4ab | pipeline.verb | 323 chars | 8 declarations

tcg_d128_flat_cp_direct_narrow:608 design.tile_pipeline.phase.split.role.Main.info.verb.write; tcg_d128_flat_cp_direct_wide:622 design.tile_pipeline.phase.split.role.Main.info.verb.write; tcg_d128_flat_tma_direct_narrow:601 design.tile_pipeline.phase.split.role.Main.info.verb.write; tcg_d128_flat_tma_direct_wide:615 design.tile_pipeline.phase.split.role.Main.info.verb.write; tcg_d128_segmented_cp_direct_narrow:699 design.tile_pipeline.phase.split.role.Main.info.verb.write; tcg_d128_segmented_cp_direct_wide:716 design.tile_pipeline.phase.split.role.Main.info.verb.write; tcg_d128_segmented_tma_direct_narrow:691 design.tile_pipeline.phase.split.role.Main.info.verb.write; tcg_d128_segmented_tma_direct_wide:710 design.tile_pipeline.phase.split.role.Main.info.verb.write

Nonempty valid rows load MTail and compute weight. Valid rows directly issue 16-byte FP32 stores from weighted NumState to PartA and a scalar weighted DenState store to PartB. Empty splits still store zeros. No shared transpose staging or O/LSE store occurs. done means ordinary stores issued and register sources consumed.

## t_b8fa9981ee9b3702 | pipeline.verb | 438 chars | 2 declarations

tcg_d128_flat_cp_xpose_narrow:603 design.tile_pipeline.phase.split.role.Main.info.verb.stage; tcg_d128_flat_cp_xpose_wide:586 design.tile_pipeline.phase.split.role.Main.info.verb.stage

Each thread executes wait_group<1> to prove current K/V plus atoms while the next group may remain in flight. The final empty commit preserves this proof at the tail. Fence shared async access and CTA-sync, write current KeyScale and ValueScale, wait for TMEM stores, fence and CTA-sync. Prior PV was already waited, protecting the scale/operand slots. These joins retire the previous GroupPV readers before QK overwrites the same region.

## t_b92198b20b6faa15 | pipeline.verb | 271 chars | 8 declarations

tcg_d128_flat_cp_direct_narrow:483 design.tile_pipeline.phase.flat_scan.role.Main.info.verb.scan; tcg_d128_flat_cp_direct_wide:497 design.tile_pipeline.phase.flat_scan.role.Main.info.verb.scan; tcg_d128_flat_cp_xpose_narrow:492 design.tile_pipeline.phase.flat_scan.role.Main.info.verb.scan; tcg_d128_flat_cp_xpose_wide:475 design.tile_pipeline.phase.flat_scan.role.Main.info.verb.scan; tcg_d128_flat_tma_direct_narrow:479 design.tile_pipeline.phase.flat_scan.role.Main.info.verb.scan; tcg_d128_flat_tma_direct_wide:493 design.tile_pipeline.phase.flat_scan.role.Main.info.verb.scan; tcg_d128_flat_tma_xpose_narrow:483 design.tile_pipeline.phase.flat_scan.role.Main.info.verb.scan; tcg_d128_flat_tma_xpose_wide:501 design.tile_pipeline.phase.flat_scan.role.Main.info.verb.scan

Each valid thread initializes one ScanState and loads batches of 64 consecutive groups into Batch before the ordered recurrence and snapshot stores, then handles batches of 16 and a scalar remainder. Consume each batch before reuse. No cross-thread scan reduction occurs.

## t_b96e54e09a25200d | pipeline.verb | 275 chars | 2 declarations

tc_split_segmented_narrow:591 design.tile_pipeline.phase.segmax.role.Main.info.verb.produce; tc_split_segmented_wide:625 design.tile_pipeline.phase.segmax.role.Main.info.verb.produce

Consume SegmentBatch through the segment step's ordered fmax and store SegmentMax[s,r], including s=0. There is no CTA barrier or final trigger. Ordinary stores consume their sources; kernel_done supplies visibility. No local state persists into the separately launched scan.

## t_ba4a4470502aa1a7 | pipeline.verb | 545 chars | 2 declarations

tcg_d128_fscan_cp_direct_narrow:548 design.tile_pipeline.phase.split.role.Main.info.verb.prefix; tcg_d128_fscan_cp_direct_wide:565 design.tile_pipeline.phase.split.role.Main.info.verb.prefix

Initialize DenState to zero and PrefixState to -inf. For Ng>0 every thread commits an empty copy group without waiting; Ng=0 performs no commit. All threads, including dead warps, then walk GroupMax[0:Tw(s),r] in ordered batches of 16 followed by a scalar remainder. Consume PrefixBatch before reuse. Empty splits perform the same bounded walk. MaxState continues in the same register as PrefixState; for Ng>0 load CurrentGroupMax at cLo, otherwise use -inf. This work can overlap primed K/V transfers. No global mstate array is read or written.

## t_ba937eae7754b5e7 | pipeline.verb | 471 chars | 2 declarations

tcg_d256_flat_cp_narrow:414 design.tile_pipeline.phase.group_max.role.Main.info.verb.land; tcg_d256_flat_cp_wide:428 design.tile_pipeline.phase.group_max.role.Main.info.verb.land

Issue K(g+1) and its atom into the other slot when present. Commit a cp.async group even at the tail, then wait_group 1 proves current payload and atoms complete. Apply the shared async-proxy fence and CTA-sync to publish them. Warps 0..3 copy current atoms to KeyScale, wait for TMEM stores, then fence before thread-sync and CTA-sync again. Current operands are ready; previous QK and its final barrier retired old tile/atom/scale readers before conflicting slot reuse.

## t_bae93903173bec16 | pipeline.verb | 273 chars | 7 declarations

tcg_d256_flat_cp_narrow:631 design.tile_pipeline.phase.split.role.Main.info.verb.write; tcg_d256_flat_cp_wide:645 design.tile_pipeline.phase.split.role.Main.info.verb.write; tcg_d256_flat_tma_narrow:626 design.tile_pipeline.phase.split.role.Main.info.verb.write; tcg_d256_flat_tma_wide:640 design.tile_pipeline.phase.split.role.Main.info.verb.write; tcg_d256_segmented_cp_narrow:699 design.tile_pipeline.phase.split.role.Main.info.verb.write; tcg_d256_segmented_tma_narrow:714 design.tile_pipeline.phase.split.role.Main.info.verb.write; tcg_d256_segmented_tma_wide:699 design.tile_pipeline.phase.split.role.Main.info.verb.write

CTA-sync publishes PartAStaging. All eight warps cooperatively read valid rows in contiguous 16-byte chunks and issue PartA stores with the inverse row mapping; half zero writes PartB. done consumes store sources, not global completion. No O/LSE write occurs in this phase.

## t_bb7cd2cebe6c02d9 | pipeline.verb | 631 chars | 1 declarations

tcg_d256_segmented_cp_wide:787 design.tile_pipeline.phase.split.role.Main.info.verb.advance

Apply normalize from ReplayPrevious/Current and retained Score; move ReplayCurrent/Next forward. Eligible warps rescale their 128 NumState channels only after the first group and when some alpha differs from one, using 32-channel TMEM round trips and store-completion waits. Update DenPart and pack Prob8 into TMEM. Wholly padded PSK warps skip rescaling and probability writes; their retained, possibly uninitialized Prob8 rows still feed PV. All warps wait for their stores, fence before thread-sync and CTA-sync, publishing live probabilities and NumState. Score is dead after this action; the barrier also retires atom readers.

## t_bbb039b768728d7f | pipeline.verb | 327 chars | 2 declarations

tcg_d128_flat_cp_xpose_narrow:426 design.tile_pipeline.phase.group_max.role.Main.info.verb.prime; tcg_d128_flat_cp_xpose_wide:409 design.tile_pipeline.phase.group_max.role.Main.info.verb.prime

For Ng>0 enqueue K(cLo) and its 512-byte scale atom into slot zero using 16-byte cp.async copies, then each thread commits its group. Invalid payload copies zero-fill from a clamped source address; allocated scale padding bytes are copied unchanged. For Ng=0 issue no copy or commit. done means enqueue and commit, not arrival.

## t_bcea0095ce04b4cd | pipeline.verb | 464 chars | 1 declarations

tc_split_segmented_wide:802 design.tile_pipeline.phase.combine.role.Main.info.verb.write

Only group zero reads the shared partials. Each of its threads computes its own SumB replica, then SumA and BF16 O for its owned channels. Only thread zero reads MaxTail and writes present LSE. Other groups perform no further data accesses; the role's write event remains defined. done means final stores have consumed their sources; kernel_done supplies caller-stream visibility. There is no second barrier, early trigger, next worktile or cross-call local state.

## t_bdbe19499b08ce5a | pipeline.verb | 451 chars | 1 declarations

root:556 design.tile_pipeline.phase.main.role.Main.info.verb.prefetch

Head(g) enqueues K(g+1) and KeyAtom(g+1); ValueNext(g) enqueues V(g+1) and ValueAtom(g+1) after retire:Previous(g). Each is a no-op at g=cHi, with no payload read or written. Otherwise its done denotes enqueue completion only, using the transaction accounting defined by prime. Slots are (g+1-cLo)%2: prior QK completion frees the K slot, and prior PV retirement frees the V slot. Incoming copies may remain in flight while subsequent actions execute.

## t_be867cfc2255eb22 | pipeline.verb | 457 chars | 1 declarations

fused_d256_tma:392 design.tile_pipeline.phase.main.role.Main.info.verb.land

Enqueue V(g+1) when it exists into the two-slot location freed by retire; at the first group this location has no prior occupant. Then wait for K(g+1) when it exists and for V(g). With iter=g-cLo, K uses slot (iter+1)%3 and parity floor((iter+1)/3)%2; V uses slot iter%2 and parity floor(iter/2)%2. Apply the shared async-proxy and before-thread-sync fences. done proves these operands ready; exchange provides the CTA ordering. V(g+1) may remain in flight.

## t_bf4b19ab9d4f4363 | pipeline.verb | 247 chars | 2 declarations

tcg_d128_segmented_cp_direct_narrow:666 design.tile_pipeline.phase.split.role.Main.info.verb.prime; tcg_d128_segmented_cp_direct_wide:683 design.tile_pipeline.phase.split.role.Main.info.verb.prime

For Ng>0 cooperatively enqueue K/V(cLo) and their atoms into slot zero and commit each thread's copy group. Payload copies zero-fill beyond Skv; atoms include allocated scale padding. For Ng=0 issue no copy or commit. done is enqueue, not arrival.

## t_bf5acaff9b5c848c | pipeline.verb | 178 chars | 4 declarations

tcg_d128_fscan_cp_xpose_narrow:569 design.tile_pipeline.phase.split.role.Main.info.verb.qk; tcg_d128_fscan_cp_xpose_wide:586 design.tile_pipeline.phase.split.role.Main.info.verb.qk; tcg_d128_fscan_tma_xpose_wide:574 design.tile_pipeline.phase.split.role.Main.info.verb.qk; tcg_d128_segmented_cp_xpose_wide:720 design.tile_pipeline.phase.split.role.Main.info.verb.qk

Enqueue four ordered QK contractions, first overwriting DotTile, commit, then wait for QK completion and apply the completion fence. No maximum readout is performed in this pass.

## t_c0154c74c6acd433 | pipeline.verb | 586 chars | 2 declarations

fused_d256_cp:392 design.tile_pipeline.phase.main.role.Main.info.verb.advance; fused_d256_tma:410 design.tile_pipeline.phase.main.role.Main.info.verb.advance

Apply normalize to each half's retained scores and DenPart. For g>cLo, warps with any alpha unequal to one rescale their own 128 NumState channels in 32-channel TMEM round trips and wait for stores; the first group never reads NumState. Each half packs its 64 Prob8 entries into the single TMEM probability tile and updates its own denominator. Wait for probability stores, then fence before thread synchronization and CTA-sync. This publishes Prob8 and rescaled NumState before PV and retires atom/Exchange readers before reuse. Score and group temporaries are dead; RowState persists.

## t_c1fce3102a963173 | pipeline.verb | 394 chars | 1 declarations

fused_d128_group:414 design.tile_pipeline.phase.main.role.Even.info.verb.lookahead

If g+2 exists, read its landed K atom into this role's single KeyScale slot, whose QK reader was just retired. wait::st, before fence and role barrier publish it. The leading thread applies the after fence, enqueues QK(g+2) into the just-freed RawScore tile and commits its QK barrier; done means enqueue. At the tail the bound check completes without atom reads or writes, MMA issue or commit.

## t_c213cae9addb08cf | pipeline.verb | 409 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:682 design.tile_pipeline.phase.split.role.Main.info.verb.setup

Establish coordinates and the head-major row-valid predicate; rowsLive is always true. Thread zero initializes QK/PV completion barriers with phase zero and issues MMA. All four warps remain cooperative participants; each thread maps to one row. All warps copy payload, the first warp copies K/V scale atoms, and each thread commits and waits its own copy groups. The exact grid never takes q0>=Sq early exit.

## t_c22f6db015dd328f | pipeline.verb | 328 chars | 3 declarations

tcg_d128_fscan_cp_xpose_narrow:432 design.tile_pipeline.phase.group_max.role.Main.info.verb.score; tcg_d128_fscan_cp_xpose_wide:449 design.tile_pipeline.phase.group_max.role.Main.info.verb.score; tcg_d128_segmented_cp_xpose_wide:519 design.tile_pipeline.phase.group_max.role.Main.info.verb.score

For non-Long enqueue this group's four 32-channel QK contractions and commit; for Long the preceding start/advance already issued them. Wait for the QK completion phase, toggle its parity and apply the completion fence. DotTile is now readable and K/KeyScale reads are retired. The first contraction overwrites the score region.

## t_c2871314fa7ad490 | pipeline.verb | 949 chars | 3 declarations

tcg_d128_fscan_cp_xpose_narrow:586 design.tile_pipeline.phase.split.role.Main.info.verb.stage_partials; tcg_d128_fscan_cp_xpose_wide:603 design.tile_pipeline.phase.split.role.Main.info.verb.stage_partials; tcg_d128_fscan_tma_xpose_wide:591 design.tile_pipeline.phase.split.role.Main.info.verb.stage_partials

Nonempty valid rows stage NumState times one; empty splits evaluate zeros. Each valid row writes its D channels into the PartA view of OperandArena; padded rows are neither staged nor later read. The 81920-byte operand capacity accommodates the 67584-byte staging span, including row spacing. For Ng>0, stage's CTA joins completed all Query writes, QK/PV waits retired operand reads, and the last group enqueued no future transfer, so the arena is free before these writes. For Ng=0, no matrix iteration or its CTA joins occur: a warp's zero staging writes can overlap another warp's unfinished Query writes in the same shared region. No pre-staging barrier orders those writes, so the staged bytes need not encode zero. The source's CTA barrier is after the staging writes. It joins all preparation/staging writes and any final TMEM readouts, making the resulting bytes readable; it does not repair the empty-split race. done includes this barrier.

## t_c33d64f39b06b6ac | pipeline.verb | 186 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:800 design.tile_pipeline.phase.combine.role.Main.info.verb.dependency

All threads execute the PDL dependency wait at kernel entry. done guarantees every PartA/PartB producer has completed; the earlier scan is complete transitively, including MTail for LSE.

## t_c3cf76e1f18289f9 | pipeline.verb | 236 chars | 3 declarations

tcg_d128_fscan_cp_xpose_narrow:550 design.tile_pipeline.phase.split.role.Main.info.verb.dependency.meaning; tcg_d128_fscan_cp_xpose_wide:567 design.tile_pipeline.phase.split.role.Main.info.verb.dependency.meaning; tcg_d128_fscan_tma_xpose_wide:557 design.tile_pipeline.phase.split.role.Main.info.verb.dependency.meaning

Each thread executes the PDL predecessor-grid wait after its own preparation and before its first GroupMax read. done guarantees all group_max results are visible. This wait does not join this CTA's warps or complete their Query writes.

## t_c3dcc76ac73d8bf8 | pipeline.verb | 437 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:722 design.tile_pipeline.phase.split.role.Main.info.verb.probability

All warps advance replayed MaxState/CurrentMax, rescale DenState and NumState, then consume DotTile in 32-key chunks through ScoreChunk and normalize, including padded rows. Pack every Prob8 chunk into TMEM and wait for its stores. Fence and CTA-sync before PV; this join publishes all probability rows and retires every DotTile reader before PV overwrites the same columns. Padded rows are computed but never written to global partials.

## t_c45c1b0c52f9f11c | pipeline.verb | 107 chars | 1 declarations

tcg_d128_segmented_cp_direct_wide:776 design.tile_pipeline.phase.combine.role.Main.info.verb.join

All sixteen warps execute a CTA barrier, publishing every MergeA and MergeB entry before cross-group reads.

## t_c638c09033a3ffda | pipeline.verb | 218 chars | 1 declarations

tc_fused:355 design.tile_pipeline.phase.main.role.Main.info.verb.close

Add E's component pairs to the rescaled DenPart. A CTA barrier joins current raw-input, decoded-scale and Vbf readers before the next prefetch/decode. Score and E are dead; NumState, MaxState and DenPart carry forward.

## t_c63f3b175793f094 | pipeline.verb | 425 chars | 1 declarations

root:614 design.tile_pipeline.phase.main.role.Main.info.verb.pv

After the issuer's after-thread-sync fence, enqueue the four ordered PV contractions and commit one PV completion phase. done denotes enqueue only; retire/drain own completion. Read Prob8 from TMEM, Value from its current shared slot, and the unit/V scale atoms. The first contraction of the first group overwrites NumState, later contractions accumulate into it. There is no per-group numerator readback or tail CTA barrier.

## t_c64fa3e1c08c5cec | pipeline.verb | 319 chars | 4 declarations

tcg_d128_segmented_cp_direct_narrow:564 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.test; tcg_d128_segmented_cp_direct_wide:581 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.test; tcg_d128_segmented_tma_direct_narrow:560 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.test; tcg_d128_segmented_tma_direct_wide:579 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.test

For chunk c load SegBatch summaries into SummaryBatch, with -inf for missing terminal segments. Test every summary against the same incoming ScanState and retain Cut(c,r) in ReplayStart. All summary values are consumed before entering the replay walk; updated states do not cause further summary tests within the chunk.

## t_c78e72f7d3f697a8 | pipeline.verb | 337 chars | 1 declarations

tcg_d256_segmented_cp_wide:629 design.tile_pipeline.phase.scan.role.Main.info.verb.first

Valid threads replay segment 0 unconditionally. The replay helper loads each full SegLen-group batch into GroupBatch before advancing ScanState and writing its states in order; an incomplete final batch uses scalar load/advance/store. This same helper is used by replay below. GroupBatch reads retire before the next batch overwrites it.

## t_c798384a4debc7e2 | pipeline.verb | 487 chars | 1 declarations

fused_d128_cp:322 design.tile_pipeline.phase.main.role.Main.info.verb.setup

Establish the work_schedule coordinates, row masks and QK/PV completion barriers, with wait parities initially zero. A q0>=Sq guard exits before allocation or copying; the exact launch grid never takes it. The allocation barrier in prepare publishes initialization. All 128 rows execute, including padding; each thread owns one row for score readout and row-state arithmetic. Thread 0 issues matrix products. All four warps copy K/V payloads, and the first warp copies their scale atoms.

## t_c7b39d3f50d95ae6 | pipeline.verb | 1326 chars | 1 declarations

fused_d256_cp:354 design.tile_pipeline.phase.main.role.Main.info.verb.ahead

Enqueue K(g+1) and V(g+1) when g+1<=cHi into slot (g+1-cLo)%2; K's allocated slot 2 remains unused. At the first group the destination has no old occupant; thereafter the preceding group's QK and PV waits have retired its readers. Enqueue both atoms(g+3) when g+3<=cHi into slot (g+3-cLo)%3. prime or the preceding staging barrier retires old atom readers before refill. Commit one cp.async group per thread, even if empty; wait_group 1 completes all older groups, including current K/V. The newest future copies may continue during computation. Warps 0..3 copy atoms(g+1) from slot (g+1-cLo)%3 into KeyScale(g+1)/ValueScale(g+1) and wait for TMEM stores. For g+1<=cHi, prime or a preceding iteration already completed and CTA-published those atom bytes. Both TMEM scale rings use parity g%2: current scales are preserved, and old readers of the next slot completed in the preceding group. The final group still copies atoms(g+1): stale bytes, or uninitialized bytes when Ng is one or two, reach an in-bounds TMEM slot with no matrix-product consumer. The source neither initializes nor suppresses this copy. Apply the shared async-proxy and before-thread-sync fences, then CTA-sync to publish current K/V and next scales and join old atom readers. Out-of-range prefetches issue no payload; commit/wait and publication remain.

## t_c7ff54d3e2ecf80a | pipeline.verb | 412 chars | 1 declarations

tcg_d128_fscan_tma_xpose_narrow:643 design.tile_pipeline.phase.combine.role.Main.info.verb.prime

Each thread reads MTail and owns one output channel. One thread per s<Ns loads PartB into DenParts and forms Weight[s] from PartM, evaluating combine's guarded expf once per split. Load the first eight numerator partials into NumeratorBatch, substituting zero for missing entries, then CTA-sync to publish DenParts and Weight. Selection guarantees Ns<=128; unused shared capacity is neither initialized nor read.

## t_c952ef1d88f35dbc | pipeline.verb | 391 chars | 1 declarations

fused_d128_group:494 design.tile_pipeline.phase.main.role.Even.info.verb.finish

A before fence and CTA barrier join final TMEM readers; the first warp deallocates the 512-column region. All threads reach fmha_pdl_trigger. This ordinary fused launch has no PDL input wait or dependent phase. Kernel completion supplies caller-stream visibility. No next worktile is assigned; later invocations initialize fresh numerical state, while host caches carry launch metadata only.

## t_c968817624c3bd5f | pipeline.verb | 120 chars | 7 declarations

tcg_d256_flat_cp_narrow:662 design.tile_pipeline.phase.combine.role.Main.info.verb.await; tcg_d256_flat_cp_wide:676 design.tile_pipeline.phase.combine.role.Main.info.verb.await; tcg_d256_flat_tma_narrow:657 design.tile_pipeline.phase.combine.role.Main.info.verb.await; tcg_d256_flat_tma_wide:672 design.tile_pipeline.phase.combine.role.Main.info.verb.await; tcg_d256_segmented_cp_narrow:730 design.tile_pipeline.phase.combine.role.Main.info.verb.await; tcg_d256_segmented_tma_narrow:745 design.tile_pipeline.phase.combine.role.Main.info.verb.await; tcg_d256_segmented_tma_wide:730 design.tile_pipeline.phase.combine.role.Main.info.verb.await

Every thread waits for the predecessor grid at entry, proving PartA/PartB visible and transitively proving scan's Mtail.

## t_cabdf20771a2bf90 | pipeline.verb | 538 chars | 1 declarations

root:532 design.tile_pipeline.phase.main.role.Main.info.verb.setup

Establish the work_schedule coordinates, row masks and completion protocols. A q0>=Sq guard exits before allocation or copying; the exact launch grid never takes it. Initialize separate two-slot K/V transaction barriers and QK/PV completion phases, each initially at parity zero, with initialization published before bulk use. All 128 rows execute in root, including padding; each thread owns one row for score readout and row-state arithmetic. Thread 0 issues the matrix products and TMA requests; the four warps cooperate on other work.

## t_cb9d3c515310f0b5 | pipeline.verb | 288 chars | 2 declarations

tcg_d128_flat_cp_direct_narrow:573 design.tile_pipeline.phase.split.role.Main.info.verb.setup; tcg_d128_flat_cp_direct_wide:587 design.tile_pipeline.phase.split.role.Main.info.verb.setup

Establish coordinates and PSK valid/live-warp predicates. Initialize QK/PV completion barriers at parity zero. All four warps cooperate on copies and synchronization; one thread maps to each row and owns an initially empty copy-group stream. Thread zero issues MMA. q0>=Sq is unreachable.

## t_cc8fc8529ab6513f | pipeline.verb | 315 chars | 1 declarations

tcg_d128_fscan_tma_xpose_narrow:647 design.tile_pipeline.phase.combine.role.Main.info.verb.sum_a

Consume the primed numerator batch, then load and consume further batches of eight. Weight valid split entries and accumulate in combine's order; missing entries add zero without reading Weight. Reuse a register batch after consumption. Shared weights remain unchanged through all numerator and denominator readers.

## t_ce99da86d58cb9de | pipeline.verb | 406 chars | 1 declarations

tcg_d256_flat_tma_wide:693 design.tile_pipeline.phase.combine.role.Main.info.verb.write

Group zero alone merges GroupDen into each thread's DenCopies and GroupNum for its two channels, then converts/writes O. Its thread zero alone reads Mtail and writes LSE when present. Other groups make no output access. Shared sums remain unchanged through all final reads; no next worktile reuses them. No extra wait or trigger follows ordinary stores; kernel completion provides caller-stream visibility.

## t_cef0afc3ce91da57 | pipeline.verb | 270 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:592 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.signal

Signal permission for split to start immediately after that wait, before any scan arithmetic. Out-of-range lanes also signal; their subsequent scan/tail actions have no data effects, matching the source early return. This launch_ready permission does not publish MState.

## t_d160430728334867 | pipeline.verb | 550 chars | 1 declarations

tc_split_segmented_wide:790 design.tile_pipeline.phase.combine.role.Main.info.verb.accumulate

Each group traverses its own interval. For a full batch, load eight independent FP32 PartialA values per thread into NumeratorBatch and then add them to the first owned GroupA channel. At D=256 do the same for the second channel before advancing to the next split batch. All batch reads end before its registers are reused. Consume the remaining splits one at a time for all owned channels, without zero-padding additions. Store every GroupFinalA entry to shared memory. There is no combine-side weight, CTA barrier or cross-group sum in this action.

## t_d22571beb0722177 | pipeline.verb | 247 chars | 2 declarations

tcg_d128_fscan_tma_direct_narrow:546 design.tile_pipeline.phase.split.role.Main.info.verb.prefetch; tcg_d128_fscan_tma_direct_wide:564 design.tile_pipeline.phase.split.role.Main.info.verb.prefetch

Load NextGroupMax at g+1 when g<cHi, otherwise use -inf, including in dead warps. Enqueue next K/V and atoms in the alternate slot only when g<cHi. Prior QK/PV completion retired that slot's old operand reads; done does not wait for new transfers.

## t_d2ab8bc1fcbc7990 | pipeline.verb | 275 chars | 4 declarations

tc_split_flat_narrow:467 design.tile_pipeline.phase.group_max.role.Main.info.verb.setup; tc_split_flat_wide:484 design.tile_pipeline.phase.group_max.role.Main.info.verb.setup; tc_split_segmented_narrow:542 design.tile_pipeline.phase.group_max.role.Main.info.verb.setup; tc_split_segmented_wide:576 design.tile_pipeline.phase.group_max.role.Main.info.verb.setup

Establish rows and gmLo/gmHi. Each warp owns 16 rows; a four-lane subgroup shares two rows and partitions key pairs. The exact grid cannot take q0>=Sq. Subgroup representatives write the excluded Wgm prefix/suffix, even on empty intervals; other maximum cells stay untouched.

## t_d3b663d2b447f8a2 | pipeline.verb | 212 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:590 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.dependency

Every thread executes the PDL grid-dependency wait before any early exit or input load. done guarantees segment_max completion and SegMax visibility, and transitively group_max completion and GroupMax visibility.

## t_d3cdf1692f440ddc | pipeline.verb | 974 chars | 1 declarations

tcg_d256_segmented_cp_wide:763 design.tile_pipeline.phase.split.role.Main.info.verb.ahead

Prefetch Mscan[g+1] when g+1<=e, otherwise set ReplayNext=-inf. Enqueue K/V(g+1) into the other two-slot payload stage when present, and atoms(g+3) into slot (g+3-l)%3 when present. Commit payloads and atoms in one cp.async group, even if empty, then wait_group 1, proving current K/V and any atoms(g+2) from the preceding commit; prime already proved the initial copies. Warps 0..3 copy atoms(g+1) to the other TMEM scale parity and wait for stores. These bytes were completed and CTA-published by prime or an earlier iteration when g+1<=e. Prior QK/PV retirement frees the TMEM parity and tile slot; the preceding head barrier orders old atom readers before refill. The final iteration still copies atoms(e+1): bytes may be stale or uninitialized for C=1 or 2, but the in-bounds TMEM destination has no matrix consumer. No extra initialization is performed. Shared async-proxy and before-thread-sync fences plus CTA-sync publish current payload and the ahead scale stores.

## t_d487b805395cc123 | pipeline.verb | 551 chars | 2 declarations

tcg_d128_fscan_cp_direct_narrow:626 design.tile_pipeline.phase.combine.role.Main.info.verb.sum_a; tcg_d128_fscan_tma_direct_narrow:619 design.tile_pipeline.phase.combine.role.Main.info.verb.sum_a

Each thread reads MTail; threads cooperatively load Ns PartB entries into DenParts and compute one Weight per split from PartM into shared storage. Ns<=128; unused capacity remains unwritten and unread. Each thread owns one channel. Load the first eight numerator partials into NumeratorBatch, then CTA-sync to publish DenParts and Weight before weighted additions. Sum increasing splits, loading later eight-entry batches after the join and consuming each batch before register reuse. Missing entries add zero without reading or multiplying a weight.

## t_d4eaca3e738fadf4 | pipeline.verb | 316 chars | 7 declarations

tcg_d128_flat_cp_xpose_narrow:610 design.tile_pipeline.phase.split.role.Main.info.verb.probability; tcg_d128_flat_cp_xpose_wide:593 design.tile_pipeline.phase.split.role.Main.info.verb.probability; tcg_d128_flat_tma_xpose_narrow:597 design.tile_pipeline.phase.split.role.Main.info.verb.probability; tcg_d128_flat_tma_xpose_wide:616 design.tile_pipeline.phase.split.role.Main.info.verb.probability; tcg_d128_segmented_cp_xpose_narrow:669 design.tile_pipeline.phase.split.role.Main.info.verb.probability; tcg_d128_segmented_tma_xpose_narrow:685 design.tile_pipeline.phase.split.role.Main.info.verb.probability; tcg_d128_segmented_tma_xpose_wide:669 design.tile_pipeline.phase.split.role.Main.info.verb.probability

All warps advance Replay, rescale DenState/NumState, consume DotTile in 32-key chunks and write Prob8, including padded rows. Wait for TMEM stores, fence and CTA-sync to publish every probability row and retire all DotTile readers before PV overwrites it. Padded rows are computed but do not produce global partials.

## t_d51f39d2c49c34fc | pipeline.verb | 1112 chars | 1 declarations

fused_d256_tma:360 design.tile_pipeline.phase.main.role.Main.info.verb.ahead

Enqueue K(g+2) when g+2<=cHi and both atoms(g+3) when g+3<=cHi, in slots (g+2-cLo)%3 and (g+3-cLo)%3 respectively. The first K lookahead slot has no old occupant; thereafter QK(g-1) completion frees it. The preceding publication barrier, or prime at g=cLo, orders old atom reads before their slot refill. Every iteration commits one cp.async group, even if empty, and wait_group 1 proves all but the newest group complete. Warps 0..3 then copy atoms(g+1) from slot (g+1-cLo)%3 to KeyScale(g+1) and ValueScale(g+1), waiting for their TMEM stores. For g+1<=cHi, these bytes were completed and CTA-published by prime or a preceding iteration. Two K scale slots avoid QK(g); three V scale slots avoid the still-live PV(g-1). The newest cp.async and TMA requests may remain in flight. This scale copy also executes at g=cHi: it reads stale atom bytes, or uninitialized bytes when Ng is one or two, and writes an in-bounds TMEM slot with no matrix-product consumer. The source neither suppresses nor initializes this copy. Out-of-range prefetches issue no payload requests; commit/wait and scale copying still execute.

## t_d5729c4e211ac14a | pipeline.verb | 500 chars | 4 declarations

tcg_d128_fscan_cp_direct_narrow:562 design.tile_pipeline.phase.split.role.Main.info.verb.probability; tcg_d128_fscan_cp_direct_wide:579 design.tile_pipeline.phase.split.role.Main.info.verb.probability; tcg_d128_fscan_tma_direct_narrow:555 design.tile_pipeline.phase.split.role.Main.info.verb.probability; tcg_d128_fscan_tma_direct_wide:573 design.tile_pipeline.phase.split.role.Main.info.verb.probability

Live warps advance MaxState from CurrentGroupMax, move NextGroupMax into its current register, rescale DenState/NumState, consume DotTile in 32-key chunks and write Prob8. Dead warps skip scalar readout, rescaling and probability writes; their P rows remain uninitialized. All warps wait for their TMEM stores, fence and CTA-sync, retiring all DotTile readers before PV overwrites it. PV evaluates all 128 rows; unwritten P rows feed only discarded result rows and are not filled with invented zeros.

## t_d5ae777bb071a307 | pipeline.verb | 437 chars | 4 declarations

tc_split_flat_narrow:488 design.tile_pipeline.phase.group_max.role.Main.info.verb.write; tc_split_flat_wide:505 design.tile_pipeline.phase.group_max.role.Main.info.verb.write; tc_split_segmented_narrow:563 design.tile_pipeline.phase.group_max.role.Main.info.verb.write; tc_split_segmented_wide:597 design.tile_pipeline.phase.group_max.role.Main.info.verb.write

The source falls through the non-split output epilogue with zero denominator, storing OTemporary and present LseTemporary. Every interval writes these same caller locations, including empty intervals. Ordinary stores consume their sources; kernel_done supplies stream visibility. There is no programmatic trigger. Every payload copy was waited before use; the final empty commit has no payload to drain. Empty intervals issued no copies.

## t_d63e288672cf4075 | pipeline.verb | 422 chars | 2 declarations

fused_d256_cp:334 design.tile_pipeline.phase.main.role.Main.info.verb.prepare; fused_d256_tma:334 design.tile_pipeline.phase.main.role.Main.info.verb.prepare

Prefetch Q payload and scale words into QueryPrefetch before allocation. The first warp allocates TensorMemory; a CTA barrier publishes its base before that warp relinquishes the permit. All warps write Query; warps 0..3 write QueryScale and UnitScale and wait for TMEM stores. Discard QueryPrefetch, initialize RowState as in normalize, and leave NumState uninitialized. Query and its invariant scales survive all groups.

## t_d7504e46b55b0a7b | pipeline.verb | 184 chars | 4 declarations

tcg_d128_segmented_cp_direct_narrow:570 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.tail; tcg_d128_segmented_cp_direct_wide:587 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.tail; tcg_d128_segmented_tma_direct_narrow:566 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.tail; tcg_d128_segmented_tma_direct_wide:585 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.tail

Issue MTail after the final snapshot. All R*T snapshots and R tails are rewritten; kernel completion supplies visibility. This CTA has no further worktile and discards its local state.

## t_d7db00ff5bf8a8f0 | pipeline.verb | 266 chars | 4 declarations

tc_split_flat_narrow:476 design.tile_pipeline.phase.group_max.role.Main.info.verb.prefetch; tc_split_flat_wide:493 design.tile_pipeline.phase.group_max.role.Main.info.verb.prefetch; tc_split_segmented_narrow:551 design.tile_pipeline.phase.group_max.role.Main.info.verb.prefetch; tc_split_segmented_wide:585 design.tile_pipeline.phase.group_max.role.Main.info.verb.prefetch

For g<gmHi enqueue the next Kraw and KeyAtom into (g+1-gmLo)%2. Commit every iteration, including an empty final group. done is enqueue/commit only. The target slot is unused initially; later the preceding group-end barrier has joined all readers of its old version.

## t_d7fdd278d82b6c01 | pipeline.verb | 175 chars | 6 declarations

tcg_d256_flat_cp_narrow:665 design.tile_pipeline.phase.combine.role.Main.info.verb.prepare; tcg_d256_flat_cp_wide:679 design.tile_pipeline.phase.combine.role.Main.info.verb.prepare; tcg_d256_flat_tma_narrow:660 design.tile_pipeline.phase.combine.role.Main.info.verb.prepare; tcg_d256_segmented_cp_narrow:733 design.tile_pipeline.phase.combine.role.Main.info.verb.prepare; tcg_d256_segmented_tma_narrow:748 design.tile_pipeline.phase.combine.role.Main.info.verb.prepare; tcg_d256_segmented_tma_wide:733 design.tile_pipeline.phase.combine.role.Main.info.verb.prepare

Thread tid loads PartB[tid,row] to SharedDen when tid<Ns and owns channels tid and tid+128. Initialize numerator sums to zero; unused shared slots remain unwritten and unread.

## t_d82ba327a950962a | pipeline.verb | 351 chars | 1 declarations

tcg_d128_fscan_tma_xpose_wide:671 design.tile_pipeline.phase.combine.role.Main.info.verb.prime

Every group loads its first eight numerator partials per channel into NumeratorBatch, using zero without a read for missing entries. Then all 16 warps execute a CTA barrier, publishing Weight before any weighted products. done covers that barrier and the first batch. Weights remain unchanged until every numerator and denominator reader has finished.

## t_d8358a50726853c8 | pipeline.verb | 587 chars | 6 declarations

tcg_d128_flat_cp_xpose_narrow:626 design.tile_pipeline.phase.split.role.Main.info.verb.write; tcg_d128_flat_cp_xpose_wide:609 design.tile_pipeline.phase.split.role.Main.info.verb.write; tcg_d128_flat_tma_xpose_wide:632 design.tile_pipeline.phase.split.role.Main.info.verb.write; tcg_d128_segmented_cp_xpose_narrow:685 design.tile_pipeline.phase.split.role.Main.info.verb.write; tcg_d128_segmented_tma_xpose_narrow:701 design.tile_pipeline.phase.split.role.Main.info.verb.write; tcg_d128_segmented_tma_xpose_wide:685 design.tile_pipeline.phase.split.role.Main.info.verb.write

All threads cooperatively read 16-byte chunks of actual shared bytes for valid staged rows and write global PartA through domain's head-major map. FP32 bits are preserved, including possible empty-split corruption. Each valid row owner separately writes weighted DenState to PartB; empty splits write PartB=0 from registers. done means stores issued with sources consumed; no later local reuse occurs and kernel completion provides global visibility. No O/LSE is written here. Combine reads every PartA, so corruption can reach TotalA and O; there is no PartA data path to TotalB or LSE.

## t_d95c0a1d019cdd1c | pipeline.verb | 304 chars | 1 declarations

tcg_d128_segmented_cp_direct_wide:773 design.tile_pipeline.phase.combine.role.Main.info.verb.sum_b

Each group's 128 threads load PartB[a_z+d,r] when a_z+d<b_z, otherwise use zero. Apply combine's 32-entry tree with shuffle-down offsets 16,8,4,2,1; each warp's lane zero writes MergeB[z,w]. Even warps containing only padding contribute their zero partials. No fused-scan weight array is read or written.

## t_d96693404add7034 | pipeline.verb | 195 chars | 20 declarations

tcg_d128_flat_cp_direct_narrow:443 design.tile_pipeline.phase.group_max.role.Main.info.verb.advance; tcg_d128_flat_cp_direct_wide:457 design.tile_pipeline.phase.group_max.role.Main.info.verb.advance; tcg_d128_flat_cp_xpose_narrow:452 design.tile_pipeline.phase.group_max.role.Main.info.verb.advance; tcg_d128_flat_cp_xpose_wide:435 design.tile_pipeline.phase.group_max.role.Main.info.verb.advance; tcg_d128_flat_tma_direct_narrow:440 design.tile_pipeline.phase.group_max.role.Main.info.verb.advance; tcg_d128_flat_tma_direct_wide:454 design.tile_pipeline.phase.group_max.role.Main.info.verb.advance; tcg_d128_flat_tma_xpose_narrow:444 design.tile_pipeline.phase.group_max.role.Main.info.verb.advance; tcg_d128_flat_tma_xpose_wide:462 design.tile_pipeline.phase.group_max.role.Main.info.verb.advance; tcg_d128_fscan_cp_direct_narrow:434 design.tile_pipeline.phase.group_max.role.Main.info.verb.advance; tcg_d128_fscan_cp_direct_wide:451 design.tile_pipeline.phase.group_max.role.Main.info.verb.advance; tcg_d128_fscan_tma_direct_narrow:429 design.tile_pipeline.phase.group_max.role.Main.info.verb.advance; tcg_d128_fscan_tma_direct_wide:447 design.tile_pipeline.phase.group_max.role.Main.info.verb.advance; tcg_d128_fscan_tma_xpose_narrow:435 design.tile_pipeline.phase.group_max.role.Main.info.verb.advance; tcg_d128_segmented_cp_direct_narrow:499 design.tile_pipeline.phase.group_max.role.Main.info.verb.advance; tcg_d128_segmented_cp_direct_wide:516 design.tile_pipeline.phase.group_max.role.Main.info.verb.advance; tcg_d128_segmented_cp_xpose_narrow:485 design.tile_pipeline.phase.group_max.role.Main.info.verb.advance; tcg_d128_segmented_tma_direct_narrow:495 design.tile_pipeline.phase.group_max.role.Main.info.verb.advance; tcg_d128_segmented_tma_direct_wide:514 design.tile_pipeline.phase.group_max.role.Main.info.verb.advance; tcg_d128_segmented_tma_xpose_narrow:501 design.tile_pipeline.phase.group_max.role.Main.info.verb.advance; tcg_d128_segmented_tma_xpose_wide:485 design.tile_pipeline.phase.group_max.role.Main.info.verb.advance

For Long and g<cHi enqueue and commit QK(g+1) using next K and its alternating scale; done means enqueue. Otherwise perform no product or wait. This permits QK(g+1) under the next prefetch/stage.

## t_da74606dadbd67ad | pipeline.verb | 743 chars | 1 declarations

tcg_d128_fscan_cp_direct_wide:643 design.tile_pipeline.phase.combine.role.Main.info.verb.sum_a

Fix compile-time MergeGroups=4 and MergeWidth=128. Four consecutive four-warp groups z own the split intervals [Slo(z),Shi(z)); each thread owns one channel and one potential denominator term. Clear GroupAccumulator and read MTail. For each local slot t, store Weight[Slo(z)+t] at shared slot (z,t) when it exists, otherwise zero without reading PartM. Load the first eight numerator entries per channel into NumeratorBatch with guarded zero fills, then CTA-sync to publish weights before any product. Perform combine's ordered local numerator sum; later batches reuse registers only after prior values are consumed. Store GroupA[z,:] in shared memory. done covers each thread's own GroupA store; the sum_b barrier publishes them collectively.

## t_daf4bd87a21d5feb | pipeline.verb | 92 chars | 4 declarations

tcg_d256_flat_cp_narrow:460 design.tile_pipeline.phase.scan.role.Main.info.verb.await; tcg_d256_flat_cp_wide:474 design.tile_pipeline.phase.scan.role.Main.info.verb.await; tcg_d256_flat_tma_narrow:458 design.tile_pipeline.phase.scan.role.Main.info.verb.await; tcg_d256_flat_tma_wide:472 design.tile_pipeline.phase.scan.role.Main.info.verb.await

Every thread performs the PDL dependency wait at entry, proving all GroupMax writes visible.

## t_db717f83098ff389 | pipeline.verb | 240 chars | 1 declarations

tc_fused:344 design.tile_pipeline.phase.main.role.Main.info.verb.advance

Apply normalize's maximum update. If any lane in a warp has a nonunit alpha for either owned row, rescale that warp's NumState and DenPart; otherwise leave them unchanged. Initialize E to zero. Score survives all current probability chunks.

## t_dba2c0a0724b7577 | pipeline.verb | 366 chars | 3 declarations

tcg_d256_segmented_cp_narrow:502 design.tile_pipeline.phase.segment_max.role.Main.info.verb.produce; tcg_d256_segmented_tma_narrow:517 design.tile_pipeline.phase.segment_max.role.Main.info.verb.produce; tcg_d256_segmented_tma_wide:502 design.tile_pipeline.phase.segment_max.role.Main.info.verb.produce

Valid threads load all SegLen entries of SegmentGroups into GroupBatch before reducing; absent groups use -inf without a read. Apply segment_max and write SegMax[f,row]. Invalid rows have no data accesses. GroupBatch dies after reduction; there is no CTA barrier or exit trigger. Kernel completion publishes the results to scan; no local resource crosses the launch.

## t_dcb982e4882940f8 | pipeline.verb | 435 chars | 1 declarations

root:594 design.tile_pipeline.phase.main.role.Main.info.verb.retire

Previous(g) waits for PV(g-1) when g>cLo; at g=cLo it is a no-op with no previous accumulator or operand reads. A real wait observes the committed PV phase, advances its parity once and applies the after-thread-sync fence. It proves all reads of that PV's Value, ValueScale and Prob8 are finished and its NumState updates are complete. Previous occurs after both current QK halves, allowing PV to overlap their preparation and readout.

## t_dd270aec6f8db16e | pipeline.verb | 546 chars | 1 declarations

fused_d128_key:325 design.tile_pipeline.phase.main.role.Main.info.verb.setup

Establish work coordinates and row masks; the q0>=Sq guard is unreachable under the exact grid. Initialize separate QK/PV completion protocols with parity zero. Threads u and 128+u own the same row: warpgroup z handles keys [64*z,64*z+64) for score/Prob8 work and channels [64*z,64*z+64) for NumState rescale and output. Both retain MaxState; each retains its own DenSlice. All eight warps participate, including padded rows. Thread 0 issues every QK/PV chain; warps 0..3 alone write scales into the shared TMEM lanes, avoiding duplicate writers.

## t_ddacb9de1e78c895 | pipeline.verb | 944 chars | 1 declarations

tcg_d256_segmented_cp_wide:859 design.tile_pipeline.phase.combine.role.Output.info.verb.reduce

GroupPartials denotes the GroupNum and GroupDen contributions. Group p consists of threads 128*p through 128*p+127; each thread's local index d0 owns channels d0 and d0+128. Output is group zero and Partners comprises groups 1 through SP-1. Initialize LocalNumerator to zero. For the group's interval [sLo(p),sHi(p)), issue eight independent PartA loads for one channel and add in source order, then do the other channel; repeat full batches and finish remaining splits singly. No out-of-interval load or zero-padded numerator addition occurs. Each batch dies after accumulation. Store the two sums in GroupNum[p,:], including zeros for an empty interval. Each thread then loads PartB[sLo(p)+d0,row] once if in range, otherwise uses zero. All lanes apply output's reduce32 tree; each warp's lane zero stores its GroupDen entry. Numerator sources and DenLane die after their shared stores; done does not yet publish those stores to other groups.

## t_de78261d637b5c0f | pipeline.verb | 342 chars | 1 declarations

tc_split_segmented_wide:786 design.tile_pipeline.phase.combine.role.Main.info.verb.entry

Wait for grid dependencies before partial reads. The ordinary split predecessor emits no early trigger, so delivery requires its completion. Group sg occupies warps sg*CombineBt/32..(sg+1)*CombineBt/32-1. Its local thread d0 owns channels d0+c*CombineBt, c=0..D/CombineBt-1, and initializes its GroupA entries to zero. No row guard is needed.

## t_de7ad62969f4478f | pipeline.verb | 308 chars | 2 declarations

tcg_d128_fscan_cp_xpose_narrow:423 design.tile_pipeline.phase.group_max.role.Main.info.verb.prefetch; tcg_d128_fscan_cp_xpose_wide:440 design.tile_pipeline.phase.group_max.role.Main.info.verb.prefetch

Enqueue K(g+2) and its atom for Long, otherwise K(g+1) and its atom, only when that group is <=cHi. Every thread then commits a copy group, including an empty group past the tail. done is enqueue only. The slot being refilled has no remaining QK readers; its scale bytes were consumed at an earlier CTA join.

## t_df4b5def3696da37 | pipeline.verb | 388 chars | 3 declarations

tcg_d128_fscan_cp_xpose_narrow:582 design.tile_pipeline.phase.split.role.Main.info.verb.tail.meaning; tcg_d128_fscan_cp_xpose_wide:599 design.tile_pipeline.phase.split.role.Main.info.verb.tail.meaning; tcg_d128_fscan_tma_xpose_wide:587 design.tile_pipeline.phase.split.role.Main.info.verb.tail.meaning

Valid row owners in s=Ns-1 issue MTail stores from MaxState after their matrix loop or empty split's prefix, ordered by each thread's source instruction sequence. Other splits and padded rows issue no stores. done covers this action across the CTA without adding a pre-staging join; each warp can proceed independently. Only combine consumes MTail, after the split kernel completion wait.

## t_e111c213cb230216 | pipeline.verb | 563 chars | 1 declarations

root:577 design.tile_pipeline.phase.main.role.Main.info.verb.qk

Enqueue the scores step's four native block-scaled products for half z into ScoreHalf, then commit one QK completion. done here means enqueue only. Each half's first contraction overwrites. For z=1, a before-thread-sync fence and CTA barrier first prove that all warps have completed half zero's readout before the same 64 columns are overwritten; the issuer uses the matching after-thread-sync fence. The next group's first half is behind the previous probability-publication barrier and this group's land:Key barrier, so its old score readers have also retired.

## t_e18c0385c83fbc0d | pipeline.verb | 364 chars | 1 declarations

tcg_d256_segmented_cp_wide:813 design.tile_pipeline.phase.split.role.Main.info.verb.stage

Each half reads its 128 NumState channels in 32-channel self-retiring TMEM loads when C>0, or synthesizes zero chunks when C=0. Valid rows multiply by weight and write PartAStaging at raw shared base with 1040-byte row pitch, including 16-byte padding per row. This overwrites the retired Query/Key/Value regions in OperandScratch. Padded rows do not stage output.

## t_e1cd4ad3005c3dbc | pipeline.verb | 831 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:595 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.scan

One valid thread owns one row and initializes ScanState. Walk segment zero unconditionally, using WalkBatch to load 16 groups before their ordered recurrence and stores, with a scalar tail for a partial segment. For each later segment batch, load all SegmentBounds registers with -inf for missing segments, then consume every bound to build the 16-bit jump mask against the same incoming state. Consume all bounds before calling the walk helper. The lowest set bit selects the suffix: a zero mask performs only constant MState stores, otherwise store the unchanged prefix and walk the whole remaining batch. WalkBatch is reused only after its values have been consumed. ScanState carries across helper calls and constant fills. No cross-thread reduction or CTA join is needed. Rows outside R perform no loads, arithmetic or stores.

## t_e21e5db22f9ac34a | pipeline.verb | 728 chars | 1 declarations

tcg_d128_fscan_tma_xpose_wide:561 design.tile_pipeline.phase.split.role.Main.info.verb.prefix.meaning

Each thread initializes MaxState=-inf and DenState=0 and performs scan's bounded prefix, loading batches of 16 GroupMax values into PrefixBatch before ordered updates, then a scalar remainder. The register batch is consumed before reuse. All rows participate; padding uses that head's first real row. Load CurrentMax from GroupMax[cLo,r] only for Ng>0, otherwise keep -inf. CurrentMax/NextMax cache GroupMax values, not delayed states. No global scan snapshot or MTail is read. done covers all prefix work; a warp can continue with its own registers before others finish. The following stage or stage_partials CTA join covers collective completion. This work can overlap primed K/V transactions and also occurs for empty splits.

## t_e2e3c12e934f969b | pipeline.verb | 421 chars | 1 declarations

tcg_d256_segmented_cp_wide:509 design.tile_pipeline.phase.group_max.role.Main.info.verb.prepare

Prefetch Q and its scale words into QueryPrefetch before the first warp allocates 256 TMEM columns. CTA-sync publishes the base, then that warp relinquishes the allocation permit. All warps write Query; warps 0..3 write QueryScale and wait for TMEM stores. QueryPrefetch is dead after these stores. With C>0, commit an additional empty cp.async group, matching the hoisted prime's accounting; C=0 performs no commit here.

## t_e2ee7a2b63d8ea63 | pipeline.verb | 477 chars | 1 declarations

fused_d128_cp:346 design.tile_pipeline.phase.main.role.Main.info.verb.prefetch

Next(g) enqueues Key, Value and both atoms for g+1 into slot (g+1-cLo)%2 using prime's copy rule, then commits one cp.async group per thread. At g=cHi it reads and writes no payload or atom, but still commits an empty group. done denotes enqueue and commit completion only. For g>cLo, the prior iteration's QK readout and PV retirement have freed this slot's old operands; its atom reads ended at the prior land. The next copies may remain in flight during current computation.

## t_e381a307bd092cb1 | pipeline.verb | 253 chars | 1 declarations

tcg_d128_fscan_tma_xpose_narrow:553 design.tile_pipeline.phase.split.role.Main.info.verb.prefetch

Load NextMax=GroupMax[g+1,r] when g<cHi, otherwise use -inf, including for padded rows. Enqueue next K/V and atoms in the alternate slot only when g<cHi. Prior QK/PV completion retired that slot's old operand reads; done does not wait for new transfers.

## t_e410db6a2c70cb03 | pipeline.verb | 566 chars | 1 declarations

tc_split_flat_wide:675 design.tile_pipeline.phase.combine.role.Main.info.verb.write

Only group zero sums all GroupB entries, forming one denominator replica per thread, and merges GroupA for its channels before the combine step writes BF16 O. Only its d0=0 thread reads MaxTail and writes present LSE. Other groups perform no further numerical work; this action still completes for the role with no additional reads or stores on those groups. Neither shared buffer is overwritten, so no final CTA barrier is needed. done consumes output-store sources; kernel_done supplies caller-stream visibility. No next worktile or cross-call local state remains.

## t_e501ee254935fa70 | pipeline.verb | 432 chars | 8 declarations

tcg_d128_flat_cp_direct_narrow:600 design.tile_pipeline.phase.split.role.Main.info.verb.probability; tcg_d128_flat_cp_direct_wide:614 design.tile_pipeline.phase.split.role.Main.info.verb.probability; tcg_d128_flat_tma_direct_narrow:593 design.tile_pipeline.phase.split.role.Main.info.verb.probability; tcg_d128_flat_tma_direct_wide:607 design.tile_pipeline.phase.split.role.Main.info.verb.probability; tcg_d128_segmented_cp_direct_narrow:691 design.tile_pipeline.phase.split.role.Main.info.verb.probability; tcg_d128_segmented_cp_direct_wide:708 design.tile_pipeline.phase.split.role.Main.info.verb.probability; tcg_d128_segmented_tma_direct_narrow:683 design.tile_pipeline.phase.split.role.Main.info.verb.probability; tcg_d128_segmented_tma_direct_wide:702 design.tile_pipeline.phase.split.role.Main.info.verb.probability

Live warps advance Replay, rescale DenState/NumState, consume DotTile in 32-key chunks and write Prob8. Dead warps skip scalar readout, rescaling and probability writes; their P rows remain uninitialized. All warps wait for their TMEM stores, fence and CTA-sync, retiring all DotTile readers before PV overwrites it. PV evaluates all 128 rows; unwritten P rows feed only discarded result rows and are not filled with invented zeros.

## t_e51131686cf95479 | pipeline.verb | 371 chars | 1 declarations

fused_d128_cp:330 design.tile_pipeline.phase.main.role.Main.info.verb.prime

Enqueue group cLo into slot zero: 16384 bytes each for Key and Value, plus 512 bytes each for KeyAtom and ValueAtom. Cooperative 16-byte cp.async copies zero-fill out-of-range key payloads; atoms still read the allocated padded scale bytes. Each thread commits its K/V and atom copies together. done means enqueue and commit completion; land owns the readiness guarantee.

## t_e675161b2e4bc746 | pipeline.verb | 128 chars | 1 declarations

tcg_d256_segmented_cp_wide:739 design.tile_pipeline.phase.split.role.Main.info.verb.await

Perform the PDL dependency wait after independent preparation, before any scan-output read; done proves Mscan and Mtail visible.

## t_e69af6adbad2ae2a | pipeline.verb | 558 chars | 1 declarations

tcg_d256_segmented_cp_wide:742 design.tile_pipeline.phase.split.role.Main.info.verb.replay

Initialize both DenPart copies to zero. Load ReplayPrevious at l-1 when l>0, otherwise use -inf; load ReplayCurrent at l only when C>0. Every thread performs these source reads, including padding, using query index zero for invalid rows. The l-1 load has no C>0 or l<=T guard: an empty split can read beyond the logical Mscan array into subsequent workspace bytes. Its value is discarded without influencing the zero partials; no initialization, bound clamp or additional wait is supplied. NumState storage remains uninitialized until the first PV overwrite.

## t_e6a3c279658b1d15 | pipeline.verb | 172 chars | 2 declarations

tc_split_flat_narrow:656 design.tile_pipeline.phase.combine.role.Main.info.verb.publish; tc_split_segmented_narrow:763 design.tile_pipeline.phase.combine.role.Main.info.verb.publish

A CTA barrier publishes the complete valid Denominator region after all numerator batches for all channel slices. This shared vector is consumed once and never overwritten.

## t_e6ae62cacc60e94d | pipeline.verb | 561 chars | 1 declarations

tc_fused:360 design.tile_pipeline.phase.main.role.Main.info.verb.write

Merge DenPart with two four-lane shuffle-add stages and perform value_output. Each lane writes paired BF16 output channels; one lane per subgroup writes optional LSE for each valid owned row. Padding issues no output stores. done means ordinary stores consumed their register sources, without a host or device-wide completion wait. No next worktile is assigned. All payload copies were waited before use; the last empty commit adds no payload to drain. Kernel completion provides caller-stream visibility. Later invocations initialize fresh CTA numerical state.

## t_e79191f31516e0ef | pipeline.verb | 288 chars | 2 declarations

tcg_d128_segmented_cp_direct_narrow:474 design.tile_pipeline.phase.group_max.role.Main.info.verb.prime; tcg_d128_segmented_cp_direct_wide:491 design.tile_pipeline.phase.group_max.role.Main.info.verb.prime

For Ng>0 cooperatively enqueue K(cLo) and its 512-byte scale atom into slot zero, then commit each thread's copy group. Payload copies zero-fill beyond Skv; atoms include allocated scale padding without a key-valid guard. For Ng=0 issue no copy or commit. done means enqueue, not arrival.

## t_e840ac64be1e2ce3 | pipeline.verb | 187 chars | 1 declarations

tcg_d256_segmented_cp_wide:876 design.tile_pipeline.phase.combine.role.Output.info.verb.join

All SP groups rendezvous at one CTA barrier, publishing every GroupNum and GroupDen entry. This is the only combine barrier; there is no fused-weight table or earlier publication barrier.

## t_e8ac163fa1c099a1 | pipeline.verb | 179 chars | 2 declarations

tcg_d128_fscan_cp_xpose_narrow:655 design.tile_pipeline.phase.combine.role.Main.info.verb.dependency; tcg_d128_fscan_cp_xpose_wide:674 design.tile_pipeline.phase.combine.role.Main.info.verb.dependency

All threads execute the PDL dependency wait at kernel entry. done guarantees every PartA, PartB, PartM and MTail producer in split has completed and its global writes are visible.

## t_e935285585b70c53 | pipeline.verb | 177 chars | 1 declarations

tcg_d256_segmented_cp_wide:578 design.tile_pipeline.phase.segment_max.role.Main.info.verb.await

Every thread then performs the PDL grid-dependency wait, proving GroupMax visible. Each thread owns row r0+thread_id; out-of-range threads return only after both protocol calls.

## t_e9c958761baebe5c | pipeline.verb | 236 chars | 3 declarations

tcg_d256_segmented_cp_narrow:527 design.tile_pipeline.phase.scan.role.Main.info.verb.begin; tcg_d256_segmented_tma_narrow:542 design.tile_pipeline.phase.scan.role.Main.info.verb.begin; tcg_d256_segmented_tma_wide:527 design.tile_pipeline.phase.scan.role.Main.info.verb.begin

Trigger split launch after await and before scanning. Thread tid owns r0+tid; out-of-range threads return after triggering. Valid threads initialize ScanState, the current Mscan state, to -inf. The trigger does not publish scan results.

## t_eb47e5301a69e30e | pipeline.verb | 597 chars | 2 declarations

tcg_d128_flat_cp_direct_narrow:584 design.tile_pipeline.phase.split.role.Main.info.verb.replay; tcg_d128_flat_cp_direct_wide:598 design.tile_pipeline.phase.split.role.Main.info.verb.replay

Initialize DenState to zero and MaxState to -inf. Every thread loads MState at (cLo-1)*R+rowG when cLo>0, including empty splits; load CurrentMax at cLo only when Ng>0. rowG is domain's r. The source has no any_compute guard or T-bound clamp on the entrance load: a logically out-of-range empty-split read is not given an initialization or safety guarantee here. Its loaded value is discarded from empty-split results; dead-warps' loads likewise do not affect output or subsequent addresses. For Ng>0 each thread then commits an empty group at the original prime position; Ng=0 performs no commit.

## t_eb7910d8121be702 | pipeline.verb | 161 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:563 design.tile_pipeline.phase.segment_max.role.Main.info.verb.dependency

Every thread waits for the group_max grid before the row-range guard or first GroupMax load. done guarantees completion and visibility of all GroupMax producers.

## t_ebcc3f246308b7aa | pipeline.verb | 177 chars | 12 declarations

tcg_d128_flat_tma_direct_narrow:418 design.tile_pipeline.phase.group_max.role.Main.info.verb.prime; tcg_d128_flat_tma_direct_wide:432 design.tile_pipeline.phase.group_max.role.Main.info.verb.prime; tcg_d128_flat_tma_xpose_narrow:422 design.tile_pipeline.phase.group_max.role.Main.info.verb.prime; tcg_d128_flat_tma_xpose_wide:440 design.tile_pipeline.phase.group_max.role.Main.info.verb.prime; tcg_d128_fscan_tma_direct_narrow:407 design.tile_pipeline.phase.group_max.role.Main.info.verb.prime; tcg_d128_fscan_tma_direct_wide:425 design.tile_pipeline.phase.group_max.role.Main.info.verb.prime; tcg_d128_fscan_tma_xpose_narrow:413 design.tile_pipeline.phase.group_max.role.Main.info.verb.prime; tcg_d128_segmented_cp_xpose_narrow:463 design.tile_pipeline.phase.group_max.role.Main.info.verb.prime; tcg_d128_segmented_tma_direct_narrow:473 design.tile_pipeline.phase.group_max.role.Main.info.verb.prime; tcg_d128_segmented_tma_direct_wide:492 design.tile_pipeline.phase.group_max.role.Main.info.verb.prime; tcg_d128_segmented_tma_xpose_narrow:479 design.tile_pipeline.phase.group_max.role.Main.info.verb.prime; tcg_d128_segmented_tma_xpose_wide:463 design.tile_pipeline.phase.group_max.role.Main.info.verb.prime

For Ng>0 enqueue K(cLo) and its 512-byte scale atom into slot zero, with one barrier accounting for 16384+512 bytes; for Ng=0 issue no transfer. done means enqueue, not arrival.

## t_ed18a46d4a03bac3 | pipeline.verb | 396 chars | 1 declarations

fused_d128_group:486 design.tile_pipeline.phase.main.role.Even.info.verb.write

Role w reads output channels [64*w,64*w+64) in two self-retiring 32-channel chunks and applies value_output. Valid rows issue BF16 RNE stores directly into caller O; Even alone writes present LSE. Padding still reads TMEM but writes no caller output. Initial destination contents are not read. done means stores issued and register sources consumed, without a device-wide or host completion wait.

## t_ef3f8e6deda2b697 | pipeline.verb | 214 chars | 4 declarations

tcg_d128_segmented_cp_direct_narrow:559 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.signal; tcg_d128_segmented_cp_direct_wide:576 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.signal; tcg_d128_segmented_tma_direct_narrow:555 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.signal; tcg_d128_segmented_tma_direct_wide:574 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.signal

All threads signal split launch permission after the wait but before scan arithmetic. This does not publish snapshots or tails. Out-of-range rows also signal; their later arithmetic and stores have no data effects.

## t_f032388ade633537 | pipeline.verb | 324 chars | 4 declarations

tcg_d128_fscan_cp_xpose_narrow:435 design.tile_pipeline.phase.group_max.role.Main.info.verb.reduce; tcg_d128_fscan_cp_xpose_wide:452 design.tile_pipeline.phase.group_max.role.Main.info.verb.reduce; tcg_d128_fscan_tma_xpose_wide:448 design.tile_pipeline.phase.group_max.role.Main.info.verb.reduce; tcg_d128_segmented_cp_xpose_wide:522 design.tile_pipeline.phase.group_max.role.Main.info.verb.reduce

All warps reduce masked score chunks through MaxChunk or use the native unmasked read-and-max path, each with its TMEM-load wait. Only valid rows issue GroupMax stores. A CTA barrier joins every score reader before the next QK overwrites DotTile. No whole-group score register array, PV, or cross-warpgroup reduction exists.

## t_f0bbb5a5701e40f3 | pipeline.verb | 309 chars | 1 declarations

tcg_d256_flat_tma_wide:679 design.tile_pipeline.phase.combine.role.Main.info.verb.numerator

Walk complete eight-split batches within the group's interval. Per batch, load and accumulate channel d0, then d0+128; NumeratorBatch dies before reuse. Process remaining splits one at a time for both channels, without tail zero additions or out-of-range loads. Empty groups retain zero without reading PartA.

## t_f1c279e813997b53 | pipeline.verb | 272 chars | 1 declarations

tc_split_segmented_wide:796 design.tile_pipeline.phase.combine.role.Main.info.verb.denominator

Each thread loads PartialB at sLo(sg)+d0, or zero outside the group's interval, into F. Five warp shuffle-down stages implement the combine step's tree. Lane zero of every warp stores its GroupB entry. All numerator and denominator partials are written before publication.

## t_f1f94f09ba1a248a | pipeline.verb | 133 chars | 2 declarations

tc_split_flat_narrow:648 design.tile_pipeline.phase.combine.role.Main.info.verb.stage; tc_split_segmented_narrow:755 design.tile_pipeline.phase.combine.role.Main.info.verb.stage

One thread per split loads PartialB into shared Denominator. There is no barrier here; unused slots are neither initialized nor read.

## t_f25c3a7d923a3bb9 | pipeline.verb | 302 chars | 3 declarations

tcg_d256_segmented_cp_narrow:543 design.tile_pipeline.phase.scan.role.Main.info.verb.replay; tcg_d256_segmented_tma_narrow:558 design.tile_pipeline.phase.scan.role.Main.info.verb.replay; tcg_d256_segmented_tma_wide:543 design.tile_pipeline.phase.scan.role.Main.info.verb.replay

Apply scan's prefix fill and suffix replay. The fill or the suffix may be empty without deleting this action; together they write every state in the chunk. Only the suffix reads GroupMax. Each valid row retains ScanState for its next chunk; registers have no other thread's readers and need no barrier.

## t_f2cbe6d23ec13d5d | pipeline.verb | 552 chars | 1 declarations

tcg_d128_segmented_tma_direct_wide:760 design.tile_pipeline.phase.combine.role.Main.info.verb.partials

Group z consists of four consecutive warps. Each of its 128 threads owns one channel and initializes GroupAccum to zero. Load full numerator batches of eight into NumeratorBatch, consume each before register reuse, then add only existing scalar tail entries and store GroupA. Each thread also loads one PartB in its group's interval, or zero past its end. Warp reductions implement combine's tree; one lane per warp stores GroupB. Empty intervals still store zeros. done covers local computation and shared stores; there is no intermediate CTA barrier.

## t_f315be3f5442453d | pipeline.verb | 333 chars | 2 declarations

tc_split_flat_narrow:512 design.tile_pipeline.phase.scan.role.Main.info.verb.entry; tc_split_flat_wide:529 design.tile_pipeline.phase.scan.role.Main.info.verb.entry

All threads execute the grid-dependency wait and then the scan-to-split launch-completion trigger before the row>=R return. The ordinary maximum predecessor has no early trigger, so delivery requires its completion. ScanState is not ready at this early scan trigger; the ordinary split launch in this node waits for scan.kernel_done.

## t_f4be70a5074a9b62 | pipeline.verb | 456 chars | 1 declarations

tcg_d128_flat_cp_direct_wide:681 design.tile_pipeline.phase.combine.role.Main.info.verb.write

Group zero independently sums GroupB for each output thread, sums GroupA for its channel and writes the normalized BF16 value. Its first thread reads MTail and writes LSE when present. Other groups have no data effects in this action. Shared partials remain unchanged through all reads; there is no subsequent worktile or local reuse. Initial destinations are never read. Kernel completion supplies output visibility; no launch-permission signal is issued.

## t_f51bcfb43c9103bf | pipeline.verb | 566 chars | 1 declarations

fused_d128_cp:362 design.tile_pipeline.phase.main.role.Main.info.verb.qk

Enqueue the scores step's four native block-scaled products for half z into ScoreHalf, then commit one QK completion. done here means enqueue only. Each half's first contraction overwrites. For z=1, a before-thread-sync fence and CTA barrier first prove that all warps have completed half zero's readout before the same 64 columns are overwritten; the issuer uses the matching after-thread-sync fence. The next group's first half is behind the previous probability-publication barrier and this group's land:Inputs barrier, so its old score readers have also retired.

## t_f5628eb9b6b31291 | pipeline.verb | 233 chars | 6 declarations

tcg_d256_flat_cp_narrow:678 design.tile_pipeline.phase.combine.role.Main.info.verb.write; tcg_d256_flat_cp_wide:692 design.tile_pipeline.phase.combine.role.Main.info.verb.write; tcg_d256_flat_tma_narrow:673 design.tile_pipeline.phase.combine.role.Main.info.verb.write; tcg_d256_segmented_cp_narrow:746 design.tile_pipeline.phase.combine.role.Main.info.verb.write; tcg_d256_segmented_tma_narrow:761 design.tile_pipeline.phase.combine.role.Main.info.verb.write; tcg_d256_segmented_tma_wide:746 design.tile_pipeline.phase.combine.role.Main.info.verb.write

Each thread converts/writes its two O channels. Thread zero alone reads Mtail and writes LSE when present. No additional wait or trigger follows; kernel completion gives caller-stream visibility after ordinary output stores complete.

## t_f5840b724bb67bfc | pipeline.verb | 179 chars | 4 declarations

tcg_d256_flat_cp_narrow:466 design.tile_pipeline.phase.scan.role.Main.info.verb.batch64; tcg_d256_flat_cp_wide:480 design.tile_pipeline.phase.scan.role.Main.info.verb.batch64; tcg_d256_flat_tma_narrow:464 design.tile_pipeline.phase.scan.role.Main.info.verb.batch64; tcg_d256_flat_tma_wide:478 design.tile_pipeline.phase.scan.role.Main.info.verb.batch64

Each valid thread loads 64 groups starting at 64*j into Batch before replaying them in order, writing each Mscan entry. Storage is [group][row]; the batch dies before local reuse.

## t_f59c18af6ccbffd8 | pipeline.verb | 196 chars | 20 declarations

tcg_d128_flat_cp_direct_narrow:604 design.tile_pipeline.phase.split.role.Main.info.verb.pv; tcg_d128_flat_cp_direct_wide:618 design.tile_pipeline.phase.split.role.Main.info.verb.pv; tcg_d128_flat_cp_xpose_narrow:613 design.tile_pipeline.phase.split.role.Main.info.verb.pv; tcg_d128_flat_cp_xpose_wide:596 design.tile_pipeline.phase.split.role.Main.info.verb.pv; tcg_d128_flat_tma_direct_narrow:597 design.tile_pipeline.phase.split.role.Main.info.verb.pv; tcg_d128_flat_tma_direct_wide:611 design.tile_pipeline.phase.split.role.Main.info.verb.pv; tcg_d128_flat_tma_xpose_narrow:600 design.tile_pipeline.phase.split.role.Main.info.verb.pv; tcg_d128_flat_tma_xpose_wide:619 design.tile_pipeline.phase.split.role.Main.info.verb.pv; tcg_d128_fscan_cp_direct_narrow:567 design.tile_pipeline.phase.split.role.Main.info.verb.pv; tcg_d128_fscan_cp_direct_wide:584 design.tile_pipeline.phase.split.role.Main.info.verb.pv; tcg_d128_fscan_tma_direct_narrow:560 design.tile_pipeline.phase.split.role.Main.info.verb.pv; tcg_d128_fscan_tma_direct_wide:578 design.tile_pipeline.phase.split.role.Main.info.verb.pv; tcg_d128_fscan_tma_xpose_narrow:566 design.tile_pipeline.phase.split.role.Main.info.verb.pv; tcg_d128_segmented_cp_direct_narrow:695 design.tile_pipeline.phase.split.role.Main.info.verb.pv; tcg_d128_segmented_cp_direct_wide:712 design.tile_pipeline.phase.split.role.Main.info.verb.pv; tcg_d128_segmented_cp_xpose_narrow:672 design.tile_pipeline.phase.split.role.Main.info.verb.pv; tcg_d128_segmented_tma_direct_narrow:687 design.tile_pipeline.phase.split.role.Main.info.verb.pv; tcg_d128_segmented_tma_direct_wide:706 design.tile_pipeline.phase.split.role.Main.info.verb.pv; tcg_d128_segmented_tma_xpose_narrow:688 design.tile_pipeline.phase.split.role.Main.info.verb.pv; tcg_d128_segmented_tma_xpose_wide:672 design.tile_pipeline.phase.split.role.Main.info.verb.pv

Enqueue four native PV contractions into GroupPV, overwriting on the first contraction of every group. Commit, wait, toggle parity and fence. Value, ValueScale and Prob8 operand reads are retired.

## t_f5eac283b1a79a02 | pipeline.verb | 404 chars | 2 declarations

tcg_d128_flat_cp_direct_narrow:593 design.tile_pipeline.phase.split.role.Main.info.verb.stage; tcg_d128_flat_cp_direct_wide:607 design.tile_pipeline.phase.split.role.Main.info.verb.stage

Each thread waits until all but its newest copy group complete, proving current K/V and both atoms even after the last empty commit. Fence shared async access and CTA-sync, write current KeyScale and ValueScale, wait for TMEM stores, fence and CTA-sync. Prior PV was already waited, protecting the scale/operand slots. These joins retire the previous GroupPV readers before QK overwrites the same region.

## t_f5f82d91970bf274 | pipeline.verb | 397 chars | 18 declarations

tcg_d128_flat_cp_xpose_narrow:632 design.tile_pipeline.phase.split.role.Main.info.verb.finish; tcg_d128_flat_cp_xpose_wide:615 design.tile_pipeline.phase.split.role.Main.info.verb.finish; tcg_d128_flat_tma_direct_narrow:604 design.tile_pipeline.phase.split.role.Main.info.verb.finish; tcg_d128_flat_tma_direct_wide:618 design.tile_pipeline.phase.split.role.Main.info.verb.finish; tcg_d128_flat_tma_xpose_narrow:618 design.tile_pipeline.phase.split.role.Main.info.verb.finish; tcg_d128_flat_tma_xpose_wide:638 design.tile_pipeline.phase.split.role.Main.info.verb.finish; tcg_d128_fscan_cp_direct_narrow:576 design.tile_pipeline.phase.split.role.Main.info.verb.finish; tcg_d128_fscan_cp_direct_wide:593 design.tile_pipeline.phase.split.role.Main.info.verb.finish; tcg_d128_fscan_tma_direct_narrow:569 design.tile_pipeline.phase.split.role.Main.info.verb.finish; tcg_d128_fscan_tma_direct_wide:587 design.tile_pipeline.phase.split.role.Main.info.verb.finish; tcg_d128_fscan_tma_xpose_narrow:587 design.tile_pipeline.phase.split.role.Main.info.verb.finish; tcg_d128_segmented_cp_direct_narrow:702 design.tile_pipeline.phase.split.role.Main.info.verb.finish; tcg_d128_segmented_cp_direct_wide:719 design.tile_pipeline.phase.split.role.Main.info.verb.finish; tcg_d128_segmented_cp_xpose_narrow:691 design.tile_pipeline.phase.split.role.Main.info.verb.finish; tcg_d128_segmented_tma_direct_narrow:694 design.tile_pipeline.phase.split.role.Main.info.verb.finish; tcg_d128_segmented_tma_direct_wide:713 design.tile_pipeline.phase.split.role.Main.info.verb.finish; tcg_d128_segmented_tma_xpose_narrow:707 design.tile_pipeline.phase.split.role.Main.info.verb.finish; tcg_d128_segmented_tma_xpose_wide:691 design.tile_pipeline.phase.split.role.Main.info.verb.finish

Fence and CTA-sync join final TMEM users; first warp deallocates. All requested transfers and products were already waited. Signal PDL completion after partial stores and teardown. No CTA receives another tile; later kernels have fresh local resources. Global intermediates survive all declared readers, and caller-stream completion orders subsequent invocations without retaining numerical state.

## t_f6fb3315bc2317ce | pipeline.verb | 767 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:705 design.tile_pipeline.phase.split.role.Main.info.verb.replay.meaning

After its dependency wait, each thread initializes DenState to zero and MaxState to -inf and loads the entrance snapshot at (cLo-1)*R+rowG whenever cLo>0, even for an empty split. CurrentMax at cLo is loaded only for Ng>0. rowG is the valid row r or that head's first real row for padding. done covers all replay work; threads can continue with their own initialized registers before other warps finish. The subsequent stage or stage_partials CTA join covers completion. For Ng>0 each thread then commits an empty copy group at the original in-body prime; Ng=0 has no commit or extra join. The entrance index is not clamped to T-1 or guarded by any_compute; an out-of-range empty-split read is not promised initialized or safe. Its value is discarded on empty splits.

## t_f76af1e869250160 | pipeline.verb | 211 chars | 1 declarations

tcg_d128_segmented_tma_direct_wide:765 design.tile_pipeline.phase.combine.role.Main.info.verb.join

A CTA barrier publishes MergePartials, the pair of GroupA and GroupB arrays, to group zero. Both arrays remain unchanged through all final readers. No shared weight array is consumed on this non-fused scan path.

## t_f7864ce348dcf56c | pipeline.verb | 358 chars | 1 declarations

tcg_d128_fscan_cp_xpose_narrow:657 design.tile_pipeline.phase.combine.role.Main.info.verb.weights

Every thread reads the row's MTail; one thread per s<Ns loads PartB into DenParts and PartM to compute Weight[s] once using combine's guarded expf. Ns<=128, so one round covers all splits. No unused shared entry is initialized or read. done means each lane's own stores are issued; publication occurs at prime's CTA join. Each thread owns one output channel.

## t_f7e456399b595d8c | pipeline.verb | 333 chars | 2 declarations

tcg_d128_fscan_cp_direct_narrow:537 design.tile_pipeline.phase.split.role.Main.info.verb.setup; tcg_d128_fscan_cp_direct_wide:554 design.tile_pipeline.phase.split.role.Main.info.verb.setup

Establish coordinates and PSK valid/live-warp predicates. Thread zero initializes QK/PV completion barriers at parity zero; prepare's CTA join publishes initialization. All four warps stage payloads, the first also copies scale atoms, and every thread commits/waits its own copy groups. Thread zero issues MMA. q0>=Sq is unreachable.

## t_f8809bc33b205cc6 | pipeline.verb | 487 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:528 design.tile_pipeline.phase.group_max.role.Main.info.verb.finish

All enqueued K payload/atom copies and QK products have been waited; only empty copy groups may remain. A before-thread-sync fence and CTA barrier join final TMEM users; the first warp deallocates 256 columns. Each CTA signals PDL launch completion after its global stores and local teardown; launch_ready applies when the whole grid satisfies the protocol. Actual global result visibility is established by kernel completion and the successor dependency wait, not by enqueueing a store.

## t_f893935f19a90202 | pipeline.verb | 403 chars | 1 declarations

fused_d128_group:455 design.tile_pipeline.phase.main.role.Even.info.verb.rescale

Except at g=cLo, wait for the other role's PV(g-1), advance its completion counter and apply the after fence. Only then may a warp with any alpha unequal to one read, rescale and store all 128 NumState channels in four 32-channel chunks; wait for those stores. At the first group neither this PV wait nor a NumState read is performed. done includes matrix completion, not just the predecessor's enqueue.

## t_fa7de64ea285bbcd | pipeline.verb | 128 chars | 20 declarations

tcg_d128_flat_cp_direct_narrow:598 design.tile_pipeline.phase.split.role.Main.info.verb.qk; tcg_d128_flat_cp_direct_wide:612 design.tile_pipeline.phase.split.role.Main.info.verb.qk; tcg_d128_flat_cp_xpose_narrow:608 design.tile_pipeline.phase.split.role.Main.info.verb.qk; tcg_d128_flat_cp_xpose_wide:591 design.tile_pipeline.phase.split.role.Main.info.verb.qk; tcg_d128_flat_tma_direct_narrow:591 design.tile_pipeline.phase.split.role.Main.info.verb.qk; tcg_d128_flat_tma_direct_wide:605 design.tile_pipeline.phase.split.role.Main.info.verb.qk; tcg_d128_flat_tma_xpose_narrow:595 design.tile_pipeline.phase.split.role.Main.info.verb.qk; tcg_d128_flat_tma_xpose_wide:614 design.tile_pipeline.phase.split.role.Main.info.verb.qk; tcg_d128_fscan_cp_direct_narrow:560 design.tile_pipeline.phase.split.role.Main.info.verb.qk; tcg_d128_fscan_cp_direct_wide:577 design.tile_pipeline.phase.split.role.Main.info.verb.qk; tcg_d128_fscan_tma_direct_narrow:553 design.tile_pipeline.phase.split.role.Main.info.verb.qk; tcg_d128_fscan_tma_direct_wide:571 design.tile_pipeline.phase.split.role.Main.info.verb.qk; tcg_d128_fscan_tma_xpose_narrow:560 design.tile_pipeline.phase.split.role.Main.info.verb.qk; tcg_d128_segmented_cp_direct_narrow:689 design.tile_pipeline.phase.split.role.Main.info.verb.qk; tcg_d128_segmented_cp_direct_wide:706 design.tile_pipeline.phase.split.role.Main.info.verb.qk; tcg_d128_segmented_cp_xpose_narrow:667 design.tile_pipeline.phase.split.role.Main.info.verb.qk; tcg_d128_segmented_tma_direct_narrow:681 design.tile_pipeline.phase.split.role.Main.info.verb.qk; tcg_d128_segmented_tma_direct_wide:700 design.tile_pipeline.phase.split.role.Main.info.verb.qk; tcg_d128_segmented_tma_xpose_narrow:683 design.tile_pipeline.phase.split.role.Main.info.verb.qk; tcg_d128_segmented_tma_xpose_wide:667 design.tile_pipeline.phase.split.role.Main.info.verb.qk

Enqueue four QK contractions with first-contraction overwrite, commit, wait for completion and fence. No maximum readout occurs.

## t_fb1dfd6e2f689cac | pipeline.verb | 296 chars | 2 declarations

tcg_d128_fscan_cp_xpose_narrow:412 design.tile_pipeline.phase.group_max.role.Main.info.verb.prime; tcg_d128_fscan_cp_xpose_wide:429 design.tile_pipeline.phase.group_max.role.Main.info.verb.prime

For Ng>0 cooperatively enqueue K(cLo) and its 512-byte scale atom into slot zero, then each thread commits its copies as one group. Payload copies zero-fill keys beyond Skv; atom copies read the allocated scale bytes. For Ng=0 issue neither copies nor a commit. done denotes enqueue, not arrival.

## t_fb93cfbff8ec8598 | pipeline.verb | 529 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:505 design.tile_pipeline.phase.group_max.role.Main.info.verb.start

For Ng>0 each thread first commits an empty copy group, preserving the in-body prime after the hoisted prime. For Long then enqueue K(cLo+1) and its atom and commit another group. Wait until at most one copy group remains pending per thread, proving slot zero's payload and atom; fence shared async access and CTA-sync. Publish the first KeyScale, wait for its TMEM stores, fence and CTA-sync, then enqueue QK(cLo) and commit its completion phase. For non-Long there is no copy wait, CTA join or QK here; Ng=0 also has no commit.

## t_fbcac786b51c7154 | pipeline.verb | 808 chars | 5 declarations

tcg_d256_flat_tma_narrow:577 design.tile_pipeline.phase.split.role.Main.info.verb.ahead; tcg_d256_flat_tma_wide:591 design.tile_pipeline.phase.split.role.Main.info.verb.ahead; tcg_d256_segmented_cp_narrow:650 design.tile_pipeline.phase.split.role.Main.info.verb.ahead; tcg_d256_segmented_tma_narrow:665 design.tile_pipeline.phase.split.role.Main.info.verb.ahead; tcg_d256_segmented_tma_wide:650 design.tile_pipeline.phase.split.role.Main.info.verb.ahead

Prefetch Mscan[g+1] when present, otherwise ReplayNext=-inf. Enqueue K/V(g+1) into the other two-slot stage and atoms(g+3) into slot (g+3-l)%3 when each group exists. Commit even if empty and wait_group 1. Copy atoms(g+1) to the alternate TMEM scale parity with warps 0..3 and wait for stores. For g+1<=e, prime or an earlier iteration completed and CTA-published these bytes. Prior QK/PV retirement frees old tile/scale slots; prior head synchronization orders atom readers before refill. At g=e the scale copy still executes on stale or, for C=1 or C=2, uninitialized atom bytes, writing an in-bounds unused TMEM slot. Wait current combined TMA with parity floor((g-l)/2)%2, fence the shared proxy and before thread-sync, then CTA-sync to publish current K/V and ahead scales. No extra tail initialization.

## t_fc20628b6005ad14 | pipeline.verb | 452 chars | 1 declarations

root:627 design.tile_pipeline.phase.main.role.Main.info.verb.write

Read NumState in 32-channel chunks, each with its TMEM-load wait. Valid rows issue scaled BF16 RNE stores directly to caller O and, when present, one FP32 LSE store using value_output. Initial destination bytes are never read. Padding still participates in computation/readout but issues no output store. done means these ordinary stores have been issued and their register sources consumed; it does not introduce a device-wide or host completion wait.

## t_fc29b074c4c7bd15 | pipeline.verb | 168 chars | 8 declarations

tcg_d128_flat_cp_direct_narrow:479 design.tile_pipeline.phase.flat_scan.role.Main.info.verb.dependency; tcg_d128_flat_cp_direct_wide:493 design.tile_pipeline.phase.flat_scan.role.Main.info.verb.dependency; tcg_d128_flat_cp_xpose_narrow:488 design.tile_pipeline.phase.flat_scan.role.Main.info.verb.dependency; tcg_d128_flat_cp_xpose_wide:471 design.tile_pipeline.phase.flat_scan.role.Main.info.verb.dependency; tcg_d128_flat_tma_direct_narrow:475 design.tile_pipeline.phase.flat_scan.role.Main.info.verb.dependency; tcg_d128_flat_tma_direct_wide:489 design.tile_pipeline.phase.flat_scan.role.Main.info.verb.dependency; tcg_d128_flat_tma_xpose_narrow:479 design.tile_pipeline.phase.flat_scan.role.Main.info.verb.dependency; tcg_d128_flat_tma_xpose_wide:497 design.tile_pipeline.phase.flat_scan.role.Main.info.verb.dependency

Execute the PDL grid-dependency wait before any early return or GroupMax load. done guarantees the full predecessor's completion and visibility for the consumed region.

## t_fcd6cc1b45dfb5a7 | pipeline.verb | 502 chars | 1 declarations

tcg_d128_flat_tma_xpose_wide:702 design.tile_pipeline.phase.combine.role.Main.info.verb.write

Group zero alone reads MergeA and MergeB. Each of its threads computes the same TotalB in group/segment order and its channel's TotalA in group order, then writes that BF16 O channel. Thread zero also reads MTail and writes LSE when present. Other groups have no further shared accesses or output work after the publication barrier. Shared partials stay unchanged through all readers. Initial caller destinations are not read; kernel completion supplies visibility. No next tile or local reuse follows.

## t_fd6e461bf9c01b83 | pipeline.verb | 332 chars | 4 declarations

tcg_d128_segmented_cp_direct_narrow:567 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.replay; tcg_d128_segmented_cp_direct_wide:584 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.replay; tcg_d128_segmented_tma_direct_narrow:563 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.replay; tcg_d128_segmented_tma_direct_wide:582 design.tile_pipeline.phase.segmented_scan.role.Main.info.verb.replay

Fill the prefix before ReplayStart with unchanged ScanState, without GroupMax loads. Replay the entire remaining suffix as in first. Empty suffixes issue no walk loads or updates. Consume each ReplayBatch before reuse and ReplayStart before the next test overwrites it. Each row progresses independently, without CTA scan reduction.

## t_fdf52bdfa259c74a | pipeline.verb | 230 chars | 3 declarations

tcg_d256_segmented_cp_narrow:549 design.tile_pipeline.phase.scan.role.Main.info.verb.finish; tcg_d256_segmented_tma_narrow:564 design.tile_pipeline.phase.scan.role.Main.info.verb.finish; tcg_d256_segmented_tma_wide:549 design.tile_pipeline.phase.scan.role.Main.info.verb.finish

Valid threads write Mtail and discard ScanState. No CTA barrier or exit trigger follows. Kernel completion, observed by split's late data wait, publishes every Mscan entry and Mtail; no local resource identity crosses this launch.

## t_03ab04b09d6ff616 | realization.choice | 208 chars | 1 declarations

tcg_d256_segmented_cp_wide:921 realization.require[1].choice

Use native block_scale.scale_vec::1X tcgen05 descriptors with SWIZZLE_128B operands, QK K-major Q/K, and PV TMEM probabilities with MN-major V. Preserve 1024-byte operand alignment and replicated scale atoms.

## t_04441df2b1b8d9fd | realization.choice | 190 chars | 4 declarations

tcg_d256_flat_cp_narrow:709 realization.require[2].choice; tcg_d256_flat_cp_wide:723 realization.require[2].choice; tcg_d256_flat_tma_narrow:702 realization.require[2].choice; tcg_d256_flat_tma_wide:721 realization.require[2].choice

Use programmatic stream serialization on scan, split and combine, cudaTriggerProgrammaticLaunchCompletion at the declared trigger points, and cudaGridDependencySynchronize at the data waits.

## t_0d12b95b511b691d | realization.choice | 527 chars | 1 declarations

tcg_d128_fscan_tma_xpose_wide:705 realization.require[0].choice

Bind matrix completion to tcg_commit plus parity mbar_wait, TMEM readout to tcg_ld32/tcg_ld32_redmax with their wait::ld, stores to wait::st, and TMEM publication to the before/after thread-sync fences at the stated CTA joins. TMA payloads and bulk atoms use the same transaction-byte completion for each indicated slot. Preserve SWIZZLE_128B operands, 1024-byte shared alignment and the replicated scale representation expected by native block_scale.scale_vec::1X descriptors; QK is K-major and PV uses TMEM A with MN-major V.

## t_0fb46c6fe0d1fd6f | realization.choice | 375 chars | 2 declarations

tcg_d256_flat_cp_narrow:694 realization.require[0].choice; tcg_d256_flat_cp_wide:708 realization.require[0].choice

Bind matrix completion to tcg_commit plus parity mbar_wait; TMEM reads use the wrappers' wait::ld and stores use wait::st. Retain before/after thread-sync fences. Bind payload copies to 16-byte cp.async.cg.shared.global with source size zero for padded keys, and payload/atom completion to the declared per-thread commit_group/wait_group sequence followed by CTA publication.

## t_1167bea6a95cf317 | realization.choice | 587 chars | 1 declarations

fused_d256_cp:477 realization.prefer[0].choice

launch_tcg requests TcgSmem<256,PASS_FUSED,2>::bytes=222208 and caches tmem_safe_smem's result. This includes Key's unused third slot. Beyond the listed shared resources it reserves 16384 unused probability bytes, 1024 unused Q-scale bytes and 1024 alignment bytes. The helper publishes the dynamic-memory attribute and queries occupancy against blocks*512<=512. This request already exceeds its 220000-byte growth limit, so it is returned without growth, including on API failure after retrying any unpublished attribute and clearing the error. The request is not a residency guarantee.

## t_2bd8f46bf9d3b26a | realization.choice | 334 chars | 2 declarations

fused_d128_cp:453 realization.require[1].choice; root:675 realization.require[1].choice

Preserve the source's compatible SWIZZLE_128B shared operands and tcgen05 block_scale.scale_vec::1X descriptors: QK uses K-major Q/K, PV uses the packed TMEM A operand and MN-major V. The shared operand window is aligned to 1024 bytes, and each atom is placed in the replicated TMEM scale representation expected by these descriptors.

## t_4182edda1a466482 | realization.choice | 409 chars | 1 declarations

fused_d256_cp:455 realization.require[0].choice

Bind completion and visibility to the fixed fmha_tcg_ptx.cuh wrappers: tcg_commit plus parity mbar_wait for QK/PV chains, tcg_ld32's own wait::ld, wait::st for TMEM stores and the before/after thread-sync fences at the stated synchronization points. Payload and atom staging use fmha_common.cuh's predicated 16-byte cp.async.cg copies, commit_group, wait_group 0 for prime and wait_group 1 in the steady loop.

## t_4915b10a299b0f07 | realization.choice | 550 chars | 2 declarations

tcg_d128_fscan_cp_direct_narrow:644 realization.require[0].choice; tcg_d128_fscan_cp_direct_wide:667 realization.require[0].choice

Bind matrix completion to tcg_commit plus parity mbar_wait; TMEM readout to tcg_ld32/tcg_ld32_redmax with their wait::ld; stores to wait::st; and publication to before/after-thread-sync fences at the stated CTA joins. Bind payload/atom copies to cp.async.cg.shared.global, commit_group and wait_group 1, preserving empty commits and per-thread completion scope. Preserve SWIZZLE_128B shared operands, 1024-byte shared alignment and replicated E8M0 atoms for native block_scale.scale_vec::1X descriptors: QK is K-major; PV uses TMEM A with MN-major V.

## t_4ef3d2cc7c242584 | realization.choice | 648 chars | 2 declarations

tcg_d128_flat_cp_direct_narrow:676 realization.require[0].choice; tcg_d128_flat_cp_direct_wide:693 realization.require[0].choice

Bind matrix completion to tcg_commit plus parity mbar_wait; TMEM readout to tcg_ld32/tcg_ld32_redmax with their wait::ld; stores to wait::st; and publication to before/after-thread-sync fences at the stated CTA joins. Bind payload/atom copies to cp.async.cg.shared.global with a 16-byte destination and source size 16 or zero, followed by cp.async.commit_group, including the empty commits, and cp.async.wait_group 1 with the stated shared fence and CTA join. Preserve SWIZZLE_128B shared operands, 1024-byte shared alignment and replicated E8M0 atoms for native block_scale.scale_vec::1X descriptors: QK is K-major; PV uses TMEM A with MN-major V.

## t_504a2cdf181f9766 | realization.choice | 263 chars | 1 declarations

fused_d128_group:557 realization.require[0].choice

Preserve SWIZZLE_128B shared operands and compatible tcgen05 block_scale.scale_vec::1X descriptors: QK consumes K-major Q/K, PV uses packed TMEM A and MN-major V. Align the shared operand window to 1024 bytes and replicate E8M0 atoms as those descriptors require.

## t_5646f4020bb0b42f | realization.choice | 281 chars | 4 declarations

tc_split_flat_narrow:672 realization.require[0].choice; tc_split_flat_wide:688 realization.require[0].choice; tc_split_segmented_narrow:779 realization.require[0].choice; tc_split_segmented_wide:814 realization.require[0].choice

Bind copy enqueue/land to cp_async16, cp_async_commit and cp_async_wait<1>, with the declared CTA barriers and final empty commits. Compatible fp8_tile_offset/bf16_tile_offset layouts and ldmatrix loads feed mma_fp8_16832 and mma_bf16_16816 for the applicable synchronous products.

## t_578e66cba4a8b342 | realization.choice | 493 chars | 6 declarations

tcg_d128_flat_tma_direct_narrow:669 realization.require[0].choice; tcg_d128_flat_tma_direct_wide:688 realization.require[0].choice; tcg_d128_fscan_tma_direct_narrow:637 realization.require[0].choice; tcg_d128_fscan_tma_direct_wide:660 realization.require[0].choice; tcg_d128_segmented_tma_direct_narrow:759 realization.require[0].choice; tcg_d128_segmented_tma_direct_wide:781 realization.require[0].choice

Bind matrix completion to tcg_commit plus parity mbar_wait; TMEM readout to tcg_ld32/tcg_ld32_redmax with their wait::ld; stores to wait::st; and publication to before/after-thread-sync fences at the stated CTA joins. TMA payloads and bulk atoms share transaction-byte completion for each indicated slot. Preserve SWIZZLE_128B shared operands, 1024-byte shared alignment and replicated E8M0 atoms for native block_scale.scale_vec::1X descriptors: QK is K-major; PV uses TMEM A with MN-major V.

## t_67f4c38b92f82678 | realization.choice | 275 chars | 2 declarations

tc_split_segmented_narrow:785 realization.require[1].choice; tc_split_segmented_wide:820 realization.require[1].choice

Keep launch_tc ordinary for both MMA passes. Use pdl_launch with cudaLaunchAttributeProgrammaticStreamSerialization for segmax/scan/combine, and bind waits and triggers to fmha_pdl_wait/fmha_pdl_trigger/fmha_pdl_scan_trigger, including the source compile-time no-op branches.

## t_6c08afb7cd72b045 | realization.choice | 572 chars | 4 declarations

tcg_d128_fscan_cp_xpose_narrow:699 realization.prefer[0].choice; tcg_d128_fscan_cp_xpose_wide:719 realization.prefer[0].choice; tcg_d128_fscan_tma_xpose_wide:720 realization.prefer[0].choice; tcg_d128_segmented_cp_xpose_wide:843 realization.prefer[0].choice

launch_tcg begins with dynamic shared requests of 69632 bytes for group-max and 102912 for split. In addition to live buffers these include 512 bytes for shared Q scales, 1024 for exchange and 1024 for alignment; split also reserves an unused 16384-byte shared P tile. tmem_safe_smem queries occupancy and increases the request by 4096 while blocks*256>512, for at most 96 attempts, stopping on API error or a still-failing request >=220000. It retries an unpublished attribute, clears the error and returns/caches the last request; it does not guarantee actual residency.

## t_6e3cf78b1ff75011 | realization.choice | 279 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:835 realization.require[1].choice

Use the fixed PDL wrappers: cudaLaunchKernelEx/ExC with programmatic stream serialization for the four successors, cudaTriggerProgrammaticLaunchCompletion at the stated signal points and cudaGridDependencySynchronize at dependency waits. group_max itself uses an ordinary launch.

## t_754ab6ef6409f73c | realization.choice | 333 chars | 1 declarations

fused_d128_key:465 realization.require[0].choice

Bind the stated completion/visibility guarantees to the fixed wrappers: cp.async commit_group/wait_group 1 and fence.proxy.async.shared::cta for shared staging; tcg_commit plus parity mbar_wait for QK/PV; tcg_ld32's own wait::ld, wait::st for TMEM stores, and before/after thread-sync fences around the stated synchronization points.

## t_78ef06fcdaf861e5 | realization.choice | 636 chars | 1 declarations

fused_d128_key:487 realization.prefer[0].choice

launch_tcg requests TcgSmem<128,PASS_FUSED,2>::bytes=170496 before tmem_safe_smem adjusts the request. Besides the live shared resources, this includes an unused 16384-byte probability reservation, 512-byte Q-scale reservation and 1024-byte alignment allowance. The helper publishes the dynamic-memory attribute and queries occupancy, increasing by 4096 bytes while blocks*512>512. It stops on an API error, after 96 attempts, or when a request at least 220000 bytes still fails that bound; it retries any unpublished attribute, clears the error and returns the last request. launch_tcg caches that value, without a residency guarantee.

## t_7c793fda11f5a2ed | realization.choice | 564 chars | 1 declarations

fused_d128_cp:438 realization.require[0].choice

Bind completion and visibility to the fixed fmha_tcg_ptx.cuh wrappers: tcg_commit plus parity mbar_wait for the issued QK/PV chains, tcg_ld32's own wait::ld, wait::st for TMEM stores, and the before/after thread-sync fences around the stated CTA synchronization points. Bind K/V and atom staging to fmha_common.cuh's cp_async16, cp_async_commit and cp_async_wait<1>, with zero source size for padded payloads and the shared async-proxy fence at land. These mechanisms retain the distinct enqueue, readable and reusable conditions specified by the action contracts.

## t_8031b24a1d597204 | realization.choice | 203 chars | 3 declarations

tcg_d256_segmented_cp_narrow:775 realization.require[2].choice; tcg_d256_segmented_tma_narrow:790 realization.require[2].choice; tcg_d256_segmented_tma_wide:775 realization.require[2].choice

Use programmatic stream serialization on segment_max, scan, split and combine, cudaTriggerProgrammaticLaunchCompletion at the declared trigger points, and cudaGridDependencySynchronize at the data waits.

## t_82724b745a8cbe72 | realization.choice | 476 chars | 7 declarations

tcg_d256_flat_cp_narrow:716 realization.prefer[0].choice; tcg_d256_flat_cp_wide:730 realization.prefer[0].choice; tcg_d256_flat_tma_narrow:709 realization.prefer[0].choice; tcg_d256_flat_tma_wide:728 realization.prefer[0].choice; tcg_d256_segmented_cp_narrow:782 realization.prefer[0].choice; tcg_d256_segmented_tma_narrow:797 realization.prefer[0].choice; tcg_d256_segmented_tma_wide:782 realization.prefer[0].choice

Initial shared requests are 103424 bytes for group_max and 189440 for split. Cache tmem_safe_smem per instantiation: publish the attribute, query occupancy, and grow by 4096 until blocks*TMEM_columns<=512, the growth limit of 220000 bytes, or 96 attempts. On API failure stop, retry an unpublished attribute, clear the error and return the current request. Reserved unused Q-scale/probability space and alignment slack remain in the request; this is not a residency guarantee.

## t_85f7d02eeb2f4a15 | realization.choice | 333 chars | 1 declarations

tc_fused:387 realization.require[0].choice

Bind enqueue/land to fmha_common.cuh's cp_async16, cp_async_commit and cp_async_wait<1>, with __syncthreads at the stated publication and reuse boundaries. Preserve the final empty commit. Use compatible fp8_tile_offset and bf16_tile_offset layouts with ldmatrix loads, mma_fp8_16832 and mma_bf16_16816 for register-result QK and PV.

## t_8606fadc09f1dd79 | realization.choice | 545 chars | 1 declarations

fused_d256_tma:497 realization.prefer[0].choice

launch_tcg requests TcgSmem<256,PASS_FUSED,2>::bytes=222208 and caches tmem_safe_smem's result. Beyond live shared resources, this reserves 16384 unused probability bytes, 1024 unused Q-scale bytes and 1024 alignment bytes. The helper publishes the dynamic-memory attribute and queries occupancy against blocks*512<=512. This request already exceeds its 220000-byte growth limit, so it is returned without growth, including on API failure after retrying any unpublished attribute and clearing the error. The request is not a residency guarantee.

## t_8615400caa631ccc | realization.choice | 356 chars | 1 declarations

tcg_d256_segmented_cp_wide:913 realization.require[0].choice

Bind matrix completion to tcg_commit and parity mbar_wait, TMEM loads to the wrappers' wait::ld, and TMEM stores to wait::st; retain before/after thread-sync fences around CTA synchronization. Use cp.async.cg.shared.global 16-byte copies with zero source size for invalid payload rows, and commit_group/wait_group for the declared payload and scale groups.

## t_8756adc52b58c5c6 | realization.choice | 521 chars | 1 declarations

tcg_d256_segmented_cp_wide:934 realization.prefer[0].choice

The launcher's initial dynamic shared requests are 103424 bytes for group_max and 189440 bytes for split. Cache tmem_safe_smem separately per instantiation; it publishes the requested attribute and may grow by 4096 bytes until occupancy fits blocks*TMEM_columns<=512, growth reaches 220000 bytes, or 96 attempts end. API failure stops probing, retries an unpublished attribute, clears the error and returns the current request. Reserved but unused Q-scale/probability space and alignment slack remain part of the request.

## t_8ab618f44699c648 | realization.choice | 278 chars | 3 declarations

tcg_d128_fscan_cp_xpose_narrow:691 realization.require[1].choice; tcg_d128_fscan_cp_xpose_wide:711 realization.require[1].choice; tcg_d128_fscan_tma_xpose_wide:712 realization.require[1].choice

Use the fixed PDL wrappers: cudaLaunchKernelEx/ExC with programmatic stream serialization for the two successors, cudaTriggerProgrammaticLaunchCompletion at the stated signal points and cudaGridDependencySynchronize at dependency waits. group_max itself uses an ordinary launch.

## t_8af22b58742d6d8a | realization.choice | 254 chars | 5 declarations

tcg_d128_fscan_cp_direct_narrow:660 realization.require[1].choice; tcg_d128_fscan_cp_direct_wide:683 realization.require[1].choice; tcg_d128_fscan_tma_direct_narrow:651 realization.require[1].choice; tcg_d128_fscan_tma_direct_wide:674 realization.require[1].choice; tcg_d128_fscan_tma_xpose_narrow:679 realization.require[1].choice

Use cudaLaunchKernelEx/ExC with programmatic stream serialization for the two successors, cudaTriggerProgrammaticLaunchCompletion at the stated signal points and cudaGridDependencySynchronize at dependency waits. group_max itself uses an ordinary launch.

## t_8dddb945c5866d7d | realization.choice | 396 chars | 1 declarations

fused_d256_tma:475 realization.require[0].choice

Bind completion and visibility to the fixed fmha_tcg_ptx.cuh wrappers: tcg_commit plus parity mbar_wait for QK/PV chains, tcg_ld32's own wait::ld, wait::st for TMEM stores and the before/after thread-sync fences at the stated synchronization points. TMA uses independent K/V transaction-byte completion. Atom cp.async uses commit_group, wait_group 0 for prime and wait_group 1 in the steady loop.

## t_93e5167014f53f1d | realization.choice | 212 chars | 1 declarations

tcg_d256_segmented_cp_wide:927 realization.require[2].choice

Use programmatic stream serialization on segment_max, scan, split and combine, cudaTriggerProgrammaticLaunchCompletion at the declared trigger points, and cudaGridDependencySynchronize at the declared data waits.

## t_9a3c116b8c31347a | realization.choice | 616 chars | 1 declarations

fused_d128_group:580 realization.prefer[0].choice

launch_tcg requests TcgSmem<128,PASS_FUSED,2,true>::bytes=190976 before tmem_safe_smem adjusts it. Besides live shared resources this reserves 512 unused Q-scale bytes and a 1024-byte alignment allowance, with no shared probability reservation. The helper publishes the dynamic-memory attribute and queries occupancy, growing by 4096 bytes while blocks*512>512. It stops on an API error, after 96 attempts, or when a request at least 220000 bytes still fails that bound; it retries an unpublished attribute, clears the error and returns the last request. The launch caches that request without a residency guarantee.

## t_af39a65c0a192508 | realization.choice | 650 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:826 realization.require[0].choice

Bind matrix completion to tcg_commit plus parity mbar_wait, TMEM readout to tcg_ld32/tcg_ld32_redmax with their wait::ld, stores to wait::st, and TMEM publication to the before/after thread-sync fences at the stated CTA joins. Bind payload and atom copies to cp.async.cg.shared.global with src-size zero-fill for invalid payload rows, per-thread commit_group and wait_group 1, retaining the empty commits and following shared fences/CTA joins. Preserve SWIZZLE_128B operands, 1024-byte shared alignment and the replicated scale representation expected by native block_scale.scale_vec::1X descriptors; QK is K-major and PV uses TMEM A with MN-major V.

## t_b81353d22b0654cf | realization.choice | 255 chars | 7 declarations

tcg_d128_segmented_cp_direct_narrow:784 realization.require[1].choice; tcg_d128_segmented_cp_direct_wide:808 realization.require[1].choice; tcg_d128_segmented_cp_xpose_narrow:777 realization.require[1].choice; tcg_d128_segmented_tma_direct_narrow:773 realization.require[1].choice; tcg_d128_segmented_tma_direct_wide:795 realization.require[1].choice; tcg_d128_segmented_tma_xpose_narrow:793 realization.require[1].choice; tcg_d128_segmented_tma_xpose_wide:777 realization.require[1].choice

Use cudaLaunchKernelEx/ExC with programmatic stream serialization for the four successors, cudaTriggerProgrammaticLaunchCompletion at the stated signal points and cudaGridDependencySynchronize at dependency waits. group_max itself uses an ordinary launch.

## t_ba0a9e1e1e9151ed | realization.choice | 718 chars | 2 declarations

tcg_d128_fscan_cp_xpose_narrow:683 realization.require[0].choice; tcg_d128_fscan_cp_xpose_wide:703 realization.require[0].choice

Bind matrix completion to tcg_commit plus parity mbar_wait, TMEM readout to tcg_ld32/tcg_ld32_redmax with their wait::ld, stores to wait::st, and TMEM publication to the before/after thread-sync fences at the stated CTA joins. Bind payload and atom copies to cp.async.cg.shared.global with 16-byte transfers and source-size zero-fill, and the declared copy-group accounting to cp.async.commit_group / cp.async.wait_group 1. These waits are per thread; retain the stated CTA joins to publish all threads' copies. Preserve SWIZZLE_128B operands, 1024-byte shared alignment and the replicated scale representation expected by native block_scale.scale_vec::1X descriptors; QK is K-major and PV uses TMEM A with MN-major V.

## t_ba79936c76428f91 | realization.choice | 287 chars | 5 declarations

tcg_d256_flat_tma_narrow:689 realization.require[0].choice; tcg_d256_flat_tma_wide:708 realization.require[0].choice; tcg_d256_segmented_cp_narrow:762 realization.require[0].choice; tcg_d256_segmented_tma_narrow:777 realization.require[0].choice; tcg_d256_segmented_tma_wide:762 realization.require[0].choice

Bind matrix completion to tcg_commit plus parity mbar_wait; TMEM reads use the wrappers' wait::ld and stores use wait::st. Retain before/after thread-sync fences. TMA byte completion covers K alone in group_max and combined K/V in split; cp.async commit_group/wait_group completes atoms.

## t_c333bf2ec185fa05 | realization.choice | 204 chars | 7 declarations

tcg_d256_flat_cp_narrow:703 realization.require[1].choice; tcg_d256_flat_cp_wide:717 realization.require[1].choice; tcg_d256_flat_tma_narrow:696 realization.require[1].choice; tcg_d256_flat_tma_wide:715 realization.require[1].choice; tcg_d256_segmented_cp_narrow:769 realization.require[1].choice; tcg_d256_segmented_tma_narrow:784 realization.require[1].choice; tcg_d256_segmented_tma_wide:769 realization.require[1].choice

Use native block_scale.scale_vec::1X descriptors with SWIZZLE_128B operands, K-major Q/K, and TMEM probabilities with MN-major V for PV. Retain 1024-byte operand alignment and replicated E8M0 scale atoms.

## t_d1aefbfb635d1b19 | realization.choice | 271 chars | 2 declarations

tc_split_flat_narrow:678 realization.require[1].choice; tc_split_flat_wide:694 realization.require[1].choice

Keep launch_tc ordinary for both MMA passes. Use pdl_launch with cudaLaunchAttributeProgrammaticStreamSerialization for scan/combine, and bind their entry waits and the scan trigger to fmha_pdl_wait/fmha_pdl_scan_trigger, including the source compile-time no-op branches.

## t_d7dc5c419e7aaf90 | realization.choice | 451 chars | 1 declarations

root:662 realization.require[0].choice

Bind completion and visibility to the fixed fmha_tcg_ptx.cuh wrappers: tcg_commit plus parity mbar_wait for the issued QK/PV chains, tcg_ld32's own wait::ld, wait::st for TMEM stores, and the before/after thread-sync fences around the stated CTA synchronization points. TMA payloads and bulk-copied atoms use transaction-byte completion. These mechanisms retain the distinct enqueue, readable and reusable conditions specified by the action contracts.

## t_dff92e7f993f42c9 | realization.choice | 720 chars | 2 declarations

tcg_d128_flat_cp_xpose_narrow:704 realization.require[0].choice; tcg_d128_flat_cp_xpose_wide:687 realization.require[0].choice

Bind matrix completion to tcg_commit plus parity mbar_wait; TMEM readout to tcg_ld32/tcg_ld32_redmax with their wait::ld; stores to wait::st; and TMEM publication to before/after-thread-sync fences at the stated CTA joins. Bind payload/atom staging to cp.async.cg.shared.global with source size 16 or zero, cp.async.commit_group and cp.async.wait_group 1. The wait completes each thread's groups older than the newest; the following shared fence and CTA join publish all copied bytes. Preserve the stated empty commits, fp8_tile_offset's 128-byte shared operand swizzle, 1024-byte shared alignment and replicated E8M0 atoms for native block_scale.scale_vec::1X descriptors: QK is K-major; PV uses TMEM A with MN-major V.

## t_e26bd3a106068b36 | realization.choice | 552 chars | 20 declarations

tcg_d128_flat_cp_direct_narrow:702 realization.prefer[0].choice; tcg_d128_flat_cp_direct_wide:719 realization.prefer[0].choice; tcg_d128_flat_cp_xpose_narrow:731 realization.prefer[0].choice; tcg_d128_flat_cp_xpose_wide:714 realization.prefer[0].choice; tcg_d128_flat_tma_direct_narrow:693 realization.prefer[0].choice; tcg_d128_flat_tma_direct_wide:712 realization.prefer[0].choice; tcg_d128_flat_tma_xpose_narrow:716 realization.prefer[0].choice; tcg_d128_flat_tma_xpose_wide:740 realization.prefer[0].choice; tcg_d128_fscan_cp_direct_narrow:670 realization.prefer[0].choice; tcg_d128_fscan_cp_direct_wide:693 realization.prefer[0].choice; tcg_d128_fscan_tma_direct_narrow:661 realization.prefer[0].choice; tcg_d128_fscan_tma_direct_wide:684 realization.prefer[0].choice; tcg_d128_fscan_tma_xpose_narrow:689 realization.prefer[0].choice; tcg_d128_segmented_cp_direct_narrow:794 realization.prefer[0].choice; tcg_d128_segmented_cp_direct_wide:818 realization.prefer[0].choice; tcg_d128_segmented_cp_xpose_narrow:787 realization.prefer[0].choice; tcg_d128_segmented_tma_direct_narrow:783 realization.prefer[0].choice; tcg_d128_segmented_tma_direct_wide:805 realization.prefer[0].choice; tcg_d128_segmented_tma_xpose_narrow:803 realization.prefer[0].choice; tcg_d128_segmented_tma_xpose_wide:787 realization.prefer[0].choice

launch_tcg starts with dynamic shared requests of 69632 bytes for group-max and 102912 for split. Besides live buffers these reserve 512 bytes of Q scales, 1024 bytes of exchange and 1024 bytes of alignment; split also reserves an unused 16384-byte shared P tile. tmem_safe_smem queries occupancy and increases the request by 4096 while blocks*256>512, up to 96 attempts, stopping on API error or a still-failing request >=220000. It retries an unpublished attribute, clears the error, and returns/caches the last request without a residency guarantee.

## t_e7d349c774797690 | realization.choice | 763 chars | 2 declarations

fused_d128_cp:465 realization.prefer[0].choice; root:687 realization.prefer[0].choice

launch_tcg requests TcgSmem<128,PASS_FUSED,1>::bytes=102912 before tmem_safe_smem applies its runtime occupancy-based request adjustment. This includes the unused 16384-byte shared probability reservation, 512-byte shared Q-scale reservation, 1024-byte exchange reservation and 1024-byte alignment allowance in addition to the live shared resources. The helper publishes the dynamic-memory attribute and queries occupancy, increasing the request by 4096 bytes while blocks*256>512. It stops on an API error, at 96 attempts, or when a request at least 220000 bytes still fails that bound. It returns the last request after retrying any unpublished attribute and clearing the error; this is not a residency guarantee. launch_tcg caches this request for later calls.

## t_ea5aae58e2135e49 | realization.choice | 498 chars | 6 declarations

tcg_d128_flat_tma_xpose_narrow:692 realization.require[0].choice; tcg_d128_flat_tma_xpose_wide:716 realization.require[0].choice; tcg_d128_fscan_tma_xpose_narrow:665 realization.require[0].choice; tcg_d128_segmented_cp_xpose_narrow:763 realization.require[0].choice; tcg_d128_segmented_tma_xpose_narrow:779 realization.require[0].choice; tcg_d128_segmented_tma_xpose_wide:763 realization.require[0].choice

Bind matrix completion to tcg_commit plus parity mbar_wait; TMEM readout to tcg_ld32/tcg_ld32_redmax with their wait::ld; stores to wait::st; and TMEM publication to before/after-thread-sync fences at the stated CTA joins. TMA payloads and bulk atoms share transaction-byte completion for each indicated slot. Preserve SWIZZLE_128B shared operands, 1024-byte shared alignment and replicated E8M0 atoms for native block_scale.scale_vec::1X descriptors: QK is K-major; PV uses TMEM A with MN-major V.

## t_f087122d9148c188 | realization.choice | 256 chars | 8 declarations

tcg_d128_flat_cp_direct_narrow:692 realization.require[1].choice; tcg_d128_flat_cp_direct_wide:709 realization.require[1].choice; tcg_d128_flat_cp_xpose_narrow:721 realization.require[1].choice; tcg_d128_flat_cp_xpose_wide:704 realization.require[1].choice; tcg_d128_flat_tma_direct_narrow:683 realization.require[1].choice; tcg_d128_flat_tma_direct_wide:702 realization.require[1].choice; tcg_d128_flat_tma_xpose_narrow:706 realization.require[1].choice; tcg_d128_flat_tma_xpose_wide:730 realization.require[1].choice

Use cudaLaunchKernelEx/ExC with programmatic stream serialization for the three successors, cudaTriggerProgrammaticLaunchCompletion at the stated signal points and cudaGridDependencySynchronize at dependency waits. group_max itself uses an ordinary launch.

## t_f2be61437d55399b | realization.choice | 304 chars | 1 declarations

fused_d128_key:476 realization.require[1].choice

Preserve the compatible SWIZZLE_128B shared operands and tcgen05 block_scale.scale_vec::1X descriptors: QK uses K-major Q/K, while PV uses packed TMEM A and MN-major V. Align the shared operand window to 1024 bytes and place E8M0 atoms in the replicated TMEM representation expected by those descriptors.

## t_f4062a0f0916f91f | realization.choice | 260 chars | 2 declarations

fused_d256_cp:467 realization.require[1].choice; fused_d256_tma:487 realization.require[1].choice

Use compatible SWIZZLE_128B shared operands and tcgen05 block_scale.scale_vec::1X descriptors: QK consumes K-major Q/K, PV consumes packed TMEM A and MN-major V. Align the shared operand window to 1024 bytes and retain the replicated TMEM scale representation.

## t_fd476151e4862660 | realization.choice | 373 chars | 1 declarations

fused_d128_group:566 realization.require[1].choice

Bind role publication to bar.sync with 128 arrivals and before/after thread-sync fences. Bind state transfers to shared stores followed by mbarrier.arrive and acquire parity waits with 128 arrivals; QK/PV use tcgen05.commit with one arrival. Retain distinct producer/parity barriers, self-retiring tcgen05.ld and TMEM wait::st. cp.async waits retain their per-thread scope.

## t_fdfad642301f05f7 | realization.choice | 657 chars | 2 declarations

tcg_d128_segmented_cp_direct_narrow:767 realization.require[0].choice; tcg_d128_segmented_cp_direct_wide:791 realization.require[0].choice

Bind matrix completion to tcg_commit plus parity mbar_wait; TMEM readout to tcg_ld32/tcg_ld32_redmax with their wait::ld; stores to wait::st; and publication to before/after-thread-sync fences at the stated CTA joins. Bind payload/atom copies to 16-byte cp.async.cg with commit_group and wait_group 1, preserving the empty commits and payload zero-fill source-size predicate. Copy completion is per issuing thread; shared-proxy fences and CTA joins publish all copies. Preserve SWIZZLE_128B shared operands, 1024-byte shared alignment and replicated E8M0 atoms for native block_scale.scale_vec::1X descriptors: QK is K-major; PV uses TMEM A with MN-major V.

## t_050e34fedb7925c0 | realization.reason | 205 chars | 2 declarations

fused_d256_cp:487 realization.prefer[0].reason; fused_d256_tma:506 realization.prefer[0].reason

Actual shared-memory reservation constrains CTA capacity even though probabilities and query scales reside in TMEM; the host checks occupancy because the runtime model does not account for TMEM allocation.

## t_0e6daadd5a941b26 | realization.reason | 79 chars | 4 declarations

tcg_d256_flat_cp_narrow:713 realization.require[2].reason; tcg_d256_flat_cp_wide:727 realization.require[2].reason; tcg_d256_flat_tma_narrow:706 realization.require[2].reason; tcg_d256_flat_tma_wide:725 realization.require[2].reason

Early scan launch permission must not be interpreted as scan-output visibility.

## t_0eb159dcc4689916 | realization.reason | 263 chars | 1 declarations

tc_fused:393 realization.require[0].reason

Copy-group counting establishes current-slot readiness while the next transfer may continue. Compatible layouts and synchronous register-result products supply the operand mapping and completed-read guarantees needed by the single Vbf slot and register lifetimes.

## t_1b1085c00ac46a66 | realization.reason | 122 chars | 1 declarations

tcg_d128_fscan_tma_xpose_narrow:684 realization.require[1].reason

Launch permission is separate from producer visibility; split's independent preparation remains before its group-max wait.

## t_1e6f4c27432a7466 | realization.reason | 135 chars | 2 declarations

fused_d256_cp:472 realization.require[1].reason; fused_d256_tma:492 realization.require[1].reason

Payloads, probability packing and E8M0 atoms must address the same row/key/channel contractions across both 128-channel chunks of D256.

## t_20ca5b2971ddfb14 | realization.reason | 152 chars | 1 declarations

tcg_d256_segmented_cp_wide:931 realization.require[2].reason

The segment_max entry trigger permits scan launch before its data is ready; the scan trigger permits split preparation before scan outputs are readable.

## t_266683c200ebb2e6 | realization.reason | 142 chars | 1 declarations

fused_d128_group:590 realization.prefer[0].reason

Shared-memory reservation affects capacity; the helper compensates for occupancy accounting that does not include this CTA's 512 TMEM columns.

## t_2d96fd91fbb64bf0 | realization.reason | 212 chars | 2 declarations

fused_d128_cp:459 realization.require[1].reason; root:681 realization.require[1].reason

Payloads, packed probabilities and E8M0 atoms must describe the same row/key/channel contractions. In particular the two QK halves share the 64-column destination while selecting different K and scale subregions.

## t_30162fc3c9cd708a | realization.reason | 138 chars | 1 declarations

fused_d128_group:562 realization.require[0].reason

The two score/probability regions and shared numerator must address the declared products with matching packed payloads and scale factors.

## t_30cf98e0831086a8 | realization.reason | 206 chars | 2 declarations

tc_split_flat_narrow:681 realization.require[1].reason; tc_split_flat_wide:697 realization.require[1].reason

The MMA predecessors do not signal early completion, and the ordinary split successor does not consume the early scan launch permission. That permission cannot replace the declared kernel_done dependencies.

## t_331f1c2f4c82275c | realization.reason | 198 chars | 1 declarations

fused_d256_cp:462 realization.require[0].reason

CTA synchronization alone cannot complete asynchronous products, retire TMEM access or prove copy arrival. The distinct enqueue, readable and reusable guarantees require these completion mechanisms.

## t_33486c0b0b6d11fc | realization.reason | 197 chars | 1 declarations

fused_d128_key:471 realization.require[0].reason

CTA barriers alone do not complete asynchronous copies or matrix products. These distinct mechanisms protect the four-stage ring, alternating scores/scales and single probability/numerator storage.

## t_34ab4cbaded732ba | realization.reason | 98 chars | 1 declarations

tcg_d256_segmented_cp_wide:919 realization.require[0].reason

CTA barriers alone establish none of the asynchronous completion or storage-retirement guarantees.

## t_3a48bc74d39ffde4 | realization.reason | 246 chars | 2 declarations

tcg_d128_fscan_cp_direct_narrow:654 realization.require[0].reason; tcg_d128_fscan_cp_direct_wide:677 realization.require[0].reason

CTA barriers alone do not complete asynchronous matrix or copy work; copy-group waits alone do not join the other issuers. These bindings protect score/PV aliasing, scale/probability reuse and matching payload, scale and probability contractions.

## t_3a8de68c8d11661c | realization.reason | 192 chars | 18 declarations

tcg_d128_flat_cp_direct_narrow:687 realization.require[0].reason; tcg_d128_flat_cp_direct_wide:704 realization.require[0].reason; tcg_d128_flat_cp_xpose_narrow:716 realization.require[0].reason; tcg_d128_flat_cp_xpose_wide:699 realization.require[0].reason; tcg_d128_flat_tma_direct_narrow:678 realization.require[0].reason; tcg_d128_flat_tma_direct_wide:697 realization.require[0].reason; tcg_d128_flat_tma_xpose_narrow:701 realization.require[0].reason; tcg_d128_flat_tma_xpose_wide:725 realization.require[0].reason; tcg_d128_fscan_tma_direct_narrow:646 realization.require[0].reason; tcg_d128_fscan_tma_direct_wide:669 realization.require[0].reason; tcg_d128_fscan_tma_xpose_narrow:674 realization.require[0].reason; tcg_d128_segmented_cp_direct_narrow:779 realization.require[0].reason; tcg_d128_segmented_cp_direct_wide:803 realization.require[0].reason; tcg_d128_segmented_cp_xpose_narrow:772 realization.require[0].reason; tcg_d128_segmented_tma_direct_narrow:768 realization.require[0].reason; tcg_d128_segmented_tma_direct_wide:790 realization.require[0].reason; tcg_d128_segmented_tma_xpose_narrow:788 realization.require[0].reason; tcg_d128_segmented_tma_xpose_wide:772 realization.require[0].reason

CTA barriers alone do not complete asynchronous matrix or copy work. These bindings protect score/PV aliasing, scale/probability reuse and matching payload, scale and probability contractions.

## t_4cac0a6b0583ea49 | realization.reason | 198 chars | 1 declarations

fused_d128_key:497 realization.prefer[0].reason

The actual shared-memory reservation affects CTA capacity even though Prob8 resides in TMEM. The helper compensates for occupancy accounting that does not include the 512 TMEM columns used by a CTA.

## t_59fbbca981c3e483 | realization.reason | 111 chars | 1 declarations

tcg_d256_segmented_cp_wide:943 realization.prefer[0].reason

Physical reservations affect occupancy beyond the live resource views; the helper is not a residency guarantee.

## t_66b2433b74ecde08 | realization.reason | 219 chars | 2 declarations

tcg_d128_fscan_cp_xpose_narrow:688 realization.require[0].reason; tcg_d128_fscan_cp_xpose_wide:708 realization.require[0].reason

CTA barriers alone do not complete matrix or asynchronous copy operations. These bindings protect score/PV aliasing, single scale/probability slots and the correspondence between payload, probability and scale operands.

## t_68cfd0e71316a4a5 | realization.reason | 256 chars | 2 declarations

tc_split_segmented_narrow:788 realization.require[1].reason; tc_split_segmented_wide:823 realization.require[1].reason

The segment trigger permits early scan launch, while its entry wait protects completed summaries. Neither MMA predecessor signals early completion, and the ordinary split successor cannot replace its kernel_done dependency with the scan's early permission.

## t_6e16471ff124d909 | realization.reason | 212 chars | 2 declarations

fused_d128_cp:477 realization.prefer[0].reason; root:699 realization.prefer[0].reason

The actual shared-memory reservation affects CTA capacity even though probabilities reside in TMEM. The host helper compensates for runtime occupancy accounting that does not include the 256 TMEM columns per CTA.

## t_7dd538aedd0b93cf | realization.reason | 140 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:837 realization.require[1].reason

Launch permission alone cannot make the producer's global stores readable; split's independent preparation must remain before its scan wait.

## t_933dbffbc97dac07 | realization.reason | 145 chars | 3 declarations

tcg_d128_fscan_cp_xpose_narrow:693 realization.require[1].reason; tcg_d128_fscan_cp_xpose_wide:713 realization.require[1].reason; tcg_d128_fscan_tma_xpose_wide:714 realization.require[1].reason

Launch permission alone cannot make the producer's global stores readable; split's independent preparation must remain before its group-max wait.

## t_945928316c78f40e | realization.reason | 102 chars | 1 declarations

tcg_d256_segmented_cp_wide:925 realization.require[1].reason

Payload, scale and probability addressing must describe the same contractions across both D256 chunks.

## t_9543f91d59340ea8 | realization.reason | 88 chars | 7 declarations

tcg_d256_flat_cp_narrow:707 realization.require[1].reason; tcg_d256_flat_cp_wide:721 realization.require[1].reason; tcg_d256_flat_tma_narrow:700 realization.require[1].reason; tcg_d256_flat_tma_wide:719 realization.require[1].reason; tcg_d256_segmented_cp_narrow:773 realization.require[1].reason; tcg_d256_segmented_tma_narrow:788 realization.require[1].reason; tcg_d256_segmented_tma_wide:773 realization.require[1].reason

Operand and scale addresses must describe the same contractions across both D256 chunks.

## t_a8afab1ac62bbcdd | realization.reason | 181 chars | 4 declarations

tc_split_flat_narrow:675 realization.require[0].reason; tc_split_flat_wide:691 realization.require[0].reason; tc_split_segmented_narrow:782 realization.require[0].reason; tc_split_segmented_wide:817 realization.require[0].reason

Copy-group counting establishes current-slot readability while next-group transfers may continue; synchronous register results establish completed operand reads before buffer reuse.

## t_aac35ab285acfbe1 | realization.reason | 133 chars | 1 declarations

fused_d128_key:482 realization.require[1].reason

Payloads, packed probabilities and scales must address the same contractions, including the two key slices and output-channel slices.

## t_acec4122e64f70ee | realization.reason | 213 chars | 1 declarations

tcg_d128_fscan_tma_xpose_wide:709 realization.require[0].reason

CTA barriers alone do not complete matrix or transaction operations. These bindings protect score/PV aliasing, single scale/probability slots and the correspondence between payload, probability and scale operands.

## t_ae5e9ff04607c314 | realization.reason | 117 chars | 15 declarations

tcg_d128_flat_cp_direct_narrow:697 realization.require[1].reason; tcg_d128_flat_cp_direct_wide:714 realization.require[1].reason; tcg_d128_flat_cp_xpose_narrow:726 realization.require[1].reason; tcg_d128_flat_cp_xpose_wide:709 realization.require[1].reason; tcg_d128_flat_tma_direct_narrow:688 realization.require[1].reason; tcg_d128_flat_tma_direct_wide:707 realization.require[1].reason; tcg_d128_flat_tma_xpose_narrow:711 realization.require[1].reason; tcg_d128_flat_tma_xpose_wide:735 realization.require[1].reason; tcg_d128_segmented_cp_direct_narrow:789 realization.require[1].reason; tcg_d128_segmented_cp_direct_wide:813 realization.require[1].reason; tcg_d128_segmented_cp_xpose_narrow:782 realization.require[1].reason; tcg_d128_segmented_tma_direct_narrow:778 realization.require[1].reason; tcg_d128_segmented_tma_direct_wide:800 realization.require[1].reason; tcg_d128_segmented_tma_xpose_narrow:798 realization.require[1].reason; tcg_d128_segmented_tma_xpose_wide:782 realization.require[1].reason

Launch permission is separate from producer visibility; split's independent preparation remains before its scan wait.

## t_b71013b9a0f4f82e | realization.reason | 94 chars | 7 declarations

tcg_d256_flat_cp_narrow:701 realization.require[0].reason; tcg_d256_flat_cp_wide:715 realization.require[0].reason; tcg_d256_flat_tma_narrow:694 realization.require[0].reason; tcg_d256_flat_tma_wide:713 realization.require[0].reason; tcg_d256_segmented_cp_narrow:767 realization.require[0].reason; tcg_d256_segmented_tma_narrow:782 realization.require[0].reason; tcg_d256_segmented_tma_wide:767 realization.require[0].reason

CTA synchronization alone does not establish asynchronous completion or retire source storage.

## t_b7e78dd0992dd4ff | realization.reason | 212 chars | 2 declarations

fused_d128_cp:448 realization.require[0].reason; root:670 realization.require[0].reason

A CTA barrier alone neither completes the asynchronous matrix products nor retires TMEM loads or establishes shared async-proxy visibility; removing these bindings would invalidate the single score/scale/P slots.

## t_cba77b1207ff3f21 | realization.reason | 280 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:831 realization.require[0].reason

CTA barriers alone do not complete matrix or async-copy operations, and a copy-group wait alone does not publish other threads' copies. These bindings protect score/PV aliasing, single scale/probability slots and the correspondence between payload, probability and scale operands.

## t_cd09d828ffbd145b | realization.reason | 137 chars | 20 declarations

tcg_d128_flat_cp_direct_narrow:712 realization.prefer[0].reason; tcg_d128_flat_cp_direct_wide:729 realization.prefer[0].reason; tcg_d128_flat_cp_xpose_narrow:741 realization.prefer[0].reason; tcg_d128_flat_cp_xpose_wide:724 realization.prefer[0].reason; tcg_d128_flat_tma_direct_narrow:703 realization.prefer[0].reason; tcg_d128_flat_tma_direct_wide:722 realization.prefer[0].reason; tcg_d128_flat_tma_xpose_narrow:726 realization.prefer[0].reason; tcg_d128_flat_tma_xpose_wide:750 realization.prefer[0].reason; tcg_d128_fscan_cp_direct_narrow:680 realization.prefer[0].reason; tcg_d128_fscan_cp_direct_wide:703 realization.prefer[0].reason; tcg_d128_fscan_tma_direct_narrow:671 realization.prefer[0].reason; tcg_d128_fscan_tma_direct_wide:694 realization.prefer[0].reason; tcg_d128_fscan_tma_xpose_narrow:699 realization.prefer[0].reason; tcg_d128_segmented_cp_direct_narrow:804 realization.prefer[0].reason; tcg_d128_segmented_cp_direct_wide:828 realization.prefer[0].reason; tcg_d128_segmented_cp_xpose_narrow:797 realization.prefer[0].reason; tcg_d128_segmented_tma_direct_narrow:793 realization.prefer[0].reason; tcg_d128_segmented_tma_direct_wide:815 realization.prefer[0].reason; tcg_d128_segmented_tma_xpose_narrow:813 realization.prefer[0].reason; tcg_d128_segmented_tma_xpose_wide:797 realization.prefer[0].reason

Shared reservations affect capacity even when probabilities occupy TMEM; the helper compensates for occupancy accounting that omits TMEM.

## t_d51aa3186ab02a3e | realization.reason | 119 chars | 7 declarations

tcg_d256_flat_cp_narrow:724 realization.prefer[0].reason; tcg_d256_flat_cp_wide:738 realization.prefer[0].reason; tcg_d256_flat_tma_narrow:717 realization.prefer[0].reason; tcg_d256_flat_tma_wide:736 realization.prefer[0].reason; tcg_d256_segmented_cp_narrow:790 realization.prefer[0].reason; tcg_d256_segmented_tma_narrow:805 realization.prefer[0].reason; tcg_d256_segmented_tma_wide:790 realization.prefer[0].reason

Physical reservations constrain occupancy beyond the live resource views, while the runtime occupancy model omits TMEM.

## t_e12a1fa2fcdcb2d6 | realization.reason | 141 chars | 4 declarations

tcg_d128_fscan_cp_xpose_narrow:704 realization.prefer[0].reason; tcg_d128_fscan_cp_xpose_wide:724 realization.prefer[0].reason; tcg_d128_fscan_tma_xpose_wide:725 realization.prefer[0].reason; tcg_d128_segmented_cp_xpose_wide:848 realization.prefer[0].reason

Shared reservations affect CTA capacity even when P resides in TMEM; the helper compensates for occupancy accounting that omits TMEM columns.

## t_f4a183b69530f412 | realization.reason | 121 chars | 4 declarations

tcg_d128_fscan_cp_direct_narrow:665 realization.require[1].reason; tcg_d128_fscan_cp_direct_wide:688 realization.require[1].reason; tcg_d128_fscan_tma_direct_narrow:656 realization.require[1].reason; tcg_d128_fscan_tma_direct_wide:679 realization.require[1].reason

Launch permission is separate from producer visibility; split's independent preparation remains before its GroupMax wait.

## t_f880080fe2c5d65a | realization.reason | 236 chars | 1 declarations

fused_d128_group:573 realization.require[1].reason

A parity is not an unbounded completion ticket. Separate PV barriers and two denominator barriers per producer prevent advancement twice past a consumer's waited phase. CTA barriers do not replace asynchronous copy or matrix completion.

## t_faee3a68d0dcba0d | realization.reason | 198 chars | 1 declarations

fused_d256_tma:482 realization.require[0].reason

CTA synchronization alone cannot complete asynchronous products, retire TMEM access or prove atom arrival. The distinct enqueue, readable and reusable guarantees require these completion mechanisms.

## t_ffcd01174859ec75 | realization.reason | 107 chars | 3 declarations

tcg_d256_segmented_cp_narrow:779 realization.require[2].reason; tcg_d256_segmented_tma_narrow:794 realization.require[2].reason; tcg_d256_segmented_tma_wide:779 realization.require[2].reason

Neither the segment_max entry trigger nor the scan trigger establishes output visibility for its successor.

## t_061f029418cfede0 | schedule.describe | 106 chars | 3 declarations

tcg_d128_segmented_cp_xpose_narrow:281 design.work_schedule.phase.segment_max.describe; tcg_d128_segmented_tma_xpose_narrow:297 design.work_schedule.phase.segment_max.describe; tcg_d128_segmented_tma_xpose_wide:281 design.work_schedule.phase.segment_max.describe

Independently reduce each segment for every row, including segment zero whose summary is unused by replay.

## t_1042cd4abb9d694f | schedule.describe | 374 chars | 2 declarations

tc_split_segmented_narrow:326 design.work_schedule.phase.group_max.describe; tc_split_segmented_wide:355 design.work_schedule.phase.group_max.describe

Compute the covered maximum table and temporary caller-buffer values. Model.forward resolves HostPlan as in when; no tensor maps are encoded. Cached graph replay, successful capture/replay, and direct submission execute this same five-kernel sequence on the caller stream. Capture records without execution; no scratch-initialization kernel or host completion wait is added.

## t_10714c5fd9312c8e | schedule.describe | 603 chars | 4 declarations

fused_d128_cp:184 design.work_schedule.phase.main.describe; fused_d256_cp:180 design.work_schedule.phase.main.describe; fused_d256_tma:180 design.work_schedule.phase.main.describe; root:394 design.work_schedule.phase.main.describe

One fused tcgen05 kernel computes scores, normalization, PV, O and optional LSE. Model.forward resolves HostPlan and MapsValid as in when. A cached graph is replayed when available. If no replay succeeds, an eligible call tries capture, instantiation and graph launch; capture records launch_sequence without executing its kernels. If still not launched, launch_sequence submits directly. These routes execute the same single main phase on the caller stream. The selected computation neither initializes scratch nor submits an auxiliary computation kernel; its launch does not wait for main to complete.

## t_1570b8e8285ac423 | schedule.describe | 442 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:271 design.work_schedule.phase.group_max.describe

Compute all GroupMax entries, including explicit -inf outside each query tile's useful interval. Model.forward resolves HostPlan and MapsValid. A successful cached graph replay executes this same five-kernel sequence; eligible calls may capture and instantiate it before replay. Capture records without executing the kernels; absent a successful replay, launch_sequence submits directly on the caller stream. No host completion wait is added.

## t_16fc069ef9c4ad77 | schedule.describe | 64 chars | 1 declarations

tcg_d256_segmented_cp_wide:367 design.work_schedule.phase.split.describe

Recompute QK, replay Mscan and produce weighted PartA and PartB.

## t_22dcab09473bb81c | schedule.describe | 73 chars | 1 declarations

tcg_d256_segmented_cp_wide:341 design.work_schedule.phase.scan.describe

Produce all Mscan entries and Mtail with independent segmented row scans.

## t_23307ec408fb69d9 | schedule.describe | 98 chars | 4 declarations

tc_split_flat_narrow:357 design.work_schedule.phase.split.describe; tc_split_flat_wide:371 design.work_schedule.phase.split.describe; tc_split_segmented_narrow:433 design.work_schedule.phase.split.describe; tc_split_segmented_wide:462 design.work_schedule.phase.split.describe

Recompute QK and produce weighted FP32 numerator and denominator partials for every row and split.

## t_234bba74dc0d19ec | schedule.describe | 85 chars | 7 declarations

tcg_d128_flat_cp_xpose_narrow:337 design.work_schedule.phase.combine.describe; tcg_d128_flat_cp_xpose_wide:320 design.work_schedule.phase.combine.describe; tcg_d128_flat_tma_xpose_narrow:334 design.work_schedule.phase.combine.describe; tcg_d128_flat_tma_xpose_wide:350 design.work_schedule.phase.combine.describe; tcg_d128_segmented_cp_xpose_narrow:374 design.work_schedule.phase.combine.describe; tcg_d128_segmented_tma_xpose_narrow:390 design.work_schedule.phase.combine.describe; tcg_d128_segmented_tma_xpose_wide:374 design.work_schedule.phase.combine.describe

Merge all materialized partials for one real output row and write O and optional LSE.

## t_307136ce20c133ca | schedule.describe | 95 chars | 8 declarations

tcg_d128_flat_cp_direct_narrow:264 design.work_schedule.phase.flat_scan.describe; tcg_d128_flat_cp_direct_wide:277 design.work_schedule.phase.flat_scan.describe; tcg_d128_flat_cp_xpose_narrow:271 design.work_schedule.phase.flat_scan.describe; tcg_d128_flat_cp_xpose_wide:254 design.work_schedule.phase.flat_scan.describe; tcg_d128_flat_tma_direct_narrow:265 design.work_schedule.phase.flat_scan.describe; tcg_d128_flat_tma_direct_wide:278 design.work_schedule.phase.flat_scan.describe; tcg_d128_flat_tma_xpose_narrow:268 design.work_schedule.phase.flat_scan.describe; tcg_d128_flat_tma_xpose_wide:284 design.work_schedule.phase.flat_scan.describe

Independently scan each row's T group maxima in order and write all MState snapshots and MTail.

## t_3c6c9e5a81e743e2 | schedule.describe | 471 chars | 1 declarations

tcg_d256_segmented_cp_wide:268 design.work_schedule.phase.group_max.describe

Compute and store GroupMax, including excluded tails. Model.forward resolves HostPlan and MapsValid, then executes this five-kernel sequence through a cached graph, a successfully captured/instantiated graph, or direct launch_sequence on the caller stream. Capture records rather than executes kernels. No scratch-initialization kernel is submitted. Submission does not wait for caller-visible completion. PDL launch permission is distinct from the successor's data wait.

## t_451378a4a6c83692 | schedule.describe | 71 chars | 4 declarations

tc_split_flat_narrow:403 design.work_schedule.phase.combine.describe; tc_split_flat_wide:417 design.work_schedule.phase.combine.describe; tc_split_segmented_narrow:479 design.work_schedule.phase.combine.describe; tc_split_segmented_wide:508 design.work_schedule.phase.combine.describe

Merge all FP32 partials for one row and write final O and optional LSE.

## t_498a47226343f7de | schedule.describe | 85 chars | 4 declarations

tcg_d128_segmented_cp_direct_narrow:293 design.work_schedule.phase.segment_max.describe; tcg_d128_segmented_cp_direct_wide:307 design.work_schedule.phase.segment_max.describe; tcg_d128_segmented_tma_direct_narrow:293 design.work_schedule.phase.segment_max.describe; tcg_d128_segmented_tma_direct_wide:309 design.work_schedule.phase.segment_max.describe

Materialize every segment summary, including the unconsumed summary for segment zero.

## t_4be710a6a91fd945 | schedule.describe | 379 chars | 3 declarations

tcg_d256_segmented_cp_narrow:220 design.work_schedule.phase.group_max.describe; tcg_d256_segmented_tma_narrow:235 design.work_schedule.phase.group_max.describe; tcg_d256_segmented_tma_wide:220 design.work_schedule.phase.group_max.describe

Compute GroupMax and excluded -inf tails. Model.forward resolves HostPlan and MapsValid, then submits this five-kernel sequence through a cached graph, successful capture/instantiation, or direct launch_sequence on the caller stream. Capture records without executing kernels. No scratch-initialization kernel is submitted; submission does not wait for caller-visible completion.

## t_541013a2fc98cab8 | schedule.describe | 78 chars | 3 declarations

tcg_d256_segmented_cp_narrow:289 design.work_schedule.phase.scan.describe; tcg_d256_segmented_tma_narrow:304 design.work_schedule.phase.scan.describe; tcg_d256_segmented_tma_wide:289 design.work_schedule.phase.scan.describe

Produce all delayed states and the final maximum with one serial scan per row.

## t_547644e7520173bd | schedule.describe | 63 chars | 3 declarations

tcg_d256_segmented_cp_narrow:264 design.work_schedule.phase.segment_max.describe; tcg_d256_segmented_tma_narrow:279 design.work_schedule.phase.segment_max.describe; tcg_d256_segmented_tma_wide:264 design.work_schedule.phase.segment_max.describe

Compute each segment maximum independently for a block of rows.

## t_60b76e71563bd639 | schedule.describe | 61 chars | 7 declarations

tcg_d256_flat_cp_narrow:319 design.work_schedule.phase.combine.describe; tcg_d256_flat_cp_wide:333 design.work_schedule.phase.combine.describe; tcg_d256_flat_tma_narrow:319 design.work_schedule.phase.combine.describe; tcg_d256_flat_tma_wide:331 design.work_schedule.phase.combine.describe; tcg_d256_segmented_cp_narrow:356 design.work_schedule.phase.combine.describe; tcg_d256_segmented_tma_narrow:371 design.work_schedule.phase.combine.describe; tcg_d256_segmented_tma_wide:356 design.work_schedule.phase.combine.describe

Sum all Ns partials for one row and write O and optional LSE.

## t_67101f8624e0785d | schedule.describe | 97 chars | 8 declarations

tcg_d128_flat_cp_direct_narrow:289 design.work_schedule.phase.split.describe; tcg_d128_flat_cp_direct_wide:302 design.work_schedule.phase.split.describe; tcg_d128_flat_tma_direct_narrow:290 design.work_schedule.phase.split.describe; tcg_d128_flat_tma_direct_wide:303 design.work_schedule.phase.split.describe; tcg_d128_segmented_cp_direct_narrow:345 design.work_schedule.phase.split.describe; tcg_d128_segmented_cp_direct_wide:359 design.work_schedule.phase.split.describe; tcg_d128_segmented_tma_direct_narrow:345 design.work_schedule.phase.split.describe; tcg_d128_segmented_tma_direct_wide:361 design.work_schedule.phase.split.describe

Recompute QK, replay scan states and write weighted FP32 PartA and PartB for one numerator split.

## t_6907184f34286839 | schedule.describe | 96 chars | 7 declarations

tcg_d128_flat_cp_xpose_narrow:296 design.work_schedule.phase.split.describe; tcg_d128_flat_cp_xpose_wide:279 design.work_schedule.phase.split.describe; tcg_d128_flat_tma_xpose_narrow:293 design.work_schedule.phase.split.describe; tcg_d128_flat_tma_xpose_wide:309 design.work_schedule.phase.split.describe; tcg_d128_segmented_cp_xpose_narrow:333 design.work_schedule.phase.split.describe; tcg_d128_segmented_tma_xpose_narrow:349 design.work_schedule.phase.split.describe; tcg_d128_segmented_tma_xpose_wide:333 design.work_schedule.phase.split.describe

Recompute QK, replay scan states and write FP32 PartA and PartB under the partials step's cases.

## t_69c7a9c4f956bc2a | schedule.describe | 110 chars | 4 declarations

tcg_d128_segmented_cp_direct_narrow:319 design.work_schedule.phase.segmented_scan.describe; tcg_d128_segmented_cp_direct_wide:333 design.work_schedule.phase.segmented_scan.describe; tcg_d128_segmented_tma_direct_narrow:319 design.work_schedule.phase.segmented_scan.describe; tcg_d128_segmented_tma_direct_wide:335 design.work_schedule.phase.segmented_scan.describe

Replay segment zero, then fill constant prefixes and replay suffixes selected by summary batches for each row.

## t_6b19259de09e5f94 | schedule.describe | 379 chars | 4 declarations

tcg_d256_flat_cp_narrow:210 design.work_schedule.phase.group_max.describe; tcg_d256_flat_cp_wide:224 design.work_schedule.phase.group_max.describe; tcg_d256_flat_tma_narrow:210 design.work_schedule.phase.group_max.describe; tcg_d256_flat_tma_wide:222 design.work_schedule.phase.group_max.describe

Compute GroupMax and excluded -inf tails. Model.forward resolves HostPlan and MapsValid, then submits this four-kernel sequence through a cached graph, successful capture/instantiation, or direct launch_sequence on the caller stream. Capture records without executing kernels. No scratch-initialization kernel is submitted; submission does not wait for caller-visible completion.

## t_6cd931180a4b1738 | schedule.describe | 94 chars | 2 declarations

tc_split_flat_narrow:327 design.work_schedule.phase.scan.describe; tc_split_flat_wide:341 design.work_schedule.phase.scan.describe

Replay delayed maxima for all logical rows, including the region absent from the maximum grid.

## t_7434ae13093ee940 | schedule.describe | 405 chars | 8 declarations

tcg_d128_flat_cp_direct_narrow:222 design.work_schedule.phase.group_max.describe; tcg_d128_flat_cp_direct_wide:235 design.work_schedule.phase.group_max.describe; tcg_d128_flat_cp_xpose_narrow:229 design.work_schedule.phase.group_max.describe; tcg_d128_flat_cp_xpose_wide:212 design.work_schedule.phase.group_max.describe; tcg_d128_flat_tma_direct_narrow:223 design.work_schedule.phase.group_max.describe; tcg_d128_flat_tma_direct_wide:236 design.work_schedule.phase.group_max.describe; tcg_d128_flat_tma_xpose_narrow:226 design.work_schedule.phase.group_max.describe; tcg_d128_flat_tma_xpose_wide:242 design.work_schedule.phase.group_max.describe

Compute GroupMax, including explicit -inf outside useful intervals. Model.forward resolves HostPlan and MapsValid. A cached graph, or an eligible capture/instantiation followed by successful graph launch, executes this same four-kernel sequence. Capture records without executing kernels; otherwise launch_sequence submits directly on the caller stream. The host does not wait for this sequence to finish.

## t_78fcab87f41a8c47 | schedule.describe | 678 chars | 2 declarations

fused_d128_group:183 design.work_schedule.phase.main.describe; fused_d128_key:186 design.work_schedule.phase.main.describe

One fused tcgen05 kernel computes scores, normalization, PV, O and optional LSE. Model.forward resolves HostPlan as in when. The host also attempts K/V descriptor encoding, whose result this cp.async path ignores. A cached graph is replayed when available. If no replay succeeds, an eligible call tries capture, instantiation and graph launch; capture records launch_sequence without executing its kernels. If still not launched, launch_sequence submits directly. These routes execute the same single main phase on the caller stream. The selected computation neither initializes scratch nor submits an auxiliary computation kernel; its launch does not wait for main to complete.

## t_7be1b6200640d5ab | schedule.describe | 443 chars | 3 declarations

tcg_d128_fscan_cp_xpose_narrow:237 design.work_schedule.phase.group_max.describe; tcg_d128_fscan_cp_xpose_wide:251 design.work_schedule.phase.group_max.describe; tcg_d128_fscan_tma_xpose_wide:251 design.work_schedule.phase.group_max.describe

Compute all GroupMax entries, including explicit -inf outside each query tile's useful interval. Model.forward resolves HostPlan and MapsValid. A successful cached graph replay executes this same three-kernel sequence; eligible calls may capture and instantiate it before replay. Capture records without executing the kernels; absent a successful replay, launch_sequence submits directly on the caller stream. No host completion wait is added.

## t_7fe9800d6517ebf4 | schedule.describe | 112 chars | 3 declarations

tcg_d128_fscan_cp_xpose_narrow:321 design.work_schedule.phase.combine.describe; tcg_d128_fscan_cp_xpose_wide:335 design.work_schedule.phase.combine.describe; tcg_d128_fscan_tma_xpose_wide:341 design.work_schedule.phase.combine.describe

Weight and sum all materialized partials for one real row, then write its D BF16 channels and optional FP32 LSE.

## t_813bf9016f624756 | schedule.describe | 142 chars | 4 declarations

tcg_d128_fscan_cp_direct_narrow:274 design.work_schedule.phase.split.describe; tcg_d128_fscan_cp_direct_wide:289 design.work_schedule.phase.split.describe; tcg_d128_fscan_tma_direct_narrow:274 design.work_schedule.phase.split.describe; tcg_d128_fscan_tma_direct_wide:290 design.work_schedule.phase.split.describe

Scan the prefix, recompute QK and advance local states, then write unweighted FP32 PartA/PartB and PartM. The last split also publishes MTail.

## t_819dc6f3eabfc97e | schedule.describe | 100 chars | 2 declarations

tc_split_segmented_narrow:370 design.work_schedule.phase.segmax.describe; tc_split_segmented_wide:399 design.work_schedule.phase.segmax.describe

Summarize every segment of the maximum table, including untouched rows and the unused first segment.

## t_9145fd24bbc6d7e9 | schedule.describe | 90 chars | 3 declarations

tcg_d128_segmented_cp_xpose_narrow:307 design.work_schedule.phase.segmented_scan.describe; tcg_d128_segmented_tma_xpose_narrow:323 design.work_schedule.phase.segmented_scan.describe; tcg_d128_segmented_tma_xpose_wide:307 design.work_schedule.phase.segmented_scan.describe

Independently screen and replay each row in order, writing all MState snapshots and MTail.

## t_917370db55c36126 | schedule.describe | 136 chars | 1 declarations

tcg_d128_fscan_tma_xpose_narrow:279 design.work_schedule.phase.split.describe

Walk the GroupMax prefix, recompute QK/PV while advancing the local scan, and write PartA/PartB/PartM; the last split also writes MTail.

## t_91f2d5ac857b4cfa | schedule.describe | 144 chars | 3 declarations

tcg_d128_fscan_cp_xpose_narrow:280 design.work_schedule.phase.split.describe; tcg_d128_fscan_cp_xpose_wide:294 design.work_schedule.phase.split.describe; tcg_d128_fscan_tma_xpose_wide:300 design.work_schedule.phase.split.describe

Walk the prefix of GroupMax, recompute QK/PV while advancing the local scan, and write PartA, PartB and PartM; the last split also writes MTail.

## t_9613cbcd1904a2d5 | schedule.describe | 478 chars | 1 declarations

tc_fused:186 design.work_schedule.phase.main.describe

One fused kernel computes scores, normalization, PV, O and optional LSE. The prepared Model.forward resolves HostPlan as in when; this path does not encode tensor maps. The host may replay a cached graph, capture and instantiate this sequence, or submit directly if no graph launch succeeds. Capture records without executing kernels. Each successful route executes the same single main phase on the caller stream without a host completion wait or scratch-initialization kernel.

## t_a55766c0c9753eaf | schedule.describe | 374 chars | 2 declarations

tc_split_flat_narrow:283 design.work_schedule.phase.group_max.describe; tc_split_flat_wide:297 design.work_schedule.phase.group_max.describe

Compute the covered maximum table and temporary caller-buffer values. Model.forward resolves HostPlan as in when; no tensor maps are encoded. Cached graph replay, successful capture/replay, and direct submission execute this same four-kernel sequence on the caller stream. Capture records without execution; no scratch-initialization kernel or host completion wait is added.

## t_a8a845a2c530a57c | schedule.describe | 79 chars | 8 declarations

tcg_d128_flat_cp_direct_narrow:330 design.work_schedule.phase.combine.describe; tcg_d128_flat_cp_direct_wide:343 design.work_schedule.phase.combine.describe; tcg_d128_flat_tma_direct_narrow:331 design.work_schedule.phase.combine.describe; tcg_d128_flat_tma_direct_wide:344 design.work_schedule.phase.combine.describe; tcg_d128_segmented_cp_direct_narrow:386 design.work_schedule.phase.combine.describe; tcg_d128_segmented_cp_direct_wide:400 design.work_schedule.phase.combine.describe; tcg_d128_segmented_tma_direct_narrow:386 design.work_schedule.phase.combine.describe; tcg_d128_segmented_tma_direct_wide:402 design.work_schedule.phase.combine.describe

Merge all weighted splits for one real output row and write O and optional LSE.

## t_b0e286780410dbeb | schedule.describe | 405 chars | 7 declarations

tcg_d128_segmented_cp_direct_narrow:251 design.work_schedule.phase.group_max.describe; tcg_d128_segmented_cp_direct_wide:265 design.work_schedule.phase.group_max.describe; tcg_d128_segmented_cp_xpose_narrow:239 design.work_schedule.phase.group_max.describe; tcg_d128_segmented_tma_direct_narrow:251 design.work_schedule.phase.group_max.describe; tcg_d128_segmented_tma_direct_wide:267 design.work_schedule.phase.group_max.describe; tcg_d128_segmented_tma_xpose_narrow:255 design.work_schedule.phase.group_max.describe; tcg_d128_segmented_tma_xpose_wide:239 design.work_schedule.phase.group_max.describe

Compute GroupMax, including explicit -inf outside useful intervals. Model.forward resolves HostPlan and MapsValid. A cached graph, or an eligible capture/instantiation followed by successful graph launch, executes this same five-kernel sequence. Capture records without executing kernels; otherwise launch_sequence submits directly on the caller stream. The host does not wait for this sequence to finish.

## t_bc775cbf2b7115af | schedule.describe | 68 chars | 1 declarations

tcg_d256_segmented_cp_wide:314 design.work_schedule.phase.segment_max.describe

Produce every SegMax segment independently across rows and segments.

## t_be4b13fc887d3684 | schedule.describe | 64 chars | 1 declarations

tcg_d256_segmented_cp_wide:410 design.work_schedule.phase.combine.describe

Reduce all Ns partials and write O and optional LSE for one row.

## t_ca72c96877e803f7 | schedule.describe | 96 chars | 1 declarations

tcg_d128_fscan_tma_xpose_narrow:322 design.work_schedule.phase.combine.describe

Weight and merge all materialized partials for one real output row and write O and optional LSE.

## t_cd9f28bece0019c1 | schedule.describe | 118 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:366 design.work_schedule.phase.split.describe

Recompute QK, replay the scan, and write FP32 PartA and PartB for one numerator split under the partials step's cases.

## t_d9abcf4f51980626 | schedule.describe | 58 chars | 7 declarations

tcg_d256_flat_cp_narrow:276 design.work_schedule.phase.split.describe; tcg_d256_flat_cp_wide:290 design.work_schedule.phase.split.describe; tcg_d256_flat_tma_narrow:276 design.work_schedule.phase.split.describe; tcg_d256_flat_tma_wide:288 design.work_schedule.phase.split.describe; tcg_d256_segmented_cp_narrow:313 design.work_schedule.phase.split.describe; tcg_d256_segmented_tma_narrow:328 design.work_schedule.phase.split.describe; tcg_d256_segmented_tma_wide:313 design.work_schedule.phase.split.describe

Recompute QK, replay Mscan and write weighted PartA/PartB.

## t_d9e43d41408e442a | schedule.describe | 107 chars | 4 declarations

tcg_d128_fscan_cp_direct_narrow:317 design.work_schedule.phase.combine.describe; tcg_d128_fscan_cp_direct_wide:332 design.work_schedule.phase.combine.describe; tcg_d128_fscan_tma_direct_narrow:317 design.work_schedule.phase.combine.describe; tcg_d128_fscan_tma_direct_wide:333 design.work_schedule.phase.combine.describe

Form weights from PartM and MTail, merge all partials for one real output row and write O and optional LSE.

## t_dbf61fa750e80a65 | schedule.describe | 406 chars | 5 declarations

tcg_d128_fscan_cp_direct_narrow:232 design.work_schedule.phase.group_max.describe; tcg_d128_fscan_cp_direct_wide:246 design.work_schedule.phase.group_max.describe; tcg_d128_fscan_tma_direct_narrow:232 design.work_schedule.phase.group_max.describe; tcg_d128_fscan_tma_direct_wide:248 design.work_schedule.phase.group_max.describe; tcg_d128_fscan_tma_xpose_narrow:237 design.work_schedule.phase.group_max.describe

Compute GroupMax, including explicit -inf outside useful intervals. Model.forward resolves HostPlan and MapsValid. A cached graph, or an eligible capture/instantiation followed by successful graph launch, executes this same three-kernel sequence. Capture records without executing kernels; otherwise launch_sequence submits directly on the caller stream. The host does not wait for this sequence to finish.

## t_dc03466dbc421099 | schedule.describe | 80 chars | 4 declarations

tcg_d256_flat_cp_narrow:254 design.work_schedule.phase.scan.describe; tcg_d256_flat_cp_wide:268 design.work_schedule.phase.scan.describe; tcg_d256_flat_tma_narrow:254 design.work_schedule.phase.scan.describe; tcg_d256_flat_tma_wide:266 design.work_schedule.phase.scan.describe

Compute independent full-length delayed-maximum recurrences for a block of rows.

## t_dd610376c884b6e8 | schedule.describe | 111 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:314 design.work_schedule.phase.segment_max.describe

Reduce each segment's group maxima independently for every row, including segment zero whose summary is unused.

## t_dfb175733db52cad | schedule.describe | 102 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:340 design.work_schedule.phase.segmented_scan.describe

Use SegMax to select replay suffixes independently for each row; publish every MState entry and MTail.

## t_ee7ff95e53051f5a | schedule.describe | 101 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:405 design.work_schedule.phase.combine.describe

Sum all materialized partials for one real row, then write its D BF16 channels and optional FP32 LSE.

## t_f888414f467aa9c0 | schedule.describe | 108 chars | 2 declarations

tc_split_segmented_narrow:400 design.work_schedule.phase.scan.describe; tc_split_segmented_wide:429 design.work_schedule.phase.scan.describe

Fill constant prefixes and replay suffixes for every row, including the region absent from the maximum grid.

## t_01e111cc93a4c586 | schedule.strategy | 111 chars | 4 declarations

tcg_d128_fscan_cp_xpose_narrow:352 design.work_schedule.phase.combine.schedule.strategy; tcg_d128_fscan_cp_xpose_wide:368 design.work_schedule.phase.combine.schedule.strategy; tcg_d128_fscan_tma_xpose_wide:373 design.work_schedule.phase.combine.schedule.strategy; tcg_d128_segmented_cp_xpose_wide:435 design.work_schedule.phase.combine.schedule.strategy

LSE stores occur only with a caller destination. No CTA shares output elements or local resources with another.

## t_044ed26e6c39c00a | schedule.strategy | 149 chars | 3 declarations

tcg_d128_segmented_cp_xpose_narrow:283 design.work_schedule.phase.segment_max.decomposition.strategy; tcg_d128_segmented_tma_xpose_narrow:299 design.work_schedule.phase.segment_max.decomposition.strategy; tcg_d128_segmented_tma_xpose_wide:283 design.work_schedule.phase.segment_max.decomposition.strategy

A tile owns segment z and rows starting at r0=x*256, with SegRows=min(256,R-r0). Groups z*SegLen through min((z+1)*SegLen,T)-1 are a local reduction.

## t_1a78c8aa1f98d006 | schedule.strategy | 343 chars | 1 declarations

tcg_d128_flat_cp_direct_wide:345 design.work_schedule.phase.combine.decomposition.strategy

MergeGroups=4. One row per CTA on grid=(Sq,H,B); worker_id=r=i+Sq*(h+H*b). All Ns partials are consumed locally using combine's split-group intervals, not separate worktiles. HostPlan caps Ns at 128*MergeGroups, so MergeChunk<=128. D128 uses one channel slice and no further reduction. MTail is consumed only when LSE has a caller destination.

## t_1cec673abeb17259 | schedule.strategy | 177 chars | 2 declarations

tc_split_segmented_narrow:390 design.work_schedule.phase.segmax.schedule.strategy; tc_split_segmented_wide:419 design.work_schedule.phase.segmax.schedule.strategy

Grid=(ceil_div(R,ScanBt),Nseg,1), s=blockIdx.y, worker_id=blockIdx.x+ceil_div(R,ScanBt)*s. Each CTA owns one segment and row block, with no cross-CTA reduction or work stealing.

## t_1f36ae40ecd7d318 | schedule.strategy | 528 chars | 4 declarations

tc_split_flat_narrow:288 design.work_schedule.phase.group_max.decomposition.strategy; tc_split_flat_wide:302 design.work_schedule.phase.group_max.decomposition.strategy; tc_split_segmented_narrow:331 design.work_schedule.phase.group_max.decomposition.strategy; tc_split_segmented_wide:360 design.work_schedule.phase.group_max.decomposition.strategy

Br=64, Nm=HostPlan.gm_nsplit. Each (kvflat,a,z) uses kvflat=b*KH+kh and the domain row map. Locally gmLo=L(a,z,Nm), gmHi=U(a,z,Nm). Wgm is the union of the nonempty gmLo..gmHi interval, 0..gLo(a)-1 when z=0, and gHi(a)+1..T-1 when z=Nm-1. The actual Ag grid with Qph rows per head produces only i<Covered. BmCovered excludes the untouched PriorMax region. Every interval, including empty ones, also writes OTemporary and present LseTemporary to the same rows as the other intervals; these overlapping stores are not a reduction.

## t_1ff229b330977910 | schedule.strategy | 235 chars | 4 declarations

tcg_d128_segmented_cp_direct_narrow:295 design.work_schedule.phase.segment_max.decomposition.strategy; tcg_d128_segmented_cp_direct_wide:309 design.work_schedule.phase.segment_max.decomposition.strategy; tcg_d128_segmented_tma_direct_narrow:295 design.work_schedule.phase.segment_max.decomposition.strategy; tcg_d128_segmented_tma_direct_wide:311 design.work_schedule.phase.segment_max.decomposition.strategy

ScanThreads=256 for both scan phases; nr=ceil_div(R,ScanThreads). A tile (z,x) starts at r0=x*ScanThreads, contains ScanRows=min(ScanThreads,R-r0) rows, and traverses g=Seg*z through min(Seg*(z+1),T)-1. Each segment reduction is local.

## t_269c57c9bcea3afe | schedule.strategy | 75 chars | 7 declarations

tcg_d256_flat_cp_narrow:346 design.work_schedule.phase.combine.schedule.strategy; tcg_d256_flat_cp_wide:360 design.work_schedule.phase.combine.schedule.strategy; tcg_d256_flat_tma_narrow:346 design.work_schedule.phase.combine.schedule.strategy; tcg_d256_flat_tma_wide:359 design.work_schedule.phase.combine.schedule.strategy; tcg_d256_segmented_cp_narrow:383 design.work_schedule.phase.combine.schedule.strategy; tcg_d256_segmented_tma_narrow:398 design.work_schedule.phase.combine.schedule.strategy; tcg_d256_segmented_tma_wide:383 design.work_schedule.phase.combine.schedule.strategy

Grid=(Sq,H,B), coordinates (i,h,b), worker_id=row; one worktile per worker.

## t_284cb6d40f6ed3c8 | schedule.strategy | 225 chars | 1 declarations

tcg_d128_fscan_tma_xpose_narrow:324 design.work_schedule.phase.combine.decomposition.strategy

One row per CTA on grid=(Sq,H,B); worker_id=r=i+Sq*(h+H*b). All Ns partials are consumed locally; no further reduction follows. MTail supplies weights for every call and also supplies LSE when a caller destination is present.

## t_29f0511a4d450d02 | schedule.strategy | 302 chars | 4 declarations

tcg_d128_segmented_cp_direct_narrow:321 design.work_schedule.phase.segmented_scan.decomposition.strategy; tcg_d128_segmented_cp_direct_wide:335 design.work_schedule.phase.segmented_scan.decomposition.strategy; tcg_d128_segmented_tma_direct_narrow:321 design.work_schedule.phase.segmented_scan.decomposition.strategy; tcg_d128_segmented_tma_direct_wide:337 design.work_schedule.phase.segmented_scan.decomposition.strategy

Use segment_max's ScanThreads, nr and ScanRows with x=worker_id. A tile scans its rows across all groups. Traverse SegMax from segment one in SegBatch-sized batches and read GroupMax for segment zero and the suffixes defined by scan. These are local traversals; every MState entry and MTail is written.

## t_2b8faa7d5f95e0c6 | schedule.strategy | 88 chars | 3 declarations

tcg_d256_segmented_cp_narrow:283 design.work_schedule.phase.segment_max.schedule.strategy; tcg_d256_segmented_tma_narrow:298 design.work_schedule.phase.segment_max.schedule.strategy; tcg_d256_segmented_tma_wide:283 design.work_schedule.phase.segment_max.schedule.strategy

Grid=(ceil_div(R,ScanRows),Nseg), coordinates (row_tile,f), worker_id=row_tile+grid.x*f.

## t_2e9dd21078b302f2 | schedule.strategy | 185 chars | 3 declarations

tcg_d256_segmented_cp_narrow:266 design.work_schedule.phase.segment_max.decomposition.strategy; tcg_d256_segmented_tma_narrow:281 design.work_schedule.phase.segment_max.decomposition.strategy; tcg_d256_segmented_tma_wide:266 design.work_schedule.phase.segment_max.decomposition.strategy

SegLen=16, SegChunk=16, ScanRows=256. One worktile owns segment f and rows r0=row_tile*ScanRows through min(r0+ScanRows,R)-1, consuming min(SegLen,GCount-f*SegLen) group maxima per row.

## t_35faed86570fdfb7 | schedule.strategy | 427 chars | 1 declarations

tc_split_flat_wide:419 design.work_schedule.phase.combine.decomposition.strategy

One worktile covers r=(b*H+h)*Sq+i and all D channels. The four intervals aLo(f)..aHi(f)-1 remain local to this worktile. For 0<=k<ceil_div(Cb,8), numerator batch k contains aLo(f)+8*k..min(aLo(f)+8*k+8,aHi(f))-1 in each interval, with Nbatch(k)=sum_{f=0}^{3} max(0,min(8,n(f)-8*k)) entries in total. Empty subranges contribute no entries; the final batch contains only remaining splits. The denominator covers all Ns partials.

## t_38c385351e838982 | schedule.strategy | 134 chars | 4 declarations

tc_split_flat_narrow:316 design.work_schedule.phase.group_max.schedule.strategy; tc_split_flat_wide:330 design.work_schedule.phase.group_max.schedule.strategy; tc_split_segmented_narrow:359 design.work_schedule.phase.group_max.schedule.strategy; tc_split_segmented_wide:388 design.work_schedule.phase.group_max.schedule.strategy

Grid=(Ag,B*KH,Nm), worker_id=a+Ag*(kvflat+B*KH*z). Each CTA completes one assigned tile; there is no query remapping or work stealing.

## t_3c16761102fd3ae9 | schedule.strategy | 325 chars | 3 declarations

tcg_d128_segmented_cp_xpose_narrow:309 design.work_schedule.phase.segmented_scan.decomposition.strategy; tcg_d128_segmented_tma_xpose_narrow:325 design.work_schedule.phase.segmented_scan.decomposition.strategy; tcg_d128_segmented_tma_xpose_wide:309 design.work_schedule.phase.segmented_scan.decomposition.strategy

ScanThreads=256. A tile starts at r0=worker_id*ScanThreads and contains ScanRows=min(ScanThreads,R-r0) rows. Each row follows scan's segment batches and selected group walks locally. SegMax traversal begins at segment one; GroupMax supplies segment zero and selected suffixes. Neither traversal creates independent work here.

## t_3dd971502df49d72 | schedule.strategy | 439 chars | 1 declarations

tc_fused:195 design.work_schedule.phase.main.decomposition.strategy

Br=64. Use the domain step's G, Qph, row map and cLo..cHi interval. Each (kvflat,a) tile has kvflat=b*KH+kh and covers G heads with Qph queries per head. The last tile computes padding but writes only i<Sq. All maxima, denominator reductions and channel contractions are local to this worktile. LSE is written only when its destination exists. D/Block QK contractions and D/128 V-scale atoms parameterize both supported channel dimensions.

## t_416727ad9461e638 | schedule.strategy | 934 chars | 1 declarations

tcg_d128_fscan_tma_xpose_wide:256 design.work_schedule.phase.group_max.decomposition.strategy

Br=128, Ns=HostPlan.nsplit, Ngm=HostPlan.gm_nsplit, nx=ceil_div(Sq,Qph), ny=B*KH. Each tile is (s,kvflat=b*KH+kh,a). Use the domain step's row map, U and interval with N=Ngm. In this phase define cLo=L(Ngm,s), cHi=Hlast(Ngm,s), Ng=Count(Ngm,s). WrittenG contains integers cLo<=g<=cHi, also 0<=g<gLoAll for s=0 and gHiAll<g<T for s=Ngm-1; empty ranges contribute nothing. Let Base=nx*ny. Non-PSK gives R>64*Base. In wide's low-row arm, Base<SmCount/32 and the plan gives Ngm>=min(32,floor(T/2)); in its other arm Ngm>=Ns>128. Without a window, scan fusion bounds Span by 160. With a window, the last query tile's span is at most 48 and any earlier tile adds at most ceil_div(Sq-1,128)<=64 groups. Consequently Ng<=5: group-max uses the sequential traversal while retaining its three-slot allocation. Maximum reduction is local to this tile. Empty splits still own their assigned outside-range fills, if any, and execute setup/teardown.

## t_49f7bb80fe1bfcb5 | schedule.strategy | 402 chars | 9 declarations

tcg_d128_flat_cp_direct_narrow:291 design.work_schedule.phase.split.decomposition.strategy; tcg_d128_flat_cp_direct_wide:304 design.work_schedule.phase.split.decomposition.strategy; tcg_d128_flat_tma_direct_narrow:292 design.work_schedule.phase.split.decomposition.strategy; tcg_d128_flat_tma_direct_wide:305 design.work_schedule.phase.split.decomposition.strategy; tcg_d128_flat_tma_xpose_narrow:295 design.work_schedule.phase.split.decomposition.strategy; tcg_d128_segmented_cp_direct_narrow:347 design.work_schedule.phase.split.decomposition.strategy; tcg_d128_segmented_cp_direct_wide:361 design.work_schedule.phase.split.decomposition.strategy; tcg_d128_segmented_tma_direct_narrow:347 design.work_schedule.phase.split.decomposition.strategy; tcg_d128_segmented_tma_direct_wide:363 design.work_schedule.phase.split.decomposition.strategy

Use group_max's Br, Ns, nx and ny and domain's row map. Each tile (s,kvflat,a) sets cLo=L(Ns,s), cHi=Hlast(Ns,s), Ng=Count(Ns,s). Key groups are a local traversal. Empty splits still write zero partials for every valid row. The entrance snapshot is MState[cLo-1,r] when cLo>0; extra discarded reads by padded rows or empty splits are recorded in the pipeline rather than treated as mathematical inputs.

## t_4dcb10953b51b42d | schedule.strategy | 431 chars | 19 declarations

tcg_d128_flat_cp_direct_narrow:227 design.work_schedule.phase.group_max.decomposition.strategy; tcg_d128_flat_cp_direct_wide:240 design.work_schedule.phase.group_max.decomposition.strategy; tcg_d128_flat_cp_xpose_narrow:234 design.work_schedule.phase.group_max.decomposition.strategy; tcg_d128_flat_cp_xpose_wide:217 design.work_schedule.phase.group_max.decomposition.strategy; tcg_d128_flat_tma_direct_narrow:228 design.work_schedule.phase.group_max.decomposition.strategy; tcg_d128_flat_tma_direct_wide:241 design.work_schedule.phase.group_max.decomposition.strategy; tcg_d128_flat_tma_xpose_narrow:231 design.work_schedule.phase.group_max.decomposition.strategy; tcg_d128_flat_tma_xpose_wide:247 design.work_schedule.phase.group_max.decomposition.strategy; tcg_d128_fscan_cp_direct_narrow:237 design.work_schedule.phase.group_max.decomposition.strategy; tcg_d128_fscan_tma_direct_narrow:237 design.work_schedule.phase.group_max.decomposition.strategy; tcg_d128_fscan_tma_direct_wide:253 design.work_schedule.phase.group_max.decomposition.strategy; tcg_d128_fscan_tma_xpose_narrow:242 design.work_schedule.phase.group_max.decomposition.strategy; tcg_d128_segmented_cp_direct_narrow:256 design.work_schedule.phase.group_max.decomposition.strategy; tcg_d128_segmented_cp_direct_wide:270 design.work_schedule.phase.group_max.decomposition.strategy; tcg_d128_segmented_cp_xpose_narrow:244 design.work_schedule.phase.group_max.decomposition.strategy; tcg_d128_segmented_tma_direct_narrow:256 design.work_schedule.phase.group_max.decomposition.strategy; tcg_d128_segmented_tma_direct_wide:272 design.work_schedule.phase.group_max.decomposition.strategy; tcg_d128_segmented_tma_xpose_narrow:260 design.work_schedule.phase.group_max.decomposition.strategy; tcg_d128_segmented_tma_xpose_wide:244 design.work_schedule.phase.group_max.decomposition.strategy

Br=128, Ns=HostPlan.nsplit, Ngm=HostPlan.gm_nsplit, nx=ceil_div(Sq,Qph), ny=B*KH. Each tile is (s,kvflat=b*KH+kh,a). Use domain's row map and set cLo=L(Ngm,s), cHi=Hlast(Ngm,s), Ng=Count(Ngm,s). WrittenG contains integers cLo<=g<=cHi, also 0<=g<gLo for s=0 and gHi<g<T for s=Ngm-1; empty ranges add nothing. Long means Ng>=8. Maxima are local reductions. Empty splits still perform their assigned outside fills, setup and teardown.

## t_4ed36d0535dbcbfe | schedule.strategy | 160 chars | 7 declarations

tcg_d256_flat_cp_narrow:309 design.work_schedule.phase.split.schedule.strategy; tcg_d256_flat_cp_wide:323 design.work_schedule.phase.split.schedule.strategy; tcg_d256_flat_tma_narrow:309 design.work_schedule.phase.split.schedule.strategy; tcg_d256_flat_tma_wide:321 design.work_schedule.phase.split.schedule.strategy; tcg_d256_segmented_cp_narrow:346 design.work_schedule.phase.split.schedule.strategy; tcg_d256_segmented_tma_narrow:361 design.work_schedule.phase.split.schedule.strategy; tcg_d256_segmented_tma_wide:346 design.work_schedule.phase.split.schedule.strategy

Grid=(ceil_div(Sq,Qph),B*KH,Ns), coordinates (a,kvflat,s), worker_id=a+grid.x*(kvflat+grid.y*s), lpt=0. No cross-split communication is used inside this kernel.

## t_527964d14dcbc176 | schedule.strategy | 257 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:394 design.work_schedule.phase.split.schedule.strategy

Physical grid=(nx,ny,Ns) and worker_id=x+nx*(y+ny*z) owns (s=z,kvflat=y,a=x), without CTA remapping. MState also supplies the entrance value at cLo-1 when cLo>0; the pipeline records extra padding/empty-split loads separately from these mathematical inputs.

## t_5682d4be6c2b351b | schedule.strategy | 198 chars | 2 declarations

tcg_d128_fscan_cp_direct_narrow:319 design.work_schedule.phase.combine.decomposition.strategy; tcg_d128_fscan_tma_direct_narrow:319 design.work_schedule.phase.combine.decomposition.strategy

One row per CTA on grid=(Sq,H,B); worker_id=r=i+Sq*(h+H*b). All Ns partials are consumed locally; no further reduction follows. MTail is consumed for weights even when LSE has no caller destination.

## t_59fa2623a1f319bc | schedule.strategy | 88 chars | 20 declarations

tcg_d128_flat_cp_direct_narrow:320 design.work_schedule.phase.split.schedule.strategy; tcg_d128_flat_cp_direct_wide:333 design.work_schedule.phase.split.schedule.strategy; tcg_d128_flat_cp_xpose_narrow:327 design.work_schedule.phase.split.schedule.strategy; tcg_d128_flat_cp_xpose_wide:310 design.work_schedule.phase.split.schedule.strategy; tcg_d128_flat_tma_direct_narrow:321 design.work_schedule.phase.split.schedule.strategy; tcg_d128_flat_tma_direct_wide:334 design.work_schedule.phase.split.schedule.strategy; tcg_d128_flat_tma_xpose_narrow:324 design.work_schedule.phase.split.schedule.strategy; tcg_d128_flat_tma_xpose_wide:340 design.work_schedule.phase.split.schedule.strategy; tcg_d128_fscan_cp_direct_narrow:307 design.work_schedule.phase.split.schedule.strategy; tcg_d128_fscan_cp_direct_wide:322 design.work_schedule.phase.split.schedule.strategy; tcg_d128_fscan_tma_direct_narrow:307 design.work_schedule.phase.split.schedule.strategy; tcg_d128_fscan_tma_direct_wide:323 design.work_schedule.phase.split.schedule.strategy; tcg_d128_fscan_tma_xpose_narrow:312 design.work_schedule.phase.split.schedule.strategy; tcg_d128_segmented_cp_direct_narrow:376 design.work_schedule.phase.split.schedule.strategy; tcg_d128_segmented_cp_direct_wide:390 design.work_schedule.phase.split.schedule.strategy; tcg_d128_segmented_cp_xpose_narrow:364 design.work_schedule.phase.split.schedule.strategy; tcg_d128_segmented_tma_direct_narrow:376 design.work_schedule.phase.split.schedule.strategy; tcg_d128_segmented_tma_direct_wide:392 design.work_schedule.phase.split.schedule.strategy; tcg_d128_segmented_tma_xpose_narrow:380 design.work_schedule.phase.split.schedule.strategy; tcg_d128_segmented_tma_xpose_wide:364 design.work_schedule.phase.split.schedule.strategy

Grid=(nx,ny,Ns); worker_id=x+nx*(y+ny*z) owns (s=z,kvflat=y,a=x), without CTA remapping.

## t_5a3ac4f70725500e | schedule.strategy | 156 chars | 1 declarations

tc_split_flat_wide:448 design.work_schedule.phase.combine.schedule.strategy

Grid=(Sq,H,B), worker_id=i+Sq*(h+H*b). Each CTA completes exactly one row without D slicing; local intervals advance independently before their final merge.

## t_5c986390241b2268 | schedule.strategy | 444 chars | 2 declarations

fused_d256_cp:191 design.work_schedule.phase.main.decomposition.strategy; fused_d256_tma:191 design.work_schedule.phase.main.decomposition.strategy

Br=128. Use the domain step's G, Qph, row map and cLo..cHi interval. Each worktile is (kvflat=b*KH+kh, a), covering G heads and Qph query positions. Its last query tile is padded for computation but targets only i<Sq. The two key-half maxima combine within each group and the two carried denominators combine after the last group, all within this worktile. LSE is produced only when its caller destination exists; no worktile reduction follows.

## t_5e54074c6560cda0 | schedule.strategy | 361 chars | 2 declarations

tc_split_segmented_narrow:372 design.work_schedule.phase.segmax.decomposition.strategy; tc_split_segmented_wide:401 design.work_schedule.phase.segmax.decomposition.strategy

ScanBt=256. Each worktile owns segment s and rows r0..min(r0+ScanBt,R)-1, r0=ScanBt*blockIdx.x. Its group interval is s*Seg..min((s+1)*Seg,T)-1. Cblock and Ublock are the owned rows whose query i=r mod Sq is respectively below Covered or at least Covered. The two views supply disjoint rows of the original maximum table; this pass does not initialize PriorMax.

## t_6113d49727c9f16c | schedule.strategy | 185 chars | 10 declarations

tcg_d128_flat_cp_direct_narrow:332 design.work_schedule.phase.combine.decomposition.strategy; tcg_d128_flat_cp_xpose_narrow:339 design.work_schedule.phase.combine.decomposition.strategy; tcg_d128_flat_cp_xpose_wide:322 design.work_schedule.phase.combine.decomposition.strategy; tcg_d128_flat_tma_direct_narrow:333 design.work_schedule.phase.combine.decomposition.strategy; tcg_d128_flat_tma_xpose_narrow:336 design.work_schedule.phase.combine.decomposition.strategy; tcg_d128_segmented_cp_direct_narrow:388 design.work_schedule.phase.combine.decomposition.strategy; tcg_d128_segmented_cp_xpose_narrow:376 design.work_schedule.phase.combine.decomposition.strategy; tcg_d128_segmented_tma_direct_narrow:388 design.work_schedule.phase.combine.decomposition.strategy; tcg_d128_segmented_tma_xpose_narrow:392 design.work_schedule.phase.combine.decomposition.strategy; tcg_d128_segmented_tma_xpose_wide:376 design.work_schedule.phase.combine.decomposition.strategy

One row per CTA on grid=(Sq,H,B); worker_id=r=i+Sq*(h+H*b). All Ns partials are consumed locally; no further reduction follows. MTail is consumed only when LSE has a caller destination.

## t_643b4b90cf0b40d9 | schedule.strategy | 357 chars | 1 declarations

tcg_d128_flat_tma_xpose_wide:352 design.work_schedule.phase.combine.decomposition.strategy

SP=4. One row per CTA on grid=(Sq,H,B); worker_id=r=i+Sq*(h+H*b), without channel slicing. All Ns partials are consumed locally using combine's SP intervals; these groups are not independent worktiles. HostPlan caps Ns at 128*SP, meeting the denominator coverage bound. No further reduction follows. MTail is consumed only when LSE has a caller destination.

## t_6474a974fea8ca78 | schedule.strategy | 249 chars | 7 declarations

tcg_d256_flat_cp_narrow:243 design.work_schedule.phase.group_max.schedule.strategy; tcg_d256_flat_cp_wide:257 design.work_schedule.phase.group_max.schedule.strategy; tcg_d256_flat_tma_narrow:243 design.work_schedule.phase.group_max.schedule.strategy; tcg_d256_flat_tma_wide:255 design.work_schedule.phase.group_max.schedule.strategy; tcg_d256_segmented_cp_narrow:253 design.work_schedule.phase.group_max.schedule.strategy; tcg_d256_segmented_tma_narrow:268 design.work_schedule.phase.group_max.schedule.strategy; tcg_d256_segmented_tma_wide:253 design.work_schedule.phase.group_max.schedule.strategy

Grid=(ceil_div(Sq,Qph),B*KH,Nm), coordinates (a,kvflat,s), kvflat=b*KH+kh. worker_id=a+grid.x*(kvflat+grid.y*s). This path keeps lpt=0. Each worker handles one worktile, including empty spans; assignment implies neither SM placement nor concurrency.

## t_6645f9906b199420 | schedule.strategy | 118 chars | 1 declarations

tcg_d128_fscan_cp_xpose_narrow:323 design.work_schedule.phase.combine.decomposition.strategy

One row per CTA, with grid=(Sq,H,B), r=i+Sq*(h+H*b). All Ns splits are consumed locally; no further reduction follows.

## t_6e4bb8fa142c5f27 | schedule.strategy | 90 chars | 3 declarations

tcg_d128_segmented_cp_xpose_narrow:298 design.work_schedule.phase.segment_max.schedule.strategy; tcg_d128_segmented_tma_xpose_narrow:314 design.work_schedule.phase.segment_max.schedule.strategy; tcg_d128_segmented_tma_xpose_wide:298 design.work_schedule.phase.segment_max.schedule.strategy

Grid=(ceil_div(R,256),Nseg); worker_id=x+ceil_div(R,256)*z owns row block x and segment z.

## t_6eec502ad571902d | schedule.strategy | 414 chars | 4 declarations

tc_split_flat_narrow:359 design.work_schedule.phase.split.decomposition.strategy; tc_split_flat_wide:373 design.work_schedule.phase.split.decomposition.strategy; tc_split_segmented_narrow:435 design.work_schedule.phase.split.decomposition.strategy; tc_split_segmented_wide:464 design.work_schedule.phase.split.decomposition.strategy

Each (kvflat,a,z) traverses cLo=L(a,z,Ns)..cHi=U(a,z,Ns). Q is reused across groups and K/V across grouped heads. ScanState provides current and preceding states; g=0 uses -inf for the predecessor. Padded rows use rho(u), clamping their scratch query coordinate to zero, and produce no global result. Empty intervals still stage Q and write zero partials. The discarded cLo-1 entry read is tracked in the pipeline.

## t_6f1f311c3e9fee16 | schedule.strategy | 291 chars | 2 declarations

tc_split_flat_narrow:405 design.work_schedule.phase.combine.decomposition.strategy; tc_split_segmented_narrow:481 design.work_schedule.phase.combine.decomposition.strategy

CombineBt=128. One worktile covers r=(b*H+h)*Sq+i and all D channels. Ns<=CombineBt lets one cooperative load cover the whole denominator vector. Numerator batches preserve ascending split order; batch base s0=8*k for k=0,...,ceil_div(Ns,8)-1. The final batch pads missing splits with zeros.

## t_6f2aa3c72c1d3647 | schedule.strategy | 351 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:342 design.work_schedule.phase.segmented_scan.decomposition.strategy

ScanThreads=256. A worktile covers rows r0=worker_id*ScanThreads through min(r0+ScanThreads,R)-1; let ScanRows=min(ScanThreads,R-r0). Each row follows scan's segment batches and conditional group walks locally. SegMax traversal begins at segment one; GroupMax supplies segment zero and each selected suffix. Neither axis creates independent work here.

## t_709360cff23a2a27 | schedule.strategy | 524 chars | 3 declarations

tcg_d128_fscan_cp_xpose_narrow:242 design.work_schedule.phase.group_max.decomposition.strategy; tcg_d128_fscan_cp_xpose_wide:256 design.work_schedule.phase.group_max.decomposition.strategy; tcg_d128_segmented_cp_xpose_wide:276 design.work_schedule.phase.group_max.decomposition.strategy

Br=128, Ns=HostPlan.nsplit, Ngm=HostPlan.gm_nsplit, nx=ceil_div(Sq,Qph), ny=B*KH. Each tile is (s,kvflat=b*KH+kh,a). Use the domain step's row map, U and interval with N=Ngm. In this phase define cLo=L(Ngm,s), cHi=Hlast(Ngm,s), Ng=Count(Ngm,s). WrittenG contains integers cLo<=g<=cHi, also 0<=g<gLoAll for s=0 and gHiAll<g<T for s=Ngm-1; empty ranges contribute nothing. Long means Ng>=8. Maximum reduction is local to this tile. Empty splits still own their assigned outside-range fills, if any, and execute setup/teardown.

## t_7444af4732ef7050 | schedule.strategy | 783 chars | 2 declarations

fused_d128_cp:222 design.work_schedule.phase.main.schedule.strategy; root:432 design.work_schedule.phase.main.schedule.strategy

Let nx=ceil_div(Sq,Qph), ny=B*KH. Use CtaCount and SmCount from meta_data.dispatch.when. Physical grid=(nx,ny,1); worker_id=d=x+nx*y. The following bijection defines (a(d),k(d)) and WorkId(d)=k(d)*nx+a(d), the logical ascending worktile number. Without a causal/window mask, a(d)=d mod nx, k(d)=floor(d/nx). With a mask and CtaCount>2*SmCount, let r=CtaCount-1-d: a(d)=floor(r/ny), k(d)=r mod ny (source lpt=1). With a mask and SmCount<CtaCount<=2*SmCount (source lpt=2): if d<SmCount, a(d)=nx-1-floor(d/ny), k(d)=d mod ny; otherwise let r=d-SmCount, a(d)=floor(r/ny), k(d)=ny-1-(r mod ny). This is a work assignment, not a guarantee of SM placement or simultaneous execution. Each worker ends after its single tile; different CTAs share no recurrence or writable intermediate state.

## t_76d5d7ee81f836a7 | schedule.strategy | 406 chars | 1 declarations

fused_d128_key:198 design.work_schedule.phase.main.decomposition.strategy

Br=128. Use the domain step's G, Qph, row map and cLo..cHi interval. Each worktile is (kvflat=b*KH+kh, a), covering G heads and Qph query positions. Its last query tile is padded for computation but targets only i<Sq. Both the per-group key-slice maximum exchange and the final DenSlice sum are local to this worktile. LSE is produced only when its caller destination exists; no worktile reduction follows.

## t_76df0db53ab17f58 | schedule.strategy | 469 chars | 3 declarations

tcg_d128_fscan_cp_xpose_narrow:282 design.work_schedule.phase.split.decomposition.strategy; tcg_d128_fscan_cp_xpose_wide:296 design.work_schedule.phase.split.decomposition.strategy; tcg_d128_fscan_tma_xpose_wide:302 design.work_schedule.phase.split.decomposition.strategy

Use the group_max phase's Br, Ns, nx, ny and the domain step's head-major row map. Each tile (s,kvflat,a) sets cLo=L(Ns,s), cHi=Hlast(Ns,s), Ng=Count(Ns,s). Every split first walks groups 0..min(cLo,T)-1, then advances the scan alongside matrix groups cLo..cHi. Empty splits still perform the bounded prefix and write all three partials, with the partials step's PartA limitation. TailRows=U for s=Ns-1, otherwise 0; this assigns each MTail row to exactly one worktile.

## t_7725acae067fc6d8 | schedule.strategy | 541 chars | 1 declarations

tcg_d128_fscan_tma_xpose_narrow:281 design.work_schedule.phase.split.decomposition.strategy

Use group_max's Br, Ns, nx and ny and domain's row map. Each tile (s,kvflat,a) sets cLo=L(Ns,s), cHi=Hlast(Ns,s), Ng=Count(Ns,s). Walk groups 0..min(cLo,T)-1, then advance the scan with matrix groups cLo..cHi. Empty splits still perform the bounded prefix and write partials under the partials step's limitation. TailRows=U for s=Ns-1, otherwise 0. GroupMax reads use domain's r, including padded rows' first-head-row reads. Their distinct row count is ScanRows=U+G if U<Br and q0>0, otherwise U; the padded results never become output rows.

## t_77b5052639459b47 | schedule.strategy | 208 chars | 3 declarations

tcg_d128_fscan_cp_xpose_narrow:310 design.work_schedule.phase.split.schedule.strategy; tcg_d128_fscan_cp_xpose_wide:324 design.work_schedule.phase.split.schedule.strategy; tcg_d128_fscan_tma_xpose_wide:330 design.work_schedule.phase.split.schedule.strategy

Physical grid=(nx,ny,Ns) and worker_id=x+nx*(y+ny*z) owns (s=z,kvflat=y,a=x), without CTA remapping. GroupMax supplies both the prefix and matrix-loop maxima; no split reads MTail or another split's partials.

## t_7817101c1727f8ed | schedule.strategy | 240 chars | 1 declarations

tcg_d256_flat_tma_wide:333 design.work_schedule.phase.combine.decomposition.strategy

A worktile covers row=(b*H+h)*Sq+i and all D channels. SP=4; the host clamps Ns<=128*SP, so Chunk<=128. The output step's SP split intervals are local traversals, not extra worktiles. There is no channel slicing; Mtail is used only for LSE.

## t_7b4b204c60cc3b18 | schedule.strategy | 385 chars | 1 declarations

tcg_d256_segmented_cp_wide:343 design.work_schedule.phase.scan.decomposition.strategy

A worktile covers rows starting at r0=worker_id*ScanRows and the entire T-group range. Segment 0 is replayed directly; subsequent local iterations c test SegChunk segment maxima and fill or replay the chunk as defined by scan. f0=1+c*SegChunk. GroupMax reads cover the initial segment and each selected suffix only. SegMax[0] is not consumed; every Mscan and Mtail element is produced.

## t_7ca3a5c8ed1cd60e | schedule.strategy | 258 chars | 8 declarations

tcg_d128_flat_cp_direct_narrow:266 design.work_schedule.phase.flat_scan.decomposition.strategy; tcg_d128_flat_cp_direct_wide:279 design.work_schedule.phase.flat_scan.decomposition.strategy; tcg_d128_flat_cp_xpose_narrow:273 design.work_schedule.phase.flat_scan.decomposition.strategy; tcg_d128_flat_cp_xpose_wide:256 design.work_schedule.phase.flat_scan.decomposition.strategy; tcg_d128_flat_tma_direct_narrow:267 design.work_schedule.phase.flat_scan.decomposition.strategy; tcg_d128_flat_tma_direct_wide:280 design.work_schedule.phase.flat_scan.decomposition.strategy; tcg_d128_flat_tma_xpose_narrow:270 design.work_schedule.phase.flat_scan.decomposition.strategy; tcg_d128_flat_tma_xpose_wide:286 design.work_schedule.phase.flat_scan.decomposition.strategy

ScanThreads=256 when T<128. Otherwise select the first value in [32,64,128] satisfying ceil_div(R,ScanThreads)<=SmCount, or 256 if none. A tile starts at r0=worker_id*ScanThreads and contains ScanRows=min(ScanThreads,R-r0) rows. Groups are a local traversal.

## t_814fceb8bedf17c5 | schedule.strategy | 397 chars | 1 declarations

tcg_d256_segmented_cp_wide:369 design.work_schedule.phase.split.decomposition.strategy

Use the domain row map and n=Ns, l=L(Ns,s), e=Hlast(Ns,s), C=max(0,e-l+1). Q is reused through this local group interval; nonempty partitions read their entry state at l-1 when l>0 and current states follow g; the discarded empty-partition read is recorded in the pipeline. Mtail is read only for nonempty valid-row partial weighting. Empty worktiles still write zero partials for every valid row.

## t_8849a8abef559f80 | schedule.strategy | 64 chars | 4 declarations

tcg_d128_segmented_cp_direct_narrow:310 design.work_schedule.phase.segment_max.schedule.strategy; tcg_d128_segmented_cp_direct_wide:324 design.work_schedule.phase.segment_max.schedule.strategy; tcg_d128_segmented_tma_direct_narrow:310 design.work_schedule.phase.segment_max.schedule.strategy; tcg_d128_segmented_tma_direct_wide:326 design.work_schedule.phase.segment_max.schedule.strategy

Grid=(nr,Nseg); worker_id=x+nr*z owns one segment and row block.

## t_8b86a50968c38070 | schedule.strategy | 380 chars | 1 declarations

tcg_d128_segmented_tma_direct_wide:404 design.work_schedule.phase.combine.decomposition.strategy

One row per CTA on grid=(Sq,H,B); worker_id=r=i+Sq*(h+H*b). MergeGroups=4 local groups consume combine's disjoint split intervals, each covering all D channels. HostPlan caps Ns at 128*MergeGroups. These intervals are intra-CTA reductions within one worktile; all Ns partials are consumed and no further reduction follows. MTail is consumed only when LSE has a caller destination.

## t_8bbcef2db01ac610 | schedule.strategy | 419 chars | 4 declarations

tcg_d128_fscan_cp_direct_narrow:277 design.work_schedule.phase.split.decomposition.strategy; tcg_d128_fscan_cp_direct_wide:292 design.work_schedule.phase.split.decomposition.strategy; tcg_d128_fscan_tma_direct_narrow:277 design.work_schedule.phase.split.decomposition.strategy; tcg_d128_fscan_tma_direct_wide:293 design.work_schedule.phase.split.decomposition.strategy

Use group_max's Br, Ns, nx and ny and domain's row map. Each tile (s,kvflat,a) sets cLo=L(Ns,s), cHi=Hlast(Ns,s), Ng=Count(Ns,s). Read GroupMax in a local prefix t=0..Tw(s)-1 before the local g=cLo..cHi attention traversal. Empty splits still scan the prefix and write PartM and zero partials. TailRows=U for s=Ns-1, otherwise 0; even an empty last split owns every valid tail. No split consumes another split's output.

## t_8d92d92a6f6389c2 | schedule.strategy | 357 chars | 2 declarations

fused_d128_group:222 design.work_schedule.phase.main.schedule.strategy; fused_d128_key:225 design.work_schedule.phase.main.schedule.strategy

Let nx=ceil_div(Sq,Qph), ny=B*KH. Physical grid=(nx,ny,1), worker_id=x+nx*y, a=x and kvflat=y. Both host remapping gates are false because CtaCount<=SmCount, including masked calls (lpt=0). Each CTA ends after its single tile; the grid order gives no SM placement or simultaneous-execution guarantee. CTAs share no recurrence or writable intermediate state.

## t_916e53fe994699fe | schedule.strategy | 197 chars | 4 declarations

tcg_d256_flat_cp_narrow:256 design.work_schedule.phase.scan.decomposition.strategy; tcg_d256_flat_cp_wide:270 design.work_schedule.phase.scan.decomposition.strategy; tcg_d256_flat_tma_narrow:256 design.work_schedule.phase.scan.decomposition.strategy; tcg_d256_flat_tma_wide:268 design.work_schedule.phase.scan.decomposition.strategy

ScanRows=256 when GCount<128; otherwise take the first b in {32,64,128} with ceil_div(R,b)<=SmCount, or 256 if none qualifies. r0=worker_id*ScanRows. A worktile traverses all GCount groups locally.

## t_94029c39bccbe215 | schedule.strategy | 348 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:407 design.work_schedule.phase.combine.decomposition.strategy

MergeGroups=4. One row per CTA, with grid=(Sq,H,B), r=i+Sq*(h+H*b). All Ns splits are consumed locally using combine's contiguous intervals; they do not create independent worktiles. HostPlan caps Ns at 128*MergeGroups, so each interval has at most 128 splits. The final channel range is the full D; no channel slicing or further reduction follows.

## t_945e0aade4e013d2 | schedule.strategy | 328 chars | 1 declarations

tcg_d128_fscan_tma_direct_wide:335 design.work_schedule.phase.combine.decomposition.strategy

One row per CTA on grid=(Sq,H,B); worker_id=r=i+Sq*(h+H*b). MergeGroups intervals are local subdivisions of this worktile, and their final sum is local. D=128 uses the whole channel range in one CTA. HostPlan caps Ns at 512, so each interval has at most 128 splits. MTail is consumed for weights even without an LSE destination.

## t_99e69f90f7bc23d2 | schedule.strategy | 386 chars | 7 declarations

tcg_d256_flat_cp_narrow:278 design.work_schedule.phase.split.decomposition.strategy; tcg_d256_flat_cp_wide:292 design.work_schedule.phase.split.decomposition.strategy; tcg_d256_flat_tma_narrow:278 design.work_schedule.phase.split.decomposition.strategy; tcg_d256_flat_tma_wide:290 design.work_schedule.phase.split.decomposition.strategy; tcg_d256_segmented_cp_narrow:315 design.work_schedule.phase.split.decomposition.strategy; tcg_d256_segmented_tma_narrow:330 design.work_schedule.phase.split.decomposition.strategy; tcg_d256_segmented_tma_wide:315 design.work_schedule.phase.split.decomposition.strategy

Use n=Ns, l=L(Ns,s), e=End(Ns,s), C=max(0,e-l+1) and the domain row map. VRows has the group_max definition for the same (b,kh,a). Q is invariant over the local group interval. Nonempty work uses entry state l-1 when l>0, then states g; Mtail supplies the final weight. The discarded empty-partition read is recorded in replay. Empty work still writes zero partials for every valid row.

## t_9a61b3a6b4705197 | schedule.strategy | 74 chars | 3 declarations

tc_split_flat_narrow:432 design.work_schedule.phase.combine.schedule.strategy; tc_split_segmented_narrow:508 design.work_schedule.phase.combine.schedule.strategy; tc_split_segmented_wide:541 design.work_schedule.phase.combine.schedule.strategy

Grid=(Sq,H,B), worker_id=i+Sq*(h+H*b). Each CTA completes exactly one row.

## t_9aa3fe840b7b7591 | schedule.strategy | 175 chars | 1 declarations

tcg_d256_segmented_cp_wide:440 design.work_schedule.phase.combine.schedule.strategy

Grid=(Sq,H,B), coordinates (i,h,b), worker_id=row. Each CTA owns its entire output row without channel slicing or cross-CTA combination. Initial output bytes are not operands.

## t_a07710a627bea075 | schedule.strategy | 202 chars | 1 declarations

tcg_d256_segmented_cp_wide:316 design.work_schedule.phase.segment_max.decomposition.strategy

SegLen=16, SegChunk=16 and ScanRows=256. A worktile covers segment f and rows beginning at r0=row_tile*ScanRows; it consumes the segment's min(SegLen,T-f*SegLen) groups. There is no cross-row reduction.

## t_a14f6ddf67467f32 | schedule.strategy | 387 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:368 design.work_schedule.phase.split.decomposition.strategy

Use the group_max phase's Br, Ns, nx, ny and the domain step's head-major row map. Each tile (s,kvflat,a) sets cLo=L(Ns,s), cHi=Hlast(Ns,s), Ng=Count(Ns,s). Each nonempty split scans g=cLo..cHi locally; reductions across keys stay within the CTA. Empty splits have no matrix iteration but still write PartA and PartB for all U valid rows, with the partials step's empty-split limitation.

## t_a1a8ed8eecf0f5dc | schedule.strategy | 456 chars | 1 declarations

tcg_d128_fscan_cp_direct_wide:251 design.work_schedule.phase.group_max.decomposition.strategy

Br=128, Ns=HostPlan.nsplit, Ngm=HostPlan.gm_nsplit, nx=ceil_div(Sq,Qph), ny=B*KH. HostPlan clamps Ns<=512. Each tile is (s,kvflat=b*KH+kh,a). Use domain's row map and set cLo=L(Ngm,s), cHi=Hlast(Ngm,s), Ng=Count(Ngm,s). WrittenG contains integers cLo<=g<=cHi, also 0<=g<gLo for s=0 and gHi<g<T for s=Ngm-1; empty ranges add nothing. Long means Ng>=8. Maxima are local reductions. Empty splits still perform their assigned outside fills, setup and teardown.

## t_a416fb9dd1f2af61 | schedule.strategy | 255 chars | 1 declarations

tcg_d128_fscan_cp_direct_wide:334 design.work_schedule.phase.combine.decomposition.strategy

One row per CTA on grid=(Sq,H,B); worker_id=r=i+Sq*(h+H*b). All Ns partials are consumed by combine's grouped local reduction. D128 has no channel slicing or later cross-CTA reduction. MTail is consumed for weights even when LSE has no caller destination.

## t_a57cbf06c3b35c4b | schedule.strategy | 99 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:331 design.work_schedule.phase.segment_max.schedule.strategy

Physical grid=(ceil_div(R,256),Nseg); worker_id=x+ceil_div(R,256)*z owns row block x and segment z.

## t_aa1f332dbd955b81 | schedule.strategy | 590 chars | 1 declarations

tc_split_segmented_wide:510 design.work_schedule.phase.combine.decomposition.strategy

CombineGroups=4 and CombineBt=128. One worktile covers r=(b*H+h)*Sq+i, all D channels and all Ns partials; the fixed build does not slice D across CTAs. HostPlan caps Ns at CombineGroups*CombineBt, so each group's denominator fits its cooperative load. Each sg independently traverses sLo(sg)..sHi(sg)-1 within this worktile. Numerator batch k has s0=sLo(sg)+8*k, k=0,...,ceil_div(Count(sg),8)-1. Full batches contain eight splits; a partial final batch is consumed one split at a time without padding. Group partials and the final sums remain local, with no extra grid or work acquisition.

## t_aaf133f5a26a5805 | schedule.strategy | 354 chars | 1 declarations

tcg_d128_fscan_cp_xpose_wide:337 design.work_schedule.phase.combine.decomposition.strategy

One row per CTA, with grid=(Sq,H,B), r=i+Sq*(h+H*b). MergeGroups=4 partitions the split interval inside each worktile; all D channels remain in this CTA and no further reduction follows. HostPlan caps Ns at 128*MergeGroups, so Chunk<=128 and the local denominator reduction covers every split. This is the actual host cap, not an added input restriction.

## t_ab0ab9ec58a08d13 | schedule.strategy | 628 chars | 2 declarations

fused_d256_cp:219 design.work_schedule.phase.main.schedule.strategy; fused_d256_tma:219 design.work_schedule.phase.main.schedule.strategy

Let nx=ceil_div(Sq,Qph), ny=B*KH. Use CtaCount and SmCount from meta_data.dispatch.when. Physical grid=(nx,ny,1); worker_id=d=x+nx*y. The following bijection defines (a(d),k(d)) and WorkId(d)=k(d)*nx+a(d), the logical ascending worktile number. Without a causal/window mask or when CtaCount<=SmCount, a(d)=d mod nx, k(d)=floor(d/nx). With a mask and CtaCount>SmCount, let r=CtaCount-1-d: a(d)=floor(r/ny), k(d)=r mod ny (source lpt=1). This is a work assignment, not a guarantee of SM placement or simultaneous execution. Each worker ends after its single tile; different CTAs share no recurrence or writable intermediate state.

## t_af3aec7f7d6d469a | schedule.strategy | 350 chars | 7 declarations

tcg_d256_flat_cp_narrow:218 design.work_schedule.phase.group_max.decomposition.strategy; tcg_d256_flat_cp_wide:232 design.work_schedule.phase.group_max.decomposition.strategy; tcg_d256_flat_tma_narrow:218 design.work_schedule.phase.group_max.decomposition.strategy; tcg_d256_flat_tma_wide:230 design.work_schedule.phase.group_max.decomposition.strategy; tcg_d256_segmented_cp_narrow:228 design.work_schedule.phase.group_max.decomposition.strategy; tcg_d256_segmented_tma_narrow:243 design.work_schedule.phase.group_max.decomposition.strategy; tcg_d256_segmented_tma_wide:228 design.work_schedule.phase.group_max.decomposition.strategy

Br=128, Ns=HostPlan.nsplit, Nm=HostPlan.gm_nsplit, PSK=(2*Sq*G<=128*ceil_div(Sq,Qph)). Use n=Nm, l=L(Nm,s), e=End(Nm,s), C=max(0,e-l+1), VRows=G*min(Qph,Sq-q0). GMGroups comprises l..e, plus 0..gLo-1 when s=0 and gHi+1..GCount-1 when s=Nm-1. Across splits these disjoint regions cover every GroupMax entry, including cases with empty terminal splits.

## t_b123958e3705d41d | schedule.strategy | 293 chars | 1 declarations

tcg_d128_flat_tma_direct_wide:346 design.work_schedule.phase.combine.decomposition.strategy

MergeGroups=4. One row per CTA on grid=(Sq,H,B); worker_id=r=i+Sq*(h+H*b). The host caps Ns at 512, so Cmerge<=128. The combine step's groups partition all Ns splits within this worktile. D128 has no channel slicing or cross-CTA merge; MTail is consumed only when LSE has a caller destination.

## t_b13a1ffaf7786f7a | schedule.strategy | 245 chars | 1 declarations

tc_fused:223 design.work_schedule.phase.main.schedule.strategy

Physical grid=(ceil_div(Sq,Qph),B*KH,1), with x=a and y=kvflat. worker_id=x+ceil_div(Sq,Qph)*y is the logical worktile number. Each CTA finishes its one assigned tile; masks do not remap CTAs. Different CTAs share no writable intermediate state.

## t_b2206ded7a0cba16 | schedule.strategy | 178 chars | 6 declarations

tcg_d256_flat_cp_narrow:321 design.work_schedule.phase.combine.decomposition.strategy; tcg_d256_flat_cp_wide:335 design.work_schedule.phase.combine.decomposition.strategy; tcg_d256_flat_tma_narrow:321 design.work_schedule.phase.combine.decomposition.strategy; tcg_d256_segmented_cp_narrow:358 design.work_schedule.phase.combine.decomposition.strategy; tcg_d256_segmented_tma_narrow:373 design.work_schedule.phase.combine.decomposition.strategy; tcg_d256_segmented_tma_wide:358 design.work_schedule.phase.combine.decomposition.strategy

A worktile covers row=(b*H+h)*Sq+i and all D channels. Narrow selection guarantees Ns<=128. Mtail is used only for LSE. The local split traversal has s0=8*j, 0<=j<ceil_div(Ns,8).

## t_bb5ccf1527d6ef83 | schedule.strategy | 164 chars | 1 declarations

tcg_d256_segmented_cp_wide:400 design.work_schedule.phase.split.schedule.strategy

Grid=(ceil_div(Sq,Qph),B*KH,Ns), coordinates (a,kvflat,s), worker_id=a+grid.x*(kvflat+grid.y*s), with lpt=0. No cross-split communication occurs inside this kernel.

## t_c6f4010d16a6eedc | schedule.strategy | 81 chars | 4 declarations

tc_split_flat_narrow:349 design.work_schedule.phase.scan.schedule.strategy; tc_split_flat_wide:363 design.work_schedule.phase.scan.schedule.strategy; tc_split_segmented_narrow:425 design.work_schedule.phase.scan.schedule.strategy; tc_split_segmented_wide:454 design.work_schedule.phase.scan.schedule.strategy

Grid=(ceil_div(R,ScanBt),1,1), worker_id=blockIdx.x. Each CTA owns one row block.

## t_c712803062c2a75c | schedule.strategy | 407 chars | 1 declarations

fused_d128_group:195 design.work_schedule.phase.main.decomposition.strategy

Br=128. Use the domain step's G, Qph, row map and cLo..cHi interval. Each worktile is (kvflat=b*KH+kh, a), covering G heads and Qph query positions. Its last query tile is padded for computation but targets only i<Sq. The entire useful-group traversal and its carried states remain within this worktile; Ng is positive. LSE is produced only when its caller destination exists; no worktile reduction follows.

## t_c8a7d617fb0da9d5 | schedule.strategy | 423 chars | 1 declarations

tcg_d256_segmented_cp_wide:412 design.work_schedule.phase.combine.decomposition.strategy

SP=4. Each worktile is one row=(b*H+h)*Sq+i, with all D channels and all Ns splits. The host caps Ns at SP*128, so Cp<=128 and every denominator split has a contributor. The output step's SP intervals are local reductions within this one worktile. PartA and PartB describe its full immutable input; the numerator reduction streams each interval in ascending split order. Mtail is consumed only when the caller supplies LSE.

## t_cc6ea8d52c210d25 | schedule.strategy | 123 chars | 4 declarations

tc_split_flat_narrow:392 design.work_schedule.phase.split.schedule.strategy; tc_split_flat_wide:406 design.work_schedule.phase.split.schedule.strategy; tc_split_segmented_narrow:468 design.work_schedule.phase.split.schedule.strategy; tc_split_segmented_wide:497 design.work_schedule.phase.split.schedule.strategy

Grid=(Aq,B*KH,Ns), worker_id=a+Aq*(kvflat+B*KH*z). All contractions and the four-part denominator merge belong to this CTA.

## t_d3703fb4f52fe50b | schedule.strategy | 450 chars | 6 declarations

tcg_d128_flat_cp_xpose_narrow:298 design.work_schedule.phase.split.decomposition.strategy; tcg_d128_flat_cp_xpose_wide:281 design.work_schedule.phase.split.decomposition.strategy; tcg_d128_flat_tma_xpose_wide:311 design.work_schedule.phase.split.decomposition.strategy; tcg_d128_segmented_cp_xpose_narrow:335 design.work_schedule.phase.split.decomposition.strategy; tcg_d128_segmented_tma_xpose_narrow:351 design.work_schedule.phase.split.decomposition.strategy; tcg_d128_segmented_tma_xpose_wide:335 design.work_schedule.phase.split.decomposition.strategy

Use group_max's Br, Ns, nx and ny and domain's row map. Each tile (s,kvflat,a) sets cLo=L(Ns,s), cHi=Hlast(Ns,s), Ng=Count(Ns,s). Key groups are a local traversal. Empty splits still write partials for every valid row, with the partials step's materialization limitation. The entrance snapshot is MState[cLo-1,r] when cLo>0; extra discarded reads by padded rows or empty splits are recorded in the pipeline rather than treated as mathematical inputs.

## t_d90d242df600eb35 | schedule.strategy | 415 chars | 2 declarations

fused_d128_cp:195 design.work_schedule.phase.main.decomposition.strategy; root:405 design.work_schedule.phase.main.decomposition.strategy

Br=128. Use the domain step's G, Qph, row map and cLo..cHi interval. Each worktile is (kvflat=b*KH+kh, a), covering G heads and Qph query positions. Its last query tile is padded for computation but targets only i<Sq. All reductions, including the two score-half maxima and denominator partial sums, are local to this worktile. LSE is produced only when its caller destination exists; no worktile reduction follows.

## t_dc71b4577a2733f6 | schedule.strategy | 413 chars | 2 declarations

tc_split_flat_narrow:329 design.work_schedule.phase.scan.decomposition.strategy; tc_split_flat_wide:343 design.work_schedule.phase.scan.decomposition.strategy

ScanBt=256 if T<128; otherwise choose the smallest value in {32,64,128} satisfying ceil_div(R,ScanBt)<=SmCount, or 256 if none qualifies. The worktile contains r0..min(r0+ScanBt,R)-1, r0=ScanBt*blockIdx.x. Cblock and Ublock are the subsets with query coordinate i=r mod Sq respectively below Covered or at least Covered. The ascending group loop consumes disjoint row views of the same original [group,row] array.

## t_e625cff920afb0ba | schedule.strategy | 125 chars | 1 declarations

tcg_d256_segmented_cp_wide:333 design.work_schedule.phase.segment_max.schedule.strategy

Grid=(ceil_div(R,ScanRows),Nseg), coordinates (row_tile,f), worker_id=row_tile+grid.x*f. Every worker processes one worktile.

## t_e6d03c5bd46609f4 | schedule.strategy | 438 chars | 2 declarations

tc_split_segmented_narrow:402 design.work_schedule.phase.scan.decomposition.strategy; tc_split_segmented_wide:431 design.work_schedule.phase.scan.decomposition.strategy

The worktile contains r0..min(r0+ScanBt,R)-1, r0=ScanBt*blockIdx.x. Cblock and Ublock are the subsets with query coordinate i=r mod Sq respectively below Covered or at least Covered. After the unconditional first-segment walk, chunk c has sb=1+c*Sc. Its summaries select the suffix that rereads group maxima; all groups receive ScanState writes. The ascending chunk loop and each row's ascending suffix walk remain local to this worktile.

## t_e90290a01c8f7ae9 | schedule.strategy | 168 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:316 design.work_schedule.phase.segment_max.decomposition.strategy

Each tile owns segment z and rows r0=x*256 through min(r0+256,R)-1, with SegRows=min(256,R-r0). The groups z*SegLen through min((z+1)*SegLen,T)-1 are a local reduction.

## t_eef6120362f5109d | schedule.strategy | 123 chars | 4 declarations

tcg_d128_fscan_cp_xpose_narrow:270 design.work_schedule.phase.group_max.schedule.strategy; tcg_d128_fscan_cp_xpose_wide:284 design.work_schedule.phase.group_max.schedule.strategy; tcg_d128_fscan_tma_xpose_wide:290 design.work_schedule.phase.group_max.schedule.strategy; tcg_d128_segmented_cp_xpose_wide:304 design.work_schedule.phase.group_max.schedule.strategy

Physical grid=(nx,ny,Ngm); worker_id=x+nx*(y+ny*z) owns (s=z,kvflat=y,a=x). p.lpt=0: no mask-dependent remapping or refill.

## t_f373703fc54ebf5d | schedule.strategy | 261 chars | 3 declarations

tcg_d256_segmented_cp_narrow:291 design.work_schedule.phase.scan.decomposition.strategy; tcg_d256_segmented_tma_narrow:306 design.work_schedule.phase.scan.decomposition.strategy; tcg_d256_segmented_tma_wide:291 design.work_schedule.phase.scan.decomposition.strategy

r0=worker_id*ScanRows. Each worktile covers all GCount groups for its rows. It replays segment 0, then visits chunks c with f0=1+c*SegChunk, filling or replaying as defined by scan. GroupMax reads cover only segment 0 and selected suffixes; SegMax[0] is unused.

## t_f546ae298e341530 | schedule.strategy | 226 chars | 1 declarations

tcg_d128_fscan_tma_xpose_wide:343 design.work_schedule.phase.combine.decomposition.strategy

Cg=4. One row per CTA, with grid=(Sq,H,B), r=i+Sq*(h+H*b). The combine step's Cg contiguous split intervals are consumed locally and merged within this worktile; they are not independent worktiles. D is not sliced across CTAs.

## t_f5ce52df60af9c7e | schedule.strategy | 109 chars | 20 declarations

tcg_d128_flat_cp_direct_narrow:254 design.work_schedule.phase.group_max.schedule.strategy; tcg_d128_flat_cp_direct_wide:267 design.work_schedule.phase.group_max.schedule.strategy; tcg_d128_flat_cp_xpose_narrow:261 design.work_schedule.phase.group_max.schedule.strategy; tcg_d128_flat_cp_xpose_wide:244 design.work_schedule.phase.group_max.schedule.strategy; tcg_d128_flat_tma_direct_narrow:255 design.work_schedule.phase.group_max.schedule.strategy; tcg_d128_flat_tma_direct_wide:268 design.work_schedule.phase.group_max.schedule.strategy; tcg_d128_flat_tma_xpose_narrow:258 design.work_schedule.phase.group_max.schedule.strategy; tcg_d128_flat_tma_xpose_wide:274 design.work_schedule.phase.group_max.schedule.strategy; tcg_d128_fscan_cp_direct_narrow:264 design.work_schedule.phase.group_max.schedule.strategy; tcg_d128_fscan_cp_direct_wide:279 design.work_schedule.phase.group_max.schedule.strategy; tcg_d128_fscan_tma_direct_narrow:264 design.work_schedule.phase.group_max.schedule.strategy; tcg_d128_fscan_tma_direct_wide:280 design.work_schedule.phase.group_max.schedule.strategy; tcg_d128_fscan_tma_xpose_narrow:269 design.work_schedule.phase.group_max.schedule.strategy; tcg_d128_segmented_cp_direct_narrow:283 design.work_schedule.phase.group_max.schedule.strategy; tcg_d128_segmented_cp_direct_wide:297 design.work_schedule.phase.group_max.schedule.strategy; tcg_d128_segmented_cp_xpose_narrow:271 design.work_schedule.phase.group_max.schedule.strategy; tcg_d128_segmented_tma_direct_narrow:283 design.work_schedule.phase.group_max.schedule.strategy; tcg_d128_segmented_tma_direct_wide:299 design.work_schedule.phase.group_max.schedule.strategy; tcg_d128_segmented_tma_xpose_narrow:287 design.work_schedule.phase.group_max.schedule.strategy; tcg_d128_segmented_tma_xpose_wide:271 design.work_schedule.phase.group_max.schedule.strategy

Grid=(nx,ny,Ngm); worker_id=x+nx*(y+ny*z) owns (s=z,kvflat=y,a=x). p.lpt=0, without mask-dependent remapping.

## t_f7f491cd502b5c4f | schedule.strategy | 422 chars | 1 declarations

tcg_d256_segmented_cp_wide:277 design.work_schedule.phase.group_max.decomposition.strategy

Br=128. PSK=(2*Sq*G<=128*ceil_div(Sq,Qph)); its fixed-default padding guard selects the domain row map. Use n=Nm and s as the split coordinate, with l=L(Nm,s), e=Hlast(Nm,s), C=max(0,e-l+1). Define GMGroups={g | l<=g<=e}, additionally including [0,gLo) when s=0 and (gHi,T) when s=Nm-1. These disjoint regions cover [0,T) once per valid row even when the last split is empty. V and probabilities are absent from this pass.

## t_f9d01606282d765f | schedule.strategy | 255 chars | 1 declarations

tcg_d256_segmented_cp_wide:303 design.work_schedule.phase.group_max.schedule.strategy

Grid=(ceil_div(Sq,Qph),B*KH,Nm), coordinates (a,kvflat,s), kvflat=b*KH+kh. worker_id=a+grid.x*(kvflat+grid.y*s). The split sequence leaves lpt=0 and uses these grid coordinates directly. Each worker processes one worktile and exits, including empty spans.

## t_fc1030504eeb1719 | schedule.strategy | 362 chars | 1 declarations

tcg_d128_segmented_cp_direct_wide:402 design.work_schedule.phase.combine.decomposition.strategy

One row per CTA on grid=(Sq,H,B); worker_id=r=i+Sq*(h+H*b). MergeGroups=4; the intervals a_z..b_z-1 and their final merge are local to this worktile. HostPlan clamps Ns<=128*MergeGroups, so Cmerge<=128 and every denominator entry is covered. D=128 uses one channel slice and no further reduction follows. MTail is consumed only when LSE has a caller destination.

## t_02c8942b617563cc | schedule.thesis | 344 chars | 2 declarations

fused_d256_cp:170 design.work_schedule.thesis; fused_d256_tma:170 design.work_schedule.thesis

Give each CTA all grouped heads for one query tile and the complete useful key interval, reusing Q across groups and K/V across those heads. This eliminates split-state traffic and the combine kernel, at the cost of one sequential group recurrence per CTA; mask-dependent CTA remapping spreads long and short work without dynamic work claiming.

## t_038abd6bbbc2794f | schedule.thesis | 238 chars | 1 declarations

tcg_d128_segmented_tma_direct_wide:246 design.work_schedule.thesis

Allocate independent CTAs along the useful key span while reusing Q within each CTA and K/V across grouped heads. Group-max, segment summaries, row replay, numerator splits and wide merge are five submissions linked by global FP32 values.

## t_0958c22a4aee73bd | schedule.thesis | 217 chars | 1 declarations

tc_split_segmented_wide:337 design.work_schedule.thesis

Five static grids exchange FP32 intermediates to parallelize value intervals. A row-by-segment maximum grid precedes row-wise replay; each wide-combine CTA owns one row and splits its partials among four local groups.

## t_134e89d7fbed038c | schedule.thesis | 274 chars | 1 declarations

tcg_d128_fscan_cp_xpose_narrow:222 design.work_schedule.thesis

Split the useful key interval across independent CTAs while reusing Q and sharing K/V across grouped heads. Group-max, split-local scans with QK/PV, and weighted narrow merge form three submissions; repeated prefixes and FP32 partial traffic pay for parallel numerator work.

## t_201e935b6466dfc6 | schedule.thesis | 340 chars | 1 declarations

tcg_d128_flat_cp_direct_wide:217 design.work_schedule.thesis

Allocate independent CTAs along the useful key span while reusing Q within each CTA and K/V across grouped heads. Group-max, ordered row scan, numerator splits and wide merge are four submissions linked by global FP32 values. Four local split groups shorten the merge's serial numerator chains at the cost of a shared exchange and CTA join.

## t_22e6764092b49132 | schedule.thesis | 196 chars | 1 declarations

tc_split_flat_narrow:267 design.work_schedule.thesis

Four static grids exchange FP32 intermediates to parallelize value intervals. QK is recomputed, the flat scan retains a serial chain per row, and narrow combine assigns one CTA to each output row.

## t_268db8259433f0d7 | schedule.thesis | 274 chars | 1 declarations

tcg_d128_fscan_cp_direct_wide:230 design.work_schedule.thesis

Allocate independent CTAs along the useful key span, reusing Q within each CTA and K/V across grouped heads. Group-max, locally scanning numerator splits and weighted wide merge are three submissions; prefix walks repeat to avoid global per-group scan-state materialization.

## t_3ba104aceebd0f7d | schedule.thesis | 339 chars | 1 declarations

fused_d128_group:173 design.work_schedule.thesis

Give each CTA all grouped heads for one query tile and the complete useful key interval, reusing Q across groups and K/V across those heads. This eliminates split-state traffic and a separate combine kernel, at the cost of a sequential group recurrence per CTA. This selection has at most SmCount CTAs and uses natural query/KV grid order.

## t_3f223506b322e38d | schedule.thesis | 221 chars | 1 declarations

tc_split_segmented_narrow:308 design.work_schedule.thesis

Five static grids exchange FP32 intermediates to parallelize value intervals. A row-by-segment maximum grid precedes row-wise replay, trading another submission for fewer replay reads; narrow combine owns one row per CTA.

## t_3f5d90204517e2fc | schedule.thesis | 369 chars | 1 declarations

fused_d128_key:176 design.work_schedule.thesis

Give each CTA all grouped heads for one query tile and the complete useful key interval, reusing Q across groups and K/V across those heads. This eliminates split-state traffic and a separate combine kernel, at the cost of a sequential group recurrence per CTA and local key-slice exchanges. This selection has at most SmCount CTAs and uses natural query/KV grid order.

## t_41f5e37e283f81bd | schedule.thesis | 240 chars | 2 declarations

tcg_d128_segmented_cp_direct_narrow:230 design.work_schedule.thesis; tcg_d128_segmented_tma_direct_narrow:230 design.work_schedule.thesis

Allocate independent CTAs along the useful key span while reusing Q within each CTA and K/V across grouped heads. Group-max, segment summaries, row replay, numerator splits and narrow merge are five submissions linked by global FP32 values.

## t_5636c37f57dec166 | schedule.thesis | 242 chars | 3 declarations

tcg_d128_segmented_cp_xpose_narrow:219 design.work_schedule.thesis; tcg_d128_segmented_tma_xpose_narrow:235 design.work_schedule.thesis; tcg_d128_segmented_tma_xpose_wide:219 design.work_schedule.thesis

Allocate independent CTAs along the useful key span while reusing Q within each CTA and K/V across grouped heads. Group-max, segment-max, ordered row replay, numerator splits and narrow merge are five submissions linked by global FP32 values.

## t_5b953010aced632f | schedule.thesis | 487 chars | 1 declarations

tcg_d128_segmented_cp_xpose_wide:250 design.work_schedule.thesis

Split the useful key interval across independent CTAs while each CTA reuses Q and shares K/V across grouped heads. Separate group-max, segment-max, row replay and wide merge supply the dependencies that make numerator splits independent; segment-parallel reduction adds a launch and global FP32 storage while letting row replay avoid unchanged prefix loads. Each merge CTA parallelizes its split reduction across four intervals and locally combines their partials before writing the row.

## t_600eb2e0970ba493 | schedule.thesis | 296 chars | 1 declarations

tcg_d256_segmented_cp_wide:246 design.work_schedule.thesis

Split the useful key interval independently for group maxima and PV, then join through parallel segment maxima, a serial scan per row and one wide CTA per output row. Extra QK work and global partials buy more compute CTAs; no work is dynamically claimed and no CTA ordering implies SM placement.

## t_75be4e54a52a68fa | schedule.thesis | 338 chars | 2 declarations

fused_d128_cp:174 design.work_schedule.thesis; root:384 design.work_schedule.thesis

Give each CTA all grouped heads for one query tile and the complete useful key interval, reusing Q across groups and K/V across those heads. This eliminates split-state traffic and combine work, at the cost of one sequential group recurrence per CTA; mask-dependent CTA remapping spreads long and short work without dynamic work claiming.

## t_a2a220753c48bfda | schedule.thesis | 319 chars | 1 declarations

tcg_d128_flat_tma_xpose_wide:224 design.work_schedule.thesis

Allocate independent CTAs along the useful key span while reusing Q within each CTA and K/V across grouped heads. Group-max, ordered row scan, numerator splits and wide merge are four submissions linked by global FP32 values. Four merge groups shorten local split walks at the cost of shared partials and a CTA barrier.

## t_a6ae7605b4a2b896 | schedule.thesis | 212 chars | 3 declarations

tcg_d256_segmented_cp_narrow:200 design.work_schedule.thesis; tcg_d256_segmented_tma_narrow:215 design.work_schedule.thesis; tcg_d256_segmented_tma_wide:200 design.work_schedule.thesis

Independent key partitions increase compute CTA count, with a segmented row scan and narrow combine trading extra launches and intermediate traffic for shorter local numerator chains. Work is statically assigned.

## t_b1a27e6b4d9e6022 | schedule.thesis | 235 chars | 1 declarations

tcg_d128_fscan_tma_xpose_narrow:222 design.work_schedule.thesis

Allocate independent key-split CTAs that reuse Q and share K/V across grouped heads. Group-max, split-local scan with QK/PV, and weighted narrow merge form three submissions; repeated prefixes replace the separate global snapshot pass.

## t_b4dc1ed3393d7eb3 | schedule.thesis | 211 chars | 1 declarations

tcg_d256_flat_tma_wide:204 design.work_schedule.thesis

Independent key partitions increase compute CTA count, with a full per-row scan and a wide combine trading extra launches and intermediate traffic for shorter local numerator chains. Work is statically assigned.

## t_b85569a8a66a361b | schedule.thesis | 272 chars | 1 declarations

tcg_d128_fscan_cp_xpose_wide:236 design.work_schedule.thesis

Split the useful key interval across independent CTAs while reusing Q and sharing K/V across grouped heads. Group-max, split-local scans with QK/PV, and weighted wide merge form three submissions; repeated prefixes and FP32 partial traffic pay for parallel numerator work.

## t_c0c48cbba2b64113 | schedule.thesis | 308 chars | 1 declarations

tcg_d128_fscan_tma_xpose_wide:236 design.work_schedule.thesis

Split the useful key interval across independent CTAs while reusing Q and sharing K/V across grouped heads. Group-max, split-local scans with QK/PV, and weighted wide merge form three submissions; the merge divides each row's splits among four local groups at the cost of shared partials and their reduction.

## t_c1c76ef0492842b7 | schedule.thesis | 269 chars | 1 declarations

tc_fused:177 design.work_schedule.thesis

Assign one CTA all grouped heads for a query tile and its complete useful key interval. Q is reused across groups and K/V across heads; the full recurrence remains local, avoiding split workspace and combine work at the cost of a sequential key-group traversal per CTA.

## t_c241b414b330f1eb | schedule.thesis | 280 chars | 1 declarations

tcg_d128_fscan_tma_direct_wide:232 design.work_schedule.thesis

Allocate independent CTAs along the useful key span, reusing Q within each CTA and K/V across grouped heads. Group-max, locally scanning numerator splits and weighted four-group merge are three submissions; prefix walks repeat to avoid global per-group scan-state materialization.

## t_c28db67b26c517f0 | schedule.thesis | 286 chars | 1 declarations

tc_split_flat_wide:280 design.work_schedule.thesis

Four static grids exchange FP32 intermediates to parallelize value intervals. QK is recomputed, the flat scan retains a serial chain per row, and wide combine merges four local split intervals within each output-row CTA, trading shorter local sums for shared partials and a final merge.

## t_d65151586a61dc3b | schedule.thesis | 227 chars | 5 declarations

tcg_d128_flat_cp_direct_narrow:205 design.work_schedule.thesis; tcg_d128_flat_cp_xpose_narrow:212 design.work_schedule.thesis; tcg_d128_flat_cp_xpose_wide:195 design.work_schedule.thesis; tcg_d128_flat_tma_direct_narrow:206 design.work_schedule.thesis; tcg_d128_flat_tma_xpose_narrow:209 design.work_schedule.thesis

Allocate independent CTAs along the useful key span while reusing Q within each CTA and K/V across grouped heads. Group-max, ordered row scan, numerator splits and narrow merge are four submissions linked by global FP32 values.

## t_e51e3affa36ff83b | schedule.thesis | 353 chars | 1 declarations

tcg_d128_segmented_cp_direct_wide:243 design.work_schedule.thesis

Allocate independent CTAs along the useful key span while reusing Q within each CTA and K/V across grouped heads. Group-max, segment summaries, row replay, numerator splits and wide merge are five submissions linked by global FP32 values. Wide merge divides each row's split traversal into four local intervals, paying for a shared partial-sum exchange.

## t_ead0b40b74769fd4 | schedule.thesis | 276 chars | 2 declarations

tcg_d128_fscan_cp_direct_narrow:216 design.work_schedule.thesis; tcg_d128_fscan_tma_direct_narrow:216 design.work_schedule.thesis

Allocate independent CTAs along the useful key span, reusing Q within each CTA and K/V across grouped heads. Group-max, locally scanning numerator splits and weighted narrow merge are three submissions; prefix walks repeat to avoid global per-group scan-state materialization.

## t_ed415b852054bcc2 | schedule.thesis | 213 chars | 3 declarations

tcg_d256_flat_cp_narrow:192 design.work_schedule.thesis; tcg_d256_flat_cp_wide:206 design.work_schedule.thesis; tcg_d256_flat_tma_narrow:192 design.work_schedule.thesis

Independent key partitions increase compute CTA count, with a full per-row scan and a narrow combine trading extra launches and intermediate traffic for shorter local numerator chains. Work is statically assigned.

## t_f59892fef2f754ae | schedule.thesis | 321 chars | 1 declarations

tcg_d128_flat_tma_direct_wide:218 design.work_schedule.thesis

Allocate independent CTAs along the useful key span while reusing Q within each CTA and K/V across grouped heads. Group-max, ordered row scan, numerator splits and grouped merge are four submissions linked by global FP32 values; the merge divides splits locally and exchanges partial sums before one group writes the row.
