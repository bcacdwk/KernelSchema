# template_gdn_prefill: full open-text inventory

These are exact-text groups, not proof of semantic equivalence across contexts.

## t_09eb3c457258c03d | algorithm_contract.gap | 160 chars | 1 declarations

root:53 contract.algorithm_requirement.golden_reference.gap

The oracle evaluates the recurrence with FP32 block_size=32 inversion and log-space decay; that blocked evaluation defines the floating-point comparison values.

## t_328abd3390325a8a | algorithm_contract.math | 806 chars | 1 declarations

root:33 contract.algorithm_requirement.compute[0].math

All tensors are contiguous CUDA tensors. Before key normalization, q/k/v rows have independent Normal(0,0.05) centers and Uniform(center-0.25,center+0.25) elements rounded to BF16. Independently for each 0<=b<B and 0<=h<HV, let hq=floor(h/(HV/HQ)), L_b=cu_seqlens[b+1]-cu_seqlens[b]>0 and State(-1)=initial_state[b,h]. For 0<=t<L_b, x=cu_seqlens[b]+t, 0<=j<DV and 0<=d<DK: D=(alpha[x,h]+epsilon)*State(t-1). delta[j]=beta[x,h]*(v[x,h,j]-sum_d D[j,d]*k[x,hq,d]). State(t)[j,d]=D[j,d]+delta[j]*k[x,hq,d]. O[x,h,j]=bf16(scale*sum_d State(t)[j,d]*q[x,hq,d]). S_final[b,h]=State(L_b-1); only output and output_state are mutated. Timing includes Model.forward and its int64-to-int32 boundary conversion; input generation, normalization, continuation-prefix evaluation, allocation and NaN fill are outside timing.

## t_02228a924ff40f4e | algorithm_contract.meaning | 16 chars | 1 declarations

root:15 contract.algorithm_requirement.constant.DV.meaning

Value dimension.

## t_238bbb50b1f21d01 | algorithm_contract.meaning | 35 chars | 1 declarations

root:25 contract.algorithm_requirement.input.scale.meaning

Mandatory Python output multiplier.

## t_33bf0b815560f8e4 | algorithm_contract.meaning | 44 chars | 1 declarations

root:47 contract.algorithm_requirement.compute[0].symbol.State.meaning

Updated reference state after local token t.

## t_448cee0c44d458bc | algorithm_contract.meaning | 27 chars | 1 declarations

root:24 contract.algorithm_requirement.input.cu_seqlens.meaning

Packed sequence boundaries.

## t_48e86097184ce227 | algorithm_contract.meaning | 7 chars | 1 declarations

root:20 contract.algorithm_requirement.input.v.meaning

Values.

## t_4c9d80558e36cf51 | algorithm_contract.meaning | 37 chars | 1 declarations

root:23 contract.algorithm_requirement.input.initial_state.meaning

Explicit incoming value-by-key state.

## t_56a2488c70b8ff61 | algorithm_contract.meaning | 14 chars | 1 declarations

root:14 contract.algorithm_requirement.constant.DK.meaning

Key dimension.

## t_69db0dca0637b672 | algorithm_contract.meaning | 27 chars | 1 declarations

root:13 contract.algorithm_requirement.constant.HV.meaning

Value and state head count.

## t_6ae617e7af900e94 | algorithm_contract.meaning | 27 chars | 1 declarations

root:26 contract.algorithm_requirement.input.output.meaning

Preallocated output buffer.

## t_7897b9c05e6a6412 | algorithm_contract.meaning | 60 chars | 1 declarations

root:29 contract.algorithm_requirement.output.O.meaning

First returned tensor aliasing and fully overwriting output.

## t_7f64c16fca084b63 | algorithm_contract.meaning | 20 chars | 1 declarations

root:19 contract.algorithm_requirement.input.k.meaning

Pre-normalized keys.

## t_9106e71a77a1e6f2 | algorithm_contract.meaning | 32 chars | 1 declarations

root:27 contract.algorithm_requirement.input.output_state.meaning

Preallocated final-state buffer.

## t_b5c49c3a57a8ea60 | algorithm_contract.meaning | 67 chars | 1 declarations

root:30 contract.algorithm_requirement.output.S_final.meaning

Second returned tensor aliasing and fully overwriting output_state.

## t_c9c4342f956bf0f5 | algorithm_contract.meaning | 23 chars | 1 declarations

root:22 contract.algorithm_requirement.input.beta.meaning

Delta-update strengths.

## t_dfe8ee431d8b0f73 | algorithm_contract.meaning | 25 chars | 1 declarations

root:12 contract.algorithm_requirement.constant.HQ.meaning

Query and key head count.

## t_e49b2328ca2a64bc | algorithm_contract.meaning | 21 chars | 1 declarations

root:21 contract.algorithm_requirement.input.alpha.meaning

Linear decay factors.

## t_ec06ba8ce47a66c9 | algorithm_contract.meaning | 21 chars | 1 declarations

root:18 contract.algorithm_requirement.input.q.meaning

Unnormalized queries.

## t_fecf230a6f4d2abd | algorithm_contract.meaning | 37 chars | 1 declarations

root:16 contract.algorithm_requirement.constant.epsilon.meaning

Additive stabilizer for linear decay.

## t_044937ae45a516d3 | algorithm_contract.range | 13 chars | 1 declarations

root:22 contract.algorithm_requirement.input.beta.range

0 <= beta < 1

## t_0588c07a723593e9 | algorithm_contract.range | 116 chars | 1 declarations

root:23 contract.algorithm_requirement.input.initial_state.range

All zeros or the final state of an independent 32-token reference prefix per sequence under the same input contract.

## t_54366254e437de83 | algorithm_contract.range | 14 chars | 1 declarations

root:21 contract.algorithm_requirement.input.alpha.range

0 <= alpha < 1

## t_5485d7ce1226bf07 | algorithm_contract.range | 89 chars | 1 declarations

root:19 contract.algorithm_requirement.input.k.range

BF16 result of torch.nn.functional.normalize(p=2.0, dim=-1) on the BF16 generator output.

## t_b63ee7c06b98f0ed | algorithm_contract.range | 66 chars | 1 declarations

root:24 contract.algorithm_requirement.input.cu_seqlens.range

Starts at 0, strictly increases, ends at T; 1<=B<=16 and T<=16384.

## t_b958fd734bd04fee | algorithm_contract.range | 60 chars | 2 declarations

root:26 contract.algorithm_requirement.input.output.range; root:27 contract.algorithm_requirement.input.output_state.range

Arbitrary contents including NaN; previous contents ignored.

## t_c32d83714413cf4b | algorithm_contract.range | 64 chars | 1 declarations

root:25 contract.algorithm_requirement.input.scale.range

Python binary64 1.0/math.sqrt(DK), equal to 0.08838834764831843.

## t_cd916003912abd86 | contract.compute_name | 51 chars | 1 declarations

root:32 contract.algorithm_requirement.compute[0].name

Decay and update the state before inclusive readout

## t_5c985cbceafd5785 | contract.tolerance | 90 chars | 1 declarations

root:51 contract.algorithm_requirement.golden_reference.tolerance.O

atol=0.01, rtol=0.01; every element of both the returned tensor and mutated_inputs.output.

## t_6164115ce2fe99a1 | contract.tolerance | 98 chars | 1 declarations

root:52 contract.algorithm_requirement.golden_reference.tolerance.S_final

atol=0.005, rtol=0.001; every element of both the returned tensor and mutated_inputs.output_state.

## t_0f7376c7514cf6c5 | dispatch.md | 122 chars | 1 declarations

dispatch.md:15 dispatch.md

树与 `meta_data.dispatch.parent` 对应；两份设计按各自的 `meta_data.dispatch.when` 选择，共同使用 root 的 Contract。修订号见各节点的 `meta_data.version`。

## t_3f7cb032ce65ad8b | dispatch.md | 29 chars | 1 declarations

dispatch.md:10 dispatch.md

| 节点 | Workload 与相对父节点的设计变化 |

## t_73df9924b3c14109 | dispatch.md | 88 chars | 1 declarations

dispatch.md:13 dispatch.md

| [m64_single_sequence](m64_single_sequence.yaml) | 单序列；固定加载边界，使用四个 chunk 的标量预取与对应缓冲协议 |

## t_779345bc7e6772ef | dispatch.md | 70 chars | 1 declarations

dispatch.md:17 dispatch.md

m64 表示每个 CTA 处理 64 行 value/state。长度对齐与 zero/continuation 初态模式均留在所属节点内。

## t_848563f239f8b32a | dispatch.md | 56 chars | 1 declarations

dispatch.md:5 dispatch.md

```mermaid flowchart TD root --> m64_single_sequence ```

## t_886460849560fe48 | dispatch.md | 99 chars | 1 declarations

dispatch.md:3 dispatch.md

本任务以支持打包多序列、逐序列更新加载边界的设计作为 root；单序列从中派生，使用固定边界和不同的标量预取流水线。完整 Contract 只在 `root.yaml` 中定义，覆盖单序列与多序列。

## t_8c547aad00c18067 | dispatch.md | 20 chars | 1 declarations

dispatch.md:1 dispatch.md

# gdn_prefill schema

## t_9bf940e6b6fec2d5 | dispatch.md | 58 chars | 1 declarations

dispatch.md:12 dispatch.md

| [root](root.yaml) | 打包多序列；逐序列更新加载边界，标量处理与输出工作共用一个 warp |

## t_89bb4b6e5937114b | dispatch.when | 24 chars | 1 declarations

root:8 meta_data.dispatch.when

Select exactly when B>1.

## t_973d554715a852a3 | dispatch.when | 24 chars | 1 declarations

m64_single_sequence:9 meta_data.dispatch.when

Select exactly when B=1.

## t_680d6642a8764eac | hardware.execution | 55 chars | 1 declarations

root:59 contract.hardware_specification.execution.CTA

Warps co-resident on one SM, sharing SMEM and barriers.

## t_6f5eff329130bc7c | hardware.execution | 63 chars | 1 declarations

root:60 contract.hardware_specification.execution.cluster

CTAs co-scheduled within one GPC, with cluster synchronization.

## t_707ed1a5b6536df5 | hardware.execution | 31 chars | 1 declarations

root:58 contract.hardware_specification.execution.warp

32 SIMT threads within one CTA.

## t_263d5b133f5e61d6 | hardware.feature | 56 chars | 1 declarations

root:64 contract.hardware_specification.feature.register

Thread-private registers backed by the SM register file.

## t_6c67314fe3205270 | hardware.feature | 61 chars | 1 declarations

root:63 contract.hardware_specification.feature.TMEM

Explicitly allocated Tensor Core storage, separate from SMEM.

## t_bf65cfd621166b87 | hardware.feature | 101 chars | 1 declarations

root:65 contract.hardware_specification.feature.asynchronous_execution

Asynchronous transfers and MMA require their primitive-specific completion and visibility guarantees.

## t_fbe55b4f6cea5249 | hardware.feature | 88 chars | 1 declarations

root:62 contract.hardware_specification.feature.SMEM

CTA-lifetime shared storage, addressable by cluster peers while its owner remains alive.

## t_4f201fb09921413f | hardware.stack | 14 chars | 1 declarations

root:67 contract.hardware_specification.stack.language

CuTe DSL 4.6.1

## t_7e4016ec26766b60 | hardware.stack | 135 chars | 1 declarations

root:68 contract.hardware_specification.stack.toolchain

CuTe DSL 4.6.1 bundled compiler; torch 2.9.0+cu130; cuda-python 13.1.1. CUDA compiler and driver versions were not recorded separately.

## t_18bdcc1dac780fe8 | math.arithmetic | 82 chars | 2 declarations

m64_single_sequence:37 design.math_reshape.rewrite[0].arithmetic.accumulate; root:102 design.math_reshape.rewrite[0].arithmetic.accumulate

Scan each 32-token half in FP32, then add the first-half total to the second half.

## t_51298e81989315e3 | math.arithmetic | 192 chars | 2 declarations

m64_single_sequence:36 design.math_reshape.rewrite[0].arithmetic.approx; root:101 design.math_reshape.rewrite[0].arithmetic.approx

log2_fast/exp2_fast are approximate FP32 functions on stabilized gates and prefix differences; unknown: a reliable error bound for the source log2_fast/exp2_fast path has not been established.

## t_e4c62488f606729c | math.arithmetic | 29 chars | 2 declarations

m64_single_sequence:75 design.math_reshape.rewrite[2].arithmetic.approx; root:140 design.math_reshape.rewrite[2].arithmetic.approx

exp2_fast follows step chunk.

## t_0298fa2c8bae14f7 | math.ledger | 16 chars | 2 declarations

m64_single_sequence:80 design.math_reshape.ledger.vector[0].item; root:145 design.math_reshape.ledger.vector[0].item

Token recurrence

## t_05c7b177f0175ad8 | math.ledger | 16 chars | 2 declarations

m64_single_sequence:78 design.math_reshape.ledger.depth[0].item; root:143 design.math_reshape.ledger.depth[0].item

State dependency

## t_157dab9ce93a9558 | math.ledger | 60 chars | 2 declarations

m64_single_sequence:78 design.math_reshape.ledger.depth[0].change; root:143 design.math_reshape.ledger.depth[0].change

L_b token updates -> Np chunk updates per sequence and head.

## t_19c80c00542ebb48 | math.ledger | 114 chars | 2 declarations

m64_single_sequence:77 design.math_reshape.ledger.baseline; root:142 design.math_reshape.ledger.baseline

Direct FP32 token recurrence with one live DV-by-DK state per sequence and head and no global intermediate states.

## t_1e1d496686c459aa | math.ledger | 87 chars | 2 declarations

m64_single_sequence:81 design.math_reshape.ledger.footprint[0].change; root:146 design.math_reshape.ledger.footprint[0].change

+O(C*C+C*(DK+DV)) elements beyond the DV*DK FP32 state, independent of sequence length.

## t_25be23c7ed3f3c89 | math.ledger | 10 chars | 10 declarations

m64_single_sequence:78 design.math_reshape.ledger.depth[0].by[2]; m64_single_sequence:79 design.math_reshape.ledger.matmul[0].by[1]; m64_single_sequence:80 design.math_reshape.ledger.vector[0].by[1]; m64_single_sequence:81 design.math_reshape.ledger.footprint[0].by[2]; m64_single_sequence:82 design.math_reshape.ledger.error[0].by[2]; root:143 design.math_reshape.ledger.depth[0].by[2]; root:144 design.math_reshape.ledger.matmul[0].by[1]; root:145 design.math_reshape.ledger.vector[0].by[1]; root:146 design.math_reshape.ledger.footprint[0].by[2]; root:147 design.math_reshape.ledger.error[0].by[2]

recurrence

## t_2bd79a3681ec0911 | math.ledger | 5 chars | 6 declarations

m64_single_sequence:78 design.math_reshape.ledger.depth[0].by[0]; m64_single_sequence:81 design.math_reshape.ledger.footprint[0].by[0]; m64_single_sequence:82 design.math_reshape.ledger.error[0].by[0]; root:143 design.math_reshape.ledger.depth[0].by[0]; root:146 design.math_reshape.ledger.footprint[0].by[0]; root:147 design.math_reshape.ledger.error[0].by[0]

chunk

## t_3d4bb15128e038dd | math.ledger | 14 chars | 2 declarations

m64_single_sequence:82 design.math_reshape.ledger.error[0].item; root:147 design.math_reshape.ledger.error[0].item

Numerical path

## t_4645f39032fa9869 | math.ledger | 102 chars | 2 declarations

m64_single_sequence:82 design.math_reshape.ledger.error[0].change; root:147 design.math_reshape.ledger.error[0].change

+fast log2/exp2 and BF16 inverse, snapshot, residual and update rounding; FP32 carried state retained.

## t_50f410388ae5159e | math.ledger | 14 chars | 2 declarations

m64_single_sequence:79 design.math_reshape.ledger.matmul[0].item; root:144 design.math_reshape.ledger.matmul[0].item

Chunk products

## t_8280173fa2d36e48 | math.ledger | 7 chars | 10 declarations

m64_single_sequence:78 design.math_reshape.ledger.depth[0].by[1]; m64_single_sequence:79 design.math_reshape.ledger.matmul[0].by[0]; m64_single_sequence:80 design.math_reshape.ledger.vector[0].by[0]; m64_single_sequence:81 design.math_reshape.ledger.footprint[0].by[1]; m64_single_sequence:82 design.math_reshape.ledger.error[0].by[1]; root:143 design.math_reshape.ledger.depth[0].by[1]; root:144 design.math_reshape.ledger.matmul[0].by[0]; root:145 design.math_reshape.ledger.vector[0].by[0]; root:146 design.math_reshape.ledger.footprint[0].by[1]; root:147 design.math_reshape.ledger.error[0].by[1]

inverse

## t_942574c9c058fc3c | math.ledger | 19 chars | 2 declarations

m64_single_sequence:81 design.math_reshape.ledger.footprint[0].item; root:146 design.math_reshape.ledger.footprint[0].item

Local intermediates

## t_c9cc86f85a733d23 | math.ledger | 148 chars | 2 declarations

m64_single_sequence:80 design.math_reshape.ledger.vector[0].change; root:145 design.math_reshape.ledger.vector[0].change

Removes 3*L_b*DV*DK projection/update MACs per sequence and head; replaces them with the block products and O(Np*(DV*DK+C*DV+C*C)) elementwise work.

## t_d47420ff649e9ff6 | math.ledger | 98 chars | 2 declarations

m64_single_sequence:79 design.math_reshape.ledger.matmul[0].change; root:144 design.math_reshape.ledger.matmul[0].change

+7 dense products and one C-by-C hierarchical inverse per padded chunk before value-row partition.

## t_03e7f003d1a1e456 | math.math | 450 chars | 2 declarations

m64_single_sequence:40 design.math_reshape.rewrite[1].math; root:105 design.math_reshape.rewrite[1].math

KK=mm32(K,Kᵀ); A=bf16(KK ⊙ W ⊙ Beta[:,None]); J=I+strict_lower(A). Invert each diagonal 8x8 block of J by FP32 Gauss-Jordan elimination and round it to BF16. Merge widths 8->16->32->64: for Jblock=[[X,0],[Z,Y]], its off-diagonal inverse is bf16(mm32(bf16(-mm32(Yinv,Z)),Xinv)); retain the two inverse diagonal blocks and zero upper half. With the resulting inverse Iinv, Tinv=bf16(Iinv ⊙ Beta[None,:]). QK=mm32(Q,Kᵀ); Bmat=bf16((QK ⊙ W)*fp32(scale)).

## t_268eeef78f301be6 | math.math | 427 chars | 1 declarations

m64_single_sequence:16 design.math_reshape.rewrite[0].math

Cu32=int32(cu_seqlens). For b=0 and each h, hq=floor(h/(HV/HQ)), L_b=T, Np=2*ceil(L_b/(2*C)); 0<=c<Np and 0<=i<C. At x=c*C+i<L_b, Q(c)[i,:]=q[x,hq,:], K(c)[i,:]=k[x,hq,:], V(c)[i,:]=v[x,h,:], Alpha(c)[i]=alpha[x,h], Beta(c)[i]=beta[x,h]. Outside L_b, use Q=K=V=0, Alpha=1, Beta=0. For each chunk, G[i]=sum32_{j=0}^{i} log2_fast(Alpha[j]+epsilon), Gamma[i]=exp2_fast(G[i]), and W[i,j]=exp2_fast(G[i]-G[j]) for j<=i, otherwise 0.

## t_40df5dc7792535c7 | math.math | 448 chars | 1 declarations

root:120 design.math_reshape.rewrite[2].math

S(-1)=initial_state[b,h]. For c=0..Np-1, omit c on local quantities: Sbar=bf16(S(c-1)); Sdec=Gamma[C-1]*S(c-1); KS=mm32(Sbar,Kᵀ); QS=mm32(Sbar,Qᵀ). R=bf16(Vᵀ-bf16(KS ⊙ Gamma[None,:])); U=mm32(R,Tinvᵀ). Ubar=bf16(U); Decay[i]=exp2_fast(G[C-1]-G[i]); Udec=bf16(U ⊙ Decay[None,:]). QSscaled=(QS ⊙ Gamma[None,:])*fp32(scale); Y=mma32(Ubar,Bmatᵀ,QSscaled). S(c)=mma32(Udec,K,Sdec); for c*C+i<L_b, O[Cu32[b]+c*C+i,h,:]=bf16(Y[:,i]); S_final[b,h]=S(Np-1).

## t_83944359c0217afb | math.math | 440 chars | 1 declarations

m64_single_sequence:55 design.math_reshape.rewrite[2].math

S(-1)=initial_state[b,h]. For c=0..Np-1, omit c on local quantities: Sbar=bf16(S(c-1)); Sdec=Gamma[C-1]*S(c-1); KS=mm32(Sbar,Kᵀ); QS=mm32(Sbar,Qᵀ). R=bf16(Vᵀ-bf16(KS ⊙ Gamma[None,:])); U=mm32(R,Tinvᵀ). Ubar=bf16(U); Decay[i]=exp2_fast(G[C-1]-G[i]); Udec=bf16(U ⊙ Decay[None,:]). QSscaled=(QS ⊙ Gamma[None,:])*fp32(scale); Y=mma32(Ubar,Bmatᵀ,QSscaled). S(c)=mma32(Udec,K,Sdec); for c*C+i<L_b, O[c*C+i,h,:]=bf16(Y[:,i]); S_final[b,h]=S(Np-1).

## t_be86032434361a08 | math.math | 456 chars | 1 declarations

root:81 design.math_reshape.rewrite[0].math

Cu32=int32(cu_seqlens). For each b,h, hq=floor(h/(HV/HQ)), L_b=Cu32[b+1]-Cu32[b], Np=2*ceil(L_b/(2*C)); 0<=c<Np and 0<=i<C. At x=Cu32[b]+c*C+i with c*C+i<L_b, Q(c)[i,:]=q[x,hq,:], K(c)[i,:]=k[x,hq,:], V(c)[i,:]=v[x,h,:], Alpha(c)[i]=alpha[x,h], Beta(c)[i]=beta[x,h]. Outside L_b, use Q=K=V=0, Alpha=1, Beta=0. For each chunk, G[i]=sum32_{j=0}^{i} log2_fast(Alpha[j]+epsilon), Gamma[i]=exp2_fast(G[i]), and W[i,j]=exp2_fast(G[i]-G[j]) for j<=i, otherwise 0.

## t_06b74a4b00421eb0 | math.meaning | 29 chars | 2 declarations

m64_single_sequence:31 design.math_reshape.rewrite[0].symbol.Beta.meaning; root:96 design.math_reshape.rewrite[0].symbol.Beta.meaning

Zero-padded update strengths.

## t_0aaef773eb6fa619 | math.meaning | 38 chars | 2 declarations

m64_single_sequence:33 design.math_reshape.rewrite[0].symbol.Gamma.meaning; root:98 design.math_reshape.rewrite[0].symbol.Gamma.meaning

Decay from incoming state to each row.

## t_0f986bb23bb1428a | math.meaning | 28 chars | 2 declarations

m64_single_sequence:69 design.math_reshape.rewrite[2].symbol.U.meaning; root:134 design.math_reshape.rewrite[2].symbol.U.meaning

FP32 corrected value update.

## t_12cc71bc981ebf96 | math.meaning | 27 chars | 2 declarations

m64_single_sequence:70 design.math_reshape.rewrite[2].symbol.Ubar.meaning; root:135 design.math_reshape.rewrite[2].symbol.Ubar.meaning

Rounded update for readout.

## t_13953c5deb7bde1c | math.meaning | 43 chars | 2 declarations

m64_single_sequence:72 design.math_reshape.rewrite[2].symbol.Udec.meaning; root:137 design.math_reshape.rewrite[2].symbol.Udec.meaning

End-decayed update rounded directly from U.

## t_3964705ab301e71b | math.meaning | 17 chars | 2 declarations

m64_single_sequence:28 design.math_reshape.rewrite[0].symbol.K.meaning; root:93 design.math_reshape.rewrite[0].symbol.K.meaning

Padded key chunk.

## t_4404ed6cfde10939 | math.meaning | 52 chars | 2 declarations

m64_single_sequence:34 design.math_reshape.rewrite[0].symbol.W.meaning; root:99 design.math_reshape.rewrite[0].symbol.W.meaning

Causal decay ratios reused for feedback and readout.

## t_490580c9f75b1c57 | math.meaning | 54 chars | 2 declarations

m64_single_sequence:52 design.math_reshape.rewrite[1].symbol.Bmat.meaning; root:117 design.math_reshape.rewrite[1].symbol.Bmat.meaning

Causal decay-weighted and scaled readout coefficients.

## t_5777bfa1fc55f7b5 | math.meaning | 28 chars | 2 declarations

m64_single_sequence:32 design.math_reshape.rewrite[0].symbol.G.meaning; root:97 design.math_reshape.rewrite[0].symbol.G.meaning

Inclusive log2 decay prefix.

## t_57cacaaa34edef2b | math.meaning | 41 chars | 2 declarations

m64_single_sequence:50 design.math_reshape.rewrite[1].symbol.Tinv.meaning; root:115 design.math_reshape.rewrite[1].symbol.Tinv.meaning

Rounded inverse with column beta scaling.

## t_5cf4671fbaf99a2f | math.meaning | 30 chars | 2 declarations

m64_single_sequence:66 design.math_reshape.rewrite[2].symbol.KS.meaning; root:131 design.math_reshape.rewrite[2].symbol.KS.meaning

Incoming-state key projection.

## t_649331bc4009f6a8 | math.meaning | 53 chars | 1 declarations

m64_single_sequence:26 design.math_reshape.rewrite[0].symbol.Cu32.meaning

Converted offsets unused by the single-sequence core.

## t_6db7812837549460 | math.meaning | 43 chars | 2 declarations

m64_single_sequence:68 design.math_reshape.rewrite[2].symbol.R.meaning; root:133 design.math_reshape.rewrite[2].symbol.R.meaning

Residual after two separate BF16 roundings.

## t_6e0582597a0fa3dc | math.meaning | 36 chars | 2 declarations

m64_single_sequence:64 design.math_reshape.rewrite[2].symbol.Sbar.meaning; root:129 design.math_reshape.rewrite[2].symbol.Sbar.meaning

Rounded state snapshot before decay.

## t_7cf3756998bf8ddb | math.meaning | 27 chars | 2 declarations

m64_single_sequence:24 design.math_reshape.rewrite[0].symbol.C.meaning; root:89 design.math_reshape.rewrite[0].symbol.C.meaning

Tokens per recurrent chunk.

## t_7df165f0a63b19f7 | math.meaning | 21 chars | 2 declarations

m64_single_sequence:48 design.math_reshape.rewrite[1].symbol.KK.meaning; root:113 design.math_reshape.rewrite[1].symbol.KK.meaning

Key Gram accumulator.

## t_883ca107f3905f83 | math.meaning | 19 chars | 2 declarations

m64_single_sequence:29 design.math_reshape.rewrite[0].symbol.V.meaning; root:94 design.math_reshape.rewrite[0].symbol.V.meaning

Padded value chunk.

## t_8a1eba8e4d9d0f23 | math.meaning | 49 chars | 1 declarations

root:91 design.math_reshape.rewrite[0].symbol.Cu32.meaning

Converted packed boundaries consumed by the core.

## t_8afdbb706b6006ef | math.meaning | 23 chars | 2 declarations

m64_single_sequence:74 design.math_reshape.rewrite[2].symbol.Y.meaning; root:139 design.math_reshape.rewrite[2].symbol.Y.meaning

Unrounded output chunk.

## t_8b92db6188a380a0 | math.meaning | 30 chars | 2 declarations

m64_single_sequence:65 design.math_reshape.rewrite[2].symbol.Sdec.meaning; root:130 design.math_reshape.rewrite[2].symbol.Sdec.meaning

State scaled to the chunk end.

## t_91c93ad2e2a72ccf | math.meaning | 39 chars | 2 declarations

m64_single_sequence:63 design.math_reshape.rewrite[2].symbol.S.meaning; root:128 design.math_reshape.rewrite[2].symbol.S.meaning

FP32 chunk-boundary value-by-key state.

## t_97709533776d8965 | math.meaning | 32 chars | 2 declarations

m64_single_sequence:67 design.math_reshape.rewrite[2].symbol.QS.meaning; root:132 design.math_reshape.rewrite[2].symbol.QS.meaning

Incoming-state query projection.

## t_9b8a0c46171a2d65 | math.meaning | 22 chars | 2 declarations

m64_single_sequence:51 design.math_reshape.rewrite[1].symbol.QK.meaning; root:116 design.math_reshape.rewrite[1].symbol.QK.meaning

Query-key accumulator.

## t_ab649368db08e1c1 | math.meaning | 48 chars | 2 declarations

m64_single_sequence:49 design.math_reshape.rewrite[1].symbol.A.meaning; root:114 design.math_reshape.rewrite[1].symbol.A.meaning

Rounded feedback and in-place inverse workspace.

## t_c04aea241f61cff9 | math.meaning | 29 chars | 2 declarations

m64_single_sequence:30 design.math_reshape.rewrite[0].symbol.Alpha.meaning; root:95 design.math_reshape.rewrite[0].symbol.Alpha.meaning

Neutral-padded linear decays.

## t_d0491b410260f8d3 | math.meaning | 30 chars | 2 declarations

m64_single_sequence:73 design.math_reshape.rewrite[2].symbol.QSscaled.meaning; root:138 design.math_reshape.rewrite[2].symbol.QSscaled.meaning

Scaled incoming-state readout.

## t_d44624e5ca4f3ac3 | math.meaning | 49 chars | 2 declarations

m64_single_sequence:71 design.math_reshape.rewrite[2].symbol.Decay.meaning; root:136 design.math_reshape.rewrite[2].symbol.Decay.meaning

Remaining decay from each token to the chunk end.

## t_fa4216613f3e4884 | math.meaning | 19 chars | 2 declarations

m64_single_sequence:27 design.math_reshape.rewrite[0].symbol.Q.meaning; root:92 design.math_reshape.rewrite[0].symbol.Q.meaning

Padded query chunk.

## t_fcf3e639e4c70fc0 | math.meaning | 49 chars | 2 declarations

m64_single_sequence:25 design.math_reshape.rewrite[0].symbol.Np.meaning; root:90 design.math_reshape.rewrite[0].symbol.Np.meaning

Even padded chunk count for the current sequence.

## t_432c9e0b148109b9 | math.name | 55 chars | 2 declarations

m64_single_sequence:39 design.math_reshape.rewrite[1].name; root:104 design.math_reshape.rewrite[1].name

Invert local feedback with explicit BF16 merge rounding

## t_de20a419c9fd7490 | math.name | 45 chars | 2 declarations

m64_single_sequence:15 design.math_reshape.rewrite[0].name; root:80 design.math_reshape.rewrite[0].name

Pad to complete pairs and form decay prefixes

## t_f20bc386256985f1 | math.name | 41 chars | 2 declarations

m64_single_sequence:54 design.math_reshape.rewrite[2].name; root:119 design.math_reshape.rewrite[2].name

Correct values and update the local state

## t_48857c9a607fe610 | math.step | 7 chars | 2 declarations

m64_single_sequence:38 design.math_reshape.rewrite[1].step; root:103 design.math_reshape.rewrite[1].step

inverse

## t_4e32c8e46d5b4f61 | math.step | 10 chars | 2 declarations

m64_single_sequence:53 design.math_reshape.rewrite[2].step; root:118 design.math_reshape.rewrite[2].step

recurrence

## t_7b767d78cd5bd935 | math.step | 5 chars | 2 declarations

m64_single_sequence:14 design.math_reshape.rewrite[0].step; root:79 design.math_reshape.rewrite[0].step

chunk

## t_77748160509f3f78 | math.thesis | 165 chars | 2 declarations

m64_single_sequence:12 design.math_reshape.thesis; root:77 design.math_reshape.thesis

Replace the token dependency chain with chunk matrix products and a hierarchical inverse, paying local quadratic work and BF16 rounding while carrying state in FP32.

## t_6f05304359c0b22c | metadata.describe | 117 chars | 2 declarations

m64_single_sequence:3 meta_data.describe; root:3 meta_data.describe

Apply grouped-head gated delta updates to packed sequences and overwrite the supplied output and final-state buffers.

## t_9ad391a797c48070 | pipeline.thesis | 212 chars | 1 declarations

root:186 design.tile_pipeline.thesis

Separate pairwise inverse preparation from the recurrent chain with two compute groups and two MMA issuers; unequal rings and early accumulator returns support overlap while scalar and output work share one warp.

## t_d870b1a3c36b1cb6 | pipeline.thesis | 191 chars | 1 declarations

m64_single_sequence:119 design.tile_pipeline.thesis

Separate pairwise inverse preparation from the recurrent chain with two compute groups and two MMA issuers; four-chunk scalar lookahead uses output progress to protect five ready-only stages.

## t_0f07081619eca457 | pipeline.verb | 84 chars | 1 declarations

root:304 design.tile_pipeline.phase.gdn.role.ScalarOutput.info.verb.load.meaning

Acquire a separate beta stage and copy Beta with explicit zero-fill for padded rows.

## t_106cd544b6cbc270 | pipeline.verb | 84 chars | 1 declarations

root:302 design.tile_pipeline.phase.gdn.role.ScalarOutput.info.verb.compute

Load predicated Alpha into registers and form G/Gamma before acquiring a gate stage.

## t_1760e93dd2d618d9 | pipeline.verb | 99 chars | 1 declarations

root:243 design.tile_pipeline.phase.gdn.role.Inverse.info.verb.publish

Tinv applies column Beta and returns its beta stage; Bmat weights the captured QK using retained W.

## t_18aca5f842ce6acc | pipeline.verb | 83 chars | 1 declarations

root:303 design.tile_pipeline.phase.gdn.role.ScalarOutput.info.verb.publish

Gate writes G/Gamma and publishes them together after acquiring their shared stage.

## t_1c4ecd3132b3b929 | pipeline.verb | 90 chars | 2 declarations

m64_single_sequence:195 design.tile_pipeline.phase.gdn.role.Recurrence.info.verb.load; root:268 design.tile_pipeline.phase.gdn.role.Recurrence.info.verb.load

Initialize S from this worktile's explicit initial_state, including all-zero input states.

## t_2d944261e9874f56 | pipeline.verb | 472 chars | 1 declarations

m64_single_sequence:196 design.tile_pipeline.phase.gdn.role.Recurrence.info.verb.publish

StateOperand reads each incoming S subtile once, snapshots Sbar and decays S in place before publication; R captures/releases KS; UpdateOperand writes Ubar/Udec from the captured FP32 U; O rounds Y into SMEM and releases its TMEM slot. StateOperand, R and UpdateOperand share one ready-event stream. KS, U and state completion gate successive publications; UpdateOperand makes both following MMA products ready without a second decay notification or empty-stage handshake.

## t_304cb8b6f5a7d0fd | pipeline.verb | 126 chars | 2 declarations

m64_single_sequence:161 design.tile_pipeline.phase.gdn.role.MMA0.info.verb.matmul; root:228 design.tile_pipeline.phase.gdn.role.MMA0.info.verb.matmul

Issue the named upstream product with zero FP32 accumulator; completion covers this issuer's prior products and operand reads.

## t_373e3471edbe11bf | pipeline.verb | 88 chars | 1 declarations

root:239 design.tile_pipeline.phase.gdn.role.Inverse.info.verb.prepare

Form W for both chunks, return their gate stages, and retain W through Bmat publication.

## t_3f802932ce3ebc5c | pipeline.verb | 61 chars | 1 declarations

root:305 design.tile_pipeline.phase.gdn.role.ScalarOutput.info.verb.store

Store O and return its sole slot after bulk-store completion.

## t_43bffef6cb85ae17 | pipeline.verb | 186 chars | 1 declarations

root:220 design.tile_pipeline.phase.gdn.role.Load.info.verb.bound.meaning

Update this worker's Q/K/V descriptors to Cu32[b+1] and fence before use; earlier requests may remain in flight. Loads start at Cu32[b]+c*C and zero-fill beyond the current sequence end.

## t_47ae33007ef4ae5b | pipeline.verb | 87 chars | 1 declarations

m64_single_sequence:228 design.tile_pipeline.phase.gdn.role.ScalarOutput.info.verb.prime.meaning

Issue raw Alpha/Beta transfers for chunks 0/1, each spanning four adjacent value heads.

## t_4908bdeb7a81806c | pipeline.verb | 166 chars | 2 declarations

m64_single_sequence:181 design.tile_pipeline.phase.gdn.role.MMA1.info.verb.matmul; root:253 design.tile_pipeline.phase.gdn.role.MMA1.info.verb.matmul

Issue the named upstream product; KS/QS/U start from zero, Y accumulates into QSscaled, S into Sdec. Completion covers this issuer's prior products and operand reads.

## t_513dd7d932964669 | pipeline.verb | 160 chars | 2 declarations

m64_single_sequence:124 design.tile_pipeline.phase.index_cast.role.Convert.info.verb.produce; root:191 design.tile_pipeline.phase.index_cast.role.Convert.info.verb.produce

Perform the upstream Cu32 conversion with the contiguous PyTorch copy kernel; only valid B+1 elements are read and written, with no internal asynchronous stage.

## t_64d091036abe724f | pipeline.verb | 86 chars | 1 declarations

root:240 design.tile_pipeline.phase.gdn.role.Inverse.info.verb.capture

Capture both chunks' row Beta; keep column Beta live in SMEM through Tinv publication.

## t_7a8b041439bedb57 | pipeline.verb | 122 chars | 1 declarations

root:273 design.tile_pipeline.phase.gdn.role.Recurrence.info.verb.compute

Capture Gamma and end-decay factors after StateOperand publication, return this role's gate stage, and overlap with KS/QS.

## t_83652c75fb992cb9 | pipeline.verb | 96 chars | 1 declarations

root:301 design.tile_pipeline.phase.gdn.role.ScalarOutput.info.verb.bound

Update this worker's O descriptor to Cu32[b+1] and fence before stores, suppressing padded rows.

## t_883da689cc69177a | pipeline.verb | 62 chars | 2 declarations

m64_single_sequence:201 design.tile_pipeline.phase.gdn.role.Recurrence.info.verb.read; root:274 design.tile_pipeline.phase.gdn.role.Recurrence.info.verb.read

Capture U in registers and return the shared KS/U accumulator.

## t_94171abdbbd93873 | pipeline.verb | 457 chars | 1 declarations

root:269 design.tile_pipeline.phase.gdn.role.Recurrence.info.verb.publish

StateOperand reads each incoming S subtile once, snapshots Sbar and decays S in place before publication; R captures/releases KS; Ubar and Udec are written from the captured FP32 U before their separate publications; O rounds Y into SMEM and releases its TMEM slot. StateOperand, R, Ubar and Udec have independent notification streams. Producers omit empty-stage acquisition: subsequent KS/U/state completion transitively protects their fixed operand slots.

## t_983106a340af33ce | pipeline.verb | 103 chars | 1 declarations

m64_single_sequence:234 design.tile_pipeline.phase.gdn.role.ScalarOutput.info.verb.store

Store O with the embedded descriptor bounded by T and return its sole slot after bulk-store completion.

## t_b030da1c7751ec8f | pipeline.verb | 336 chars | 1 declarations

m64_single_sequence:229 design.tile_pipeline.phase.gdn.role.ScalarOutput.info.verb.prepare

Wait for both raw transfers, capture column h%4 and finish raw reads before reissuing chunk c+2 when present; issue that transfer before computing G/Gamma. Raw stages use c%2; write G/Gamma/Beta into processed stage c%5. No empty-stage acquisition or consumer return: preparing c+5 follows store:O(c), protecting all old scalar readers.

## t_c145f25b39585b2a | pipeline.verb | 91 chars | 2 declarations

m64_single_sequence:173 design.tile_pipeline.phase.gdn.role.Inverse.info.verb.read; root:241 design.tile_pipeline.phase.gdn.role.Inverse.info.verb.read

Capture KK/QK in registers and return its TMEM slot before subsequent weighting arithmetic.

## t_c47de18ba3ae684c | pipeline.verb | 156 chars | 2 declarations

m64_single_sequence:174 design.tile_pipeline.phase.gdn.role.Inverse.info.verb.invert; root:242 design.tile_pipeline.phase.gdn.role.Inverse.info.verb.invert

Invert both chunks in place with group rendezvous between levels; two warps per chunk at widths 8/32/64, all four across each chunk's 16x16 diagonal blocks.

## t_cb27bc7b5078acf8 | pipeline.verb | 61 chars | 1 declarations

m64_single_sequence:233 design.tile_pipeline.phase.gdn.role.ScalarOutput.info.verb.publish

Scalar signals joint readiness after all G/Gamma/Beta writes.

## t_cec860337c98f00f | pipeline.verb | 98 chars | 1 declarations

m64_single_sequence:200 design.tile_pipeline.phase.gdn.role.Recurrence.info.verb.compute

Capture Gamma and end-decay factors after StateOperand publication, permitting overlap with KS/QS.

## t_d61b4c2a7ef13a4f | pipeline.verb | 72 chars | 1 declarations

m64_single_sequence:175 design.tile_pipeline.phase.gdn.role.Inverse.info.verb.publish

Tinv applies column Beta; Bmat weights the captured QK using retained W.

## t_dd4a2966a273836b | pipeline.verb | 32 chars | 2 declarations

m64_single_sequence:202 design.tile_pipeline.phase.gdn.role.Recurrence.info.verb.rescale; root:275 design.tile_pipeline.phase.gdn.role.Recurrence.info.verb.rescale

Replace QS by QSscaled in place.

## t_dd6e50e80e7b8620 | pipeline.verb | 130 chars | 1 declarations

m64_single_sequence:172 design.tile_pipeline.phase.gdn.role.Inverse.info.verb.prepare

Capture both chunks' W and row Beta; retain W through Bmat publication and keep column Beta live in SMEM through Tinv publication.

## t_fd3ce3eba9be55e9 | pipeline.verb | 84 chars | 1 declarations

m64_single_sequence:153 design.tile_pipeline.phase.gdn.role.Load.info.verb.load

Use embedded descriptors bounded by T; partial and synthetic chunks zero-fill Q/K/V.

## t_77f0d01467df493b | realization.choice | 143 chars | 1 declarations

root:324 realization.require[3].choice

Separate PipelineAsyncUmma notification streams for StateOperand, R, Ubar and Udec, retaining the source's omitted producer acquire operations.

## t_7dfe0e091a8ad1f2 | realization.choice | 68 chars | 2 declarations

m64_single_sequence:242 realization.require[1].choice; root:318 realization.require[1].choice

BF16 warp-level mma.sync for inverse merges, with FP32 accumulators.

## t_b1a00a9ae185d899 | realization.choice | 89 chars | 2 declarations

m64_single_sequence:239 realization.require[0].choice; root:315 realization.require[0].choice

Single-CTA tcgen05.mma with TMEM access and issuer-local cumulative completion signaling.

## t_c8e7b4b4b6ebbe7e | realization.choice | 161 chars | 1 declarations

root:321 realization.require[2].choice

Per-worker mutable TMA Q/K/V/O descriptors via TensorMapManager GMEM updates and fences; PipelineCpAsync for Beta and PipelineAsync for gate and output handoffs.

## t_fdd0d988c0872744 | realization.choice | 203 chars | 1 declarations

m64_single_sequence:245 realization.require[2].choice

Embedded TMA descriptors; transaction-count mbarriers for raw scalars; warp rendezvous and async-shared fence before raw-stage reuse; ready-only scalar and compact Recurrence MbarrierArray notifications.

## t_6390584ff8db3df0 | realization.reason | 101 chars | 2 declarations

m64_single_sequence:240 realization.require[0].reason; root:316 realization.require[0].reason

Supplies the ordering and operand-read completion used by accumulator aliases and input-slot returns.

## t_731675cc45401616 | realization.reason | 75 chars | 2 declarations

m64_single_sequence:243 realization.require[1].reason; root:319 realization.require[1].reason

Keeps inverse preparation independent of the two asynchronous TMEM issuers.

## t_768b71c4eecdfa3b | realization.reason | 169 chars | 1 declarations

root:322 realization.require[2].reason

Supplies sequence-tail isolation and independent scalar returns; the ordering of old in-flight descriptor reads across TensorMapManager updates has not been established.

## t_9457c507c163ecd7 | realization.reason | 106 chars | 1 declarations

root:325 realization.require[3].reason

Operand reuse follows the recurrent dependencies; a full handshake on each input would change the overlap.

## t_fafd588dd07dab16 | realization.reason | 122 chars | 1 declarations

m64_single_sequence:246 realization.require[2].reason

Supplies transfer visibility and ordered publication without adding full/empty handshakes to the two ready-only protocols.

## t_1a5ed071695eca1f | schedule.describe | 113 chars | 1 declarations

root:164 design.work_schedule.phase.gdn.describe

Fused M64 recurrence with per-worktile sequence-end descriptors and separate scalar and Recurrence input streams.

## t_71e6999a09164fa8 | schedule.describe | 96 chars | 1 declarations

m64_single_sequence:98 design.work_schedule.phase.gdn.describe

Fused M64 recurrence with fixed sequence bounds [0,T), scalar TMA and compact Recurrence events.

## t_b5977bf32348dc3a | schedule.describe | 84 chars | 2 declarations

m64_single_sequence:89 design.work_schedule.phase.index_cast.describe; root:155 design.work_schedule.phase.index_cast.describe

Timed PyTorch 2.9.0 contiguous int64-to-int32 copy, submitted before the GDN kernel.

## t_02b7c3cf9d41a509 | schedule.strategy | 115 chars | 2 declarations

m64_single_sequence:91 design.work_schedule.phase.index_cast.decomposition.strategy; root:157 design.work_schedule.phase.index_cast.decomposition.strategy

The complete B+1 boundary array is one worktile; 2 <= B+1 <= 17 fits the library kernel's 512-element CTA capacity.

## t_2a0f6a7113cc1467 | schedule.strategy | 163 chars | 1 declarations

root:182 design.work_schedule.phase.gdn.schedule.strategy

All roles follow the same preassigned sequence; ring cursors continue across worktiles while S reloads each worktile's initial_state. No per-worktile global drain.

## t_3b33887778ec93dc | schedule.strategy | 150 chars | 1 declarations

m64_single_sequence:112 design.work_schedule.phase.gdn.mapping.strategy

P=min(B*HV*(DV/64),num_sm), with num_sm queried at runtime. For B=1, linear id=2*h+row_half; the recorded 148-SM run gives 64 CTAs, one worktile each.

## t_56041319279124d3 | schedule.strategy | 88 chars | 1 declarations

root:166 design.work_schedule.phase.gdn.decomposition.strategy

Traverse Np chunks per worktile using Cu32[b:b+2]; the two value-row halves never merge.

## t_5b334a0189d7d8e5 | schedule.strategy | 132 chars | 1 declarations

root:178 design.work_schedule.phase.gdn.mapping.strategy

P=min(B*HV*(DV/64),num_sm), with num_sm queried at runtime; linear id=b*(2*HV)+2*h+row_half. The source L20D run records num_sm=148.

## t_cfdbf59620953e19 | schedule.strategy | 106 chars | 1 declarations

m64_single_sequence:100 design.work_schedule.phase.gdn.decomposition.strategy

Traverse Np chunks per worktile; the two row halves never merge. Cu32 is passed but not read by this core.

## t_e65d1c79fa3e54f1 | schedule.thesis | 172 chars | 2 declarations

m64_single_sequence:84 design.work_schedule.thesis; root:149 design.work_schedule.thesis

Split value heads into independent 64-row halves; each CTA retains the full chunk recurrence, doubling work count while duplicating Q/K traffic, Gram products and inverses.
