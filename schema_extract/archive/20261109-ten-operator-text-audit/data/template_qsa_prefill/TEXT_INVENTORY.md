# template_qsa_prefill: full open-text inventory

These are exact-text groups, not proof of semantic equivalence across contexts.

## t_29954f805f650a11 | algorithm_contract.math | 854 chars | 1 declarations

root:40 contract.algorithm_requirement.compute[0].math

K/V share storage with element strides [2157376,512,256,1]; v_cache starts 65536 elements after k_cache. For each independent row r and head h, v=floor(h/GroupSize), J(r)={0,...,logical_indices[r,SelectedWidth]-1}. For j in J(r), k=logical_indices[r,j], page=block_table[0,floor(k/PageSize)], offset=k%PageSize. K_j=k_cache[page,offset,v,:], V_j=v_cache[page,offset,v,:]; score[j]=scale*sum_{d=0}^{D-1} q[r,h,d]*K_j[d]. a=max_{j in J(r)} score[j]; z=sum_{j in J(r)} exp(score[j]-a). A[r,h,d]=bf16(sum_{j in J(r)} (exp(score[j]-a)/z)*V_j[d]). O[r,h,d]=bf16(A[r,h,d]/(1+exp(-output_gate[r,h,d]))). The supplied selection encodes causality; add no further mask. Write every out element and leave other inputs unchanged. The timed callable includes selection preprocessing, attention, gating and output allocation when needed; input construction is excluded.

## t_00ba45850b847099 | algorithm_contract.meaning | 27 chars | 1 declarations

root:33 contract.algorithm_requirement.input.use_prefill_config.meaning

Ignored configuration hint.

## t_0aa9a15fdef03d31 | algorithm_contract.meaning | 39 chars | 1 declarations

root:52 contract.algorithm_requirement.compute[0].symbol.A.meaning

Attention result rounded before gating.

## t_0c8625f40aa57cb7 | algorithm_contract.meaning | 32 chars | 1 declarations

root:17 contract.algorithm_requirement.constant.GroupSize.meaning

Query heads sharing one KV head.

## t_1dabe8dcdfc643da | algorithm_contract.meaning | 24 chars | 1 declarations

root:32 contract.algorithm_requirement.input.token_to_req.meaning

Request owning each row.

## t_2039fe3883d209b0 | algorithm_contract.meaning | 27 chars | 1 declarations

root:34 contract.algorithm_requirement.input.output_gate.meaning

Elementwise sigmoid logits.

## t_435ef2cea1740fc3 | algorithm_contract.meaning | 14 chars | 1 declarations

root:11 contract.algorithm_requirement.constant.KVHeads.meaning

KV head count.

## t_5e8ee9d61827b0f0 | algorithm_contract.meaning | 26 chars | 1 declarations

root:12 contract.algorithm_requirement.constant.D.meaning

Query and value dimension.

## t_77a94d685c172ff2 | algorithm_contract.meaning | 38 chars | 1 declarations

root:51 contract.algorithm_requirement.compute[0].symbol.J.meaning

Valid selection-slot indices of a row.

## t_797f3063f57d252c | algorithm_contract.meaning | 31 chars | 1 declarations

root:13 contract.algorithm_requirement.constant.PageSize.meaning

Tokens per physical cache page.

## t_813fb722718b5998 | algorithm_contract.meaning | 25 chars | 1 declarations

root:15 contract.algorithm_requirement.constant.TableWidth.meaning

Logical page-table width.

## t_84b0376aae29ffbd | algorithm_contract.meaning | 22 chars | 1 declarations

root:18 contract.algorithm_requirement.constant.scale.meaning

Attention score scale.

## t_8a16f0e1f26e0455 | algorithm_contract.meaning | 11 chars | 1 declarations

root:20 contract.algorithm_requirement.input.q.meaning

Query rows.

## t_987248ce8bad2c35 | algorithm_contract.meaning | 20 chars | 1 declarations

root:9 contract.algorithm_requirement.constant.Rows.meaning

CP-local query rows.

## t_b1f8409c80ee3f65 | algorithm_contract.meaning | 62 chars | 1 declarations

root:22 contract.algorithm_requirement.input.v_cache.meaning

Paged values sharing the interleaved K allocation and strides.

## t_b5cbb79b2e85d4c2 | algorithm_contract.meaning | 44 chars | 1 declarations

root:16 contract.algorithm_requirement.constant.SelectedWidth.meaning

Selection slots before the valid-count word.

## t_c2633d8a14eaba39 | algorithm_contract.meaning | 57 chars | 1 declarations

root:35 contract.algorithm_requirement.input.out.meaning

Optional destination whose initial contents are not read.

## t_ce55bc0d35c5fb2d | algorithm_contract.meaning | 26 chars | 1 declarations

root:14 contract.algorithm_requirement.constant.CachePages.meaning

Physical cache page count.

## t_d867ea4d21dd5ba1 | algorithm_contract.meaning | 11 chars | 1 declarations

root:21 contract.algorithm_requirement.input.k_cache.meaning

Paged keys.

## t_dc77c36eaf806e1b | algorithm_contract.meaning | 57 chars | 1 declarations

root:37 contract.algorithm_requirement.output.O.meaning

Result written into out and returned as that same tensor.

## t_f754587975cffeb6 | algorithm_contract.meaning | 55 chars | 1 declarations

root:24 contract.algorithm_requirement.input.logical_indices.meaning

Selected logical token IDs followed by the valid count.

## t_fafc62027311f974 | algorithm_contract.meaning | 17 chars | 1 declarations

root:10 contract.algorithm_requirement.constant.QueryHeads.meaning

Query head count.

## t_fea9502fdbafb960 | algorithm_contract.meaning | 32 chars | 1 declarations

root:31 contract.algorithm_requirement.input.block_table.meaning

Request-to-physical-page lookup.

## t_26c276f6263ee19e | algorithm_contract.range | 273 chars | 1 declarations

root:27 contract.algorithm_requirement.input.logical_indices.range

One invocation-wide CP rank rho is in {0,1}; c_r=floor((15*8192+rho*Rows+r+1)/4), t_r=(r+1)%4. The first 2048 entries are 512 quads 4*g+[0,1,2,3] with distinct 0<=g<c_r. Entries 2048+l equal 4*c_r+l for 0<=l<t_r; remaining selection slots are -1; the last word is 2048+t_r.

## t_8ec9301d7fa8f80d | algorithm_contract.range | 52 chars | 1 declarations

root:31 contract.algorithm_requirement.input.block_table.range

block_table[0,p]=1352+p for 0<=p<1024; -1 otherwise.

## t_c2322bbd50b03c04 | algorithm_contract.range | 99 chars | 2 declarations

root:21 contract.algorithm_requirement.input.k_cache.range; root:22 contract.algorithm_requirement.input.v_cache.range

Mapped pages are finite; other storage may contain arbitrary BF16 bits, including NaN and infinity.

## t_f3947aadafa51c6e | algorithm_contract.range | 48 chars | 1 declarations

root:35 contract.algorithm_requirement.input.out.range

Arbitrary BF16 bits, including NaN and infinity.

## t_3f634870d998d1d9 | contract.compute_name | 56 chars | 1 declarations

root:39 contract.algorithm_requirement.compute[0].name

Resolve the supplied selection and apply gated attention

## t_b97a8f5653796085 | contract.tolerance | 71 chars | 1 declarations

root:55 contract.algorithm_requirement.golden_reference.tolerance.O

atol=3.2e-5, rtol=0.02; check both the returned tensor and mutated out.

## t_65834f6b180e82b6 | dispatch.md | 32 chars | 1 declarations

dispatch.md:5 dispatch.md

```mermaid flowchart TD root ```

## t_67f66c06820293de | dispatch.md | 68 chars | 1 declarations

dispatch.md:10 dispatch.md

唯一入口为 [root](root.yaml)。它省略 `dispatch`，节点 ID 缺省为 `root`，设计覆盖整个契约有效域。

## t_74feea2ee8dbafdc | dispatch.md | 171 chars | 1 declarations

dispatch.md:12 dispatch.md

该域为 NVIDIA L20D / SM103 上的 single-request fitted-step15，采用串行 eager 调用。具体输入、布局和有效域见 root 的 `contract`，修订号见 `meta_data.version`；`use_prefill_config` 的取值和 `out` 是否传入均沿用这一份设计。

## t_dbdae945f065d322 | dispatch.md | 13 chars | 1 declarations

dispatch.md:1 dispatch.md

# qsa_prefill

## t_e12a988493524956 | dispatch.md | 97 chars | 1 declarations

dispatch.md:3 dispatch.md

本任务只有 root，保存完整 Contract 和核心设计：相邻 query 行共用索引并集，以 persistent 调度接续工作，并完成 attention 与 sigmoid gate。

## t_680d6642a8764eac | hardware.execution | 55 chars | 1 declarations

root:61 contract.hardware_specification.execution.CTA

Warps co-resident on one SM, sharing SMEM and barriers.

## t_6f5eff329130bc7c | hardware.execution | 63 chars | 1 declarations

root:62 contract.hardware_specification.execution.cluster

CTAs co-scheduled within one GPC, with cluster synchronization.

## t_707ed1a5b6536df5 | hardware.execution | 31 chars | 1 declarations

root:60 contract.hardware_specification.execution.warp

32 SIMT threads within one CTA.

## t_263d5b133f5e61d6 | hardware.feature | 56 chars | 1 declarations

root:66 contract.hardware_specification.feature.register

Thread-private registers backed by the SM register file.

## t_6c67314fe3205270 | hardware.feature | 61 chars | 1 declarations

root:65 contract.hardware_specification.feature.TMEM

Explicitly allocated Tensor Core storage, separate from SMEM.

## t_bf65cfd621166b87 | hardware.feature | 101 chars | 1 declarations

root:67 contract.hardware_specification.feature.asynchronous_execution

Asynchronous transfers and MMA require their primitive-specific completion and visibility guarantees.

## t_fbe55b4f6cea5249 | hardware.feature | 88 chars | 1 declarations

root:64 contract.hardware_specification.feature.SMEM

CTA-lifetime shared storage, addressable by cluster peers while its owner remains alive.

## t_2bd1510fef32c674 | hardware.stack | 82 chars | 1 declarations

root:70 contract.hardware_specification.stack.toolchain

CuTe DSL 4.6.1 bundled compiler; CUDA 13.0; cuda-python 13.1.1; torch 2.9.0+cu130.

## t_d054d71928dc57ba | hardware.stack | 36 chars | 1 declarations

root:69 contract.hardware_specification.stack.language

CuTe DSL 4.6.1 (nvidia-cutlass-dsl).

## t_25ffffa2c36c0234 | math.arithmetic | 130 chars | 1 declarations

root:126 design.math_reshape.rewrite[1].arithmetic.approx

exp2_fast uses approximate FP32 exp2 on the shown arguments; unknown: no numerical error bound has been established for exp2_fast.

## t_583d095316e3f494 | math.arithmetic | 160 chars | 1 declarations

root:134 design.math_reshape.rewrite[2].arithmetic.approx

exp2_fast follows online; rcp_fast is approximate FP32 reciprocal for nonzero denominators; unknown: no numerical error bound has been established for rcp_fast.

## t_01c5d260b33458e0 | math.ledger | 9 chars | 1 declarations

root:141 design.math_reshape.ledger.matmul[0].item

QK and PV

## t_06311df5f243ef15 | math.ledger | 6 chars | 4 declarations

root:139 design.math_reshape.ledger.traffic[1].by[0]; root:141 design.math_reshape.ledger.matmul[0].by[1]; root:143 design.math_reshape.ledger.footprint[0].by[0]; root:148 design.math_reshape.ledger.error[0].by[1]

online

## t_19b2f56c8c40a936 | math.ledger | 14 chars | 1 declarations

root:146 design.math_reshape.ledger.pass[0].item

Sparse indices

## t_22910388926326ab | math.ledger | 112 chars | 1 declarations

root:141 design.math_reshape.ledger.matmul[0].change

2*D*GroupSize*sum_{r in Rb} count(J(r)) -> 2*D*M*Bk*N(b) MAC per (b,v), including padded rows and masked columns

## t_3104652cc8c8de8b | math.ledger | 43 chars | 1 declarations

root:146 design.math_reshape.ledger.pass[0].change

+1 preprocessing traversal before attention

## t_3d4bb15128e038dd | math.ledger | 14 chars | 1 declarations

root:148 design.math_reshape.ledger.error[0].item

Numerical path

## t_4f9d1592bf43c18e | math.ledger | 142 chars | 1 declarations

root:136 design.math_reshape.ledger.baseline

Direct selected attention with one KV gather per query and KV head, sharing KV across grouped heads, and global FP32 scores and probabilities.

## t_73be1d280808235d | math.ledger | 55 chars | 1 declarations

root:139 design.math_reshape.ledger.traffic[1].change

-16*GroupSize*sum_{r in Rb} count(J(r)) bytes per (b,v)

## t_85b2b4b5d60d908b | math.ledger | 73 chars | 1 declarations

root:144 design.math_reshape.ledger.footprint[1].change

+16*Count(b)+4 live bytes per row block, shared by both KV-head worktiles

## t_b6060850ad395dc4 | math.ledger | 5 chars | 5 declarations

root:138 design.math_reshape.ledger.traffic[0].by[0]; root:141 design.math_reshape.ledger.matmul[0].by[0]; root:144 design.math_reshape.ledger.footprint[1].by[0]; root:146 design.math_reshape.ledger.pass[0].by[0]; root:148 design.math_reshape.ledger.error[0].by[0]

union

## t_cec5cbdcdc91f3ee | math.ledger | 10 chars | 1 declarations

root:138 design.math_reshape.ledger.traffic[0].item

KV payload

## t_cf8108a6a7daaaa3 | math.ledger | 135 chars | 1 declarations

root:148 design.math_reshape.ledger.error[0].change

BF16 P adds one rounding per PV weight; union order, split FP32 denominators, lagged reference, fast exp2 and reciprocal alter rounding

## t_dcf5c94273b88ae2 | math.ledger | 93 chars | 1 declarations

root:138 design.math_reshape.ledger.traffic[0].change

4*D*sum_{r in Rb} count(J(r)) -> 4*D*Bk*N(b) bytes per (b,v), including gathered tail padding

## t_e46e64e375a66d15 | math.ledger | 33 chars | 1 declarations

root:139 design.math_reshape.ledger.traffic[1].item

Score and probability round trips

## t_e71f38bf4a5e1d75 | math.ledger | 4 chars | 1 declarations

root:148 design.math_reshape.ledger.error[0].by[2]

gate

## t_f447b9ff043fad9b | math.ledger | 163 chars | 1 declarations

root:143 design.math_reshape.ledger.footprint[0].change

8*GroupSize*sum_{r in Rb} count(J(r)) global score/probability bytes -> 4*M*(D+3) logical carried-state bytes per (b,v), excluding transient tiles and replicated m

## t_f59e242a156084cc | math.ledger | 18 chars | 1 declarations

root:143 design.math_reshape.ledger.footprint[0].item

Intermediate state

## t_ffd1c6495328e4d9 | math.ledger | 15 chars | 1 declarations

root:144 design.math_reshape.ledger.footprint[1].item

Union workspace

## t_5950a8358d727ae0 | math.math | 830 chars | 1 declarations

root:83 design.math_reshape.rewrite[0].math

Rb={r | b*Br<=r<min((b+1)*Br,Rows)}; MaxE=Br*(floor(SelectedWidth/4)+1). Union(b) enumerates distinct (req=0,g=floor(logical_indices[r,j]/4)) over r in Rb, j in J(r); Count(b) is its length. Each entry stores (req,g,lo,hi), with bit 4*(r-b*Br)+l set iff row r selects token 4*g+l; order need not be sorted. For each KV head v, pack p=GroupSize*(r-b*Br)+h: Qblk[p,:]=q[r,v*GroupSize+h,:], 0<=h<GroupSize. Other packed rows remain unwritten and produce no output; their influence on the shared correction vote has not been established. N(b)=ceil(Count(b)/Etile)>=32. Tile i column u uses entry e=i*Etile+floor(u/4) and token offset u%4. Kblk(i)[u,:] and Vblk(i)[u,:] gather that group's paged KV rows; e>=Count(b) rereads entry zero's finite payload. selected(p,i,u) is the row's membership bit, false for inactive p or e>=Count(b).

## t_6e37617b75a6a7d2 | math.math | 328 chars | 1 declarations

root:129 design.math_reshape.rewrite[2].math

ell[p]=L(N(b)-1)[0,p]+L(N(b)-1)[1,p]; inv[p]=rcp_fast(ell[p]) if ell[p]!=0, otherwise 0. For each real packed row p=GroupSize*(r-b*Br)+h, A[r,v*GroupSize+h,d]=bf16(Y(N(b)-1)[p,d]*inv[p]) if ell[p]!=0, otherwise 0. sig=1/(1+exp2_fast(-output_gate[r,v*GroupSize+h,d]*log2(e))); O[r,v*GroupSize+h,d]=bf16(A[r,v*GroupSize+h,d]*sig).

## t_d179d03136c55dae | math.math | 892 chars | 1 declarations

root:104 design.math_reshape.rewrite[1].math

For each (b,v), c=scale*log2(e); m(-1)[p]=-inf and L(-1)[w,p]=0 for w in {0,1}, 0<=p<M. For 0<=i<N(b), S(i)=mm32(Qblk,Kblk(i)ᵀ); vmax(i,p)=c*max_{0<=u<Bk} S(i)[p,u], before membership masking. first(i,p)=(m(i-1)[p]==-inf); drift(i,p)=0 if first(i,p), otherwise vmax(i,p)-m(i-1)[p]. move_ref(i,p)=any_{p' in rescale_peer(p)}(drift(i,p')>tau). m(i)[p]=vmax(i,p) if first(i,p) or move_ref(i,p), otherwise m(i-1)[p]; a triggered update may decrease m. The two halves keep replicas m_A=m_B=m. alpha(i,p)=exp2_fast(m(i-1)[p]-m(i)[p]) if move_ref(i,p), otherwise 1. Evaluate exp2_fast(c*S(i)[p,u]-m(i)[p]) for every column; E(i)[p,u] keeps it iff selected(p,i,u), otherwise 0. P(i)=bf16(E(i)); P_A(i)=P(i)[:,0:Bk/2], P_B(i)=P(i)[:,Bk/2:Bk]. L(i)[w,p]=alpha(i,p)*L(i-1)[w,p]+sum32_{w*Bk/2<=u<(w+1)*Bk/2} E(i)[p,u]. Y(0)=mm32(P(0),Vblk(0)); for i>0, Y(i)=mma32(P(i),Vblk(i),alpha(i)[:,None] ⊙ Y(i-1)).

## t_0bf40d76d34113a6 | math.meaning | 29 chars | 1 declarations

root:93 design.math_reshape.rewrite[0].symbol.Br.meaning

Query rows sharing one union.

## t_16342f46136e630c | math.meaning | 43 chars | 1 declarations

root:97 design.math_reshape.rewrite[0].symbol.Union.meaning

Compacted request/group/membership entries.

## t_18ea1f0574d57742 | math.meaning | 23 chars | 1 declarations

root:124 design.math_reshape.rewrite[1].symbol.Y.meaning

Unnormalized numerator.

## t_219b19b0425e90b9 | math.meaning | 32 chars | 1 declarations

root:123 design.math_reshape.rewrite[1].symbol.L.meaning

Separate FP32 half-denominators.

## t_2672ef2832cca609 | math.meaning | 28 chars | 1 declarations

root:95 design.math_reshape.rewrite[0].symbol.Bk.meaning

Token columns per iteration.

## t_2b0ff0fc7ee897df | math.meaning | 20 chars | 1 declarations

root:100 design.math_reshape.rewrite[0].symbol.Kblk.meaning

Gathered union keys.

## t_311dd9e5c7926e36 | math.meaning | 22 chars | 1 declarations

root:101 design.math_reshape.rewrite[0].symbol.Vblk.meaning

Gathered union values.

## t_4019cf79d812dbb1 | math.meaning | 23 chars | 1 declarations

root:120 design.math_reshape.rewrite[1].symbol.P_A.meaning

First column half of P.

## t_43f81063399e33f6 | math.meaning | 46 chars | 1 declarations

root:122 design.math_reshape.rewrite[1].symbol.m.meaning

Common reference replicated by the two halves.

## t_93ceed22ebd44593 | math.meaning | 26 chars | 1 declarations

root:119 design.math_reshape.rewrite[1].symbol.P.meaning

Rounded numerator weights.

## t_ab123af9cf756d3c | math.meaning | 23 chars | 1 declarations

root:98 design.math_reshape.rewrite[0].symbol.Count.meaning

Live union entry count.

## t_ae8e2725397370be | math.meaning | 23 chars | 1 declarations

root:99 design.math_reshape.rewrite[0].symbol.Qblk.meaning

Packed query/head rows.

## t_dd7b90e79109b916 | math.meaning | 36 chars | 1 declarations

root:116 design.math_reshape.rewrite[1].symbol.tau.meaning

Base-two reference-update threshold.

## t_ec11670d931428b0 | math.meaning | 38 chars | 1 declarations

root:117 design.math_reshape.rewrite[1].symbol.rescale_peer.meaning

Packed rows sharing a correction vote.

## t_ec5c06b3b419ea22 | math.meaning | 28 chars | 1 declarations

root:94 design.math_reshape.rewrite[0].symbol.Etile.meaning

Union entries per iteration.

## t_ed1f323d723cb34a | math.meaning | 22 chars | 1 declarations

root:96 design.math_reshape.rewrite[0].symbol.M.meaning

Packed query capacity.

## t_f7933849de56c090 | math.meaning | 24 chars | 1 declarations

root:121 design.math_reshape.rewrite[1].symbol.P_B.meaning

Second column half of P.

## t_f9e097adbe2406ff | math.meaning | 19 chars | 1 declarations

root:118 design.math_reshape.rewrite[1].symbol.S.meaning

Unmasked QK scores.

## t_888f1af38501ec87 | math.name | 44 chars | 1 declarations

root:128 design.math_reshape.rewrite[2].name

Normalize and preserve both output roundings

## t_9651edc8219a24d0 | math.name | 62 chars | 1 declarations

root:103 design.math_reshape.rewrite[1].name

Stream two half-denominators against a shared lagged reference

## t_c8b68f28ad1875a7 | math.name | 49 chars | 1 declarations

root:82 design.math_reshape.rewrite[0].name

Materialize four-token unions with row membership

## t_1ea39aef19f89ac1 | math.step | 4 chars | 1 declarations

root:127 design.math_reshape.rewrite[2].step

gate

## t_bf711e10cb54fbf4 | math.step | 5 chars | 1 declarations

root:81 design.math_reshape.rewrite[0].step

union

## t_cd9bee15f13aa998 | math.step | 6 chars | 1 declarations

root:102 design.math_reshape.rewrite[1].step

online

## t_8c86186f2e768d6b | math.thesis | 189 chars | 1 declarations

root:79 design.math_reshape.thesis

Share a union of nearby rows' selected groups across packed GQA heads; streaming softmax saves KV gathers and global probabilities at the cost of masked products and one preprocessing pass.

## t_ec9553c32596e53e | metadata.describe | 100 chars | 1 declarations

root:3 meta_data.describe

Compute selected paged attention, round the attention result, and apply an elementwise sigmoid gate.

## t_d3a989ead7a46866 | pipeline.thesis | 184 chars | 1 declarations

root:195 design.tile_pipeline.thesis

Two softmax column halves overlap three leading QKs with PV; KV rings span worktiles, and Q refill overlaps output drain while the next first PV waits for both Y halves to be released.

## t_161b7ea0049cdf46 | pipeline.verb | 65 chars | 1 declarations

root:295 design.tile_pipeline.phase.attention.role.MMA.info.verb.wait.meaning

Wait for the initial Q publication even when Ticket0 is terminal.

## t_201acf706699d407 | pipeline.verb | 138 chars | 1 declarations

root:276 design.tile_pipeline.phase.attention.role.GatherV.info.verb.load

Follow GatherK's address lookahead and delayed publication for ValueRing, adding the V base delta and using the value-product orientation.

## t_2ff9fd2d683de324 | pipeline.verb | 54 chars | 1 declarations

root:361 design.tile_pipeline.phase.attention.role.SoftmaxB.info.verb.combine

Follow A's HalfSum exchange and normalization capture.

## t_4db605e0d7315757 | pipeline.verb | 172 chars | 1 declarations

root:296 design.tile_pipeline.phase.attention.role.MMA.info.verb.mma

QK initializes S and each worktile's first PV initializes Y. Cumulative completion includes operand reads; PV(n) reads P before a later QK overwrites its aliased S/P stage.

## t_603b84fcbb16bf9f | pipeline.verb | 74 chars | 5 declarations

root:255 design.tile_pipeline.phase.attention.role.GatherK.info.verb.accept.meaning; root:275 design.tile_pipeline.phase.attention.role.GatherV.info.verb.accept.meaning; root:294 design.tile_pipeline.phase.attention.role.MMA.info.verb.accept.meaning; root:319 design.tile_pipeline.phase.attention.role.SoftmaxA.info.verb.accept; root:358 design.tile_pipeline.phase.attention.role.SoftmaxB.info.verb.accept

Copy and acknowledge the published ticket; terminal tickets start no work.

## t_6d6711ca5067a5ea | pipeline.verb | 88 chars | 1 declarations

root:329 design.tile_pipeline.phase.attention.role.SoftmaxA.info.verb.prefetch.meaning

Issue the next valid ticket's Q gather into Qblk after the current final QK has retired.

## t_7899cfbcb718928b | pipeline.verb | 69 chars | 2 declarations

root:333 design.tile_pipeline.phase.attention.role.SoftmaxA.info.verb.store; root:363 design.tile_pipeline.phase.attention.role.SoftmaxB.info.verb.store

Gate and directly store the retained final output piece on real rows.

## t_7de653cd4e690396 | pipeline.verb | 364 chars | 1 declarations

root:241 design.tile_pipeline.phase.attention.role.Scheduler.info.verb.publish

Claim and post a ticket; done means readable. Before claiming the following ticket, await all eleven consumer warps' copies and acknowledgements of the current ticket. Acknowledgement permits one-worktile lookahead while that work is still executing. Ticket0 is the first ticket; Next(t) includes the terminal ticket, whose acknowledgement is required before exit.

## t_8b6274ff2fcbe683 | pipeline.verb | 156 chars | 1 declarations

root:331 design.tile_pipeline.phase.attention.role.SoftmaxA.info.verb.drain

Read Y[:,0:D/2] in 32-column pieces, gate/store the first pieces and retain the last in registers; done releases this half of Y before its final gate/store.

## t_9117da571fe58f9d | pipeline.verb | 106 chars | 1 declarations

root:330 design.tile_pipeline.phase.attention.role.SoftmaxA.info.verb.combine

Exchange final L halves through HalfSum with a 256-thread rendezvous and capture the normalization factor.

## t_aa4b710cb69eb65b | pipeline.verb | 1076 chars | 1 declarations

root:320 design.tile_pipeline.phase.attention.role.SoftmaxA.info.verb.publish

Qblk0 gathers real packed rows for min(Ticket0,worktile_count-1), including unused payload on an initial terminal ticket. HalfMax_A captures all S[:,0:Bk/2] into registers, writes this HalfMax half and joins B's 256-thread rendezvous. Both score halves have been captured before either group overwrites the aliased stage with P; HalfMax_A done precedes peer-maximum capture. P_A captures the peer maximum and evaluates online from the retained register scores. rescale_peer(p) is the 32-row interval containing p, including padding. Only warps with move_ref true wait for PV(n-1) when i>0, then A rescales all D columns of their Y rows before publishing P_A. Prime membership per worktile; retain the current packed mask through weight formation while prefetching raw words for min(i+1,N(b)-1) in registers. Count masks dead entries when packing the next mask. HalfMax overwrite safety has not been established; no extra read-complete barrier is assumed. Qnext completes the issued Q refill and publishes readiness after this group's final Y read; terminal Next(t) is a no-op.

## t_b0e53891338ed25c | pipeline.verb | 81 chars | 1 declarations

root:362 design.tile_pipeline.phase.attention.role.SoftmaxB.info.verb.drain

Follow A's drain for Y[:,D/2:D], releasing this half before the final gate/store.

## t_be21bf11fd93b288 | pipeline.verb | 179 chars | 1 declarations

root:213 design.tile_pipeline.phase.preprocess.role.Main.info.verb.compact

Each warp prefix-scans contiguous hash-slot ranges, atomically reserves an output interval, and emits Union; publish Count after all writes. Warp reservation order is unspecified.

## t_d748bc91f3cce424 | pipeline.verb | 477 chars | 1 declarations

root:256 design.tile_pipeline.phase.attention.role.GatherK.info.verb.load

Prime address lookahead per worktile, then resolve Union two tiles ahead and block_table one ahead, clamped to that worktile's last tile. Resolve one base per union-entry position and share it within the gather warp across that entry's token/channel copies; issue the current paged copy into KeyRing. load:Kblk(n) becomes done only after issuing tile n+1 and publishing n's completed copy. Carry delayed publication across worktiles; drain publishes the final outstanding tile.

## t_ddac58e89193981e | pipeline.verb | 208 chars | 1 declarations

root:359 design.tile_pipeline.phase.attention.role.SoftmaxB.info.verb.publish

Capture S[:,Bk/2:Bk] into registers before the shared HalfMax rendezvous; follow A's peer capture, correction vote and mask lookahead, forming this half of L and P from that register copy without modifying Y.

## t_e894309d73eab610 | pipeline.verb | 312 chars | 1 declarations

root:209 design.tile_pipeline.phase.preprocess.role.Main.info.verb.build

Clear Hash and CompactCount, cache row counts/requests and request-zero PageTable, then rendezvous. Stream 16-byte index quads one item ahead in registers; one CAS/linear-probing insertion per nonempty group merges membership with atomic OR by every inserter, including the creator. Rendezvous before compaction.

## t_e9b45ab6030b95bb | pipeline.verb | 103 chars | 1 declarations

root:244 design.tile_pipeline.phase.attention.role.Scheduler.info.verb.wait

Join all consumer acknowledgements before posting the following ticket or retiring the terminal ticket.

## t_f0d7d3a98bc30d13 | pipeline.verb | 206 chars | 2 declarations

root:258 design.tile_pipeline.phase.attention.role.GatherK.info.verb.drain.meaning; root:277 design.tile_pipeline.phase.attention.role.GatherV.info.verb.drain.meaning

After the terminal ticket, finish and publish the last outstanding gather if any data was loaded. If Ticket0 was already terminal, no gather exists and this action completes without waiting for a last tile.

## t_9fba041bc2174ea3 | realization.choice | 95 chars | 1 declarations

root:382 realization.require[0].choice

Single-CTA tcgen05.mma with TMEM accumulators, TMEM P for PV and cumulative completion commits.

## t_ead8c8bc19c151ff | realization.choice | 144 chars | 1 declarations

root:383 realization.require[1].choice

cp.async with wait_group(1) for delayed KV publication and wait_group(0) for final drain/Q readiness, plus shared-memory visibility publication.

## t_f0777944b2af88d1 | realization.choice | 113 chars | 1 declarations

root:384 realization.require[2].choice

Contiguous 16-byte-aligned logical_indices; gate/out last-axis stride 1 and outer element strides divisible by 8.

## t_3a8f1db74940243c | realization.reason | 91 chars | 1 declarations

root:382 realization.require[0].reason

S/P alias reuse and Y correction depend on direct TMEM handoff and operand-read completion.

## t_ca5fcb8a29992913 | realization.reason | 88 chars | 1 declarations

root:384 realization.require[2].reason

Packed index quads and vector gate/output transfers require these addressing conditions.

## t_e9168b73e10a3446 | realization.reason | 88 chars | 1 declarations

root:383 realization.require[1].reason

These completion guarantees preserve two-slot cross-worktile gathers and next-Q overlap.

## t_3af40f743ab81c09 | schedule.describe | 48 chars | 1 declarations

root:158 design.work_schedule.phase.preprocess.describe

Build one union per consecutive query-row block.

## t_6413cfb1d2733e8f | schedule.describe | 91 chars | 1 declarations

root:173 design.work_schedule.phase.attention.describe

Keep one row-block/KV-head union traversal and all its grouped outputs inside one worktile.

## t_3b05b9bc4ac1d535 | schedule.exit | 106 chars | 1 declarations

root:193 design.work_schedule.phase.attention.schedule.worker_assignment.exit

Terminal ticket acknowledged by all roles, all acquired work complete, and all in-flight accesses drained.

## t_daae652bafc5e351 | schedule.refill | 177 chars | 1 declarations

root:192 design.work_schedule.phase.attention.schedule.worker_assignment.refill.atomic_counter

One scheduler thread per CTA atomically claims one ticket minus the launch base; IDs below worktile_count follow worktile_order, and each CTA stops on its first terminal ticket.

## t_17aa633fad8ac866 | schedule.strategy | 42 chars | 1 declarations

root:187 design.work_schedule.phase.attention.mapping.strategy

SMCount is queried from device properties.

## t_4fcd1d929ce6f646 | schedule.strategy | 6 chars | 1 declarations

root:160 design.work_schedule.phase.preprocess.decomposition.strategy

Br=10.

## t_d8a13347234faecd | schedule.strategy | 230 chars | 1 declarations

root:189 design.work_schedule.phase.attention.schedule.strategy

Cached device ticket and host base start at zero; each serialized eager call consumes worktile_count+worker_count tickets and advances the host base by that amount, using signed-int32 wrap-safe subtraction without a counter reset.

## t_e2506e283a9169a2 | schedule.strategy | 106 chars | 1 declarations

root:175 design.work_schedule.phase.attention.decomposition.strategy

M=128; Etile=16; Bk=4*Etile. The final row block contains six rows; no cross-worktile reduction is needed.

## t_7909b8a700318457 | schedule.thesis | 142 chars | 1 declarations

root:150 design.work_schedule.thesis

Build shared unions first, then balance variable-length row-block/head traversals with dynamic persistent CTAs and descending row-block order.
