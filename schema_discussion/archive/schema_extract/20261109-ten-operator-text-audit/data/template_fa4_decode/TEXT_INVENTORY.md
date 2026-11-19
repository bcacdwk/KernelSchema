# template_fa4_decode: full open-text inventory

These are exact-text groups, not proof of semantic equivalence across contexts.

## t_612303295b5a07c0 | algorithm_contract.math | 620 chars | 1 declarations

root:58 contract.algorithm_requirement.compute[1].math

For independent b,t,h with 0<=b<BatchCount, 0<=t<q_len_per_req, 0<=h<HeadCount: visible_key(b,t)={k in integers | 0<=k<=L_b-q_len_per_req+t}. score[b,t,h,k]=bmm1_scale*sum_{d=0}^{D-1} query[b*q_len_per_req+t,h,d]*K[b,k,d]. m=max_{k in visible_key(b,t)} score[b,t,h,k]; Z=sum_{k in visible_key(b,t)} exp(score[b,t,h,k]-m). probability[b,t,h,k]=exp(score[b,t,h,k]-m)/Z on visible_key(b,t), and zero otherwise. output[b*q_len_per_req+t,h,d]=bf16(bmm2_scale*sum_{k in visible_key(b,t)} probability[b,t,h,k]*V[b,k,d]). All visible sets are nonempty. out_after=output; return the same tensor object as out; no LSE is returned.

## t_aa7114bc4645fdfe | algorithm_contract.math | 662 chars | 1 declarations

root:45 contract.algorithm_requirement.compute[0].math

python_int denotes a Python integer; fp64 scalar arguments are Python binary64 floats. python_none denotes the Python None singleton and carries no numeric payload. TotalQuery=BatchCount*q_len_per_req; max_seq_len=max_b seq_lens[b]; L_b=seq_lens[b]. All tensors are contiguous on one CUDA device; sinks=None and o_sf_scale=None. For 0<=b<BatchCount, the first ceil(L_b/PageSize) block-table entries are valid pages. For 0<=k<L_b and 0<=d<D, let page=block_tables[b,floor(k/PageSize)]: K[b,k,d]=kv_cache[page,0,0,k%PageSize,d]; V[b,k,d]=kv_cache[page,1,0,k%PageSize,d]. The timed boundary includes any partial-result combination; cache construction is outside it.

## t_00ff08e2416a0872 | algorithm_contract.meaning | 36 chars | 1 declarations

root:67 contract.algorithm_requirement.compute[1].symbol.visible_key.meaning

Valid causal keys for one tail token

## t_01745f0cc1cb3d84 | algorithm_contract.meaning | 64 chars | 1 declarations

root:27 contract.algorithm_requirement.input.workspace_buffer.meaning

Scratch zeroed before first isolated use; not a semantic output.

## t_06d88af6b525635b | algorithm_contract.meaning | 29 chars | 1 declarations

root:28 contract.algorithm_requirement.input.block_tables.meaning

Logical-to-physical page map.

## t_11cfdb86b5f29d96 | algorithm_contract.meaning | 30 chars | 1 declarations

root:55 contract.algorithm_requirement.compute[0].symbol.K.meaning

Logical keys after page lookup

## t_24bd1249d880994e | algorithm_contract.meaning | 28 chars | 1 declarations

root:33 contract.algorithm_requirement.input.bmm2_scale.meaning

Composite value-output scale

## t_2743d856663b5ee5 | algorithm_contract.meaning | 39 chars | 1 declarations

root:39 contract.algorithm_requirement.input.q_len_per_req.meaning

Number of query-tail tokens per request

## t_29e4269e2f113ad9 | algorithm_contract.meaning | 39 chars | 1 declarations

root:35 contract.algorithm_requirement.input.sinks.meaning

Required None sentinel; sinks disabled.

## t_30162d8698ea3177 | algorithm_contract.meaning | 37 chars | 1 declarations

root:30 contract.algorithm_requirement.input.seq_lens.meaning

KV lengths including the query tails.

## t_36a68eb2994a8ec8 | algorithm_contract.meaning | 55 chars | 1 declarations

root:36 contract.algorithm_requirement.input.o_sf_scale.meaning

Required None sentinel; output scaling tensor disabled.

## t_3ac611c21d592f15 | algorithm_contract.meaning | 22 chars | 1 declarations

root:21 contract.algorithm_requirement.constant.PageSize.meaning

Tokens per cache page.

## t_435ef2cea1740fc3 | algorithm_contract.meaning | 14 chars | 1 declarations

root:19 contract.algorithm_requirement.constant.KVHeadCount.meaning

KV head count.

## t_448671885442cbcc | algorithm_contract.meaning | 31 chars | 1 declarations

root:20 contract.algorithm_requirement.constant.D.meaning

Query, key and value dimension.

## t_46702e1f254793cf | algorithm_contract.meaning | 61 chars | 1 declarations

root:32 contract.algorithm_requirement.input.bmm1_scale.meaning

Composite score scale including captured quantization scales.

## t_4aae3203399a14c8 | algorithm_contract.meaning | 43 chars | 1 declarations

root:37 contract.algorithm_requirement.input.out.meaning

Destination; incoming contents are ignored.

## t_4ad6442e949ae9e0 | algorithm_contract.meaning | 32 chars | 1 declarations

root:56 contract.algorithm_requirement.compute[0].symbol.V.meaning

Logical values after page lookup

## t_4d9e07aceb4cf0d2 | algorithm_contract.meaning | 41 chars | 1 declarations

root:69 contract.algorithm_requirement.compute[1].symbol.probability.meaning

Softmax probability on valid logical keys

## t_59db9d4c7975b1cb | algorithm_contract.meaning | 32 chars | 1 declarations

root:22 contract.algorithm_requirement.constant.TableWidth.meaning

Block-table entries per request.

## t_7571558d679b5672 | algorithm_contract.meaning | 10 chars | 1 declarations

root:24 contract.algorithm_requirement.input.query.meaning

Query tail

## t_8568ad127e4a96f3 | algorithm_contract.meaning | 23 chars | 1 declarations

root:34 contract.algorithm_requirement.input.window_left.meaning

Full-attention sentinel

## t_865493c154d4b2d6 | algorithm_contract.meaning | 44 chars | 1 declarations

root:68 contract.algorithm_requirement.compute[1].symbol.score.meaning

Scaled query-key score on valid logical keys

## t_9999918522567c66 | algorithm_contract.meaning | 30 chars | 1 declarations

root:41 contract.algorithm_requirement.output.output.meaning

Returned tensor, aliasing out.

## t_9a1b53c80650e623 | algorithm_contract.meaning | 56 chars | 1 declarations

root:25 contract.algorithm_requirement.input.kv_cache.meaning

HND cache with key at selector 0 and value at selector 1

## t_dc318633beaf917c | algorithm_contract.meaning | 55 chars | 1 declarations

root:42 contract.algorithm_requirement.output.out_after.meaning

In-place replacement of input out; identical to output.

## t_e4b3ad7dc97c1210 | algorithm_contract.meaning | 25 chars | 1 declarations

root:31 contract.algorithm_requirement.input.max_seq_len.meaning

Maximum entry of seq_lens

## t_fafc62027311f974 | algorithm_contract.meaning | 17 chars | 1 declarations

root:18 contract.algorithm_requirement.constant.HeadCount.meaning

Query head count.

## t_0845eca47b8e4ef5 | algorithm_contract.range | 16 chars | 2 declarations

root:35 contract.algorithm_requirement.input.sinks.range; root:36 contract.algorithm_requirement.input.o_sf_scale.range

Python None only

## t_340e25ab552fe800 | algorithm_contract.range | 50 chars | 1 declarations

root:37 contract.algorithm_requirement.input.out.range

Any BF16 bit pattern; never read before overwrite.

## t_5ac8acf468778595 | algorithm_contract.range | 23 chars | 1 declarations

root:39 contract.algorithm_requirement.input.q_len_per_req.range

q_len_per_req in {1, 4}

## t_8751851ba8eb951e | algorithm_contract.range | 59 chars | 1 declarations

root:28 contract.algorithm_requirement.input.block_tables.range

-1 <= block_tables < PageCount; valid pages are nonnegative

## t_8d6dd8b53b9c3dfa | algorithm_contract.range | 48 chars | 1 declarations

root:30 contract.algorithm_requirement.input.seq_lens.range

q_len_per_req <= seq_lens <= PageSize*TableWidth

## t_359a5123bc339c71 | contract.compute_name | 38 chars | 1 declarations

root:57 contract.algorithm_requirement.compute[1].name

Attend to the right-aligned query tail

## t_5dd2afe59012b391 | contract.compute_name | 40 chars | 1 declarations

root:44 contract.algorithm_requirement.compute[0].name

Interpret the captured callable boundary

## t_28c8cfebc4cd144c | contract.tolerance | 20 chars | 2 declarations

root:72 contract.algorithm_requirement.golden_reference.tolerance.output; root:72 contract.algorithm_requirement.golden_reference.tolerance.out_after

atol=0.06, rtol=0.04

## t_1d386832a69e9cc6 | dispatch.md | 89 chars | 1 declarations

dispatch.md:18 dispatch.md

| [q4_compact](q4_compact.yaml) | q4 的特定长序列 batch；保留名义 split 布局，合并线程串行处理活跃 split 并跳过空负载 |

## t_2a87548cee5693af | dispatch.md | 19 chars | 1 declarations

dispatch.md:1 dispatch.md

# fa4_decode schema

## t_36a5cbc1fbb83399 | dispatch.md | 93 chars | 1 declarations

dispatch.md:20 dispatch.md

| [q4_direct](q4_direct.yaml) | q4 的单 KV block；由单 CTA 完成各 query tile，省去 split workspace 与合并 |

## t_3f7cb032ce65ad8b | dispatch.md | 29 chars | 1 declarations

dispatch.md:15 dispatch.md

| 节点 | Workload 与相对父节点的设计变化 |

## t_50454f5373730b04 | dispatch.md | 77 chars | 1 declarations

dispatch.md:22 dispatch.md

| [q1_cluster](q1_cluster.yaml) | q1 的 cluster workload；改为 cluster 内合并与直接写回 |

## t_5c3dc93de3f9aae1 | dispatch.md | 180 chars | 1 declarations

dispatch.md:3 dispatch.md

本任务以每请求 4 个 query token、KV 分段后全局合并的通用设计作为 root。它包含分段计算、FP32 部分结果和独立合并 kernel；其他节点改变合并协议、缩减到单 block，或调整 query 打包与 softmax 分工。完整 Contract 只在 `root.yaml` 中定义，包含 query 长度 1 和 4 的共同有效域。

## t_67c626bc1bbd6796 | dispatch.md | 107 chars | 1 declarations

dispatch.md:27 dispatch.md

q1/q4 表示每请求的 query token 数；global、cluster、direct、compact 概括合并方式。cluster 成员数、global split 数及循环长度的变化保留在所属节点内。

## t_760fc5b9bc976b15 | dispatch.md | 168 chars | 1 declarations

dispatch.md:25 dispatch.md

树与各节点的 `meta_data.dispatch.parent` 对应，表示设计派生关系。root 和其他节点均按各自的 `when` 在共同契约域内参与选择；精确条件见 `meta_data.dispatch.when`，修订号见 `meta_data.version`，工作参数见 `design.work_schedule`。

## t_7d2d006647fba19c | dispatch.md | 102 chars | 1 declarations

dispatch.md:21 dispatch.md

| [q1_global](q1_global.yaml) | 每请求一个 query token；调整 query 打包，两个 softmax warpgroup 交替处理 block，保留全局合并 |

## t_80823c08c0d238c3 | dispatch.md | 97 chars | 1 declarations

dispatch.md:19 dispatch.md

| [q4_cluster](q4_cluster.yaml) | q4 的较短 KV 等 workload；改为 cluster 内交换统计量、BF16 reduce-add，直接写回输出 |

## t_83cd68420f42ee12 | dispatch.md | 71 chars | 1 declarations

dispatch.md:17 dispatch.md

| [root](root.yaml) | q4 的通用全局合并路径，每请求两个 query tile，分段结果由独立 kernel 合并 |

## t_b63dbe986757b882 | dispatch.md | 154 chars | 1 declarations

dispatch.md:5 dispatch.md

```mermaid flowchart TD root --> q4_compact root --> q4_cluster root --> q4_direct root --> q1_global q1_global --> q1_cluster q1_global --> q1_direct ```

## t_bc82ac6633c65f86 | dispatch.md | 64 chars | 1 declarations

dispatch.md:23 dispatch.md

| [q1_direct](q1_direct.yaml) | q1 的单 KV block；单 CTA 直接完成归约与输出 |

## t_4598d34e4c939b06 | dispatch.when | 64 chars | 1 declarations

q4_direct:9 meta_data.dispatch.when

Select exactly when q_len_per_req=4 and ceil(max_seq_len/256)=1.

## t_4b25ca98ef427c85 | dispatch.when | 398 chars | 1 declarations

q1_global:9 meta_data.dispatch.when

Let B=BatchCount, L=max_seq_len, E=64*PageCount, C=ceil(L/256), A=floor(E/B), skew=(2*E<B*L), and U=(B>=5 and L>=32768 and A>=(65536 if B=8, otherwise 32768) and not skew). G=(B<=4 and L>=24576) or (U and B>=8) or (skew and 5<=B<=8 and L>=65536) or (not skew and L>=24576 and 16<=B<=17). Select exactly when q_len_per_req=1, C>1 and G. Changes in the global split count share one two-kernel design.

## t_57c98f9651c308c5 | dispatch.when | 364 chars | 1 declarations

q4_compact:9 meta_data.dispatch.when

Let B=BatchCount, L=max_seq_len, P=PageCount, C=ceil(L/256), skew=(128*P<B*L), compact=(L>=24576 and ((not skew and B in {10,11}) or (skew and 10<=B<=17 and P>ceil(L/64)+16*(B-1) and (B<12 or P<B*ceil(L/64))))). Select exactly when q_len_per_req=4 and compact. Nominal S=8/32, near-uniform and proportional active counts share the serial-per-thread combine design.

## t_6ad6445ecac5df07 | dispatch.when | 269 chars | 1 declarations

q4_cluster:9 meta_data.dispatch.when

Let B=BatchCount, L=max_seq_len, C=ceil(L/256), and skew=(128*PageCount<B*L). Select exactly when q_len_per_req=4, C>1, and either L<=2048, or 2048<L<24576 and (B<=4 or (B>=10 and (B<17 or skew))). Short, small-batch and four-member policies share the cluster protocol.

## t_7b5c490da93280ea | dispatch.when | 64 chars | 1 declarations

q1_direct:9 meta_data.dispatch.when

Select exactly when q_len_per_req=1 and ceil(max_seq_len/256)=1.

## t_91fcf1fa6d850990 | dispatch.when | 394 chars | 1 declarations

q1_cluster:9 meta_data.dispatch.when

Let B=BatchCount, L=max_seq_len, E=64*PageCount, C=ceil(L/256), A=floor(E/B), skew=(2*E<B*L), and U=(B>=5 and L>=32768 and A>=(65536 if B=8, otherwise 32768) and not skew). G=(B<=4 and L>=24576) or (U and B>=8) or (skew and 5<=B<=8 and L>=65536) or (not skew and L>=24576 and 16<=B<=17). Select exactly when q_len_per_req=1, C>1 and not G. The nominal cluster sizes 4, 8 and 16 share this path.

## t_984b54c25216600a | dispatch.when | 418 chars | 1 declarations

root:8 meta_data.dispatch.when

Let B=BatchCount, L=max_seq_len, P=PageCount, C=ceil(L/256), skew=(128*P<B*L), compact=(L>=24576 and ((not skew and B in {10,11}) or (skew and 10<=B<=17 and P>ceil(L/64)+16*(B-1) and (B<12 or P<B*ceil(L/64))))). Select exactly when q_len_per_req=4, C>1, not compact, and either L>=24576 or 2048<L<24576 and ((5<=B<=9) or (B>=17 and not skew)). The wave-fit and occupancy split counts retain the same two-kernel design.

## t_680d6642a8764eac | hardware.execution | 55 chars | 1 declarations

root:78 contract.hardware_specification.execution.CTA

Warps co-resident on one SM, sharing SMEM and barriers.

## t_6f5eff329130bc7c | hardware.execution | 63 chars | 1 declarations

root:79 contract.hardware_specification.execution.cluster

CTAs co-scheduled within one GPC, with cluster synchronization.

## t_707ed1a5b6536df5 | hardware.execution | 31 chars | 1 declarations

root:77 contract.hardware_specification.execution.warp

32 SIMT threads within one CTA.

## t_263d5b133f5e61d6 | hardware.feature | 56 chars | 1 declarations

root:83 contract.hardware_specification.feature.register

Thread-private registers backed by the SM register file.

## t_6c67314fe3205270 | hardware.feature | 61 chars | 1 declarations

root:82 contract.hardware_specification.feature.TMEM

Explicitly allocated Tensor Core storage, separate from SMEM.

## t_bf65cfd621166b87 | hardware.feature | 101 chars | 1 declarations

root:84 contract.hardware_specification.feature.asynchronous_execution

Asynchronous transfers and MMA require their primitive-specific completion and visibility guarantees.

## t_fbe55b4f6cea5249 | hardware.feature | 88 chars | 1 declarations

root:81 contract.hardware_specification.feature.SMEM

CTA-lifetime shared storage, addressable by cluster peers while its owner remains alive.

## t_049cabe491d2aa4b | hardware.stack | 128 chars | 1 declarations

root:85 contract.hardware_specification.stack.toolchain

nvidia-cutlass-dsl 4.6.1 compiler; PyTorch 2.9.0+cu130; cuda-python 13.1.1; CUDA compiler and driver versions were not recorded.

## t_e8e7946973e5b096 | hardware.stack | 34 chars | 1 declarations

root:85 contract.hardware_specification.stack.language

CuTe DSL, nvidia-cutlass-dsl 4.6.1

## t_0c51c9bd9a11185d | math.arithmetic | 282 chars | 2 declarations

q1_global:112 design.math_reshape.rewrite[4].arithmetic.accumulate; root:189 design.math_reshape.rewrite[4].arithmetic.accumulate

Each warp reduces maxima/sums over at most two splits per thread lane; it broadcasts weights and accumulates all nominal numerators in ascending split order. Scalar contraction is unknown: FP32 multiply-add contraction has not been confirmed for the recorded CuTe DSL compiler path.

## t_2452f4ca178f9929 | math.arithmetic | 124 chars | 2 declarations

q1_direct:60 design.math_reshape.rewrite[2].arithmetic.accumulate; q4_direct:60 design.math_reshape.rewrite[2].arithmetic.accumulate

Each reduction slice accumulates its leading product then its residual into the same FP32 Y; Llocal sums four warp partials.

## t_452b1db30d433831 | math.arithmetic | 305 chars | 2 declarations

q1_cluster:100 design.math_reshape.rewrite[3].arithmetic.accumulate; q4_cluster:99 design.math_reshape.rewrite[3].arithmetic.accumulate

Cluster max and sum use XOR-partner rounds; output accumulation takes separately BF16-rounded split contributions and has no fixed split order. Native BF16 reduce-add internal arithmetic and subnormal handling are unknown: native BF16 TMA reduce-add rounding and subnormal behavior have not been verified.

## t_505b01e9b6cb4ac7 | math.arithmetic | 138 chars | 5 declarations

q1_cluster:99 design.math_reshape.rewrite[3].arithmetic.approx; q1_global:111 design.math_reshape.rewrite[4].arithmetic.approx; q4_cluster:98 design.math_reshape.rewrite[3].arithmetic.approx; q4_compact:110 design.math_reshape.rewrite[4].arithmetic.approx; root:188 design.math_reshape.rewrite[4].arithmetic.approx

exp2_fast follows step exponent; rcp_fast is the source approximate reciprocal; unknown: no verified error bound for this reciprocal path.

## t_60c370728e194490 | math.arithmetic | 113 chars | 2 declarations

q1_direct:48 design.math_reshape.rewrite[1].arithmetic.approx; q4_direct:48 design.math_reshape.rewrite[1].arithmetic.approx

exp2_fast is the source fast-math base-two exponential; unknown: no verified error bound for this fast-exp2 path.

## t_6629e1de979255ab | math.arithmetic | 255 chars | 1 declarations

q4_compact:111 design.math_reshape.rewrite[4].arithmetic.accumulate

Each output thread serially recomputes the maximum, then scans active splits in ascending order for the denominator and numerator. Scalar contraction is unknown: FP32 multiply-add contraction has not been confirmed for the recorded CuTe DSL compiler path.

## t_8720ec2de5200945 | math.arithmetic | 32 chars | 5 declarations

q1_cluster:81 design.math_reshape.rewrite[2].arithmetic.approx; q1_global:81 design.math_reshape.rewrite[2].arithmetic.approx; q4_cluster:80 design.math_reshape.rewrite[2].arithmetic.approx; q4_compact:81 design.math_reshape.rewrite[2].arithmetic.approx; root:158 design.math_reshape.rewrite[2].arithmetic.approx

exp2_fast follows step exponent.

## t_8d178fa3ae170a75 | math.arithmetic | 105 chars | 2 declarations

q1_direct:59 design.math_reshape.rewrite[2].arithmetic.approx; q4_direct:59 design.math_reshape.rewrite[2].arithmetic.approx

rcp_fast is the source approximate reciprocal; unknown: no verified error bound for this reciprocal path.

## t_ad776cbb8052eda7 | math.arithmetic | 278 chars | 5 declarations

q1_cluster:82 design.math_reshape.rewrite[2].arithmetic.accumulate; q1_global:82 design.math_reshape.rewrite[2].arithmetic.accumulate; q4_cluster:81 design.math_reshape.rewrite[2].arithmetic.accumulate; q4_compact:82 design.math_reshape.rewrite[2].arithmetic.accumulate; root:159 design.math_reshape.rewrite[2].arithmetic.accumulate

Each native V reduction slice accumulates P_hi then P_lo into the same FP32 Y. Lbase scales the old state and current partial separately before addition; scalar contraction is unknown: FP32 multiply-add contraction has not been confirmed for the recorded CuTe DSL compiler path.

## t_ebbe507d9c868402 | math.arithmetic | 115 chars | 5 declarations

q1_cluster:57 design.math_reshape.rewrite[1].arithmetic.approx; q1_global:57 design.math_reshape.rewrite[1].arithmetic.approx; q4_cluster:56 design.math_reshape.rewrite[1].arithmetic.approx; q4_compact:57 design.math_reshape.rewrite[1].arithmetic.approx; root:134 design.math_reshape.rewrite[1].arithmetic.approx

exp2_fast is the source's fast-math base-two exponential; unknown: no verified error bound for this fast-exp2 path.

## t_123eee0091eaca78 | math.ledger | 11 chars | 3 declarations

q1_global:121 design.math_reshape.ledger.traffic[1].by[0]; q4_compact:120 design.math_reshape.ledger.traffic[1].by[0]; root:198 design.math_reshape.ledger.traffic[1].by[0]

materialize

## t_17cff8816150360a | math.ledger | 106 chars | 5 declarations

q1_cluster:112 design.math_reshape.ledger.depth[0].change; q1_global:123 design.math_reshape.ledger.depth[0].change; q4_cluster:111 design.math_reshape.ledger.depth[0].change; q4_compact:122 design.math_reshape.ledger.depth[0].change; root:200 design.math_reshape.ledger.depth[0].change

T_b -> max_r N_r sequential block updates per query, followed by a SplitCount-way normalization reduction.

## t_2ae76a85c8e1dc75 | math.ledger | 77 chars | 5 declarations

q1_cluster:108 design.math_reshape.ledger.matmul[0].change; q1_global:117 design.math_reshape.ledger.matmul[0].change; q4_cluster:107 design.math_reshape.ledger.matmul[0].change; q4_compact:116 design.math_reshape.ledger.matmul[0].change; root:194 design.math_reshape.ledger.matmul[0].change

2*D*L_b -> 3*D*Bk*T_b MAC per query head; the added contraction applies P_lo.

## t_3159361f1d06f44b | math.ledger | 105 chars | 2 declarations

q1_direct:68 design.math_reshape.ledger.error[0].change; q4_direct:68 design.math_reshape.ledger.error[0].change

+two FP8 roundings per exponential; denominator remains FP32; fast exp2 and reciprocal add approximation.

## t_351652649fcc3acf | math.ledger | 8 chars | 19 declarations

q1_cluster:108 design.math_reshape.ledger.matmul[0].by[1]; q1_cluster:110 design.math_reshape.ledger.traffic[0].by[1]; q1_cluster:115 design.math_reshape.ledger.error[0].by[0]; q1_direct:64 design.math_reshape.ledger.traffic[0].by[1]; q1_direct:68 design.math_reshape.ledger.error[0].by[0]; q1_global:117 design.math_reshape.ledger.matmul[0].by[1]; q1_global:119 design.math_reshape.ledger.traffic[0].by[1]; q1_global:126 design.math_reshape.ledger.error[0].by[0]; q4_cluster:107 design.math_reshape.ledger.matmul[0].by[1]; q4_cluster:109 design.math_reshape.ledger.traffic[0].by[1]; q4_cluster:114 design.math_reshape.ledger.error[0].by[0]; q4_compact:116 design.math_reshape.ledger.matmul[0].by[1]; q4_compact:118 design.math_reshape.ledger.traffic[0].by[1]; q4_compact:125 design.math_reshape.ledger.error[0].by[0]; q4_direct:64 design.math_reshape.ledger.traffic[0].by[1]; q4_direct:68 design.math_reshape.ledger.error[0].by[0]; root:194 design.math_reshape.ledger.matmul[0].by[1]; root:196 design.math_reshape.ledger.traffic[0].by[1]; root:203 design.math_reshape.ledger.error[0].by[0]

exponent

## t_4a33e1385fcc6825 | math.ledger | 26 chars | 2 declarations

q1_direct:68 design.math_reshape.ledger.error[0].item; q4_direct:68 design.math_reshape.ledger.error[0].item

Probability representation

## t_4dacd6585b42cc7d | math.ledger | 143 chars | 3 declarations

q1_global:120 design.math_reshape.ledger.traffic[1].change; q4_compact:119 design.math_reshape.ledger.traffic[1].change; root:197 design.math_reshape.ledger.traffic[1].change

+8*SplitCount*q_len_per_req*HeadCount*(D+2) bytes per request as the full-active write/read payload bound; inactive numerators are not written.

## t_54a6c1160df95724 | math.ledger | 15 chars | 3 declarations

q1_global:120 design.math_reshape.ledger.traffic[1].item; q4_compact:119 design.math_reshape.ledger.traffic[1].item; root:197 design.math_reshape.ledger.traffic[1].item

Split workspace

## t_54b9fc89ab2b4034 | math.ledger | 196 chars | 2 declarations

q1_cluster:114 design.math_reshape.ledger.error[0].change; q4_cluster:113 design.math_reshape.ledger.error[0].change

+two FP8 probability roundings, fast exp2/reciprocal, parity and split regrouping; the denominator keeps unquantized FP32 probabilities. Each split also rounds to BF16 before unordered reduce-add.

## t_626aff5e7c43649a | math.ledger | 6 chars | 10 declarations

q1_cluster:108 design.math_reshape.ledger.matmul[0].by[2]; q1_cluster:115 design.math_reshape.ledger.error[0].by[1]; q1_global:117 design.math_reshape.ledger.matmul[0].by[2]; q1_global:126 design.math_reshape.ledger.error[0].by[1]; q4_cluster:107 design.math_reshape.ledger.matmul[0].by[2]; q4_cluster:114 design.math_reshape.ledger.error[0].by[1]; q4_compact:116 design.math_reshape.ledger.matmul[0].by[2]; q4_compact:125 design.math_reshape.ledger.error[0].by[1]; root:194 design.math_reshape.ledger.matmul[0].by[2]; root:203 design.math_reshape.ledger.error[0].by[1]

parity

## t_6467a32e67178bdc | math.ledger | 136 chars | 3 declarations

q1_global:125 design.math_reshape.ledger.error[0].change; q4_compact:124 design.math_reshape.ledger.error[0].change; root:202 design.math_reshape.ledger.error[0].change

+two FP8 probability roundings, fast exp2/reciprocal, parity and split regrouping; the denominator keeps unquantized FP32 probabilities.

## t_6df0883818c9b22a | math.ledger | 6 chars | 15 declarations

q1_cluster:108 design.math_reshape.ledger.matmul[0].by[0]; q1_cluster:110 design.math_reshape.ledger.traffic[0].by[0]; q1_cluster:112 design.math_reshape.ledger.depth[0].by[0]; q1_global:117 design.math_reshape.ledger.matmul[0].by[0]; q1_global:119 design.math_reshape.ledger.traffic[0].by[0]; q1_global:123 design.math_reshape.ledger.depth[0].by[0]; q4_cluster:107 design.math_reshape.ledger.matmul[0].by[0]; q4_cluster:109 design.math_reshape.ledger.traffic[0].by[0]; q4_cluster:111 design.math_reshape.ledger.depth[0].by[0]; q4_compact:116 design.math_reshape.ledger.matmul[0].by[0]; q4_compact:118 design.math_reshape.ledger.traffic[0].by[0]; q4_compact:122 design.math_reshape.ledger.depth[0].by[0]; root:194 design.math_reshape.ledger.matmul[0].by[0]; root:196 design.math_reshape.ledger.traffic[0].by[0]; root:200 design.math_reshape.ledger.depth[0].by[0]

stripe

## t_6edf48098570bb81 | math.ledger | 9 chars | 17 declarations

q1_cluster:112 design.math_reshape.ledger.depth[0].by[1]; q1_cluster:115 design.math_reshape.ledger.error[0].by[2]; q1_direct:66 design.math_reshape.ledger.matmul[0].by[1]; q1_direct:68 design.math_reshape.ledger.error[0].by[1]; q1_global:121 design.math_reshape.ledger.traffic[1].by[1]; q1_global:123 design.math_reshape.ledger.depth[0].by[1]; q1_global:126 design.math_reshape.ledger.error[0].by[2]; q4_cluster:111 design.math_reshape.ledger.depth[0].by[1]; q4_cluster:114 design.math_reshape.ledger.error[0].by[2]; q4_compact:120 design.math_reshape.ledger.traffic[1].by[1]; q4_compact:122 design.math_reshape.ledger.depth[0].by[1]; q4_compact:125 design.math_reshape.ledger.error[0].by[2]; q4_direct:66 design.math_reshape.ledger.matmul[0].by[1]; q4_direct:68 design.math_reshape.ledger.error[0].by[1]; root:198 design.math_reshape.ledger.traffic[1].by[1]; root:200 design.math_reshape.ledger.depth[0].by[1]; root:203 design.math_reshape.ledger.error[0].by[2]

normalize

## t_87bd5d06faf37059 | math.ledger | 96 chars | 2 declarations

q1_direct:64 design.math_reshape.ledger.traffic[0].change; q4_direct:64 design.math_reshape.ledger.traffic[0].change

-16*q_len_per_req*HeadCount*seq_lens[b] bytes per request of score/probability writes and reads.

## t_9d3577b87dffa1c1 | math.ledger | 88 chars | 2 declarations

q1_direct:66 design.math_reshape.ledger.matmul[0].change; q4_direct:66 design.math_reshape.ledger.matmul[0].change

2*D*seq_lens[b] -> 3*D*Bk MAC per query head; padding and the residual product add work.

## t_a6bc41e9ce2671f1 | math.ledger | 19 chars | 7 declarations

q1_cluster:110 design.math_reshape.ledger.traffic[0].item; q1_direct:64 design.math_reshape.ledger.traffic[0].item; q1_global:119 design.math_reshape.ledger.traffic[0].item; q4_cluster:109 design.math_reshape.ledger.traffic[0].item; q4_compact:118 design.math_reshape.ledger.traffic[0].item; q4_direct:64 design.math_reshape.ledger.traffic[0].item; root:196 design.math_reshape.ledger.traffic[0].item

Dense intermediates

## t_ade0e9694a4d3dbd | math.ledger | 89 chars | 5 declarations

q1_cluster:110 design.math_reshape.ledger.traffic[0].change; q1_global:119 design.math_reshape.ledger.traffic[0].change; q4_cluster:109 design.math_reshape.ledger.traffic[0].change; q4_compact:118 design.math_reshape.ledger.traffic[0].change; root:196 design.math_reshape.ledger.traffic[0].change

-16*q_len_per_req*HeadCount*L_b bytes per request for score/probability writes and reads.

## t_b84cabce2369c341 | math.ledger | 5 chars | 4 declarations

q1_direct:64 design.math_reshape.ledger.traffic[0].by[0]; q1_direct:66 design.math_reshape.ledger.matmul[0].by[0]; q4_direct:64 design.math_reshape.ledger.traffic[0].by[0]; q4_direct:66 design.math_reshape.ledger.matmul[0].by[0]

block

## t_bc9351ae15cd7d52 | math.ledger | 8 chars | 5 declarations

q1_cluster:112 design.math_reshape.ledger.depth[0].item; q1_global:123 design.math_reshape.ledger.depth[0].item; q4_cluster:111 design.math_reshape.ledger.depth[0].item; q4_compact:122 design.math_reshape.ledger.depth[0].item; root:200 design.math_reshape.ledger.depth[0].item

KV chain

## t_d3690b7906f854fa | math.ledger | 12 chars | 7 declarations

q1_cluster:108 design.math_reshape.ledger.matmul[0].item; q1_direct:66 design.math_reshape.ledger.matmul[0].item; q1_global:117 design.math_reshape.ledger.matmul[0].item; q4_cluster:107 design.math_reshape.ledger.matmul[0].item; q4_compact:116 design.math_reshape.ledger.matmul[0].item; q4_direct:66 design.math_reshape.ledger.matmul[0].item; root:194 design.math_reshape.ledger.matmul[0].item

Contractions

## t_de21e27bf409662a | math.ledger | 77 chars | 7 declarations

q1_cluster:106 design.math_reshape.ledger.baseline; q1_direct:62 design.math_reshape.ledger.baseline; q1_global:115 design.math_reshape.ledger.baseline; q4_cluster:105 design.math_reshape.ledger.baseline; q4_compact:114 design.math_reshape.ledger.baseline; q4_direct:62 design.math_reshape.ledger.baseline; root:192 design.math_reshape.ledger.baseline

Dense FP32 scores/probabilities in global memory, with one value contraction.

## t_ec2acf6b1781adf9 | math.ledger | 8 chars | 5 declarations

q1_cluster:114 design.math_reshape.ledger.error[0].item; q1_global:125 design.math_reshape.ledger.error[0].item; q4_cluster:113 design.math_reshape.ledger.error[0].item; q4_compact:124 design.math_reshape.ledger.error[0].item; root:202 design.math_reshape.ledger.error[0].item

Numerics

## t_20d78f5e90e9f721 | math.math | 469 chars | 2 declarations

q1_direct:18 design.math_reshape.rewrite[0].math; q4_direct:18 design.math_reshape.rewrite[0].math

For request b and query block q, x=h+HeadCount*t, 0<=t<QueryTile, 0<=h<HeadCount, t_abs=q*QueryTile+t, and Qpack[x,:]=query[b*q_len_per_req+t_abs,h,:]. For 0<=u<Bk, let page=floor(u/PageSize), id=block_tables[b,page]. Kblk[u,:]=kv_cache[id,0,0,u%PageSize,:], Vblk[u,:]=kv_cache[id,1,0,u%PageSize,:] if id>=0; otherwise zero. The speculative first-page lookup includes unused table entries; visibility is enforced on S, not on gathered K/V. ScoreBlock=mm32(Kblk,Qpackᵀ).

## t_31c1bfd083f1619f | math.math | 187 chars | 2 declarations

q1_direct:51 design.math_reshape.rewrite[2].math; q4_direct:51 design.math_reshape.rewrite[2].math

Y=mma32(Vblkᵀ,P_lo,mm32(Vblkᵀ,P_hi)). Normalization[x]=rcp_fast(Llocal[x])*fp32(bmm2_scale/rho). output[b*q_len_per_req+q*QueryTile+t,h,d]=bf16(Y[d,x]*Normalization[x]); out_after=output.

## t_44b6922db5c6bf23 | math.math | 352 chars | 2 declarations

q1_global:86 design.math_reshape.rewrite[3].math; root:163 design.math_reshape.rewrite[3].math

For x=h+HeadCount*t and t_abs=q*QueryTile+t: Mpartial[r,b,t_abs,h]=Mlocal[x]. Lpartial[r,b,t_abs,h]=Llocal[x]. Opartial[r,b,t_abs,h,d]=Ylocal[d,x] if N_r>0. An empty split writes Mpartial=-inf and Lpartial=0; its Opartial payload is not overwritten. For N_r=0, the Opartial scratch view retains its prior finite FP32 value; it is not a computed Ylocal.

## t_8573899d736ecd67 | math.math | 410 chars | 2 declarations

q1_direct:35 design.math_reshape.rewrite[1].math; q4_direct:35 design.math_reshape.rewrite[1].math

c=fp32(bmm1_scale*log2(e)); rho=448; MinFinite=-3.4028234663852886e38. S[u,x]=ScoreBlock[u,x] if u is in visible_key(b,q*QueryTile+floor(x/HeadCount)), otherwise -inf. m[x]=c*max(MinFinite,max_u S[u,x]). E[u,x]=exp2_fast(c*S[u,x]-m[x]) for visible keys; otherwise zero. P_hi=fp8_e4m3fn(rho*E); P_lo=fp8_e4m3fn(rho*E-P_hi). ExponentialSum[v,x]=E[v,x]+E[v+128,x], 0<=v<128; Llocal[x]=sum32_v ExponentialSum[v,x].

## t_8c887532ff65ad27 | math.math | 452 chars | 1 declarations

q4_compact:100 design.math_reshape.rewrite[4].math

For each (b,t_abs,h), let R={r | 0<=r<SplitCount and Mpartial[r,b,t_abs,h]>-inf}. M=max_{r in R} Mpartial[r,b,t_abs,h]. w[r]=exp2_fast(Mpartial[r,b,t_abs,h]-M) for r in R; otherwise 0. Z=sum32_{r in R} w[r]*Lpartial[r,b,t_abs,h]. U[d]=sum32_{r in R} w[r]*Opartial[r,b,t_abs,h,d]. output[b*q_len_per_req+t_abs,h,d]=bf16(U[d]*(rcp_fast(Z)*fp32(bmm2_scale/rho))). Empty splits have zero weight; their scratch payload does not contribute. out_after=output.

## t_a3491d84ac88f5f9 | math.math | 249 chars | 1 declarations

q4_compact:86 design.math_reshape.rewrite[3].math

For x=h+HeadCount*t and t_abs=q*QueryTile+t: Mpartial[r,b,t_abs,h]=Mlocal[x]. Lpartial[r,b,t_abs,h]=Llocal[x]. Opartial[r,b,t_abs,h,d]=Ylocal[d,x] if N_r>0. An empty split writes Mpartial=-inf and Lpartial=0; its Opartial payload is not overwritten.

## t_c5aea2665a87787d | math.math | 590 chars | 5 declarations

q1_cluster:22 design.math_reshape.rewrite[0].math; q1_global:22 design.math_reshape.rewrite[0].math; q4_cluster:21 design.math_reshape.rewrite[0].math; q4_compact:22 design.math_reshape.rewrite[0].math; root:99 design.math_reshape.rewrite[0].math

For request b and query block q, let x=h+HeadCount*t, 0<=t<QueryTile, 0<=h<HeadCount, t_abs=q*QueryTile+t, Qpack[x,:]=query[b*q_len_per_req+t_abs,h,:], and T_b=ceil(seq_lens[b]/Bk). For split r in [0,SplitCount), N_r=max(0,ceil((T_b-r)/WorkSplit(b))) if r<WorkSplit(b); otherwise 0. For 0<=i<N_r, 0<=u<Bk, k=(r+i*WorkSplit(b))*Bk+u: Let page=floor(k/PageSize); id=block_tables[b,page] if i=0 or page<ceil(seq_lens[b]/PageSize), otherwise -1. Kblk(i)[u,:]=kv_cache[id,0,0,k%PageSize,:], Vblk(i)[u,:]=kv_cache[id,1,0,k%PageSize,:] if id>=0; otherwise zero. ScoreBlock(i)=mm32(Kblk(i),Qpackᵀ).

## t_cc1683cb88052337 | math.math | 641 chars | 2 declarations

q1_cluster:86 design.math_reshape.rewrite[3].math; q4_cluster:85 design.math_reshape.rewrite[3].math

For each (b,q,x), with split indices restored: M=max_{r=0}^{SplitCount-1} Mlocal(r)[x]. w[r]=exp2_fast(Mlocal(r)[x]-M) if Mlocal(r)[x]>-inf; otherwise 0. Z=sum32_{r=0}^{SplitCount-1} w[r]*Llocal(r)[x]. Normalization(r)[x]=w[r]*rcp_fast(Z)*fp32(bmm2_scale/rho). Contribution(r)[d,x]=bf16(Ylocal(r)[d,x]*Normalization(r)[x]) only for r with N_r>0. For x=h+HeadCount*t and t_abs=q*QueryTile+t: output[b*q_len_per_req+t_abs,h,d]=reduce_add_bf16_{r:N_r>0} Contribution(r)[d,x]. reduce_add_bf16 starts from BF16 zero and adds the BF16 contributions in runtime completion order into the BF16 destination; empty splits add nothing. out_after=output.

## t_e013aa9961e40ad1 | math.math | 688 chars | 5 declarations

q1_cluster:60 design.math_reshape.rewrite[2].math; q1_global:60 design.math_reshape.rewrite[2].math; q4_cluster:59 design.math_reshape.rewrite[2].math; q4_compact:60 design.math_reshape.rewrite[2].math; root:137 design.math_reshape.rewrite[2].math

Initialize Y(-2)=Y(-1)=0; Lbase(0)=Lbase(1)=0; omit (b,q,r). For 0<=i<N_r: a(i)[x]=0 if i<2, otherwise exp2_fast(m(i-2)[x]-m(i)[x]). Y(i)=mma32(Vblk(i)ᵀ,P_lo(i),mma32(Vblk(i)ᵀ,P_hi(i),Y(i-2) ⊙ a(i)[None,:])). If i>=2: Lbase(i)=a(i)[None,:] ⊙ Lbase(i-2)+a(i)[None,:] ⊙ ExponentialSum(i-2). If N_r=0: Mlocal=-inf, Llocal=0; no Ylocal payload is produced. If N_r>0, let f=N_r-1, p=N_r-2; z=exp2_fast(m(p)-m(f)) if N_r>1, otherwise 1: Mlocal=m(f); Ylocal=Y(f)+(Y(p) ⊙ z[None,:] if N_r>1, otherwise zero). Lt=Lbase(f)+ExponentialSum(f)+(z[None,:] ⊙ (Lbase(p)+ExponentialSum(p)) if N_r>1, otherwise zero). FinalSum[x,g]=sum32_{v=32*g}^{32*g+31} Lt[v,x], 0<=g<4; Llocal[x]=sum32_g FinalSum[x,g].

## t_e29b372af0983234 | math.math | 489 chars | 5 declarations

q1_cluster:42 design.math_reshape.rewrite[1].math; q1_global:42 design.math_reshape.rewrite[1].math; q4_cluster:41 design.math_reshape.rewrite[1].math; q4_compact:42 design.math_reshape.rewrite[1].math; root:119 design.math_reshape.rewrite[1].math

For each (b,q,r), initialize m(-1)=-inf; c=fp32(bmm1_scale*log2(e)). For 0<=i<N_r: S(i)[u,x]=ScoreBlock(i)[u,x] if (r+i*WorkSplit(b))*Bk+u is in visible_key(b,q*QueryTile+floor(x/HeadCount)), otherwise -inf. m(i)[x]=max(m(i-1)[x],c*max(MinFinite,max_u S(i)[u,x])). Eblock(i)[u,x]=exp2_fast(c*S(i)[u,x]-m(i)[x]) on visible keys, otherwise zero. P_hi(i)=fp8_e4m3fn(rho*Eblock(i)); P_lo(i)=fp8_e4m3fn(rho*Eblock(i)-P_hi(i)). ExponentialSum(i)[v,x]=Eblock(i)[v,x]+Eblock(i)[v+128,x], 0<=v<128.

## t_e3c759f8d74c5689 | math.math | 511 chars | 2 declarations

q1_global:101 design.math_reshape.rewrite[4].math; root:178 design.math_reshape.rewrite[4].math

For each (b,t_abs,h), let R={r | 0<=r<SplitCount and Mpartial[r,b,t_abs,h]>-inf}. M=max_{r in R} Mpartial[r,b,t_abs,h]. w[r]=exp2_fast(Mpartial[r,b,t_abs,h]-M) for r in R; otherwise 0. Z=sum32_{r in R} w[r]*Lpartial[r,b,t_abs,h]. U[d]=sum32_{r=0}^{SplitCount-1} w[r]*Opartial[r,b,t_abs,h,d]. output[b*q_len_per_req+t_abs,h,d]=bf16(U[d]*(rcp_fast(Z)*fp32(bmm2_scale/rho))). Empty payloads are read with zero weight and must remain finite; first-use zeroing establishes this for untouched slots. out_after=output.

## t_2101339577fa0c25 | math.meaning | 28 chars | 5 declarations

q1_cluster:31 design.math_reshape.rewrite[0].symbol.SplitCount.meaning; q1_global:31 design.math_reshape.rewrite[0].symbol.SplitCount.meaning; q4_cluster:30 design.math_reshape.rewrite[0].symbol.SplitCount.meaning; q4_compact:31 design.math_reshape.rewrite[0].symbol.SplitCount.meaning; root:108 design.math_reshape.rewrite[0].symbol.SplitCount.meaning

Number of nominal KV splits.

## t_219036233c3c8bc6 | math.meaning | 18 chars | 2 declarations

q1_direct:29 design.math_reshape.rewrite[0].symbol.Qpack.meaning; q4_direct:29 design.math_reshape.rewrite[0].symbol.Qpack.meaning

Packed query rows.

## t_21e5ba8f17859b0f | math.meaning | 32 chars | 3 declarations

q1_global:96 design.math_reshape.rewrite[3].symbol.Lpartial.meaning; q4_compact:95 design.math_reshape.rewrite[3].symbol.Lpartial.meaning; root:173 design.math_reshape.rewrite[3].symbol.Lpartial.meaning

Materialized split denominators.

## t_24b435bd9396e4d1 | math.meaning | 38 chars | 5 declarations

q1_cluster:36 design.math_reshape.rewrite[0].symbol.Qpack.meaning; q1_global:36 design.math_reshape.rewrite[0].symbol.Qpack.meaning; q4_cluster:35 design.math_reshape.rewrite[0].symbol.Qpack.meaning; q4_compact:36 design.math_reshape.rewrite[0].symbol.Qpack.meaning; root:113 design.math_reshape.rewrite[0].symbol.Qpack.meaning

Packed query-head and query-tail rows.

## t_2ac158af88aa4fc3 | math.meaning | 48 chars | 5 declarations

q1_cluster:54 design.math_reshape.rewrite[1].symbol.P_hi.meaning; q1_global:54 design.math_reshape.rewrite[1].symbol.P_hi.meaning; q4_cluster:53 design.math_reshape.rewrite[1].symbol.P_hi.meaning; q4_compact:54 design.math_reshape.rewrite[1].symbol.P_hi.meaning; root:131 design.math_reshape.rewrite[1].symbol.P_hi.meaning

Leading FP8 component of the scaled exponential.

## t_34f018092779c4e6 | math.meaning | 50 chars | 5 declarations

q1_cluster:72 design.math_reshape.rewrite[2].symbol.Y.meaning; q1_global:72 design.math_reshape.rewrite[2].symbol.Y.meaning; q4_cluster:71 design.math_reshape.rewrite[2].symbol.Y.meaning; q4_compact:72 design.math_reshape.rewrite[2].symbol.Y.meaning; root:149 design.math_reshape.rewrite[2].symbol.Y.meaning

FP32 numerator of one parity, still scaled by rho.

## t_521dd4ff170e7759 | math.meaning | 31 chars | 2 declarations

q1_direct:28 design.math_reshape.rewrite[0].symbol.PackedColumns.meaning; q4_direct:28 design.math_reshape.rewrite[0].symbol.PackedColumns.meaning

Packed query-head column count.

## t_52acfbc246509f0c | math.meaning | 25 chars | 5 declarations

q1_cluster:33 design.math_reshape.rewrite[0].symbol.Bk.meaning; q1_global:33 design.math_reshape.rewrite[0].symbol.Bk.meaning; q4_cluster:32 design.math_reshape.rewrite[0].symbol.Bk.meaning; q4_compact:33 design.math_reshape.rewrite[0].symbol.Bk.meaning; root:110 design.math_reshape.rewrite[0].symbol.Bk.meaning

Logical key block length.

## t_5e21e7576fa9a618 | math.meaning | 38 chars | 5 declarations

q1_cluster:34 design.math_reshape.rewrite[0].symbol.QueryTile.meaning; q1_global:34 design.math_reshape.rewrite[0].symbol.QueryTile.meaning; q4_cluster:33 design.math_reshape.rewrite[0].symbol.QueryTile.meaning; q4_compact:34 design.math_reshape.rewrite[0].symbol.QueryTile.meaning; root:111 design.math_reshape.rewrite[0].symbol.QueryTile.meaning

Tail queries sharing one KV traversal.

## t_634ad219ea9d9d59 | math.meaning | 23 chars | 5 declarations

q1_cluster:76 design.math_reshape.rewrite[2].symbol.Ylocal.meaning; q1_global:76 design.math_reshape.rewrite[2].symbol.Ylocal.meaning; q4_cluster:75 design.math_reshape.rewrite[2].symbol.Ylocal.meaning; q4_compact:76 design.math_reshape.rewrite[2].symbol.Ylocal.meaning; root:153 design.math_reshape.rewrite[2].symbol.Ylocal.meaning

Active split numerator.

## t_671c4977cc26a762 | math.meaning | 30 chars | 2 declarations

q1_direct:46 design.math_reshape.rewrite[1].symbol.ExponentialSum.meaning; q4_direct:46 design.math_reshape.rewrite[1].symbol.ExponentialSum.meaning

FP32 probability partial sums.

## t_6b95040864838f9c | math.meaning | 14 chars | 5 declarations

q1_cluster:74 design.math_reshape.rewrite[2].symbol.Mlocal.meaning; q1_global:74 design.math_reshape.rewrite[2].symbol.Mlocal.meaning; q4_cluster:73 design.math_reshape.rewrite[2].symbol.Mlocal.meaning; q4_compact:74 design.math_reshape.rewrite[2].symbol.Mlocal.meaning; root:151 design.math_reshape.rewrite[2].symbol.Mlocal.meaning

Split maximum.

## t_6ce8060460ace5c0 | math.meaning | 52 chars | 1 declarations

q4_compact:96 design.math_reshape.rewrite[3].symbol.Opartial.meaning

Active split numerators; empty slots remain scratch.

## t_7df8a9c58ebe6e79 | math.meaning | 93 chars | 2 declarations

q1_global:97 design.math_reshape.rewrite[3].symbol.Opartial.meaning; root:174 design.math_reshape.rewrite[3].symbol.Opartial.meaning

FP32 scratch view; active numerators overwrite it, inactive slots retain finite prior values.

## t_87c8306e6b6c04a6 | math.meaning | 45 chars | 5 declarations

q1_cluster:35 design.math_reshape.rewrite[0].symbol.PackedColumns.meaning; q1_global:35 design.math_reshape.rewrite[0].symbol.PackedColumns.meaning; q4_cluster:34 design.math_reshape.rewrite[0].symbol.PackedColumns.meaning; q4_compact:35 design.math_reshape.rewrite[0].symbol.PackedColumns.meaning; root:112 design.math_reshape.rewrite[0].symbol.PackedColumns.meaning

Head-query column count, HeadCount*QueryTile.

## t_88f9ece419a83e1c | math.meaning | 25 chars | 2 declarations

q1_direct:30 design.math_reshape.rewrite[0].symbol.Kblk.meaning; q4_direct:30 design.math_reshape.rewrite[0].symbol.Kblk.meaning

Padded logical key block.

## t_904b93b7647d420b | math.meaning | 23 chars | 2 declarations

q1_direct:56 design.math_reshape.rewrite[2].symbol.Y.meaning; q4_direct:56 design.math_reshape.rewrite[2].symbol.Y.meaning

Scaled local numerator.

## t_97c9a5bf4ccd8bc2 | math.meaning | 56 chars | 5 declarations

q1_cluster:77 design.math_reshape.rewrite[2].symbol.Lbase.meaning; q1_global:77 design.math_reshape.rewrite[2].symbol.Lbase.meaning; q4_cluster:76 design.math_reshape.rewrite[2].symbol.Lbase.meaning; q4_compact:77 design.math_reshape.rewrite[2].symbol.Lbase.meaning; root:154 design.math_reshape.rewrite[2].symbol.Lbase.meaning

Corrected per-thread denominator carried on each parity.

## t_9f07886eb6ad13de | math.meaning | 34 chars | 5 declarations

q1_cluster:32 design.math_reshape.rewrite[0].symbol.WorkSplit.meaning; q1_global:32 design.math_reshape.rewrite[0].symbol.WorkSplit.meaning; q4_cluster:31 design.math_reshape.rewrite[0].symbol.WorkSplit.meaning; q4_compact:32 design.math_reshape.rewrite[0].symbol.WorkSplit.meaning; root:109 design.math_reshape.rewrite[0].symbol.WorkSplit.meaning

Request-to-active-split-count map.

## t_a7ab7526b03c2160 | math.meaning | 35 chars | 5 declarations

q1_cluster:55 design.math_reshape.rewrite[1].symbol.P_lo.meaning; q1_global:55 design.math_reshape.rewrite[1].symbol.P_lo.meaning; q4_cluster:54 design.math_reshape.rewrite[1].symbol.P_lo.meaning; q4_compact:55 design.math_reshape.rewrite[1].symbol.P_lo.meaning; root:132 design.math_reshape.rewrite[1].symbol.P_lo.meaning

Independently rounded FP8 residual.

## t_a9df29fdb884578f | math.meaning | 31 chars | 2 declarations

q1_direct:27 design.math_reshape.rewrite[0].symbol.QueryTile.meaning; q4_direct:27 design.math_reshape.rewrite[0].symbol.QueryTile.meaning

Queries sharing a KV traversal.

## t_a9e0c7c9fe6a25a3 | math.meaning | 35 chars | 2 declarations

q1_cluster:104 design.math_reshape.rewrite[3].symbol.Contribution.meaning; q4_cluster:103 design.math_reshape.rewrite[3].symbol.Contribution.meaning

Normalized BF16 split contribution.

## t_aa19ea4245377bd5 | math.meaning | 24 chars | 5 declarations

q1_cluster:37 design.math_reshape.rewrite[0].symbol.Kblk.meaning; q1_global:37 design.math_reshape.rewrite[0].symbol.Kblk.meaning; q4_cluster:36 design.math_reshape.rewrite[0].symbol.Kblk.meaning; q4_compact:37 design.math_reshape.rewrite[0].symbol.Kblk.meaning; root:114 design.math_reshape.rewrite[0].symbol.Kblk.meaning

Logical paged key block.

## t_adc427b7c1c3fe69 | math.meaning | 17 chars | 4 declarations

q1_cluster:103 design.math_reshape.rewrite[3].symbol.Normalization.meaning; q1_direct:57 design.math_reshape.rewrite[2].symbol.Normalization.meaning; q4_cluster:102 design.math_reshape.rewrite[3].symbol.Normalization.meaning; q4_direct:57 design.math_reshape.rewrite[2].symbol.Normalization.meaning

Final multiplier.

## t_b1e8f948772bac7e | math.meaning | 31 chars | 2 declarations

q1_direct:45 design.math_reshape.rewrite[1].symbol.P_lo.meaning; q4_direct:45 design.math_reshape.rewrite[1].symbol.P_lo.meaning

Independently rounded residual.

## t_bb0a00e740b6f995 | math.meaning | 26 chars | 5 declarations

q1_cluster:38 design.math_reshape.rewrite[0].symbol.Vblk.meaning; q1_global:38 design.math_reshape.rewrite[0].symbol.Vblk.meaning; q4_cluster:37 design.math_reshape.rewrite[0].symbol.Vblk.meaning; q4_compact:38 design.math_reshape.rewrite[0].symbol.Vblk.meaning; root:115 design.math_reshape.rewrite[0].symbol.Vblk.meaning

Logical paged value block.

## t_bd865aef5ad4ff41 | math.meaning | 18 chars | 7 declarations

q1_cluster:75 design.math_reshape.rewrite[2].symbol.Llocal.meaning; q1_direct:47 design.math_reshape.rewrite[1].symbol.Llocal.meaning; q1_global:75 design.math_reshape.rewrite[2].symbol.Llocal.meaning; q4_cluster:74 design.math_reshape.rewrite[2].symbol.Llocal.meaning; q4_compact:75 design.math_reshape.rewrite[2].symbol.Llocal.meaning; q4_direct:47 design.math_reshape.rewrite[1].symbol.Llocal.meaning; root:152 design.math_reshape.rewrite[2].symbol.Llocal.meaning

Split denominator.

## t_cd017efa2002c5f2 | math.meaning | 46 chars | 5 declarations

q1_cluster:56 design.math_reshape.rewrite[1].symbol.ExponentialSum.meaning; q1_global:56 design.math_reshape.rewrite[1].symbol.ExponentialSum.meaning; q4_cluster:55 design.math_reshape.rewrite[1].symbol.ExponentialSum.meaning; q4_compact:56 design.math_reshape.rewrite[1].symbol.ExponentialSum.meaning; root:133 design.math_reshape.rewrite[1].symbol.ExponentialSum.meaning

Unrounded per-thread probability partial sums.

## t_cee9156ccaed5cc1 | math.meaning | 17 chars | 2 declarations

q1_direct:26 design.math_reshape.rewrite[0].symbol.Bk.meaning; q4_direct:26 design.math_reshape.rewrite[0].symbol.Bk.meaning

Key-block length.

## t_d0c4e19dcb50e792 | math.meaning | 27 chars | 2 declarations

q1_direct:31 design.math_reshape.rewrite[0].symbol.Vblk.meaning; q4_direct:31 design.math_reshape.rewrite[0].symbol.Vblk.meaning

Padded logical value block.

## t_e04a4e4b2c3ead57 | math.meaning | 26 chars | 5 declarations

q1_cluster:51 design.math_reshape.rewrite[1].symbol.rho.meaning; q1_global:51 design.math_reshape.rewrite[1].symbol.rho.meaning; q4_cluster:50 design.math_reshape.rewrite[1].symbol.rho.meaning; q4_compact:51 design.math_reshape.rewrite[1].symbol.rho.meaning; root:128 design.math_reshape.rewrite[1].symbol.rho.meaning

Probability-operand scale.

## t_e079defeeff58ff4 | math.meaning | 46 chars | 7 declarations

q1_cluster:39 design.math_reshape.rewrite[0].symbol.ScoreBlock.meaning; q1_direct:32 design.math_reshape.rewrite[0].symbol.ScoreBlock.meaning; q1_global:39 design.math_reshape.rewrite[0].symbol.ScoreBlock.meaning; q4_cluster:38 design.math_reshape.rewrite[0].symbol.ScoreBlock.meaning; q4_compact:39 design.math_reshape.rewrite[0].symbol.ScoreBlock.meaning; q4_direct:32 design.math_reshape.rewrite[0].symbol.ScoreBlock.meaning; root:116 design.math_reshape.rewrite[0].symbol.ScoreBlock.meaning

Raw key-query product before the softmax mask.

## t_e60c4569f9da8022 | math.meaning | 27 chars | 2 declarations

q1_direct:44 design.math_reshape.rewrite[1].symbol.P_hi.meaning; q4_direct:44 design.math_reshape.rewrite[1].symbol.P_hi.meaning

Leading scaled probability.

## t_f016d8d441db5390 | math.meaning | 26 chars | 3 declarations

q1_global:94 design.math_reshape.rewrite[3].symbol.Mpartial.meaning; q4_compact:93 design.math_reshape.rewrite[3].symbol.Mpartial.meaning; root:171 design.math_reshape.rewrite[3].symbol.Mpartial.meaning

Materialized split maxima.

## t_f0da1bb97e99989e | math.meaning | 47 chars | 5 declarations

q1_cluster:52 design.math_reshape.rewrite[1].symbol.MinFinite.meaning; q1_global:52 design.math_reshape.rewrite[1].symbol.MinFinite.meaning; q4_cluster:51 design.math_reshape.rewrite[1].symbol.MinFinite.meaning; q4_compact:52 design.math_reshape.rewrite[1].symbol.MinFinite.meaning; root:129 design.math_reshape.rewrite[1].symbol.MinFinite.meaning

Finite initializer of the masked score maximum.

## t_f388eb30f12b6df5 | math.meaning | 25 chars | 7 declarations

q1_cluster:53 design.math_reshape.rewrite[1].symbol.m.meaning; q1_direct:43 design.math_reshape.rewrite[1].symbol.m.meaning; q1_global:53 design.math_reshape.rewrite[1].symbol.m.meaning; q4_cluster:52 design.math_reshape.rewrite[1].symbol.m.meaning; q4_compact:53 design.math_reshape.rewrite[1].symbol.m.meaning; q4_direct:43 design.math_reshape.rewrite[1].symbol.m.meaning; root:130 design.math_reshape.rewrite[1].symbol.m.meaning

Running base-two maximum.

## t_f92d486c5b251171 | math.meaning | 31 chars | 5 declarations

q1_cluster:79 design.math_reshape.rewrite[2].symbol.FinalSum.meaning; q1_global:79 design.math_reshape.rewrite[2].symbol.FinalSum.meaning; q4_cluster:78 design.math_reshape.rewrite[2].symbol.FinalSum.meaning; q4_compact:79 design.math_reshape.rewrite[2].symbol.FinalSum.meaning; root:156 design.math_reshape.rewrite[2].symbol.FinalSum.meaning

Warp-grouped final denominator.

## t_09e6cca0c0565fc0 | math.name | 50 chars | 5 declarations

q1_cluster:21 design.math_reshape.rewrite[0].name; q1_global:21 design.math_reshape.rewrite[0].name; q4_cluster:20 design.math_reshape.rewrite[0].name; q4_compact:21 design.math_reshape.rewrite[0].name; root:98 design.math_reshape.rewrite[0].name

Pack query heads and stripe the logical key blocks

## t_39cd8fef2d8e1919 | math.name | 43 chars | 3 declarations

q1_global:85 design.math_reshape.rewrite[3].name; q4_compact:85 design.math_reshape.rewrite[3].name; root:162 design.math_reshape.rewrite[3].name

Save split statistics and active numerators

## t_3b24f08dd6e3f5f4 | math.name | 50 chars | 5 declarations

q1_cluster:41 design.math_reshape.rewrite[1].name; q1_global:41 design.math_reshape.rewrite[1].name; q4_cluster:40 design.math_reshape.rewrite[1].name; q4_compact:41 design.math_reshape.rewrite[1].name; root:118 design.math_reshape.rewrite[1].name

Use two FP8 components for each scaled exponential

## t_3c4bdaad9eec0573 | math.name | 67 chars | 2 declarations

q1_direct:34 design.math_reshape.rewrite[1].name; q4_direct:34 design.math_reshape.rewrite[1].name

Split scaled probabilities into leading and residual FP8 components

## t_4094b0d181b7b2dd | math.name | 49 chars | 5 declarations

q1_cluster:59 design.math_reshape.rewrite[2].name; q1_global:59 design.math_reshape.rewrite[2].name; q4_cluster:58 design.math_reshape.rewrite[2].name; q4_compact:59 design.math_reshape.rewrite[2].name; root:136 design.math_reshape.rewrite[2].name

Carry two interleaved numerators and denominators

## t_53be5be152ad7169 | math.name | 39 chars | 2 declarations

q1_direct:17 design.math_reshape.rewrite[0].name; q4_direct:17 design.math_reshape.rewrite[0].name

Pack the query and retain one key block

## t_6cbae0a022d625f3 | math.name | 45 chars | 2 declarations

q1_direct:50 design.math_reshape.rewrite[2].name; q4_direct:50 design.math_reshape.rewrite[2].name

Normalize the two-component numerator locally

## t_cb4ecb4f0cc6d9aa | math.name | 56 chars | 2 declarations

q1_cluster:85 design.math_reshape.rewrite[3].name; q4_cluster:84 design.math_reshape.rewrite[3].name

Share normalization and add narrowed split contributions

## t_f7dd4baab2d19877 | math.name | 42 chars | 3 declarations

q1_global:100 design.math_reshape.rewrite[4].name; q4_compact:99 design.math_reshape.rewrite[4].name; root:177 design.math_reshape.rewrite[4].name

Normalize after the global split reduction

## t_314368b8c3409398 | math.step | 9 chars | 7 declarations

q1_cluster:84 design.math_reshape.rewrite[3].step; q1_direct:49 design.math_reshape.rewrite[2].step; q1_global:99 design.math_reshape.rewrite[4].step; q4_cluster:83 design.math_reshape.rewrite[3].step; q4_compact:98 design.math_reshape.rewrite[4].step; q4_direct:49 design.math_reshape.rewrite[2].step; root:176 design.math_reshape.rewrite[4].step

normalize

## t_58a05d1bccff45b4 | math.step | 8 chars | 7 declarations

q1_cluster:40 design.math_reshape.rewrite[1].step; q1_direct:33 design.math_reshape.rewrite[1].step; q1_global:40 design.math_reshape.rewrite[1].step; q4_cluster:39 design.math_reshape.rewrite[1].step; q4_compact:40 design.math_reshape.rewrite[1].step; q4_direct:33 design.math_reshape.rewrite[1].step; root:117 design.math_reshape.rewrite[1].step

exponent

## t_78d97953586155c0 | math.step | 11 chars | 3 declarations

q1_global:84 design.math_reshape.rewrite[3].step; q4_compact:84 design.math_reshape.rewrite[3].step; root:161 design.math_reshape.rewrite[3].step

materialize

## t_7ff2243d621ba0c2 | math.step | 6 chars | 5 declarations

q1_cluster:20 design.math_reshape.rewrite[0].step; q1_global:20 design.math_reshape.rewrite[0].step; q4_cluster:19 design.math_reshape.rewrite[0].step; q4_compact:20 design.math_reshape.rewrite[0].step; root:97 design.math_reshape.rewrite[0].step

stripe

## t_92b8bd785b35029f | math.step | 6 chars | 5 declarations

q1_cluster:58 design.math_reshape.rewrite[2].step; q1_global:58 design.math_reshape.rewrite[2].step; q4_cluster:57 design.math_reshape.rewrite[2].step; q4_compact:58 design.math_reshape.rewrite[2].step; root:135 design.math_reshape.rewrite[2].step

parity

## t_f89026891a8b3202 | math.step | 5 chars | 2 declarations

q1_direct:16 design.math_reshape.rewrite[0].step; q4_direct:16 design.math_reshape.rewrite[0].step

block

## t_2c56c8c4f4ddc79c | math.thesis | 203 chars | 2 declarations

q1_direct:13 design.math_reshape.thesis; q4_direct:13 design.math_reshape.thesis

Stream the sole key block without score/probability materialization. Pack heads and query tokens into matrix columns; two FP8 probability components feed the numerator while its denominator remains FP32.

## t_971b3f3f15c8e598 | math.thesis | 180 chars | 5 declarations

q1_cluster:17 design.math_reshape.thesis; q1_global:17 design.math_reshape.thesis; q4_cluster:16 design.math_reshape.thesis; q4_compact:17 design.math_reshape.thesis; root:94 design.math_reshape.thesis

Stripe KV blocks into shorter chains; keep two FP32 numerator parities and use leading/residual FP8 probabilities, trading an extra value product for no dense intermediate storage.

## t_c887d7433e049ced | metadata.describe | 88 chars | 7 declarations

q1_cluster:3 meta_data.describe; q1_direct:3 meta_data.describe; q1_global:3 meta_data.describe; q4_cluster:3 meta_data.describe; q4_compact:3 meta_data.describe; q4_direct:3 meta_data.describe; root:3 meta_data.describe

Compute right-aligned causal attention against a paged FP8 KV cache; overwrite BF16 out.

## t_76a3f9f1d14981d7 | pipeline.thesis | 135 chars | 1 declarations

q1_direct:88 design.tile_pipeline.thesis

Retain dedicated load and matrix-issue warps for the one-block path; the unused parity still participates in the common drain protocol.

## t_7dad1a7ede2dd831 | pipeline.thesis | 259 chars | 2 declarations

q1_cluster:145 design.tile_pipeline.thesis; q1_global:179 design.tile_pipeline.thesis

Separate K/V transfer and score/value issue roles; bounded issue lead and two numerator parities overlap softmax with correction. Buffer reuse follows operand completion, not issue. Two softmax warpgroups alternate blocks without role register redistribution.

## t_bff0800a47ba01e2 | pipeline.thesis | 297 chars | 3 declarations

q4_cluster:142 design.tile_pipeline.thesis; q4_compact:177 design.tile_pipeline.thesis; root:259 design.tile_pipeline.thesis

Separate K/V transfer and score/value issue roles; bounded issue lead and two numerator parities overlap softmax with correction. Buffer reuse follows operand completion, not issue. One softmax warpgroup processes every block; issuer/softmax/correction budgets are 64/168/256 registers per thread.

## t_f1b9ffbf9e63bfc3 | pipeline.thesis | 162 chars | 1 declarations

q4_direct:88 design.tile_pipeline.thesis

Pack two tail queries per CTA and use one softmax warpgroup; retain the initialized empty parity for the one-block drain and redistribute registers to Correction.

## t_011327771f5decc1 | pipeline.verb | 1091 chars | 1 declarations

q1_global:221 design.tile_pipeline.phase.partial.role.LoadK.info.verb.prepare.meaning

Use the fixed TableWidth stride to prefetch the first four page indices while the sequence-length fetch, descriptor prefetch and TMEM/pipeline setup progress. The initial lookup does not wait for seq_lens; unused entries may be -1 or name finite cache data, which the subsequent score mask excludes. Join CTA setup, including initialized statistics, before any stream work or empty-split handling; Q is then issued independently of later page lookups. The first page-table address is b*TableWidth+4*r. Let R=PackedColumns, T=512, Csm=232448. T is in TMEM columns and Csm is in bytes; these are the values returned by the original sm_100 helper calls in CUTLASS DSL 4.6.1. ScoreStageCount=min(4,floor((T-6*R)/(2*R))); Lead=min(2,ScoreStageCount-1). Allocate next_power_of_two((6+2*ScoreStageCount)*R) TMEM columns. A=180+2324*R+16*ScoreStageCount; pad=128-A%128; K0=floor((Csm-A-pad)/16384); KVStageCount=K0-int(16*K0>pad). Kslot(i,u)=(4*i+4*max(i-Lead,0)+u)%KVStageCount; Vslot(i,u)=(4*(i+min(i+Lead+1,N_r))+u)%KVStageCount. Each slice is 128-key by 128-d FP8; u orders key-half then d-half.

## t_083e248283ce8317 | pipeline.verb | 141 chars | 2 declarations

q1_cluster:217 design.tile_pipeline.phase.attention.role.LoadV.info.verb.store; q4_cluster:216 design.tile_pipeline.phase.attention.role.LoadV.info.verb.store

For N_r>0, wait for Ostage and acquire member 0 zero_done, execute an async-proxy fence, then BF16 TMA reduce-add to out; N_r=0 adds nothing.

## t_0a09debffc775d9c | pipeline.verb | 128 chars | 2 declarations

q1_direct:161 design.tile_pipeline.phase.attention.role.Value.info.verb.retire.meaning; q4_direct:163 design.tile_pipeline.phase.attention.role.Value.info.verb.retire.meaning

Free TMEM after Correction has captured and returned both Y parities; normalization and Ostage publication may still be pending.

## t_1292ae9d6dafb272 | pipeline.verb | 429 chars | 5 declarations

q1_cluster:236 design.tile_pipeline.phase.attention.role.Score.info.verb.score; q1_global:267 design.tile_pipeline.phase.partial.role.Score.info.verb.score; q4_cluster:236 design.tile_pipeline.phase.attention.role.Score.info.verb.score; q4_compact:269 design.tile_pipeline.phase.partial.role.Score.info.verb.score; root:350 design.tile_pipeline.phase.partial.role.Score.info.verb.score

Acquire a score slot; form two 128-key products reduced over both 128-d slices. Return K slices after operand reads; publish only the complete ScoreBlock. An independent KQ/VP issue-turn protocol follows the same Lead window as K/V loads, releasing its peer on final operand acquisition; short loops relay no-payload turns, and N_r=0 skips it. Wait for each operand slice separately; later transfers may overlap earlier products.

## t_17ed36e8ac6f1588 | pipeline.verb | 120 chars | 3 declarations

q1_global:262 design.tile_pipeline.phase.partial.role.Score.info.verb.store; q4_compact:264 design.tile_pipeline.phase.partial.role.Score.info.verb.store; root:345 design.tile_pipeline.phase.partial.role.Score.info.verb.store

Store Mpartial on FinalMaximum and Lpartial after reducing four FinalSum entries; visibility is required at kernel_done.

## t_1ce27ef9931bd32a | pipeline.verb | 75 chars | 2 declarations

q1_direct:182 design.tile_pipeline.phase.attention.role.Correction.info.verb.read; q4_direct:173 design.tile_pipeline.phase.attention.role.Correction.info.verb.read

Capture the sole maximum; drain empty sum parity 1 then the valid parity 0.

## t_1ea4daecd026b494 | pipeline.verb | 104 chars | 2 declarations

q1_direct:148 design.tile_pipeline.phase.attention.role.Score.info.verb.wait.meaning; q4_direct:149 design.tile_pipeline.phase.attention.role.Score.info.verb.wait.meaning

Join Correction's existing final-sum rendezvous; allow its final Y capture only after this peer arrival.

## t_1f0d50119909444b | pipeline.verb | 118 chars | 5 declarations

q1_cluster:216 design.tile_pipeline.phase.attention.role.LoadV.info.verb.load; q1_global:248 design.tile_pipeline.phase.partial.role.LoadV.info.verb.load; q4_cluster:215 design.tile_pipeline.phase.attention.role.LoadV.info.verb.load; q4_compact:249 design.tile_pipeline.phase.partial.role.LoadV.info.verb.load; root:330 design.tile_pipeline.phase.partial.role.LoadV.info.verb.load

Capture PageIndex(i); load four V slices under the K/V issue-turn and per-slice reuse protocols defined by LoadK/load.

## t_22e9e9351c83a355 | pipeline.verb | 137 chars | 2 declarations

q1_direct:118 design.tile_pipeline.phase.attention.role.LoadK.info.verb.load; q4_direct:118 design.tile_pipeline.phase.attention.role.LoadK.info.verb.load

Issue Q first, independently of the prefetched page table. K uses four slices; a reused slice waits for its previous matrix operand read.

## t_24a77bfd15116f86 | pipeline.verb | 353 chars | 2 declarations

q1_global:341 design.tile_pipeline.phase.partial.role.Correction.info.verb.publish; q4_compact:331 design.tile_pipeline.phase.partial.role.Correction.info.verb.publish

FinalMaximum releases the final m before the last parity correction. FinalSum publishes four warp sums (zero for N_r=0) and arrives at Score's rendezvous. Ostage stages captured FP32 Ylocal without normalization over the retired KV region; For N_r=0 the Ostage action remains in the pipe and completes as a no-op without forming or publishing a payload.

## t_265965ab6aa54024 | pipeline.verb | 93 chars | 5 declarations

q1_cluster:284 design.tile_pipeline.phase.attention.role.SoftmaxEven.info.verb.initialize; q1_global:315 design.tile_pipeline.phase.partial.role.SoftmaxEven.info.verb.initialize; q4_cluster:282 design.tile_pipeline.phase.attention.role.Softmax.info.verb.initialize; q4_compact:315 design.tile_pipeline.phase.partial.role.Softmax.info.verb.initialize; root:396 design.tile_pipeline.phase.partial.role.Softmax.info.verb.initialize

CTA setup initializes the shared maximum to -inf and FinalSum to zero, even for empty splits.

## t_2eee33663e37e2f5 | pipeline.verb | 152 chars | 2 declarations

q1_direct:183 design.tile_pipeline.phase.attention.role.Correction.info.verb.publish; q4_direct:174 design.tile_pipeline.phase.attention.role.Correction.info.verb.publish

FinalSum publishes four warp partials and arrives at Score's rendezvous; Ostage normalizes the valid Y, casts to BF16 and aliases the retired KV region.

## t_31809e74e46303ec | pipeline.verb | 105 chars | 5 declarations

q1_cluster:260 design.tile_pipeline.phase.attention.role.Value.info.verb.release.meaning; q1_global:291 design.tile_pipeline.phase.partial.role.Value.info.verb.release.meaning; q4_cluster:259 design.tile_pipeline.phase.attention.role.Value.info.verb.release.meaning; q4_compact:292 design.tile_pipeline.phase.partial.role.Value.info.verb.release.meaning; root:373 design.tile_pipeline.phase.partial.role.Value.info.verb.release.meaning

Guarantee V/P operand reads complete, without waiting for final Y capture or adding a collective barrier.

## t_32c2e6c37cd2456c | pipeline.verb | 111 chars | 2 declarations

q1_direct:172 design.tile_pipeline.phase.attention.role.SoftmaxEven.info.verb.publish_max; q4_direct:186 design.tile_pipeline.phase.attention.role.Softmax.info.verb.publish_max

Update the masked maximum and rendezvous with Correction so it captures m before that shared storage is reused.

## t_3443e86655f6076f | pipeline.verb | 120 chars | 5 declarations

q1_cluster:229 design.tile_pipeline.phase.attention.role.Score.info.verb.wait.meaning; q1_global:261 design.tile_pipeline.phase.partial.role.Score.info.verb.wait.meaning; q4_cluster:229 design.tile_pipeline.phase.attention.role.Score.info.verb.wait.meaning; q4_compact:263 design.tile_pipeline.phase.partial.role.Score.info.verb.wait.meaning; root:344 design.tile_pipeline.phase.partial.role.Score.info.verb.wait.meaning

Join Correction's existing FinalSum rendezvous; completion admits this role's sum read and Correction's final Y capture.

## t_35a6a78aee22c5e4 | pipeline.verb | 343 chars | 2 declarations

q1_cluster:311 design.tile_pipeline.phase.attention.role.Correction.info.verb.publish; q4_cluster:299 design.tile_pipeline.phase.attention.role.Correction.info.verb.publish

FinalMaximum releases the final m before the last parity correction. FinalSum publishes four warp sums (zero for N_r=0) and arrives at Score's rendezvous. Ostage normalizes and casts captured Y to BF16 over the retired KV region; For N_r=0 the Ostage action remains in the pipe and completes as a no-op without forming or publishing a payload.

## t_366bc06b01a49b70 | pipeline.verb | 162 chars | 2 declarations

q1_global:249 design.tile_pipeline.phase.partial.role.LoadV.info.verb.store; q4_compact:250 design.tile_pipeline.phase.partial.role.LoadV.info.verb.store

For N_r>0, store Ostage to the FP32 Opartial slice; for N_r=0 this action remains in the pipe and completes as a no-op without reading Ostage or writing Opartial.

## t_3696342d3667e348 | pipeline.verb | 186 chars | 5 declarations

q1_cluster:313 design.tile_pipeline.phase.attention.role.Correction.info.verb.update; q1_global:343 design.tile_pipeline.phase.partial.role.Correction.info.verb.update; q4_cluster:301 design.tile_pipeline.phase.attention.role.Correction.info.verb.update; q4_compact:333 design.tile_pipeline.phase.partial.role.Correction.info.verb.update; root:414 design.tile_pipeline.phase.partial.role.Correction.info.verb.update

Update the target Lbase from its parity predecessor and the just-captured exponential sum using step parity; this register work follows Y-slot return and can overlap the next VP product.

## t_379afcbcb34aae32 | pipeline.verb | 298 chars | 9 declarations

q1_cluster:284 design.tile_pipeline.phase.attention.role.SoftmaxEven.info.verb.read; q1_cluster:295 design.tile_pipeline.phase.attention.role.SoftmaxOdd.info.verb.read; q1_direct:170 design.tile_pipeline.phase.attention.role.SoftmaxEven.info.verb.read; q1_global:315 design.tile_pipeline.phase.partial.role.SoftmaxEven.info.verb.read; q1_global:326 design.tile_pipeline.phase.partial.role.SoftmaxOdd.info.verb.read; q4_cluster:283 design.tile_pipeline.phase.attention.role.Softmax.info.verb.read; q4_compact:316 design.tile_pipeline.phase.partial.role.Softmax.info.verb.read; q4_direct:184 design.tile_pipeline.phase.attention.role.Softmax.info.verb.read; root:397 design.tile_pipeline.phase.partial.role.Softmax.info.verb.read

Capture raw ScoreBlock in registers and immediately return its TMEM slot. Let last=ceil(seq_lens[b]/Bk)-1. Apply the contract mask in registers only for logical block last when q_len_per_req>1 or seq_lens[b]%Bk!=0, and also last-1 when 0<seq_lens[b]%Bk<q_len_per_req; interior blocks are mask-free.

## t_398b0eee4268b734 | pipeline.verb | 340 chars | 2 declarations

q1_cluster:306 design.tile_pipeline.phase.attention.role.Correction.info.verb.zero; q4_cluster:294 design.tile_pipeline.phase.attention.role.Correction.info.verb.zero

For N_r>0 initialize both Y slots, empty sum parity 1 and Lbase; fence before admitting softmax. For N_r=0 active is empty, so no zero action or wait is created; setup statistics remain in use. For output, member 0's first correction warp writes BF16 zero, fences at GPU scope and notifies all peers through DSMEM zero_done during the ramp.

## t_39c98e2e46e39c2e | pipeline.verb | 159 chars | 3 declarations

q1_global:265 design.tile_pipeline.phase.partial.role.Score.info.verb.launch.meaning; q4_compact:267 design.tile_pipeline.phase.partial.role.Score.info.verb.launch.meaning; root:348 design.tile_pipeline.phase.partial.role.Score.info.verb.launch.meaning

Issue this CTA's PDL release after statistic-store issue, contributing to grid-wide successor permission; statistic and Opartial stores may remain outstanding.

## t_3aae838c6eed3bde | pipeline.verb | 110 chars | 5 declarations

q1_cluster:295 design.tile_pipeline.phase.attention.role.SoftmaxOdd.info.verb.prepare; q1_global:326 design.tile_pipeline.phase.partial.role.SoftmaxOdd.info.verb.prepare; q4_cluster:282 design.tile_pipeline.phase.attention.role.Softmax.info.verb.prepare; q4_compact:315 design.tile_pipeline.phase.partial.role.Softmax.info.verb.prepare; root:396 design.tile_pipeline.phase.partial.role.Softmax.info.verb.prepare

For N_r=1, signal initialized ExponentialSum parity 1 before the block loop; otherwise this action is a no-op.

## t_3f466e0fbb1a8e28 | pipeline.verb | 957 chars | 2 declarations

q1_direct:122 design.tile_pipeline.phase.attention.role.LoadK.info.verb.prepare.meaning; q4_direct:122 design.tile_pipeline.phase.attention.role.LoadK.info.verb.prepare.meaning

Use the fixed TableWidth stride to prefetch the first four page indices while the sequence-length fetch, descriptor prefetch and TMEM/pipeline setup progress. The initial lookup does not wait for seq_lens; unused entries may be -1 or name finite cache data, which the subsequent score mask excludes. Join CTA setup, including initialized statistics, before any stream work or empty-split handling; Q is then issued independently of later page lookups. The first page-table address is b*TableWidth. Let R=PackedColumns, T=512, Csm=232448. T is in TMEM columns and Csm is in bytes; these are the values returned by the original sm_100 helper calls in CUTLASS DSL 4.6.1. ScoreStageCount=min(4,floor((T-6*R)/(2*R))); allocate next_power_of_two((6+2*ScoreStageCount)*R) TMEM columns. A=180+2324*R+16*ScoreStageCount; pad=128-A%128; K0=floor((Csm-A-pad)/16384); KVStageCount=K0-int(16*K0>pad). Each KV slice is 128-key by 128-d FP8; u orders key-half then d-half.

## t_3fbdb14a32fa5618 | pipeline.verb | 159 chars | 2 declarations

q1_direct:184 design.tile_pipeline.phase.attention.role.Correction.info.verb.capture; q4_direct:175 design.tile_pipeline.phase.attention.role.Correction.info.verb.capture

Capture empty parity 1 and valid Y parity 0 into TailNumerator[0:2] (older, newer), returning both TMEM slots; retain these registers through Ostage formation.

## t_43ceb822edcd9775 | pipeline.verb | 237 chars | 5 declarations

q1_cluster:181 design.tile_pipeline.phase.attention.role.LoadK.info.verb.advance.meaning; q1_global:214 design.tile_pipeline.phase.partial.role.LoadK.info.verb.advance.meaning; q4_cluster:178 design.tile_pipeline.phase.attention.role.LoadK.info.verb.advance.meaning; q4_compact:212 design.tile_pipeline.phase.partial.role.LoadK.info.verb.advance.meaning; root:294 design.tile_pipeline.phase.partial.role.LoadK.info.verb.advance.meaning

Complete PageIndex(i), prefetch i+1 before the K issue turn, and cycle four slots. For Lead=2 and i+4<N_r, V capture before same-slot prefetch overwrite is unproved: no established cross-role ordering proof protects this page-slot reuse.

## t_4882434a62a9fc9f | pipeline.verb | 145 chars | 5 declarations

q1_cluster:307 design.tile_pipeline.phase.attention.role.Correction.info.verb.read; q1_global:338 design.tile_pipeline.phase.partial.role.Correction.info.verb.read; q4_cluster:295 design.tile_pipeline.phase.attention.role.Correction.info.verb.read; q4_compact:328 design.tile_pipeline.phase.partial.role.Correction.info.verb.read; root:409 design.tile_pipeline.phase.partial.role.Correction.info.verb.read

Capture and release m or ExponentialSum, retaining two maxima and Lbase parities. For N_r=1 drain initialized sum parity 1 before valid parity 0.

## t_51d73ba60222b576 | pipeline.verb | 1198 chars | 1 declarations

root:301 design.tile_pipeline.phase.partial.role.LoadK.info.verb.prepare.meaning

Use the fixed TableWidth stride to prefetch the first four page indices while the sequence-length fetch, descriptor prefetch and TMEM/pipeline setup progress. The initial lookup does not wait for seq_lens; unused entries may be -1 or name finite cache data, which the subsequent score mask excludes. Join CTA setup, including initialized statistics, before any stream work or empty-split handling; Q is then issued independently of later page lookups. The first page-table address is b*TableWidth+4*r. Empty CTAs bypass stream-role register redistribution; their reduction warp retains the launch allocation. Let R=PackedColumns, T=512, Csm=232448. T is in TMEM columns and Csm is in bytes; these are the values returned by the original sm_100 helper calls in CUTLASS DSL 4.6.1. ScoreStageCount=min(4,floor((T-6*R)/(2*R))); Lead=min(2,ScoreStageCount-1). Allocate next_power_of_two((6+2*ScoreStageCount)*R) TMEM columns. A=180+2324*R+16*ScoreStageCount; pad=128-A%128; K0=floor((Csm-A-pad)/16384); KVStageCount=K0-int(16*K0>pad). Kslot(i,u)=(4*i+4*max(i-Lead,0)+u)%KVStageCount; Vslot(i,u)=(4*(i+min(i+Lead+1,N_r))+u)%KVStageCount. Each slice is 128-key by 128-d FP8; u orders key-half then d-half.

## t_566b43efd45d4126 | pipeline.verb | 99 chars | 5 declarations

q1_cluster:239 design.tile_pipeline.phase.attention.role.Score.info.verb.yield_turn.meaning; q1_global:270 design.tile_pipeline.phase.partial.role.Score.info.verb.yield_turn.meaning; q4_cluster:239 design.tile_pipeline.phase.attention.role.Score.info.verb.yield_turn.meaning; q4_compact:272 design.tile_pipeline.phase.partial.role.Score.info.verb.yield_turn.meaning; root:353 design.tile_pipeline.phase.partial.role.Score.info.verb.yield_turn.meaning

Release the peer MMA turn after final K operand acquisition; the score result may remain in flight.

## t_56e64a89090edb46 | pipeline.verb | 104 chars | 2 declarations

q1_direct:138 design.tile_pipeline.phase.attention.role.LoadV.info.verb.load; q4_direct:139 design.tile_pipeline.phase.attention.role.LoadV.info.verb.load

Load four V slices through the shared per-slice acquire protocol; use the same completed PageIndex as K.

## t_5fde249b997272d0 | pipeline.verb | 105 chars | 5 declarations

q1_cluster:270 design.tile_pipeline.phase.attention.role.Value.info.verb.yield_turn.meaning; q1_global:301 design.tile_pipeline.phase.partial.role.Value.info.verb.yield_turn.meaning; q4_cluster:269 design.tile_pipeline.phase.attention.role.Value.info.verb.yield_turn.meaning; q4_compact:302 design.tile_pipeline.phase.partial.role.Value.info.verb.yield_turn.meaning; root:383 design.tile_pipeline.phase.partial.role.Value.info.verb.yield_turn.meaning

Release the peer MMA turn after final V/P operand acquisition; the numerator result may remain in flight.

## t_64f19f96f28492bf | pipeline.verb | 71 chars | 7 declarations

q1_cluster:288 design.tile_pipeline.phase.attention.role.SoftmaxEven.info.verb.publish; q1_cluster:299 design.tile_pipeline.phase.attention.role.SoftmaxOdd.info.verb.publish; q1_global:319 design.tile_pipeline.phase.partial.role.SoftmaxEven.info.verb.publish; q1_global:330 design.tile_pipeline.phase.partial.role.SoftmaxOdd.info.verb.publish; q4_cluster:286 design.tile_pipeline.phase.attention.role.Softmax.info.verb.publish; q4_compact:319 design.tile_pipeline.phase.partial.role.Softmax.info.verb.publish; root:400 design.tile_pipeline.phase.partial.role.Softmax.info.verb.publish

Publish P_hi/P_lo together, then the independent ExponentialSum parity.

## t_68ae2951458ab017 | pipeline.verb | 95 chars | 5 declarations

q1_cluster:228 design.tile_pipeline.phase.attention.role.Score.info.verb.release.meaning; q1_global:260 design.tile_pipeline.phase.partial.role.Score.info.verb.release.meaning; q4_cluster:228 design.tile_pipeline.phase.attention.role.Score.info.verb.release.meaning; q4_compact:262 design.tile_pipeline.phase.partial.role.Score.info.verb.release.meaning; root:343 design.tile_pipeline.phase.partial.role.Score.info.verb.release.meaning

Guarantee completion of K operand reads for the Ostage alias; no additional collective barrier.

## t_6a50892326aab903 | pipeline.verb | 68 chars | 3 declarations

q1_global:367 design.tile_pipeline.phase.combine.role.Main.info.verb.produce; q4_compact:356 design.tile_pipeline.phase.combine.role.Main.info.verb.produce; root:438 design.tile_pipeline.phase.combine.role.Main.info.verb.produce

Apply upstream normalize and cast once to BF16 when overwriting out.

## t_6e9dfa4956b1e98c | pipeline.verb | 1352 chars | 1 declarations

q4_cluster:185 design.tile_pipeline.phase.attention.role.LoadK.info.verb.prepare.meaning

Use the fixed TableWidth stride to prefetch the first four page indices while the sequence-length fetch, descriptor prefetch and TMEM/pipeline setup progress. The initial lookup does not wait for seq_lens; unused entries may be -1 or name finite cache data, which the subsequent score mask excludes. Join CTA setup, including initialized statistics, before any stream work or empty-split handling; Q is then issued independently of later page lookups. The first page-table address is b*TableWidth+4*r. The startup cluster rendezvous publishes every member's DSMEM statistic/zero_done receiver initialization; it does not wait for output zeroing. Empty CTAs bypass stream-role register redistribution; their reduction warp retains the launch allocation. Let R=PackedColumns, T=512, Csm=232448. T is in TMEM columns and Csm is in bytes; these are the values returned by the original sm_100 helper calls in CUTLASS DSL 4.6.1. ScoreStageCount=min(4,floor((T-6*R)/(2*R))); Lead=min(2,ScoreStageCount-1). Allocate next_power_of_two((6+2*ScoreStageCount)*R) TMEM columns. A=180+2324*R+16*ScoreStageCount+(32*R+88); pad=128-A%128; K0=floor((Csm-A-pad)/16384); KVStageCount=K0-int(16*K0>pad). Kslot(i,u)=(4*i+4*max(i-Lead,0)+u)%KVStageCount; Vslot(i,u)=(4*(i+min(i+Lead+1,N_r))+u)%KVStageCount. Each slice is 128-key by 128-d FP8; u orders key-half then d-half.

## t_6f48a8d8446c4ed5 | pipeline.verb | 145 chars | 5 declarations

q1_cluster:263 design.tile_pipeline.phase.attention.role.Value.info.verb.retire.meaning; q1_global:294 design.tile_pipeline.phase.partial.role.Value.info.verb.retire.meaning; q4_cluster:262 design.tile_pipeline.phase.attention.role.Value.info.verb.retire.meaning; q4_compact:295 design.tile_pipeline.phase.partial.role.Value.info.verb.retire.meaning; root:376 design.tile_pipeline.phase.partial.role.Value.info.verb.retire.meaning

Free TMEM after Correction returns the final Y slots, or immediately for N_r=0; final normalization and Ostage publication need not finish first.

## t_7911025a4b6beadb | pipeline.verb | 1245 chars | 1 declarations

q1_cluster:188 design.tile_pipeline.phase.attention.role.LoadK.info.verb.prepare.meaning

Use the fixed TableWidth stride to prefetch the first four page indices while the sequence-length fetch, descriptor prefetch and TMEM/pipeline setup progress. The initial lookup does not wait for seq_lens; unused entries may be -1 or name finite cache data, which the subsequent score mask excludes. Join CTA setup, including initialized statistics, before any stream work or empty-split handling; Q is then issued independently of later page lookups. The first page-table address is b*TableWidth+4*r. The startup cluster rendezvous publishes every member's DSMEM statistic/zero_done receiver initialization; it does not wait for output zeroing. Let R=PackedColumns, T=512, Csm=232448. T is in TMEM columns and Csm is in bytes; these are the values returned by the original sm_100 helper calls in CUTLASS DSL 4.6.1. ScoreStageCount=min(4,floor((T-6*R)/(2*R))); Lead=min(2,ScoreStageCount-1). Allocate next_power_of_two((6+2*ScoreStageCount)*R) TMEM columns. A=180+2324*R+16*ScoreStageCount+(32*R+88); pad=128-A%128; K0=floor((Csm-A-pad)/16384); KVStageCount=K0-int(16*K0>pad). Kslot(i,u)=(4*i+4*max(i-Lead,0)+u)%KVStageCount; Vslot(i,u)=(4*(i+min(i+Lead+1,N_r))+u)%KVStageCount. Each slice is 128-key by 128-d FP8; u orders key-half then d-half.

## t_7e8887715d30d383 | pipeline.verb | 121 chars | 1 declarations

q4_direct:187 design.tile_pipeline.phase.attention.role.Softmax.info.verb.signal

For this single-block path, signal initialized ExponentialSum parity 1 before computing and publishing the real parity 0.

## t_8215187bff0459f7 | pipeline.verb | 129 chars | 2 declarations

q1_cluster:235 design.tile_pipeline.phase.attention.role.Score.info.verb.publish.meaning; q4_cluster:235 design.tile_pipeline.phase.attention.role.Score.info.verb.publish.meaning

Sum four local warp entries, rescale to ClusterMax, repeat the XOR tree for denominators, and publish Normalization over Maximum.

## t_84fc90e5833d1907 | pipeline.verb | 163 chars | 1 declarations

root:331 design.tile_pipeline.phase.partial.role.LoadV.info.verb.store

For N_r>0, store Ostage to the FP32 Opartial slice. For N_r=0, this action remains in the pipe and completes as a no-op without reading Ostage or writing Opartial.

## t_8ae042c014d2ee27 | pipeline.verb | 75 chars | 3 declarations

q1_global:363 design.tile_pipeline.phase.combine.role.Main.info.verb.wait; q4_compact:353 design.tile_pipeline.phase.combine.role.Main.info.verb.wait; root:434 design.tile_pipeline.phase.combine.role.Main.info.verb.wait

Execute the PDL predecessor-completion wait before any partial-result read.

## t_8b50d5b838335ee8 | pipeline.verb | 293 chars | 2 declarations

q1_global:363 design.tile_pipeline.phase.combine.role.Main.info.verb.compute; root:434 design.tile_pipeline.phase.combine.role.Main.info.verb.compute

Four warps each own 64 d elements, two per thread. split_weight reduces M/Z within each warp from at most two splits per thread and retains their two weights; Numerator(r) broadcasts the corresponding weight and accumulates every nominal slot in split order, including zero-weight empty reads.

## t_8c3518a7cf816cc2 | pipeline.verb | 94 chars | 2 declarations

q1_direct:149 design.tile_pipeline.phase.attention.role.Score.info.verb.publish.meaning; q4_direct:150 design.tile_pipeline.phase.attention.role.Score.info.verb.publish.meaning

Reduce four FinalSum warp entries and publish Normalization over the captured maximum storage.

## t_96057ea4f2b4f97a | pipeline.verb | 268 chars | 5 declarations

q1_cluster:309 design.tile_pipeline.phase.attention.role.Correction.info.verb.capture; q1_global:339 design.tile_pipeline.phase.partial.role.Correction.info.verb.capture; q4_cluster:297 design.tile_pipeline.phase.attention.role.Correction.info.verb.capture; q4_compact:329 design.tile_pipeline.phase.partial.role.Correction.info.verb.capture; root:410 design.tile_pipeline.phase.partial.role.Correction.info.verb.capture

Each tail action captures its Y parity into the corresponding older/newer entry of TailNumerator and returns that TMEM slot. For N_r=1, the sole action first drains initialized empty parity 1, then valid parity 0. Retain both register entries through Ostage formation.

## t_968085f1581dc6df | pipeline.verb | 95 chars | 5 declarations

q1_cluster:205 design.tile_pipeline.phase.attention.role.LoadK.info.verb.yield_turn.meaning; q1_global:237 design.tile_pipeline.phase.partial.role.LoadK.info.verb.yield_turn.meaning; q4_cluster:203 design.tile_pipeline.phase.attention.role.LoadK.info.verb.yield_turn.meaning; q4_compact:237 design.tile_pipeline.phase.partial.role.LoadK.info.verb.yield_turn.meaning; root:318 design.tile_pipeline.phase.partial.role.LoadK.info.verb.yield_turn.meaning

Release the peer load turn after acquiring the final K slice; K transfers may remain in flight.

## t_9b84e412251c5ce4 | pipeline.verb | 351 chars | 1 declarations

root:412 design.tile_pipeline.phase.partial.role.Correction.info.verb.publish

FinalMaximum releases the final m before the last parity correction. FinalSum publishes four warp sums (zero for N_r=0) and arrives at Score's rendezvous. Ostage stages captured FP32 Ylocal without normalization over the retired KV region. For N_r=0, publish:Ostage remains in the pipe and completes as a no-op without forming or publishing a payload.

## t_9c80ff1f08125181 | pipeline.verb | 137 chars | 2 declarations

q1_cluster:232 design.tile_pipeline.phase.attention.role.Score.info.verb.reduce.meaning; q4_cluster:232 design.tile_pipeline.phase.attention.role.Score.info.verb.reduce.meaning

After FinalMaximum, exchange maxima with peer r xor 2^j for j=0..log2(SplitCount)-1; each DSMEM transfer completes before the next round.

## t_a2ce5c776ea7549d | pipeline.verb | 176 chars | 3 declarations

q4_cluster:285 design.tile_pipeline.phase.attention.role.Softmax.info.verb.publish_max; q4_compact:318 design.tile_pipeline.phase.partial.role.Softmax.info.verb.publish_max; root:399 design.tile_pipeline.phase.partial.role.Softmax.info.verb.publish_max

Update/capture m; the next update waits for Correction to capture it. Acquire the P-ring slot before publishing this maximum to Correction, as in the source maximum rendezvous.

## t_a3e6c657a2a1a8a7 | pipeline.verb | 103 chars | 1 declarations

q1_direct:178 design.tile_pipeline.phase.attention.role.SoftmaxOdd.info.verb.signal

Signal the initialized empty ExponentialSum parity 1; this role computes no score block in this domain.

## t_a3e7f2a83aea87c0 | pipeline.verb | 160 chars | 5 declarations

q1_cluster:308 design.tile_pipeline.phase.attention.role.Correction.info.verb.rescale; q1_global:338 design.tile_pipeline.phase.partial.role.Correction.info.verb.rescale; q4_cluster:296 design.tile_pipeline.phase.attention.role.Correction.info.verb.rescale; q4_compact:328 design.tile_pipeline.phase.partial.role.Correction.info.verb.rescale; root:409 design.tile_pipeline.phase.partial.role.Correction.info.verb.rescale

With captured m(i+2), wait for Y(i), correct that numerator, fence its TMEM writes and return the slot for block i+2; the denominator update follows separately.

## t_a3fcc252e10e8495 | pipeline.verb | 179 chars | 5 declarations

q1_cluster:268 design.tile_pipeline.phase.attention.role.Value.info.verb.release_p; q1_global:299 design.tile_pipeline.phase.partial.role.Value.info.verb.release_p; q4_cluster:267 design.tile_pipeline.phase.attention.role.Value.info.verb.release_p; q4_compact:300 design.tile_pipeline.phase.partial.role.Value.info.verb.release_p; root:381 design.tile_pipeline.phase.partial.role.Value.info.verb.release_p

Signal completion of all leading/residual P operand reads for block i, returning its P-ring slot independently of the Y result-commit notification; no Correction wait is involved.

## t_aabbba0a83e94b8c | pipeline.verb | 196 chars | 2 declarations

q1_direct:150 design.tile_pipeline.phase.attention.role.Score.info.verb.score; q4_direct:151 design.tile_pipeline.phase.attention.role.Score.info.verb.score

Complete both key halves and both reduction slices into ScoreBlock; return K slices after their operand reads. Wait for each operand slice separately; later transfers may overlap earlier products.

## t_b3bc71bba2ff28d2 | pipeline.verb | 96 chars | 2 declarations

q1_direct:182 design.tile_pipeline.phase.attention.role.Correction.info.verb.zero; q4_direct:173 design.tile_pipeline.phase.attention.role.Correction.info.verb.zero

Zero both Y parities and ExponentialSum parity 1; fence initialization before admitting softmax.

## t_b9004354d10b5f3b | pipeline.verb | 241 chars | 2 declarations

q1_direct:162 design.tile_pipeline.phase.attention.role.Value.info.verb.accumulate; q4_direct:164 design.tile_pipeline.phase.attention.role.Value.info.verb.accumulate

P_pair comprises P_hi and P_lo; accumulate both per V reduction slice into Y parity 0; return V/P after both reads and signal completion of empty parity 1. Wait for each operand slice separately; later transfers may overlap earlier products.

## t_bbe874943b805358 | pipeline.verb | 89 chars | 2 declarations

q1_direct:147 design.tile_pipeline.phase.attention.role.Score.info.verb.yield_turn.meaning; q4_direct:148 design.tile_pipeline.phase.attention.role.Score.info.verb.yield_turn.meaning

Release Value after the final K operand is acquired, before score completion is required.

## t_c5573fc61dce69ee | pipeline.verb | 113 chars | 2 declarations

q1_direct:172 design.tile_pipeline.phase.attention.role.SoftmaxEven.info.verb.publish; q4_direct:186 design.tile_pipeline.phase.attention.role.Softmax.info.verb.publish

Publish both FP8 P components together; then publish 128 FP32 partial sums per column in ExponentialSum parity 0.

## t_c74b6f9b833aeee1 | pipeline.verb | 95 chars | 2 declarations

q1_direct:119 design.tile_pipeline.phase.attention.role.LoadK.info.verb.yield_turn.meaning; q4_direct:119 design.tile_pipeline.phase.attention.role.LoadK.info.verb.yield_turn.meaning

Release the V issue turn after the last K slice is acquired; transfers need not have completed.

## t_c84d039e26631130 | pipeline.verb | 95 chars | 5 declarations

q1_cluster:218 design.tile_pipeline.phase.attention.role.LoadV.info.verb.yield_turn.meaning; q1_global:250 design.tile_pipeline.phase.partial.role.LoadV.info.verb.yield_turn.meaning; q4_cluster:217 design.tile_pipeline.phase.attention.role.LoadV.info.verb.yield_turn.meaning; q4_compact:251 design.tile_pipeline.phase.partial.role.LoadV.info.verb.yield_turn.meaning; root:332 design.tile_pipeline.phase.partial.role.LoadV.info.verb.yield_turn.meaning

Release the peer load turn after acquiring the final V slice; V transfers may remain in flight.

## t_c9bb4c4d159140d3 | pipeline.verb | 1345 chars | 1 declarations

q4_compact:219 design.tile_pipeline.phase.partial.role.LoadK.info.verb.prepare.meaning

Use the fixed TableWidth stride to prefetch the first four page indices while the sequence-length fetch, descriptor prefetch and TMEM/pipeline setup progress. The initial lookup does not wait for seq_lens; unused entries may be -1 or name finite cache data, which the subsequent score mask excludes. Join CTA setup, including initialized statistics, before any stream work or empty-split handling; Q is then issued independently of later page lookups. The first page-table address is b*TableWidth+4*r. Each nominal CTA computes WorkSplit on device before early exit, using per-warp batch min/max reductions with the declared finite min initializer. Empty CTAs bypass stream-role register redistribution; their reduction warp retains the launch allocation. Let R=PackedColumns, T=512, Csm=232448. T is in TMEM columns and Csm is in bytes; these are the values returned by the original sm_100 helper calls in CUTLASS DSL 4.6.1. ScoreStageCount=min(4,floor((T-6*R)/(2*R))); Lead=min(2,ScoreStageCount-1). Allocate next_power_of_two((6+2*ScoreStageCount)*R) TMEM columns. A=180+2324*R+16*ScoreStageCount; pad=128-A%128; K0=floor((Csm-A-pad)/16384); KVStageCount=K0-int(16*K0>pad). Kslot(i,u)=(4*i+4*max(i-Lead,0)+u)%KVStageCount; Vslot(i,u)=(4*(i+min(i+Lead+1,N_r))+u)%KVStageCount. Each slice is 128-key by 128-d FP8; u orders key-half then d-half.

## t_cb7705d2a4404e91 | pipeline.verb | 186 chars | 1 declarations

q4_compact:353 design.tile_pipeline.phase.combine.role.Main.info.verb.compute

Each of 128 threads owns two d elements. split_weight scans all maxima to obtain M and initializes Z=0; Numerator(r) recomputes w[r] and updates Z/U for active splits in ascending order.

## t_d537b3e0b0d47309 | pipeline.verb | 242 chars | 1 declarations

q4_compact:354 design.tile_pipeline.phase.combine.role.Main.info.verb.load

PartialStatistic stages Mpartial/Lpartial together with asynchronous global-to-SMEM copies, then waits and establishes CTA-wide visibility. For Opartial(r), skip Mpartial[r]=-inf before any payload read; active payloads stream into registers.

## t_da610921ab34d284 | pipeline.verb | 76 chars | 2 declarations

q1_direct:173 design.tile_pipeline.phase.attention.role.SoftmaxEven.info.verb.initialize; q4_direct:188 design.tile_pipeline.phase.attention.role.Softmax.info.verb.initialize

During CTA setup initialize the shared maximum to -inf and FinalSum to zero.

## t_e1a4acb31b3c465a | pipeline.verb | 372 chars | 5 declarations

q1_cluster:265 design.tile_pipeline.phase.attention.role.Value.info.verb.accumulate; q1_global:296 design.tile_pipeline.phase.partial.role.Value.info.verb.accumulate; q4_cluster:264 design.tile_pipeline.phase.attention.role.Value.info.verb.accumulate; q4_compact:297 design.tile_pipeline.phase.partial.role.Value.info.verb.accumulate; root:378 design.tile_pipeline.phase.partial.role.Value.info.verb.accumulate

P_pair is P_hi/P_lo. Acquire Y after initialization/correction; accumulate leading then residual per V slice. Return each V slice after both component reads; release_p returns the P slot. N_r=1 also signals the initialized empty Y parity. Follow Score/score's KQ/VP issue-turn protocol. Wait for each operand slice separately; later transfers may overlap earlier products.

## t_e7425e141d7560bd | pipeline.verb | 361 chars | 5 declarations

q1_cluster:183 design.tile_pipeline.phase.attention.role.LoadK.info.verb.load; q1_global:216 design.tile_pipeline.phase.partial.role.LoadK.info.verb.load; q4_cluster:180 design.tile_pipeline.phase.attention.role.LoadK.info.verb.load; q4_compact:214 design.tile_pipeline.phase.partial.role.LoadK.info.verb.load; root:296 design.tile_pipeline.phase.partial.role.LoadK.info.verb.load

Load Q once before K. Acquire four K slices independently; reuse waits for their prior KQ/VP operand reads, and transfers complete per slice. The K/V issue-turn protocol admits K0..K(Lead),V0,K(Lead+1),V1,... and the V tail; each final-slice acquisition releases the peer before transfer completion. Short loops relay no-payload turns; N_r=0 skips the protocol.

## t_e8000047de03b6c5 | pipeline.verb | 250 chars | 2 declarations

q1_global:365 design.tile_pipeline.phase.combine.role.Main.info.verb.load; root:436 design.tile_pipeline.phase.combine.role.Main.info.verb.load

PartialStatistic stages Mpartial/Lpartial together with asynchronous global-to-SMEM copies, then waits and establishes CTA-wide visibility. For Opartial(r), read every nominal slot into registers, including the finite scratch value of an empty split.

## t_ef56c1145ae9c2cc | pipeline.verb | 248 chars | 1 declarations

root:408 design.tile_pipeline.phase.partial.role.Correction.info.verb.zero

For N_r>0, active=0 labels one initialization of both Y slots, empty sum parity 1 and Lbase; it is not a Y parity. Fence before admitting softmax. For N_r=0, active is empty: no zero action or wait for it exists, and setup statistics remain in use.

## t_f3787dcaa9b52150 | pipeline.verb | 238 chars | 4 declarations

q1_cluster:286 design.tile_pipeline.phase.attention.role.SoftmaxEven.info.verb.publish_max; q1_cluster:297 design.tile_pipeline.phase.attention.role.SoftmaxOdd.info.verb.publish_max; q1_global:317 design.tile_pipeline.phase.partial.role.SoftmaxEven.info.verb.publish_max; q1_global:328 design.tile_pipeline.phase.partial.role.SoftmaxOdd.info.verb.publish_max

Update/capture m under the maximum handshake; the next block waits for Correction to capture it. Q1 groups alternate under a mutex. Acquire the P-ring slot before publishing this maximum to Correction, as in the source maximum rendezvous.

## t_f555c62ae7f90de4 | pipeline.verb | 193 chars | 2 declarations

q1_global:337 design.tile_pipeline.phase.partial.role.Correction.info.verb.zero; q4_compact:327 design.tile_pipeline.phase.partial.role.Correction.info.verb.zero

For N_r>0 initialize both Y slots, empty sum parity 1 and Lbase; fence before admitting softmax. For N_r=0 active is empty, so no zero action or wait is created; setup statistics remain in use.

## t_f67b12aabe4a71eb | pipeline.verb | 83 chars | 2 declarations

q1_direct:138 design.tile_pipeline.phase.attention.role.LoadV.info.verb.store; q4_direct:139 design.tile_pipeline.phase.attention.role.LoadV.info.verb.store

Transfer BF16 Ostage to out and complete its asynchronous store before worker exit.

## t_04be6a4a867820b1 | realization.choice | 103 chars | 3 declarations

q1_global:373 realization.require[0].choice; q4_compact:362 realization.require[0].choice; root:444 realization.require[0].choice

A byte length divisible by four and at least 4*BatchCount*q_len_per_req*HeadCount*(SplitCount*(D+2)+1).

## t_695c381c8b53694e | realization.choice | 154 chars | 3 declarations

q4_cluster:326 realization.require[3].choice; q4_compact:376 realization.require[4].choice; root:458 realization.require[4].choice

For nonempty CTAs, setmaxnreg redistributes the 168-register launch allocation: issuers decrease to 64, Softmax stays at 168, Correction increases to 256.

## t_75414399db6e3fc1 | realization.choice | 122 chars | 1 declarations

q4_direct:200 realization.require[2].choice

setmaxnreg: issuers decrease from the 168-register launch budget to 64; Softmax stays at 168; Correction increases to 256.

## t_88bce51928dcf072 | realization.choice | 208 chars | 5 declarations

q1_cluster:329 realization.require[0].choice; q1_global:377 realization.require[1].choice; q4_cluster:316 realization.require[0].choice; q4_compact:366 realization.require[1].choice; root:448 realization.require[1].choice

tcgen05.mma with FP8 E4M3FN operands and FP32 TMEM accumulation; completion-aware operand release. Use the operand-consumer release separately from the accumulator-producer commit for P reuse and Y readiness.

## t_9f3d99ccf993d331 | realization.choice | 79 chars | 5 declarations

q1_cluster:339 realization.require[3].choice; q1_global:387 realization.require[4].choice; q4_cluster:329 realization.require[4].choice; q4_compact:379 realization.require[5].choice; root:461 realization.require[5].choice

Named-barrier handshakes for the independent load and MMA issue-turn protocols.

## t_d3491b4505dda2cc | realization.choice | 81 chars | 3 declarations

q1_global:384 realization.require[3].choice; q4_compact:373 realization.require[3].choice; root:455 realization.require[3].choice

PDL launch-dependents publication and griddepcontrol predecessor-completion wait.

## t_e319e3fd9f86092c | realization.choice | 98 chars | 2 declarations

q1_direct:195 realization.require[0].choice; q4_direct:194 realization.require[0].choice

tcgen05.mma with FP8 E4M3FN operands and FP32 TMEM accumulation; completion-aware operand release.

## t_e62d7b771e9498da | realization.choice | 98 chars | 7 declarations

q1_cluster:333 realization.require[1].choice; q1_direct:198 realization.require[1].choice; q1_global:381 realization.require[2].choice; q4_cluster:320 realization.require[1].choice; q4_compact:370 realization.require[2].choice; q4_direct:197 realization.require[1].choice; root:452 realization.require[2].choice

TMA query/KV/output transfers; cp.async page-table prefetch; per-slice producer/consumer barriers.

## t_fd2a7b0a22207efa | realization.choice | 175 chars | 2 declarations

q1_cluster:336 realization.require[2].choice; q4_cluster:323 realization.require[2].choice

Startup cluster barrier for receiver initialization; DSMEM exchanges with completion notifications; GPU-scope zeroing fence and async-proxy acquire before BF16 TMA reduce-add.

## t_19323a90ca13b01a | realization.reason | 186 chars | 3 declarations

q1_global:374 realization.require[0].reason; q4_compact:363 realization.require[0].reason; root:445 realization.require[0].reason

The entry reinterprets scratch as FP32 and slices the three partial tensors plus an additional statistics region; that final region is reserved even though the kernels do not consume it.

## t_20c1144d35f8c6c7 | realization.reason | 119 chars | 3 declarations

q4_cluster:327 realization.require[3].reason; q4_compact:377 realization.require[4].reason; root:459 realization.require[4].reason

Preserve the wide-column role allocation without asking the softmax warpgroup to increase beyond the launch allocation.

## t_2be8c34c571c3ce4 | realization.reason | 66 chars | 7 declarations

q1_cluster:334 realization.require[1].reason; q1_direct:199 realization.require[1].reason; q1_global:382 realization.require[2].reason; q4_cluster:321 realization.require[1].reason; q4_compact:371 realization.require[2].reason; q4_direct:198 realization.require[1].reason; root:453 realization.require[2].reason

Paged HND inputs are consumed directly through the shared KV ring.

## t_42df31af7dc75930 | realization.reason | 57 chars | 3 declarations

q1_global:385 realization.require[3].reason; q4_compact:374 realization.require[3].reason; root:456 realization.require[3].reason

Early successor launch must not make partial reads early.

## t_4f7a5460762b0e64 | realization.reason | 133 chars | 2 declarations

q1_cluster:337 realization.require[2].reason; q4_cluster:324 realization.require[2].reason

The cluster must share a common normalization and observe zero initialization before adding separately narrowed contributions to out.

## t_c97baa88114bd476 | realization.reason | 105 chars | 1 declarations

q4_direct:201 realization.require[2].reason

The issuer reduction funds Correction; Softmax must not request an increase beyond its launch allocation.

## t_dcea6b4ddf720ef2 | realization.reason | 64 chars | 5 declarations

q1_cluster:340 realization.require[3].reason; q1_global:388 realization.require[4].reason; q4_cluster:330 realization.require[4].reason; q4_compact:380 realization.require[5].reason; root:462 realization.require[5].reason

Issue turns release peers before transfer or product completion.

## t_e0af795a5949dff9 | realization.reason | 82 chars | 7 declarations

q1_cluster:331 realization.require[0].reason; q1_direct:196 realization.require[0].reason; q1_global:379 realization.require[1].reason; q4_cluster:318 realization.require[0].reason; q4_compact:368 realization.require[1].reason; q4_direct:195 realization.require[0].reason; root:450 realization.require[1].reason

The pipeline exchanges TMEM accumulators and reuses KV slices after operand reads.

## t_39be4744384b79d9 | schedule.describe | 52 chars | 3 declarations

q1_global:139 design.work_schedule.phase.partial.describe; q4_compact:138 design.work_schedule.phase.partial.describe; root:216 design.work_schedule.phase.partial.describe

Compute split-local online statistics and numerator.

## t_41ff9d173d204695 | schedule.describe | 77 chars | 3 declarations

q1_global:166 design.work_schedule.phase.combine.describe; q4_compact:164 design.work_schedule.phase.combine.describe; root:244 design.work_schedule.phase.combine.describe

Combine all nominal slots for one query head and overwrite its output vector.

## t_781c06d1c4f7017c | schedule.describe | 65 chars | 4 declarations

q1_cluster:121 design.work_schedule.phase.attention.describe; q1_direct:74 design.work_schedule.phase.attention.describe; q4_cluster:120 design.work_schedule.phase.attention.describe; q4_direct:74 design.work_schedule.phase.attention.describe

Compute and normalize one query block within its assigned worker.

## t_021853ef880b9de2 | schedule.strategy | 72 chars | 1 declarations

root:240 design.work_schedule.phase.partial.schedule.strategy

Enumerate batch outside query_tail outside split; split changes fastest.

## t_0743e7f302231b85 | schedule.strategy | 677 chars | 1 declarations

root:218 design.work_schedule.phase.partial.decomposition.strategy

Bk=256; QueryTile=2; PackedColumns=HeadCount*QueryTile; C=ceil(max_seq_len/Bk). Let B=BatchCount, L=max_seq_len, M=2*B, skew=(128*PageCount<B*L), f2(n)=largest power of two <=max(1,n), base=f2(floor(148/M)). SplitCount=min(C,max(4,base)) if L<24576; otherwise min(C,floor(148/M)) if not skew and M in {10,12,14,16,24,26,28}; otherwise min(C,floor(296/M)) if (not skew and M in {30,32}) or (skew and M in {16,18}); otherwise min(C,max(base,32 if skew, otherwise 8)). WorkSplit(b)=SplitCount. 148 and 296 are source scheduling budgets. Each CTA traverses blocks r+i*WorkSplit(b) through four HND page indices; empty splits leave Opartial untouched and publish neutral statistics.

## t_0a0b7e611783fec3 | schedule.strategy | 168 chars | 1 declarations

q4_direct:76 design.work_schedule.phase.attention.decomposition.strategy

Bk=256; QueryTile=2; PackedColumns=HeadCount*QueryTile. The dispatch domain has exactly one KV block per request; load its four logical pages directly from HND storage.

## t_13cf44c47a12bd49 | schedule.strategy | 738 chars | 1 declarations

q4_compact:140 design.work_schedule.phase.partial.decomposition.strategy

Bk=256; QueryTile=2; PackedColumns=HeadCount*QueryTile; let B=BatchCount, L=max_seq_len. SplitCount=8 if 128*PageCount>=B*L, otherwise 32. near=(4*min(131072,min_b seq_lens[b])>=3*L); the finite initializer is retained. If SplitCount=32: WorkSplit(b)=max(4,ceil(seq_lens[b]*SplitCount/L)). Otherwise, if near: WorkSplit(b)=7 for B=10 and 6 for B=11; otherwise WorkSplit(b)=ceil(seq_lens[b]*8/L) for B=10 and 8 for B=11. Nominal split r owns blocks r+i*WorkSplit(b); r>=WorkSplit(b) or r>=ceil(seq_lens[b]/Bk) is empty. Every nominal CTA is launched; empty CTAs publish Mpartial=-inf,Lpartial=0 without Opartial. Each active block gathers four HND page indices. The host's B=12/S=8 compact clause is unreachable under the default selector.

## t_23ad979aaa431c8e | schedule.strategy | 759 chars | 1 declarations

q1_global:141 design.work_schedule.phase.partial.decomposition.strategy

Bk=256; QueryTile=1; PackedColumns=HeadCount*QueryTile; C=ceil(max_seq_len/Bk). Let B=BatchCount, L=max_seq_len, E=PageSize*PageCount, A=floor(E/B), skew=(2*E<B*L), U=(B>=5 and L>=32768 and A>=(65536 if B=8, otherwise 32768) and not skew). f2(n)=largest power of two <=max(1,n). In first-matching order: SplitCount=min(C,max(16,f2(floor(64/B)))) if B<=4 and L>=24576; otherwise min(C,max(8,f2(floor(148/B)))) if U; otherwise 32 if skew and 5<=B<=8 and L>=65536; otherwise 16 if 5*B*L>=9*E, otherwise 8 (the remaining domain has 16<=B<=17). WorkSplit(b)=SplitCount. Each CTA traverses block r+i*WorkSplit(b) via four HND page indices. Empty splits publish neutral statistics and leave Opartial untouched. The constants 64 and 148 are source scheduling budgets.

## t_2d78a42d262c3f24 | schedule.strategy | 70 chars | 1 declarations

root:255 design.work_schedule.phase.combine.schedule.strategy

Enumerate batch outside query_tail outside head; head changes fastest.

## t_3926d433b2d38741 | schedule.strategy | 744 chars | 1 declarations

q1_cluster:123 design.work_schedule.phase.attention.decomposition.strategy

Bk=256; QueryTile=1; PackedColumns=HeadCount*QueryTile; C=ceil(max_seq_len/Bk). Let B=BatchCount, L=max_seq_len, E=PageSize*PageCount, A=floor(E/B), skew=(2*E<B*L). f2(n)=largest power of two <=max(1,n). X16=(B<=4) or (skew and L>=32768) or (not skew and 5<=B<=7 and (12288<=L<=20480 or (L>=24576 and A>=24576))). X8=(B<=8) or skew or (L>=24576 and 9<=B<=15). SplitCount=f2(min(C,16 if X16, otherwise 8 if X8, otherwise 4)); WorkSplit(b)=SplitCount. Each cluster member r independently traverses block r+i*WorkSplit(b) through four HND page indices. Iteration shapes collect all members without lockstep. Empty members still join statistics exchange. Member 0 zeroes out during the pipeline ramp; all reduce-adds wait for its visibility signal.

## t_443875b3cca60dd9 | schedule.strategy | 168 chars | 1 declarations

q1_direct:76 design.work_schedule.phase.attention.decomposition.strategy

Bk=256; QueryTile=1; PackedColumns=HeadCount*QueryTile. The dispatch domain has exactly one KV block per request; load its four logical pages directly from HND storage.

## t_647f2dd58e573aca | schedule.strategy | 619 chars | 1 declarations

q4_cluster:122 design.work_schedule.phase.attention.decomposition.strategy

Bk=256; QueryTile=2; PackedColumns=HeadCount*QueryTile; C=ceil(max_seq_len/Bk). Let B=BatchCount, L=max_seq_len, M=2*B; f2(n)=largest power of two <=max(1,n). SplitCount=f2(min(C,min(8,max(2,f2(ceil(96/M)))))) if L<=2048; otherwise f2(min(C,min(16,max(4,f2(ceil(96/M)))))) if B<=4, otherwise 4. WorkSplit(b)=SplitCount; 96 is a source scheduling budget. Member r independently traverses blocks r+i*WorkSplit(b) via four HND page indices. Iteration shapes collect all members without lockstep; empty members still join statistics exchange. Member 0 zeroes out during the ramp; reduce-adds wait for its visibility signal.

## t_922c6259583cbd36 | schedule.strategy | 121 chars | 2 declarations

q1_cluster:140 design.work_schedule.phase.attention.mapping.strategy; q4_cluster:137 design.work_schedule.phase.attention.mapping.strategy

Cluster member r, 0<=r<SplitCount, owns exactly the striped split r; query operands are separately loaded by each member.

## t_119fa15b941651b5 | schedule.thesis | 137 chars | 1 declarations

q1_global:128 design.work_schedule.thesis

Run independent striped split CTAs, then one combine CTA per query head; retain FP32 partials and permit early combine startup under PDL.

## t_2af9a8670e197997 | schedule.thesis | 137 chars | 1 declarations

q4_cluster:116 design.work_schedule.thesis

Two query tiles per request each receive a striped CTA cluster; exchange maxima/sums and accumulate BF16 contributions directly into out.

## t_64210b3cf5d53f68 | schedule.thesis | 91 chars | 2 declarations

q1_direct:70 design.work_schedule.thesis; q4_direct:70 design.work_schedule.thesis

Keep the one-block KV reduction inside one CTA; no split workspace or collective is needed.

## t_87ce50aa8e425927 | schedule.thesis | 170 chars | 1 declarations

q4_compact:127 design.work_schedule.thesis

Preserve nominal launch and scratch dimensions while varying active striped splits; combine skips empty payloads and recomputes statistics serially in each output thread.

## t_c7af3ec5e988123b | schedule.thesis | 161 chars | 1 declarations

q1_cluster:117 design.work_schedule.thesis

Assign one query tile to one cluster; members stripe KV blocks and exchange statistics before BF16 reduce-add, avoiding a second kernel and global FP32 partials.

## t_eb490df8a2935e06 | schedule.thesis | 120 chars | 1 declarations

root:205 design.work_schedule.thesis

Expose two query tiles per request and independent KV splits; combine FP32 partials in one CTA per query head under PDL.
