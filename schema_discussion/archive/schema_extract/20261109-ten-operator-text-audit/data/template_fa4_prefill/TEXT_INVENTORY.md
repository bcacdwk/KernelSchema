# template_fa4_prefill: full open-text inventory

These are exact-text groups, not proof of semantic equivalence across contexts.

## t_4320672ba5771a94 | algorithm_contract.math | 743 chars | 1 declarations

root:42 contract.algorithm_requirement.compute[0].math

python_int is a Python integer; fp64 scalars are Python floats; python_none is the singleton Python None ABI type, not a tensor. Tensors are contiguous in the stated axis order on one CUDA device. batch_size=BatchCount. cum_seq_lens_q[0]=cum_seq_lens_kv[0]=0; cum_seq_lens_q[BatchCount]=TotalQ. Qlen(b)=cum_seq_lens_q[b+1]-cum_seq_lens_q[b]; L(b)=seq_lens[b]; 0<Qlen(b)<=L(b). max_q_len=max_b Qlen(b); max_kv_len=max_b L(b); Pages=sum_b ceil(L(b)/Page). cum_seq_lens_kv[b+1]-cum_seq_lens_kv[b]=ceil(L(b)/Page)<=TableColumns. For 0<=j<L(b), Kref[b,j,d]=kv_cache[block_tables[b,floor(j/Page)],0,0,j%Page,d], with Vref using kind=1. The complete standalone callable is timed. Only out is an observable input mutation; workspace_buffer is scratch.

## t_b34a8da0898b51ff | algorithm_contract.math | 430 chars | 1 declarations

root:57 contract.algorithm_requirement.compute[1].math

For each b, 0<=q<Qlen(b), 0<=h<Hq and 0<=d<D, let a=cum_seq_lens_q[b]+q. J(b,q)={j in integers | 0<=j<L(b) and j<=L(b)-Qlen(b)+q}. score[j]=bmm1_scale*sum_{d=0}^{D-1} query[a,h,d]*Kref[b,j,d] for j in J(b,q). p[j]=exp(score[j]-max_k score[k])/sum_{k in J(b,q)} exp(score[k]-max_u score[u]), with all maxima over J(b,q). out_after[a,h,d]=bf16(bmm2_scale*sum_{j in J(b,q)} p[j]*Vref[b,j,d]); output is the same tensor object as out.

## t_0e23b91f87366271 | algorithm_contract.meaning | 40 chars | 1 declarations

root:38 contract.algorithm_requirement.output.output.meaning

Returned alias of the updated input out.

## t_161d515cd4827d79 | algorithm_contract.meaning | 23 chars | 1 declarations

root:29 contract.algorithm_requirement.input.bmm2_scale.meaning

Composite output scale.

## t_302dec2005cdfd83 | algorithm_contract.meaning | 35 chars | 1 declarations

root:52 contract.algorithm_requirement.compute[0].symbol.Qlen.meaning

Query length function of request b.

## t_33d00ca80e470ab0 | algorithm_contract.meaning | 48 chars | 1 declarations

root:23 contract.algorithm_requirement.input.workspace_buffer.meaning

Caller scratch zeroed before first isolated use.

## t_3932d15b415420a2 | algorithm_contract.meaning | 24 chars | 1 declarations

root:31 contract.algorithm_requirement.input.cum_seq_lens_q.meaning

Query prefix boundaries.

## t_413ef00d1623b854 | algorithm_contract.meaning | 29 chars | 1 declarations

root:26 contract.algorithm_requirement.input.max_q_len.meaning

Maximum request query length.

## t_435ef2cea1740fc3 | algorithm_contract.meaning | 14 chars | 1 declarations

root:17 contract.algorithm_requirement.constant.Hkv.meaning

KV head count.

## t_5f40e2f2a5ff3a66 | algorithm_contract.meaning | 37 chars | 1 declarations

root:53 contract.algorithm_requirement.compute[0].symbol.L.meaning

Used KV length function of request b.

## t_61344414b1f0da4e | algorithm_contract.meaning | 54 chars | 1 declarations

root:36 contract.algorithm_requirement.input.out.meaning

Caller destination whose initial contents are ignored.

## t_771571ce59cfbce7 | algorithm_contract.meaning | 44 chars | 1 declarations

root:54 contract.algorithm_requirement.compute[0].symbol.Kref.meaning

Logical keys on each request's valid domain.

## t_7a81fe920558b776 | algorithm_contract.meaning | 38 chars | 1 declarations

root:24 contract.algorithm_requirement.input.block_tables.meaning

Logical page to physical page mapping.

## t_92dd8f4bdba13f98 | algorithm_contract.meaning | 32 chars | 1 declarations

root:34 contract.algorithm_requirement.input.sinks.meaning

Required disabled sink argument.

## t_cd0f1797bd10b750 | algorithm_contract.meaning | 31 chars | 1 declarations

root:39 contract.algorithm_requirement.output.out_after.meaning

In-place contents of input out.

## t_cde865389d0fdbdf | algorithm_contract.meaning | 29 chars | 1 declarations

root:21 contract.algorithm_requirement.input.query.meaning

Concatenated request queries.

## t_d004f487e4faf6a4 | algorithm_contract.meaning | 30 chars | 1 declarations

root:18 contract.algorithm_requirement.constant.D.meaning

Query and value channel count.

## t_d0c960a5c6be1f7f | algorithm_contract.meaning | 23 chars | 1 declarations

root:27 contract.algorithm_requirement.input.max_kv_len.meaning

Maximum used KV length.

## t_d25f17179de788c9 | algorithm_contract.meaning | 28 chars | 1 declarations

root:19 contract.algorithm_requirement.constant.Page.meaning

Tokens per physical KV page.

## t_d38b1c38833b286c | algorithm_contract.meaning | 37 chars | 1 declarations

root:33 contract.algorithm_requirement.input.window_left.meaning

Unrestricted causal history sentinel.

## t_d4e2481ddeebe933 | algorithm_contract.meaning | 52 chars | 1 declarations

root:32 contract.algorithm_requirement.input.cum_seq_lens_kv.meaning

Prefix counts of allocated pages rather than tokens.

## t_d6e441742dec3fca | algorithm_contract.meaning | 49 chars | 1 declarations

root:22 contract.algorithm_requirement.input.kv_cache.meaning

HND cache with keys then values on the kind axis.

## t_e361d613a56cdba6 | algorithm_contract.meaning | 46 chars | 1 declarations

root:64 contract.algorithm_requirement.compute[1].symbol.J.meaning

Visible key set function of request and query.

## t_e42d2129a8323318 | algorithm_contract.meaning | 16 chars | 1 declarations

root:25 contract.algorithm_requirement.input.seq_lens.meaning

Used KV lengths.

## t_e66af7be4d3dbae2 | algorithm_contract.meaning | 47 chars | 1 declarations

root:35 contract.algorithm_requirement.input.o_sf_scale.meaning

Required disabled output quantization argument.

## t_e7bba2709488483e | algorithm_contract.meaning | 18 chars | 1 declarations

root:30 contract.algorithm_requirement.input.batch_size.meaning

ABI request count.

## t_f441d1be8f8a8404 | algorithm_contract.meaning | 49 chars | 1 declarations

root:28 contract.algorithm_requirement.input.bmm1_scale.meaning

Composite QK scale including quantization scales.

## t_f7685d5364954804 | algorithm_contract.meaning | 46 chars | 1 declarations

root:55 contract.algorithm_requirement.compute[0].symbol.Vref.meaning

Logical values on each request's valid domain.

## t_fafc62027311f974 | algorithm_contract.meaning | 17 chars | 1 declarations

root:16 contract.algorithm_requirement.constant.Hq.meaning

Query head count.

## t_5996dd1e60b2e83a | algorithm_contract.range | 44 chars | 1 declarations

root:36 contract.algorithm_requirement.input.out.range

Any BF16 bit pattern, including inf and NaN.

## t_c8859c9506de4d0d | algorithm_contract.range | 49 chars | 1 declarations

root:24 contract.algorithm_requirement.input.block_tables.range

Used entries in [0,Pages); unused entries are -1.

## t_c8e2f190af93b55c | algorithm_contract.range | 9 chars | 1 declarations

root:25 contract.algorithm_requirement.input.seq_lens.range

Positive.

## t_fc93f7b072d442a0 | algorithm_contract.range | 12 chars | 2 declarations

root:34 contract.algorithm_requirement.input.sinks.range; root:35 contract.algorithm_requirement.input.o_sf_scale.range

Python None.

## t_101c8a69d5709542 | contract.compute_name | 32 chars | 1 declarations

root:41 contract.algorithm_requirement.compute[0].name

Interpret the callable and pages

## t_a484e2476965f9d7 | contract.compute_name | 30 chars | 1 declarations

root:56 contract.algorithm_requirement.compute[1].name

Right-aligned causal attention

## t_28c8cfebc4cd144c | contract.tolerance | 20 chars | 2 declarations

root:67 contract.algorithm_requirement.golden_reference.tolerance.output; root:67 contract.algorithm_requirement.golden_reference.tolerance.out_after

atol=0.06, rtol=0.04

## t_115926a57ee09c4d | dispatch.md | 92 chars | 1 declarations

dispatch.md:30 dispatch.md

| [splitkv_dual_load](splitkv_dual_load.yaml) | query 工作不足时；增加 KV 分段、FP32 部分结果和独立合并 kernel |

## t_209ca87aa0ebe05e | dispatch.md | 433 chars | 1 declarations

dispatch.md:5 dispatch.md

```mermaid flowchart TD root --> rect_joint_load root --> packed_persistent packed_persistent --> single_persistent_dual_load single_persistent_dual_load --> single_persistent_joint_load packed_persistent --> small_persistent_lpt small_persistent_lpt --> small_persistent_shared_rescale root --> small_static_lpt small_static_lpt --> small_static_shared_rescale root --> splitkv_dual_load splitkv_dual_load --> splitkv_joint_load ```

## t_292cae790405e474 | dispatch.md | 104 chars | 1 declarations

dispatch.md:29 dispatch.md

| [small_static_shared_rescale](small_static_shared_rescale.yaml) | 特定小 batch；static 调度下改为共享已计算的 alpha |

## t_3f7cb032ce65ad8b | dispatch.md | 29 chars | 1 declarations

dispatch.md:19 dispatch.md

| 节点 | Workload 与相对父节点的设计变化 |

## t_4572131c25bfdc59 | dispatch.md | 104 chars | 1 declarations

dispatch.md:23 dispatch.md

| [packed_persistent](packed_persistent.yaml) | 中等 batch；折叠 query 工作并用 persistent worker 接续多份工作，保留请求次序 |

## t_565310feb6edd641 | dispatch.md | 103 chars | 1 declarations

dispatch.md:25 dispatch.md

| [single_persistent_joint_load](single_persistent_joint_load.yaml) | 单请求；persistent 调度下由同一加载角色处理 K/V |

## t_66b1d3bb843a4688 | dispatch.md | 97 chars | 1 declarations

dispatch.md:28 dispatch.md

| [small_static_lpt](small_static_lpt.yaml) | 小 batch；按 KV 长度降序排列并折叠 query 工作，每个 worker 只承担一份工作 |

## t_6e8906c949e88b5d | dispatch.md | 20 chars | 1 declarations

dispatch.md:1 dispatch.md

# FA4 prefill schema

## t_7185c4611eb73d1f | dispatch.md | 147 chars | 1 declarations

dispatch.md:33 dispatch.md

树与各节点的 `meta_data.dispatch.parent` 对应，表示设计派生关系。root 和其他节点均按各自的 `when` 在共同契约域内参与选择；精确条件与修订号分别见 YAML 的 `meta_data.dispatch.when`、`meta_data.version`。

## t_760c0650d972f570 | dispatch.md | 81 chars | 1 declarations

dispatch.md:31 dispatch.md

| [splitkv_joint_load](splitkv_joint_load.yaml) | 单请求；保留 KV 分段与合并，由同一加载角色处理 K/V |

## t_854c94f126d73f55 | dispatch.md | 121 chars | 1 declarations

dispatch.md:27 dispatch.md

| [small_persistent_shared_rescale](small_persistent_shared_rescale.yaml) | 特定小 batch；softmax 向 correction 传递已计算的 alpha |

## t_863f83078d0f22b1 | dispatch.md | 92 chars | 1 declarations

dispatch.md:35 dispatch.md

LPT 指按 KV 长度降序排列请求；persistent 按固定步长接续多份工作，static 每个 worker 只承担一份工作。split 数、长度阈值等参数变化留在所属节点内。

## t_9e57ca613b5f2609 | dispatch.md | 108 chars | 1 declarations

dispatch.md:3 dispatch.md

本任务以矩形 static 调度、K/V 独立加载的设计作为 root。它保留通用的请求与 query 网格，其他节点针对工作量、请求组织或加载方式派生。完整 Contract 只在 `root.yaml` 中定义。

## t_c9b880c990d844b5 | dispatch.md | 96 chars | 1 declarations

dispatch.md:24 dispatch.md

| [single_persistent_dual_load](single_persistent_dual_load.yaml) | 单请求；使用单请求的 persistent 工作映射 |

## t_cd4085da26f0fb4d | dispatch.md | 68 chars | 1 declarations

dispatch.md:22 dispatch.md

| [rect_joint_load](rect_joint_load.yaml) | 单请求；矩形调度下由同一加载角色处理 K/V |

## t_dfcaa80c0ed7db86 | dispatch.md | 67 chars | 1 declarations

dispatch.md:21 dispatch.md

| [root](root.yaml) | 通用矩形 static 双加载设计，承接大 batch 及部分单请求 workload |

## t_ecf0996549f6bab5 | dispatch.md | 90 chars | 1 declarations

dispatch.md:26 dispatch.md

| [small_persistent_lpt](small_persistent_lpt.yaml) | 小 batch；persistent 遍历按 KV 长度降序排列请求 |

## t_4209f7e147edb1ea | dispatch.when | 629 chars | 1 declarations

small_persistent_shared_rescale:9 meta_data.dispatch.when

Use packaged defaults with no FA_HD256 environment overrides, under the root contract. Let B=BatchCount, T=TotalQ, Q=max_q_len, K=64*ceil(max_kv_len/64), SM=runtime multiprocessor count. C=ceil(T/16)*B; G=max(1,floor(SM/2)); Split=(2*C<=G and ceil(K/128)>=16 and 2*T*16*256*4<=24000000). nc=ceil((T+15*(B-1))/16); g=max(1,min(floor(SM/2),nc)); w=T*T/(4096*B). qlo=min(max(1,T-(B-1)*Q),Q); A=w+T/256; F=max(0,min(qlo*(K-63-qlo),Q*(K-63-Q)))/2048. PersistSmall=(ceil(nc/g)>=4 and (ceil(K/128)*g/w<=0.30 or (B*K-T)*8<=K or (K<=8*Q and ceil((K+15)/128)*g/(A+F)<=0.60))). Select exactly when not Split, B=4, T=16384, and PersistSmall.

## t_4b280404384bb5a3 | dispatch.when | 647 chars | 1 declarations

small_persistent_lpt:9 meta_data.dispatch.when

Use packaged defaults with no FA_HD256 environment overrides, under the root contract. Let B=BatchCount, T=TotalQ, Q=max_q_len, K=64*ceil(max_kv_len/64), SM=runtime multiprocessor count. C=ceil(T/16)*B; G=max(1,floor(SM/2)); Split=(2*C<=G and ceil(K/128)>=16 and 2*T*16*256*4<=24000000). nc=ceil((T+15*(B-1))/16); g=max(1,min(floor(SM/2),nc)); w=T*T/(4096*B). qlo=min(max(1,T-(B-1)*Q),Q); A=w+T/256; F=max(0,min(qlo*(K-63-qlo),Q*(K-63-Q)))/2048. PersistSmall=(ceil(nc/g)>=4 and (ceil(K/128)*g/w<=0.30 or (B*K-T)*8<=K or (K<=8*Q and ceil((K+15)/128)*g/(A+F)<=0.60))). Select exactly when not Split, 2<=B<=4, PersistSmall, and not (B=4 and T=16384).

## t_5117abcbb17b7105 | dispatch.when | 633 chars | 1 declarations

small_static_shared_rescale:9 meta_data.dispatch.when

Use packaged defaults with no FA_HD256 environment overrides, under the root contract. Let B=BatchCount, T=TotalQ, Q=max_q_len, K=64*ceil(max_kv_len/64), SM=runtime multiprocessor count. C=ceil(T/16)*B; G=max(1,floor(SM/2)); Split=(2*C<=G and ceil(K/128)>=16 and 2*T*16*256*4<=24000000). nc=ceil((T+15*(B-1))/16); g=max(1,min(floor(SM/2),nc)); w=T*T/(4096*B). qlo=min(max(1,T-(B-1)*Q),Q); A=w+T/256; F=max(0,min(qlo*(K-63-qlo),Q*(K-63-Q)))/2048. PersistSmall=(ceil(nc/g)>=4 and (ceil(K/128)*g/w<=0.30 or (B*K-T)*8<=K or (K<=8*Q and ceil((K+15)/128)*g/(A+F)<=0.60))). Select exactly when not Split, B=4, T=16384, and not PersistSmall.

## t_543b5759eefcb8fd | dispatch.when | 336 chars | 1 declarations

splitkv_dual_load:9 meta_data.dispatch.when

Use packaged defaults with no FA_HD256 environment overrides, under the root contract. Let B=BatchCount, T=TotalQ, Q=max_q_len, K=64*ceil(max_kv_len/64), SM=runtime multiprocessor count. C=ceil(T/16)*B; G=max(1,floor(SM/2)); Split=(2*C<=G and ceil(K/128)>=16 and 2*T*16*256*4<=24000000). Select exactly when Split and (B>1 or K>=98304).

## t_69c8c431c5aabf76 | dispatch.when | 363 chars | 1 declarations

single_persistent_dual_load:9 meta_data.dispatch.when

Use packaged defaults with no FA_HD256 environment overrides, under the root contract. Let B=BatchCount, T=TotalQ, Q=max_q_len, K=64*ceil(max_kv_len/64), SM=runtime multiprocessor count. C=ceil(T/16)*B; G=max(1,floor(SM/2)); Split=(2*C<=G and ceil(K/128)>=16 and 2*T*16*256*4<=24000000). Select exactly when not Split, B=1, ceil(Q/8)>SM+floor(SM/2), and K>=98304.

## t_709730c14ec9203e | dispatch.when | 363 chars | 1 declarations

rect_joint_load:9 meta_data.dispatch.when

Use packaged defaults with no FA_HD256 environment overrides, under the root contract. Let B=BatchCount, T=TotalQ, Q=max_q_len, K=64*ceil(max_kv_len/64), SM=runtime multiprocessor count. C=ceil(T/16)*B; G=max(1,floor(SM/2)); Split=(2*C<=G and ceil(K/128)>=16 and 2*T*16*256*4<=24000000). Select exactly when not Split, B=1, ceil(Q/8)<=SM+floor(SM/2), and K<98304.

## t_76d23997a65d73bd | dispatch.when | 381 chars | 1 declarations

root:8 meta_data.dispatch.when

Use packaged defaults with no FA_HD256 environment overrides, under the root contract. Let B=BatchCount, T=TotalQ, Q=max_q_len, K=64*ceil(max_kv_len/64), SM=runtime multiprocessor count. C=ceil(T/16)*B; G=max(1,floor(SM/2)); Split=(2*C<=G and ceil(K/128)>=16 and 2*T*16*256*4<=24000000). Select exactly when not Split and (B>16 or (B=1 and ceil(Q/8)<=SM+floor(SM/2) and K>=98304)).

## t_84497ad4c2c2cbd1 | dispatch.when | 332 chars | 1 declarations

splitkv_joint_load:9 meta_data.dispatch.when

Use packaged defaults with no FA_HD256 environment overrides, under the root contract. Let B=BatchCount, T=TotalQ, Q=max_q_len, K=64*ceil(max_kv_len/64), SM=runtime multiprocessor count. C=ceil(T/16)*B; G=max(1,floor(SM/2)); Split=(2*C<=G and ceil(K/128)>=16 and 2*T*16*256*4<=24000000). Select exactly when Split, B=1, and K<98304.

## t_a15d2ae59ab8c976 | dispatch.when | 331 chars | 1 declarations

packed_persistent:9 meta_data.dispatch.when

Use packaged defaults with no FA_HD256 environment overrides, under the root contract. Let B=BatchCount, T=TotalQ, Q=max_q_len, K=64*ceil(max_kv_len/64), SM=runtime multiprocessor count. C=ceil(T/16)*B; G=max(1,floor(SM/2)); Split=(2*C<=G and ceil(K/128)>=16 and 2*T*16*256*4<=24000000). Select exactly when not Split and 5<=B<=16.

## t_c64f2e95260985b4 | dispatch.when | 362 chars | 1 declarations

single_persistent_joint_load:9 meta_data.dispatch.when

Use packaged defaults with no FA_HD256 environment overrides, under the root contract. Let B=BatchCount, T=TotalQ, Q=max_q_len, K=64*ceil(max_kv_len/64), SM=runtime multiprocessor count. C=ceil(T/16)*B; G=max(1,floor(SM/2)); Split=(2*C<=G and ceil(K/128)>=16 and 2*T*16*256*4<=24000000). Select exactly when not Split, B=1, ceil(Q/8)>SM+floor(SM/2), and K<98304.

## t_d527482167e99293 | dispatch.when | 651 chars | 1 declarations

small_static_lpt:9 meta_data.dispatch.when

Use packaged defaults with no FA_HD256 environment overrides, under the root contract. Let B=BatchCount, T=TotalQ, Q=max_q_len, K=64*ceil(max_kv_len/64), SM=runtime multiprocessor count. C=ceil(T/16)*B; G=max(1,floor(SM/2)); Split=(2*C<=G and ceil(K/128)>=16 and 2*T*16*256*4<=24000000). nc=ceil((T+15*(B-1))/16); g=max(1,min(floor(SM/2),nc)); w=T*T/(4096*B). qlo=min(max(1,T-(B-1)*Q),Q); A=w+T/256; F=max(0,min(qlo*(K-63-qlo),Q*(K-63-Q)))/2048. PersistSmall=(ceil(nc/g)>=4 and (ceil(K/128)*g/w<=0.30 or (B*K-T)*8<=K or (K<=8*Q and ceil((K+15)/128)*g/(A+F)<=0.60))). Select exactly when not Split, 2<=B<=4, not PersistSmall, and not (B=4 and T=16384).

## t_680d6642a8764eac | hardware.execution | 55 chars | 1 declarations

root:73 contract.hardware_specification.execution.CTA

Warps co-resident on one SM, sharing SMEM and barriers.

## t_6f5eff329130bc7c | hardware.execution | 63 chars | 1 declarations

root:74 contract.hardware_specification.execution.cluster

CTAs co-scheduled within one GPC, with cluster synchronization.

## t_707ed1a5b6536df5 | hardware.execution | 31 chars | 1 declarations

root:72 contract.hardware_specification.execution.warp

32 SIMT threads within one CTA.

## t_263d5b133f5e61d6 | hardware.feature | 56 chars | 1 declarations

root:78 contract.hardware_specification.feature.register

Thread-private registers backed by the SM register file.

## t_6c67314fe3205270 | hardware.feature | 61 chars | 1 declarations

root:77 contract.hardware_specification.feature.TMEM

Explicitly allocated Tensor Core storage, separate from SMEM.

## t_bf65cfd621166b87 | hardware.feature | 101 chars | 1 declarations

root:79 contract.hardware_specification.feature.asynchronous_execution

Asynchronous transfers and MMA require their primitive-specific completion and visibility guarantees.

## t_fbe55b4f6cea5249 | hardware.feature | 88 chars | 1 declarations

root:76 contract.hardware_specification.feature.SMEM

CTA-lifetime shared storage, addressable by cluster peers while its owner remains alive.

## t_49c12c04800f7f35 | hardware.stack | 134 chars | 1 declarations

root:82 contract.hardware_specification.stack.toolchain

nvidia-cutlass-dsl 4.6.1 bundled compiler; torch 2.9.0+cu130; cuda-python 13.1.1; CUDA compiler and driver versions were not recorded.

## t_b3bd6578aa86a562 | hardware.stack | 35 chars | 1 declarations

root:81 contract.hardware_specification.stack.language

CuTe DSL, nvidia-cutlass-dsl==4.6.1

## t_05d9ebfd803511d2 | math.arithmetic | 168 chars | 11 declarations

packed_persistent:59 design.math_reshape.rewrite[1].arithmetic.accumulate; rect_joint_load:59 design.math_reshape.rewrite[1].arithmetic.accumulate; root:134 design.math_reshape.rewrite[1].arithmetic.accumulate; single_persistent_dual_load:59 design.math_reshape.rewrite[1].arithmetic.accumulate; single_persistent_joint_load:59 design.math_reshape.rewrite[1].arithmetic.accumulate; small_persistent_lpt:62 design.math_reshape.rewrite[1].arithmetic.accumulate; small_persistent_shared_rescale:62 design.math_reshape.rewrite[1].arithmetic.accumulate; small_static_lpt:62 design.math_reshape.rewrite[1].arithmetic.accumulate; small_static_shared_rescale:62 design.math_reshape.rewrite[1].arithmetic.accumulate; splitkv_dual_load:65 design.math_reshape.rewrite[1].arithmetic.accumulate; splitkv_joint_load:65 design.math_reshape.rewrite[1].arithmetic.accumulate

Seed both components of one FP32 pair with ell_previous*(alpha*0.5), seed three other pairs with zero, add four E fragments, merge (g0+g1)+(g2+g3), then add components.

## t_295a565bb7ad4dd4 | math.arithmetic | 71 chars | 2 declarations

splitkv_dual_load:81 design.math_reshape.rewrite[2].arithmetic.approx; splitkv_joint_load:81 design.math_reshape.rewrite[2].arithmetic.approx

Native fast exp2 and division have the unknown bounds stated in online.

## t_ace6651be84050cb | math.arithmetic | 134 chars | 2 declarations

splitkv_dual_load:82 design.math_reshape.rewrite[2].arithmetic.accumulate; splitkv_joint_load:82 design.math_reshape.rewrite[2].arithmetic.accumulate

Warp max and sum over padded split slots determine weights; output accumulates positive-weight terms in ascending split order in FP32.

## t_b54b6b0dabad52d2 | math.arithmetic | 152 chars | 9 declarations

packed_persistent:58 design.math_reshape.rewrite[1].arithmetic.approx; rect_joint_load:58 design.math_reshape.rewrite[1].arithmetic.approx; root:133 design.math_reshape.rewrite[1].arithmetic.approx; single_persistent_dual_load:58 design.math_reshape.rewrite[1].arithmetic.approx; single_persistent_joint_load:58 design.math_reshape.rewrite[1].arithmetic.approx; small_persistent_lpt:61 design.math_reshape.rewrite[1].arithmetic.approx; small_persistent_shared_rescale:61 design.math_reshape.rewrite[1].arithmetic.approx; small_static_lpt:61 design.math_reshape.rewrite[1].arithmetic.approx; small_static_shared_rescale:61 design.math_reshape.rewrite[1].arithmetic.approx

exp2_fast is native fast exp2 on SM103 without polynomial substitution; unknown: standalone fast-exp2 and output-division error bounds are not supplied.

## t_ff3ae9ec2975ed5d | math.arithmetic | 159 chars | 2 declarations

splitkv_dual_load:64 design.math_reshape.rewrite[1].arithmetic.approx; splitkv_joint_load:64 design.math_reshape.rewrite[1].arithmetic.approx

exp2_fast and log_fast are native fast math on SM103 without polynomial substitution; unknown: standalone exp2, log and division error bounds are not supplied.

## t_06311df5f243ef15 | math.ledger | 6 chars | 24 declarations

packed_persistent:63 design.math_reshape.ledger.footprint[0].by[1]; packed_persistent:65 design.math_reshape.ledger.error[0].by[0]; rect_joint_load:63 design.math_reshape.ledger.footprint[0].by[1]; rect_joint_load:65 design.math_reshape.ledger.error[0].by[0]; root:138 design.math_reshape.ledger.footprint[0].by[1]; root:140 design.math_reshape.ledger.error[0].by[0]; single_persistent_dual_load:63 design.math_reshape.ledger.footprint[0].by[1]; single_persistent_dual_load:65 design.math_reshape.ledger.error[0].by[0]; single_persistent_joint_load:63 design.math_reshape.ledger.footprint[0].by[1]; single_persistent_joint_load:65 design.math_reshape.ledger.error[0].by[0]; small_persistent_lpt:66 design.math_reshape.ledger.footprint[0].by[1]; small_persistent_lpt:68 design.math_reshape.ledger.error[0].by[0]; small_persistent_shared_rescale:66 design.math_reshape.ledger.footprint[0].by[1]; small_persistent_shared_rescale:68 design.math_reshape.ledger.error[0].by[0]; small_static_lpt:66 design.math_reshape.ledger.footprint[0].by[1]; small_static_lpt:68 design.math_reshape.ledger.error[0].by[0]; small_static_shared_rescale:66 design.math_reshape.ledger.footprint[0].by[1]; small_static_shared_rescale:68 design.math_reshape.ledger.error[0].by[0]; splitkv_dual_load:86 design.math_reshape.ledger.footprint[0].by[1]; splitkv_dual_load:88 design.math_reshape.ledger.traffic[0].by[0]; splitkv_dual_load:90 design.math_reshape.ledger.error[0].by[0]; splitkv_joint_load:86 design.math_reshape.ledger.footprint[0].by[1]; splitkv_joint_load:88 design.math_reshape.ledger.traffic[0].by[0]; splitkv_joint_load:90 design.math_reshape.ledger.error[0].by[0]

online

## t_1331b1b38b550457 | math.ledger | 109 chars | 9 declarations

packed_persistent:63 design.math_reshape.ledger.footprint[0].change; rect_joint_load:63 design.math_reshape.ledger.footprint[0].change; root:138 design.math_reshape.ledger.footprint[0].change; single_persistent_dual_load:63 design.math_reshape.ledger.footprint[0].change; single_persistent_joint_load:63 design.math_reshape.ledger.footprint[0].change; small_persistent_lpt:66 design.math_reshape.ledger.footprint[0].change; small_persistent_shared_rescale:66 design.math_reshape.ledger.footprint[0].change; small_static_lpt:66 design.math_reshape.ledger.footprint[0].change; small_static_shared_rescale:66 design.math_reshape.ledger.footprint[0].change

8*Hq*sum_b(Qlen(b)*L(b)) bytes -> M*(D+2) FP32 recurrent scalars per active worktile plus block-local values.

## t_425972b26b87b9e2 | math.ledger | 144 chars | 2 declarations

splitkv_dual_load:88 design.math_reshape.ledger.traffic[0].change; splitkv_joint_load:88 design.math_reshape.ledger.traffic[0].change

Up to +8*Ssplit*TotalQ*Hq*(D+1) logical bytes for one write/read per partial; empty splits leave O unwritten, and combine tiling may reread LSE.

## t_4c7dbfb7dcb4ee8e | math.ledger | 180 chars | 2 declarations

splitkv_dual_load:86 design.math_reshape.ledger.footprint[0].change; splitkv_joint_load:86 design.math_reshape.ledger.footprint[0].change

8*Hq*sum_b(Qlen(b)*L(b)) bytes -> M*(D+2) FP32 recurrent scalars per active worktile plus block-local values and 4*Ssplit*TotalQ*Hq*(D+1) bytes of reserved global partial capacity.

## t_57aaaf43c8724e4a | math.ledger | 6 chars | 11 declarations

packed_persistent:63 design.math_reshape.ledger.footprint[0].by[0]; rect_joint_load:63 design.math_reshape.ledger.footprint[0].by[0]; root:138 design.math_reshape.ledger.footprint[0].by[0]; single_persistent_dual_load:63 design.math_reshape.ledger.footprint[0].by[0]; single_persistent_joint_load:63 design.math_reshape.ledger.footprint[0].by[0]; small_persistent_lpt:66 design.math_reshape.ledger.footprint[0].by[0]; small_persistent_shared_rescale:66 design.math_reshape.ledger.footprint[0].by[0]; small_static_lpt:66 design.math_reshape.ledger.footprint[0].by[0]; small_static_shared_rescale:66 design.math_reshape.ledger.footprint[0].by[0]; splitkv_dual_load:86 design.math_reshape.ledger.footprint[0].by[0]; splitkv_joint_load:86 design.math_reshape.ledger.footprint[0].by[0]

stream

## t_6e6008f1013d18be | math.ledger | 103 chars | 11 declarations

packed_persistent:61 design.math_reshape.ledger.baseline; rect_joint_load:61 design.math_reshape.ledger.baseline; root:136 design.math_reshape.ledger.baseline; single_persistent_dual_load:61 design.math_reshape.ledger.baseline; single_persistent_joint_load:61 design.math_reshape.ledger.baseline; small_persistent_lpt:64 design.math_reshape.ledger.baseline; small_persistent_shared_rescale:64 design.math_reshape.ledger.baseline; small_static_lpt:64 design.math_reshape.ledger.baseline; small_static_shared_rescale:64 design.math_reshape.ledger.baseline; splitkv_dual_load:84 design.math_reshape.ledger.baseline; splitkv_joint_load:84 design.math_reshape.ledger.baseline

Materialize FP32 scores and probabilities for every request and query head, each written and read once.

## t_786c17c865d400d6 | math.ledger | 78 chars | 11 declarations

packed_persistent:65 design.math_reshape.ledger.error[0].change; rect_joint_load:65 design.math_reshape.ledger.error[0].change; root:140 design.math_reshape.ledger.error[0].change; single_persistent_dual_load:65 design.math_reshape.ledger.error[0].change; single_persistent_joint_load:65 design.math_reshape.ledger.error[0].change; small_persistent_lpt:68 design.math_reshape.ledger.error[0].change; small_persistent_shared_rescale:68 design.math_reshape.ledger.error[0].change; small_static_lpt:68 design.math_reshape.ledger.error[0].change; small_static_shared_rescale:68 design.math_reshape.ledger.error[0].change; splitkv_dual_load:90 design.math_reshape.ledger.error[0].change; splitkv_joint_load:90 design.math_reshape.ledger.error[0].change

+1 E4M3FN rounding per PV multiplicand; denominator retains FP32 exponentials.

## t_8d1a25742302e8ef | math.ledger | 7 chars | 2 declarations

splitkv_dual_load:88 design.math_reshape.ledger.traffic[0].by[1]; splitkv_joint_load:88 design.math_reshape.ledger.traffic[0].by[1]

combine

## t_985d31bab877ec22 | math.ledger | 21 chars | 11 declarations

packed_persistent:65 design.math_reshape.ledger.error[0].item; rect_joint_load:65 design.math_reshape.ledger.error[0].item; root:140 design.math_reshape.ledger.error[0].item; single_persistent_dual_load:65 design.math_reshape.ledger.error[0].item; single_persistent_joint_load:65 design.math_reshape.ledger.error[0].item; small_persistent_lpt:68 design.math_reshape.ledger.error[0].item; small_persistent_shared_rescale:68 design.math_reshape.ledger.error[0].item; small_static_lpt:68 design.math_reshape.ledger.error[0].item; small_static_shared_rescale:68 design.math_reshape.ledger.error[0].item; splitkv_dual_load:90 design.math_reshape.ledger.error[0].item; splitkv_joint_load:90 design.math_reshape.ledger.error[0].item

Probability precision

## t_a6bc41e9ce2671f1 | math.ledger | 19 chars | 11 declarations

packed_persistent:63 design.math_reshape.ledger.footprint[0].item; rect_joint_load:63 design.math_reshape.ledger.footprint[0].item; root:138 design.math_reshape.ledger.footprint[0].item; single_persistent_dual_load:63 design.math_reshape.ledger.footprint[0].item; single_persistent_joint_load:63 design.math_reshape.ledger.footprint[0].item; small_persistent_lpt:66 design.math_reshape.ledger.footprint[0].item; small_persistent_shared_rescale:66 design.math_reshape.ledger.footprint[0].item; small_static_lpt:66 design.math_reshape.ledger.footprint[0].item; small_static_shared_rescale:66 design.math_reshape.ledger.footprint[0].item; splitkv_dual_load:86 design.math_reshape.ledger.footprint[0].item; splitkv_joint_load:86 design.math_reshape.ledger.footprint[0].item

Dense intermediates

## t_af461efc9ac156d4 | math.ledger | 14 chars | 2 declarations

splitkv_dual_load:88 design.math_reshape.ledger.traffic[0].item; splitkv_joint_load:88 design.math_reshape.ledger.traffic[0].item

Split partials

## t_2aa9ec0d460fa91f | math.math | 507 chars | 2 declarations

splitkv_dual_load:68 design.math_reshape.rewrite[2].math; splitkv_joint_load:68 design.math_reshape.rewrite[2].math

For every token a and head h, z=max_s PartialLSE[s,h,a]; safez=0 if z=-inf, otherwise z. w_s=exp2_fast(PartialLSE[s,h,a]*log2(e)-safez*log2(e)); denom=sum32_s w_s. invdenom=1/denom if denom is nonzero and not NaN, otherwise zero; Weights[s]=w_s*invdenom. Facc(-1)=0; for s=0..Ssplit-1, Facc(s)=Facc(s-1)+Weights[s]*PartialO[s,a,h,:] if Weights[s]>0, otherwise Facc(s-1). Zero-weight partials never enter arithmetic. Fblock=bf16(Facc(Ssplit-1)); store its valid slice into out_after and return out as output.

## t_39c4d91fa4c3a8e4 | math.math | 893 chars | 2 declarations

small_persistent_shared_rescale:41 design.math_reshape.rewrite[1].math; small_static_shared_rescale:41 design.math_reshape.rewrite[1].math

For each valid row independently, c=fp32(bmm1_scale*log2(e)), with the expression inside the cast evaluated in Python binary64. Initialize mhat(-1)=-inf, ell(-1)=0, Y(-1)=0. For i=0..N-1, a=max(mhat(i-1),max_k X(i)[r,k]); safe=0 if a=-inf, otherwise a. keep=(c*(mhat(i-1)-safe)>=-tau); mhat(i)=mhat(i-1) if keep, otherwise a; base=mhat(i-1) if keep, otherwise safe. alpha(i)=1 if keep, otherwise exp2_fast(c*(mhat(i-1)-safe)); Stats(i)[r,0]=alpha(i). E(i)[r,k]=exp2_fast(fma32(c,X(i)[r,k],offset-c*base)); masked -inf maps to zero; P(i)=fp8_e4m3fn(E(i)). ell(i)=alpha(i)*ell(i-1)+sum32_k E(i)[r,k]. For d in {0,1}, Y(i)[:,128*d:128*(d+1)]=mma32(P(i),Vc(i)[:,128*d:128*(d+1)],alpha(i)*Y(i-1)[:,128*d:128*(d+1)]). invell=1/ell(N-1) if ell is nonzero and not NaN, otherwise zero; Oblock=bf16(Y(N-1)*invell). Store valid rows of Oblock into out_after by undoing head packing; return out as output.

## t_66ee124f50fdb01f | math.math | 1153 chars | 2 declarations

splitkv_dual_load:39 design.math_reshape.rewrite[1].math; splitkv_joint_load:39 design.math_reshape.rewrite[1].math

For each split s and valid row independently, c=fp32(bmm1_scale*log2(e)), with the expression inside the cast evaluated in Python binary64. Initialize mhat(-1)=-inf, ell(-1)=0, Y(-1)=0. For i=0..N-1, a=max(mhat(i-1),max_k X(i)[r,k]); safe=0 if a=-inf, otherwise a. keep=(c*(mhat(i-1)-safe)>=-tau); mhat(i)=mhat(i-1) if keep, otherwise a; base=mhat(i-1) if keep, otherwise safe. alpha(i)=1 if keep, otherwise exp2_fast(c*(mhat(i-1)-safe)); Stats(i)[r,:]=(mhat(i-1),base). E(i)[r,k]=exp2_fast(fma32(c,X(i)[r,k],offset-c*base)); masked -inf maps to zero; P(i)=fp8_e4m3fn(E(i)). ell(i)=alpha(i)*ell(i-1)+sum32_k E(i)[r,k]. For d in {0,1}, Y(i)[:,128*d:128*(d+1)]=mma32(P(i),Vc(i)[:,128*d:128*(d+1)],alpha(i)*Y(i-1)[:,128*d:128*(d+1)]). For N>0, invell=1/ell(N-1) if ell is nonzero and not NaN, otherwise zero; Opart=fp32(Y(N-1)*invell). LSEpart=fp32(bmm1_scale)*mhat(N-1)+log_fast(ell(N-1))-offset*ln(2) when ell is nonzero and not NaN, otherwise -inf. For r=h+Hq*u and a=cum_seq_lens_q[b]+16*m+u, store valid rows to PartialO[s,a,h,:]=Opart[r,:] and PartialLSE[s,h,a]=LSEpart[r]. For N=0, write only PartialLSE=-inf for valid rows; PartialO is unspecified.

## t_e8c20b1e4ec5817b | math.math | 596 chars | 11 declarations

packed_persistent:20 design.math_reshape.rewrite[0].math; rect_joint_load:20 design.math_reshape.rewrite[0].math; root:95 design.math_reshape.rewrite[0].math; single_persistent_dual_load:20 design.math_reshape.rewrite[0].math; single_persistent_joint_load:20 design.math_reshape.rewrite[0].math; small_persistent_lpt:23 design.math_reshape.rewrite[0].math; small_persistent_shared_rescale:23 design.math_reshape.rewrite[0].math; small_static_lpt:23 design.math_reshape.rewrite[0].math; small_static_shared_rescale:23 design.math_reshape.rewrite[0].math; splitkv_dual_load:20 design.math_reshape.rewrite[0].math; splitkv_joint_load:20 design.math_reshape.rewrite[0].math

For request b and query block m, Qc[r,d]=query[cum_seq_lens_q[b]+16*m+floor(r/Hq),r%Hq,d], 0<=r<M, on valid query rows. For 0<=i<N and 0<=k<Bk, j=Bk*sigma(i)+k and p=floor(j/Page). Read page p when p*Page<L(b), otherwise this block's first page; Kc(i)[k,d] and Vc(i)[k,d] use kind 0 and 1 of kv_cache. These are block views; no full-cache gather is materialized. Invalid rows and keys are excluded from results. S(i,0)=mm32(Qc[:,0:128],transpose(Kc(i)[:,0:128])); S(i,1)=mma32(Qc[:,128:256],transpose(Kc(i)[:,128:256]),S(i,0)). X(i)[r,k]=S(i,1)[r,k] if j in J(b,16*m+floor(r/Hq)), otherwise -inf.

## t_f6b241b38e7d2a6a | math.math | 901 chars | 7 declarations

packed_persistent:38 design.math_reshape.rewrite[1].math; rect_joint_load:38 design.math_reshape.rewrite[1].math; root:113 design.math_reshape.rewrite[1].math; single_persistent_dual_load:38 design.math_reshape.rewrite[1].math; single_persistent_joint_load:38 design.math_reshape.rewrite[1].math; small_persistent_lpt:41 design.math_reshape.rewrite[1].math; small_static_lpt:41 design.math_reshape.rewrite[1].math

For each valid row independently, c=fp32(bmm1_scale*log2(e)), with the expression inside the cast evaluated in Python binary64. Initialize mhat(-1)=-inf, ell(-1)=0, Y(-1)=0. For i=0..N-1, a=max(mhat(i-1),max_k X(i)[r,k]); safe=0 if a=-inf, otherwise a. keep=(c*(mhat(i-1)-safe)>=-tau); mhat(i)=mhat(i-1) if keep, otherwise a; base=mhat(i-1) if keep, otherwise safe. alpha(i)=1 if keep, otherwise exp2_fast(c*(mhat(i-1)-safe)); Stats(i)[r,:]=(mhat(i-1),base). E(i)[r,k]=exp2_fast(fma32(c,X(i)[r,k],offset-c*base)); masked -inf maps to zero; P(i)=fp8_e4m3fn(E(i)). ell(i)=alpha(i)*ell(i-1)+sum32_k E(i)[r,k]. For d in {0,1}, Y(i)[:,128*d:128*(d+1)]=mma32(P(i),Vc(i)[:,128*d:128*(d+1)],alpha(i)*Y(i-1)[:,128*d:128*(d+1)]). invell=1/ell(N-1) if ell is nonzero and not NaN, otherwise zero; Oblock=bf16(Y(N-1)*invell). Store valid rows of Oblock into out_after by undoing head packing; return out as output.

## t_0049d7b384b364f3 | math.meaning | 25 chars | 11 declarations

packed_persistent:29 design.math_reshape.rewrite[0].symbol.Bk.meaning; rect_joint_load:29 design.math_reshape.rewrite[0].symbol.Bk.meaning; root:104 design.math_reshape.rewrite[0].symbol.Bk.meaning; single_persistent_dual_load:29 design.math_reshape.rewrite[0].symbol.Bk.meaning; single_persistent_joint_load:29 design.math_reshape.rewrite[0].symbol.Bk.meaning; small_persistent_lpt:32 design.math_reshape.rewrite[0].symbol.Bk.meaning; small_persistent_shared_rescale:32 design.math_reshape.rewrite[0].symbol.Bk.meaning; small_static_lpt:32 design.math_reshape.rewrite[0].symbol.Bk.meaning; small_static_shared_rescale:32 design.math_reshape.rewrite[0].symbol.Bk.meaning; splitkv_dual_load:29 design.math_reshape.rewrite[0].symbol.Bk.meaning; splitkv_joint_load:29 design.math_reshape.rewrite[0].symbol.Bk.meaning

Keys per local iteration.

## t_038c3d5bb46c0ac2 | math.meaning | 36 chars | 11 declarations

packed_persistent:49 design.math_reshape.rewrite[1].symbol.tau.meaning; rect_joint_load:49 design.math_reshape.rewrite[1].symbol.tau.meaning; root:124 design.math_reshape.rewrite[1].symbol.tau.meaning; single_persistent_dual_load:49 design.math_reshape.rewrite[1].symbol.tau.meaning; single_persistent_joint_load:49 design.math_reshape.rewrite[1].symbol.tau.meaning; small_persistent_lpt:52 design.math_reshape.rewrite[1].symbol.tau.meaning; small_persistent_shared_rescale:52 design.math_reshape.rewrite[1].symbol.tau.meaning; small_static_lpt:52 design.math_reshape.rewrite[1].symbol.tau.meaning; small_static_shared_rescale:52 design.math_reshape.rewrite[1].symbol.tau.meaning; splitkv_dual_load:52 design.math_reshape.rewrite[1].symbol.tau.meaning; splitkv_joint_load:52 design.math_reshape.rewrite[1].symbol.tau.meaning

Maximum lag in base-two score units.

## t_0929fab35a683f2f | math.meaning | 30 chars | 2 declarations

splitkv_dual_load:76 design.math_reshape.rewrite[2].symbol.Dc.meaning; splitkv_joint_load:76 design.math_reshape.rewrite[2].symbol.Dc.meaning

Channels per combine worktile.

## t_09e928cb262d5e30 | math.meaning | 26 chars | 2 declarations

splitkv_dual_load:75 design.math_reshape.rewrite[2].symbol.Mc.meaning; splitkv_joint_load:75 design.math_reshape.rewrite[2].symbol.Mc.meaning

Rows per combine worktile.

## t_0bcfc49baf340b57 | math.meaning | 30 chars | 2 declarations

splitkv_dual_load:62 design.math_reshape.rewrite[1].symbol.PartialLSE.meaning; splitkv_joint_load:62 design.math_reshape.rewrite[1].symbol.PartialLSE.meaning

Materialized split normalizer.

## t_2518ac8186edf3bb | math.meaning | 42 chars | 9 declarations

packed_persistent:55 design.math_reshape.rewrite[1].symbol.Stats.meaning; rect_joint_load:55 design.math_reshape.rewrite[1].symbol.Stats.meaning; root:130 design.math_reshape.rewrite[1].symbol.Stats.meaning; single_persistent_dual_load:55 design.math_reshape.rewrite[1].symbol.Stats.meaning; single_persistent_joint_load:55 design.math_reshape.rewrite[1].symbol.Stats.meaning; small_persistent_lpt:58 design.math_reshape.rewrite[1].symbol.Stats.meaning; small_static_lpt:58 design.math_reshape.rewrite[1].symbol.Stats.meaning; splitkv_dual_load:58 design.math_reshape.rewrite[1].symbol.Stats.meaning; splitkv_joint_load:58 design.math_reshape.rewrite[1].symbol.Stats.meaning

Previous maximum and guarded current base.

## t_26fabafeeda8fcba | math.meaning | 42 chars | 2 declarations

splitkv_dual_load:77 design.math_reshape.rewrite[2].symbol.Weights.meaning; splitkv_joint_load:77 design.math_reshape.rewrite[2].symbol.Weights.meaning

Normalized split weights for combine rows.

## t_2cb2e8c963a0693e | math.meaning | 26 chars | 2 declarations

splitkv_dual_load:61 design.math_reshape.rewrite[1].symbol.PartialO.meaning; splitkv_joint_load:61 design.math_reshape.rewrite[1].symbol.PartialO.meaning

Materialized split output.

## t_5bc04719ba442986 | math.meaning | 19 chars | 9 declarations

packed_persistent:56 design.math_reshape.rewrite[1].symbol.Oblock.meaning; rect_joint_load:56 design.math_reshape.rewrite[1].symbol.Oblock.meaning; root:131 design.math_reshape.rewrite[1].symbol.Oblock.meaning; single_persistent_dual_load:56 design.math_reshape.rewrite[1].symbol.Oblock.meaning; single_persistent_joint_load:56 design.math_reshape.rewrite[1].symbol.Oblock.meaning; small_persistent_lpt:59 design.math_reshape.rewrite[1].symbol.Oblock.meaning; small_persistent_shared_rescale:59 design.math_reshape.rewrite[1].symbol.Oblock.meaning; small_static_lpt:59 design.math_reshape.rewrite[1].symbol.Oblock.meaning; small_static_shared_rescale:59 design.math_reshape.rewrite[1].symbol.Oblock.meaning

Final output block.

## t_61f451888e8e35e4 | math.meaning | 27 chars | 2 declarations

splitkv_dual_load:79 design.math_reshape.rewrite[2].symbol.Fblock.meaning; splitkv_joint_load:79 design.math_reshape.rewrite[2].symbol.Fblock.meaning

Final combine output slice.

## t_6de37aed25fccb86 | math.meaning | 26 chars | 2 declarations

splitkv_dual_load:78 design.math_reshape.rewrite[2].symbol.Facc.meaning; splitkv_joint_load:78 design.math_reshape.rewrite[2].symbol.Facc.meaning

Weighted split-output sum.

## t_747f29e9f0e9b6f8 | math.meaning | 22 chars | 11 declarations

packed_persistent:33 design.math_reshape.rewrite[0].symbol.Kc.meaning; rect_joint_load:33 design.math_reshape.rewrite[0].symbol.Kc.meaning; root:108 design.math_reshape.rewrite[0].symbol.Kc.meaning; single_persistent_dual_load:33 design.math_reshape.rewrite[0].symbol.Kc.meaning; single_persistent_joint_load:33 design.math_reshape.rewrite[0].symbol.Kc.meaning; small_persistent_lpt:36 design.math_reshape.rewrite[0].symbol.Kc.meaning; small_persistent_shared_rescale:36 design.math_reshape.rewrite[0].symbol.Kc.meaning; small_static_lpt:36 design.math_reshape.rewrite[0].symbol.Kc.meaning; small_static_shared_rescale:36 design.math_reshape.rewrite[0].symbol.Kc.meaning; splitkv_dual_load:34 design.math_reshape.rewrite[0].symbol.Kc.meaning; splitkv_joint_load:34 design.math_reshape.rewrite[0].symbol.Kc.meaning

Page-backed key block.

## t_79f97fd368d2c1ab | math.meaning | 40 chars | 11 declarations

packed_persistent:52 design.math_reshape.rewrite[1].symbol.ell.meaning; rect_joint_load:52 design.math_reshape.rewrite[1].symbol.ell.meaning; root:127 design.math_reshape.rewrite[1].symbol.ell.meaning; single_persistent_dual_load:52 design.math_reshape.rewrite[1].symbol.ell.meaning; single_persistent_joint_load:52 design.math_reshape.rewrite[1].symbol.ell.meaning; small_persistent_lpt:55 design.math_reshape.rewrite[1].symbol.ell.meaning; small_persistent_shared_rescale:55 design.math_reshape.rewrite[1].symbol.ell.meaning; small_static_lpt:55 design.math_reshape.rewrite[1].symbol.ell.meaning; small_static_shared_rescale:55 design.math_reshape.rewrite[1].symbol.ell.meaning; splitkv_dual_load:55 design.math_reshape.rewrite[1].symbol.ell.meaning; splitkv_joint_load:55 design.math_reshape.rewrite[1].symbol.ell.meaning

Denominator from unrounded exponentials.

## t_7c467842d95ac0d6 | math.meaning | 43 chars | 11 declarations

packed_persistent:35 design.math_reshape.rewrite[0].symbol.S.meaning; rect_joint_load:35 design.math_reshape.rewrite[0].symbol.S.meaning; root:110 design.math_reshape.rewrite[0].symbol.S.meaning; single_persistent_dual_load:35 design.math_reshape.rewrite[0].symbol.S.meaning; single_persistent_joint_load:35 design.math_reshape.rewrite[0].symbol.S.meaning; small_persistent_lpt:38 design.math_reshape.rewrite[0].symbol.S.meaning; small_persistent_shared_rescale:38 design.math_reshape.rewrite[0].symbol.S.meaning; small_static_lpt:38 design.math_reshape.rewrite[0].symbol.S.meaning; small_static_shared_rescale:38 design.math_reshape.rewrite[0].symbol.S.meaning; splitkv_dual_load:36 design.math_reshape.rewrite[0].symbol.S.meaning; splitkv_joint_load:36 design.math_reshape.rewrite[0].symbol.S.meaning

Raw score after a channel-half contraction.

## t_989bd323a430b3d8 | math.meaning | 24 chars | 11 declarations

packed_persistent:34 design.math_reshape.rewrite[0].symbol.Vc.meaning; rect_joint_load:34 design.math_reshape.rewrite[0].symbol.Vc.meaning; root:109 design.math_reshape.rewrite[0].symbol.Vc.meaning; single_persistent_dual_load:34 design.math_reshape.rewrite[0].symbol.Vc.meaning; single_persistent_joint_load:34 design.math_reshape.rewrite[0].symbol.Vc.meaning; small_persistent_lpt:37 design.math_reshape.rewrite[0].symbol.Vc.meaning; small_persistent_shared_rescale:37 design.math_reshape.rewrite[0].symbol.Vc.meaning; small_static_lpt:37 design.math_reshape.rewrite[0].symbol.Vc.meaning; small_static_shared_rescale:37 design.math_reshape.rewrite[0].symbol.Vc.meaning; splitkv_dual_load:35 design.math_reshape.rewrite[0].symbol.Vc.meaning; splitkv_joint_load:35 design.math_reshape.rewrite[0].symbol.Vc.meaning

Page-backed value block.

## t_99fd84ba6cd5c6fa | math.meaning | 32 chars | 2 declarations

splitkv_dual_load:59 design.math_reshape.rewrite[1].symbol.Opart.meaning; splitkv_joint_load:59 design.math_reshape.rewrite[1].symbol.Opart.meaning

Locally normalized split output.

## t_a87c2d1020c8f6ca | math.meaning | 25 chars | 11 declarations

packed_persistent:28 design.math_reshape.rewrite[0].symbol.M.meaning; rect_joint_load:28 design.math_reshape.rewrite[0].symbol.M.meaning; root:103 design.math_reshape.rewrite[0].symbol.M.meaning; single_persistent_dual_load:28 design.math_reshape.rewrite[0].symbol.M.meaning; single_persistent_joint_load:28 design.math_reshape.rewrite[0].symbol.M.meaning; small_persistent_lpt:31 design.math_reshape.rewrite[0].symbol.M.meaning; small_persistent_shared_rescale:31 design.math_reshape.rewrite[0].symbol.M.meaning; small_static_lpt:31 design.math_reshape.rewrite[0].symbol.M.meaning; small_static_shared_rescale:31 design.math_reshape.rewrite[0].symbol.M.meaning; splitkv_dual_load:28 design.math_reshape.rewrite[0].symbol.M.meaning; splitkv_joint_load:28 design.math_reshape.rewrite[0].symbol.M.meaning

Packed rows per worktile.

## t_acf33ce1dcd115ee | math.meaning | 22 chars | 11 declarations

packed_persistent:54 design.math_reshape.rewrite[1].symbol.P.meaning; rect_joint_load:54 design.math_reshape.rewrite[1].symbol.P.meaning; root:129 design.math_reshape.rewrite[1].symbol.P.meaning; single_persistent_dual_load:54 design.math_reshape.rewrite[1].symbol.P.meaning; single_persistent_joint_load:54 design.math_reshape.rewrite[1].symbol.P.meaning; small_persistent_lpt:57 design.math_reshape.rewrite[1].symbol.P.meaning; small_persistent_shared_rescale:57 design.math_reshape.rewrite[1].symbol.P.meaning; small_static_lpt:57 design.math_reshape.rewrite[1].symbol.P.meaning; small_static_shared_rescale:57 design.math_reshape.rewrite[1].symbol.P.meaning; splitkv_dual_load:57 design.math_reshape.rewrite[1].symbol.P.meaning; splitkv_joint_load:57 design.math_reshape.rewrite[1].symbol.P.meaning

Value-product operand.

## t_bd25cb36761ad3d7 | math.meaning | 27 chars | 11 declarations

packed_persistent:51 design.math_reshape.rewrite[1].symbol.mhat.meaning; rect_joint_load:51 design.math_reshape.rewrite[1].symbol.mhat.meaning; root:126 design.math_reshape.rewrite[1].symbol.mhat.meaning; single_persistent_dual_load:51 design.math_reshape.rewrite[1].symbol.mhat.meaning; single_persistent_joint_load:51 design.math_reshape.rewrite[1].symbol.mhat.meaning; small_persistent_lpt:54 design.math_reshape.rewrite[1].symbol.mhat.meaning; small_persistent_shared_rescale:54 design.math_reshape.rewrite[1].symbol.mhat.meaning; small_static_lpt:54 design.math_reshape.rewrite[1].symbol.mhat.meaning; small_static_shared_rescale:54 design.math_reshape.rewrite[1].symbol.mhat.meaning; splitkv_dual_load:54 design.math_reshape.rewrite[1].symbol.mhat.meaning; splitkv_joint_load:54 design.math_reshape.rewrite[1].symbol.mhat.meaning

Retained raw-score maximum.

## t_c27b5dc640a5d436 | math.meaning | 32 chars | 11 declarations

packed_persistent:31 design.math_reshape.rewrite[0].symbol.sigma.meaning; rect_joint_load:31 design.math_reshape.rewrite[0].symbol.sigma.meaning; root:106 design.math_reshape.rewrite[0].symbol.sigma.meaning; single_persistent_dual_load:31 design.math_reshape.rewrite[0].symbol.sigma.meaning; single_persistent_joint_load:31 design.math_reshape.rewrite[0].symbol.sigma.meaning; small_persistent_lpt:34 design.math_reshape.rewrite[0].symbol.sigma.meaning; small_persistent_shared_rescale:34 design.math_reshape.rewrite[0].symbol.sigma.meaning; small_static_lpt:34 design.math_reshape.rewrite[0].symbol.sigma.meaning; small_static_shared_rescale:34 design.math_reshape.rewrite[0].symbol.sigma.meaning; splitkv_dual_load:32 design.math_reshape.rewrite[0].symbol.sigma.meaning; splitkv_joint_load:32 design.math_reshape.rewrite[0].symbol.sigma.meaning

Local iteration to KV block map.

## t_c5783d2c0f2c773a | math.meaning | 25 chars | 11 declarations

packed_persistent:30 design.math_reshape.rewrite[0].symbol.N.meaning; rect_joint_load:30 design.math_reshape.rewrite[0].symbol.N.meaning; root:105 design.math_reshape.rewrite[0].symbol.N.meaning; single_persistent_dual_load:30 design.math_reshape.rewrite[0].symbol.N.meaning; single_persistent_joint_load:30 design.math_reshape.rewrite[0].symbol.N.meaning; small_persistent_lpt:33 design.math_reshape.rewrite[0].symbol.N.meaning; small_persistent_shared_rescale:33 design.math_reshape.rewrite[0].symbol.N.meaning; small_static_lpt:33 design.math_reshape.rewrite[0].symbol.N.meaning; small_static_shared_rescale:33 design.math_reshape.rewrite[0].symbol.N.meaning; splitkv_dual_load:31 design.math_reshape.rewrite[0].symbol.N.meaning; splitkv_joint_load:31 design.math_reshape.rewrite[0].symbol.N.meaning

Local KV iteration count.

## t_e3258a3551f02a45 | math.meaning | 19 chars | 11 declarations

packed_persistent:32 design.math_reshape.rewrite[0].symbol.Qc.meaning; rect_joint_load:32 design.math_reshape.rewrite[0].symbol.Qc.meaning; root:107 design.math_reshape.rewrite[0].symbol.Qc.meaning; single_persistent_dual_load:32 design.math_reshape.rewrite[0].symbol.Qc.meaning; single_persistent_joint_load:32 design.math_reshape.rewrite[0].symbol.Qc.meaning; small_persistent_lpt:35 design.math_reshape.rewrite[0].symbol.Qc.meaning; small_persistent_shared_rescale:35 design.math_reshape.rewrite[0].symbol.Qc.meaning; small_static_lpt:35 design.math_reshape.rewrite[0].symbol.Qc.meaning; small_static_shared_rescale:35 design.math_reshape.rewrite[0].symbol.Qc.meaning; splitkv_dual_load:33 design.math_reshape.rewrite[0].symbol.Qc.meaning; splitkv_joint_load:33 design.math_reshape.rewrite[0].symbol.Qc.meaning

Packed query block.

## t_e7dafb796defead3 | math.meaning | 25 chars | 2 declarations

small_persistent_shared_rescale:58 design.math_reshape.rewrite[1].symbol.Stats.meaning; small_static_shared_rescale:58 design.math_reshape.rewrite[1].symbol.Stats.meaning

Shared correction factor.

## t_ed772745ff2816d0 | math.meaning | 44 chars | 11 declarations

packed_persistent:50 design.math_reshape.rewrite[1].symbol.offset.meaning; rect_joint_load:50 design.math_reshape.rewrite[1].symbol.offset.meaning; root:125 design.math_reshape.rewrite[1].symbol.offset.meaning; single_persistent_dual_load:50 design.math_reshape.rewrite[1].symbol.offset.meaning; single_persistent_joint_load:50 design.math_reshape.rewrite[1].symbol.offset.meaning; small_persistent_lpt:53 design.math_reshape.rewrite[1].symbol.offset.meaning; small_persistent_shared_rescale:53 design.math_reshape.rewrite[1].symbol.offset.meaning; small_static_lpt:53 design.math_reshape.rewrite[1].symbol.offset.meaning; small_static_shared_rescale:53 design.math_reshape.rewrite[1].symbol.offset.meaning; splitkv_dual_load:53 design.math_reshape.rewrite[1].symbol.offset.meaning; splitkv_joint_load:53 design.math_reshape.rewrite[1].symbol.offset.meaning

Base-two shift before probability narrowing.

## t_f48ebd7d53175717 | math.meaning | 33 chars | 11 declarations

packed_persistent:53 design.math_reshape.rewrite[1].symbol.Y.meaning; rect_joint_load:53 design.math_reshape.rewrite[1].symbol.Y.meaning; root:128 design.math_reshape.rewrite[1].symbol.Y.meaning; single_persistent_dual_load:53 design.math_reshape.rewrite[1].symbol.Y.meaning; single_persistent_joint_load:53 design.math_reshape.rewrite[1].symbol.Y.meaning; small_persistent_lpt:56 design.math_reshape.rewrite[1].symbol.Y.meaning; small_persistent_shared_rescale:56 design.math_reshape.rewrite[1].symbol.Y.meaning; small_static_lpt:56 design.math_reshape.rewrite[1].symbol.Y.meaning; small_static_shared_rescale:56 design.math_reshape.rewrite[1].symbol.Y.meaning; splitkv_dual_load:56 design.math_reshape.rewrite[1].symbol.Y.meaning; splitkv_joint_load:56 design.math_reshape.rewrite[1].symbol.Y.meaning

Numerator from FP8 probabilities.

## t_fce232dcd3f45c2c | math.meaning | 15 chars | 2 declarations

splitkv_dual_load:30 design.math_reshape.rewrite[0].symbol.Ssplit.meaning; splitkv_joint_load:30 design.math_reshape.rewrite[0].symbol.Ssplit.meaning

KV split count.

## t_fce2f37637e0ef05 | math.meaning | 29 chars | 2 declarations

splitkv_dual_load:60 design.math_reshape.rewrite[1].symbol.LSEpart.meaning; splitkv_joint_load:60 design.math_reshape.rewrite[1].symbol.LSEpart.meaning

Natural-log split normalizer.

## t_4fbfc903557f6ab4 | math.name | 44 chars | 2 declarations

splitkv_dual_load:67 design.math_reshape.rewrite[2].name; splitkv_joint_load:67 design.math_reshape.rewrite[2].name

Merge normalized partials by log normalizers

## t_75e75d24e9daf384 | math.name | 46 chars | 11 declarations

packed_persistent:19 design.math_reshape.rewrite[0].name; rect_joint_load:19 design.math_reshape.rewrite[0].name; root:94 design.math_reshape.rewrite[0].name; single_persistent_dual_load:19 design.math_reshape.rewrite[0].name; single_persistent_joint_load:19 design.math_reshape.rewrite[0].name; small_persistent_lpt:22 design.math_reshape.rewrite[0].name; small_persistent_shared_rescale:22 design.math_reshape.rewrite[0].name; small_static_lpt:22 design.math_reshape.rewrite[0].name; small_static_shared_rescale:22 design.math_reshape.rewrite[0].name; splitkv_dual_load:19 design.math_reshape.rewrite[0].name; splitkv_joint_load:19 design.math_reshape.rewrite[0].name

Pack heads and stream page-backed score blocks

## t_90ab6038e36612bf | math.name | 63 chars | 11 declarations

packed_persistent:37 design.math_reshape.rewrite[1].name; rect_joint_load:37 design.math_reshape.rewrite[1].name; root:112 design.math_reshape.rewrite[1].name; single_persistent_dual_load:37 design.math_reshape.rewrite[1].name; single_persistent_joint_load:37 design.math_reshape.rewrite[1].name; small_persistent_lpt:40 design.math_reshape.rewrite[1].name; small_persistent_shared_rescale:40 design.math_reshape.rewrite[1].name; small_static_lpt:40 design.math_reshape.rewrite[1].name; small_static_shared_rescale:40 design.math_reshape.rewrite[1].name; splitkv_dual_load:38 design.math_reshape.rewrite[1].name; splitkv_joint_load:38 design.math_reshape.rewrite[1].name

Lagged maximum and distinct denominator and numerator precision

## t_cd9bee15f13aa998 | math.step | 6 chars | 11 declarations

packed_persistent:36 design.math_reshape.rewrite[1].step; rect_joint_load:36 design.math_reshape.rewrite[1].step; root:111 design.math_reshape.rewrite[1].step; single_persistent_dual_load:36 design.math_reshape.rewrite[1].step; single_persistent_joint_load:36 design.math_reshape.rewrite[1].step; small_persistent_lpt:39 design.math_reshape.rewrite[1].step; small_persistent_shared_rescale:39 design.math_reshape.rewrite[1].step; small_static_lpt:39 design.math_reshape.rewrite[1].step; small_static_shared_rescale:39 design.math_reshape.rewrite[1].step; splitkv_dual_load:37 design.math_reshape.rewrite[1].step; splitkv_joint_load:37 design.math_reshape.rewrite[1].step

online

## t_deb99bb16abaca20 | math.step | 7 chars | 2 declarations

splitkv_dual_load:66 design.math_reshape.rewrite[2].step; splitkv_joint_load:66 design.math_reshape.rewrite[2].step

combine

## t_edf4686bf66b674d | math.step | 6 chars | 11 declarations

packed_persistent:18 design.math_reshape.rewrite[0].step; rect_joint_load:18 design.math_reshape.rewrite[0].step; root:93 design.math_reshape.rewrite[0].step; single_persistent_dual_load:18 design.math_reshape.rewrite[0].step; single_persistent_joint_load:18 design.math_reshape.rewrite[0].step; small_persistent_lpt:21 design.math_reshape.rewrite[0].step; small_persistent_shared_rescale:21 design.math_reshape.rewrite[0].step; small_static_lpt:21 design.math_reshape.rewrite[0].step; small_static_shared_rescale:21 design.math_reshape.rewrite[0].step; splitkv_dual_load:18 design.math_reshape.rewrite[0].step; splitkv_joint_load:18 design.math_reshape.rewrite[0].step

stream

## t_9c07758b497a2ef3 | math.thesis | 188 chars | 2 declarations

splitkv_dual_load:16 design.math_reshape.thesis; splitkv_joint_load:16 design.math_reshape.thesis

Stream each contiguous KV split with lagged-max FP8 online attention, then combine normalized FP32 partials using their natural-log weights; scratch and a second launch buy KV parallelism.

## t_b191ca68f04fd5a2 | math.thesis | 192 chars | 9 declarations

packed_persistent:16 design.math_reshape.thesis; rect_joint_load:16 design.math_reshape.thesis; root:91 design.math_reshape.thesis; single_persistent_dual_load:16 design.math_reshape.thesis; single_persistent_joint_load:16 design.math_reshape.thesis; small_persistent_lpt:19 design.math_reshape.thesis; small_persistent_shared_rescale:19 design.math_reshape.thesis; small_static_lpt:19 design.math_reshape.thesis; small_static_shared_rescale:19 design.math_reshape.thesis

Pack GQA heads into rows and replace dense attention intermediates with online state; lag the maximum and offset exponentials before FP8 rounding, trading numerator precision for FP8 products.

## t_67ec5eed05852bce | metadata.describe | 106 chars | 11 declarations

packed_persistent:3 meta_data.describe; rect_joint_load:3 meta_data.describe; root:3 meta_data.describe; single_persistent_dual_load:3 meta_data.describe; single_persistent_joint_load:3 meta_data.describe; small_persistent_lpt:3 meta_data.describe; small_persistent_shared_rescale:3 meta_data.describe; small_static_lpt:3 meta_data.describe; small_static_shared_rescale:3 meta_data.describe; splitkv_dual_load:3 meta_data.describe; splitkv_joint_load:3 meta_data.describe

Compute right-aligned causal GQA from FP8 queries and an HND paged KV cache into the supplied BF16 output.

## t_527f908e5c416728 | pipeline.thesis | 209 chars | 1 declarations

splitkv_dual_load:147 design.tile_pipeline.thesis

Independent K and V producers feed separate four-stage rings for each split; QK leads PV by one block while softmax and correction exchange numerator ownership. A separate cooperative kernel combines partials.

## t_78e1d33a31a0cebd | pipeline.thesis | 189 chars | 1 declarations

rect_joint_load:97 design.tile_pipeline.thesis

One loader follows the QK-ahead-of-PV issue order through a shared four-token stream; separate K/V arrays and aliased score slots support softmax and correction overlap within one worktile.

## t_9568d6de9290314d | pipeline.thesis | 197 chars | 1 declarations

single_persistent_joint_load:99 design.tile_pipeline.thesis

One loader follows the QK-ahead-of-PV issue order through a shared four-token stream; separate K/V arrays and aliased score slots support softmax and correction overlap across persistent worktiles.

## t_a2744a6c1b1e6196 | pipeline.thesis | 192 chars | 3 declarations

packed_persistent:99 design.tile_pipeline.thesis; single_persistent_dual_load:99 design.tile_pipeline.thesis; small_persistent_lpt:104 design.tile_pipeline.thesis

Independent K and V producers feed separate four-stage rings; QK leads PV by one block while softmax and correction exchange aliased scores and numerator ownership across persistent worktiles.

## t_a3012fa5216a027f | pipeline.thesis | 209 chars | 1 declarations

small_persistent_shared_rescale:104 design.tile_pipeline.thesis

Independent K and V producers feed separate four-stage rings; QK leads PV by one block while softmax sends a one-column shared alpha to correction and exchanges numerator ownership across persistent worktiles.

## t_be34cac52f128adf | pipeline.thesis | 184 chars | 2 declarations

root:174 design.tile_pipeline.thesis; small_static_lpt:104 design.tile_pipeline.thesis

Independent K and V producers feed separate four-stage rings; QK leads PV by one block while softmax and correction exchange aliased scores and numerator ownership within one worktile.

## t_d5862581a9a20229 | pipeline.thesis | 201 chars | 1 declarations

small_static_shared_rescale:104 design.tile_pipeline.thesis

Independent K and V producers feed separate four-stage rings; QK leads PV by one block while softmax sends a one-column shared alpha to correction and exchanges numerator ownership within one worktile.

## t_e33e3dae98888e96 | pipeline.thesis | 245 chars | 1 declarations

splitkv_joint_load:147 design.tile_pipeline.thesis

One loader follows the QK-ahead-of-PV issue order through a shared four-token stream; separate K/V arrays and aliased score slots support softmax and correction overlap within each nonempty split; a separate cooperative kernel combines partials.

## t_2c617ee62b3ac57e | pipeline.verb | 158 chars | 9 declarations

packed_persistent:206 design.tile_pipeline.phase.attention.role.Correction.info.verb.rescale; rect_joint_load:181 design.tile_pipeline.phase.attention.role.Correction.info.verb.rescale; root:269 design.tile_pipeline.phase.attention.role.Correction.info.verb.rescale; single_persistent_dual_load:206 design.tile_pipeline.phase.attention.role.Correction.info.verb.rescale; single_persistent_joint_load:193 design.tile_pipeline.phase.attention.role.Correction.info.verb.rescale; small_persistent_lpt:211 design.tile_pipeline.phase.attention.role.Correction.info.verb.rescale; small_static_lpt:203 design.tile_pipeline.phase.attention.role.Correction.info.verb.rescale; splitkv_dual_load:249 design.tile_pipeline.phase.attention.role.Correction.info.verb.rescale; splitkv_joint_load:237 design.tile_pipeline.phase.attention.role.Correction.info.verb.rescale

Evaluate alpha from Stats and correct both Y halves after previous PV; a warp vote skips rewriting when no row has alpha<1, retaining the ownership handshake.

## t_2d1fac0abee9b0f4 | pipeline.verb | 66 chars | 2 declarations

splitkv_dual_load:295 design.tile_pipeline.phase.combine.role.Main.info.verb.store.meaning; splitkv_joint_load:283 design.tile_pipeline.phase.combine.role.Main.info.verb.store.meaning

Narrow Facc to BF16 and coalescently store its valid output slice.

## t_2ea662537d951788 | pipeline.verb | 173 chars | 2 declarations

splitkv_dual_load:289 design.tile_pipeline.phase.combine.role.Main.info.verb.publish.meaning; splitkv_joint_load:277 design.tile_pipeline.phase.combine.role.Main.info.verb.publish.meaning

Wait for LSE copies and rendezvous the CTA; compute combine weights and per-row LastValidSplit, overwrite LSEWeight, then rendezvous before consuming the transposed weights.

## t_338c5553b8c7bc6d | pipeline.verb | 125 chars | 4 declarations

rect_joint_load:165 design.tile_pipeline.phase.attention.role.Softmax.info.verb.publish; root:253 design.tile_pipeline.phase.attention.role.Softmax.info.verb.publish; splitkv_dual_load:231 design.tile_pipeline.phase.attention.role.Softmax.info.verb.publish; splitkv_joint_load:219 design.tile_pipeline.phase.attention.role.Softmax.info.verb.publish

Stats carries old maximum and base; P applies online with per-tile state reset; FinalSum publishes ell(N-1) after all blocks.

## t_36584397ce8761c1 | pipeline.verb | 292 chars | 1 declarations

small_static_shared_rescale:186 design.tile_pipeline.phase.attention.role.Softmax.info.verb.publish

Publish the elected thread's mapped request and query coordinates once with CTA-wide visibility; Stats publishes the single FP32 alpha evaluated by Softmax and reused by both denominator and Correction; P applies online with per-tile state reset; FinalSum publishes ell(N-1) after all blocks.

## t_44b25f0d1b31e8e9 | pipeline.verb | 101 chars | 5 declarations

packed_persistent:210 design.tile_pipeline.phase.attention.role.Correction.info.verb.release; single_persistent_dual_load:210 design.tile_pipeline.phase.attention.role.Correction.info.verb.release; single_persistent_joint_load:197 design.tile_pipeline.phase.attention.role.Correction.info.verb.release; small_persistent_lpt:215 design.tile_pipeline.phase.attention.role.Correction.info.verb.release; small_persistent_shared_rescale:215 design.tile_pipeline.phase.attention.role.Correction.info.verb.release

All Y reads are complete; the next tile may initialize Y while the final staged half is being stored.

## t_4627114363c20b62 | pipeline.verb | 253 chars | 1 declarations

root:229 design.tile_pipeline.phase.attention.role.MMA.info.verb.qk

Issue one QK channel half; d=0 initializes S and d=1 completes it. Done returns K; last QK returns Q. Products may overlap; a product's done covers all earlier products issued by this role. Earlier PV reads P before later QK overwrites its aliased slot.

## t_5cb5be92e4a4a4c9 | pipeline.verb | 124 chars | 9 declarations

packed_persistent:209 design.tile_pipeline.phase.attention.role.Correction.info.verb.store; rect_joint_load:184 design.tile_pipeline.phase.attention.role.Correction.info.verb.store; root:272 design.tile_pipeline.phase.attention.role.Correction.info.verb.store; single_persistent_dual_load:209 design.tile_pipeline.phase.attention.role.Correction.info.verb.store; single_persistent_joint_load:196 design.tile_pipeline.phase.attention.role.Correction.info.verb.store; small_persistent_lpt:214 design.tile_pipeline.phase.attention.role.Correction.info.verb.store; small_persistent_shared_rescale:214 design.tile_pipeline.phase.attention.role.Correction.info.verb.store; small_static_lpt:206 design.tile_pipeline.phase.attention.role.Correction.info.verb.store; small_static_shared_rescale:206 design.tile_pipeline.phase.attention.role.Correction.info.verb.store

Coalesced valid-row stores undo head packing; completion ends OutputStage reads and includes the group barrier before reuse.

## t_6c7ff93a7fd13ed1 | pipeline.verb | 163 chars | 2 declarations

small_persistent_shared_rescale:211 design.tile_pipeline.phase.attention.role.Correction.info.verb.rescale; small_static_shared_rescale:203 design.tile_pipeline.phase.attention.role.Correction.info.verb.rescale

Read alpha directly from Stats and correct both Y halves after previous PV; a warp vote skips rewriting when no row has alpha<1, retaining the ownership handshake.

## t_7404fea5c1001a3b | pipeline.verb | 183 chars | 7 declarations

packed_persistent:161 design.tile_pipeline.phase.attention.role.MMA.info.verb.qk; single_persistent_dual_load:161 design.tile_pipeline.phase.attention.role.MMA.info.verb.qk; small_persistent_lpt:166 design.tile_pipeline.phase.attention.role.MMA.info.verb.qk; small_persistent_shared_rescale:166 design.tile_pipeline.phase.attention.role.MMA.info.verb.qk; small_static_lpt:162 design.tile_pipeline.phase.attention.role.MMA.info.verb.qk; small_static_shared_rescale:162 design.tile_pipeline.phase.attention.role.MMA.info.verb.qk; splitkv_dual_load:205 design.tile_pipeline.phase.attention.role.MMA.info.verb.qk

Issue one QK channel half; d=0 initializes S and d=1 completes it. Done returns K; last QK returns Q. Ordered pair products read earlier P before later QK overwrites its aliased slot.

## t_7465bbd91a3e7371 | pipeline.verb | 239 chars | 2 declarations

splitkv_dual_load:293 design.tile_pipeline.phase.combine.role.Main.info.verb.accumulate.meaning; splitkv_joint_load:281 design.tile_pipeline.phase.combine.role.Main.info.verb.accumulate.meaning

Wait for the current partial stage; initialize Facc before s=0 and apply combine. Each warp owns one row; skip nonpositive weights and iterations beyond its last valid split. Reads are thread-private, so no per-stage CTA barrier is needed.

## t_806327463a254c76 | pipeline.verb | 222 chars | 1 declarations

small_static_lpt:186 design.tile_pipeline.phase.attention.role.Softmax.info.verb.publish

Publish the elected thread's mapped request and query coordinates once with CTA-wide visibility; Stats carries old maximum and base; P applies online with per-tile state reset; FinalSum publishes ell(N-1) after all blocks.

## t_8472b8bf5f6d80b9 | pipeline.verb | 217 chars | 2 declarations

splitkv_dual_load:285 design.tile_pipeline.phase.combine.role.Main.info.verb.load; splitkv_joint_load:273 design.tile_pipeline.phase.combine.role.Main.info.verb.load

Issue cp.async copies; PartialLSE includes -inf padding. The first three partial prefetches may read unspecified empty-split payloads; later copies skip splits beyond each thread-owned row group's last nonempty split.

## t_94309e835096e85f | pipeline.verb | 148 chars | 11 declarations

packed_persistent:186 design.tile_pipeline.phase.attention.role.Softmax.info.verb.capture; rect_joint_load:164 design.tile_pipeline.phase.attention.role.Softmax.info.verb.capture; root:252 design.tile_pipeline.phase.attention.role.Softmax.info.verb.capture; single_persistent_dual_load:186 design.tile_pipeline.phase.attention.role.Softmax.info.verb.capture; single_persistent_joint_load:173 design.tile_pipeline.phase.attention.role.Softmax.info.verb.capture; small_persistent_lpt:191 design.tile_pipeline.phase.attention.role.Softmax.info.verb.capture; small_persistent_shared_rescale:191 design.tile_pipeline.phase.attention.role.Softmax.info.verb.capture; small_static_lpt:185 design.tile_pipeline.phase.attention.role.Softmax.info.verb.capture; small_static_shared_rescale:185 design.tile_pipeline.phase.attention.role.Softmax.info.verb.capture; splitkv_dual_load:229 design.tile_pipeline.phase.attention.role.Softmax.info.verb.capture; splitkv_joint_load:217 design.tile_pipeline.phase.attention.role.Softmax.info.verb.capture

Capture scores and return their slot; use the load-reduced maximum for the fully visible prefix, and recompute after masking the causal/tail suffix.

## t_94ac361b1ae881e0 | pipeline.verb | 83 chars | 2 declarations

splitkv_dual_load:251 design.tile_pipeline.phase.attention.role.Correction.info.verb.stage; splitkv_joint_load:239 design.tile_pipeline.phase.attention.role.Correction.info.verb.stage

Normalize one Y half into FP32 OutputStage; completion covers the correction group.

## t_9bb88449bf8dbd38 | pipeline.verb | 280 chars | 3 declarations

rect_joint_load:147 design.tile_pipeline.phase.attention.role.MMA.info.verb.contract; single_persistent_joint_load:155 design.tile_pipeline.phase.attention.role.MMA.info.verb.contract; splitkv_joint_load:200 design.tile_pipeline.phase.attention.role.MMA.info.verb.contract

K tokens perform QK, initializing S at d=0 and completing at d=1; V tokens perform PV, initializing Y at i=0 and using corrected Y thereafter. Completion covers both CTAs and returns the operand token. Ordered products read earlier P before a later QK overwrites its aliased slot.

## t_9fc2bd79eb80daf3 | pipeline.verb | 188 chars | 2 declarations

splitkv_dual_load:252 design.tile_pipeline.phase.attention.role.Correction.info.verb.store; splitkv_joint_load:240 design.tile_pipeline.phase.attention.role.Correction.info.verb.store

Coalesced stores write the split PartialO slice; completion ends OutputStage reads and includes the group barrier. EmptyLSE writes only PartialLSE=-inf without waiting on unused pipelines.

## t_a5cca1d18a2e6ed8 | pipeline.verb | 89 chars | 9 declarations

packed_persistent:208 design.tile_pipeline.phase.attention.role.Correction.info.verb.stage; rect_joint_load:183 design.tile_pipeline.phase.attention.role.Correction.info.verb.stage; root:271 design.tile_pipeline.phase.attention.role.Correction.info.verb.stage; single_persistent_dual_load:208 design.tile_pipeline.phase.attention.role.Correction.info.verb.stage; single_persistent_joint_load:195 design.tile_pipeline.phase.attention.role.Correction.info.verb.stage; small_persistent_lpt:213 design.tile_pipeline.phase.attention.role.Correction.info.verb.stage; small_persistent_shared_rescale:213 design.tile_pipeline.phase.attention.role.Correction.info.verb.stage; small_static_lpt:205 design.tile_pipeline.phase.attention.role.Correction.info.verb.stage; small_static_shared_rescale:205 design.tile_pipeline.phase.attention.role.Correction.info.verb.stage

Normalize and narrow one Y half into OutputStage; completion covers the correction group.

## t_a85b960ddb3dab48 | pipeline.verb | 284 chars | 1 declarations

small_persistent_shared_rescale:192 design.tile_pipeline.phase.attention.role.Softmax.info.verb.publish

Build immutable RequestMap once per CTA and release all roles after CTA-wide visibility; Stats publishes the single FP32 alpha evaluated by Softmax and reused by both denominator and Correction; P applies online with per-tile state reset; FinalSum publishes ell(N-1) after all blocks.

## t_bf61446f0729da28 | pipeline.verb | 119 chars | 2 declarations

splitkv_dual_load:230 design.tile_pipeline.phase.attention.role.Softmax.info.verb.produce; splitkv_joint_load:218 design.tile_pipeline.phase.attention.role.Softmax.info.verb.produce

Compute and store LSEpart into PartialLSE for valid rows after publishing FinalSum; correction can already stage Opart.

## t_c100bdea654dace0 | pipeline.verb | 360 chars | 3 declarations

rect_joint_load:130 design.tile_pipeline.phase.attention.role.Load.info.verb.load; single_persistent_joint_load:135 design.tile_pipeline.phase.attention.role.Load.info.verb.load; splitkv_joint_load:182 design.tile_pipeline.phase.attention.role.Load.info.verb.load

KVStage is a token, with K at p=max(0,2*i-1) and V at p=min(2*i+2,2*N-1), each with halves d=0,1. Issue K0,K1,V0,K2,V1,...,Klast,Vpenultimate,Vlast; N=1 uses K0,V0. Separate payload arrays share one four-token acquire/release stream. Q contains this member's rows; K uses one page per member/half and V uses two. Both members contribute to pair-wide readiness.

## t_ce274e8894eaf92a | pipeline.verb | 144 chars | 8 declarations

packed_persistent:162 design.tile_pipeline.phase.attention.role.MMA.info.verb.pv; root:230 design.tile_pipeline.phase.attention.role.MMA.info.verb.pv; single_persistent_dual_load:162 design.tile_pipeline.phase.attention.role.MMA.info.verb.pv; small_persistent_lpt:167 design.tile_pipeline.phase.attention.role.MMA.info.verb.pv; small_persistent_shared_rescale:167 design.tile_pipeline.phase.attention.role.MMA.info.verb.pv; small_static_lpt:163 design.tile_pipeline.phase.attention.role.MMA.info.verb.pv; small_static_shared_rescale:163 design.tile_pipeline.phase.attention.role.MMA.info.verb.pv; splitkv_dual_load:206 design.tile_pipeline.phase.attention.role.MMA.info.verb.pv

Issue one PV output half; i=0 initializes Y and later blocks use corrected Y. Done returns V; d=1 publishes Y and returns P, covering both CTAs.

## t_d9cd95fa7788ec17 | pipeline.verb | 198 chars | 1 declarations

root:268 design.tile_pipeline.phase.attention.role.Correction.info.verb.capture

Stats and FinalSum refer to this CTA's Softmax publications. Acknowledge first-block Stats without reading its payload; capture later Stats and final ell into registers before returning their slots.

## t_e512ef5cedb4d1e1 | pipeline.verb | 137 chars | 10 declarations

packed_persistent:205 design.tile_pipeline.phase.attention.role.Correction.info.verb.capture; rect_joint_load:180 design.tile_pipeline.phase.attention.role.Correction.info.verb.capture; single_persistent_dual_load:205 design.tile_pipeline.phase.attention.role.Correction.info.verb.capture; single_persistent_joint_load:192 design.tile_pipeline.phase.attention.role.Correction.info.verb.capture; small_persistent_lpt:210 design.tile_pipeline.phase.attention.role.Correction.info.verb.capture; small_persistent_shared_rescale:210 design.tile_pipeline.phase.attention.role.Correction.info.verb.capture; small_static_lpt:202 design.tile_pipeline.phase.attention.role.Correction.info.verb.capture; small_static_shared_rescale:202 design.tile_pipeline.phase.attention.role.Correction.info.verb.capture; splitkv_dual_load:248 design.tile_pipeline.phase.attention.role.Correction.info.verb.capture; splitkv_joint_load:236 design.tile_pipeline.phase.attention.role.Correction.info.verb.capture

Acknowledge first-block Stats without reading its payload; capture later Stats and final ell into registers before returning their slots.

## t_ee029bd4bfe3c351 | pipeline.verb | 123 chars | 8 declarations

packed_persistent:150 design.tile_pipeline.phase.attention.role.LoadV.info.verb.load; root:219 design.tile_pipeline.phase.attention.role.LoadV.info.verb.load; single_persistent_dual_load:150 design.tile_pipeline.phase.attention.role.LoadV.info.verb.load; small_persistent_lpt:155 design.tile_pipeline.phase.attention.role.LoadV.info.verb.load; small_persistent_shared_rescale:155 design.tile_pipeline.phase.attention.role.LoadV.info.verb.load; small_static_lpt:151 design.tile_pipeline.phase.attention.role.LoadV.info.verb.load; small_static_shared_rescale:151 design.tile_pipeline.phase.attention.role.LoadV.info.verb.load; splitkv_dual_load:195 design.tile_pipeline.phase.attention.role.LoadV.info.verb.load

Each V half uses two page transfers per member; done covers both and contributes to pair-wide readiness independently of K.

## t_f98ffc677442ff07 | pipeline.verb | 134 chars | 8 declarations

packed_persistent:133 design.tile_pipeline.phase.attention.role.LoadQK.info.verb.load; root:205 design.tile_pipeline.phase.attention.role.LoadQK.info.verb.load; single_persistent_dual_load:133 design.tile_pipeline.phase.attention.role.LoadQK.info.verb.load; small_persistent_lpt:138 design.tile_pipeline.phase.attention.role.LoadQK.info.verb.load; small_persistent_shared_rescale:138 design.tile_pipeline.phase.attention.role.LoadQK.info.verb.load; small_static_lpt:136 design.tile_pipeline.phase.attention.role.LoadQK.info.verb.load; small_static_shared_rescale:136 design.tile_pipeline.phase.attention.role.LoadQK.info.verb.load; splitkv_dual_load:180 design.tile_pipeline.phase.attention.role.LoadQK.info.verb.load

Q holds this member's rows for the KV traversal; each K half uses one page per member. Both members contribute to pair-wide readiness.

## t_ff12b271ba9c3460 | pipeline.verb | 214 chars | 4 declarations

packed_persistent:187 design.tile_pipeline.phase.attention.role.Softmax.info.verb.publish; single_persistent_dual_load:187 design.tile_pipeline.phase.attention.role.Softmax.info.verb.publish; single_persistent_joint_load:174 design.tile_pipeline.phase.attention.role.Softmax.info.verb.publish; small_persistent_lpt:192 design.tile_pipeline.phase.attention.role.Softmax.info.verb.publish

Build immutable RequestMap once per CTA and release all roles after CTA-wide visibility; Stats carries old maximum and base; P applies online with per-tile state reset; FinalSum publishes ell(N-1) after all blocks.

## t_0425c7d2822cadd3 | realization.choice | 102 chars | 11 declarations

packed_persistent:230 realization.require[0].choice; rect_joint_load:202 realization.require[0].choice; root:290 realization.require[0].choice; single_persistent_dual_load:230 realization.require[0].choice; single_persistent_joint_load:217 realization.require[0].choice; small_persistent_lpt:235 realization.require[0].choice; small_persistent_shared_rescale:235 realization.require[0].choice; small_static_lpt:224 realization.require[0].choice; small_static_shared_rescale:224 realization.require[0].choice; splitkv_dual_load:311 realization.require[0].choice; splitkv_joint_load:299 realization.require[0].choice

Use two-CTA tcgen05 MMA with cumulative completion and earlier-PV-read-before-later-QK-write ordering.

## t_29fda24ab1b4f9e7 | realization.choice | 64 chars | 2 declarations

splitkv_dual_load:314 realization.require[3].choice; splitkv_joint_load:302 realization.require[3].choice

Use cp.async commit/wait groups for the four-stage partial ring.

## t_a009767f679ced1d | realization.choice | 65 chars | 11 declarations

packed_persistent:231 realization.require[1].choice; rect_joint_load:203 realization.require[1].choice; root:291 realization.require[1].choice; single_persistent_dual_load:231 realization.require[1].choice; single_persistent_joint_load:218 realization.require[1].choice; small_persistent_lpt:236 realization.require[1].choice; small_persistent_shared_rescale:236 realization.require[1].choice; small_static_lpt:225 realization.require[1].choice; small_static_shared_rescale:225 realization.require[1].choice; splitkv_dual_load:312 realization.require[1].choice; splitkv_joint_load:300 realization.require[1].choice

Use native page-64 TMA for HND FP8 KV and descriptor views for Q.

## t_b26038334e8e6428 | realization.choice | 97 chars | 11 declarations

packed_persistent:232 realization.require[2].choice; rect_joint_load:204 realization.require[2].choice; root:292 realization.require[2].choice; single_persistent_dual_load:232 realization.require[2].choice; single_persistent_joint_load:219 realization.require[2].choice; small_persistent_lpt:237 realization.require[2].choice; small_persistent_shared_rescale:237 realization.require[2].choice; small_static_lpt:226 realization.require[2].choice; small_static_shared_rescale:226 realization.require[2].choice; splitkv_dual_load:313 realization.require[2].choice; splitkv_joint_load:301 realization.require[2].choice

Launch 384 threads per CTA with setmaxnreg budgets and one contiguous 512-column TMEM allocation.

## t_381c2211d1ce88c5 | realization.reason | 74 chars | 11 declarations

packed_persistent:231 realization.require[1].reason; rect_joint_load:203 realization.require[1].reason; root:291 realization.require[1].reason; single_persistent_dual_load:231 realization.require[1].reason; single_persistent_joint_load:218 realization.require[1].reason; small_persistent_lpt:236 realization.require[1].reason; small_persistent_shared_rescale:236 realization.require[1].reason; small_static_lpt:225 realization.require[1].reason; small_static_shared_rescale:225 realization.require[1].reason; splitkv_dual_load:312 realization.require[1].reason; splitkv_joint_load:300 realization.require[1].reason

Pair-wide transaction completion implements the declared operand handoffs.

## t_539672a8e7f29b78 | realization.reason | 91 chars | 11 declarations

packed_persistent:230 realization.require[0].reason; rect_joint_load:202 realization.require[0].reason; root:290 realization.require[0].reason; single_persistent_dual_load:230 realization.require[0].reason; single_persistent_joint_load:217 realization.require[0].reason; small_persistent_lpt:235 realization.require[0].reason; small_persistent_shared_rescale:235 realization.require[0].reason; small_static_lpt:224 realization.require[0].reason; small_static_shared_rescale:224 realization.require[0].reason; splitkv_dual_load:311 realization.require[0].reason; splitkv_joint_load:299 realization.require[0].reason

Pair-wide operand returns and aliased score/probability storage depend on these guarantees.

## t_8335a20ddb3a8a9e | realization.reason | 111 chars | 2 declarations

splitkv_dual_load:314 realization.require[3].reason; splitkv_joint_load:302 realization.require[3].reason

Three-stage prefetch overlaps copies and weighted accumulation while preserving thread-private stage readiness.

## t_b1cb841883ea8348 | realization.reason | 96 chars | 11 declarations

packed_persistent:232 realization.require[2].reason; rect_joint_load:204 realization.require[2].reason; root:292 realization.require[2].reason; single_persistent_dual_load:232 realization.require[2].reason; single_persistent_joint_load:219 realization.require[2].reason; small_persistent_lpt:237 realization.require[2].reason; small_persistent_shared_rescale:237 realization.require[2].reason; small_static_lpt:226 realization.require[2].reason; small_static_shared_rescale:226 realization.require[2].reason; splitkv_dual_load:313 realization.require[2].reason; splitkv_joint_load:301 realization.require[2].reason

Idle warps relinquish registers and the score/numerator resources share one allocation lifetime.

## t_45f4169d210726fd | schedule.describe | 73 chars | 2 declarations

single_persistent_dual_load:72 design.work_schedule.phase.attention.describe; single_persistent_joint_load:72 design.work_schedule.phase.attention.describe

Final BF16 attention with a single-request flattened persistent schedule.

## t_4dab27ea52cdb39f | schedule.describe | 51 chars | 2 declarations

splitkv_dual_load:128 design.work_schedule.phase.combine.describe; splitkv_joint_load:128 design.work_schedule.phase.combine.describe

Reduce all splits into one final BF16 output slice.

## t_53dddb1c944a2d53 | schedule.describe | 74 chars | 2 declarations

small_persistent_lpt:75 design.work_schedule.phase.attention.describe; small_persistent_shared_rescale:75 design.work_schedule.phase.attention.describe

Final BF16 attention with a small-batch LPT flattened persistent schedule.

## t_63921062fe5b1306 | schedule.describe | 81 chars | 2 declarations

splitkv_dual_load:100 design.work_schedule.phase.attention.describe; splitkv_joint_load:100 design.work_schedule.phase.attention.describe

Produce FP32 partial outputs and LSE with a rectangular static split-KV schedule.

## t_856c6813706c7a37 | schedule.describe | 81 chars | 1 declarations

packed_persistent:72 design.work_schedule.phase.attention.describe

Final BF16 attention with a original-request-order flattened persistent schedule.

## t_909561ef5d3e2cbc | schedule.describe | 70 chars | 2 declarations

small_static_lpt:75 design.work_schedule.phase.attention.describe; small_static_shared_rescale:75 design.work_schedule.phase.attention.describe

Final BF16 attention with a small-batch LPT flattened static schedule.

## t_f8d68a32ae40dfb4 | schedule.describe | 56 chars | 2 declarations

rect_joint_load:72 design.work_schedule.phase.attention.describe; root:147 design.work_schedule.phase.attention.describe

Final BF16 attention with a rectangular static schedule.

## t_044b296d58f01412 | schedule.strategy | 156 chars | 1 declarations

root:162 design.work_schedule.phase.attention.mapping.strategy

Member list entry a is CTA rank a, for a=0,1. Rank a supplies K tokens [64*a,64*(a+1)) per channel half and V columns [128*d+64*a,128*d+64*(a+1)) for d=0,1.

## t_0f2409562337689a | schedule.strategy | 239 chars | 1 declarations

packed_persistent:93 design.work_schedule.phase.attention.schedule.strategy

Keep input request order. Build the request-id/prefix map once per CTA; find each segment with a fixed four-step binary search of its prefix boundaries. Within a segment, ordinal x maps to m=2*x if x<ceil(C_b/2), otherwise m=2*(C_b-1-x)+1.

## t_6381618b4f7d8e3a | schedule.strategy | 155 chars | 5 declarations

packed_persistent:87 design.work_schedule.phase.attention.mapping.strategy; single_persistent_dual_load:87 design.work_schedule.phase.attention.mapping.strategy; single_persistent_joint_load:87 design.work_schedule.phase.attention.mapping.strategy; small_persistent_lpt:90 design.work_schedule.phase.attention.mapping.strategy; small_persistent_shared_rescale:90 design.work_schedule.phase.attention.mapping.strategy

SMCount is the runtime multiprocessor count; rank a supplies K tokens [64*a,64*(a+1)) per channel half and V columns [128*d+64*a,128*d+64*(a+1)) for d=0,1.

## t_92b4ae1649eb5a2c | schedule.strategy | 149 chars | 3 declarations

rect_joint_load:93 design.work_schedule.phase.attention.schedule.strategy; splitkv_dual_load:124 design.work_schedule.phase.attention.schedule.strategy; splitkv_joint_load:124 design.work_schedule.phase.attention.schedule.strategy

For ordinal x in [0,Crect), map to m=2*x if x<ceil(Crect/2), otherwise m=2*(Crect-1-x)+1; use grid coordinates directly, with query clusters fastest.

## t_94808f9cd9fcbbc0 | schedule.strategy | 265 chars | 2 declarations

rect_joint_load:74 design.work_schedule.phase.attention.decomposition.strategy; root:149 design.work_schedule.phase.attention.decomposition.strategy

M=256; Bk=128; Crect=ceil(TotalQ/16), independently for every request. A worktile covers 16 query tokens and all Hq heads; N=min(ceil(L(b)/Bk),ceil((L(b)-Qlen(b)+16*(m+1))/Bk)); sigma(i)=i. Reject m>=ceil(Qlen(b)/16) before pipeline actions. Active tiles have N>=1.

## t_a78539136fa99eb1 | schedule.strategy | 110 chars | 5 declarations

rect_joint_load:87 design.work_schedule.phase.attention.mapping.strategy; small_static_lpt:90 design.work_schedule.phase.attention.mapping.strategy; small_static_shared_rescale:90 design.work_schedule.phase.attention.mapping.strategy; splitkv_dual_load:118 design.work_schedule.phase.attention.mapping.strategy; splitkv_joint_load:118 design.work_schedule.phase.attention.mapping.strategy

Rank a supplies K tokens [64*a,64*(a+1)) per channel half and V columns [128*d+64*a,128*d+64*(a+1)) for d=0,1.

## t_b0cd643f1a97dbe0 | schedule.strategy | 339 chars | 7 declarations

packed_persistent:74 design.work_schedule.phase.attention.decomposition.strategy; single_persistent_dual_load:74 design.work_schedule.phase.attention.decomposition.strategy; single_persistent_joint_load:74 design.work_schedule.phase.attention.decomposition.strategy; small_persistent_lpt:77 design.work_schedule.phase.attention.decomposition.strategy; small_persistent_shared_rescale:77 design.work_schedule.phase.attention.decomposition.strategy; small_static_lpt:77 design.work_schedule.phase.attention.decomposition.strategy; small_static_shared_rescale:77 design.work_schedule.phase.attention.decomposition.strategy

M=256; Bk=128; C_b=ceil(Qlen(b)/16); Cpad=ceil((TotalQ+15*(BatchCount-1))/16). A worktile covers 16 query tokens and all Hq heads; N=min(ceil(L(b)/Bk),ceil((L(b)-Qlen(b)+16*(m+1))/Bk)); sigma(i)=i. Flatten request segments of C_b tiles into Cpad positions; reject positions beyond sum_b C_b before pipeline actions. Active tiles have N>=1.

## t_d582ae5d06eca3ad | schedule.strategy | 645 chars | 2 declarations

splitkv_dual_load:102 design.work_schedule.phase.attention.decomposition.strategy; splitkv_joint_load:102 design.work_schedule.phase.attention.decomposition.strategy

M=256; Bk=128; Crect=ceil(TotalQ/16), independently for every request. SMCount is the runtime multiprocessor count; C=BatchCount*Crect; G=max(1,floor(SMCount/2)); Kblocks=ceil(64*ceil(max_kv_len/64)/Bk). Ssplit=min(floor(G/C),floor(Kblocks/2),floor(24000000/(TotalQ*Hq*D*4)),64); this dispatch guarantees Ssplit>=2. Nfull=min(ceil(L(b)/Bk),ceil((L(b)-Qlen(b)+16*(m+1))/Bk)); span=ceil(ceil(L(b)/Bk)/Ssplit). Split s owns [lo,hi), where lo=s*span, hi=min((s+1)*span,ceil(L(b)/Bk),Nfull); N=max(0,hi-lo); sigma(i)=lo+i. Reject m>=ceil(Qlen(b)/16). A valid query tile with N=0 writes only -inf LSE and performs no operand traffic or pipeline waits.

## t_d5cce79e73e9346a | schedule.strategy | 401 chars | 1 declarations

small_static_shared_rescale:96 design.work_schedule.phase.attention.schedule.strategy

Order requests by descending L(b), using strict-greater compare-swaps on request IDs initially in input order. Compare position pairs (0,1),(2,3),(0,2),(1,3),(1,2); ties are not generally stable. An elected thread resolves the assigned segment once and publishes the request and local query coordinates to its CTA. Within a segment, ordinal x maps to m=2*x if x<ceil(C_b/2), otherwise m=2*(C_b-1-x)+1.

## t_d7fd3f1512f83c2f | schedule.strategy | 455 chars | 2 declarations

splitkv_dual_load:130 design.work_schedule.phase.combine.decomposition.strategy; splitkv_joint_load:130 design.work_schedule.phase.combine.decomposition.strategy

Mc=8; Dc=128; Smax=32 if Ssplit<=32, otherwise 64; pad absent LSE slots with -inf. A request-local row x maps to h=floor(x/Qlen(b)), q=x%Qlen(b), a=cum_seq_lens_q[b]+q; combine is head-major, attention is head-fast. Read PartialO[s,a,h,d] and PartialLSE[s,h,a]; the combine interface takes a zero-copy [split,token,head] transpose of PartialLSE. Launch TotalQ*Hq rows for each request; reject blocks beyond Qlen(b)*Hq and mask its final partial row block.

## t_e0f245e8e619776f | schedule.strategy | 447 chars | 1 declarations

small_static_lpt:96 design.work_schedule.phase.attention.schedule.strategy

Order requests by descending L(b), using strict-greater compare-swaps on request IDs initially in input order. Pair positions are (0,1) for B=2; (0,1),(1,2),(0,1) for B=3; (0,1),(2,3),(0,2),(1,3),(1,2) for B=4; ties are not generally stable. An elected thread resolves the assigned segment once and publishes the request and local query coordinates to its CTA. Within a segment, ordinal x maps to m=2*x if x<ceil(C_b/2), otherwise m=2*(C_b-1-x)+1.

## t_e981ca6a6599f148 | schedule.strategy | 199 chars | 1 declarations

root:168 design.work_schedule.phase.attention.schedule.strategy

Enumerate request outside query_cluster, so query_cluster changes fastest. For its ordinal x in [0,Crect), map to m=2*x if x<ceil(Crect/2), otherwise m=2*(Crect-1-x)+1; use grid coordinates directly.

## t_ed0b29bd56dcfc85 | schedule.strategy | 415 chars | 1 declarations

small_persistent_lpt:96 design.work_schedule.phase.attention.schedule.strategy

Order requests by descending L(b), using strict-greater compare-swaps on request IDs initially in input order. Pair positions are (0,1) for B=2; (0,1),(1,2),(0,1) for B=3; (0,1),(2,3),(0,2),(1,3),(1,2) for B=4; ties are not generally stable. Build this request-id/prefix map once per CTA; resolve each position by linear lookup. Within a segment, ordinal x maps to m=2*x if x<ceil(C_b/2), otherwise m=2*(C_b-1-x)+1.

## t_f41f272e46bf56c0 | schedule.strategy | 203 chars | 2 declarations

single_persistent_dual_load:93 design.work_schedule.phase.attention.schedule.strategy; single_persistent_joint_load:93 design.work_schedule.phase.attention.schedule.strategy

The request order is identity. Build the request-id/prefix map once per CTA; resolve each position by linear lookup. Within a segment, ordinal x maps to m=2*x if x<ceil(C_b/2), otherwise m=2*(C_b-1-x)+1.

## t_f42a81f4f55bd391 | schedule.strategy | 369 chars | 1 declarations

small_persistent_shared_rescale:96 design.work_schedule.phase.attention.schedule.strategy

Order requests by descending L(b), using strict-greater compare-swaps on request IDs initially in input order. Compare position pairs (0,1),(2,3),(0,2),(1,3),(1,2); ties are not generally stable. Build this request-id/prefix map once per CTA; resolve each position by linear lookup. Within a segment, ordinal x maps to m=2*x if x<ceil(C_b/2), otherwise m=2*(C_b-1-x)+1.

## t_11987257cbd81380 | schedule.thesis | 139 chars | 2 declarations

splitkv_dual_load:92 design.work_schedule.thesis; splitkv_joint_load:92 design.work_schedule.thesis

Distribute contiguous KV splits over a rectangular CTA-pair grid, then launch independent eight-row combine CTAs after attention completes.

## t_4ee18d17dc7e402c | schedule.thesis | 140 chars | 2 declarations

small_static_lpt:70 design.work_schedule.thesis; small_static_shared_rescale:70 design.work_schedule.thesis

Front-load longer KV requests and fold their query tiles; launch one CTA pair per position so hardware can schedule each tile independently.

## t_66ed307219a81919 | schedule.thesis | 125 chars | 2 declarations

rect_joint_load:67 design.work_schedule.thesis; root:142 design.work_schedule.thesis

Use a rectangular request/query grid with one CTA pair per position; fold the query axis and reject excess positions locally.

## t_7a82e33ea2268243 | schedule.thesis | 125 chars | 3 declarations

packed_persistent:67 design.work_schedule.thesis; single_persistent_dual_load:67 design.work_schedule.thesis; single_persistent_joint_load:67 design.work_schedule.thesis

Reuse a bounded grid of CTA pairs over folded query work; retain mapping and pipeline state across statically assigned tiles.

## t_87f5385b19ce1801 | schedule.thesis | 161 chars | 2 declarations

small_persistent_lpt:70 design.work_schedule.thesis; small_persistent_shared_rescale:70 design.work_schedule.thesis

Front-load longer KV requests, fold their query tiles, and stride a bounded persistent grid over the resulting sequence; amortize request-map and pipeline setup.
