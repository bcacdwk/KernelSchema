# extract_gemm: full open-text inventory

These are exact-text groups, not proof of semantic equivalence across contexts.

## t_8b54553cca183883 | algorithm_contract.math | 618 chars | 1 declarations

root:92 contract.algorithm_requirement.compute[2].math

For each integer 0 <= g < E, m_indptr[g] <= i < m_indptr[g+1] and 0 <= j < N: out[i,j] = bf16(sum_{t=0}^{K-1} A_value[i,t] * B_value[g,j,t]). Multiplication and accumulation use FP32. The final BF16 conversion uses RNE without clamping, including conversion of non-finite FP32 results. Empty groups have no output rows. The groups partition all M rows; no prior destination value, additive term or logical-row mask is used. Overwrite the entire out_buffer view and return it as out, preserving object and storage identity. Other inputs and storage outside that view, including backing-storage guards, remain unchanged.

## t_b855bdfda2c21bf2 | algorithm_contract.math | 670 chars | 1 declarations

root:69 contract.algorithm_requirement.compute[0].math

1 <= M <= 65536; E is in {8, 16, 32, 64}. (N, K) is in {(1536, 2048), (3072, 4096), (4096, 1536), (4096, 7168), (7168, 2048)}; N is divisible by ScaleN and K by ScaleK. m_indptr[0] = 0 and m_indptr[E] = M. For each integer g with 0 <= g < E, the difference m_indptr[g+1] - m_indptr[g] is in [0, 8192] and is divisible by 4. All tensor arguments are contiguous CUDA tensors on the same device. All listed inputs are supplied; out_buffer and out_dtype are non-None. The scale widths and selectors have exactly the values specified in input. Logical group sizes and generator modes are descriptive metadata, not extra arguments or row masks; every physical row is computed.

## t_c0ce2b20ec02d0ba | algorithm_contract.math | 327 chars | 1 declarations

root:82 contract.algorithm_requirement.compute[1].math

For integers 0 <= i < M, 0 <= j < N, 0 <= t < K and 0 <= g < E: A_value[i,t] = a[i,t] * a_scale[floor(t/ScaleK),i]. B_value[g,j,t] = b[g,j,t] * b_scale[g,floor(t/ScaleK),floor(j/ScaleN)]. Products use FP32; stored payloads and scales are the numerical inputs, with no re-quantization or reconstruction of pre-quantization data.

## t_0e3776e379253e1c | algorithm_contract.meaning | 57 chars | 1 declarations

root:62 contract.algorithm_requirement.input.out_dtype.meaning

Requested output type represented by a torch.dtype object

## t_1400e4bac74024b0 | algorithm_contract.meaning | 54 chars | 1 declarations

root:48 contract.algorithm_requirement.input.b_scale.meaning

Weight scale for each group and reduction-column block

## t_18a078d160a54ac7 | algorithm_contract.meaning | 47 chars | 1 declarations

root:63 contract.algorithm_requirement.input.backend.meaning

Backend selector represented by a Python string

## t_3450b624cbe078a9 | algorithm_contract.meaning | 48 chars | 1 declarations

root:49 contract.algorithm_requirement.input.m_indptr.meaning

Cumulative physical row boundaries of the groups

## t_38b61d84dc4fc540 | algorithm_contract.meaning | 75 chars | 1 declarations

root:58 contract.algorithm_requirement.input.out_buffer.meaning

Caller-owned destination passed as out; its incoming contents are not read.

## t_4570ae0acf7f6eca | algorithm_contract.meaning | 30 chars | 1 declarations

root:64 contract.algorithm_requirement.input.segment_alignment.meaning

API segment alignment selector

## t_56b66f5ef995a12e | algorithm_contract.meaning | 36 chars | 1 declarations

root:46 contract.algorithm_requirement.input.b.meaning

Stored weight payload for each group

## t_58391e5359030431 | algorithm_contract.meaning | 35 chars | 1 declarations

root:90 contract.algorithm_requirement.compute[1].symbol.B_value.meaning

Scaled weight values for each group

## t_5a2833edce0ff7ce | algorithm_contract.meaning | 24 chars | 1 declarations

root:89 contract.algorithm_requirement.compute[1].symbol.A_value.meaning

Scaled activation values

## t_5dec2fb6407920ae | algorithm_contract.meaning | 30 chars | 1 declarations

root:56 contract.algorithm_requirement.input.mma_sm.meaning

API MMA configuration selector

## t_7bfdad4c9331c537 | algorithm_contract.meaning | 72 chars | 1 declarations

root:66 contract.algorithm_requirement.output.out.meaning

Grouped product written into out_buffer and returned as that same object

## t_7c65899306d5c466 | algorithm_contract.meaning | 99 chars | 1 declarations

root:51 contract.algorithm_requirement.input.scale_granularity_mnk.meaning

Python tuple of three integers specifying activation-row, weight-column and reduction scale widths.

## t_b1e31db38783565e | algorithm_contract.meaning | 52 chars | 1 declarations

root:55 contract.algorithm_requirement.input.scale_major_mode.meaning

Scale layout selector represented by a Python string

## t_c6e998425092276d | algorithm_contract.meaning | 49 chars | 1 declarations

root:47 contract.algorithm_requirement.input.a_scale.meaning

Activation scale for each row and reduction block

## t_e73564cb737c8350 | algorithm_contract.meaning | 25 chars | 1 declarations

root:45 contract.algorithm_requirement.input.a.meaning

Stored activation payload

## t_f7aab36a67a1de58 | algorithm_contract.meaning | 47 chars | 1 declarations

root:43 contract.algorithm_requirement.constant.ScaleK.meaning

Number of reduction elements sharing one scale.

## t_ffaa23a70cd6e130 | algorithm_contract.meaning | 50 chars | 1 declarations

root:42 contract.algorithm_requirement.constant.ScaleN.meaning

Number of weight output columns sharing one scale.

## t_0d3dc270bf299b3a | algorithm_contract.range | 23 chars | 1 declarations

root:62 contract.algorithm_requirement.input.out_dtype.range

Exactly torch.bfloat16.

## t_2d994ddfe94b90d1 | algorithm_contract.range | 13 chars | 1 declarations

root:55 contract.algorithm_requirement.input.scale_major_mode.range

Exactly 'MN'.

## t_510ea2ff67111383 | algorithm_contract.range | 28 chars | 1 declarations

root:54 contract.algorithm_requirement.input.scale_granularity_mnk.range

Exactly (1, ScaleN, ScaleK).

## t_9f2dad929d31e774 | algorithm_contract.range | 18 chars | 1 declarations

root:49 contract.algorithm_requirement.input.m_indptr.range

0 <= m_indptr <= M

## t_9fd78cba1611a823 | algorithm_contract.range | 17 chars | 1 declarations

root:63 contract.algorithm_requirement.input.backend.range

Exactly 'trtllm'.

## t_fc9ecae5db46e54f | algorithm_contract.range | 45 chars | 1 declarations

root:61 contract.algorithm_requirement.input.out_buffer.range

Any BF16 value, including NaN and infinities.

## t_3a8deb15073781b7 | contract.compute_name | 12 chars | 1 declarations

root:68 contract.algorithm_requirement.compute[0].name

Valid domain

## t_c3e757d4a56c2841 | contract.compute_name | 20 chars | 1 declarations

root:81 contract.algorithm_requirement.compute[1].name

Decode stored scales

## t_f60dc547f4910f50 | contract.compute_name | 36 chars | 1 declarations

root:91 contract.algorithm_requirement.compute[2].name

Grouped matrix product and writeback

## t_5af34fadb420a5ab | contract.tolerance | 524 chars | 1 declarations

root:106 contract.algorithm_requirement.golden_reference.tolerance.out

Source record: problem/metadata.json declares atol=0.01, rtol=0.05 and max_mismatch_fraction=0.0 for the full output view against the reference. Nonwrite input storage and storage outside the output view must remain unchanged. performance/CERTIFIED_BEST.json states that comparator/tolerances are unchanged, but the bound materials do not provide the comparison formula, relative-error basis, comparator arithmetic precision or NaN/Inf policy. This source record does not adopt the schema's default atol/rtol interpretation.

## t_20e05bd83a6bde6c | dispatch.md | 109 chars | 1 declarations

dispatch.md:23 dispatch.md

V0 与 wide 的源 TMEM drain 均存在行组寻址遗漏。索引中的完整乘积与分行设计依据乘积生产、TMEM 容量、尺度和写回，不能据路径覆盖认定源 kernel 已正确实现该行分工；该缺陷不产生新的选择分支。

## t_2304aaf0197a9461 | dispatch.md | 7 chars | 1 declarations

dispatch.md:19 dispatch.md

## 覆盖缺口

## t_32b044faa5d8348b | dispatch.md | 38 chars | 1 declarations

dispatch.md:3 dispatch.md

| 节点 | 直接父 | workload | 相对父差异 | 选择依据 |

## t_37414fad057b9a22 | dispatch.md | 231 chars | 1 declarations

dispatch.md:9 dispatch.md

| [wide_recovery](wide_recovery.yaml) | wide | 同一次请求中发生两次实际 wide 计算的恢复路径。 | 在单次 wide 设计上增加一个独立提交，分别绑定各次 F/G 特化及录制参数；同一 stream 顺序完成，两次都重算并覆盖输出，不共享局部资源或消费前次输出。 | UMMA 的 wide 路由发生图转直接或 G 转 F，实际 L 属于 [F,F]、[G,G]、[G,F]；未实际提交的尝试不占序列位置。 |

## t_5922cfeee4521bfa | dispatch.md | 61 chars | 1 declarations

dispatch.md:21 dispatch.md

上表覆盖正常构建中可进入实际计算的 V0、wide、sync 及其恢复提交序列；不把诊断宏强制启用的其他特化当作当前路径。

## t_7613329559610967 | dispatch.md | 200 chars | 1 declarations

dispatch.md:13 dispatch.md

V0 指正常构建的 `umma_gemm<0>`；F 指 `umma_wide<4,32768,0>`（w1f），G 指 `umma_wide<3,65536,1>`（w1g）。P 是实际调用的 UMMA 模块中 `sched_policy` 返回的选择值；其几何派生量、阈值、缓存命中和读取失败行为由 root 的选择定义说明。命中策略缓存时复用选择，不能按当次设备端分组内容重算 host 选择。

## t_7875020cff23f0cd | dispatch.md | 6 chars | 1 declarations

dispatch.md:1 dispatch.md

# 路径索引

## t_7c4b1099056d862b | dispatch.md | 201 chars | 1 declarations

dispatch.md:5 dispatch.md

| [root](root.yaml) | 无 | UMMA 默认选择域内的一次 V0 分组计算，含空组和尾行块。 | 代表设计：128×128 输出块，两个相邻 K=128 块成对推进；持久 CTA、三槽输入环、双槽 TMEM、有序块尺度 FMA；单/双发射参数化。 | Model 初始化选择 UMMA，sched_policy 返回 P=0，实际提交序列 L=[V0]；图回放或直接提交均可。 |

## t_7e6b6344446c3725 | dispatch.md | 268 chars | 1 declarations

dispatch.md:6 dispatch.md

| [mma_sync](mma_sync.yaml) | root | 初始化未获得 UMMA 引擎时的一次 mma.sync 分组计算。 | 输出块改为 64×128，逐 K=128 块计算；八个 warp 协作执行 cp.async、mma.sync 和尺度累加，使用三槽共享输入环；排空后复用输入存储暂存 BF16 输出；grid 上限改为 2*nSM。 | 初始化转入 sync 且构建加载成功，Model._fn 经 fp8gg_run 实际提交一次 fp8gg_bm64；UMMA 执行期报错不会自动切换到 sync。 |

## t_a163a22de4af92d5 | dispatch.md | 228 chars | 1 declarations

dispatch.md:11 dispatch.md

| [wide_recovery_full](wide_recovery_full.yaml) | wide_recovery_chain | 同一次请求完整经历 G、F 两级图转直接的四次实际 wide 计算。 | 相对三提交设计增加末次 F 直接提交；四次重算、输出覆盖、独立局部资源和逐次 stream 完成关系均显式保留。 | P=2，G 图调用、G 直接调用及 F 图调用报错后均已有实际计算提交，随后 F 直接提交，L=[G,G,F,F]。 |

## t_a776498afc0a484b | dispatch.md | 68 chars | 1 declarations

dispatch.md:17 dispatch.md

V0 的发射成员数和 wide 的写回批量在各自节点内参数化；恢复链的不同实际提交次数对应不同节点。精确条件和版本以各 YAML 为准。

## t_abb32fe04e5ebebf | dispatch.md | 221 chars | 1 declarations

dispatch.md:7 dispatch.md

| [v0_recovery](v0_recovery.yaml) | root | 同一次请求在 V0 图调用报错后又直接提交，且两次计算均已实际提交。 | 单次计算沿用 V0；改为两个独立计算提交及 worker/局部资源实例，按同一 stream 的完成关系接续；两次完整重算和输出覆盖均保留，后次不读取前次输出。 | UMMA 且 P=0，图调用失败后走直接入口，L=[V0,V0]；若报错的图调用没有提交计算，则仍归 root。 |

## t_db5d6f457203bfbd | dispatch.md | 115 chars | 1 declarations

dispatch.md:27 dispatch.md

共享 Contract 已保留外部比较器定义缺项：已知 atol、rtol 与零失配比例阈值，但比较公式、相对误差基准和非有限值策略没有完整来源。该缺口不改变路径划分，也不据认证记录或本索引宣称全域数值误差或 GPU 正确性保证。

## t_dd00d49c4f6d7611 | dispatch.md | 244 chars | 1 declarations

dispatch.md:10 dispatch.md

| [wide_recovery_chain](wide_recovery_chain.yaml) | wide_recovery | 同一次请求中发生三次实际 wide 计算的连续恢复路径。 | 相对双提交设计再增加一次完整 wide 提交及独立 worker/局部资源实例；保留三次覆盖以及逐次 stream 完成接续，各次特化独立绑定。 | P=2 且进入 G 到 F 的恢复链，实际 L 为 [G,G,F] 或 [G,F,F]；分别保留 G 或 F 图调用报错前已提交的计算。 |

## t_f50b7942d92bdd09 | dispatch.md | 240 chars | 1 declarations

dispatch.md:25 dispatch.md

host 的拒绝和失败边界仍存在：两个引擎都拒绝 N 或 K 不被 128 整除的输入，组数超出各自上限也报错；这些范围外输入没有通用回退。UMMA 的基础 A/B/输出 tensor map 构建失败、首次配置时任一特化共享内存预算不满足，以及 sync 的配置失败，都可能使请求无法形成成功输出。缩小 A 行高的辅助 map 失败只退回完整行高，不另立节点。UMMA 运行期非零错误经 Model 抛出，不触发初始化时的 sync 回退；无实际计算提交的失败不虚构空计算节点。

## t_fdcc44a5d1f1c2cb | dispatch.md | 303 chars | 1 declarations

dispatch.md:8 dispatch.md

| [wide](wide.yaml) | root | UMMA 宽列选择域内的一次 wide 计算，覆盖 w1f 和 w1g。 | 输出块改为 128×256，逐 K=128 块推进；320 线程、单发射角色、两组列 epilogue、寄存器尺度预取，两个 TMEM 槽各存一个块乘积。完整行块每批列组数 J=1（F）或 2（G）：写回批数 2/J、暂存 32*J KiB、输入环深 5-J；每批按等源读取、填充、发布、写回接续，尾块直接写有效行。 | UMMA 且 P 为 1 或 2，L=[F] 或 [G]；P=2 可因 g_w1g_off 或当次失败转 F，没有实际计算前缀的转换仍归此节点。 |

## t_ff234e4d38aeee1f | dispatch.md | 236 chars | 1 declarations

dispatch.md:15 dispatch.md

L 按源码调用顺序只列本次请求实际提交的计算；API 返回错误本身不证明已经提交，也不能抹去已提交前缀。V0 调用链最多包含图调用、直接调用两处；wide 调用链最多依次包含 G 图、G 直接、F 图、F 直接四处。恢复节点仅在相应实际序列成立时适用，不承诺出错执行一定完成或产生正确结果。绑定缓存、ctypes、图创建和图实例化本身不增设计算节点。纯录图不产生计算完成事件；外层图实际执行时按已录制的提交序列、host 选择及按值参数归类，设备读取仍使用执行时张量内容。

## t_030aacca5a8b7062 | dispatch.when | 2526 chars | 1 declarations

wide_recovery_full:10 meta_data.dispatch.when

Let U be the engine chosen by Model initialization. U is umma when the device-0 major capability queried by _Engines is 10 and the UMMA library builds and loads successfully; otherwise initialization tries the sync engine. A binding-module failure alone retains UMMA via ctypes. Let S be the device-0 multiprocessor count passed by that wrapper. Let P be the engine selector returned by sched_policy in the UMMA module actually invoked. Its cache key is (address(m_indptr), E, M, N, K). A cache hit reuses P, without deriving it again from current tensor contents. On a miss, the host attempts a device-to-host pointer-array copy and stream synchronization. P defaults to 0 if either fails or the copied boundaries are not nonnegative, nondecreasing and at most M. After a successful read, let h be the copied boundaries, A = count of integers 0 <= g < E with h[g+1] > h[g], T = sum over 0 <= g < E of ceil((h[g+1]-h[g])/128), B = K/128 and I = T*(N/256). Let W mean N is divisible by 256, A > 0, S > 0, and at least one of: (B >= 32 and ((I >= 4*S and T >= 4*A) or (B <= 48 and 2*I >= 5*S and T >= A))); (B >= 16 and I >= 8*S and (T >= 2*A or T == A)); (8 <= B <= 12 and I >= 10*S and T >= A); (8 <= B <= 20 and I >= 4*S and T >= 2*A). The new selector is P=0 when W is false, P=2 when W is true and B <= 20 and 2*T >= 3*A, and P=1 otherwise. The normal build sets sareg=0. Let L be this request's actual compute-kernel submissions, including any submission preceding an API-error fallback. Graph construction and instantiation alone add no computation to L. For a recorded request, use its recorded host selection and by-value arguments at replay; device reads still use the tensor contents at execution time. Let F denote umma_wide<4,32768,0> and G denote umma_wide<3,65536,1>. Select exactly when U is umma, P=2, and L is exactly [G,G,F,F]. These are four actual compute submissions in one request, ordered by the same stream. G starts while g_w1g_off is false. Its graph replay reports an error after submitting computation, then direct G submits and reports an error, setting g_w1g_off and continuing to F. F's graph replay also reports an error after submitting computation, then direct F submits the fourth computation and returns its status. Count an attempt in L only if it actually submitted computation; an API error cannot remove an already submitted prefix. Construction or instantiation without replay adds no computation. A pure recording describes this future execution, with no extra invocation at recording.

## t_3edf9c6658d30db6 | dispatch.when | 2473 chars | 1 declarations

wide_recovery_chain:10 meta_data.dispatch.when

Let U be the engine chosen by Model initialization. U is umma when the device-0 major capability queried by _Engines is 10 and the UMMA library builds and loads successfully; otherwise initialization tries the sync engine. A binding-module failure alone retains UMMA via ctypes. Let S be the device-0 multiprocessor count passed by that wrapper. Let P be the engine selector returned by sched_policy in the UMMA module actually invoked. Its cache key is (address(m_indptr), E, M, N, K). A cache hit reuses P, without deriving it again from current tensor contents. On a miss, the host attempts a device-to-host pointer-array copy and stream synchronization. P defaults to 0 if either fails or the copied boundaries are not nonnegative, nondecreasing and at most M. After a successful read, let h be the copied boundaries, A = count of integers 0 <= g < E with h[g+1] > h[g], T = sum over 0 <= g < E of ceil((h[g+1]-h[g])/128), B = K/128 and I = T*(N/256). Let W mean N is divisible by 256, A > 0, S > 0, and at least one of: (B >= 32 and ((I >= 4*S and T >= 4*A) or (B <= 48 and 2*I >= 5*S and T >= A))); (B >= 16 and I >= 8*S and (T >= 2*A or T == A)); (8 <= B <= 12 and I >= 10*S and T >= A); (8 <= B <= 20 and I >= 4*S and T >= 2*A). The new selector is P=0 when W is false, P=2 when W is true and B <= 20 and 2*T >= 3*A, and P=1 otherwise. The normal build sets sareg=0. Let L be this request's actual compute-kernel submissions, including any submission preceding an API-error fallback. Graph construction and instantiation alone add no computation to L. For a recorded request, use its recorded host selection and by-value arguments at replay; device reads still use the tensor contents at execution time. Let F denote umma_wide<4,32768,0> and G denote umma_wide<3,65536,1>. Select exactly when U is umma, P=2, and L is exactly [G,G,F] or [G,F,F]. These are three actual compute submissions in one request, ordered by the same stream. G is attempted while g_w1g_off is false. A failed G graph replay disables that graph path and continues to direct G; a failed direct G attempt sets g_w1g_off and continues to F. A failed F graph replay disables its graph path and continues to direct F. Count an attempt in L only if it actually submitted computation; API failure cannot remove an already submitted prefix. Construction or instantiation without replay adds no computation. A pure recording describes this future execution, with no additional invocation at recording.

## t_414f8bf4bded40ac | dispatch.when | 968 chars | 1 declarations

mma_sync:10 meta_data.dispatch.when

Let U be the engine bound by Model initialization. The wrapper queries capability and multiprocessor count on device 0. U is sync exactly when UMMA was not selected (the queried major capability is not 10, or the UMMA library failed to build or load) and the sync library builds and loads successfully. Failure of the optional UMMA binding alone retains UMMA through ctypes. Let L be this request's actual compute-kernel submissions. Select exactly when U is sync and L consists of one fp8gg_bm64 invocation submitted through Model.forward and fp8gg_run. The sync path has no execution-time engine fallback: a nonzero entry return code raises an error. Initialization or prelaunch failure does not create a compute invocation. Pure graph recording describes a future execution; it does not execute the recorded work. For a recorded request, use its recorded host choice, tensor addresses and by-value parameters; device loads use the tensor contents at execution time.

## t_54b4d0a1b1e78de2 | dispatch.when | 2283 chars | 1 declarations

v0_recovery:9 meta_data.dispatch.when

Let U be the engine chosen by Model initialization. U is umma when the device-0 major capability queried by _Engines is 10 and the UMMA library builds and loads successfully; otherwise initialization tries the sync engine. A binding-module failure alone retains UMMA via ctypes. Let S be the device-0 multiprocessor count passed by that wrapper. Let P be the engine selector returned by sched_policy in the UMMA module actually invoked. Its cache key is (address(m_indptr), E, M, N, K). A cache hit reuses P, without deriving it again from current tensor contents. On a miss, the host attempts a device-to-host pointer-array copy and stream synchronization. P defaults to 0 if either fails or the copied boundaries are not nonnegative, nondecreasing and at most M. After a successful read, let h be the copied boundaries, A = count of integers 0 <= g < E with h[g+1] > h[g], T = sum over 0 <= g < E of ceil((h[g+1]-h[g])/128), B = K/128 and I = T*(N/256). Let W mean N is divisible by 256, A > 0, S > 0, and at least one of: (B >= 32 and ((I >= 4*S and T >= 4*A) or (B <= 48 and 2*I >= 5*S and T >= A))); (B >= 16 and I >= 8*S and (T >= 2*A or T == A)); (8 <= B <= 12 and I >= 10*S and T >= A); (8 <= B <= 20 and I >= 4*S and T >= 2*A). The new selector is P=0 when W is false, P=2 when W is true and B <= 20 and 2*T >= 3*A, and P=1 otherwise. The normal build sets sareg=0. Let L be this request's actual compute-kernel submissions, including any submission preceding an API-error fallback. Graph construction and instantiation alone add no computation to L. For a recorded request, use its recorded host selection and by-value arguments at replay; device reads still use the tensor contents at execution time. Select exactly when U is umma, P=0, and L consists of two invocations of umma_gemm<0>: a graph replay whose API call reports an error after actually submitting its kernel, followed by the direct recovery submission from the same fp8umma_run call on the same stream. This includes both a cached graph and a newly instantiated graph. If the failed graph call submitted no compute kernel, the one-submission request belongs to root. API failure alone does not establish either submission or completion. A pure recording describes the future execution and adds no invocation.

## t_850f58c14f6294f1 | dispatch.when | 2464 chars | 1 declarations

wide_recovery:10 meta_data.dispatch.when

Let U be the engine chosen by Model initialization. U is umma when the device-0 major capability queried by _Engines is 10 and the UMMA library builds and loads successfully; otherwise initialization tries the sync engine. A binding-module failure alone retains UMMA via ctypes. Let S be the device-0 multiprocessor count passed by that wrapper. Let P be the engine selector returned by sched_policy in the UMMA module actually invoked. Its cache key is (address(m_indptr), E, M, N, K). A cache hit reuses P, without deriving it again from current tensor contents. On a miss, the host attempts a device-to-host pointer-array copy and stream synchronization. P defaults to 0 if either fails or the copied boundaries are not nonnegative, nondecreasing and at most M. After a successful read, let h be the copied boundaries, A = count of integers 0 <= g < E with h[g+1] > h[g], T = sum over 0 <= g < E of ceil((h[g+1]-h[g])/128), B = K/128 and I = T*(N/256). Let W mean N is divisible by 256, A > 0, S > 0, and at least one of: (B >= 32 and ((I >= 4*S and T >= 4*A) or (B <= 48 and 2*I >= 5*S and T >= A))); (B >= 16 and I >= 8*S and (T >= 2*A or T == A)); (8 <= B <= 12 and I >= 10*S and T >= A); (8 <= B <= 20 and I >= 4*S and T >= 2*A). The new selector is P=0 when W is false, P=2 when W is true and B <= 20 and 2*T >= 3*A, and P=1 otherwise. The normal build sets sareg=0. Let L be this request's actual compute-kernel submissions, including any submission preceding an API-error fallback. Graph construction and instantiation alone add no computation to L. For a recorded request, use its recorded host selection and by-value arguments at replay; device reads still use the tensor contents at execution time. Let F denote umma_wide<4,32768,0> and G denote umma_wide<3,65536,1>. Select exactly when U is umma, P is 1 or 2, and L is exactly [F,F], [G,G] or [G,F]. These are two actual compute submissions in one request, ordered by the same stream. P=1 attempts F; P=2 attempts G unless the permanent g_w1g_off flag redirects it to F. A failed graph replay disables that engine's graph path and continues to its direct attempt. A failed direct G attempt sets g_w1g_off and continues to F. Count an attempt in L only if it actually submitted computation; API failure cannot remove an already submitted prefix. Construction or instantiation without replay adds no computation. A pure recording describes this future execution, with no extra invocation at recording.

## t_97a89826281a0aa3 | dispatch.when | 1922 chars | 1 declarations

root:7 meta_data.dispatch.when

Let U be the engine chosen by Model initialization. U is umma when the device-0 major capability queried by _Engines is 10 and the UMMA library builds and loads successfully; otherwise initialization tries the sync engine. A binding-module failure alone retains UMMA via ctypes. Let S be the device-0 multiprocessor count passed by that wrapper. Let P be the engine selector returned by sched_policy in the UMMA module actually invoked. Its cache key is (address(m_indptr), E, M, N, K). A cache hit reuses P, without deriving it again from current tensor contents. On a miss, the host attempts a device-to-host pointer-array copy and stream synchronization. P defaults to 0 if either fails or the copied boundaries are not nonnegative, nondecreasing and at most M. After a successful read, let h be the copied boundaries, A = count of integers 0 <= g < E with h[g+1] > h[g], T = sum over 0 <= g < E of ceil((h[g+1]-h[g])/128), B = K/128 and I = T*(N/256). Let W mean N is divisible by 256, A > 0, S > 0, and at least one of: (B >= 32 and ((I >= 4*S and T >= 4*A) or (B <= 48 and 2*I >= 5*S and T >= A))); (B >= 16 and I >= 8*S and (T >= 2*A or T == A)); (8 <= B <= 12 and I >= 10*S and T >= A); (8 <= B <= 20 and I >= 4*S and T >= 2*A). The new selector is P=0 when W is false, P=2 when W is true and B <= 20 and 2*T >= 3*A, and P=1 otherwise. The normal build sets sareg=0. Let L be this request's actual compute-kernel submissions, including any submission preceding an API-error fallback. Graph construction and instantiation alone add no computation to L. For a recorded request, use its recorded host selection and by-value arguments at replay; device reads still use the tensor contents at execution time. Select exactly when U is umma, P=0, and L consists of one invocation of umma_gemm<0>, through graph replay or direct launch. A pure recording describes that future execution and is not an additional invocation.

## t_a29b153f18b89e37 | dispatch.when | 2471 chars | 1 declarations

wide:10 meta_data.dispatch.when

Let U be the engine chosen by Model initialization. U is umma when the device-0 major capability queried by _Engines is 10 and the UMMA library builds and loads successfully; otherwise initialization tries the sync engine. A binding-module failure alone retains UMMA via ctypes. Let S be the device-0 multiprocessor count passed by that wrapper. Let P be the engine selector returned by sched_policy in the UMMA module actually invoked. Its cache key is (address(m_indptr), E, M, N, K). A cache hit reuses P, without deriving it again from current tensor contents. On a miss, the host attempts a device-to-host pointer-array copy and stream synchronization. P defaults to 0 if either fails or the copied boundaries are not nonnegative, nondecreasing and at most M. After a successful read, let h be the copied boundaries, A = count of integers 0 <= g < E with h[g+1] > h[g], T = sum over 0 <= g < E of ceil((h[g+1]-h[g])/128), B = K/128 and I = T*(N/256). Let W mean N is divisible by 256, A > 0, S > 0, and at least one of: (B >= 32 and ((I >= 4*S and T >= 4*A) or (B <= 48 and 2*I >= 5*S and T >= A))); (B >= 16 and I >= 8*S and (T >= 2*A or T == A)); (8 <= B <= 12 and I >= 10*S and T >= A); (8 <= B <= 20 and I >= 4*S and T >= 2*A). The new selector is P=0 when W is false, P=2 when W is true and B <= 20 and 2*T >= 3*A, and P=1 otherwise. The normal build sets sareg=0. Let L be this request's actual compute-kernel submissions, including any submission preceding an API-error fallback. Graph construction and instantiation alone add no computation to L. For a recorded request, use its recorded host selection and by-value arguments at replay; device reads still use the tensor contents at execution time. Let F denote umma_wide<4,32768,0> and G denote umma_wide<3,65536,1>. Select exactly when U is umma, P is 1 or 2, and L is [F] or [G]. P=1 attempts F. P=2 attempts G unless the module's permanent g_w1g_off flag is set; a failed direct G attempt also sets that flag and proceeds to F. Such redirection belongs here only when L still has exactly one compute submission. A failed graph or direct attempt that has actually submitted computation remains in L; multiple-submission paths do not satisfy this condition. Graph construction failure without submission may proceed directly without adding to L. Each engine's graph-disable flag is permanent after graph failure. A pure recording describes the selected future execution and is not an additional invocation.

## t_236df50fcb96a6d5 | math.arithmetic | 155 chars | 1 declarations

mma_sync:46 design.math_reshape.rewrite[0].arithmetic.accumulate

Each Dot starts with a zero FP32 accumulator and absorbs four consecutive 32-element reduction slices in increasing order, using FP8 multiplication inputs.

## t_831393721c89008a | math.arithmetic | 198 chars | 6 declarations

root:152 design.math_reshape.rewrite[0].arithmetic.accumulate; v0_recovery:77 design.math_reshape.rewrite[0].arithmetic.accumulate; wide:73 design.math_reshape.rewrite[0].arithmetic.accumulate; wide_recovery:76 design.math_reshape.rewrite[0].arithmetic.accumulate; wide_recovery_chain:76 design.math_reshape.rewrite[0].arithmetic.accumulate; wide_recovery_full:77 design.math_reshape.rewrite[0].arithmetic.accumulate

Each Dot uses four consecutive 32-element reduction slices, in increasing order, with FP8 inputs and a FP32 accumulator initialized by the first slice. Subsequent slices accumulate into that result.

## t_018ecdd3a34258da | math.ledger | 137 chars | 1 declarations

wide_recovery:132 design.math_reshape.ledger.vector[0].change

M*K+E*N*K FP32 operand multiplications -> 2*Mp*Tn*KB*(Bn/ScaleN) scale multiplications and 2*Mp*N*KB FP32 fused multiply-adds per request

## t_05a88599912e08d7 | math.ledger | 16 chars | 3 declarations

wide_recovery:146 design.math_reshape.ledger.traffic[0].item; wide_recovery_chain:146 design.math_reshape.ledger.traffic[0].item; wide_recovery_full:147 design.math_reshape.ledger.traffic[0].item

Output writeback

## t_0979e6342edfb94a | math.ledger | 17 chars | 7 declarations

mma_sync:79 design.math_reshape.ledger.vector[0].item; root:189 design.math_reshape.ledger.vector[0].item; v0_recovery:117 design.math_reshape.ledger.vector[0].item; wide:112 design.math_reshape.ledger.vector[0].item; wide_recovery:131 design.math_reshape.ledger.vector[0].item; wide_recovery_chain:131 design.math_reshape.ledger.vector[0].item; wide_recovery_full:132 design.math_reshape.ledger.vector[0].item

Scale application

## t_1c01c708f4510b87 | math.ledger | 137 chars | 1 declarations

wide_recovery_full:133 design.math_reshape.ledger.vector[0].change

M*K+E*N*K FP32 operand multiplications -> 4*Mp*Tn*KB*(Bn/ScaleN) scale multiplications and 4*Mp*N*KB FP32 fused multiply-adds per request

## t_1dcc709e39f405e6 | math.ledger | 100 chars | 4 declarations

mma_sync:82 design.math_reshape.ledger.footprint[0].change; root:191 design.math_reshape.ledger.footprint[0].change; v0_recovery:121 design.math_reshape.ledger.footprint[0].change; wide:118 design.math_reshape.ledger.footprint[0].change

4*(M*K+E*N*K) bytes of FP32 operand materialization -> 0 bytes; each live Dot and F is 4*Bm*Bn bytes

## t_21154792abc7f0d7 | math.ledger | 86 chars | 3 declarations

mma_sync:76 design.math_reshape.ledger.matmul[0].change; root:187 design.math_reshape.ledger.matmul[0].change; wide:109 design.math_reshape.ledger.matmul[0].change

M*N*K FP32-input MACs -> Mp*N*K FP8-input MACs; (Mp-M)*N*K MACs compute discarded rows

## t_22c119277d05a3c4 | math.ledger | 121 chars | 1 declarations

wide:113 design.math_reshape.ledger.vector[0].change

M*K+E*N*K FP32 operand multiplications -> Mp*Tn*KB*(Bn/ScaleN) scale multiplications and Mp*N*KB FP32 fused multiply-adds

## t_474c0db4da747bbf | math.ledger | 137 chars | 1 declarations

wide_recovery_chain:132 design.math_reshape.ledger.vector[0].change

M*K+E*N*K FP32 operand multiplications -> 3*Mp*Tn*KB*(Bn/ScaleN) scale multiplications and 3*Mp*N*KB FP32 fused multiply-adds per request

## t_4782b36b669b1d6e | math.ledger | 97 chars | 1 declarations

wide_recovery:147 design.math_reshape.ledger.traffic[0].change

2*M*N -> 4*M*N logical output bytes written per request; neither evaluation reads previous output

## t_4c7e98c4062d417b | math.ledger | 102 chars | 1 declarations

wide_recovery_full:128 design.math_reshape.ledger.matmul[0].change

M*N*K FP32-input MACs -> 4*Mp*N*K FP8-input MACs per request; 4*(Mp-M)*N*K MACs compute discarded rows

## t_56d9c0133e02dbc5 | math.ledger | 102 chars | 1 declarations

wide_recovery:127 design.math_reshape.ledger.matmul[0].change

M*N*K FP32-input MACs -> 2*Mp*N*K FP8-input MACs per request; 2*(Mp-M)*N*K MACs compute discarded rows

## t_5e31252af2089088 | math.ledger | 9 chars | 21 declarations

mma_sync:77 design.math_reshape.ledger.matmul[0].by[0]; mma_sync:83 design.math_reshape.ledger.footprint[0].by[0]; mma_sync:86 design.math_reshape.ledger.error[0].by[0]; root:187 design.math_reshape.ledger.matmul[0].by[0]; root:191 design.math_reshape.ledger.footprint[0].by[0]; root:193 design.math_reshape.ledger.error[0].by[0]; v0_recovery:115 design.math_reshape.ledger.matmul[0].by[0]; v0_recovery:121 design.math_reshape.ledger.footprint[0].by[0]; v0_recovery:123 design.math_reshape.ledger.error[0].by[0]; wide:110 design.math_reshape.ledger.matmul[0].by[0]; wide:119 design.math_reshape.ledger.footprint[0].by[0]; wide:124 design.math_reshape.ledger.error[0].by[0]; wide_recovery:129 design.math_reshape.ledger.matmul[0].by[0]; wide_recovery:139 design.math_reshape.ledger.footprint[0].by[0]; wide_recovery:144 design.math_reshape.ledger.error[0].by[0]; wide_recovery_chain:129 design.math_reshape.ledger.matmul[0].by[0]; wide_recovery_chain:139 design.math_reshape.ledger.footprint[0].by[0]; wide_recovery_chain:144 design.math_reshape.ledger.error[0].by[0]; wide_recovery_full:130 design.math_reshape.ledger.matmul[0].by[0]; wide_recovery_full:140 design.math_reshape.ledger.footprint[0].by[0]; wide_recovery_full:145 design.math_reshape.ledger.error[0].by[0]

block_dot

## t_5f9c55d6bd25b5a5 | math.ledger | 92 chars | 1 declarations

wide_recovery_full:148 design.math_reshape.ledger.traffic[0].change

2*M*N -> 8*M*N logical output bytes written per request; no evaluation reads previous output

## t_605c264417ae5823 | math.ledger | 113 chars | 1 declarations

v0_recovery:117 design.math_reshape.ledger.vector[0].change

M*K+E*N*K FP32 operand multiplications -> 2*Mp*Tn*KB scale multiplications and 2*Mp*N*KB FP32 fused multiply-adds

## t_7578a9416456e996 | math.ledger | 121 chars | 3 declarations

wide_recovery:137 design.math_reshape.ledger.footprint[0].change; wide_recovery_chain:137 design.math_reshape.ledger.footprint[0].change; wide_recovery_full:138 design.math_reshape.ledger.footprint[0].change

4*(M*K+E*N*K) bytes of FP32 operand materialization -> 0 bytes; each live Dot and F is 4*Bm*Bn bytes and belongs to one q

## t_758c45960520bdef | math.ledger | 198 chars | 1 declarations

wide_recovery_chain:123 design.math_reshape.ledger.baseline

One direct Contract evaluation with materialized scaled FP32 A_value and B_value, followed by their grouped FP32 matrix product. The design counts cover all three evaluations, including padded rows.

## t_7996c77f8a5040a4 | math.ledger | 167 chars | 6 declarations

root:193 design.math_reshape.ledger.error[0].change; v0_recovery:123 design.math_reshape.ledger.error[0].change; wide:122 design.math_reshape.ledger.error[0].change; wide_recovery:142 design.math_reshape.ledger.error[0].change; wide_recovery_chain:142 design.math_reshape.ledger.error[0].change; wide_recovery_full:143 design.math_reshape.ledger.error[0].change

Per-operand scaling before the product -> rounded scale products applied after each block dot, then ordered FP32 FMA; no source-provided bound on final reference error

## t_8857a90d75d0be7c | math.ledger | 183 chars | 1 declarations

mma_sync:73 design.math_reshape.ledger.baseline

Direct Contract evaluation with materialized scaled FP32 A_value and B_value, followed by their grouped FP32 matrix product. Counts cover one invocation and include evaluated padding.

## t_8fddb632b8567e4e | math.ledger | 90 chars | 1 declarations

v0_recovery:115 design.math_reshape.ledger.matmul[0].change

M*N*K FP32-input MACs -> 2*Mp*N*K FP8-input MACs; 2*(Mp-M)*N*K MACs compute discarded rows

## t_900d6c66a534d62f | math.ledger | 94 chars | 1 declarations

v0_recovery:119 design.math_reshape.ledger.traffic[0].change

2*M*N -> 4*M*N bytes written to the same output view; the first complete result is overwritten

## t_9724aade0ba7fb11 | math.ledger | 10 chars | 25 declarations

mma_sync:80 design.math_reshape.ledger.vector[0].by[0]; mma_sync:83 design.math_reshape.ledger.footprint[0].by[1]; mma_sync:86 design.math_reshape.ledger.error[0].by[1]; root:189 design.math_reshape.ledger.vector[0].by[0]; root:191 design.math_reshape.ledger.footprint[0].by[1]; root:193 design.math_reshape.ledger.error[0].by[1]; v0_recovery:117 design.math_reshape.ledger.vector[0].by[0]; v0_recovery:119 design.math_reshape.ledger.traffic[0].by[0]; v0_recovery:121 design.math_reshape.ledger.footprint[0].by[1]; v0_recovery:123 design.math_reshape.ledger.error[0].by[1]; wide:115 design.math_reshape.ledger.vector[0].by[0]; wide:119 design.math_reshape.ledger.footprint[0].by[1]; wide:124 design.math_reshape.ledger.error[0].by[1]; wide_recovery:134 design.math_reshape.ledger.vector[0].by[0]; wide_recovery:139 design.math_reshape.ledger.footprint[0].by[1]; wide_recovery:144 design.math_reshape.ledger.error[0].by[1]; wide_recovery:148 design.math_reshape.ledger.traffic[0].by[0]; wide_recovery_chain:134 design.math_reshape.ledger.vector[0].by[0]; wide_recovery_chain:139 design.math_reshape.ledger.footprint[0].by[1]; wide_recovery_chain:144 design.math_reshape.ledger.error[0].by[1]; wide_recovery_chain:148 design.math_reshape.ledger.traffic[0].by[0]; wide_recovery_full:135 design.math_reshape.ledger.vector[0].by[0]; wide_recovery_full:140 design.math_reshape.ledger.footprint[0].by[1]; wide_recovery_full:145 design.math_reshape.ledger.error[0].by[1]; wide_recovery_full:149 design.math_reshape.ledger.traffic[0].by[0]

scaled_sum

## t_9dca23e8103d7f35 | math.ledger | 14 chars | 7 declarations

mma_sync:76 design.math_reshape.ledger.matmul[0].item; root:187 design.math_reshape.ledger.matmul[0].item; v0_recovery:115 design.math_reshape.ledger.matmul[0].item; wide:109 design.math_reshape.ledger.matmul[0].item; wide_recovery:126 design.math_reshape.ledger.matmul[0].item; wide_recovery_chain:126 design.math_reshape.ledger.matmul[0].item; wide_recovery_full:127 design.math_reshape.ledger.matmul[0].item

Dense products

## t_9fac3f4716e62147 | math.ledger | 13 chars | 7 declarations

mma_sync:85 design.math_reshape.ledger.error[0].item; root:193 design.math_reshape.ledger.error[0].item; v0_recovery:123 design.math_reshape.ledger.error[0].item; wide:121 design.math_reshape.ledger.error[0].item; wide_recovery:141 design.math_reshape.ledger.error[0].item; wide_recovery_chain:141 design.math_reshape.ledger.error[0].item; wide_recovery_full:142 design.math_reshape.ledger.error[0].item

Reassociation

## t_a12bd5286b2f32bb | math.ledger | 221 chars | 1 declarations

v0_recovery:110 design.math_reshape.ledger.baseline

Direct Contract evaluation with materialized scaled FP32 A_value and B_value, followed by their grouped FP32 matrix product, evaluated once. Post-rewrite counts cover this request's two evaluations, including padded rows.

## t_a86cc7561f7997b9 | math.ledger | 15 chars | 7 declarations

mma_sync:82 design.math_reshape.ledger.footprint[0].item; root:191 design.math_reshape.ledger.footprint[0].item; v0_recovery:121 design.math_reshape.ledger.footprint[0].item; wide:117 design.math_reshape.ledger.footprint[0].item; wide_recovery:136 design.math_reshape.ledger.footprint[0].item; wide_recovery_chain:136 design.math_reshape.ledger.footprint[0].item; wide_recovery_full:137 design.math_reshape.ledger.footprint[0].item

Scaled operands

## t_b5cf99aad4583d89 | math.ledger | 183 chars | 2 declarations

root:182 design.math_reshape.ledger.baseline; wide:106 design.math_reshape.ledger.baseline

Direct Contract evaluation with materialized scaled FP32 A_value and B_value, followed by their grouped FP32 matrix product. Counts below cover one invocation and include padded rows.

## t_b8893fbd55afcd2a | math.ledger | 138 chars | 1 declarations

mma_sync:85 design.math_reshape.ledger.error[0].change

Per-operand scaling -> rounded scale products after FP8 block dots and ordered FP32 FMA; no source-provided bound on final reference error

## t_b992c9fc857b3de5 | math.ledger | 13 chars | 1 declarations

v0_recovery:119 design.math_reshape.ledger.traffic[0].item

Output writes

## t_bac043e98d0d64e1 | math.ledger | 92 chars | 1 declarations

wide_recovery_chain:147 design.math_reshape.ledger.traffic[0].change

2*M*N -> 6*M*N logical output bytes written per request; no evaluation reads previous output

## t_bc61629c6287f783 | math.ledger | 193 chars | 1 declarations

wide_recovery:123 design.math_reshape.ledger.baseline

One direct Contract evaluation with materialized scaled FP32 A_value and B_value, followed by their grouped FP32 matrix product. The design counts cover both evaluations, including padded rows.

## t_d59bfa45f5aa2d92 | math.ledger | 102 chars | 1 declarations

wide_recovery_chain:127 design.math_reshape.ledger.matmul[0].change

M*N*K FP32-input MACs -> 3*Mp*N*K FP8-input MACs per request; 3*(Mp-M)*N*K MACs compute discarded rows

## t_d96125499c25e85e | math.ledger | 109 chars | 2 declarations

mma_sync:79 design.math_reshape.ledger.vector[0].change; root:189 design.math_reshape.ledger.vector[0].change

M*K+E*N*K FP32 operand multiplications -> Mp*Tn*KB scale multiplications and Mp*N*KB FP32 fused multiply-adds

## t_f076418a0c0d3919 | math.ledger | 197 chars | 1 declarations

wide_recovery_full:124 design.math_reshape.ledger.baseline

One direct Contract evaluation with materialized scaled FP32 A_value and B_value, followed by their grouped FP32 matrix product. The design counts cover all four evaluations, including padded rows.

## t_0d240e644004f2f9 | math.math | 824 chars | 1 declarations

v0_recovery:90 design.math_reshape.rewrite[1].math

For each evaluation r and nonempty output worktile, F(-1)[u,v] = 0. For kb=0,...,KB-1 in ascending order: sa(kb,u) = a_scale[kb,r0+u] if r0+u < M, otherwise 0. sb(kb) = b_scale[g,kb,y]. Scale(kb)[u] = sa(kb,u)*sb(kb). F(kb)[u,v] = fma32(Scale(kb)[u], Dot(kb)[u,v], F(kb-1)[u,v]). The two blocks kb=2*q and kb=2*q+1, 0 <= q < Q, have independent Dot values and scales; F absorbs the even block before the odd block. For 0 <= u < V and 0 <= v < Bn: out[r0+u,c0+v] = bf16(F(KB-1)[u,v]). Discard F for u >= V. Both evaluations overwrite the same complete out_buffer view, with evaluation 1 replacing evaluation 0. Neither reads incoming output values or the other evaluation's results. After evaluation 1, return out with the object identity and other side effects in Contract's Grouped matrix product and writeback requirement.

## t_355c43164785cf74 | math.math | 568 chars | 1 declarations

mma_sync:33 design.math_reshape.rewrite[0].math

Let KB=K/ScaleK and Tn=N/Bn. For each group g, let Rg=m_indptr[g+1]-m_indptr[g], Tm(g)=ceil_div(Rg,Bm). For 0 <= x < Tm(g), 0 <= y < Tn, define r0=m_indptr[g]+x*Bm, c0=y*Bn and V=min(Bm,Rg-x*Bm). For 0 <= kb < KB, 0 <= u < Bm, 0 <= v < Bn, 0 <= z < ScaleK: A_tile(kb)[u,z] = a[r0+u,kb*ScaleK+z] if u<V, otherwise 0. B_tile(kb)[v,z] = b[g,c0+v,kb*ScaleK+z]. Dot(kb) = mm32(A_tile(kb),B_tile(kb)^T). Every Bm-row block is evaluated, including rows u>=V whose results are discarded. Empty groups have no instances. Let Mp=Bm*sum_g Tm(g) be the total evaluated row extent.

## t_5ee6ddf242a403d2 | math.math | 829 chars | 1 declarations

wide_recovery_chain:59 design.math_reshape.rewrite[0].math

Let q in {0,1,2} index three complete evaluations. No evaluation reads another evaluation's result. The notation suppresses q on local values. Let KB = K/ScaleK and Tn = N/Bn. For each group g, let Rg = m_indptr[g+1]-m_indptr[g], Tm(g) = ceil_div(Rg,Bm). For 0 <= x < Tm(g), 0 <= y < Tn, define r0 = m_indptr[g]+x*Bm, c0 = y*Bn and V = min(Bm,Rg-x*Bm). For 0 <= kb < KB, 0 <= u < Bm, 0 <= v < Bn, 0 <= z < ScaleK: A_tile(kb)[u,z] = a[r0+u,kb*ScaleK+z] if u < H and r0+u < M; = 0 if u < H and r0+u >= M; = the retained, unspecified FP8 row payload if u >= H. B_tile(kb)[v,z] = b[g,c0+v,kb*ScaleK+z]. Dot(kb) = mm32(A_tile(kb), B_tile(kb)^T). All Bm rows are evaluated, including u >= V; no result from those rows is used in an output. Empty groups have no such instances. Let Mp = Bm*sum_g Tm(g) be the total evaluated row extent.

## t_6cd9c7a2adeea1b0 | math.math | 929 chars | 1 declarations

v0_recovery:58 design.math_reshape.rewrite[0].math

For each evaluation r in {0,1}, independently compute the following values from the request inputs. Worktile values are scoped by (r,g,x,y); no computed value is reused between evaluations. Let KB = K/ScaleK, Q = KB/2 and Tn = N/Bn. KB is even throughout the Contract domain. For each group g, let Rg = m_indptr[g+1]-m_indptr[g], Tm(g) = ceil_div(Rg,Bm). For 0 <= x < Tm(g), 0 <= y < Tn, define r0 = m_indptr[g]+x*Bm, c0 = y*Bn and V = min(Bm,Rg-x*Bm). For 0 <= kb < KB, 0 <= u < Bm, 0 <= v < Bn, 0 <= z < ScaleK: A_tile(kb)[u,z] = a[r0+u,kb*ScaleK+z] if u < H and r0+u < M; = 0 if u < H and r0+u >= M; = the retained, unspecified FP8 row payload if u >= H. B_tile(kb)[v,z] = b[g,c0+v,kb*ScaleK+z]. Dot(kb) = mm32(A_tile(kb), B_tile(kb)^T). All Bm rows are evaluated, including u >= V; no result from those rows is used in an output. Empty groups have no such instances. Let Mp = Bm*sum_g Tm(g) be the total evaluated row extent.

## t_800cb3a9b15facd4 | math.math | 830 chars | 1 declarations

wide_recovery_full:60 design.math_reshape.rewrite[0].math

Let q in {0,1,2,3} index four complete evaluations. No evaluation reads another evaluation's result. The notation suppresses q on local values. Let KB = K/ScaleK and Tn = N/Bn. For each group g, let Rg = m_indptr[g+1]-m_indptr[g], Tm(g) = ceil_div(Rg,Bm). For 0 <= x < Tm(g), 0 <= y < Tn, define r0 = m_indptr[g]+x*Bm, c0 = y*Bn and V = min(Bm,Rg-x*Bm). For 0 <= kb < KB, 0 <= u < Bm, 0 <= v < Bn, 0 <= z < ScaleK: A_tile(kb)[u,z] = a[r0+u,kb*ScaleK+z] if u < H and r0+u < M; = 0 if u < H and r0+u >= M; = the retained, unspecified FP8 row payload if u >= H. B_tile(kb)[v,z] = b[g,c0+v,kb*ScaleK+z]. Dot(kb) = mm32(A_tile(kb), B_tile(kb)^T). All Bm rows are evaluated, including u >= V; no result from those rows is used in an output. Empty groups have no such instances. Let Mp = Bm*sum_g Tm(g) be the total evaluated row extent.

## t_86adf91408b9a3c3 | math.math | 697 chars | 1 declarations

wide_recovery_chain:97 design.math_reshape.rewrite[1].math

For each q and each nonempty output worktile, F(-1)[u,v] = 0. For kb=0,...,KB-1 in ascending order and 0 <= h < Bn/ScaleN: sa(kb,u) = a_scale[kb,r0+u] if r0+u < M, otherwise 0. sb(kb,h) = b_scale[g,kb,y*(Bn/ScaleN)+h]. Scale(kb)[u,h] = sa(kb,u)*sb(kb,h). For 0 <= u < Bm and 0 <= v < Bn: F(kb)[u,v] = fma32(Scale(kb)[u,floor(v/ScaleN)], Dot(kb)[u,v], F(kb-1)[u,v]). For 0 <= u < V and 0 <= v < Bn: out[r0+u,c0+v] = bf16(F(KB-1)[u,v]). Discard F for u >= V. Each evaluation overwrites the entire output view without reading its prior contents. Evaluation 2 supplies the final output. The output identity and all other side effects follow Contract's Grouped matrix product and writeback requirement.

## t_970f4cb3059d86d6 | math.math | 697 chars | 1 declarations

wide_recovery:97 design.math_reshape.rewrite[1].math

For each q and each nonempty output worktile, F(-1)[u,v] = 0. For kb=0,...,KB-1 in ascending order and 0 <= h < Bn/ScaleN: sa(kb,u) = a_scale[kb,r0+u] if r0+u < M, otherwise 0. sb(kb,h) = b_scale[g,kb,y*(Bn/ScaleN)+h]. Scale(kb)[u,h] = sa(kb,u)*sb(kb,h). For 0 <= u < Bm and 0 <= v < Bn: F(kb)[u,v] = fma32(Scale(kb)[u,floor(v/ScaleN)], Dot(kb)[u,v], F(kb-1)[u,v]). For 0 <= u < V and 0 <= v < Bn: out[r0+u,c0+v] = bf16(F(KB-1)[u,v]). Discard F for u >= V. Each evaluation overwrites the entire output view without reading its prior contents. Evaluation 1 supplies the final output. The output identity and all other side effects follow Contract's Grouped matrix product and writeback requirement.

## t_9b8c50414c23133c | math.math | 560 chars | 1 declarations

wide:85 design.math_reshape.rewrite[1].math

For each nonempty output worktile, F(-1)[u,v] = 0. For kb=0,...,KB-1 in ascending order and 0 <= h < Bn/ScaleN: sa(kb,u) = a_scale[kb,r0+u] if r0+u < M, otherwise 0. sb(kb,h) = b_scale[g,kb,y*(Bn/ScaleN)+h]. Scale(kb)[u,h] = sa(kb,u)*sb(kb,h). For 0 <= u < Bm and 0 <= v < Bn: F(kb)[u,v] = fma32(Scale(kb)[u,floor(v/ScaleN)], Dot(kb)[u,v], F(kb-1)[u,v]). For 0 <= u < V and 0 <= v < Bn: out[r0+u,c0+v] = bf16(F(KB-1)[u,v]). Discard F for u >= V. The output identity and all other side effects follow Contract's Grouped matrix product and writeback requirement.

## t_9fa9e57d9e72a342 | math.math | 686 chars | 1 declarations

wide:58 design.math_reshape.rewrite[0].math

Let KB = K/ScaleK and Tn = N/Bn. For each group g, let Rg = m_indptr[g+1]-m_indptr[g], Tm(g) = ceil_div(Rg,Bm). For 0 <= x < Tm(g), 0 <= y < Tn, define r0 = m_indptr[g]+x*Bm, c0 = y*Bn and V = min(Bm,Rg-x*Bm). For 0 <= kb < KB, 0 <= u < Bm, 0 <= v < Bn, 0 <= z < ScaleK: A_tile(kb)[u,z] = a[r0+u,kb*ScaleK+z] if u < H and r0+u < M; = 0 if u < H and r0+u >= M; = the retained, unspecified FP8 row payload if u >= H. B_tile(kb)[v,z] = b[g,c0+v,kb*ScaleK+z]. Dot(kb) = mm32(A_tile(kb), B_tile(kb)^T). All Bm rows are evaluated, including u >= V; no result from those rows is used in an output. Empty groups have no such instances. Let Mp = Bm*sum_g Tm(g) be the total evaluated row extent.

## t_c5ea9730799bd968 | math.math | 603 chars | 1 declarations

root:165 design.math_reshape.rewrite[1].math

For each nonempty output worktile, F(-1)[u,v] = 0. For kb=0,...,KB-1 in ascending order: sa(kb,u) = a_scale[kb,r0+u] if r0+u < M, otherwise 0. sb(kb) = b_scale[g,kb,y]. Scale(kb)[u] = sa(kb,u)*sb(kb). F(kb)[u,v] = fma32(Scale(kb)[u], Dot(kb)[u,v], F(kb-1)[u,v]). The two blocks kb=2*q and kb=2*q+1, 0 <= q < Q, have independent Dot values and scales; F absorbs the even block before the odd block. For 0 <= u < V and 0 <= v < Bn: out[r0+u,c0+v] = bf16(F(KB-1)[u,v]). Discard F for u >= V. The output identity and all other side effects follow Contract's Grouped matrix product and writeback requirement.

## t_cff0f4f218adbaf7 | math.math | 883 chars | 1 declarations

wide_recovery:58 design.math_reshape.rewrite[0].math

Let q in {0,1} index two complete evaluations. The values below belong to independent evaluations; no result of evaluation 0 is an input to evaluation 1. The notation suppresses q on local values. Let KB = K/ScaleK and Tn = N/Bn. For each group g, let Rg = m_indptr[g+1]-m_indptr[g], Tm(g) = ceil_div(Rg,Bm). For 0 <= x < Tm(g), 0 <= y < Tn, define r0 = m_indptr[g]+x*Bm, c0 = y*Bn and V = min(Bm,Rg-x*Bm). For 0 <= kb < KB, 0 <= u < Bm, 0 <= v < Bn, 0 <= z < ScaleK: A_tile(kb)[u,z] = a[r0+u,kb*ScaleK+z] if u < H and r0+u < M; = 0 if u < H and r0+u >= M; = the retained, unspecified FP8 row payload if u >= H. B_tile(kb)[v,z] = b[g,c0+v,kb*ScaleK+z]. Dot(kb) = mm32(A_tile(kb), B_tile(kb)^T). All Bm rows are evaluated, including u >= V; no result from those rows is used in an output. Empty groups have no such instances. Let Mp = Bm*sum_g Tm(g) be the total evaluated row extent.

## t_df52b6a51ee5813f | math.math | 739 chars | 1 declarations

root:136 design.math_reshape.rewrite[0].math

Let KB = K/ScaleK, Q = KB/2 and Tn = N/Bn. KB is even throughout the Contract domain. For each group g, let Rg = m_indptr[g+1]-m_indptr[g], Tm(g) = ceil_div(Rg,Bm). For 0 <= x < Tm(g), 0 <= y < Tn, define r0 = m_indptr[g]+x*Bm, c0 = y*Bn and V = min(Bm,Rg-x*Bm). For 0 <= kb < KB, 0 <= u < Bm, 0 <= v < Bn, 0 <= z < ScaleK: A_tile(kb)[u,z] = a[r0+u,kb*ScaleK+z] if u < H and r0+u < M; = 0 if u < H and r0+u >= M; = the retained, unspecified FP8 row payload if u >= H. B_tile(kb)[v,z] = b[g,c0+v,kb*ScaleK+z]. Dot(kb) = mm32(A_tile(kb), B_tile(kb)^T). All Bm rows are evaluated, including u >= V; no result from those rows is used in an output. Empty groups have no such instances. Let Mp = Bm*sum_g Tm(g) be the total evaluated row extent.

## t_e7183405da1a2052 | math.math | 697 chars | 1 declarations

wide_recovery_full:98 design.math_reshape.rewrite[1].math

For each q and each nonempty output worktile, F(-1)[u,v] = 0. For kb=0,...,KB-1 in ascending order and 0 <= h < Bn/ScaleN: sa(kb,u) = a_scale[kb,r0+u] if r0+u < M, otherwise 0. sb(kb,h) = b_scale[g,kb,y*(Bn/ScaleN)+h]. Scale(kb)[u,h] = sa(kb,u)*sb(kb,h). For 0 <= u < Bm and 0 <= v < Bn: F(kb)[u,v] = fma32(Scale(kb)[u,floor(v/ScaleN)], Dot(kb)[u,v], F(kb-1)[u,v]). For 0 <= u < V and 0 <= v < Bn: out[r0+u,c0+v] = bf16(F(KB-1)[u,v]). Discard F for u >= V. Each evaluation overwrites the entire output view without reading its prior contents. Evaluation 3 supplies the final output. The output identity and all other side effects follow Contract's Grouped matrix product and writeback requirement.

## t_fe45db25bea5f634 | math.math | 476 chars | 1 declarations

mma_sync:57 design.math_reshape.rewrite[1].math

For each output worktile, F(-1)[u,v]=0 for 0 <= u < Bm, 0 <= v < Bn. For kb=0,...,KB-1 in increasing order: sa(kb,u) = a_scale[kb,r0+u] if u<V, otherwise 0. sb(kb) = b_scale[g,kb,y]. Scale(kb)[u] = sa(kb,u)*sb(kb). F(kb)[u,v] = fma32(Scale(kb)[u],Dot(kb)[u,v],F(kb-1)[u,v]). For 0 <= u < V and 0 <= v < Bn: out[r0+u,c0+v] = bf16(F(KB-1)[u,v]). Discard F for u>=V. Preserve output identity and other side effects as specified by Contract's Grouped matrix product and writeback.

## t_36793cf7cb031f0f | math.meaning | 25 chars | 7 declarations

mma_sync:70 design.math_reshape.rewrite[1].symbol.F.meaning; root:180 design.math_reshape.rewrite[1].symbol.F.meaning; v0_recovery:108 design.math_reshape.rewrite[1].symbol.F.meaning; wide:103 design.math_reshape.rewrite[1].symbol.F.meaning; wide_recovery:118 design.math_reshape.rewrite[1].symbol.F.meaning; wide_recovery_chain:118 design.math_reshape.rewrite[1].symbol.F.meaning; wide_recovery_full:119 design.math_reshape.rewrite[1].symbol.F.meaning

Scaled output accumulator

## t_38789eb357400da4 | math.meaning | 46 chars | 6 declarations

root:160 design.math_reshape.rewrite[0].symbol.A_tile.meaning; v0_recovery:85 design.math_reshape.rewrite[0].symbol.A_tile.meaning; wide:79 design.math_reshape.rewrite[0].symbol.A_tile.meaning; wide_recovery:83 design.math_reshape.rewrite[0].symbol.A_tile.meaning; wide_recovery_chain:83 design.math_reshape.rewrite[0].symbol.A_tile.meaning; wide_recovery_full:84 design.math_reshape.rewrite[0].symbol.A_tile.meaning

Activation payload consumed by a block product

## t_53e5ac25b0fe585d | math.meaning | 27 chars | 7 declarations

mma_sync:50 design.math_reshape.rewrite[0].symbol.Bn.meaning; root:158 design.math_reshape.rewrite[0].symbol.Bn.meaning; v0_recovery:83 design.math_reshape.rewrite[0].symbol.Bn.meaning; wide:77 design.math_reshape.rewrite[0].symbol.Bn.meaning; wide_recovery:80 design.math_reshape.rewrite[0].symbol.Bn.meaning; wide_recovery_chain:80 design.math_reshape.rewrite[0].symbol.Bn.meaning; wide_recovery_full:81 design.math_reshape.rewrite[0].symbol.Bn.meaning

Columns per output worktile

## t_74ce90f3900d27d6 | math.meaning | 34 chars | 7 declarations

mma_sync:49 design.math_reshape.rewrite[0].symbol.Bm.meaning; root:157 design.math_reshape.rewrite[0].symbol.Bm.meaning; v0_recovery:82 design.math_reshape.rewrite[0].symbol.Bm.meaning; wide:76 design.math_reshape.rewrite[0].symbol.Bm.meaning; wide_recovery:79 design.math_reshape.rewrite[0].symbol.Bm.meaning; wide_recovery_chain:79 design.math_reshape.rewrite[0].symbol.Bm.meaning; wide_recovery_full:80 design.math_reshape.rewrite[0].symbol.Bm.meaning

Evaluated rows per output worktile

## t_794e3a9fc683451f | math.meaning | 12 chars | 1 declarations

mma_sync:52 design.math_reshape.rewrite[0].symbol.B_tile.meaning

Weight block

## t_89ebd7b5ddd8987d | math.meaning | 42 chars | 6 declarations

root:161 design.math_reshape.rewrite[0].symbol.B_tile.meaning; v0_recovery:86 design.math_reshape.rewrite[0].symbol.B_tile.meaning; wide:80 design.math_reshape.rewrite[0].symbol.B_tile.meaning; wide_recovery:87 design.math_reshape.rewrite[0].symbol.B_tile.meaning; wide_recovery_chain:87 design.math_reshape.rewrite[0].symbol.B_tile.meaning; wide_recovery_full:88 design.math_reshape.rewrite[0].symbol.B_tile.meaning

Weight payload consumed by a block product

## t_92e5e1f4e7dc1941 | math.meaning | 35 chars | 1 declarations

mma_sync:69 design.math_reshape.rewrite[1].symbol.Scale.meaning

Rounded product of the block scales

## t_94874b457110eb36 | math.meaning | 49 chars | 1 declarations

mma_sync:53 design.math_reshape.rewrite[0].symbol.Dot.meaning

Unscaled block product retained through promotion

## t_b4e7ce6b6e145622 | math.meaning | 32 chars | 6 declarations

root:162 design.math_reshape.rewrite[0].symbol.Dot.meaning; v0_recovery:87 design.math_reshape.rewrite[0].symbol.Dot.meaning; wide:81 design.math_reshape.rewrite[0].symbol.Dot.meaning; wide_recovery:91 design.math_reshape.rewrite[0].symbol.Dot.meaning; wide_recovery_chain:91 design.math_reshape.rewrite[0].symbol.Dot.meaning; wide_recovery_full:92 design.math_reshape.rewrite[0].symbol.Dot.meaning

Unscaled reduction-block product

## t_bb287895fa43c114 | math.meaning | 88 chars | 4 declarations

wide:100 design.math_reshape.rewrite[1].symbol.Scale.meaning; wide_recovery:114 design.math_reshape.rewrite[1].symbol.Scale.meaning; wide_recovery_chain:114 design.math_reshape.rewrite[1].symbol.Scale.meaning; wide_recovery_full:115 design.math_reshape.rewrite[1].symbol.Scale.meaning

Rounded product of the activation and weight scales for each row and column-scale block.

## t_d4396d60d48ad1c1 | math.meaning | 39 chars | 2 declarations

root:179 design.math_reshape.rewrite[1].symbol.Scale.meaning; v0_recovery:107 design.math_reshape.rewrite[1].symbol.Scale.meaning

Rounded product of the two block scales

## t_d6b00928af6f93e6 | math.meaning | 53 chars | 6 declarations

root:159 design.math_reshape.rewrite[0].symbol.H.meaning; v0_recovery:84 design.math_reshape.rewrite[0].symbol.H.meaning; wide:78 design.math_reshape.rewrite[0].symbol.H.meaning; wide_recovery:81 design.math_reshape.rewrite[0].symbol.H.meaning; wide_recovery_chain:81 design.math_reshape.rewrite[0].symbol.H.meaning; wide_recovery_full:82 design.math_reshape.rewrite[0].symbol.H.meaning

Loaded activation row extent for the current worktile

## t_d90c32e719c5e385 | math.meaning | 23 chars | 1 declarations

mma_sync:51 design.math_reshape.rewrite[0].symbol.A_tile.meaning

Padded activation block

## t_3e91d26791c95984 | math.name | 23 chars | 7 declarations

mma_sync:32 design.math_reshape.rewrite[0].name; root:135 design.math_reshape.rewrite[0].name; v0_recovery:57 design.math_reshape.rewrite[0].name; wide:57 design.math_reshape.rewrite[0].name; wide_recovery:57 design.math_reshape.rewrite[0].name; wide_recovery_chain:58 design.math_reshape.rewrite[0].name; wide_recovery_full:59 design.math_reshape.rewrite[0].name

Unscaled block products

## t_6d5f5d1440254353 | math.name | 38 chars | 1 declarations

mma_sync:56 design.math_reshape.rewrite[1].name

Block scaling and ordered accumulation

## t_c2f7251f92dbdd82 | math.name | 45 chars | 6 declarations

root:164 design.math_reshape.rewrite[1].name; v0_recovery:89 design.math_reshape.rewrite[1].name; wide:84 design.math_reshape.rewrite[1].name; wide_recovery:96 design.math_reshape.rewrite[1].name; wide_recovery_chain:96 design.math_reshape.rewrite[1].name; wide_recovery_full:97 design.math_reshape.rewrite[1].name

Block scaling and ordered output accumulation

## t_89511be0e5c07e9d | math.step | 9 chars | 7 declarations

mma_sync:31 design.math_reshape.rewrite[0].step; root:134 design.math_reshape.rewrite[0].step; v0_recovery:56 design.math_reshape.rewrite[0].step; wide:56 design.math_reshape.rewrite[0].step; wide_recovery:56 design.math_reshape.rewrite[0].step; wide_recovery_chain:57 design.math_reshape.rewrite[0].step; wide_recovery_full:58 design.math_reshape.rewrite[0].step

block_dot

## t_e6ce88a418703af9 | math.step | 10 chars | 7 declarations

mma_sync:55 design.math_reshape.rewrite[1].step; root:163 design.math_reshape.rewrite[1].step; v0_recovery:88 design.math_reshape.rewrite[1].step; wide:83 design.math_reshape.rewrite[1].step; wide_recovery:95 design.math_reshape.rewrite[1].step; wide_recovery_chain:95 design.math_reshape.rewrite[1].step; wide_recovery_full:96 design.math_reshape.rewrite[1].step

scaled_sum

## t_34f0d16b88c13e99 | math.thesis | 303 chars | 1 declarations

wide_recovery_chain:53 design.math_reshape.thesis

Evaluate the grouped product three times from the stored FP8 payloads, applying block scales after each dot and overwriting the output on each evaluation. Recovery triples the dense work and writeback while retaining the single-evaluation numerical path and avoiding scaled FP32 operand materialization.

## t_5670ba4d82b3eef3 | math.thesis | 436 chars | 1 declarations

v0_recovery:48 design.math_reshape.thesis

Multiply the stored FP8 payloads before applying each 128-element block's two scales. This avoids materializing scaled FP32 operands, but adds an ordered FP32 scale-and-accumulate step per output and reduction block; padded rows still undergo the dense computation before being discarded. Evaluate the entire product twice without sharing computed values; the second evaluation replaces the first output and doubles the evaluation work.

## t_67d8735720f757ed | math.thesis | 288 chars | 2 declarations

root:128 design.math_reshape.thesis; wide:52 design.math_reshape.thesis

Multiply the stored FP8 payloads before applying each 128-element block's two scales. This avoids materializing scaled FP32 operands, but adds an ordered FP32 scale-and-accumulate step per output and reduction block; padded rows still undergo the dense computation before being discarded.

## t_7f95893d299fa97e | math.thesis | 240 chars | 1 declarations

mma_sync:27 design.math_reshape.thesis

Form unscaled FP8 block products, then apply each reduction block's scales with an ordered FP32 accumulation. This removes full scaled-operand materialization at the cost of per-block promotion and dense evaluation of zero-padded tail rows.

## t_a8ce334467316b5c | math.thesis | 305 chars | 1 declarations

wide_recovery_full:54 design.math_reshape.thesis

Evaluate the grouped product four times from the stored FP8 payloads, applying block scales after each dot and overwriting the output on each evaluation. Recovery quadruples the dense work and writeback while retaining the single-evaluation numerical path and avoiding scaled FP32 operand materialization.

## t_b345533fb70e1a7d | math.thesis | 297 chars | 1 declarations

wide_recovery:52 design.math_reshape.thesis

Evaluate the grouped product twice from the stored FP8 payloads, applying block scales after each dot and overwriting the output on each evaluation. Recovery doubles the dense work and writeback while retaining the single-evaluation numerical path and avoiding scaled FP32 operand materialization.

## t_8f158114b190ab35 | metadata.describe | 137 chars | 7 declarations

mma_sync:3 meta_data.describe; root:3 meta_data.describe; v0_recovery:3 meta_data.describe; wide:3 meta_data.describe; wide_recovery:3 meta_data.describe; wide_recovery_chain:3 meta_data.describe; wide_recovery_full:3 meta_data.describe

Multiply contiguous row groups of FP8 activations by their FP8 weight matrices with block scales, overwriting a caller-owned BF16 output.

## t_0c0ec5a637fdc4a0 | pipeline.thesis | 284 chars | 1 declarations

mma_sync:147 design.tile_pipeline.thesis

All eight warps cooperate in a three-slot input pipeline, overlapping future copies with synchronous block products and scale accumulation. Drain copies before reusing input storage for output staging; this saves a separate output allocation but prevents overlap across that boundary.

## t_1b64ec9b91b79bb0 | pipeline.thesis | 516 chars | 1 declarations

wide_recovery_full:256 design.tile_pipeline.thesis

Each submission separately overlaps single-block operand loads, FP8 products and two column groups of epilogue work. Its specialization chooses a deeper input ring with two store batches (F) or a shallower ring with one larger batch (G); all local state and outstanding operations close within that submission. The source TMEM drain omits the row-block address contribution for its row-owning warps; this design retains the complete products and distinct row ownership established by the producer, scales and stores.

## t_a8d2d83df2fff2fa | pipeline.thesis | 573 chars | 1 declarations

wide_recovery_chain:251 design.tile_pipeline.thesis

Each submission separately overlaps single-block operand loads, FP8 products and two column groups of epilogue work. Its specialization chooses a deeper input ring with two store batches (F) or a shallower ring with one larger batch (G); all local state and outstanding operations close within that submission. The source TMEM drain omits the row-group address contribution in every F/G instance of first, retry and final; complete 128x256 products and distinct row ownership remain supported by product generation, 512-column TMEM capacity, row-specific scales and stores.

## t_c8e41e2af9602b90 | pipeline.thesis | 579 chars | 1 declarations

wide_recovery:248 design.tile_pipeline.thesis

Each submission separately overlaps single-block operand loads, FP8 products and two column groups of epilogue work. Its specialization chooses a deeper input ring with two store batches (F) or a shallower ring with one larger batch (G); all local state and outstanding operations close within that submission. The source TMEM drain omits the row-group address contribution in every F/G instance of first and retry; this design retains complete 128x256 products and distinct row ownership supported by product generation, 512-column TMEM capacity, row-specific scales and stores.

## t_da2153885c0b4968 | pipeline.thesis | 517 chars | 1 declarations

wide:192 design.tile_pipeline.thesis

Overlap single-block loads, FP8 products and two column groups of epilogue work. F trades a deeper operand ring for two sequential store batches; G uses a shallower ring and one larger batch. Product slots return after all TMEM reads, while final register FMA and global stores can continue. The source wide TMEM drain in both F and G omits the row-group address contribution; this design retains the complete products and distinct row ownership supported by product generation, full TMEM capacity, scales and stores.

## t_dc5a708387e88288 | pipeline.thesis | 707 chars | 1 declarations

v0_recovery:198 design.tile_pipeline.thesis

Overlap pair loads, independent FP8 products and scaled epilogue work using a three-slot operand ring and two TMEM pair slots. TMEM is released after all reads, before the final register FMA; output staging waits only for prior source reads, allowing global writes to continue into later work. Each phase instantiates this pipeline in its own workers; resource contents, ring generations and outstanding operations do not carry between kernels. Both graph and direct execute a source V0 TMEM drain that omits the row-group address contribution; this design retains the complete products, distinct row ownership and resource strategy supported by product generation, TMEM capacity, per-row scales and stores.

## t_e6d03b7a509f5242 | pipeline.thesis | 498 chars | 1 declarations

root:256 design.tile_pipeline.thesis

Overlap pair loads, independent FP8 products and scaled epilogue work using a three-slot operand ring and two TMEM pair slots. TMEM is released after all reads, before the final register FMA; output staging waits only for prior source reads, allowing global writes to continue into later work. The source V0 TMEM drain omits the row-group address contribution; this design retains the complete products and distinct row ownership established by product generation, TMEM capacity, scales and stores.

## t_0865c15e16299fdf | pipeline.verb | 275 chars | 1 declarations

mma_sync:203 design.tile_pipeline.phase.main.role.Main.info.verb.publish

Convert the entire final F to BF16 in Output(t), using the first InputRing slot after drain. This view holds the Bm-by-Bn block plus implementation row padding, within that slot's capacity. A CTA rendezvous makes every staged row readable. Schedule remains disjoint and live.

## t_0bf9de0efb9ae9c1 | pipeline.verb | 291 chars | 2 declarations

root:392 design.tile_pipeline.phase.main.role.Epilogue.info.verb.prepare; v0_recovery:337 design.tile_pipeline.phase.graph.role.Epilogue.info.verb.prepare

Work(t) decodes its coordinates from Schedule and initializes F to the mathematical initial state. Each epilogue member owns disjoint rows and all Bn columns of those rows. Define Full(t)=1 when V=Bm, otherwise 0. All Bm rows participate in computation and release, including discarded rows.

## t_0d44f5a55baa3664 | pipeline.verb | 421 chars | 2 declarations

root:439 design.tile_pipeline.phase.main.role.Epilogue.info.verb.release; v0_recovery:384 design.tile_pipeline.phase.graph.role.Epilogue.info.verb.release

DotPair(n) joins the read completion of every epilogue row and returns its TMEM pair slot; the final register FMA has not yet executed. Staging(t,f) instead waits, through the bulk source-read completion primitive, for every outstanding earlier full-tile store to stop reading OutputStaging, then rendezvous all epilogue members. The first full tile has no earlier store to wait for. Tail tiles retain this store history.

## t_12183ed223825f55 | pipeline.verb | 66 chars | 4 declarations

wide:314 design.tile_pipeline.phase.main.role.Epilogue.info.verb.await; wide_recovery:395 design.tile_pipeline.phase.first.role.Epilogue.info.verb.await; wide_recovery_chain:398 design.tile_pipeline.phase.first.role.Epilogue.info.verb.await; wide_recovery_full:402 design.tile_pipeline.phase.first.role.Epilogue.info.verb.await

Complete after the referenced TMEM fragment delivery has finished.

## t_16da99180475ebe6 | pipeline.verb | 177 chars | 1 declarations

mma_sync:201 design.tile_pipeline.phase.main.role.Main.info.verb.drain

Copies(t) waits for every committed copy group to finish, including the final empty groups, and rendezvous the CTA. InputRing is then free of input readers and in-flight writes.

## t_28850f7557241d40 | pipeline.verb | 430 chars | 1 declarations

wide:264 design.tile_pipeline.phase.main.role.Producer.info.verb.initialize.meaning

Schedule is the CTA-cooperative entry: copy m_indptr, build prefix, initialize resource generations and allocate ProductRing with the issuing warp. Completion includes the CTA rendezvous before role divergence. Schedule holds boundaries and prefix. Let J=1 for the submitted F instance and J=2 for G; these instances are defined in dispatch.when. Set R=5-J for this worker's lifetime. ProductRing holds two independent Dot blocks.

## t_292ce8a0c81de3c0 | pipeline.verb | 386 chars | 1 declarations

mma_sync:189 design.tile_pipeline.phase.main.role.Main.info.verb.load.meaning

Stage(t,p) issues and commits one per-thread copy group. For p<KB, fill A(t,p), B(t,p) and AS(t,p) with A_tile(p), B_tile(p) and sa(p,u). The predicated A and scale copies zero-fill rows u>=V without reading them; B is loaded fully. Done covers every participating thread's transfers. For p>=KB, commit an empty group, with no storage write; its issue and completion events still exist.

## t_2bc4b4e440fbf53c | pipeline.verb | 440 chars | 2 declarations

root:400 design.tile_pipeline.phase.main.role.Epilogue.info.verb.prefetch_a.meaning; v0_recovery:345 design.tile_pipeline.phase.graph.role.Epilogue.info.verb.prefetch_a.meaning

AS(t,p) commits one row-private copy group for both scales of pair p when p<Q. Rows with r0+u>=M receive zero without reading the source; rows in another physical group are still read. For p>=Q an empty group is committed and no scale slot is written. Each worktile primes two groups, then commits one replacement per consumed pair. The final two groups are empty; the group stream is not explicitly drained or reset at worktile boundaries.

## t_36c41cb3bb20ccff | pipeline.verb | 227 chars | 1 declarations

mma_sync:186 design.tile_pipeline.phase.main.role.Main.info.verb.prepare

Decode Work(t) from Schedule according to the assigned w. Warp h owns rows [32*floor(h/4),32*(floor(h/4)+1)) and columns [32*(h%4),32*(h%4+1)) of Dot and F. The copy and output-store work is distributed across all role members.

## t_4227e3e6ac2f9cf7 | pipeline.verb | 590 chars | 2 declarations

root:368 design.tile_pipeline.phase.main.role.Issuer.info.verb.mma; v0_recovery:313 design.tile_pipeline.phase.graph.role.Issuer.info.verb.mma

NI=2 when Wub*Q >= 48*C, otherwise NI=1; it is fixed from the submitted scalar parameters for this worker's lifetime. With one member its elected lane issues both block products; with two, one elected lane issues each block. A_pair(n) and B_pair(n) are the two operand views of Operands(n). They produce DotPair(n), containing Dot(2*q) and Dot(2*q+1), each independently initialized by its first reduction slice. The action's completion joins all NI member completions: both products are readable and all operand reads have ended. The same completion releases the operand stage to Producer.

## t_44b4e82003555377 | pipeline.verb | 505 chars | 4 declarations

wide:300 design.tile_pipeline.phase.main.role.Epilogue.info.verb.prefetch.meaning; wide_recovery:380 design.tile_pipeline.phase.first.role.Epilogue.info.verb.prefetch.meaning; wide_recovery_chain:383 design.tile_pipeline.phase.first.role.Epilogue.info.verb.prefetch.meaning; wide_recovery_full:387 design.tile_pipeline.phase.first.role.Epilogue.info.verb.prefetch.meaning

RawScale(t,p) loads the activation and weight scales of block p into ScalePrefetch when p<KB. Activation scales are replicated across column groups; weight scales across rows. A row beyond global M receives zero without reading a_scale, while rows in another physical group are still read. For p>=KB both scales are set to zero, complete at issue, and perform no global read. Each worktile primes blocks 0 and 1; consuming block kb allows replacement by kb+2. The captured Scale survives that replacement.

## t_4c19940dc1c34e7f | pipeline.verb | 286 chars | 2 declarations

root:338 design.tile_pipeline.phase.main.role.Producer.info.verb.initialize.meaning; v0_recovery:283 design.tile_pipeline.phase.graph.role.Producer.info.verb.initialize.meaning

Schedule is the CTA-cooperative entry: copy m_indptr and have the leader compute prefix; initialize empty resource generations and allocate ProductRing with the issuing warp. Completion includes the CTA entry rendezvous before role divergence. Schedule stores the boundaries and prefix.

## t_4f3842b91450ac4d | pipeline.verb | 221 chars | 4 declarations

wide:305 design.tile_pipeline.phase.main.role.Epilogue.info.verb.capture; wide_recovery:386 design.tile_pipeline.phase.first.role.Epilogue.info.verb.capture; wide_recovery_chain:389 design.tile_pipeline.phase.first.role.Epilogue.info.verb.capture; wide_recovery_full:393 design.tile_pipeline.phase.first.role.Epilogue.info.verb.capture

Scale(n) consumes RawScale(t,kb) and computes mathematical Scale(kb) into CurrentScale. Done ends those raw-scale reads. Each row and column group's scale is retained through all four fragment accumulations of this block.

## t_53b69dd9a068a996 | pipeline.verb | 222 chars | 2 declarations

root:453 design.tile_pipeline.phase.main.role.Epilogue.info.verb.store.meaning; v0_recovery:398 design.tile_pipeline.phase.graph.role.Epilogue.info.verb.store.meaning

Output(t,f) issues the two full-height 64-column bulk transfers to the owned out region and commits them as one group. Done means global writes are complete. The next worktile may start while these writes remain in flight.

## t_563273446538a7f4 | pipeline.verb | 429 chars | 4 declarations

wide:317 design.tile_pipeline.phase.main.role.Epilogue.info.verb.release; wide_recovery:398 design.tile_pipeline.phase.first.role.Epilogue.info.verb.release; wide_recovery_chain:401 design.tile_pipeline.phase.first.role.Epilogue.info.verb.release; wide_recovery_full:405 design.tile_pipeline.phase.first.role.Epilogue.info.verb.release

Product(n) joins completed TMEM reads from all epilogue rows and both column groups, returning its slot before the final register FMA. Staging(t,f) instead waits for every outstanding earlier bulk store to stop reading OutputStaging, then rendezvous all epilogue members. The first batch has no prior store to wait for; later batches include stores from this or earlier worktiles. Tail worktiles preserve this bulk-store history.

## t_5d0da106f5c337e2 | pipeline.verb | 578 chars | 1 declarations

wide_recovery:332 design.tile_pipeline.phase.first.role.Producer.info.verb.initialize.meaning

Schedule is the CTA-cooperative entry: copy m_indptr, build prefix, initialize resource generations and allocate ProductRing with the issuing warp. Completion includes the CTA rendezvous before role divergence. Schedule holds boundaries and prefix. Let J=1 when this phase uses L[q]=F and J=2 when L[q]=G, with q bound by work_schedule. Set R=5-J for this worker's lifetime. ProductRing holds two independent Dot blocks. Each launch starts its own ring positions, generations, scale state and store history; no local resource or event instance is shared between first and retry.

## t_638f7cf096523b77 | pipeline.verb | 360 chars | 1 declarations

mma_sync:206 design.tile_pipeline.phase.main.role.Main.info.verb.store

Cooperatively read Output(t) and issue ordinary vector stores to the owned out region for rows u<V only. Done includes the trailing CTA rendezvous and ends all staging reads, permitting the next worktile to overwrite InputRing. Destination writes are covered by main.kernel_done; no host synchronization or separate asynchronous bulk-store queue is introduced.

## t_69676bc6db27b2b3 | pipeline.verb | 209 chars | 2 declarations

root:428 design.tile_pipeline.phase.main.role.Epilogue.info.verb.read.meaning; v0_recovery:373 design.tile_pipeline.phase.graph.role.Epilogue.info.verb.read.meaning

Fragment(n,d) asynchronously reads all Bm rows of columns [32*(d%4),32*(d%4+1)) from block floor(d/4) of DotPair(n) into DrainRegister. Done guarantees that this TMEM read and register delivery have completed.

## t_699e0bdd982ace3b | pipeline.verb | 242 chars | 2 declarations

root:463 design.tile_pipeline.phase.main.role.Epilogue.info.verb.drain; v0_recovery:408 design.tile_pipeline.phase.graph.role.Epilogue.info.verb.drain

Stores waits for global completion of all previously issued bulk-store groups, or has no data effect when none were issued. The final CTA rendezvous and ProductRing deallocation follow role completion, including for workers with no worktiles.

## t_69b3a9efde19b6dd | pipeline.verb | 243 chars | 2 declarations

root:421 design.tile_pipeline.phase.main.role.Epilogue.info.verb.acquire; v0_recovery:366 design.tile_pipeline.phase.graph.role.Epilogue.info.verb.acquire

DotPair(n) waits for both products to be ready and establishes their visibility to the epilogue. It then evaluates both mathematical Scale values, replacing the captured raw scales in CurrentScales and retaining the products through this pair.

## t_726123fe5726f3d0 | pipeline.verb | 583 chars | 1 declarations

wide_recovery_chain:335 design.tile_pipeline.phase.first.role.Producer.info.verb.initialize.meaning

Schedule is the CTA-cooperative entry: copy m_indptr, build prefix, initialize resource generations and allocate ProductRing with the issuing warp. Completion includes the CTA rendezvous before role divergence. Schedule holds boundaries and prefix. Let J=1 when this phase uses L[q]=F and J=2 when L[q]=G, with q bound by work_schedule. Set R=5-J for this worker's lifetime. ProductRing holds two independent Dot blocks. Each launch starts its own ring positions, generations, scale state and store history; no local resource or event instance is shared among first, retry and final.

## t_75bbbc5de173ffb9 | pipeline.verb | 648 chars | 2 declarations

root:344 design.tile_pipeline.phase.main.role.Producer.info.verb.load; v0_recovery:289 design.tile_pipeline.phase.graph.role.Producer.info.verb.load

The elected lane loads Operands(n), the two A_tile/B_tile pairs for work t and pair q. A transfers cover H rows; rows beyond the global M extent are zero-filled, rows above H retain their previous storage without initialization. B transfers cover Bn rows. Completion covers all actual transfer bytes, not the untouched A rows. Ring position and completion generations continue across worktiles. Bstream is the actual submitted bstream flag: policy defaults it to zero on a failed or invalid host read, otherwise sets it when copied total row tiles equal copied nonempty groups; cache hits reuse the flag and graph execution uses its recorded value.

## t_86462b1db52be507 | pipeline.verb | 176 chars | 2 declarations

root:447 design.tile_pipeline.phase.main.role.Epilogue.info.verb.publish; v0_recovery:392 design.tile_pipeline.phase.graph.role.Epilogue.info.verb.publish

Output(t,f) converts F to BF16 in OutputStaging for a full tile; all writers publish to the asynchronous access domain and rendezvous before the elected store lane may proceed.

## t_8b30b3f661e93795 | pipeline.verb | 182 chars | 1 declarations

mma_sync:184 design.tile_pipeline.phase.main.role.Main.info.verb.initialize

Cooperatively copy m_indptr into Schedule, rendezvous, have the CTA leader form prefix, and rendezvous again. Both arrays remain live for the worker, including when it owns no tiles.

## t_8cfcaf7ad21abf57 | pipeline.verb | 362 chars | 1 declarations

mma_sync:197 design.tile_pipeline.phase.main.role.Main.info.verb.compute

Block(t,kb) performs block_dot into the register Dot, reads AS(t,kb) and the corresponding global b_scale, and performs scaled_sum into F. Dot is initialized anew for this block. All Bm rows participate, including discarded rows. Done ends the block's operand and scale reads and its Dot consumption; the next acquire joins all warps before any input-slot reuse.

## t_9356af0145f414aa | pipeline.verb | 235 chars | 4 declarations

wide:332 design.tile_pipeline.phase.main.role.Epilogue.info.verb.drain; wide_recovery:413 design.tile_pipeline.phase.first.role.Epilogue.info.verb.drain; wide_recovery_chain:416 design.tile_pipeline.phase.first.role.Epilogue.info.verb.drain; wide_recovery_full:420 design.tile_pipeline.phase.first.role.Epilogue.info.verb.drain

Stores waits for destination completion of all earlier bulk groups; it has no data effect when none were issued. After all roles finish, the CTA rendezvous precedes ProductRing deallocation, including for workers assigned no worktiles.

## t_9a3b2c70cbbb9272 | pipeline.verb | 194 chars | 2 declarations

root:458 design.tile_pipeline.phase.main.role.Epilogue.info.verb.store_tail; v0_recovery:403 design.tile_pipeline.phase.graph.role.Epilogue.info.verb.store_tail

out(t,z) converts and directly stores only rows u<V; discarded rows do not write, and OutputStaging and prior bulk-store history remain untouched. F is no longer read when this action completes.

## t_a063941855ab4446 | pipeline.verb | 308 chars | 4 declarations

wide:295 design.tile_pipeline.phase.main.role.Epilogue.info.verb.prepare; wide_recovery:374 design.tile_pipeline.phase.first.role.Epilogue.info.verb.prepare; wide_recovery_chain:377 design.tile_pipeline.phase.first.role.Epilogue.info.verb.prepare; wide_recovery_full:381 design.tile_pipeline.phase.first.role.Epilogue.info.verb.prepare

Work(t) decodes coordinates from Schedule and initializes F to its mathematical initial state. Two four-warp groups each own all Bm rows and one disjoint ScaleN-column group; within each group, members own disjoint rows. All Bm rows participate in computation and release. Let Full(t)=1 if V=Bm, otherwise 0.

## t_a1dfe6630488e01e | pipeline.verb | 593 chars | 1 declarations

wide_recovery_full:339 design.tile_pipeline.phase.first.role.Producer.info.verb.initialize.meaning

Schedule is the CTA-cooperative entry: copy m_indptr, build prefix, initialize resource generations and allocate ProductRing with the issuing warp. Completion includes the CTA rendezvous before role divergence. Schedule holds boundaries and prefix. Let J=1 when this phase uses L[q]=F and J=2 when L[q]=G, with q bound by work_schedule. Set R=5-J for this worker's lifetime. ProductRing holds two independent Dot blocks. Each launch starts its own ring positions, generations, scale state and store history; no local resource or event instance is shared among first, retry, fallback and final.

## t_a5b5079e6a9f8a7b | pipeline.verb | 214 chars | 2 declarations

root:411 design.tile_pipeline.phase.main.role.Epilogue.info.verb.prefetch_b.meaning; v0_recovery:356 design.tile_pipeline.phase.graph.role.Epilogue.info.verb.prefetch_b.meaning

BS(t,p) loads both weight scales for pair p into WeightPrefetch, replicated for epilogue rows, when p<Q; p=Q assigns zeros without global reads. Captured current scales survive overwriting these prefetch registers.

## t_ae6b2f6118db005a | pipeline.verb | 246 chars | 2 declarations

root:416 design.tile_pipeline.phase.main.role.Epilogue.info.verb.capture; v0_recovery:361 design.tile_pipeline.phase.graph.role.Epilogue.info.verb.capture

Scales(n) waits until all but the newest committed activation copy group are complete, then captures AS(t,q) and BS(t,q) in CurrentScales. Its done ends this pair's scale-slot reads. The next weight pair can be prefetched before DotPair is ready.

## t_b3d39f804b650aab | pipeline.verb | 181 chars | 4 declarations

wide:312 design.tile_pipeline.phase.main.role.Epilogue.info.verb.read.meaning; wide_recovery:393 design.tile_pipeline.phase.first.role.Epilogue.info.verb.read.meaning; wide_recovery_chain:396 design.tile_pipeline.phase.first.role.Epilogue.info.verb.read.meaning; wide_recovery_full:400 design.tile_pipeline.phase.first.role.Epilogue.info.verb.read.meaning

Fragment(n,d) reads columns [32*d,32*(d+1)) of each column group from Product(n) into DrainRegister for all Bm rows. Done includes delivery of both groups' fragments into registers.

## t_b88d8bda803e699c | pipeline.verb | 240 chars | 4 declarations

wide:322 design.tile_pipeline.phase.main.role.Epilogue.info.verb.publish; wide_recovery:403 design.tile_pipeline.phase.first.role.Epilogue.info.verb.publish; wide_recovery_chain:406 design.tile_pipeline.phase.first.role.Epilogue.info.verb.publish; wide_recovery_full:410 design.tile_pipeline.phase.first.role.Epilogue.info.verb.publish

Output(t,f) converts the J column groups numbered J*f through J*f+J-1 from F to BF16 in OutputStaging. All epilogue members rendezvous; the writers publish their data to the asynchronous access domain before the elected store lane proceeds.

## t_ca24b3ca29c08010 | pipeline.verb | 311 chars | 4 declarations

wide:283 design.tile_pipeline.phase.main.role.Issuer.info.verb.mma; wide_recovery:357 design.tile_pipeline.phase.first.role.Issuer.info.verb.mma; wide_recovery_chain:360 design.tile_pipeline.phase.first.role.Issuer.info.verb.mma; wide_recovery_full:364 design.tile_pipeline.phase.first.role.Issuer.info.verb.mma

The elected lane computes Dot(kb) into Product(n) using the four reduction slices defined by block_dot, initializing on the first slice. Done is the committed completion that makes the product readable and ends all reads of Operands(n). Producer and Epilogue observe this same completion for different purposes.

## t_e1fdd723c55de1b2 | pipeline.verb | 242 chars | 4 declarations

wide:327 design.tile_pipeline.phase.main.role.Epilogue.info.verb.store.meaning; wide_recovery:408 design.tile_pipeline.phase.first.role.Epilogue.info.verb.store.meaning; wide_recovery_chain:411 design.tile_pipeline.phase.first.role.Epilogue.info.verb.store.meaning; wide_recovery_full:415 design.tile_pipeline.phase.first.role.Epilogue.info.verb.store.meaning

Output(t,f) issues 2*J full-height 64-column bulk transfers to out starting at column c0+J*f*ScaleN and commits one group. Done means destination writes are complete. The next batch or worktile may advance while those writes remain in flight.

## t_e2a747c2c30fb92a | pipeline.verb | 114 chars | 4 declarations

wide:308 design.tile_pipeline.phase.main.role.Epilogue.info.verb.acquire; wide_recovery:389 design.tile_pipeline.phase.first.role.Epilogue.info.verb.acquire; wide_recovery_chain:392 design.tile_pipeline.phase.first.role.Epilogue.info.verb.acquire; wide_recovery_full:396 design.tile_pipeline.phase.first.role.Epilogue.info.verb.acquire

Product(n) observes the issuer's committed completion and establishes TMEM visibility before any fragment is read.

## t_e3a6ac1b3df66c2c | pipeline.verb | 291 chars | 1 declarations

mma_sync:194 design.tile_pipeline.phase.main.role.Main.info.verb.acquire

Ready(t,kb) waits until all but the newest committed group have completed on each thread, then rendezvous the CTA. Done makes the current stage readable by every warp and joins the previous Block's reads before the next refill overwrites its slot. Stage indices restart at zero per worktile.

## t_e3c3a16755b66282 | pipeline.verb | 242 chars | 2 declarations

root:434 design.tile_pipeline.phase.main.role.Epilogue.info.verb.compute; v0_recovery:379 design.tile_pipeline.phase.graph.role.Epilogue.info.verb.compute

Accum(n,d) consumes Fragment(n,d) and the corresponding retained Scale values to perform scaled_sum for those columns. Each scale product is reused across its four column fragments. Done ends all reads of that register fragment and updates F.

## t_f899bb48cd6f38b5 | pipeline.verb | 78 chars | 2 declarations

root:433 design.tile_pipeline.phase.main.role.Epilogue.info.verb.await; v0_recovery:378 design.tile_pipeline.phase.graph.role.Epilogue.info.verb.await

Complete only after the explicitly referenced fragment delivery is guaranteed.

## t_fabb378485686a74 | pipeline.verb | 185 chars | 4 declarations

wide:330 design.tile_pipeline.phase.main.role.Epilogue.info.verb.store_tail; wide_recovery:411 design.tile_pipeline.phase.first.role.Epilogue.info.verb.store_tail; wide_recovery_chain:414 design.tile_pipeline.phase.first.role.Epilogue.info.verb.store_tail; wide_recovery_full:418 design.tile_pipeline.phase.first.role.Epilogue.info.verb.store_tail

out(t,z) converts and directly stores F for rows u<V in both column groups. Other rows do not write. Done ends all F reads; OutputStaging and the prior bulk-store history are untouched.

## t_fdbc2e707f1ded94 | pipeline.verb | 183 chars | 4 declarations

wide:315 design.tile_pipeline.phase.main.role.Epilogue.info.verb.compute; wide_recovery:396 design.tile_pipeline.phase.first.role.Epilogue.info.verb.compute; wide_recovery_chain:399 design.tile_pipeline.phase.first.role.Epilogue.info.verb.compute; wide_recovery_full:403 design.tile_pipeline.phase.first.role.Epilogue.info.verb.compute

Accum(n,d) uses Fragment(n,d) and the retained Scale(n) to perform scaled_sum for those columns in F. Done ends reads of the fragment, permitting that DrainRegister slot to be reused.

## t_ff2b1c348d5b3f07 | pipeline.verb | 634 chars | 4 declarations

wide:269 design.tile_pipeline.phase.main.role.Producer.info.verb.load; wide_recovery:338 design.tile_pipeline.phase.first.role.Producer.info.verb.load; wide_recovery_chain:341 design.tile_pipeline.phase.first.role.Producer.info.verb.load; wide_recovery_full:345 design.tile_pipeline.phase.first.role.Producer.info.verb.load

The elected lane loads A_tile(kb) and B_tile(kb) into A_block(n) and B_block(n), jointly named Operands(n). The A box covers H rows and zero-fills beyond global M; rows above H retain their storage contents. B covers all Bn columns. Done covers all actual transfer bytes, without requiring a write to untouched A rows. Ring positions and generations continue across worktiles. Bstream is the submitted bstream flag: on a policy miss it is zero after a failed or invalid host read, otherwise one exactly when copied row tiles equal copied nonempty groups. Policy hits reuse the flag; graph replay uses the value recorded in that graph.

## t_336d1672dae971e6 | realization.choice | 244 chars | 2 declarations

root:532 realization.require[3].choice; v0_recovery:483 realization.require[4].choice

Match the BF16 staging layout to the UINT16 64-by-128 output tensor maps. Publish with fence.proxy.async.shared::cta and an epilogue rendezvous; use bulk-group stores, wait_group.read 0 before reuse and wait_group 0 for final global completion.

## t_414786754c83928e | realization.choice | 145 chars | 6 declarations

root:508 realization.require[0].choice; v0_recovery:459 realization.require[1].choice; wide:365 realization.require[0].choice; wide_recovery:451 realization.require[0].choice; wide_recovery_chain:455 realization.require[0].choice; wide_recovery_full:461 realization.require[0].choice

Use tensor-map 2D asynchronous loads with byte-count completion barriers; the operand maps and MMA descriptors use compatible 128-byte swizzling.

## t_4501a0b3a8f6743c | realization.choice | 186 chars | 1 declarations

mma_sync:236 realization.require[1].choice

Form Dot with mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 and ldmatrix.x4 shared operand reads. Match the copy layout and the FP8 byte packing consumed by those matrix fragments.

## t_4d602afeaf4d9d35 | realization.choice | 207 chars | 4 declarations

wide:378 realization.require[3].choice; wide_recovery:466 realization.require[3].choice; wide_recovery_chain:470 realization.require[3].choice; wide_recovery_full:476 realization.require[3].choice

Match BF16 staging to UINT16 64-by-128 tensor maps. Publish with fence.proxy.async.shared::cta and an epilogue rendezvous; use bulk-group stores, wait_group.read 0 for source reuse and wait_group 0 for exit.

## t_4fe0868918b77fe0 | realization.choice | 68 chars | 6 declarations

root:542 realization.prefer[0].choice; v0_recovery:493 realization.prefer[0].choice; wide:384 realization.prefer[0].choice; wide_recovery:472 realization.prefer[0].choice; wide_recovery_chain:476 realization.prefer[0].choice; wide_recovery_full:482 realization.prefer[0].choice

Use the L2 evict_first hint for B loads exactly when Bstream is set.

## t_703fc340b60bdbb3 | realization.choice | 263 chars | 2 declarations

root:523 realization.require[2].choice; v0_recovery:474 realization.require[3].choice

Implement activation-scale groups with predicated four-byte cp.async, commit_group and wait_group 1, including empty final commits. Use tcgen05.ld with tcgen05.wait::ld for TMEM fragment completion and tcgen05.fence::after_thread_sync when acquiring the products.

## t_7c209b877849283c | realization.choice | 90 chars | 1 declarations

v0_recovery:454 realization.require[0].choice

Submit graph replay and its direct recovery on the same CUDA stream passed to fp8umma_run.

## t_9dc6a45dbc96d995 | realization.choice | 321 chars | 1 declarations

mma_sync:230 realization.require[0].choice

Implement stage loads with predicated 16-byte cp.async.cg for FP8 payloads and four-byte cp.async.ca for activation scales, using zero source size for masked rows. Commit on every thread, including the two final empty groups. Use wait_group 1 plus __syncthreads for acquire, and wait_group 0 plus __syncthreads for drain.

## t_a6b5961e94c0a09a | realization.choice | 202 chars | 2 declarations

root:515 realization.require[1].choice; v0_recovery:466 realization.require[2].choice

Use tcgen05.mma.cta_group::1.kind::f8f6f4 for the FP8-to-FP32 products, tcgen05.commit completion arrivals for each issuing member, and tcgen05.fence::after_thread_sync after the relevant barrier waits.

## t_acaeb302eb584e18 | realization.choice | 125 chars | 4 declarations

wide:374 realization.require[2].choice; wide_recovery:461 realization.require[2].choice; wide_recovery_chain:465 realization.require[2].choice; wide_recovery_full:471 realization.require[2].choice

Use tcgen05.ld with tcgen05.wait::ld for fragment delivery, and tcgen05.fence::after_thread_sync when acquiring each product.

## t_b18a6353bdfed869 | realization.choice | 273 chars | 1 declarations

mma_sync:241 realization.require[2].choice

Match BF16 stmatrix.x4 output packing to the shared-to-global 16-byte vector readers. The source layout uses a 272-byte row pitch; its 17408-byte staging footprint fits the 24832-byte first input slot. Use __syncthreads after staging writes and after the vector-store work.

## t_c117b5b4d38cd662 | realization.choice | 167 chars | 4 declarations

wide:370 realization.require[1].choice; wide_recovery:456 realization.require[1].choice; wide_recovery_chain:460 realization.require[1].choice; wide_recovery_full:466 realization.require[1].choice

Use tcgen05.mma.cta_group::1.kind::f8f6f4 and tcgen05.commit for the product/operand completion; use tcgen05.fence::after_thread_sync after the relevant barrier waits.

## t_032bd6e37cb99b34 | realization.reason | 140 chars | 2 declarations

root:537 realization.require[3].reason; v0_recovery:488 realization.require[4].reason

The single staging buffer is shared by all row writers and its reuse permission intentionally precedes completion of the destination writes.

## t_3dc31f613310f114 | realization.reason | 118 chars | 4 declarations

wide:380 realization.require[3].reason; wide_recovery:468 realization.require[3].reason; wide_recovery_chain:472 realization.require[3].reason; wide_recovery_full:478 realization.require[3].reason

The J-group buffer is shared by its writers; permission to overwrite it precedes completion of the destination writes.

## t_3ff63f169a9c0949 | realization.reason | 143 chars | 2 declarations

root:528 realization.require[2].reason; v0_recovery:479 realization.require[3].reason

Scale capture depends on the group stream, while fragment consumption and early TMEM release depend on completed reads, not merely their issue.

## t_4195724f56e00725 | realization.reason | 126 chars | 4 declarations

wide:367 realization.require[0].reason; wide_recovery:453 realization.require[0].reason; wide_recovery_chain:457 realization.require[0].reason; wide_recovery_full:463 realization.require[0].reason

Readiness covers the selected H-row A box and full B box while preserving the untouched A rows and the R-slot operand storage.

## t_50f2c0e7e801f558 | realization.reason | 112 chars | 4 declarations

wide:372 realization.require[1].reason; wide_recovery:458 realization.require[1].reason; wide_recovery_chain:462 realization.require[1].reason; wide_recovery_full:468 realization.require[1].reason

These bind the TMEM product path and the completion observed both for epilogue visibility and for operand reuse.

## t_5e29a9583b347a5c | realization.reason | 204 chars | 2 declarations

root:543 realization.prefer[0].reason; v0_recovery:494 realization.prefer[0].reason

The host policy targets one row tile per nonempty group, where B has no row-tile reuse. A retained policy can outlive that tensor content; the hint does not assert that the current execution has no reuse.

## t_a62d9ff9a382d468 | realization.reason | 128 chars | 1 declarations

v0_recovery:455 realization.require[0].reason

Stream ordering provides the declared graph.kernel_done startup dependency without a host synchronization or an output readback.

## t_a9336c2f06a3b8be | realization.reason | 133 chars | 1 declarations

mma_sync:244 realization.require[2].reason

These rendezvous provide cross-warp output visibility and end all staging reads before the input ring is reused by the next worktile.

## t_c472ae966b631011 | realization.reason | 141 chars | 2 declarations

root:519 realization.require[1].reason; v0_recovery:470 realization.require[2].reason

These bind the TMEM product path and the shared completion used both for product visibility and operand reuse, including the two-member join.

## t_ccbca337cad325db | realization.reason | 132 chars | 1 declarations

mma_sync:238 realization.require[1].reason

This binds the warp-cooperative FP8-to-FP32 register product path and its ordered reduction slices without introducing TMEM storage.

## t_cce71cfc78885712 | realization.reason | 133 chars | 2 declarations

root:511 realization.require[0].reason; v0_recovery:462 realization.require[1].reason

Readiness must cover the selected H-row A boxes and full B boxes while preserving untouched rows and the three-stage operand storage.

## t_e6cf77abb47d95ca | realization.reason | 212 chars | 4 declarations

wide:385 realization.prefer[0].reason; wide_recovery:473 realization.prefer[0].reason; wide_recovery_chain:477 realization.prefer[0].reason; wide_recovery_full:483 realization.prefer[0].reason

The host policy targets one row tile per nonempty group, where B has no row-tile reuse. A retained policy can outlive those tensor contents; the hint does not assert absence of reuse during the current execution.

## t_eb3fe1317433618f | realization.reason | 192 chars | 1 declarations

mma_sync:233 realization.require[0].reason

The copy-group window supplies stage readiness; the CTA rendezvous also closes all old readers before shared-slot reuse. Drain must precede reinterpretation of the same storage as BF16 output.

## t_eeeeecea1f5b7e5b | realization.reason | 112 chars | 4 declarations

wide:376 realization.require[2].reason; wide_recovery:463 realization.require[2].reason; wide_recovery_chain:467 realization.require[2].reason; wide_recovery_full:473 realization.require[2].reason

Rotated fragment consumption and early ProductRing release rely on completed TMEM reads, not merely their issue.

## t_2f5550d08a9a2578 | schedule.describe | 92 chars | 2 declarations

root:204 design.work_schedule.phase.main.describe; wide:132 design.work_schedule.phase.main.describe

Compute block_dot and scaled_sum for every grouped output tile and overwrite its owned rows.

## t_8ffbd4ea4e68ed42 | schedule.describe | 306 chars | 1 declarations

v0_recovery:137 design.work_schedule.phase.graph.describe

Compute block_dot and scaled_sum for every grouped output tile and overwrite its owned rows. The graph phase performs evaluation r=0; the direct phase performs evaluation r=1. Each is a complete evaluation with new workers, independent Dot and F values, and no consumption of the other evaluation's output.

## t_bc543ecbf5bd02b7 | schedule.describe | 103 chars | 1 declarations

mma_sync:96 design.work_schedule.phase.main.describe

Compute block_dot and scaled_sum for all grouped output tiles and overwrite their owned output regions.

## t_d6b53592c7372562 | schedule.describe | 113 chars | 3 declarations

wide_recovery:160 design.work_schedule.phase.first.describe; wide_recovery_chain:162 design.work_schedule.phase.first.describe; wide_recovery_full:165 design.work_schedule.phase.first.describe

Compute block_dot and scaled_sum for one evaluation q over every grouped output tile, overwriting its owned rows.

## t_096e91b8f89c94cb | schedule.strategy | 1354 chars | 1 declarations

wide_recovery:219 design.work_schedule.phase.first.schedule.strategy

Each CTA independently copies the execution-time m_indptr and builds prefix once. Producer and epilogue scan groups monotonically from their previous group for each increasing assigned w. Empty groups are skipped. Workers with worker_id >= W do initialization and teardown but execute no worktile. The two launches use the same request stream. first.kernel_done includes all first-grid output writes and worker teardown before retry begins; it does not require a blocking host wait. The grids have distinct worker and local-resource instances even if their mappings are identical. There is no cross-phase data handoff: retry reads the original inputs, reinitializes its local accumulators and overwrites out without reading first's output. The graph for either phase retains its own ParamsW and tensor maps, including its recorded bstream flag; a direct submission uses the current parameters. The grid and activation-map availability also belong to that submission. Required A, output and wide B maps and shared-memory capacity are checked on the host. Reduced-height map failure only disables H reduction. An attempt with no actual computation adds no entry to L; the selection here requires both actual compute submissions. Subsequent API errors do not retract them. Model.forward propagates a nonzero return code without switching to the sync engine.

## t_36a7cbb00e7d7854 | schedule.strategy | 461 chars | 1 declarations

mma_sync:132 design.work_schedule.phase.main.schedule.strategy

Each CTA copies execution-time m_indptr and builds prefix once. For each assigned w it searches from group zero, skipping empty groups, and recovers (g,x,y). Workers with worker_id>=W initialize Schedule but execute no worktiles. Each actual submission has new worker-local state. Repeated calls execute again even on a wrapper memo hit; tensor contents are never memoized. Recorded execution retains its host-bound parameters and reads current device contents.

## t_3e0b03bf36a10955 | schedule.strategy | 1342 chars | 1 declarations

wide_recovery_full:225 design.work_schedule.phase.first.schedule.strategy

Each CTA independently copies the execution-time m_indptr and builds prefix once. Producer and epilogue scan groups monotonically from their previous group for each increasing assigned w. Empty groups are skipped. Workers with worker_id >= W do initialization and teardown but execute no worktile. All four launches use the same request stream. Each predecessor's kernel_done includes its output writes and worker teardown before the next grid begins; no blocking host wait is required. Each grid has distinct worker and local-resource instances even if mappings match. There is no cross-phase data handoff: every phase reads the original inputs, reinitializes its local accumulators and overwrites out without reading a prior phase's output. A graph retains its own ParamsW and tensor maps, including its recorded bstream flag; a direct submission uses the current parameters. The grid and activation-map availability also belong to that submission. Required A, output and wide B maps and shared-memory capacity are checked on the host. Reduced-height map failure only disables H reduction. An attempt with no actual computation adds no entry to L; the selection here requires all four actual compute submissions. Subsequent API errors do not retract them. Model.forward propagates a nonzero return code without switching to the sync engine.

## t_441968a10cab0871 | schedule.strategy | 1078 chars | 1 declarations

wide_recovery_full:168 design.work_schedule.phase.first.decomposition.strategy

Bind q=0 in first (G graph), q=1 in retry (direct G), q=2 in fallback (F graph) and q=3 in final (direct F). Use L[q] and that actual submission's by-value parameters and tensor maps. Tensor bindings and host choices in a graph come from recording or explicit updates; reads of tensor contents occur during this execution. Bm = 128 and Bn = 256. Let prefix(0)=0 and prefix(g+1)=prefix(g)+Tm(g); W=prefix(E)*Tn. The work numbered w has the unique nonempty group g satisfying prefix(g)*Tn <= w < prefix(g+1)*Tn. Let z=w-prefix(g)*Tn; y=floor(z/Tm(g)), x=z%Tm(g). This binds the mathematical worktile (q,g,x,y), including r0,c0,V. The activation extent H is the smallest member of {16,32,64,128} at least V when all three reduced-height tensor maps were obtained; otherwise H=128 for this submission. Only H rows are loaded, but every block product evaluates Bm rows as defined in block_dot. Each local iteration consumes block kb, kb in [0,KB). All scales and all K contributions are applied inside this worktile; no cross-worktile reduction or intermediate global matrix is used.

## t_656fbb69e1c20217 | schedule.strategy | 1019 chars | 1 declarations

wide_recovery_chain:165 design.work_schedule.phase.first.decomposition.strategy

Bind q=0 in first, q=1 in retry and q=2 in final. Use L[q] and that actual submission's by-value parameters and tensor maps. Tensor bindings and host choices in a graph come from recording or explicit updates; reads of tensor contents occur during this execution. Bm = 128 and Bn = 256. Let prefix(0)=0 and prefix(g+1)=prefix(g)+Tm(g); W=prefix(E)*Tn. The work numbered w has the unique nonempty group g satisfying prefix(g)*Tn <= w < prefix(g+1)*Tn. Let z=w-prefix(g)*Tn; y=floor(z/Tm(g)), x=z%Tm(g). This binds the mathematical worktile (q,g,x,y), including r0,c0,V. The activation extent H is the smallest member of {16,32,64,128} at least V when all three reduced-height tensor maps were obtained; otherwise H=128 for this submission. Only H rows are loaded, but every block product evaluates Bm rows as defined in block_dot. Each local iteration consumes block kb, kb in [0,KB). All scales and all K contributions are applied inside this worktile; no cross-worktile reduction or intermediate global matrix is used.

## t_77094d9f9f350f53 | schedule.strategy | 662 chars | 1 declarations

v0_recovery:180 design.work_schedule.phase.graph.schedule.strategy

Each CTA independently copies the execution-time m_indptr and builds prefix once. Producer and epilogue scan groups monotonically from their previous group for each increasing assigned w. Empty groups are skipped. Workers with worker_id >= W do initialization and teardown but execute no worktile. Host policy/cache handling is as defined by dispatch.when. The graph phase uses its captured by-value parameters and tensor maps; direct uses the parameters and maps supplied at the recovery launch. Each device schedule reads current boundaries, and all resource and work indices start in its own worker instance. The first output is not a handoff input to direct.

## t_7f482336dfc7513c | schedule.strategy | 1024 chars | 1 declarations

wide_recovery:163 design.work_schedule.phase.first.decomposition.strategy

Bind q=0 in first and q=1 in retry. Use the specialization L[q] and that actual submission's by-value parameters and tensor maps. Tensor bindings and host choices in a graph come from recording or explicit updates; reads of tensor contents occur during this execution. Bm = 128 and Bn = 256. Let prefix(0)=0 and prefix(g+1)=prefix(g)+Tm(g); W=prefix(E)*Tn. The work numbered w has the unique nonempty group g satisfying prefix(g)*Tn <= w < prefix(g+1)*Tn. Let z=w-prefix(g)*Tn; y=floor(z/Tm(g)), x=z%Tm(g). This binds the mathematical worktile (q,g,x,y), including r0,c0,V. The activation extent H is the smallest member of {16,32,64,128} at least V when all three reduced-height tensor maps were obtained; otherwise H=128 for this submission. Only H rows are loaded, but every block product evaluates Bm rows as defined in block_dot. Each local iteration consumes block kb, kb in [0,KB). All scales and all K contributions are applied inside this worktile; no cross-worktile reduction or intermediate global matrix is used.

## t_845d59005ad6d477 | schedule.strategy | 460 chars | 1 declarations

root:242 design.work_schedule.phase.main.schedule.strategy

Each CTA independently copies the execution-time m_indptr and builds prefix once. Producer and epilogue scan groups monotonically from their previous group for each increasing assigned w. Empty groups are skipped. Workers with worker_id >= W do initialization and teardown but execute no worktile. Host policy/cache handling is as defined by dispatch.when; graph replay retains its recorded scalar parameters while this device schedule uses current boundaries.

## t_86ecb02a91b1391d | schedule.strategy | 473 chars | 1 declarations

mma_sync:98 design.work_schedule.phase.main.decomposition.strategy

Bm=64 and Bn=128. Let prefix(0)=0, prefix(g+1)=prefix(g)+Tm(g), and W=prefix(E)*Tn. Work w belongs to the unique nonempty group g with prefix(g)*Tn <= w < prefix(g+1)*Tn. Let z=w-prefix(g)*Tn; y=floor(z/Tm(g)) and x=z%Tm(g). These coordinates bind r0,c0,V in block_dot. Each worktile traverses kb in [0,KB), one ScaleK block at a time, including all block products and scale accumulation for its output. No split-K work, cross-tile reduction or global intermediate is used.

## t_97dcf8a16cc57016 | schedule.strategy | 753 chars | 1 declarations

wide:134 design.work_schedule.phase.main.decomposition.strategy

Bm = 128 and Bn = 256. Let prefix(0)=0 and prefix(g+1)=prefix(g)+Tm(g); W=prefix(E)*Tn. The work numbered w has the unique nonempty group g satisfying prefix(g)*Tn <= w < prefix(g+1)*Tn. Let z=w-prefix(g)*Tn; y=floor(z/Tm(g)), x=z%Tm(g). This binds the mathematical worktile (g,x,y), including r0,c0,V. The activation extent H is the smallest member of {16,32,64,128} at least V when all three reduced-height tensor maps were obtained; otherwise H=128 for this submission. Only H rows are loaded, but every block product evaluates Bm rows as defined in block_dot. Each local iteration consumes block kb, kb in [0,KB). All scales and all K contributions are applied inside this worktile; no cross-worktile reduction or intermediate global matrix is used.

## t_9feeba0da5a83532 | schedule.strategy | 1344 chars | 1 declarations

wide_recovery_chain:221 design.work_schedule.phase.first.schedule.strategy

Each CTA independently copies the execution-time m_indptr and builds prefix once. Producer and epilogue scan groups monotonically from their previous group for each increasing assigned w. Empty groups are skipped. Workers with worker_id >= W do initialization and teardown but execute no worktile. All three launches use the same request stream. Each predecessor's kernel_done includes its output writes and worker teardown before the next grid begins; no blocking host wait is required. Each grid has distinct worker and local-resource instances even if mappings match. There is no cross-phase data handoff: every phase reads the original inputs, reinitializes its local accumulators and overwrites out without reading a prior phase's output. A graph retains its own ParamsW and tensor maps, including its recorded bstream flag; a direct submission uses the current parameters. The grid and activation-map availability also belong to that submission. Required A, output and wide B maps and shared-memory capacity are checked on the host. Reduced-height map failure only disables H reduction. An attempt with no actual computation adds no entry to L; the selection here requires all three actual compute submissions. Subsequent API errors do not retract them. Model.forward propagates a nonzero return code without switching to the sync engine.

## t_a26f95de8d88b7f9 | schedule.strategy | 246 chars | 4 declarations

wide:163 design.work_schedule.phase.main.mapping.strategy; wide_recovery:211 design.work_schedule.phase.first.mapping.strategy; wide_recovery_chain:213 design.work_schedule.phase.first.mapping.strategy; wide_recovery_full:217 design.work_schedule.phase.first.mapping.strategy

Let Wub=(ceil_div(M,Bm)+E)*Tn and C=max(1,min(Wub,S)), where S is the wrapper's multiprocessor count used for this submission. The kernel uses 320 threads per CTA. C is a submission count, not a claim that all workers are resident simultaneously.

## t_a349bc0c4ccb9bd0 | schedule.strategy | 874 chars | 1 declarations

wide:171 design.work_schedule.phase.main.schedule.strategy

Each CTA independently copies the execution-time m_indptr and builds prefix once. Producer and epilogue scan groups monotonically from their previous group for each increasing assigned w. Empty groups are skipped. Workers with worker_id >= W do initialization and teardown but execute no worktile. Host policy/cache handling is as defined by dispatch.when; graph replay retains its recorded scalar parameters while this device schedule uses current boundaries. This node has one main compute submission and no cross-phase handoff. The host obtains A and output tensor maps, configures shared memory, then obtains the wide B map. Missing required maps or insufficient shared memory return an error before this compute submission; reduced-height map failure only disables H reduction. Model.forward propagates a nonzero kernel return code without switching to the sync engine.

## t_a5e7d840ee7de38a | schedule.strategy | 760 chars | 1 declarations

root:206 design.work_schedule.phase.main.decomposition.strategy

Bm = 128 and Bn = 128. Let prefix(0)=0 and prefix(g+1)=prefix(g)+Tm(g); W=prefix(E)*Tn. The work numbered w has the unique nonempty group g satisfying prefix(g)*Tn <= w < prefix(g+1)*Tn. Let z=w-prefix(g)*Tn; y=floor(z/Tm(g)), x=z%Tm(g). This binds the mathematical worktile (g,x,y), including r0,c0,V. The activation extent H is the smallest member of {16,32,64,128} at least V when all three reduced-height tensor maps were obtained; otherwise H=128 for this submission. Only H rows are loaded, but every block product evaluates Bm rows as defined in block_dot. Each local pair q consumes blocks 2*q and 2*q+1, q in [0,Q). All scales and all K contributions are applied inside this worktile; no cross-worktile reduction or intermediate global matrix is used.

## t_b0f9e0beaccbd3c2 | schedule.strategy | 246 chars | 2 declarations

root:234 design.work_schedule.phase.main.mapping.strategy; v0_recovery:172 design.work_schedule.phase.graph.mapping.strategy

Let Wub=(ceil_div(M,Bm)+E)*Tn and C=max(1,min(Wub,S)), where S is the wrapper's multiprocessor count used for this submission. The kernel uses 256 threads per CTA. C is a submission count, not a claim that all workers are resident simultaneously.

## t_db200f0a6670da25 | schedule.strategy | 762 chars | 1 declarations

v0_recovery:144 design.work_schedule.phase.graph.decomposition.strategy

Bm = 128 and Bn = 128. Let prefix(0)=0 and prefix(g+1)=prefix(g)+Tm(g); W=prefix(E)*Tn. The work numbered w has the unique nonempty group g satisfying prefix(g)*Tn <= w < prefix(g+1)*Tn. Let z=w-prefix(g)*Tn; y=floor(z/Tm(g)), x=z%Tm(g). This binds the mathematical worktile (r,g,x,y), including r0,c0,V. The activation extent H is the smallest member of {16,32,64,128} at least V when all three reduced-height tensor maps were obtained; otherwise H=128 for this submission. Only H rows are loaded, but every block product evaluates Bm rows as defined in block_dot. Each local pair q consumes blocks 2*q and 2*q+1, q in [0,Q). All scales and all K contributions are applied inside this worktile; no cross-worktile reduction or intermediate global matrix is used.

## t_fc5a73485c55b99f | schedule.strategy | 189 chars | 1 declarations

mma_sync:125 design.work_schedule.phase.main.mapping.strategy

Let S be the device-0 multiprocessor count passed by the wrapper, Wub=(ceil_div(M,Bm)+E)*Tn and C=max(1,min(Wub,2*S)). Submit C CTAs of 256 threads; C is not a simultaneous residency claim.

## t_1c42235812b61182 | schedule.thesis | 347 chars | 1 declarations

v0_recovery:125 design.work_schedule.thesis

Submit the complete grouped product twice, with the direct recovery starting after the graph kernel completes. Independent persistent CTA grids perform both output overwrites without a data handoff; group-major, column-then-row work ordering exposes weight reuse within each evaluation, while conservative grid sizing can submit CTAs with no work.

## t_40085957ff7e9fc1 | schedule.thesis | 247 chars | 2 declarations

root:195 design.work_schedule.thesis; wide:126 design.work_schedule.thesis

Assign complete output tiles to a bounded persistent CTA grid, with K reduced locally. Group-major, column-then-row ordering exposes weight reuse; a conservative work upper bound avoids a host read for grid sizing but can submit CTAs with no work.

## t_431e5befda17af91 | schedule.thesis | 275 chars | 1 declarations

wide_recovery_full:151 design.work_schedule.thesis

Submit four independent persistent wide grids on the request stream, each starting after its predecessor completes. Each grid recomputes and overwrites the complete output; recovery adds three full evaluations and writebacks without passing intermediate values between grids.

## t_9db8bdebfa0b8e66 | schedule.thesis | 289 chars | 1 declarations

wide_recovery:150 design.work_schedule.thesis

Submit two independent persistent wide grids on the request stream, waiting for the first kernel to finish before the second begins. Each grid owns and recomputes the complete output; recovery incurs a second full evaluation and writeback without passing intermediate values between grids.

## t_a5d157b54fb5e9af | schedule.thesis | 274 chars | 1 declarations

wide_recovery_chain:150 design.work_schedule.thesis

Submit three independent persistent wide grids on the request stream, each starting after its predecessor completes. Each grid recomputes and overwrites the complete output; recovery adds two full evaluations and writebacks without passing intermediate values between grids.

## t_c2e004b2814c77a3 | schedule.thesis | 272 chars | 1 declarations

mma_sync:88 design.work_schedule.thesis

Give each persistent CTA complete output tiles and keep K reduction local. Group-major, column-then-row numbering offers weight-panel reuse, while a grid capped at twice the supplied SM count avoids host boundary reads for grid sizing but can submit workers with no tiles.
