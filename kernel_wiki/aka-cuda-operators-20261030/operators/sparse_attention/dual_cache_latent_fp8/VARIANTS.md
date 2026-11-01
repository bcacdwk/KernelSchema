# dual_cache_latent_fp8 架构与计算入口索引

按架构查计算/启动文件；输入选路条件见 [runtime.py](candidate/runtime.py)。

variant以kernel/host为界；program负责组合启动和split归约。`kernel_dependency`只提供program链接所需的device符号，没有多余的独立FFI启动器。

## sm_100a

| 入口 | 类型 | 源文件 | 依赖 |
|---|---|---|---|
| `fp8_h128_prefill_source_persistent` | 直接调用的variant | [fp8_h128_prefill_source_persistent_sm_100a_kernel.cu](candidate/fp8_h128_prefill_source_persistent_sm_100a_kernel.cu)、[fp8_h128_prefill_source_persistent_sm_100a_launch.cu](candidate/fp8_h128_prefill_source_persistent_sm_100a_launch.cu) | — |
| `fp8_h128_prefill_source_persistent_uniform` | 直接调用的variant | [fp8_h128_prefill_source_persistent_uniform_sm_100a_kernel.cu](candidate/fp8_h128_prefill_source_persistent_uniform_sm_100a_kernel.cu)、[fp8_h128_prefill_source_persistent_uniform_sm_100a_launch.cu](candidate/fp8_h128_prefill_source_persistent_uniform_sm_100a_launch.cu) | — |
| `fp8_h64_prefill_source_persistent_m64` | 直接调用的variant | [fp8_h64_prefill_source_persistent_m64_sm_100a_kernel.cu](candidate/fp8_h64_prefill_source_persistent_m64_sm_100a_kernel.cu)、[fp8_h64_prefill_source_persistent_m64_sm_100a_launch.cu](candidate/fp8_h64_prefill_source_persistent_m64_sm_100a_launch.cu) | — |
| `fp8_h64_source_exact` | 直接调用的variant | [fp8_h64_source_exact_sm_100a_kernel.cu](candidate/fp8_h64_source_exact_sm_100a_kernel.cu)、[fp8_h64_source_exact_sm_100a_launch.cu](candidate/fp8_h64_source_exact_sm_100a_launch.cu) | — |
| `fp8_h8_h16_source_exact` | 直接调用的variant | [fp8_h8_h16_source_exact_sm_100a_kernel.cu](candidate/fp8_h8_h16_source_exact_sm_100a_kernel.cu)、[fp8_h8_h16_source_exact_sm_100a_launch.cu](candidate/fp8_h8_h16_source_exact_sm_100a_launch.cu) | — |
| `fp8_lowhead_h64` | 直接调用的variant | [fp8_lowhead_h64_sm_100a_kernel.cu](candidate/fp8_lowhead_h64_sm_100a_kernel.cu)、[fp8_lowhead_h64_sm_100a_launch.cu](candidate/fp8_lowhead_h64_sm_100a_launch.cu) | — |
| `fp8_lowhead_one_partition` | 直接调用的variant | [fp8_lowhead_one_partition_sm_100a_kernel.cu](candidate/fp8_lowhead_one_partition_sm_100a_kernel.cu)、[fp8_lowhead_one_partition_sm_100a_launch.cu](candidate/fp8_lowhead_one_partition_sm_100a_launch.cu) | — |
| `fp8_lowhead_prefill` | 直接调用的variant | [fp8_lowhead_prefill_sm_100a_kernel.cu](candidate/fp8_lowhead_prefill_sm_100a_kernel.cu)、[fp8_lowhead_prefill_sm_100a_launch.cu](candidate/fp8_lowhead_prefill_sm_100a_launch.cu) | — |

## sm_103a

| 入口 | 类型 | 源文件 | 依赖 |
|---|---|---|---|
| `fp8_h128_prefill_source_persistent` | 直接调用的variant | [fp8_h128_prefill_source_persistent_sm_103a_kernel.cu](candidate/fp8_h128_prefill_source_persistent_sm_103a_kernel.cu)、[fp8_h128_prefill_source_persistent_sm_103a_launch.cu](candidate/fp8_h128_prefill_source_persistent_sm_103a_launch.cu) | — |
| `fp8_h128_prefill_source_persistent_uniform` | 直接调用的variant | [fp8_h128_prefill_source_persistent_uniform_sm_103a_kernel.cu](candidate/fp8_h128_prefill_source_persistent_uniform_sm_103a_kernel.cu)、[fp8_h128_prefill_source_persistent_uniform_sm_103a_launch.cu](candidate/fp8_h128_prefill_source_persistent_uniform_sm_103a_launch.cu) | — |
| `fp8_h64_prefill_source_persistent_m64` | 直接调用的variant | [fp8_h64_prefill_source_persistent_m64_sm_103a_kernel.cu](candidate/fp8_h64_prefill_source_persistent_m64_sm_103a_kernel.cu)、[fp8_h64_prefill_source_persistent_m64_sm_103a_launch.cu](candidate/fp8_h64_prefill_source_persistent_m64_sm_103a_launch.cu) | — |
| `fp8_h64_source_exact` | 直接调用的variant | [fp8_h64_source_exact_sm_103a_kernel.cu](candidate/fp8_h64_source_exact_sm_103a_kernel.cu)、[fp8_h64_source_exact_sm_103a_launch.cu](candidate/fp8_h64_source_exact_sm_103a_launch.cu) | — |
| `fp8_h8_h16_source_exact` | 直接调用的variant | [fp8_h8_h16_source_exact_sm_103a_kernel.cu](candidate/fp8_h8_h16_source_exact_sm_103a_kernel.cu)、[fp8_h8_h16_source_exact_sm_103a_launch.cu](candidate/fp8_h8_h16_source_exact_sm_103a_launch.cu) | — |
| `fp8_lowhead_h64` | 直接调用的variant | [fp8_lowhead_h64_sm_103a_kernel.cu](candidate/fp8_lowhead_h64_sm_103a_kernel.cu)、[fp8_lowhead_h64_sm_103a_launch.cu](candidate/fp8_lowhead_h64_sm_103a_launch.cu) | — |
| `fp8_lowhead_one_partition` | 直接调用的variant | [fp8_lowhead_one_partition_sm_103a_kernel.cu](candidate/fp8_lowhead_one_partition_sm_103a_kernel.cu)、[fp8_lowhead_one_partition_sm_103a_launch.cu](candidate/fp8_lowhead_one_partition_sm_103a_launch.cu) | — |
| `fp8_lowhead_prefill` | 直接调用的variant | [fp8_lowhead_prefill_sm_103a_kernel.cu](candidate/fp8_lowhead_prefill_sm_103a_kernel.cu)、[fp8_lowhead_prefill_sm_103a_launch.cu](candidate/fp8_lowhead_prefill_sm_103a_launch.cu) | — |
| `fp8_m64_partitioned_v4` | 直接调用的variant | [fp8_m64_partitioned_v4_sm_103a_kernel.cu](candidate/fp8_m64_partitioned_v4_sm_103a_kernel.cu)、[fp8_m64_partitioned_v4_sm_103a_launch.cu](candidate/fp8_m64_partitioned_v4_sm_103a_launch.cu) | — |

