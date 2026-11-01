#pragma once

// ---- include/deep_gemm/impls/sm100_bf16_gemm.cuh ----
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

// ---- include/dg_pure/dg_pure.cuh ----
// dg_pure -- pure CUDA C++ plus inline PTX replacement for the CUTLASS/CuTe
// subset that the pinned DeepGEMM e1f418c2 SM100 GEMM/MoE device closure uses.
//
// Nothing here depends on CUTLASS or CuTe headers, templates or namespaces.
// Every entity is one of:
//   * a std/CUDA C++ alias (type traits, numeric_limits, index_sequence),
//   * a PTX-ISA bitfield/union layout (UMMA descriptors),
//   * one inline PTX instruction sequence (TMA / TMEM / mbarrier / cluster).
//
// Extraction provenance (read-only, pinned in this repository):
//   CUTLASS + CuTe revision f3fde58372d33e9a5650ba7b80fc48b3b49d40c8,
//   BSD-3-Clause (NVIDIA CORPORATION & AFFILIATES). Files consulted:
//     cutlass/detail/helper_macros.hpp   -> DG_HOST_DEVICE / DG_DEVICE / DG_GLOBAL
//     cutlass/cutlass.h                  -> canonical_warp_idx_sync
//     cutlass/arch/barrier.h             -> ClusterBarrier / ClusterTransactionBarrier /
//                                           NamedBarrier / fence_* / umma_arrive*
//     cutlass/arch/memory_sm80.h         -> cp_async_fence / cp_async_wait
//     cutlass/arch/reg_reconfig.h        -> warpgroup_reg_alloc / _dealloc
//     cutlass/arch/grid_dependency_control.h -> launch_dependent_grids
//     cutlass/bfloat16.h, cutlass/float8.h, cutlass/float_subbyte.h -> numeric tags
//     cute/arch/util.hpp                 -> cast_smem_ptr_to_uint
//     cute/arch/cluster_sm90.hpp         -> cluster_* / block_rank_in_cluster / elect_one_sync
//     cute/arch/copy_sm90_desc.hpp       -> TmaDescriptor / prefetch_tma_descriptor / CacheHint*
//     cute/arch/copy_sm90_tma.hpp        -> SM90_TMA_{LOAD,LOAD_MULTICAST,STORE,REDUCE_ADD}_{2D,3D} /
//                                           tma_store_{fence,arrive,wait}
//     cute/arch/copy_sm100_tma.hpp       -> SM100_TMA_2SM_LOAD_{2D,3D} / Sm100MmaPeerBitMask
//     cute/arch/copy_sm100.hpp           -> SM100_TMEM_LOAD_* / SM100_UTCCP_*
//     cute/arch/tmem_allocator_sm100.hpp -> TMEM::Allocator1Sm / Allocator2Sm
//     cute/arch/mma_sm100_desc.hpp       -> UMMA::*
//     cute/numeric/math.hpp              -> min / max
//     cute/util/type_traits.hpp          -> is_same_v / conditional_t / numeric_limits / index_sequence
//   Full license texts accompany this candidate in ../LICENSES/.
//
// Deliberate deviation from the pinned sources (documented, numerics-neutral):
//   SM100_UTCCP_4x32dp128bit_{1cta,2cta}::copy in cute/arch/copy_sm100.hpp of this
//   revision is missing a comma between two asm operand constraints, so the pinned
//   header cannot be instantiated. The comma is restored here; no instruction,
//   operand order or constraint letter is changed.
//
// Correction (2026-10-10): the 16dp256b tcgen05 shims were first written with the
// 32dp32b repetition rule (`.xN` == N registers). For the 16x256b shape `.xN` counts
// 256-bit groups, i.e. 4*N registers, so ptxas rejected them with "Argument 0 of
// instruction 'tcgen05.ld': Argument vector size mismatch" as soon as a kernel took
// the swap-AB epilogue (grouped BF16 GEMM, mega-MoE). The suffixes are now taken
// verbatim from cute/arch/copy_sm100.hpp: LOAD 16dp256b{1,2,4}x -> .x{1,2,4} and
// STORE 16dp256b{1,4}x -> .x{1,4}. Register counts, operand order and constraint
// letters were already correct and are unchanged.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

#if !defined(__CUDACC_RTC__)
#include <cuda.h>
#endif

// --------------------------------------------------------------------------------------
// Architecture feature gates.
//
// These reproduce, in terms of `__CUDA_ARCH__` only, the branch selection that the
// pinned build gets from the upstream `..._ARCH_MMA_SM1xx{A,F}_ENABLED` feature macros,
// which DeepJIT triggers by compiling with
// `-arch=sm_103f` (i.e. `__CUDA_ARCH__ == 1030`, `__CUDA_ARCH_FAMILY_SPECIFIC__`).
// --------------------------------------------------------------------------------------
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#define DG_ARCH_CP_ASYNC_ENABLED 1
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define DG_ARCH_CLUSTER_ENABLED 1
#define DG_ARCH_ELECT_ONE_ENABLED 1
#define DG_ARCH_TMA_SM90_ENABLED 1
#define DG_ARCH_CTA_RECONFIG_ENABLED 1
#define DG_ARCH_GDC_ENABLED 1
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDACC_VER_MAJOR__ >= 12)
#define DG_ARCH_BARRIER_ENABLED 1
#else
#define DG_ARCH_BARRIER_ENABLED 0
#endif

// SM100 / SM101 / SM103 / SM110 (tcgen05 + 2-SM TMA). SM120/SM121 are excluded on
// purpose: they use `shared::cta` TMA and have no tcgen05 tensor memory.
#if defined(__CUDA_ARCH__) && ((__CUDA_ARCH__ == 1000) || (__CUDA_ARCH__ == 1010) || \
                               (__CUDA_ARCH__ == 1030) || (__CUDA_ARCH__ == 1100))
#define DG_ARCH_TCGEN05_ENABLED 1
#define DG_ARCH_TMA_SM100_ENABLED 1
#endif

// SM120/SM121 select the `shared::cta` form of the SM90-style TMA loads.
#if defined(__CUDA_ARCH__) && ((__CUDA_ARCH__ == 1200) || (__CUDA_ARCH__ == 1210))
#define DG_ARCH_TMA_SM120_ENABLED 1
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
#define DG_ARCH_TCGEN_ENABLED 1
#endif

// --------------------------------------------------------------------------------------
// Function qualification macros (replaces cutlass/detail/helper_macros.hpp).
// --------------------------------------------------------------------------------------
#define DG_HOST_DEVICE __host__ __device__
#define DG_DEVICE __device__
#define DG_GLOBAL __global__

#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
#define DG_HOST_DEVICE_NOINLINE __host__ __device__ __noinline__
#define DG_DEVICE_NOINLINE __device__ __noinline__
#elif defined(__CUDACC_RTC__)
#define DG_HOST_DEVICE_NOINLINE __device__ __noinline__
#define DG_DEVICE_NOINLINE __device__ __noinline__
#else
#define DG_HOST_DEVICE_NOINLINE
#define DG_DEVICE_NOINLINE
#endif

// Cache-key token consumed by csrc/runtime/jit.hpp in place of the upstream version macro.
#if !defined(DG_PURE_VERSION)
#define DG_PURE_VERSION 1
#endif

namespace dg {

// --------------------------------------------------------------------------------------
// Type traits and small numeric helpers (replaces cute/util/type_traits.hpp,
// cute/numeric/math.hpp).
// --------------------------------------------------------------------------------------
using std::conditional_t;
using std::index_sequence;
using std::make_index_sequence;
using std::numeric_limits;

template <class A, class B>
inline constexpr bool is_same_v = std::is_same<A, B>::value;

template <class T, class U,
          typename std::enable_if<std::is_arithmetic<T>::value && std::is_arithmetic<U>::value, int>::type = 0>
DG_HOST_DEVICE constexpr auto max(T const& t, U const& u) {
    return t < u ? u : t;
}

template <class T, class U,
          typename std::enable_if<std::is_arithmetic<T>::value && std::is_arithmetic<U>::value, int>::type = 0>
DG_HOST_DEVICE constexpr auto min(T const& t, U const& u) {
    return static_cast<typename std::common_type<T, U>::type>(t) <
           static_cast<typename std::common_type<T, U>::type>(u) ? t : u;
}

template <class T,
          typename std::enable_if<std::is_arithmetic<T>::value, int>::type = 0>
DG_HOST_DEVICE constexpr auto abs(T const& t) {
    if constexpr (std::is_signed<T>::value)
        return t < T(0) ? -t : t;
    else
        return t;
}

// `dg::get` backs the DG_TIE_* macro family in deep_gemm/common/dg_tie.cuh.
// Only std tuple-like operands are reachable from the SM100 GEMM/MoE closure.
template <size_t kIndex, class T>
DG_HOST_DEVICE constexpr auto get(T&& value) -> decltype(std::get<kIndex>(std::forward<T>(value))) {
    return std::get<kIndex>(std::forward<T>(value));
}

// --------------------------------------------------------------------------------------
// Numeric tag types (replaces cutlass/bfloat16.h, cutlass/float8.h,
// cutlass/float_subbyte.h). Sizes, alignments and stored payloads match the pinned
// CUTLASS types so that every `sizeof`/TMA/swizzle computation is unchanged.
// --------------------------------------------------------------------------------------
struct alignas(2) bfloat16_t {
    uint16_t storage;

    DG_HOST_DEVICE bfloat16_t() = default;
    DG_HOST_DEVICE explicit bfloat16_t(const uint16_t& raw): storage(raw) {}
    DG_HOST_DEVICE bfloat16_t(const float& value): storage(bitcast(__float2bfloat16(value))) {}
    DG_HOST_DEVICE operator float() const { return __bfloat162float(bitcast_raw()); }
    DG_HOST_DEVICE __nv_bfloat16 bitcast_raw() const {
        __nv_bfloat16 raw;
        *reinterpret_cast<uint16_t*>(&raw) = storage;
        return raw;
    }
    DG_HOST_DEVICE static uint16_t bitcast(const __nv_bfloat16& raw) {
        return *reinterpret_cast<const uint16_t*>(&raw);
    }
};

static_assert(sizeof(bfloat16_t) == 2, "bfloat16_t must stay a 2-byte type");

using half_t = __half;

// 4-byte tag; only reachable through tf32 kernels that are outside this bundle.
struct alignas(4) tfloat32_t {
    uint32_t storage;
    DG_HOST_DEVICE tfloat32_t() = default;
    DG_HOST_DEVICE explicit tfloat32_t(const uint32_t& raw): storage(raw) {}
};

struct alignas(1) float_e4m3_t {
    uint8_t storage;

    DG_HOST_DEVICE float_e4m3_t() = default;
    DG_HOST_DEVICE explicit float_e4m3_t(const uint8_t& raw): storage(raw) {}
    DG_HOST_DEVICE float_e4m3_t(const float& value)
        : storage(__nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3)) {}
    DG_HOST_DEVICE operator float() const {
        const __half_raw raw = __nv_cvt_fp8_to_halfraw(storage, __NV_E4M3);
        return __half2float(*reinterpret_cast<const __half*>(&raw));
    }
};

struct alignas(1) float_e5m2_t {
    uint8_t storage;

    DG_HOST_DEVICE float_e5m2_t() = default;
    DG_HOST_DEVICE explicit float_e5m2_t(const uint8_t& raw): storage(raw) {}
    DG_HOST_DEVICE float_e5m2_t(const float& value)
        : storage(__nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E5M2)) {}
    DG_HOST_DEVICE operator float() const {
        const __half_raw raw = __nv_cvt_fp8_to_halfraw(storage, __NV_E5M2);
        return __half2float(*reinterpret_cast<const __half*>(&raw));
    }
};

// Sub-byte / scale-factor payloads travel through TMA and UMMA as opaque bytes: the
// kernels never convert them on the host or in registers, so these stay storage tags.
struct alignas(1) float_e2m1_t {
    uint8_t storage;
    DG_HOST_DEVICE float_e2m1_t() = default;
    DG_HOST_DEVICE explicit float_e2m1_t(const uint8_t& raw): storage(raw) {}
};

struct alignas(1) float_ue8m0_t {
    uint8_t storage;
    DG_HOST_DEVICE float_ue8m0_t() = default;
    DG_HOST_DEVICE explicit float_ue8m0_t(const uint8_t& raw): storage(raw) {}
};

struct alignas(1) float_ue4m3_t {
    uint8_t storage;
    DG_HOST_DEVICE float_ue4m3_t() = default;
    DG_HOST_DEVICE explicit float_ue4m3_t(const uint8_t& raw): storage(raw) {}
};

namespace detail {

// FP4 kept unpacked in shared memory: one byte per logical element, MMA still reads E2M1.
struct alignas(1) float_e2m1_unpacksmem_t {
    uint8_t storage;
    DG_HOST_DEVICE float_e2m1_unpacksmem_t() = default;
    DG_HOST_DEVICE explicit float_e2m1_unpacksmem_t(const uint8_t& raw): storage(raw) {}
};

struct alignas(1) float_e2m3_unpacksmem_t {
    uint8_t storage;
};

struct alignas(1) float_e3m2_unpacksmem_t {
    uint8_t storage;
};

} // namespace detail

static_assert(sizeof(float_e4m3_t) == 1 and sizeof(float_e2m1_t) == 1 and
              sizeof(float_ue8m0_t) == 1 and sizeof(detail::float_e2m1_unpacksmem_t) == 1,
              "Low-precision tags must stay 1-byte types");

// --------------------------------------------------------------------------------------
// Warp / CTA / cluster helpers.
// --------------------------------------------------------------------------------------
DG_HOST_DEVICE inline uint32_t cast_smem_ptr_to_uint(void const* const ptr) {
#if defined(__CUDA_ARCH__)
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
#else
    (void) ptr;
    return 0;
#endif
}

DG_HOST_DEVICE inline uint32_t elect_one_sync() {
#if defined(DG_ARCH_ELECT_ONE_ENABLED)
    uint32_t pred = 0;
    uint32_t laneid = 0;
    asm volatile(
        "{\n"
        ".reg .b32 %%rx;\n"
        ".reg .pred %%px;\n"
        "     elect.sync %%rx|%%px, %2;\n"
        "@%%px mov.s32 %1, 1;\n"
        "     mov.s32 %0, %%rx;\n"
        "}\n"
        : "+r"(laneid), "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
#elif defined(__CUDA_ARCH__)
    return (threadIdx.x % 32) == 0;
#else
    return 1;
#endif
}

DG_DEVICE inline uint32_t block_rank_in_cluster() {
#if defined(DG_ARCH_CLUSTER_ENABLED)
    uint32_t rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rank) :);
    return rank;
#else
    return 0;
#endif
}

DG_DEVICE inline dim3 cluster_id_in_grid() {
#if defined(DG_ARCH_CLUSTER_ENABLED)
    uint32_t x, y, z;
    asm volatile("mov.u32 %0, %%clusterid.x;\n" : "=r"(x) :);
    asm volatile("mov.u32 %0, %%clusterid.y;\n" : "=r"(y) :);
    asm volatile("mov.u32 %0, %%clusterid.z;\n" : "=r"(z) :);
    return {x, y, z};
#elif defined(__CUDA_ARCH__)
    return blockIdx;
#else
    return {0, 0, 0};
#endif
}

DG_DEVICE inline void cluster_arrive_relaxed() {
#if defined(DG_ARCH_CLUSTER_ENABLED)
    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" : : );
#endif
}

DG_DEVICE inline void cluster_arrive() {
#if defined(DG_ARCH_CLUSTER_ENABLED)
    asm volatile("barrier.cluster.arrive.aligned;\n" : : );
#endif
}

DG_DEVICE inline void cluster_wait() {
#if defined(DG_ARCH_CLUSTER_ENABLED)
    asm volatile("barrier.cluster.wait.aligned;\n" : : );
#endif
}

DG_DEVICE inline void cluster_sync() {
    cluster_arrive();
    cluster_wait();
}

// `NumThreadsPerWarp` in cutlass/cutlass.h.
static constexpr int kNumThreadsPerWarp = 32;

DG_DEVICE inline int canonical_warp_idx_sync() {
#if defined(__CUDA_ARCH__)
    return __shfl_sync(0xffffffff, threadIdx.x / kNumThreadsPerWarp, 0);
#else
    return 0;
#endif
}

DG_DEVICE inline int canonical_lane_idx() {
#if defined(__CUDA_ARCH__)
    return threadIdx.x % kNumThreadsPerWarp;
#else
    return 0;
#endif
}

// --------------------------------------------------------------------------------------
// TMA descriptors and cache hints (cute/arch/copy_sm90_desc.hpp).
// --------------------------------------------------------------------------------------
#if (__CUDACC_VER_MAJOR__ >= 12) && !defined(__CUDACC_RTC__)
using TmaDescriptor = CUtensorMap;
#else
using TmaDescriptor = struct alignas(64) { char bytes[128]; };
#endif

namespace TMA {

enum class CacheHintSm90 : uint64_t {
    EVICT_NORMAL = 0x1000000000000000,
    EVICT_FIRST  = 0x12F0000000000000,
    EVICT_LAST   = 0x14F0000000000000,
};

enum class CacheHintSm100 : uint64_t {
    EVICT_NORMAL = 0x1000000000000000,
    EVICT_FIRST  = 0x12F0000000000000,
    EVICT_LAST   = 0x14F0000000000000,
};

} // namespace TMA

DG_HOST_DEVICE inline void prefetch_tma_descriptor(TmaDescriptor const* desc_ptr) {
#if defined(DG_ARCH_TMA_SM90_ENABLED)
    const uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    asm volatile("prefetch.tensormap [%0];" : : "l"(gmem_int_desc) : "memory");
#endif
}

// --------------------------------------------------------------------------------------
// SM90-style TMA loads / stores / reductions (cute/arch/copy_sm90_tma.hpp).
// --------------------------------------------------------------------------------------
struct SM90_TMA_LOAD_2D {
    DG_HOST_DEVICE static void copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
                                    void* smem_ptr, int32_t const& crd0, int32_t const& crd1) {
#if defined(DG_ARCH_TMA_SM90_ENABLED)
        const uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
        const uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
        const uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
#if defined(DG_ARCH_TMA_SM120_ENABLED)
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
            " [%0], [%1, {%3, %4}], [%2], %5;"
            :
            : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
              "r"(crd0), "r"(crd1), "l"(cache_hint)
            : "memory");
#else
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
            " [%0], [%1, {%3, %4}], [%2], %5;"
            :
            : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
              "r"(crd0), "r"(crd1), "l"(cache_hint)
            : "memory");
#endif
#endif
    }
};

struct SM90_TMA_LOAD_3D {
    DG_HOST_DEVICE static void copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
                                    void* smem_ptr, int32_t const& crd0, int32_t const& crd1,
                                    int32_t const& crd2) {
#if defined(DG_ARCH_TMA_SM90_ENABLED)
        const uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
        const uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
        const uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
#if defined(DG_ARCH_TMA_SM120_ENABLED)
        asm volatile(
            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
            " [%0], [%1, {%3, %4, %5}], [%2], %6;"
            :
            : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
              "r"(crd0), "r"(crd1), "r"(crd2), "l"(cache_hint)
            : "memory");
#else
        asm volatile(
            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
            " [%0], [%1, {%3, %4, %5}], [%2], %6;"
            :
            : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
              "r"(crd0), "r"(crd1), "r"(crd2), "l"(cache_hint)
            : "memory");
#endif
#endif
    }
};

struct SM90_TMA_LOAD_MULTICAST_2D {
    DG_HOST_DEVICE static void copy(void const* desc_ptr, uint64_t* mbar_ptr, uint16_t multicast_mask,
                                    uint64_t cache_hint, void* smem_ptr,
                                    int32_t const& crd0, int32_t const& crd1) {
#if defined(DG_ARCH_TMA_SM90_ENABLED) && !defined(DG_ARCH_TMA_SM120_ENABLED)
        const uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
        const uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
        const uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
            " [%0], [%1, {%4, %5}], [%2], %3, %6;"
            :
            : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
              "h"(multicast_mask), "r"(crd0), "r"(crd1), "l"(cache_hint)
            : "memory");
#endif
    }
};

struct SM90_TMA_LOAD_MULTICAST_3D {
    DG_HOST_DEVICE static void copy(void const* desc_ptr, uint64_t* mbar_ptr, uint16_t multicast_mask,
                                    uint64_t cache_hint, void* smem_ptr,
                                    int32_t const& crd0, int32_t const& crd1, int32_t const& crd2) {
#if defined(DG_ARCH_TMA_SM90_ENABLED) && !defined(DG_ARCH_TMA_SM120_ENABLED)
        const uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
        const uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
        const uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
        asm volatile(
            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
            " [%0], [%1, {%4, %5, %6}], [%2], %3, %7;"
            :
            : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
              "h"(multicast_mask), "r"(crd0), "r"(crd1), "r"(crd2), "l"(cache_hint)
            : "memory");
#endif
    }
};

struct SM90_TMA_STORE_2D {
    DG_HOST_DEVICE static void copy(void const* desc_ptr, void const* smem_ptr,
                                    int32_t const& crd0, int32_t const& crd1) {
#if defined(DG_ARCH_TMA_SM90_ENABLED)
        const uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
        const uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
        asm volatile(
            "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
            :
            : "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0), "r"(crd1)
            : "memory");
#endif
    }
};

struct SM90_TMA_STORE_3D {
    DG_HOST_DEVICE static void copy(void const* desc_ptr, void const* smem_ptr,
                                    int32_t const& crd0, int32_t const& crd1, int32_t const& crd2) {
#if defined(DG_ARCH_TMA_SM90_ENABLED)
        const uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
        const uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
        asm volatile(
            "cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%2, %3, %4}], [%1];"
            :
            : "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0), "r"(crd1), "r"(crd2)
            : "memory");
#endif
    }
};

struct SM90_TMA_REDUCE_ADD_2D {
    DG_HOST_DEVICE static void copy(void const* const desc_ptr, void const* const smem_ptr,
                                    int32_t const& crd0, int32_t const& crd1) {
#if defined(DG_ARCH_TMA_SM90_ENABLED)
        const uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
        const uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
        asm volatile(
            "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group [%0, {%2, %3}], [%1];"
            :
            : "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0), "r"(crd1)
            : "memory");
#endif
    }
};

struct SM90_TMA_REDUCE_ADD_3D {
    DG_HOST_DEVICE static void copy(void const* const desc_ptr, void const* const smem_ptr,
                                    int32_t const& crd0, int32_t const& crd1, int32_t const& crd2) {
#if defined(DG_ARCH_TMA_SM90_ENABLED)
        const uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
        const uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
        asm volatile(
            "cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.bulk_group [%0, {%2, %3, %4}], [%1];"
            :
            : "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0), "r"(crd1), "r"(crd2)
            : "memory");
#endif
    }
};

DG_HOST_DEVICE inline void tma_store_fence() {
#if defined(DG_ARCH_TMA_SM90_ENABLED)
    asm volatile("fence.proxy.async.shared::cta;");
#endif
}

DG_HOST_DEVICE inline void tma_store_arrive() {
#if defined(DG_ARCH_TMA_SM90_ENABLED)
    asm volatile("cp.async.bulk.commit_group;");
#endif
}

template <int Count>
DG_HOST_DEVICE inline void tma_store_wait() {
#if defined(DG_ARCH_TMA_SM90_ENABLED)
    asm volatile("cp.async.bulk.wait_group.read %0;" : : "n"(Count) : "memory");
#endif
}

DG_HOST_DEVICE inline void tma_desc_commit_group() {
#if defined(DG_ARCH_TMA_SM90_ENABLED)
    asm volatile("cp.async.bulk.commit_group;");
#endif
}

DG_HOST_DEVICE inline void tma_desc_wait_group() {
#if defined(DG_ARCH_TMA_SM90_ENABLED)
    asm volatile("cp.async.bulk.wait_group.read %0;" : : "n"(0) : "memory");
#endif
}

// --------------------------------------------------------------------------------------
// SM100 2-SM TMA loads (cute/arch/copy_sm100_tma.hpp).
// --------------------------------------------------------------------------------------
// Clears the CTA-pair "peer" bit so the transaction bytes land on CTA0's mbarrier.
static constexpr uint32_t Sm100MmaPeerBitMask = 0xFEFFFFFFu;

struct SM100_TMA_2SM_LOAD_2D {
    DG_HOST_DEVICE static void copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
                                    void* smem_ptr, int32_t const& crd0, int32_t const& crd1) {
#if defined(DG_ARCH_TMA_SM100_ENABLED)
        const uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
        const uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr) & Sm100MmaPeerBitMask;
        const uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
        asm volatile(
            "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
            " [%0], [%1, {%3, %4}], [%2], %5;"
            :
            : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
              "r"(crd0), "r"(crd1), "l"(cache_hint)
            : "memory");
#endif
    }
};

struct SM100_TMA_2SM_LOAD_3D {
    DG_HOST_DEVICE static void copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
                                    void* smem_ptr, int32_t const& crd0, int32_t const& crd1,
                                    int32_t const& crd2) {
#if defined(DG_ARCH_TMA_SM100_ENABLED)
        const uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
        const uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr) & Sm100MmaPeerBitMask;
        const uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
        asm volatile(
            "cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
            " [%0], [%1, {%3, %4, %5}], [%2], %6;"
            :
            : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
              "r"(crd0), "r"(crd1), "r"(crd2), "l"(cache_hint)
            : "memory");
#endif
    }
};

// --------------------------------------------------------------------------------------
// Tensor-memory loads, stores and copies (cute/arch/copy_sm100.hpp).
// Generated explicitly (no variadic macros) so the emitted PTX is literally the pinned
// instruction: `tcgen05.{ld,st}.sync.aligned.<shape>.x<N>.b32`.
// --------------------------------------------------------------------------------------
struct SM100_TMEM_LOAD_32dp32b4x {
    using SRegisters = uint32_t[1];
    using DRegisters = uint32_t[4];

    DG_HOST_DEVICE static void copy(uint32_t const& src_addr, uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3) {
#if defined(DG_ARCH_TCGEN05_ENABLED)
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0, %1, %2, %3}, [%4];"
            : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
            : "r"(src_addr));
#endif
    }
};
struct SM100_TMEM_LOAD_32dp32b8x {
    using SRegisters = uint32_t[1];
    using DRegisters = uint32_t[8];

    DG_HOST_DEVICE static void copy(uint32_t const& src_addr, uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3, uint32_t& dst4, uint32_t& dst5, uint32_t& dst6, uint32_t& dst7) {
#if defined(DG_ARCH_TCGEN05_ENABLED)
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3), "=r"(dst4), "=r"(dst5), "=r"(dst6), "=r"(dst7)
            : "r"(src_addr));
#endif
    }
};
struct SM100_TMEM_LOAD_32dp32b16x {
    using SRegisters = uint32_t[1];
    using DRegisters = uint32_t[16];

    DG_HOST_DEVICE static void copy(uint32_t const& src_addr, uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3, uint32_t& dst4, uint32_t& dst5, uint32_t& dst6, uint32_t& dst7, uint32_t& dst8, uint32_t& dst9, uint32_t& dst10, uint32_t& dst11, uint32_t& dst12, uint32_t& dst13, uint32_t& dst14, uint32_t& dst15) {
#if defined(DG_ARCH_TCGEN05_ENABLED)
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
            : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3), "=r"(dst4), "=r"(dst5), "=r"(dst6), "=r"(dst7), "=r"(dst8), "=r"(dst9), "=r"(dst10), "=r"(dst11), "=r"(dst12), "=r"(dst13), "=r"(dst14), "=r"(dst15)
            : "r"(src_addr));
#endif
    }
};
struct SM100_TMEM_LOAD_32dp32b32x {
    using SRegisters = uint32_t[1];
    using DRegisters = uint32_t[32];

    DG_HOST_DEVICE static void copy(uint32_t const& src_addr, uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3, uint32_t& dst4, uint32_t& dst5, uint32_t& dst6, uint32_t& dst7, uint32_t& dst8, uint32_t& dst9, uint32_t& dst10, uint32_t& dst11, uint32_t& dst12, uint32_t& dst13, uint32_t& dst14, uint32_t& dst15, uint32_t& dst16, uint32_t& dst17, uint32_t& dst18, uint32_t& dst19, uint32_t& dst20, uint32_t& dst21, uint32_t& dst22, uint32_t& dst23, uint32_t& dst24, uint32_t& dst25, uint32_t& dst26, uint32_t& dst27, uint32_t& dst28, uint32_t& dst29, uint32_t& dst30, uint32_t& dst31) {
#if defined(DG_ARCH_TCGEN05_ENABLED)
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
            : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3), "=r"(dst4), "=r"(dst5), "=r"(dst6), "=r"(dst7), "=r"(dst8), "=r"(dst9), "=r"(dst10), "=r"(dst11), "=r"(dst12), "=r"(dst13), "=r"(dst14), "=r"(dst15), "=r"(dst16), "=r"(dst17), "=r"(dst18), "=r"(dst19), "=r"(dst20), "=r"(dst21), "=r"(dst22), "=r"(dst23), "=r"(dst24), "=r"(dst25), "=r"(dst26), "=r"(dst27), "=r"(dst28), "=r"(dst29), "=r"(dst30), "=r"(dst31)
            : "r"(src_addr));
#endif
    }
};
struct SM100_TMEM_LOAD_32dp32b64x {
    using SRegisters = uint32_t[1];
    using DRegisters = uint32_t[64];

    DG_HOST_DEVICE static void copy(uint32_t const& src_addr, uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3, uint32_t& dst4, uint32_t& dst5, uint32_t& dst6, uint32_t& dst7, uint32_t& dst8, uint32_t& dst9, uint32_t& dst10, uint32_t& dst11, uint32_t& dst12, uint32_t& dst13, uint32_t& dst14, uint32_t& dst15, uint32_t& dst16, uint32_t& dst17, uint32_t& dst18, uint32_t& dst19, uint32_t& dst20, uint32_t& dst21, uint32_t& dst22, uint32_t& dst23, uint32_t& dst24, uint32_t& dst25, uint32_t& dst26, uint32_t& dst27, uint32_t& dst28, uint32_t& dst29, uint32_t& dst30, uint32_t& dst31, uint32_t& dst32, uint32_t& dst33, uint32_t& dst34, uint32_t& dst35, uint32_t& dst36, uint32_t& dst37, uint32_t& dst38, uint32_t& dst39, uint32_t& dst40, uint32_t& dst41, uint32_t& dst42, uint32_t& dst43, uint32_t& dst44, uint32_t& dst45, uint32_t& dst46, uint32_t& dst47, uint32_t& dst48, uint32_t& dst49, uint32_t& dst50, uint32_t& dst51, uint32_t& dst52, uint32_t& dst53, uint32_t& dst54, uint32_t& dst55, uint32_t& dst56, uint32_t& dst57, uint32_t& dst58, uint32_t& dst59, uint32_t& dst60, uint32_t& dst61, uint32_t& dst62, uint32_t& dst63) {
#if defined(DG_ARCH_TCGEN05_ENABLED)
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x64.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
            : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3), "=r"(dst4), "=r"(dst5), "=r"(dst6), "=r"(dst7), "=r"(dst8), "=r"(dst9), "=r"(dst10), "=r"(dst11), "=r"(dst12), "=r"(dst13), "=r"(dst14), "=r"(dst15), "=r"(dst16), "=r"(dst17), "=r"(dst18), "=r"(dst19), "=r"(dst20), "=r"(dst21), "=r"(dst22), "=r"(dst23), "=r"(dst24), "=r"(dst25), "=r"(dst26), "=r"(dst27), "=r"(dst28), "=r"(dst29), "=r"(dst30), "=r"(dst31), "=r"(dst32), "=r"(dst33), "=r"(dst34), "=r"(dst35), "=r"(dst36), "=r"(dst37), "=r"(dst38), "=r"(dst39), "=r"(dst40), "=r"(dst41), "=r"(dst42), "=r"(dst43), "=r"(dst44), "=r"(dst45), "=r"(dst46), "=r"(dst47), "=r"(dst48), "=r"(dst49), "=r"(dst50), "=r"(dst51), "=r"(dst52), "=r"(dst53), "=r"(dst54), "=r"(dst55), "=r"(dst56), "=r"(dst57), "=r"(dst58), "=r"(dst59), "=r"(dst60), "=r"(dst61), "=r"(dst62), "=r"(dst63)
            : "r"(src_addr));
#endif
    }
};
struct SM100_TMEM_LOAD_32dp32b128x {
    using SRegisters = uint32_t[1];
    using DRegisters = uint32_t[128];

    DG_HOST_DEVICE static void copy(uint32_t const& src_addr, uint32_t& dst000, uint32_t& dst001, uint32_t& dst002, uint32_t& dst003, uint32_t& dst004, uint32_t& dst005, uint32_t& dst006, uint32_t& dst007, uint32_t& dst008, uint32_t& dst009, uint32_t& dst010, uint32_t& dst011, uint32_t& dst012, uint32_t& dst013, uint32_t& dst014, uint32_t& dst015, uint32_t& dst016, uint32_t& dst017, uint32_t& dst018, uint32_t& dst019, uint32_t& dst020, uint32_t& dst021, uint32_t& dst022, uint32_t& dst023, uint32_t& dst024, uint32_t& dst025, uint32_t& dst026, uint32_t& dst027, uint32_t& dst028, uint32_t& dst029, uint32_t& dst030, uint32_t& dst031, uint32_t& dst032, uint32_t& dst033, uint32_t& dst034, uint32_t& dst035, uint32_t& dst036, uint32_t& dst037, uint32_t& dst038, uint32_t& dst039, uint32_t& dst040, uint32_t& dst041, uint32_t& dst042, uint32_t& dst043, uint32_t& dst044, uint32_t& dst045, uint32_t& dst046, uint32_t& dst047, uint32_t& dst048, uint32_t& dst049, uint32_t& dst050, uint32_t& dst051, uint32_t& dst052, uint32_t& dst053, uint32_t& dst054, uint32_t& dst055, uint32_t& dst056, uint32_t& dst057, uint32_t& dst058, uint32_t& dst059, uint32_t& dst060, uint32_t& dst061, uint32_t& dst062, uint32_t& dst063, uint32_t& dst064, uint32_t& dst065, uint32_t& dst066, uint32_t& dst067, uint32_t& dst068, uint32_t& dst069, uint32_t& dst070, uint32_t& dst071, uint32_t& dst072, uint32_t& dst073, uint32_t& dst074, uint32_t& dst075, uint32_t& dst076, uint32_t& dst077, uint32_t& dst078, uint32_t& dst079, uint32_t& dst080, uint32_t& dst081, uint32_t& dst082, uint32_t& dst083, uint32_t& dst084, uint32_t& dst085, uint32_t& dst086, uint32_t& dst087, uint32_t& dst088, uint32_t& dst089, uint32_t& dst090, uint32_t& dst091, uint32_t& dst092, uint32_t& dst093, uint32_t& dst094, uint32_t& dst095, uint32_t& dst096, uint32_t& dst097, uint32_t& dst098, uint32_t& dst099, uint32_t& dst100, uint32_t& dst101, uint32_t& dst102, uint32_t& dst103, uint32_t& dst104, uint32_t& dst105, uint32_t& dst106, uint32_t& dst107, uint32_t& dst108, uint32_t& dst109, uint32_t& dst110, uint32_t& dst111, uint32_t& dst112, uint32_t& dst113, uint32_t& dst114, uint32_t& dst115, uint32_t& dst116, uint32_t& dst117, uint32_t& dst118, uint32_t& dst119, uint32_t& dst120, uint32_t& dst121, uint32_t& dst122, uint32_t& dst123, uint32_t& dst124, uint32_t& dst125, uint32_t& dst126, uint32_t& dst127) {
#if defined(DG_ARCH_TCGEN05_ENABLED)
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x128.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
            : "=r"(dst000), "=r"(dst001), "=r"(dst002), "=r"(dst003), "=r"(dst004), "=r"(dst005), "=r"(dst006), "=r"(dst007), "=r"(dst008), "=r"(dst009), "=r"(dst010), "=r"(dst011), "=r"(dst012), "=r"(dst013), "=r"(dst014), "=r"(dst015), "=r"(dst016), "=r"(dst017), "=r"(dst018), "=r"(dst019), "=r"(dst020), "=r"(dst021), "=r"(dst022), "=r"(dst023), "=r"(dst024), "=r"(dst025), "=r"(dst026), "=r"(dst027), "=r"(dst028), "=r"(dst029), "=r"(dst030), "=r"(dst031), "=r"(dst032), "=r"(dst033), "=r"(dst034), "=r"(dst035), "=r"(dst036), "=r"(dst037), "=r"(dst038), "=r"(dst039), "=r"(dst040), "=r"(dst041), "=r"(dst042), "=r"(dst043), "=r"(dst044), "=r"(dst045), "=r"(dst046), "=r"(dst047), "=r"(dst048), "=r"(dst049), "=r"(dst050), "=r"(dst051), "=r"(dst052), "=r"(dst053), "=r"(dst054), "=r"(dst055), "=r"(dst056), "=r"(dst057), "=r"(dst058), "=r"(dst059), "=r"(dst060), "=r"(dst061), "=r"(dst062), "=r"(dst063), "=r"(dst064), "=r"(dst065), "=r"(dst066), "=r"(dst067), "=r"(dst068), "=r"(dst069), "=r"(dst070), "=r"(dst071), "=r"(dst072), "=r"(dst073), "=r"(dst074), "=r"(dst075), "=r"(dst076), "=r"(dst077), "=r"(dst078), "=r"(dst079), "=r"(dst080), "=r"(dst081), "=r"(dst082), "=r"(dst083), "=r"(dst084), "=r"(dst085), "=r"(dst086), "=r"(dst087), "=r"(dst088), "=r"(dst089), "=r"(dst090), "=r"(dst091), "=r"(dst092), "=r"(dst093), "=r"(dst094), "=r"(dst095), "=r"(dst096), "=r"(dst097), "=r"(dst098), "=r"(dst099), "=r"(dst100), "=r"(dst101), "=r"(dst102), "=r"(dst103), "=r"(dst104), "=r"(dst105), "=r"(dst106), "=r"(dst107), "=r"(dst108), "=r"(dst109), "=r"(dst110), "=r"(dst111), "=r"(dst112), "=r"(dst113), "=r"(dst114), "=r"(dst115), "=r"(dst116), "=r"(dst117), "=r"(dst118), "=r"(dst119), "=r"(dst120), "=r"(dst121), "=r"(dst122), "=r"(dst123), "=r"(dst124), "=r"(dst125), "=r"(dst126), "=r"(dst127)
            : "r"(src_addr));
#endif
    }
};
struct SM100_TMEM_LOAD_16dp256b1x {
    using SRegisters = uint32_t[1];
    using DRegisters = uint32_t[4];

    DG_HOST_DEVICE static void copy(uint32_t const& src_addr, uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3) {
#if defined(DG_ARCH_TCGEN05_ENABLED)
        asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0, %1, %2, %3}, [%4];"
            : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
            : "r"(src_addr));
#endif
    }
};
struct SM100_TMEM_LOAD_16dp256b2x {
    using SRegisters = uint32_t[1];
    using DRegisters = uint32_t[8];

    DG_HOST_DEVICE static void copy(uint32_t const& src_addr, uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3, uint32_t& dst4, uint32_t& dst5, uint32_t& dst6, uint32_t& dst7) {
#if defined(DG_ARCH_TCGEN05_ENABLED)
        asm volatile("tcgen05.ld.sync.aligned.16x256b.x2.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3), "=r"(dst4), "=r"(dst5), "=r"(dst6), "=r"(dst7)
            : "r"(src_addr));
#endif
    }
};
struct SM100_TMEM_LOAD_16dp256b4x {
    using SRegisters = uint32_t[1];
    using DRegisters = uint32_t[16];

    DG_HOST_DEVICE static void copy(uint32_t const& src_addr, uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3, uint32_t& dst4, uint32_t& dst5, uint32_t& dst6, uint32_t& dst7, uint32_t& dst8, uint32_t& dst9, uint32_t& dst10, uint32_t& dst11, uint32_t& dst12, uint32_t& dst13, uint32_t& dst14, uint32_t& dst15) {
#if defined(DG_ARCH_TCGEN05_ENABLED)
        asm volatile("tcgen05.ld.sync.aligned.16x256b.x4.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
            : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3), "=r"(dst4), "=r"(dst5), "=r"(dst6), "=r"(dst7), "=r"(dst8), "=r"(dst9), "=r"(dst10), "=r"(dst11), "=r"(dst12), "=r"(dst13), "=r"(dst14), "=r"(dst15)
            : "r"(src_addr));
#endif
    }
};
struct SM100_TMEM_STORE_16dp256b1x {
    using SRegisters = uint32_t[4];
    using DRegisters = uint32_t[1];

    DG_HOST_DEVICE static void copy(uint32_t const& src00, uint32_t const& src01, uint32_t const& src02, uint32_t const& src03, uint32_t const& dst_addr) {
#if defined(DG_ARCH_TCGEN05_ENABLED)
        asm volatile("tcgen05.st.sync.aligned.16x256b.x1.b32 [%0], {%1, %2, %3, %4};"
            :
            : "r"(dst_addr), "r"(src00), "r"(src01), "r"(src02), "r"(src03));
#endif
    }
};
struct SM100_TMEM_STORE_16dp256b4x {
    using SRegisters = uint32_t[16];
    using DRegisters = uint32_t[1];

    DG_HOST_DEVICE static void copy(uint32_t const& src00, uint32_t const& src01, uint32_t const& src02, uint32_t const& src03, uint32_t const& src04, uint32_t const& src05, uint32_t const& src06, uint32_t const& src07, uint32_t const& src08, uint32_t const& src09, uint32_t const& src10, uint32_t const& src11, uint32_t const& src12, uint32_t const& src13, uint32_t const& src14, uint32_t const& src15, uint32_t const& dst_addr) {
#if defined(DG_ARCH_TCGEN05_ENABLED)
        asm volatile("tcgen05.st.sync.aligned.16x256b.x4.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
            :
            : "r"(dst_addr), "r"(src00), "r"(src01), "r"(src02), "r"(src03), "r"(src04), "r"(src05), "r"(src06), "r"(src07), "r"(src08), "r"(src09), "r"(src10), "r"(src11), "r"(src12), "r"(src13), "r"(src14), "r"(src15));
#endif
    }
};
// tcgen05 shared-memory -> tensor-memory copy, 32x128b, replicated over 4 warps.
// The pinned CuTe revision omits the comma between the two asm operand constraints;
// it is restored here (see the deviation note at the top of this file).
struct SM100_UTCCP_4x32dp128bit_1cta {
    using SRegisters = uint64_t[1];
    using DRegisters = uint32_t[1];

    DG_HOST_DEVICE static void copy(uint64_t const& src_addr, uint32_t const& dst_addr) {
#if defined(DG_ARCH_TCGEN05_ENABLED)
        asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
            :
            : "r"(dst_addr), "l"(src_addr));
#endif
    }
};

struct SM100_UTCCP_4x32dp128bit_2cta {
    using SRegisters = uint64_t[1];
    using DRegisters = uint32_t[1];

    DG_HOST_DEVICE static void copy(uint64_t const& src_addr, uint32_t const& dst_addr) {
#if defined(DG_ARCH_TCGEN05_ENABLED)
        asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
            :
            : "r"(dst_addr), "l"(src_addr));
#endif
    }
};

// --------------------------------------------------------------------------------------
// Tensor-memory allocator (cute/arch/tmem_allocator_sm100.hpp).
// --------------------------------------------------------------------------------------
namespace TMEM {

class Allocator1Sm {
public:
    static constexpr int ColumnsPerAllocationSlice = 32;
    static constexpr int Sm100TmemCapacityColumns = 512;

    DG_DEVICE Allocator1Sm() {}

    DG_HOST_DEVICE void allocate(int num_columns, uint32_t* dst_ptr) {
#if defined(DG_ARCH_TCGEN05_ENABLED)
        const uint32_t dst_intptr = cast_smem_ptr_to_uint(dst_ptr);
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
            :
            : "r"(dst_intptr), "r"(num_columns));
#endif
    }

    DG_DEVICE void free(uint32_t tmem_ptr, int num_columns) {
#if defined(DG_ARCH_TCGEN05_ENABLED)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
            :
            : "r"(tmem_ptr), "r"(num_columns));
#endif
    }

    DG_DEVICE void release_allocation_lock() {
#if defined(DG_ARCH_TCGEN05_ENABLED)
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::);
#endif
    }
};

class Allocator2Sm {
public:
    static constexpr int ColumnsPerAllocationSlice = 32;
    static constexpr int Sm100TmemCapacityColumns = 512;

    DG_DEVICE Allocator2Sm() {}

    DG_HOST_DEVICE void allocate(int num_columns, uint32_t* dst_ptr) {
#if defined(DG_ARCH_TCGEN05_ENABLED)
        const uint32_t dst_intptr = cast_smem_ptr_to_uint(dst_ptr);
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
            :
            : "r"(dst_intptr), "r"(num_columns));
#endif
    }

    DG_DEVICE void free(uint32_t tmem_ptr, int num_columns) {
#if defined(DG_ARCH_TCGEN05_ENABLED)
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
            :
            : "r"(tmem_ptr), "r"(num_columns));
#endif
    }

    DG_DEVICE void release_allocation_lock() {
#if defined(DG_ARCH_TCGEN05_ENABLED)
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::);
#endif
    }
};

} // namespace TMEM

// --------------------------------------------------------------------------------------
// Barriers, fences and UMMA arrival (cutlass/arch/barrier.h).
// --------------------------------------------------------------------------------------
namespace arch {

// CUTLASS reserves named barriers 1..7; user IDs start at 8.
static constexpr uint32_t kReservedNamedBarrierCount = 8;
static constexpr uint32_t kHardwareMaxNumNamedBarriers = 16;

struct NamedBarrier {
    DG_DEVICE static void arrive_and_wait(uint32_t num_threads, uint32_t barrier_id) {
#if DG_ARCH_BARRIER_ENABLED
        asm volatile("bar.sync %0, %1;" : : "r"(barrier_id + kReservedNamedBarrierCount), "r"(num_threads));
#endif
    }

    DG_DEVICE static void arrive(uint32_t num_threads, uint32_t barrier_id) {
#if DG_ARCH_BARRIER_ENABLED
        asm volatile("bar.arrive %0, %1;" : : "r"(barrier_id + kReservedNamedBarrierCount), "r"(num_threads));
#endif
    }

    DG_DEVICE static void sync(uint32_t num_threads, uint32_t barrier_id) {
        arrive_and_wait(num_threads, barrier_id);
    }
};

struct ClusterBarrier {
    using ValueType = uint64_t;

protected:
    // Never constructed in registers: the object only ever aliases shared memory.
    ValueType barrier_;

public:
    DG_DEVICE ClusterBarrier() = delete;

    DG_DEVICE void init(uint32_t arrive_count) const { ClusterBarrier::init(&this->barrier_, arrive_count); }
    DG_DEVICE void wait(uint32_t phase) const { ClusterBarrier::wait(&this->barrier_, phase); }
    DG_DEVICE bool test_wait(uint32_t phase, uint32_t pred = true) const {
        return ClusterBarrier::test_wait(&this->barrier_, phase, pred);
    }
    DG_DEVICE bool try_wait(uint32_t phase) const { return ClusterBarrier::try_wait(&this->barrier_, phase); }
    DG_DEVICE void arrive() const { ClusterBarrier::arrive(&this->barrier_); }
    DG_DEVICE void arrive(uint32_t cta_id, uint32_t pred = true) const {
        ClusterBarrier::arrive(&this->barrier_, cta_id, pred);
    }

    DG_HOST_DEVICE static void init(ValueType const* smem_ptr, uint32_t arrive_count) {
#if DG_ARCH_BARRIER_ENABLED
        const uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);
        asm volatile("mbarrier.init.shared::cta.b64 [%1], %0;" : : "r"(arrive_count), "r"(smem_addr));
#endif
    }

    DG_HOST_DEVICE static void wait(ValueType const* smem_ptr, uint32_t phase) {
#if DG_ARCH_BARRIER_ENABLED
        const uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);
        // Arbitrarily large timer value after which try-wait expires and re-tries.
        const uint32_t ticks = 0x989680;
        asm volatile(
            "{\n\t"
            ".reg .pred       P1; \n\t"
            "LAB_WAIT: \n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; \n\t"
            "@P1 bra DONE; \n\t"
            "bra     LAB_WAIT; \n\t"
            "DONE: \n\t"
            "}"
            :
            : "r"(smem_addr), "r"(phase), "r"(ticks));
#endif
    }

    DG_HOST_DEVICE static bool test_wait(ValueType const* smem_ptr, uint32_t phase, uint32_t pred) {
#if DG_ARCH_BARRIER_ENABLED
        const uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);
        uint32_t wait_complete;
        asm volatile(
            "{\n\t"
            ".reg .pred P1; \n\t"
            ".reg .pred P2; \n\t"
            "setp.eq.u32 P2, %3, 1;\n\t"
            "@P2 mbarrier.test_wait.parity.shared::cta.b64 P1, [%1], %2; \n\t"
            "selp.b32 %0, 1, 0, P1; \n\t"
            "}"
            : "=r"(wait_complete)
            : "r"(smem_addr), "r"(phase), "r"(pred));
        return static_cast<bool>(wait_complete);
#else
        return false;
#endif
    }

    DG_HOST_DEVICE static bool try_wait(ValueType const* smem_ptr, uint32_t phase) {
#if DG_ARCH_BARRIER_ENABLED
        const uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);
        uint32_t wait_complete;
        asm volatile(
            "{\n\t"
            ".reg .pred P1; \n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2; \n\t"
            "selp.b32 %0, 1, 0, P1; \n\t"
            "}"
            : "=r"(wait_complete)
            : "r"(smem_addr), "r"(phase));
        return static_cast<bool>(wait_complete);
#else
        return false;
#endif
    }

    DG_HOST_DEVICE static void arrive(ValueType const* smem_ptr, uint32_t cta_id, uint32_t pred) {
#if DG_ARCH_BARRIER_ENABLED
        const uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);
        if (pred) {
            asm volatile(
                "{\n\t"
                ".reg .b32 remAddr32;\n\t"
                "mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
                "mbarrier.arrive.shared::cluster.b64  _, [remAddr32];\n\t"
                "}"
                :
                : "r"(smem_addr), "r"(cta_id));
        }
#endif
    }

    DG_HOST_DEVICE static void arrive(ValueType const* smem_ptr) {
#if DG_ARCH_BARRIER_ENABLED
        const uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" : : "r"(smem_addr));
#endif
    }

    DG_HOST_DEVICE static void invalidate(ValueType const* smem_ptr) {
#if DG_ARCH_BARRIER_ENABLED
        const uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);
        asm volatile("mbarrier.inval.shared::cta.b64 [%0];" : : "r"(smem_addr));
#endif
    }
};

struct ClusterTransactionBarrier : public ClusterBarrier {
    DG_DEVICE ClusterTransactionBarrier() = delete;

    DG_DEVICE void arrive_and_expect_tx(uint32_t transaction_bytes) const {
        ClusterTransactionBarrier::arrive_and_expect_tx(&this->barrier_, transaction_bytes);
    }

    DG_DEVICE void arrive_and_expect_tx(uint32_t transaction_bytes, uint32_t cta_id, uint32_t pred = 1u) const {
        ClusterTransactionBarrier::arrive_and_expect_tx(&this->barrier_, transaction_bytes, cta_id, pred);
    }

    DG_DEVICE void expect_transaction(uint32_t transaction_bytes) const {
        ClusterTransactionBarrier::expect_transaction(&this->barrier_, transaction_bytes);
    }

    DG_HOST_DEVICE static void arrive_and_expect_tx(ValueType const* smem_ptr, uint32_t transaction_bytes) {
#if DG_ARCH_BARRIER_ENABLED
        const uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0;"
            :
            : "r"(transaction_bytes), "r"(smem_addr));
#endif
    }

    DG_HOST_DEVICE static void arrive_and_expect_tx(ValueType const* smem_ptr, uint32_t transaction_bytes,
                                                    uint32_t cta_id, uint32_t pred) {
#if DG_ARCH_BARRIER_ENABLED
        const uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 remAddr32;\n\t"
            "setp.eq.u32 p, %2, 1;\n\t"
            "@p mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
            "@p mbarrier.arrive.expect_tx.shared::cluster.b64  _, [remAddr32], %3;\n\t"
            "}"
            :
            : "r"(smem_addr), "r"(cta_id), "r"(pred), "r"(transaction_bytes));
#endif
    }

    DG_HOST_DEVICE static void expect_transaction(ValueType const* smem_ptr, uint32_t transaction_bytes) {
#if DG_ARCH_BARRIER_ENABLED
        const uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);
        asm volatile("mbarrier.expect_tx.shared::cta.b64 [%1], %0;"
            :
            : "r"(transaction_bytes), "r"(smem_addr));
#endif
    }
};

static_assert(sizeof(ClusterTransactionBarrier) == 8, "mbarrier object must stay 8 bytes");

DG_DEVICE inline void fence_barrier_init() {
#if DG_ARCH_BARRIER_ENABLED
    asm volatile("fence.mbarrier_init.release.cluster;" ::);
#endif
}

DG_DEVICE inline void fence_view_async_shared() {
#if DG_ARCH_BARRIER_ENABLED
    asm volatile("fence.proxy.async.shared::cta;");
#endif
}

DG_DEVICE inline void fence_view_async_tmem_load() {
#if defined(DG_ARCH_TCGEN_ENABLED)
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::);
#endif
}

DG_DEVICE inline void fence_view_async_tmem_store() {
#if defined(DG_ARCH_TCGEN_ENABLED)
    asm volatile("tcgen05.wait::st.sync.aligned;" ::);
#endif
}

DG_HOST_DEVICE inline void umma_arrive(uint64_t const* smem_ptr) {
#if defined(DG_ARCH_TCGEN_ENABLED)
    const uint32_t bar_intptr = cast_smem_ptr_to_uint(smem_ptr);
    if (elect_one_sync())
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
            :
            : "r"(bar_intptr));
#endif
}

DG_HOST_DEVICE inline void umma_arrive_2x1SM(uint64_t const* smem_ptr) {
#if defined(DG_ARCH_TCGEN_ENABLED)
    const uint32_t bar_intptr = cast_smem_ptr_to_uint(smem_ptr);
    if (elect_one_sync())
        asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];"
            :
            : "r"(bar_intptr));
#endif
}

DG_HOST_DEVICE inline void umma_arrive_multicast(uint64_t const* smem_ptr, uint16_t cta_mask) {
#if defined(DG_ARCH_TCGEN_ENABLED)
    const uint32_t bar_intptr = cast_smem_ptr_to_uint(smem_ptr);
    if (elect_one_sync())
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
            :
            : "r"(bar_intptr), "h"(cta_mask));
#endif
}

DG_HOST_DEVICE inline void umma_arrive_multicast_2x1SM(uint64_t const* smem_ptr, uint16_t cta_mask) {
#if defined(DG_ARCH_TCGEN_ENABLED)
    const uint32_t bar_intptr = cast_smem_ptr_to_uint(smem_ptr);
    if (elect_one_sync())
        asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
            :
            : "r"(bar_intptr), "h"(cta_mask));
#endif
}

DG_HOST_DEVICE inline void cpasync_barrier_arrive_noinc(uint64_t const* smem_ptr) {
#if DG_ARCH_BARRIER_ENABLED
    const uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" : : "r"(smem_addr));
#endif
}

// cutlass/arch/memory_sm80.h
DG_DEVICE inline void cp_async_fence() {
#if defined(DG_ARCH_CP_ASYNC_ENABLED)
    asm volatile("cp.async.commit_group;" ::);
#endif
}

template <int N>
DG_DEVICE inline void cp_async_wait() {
#if defined(DG_ARCH_CP_ASYNC_ENABLED)
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
#endif
}

// cutlass/arch/reg_reconfig.h
template <uint32_t RegCount>
DG_DEVICE inline void warpgroup_reg_alloc() {
#if defined(DG_ARCH_CTA_RECONFIG_ENABLED)
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" : : "n"(RegCount));
#endif
}

template <uint32_t RegCount>
DG_DEVICE inline void warpgroup_reg_dealloc() {
#if defined(DG_ARCH_CTA_RECONFIG_ENABLED)
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" : : "n"(RegCount));
#endif
}

// cutlass/arch/grid_dependency_control.h
DG_DEVICE inline void launch_dependent_grids() {
#if defined(DG_ARCH_GDC_ENABLED)
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

DG_DEVICE inline void wait_on_dependent_grids() {
#if defined(DG_ARCH_GDC_ENABLED)
    asm volatile("griddepcontrol.wait;");
#endif
}

} // namespace arch

// --------------------------------------------------------------------------------------
// UMMA descriptors (cute/arch/mma_sm100_desc.hpp).
// --------------------------------------------------------------------------------------
namespace UMMA {

enum class Major : uint8_t {
    K  = 0,
    MN = 1
};

enum class ScaleIn : uint8_t {
    One = 0,
    Neg = 1
};

enum class ScaleOut : uint8_t {
    Zero = 0,
    One  = 1
};

enum class Saturate : uint8_t {
    False = 0,
    True  = 1
};

enum class LayoutType : uint8_t {
    SWIZZLE_NONE           = 0,
    SWIZZLE_128B_BASE32B   = 1,
    SWIZZLE_128B           = 2,
    SWIZZLE_64B            = 4,
    SWIZZLE_32B            = 6
};

enum class F16F32Format : uint8_t {
    F16  = 0,
    BF16 = 1,
    TF32 = 2,
};

enum class S8Format : uint8_t {
    UINT8 = 0,
    INT8  = 1,
};

enum class MXF8F6F4Format : uint8_t {
    E4M3    = 0,
    E5M2    = 1,
    E2M3    = 3,
    E3M2    = 4,
    E2M1    = 5,
    INVALID = 7
};

enum class MXF4Format : uint8_t {
    E2M1 = 1,
};

enum class ScaleFormat : uint8_t {
    UE4M3 = 0,
    UE8M0 = 1,
};

enum class CFormat : uint8_t {
    F16 = 0,
    F32 = 1,
    S32 = 2,
};

enum class MaxShift : uint8_t {
    NoShift    = 0,
    MaxShift8  = 1,
    MaxShift16 = 2,
    MaxShift32 = 3
};

union SmemDescriptor {
    uint64_t desc_ = 0;
    struct {
        // start_address, bit [0,14), 4LSB not included
        uint16_t start_address_ : 14, : 2;
        // leading dimension byte offset, bit [16,30), 4LSB not included
        uint16_t leading_byte_offset_ : 14, : 2;
        // stride dimension byte offset, bit [32,46), 4LSB not included
        uint16_t stride_byte_offset_ : 14, version_ : 2;
        // base_offset, bit [49,52); leading_byte_offset_mode, bit [52,53)
        uint8_t : 1, base_offset_ : 3, lbo_mode_ : 1, : 3;
        // layout type, bit [61,64)
        uint8_t : 5, layout_type_ : 3;
    };
    struct {
        uint32_t lo;
        uint32_t hi;
    };

    DG_HOST_DEVICE constexpr operator uint64_t() const noexcept { return desc_; }
};

static_assert(sizeof(SmemDescriptor) == 8, "UMMA shared-memory descriptor must stay 8 bytes");

union InstrDescriptor {
    uint32_t desc_;

    struct {
        uint16_t sparse_id2_    : 2,  // bit [ 0, 2)
                 sparse_flag_   : 1,  // bit [ 2, 3)
                 saturate_      : 1,  // bit [ 3, 4)
                 c_format_      : 2,  // bit [ 4, 6)
                                : 1,
                 a_format_      : 3,  // bit [ 7,10)
                 b_format_      : 3,  // bit [10,13)
                 a_negate_      : 1,  // bit [13,14)
                 b_negate_      : 1,  // bit [14,15)
                 a_major_       : 1;  // bit [15,16)
        uint16_t b_major_       : 1,  // bit [16,17)
                 n_dim_         : 6,  // bit [17,23)
                                : 1,
                 m_dim_         : 5,  // bit [24,29)
                                : 1,
                 max_shift_     : 2;  // bit [30,32)
    };

    DG_HOST_DEVICE constexpr explicit operator uint32_t() const noexcept { return desc_; }
};

union InstrDescriptorBlockScaled {
    uint32_t desc_;

    struct {
        uint16_t sparse_id2_    : 2,  // bit [ 0, 2)
                 sparse_flag_   : 1,  // bit [ 2, 3)
                                : 1,
                 b_sf_id_       : 2,  // bit [ 4, 6)
                                : 1,
                 a_format_      : 3,  // bit [ 7, 9)
                 b_format_      : 3,  // bit [10,12)
                 a_negate_      : 1,  // bit [13,14)
                 b_negate_      : 1,  // bit [14,15)
                 a_major_       : 1;  // bit [15,16)
        uint16_t b_major_       : 1,  // bit [16,17)
                 n_dim_         : 6,  // bit [17,23)
                 scale_format_  : 1,  // bit [23,24)
                 m_dim_         : 5,  // bit [24,29)
                 a_sf_id_       : 2,  // bit [29,31)
                 k_size_        : 1;  // bit [31,32)
    };

    DG_HOST_DEVICE constexpr operator uint32_t() const noexcept { return desc_; }
};

static_assert(sizeof(InstrDescriptor) == 4 and sizeof(InstrDescriptorBlockScaled) == 4,
              "UMMA instruction descriptors must stay 4 bytes");

template <class T>
DG_HOST_DEVICE constexpr uint8_t to_UMMAFormat() {
    if constexpr (is_same_v<T, half_t>)                             return uint8_t(F16F32Format::F16);
    else if constexpr (is_same_v<T, bfloat16_t>)                    return uint8_t(F16F32Format::BF16);
    else if constexpr (is_same_v<T, tfloat32_t>)                    return uint8_t(F16F32Format::TF32);
    else if constexpr (is_same_v<T, uint8_t>)                       return uint8_t(S8Format::UINT8);
    else if constexpr (is_same_v<T, int8_t>)                        return uint8_t(S8Format::INT8);
    else if constexpr (is_same_v<T, float_e4m3_t>)                  return uint8_t(MXF8F6F4Format::E4M3);
    else if constexpr (is_same_v<T, float_e5m2_t>)                  return uint8_t(MXF8F6F4Format::E5M2);
    else if constexpr (is_same_v<T, detail::float_e2m3_unpacksmem_t>) return uint8_t(MXF8F6F4Format::E2M3);
    else if constexpr (is_same_v<T, detail::float_e3m2_unpacksmem_t>) return uint8_t(MXF8F6F4Format::E3M2);
    else if constexpr (is_same_v<T, detail::float_e2m1_unpacksmem_t>) return uint8_t(MXF8F6F4Format::E2M1);
    else if constexpr (is_same_v<T, float_e2m1_t>)                  return uint8_t(MXF4Format::E2M1);
    else { static_assert(sizeof(T) == 0, "Unknown type for UMMAFormat"); }
}

template <class T>
DG_HOST_DEVICE constexpr uint8_t to_CFormat() {
    if constexpr (is_same_v<T, half_t>)   return uint8_t(CFormat::F16);
    else if constexpr (is_same_v<T, float>)  return uint8_t(CFormat::F32);
    else if constexpr (is_same_v<T, int32_t>) return uint8_t(CFormat::S32);
    else { static_assert(sizeof(T) == 0, "Unknown type for CFormat"); }
}

template <class T>
DG_HOST_DEVICE constexpr uint8_t to_ScaleFormat() {
    if constexpr (is_same_v<T, float_ue4m3_t>)  return uint8_t(ScaleFormat::UE4M3);
    else if constexpr (is_same_v<T, float_ue8m0_t>) return uint8_t(ScaleFormat::UE8M0);
    else { static_assert(sizeof(T) == 0, "Unknown type for ScaleFormat"); }
}

template <class a_type, class b_type, class c_type,
          int M, int N, Major a_major, Major b_major,
          ScaleIn a_neg = ScaleIn::One, ScaleIn b_neg = ScaleIn::One,
          Saturate c_sat = Saturate::False,
          bool is_sparse = false,
          MaxShift max_shift = MaxShift::NoShift>
DG_HOST_DEVICE constexpr InstrDescriptor make_instr_desc() {
    InstrDescriptor desc_i = {};

    desc_i.a_format_ = to_UMMAFormat<a_type>();
    desc_i.b_format_ = to_UMMAFormat<b_type>();
    desc_i.c_format_ = to_CFormat<c_type>();

    desc_i.m_dim_ = (M >> 4);
    desc_i.n_dim_ = (N >> 3);

    desc_i.a_major_ = uint8_t(a_major);
    desc_i.b_major_ = uint8_t(b_major);

    desc_i.a_negate_ = uint8_t(a_neg);
    desc_i.b_negate_ = uint8_t(b_neg);
    desc_i.saturate_ = uint8_t(c_sat);

    desc_i.sparse_flag_ = is_sparse;
    desc_i.sparse_id2_ = 0;

    desc_i.max_shift_ = uint8_t(max_shift);

    return desc_i;
}

// Upper 32b carry the instruction descriptor; lower 32b carry the sparse-metadata TMEM
// address and stay zero for the dense kernels in scope.
template <class a_type, class b_type, class c_type,
          int M, int N, Major a_major, Major b_major,
          ScaleIn a_neg = ScaleIn::One, ScaleIn b_neg = ScaleIn::One,
          Saturate c_sat = Saturate::False,
          bool is_sparse = false,
          MaxShift max_shift = MaxShift::NoShift>
DG_HOST_DEVICE constexpr uint64_t make_runtime_instr_desc(uint16_t sparse_id2 = 0u, uint32_t tmem_e = 0u) {
    InstrDescriptor desc_i = make_instr_desc<a_type, b_type, c_type, M, N, a_major, b_major,
                                             a_neg, b_neg, c_sat, is_sparse, max_shift>();
    (void) tmem_e;
    if constexpr (is_sparse)
        desc_i.sparse_id2_ = sparse_id2;
    else
        assert(sparse_id2 == 0u);
    return static_cast<uint64_t>(static_cast<uint32_t>(desc_i)) << 32;
}

template <bool is_sparse = false>
DG_HOST_DEVICE constexpr uint64_t make_runtime_instr_desc(InstrDescriptor desc_i,
                                                          uint16_t sparse_id2 = 0u,
                                                          uint32_t tmem_e = 0u) {
    (void) tmem_e;
    if constexpr (is_sparse)
        desc_i.sparse_id2_ = sparse_id2;
    else
        assert(sparse_id2 == 0u);
    return static_cast<uint64_t>(static_cast<uint32_t>(desc_i)) << 32;
}

template <class a_type, class b_type, class c_type, class sf_type,
          int M, int N, Major a_major, Major b_major,
          ScaleIn a_neg = ScaleIn::One, ScaleIn b_neg = ScaleIn::One,
          bool is_sparse = false>
DG_HOST_DEVICE constexpr InstrDescriptorBlockScaled make_instr_desc_block_scaled() {
    InstrDescriptorBlockScaled desc_i = {};

    desc_i.a_format_ = to_UMMAFormat<a_type>();
    desc_i.b_format_ = to_UMMAFormat<b_type>();

    desc_i.scale_format_ = to_ScaleFormat<sf_type>();
    desc_i.a_sf_id_ = 0;
    desc_i.b_sf_id_ = 0;

    desc_i.m_dim_ = (M >> 4);
    desc_i.n_dim_ = (N >> 3);

    desc_i.a_major_ = uint8_t(a_major);
    desc_i.b_major_ = uint8_t(b_major);

    desc_i.a_negate_ = uint8_t(a_neg);
    desc_i.b_negate_ = uint8_t(b_neg);
    desc_i.sparse_flag_ = is_sparse;
    desc_i.sparse_id2_ = 0;

    return desc_i;
}

template <class a_type, class b_type, class c_type, class sf_type,
          int M, int N, Major a_major, Major b_major,
          ScaleIn a_neg = ScaleIn::One, ScaleIn b_neg = ScaleIn::One,
          bool is_sparse = false>
DG_HOST_DEVICE constexpr uint64_t make_runtime_instr_desc_block_scaled(uint32_t const tmem_sfa_addr,
                                                                      uint32_t const tmem_sfb_addr,
                                                                      uint16_t const sparse_id2 = 0u,
                                                                      uint32_t const tmem_e = 0u) {
    InstrDescriptorBlockScaled desc_i = make_instr_desc_block_scaled<a_type, b_type, c_type, sf_type,
                                                                     M, N, a_major, b_major,
                                                                     a_neg, b_neg, is_sparse>();
    (void) tmem_e;
    // The first 2 bits of a TMEM address carry the byte address.
    desc_i.a_sf_id_ = (tmem_sfa_addr & 0xC0000000) >> 30;
    desc_i.b_sf_id_ = (tmem_sfb_addr & 0xC0000000) >> 30;

    if constexpr (is_sparse)
        desc_i.sparse_id2_ = sparse_id2;
    else
        assert(sparse_id2 == 0u);
    return static_cast<uint64_t>(static_cast<uint32_t>(desc_i)) << 32;
}

template <bool is_sparse = false>
DG_HOST_DEVICE constexpr uint64_t make_runtime_instr_desc_block_scaled(InstrDescriptorBlockScaled desc_i,
                                                                      uint32_t const tmem_sfa_addr,
                                                                      uint32_t const tmem_sfb_addr,
                                                                      uint16_t const sparse_id2 = 0u,
                                                                      uint32_t const tmem_e = 0u) {
    (void) tmem_e;
    desc_i.a_sf_id_ = (tmem_sfa_addr & 0xC0000000) >> 30;
    desc_i.b_sf_id_ = (tmem_sfb_addr & 0xC0000000) >> 30;

    if constexpr (is_sparse)
        desc_i.sparse_id2_ = sparse_id2;
    else
        assert(sparse_id2 == 0u);
    return static_cast<uint64_t>(static_cast<uint32_t>(desc_i)) << 32;
}

} // namespace UMMA

} // namespace dg

// ---- include/deep_gemm/comm/barrier.cuh ----

namespace deep_gemm::comm {
DG_DEVICE void cluster_sync_with_relaxed_arrive() {
    dg::cluster_arrive_relaxed();
    dg::cluster_wait();
}
}

// ---- include/deep_gemm/scheduler/gemm.cuh ----

// ---- include/deep_gemm/common/math.cuh ----

#include <cuda/std/cstdint>

// ---- include/deep_gemm/common/compile.cuh ----

#if defined(__NVCC__) or (defined(__clang__) and defined(__CUDA__)) or defined(__CUDACC_RTC__) or defined(__CLION_IDE__)
#define DG_IN_CUDA_COMPILATION
#endif

#if defined(__NVCC__) || (defined(__clang__) and defined(__CUDA__))
#define DG_HOST_DEVICE_NOINLINE  __device__ __host__
#define DG_DEVICE_NOINLINE __device__
#elif defined(__CUDACC_RTC__)
#define DG_HOST_DEVICE_NOINLINE __device__
#define DG_DEVICE_NOINLINE __device__
#else
#define DG_HOST_DEVICE_NOINLINE
#define DG_DEVICE_NOINLINE
#endif

// ---- include/deep_gemm/common/exception.cuh ----
#include <cuda/std/cstdint>
// A whole-program build compiles this header's host pass too, where `__CUDA_ARCH__`
// is undefined, so a device source's `#else` arch-trap branch is instantiated and
// `DG_DEVICE_ASSERT` needs a declared `printf`.  The delivered JIT compiled
// device-only cubins and never hit it.  Include-only change: no generated
// instruction, constant or code path is altered.
#include <cstdio>

#ifdef __CLION_IDE__

DG_HOST_DEVICE void host_device_printf(const char* format, ...) {
    asm volatile("trap;");
}

#define printf host_device_printf
#endif

// A whole-program build compiles the host pass too, where `__CUDA_ARCH__` is
// undefined.  `trap` is a PTX instruction, so the bare `asm("trap;")` that the
// delivered macros expand to reaches the *host* assembler whenever an assert sits
// in a `DG_HOST_DEVICE` function, and gas rejects it with "Error: no such
// instruction: `trap'".  The delivered JIT compiled device-only cubins and never
// hit it.  The device branch below is byte-identical to the delivered macros; only
// the host pass loses a trap it can never execute.
#ifdef __CUDA_ARCH__
#ifndef DG_DEVICE_ASSERT
#define DG_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) { \
        printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, #cond); \
        asm("trap;"); \
    } \
} while (0)
#endif

#ifndef DG_TRAP_ONLY_DEVICE_ASSERT
#define DG_TRAP_ONLY_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) \
        asm("trap;"); \
} while (0)
#endif
#else
#ifndef DG_DEVICE_ASSERT
#define DG_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) { \
        printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, #cond); \
    } \
} while (0)
#endif

#ifndef DG_TRAP_ONLY_DEVICE_ASSERT
#define DG_TRAP_ONLY_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) { \
    } \
} while (0)
#endif
#endif

#ifndef DG_STATIC_ASSERT
#define DG_STATIC_ASSERT(cond, ...) static_assert(cond, __VA_ARGS__)
#endif

#ifndef DG_UNIFIED_ASSERT
#ifdef DG_IN_CUDA_COMPILATION
#define DG_UNIFIED_ASSERT(cond) DG_DEVICE_ASSERT(cond)
#else
#define DG_UNIFIED_ASSERT(cond) DG_HOST_ASSERT(cond)
#endif
#endif

// ---- include/deep_gemm/ptx/utils.cuh ----
#include <cuda/std/cstdint>
#include <cuda_bf16.h>

#if defined(DG_IN_CUDA_COMPILATION)
namespace deep_gemm::ptx {

DG_DEVICE uint32_t get_sm_idx() {
    uint32_t sm_idx;
    asm ("mov.u32 %0, %%smid;" : "=r"(sm_idx));
    return sm_idx;
}

DG_DEVICE uint32_t get_lane_idx() {
    uint32_t lane_id;
    asm ("mov.u32 %0, %%laneid;" : "=r"(lane_id));
    return lane_id;
}

DG_DEVICE uint64_t get_grid_idx() {
    uint64_t grid_idx;
    asm volatile("mov.u64 %0, %%gridid;" : "=l"(grid_idx));
    return grid_idx;
}

DG_DEVICE void sync_aligned(const uint32_t& num_threads, const uint32_t& barrier_idx) {
    asm volatile("bar.sync %0, %1;" : : "r"(barrier_idx), "r"(num_threads));
}

DG_DEVICE void sync_unaligned(const uint32_t& num_threads, const uint32_t& barrier_idx) {
    asm volatile("barrier.sync %0, %1;" : : "r"(barrier_idx), "r"(num_threads));
}

template <typename dtype_t>
DG_DEVICE dtype_t exchange(dtype_t ptr, const uint32_t& src_lane_idx) {
    DG_STATIC_ASSERT(sizeof(dtype_t) % sizeof(uint32_t) == 0, "");
    const auto send_int_values = reinterpret_cast<uint32_t*>(&ptr);
    dtype_t recv_dtype;
    auto recv_int_values = reinterpret_cast<uint32_t*>(&recv_dtype);
    #pragma unroll
    for (uint32_t i = 0; i < sizeof(dtype_t) / sizeof(uint32_t); ++ i)
        recv_int_values[i] = __shfl_sync(0xffffffff, send_int_values[i], static_cast<int>(src_lane_idx));
    return recv_dtype;
}

DG_DEVICE float reduce_max_sync(const float& value) {
    float ret;
    asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(ret) : "f"(value));
    return ret;
}

DG_DEVICE nv_bfloat162 cvt_relu_bf16x2_f32(const float2& v) {
#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)
    uint32_t packed;
    asm volatile("cvt.rn.relu.bf16x2.f32 %0, %1, %2;\n" : "=r"(packed) : "f"(v.y), "f"(v.x));
    return *reinterpret_cast<const nv_bfloat162*>(&packed);
#else
    return __floats2bfloat162_rn(fmaxf(v.x, 0.0f), fmaxf(v.y, 0.0f));
#endif
}

DG_DEVICE void accumulate(float2& a, nv_bfloat162 b) {
#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)
    // Use `add.rn.f32.bf16` instruction to perform fused (cast + add) operation on SM100
    asm("add.rn.f32.bf16 %0, %1, %0;\n" : "+f"(a.x) : "h"(*reinterpret_cast<uint16_t*>(&b.x)));
    asm("add.rn.f32.bf16 %0, %1, %0;\n" : "+f"(a.y) : "h"(*reinterpret_cast<uint16_t*>(&b.y)));
#else
    const auto [x, y] = __bfloat1622float2(b);
    a.x += x, a.y += y;
#endif
}

DG_DEVICE void accumulate_square(float2& a, nv_bfloat162 b) {
#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)
    // Accumulate BF16 squares directly into FP32 without separate conversion instructions.
    asm("fma.rn.f32.bf16 %0, %1, %1, %0;\n" : "+f"(a.x) : "h"(*reinterpret_cast<uint16_t*>(&b.x)));
    asm("fma.rn.f32.bf16 %0, %1, %1, %0;\n" : "+f"(a.y) : "h"(*reinterpret_cast<uint16_t*>(&b.y)));
#else
    const float2 b_f32 = __bfloat1622float2(b);
    a.x += b_f32.x * b_f32.x;
    a.y += b_f32.y * b_f32.y;
#endif
}

} // namespace deep_gemm::ptx
#endif

namespace deep_gemm::math {

/// Pointer operations
template <typename dtype_t = void>
DG_HOST_DEVICE dtype_t* advance_ptr(void* ptr, const uint64_t num_bytes) {
    return reinterpret_cast<dtype_t*>(static_cast<uint8_t*>(ptr) + num_bytes);
}

/// Math functions
template <typename T>
DG_HOST_DEVICE T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

template <typename T>
DG_HOST_DEVICE constexpr T constexpr_ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

template <typename T, bool kDoCeilAlignment = true>
DG_HOST_DEVICE T align(T a, T b) {
    return (kDoCeilAlignment ? ceil_div(a, b) : (a / b)) * b;
}

template <typename T>
DG_HOST_DEVICE constexpr T constexpr_align(T a, T b) {
    return constexpr_ceil_div(a, b) * b;
}

template <typename T>
DG_HOST_DEVICE constexpr T constexpr_gcd(T a, T b) {
    return b == 0 ? a : constexpr_gcd(b, a % b);
}

template <typename T>
DG_HOST_DEVICE constexpr T constexpr_min(T a, T b) {
    return a < b ? a : b;
}

template <typename T>
DG_DEVICE void swap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

#ifdef DG_IN_CUDA_COMPILATION
DG_DEVICE float2 fma2(const float2& a, const float2& b, const float2& c) {
#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)
    return __ffma2_rn(a, b, c);
#else
    return make_float2(
        __fmaf_rn(a.x, b.x, c.x),
        __fmaf_rn(a.y, b.y, c.y)
    );
#endif
}

DG_HOST_DEVICE float fast_rcp(const float& x) {
#ifdef __CUDA_ARCH__
    float ret;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(ret) : "f"(x));
    return ret;
#else
    // A whole-program build compiles this `__host__ __device__` body in the host
    // pass too, where `__CUDA_ARCH__` is undefined and host inline asm rejects the
    // PTX "f" constraints ("output constraint 0 must specify a single register").
    // The delivered JIT compiled device-only cubins and never hit it.  The device
    // branch above is unchanged, so every kernel still instantiates the same
    // rcp.approx.ftz.f32; this branch is host-only and is never launched.
    return 1.0f / x;
#endif
}

/// Casting
template <typename old_t>
DG_DEVICE nv_bfloat162 cast_into_bf16x2(old_t& x, old_t& y) {
    return __float22bfloat162_rn({*reinterpret_cast<float*>(&x), *reinterpret_cast<float*>(&y)});
}

template <typename old_t>
DG_DEVICE int cast_into_bf16_and_pack(old_t& x, old_t& y) {
    auto bf16x2 = cast_into_bf16x2(x, y);
    return *reinterpret_cast<int*>(&bf16x2);
}

/// Packed BF16 quantization
// Tree-reduce the amax of packed BF16 pairs (HMNMX2)
template <uint32_t kNumPacked>
DG_DEVICE nv_bfloat16 get_packed_bf16_amax(const nv_bfloat162 (&values)[kNumPacked]) {
    DG_STATIC_ASSERT(kNumPacked >= 2 and (kNumPacked & (kNumPacked - 1)) == 0,
                     "The tree requires a power-of-two size");
    nv_bfloat162 tree[kNumPacked / 2];
    #pragma unroll
    for (uint32_t i = 0; i < kNumPacked / 2; ++ i)
        tree[i] = __hmax2(__habs2(values[i]), __habs2(values[i + kNumPacked / 2]));
    #pragma unroll
    for (uint32_t stride = kNumPacked / 4; stride > 0; stride /= 2) {
        #pragma unroll
        for (uint32_t i = 0; i < stride; ++ i)
            tree[i] = __hmax2(tree[i], tree[i + stride]);
    }
    return __hmax(tree[0].x, tree[0].y);
}

// Scale two BF16 pairs by power-of-two BF16 sf_invs (exact) and pack into four FP8 bytes
DG_DEVICE uint32_t scale_bf16x2_into_fp8x4(const nv_bfloat162& lower, const nv_bfloat162& upper,
                                                const nv_bfloat162& sf_inv_lower, const nv_bfloat162& sf_inv_upper) {
    return __nv_fp8x4_e4m3(__hmul2(lower, sf_inv_lower), __hmul2(upper, sf_inv_upper)).__x;
}

// Select a power-of-two UE8M0 SF mapping `amax` into the finite range of `quant_dtype_t`:
// the carry of the integer addition performs the exponent ceiling (carry iff the amax
// mantissa exceeds the max finite value's mantissa)
template <typename quant_dtype_t = dg::float_e4m3_t, typename dtype_t>
DG_DEVICE uint32_t get_ue8m0_sf_exp(const dtype_t& amax) {
    constexpr bool kIsFP32 = dg::is_same_v<dtype_t, float>;
    DG_STATIC_ASSERT((kIsFP32 or dg::is_same_v<dtype_t, nv_bfloat16>), "The input type must be FP32 or BF16");
    DG_STATIC_ASSERT((dg::is_same_v<quant_dtype_t, dg::float_e4m3_t> or dg::is_same_v<quant_dtype_t, dg::float_e2m1_t>),
                     "The quantized type must be E4M3 or E2M1");
    constexpr bool kIsFP8 = dg::is_same_v<quant_dtype_t, dg::float_e4m3_t>;
    constexpr uint32_t kMantissaBits = kIsFP32 ? 23 : 7;
    constexpr uint32_t kMantissaMask = (1u << kMantissaBits) - 1;
    constexpr uint32_t kQuantMaxMantissa = (kIsFP8 ? 0x60u : 0x40u) << (kMantissaBits - 7);   // mantissa(1.75) or mantissa(1.5)
    constexpr uint32_t kQuantMaxExponent = kIsFP8 ? 8 : 2;                                    // 448 = 1.75 * 2 ^ 8, 6 = 1.5 * 2 ^ 2
    // The exponent floors match the per-token cast kernel's amax clamps:
    // `max(amax, 1e-4)` for E4M3 (2 ^ -22), `max(amax, 6 * 2 ^ -126)` for E2M1 (2 ^ -126)
    constexpr uint32_t kMinSFExponent = kIsFP8 ? 105 : 1;
    uint32_t amax_bits;
    if constexpr (kIsFP32)
        amax_bits = __float_as_uint(amax);
    else
        amax_bits = __bfloat16_as_ushort(amax);
    const auto rounded_exp = (amax_bits + kMantissaMask - kQuantMaxMantissa) >> kMantissaBits;
    return dg::max(rounded_exp, kMinSFExponent + kQuantMaxExponent) - kQuantMaxExponent;
}

// The `dtype_t` sf_inv of a UE8M0 SF exponent (reciprocal biased exponents sum to 254):
// a power of two, exact in both output formats
template <typename dtype_t>
DG_DEVICE auto get_ue8m0_sf_inv(const uint32_t& sf_exp) {
    if constexpr (dg::is_same_v<dtype_t, float>) {
        return __uint_as_float((254u - sf_exp) << 23);
    } else {
        DG_STATIC_ASSERT((dg::is_same_v<dtype_t, nv_bfloat16>), "The output type must be FP32 or BF16");
        return __ushort_as_bfloat16((254u - sf_exp) << 7);
    }
}

/// Reduction
template <typename T>
DG_DEVICE T warp_inclusive_sum(T value, const uint32_t lane_idx) {
    #pragma unroll
    for (uint32_t offset = 1; offset < 32; offset <<= 1) {
        const T synced = __shfl_up_sync(0xffffffff, value, offset);
        if (lane_idx >= offset)
            value += synced;
    }
    return value;
}

// Block-wide prefix sum. Synchronize the block before reusing `warp_sums`.
template <uint32_t kNumThreads, typename T>
DG_DEVICE T cta_exclusive_sum(const T value, T* warp_sums, T& total) {
    constexpr uint32_t kNumWarps = kNumThreads / 32;
    DG_STATIC_ASSERT(kNumThreads >= 32 and kNumThreads % 32 == 0 and kNumWarps <= 32, "Invalid CTA scan shape");

    const uint32_t lane_idx = ptx::get_lane_idx();
    const uint32_t warp_idx = dg::canonical_warp_idx_sync();
    const T lane_sum = warp_inclusive_sum(value, lane_idx);
    if (lane_idx == 31)
        warp_sums[warp_idx] = lane_sum;
    __syncthreads();

    const T warp_total = lane_idx < kNumWarps ? warp_sums[lane_idx] : 0;
    const T warp_sum = warp_inclusive_sum(warp_total, lane_idx);
    total = ptx::exchange(warp_sum, kNumWarps - 1);
    return lane_sum - value + ptx::exchange(warp_sum - warp_total, warp_idx);
}

template <uint32_t kNumThreads, typename T>
DG_DEVICE T cta_exclusive_sum(const T value, T* warp_sums) {
    T total;
    return cta_exclusive_sum<kNumThreads>(value, warp_sums, total);
}

// Operation functors
// NOTES: `fmaxf`/`fminf` lower to single `FMNMX` instructions; the generic float ternaries
//        cannot (mismatched NaN semantics) and lower to `FSETP + FSEL` pairs instead,
//        doubling both the instruction count and the dependency chain latency
template <typename T> struct ReduceSum { DG_DEVICE T operator()(T a, T b) const { return a + b; } };
template <typename T> struct ReduceMax { DG_DEVICE T operator()(T a, T b) const { return a > b ? a : b; } };
template <typename T> struct ReduceMin { DG_DEVICE T operator()(T a, T b) const { return a < b ? a : b; } };
template <> struct ReduceMax<float> { DG_DEVICE float operator()(float a, float b) const { return fmaxf(a, b); } };
template <> struct ReduceMin<float> { DG_DEVICE float operator()(float a, float b) const { return fminf(a, b); } };
template <> struct ReduceMax<nv_bfloat16> { DG_DEVICE nv_bfloat16 operator()(nv_bfloat16 a, nv_bfloat16 b) const { return __hmax(a, b); } };
template <> struct ReduceMax<nv_bfloat162> { DG_DEVICE nv_bfloat162 operator()(nv_bfloat162 a, nv_bfloat162 b) const { return __hmax2(a, b); } };
template <typename T> struct ReduceAnd { DG_DEVICE T operator()(T a, T b) const { return a & b; } };
template <typename T> struct ReduceOr  { DG_DEVICE T operator()(T a, T b) const { return a | b; } };

// Unified reduction function
template <uint32_t kNumLanesPerGroup, bool kIntergroupReduce, typename T, typename Op>
DG_DEVICE T warp_reduce(T value, Op op) {
    DG_STATIC_ASSERT(kNumLanesPerGroup == 32 or kNumLanesPerGroup == 16 or kNumLanesPerGroup == 8 or
                     kNumLanesPerGroup ==  4 or kNumLanesPerGroup == 2  or kNumLanesPerGroup == 1,
                     "Invalid number of lanes");
    constexpr uint32_t mask = 0xffffffff;
    if constexpr (kIntergroupReduce) {
        if constexpr (kNumLanesPerGroup <=  1) value = op(value, __shfl_xor_sync(mask, value,  1));
        if constexpr (kNumLanesPerGroup <=  2) value = op(value, __shfl_xor_sync(mask, value,  2));
        if constexpr (kNumLanesPerGroup <=  4) value = op(value, __shfl_xor_sync(mask, value,  4));
        if constexpr (kNumLanesPerGroup <=  8) value = op(value, __shfl_xor_sync(mask, value,  8));
        if constexpr (kNumLanesPerGroup <= 16) value = op(value, __shfl_xor_sync(mask, value, 16));
    } else {
        if constexpr (kNumLanesPerGroup >= 32) value = op(value, __shfl_xor_sync(mask, value, 16));
        if constexpr (kNumLanesPerGroup >= 16) value = op(value, __shfl_xor_sync(mask, value,  8));
        if constexpr (kNumLanesPerGroup >=  8) value = op(value, __shfl_xor_sync(mask, value,  4));
        if constexpr (kNumLanesPerGroup >=  4) value = op(value, __shfl_xor_sync(mask, value,  2));
        if constexpr (kNumLanesPerGroup >=  2) value = op(value, __shfl_xor_sync(mask, value,  1));
    }
    return value;
}

// Convenience aliases
template <uint32_t kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
DG_DEVICE T warp_reduce_sum(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceSum<T>{});
}

template <uint32_t kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
DG_DEVICE T warp_reduce_max(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceMax<T>{});
}

#endif

} // namespace deep_gemm

// ---- include/deep_gemm/common/types.cuh ----

namespace deep_gemm {

enum class MmaKind {
    BF16        = 0,
    MXFP8FP4    = 1,
    MXF4        = 2,
};

enum class GemmType {
    Normal                              = 0,
    MGroupedContiguous                  = 1,
    MGroupedMasked                      = 2,
    KGroupedContiguous                  = 3,
    Batched                             = 4,
    MGroupedContiguousWithPsumLayout    = 5,
    KGroupedContiguousWithPsumLayout    = 6,
};

constexpr DG_HOST_DEVICE bool is_m_grouped_contiguous(const GemmType& gemm_type) {
    switch (gemm_type) {
        case GemmType::MGroupedContiguous:                  return true;
        case GemmType::MGroupedContiguousWithPsumLayout:    return true;
        default: return false;
    }
}

constexpr DG_HOST_DEVICE bool is_k_grouped_contiguous(const GemmType& gemm_type) {
    switch (gemm_type) {
        case GemmType::KGroupedContiguous:                  return true;
        case GemmType::KGroupedContiguousWithPsumLayout:    return true;
        default: return false;
    }
}

enum class KernelType {
    Kernel1D1D = 0,
    Kernel1D2D = 1,
    KernelNoSF = 2
};

// Host-visible runtime state of the `epilogue::transform` operators, which derive from
// this struct without adding members, so the host fully determines the operator on launch.
// Only the fields consumed by the selected epilogue type are meaningful
struct EpilogueArgs {
    uint32_t* sfd = nullptr;
    uint32_t sfd_stride = 0;
    uint32_t shape_m = 0, shape_n = 0;
    float alpha = 1.0f;
};

} // namespace deep_gemm

namespace deep_gemm::sched {

enum class IndexType {
    MN,
    K,
    SF_K,
};

template <GemmType kGemmType, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumSMs, bool kIsMulticastOnA>
static constexpr uint32_t get_num_1d_blocks_per_group() {
    // Select the best from candidates
    uint32_t num_best_blocks = 0, min_usage = dg::numeric_limits<uint32_t>::max();
    for (const auto candidate: {8u, 16u}) {
        const auto usage = kIsMulticastOnA ?
            candidate * BLOCK_N + math::constexpr_ceil_div(kNumSMs, candidate) * BLOCK_M: // Grouping on N
            candidate * BLOCK_M + math::constexpr_ceil_div(kNumSMs, candidate) * BLOCK_N; // Grouping on M
        if (usage < min_usage)
            min_usage = usage, num_best_blocks = candidate;
    }
    return num_best_blocks;
}

#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-pro-type-member-init"
template <GemmType kGemmType,
          uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumGroups,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs,
          bool kEnsureZeroPadding = true,
          uint32_t kKAlignment = 128u,     // psum k-group start alignment
          uint32_t kSFKSpan = 128u,        // K covered by one k-grouped SF row
          uint32_t kNum1DBlocksPerGroup = get_num_1d_blocks_per_group<kGemmType, BLOCK_M, BLOCK_N, kNumSMs, kIsMulticastOnA>(),
          uint32_t kSplitKFactor = 1>
struct Scheduler {
    // A/B group starts must be aligned to whole K blocks. SF rows are packed
    // independently per group and tracked by `current_sf_k_cumsum`.
    DG_STATIC_ASSERT(not is_k_grouped_contiguous(kGemmType) or kKAlignment % 128 == 0,
                     "K alignment must be a multiple of BLOCK_K (128)");

    // Only `Normal` has both halves of split-K: the constructor inflates `num_blocks` by
    // `kSplitKFactor`, and `get_next_block`'s final `else` decomposes the raw index back into
    // `mn_block_idx` / `split_k_idx`. Every other `GemmType` has at most one half:
    //   - `Batched` is inflated but its own branch never sets `split_k_idx`, and it derives
    //     `current_group_idx` from the inflated `num_blocks` -- wrong group indexing;
    //   - `MGroupedContiguous` falls into the same final `else` as `Normal` but is NOT
    //     inflated, so `split_k_idx` is always 0 -- every block writes partition 0 and
    //     partitions 1..n-1 keep the uninitialised workspace `sm120_split_k_reduce` sums.
    // Both are silent wrong numerics, so encode the invariant rather than the two symptoms.
    // The `Batched` defect also exists upstream in nv_dev; we deliberately do not fix it here.
    DG_STATIC_ASSERT(kSplitKFactor == 1 or kGemmType == GemmType::Normal,
                     "Split-K is only supported for Normal GEMM: it is the only GemmType whose "
                     "constructor inflates num_blocks by kSplitKFactor AND whose get_next_block "
                     "branch decomposes the index into mn_block_idx/split_k_idx");

    int current_iter = -1;

    // Block configs
    uint32_t num_blocks;
    uint32_t num_m_blocks;
    uint32_t num_n_blocks;

    // Split-K state (inert unless kSplitKFactor > 1)
    uint32_t num_mn_blocks;
    uint32_t split_k_idx;

    // For SM90 multicast checks
    uint32_t num_blocks_in_group;
    bool is_peer_cta_alive = true;

    // For grouped GEMM
    int* grouped_layout;
    uint32_t current_group_idx = 0;
    // Only used for masked layout
    uint32_t current_m_cumsum = 0;
    // Only used for contiguous psum layout
    uint32_t last_psum_m = 0, current_psum_m, current_m_block_cumsum = 0;
    // Only used for k-grouped layout
    // NOTES: `current_k_start` is the current group's physical K start offset
    // (always a multiple of `kKAlignment`), maintained by both psum and non-psum paths
    uint32_t current_shape_k, current_k_start = 0, current_sf_k_cumsum = 0;
    // Only used for `KGroupedContiguousWithPsumLayout`
    uint32_t current_k_end = 0;

    // Load the K-group selected by `current_group_idx`.
    DG_DEVICE void get_next_k_group() {
        if constexpr (kGemmType == GemmType::KGroupedContiguousWithPsumLayout) {
            // `grouped_layout[i]` is the psum end offset in K elements.
            const auto next_k_end = static_cast<uint32_t>(grouped_layout[current_group_idx]);
            current_k_start = math::align(current_k_end, kKAlignment);
            current_shape_k = next_k_end - current_k_start;
            current_k_end = next_k_end;
        } else {
            current_k_start += current_shape_k;
            current_shape_k = grouped_layout[current_group_idx];
        }
    }

    // ReSharper disable once CppPossiblyUninitializedMember
    DG_DEVICE explicit Scheduler(const uint32_t& shape_m, const uint32_t& shape_n,
                                       const uint32_t& shape_k, int* grouped_layout = nullptr) {
        num_m_blocks = math::ceil_div(shape_m, BLOCK_M);
        num_n_blocks = math::ceil_div(shape_n, BLOCK_N);
        current_shape_k = is_k_grouped_contiguous(kGemmType) ? 0 : shape_k;
        num_mn_blocks = num_m_blocks * num_n_blocks;
        if constexpr (kGemmType == GemmType::Normal or kGemmType == GemmType::Batched) {
            num_blocks = num_mn_blocks * kSplitKFactor;
        } else if constexpr (kGemmType == GemmType::MGroupedContiguous) {
            num_blocks = num_m_blocks * num_n_blocks;
            this->grouped_layout = grouped_layout;
        } else if constexpr (kGemmType == GemmType::MGroupedMasked) {
            this->grouped_layout = grouped_layout;
        } else if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            this->grouped_layout = grouped_layout;
            current_psum_m = grouped_layout[0];
            num_m_blocks = math::ceil_div(current_psum_m, BLOCK_M);
        } else if constexpr (is_k_grouped_contiguous(kGemmType)) {
            num_blocks = num_m_blocks * num_n_blocks;
            this->grouped_layout = grouped_layout;
            get_next_k_group();
        }
    }

    DG_DEVICE void get_swizzled_block_idx(const uint32_t& block_idx, uint32_t& m_block_idx, uint32_t& n_block_idx) {
        DG_STATIC_ASSERT(kNum1DBlocksPerGroup % kNumMulticast == 0, "Invalid group size");

        // Swizzle for better L2 usages
        const auto primary_num_blocks = kIsMulticastOnA ? num_n_blocks : num_m_blocks;
        const auto secondary_num_blocks = kIsMulticastOnA ? num_m_blocks : num_n_blocks;
        const auto num_blocks_per_group = secondary_num_blocks * kNum1DBlocksPerGroup;
        const auto group_idx = block_idx / num_blocks_per_group;
        auto first_block_idx = group_idx * kNum1DBlocksPerGroup;
        auto in_group_idx = block_idx % num_blocks_per_group;
        num_blocks_in_group = min(kNum1DBlocksPerGroup, primary_num_blocks - first_block_idx);

        // Fix unaligned TMA multicast
        // NOTES: for SM90 only, as SM90 can dynamically disable TMA multicast
        // while SM100 uses 2-CTA, which can not be dynamically disabled
#if __CUDA_ARCH__ < 1000
        if (kNumMulticast > 1 and num_blocks_in_group % 2 != 0) {
            if (in_group_idx < (num_blocks_in_group ^ 1) * secondary_num_blocks) {
                num_blocks_in_group = num_blocks_in_group ^ 1;
            } else {
                in_group_idx = in_group_idx - (num_blocks_in_group ^ 1) * secondary_num_blocks;
                first_block_idx += num_blocks_in_group ^ 1;
                num_blocks_in_group = 1;
            }
        }
#endif

        // Convert to final M/N block indices
        // `kIsMulticastOnA == true` leads to groups on N
        if constexpr (kIsMulticastOnA) {
            m_block_idx = in_group_idx / num_blocks_in_group;
            n_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
        } else {
            m_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
            n_block_idx = in_group_idx / num_blocks_in_group;
        }
    }

    template <bool kWithGroupOffset, IndexType kIndexType = IndexType::MN>
    DG_DEVICE uint32_t get_global_idx(const uint32_t shape_dim, const uint32_t block_size,
                                             const uint32_t& block_idx, const uint32_t& m_block_idx = 0) {
        if constexpr (kGemmType == GemmType::Normal) {
            return block_idx * block_size;
        } else if constexpr (kGemmType == GemmType::MGroupedContiguous) {
            const auto offset = kWithGroupOffset ? dg::max(0, grouped_layout[m_block_idx * BLOCK_M]) : 0;
            return offset * shape_dim + block_idx * block_size;
        } else if constexpr (kGemmType == GemmType::MGroupedMasked or kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            const auto offset = kWithGroupOffset ? current_group_idx : 0;
            return offset * shape_dim + block_idx * block_size;
        } else if constexpr (is_k_grouped_contiguous(kGemmType)) {
            auto offset = 0;
            if constexpr (kWithGroupOffset) {
                if constexpr (kIndexType == IndexType::MN) {
                    offset = current_group_idx * shape_dim;
                } else if constexpr (kIndexType == IndexType::K) {
                    offset = current_k_start;
                } else if constexpr (kIndexType == IndexType::SF_K) {
                    offset = current_sf_k_cumsum;
                }
            }
            return offset + block_idx * block_size;
        } else if constexpr (kGemmType == GemmType::Batched) {
            // Ignore kWithGroupOffset, and apply offset for IndexType::SF_K
            const auto offset = kIndexType == IndexType::SF_K ? current_group_idx : 0;
            return offset * shape_dim + block_idx * block_size;
        }
    }

    // For swap A/B and psum layout only
    DG_DEVICE uint32_t get_aligned_effective_m_in_block(const uint32_t& m_block_idx) const {
        constexpr uint32_t UMMA_STEP_N = 16;
        DG_STATIC_ASSERT(BLOCK_M % UMMA_STEP_N == 0, "Invalid alignment");
        if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout and not kEnsureZeroPadding)
            return math::align(m_block_idx == last_psum_m / BLOCK_M + num_m_blocks - 1 ? current_psum_m - m_block_idx * BLOCK_M : BLOCK_M, UMMA_STEP_N);
        return BLOCK_M;
    }

    DG_DEVICE bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
        const auto next_block_idx = (++ current_iter) * kNumSMs + blockIdx.x;

        if constexpr (kGemmType == GemmType::MGroupedMasked) {
            while (true) {
                // End of the task
                if (current_group_idx == kNumGroups)
                    return false;

                // Within current group
                num_m_blocks = math::ceil_div(static_cast<uint32_t>(grouped_layout[current_group_idx]), BLOCK_M);
                const auto current_m_block_cumsum = current_m_cumsum + num_m_blocks;
                if (next_block_idx < current_m_block_cumsum * num_n_blocks)
                    break;

                // Move to check the next group
                current_group_idx ++, current_m_cumsum = current_m_block_cumsum;
            }

            get_swizzled_block_idx(next_block_idx - current_m_cumsum * num_n_blocks, m_block_idx, n_block_idx);
        } else if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) { 
            while (true) {
                // Within current group
                if (next_block_idx < (current_m_block_cumsum + num_m_blocks) * num_n_blocks)
                    break;

                // Move to check the next group
                if (++ current_group_idx == kNumGroups)
                    return false;

                // NOTES: `num_m_blocks` varies with the increase of the group index
                last_psum_m = math::align(current_psum_m, BLOCK_M);
                current_psum_m = grouped_layout[current_group_idx];
                current_m_block_cumsum += num_m_blocks;
                num_m_blocks = math::ceil_div(current_psum_m - last_psum_m, BLOCK_M);
            }

            get_swizzled_block_idx(next_block_idx - current_m_block_cumsum * num_n_blocks, m_block_idx, n_block_idx);

            // NOTES: `last_psum_m` is aligned with block M
            m_block_idx += last_psum_m / BLOCK_M;
        } else if constexpr (is_k_grouped_contiguous(kGemmType)) {
            while (true) {
                // End of the task
                if (current_group_idx == kNumGroups)
                    return false;

                // Within current group
                if (next_block_idx < (current_group_idx + 1) * num_blocks)
                    break;

                // Move to check the next group
                current_group_idx ++;
                if (current_group_idx >= kNumGroups)
                    return false;

                const auto aligned_shape_k = math::align(current_shape_k, kKAlignment);
                current_sf_k_cumsum += math::ceil_div(aligned_shape_k, kSFKSpan);
                get_next_k_group();
            }

            get_swizzled_block_idx(next_block_idx - current_group_idx * num_blocks, m_block_idx, n_block_idx);
        } else if constexpr (kGemmType == GemmType::Batched) {
            if (next_block_idx >= num_blocks * kNumGroups)
                return false;

            current_group_idx = next_block_idx / num_blocks;
            const auto block_idx = next_block_idx - current_group_idx * num_blocks;
            if constexpr (kIsMulticastOnA) {
                m_block_idx = block_idx / num_n_blocks;
                n_block_idx = block_idx % num_n_blocks;
            } else {
                m_block_idx = block_idx % num_m_blocks;
                n_block_idx = block_idx / num_m_blocks;
            }
        } else {
            // NOTES: the bounds check stays on the RAW index against the inflated
            // `num_blocks`, or split-K never terminates.
            if (next_block_idx >= num_blocks)
                return false;

            uint32_t mn_block_idx = next_block_idx;
            if constexpr (kSplitKFactor > 1) {
                mn_block_idx = next_block_idx % num_mn_blocks;
                split_k_idx  = next_block_idx / num_mn_blocks;
            } else {
                split_k_idx = 0;
            }

            // For SM90 only
            // NOTES: we don't have to set `is_peer_cta_alive` for masked grouped GEMM, as it must be aligned
            is_peer_cta_alive = num_n_blocks % kNumMulticast == 0 or                  // Always aligned on N (constant bypass)
                                num_m_blocks % kNumMulticast == 0 or                  // Always aligned on M (constant bypass)
                                (mn_block_idx ^ 1) < num_mn_blocks;                   // Peer CTA in bound
            get_swizzled_block_idx(mn_block_idx, m_block_idx, n_block_idx);
        }
        return true;
    }

    // For SM90 only
    DG_DEVICE bool is_tma_multicast_valid(const uint32_t& m_block_idx) const {
        if (num_blocks_in_group == 1)
            return false;
        if constexpr (kGemmType == GemmType::Normal or kGemmType == GemmType::MGroupedMasked or
                      is_k_grouped_contiguous(kGemmType) or kGemmType == GemmType::Batched or
                      kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            return true;
        } else {
            DG_STATIC_ASSERT(kGemmType == GemmType::MGroupedContiguous, "Invalid Gemm type");
            if constexpr (kIsMulticastOnA) {
                return true;
            } else {
                const auto group_idx = grouped_layout[m_block_idx * BLOCK_M];
                const auto peer_group_idx = grouped_layout[(m_block_idx ^ 1) * BLOCK_M];
                return group_idx == peer_group_idx;
            }
        }
    }

    // For SM90 only
    // ReSharper disable once CppNotAllPathsReturnValue
    DG_DEVICE bool is_computation_valid(const uint32_t& m_block_idx, const uint32_t& m_offset) const {
        if constexpr (kGemmType == GemmType::Normal or kGemmType == GemmType::Batched) {
            return true;
        } else if constexpr (kGemmType == GemmType::MGroupedContiguous) {
            return grouped_layout[m_offset + m_block_idx * BLOCK_M] >= 0;
        } else if constexpr (kGemmType == GemmType::MGroupedMasked) {
            return m_offset + m_block_idx * BLOCK_M < grouped_layout[current_group_idx];
        } else if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            return m_offset + m_block_idx * BLOCK_M < current_psum_m;
        } else {
            // Unreachable 
            DG_TRAP_ONLY_DEVICE_ASSERT(false);
        }
    }
};

#pragma clang diagnostic pop

} // namespace deep_gemm::sched

// ---- include/deep_gemm/common/tma_copy.cuh ----

// ---- include/deep_gemm/common/packing.cuh ----

namespace deep_gemm {

template <typename dtype_t>
constexpr DG_HOST_DEVICE uint32_t get_smem_pack_factor() {
    // Packed FP4 stores two logical elements per byte in shared memory.
    return dg::is_same_v<dtype_t, dg::float_e2m1_t> ? 2 : 1;
}

} // namespace deep_gemm

namespace deep_gemm::tma {

template <uint32_t BLOCK_INNER, uint32_t kSwizzleMode, uint32_t kPackFactor, typename dtype_t>
constexpr uint32_t get_inner_block_atom_size() {
    return kSwizzleMode == 0 ? BLOCK_INNER / kPackFactor : kSwizzleMode / sizeof(dtype_t);
}

template <uint32_t BLOCK_INNER, uint32_t BLOCK_OUTER,
          uint32_t kSwizzleMode,
          typename dtype_t, bool kIs3DTMA = false,
          uint64_t kCacheHint = static_cast<uint64_t>(dg::TMA::CacheHintSm100::EVICT_NORMAL)>
DG_DEVICE void
copy(void const* desc_ptr, dg::arch::ClusterTransactionBarrier* barrier_ptr,
     dtype_t* smem_ptr, const uint32_t& inner_idx, const uint32_t& outer_idx,
     const uint32_t& num_tma_multicast = 1, const uint32_t& batch_idx = 0) {
    DG_STATIC_ASSERT(static_cast<uint64_t>(dg::TMA::CacheHintSm90::EVICT_NORMAL) ==
                     static_cast<uint64_t>(dg::TMA::CacheHintSm100::EVICT_NORMAL), "Invalid cache hint");
    constexpr uint32_t kPackFactor = get_smem_pack_factor<dtype_t>();
    // NOTES: for sub-byte packed types, the SMEM atom stride stays in
    //        bytes (`sizeof(dtype_t) == 1`), while the GMEM index and loop count run over logical elements
    DG_STATIC_ASSERT(kPackFactor == 1 or sizeof(dtype_t) == 1, "Packing expects a 1-byte storage type");
    constexpr uint32_t BLOCK_INNER_ATOM_STORAGE = get_inner_block_atom_size<BLOCK_INNER, kSwizzleMode, kPackFactor, dtype_t>();
    constexpr uint32_t BLOCK_INNER_ATOM_LOGICAL = BLOCK_INNER_ATOM_STORAGE * kPackFactor;
    DG_STATIC_ASSERT(BLOCK_INNER % BLOCK_INNER_ATOM_LOGICAL == 0, "TMA inner block must contain whole atoms");

    if constexpr (not kIs3DTMA) {
        if (num_tma_multicast == 1) {
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM_LOGICAL; ++ i) {
                dg::SM90_TMA_LOAD_2D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                             kCacheHint,
                                             smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM_STORAGE,
                                             inner_idx + i * BLOCK_INNER_ATOM_LOGICAL, outer_idx);
            }
        } else {
            #if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000))
                // 2-CTA function will send signals to the leader CTA only
                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM_LOGICAL; ++ i) {
                    dg::SM100_TMA_2SM_LOAD_2D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                                      kCacheHint,
                                                      smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM_STORAGE,
                                                      inner_idx + i * BLOCK_INNER_ATOM_LOGICAL, outer_idx);
                }
            #elif (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900))
                if (dg::block_rank_in_cluster() == 0) {
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM_LOGICAL; ++ i) {
                        dg::SM90_TMA_LOAD_MULTICAST_2D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                                               (1 << num_tma_multicast) - 1, kCacheHint,
                                                               smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM_STORAGE,
                                                               inner_idx + i * BLOCK_INNER_ATOM_LOGICAL, outer_idx);
                    }
                }
            #endif
        }
    } else {
        if (num_tma_multicast == 1) {
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM_LOGICAL; ++ i) {
                dg::SM90_TMA_LOAD_3D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                            static_cast<uint64_t>(dg::TMA::CacheHintSm100::EVICT_NORMAL),
                                            smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM_STORAGE,
                                            inner_idx + i * BLOCK_INNER_ATOM_LOGICAL, outer_idx, batch_idx);
            }
        } else {
            #if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000))
                // 2-CTA function will send signals to the leader CTA only
                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM_LOGICAL; ++ i) {
                    dg::SM100_TMA_2SM_LOAD_3D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                                      static_cast<uint64_t>(dg::TMA::CacheHintSm100::EVICT_NORMAL),
                                                      smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM_STORAGE,
                                                      inner_idx + i * BLOCK_INNER_ATOM_LOGICAL, outer_idx, batch_idx);
                }
            #elif (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900))
                if (dg::block_rank_in_cluster() == 0) {
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM_LOGICAL; ++ i) {
                        dg::SM90_TMA_LOAD_MULTICAST_3D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                                               (1 << num_tma_multicast) - 1, static_cast<uint64_t>(dg::TMA::CacheHintSm90::EVICT_NORMAL),
                                                               smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM_STORAGE,
                                                               inner_idx + i * BLOCK_INNER_ATOM_LOGICAL, outer_idx, batch_idx);
                    }
                }
            #endif
        }
    }
}

} // namespace deep_gemm::tma

// ---- include/deep_gemm/common/utils.cuh ----
#include <type_traits>
#include <utility>

#include <cuda/std/cstdint>

namespace deep_gemm::utils {

template <typename FuncT>
struct PatternVisitor {
    FuncT func;

    DG_HOST_DEVICE
    explicit PatternVisitor(FuncT&& func): func(std::forward<FuncT>(func)) {}

    DG_HOST_DEVICE
    auto operator [](const uint32_t& i) const {
        return func(i);
    }
};

template <uint32_t kNumBytes>
struct Vectorized {
    static auto zeros() {
        // TODO: add `ulonglong4` for SM100 once `__ldg` support this
        if constexpr (kNumBytes > 0 and kNumBytes % 16 == 0) {
            return make_uint4(0, 0, 0, 0);
        } else if constexpr (kNumBytes > 0 and kNumBytes % 8 == 0) {
            return make_uint2(0, 0);
        } else if constexpr (kNumBytes > 0 and kNumBytes % 4 == 0) {
            return 0;
        } else {
            DG_STATIC_ASSERT(kNumBytes > 0 and kNumBytes % 4 == 0, "Invalid vectorization");
        }
    }

    using vec_t = decltype(zeros());
};

template <uint32_t kNumCols>
DG_DEVICE constexpr uint32_t get_num_aligned_tmem_cols() {
    DG_STATIC_ASSERT(kNumCols <= 512, "Too many tensor memory columns");
    if constexpr (kNumCols <=  32) return  32;
    if constexpr (kNumCols <=  64) return  64;
    if constexpr (kNumCols <= 128) return 128;
    if constexpr (kNumCols <= 256) return 256;
    return 512;
}

template <typename T>
__device__ __forceinline__ T shfl_sync(unsigned mask, T var, int srcLane, int width = 32) {

    using shfl_t = std::conditional_t<sizeof(T) == 4, int,
                   std::conditional_t<sizeof(T) == 8, long long, long long>>;

    T result;
    shfl_t* var_ptr = reinterpret_cast<shfl_t*>(&var);
    shfl_t* result_ptr = reinterpret_cast<shfl_t*>(&result);
    *result_ptr = __shfl_sync(mask, *var_ptr, srcLane, width);

    if constexpr (sizeof(T) == 16) {
        *(result_ptr + 1) = __shfl_sync(mask, *(var_ptr + 1), srcLane, width);
    }

    return result;
}

} // namespace deep_gemm::utils

// ---- include/deep_gemm/epilogue/sm100_store_cd.cuh ----

// ---- include/deep_gemm/epilogue/transform.cuh ----

namespace deep_gemm::epilogue::transform {

// Epilogue operators add behavior (never state) to `EpilogueArgs`, so kernels take
// the host-constructed operator directly as a kernel argument
// NOTES: the operators do not compose with each other
struct EpilogueIdentity: EpilogueArgs {
    template <uint32_t STORE_BLOCK_N>
    DG_DEVICE static uint32_t apply_index_n(const uint32_t& n_idx) {
        return n_idx;
    }

    template <uint32_t kNumValues>
    DG_DEVICE void apply_values(uint32_t (&)[kNumValues]) const {}
};

// Scale only the product term of a BLAS-style linear combination. Supporting an
// arbitrary beta requires a separate C load/initialization path and does not belong
// in this value-only transform.
struct EpilogueWithAlpha: EpilogueIdentity {
    template <uint32_t kNumValues>
    DG_DEVICE void apply_values(uint32_t (&values)[kNumValues]) const {
        DG_STATIC_ASSERT(kNumValues % 2 == 0, "Alpha scaling requires float2 alignment");
        const auto values_f32x2 = reinterpret_cast<float2*>(values);
        const auto alpha_f32x2 = make_float2(alpha, alpha);
        #pragma unroll
        for (uint32_t value_idx = 0; value_idx < kNumValues / 2; ++ value_idx)
            values_f32x2[value_idx] = __fmul2_rn(values_f32x2[value_idx], alpha_f32x2);
    }
};

template <uint32_t kLeft, uint32_t kMid, uint32_t kRight>
struct EpilogueHeadSplits: EpilogueIdentity {
    template <uint32_t STORE_BLOCK_N>
    DG_DEVICE static uint32_t apply_index_n(const uint32_t& n_idx) {
        DG_STATIC_ASSERT(kLeft % STORE_BLOCK_N == 0 and kMid % STORE_BLOCK_N == 0 and
                         kRight % STORE_BLOCK_N == 0, "Invalid head splits config");
        return n_idx + (n_idx + kRight) / (kLeft + kRight) * kMid;
    }
};

// Cast D into FP8 with dynamic per-row, per-32-column UE8M0 SFs, packed into `uint32_t`
// words in a TMA-aligned MN-major layout (the same layout accepted for SFA)
// NOTES: the accumulator is rounded into BF16 before the amax/scale/cast steps, so the
//        output bitwise matches a BF16 D followed by the standalone per-token cast kernel
struct EpilogueDynamicScaledFP8: EpilogueIdentity {
    static constexpr uint32_t kSFGranN = 32;

    // Store one SF byte (`uint8_t`), or all four SF bytes of one packed word at once
    // (`uint32_t`, fully coalesced; only valid when `shape_n % 128 == 0`, so that a word
    // never crosses a batch or shape boundary)
    // NOTES: batched GEMMs flatten the SF columns of all batches along `(batch_idx, n)`,
    //        matching a D viewed by the consumer as a `(shape_m, num_batches * shape_n)` 2D tensor
    template <typename sf_t>
    DG_DEVICE void store_sf(const uint32_t& row_idx, const uint32_t& group_n_idx,
                                 const uint32_t& batch_idx, const sf_t& sf) const {
        DG_STATIC_ASSERT(dg::is_same_v<sf_t, uint8_t> or dg::is_same_v<sf_t, uint32_t>,
                         "The SF must be a single byte or a whole packed word");
        if (row_idx >= shape_m or group_n_idx >= shape_n)
            return;
        const auto sf_idx = (batch_idx * shape_n + group_n_idx) / kSFGranN;
        const auto sf_word_ptr = sfd + (sf_idx / 4) * sfd_stride + row_idx;
        if constexpr (dg::is_same_v<sf_t, uint32_t>) {
            *sf_word_ptr = sf;
        } else {
            reinterpret_cast<uint8_t*>(sf_word_ptr)[sf_idx % 4] = sf;
        }
    }
};

} // namespace deep_gemm::epilogue::transform

// ---- include/deep_gemm/ptx/ld_st.cuh ----
#include <cuda/std/cstdint>
#include <cuda_bf16.h>

#if defined(DG_IN_CUDA_COMPILATION)
namespace deep_gemm::ptx {

// Compatibility: 256 bits LD/ST instructions
#if defined(CUDART_VERSION) and CUDART_VERSION >= 13000
using longlong4_t = longlong4_32a;
#define make_longlong4_t make_longlong4_32a
#else
struct alignas(32) longlong4_t { long long x, y, z, w; };
DG_HOST_DEVICE longlong4_t make_longlong4_t(
    const long long& x, const long long& y, const long long& z, const long long& w) {
    return {x, y, z, w};
}
#endif

/// LD/ST matrix
// TODO: remove `struct`
struct SM90_U32x2_LDSM_N {
    DG_DEVICE static void
    copy(uint32_t& dst_0, uint32_t& dst_1, void* smem_src) {
        asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
                     : "=r"(dst_0), "=r"(dst_1)
                     : "l"(__cvta_generic_to_shared(smem_src)));
    }
};

struct SM90_U32x4_LDSM_N {
    DG_DEVICE static void
    copy(uint32_t& dst_0, uint32_t& dst_1, uint32_t& dst_2, uint32_t& dst_3, void* smem_src) {
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(dst_0), "=r"(dst_1), "=r"(dst_2), "=r"(dst_3)
                     : "l"(__cvta_generic_to_shared(smem_src)));
    }
};

template <typename dtype_t>
struct SM90_U32x2_STSM_N {
    DG_DEVICE static void
    copy(dtype_t src_0, dtype_t src_1, void* smem_dst) {
        DG_STATIC_ASSERT(sizeof(dtype_t) == sizeof(uint32_t), "Invalid dtype");
        const uint32_t src[2] = {*reinterpret_cast<uint32_t*>(&src_0), *reinterpret_cast<uint32_t*>(&src_1)};
        asm volatile("stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n"
                     :: "l"(__cvta_generic_to_shared(smem_dst)), "r"(src[0]), "r"(src[1]));
    }
};

template <typename dtype_t>
struct SM90_U32x2_STSM_T {
    DG_DEVICE static void
    copy(dtype_t src_0, dtype_t src_1, void* smem_dst) {
        DG_STATIC_ASSERT(sizeof(dtype_t) == sizeof(uint32_t), "Invalid dtype");
        const uint32_t src[2] = {*reinterpret_cast<uint32_t*>(&src_0), *reinterpret_cast<uint32_t*>(&src_1)};
        asm volatile("stmatrix.sync.aligned.x2.m8n8.shared.b16.trans [%0], {%1, %2};\n"
                     :: "l"(__cvta_generic_to_shared(smem_dst)),
                        "r"(src[0]), "r"(src[1]));
    }
};

template <typename dtype_t>
struct SM90_U32x4_STSM_T {
    DG_DEVICE static void
    copy(dtype_t src_0, dtype_t src_1, dtype_t src_2, dtype_t src_3, void* smem_dst) {
        DG_STATIC_ASSERT(sizeof(dtype_t) == sizeof(uint32_t), "Invalid dtype");
        const uint32_t src[4] = {*reinterpret_cast<uint32_t*>(&src_0), *reinterpret_cast<uint32_t*>(&src_1),
                                 *reinterpret_cast<uint32_t*>(&src_2), *reinterpret_cast<uint32_t*>(&src_3)};
        asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16.trans [%0], {%1, %2, %3, %4};\n"
                     :: "l"(__cvta_generic_to_shared(smem_dst)),
                        "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3]));
    }
};

template <typename dtype_t>
struct SM100_U8x4_STSM_T {
    DG_DEVICE static void
    copy(dtype_t src_0, void* smem_dst) {
        DG_STATIC_ASSERT(sizeof(dtype_t) == sizeof(uint32_t), "Invalid dtype");
        const uint32_t src = *reinterpret_cast<uint32_t*>(&src_0);
        asm volatile("stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [%0], {%1};\n"
                     :: "l"(__cvta_generic_to_shared(smem_dst)), "r"(src));
    }
};

template <typename dtype_t>
struct SM100_U8x8_STSM_T {
    DG_DEVICE static void
    copy(dtype_t src_0, dtype_t src_1, void* smem_dst) {
        DG_STATIC_ASSERT(sizeof(dtype_t) == sizeof(uint32_t), "Invalid dtype");
        const uint32_t src[2] = {*reinterpret_cast<uint32_t*>(&src_0), *reinterpret_cast<uint32_t*>(&src_1)};
        asm volatile("stmatrix.sync.aligned.m16n8.x2.trans.shared.b8 [%0], {%1, %2};\n"
                     :: "l"(__cvta_generic_to_shared(smem_dst)), "r"(src[0]), "r"(src[1]));
    }
};

/// Memory ordering
DG_DEVICE void fence_acq_rel_cta() {
    asm volatile("fence.acq_rel.cta;" ::: "memory");
}

/// Shared memory
DG_DEVICE uint32_t ld_shared(const uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(ret) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

DG_DEVICE float2 ld_shared(const float2* ptr) {
    float2 ret;
    asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];" : "=f"(ret.x), "=f"(ret.y) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

DG_DEVICE float4 ld_shared(const float4* ptr) {
    float4 ret;
    asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

DG_DEVICE uint4 ld_shared(const uint4* ptr) {
    uint4 ret;
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

DG_DEVICE float ld_shared(const float* ptr) {
    float ret;
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(ret) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

DG_DEVICE void st_shared(const float* ptr, float val) {
    asm volatile("st.shared.f32 [%0], %1;" :: "l"(__cvta_generic_to_shared(ptr)), "f"(val));
}

DG_DEVICE void st_shared(const float2* ptr, float2 val) {
    asm volatile("st.shared.v2.f32 [%0], {%1, %2};" :: "l"(__cvta_generic_to_shared(ptr)), "f"(val.x), "f"(val.y));
}

DG_DEVICE void st_shared(const uint32_t* ptr, uint32_t val) {
    asm volatile("st.shared.u32 [%0], %1;" :: "l"(__cvta_generic_to_shared(ptr)), "r"(val));
}

DG_DEVICE void st_shared(const void* ptr, uint32_t x, uint32_t y) {
    asm volatile("st.shared.v2.u32 [%0], {%1, %2};" :: "l"(__cvta_generic_to_shared(ptr)), "r"(x), "r"(y));
}

DG_DEVICE void st_shared(const void* ptr, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"(__cvta_generic_to_shared(ptr)), "r"(x), "r"(y), "r"(z), "r"(w));
}

DG_DEVICE void st_shared(const __int128_t* ptr, __int128_t val) {
    asm volatile("st.shared.b128 [%0], %1;" :: "l"(__cvta_generic_to_shared(ptr)), "q"(val));
}

template <uint32_t kNumL2PrefetchBytes>
DG_DEVICE void cp_async_cg(const uint4* global_ptr, uint4* shared_ptr) {
    DG_STATIC_ASSERT(kNumL2PrefetchBytes == 64 or kNumL2PrefetchBytes == 256, "Invalid L2 prefetch size");
    if constexpr (kNumL2PrefetchBytes == 64) {
        asm volatile("cp.async.cg.shared::cta.global.L2::64B [%0], [%1], 16;" ::
            "r"(static_cast<uint32_t>(__cvta_generic_to_shared(shared_ptr))), "l"(global_ptr));
    } else {
        asm volatile("cp.async.cg.shared::cta.global.L2::256B [%0], [%1], 16;" ::
            "r"(static_cast<uint32_t>(__cvta_generic_to_shared(shared_ptr))), "l"(global_ptr));
    }
}

DG_DEVICE uint32_t mapa_shared(const uint32_t& ptr, const uint32_t& dst_cta_idx) {
    uint32_t mapped;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(mapped) : "r"(ptr), "r"(dst_cta_idx));
    return mapped;
}

template <class T, class Barrier>
DG_DEVICE void st_async_cluster(T* dst, const T& src, const uint32_t& dst_cta_idx, Barrier& barrier) {
    DG_STATIC_ASSERT(sizeof(T) % 16 == 0, "Invalid cluster shared store size");

    // Map address
    const auto mapped_barrier_ptr = mapa_shared(static_cast<uint32_t>(__cvta_generic_to_shared(&barrier)), dst_cta_idx);
    const auto mapped_dst_ptr = mapa_shared(static_cast<uint32_t>(__cvta_generic_to_shared(dst)), dst_cta_idx);

    // Move
    const auto u32_view = reinterpret_cast<const uint32_t*>(&src);
    for (int i = 0; i < sizeof(T) / 16; ++ i) {
        asm volatile(
            "st.async.shared::cluster.mbarrier::complete_tx::bytes.u32.v4 [%0], {%1, %2, %3, %4}, [%5];" ::
            "r"(mapped_dst_ptr + i * 16),
            "r"(u32_view[i * 4]), "r"(u32_view[i * 4 + 1]), "r"(u32_view[i * 4 + 2]), "r"(u32_view[i * 4 + 3]),
            "r"(mapped_barrier_ptr)
        );
    }
}

DG_DEVICE void st_shared_bulk(void* smem_ptr, const uint32_t& num_bytes) {
    // `size` must be 64-bit before PTX ISA 9.0
    DG_DEVICE_ASSERT(num_bytes % 8 == 0);
    asm volatile("st.bulk.weak.shared::cta [%0], %1, 0;" ::
                 "l"(__cvta_generic_to_shared(smem_ptr)), "l"(static_cast<uint64_t>(num_bytes)));
}

/// Global memory
constexpr uint64_t kEvictFirstCacheHint = 0x12f0000000000000;

DG_DEVICE uint4 ld_evict_first(const uint4* ptr) {
    uint4 value;
    asm volatile("ld.weak.global.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                 : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                 : "l"(ptr), "l"(kEvictFirstCacheHint));
    return value;
}

DG_DEVICE float ld_global(const float* ptr) {
    float ret;
    asm volatile("ld.weak.global.f32 %0, [%1];" : "=f"(ret) : "l"(ptr) : "memory");
    return ret;
}

DG_DEVICE float4 ld_global(const float4* ptr) {
    float4 ret;
    asm volatile("ld.weak.global.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w)
                 : "l"(ptr) : "memory");
    return ret;
}

DG_DEVICE void st_global(float* ptr, const float& value) {
    asm volatile("st.weak.global.f32 [%0], %1;" :: "l"(ptr), "f"(value) : "memory");
}

DG_DEVICE uint64_t ld_volatile(const uint64_t* ptr) {
    uint64_t ret;
    asm volatile("ld.volatile.global.b64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

DG_DEVICE uint32_t ld_volatile(const uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ld.volatile.global.b32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

DG_DEVICE uint32_t ld_acq(const uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(ret) : "l"(ptr) : "memory");
    return ret;
}

DG_DEVICE uint64_t ld_acq_sys(const uint64_t* ptr) {
    uint64_t ret;
    asm volatile("ld.acquire.sys.global.b64 %0, [%1];" : "=l"(ret) : "l"(ptr) : "memory");
    return ret;
}

DG_DEVICE void st_relaxed_sys(const uint64_t* ptr, const uint64_t& value) {
    asm volatile("st.L1::no_allocate.relaxed.sys.global.u64 [%0], %1;" :: "l"(ptr), "l"(value));
}

/// Atomics
DG_DEVICE uint32_t atomic_add(const uint32_t* ptr, const uint32_t& value) {
    uint32_t ret;
    asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(value));
    return ret;
}

DG_DEVICE uint64_t atomic_add(const uint64_t* ptr, const uint64_t& value) {
    uint64_t ret;
    asm volatile("atom.global.add.u64 %0, [%1], %2;" : "=l"(ret) : "l"(ptr), "l"(value));
    return ret;
}

DG_DEVICE uint64_t atomic_add_sys(const uint64_t* ptr, const uint64_t& value) {
    uint64_t ret;
    asm volatile("atom.sys.global.add.u64 %0, [%1], %2;" : "=l"(ret) : "l"(ptr), "l"(value));
    return ret;
}

DG_DEVICE uint32_t atomic_add_rel(const uint32_t* ptr, const uint32_t& value) {
    uint32_t ret;
    asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(value) : "memory");
    return ret;
}

DG_DEVICE void st_rel(const uint64_t* ptr, const uint64_t value) {
    asm volatile("st.release.gpu.global.u64 [%0], %1;" :: "l"(ptr), "l"(value) : "memory");
}

DG_DEVICE void red_add(const int* ptr, const int& value) {
    asm volatile("red.gpu.global.add.s32 [%0], %1;" :: "l"(ptr), "r"(value));
}

DG_DEVICE void red_add(const uint32_t* ptr, const uint32_t& value) {
    asm volatile("red.gpu.global.add.u32 [%0], %1;" :: "l"(ptr), "r"(value));
}

DG_DEVICE void red_or_rel_sys(const uint64_t* ptr, const uint64_t& value) {
    asm volatile("red.release.sys.global.or.b64 [%0], %1;" :: "l"(ptr), "l"(value));
}

DG_DEVICE void red_or_rel_gpu(uint64_t* ptr, const uint64_t& value) {
    asm volatile("red.release.gpu.global.or.b64 [%0], %1;" :: "l"(ptr), "l"(value));
}

DG_DEVICE void red_xor_rel(const uint64_t* ptr, const uint64_t& value) {
    asm volatile("red.release.gpu.global.xor.b64 [%0], %1;" :: "l"(ptr), "l"(value) : "memory");
}

DG_DEVICE void st_rel_sys(const uint64_t* ptr, const uint64_t& value) {
    asm volatile("st.release.sys.global.u64 [%0], %1;" :: "l"(ptr), "l"(value) : "memory");
}

DG_DEVICE void red_add_rel(const uint32_t* ptr, const uint32_t& value) {
    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"(ptr), "r"(value));
}

DG_DEVICE void red_async_inc_rel(const uint64_t* ptr) {
    asm volatile("red.async.release.gpu.global.add.u64 [%0], 1;" :: "l"(ptr) : "memory");
}

DG_DEVICE void red_add_rel_sys(const int* ptr, const int& value) {
    asm volatile("red.release.sys.global.add.s32 [%0], %1;" :: "l"(ptr), "r"(value));
}

DG_DEVICE int ld_acq_sys(const int* ptr) {
    int ret;
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

DG_DEVICE uint32_t ld_acq_sys(const uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

DG_DEVICE uint64_t ld_acq_gpu(const uint64_t* ptr) {
    uint64_t ret;
    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr) : "memory");
    return ret;
}

/// Predicated loads
DG_DEVICE longlong4_t ld_gez_pred(const longlong4_t* ptr, const int& pred) {
    longlong4_t ret = make_longlong4_t(0, 0, 0, 0);
    asm volatile(
        "{\n\t"
        "  .reg .pred p;\n\t"
        "  setp.ge.s32 p, %5, 0;\n\t"
        "  @p ld.global.L2::256B.v4.s64 {%0, %1, %2, %3}, [%4];\n\t"
        "}"
        : "+l"(ret.x), "+l"(ret.y), "+l"(ret.z), "+l"(ret.w)
        : "l"(ptr), "r"(pred)
        : "memory");
    return ret;
}

/// Prefetch
DG_DEVICE void prefetch_l1(void *ptr) {
    asm volatile("prefetch.global.L1 [%0];" :: "l"(ptr));
}

} // namespace deep_gemm::ptx
#endif

// ---- include/deep_gemm/ptx/tcgen05.cuh ----

namespace deep_gemm::ptx {

/// UMMA versions with relaxed assertions
struct SM100_MMA_F16BF16_SS {
    DG_DEVICE static void
    fma(uint64_t const& desc_a,
        uint64_t const& desc_b,
        uint32_t const& tmem_c,
        uint32_t const& scale_c,
        uint64_t const& desc) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p; \n\t"
            "}\n"
            :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(desc >> 32)), "r"(scale_c));
    }
};

struct SM100_MMA_F16BF16_2x1SM_SS {
    DG_DEVICE static void
    fma(uint64_t const& desc_a,
        uint64_t const& desc_b,
        uint32_t const& tmem_c,
        uint32_t const& scale_c,
        uint64_t const& desc) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p; \n\t"
            "}\n"
            :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(desc >> 32)), "r"(scale_c));
    }
};

struct SM100_MMA_MXF8F6F4_SS {
    DG_DEVICE static void
    fma(uint64_t const& desc_a,
        uint64_t const& desc_b,
        uint32_t const& tmem_c,
        uint32_t const& scale_c,
        uint64_t const& desc,
        uint32_t const& tmem_sfa,
        uint32_t const& tmem_sfb) {
        asm volatile(
          "{\n\t"
          ".reg .pred p;\n\t"
          "setp.ne.b32 p, %4, 0;\n\t"
          "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p; \n\t"
          "}\n"
          :
          : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(desc >> 32)), "r"(scale_c),
            "r"(tmem_sfa), "r"(tmem_sfb));
    }
};

struct SM100_MMA_MXF8F6F4_2x1SM_SS {
    DG_DEVICE static void
    fma(uint64_t const& desc_a,
        uint64_t const& desc_b,
        uint32_t const& tmem_c,
        uint32_t const& scale_c,
        uint64_t const& desc,
        uint32_t const& tmem_sfa,
        uint32_t const& tmem_sfb) {
        asm volatile(
          "{\n\t"
          ".reg .pred p;\n\t"
          "setp.ne.b32 p, %4, 0;\n\t"
          "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p; \n\t"
          "}\n"
          :
          : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(desc >> 32)), "r"(scale_c),
            "r"(tmem_sfa), "r"(tmem_sfb));
    }
};

struct SM100_MMA_F8F6F4_SS {
    DG_DEVICE static void
    fma(uint64_t const& desc_a,
        uint64_t const& desc_b,
        uint32_t const& tmem_c,
        uint32_t const& scale_c,
        uint64_t const& desc) {
        asm volatile(
          "{\n\t"
          ".reg .pred p;\n\t"
          "setp.ne.b32 p, %4, 0;\n\t"
          "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p; \n\t"
          "}\n"
          :
          : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(desc >> 32)), "r"(scale_c));
    }
};

struct SM100_MMA_F8F6F4_2x1SM_SS {
    DG_DEVICE static void
    fma(uint64_t const& desc_a,
        uint64_t const& desc_b,
        uint32_t const& tmem_c,
        uint32_t const& scale_c,
        uint64_t const& desc) {
        asm volatile(
          "{\n\t"
          ".reg .pred p;\n\t"
          "setp.ne.b32 p, %4, 0;\n\t"
          "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], %1, %2, %3, p; \n\t"
          "}\n"
          :
          : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(desc >> 32)), "r"(scale_c));
    }
};

struct SM100_MMA_MXF4_SS {
    DG_DEVICE static void
    fma(uint64_t const& desc_a,
        uint64_t const& desc_b,
        uint32_t const& tmem_c,
        uint32_t const& scale_c,
        uint64_t const& desc,
        uint32_t const& tmem_sfa,
        uint32_t const& tmem_sfb) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9)
            "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32 [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#else
            "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#endif
            "}\n"
            :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(desc >> 32)), "r"(scale_c),
               "r"(tmem_sfa), "r"(tmem_sfb));
    }
};

struct SM100_MMA_MXF4_2x1SM_SS {
    DG_DEVICE static void
    fma(uint64_t const& desc_a,
        uint64_t const& desc_b,
        uint32_t const& tmem_c,
        uint32_t const& scale_c,
        uint64_t const& desc,
        uint32_t const& tmem_sfa,
        uint32_t const& tmem_sfb) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9)
            "tcgen05.mma.cta_group::2.kind::mxf4.block_scale.block32 [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#else
            "tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#endif
            "}\n"
            :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(desc >> 32)), "r"(scale_c),
               "r"(tmem_sfa), "r"(tmem_sfb));
    }
};

struct SM100_MMA_F16BF16_WS_SS {
    DG_DEVICE static void
    fma(uint64_t const& desc_a,
        uint64_t const& desc_b,
        uint32_t const& tmem_c,
        uint32_t const& scale_c,
        uint64_t const& desc) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], %1, %2, %3, p; \n\t"
            "}\n"
            :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(desc >> 32)), "r"(scale_c));
    }
};

/// Tensor memory operations
template <uint32_t kNumValues>
DG_DEVICE void tmem_load_32dp32b(const uint32_t tmem_addr, uint32_t* values) {
    DG_STATIC_ASSERT(kNumValues == 4 or kNumValues == 8 or kNumValues == 16 or
                     kNumValues == 32 or kNumValues == 64 or kNumValues == 128,
                     "Invalid TMEM load width");
    using Loader = dg::conditional_t<kNumValues == 4, dg::SM100_TMEM_LOAD_32dp32b4x,
                   dg::conditional_t<kNumValues == 8, dg::SM100_TMEM_LOAD_32dp32b8x,
                   dg::conditional_t<kNumValues == 16, dg::SM100_TMEM_LOAD_32dp32b16x,
                   dg::conditional_t<kNumValues == 32, dg::SM100_TMEM_LOAD_32dp32b32x,
                   dg::conditional_t<kNumValues == 64, dg::SM100_TMEM_LOAD_32dp32b64x,
                                                               dg::SM100_TMEM_LOAD_32dp32b128x>>>>>;
    [&]<size_t... Is>(dg::index_sequence<Is...>) {
        Loader::copy(tmem_addr, values[Is]...);
    }(dg::make_index_sequence<kNumValues>{});
}

DG_DEVICE void umma_arrive_no_elect(dg::arch::ClusterTransactionBarrier& barrier) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&barrier))));
}

DG_DEVICE void tcgen05_before_thread_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;");
}

DG_DEVICE void tcgen05_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}

} // namespace deep_gemm::ptx

namespace deep_gemm::epilogue {

template <uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t STORE_BLOCK_M, uint32_t STORE_BLOCK_N,
          uint32_t kSwizzleCDMode,
          uint32_t kNumTMAStoreStages,
          uint32_t kNumUMMAStoreThreads,
          uint32_t kNumOverlappedTmemCols,
          GemmType kGemmType, bool kWithAccumulation,
          typename epilogue_op_t,
          typename smem_t>
DG_DEVICE void
sm100_store_cd(smem_t& smem, uint32_t& tma_stage_idx,
               const uint32_t& tmem_base_addr,
               const uint32_t& base_m_idx, const uint32_t& base_n_idx, const uint32_t& batch_idx,
               const bool& is_empty_group,
               const uint32_t& epilogue_warp_idx, const uint32_t& lane_idx,
               const epilogue_op_t& epilogue_op,
               const bool& reverse_store_order,
               const dg::arch::ClusterTransactionBarrier* tmem_overlap_barrier,
               const dg::arch::ClusterTransactionBarrier* tmem_empty_barrier,
               const dg::TmaDescriptor& tensor_map_cd) {
    using cd_dtype_t = typename smem_t::cd_dtype;
    // Whether to cast D into FP8 with dynamic per-`kSFGranN` UE8M0 SFs (see `EpilogueDynamicScaledFP8`)
    constexpr bool kWithOutputSF = dg::is_same_v<epilogue_op_t, transform::EpilogueDynamicScaledFP8>;
    constexpr uint32_t kSFGranN = transform::EpilogueDynamicScaledFP8::kSFGranN;
    // TMA checks
    constexpr uint32_t kNumBankGroupBytes = 16;
    constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(cd_dtype_t);
    DG_STATIC_ASSERT(kSwizzleCDMode > 0, "TMA D must be swizzled");
    DG_STATIC_ASSERT(STORE_BLOCK_N % kNumElemsPerBankGroup == 0, "Invalid swizzling");
    DG_STATIC_ASSERT(BLOCK_M % STORE_BLOCK_M == 0, "Invalid block sizes");
    DG_STATIC_ASSERT(BLOCK_N % STORE_BLOCK_N == 0, "Invalid block sizes");
    DG_STATIC_ASSERT(not kWithOutputSF or dg::is_same_v<cd_dtype_t, dg::float_e4m3_t>, "FP8 output requires an E4M3 D");
    DG_STATIC_ASSERT(not kWithOutputSF or STORE_BLOCK_N % kSFGranN == 0, "A store must cover complete SF groups");

    // Share store pipeline between blocks
    auto advance_store_pipeline = [&]() {
        tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages;
    };

    // Iterate over M waves
    constexpr auto kNumMWaves = BLOCK_M / STORE_BLOCK_M;
    #pragma unroll
    for (uint32_t w = 0; w < kNumMWaves; ++ w) {
        // Issue every swizzled atom and pipeline STSM and TMA store
        constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_N;
        #pragma unroll
        for (uint32_t s = 0; s < kNumStores; ++ s, advance_store_pipeline()) {
            const auto store_idx = reverse_store_order ? kNumStores - 1 - s : s;
            auto smem_base_ptr = reinterpret_cast<uint8_t*>(smem.cd[tma_stage_idx]);

            // Swizzled shared memory address of the `bank_group_idx`-th bank group in this
            // warp's atom, reshaping the atom in another view:
            //  - original: `(LAYOUT_AD_M, kSwizzleCDMode / kNumBankGroupBytes)`
            //  - new: `(LAYOUT_AD_M * kSwizzleCDMode / kNumBankGroupBytes / 8, 8)`
            // NOTES: "8" is the number of bank groups, "16" is the swizzling pattern
            const auto get_swizzled_smem_ptr = [&](const uint32_t& bank_group_idx) {
                constexpr bool kHasShortcut = (kSwizzleCDMode / kNumBankGroupBytes) == 8;
                const auto shifted_idx = bank_group_idx + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes);
                auto row = kHasShortcut ? (bank_group_idx / 8 + lane_idx) : (shifted_idx / 8);
                auto col = kHasShortcut ? bank_group_idx : (shifted_idx % 8);
                col ^= row % (kSwizzleCDMode / 16);
                return smem_base_ptr +                                              // Base pointer
                       epilogue_warp_idx * 32 * kSwizzleCDMode +                    // Warp offset
                       row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;   // In-atom offset
            };

            // Wait shared memory to be released
            if (epilogue_warp_idx == 0)
                dg::tma_store_wait<kNumTMAStoreStages - 1>();
            dg::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

            // The pipeline stage
            const auto m_idx = base_m_idx + w * STORE_BLOCK_M;
            const auto n_idx = epilogue_op_t::apply_index_n<STORE_BLOCK_N>(base_n_idx + store_idx * STORE_BLOCK_N);

            // Iterate over TMEM loads: each covers one swizzled bank group per lane, or one
            // complete SF group per lane (one row, `kSFGranN` columns) with FP8 output
            constexpr uint32_t kNumElemsPerLoad = kWithOutputSF ? kSFGranN : kNumElemsPerBankGroup;
            constexpr uint32_t kNumLoads = STORE_BLOCK_N / kNumElemsPerLoad;
            constexpr auto kNumOverlapLoads = math::constexpr_ceil_div(kNumOverlappedTmemCols, kNumElemsPerLoad);
            DG_STATIC_ASSERT(kNumOverlapLoads <= kNumLoads, "The first store block must cover all overlapped columns");

            // A store covering a whole packed word (with word-aligned batches) writes all 4 SF bytes at once
            const auto row_idx = m_idx + epilogue_warp_idx * 32 + lane_idx;
            const bool store_whole_sf_word = kWithOutputSF and kNumLoads == 4 and epilogue_op.shape_n % (4 * kSFGranN) == 0;
            uint32_t sf_word = 0;

            #pragma unroll
            for (uint32_t i = 0; i < kNumLoads; ++ i) {
                const auto load_idx = reverse_store_order ? kNumLoads - 1 - i : i;
                const auto tmem_addr = tmem_base_addr +                                     // Accumulator offset
                                       w * BLOCK_N +                                        // Wave offset
                                       store_idx * STORE_BLOCK_N + load_idx * kNumElemsPerLoad; // In-block offset

                // Read from tensor memory into registers
                // NOTES: empty groups also read the accumulator (safe, as at least one UMMA is
                //        issued) and select zeros afterwards, keeping the loads branchless
                uint32_t values[kNumElemsPerLoad];
                DG_STATIC_ASSERT(kNumElemsPerLoad == 4 or kNumElemsPerLoad == 8 or kNumElemsPerLoad == 32, "Invalid load width");
                using tmem_load_t = dg::conditional_t<kNumElemsPerLoad == 32, dg::SM100_TMEM_LOAD_32dp32b32x,
                                    dg::conditional_t<kNumElemsPerLoad ==  8, dg::SM100_TMEM_LOAD_32dp32b8x,
                                                                                dg::SM100_TMEM_LOAD_32dp32b4x>>;
                [&]<size_t... Is>(dg::index_sequence<Is...>) {
                    tmem_load_t::copy(tmem_addr, values[Is]...);
                }(dg::make_index_sequence<kNumElemsPerLoad>{});
                dg::arch::fence_view_async_tmem_load();
                epilogue_op.apply_values(values);
                #pragma unroll
                for (uint32_t value_idx = 0; value_idx < kNumElemsPerLoad; ++ value_idx)
                    values[value_idx] = is_empty_group ? 0u : values[value_idx];

                // Notify the MMA warp once all overlapped TMEM columns have been read,
                // before any of the cast/store work
                if constexpr (kNumOverlapLoads > 0) {
                    if (w == 0 and s == 0 and i + 1 == kNumOverlapLoads) {
                        ptx::tcgen05_before_thread_sync();
                        tmem_overlap_barrier->arrive(0u);
                    }
                }
                // Notify tensor memory empty (only at the leader CTA) as soon as all loads are in registers
                if (w == kNumMWaves - 1 and s == kNumStores - 1 and i == kNumLoads - 1) {
                    ptx::tcgen05_before_thread_sync();
                    tmem_empty_barrier->arrive(0u);
                }

                // Store into shared memory
                if constexpr (kWithOutputSF) {
                    // Round the accumulator into BF16 pairs first, so the output bitwise matches
                    // a BF16 D followed by the standalone cast (power-of-two scaling of BF16 is exact)
                    nv_bfloat162 values_bf16x2[kNumElemsPerLoad / 2];
                    #pragma unroll
                    for (uint32_t pair_idx = 0; pair_idx < kNumElemsPerLoad / 2; ++ pair_idx)
                        values_bf16x2[pair_idx] = math::cast_into_bf16x2(values[pair_idx * 2], values[pair_idx * 2 + 1]);

                    // Reduce the group amax and compute the SF (the SF exponent floor covers the amax clamp),
                    // then splat the BF16 sf_inv into both packed halves
                    const auto sf_exp = math::get_ue8m0_sf_exp<cd_dtype_t>(math::get_packed_bf16_amax(values_bf16x2));
                    const auto sf_inv = __bfloat162bfloat162(math::get_ue8m0_sf_inv<nv_bfloat16>(sf_exp));

                    // Cast into FP8 and store into shared memory
                    constexpr uint32_t kNumBankGroupsPerLoad = kNumElemsPerLoad / kNumElemsPerBankGroup;
                    #pragma unroll
                    for (uint32_t bank_group_iter_idx = 0; bank_group_iter_idx < kNumBankGroupsPerLoad; ++ bank_group_iter_idx) {
                        const auto smem_ptr = get_swizzled_smem_ptr(load_idx * kNumBankGroupsPerLoad + bank_group_iter_idx);
                        const auto pair_idx = bank_group_iter_idx * (kNumElemsPerBankGroup / 2);
                        ptx::st_shared(smem_ptr,
                                       math::scale_bf16x2_into_fp8x4(values_bf16x2[pair_idx + 0], values_bf16x2[pair_idx + 1], sf_inv, sf_inv),
                                       math::scale_bf16x2_into_fp8x4(values_bf16x2[pair_idx + 2], values_bf16x2[pair_idx + 3], sf_inv, sf_inv),
                                       math::scale_bf16x2_into_fp8x4(values_bf16x2[pair_idx + 4], values_bf16x2[pair_idx + 5], sf_inv, sf_inv),
                                       math::scale_bf16x2_into_fp8x4(values_bf16x2[pair_idx + 6], values_bf16x2[pair_idx + 7], sf_inv, sf_inv));
                    }

                    // Accumulate the SF byte into the packed word, or store it directly
                    if (store_whole_sf_word)
                        sf_word |= sf_exp << (load_idx * 8);
                    else
                        epilogue_op.store_sf(row_idx, n_idx + load_idx * kSFGranN, batch_idx, static_cast<uint8_t>(sf_exp));
                } else if constexpr (dg::is_same_v<cd_dtype_t, float>) {
                    ptx::st_shared(get_swizzled_smem_ptr(load_idx), values[0], values[1], values[2], values[3]);
                } else {
                    ptx::st_shared(
                        get_swizzled_smem_ptr(load_idx),
                        math::cast_into_bf16_and_pack(values[0], values[1]),
                        math::cast_into_bf16_and_pack(values[2], values[3]),
                        math::cast_into_bf16_and_pack(values[4], values[5]),
                        math::cast_into_bf16_and_pack(values[6], values[7])
                    );
                }
            }
            if constexpr (kWithOutputSF) {
                if (store_whole_sf_word)
                    epilogue_op.store_sf(row_idx, n_idx, batch_idx, sf_word);
            }

            // Synchronize all threads and issue TMA
            dg::tma_store_fence();
            dg::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
            if (epilogue_warp_idx == 0 and dg::elect_one_sync()) {
                if constexpr (kGemmType == GemmType::Batched or is_k_grouped_contiguous(kGemmType)) {
                    using dg_tma_t = dg::conditional_t<kWithAccumulation,
                                            dg::SM90_TMA_REDUCE_ADD_3D, dg::SM90_TMA_STORE_3D>;
                    dg_tma_t::copy(&tensor_map_cd, smem_base_ptr, n_idx, m_idx, batch_idx);
                } else {
                    using dg_tma_t = dg::conditional_t<kWithAccumulation,
                                            dg::SM90_TMA_REDUCE_ADD_2D, dg::SM90_TMA_STORE_2D>;
                    dg_tma_t::copy(&tensor_map_cd, smem_base_ptr, n_idx, m_idx);
                }
                dg::tma_store_arrive();
            }
            __syncwarp();
        }
    }
}

} // namespace deep_gemm::epilogue

// ---- include/deep_gemm/epilogue/sm100_store_cd_swap_ab.cuh ----

namespace deep_gemm::epilogue {

template <uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t STORE_BLOCK_M, uint32_t STORE_BLOCK_N,
          uint32_t kSwizzleCDMode,
          uint32_t kNumTMAStoreStages,
          uint32_t kNumUMMAStoreThreads,
          uint32_t kNumOverlappedTmemCols,
          GemmType kGemmType, bool kWithAccumulation,
          typename epilogue_op_t,
          typename smem_t>
DG_DEVICE void
sm100_store_cd_swap_ab(smem_t& smem, uint32_t& tma_stage_idx,
                       const uint32_t& tmem_base_addr,
                       const uint32_t& base_m_idx, const uint32_t& base_n_idx, const uint32_t& batch_idx,
                       const bool& is_empty_group,
                       const uint32_t& effective_m,
                       const uint32_t& epilogue_warp_idx, const uint32_t& lane_idx,
                       const epilogue_op_t& epilogue_op,
                       const bool& reverse_store_order,
                       const dg::arch::ClusterTransactionBarrier* tmem_overlap_barrier,
                       const dg::arch::ClusterTransactionBarrier* tmem_empty_barrier,
                       const dg::TmaDescriptor& tensor_map_cd) {
    using cd_dtype_t = typename smem_t::cd_dtype;
    // NOTES: The epilogue requires a full warpgroup to read all 128 TMEM rows,
    //          implying STORE_BLOCK_N must be 128.
    DG_STATIC_ASSERT(STORE_BLOCK_N == 128, "STORE_BLOCK_N must be 128 to match TMEM rows");

    // Whether to cast D into FP8 with dynamic per-`kSFGranN` UE8M0 SFs (see `EpilogueDynamicScaledFP8`)
    constexpr bool kWithOutputSF = dg::is_same_v<epilogue_op_t, transform::EpilogueDynamicScaledFP8>;
    constexpr uint32_t kSFGranN = transform::EpilogueDynamicScaledFP8::kSFGranN;

    // TMA checks
    constexpr uint32_t STORE_BLOCK_N_ATOM = kSwizzleCDMode / sizeof(cd_dtype_t);
    constexpr uint32_t kNumBankGroupBytes = 16;
    constexpr uint32_t kNumSwizzleAtomRows = 8;
    DG_STATIC_ASSERT(kSwizzleCDMode == 128, "TMA D must be 128B swizzled");
    DG_STATIC_ASSERT(BLOCK_M % STORE_BLOCK_M == 0, "Invalid block sizes");
    DG_STATIC_ASSERT(BLOCK_N % STORE_BLOCK_N == 0, "Invalid block sizes");
    DG_STATIC_ASSERT(STORE_BLOCK_M % kNumSwizzleAtomRows == 0, "Invalid swizzling");
    DG_STATIC_ASSERT(STORE_BLOCK_N % STORE_BLOCK_N_ATOM == 0, "Invalid swizzling");
    DG_STATIC_ASSERT(not kWithOutputSF or dg::is_same_v<cd_dtype_t, dg::float_e4m3_t>, "FP8 output requires an E4M3 D");
    DG_STATIC_ASSERT(not kWithOutputSF or (STORE_BLOCK_M == 16 and kNumUMMAStoreThreads == 128),
                     "Swap-AB FP8 output requires one warpgroup and a 16x128 store shape");

    // Share store pipeline between blocks
    auto advance_store_pipeline = [&]() {
        tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages;
    };

    // Iterate over M blocks. The scheduler skips empty blocks and aligns dynamic effective M to STORE_BLOCK_M.
    const auto num_stores = effective_m / STORE_BLOCK_M;
    for (uint32_t s = 0; s < num_stores; ++ s, advance_store_pipeline()) {
        const auto store_idx = reverse_store_order ? num_stores - 1 - s : s;
        // Wait shared memory to be released
        if (epilogue_warp_idx == 0)
            dg::tma_store_wait<kNumTMAStoreStages - 1>();
        dg::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

        // Store into shared memory
        constexpr uint32_t kNumTmemLoads = STORE_BLOCK_M / kNumSwizzleAtomRows;
        #pragma unroll
        for (uint32_t i = 0; i < kNumTmemLoads; ++ i) {
            const auto load_idx = reverse_store_order ? kNumTmemLoads - 1 - i : i;
            uint32_t tmem_addr = tmem_base_addr +
                                 store_idx * STORE_BLOCK_M +    // Store stage offset
                                 load_idx * kNumSwizzleAtomRows; // In-block offset
            uint32_t values[kNumSwizzleAtomRows];

            // Warps cooperatively write an atomic block to shared memory
            DG_STATIC_ASSERT(STORE_BLOCK_N_ATOM % 32 == 0, "Invalid block sizes");
            constexpr uint32_t kNumWarpsPerAtom = STORE_BLOCK_N_ATOM / 32;
            uint32_t outer_atom_offset = (epilogue_warp_idx / kNumWarpsPerAtom) * STORE_BLOCK_M * kSwizzleCDMode;
            uint32_t inner_atom_offset = load_idx * kNumSwizzleAtomRows * kSwizzleCDMode;
            auto smem_base_ptr = reinterpret_cast<uint8_t*>(smem.cd[tma_stage_idx]) + outer_atom_offset + inner_atom_offset;

            // Read from tensor memory into registers
            // NOTES: empty groups also read the accumulator (safe, as at least one UMMA is
            //        issued) and select zeros afterwards, keeping the loads branchless
            if constexpr (dg::is_same_v<cd_dtype_t, float>) {
                // NOTES: the FP32 store does not use STSM, so the plain `.32x32b` layout works
                dg::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_addr, values[0], values[1], values[2], values[3],
                                                                 values[4], values[5], values[6], values[7]);
                dg::arch::fence_view_async_tmem_load();
                epilogue_op.apply_values(values);
            } else {
                // Load from TMEM using `.16x256b` shape to satisfy the STSM layout requirements
                // (`.b16` for the BF16 store, `.b8` for the FP8 store): each lane receives
                // 2 rows (TMEM columns `2 * (lane % 4)` onwards) x 2 columns (dps `lane / 4` and `lane / 4 + 8`)
                // Start from lane index 0
                dg::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                                                       values[0], values[1], values[2], values[3]);
                // Start from lane index 16
                dg::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                                                       values[4], values[5], values[6], values[7]);
                dg::arch::fence_view_async_tmem_load();
                epilogue_op.apply_values(values);
            }
            #pragma unroll
            for (uint32_t value_idx = 0; value_idx < kNumSwizzleAtomRows; ++ value_idx)
                values[value_idx] = is_empty_group ? 0u : values[value_idx];

            // Notify the MMA warp once all overlapped TMEM columns have been read,
            // before any of the cast/store work
            constexpr auto kNumOverlapLoads = math::constexpr_ceil_div(kNumOverlappedTmemCols, kNumSwizzleAtomRows);
            if constexpr (kNumOverlapLoads > 0) {
                // NOTES: dynamic UMMA N writes only `effective_m` columns, so compile-time overlap
                //        columns beyond the available TMEM loads were never produced
                const auto num_loads_before_arrive = dg::min<uint32_t>(kNumOverlapLoads, num_stores * kNumTmemLoads);
                if (s * kNumTmemLoads + i + 1 == num_loads_before_arrive) {
                    ptx::tcgen05_before_thread_sync();
                    tmem_overlap_barrier->arrive(0u);
                }
            }
            // Notify tensor memory empty (only at the leader CTA) as soon as all loads are in registers
            if (s == num_stores - 1 and i == kNumTmemLoads - 1) {
                ptx::tcgen05_before_thread_sync();
                tmem_empty_barrier->arrive(0u);
            }

            // Store into shared memory
            if constexpr (kWithOutputSF) {
                // Round the accumulator into BF16 pairs first, so the output bitwise matches
                // a BF16 D followed by the standalone cast (power-of-two scaling of BF16 is exact)
                // NOTES: with the `.16x256b` load, every pair holds this lane's row pair at one
                //        of its 4 columns, so all packed multiplications share one per-half sf_inv
                nv_bfloat162 values_bf16x2[kNumSwizzleAtomRows / 2];
                #pragma unroll
                for (uint32_t pair_idx = 0; pair_idx < kNumSwizzleAtomRows / 2; ++ pair_idx)
                    values_bf16x2[pair_idx] = math::cast_into_bf16x2(values[pair_idx * 2], values[pair_idx * 2 + 1]);

                // Reduce the row pair's amax and compute the SFs (the SF exponent floor covers
                // the amax clamp): the 8 lanes sharing `lane_idx % 4` cover the pair's two SF
                // groups, so a packed tree plus a cross-group partial reduction is enough
                const auto amax_pair = math::warp_reduce_max<4, true>(
                    __hmax2(__hmax2(__habs2(values_bf16x2[0]), __habs2(values_bf16x2[1])),
                            __hmax2(__habs2(values_bf16x2[2]), __habs2(values_bf16x2[3]))));
                const auto sf_exp_lower = math::get_ue8m0_sf_exp<cd_dtype_t>(amax_pair.x);
                const auto sf_exp_upper = math::get_ue8m0_sf_exp<cd_dtype_t>(amax_pair.y);
                const auto sf_inv = nv_bfloat162(math::get_ue8m0_sf_inv<nv_bfloat16>(sf_exp_lower),
                                                 math::get_ue8m0_sf_inv<nv_bfloat16>(sf_exp_upper));

                // Cast into FP8 and store with a transposing STSM: the loaded fragment matches
                // `stmatrix.m16n8.trans.b8`, with one 8x16 matrix per 16 columns
                const auto smem_ptr = smem_base_ptr + (lane_idx % 8) * kSwizzleCDMode +
                                      ((epilogue_warp_idx * 2 + lane_idx / 8) ^ (lane_idx % 8)) * kNumBankGroupBytes;
                ptx::SM100_U8x8_STSM_T<uint32_t>::copy(
                    math::scale_bf16x2_into_fp8x4(values_bf16x2[0], values_bf16x2[1], sf_inv, sf_inv),
                    math::scale_bf16x2_into_fp8x4(values_bf16x2[2], values_bf16x2[3], sf_inv, sf_inv),
                    smem_ptr);

                // Store the SF byte into the packed word
                // NOTES: lanes 0-3 own their row pair's even row, lanes 4-7 the odd row
                if (lane_idx < kNumSwizzleAtomRows)
                    epilogue_op.store_sf(base_m_idx + store_idx * STORE_BLOCK_M + load_idx * kNumSwizzleAtomRows +
                                          2 * (lane_idx % 4) + lane_idx / 4,
                                      base_n_idx + epilogue_warp_idx * kSFGranN, batch_idx,
                                      static_cast<uint8_t>(lane_idx < 4 ? sf_exp_lower : sf_exp_upper));
            } else if constexpr (dg::is_same_v<cd_dtype_t, float>) {
                uint32_t col = lane_idx / 4;

                #pragma unroll
                for (uint32_t row = 0; row < kNumSwizzleAtomRows; ++ row) {
                    auto smem_ptr = smem_base_ptr + row * (kNumBankGroupBytes * 8)
                                                  + (col ^ row) * kNumBankGroupBytes
                                                  + (lane_idx % 4) * sizeof(float);
                    ptx::st_shared(reinterpret_cast<uint32_t*>(smem_ptr), values[row]);
                }
            } else {
                // Destination shared memory address
                uint32_t row = lane_idx % 8;
                uint32_t col = (epilogue_warp_idx % 2) * 4 + lane_idx / 8;
                auto smem_ptr = smem_base_ptr + row * (kNumBankGroupBytes * 8)
                                              + (col ^ row) * kNumBankGroupBytes;

                // Store matrix with transposition
                ptx::SM90_U32x4_STSM_T<int>::copy(math::cast_into_bf16_and_pack(values[0], values[1]),
                                                  math::cast_into_bf16_and_pack(values[2], values[3]),
                                                  math::cast_into_bf16_and_pack(values[4], values[5]),
                                                  math::cast_into_bf16_and_pack(values[6], values[7]),
                                                  smem_ptr);
            }
        }

        // Synchronize all threads and issue TMA
        dg::tma_store_fence();
        dg::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
        if (epilogue_warp_idx == 0 and dg::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < STORE_BLOCK_N / STORE_BLOCK_N_ATOM; ++ i) {
                auto smem_ptr = smem.cd[tma_stage_idx] + i * STORE_BLOCK_M * STORE_BLOCK_N_ATOM;
                uint32_t m_idx = base_m_idx + store_idx * STORE_BLOCK_M;
                uint32_t n_idx = epilogue_op_t::apply_index_n<STORE_BLOCK_N_ATOM>(base_n_idx + i * STORE_BLOCK_N_ATOM);

                // Issue 2D or 3D TMA store
                if constexpr (kGemmType == GemmType::Batched or is_k_grouped_contiguous(kGemmType)) {
                    using dg_tma_t = dg::conditional_t<kWithAccumulation,
                        dg::SM90_TMA_REDUCE_ADD_3D, dg::SM90_TMA_STORE_3D>;
                    dg_tma_t::copy(&tensor_map_cd, smem_ptr, n_idx, m_idx, batch_idx);
                } else {
                    using dg_tma_t = dg::conditional_t<kWithAccumulation,
                        dg::SM90_TMA_REDUCE_ADD_2D, dg::SM90_TMA_STORE_2D>;
                    dg_tma_t::copy(&tensor_map_cd, smem_ptr, n_idx, m_idx);
                }
            }
            dg::tma_store_arrive();
        }
        __syncwarp();
    }
}

} // namespace deep_gemm::epilogue

// ---- include/deep_gemm/layout/gemm.cuh ----

namespace deep_gemm::layout {

template <uint32_t kNumStages, uint32_t kNumEpilogueStages, uint32_t kNumTMAStoreStages,
          uint32_t LOAD_BLOCK_M, uint32_t LOAD_BLOCK_N, uint32_t BLOCK_K,
          uint32_t STORE_BLOCK_M, uint32_t STORE_BLOCK_N,
          typename cd_dtype_t>
struct SM100BF16GemmSharedStorage {
    using Barrier = dg::arch::ClusterTransactionBarrier;
    using cd_dtype = cd_dtype_t;
    using cd_stage_t = cd_dtype_t[STORE_BLOCK_M * STORE_BLOCK_N];
    using a_stage_t = dg::bfloat16_t[LOAD_BLOCK_M * BLOCK_K];
    using b_stage_t = dg::bfloat16_t[LOAD_BLOCK_N * BLOCK_K];
    DG_STATIC_ASSERT(kNumTMAStoreStages * sizeof(cd_stage_t) % 1024 == 0 and
                     sizeof(a_stage_t) % 1024 == 0 and sizeof(b_stage_t) % 1024 == 0,
                     "Shared memory of A/B must be aligned to 1024 bytes");
    alignas(1024) cd_stage_t cd[kNumTMAStoreStages];
    a_stage_t a[kNumStages];
    b_stage_t b[kNumStages];
    Barrier full_barriers[kNumStages];
    Barrier empty_barriers[kNumStages];
    Barrier tmem_full_barriers[kNumEpilogueStages];
    Barrier tmem_empty_barriers[kNumEpilogueStages];
    // Keep the BF16 barrier layout aligned with the block-scaled GEMM layout.
    Barrier reserved_barriers[kNumStages];
    Barrier tensor_core_full_barrier;
    uint32_t tmem_ptr;
};

template <uint32_t kNumStages, uint32_t kNumEpilogueStages, uint32_t kNumTMAStoreStages,
          uint32_t LOAD_BLOCK_M, uint32_t LOAD_BLOCK_N, uint32_t BLOCK_K,
          uint32_t STORE_BLOCK_M, uint32_t STORE_BLOCK_N,
          uint32_t SF_BLOCK_M, uint32_t SF_BLOCK_N, uint32_t SF_BLOCK_K,
          typename a_dtype_t, typename b_dtype_t, typename cd_dtype_t>
struct SM100FP8FP4GemmSharedStorage {
    using Barrier = dg::arch::ClusterTransactionBarrier;
    using cd_dtype = cd_dtype_t;
    using cd_stage_t = cd_dtype_t[STORE_BLOCK_M * STORE_BLOCK_N];
    using a_stage_t = a_dtype_t[LOAD_BLOCK_M * BLOCK_K / get_smem_pack_factor<a_dtype_t>()];
    using b_stage_t = b_dtype_t[LOAD_BLOCK_N * BLOCK_K / get_smem_pack_factor<b_dtype_t>()];
    using sfa_stage_t = uint32_t[SF_BLOCK_M * SF_BLOCK_K];
    using sfb_stage_t = uint32_t[SF_BLOCK_N * SF_BLOCK_K];
    DG_STATIC_ASSERT(kNumTMAStoreStages * sizeof(cd_stage_t) % 1024 == 0 and
                     sizeof(a_stage_t) % 1024 == 0 and sizeof(b_stage_t) % 1024 == 0,
                     "Shared memory of A/B must be aligned to 1024 bytes");
    alignas(1024) cd_stage_t cd[kNumTMAStoreStages];
    a_stage_t a[kNumStages];
    b_stage_t b[kNumStages];
    sfa_stage_t sfa[kNumStages];
    sfb_stage_t sfb[kNumStages];
    Barrier full_barriers[kNumStages];
    Barrier sf_full_barriers[kNumStages];
    Barrier empty_barriers[kNumStages];
    Barrier tmem_full_barriers[kNumEpilogueStages];
    Barrier tmem_empty_barriers[kNumEpilogueStages];
    Barrier tmem_overlap_barriers[kNumEpilogueStages];
    uint32_t tmem_ptr;
};

} // namespace deep_gemm::layout

// ---- include/deep_gemm/mma/sm100.cuh ----

namespace deep_gemm::mma::sm100 {

/// Shared memory descriptor
DG_DEVICE
dg::UMMA::SmemDescriptor make_smem_desc(dg::UMMA::LayoutType layout, void* smem_ptr,
                                          const uint32_t& stride_byte_offset, const uint32_t& leading_byte_offset) {
    dg::UMMA::SmemDescriptor desc;

    // Set the version for SM100
    desc.version_ = 1;

    // Legacy mode
    desc.lbo_mode_ = 0;

    // Layout
    desc.layout_type_ = static_cast<uint8_t>(layout);

    // Start address
    const auto uint_ptr = dg::cast_smem_ptr_to_uint(smem_ptr);
    desc.start_address_ = static_cast<uint16_t>(uint_ptr >> 4);

    // Base offset
    desc.base_offset_ = 0;

    // SBO and LBO
    desc.stride_byte_offset_ = stride_byte_offset >> 4;
    desc.leading_byte_offset_ = leading_byte_offset >> 4;

    return desc;
}

DG_DEVICE
dg::UMMA::SmemDescriptor make_sf_desc(void* smem_ptr) {
    // NOTES: the UTCCP layout is K-major by default
    // Atom size: 8 x 128 bits
    // {SBO, LBO} means the byte stride between atoms on {MN, K}
    // Since the UTCCP we used is 128b-wide (only 1 atom on K), so LBO can be zero
    return make_smem_desc(dg::UMMA::LayoutType::SWIZZLE_NONE, smem_ptr, 8 * 16, 0);
}

DG_DEVICE
void replace_smem_desc_addr(dg::UMMA::SmemDescriptor& desc, const void* smem_ptr) {
    const auto uint_ptr = dg::cast_smem_ptr_to_uint(smem_ptr);
    desc.start_address_ = static_cast<uint16_t>(uint_ptr >> 4);
}

DG_DEVICE
static uint32_t get_atom_base(const dg::UMMA::LayoutType& layout_type) {
    return layout_type == dg::UMMA::LayoutType::SWIZZLE_128B_BASE32B ? 32 : 16;
}

/// UMMA descriptors
// ReSharper disable once CppNotAllPathsReturnValue
template <dg::UMMA::Major kMajorMode, uint32_t kSwizzleMode, bool kUseBase32, typename dtype_t>
constexpr static dg::UMMA::LayoutType to_umma_layout_type() {
    DG_STATIC_ASSERT(kSwizzleMode == 0 or kSwizzleMode == 16 or
                     kSwizzleMode == 32 or kSwizzleMode == 64 or
                     kSwizzleMode == 128, "Invalid swizzling mode");
    // A special case
    if constexpr ((dg::is_same_v<dtype_t, float> and kMajorMode == dg::UMMA::Major::MN) or kUseBase32) {
        DG_STATIC_ASSERT(kUseBase32, "Invalid swizzling base");
        return dg::UMMA::LayoutType::SWIZZLE_128B_BASE32B;
    }

    // Normal cases
    if constexpr (kSwizzleMode == 0)   return dg::UMMA::LayoutType::SWIZZLE_NONE;
    if constexpr (kSwizzleMode == 16)  return dg::UMMA::LayoutType::SWIZZLE_NONE;
    if constexpr (kSwizzleMode == 32)  return dg::UMMA::LayoutType::SWIZZLE_32B;
    if constexpr (kSwizzleMode == 64)  return dg::UMMA::LayoutType::SWIZZLE_64B;
    if constexpr (kSwizzleMode == 128) return dg::UMMA::LayoutType::SWIZZLE_128B;
}

template <dg::UMMA::Major kMajorMode, uint32_t BLOCK_MN, uint32_t kSwizzleMode, typename dtype_t>
DG_DEVICE
constexpr uint32_t get_umma_desc_stride_k() {
    return kMajorMode == dg::UMMA::Major::K ? 1 : tma::get_inner_block_atom_size<BLOCK_MN, kSwizzleMode, 1, dtype_t>();
}

template <dg::UMMA::Major kMajorMode, uint32_t BLOCK_MN, uint32_t kSwizzleMode, typename dtype_t>
DG_DEVICE
uint32_t advance_umma_desc_lo(const uint32_t& base, const uint32_t& offset, const uint32_t& k_idx) {
    constexpr uint32_t kPackFactor = get_smem_pack_factor<dtype_t>();
    const uint32_t byte_offset = (offset + k_idx * get_umma_desc_stride_k<kMajorMode, BLOCK_MN, kSwizzleMode, dtype_t>())
                                 * static_cast<uint32_t>(sizeof(dtype_t)) / kPackFactor;
    return base + (byte_offset >> 4u);
}

template <dg::UMMA::Major kMajorMode, uint32_t BLOCK_MN, uint32_t BLOCK_K, uint32_t kSwizzleMode,
          bool kUseBase32 = false, typename dtype_t>
DG_DEVICE
dg::UMMA::SmemDescriptor make_umma_desc(dtype_t* base_smem_ptr, uint32_t mn_idx, uint32_t k_idx) {
    // NOTES: `base_smem_ptr` must use the logical SMEM element type used by UMMA descriptors.
    constexpr uint32_t kPackFactor = get_smem_pack_factor<dtype_t>();
    DG_STATIC_ASSERT(kPackFactor == 1 or sizeof(dtype_t) == 1, "Packing expects a 1-byte storage type");
    const uint32_t stride_k = get_umma_desc_stride_k<kMajorMode, BLOCK_MN, kSwizzleMode, dtype_t>();
    const auto layout_type = to_umma_layout_type<kMajorMode, kSwizzleMode, kUseBase32, dtype_t>();
    const auto num_non_contiguous = 128 / get_atom_base(layout_type);
    if constexpr (kMajorMode == dg::UMMA::Major::K) {
        // NOTES: for K-major layout, the swizzle must be the same as `BLOCK_K` elements in bytes
        // also, atom index must be 0, so that each block has exactly one swizzle atom on the K axis
        DG_STATIC_ASSERT(kSwizzleMode * kPackFactor == BLOCK_K * sizeof(dtype_t), "Unexpected value");

        // Atom size: 8 x `kSwizzleMode` (in bytes, on K)
        // {SBO, LBO} means the byte stride between atoms on {MN, K}
        // NOTES: on K, there is only 1 atom as asserted previously, so LBO can be 0
        const uint32_t stride_byte_offset = num_non_contiguous * BLOCK_K * sizeof(dtype_t) / kPackFactor;
        const uint32_t leading_byte_offset = 0;
        const auto byte_ptr = reinterpret_cast<uint8_t*>(base_smem_ptr) +
                              (mn_idx * BLOCK_K + k_idx * stride_k) * sizeof(dtype_t) / kPackFactor;
        return make_smem_desc(layout_type, byte_ptr, stride_byte_offset, leading_byte_offset);
    } else {
        DG_STATIC_ASSERT(kPackFactor <= 1, "Packing only supports K-major");
        constexpr uint32_t BLOCK_MN_ATOM = tma::get_inner_block_atom_size<BLOCK_MN, kSwizzleMode, 1, dtype_t>();

        // Must have no in-atom MN-idx
        // NOTES: no worries for the runtime assert, the `mn_idx` are constants at compilation time
        DG_DEVICE_ASSERT(mn_idx % BLOCK_MN_ATOM == 0);
        DG_STATIC_ASSERT(kSwizzleMode > 0, "Invalid swizzling");

        // Atom size: `kSwizzleMode` (in bytes, on MN) x 8
        // NOTES: `kSwizzleMode == 16` mean non-swizzling but interleaving
        // {SBO, LBO} means the byte stride between atoms on {K, MN} for swizzling
        // {SBO, LBO} means the byte stride between atoms on {MN, K} for non-swizzling
        uint32_t stride_byte_offset = num_non_contiguous * BLOCK_MN_ATOM * sizeof(dtype_t);
        uint32_t leading_byte_offset = BLOCK_K * BLOCK_MN_ATOM * sizeof(dtype_t);
        if constexpr (kSwizzleMode == 16)
            math::swap(stride_byte_offset, leading_byte_offset);
        return make_smem_desc(layout_type,
                              base_smem_ptr + mn_idx * BLOCK_K + k_idx * stride_k,
                              stride_byte_offset, leading_byte_offset);
    }
}

DG_DEVICE uint64_t make_runtime_instr_desc_with_sf_id(
    dg::UMMA::InstrDescriptorBlockScaled desc, const uint32_t& sfa_id, const uint32_t& sfb_id) {
    desc.a_sf_id_ = sfa_id, desc.b_sf_id_ = sfb_id;
    return static_cast<uint64_t>(static_cast<uint32_t>(desc)) << 32;
}

DG_DEVICE void update_instr_desc_with_umma_n(
    dg::UMMA::InstrDescriptorBlockScaled& desc, const uint32_t& umma_n) {
    desc.n_dim_ = umma_n >> 3;
}

DG_DEVICE void update_instr_desc_with_umma_n(
    dg::UMMA::InstrDescriptor& desc, const uint32_t& umma_n) {
    desc.n_dim_ = umma_n >> 3;
}

} // namespace deep_gemm::mma::sm100

namespace deep_gemm {

// Executable BF16 GEMM bodies live in include/aka/*.inc and are included
// directly by explicit non-template entries, including dynamically bound configs.
// This header retains only their shared dependencies; the duplicate unused
// sm100_bf16_gemm_impl global template has been removed.

};  // namespace deep_gemm

#pragma clang diagnostic pop
