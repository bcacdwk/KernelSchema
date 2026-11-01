#pragma once
// include/dg_pure/dg_pure.cuh

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

// include/deep_gemm/common/math.cuh

#include <cuda/std/cstdint>

// include/deep_gemm/common/compile.cuh

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

// include/deep_gemm/common/exception.cuh
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

// include/deep_gemm/ptx/utils.cuh
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

// include/deep_gemm/common/tma_copy.cuh

// include/deep_gemm/common/packing.cuh

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

// include/deep_gemm/common/utils.cuh
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

