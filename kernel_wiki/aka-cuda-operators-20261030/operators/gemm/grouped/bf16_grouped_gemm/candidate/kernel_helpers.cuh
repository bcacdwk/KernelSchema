#pragma once

// include/deep_gemm/impls/sm100_bf16_gemm.cuh
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include "cuda_primitives.cuh"

// include/deep_gemm/comm/barrier.cuh

namespace deep_gemm::comm {
DG_DEVICE void cluster_sync_with_relaxed_arrive() {
    dg::cluster_arrive_relaxed();
    dg::cluster_wait();
}
}

// include/deep_gemm/scheduler/gemm.cuh

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

// include/deep_gemm/common/types.cuh

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

// include/deep_gemm/epilogue/sm100_store_cd.cuh

// include/deep_gemm/epilogue/transform.cuh

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

// include/deep_gemm/ptx/ld_st.cuh
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

// include/deep_gemm/ptx/tcgen05.cuh

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

// include/deep_gemm/epilogue/sm100_store_cd_swap_ab.cuh

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

// include/deep_gemm/layout/gemm.cuh

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

// include/deep_gemm/mma/sm100.cuh

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
