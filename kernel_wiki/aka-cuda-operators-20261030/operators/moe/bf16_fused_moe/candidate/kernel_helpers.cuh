#pragma once

// include/deep_gemm/impls/sm100_bf16_mega_moe.cuh
#include <cstdint>

#include "cuda_primitives.cuh"

// include/deep_gemm/comm/barrier.cuh

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

// include/deep_gemm/layout/sym_buffer.cuh

namespace deep_gemm::layout {

constexpr static uint32_t kNumMaxRanks = 72;

template <uint32_t kNumRanks = kNumMaxRanks>
struct SymBuffer {
    int64_t base;
    int64_t offsets[kNumMaxRanks];
    uint32_t rank_idx;

    DG_STATIC_ASSERT(kNumRanks <= kNumMaxRanks, "Too many ranks");

    SymBuffer() = default;

    template <typename Container>
    explicit SymBuffer(const Container& c, const uint32_t& rank_idx): rank_idx(rank_idx) {
        const auto size = static_cast<uint32_t>(c.size());
        base = c[rank_idx];
        for (uint32_t i = 0; i < kNumMaxRanks; ++ i)
            offsets[i] = i < size ? (c[i] - base) : 0;
    }

#if defined(__CUDA_ARCH__) or defined(__CLION_IDE__)
    template <typename ptr_t = void*>
    DG_DEVICE ptr_t get_base_ptr() const {
        return reinterpret_cast<ptr_t>(base);
    }

    template <typename ptr_t>
    DG_DEVICE ptr_t map(const ptr_t& ptr, const uint32_t& dst_rank_idx) const {
        if constexpr (kNumRanks == 1)
            return ptr;

        int64_t mapped_ptr = offsets[dst_rank_idx] + reinterpret_cast<int64_t>(ptr);
        return *reinterpret_cast<ptr_t*>(&mapped_ptr);
    }
#endif
};

} // namespace deep_gemm::layout

// include/deep_gemm/layout/mega_moe.cuh
#include <cstddef>

namespace deep_gemm::layout {

static constexpr int kNumCandidateBlockMs = 8;
static constexpr int kCandidateBlockM[kNumCandidateBlockMs] = {8, 16, 32, 64, 96, 128, 192, 240};
static constexpr int kMaxCandidateBlockM = 240;
static constexpr int kMinCandidateBlockM = 8;
static constexpr int kLCMCandidateBlockM = 1920;

// Pool capacity for shared expert token pool: worst-case total tokens + per-expert BLOCK_M alignment padding, among all possible BLOCK_M
template <typename T>
DG_HOST_DEVICE constexpr T get_num_max_pool_tokens(T num_ranks, T num_max_tokens_per_rank, T num_topk,
                                                        T num_experts_per_rank) {
    const auto num_max_recv_tokens = num_ranks * num_max_tokens_per_rank;
    const auto num_max_experts_per_token = math::constexpr_min(num_topk, num_experts_per_rank);
    return math::constexpr_align(
        num_max_recv_tokens * num_max_experts_per_token + num_experts_per_rank * (static_cast<T>(kMaxCandidateBlockM) - 1),
        static_cast<T>(kLCMCandidateBlockM));
}

// SF pool capacity: all experts share a contiguous SF region, sized by pool blocks × SF_BLOCK_M
template <typename T>
DG_HOST_DEVICE constexpr T get_num_sf_ring_tokens(T num_ring_tokens, T block_m) {
    return (num_ring_tokens / block_m) * math::constexpr_align(block_m, static_cast<T>(128));
}

// Shared L2 input SF capacity: worst-case aligned SF pages over all candidate BLOCK_M.
template <typename T>
DG_HOST_DEVICE constexpr T get_num_max_shared_sf_tokens(const T& num_max_tokens_per_rank) {
    return math::constexpr_ceil_div<T>(num_max_tokens_per_rank, kMinCandidateBlockM) * 128;
}

// Per-token source metadata for combine write-back
struct TokenSrcMetadata {
    uint32_t rank_idx;
    uint32_t token_idx;
    uint32_t topk_idx;

    TokenSrcMetadata() = default;

    DG_DEVICE TokenSrcMetadata(const uint32_t& rank_idx, const uint32_t& token_idx, const uint32_t& topk_idx):
        rank_idx(rank_idx), token_idx(token_idx), topk_idx(topk_idx) {}
};

struct alignas(128) MegaMoESignals {
    static constexpr uint32_t kNumMaxGridSyncCounters = 4;
    static constexpr uint32_t kNumMaxExperts = 2048;
    static constexpr uint32_t kNumMaxExpertsPerRank = 512;
    static constexpr uint32_t kNumMaxRingBlocks = 1u << 20;  // 20 MiB for the ring signals
    static constexpr uint32_t kNumMaxSharedL2Blocks = 1u << 15;  // 128 KiB

    // Grid and NVLink synchronization
    uint32_t grid_sync_count[kNumMaxGridSyncCounters];
    uint32_t nvl_barrier_counter;
    int nvl_barrier_signals[2];

    // Task scheduling
    uint32_t l1_task_count;
    uint32_t l2_task_count;
    uint32_t shared_l1_task_count;
    uint32_t shared_l2_task_count;

    // Combine readiness: `combine_ready_grid_idx[peer] == own grid index` means the peer's L2 writes into this
    // rank are done; grid indices are unique per launch, so no reset is needed. Peers push theirs during dispatch.
    alignas(128) uint64_t combine_ready_grid_idx[kNumMaxRanks];
    alignas(128) uint64_t peer_grid_idx[kNumMaxRanks];

    // Expert token counts
    alignas(128) uint64_t expert_send_count[kNumMaxExperts];
    uint64_t expert_recv_count[kNumMaxExperts];
    uint64_t expert_recv_count_sum[kNumMaxExpertsPerRank];

    // Routed-expert ring signals, `l2_full_mask` has one bit per L1 N block
    uint32_t l1_full_count[kNumMaxRingBlocks];
    uint32_t l1_empty_count[kNumMaxRingBlocks];
    uint64_t l2_full_mask[kNumMaxRingBlocks];
    uint32_t l2_empty_count[kNumMaxRingBlocks];

    // Shared-expert signals
    uint32_t shared_l2_full_count[kNumMaxSharedL2Blocks];
};
DG_STATIC_ASSERT(offsetof(MegaMoESignals, combine_ready_grid_idx) == 128, "Invalid control signal layout");

struct Workspace {
    MegaMoESignals* signals;
    uint32_t num_ranks, num_experts;
    uint32_t num_experts_per_rank;
    uint32_t num_max_tokens_per_rank;
    uint32_t num_shared_l2_pool_blocks;

    // Full-pool span used by non-ring token metadata
    uint32_t num_max_pool_tokens;

    Workspace() = default;

    DG_HOST_DEVICE
    Workspace(void* base,
              const uint32_t& num_ranks,
              const uint32_t& num_experts,
              const uint32_t& num_max_tokens_per_rank,
              const uint32_t& num_topk,
              const uint32_t& num_ring_tokens):
        signals(static_cast<MegaMoESignals*>(base)),
        num_ranks(num_ranks), num_experts(num_experts),
        num_max_tokens_per_rank(num_max_tokens_per_rank) {
        num_experts_per_rank = num_experts / num_ranks;
        num_max_pool_tokens = get_num_max_pool_tokens(num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
        num_shared_l2_pool_blocks = math::ceil_div<uint32_t>(num_max_tokens_per_rank, kMinCandidateBlockM);

        DG_UNIFIED_ASSERT(num_ranks > 0);
        DG_UNIFIED_ASSERT(num_ranks <= kNumMaxRanks);
        DG_UNIFIED_ASSERT(num_experts % num_ranks == 0);
        DG_UNIFIED_ASSERT(num_experts <= MegaMoESignals::kNumMaxExperts);
        DG_UNIFIED_ASSERT(num_experts_per_rank <= MegaMoESignals::kNumMaxExpertsPerRank);
        DG_UNIFIED_ASSERT(num_ring_tokens <= MegaMoESignals::kNumMaxRingBlocks * kMinCandidateBlockM);
        DG_UNIFIED_ASSERT(num_shared_l2_pool_blocks <= MegaMoESignals::kNumMaxSharedL2Blocks);
    }

    DG_HOST_DEVICE
    uint64_t get_num_bytes() const {
        uint64_t num_bytes = 0;

        // Fixed-position global and expert signals
        num_bytes += sizeof(MegaMoESignals);

        // Source token-topk: [local expert][source rank][token], with distinct experts per token
        num_bytes += num_experts * num_max_tokens_per_rank * sizeof(uint32_t);

        // Combine push source indices (full)
        num_bytes += num_max_pool_tokens * sizeof(TokenSrcMetadata);

        // Align to TMA descriptor requirements
        num_bytes = math::align<uint64_t>(num_bytes, 16);
        return num_bytes;
    }

    DG_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(signals, get_num_bytes());
    }

    template <uint32_t kIndex = 0>
    DG_DEVICE
    uint32_t* get_grid_sync_count_ptr() const {
        DG_STATIC_ASSERT(kIndex < MegaMoESignals::kNumMaxGridSyncCounters, "Grid sync index out of bounds");
        return signals->grid_sync_count + kIndex;
    }

    DG_DEVICE
    uint32_t* get_nvl_barrier_counter_ptr() const {
        return &signals->nvl_barrier_counter;
    }

    DG_DEVICE
    int* get_nvl_barrier_signal_ptr(const uint32_t& phase) const {
        // NOTES: the signal is signed, as we may minus
        return signals->nvl_barrier_signals + phase;
    }

    DG_DEVICE
    uint64_t* get_combine_ready_grid_idx_ptr(const uint32_t& peer_rank_idx = 0) const {
        return signals->combine_ready_grid_idx + peer_rank_idx;
    }

    DG_DEVICE
    uint64_t* get_peer_grid_idx_ptr(const uint32_t& peer_rank_idx = 0) const {
        return signals->peer_grid_idx + peer_rank_idx;
    }

    DG_DEVICE
    uint32_t* get_l1_task_count_ptr() const {
        return &signals->l1_task_count;
    }

    DG_DEVICE
    uint32_t* get_l2_task_count_ptr() const {
        return &signals->l2_task_count;
    }

    DG_DEVICE
    uint32_t* get_shared_l1_task_count_ptr() const {
        return &signals->shared_l1_task_count;
    }

    DG_DEVICE
    uint32_t* get_shared_l2_task_count_ptr() const {
        return &signals->shared_l2_task_count;
    }

    DG_DEVICE
    uint64_t* get_expert_send_count_ptr(const uint32_t& expert_idx = 0) const {
        return signals->expert_send_count + expert_idx;
    }

    DG_DEVICE
    uint64_t* get_expert_recv_count_ptr(
        const uint32_t& rank_idx = 0, const uint32_t& expert_idx = 0) const {
        return signals->expert_recv_count + rank_idx * num_experts_per_rank + expert_idx;
    }

    DG_DEVICE
    uint64_t* get_expert_recv_count_sum_ptr(const uint32_t& expert_idx = 0) const {
        return signals->expert_recv_count_sum + expert_idx;
    }

    DG_DEVICE
    uint32_t* get_l1_full_count_ptr(const uint32_t& ring_block_idx = 0) const {
        return signals->l1_full_count + ring_block_idx;
    }

    DG_DEVICE
    uint32_t* get_l1_empty_count_ptr(const uint32_t& ring_block_idx = 0) const {
        return signals->l1_empty_count + ring_block_idx;
    }

    DG_DEVICE
    uint64_t* get_l2_full_mask_ptr(const uint32_t& ring_block_idx = 0) const {
        return signals->l2_full_mask + ring_block_idx;
    }

    DG_DEVICE
    uint32_t* get_l2_empty_count_ptr(const uint32_t& ring_block_idx = 0) const {
        return signals->l2_empty_count + ring_block_idx;
    }

    DG_DEVICE
    uint32_t* get_shared_l2_full_count_ptr(const uint32_t& block_idx = 0) const {
        return signals->shared_l2_full_count + block_idx;
    }

    // For dispatch pulling
    DG_DEVICE
    uint32_t* get_src_token_topk_idx_ptr(
        const uint32_t& expert_idx = 0, const uint32_t& rank_idx = 0, const uint32_t& token_idx = 0) const {
        const auto offset = (expert_idx * num_ranks + rank_idx) * num_max_tokens_per_rank + token_idx;
        return math::advance_ptr<uint32_t>(signals, sizeof(MegaMoESignals)) + offset;
    }

    // For combine usages (full)
    DG_DEVICE
    TokenSrcMetadata* get_token_src_metadata_ptr(const uint32_t& pool_token_idx = 0) const {
        const auto base = reinterpret_cast<TokenSrcMetadata*>(get_src_token_topk_idx_ptr(num_experts_per_rank));
        return base + pool_token_idx;
    }
};

struct Data {
    uint32_t num_bytes;
    bool require_tma_alignment;
    void* base;

    Data() = default;

    DG_HOST_DEVICE
    constexpr explicit Data(
        const uint32_t& num_bytes,
        const bool& require_tma_alignment = true,
        void* base = nullptr) :
        num_bytes(num_bytes), require_tma_alignment(require_tma_alignment), base(base) {
        DG_UNIFIED_ASSERT(num_bytes % 16 == 0 or not require_tma_alignment);
    }

    template <typename dtype_t = uint32_t>
    DG_HOST_DEVICE constexpr dtype_t get_num_bytes() const {
        return static_cast<dtype_t>(num_bytes);
    }

    template <typename dtype_t = void>
    DG_HOST_DEVICE dtype_t* get_base_ptr() const {
        return static_cast<dtype_t*>(base);
    }

    DG_HOST_DEVICE void set_base_ptr(void* ptr) {
        base = ptr;
    }
};

struct Buffer {
    Data data_layout;
    uint32_t num_ranks;
    uint32_t num_max_tokens_per_rank;

    void* base;

    Buffer() = default;

    DG_HOST_DEVICE
    Buffer(const Data& data_layout,
           const uint32_t& num_ranks,
           const uint32_t& num_max_tokens_per_rank,
           void* base = nullptr) :
        data_layout(data_layout),
        num_ranks(num_ranks), num_max_tokens_per_rank(num_max_tokens_per_rank),
        base(base) {}

    DG_HOST_DEVICE
    uint64_t get_num_bytes_per_rank() const {
        return num_max_tokens_per_rank * data_layout.get_num_bytes<uint64_t>();
    }

    DG_HOST_DEVICE
    uint64_t get_num_bytes() const {
        return get_num_bytes_per_rank() * num_ranks;
    }

    template <typename dtype_t = void>
    DG_HOST_DEVICE dtype_t* get_base_ptr() const {
        return static_cast<dtype_t*>(base);
    }

    DG_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    DG_HOST_DEVICE
    Buffer get_rank_buffer(const uint32_t& rank_idx) const {
        return {
            data_layout,
            1, num_max_tokens_per_rank,
            math::advance_ptr(base, get_num_bytes_per_rank() * rank_idx)
        };
    }

    DG_HOST_DEVICE
    Data get_data_buffer(const uint32_t& token_idx, const bool& global = false) const {
        DG_DEVICE_ASSERT(num_ranks == 1 or global);
        return Data(
            data_layout.num_bytes,
            data_layout.require_tma_alignment,
            math::advance_ptr(base, data_layout.get_num_bytes<uint64_t>() * token_idx)
        );
    }
};

struct MegaMoEBuffer {
    Workspace workspace;

    // Input buffers (per-rank)
    Buffer input_token_buffer,
           input_sf_buffer,
           input_topk_idx_buffer,
           input_topk_weights_buffer;

    // Routed expert ring buffers
    // NOTE: shared L1 tokens reuse `input_token_buffer`.
    Buffer shared_l1_token_buffer, shared_l1_sf_buffer,
           shared_l2_token_buffer, shared_l2_sf_buffer;

    // Routed expert ring buffers
    Buffer l1_token_buffer,
           l1_sf_buffer,
           l1_topk_weights_buffer,
           l2_token_buffer,
           l2_sf_buffer,
           combine_token_buffer;

    DG_HOST_DEVICE
    MegaMoEBuffer(void* base,
                  const uint32_t& hidden,
                  const uint32_t& intermediate_hidden,
                  const uint32_t& num_ranks,
                  const uint32_t& num_experts,
                  const uint32_t& num_max_tokens_per_rank,
                  const uint32_t& num_topk,
                  const uint32_t& num_ring_tokens,
                  const uint32_t& num_sf_ring_tokens,
                  const bool& with_sf,
                  const uint32_t& num_shared_experts = 0) {
        // Workspace
        workspace = Workspace(base, num_ranks, num_experts,
                              num_max_tokens_per_rank, num_topk, num_ring_tokens);

        // Shared
        const auto shared_intermediate_hidden = intermediate_hidden * num_shared_experts;
        const auto num_max_shared_sf_tokens = with_sf ? get_num_max_shared_sf_tokens(num_max_tokens_per_rank) : 0u;

        // Layouts
        const uint32_t num_mma_elem_bytes = with_sf ? 1 : 2;
        const auto input_token_layout = layout::Data(hidden * num_mma_elem_bytes);
        const auto bf16_token_layout = layout::Data(hidden * 2);
        const auto intermediate_token_layout = layout::Data(intermediate_hidden * num_mma_elem_bytes);
        const auto shared_intermediate_token_layout = layout::Data(shared_intermediate_hidden * num_mma_elem_bytes);
        const auto input_sf_layout = layout::Data(with_sf ? hidden / 32 : 0, false);
        const auto intermediate_sf_layout = layout::Data(with_sf ? intermediate_hidden / 32 : 0, false);
        const auto shared_intermediate_sf_layout = layout::Data(with_sf ? shared_intermediate_hidden / 32 : 0, false);
        const auto input_topk_idx_layout = layout::Data(num_topk * sizeof(int64_t), false);
        const auto input_topk_weights_layout = layout::Data(num_topk * sizeof(float), false);
        const auto l1_topk_weights_layout = layout::Data(sizeof(float), false);

        // Input buffers
        input_token_buffer = Buffer(
            input_token_layout, 1, num_max_tokens_per_rank,
            workspace.get_end_ptr());
        input_sf_buffer = Buffer(
            input_sf_layout, 1, num_max_tokens_per_rank,
            input_token_buffer.get_end_ptr());
        input_topk_idx_buffer = Buffer(
            input_topk_idx_layout, 1, num_max_tokens_per_rank,
            with_sf ? input_sf_buffer.get_end_ptr() : input_token_buffer.get_end_ptr());
        input_topk_weights_buffer = Buffer(
            input_topk_weights_layout, 1, num_max_tokens_per_rank,
            input_topk_idx_buffer.get_end_ptr());

        // Shared expert buffers
        shared_l1_token_buffer = input_token_buffer;
        shared_l1_sf_buffer = Buffer(
            input_sf_layout, 1, num_shared_experts > 0 ? num_max_shared_sf_tokens : 0,
            input_topk_weights_buffer.get_end_ptr());
        shared_l2_token_buffer = Buffer(
            shared_intermediate_token_layout, 1, num_shared_experts > 0 ? num_max_tokens_per_rank : 0,
            with_sf ? shared_l1_sf_buffer.get_end_ptr() : input_topk_weights_buffer.get_end_ptr());
        shared_l2_sf_buffer = Buffer(
            shared_intermediate_sf_layout, 1, num_shared_experts > 0 ? num_max_shared_sf_tokens : 0,
            shared_l2_token_buffer.get_end_ptr());

        // Routed expert ring buffers
        l1_token_buffer = Buffer(
            input_token_layout, 1, num_ring_tokens,
            num_shared_experts > 0 ?
                (with_sf ? shared_l2_sf_buffer.get_end_ptr() : shared_l2_token_buffer.get_end_ptr()) :
                input_topk_weights_buffer.get_end_ptr()
        );
        l1_sf_buffer = Buffer(
            input_sf_layout, 1, num_sf_ring_tokens,
            l1_token_buffer.get_end_ptr());
        l1_topk_weights_buffer = Buffer(
            l1_topk_weights_layout, 1, num_ring_tokens,
            with_sf ? l1_sf_buffer.get_end_ptr() : l1_token_buffer.get_end_ptr());

        l2_token_buffer = Buffer(
            intermediate_token_layout, 1, num_ring_tokens,
            l1_topk_weights_buffer.get_end_ptr());
        l2_sf_buffer = Buffer(
            intermediate_sf_layout, 1, num_sf_ring_tokens,
            l2_token_buffer.get_end_ptr());

        combine_token_buffer = Buffer(
            bf16_token_layout, num_topk + (num_shared_experts > 0 ? 1u : 0u), num_max_tokens_per_rank,
            with_sf ? l2_sf_buffer.get_end_ptr() : l2_token_buffer.get_end_ptr());
    }

    DG_HOST_DEVICE
    int64_t get_num_bytes() const {
        return static_cast<uint8_t*>(combine_token_buffer.get_end_ptr())
               - reinterpret_cast<uint8_t*>(workspace.signals);
    }
};

} // namespace deep_gemm::layout

namespace deep_gemm::comm {

// 300s timeout, at 2 GHz
constexpr int64_t kNumTimeoutCycles = 300ll * 2000000000ll;

// Spin until `pred()` holds; on timeout, `print_timeout()` runs before the assertion
template <typename pred_t, typename print_timeout_t>
DG_DEVICE void wait_until(const pred_t& pred, const print_timeout_t& print_timeout) {
    const auto start_clock = clock64();
    while (not pred()) {
        if (clock64() - start_clock >= kNumTimeoutCycles) {
            print_timeout();
            DG_DEVICE_ASSERT(false and "Timeout");
        }
    }
}

DG_DEVICE void cluster_sync_with_relaxed_arrive() {
    // Perform cluster_sync with `barrier.cluster.arrive.relaxed`
    // This is slightly faster than `dg::cluster_sync` but has weaker memory ordering guarantee
    dg::cluster_arrive_relaxed();
    dg::cluster_wait();
}

template <uint32_t kNumSMs, uint32_t kGridSyncIndex = 0, typename sync_scope_t>
DG_DEVICE void grid_sync(const layout::Workspace& workspace,
                              const uint32_t& sm_idx, const uint32_t& thread_idx,
                              const sync_scope_t& sync_scope) {
    // NOTES: the implementation idea is from `cooperative_groups::this_grid().sync()`
    static constexpr uint32_t kFinishSumTag = 0x80000000u;
    sync_scope();
    if (thread_idx == 0) {
        const auto count_ptr = workspace.get_grid_sync_count_ptr<kGridSyncIndex>();
        const auto old_value = ptx::atomic_add_rel(
            count_ptr, sm_idx == 0 ? (kFinishSumTag - (kNumSMs - 1)) : 1);
        uint32_t new_value;
        wait_until([&]() { return ((new_value = ptx::ld_acq(count_ptr)) ^ old_value) & kFinishSumTag; }, [&]() {
            printf("DeepGEMM grid sync timeout: sm=%u, thread=%u, grid_sync_idx=%u, old=%u, current=%u, expected_tag=%u\n",
                   sm_idx, thread_idx, kGridSyncIndex, old_value, new_value, old_value ^ kFinishSumTag);
        });
    }
    sync_scope();
}

template <uint32_t kNumRanks, uint32_t kNumSMs, uint32_t kNumThreads, uint32_t kGridSyncIndex, uint32_t kTag, typename sync_scope_t>
DG_DEVICE void nvlink_barrier(const layout::Workspace& workspace,
                                   const layout::SymBuffer<kNumRanks>& sym_buffer,
                                   const uint32_t& sm_idx, const uint32_t& thread_idx,
                                   const sync_scope_t& sync_scope,
                                   const bool& sync_prologue = true,
                                   const bool& sync_epilogue = true) {
    DG_STATIC_ASSERT(kNumRanks <= kNumThreads, "Insufficient threads");

    // Grid sync before NVLink signaling
    if (sync_prologue)
        grid_sync<kNumSMs, kGridSyncIndex>(workspace, sm_idx, thread_idx, sync_scope);

    // NVLink cross-rank barrier, only SM 0 participates
    if (sm_idx == 0) {
        auto* counter_ptr = workspace.get_nvl_barrier_counter_ptr();
        const auto status = (*counter_ptr) & 3;
        const auto signal_phase = status & 1, signal_sign = status >> 1;
        auto* signal_ptr = workspace.get_nvl_barrier_signal_ptr(signal_phase);

        // Send signals to remote ranks
        if (thread_idx < kNumRanks)
            ptx::red_add_rel_sys(sym_buffer.map(signal_ptr, thread_idx), signal_sign ? -1 : 1);
        sync_scope();

        // Update status and wait arrival
        if (thread_idx == 0) {
            ptx::red_add(counter_ptr, 1);
            const int target = signal_sign ? 0 : static_cast<int>(kNumRanks);
            wait_until([&]() { return ptx::ld_acq_sys(signal_ptr) == target; }, [&]() {
                printf("DeepGEMM NVLink barrier timeout: rank=%d, counter=%d, signal=%d, target=%d, phase=%d, sign=%d, tag=%d\n",
                       sym_buffer.rank_idx, *counter_ptr, ptx::ld_acq_sys(signal_ptr), target, signal_phase, signal_sign, kTag);
            });
        }
    }

    // Grid sync after NVLink completion
    if (sync_epilogue)
        grid_sync<kNumSMs, kGridSyncIndex>(workspace, sm_idx, thread_idx, sync_scope);
}

} // namespace deep_gemm::comm

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

// include/deep_gemm/scheduler/mega_moe.cuh

// include/deep_gemm/common/dg_tie.cuh

namespace dg {

struct ignore_t {
    template <typename T>
    constexpr const ignore_t& operator=(T&&) const noexcept {
        return *this;
    }
};

inline constexpr ignore_t ignore{};

} // namespace dg

#define DG_TIE_CONCAT_IMPL(A, B) A##B
#define DG_TIE_CONCAT(A, B) DG_TIE_CONCAT_IMPL(A, B)

#define DG_TIE_GET_NTH_ARG(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, N, ...) N
#define DG_TIE_COUNT_ARGS(...) \
    DG_TIE_GET_NTH_ARG(__VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)

#define DG_TIE_OP_DECL(I, TUPLE, VAR) auto VAR = ::dg::get<I>(TUPLE)
#define DG_TIE_OP_ASSIGN(I, TUPLE, VAR) VAR = ::dg::get<I>(TUPLE)

#define DG_TIE_APPLY_OP_1(OP, T, V1) OP(0, T, V1);
#define DG_TIE_APPLY_OP_2(OP, T, V1, V2) OP(0, T, V1); OP(1, T, V2);
#define DG_TIE_APPLY_OP_3(OP, T, V1, V2, V3) OP(0, T, V1); OP(1, T, V2); OP(2, T, V3);
#define DG_TIE_APPLY_OP_4(OP, T, V1, V2, V3, V4) OP(0, T, V1); OP(1, T, V2); OP(2, T, V3); OP(3, T, V4);
#define DG_TIE_APPLY_OP_5(OP, T, V1, V2, V3, V4, V5) OP(0, T, V1); OP(1, T, V2); OP(2, T, V3); OP(3, T, V4); OP(4, T, V5);

#define DG_TIE_DECL(TUPLE_EXPR, ...) \
    auto&& DG_TIE_CONCAT(dg_tie__temp_tuple_, __LINE__) = (TUPLE_EXPR); \
    DG_TIE_CONCAT(DG_TIE_APPLY_OP_, DG_TIE_COUNT_ARGS(__VA_ARGS__)) ( \
        DG_TIE_OP_DECL, \
        DG_TIE_CONCAT(dg_tie__temp_tuple_, __LINE__), \
        __VA_ARGS__ \
    )

#define DG_TIE(TUPLE_EXPR, ...) \
    do { \
        auto&& DG_TIE_CONCAT(dg_tie__temp_tuple_, __LINE__) = (TUPLE_EXPR); \
        DG_TIE_CONCAT(DG_TIE_APPLY_OP_, DG_TIE_COUNT_ARGS(__VA_ARGS__)) ( \
            DG_TIE_OP_ASSIGN, \
            DG_TIE_CONCAT(dg_tie__temp_tuple_, __LINE__), \
            __VA_ARGS__ \
        ); \
    } while (0)

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

// TODO: add MegaMoE namespace
namespace deep_gemm::sched {

// Get minimal L1 warmup waves to ensure no L1 -> L2 deadlock
// constexpr func, no runtime overhead in kernel
DG_HOST_DEVICE constexpr
int get_num_l1_warmup_waves(
    const int& num_total_m_blocks,
    const int& num_clusters,
    const int& num_l1_n_clusters,
    const int& num_l2_n_clusters) {
    // The first L2 wave may touch multiple M blocks; all their L1 N tasks must be issued first.
    const int num_first_l2_wave_m_blocks = math::constexpr_ceil_div(num_clusters, num_l2_n_clusters);
    const int num_l1_warmup_clusters_for_first_l2_wave = math::constexpr_ceil_div(
        num_first_l2_wave_m_blocks * num_l1_n_clusters, num_clusters);

    const int num_interleave_cluster_diff_per_m_block =
        num_l1_n_clusters > num_l2_n_clusters ? num_l1_n_clusters - num_l2_n_clusters : 0;
    const int num_warmup_waves_for_interleave_schedule = math::constexpr_ceil_div(
        num_l1_n_clusters + (num_total_m_blocks - 1) * num_interleave_cluster_diff_per_m_block,
        num_clusters) + 1;

    // TODO: may delay combine NVLink
    return dg::max(num_l1_warmup_clusters_for_first_l2_wave, num_warmup_waves_for_interleave_schedule);
}

// Host-side ring capacity helper for the MegaMoE schedule.
DG_HOST_DEVICE constexpr int get_num_max_live_pool_blocks(
    const int& num_total_m_blocks,
    const int& num_sms,
    const int& hidden,
    const int& intermediate_hidden) {
    constexpr int kMegaMoEBlockN = 128;
    constexpr int kNumCTAsPerCluster = 2;

    DG_UNIFIED_ASSERT((intermediate_hidden * 2) % (kNumCTAsPerCluster * kMegaMoEBlockN) == 0);
    DG_UNIFIED_ASSERT(hidden % (kNumCTAsPerCluster * kMegaMoEBlockN) == 0);
    const int num_clusters = num_sms / kNumCTAsPerCluster;
    const int num_l1_n_clusters = intermediate_hidden * 2 / (kNumCTAsPerCluster * kMegaMoEBlockN);
    const int num_l2_n_clusters = hidden / (kNumCTAsPerCluster * kMegaMoEBlockN);
    const int num_l1_clusters = num_total_m_blocks * num_l1_n_clusters;
    const int num_l1_waves = math::constexpr_ceil_div(num_l1_clusters, num_clusters);
    const int num_min_l1_warmup_waves = get_num_l1_warmup_waves(
        num_total_m_blocks, num_clusters, num_l1_n_clusters, num_l2_n_clusters);
    const int num_l1_warmup_waves = dg::min(num_min_l1_warmup_waves, num_l1_waves);
    const int num_l1_warmup_clusters = dg::min(num_l1_warmup_waves * num_clusters, num_l1_clusters);
    const int num_live_blocks_after_warmup = math::constexpr_ceil_div(num_l1_warmup_clusters, num_l1_n_clusters);

    // Conservative closed-form bound for the L2/L1 alternating tail. If L1 advances faster in
    // M-block space (B1 < B2), live blocks can grow by about total_m_blocks * (B2 - B1) / B2.
    // Add one global-wave margin to cover partial waves / pipeline lag.
    // This is slightly larger than the tight bound, but cheaper and simpler.
    // TODO: refactor here
    const int frontier_growth = num_l2_n_clusters > num_l1_n_clusters ?
        math::constexpr_ceil_div(
            num_total_m_blocks * (num_l2_n_clusters - num_l1_n_clusters), num_l2_n_clusters) : 0;
    const int wave_margin = math::constexpr_ceil_div(
        num_clusters, dg::min(num_l1_n_clusters, num_l2_n_clusters));
    return dg::min(num_total_m_blocks, num_live_blocks_after_warmup + frontier_growth + wave_margin);
}

// Computation phase for the current block
enum class BlockPhase : uint32_t {
    None = 0,
    Linear1 = 1,
    Linear2 = 2,
    SharedLinear1 = 3,
    SharedLinear2 = 4
};

template <bool kHasSharedExperts>
struct alignas(16) TaskInfo {
    BlockPhase block_phase;
    uint32_t local_expert_idx;
    uint32_t m_block_idx;
    uint32_t n_cluster_idx;
    uint32_t pool_block_idx;
    uint32_t valid_m;
    uint32_t shape_n;
    uint32_t shape_k;

    DG_HOST_DEVICE
    TaskInfo(): TaskInfo(BlockPhase::None, 0, 0, 0, 0, 0, 0, 0) {}

    DG_HOST_DEVICE
    TaskInfo(const BlockPhase& block_phase,
             const uint32_t& local_expert_idx,
             const uint32_t& m_block_idx,
             const uint32_t& pair_n_block_idx,
             const uint32_t& pool_block_idx,
             const uint32_t& valid_m,
             const uint32_t& shape_n,
             const uint32_t& shape_k):
        block_phase(block_phase),
        local_expert_idx(local_expert_idx),
        m_block_idx(m_block_idx),
        n_cluster_idx(pair_n_block_idx),
        pool_block_idx(pool_block_idx),
        valid_m(valid_m), shape_n(shape_n), shape_k(shape_k) {}

    DG_DEVICE
    uint32_t is_valid() const {
        return (block_phase != BlockPhase::None);
    }

    DG_DEVICE
    uint32_t get_umma_aligned_valid_m() const {
        return math::align(valid_m, 16u);
    }

    DG_DEVICE uint32_t is_shared() const {
        return kHasSharedExperts ? (block_phase > BlockPhase::Linear2) : false;
    }
};

DG_STATIC_ASSERT(sizeof(sched::TaskInfo<true>) == sizeof(sched::TaskInfo<false>), "Invalid layout");

#if defined(__CUDACC__) or defined(__CLION_IDE__)

// Each finished L1 N block toggles its bit in `l2_full_mask`, so L2 K blocks can start as soon as they are fed
// NOTES: the mask parity per ring generation relies on the ring capacity, see `get_num_max_live_pool_blocks`
template <uint32_t L1_SHAPE_N, uint32_t BLOCK_N, uint32_t BLOCK_K>
struct L2KBlockDependency {
    static constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
    static constexpr uint32_t kNumL1BlockNsPerL2KBlock = BLOCK_K / (BLOCK_N / 2);
    static constexpr uint64_t kFullMask = kNumL1BlockNs == 64 ? ~0ull : (1ull << kNumL1BlockNs) - 1;
    DG_STATIC_ASSERT(kNumL1BlockNs <= 64, "Too many L1 N blocks for the L2 readiness mask");
    DG_STATIC_ASSERT(BLOCK_K % (BLOCK_N / 2) == 0, "Invalid L1/L2 shape relationship");

    const uint64_t* mask_ptr;
    uint64_t expected_mask, pending_mask = kFullMask;

    DG_DEVICE L2KBlockDependency(const uint64_t* mask_ptr, const uint32_t& generation_idx):
        mask_ptr(mask_ptr), expected_mask(generation_idx & 1 ? 0ull : kFullMask) {}

    DG_DEVICE static void arrive(const uint64_t* mask_ptr, const uint32_t& n_block_idx) {
        ptx::red_xor_rel(mask_ptr, 1ull << n_block_idx);
    }

    // Re-read the mask only when the K block is not fed yet
    DG_DEVICE void wait(const uint32_t& k_block_idx) {
        const auto k_block_mask = ((1ull << kNumL1BlockNsPerL2KBlock) - 1) << (k_block_idx * kNumL1BlockNsPerL2KBlock);
        while (pending_mask & k_block_mask)
            pending_mask = ptx::ld_acq_gpu(mask_ptr) ^ expected_mask;
    }
};

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t L1_SHAPE_N, uint32_t L1_SHAPE_K,
          uint32_t L2_SHAPE_N, uint32_t L2_SHAPE_K,
          uint32_t kNumExpertsPerRank,
          uint32_t kNumSMs, uint32_t kNumRanks,
          uint32_t kNumRingBlocks,
          uint32_t kNumSharedExperts = 0,
          uint32_t kNumExpertsPerLane = math::constexpr_ceil_div(kNumExpertsPerRank, 32u),
          uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N,
          uint32_t kNumL2BlockNs = L2_SHAPE_N / BLOCK_N,
          uint32_t kNumL1Clusters = kNumL1BlockNs / 2,
          uint32_t kNumL2Clusters = kNumL2BlockNs / 2>
struct MegaMoEScheduler {
    static constexpr bool kHasShared = kNumSharedExperts > 0;
    static constexpr uint32_t SHARED_L1_SHAPE_N = L1_SHAPE_N * kNumSharedExperts;
    static constexpr uint32_t SHARED_L1_SHAPE_K = L1_SHAPE_K;
    static constexpr uint32_t SHARED_L2_SHAPE_N = L2_SHAPE_N;
    static constexpr uint32_t SHARED_L2_SHAPE_K = L2_SHAPE_K * kNumSharedExperts;
    using task_info_t = TaskInfo<kHasShared>;

    DG_STATIC_ASSERT(L1_SHAPE_N % (BLOCK_N * 2) == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_N % (BLOCK_N * 2) == 0, "Invalid shape");
    DG_STATIC_ASSERT(L1_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    DG_STATIC_ASSERT(SHARED_L1_SHAPE_N % (BLOCK_N * 2) == 0, "Invalid shared shape");
    DG_STATIC_ASSERT(SHARED_L2_SHAPE_N % (BLOCK_N * 2) == 0, "Invalid shared shape");
    DG_STATIC_ASSERT(SHARED_L1_SHAPE_K % BLOCK_K == 0, "Invalid shared shape");
    DG_STATIC_ASSERT(SHARED_L2_SHAPE_K % BLOCK_K == 0, "Invalid shared shape");

    // NOTES: N block counts must be even so that 2 adjacent CTAs in a cluster
    // always land on the same m_block_idx with n_block_idx differing by 1
    DG_STATIC_ASSERT(kNumSMs % 2 == 0, "Number of SMs must be even for 2-CTA cluster");
    DG_STATIC_ASSERT(kNumRingBlocks > 0, "Invalid ring buffer config");

    // Workspace
    const layout::Workspace& workspace;

    // Scheduler configs
    static constexpr uint32_t kNumScheduleStages = 2;
    using Barrier = dg::arch::ClusterTransactionBarrier;
    uint32_t sched_stage_idx = 0;
    uint32_t sched_phase = 0;
    Barrier* task_info_full_barriers = nullptr;
    Barrier* task_info_empty_barriers = nullptr;
    task_info_t* task_infos = nullptr;

    // Pre-cached per-expert token counts.
    // Layout: `stored_num_tokens_per_expert[i]` holds expert (i * 32 + lane_idx)'s count.
    uint32_t stored_num_tokens_per_expert[kNumExpertsPerLane] = {};
    uint32_t num_total_m_blocks = 0;

    // Per-scheduler warmup waves; all CTA-pair schedulers together form one global wave.
    static constexpr uint32_t kNumSchedL1WavesDone = 0xffffffffu;
    uint32_t num_sched_l1_waves = 0;

    DG_DEVICE explicit MegaMoEScheduler(const layout::Workspace& workspace):
        workspace(workspace) {}

    DG_DEVICE MegaMoEScheduler(const layout::Workspace& workspace,
                                    Barrier* task_info_full_barriers,
                                    Barrier* task_info_empty_barriers,
                                    task_info_t* task_infos):
        workspace(workspace),
        task_info_full_barriers(task_info_full_barriers),
        task_info_empty_barriers(task_info_empty_barriers),
        task_infos(task_infos) {}

    DG_DEVICE void advance_sched_pipeline() {
        DG_STATIC_ASSERT(kNumScheduleStages == 2, "Invalid stages");
        sched_stage_idx ^= 1;
        sched_phase ^= sched_stage_idx == 0;
    }

    DG_DEVICE bool get_next_task(task_info_t& task_info) {
        task_info_full_barriers[sched_stage_idx].wait(sched_phase);
        task_info = task_infos[sched_stage_idx];
        advance_sched_pipeline();
        return task_info.is_valid();
    }

    DG_DEVICE void release_task_info() const {
        // Complete metadata reads before the scheduler can overwrite this slot.
        ptx::fence_acq_rel_cta();
        task_info_empty_barriers[sched_stage_idx ^ 1].arrive(0u);
    }

    DG_DEVICE uint32_t get_num_tokens(const uint32_t& expert_idx) const {
        uint32_t valid_value = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            valid_value = (expert_idx == i * 32 + ptx::get_lane_idx()) ?
                stored_num_tokens_per_expert[i] : valid_value;
        }
        return ptx::exchange(valid_value, expert_idx % 32);
    }

    // Get pool block offset for a given expert index from a per-lane token count array.
    DG_DEVICE uint32_t get_pool_block_offset(const uint32_t& expert_idx) const {
        uint32_t num_blocks = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            if (i * 32 + ptx::get_lane_idx() < expert_idx)
                num_blocks += math::ceil_div(stored_num_tokens_per_expert[i], BLOCK_M);
        }
        return __reduce_add_sync(0xffffffff, num_blocks);
    }

    DG_DEVICE uint32_t get_num_total_pool_blocks() const {
        return get_pool_block_offset(kNumExpertsPerRank);
    }

    DG_DEVICE void fetch_expert_recv_count() {
        // NOTES: each lane caches experts at indices (i * 32 + lane_idx)
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            const auto expert_idx = i * 32 + ptx::get_lane_idx();
            uint64_t value = 0;
            if (expert_idx < kNumExpertsPerRank) {
                do {
                    value = ptx::ld_volatile(workspace.get_expert_recv_count_sum_ptr(expert_idx));
                } while (static_cast<uint32_t>(value >> 32) != kNumSMs * kNumRanks);
            }
            stored_num_tokens_per_expert[i] = static_cast<uint32_t>(value);
        }
        __syncwarp();

        num_total_m_blocks = get_num_total_pool_blocks();
        const uint32_t num_total_l1_tasks = num_total_m_blocks * kNumL1Clusters;
        const uint32_t num_total_l1_waves = math::ceil_div(num_total_l1_tasks, kNumSMs / 2);
        const uint32_t min_l1_warmup_waves = get_num_l1_warmup_waves(
            num_total_m_blocks, kNumSMs / 2, kNumL1Clusters, kNumL2Clusters);
        num_sched_l1_waves = dg::min(min_l1_warmup_waves, num_total_l1_waves);
    }

    DG_DEVICE task_info_t create_task(const BlockPhase& block_phase,
                                        const uint32_t& task_idx,
                                        const uint32_t& num_clusters,
                                        const uint32_t& shape_n,
                                        const uint32_t& shape_k) const {
        const uint32_t lane_idx = ptx::get_lane_idx();
        const uint32_t m_block_idx = task_idx / num_clusters;
        const uint32_t n_cluster_idx = task_idx % num_clusters;

        task_info_t result(block_phase, 0, 0, n_cluster_idx, m_block_idx, 0, shape_n, shape_k);
        uint32_t block_offset = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            // Reduce whether task fall in the expert
            const uint32_t expert_idx = i * 32 + lane_idx;
            const uint32_t num_tokens = stored_num_tokens_per_expert[i];
            const uint32_t num_m_blocks = math::ceil_div(num_tokens, BLOCK_M);
            const uint32_t inclusive_num_m_blocks = math::warp_inclusive_sum(num_m_blocks, lane_idx);
            const uint32_t lane_pool_block_offset = block_offset + inclusive_num_m_blocks - num_m_blocks;
            const bool is_owner = expert_idx < kNumExpertsPerRank and
                m_block_idx >= lane_pool_block_offset and m_block_idx < lane_pool_block_offset + num_m_blocks;
            const uint32_t owner_mask = __ballot_sync(0xffffffff, is_owner);

            // Exchange the expert info
            if (owner_mask) {
                const uint32_t owner_lane_idx = static_cast<uint32_t>(__ffs(owner_mask) - 1);
                const uint32_t owner_m_block_idx = m_block_idx - lane_pool_block_offset;
                const uint32_t owner_valid_m = dg::min(num_tokens - owner_m_block_idx * BLOCK_M, BLOCK_M);
                result.local_expert_idx = ptx::exchange(expert_idx, owner_lane_idx);
                result.m_block_idx = ptx::exchange(owner_m_block_idx, owner_lane_idx);
                result.valid_m = ptx::exchange(owner_valid_m, owner_lane_idx);
            }
            block_offset += ptx::exchange(inclusive_num_m_blocks, 31);
        }
        return result;
    }

    static DG_DEVICE uint32_t get_next_task_idx(const uint32_t* global_task_count_ptr) {
        uint32_t result = 0;
        if (dg::elect_one_sync())
            result = ptx::atomic_add(global_task_count_ptr, 1u);
        return ptx::exchange(result, 0);
    }

    DG_DEVICE task_info_t get_next_task() {
        while (true) {
            if (num_sched_l1_waves != kNumSchedL1WavesDone and num_sched_l1_waves) {
                // One local L1 task per scheduler; globally this is one CTA-pair wave.
                -- num_sched_l1_waves;

                // No more L1 tasks
                const uint32_t l1_task_idx = get_next_task_idx(workspace.get_l1_task_count_ptr());
                if (l1_task_idx >= num_total_m_blocks * kNumL1Clusters) {
                    num_sched_l1_waves = kNumSchedL1WavesDone;
                    continue;
                }

                // Create task
                return create_task(BlockPhase::Linear1, l1_task_idx, kNumL1Clusters, L1_SHAPE_N, L1_SHAPE_K);
            } else {
                const uint32_t l2_task_idx = get_next_task_idx(workspace.get_l2_task_count_ptr());
                if (l2_task_idx >= num_total_m_blocks * kNumL2Clusters)
                    break;

                // The next task should be L1
                if (num_sched_l1_waves != kNumSchedL1WavesDone)
                    num_sched_l1_waves = 1;

                // Create task
                auto task_info = create_task(BlockPhase::Linear2, l2_task_idx, kNumL2Clusters, L2_SHAPE_N, L2_SHAPE_K);

                // Wait until all required L1 tasks are fetched
                const auto num_required_l1_tasks = (task_info.pool_block_idx + 1) * kNumL1Clusters;
                while (ptx::ld_volatile(workspace.get_l1_task_count_ptr()) < num_required_l1_tasks) {}
                return task_info;
            }
        }
        return task_info_t(BlockPhase::None, 0, 0, 0, 0, 0, 0, 0);
    }

    DG_DEVICE void publish_task(const task_info_t& task_info, const uint32_t& lane_idx) {
        if (lane_idx < 2) {
            task_info_full_barriers[sched_stage_idx].arrive_and_expect_tx(sizeof(task_info_t), lane_idx);
            ptx::st_async_cluster(
                task_infos + sched_stage_idx, task_info,
                lane_idx, task_info_full_barriers[sched_stage_idx]
            );
        }
        __syncwarp();
        advance_sched_pipeline();
    }

    template <BlockPhase kBlockPhase, uint32_t kShapeN, uint32_t kShapeK>
    DG_DEVICE void shared_mainloop(const uint32_t& num_tokens, const uint32_t& lane_idx, const uint32_t* task_count_ptr) {
        constexpr uint32_t kNumNClusters = kShapeN / BLOCK_N / 2;
        const uint32_t num_m_blocks = math::ceil_div(num_tokens, BLOCK_M);
        const uint32_t num_tasks = num_m_blocks * kNumNClusters;
        while (true) {
            task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);

            // Use dynamic scheduling to reduce tailing across shared L1/L2 tile shapes.
            const uint32_t task_idx = get_next_task_idx(task_count_ptr);
            if (task_idx >= num_tasks)
                break;
            const uint32_t m_block_idx = task_idx / kNumNClusters;
            const uint32_t n_cluster_idx = task_idx % kNumNClusters;
            const uint32_t valid_m = dg::min(num_tokens - m_block_idx * BLOCK_M, BLOCK_M);
            publish_task(task_info_t(kBlockPhase, 0, m_block_idx, n_cluster_idx, m_block_idx, valid_m, kShapeN, kShapeK), lane_idx);
        }
    }

    DG_DEVICE void mainloop(const uint32_t& num_tokens) {
        const auto lane_idx = ptx::get_lane_idx();

        // Shared L2 tasks go before the routed tasks if one wave fits the dispatch gap, else at the tail
        constexpr uint32_t kNumSharedL2Clusters = SHARED_L2_SHAPE_N / BLOCK_N / 2;
        const bool is_shared_l2_early = math::ceil_div(num_tokens, BLOCK_M) * kNumSharedL2Clusters <= kNumSMs / 2;
        if constexpr (kHasShared) {
            // Shared expert L1 tasks do not depend on dispatch.
            shared_mainloop<BlockPhase::SharedLinear1, SHARED_L1_SHAPE_N, SHARED_L1_SHAPE_K>(
                num_tokens, lane_idx, workspace.get_shared_l1_task_count_ptr());
            if (is_shared_l2_early)
                shared_mainloop<BlockPhase::SharedLinear2, SHARED_L2_SHAPE_N, SHARED_L2_SHAPE_K>(
                    num_tokens, lane_idx, workspace.get_shared_l2_task_count_ptr());
        }

        // Wait dispatch's results
        fetch_expert_recv_count();

        // Generate routed tasks. Keep the original wait -> claim -> publish ordering:
        // `get_next_task()` advances global task counters and must not run before the
        // schedule slot is released by consumers.
        task_info_t task_info;
        do {
            task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);
            task_info = get_next_task();
            if (task_info.is_valid()) publish_task(task_info, lane_idx);
        } while (task_info.is_valid());

        if constexpr (kHasShared) {
            if (not is_shared_l2_early)
                shared_mainloop<BlockPhase::SharedLinear2, SHARED_L2_SHAPE_N, SHARED_L2_SHAPE_K>(
                    num_tokens, lane_idx, workspace.get_shared_l2_task_count_ptr());
        }

        // Sentinel.
        task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);
        publish_task(task_info_t(BlockPhase::None, 0, 0, 0, 0, 0, 0, 0), lane_idx);
    }
};

#endif

} // namespace deep_gemm::sched

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

// include/deep_gemm/ptx/tma.cuh

namespace deep_gemm::ptx {

// Tensor-map instructions
DG_DEVICE void tensor_map_release_gpu() {
    asm volatile ("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
}

DG_DEVICE void tensor_map_acquire_gpu(const dg::TmaDescriptor* gmem_desc_ptr) {
    auto gmem_int_desc = reinterpret_cast<uint64_t>(gmem_desc_ptr);
    asm volatile ("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"(gmem_int_desc) : "memory");
}

DG_DEVICE void tensor_map_replace_global_addr_in_smem(dg::TmaDescriptor* smem_desc, const void* new_addr) {
    auto smem_int_desc = static_cast<uint32_t>(__cvta_generic_to_shared(smem_desc));
    const auto new_int64_addr = reinterpret_cast<uint64_t>(new_addr);
    asm volatile ("tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%0], %1;" :: "r"(smem_int_desc), "l"(new_int64_addr));
}

DG_DEVICE void tensor_map_replace_global_inner_dim_stride_in_smem(dg::TmaDescriptor* smem_desc, const uint32_t& new_dim, const uint64_t& new_stride) {
    auto smem_int_desc = __cvta_generic_to_shared(smem_desc);
    asm volatile ("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0, %1;" :: "l"(smem_int_desc), "r"(new_dim));
#if ((__CUDACC_VER_MAJOR__ > 12) or ((__CUDACC_VER_MAJOR__ == 12) and (__CUDACC_VER_MINOR__ >= 3)))
    asm volatile("tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 0, %1;" :: "l"(smem_int_desc), "l"(new_stride));
#else
    DG_STATIC_ASSERT(false, "Invalid CUDA version");
#endif
}

/// TMA instructions
DG_DEVICE void mbarrier_arrive(
    dg::arch::ClusterTransactionBarrier* ptr) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0]; \n\t" ::
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))));
}

DG_DEVICE void mbarrier_arrive_count(
    dg::arch::ClusterTransactionBarrier& barrier, const uint32_t count) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;" ::
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&barrier))), "r"(count));
}

DG_DEVICE void mbarrier_arrive_count_pred(
    dg::arch::ClusterTransactionBarrier& barrier, const uint32_t count, const bool pred) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %2, 0;\n\t"
        "@p mbarrier.arrive.shared::cta.b64 _, [%0], %1;\n\t"
        "}" ::
        "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&barrier))), "r"(count), "r"(static_cast<uint32_t>(pred)));
}

DG_DEVICE void mbarrier_arrive_and_set_tx(
    dg::arch::ClusterTransactionBarrier* ptr, const uint32_t& num_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0; \n\t" ::
                 "r"(num_bytes), "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))));
}

DG_DEVICE void mbarrier_wait_and_flip_phase(
    dg::arch::ClusterTransactionBarrier* ptr, uint32_t& phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred       P1; \n\t"
        "LAB_WAIT: \n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; \n\t"
        "@P1 bra DONE; \n\t"
        "bra     LAB_WAIT; \n\t"
        "DONE: \n\t"
        "}" ::
        "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))),
        "r"(phase), "r"(0x989680));
    phase ^= 1;
}

DG_DEVICE void tma_load_1d(
    const void* dst_ptr, const void* src_ptr,
    dg::arch::ClusterTransactionBarrier* mbarrier_ptr,
    const uint32_t& num_bytes,
    const dg::TMA::CacheHintSm100& hint = dg::TMA::CacheHintSm100::EVICT_FIRST) {
    // NOTES: normally, the loaded part will be evicted soon
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n" ::
        "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst_ptr))),
        "l"(src_ptr),
        "r"(num_bytes),
        "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbarrier_ptr))),
        "l"(hint)
        : "memory");
}

DG_DEVICE void tma_store_1d(
    const void* dst_ptr, const void* src_ptr, const uint32_t& num_bytes,
    const dg::TMA::CacheHintSm100& hint = dg::TMA::CacheHintSm100::EVICT_NORMAL) {
    // NOTES: normally, the stored part will be used soon
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0], [%1], %2, %3;\n" ::
                 "l"(dst_ptr),
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src_ptr))),
                 "r"(num_bytes),
                 "l"(hint)
                 : "memory");
}

DG_DEVICE void tma_store_2d(
    const void* desc_ptr, const void* smem_ptr,
    const uint32_t& crd_0, const uint32_t& crd_1,
    const dg::TMA::CacheHintSm100& hint = dg::TMA::CacheHintSm100::EVICT_NORMAL) {
    const auto gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    const auto smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group.L2::cache_hint "
        "[%0, {%2, %3}], [%1], %4;\n" ::
        "l"(gmem_int_desc), "r"(smem_int_ptr),
        "r"(crd_0), "r"(crd_1), "l"(hint)
        : "memory");
}

DG_DEVICE void tma_store_3d(
    const void* desc_ptr, const void* smem_ptr,
    const uint32_t& crd_0, const uint32_t& crd_1, const uint32_t& crd_2,
    const dg::TMA::CacheHintSm100& hint) {
    const auto gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    const auto smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.bulk_group.L2::cache_hint "
        "[%0, {%2, %3, %4}], [%1], %5;\n" ::
        "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd_0), "r"(crd_1), "r"(crd_2), "l"(hint) : "memory");
}

template <int kNumRemainingWaits = 0>
__forceinline__ __device__ void tma_store_wait() {
    // NOTES: this function does not have `.read`
    asm volatile("cp.async.bulk.wait_group %0;" ::"n"(kNumRemainingWaits) : "memory");
}

DG_DEVICE
void tma_gather4(const void* desc_ptr, dg::arch::ClusterTransactionBarrier& mbarrier,
                 void* smem_ptr, const uint32_t& col_idx, const int4& row_idxs, const uint64_t& cache_hint) {
    const auto smem_addr = dg::cast_smem_ptr_to_uint(smem_ptr);
    const auto mbarrier_addr = dg::cast_smem_ptr_to_uint(&mbarrier);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;\n"
        :
        : "r"(smem_addr), "l"(desc_ptr), "r"(col_idx),
          "r"(row_idxs.x), "r"(row_idxs.y), "r"(row_idxs.z), "r"(row_idxs.w),
          "r"(mbarrier_addr), "l"(cache_hint)
        : "memory"
    );
}

} // namespace deep_gemm::ptx

namespace deep_gemm {

// The explicit entries bind all original parameters and include the one
// editable kernels/sm100_bf16_mega_moe.inc body. No duplicate template
// global is retained; dynamic entries use the same bound body.

} // namespace deep_gemm
