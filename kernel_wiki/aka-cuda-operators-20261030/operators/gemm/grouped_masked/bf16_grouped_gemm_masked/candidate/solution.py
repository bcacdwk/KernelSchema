"""Independent CUDA bf16 masked grouped GEMM for L20D (sm_103).

Own kernel: cp.async multi-stage pipeline + ldmatrix + mma.sync m16n8k16 bf16
with fp32 accumulation, per-expert device-side masked block skipping, swizzled
shared memory, staged vectorized epilogue. No GEMM libraries, no CuTe/CUTLASS,
no Triton, no Torch numerical fallback. Mechanisms (tile pipelining, NT
fragment layouts, masked scheduling) informed by the public PTX ISA and the
DeepGEMM reference source at revision e1f418c2a4f20818221f6b0e578b4c2f634d4c3f;
the implementation is independent.

ABI: forward(a, b, d, masked_m, expected_m, compiled_dims='nk') -> None
  a: (E, CAP, K) bf16 contiguous
  b: (E, N, K)   bf16 contiguous
  d: (E, CAP, N) bf16 contiguous (in/out; only d[e, :masked_m[e]] is semantic,
     capacity padding may be written; guard/backing storage untouched)
  masked_m: (E,) int32 on device (never read on host: no sync in forward)
"""
import ctypes
import os
from pathlib import Path
import subprocess
import tempfile

import torch

CUDA_SOURCE = r'''
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace ggm {

#define DEVINL __device__ __forceinline__

DEVINL uint32_t smem_addr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

DEVINL void cp_async16(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
}
DEVINL void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
DEVINL void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

DEVINL void ldm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

DEVINL void mma_bf16(float& c0, float& c1, float& c2, float& c3,
                     uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                     uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int BM, int BN, int BK, int STAGES, int NW_M, int NW_N>
__global__ void __launch_bounds__(256, 1)
ggm_kernel(const __nv_bfloat16* __restrict__ A,
           const __nv_bfloat16* __restrict__ B,
           __nv_bfloat16* __restrict__ D,
           const int* __restrict__ masked_m,
           int CAP, int N, int K, int MB, int NB) {
    static_assert(BK == 64, "BK must be 64 (128B rows, 8x16B swizzle atoms)");
    static_assert(NW_M * NW_N == 8, "256 threads = 8 warps");
    static_assert(BM % (NW_M * 16) == 0 && BN % (NW_N * 8) == 0, "warp tiling");
    static_assert((BN / 8) % 2 == 0, "B ldmatrix pairs");

    constexpr int WM = BM / NW_M;
    constexpr int WN = BN / NW_N;
    constexpr int MI = WM / 16;
    constexpr int NJ = WN / 8;
    constexpr int A_STAGE = BM * BK * 2;
    constexpr int B_STAGE = BN * BK * 2;

    const int bid = blockIdx.x;
    const int e = bid / (MB * NB);
    const int rem = bid - e * (MB * NB);
    const int nb = rem / MB;
    const int mb = rem - nb * MB;

    const int mm = __ldg(masked_m + e);
    const int m0 = mb * BM;
    if (m0 >= mm) return;
    const int n0 = nb * BN;

    extern __shared__ __align__(128) char smem[];
    char* smemA = smem;
    char* smemB = smem + STAGES * A_STAGE;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warp_m = warp / NW_N;
    const int warp_n = warp % NW_N;

    const size_t aBase = (size_t)e * CAP * K + (size_t)m0 * K;
    const size_t bBase = (size_t)e * N * K + (size_t)n0 * K;

    const int num_k = K / BK;

    auto load_stage = [&](int stage, int kb) {
        char* sa = smemA + stage * A_STAGE;
        char* sb = smemB + stage * B_STAGE;
        const __nv_bfloat16* ga = A + aBase + (size_t)kb * BK;
        const __nv_bfloat16* gb = B + bBase + (size_t)kb * BK;
        #pragma unroll
        for (int i = 0; i < BM / 32; ++i) {
            int c = tid + i * 256;
            int r = c >> 3, ch = c & 7;
            cp_async16(smem_addr(sa + r * (BK * 2) + ((ch ^ (r & 7)) * 16)),
                       ga + (size_t)r * K + ch * 8);
        }
        #pragma unroll
        for (int i = 0; i < BN / 32; ++i) {
            int c = tid + i * 256;
            int r = c >> 3, ch = c & 7;
            cp_async16(smem_addr(sb + r * (BK * 2) + ((ch ^ (r & 7)) * 16)),
                       gb + (size_t)r * K + ch * 8);
        }
    };

    float acc[MI][NJ][4];
    #pragma unroll
    for (int i = 0; i < MI; ++i)
        #pragma unroll
        for (int j = 0; j < NJ; ++j) {
            acc[i][j][0] = 0.f; acc[i][j][1] = 0.f;
            acc[i][j][2] = 0.f; acc[i][j][3] = 0.f;
        }

    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < num_k) load_stage(s, s);
        cp_commit();
    }

    int sc = 0, sl = STAGES - 1;
    #pragma unroll 1
    for (int kb = 0; kb < num_k; ++kb) {
        cp_wait<STAGES - 2>();
        __syncthreads();
        int kn = kb + STAGES - 1;
        if (kn < num_k) load_stage(sl, kn);
        cp_commit();

        const char* sa = smemA + sc * A_STAGE;
        const char* sb = smemB + sc * B_STAGE;
        uint32_t afr[MI][4];
        uint32_t bfr[NJ][2];
        #pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
            #pragma unroll
            for (int i = 0; i < MI; ++i) {
                int row = warp_m * WM + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                int chunk = kk * 2 + (lane >> 4);
                ldm_x4(afr[i][0], afr[i][1], afr[i][2], afr[i][3],
                       smem_addr(sa + row * (BK * 2) + ((chunk ^ (row & 7)) * 16)));
            }
            #pragma unroll
            for (int j = 0; j < NJ; j += 2) {
                int rown = warp_n * WN + j * 8 + ((lane >> 4) & 1) * 8 + (lane & 7);
                int chunk = kk * 2 + ((lane >> 3) & 1);
                ldm_x4(bfr[j][0], bfr[j][1], bfr[j + 1][0], bfr[j + 1][1],
                       smem_addr(sb + rown * (BK * 2) + ((chunk ^ (rown & 7)) * 16)));
            }
            #pragma unroll
            for (int i = 0; i < MI; ++i)
                #pragma unroll
                for (int j = 0; j < NJ; ++j)
                    mma_bf16(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3],
                             afr[i][0], afr[i][1], afr[i][2], afr[i][3],
                             bfr[j][0], bfr[j][1]);
        }
        sc = (sc + 1 == STAGES) ? 0 : sc + 1;
        sl = (sl + 1 == STAGES) ? 0 : sl + 1;
    }


    // Epilogue: stage C through shared memory, then vectorized global stores.
    __syncthreads();
    constexpr int EPI_ROW = BN * 2;          // bytes per staged row
    constexpr int EPI_CPR = BN / 8;          // 16B chunks per row
    char* ep = smem;
    const int r_lane = lane >> 2;            // 0..7
    const int c_lane = (lane & 3) * 2;       // 0,2,4,6
    #pragma unroll
    for (int i = 0; i < MI; ++i) {
        int rowA = warp_m * WM + i * 16 + r_lane;
        #pragma unroll
        for (int j = 0; j < NJ; ++j) {
            int col = warp_n * WN + j * 8 + c_lane;
            int ch = col >> 3, cin = (col & 7) * 2;
            int rowB = rowA + 8;
            __nv_bfloat162 v0 = __floats2bfloat162_rn(acc[i][j][0], acc[i][j][1]);
            __nv_bfloat162 v1 = __floats2bfloat162_rn(acc[i][j][2], acc[i][j][3]);
            *reinterpret_cast<uint32_t*>(ep + rowA * EPI_ROW + ((ch ^ (rowA & (EPI_CPR - 1))) * 16) + cin)
                = *reinterpret_cast<uint32_t*>(&v0);
            *reinterpret_cast<uint32_t*>(ep + rowB * EPI_ROW + ((ch ^ (rowB & (EPI_CPR - 1))) * 16) + cin)
                = *reinterpret_cast<uint32_t*>(&v1);
        }
    }
    __syncthreads();
    __nv_bfloat16* dg = D + (size_t)e * CAP * N + (size_t)m0 * N + n0;
    #pragma unroll
    for (int i = 0; i < BM * EPI_CPR / 256; ++i) {
        int c = tid + i * 256;
        int r = c / EPI_CPR, ch = c % EPI_CPR;
        float4 v = *reinterpret_cast<const float4*>(ep + r * EPI_ROW + ((ch ^ (r & (EPI_CPR - 1))) * 16));
        *reinterpret_cast<float4*>(dg + (size_t)r * N + ch * 8) = v;
    }
}

template <int BM, int BN, int STAGES>
void launch_cfg(const void* A, const void* B, void* D, const int* mm,
                int E, int CAP, int N, int K, cudaStream_t st) {
    constexpr int BK = 64;
    constexpr int SMEM = STAGES * (BM + BN) * BK * 2;
    static bool inited = false;
    if (!inited) {
        cudaFuncSetAttribute((const void*)ggm_kernel<BM, BN, BK, STAGES, 2, 4>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        inited = true;
    }
    int MB = CAP / BM, NB = N / BN;
    ggm_kernel<BM, BN, BK, STAGES, 2, 4>
        <<<E * MB * NB, 256, SMEM, st>>>((const __nv_bfloat16*)A, (const __nv_bfloat16*)B,
                                         (__nv_bfloat16*)D, mm, CAP, N, K, MB, NB);
}

}  // namespace ggm

extern "C" int ggm_launch(const void* A, const void* B, void* D, const int* masked_m,
                          int E, int CAP, int N, int K,
                          int expected_m, int bm_ovr, int bn_ovr, cudaStream_t stream) {
    int bm = bm_ovr > 0 ? bm_ovr : (expected_m <= 16 ? 32 : (expected_m <= 96 ? 64 : 128));
    int bn = bn_ovr > 0 ? bn_ovr : (bm == 128 ? 256 : 128);
    if (CAP % bm != 0 || N % bn != 0 || K % 64 != 0) return -2;
    if (bm == 32 && bn == 128)
        ggm::launch_cfg<32, 128, 8>(A, B, D, masked_m, E, CAP, N, K, stream);
    else if (bm == 64 && bn == 128)
        ggm::launch_cfg<64, 128, 6>(A, B, D, masked_m, E, CAP, N, K, stream);
    else if (bm == 128 && bn == 128)
        ggm::launch_cfg<128, 128, 4>(A, B, D, masked_m, E, CAP, N, K, stream);
    else if (bm == 128 && bn == 256)
        ggm::launch_cfg<128, 256, 4>(A, B, D, masked_m, E, CAP, N, K, stream);
    else if (bm == 64 && bn == 256)
        ggm::launch_cfg<64, 256, 4>(A, B, D, masked_m, E, CAP, N, K, stream);
    else
        return -3;
    return (int)cudaGetLastError();
}
'''



CUDA_V1 = r'''
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <map>

// v1: persistent tcgen05 (UMMA) masked grouped GEMM for SM100-family (sm_103a).
// Own implementation; mechanisms (TMA+mbarrier pipeline, UMMA smem/instr
// descriptors, tmem double-buffered epilogue, masked persistent scheduling)
// informed by the public PTX ISA and DeepGEMM e1f418c sources. No CUTLASS/CuTe.

namespace ggm2 {

#define DEVINL __device__ __forceinline__

DEVINL uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ---- mbarrier ----
DEVINL void mbar_init(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(smem_u32(bar)), "r"(count));
}
DEVINL void mbar_init_fence() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
DEVINL void mbar_arrive(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(smem_u32(bar)));
}
DEVINL void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                 :: "r"(smem_u32(bar)), "r"(bytes));
}
DEVINL void mbar_wait(uint64_t* bar, uint32_t parity) {
    asm volatile(
        "{\n\t.reg .pred P;\n\t"
        "WAIT_%=:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1, %2;\n\t"
        "@P bra DONE_%=;\n\t"
        "bra WAIT_%=;\n\t"
        "DONE_%=:\n\t}\n"
        :: "r"(smem_u32(bar)), "r"(parity), "r"(0x989680));
}

// ---- TMA ----
DEVINL void tma_load_2d(const CUtensorMap* desc, uint64_t* bar, void* smem_dst, int c0, int c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(smem_u32(smem_dst)), "l"(reinterpret_cast<uint64_t>(desc)),
           "r"(c0), "r"(c1), "r"(smem_u32(bar))
        : "memory");
}
DEVINL void tma_load_2d_h(const CUtensorMap* desc, uint64_t* bar, void* smem_dst, int c0, int c1, uint64_t hint) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
        "[%0], [%1, {%2, %3}], [%4], %5;\n"
        :: "r"(smem_u32(smem_dst)), "l"(reinterpret_cast<uint64_t>(desc)),
           "r"(c0), "r"(c1), "r"(smem_u32(bar)), "l"(hint)
        : "memory");
}
DEVINL void tma_store_2d(const CUtensorMap* desc, const void* smem_src, int c0, int c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n"
        :: "l"(reinterpret_cast<uint64_t>(desc)), "r"(c0), "r"(c1), "r"(smem_u32(smem_src))
        : "memory");
}
DEVINL void tma_store_2d_h(const CUtensorMap* desc, const void* smem_src, int c0, int c1, uint64_t hint) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group.L2::cache_hint [%0, {%1, %2}], [%3], %4;\n"
        :: "l"(reinterpret_cast<uint64_t>(desc)), "r"(c0), "r"(c1), "r"(smem_u32(smem_src)), "l"(hint)
        : "memory");
}
DEVINL void tma_store_commit() { asm volatile("cp.async.bulk.commit_group;\n"); }
template <int N>
DEVINL void tma_store_wait() { asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N)); }
DEVINL void tma_store_fence() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }

DEVINL uint64_t policy_evict_first() {
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
    return p;
}
DEVINL uint64_t policy_evict_last() {
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
    return p;
}

// ---- tensor memory ----
DEVINL void tmem_alloc(uint32_t dst_word, uint32_t cols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
                 :: "r"(dst_word), "r"(cols));
}
DEVINL void tmem_relinquish() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
}
DEVINL void tmem_dealloc(uint32_t base, uint32_t cols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" :: "r"(base), "r"(cols));
}
DEVINL void tmem_ld_x8(uint32_t addr, uint32_t v[8]) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n"
                 : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]),
                   "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7])
                 : "r"(addr));
}
DEVINL void tmem_ld_wait() { asm volatile("tcgen05.wait::ld.sync.aligned;\n"); }

// ---- tcgen05 MMA ----
DEVINL void umma_f16(uint32_t tmem_c, uint64_t adesc, uint64_t bdesc, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
        :: "r"(tmem_c), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accum));
}
DEVINL void umma_commit(uint64_t* bar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
                 :: "r"(smem_u32(bar)));
}
DEVINL void tc_fence_before_sync() { asm volatile("tcgen05.fence::before_thread_sync;\n"); }
DEVINL void tc_fence_after_sync() { asm volatile("tcgen05.fence::after_thread_sync;\n"); }

DEVINL void bar_sync(int id, int count) {
    asm volatile("barrier.sync.aligned %0, %1;\n" :: "r"(id), "r"(count));
}

struct Sched {
    const int* masked_m;
    int E, NB;
    int g;
    int iter;
    long long cum;
};

DEVINL bool sched_next(Sched& s, int& e, int& mb, int& nb, int& mblk_out) {
    const long long idx = (long long)s.iter * gridDim.x + blockIdx.x;
    s.iter++;
    while (true) {
        if (s.g >= s.E) return false;
        const int mmv = __ldg(s.masked_m + s.g);
        const int mblk = (mmv + 127) >> 7;
        const long long nxt = s.cum + (long long)mblk * s.NB;
        if (idx < nxt) {
            const long long in = idx - s.cum;
            e = s.g;
            nb = (int)(in / mblk);
            mb = (int)(in - (long long)nb * mblk);
            mblk_out = mblk;
            return true;
        }
        s.g++;
        s.cum = nxt;
    }
}

template <int BN, int STAGES, bool AC>
__global__ void __launch_bounds__(256, 1)
ggm2_kernel(const __grid_constant__ CUtensorMap tmapA,
            const __grid_constant__ CUtensorMap tmapA8,
            const __grid_constant__ CUtensorMap tmapA32,
            const __grid_constant__ CUtensorMap tmapB,
            const __grid_constant__ CUtensorMap tmapD,
            const int* __restrict__ masked_m,
            int E, int CAP, int N, int K, int hints) {
    constexpr int BM = 128;
    constexpr int BK = 64;
    constexpr int A_STAGE = BM * BK * 2;      // 16KB
    constexpr int B_STAGE = BN * BK * 2;      // 16/32KB
    constexpr int CD_BYTES = 2 * BM * 64 * 2; // 2 TMA-store stages x (128 x 64 bf16)
    constexpr int SB = BN / 64;               // store blocks per tile

    extern __shared__ __align__(1024) char smem[];
    char* sA = smem;
    char* sB = smem + STAGES * A_STAGE;
    char* sCD = smem + STAGES * (A_STAGE + B_STAGE);
    uint64_t* full_bar = reinterpret_cast<uint64_t*>(sCD + CD_BYTES);
    uint64_t* empty_bar = full_bar + STAGES;
    uint64_t* tmem_full = empty_bar + STAGES;
    uint64_t* tmem_empty = tmem_full + 2;
    uint32_t* tmem_word = reinterpret_cast<uint32_t*>(tmem_empty + 2);
    uint32_t* has_word = tmem_word + 1;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int num_k = K / BK;
    const int NB = N / BN;

    if (warp == 3) {
        // Parallel probe on the idle warp (overlaps alloc/init): does this
        // CTA own any tile? Uniform across the CTA.
        long long tot = 0;
        for (int g = lane; g < E; g += 32)
            tot += ((__ldg(masked_m + g) + 127) >> 7);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            tot += __shfl_down_sync(0xffffffffu, tot, o);
        if (lane == 0)
            *(reinterpret_cast<volatile uint32_t*>(has_word)) =
                ((tot * NB) > (long long)blockIdx.x) ? 1u : 0u;
    } else if (warp == 2) {
        tmem_alloc(smem_u32(tmem_word), 2 * BN);
        tmem_relinquish();
    } else if (warp == 0 && lane == 0) {
        #pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            mbar_init(full_bar + s, 1);
            mbar_init(empty_bar + s, 1);
        }
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            mbar_init(tmem_full + i, 1);
            mbar_init(tmem_empty + i, 128);
        }
        mbar_init_fence();
    }
    __syncthreads();
    const uint32_t tmem_base = *(reinterpret_cast<volatile uint32_t*>(tmem_word));
    if (*(reinterpret_cast<volatile uint32_t*>(has_word)) == 0u) {
        if (warp == 2) tmem_dealloc(tmem_base, 2 * BN);
        return;
    }

    // PDL: overlap launch/init with the previous stream kernel when launched
    // with the programmatic-serialization attribute; all predecessor writes
    // (d) are complete before any role touches global memory. The masked_m
    // probe above only reads an unmutated input tensor. No-op without the
    // attribute.
    cudaGridDependencySynchronize();

    Sched sched{masked_m, E, NB, 0, 0, 0};

    if (warp == 0) {
        // TMA producer (single lane)
        if (lane == 0) {
            uint32_t st = 0, ph = 0;
            // bit4: unconditional A evict_last; bit16: SELECTIVE A-el - pin A
            // only when the whole A operand fits an L2-resident budget (A is
            // re-read across NB tiles; pinning saves DRAM re-fetch, but when
            // A exceeds L2 the pinning thrashes and squeezes B: measured
            // case9 -0.5..-0.7% with unconditional A-el, J1b/J1c).
            const bool ael = (hints & 4) ||
                ((hints & 16) && ((long long)E * CAP * K * 2 <= (100LL << 20)));
            const uint64_t hintA = ael ? policy_evict_last() : 0;
            const uint64_t hintB_static = (hints & 1) ? policy_evict_first() : 0;
            const uint64_t policy_ef = (hints & 8) ? policy_evict_first() : 0;
            int e, mb, nb, mblk;
            while (sched_next(sched, e, mb, nb, mblk)) {
                const int outerA = e * CAP + mb * BM;
                const int outerB = e * N + nb * BN;
                // GGM_ACLAMP (AC): row-clamped A loads. Partial blocks load
                // ceil(v/8) 8-row sub-boxes through tmapA8 (SW128-compatible:
                // XOR keys off row%8; r0 % 8 == 0 keeps the smem layout
                // bit-identical). Padding rows keep stale smem; the garbage
                // reaches only capacity-padding D rows (legal: write-optional,
                // pskip precedent). expect_tx tracks exact delivered bytes.
                // Full blocks: identical single-load path.
                int rows8 = BM;
                if (AC) {
                    const int vrows = __ldg(masked_m + e) - mb * BM;
                    rows8 = vrows >= BM ? BM : ((vrows + 7) & ~7);
                }
                // bit3: device-adaptive B evict_first when this expert's B tile
                // has no cross-m-block L2 reuse (mblk == 1).
                const uint64_t hintB = hintB_static ? hintB_static
                                       : ((hints & 8) && mblk == 1 ? policy_ef : 0);
                #pragma unroll 1
                for (int kb = 0; kb < num_k; ++kb) {
                    mbar_wait(empty_bar + st, ph ^ 1);
                    mbar_expect_tx(full_bar + st, rows8 * (BK * 2) + B_STAGE);
                    if (AC && rows8 < BM) {
                        // Mixed radix: 32-row boxes then <=3 8-row boxes.
                        // Arm-1 (8-row only) cost c6 +3.23% (v=64 => 8
                        // issues/stage at ~3.4ns marginal each); mixed
                        // radix keeps c0 at 1 issue and c6 at 2.
                        const int k32 = rows8 >> 5;
                        const int k8 = (rows8 & 31) >> 3;
                        #pragma unroll 1
                        for (int i = 0; i < k32; ++i) {
                            const int r0 = i << 5;
                            if (hintA) tma_load_2d_h(&tmapA32, full_bar + st, sA + st * A_STAGE + r0 * (BK * 2), kb * BK, outerA + r0, hintA);
                            else       tma_load_2d(&tmapA32, full_bar + st, sA + st * A_STAGE + r0 * (BK * 2), kb * BK, outerA + r0);
                        }
                        #pragma unroll 1
                        for (int i = 0; i < k8; ++i) {
                            const int r0 = (k32 << 5) + (i << 3);
                            if (hintA) tma_load_2d_h(&tmapA8, full_bar + st, sA + st * A_STAGE + r0 * (BK * 2), kb * BK, outerA + r0, hintA);
                            else       tma_load_2d(&tmapA8, full_bar + st, sA + st * A_STAGE + r0 * (BK * 2), kb * BK, outerA + r0);
                        }
                    } else {
                        if (hintA) tma_load_2d_h(&tmapA, full_bar + st, sA + st * A_STAGE, kb * BK, outerA, hintA);
                        else       tma_load_2d(&tmapA, full_bar + st, sA + st * A_STAGE, kb * BK, outerA);
                    }
                    if (hintB) tma_load_2d_h(&tmapB, full_bar + st, sB + st * B_STAGE, kb * BK, outerB, hintB);
                    else       tma_load_2d(&tmapB, full_bar + st, sB + st * B_STAGE, kb * BK, outerB);
                    if (++st == STAGES) { st = 0; ph ^= 1; }
                }
            }
        }
    } else if (warp == 1) {
        // UMMA issuer (single lane)
        if (lane == 0) {
            constexpr uint64_t kDescHi = (2ULL << 61) | (1ULL << 46) | ((1024ULL >> 4) << 32);
            const uint64_t ad0 = kDescHi | ((smem_u32(sA) >> 4) & 0x3FFFULL);
            const uint64_t bd0 = kDescHi | ((smem_u32(sB) >> 4) & 0x3FFFULL);
            const uint32_t idesc = (8u << 24) | ((uint32_t)(BN >> 3) << 17) |
                                   (1u << 10) | (1u << 7) | (1u << 4);
            // PROBE ONLY (GGM_PROBE, hints bits 24-29, forced-impl slow path):
            // runtime-shrink idesc.n_dim to measure UMMA N-rate scaling.
            // Produces WRONG output (tmem cols beyond probe N are stale);
            // never set in dispatch defaults.
            const uint32_t probe_n8 = ((uint32_t)hints >> 24) & 0x3fu;
            const uint32_t idesc_p = probe_n8
                ? ((idesc & ~(0x3fu << 17)) | (probe_n8 << 17)) : idesc;
            uint32_t st = 0, ph = 0;
            int acc_it = 0;
            int e, mb, nb, mblk;
            while (sched_next(sched, e, mb, nb, mblk)) {
                const int acc = acc_it & 1;
                const uint32_t aphase = (uint32_t)((acc_it >> 1) & 1);
                acc_it++;
                mbar_wait(tmem_empty + acc, aphase ^ 1);
                tc_fence_after_sync();
                const uint32_t tc = tmem_base + (uint32_t)(acc * BN);
                #pragma unroll 1
                for (int kb = 0; kb < num_k; ++kb) {
                    mbar_wait(full_bar + st, ph);
                    tc_fence_after_sync();
                    const uint64_t ad = ad0 + (uint64_t)((st * A_STAGE) >> 4);
                    const uint64_t bd = bd0 + (uint64_t)((st * B_STAGE) >> 4);
                    #pragma unroll
                    for (int i = 0; i < 4; ++i)
                        umma_f16(tc, ad + 2 * i, bd + 2 * i, idesc_p, (uint32_t)((kb | i) != 0));
                    umma_commit(empty_bar + st);
                    if (kb == num_k - 1) umma_commit(tmem_full + acc);
                    if (++st == STAGES) { st = 0; ph ^= 1; }
                }
            }
        }
    } else if (warp >= 4) {
        // Epilogue warpgroup (threads 128..255)
        const int ew = warp - 4;
        const int row = ew * 32 + lane;
        const int row_swz = row & 7;
        const uint64_t hintD = (hints & 2) ? policy_evict_first() : 0;
        int acc_it = 0;
        int ts = 0;
        int e, mb, nb, mblk;
        while (sched_next(sched, e, mb, nb, mblk)) {
            const int acc = acc_it & 1;
            const uint32_t aphase = (uint32_t)((acc_it >> 1) & 1);
            acc_it++;
            mbar_wait(tmem_full + acc, aphase);
            tc_fence_after_sync();
            const int m_out = e * CAP + mb * BM;
            const int n_base = nb * BN;
            const uint32_t tcol = tmem_base + (uint32_t)(acc * BN);
            char* cd = sCD + ts * (BM * 64 * 2);
            #pragma unroll 1
            for (int sb = 0; sb < SB; ++sb) {
                if (ew == 0) tma_store_wait<1>();
                bar_sync(1, 128);
                uint32_t vals[8][8];
                #pragma unroll
                for (int ld = 0; ld < 8; ++ld)
                    tmem_ld_x8(tcol + (uint32_t)(sb * 64 + ld * 8), vals[ld]);
                tmem_ld_wait();
                if (sb == SB - 1) {
                    tc_fence_before_sync();
                    mbar_arrive(tmem_empty + acc);
                }
                #pragma unroll
                for (int ld = 0; ld < 8; ++ld) {
                    uint4 packed;
                    __nv_bfloat162 p0 = __floats2bfloat162_rn(__uint_as_float(vals[ld][0]), __uint_as_float(vals[ld][1]));
                    __nv_bfloat162 p1 = __floats2bfloat162_rn(__uint_as_float(vals[ld][2]), __uint_as_float(vals[ld][3]));
                    __nv_bfloat162 p2 = __floats2bfloat162_rn(__uint_as_float(vals[ld][4]), __uint_as_float(vals[ld][5]));
                    __nv_bfloat162 p3 = __floats2bfloat162_rn(__uint_as_float(vals[ld][6]), __uint_as_float(vals[ld][7]));
                    packed.x = *reinterpret_cast<uint32_t*>(&p0);
                    packed.y = *reinterpret_cast<uint32_t*>(&p1);
                    packed.z = *reinterpret_cast<uint32_t*>(&p2);
                    packed.w = *reinterpret_cast<uint32_t*>(&p3);
                    *reinterpret_cast<uint4*>(cd + row * 128 + ((ld ^ row_swz) * 16)) = packed;
                }
                tma_store_fence();
                bar_sync(1, 128);
                if (ew == 0 && lane == 0) {
                    if (hintD) tma_store_2d_h(&tmapD, cd, n_base + sb * 64, m_out, hintD);
                    else       tma_store_2d(&tmapD, cd, n_base + sb * 64, m_out);
                    tma_store_commit();
                }
                ts ^= 1;
                cd = sCD + ts * (BM * 64 * 2);
            }
        }
        if (ew == 0 && lane == 0) tma_store_wait<0>();
    }

    __syncthreads();
    if (warp == 2) {
        tmem_dealloc(tmem_base, 2 * BN);
    }
}

namespace {

struct DescKey {
    const void* a; const void* b; const void* d;
    int E, CAP, N, K, bn;
    bool operator<(const DescKey& o) const {
        if (a != o.a) return a < o.a;
        if (b != o.b) return b < o.b;
        if (d != o.d) return d < o.d;
        if (E != o.E) return E < o.E;
        if (CAP != o.CAP) return CAP < o.CAP;
        if (N != o.N) return N < o.N;
        if (K != o.K) return K < o.K;
        return bn < o.bn;
    }
};

struct DescVal { CUtensorMap a, a8, a32, b, d; };

std::map<DescKey, DescVal> g_desc_cache;

inline CUtensorMapL2promotion l2_promo_env() {
    static int v = -2;
    if (v == -2) {
        const char* e = getenv("GGM_L2PROMO");
        v = e ? atoi(e) : 2;
    }
    return v == 0 ? CU_TENSOR_MAP_L2_PROMOTION_NONE :
           v == 1 ? CU_TENSOR_MAP_L2_PROMOTION_L2_64B :
           v == 3 ? CU_TENSOR_MAP_L2_PROMOTION_L2_128B :
                    CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
}

int encode_2d(CUtensorMap* out, const void* ptr, uint64_t inner, uint64_t outer,
              uint64_t outer_stride_bytes, uint32_t box_inner, uint32_t box_outer) {
    const cuuint64_t dims[2] = {inner, outer};
    const cuuint64_t strides[1] = {outer_stride_bytes};
    const cuuint32_t box[2] = {box_inner, box_outer};
    const cuuint32_t elem[2] = {1, 1};
    CUresult r = cuTensorMapEncodeTiled(
        out, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr),
        dims, strides, box, elem,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        l2_promo_env(), CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return (int)r;
}

template <int BN, int STAGES, bool AC>
int launch_one(const DescVal& v, const int* mm, int E, int CAP, int N, int K,
               int num_sms, int hints, int pdl, cudaStream_t stream) {
    constexpr int BM = 128, BK = 64;
    constexpr int SMEM = STAGES * (BM * BK * 2 + BN * BK * 2) + 2 * BM * 64 * 2
                       + (2 * STAGES + 4) * 8 + 8 + 128;
    static bool inited = false;
    if (!inited) {
        cudaError_t e = cudaFuncSetAttribute((const void*)ggm2_kernel<BN, STAGES, AC>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        if (e != cudaSuccess) return (int)e;
        inited = true;
    }
    if (pdl) {
        cudaLaunchConfig_t cfg = {};
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = attrs;
        cfg.numAttrs = 1;
        cfg.gridDim = dim3((unsigned)num_sms, 1, 1);
        cfg.blockDim = dim3(256, 1, 1);
        cfg.dynamicSmemBytes = SMEM;
        cfg.stream = stream;
        cudaError_t rc = cudaLaunchKernelEx(&cfg, ggm2_kernel<BN, STAGES, AC>,
                                            v.a, v.a8, v.a32, v.b, v.d, mm, E, CAP, N, K, hints);
        return (int)rc;
    }
    ggm2_kernel<BN, STAGES, AC><<<num_sms, 256, SMEM, stream>>>(v.a, v.a8, v.a32, v.b, v.d, mm, E, CAP, N, K, hints);
    return 0;
}

}  // namespace

extern "C" int ggm2_launch(const void* A, const void* B, void* D, const int* masked_m,
                           int E, int CAP, int N, int K, int bn_ovr, int hints,
                           int pdl, int grid_ovr, cudaStream_t stream) {
    static int num_sms = -1;
    if (num_sms < 0) {
        if (cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess)
            return -10;
    }
    // grid_ovr>0: persistent-grid wave balancing (any grid >=1 is exact; the
    // sched walk adapts). 0 = all SMs.
    const int grid = grid_ovr > 0 ? grid_ovr : num_sms;
    const int bn = bn_ovr > 0 ? bn_ovr : (N % 256 == 0 ? 256 : 128);
    if (CAP % 128 != 0 || K % 64 != 0 || N % bn != 0 || (bn != 64 && bn != 128 && bn != 256))
        return -2;
    DescKey key{A, B, D, E, CAP, N, K, bn};
    auto it = g_desc_cache.find(key);
    if (it == g_desc_cache.end()) {
        if (g_desc_cache.size() > 64) g_desc_cache.clear();
        DescVal v;
        int rc = encode_2d(&v.a, A, (uint64_t)K, (uint64_t)E * CAP, (uint64_t)K * 2, 64, 128);
        // GGM_ACLAMP: 8-row-box A map for row-clamped partial-block loads.
        if (rc == 0) rc = encode_2d(&v.a8, A, (uint64_t)K, (uint64_t)E * CAP, (uint64_t)K * 2, 64, 8);
        // GGM_ACLAMP: 32-row-box A map (mixed-radix head; 8-row tail).
        if (rc == 0) rc = encode_2d(&v.a32, A, (uint64_t)K, (uint64_t)E * CAP, (uint64_t)K * 2, 64, 32);
        if (rc == 0) rc = encode_2d(&v.b, B, (uint64_t)K, (uint64_t)E * N, (uint64_t)K * 2, 64, (uint32_t)bn);
        if (rc == 0) rc = encode_2d(&v.d, D, (uint64_t)N, (uint64_t)E * CAP, (uint64_t)N * 2, 64, 128);
        if (rc != 0) return rc;
        it = g_desc_cache.emplace(key, v).first;
    }
    int rc;
    // GGM_ACLAMP: v1 hints bit7 selects the row-clamped-A instantiation.
    // bit7 is free in the default 113; the kernel-local probe field uses
    // bits 24-29 of the hints int and never sees bit7.
    if (hints & 128) {
        if (bn == 256)      rc = launch_one<256, 4, true>(it->second, masked_m, E, CAP, N, K, grid, hints, pdl, stream);
        else if (bn == 128) rc = launch_one<128, 6, true>(it->second, masked_m, E, CAP, N, K, grid, hints, pdl, stream);
        else                rc = launch_one<64, 8, true>(it->second, masked_m, E, CAP, N, K, grid, hints, pdl, stream);
    } else {
        if (bn == 256)      rc = launch_one<256, 4, false>(it->second, masked_m, E, CAP, N, K, grid, hints, pdl, stream);
        else if (bn == 128) rc = launch_one<128, 6, false>(it->second, masked_m, E, CAP, N, K, grid, hints, pdl, stream);
        else                rc = launch_one<64, 8, false>(it->second, masked_m, E, CAP, N, K, grid, hints, pdl, stream);
    }
    if (rc != 0) return rc;
    return (int)cudaGetLastError();
}

// ---------------- split-K variant (experimental, impl 4) ----------------
// v1-derived 1-CTA tcgen05 kernel writing fp32 partials to a workspace, plus
// a deterministic reduce kernel to bf16 d. Targets low-occupancy small-m
// classes (few m-blocks, long K) where wave quantization costs 10-20%.

template <int BN, int STAGES, int SPLIT>
__global__ void __launch_bounds__(256, 1)
ggm4_kernel(const __grid_constant__ CUtensorMap tmapA,
            const __grid_constant__ CUtensorMap tmapB,
            const __grid_constant__ CUtensorMap tmapW,
            const int* __restrict__ masked_m,
            int E, int CAP, int N, int K) {
    constexpr int BM = 128;
    constexpr int BK = 64;
    constexpr int A_STAGE = BM * BK * 2;
    constexpr int B_STAGE = BN * BK * 2;
    constexpr int CD_BYTES = 2 * BM * 32 * 4;   // 2 stages x (128 rows x 32 fp32)
    constexpr int SB = BN / 32;                 // 32-col fp32 store blocks

    extern __shared__ __align__(1024) char smem[];
    char* sA = smem;
    char* sB = smem + STAGES * A_STAGE;
    char* sCD = smem + STAGES * (A_STAGE + B_STAGE);
    uint64_t* full_bar = reinterpret_cast<uint64_t*>(sCD + CD_BYTES);
    uint64_t* empty_bar = full_bar + STAGES;
    uint64_t* tmem_full = empty_bar + STAGES;
    uint64_t* tmem_empty = tmem_full + 2;
    uint32_t* tmem_word = reinterpret_cast<uint32_t*>(tmem_empty + 2);

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int num_k = K / BK;
    const int NB = N / BN;

    if (warp == 0 && lane == 0) {
        #pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            mbar_init(full_bar + s, 1);
            mbar_init(empty_bar + s, 1);
        }
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            mbar_init(tmem_full + i, 1);
            mbar_init(tmem_empty + i, 128);
        }
        mbar_init_fence();
    } else if (warp == 2) {
        tmem_alloc(smem_u32(tmem_word), 2 * BN);
        tmem_relinquish();
    }
    __syncthreads();
    const uint32_t tmem_base = *(reinterpret_cast<volatile uint32_t*>(tmem_word));

    struct SchedS { int g, iter; long long cum; };
    SchedS sched{0, 0, 0};
    auto sched_next_s = [&](int& e, int& mb, int& nb, int& sp) {
        const long long idx = (long long)sched.iter * gridDim.x + blockIdx.x;
        sched.iter++;
        while (true) {
            if (sched.g >= E) return false;
            const int mmv = __ldg(masked_m + sched.g);
            const int mblk = (mmv + 127) >> 7;
            const long long nxt = sched.cum + (long long)mblk * NB * SPLIT;
            if (idx < nxt) {
                const long long in = idx - sched.cum;
                e = sched.g;
                sp = (int)(in % SPLIT);
                const long long rest = in / SPLIT;
                nb = (int)(rest / mblk);
                mb = (int)(rest - (long long)nb * mblk);
                return true;
            }
            sched.g++;
            sched.cum = nxt;
        }
    };

    if (warp == 0) {
        if (lane == 0) {
            uint32_t st = 0, ph = 0;
            const uint64_t hintB = policy_evict_first();
            int e, mb, nb, sp;
            while (sched_next_s(e, mb, nb, sp)) {
                const int outerA = e * CAP + mb * BM;
                const int outerB = e * N + nb * BN;
                const int kb0 = sp * num_k / SPLIT;
                const int kb1 = (sp + 1) * num_k / SPLIT;
                #pragma unroll 1
                for (int kb = kb0; kb < kb1; ++kb) {
                    mbar_wait(empty_bar + st, ph ^ 1);
                    mbar_expect_tx(full_bar + st, A_STAGE + B_STAGE);
                    tma_load_2d(&tmapA, full_bar + st, sA + st * A_STAGE, kb * BK, outerA);
                    tma_load_2d_h(&tmapB, full_bar + st, sB + st * B_STAGE, kb * BK, outerB, hintB);
                    if (++st == STAGES) { st = 0; ph ^= 1; }
                }
            }
        }
    } else if (warp == 1) {
        if (lane == 0) {
            constexpr uint64_t kDescHi = (2ULL << 61) | (1ULL << 46) | ((1024ULL >> 4) << 32);
            const uint64_t ad0 = kDescHi | ((smem_u32(sA) >> 4) & 0x3FFFULL);
            const uint64_t bd0 = kDescHi | ((smem_u32(sB) >> 4) & 0x3FFFULL);
            const uint32_t idesc = (8u << 24) | ((uint32_t)(BN >> 3) << 17) |
                                   (1u << 10) | (1u << 7) | (1u << 4);
            uint32_t st = 0, ph = 0;
            int acc_it = 0;
            int e, mb, nb, sp;
            while (sched_next_s(e, mb, nb, sp)) {
                const int acc = acc_it & 1;
                const uint32_t aphase = (uint32_t)((acc_it >> 1) & 1);
                acc_it++;
                mbar_wait(tmem_empty + acc, aphase ^ 1);
                tc_fence_after_sync();
                const uint32_t tc = tmem_base + (uint32_t)(acc * BN);
                const int kb0 = sp * num_k / SPLIT;
                const int kb1 = (sp + 1) * num_k / SPLIT;
                #pragma unroll 1
                for (int kb = kb0; kb < kb1; ++kb) {
                    mbar_wait(full_bar + st, ph);
                    tc_fence_after_sync();
                    const uint64_t ad = ad0 + (uint64_t)((st * A_STAGE) >> 4);
                    const uint64_t bd = bd0 + (uint64_t)((st * B_STAGE) >> 4);
                    #pragma unroll
                    for (int i = 0; i < 4; ++i)
                        umma_f16(tc, ad + 2 * i, bd + 2 * i, idesc,
                                 (uint32_t)((kb > kb0) | (i != 0)));
                    umma_commit(empty_bar + st);
                    if (kb == kb1 - 1) umma_commit(tmem_full + acc);
                    if (++st == STAGES) { st = 0; ph ^= 1; }
                }
            }
        }
    } else if (warp >= 4) {
        const int ew = warp - 4;
        const int row = ew * 32 + lane;
        const int row_swz = row & 7;
        int acc_it = 0;
        int ts = 0;
        int e, mb, nb, sp;
        while (sched_next_s(e, mb, nb, sp)) {
            const int acc = acc_it & 1;
            const uint32_t aphase = (uint32_t)((acc_it >> 1) & 1);
            acc_it++;
            mbar_wait(tmem_full + acc, aphase);
            tc_fence_after_sync();
            const int m_out = sp * (E * CAP) + e * CAP + mb * BM;   // W outer coord
            const int n_base = nb * BN;
            const uint32_t tcol = tmem_base + (uint32_t)(acc * BN);
            char* cd = sCD + ts * (BM * 32 * 4);
            #pragma unroll 1
            for (int sb = 0; sb < SB; ++sb) {
                if (ew == 0) tma_store_wait<1>();
                bar_sync(1, 128);
                uint32_t vals[4][8];
                #pragma unroll
                for (int ld = 0; ld < 4; ++ld)
                    tmem_ld_x8(tcol + (uint32_t)(sb * 32 + ld * 8), vals[ld]);
                tmem_ld_wait();
                if (sb == SB - 1) {
                    tc_fence_before_sync();
                    mbar_arrive(tmem_empty + acc);
                }
                #pragma unroll
                for (int ld = 0; ld < 4; ++ld) {
                    uint4 q0 = make_uint4(vals[ld][0], vals[ld][1], vals[ld][2], vals[ld][3]);
                    uint4 q1 = make_uint4(vals[ld][4], vals[ld][5], vals[ld][6], vals[ld][7]);
                    *reinterpret_cast<uint4*>(cd + row * 128 + (((ld * 2 + 0) ^ row_swz) * 16)) = q0;
                    *reinterpret_cast<uint4*>(cd + row * 128 + (((ld * 2 + 1) ^ row_swz) * 16)) = q1;
                }
                tma_store_fence();
                bar_sync(1, 128);
                if (ew == 0 && lane == 0) {
                    tma_store_2d(&tmapW, cd, n_base + sb * 32, m_out);
                    tma_store_commit();
                }
                ts ^= 1;
                cd = sCD + ts * (BM * 32 * 4);
            }
        }
        if (ew == 0 && lane == 0) tma_store_wait<0>();
    }

    __syncthreads();
    if (warp == 2) {
        tmem_dealloc(tmem_base, 2 * BN);
    }
}

__global__ void ggm4_reduce(const float* __restrict__ W, __nv_bfloat16* __restrict__ D,
                            const int* __restrict__ masked_m,
                            int E, int CAP, int N, int SPLIT) {
    const int NVEC = N >> 3;
    const long long rows_total = (long long)E * CAP;
    const long long total = rows_total * NVEC;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < total; t += stride) {
        const long long rc = t / NVEC;                 // flattened (e, row)
        const int col = (int)(t - rc * NVEC) * 8;
        const int e = (int)(rc / CAP);
        const int row = (int)(rc - (long long)e * CAP);
        if (row >= __ldg(masked_m + e)) continue;
        const float* w = W + rc * N + col;
        float4 x0 = *reinterpret_cast<const float4*>(w);
        float4 x1 = *reinterpret_cast<const float4*>(w + 4);
        for (int s = 1; s < SPLIT; ++s) {
            const float* ws = W + ((long long)s * rows_total + rc) * N + col;
            const float4 y0 = *reinterpret_cast<const float4*>(ws);
            const float4 y1 = *reinterpret_cast<const float4*>(ws + 4);
            x0.x += y0.x; x0.y += y0.y; x0.z += y0.z; x0.w += y0.w;
            x1.x += y1.x; x1.y += y1.y; x1.z += y1.z; x1.w += y1.w;
        }
        __nv_bfloat162 p0 = __floats2bfloat162_rn(x0.x, x0.y);
        __nv_bfloat162 p1 = __floats2bfloat162_rn(x0.z, x0.w);
        __nv_bfloat162 p2 = __floats2bfloat162_rn(x1.x, x1.y);
        __nv_bfloat162 p3 = __floats2bfloat162_rn(x1.z, x1.w);
        uint4 outv;
        outv.x = *reinterpret_cast<uint32_t*>(&p0);
        outv.y = *reinterpret_cast<uint32_t*>(&p1);
        outv.z = *reinterpret_cast<uint32_t*>(&p2);
        outv.w = *reinterpret_cast<uint32_t*>(&p3);
        *reinterpret_cast<uint4*>(D + rc * N + col) = outv;
    }
}

template <int BN, int STAGES, int SPLIT>
int launch_one4(const DescVal& v, const CUtensorMap& wmap, const int* mm,
                int E, int CAP, int N, int K, int num_sms, cudaStream_t stream) {
    constexpr int BM = 128, BK = 64;
    constexpr int SMEM = STAGES * (BM * BK * 2 + BN * BK * 2) + 2 * BM * 32 * 4
                       + (2 * STAGES + 4) * 8 + 8 + 128;
    static bool inited = false;
    if (!inited) {
        cudaError_t e = cudaFuncSetAttribute((const void*)ggm4_kernel<BN, STAGES, SPLIT>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        if (e != cudaSuccess) return (int)e;
        inited = true;
    }
    ggm4_kernel<BN, STAGES, SPLIT><<<num_sms, 256, SMEM, stream>>>(v.a, v.b, wmap, mm, E, CAP, N, K);
    return 0;
}

extern "C" int ggm4_launch(const void* A, const void* B, void* W, const int* masked_m,
                           int E, int CAP, int N, int K, int split, cudaStream_t stream) {
    static int num_sms = -1;
    if (num_sms < 0) {
        if (cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess)
            return -10;
    }
    if (split != 2 || CAP % 128 != 0 || K % 128 != 0 || N % 256 != 0) return -2;
    static std::map<DescKey, DescVal> cache4;
    DescKey key{A, B, W, E, CAP, N, K, split};
    auto it = cache4.find(key);
    if (it == cache4.end()) {
        if (cache4.size() > 64) cache4.clear();
        DescVal v;
        int rc = encode_2d(&v.a, A, (uint64_t)K, (uint64_t)E * CAP, (uint64_t)K * 2, 64, 128);
        if (rc == 0) rc = encode_2d(&v.b, B, (uint64_t)K, (uint64_t)E * N, (uint64_t)K * 2, 64, 256);
        if (rc != 0) return rc;
        it = cache4.emplace(key, v).first;
    }
    // W descriptor: fp32, dims {N, split*E*CAP}, box {32, 128}
    CUtensorMap wmap;
    {
        const cuuint64_t dims[2] = {(uint64_t)N, (uint64_t)split * E * CAP};
        const cuuint64_t strides[1] = {(uint64_t)N * 4};
        const cuuint32_t box[2] = {32, 128};
        const cuuint32_t elem[2] = {1, 1};
        CUresult r = cuTensorMapEncodeTiled(&wmap, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, W,
            dims, strides, box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (r != CUDA_SUCCESS) return (int)r;
    }
    int rc = launch_one4<256, 4, 2>(it->second, wmap, masked_m, E, CAP, N, K, num_sms, stream);
    if (rc != 0) return rc;
    return (int)cudaGetLastError();
}

extern "C" int ggm4_reduce_launch(const void* W, void* D, const int* masked_m,
                                  int E, int CAP, int N, int split, cudaStream_t stream) {
    const long long total = (long long)E * CAP * (N >> 3);
    int blocks = (int)((total + 255) / 256);
    if (blocks > 8192) blocks = 8192;
    if (blocks < 1) blocks = 1;
    ggm4_reduce<<<blocks, 256, 0, stream>>>((const float*)W, (__nv_bfloat16*)D, masked_m,
                                            E, CAP, N, split);
    return (int)cudaGetLastError();
}

}  // namespace ggm2
'''


CUDA_V2 = r'''
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <map>

// v2: 2-CTA cluster tcgen05 (cta_group::2, UMMA M=256) persistent masked
// grouped GEMM for SM100-family. Each CTA holds its 128 A-rows and BN/2
// B-columns per stage; the pair's leader issues the 2SM UMMA; T completions
// signal the leader's full barrier; empty/tmem_full commits multicast to both
// CTAs; each CTA epilogues its own 128 D rows from its own tmem.
// Independent implementation; mechanisms informed by the public PTX ISA and
// DeepGEMM e1f418c / atrex-fa 2-CTA references. No CUTLASS/CuTe.

namespace ggm3 {

#define DEVINL __device__ __forceinline__

DEVINL uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
DEVINL uint32_t cluster_rank() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}
DEVINL uint32_t mapa0(uint32_t local_addr) {  // same offset in CTA rank 0
    uint32_t r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(local_addr), "r"(0));
    return r;
}
DEVINL void cluster_sync() {
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

// ---- mbarrier ----
DEVINL void mbar_init(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(smem_u32(bar)), "r"(count));
}
DEVINL void mbar_init_fence() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
DEVINL void mbar_arrive_local(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(smem_u32(bar)));
}
DEVINL void mbar_arrive_rank0(uint64_t* bar) {  // remote arrive on leader copy
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];\n" :: "r"(mapa0(smem_u32(bar))) : "memory");
}
DEVINL void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                 :: "r"(smem_u32(bar)), "r"(bytes));
}
DEVINL void mbar_wait(uint64_t* bar, uint32_t parity) {
    asm volatile(
        "{\n\t.reg .pred P;\n\t"
        "WAIT_%=:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1, %2;\n\t"
        "@P bra DONE_%=;\n\t"
        "bra WAIT_%=;\n\t"
        "DONE_%=:\n\t}\n"
        :: "r"(smem_u32(bar)), "r"(parity), "r"(0x989680));
}

// ---- TMA (cta_group::2 loads signal the pair leader's barrier; mapa0 makes
// the operand the leader's cluster address under either HW interpretation) ----
DEVINL void tma_load_2d_c2(const CUtensorMap* desc, uint64_t* bar, void* smem_dst, int c0, int c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(smem_u32(smem_dst)), "l"(reinterpret_cast<uint64_t>(desc)),
           "r"(c0), "r"(c1), "r"(mapa0(smem_u32(bar)))
        : "memory");
}
DEVINL void tma_load_2d_c2_h(const CUtensorMap* desc, uint64_t* bar, void* smem_dst, int c0, int c1, uint64_t hint) {
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
        "[%0], [%1, {%2, %3}], [%4], %5;\n"
        :: "r"(smem_u32(smem_dst)), "l"(reinterpret_cast<uint64_t>(desc)),
           "r"(c0), "r"(c1), "r"(mapa0(smem_u32(bar))), "l"(hint)
        : "memory");
}
DEVINL void tma_store_2d(const CUtensorMap* desc, const void* smem_src, int c0, int c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n"
        :: "l"(reinterpret_cast<uint64_t>(desc)), "r"(c0), "r"(c1), "r"(smem_u32(smem_src))
        : "memory");
}
DEVINL void tma_store_2d_h(const CUtensorMap* desc, const void* smem_src, int c0, int c1, uint64_t hint) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group.L2::cache_hint [%0, {%1, %2}], [%3], %4;\n"
        :: "l"(reinterpret_cast<uint64_t>(desc)), "r"(c0), "r"(c1), "r"(smem_u32(smem_src)), "l"(hint)
        : "memory");
}
DEVINL void tma_store_commit() { asm volatile("cp.async.bulk.commit_group;\n"); }
template <int N>
DEVINL void tma_store_wait() { asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N)); }
DEVINL void tma_store_fence() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }

DEVINL uint64_t policy_evict_first() {
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
    return p;
}
DEVINL uint64_t policy_evict_last() {
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
    return p;
}

// ---- tensor memory (cta_group::2) ----
DEVINL void tmem_alloc2(uint32_t dst_word, uint32_t cols) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;\n"
                 :: "r"(dst_word), "r"(cols));
}
DEVINL void tmem_relinquish2() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;\n");
}
DEVINL void tmem_dealloc2(uint32_t base, uint32_t cols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;\n" :: "r"(base), "r"(cols));
}
DEVINL void tmem_ld_x8(uint32_t addr, uint32_t v[8]) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n"
                 : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]),
                   "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7])
                 : "r"(addr));
}
DEVINL void tmem_ld_wait() { asm volatile("tcgen05.wait::ld.sync.aligned;\n"); }

// ---- tcgen05 MMA (2SM) ----
DEVINL void umma_f16_2sm(uint32_t tmem_c, uint64_t adesc, uint64_t bdesc, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
        :: "r"(tmem_c), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accum));
}
DEVINL void umma_commit2_mc(uint64_t* bar) {  // multicast commit to both CTAs
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;\n"
        :: "r"(smem_u32(bar)), "h"((uint16_t)0x3));
}
DEVINL void tc_fence_before_sync() { asm volatile("tcgen05.fence::before_thread_sync;\n"); }
DEVINL void tc_fence_after_sync() { asm volatile("tcgen05.fence::after_thread_sync;\n"); }

DEVINL void bar_sync(int id, int count) {
    asm volatile("barrier.sync.aligned %0, %1;\n" :: "r"(id), "r"(count));
}

struct Sched {
    const int* masked_m;
    int E, NB;
    int g;
    int iter;
    long long cum;
};

DEVINL bool sched_next(Sched& s, int& e, int& mb, int& nb, int& mblk_out,
                       int& mmv_out) {
    const long long idx = (long long)s.iter * (long long)(gridDim.x >> 1)
                        + (long long)(blockIdx.x >> 1);
    s.iter++;
    while (true) {
        if (s.g >= s.E) return false;
        const int mmv = __ldg(s.masked_m + s.g);
        const int mblk = (mmv + 255) >> 8;
        const long long nxt = s.cum + (long long)mblk * s.NB;
        if (idx < nxt) {
            const long long in = idx - s.cum;
            e = s.g;
            nb = (int)(in / mblk);
            mb = (int)(in - (long long)nb * mblk);
            mblk_out = mblk;
            mmv_out = mmv;
            return true;
        }
        s.g++;
        s.cum = nxt;
    }
}

template <int BN, int STAGES, int CD_STAGES, int BK, int EP>
__global__ void __launch_bounds__(256, 1)
ggm3_kernel(const __grid_constant__ CUtensorMap tmapA,
            const __grid_constant__ CUtensorMap tmapB,
            const __grid_constant__ CUtensorMap tmapD,
            const __grid_constant__ CUtensorMap tmapD32,
            const int* __restrict__ masked_m,
            int E, int CAP, int N, int K, int hints) {
    constexpr int BM2 = 256;                  // cluster tile rows
    constexpr int BMH = 128;                  // per-CTA rows
    constexpr int BH = BN / 2;                // per-CTA B columns
    constexpr int ATOMS = BK / 64;            // 128B-swizzle k atoms per stage
    constexpr int A_STAGE = BMH * BK * 2;     // 16KB
    constexpr int B_STAGE = BH * BK * 2;      // 16KB (BN=256)
    constexpr int CD_BYTES = CD_STAGES * BMH * 64 * 2;
    constexpr int SB = BN / 64;               // full 64-col store chunks
    constexpr int TAILW = BN % 64;            // partial tail chunk (0 or 32)
    constexpr int NCHUNK = SB + (TAILW ? 1 : 0);
    // tcgen05.alloc requires a power-of-two column count (>=32).
    constexpr int TMEM_COLS = ((2 * BN) & (2 * BN - 1)) == 0 ? 2 * BN : 512;

    extern __shared__ __align__(1024) char smem[];
    char* sA = smem;
    char* sB = smem + STAGES * A_STAGE;
    char* sCD = smem + STAGES * (A_STAGE + B_STAGE);
    uint64_t* full_bar = reinterpret_cast<uint64_t*>(sCD + CD_BYTES);
    uint64_t* empty_bar = full_bar + STAGES;
    uint64_t* tmem_full = empty_bar + STAGES;
    uint64_t* tmem_empty = tmem_full + 2;
    uint32_t* tmem_word = reinterpret_cast<uint32_t*>(tmem_empty + 2);
    uint32_t* has_word = tmem_word + 1;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const uint32_t rank = cluster_rank();
    const bool leader = (rank == 0);
    const int num_k = K / BK;
    const int NB = N / BN;

    if constexpr (EP) {
        // Empty-path variant (GGM_VAR=4): probe FIRST; a cluster that owns no
        // tile returns before any tmem alloc (no alloc => no dealloc, no
        // relinquish, one fewer syncthreads) - all-empty launches (case1)
        // skip the whole TMEM handshake. Working clusters pay a serialized
        // alloc (~0.3us) that the default variant overlaps with the probe.
        if (warp == 3) {
            long long tot = 0;
            for (int g = lane; g < E; g += 32)
                tot += ((__ldg(masked_m + g) + 255) >> 8);
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                tot += __shfl_down_sync(0xffffffffu, tot, o);
            if (lane == 0)
                *(reinterpret_cast<volatile uint32_t*>(has_word)) =
                    ((tot * NB) > (long long)(blockIdx.x >> 1)) ? 1u : 0u;
        }
        __syncthreads();
        if (*(reinterpret_cast<volatile uint32_t*>(has_word)) == 0u) {
            // uniform per pair; no cluster collective has run at this point
            return;
        }
        if (warp == 2) {
            tmem_alloc2(smem_u32(tmem_word), TMEM_COLS);
            tmem_relinquish2();
        } else if (warp == 0 && lane == 0) {
            #pragma unroll
            for (int s = 0; s < STAGES; ++s) {
                mbar_init(full_bar + s, 2);
                mbar_init(empty_bar + s, 1);
            }
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                mbar_init(tmem_full + i, 1);
                mbar_init(tmem_empty + i, 256);
            }
            mbar_init_fence();
        }
        __syncthreads();
    } else {
        if (warp == 3) {
            // Parallel probe on the idle warp (overlaps alloc/init): does this
            // cluster own any tile? Both CTAs of a pair derive the identical
            // answer from the same cluster id, so the pair uniformly skips or
            // joins every cluster collective below.
            long long tot = 0;
            for (int g = lane; g < E; g += 32)
                tot += ((__ldg(masked_m + g) + 255) >> 8);
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                tot += __shfl_down_sync(0xffffffffu, tot, o);
            if (lane == 0)
                *(reinterpret_cast<volatile uint32_t*>(has_word)) =
                    ((tot * NB) > (long long)(blockIdx.x >> 1)) ? 1u : 0u;
        } else if (warp == 2) {
            tmem_alloc2(smem_u32(tmem_word), TMEM_COLS);
            tmem_relinquish2();
        } else if (warp == 0 && lane == 0) {
            #pragma unroll
            for (int s = 0; s < STAGES; ++s) {
                mbar_init(full_bar + s, 2);
                mbar_init(empty_bar + s, 1);
            }
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                mbar_init(tmem_full + i, 1);
                mbar_init(tmem_empty + i, 256);
            }
            mbar_init_fence();
        }
        __syncthreads();
    }
    const uint32_t tmem_base = *(reinterpret_cast<volatile uint32_t*>(tmem_word));
    if (!EP && *(reinterpret_cast<volatile uint32_t*>(has_word)) == 0u) {
        // uniform per pair: no cluster collective below is left unbalanced
        if (warp == 2) tmem_dealloc2(tmem_base, TMEM_COLS);
        return;
    }
    cluster_sync();

    // PDL: overlap launch/init with the previous stream kernel (the harness's
    // d restore); all predecessor writes (d) are complete before any role
    // touches global memory. No-op when launched without the PDL attribute.
    cudaGridDependencySynchronize();

    Sched sched{masked_m, E, NB, 0, 0, 0};

    if (warp == 0) {
        // TMA producer on BOTH CTAs: each loads its A-row half and B-col half
        if (lane == 0) {
            uint32_t st = 0, ph = 0;
            // bit4: unconditional A evict_last; bit16: SELECTIVE A-el - pin A
            // only when the whole A operand fits an L2-resident budget (A is
            // re-read across NB tiles; pinning saves DRAM re-fetch, but when
            // A exceeds L2 the pinning thrashes and squeezes B: measured
            // case9 -0.5..-0.7% with unconditional A-el, J1b/J1c).
            const bool ael = (hints & 4) ||
                ((hints & 16) && ((long long)E * CAP * K * 2 <= (100LL << 20)));
            const uint64_t hintA = ael ? policy_evict_last() : 0;
            const uint64_t hintB_static = (hints & 1) ? policy_evict_first() : 0;
            const uint64_t policy_ef = (hints & 8) ? policy_evict_first() : 0;
            int e, mb, nb, mblk, mmv;
            while (sched_next(sched, e, mb, nb, mblk, mmv)) {
                const int outerA = e * CAP + mb * BM2 + (int)rank * BMH;
                const int outerB = e * N + nb * BN + (int)rank * BH;
                // bit3: device-adaptive B evict_first when this expert's B tile
                // has no cross-m-block L2 reuse (mblk == 1).
                const uint64_t hintB = hintB_static ? hintB_static
                                       : ((hints & 8) && mblk == 1 ? policy_ef : 0);
                // bit6: skip rank-1's A loads when its 128 rows are all
                // padding (same condition as the epilogue's bit5 store skip).
                // The pair UMMA consumes stale smem for those rows; the
                // garbage results are never stored (bit5). Both CTAs run the
                // identical walk, so the leader's trimmed expect_tx byte
                // accounting matches the actual TMA arrivals exactly.
                const bool r1skip = (hints & 64) && (mmv <= mb * BM2 + BMH);
                const bool skipA = r1skip && (rank == 1);
                const uint32_t a_bytes = (r1skip ? 1u : 2u) * A_STAGE;
                #pragma unroll 1
                for (int kb = 0; kb < num_k; ++kb) {
                    mbar_wait(empty_bar + st, ph ^ 1);
                    if (leader) mbar_expect_tx(full_bar + st, a_bytes + 2 * B_STAGE);
                    else        mbar_arrive_rank0(full_bar + st);
                    #pragma unroll
                    for (int at = 0; at < ATOMS; ++at) {
                        const int koff = kb * BK + at * 64;
                        char* da = sA + st * A_STAGE + at * (BMH * 64 * 2);
                        char* db = sB + st * B_STAGE + at * (BH * 64 * 2);
                        if (!skipA) {
                            if (hintA) tma_load_2d_c2_h(&tmapA, full_bar + st, da, koff, outerA, hintA);
                            else       tma_load_2d_c2(&tmapA, full_bar + st, da, koff, outerA);
                        }
                        if (hintB) tma_load_2d_c2_h(&tmapB, full_bar + st, db, koff, outerB, hintB);
                        else       tma_load_2d_c2(&tmapB, full_bar + st, db, koff, outerB);
                    }
                    if (++st == STAGES) { st = 0; ph ^= 1; }
                }
            }
        }
    } else if (warp == 1) {
        // Pair UMMA issuer: leader CTA only
        if (leader && lane == 0) {
            constexpr uint64_t kDescHi = (2ULL << 61) | (1ULL << 46) | ((1024ULL >> 4) << 32);
            const uint64_t ad0 = kDescHi | ((smem_u32(sA) >> 4) & 0x3FFFULL);
            const uint64_t bd0 = kDescHi | ((smem_u32(sB) >> 4) & 0x3FFFULL);
            const uint32_t idesc = (16u << 24) | ((uint32_t)(BN >> 3) << 17) |
                                   (1u << 10) | (1u << 7) | (1u << 4);
            // PROBE ONLY (see v1): runtime n_dim shrink for rate measurements.
            const uint32_t probe_n8 = ((uint32_t)hints >> 24) & 0x3fu;
            const uint32_t idesc_p = probe_n8
                ? ((idesc & ~(0x3fu << 17)) | (probe_n8 << 17)) : idesc;
            uint32_t st = 0, ph = 0;
            int acc_it = 0;
            int e, mb, nb, mblk, mmv_unused;
            while (sched_next(sched, e, mb, nb, mblk, mmv_unused)) {
                const int acc = acc_it & 1;
                const uint32_t aphase = (uint32_t)((acc_it >> 1) & 1);
                acc_it++;
                mbar_wait(tmem_empty + acc, aphase ^ 1);
                tc_fence_after_sync();
                const uint32_t tc = tmem_base + (uint32_t)(acc * BN);
                #pragma unroll 1
                for (int kb = 0; kb < num_k; ++kb) {
                    mbar_wait(full_bar + st, ph);
                    tc_fence_after_sync();
                    const uint64_t ad = ad0 + (uint64_t)((st * A_STAGE) >> 4);
                    const uint64_t bd = bd0 + (uint64_t)((st * B_STAGE) >> 4);
                    constexpr uint64_t kAtomA = (uint64_t)((BMH * 64 * 2) >> 4);
                    constexpr uint64_t kAtomB = (uint64_t)((BH * 64 * 2) >> 4);
                    #pragma unroll
                    for (int i = 0; i < BK / 16; ++i)
                        umma_f16_2sm(tc,
                                     ad + (i / 4) * kAtomA + 2 * (i % 4),
                                     bd + (i / 4) * kAtomB + 2 * (i % 4),
                                     idesc_p, (uint32_t)((kb | i) != 0));
                    umma_commit2_mc(empty_bar + st);
                    if (kb == num_k - 1) umma_commit2_mc(tmem_full + acc);
                    if (++st == STAGES) { st = 0; ph ^= 1; }
                }
            }
        }
    } else if (warp >= 4) {
        // Epilogue warpgroup on BOTH CTAs: own 128 rows x BN cols
        const int ew = warp - 4;
        const int row = ew * 32 + lane;
        const int row_swz = row & 7;
        const uint64_t hintD = (hints & 2) ? policy_evict_first() : 0;
        int acc_it = 0;
        int ts = 0;
        int e, mb, nb, mblk, mmv;
        // hints bit5: rank-1 skips all-padding tiles (mm <= mb*256+128) - its
        // 128 rows hold no masked data, so the TMA store and the tmem drain
        // feeding it are pure waste. Skipping also tightens the metadata
        // region contract ("all remaining logical ... storage unchanged")
        // and frees the accumulator earlier for the pair's cd2 pipeline.
        const bool pskip = (hints & 32) && (rank == 1);
        while (sched_next(sched, e, mb, nb, mblk, mmv)) {
            const int acc = acc_it & 1;
            const uint32_t aphase = (uint32_t)((acc_it >> 1) & 1);
            acc_it++;
            mbar_wait(tmem_full + acc, aphase);
            tc_fence_after_sync();
            if (pskip && mmv <= mb * BM2 + BMH) {
                tc_fence_before_sync();
                mbar_arrive_rank0(tmem_empty + acc);
                continue;
            }
            const int m_out = e * CAP + mb * BM2 + (int)rank * BMH;
            const int n_base = nb * BN;
            const uint32_t tcol = tmem_base + (uint32_t)(acc * BN);
            char* cd = sCD + ts * (BMH * 64 * 2);
            #pragma unroll 1
            for (int sb = 0; sb < NCHUNK; ++sb) {
                // BN%64 != 0: final chunk is TAILW cols, staged linearly and
                // stored through the SWIZZLE_NONE box{TAILW,128} descriptor.
                const bool is_tail = (TAILW != 0) && (sb == SB);
                if (ew == 0) tma_store_wait<CD_STAGES - 1>();
                bar_sync(1, 128);
                uint32_t vals[8][8];
                if (!is_tail) {
                    #pragma unroll
                    for (int ld = 0; ld < 8; ++ld)
                        tmem_ld_x8(tcol + (uint32_t)(sb * 64 + ld * 8), vals[ld]);
                } else {
                    #pragma unroll
                    for (int ld = 0; ld < TAILW / 8; ++ld)
                        tmem_ld_x8(tcol + (uint32_t)(sb * 64 + ld * 8), vals[ld]);
                }
                tmem_ld_wait();
                if (sb == NCHUNK - 1) {
                    tc_fence_before_sync();
                    mbar_arrive_rank0(tmem_empty + acc);
                }
                if (!is_tail) {
                    #pragma unroll
                    for (int ld = 0; ld < 8; ++ld) {
                        uint4 packed;
                        __nv_bfloat162 p0 = __floats2bfloat162_rn(__uint_as_float(vals[ld][0]), __uint_as_float(vals[ld][1]));
                        __nv_bfloat162 p1 = __floats2bfloat162_rn(__uint_as_float(vals[ld][2]), __uint_as_float(vals[ld][3]));
                        __nv_bfloat162 p2 = __floats2bfloat162_rn(__uint_as_float(vals[ld][4]), __uint_as_float(vals[ld][5]));
                        __nv_bfloat162 p3 = __floats2bfloat162_rn(__uint_as_float(vals[ld][6]), __uint_as_float(vals[ld][7]));
                        packed.x = *reinterpret_cast<uint32_t*>(&p0);
                        packed.y = *reinterpret_cast<uint32_t*>(&p1);
                        packed.z = *reinterpret_cast<uint32_t*>(&p2);
                        packed.w = *reinterpret_cast<uint32_t*>(&p3);
                        *reinterpret_cast<uint4*>(cd + row * 128 + ((ld ^ row_swz) * 16)) = packed;
                    }
                } else {
                    #pragma unroll
                    for (int ld = 0; ld < TAILW / 8; ++ld) {
                        uint4 packed;
                        __nv_bfloat162 p0 = __floats2bfloat162_rn(__uint_as_float(vals[ld][0]), __uint_as_float(vals[ld][1]));
                        __nv_bfloat162 p1 = __floats2bfloat162_rn(__uint_as_float(vals[ld][2]), __uint_as_float(vals[ld][3]));
                        __nv_bfloat162 p2 = __floats2bfloat162_rn(__uint_as_float(vals[ld][4]), __uint_as_float(vals[ld][5]));
                        __nv_bfloat162 p3 = __floats2bfloat162_rn(__uint_as_float(vals[ld][6]), __uint_as_float(vals[ld][7]));
                        packed.x = *reinterpret_cast<uint32_t*>(&p0);
                        packed.y = *reinterpret_cast<uint32_t*>(&p1);
                        packed.z = *reinterpret_cast<uint32_t*>(&p2);
                        packed.w = *reinterpret_cast<uint32_t*>(&p3);
                        *reinterpret_cast<uint4*>(cd + row * (TAILW * 2) + ld * 16) = packed;
                    }
                }
                tma_store_fence();
                bar_sync(1, 128);
                if (ew == 0 && lane == 0) {
                    const CUtensorMap* dm = is_tail ? &tmapD32 : &tmapD;
                    if (hintD) tma_store_2d_h(dm, cd, n_base + sb * 64, m_out, hintD);
                    else       tma_store_2d(dm, cd, n_base + sb * 64, m_out);
                    tma_store_commit();
                }
                ts = (ts + 1 == CD_STAGES) ? 0 : ts + 1;
                cd = sCD + ts * (BMH * 64 * 2);
            }
        }
        if (ew == 0 && lane == 0) tma_store_wait<0>();
    }

    __syncthreads();
    cluster_sync();
    if (warp == 2) {
        tmem_dealloc2(tmem_base, TMEM_COLS);
    }
}

namespace {

struct DescKey {
    const void* a; const void* b; const void* d;
    int E, CAP, N, K, bn;
    bool operator<(const DescKey& o) const {
        if (a != o.a) return a < o.a;
        if (b != o.b) return b < o.b;
        if (d != o.d) return d < o.d;
        if (E != o.E) return E < o.E;
        if (CAP != o.CAP) return CAP < o.CAP;
        if (N != o.N) return N < o.N;
        if (K != o.K) return K < o.K;
        return bn < o.bn;
    }
};

struct DescVal { CUtensorMap a, b, d, d32; };

std::map<DescKey, DescVal> g_desc_cache;

inline CUtensorMapL2promotion l2_promo_env() {
    static int v = -2;
    if (v == -2) {
        const char* e = getenv("GGM_L2PROMO");
        v = e ? atoi(e) : 2;
    }
    return v == 0 ? CU_TENSOR_MAP_L2_PROMOTION_NONE :
           v == 1 ? CU_TENSOR_MAP_L2_PROMOTION_L2_64B :
           v == 3 ? CU_TENSOR_MAP_L2_PROMOTION_L2_128B :
                    CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
}

int encode_2d(CUtensorMap* out, const void* ptr, uint64_t inner, uint64_t outer,
              uint64_t outer_stride_bytes, uint32_t box_inner, uint32_t box_outer,
              CUtensorMapSwizzle swz = CU_TENSOR_MAP_SWIZZLE_128B) {
    const cuuint64_t dims[2] = {inner, outer};
    const cuuint64_t strides[1] = {outer_stride_bytes};
    const cuuint32_t box[2] = {box_inner, box_outer};
    const cuuint32_t elem[2] = {1, 1};
    CUresult r = cuTensorMapEncodeTiled(
        out, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr),
        dims, strides, box, elem,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swz,
        l2_promo_env(), CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return (int)r;
}

template <int BN, int STAGES, int CD_STAGES, int BK, int EP = 0>
int launch_one(const DescVal& v, const int* mm, int E, int CAP, int N, int K,
               int num_sms, int hints, int pdl, int grid_ovr, cudaStream_t stream) {
    constexpr int BMH = 128;
    constexpr int SMEM = STAGES * (BMH * BK * 2 + (BN / 2) * BK * 2) + CD_STAGES * BMH * 64 * 2
                       + (2 * STAGES + 4) * 8 + 8 + 128;
    static bool inited = false;
    if (!inited) {
        cudaError_t e = cudaFuncSetAttribute((const void*)ggm3_kernel<BN, STAGES, CD_STAGES, BK, EP>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        if (e != cudaSuccess) return (int)e;
        inited = true;
    }
    cudaLaunchConfig_t cfg = {};
    cudaLaunchAttribute attrs[3];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    if (pdl) {
        attrs[cfg.numAttrs].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[cfg.numAttrs].val.programmaticStreamSerializationAllowed = 1;
        cfg.numAttrs += 1;
    }
    static int policy = -2;
    if (policy == -2) {
        const char* pv = getenv("GGM_CLUSTER_POLICY");
        policy = pv ? atoi(pv) : -1;
    }
    if (policy >= 0) {
        attrs[cfg.numAttrs].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
        attrs[cfg.numAttrs].val.clusterSchedulingPolicyPreference =
            policy == 0 ? cudaClusterSchedulingPolicyDefault :
            policy == 1 ? cudaClusterSchedulingPolicySpread :
                          cudaClusterSchedulingPolicyLoadBalancing;
        cfg.numAttrs += 1;
    }
    static int grid_override = -2;
    if (grid_override == -2) {
        const char* gv = getenv("GGM_NCLUSTERS");
        grid_override = gv ? atoi(gv) : 0;
    }
    // Per-call grid (clusters) beats the process-static env knob; any even
    // grid >= 2 is exact for the persistent sched walk (wave balancing).
    int grid = num_sms;
    if (grid_ovr > 0) grid = (grid_ovr * 2) & ~1;
    else if (grid_override > 0) grid = (grid_override * 2) & ~1;
    cfg.gridDim = dim3((unsigned)grid, 1, 1);
    cfg.blockDim = dim3(256, 1, 1);
    cfg.dynamicSmemBytes = SMEM;
    cfg.stream = stream;
    cudaError_t rc = cudaLaunchKernelEx(&cfg, ggm3_kernel<BN, STAGES, CD_STAGES, BK, EP>,
                                        v.a, v.b, v.d, v.d32, mm, E, CAP, N, K, hints);
    return (int)rc;
}

}  // namespace

extern "C" int ggm3_launch(const void* A, const void* B, void* D, const int* masked_m,
                           int E, int CAP, int N, int K, int bn_ovr, int hints,
                           int variant, int pdl, int cap128, int grid_ovr,
                           cudaStream_t stream) {
    static int num_sms = -1;
    if (num_sms < 0) {
        if (cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess)
            return -10;
        num_sms &= ~1;  // cluster of 2 requires an even grid
        if (num_sms < 2) return -11;
    }
    const int bn = bn_ovr > 0 ? bn_ovr : (N % 256 == 0 ? 256 : 128);
    if (K % 64 != 0 || N % bn != 0 || (bn != 128 && bn != 224 && bn != 256))
        return -2;
    // CAP%256==0: rank-1 rows may hold real data. CAP%128==0 (gated): every
    // rank-1 half-tile is capacity padding because mm <= CAP < mb*256+256 for
    // the last block; force both rank-1 skip hints so its A loads and D stores
    // never touch the next expert's rows (bit32 is a storage-safety
    // requirement here, bit64 trims the expect_tx accounting to match).
    if (CAP % 256 != 0) {
        if (!cap128 || (CAP % 128 != 0)) return -2;
        hints |= 96;
    }
    DescKey key{A, B, D, E, CAP, N, K, bn};
    auto it = g_desc_cache.find(key);
    if (it == g_desc_cache.end()) {
        if (g_desc_cache.size() > 64) g_desc_cache.clear();
        DescVal v;
        int rc = encode_2d(&v.a, A, (uint64_t)K, (uint64_t)E * CAP, (uint64_t)K * 2, 64, 128);
        if (rc == 0) rc = encode_2d(&v.b, B, (uint64_t)K, (uint64_t)E * N, (uint64_t)K * 2, 64, (uint32_t)(bn / 2));
        if (rc == 0) rc = encode_2d(&v.d, D, (uint64_t)N, (uint64_t)E * CAP, (uint64_t)N * 2, 64, 128);
        if (rc == 0) {
            // Tail-store descriptor for BN%64!=0 (linear-staged 32-col chunks).
            if (bn % 64 != 0)
                rc = encode_2d(&v.d32, D, (uint64_t)N, (uint64_t)E * CAP,
                               (uint64_t)N * 2, 32, 128, CU_TENSOR_MAP_SWIZZLE_NONE);
            else
                __builtin_memset(&v.d32, 0, sizeof(v.d32));
        }
        if (rc != 0) return rc;
        it = g_desc_cache.emplace(key, v).first;
    }
    int rc;
    if (bn == 256) {
        switch (variant) {
            case 1:  rc = launch_one<256, 5, 3, 64>(it->second, masked_m, E, CAP, N, K, num_sms, hints, pdl, grid_ovr, stream); break;
            case 2:  rc = launch_one<256, 4, 4, 64>(it->second, masked_m, E, CAP, N, K, num_sms, hints, pdl, grid_ovr, stream); break;
            case 3:  if (K % 128 != 0) return -2;
                     rc = launch_one<256, 3, 2, 128>(it->second, masked_m, E, CAP, N, K, num_sms, hints, pdl, grid_ovr, stream); break;
            case 4:  rc = launch_one<256, 6, 2, 64, 1>(it->second, masked_m, E, CAP, N, K, num_sms, hints, pdl, grid_ovr, stream); break;  // EP: probe-before-alloc empty path
            default: rc = launch_one<256, 6, 2, 64>(it->second, masked_m, E, CAP, N, K, num_sms, hints, pdl, grid_ovr, stream); break;
        }
    } else if (bn == 224) {
        // N=7168 class: 32 n-tiles; wave makespan on c7 matches BN=256 while
        // the m256n224k16 UMMA measures ~3.6% faster per tile (probe job
        // 20261024T040011Z). 6 stages x 30KB + 2x16KB CD = 217KB smem.
        rc = launch_one<224, 6, 2, 64>(it->second, masked_m, E, CAP, N, K, num_sms, hints, pdl, grid_ovr, stream);
    } else {
        rc = launch_one<128, 8, 2, 64>(it->second, masked_m, E, CAP, N, K, num_sms, hints, pdl, grid_ovr, stream);
    }
    if (rc != 0) return rc;
    return (int)cudaGetLastError();
}

// ---- host single-node CUDA graph launch cache (GGM_GRAPH knob) ----
// Replays an identical launch (same pointers/dims/flags) via cudaGraphLaunch
// to cut host-side launch cost. Semantics preserved: every replay re-executes
// the same kernel reading live buffer contents, ordered on the caller's
// current stream; only launch configuration is cached, never data. Capture is
// host-only (private non-blocking stream, ThreadLocal mode) and attempted on
// the second sighting of a key; any failure or an externally capturing stream
// falls back to direct launch for that key (state 3, never retried).
struct GKey {
    const void* a; const void* b; void* d; const void* m;
    int e, cap, n, k, bn, hints, variant, pdl, cap128, grid;
    bool operator==(const GKey& o) const {
        return a == o.a && b == o.b && d == o.d && m == o.m && e == o.e
            && cap == o.cap && n == o.n && k == o.k && bn == o.bn
            && hints == o.hints && variant == o.variant && pdl == o.pdl
            && cap128 == o.cap128 && grid == o.grid;
    }
};
struct GEntry {
    GKey key{nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    cudaGraphExec_t exec = nullptr;
    int state = 0;   // 0 unused, 1 seen once, 2 graph ready, 3 direct-only
    bool used = false;
};
GEntry g_gcache[64];
cudaStream_t g_capstream = nullptr;
unsigned long long g_stat_built = 0, g_stat_dead = 0, g_stat_replay = 0, g_stat_direct = 0;

inline uint64_t gmix(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33; return x;
}
inline int g_slot(const GKey& k) {
    uint64_t h = gmix((uint64_t)k.a) ^ (gmix((uint64_t)k.b) >> 7)
               ^ (gmix((uint64_t)k.d) << 5) ^ (gmix((uint64_t)k.m) >> 3)
               ^ gmix(((uint64_t)(unsigned)k.e << 48) | ((uint64_t)(unsigned)k.cap << 32)
                      | ((uint64_t)(unsigned)k.n << 16) | (uint64_t)(unsigned)k.k)
               ^ gmix(((uint64_t)(unsigned)k.bn << 24) | ((uint64_t)(unsigned)k.hints << 16)
                      | ((uint64_t)(unsigned)k.variant << 8) | (uint64_t)(unsigned)k.pdl)
               ^ (gmix((uint64_t)(unsigned)k.cap128) << 11)
               ^ (gmix((uint64_t)(unsigned)k.grid) << 7);
    return (int)(h & 63ULL);
}

extern "C" void ggm3_graph_stats(unsigned long long* built, unsigned long long* dead,
                                 unsigned long long* replay, unsigned long long* direct) {
    if (built) *built = g_stat_built;
    if (dead) *dead = g_stat_dead;
    if (replay) *replay = g_stat_replay;
    if (direct) *direct = g_stat_direct;
}

extern "C" int ggm3_launch_g(const void* A, const void* B, void* D, const int* masked_m,
                             int E, int CAP, int N, int K, int bn_ovr, int hints,
                             int variant, int pdl, int cap128, int grid_ovr,
                             cudaStream_t stream) {
    const int bn = bn_ovr > 0 ? bn_ovr : (N % 256 == 0 ? 256 : 128);
    GKey key{A, B, D, masked_m, E, CAP, N, K, bn, hints, variant, pdl, cap128, grid_ovr};
    GEntry& en = g_gcache[g_slot(key)];
    if (!(en.used && en.key == key)) {
        // first sighting (or slot reuse): evict stale exec, direct launch
        if (en.exec) { cudaGraphExecDestroy(en.exec); en.exec = nullptr; }
        en.key = key; en.used = true; en.state = 1;
        int rc = ggm3_launch(A, B, D, masked_m, E, CAP, N, K, bn_ovr, hints, variant, pdl, cap128, grid_ovr, stream);
        if (rc == -2) en.state = 3;  // geometry unsupported: never graph this key
        g_stat_direct++;
        return rc;
    }
    if (en.state == 2) {
        if (cudaGraphLaunch(en.exec, stream) == cudaSuccess) { g_stat_replay++; return 0; }
        cudaGetLastError();
        cudaGraphExecDestroy(en.exec); en.exec = nullptr; en.state = 3; g_stat_dead++;
        g_stat_direct++;
        return ggm3_launch(A, B, D, masked_m, E, CAP, N, K, bn_ovr, hints, variant, pdl, cap128, grid_ovr, stream);
    }
    if (en.state == 3) {
        g_stat_direct++;
        return ggm3_launch(A, B, D, masked_m, E, CAP, N, K, bn_ovr, hints, variant, pdl, cap128, grid_ovr, stream);
    }
    // state 1: second sighting -> try host-only capture, then replay for this call.
    bool ok = true;
    cudaStreamCaptureStatus capst = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capst) != cudaSuccess
        || capst != cudaStreamCaptureStatusNone) ok = false;  // engine is capturing: don't nest
    if (ok && !g_capstream)
        ok = (cudaStreamCreateWithFlags(&g_capstream, cudaStreamNonBlocking) == cudaSuccess);
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    if (ok) {
        if (cudaStreamBeginCapture(g_capstream, cudaStreamCaptureModeThreadLocal) == cudaSuccess) {
            int lrc = ggm3_launch(A, B, D, masked_m, E, CAP, N, K, bn_ovr, hints, variant, pdl,
                                  cap128, grid_ovr, g_capstream);
            cudaError_t erc = cudaStreamEndCapture(g_capstream, &graph);
            ok = (lrc == 0 && erc == cudaSuccess && graph != nullptr);
            if (ok) ok = (cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess && exec != nullptr);
            if (graph) cudaGraphDestroy(graph);
            if (!ok && exec) { cudaGraphExecDestroy(exec); exec = nullptr; }
        } else {
            ok = false;
        }
        if (!ok) cudaGetLastError();
    }
    if (ok && exec) {
        en.exec = exec; en.state = 2; g_stat_built++;
        cudaGraphUpload(exec, stream); cudaGetLastError();  // one-time pre-upload; harmless if it fails
        if (cudaGraphLaunch(exec, stream) == cudaSuccess) { g_stat_replay++; return 0; }
        cudaGetLastError();
        cudaGraphExecDestroy(en.exec); en.exec = nullptr; en.state = 3; g_stat_dead++;
    } else {
        en.state = 3; g_stat_dead++;
    }
    g_stat_direct++;
    return ggm3_launch(A, B, D, masked_m, E, CAP, N, K, bn_ovr, hints, variant, pdl, cap128, grid_ovr, stream);
}

}  // namespace ggm3
'''


def _find_nvcc():
    for cand in ('/usr/local/cuda/bin/nvcc',):
        if os.path.exists(cand):
            return cand
    import shutil
    found = shutil.which('nvcc')
    if found:
        return found
    raise RuntimeError('nvcc not found')


FASTCALL_C = r"""
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <dlfcn.h>

typedef int (*g3_fn)(const void*, const void*, void*, const int*,
                     int, int, int, int, int, int, int, int, int, int, void*);
typedef int (*g2_fn)(const void*, const void*, void*, const int*,
                     int, int, int, int, int, int, int, int, void*);
typedef void (*gstat_fn)(unsigned long long*, unsigned long long*,
                         unsigned long long*, unsigned long long*);
static g3_fn f3 = 0;
static g3_fn f3g = 0;   /* graph-cached v2 launch (optional symbol) */
static gstat_fn f3s = 0;  /* POSIX fstat() name is taken */
static g2_fn f2 = 0;

static PyObject* bind(PyObject* self, PyObject* args) {
    const char* p1 = "";
    const char* p2 = "";
    if (!PyArg_ParseTuple(args, "|ss", &p1, &p2)) return NULL;
    if (p1[0]) {
        void* h = dlopen(p1, RTLD_NOW | RTLD_LOCAL);
        if (!h) { PyErr_SetString(PyExc_RuntimeError, dlerror()); return NULL; }
        f2 = (g2_fn)dlsym(h, "ggm2_launch");
    }
    if (p2[0]) {
        void* h = dlopen(p2, RTLD_NOW | RTLD_LOCAL);
        if (!h) { PyErr_SetString(PyExc_RuntimeError, dlerror()); return NULL; }
        f3 = (g3_fn)dlsym(h, "ggm3_launch");
        f3g = (g3_fn)dlsym(h, "ggm3_launch_g");
        f3s = (gstat_fn)dlsym(h, "ggm3_graph_stats");
    }
    Py_RETURN_NONE;
}

/* launch3(a, b, d, mm, E, CAP, N, K, em, flags, stream) -> rc ; combined
 * v2/v1 selector. flags bit21: use graph-cached launcher when available
 * (replay reads live buffers; falls back to direct launch internally on any
 * doubt). v1-preference rule (expected_m is the frozen ABI host scalar, an
 * estimate - both kernels are exact for any masked_m, this only picks the
 * faster one): when the expected fill leaves v2's 256-row blocks >50% padded
 * AND the shape is not DRAM-bound enough to absorb the padded tensor work
 * (n<=3072, k<=2048), the 1-CTA BM128 kernel halves padded UMMA work.
 * Evidence: 8 same-job dev A/B runs - c6 (em=112,n1536,k2048,E64) v1 wins
 * every pair (-2.5% avg); c2 (k=4096) and c7 (n=7168) falsify padding-only
 * rules (v2 wins there); c3/c8 (em=224) stay v2/cannot be split robustly. */
static PyObject* launch3(PyObject* self, PyObject* const* a, Py_ssize_t n) {
    if (n != 11 || (!f3 && !f2)) {
        PyErr_SetString(PyExc_TypeError, "launch3(a,b,d,mm,E,CAP,N,K,em,flags,stream)");
        return NULL;
    }
    unsigned long long A = PyLong_AsUnsignedLongLongMask(a[0]);
    unsigned long long B = PyLong_AsUnsignedLongLongMask(a[1]);
    unsigned long long D = PyLong_AsUnsignedLongLongMask(a[2]);
    unsigned long long M = PyLong_AsUnsignedLongLongMask(a[3]);
    long E = PyLong_AsLong(a[4]);
    long CAP = PyLong_AsLong(a[5]);
    long N = PyLong_AsLong(a[6]);
    long K = PyLong_AsLong(a[7]);
    long EM = PyLong_AsLong(a[8]);
    unsigned long long fl = PyLong_AsUnsignedLongLongMask(a[9]);
    unsigned long long S = PyLong_AsUnsignedLongLongMask(a[10]);
    if (PyErr_Occurred()) return NULL;
    const int v1pref = ((EM & 255L) > 0) && ((EM & 255L) <= 128)
                       && (N <= 3072) && (K <= 2048);
    const int cap128 = (int)((fl >> 22) & 1ULL);
    const int pdl = (int)((fl >> 20) & 1ULL);
    const int grid = (int)((fl >> 40) & 0xffffULL);  /* v2: clusters, v1: CTAs */
    int rc;
    if (!v1pref && f3 && ((CAP & 255L) == 0 || (cap128 && (CAP & 127L) == 0))) {
        g3_fn fn = (((fl >> 21) & 1ULL) && f3g) ? f3g : f3;
        rc = fn((const void*)A, (const void*)B, (void*)D, (const int*)M,
                (int)E, (int)CAP, (int)N, (int)K,
                (int)(fl & 0xff), (int)((fl >> 8) & 0xff),
                (int)((fl >> 16) & 0xf), pdl, cap128, grid, (void*)S);
    } else if (f2 && (CAP & 127L) == 0) {
        /* bits 24-31: v1-path hints (may differ from v2's bits 8-15; python
         * defaults them equal, so behavior is unchanged unless GGM_HINTS_V1
         * is set -- e.g. static B-evict-first for the v1pref/odd-CAP shapes
         * where adaptive bit8 under-fires on BM128 blocks). */
        rc = f2((const void*)A, (const void*)B, (void*)D, (const int*)M,
                (int)E, (int)CAP, (int)N, (int)K,
                (int)(fl & 0xff), (int)((fl >> 24) & 0xff), pdl, grid, (void*)S);
    } else {
        rc = -2;
    }
    return PyLong_FromLong(rc);
}

/* launch2(a, b, d, mm, E, CAP, N, K, flags, stream) -> rc ; v1 path */
static PyObject* launch2(PyObject* self, PyObject* const* a, Py_ssize_t n) {
    if (n != 10 || !f2) {
        PyErr_SetString(PyExc_TypeError, "launch2(a,b,d,mm,E,CAP,N,K,flags,stream)");
        return NULL;
    }
    unsigned long long A = PyLong_AsUnsignedLongLongMask(a[0]);
    unsigned long long B = PyLong_AsUnsignedLongLongMask(a[1]);
    unsigned long long D = PyLong_AsUnsignedLongLongMask(a[2]);
    unsigned long long M = PyLong_AsUnsignedLongLongMask(a[3]);
    long E = PyLong_AsLong(a[4]);
    long CAP = PyLong_AsLong(a[5]);
    long N = PyLong_AsLong(a[6]);
    long K = PyLong_AsLong(a[7]);
    unsigned long long fl = PyLong_AsUnsignedLongLongMask(a[8]);
    unsigned long long S = PyLong_AsUnsignedLongLongMask(a[9]);
    if (PyErr_Occurred()) return NULL;
    int rc = f2((const void*)A, (const void*)B, (void*)D, (const int*)M,
                (int)E, (int)CAP, (int)N, (int)K,
                (int)(fl & 0xff), (int)((fl >> 8) & 0xff),
                (int)((fl >> 20) & 1ULL), (int)((fl >> 40) & 0xffffULL), (void*)S);
    return PyLong_FromLong(rc);
}

static PyObject* graphcap(PyObject* self, PyObject* noarg) {
    (void)noarg;
    return PyLong_FromLong(f3g ? 1 : 0);
}

static PyObject* graphstats(PyObject* self, PyObject* noarg) {
    (void)noarg;
    unsigned long long b = 0, d = 0, r = 0, dd = 0;
    if (f3s) f3s(&b, &d, &r, &dd);
    return Py_BuildValue("(KKKK)", b, d, r, dd);
}

static PyMethodDef methods[] = {
    {"bind", bind, METH_VARARGS, "bind v1/v2 shared objects"},
    {"launch3", (PyCFunction)launch3, METH_FASTCALL, "v2 cluster launch"},
    {"launch2", (PyCFunction)launch2, METH_FASTCALL, "v1 launch"},
    {"graphcap", (PyCFunction)graphcap, METH_NOARGS, "graph launcher available"},
    {"graphstats", (PyCFunction)graphstats, METH_NOARGS, "(built,dead,replay,direct)"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef mod = {PyModuleDef_HEAD_INIT, "ggm_fast", NULL, -1, methods};

PyMODINIT_FUNC PyInit_ggm_fast(void) { return PyModule_Create(&mod); }
"""


_LIB0 = None
_LIB1 = None
_LIB2 = None
_BUILD_LOGS = {}
_LIB_PATHS = {}


def _compile(tag, source, arch_flags, extra=()):
    """arch_flags: list of nvcc arch flags (explicit --generate-code preferred)."""
    import hashlib
    if isinstance(arch_flags, str):
        arch_flags = [f'-arch={arch_flags}']
    key = hashlib.sha256((source + '|' + ' '.join(arch_flags) + '|' + tag + '|rev3').encode()).hexdigest()[:16]
    base = Path(os.environ.get('GGM_BUILD_DIR')
                or os.environ.get('TORCH_EXTENSIONS_DIR')
                or tempfile.gettempdir()) / ('ggm-' + tag + '-' + key)
    base.mkdir(parents=True, exist_ok=True)
    src = base / (tag + '.cu')
    lib = base / (tag + '.so')
    src.write_text(source)
    logf = base / 'build.log'
    if not lib.exists():
        cmd = [_find_nvcc(), '-O3', '-std=c++17', '-shared', '-Xcompiler', '-fPIC',
               *arch_flags, '--ptxas-options=-v', *extra, str(src), '-o', str(lib)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        log = ' '.join(cmd) + '\n' + (proc.stdout or '') + (proc.stderr or '')
        logf.write_text(log)
        if proc.returncode != 0 or not lib.exists():
            raise RuntimeError(f'nvcc build failed ({tag}):\n' + log)
    else:
        log = logf.read_text() if logf.exists() else 'cached'
    _BUILD_LOGS[tag + ':' + ' '.join(arch_flags)] = log
    _LIB_PATHS[tag] = str(lib)
    return ctypes.CDLL(str(lib))


def _arch_parts():
    major, minor = torch.cuda.get_device_capability(0)
    return (os.environ.get('GGM_ARCH') or f'sm_{major}{minor}'), major, minor


def _build_v0():
    global _LIB0
    if _LIB0 is None:
        arch, _, _ = _arch_parts()
        lib = _compile('ggm_v0', CUDA_SOURCE, arch)
        lib.ggm_launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 7 + [ctypes.c_void_p]
        lib.ggm_launch.restype = ctypes.c_int
        _LIB0 = lib
    return _LIB0


def _build_v1():
    """tcgen05 kernel; False when device family lacks tcgen05 or build fails."""
    global _LIB1
    if _LIB1 is None:
        arch, major, minor = _arch_parts()
        if major != 10:
            _LIB1 = False
            _BUILD_LOGS['v1:skip'] = f'tcgen05 path not built for cc {major}.{minor}'
            return _LIB1
        lib = None
        errs = []
        for variant in ('a', 'f'):
            flags = [f'--generate-code=arch=compute_{major}{minor}{variant},'
                     f'code=sm_{major}{minor}{variant}']
            try:
                lib = _compile(f'ggm_v1{variant}', CUDA_V1, flags, extra=['-lcuda'])
                break
            except RuntimeError as ex:
                errs.append(str(ex)[-1500:])
        if lib is None:
            _BUILD_LOGS['v1:error'] = '\n'.join(errs)
            _LIB1 = False
            return _LIB1
        lib.ggm2_launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 8 + [ctypes.c_void_p]
        lib.ggm2_launch.restype = ctypes.c_int
        lib.ggm4_launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
        lib.ggm4_launch.restype = ctypes.c_int
        lib.ggm4_reduce_launch.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
        lib.ggm4_reduce_launch.restype = ctypes.c_int
        _LIB1 = lib
    return _LIB1


def _build_v2():
    """2-CTA cluster tcgen05 kernel; False when unavailable."""
    global _LIB2
    if _LIB2 is None:
        arch, major, minor = _arch_parts()
        if major != 10:
            _LIB2 = False
            _BUILD_LOGS['v2:skip'] = f'cluster tcgen05 path not built for cc {major}.{minor}'
            return _LIB2
        lib = None
        errs = []
        for variant in ('a', 'f'):
            flags = [f'--generate-code=arch=compute_{major}{minor}{variant},'
                     f'code=sm_{major}{minor}{variant}']
            try:
                lib = _compile(f'ggm_v2{variant}', CUDA_V2, flags, extra=['-lcuda'])
                break
            except RuntimeError as ex:
                errs.append(str(ex)[-1500:])
        if lib is None:
            _BUILD_LOGS['v2:error'] = '\n'.join(errs)
            _LIB2 = False
            return _LIB2
        lib.ggm3_launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 10 + [ctypes.c_void_p]
        lib.ggm3_launch.restype = ctypes.c_int
        _LIB2 = lib
    return _LIB2


def _build_fastcall():
    """Optional METH_FASTCALL dispatcher (pure C, no torch headers).

    Cuts the per-call host cost from ~3.6us (13-arg ctypes) to ~0.4us;
    falls back to None (ctypes path) when the toolchain or Python headers
    are unavailable.
    """
    try:
        import sysconfig
        inc = sysconfig.get_paths()['include']
        if not (Path(inc) / 'Python.h').exists():
            return None
        key = __import__('hashlib').sha256(FASTCALL_C.encode()).hexdigest()[:12]
        base = Path(os.environ.get('GGM_BUILD_DIR')
                    or os.environ.get('TORCH_EXTENSIONS_DIR')
                    or tempfile.gettempdir()) / ('ggm-fast-' + key)
        base.mkdir(parents=True, exist_ok=True)
        src = base / 'ggm_fast.c'
        out = base / 'ggm_fast.so'
        src.write_text(FASTCALL_C)
        if not out.exists():
            cc = os.environ.get('CC') or 'cc'
            cmd = [cc, '-O2', '-shared', '-fPIC', '-I' + inc, str(src), '-o', str(out), '-ldl']
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            _BUILD_LOGS['fastcall'] = ' '.join(cmd) + '\n' + (proc.stdout or '') + (proc.stderr or '')
            if proc.returncode != 0 or not out.exists():
                return None
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location('ggm_fast', str(out))
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception as ex:  # never fatal: ctypes path remains
        _BUILD_LOGS['fastcall_error'] = repr(ex)
        return None


def build_log():
    """Expose all ptxas/build logs for diagnostics."""
    _build_v0()
    _build_v1()
    _build_v2()
    return '\n'.join(f'== {k} ==\n{v}' for k, v in _BUILD_LOGS.items())


class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        v0 = _build_v0()
        v1 = _build_v1()
        v2 = _build_v2()
        self._f0 = v0.ggm_launch
        self._f1 = v1.ggm2_launch if v1 else None
        self._f4 = v1.ggm4_launch if v1 else None
        self._f4r = v1.ggm4_reduce_launch if v1 else None
        self._f2 = v2.ggm3_launch if v2 else None
        self._sms = torch.cuda.get_device_properties(0).multi_processor_count
        # 0 = auto, 1 = v0 mma.sync, 2 = v1 tcgen05 1-CTA, 3 = v2 tcgen05 2-CTA cluster
        self._impl = int(os.environ.get('GGM_IMPL', '0'))
        self._bm = int(os.environ.get('GGM_BM', '0'))     # v0 BM override
        self._bn = int(os.environ.get('GGM_BN', '0'))     # v0 BN override
        self._bn2 = int(os.environ.get('GGM_BN2', '0'))   # v1/v2 BN override
        # L2 hint bits: 1 = B evict_first (static), 2 = D store evict_first,
        # 4 = A evict_last (unconditional), 8 = device-adaptive B evict_first
        # (reuse==1 experts), 16 = SELECTIVE A evict_last (device gate:
        # E*CAP*K*2 <= 100MB, i.e. whole-A L2-resident; avoids the case9/5
        # thrash measured for unconditional bit4), 32 = rank-1 padding
        # D-store skip (v2), 64 = rank-1 padding A-load skip (v2; trimmed
        # expect_tx). Default v2 = 120 (adaptive B-ef + selective A-el + both
        # skips), v1 = 113 (static B-ef + selective A-el; v1 ignores 32/64).
        # Evidence: skips (runs 20261021T211859Z/222716Z, arm-a); A-el raw
        # +1.03% zero-regression (J1b dv_523459623739), replicated pieces
        # case2 +1.64 x2 / case3 +4.15 x2 (J1c dv_f136a02d4f90, J1d
        # dv_2444676661f5); v1 static-B-ef case6 +2.81..+3.17 (J1c H5, J1d
        # V1S); COMBO case6 +5.71 additive, bench geomean +1.38% (J1d);
        # selective gate protects case9 (-0.5..-0.7% with raw bit4).
        self._hints = int(os.environ.get('GGM_HINTS', '120'))
        # v1-path hints (packed in flags bits 24-31). Static B-ef (bit1)
        # because adaptive bit8 under-fires on BM128 blocks (mask-256
        # experts have mblk128=2). Set GGM_HINTS_V1=-1 to mirror GGM_HINTS.
        self._hints_v1 = int(os.environ.get('GGM_HINTS_V1', '113'))
        if self._hints_v1 < 0:
            self._hints_v1 = self._hints
        # GGM_ACLAMP (ADOPTED default-on; v1 hints bit7 -> AC template
        # instantiation): row-clamped A loads for partial m-blocks;
        # removes padding-row A over-read. Evidence: c0 med -2.5% x3 arms
        # (dv_ce7b4ff831f1/dv_4b860bd7c7fd/dv_6877aa580a49), all-pass
        # correctness, domain zero-regression arithmetic; 8/10 fixed cases
        # byte-identical (v2 untouched). Set GGM_ACLAMP=0 to disable.
        self._aclamp = int(os.environ.get('GGM_ACLAMP', '1')) & 1
        if self._aclamp:
            self._hints_v1 |= 0x80
        # v2 pipeline variants: 0 = S6/cd2 (default), 1 = S5/cd3, 2 = S4/cd4,
        # 3 = S3/BK128/cd2, 4 = S6/cd2 + EP (empty-path: probe-before-alloc,
        # all-empty launches skip tmem alloc/dealloc entirely)
        self._var = int(os.environ.get('GGM_VAR', '0'))
        # PDL: overlap kernel launch/init with the previous stream kernel
        self._pdl = int(os.environ.get('GGM_PDL', '0'))
        # Host graph-replay launcher (bit 21 of packed flags; v2 fast path
        # only). Replays the identical launch via cudaGraphLaunch to cut CPU
        # launch cost; kernel still recomputes from live buffers every call.
        self._graph = int(os.environ.get('GGM_GRAPH', '0'))
        # v2 on CAP%128!=0 shapes (bit 22 of packed flags): rank-1 half-tiles
        # are pure capacity padding there; the launcher force-enables both
        # rank-1 skip hints (A-load and D-store) so no cross-expert access
        # happens. Experimental gate; default off = certified behavior.
        self._cap128 = int(os.environ.get('GGM_CAP128', '0'))
        # PROBE ONLY (timing experiments, forced-impl slow path): runtime
        # n_dim value (N>>3) smuggled in hints bits 24-29 to measure UMMA
        # N-rate scaling. Wrong outputs when nonzero; never used by dispatch.
        self._probe = int(os.environ.get('GGM_PROBE', '0')) & 0x3f
        # Persistent-grid wave balancing: 0 = all SMs (certified default).
        # v1 path: value is a CTA count; v2 path: value is a CLUSTER count
        # (grid CTAs = 2*clusters). Any positive grid is numerically exact;
        # it only changes how tiles spread over the persistent walk.
        self._grid = int(os.environ.get('GGM_GRID', '0')) & 0xffff
        self._ws = None  # split-K fp32 workspace (impl 4, dev experiment)
        try:
            self._gs = torch._C._cuda_getCurrentRawStream
        except AttributeError:
            self._gs = lambda idx: torch.cuda.current_stream().cuda_stream
        self._stream_fn = self._gs  # dev-harness compatibility
        self._flags3 = ((self._bn2 & 0xff) | ((self._hints & 0xff) << 8)
                        | ((self._var & 0xf) << 16) | ((self._pdl & 1) << 20)
                        | ((self._graph & 1) << 21)
                        | ((self._cap128 & 1) << 22)
                        | ((self._hints_v1 & 0xff) << 24)
                        | (self._grid << 40))
        self._flags2 = ((self._bn2 & 0xff) | ((self._hints_v1 & 0xff) << 8)
                        | ((self._pdl & 1) << 20) | (self._grid << 40))
        # Auto per-shape v2 tuning (GGM_AUTO, default on). Three replicated
        # same-job rules keyed only on (E, CAP, N, K, expected_m) geometry:
        #  R1 grid balance: many-wave v2 shapes whose estimated last wave is
        #     ragged run on ceil(est/waves) clusters when that leaves >=4
        #     spare (c4: -2.5/-3.4/-2.6% in 3 jobs; never raises waves).
        #  R2 BN=224: n%224==0 shapes where 224's extra n-tiles do not add a
        #     wave, since m256n224k16 UMMA measures ~3.6% faster per tile
        #     (c7: -3.8%; c4/c9 excluded by the wave rule, measured +1.3/+8.9%).
        #  R3 static B evict_first for K<=1536 (c3: -4.1/-4.2% twice; c8 0%).
        # Disabled by any explicit GGM_BN2/GGM_GRID/GGM_HINTS override.
        self._auto = (int(os.environ.get('GGM_AUTO', '1')) != 0
                      and self._impl == 0 and self._bn2 == 0 and self._grid == 0
                      and self._hints == 120 and self._var == 0
                      and self._f2 is not None)
        self._clusters = max(1, self._sms // 2)
        self._autocache = {}
        # Optional METH_FASTCALL dispatcher (host-side only; same GPU launches)
        self._fast = None
        self._l3 = None
        self._l2 = None
        fast = _build_fastcall()
        if fast is not None:
            try:
                v1so = next((v for k, v in _LIB_PATHS.items() if k.startswith('ggm_v1')), '')
                v2so = next((v for k, v in _LIB_PATHS.items() if k.startswith('ggm_v2')), '')
                fast.bind(v1so, v2so)
                self._fast = fast
                if self._f2 is not None:
                    self._l3 = fast.launch3
                    gc = getattr(fast, 'graphcap', None)
                    if self._graph and (gc is None or not gc()):
                        # graph launcher symbol missing: keep direct path only
                        self._graph = 0
                        self._flags3 &= ~(1 << 21)
                if self._f1 is not None:
                    self._l2 = fast.launch2
            except Exception as ex:
                _BUILD_LOGS['fastcall_bind_error'] = repr(ex)
                self._fast = None
        if self._impl == 2 and self._f1 is None:
            raise RuntimeError('v1 forced but unavailable: '
                               + _BUILD_LOGS.get('v1:error', _BUILD_LOGS.get('v1:skip', 'n/a'))[-3000:])
        if self._impl == 3 and self._f2 is None:
            raise RuntimeError('v2 forced but unavailable: '
                               + _BUILD_LOGS.get('v2:error', _BUILD_LOGS.get('v2:skip', 'n/a'))[-3000:])

    def _auto_flags(self, E, cap, n, k, em):
        """Per-shape packed flags for the fast/auto dispatch (cached)."""
        key = (E, cap, n, k, em)
        f = self._autocache.get(key)
        if f is not None:
            return f
        f = self._flags3
        if (em >= 0 and cap % 256 == 0 and k % 64 == 0 and n % 128 == 0
                and not (0 < em % 256 <= 128 and n <= 3072 and k <= 2048)):
            # v2-eligible shape (mirrors the C launch3 gate incl. v1pref)
            bn = 256 if n % 256 == 0 else 128
            hints = self._hints
            grid = 0
            cl = self._clusters
            est_mblk = max(1, (em + 255) // 256)
            if k <= 1536:
                hints |= 1                       # R3
            if bn == 256 and n % 224 == 0:       # R2
                t224 = E * est_mblk * (n // 224)
                t256 = E * est_mblk * (n // 256)
                if -(-t224 // cl) <= -(-t256 // cl):
                    bn = 224
            tiles = E * est_mblk * (n // bn)     # R1
            waves = -(-tiles // cl)
            if em > 256 and waves >= 3:
                g = -(-tiles // waves)
                if 2 <= g <= cl - 4:
                    grid = g
            f = ((bn & 0xff) | ((hints & 0xff) << 8)
                 | ((self._var & 0xf) << 16) | ((self._pdl & 1) << 20)
                 | ((self._graph & 1) << 21) | ((self._cap128 & 1) << 22)
                 | ((self._hints_v1 & 0xff) << 24) | ((grid & 0xffff) << 40))
        elif em >= 0 and cap % 128 == 0 and k % 64 == 0 and n % 128 == 0:
            # R4 (v1-path shapes: v1pref or CAP%256!=0): grid wave-balance.
            # Order-controlled evidence (200-iter, base==base2==73.760):
            # c6 g128 -1.41/-1.56% both orders (g144 -0.04% control), c0 g128
            # -0.53% sandwiched. Guards exclude every measured-loss shape:
            # k8-style per-SM-BW loss (trim <= sms-8), k6-style est-error
            # wave growth (g >= 0.8*sms), single-wave empties (waves>=2),
            # and require >=10% tail waste to be worth trimming at all.
            nb1 = n // 256 if n % 256 == 0 else n // 128
            est = E * max(1, (em + 127) // 128) * nb1
            sms = self._sms
            waves = -(-est // sms)
            if waves >= 2:
                g = -(-est // waves)
                if ((sms * 4) // 5 <= g <= sms - 8
                        and (waves * sms - est) * 10 >= est):
                    f = (self._flags3 & ~(0xffff << 40)) | ((g & 0xffff) << 40)
        if len(self._autocache) > 512:
            self._autocache.clear()
        self._autocache[key] = f
        return f

    def _splitk_gate(self, E, CAP, N, K, em):
        # FAILED ARM (kept gated off; dev dv run 20261021T1037Z): c0 split-K=2
        # measured 183us vs 154.5 baseline - fp32 workspace traffic (+2x E*CAP*N
        # x4B write+read incl. padding rows) and the reduce launch outweigh the
        # wave-quantization gain. Not used by auto dispatch.
        # Low-occupancy small-m class: few tiles, long K, wave quantization.
        if self._f4 is None:
            return False
        if em > 32 or K < 3072 or K % 128 or CAP % 128 or N % 256:
            return False
        tiles = E * max(1, (em + 127) // 128) * (N // 256)
        return 96 <= tiles < 2 * self._sms

    def forward(self, a, b, d, masked_m, expected_m, compiled_dims='nk'):
        # Fast host path (auto dispatch only): METH_FASTCALL launcher, packed
        # flags, minimal attribute traffic. Same kernels/descriptors as the
        # slow path; falls through on any geometry or availability doubt.
        if self._impl == 0 and self._fast is not None:
            sa = a.shape
            cap = sa[1]
            k = sa[2]
            if k % 64 == 0:
                n = b.shape[1]
                if n % 128 == 0:
                    st = self._gs(a.get_device())
                    dp = masked_m.data_ptr()
                    ap = a.data_ptr()
                    bp = b.data_ptr()
                    dpp = d.data_ptr()
                    rc = -2
                    l3 = self._l3
                    if l3 is not None:
                        # combined selector: v1-preference rule lives in C
                        # (2ns) keyed on the frozen expected_m host scalar
                        em = expected_m if type(expected_m) is int else -1
                        flags3 = self._auto_flags(sa[0], cap, n, k, em) if self._auto else self._flags3
                        rc = l3(ap, bp, dpp, dp, sa[0], cap, n, k, em, flags3, st)
                    if rc == -2:
                        l2 = self._l2
                        if l2 is not None and cap % 128 == 0:
                            rc = l2(ap, bp, dpp, dp, sa[0], cap, n, k, self._flags2, st)
                    if rc == 0:
                        return None
                    if rc != -2:
                        raise RuntimeError(f'ggm launch failed rc={rc}')
        return self._slow(a, b, d, masked_m, expected_m, compiled_dims)

    def _slow(self, a, b, d, masked_m, expected_m, compiled_dims='nk'):
        E, CAP, K = a.shape
        N = b.shape[1]
        stream = self._stream_fn(a.get_device())
        dp_a = a.data_ptr()
        dp_b = b.data_ptr()
        dp_d = d.data_ptr()
        dp_m = masked_m.data_ptr()
        rc = None
        geo_v1 = (CAP % 128 == 0) and (K % 64 == 0) and (N % 128 == 0)
        if self._impl == 4 and self._splitk_gate(E, CAP, N, K, int(expected_m)):
            ws = self._ws
            need = (2 * E * CAP, N)
            if ws is None or ws.shape[0] < need[0] or ws.shape[1] != N or ws.device != a.device:
                ws = torch.empty(need, dtype=torch.float32, device=a.device)
                self._ws = ws
            rc = self._f4(dp_a, dp_b, ws.data_ptr(), dp_m, E, CAP, N, K, 2, stream)
            if rc == 0:
                rc = self._f4r(ws.data_ptr(), dp_d, dp_m, E, CAP, N, 2, stream)
            if rc:
                raise RuntimeError(f'ggm splitk failed rc={rc}')
            return None
        geo_v2 = (CAP % 256 == 0) or (self._cap128 and CAP % 128 == 0)
        probe_h = self._probe << 24  # 0 unless GGM_PROBE set (timing experiments)
        if self._impl == 3:
            if self._f2 is not None and geo_v2 and geo_v1:
                rc = self._f2(dp_a, dp_b, dp_d, dp_m, E, CAP, N, K, self._bn2, self._hints | probe_h, self._var, self._pdl, self._cap128, self._grid, stream)
            elif self._f1 is not None and geo_v1:
                rc = self._f1(dp_a, dp_b, dp_d, dp_m, E, CAP, N, K, self._bn2, self._hints | probe_h, self._pdl, self._grid, stream)
        elif self._impl == 2:
            if self._f1 is None or not geo_v1:
                raise RuntimeError('v1 forced but unavailable/unsupported: '
                                   + _BUILD_LOGS.get('v1:error', _BUILD_LOGS.get('v1:skip', 'geometry'))[-2000:])
            rc = self._f1(dp_a, dp_b, dp_d, dp_m, E, CAP, N, K, self._bn2, self._hints | probe_h, self._pdl, self._grid, stream)
        elif self._impl in (0, 4):
            # auto (also the fallback when impl 4's split-K gate rejects):
            # v2 cluster when geometry allows (same-job A/B dv_b80d7f1a33a4:
            # v2n256 geomean 96.9us vs v1n256 101.0us; wins c3/c4/c5/c7/c9 by
            # 7-11%), EXCEPT the v1-preference rule (mirrors launch3 in C):
            # expected fill leaving v2's 256-row blocks >50% padded on shapes
            # not DRAM-bound enough to absorb it (8 same-job A/B runs: c6 v1
            # wins every pair -2.5% avg; c2 k=4096 / c7 n=7168 falsify a
            # padding-only rule). expected_m is an estimate - both kernels are
            # exact for any masked_m; the rule only picks the faster one.
            em = expected_m
            v1pref = (type(em) is int and 0 < em % 256 <= 128
                      and N <= 3072 and K <= 2048)
            if self._f2 is not None and geo_v2 and geo_v1 and not v1pref:
                if self._auto:
                    fl = self._auto_flags(E, CAP, N, K, em if type(em) is int else -1)
                    bn_a = fl & 0xff
                    hints_a = (fl >> 8) & 0xff
                    grid_a = (fl >> 40) & 0xffff
                else:
                    bn_a, hints_a, grid_a = self._bn2, self._hints, self._grid
                rc = self._f2(dp_a, dp_b, dp_d, dp_m, E, CAP, N, K, bn_a, hints_a, self._var, self._pdl, self._cap128, grid_a, stream)
            elif self._f1 is not None and geo_v1:
                if self._auto:
                    grid_v1 = (self._auto_flags(
                        E, CAP, N, K, em if type(em) is int else -1) >> 40) & 0xffff
                else:
                    grid_v1 = self._grid
                rc = self._f1(dp_a, dp_b, dp_d, dp_m, E, CAP, N, K, self._bn2, self._hints_v1, self._pdl, grid_v1, stream)
        if rc is None:
            rc = self._f0(dp_a, dp_b, dp_d, dp_m, E, CAP, N, K,
                          int(expected_m), self._bm, self._bn, stream)
        if rc:
            raise RuntimeError(f'ggm launch failed rc={rc}')
        return None


# One frame for the engine's model(*args, **kwargs): the ABI is a plain call.
Model.__call__ = Model.forward
