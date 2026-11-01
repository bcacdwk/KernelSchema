#pragma once
/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* D13 NVFP4 MLA decode: the code below is the editable algorithm source.
 *
 * Scores use dequantized E2M1 Q/K with UE4M3 scales (one per 16 elements).
 * For each KV tile the softmax role updates its base-2 maximum m and sum l:
 *   m' = max(m, max(score)); alpha = exp2(m - m');
 *   l' = alpha*l + sum(exp2(score - m')).
 * The correction role rescales the accumulated PV result by alpha. The
 * probability tile is rounded/saturated to E4M3 before tensor-core PV;
 * preserve this quantization and the existing normalization order.
 * A seeded sink contributes to the normalizer, with no value contribution.
 *
 * Two CTAs cooperate on a 128-row query tile and a 128-token KV tile.
 * Warp roles in each 512-thread CTA:
 *   0..3 softmax: read score TMEM, update row statistics, publish E4M3 P;
 *   4..7 correction: prepare Q/SF, rescale PV and publish normalization;
 *   8..13 transform: unpack/rearrange raw KV into the MMA V/SF layouts;
 *   14 load: page-table addressing and asynchronous TMA Q/KV loads;
 *   15 MMA: issue QK/PV and commit completion barriers for both CTAs.
 *
 * Shared memory is 189440 bytes: barriers/allocator slot in [0,1024),
 * Q and Q scales in [1024,19456), three raw-KV stages in [19456,105472),
 * two V stages in [105472,171008), two P stages in [171008,187392),
 * then row state, cross-warp max/sum exchange and page IDs. KQ/KO views
 * deliberately alias the raw-KV allocation. *_OFF are byte offsets;
 * *_STRIDE advances pipeline stages, and is not always *_STAGE_BYTES.
 * TMEM uses 480 of 512 allocated columns: scores [0,64), PV [64,320),
 * Q scales [320,352), K scales [352,416), and transformed Q [416,480).
 *
 * Every full/empty barrier is a producer/consumer ownership transfer.
 * A role advances its local stage and flips phase only on ring wrap;
 * phases continue across persistent work items. Keep arrivals, waits and
 * tcgen05 fences paired when changing any role. The two V-half work items
 * are consecutive; all roles consume the same unit_first/work_table plan.
 * Single-split work writes caller O/LSE directly; multi-split work writes
 * BF16 partial O and log2 partial LSE for the separate precise reduce TU.
 */

typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Cake requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) CakeTensorMap { uint64_t opaque[16]; };
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(CakeTensorMap) >= alignof(CUtensorMap), "CakeTensorMap alignment must cover the CUtensorMap CUDA ABI");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 480
#define TMEM_ACC_OFFSET 0
#define TMEM_SFA0_OFFSET 320
#define TMEM_SFA1_OFFSET 328
#define TMEM_SFA2_OFFSET 336
#define TMEM_SFA3_OFFSET 344
#define TMEM_SFB0_OFFSET 352
#define TMEM_SFB1_OFFSET 360
#define TMEM_SFB2_OFFSET 368
#define TMEM_SFB3_OFFSET 376
#define TMEM_SFB4_OFFSET 384
#define TMEM_SFB5_OFFSET 392
#define TMEM_SFB6_OFFSET 400
#define TMEM_SFB7_OFFSET 408
#define TMEM_Q_OFFSET 416
#define NUM_RAW_PIPE_STAGES 3
#define NUM_V_PIPE_STAGES 2
#define NUM_P_PIPE_STAGES 2
#define NUM_S_PIPE_STAGES 2
#define NUM_SF_PIPE_STAGES 2
#define SMEM_Q0_OFF 1024
#define SMEM_Q0_STAGE_BYTES 4096
#define SMEM_Q0_STRIDE 4096
#define SMEM_Q1_OFF 5120
#define SMEM_Q1_STAGE_BYTES 4096
#define SMEM_Q1_STRIDE 4096
#define SMEM_Q2_OFF 9216
#define SMEM_Q2_STAGE_BYTES 4096
#define SMEM_Q2_STRIDE 4096
#define SMEM_Q3_OFF 13312
#define SMEM_Q3_STAGE_BYTES 4096
#define SMEM_Q3_STRIDE 4096
#define SMEM_QS_OFF 17408
#define SMEM_QS_STAGE_BYTES 2048
#define SMEM_QS_STRIDE 2048
#define SMEM_K0H0_OFF 19456
#define SMEM_K0H0_STAGE_BYTES 2048
#define SMEM_K0H0_STRIDE 28672
#define SMEM_K1H0_OFF 23552
#define SMEM_K1H0_STAGE_BYTES 2048
#define SMEM_K1H0_STRIDE 28672
#define SMEM_K2H0_OFF 27648
#define SMEM_K2H0_STAGE_BYTES 2048
#define SMEM_K2H0_STRIDE 28672
#define SMEM_K3H0_OFF 31744
#define SMEM_K3H0_STAGE_BYTES 2048
#define SMEM_K3H0_STRIDE 28672
#define SMEM_K0H1_OFF 21504
#define SMEM_K0H1_STAGE_BYTES 2048
#define SMEM_K0H1_STRIDE 28672
#define SMEM_K1H1_OFF 25600
#define SMEM_K1H1_STAGE_BYTES 2048
#define SMEM_K1H1_STRIDE 28672
#define SMEM_K2H1_OFF 29696
#define SMEM_K2H1_STAGE_BYTES 2048
#define SMEM_K2H1_STRIDE 28672
#define SMEM_K3H1_OFF 33792
#define SMEM_K3H1_STAGE_BYTES 2048
#define SMEM_K3H1_STRIDE 28672
#define SMEM_KQ0H0_OFF 19456
#define SMEM_KQ0H0_STAGE_BYTES 4096
#define SMEM_KQ0H0_STRIDE 28672
#define SMEM_KQ1H0_OFF 23552
#define SMEM_KQ1H0_STAGE_BYTES 4096
#define SMEM_KQ1H0_STRIDE 28672
#define SMEM_KQ2H0_OFF 27648
#define SMEM_KQ2H0_STAGE_BYTES 4096
#define SMEM_KQ2H0_STRIDE 28672
#define SMEM_KQ3H0_OFF 31744
#define SMEM_KQ3H0_STAGE_BYTES 4096
#define SMEM_KQ3H0_STRIDE 28672
#define SMEM_KQ0H1_OFF 21504
#define SMEM_KQ0H1_STAGE_BYTES 4096
#define SMEM_KQ0H1_STRIDE 28672
#define SMEM_KQ1H1_OFF 25600
#define SMEM_KQ1H1_STAGE_BYTES 4096
#define SMEM_KQ1H1_STRIDE 28672
#define SMEM_KQ2H1_OFF 29696
#define SMEM_KQ2H1_STAGE_BYTES 4096
#define SMEM_KQ2H1_STRIDE 28672
#define SMEM_KQ3H1_OFF 33792
#define SMEM_KQ3H1_STAGE_BYTES 4096
#define SMEM_KQ3H1_STRIDE 28672
#define SMEM_KO0H0_OFF 35840
#define SMEM_KO0H0_STAGE_BYTES 2048
#define SMEM_KO0H0_STRIDE 28672
#define SMEM_KO1H0_OFF 39936
#define SMEM_KO1H0_STAGE_BYTES 2048
#define SMEM_KO1H0_STRIDE 28672
#define SMEM_KO0H1_OFF 37888
#define SMEM_KO0H1_STAGE_BYTES 2048
#define SMEM_KO0H1_STRIDE 28672
#define SMEM_KO1H1_OFF 41984
#define SMEM_KO1H1_STAGE_BYTES 2048
#define SMEM_KO1H1_STRIDE 28672
#define SMEM_KS_OFF 44032
#define SMEM_KS_STAGE_BYTES 4096
#define SMEM_KS_STRIDE 28672
#define SMEM_RAW_FLAT_OFF 19456
#define SMEM_RAW_FLAT_STAGE_BYTES 86016
#define SMEM_RAW_FLAT_STRIDE 86016
#define SMEM_V0_OFF 105472
#define SMEM_V0_STAGE_BYTES 16384
#define SMEM_V0_STRIDE 32768
#define SMEM_V1_OFF 121856
#define SMEM_V1_STAGE_BYTES 16384
#define SMEM_V1_STRIDE 32768
#define SMEM_P_OFF 171008
#define SMEM_P_STAGE_BYTES 8192
#define SMEM_P_STRIDE 8192
#define SMEM_ROW_STATE_OFF 187392
#define SMEM_ROW_STATE_STAGE_BYTES 512
#define SMEM_ROW_STATE_STRIDE 512
#define SMEM_XCHG_MAX_OFF 187904
#define SMEM_XCHG_MAX_STAGE_BYTES 512
#define SMEM_XCHG_MAX_STRIDE 512
#define SMEM_XCHG_SUM_OFF 188416
#define SMEM_XCHG_SUM_STAGE_BYTES 512
#define SMEM_XCHG_SUM_STRIDE 512
#define SMEM_PAGES_OFF 188928
#define SMEM_PAGES_STAGE_BYTES 128
#define SMEM_PAGES_STRIDE 128
#define SMEM_TOTAL 189440
#define THREADS 512
#define SKIP_XFORM 0
#define LAZY_RESCALE 1
#define IDENTITY_PAGES 0
#define SKIP_EXP 0
#define SKIP_KPUB 0
#define SKIP_PSTORE 0
#define SKIP_QK 0
#define SKIP_PV 0
#define SKIP_SMATH 0
#define TRACE 0

#include <math_constants.h>

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}

__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(count) : "memory");
}

// CTA-local pipelines have short, resident producer/consumer edges.  Omitting
// suspendTimeHint keeps a miss on the lightweight TRYWAIT retry path; the
// explicit loop still makes this helper blocking until acquire succeeds.
__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

// Source-faithful relaxed CTA wait used only by a typed protocol that does
// not attach the PTX acquire qualifier, such as FA4's interior P-ready edge.

// Exact source ports may request the PTX suspendTimeHint operand explicitly.
// The hint is expressed in nanoseconds and is kept separate from the canonical
// no-hint CTA helper so unrelated schedules retain their existing retry path.

// Exact unqualified CTA wait used by source schedules whose PTX intentionally
// omits the acquire qualifier while retaining a typed suspendTimeHint operand.

union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};

__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}

__device__ __forceinline__ void tmem_st_x32_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x32.b32"
        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8,"
        "  %9, %10, %11, %12, %13, %14, %15, %16,"
        "  %17, %18, %19, %20, %21, %22, %23, %24,"
        "  %25, %26, %27, %28, %29, %30, %31, %32};"
        :: "r"(tmem_addr),
           "f"(src[0]),  "f"(src[1]),  "f"(src[2]),  "f"(src[3]),
           "f"(src[4]),  "f"(src[5]),  "f"(src[6]),  "f"(src[7]),
           "f"(src[8]),  "f"(src[9]),  "f"(src[10]), "f"(src[11]),
           "f"(src[12]), "f"(src[13]), "f"(src[14]), "f"(src[15]),
           "f"(src[16]), "f"(src[17]), "f"(src[18]), "f"(src[19]),
           "f"(src[20]), "f"(src[21]), "f"(src[22]), "f"(src[23]),
           "f"(src[24]), "f"(src[25]), "f"(src[26]), "f"(src[27]),
           "f"(src[28]), "f"(src[29]), "f"(src[30]), "f"(src[31]));
}

__device__ __forceinline__ uint32_t smem_addr(const void* ptr) {
    uint32_t addr;
    asm("{\n\t"
        ".reg .u64 u64addr;\n\t"
        "cvta.to.shared.u64 u64addr, %1;\n\t"
        "cvt.u32.u64 %0, u64addr;\n\t"
        "}\n" : "=r"(addr) : "l"(ptr));
    return addr;
}

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}

__device__ __forceinline__ float row_max_reduce(float2 acc) {
    return max_noftz(acc.x, acc.y);
}

__device__ __forceinline__ float2 ex2_emulation_f32x2_value(float2 value) {
    const float c0 = 1.0f, c1 = 0.695146143436431884765625f;
    const float c2 = 0.227564394474029541015625f, c3 = 0.077119089663028717041015625f;
    const float magic = 12582912.0f;
    float x0 = max_noftz(value.x, -127.0f), x1 = max_noftz(value.y, -127.0f);
    float2 xc2 = make_float2(x0, x1), magic2 = make_float2(magic, magic);
    float2 xr2;
    asm("add.rm.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xr2)
        : "l"(*(unsigned long long*)&xc2), "l"(*(unsigned long long*)&magic2));
    float2 c3_2 = make_float2(c3, c3), c2_2 = make_float2(c2, c2);
    float2 c1_2 = make_float2(c1, c1), c0_2 = make_float2(c0, c0);
    float2 xrb2, xfrac2;
    asm("sub.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xrb2)
        : "l"(*(unsigned long long*)&xr2), "l"(*(unsigned long long*)&magic2));
    asm("sub.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xfrac2)
        : "l"(*(unsigned long long*)&xc2), "l"(*(unsigned long long*)&xrb2));
    float2 poly2;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&c3_2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c2_2));
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&poly2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c1_2));
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&poly2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c0_2));
    int x0r_i, x1r_i, p0_i, p1_i;
    asm("mov.b64 {%0, %1}, %2;" : "=r"(x0r_i), "=r"(x1r_i) : "l"(*(unsigned long long*)&xr2));
    asm("mov.b64 {%0, %1}, %2;" : "=r"(p0_i), "=r"(p1_i) : "l"(*(unsigned long long*)&poly2));
    float r0, r1;
    asm("mov.b32 %0, %1;" : "=f"(r0) : "r"((x0r_i << 23) + p0_i));
    asm("mov.b32 %0, %1;" : "=f"(r1) : "r"((x1r_i << 23) + p1_i));
    return make_float2(r0, r1);
}

__device__ __forceinline__ void ex2_emulation_f32x2(float* x0_ptr, float* x1_ptr) {
    float2 result = ex2_emulation_f32x2_value(make_float2(*x0_ptr, *x1_ptr));
    *x0_ptr = result.x; *x1_ptr = result.y;
}

__device__ __forceinline__ void softmax_frag_exp2_cast(
    float* sv, uint32_t* pv, int use_emu)
{
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        if (use_emu && j >= 12)
            ex2_emulation_f32x2(&sv[j*2], &sv[j*2+1]);
        else {
            sv[j*2]   = approx_exp2(sv[j*2]);
            sv[j*2+1] = approx_exp2(sv[j*2+1]);
        }
    }
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        __nv_bfloat162 bf = __float22bfloat162_rn({sv[j*2], sv[j*2+1]});
        pv[j] = reinterpret_cast<uint32_t&>(bf);
    }
}

__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ float2 add_f32x2(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    return r;
}

// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}

__device__ __forceinline__ void tma_store_2d(
    const void *tmap, int x, int y, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2}], [%3];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(smem_addr) : "memory");
}

__device__ __forceinline__ void tcgen05_commit_cg2_multicast(int mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "{\n\t"
        ".reg .b16 lo, hi;\n\t"
        "mov.b32 {lo, hi}, %1;\n\t"
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one"
        ".shared::cluster.multicast::cluster.b64 [%0], lo;\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"((uint32_t)cta_mask) : "memory");
}

__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

