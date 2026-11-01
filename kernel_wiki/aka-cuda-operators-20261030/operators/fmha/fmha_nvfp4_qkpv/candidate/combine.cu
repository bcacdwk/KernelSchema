// Shared SM100/SM103 combine; build.py supplies the original module symbols.
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

#include "math.cuh"

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 128

#include <math_constants.h>

// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

extern "C" {

__global__ __launch_bounds__(128) void
AKA_COMBINE_KERNEL(__half* __restrict__ partial_O, float* __restrict__ partial_ML, int* __restrict__ combine_table, int* __restrict__ seg_begin, int* __restrict__ seg_len, __nv_bfloat16* __restrict__ O, int num_heads)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int unit = blockIdx.x / 128;
    int cluster_row = blockIdx.x % 128 * 4 + warp;
    int rec = unit * 5;
    int packed = combine_table[rec + 1];
    int first_slot = combine_table[rec + 2];
    int splits = combine_table[rec + 3];
    // This unit's cluster tile may be a segment tail with far fewer than
    // PARTIAL_ROWS valid packed rows, but the grid is uniform -- one
    // COMBINE_CTAS_PER_UNIT CTAs per split unit -- so most of a tail unit's warps
    // own rows that do not exist.  They used to fall through to the partial fetch
    // below and only then hit the ``local_row < doc_len`` guard, which for shape 8
    // meant 511 of one unit's 512 rows pulling six splits' worth of partials the
    // attention program never wrote: ~20% of this kernel's entire read traffic.
    // ``combine_table[rec + 4]`` is that row count, so the verdict arrives in the
    // same load wave as ``first_slot``/``splits`` -- no extra dependent miss --
    // and the warps that own nothing leave before issuing a single fetch.  Pure
    // dead-traffic removal: those warps computed nothing and stored nothing
    // before either, so no delivered value changes.
    // The ``local_row < doc_len`` guard below is implied by this early-out --
    // both compare the same packed row against the same ``plan.seg_len[s]`` --
    // but it stays: it is what makes the output store safe if the two ever
    // disagreed, and after the fetches are in flight it costs nothing.
    if (cluster_row >= combine_table[rec + 4]) return;
    int head = packed >> 16;
    int c = packed & 65535;
    int local_row = c * 512 + cluster_row;
    int d_base = lane * 4;
    // Every split's statistics and partial output are fetched up front, one
    // 8-byte load each.  The reduction below is a strict ascending-``i``
    // dependency chain (``denom`` and ``acc`` accumulate in split order and
    // must keep that order to stay bit-identical), so a load issued from
    // inside it can only start once the previous split's arithmetic has
    // retired: the stage then runs at one in-flight cache line per thread
    // instead of ``splits`` of them.  Hoisting is pure reordering of the
    // *loads*; no arithmetic operation, operand or sequence changes.
    // Inactive splits keep the identity values the loop below already
    // skips, so ``row_max`` and the guard behave exactly as before.
    //
    // They are also hoisted *above* the row guard, and ``combine_table[rec]``
    // with the ``seg_begin``/``seg_len`` pair it feeds is read below instead of
    // above.  The guard needs two dependent global loads (segment index, then
    // that segment's bounds) before a single partial could be fetched, so the
    // whole kernel -- four microseconds long, of which roughly two is launch --
    // began with three serialised cache misses.  The partial addresses depend
    // only on ``first_slot``, ``splits`` and ``cluster_row``, so they no longer
    // wait for the bounds, and the bounds loads now overlap them.  Reading a
    // guarded-out row's partials is safe and was already happening for every
    // inactive split: ``first_slot + i`` stays inside the partial workspace for
    // all ``i < splits`` and ``cluster_row < 512`` always, so the addresses are
    // in bounds for every row of every unit, and the guard still decides
    // whether anything is computed or written.
    float2 split_ml[8];
    uint2 split_ov[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        split_ml[i] = make_float2(-CAKE_INF, 0.0f);
        split_ov[i] = make_uint2(0u, 0u);
        if (splits > i) {
            int stat_row = (first_slot + i) * 512 + cluster_row;
            split_ml[i] = *reinterpret_cast<const float2*>(partial_ML + stat_row * 2);
            split_ov[i] = *reinterpret_cast<const uint2*>(partial_O + stat_row * 128 + d_base);
        }
    }
    int seg = combine_table[rec];
    int doc_begin = seg_begin[seg];
    int doc_len = seg_len[seg];
    if (local_row < doc_len) {
        float row_max = split_ml[0].x;
        #pragma unroll
        for (int i_1 = 1; i_1 < 8; i_1++) {
            float _max_0 = max_noftz(row_max, split_ml[i_1].x);
            row_max = _max_0;
        }
        float acc[4];
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            acc[e] = 0.0f;
        }
        float denom = 0.0f;
        #pragma unroll
        for (int i_2 = 0; i_2 < 8; i_2++) {
            if (splits > i_2) {
                float _exp2_0 = approx_exp2(split_ml[i_2].x - row_max);
                float weight = ((split_ml[i_2].x == -CAKE_INF) ? 0.0f : _exp2_0 * split_ml[i_2].y);
                denom = denom + weight;
                float vals[4];
                {
                    uint2 _vld_0;
                    _vld_0 = split_ov[i_2];
                    uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
                    #pragma unroll
                    for (int _pair = 0; _pair < 2; _pair++) {
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 h_lo, h_hi;\n\t"
                            ".reg .b32 f_lo, f_hi;\n\t"
                            "mov.b32 {h_lo, h_hi}, %1;\n\t"
                            "cvt.f32.f16 f_lo, h_lo;\n\t"
                            "cvt.f32.f16 f_hi, h_hi;\n\t"
                            "mov.b64 %0, {f_lo, f_hi};\n\t"
                            "}\n"
                            : "=l"(*reinterpret_cast<unsigned long long*>(&vals[0 + _pair * 2]))
                            : "r"(_vpairs_0[_pair]));
                    }
                }
                #pragma unroll
                for (int e_1 = 0; e_1 < 4; e_1++) {
                    float _fma_0 = __fmaf_rn(vals[e_1], weight, acc[e_1]);
                    acc[e_1] = _fma_0;
                }
            }
        }
        float _rcp_0 = approx_rcp(denom);
        float inv = ((denom > 0.0f) ? _rcp_0 : 0.0f);
        int out_row = (doc_begin + local_row) * num_heads + head;
        {
            const float2 _prescale2_1 = {inv, inv};
            #if __CUDA_ARCH__ >= 1000
            #pragma unroll
            for (int _ps = 0; _ps < 2; _ps++)
                mul_f32x2_inplace(&reinterpret_cast<float2*>(&acc[0])[_ps], _prescale2_1);
            #else
            #pragma unroll
            for (int _ps = 0; _ps < 4; _ps++)
                acc[0 + _ps] *= inv;
            #endif
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(acc[0 + 0], acc[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(acc[0 + 2], acc[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(O + (out_row * 128 + d_base)))[0]) = _pk2;
        }
    }
}

} // extern "C"
