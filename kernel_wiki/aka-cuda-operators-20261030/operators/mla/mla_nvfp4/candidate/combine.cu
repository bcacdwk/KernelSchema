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


/* Split-KV merge, one warp per (query, head), 512 output elements per head.
 * Each lane reads two of at most 64 split LSEs. Warp max/sum reductions
 * form global_lse2 = max_lse2 + log2(sum(exp2(split_lse2-max_lse2))).
 * BF16 partial outputs are loaded in 16-byte vectors, weighted in FP32 by
 * exp2(split_lse2-global_lse2), then packed to BF16. Final LSE is natural
 * log. used==1 was already written by the main kernel and is untouched.
 * This TU keeps the original precise compiler flags; do not inherit the
 * main kernel's fast-math option or change the partial-output rounding.
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
template <int N>
struct __align__(128) CakeTensorMapPack { CakeTensorMap maps[N]; };

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(CakeTensorMap) >= alignof(CUtensorMap), "CakeTensorMap alignment must cover the CUtensorMap CUDA ABI");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}

extern "C" {

__global__ __launch_bounds__(256, 1) void
kernel_cake_nvfp4_mla_decode_ff76ef6d8e6453ca5e3c(__nv_bfloat16* __restrict__ o, float* __restrict__ lse, __nv_bfloat16* __restrict__ partial_o, float* __restrict__ partial_lse, int* __restrict__ row_splits, int total_q, int num_heads, int max_splits)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    float NEG_INF = -CUDART_INF_F;
    int query_index = blockIdx.x;
    int head_index = blockIdx.y * 8 + warp;
    int output_lane = lane;
    if (head_index < num_heads) {
        int row = query_index * num_heads + head_index;
        int num_splits = row_splits[query_index];
        int lse_base = row * max_splits;
        if (num_splits > 1) {
            float lane_split_lse2[2];
            float max_split_lse2 = NEG_INF;
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                int split_index = output_lane + i * 32;
                float split_lse2 = ((split_index < num_splits) ? partial_lse[lse_base + split_index] : NEG_INF);
                lane_split_lse2[i] = split_lse2;
                float max_with_split = max_noftz(max_split_lse2, split_lse2);
                max_split_lse2 = max_with_split;
            }
            float warp_max_lse2 = max_split_lse2;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                warp_max_lse2 = max_noftz(warp_max_lse2, __shfl_xor_sync(0xFFFFFFFF, warp_max_lse2, offset));
            max_split_lse2 = warp_max_lse2;
            max_split_lse2 = ((max_split_lse2 != NEG_INF) ? max_split_lse2 : 0.0f);
            float normalizer = 0.0f;
            #pragma unroll
            for (int lane_split = 0; lane_split < 2; lane_split++) {
                float unnormalized_weight = approx_exp2(lane_split_lse2[lane_split] - max_split_lse2);
                normalizer = normalizer + unnormalized_weight;
            }
            float warp_normalizer = normalizer;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                warp_normalizer += __shfl_xor_sync(0xFFFFFFFF, warp_normalizer, offset);
            normalizer = warp_normalizer;
            float log2_normalizer;
            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(log2_normalizer) : "f"(normalizer));
            float global_lse2 = max_split_lse2 + log2_normalizer;
            if (output_lane == 0) {
                *(reinterpret_cast<float*>(lse + row) + (0)) = global_lse2 * 0.6931471805599453f;
            }
            float output_accum[16];
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                output_accum[j] = 0.0f;
            }
            int split_stride = total_q * num_heads * 512;
            int o_base = row * 512 + output_lane * 16;
            #pragma unroll 1
            for (int split = 0; split < num_splits; split++) {
                float normalized_weight = approx_exp2(partial_lse[lse_base + split] - global_lse2);
                float split_weight = normalized_weight;
                #pragma unroll
                for (int vector_offset = 0; vector_offset < 16; vector_offset += 8) {
                    float partial_values[8];
                    {
                        const uint4* partial_vector_ptr = reinterpret_cast<const uint4*>(partial_o + (o_base + split * split_stride + vector_offset) + 0);
                        uint4 partial_vectors[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            partial_vectors[_blk] = partial_vector_ptr[_blk];
                            uint32_t* partial_bf16_pairs = reinterpret_cast<uint32_t*>(&partial_vectors[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&partial_values[0 + _blk * 8 + _pair * 2])[0]), "=f"((&partial_values[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(partial_bf16_pairs[_pair]));
                            }
                        }
                    }
                    #pragma unroll
                    for (int vector_element = 0; vector_element < 8; vector_element++) {
                        output_accum[vector_offset + vector_element] = output_accum[vector_offset + vector_element] + partial_values[vector_element] * split_weight;
                    }
                }
            }
            {
                __nv_bfloat162 packed_output[8];
                packed_output[0] = __floats2bfloat162_rn(output_accum[0 + 0], output_accum[0 + 1]);
                packed_output[1] = __floats2bfloat162_rn(output_accum[0 + 2], output_accum[0 + 3]);
                packed_output[2] = __floats2bfloat162_rn(output_accum[0 + 4], output_accum[0 + 5]);
                packed_output[3] = __floats2bfloat162_rn(output_accum[0 + 6], output_accum[0 + 7]);
                packed_output[4] = __floats2bfloat162_rn(output_accum[0 + 8], output_accum[0 + 9]);
                packed_output[5] = __floats2bfloat162_rn(output_accum[0 + 10], output_accum[0 + 11]);
                packed_output[6] = __floats2bfloat162_rn(output_accum[0 + 12], output_accum[0 + 13]);
                packed_output[7] = __floats2bfloat162_rn(output_accum[0 + 14], output_accum[0 + 15]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(o + (row * 512 + output_lane * 16)))[0]) = *reinterpret_cast<uint4*>(&packed_output[0]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(o + (row * 512 + output_lane * 16)))[8]) = *reinterpret_cast<uint4*>(&packed_output[4]);
            }
        }
    }
}

} // extern "C"
