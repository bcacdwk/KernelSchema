/* S4 local attention entry: out = softmax(Q K_selected^T * scale1) V * scale2.
 * Q/K are contiguous E4M3 [T,1,H,D] / physical token rows; V is K's first512
 * features. Int32 IDs and row lengths stay on device; -1 markers are masked.
 * CTA=512 threads. Seven concurrent roles are documented at their entry blocks.
 * Layout constants below name shared offsets; q/k/p rings and TMEM parity remain
 * explicit. Gather4 swizzle uses feature*16384 + index_group*512; the D576 tail
 * is a64-feature logical extent in a128-byte zero-filled TMA box.
 * Each tile advances the same producer/consumer barrier parity. The seven work
 * loops allow total_work_items+1 claims, including the final invalid CLC reply.
 * M64 head tiles,512 TMEM columns; four load warps read uniform indices and
 * balance32 gathers each (D512), or40 including the D576 tail.
 * The epilogue writes the supplied output buffer directly; no output alias is
 * replaced. H128/D576 uses aligned TMA output slots, drained before each reuse.
 */
/*
 * Copyright (c) 2023 by FlashInfer team.
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

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_TMEM_SCRATCH_OFFSET 0
#define NUM_K_PIPE_STAGES 2
#define NUM_V_PIPE_STAGES 2
#define NUM_KV_PIPE_STAGES 1
#define NUM_INDEX_PIPE_STAGES 6
#define NUM_SOURCE_WORK_PIPE_STAGES 2
#define NUM_SOURCE_THROTTLE_PIPE_STAGES 2
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 8192
#define SMEM_SMEM_Q_STRIDE 8192
#define SMEM_SMEM_KV_OFF 33792
#define SMEM_SMEM_KV_STAGE_BYTES 16384
#define SMEM_SMEM_KV_STRIDE 16384
#define SMEM_SMEM_V_OFF 33792
#define SMEM_SMEM_V_STAGE_BYTES 16384
#define SMEM_SMEM_V_STRIDE 16384
#define SMEM_SMEM_Q_FULL_OFF 1024
#define SMEM_SMEM_Q_FULL_STAGE_BYTES 32768
#define SMEM_SMEM_Q_FULL_STRIDE 32768
#define SMEM_SMEM_K_FULL_OFF 33792
#define SMEM_SMEM_K_FULL_STAGE_BYTES 65536
#define SMEM_SMEM_K_FULL_STRIDE 65536
#define SMEM_SMEM_V_FULL_OFF 33792
#define SMEM_SMEM_V_FULL_STAGE_BYTES 65536
#define SMEM_SMEM_V_FULL_STRIDE 65536
#define SMEM_SMEM_STATS_MAX_OFF 164864
#define SMEM_SMEM_STATS_MAX_STAGE_BYTES 1024
#define SMEM_SMEM_STATS_MAX_STRIDE 1024
#define SMEM_SMEM_STATS_SUM_OFF 165888
#define SMEM_SMEM_STATS_SUM_STAGE_BYTES 512
#define SMEM_SMEM_STATS_SUM_STRIDE 512
#define SMEM_SMEM_STATS_FINAL_MAX_OFF 166400
#define SMEM_SMEM_STATS_FINAL_MAX_STAGE_BYTES 512
#define SMEM_SMEM_STATS_FINAL_MAX_STRIDE 512
#define SMEM_SMEM_SOFTMAX_WARP_PAIR_EXCHANGE_OFF 166912
#define SMEM_SMEM_SOFTMAX_WARP_PAIR_EXCHANGE_STAGE_BYTES 1024
#define SMEM_SMEM_SOFTMAX_WARP_PAIR_EXCHANGE_STRIDE 1024
#define SMEM_SMEM_CORR_WARP_PAIR_EXCHANGE_OFF 167936
#define SMEM_SMEM_CORR_WARP_PAIR_EXCHANGE_STAGE_BYTES 512
#define SMEM_SMEM_CORR_WARP_PAIR_EXCHANGE_STRIDE 512
#define SMEM_SMEM_SPARSE_INDICES_OFF 168448
#define SMEM_SMEM_SPARSE_INDICES_STAGE_BYTES 1024
#define SMEM_SMEM_SPARSE_INDICES_STRIDE 1024
#define SMEM_SMEM_P_FP8_OFF 174592
#define SMEM_SMEM_P_FP8_STAGE_BYTES 8192
#define SMEM_SMEM_P_FP8_STRIDE 8192
#define SMEM_WORK_RESPONSE_OFF 190976
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define THREADS 512

#include <math_constants.h>

#include "cuda_primitives.cuh"



#define S4_ENTRY aka_s4_indexed_m64_d512
#define S4_ROPE 0
#define S4_SPLIT 0
#define SMEM_TOTAL 191104
#include "kernel_m64.cuh"
#undef SMEM_TOTAL
#undef S4_SPLIT
#undef S4_ROPE
#undef S4_ENTRY

#define S4_ENTRY aka_s4_indexed_m64_d576
#define S4_ROPE 1
#define S4_SPLIT 0
#define SMEM_TOTAL 232064
#include "kernel_m64.cuh"
#undef SMEM_TOTAL
#undef S4_SPLIT
#undef S4_ROPE
#undef S4_ENTRY

#define S4_ENTRY aka_s4_split8_d512
#define S4_ROPE 0
#define S4_SPLIT 1
#define SMEM_TOTAL 191104
#include "kernel_m64.cuh"
#undef SMEM_TOTAL
#undef S4_SPLIT
#undef S4_ROPE
#undef S4_ENTRY

#define S4_ENTRY aka_s4_split8_d576
#define S4_ROPE 1
#define S4_SPLIT 1
#define SMEM_TOTAL 232064
#include "kernel_m64.cuh"
#undef SMEM_TOTAL
#undef S4_SPLIT
#undef S4_ROPE
#undef S4_ENTRY
