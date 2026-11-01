// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
/* gdn_post_conv_mtp -- device kernels for the GDN decode/MTP post-convolution stage.
 *
 * Upstream: vLLM revision c723a831a81cb4ff89ea6d61b2d15109306ed1bc, file
 * csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu (SHA256
 * 9581d256847b1922b21a54c39086d92da4f402b7310b0e8649e45ec0cae341fe).
 * One typed CUDA template preserves the original mathematical body and 20 specializations.
 * Original restricted parameter qualifiers and launch bounds are retained; source identity
 * are preserved in each specialization. Pure CUDA C++ plus
 * inline PTX (cp.async): no CUTLASS/CuTe, no cuBLAS, no Triton, no vLLM at build or run time.
 *
 * WHAT IT COMPUTES
 *   One block handles one (request, value_head) pair.  With H = key heads, HV = value heads,
 *   D = kDimK = kDimV = 128, and t indexing the request's MTP tokens (bos + t):
 *
 *     q_t = mixed_qkv[token, key_head*D .. +D)                 (bf16 -> f32)
 *     k_t = mixed_qkv[token, H*D + key_head*D .. +D)
 *     v_t = mixed_qkv[token, 2*H*D + value_head*D .. +D)
 *     q_t *= rsqrt(sum(q_t^2) + 1e-6) * scale                  (QK L2 normalization, eps fixed 1e-6)
 *     k_t *= rsqrt(sum(k_t^2) + 1e-6)
 *     g_t     = -exp(A_log[value_head]) * softplus(a[token,value_head] + dt_bias[value_head])
 *     decay_t = exp(g_t);  beta_t = sigmoid(b[token,value_head])
 *
 *     h <- h * decay_t                                  (state decay, FP32 registers)
 *     delta_t = (v_t[value] - sum_k(h[value,k] * k_t[k])) * beta_t
 *     h <- h + k_t * delta_t                            (delta-rule update)
 *     raw_t[value] = sum_k(h[value,k] * q_t[k])         (read-out)
 *
 *     out[token,value_head,value] =
 *         bf16( raw_t[value] * rsqrt(sum(raw_t^2)/128 + norm_eps) * norm_weight[value]
 *               * gate(output_gate[token,value_head,value]) )
 *     with gate = sigmoid when output_gate_activation == "sigmoid", else silu(x) = x*sigmoid(x).
 *
 *   key_head = value_head / ValueHeadsPerKeyHead, so ValueHeadsPerKeyHead is a compile-time
 *   constant per entry (HV/H in {1,2,3,4,8}).
 *
 * STATE (the in-place, cross-call part of the contract)
 *   source_slot = state_indices[request, accepted-1] when 0 < accepted <= state_indices_width,
 *   else 0.  If source_slot <= 0, or the request has more than kMaxMtpTokens = 8 tokens, the
 *   block zero-fills this request's out rows and returns without touching state -- that is the
 *   reserved-slot-0 / inactive-request semantics, not an error path.
 *   Otherwise h is seeded from state[source_slot, value_head, :, :] and, after each token t,
 *   written back to state[state_indices[request, t], value_head, :, :] whenever that slot is
 *   positive.  Reads and writes alias the same caller-owned pool; slot 0 and non-positive slots
 *   are never written.
 *
 * LAYOUT AND WORK SPLIT
 *   grid  = (num_requests, num_value_heads);  block = kThreads = 256 threads = kWarps = 8 warps.
 *   Prologue: warp t (only for t < num_tokens) loads token t's q/k/v, reduces the two squared
 *     norms with warp_reduce_sum_pair, broadcasts the two rsqrt scales from lane 0 via
 *     __shfl_sync, and lane 0 computes decay_t / beta_t.  Each lane owns 4 K dims per unrolled
 *     step (dim = lane + i*32, i < 4), i.e. the warp spans all 128 K dims.
 *   Recurrence: value rows are processed in kNumChunks = 4 chunks of kChunkV = 32 rows.  Inside
 *     a chunk each warp owns kRowsPerWarp = kChunkV/kWarps = 4 rows (rows[row] = warp + row*8)
 *     and each lane owns k_base = lane*4 .. +4 K dims, so h[4][4] is a 16-float register tile
 *     per lane.  The two reductions per token (h.k and h.q) are warp-scoped shuffles.
 *   Epilogue: warp t re-reads its 128 raw values, warp_reduce_sum's the squares for RMSNorm,
 *     applies norm_weight and the gate, and writes bf16 out.
 *
 * SHARED MEMORY
 *   shared_state[2][32][128]  StateT   double-buffered state chunk, 2 cp.async stages
 *   shared_q[8][128], shared_k[8][128]  float   normalized Q / K per MTP token
 *   shared_v[8][128], shared_out[8][128]  __nv_bfloat16   V and raw read-out per token
 *   shared_decay[8], shared_beta[8]  float
 *   Peak = 2*32*128*sizeof(StateT) + 2*8*128*4 + 2*8*128*2 + 16*4 bytes, i.e. 45,120 B for
 *   FP32 state and 28,736 B for BF16 state.  No dynamic shared memory, no cluster, no TMA.
 *
 * ASYNC COPY AND BARRIERS
 *   copy_state_chunk issues 16-byte `cp.async.cg.shared.global` per thread (16/sizeof(StateT)
 *   elements each) over one 32x128 chunk and closes a `cp.async.commit_group`; chunk 0 is
 *   prefetched before the prologue's __syncthreads, and each loop iteration does
 *   `cp.async.wait_all` + __syncthreads before consuming stage chunk&1 while prefetching
 *   chunk+1 into stage (chunk+1)&1.  __syncthreads also separates the prologue from the
 *   recurrence and the recurrence from the epilogue.
 *
 * NUMERICS THAT MUST NOT DRIFT
 *   All recurrence arithmetic is FP32 in registers regardless of the state dtype; state is
 *   converted on load (load_state4) and on store (store_state4, __floats2bfloat162_rn).  The
 *   h *= decay / dot / h += k*delta / dot order inside the 4-wide unroll is the upstream order
 *   and is what the reference tolerance was measured against.  sigmoid/silu/softplus use
 *   __expf and log1pf and are built with --use_fast_math; both are part of the pinned build
 *   contract and are preserved in the operator manifest.
 */
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace gdn_post {
namespace {

template <typename StateT>
__device__ __forceinline__ void cp_async_16b(StateT* smem_ptr,
                                             const StateT* gmem_ptr) {
  const uint32_t smem_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :
               : "r"(smem_addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;\n" ::: "memory");
}

template <typename StateT, int ChunkV, int DimK, int Stages>
__device__ __forceinline__ void copy_state_chunk(StateT* shared_state,
                                                 const StateT* state, int chunk,
                                                 int thread, int threads) {
  constexpr int kElementsPerCopy = 16 / sizeof(StateT);
  constexpr int kCopiesPerChunk = ChunkV * DimK / kElementsPerCopy;
  const int stage = chunk % Stages;
  for (int copy = thread; copy < kCopiesPerChunk; copy += threads) {
    const int element = copy * kElementsPerCopy;
    cp_async_16b(shared_state + stage * ChunkV * DimK + element,
                 state + chunk * ChunkV * DimK + element);
  }
  cp_async_commit();
}

template <typename StateT>
__device__ __forceinline__ float4 load_state4(const StateT* state);

template <>
__device__ __forceinline__ float4 load_state4<float>(const float* state) {
  return *reinterpret_cast<const float4*>(state);
}

template <>
__device__ __forceinline__ float4
load_state4<__nv_bfloat16>(const __nv_bfloat16* state) {
  const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(state);
  const __nv_bfloat162 hi = *reinterpret_cast<const __nv_bfloat162*>(state + 2);
  return make_float4(__bfloat162float(lo.x), __bfloat162float(lo.y),
                     __bfloat162float(hi.x), __bfloat162float(hi.y));
}

template <typename StateT>
__device__ __forceinline__ void store_state4(StateT* state, float4 value);

template <>
__device__ __forceinline__ void store_state4<float>(float* state,
                                                    float4 value) {
  *reinterpret_cast<float4*>(state) = value;
}

template <>
__device__ __forceinline__ void store_state4<__nv_bfloat16>(
    __nv_bfloat16* state, float4 value) {
  *reinterpret_cast<__nv_bfloat162*>(state) =
      __floats2bfloat162_rn(value.x, value.y);
  *reinterpret_cast<__nv_bfloat162*>(state + 2) =
      __floats2bfloat162_rn(value.z, value.w);
}

constexpr int kDimK = 128;
constexpr int kDimV = 128;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kChunkV = 32;
constexpr int kNumChunks = kDimV / kChunkV;
constexpr int kRowsPerWarp = kChunkV / kWarps;
constexpr int kMaxMtpTokens = 8;
constexpr int kDtBiasFloat32 = 0;
constexpr int kDtBiasBFloat16 = 1;
constexpr int kDtBiasFloat16 = 2;

struct GdnDecodeStrides {
  int64_t mixed_row;
  int64_t a_row;
  int64_t b_row;
  int64_t gate_row;
  int64_t state_slot;
};

__device__ __forceinline__ float sigmoid_fast(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float silu_fast(float x) {
  return x * sigmoid_fast(x);
}

__device__ __forceinline__ float softplus_fast(float x) {
  return x > 20.0f ? x : log1pf(__expf(x));
}

__device__ __forceinline__ float load_dt_bias(const void* dt_bias, int head,
                                              int dt_bias_type) {
  if (dt_bias_type == kDtBiasBFloat16) {
    return __bfloat162float(static_cast<const __nv_bfloat16*>(dt_bias)[head]);
  }
  if (dt_bias_type == kDtBiasFloat16) {
    return __half2float(static_cast<const __half*>(dt_bias)[head]);
  }
  return static_cast<const float*>(dt_bias)[head];
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

struct Sum2 {
  float x;
  float y;
};

__device__ __forceinline__ Sum2 warp_reduce_sum_pair(float x, float y) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_xor_sync(0xffffffffu, x, offset);
    y += __shfl_xor_sync(0xffffffffu, y, offset);
  }
  return {x, y};
}

// Host-only argument carrier. Device entry parameters remain individually restricted,
// as in the original P kernel; this carrier does not cross the CUDA launch ABI.
struct GdnPostConvParams {
  const __nv_bfloat16* __restrict__ mixed_qkv;
  const __nv_bfloat16* __restrict__ a;
  const __nv_bfloat16* __restrict__ b;
  const float* __restrict__ a_log;
  const void* __restrict__ dt_bias;
  const int* __restrict__ state_indices;
  const int* __restrict__ cu_seqlens;
  const int* __restrict__ num_accepted_tokens;
  void* __restrict__ state;  // StateT*; re-typed inside the body
  const __nv_bfloat16* __restrict__ output_gate;
  const void* __restrict__ norm_weight;
  __nv_bfloat16* __restrict__ out;
  int H;
  int HV;
  int state_indices_width;
  int dt_bias_type;
  bool norm_weight_is_bf16;
  float scale;
  float norm_eps;
  GdnDecodeStrides strides;
};

template<typename C_StateT, int C_HeadRatio, bool C_SigmoidGate>
__global__ __launch_bounds__(kThreads, 2) void gdn_post_conv_mtp(
    const __nv_bfloat16* __restrict__ mixed_qkv,
    const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b,
    const float* __restrict__ a_log, const void* __restrict__ dt_bias,
    const int* __restrict__ state_indices, const int* __restrict__ cu_seqlens,
    const int* __restrict__ num_accepted_tokens, C_StateT* __restrict__ state,
    const __nv_bfloat16* __restrict__ output_gate,
    const void* __restrict__ norm_weight, __nv_bfloat16* __restrict__ out,
    int H, int HV, int state_indices_width, int dt_bias_type,
    bool norm_weight_is_bf16, float scale, float norm_eps,
    GdnDecodeStrides strides) {
  using StateT = C_StateT;
  constexpr int ValueHeadsPerKeyHead = C_HeadRatio;
  constexpr bool SigmoidGate = C_SigmoidGate;
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Shared original P body, included inside each explicit entry. No include guard.

  const int request = blockIdx.x;
  const int value_head = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int bos = cu_seqlens[request];
  const int eos = cu_seqlens[request + 1];
  const int num_tokens = eos - bos;
  if (num_tokens <= 0) {
    return;
  }

  const int accepted = num_accepted_tokens[request];
  const int source_slot =
      accepted > 0 && accepted <= state_indices_width
          ? state_indices[request * state_indices_width + accepted - 1]
          : 0;
  if (source_slot <= 0 || num_tokens > kMaxMtpTokens) {
    for (int linear = tid; linear < num_tokens * kDimV; linear += kThreads) {
      const int token = bos + linear / kDimV;
      const int value = linear % kDimV;
      const int64_t out_offset =
          (static_cast<int64_t>(token) * HV + value_head) * kDimV + value;
      out[out_offset] = __float2bfloat16(0.0f);
    }
    return;
  }

  const int key_head = value_head / ValueHeadsPerKeyHead;
  __shared__ StateT shared_state[2][kChunkV][kDimK];
  __shared__ float shared_q[kMaxMtpTokens][kDimK];
  __shared__ float shared_k[kMaxMtpTokens][kDimK];
  __shared__ __nv_bfloat16 shared_v[kMaxMtpTokens][kDimV];
  __shared__ __nv_bfloat16 shared_out[kMaxMtpTokens][kDimV];
  __shared__ float shared_decay[kMaxMtpTokens];
  __shared__ float shared_beta[kMaxMtpTokens];

  StateT* source_state =
      state + static_cast<int64_t>(source_slot) * strides.state_slot +
      value_head * kDimV * kDimK;
  copy_state_chunk<StateT, kChunkV, kDimK, 2>(&shared_state[0][0][0],
                                              source_state, 0, tid, kThreads);

  if (warp < num_tokens) {
    const int t = warp;
    const int token = bos + t;
    const int64_t mixed_base = static_cast<int64_t>(token) * strides.mixed_row;
    float q_values[4];
    float k_values[4];
    float q_square = 0.0f;
    float k_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * 32;
      q_values[i] =
          __bfloat162float(mixed_qkv[mixed_base + key_head * kDimK + dim]);
      k_values[i] = __bfloat162float(
          mixed_qkv[mixed_base + H * kDimK + key_head * kDimK + dim]);
      shared_v[t][dim] =
          mixed_qkv[mixed_base + 2 * H * kDimK + value_head * kDimV + dim];
      q_square += q_values[i] * q_values[i];
      k_square += k_values[i] * k_values[i];
    }
    const Sum2 qk_sums = warp_reduce_sum_pair(q_square, k_square);
    const float q_scale = __shfl_sync(
        0xffffffffu, lane == 0 ? rsqrtf(qk_sums.x + 1.0e-6f) * scale : 0.0f, 0);
    const float k_scale = __shfl_sync(
        0xffffffffu, lane == 0 ? rsqrtf(qk_sums.y + 1.0e-6f) : 0.0f, 0);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * 32;
      shared_q[t][dim] = q_values[i] * q_scale;
      shared_k[t][dim] = k_values[i] * k_scale;
    }
    if (lane == 0) {
      const float a_value = __bfloat162float(
          a[static_cast<int64_t>(token) * strides.a_row + value_head]);
      const float b_value = __bfloat162float(
          b[static_cast<int64_t>(token) * strides.b_row + value_head]);
      const float g = -__expf(a_log[value_head]) *
                      softplus_fast(a_value + load_dt_bias(dt_bias, value_head,
                                                           dt_bias_type));
      shared_decay[t] = __expf(g);
      shared_beta[t] = sigmoid_fast(b_value);
    }
  }
  __syncthreads();

  const int k_base = lane * 4;
  int rows[kRowsPerWarp];
#pragma unroll
  for (int row = 0; row < kRowsPerWarp; ++row) {
    rows[row] = warp + row * kWarps;
  }

#pragma unroll
  for (int chunk = 0; chunk < kNumChunks; ++chunk) {
    cp_async_wait_all();
    __syncthreads();
    if (chunk + 1 < kNumChunks) {
      copy_state_chunk<StateT, kChunkV, kDimK, 2>(
          &shared_state[0][0][0], source_state, chunk + 1, tid, kThreads);
    }

    float h[kRowsPerWarp][4];
#pragma unroll
    for (int row = 0; row < kRowsPerWarp; ++row) {
      const float4 state_value =
          load_state4(&shared_state[chunk & 1][rows[row]][k_base]);
      h[row][0] = state_value.x;
      h[row][1] = state_value.y;
      h[row][2] = state_value.z;
      h[row][3] = state_value.w;
    }

    for (int t = 0; t < num_tokens; ++t) {
      const float4 q4 = *reinterpret_cast<const float4*>(&shared_q[t][k_base]);
      const float4 k4 = *reinterpret_cast<const float4*>(&shared_k[t][k_base]);
      const float q_values[4] = {q4.x, q4.y, q4.z, q4.w};
      const float k_values[4] = {k4.x, k4.y, k4.z, k4.w};

      float dot_hk[kRowsPerWarp] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          h[row][i] *= shared_decay[t];
          dot_hk[row] += h[row][i] * k_values[i];
        }
      }
      const Sum2 dot_hk_01 = warp_reduce_sum_pair(dot_hk[0], dot_hk[1]);
      const Sum2 dot_hk_23 = warp_reduce_sum_pair(dot_hk[2], dot_hk[3]);
      const float reduced_hk[kRowsPerWarp] = {dot_hk_01.x, dot_hk_01.y,
                                              dot_hk_23.x, dot_hk_23.y};

      float dot_hq[kRowsPerWarp] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
        const int value = chunk * kChunkV + rows[row];
        const float delta =
            (__bfloat162float(shared_v[t][value]) - reduced_hk[row]) *
            shared_beta[t];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          h[row][i] += k_values[i] * delta;
          dot_hq[row] += h[row][i] * q_values[i];
        }
      }
      const Sum2 dot_hq_01 = warp_reduce_sum_pair(dot_hq[0], dot_hq[1]);
      const Sum2 dot_hq_23 = warp_reduce_sum_pair(dot_hq[2], dot_hq[3]);
      if (lane == 0) {
        shared_out[t][chunk * kChunkV + rows[0]] =
            __float2bfloat16(dot_hq_01.x);
        shared_out[t][chunk * kChunkV + rows[1]] =
            __float2bfloat16(dot_hq_01.y);
        shared_out[t][chunk * kChunkV + rows[2]] =
            __float2bfloat16(dot_hq_23.x);
        shared_out[t][chunk * kChunkV + rows[3]] =
            __float2bfloat16(dot_hq_23.y);
      }

      const int destination_slot =
          state_indices[request * state_indices_width + t];
      if (destination_slot > 0) {
        StateT* destination_state =
            state +
            static_cast<int64_t>(destination_slot) * strides.state_slot +
            value_head * kDimV * kDimK;
#pragma unroll
        for (int row = 0; row < kRowsPerWarp; ++row) {
          const int value = chunk * kChunkV + rows[row];
          store_state4(destination_state + value * kDimK + k_base,
                       make_float4(h[row][0], h[row][1], h[row][2], h[row][3]));
        }
      }
    }
  }
  __syncthreads();

  if (warp < num_tokens) {
    const int t = warp;
    float output_values[4];
    float sum_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int value = lane + i * 32;
      output_values[i] = __bfloat162float(shared_out[t][value]);
      sum_square += output_values[i] * output_values[i];
    }
    sum_square = warp_reduce_sum(sum_square);
    const float rstd =
        rsqrtf(sum_square / static_cast<float>(kDimV) + norm_eps);
    const int token = bos + t;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int value = lane + i * 32;
      const float gate_input = __bfloat162float(
          output_gate[static_cast<int64_t>(token) * strides.gate_row +
                      value_head * kDimV + value]);
      const float gate =
          SigmoidGate ? sigmoid_fast(gate_input) : silu_fast(gate_input);
      const float weight =
          norm_weight_is_bf16
              ? __bfloat162float(
                    static_cast<const __nv_bfloat16*>(norm_weight)[value])
              : static_cast<const float*>(norm_weight)[value];
      const int64_t out_offset =
          (static_cast<int64_t>(token) * HV + value_head) * kDimV + value;
      out[out_offset] =
          __float2bfloat16(output_values[i] * rstd * weight * gate);
    }
  }


}
} // namespace
} // namespace gdn_post
