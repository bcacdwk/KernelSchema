// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Copyright (c) 2024, Tri Dao.
/* causal_conv1d_update -- device kernel for the GDN conv/MTP sliding-window state update.
 *
 * Original source: vLLM c723a831a81cb4ff89ea6d61b2d15109306ed1bc,
 * vllm/model_executor/layers/mamba/ops/causal_conv1d.py,
 * SHA256 cb16cc9250c4195c09d6d83b43bba607c2d749b621d8d135a920443d1577265f.
 * Intermediate corrected S: OLD/problems/pure-cuda-single-file/build/gdnconv-fix/pure/
 * causal_conv1d_update_pure.cu (sha256 e90c39b84686a6c0..., 735 lines), the REPAIRED
 * hand-written pure CUDA + inline PTX port of the single vendored Triton kernel
 * `_causal_conv1d_update_kernel` from vLLM c723a831a81cb4ff89ea6d61b2d15109306ed1bc.
 * Derived mechanically by families/gdn-conv/tools/extract_conv.py.
 *
 * The intermediate repaired port and its historical receipts are separate from original P.
 * Current source-bound v4 validation is recorded in each operator's manifest/WORKLOG.
 *
 *   identity   `patched_native`.  It carries native-fix `fp32-convolution-product`
 *              (repair_id core-native-repair-20261013): both operands are widened BEFORE the
 *              multiply and the product is formed in FP32.  The unrepaired sibling
 *              (sha256 52d519e8...) multiplied in BF16 and reproduced the Triton rounding;
 *              that is the rounding the accepted baseline removes, so the two are NOT
 *              interchangeable and swapping them silently changes results.
 *   pinned arithmetic  every numeric choice was read off the dumped Triton PTX
 *              (pure/evidence/triton-seqlen{2,4,8}.ptx): non-contractible __fadd_rn/__fmul_rn
 *              accumulation, and silu as the exact instruction sequence
 *              sub.f32 / mul.f32 by log2(e) / ex2.approx.f32 / add.f32 1.0 / div.full.f32.
 *              Those are inline PTX or non-contractible intrinsics, so the result does not
 *              depend on --use_fast_math.
 *   grid       (batch, cdiv(dim, kBlockN=256)), one thread per feature; no shared memory, no
 *              cross-lane reduction, no atomics, so the lane mapping cannot reach a result.
 *   preserved  every stride/offset/mask, the varlen `state_len -= seqlen - (qe-qs)` rewrite,
 *              the spec-decode `num_accepted-1` token offset, the null-block early exit, the
 *              APC dual conv_state_indices lookup, the `where(mask, old_state, x)` sliding
 *              window state write, tl.debug_barrier() -> __syncthreads() at the same point,
 *              and griddepcontrol.wait / .launch_dependents at the same three points.
 *   in place   conv_state is read and written in the caller's own padded backing; slot 0 and
 *              unscheduled slots are left alone.  out may alias x.
 *   domain     conv_state/x/weight/bias/out all bf16; KERNEL_WIDTH 2..6; NP2_STATELEN
 *              {1,2,4,8,16}.  Anything else is refused by launch_dispatch rather than
 *              computed by an unexercised instantiation.
 *   historical The intermediate port reports a 120-case instruction-equivalence study.
 *              This is historical evidence, not current v4 acceptance and not proof that the
 *              repaired FP32 product equals the pristine BF16 product. Original P still fails
 *              six v4 frozen shapes; corrected S and current N have distinct identities.
 */
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace sfc_conv_pure {
namespace {

constexpr int kBlockN = 256;              // conv.py: BLOCK_N=256

// ---------------------------------------------------------------------------
// instruction-level primitives, pinned to the dumped Triton PTX
// ---------------------------------------------------------------------------

__device__ __forceinline__ void gdc_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}

__device__ __forceinline__ void gdc_launch_dependents() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

// conv.py:306 as REPAIRED by exports/core-atrex-bench-20261013/dependencies/native-fixes
// (`fp32-convolution-product`, repair_id core-native-repair-20261013):
//
//     acc += matrix_x.to(tl.float32) * matrix_w.to(tl.float32)
//
// BF16 storage does not imply a separately rounded BF16 product in this FP32 convolution, so both
// operands are widened BEFORE the multiply and the product is formed in FP32.  The unrepaired
// conv.py multiplied in BF16 and widened only the product; this port reproduced that instruction
// sequence byte-faithfully (an inline-PTX `mul.rn.bf16` helper existed solely to pin it), and it
// is that reproduced rounding the accepted `patched_native` baseline removes.  Carrying the repair
// therefore deletes the helper: nothing on the repaired path multiplies in BF16 any more.
__device__ __forceinline__ float to_f32(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float to_f32(float v) { return v; }

template <class T>
__device__ __forceinline__ float acc_add(float acc, T x, T w) {
  return __fadd_rn(acc, __fmul_rn(to_f32(x), to_f32(w)));
}

// store conversion: cvt.rn.bf16x2.f32 in the PTX == per-element cvt.rn.bf16.f32
__device__ __forceinline__ __nv_bfloat16 from_f32(__nv_bfloat16*, float v) {
  return __float2bfloat16_rn(v);
}
__device__ __forceinline__ float from_f32(float*, float v) { return v; }

__device__ __forceinline__ __nv_bfloat16 zero_of(__nv_bfloat16*) {
  return __float2bfloat16_rn(0.0f);
}
__device__ __forceinline__ float zero_of(float*) { return 0.0f; }

// conv.py:330  `acc / (1 + tl.exp(-acc))`, verbatim instruction sequence.
__device__ __forceinline__ float silu_f32(float acc) {
  float result;
  asm("{\n\t"
      ".reg .f32 nacc, scaled, evalue, denom;\n\t"
      "sub.f32        nacc,   0f00000000, %1;\n\t"
      "mul.f32        scaled, nacc, 0f3FB8AA3B;\n\t"
      "ex2.approx.f32 evalue, scaled;\n\t"
      "add.f32        denom,  evalue, 0f3F800000;\n\t"
      "div.full.f32   %0,     %1, denom;\n\t"
      "}"
      : "=f"(result)
      : "f"(acc));
  return result;
}

// The 35 kernel parameters, in upstream order.  Only the five `T*` pointers are
// erased to void*, because one struct has to serve every entry while T is a
// template argument; the body re-types them in its alias block.
struct ConvParams {
  const void* x_ptr;
  const void* w_ptr;
  const void* bias_ptr;
  void* conv_state_ptr;
  const int32_t* __restrict__ conv_state_indices_ptr;
  const int32_t* __restrict__ num_accepted_tokens_ptr;
  const int32_t* __restrict__ query_start_loc_ptr;
  const int32_t* __restrict__ block_idx_last_scheduled_token;
  const int32_t* __restrict__ initial_state_idx;
  void* o_ptr;
  int batch;
  int dim;
  int seqlen_in;
  int state_len_in;
  int num_cache_lines;
  int64_t stride_x_seq;
  int64_t stride_x_dim;
  int64_t stride_x_token;
  int64_t stride_w_dim;
  int64_t stride_w_width;
  int64_t stride_conv_state_seq;
  int64_t stride_conv_state_dim;
  int64_t stride_conv_state_tok;
  int64_t stride_state_indices;
  int64_t stride_o_seq;
  int64_t stride_o_dim;
  int64_t stride_o_token;
  int64_t null_block_id;
  bool has_bias;
  bool silu_activation;
  bool is_varlen;
  bool is_apc_enabled;
  bool is_spec_decoding;
  bool has_null_block;
  bool launch_pdl;
};

// The one maintained kernel body.  KERNEL_WIDTH sizes the unrolled tap loop and
// NP2_STATELEN sizes the register array holding the rolling conv history.
template <class T, int KERNEL_WIDTH, int NP2_STATELEN, bool PRODUCTION = false>
__device__ __forceinline__ void causal_conv1d_update_body(
    const ConvParams& p) {
  // One alias per upstream kernel parameter, each re-declared with the upstream
  // type including its __restrict__ and const spelling.  This block is the whole
  // reason the body below can stay byte-identical to the verified port: every name
  // the body uses still exists, with the same type and the same qualifiers.
  const T* __restrict__ x_ptr = static_cast<const T*>(p.x_ptr);
  const T* __restrict__ w_ptr = static_cast<const T*>(p.w_ptr);
  const T* __restrict__ bias_ptr = static_cast<const T*>(p.bias_ptr);
  T* __restrict__ conv_state_ptr = static_cast<T*>(p.conv_state_ptr);
  const int32_t* __restrict__ conv_state_indices_ptr = p.conv_state_indices_ptr;
  const int32_t* __restrict__ num_accepted_tokens_ptr = p.num_accepted_tokens_ptr;
  const int32_t* __restrict__ query_start_loc_ptr = p.query_start_loc_ptr;
  const int32_t* __restrict__ block_idx_last_scheduled_token = p.block_idx_last_scheduled_token;
  const int32_t* __restrict__ initial_state_idx = p.initial_state_idx;
  T* __restrict__ o_ptr = static_cast<T*>(p.o_ptr);
  int batch = p.batch;
  int dim = p.dim;
  int seqlen_in = p.seqlen_in;
  int state_len_in = p.state_len_in;
  int num_cache_lines = p.num_cache_lines;
  int64_t stride_x_seq = p.stride_x_seq;
  int64_t stride_x_dim = PRODUCTION ? 1 : p.stride_x_dim;
  int64_t stride_x_token = p.stride_x_token;
  int64_t stride_w_dim = PRODUCTION ? 4 : p.stride_w_dim;
  int64_t stride_w_width = PRODUCTION ? 1 : p.stride_w_width;
  int64_t stride_conv_state_seq = p.stride_conv_state_seq;
  int64_t stride_conv_state_dim = PRODUCTION ? 1 : p.stride_conv_state_dim;
  int64_t stride_conv_state_tok = p.stride_conv_state_tok;
  int64_t stride_state_indices = p.stride_state_indices;
  int64_t stride_o_seq = p.stride_o_seq;
  int64_t stride_o_dim = PRODUCTION ? 1 : p.stride_o_dim;
  int64_t stride_o_token = p.stride_o_token;
  int64_t null_block_id = PRODUCTION ? 0 : p.null_block_id;
  bool has_bias = PRODUCTION ? false : p.has_bias;
  bool silu_activation = PRODUCTION ? true : p.silu_activation;
  bool is_varlen = PRODUCTION ? true : p.is_varlen;
  bool is_apc_enabled = PRODUCTION ? false : p.is_apc_enabled;
  bool is_spec_decoding = PRODUCTION ? true : p.is_spec_decoding;
  bool has_null_block = PRODUCTION ? true : p.has_null_block;
  bool launch_pdl = PRODUCTION ? true : p.launch_pdl;
  if (launch_pdl) {
    gdc_wait();
  }

  const int idx_seq = blockIdx.x;
  if (idx_seq >= batch) {
    if (launch_pdl) {
      gdc_launch_dependents();
    }
    return;
  }

  const int idx_feat = blockIdx.y * kBlockN + threadIdx.x;
  const bool feat_ok = idx_feat < dim;

  int conv_state_init = 0;
  int current_last_index = 0;
  if (is_apc_enabled) {
    conv_state_init = initial_state_idx[idx_seq];
    current_last_index = block_idx_last_scheduled_token[idx_seq];
  }

  const int64_t coord = static_cast<int64_t>(
      conv_state_indices_ptr[idx_seq * stride_state_indices + conv_state_init]);

  if (has_null_block && coord == null_block_id) {
    if (launch_pdl) {
      gdc_launch_dependents();
    }
    return;
  }

  int seqlen = seqlen_in;
  int state_len = state_len_in;
  int64_t query_start_index = 0;
  int64_t query_end_index = 0;
  int64_t x_offset = 0;
  int64_t o_offset = 0;
  if (is_varlen) {
    query_start_index = static_cast<int64_t>(query_start_loc_ptr[idx_seq]);
    query_end_index = static_cast<int64_t>(query_start_loc_ptr[idx_seq + 1]);
    // revise state_len and seqlen
    state_len = state_len - (seqlen - static_cast<int>(query_end_index - query_start_index));
    seqlen = static_cast<int>(query_end_index - query_start_index);
    x_offset = query_start_index * stride_x_token;
    o_offset = query_start_index * stride_o_token;
  } else {
    query_start_index = static_cast<int64_t>(idx_seq) * seqlen;
    query_end_index = query_start_index + seqlen;
    x_offset = idx_seq * stride_x_seq;
    o_offset = idx_seq * stride_o_seq;
  }

  if (query_start_index == query_end_index) {
    if (launch_pdl) {
      gdc_launch_dependents();
    }
    return;
  }

  const int64_t token_offset =
      is_spec_decoding ? (static_cast<int64_t>(num_accepted_tokens_ptr[idx_seq]) - 1) : 0;

  // STEP 1: READ init_state data (KERNEL_WIDTH-1 history columns)
  // `= {}` on purpose: under torch's classic -D__CUDA_NO_BFLOAT16_CONVERSIONS__ the
  // __nv_bfloat16 conversion constructors (including the defaulted one) are compiled
  // out and the type becomes an aggregate, so a bare declaration would not compile.
  T col[KERNEL_WIDTH > 1 ? KERNEL_WIDTH - 1 : 1] = {};
#pragma unroll
  for (int j = 0; j < KERNEL_WIDTH - 1; ++j) {
    col[j] = zero_of(static_cast<T*>(nullptr));          // tl.load(..., other=0.0)
  }
  if (feat_ok) {
    const T* prior = conv_state_ptr + coord * stride_conv_state_seq +
                     static_cast<int64_t>(idx_feat) * stride_conv_state_dim +
                     token_offset * stride_conv_state_tok;
#pragma unroll
    for (int j = 0; j < KERNEL_WIDTH - 1; ++j) {
      col[j] = prior[j * stride_conv_state_tok];
    }
  }

  // STEP 2: assume state_len > seqlen
  const int val = state_len - seqlen;
  const T* x_base = x_ptr + x_offset + static_cast<int64_t>(idx_feat) * stride_x_dim;

  T new_conv_state[NP2_STATELEN] = {};
#pragma unroll
  for (int row = 0; row < NP2_STATELEN; ++row) {
    const bool keep_state = (coord < num_cache_lines) && ((row + seqlen) < state_len) && feat_ok;
    T state_value = zero_of(static_cast<T*>(nullptr));
    if (keep_state) {
      state_value = conv_state_ptr[coord * stride_conv_state_seq +
                                   token_offset * stride_conv_state_tok +
                                   static_cast<int64_t>(idx_feat) * stride_conv_state_dim +
                                   (static_cast<int64_t>(row) +
                                    (is_spec_decoding ? 1 : seqlen)) *
                                       stride_conv_state_tok];
    }
    const int x_token = row - val;
    T x_value = zero_of(static_cast<T*>(nullptr));
    if (x_token >= 0 && x_token < seqlen && feat_ok) {
      x_value = x_base[x_token * stride_x_token];
    }
    new_conv_state[row] = keep_state ? state_value : x_value;   // tl.where(mask, state, x)
  }

  // tl.debug_barrier().  Safe despite the three early returns above: every one of
  // them depends only on blockIdx.x / per-sequence scalars, never on threadIdx, so a
  // CTA either reaches this barrier with all kBlockN threads or has already exited whole.
  __syncthreads();

  {
    const int64_t target_coord = static_cast<int64_t>(
        conv_state_indices_ptr[idx_seq * stride_state_indices + current_last_index]);
    T* target = conv_state_ptr + target_coord * stride_conv_state_seq +
                static_cast<int64_t>(idx_feat) * stride_conv_state_dim;
#pragma unroll
    for (int row = 0; row < NP2_STATELEN; ++row) {
      if (row < state_len && feat_ok) {
        target[row * stride_conv_state_tok] = new_conv_state[row];
      }
    }
  }

  // STEP 3: init accumulator
  float acc_preload = 0.0f;
  if (has_bias && feat_ok) {
    acc_preload = to_f32(bias_ptr[idx_feat]);
  }

  // STEP 4: PRE-LOAD WEIGHTS
  T w_col[KERNEL_WIDTH] = {};
#pragma unroll
  for (int j = 0; j < KERNEL_WIDTH; ++j) {
    w_col[j] = zero_of(static_cast<T*>(nullptr));
  }
  if (feat_ok) {
    const T* w_base = w_ptr + static_cast<int64_t>(idx_feat) * stride_w_dim;
#pragma unroll
    for (int j = 0; j < KERNEL_WIDTH; ++j) {
      w_col[j] = w_base[j * stride_w_width];
    }
  }

  // STEP 5: compute each token
  if (launch_pdl) {
    gdc_launch_dependents();
  }

  for (int idx_token = 0; idx_token < seqlen; ++idx_token) {
    float acc = acc_preload;
#pragma unroll
    for (int j = 0; j < KERNEL_WIDTH - 1; ++j) {
      acc = acc_add(acc, col[j], w_col[j]);
    }
    T x_now = zero_of(static_cast<T*>(nullptr));
    if (feat_ok) {
      x_now = x_base[idx_token * stride_x_token];
    }
    acc = acc_add(acc, x_now, w_col[KERNEL_WIDTH - 1]);

#pragma unroll
    for (int j = 0; j < KERNEL_WIDTH - 2; ++j) {
      col[j] = col[j + 1];
    }
    col[KERNEL_WIDTH - 2] = x_now;

    if (silu_activation) {
      acc = silu_f32(acc);
    }
    if (feat_ok) {
      o_ptr[o_offset + static_cast<int64_t>(idx_token) * stride_o_token +
            static_cast<int64_t>(idx_feat) * stride_o_dim] =
          from_f32(static_cast<T*>(nullptr), acc);
    }
  }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Copyright (c) 2024, Tri Dao.
// Production metadata specialization: 128 threads own 256 adjacent BF16 features,
// preserving the original Triton BLOCK_N=256 / four-warp mapping. FP32 products
// are the separately identified correctness repair, identical to corrected S.
// Generic layouts and other convolution widths continue through the old body.
__device__ __forceinline__ uint32_t conv_load_pair(const __nv_bfloat16* p) {
  return *reinterpret_cast<const uint32_t*>(p);
}
__device__ __forceinline__ void conv_store_pair(__nv_bfloat16* p, uint32_t v) {
  *reinterpret_cast<uint32_t*>(p) = v;
}
__device__ __forceinline__ float2 conv_pair_float(uint32_t v) {
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v));
}
__device__ __forceinline__ float2 conv_pair_acc(float2 acc, uint32_t x, uint32_t w) {
  float2 a = conv_pair_float(x), b = conv_pair_float(w);
  return make_float2(__fadd_rn(acc.x,__fmul_rn(a.x,b.x)),
                    __fadd_rn(acc.y,__fmul_rn(a.y,b.y)));
}

template<int NP2_STATELEN>
__device__ __forceinline__ void causal_conv1d_update_pair(const ConvParams& p) {
  gdc_wait();
  const int seq=blockIdx.x;
  if (seq>=p.batch) { gdc_launch_dependents(); return; }
  const int feat=blockIdx.y*kBlockN+threadIdx.x*2;
  const bool active=feat<p.dim;
  const int64_t slot=p.conv_state_indices_ptr[seq*p.stride_state_indices];
  const int64_t start=p.query_start_loc_ptr[seq],end=p.query_start_loc_ptr[seq+1];
  if (slot==0 || start==end) { gdc_launch_dependents(); return; }
  const int length=static_cast<int>(end-start);
  const int state_length=p.state_len_in-(p.seqlen_in-length);
  const int64_t accepted_offset=p.num_accepted_tokens_ptr[seq]-1;
  const auto* x=static_cast<const __nv_bfloat16*>(p.x_ptr)+start*p.stride_x_token+feat;
  auto* state=static_cast<__nv_bfloat16*>(p.conv_state_ptr)+slot*p.stride_conv_state_seq+feat;
  const auto* history=state+accepted_offset*p.stride_conv_state_tok;
  const auto* weights=static_cast<const __nv_bfloat16*>(p.w_ptr);
  auto* out=static_cast<__nv_bfloat16*>(p.o_ptr)+start*p.stride_o_token+feat;
  uint32_t col[3]={0,0,0};
#pragma unroll
  for (int j=0;j<3;++j) if(active)col[j]=conv_load_pair(history+j*p.stride_conv_state_tok);
  uint32_t next_state[NP2_STATELEN]={};
#pragma unroll
  for(int row=0;row<NP2_STATELEN;++row) {
    const bool keep=slot<p.num_cache_lines && row+length<state_length && active;
    uint32_t old=0,now=0;
    if(keep)old=conv_load_pair(history+(row+1)*p.stride_conv_state_tok);
    const int token=row-(state_length-length);
    if(token>=0 && token<length && active)now=conv_load_pair(x+token*p.stride_x_token);
    next_state[row]=keep?old:now;
  }
  __syncthreads();
#pragma unroll
  for(int row=0;row<NP2_STATELEN;++row)
    if(row<state_length && active)conv_store_pair(state+row*p.stride_conv_state_tok,next_state[row]);
  uint32_t weight[4]={};
  if(active) {
    const uint2 a=*reinterpret_cast<const uint2*>(weights+feat*4);
    const uint2 b=*reinterpret_cast<const uint2*>(weights+(feat+1)*4);
    uint32_t wa[2]={a.x,a.y},wb[2]={b.x,b.y};
#pragma unroll
    for(int j=0;j<4;++j)weight[j]=((wa[j/2]>>(16*(j&1)))&65535u) | ((wb[j/2]>>(16*(j&1)))<<16);
  }
  gdc_launch_dependents();
  for(int token=0;token<length;++token) {
    float2 acc=make_float2(0.0f,0.0f);
#pragma unroll
    for(int j=0;j<3;++j)acc=conv_pair_acc(acc,col[j],weight[j]);
    uint32_t now=active?conv_load_pair(x+token*p.stride_x_token):0;
    acc=conv_pair_acc(acc,now,weight[3]);
    col[0]=col[1];col[1]=col[2];col[2]=now;
    acc.x=silu_f32(acc.x);acc.y=silu_f32(acc.y);
    __nv_bfloat162 result=__floats2bfloat162_rn(acc.x,acc.y);
    if(active)conv_store_pair(out+token*p.stride_o_token,*reinterpret_cast<uint32_t*>(&result));
  }
}

// Paired-feature production path and generic layout fallback retain their launch bounds.
template<int StateLength>
__global__ void __launch_bounds__(128) causal_conv_pair(ConvParams p) {
  causal_conv1d_update_pair<StateLength>(p);
}
template<int Width, int StateLength>
__global__ void __launch_bounds__(kBlockN) causal_conv_generic(ConvParams p) {
  causal_conv1d_update_body<__nv_bfloat16, Width, StateLength>(p);
}
}  // namespace

}  // namespace sfc_conv_pure
