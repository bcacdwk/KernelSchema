/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 */

// FMHA-COMPAT host layer -- the graded route's `LaunchCompat`, rewritten framework-free.
//
// This file is a *mechanical* re-expression of the host half of
// `OLD/merged/FMHA_COMPAT_V1__ABS.cu` (region `[FILE csrc/cake_fmha/cake_fmha_jit_binding.cu]`,
// `namespace flashinfer::cake_fmha`). Geometry and mathematical arguments are
// retained; host domain gates extend HD64 and noncausal left windows below.
// Framework facts arrive through a thin local ABI:
//
//   before: `tvm::ffi::TensorView` / `Optional<TensorView>` / `Variant<double, Tensor>`, with
//           `TVM_FFI_ICHECK*` throwing, and `get_stream(device)` plus a `CUDADeviceGuard`.
//   after:  plain pointers, sizes, strides and dtype ids in `CompatArgs`, filled by a thin Torch
//           binding; failures come back as a `LaunchStatus` the binding turns into TORCH_CHECK.
//           The stream is passed in by the caller (the current Torch stream), and there is no
//           device guard because the binding already runs on the tensors' own device.
//
// Nothing here computes anything.  It validates, derives the launch geometry and the scalar
// arguments, and calls the launcher in `kernel.cuh`.
//
// `ScalarScale` is gone on purpose: the pinned route accepted either a host double or a device
// tensor for bmm1_scale/bmm2_scale and *threw* if handed a tensor, demanding the caller
// materialise it first.  The problem always passes a Python float, so the accepted type here is
// simply `double` -- the tensor branch was unreachable and rejecting it was the whole behaviour.

#ifndef CAKE_FMHA_COMPAT_V1_HOST_CUH_
#define CAKE_FMHA_COMPAT_V1_HOST_CUH_

#include <cuda_runtime_api.h>

#include "compat_kernel.cuh"

namespace fmha_compat {

// Paged-KV layout id and payload dtype ids.  Values must match what the kernel in kernel.cuh
// switches on; they are the graded route's own constants, not a new encoding.
constexpr int kHndLayout = 0;      // index = page*s0 + head*s1 + token*s2 + dim*s3
constexpr int kNhdLayout = 1;      // index = page*s0 + token*s1 + head*s2 + dim*s3
constexpr int kBFloat16 = 0;
constexpr int kFloat16 = 1;
constexpr int kFloat8E4M3 = 2;
constexpr int kNvFp4 = 3;          // packed: two e2m1 values per byte
// Ids for the auxiliary tensors, which are never NVFP4.  They continue the same numbering so one
// `cake_dtype` field can describe every argument; the kernel itself only switches on 0..3.
constexpr int kFloat32 = 4;        // attention_sinks, lse
constexpr int kInt32 = 5;          // q_indptr

// Plain metadata for one tensor argument.  `present == false` is the replacement for
// `Optional<TensorView>`; an absent tensor contributes a null pointer and no checks.
struct TensorMeta {
  const void* ptr = nullptr;
  int ndim = 0;
  long long size[4] = {0, 0, 0, 0};
  long long stride[4] = {0, 0, 0, 0};
  int cake_dtype = -1;
  bool present = false;

  long long dim(int i) const { return i < ndim ? size[i] : 0; }
  long long st(int i) const { return i < ndim ? stride[i] : 0; }
};

struct CompatArgs {
  TensorMeta out;
  TensorMeta out_scale_factor;
  TensorMeta query;
  TensorMeta key_cache;
  TensorMeta value_cache;
  TensorMeta block_tables;
  TensorMeta seq_lens;
  TensorMeta q_indptr;             // absent => uniform decode, built into workspace below
  TensorMeta attention_sinks;
  TensorMeta key_block_scales;     // required when the KV payload is packed NVFP4
  TensorMeta value_block_scales;
  TensorMeta lse;
  // Declared scratch, already checked by the binding to be a writable region of the problem's
  // own contract.  Only ever used to materialise a uniform q_indptr.
  const void* workspace_ptr = nullptr;
  long long workspace_bytes = 0;
  long long batch_size = 0;
  long long uniform_q_len = 0;     // > 0 selects the decode route's uniform indptr
  long long window_left = -1;
  bool is_causal = true;
  bool uses_shared_paged_kv_idx = true;
  double bmm1_scale = 0.0;         // sm_scale: softmax scale with q/k scales folded in
  double bmm2_scale = 0.0;         // v_scale
  double o_sf_scale = 0.0;
  long long o_sf_start_index = 0;
  long long lse_stride_tokens = 0;
  long long lse_stride_heads = 0;
  int kv_layout = kHndLayout;
};

// code == 0 on success.  Otherwise a cudaError_t value, plus a static message naming the
// precondition that failed -- the same text the graded route threw, so a refusal is attributable
// to the same rule it always was.
struct LaunchStatus {
  int code = 0;
  const char* message = "ok";
};

inline LaunchStatus reject(const char* message) {
  return LaunchStatus{static_cast<int>(cudaErrorInvalidValue), message};
}

#define FMHA_COMPAT_CHECK(cond, msg) \
  do { if (!(cond)) return reject(msg); } while (0)

// Validate and derive exactly what `LaunchCompat` did, then launch.
inline LaunchStatus launch_compat(const CompatArgs& args, cudaStream_t stream) {
  FMHA_COMPAT_CHECK(args.query.present && args.query.ndim == 3,
                    "query must be rank 3 [tokens, heads, dim]");
  FMHA_COMPAT_CHECK(args.key_cache.present && args.key_cache.ndim == 4,
                    "key cache must be rank 4 HND");
  FMHA_COMPAT_CHECK(args.value_cache.present && args.value_cache.ndim == 4,
                    "value cache must be rank 4 HND");
  FMHA_COMPAT_CHECK(args.out.present && args.out.ndim == 3, "output must be rank 3");
  FMHA_COMPAT_CHECK(args.query.st(2) == 1, "query head dimension must be contiguous");
  FMHA_COMPAT_CHECK(args.key_cache.st(3) == 1, "key cache head dimension must be contiguous");
  FMHA_COMPAT_CHECK(args.value_cache.st(3) == 1, "value cache head dimension must be contiguous");
  FMHA_COMPAT_CHECK(args.out.st(2) == 1, "output head dimension must be contiguous");
  FMHA_COMPAT_CHECK(args.batch_size > 0, "batch size must be positive");
  FMHA_COMPAT_CHECK(args.seq_lens.present && args.seq_lens.ndim == 1, "seq_lens must be rank 1");
  FMHA_COMPAT_CHECK(args.seq_lens.dim(0) == args.batch_size,
                    "seq_lens must contain one value per request");

  const int q_dtype = args.query.cake_dtype;
  const int kv_dtype = args.key_cache.cake_dtype;
  const int o_dtype = args.out.cake_dtype;
  FMHA_COMPAT_CHECK(q_dtype == kBFloat16 || q_dtype == kFloat16 || q_dtype == kFloat8E4M3,
                    "query must use bfloat16, float16 or float8_e4m3fn");
  FMHA_COMPAT_CHECK(kv_dtype == kBFloat16 || kv_dtype == kFloat16 || kv_dtype == kFloat8E4M3 ||
                        kv_dtype == kNvFp4,
                    "KV cache must use bfloat16, float16, float8_e4m3fn, or packed NVFP4 uint8");
  FMHA_COMPAT_CHECK(o_dtype == kBFloat16 || o_dtype == kFloat16 || o_dtype == kFloat8E4M3 ||
                        o_dtype == kNvFp4,
                    "output must use bfloat16, float16, float8_e4m3fn, or packed NVFP4 uint8");
  FMHA_COMPAT_CHECK(args.key_cache.cake_dtype == args.value_cache.cake_dtype,
                    "key and value cache dtypes must match");

  const int num_q_heads = static_cast<int>(args.query.dim(1));
  const int num_kv_heads = static_cast<int>(args.key_cache.dim(1));
  const int head_dim = static_cast<int>(args.query.dim(2));
  // A packed NVFP4 tensor stores two dimensions per byte, so the *logical* head dimension is
  // twice the physical last extent.  Comparing logical to logical is what makes the next two
  // checks mean the same thing for every dtype.
  const int kv_head_dim = static_cast<int>(args.key_cache.dim(3)) * (kv_dtype == kNvFp4 ? 2 : 1);
  const int out_head_dim = static_cast<int>(args.out.dim(2)) * (o_dtype == kNvFp4 ? 2 : 1);
  const int page_size = static_cast<int>(args.key_cache.dim(2));

  // The CUDA body masks every dimension load/store and reduces a fixed 256 lanes.
  // HD64 therefore uses the same arithmetic with 192 neutral lanes. Keep this
  // extension explicit: the upstream compatibility gate admitted only HD128/256.
  FMHA_COMPAT_CHECK(head_dim == 64 || head_dim == 128 || head_dim == 256,
                    "Pure CUDA FMHA supports head_dim 64, 128 or 256");
  FMHA_COMPAT_CHECK(kv_head_dim == head_dim, "query and KV head dimensions must match");
  FMHA_COMPAT_CHECK(out_head_dim == head_dim, "query and output head dimensions must match");
  FMHA_COMPAT_CHECK(num_kv_heads > 0, "num_kv_heads must be positive");
  FMHA_COMPAT_CHECK(num_q_heads % num_kv_heads == 0,
                    "num_q_heads must be divisible by num_kv_heads");
  // Non-causal sliding windows impose only the left boundary, as in the frozen
  // reference. The kernel already applies this via kv_begin independently of
  // is_causal; retaining the upstream host rejection would incorrectly shrink the domain.
  FMHA_COMPAT_CHECK(args.block_tables.present, "block_tables must be present");

  const void* k_scales_ptr = nullptr;
  const void* v_scales_ptr = nullptr;
  long long ksf_s0 = 0, ksf_s1 = 0, ksf_s2 = 0, ksf_s3 = 0;
  long long vsf_s0 = 0, vsf_s1 = 0, vsf_s2 = 0, vsf_s3 = 0;
  if (kv_dtype == kNvFp4) {
    FMHA_COMPAT_CHECK(args.key_block_scales.present && args.value_block_scales.present,
                      "packed NVFP4 KV requires key and value scale tensors");
    const TensorMeta& ksf = args.key_block_scales;
    const TensorMeta& vsf = args.value_block_scales;
    FMHA_COMPAT_CHECK(ksf.ndim == 4, "key_block_scales must be rank 4");
    FMHA_COMPAT_CHECK(vsf.ndim == 4, "value_block_scales must be rank 4");
    FMHA_COMPAT_CHECK(ksf.cake_dtype == kFloat8E4M3, "key_block_scales must be float8_e4m3fn");
    FMHA_COMPAT_CHECK(vsf.cake_dtype == kFloat8E4M3, "value_block_scales must be float8_e4m3fn");
    k_scales_ptr = ksf.ptr;
    v_scales_ptr = vsf.ptr;
    ksf_s0 = ksf.st(0); ksf_s1 = ksf.st(1); ksf_s2 = ksf.st(2); ksf_s3 = ksf.st(3);
    vsf_s0 = vsf.st(0); vsf_s1 = vsf.st(1); vsf_s2 = vsf.st(2); vsf_s3 = vsf.st(3);
  }

  void* output_scales_ptr = nullptr;
  int output_scale_columns = 0;
  if (o_dtype == kNvFp4) {
    FMHA_COMPAT_CHECK(q_dtype == kFloat8E4M3, "packed NVFP4 output requires float8_e4m3 query");
    FMHA_COMPAT_CHECK(kv_dtype == kFloat8E4M3, "packed NVFP4 output requires float8_e4m3 KV");
    FMHA_COMPAT_CHECK(args.out_scale_factor.present,
                      "packed NVFP4 output requires an output scale tensor");
    const TensorMeta& osf = args.out_scale_factor;
    FMHA_COMPAT_CHECK(osf.ndim == 2, "out_scale_factor must be rank 2");
    FMHA_COMPAT_CHECK(osf.cake_dtype == kFloat8E4M3, "out_scale_factor must be float8_e4m3fn");
    FMHA_COMPAT_CHECK(osf.st(1) == 1, "out_scale_factor must be contiguous");
    FMHA_COMPAT_CHECK(args.o_sf_scale > 0.0, "o_sf_scale must be positive for NVFP4 output");
    output_scales_ptr = const_cast<void*>(osf.ptr);
    output_scale_columns = static_cast<int>(osf.dim(1));
  }

  const float* sinks_ptr = nullptr;
  if (args.attention_sinks.present) {
    FMHA_COMPAT_CHECK(args.attention_sinks.cake_dtype == kFloat32, "attention_sinks must be float32");
    FMHA_COMPAT_CHECK(args.attention_sinks.dim(0) == num_q_heads ||
                          args.attention_sinks.stride[0] == 1,
                      "attention_sinks must hold one value per query head");
    sinks_ptr = static_cast<const float*>(args.attention_sinks.ptr);
  }

  float* lse_ptr = nullptr;
  if (args.lse.present) {
    FMHA_COMPAT_CHECK(args.lse.cake_dtype == kFloat32, "lse must be float32");
    FMHA_COMPAT_CHECK(args.lse.ndim == 2, "lse must be rank 2");
    FMHA_COMPAT_CHECK(args.lse.dim(0) == args.query.dim(0), "lse must have one row per query token");
    FMHA_COMPAT_CHECK(args.lse.dim(1) == num_q_heads, "lse must have one column per query head");
    FMHA_COMPAT_CHECK(args.lse_stride_tokens == num_q_heads,
                      "Cake FMHA currently requires contiguous LSE");
    FMHA_COMPAT_CHECK(args.lse_stride_heads == 1,
                      "Cake FMHA currently requires contiguous LSE");
    lse_ptr = const_cast<float*>(static_cast<const float*>(args.lse.ptr));
  }

  const int* table_base = static_cast<const int*>(args.block_tables.ptr);
  const int* table_k = table_base;
  const int* table_v = table_base;
  const long long table_k_s0 = args.block_tables.st(0);
  long long table_v_s0 = table_k_s0;
  if (!args.uses_shared_paged_kv_idx) {
    FMHA_COMPAT_CHECK(args.block_tables.ndim == 3,
                      "separate K/V page tables must have shape [batch, 2, pages]");
    FMHA_COMPAT_CHECK(args.block_tables.dim(1) == 2,
                      "separate K/V page tables must have shape [batch, 2, pages]");
    table_v = table_base + args.block_tables.st(1);
  } else {
    FMHA_COMPAT_CHECK(args.block_tables.ndim == 2,
                      "shared page tables must have shape [batch, pages]");
  }

  const int* q_indptr_ptr = nullptr;
  if (args.q_indptr.present) {
    FMHA_COMPAT_CHECK(args.q_indptr.cake_dtype == kInt32, "q_indptr must be int32");
    FMHA_COMPAT_CHECK(args.q_indptr.ndim == 1, "q_indptr must be rank 1");
    FMHA_COMPAT_CHECK(args.q_indptr.dim(0) == args.batch_size + 1,
                      "q_indptr must have batch_size + 1 entries");
    FMHA_COMPAT_CHECK(args.q_indptr.st(0) == 1, "q_indptr must be contiguous");
    q_indptr_ptr = static_cast<const int*>(args.q_indptr.ptr);
  } else {
    // Decode route: the caller passes one q_len for every request, so the indptr is uniform and
    // is materialised into the problem's own declared scratch.  Writing it anywhere else, or
    // enlarging the caller's workspace to fit it, would both be contract violations.
    FMHA_COMPAT_CHECK(args.uniform_q_len > 0, "uniform_q_len must be positive without q_indptr");
    FMHA_COMPAT_CHECK(args.query.dim(0) == args.batch_size * args.uniform_q_len,
                      "uniform decode query rows must equal batch_size * q_len");
    const long long required_bytes = (args.batch_size + 1) * static_cast<long long>(sizeof(int));
    FMHA_COMPAT_CHECK(args.workspace_ptr != nullptr && args.workspace_bytes >= required_bytes,
                      "workspace is too small for Cake FMHA query indptr");
    q_indptr_ptr = static_cast<const int*>(const_cast<void*>(args.workspace_ptr));
    // This launcher reports a plain cudaError_t (it is a one-kernel fill with no preconditions of
    // its own), not a LaunchStatus, so the check compares against cudaSuccess.
    const cudaError_t filled = cake_fmha_fill_uniform_q_indptr(
        const_cast<int*>(q_indptr_ptr), static_cast<int>(args.batch_size),
        static_cast<int>(args.uniform_q_len), stream);
    FMHA_COMPAT_CHECK(filled == cudaSuccess,
                      "failed to initialize Cake FMHA uniform query indptr");
  }

  const cudaError_t status = cake_fmha_launch_compat_v1(
      args.query.ptr, args.key_cache.ptr, args.value_cache.ptr, k_scales_ptr, v_scales_ptr,
      const_cast<void*>(args.out.ptr), output_scales_ptr, lse_ptr, table_k, table_v, q_indptr_ptr,
      static_cast<const int*>(args.seq_lens.ptr), sinks_ptr, static_cast<int>(args.batch_size),
      num_q_heads, num_kv_heads, head_dim, page_size, args.kv_layout, q_dtype, kv_dtype, o_dtype,
      static_cast<int>(args.is_causal), static_cast<int>(args.window_left),
      static_cast<int>(sinks_ptr != nullptr), static_cast<int>(lse_ptr != nullptr),
      // q_scale/k_scale/o_scale are 1.0 because the problem folds the query and key scaling into
      // bmm1_scale and the value scaling into bmm2_scale; the graded route passed the same ones.
      1.0f, 1.0f, static_cast<float>(args.bmm2_scale), 1.0f, static_cast<float>(args.bmm1_scale),
      o_dtype == kNvFp4 ? static_cast<float>(args.o_sf_scale) : 1.0f,
      static_cast<int>(args.o_sf_start_index), output_scale_columns, args.query.st(0),
      args.query.st(1), args.key_cache.st(0), args.key_cache.st(1), args.key_cache.st(2),
      args.key_cache.st(3), args.value_cache.st(0), args.value_cache.st(1), args.value_cache.st(2),
      args.value_cache.st(3), ksf_s0, ksf_s1, ksf_s2, ksf_s3, vsf_s0, vsf_s1, vsf_s2, vsf_s3,
      table_k_s0, table_v_s0, args.out.st(0), args.out.st(1),
      static_cast<unsigned int>(args.query.dim(0)), static_cast<unsigned int>(num_q_heads), stream);
  FMHA_COMPAT_CHECK(status == cudaSuccess, "Cake FMHA compatibility launch failed");
  return LaunchStatus{0, "ok"};
}

#undef FMHA_COMPAT_CHECK

}  // namespace fmha_compat

#endif  // CAKE_FMHA_COMPAT_V1_HOST_CUH_
