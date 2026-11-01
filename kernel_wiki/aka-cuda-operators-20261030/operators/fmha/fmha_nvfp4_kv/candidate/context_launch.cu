/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/container/variant.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "context_cake_fmha.cuh"
#include "host.cuh"

#ifndef NUM_M_BLOCKS
#error "NUM_M_BLOCKS must be supplied by the route-specific JIT"
#endif
#ifndef NUM_Q_HEADS
#error "NUM_Q_HEADS must be supplied by the route-specific JIT"
#endif
#ifndef HEADS_PER_GROUP
#error "HEADS_PER_GROUP must be supplied by the route-specific JIT"
#endif
#ifndef PACK_G
#error "PACK_G must be supplied by the route-specific JIT"
#endif
#ifndef TOK_PER_STAGE
#error "TOK_PER_STAGE must be supplied by the route-specific JIT"
#endif
#ifndef PAGE_SIZE
#error "PAGE_SIZE must be supplied by the route-specific JIT"
#endif
#ifndef CAKE_FMHA_CONTEXT_IS_CAUSAL
#error "CAKE_FMHA_CONTEXT_IS_CAUSAL must be supplied by the route-specific JIT"
#endif
#ifndef CAKE_FMHA_CONTEXT_RETURN_LSE
#error "CAKE_FMHA_CONTEXT_RETURN_LSE must be supplied by the route-specific JIT"
#endif
#ifndef CAKE_FMHA_CONTEXT_ENABLE_SINK
#error "CAKE_FMHA_CONTEXT_ENABLE_SINK must be supplied by the route-specific JIT"
#endif
#ifndef CAKE_FMHA_CONTEXT_NVFP4
#define CAKE_FMHA_CONTEXT_NVFP4 0
#endif

using tvm::ffi::Optional;
using tvm::ffi::Variant;

// Kernel launch configuration for this route.
#include <cuda.h>
/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "context_cake_fmha.cuh"

extern "C" __global__ void kernel_cake_fmha_context_fp8(const __grid_constant__ CUtensorMap Q_value, const __grid_constant__ CUtensorMap K_value, const __grid_constant__ CUtensorMap V_value, __nv_bfloat16* O_ptr, float* LSE_ptr, float* sinks, int* page_table_k, int* page_table_v, int* seq_lens_q, int* seq_lens_kv, int* cu_seq_lens_q, float softmax_scale_log2, float output_scale, int total_bh, int page_row_stride, int num_ctas, uint32_t* dynamic_counter);

extern "C" cudaError_t cake_fmha_launch_context_fp8(
    CakeFmhaTensorMap const* Q,
    CakeFmhaTensorMap const* K,
    CakeFmhaTensorMap const* V,
    __nv_bfloat16* O_ptr,
    float* LSE_ptr,
    float* sinks,
    int* page_table_k,
    int* page_table_v,
    int* seq_lens_q,
    int* seq_lens_kv,
    int* cu_seq_lens_q,
    float softmax_scale_log2,
    float output_scale,
    int total_bh,
    int page_row_stride,
    int num_ctas,
    uint32_t* dynamic_counter,
    unsigned int grid_x,
    unsigned int grid_y,
    unsigned int grid_z,
    cudaStream_t stream) {
    cudaError_t status = cudaFuncSetAttribute(
        reinterpret_cast<const void*>(kernel_cake_fmha_context_fp8),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        86528);
    if (status != cudaSuccess) {
        return status;
    }
    void* kernel_args[] = {
        const_cast<void*>(reinterpret_cast<const void*>(Q)),
        const_cast<void*>(reinterpret_cast<const void*>(K)),
        const_cast<void*>(reinterpret_cast<const void*>(V)),
        const_cast<void*>(reinterpret_cast<const void*>(&O_ptr)),
        const_cast<void*>(reinterpret_cast<const void*>(&LSE_ptr)),
        const_cast<void*>(reinterpret_cast<const void*>(&sinks)),
        const_cast<void*>(reinterpret_cast<const void*>(&page_table_k)),
        const_cast<void*>(reinterpret_cast<const void*>(&page_table_v)),
        const_cast<void*>(reinterpret_cast<const void*>(&seq_lens_q)),
        const_cast<void*>(reinterpret_cast<const void*>(&seq_lens_kv)),
        const_cast<void*>(reinterpret_cast<const void*>(&cu_seq_lens_q)),
        const_cast<void*>(reinterpret_cast<const void*>(&softmax_scale_log2)),
        const_cast<void*>(reinterpret_cast<const void*>(&output_scale)),
        const_cast<void*>(reinterpret_cast<const void*>(&total_bh)),
        const_cast<void*>(reinterpret_cast<const void*>(&page_row_stride)),
        const_cast<void*>(reinterpret_cast<const void*>(&num_ctas)),
        const_cast<void*>(reinterpret_cast<const void*>(&dynamic_counter))
    };
    return cudaLaunchKernel(
        reinterpret_cast<const void*>(kernel_cake_fmha_context_fp8),
        dim3(grid_x, grid_y, grid_z),
        dim3(512, 1, 1),
        kernel_args,
        86528,
        stream);
}

namespace flashinfer {
namespace cake_fmha {
namespace {

using tvm::ffi::TensorView;



double ScalarScale(Variant<double, ffi::Tensor> scale, const char* name) {
  auto scalar = scale.as<double>();
  TVM_FFI_ICHECK(scalar.has_value()) << name << " must be a host scalar on this specialization";
  return scalar.value();
}


CUtensorMap EncodeTmaQ(TensorView tensor) {
  TVM_FFI_ICHECK_EQ(tensor.ndim(), 3);
  TVM_FFI_ICHECK_EQ(tensor.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK_EQ(tensor.size(1), NUM_Q_HEADS);
  TVM_FFI_ICHECK_EQ(tensor.size(2), D06_OUTPUT_HEAD_DIM);
  TVM_FFI_ICHECK_EQ(tensor.stride(2), 1);
  uint64_t global_dim[5] = {D06_OUTPUT_HEAD_DIM, PACK_G, static_cast<uint64_t>(tensor.size(0)), 1u,
                            NUM_Q_HEADS / PACK_G};
  uint64_t global_strides[4] = {static_cast<uint64_t>(tensor.stride(1)),
                                static_cast<uint64_t>(tensor.stride(0)), 128u,
                                static_cast<uint64_t>(tensor.stride(1) * PACK_G)};
  uint32_t box_dim[5] = {128u, PACK_G, TOK_PER_STAGE, 1u, 1u};
  uint32_t elem_strides[5] = {1u, 1u, 1u, 1u, 1u};
  CUtensorMap tm;
  CUresult result = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 5, tensor.data_ptr(), global_dim, global_strides, box_dim,
      elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS) << "failed to encode Cake FMHA context query map";
  return tm;
}

CUtensorMap EncodeTmaPagedKv(TensorView tensor, const char* name) {
  TVM_FFI_ICHECK_EQ(tensor.ndim(), 4) << name << " must be rank-4 HND paged KV";
  TVM_FFI_ICHECK_EQ(tensor.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK_EQ(tensor.size(1), NUM_Q_HEADS / HEADS_PER_GROUP);
  TVM_FFI_ICHECK_EQ(tensor.size(2), PAGE_SIZE);
  TVM_FFI_ICHECK_EQ(tensor.size(3), D06_OUTPUT_HEAD_DIM);
  TVM_FFI_ICHECK_EQ(tensor.stride(3), 1);
  uint64_t global_dim[5] = {D06_OUTPUT_HEAD_DIM, PAGE_SIZE, 1u, static_cast<uint64_t>(tensor.size(1)),
                            static_cast<uint64_t>(tensor.size(0))};
  uint64_t global_strides[4] = {static_cast<uint64_t>(tensor.stride(2)), 128u,
                                static_cast<uint64_t>(tensor.stride(1)),
                                static_cast<uint64_t>(tensor.stride(0))};
  uint32_t box_dim[5] = {128u, 16u, 1u, 1u, 1u};
  uint32_t elem_strides[5] = {1u, 1u, 1u, 1u, 1u};
  CUtensorMap tm;
  CUresult result = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 5, tensor.data_ptr(), global_dim, global_strides, box_dim,
      elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS)
      << "failed to encode Cake FMHA context " << name << " map";
  return tm;
}


__global__ void PrepareContextMetadata(const int* cum_seq_lens_q, const int* seq_lens_kv,
                                       int* expanded_q, int* expanded_kv, int* expanded_cu_q,
                                       int total_bh, int units_per_batch) {
  int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < total_bh) {
    int batch = index / units_per_batch;
    expanded_q[index] = cum_seq_lens_q[batch + 1] - cum_seq_lens_q[batch];
    expanded_kv[index] = seq_lens_kv[batch];
    expanded_cu_q[index] = cum_seq_lens_q[batch];
  }
}

int64_t AlignUp(int64_t value, int64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

}  // namespace

void cake_paged_attention_context(
    TensorView out, Optional<TensorView> out_scale_factor, TensorView query, TensorView key_cache,
    TensorView value_cache, TensorView workspace_buffer, TensorView multi_ctas_kv_counter_buffer,
    TensorView block_tables, TensorView seq_lens, int64_t max_q_len, int64_t max_kv_len,
    Variant<double, ffi::Tensor> bmm1_scale, Variant<double, ffi::Tensor> bmm2_scale,
    double o_sf_scale, int64_t o_sf_vec_size, int64_t o_sf_start_index, int64_t batch_size,
    int64_t window_left, TensorView cum_seq_lens_q, TensorView cum_seq_lens_kv, int64_t sm_count,
    bool enable_pdl, int64_t workspace_size, Optional<TensorView> attention_sinks,
    Optional<TensorView> key_block_scales, Optional<TensorView> value_block_scales,
    Optional<float> skip_softmax_threshold_scale_factor, Optional<bool> uses_shared_paged_kv_idx,
    Optional<bool> use_fp16_softmax, Optional<bool> uses_spcompress, bool is_causal,
    Optional<TensorView> lse, int64_t lse_stride_tokens, int64_t lse_stride_heads, int64_t stream_handle) {
  TVM_FFI_ICHECK_EQ(query.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK_EQ(key_cache.dtype(),
                    dl_float8_e4m3fn
  );
  TVM_FFI_ICHECK_EQ(value_cache.dtype(), key_cache.dtype());
  TVM_FFI_ICHECK_EQ(out.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK(!out_scale_factor.has_value());
  TVM_FFI_ICHECK(!key_block_scales.has_value() && !value_block_scales.has_value());
  TVM_FFI_ICHECK_EQ(skip_softmax_threshold_scale_factor.value_or(0.0f), 0.0f);
  TVM_FFI_ICHECK(!use_fp16_softmax.value_or(false));
  TVM_FFI_ICHECK(!uses_spcompress.value_or(false));
  TVM_FFI_ICHECK_EQ(window_left, D06_WINDOW_LEFT);
  TVM_FFI_ICHECK_EQ(o_sf_scale, -1.0);
  TVM_FFI_ICHECK_EQ(o_sf_vec_size, -1);
  TVM_FFI_ICHECK_EQ(o_sf_start_index, 0);
  float output_scale = static_cast<float>(ScalarScale(bmm2_scale, "bmm2_scale"));
  TVM_FFI_ICHECK_EQ(is_causal ? 1 : 0, CAKE_FMHA_CONTEXT_IS_CAUSAL);
  TVM_FFI_ICHECK_EQ(attention_sinks.has_value() ? 1 : 0, CAKE_FMHA_CONTEXT_ENABLE_SINK);
  TVM_FFI_ICHECK_EQ(lse.has_value() ? 1 : 0, CAKE_FMHA_CONTEXT_RETURN_LSE);
  TVM_FFI_ICHECK_GT(batch_size, 0);
  TVM_FFI_ICHECK_GT(max_q_len, 0);
  TVM_FFI_ICHECK_GT(max_kv_len, 0);
  TVM_FFI_ICHECK_GT(sm_count, 0);

  TVM_FFI_ICHECK_EQ(query.ndim(), 3);
  TVM_FFI_ICHECK_EQ(query.size(1), NUM_Q_HEADS);
  TVM_FFI_ICHECK_EQ(query.size(2), D06_OUTPUT_HEAD_DIM);
  TVM_FFI_ICHECK_EQ(query.stride(2), 1);
  TVM_FFI_ICHECK_EQ(out.ndim(), 3);
  TVM_FFI_ICHECK_EQ(out.size(0), query.size(0));
  TVM_FFI_ICHECK_EQ(out.size(1), query.size(1));
  TVM_FFI_ICHECK_EQ(out.size(2), D06_OUTPUT_HEAD_DIM);
  TVM_FFI_ICHECK(out.IsContiguous());
  TVM_FFI_ICHECK_EQ(key_cache.ndim(), 4);
  TVM_FFI_ICHECK_EQ(value_cache.ndim(), 4);
  TVM_FFI_ICHECK_EQ(key_cache.size(0), value_cache.size(0));
  TVM_FFI_ICHECK_EQ(key_cache.size(1), NUM_Q_HEADS / HEADS_PER_GROUP);
  TVM_FFI_ICHECK_EQ(value_cache.size(1), key_cache.size(1));
  TVM_FFI_ICHECK_EQ(key_cache.size(2), PAGE_SIZE);
  TVM_FFI_ICHECK_EQ(value_cache.size(2), PAGE_SIZE);
  TVM_FFI_ICHECK_EQ(key_cache.size(3), D06_OUTPUT_HEAD_DIM);
  TVM_FFI_ICHECK_EQ(value_cache.size(3), D06_OUTPUT_HEAD_DIM);
  TVM_FFI_ICHECK_EQ(key_cache.stride(3), 1);
  TVM_FFI_ICHECK_EQ(value_cache.stride(3), 1);

  TVM_FFI_ICHECK_EQ(seq_lens.ndim(), 1);
  TVM_FFI_ICHECK_EQ(seq_lens.size(0), batch_size);
  TVM_FFI_ICHECK_EQ(seq_lens.dtype(), dl_int32);
  TVM_FFI_ICHECK(seq_lens.IsContiguous());
  TVM_FFI_ICHECK_EQ(cum_seq_lens_q.ndim(), 1);
  TVM_FFI_ICHECK_EQ(cum_seq_lens_q.size(0), batch_size + 1);
  TVM_FFI_ICHECK_EQ(cum_seq_lens_q.dtype(), dl_int32);
  TVM_FFI_ICHECK(cum_seq_lens_q.IsContiguous());
  TVM_FFI_ICHECK_EQ(cum_seq_lens_kv.ndim(), 1);
  TVM_FFI_ICHECK_EQ(cum_seq_lens_kv.size(0), batch_size + 1);
  TVM_FFI_ICHECK_EQ(cum_seq_lens_kv.dtype(), dl_int32);
  TVM_FFI_ICHECK(cum_seq_lens_kv.IsContiguous());
  TVM_FFI_ICHECK(block_tables.dtype() == dl_int32 || block_tables.dtype() == dl_uint32);
  TVM_FFI_ICHECK(workspace_buffer.IsContiguous());
  TVM_FFI_ICHECK(multi_ctas_kv_counter_buffer.IsContiguous());

  CheckSameDevice(query, key_cache, "key_cache");
  CheckSameDevice(query, value_cache, "value_cache");
  CheckSameDevice(query, out, "out");
  CheckSameDevice(query, workspace_buffer, "workspace_buffer");
  CheckSameDevice(query, multi_ctas_kv_counter_buffer, "multi_ctas_kv_counter_buffer");
  CheckSameDevice(query, block_tables, "block_tables");
  CheckSameDevice(query, seq_lens, "seq_lens");
  CheckSameDevice(query, cum_seq_lens_q, "cum_seq_lens_q");
  CheckSameDevice(query, cum_seq_lens_kv, "cum_seq_lens_kv");
  if (attention_sinks.has_value()) {
    CheckSameDevice(query, attention_sinks.value(), "attention_sinks");
  }
  if (lse.has_value()) CheckSameDevice(query, lse.value(), "lse");

  int* table_base = static_cast<int*>(block_tables.data_ptr());
  int* table_k = table_base;
  int* table_v = table_base;
  int64_t page_row_stride = block_tables.stride(0);
  if (uses_shared_paged_kv_idx.value_or(true)) {
    TVM_FFI_ICHECK_EQ(block_tables.ndim(), 2);
    TVM_FFI_ICHECK_EQ(block_tables.size(0), batch_size);
    TVM_FFI_ICHECK_EQ(block_tables.stride(1), 1);
  } else {
    TVM_FFI_ICHECK_EQ(block_tables.ndim(), 3);
    TVM_FFI_ICHECK_EQ(block_tables.size(0), batch_size);
    TVM_FFI_ICHECK_EQ(block_tables.size(1), 2);
    TVM_FFI_ICHECK_EQ(block_tables.stride(2), 1);
    table_v = table_base + block_tables.stride(1);
  }

  ffi::CUDADeviceGuard device_guard(query.device().device_id);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
  auto* workspace = static_cast<uint8_t*>(workspace_buffer.data_ptr());
  int64_t actual_workspace_bytes = workspace_buffer.numel() * get_element_size(workspace_buffer);
  CUtensorMap h_q = EncodeTmaQ(query);
  CUtensorMap h_k = EncodeTmaPagedKv(key_cache, "key_cache");
  CUtensorMap h_v = EncodeTmaPagedKv(value_cache, "value_cache");
  constexpr int units_per_batch = NUM_Q_HEADS / PACK_G;
  int64_t total_bh_64 = batch_size * units_per_batch;
  TVM_FFI_ICHECK_LE(total_bh_64, static_cast<int64_t>(INT32_MAX));
  int total_bh = static_cast<int>(total_bh_64);
  int64_t seq_q_offset = 0;
  int64_t seq_kv_offset =
      AlignUp(seq_q_offset + total_bh_64 * static_cast<int64_t>(sizeof(int)), 16);
  int64_t cu_q_offset = seq_kv_offset + total_bh_64 * static_cast<int64_t>(sizeof(int));
  int64_t cursor = AlignUp(cu_q_offset + total_bh_64 * static_cast<int64_t>(sizeof(int)), 16);
  TVM_FFI_ICHECK_GE(actual_workspace_bytes, cursor)
      << "Cake FMHA context metadata workspace requires " << cursor << " bytes";
  TVM_FFI_ICHECK_GE(workspace_size, cursor);
  int64_t counter_bytes =
      multi_ctas_kv_counter_buffer.numel() * get_element_size(multi_ctas_kv_counter_buffer);
  TVM_FFI_ICHECK_GE(counter_bytes, 2 * static_cast<int64_t>(sizeof(uint32_t)));

  auto* seq_q_expanded = reinterpret_cast<int*>(workspace + seq_q_offset);
  auto* seq_kv_expanded = reinterpret_cast<int*>(workspace + seq_kv_offset);
  auto* cu_q_expanded = reinterpret_cast<int*>(workspace + cu_q_offset);
  // Metadata was written by the preceding decode on the same stream.

  float* lse_ptr = reinterpret_cast<float*>(seq_q_expanded);
  if (lse.has_value()) {
    auto const& lse_tensor = lse.value();
    TVM_FFI_ICHECK_EQ(lse_tensor.dtype(), dl_float32);
    TVM_FFI_ICHECK_EQ(lse_tensor.ndim(), 2);
    TVM_FFI_ICHECK_EQ(lse_tensor.size(0), query.size(0));
    TVM_FFI_ICHECK_EQ(lse_tensor.size(1), NUM_Q_HEADS);
    TVM_FFI_ICHECK_EQ(lse_stride_tokens, NUM_Q_HEADS);
    TVM_FFI_ICHECK_EQ(lse_stride_heads, 1);
    lse_ptr = static_cast<float*>(lse_tensor.data_ptr());
  }
  float* sinks_ptr = reinterpret_cast<float*>(seq_q_expanded);
  if (attention_sinks.has_value()) {
    auto const& sinks = attention_sinks.value();
    TVM_FFI_ICHECK_EQ(sinks.dtype(), dl_float32);
    TVM_FFI_ICHECK_EQ(sinks.numel(), NUM_Q_HEADS);
    TVM_FFI_ICHECK(sinks.IsContiguous());
    sinks_ptr = static_cast<float*>(sinks.data_ptr());
  }

  float softmax_scale_log2 =
      static_cast<float>(ScalarScale(bmm1_scale, "bmm1_scale") * 1.4426950408889634);
  unsigned int total_tiles = static_cast<unsigned int>(NUM_M_BLOCKS * total_bh);
  unsigned int grid_x = std::min<unsigned int>(static_cast<unsigned int>(sm_count), total_tiles);
  auto* dynamic_counter = static_cast<uint32_t*>(multi_ctas_kv_counter_buffer.data_ptr());
  auto const* p_q = reinterpret_cast<CakeFmhaTensorMap const*>(&h_q);
  auto const* p_k = reinterpret_cast<CakeFmhaTensorMap const*>(&h_k);
  auto const* p_v = reinterpret_cast<CakeFmhaTensorMap const*>(&h_v);
  cudaError_t status = cake_fmha_launch_context_fp8(
      p_q, p_k, p_v, static_cast<__nv_bfloat16*>(out.data_ptr()), lse_ptr, sinks_ptr, table_k, table_v,
      seq_q_expanded, seq_kv_expanded, cu_q_expanded, softmax_scale_log2, output_scale, total_bh,
      static_cast<int>(page_row_stride), static_cast<int>(grid_x), dynamic_counter, grid_x, 1, 1,
      stream);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "Cake FMHA context launch failed: " << cudaGetErrorString(status);

  (void)enable_pdl;
}

}  // namespace cake_fmha
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_paged_attention_context,
                              flashinfer::cake_fmha::cake_paged_attention_context);

// D06 call-local views into one caller/private scratch allocation. No pointer cache.
using D06TV = tvm::ffi::TensorView;
void d06_prepare_metadata(D06TV,D06TV,D06TV,D06TV,D06TV,D06TV,D06TV,D06TV,D06TV,int64_t,D06TV,D06TV,D06TV);
void d06_prepare_window_metadata(D06TV,D06TV,D06TV,D06TV,D06TV,D06TV,D06TV,D06TV,D06TV,
                        D06TV,D06TV,D06TV,D06TV,int64_t,int64_t,int64_t,D06TV,D06TV,D06TV);

namespace flashinfer { namespace cake_fmha {
struct D06LocalView {
  DLTensor tensor{};
  int64_t shape[4]{},stride[4]{};
  void reset(void* data,DLDevice device,DLDataType dtype,std::initializer_list<int64_t> dimensions) {
    tensor.data=data;tensor.device=device;tensor.dtype=dtype;
    tensor.ndim=dimensions.size();tensor.shape=shape;tensor.strides=stride;
    int index=0;for(auto size:dimensions)shape[index++]=size;
    int64_t span=1;for(int i=tensor.ndim-1;i>=0;--i){stride[i]=span;span*=shape[i];}
  }
  D06TV view() const {return D06TV(&tensor);}
};

void d06_fused_context(D06TV out,Optional<D06TV> lse,D06TV query,D06TV kp,D06TV vp,
 D06TV ks,D06TV vs,D06TV scratch,D06TV tables,D06TV lengths,D06TV cuq,D06TV cukv,
 int64_t maxq,int64_t maxkv,double scale1,double scale2,int64_t batch,int64_t sms,
 Optional<D06TV> sinks,bool causal,bool shared,int64_t window,int64_t capacity,bool compact,int64_t stream) {
  const int64_t rows=query.size(0)*NUM_Q_HEADS;
  int64_t cursor=AlignUp(3*batch*(NUM_Q_HEADS/PACK_G)*4+64,128);
#ifdef D06_CONTEXT_SPLITS
  cursor+=rows*D06_CONTEXT_SPLITS*(D06_OUTPUT_HEAD_DIM+2)*4;
#endif
  auto* base=static_cast<uint8_t*>(scratch.data_ptr());
  auto allocate=[&](D06LocalView& view,DLDataType dtype,std::initializer_list<int64_t> dimensions){
    cursor=((reinterpret_cast<uintptr_t>(base)+cursor+127)&~uintptr_t(127))-reinterpret_cast<uintptr_t>(base);
    view.reset(base+cursor,scratch.device(),dtype,dimensions);
    int64_t count=1;for(auto size:dimensions)count*=size;
    cursor+=count*((dtype.bits*dtype.lanes+7)/8);
  };
  const int64_t pages=compact?batch*capacity:kp.size(0);
  D06LocalView key,value,pad,counter,mapped;
  allocate(key,query.dtype(),{pages,kp.size(1),kp.size(2),D06_OUTPUT_HEAD_DIM});
  allocate(value,query.dtype(),{pages,kp.size(1),kp.size(2),D06_OUTPUT_HEAD_DIM});
  if(D06_OUTPUT_HEAD_DIM==64)allocate(pad,query.dtype(),{query.size(0),NUM_Q_HEADS,D06_OUTPUT_HEAD_DIM});
  allocate(counter,DLDataType{kDLUInt,8,1},{8});
  if(compact){
    if(tables.ndim()==2)allocate(mapped,tables.dtype(),{tables.size(0),tables.size(1)});
    else allocate(mapped,tables.dtype(),{tables.size(0),tables.size(1),tables.size(2)});
  }
  TVM_FFI_ICHECK_LE(cursor,scratch.numel()*get_element_size(scratch));
  D06TV qp=D06_OUTPUT_HEAD_DIM==64?pad.view():query;
  if(compact)::d06_prepare_window_metadata(query,qp,kp,vp,ks,vs,key.view(),value.view(),counter.view(),
      tables,cuq,lengths,mapped.view(),capacity,window,stream,cuq,lengths,scratch);
  else ::d06_prepare_metadata(query,qp,kp,vp,ks,vs,key.view(),value.view(),counter.view(),stream,cuq,lengths,scratch);
  cake_paged_attention_context(out,Optional<D06TV>(),qp,key.view(),value.view(),scratch,
      counter.view(),compact?mapped.view():tables,lengths,maxq,maxkv,scale1,scale2,-1.0,-1,0,
      batch,window,cuq,cukv,sms,false,scratch.numel()*get_element_size(scratch),sinks,
      Optional<D06TV>(),Optional<D06TV>(),Optional<float>(),Optional<bool>(shared),
      Optional<bool>(false),Optional<bool>(false),causal,lse,
      lse.has_value()?lse.value().stride(0):0,lse.has_value()?lse.value().stride(1):0,stream);
}
}}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(d06_fused_context,flashinfer::cake_fmha::d06_fused_context);
