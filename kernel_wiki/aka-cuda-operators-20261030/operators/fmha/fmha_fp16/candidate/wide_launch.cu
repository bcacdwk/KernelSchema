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

#include "wide_cake_fmha.cuh"
#include "host.cuh"

#ifndef CAKE_FMHA_HD256_FP8
#error "CAKE_FMHA_HD256_FP8 must be supplied by the route-specific JIT"
#endif
#ifndef CAKE_FMHA_SOURCE_PAGE_SIZE
#error "CAKE_FMHA_SOURCE_PAGE_SIZE must be supplied by the route-specific JIT"
#endif

extern "C" cudaError_t aka_launch_context_bf16_hd256(
    CakeFmhaTensorMap const* Q,
    CakeFmhaTensorMap const* K,
    CakeFmhaTensorMap const* V,
    __half* O_ptr,
    int* page_table,
    int* seq_lens_q,
    int* seq_lens_kv,
    int* cu_seq_lens_q,
    float softmax_scale_log2,
    int total_bh,
    int max_pages_per_seq,
    uint32_t* dynamic_counter,
    float* LSE_ptr,
    const int* original_q_indptr,
    unsigned int grid_x,
    unsigned int grid_y,
    unsigned int grid_z,
    cudaStream_t stream);

using tvm::ffi::Optional;
using tvm::ffi::Variant;

// Kernel launch configuration for this route.
/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "wide_cake_fmha.cuh"

extern "C" __global__ void kernel_aka_context_bf16_hd256(CakeFmhaTensorMap const* Q, CakeFmhaTensorMap const* K, CakeFmhaTensorMap const* V, __half* O_ptr, int* page_table, int* seq_lens_q, int* seq_lens_kv, int* cu_seq_lens_q, float softmax_scale_log2, int total_bh, int max_pages_per_seq, uint32_t* dynamic_counter, float* LSE_ptr, const int* original_q_indptr);

extern "C" cudaError_t aka_launch_context_bf16_hd256(
    CakeFmhaTensorMap const* Q,
    CakeFmhaTensorMap const* K,
    CakeFmhaTensorMap const* V,
    __half* O_ptr,
    int* page_table,
    int* seq_lens_q,
    int* seq_lens_kv,
    int* cu_seq_lens_q,
    float softmax_scale_log2,
    int total_bh,
    int max_pages_per_seq,
    uint32_t* dynamic_counter,
    float* LSE_ptr,
    const int* original_q_indptr,
    unsigned int grid_x,
    unsigned int grid_y,
    unsigned int grid_z,
    cudaStream_t stream) {
    cudaError_t status = cudaFuncSetAttribute(
        reinterpret_cast<const void*>(kernel_aka_context_bf16_hd256),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        200960);
    if (status != cudaSuccess) {
        return status;
    }
    void* kernel_args[] = {
        const_cast<void*>(reinterpret_cast<const void*>(&Q)),
        const_cast<void*>(reinterpret_cast<const void*>(&K)),
        const_cast<void*>(reinterpret_cast<const void*>(&V)),
        const_cast<void*>(reinterpret_cast<const void*>(&O_ptr)),
        const_cast<void*>(reinterpret_cast<const void*>(&page_table)),
        const_cast<void*>(reinterpret_cast<const void*>(&seq_lens_q)),
        const_cast<void*>(reinterpret_cast<const void*>(&seq_lens_kv)),
        const_cast<void*>(reinterpret_cast<const void*>(&cu_seq_lens_q)),
        const_cast<void*>(reinterpret_cast<const void*>(&softmax_scale_log2)),
        const_cast<void*>(reinterpret_cast<const void*>(&total_bh)),
        const_cast<void*>(reinterpret_cast<const void*>(&max_pages_per_seq)),
        const_cast<void*>(reinterpret_cast<const void*>(&dynamic_counter)),
        const_cast<void*>(reinterpret_cast<const void*>(&LSE_ptr)),
        const_cast<void*>(reinterpret_cast<const void*>(&original_q_indptr))
    };
    return cudaLaunchKernel(
        reinterpret_cast<const void*>(kernel_aka_context_bf16_hd256),
        dim3(grid_x, grid_y, grid_z),
        dim3(384, 1, 1),
        kernel_args,
        200960,
        stream);
}

namespace flashinfer {
namespace cake_fmha {
namespace {

using tvm::ffi::TensorView;

constexpr int kHeadDim = 256;
constexpr int kBlockM = 128;
constexpr int kMicroPage = 128;
constexpr int kSupportThreads = 256;

int64_t AlignUp(int64_t value, int64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

double ScalarScale(Variant<double, ffi::Tensor> scale, const char* name) {
  auto scalar = scale.as<double>();
  TVM_FFI_ICHECK(scalar.has_value()) << name << " must be a host scalar";
  return scalar.value();
}



#include "tma_cache.cuh"

CUtensorMap EncodeDirectQ(TensorView query) {
  uint64_t global_dim[4] = {64u, static_cast<uint64_t>(query.size(0)), 4u, NUM_Q_HEADS};
  uint64_t global_strides[3] = {static_cast<uint64_t>(query.stride(0))*2u, 128u,
                              static_cast<uint64_t>(query.stride(1))*2u};
  uint32_t box_dim[4] = {64u, 128u, 2u, 1u};
  uint32_t elem_strides[4] = {1u,1u,1u,1u};
  CUtensorMap tm;
  auto status=cuTensorMapEncodeTiled(&tm,CU_TENSOR_MAP_DATA_TYPE_FLOAT16,4,
      query.data_ptr(),global_dim,global_strides,box_dim,elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK_EQ(status,CUDA_SUCCESS) << "HD256 direct Q tensor map";
  return tm;
}

CUtensorMap EncodeStagedKv(void* pointer, int64_t total_micro_pages) {
#if CAKE_FMHA_HD256_FP8
  uint64_t global_dim[4] = {
      128u, kMicroPage, 2u, static_cast<uint64_t>(total_micro_pages)};
  uint64_t global_strides[3] = {256u, 128u, uint64_t(kMicroPage) * 256u};
  uint32_t box_dim[4] = {128u, kMicroPage, 1u, 1u};
  CUtensorMapDataType dtype = CU_TENSOR_MAP_DATA_TYPE_UINT8;
#else
  uint64_t global_dim[4] = {
      64u, kMicroPage, 4u, static_cast<uint64_t>(total_micro_pages)};
  uint64_t global_strides[3] = {512u, 128u, uint64_t(kMicroPage) * 512u};
  uint32_t box_dim[4] = {64u, kMicroPage, 1u, 1u};
  CUtensorMapDataType dtype = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
#endif
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm;
  CUresult result = cuTensorMapEncodeTiled(
      &tm, dtype, 4, pointer, global_dim, global_strides, box_dim,
      elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS)
      << "failed to encode Cake FMHA HD256 staged KV map";
  return tm;
}

unsigned int SupportGrid(int64_t items) {
  return static_cast<unsigned int>((items + kSupportThreads - 1) /
                                   kSupportThreads);
}

}  // namespace

void cake_paged_attention_context(
    TensorView out, Optional<TensorView> out_scale_factor, TensorView query,
    TensorView key_cache, TensorView value_cache, TensorView workspace_buffer,
    TensorView multi_ctas_kv_counter_buffer, TensorView block_tables,
    TensorView seq_lens, int64_t max_q_len, int64_t max_kv_len,
    Variant<double, ffi::Tensor> bmm1_scale,
    Variant<double, ffi::Tensor> bmm2_scale, double o_sf_scale,
    int64_t o_sf_vec_size, int64_t o_sf_start_index, int64_t batch_size,
    int64_t window_left, TensorView cum_seq_lens_q,
    TensorView cum_seq_lens_kv, int64_t sm_count, bool enable_pdl,
    int64_t workspace_size, Optional<TensorView> attention_sinks,
    Optional<TensorView> key_block_scales,
    Optional<TensorView> value_block_scales,
    Optional<float> skip_softmax_threshold_scale_factor,
    Optional<bool> uses_shared_paged_kv_idx,
    Optional<bool> use_fp16_softmax, Optional<bool> uses_spcompress,
    bool is_causal, Optional<TensorView> lse, int64_t lse_stride_tokens,
    int64_t lse_stride_heads) {
#if CAKE_FMHA_HD256_FP8
  TVM_FFI_ICHECK_EQ(query.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK_EQ(key_cache.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK_EQ(value_cache.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK_EQ(out.dtype(), dl_float16);
  TVM_FFI_ICHECK(is_causal);
  constexpr int kInputBytes = kHeadDim;
  constexpr int kOutputBytes = 2 * kHeadDim;
#else
  TVM_FFI_ICHECK_EQ(query.dtype(), dl_float16);
  TVM_FFI_ICHECK_EQ(key_cache.dtype(), dl_float16);
  TVM_FFI_ICHECK_EQ(value_cache.dtype(), dl_float16);
  TVM_FFI_ICHECK_EQ(out.dtype(), dl_float16);
  TVM_FFI_ICHECK(is_causal);
  constexpr int kInputBytes = 2 * kHeadDim;
  constexpr int kOutputBytes = 2 * kHeadDim;
  TVM_FFI_ICHECK_EQ(ScalarScale(bmm2_scale, "bmm2_scale"), 1.0);
#endif
  TVM_FFI_ICHECK(!out_scale_factor.has_value());
  TVM_FFI_ICHECK(!attention_sinks.has_value());
  TVM_FFI_ICHECK(!key_block_scales.has_value() &&
                 !value_block_scales.has_value());
  float* lse_pointer = nullptr;
  if (lse.has_value()) {
    auto target=lse.value();
    TVM_FFI_ICHECK(target.dtype()==dl_float32 && target.IsContiguous());
    TVM_FFI_ICHECK_EQ(target.numel(),query.size(0)*NUM_Q_HEADS);
    CheckSameDevice(query,target,"lse");
    TVM_FFI_ICHECK_EQ(lse_stride_tokens,NUM_Q_HEADS);
    TVM_FFI_ICHECK_EQ(lse_stride_heads,1);
    lse_pointer=static_cast<float*>(target.data_ptr());
  }
  TVM_FFI_ICHECK_EQ(
      skip_softmax_threshold_scale_factor.value_or(0.0f), 0.0f);
  TVM_FFI_ICHECK(!use_fp16_softmax.value_or(false));
  TVM_FFI_ICHECK(!uses_spcompress.value_or(false));
  TVM_FFI_ICHECK_EQ(window_left, -1);
  TVM_FFI_ICHECK_EQ(o_sf_scale, -1.0);
  TVM_FFI_ICHECK_EQ(o_sf_vec_size, -1);
  TVM_FFI_ICHECK_EQ(o_sf_start_index, 0);
  TVM_FFI_ICHECK(uses_shared_paged_kv_idx.value_or(true));
  TVM_FFI_ICHECK_GT(batch_size, 0);
  TVM_FFI_ICHECK_GT(max_q_len, 0);
  TVM_FFI_ICHECK_GT(max_kv_len, 0);
  TVM_FFI_ICHECK_GT(sm_count, 0);
  TVM_FFI_ICHECK_EQ((max_q_len + kBlockM - 1) / kBlockM,
                    NUM_M_BLOCKS);

  TVM_FFI_ICHECK_EQ(query.ndim(), 3);
  TVM_FFI_ICHECK_EQ(query.size(1), NUM_Q_HEADS);
  TVM_FFI_ICHECK_EQ(query.size(2), kHeadDim);
  TVM_FFI_ICHECK_EQ(query.stride(2), 1);
  TVM_FFI_ICHECK_EQ(out.ndim(), 3);
  TVM_FFI_ICHECK_EQ(out.size(0), query.size(0));
  TVM_FFI_ICHECK_EQ(out.size(1), query.size(1));
  TVM_FFI_ICHECK_EQ(out.size(2), query.size(2));
  TVM_FFI_ICHECK(out.IsContiguous());
  TVM_FFI_ICHECK_EQ(key_cache.ndim(), 4);
  TVM_FFI_ICHECK_EQ(value_cache.ndim(), 4);
  TVM_FFI_ICHECK_EQ(key_cache.size(0), value_cache.size(0));
  TVM_FFI_ICHECK_EQ(key_cache.size(1), NUM_KV_HEADS);
  TVM_FFI_ICHECK_EQ(value_cache.size(1), NUM_KV_HEADS);
  TVM_FFI_ICHECK_EQ(key_cache.size(2), CAKE_FMHA_SOURCE_PAGE_SIZE);
  TVM_FFI_ICHECK_EQ(value_cache.size(2), CAKE_FMHA_SOURCE_PAGE_SIZE);
  TVM_FFI_ICHECK_EQ(key_cache.size(3), kHeadDim);
  TVM_FFI_ICHECK_EQ(value_cache.size(3), kHeadDim);
  TVM_FFI_ICHECK_EQ(key_cache.stride(3), 1);
  TVM_FFI_ICHECK_EQ(value_cache.stride(3), 1);
  TVM_FFI_ICHECK_EQ(block_tables.ndim(), 2);
  TVM_FFI_ICHECK_EQ(block_tables.size(0), batch_size);
  TVM_FFI_ICHECK(block_tables.dtype() == dl_int32 ||
                 block_tables.dtype() == dl_uint32);
  TVM_FFI_ICHECK_EQ(block_tables.stride(1), 1);
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

  CheckSameDevice(query, key_cache, "key_cache");
  CheckSameDevice(query, value_cache, "value_cache");
  CheckSameDevice(query, out, "out");
  CheckSameDevice(query, workspace_buffer, "workspace_buffer");
  CheckSameDevice(query, multi_ctas_kv_counter_buffer,
                  "multi_ctas_kv_counter_buffer");
  CheckSameDevice(query, block_tables, "block_tables");
  CheckSameDevice(query, seq_lens, "seq_lens");
  CheckSameDevice(query, cum_seq_lens_q, "cum_seq_lens_q");
  CheckSameDevice(query, cum_seq_lens_kv, "cum_seq_lens_kv");

  int64_t padded_q = NUM_M_BLOCKS * kBlockM;
  int64_t max_micro_pages = (max_kv_len + 127) / 128;
  int64_t total_q_rows = batch_size * NUM_Q_HEADS * padded_q;
  int64_t total_micro_pages =
      batch_size * NUM_KV_HEADS * max_micro_pages;
  int64_t cursor = 0;
  int64_t q_offset = cursor;
  // Q is read directly; no packed Q allocation.
  cursor = AlignUp(cursor, 16);
  int64_t k_offset = cursor;
  cursor += total_micro_pages * kMicroPage * kInputBytes;
  cursor = AlignUp(cursor, 16);
  int64_t v_offset = cursor;
  cursor += total_micro_pages * kMicroPage * kInputBytes;
  cursor = AlignUp(cursor, 16);
  int64_t o_offset = cursor;
  // Direct output uses the caller buffer; no packed output allocation.
  cursor = AlignUp(cursor, 16);
  int64_t seq_q_offset = cursor;
  cursor += batch_size * NUM_Q_HEADS * static_cast<int64_t>(sizeof(int));
  int64_t seq_kv_offset = cursor;
  cursor += batch_size * NUM_Q_HEADS * static_cast<int64_t>(sizeof(int));
  int64_t cu_q_offset = cursor;
  cursor += batch_size * NUM_Q_HEADS * static_cast<int64_t>(sizeof(int));
  int64_t page_offset = cursor;
  cursor += total_micro_pages * static_cast<int64_t>(sizeof(int));

  TVM_FFI_ICHECK(workspace_buffer.IsContiguous());
  TVM_FFI_ICHECK_EQ(
      reinterpret_cast<uintptr_t>(workspace_buffer.data_ptr()) % 16, 0);
  int64_t actual_workspace_bytes =
      workspace_buffer.numel() * get_element_size(workspace_buffer);
  TVM_FFI_ICHECK_GE(actual_workspace_bytes, cursor)
      << "Cake FMHA HD256 context workspace requires " << cursor << " bytes";
  TVM_FFI_ICHECK_GE(workspace_size, cursor);
  int64_t counter_bytes = multi_ctas_kv_counter_buffer.numel() *
                          get_element_size(multi_ctas_kv_counter_buffer);
  TVM_FFI_ICHECK_GE(counter_bytes, static_cast<int64_t>(sizeof(uint32_t)));

  auto* workspace = static_cast<uint8_t*>(workspace_buffer.data_ptr());
  auto* q_packed = workspace + q_offset;
  auto* k_packed = workspace + k_offset;
  auto* v_packed = workspace + v_offset;
  auto* o_packed = workspace + o_offset;
  auto* seq_lens_q = reinterpret_cast<int*>(workspace + seq_q_offset);
  auto* seq_lens_kv = reinterpret_cast<int*>(workspace + seq_kv_offset);
  auto* cu_seq_lens_q = reinterpret_cast<int*>(workspace + cu_q_offset);
  auto* kernel_page_table = reinterpret_cast<int*>(workspace + page_offset);
  auto* dynamic_counter =
      static_cast<uint32_t*>(multi_ctas_kv_counter_buffer.data_ptr());

  for (auto tensor : {query,key_cache,value_cache}) {
    TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(tensor.data_ptr())%16,0);
    for (int axis=0;axis<tensor.ndim()-1;++axis)
      TVM_FFI_ICHECK_EQ((tensor.stride(axis)*get_element_size(tensor))%16,0);
  }
  ffi::CUDADeviceGuard device_guard(query.device().device_id);
  cudaStream_t stream = get_stream(query.device());
  unsigned int stage_q_grid = SupportGrid(total_q_rows * kInputBytes / 16);
  unsigned int stage_kv_grid = SupportGrid(
      total_micro_pages * kMicroPage * static_cast<int64_t>(kInputBytes) / 16);
  unsigned int metadata_grid = SupportGrid(std::max<int64_t>(
      batch_size * NUM_Q_HEADS, total_micro_pages));
  unsigned int scatter_grid = SupportGrid(total_q_rows * kOutputBytes);

  cudaError_t status = cudaSuccess;
  status = cake_fmha_launch_hd256_stage_kv(
      static_cast<const uint8_t*>(key_cache.data_ptr()),
      static_cast<const uint8_t*>(value_cache.data_ptr()), k_packed, v_packed,
      static_cast<const int*>(block_tables.data_ptr()),
      static_cast<const int*>(seq_lens.data_ptr()),
      static_cast<const int*>(cum_seq_lens_q.data_ptr()),
      seq_lens_q,seq_lens_kv,cu_seq_lens_q,kernel_page_table,dynamic_counter,NUM_Q_HEADS,
      static_cast<int>(batch_size),
      NUM_KV_HEADS, CAKE_FMHA_SOURCE_PAGE_SIZE,
      static_cast<int>(max_micro_pages), kInputBytes,
      key_cache.stride(0) * get_element_size(key_cache),
      key_cache.stride(2) * get_element_size(key_cache),
      key_cache.stride(1) * get_element_size(key_cache), block_tables.stride(0),
      0, stage_kv_grid, stream);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "Cake FMHA HD256 KV staging failed: " << cudaGetErrorString(status);

  CUtensorMap h_q = EncodeDirectQ(query);
  CUtensorMap h_k = EncodeStagedKv(k_packed, total_micro_pages);
  CUtensorMap h_v = EncodeStagedKv(v_packed, total_micro_pages);
  float softmax_scale_log2 =
      static_cast<float>(ScalarScale(bmm1_scale, "bmm1_scale") *
                         1.4426950408889634);
  unsigned int total_tiles = static_cast<unsigned int>(
      NUM_M_BLOCKS * batch_size * NUM_Q_HEADS);
  unsigned int grid_x =
      std::min<unsigned int>(static_cast<unsigned int>(sm_count), total_tiles);
  auto q_slot = TmaDeviceSlot(h_q, query.device().device_id, stream);
  auto k_slot = TmaDeviceSlot(h_k, query.device().device_id, stream);
  auto v_slot = TmaDeviceSlot(h_v, query.device().device_id, stream);
  auto const* p_q = reinterpret_cast<CakeFmhaTensorMap const*>(q_slot.pointer);
  auto const* p_k = reinterpret_cast<CakeFmhaTensorMap const*>(k_slot.pointer);
  auto const* p_v = reinterpret_cast<CakeFmhaTensorMap const*>(v_slot.pointer);
#if CAKE_FMHA_HD256_FP8
  status = cake_fmha_launch_context_fp8_hd256(
      p_q, p_k, p_v, o_packed, kernel_page_table, seq_lens_q, seq_lens_kv,
      cu_seq_lens_q, softmax_scale_log2,
      static_cast<float>(ScalarScale(bmm2_scale, "bmm2_scale")),
      static_cast<int>(batch_size * NUM_Q_HEADS),
      static_cast<int>(max_micro_pages), dynamic_counter, grid_x, 1, 1,
      stream);
#else
  status = aka_launch_context_bf16_hd256(
      p_q, p_k, p_v, static_cast<__half*>(out.data_ptr()), kernel_page_table,
      seq_lens_q, seq_lens_kv, cu_seq_lens_q, softmax_scale_log2,
      static_cast<int>(batch_size * NUM_Q_HEADS),
      static_cast<int>(max_micro_pages), dynamic_counter, lse_pointer,
      static_cast<const int*>(cum_seq_lens_q.data_ptr()), grid_x, 1, 1,
      stream);
#endif
  RecordTmaDeviceSlotUses({q_slot, k_slot, v_slot}, stream);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "Cake FMHA HD256 context launch failed: "
      << cudaGetErrorString(status);

  (void)cum_seq_lens_kv;
  (void)enable_pdl;
}

}  // namespace cake_fmha
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    cake_paged_attention_context,
    flashinfer::cake_fmha::cake_paged_attention_context);
