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

// Editable source; provenance is recorded in ../LICENSES/.
// tvm-ffi direct-source launcher for Cake kernel 'kernel_cake_minimax_h3_varlen_attention_a9de8bf510ed0bfbb02a'.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "ffi.cuh"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" __global__ void kernel_cake_minimax_h3_varlen_attention_a9de8bf510ed0bfbb02a(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap Vt, const __grid_constant__ CUtensorMap SFQ, const __grid_constant__ CUtensorMap SFK, const __grid_constant__ CUtensorMap SFVtLo, const __grid_constant__ CUtensorMap SFVtHi, __nv_bfloat16* __restrict__ O, int* __restrict__ cl_head, int* __restrict__ cl_seg_begin, int* __restrict__ cl_seg_len, int* __restrict__ cl_kv_base, int* __restrict__ cl_q_block, int* __restrict__ cl_kv_begin, int* __restrict__ cl_kv_blocks, int* __restrict__ cl_ws_slot, __half* __restrict__ partial_O, float* __restrict__ partial_ML, unsigned int num_tiles, int total_clusters, int heads, int PB, float softmax_scale_log2, float* __restrict__ prefix_max, int prefix_stride, const int* __restrict__ pf_tile, int pf_total, int* __restrict__ pf_flag, const int* __restrict__ pf_info);


namespace cake_host_shim_a442e6313e216d3d {

using tvm::ffi::Optional;
using tvm::ffi::TensorView;

#include "host_checks.cuh"

// 2D TMA descriptor for buffer 'Q' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_Q(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'Q' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'Q' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError)
      << "TMA source 'Q' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  CheckDenseLeadingFold(t, 1, "Q");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'Q' physical strides must be positive";
  uint64_t global_dim[2] = {(uint64_t)(d1), (uint64_t)(outer1)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0, ValueError)
      << "TMA descriptor for 'Q' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 128u <= global_dim[1], ValueError)
      << "TMA box (64, 128) exceeds resolved global dims for 'Q'";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'Q' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'Q' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 8) % 8 == 0, ValueError)
      << "TMA descriptor for 'Q' resolved global stride 1 to a non-whole-byte offset";
  uint64_t global_strides[1] = {
      (uint64_t)((carrier_stride_0 * 8) / 8),
  };
  uint32_t box_dim[2] = {64u, 128u};
  uint32_t elem_strides[2] = {1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (2D, 'Q') failed: CUresult=" << (int)r;
  return tm;
}

// 2D TMA descriptor for buffer 'K' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_K(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'K' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'K' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError)
      << "TMA source 'K' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  CheckDenseLeadingFold(t, 1, "K");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'K' physical strides must be positive";
  uint64_t global_dim[2] = {(uint64_t)(d1), (uint64_t)(outer1)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0, ValueError)
      << "TMA descriptor for 'K' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 64u <= global_dim[1], ValueError)
      << "TMA box (64, 64) exceeds resolved global dims for 'K'";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'K' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'K' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 8) % 8 == 0, ValueError)
      << "TMA descriptor for 'K' resolved global stride 1 to a non-whole-byte offset";
  uint64_t global_strides[1] = {
      (uint64_t)((carrier_stride_0 * 8) / 8),
  };
  uint32_t box_dim[2] = {64u, 64u};
  uint32_t elem_strides[2] = {1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (2D, 'K') failed: CUresult=" << (int)r;
  return tm;
}

// 2D TMA descriptor for buffer 'Vt' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_Vt(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'Vt' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'Vt' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError)
      << "TMA source 'Vt' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  CheckDenseLeadingFold(t, 1, "Vt");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'Vt' physical strides must be positive";
  uint64_t global_dim[2] = {(uint64_t)(d1), (uint64_t)(outer1)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0, ValueError)
      << "TMA descriptor for 'Vt' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 64u <= global_dim[1], ValueError)
      << "TMA box (64, 64) exceeds resolved global dims for 'Vt'";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'Vt' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'Vt' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 8) % 8 == 0, ValueError)
      << "TMA descriptor for 'Vt' resolved global stride 1 to a non-whole-byte offset";
  uint64_t global_strides[1] = {
      (uint64_t)((carrier_stride_0 * 8) / 8),
  };
  uint32_t box_dim[2] = {64u, 64u};
  uint32_t elem_strides[2] = {1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (2D, 'Vt') failed: CUresult=" << (int)r;
  return tm;
}

// 2D TMA descriptor for buffer 'SFQ' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_SFQ(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'SFQ' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'SFQ' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError)
      << "TMA source 'SFQ' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  CheckDenseLeadingFold(t, 1, "SFQ");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'SFQ' physical strides must be positive";
  uint64_t global_dim[2] = {(uint64_t)(d1), (uint64_t)(outer1)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0, ValueError)
      << "TMA descriptor for 'SFQ' resolved a non-positive global dim";
  TVM_FFI_CHECK(32u <= global_dim[0] && 32u <= global_dim[1], ValueError)
      << "TMA box (32, 32) exceeds resolved global dims for 'SFQ'";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'SFQ' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'SFQ' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 8) % 8 == 0, ValueError)
      << "TMA descriptor for 'SFQ' resolved global stride 1 to a non-whole-byte offset";
  uint64_t global_strides[1] = {
      (uint64_t)((carrier_stride_0 * 8) / 8),
  };
  uint32_t box_dim[2] = {32u, 32u};
  uint32_t elem_strides[2] = {1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (2D, 'SFQ') failed: CUresult=" << (int)r;
  return tm;
}

// 2D TMA descriptor for buffer 'SFK' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_SFK(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'SFK' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'SFK' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError)
      << "TMA source 'SFK' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  CheckDenseLeadingFold(t, 1, "SFK");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'SFK' physical strides must be positive";
  uint64_t global_dim[2] = {(uint64_t)(d1), (uint64_t)(outer1)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0, ValueError)
      << "TMA descriptor for 'SFK' resolved a non-positive global dim";
  TVM_FFI_CHECK(32u <= global_dim[0] && 32u <= global_dim[1], ValueError)
      << "TMA box (32, 32) exceeds resolved global dims for 'SFK'";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'SFK' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'SFK' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 8) % 8 == 0, ValueError)
      << "TMA descriptor for 'SFK' resolved global stride 1 to a non-whole-byte offset";
  uint64_t global_strides[1] = {
      (uint64_t)((carrier_stride_0 * 8) / 8),
  };
  uint32_t box_dim[2] = {32u, 32u};
  uint32_t elem_strides[2] = {1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (2D, 'SFK') failed: CUresult=" << (int)r;
  return tm;
}

// 2D TMA descriptor for buffer 'SFVtLo' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_SFVtLo(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'SFVtLo' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'SFVtLo' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError)
      << "TMA source 'SFVtLo' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  CheckDenseLeadingFold(t, 1, "SFVtLo");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'SFVtLo' physical strides must be positive";
  uint64_t global_dim[2] = {(uint64_t)(d1), (uint64_t)(outer1)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0, ValueError)
      << "TMA descriptor for 'SFVtLo' resolved a non-positive global dim";
  TVM_FFI_CHECK(32u <= global_dim[0] && 16u <= global_dim[1], ValueError)
      << "TMA box (32, 16) exceeds resolved global dims for 'SFVtLo'";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'SFVtLo' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'SFVtLo' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 8) % 8 == 0, ValueError)
      << "TMA descriptor for 'SFVtLo' resolved global stride 1 to a non-whole-byte offset";
  uint64_t global_strides[1] = {
      (uint64_t)((carrier_stride_0 * 8) / 8),
  };
  uint32_t box_dim[2] = {32u, 16u};
  uint32_t elem_strides[2] = {1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (2D, 'SFVtLo') failed: CUresult=" << (int)r;
  return tm;
}

// 2D TMA descriptor for buffer 'SFVtHi' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_SFVtHi(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'SFVtHi' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'SFVtHi' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError)
      << "TMA source 'SFVtHi' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  CheckDenseLeadingFold(t, 1, "SFVtHi");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'SFVtHi' physical strides must be positive";
  uint64_t global_dim[2] = {(uint64_t)(d1), (uint64_t)(outer1)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0, ValueError)
      << "TMA descriptor for 'SFVtHi' resolved a non-positive global dim";
  TVM_FFI_CHECK(32u <= global_dim[0] && 16u <= global_dim[1], ValueError)
      << "TMA box (32, 16) exceeds resolved global dims for 'SFVtHi'";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'SFVtHi' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'SFVtHi' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 8) % 8 == 0, ValueError)
      << "TMA descriptor for 'SFVtHi' resolved global stride 1 to a non-whole-byte offset";
  uint64_t global_strides[1] = {
      (uint64_t)((carrier_stride_0 * 8) / 8),
  };
  uint32_t box_dim[2] = {32u, 16u};
  uint32_t elem_strides[2] = {1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (2D, 'SFVtHi') failed: CUresult=" << (int)r;
  return tm;
}

void Run(TensorView arg_Q, TensorView arg_K, TensorView arg_Vt, TensorView arg_SFQ, TensorView arg_SFK, TensorView arg_SFVtLo, TensorView arg_SFVtHi, TensorView arg_O, TensorView arg_cl_head, TensorView arg_cl_seg_begin, TensorView arg_cl_seg_len, TensorView arg_cl_kv_base, TensorView arg_cl_q_block, TensorView arg_cl_kv_begin, TensorView arg_cl_kv_blocks, TensorView arg_cl_ws_slot, TensorView arg_partial_O, TensorView arg_partial_ML, int64_t arg_num_tiles, int64_t arg_total_clusters, int64_t arg_heads, int64_t arg_PB, double arg_softmax_scale_log2, int64_t grid_x, int64_t grid_y, int64_t grid_z, TensorView arg_prefix_max, int64_t arg_prefix_stride, TensorView arg_pf_tile, int64_t arg_pf_total, TensorView arg_pf_flag, TensorView arg_pf_info) {
  // D16_PREFIX_REPAIR: caller-owned, per-call maxima scratch.
  CheckCudaTensor(arg_prefix_max, "prefix_max");
  CheckSameCudaDevice(arg_prefix_max, arg_Q, "prefix_max", "Q");
  CheckDtype(arg_prefix_max, "prefix_max", 2, 32, 1);
  CheckContiguous(arg_prefix_max, "prefix_max");
  TVM_FFI_CHECK(arg_prefix_stride > 0 && arg_prefix_stride <= INT32_MAX, ValueError)
      << "prefix_stride must be a positive int32";
  TVM_FFI_CHECK(arg_prefix_max.numel() >= arg_heads * arg_PB * 128 * arg_prefix_stride, ValueError)
      << "prefix_max is smaller than the packed query rows times prefix_stride";
  // Fused prepass tables. pf_tile is indexed by (round, cluster) with the same
  // stride the kernel uses, so its length must be a whole number of rounds of
  // the launched cluster count; pf_info holds three int32 per slot.
  CheckCudaTensor(arg_pf_tile, "pf_tile");
  CheckSameCudaDevice(arg_pf_tile, arg_Q, "pf_tile", "Q");
  CheckDtype(arg_pf_tile, "pf_tile", 0, 32, 1);
  CheckContiguous(arg_pf_tile, "pf_tile");
  CheckCudaTensor(arg_pf_flag, "pf_flag");
  CheckSameCudaDevice(arg_pf_flag, arg_Q, "pf_flag", "Q");
  CheckDtype(arg_pf_flag, "pf_flag", 0, 32, 1);
  CheckContiguous(arg_pf_flag, "pf_flag");
  CheckCudaTensor(arg_pf_info, "pf_info");
  CheckSameCudaDevice(arg_pf_info, arg_Q, "pf_info", "Q");
  CheckDtype(arg_pf_info, "pf_info", 0, 32, 1);
  CheckContiguous(arg_pf_info, "pf_info");
  TVM_FFI_CHECK(arg_pf_total >= 0 && arg_pf_total <= arg_pf_tile.numel(), ValueError)
      << "pf_total " << arg_pf_total << " is outside [0, pf_tile.numel()="
      << arg_pf_tile.numel() << "]";
  TVM_FFI_CHECK(grid_x > 0 && grid_x % 2 == 0 && arg_pf_total % (grid_x / 2) == 0, ValueError)
      << "pf_total " << arg_pf_total
      << " must be a whole number of rounds of the launched cluster count, and grid_x "
      << grid_x << " must be a positive multiple of the cluster width 2";
  TVM_FFI_CHECK(arg_pf_info.numel() >= 3 * arg_num_tiles, ValueError)
      << "pf_info holds " << arg_pf_info.numel() << " int32 but " << arg_num_tiles
      << " slots need " << (3 * arg_num_tiles);
  float* p_prefix_max = static_cast<float*>(arg_prefix_max.data_ptr());
  int32_t v_prefix_stride = (int32_t)arg_prefix_stride;
  int* p_pf_tile = static_cast<int*>(arg_pf_tile.data_ptr());
  int32_t v_pf_total = (int32_t)arg_pf_total;
  int* p_pf_flag = static_cast<int*>(arg_pf_flag.data_ptr());
  int* p_pf_info = static_cast<int*>(arg_pf_info.data_ptr());
  DLDevice dev = arg_Q.device();
  ScopedCudaDevice device_guard(dev.device_id);
  CheckCudaTensor(arg_Q, "Q");
  CheckDtype(arg_Q, "Q", 1, 8, 1);
  CheckCudaTensor(arg_K, "K");
  CheckDtype(arg_K, "K", 1, 8, 1);
  CheckCudaTensor(arg_Vt, "Vt");
  CheckDtype(arg_Vt, "Vt", 1, 8, 1);
  CheckCudaTensor(arg_SFQ, "SFQ");
  CheckDtype(arg_SFQ, "SFQ", 1, 8, 1);
  CheckCudaTensor(arg_SFK, "SFK");
  CheckDtype(arg_SFK, "SFK", 1, 8, 1);
  CheckCudaTensor(arg_SFVtLo, "SFVtLo");
  CheckDtype(arg_SFVtLo, "SFVtLo", 1, 8, 1);
  CheckCudaTensor(arg_SFVtHi, "SFVtHi");
  CheckDtype(arg_SFVtHi, "SFVtHi", 1, 8, 1);
  CheckCudaTensor(arg_O, "O");
  CheckDtype(arg_O, "O", 4, 16, 1);
  CheckContiguous(arg_O, "O");
  CheckCudaTensor(arg_cl_head, "cl_head");
  CheckDtype(arg_cl_head, "cl_head", 0, 32, 1);
  CheckContiguous(arg_cl_head, "cl_head");
  CheckCudaTensor(arg_cl_seg_begin, "cl_seg_begin");
  CheckDtype(arg_cl_seg_begin, "cl_seg_begin", 0, 32, 1);
  CheckContiguous(arg_cl_seg_begin, "cl_seg_begin");
  CheckCudaTensor(arg_cl_seg_len, "cl_seg_len");
  CheckDtype(arg_cl_seg_len, "cl_seg_len", 0, 32, 1);
  CheckContiguous(arg_cl_seg_len, "cl_seg_len");
  CheckCudaTensor(arg_cl_kv_base, "cl_kv_base");
  CheckDtype(arg_cl_kv_base, "cl_kv_base", 0, 32, 1);
  CheckContiguous(arg_cl_kv_base, "cl_kv_base");
  CheckCudaTensor(arg_cl_q_block, "cl_q_block");
  CheckDtype(arg_cl_q_block, "cl_q_block", 0, 32, 1);
  CheckContiguous(arg_cl_q_block, "cl_q_block");
  CheckCudaTensor(arg_cl_kv_begin, "cl_kv_begin");
  CheckDtype(arg_cl_kv_begin, "cl_kv_begin", 0, 32, 1);
  CheckContiguous(arg_cl_kv_begin, "cl_kv_begin");
  CheckCudaTensor(arg_cl_kv_blocks, "cl_kv_blocks");
  CheckDtype(arg_cl_kv_blocks, "cl_kv_blocks", 0, 32, 1);
  CheckContiguous(arg_cl_kv_blocks, "cl_kv_blocks");
  CheckCudaTensor(arg_cl_ws_slot, "cl_ws_slot");
  CheckDtype(arg_cl_ws_slot, "cl_ws_slot", 0, 32, 1);
  CheckContiguous(arg_cl_ws_slot, "cl_ws_slot");
  CheckCudaTensor(arg_partial_O, "partial_O");
  CheckDtype(arg_partial_O, "partial_O", 2, 16, 1);
  CheckContiguous(arg_partial_O, "partial_O");
  CheckCudaTensor(arg_partial_ML, "partial_ML");
  CheckDtype(arg_partial_ML, "partial_ML", 2, 32, 1);
  CheckContiguous(arg_partial_ML, "partial_ML");
  TVM_FFI_CHECK(arg_num_tiles >= 0LL && arg_num_tiles <= 4294967295LL, ValueError)
      << "scalar 'num_tiles' value " << arg_num_tiles
      << " is outside u32 range [0, 4294967295]";
  TVM_FFI_CHECK(arg_total_clusters >= -2147483648LL && arg_total_clusters <= 2147483647LL, ValueError)
      << "scalar 'total_clusters' value " << arg_total_clusters
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_heads >= -2147483648LL && arg_heads <= 2147483647LL, ValueError)
      << "scalar 'heads' value " << arg_heads
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_PB >= -2147483648LL && arg_PB <= 2147483647LL, ValueError)
      << "scalar 'PB' value " << arg_PB
      << " is outside i32 range [-2147483648, 2147483647]";
  CheckSameCudaDevice(arg_K, arg_Q, "K", "Q");
  CheckSameCudaDevice(arg_Vt, arg_Q, "Vt", "Q");
  CheckSameCudaDevice(arg_SFQ, arg_Q, "SFQ", "Q");
  CheckSameCudaDevice(arg_SFK, arg_Q, "SFK", "Q");
  CheckSameCudaDevice(arg_SFVtLo, arg_Q, "SFVtLo", "Q");
  CheckSameCudaDevice(arg_SFVtHi, arg_Q, "SFVtHi", "Q");
  CheckSameCudaDevice(arg_O, arg_Q, "O", "Q");
  CheckSameCudaDevice(arg_cl_head, arg_Q, "cl_head", "Q");
  CheckSameCudaDevice(arg_cl_seg_begin, arg_Q, "cl_seg_begin", "Q");
  CheckSameCudaDevice(arg_cl_seg_len, arg_Q, "cl_seg_len", "Q");
  CheckSameCudaDevice(arg_cl_kv_base, arg_Q, "cl_kv_base", "Q");
  CheckSameCudaDevice(arg_cl_q_block, arg_Q, "cl_q_block", "Q");
  CheckSameCudaDevice(arg_cl_kv_begin, arg_Q, "cl_kv_begin", "Q");
  CheckSameCudaDevice(arg_cl_kv_blocks, arg_Q, "cl_kv_blocks", "Q");
  CheckSameCudaDevice(arg_cl_ws_slot, arg_Q, "cl_ws_slot", "Q");
  CheckSameCudaDevice(arg_partial_O, arg_Q, "partial_O", "Q");
  CheckSameCudaDevice(arg_partial_ML, arg_Q, "partial_ML", "Q");
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << "launch grid dimensions must be positive, got (" << grid_x << ", " << grid_y
      << ", " << grid_z << ")";
  TVM_FFI_CHECK(grid_x % 2 == 0 && grid_y % 1 == 0 && grid_z % 1 == 0, ValueError)
      << "launch grid (" << grid_x << ", " << grid_y << ", " << grid_z
      << ") must be divisible by cluster dims (2, 1, 1)";

  cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(dev.device_type, dev.device_id);
  CUtensorMap p_Q = EncodeTma_Q(arg_Q);
  CUtensorMap p_K = EncodeTma_K(arg_K);
  CUtensorMap p_Vt = EncodeTma_Vt(arg_Vt);
  CUtensorMap p_SFQ = EncodeTma_SFQ(arg_SFQ);
  CUtensorMap p_SFK = EncodeTma_SFK(arg_SFK);
  CUtensorMap p_SFVtLo = EncodeTma_SFVtLo(arg_SFVtLo);
  CUtensorMap p_SFVtHi = EncodeTma_SFVtHi(arg_SFVtHi);
  __nv_bfloat16* p_O = static_cast<__nv_bfloat16*>(arg_O.data_ptr());
  int* p_cl_head = static_cast<int*>(arg_cl_head.data_ptr());
  int* p_cl_seg_begin = static_cast<int*>(arg_cl_seg_begin.data_ptr());
  int* p_cl_seg_len = static_cast<int*>(arg_cl_seg_len.data_ptr());
  int* p_cl_kv_base = static_cast<int*>(arg_cl_kv_base.data_ptr());
  int* p_cl_q_block = static_cast<int*>(arg_cl_q_block.data_ptr());
  int* p_cl_kv_begin = static_cast<int*>(arg_cl_kv_begin.data_ptr());
  int* p_cl_kv_blocks = static_cast<int*>(arg_cl_kv_blocks.data_ptr());
  int* p_cl_ws_slot = static_cast<int*>(arg_cl_ws_slot.data_ptr());
  __half* p_partial_O = static_cast<__half*>(arg_partial_O.data_ptr());
  float* p_partial_ML = static_cast<float*>(arg_partial_ML.data_ptr());
  uint32_t v_num_tiles = (uint32_t)arg_num_tiles;
  int32_t v_total_clusters = (int32_t)arg_total_clusters;
  int32_t v_heads = (int32_t)arg_heads;
  int32_t v_PB = (int32_t)arg_PB;
  // Preserve the original FP32 natural softmax scale before QDQ.
  float v_softmax_scale_log2 = (float)(arg_softmax_scale_log2 * 0.693147180559945309417);
  void* kargs[] = {&p_Q, &p_K, &p_Vt, &p_SFQ, &p_SFK, &p_SFVtLo, &p_SFVtHi, &p_O, &p_cl_head, &p_cl_seg_begin, &p_cl_seg_len, &p_cl_kv_base, &p_cl_q_block, &p_cl_kv_begin, &p_cl_kv_blocks, &p_cl_ws_slot, &p_partial_O, &p_partial_ML, &v_num_tiles, &v_total_clusters, &v_heads, &v_PB, &v_softmax_scale_log2, &p_prefix_max, &v_prefix_stride, &p_pf_tile, &v_pf_total, &p_pf_flag, &p_pf_info};

  static std::mutex smem_mu;
  static auto* smem_status_by_device =
      new std::unordered_map<int, cudaError_t>();
  cudaError_t smem_status = cudaSuccess;
  {
    std::lock_guard<std::mutex> lock(smem_mu);
    auto it = smem_status_by_device->find(dev.device_id);
    if (it == smem_status_by_device->end()) {
      int current_device = -1;
      smem_status = cudaGetDevice(&current_device);
      if (smem_status == cudaSuccess && current_device != dev.device_id) {
        smem_status = cudaErrorInvalidDevice;
      }
      if (smem_status == cudaSuccess) {
        smem_status = cudaFuncSetAttribute(
            reinterpret_cast<const void*>(kernel_cake_minimax_h3_varlen_attention_a9de8bf510ed0bfbb02a),
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            83968);
      }
      smem_status_by_device->emplace(dev.device_id, smem_status);
    } else {
      smem_status = it->second;
    }
  }
  TVM_FFI_CHECK(smem_status == cudaSuccess, RuntimeError)
      << "cudaFuncSetAttribute for kernel_cake_minimax_h3_varlen_attention_a9de8bf510ed0bfbb02a failed: "
      << cudaGetErrorString(smem_status);
  dim3 grid((uint32_t)grid_x, (uint32_t)grid_y, (uint32_t)grid_z);
  dim3 block(512u, 1u, 1u);

  // Extended direct-source launch for a linked CUDA kernel symbol.
  cudaLaunchAttribute attrs[3]{};
  int n = 0;
  attrs[n].id = cudaLaunchAttributeClusterDimension;
  attrs[n].val.clusterDim.x = 2u;
  attrs[n].val.clusterDim.y = 1u;
  attrs[n].val.clusterDim.z = 1u;
  ++n;
  attrs[n].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
  attrs[n].val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicySpread;
  ++n;
  attrs[n].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[n].val.programmaticStreamSerializationAllowed = 1;
  ++n;
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = 83968u;
  config.stream = stream;
  config.attrs = attrs;
  config.numAttrs = n;
  cudaError_t launch_status = cudaLaunchKernelExC(
      &config, reinterpret_cast<const void*>(kernel_cake_minimax_h3_varlen_attention_a9de8bf510ed0bfbb02a), kargs);
  TVM_FFI_CHECK(launch_status == cudaSuccess, RuntimeError)
      << "cudaLaunchKernelExC for kernel_cake_minimax_h3_varlen_attention_a9de8bf510ed0bfbb02a failed: "
      << cudaGetErrorString(launch_status);
}

}  // namespace cake_host_shim_a442e6313e216d3d

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, cake_host_shim_a442e6313e216d3d::Run);
