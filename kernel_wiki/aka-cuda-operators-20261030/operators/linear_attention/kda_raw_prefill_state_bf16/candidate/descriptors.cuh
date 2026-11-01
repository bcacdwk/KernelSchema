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

#pragma once

#include "host_helpers.cuh"

namespace flashinfer {
namespace flash_kda {

constexpr int64_t kBt16ChunkTokens = 16;

inline void CheckBt16DenseTensor(const TensorView& tensor, const char* name, int32_t device_id,
                                 DLDataType dtype) {
  CheckCudaTensor(tensor, name, device_id);
  CheckDtype(tensor, name, dtype);
}

inline void CheckBt16TokenTensor(const TensorView& tensor, const char* name, int32_t device_id,
                                 int64_t token_count, int64_t num_heads) {
  CheckBt16DenseTensor(tensor, name, device_id, dl_bfloat16);
  TVM_FFI_ICHECK(tensor.ndim() >= 3 && tensor.size(tensor.ndim() - 2) == num_heads &&
                 tensor.size(tensor.ndim() - 1) == kHeadDim &&
                 tensor.numel() == token_count * num_heads * kHeadDim)
      << name << " must match flattened [tokens, H, 128] storage";
}

inline int64_t CheckBt16PrepareInputs(
    const TensorView& q, const TensorView& k, const TensorView& raw_gate,
    const TensorView& beta_logits, const TensorView& a_log, const TensorView& dt_bias,
    const TensorView& cu_seqlens,
    const TensorView& ws_qd, const TensorView& ws_kd, const TensorView& ws_w,
    const TensorView& ws_qk_t, const TensorView& ws_diag, int64_t prepare_total_ctas, int64_t total_chunks,
    int64_t num_heads, double gate_lower_bound) {
  TVM_FFI_ICHECK(q.device().device_type == kDLCUDA) << "q must be a CUDA tensor";
  const int32_t device_id = q.device().device_id;
  CheckFlashKDATarget(device_id);
  TVM_FFI_ICHECK(num_heads > 0 && num_heads <= std::numeric_limits<int32_t>::max())
      << "num_heads must be in the positive int32 range";
  TVM_FFI_ICHECK(total_chunks > 0 && total_chunks <= std::numeric_limits<int32_t>::max())
      << "total_chunks must be in the positive int32 range";
  TVM_FFI_ICHECK(prepare_total_ctas >= num_heads &&
                 prepare_total_ctas <= num_heads * total_chunks &&
                 prepare_total_ctas <= std::numeric_limits<uint32_t>::max())
      << "prepare_total_ctas must be in [H, H * total_chunks]";
  TVM_FFI_ICHECK(std::isfinite(gate_lower_bound) && gate_lower_bound < 0.0 &&
                 std::isfinite(static_cast<float>(gate_lower_bound)))
      << "gate_lower_bound must be finite, negative, and representable as float32";

  CheckBt16DenseTensor(q, "q", device_id, dl_bfloat16);
  TVM_FFI_ICHECK(q.ndim() >= 3 && q.size(q.ndim() - 2) == num_heads &&
                 q.size(q.ndim() - 1) == kHeadDim)
      << "q must have trailing [H, 128] dimensions";
  const int64_t token_count = q.numel() / (num_heads * kHeadDim);
  TVM_FFI_ICHECK(token_count > 0) << "q must contain at least one token";
  CheckBt16TokenTensor(k, "k", device_id, token_count, num_heads);
  CheckBt16TokenTensor(raw_gate, "raw_gate", device_id, token_count, num_heads);

  CheckBt16DenseTensor(beta_logits, "beta_logits", device_id, dl_bfloat16);
  TVM_FFI_ICHECK(beta_logits.ndim() >= 2 && beta_logits.size(beta_logits.ndim() - 1) == num_heads &&
                 beta_logits.numel() == token_count * num_heads)
      << "beta_logits must match flattened [tokens, H] storage";
  CheckBt16DenseTensor(a_log, "a_log", device_id, dl_float32);
  CheckBt16DenseTensor(dt_bias, "dt_bias", device_id, dl_float32);
  TVM_FFI_ICHECK(a_log.numel() == num_heads) << "a_log must contain H elements";
  TVM_FFI_ICHECK(dt_bias.numel() == num_heads * kHeadDim)
      << "dt_bias must contain H * 128 elements";
  CheckBt16DenseTensor(cu_seqlens, "cu_seqlens", device_id, dl_int64);
  TVM_FFI_ICHECK(cu_seqlens.ndim() == 1 && cu_seqlens.numel() >= 2)
      << "cu_seqlens must contain N + 1 entries";

  const int64_t padded_tokens = total_chunks * kBt16ChunkTokens;
  for (const auto& named : {std::pair<const TensorView*, const char*>(&ws_qd, "ws_qd"),
                            std::pair<const TensorView*, const char*>(&ws_kd, "ws_kd"),
                            std::pair<const TensorView*, const char*>(&ws_w, "ws_w")}) {
    CheckBt16DenseTensor(*named.first, named.second, device_id, dl_bfloat16);
    TVM_FFI_ICHECK(named.first->ndim() == 4 && named.first->size(0) == 1 &&
                   named.first->size(1) == num_heads && named.first->size(2) == padded_tokens &&
                   named.first->size(3) == kHeadDim)
        << named.second << " must have shape [1, H, total_chunks * 16, 128]";
  }
  CheckBt16DenseTensor(ws_qk_t, "ws_qk_t", device_id, dl_bfloat16);
  TVM_FFI_ICHECK(ws_qk_t.ndim() == 5 && ws_qk_t.size(0) == 1 && ws_qk_t.size(1) == num_heads &&
                 ws_qk_t.size(2) == total_chunks && ws_qk_t.size(3) == kBt16ChunkTokens &&
                 ws_qk_t.size(4) == kBt16ChunkTokens)
      << "ws_qk_t must have shape [1, H, total_chunks, 16, 16]";
  CheckBt16DenseTensor(ws_diag, "ws_diag", device_id, dl_float32);
  TVM_FFI_ICHECK(ws_diag.ndim() == 4 && ws_diag.size(0) == 1 && ws_diag.size(1) == num_heads &&
                 ws_diag.size(2) == total_chunks && ws_diag.size(3) == kHeadDim)
      << "ws_diag must have shape [1, H, total_chunks, 128]";
  return token_count;
}

template <int BoxValueSplits>
inline CUtensorMap EncodeBt16FactorTma(const TensorView& tensor, const char* name) {
  static_assert(BoxValueSplits == 1 || BoxValueSplits == 2);
  const int64_t d1 = tensor.size(tensor.ndim() - 1);
  const int64_t d2 = tensor.size(tensor.ndim() - 2);
  const int64_t outer2 = tensor.numel() / (d1 * d2);
  uint64_t global_dim[4] = {64, static_cast<uint64_t>(d2), static_cast<uint64_t>(outer2),
                            static_cast<uint64_t>(d1 / 64)};
  uint64_t global_strides[3] = {static_cast<uint64_t>(d1 * sizeof(__nv_bfloat16)),
                                static_cast<uint64_t>(d2 * d1 * sizeof(__nv_bfloat16)),
                                static_cast<uint64_t>(64 * sizeof(__nv_bfloat16))};
  uint32_t box_dim[4] = {64, 16, 1, BoxValueSplits};
  uint32_t elem_strides[4] = {1, 1, 1, 1};
  CUtensorMap map{};
  const CUresult result = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, tensor.data_ptr(), global_dim, global_strides,
      box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for " << name << " with CUresult=" << int(result);
  return map;
}

inline CUtensorMap EncodeBt16QkWorkspaceTma(const TensorView& tensor) {
  uint64_t global_dim[5] = {
      static_cast<uint64_t>(tensor.size(4)), static_cast<uint64_t>(tensor.size(3)),
      static_cast<uint64_t>(tensor.size(2)), static_cast<uint64_t>(tensor.size(1)),
      static_cast<uint64_t>(tensor.size(0))};
  uint64_t global_strides[4] = {static_cast<uint64_t>(tensor.stride(3) * sizeof(__nv_bfloat16)),
                                static_cast<uint64_t>(tensor.stride(2) * sizeof(__nv_bfloat16)),
                                static_cast<uint64_t>(tensor.stride(1) * sizeof(__nv_bfloat16)),
                                static_cast<uint64_t>(tensor.stride(0) * sizeof(__nv_bfloat16))};
  uint32_t box_dim[5] = {16, 16, 1, 1, 1};
  uint32_t elem_strides[5] = {1, 1, 1, 1, 1};
  CUtensorMap map{};
  const CUresult result = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, tensor.data_ptr(), global_dim, global_strides,
      box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_32B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for ws_qk with CUresult=" << int(result);
  return map;
}

inline CUtensorMap EncodeBt16DiagWorkspaceTma(const TensorView& tensor) {
  uint64_t global_dim[4] = {
      static_cast<uint64_t>(tensor.size(3)), static_cast<uint64_t>(tensor.size(2)),
      static_cast<uint64_t>(tensor.size(1)), static_cast<uint64_t>(tensor.size(0))};
  uint64_t global_strides[3] = {static_cast<uint64_t>(tensor.stride(2) * sizeof(float)),
                                static_cast<uint64_t>(tensor.stride(1) * sizeof(float)),
                                static_cast<uint64_t>(tensor.stride(0) * sizeof(float))};
  uint32_t box_dim[4] = {128, 1, 1, 1};
  uint32_t elem_strides[4] = {1, 1, 1, 1};
  CUtensorMap map{};
  const CUresult result = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, tensor.data_ptr(), global_dim, global_strides,
      box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for ws_diag with CUresult=" << int(result);
  return map;
}

struct Bt16PrepareTmaMaps {
  CUtensorMap q;
  CUtensorMap k;
  CUtensorMap raw_gate;
  CUtensorMap beta_logits;
  CUtensorMap ws_qd;
  CUtensorMap ws_kd;
  CUtensorMap ws_w;
};

// Match native CuTe compact beta view with a 16-byte-aligned row stride.
inline CUtensorMap EncodeBt16NativeBetaTma(const TensorView& beta) {
  const int64_t heads=beta.size(beta.ndim()-1), tokens=beta.numel()/heads;
  const int64_t lsb=heads & -heads, group=8/std::min<int64_t>(lsb,8);
  const int64_t inner=group*heads;
  TVM_FFI_ICHECK(beta.IsContiguous() && reinterpret_cast<uintptr_t>(beta.data_ptr())%16==0);
  const uint32_t cols=group==1 ? 8 : static_cast<uint32_t>(inner);
  const uint32_t rows=16/group+(group>1);
  TVM_FFI_ICHECK(cols*rows*2<=512) << "Native beta TMA tile exceeds the 512-byte stage";
  const uint64_t dimensions[2]={static_cast<uint64_t>(inner),static_cast<uint64_t>((tokens+group-1)/group)};
  const uint64_t strides[1]={static_cast<uint64_t>(inner*2)};
  const uint32_t box[2]={cols,rows}, elements[2]={1,1};
  CUtensorMap map{};
  const CUresult result=cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,
      beta.data_ptr(),dimensions,strides,box,elements,CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(result==CUDA_SUCCESS) << "Native beta TMA encode failed: " << int(result);
  return map;
}

template <bool UsesBetaTma>
inline Bt16PrepareTmaMaps EncodeBt16PrepareTma(
    const TensorView& q, const TensorView& k, const TensorView& raw_gate,
    const TensorView& beta_logits, const TensorView& ws_qd, const TensorView& ws_kd,
    const TensorView& ws_w) {
  const CUtensorMap raw_gate_map = EncodeGateTma<16>(raw_gate);
  CUtensorMap beta_map = raw_gate_map;
  if constexpr (UsesBetaTma) beta_map = EncodeBt16NativeBetaTma(beta_logits);
  return {EncodeQkTma<16>(q, "q"), EncodeQkTma<16>(k, "k"), raw_gate_map, beta_map,
          EncodeBt16FactorTma<2>(ws_qd, "ws_qd"), EncodeBt16FactorTma<2>(ws_kd, "ws_kd"),
          EncodeBt16FactorTma<1>(ws_w, "ws_w")};
}

struct Bt16ChainTmaMaps {
  CUtensorMap ws_qd;
  CUtensorMap ws_kd;
  CUtensorMap ws_w;
  CUtensorMap ws_qk;
  CUtensorMap ws_diag;
  CUtensorMap v;
  CUtensorMap out;
};

inline Bt16ChainTmaMaps EncodeBt16ChainTma(const TensorView& ws_qd, const TensorView& ws_kd,
                                               const TensorView& ws_w, const TensorView& ws_qk,
                                               const TensorView& ws_diag, const TensorView& v,
                                               const TensorView& out) {
  return {EncodeBt16FactorTma<2>(ws_qd, "ws_qd"), EncodeBt16FactorTma<2>(ws_kd, "ws_kd"),
          EncodeBt16FactorTma<2>(ws_w, "ws_w"), EncodeBt16QkWorkspaceTma(ws_qk),
          EncodeBt16DiagWorkspaceTma(ws_diag), EncodeValueTma<64, 16>(v),
          EncodeOutputTma<64, 16>(out)};
}

}  // namespace flash_kda
}  // namespace flashinfer
