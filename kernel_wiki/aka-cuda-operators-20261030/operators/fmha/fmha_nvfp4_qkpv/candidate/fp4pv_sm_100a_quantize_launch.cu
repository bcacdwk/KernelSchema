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
// tvm-ffi direct-source launcher for Cake kernel 'kernel_cake_minimax_h3_varlen_attention_e70c217406e9eafae840'.
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

extern "C" __global__ void kernel_cake_minimax_h3_varlen_attention_e70c217406e9eafae840(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, uint8_t* __restrict__ q_fp4, uint8_t* __restrict__ k_fp4, uint8_t* __restrict__ q_scale, uint8_t* __restrict__ k_scale, uint8_t* __restrict__ v_fp4_t, uint8_t* __restrict__ v_scale_lo, uint8_t* __restrict__ v_scale_hi, int* __restrict__ block_token, int* __restrict__ block_valid, int heads, int PB);


namespace cake_host_shim_e9d5e4c291b569ba {

using tvm::ffi::Optional;
using tvm::ffi::TensorView;

#include "host_checks.cuh"

void Run(TensorView arg_q, TensorView arg_k, TensorView arg_v, TensorView arg_q_fp4, TensorView arg_k_fp4, TensorView arg_q_scale, TensorView arg_k_scale, TensorView arg_v_fp4_t, TensorView arg_v_scale_lo, TensorView arg_v_scale_hi, TensorView arg_block_token, TensorView arg_block_valid, int64_t arg_heads, int64_t arg_PB, int64_t grid_x, int64_t grid_y, int64_t grid_z) {
  DLDevice dev = arg_q.device();
  ScopedCudaDevice device_guard(dev.device_id);
  CheckCudaTensor(arg_q, "q");
  CheckDtype(arg_q, "q", 4, 16, 1);
  CheckContiguous(arg_q, "q");
  CheckCudaTensor(arg_k, "k");
  CheckDtype(arg_k, "k", 4, 16, 1);
  CheckContiguous(arg_k, "k");
  CheckCudaTensor(arg_v, "v");
  CheckDtype(arg_v, "v", 4, 16, 1);
  CheckContiguous(arg_v, "v");
  CheckCudaTensor(arg_q_fp4, "q_fp4");
  CheckDtype(arg_q_fp4, "q_fp4", 1, 8, 1);
  CheckContiguous(arg_q_fp4, "q_fp4");
  CheckCudaTensor(arg_k_fp4, "k_fp4");
  CheckDtype(arg_k_fp4, "k_fp4", 1, 8, 1);
  CheckContiguous(arg_k_fp4, "k_fp4");
  CheckCudaTensor(arg_q_scale, "q_scale");
  CheckDtype(arg_q_scale, "q_scale", 1, 8, 1);
  CheckContiguous(arg_q_scale, "q_scale");
  CheckCudaTensor(arg_k_scale, "k_scale");
  CheckDtype(arg_k_scale, "k_scale", 1, 8, 1);
  CheckContiguous(arg_k_scale, "k_scale");
  CheckCudaTensor(arg_v_fp4_t, "v_fp4_t");
  CheckDtype(arg_v_fp4_t, "v_fp4_t", 1, 8, 1);
  CheckContiguous(arg_v_fp4_t, "v_fp4_t");
  CheckCudaTensor(arg_v_scale_lo, "v_scale_lo");
  CheckDtype(arg_v_scale_lo, "v_scale_lo", 1, 8, 1);
  CheckContiguous(arg_v_scale_lo, "v_scale_lo");
  CheckCudaTensor(arg_v_scale_hi, "v_scale_hi");
  CheckDtype(arg_v_scale_hi, "v_scale_hi", 1, 8, 1);
  CheckContiguous(arg_v_scale_hi, "v_scale_hi");
  CheckCudaTensor(arg_block_token, "block_token");
  CheckDtype(arg_block_token, "block_token", 0, 32, 1);
  CheckContiguous(arg_block_token, "block_token");
  CheckCudaTensor(arg_block_valid, "block_valid");
  CheckDtype(arg_block_valid, "block_valid", 0, 32, 1);
  CheckContiguous(arg_block_valid, "block_valid");
  TVM_FFI_CHECK(arg_heads >= -2147483648LL && arg_heads <= 2147483647LL, ValueError)
      << "scalar 'heads' value " << arg_heads
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_PB >= -2147483648LL && arg_PB <= 2147483647LL, ValueError)
      << "scalar 'PB' value " << arg_PB
      << " is outside i32 range [-2147483648, 2147483647]";
  CheckSameCudaDevice(arg_k, arg_q, "k", "q");
  CheckSameCudaDevice(arg_v, arg_q, "v", "q");
  CheckSameCudaDevice(arg_q_fp4, arg_q, "q_fp4", "q");
  CheckSameCudaDevice(arg_k_fp4, arg_q, "k_fp4", "q");
  CheckSameCudaDevice(arg_q_scale, arg_q, "q_scale", "q");
  CheckSameCudaDevice(arg_k_scale, arg_q, "k_scale", "q");
  CheckSameCudaDevice(arg_v_fp4_t, arg_q, "v_fp4_t", "q");
  CheckSameCudaDevice(arg_v_scale_lo, arg_q, "v_scale_lo", "q");
  CheckSameCudaDevice(arg_v_scale_hi, arg_q, "v_scale_hi", "q");
  CheckSameCudaDevice(arg_block_token, arg_q, "block_token", "q");
  CheckSameCudaDevice(arg_block_valid, arg_q, "block_valid", "q");
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << "launch grid dimensions must be positive, got (" << grid_x << ", " << grid_y
      << ", " << grid_z << ")";

  cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(dev.device_type, dev.device_id);
  __nv_bfloat16* p_q = static_cast<__nv_bfloat16*>(arg_q.data_ptr());
  __nv_bfloat16* p_k = static_cast<__nv_bfloat16*>(arg_k.data_ptr());
  __nv_bfloat16* p_v = static_cast<__nv_bfloat16*>(arg_v.data_ptr());
  uint8_t* p_q_fp4 = static_cast<uint8_t*>(arg_q_fp4.data_ptr());
  uint8_t* p_k_fp4 = static_cast<uint8_t*>(arg_k_fp4.data_ptr());
  uint8_t* p_q_scale = static_cast<uint8_t*>(arg_q_scale.data_ptr());
  uint8_t* p_k_scale = static_cast<uint8_t*>(arg_k_scale.data_ptr());
  uint8_t* p_v_fp4_t = static_cast<uint8_t*>(arg_v_fp4_t.data_ptr());
  uint8_t* p_v_scale_lo = static_cast<uint8_t*>(arg_v_scale_lo.data_ptr());
  uint8_t* p_v_scale_hi = static_cast<uint8_t*>(arg_v_scale_hi.data_ptr());
  int* p_block_token = static_cast<int*>(arg_block_token.data_ptr());
  int* p_block_valid = static_cast<int*>(arg_block_valid.data_ptr());
  int32_t v_heads = (int32_t)arg_heads;
  int32_t v_PB = (int32_t)arg_PB;
  void* kargs[] = {&p_q, &p_k, &p_v, &p_q_fp4, &p_k_fp4, &p_q_scale, &p_k_scale, &p_v_fp4_t, &p_v_scale_lo, &p_v_scale_hi, &p_block_token, &p_block_valid, &v_heads, &v_PB};


  dim3 grid((uint32_t)grid_x, (uint32_t)grid_y, (uint32_t)grid_z);
  dim3 block(256u, 1u, 1u);

  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = 8704u;
  config.stream = stream;
  cudaError_t launch_status = cudaLaunchKernelExC(
      &config, reinterpret_cast<const void*>(kernel_cake_minimax_h3_varlen_attention_e70c217406e9eafae840), kargs);
  TVM_FFI_CHECK(launch_status == cudaSuccess, RuntimeError)
      << "cudaLaunchKernelExC for kernel_cake_minimax_h3_varlen_attention_e70c217406e9eafae840 failed: "
      << cudaGetErrorString(launch_status);
}

}  // namespace cake_host_shim_e9d5e4c291b569ba

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, cake_host_shim_e9d5e4c291b569ba::Run);
