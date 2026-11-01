// Shared SM100/SM103 combine; build.py supplies the original module symbols.
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
// tvm-ffi direct-source launcher for Cake kernel 'AKA_COMBINE_KERNEL'.
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

extern "C" __global__ void AKA_COMBINE_KERNEL(__half* __restrict__ partial_O, float* __restrict__ partial_ML, int* __restrict__ combine_table, int* __restrict__ seg_begin, int* __restrict__ seg_len, __nv_bfloat16* __restrict__ O, int num_heads);


namespace AKA_COMBINE_HOST_NAMESPACE {

using tvm::ffi::Optional;
using tvm::ffi::TensorView;

#include "host_checks.cuh"

void Run(TensorView arg_partial_O, TensorView arg_partial_ML, TensorView arg_combine_table, TensorView arg_seg_begin, TensorView arg_seg_len, TensorView arg_O, int64_t arg_num_heads, int64_t grid_x, int64_t grid_y, int64_t grid_z) {
  DLDevice dev = arg_partial_O.device();
  ScopedCudaDevice device_guard(dev.device_id);
  CheckCudaTensor(arg_partial_O, "partial_O");
  CheckDtype(arg_partial_O, "partial_O", 2, 16, 1);
  CheckContiguous(arg_partial_O, "partial_O");
  CheckCudaTensor(arg_partial_ML, "partial_ML");
  CheckDtype(arg_partial_ML, "partial_ML", 2, 32, 1);
  CheckContiguous(arg_partial_ML, "partial_ML");
  CheckCudaTensor(arg_combine_table, "combine_table");
  CheckDtype(arg_combine_table, "combine_table", 0, 32, 1);
  CheckContiguous(arg_combine_table, "combine_table");
  CheckCudaTensor(arg_seg_begin, "seg_begin");
  CheckDtype(arg_seg_begin, "seg_begin", 0, 32, 1);
  CheckContiguous(arg_seg_begin, "seg_begin");
  CheckCudaTensor(arg_seg_len, "seg_len");
  CheckDtype(arg_seg_len, "seg_len", 0, 32, 1);
  CheckContiguous(arg_seg_len, "seg_len");
  CheckCudaTensor(arg_O, "O");
  CheckDtype(arg_O, "O", 4, 16, 1);
  CheckContiguous(arg_O, "O");
  TVM_FFI_CHECK(arg_num_heads >= -2147483648LL && arg_num_heads <= 2147483647LL, ValueError)
      << "scalar 'num_heads' value " << arg_num_heads
      << " is outside i32 range [-2147483648, 2147483647]";
  CheckSameCudaDevice(arg_partial_ML, arg_partial_O, "partial_ML", "partial_O");
  CheckSameCudaDevice(arg_combine_table, arg_partial_O, "combine_table", "partial_O");
  CheckSameCudaDevice(arg_seg_begin, arg_partial_O, "seg_begin", "partial_O");
  CheckSameCudaDevice(arg_seg_len, arg_partial_O, "seg_len", "partial_O");
  CheckSameCudaDevice(arg_O, arg_partial_O, "O", "partial_O");
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << "launch grid dimensions must be positive, got (" << grid_x << ", " << grid_y
      << ", " << grid_z << ")";

  cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(dev.device_type, dev.device_id);
  __half* p_partial_O = static_cast<__half*>(arg_partial_O.data_ptr());
  float* p_partial_ML = static_cast<float*>(arg_partial_ML.data_ptr());
  int* p_combine_table = static_cast<int*>(arg_combine_table.data_ptr());
  int* p_seg_begin = static_cast<int*>(arg_seg_begin.data_ptr());
  int* p_seg_len = static_cast<int*>(arg_seg_len.data_ptr());
  __nv_bfloat16* p_O = static_cast<__nv_bfloat16*>(arg_O.data_ptr());
  int32_t v_num_heads = (int32_t)arg_num_heads;
  void* kargs[] = {&p_partial_O, &p_partial_ML, &p_combine_table, &p_seg_begin, &p_seg_len, &p_O, &v_num_heads};


  dim3 grid((uint32_t)grid_x, (uint32_t)grid_y, (uint32_t)grid_z);
  dim3 block(128u, 1u, 1u);

  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = 0u;
  config.stream = stream;
  cudaError_t launch_status = cudaLaunchKernelExC(
      &config, reinterpret_cast<const void*>(AKA_COMBINE_KERNEL), kargs);
  TVM_FFI_CHECK(launch_status == cudaSuccess, RuntimeError)
      << "cudaLaunchKernelExC for AKA_COMBINE_KERNEL failed: "
      << cudaGetErrorString(launch_status);
}

}  // namespace AKA_COMBINE_HOST_NAMESPACE

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, AKA_COMBINE_HOST_NAMESPACE::Run);
