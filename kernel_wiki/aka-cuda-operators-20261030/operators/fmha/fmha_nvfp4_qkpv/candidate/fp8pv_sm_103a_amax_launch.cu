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
// tvm-ffi direct-source launcher for Cake kernel 'kernel_cake_minimax_h3_varlen_attention_09d2c9ea5209af5c2d70'.
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

extern "C" __global__ void kernel_cake_minimax_h3_varlen_attention_09d2c9ea5209af5c2d70(__nv_bfloat16* __restrict__ v, float* __restrict__ v_amax_partial, int num_vectors, int num_chunks);


namespace cake_host_shim_405a4c3c312211cd {

using tvm::ffi::Optional;
using tvm::ffi::TensorView;

#include "host_checks.cuh"

void Run(TensorView arg_v, TensorView arg_v_amax_partial, int64_t arg_num_vectors, int64_t arg_num_chunks, int64_t grid_x, int64_t grid_y, int64_t grid_z) {
  DLDevice dev = arg_v.device();
  ScopedCudaDevice device_guard(dev.device_id);
  CheckCudaTensor(arg_v, "v");
  CheckDtype(arg_v, "v", 4, 16, 1);
  CheckContiguous(arg_v, "v");
  CheckCudaTensor(arg_v_amax_partial, "v_amax_partial");
  CheckDtype(arg_v_amax_partial, "v_amax_partial", 2, 32, 1);
  CheckContiguous(arg_v_amax_partial, "v_amax_partial");
  TVM_FFI_CHECK(arg_num_vectors >= -2147483648LL && arg_num_vectors <= 2147483647LL, ValueError)
      << "scalar 'num_vectors' value " << arg_num_vectors
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_num_chunks >= -2147483648LL && arg_num_chunks <= 2147483647LL, ValueError)
      << "scalar 'num_chunks' value " << arg_num_chunks
      << " is outside i32 range [-2147483648, 2147483647]";
  CheckSameCudaDevice(arg_v_amax_partial, arg_v, "v_amax_partial", "v");
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << "launch grid dimensions must be positive, got (" << grid_x << ", " << grid_y
      << ", " << grid_z << ")";

  cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(dev.device_type, dev.device_id);
  __nv_bfloat16* p_v = static_cast<__nv_bfloat16*>(arg_v.data_ptr());
  float* p_v_amax_partial = static_cast<float*>(arg_v_amax_partial.data_ptr());
  int32_t v_num_vectors = (int32_t)arg_num_vectors;
  int32_t v_num_chunks = (int32_t)arg_num_chunks;
  void* kargs[] = {&p_v, &p_v_amax_partial, &v_num_vectors, &v_num_chunks};


  dim3 grid((uint32_t)grid_x, (uint32_t)grid_y, (uint32_t)grid_z);
  dim3 block(256u, 1u, 1u);

  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = 128u;
  config.stream = stream;
  cudaError_t launch_status = cudaLaunchKernelExC(
      &config, reinterpret_cast<const void*>(kernel_cake_minimax_h3_varlen_attention_09d2c9ea5209af5c2d70), kargs);
  TVM_FFI_CHECK(launch_status == cudaSuccess, RuntimeError)
      << "cudaLaunchKernelExC for kernel_cake_minimax_h3_varlen_attention_09d2c9ea5209af5c2d70 failed: "
      << cudaGetErrorString(launch_status);
}

}  // namespace cake_host_shim_405a4c3c312211cd

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, cake_host_shim_405a4c3c312211cd::Run);
