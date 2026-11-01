
// Editable source; provenance is recorded in ../LICENSES/.
// tvm-ffi direct-source launcher for Cake kernel 'kernel_cake_gdn_cp_1fbdb4b8eac3c2561392'.
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

extern "C" __global__ void kernel_cake_gdn_cp_1fbdb4b8eac3c2561392(float* __restrict__ local_transfer, float* __restrict__ local_state, float* __restrict__ initial_state, float* __restrict__ initial_state_workspace, float* __restrict__ fixed_state, float* __restrict__ output_state, long long* __restrict__ cu_seqlens, int chunk_len, int total_cp_chunks, int num_seqs, int num_heads);


namespace cake_host_shim_2efdcbd0d13ea373 {

using tvm::ffi::Optional;
using tvm::ffi::TensorView;

#include "host_checks.cuh"

void Run(TensorView arg_local_transfer, TensorView arg_local_state, TensorView arg_initial_state, TensorView arg_initial_state_workspace, TensorView arg_fixed_state, TensorView arg_output_state, TensorView arg_cu_seqlens, int64_t arg_chunk_len, int64_t arg_total_cp_chunks, int64_t arg_num_seqs, int64_t arg_num_heads, int64_t grid_x, int64_t grid_y, int64_t grid_z) {
  DLDevice dev = arg_local_transfer.device();
  ScopedCudaDevice device_guard(dev.device_id);
  CheckCudaTensor(arg_local_transfer, "local_transfer");
  CheckDtype(arg_local_transfer, "local_transfer", 2, 32, 1);
  CheckContiguous(arg_local_transfer, "local_transfer");
  CheckCudaTensor(arg_local_state, "local_state");
  CheckDtype(arg_local_state, "local_state", 2, 32, 1);
  CheckContiguous(arg_local_state, "local_state");
  CheckCudaTensor(arg_initial_state, "initial_state");
  CheckDtype(arg_initial_state, "initial_state", 2, 32, 1);
  CheckContiguous(arg_initial_state, "initial_state");
  CheckCudaTensor(arg_initial_state_workspace, "initial_state_workspace");
  CheckDtype(arg_initial_state_workspace, "initial_state_workspace", 2, 32, 1);
  CheckContiguous(arg_initial_state_workspace, "initial_state_workspace");
  CheckCudaTensor(arg_fixed_state, "fixed_state");
  CheckDtype(arg_fixed_state, "fixed_state", 2, 32, 1);
  CheckContiguous(arg_fixed_state, "fixed_state");
  CheckCudaTensor(arg_output_state, "output_state");
  CheckDtype(arg_output_state, "output_state", 2, 32, 1);
  CheckContiguous(arg_output_state, "output_state");
  CheckCudaTensor(arg_cu_seqlens, "cu_seqlens");
  CheckDtype(arg_cu_seqlens, "cu_seqlens", 0, 64, 1);
  CheckContiguous(arg_cu_seqlens, "cu_seqlens");
  TVM_FFI_CHECK(arg_chunk_len >= -2147483648LL && arg_chunk_len <= 2147483647LL, ValueError)
      << "scalar 'chunk_len' value " << arg_chunk_len
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_total_cp_chunks >= -2147483648LL && arg_total_cp_chunks <= 2147483647LL, ValueError)
      << "scalar 'total_cp_chunks' value " << arg_total_cp_chunks
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_num_seqs >= -2147483648LL && arg_num_seqs <= 2147483647LL, ValueError)
      << "scalar 'num_seqs' value " << arg_num_seqs
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_num_heads >= -2147483648LL && arg_num_heads <= 2147483647LL, ValueError)
      << "scalar 'num_heads' value " << arg_num_heads
      << " is outside i32 range [-2147483648, 2147483647]";
  CheckSameCudaDevice(arg_local_state, arg_local_transfer, "local_state", "local_transfer");
  CheckSameCudaDevice(arg_initial_state, arg_local_transfer, "initial_state", "local_transfer");
  CheckSameCudaDevice(arg_initial_state_workspace, arg_local_transfer, "initial_state_workspace", "local_transfer");
  CheckSameCudaDevice(arg_fixed_state, arg_local_transfer, "fixed_state", "local_transfer");
  CheckSameCudaDevice(arg_output_state, arg_local_transfer, "output_state", "local_transfer");
  CheckSameCudaDevice(arg_cu_seqlens, arg_local_transfer, "cu_seqlens", "local_transfer");
  CheckCurrentCudaDevice(arg_local_transfer, "local_transfer");
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << "launch grid dimensions must be positive, got (" << grid_x << ", " << grid_y
      << ", " << grid_z << ")";

  cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(dev.device_type, dev.device_id);
  float* p_local_transfer = static_cast<float*>(arg_local_transfer.data_ptr());
  float* p_local_state = static_cast<float*>(arg_local_state.data_ptr());
  float* p_initial_state = static_cast<float*>(arg_initial_state.data_ptr());
  float* p_initial_state_workspace = static_cast<float*>(arg_initial_state_workspace.data_ptr());
  float* p_fixed_state = static_cast<float*>(arg_fixed_state.data_ptr());
  float* p_output_state = static_cast<float*>(arg_output_state.data_ptr());
  long long* p_cu_seqlens = static_cast<long long*>(arg_cu_seqlens.data_ptr());
  int32_t v_chunk_len = (int32_t)arg_chunk_len;
  int32_t v_total_cp_chunks = (int32_t)arg_total_cp_chunks;
  int32_t v_num_seqs = (int32_t)arg_num_seqs;
  int32_t v_num_heads = (int32_t)arg_num_heads;
  void* kargs[] = {&p_local_transfer, &p_local_state, &p_initial_state, &p_initial_state_workspace, &p_fixed_state, &p_output_state, &p_cu_seqlens, &v_chunk_len, &v_total_cp_chunks, &v_num_seqs, &v_num_heads};


  dim3 grid((uint32_t)grid_x, (uint32_t)grid_y, (uint32_t)grid_z);
  dim3 block(128u, 1u, 1u);

  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = 2048u;
  config.stream = stream;
  cudaError_t launch_status = cudaLaunchKernelExC(
      &config, reinterpret_cast<const void*>(kernel_cake_gdn_cp_1fbdb4b8eac3c2561392), kargs);
  TVM_FFI_CHECK(launch_status == cudaSuccess, RuntimeError)
      << "cudaLaunchKernelExC for kernel_cake_gdn_cp_1fbdb4b8eac3c2561392 failed: "
      << cudaGetErrorString(launch_status);
}

}  // namespace cake_host_shim_2efdcbd0d13ea373

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, cake_host_shim_2efdcbd0d13ea373::Run);
