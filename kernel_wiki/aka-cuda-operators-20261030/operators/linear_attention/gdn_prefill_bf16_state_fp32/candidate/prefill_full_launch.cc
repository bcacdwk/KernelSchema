
// clang-format off
// Editable source; provenance is recorded in ../LICENSES/.
// tvm-ffi host launcher for GDN non-CP kernel 'kernel_flashinfer_blackwell_gdn_prefill_initial'.
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "ffi.cuh"
#include <tvm/ffi/extra/cuda/cubin_launcher.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

TVM_FFI_EMBED_CUBIN(flashinfer_blackwell_gdn_prefill_initial_1230278b65);

namespace cake_gdn_host_shim_785de87ac034a54f {

using tvm::ffi::Optional;
using tvm::ffi::TensorView;

#include "host_checks.cuh"

#include <dlfcn.h>

// CUDA 13.4 adds an oversized shared-memory mode (function/launch attribute
// SHARED_MEMORY_MODE = 3) that lets a kernel exceed the standard
// MaxSharedMemoryPerBlockOptin ceiling up to device attribute 150
// (OversizedSharedMemoryPerBlock), at the cost of an 8 KiB L1 carveout.
#if (TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API && defined(CUDA_VERSION) && CUDA_VERSION >= 13040) || \
    (!TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API && defined(CUDART_VERSION) && CUDART_VERSION >= 13040)
#define CAKE_GDN_HAS_OVERSIZED_SMEM 1
#else
#define CAKE_GDN_HAS_OVERSIZED_SMEM 0
#endif

inline bool CakeGDNConfigureDynamicSmem(tvm::ffi::CubinKernel& kernel, int device_id,
                                     int smem_bytes, signed char* cache, int cache_len) {
  namespace cuda_api = tvm::ffi::cuda_api;
  TVM_FFI_CHECK(device_id >= 0 && device_id < cache_len, RuntimeError)
      << "dynamic-SMEM opt-in cache does not cover cuda:" << device_id;
  if (cache[device_id] != 0) {
    return cache[device_id] == 2;
  }
  auto device = cuda_api::GetDeviceHandle(device_id);
  int optin_max = 0;
  cuda_api::ResultType err = cuda_api::GetDeviceAttribute(
      &optin_max,
      /* CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN /
         cudaDevAttrMaxSharedMemoryPerBlockOptin */
      cuda_api::DeviceAttrType(97), device);
  TVM_FFI_CHECK(err == cuda_api::kSuccess, RuntimeError)
      << "querying MaxSharedMemoryPerBlockOptin failed for cuda:" << device_id;
  if (smem_bytes <= optin_max) {
    err = cuda_api::SetKernelMaxDynamicSharedMem(kernel.GetHandle(), smem_bytes, device);
    TVM_FFI_CHECK(err == cuda_api::kSuccess, RuntimeError)
        << "MAX_DYNAMIC_SHARED_SIZE_BYTES=" << smem_bytes << " rejected for cuda:" << device_id;
    cache[device_id] = 1;
    return false;
  }
#if CAKE_GDN_HAS_OVERSIZED_SMEM
  int oversized_max = 0;
  err = cuda_api::GetDeviceAttribute(
      &oversized_max,
      /* CU_DEVICE_ATTRIBUTE_MAX_OVERSIZED_SHARED_MEMORY_PER_BLOCK /
         cudaDevAttrOversizedSharedMemoryPerBlock */
      cuda_api::DeviceAttrType(150), device);
  TVM_FFI_CHECK(err == cuda_api::kSuccess && oversized_max >= smem_bytes, RuntimeError)
      << "dynamic smem " << smem_bytes << " B exceeds the standard opt-in ceiling ("
      << optin_max << " B) on cuda:" << device_id << " and the oversized ceiling is "
      << oversized_max << " B";
  // Also set the function-level shared-memory mode: profiler-instrumented
  // launches (ncu) honor only the function-level attribute for oversized
  // dynamic SMEM and fail with LaunchFailed on the launch attribute alone.
#if TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API
  err = cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE,
                             CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY,
                             kernel.GetHandle(), device);
  TVM_FFI_CHECK(err == cuda_api::kSuccess, RuntimeError)
      << "setting SHARED_MEMORY_MODE=ALLOW_OVERSIZED failed for cuda:" << device_id
      << " (error " << static_cast<int>(err) << ")";
#else
  // The process cudart may predate 13.4 (e.g. torch's pip cudart 13.0) and
  // reject cudaFuncAttributeSharedMemoryMode with cudaErrorInvalidValue even
  // though the driver supports the mode, so call the driver entry point
  // directly: cudaKernel_t is interchangeable with CUkernel, the runtime
  // device ordinal is the CUdevice, and libcuda.so.1 is already loaded.
  {
    using CakeGDNCuKernelSetAttributeFn = int (*)(int attrib, int val, void* kernel, int dev);
    static const auto cake_gdn_cu_kernel_set_attribute =
        reinterpret_cast<CakeGDNCuKernelSetAttributeFn>(dlsym(RTLD_DEFAULT, "cuKernelSetAttribute"));
    TVM_FFI_CHECK(cake_gdn_cu_kernel_set_attribute != nullptr, RuntimeError)
        << "cuKernelSetAttribute not resolvable while enabling the oversized shared-memory mode";
    int drv_err = cake_gdn_cu_kernel_set_attribute(
        /* CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE */ 17,
        /* CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY */ 3,
        reinterpret_cast<void*>(kernel.GetHandle()), device_id);
    TVM_FFI_CHECK(drv_err == 0, RuntimeError)
        << "setting SHARED_MEMORY_MODE=ALLOW_OVERSIZED failed for cuda:" << device_id
        << " (driver error " << drv_err << ")";
  }
#endif
  cache[device_id] = 2;
  return true;
#else
  TVM_FFI_THROW(RuntimeError)
      << "dynamic smem " << smem_bytes << " B exceeds the standard opt-in ceiling ("
      << optin_max << " B) on cuda:" << device_id
      << " and this CUDA toolkit predates the 13.4 oversized shared-memory mode";
#endif
}

// 3D TMA descriptor for buffer 'Q' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_Q(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 3, ValueError)
      << "TMA source 'Q' must have at least 3 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'Q' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  int64_t d3 = t.size(t.ndim() - 3);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0 && d3 > 0, ValueError)
      << "TMA source 'Q' trailing dims must be positive";
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'Q' physical strides must be positive";
  int64_t s3 = t.stride(t.ndim() - 3) * 1;
  TVM_FFI_CHECK(s3 > 0, ValueError)
      << "TMA source 'Q' physical strides must be positive";
  uint64_t global_dim[3] = {(uint64_t)(d1), (uint64_t)(d3), (uint64_t)(d2)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'Q' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 1u <= global_dim[2], ValueError)
      << "TMA box (64, 64, 1) exceeds resolved global dims for 'Q'";
  uint64_t global_strides[2] = {
      (uint64_t)((s3 * 16) / 8),
      (uint64_t)((s2 * 16) / 8),
  };
  uint32_t box_dim[3] = {64u, 64u, 1u};
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm{};
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, t.data_ptr(), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'Q') failed: CUresult=" << (int)r;
  return tm;
}

// 3D TMA descriptor for buffer 'K' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_K(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 3, ValueError)
      << "TMA source 'K' must have at least 3 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'K' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  int64_t d3 = t.size(t.ndim() - 3);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0 && d3 > 0, ValueError)
      << "TMA source 'K' trailing dims must be positive";
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'K' physical strides must be positive";
  int64_t s3 = t.stride(t.ndim() - 3) * 1;
  TVM_FFI_CHECK(s3 > 0, ValueError)
      << "TMA source 'K' physical strides must be positive";
  uint64_t global_dim[3] = {(uint64_t)(d1), (uint64_t)(d3), (uint64_t)(d2)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'K' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 1u <= global_dim[2], ValueError)
      << "TMA box (64, 64, 1) exceeds resolved global dims for 'K'";
  uint64_t global_strides[2] = {
      (uint64_t)((s3 * 16) / 8),
      (uint64_t)((s2 * 16) / 8),
  };
  uint32_t box_dim[3] = {64u, 64u, 1u};
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm{};
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, t.data_ptr(), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'K') failed: CUresult=" << (int)r;
  return tm;
}

// 3D TMA descriptor for buffer 'V' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_V(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 3, ValueError)
      << "TMA source 'V' must have at least 3 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'V' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  int64_t d3 = t.size(t.ndim() - 3);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0 && d3 > 0, ValueError)
      << "TMA source 'V' trailing dims must be positive";
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'V' physical strides must be positive";
  int64_t s3 = t.stride(t.ndim() - 3) * 1;
  TVM_FFI_CHECK(s3 > 0, ValueError)
      << "TMA source 'V' physical strides must be positive";
  uint64_t global_dim[3] = {(uint64_t)(d1), (uint64_t)(d3), (uint64_t)(d2)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'V' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 1u <= global_dim[2], ValueError)
      << "TMA box (64, 64, 1) exceeds resolved global dims for 'V'";
  uint64_t global_strides[2] = {
      (uint64_t)((s3 * 16) / 8),
      (uint64_t)((s2 * 16) / 8),
  };
  uint32_t box_dim[3] = {64u, 64u, 1u};
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm{};
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, t.data_ptr(), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'V') failed: CUresult=" << (int)r;
  return tm;
}

// 3D TMA descriptor for buffer 'O' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_O(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 3, ValueError)
      << "TMA source 'O' must have at least 3 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'O' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  int64_t d3 = t.size(t.ndim() - 3);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0 && d3 > 0, ValueError)
      << "TMA source 'O' trailing dims must be positive";
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'O' physical strides must be positive";
  int64_t s3 = t.stride(t.ndim() - 3) * 1;
  TVM_FFI_CHECK(s3 > 0, ValueError)
      << "TMA source 'O' physical strides must be positive";
  uint64_t global_dim[3] = {(uint64_t)(d1), (uint64_t)(d3), (uint64_t)(d2)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'O' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 1u <= global_dim[2], ValueError)
      << "TMA box (64, 64, 1) exceeds resolved global dims for 'O'";
  uint64_t global_strides[2] = {
      (uint64_t)((s3 * 16) / 8),
      (uint64_t)((s2 * 16) / 8),
  };
  uint32_t box_dim[3] = {64u, 64u, 1u};
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm{};
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, t.data_ptr(), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'O') failed: CUresult=" << (int)r;
  return tm;
}

void Run(TensorView arg_Q, TensorView arg_K, TensorView arg_V, TensorView arg_O, TensorView arg_gate, TensorView arg_beta, TensorView arg_cu_seqlens, TensorView arg_state_indices, TensorView arg_initial_state, TensorView arg_output_state, TensorView arg_checkpoint_state, TensorView arg_cu_checkpoints, TensorView arg_tensormap_workspace, int64_t arg_initial_state_stride_slot, int64_t arg_output_state_stride_slot, int64_t arg_checkpoint_every_n_tokens, double arg_scale, int64_t arg_num_seqs, int64_t arg_num_q_heads, int64_t arg_num_v_heads, int64_t arg_total_tiles, int64_t grid_x, int64_t grid_y, int64_t grid_z) {
  DLDevice dev = arg_Q.device();
  ScopedCudaDevice device_guard(dev.device_id);
  CheckCudaTensor(arg_Q, "Q");
  CheckDtype(arg_Q, "Q", 4, 16, 1);
  CheckCudaTensor(arg_K, "K");
  CheckDtype(arg_K, "K", 4, 16, 1);
  CheckCudaTensor(arg_V, "V");
  CheckDtype(arg_V, "V", 4, 16, 1);
  CheckCudaTensor(arg_O, "O");
  CheckDtype(arg_O, "O", 4, 16, 1);
  CheckCudaTensor(arg_gate, "gate");
  CheckDtype(arg_gate, "gate", 2, 32, 1);
  CheckContiguous(arg_gate, "gate");
  CheckCudaTensor(arg_beta, "beta");
  CheckDtype(arg_beta, "beta", 2, 32, 1);
  CheckContiguous(arg_beta, "beta");
  CheckCudaTensor(arg_cu_seqlens, "cu_seqlens");
  CheckDtype(arg_cu_seqlens, "cu_seqlens", 0, 32, 1);
  CheckContiguous(arg_cu_seqlens, "cu_seqlens");
  CheckCudaTensor(arg_state_indices, "state_indices");
  CheckDtype(arg_state_indices, "state_indices", 0, 32, 1);
  CheckContiguous(arg_state_indices, "state_indices");
  CheckCudaTensor(arg_initial_state, "initial_state");
  CheckDtype(arg_initial_state, "initial_state", 2, 32, 1);
  CheckCudaTensor(arg_output_state, "output_state");
  CheckDtype(arg_output_state, "output_state", 2, 32, 1);
  CheckCudaTensor(arg_checkpoint_state, "checkpoint_state");
  CheckDtype(arg_checkpoint_state, "checkpoint_state", 2, 32, 1);
  CheckContiguous(arg_checkpoint_state, "checkpoint_state");
  CheckCudaTensor(arg_cu_checkpoints, "cu_checkpoints");
  CheckDtype(arg_cu_checkpoints, "cu_checkpoints", 0, 32, 1);
  CheckContiguous(arg_cu_checkpoints, "cu_checkpoints");
  CheckCudaTensor(arg_tensormap_workspace, "tensormap_workspace");
  CheckDtype(arg_tensormap_workspace, "tensormap_workspace", 1, 8, 1);
  CheckContiguous(arg_tensormap_workspace, "tensormap_workspace");
  TVM_FFI_CHECK(arg_checkpoint_every_n_tokens >= -2147483648LL && arg_checkpoint_every_n_tokens <= 2147483647LL, ValueError)
      << "scalar 'checkpoint_every_n_tokens' value " << arg_checkpoint_every_n_tokens
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_num_seqs >= -2147483648LL && arg_num_seqs <= 2147483647LL, ValueError)
      << "scalar 'num_seqs' value " << arg_num_seqs
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_num_q_heads >= -2147483648LL && arg_num_q_heads <= 2147483647LL, ValueError)
      << "scalar 'num_q_heads' value " << arg_num_q_heads
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_num_v_heads >= -2147483648LL && arg_num_v_heads <= 2147483647LL, ValueError)
      << "scalar 'num_v_heads' value " << arg_num_v_heads
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_total_tiles >= -2147483648LL && arg_total_tiles <= 2147483647LL, ValueError)
      << "scalar 'total_tiles' value " << arg_total_tiles
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_num_seqs > 0 && arg_num_q_heads == AKA_NUM_Q_HEADS && arg_num_v_heads == AKA_NUM_V_HEADS, ValueError)
      << "prefill sequence/head counts do not match the selected GDN non-CP variant";
  TVM_FFI_CHECK(arg_total_tiles == arg_num_seqs * AKA_NUM_V_HEADS * 1, ValueError)
      << "prefill total_tiles does not match num_seqs and the selected physical route";
  TVM_FFI_CHECK(grid_y == 1 && grid_z == 1 && grid_x <= arg_total_tiles, ValueError)
      << "prefill launch grid does not match the selected persistent route";
  TVM_FFI_CHECK(arg_Q.ndim() == 3 && arg_Q.size(0) > 0 && arg_Q.size(1) == AKA_NUM_Q_HEADS && arg_Q.size(2) == 128LL, ValueError)
      << "Q shape does not match [tokens,Hq,128]";
  TVM_FFI_CHECK(arg_K.ndim() == 3 && arg_K.size(0) == arg_Q.size(0) && arg_K.size(1) == AKA_NUM_Q_HEADS && arg_K.size(2) == 128LL, ValueError)
      << "K shape does not match [tokens,min(Hq,Hv),128]";
  TVM_FFI_CHECK(arg_V.ndim() == 3 && arg_V.size(0) == arg_Q.size(0) && arg_V.size(1) == AKA_NUM_V_HEADS && arg_V.size(2) == 128LL, ValueError)
      << "V shape does not match [tokens,Hv,128]";
  TVM_FFI_CHECK(arg_O.ndim() == 3 && arg_O.size(0) == arg_Q.size(0) && arg_O.size(1) == AKA_NUM_V_HEADS && arg_O.size(2) == 128LL, ValueError)
      << "O shape does not match [tokens,max(Hq,Hv),128]";
  TVM_FFI_CHECK(arg_gate.ndim() == 2 && arg_gate.size(0) == arg_Q.size(0) && arg_gate.size(1) == AKA_NUM_V_HEADS, ValueError)
      << "gate shape does not match [tokens,max(Hq,Hv)]";
  TVM_FFI_CHECK(arg_beta.ndim() == 2 && arg_beta.size(0) == arg_Q.size(0) && arg_beta.size(1) == AKA_NUM_V_HEADS, ValueError)
      << "beta shape does not match [tokens,max(Hq,Hv)]";
  TVM_FFI_CHECK(arg_cu_seqlens.ndim() == 1 && arg_cu_seqlens.size(0) == arg_num_seqs + 1, ValueError)
      << "cu_seqlens must contain num_seqs+1 entries";
  TVM_FFI_CHECK(arg_tensormap_workspace.ndim() == 1 && arg_tensormap_workspace.numel() >= grid_x * 512LL, ValueError)
      << "tensormap_workspace is too small for the launch grid";
  TVM_FFI_CHECK(arg_state_indices.numel() >= 1, ValueError)
      << "state_indices dummy buffer is empty";
  TVM_FFI_CHECK(arg_initial_state.ndim() == 4 && arg_initial_state.size(0) == arg_num_seqs && arg_initial_state.size(1) == AKA_NUM_V_HEADS && arg_initial_state.size(2) == 128LL && arg_initial_state.size(3) == 128LL, ValueError)
      << "initial_state shape does not match the selected state-pool contract";
  TVM_FFI_CHECK(arg_initial_state.stride(3) == 1 && arg_initial_state.stride(2) == 128LL && arg_initial_state.stride(1) == 16384LL && arg_initial_state_stride_slot == arg_initial_state.stride(0) && arg_initial_state_stride_slot >= AKA_NUM_V_HEADS * 16384LL, ValueError)
      << "initial_state inner layout or slot stride is invalid";
  TVM_FFI_CHECK(arg_output_state.ndim() == 4 && arg_output_state.size(0) == arg_num_seqs && arg_output_state.size(1) == AKA_NUM_V_HEADS && arg_output_state.size(2) == 128LL && arg_output_state.size(3) == 128LL, ValueError)
      << "output_state shape does not match the selected state-pool contract";
  TVM_FFI_CHECK(arg_output_state.stride(3) == 1 && arg_output_state.stride(2) == 128LL && arg_output_state.stride(1) == 16384LL && arg_output_state_stride_slot == arg_output_state.stride(0) && arg_output_state_stride_slot >= AKA_NUM_V_HEADS * 16384LL, ValueError)
      << "output_state inner layout or slot stride is invalid";
  TVM_FFI_CHECK(arg_checkpoint_every_n_tokens == 0, ValueError)
      << "checkpoint interval must be zero for a non-checkpoint variant";
  TVM_FFI_CHECK(arg_checkpoint_state.numel() >= 1 && arg_cu_checkpoints.numel() >= 1, ValueError)
      << "checkpoint dummy buffers must be nonempty";
  CheckSameCudaDevice(arg_K, arg_Q, "K", "Q");
  CheckSameCudaDevice(arg_V, arg_Q, "V", "Q");
  CheckSameCudaDevice(arg_O, arg_Q, "O", "Q");
  CheckSameCudaDevice(arg_gate, arg_Q, "gate", "Q");
  CheckSameCudaDevice(arg_beta, arg_Q, "beta", "Q");
  CheckSameCudaDevice(arg_cu_seqlens, arg_Q, "cu_seqlens", "Q");
  CheckSameCudaDevice(arg_state_indices, arg_Q, "state_indices", "Q");
  CheckSameCudaDevice(arg_initial_state, arg_Q, "initial_state", "Q");
  CheckSameCudaDevice(arg_output_state, arg_Q, "output_state", "Q");
  CheckSameCudaDevice(arg_checkpoint_state, arg_Q, "checkpoint_state", "Q");
  CheckSameCudaDevice(arg_cu_checkpoints, arg_Q, "cu_checkpoints", "Q");
  CheckSameCudaDevice(arg_tensormap_workspace, arg_Q, "tensormap_workspace", "Q");
  CheckCurrentCudaDevice(arg_Q, "Q");
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << "launch grid dimensions must be positive, got (" << grid_x << ", " << grid_y
      << ", " << grid_z << ")";

  cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(dev.device_type, dev.device_id);
  CUtensorMap p_Q = EncodeTma_Q(arg_Q);
  CUtensorMap p_K = EncodeTma_K(arg_K);
  CUtensorMap p_V = EncodeTma_V(arg_V);
  CUtensorMap p_O = EncodeTma_O(arg_O);
  void* p_gate = arg_gate.data_ptr();
  void* p_beta = arg_beta.data_ptr();
  void* p_cu_seqlens = arg_cu_seqlens.data_ptr();
  void* p_state_indices = arg_state_indices.data_ptr();
  void* p_initial_state = arg_initial_state.data_ptr();
  void* p_output_state = arg_output_state.data_ptr();
  void* p_checkpoint_state = arg_checkpoint_state.data_ptr();
  void* p_cu_checkpoints = arg_cu_checkpoints.data_ptr();
  void* p_tensormap_workspace = arg_tensormap_workspace.data_ptr();
  int64_t v_initial_state_stride_slot = (int64_t)arg_initial_state_stride_slot;
  int64_t v_output_state_stride_slot = (int64_t)arg_output_state_stride_slot;
  int32_t v_checkpoint_every_n_tokens = (int32_t)arg_checkpoint_every_n_tokens;
  float v_scale = (float)arg_scale;
  int32_t v_num_seqs = (int32_t)arg_num_seqs;
  int32_t v_num_q_heads = (int32_t)arg_num_q_heads;
  int32_t v_num_v_heads = (int32_t)arg_num_v_heads;
  int32_t v_total_tiles = (int32_t)arg_total_tiles;
  void* kargs[] = {&p_Q, &p_K, &p_V, &p_O, &p_gate, &p_beta, &p_cu_seqlens, &p_state_indices, &p_initial_state, &p_output_state, &p_checkpoint_state, &p_cu_checkpoints, &p_tensormap_workspace, &v_initial_state_stride_slot, &v_output_state_stride_slot, &v_checkpoint_every_n_tokens, &v_scale, &v_num_seqs, &v_num_q_heads, &v_num_v_heads, &v_total_tiles};

  static auto kernel = EmbedCubinModule_flashinfer_blackwell_gdn_prefill_initial_1230278b65::Global()->mod.GetKernel("kernel_flashinfer_blackwell_gdn_prefill_initial");
  static signed char cake_gdn_smem_mode_cache[64] = {0};
  const bool use_oversized_smem = CakeGDNConfigureDynamicSmem(
      kernel, (int)arg_Q.device().device_id, 226048,
      cake_gdn_smem_mode_cache, 64);
  tvm::ffi::dim3 grid((uint32_t)grid_x, (uint32_t)grid_y, (uint32_t)grid_z);
  tvm::ffi::dim3 block(384u, 1u, 1u);

  if (use_oversized_smem) {

  // Extended launch — mirrors CUDAKernel.prepare_launch_cluster attributes.
  tvm::ffi::cuda_api::LaunchConfig config;
  int n = 0;
#if TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API
  CUlaunchAttribute attrs[1];
#if CAKE_GDN_HAS_OVERSIZED_SMEM
  if (use_oversized_smem) {
    attrs[n].id = CU_LAUNCH_ATTRIBUTE_SHARED_MEMORY_MODE;
    attrs[n].value.sharedMemoryMode = CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY;
    ++n;
  }
#endif
  config.gridDimX = grid.x;
  config.gridDimY = grid.y;
  config.gridDimZ = grid.z;
  config.blockDimX = block.x;
  config.blockDimY = block.y;
  config.blockDimZ = block.z;
  config.sharedMemBytes = 226048u;
  config.hStream = stream;
  config.attrs = attrs;
  config.numAttrs = n;
#else
  cudaLaunchAttribute attrs[1];
#if CAKE_GDN_HAS_OVERSIZED_SMEM
  if (use_oversized_smem) {
    attrs[n].id = cudaLaunchAttributeSharedMemoryMode;
    attrs[n].val.sharedMemoryMode = cudaSharedMemoryModeAllowOversizedSharedMemory;
    ++n;
  }
#endif
  config.gridDim = {grid.x, grid.y, grid.z};
  config.blockDim = {block.x, block.y, block.z};
  config.dynamicSmemBytes = 226048u;
  config.stream = stream;
  config.attrs = attrs;
  config.numAttrs = n;
#endif
  TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(kernel.LaunchEx(kargs, config));

  } else {
    TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(kernel.Launch(kargs, grid, block, stream, 226048u));
  }
}

}  // namespace cake_gdn_host_shim_785de87ac034a54f

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_prefill_flashinfer_blackwell_gdn_prefill_initial_f756fe574915, cake_gdn_host_shim_785de87ac034a54f::Run);
// clang-format on
