
// clang-format off
// Editable source; provenance is recorded in ../LICENSES/.
// tvm-ffi host launcher for GDN CP-prefill kernel 'kernel_flashinfer_blackwell_gdn_cp_prefill_final_bf16_v1'.
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

TVM_FFI_EMBED_CUBIN(flashinfer_blackwell_gdn_cp_prefill_final_bf16_v1_4affd36e88);

namespace gdn_cp_host_shim_21d7c0f0e4ee0c6e {

using tvm::ffi::TensorView;

#include "host_checks.cuh"

// CUDA 13.4 adds an oversized shared-memory mode (function/launch attribute
// SHARED_MEMORY_MODE = 3) that lets a kernel exceed the standard
// MaxSharedMemoryPerBlockOptin ceiling up to device attribute 150
// (OversizedSharedMemoryPerBlock), at the cost of an 8 KiB L1 carveout.
#if (TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API && defined(CUDA_VERSION) && CUDA_VERSION >= 13040) || \
    (!TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API && defined(CUDART_VERSION) && CUDART_VERSION >= 13040)
#define GDN_CP_HAS_OVERSIZED_SMEM 1
#else
#define GDN_CP_HAS_OVERSIZED_SMEM 0
#endif

inline bool GDNCPConfigureDynamicSmem(tvm::ffi::CubinKernel& kernel, int device_id,
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
#if GDN_CP_HAS_OVERSIZED_SMEM
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
  cache[device_id] = 2;
  return true;
#else
  TVM_FFI_THROW(RuntimeError)
      << "dynamic smem " << smem_bytes << " B exceeds the standard opt-in ceiling ("
      << optin_max << " B) on cuda:" << device_id
      << " and this CUDA toolkit predates the 13.4 oversized shared-memory mode";
#endif
}

// 4D TMA descriptor for buffer 'Q' — compiled from the
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
  int64_t s3 = t.stride(t.ndim() - 3) * 1;
  TVM_FFI_CHECK(s3 > 0, ValueError)
      << "TMA source 'Q' physical strides must be positive";
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'Q' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[4] = {(uint64_t)(64), (uint64_t)(d3), (uint64_t)((d1 / 64)), (uint64_t)(d2)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'Q' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 2u <= global_dim[2] && 1u <= global_dim[3], ValueError)
      << "TMA box (64, 64, 2, 1) exceeds resolved global dims for 'Q'";
  uint64_t global_strides[3] = {
      (uint64_t)((s3 * 16) / 8),
      (uint64_t)((64 * 16) / 8),
      (uint64_t)((d1 * 16) / 8),
  };
  uint32_t box_dim[4] = {64u, 64u, 2u, 1u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, t.data_ptr(), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'Q') failed: CUresult=" << (int)r;
  return tm;
}

// 4D TMA descriptor for buffer 'K' — compiled from the
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
  int64_t s3 = t.stride(t.ndim() - 3) * 1;
  TVM_FFI_CHECK(s3 > 0, ValueError)
      << "TMA source 'K' physical strides must be positive";
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'K' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[4] = {(uint64_t)(64), (uint64_t)(d3), (uint64_t)((d1 / 64)), (uint64_t)(d2)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'K' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 2u <= global_dim[2] && 1u <= global_dim[3], ValueError)
      << "TMA box (64, 64, 2, 1) exceeds resolved global dims for 'K'";
  uint64_t global_strides[3] = {
      (uint64_t)((s3 * 16) / 8),
      (uint64_t)((64 * 16) / 8),
      (uint64_t)((d1 * 16) / 8),
  };
  uint32_t box_dim[4] = {64u, 64u, 2u, 1u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, t.data_ptr(), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'K') failed: CUresult=" << (int)r;
  return tm;
}

// 4D TMA descriptor for buffer 'V' — compiled from the
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
  int64_t s3 = t.stride(t.ndim() - 3) * 1;
  TVM_FFI_CHECK(s3 > 0, ValueError)
      << "TMA source 'V' physical strides must be positive";
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'V' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[4] = {(uint64_t)(64), (uint64_t)(d3), (uint64_t)((d1 / 64)), (uint64_t)(d2)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'V' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 2u <= global_dim[2] && 1u <= global_dim[3], ValueError)
      << "TMA box (64, 64, 2, 1) exceeds resolved global dims for 'V'";
  uint64_t global_strides[3] = {
      (uint64_t)((s3 * 16) / 8),
      (uint64_t)((64 * 16) / 8),
      (uint64_t)((d1 * 16) / 8),
  };
  uint32_t box_dim[4] = {64u, 64u, 2u, 1u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, t.data_ptr(), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'V') failed: CUresult=" << (int)r;
  return tm;
}

// 4D TMA descriptor for buffer 'T' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_T(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 4, ValueError)
      << "TMA source 'T' must have at least 4 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'T' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  int64_t d3 = t.size(t.ndim() - 3);
  int64_t d4 = t.size(t.ndim() - 4);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0 && d3 > 0 && d4 > 0, ValueError)
      << "TMA source 'T' trailing dims must be positive";
  uint64_t global_dim[4] = {(uint64_t)(d1), (uint64_t)(d2), (uint64_t)(d3), (uint64_t)(d4)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'T' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 64u <= global_dim[1] && 1u <= global_dim[2] && 1u <= global_dim[3], ValueError)
      << "TMA box (64, 64, 1, 1) exceeds resolved global dims for 'T'";
  uint64_t global_strides[3] = {
      (uint64_t)((d1 * 16) / 8),
      (uint64_t)(((d2 * d1) * 16) / 8),
      (uint64_t)((((d3 * d2) * d1) * 16) / 8),
  };
  uint32_t box_dim[4] = {64u, 64u, 1u, 1u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, t.data_ptr(), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'T') failed: CUresult=" << (int)r;
  return tm;
}

// 4D TMA descriptor for buffer 'O' — compiled from the
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
  int64_t s3 = t.stride(t.ndim() - 3) * 1;
  TVM_FFI_CHECK(s3 > 0, ValueError)
      << "TMA source 'O' physical strides must be positive";
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'O' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[4] = {(uint64_t)(64), (uint64_t)(d3), (uint64_t)((d1 / 64)), (uint64_t)(d2)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'O' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 2u <= global_dim[2] && 1u <= global_dim[3], ValueError)
      << "TMA box (64, 64, 2, 1) exceeds resolved global dims for 'O'";
  uint64_t global_strides[3] = {
      (uint64_t)((s3 * 16) / 8),
      (uint64_t)((64 * 16) / 8),
      (uint64_t)((d1 * 16) / 8),
  };
  uint32_t box_dim[4] = {64u, 64u, 2u, 1u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, t.data_ptr(), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'O') failed: CUresult=" << (int)r;
  return tm;
}

void Run(TensorView arg_Q, TensorView arg_K, TensorView arg_V, TensorView arg_T, TensorView arg_O, TensorView arg_alpha, TensorView arg_cu_seqlens, TensorView arg_fixed_state, TensorView arg_initial_state_workspace, TensorView arg_tensormap_workspace, int64_t arg_cp_chunk_len, int64_t arg_source_cp_chunk_len, int64_t arg_num_q_heads, int64_t arg_num_k_heads, int64_t arg_num_v_heads, int64_t arg_num_sab_heads, double arg_scale, int64_t grid_x, int64_t grid_y, int64_t grid_z) {
  CheckCudaTensor(arg_Q, "Q");
  CheckDtype(arg_Q, "Q", 4, 16, 1);
  CheckContiguous(arg_Q, "Q");
  CheckCudaTensor(arg_K, "K");
  CheckDtype(arg_K, "K", 4, 16, 1);
  CheckContiguous(arg_K, "K");
  CheckCudaTensor(arg_V, "V");
  CheckDtype(arg_V, "V", 4, 16, 1);
  CheckContiguous(arg_V, "V");
  CheckCudaTensor(arg_T, "T");
  CheckDtype(arg_T, "T", 4, 16, 1);
  CheckContiguous(arg_T, "T");
  CheckCudaTensor(arg_O, "O");
  CheckDtype(arg_O, "O", 4, 16, 1);
  CheckContiguous(arg_O, "O");
  CheckCudaTensor(arg_alpha, "alpha");
  CheckDtype(arg_alpha, "alpha", 2, 32, 1);
  CheckContiguous(arg_alpha, "alpha");
  CheckCudaTensor(arg_cu_seqlens, "cu_seqlens");
  CheckDtype(arg_cu_seqlens, "cu_seqlens", 0, 64, 1);
  CheckContiguous(arg_cu_seqlens, "cu_seqlens");
  CheckCudaTensor(arg_fixed_state, "fixed_state");
  CheckDtype(arg_fixed_state, "fixed_state", 2, 32, 1);
  CheckContiguous(arg_fixed_state, "fixed_state");
  CheckCudaTensor(arg_initial_state_workspace, "initial_state_workspace");
  CheckDtype(arg_initial_state_workspace, "initial_state_workspace", 2, 32, 1);
  CheckContiguous(arg_initial_state_workspace, "initial_state_workspace");
  CheckCudaTensor(arg_tensormap_workspace, "tensormap_workspace");
  CheckDtype(arg_tensormap_workspace, "tensormap_workspace", 1, 8, 1);
  CheckContiguous(arg_tensormap_workspace, "tensormap_workspace");
  TVM_FFI_CHECK(arg_cp_chunk_len >= -2147483648LL && arg_cp_chunk_len <= 2147483647LL, ValueError)
      << "scalar 'cp_chunk_len' value " << arg_cp_chunk_len
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_source_cp_chunk_len >= -2147483648LL && arg_source_cp_chunk_len <= 2147483647LL, ValueError)
      << "scalar 'source_cp_chunk_len' value " << arg_source_cp_chunk_len
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_num_q_heads >= -2147483648LL && arg_num_q_heads <= 2147483647LL, ValueError)
      << "scalar 'num_q_heads' value " << arg_num_q_heads
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_num_k_heads >= -2147483648LL && arg_num_k_heads <= 2147483647LL, ValueError)
      << "scalar 'num_k_heads' value " << arg_num_k_heads
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_num_v_heads >= -2147483648LL && arg_num_v_heads <= 2147483647LL, ValueError)
      << "scalar 'num_v_heads' value " << arg_num_v_heads
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_num_sab_heads >= -2147483648LL && arg_num_sab_heads <= 2147483647LL, ValueError)
      << "scalar 'num_sab_heads' value " << arg_num_sab_heads
      << " is outside i32 range [-2147483648, 2147483647]";
  CheckSameCudaDevice(arg_K, arg_Q, "K", "Q");
  CheckSameCudaDevice(arg_V, arg_Q, "V", "Q");
  CheckSameCudaDevice(arg_T, arg_Q, "T", "Q");
  CheckSameCudaDevice(arg_O, arg_Q, "O", "Q");
  CheckSameCudaDevice(arg_alpha, arg_Q, "alpha", "Q");
  CheckSameCudaDevice(arg_cu_seqlens, arg_Q, "cu_seqlens", "Q");
  CheckSameCudaDevice(arg_fixed_state, arg_Q, "fixed_state", "Q");
  CheckSameCudaDevice(arg_initial_state_workspace, arg_Q, "initial_state_workspace", "Q");
  CheckSameCudaDevice(arg_tensormap_workspace, arg_Q, "tensormap_workspace", "Q");
  CheckCurrentCudaDevice(arg_Q, "Q");
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << "launch grid dimensions must be positive, got (" << grid_x << ", " << grid_y
      << ", " << grid_z << ")";

  DLDevice dev = arg_Q.device();
  cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(dev.device_type, dev.device_id);
  CUtensorMap p_Q = EncodeTma_Q(arg_Q);
  CUtensorMap p_K = EncodeTma_K(arg_K);
  CUtensorMap p_V = EncodeTma_V(arg_V);
  CUtensorMap p_T = EncodeTma_T(arg_T);
  CUtensorMap p_O = EncodeTma_O(arg_O);
  void* p_alpha = arg_alpha.data_ptr();
  void* p_cu_seqlens = arg_cu_seqlens.data_ptr();
  void* p_fixed_state = arg_fixed_state.data_ptr();
  void* p_initial_state_workspace = arg_initial_state_workspace.data_ptr();
  void* p_tensormap_workspace = arg_tensormap_workspace.data_ptr();
  int32_t v_cp_chunk_len = (int32_t)arg_cp_chunk_len;
  int32_t v_source_cp_chunk_len = (int32_t)arg_source_cp_chunk_len;
  int32_t v_num_q_heads = (int32_t)arg_num_q_heads;
  int32_t v_num_k_heads = (int32_t)arg_num_k_heads;
  int32_t v_num_v_heads = (int32_t)arg_num_v_heads;
  int32_t v_num_sab_heads = (int32_t)arg_num_sab_heads;
  float v_scale = (float)arg_scale;
  void* kargs[] = {&p_Q, &p_K, &p_V, &p_T, &p_O, &p_alpha, &p_cu_seqlens, &p_fixed_state, &p_initial_state_workspace, &p_tensormap_workspace, &v_cp_chunk_len, &v_source_cp_chunk_len, &v_num_q_heads, &v_num_k_heads, &v_num_v_heads, &v_num_sab_heads, &v_scale};

  static auto kernel = EmbedCubinModule_flashinfer_blackwell_gdn_cp_prefill_final_bf16_v1_4affd36e88::Global()->mod.GetKernel("kernel_flashinfer_blackwell_gdn_cp_prefill_final_bf16_v1");
  static signed char gdn_cp_smem_mode_cache[64] = {0};
  const bool use_oversized_smem = GDNCPConfigureDynamicSmem(
      kernel, (int)arg_Q.device().device_id, 224768,
      gdn_cp_smem_mode_cache, 64);
  tvm::ffi::dim3 grid((uint32_t)grid_x, (uint32_t)grid_y, (uint32_t)grid_z);
  tvm::ffi::dim3 block(384u, 1u, 1u);

  TVM_FFI_CHECK(!use_oversized_smem, RuntimeError)
      << "oversized dynamic shared memory requires the extended (cluster) launch path";
  TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(kernel.Launch(kargs, grid, block, stream, 224768u));
}

}  // namespace gdn_cp_host_shim_21d7c0f0e4ee0c6e

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_cp_prefill_bf16, gdn_cp_host_shim_21d7c0f0e4ee0c6e::Run);
// clang-format on
