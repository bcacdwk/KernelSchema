
// clang-format off
// Editable source; provenance is recorded in ../LICENSES/.
// tvm-ffi host launcher for GDN CP-prefill kernel 'kernel_flashinfer_blackwell_gdn_cp_prefill_t_precompute_bf16_v1'.
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

TVM_FFI_EMBED_CUBIN(flashinfer_blackwell_gdn_cp_prefill_t_precompute_bf16_v1_75841827ff);

namespace gdn_cp_host_shim_42d69ce8c4db8d10 {

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

void Run(TensorView arg_K, TensorView arg_beta, TensorView arg_t, TensorView arg_cu_seqlens, int64_t arg_num_k_heads, int64_t arg_num_sab_heads, int64_t arg_total_t_blocks, int64_t arg_num_seqs, int64_t grid_x, int64_t grid_y, int64_t grid_z) {
  CheckCudaTensor(arg_K, "K");
  CheckDtype(arg_K, "K", 4, 16, 1);
  CheckContiguous(arg_K, "K");
  CheckCudaTensor(arg_beta, "beta");
  CheckDtype(arg_beta, "beta", 2, 32, 1);
  CheckContiguous(arg_beta, "beta");
  CheckCudaTensor(arg_t, "t");
  CheckDtype(arg_t, "t", 4, 16, 1);
  CheckContiguous(arg_t, "t");
  CheckCudaTensor(arg_cu_seqlens, "cu_seqlens");
  CheckDtype(arg_cu_seqlens, "cu_seqlens", 0, 64, 1);
  CheckContiguous(arg_cu_seqlens, "cu_seqlens");
  TVM_FFI_CHECK(arg_num_k_heads >= -2147483648LL && arg_num_k_heads <= 2147483647LL, ValueError)
      << "scalar 'num_k_heads' value " << arg_num_k_heads
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_num_sab_heads >= -2147483648LL && arg_num_sab_heads <= 2147483647LL, ValueError)
      << "scalar 'num_sab_heads' value " << arg_num_sab_heads
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_total_t_blocks >= -2147483648LL && arg_total_t_blocks <= 2147483647LL, ValueError)
      << "scalar 'total_t_blocks' value " << arg_total_t_blocks
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_num_seqs >= -2147483648LL && arg_num_seqs <= 2147483647LL, ValueError)
      << "scalar 'num_seqs' value " << arg_num_seqs
      << " is outside i32 range [-2147483648, 2147483647]";
  CheckSameCudaDevice(arg_beta, arg_K, "beta", "K");
  CheckSameCudaDevice(arg_t, arg_K, "t", "K");
  CheckSameCudaDevice(arg_cu_seqlens, arg_K, "cu_seqlens", "K");
  CheckCurrentCudaDevice(arg_K, "K");
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << "launch grid dimensions must be positive, got (" << grid_x << ", " << grid_y
      << ", " << grid_z << ")";

  DLDevice dev = arg_K.device();
  cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(dev.device_type, dev.device_id);
  CUtensorMap p_K = EncodeTma_K(arg_K);
  void* p_beta = arg_beta.data_ptr();
  void* p_t = arg_t.data_ptr();
  void* p_cu_seqlens = arg_cu_seqlens.data_ptr();
  int32_t v_num_k_heads = (int32_t)arg_num_k_heads;
  int32_t v_num_sab_heads = (int32_t)arg_num_sab_heads;
  int32_t v_total_t_blocks = (int32_t)arg_total_t_blocks;
  int32_t v_num_seqs = (int32_t)arg_num_seqs;
  void* kargs[] = {&p_K, &p_beta, &p_t, &p_cu_seqlens, &v_num_k_heads, &v_num_sab_heads, &v_total_t_blocks, &v_num_seqs};

  static auto kernel = TVM_FFI_EMBED_CUBIN_GET_KERNEL(flashinfer_blackwell_gdn_cp_prefill_t_precompute_bf16_v1_75841827ff, "kernel_flashinfer_blackwell_gdn_cp_prefill_t_precompute_bf16_v1");
  tvm::ffi::dim3 grid((uint32_t)grid_x, (uint32_t)grid_y, (uint32_t)grid_z);
  tvm::ffi::dim3 block(128u, 1u, 1u);

  TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(kernel.Launch(kargs, grid, block, stream, 25856u));
}

}  // namespace gdn_cp_host_shim_42d69ce8c4db8d10

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_t_precompute_bf16, gdn_cp_host_shim_42d69ce8c4db8d10::Run);
// clang-format on
