
// clang-format off
// Editable source; provenance is recorded in ../LICENSES/.
// tvm-ffi host launcher for GDN CP-prefill kernel 'kernel_flashinfer_blackwell_gdn_cp_prefill_fixup_utcmma64_v1'.
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

TVM_FFI_EMBED_CUBIN(flashinfer_blackwell_gdn_cp_prefill_fixup_utcmma64_v1_e12079adf3);

namespace gdn_cp_host_shim_f0f851747a592feb {

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

// 5D TMA descriptor for buffer 'local_transfer' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_local_transfer(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 4, ValueError)
      << "TMA source 'local_transfer' must have at least 4 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'local_transfer' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  int64_t d3 = t.size(t.ndim() - 3);
  int64_t d4 = t.size(t.ndim() - 4);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0 && d3 > 0 && d4 > 0, ValueError)
      << "TMA source 'local_transfer' trailing dims must be positive";
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'local_transfer' physical strides must be positive";
  int64_t s3 = t.stride(t.ndim() - 3) * 1;
  TVM_FFI_CHECK(s3 > 0, ValueError)
      << "TMA source 'local_transfer' physical strides must be positive";
  int64_t s4 = t.stride(t.ndim() - 4) * 1;
  TVM_FFI_CHECK(s4 > 0, ValueError)
      << "TMA source 'local_transfer' physical strides must be positive";
  TVM_FFI_CHECK(d1 % 32 == 0, ValueError)
      << "TMA source 'local_transfer' extent " << d1
      << " must divide exactly by " << 32;
  uint64_t global_dim[5] = {(uint64_t)(32), (uint64_t)(d2), (uint64_t)((d1 / 32)), (uint64_t)(d3), (uint64_t)(d4)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0 && global_dim[4] > 0, ValueError)
      << "TMA descriptor for 'local_transfer' resolved a non-positive global dim";
  TVM_FFI_CHECK(32u <= global_dim[0] && 128u <= global_dim[1] && 4u <= global_dim[2] && 1u <= global_dim[3] && 1u <= global_dim[4], ValueError)
      << "TMA box (32, 128, 4, 1, 1) exceeds resolved global dims for 'local_transfer'";
  uint64_t global_strides[4] = {
      (uint64_t)((s2 * 32) / 8),
      (uint64_t)((32 * 32) / 8),
      (uint64_t)((s3 * 32) / 8),
      (uint64_t)((s4 * 32) / 8),
  };
  uint32_t box_dim[5] = {32u, 128u, 4u, 1u, 1u};
  uint32_t elem_strides[5] = {1u, 1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 5, t.data_ptr(), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (5D, 'local_transfer') failed: CUresult=" << (int)r;
  return tm;
}

// 5D TMA descriptor for buffer 'local_state' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_local_state(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 4, ValueError)
      << "TMA source 'local_state' must have at least 4 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'local_state' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  int64_t d3 = t.size(t.ndim() - 3);
  int64_t d4 = t.size(t.ndim() - 4);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0 && d3 > 0 && d4 > 0, ValueError)
      << "TMA source 'local_state' trailing dims must be positive";
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'local_state' physical strides must be positive";
  int64_t s3 = t.stride(t.ndim() - 3) * 1;
  TVM_FFI_CHECK(s3 > 0, ValueError)
      << "TMA source 'local_state' physical strides must be positive";
  int64_t s4 = t.stride(t.ndim() - 4) * 1;
  TVM_FFI_CHECK(s4 > 0, ValueError)
      << "TMA source 'local_state' physical strides must be positive";
  TVM_FFI_CHECK(d1 % 32 == 0, ValueError)
      << "TMA source 'local_state' extent " << d1
      << " must divide exactly by " << 32;
  uint64_t global_dim[5] = {(uint64_t)(32), (uint64_t)(d2), (uint64_t)((d1 / 32)), (uint64_t)(d3), (uint64_t)(d4)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0 && global_dim[4] > 0, ValueError)
      << "TMA descriptor for 'local_state' resolved a non-positive global dim";
  TVM_FFI_CHECK(32u <= global_dim[0] && 64u <= global_dim[1] && 4u <= global_dim[2] && 1u <= global_dim[3] && 1u <= global_dim[4], ValueError)
      << "TMA box (32, 64, 4, 1, 1) exceeds resolved global dims for 'local_state'";
  uint64_t global_strides[4] = {
      (uint64_t)((s2 * 32) / 8),
      (uint64_t)((32 * 32) / 8),
      (uint64_t)((s3 * 32) / 8),
      (uint64_t)((s4 * 32) / 8),
  };
  uint32_t box_dim[5] = {32u, 64u, 4u, 1u, 1u};
  uint32_t elem_strides[5] = {1u, 1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 5, t.data_ptr(), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (5D, 'local_state') failed: CUresult=" << (int)r;
  return tm;
}

void Run(TensorView arg_local_transfer, TensorView arg_local_state, TensorView arg_initial_state, TensorView arg_initial_state_workspace, TensorView arg_fixed_state, TensorView arg_output_state, TensorView arg_cu_seqlens, int64_t arg_chunk_len, int64_t arg_total_cp_chunks, int64_t arg_num_seqs, int64_t arg_num_heads, int64_t grid_x, int64_t grid_y, int64_t grid_z) {
  CheckCudaTensor(arg_local_transfer, "local_transfer");
  CheckDtype(arg_local_transfer, "local_transfer", 2, 32, 1);
  CheckCudaTensor(arg_local_state, "local_state");
  CheckDtype(arg_local_state, "local_state", 2, 32, 1);
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

  DLDevice dev = arg_local_transfer.device();
  cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(dev.device_type, dev.device_id);
  CUtensorMap p_local_transfer = EncodeTma_local_transfer(arg_local_transfer);
  CUtensorMap p_local_state = EncodeTma_local_state(arg_local_state);
  void* p_initial_state = arg_initial_state.data_ptr();
  void* p_initial_state_workspace = arg_initial_state_workspace.data_ptr();
  void* p_fixed_state = arg_fixed_state.data_ptr();
  void* p_output_state = arg_output_state.data_ptr();
  void* p_cu_seqlens = arg_cu_seqlens.data_ptr();
  int32_t v_chunk_len = (int32_t)arg_chunk_len;
  int32_t v_total_cp_chunks = (int32_t)arg_total_cp_chunks;
  int32_t v_num_seqs = (int32_t)arg_num_seqs;
  int32_t v_num_heads = (int32_t)arg_num_heads;
  void* kargs[] = {&p_local_transfer, &p_local_state, &p_initial_state, &p_initial_state_workspace, &p_fixed_state, &p_output_state, &p_cu_seqlens, &v_chunk_len, &v_total_cp_chunks, &v_num_seqs, &v_num_heads};

  static auto kernel = EmbedCubinModule_flashinfer_blackwell_gdn_cp_prefill_fixup_utcmma64_v1_e12079adf3::Global()->mod.GetKernel("kernel_flashinfer_blackwell_gdn_cp_prefill_fixup_utcmma64_v1");
  static signed char gdn_cp_smem_mode_cache[64] = {0};
  const bool use_oversized_smem = GDNCPConfigureDynamicSmem(
      kernel, (int)arg_local_transfer.device().device_id, 164864,
      gdn_cp_smem_mode_cache, 64);
  tvm::ffi::dim3 grid((uint32_t)grid_x, (uint32_t)grid_y, (uint32_t)grid_z);
  tvm::ffi::dim3 block(256u, 1u, 1u);

  TVM_FFI_CHECK(!use_oversized_smem, RuntimeError)
      << "oversized dynamic shared memory requires the extended (cluster) launch path";
  TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(kernel.Launch(kargs, grid, block, stream, 164864u));
}

}  // namespace gdn_cp_host_shim_f0f851747a592feb

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_state_fixup_utcmma64, gdn_cp_host_shim_f0f851747a592feb::Run);
// clang-format on
