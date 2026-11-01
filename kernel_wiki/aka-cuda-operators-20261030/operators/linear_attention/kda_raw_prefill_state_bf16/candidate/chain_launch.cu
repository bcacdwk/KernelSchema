/* Auto-generated public binding selector. */
static_assert(FLASHINFER_FLASH_KDA_TARGET_MINOR == 3,
              "binding compiled for the wrong exact target");

/* Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0. */
#pragma once

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

#include <cstddef>
#include <cstdint>

#include "host_helpers.cuh"
#include "descriptors.cuh"

static_assert(512 > 0);
static_assert(138240 > 0);
static_assert(0 == 0 || 0 == 1);
static_assert(1 >= 0 &&
              1 <= 3);

// The generated source is standalone and owns private fixed-width aliases.
// Isolate them from CUDA and TVM-FFI declarations in this translation unit.
#define int8_t flashkda_generated_private_int8_t
#define uint8_t flashkda_generated_private_uint8_t
#define uint16_t flashkda_generated_private_uint16_t
#define uint32_t flashkda_generated_private_uint32_t
#define uint64_t flashkda_generated_private_uint64_t
#define int32_t flashkda_generated_private_int32_t
#define int16_t flashkda_generated_private_int16_t
#define FlashKDATensorMap flashkda_generated_private_TensorMap
#define FlashKDATensorMapPack flashkda_generated_private_TensorMapPack
#define CUtensorMap flashkda_generated_private_CUtensorMap
#include "chain.cuh"
#undef CUtensorMap
#undef FlashKDATensorMapPack
#undef FlashKDATensorMap
#undef int8_t
#undef uint8_t
#undef uint16_t
#undef uint32_t
#undef uint64_t
#undef int32_t
#undef int16_t

#ifdef THREADS
static_assert(THREADS == 512,
              "generated body and selector thread counts disagree");
#endif
static_assert(SMEM_TOTAL == 138240,
              "generated body and selector shared-memory sizes disagree");

namespace flashinfer {
namespace flash_kda_generated {

using flash_kda::CheckCuda;
using flash_kda::CheckCudaTensor;
using flash_kda::CheckCudaTensorDevice;
using flash_kda::CheckDtype;
using flash_kda::CheckDynamicSmemCapacity;
using flash_kda::CheckFlashKDATarget;
using flash_kda::ResolveAndCheckServingStatePoolForDtype;

struct StatePointerSlots {
  void* initial_state;
  void* final_state;
  void* initial_state_f32;
  void* final_state_f32;
  DLDataType dtype;
  int64_t pool_slots;
};

inline DLDataType GeneratedStateDtype() {
  return dl_bfloat16;
}

inline dim3 CheckedGrid(int64_t grid_x, int64_t grid_y, int64_t grid_z) {
  for (const auto& named : {std::pair<int64_t, const char*>{grid_x, "grid_x"},
                            std::pair<int64_t, const char*>{grid_y, "grid_y"},
                            std::pair<int64_t, const char*>{grid_z, "grid_z"}}) {
    TVM_FFI_ICHECK(named.first > 0 &&
                   named.first <= std::numeric_limits<uint32_t>::max())
        << named.second << " must be in the positive uint32 range";
  }
  return dim3(static_cast<uint32_t>(grid_x), static_cast<uint32_t>(grid_y),
              static_cast<uint32_t>(grid_z));
}

inline int32_t CheckedInt32(int64_t value, const char* name) {
  TVM_FFI_ICHECK(value >= std::numeric_limits<int32_t>::min() &&
                 value <= std::numeric_limits<int32_t>::max())
      << name << " must fit int32";
  return static_cast<int32_t>(value);
}

inline void* CheckedBufferPointer(const TensorView& tensor, const char* name,
                                  int32_t device_id, DLDataType dtype,
                                  bool allow_empty = false) {
  CheckCudaTensor(tensor, name, device_id);
  CheckDtype(tensor, name, dtype);
  TVM_FFI_ICHECK(allow_empty || tensor.numel() > 0)
      << name << " must not be empty";
  return tensor.numel() == 0 ? nullptr : tensor.data_ptr();
}

inline cudaStream_t CheckedStream(int64_t cuda_stream);

inline StatePointerSlots ResolveStatePointerSlots(
    const TensorView& state_indices, const TensorView& initial_state,
    const TensorView& final_state, int32_t device_id, int64_t num_seqs,
    int64_t num_heads, int64_t state_slot_stride, int64_t use_state_indices,
    int64_t use_initial_state, int64_t store_final_state) {
#if 1 != 1 && \
    1 != 2
  TVM_FFI_ICHECK(false)
      << "this generated module does not expose serving-state pointer slots";
  return {nullptr, nullptr, nullptr, nullptr, dl_bfloat16, 0};
#else
  const DLDataType dtype = GeneratedStateDtype();
  CheckCudaTensorDevice(initial_state, "initial_state", device_id);
  CheckCudaTensorDevice(final_state, "final_state", device_id);
  CheckDtype(initial_state, "initial_state", dtype);
  CheckDtype(final_state, "final_state", dtype);
  const int64_t pool_slots = ResolveAndCheckServingStatePoolForDtype(
      state_indices, initial_state, final_state, device_id, num_seqs, num_heads,
      state_slot_stride, use_state_indices, use_initial_state, store_final_state, dtype);
  StatePointerSlots slots{nullptr, nullptr, nullptr, nullptr, dtype, pool_slots};
  slots.initial_state = initial_state.data_ptr();
  slots.final_state = final_state.data_ptr();
  return slots;
#endif
}

template <size_t Expected, size_t Actual>
constexpr void CheckArgumentCount(void* (&)[Actual]) {
  static_assert(Expected == Actual, "generated kernel ABI argument count changed");
}

static inline void ConfigureGeneratedKernelForDevice(const void* kernel,
                                                     int32_t device_id) {
  // The CUDA current device is host-thread local.  Configure this generated
  // module once for each device observed by the calling thread instead of
  // inserting repeated attribute/device queries between dependent launches.
  static thread_local int32_t configured_device_id = -1;
  static thread_local const void* configured_kernel = nullptr;
  if (configured_device_id != device_id || configured_kernel != kernel) {
    CheckDynamicSmemCapacity(device_id, 138240);
    CheckCuda(
        cudaFuncSetAttribute(kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             138240),
        "cudaFuncSetAttribute(generated FlashKDA kernel)");
    configured_device_id = device_id;
    configured_kernel = kernel;
  }
}

static inline void LaunchConfiguredGeneratedKernel(const void* kernel, dim3 grid,
                                                   cudaStream_t stream,
                                                   void** args,
                                                   const char* launch_name) {
  TVM_FFI_ICHECK(grid.x > 0 && grid.y > 0 && grid.z > 0)
      << "generated kernel grid dimensions must be positive";
  CheckCuda(cudaLaunchKernel(kernel, grid, dim3(512, 1, 1), args,
                             138240, stream),
            launch_name);
}

static inline void ConfigureAndLaunch(const void* kernel, dim3 grid,
                                      cudaStream_t stream, void** args,
                                      const char* launch_name) {
  int32_t device_id = 0;
  CheckCuda(cudaGetDevice(&device_id), "cudaGetDevice");
  ConfigureGeneratedKernelForDevice(kernel, device_id);
  LaunchConfiguredGeneratedKernel(kernel, grid, stream, args, launch_name);
}

inline cudaStream_t CheckedStream(int64_t cuda_stream) {
  TVM_FFI_ICHECK(cuda_stream >= 0) << "cuda_stream must be a non-negative stream handle";
  return reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
}

}  // namespace flash_kda_generated
}  // namespace flashinfer

namespace flashinfer::flash_kda_generated {

struct Bt16ChainArgs {
  void *ws_qd{}, *ws_kd{}, *ws_w{};
  CUtensorMap ws_qd_tma{}, ws_kd_tma{}, ws_w_tma{};
  void *ws_qk{}, *ws_diag{}, *v{};
  CUtensorMap ws_qk_tma{}, ws_diag_tma{}, v_tma{};
  void *cu_seqlens{}, *initial_state{}, *out{};
  void *final_state{};
  CUtensorMap out_tma{};
  int32_t num_heads{}, use_initial_state{}, store_final_state{};
  float scale{};
  uint64_t state_indices_addr{};
  int64_t state_slot_stride{};
  int32_t use_state_indices{};
  void *initial_state_f32{}, *final_state_f32{}, *state_checkpoints{}, *checkpoint_cu_starts{};
  int32_t checkpoint_stride_chunks{};
};

inline void LaunchBt16Chain(Bt16ChainArgs args, const StatePointerSlots& state,
                            dim3 grid, cudaStream_t stream) {
  args.initial_state = state.initial_state;
  args.final_state = state.final_state;
  args.initial_state_f32 = state.initial_state_f32;
  args.final_state_f32 = state.final_state_f32;
  void* kernel_args[] = {
      &args.ws_qd, &args.ws_qd_tma, &args.ws_kd, &args.ws_kd_tma,
      &args.ws_w, &args.ws_w_tma, &args.ws_qk, &args.ws_qk_tma,
      &args.ws_diag, &args.ws_diag_tma, &args.v, &args.v_tma,
      &args.cu_seqlens, &args.initial_state,
      &args.out, &args.out_tma, &args.final_state, &args.num_heads,
      &args.use_initial_state, &args.store_final_state, &args.scale,
      &args.state_indices_addr, &args.state_slot_stride, &args.use_state_indices,
      &args.initial_state_f32, &args.final_state_f32, &args.state_checkpoints,
      &args.checkpoint_cu_starts, &args.checkpoint_stride_chunks};
  CheckArgumentCount<29>(kernel_args);
  const void* kernel = args.checkpoint_stride_chunks > 0
      ? reinterpret_cast<const void*>(kernel_flashkda_bf16_bt16_chain_m64<true>)
      : reinterpret_cast<const void*>(kernel_flashkda_bf16_bt16_chain_m64<false>);
  ConfigureAndLaunch(kernel, grid,
                     stream, kernel_args, "generated BT16 chain launch");
}

inline void RunBt16Chain(
    TensorView ws_qd,TensorView ws_kd,TensorView ws_w,TensorView ws_qk,
    TensorView ws_diag,TensorView v,
    TensorView cu_seqlens,
    TensorView state_indices,TensorView initial_state,TensorView out,TensorView final_state,
    TensorView state_checkpoints,TensorView checkpoint_cu_starts,int64_t checkpoint_stride_chunks,
    int64_t num_heads,int64_t state_slot_stride,
    int64_t use_state_indices,int64_t use_initial_state,int64_t store_final_state,
    double scale,int64_t grid_x,int64_t grid_y,int64_t grid_z,int64_t cuda_stream) {
  TVM_FFI_ICHECK(v.device().device_type==kDLCUDA);const int32_t dev=v.device().device_id;
  ffi::CUDADeviceGuard guard(dev);CheckFlashKDATarget(dev);
  TVM_FFI_ICHECK(std::isfinite(scale)&&std::isfinite(static_cast<float>(scale)));
  for(const auto& x:{std::pair<TensorView*,const char*>{&ws_qd,"ws_qd"},{&ws_kd,"ws_kd"},
      {&ws_w,"ws_w"},{&ws_qk,"ws_qk"},{&v,"v"},{&out,"out"}})
    CheckedBufferPointer(*x.first,x.second,dev,dl_bfloat16);
  CheckedBufferPointer(ws_diag,"ws_diag",dev,dl_float32);
  CheckedBufferPointer(cu_seqlens,"cu_seqlens",dev,dl_int64);
  const int64_t n=cu_seqlens.numel()-1;
  StatePointerSlots state=ResolveStatePointerSlots(state_indices,initial_state,final_state,dev,n,
      num_heads,state_slot_stride,use_state_indices,use_initial_state,store_final_state);
  TVM_FFI_ICHECK(grid_y==1&&grid_z==1&&grid_x==2*n*num_heads)
      << "BT16 chain grid_x must equal 2 * num_sequences * num_heads";
  const cudaStream_t stream=CheckedStream(cuda_stream);
  const flash_kda::Bt16ChainTmaMaps tma=flash_kda::EncodeBt16ChainTma(
      ws_qd,ws_kd,ws_w,ws_qk,ws_diag,v,out);
  Bt16ChainArgs a{};
  a.ws_qd=ws_qd.data_ptr();a.ws_qd_tma=tma.ws_qd;
  a.ws_kd=ws_kd.data_ptr();a.ws_kd_tma=tma.ws_kd;
  a.ws_w=ws_w.data_ptr();a.ws_w_tma=tma.ws_w;
  a.ws_qk=ws_qk.data_ptr();a.ws_qk_tma=tma.ws_qk;
  a.ws_diag=ws_diag.data_ptr();a.ws_diag_tma=tma.ws_diag;
  a.v=v.data_ptr();a.v_tma=tma.v;
  a.cu_seqlens=cu_seqlens.data_ptr();
  a.out=out.data_ptr();a.out_tma=tma.out;
  a.num_heads=CheckedInt32(num_heads,"num_heads");a.use_initial_state=CheckedInt32(use_initial_state,"use_initial_state");
  a.store_final_state=CheckedInt32(store_final_state,"store_final_state");a.scale=static_cast<float>(scale);
  a.state_indices_addr=reinterpret_cast<uintptr_t>(state_indices.data_ptr());a.state_slot_stride=state_slot_stride;
  a.use_state_indices=CheckedInt32(use_state_indices,"use_state_indices");
  TVM_FFI_ICHECK(checkpoint_stride_chunks >= 0);
  if (checkpoint_stride_chunks > 0) {
    CheckedBufferPointer(state_checkpoints,"state_checkpoints",dev,dl_bfloat16);
    CheckedBufferPointer(checkpoint_cu_starts,"checkpoint_cu_starts",dev,dl_int64);
    TVM_FFI_ICHECK(checkpoint_cu_starts.numel() >= n);
  }
  a.state_checkpoints=state_checkpoints.data_ptr();a.checkpoint_cu_starts=checkpoint_cu_starts.data_ptr();
  a.checkpoint_stride_chunks=CheckedInt32(checkpoint_stride_chunks,"checkpoint_stride_chunks");
  LaunchBt16Chain(a,state,CheckedGrid(grid_x,grid_y,grid_z),stream);
}

}  // namespace flashinfer::flash_kda_generated

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run,flashinfer::flash_kda_generated::RunBt16Chain);
