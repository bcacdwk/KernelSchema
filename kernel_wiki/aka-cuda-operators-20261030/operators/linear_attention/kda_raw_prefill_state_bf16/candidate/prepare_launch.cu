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

static_assert(128 > 0);
static_assert(44032 > 0);
static_assert(0 == 0 || 0 == 1);
static_assert(0 >= 0 &&
              0 <= 3);

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
#include "prepare.cuh"
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
static_assert(THREADS == 128,
              "generated body and selector thread counts disagree");
#endif
static_assert(SMEM_TOTAL == 44032,
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

inline cudaStream_t CheckedStream(int64_t cuda_stream);

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
  if (configured_device_id != device_id) {
    CheckDynamicSmemCapacity(device_id, 44032);
    CheckCuda(
        cudaFuncSetAttribute(kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             44032),
        "cudaFuncSetAttribute(generated FlashKDA kernel)");
    configured_device_id = device_id;
  }
}

static inline void LaunchConfiguredGeneratedKernel(const void* kernel, dim3 grid,
                                                   cudaStream_t stream,
                                                   void** args,
                                                   const char* launch_name) {
  TVM_FFI_ICHECK(grid.x > 0 && grid.y > 0 && grid.z > 0)
      << "generated kernel grid dimensions must be positive";
  CheckCuda(cudaLaunchKernel(kernel, grid, dim3(128, 1, 1), args,
                             44032, stream),
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

static_assert(0 == 0,
              "BT16 prepare must not acquire serving-state pointers");

struct Bt16PrepareArgs {
  void *q{}, *k{}, *raw_gate{};
  CUtensorMap q_tma{}, k_tma{}, raw_gate_tma{};
  void *beta_logits{}, *a_log{}, *dt_bias{}, *cu_seqlens{};
  CUtensorMap beta_logits_tma{};
  void *ws_qd{}, *ws_kd{};
  CUtensorMap ws_qd_tma{};
  void *ws_w{}, *ws_qk_t{}, *ws_diag{};
  CUtensorMap ws_kd_tma{}, ws_w_tma{};
  int32_t total_chunks{}, num_heads{}, num_sequences{};
  float gate_lower_bound{};
};

inline void LaunchBt16Prepare(Bt16PrepareArgs args, dim3 grid,
                              cudaStream_t stream) {
  void* kernel_args[] = {
      &args.q, &args.q_tma, &args.k, &args.k_tma, &args.raw_gate,
      &args.raw_gate_tma, &args.beta_logits, &args.beta_logits_tma,
      &args.a_log, &args.dt_bias, &args.cu_seqlens, &args.ws_qd, &args.ws_qd_tma, &args.ws_kd,
      &args.ws_kd_tma, &args.ws_w, &args.ws_w_tma, &args.ws_qk_t,
      &args.ws_diag, &args.total_chunks, &args.num_heads,
      &args.gate_lower_bound, &args.num_sequences};
  CheckArgumentCount<23>(kernel_args);
  ConfigureAndLaunch(reinterpret_cast<const void*>(kernel_flashkda_bf16_bt16_prepare), grid,
                     stream, kernel_args, "generated BT16 prepare launch");
}

inline void RunBt16Prepare(
    TensorView q,TensorView k,TensorView raw_gate,TensorView beta_logits,
    TensorView a_log,TensorView dt_bias,
    TensorView cu_seqlens,
    TensorView ws_qd,TensorView ws_kd,TensorView ws_w,TensorView ws_qk_t,
    TensorView ws_diag,int64_t total_chunks,int64_t num_heads,double gate_lower_bound,
    int64_t grid_x,int64_t grid_y,int64_t grid_z,int64_t cuda_stream) {
  TVM_FFI_ICHECK(q.device().device_type==kDLCUDA);const int32_t dev=q.device().device_id;
  ffi::CUDADeviceGuard guard(dev);CheckFlashKDATarget(dev);
  TVM_FFI_ICHECK(grid_y==1&&grid_z==1);
  flash_kda::CheckBt16PrepareInputs(q,k,raw_gate,beta_logits,a_log,dt_bias,
      cu_seqlens,ws_qd,ws_kd,ws_w,ws_qk_t,ws_diag,
      grid_x,total_chunks,num_heads,
      gate_lower_bound);
  const cudaStream_t stream=CheckedStream(cuda_stream);
  const flash_kda::Bt16PrepareTmaMaps tma=
      flash_kda::EncodeBt16PrepareTma<true>(q,k,raw_gate,beta_logits,ws_qd,
          ws_kd,ws_w);
  Bt16PrepareArgs a{};
  a.q=q.data_ptr();a.q_tma=tma.q;a.k=k.data_ptr();a.k_tma=tma.k;
  a.raw_gate=raw_gate.data_ptr();a.raw_gate_tma=tma.raw_gate;
  a.beta_logits=beta_logits.data_ptr();a.beta_logits_tma=tma.beta_logits;
  a.ws_qd=ws_qd.data_ptr();a.ws_qd_tma=tma.ws_qd;
  a.ws_kd=ws_kd.data_ptr();a.ws_kd_tma=tma.ws_kd;
  a.ws_w=ws_w.data_ptr();a.ws_w_tma=tma.ws_w;
  a.a_log=a_log.data_ptr();a.dt_bias=dt_bias.data_ptr();a.cu_seqlens=cu_seqlens.data_ptr();
  a.ws_qk_t=ws_qk_t.data_ptr();
  a.ws_diag=ws_diag.data_ptr();a.total_chunks=CheckedInt32(total_chunks,"total_chunks");
  a.num_heads=CheckedInt32(num_heads,"num_heads");
  a.num_sequences=CheckedInt32(cu_seqlens.numel()-1,"num_sequences");a.gate_lower_bound=static_cast<float>(gate_lower_bound);
  LaunchBt16Prepare(a,CheckedGrid(grid_x,grid_y,grid_z),stream);
}

}  // namespace flashinfer::flash_kda_generated

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run,flashinfer::flash_kda_generated::RunBt16Prepare);
