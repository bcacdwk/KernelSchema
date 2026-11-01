/*
 * Copyright (c) 2023 by FlashInfer team.
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
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/device_guard.h>
#include <tvm/ffi/function.h>

#include "dlpack/dlpack.h"

using tvm::ffi::Tensor;
using tvm::ffi::TensorView;
namespace ffi = tvm::ffi;

constexpr DLDataType dl_uint32 = DLDataType{kDLUInt, 32, 1};

constexpr DLDataType dl_int32 = DLDataType{kDLInt, 32, 1};

constexpr DLDataType dl_float32 = DLDataType{kDLFloat, 32, 1};

constexpr DLDataType dl_float8_e4m3fn = DLDataType{kDLFloat8_e4m3fn, 8, 1};

constexpr DLDataType dl_bfloat16 = DLDataType{kDLBfloat, 16, 1};

inline int64_t get_element_size(ffi::Tensor x) { return (x.dtype().bits * x.dtype().lanes) / 8; }

inline int64_t get_element_size(ffi::TensorView x) {
  return (x.dtype().bits * x.dtype().lanes) / 8;
}

namespace flashinfer {
namespace cake_fmha {
namespace {
void CheckSameDevice(TensorView query, TensorView tensor, const char* name) {
  TVM_FFI_ICHECK_EQ(query.device().device_type, tensor.device().device_type)
      << name << " must be on the query device";
  TVM_FFI_ICHECK_EQ(query.device().device_id, tensor.device().device_id)
      << name << " must be on the query device";
}

}  // namespace
}  // namespace cake_fmha
}  // namespace flashinfer
