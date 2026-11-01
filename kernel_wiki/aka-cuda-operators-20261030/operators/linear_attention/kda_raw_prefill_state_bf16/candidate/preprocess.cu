// Copyright (c) 2026. Pure CUDA AKA integration.
// Sequence values are read on the GPU on every invocation. No host-value cache.
#include "ffi.cuh"
#include <cuda_runtime.h>
__global__ void aka_kda_packed_metadata(const int64_t* cu, int32_t* chunks,
                                      int32_t* chunk_to_seq, int32_t* order,
                                      int sequences, int capacity, int sort_by_length) {
  if (threadIdx.x == 0) {
    chunks[0] = 0;
    for (int s=0; s<sequences; ++s) {
      chunks[s+1] = chunks[s] + int((cu[s+1]-cu[s]+15)/16);
      int rank = 0;
      if (sort_by_length) {
        const int64_t length = cu[s+1]-cu[s];
        for (int other=0; other<sequences; ++other) {
          const int64_t other_length = cu[other+1]-cu[other];
          rank += other_length > length || (other_length == length && other < s);
        }
      } else { rank = s; }
      order[rank] = s;
    }
  }
  __syncthreads();
  for (int c=threadIdx.x; c<capacity; c+=blockDim.x) {
    int s=0;
    while (s<sequences && c>=chunks[s+1]) ++s;
    chunk_to_seq[c] = s;
  }
}
void RunMetadata(TensorView cu, TensorView chunks, TensorView map, TensorView order,
                 int64_t capacity, int64_t cuda_stream, int64_t sort_by_length) {
  ffi::CUDADeviceGuard guard(cu.device().device_id);
  TVM_FFI_ICHECK(cu.dtype() == dl_int64 && chunks.dtype() == dl_int32);
  TVM_FFI_ICHECK(cu.numel() == chunks.numel() && order.numel()+1 == cu.numel());
  TVM_FFI_ICHECK(map.numel() >= capacity);
  aka_kda_packed_metadata<<<1,128,0,reinterpret_cast<cudaStream_t>(cuda_stream)>>>(
      static_cast<int64_t*>(cu.data_ptr()),static_cast<int32_t*>(chunks.data_ptr()),
      static_cast<int32_t*>(map.data_ptr()),static_cast<int32_t*>(order.data_ptr()),
      int(cu.numel()-1),int(capacity),int(sort_by_length));
  TVM_FFI_ICHECK(cudaGetLastError()==cudaSuccess);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(metadata,RunMetadata);

// SPDX-License-Identifier: Apache-2.0
// Local value-preserving transport for the frozen row-strided BF16 beta view.
// The tensor-math kernels consume the same contiguous, 16-byte aligned values.
#include <cuda_runtime.h>
#include <cstdint>
#include "ffi.cuh"

__global__ void aka_kda_pack_beta(const uint16_t* src, uint16_t* dst,
                                int64_t n, int64_t heads,
                                int64_t row_stride, int64_t head_stride) {
  const int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if (i<n) dst[i]=src[(i/heads)*row_stride+(i%heads)*head_stride];
}

void run(TensorView source, TensorView destination, int64_t stream_handle) {
  TVM_FFI_ICHECK(source.device().device_type==kDLCUDA &&
      destination.device().device_type==kDLCUDA &&
      source.device().device_id==destination.device().device_id);
  ffi::CUDADeviceGuard guard(source.device().device_id);
  TVM_FFI_ICHECK(source.ndim()==3 && source.size(0)==1 && destination.ndim()==3);
  TVM_FFI_ICHECK(destination.size(0)==1 && source.size(1)==destination.size(1) &&
      source.size(2)==destination.size(2) && destination.IsContiguous());
  TVM_FFI_ICHECK(encode_dlpack_dtype(source.dtype())==encode_dlpack_dtype(dl_bfloat16) &&
      encode_dlpack_dtype(destination.dtype())==encode_dlpack_dtype(dl_bfloat16));
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(destination.data_ptr())%16==0);
  const int64_t n=source.numel();
  if (n) aka_kda_pack_beta<<<(n+255)/256,256,0,
      reinterpret_cast<cudaStream_t>(uintptr_t(stream_handle))>>>(
      static_cast<const uint16_t*>(source.data_ptr()),static_cast<uint16_t*>(destination.data_ptr()),
      n,source.size(2),source.stride(1),source.stride(2));
  TVM_FFI_ICHECK(cudaGetLastError()==cudaSuccess) << "KDA beta packing launch failed";
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(beta_pack,run);
