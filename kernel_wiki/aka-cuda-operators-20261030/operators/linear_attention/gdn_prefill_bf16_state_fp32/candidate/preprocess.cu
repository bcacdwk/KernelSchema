// SPDX-License-Identifier: Apache-2.0
// GDN wrapper preprocessing expressed locally in CUDA. Mathematical kernels
// (CP and non-CP) are unchanged; these operations previously used Torch ops.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include "ffi.cuh"

namespace aka_gdn {
__device__ __forceinline__ float read_scalar(const void* p, int64_t i, bool bf16) {
  return bf16 ? __bfloat162float(static_cast<const __nv_bfloat16*>(p)[i])
              : static_cast<const float*>(p)[i];
}

__global__ void prepare_scalars(const void* g, const void* beta, const void* state,
                               float* alpha, float* beta_fp32, float* state_fp32,
                               int64_t n_gate, int64_t n_state,
                               bool g_bf16, bool beta_bf16, bool state_bf16,
                               const void* cu_in, void* cu_out, int64_t n_cu,
                               bool cu_in64, bool cu_out64, bool make_cu,
                               int64_t total_tokens) {
  const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
  if (i < n_gate) {
    // Compile without --use_fast_math: preserve the wrapper's float exp.
    alpha[i] = expf(read_scalar(g, i, g_bf16));
    if (beta_bf16) beta_fp32[i] = read_scalar(beta, i, true);
  }
  if (state_bf16 && i < n_state) state_fp32[i] = read_scalar(state, i, true);
  if (i < n_cu) {
    const int64_t value = make_cu ? (i == 0 ? 0 : total_tokens)
        : (cu_in64 ? static_cast<const int64_t*>(cu_in)[i]
                   : static_cast<const int32_t*>(cu_in)[i]);
    if (cu_out64) static_cast<int64_t*>(cu_out)[i] = value;
    else static_cast<int32_t*>(cu_out)[i] = static_cast<int32_t>(value);
  }
}

// Frozen GDN operands and the writable core prefix are contiguous.
// Copies and integer metadata casts are explicit CUDA, outside Torch dispatch.
template<typename S, typename D>
__device__ __forceinline__ void copy_elements(const S* source, D* destination, int64_t count) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if (i<count) destination[i]=static_cast<D>(source[i]);
}
// Integer metadata conversions need arithmetic; identical types use stream DMA.
__global__ void copy_i32_i64(const int32_t* source, int64_t* destination, int64_t count) {
  copy_elements(source,destination,count);
}
__global__ void copy_i64_i32(const int64_t* source, int32_t* destination, int64_t count) {
  copy_elements(source,destination,count);
}
void copy_tensor(TensorView source, TensorView destination, int64_t stream_handle) {
  TVM_FFI_ICHECK(source.device().device_type==kDLCUDA && destination.device().device_type==kDLCUDA &&
                destination.device().device_id==source.device().device_id);
  TVM_FFI_ICHECK(destination.numel()>=source.numel());
  ffi::CUDADeviceGuard guard(source.device().device_id);
  const auto s=source.dtype(), d=destination.dtype();
  TVM_FFI_ICHECK(s.lanes==1 && d.lanes==1);
  auto stream=reinterpret_cast<cudaStream_t>(uintptr_t(stream_handle));
  int64_t n=source.numel(); if (!n) return;
  TVM_FFI_ICHECK(source.IsContiguous() && destination.IsContiguous());
  #define AKA_COPY(K,S,D) K<<<(n+255)/256,256,0,stream>>>(static_cast<const S*>(source.data_ptr()),static_cast<D*>(destination.data_ptr()),n)
  if (encode_dlpack_dtype(s)==encode_dlpack_dtype(d)) {
    TVM_FFI_ICHECK(s.bits==8 || s.bits==16 || s.bits==32 || s.bits==64);
    TVM_FFI_ICHECK(cudaMemcpyAsync(destination.data_ptr(), source.data_ptr(),
                    n * (s.bits / 8), cudaMemcpyDeviceToDevice, stream)==cudaSuccess)
        << "GDN stream copy failed";
  } else {
    TVM_FFI_ICHECK(s.code==kDLInt && d.code==kDLInt);
    if (s.bits==32 && d.bits==64) {AKA_COPY(copy_i32_i64,int32_t,int64_t);}
    else if (s.bits==64 && d.bits==32) {AKA_COPY(copy_i64_i32,int64_t,int32_t);}
    else TVM_FFI_ICHECK(false) << "only int32/int64 metadata casts are supported";
  }
  #undef AKA_COPY
  TVM_FFI_ICHECK(cudaGetLastError()==cudaSuccess) << "GDN local copy launch failed";
}

__global__ void normalize_qk(const __nv_bfloat16* q, const __nv_bfloat16* k,
                             __nv_bfloat16* q_out, __nv_bfloat16* k_out,
                             int64_t rows) {
  const int64_t row = int64_t(blockIdx.x)*4 + threadIdx.x/32;
  if (row >= rows) return;
  const int lane = threadIdx.x%32;
  float qv[4], kv[4], qs=0.0f, ks=0.0f;
  #pragma unroll
  for (int j=0;j<4;++j) {
    qv[j]=__bfloat162float(q[row*128+lane+j*32]);
    kv[j]=__bfloat162float(k[row*128+lane+j*32]);
    // The original wrapper performs square then sum, without fused multiply-add.
    qs=__fadd_rn(qs,__fmul_rn(qv[j],qv[j]));
    ks=__fadd_rn(ks,__fmul_rn(kv[j],kv[j]));
  }
  for (int d=16;d>0;d/=2) {
    qs=__fadd_rn(qs,__shfl_down_sync(0xffffffff,qs,d));
    ks=__fadd_rn(ks,__shfl_down_sync(0xffffffff,ks,d));
  }
  const float qr=rsqrtf(__shfl_sync(0xffffffff,qs,0)+1e-6f);
  const float kr=rsqrtf(__shfl_sync(0xffffffff,ks,0)+1e-6f);
  #pragma unroll
  for (int j=0;j<4;++j) {
    q_out[row*128+lane+j*32]=__float2bfloat16_rn(qv[j]*qr);
    k_out[row*128+lane+j*32]=__float2bfloat16_rn(kv[j]*kr);
  }
}

inline bool is_bf16(const TensorView& x) {
  TVM_FFI_ICHECK(encode_dlpack_dtype(x.dtype())==encode_dlpack_dtype(dl_bfloat16) ||
                encode_dlpack_dtype(x.dtype())==encode_dlpack_dtype(dl_float32));
  return encode_dlpack_dtype(x.dtype())==encode_dlpack_dtype(dl_bfloat16);
}

void prepare(TensorView q, TensorView k, TensorView g, TensorView beta, TensorView state,
             TensorView q_out, TensorView k_out, TensorView alpha, TensorView beta_fp32,
             TensorView state_fp32, TensorView cu_in, TensorView cu_out,
             bool make_cu, bool normalize, int64_t stream_handle) {
  TVM_FFI_ICHECK(q.device().device_type==kDLCUDA);
  const int dev=q.device().device_id;
  ffi::CUDADeviceGuard guard(dev);
  for (const auto& x : {q,k,g,beta,state,q_out,k_out,alpha,beta_fp32,state_fp32,cu_in,cu_out}) {
    TVM_FFI_ICHECK(x.device().device_type==kDLCUDA && x.device().device_id==dev);
    TVM_FFI_ICHECK(x.IsContiguous());
  }
  TVM_FFI_ICHECK(q.numel()==k.numel() && q.numel()%128==0);
  TVM_FFI_ICHECK(q_out.numel()==q.numel() && k_out.numel()==k.numel());
  TVM_FFI_ICHECK(g.numel()==beta.numel() && alpha.numel()==g.numel() && beta_fp32.numel()==beta.numel());
  TVM_FFI_ICHECK(state_fp32.numel()==state.numel());
  for (const auto& x : {q,k,q_out,k_out}) TVM_FFI_ICHECK(encode_dlpack_dtype(x.dtype())==encode_dlpack_dtype(dl_bfloat16));
  for (const auto& x : {alpha,beta_fp32,state_fp32}) TVM_FFI_ICHECK(encode_dlpack_dtype(x.dtype())==encode_dlpack_dtype(dl_float32));
  const bool gb=is_bf16(g), bb=is_bf16(beta), sb=is_bf16(state);
  TVM_FFI_ICHECK(cu_in.dtype().code==kDLInt && cu_out.dtype().code==kDLInt);
  TVM_FFI_ICHECK((cu_in.dtype().bits==32 || cu_in.dtype().bits==64) &&
                (cu_out.dtype().bits==32 || cu_out.dtype().bits==64));
  TVM_FFI_ICHECK(cu_in.numel()==cu_out.numel() && (!make_cu || cu_out.numel()==2));
  const int64_t n_cu=(make_cu || cu_in.data_ptr()!=cu_out.data_ptr())?cu_out.numel():0;
  const int64_t count=std::max(std::max(g.numel(),sb?state.numel():int64_t(0)),n_cu);
  auto stream=reinterpret_cast<cudaStream_t>(uintptr_t(stream_handle));
  if (count) prepare_scalars<<<(count+255)/256,256,0,stream>>>(g.data_ptr(),beta.data_ptr(),state.data_ptr(),
      static_cast<float*>(alpha.data_ptr()),static_cast<float*>(beta_fp32.data_ptr()),
      static_cast<float*>(state_fp32.data_ptr()),g.numel(),state.numel(),gb,bb,sb,
      cu_in.data_ptr(),cu_out.data_ptr(),n_cu,cu_in.dtype().bits==64,
      cu_out.dtype().bits==64,make_cu,q.size(0));
  if (normalize && q.numel()) normalize_qk<<<(q.numel()/128+3)/4,128,0,stream>>>(
      static_cast<const __nv_bfloat16*>(q.data_ptr()),static_cast<const __nv_bfloat16*>(k.data_ptr()),
      static_cast<__nv_bfloat16*>(q_out.data_ptr()),static_cast<__nv_bfloat16*>(k_out.data_ptr()),q.numel()/128);
  TVM_FFI_ICHECK(cudaGetLastError()==cudaSuccess) << "GDN wrapper preprocess launch failed";
}
}  // namespace aka_gdn
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run,aka_gdn::prepare);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(copy,aka_gdn::copy_tensor);
