// D06 stored NVFP4 -> finite-saturated RNE E4M3, without global descales.
// Values are converted before the original FP8 Tensor Core context pipeline.
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include "host.cuh"

// Four-bit magnitudes in half units: {0,1,2,3,4,6,8,12}.
// The packed integer lookup has no dynamically indexed thread-local array.
__device__ __forceinline__ float d06_e2m1(int code) {
  float x=float((0xc8643210u >> ((code&7)*4))&15)*0.5f;
  return (code&8)?-x:x;
}
template<int page_size,int head_dim,int output_dim>
__global__ void d06_decode_kv(const uint8_t* kp,const uint8_t* vp,
    const __nv_fp8_e4m3* ks,const __nv_fp8_e4m3* vs,
    __nv_fp8_e4m3* k,__nv_fp8_e4m3* v,long long rows, const uint8_t* q,uint8_t* qp,long long qrows,uint32_t* counter) {
  constexpr int out_groups=output_dim/16,groups=head_dim/16;
  const long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i==0){counter[0]=0;counter[1]=0;}
  if constexpr(head_dim==64){
    if(i<qrows*8){
      const int group=i%8;const long long rowq=i/8;
      uint4 data={0,0,0,0};
      if(group<4){
        const uint8_t* src=q+rowq*64+group*16;
        if((reinterpret_cast<uintptr_t>(src)&15)==0)data=*reinterpret_cast<const uint4*>(src);
        else {
          uint8_t* dst=reinterpret_cast<uint8_t*>(&data);
          #pragma unroll
          for(int j=0;j<16;++j)dst[j]=src[j];
        }
      }
      reinterpret_cast<uint4*>(qp)[i]=data;
    }
  }
  if(i>=rows*out_groups)return;
  const int g=i%out_groups;const long long row=i/out_groups;
  uint4 ko={0,0,0,0},vo={0,0,0,0};
  if(g<groups){
    const int token=row%page_size;const long long ph=row/page_size;
    const long long base=row*(head_dim/2)+g*8;
    const float kscale=float(ks[row*groups+g]);
    const float vscale=float(vs[((ph*(page_size/4)+token/4)*groups+g)*4+token%4]);
    uint32_t* kw=&ko.x;uint32_t* vw=&vo.x;
    #pragma unroll
    for(int word=0;word<4;++word){
      uint32_t a=0,b=0;
      #pragma unroll
      for(int pair=0;pair<2;++pair){
        const int kc=kp[base+word*2+pair],vc=vp[base+word*2+pair];
        const __nv_fp8x2_e4m3 ka(make_float2(d06_e2m1(kc&15)*kscale,d06_e2m1(kc>>4)*kscale));
        const __nv_fp8x2_e4m3 va(make_float2(d06_e2m1(vc&15)*vscale,d06_e2m1(vc>>4)*vscale));
        a|=uint32_t(ka.__x)<<(pair*16);b|=uint32_t(va.__x)<<(pair*16);
      }
      kw[word]=a;vw[word]=b;
    }
  }
  // These are candidate-owned, naturally aligned allocations.
  reinterpret_cast<uint4*>(k)[i]=ko;reinterpret_cast<uint4*>(v)[i]=vo;
}

void d06_prepare(TensorView q,TensorView qp,TensorView kp,TensorView vp,
    TensorView ks,TensorView vs,TensorView k,TensorView v,TensorView counter,int64_t stream_handle) {
  TVM_FFI_ICHECK(q.IsContiguous()&&kp.IsContiguous()&&vp.IsContiguous());
  TVM_FFI_ICHECK(ks.IsContiguous()&&vs.IsContiguous()&&k.IsContiguous()&&v.IsContiguous());
  TVM_FFI_ICHECK(q.size(2)==64||q.size(2)==128||q.size(2)==256);
  ffi::CUDADeviceGuard guard(q.device().device_id);cudaStream_t stream=reinterpret_cast<cudaStream_t>(stream_handle);
  TVM_FFI_ICHECK(counter.IsContiguous() && counter.numel()*get_element_size(counter)>=8);
  const long long rows=kp.size(0)*kp.size(1)*kp.size(2);
  const long long qrows=q.size(0)*q.size(1);
  const long long groups=rows*(k.size(3)/16);
  const long long qgroups=D06_OUTPUT_HEAD_DIM==64?qrows*8:0;
  const long long work=groups>qgroups?groups:qgroups;
  d06_decode_kv<D06_SOURCE_PAGE_SIZE,D06_OUTPUT_HEAD_DIM,(D06_OUTPUT_HEAD_DIM==256?256:128)><<<(work+255)/256,256,0,stream>>>(
    static_cast<const uint8_t*>(kp.data_ptr()),static_cast<const uint8_t*>(vp.data_ptr()),
    static_cast<const __nv_fp8_e4m3*>(ks.data_ptr()),static_cast<const __nv_fp8_e4m3*>(vs.data_ptr()),
    static_cast<__nv_fp8_e4m3*>(k.data_ptr()),static_cast<__nv_fp8_e4m3*>(v.data_ptr()),rows,
    static_cast<const uint8_t*>(q.data_ptr()),static_cast<uint8_t*>(qp.data_ptr()),qrows,static_cast<uint32_t*>(counter.data_ptr()));
  TVM_FFI_ICHECK_EQ(cudaGetLastError(),cudaSuccess);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(d06_prepare,d06_prepare);

template<int page_size,int head_dim,int output_dim>
__global__ void d06_decode_window_kv(const uint8_t* kp,const uint8_t* vp,
    const __nv_fp8_e4m3* ks,const __nv_fp8_e4m3* vs,
    __nv_fp8_e4m3* k,__nv_fp8_e4m3* v,long long rows, const uint8_t* q,uint8_t* qp,long long qrows,uint32_t* counter, const int* tables,const int* cuq,const int* lengths,int* compact_tables, int table_batch_stride,int table_value_stride,int compact_batch_stride,int compact_value_stride,int capacity,int kvheads,int window) {
  constexpr int out_groups=output_dim/16,groups=head_dim/16;
  const long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i==0){counter[0]=0;counter[1]=0;}
  if constexpr(head_dim==64){
    if(i<qrows*8){
      const int group=i%8;const long long rowq=i/8;
      uint4 data={0,0,0,0};
      if(group<4){
        const uint8_t* src=q+rowq*64+group*16;
        if((reinterpret_cast<uintptr_t>(src)&15)==0)data=*reinterpret_cast<const uint4*>(src);
        else {
          uint8_t* dst=reinterpret_cast<uint8_t*>(&data);
          #pragma unroll
          for(int j=0;j<16;++j)dst[j]=src[j];
        }
      }
      reinterpret_cast<uint4*>(qp)[i]=data;
    }
  }
  if(i>=rows*out_groups)return;
  const int g=i%out_groups;const long long row=i/out_groups;
  uint4 ko={0,0,0,0},vo={0,0,0,0};
  const int token=row%page_size;
  const long long output_ph=row/page_size;
  const int h=output_ph%kvheads;
  const int physical=output_ph/kvheads;
  const int b=physical/capacity,slot=physical%capacity;
  const int len=lengths[b],qlen=cuq[b+1]-cuq[b];
  // Attention traverses complete 128-token KV blocks, including the masked
  // left boundary. Retain the complete containing physical page as well.
  const int first_page=(max(0,len-qlen-window)/128*128)/page_size;
  const int logical_page=first_page+slot;
  const bool valid=logical_page<(len+page_size-1)/page_size;
  if(valid && h==0 && token==0 && g==0){
    compact_tables[b*compact_batch_stride+logical_page]=physical;
    if(compact_value_stride)compact_tables[b*compact_batch_stride+compact_value_stride+logical_page]=physical;
  }
  if(valid && g<groups){
    const int pk=tables[b*table_batch_stride+logical_page];
    const int pv=tables[b*table_batch_stride+table_value_stride+logical_page];
    const long long krow=(static_cast<long long>(pk)*kvheads+h)*page_size+token;
    const long long vph=static_cast<long long>(pv)*kvheads+h;
    const long long vrow=vph*page_size+token;
    const long long kbase=krow*(head_dim/2)+g*8,vbase=vrow*(head_dim/2)+g*8;
    const float kscale=float(ks[krow*groups+g]);
    const float vscale=float(vs[((vph*(page_size/4)+token/4)*groups+g)*4+token%4]);
    uint32_t* kw=&ko.x;uint32_t* vw=&vo.x;
    #pragma unroll
    for(int word=0;word<4;++word){
      uint32_t a=0,b=0;
      #pragma unroll
      for(int pair=0;pair<2;++pair){
        const int kc=kp[kbase+word*2+pair],vc=vp[vbase+word*2+pair];
        const __nv_fp8x2_e4m3 ka(make_float2(d06_e2m1(kc&15)*kscale,d06_e2m1(kc>>4)*kscale));
        const __nv_fp8x2_e4m3 va(make_float2(d06_e2m1(vc&15)*vscale,d06_e2m1(vc>>4)*vscale));
        a|=uint32_t(ka.__x)<<(pair*16);b|=uint32_t(va.__x)<<(pair*16);
      }
      kw[word]=a;vw[word]=b;
    }
  }
  // These are candidate-owned, naturally aligned allocations.
  reinterpret_cast<uint4*>(k)[i]=ko;reinterpret_cast<uint4*>(v)[i]=vo;
}


void d06_prepare_window(TensorView q,TensorView qp,TensorView kp,TensorView vp,
    TensorView ks,TensorView vs,TensorView k,TensorView v,TensorView counter,
    TensorView tables,TensorView cuq,TensorView lengths,TensorView compact_tables,
    int64_t capacity,int64_t window,int64_t stream_handle) {
  TVM_FFI_ICHECK(q.IsContiguous()&&kp.IsContiguous()&&vp.IsContiguous());
  TVM_FFI_ICHECK(ks.IsContiguous()&&vs.IsContiguous()&&k.IsContiguous()&&v.IsContiguous());
  TVM_FFI_ICHECK(cuq.IsContiguous()&&lengths.IsContiguous()&&compact_tables.IsContiguous());
  TVM_FFI_ICHECK(q.size(2)==64||q.size(2)==128);
  TVM_FFI_ICHECK(counter.IsContiguous()&&counter.numel()*get_element_size(counter)>=8);
  TVM_FFI_ICHECK(capacity>0&&window>=0&&k.size(0)==lengths.size(0)*capacity);
  ffi::CUDADeviceGuard guard(q.device().device_id);
  cudaStream_t stream=reinterpret_cast<cudaStream_t>(stream_handle);
  const long long rows=k.size(0)*k.size(1)*k.size(2),qrows=q.size(0)*q.size(1);
  const long long groups=rows*8,qgroups=D06_OUTPUT_HEAD_DIM==64?qrows*8:0;
  const long long work=groups>qgroups?groups:qgroups;
  const int vstride=tables.ndim()==3?tables.stride(1):0;
  const int compact_vstride=compact_tables.ndim()==3?compact_tables.stride(1):0;
  d06_decode_window_kv<D06_SOURCE_PAGE_SIZE,D06_OUTPUT_HEAD_DIM,128><<<(work+255)/256,256,0,stream>>>(
    static_cast<const uint8_t*>(kp.data_ptr()),static_cast<const uint8_t*>(vp.data_ptr()),
    static_cast<const __nv_fp8_e4m3*>(ks.data_ptr()),static_cast<const __nv_fp8_e4m3*>(vs.data_ptr()),
    static_cast<__nv_fp8_e4m3*>(k.data_ptr()),static_cast<__nv_fp8_e4m3*>(v.data_ptr()),rows,
    static_cast<const uint8_t*>(q.data_ptr()),static_cast<uint8_t*>(qp.data_ptr()),qrows,
    static_cast<uint32_t*>(counter.data_ptr()),static_cast<const int*>(tables.data_ptr()),
    static_cast<const int*>(cuq.data_ptr()),static_cast<const int*>(lengths.data_ptr()),
    static_cast<int*>(compact_tables.data_ptr()),tables.stride(0),vstride,compact_tables.stride(0),
    compact_vstride,capacity,kp.size(1),window);
  TVM_FFI_ICHECK_EQ(cudaGetLastError(),cudaSuccess);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(d06_prepare_window,d06_prepare_window);


// Context metadata fusion; legacy preparation instantiations remain unchanged.
template<int page_size,int head_dim,int output_dim>
__global__ void d06_decode_kv_metadata(const uint8_t* kp,const uint8_t* vp,
    const __nv_fp8_e4m3* ks,const __nv_fp8_e4m3* vs,
    __nv_fp8_e4m3* k,__nv_fp8_e4m3* v,long long rows, const uint8_t* q,uint8_t* qp,long long qrows,uint32_t* counter,const int* metadata_cuq,const int* metadata_lengths,int* metadata_q,int* metadata_kv,int* metadata_cu,int metadata_bh) {
  constexpr int out_groups=output_dim/16,groups=head_dim/16;
  const long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<metadata_bh){
    const int b=i/(NUM_Q_HEADS/PACK_G);
    metadata_q[i]=metadata_cuq[b+1]-metadata_cuq[b];
    metadata_kv[i]=metadata_lengths[b];
    metadata_cu[i]=metadata_cuq[b];
  }
  if(i==0){counter[0]=0;counter[1]=0;}
  if constexpr(head_dim==64){
    if(i<qrows*4){
      const int group=i%4;const long long rowq=i/4;
      uint4 data={0,0,0,0};
      if(group<4){
        const uint8_t* src=q+rowq*64+group*16;
        if((reinterpret_cast<uintptr_t>(src)&15)==0)data=*reinterpret_cast<const uint4*>(src);
        else {
          uint8_t* dst=reinterpret_cast<uint8_t*>(&data);
          #pragma unroll
          for(int j=0;j<16;++j)dst[j]=src[j];
        }
      }
      reinterpret_cast<uint4*>(qp)[i]=data;
    }
  }
  if(i>=rows*out_groups)return;
  const int g=i%out_groups;const long long row=i/out_groups;
  uint4 ko={0,0,0,0},vo={0,0,0,0};
  if(g<groups){
    const int token=row%page_size;const long long ph=row/page_size;
    const long long base=row*(head_dim/2)+g*8;
    const float kscale=float(ks[row*groups+g]);
    const float vscale=float(vs[((ph*(page_size/4)+token/4)*groups+g)*4+token%4]);
    uint32_t* kw=&ko.x;uint32_t* vw=&vo.x;
    #pragma unroll
    for(int word=0;word<4;++word){
      uint32_t a=0,b=0;
      #pragma unroll
      for(int pair=0;pair<2;++pair){
        const int kc=kp[base+word*2+pair],vc=vp[base+word*2+pair];
        const __nv_fp8x2_e4m3 ka(make_float2(d06_e2m1(kc&15)*kscale,d06_e2m1(kc>>4)*kscale));
        const __nv_fp8x2_e4m3 va(make_float2(d06_e2m1(vc&15)*vscale,d06_e2m1(vc>>4)*vscale));
        a|=uint32_t(ka.__x)<<(pair*16);b|=uint32_t(va.__x)<<(pair*16);
      }
      kw[word]=a;vw[word]=b;
    }
  }
  // These are candidate-owned, naturally aligned allocations.
  reinterpret_cast<uint4*>(k)[i]=ko;reinterpret_cast<uint4*>(v)[i]=vo;
}

void d06_prepare_metadata(TensorView q,TensorView qp,TensorView kp,TensorView vp,
    TensorView ks,TensorView vs,TensorView k,TensorView v,TensorView counter,int64_t stream_handle,TensorView metadata_cuq,TensorView metadata_lengths,TensorView metadata_workspace) {
  TVM_FFI_ICHECK(q.IsContiguous()&&kp.IsContiguous()&&vp.IsContiguous());
  TVM_FFI_ICHECK(ks.IsContiguous()&&vs.IsContiguous()&&k.IsContiguous()&&v.IsContiguous());
  TVM_FFI_ICHECK(q.size(2)==64||q.size(2)==128||q.size(2)==256);
  ffi::CUDADeviceGuard guard(q.device().device_id);cudaStream_t stream=reinterpret_cast<cudaStream_t>(stream_handle);
  TVM_FFI_ICHECK(counter.IsContiguous() && counter.numel()*get_element_size(counter)>=8);
  const long long rows=kp.size(0)*kp.size(1)*kp.size(2);
  const long long qrows=q.size(0)*q.size(1);
  const long long groups=rows*(k.size(3)/16);
  const long long qgroups=D06_OUTPUT_HEAD_DIM==64?qrows*4:0;
  const int metadata_bh=metadata_lengths.numel()*(NUM_Q_HEADS/PACK_G);
  const int64_t metadata_kv_offset=(metadata_bh*4+15)/16*16;
  const int64_t metadata_cu_offset=metadata_kv_offset+metadata_bh*4;
  TVM_FFI_ICHECK_GE(metadata_workspace.numel()*get_element_size(metadata_workspace),metadata_cu_offset+metadata_bh*4);
  const long long decode_work=groups>qgroups?groups:qgroups;
  const long long work=decode_work>metadata_bh?decode_work:metadata_bh;
  d06_decode_kv_metadata<D06_SOURCE_PAGE_SIZE,D06_OUTPUT_HEAD_DIM,D06_OUTPUT_HEAD_DIM><<<(work+255)/256,256,0,stream>>>(
    static_cast<const uint8_t*>(kp.data_ptr()),static_cast<const uint8_t*>(vp.data_ptr()),
    static_cast<const __nv_fp8_e4m3*>(ks.data_ptr()),static_cast<const __nv_fp8_e4m3*>(vs.data_ptr()),
    static_cast<__nv_fp8_e4m3*>(k.data_ptr()),static_cast<__nv_fp8_e4m3*>(v.data_ptr()),rows,
    static_cast<const uint8_t*>(q.data_ptr()),static_cast<uint8_t*>(qp.data_ptr()),qrows,static_cast<uint32_t*>(counter.data_ptr()),static_cast<const int*>(metadata_cuq.data_ptr()),static_cast<const int*>(metadata_lengths.data_ptr()),reinterpret_cast<int*>(metadata_workspace.data_ptr()),reinterpret_cast<int*>(static_cast<uint8_t*>(metadata_workspace.data_ptr())+metadata_kv_offset),reinterpret_cast<int*>(static_cast<uint8_t*>(metadata_workspace.data_ptr())+metadata_cu_offset),metadata_bh);
  TVM_FFI_ICHECK_EQ(cudaGetLastError(),cudaSuccess);
}

template<int page_size,int head_dim,int output_dim>
__global__ void d06_decode_window_kv_metadata(const uint8_t* kp,const uint8_t* vp,
    const __nv_fp8_e4m3* ks,const __nv_fp8_e4m3* vs,
    __nv_fp8_e4m3* k,__nv_fp8_e4m3* v,long long rows, const uint8_t* q,uint8_t* qp,long long qrows,uint32_t* counter, const int* tables,const int* cuq,const int* lengths,int* compact_tables, int table_batch_stride,int table_value_stride,int compact_batch_stride,int compact_value_stride,int capacity,int kvheads,int window,const int* metadata_cuq,const int* metadata_lengths,int* metadata_q,int* metadata_kv,int* metadata_cu,int metadata_bh) {
  constexpr int out_groups=output_dim/16,groups=head_dim/16;
  const long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<metadata_bh){
    const int b=i/(NUM_Q_HEADS/PACK_G);
    metadata_q[i]=metadata_cuq[b+1]-metadata_cuq[b];
    metadata_kv[i]=metadata_lengths[b];
    metadata_cu[i]=metadata_cuq[b];
  }
  if(i==0){counter[0]=0;counter[1]=0;}
  if constexpr(head_dim==64){
    if(i<qrows*4){
      const int group=i%4;const long long rowq=i/4;
      uint4 data={0,0,0,0};
      if(group<4){
        const uint8_t* src=q+rowq*64+group*16;
        if((reinterpret_cast<uintptr_t>(src)&15)==0)data=*reinterpret_cast<const uint4*>(src);
        else {
          uint8_t* dst=reinterpret_cast<uint8_t*>(&data);
          #pragma unroll
          for(int j=0;j<16;++j)dst[j]=src[j];
        }
      }
      reinterpret_cast<uint4*>(qp)[i]=data;
    }
  }
  if(i>=rows*out_groups)return;
  const int g=i%out_groups;const long long row=i/out_groups;
  uint4 ko={0,0,0,0},vo={0,0,0,0};
  const int token=row%page_size;
  const long long output_ph=row/page_size;
  const int h=output_ph%kvheads;
  const int physical=output_ph/kvheads;
  const int b=physical/capacity,slot=physical%capacity;
  const int len=lengths[b],qlen=cuq[b+1]-cuq[b];
  // Attention traverses complete 128-token KV blocks, including the masked
  // left boundary. Retain the complete containing physical page as well.
  const int first_page=(max(0,len-qlen-window)/128*128)/page_size;
  const int logical_page=first_page+slot;
  const bool valid=logical_page<(len+page_size-1)/page_size;
  if(valid && h==0 && token==0 && g==0){
    compact_tables[b*compact_batch_stride+logical_page]=physical;
    if(compact_value_stride)compact_tables[b*compact_batch_stride+compact_value_stride+logical_page]=physical;
  }
  if(valid && g<groups){
    const int pk=tables[b*table_batch_stride+logical_page];
    const int pv=tables[b*table_batch_stride+table_value_stride+logical_page];
    const long long krow=(static_cast<long long>(pk)*kvheads+h)*page_size+token;
    const long long vph=static_cast<long long>(pv)*kvheads+h;
    const long long vrow=vph*page_size+token;
    const long long kbase=krow*(head_dim/2)+g*8,vbase=vrow*(head_dim/2)+g*8;
    const float kscale=float(ks[krow*groups+g]);
    const float vscale=float(vs[((vph*(page_size/4)+token/4)*groups+g)*4+token%4]);
    uint32_t* kw=&ko.x;uint32_t* vw=&vo.x;
    #pragma unroll
    for(int word=0;word<4;++word){
      uint32_t a=0,b=0;
      #pragma unroll
      for(int pair=0;pair<2;++pair){
        const int kc=kp[kbase+word*2+pair],vc=vp[vbase+word*2+pair];
        const __nv_fp8x2_e4m3 ka(make_float2(d06_e2m1(kc&15)*kscale,d06_e2m1(kc>>4)*kscale));
        const __nv_fp8x2_e4m3 va(make_float2(d06_e2m1(vc&15)*vscale,d06_e2m1(vc>>4)*vscale));
        a|=uint32_t(ka.__x)<<(pair*16);b|=uint32_t(va.__x)<<(pair*16);
      }
      kw[word]=a;vw[word]=b;
    }
  }
  // These are candidate-owned, naturally aligned allocations.
  reinterpret_cast<uint4*>(k)[i]=ko;reinterpret_cast<uint4*>(v)[i]=vo;
}

void d06_prepare_window_metadata(TensorView q,TensorView qp,TensorView kp,TensorView vp,
    TensorView ks,TensorView vs,TensorView k,TensorView v,TensorView counter,
    TensorView tables,TensorView cuq,TensorView lengths,TensorView compact_tables,
    int64_t capacity,int64_t window,int64_t stream_handle,TensorView metadata_cuq,TensorView metadata_lengths,TensorView metadata_workspace) {
  TVM_FFI_ICHECK(q.IsContiguous()&&kp.IsContiguous()&&vp.IsContiguous());
  TVM_FFI_ICHECK(ks.IsContiguous()&&vs.IsContiguous()&&k.IsContiguous()&&v.IsContiguous());
  TVM_FFI_ICHECK(cuq.IsContiguous()&&lengths.IsContiguous()&&compact_tables.IsContiguous());
  TVM_FFI_ICHECK(q.size(2)==64||q.size(2)==128);
  TVM_FFI_ICHECK(counter.IsContiguous()&&counter.numel()*get_element_size(counter)>=8);
  TVM_FFI_ICHECK(capacity>0&&window>=0&&k.size(0)==lengths.size(0)*capacity);
  ffi::CUDADeviceGuard guard(q.device().device_id);
  cudaStream_t stream=reinterpret_cast<cudaStream_t>(stream_handle);
  const long long rows=k.size(0)*k.size(1)*k.size(2),qrows=q.size(0)*q.size(1);
  const long long groups=rows*(D06_OUTPUT_HEAD_DIM/16),qgroups=D06_OUTPUT_HEAD_DIM==64?qrows*4:0;
  const int metadata_bh=metadata_lengths.numel()*(NUM_Q_HEADS/PACK_G);
  const int64_t metadata_kv_offset=(metadata_bh*4+15)/16*16;
  const int64_t metadata_cu_offset=metadata_kv_offset+metadata_bh*4;
  TVM_FFI_ICHECK_GE(metadata_workspace.numel()*get_element_size(metadata_workspace),metadata_cu_offset+metadata_bh*4);
  const long long decode_work=groups>qgroups?groups:qgroups;
  const long long work=decode_work>metadata_bh?decode_work:metadata_bh;
  const int vstride=tables.ndim()==3?tables.stride(1):0;
  const int compact_vstride=compact_tables.ndim()==3?compact_tables.stride(1):0;
  d06_decode_window_kv_metadata<D06_SOURCE_PAGE_SIZE,D06_OUTPUT_HEAD_DIM,D06_OUTPUT_HEAD_DIM><<<(work+255)/256,256,0,stream>>>(
    static_cast<const uint8_t*>(kp.data_ptr()),static_cast<const uint8_t*>(vp.data_ptr()),
    static_cast<const __nv_fp8_e4m3*>(ks.data_ptr()),static_cast<const __nv_fp8_e4m3*>(vs.data_ptr()),
    static_cast<__nv_fp8_e4m3*>(k.data_ptr()),static_cast<__nv_fp8_e4m3*>(v.data_ptr()),rows,
    static_cast<const uint8_t*>(q.data_ptr()),static_cast<uint8_t*>(qp.data_ptr()),qrows,
    static_cast<uint32_t*>(counter.data_ptr()),static_cast<const int*>(tables.data_ptr()),
    static_cast<const int*>(cuq.data_ptr()),static_cast<const int*>(lengths.data_ptr()),
    static_cast<int*>(compact_tables.data_ptr()),tables.stride(0),vstride,compact_tables.stride(0),
    compact_vstride,capacity,kp.size(1),window,static_cast<const int*>(metadata_cuq.data_ptr()),static_cast<const int*>(metadata_lengths.data_ptr()),reinterpret_cast<int*>(metadata_workspace.data_ptr()),reinterpret_cast<int*>(static_cast<uint8_t*>(metadata_workspace.data_ptr())+metadata_kv_offset),reinterpret_cast<int*>(static_cast<uint8_t*>(metadata_workspace.data_ptr())+metadata_cu_offset),metadata_bh);
  TVM_FFI_ICHECK_EQ(cudaGetLastError(),cudaSuccess);
}
