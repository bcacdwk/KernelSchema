// Tensor binding only; metadata validation and launches live in tc_launch.cu.
#include <torch/extension.h>
#include <c10/util/Optional.h>

void s4_tc_forward(at::Tensor q, at::Tensor kv, at::Tensor ids,
    at::Tensor seq, c10::optional<at::Tensor> lens, at::Tensor out,
    double scale1, double scale2);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("s4_tc_forward", &s4_tc_forward);
}

// S4: metadata/allocation/binding only. Local M64, H128 and split8 CUDA entries compute the output.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <mutex>
#include <set>
#include "kernel_decl.cuh"

namespace {
void checked(cudaError_t error) {
  TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
}
auto encode_function() {
  // CUDA driver API is resolved once, without a link-time dependency on a
  // toolkit stub. Only the platform driver is loaded, never a device binary.
  static auto function = [] {
    void* library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    TORCH_CHECK(library, "cannot open CUDA driver");
    auto result = reinterpret_cast<decltype(&cuTensorMapEncodeTiled)>(
        dlsym(library, "cuTensorMapEncodeTiled"));
    TORCH_CHECK(result, "driver lacks cuTensorMapEncodeTiled");
    return result;
  }();
  return function;
}
CUtensorMap query_map(const at::Tensor& q) {
  uint64_t dimensions[4] = {128, static_cast<uint64_t>(q.size(-2)), 4,
                            static_cast<uint64_t>(q.size(0))};
  uint64_t strides[3] = {static_cast<uint64_t>(q.size(-1)), 128, static_cast<uint64_t>(q.stride(0))};
  uint32_t box[4] = {128,64,1,1}, step[4] = {1,1,1,1};
  CUtensorMap map{};
  auto rc = encode_function()(&map,CU_TENSOR_MAP_DATA_TYPE_UINT8,4,q.data_ptr(),
      dimensions,strides,box,step,CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(rc == CUDA_SUCCESS,"S4 query TMA descriptor: ",static_cast<int>(rc));
  return map;
}
CUtensorMap cache_map(const at::Tensor& kv) {
  uint64_t dimensions[2] = {512,static_cast<uint64_t>(kv.numel()/kv.size(-1))};
  uint64_t strides[1] = {static_cast<uint64_t>(kv.size(-1))};
  uint32_t box[2] = {128,1}, step[2] = {1,1};
  CUtensorMap map{};
  auto rc = encode_function()(&map,CU_TENSOR_MAP_DATA_TYPE_UINT8,2,kv.data_ptr(),
      dimensions,strides,box,step,CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(rc == CUDA_SUCCESS,"S4 physical-cache TMA descriptor: ",static_cast<int>(rc));
  return map;
}
CUtensorMap query_tail_map(const at::Tensor& q) {
  uint64_t dimensions[3] = {64,static_cast<uint64_t>(q.size(-2)),static_cast<uint64_t>(q.size(0))};
  uint64_t strides[2] = {576,static_cast<uint64_t>(q.stride(0))};
  uint32_t box[3] = {128,64,1}, step[3] = {1,1,1};
  CUtensorMap map{};
  auto rc=encode_function()(&map,CU_TENSOR_MAP_DATA_TYPE_UINT8,3,
      static_cast<char*>(q.data_ptr())+512,dimensions,strides,box,step,
      CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(rc==CUDA_SUCCESS,"S4 query tail TMA descriptor: ",static_cast<int>(rc));
  return map;
}
CUtensorMap cache_tail_map(const at::Tensor& kv) {
  uint64_t dimensions[2]={64,static_cast<uint64_t>(kv.numel()/576)},strides[1]={576};
  uint32_t box[2]={128,1},step[2]={1,1};
  CUtensorMap map{};
  auto rc=encode_function()(&map,CU_TENSOR_MAP_DATA_TYPE_UINT8,2,
      static_cast<char*>(kv.data_ptr())+512,dimensions,strides,box,step,
      CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(rc==CUDA_SUCCESS,"S4 cache tail TMA descriptor: ",static_cast<int>(rc));
  return map;
}

CUtensorMap output_map(const at::Tensor& out) {
  uint64_t dims[3]={64,static_cast<uint64_t>(out.numel()/512),8};
  uint64_t strides[2]={1024,128};
  uint32_t box[3]={64,32,1},step[3]={1,1,1};
  CUtensorMap map{};
  auto rc=encode_function()(&map,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,out.data_ptr(),
      dims,strides,box,step,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(rc==CUDA_SUCCESS,"S4 output TMA descriptor: ",static_cast<int>(rc));
  return map;
}
}

void s4_split_forward(at::Tensor q,at::Tensor kv,at::Tensor ids,at::Tensor seq,
    c10::optional<at::Tensor> lens,at::Tensor out,double scale1,double scale2) {
  int tokens=q.size(0),heads=q.size(-2),topk=ids.size(-1),stride=ids.stride(0);
  bool tail=q.size(-1)==576;
  auto partial=at::empty({tokens,heads,8,512},out.options().dtype(at::kFloat));
  auto stats=at::empty({tokens,heads,8,2},out.options().dtype(at::kFloat));
  auto tmq=query_map(q),tmkv=cache_map(kv);CUtensorMap tmq_tail{},tmkv_tail{};
  if(tail){tmq_tail=query_tail_map(q);tmkv_tail=cache_tail_map(kv);}
  float* part=partial.data_ptr<float>();float* st=stats.data_ptr<float>();
  const int* indices=ids.data_ptr<int>();const int* lengths=seq.data_ptr<int>();
  const int* topk_lengths=lens ? lens->data_ptr<int>() : nullptr;
  float s1=static_cast<float>(scale1),s2=static_cast<float>(scale2);
  const void* kernel=tail ? reinterpret_cast<const void*>(aka_s4_split8_d576) : reinterpret_cast<const void*>(aka_s4_split8_d512);
  int smem=tail ? 232064 : 191104;
  static std::mutex mutex;static std::set<std::pair<int,int>> configured;
  {std::lock_guard<std::mutex> lock(mutex);auto key=std::make_pair(q.get_device(),tail);
   if(!configured.count(key)){checked(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));configured.insert(key);}}
  void* args[]={&tmq,&tmkv,&part,&st,&indices,&topk_lengths,&lengths,&s1,&s2,&heads,&stride,&tokens,&topk};
  void* tail_args[]={&tmq,&tmkv,&tmq_tail,&tmkv_tail,&part,&st,&indices,&topk_lengths,&lengths,&s1,&s2,&heads,&stride,&tokens,&topk};
  cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim={1,1,1};
  cudaLaunchConfig_t config{};config.gridDim=dim3(tokens*((heads+63)/64)*8);config.blockDim=dim3(512);
  config.dynamicSmemBytes=smem;config.stream=c10::cuda::getCurrentCUDAStream(q.get_device());config.attrs=&attr;config.numAttrs=1;
  checked(cudaLaunchKernelExC(&config,kernel,tail ? tail_args : args));
  aka_s4_split8_merge<<<tokens*heads,256,0,config.stream>>>(part,st,lengths,topk_lengths,reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),heads,topk,s2);
  checked(cudaGetLastError());
}

void s4_tc_forward(at::Tensor q, at::Tensor kv, at::Tensor ids,
                  at::Tensor seq, c10::optional<at::Tensor> lens, at::Tensor out,
                  double scale1, double scale2) {
  c10::cuda::CUDAGuard guard(q.device());
  TORCH_CHECK(q.is_cuda() && q.dim()==4 && q.size(1)==1 && (q.size(-1)==512 || q.size(-1)==576));
  TORCH_CHECK(q.scalar_type()==at::kFloat8_e4m3fn && kv.scalar_type()==q.scalar_type());
  TORCH_CHECK(q.size(-2)>0 && q.size(-2)<=128 && kv.dim()==4 && kv.size(1)==1 && kv.size(-1)==q.size(-1));
  for (const auto& t : {q,kv,ids,seq,out}) {
    TORCH_CHECK(t.device()==q.device() && t.is_contiguous(),"S4 requires dense contract tensors");
  }
  TORCH_CHECK(ids.scalar_type()==at::kInt && seq.scalar_type()==at::kInt);
  TORCH_CHECK(ids.dim()==3 && ids.size(0)==q.size(0) && ids.size(1)==1 && seq.numel()==q.size(0));
  TORCH_CHECK(out.scalar_type()==at::kBFloat16 && out.dim()==4 && out.size(0)==q.size(0) && out.size(1)==1 && out.size(2)==q.size(2) && out.size(3)==512);
  if(lens) TORCH_CHECK(lens->device()==q.device() && lens->scalar_type()==at::kInt && lens->is_contiguous() && lens->numel()==q.size(0));
  if(!q.size(0)) return;
  if(q.size(0)<=32 && q.size(-2)<=64 && ids.size(-1)>=1024) {
    s4_split_forward(q,kv,ids,seq,lens,out,scale1,scale2);return;
  }
  auto tmq=query_map(q), tmkv=cache_map(kv);
  bool tail=q.size(-1)==576;
  CUtensorMap tmq_tail{},tmkv_tail{};
  if(tail) { tmq_tail=query_tail_map(q);tmkv_tail=cache_tail_map(kv); }
  bool cluster=tail && q.size(-2)==128;
  CUtensorMap tmo{}; if(cluster) tmo=output_map(out);
  const void* kernel=cluster ? reinterpret_cast<const void*>(aka_s4_indexed_h128_d576) : tail ? reinterpret_cast<const void*>(aka_s4_indexed_m64_d576)
                          : reinterpret_cast<const void*>(aka_s4_indexed_m64_d512);
  int smem=cluster ? 232448 : tail ? 232064 : 191104;
  auto output=reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  const int* indices=ids.data_ptr<int>();
  const int* lengths=seq.data_ptr<int>();
  const int* topk_lengths=lens ? lens->data_ptr<int>() : nullptr;
  float s1=static_cast<float>(scale1),s2=static_cast<float>(scale2);
  int heads=q.size(-2), index_stride=ids.stride(0), tokens=q.size(0), topk=ids.size(-1);
  static std::mutex mutex;
  static std::set<std::pair<int,int>> configured;
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto key=std::make_pair(q.get_device(),static_cast<int>(q.size(-1)) + (cluster ? 1000 : 0));
    if(!configured.count(key)) {
      checked(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
      configured.insert(key);
    }
  }
  void* args[]={&tmq,&tmkv,&output,&indices,&topk_lengths,&lengths,&s1,&s2,
                &heads,&index_stride,&tokens,&topk};
  void* tail_args[]={&tmq,&tmkv,&tmq_tail,&tmkv_tail,&output,&indices,&topk_lengths,&lengths,
                    &s1,&s2,&heads,&index_stride,&tokens,&topk};
  void* cluster_args[]={&tmq,&tmkv,&tmq_tail,&tmkv_tail,&tmo,&output,&indices,&topk_lengths,&lengths,
                    &s1,&s2,&heads,&index_stride,&tokens,&topk};
  cudaLaunchAttribute attributes[2]{};
  attributes[0].id=cudaLaunchAttributeClusterDimension;
  attributes[0].val.clusterDim={cluster ? 2U : 1U,1,1};
  attributes[1].id=cudaLaunchAttributeClusterSchedulingPolicyPreference;
  attributes[1].val.clusterSchedulingPolicyPreference=cudaClusterSchedulingPolicySpread;
  cudaLaunchConfig_t config{};
  config.gridDim=dim3(tokens*((heads+63)/64),1,1);config.blockDim=dim3(512,1,1);
  config.dynamicSmemBytes=smem;config.stream=c10::cuda::getCurrentCUDAStream(q.get_device());
  config.attrs=attributes;config.numAttrs=2;
  checked(cudaLaunchKernelExC(&config,kernel,cluster ? cluster_args : tail ? tail_args : args));
}

#include <cuda_bf16.h>
#include <math_constants.h>
#include <cuda_runtime.h>
// One output row per CTA; partial data/metadata is consumed on the same stream.
extern "C" __global__ void aka_s4_split8_merge(const float* partial,const float* stats,
    const int* seq,const int* lens,__nv_bfloat16* out,int heads,int topk,float scale) {
    __shared__ float weights[8];
    __shared__ float normalization;
    int row=blockIdx.x,query=row/heads;
    if(threadIdx.x==0) {
        int active=max(0,min(topk,lens ? min(seq[query],lens[query]) : seq[query]));
        int tiles=(active+127)/128,chunk=(tiles+7)/8;
        float maximum=-CUDART_INF_F;
        for(int s=0;s<8;++s) if(s*chunk<tiles) maximum=fmaxf(maximum,stats[(row*8+s)*2]);
        float denominator=0.f;
        for(int s=0;s<8;++s) {
            float w=s*chunk<tiles ? exp2f(stats[(row*8+s)*2]-maximum) : 0.f;
            weights[s]=w;
            if(w>0.f) denominator+=w*stats[(row*8+s)*2+1];
        }
        normalization=denominator>0.f ? scale/denominator : 0.f;
    }
    __syncthreads();
    int col=threadIdx.x*2;float x=0.f,y=0.f;
    #pragma unroll
    for(int s=0;s<8;++s) if(weights[s]>0.f) {
        float2 v=*reinterpret_cast<const float2*>(partial+(row*8+s)*512+col);
        x=fmaf(weights[s],v.x,x);y=fmaf(weights[s],v.y,y);
    }
    *reinterpret_cast<__nv_bfloat162*>(out+row*512+col)=__floats2bfloat162_rn(x*normalization,y*normalization);
}
