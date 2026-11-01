/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Allocation-free staging for Cake's head-dim-256 context kernels.
 * Product minimum architecture: sm_100a.  The copies themselves use no
 * architecture-specific instructions and share one SM100a/SM103a source.
 */

#ifdef CAKE_FMHA_DEVICE_ONLY
using uint8_t = unsigned char;
using int32_t = int;
using uint32_t = unsigned int;
using cake_fmha_int64_t = long long;
#else
#include "context_cake_fmha_hd256_support.cuh"
using cake_fmha_int64_t = int64_t;
#endif


#ifndef D06_HD256_PACK_G
#define D06_HD256_PACK_G 1
#endif

namespace {

constexpr int kThreads = 256;
constexpr int kMicroPage = 16;

// Preserve arbitrary source/output alignment. Private stage buffers are aligned;
// normal contiguous tensors take one 16-byte transaction per thread.
__device__ __forceinline__ void d06_copy16(uint8_t* dst,const uint8_t* src,bool valid) {
  if((reinterpret_cast<unsigned long long>(dst)&15)==0 &&
      (!valid || (reinterpret_cast<unsigned long long>(src)&15)==0)) {
    uint4 v={0,0,0,0};
    if(valid)v=*reinterpret_cast<const uint4*>(src);
    *reinterpret_cast<uint4*>(dst)=v;
  } else {
    #pragma unroll
    for(int j=0;j<16;++j)dst[j]=valid?src[j]:0;
  }
}


#ifndef CAKE_FMHA_DEVICE_ONLY
template <typename T>
inline void* kernel_arg(T* value) {
  return const_cast<void*>(reinterpret_cast<const void*>(value));
}
#endif

}  // namespace

extern "C" __global__ void kernel_cake_fmha_hd256_stage_q(
    const uint8_t* q,
    uint8_t* q_packed,
    const int32_t* q_indptr,
    int batch_size,
    int num_q_heads,
    int padded_q,
    int head_dim_bytes,
    cake_fmha_int64_t q_token_stride_bytes,
    cake_fmha_int64_t q_head_stride_bytes) {
  // D06 builds a private module for this runtime geometry.
  num_q_heads=NUM_Q_HEADS / D06_HD256_PACK_G; padded_q=NUM_M_BLOCKS*128; head_dim_bytes=256;

  const cake_fmha_int64_t total =
      static_cast<cake_fmha_int64_t>(batch_size) * num_q_heads * padded_q * (head_dim_bytes/16);
  for (cake_fmha_int64_t index =
           static_cast<cake_fmha_int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<cake_fmha_int64_t>(gridDim.x) * blockDim.x) {
    cake_fmha_int64_t remaining = index;
    const int byte = (remaining % (head_dim_bytes/16))*16;
    remaining /= (head_dim_bytes/16);
    const int packed_row = remaining % padded_q;
    const int token = packed_row / D06_HD256_PACK_G;
    remaining /= padded_q;
    const int head = (remaining % num_q_heads) * D06_HD256_PACK_G + packed_row % D06_HD256_PACK_G;
    const int batch = remaining / num_q_heads;
    const int32_t q_begin = q_indptr[batch];
    const int32_t q_length = q_indptr[batch + 1] - q_begin;
    const cake_fmha_int64_t source =
        static_cast<cake_fmha_int64_t>(q_begin+token)*q_token_stride_bytes+
        static_cast<cake_fmha_int64_t>(head)*q_head_stride_bytes+byte;
    d06_copy16(q_packed+index*16,q+source,token<q_length);
  }
}

extern "C" __global__ void kernel_cake_fmha_hd256_stage_kv(
    const uint8_t* k_source,
    const uint8_t* v_source,
    uint8_t* k_packed,
    uint8_t* v_packed,
    const int32_t* page_table,
    const int32_t* seq_lens,
    int batch_size,
    int num_kv_heads,
    int page_size,
    int max_micro_pages,
    int head_dim_bytes,
    cake_fmha_int64_t source_page_stride_bytes,
    cake_fmha_int64_t source_token_stride_bytes,
    cake_fmha_int64_t source_head_stride_bytes,
    cake_fmha_int64_t page_table_batch_stride,
    cake_fmha_int64_t page_table_side_stride) {
  // D06 builds a private module for this runtime geometry.
  num_kv_heads=NUM_KV_HEADS; page_size=CAKE_FMHA_SOURCE_PAGE_SIZE; head_dim_bytes=256;

  const cake_fmha_int64_t total =
      static_cast<cake_fmha_int64_t>(batch_size) * num_kv_heads * max_micro_pages *
      kMicroPage * (head_dim_bytes/16);
  for (cake_fmha_int64_t index =
           static_cast<cake_fmha_int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<cake_fmha_int64_t>(gridDim.x) * blockDim.x) {
    cake_fmha_int64_t remaining = index;
    const int byte = (remaining % (head_dim_bytes/16))*16;
    remaining /= (head_dim_bytes/16);
    const int token_in_micro = remaining % kMicroPage;
    remaining /= kMicroPage;
    const int micro_page = remaining % max_micro_pages;
    remaining /= max_micro_pages;
    const int head = remaining % num_kv_heads;
    const int batch = remaining / num_kv_heads;
    const int token = micro_page * kMicroPage + token_in_micro;
    const uint8_t* k_ptr=k_source;const uint8_t* v_ptr=v_source;
    if (token < seq_lens[batch]) {
      const int logical_page = token / page_size;
      const int page_offset = token % page_size;
      const cake_fmha_int64_t table_base =
          static_cast<cake_fmha_int64_t>(batch) * page_table_batch_stride;
      const int32_t k_page = page_table[table_base + logical_page];
      const int32_t v_page =
          page_table[table_base + page_table_side_stride + logical_page];
      const cake_fmha_int64_t inner =
          static_cast<cake_fmha_int64_t>(page_offset) * source_token_stride_bytes +
          static_cast<cake_fmha_int64_t>(head) * source_head_stride_bytes + byte;
      k_ptr=k_source+static_cast<cake_fmha_int64_t>(k_page)*source_page_stride_bytes+inner;
      v_ptr=v_source+static_cast<cake_fmha_int64_t>(v_page)*source_page_stride_bytes+inner;
    }
    d06_copy16(k_packed+index*16,k_ptr,token<seq_lens[batch]);
    d06_copy16(v_packed+index*16,v_ptr,token<seq_lens[batch]);
  }
}

extern "C" __global__ void kernel_cake_fmha_hd256_prepare_metadata(
    const int32_t* q_indptr,
    const int32_t* seq_lens,
    int32_t* seq_lens_q,
    int32_t* seq_lens_kv,
    int32_t* cu_seq_lens_q,
    int32_t* kernel_page_table,
    uint32_t* dynamic_counter,
    int batch_size,
    int num_q_heads,
    int num_kv_heads,
    int padded_q,
    int max_micro_pages) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_q_heads = batch_size * num_q_heads;
  if (index < total_q_heads) {
    const int batch = index / num_q_heads;
    seq_lens_q[index] = (q_indptr[batch + 1] - q_indptr[batch]) * D06_HD256_PACK_G;
    seq_lens_kv[index] = seq_lens[batch];
    cu_seq_lens_q[index] = index * padded_q;
  }
  const int total_kv_rows = batch_size * num_kv_heads * max_micro_pages;
  if (index < total_kv_rows) {
    kernel_page_table[index] = index;
  }
  if (index == 0) {
    dynamic_counter[0] = 0;
  }
}

extern "C" __global__ void kernel_cake_fmha_hd256_scatter_o(
    const uint8_t* o_packed,
    uint8_t* output,
    const int32_t* q_indptr,
    int batch_size,
    int num_q_heads,
    int padded_q,
    int head_dim_bytes,
    cake_fmha_int64_t output_token_stride_bytes,
    cake_fmha_int64_t output_head_stride_bytes) {
  // D06 builds a private module for this runtime geometry.
  num_q_heads=NUM_Q_HEADS / D06_HD256_PACK_G; padded_q=NUM_M_BLOCKS*128; head_dim_bytes=512;

  const cake_fmha_int64_t total =
      static_cast<cake_fmha_int64_t>(batch_size) * num_q_heads * padded_q * (head_dim_bytes/16);
  for (cake_fmha_int64_t index =
           static_cast<cake_fmha_int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<cake_fmha_int64_t>(gridDim.x) * blockDim.x) {
    cake_fmha_int64_t remaining = index;
    const int byte = (remaining % (head_dim_bytes/16))*16;
    remaining /= (head_dim_bytes/16);
    const int packed_row = remaining % padded_q;
    const int token = packed_row / D06_HD256_PACK_G;
    remaining /= padded_q;
    const int head = (remaining % num_q_heads) * D06_HD256_PACK_G + packed_row % D06_HD256_PACK_G;
    const int batch = remaining / num_q_heads;
    const int32_t q_begin = q_indptr[batch];
    const int32_t q_length = q_indptr[batch + 1] - q_begin;
    if (token < q_length) {
      const cake_fmha_int64_t destination =
          static_cast<cake_fmha_int64_t>(q_begin + token) * output_token_stride_bytes +
          static_cast<cake_fmha_int64_t>(head) * output_head_stride_bytes + byte;
      d06_copy16(output+destination,o_packed+index*16,true);
    }
  }
}

#ifndef CAKE_FMHA_DEVICE_ONLY
extern "C" cudaError_t cake_fmha_launch_hd256_stage_q(
    const uint8_t* q,
    uint8_t* q_packed,
    const int32_t* q_indptr,
    int batch_size,
    int num_q_heads,
    int padded_q,
    int head_dim_bytes,
    cake_fmha_int64_t q_token_stride_bytes,
    cake_fmha_int64_t q_head_stride_bytes,
    unsigned int grid_x,
    cudaStream_t stream) {
  void* args[] = {
      kernel_arg(&q),
      kernel_arg(&q_packed),
      kernel_arg(&q_indptr),
      kernel_arg(&batch_size),
      kernel_arg(&num_q_heads),
      kernel_arg(&padded_q),
      kernel_arg(&head_dim_bytes),
      kernel_arg(&q_token_stride_bytes),
      kernel_arg(&q_head_stride_bytes),
  };
  return cudaLaunchKernel(
      reinterpret_cast<const void*>(kernel_cake_fmha_hd256_stage_q),
      dim3(grid_x, 1, 1),
      dim3(kThreads, 1, 1),
      args,
      0,
      stream);
}

extern "C" cudaError_t cake_fmha_launch_hd256_stage_kv(
    const uint8_t* k_source,
    const uint8_t* v_source,
    uint8_t* k_packed,
    uint8_t* v_packed,
    const int32_t* page_table,
    const int32_t* seq_lens,
    int batch_size,
    int num_kv_heads,
    int page_size,
    int max_micro_pages,
    int head_dim_bytes,
    cake_fmha_int64_t source_page_stride_bytes,
    cake_fmha_int64_t source_token_stride_bytes,
    cake_fmha_int64_t source_head_stride_bytes,
    cake_fmha_int64_t page_table_batch_stride,
    cake_fmha_int64_t page_table_side_stride,
    unsigned int grid_x,
    cudaStream_t stream) {
  void* args[] = {
      kernel_arg(&k_source),
      kernel_arg(&v_source),
      kernel_arg(&k_packed),
      kernel_arg(&v_packed),
      kernel_arg(&page_table),
      kernel_arg(&seq_lens),
      kernel_arg(&batch_size),
      kernel_arg(&num_kv_heads),
      kernel_arg(&page_size),
      kernel_arg(&max_micro_pages),
      kernel_arg(&head_dim_bytes),
      kernel_arg(&source_page_stride_bytes),
      kernel_arg(&source_token_stride_bytes),
      kernel_arg(&source_head_stride_bytes),
      kernel_arg(&page_table_batch_stride),
      kernel_arg(&page_table_side_stride),
  };
  return cudaLaunchKernel(
      reinterpret_cast<const void*>(kernel_cake_fmha_hd256_stage_kv),
      dim3(grid_x, 1, 1),
      dim3(kThreads, 1, 1),
      args,
      0,
      stream);
}

extern "C" cudaError_t cake_fmha_launch_hd256_prepare_metadata(
    const int32_t* q_indptr,
    const int32_t* seq_lens,
    int32_t* seq_lens_q,
    int32_t* seq_lens_kv,
    int32_t* cu_seq_lens_q,
    int32_t* kernel_page_table,
    uint32_t* dynamic_counter,
    int batch_size,
    int num_q_heads,
    int num_kv_heads,
    int padded_q,
    int max_micro_pages,
    unsigned int grid_x,
    cudaStream_t stream) {
  void* args[] = {
      kernel_arg(&q_indptr),
      kernel_arg(&seq_lens),
      kernel_arg(&seq_lens_q),
      kernel_arg(&seq_lens_kv),
      kernel_arg(&cu_seq_lens_q),
      kernel_arg(&kernel_page_table),
      kernel_arg(&dynamic_counter),
      kernel_arg(&batch_size),
      kernel_arg(&num_q_heads),
      kernel_arg(&num_kv_heads),
      kernel_arg(&padded_q),
      kernel_arg(&max_micro_pages),
  };
  return cudaLaunchKernel(
      reinterpret_cast<const void*>(kernel_cake_fmha_hd256_prepare_metadata),
      dim3(grid_x, 1, 1),
      dim3(kThreads, 1, 1),
      args,
      0,
      stream);
}

extern "C" cudaError_t cake_fmha_launch_hd256_scatter_o(
    const uint8_t* o_packed,
    uint8_t* output,
    const int32_t* q_indptr,
    int batch_size,
    int num_q_heads,
    int padded_q,
    int head_dim_bytes,
    cake_fmha_int64_t output_token_stride_bytes,
    cake_fmha_int64_t output_head_stride_bytes,
    unsigned int grid_x,
    cudaStream_t stream) {
  void* args[] = {
      kernel_arg(&o_packed),
      kernel_arg(&output),
      kernel_arg(&q_indptr),
      kernel_arg(&batch_size),
      kernel_arg(&num_q_heads),
      kernel_arg(&padded_q),
      kernel_arg(&head_dim_bytes),
      kernel_arg(&output_token_stride_bytes),
      kernel_arg(&output_head_stride_bytes),
  };
  return cudaLaunchKernel(
      reinterpret_cast<const void*>(kernel_cake_fmha_hd256_scatter_o),
      dim3(grid_x, 1, 1),
      dim3(kThreads, 1, 1),
      args,
      0,
      stream);
}
#endif
