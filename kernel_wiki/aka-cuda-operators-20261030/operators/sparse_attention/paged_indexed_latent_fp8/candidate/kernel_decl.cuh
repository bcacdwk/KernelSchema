#pragma once
#include <cuda.h>
#include <cuda_bf16.h>
extern "C" __global__ void aka_s4_indexed_m64_d512(const __grid_constant__ CUtensorMap tmap_q,
    const __grid_constant__ CUtensorMap tmap_kv, __nv_bfloat16* __restrict__ O,
    const int* __restrict__ indices, const int* __restrict__ sparse_topk_lens,
    const int* __restrict__ seq_lens, float bmm1_scale, float bmm2_scale,
    int num_heads, int index_stride, int num_query_tokens, int sparse_topk);
extern "C" __global__ void aka_s4_indexed_m64_d576(const __grid_constant__ CUtensorMap tmap_q,
    const __grid_constant__ CUtensorMap tmap_kv,
    const __grid_constant__ CUtensorMap tmap_q_tail,
    const __grid_constant__ CUtensorMap tmap_kv_tail, __nv_bfloat16* __restrict__ O,
    const int* __restrict__ indices, const int* __restrict__ sparse_topk_lens,
    const int* __restrict__ seq_lens, float bmm1_scale, float bmm2_scale,
    int num_heads, int index_stride, int num_query_tokens, int sparse_topk);

extern "C" __global__ void aka_s4_indexed_h128_d576(const __grid_constant__ CUtensorMap tmap_q,
    const __grid_constant__ CUtensorMap tmap_kv,
    const __grid_constant__ CUtensorMap tmap_q_tail,
    const __grid_constant__ CUtensorMap tmap_kv_tail,
    const __grid_constant__ CUtensorMap tmap_o, __nv_bfloat16* __restrict__ O,
    const int* __restrict__ indices, const int* __restrict__ sparse_topk_lens,
    const int* __restrict__ seq_lens, float bmm1_scale, float bmm2_scale,
    int num_heads, int index_stride, int num_query_tokens, int sparse_topk);

extern "C" __global__ void aka_s4_split8_d512(const __grid_constant__ CUtensorMap tmap_q,
    const __grid_constant__ CUtensorMap tmap_kv, float* __restrict__ partial, float* __restrict__ split_stats,
    const int* __restrict__ indices, const int* __restrict__ sparse_topk_lens,
    const int* __restrict__ seq_lens, float bmm1_scale, float bmm2_scale,
    int num_heads, int index_stride, int num_query_tokens, int sparse_topk);

extern "C" __global__ void aka_s4_split8_d576(const __grid_constant__ CUtensorMap tmap_q,
    const __grid_constant__ CUtensorMap tmap_kv,
    const __grid_constant__ CUtensorMap tmap_q_tail,
    const __grid_constant__ CUtensorMap tmap_kv_tail, float* __restrict__ partial, float* __restrict__ split_stats,
    const int* __restrict__ indices, const int* __restrict__ sparse_topk_lens,
    const int* __restrict__ seq_lens, float bmm1_scale, float bmm2_scale,
    int num_heads, int index_stride, int num_query_tokens, int sparse_topk);

extern "C" __global__ void aka_s4_split8_merge(const float*,const float*,const int*,const int*,__nv_bfloat16*,int,int,float);
