#pragma once
// mla_absorbed_fp8 v0: general SIMT reference-grade kernel.
// One warp per (query row, head). Pure CUDA, no libraries.
// Purpose: validate ABI, geometry, masking, sinks, LSE and storage semantics.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace mla_v0 {

constexpr int kHeadDim = 576;
constexpr int kValueDim = 512;
constexpr int kWarp = 32;
constexpr int kWarpsPerBlock = 8;

__device__ __forceinline__ float fp8_byte_to_float(uint8_t b) {
  __nv_fp8_e4m3 v;
  v.__x = b;
  return float(v);
}

struct RowMap {
  int batch;
  int index;
  int qlen;
};

__device__ __forceinline__ RowMap map_row(int row, int uniform_q, const int* __restrict__ cum_q) {
  RowMap m;
  if (cum_q == nullptr) {
    m.batch = row / uniform_q;
    m.index = row - m.batch * uniform_q;
    m.qlen = uniform_q;
    return m;
  }
  int lo = 0;
  int hi = 0;
  // cum_q has B+1 entries; find b with cum_q[b] <= row < cum_q[b+1].
  // B <= 32, so a linear scan is cheap and avoids reading B from the host.
  int previous = cum_q[0];
  for (int b = 1;; ++b) {
    int next = cum_q[b];
    if (row < next) {
      hi = b;
      break;
    }
    previous = next;
    ++lo;
  }
  (void)lo;
  m.batch = hi - 1;
  m.index = row - previous;
  m.qlen = cum_q[hi] - previous;
  return m;
}

__global__ void mla_v0_kernel(
    const uint8_t* __restrict__ q,
    const uint8_t* __restrict__ kv,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const int* __restrict__ cum_q,
    uint8_t* __restrict__ out,
    float* __restrict__ lse,
    const float* __restrict__ sinks,
    int heads,
    int uniform_q,
    int page_size,
    int table_stride,
    float bmm1,
    float bmm2,
    long long total_warps) {
  const int warp_in_block = threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;
  const long long warp_id = (long long)blockIdx.x * kWarpsPerBlock + warp_in_block;
  if (warp_id >= total_warps) return;
  const long long row = warp_id / heads;
  const int head = (int)(warp_id - row * heads);
  const RowMap mapping = map_row((int)row, uniform_q, cum_q);
  const int batch = mapping.batch;
  const int kv_len = seq_lens[batch];
  const int visible = kv_len - mapping.qlen + mapping.index;

  // Per-lane dimension ownership: nope [16*lane, 16*lane+16), rope [512+2*lane, +2).
  const uint8_t* qrow = q + (row * heads + head) * kHeadDim;
  const int nope = lane * 16;
  const int rope = kValueDim + lane * 2;
  float qn[16];
  float qr[2];
  uint4 qraw = *reinterpret_cast<const uint4*>(qrow + nope);
  {
    const uint8_t* b = reinterpret_cast<const uint8_t*>(&qraw);
#pragma unroll
    for (int i = 0; i < 16; ++i) qn[i] = fp8_byte_to_float(b[i]);
  }
  uint16_t qrope_raw = *reinterpret_cast<const uint16_t*>(qrow + rope);
  {
    const uint8_t* b = reinterpret_cast<const uint8_t*>(&qrope_raw);
    qr[0] = fp8_byte_to_float(b[0]);
    qr[1] = fp8_byte_to_float(b[1]);
  }

  float acc[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) acc[i] = 0.f;
  float denom = 0.f;
  float running_max = -INFINITY;
  const int* table = block_tables + (long long)batch * table_stride;
  const int keys = visible + 1;
  for (int j = 0; j < keys; ++j) {
    const int page = table[j / page_size];
    const int offset = j - (j / page_size) * page_size;
    const uint8_t* krow = kv + ((long long)page * page_size + offset) * kHeadDim;
    uint4 kraw = *reinterpret_cast<const uint4*>(krow + nope);
    uint16_t krope_raw = *reinterpret_cast<const uint16_t*>(krow + rope);
    const uint8_t* kb = reinterpret_cast<const uint8_t*>(&kraw);
    const uint8_t* krb = reinterpret_cast<const uint8_t*>(&krope_raw);
    float partial = qr[0] * fp8_byte_to_float(krb[0]) + qr[1] * fp8_byte_to_float(krb[1]);
#pragma unroll
    for (int i = 0; i < 16; ++i) partial += qn[i] * fp8_byte_to_float(kb[i]);
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) partial += __shfl_xor_sync(0xffffffffu, partial, s);
    const float score = partial * bmm1;
    const float new_max = fmaxf(running_max, score);
    const float safe_max = new_max == -INFINITY ? 0.f : new_max;
    const float rescale = expf(running_max - safe_max);
    const float weight = expf(score - safe_max);
    running_max = new_max;
    denom = denom * rescale + weight;
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = acc[i] * rescale + weight * fp8_byte_to_float(kb[i]);
  }

  const float inv_denom = denom > 0.f ? 1.f / denom : 0.f;
  float base_lse = running_max + logf(denom);
  float scale_extra = 1.f;
  if (sinks != nullptr) {
    const float sink = sinks[head];
    const float hi = fmaxf(base_lse, sink);
    const float total = hi + logf(expf(base_lse - hi) + expf(sink - hi));
    scale_extra = expf(base_lse - total);
    base_lse = total;
  }
  __nv_bfloat16* orow = reinterpret_cast<__nv_bfloat16*>(out) + (row * heads + head) * kValueDim + nope;
  const float combined = inv_denom * bmm2 * scale_extra;
#pragma unroll
  for (int i = 0; i < 16; i += 2) {
    __nv_bfloat162 packed = __floats2bfloat162_rn(acc[i] * combined, acc[i + 1] * combined);
    *reinterpret_cast<__nv_bfloat162*>(orow + i) = packed;
  }
  if (lse != nullptr && lane == 0) lse[row * heads + head] = base_lse;
}

}  // namespace mla_v0

void mla_v0_launch(const uint8_t* q,
                   const uint8_t* kv,
                   const int* block_tables,
                   const int* seq_lens,
                   const int* cum_q,
                   uint8_t* out,
                   float* lse,
                   const float* sinks,
                   int heads,
                   int uniform_q,
                   long long rows,
                   int page_shift_unused,
                   int page_size,
                   int tstride,
                   float bmm1,
                   float bmm2,
                   cudaStream_t stream) {
  using namespace mla_v0;
  const long long warps = rows * heads;
  const long long blocks = (warps + kWarpsPerBlock - 1) / kWarpsPerBlock;
  mla_v0_kernel<<<(unsigned)blocks, kWarpsPerBlock * kWarp, 0, stream>>>(
      q, kv, block_tables, seq_lens, cum_q, out, lse, sinks, heads, uniform_q, page_size, tstride, bmm1,
      bmm2, warps);
}


