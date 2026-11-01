// mla_expanded_bf16 — td: tiny-decode warp-per-(batch,head) kernel.
//
// Regime: max_q_len == 1 && max_kv_len <= 64 && window_left < 0 (generic
// shape-derived dispatch; decode-like requests). The general engines (sync
// C2 BM=64 / tc2 BM=256) spend 6-15us here mostly on tiled-pipeline
// prologue for what is a pure GEMV-scale computation (measured: d0 kernel
// 6.2us, d2 14.3us vs ~1.5-3us achievable).
//
// Structure: one WARP per (batch, head) pair, 8 warps/CTA, no tensor cores.
// q (192 dims) lives in registers split 6 dims per lane. Scores: the warp
// walks the kv rows together - all lanes load the 12B dim-slice of row j
// (3x b32, fully coalesced 384B/row), 6 fma, 5-shfl butterfly -> s_j on
// every lane; the owning lane keeps it. Softmax in the log2 domain, then
// O accumulation with dims split 4 per lane: per row, p_j broadcasts by
// shfl and every lane adds p_j * v_j[4l..4l+3] from one coalesced 8B load.
// Right-aligned causal with qlen==1 makes every kv row visible
// (visible = kvlen-1), so causal and non-causal coincide here.
// Math matches problem/reference.py: fp32 scores/softmax, exp2 in the log2
// domain, lse = m2 + log2(d) with optional sink logaddexp2 and the matching
// output rescale; out = bmm2 * o / d.
//
// ABI: identical to kernel.cu's mla_forward so solution.py binds it the
// same way (memo/graph-replay paths unchanged).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace mla_td {

constexpr int DQK = 192;
constexpr int DV = 128;
constexpr int WARPS = 8;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float NEG_BIG = -1.0e38f;

__device__ __forceinline__ float bf2lo(uint32_t x) {
  return __uint_as_float(x << 16);
}
__device__ __forceinline__ float bf2hi(uint32_t x) {
  return __uint_as_float(x & 0xffff0000u);
}

__global__ void __launch_bounds__(WARPS * 32)
mla_td_kernel(const __nv_bfloat16* __restrict__ q,
              const __nv_bfloat16* __restrict__ k,
              const __nv_bfloat16* __restrict__ v,
              __nv_bfloat16* __restrict__ out,
              float* __restrict__ lse,
              const float* __restrict__ sinks,
              const int32_t* __restrict__ cum_q,
              const int32_t* __restrict__ cum_kv,
              int32_t total_q, int32_t batch, int32_t heads, int32_t kv_heads,
              int32_t window_left, int32_t causal, int32_t has_lse,
              int32_t has_sink, float bmm1_scale, float bmm2_scale) {
  (void)total_q; (void)window_left; (void)causal;
  const int lane = threadIdx.x & 31;
  const int pair = blockIdx.x * WARPS + (threadIdx.x >> 5);
  if (pair >= batch * heads) return;
  const int h = pair % heads;
  const int b = pair / heads;
  const int h_kv = h / (heads / kv_heads);

  const int q_base = cum_q[b];
  const int qlen = cum_q[b + 1] - q_base;      // == 1 under the td dispatch
  const int kv_base = cum_kv[b];
  const int kvlen = cum_kv[b + 1] - kv_base;   // <= 64 under the td dispatch
  if (qlen <= 0) return;

  // ---- q slice: dims [6l, 6l+6) as 3 b32 (6 bf16), fp32-converted ----
  const __nv_bfloat16* qp =
      q + (static_cast<int64_t>(q_base) * heads + h) * DQK;
  float qr[6];
  {
    const uint32_t* q32 = reinterpret_cast<const uint32_t*>(qp) + lane * 3;
    uint32_t a = q32[0], c = q32[1], d = q32[2];
    qr[0] = bf2lo(a); qr[1] = bf2hi(a);
    qr[2] = bf2lo(c); qr[3] = bf2hi(c);
    qr[4] = bf2lo(d); qr[5] = bf2hi(d);
  }

  const int64_t k_row = static_cast<int64_t>(kv_heads) * DQK;
  const int64_t v_row = static_cast<int64_t>(kv_heads) * DV;
  const __nv_bfloat16* kp =
      k + (static_cast<int64_t>(kv_base) * kv_heads + h_kv) * DQK;
  const __nv_bfloat16* vp =
      v + (static_cast<int64_t>(kv_base) * kv_heads + h_kv) * DV;

  // ---- scores: warp walks rows together; owner lane keeps s_j ----
  float srow[2] = {NEG_BIG, NEG_BIG};
  #pragma unroll 4
  for (int j = 0; j < kvlen; ++j) {
    const uint32_t* k32 =
        reinterpret_cast<const uint32_t*>(
            kp + static_cast<int64_t>(j) * k_row) + lane * 3;
    uint32_t a = k32[0], bb = k32[1], dd = k32[2];
    float acc = qr[0] * bf2lo(a) + qr[1] * bf2hi(a)
              + qr[2] * bf2lo(bb) + qr[3] * bf2hi(bb)
              + qr[4] * bf2lo(dd) + qr[5] * bf2hi(dd);
    #pragma unroll
    for (int off = 16; off; off >>= 1)
      acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (j == lane) srow[0] = acc;
    if (j == lane + 32) srow[1] = acc;
  }
  const bool v1ok = (lane + 32 < kvlen);

  // ---- softmax over the <=64 rows (log2 domain) ----
  float m = v1ok ? fmaxf(srow[0], srow[1]) : srow[0];
  #pragma unroll
  for (int off = 16; off; off >>= 1)
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
  const float sc2 = bmm1_scale * LOG2E;
  const float m2 = m * sc2;
  float p0 = 0.0f, p1 = 0.0f;
  {
    float x;
    asm("ex2.approx.f32 %0, %1;" : "=f"(x) : "f"(fmaf(srow[0], sc2, -m2)));
    p0 = x;
  }
  if (v1ok) {
    float x;
    asm("ex2.approx.f32 %0, %1;" : "=f"(x) : "f"(fmaf(srow[1], sc2, -m2)));
    p1 = x;
  }
  float d = p0 + p1;
  #pragma unroll
  for (int off = 16; off; off >>= 1)
    d += __shfl_xor_sync(0xffffffffu, d, off);

  // ---- O accumulation: lane owns dims [4l, 4l+4) ----
  float o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  #pragma unroll
  for (int c = 0; c < 2; ++c) {
    const float pmine = c ? p1 : p0;
    #pragma unroll 4
    for (int r = 0; r < 32; ++r) {
      const int j = r + c * 32;
      if (j >= kvlen) break;
      const float pj = __shfl_sync(0xffffffffu, pmine, r);
      const uint32_t* v32 =
          reinterpret_cast<const uint32_t*>(
              vp + static_cast<int64_t>(j) * v_row) + lane * 2;
      uint32_t a = v32[0], bb = v32[1];
      o[0] = fmaf(pj, bf2lo(a), o[0]);
      o[1] = fmaf(pj, bf2hi(a), o[1]);
      o[2] = fmaf(pj, bf2lo(bb), o[2]);
      o[3] = fmaf(pj, bf2hi(bb), o[3]);
    }
  }

  // ---- epilogue ----
  const float inv = d > 0.0f ? 1.0f / d : 0.0f;
  float lse2 = m2 + log2f(d);
  float extra = 1.0f;
  if (has_sink) {
    const float sink2 = sinks[h] * LOG2E;
    float t = fmaxf(lse2, sink2) + log2f(1.0f + exp2f(-fabsf(lse2 - sink2)));
    extra = exp2f(lse2 - t);
    lse2 = t;
  }
  const float g = bmm2_scale * inv * extra;
  __nv_bfloat16* orow =
      out + (static_cast<int64_t>(q_base) * heads + h) * DV + lane * 4;
  __nv_bfloat162 pk0 = __floats2bfloat162_rn(o[0] * g, o[1] * g);
  __nv_bfloat162 pk1 = __floats2bfloat162_rn(o[2] * g, o[3] * g);
  *reinterpret_cast<uint32_t*>(orow) = *reinterpret_cast<uint32_t*>(&pk0);
  *reinterpret_cast<uint32_t*>(orow + 2) = *reinterpret_cast<uint32_t*>(&pk1);
  if (has_lse && lane == 0) {
    lse[static_cast<int64_t>(q_base) * heads + h] = lse2;
  }
}

}  // namespace mla_td

extern "C" int mla_forward(const void* q, const void* k, const void* v,
                           void* out, void* lse, const void* sinks,
                           const void* cum_q, const void* cum_kv,
                           int total_q, int batch, int heads, int kv_heads,
                           int window_left, int causal, int has_lse,
                           int has_sink, float bmm1_scale, float bmm2_scale,
                           int max_kv_len, void* stream) {
  using namespace mla_td;
  (void)max_kv_len;
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const int pairs = batch * heads;
  if (pairs <= 0) return 5000;
  const int grid = (pairs + WARPS - 1) / WARPS;
  mla_td_kernel<<<dim3(grid), dim3(WARPS * 32), 0, s>>>(
      static_cast<const __nv_bfloat16*>(q),
      static_cast<const __nv_bfloat16*>(k),
      static_cast<const __nv_bfloat16*>(v),
      static_cast<__nv_bfloat16*>(out), static_cast<float*>(lse),
      static_cast<const float*>(sinks),
      static_cast<const int32_t*>(cum_q),
      static_cast<const int32_t*>(cum_kv),
      total_q, batch, heads, kv_heads, window_left, causal, has_lse, has_sink,
      bmm1_scale, bmm2_scale);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int mla_config_count() { return 1; }
