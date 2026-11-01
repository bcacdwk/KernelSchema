#pragma once
#include "kernel_tcgen.cuh"
namespace mla_v1 {
constexpr int kMaxSplits = 512;
// Target grid for the combine: about one wave of the CTAs an SM keeps resident
// (128 threads, 40 registers, ~8 KiB of static SMEM -> ~12 per SM on 148 SMs).
// MLA_COMB_TARGET exists so a job can sweep the sensitivity of that choice.
#ifndef MLA_COMB_TARGET
#define MLA_COMB_TARGET 1776
#endif
constexpr long long kCombTargetCtas = MLA_COMB_TARGET;
#ifndef MLA_COMB_MINDIMS
#define MLA_COMB_MINDIMS 32
#endif
constexpr int kCombMinDims = MLA_COMB_MINDIMS;
// 1 fetches the first split's partial_ms/partial_ds before the loads that decide
// how many splits there are. 0 is the v1.26 order and is the A/B control.
#ifndef MLA_COMB_SPEC
#define MLA_COMB_SPEC 1
#endif
// L1/shared carveout percentage pinned on every kernel this launcher starts, or
// negative to leave each function's default (kCarveout, defined above the helper
// namespace). See set_combine_attrs.
constexpr bool kCombSpec = (MLA_COMB_SPEC != 0);
#ifndef MLA_COMB_PUBNS
#define MLA_COMB_PUBNS 1
#endif
constexpr bool kCombPubNs = (MLA_COMB_PUBNS != 0);
#ifndef MLA_COMB_UNROLL
#define MLA_COMB_UNROLL 4
#endif
constexpr int kCombUnroll = MLA_COMB_UNROLL;
static_assert(kCombUnroll >= 1 && kCombUnroll <= 16, "MLA_COMB_UNROLL out of range");
#ifndef MLA_COMB_UNROLL_WIDE
#define MLA_COMB_UNROLL_WIDE 4
#endif
constexpr int kCombUnrollWide = MLA_COMB_UNROLL_WIDE;
static_assert(kCombUnrollWide >= 1 && kCombUnrollWide <= 16, "MLA_COMB_UNROLL_WIDE range");
#ifndef MLA_COMB_CYCPF
#define MLA_COMB_CYCPF 0
#endif
constexpr bool kCombCycPf = (MLA_COMB_CYCPF != 0);
#ifndef MLA_COMB_NSHOIST
#define MLA_COMB_NSHOIST 0
#endif
#if MLA_COMB_NSHOIST
#define MLA_COMB_NS ns_pub
#else
#define MLA_COMB_NS p.ns_shared[item]
#endif

template <int WARPS>
__device__ __forceinline__ float comb_reduce_max(float v, float* red) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float r = red[0];
#pragma unroll
  for (int w = 1; w < WARPS; ++w) r = fmaxf(r, red[w]);
  __syncthreads();
  return r;
}
template <int WARPS>
__device__ __forceinline__ float comb_reduce_sum(float v, float* red) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float r = red[0];
#pragma unroll
  for (int w = 1; w < WARPS; ++w) r += red[w];
  __syncthreads();
  return r;
}

// One split's weighted contribution to a thread's four output dims. The branch is
// CTA-uniform, because w comes from shared memory, and it exists only so that an
// empty split, whose partial_o was never written and may hold NaN, cannot poison
// the accumulator through 0*NaN.
__device__ __forceinline__ void comb_acc(float4& acc, float w, const float4& v) {
  if (w == 0.f) return;
  acc.x += w * v.x;
  acc.y += w * v.y;
  acc.z += w * v.z;
  acc.w += w * v.w;
}

template <int DIMS, int NSG>
__device__ __forceinline__ void mla_v1_combine_body(Params p) {
  constexpr int DT = DIMS / 4;         // dim-threads, four dims each
  constexpr int THREADS = NSG * DT;
  constexpr int WARPS = THREADS / 32;
  constexpr int UR = (NSG == 1) ? kCombUnrollWide : kCombUnroll;
  const int item = blockIdx.x;
  const int row = blockIdx.y;
  const int slab = blockIdx.z;
  const int tid = threadIdx.x;
  const size_t base0 = (size_t)item * p.n_splits * p.rows_max + row;
  const size_t bstride = (size_t)p.rows_max;
  const int sg = tid / DT;              // split group
  const int dt = tid - sg * DT;
  const int dim = slab * DIMS + dt * 4;
  float ms0 = 0.f, ds0 = 0.f;
  if (kCombSpec && tid < p.n_splits) {
    const size_t off = base0 + (size_t)tid * bstride;
    ms0 = p.partial_ms[off];
    ds0 = p.partial_ds[off];
  }
#if MLA_COMB_NSHOIST
  // item is blockIdx.x, so nothing has to be resolved before this address is known. The
  // read is in bounds for every CTA of this grid: ns_shared holds one word per item and
  // grid.x is items.
  const int ns_pub = (kCombPubNs && p.ns_shared != nullptr) ? p.ns_shared[item] : 0;
#endif
  float4 pv0 = make_float4(0.f, 0.f, 0.f, 0.f);
  if (kCombCycPf && sg < p.n_splits)
    pv0 = part_load4(p.partial_o + (base0 + (size_t)sg * bstride) * kPartWords, dim);
  // Both early-outs are CTA-uniform (they depend only on blockIdx), so the block
  // reductions below never see a partially exited CTA.
  ItemInfo info;
  if (!resolve_item(p, item, info)) return;
  // With the per-item split count published, seq_lens is not needed at all:
  // valid_rows and row0 come from info alone, and ntiles was only ever used to
  // recompute ns. resolve_row with kvlen=0 leaves visible and ntiles meaningless,
  // and this path reads neither.
  const bool pub_ns = kCombPubNs && p.ns_shared != nullptr;
  const int kvlen = pub_ns ? 0 : p.seq_lens[info.batch];
  const RowInfo ri = resolve_row(p, info, kvlen, row);
  if (row >= ri.valid_rows) return;
  const int rglobal = ri.row0 + row;
  const int q_idx = rglobal / p.heads;
  const int head = rglobal - q_idx * p.heads;
  const int ns = pub_ns ? MLA_COMB_NS : split_count(p, ri.ntiles);

  __shared__ float s_ms[kMaxSplits];
  __shared__ float s_ds[kMaxSplits];
  __shared__ float s_w[kMaxSplits];
  __shared__ float red[WARPS];
  __shared__ float s_acc[NSG][DIMS];

  // One round trip for both per-split scalars: they are independent addresses
  // and the weights below need both.
  if (kCombSpec) {
    if (tid < ns) {
      s_ms[tid] = ms0;
      s_ds[tid] = ds0;
    }
    for (int s = tid + THREADS; s < ns; s += THREADS) {
      const size_t off = base0 + (size_t)s * bstride;
      s_ms[s] = p.partial_ms[off];
      s_ds[s] = p.partial_ds[off];
    }
  } else {
    for (int s = tid; s < ns; s += THREADS) {
      const size_t off = base0 + (size_t)s * bstride;
      s_ms[s] = p.partial_ms[off];
      s_ds[s] = p.partial_ds[off];
    }
  }
  __syncthreads();

  float m = NEG_INF;
  for (int s = tid; s < ns; s += THREADS) m = fmaxf(m, s_ms[s]);
  m = comb_reduce_max<WARPS>(m, red);

  float dsum = 0.f;
  for (int s = tid; s < ns; s += THREADS) {
    const bool ok = (s_ms[s] > NEG_INF);
    const float w = ok ? ex2(s_ms[s] - m) : 0.f;
    const float wd = w * s_ds[s];
    // fp16 partials arrive already divided by their own denominator, so they are
    // re-weighted by it here; the total denominator is identical either way.
    s_w[s] = kPartFp16 ? wd : w;
    dsum += wd;
  }
  dsum = comb_reduce_sum<WARPS>(dsum, red);
  __syncthreads();   // publish s_w

  const float inv = (dsum > 0.f) ? (p.bmm2 / dsum) : 0.f;
  float base_lse = LN2 * (m + log2f(dsum) - LOG2_P_SCALE);
  float oscale = 1.f;
  if (p.sinks != nullptr) {
    const float sink = p.sinks[head];
    const float hi = fmaxf(base_lse, sink);
    const float total = hi + logf(ex2((base_lse - hi) * LOG2E) + ex2((sink - hi) * LOG2E));
    oscale = ex2((base_lse - total) * LOG2E);
    base_lse = total;
  }
  const float scale = inv * oscale;

  constexpr int step = kCombCycPf ? NSG : 1;
  const int chunk = (ns + NSG - 1) / NSG;
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  int s;
  int send;
  if (kCombCycPf) {
    if (sg < ns) comb_acc(acc, s_w[sg], pv0);
    s = sg + NSG;
    send = ns;
  } else {
    s = sg * chunk;
    send = min(ns, s + chunk);
  }
  for (; s + UR * step <= send; s += UR * step) {
    float w[UR];
    float4 v[UR];
#pragma unroll
    for (int u = 0; u < UR; ++u) w[u] = s_w[s + u * step];
#pragma unroll
    for (int u = 0; u < UR; ++u)
      v[u] = part_load4(p.partial_o + (base0 + (size_t)(s + u * step) * bstride) * kPartWords, dim);
#pragma unroll
    for (int u = 0; u < UR; ++u) comb_acc(acc, w[u], v[u]);
  }
  for (; s < send; s += step) {
    const float w = s_w[s];
    if (w == 0.f) continue;
    const float4 v = part_load4(p.partial_o + (base0 + (size_t)s * bstride) * kPartWords, dim);
    acc.x += w * v.x;
    acc.y += w * v.y;
    acc.z += w * v.z;
    acc.w += w * v.w;
  }
  if (NSG > 1) {
    s_acc[sg][dt * 4 + 0] = acc.x;
    s_acc[sg][dt * 4 + 1] = acc.y;
    s_acc[sg][dt * 4 + 2] = acc.z;
    s_acc[sg][dt * 4 + 3] = acc.w;
    __syncthreads();
    if (sg != 0) return;
    float tx = s_acc[0][dt * 4], ty = s_acc[0][dt * 4 + 1];
    float tz = s_acc[0][dt * 4 + 2], tw = s_acc[0][dt * 4 + 3];
#pragma unroll
    for (int g = 1; g < NSG; ++g) {
      tx += s_acc[g][dt * 4];
      ty += s_acc[g][dt * 4 + 1];
      tz += s_acc[g][dt * 4 + 2];
      tw += s_acc[g][dt * 4 + 3];
    }
    acc = make_float4(tx, ty, tz, tw);
  }
  if (sg == 0) {
    __nv_bfloat16* orow = reinterpret_cast<__nv_bfloat16*>(p.out) +
                          ((size_t)(info.q_start + q_idx) * p.heads + head) * DDIM + dim;
    const __nv_bfloat162 pk[2] = {__floats2bfloat162_rn(acc.x * scale, acc.y * scale),
                                  __floats2bfloat162_rn(acc.z * scale, acc.w * scale)};
    const uint32_t* pw = reinterpret_cast<const uint32_t*>(pk);
    asm volatile("st.global.v2.b32 [%0], {%1,%2};" ::"l"(orow), "r"(pw[0]), "r"(pw[1]));
  }
  if (p.lse != nullptr && tid == 0 && slab == 0)
    p.lse[(size_t)(info.q_start + q_idx) * p.heads + head] = base_lse;
}

// Output slab sizes; all entries retain the original 128-thread launch bound.
__global__ void __launch_bounds__(128) mla_combine_d512(Params p) {
  mla_v1_combine_body<512, 1>(p);
}
__global__ void __launch_bounds__(128) mla_combine_d256(Params p) {
  mla_v1_combine_body<256, 2>(p);
}
__global__ void __launch_bounds__(128) mla_combine_d128(Params p) {
  mla_v1_combine_body<128, 4>(p);
}
__global__ void __launch_bounds__(128) mla_combine_d64(Params p) {
  mla_v1_combine_body<64, 8>(p);
}
__global__ void __launch_bounds__(128) mla_combine_d32(Params p) {
  mla_v1_combine_body<32, 16>(p);
}
__global__ void __launch_bounds__(128) mla_combine_d16(Params p) {
  mla_v1_combine_body<16, 32>(p);
}
__global__ void __launch_bounds__(128) mla_combine_d8(Params p) {
  mla_v1_combine_body<8, 64>(p);
}



} // namespace mla_v1
